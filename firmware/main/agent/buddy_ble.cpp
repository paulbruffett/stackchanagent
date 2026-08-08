#include "buddy_ble.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>
#include <string_view>
#include <sys/time.h>

#include <ArduinoJson.h>
#include <esp_system.h>
#include <hal/hal.h>
#include <mooncake_log.h>
#include <stackchan/avatar/avatar/elements/emotion.h>
#include <stackchan/stackchan.h>

#include "buddy_ble_nus.h"
#include "commands.h"
#include "state.h"
#include "transport.h"

namespace agent::buddy_ble {

namespace {

constexpr const char* TAG = "buddy";
using stackchan::avatar::Emotion;

// Connection treated as dead if no snapshot in this long (REFERENCE.md ~30s).
constexpr uint32_t kSnapshotStaleMs = 30000;

// Give up on a passkey display after the SM pairing timeout (30 s) plus slack.
constexpr uint32_t kPasskeyTimeoutMs = 35000;

// Re-check that we are still advertising this often while no central is up.
constexpr uint32_t kAdvCheckIntervalMs = 5000;

// Minimum gap between buddy_prompt announcement attempts to the brain.
constexpr uint32_t kPromptRetryMs = 1000;

// ---- Snapshot state (written from the NimBLE host task, read from tick) ----
std::atomic<int> g_total{0};
std::atomic<int> g_running{0};
std::atomic<int> g_waiting{0};
std::atomic<bool> g_has_prompt{false};
std::atomic<uint32_t> g_last_snapshot_ms{0};

std::mutex g_prompt_mtx;
std::string g_prompt_id;    // guarded by g_prompt_mtx
std::string g_prompt_tool;  // guarded by g_prompt_mtx

// Pairing passkey display (set by the passkey hook, drawn by tick).
std::atomic<bool> g_showing_passkey{false};
std::atomic<uint32_t> g_passkey{0};
std::atomic<uint32_t> g_passkey_at_ms{0};
std::atomic<bool> g_pairing_failed{false};

// Fun lifetime counters reported in status.stats (in-memory, best effort).
std::atomic<uint32_t> g_appr{0};

// ---- Render state (touched only by tick) ----
enum class Desired { None, Working, Waiting };
Desired g_rendered = Desired::None;
bool g_owns_screen = false;   // buddy currently drawing a face/bubble
bool g_woke_screen = false;   // buddy turned the screen on for a prompt
bool g_passkey_drawn = false;
bool g_prompt_announced = false;  // last buddy_prompt pending state sent to brain
bool g_last_face_off = false;     // backlight state observed on the previous tick
bool g_ws_was_connected = false;  // brain-WS state observed on the previous tick
uint32_t g_last_adv_check_ms = 0;   // last "are we still advertising?" poll
uint32_t g_last_prompt_try_ms = 0;  // last buddy_prompt announcement attempt
// Set from the WS/dispatch task when the avatar is swapped (set_skin); consumed
// by tick() to force a redraw. Atomic because it crosses tasks; g_rendered et al
// stay tick-only.
std::atomic<bool> g_force_redraw{false};

// ---- LVGL helpers (each takes the lock; never call while already holding) --
void draw_bubble(const char* text)
{
    LvglLockGuard lock;
    if (text && text[0]) {
        GetStackChan().avatar().setSpeech(text);
    } else {
        GetStackChan().avatar().clearSpeech();
    }
}

void draw_emotion(Emotion e)
{
    LvglLockGuard lock;
    GetStackChan().avatar().setEmotion(e);
}

// Release buddy's hold on the screen. If we woke it for a prompt, put it back
// to sleep; otherwise just clear our bubble/face.
void release_screen(bool resleep)
{
    draw_bubble("");
    draw_emotion(Emotion::Neutral);
    if (resleep) commands::sleep_face();
    g_owns_screen = false;
    g_woke_screen = false;
    g_rendered = Desired::None;
}

// Tear down the PIN bubble. Separate from release_screen() because buddy never
// "owns" the face during pairing — but it may still have lit the screen to
// show the PIN, and nothing else would ever put it back to sleep (the brain
// believes the device is already asleep, so its timer won't fire again).
void end_passkey_display()
{
    g_showing_passkey.store(false);
    g_passkey_drawn = false;
    draw_bubble("");
    if (g_woke_screen && !g_owns_screen) {
        commands::sleep_face();
        g_woke_screen = false;
    }
}

// ---- Outbound JSON ----
// Returns buddy_nus_notify's rc (0 on success) so callers that commit local
// state on the strength of a notification can check it.
int notify_doc(JsonDocument& doc)
{
    std::string out;
    serializeJson(doc, out);
    out.push_back('\n');
    return buddy_nus_notify(out.data(), static_cast<int>(out.size()));
}

void ack(const char* cmd, bool ok, const char* error = nullptr)
{
    JsonDocument doc;
    doc["ack"] = cmd;
    doc["ok"] = ok;
    if (error) doc["error"] = error;
    notify_doc(doc);
}

void send_status()
{
    JsonDocument doc;
    doc["ack"] = "status";
    doc["ok"] = true;
    JsonObject data = doc["data"].to<JsonObject>();
    data["name"] = "Claude StackChan";
    data["sec"] = buddy_nus_authenticated();
    JsonObject bat = data["bat"].to<JsonObject>();
    bat["pct"] = GetHAL().getBatteryLevel();
    bat["usb"] = GetHAL().isBatteryCharging();
    JsonObject sys = data["sys"].to<JsonObject>();
    sys["up"] = GetHAL().millis() / 1000;
    sys["heap"] = esp_get_free_heap_size();
    JsonObject stats = data["stats"].to<JsonObject>();
    stats["appr"] = g_appr.load();
    notify_doc(doc);
}

// ---- Inbound parsing ----
void apply_time(JsonArrayConst arr)
{
    if (arr.size() < 1) return;
    time_t epoch = arr[0].as<long long>();
    struct timeval tv{epoch, 0};
    settimeofday(&tv, nullptr);
    mclog::tagInfo(TAG, "time synced: {}", static_cast<long long>(epoch));
}

void apply_snapshot(JsonDocument& doc)
{
    int total = doc["total"] | 0;
    int running = doc["running"] | 0;
    int waiting = doc["waiting"] | 0;
    g_total.store(total);
    g_running.store(running);
    g_waiting.store(waiting);
    g_last_snapshot_ms.store(GetHAL().millis());

    JsonObjectConst prompt = doc["prompt"];
    const char* tool = "none";
    if (!prompt.isNull()) {
        const char* id = prompt["id"] | "";
        tool = prompt["tool"] | "tool";
        {
            std::lock_guard<std::mutex> lk(g_prompt_mtx);
            g_prompt_id = id;
            g_prompt_tool = tool;
        }
        g_has_prompt.store(id[0] != '\0');
    } else {
        g_has_prompt.store(false);
    }
    // Diagnostic: show exactly what the desktop reports so we can tell a
    // "no prompt arrived" (desktop coverage) case from a render bug.
    mclog::tagInfo(TAG, "snapshot: total={} running={} waiting={} prompt={}",
                   total, running, waiting, tool);
}

void handle_command(std::string_view cmd, JsonDocument& doc)
{
    if (cmd == "status") {
        send_status();
    } else if (cmd == "name") {
        // Display name is fixed in the BLE advertisement; accept + ack.
        mclog::tagInfo(TAG, "name set: {}", doc["name"] | "");
        ack("name", true);
    } else if (cmd == "owner") {
        mclog::tagInfo(TAG, "owner: {}", doc["name"] | "");
        ack("owner", true);
    } else if (cmd == "unpair") {
        ack("unpair", true);   // ack before tearing the link down
        buddy_nus_unpair();
    } else if (cmd == "char_begin" || cmd == "file" || cmd == "chunk" ||
               cmd == "file_end" || cmd == "char_end") {
        // Folder push (custom face assets) is out of scope for now — decline
        // politely so the desktop doesn't hang waiting for an ack.
        ack(std::string(cmd).c_str(), false, "unsupported");
    } else {
        mclog::tagWarn(TAG, "unknown cmd: {}", cmd);
        ack(std::string(cmd).c_str(), false, "unknown");
    }
}

void on_line(const char* line, int len)
{
    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, line, len);
    if (err) {
        mclog::tagWarn(TAG, "bad json: {}", err.c_str());
        return;
    }
    if (doc["time"].is<JsonArrayConst>()) {
        apply_time(doc["time"].as<JsonArrayConst>());
        return;
    }
    const char* cmd = doc["cmd"] | static_cast<const char*>(nullptr);
    if (cmd) {
        handle_command(cmd, doc);
        return;
    }
    if (doc["evt"].is<const char*>()) {
        return;  // turn events: ignored for now
    }
    // Otherwise: a heartbeat snapshot.
    apply_snapshot(doc);
}

}  // namespace

// ---- C hooks called from buddy_ble_nus.c ----
extern "C" void buddy_on_passkey(uint32_t passkey)
{
    g_passkey.store(passkey);
    g_passkey_drawn = false;
    g_passkey_at_ms.store(GetHAL().millis());
    g_pairing_failed.store(false);
    g_showing_passkey.store(true);
}

extern "C" void buddy_on_pairing_failed(void)
{
    // Runs on the NimBLE host task; tick() owns the drawing state, so only
    // raise the flag and let it tear the PIN bubble down.
    g_pairing_failed.store(true);
}

extern "C" void buddy_on_connect(void)
{
    g_total.store(0);
    g_running.store(0);
    g_waiting.store(0);
    g_has_prompt.store(false);
}

extern "C" void buddy_on_disconnect(void)
{
    g_total.store(0);
    g_running.store(0);
    g_waiting.store(0);
    g_has_prompt.store(false);
    g_showing_passkey.store(false);
}

// ---- Public API ----
void start()
{
    buddy_nus_init(&on_line);
}

bool prompt_pending()
{
    return g_has_prompt.load();
}

void notify_avatar_swapped()
{
    // Called from the set_skin handler (WS task) after attachAvatar. The new
    // avatar starts blank, so invalidate our render state to redraw any
    // pending prompt/bubble next tick.
    g_force_redraw.store(true);
}

void approve_pending()
{
    std::string id;
    {
        std::lock_guard<std::mutex> lk(g_prompt_mtx);
        id = g_prompt_id;
    }
    if (id.empty()) return;
    JsonDocument doc;
    doc["cmd"] = "permission";
    doc["id"] = id;
    doc["decision"] = "once";
    int rc = notify_doc(doc);
    if (rc != 0) {
        // The desktop never got the decision (link gone, mbuf pool empty).
        // Leave the prompt up so the bubble stays and the tap is repeatable
        // rather than logging an approval that did not happen.
        mclog::tagWarn(TAG, "approve notify failed (rc={}); prompt {} left pending", rc, id);
        return;
    }
    g_appr.fetch_add(1);
    g_has_prompt.store(false);  // optimistic; next snapshot confirms
    mclog::tagInfo(TAG, "approved prompt {}", id);
}

void tick()
{
    uint32_t now = GetHAL().millis();

    // If the screen was woken by something other than us (a head tap, the
    // wakeword, or a brain wake/activity command), wake_face() reset the face
    // to Neutral — clobbering any approve prompt we had drawn. Force a redraw
    // so a "see it, then approve" prompt comes back. (g_woke_screen guards the
    // case where buddy itself just woke the screen for the prompt.)
    bool face_off_now = commands::face_is_off();
    if (g_last_face_off && !face_off_now && !g_woke_screen) {
        g_rendered = Desired::None;
    }
    g_last_face_off = face_off_now;

    // A skin swap (set_skin) rebuilt the avatar, wiping any prompt/PIN bubble
    // we had drawn. Force a redraw so it comes back on the next tick.
    if (g_force_redraw.exchange(false)) {
        g_rendered = Desired::None;
        g_owns_screen = false;
    }

    // On a brain-WS (re)connect, re-announce the buddy_prompt pending state:
    // our edge-trigger global survives a brain restart but the brain's view
    // resets to "not pending", so without this the device could sleep under
    // an unanswered prompt after a reconnect.
    bool ws_now = transport::is_connected();
    if (ws_now && !g_ws_was_connected) {
        g_prompt_announced = false;
    }
    g_ws_was_connected = ws_now;

    // Pairing passkey display wins until the link is authenticated.
    if (g_showing_passkey.load()) {
        if (buddy_nus_authenticated()) {
            end_passkey_display();
        } else if (g_pairing_failed.load()
                   || now - g_passkey_at_ms.load() > kPasskeyTimeoutMs) {
            // Pairing failed or never finished, and NimBLE keeps the ACL up —
            // so without an exit the stale PIN stays on screen and tick()
            // returns below on every iteration, which also freezes the
            // buddy_prompt edge announcement to the brain.
            mclog::tagWarn(TAG, "pairing did not complete; clearing PIN display");
            end_passkey_display();
        } else {
            if (!g_passkey_drawn) {
                if (commands::face_is_off()) {
                    commands::wake_face();
                    g_woke_screen = true;
                }
                char buf[20];
                std::snprintf(buf, sizeof(buf), "PIN %06u",
                              static_cast<unsigned>(g_passkey.load()));
                draw_bubble(buf);
                g_passkey_drawn = true;
            }
            return;
        }
    }

    // Advertising is otherwise only re-armed on GAP edges, and every failure
    // path there just logs — one lost ble_gap_adv_start() would leave the
    // buddy link dead until a power cycle, with no indicator anywhere.
    if (now - g_last_adv_check_ms >= kAdvCheckIntervalMs) {
        g_last_adv_check_ms = now;
        buddy_nus_ensure_advertising();
    }

    // A live conversation always wins; yield without sleeping.
    if (state::current() != state::Mode::Idle) {
        if (g_owns_screen) release_screen(/*resleep=*/false);
        return;
    }

    // Decide the desired buddy UI from the freshest snapshot.
    Desired desired = Desired::None;
    bool stale = !buddy_nus_connected() ||
                 (now - g_last_snapshot_ms.load() > kSnapshotStaleMs);
    if (!stale) {
        if (g_waiting.load() > 0 && g_has_prompt.load()) {
            desired = Desired::Waiting;
        } else if (g_running.load() > 0) {
            desired = Desired::Working;
        }
    }

    // Tell the brain whether a permission prompt is pending. The brain's sleep
    // timer is otherwise blind to BLE buddy state and would sleep the device
    // out from under an unanswered approve prompt; while pending it uses an
    // elongated timeout instead. Edge-triggered to avoid spamming the WS.
    bool pending = (desired == Desired::Waiting);
    if (pending != g_prompt_announced && now - g_last_prompt_try_ms >= kPromptRetryMs) {
        // Only commit the edge if the brain actually heard it; otherwise we
        // would believe we had announced a pending prompt it never learned
        // about, which is exactly the "sleeps under an unanswered prompt"
        // divergence the WS-reconnect reset above exists to prevent. Retried,
        // but not at the 50 Hz tick rate — send_event_json can block on a
        // half-dead socket and this runs on the main loop.
        g_last_prompt_try_ms = now;
        if (transport::send_event_json(pending
                                           ? "{\"event\":\"buddy_prompt\",\"pending\":true}"
                                           : "{\"event\":\"buddy_prompt\",\"pending\":false}")) {
            g_prompt_announced = pending;
        }
    }

    if (desired == g_rendered) return;

    switch (desired) {
        case Desired::Waiting: {
            if (commands::face_is_off()) {
                commands::wake_face();
                g_woke_screen = true;
            }
            std::string tool;
            {
                std::lock_guard<std::mutex> lk(g_prompt_mtx);
                tool = g_prompt_tool;
            }
            draw_emotion(Emotion::Doubt);
            draw_bubble(("approve: " + tool).c_str());
            g_owns_screen = true;
            break;
        }
        case Desired::Working: {
            // Don't wake the screen just to show "working" — only indicate if
            // it's already on (reuse the "…" busy bubble).
            if (!commands::face_is_off()) draw_bubble("...");
            g_owns_screen = true;
            break;
        }
        case Desired::None: {
            if (g_owns_screen) release_screen(/*resleep=*/g_woke_screen);
            break;
        }
    }
    g_rendered = desired;
}

}  // namespace agent::buddy_ble
