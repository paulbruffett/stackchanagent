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

namespace agent::buddy_ble {

namespace {

constexpr const char* TAG = "buddy";
using stackchan::avatar::Emotion;

// Connection treated as dead if no snapshot in this long (REFERENCE.md ~30s).
constexpr uint32_t kSnapshotStaleMs = 30000;

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

// Fun lifetime counters reported in status.stats (in-memory, best effort).
std::atomic<uint32_t> g_appr{0};

// ---- Render state (touched only by tick) ----
enum class Desired { None, Working, Waiting };
Desired g_rendered = Desired::None;
bool g_owns_screen = false;   // buddy currently drawing a face/bubble
bool g_woke_screen = false;   // buddy turned the screen on for a prompt
bool g_passkey_drawn = false;

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

// ---- Outbound JSON ----
void notify_doc(JsonDocument& doc)
{
    std::string out;
    serializeJson(doc, out);
    out.push_back('\n');
    buddy_nus_notify(out.data(), static_cast<int>(out.size()));
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
    data["sec"] = buddy_nus_encrypted();
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
    g_total.store(doc["total"] | 0);
    g_running.store(doc["running"] | 0);
    g_waiting.store(doc["waiting"] | 0);
    g_last_snapshot_ms.store(GetHAL().millis());

    JsonObjectConst prompt = doc["prompt"];
    if (!prompt.isNull()) {
        const char* id = prompt["id"] | "";
        const char* tool = prompt["tool"] | "tool";
        {
            std::lock_guard<std::mutex> lk(g_prompt_mtx);
            g_prompt_id = id;
            g_prompt_tool = tool;
        }
        g_has_prompt.store(id[0] != '\0');
    } else {
        g_has_prompt.store(false);
    }
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
    g_showing_passkey.store(true);
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
    notify_doc(doc);
    g_appr.fetch_add(1);
    g_has_prompt.store(false);  // optimistic; next snapshot confirms
    mclog::tagInfo(TAG, "approved prompt {}", id);
}

void tick()
{
    uint32_t now = GetHAL().millis();

    // Pairing passkey display wins until the link is encrypted.
    if (g_showing_passkey.load()) {
        if (buddy_nus_encrypted()) {
            g_showing_passkey.store(false);
            draw_bubble("");
            g_passkey_drawn = false;
        } else {
            if (!g_passkey_drawn) {
                commands::wake_face();
                char buf[20];
                std::snprintf(buf, sizeof(buf), "PIN %06u",
                              static_cast<unsigned>(g_passkey.load()));
                draw_bubble(buf);
                g_passkey_drawn = true;
            }
            return;
        }
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
