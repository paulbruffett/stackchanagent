#include "ota.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>

#include <board.h>
#include <esp_app_desc.h>
#include <esp_heap_caps.h>
#include <esp_ota_ops.h>
#include <esp_system.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <hal/hal.h>
#include <http.h>
#include <mbedtls/sha256.h>
#include <mooncake_log.h>
#include <stackchan/stackchan.h>

#include "state.h"
#include "transport.h"
#include "wakeword.h"

namespace agent::ota {

namespace {

constexpr const char* TAG = "agent.ota";

// Probation for a freshly updated image (see confirm_tick): valid after this
// long of unbroken brain link, rolled back if that hasn't happened by the
// deadline. A brain restart/deploy is ~30 s, so the deadline allows several.
constexpr int64_t kConfirmLinkMs = 30000;
constexpr int64_t kConfirmDeadlineMs = 5 * 60 * 1000;

// Per-read HTTP timeout; a stalled transfer fails after this.
constexpr int kHttpTimeoutMs = 15000;
constexpr size_t kChunk = 4096;

std::atomic<bool> g_in_progress{false};

struct Job {
    std::string url;
    size_t size;
    std::string sha256_hex;
    int id;
};

// `id` is the brain's attempt number, echoed so a late event from an earlier
// attempt can't be mistaken for the current one.
void send_state(int id, const char* state, int pct = -1, const std::string& error = {})
{
    std::string json = "{\"event\":\"ota\",\"id\":" + std::to_string(id)
                       + ",\"state\":\"" + state + "\"";
    if (pct >= 0) json += ",\"pct\":" + std::to_string(pct);
    if (!error.empty()) {
        // Errors are our own fixed strings plus esp_err names / numbers: no
        // quotes or backslashes to escape.
        json += ",\"error\":\"" + error + "\"";
    }
    json += "}";
    transport::send_event_json(json);
}

// The avatar's speech bubble + busy indicator, as the brain's set_busy uses.
void show(const std::string& text)
{
    LvglLockGuard lock;
    GetStackChan().avatar().setBusy(!text.empty());
    if (text.empty()) {
        GetStackChan().avatar().clearSpeech();
    } else {
        GetStackChan().avatar().setSpeech(text);
    }
}

std::string to_hex(const uint8_t* d, size_t n)
{
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        out += kHex[d[i] >> 4];
        out += kHex[d[i] & 0xF];
    }
    return out;
}

// Download, verify and activate. Returns "" on success (boot partition set),
// else a short reason. Everything it opens is closed/aborted on every path.
std::string run(const Job& job)
{
    const esp_partition_t* part = esp_ota_get_next_update_partition(nullptr);
    if (part == nullptr) return "no OTA partition";
    if (job.size > part->size) return "image larger than the OTA partition";

    auto* network = Board::GetInstance().GetNetwork();
    if (network == nullptr) return "no network";
    std::unique_ptr<Http> http = network->CreateHttp(0);
    http->SetTimeout(kHttpTimeoutMs);
    if (!http->Open("GET", job.url)) return "http open failed";
    int status = http->GetStatusCode();
    if (status != 200) {
        http->Close();
        return "http status " + std::to_string(status);
    }
    size_t body_len = http->GetBodyLength();
    if (body_len != job.size) {
        http->Close();
        return "content length " + std::to_string(body_len) + " != " + std::to_string(job.size);
    }

    esp_ota_handle_t handle = 0;
    esp_err_t err = esp_ota_begin(part, OTA_WITH_SEQUENTIAL_WRITES, &handle);
    if (err != ESP_OK) {
        http->Close();
        return std::string("ota begin: ") + esp_err_to_name(err);
    }
    mclog::tagInfo(TAG, "writing {} bytes to {}", job.size, part->label);

    auto* buf = static_cast<char*>(heap_caps_malloc(kChunk, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    mbedtls_sha256_context sha;
    mbedtls_sha256_init(&sha);
    mbedtls_sha256_starts(&sha, 0);

    std::string fail;
    size_t total = 0;
    int reported = 0;
    if (buf == nullptr) fail = "out of memory";
    while (fail.empty() && total < job.size) {
        int n = http->Read(buf, kChunk);
        if (n < 0) {
            fail = "download interrupted";
            break;
        }
        if (n == 0) break;  // EOF early; the size check below reports it
        if (total + n > job.size) {
            fail = "more data than announced";
            break;
        }
        mbedtls_sha256_update(&sha, reinterpret_cast<const unsigned char*>(buf), n);
        err = esp_ota_write(handle, buf, n);
        if (err != ESP_OK) {
            fail = std::string("ota write: ") + esp_err_to_name(err);
            break;
        }
        total += n;
        int pct = static_cast<int>(total * 100 / job.size);
        if (pct / 10 > reported / 10) {
            reported = pct;
            send_state(job.id, "downloading", pct);
            show("Updating " + std::to_string(pct) + "%");
        }
    }
    http->Close();
    heap_caps_free(buf);

    uint8_t digest[32];
    mbedtls_sha256_finish(&sha, digest);
    mbedtls_sha256_free(&sha);

    if (fail.empty() && total != job.size) {
        fail = "short download (" + std::to_string(total) + " bytes)";
    }
    if (fail.empty() && to_hex(digest, sizeof digest) != job.sha256_hex) {
        fail = "sha256 mismatch";
    }
    if (!fail.empty()) {
        esp_ota_abort(handle);
        return fail;
    }
    // Validates the image: header, segments, appended hash, chip, and the
    // RSA signature against the running app's key
    // (CONFIG_SECURE_SIGNED_ON_UPDATE_NO_SECURE_BOOT).
    err = esp_ota_end(handle);
    if (err != ESP_OK) return std::string("image check: ") + esp_err_to_name(err);
    err = esp_ota_set_boot_partition(part);
    if (err != ESP_OK) return std::string("set boot partition: ") + esp_err_to_name(err);
    return {};
}

void ota_task(void* arg)
{
    std::unique_ptr<Job> job(static_cast<Job*>(arg));
    std::string err = run(*job);
    if (err.empty()) {
        mclog::tagInfo(TAG, "update written; rebooting");
        send_state(job->id, "rebooting");
        show("Rebooting...");
        vTaskDelay(pdMS_TO_TICKS(1000));  // let the event and the log get out
        esp_restart();
    }
    mclog::tagError(TAG, "update failed: {}", err);
    send_state(job->id, "failed", -1, err);
    show("Update failed");
    vTaskDelay(pdMS_TO_TICKS(4000));
    show({});
    g_in_progress = false;
    wakeword::resume();
    vTaskDelete(nullptr);
}

bool valid_sha256_hex(std::string_view s)
{
    if (s.size() != 64) return false;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

// "http://HOST[:port]/…" → HOST ("" if not that shape).
std::string_view url_host(std::string_view url)
{
    constexpr std::string_view kScheme = "http://";
    if (url.substr(0, kScheme.size()) != kScheme) return {};
    url.remove_prefix(kScheme.size());
    size_t end = url.find_first_of(":/");
    return end == std::string_view::npos ? url : url.substr(0, end);
}

// Image state of the running partition, read once (it's in flash): 1 while
// a fresh OTA image is on probation, 0 otherwise, -1 not read yet. Cleared
// by confirm_tick once it marks the image valid.
std::atomic<int> g_pending{-1};

bool read_pending()
{
    int v = g_pending.load();
    if (v < 0) {
        const esp_partition_t* running = esp_ota_get_running_partition();
        esp_ota_img_states_t img_state;
        // A USB-flashed image has no OTA state (or a valid one).
        v = running != nullptr && esp_ota_get_state_partition(running, &img_state) == ESP_OK
            && img_state == ESP_OTA_IMG_PENDING_VERIFY;
        int expected = -1;
        g_pending.compare_exchange_strong(expected, v);
        v = g_pending.load();
    }
    return v == 1;
}

}  // namespace

void start(std::string_view url, size_t size, std::string_view sha256_hex, int id)
{
    if (url_host(url).empty() || size == 0 || !valid_sha256_hex(sha256_hex)) {
        mclog::tagWarn(TAG, "ota: bad arguments");
        send_state(id, "failed", -1, "bad ota command");
        return;
    }
    // Defense in depth on top of the image signature: only ever download
    // from the brain we're talking to, never a host a command names.
    std::string brain = transport::brain_ip();
    if (brain.empty() || url_host(url) != brain) {
        mclog::tagWarn(TAG, "ota: url host is not the brain ({})", brain);
        send_state(id, "failed", -1, "url host is not the brain this robot is connected to");
        return;
    }
    if (read_pending()) {
        // esp_ota_begin refuses anyway (ESP_ERR_OTA_ROLLBACK_INVALID_STATE);
        // say why in words.
        send_state(id, "failed", -1, "running image not confirmed yet; retry in a minute");
        return;
    }
    if (state::current() != state::Mode::Idle) {
        send_state(id, "failed", -1, "robot busy (not idle)");
        return;
    }
    bool expected = false;
    if (!g_in_progress.compare_exchange_strong(expected, true)) {
        send_state(id, "failed", -1, "update already running");
        return;
    }
    mclog::tagInfo(TAG, "update #{} requested: {} bytes", id, size);
    // Detection off for the duration: nothing may start a turn, and the AFE
    // is CPU we'd rather spend on the download. The wakeword/tap handlers
    // also check in_progress().
    wakeword::pause();
    show("Updating...");
    auto* job = new Job{std::string(url), size, std::string(sha256_hex), id};
    if (xTaskCreate(ota_task, "agent_ota", 8192, job, 3, nullptr) != pdPASS) {
        delete job;
        show({});
        g_in_progress = false;
        wakeword::resume();
        send_state(id, "failed", -1, "could not start update task");
    }
}

bool in_progress()
{
    return g_in_progress.load();
}

bool pending_verify()
{
    return read_pending();
}

// A new image is on probation until it proves it can do the one thing that
// matters: hold a brain link. It is marked valid once the WebSocket has been
// up for kConfirmLinkMs without a break; if that hasn't happened by
// kConfirmDeadlineMs after boot it rolls back and restarts into the previous
// image. Edge cases:
//  - It crashes, hangs into the task watchdog or is power-cycled before
//    confirming: the bootloader sees PENDING_VERIFY on the next boot and
//    boots the previous image instead (that is the rollback feature itself).
//  - The brain restarts/deploys during the window (~30 s outage): the link
//    clock starts over when it reconnects; the 5 min deadline leaves room
//    for that several times over.
//  - The brain is down for all of the first 5 min: we roll back even though
//    the new image may be fine. The old image is known good, and the console
//    shows "rolled back?" so the update can simply be sent again.
//  - Wi-Fi never comes up at all: same as above.
// While on probation, ota::start refuses further updates (the OTA API would
// too) and the brain holds off anything that could reboot us.
void confirm_tick()
{
    static bool settled = false;
    static int64_t link_since_ms = 0;
    if (settled) return;
    if (!read_pending()) {
        settled = true;
        return;
    }
    int64_t now = state::now_ms();
    if (!transport::is_connected()) {
        link_since_ms = 0;
    } else if (link_since_ms == 0) {
        link_since_ms = now;
        mclog::tagInfo(TAG, "new image reached the brain; confirming after {} s of link",
                       kConfirmLinkMs / 1000);
    } else if (now - link_since_ms >= kConfirmLinkMs) {
        esp_err_t err = esp_ota_mark_app_valid_cancel_rollback();
        mclog::tagInfo(TAG, "new image marked valid: {}", esp_err_to_name(err));
        if (err == ESP_OK) {
            g_pending = 0;
            settled = true;
            transport::send_event_json("{\"event\":\"ota\",\"state\":\"confirmed\",\"fw_sha\":\""
                                       + running_elf_sha256() + "\"}");
        }
        return;
    }
    if (now > kConfirmDeadlineMs) {
        mclog::tagError(TAG, "new image never held a brain link for {} s within {} s; rolling back",
                        kConfirmLinkMs / 1000, kConfirmDeadlineMs / 1000);
        // Does not return when a previous valid image exists.
        esp_ota_mark_app_invalid_rollback_and_reboot();
        settled = true;
    }
}

std::string running_version()
{
    return esp_app_get_description()->version;
}

std::string running_build()
{
    const esp_app_desc_t* d = esp_app_get_description();
    return std::string(d->date) + " " + d->time;
}

std::string running_elf_sha256()
{
    char hex[65] = {};
    esp_app_get_elf_sha256(hex, sizeof hex);
    return hex;
}

}  // namespace agent::ota
