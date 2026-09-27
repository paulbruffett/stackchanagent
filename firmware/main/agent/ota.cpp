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

// A new image that hasn't reached the brain by this long after boot is
// treated as broken and rolled back. Covers Wi-Fi association (bring-up
// sweep ~7 s, then DHCP) and a few rungs of transport's reconnect backoff.
constexpr int64_t kConfirmDeadlineMs = 60000;

// Per-read HTTP timeout; a stalled transfer fails after this.
constexpr int kHttpTimeoutMs = 15000;
constexpr size_t kChunk = 4096;

std::atomic<bool> g_in_progress{false};

struct Job {
    std::string url;
    size_t size;
    std::string sha256_hex;
};

void send_state(const char* state, int pct = -1, const std::string& error = {})
{
    std::string json = std::string("{\"event\":\"ota\",\"state\":\"") + state + "\"";
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
            send_state("downloading", pct);
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
    // Validates the image (header, segments, appended hash, chip).
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
        send_state("rebooting");
        show("Rebooting...");
        vTaskDelay(pdMS_TO_TICKS(1000));  // let the event and the log get out
        esp_restart();
    }
    mclog::tagError(TAG, "update failed: {}", err);
    send_state("failed", -1, err);
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

}  // namespace

void start(std::string_view url, size_t size, std::string_view sha256_hex)
{
    if (url.substr(0, 7) != "http://" || size == 0 || !valid_sha256_hex(sha256_hex)) {
        mclog::tagWarn(TAG, "ota: bad arguments");
        send_state("failed", -1, "bad ota command");
        return;
    }
    if (state::current() != state::Mode::Idle) {
        send_state("failed", -1, "robot busy (not idle)");
        return;
    }
    bool expected = false;
    if (!g_in_progress.compare_exchange_strong(expected, true)) {
        send_state("failed", -1, "update already running");
        return;
    }
    mclog::tagInfo(TAG, "update requested: {} bytes", size);
    // Detection off for the duration: nothing may start a turn, and the AFE
    // is CPU we'd rather spend on the download. The wakeword/tap handlers
    // also check in_progress().
    wakeword::pause();
    show("Updating...");
    auto* job = new Job{std::string(url), size, std::string(sha256_hex)};
    if (xTaskCreate(ota_task, "agent_ota", 8192, job, 3, nullptr) != pdPASS) {
        delete job;
        show({});
        g_in_progress = false;
        wakeword::resume();
        send_state("failed", -1, "could not start update task");
    }
}

bool in_progress()
{
    return g_in_progress.load();
}

void confirm_tick()
{
    // Read the image state once (it's in flash); after that only the
    // pending case keeps polling, and only until it resolves.
    static const esp_partition_t* running = nullptr;
    static bool checked = false;
    static bool settled = false;
    if (settled) return;
    if (!checked) {
        checked = true;
        running = esp_ota_get_running_partition();
        esp_ota_img_states_t img_state;
        // A USB-flashed image has no OTA state (or a valid one): nothing to do.
        if (running == nullptr || esp_ota_get_state_partition(running, &img_state) != ESP_OK
            || img_state != ESP_OTA_IMG_PENDING_VERIFY) {
            settled = true;
            return;
        }
        mclog::tagInfo(TAG, "{} is a new image pending verification", running->label);
    }
    if (transport::is_connected()) {
        mclog::tagInfo(TAG, "new image reached the brain; marking {} valid", running->label);
        esp_ota_mark_app_valid_cancel_rollback();
        settled = true;
        return;
    }
    if (state::now_ms() > kConfirmDeadlineMs) {
        mclog::tagError(TAG, "new image on {} never reached the brain in {} s; rolling back",
                        running->label, kConfirmDeadlineMs / 1000);
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

}  // namespace agent::ota
