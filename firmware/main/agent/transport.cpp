#include "transport.h"

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ArduinoJson.h>
#include <board.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <mooncake_log.h>
#include <lwip/inet.h>
#include <lwip/netdb.h>
#include <lwip/sockets.h>
#include <web_socket.h>

#include "buddy_ble.h"
#include "commands.h"
#include "ota.h"
#include "state.h"

namespace agent::transport {

namespace {

constexpr const char* TAG = "agent.ws";

// Reconnect backoff bounds.
constexpr uint32_t kBackoffMinMs = 1000;
constexpr uint32_t kBackoffMaxMs = 16000;

// A session must last this long before it counts as healthy enough to reset
// the backoff ladder. A brain that accepts the handshake and dies immediately
// (systemd crash-loop) would otherwise pin us at kBackoffMinMs forever: a new
// socket, a new tcp_receive task and a boot/stop_speaking/set_buddy exchange
// every second for as long as it stays broken.
constexpr uint32_t kStableSessionMs = 10000;

struct State {
    std::string host;
    int port = 0;
    // shared_ptr, not unique_ptr: senders copy this under `mu`, drop the lock
    // and only then Send(), so the socket may be swapped out from under an
    // in-flight write. The reference the sender holds keeps the object alive
    // until that write returns.
    std::shared_ptr<WebSocket> ws;
    std::atomic<bool> connected{false};
    // Protects ws_ swap + callbacks installed on ws_.
    std::mutex mu;
    AudioFrameHandler on_audio;
    JsonFrameHandler on_json;
    // IPv4 address `host` resolved to for the current connection ("" if the
    // lookup failed). Guarded by `mu`.
    std::string brain_ip;
    // state::now_ms() of the last inbound frame (audio or JSON). Feeds the
    // turn watchdog in commands.cpp.
    std::atomic<int64_t> last_rx_ms{0};
};

State& state()
{
    static State s;
    return s;
}

std::string resolve_ipv4(const std::string& host)
{
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || res == nullptr) {
        mclog::tagWarn(TAG, "could not resolve {} for the OTA host check", host);
        return {};
    }
    char buf[INET_ADDRSTRLEN] = {};
    inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr, buf, sizeof buf);
    freeaddrinfo(res);
    return buf;
}

void handle_data(const char* data, size_t len, bool binary)
{
    auto& s = state();
    s.last_rx_ms.store(state::now_ms());
    if (binary) {
        if (len < 1) return;
        uint8_t op = static_cast<uint8_t>(data[0]);
        if (op == OP_AUDIO) {
            AudioFrameHandler h;
            {
                std::lock_guard<std::mutex> lock(s.mu);
                h = s.on_audio;
            }
            if (h) {
                // Payload after opcode is byte-aligned, not int16-aligned.
                // memcpy into an aligned buffer to avoid UB.
                size_t sample_count = (len - 1) / sizeof(int16_t);
                std::vector<int16_t> samples(sample_count);
                memcpy(samples.data(), data + 1,
                       sample_count * sizeof(int16_t));
                h(samples.data(), sample_count);
            }
        }
        // Other opcodes ignored for now (no JPEG handler yet).
    } else {
        JsonFrameHandler h;
        {
            std::lock_guard<std::mutex> lock(s.mu);
            h = s.on_json;
        }
        if (h) {
            h(std::string_view(data, len));
        }
    }
}

void connection_task(void*)
{
    auto& s = state();
    auto* network = Board::GetInstance().GetNetwork();
    if (!network) {
        mclog::tagError(TAG, "no network interface — task exiting");
        vTaskDelete(nullptr);
        return;
    }

    std::string uri = "ws://" + s.host + ":" + std::to_string(s.port) + "/";
    uint32_t backoff_ms = kBackoffMinMs;

    while (true) {
        mclog::tagInfo(TAG, "connecting: {}", uri);

        std::shared_ptr<WebSocket> ws = network->CreateWebSocket(1);
        if (!ws) {
            mclog::tagError(TAG, "CreateWebSocket failed; retry in {} ms", backoff_ms);
            // Same self-heal as the disconnect edge below: with no link there
            // is nobody to send stop_listening/stop_speaking, so a wakeword or
            // tap taken while the brain was down would strand us in LISTENING.
            state::transition(state::Mode::Idle);
            vTaskDelay(pdMS_TO_TICKS(backoff_ms));
            backoff_ms = std::min(backoff_ms * 2, kBackoffMaxMs);
            continue;
        }

        // Heap-allocated and captured by value: a sender blocked inside Send()
        // can keep this WebSocket (and therefore these callbacks) alive past
        // the end of this loop iteration, so they must not point at our stack.
        auto closed = std::make_shared<std::atomic<bool>>(false);
        ws->OnData([](const char* d, size_t l, bool b) { handle_data(d, l, b); });
        ws->OnDisconnected([closed]() {
            state().connected = false;
            closed->store(true);
        });
        ws->OnError([closed](int err) {
            mclog::tagWarn(TAG, "ws error: {}", err);
            state().connected = false;
            closed->store(true);
        });

        if (!ws->Connect(uri.c_str())) {
            mclog::tagWarn(TAG, "connect failed (err={}); retry in {} ms",
                           ws->GetLastError(), backoff_ms);
            state::transition(state::Mode::Idle);
            vTaskDelay(pdMS_TO_TICKS(backoff_ms));
            backoff_ms = std::min(backoff_ms * 2, kBackoffMaxMs);
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(s.mu);
            s.ws = std::move(ws);
        }
        // The address we reached the brain at, for ota::start's URL check.
        // Our own lookup (esp-ml307 doesn't expose the socket's peer); same
        // resolver the WebSocket just used, so it names the same host.
        {
            std::string ip = resolve_ipv4(s.host);
            std::lock_guard<std::mutex> lock(s.mu);
            s.brain_ip = std::move(ip);
        }
        s.connected = true;
        const TickType_t session_start = xTaskGetTickCount();
        mclog::tagInfo(TAG, "connected");
        // Report the BLE-buddy mode this boot is running in, so the brain
        // only sends set_buddy (which reboots us) when it actually differs,
        // plus battery/volume (the brain syncs SPEAKER_VOLUME off this) and
        // the running firmware, so the console can show it.
        {
            JsonDocument boot;
            boot["event"] = "boot";
            boot["buddy"] = buddy_ble::enabled();
            boot["fw"] = ota::running_version();
            boot["fw_built"] = ota::running_build();
            boot["fw_sha"] = ota::running_elf_sha256();
            boot["fw_valid"] = !ota::pending_verify();
            std::string json;
            serializeJson(boot, json);
            // status_fields() is a ready-made `"battery":…,"volume":…`
            // fragment; splice it in before the closing brace.
            json.insert(json.size() - 1, "," + commands::status_fields());
            send_event_json(json);
        }

        // Run until disconnect / error fires.
        while (!closed->load()) {
            vTaskDelay(pdMS_TO_TICKS(200));
        }

        mclog::tagInfo(TAG, "disconnected; will reconnect");
        s.connected = false;
        // M6.8 Fix A: a brain kill mid-turn strands the firmware in SPEAKING
        // (wakeword paused, tap gated to Idle); the WS reconnect alone never
        // resets it, so the device is dead to wakeword/tap until reboot.
        // Self-heal to Idle locally — this resumes the wakeword and ungates
        // the tap handler. We intentionally do NOT relight the screen here:
        // wake_face() would wrongly wake a legitimately-sleeping device on a
        // transient brain drop (a stranded turn is always in SPEAKING, screen
        // already on, so Idle is all that's needed).
        state::transition(state::Mode::Idle);
        {
            std::lock_guard<std::mutex> lock(s.mu);
            s.ws.reset();
        }
        // Only a session that actually stood up counts as success; otherwise
        // keep climbing the ladder so an accept-then-die brain gets backed off
        // instead of being hammered once a second.
        if (xTaskGetTickCount() - session_start >= pdMS_TO_TICKS(kStableSessionMs)) {
            backoff_ms = kBackoffMinMs;
        } else {
            backoff_ms = std::min(backoff_ms * 2, kBackoffMaxMs);
        }
        vTaskDelay(pdMS_TO_TICKS(backoff_ms));
    }
}

}  // namespace

void start(const char* host, int port)
{
    auto& s = state();
    s.host = host;
    s.port = port;
    xTaskCreatePinnedToCore(connection_task, "agent_ws", 6144, nullptr, 5,
                            nullptr, 0);
}

bool is_connected()
{
    return state().connected.load();
}

namespace {

bool send_binary(uint8_t op, const uint8_t* payload, size_t len)
{
    auto& s = state();
    if (!s.connected.load()) return false;

    std::vector<uint8_t> buf;
    buf.reserve(1 + len);
    buf.push_back(op);
    buf.insert(buf.end(), payload, payload + len);

    // Never hold s.mu across the write. EspTcp::Send loops on a blocking
    // socket with no SO_SNDTIMEO, so a peer that stops reading (brain event
    // loop wedged, AP flap) parks us in send() for lwIP's whole retransmit
    // budget — minutes. Holding the module mutex there froze the mic/camera
    // pumps, the wakeword and tap events, inbound command dispatch, and
    // connection_task itself, which needs s.mu to drop the socket and
    // reconnect. WebSocket::Send has its own send_mutex_, so concurrent
    // senders are still serialised.
    std::shared_ptr<WebSocket> ws;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        ws = s.ws;
    }
    if (!ws) return false;
    return ws->Send(buf.data(), buf.size(), /*binary=*/true);
}

}  // namespace

bool send_audio(const int16_t* samples, size_t sample_count)
{
    return send_binary(OP_AUDIO,
                       reinterpret_cast<const uint8_t*>(samples),
                       sample_count * sizeof(int16_t));
}

bool send_event_json(std::string_view json)
{
    auto& s = state();
    if (!s.connected.load()) return false;
    // Copy the socket out, then write outside the lock — see send_binary.
    std::shared_ptr<WebSocket> ws;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        ws = s.ws;
    }
    if (!ws) return false;
    return ws->Send(std::string(json));
}

std::string brain_ip()
{
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    return s.connected.load() ? s.brain_ip : std::string();
}

int64_t last_rx_ms()
{
    return state().last_rx_ms.load();
}

void set_on_audio(AudioFrameHandler handler)
{
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.on_audio = std::move(handler);
}

void set_on_json(JsonFrameHandler handler)
{
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.on_json = std::move(handler);
}

}  // namespace agent::transport
