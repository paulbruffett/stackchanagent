#include "transport.h"

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <board.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <mooncake_log.h>
#include <web_socket.h>

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
// socket, a new tcp_receive task and a boot/stop_speaking/set_skin exchange
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
};

State& state()
{
    static State s;
    return s;
}

void handle_data(const char* data, size_t len, bool binary)
{
    auto& s = state();
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
        s.connected = true;
        const TickType_t session_start = xTaskGetTickCount();
        mclog::tagInfo(TAG, "connected");
        send_event_json("{\"event\":\"boot\"}");

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

bool send_jpeg(const uint8_t* jpeg, size_t len)
{
    return send_binary(OP_JPEG, jpeg, len);
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
