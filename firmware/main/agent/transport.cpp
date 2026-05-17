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

namespace agent::transport {

namespace {

constexpr const char* TAG = "agent.ws";

// Reconnect backoff bounds.
constexpr uint32_t kBackoffMinMs = 1000;
constexpr uint32_t kBackoffMaxMs = 16000;

struct State {
    std::string host;
    int port = 0;
    std::unique_ptr<WebSocket> ws;
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

        auto ws = network->CreateWebSocket(1);
        if (!ws) {
            mclog::tagError(TAG, "CreateWebSocket failed; retry in {} ms", backoff_ms);
            vTaskDelay(pdMS_TO_TICKS(backoff_ms));
            backoff_ms = std::min(backoff_ms * 2, kBackoffMaxMs);
            continue;
        }

        std::atomic<bool> closed{false};
        ws->OnData([](const char* d, size_t l, bool b) { handle_data(d, l, b); });
        ws->OnDisconnected([&closed]() {
            state().connected = false;
            closed = true;
        });
        ws->OnError([&closed](int err) {
            mclog::tagWarn(TAG, "ws error: {}", err);
            state().connected = false;
            closed = true;
        });

        if (!ws->Connect(uri.c_str())) {
            mclog::tagWarn(TAG, "connect failed (err={}); retry in {} ms",
                           ws->GetLastError(), backoff_ms);
            vTaskDelay(pdMS_TO_TICKS(backoff_ms));
            backoff_ms = std::min(backoff_ms * 2, kBackoffMaxMs);
            continue;
        }

        backoff_ms = kBackoffMinMs;
        {
            std::lock_guard<std::mutex> lock(s.mu);
            s.ws = std::move(ws);
        }
        s.connected = true;
        mclog::tagInfo(TAG, "connected");
        send_event_json("{\"event\":\"boot\"}");

        // Run until disconnect / error fires.
        while (!closed) {
            vTaskDelay(pdMS_TO_TICKS(200));
        }

        mclog::tagInfo(TAG, "disconnected; will reconnect");
        s.connected = false;
        {
            std::lock_guard<std::mutex> lock(s.mu);
            s.ws.reset();
        }
        vTaskDelay(pdMS_TO_TICKS(kBackoffMinMs));
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

bool send_audio(const int16_t* samples, size_t sample_count)
{
    auto& s = state();
    if (!s.connected.load()) return false;

    size_t payload_bytes = sample_count * sizeof(int16_t);
    std::vector<uint8_t> buf;
    buf.reserve(1 + payload_bytes);
    buf.push_back(OP_AUDIO);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(samples);
    buf.insert(buf.end(), p, p + payload_bytes);

    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.ws) return false;
    return s.ws->Send(buf.data(), buf.size(), /*binary=*/true);
}

bool send_event_json(std::string_view json)
{
    auto& s = state();
    if (!s.connected.load()) return false;
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.ws) return false;
    return s.ws->Send(std::string(json));
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
