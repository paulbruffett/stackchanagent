/*
 * Link stub for xiaozhi's Application singleton.
 *
 * We never run xiaozhi's app (our own app_main in main.cpp drives the agent
 * modules), but a handful of kept xiaozhi translation units — wifi_board.cc,
 * display.cc, lvgl_display.cc, assets.cc — still reference
 * Application::GetInstance() on paths that don't execute in this firmware
 * (wifi config mode, app-driven UI updates). GetInstance() is inline in
 * application.h, so those references pull in the constructor and any
 * out-of-line methods they name. Defining them here lets us drop
 * application.cc, mcp_server.cc and the mqtt/websocket protocol stacks from
 * the build entirely (see main/CMakeLists.txt).
 *
 * The device state machine stays at its initial state, exactly as it did
 * when application.cc was compiled but never Start()ed.
 */
#include <application.h>

#include <esp_log.h>

static const char* TAG = "app_stub";

Application::Application() = default;

Application::~Application() = default;

bool Application::SetDeviceState(DeviceState state) {
    return state_machine_.TransitionTo(state);
}

void Application::Schedule(std::function<void()>&& callback) {
    // No main loop to run it on; these only come from wifi-config paths we
    // never enter.
    ESP_LOGW(TAG, "Schedule() ignored (xiaozhi app not running)");
}

void Application::Alert(const char* status, const char* message,
                        const char* emotion, const std::string_view& sound) {
    ESP_LOGW(TAG, "Alert: %s — %s", status ? status : "",
             message ? message : "");
}

void Application::ResetProtocol() {
    protocol_.reset();
}

void Application::PlaySound(const std::string_view& sound) {
    // Sound effects were only ever played by the xiaozhi app's UI flows.
}

bool Application::CanEnterSleepMode() {
    // Mirrors the real implementation's first clause: the state machine
    // never leaves its initial state in this firmware, so this is always
    // false — PowerSaveTimer stays inert, exactly as before the strip.
    return state_machine_.GetState() == kDeviceStateIdle;
}
