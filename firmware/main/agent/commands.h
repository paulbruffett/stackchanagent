/*
 * Dispatch JSON commands from the brain to firmware state changes.
 * Registered with transport::set_on_json() during startup.
 *
 * Recognized commands (Phase 2):
 *   {"cmd":"stop_listening"}   LISTENING → IDLE  (cancel without speech)
 *   {"cmd":"start_speaking"}   LISTENING or IDLE → SPEAKING
 *   {"cmd":"stop_speaking"}    SPEAKING → IDLE
 */
#pragma once

#include <string_view>

namespace agent::commands {

void dispatch(std::string_view json);

}  // namespace agent::commands
