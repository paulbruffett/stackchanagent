# stackchan firmware

ESP32-S3 firmware for the M5StackChan Kawaii. Thin I/O client; the brain
(Python agent on the LAN) does STT, LLM, TTS, and vision.

## Build & flash

```bash
# Install PlatformIO (one-time)
pipx install platformio

cd firmware
pio run -t upload && pio device monitor
```

## Current phase: 0 (bring-up)

`src/main.cpp` cycles display → servos → mic→speaker passthrough on boot.

Expected on flash:
1. Avatar face appears with "phase 0: bring-up".
2. Head tilts up/down then pans left/right within safe range.
3. Speech bubble shows "say something" — voice is echoed to the speaker for 3s.
4. Settles on a neutral face with "bring-up ok".

If any step doesn't fire, check serial @ 115200.

## Roadmap

| Phase | Adds |
|---|---|
| 1 | WebSocket transport to brain (mDNS), audio echo loop |
| 2 | ESP-SR "hey robot" wakeword + mic streaming |
| 3 | Expression/servo/motion-rate command receiver |
| 4 | GC0308 camera + JPEG frame pump |
| 5–6 | (brain-side only) |

## Wakeword (Phase 2)

Custom "hey robot" model from Espressif's online wakeword generator
(<https://customer.espressif.com/>). Drop the `.bin` into
`data/wakeword/` and register it in `src/wakeword.cpp`.

Fallback if the Espressif turnaround stalls: Porcupine custom keyword
(Picovoice). Requires their SDK on ESP32 and a per-device access key.
