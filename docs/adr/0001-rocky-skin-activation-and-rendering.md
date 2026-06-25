# Rocky avatar skin: activation and rendering architecture

## Status

accepted

## Context & Decision

We want a second avatar "skin" — Rocky, a faceless armored-creature character rendered as full-screen bitmap **body sprites** — that turns on and off together with the existing `ROCKY_MODE` (the Hume alien-voice persona), so a single toggle makes the robot "be Rocky" in both voice and face. The default kawaii face must remain fully intact when Rocky mode is off.

Two intertwined decisions:

1. **Activation = runtime swap coupled to `ROCKY_MODE`.** Both skins ship in the firmware build. `ROCKY_MODE` (a brain config knob, voice-toggleable) is the single source of truth; the brain emits a new `{"cmd":"set_skin","value":"rocky"|"default"}` whenever it changes and on every (re)connect for initial sync. The firmware swaps the live avatar under the display lock via the existing `StackChan::attachAvatar`.

2. **`RockyAvatar` reuses the `avatar::Avatar` framework rather than a separate render path.** It is a normal `Avatar` subclass holding one full-screen `lv_image` for the body (swapped in `setEmotion`), plus **no-op stub** `leftEye/rightEye/mouth` Features and a retained text label (for the pairing PIN and buddy "approve:" prompt). Overlays (confetti/alert/zzz) are implemented as trivial static-image `Decorator`s through the existing `addDecorator`/`removeDecorator` pool.

## Why this works (and why it's surprising)

The three registered modifiers (`Breath`, `Blink`, `HeadPet`) re-fetch `stackchan.avatar().leftEye()/mouth()/getEmotion()` **fresh every tick** — none cache a `Feature*` or `Avatar*`. So a live `attachAvatar` swap cannot dangle a pointer, and the modifiers need **no** teardown or re-registration; they simply poke the new avatar next tick. That is the load-bearing reason this is cheap. The cost: a *faceless* avatar must still implement eyes/mouth as no-op stubs purely to satisfy those modifiers — which will look wrong to a future reader unless they know the modifiers are unchanged and skin-agnostic by design.

## Considered alternatives (rejected)

- **Wholesale replacement** (retire the kawaii face): rejected — the product goal is one toggle that flips between two living skins.
- **Build-time skin selection**: rejected — couples skin to a reflash; `ROCKY_MODE` is a hot runtime knob.
- **Separate Rocky render path bypassing `Avatar`/modifiers**: rejected — would duplicate the modifier wiring and the swap machinery for no benefit, since the framework swap is already pointer-safe.

## Consequences

- A new firmware command `set_skin` and brain-side wiring (emit on `ROCKY_MODE` change + on connect).
- The Rocky skin is **display-only coupling**: it follows `ROCKY_MODE` but the brain remains the source of truth; firmware never decides the skin on its own.
- Asset budget ~390 KB rodata (4 bodies + 3 overlays, RGB565A8) against ~2 MB free in the app partition — comfortable.
