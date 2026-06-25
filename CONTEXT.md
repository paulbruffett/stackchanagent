# Stackchan

A single-purpose voice agent built on M5StackChan hardware: a thin ESP32 I/O client (`firmware/`) paired with a brain on the LAN (`brain/`). This glossary covers the terms specific to the project's domain — especially the avatar display model, where the language is easy to confuse.

## Language

### Avatar & skins

**Skin**:
A complete visual style for the on-screen character, owning how every expression and overlay is rendered. Exactly one skin is active at a time. The two skins are the _default skin_ and the _Rocky skin_.

**Default skin**:
The stock kawaii face — vector-drawn eyes and mouth that morph per expression. Active whenever Rocky mode is off.
_Avoid_: kawaii avatar, M5 face.

**Rocky skin**:
A faceless armored-creature character rendered as full-screen bitmap _body sprites_ that swap wholesale per expression. Active whenever Rocky mode is on. Rocky has no eyes or mouth.

**Rocky mode**:
The single toggle that makes the robot "be Rocky" — both the Hume alien voice persona _and_ the Rocky skin. Backed by the brain's `ROCKY_MODE` config knob.

**Body sprite**:
A full-screen bitmap that is the Rocky skin's steady-state rendering of one expression (e.g. `rocky_neutral`, `rocky_angry`). Mutually exclusive — one body sprite shows at a time.
_Avoid_: avatar image, frame.

**Overlay**:
A transient graphic layered _on top of_ the current body sprite to add a signal the body alone can't convey (e.g. `confetti`, `alert`, `zzz`). Composes with a body sprite rather than replacing it. In the framework code these are `Decorator`s.
_Avoid_: decorator (in product/design discussion), badge.

**Expression**:
A named emotional state the brain requests via `set_expression` (`neutral`, `happy`, `sad`, `angry`, `surprised`, `sleepy`). Each skin decides how to render it; the Rocky skin may render one expression as a body sprite, a body sprite + overlay, or a reused body sprite.
_Avoid_: emotion (in product discussion — `Emotion` is the firmware enum, not all expressions map 1:1).
