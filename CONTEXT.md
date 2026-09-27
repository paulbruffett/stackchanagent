# Stackchan

A single-purpose voice agent built on M5StackChan hardware: a thin ESP32 I/O client (`firmware/`) paired with a brain on the LAN (`brain/`). This glossary covers the terms specific to the project's domain — especially the avatar display model, where the language is easy to confuse.

## Language

### Avatar & skins

**Skin**:
A complete visual style for the on-screen character, owning how every expression and overlay is rendered. Only the _default skin_ ships; the `Avatar` virtuals (`setBusy`, `celebrate`) remain as the seam for another.

**Default skin**:
The stock kawaii face — vector-drawn eyes and mouth that morph per expression.
_Avoid_: kawaii avatar, M5 face.

**Rocky skin** / **Rocky mode** / **Body sprite** _(removed 2026-09-26)_:
The bitmap armored-creature skin, its `ROCKY_MODE` brain knob (voice persona + skin) and its full-screen per-expression sprites. Removed along with the `set_skin` command; see ADR 0001 and docs/plan-responsiveness.md.

**Overlay**:
A transient graphic layered _on top of_ the avatar to add a signal the face alone can't convey. In the framework code these are `Decorator`s.
_Avoid_: decorator (in product/design discussion), badge.

**Expression**:
A named emotional state the brain requests via `set_expression` (`neutral`, `happy`, `sad`, `angry`, `surprised`, `sleepy`). The skin decides how to render it.
_Avoid_: emotion (in product discussion — `Emotion` is the firmware enum, not all expressions map 1:1).
