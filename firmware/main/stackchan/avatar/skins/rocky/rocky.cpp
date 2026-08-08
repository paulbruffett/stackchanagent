/*
 * SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "rocky.h"
#include "../default/default.h"
#include <hal/hal.h>

using namespace uitk;
using namespace uitk::lvgl_cpp;
using namespace stackchan::avatar;

// Body sprites (full-screen, 160x160 RGB565A8) — committed under assets/.
LV_IMAGE_DECLARE(rocky_neutral);
LV_IMAGE_DECLARE(rocky_happy);
LV_IMAGE_DECLARE(rocky_angry);
LV_IMAGE_DECLARE(rocky_doubt);
LV_IMAGE_DECLARE(rocky_busy);

// Overlays.
LV_IMAGE_DECLARE(decorator_zzz);
LV_IMAGE_DECLARE(decorator_alert);
LV_IMAGE_DECLARE(decorator_confetti);

static const lv_color_t _bg_default = lv_color_black();
static const lv_color_t _bg_sad     = lv_color_hex(0x18306E);  // muted blue

static const uint32_t _confetti_lifetime_ms = 3000;

void RockyAvatar::init(lv_obj_t* parent, const lv_font_t* font)
{
    _pannel = std::make_unique<Container>(parent);
    _pannel->align(LV_ALIGN_CENTER, 0, 0);
    _pannel->setSize(320, 240);
    _pannel->setRadius(0);
    _pannel->setBorderWidth(0);
    _pannel->setBgColor(_bg_default);
    _pannel->removeFlag(LV_OBJ_FLAG_SCROLLABLE);

    // Full-screen body sprite (the steady-state Rocky render).
    _body = std::make_unique<Image>(_pannel->get());
    _body->setSrc(&rocky_neutral);
    _body->setAlign(LV_ALIGN_CENTER);
    _body->setPos(0, 0);

    // Single overlay slot, hidden until an emotion needs it.
    _overlay = std::make_unique<Image>(_pannel->get());
    _overlay->setAlign(LV_ALIGN_CENTER);
    lv_obj_add_flag(_overlay->get(), LV_OBJ_FLAG_HIDDEN);

    // Faceless: eyes/mouth are no-op stubs purely to satisfy the modifiers.
    _key_elements.leftEye  = std::make_unique<NullFeature>();
    _key_elements.rightEye = std::make_unique<NullFeature>();
    _key_elements.mouth    = std::make_unique<NullFeature>();
    // Reuse the default speech bubble for the pairing PIN and buddy "approve:"
    // prompt (set via setSpeech from buddy_ble / set_busy).
    _key_elements.speechBubble =
        std::make_unique<DefaultSpeechBubble>(_pannel->get(), lv_color_white(), lv_color_black(), font);
}

void RockyAvatar::update()
{
    Avatar::update();

    if (_overlay_duration_ms != 0 && (GetHAL().millis() - _overlay_set_at) >= _overlay_duration_ms) {
        setOverlay(Overlay::None);
    }
}

void RockyAvatar::setEmotion(const Emotion& emotion)
{
    // Keep _emotion + no-op features/decorators in sync (skin-agnostic base).
    Avatar::setEmotion(emotion);
    _busy = false;
    applyEmotion(emotion);
}

void RockyAvatar::setBusy(bool on)
{
    // celebrate() and setEmotion() already clear _busy, so this early-out makes
    // the brain's trailing `set_busy false` a no-op after either of them. That
    // matters: without it, busy-off repaints from _emotion and hides the
    // confetti a second into its 3 s life, because the brain only clears busy
    // when the reply's first sentence lands — one API round trip after the
    // set_expression("celebrate") that armed it.
    if (_busy == on) {
        return;
    }
    _busy = on;
    if (on) {
        showBody(&rocky_busy);
    } else {
        applyEmotion(getEmotion());
    }
}

void RockyAvatar::celebrate()
{
    if (!_pannel) {
        return;
    }
    Avatar::setEmotion(Emotion::Happy);
    _busy = false;
    _pannel->setBgColor(_bg_default);
    showBody(&rocky_happy);
    setOverlay(Overlay::Confetti, _confetti_lifetime_ms);
}

// showBody/setOverlay already guard their own objects; guard _pannel the same
// way here and in celebrate() so the whole skin follows one convention — the
// three objects are created together in init(), but nothing enforces that, and
// the old mix protected the cheap paths while leaving these two bare.
void RockyAvatar::applyEmotion(const Emotion& emotion)
{
    if (!_pannel) {
        return;
    }

    const lv_image_dsc_t* body = &rocky_neutral;
    Overlay overlay            = Overlay::None;
    lv_color_t bg              = _bg_default;

    switch (emotion) {
        case Emotion::Happy:
            body = &rocky_happy;
            break;
        case Emotion::Angry:
            body = &rocky_angry;
            break;
        case Emotion::Doubt:
            body    = &rocky_doubt;
            overlay = Overlay::Alert;  // covers the buddy "approve/attention" prompt
            break;
        case Emotion::Sad:
            // No sad sprite: neutral body on a blue background tint.
            body = &rocky_neutral;
            bg   = _bg_sad;
            break;
        case Emotion::Sleepy:
            body    = &rocky_neutral;
            overlay = Overlay::Zzz;
            break;
        case Emotion::Neutral:
        default:
            break;
    }

    _pannel->setBgColor(bg);
    showBody(body);
    setOverlay(overlay);
}

void RockyAvatar::showBody(const lv_image_dsc_t* body)
{
    if (_body) {
        _body->setSrc(body);
    }
}

void RockyAvatar::setOverlay(Overlay overlay, uint32_t autoClearMs)
{
    if (!_overlay) {
        return;
    }

    if (overlay == Overlay::None) {
        lv_obj_add_flag(_overlay->get(), LV_OBJ_FLAG_HIDDEN);
        _overlay_duration_ms = 0;
        return;
    }

    const lv_image_dsc_t* src = nullptr;
    lv_align_t align          = LV_ALIGN_CENTER;
    int x_ofs = 0, y_ofs = 0;
    switch (overlay) {
        case Overlay::Zzz:
            src   = &decorator_zzz;
            align = LV_ALIGN_TOP_RIGHT;
            x_ofs = -18;
            y_ofs = 18;
            break;
        case Overlay::Alert:
            src   = &decorator_alert;
            align = LV_ALIGN_TOP_MID;
            y_ofs = 12;
            break;
        case Overlay::Confetti:
            src = &decorator_confetti;  // full-screen, centered
            break;
        default:
            return;
    }

    _overlay->setSrc(src);
    _overlay->setAlign(align);
    _overlay->setPos(x_ofs, y_ofs);
    lv_obj_remove_flag(_overlay->get(), LV_OBJ_FLAG_HIDDEN);
    _overlay_set_at      = GetHAL().millis();
    _overlay_duration_ms = autoClearMs;
}
