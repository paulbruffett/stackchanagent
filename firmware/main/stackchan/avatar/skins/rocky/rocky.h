/*
 * SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "../../avatar/avatar.h"
#include "../../avatar/elements/feature.h"
#include <lvgl.h>
#include <smooth_lvgl.hpp>
#include <cstdint>
#include <memory>

namespace stackchan::avatar {

/**
 * @brief No-op Feature for a faceless skin.
 *
 * Rocky has no eyes or mouth, but the Breath/Blink/HeadPet modifiers poke
 * avatar().leftEye()/rightEye()/mouth() fresh every tick and do not know which
 * skin is active. So a faceless avatar must still expose eye/mouth Features —
 * these simply absorb the pokes and render nothing.
 */
class NullFeature : public Feature {
public:
    void setPosition(const uitk::Vector2i&) override {}
    void setRotation(int) override {}
    void setEmotion(const Emotion&) override {}
    void setVisible(bool) override {}
    void setWeight(int) override {}
    void setSize(int) override {}
};

/**
 * @brief Rocky skin — faceless armored creature rendered as full-screen body
 *        sprites that swap wholesale per expression, plus transient overlays.
 *
 * A normal Avatar subclass: the framework swap (StackChan::attachAvatar) and
 * the modifiers are skin-agnostic, so all Rocky-specific rendering lives here.
 */
class RockyAvatar : public Avatar {
public:
    void init(lv_obj_t* parent, const lv_font_t* font = &lv_font_montserrat_16);

    void update() override;
    void setEmotion(const Emotion& emotion) override;
    void setBusy(bool on) override;
    void celebrate() override;

private:
    enum class Overlay { None, Zzz, Alert, Confetti };

    void applyEmotion(const Emotion& emotion);  // body + overlay + tint
    void showBody(const lv_image_dsc_t* body);
    void setOverlay(Overlay overlay, uint32_t autoClearMs = 0);

    std::unique_ptr<uitk::lvgl_cpp::Container> _pannel;
    std::unique_ptr<uitk::lvgl_cpp::Image> _body;
    std::unique_ptr<uitk::lvgl_cpp::Image> _overlay;

    // Elapsed-since form, not an absolute deadline: Hal::millis() is a uint32
    // that wraps every ~49.7 days and this device is meant to run unattended
    // for weeks. Unsigned subtraction stays correct across the wrap.
    uint32_t _overlay_set_at      = 0;
    uint32_t _overlay_duration_ms = 0;  // 0 = no auto-clear
    bool _busy                    = false;
};

}  // namespace stackchan::avatar
