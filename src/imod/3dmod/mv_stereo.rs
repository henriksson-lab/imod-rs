//! Translation of `IMOD/3dmod/mv_stereo.cpp` together with `mv_stereo.h`.
#![allow(dead_code, unused_variables)]

use crate::imod::three_dmod::imodv::{IMODV_STEREO_OFF, ImodvApp};
use crate::imod::three_dmod::mv_gfx::{IMODV_STEREO_HW, IMODV_STEREO_RL, IMODV_STEREO_TB};
use std::env;
use std::process::Command;

/// `imodvStereoData`.
#[derive(Clone, Debug)]
pub struct ImodvStereoData {
    pub init: i32,
    pub hw: i32,
    pub dialog_open: bool,
    pub tb_voffset: i32,
    pub cw: i32,
    pub omode: i32,
    pub width: i32,
    pub height: i32,
    pub x: i32,
    pub y: i32,
    pub rad: f32,
    pub stereo_command: Option<String>,
    pub restore_command: Option<String>,
}
impl Default for ImodvStereoData {
    fn default() -> Self {
        Self {
            init: 0,
            hw: 0,
            dialog_open: false,
            tb_voffset: 0,
            cw: 0,
            omode: 0,
            width: 0,
            height: 0,
            x: 0,
            y: 0,
            rad: 0.,
            stereo_command: None,
            restore_command: None,
        }
    }
}
/// State of Qt widgets in `ImodvStereo`; the actual form is a paired GUI boundary.
#[derive(Clone, Debug)]
pub struct ImodvStereo {
    pub combo_items: Vec<String>,
    pub combo_index: i32,
    pub slider_angle: i32,
    pub slider_offset: i32,
    pub image_checked: bool,
    pub images_enabled: bool,
    pub ctrl_pressed: bool,
    pub closed: bool,
}
impl ImodvStereo {
    pub fn new(a: &ImodvApp, data: &ImodvStereoData) -> Self {
        Self {
            combo_items: vec![
                "Stereo Off".into(),
                "Side by Side".into(),
                "Top / Bottom".into(),
            ],
            combo_index: a.stereo,
            slider_angle: 0,
            slider_offset: data.tb_voffset,
            image_checked: a.image_stereo != 0,
            images_enabled: false,
            ctrl_pressed: false,
            closed: false,
        }
    }
    pub fn new_option(&mut self, a: &mut ImodvApp, item: i32) {
        a.stereo = item;
    }
    pub fn slider_moved(
        &mut self,
        a: &mut ImodvApp,
        data: &mut ImodvStereoData,
        which: i32,
        value: i32,
        dragging: bool,
        hot_slider_active: bool,
    ) -> bool {
        if which != 0 {
            data.tb_voffset = value;
            self.slider_offset = value;
        } else {
            a.plax = value as f32 / 10.;
            self.slider_angle = value;
        }
        !dragging || hot_slider_active
    }
    pub fn image_toggled(&mut self, a: &mut ImodvApp, state: bool) {
        a.image_stereo = state as i32;
        self.image_checked = state;
        self.update(a, data_hardware_ok(a), true);
    }
    pub fn views_changed(&mut self, a: &mut ImodvApp, value: i32) {
        a.images_per_area = value;
        a.image_delta_z = a.image_delta_z.min(a.images_per_area - 1);
    }
    pub fn delta_changed(&mut self, a: &mut ImodvApp, value: i32) {
        a.image_delta_z = value;
    }
    pub fn button_pressed(&mut self, which: i32) -> Option<&'static str> {
        if which != 0 {
            Some("modvStereo.html#TOP")
        } else {
            self.closed = true;
            None
        }
    }
    pub fn update(&mut self, a: &mut ImodvApp, hardware_ok: bool, byte_images_exist: bool) {
        a.plax = a.plax.clamp(-10., 10.);
        self.slider_angle = (a.plax * 10. + 0.5).floor() as i32;
        if hardware_ok && self.combo_items.len() == 3 {
            self.combo_items.push("Hardware".into());
        }
        if !hardware_ok && self.combo_items.len() == 4 {
            self.combo_items.remove(IMODV_STEREO_HW as usize);
        }
        self.combo_index = a.stereo;
        self.images_enabled = byte_images_exist && a.tex_map != 0;
    }
    pub fn manage_hw_label(
        &self,
        hardware_ok: bool,
        key_sets_hw_stereo: bool,
        macos: bool,
    ) -> bool {
        macos && hardware_ok && !key_sets_hw_stereo
    }
    pub fn change_event(&mut self) {}
    pub fn close_event(&mut self) {
        self.closed = true;
    }
    pub fn key_press_event(&mut self, close_key: bool, hot_slider_key: bool) -> bool {
        if close_key {
            self.closed = true;
        } else if hot_slider_key {
            self.ctrl_pressed = true;
        }
        self.closed
    }
    pub fn key_release_event(&mut self, hot_slider_key: bool) {
        if hot_slider_key {
            self.ctrl_pressed = false;
        }
    }
}
/// Static `hardwareOK`.
pub fn data_hardware_ok(a: &ImodvApp) -> bool {
    (a.dbl_buf != 0 && a.enable_depth_dbst >= 0) || (a.dbl_buf == 0 && a.enable_depth_sbst >= 0)
}
/// Static `stereoInit`.
pub fn stereo_init(a: &ImodvApp, data: &mut ImodvStereoData) {
    if data.init != 0 {
        return;
    }
    data.omode = IMODV_STEREO_HW;
    data.init = 1;
    data.stereo_command = env::var("IMOD_STEREO_COMMAND").ok();
    data.restore_command = env::var("IMOD_STEREO_RESTORE").ok();
    if let Ok(v) = env::var("IMOD_STEREO_TBOFFSET") {
        if let Ok(v) = v.parse() {
            data.tb_voffset = v;
            data.omode = IMODV_STEREO_TB;
        }
    }
}
/// Static `stereoEnable`.
pub fn stereo_enable(
    a: &mut ImodvApp,
    data: &mut ImodvStereoData,
    set_buffer: &mut dyn FnMut(i32, i32, i32),
) {
    set_buffer(-1, 1, -1);
    if let Some(command) = &data.stereo_command {
        let _ = Command::new("sh").arg("-c").arg(command).status();
    }
}
/// Static `stereoDisable`.
pub fn stereo_disable(
    a: &mut ImodvApp,
    data: &mut ImodvStereoData,
    set_buffer: &mut dyn FnMut(i32, i32, i32),
) {
    set_buffer(-1, 0, -1);
    stereo_hw_off(data);
}
/// `stereoHWOff`.
pub fn stereo_hw_off(data: &mut ImodvStereoData) {
    if data.hw != 0 {
        if let Some(command) = &data.restore_command {
            let _ = Command::new("sh").arg("-c").arg(command).status();
        }
    }
    data.hw = 0;
}
/// `stereoDrawBuffer`; real GL draw-buffer selection is owned by `mv_gfx.cpp`.
pub fn stereo_draw_buffer(mode: u32, draw_buffer: &mut dyn FnMut(u32)) {
    draw_buffer(mode);
}
/// `imodvStereoVoffset`.
pub fn imodv_stereo_voffset(a: &ImodvApp, data: &ImodvStereoData, screen_height: i32) -> i32 {
    (screen_height / 2 - a.winy) + data.tb_voffset
}
/// `imodvStereoClear`.
pub fn imodv_stereo_clear(clear: &mut dyn FnMut()) {
    clear();
}
/// Static `stereoSetUp`; geometry/presentation is passed to the native window boundary.
pub fn stereo_set_up(
    a: &mut ImodvApp,
    data: &mut ImodvStereoData,
    geometry: &mut dyn FnMut(i32, i32, i32, i32),
    set_buffer: &mut dyn FnMut(i32, i32, i32),
    draw: &mut dyn FnMut(),
) {
    if a.stereo == IMODV_STEREO_HW || a.stereo == IMODV_STEREO_TB {
        data.width = a.winx;
        data.height = a.winy;
        data.rad = unsafe { a.imod.as_ref() }
            .and_then(|m| m.view.first())
            .map_or(0., |v| v.rad);
        if a.stereo == IMODV_STEREO_HW {
            data.hw = 1;
            stereo_enable(a, data, set_buffer);
        }
    } else if data.omode == IMODV_STEREO_TB || data.hw != 0 {
        if data.hw != 0 {
            stereo_disable(a, data, set_buffer);
            a.clear_after_stereo = 1;
        }
        geometry(data.x, data.y, data.width, data.height);
    }
    if a.stereo != IMODV_STEREO_OFF {
        data.omode = a.stereo;
    }
    draw();
}
/// `imodvStereoToggle`.
pub fn imodv_stereo_toggle(
    a: &mut ImodvApp,
    data: &mut ImodvStereoData,
    key_sets_hw_stereo: bool,
    geometry: &mut dyn FnMut(i32, i32, i32, i32),
    set_buffer: &mut dyn FnMut(i32, i32, i32),
    draw: &mut dyn FnMut(),
) {
    stereo_init(a, data);
    if a.stereo != IMODV_STEREO_OFF {
        a.stereo = IMODV_STEREO_OFF;
    } else {
        a.stereo = data.omode;
        if a.stereo == IMODV_STEREO_HW && (!data_hardware_ok(a) || !key_sets_hw_stereo) {
            a.stereo = IMODV_STEREO_RL;
        }
    }
    stereo_set_up(a, data, geometry, set_buffer, draw);
}
/// `imodvStereoUpdate`.
pub fn imodv_stereo_update(dialog: Option<&mut ImodvStereo>, a: &mut ImodvApp) {
    if let Some(dialog) = dialog {
        dialog.update(a, data_hardware_ok(a), false);
    }
}
/// `imodvStereoEditDialog`.
pub fn imodv_stereo_edit_dialog(
    a: &ImodvApp,
    data: &mut ImodvStereoData,
    state: i32,
) -> Option<ImodvStereo> {
    if state == 0 {
        data.dialog_open = false;
        None
    } else {
        stereo_init(a, data);
        if data.dialog_open {
            None
        } else {
            data.dialog_open = true;
            Some(ImodvStereo::new(a, data))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn toggle_falls_back_to_side_by_side_without_hardware() {
        let mut a = ImodvApp::default();
        let mut d = ImodvStereoData::default();
        let mut g = |_, _, _, _| {};
        let mut b = |_, _, _| {};
        let mut draw = || {};
        imodv_stereo_toggle(&mut a, &mut d, false, &mut g, &mut b, &mut draw);
        assert_eq!(a.stereo, IMODV_STEREO_RL);
    }
    #[test]
    fn update_clamps_angle() {
        let mut a = ImodvApp {
            plax: 20.,
            ..Default::default()
        };
        let mut d = ImodvStereoData::default();
        let mut s = ImodvStereo::new(&a, &d);
        s.update(&mut a, false, false);
        assert_eq!(a.plax, 10.);
    }
}
