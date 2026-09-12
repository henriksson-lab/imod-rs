//! Translation of `IMOD/3dmod/mv_control.cpp` and `mv_control.h`.
//!
//! The Qt form itself remains owned by the `formv_control.cpp` translation
//! closure.  This unit deliberately retains the source's state mutations;
//! callers pass the `ImodvApp` selected by the window/event boundary.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::imodel::Imod;
use crate::imod::three_dmod::imodv::{
    ImodvApp, imodv_draw, imodv_draw_imod_images, imodv_finish_chg_unit, imodv_new_model_angles,
    imodv_register_model_chg,
};
use crate::imod::three_dmod::mv_input::{imodv_rotate_model, imodv_zoomd};

pub const IMODV_ROTATION_FACTOR: f32 = 1.26;
pub const IMODV_CONTROL_NEAR: i32 = 1;
pub const IMODV_CONTROL_FAR: i32 = 2;
pub const IMODV_CONTROL_FOVY: i32 = 3;
pub const IMODV_CONTROL_ZSCALE: i32 = 4;
pub const IMODV_CONTROL_XAXIS: i32 = 1;
pub const IMODV_CONTROL_YAXIS: i32 = 2;
pub const IMODV_CONTROL_ZAXIS: i32 = 3;

const ROTATION_MAX: i32 = 100;

/// File-static `sDialog`, `sTopWin`, and last-value fields from `mv_control.cpp`.
#[derive(Clone, Debug)]
pub struct ImodvControlDialogState {
    pub dialog_open: bool,
    pub last_x: f32,
    pub last_y: f32,
    pub last_z: f32,
    pub last_scale: f32,
    pub clip_sliding: bool,
    pub zscale_sliding: bool,
}
impl Default for ImodvControlDialogState {
    fn default() -> Self {
        Self {
            dialog_open: false,
            last_x: -999.,
            last_y: -999.,
            last_z: -999.,
            last_scale: -999.,
            clip_sliding: false,
            zscale_sliding: false,
        }
    }
}

/// `imodvControlForm`/docking operations at the source form boundary.
pub trait ImodvControlNativeBoundary {
    fn remove_dialog(&mut self);
    fn close_dialog(&mut self);
    fn model_edit_update(&mut self);
    fn rotation_center_state(&mut self, state: bool);
    fn rotation_step_label(&mut self, step: f32);
    fn raise_dialog(&mut self);
    fn create_dialog(&mut self) -> bool;
    fn set_axis_text(&mut self, axis: i32, value: f32);
    fn set_scale_text(&mut self, value: f32);
    fn set_rotation_rate(&mut self, value: f32);
    fn set_speed_text(&mut self, value: f32);
    fn set_view_slider(&mut self, slider: i32, value: i32);
    fn set_kick_box(&mut self, checked: bool);
    fn update_slicer_link(&mut self, value: i32);
}

/// Original: `imodvControlZoom`.
pub fn imodv_control_zoom(a: &mut ImodvApp, zoom: i32) {
    imodv_zoomd(a, if zoom > 0 { 1.05 } else { 0.95238095 });
    unsafe { imodv_draw() };
}

/// Original: `imodvControlKickClips`.
pub fn imodv_control_kick_clips(
    a: &mut ImodvApp,
    n: &mut dyn ImodvControlNativeBoundary,
    state: bool,
) {
    // WORLD_KICKOUT_CLIPS is a source flag.  Preserve all other view bits.
    const WORLD_KICKOUT_CLIPS: u32 = 0x20;
    let first = if a.crosset != 0 {
        0
    } else {
        a.cur_mod.max(0) as usize
    };
    let last = if a.crosset != 0 {
        a.mod_.len()
    } else {
        first.saturating_add(1).min(a.mod_.len())
    };
    imodv_register_model_chg();
    for model in &a.mod_[first..last] {
        if let Some(view) = unsafe { model.as_mut() }.and_then(|m| m.view.first_mut()) {
            if state {
                view.world |= WORLD_KICKOUT_CLIPS;
            } else {
                view.world &= !WORLD_KICKOUT_CLIPS;
            }
        }
    }
    imodv_finish_chg_unit();
    unsafe { imodv_draw() };
}

/// Original: `imodvControlClip`.
pub fn imodv_control_clip(
    a: &mut ImodvApp,
    state: &mut ImodvControlDialogState,
    n: &mut dyn ImodvControlNativeBoundary,
    plane: i32,
    value: i32,
    dragging: bool,
) {
    let Some(imod) = (unsafe { a.imod.as_mut() }) else {
        return;
    };
    let Some(view) = imod.view.first_mut() else {
        return;
    };
    if !state.clip_sliding {
        imodv_register_model_chg();
        imodv_finish_chg_unit();
    }
    state.clip_sliding = dragging;
    match plane {
        IMODV_CONTROL_FAR => {
            a.cfar = value;
            if a.cnear >= a.cfar {
                a.cnear = a.cfar - 1;
                n.set_view_slider(IMODV_CONTROL_NEAR, a.cnear);
            }
        }
        IMODV_CONTROL_NEAR => {
            a.cnear = value;
            if a.cfar <= a.cnear {
                a.cfar = a.cnear + 1;
                n.set_view_slider(IMODV_CONTROL_FAR, a.cfar);
            }
        }
        IMODV_CONTROL_FOVY => {
            a.fovy = value;
        }
        _ => return,
    }
    view.cnear = a.cnear as f32 * 0.001;
    view.cfar = a.cfar as f32 * 0.001;
    view.fovy = a.fovy as f32;
    if a.crosset != 0 {
        for model in &a.mod_ {
            if let Some(other) = unsafe { model.as_mut() }.and_then(|m| m.view.first_mut()) {
                other.fovy = view.fovy;
                if plane == IMODV_CONTROL_FAR {
                    other.cfar = view.cfar;
                    if other.cnear >= other.cfar {
                        other.cnear = other.cfar - 0.001;
                    }
                }
                if plane == IMODV_CONTROL_NEAR {
                    other.cnear = view.cnear;
                    if other.cfar <= other.cnear {
                        other.cfar = other.cnear + 0.001;
                    }
                }
            }
        }
    }
    unsafe { imodv_draw() };
}

/// Original: `imodvControlZscale`.
pub fn imodv_control_zscale(
    a: &mut ImodvApp,
    state: &mut ImodvControlDialogState,
    n: &mut dyn ImodvControlNativeBoundary,
    value: i32,
    dragging: bool,
) {
    if !state.zscale_sliding {
        imodv_register_model_chg();
        imodv_finish_chg_unit();
    }
    state.zscale_sliding = dragging;
    if let Some(model) = a
        .mod_
        .get(a.cur_mod.max(0) as usize)
        .and_then(|m| unsafe { m.as_mut() })
    {
        model.zscale = value as f32 / 100.;
    }
    unsafe { imodv_draw() };
    imodv_draw_imod_images(0);
    n.model_edit_update();
}

/// Original: `imodvControlScale`.
pub fn imodv_control_scale(a: &mut ImodvApp, scale: f32) {
    if scale == 0. {
        return;
    }
    let rad = 0.5 * a.winx.min(a.winy) as f32 / scale;
    let models: &[*mut Imod] = if a.crosset != 0 {
        &a.mod_
    } else {
        a.mod_
            .get(a.cur_mod.max(0) as usize)
            .map(std::slice::from_ref)
            .unwrap_or(&[])
    };
    for model in models {
        if let Some(view) = unsafe { model.as_mut() }.and_then(|m| m.view.first_mut()) {
            view.rad = rad;
        }
    }
    unsafe { imodv_draw() };
}

/// Original: `imodvControlStart`.
pub fn imodv_control_start(a: &mut ImodvApp, n: &mut dyn ImodvControlNativeBoundary) {
    a.movie = if a.movie != 0 { 0 } else { 1 };
    a.xrot_movie = 0.;
    a.yrot_movie = 0.;
    a.zrot_movie = 0.;
    unsafe { imodv_draw() };
    n.rotation_center_state(a.movie == 1);
}

/// Original: `imodvControlAxisButton`.
pub fn imodv_control_axis_button(a: &mut ImodvApp, axis_dir: i32) {
    let d = a.delta_rot as i32;
    match axis_dir {
        IMODV_CONTROL_XAXIS => imodv_rotate_model(a, d, 0, 0),
        -1 => imodv_rotate_model(a, -d, 0, 0),
        IMODV_CONTROL_YAXIS => imodv_rotate_model(a, 0, d, 0),
        -2 => imodv_rotate_model(a, 0, -d, 0),
        IMODV_CONTROL_ZAXIS => imodv_rotate_model(a, 0, 0, d),
        -3 => imodv_rotate_model(a, 0, 0, -d),
        _ => {}
    }
    unsafe { imodv_draw() };
}

/// Original: `imodvControlAxisText`.
pub fn imodv_control_axis_text(
    a: &mut ImodvApp,
    state: &mut ImodvControlDialogState,
    axis: i32,
    rot: f32,
) {
    let first = if a.moveall == 0 {
        a.cur_mod.max(0) as usize
    } else {
        0
    };
    let last = if a.moveall == 0 {
        first.saturating_add(1).min(a.mod_.len())
    } else {
        a.mod_.len()
    };
    for model in &a.mod_[first..last] {
        if let Some(view) = unsafe { model.as_mut() }.and_then(|m| m.view.first_mut()) {
            match axis {
                IMODV_CONTROL_XAXIS => view.rot.x = rot,
                IMODV_CONTROL_YAXIS => view.rot.y = rot,
                IMODV_CONTROL_ZAXIS => view.rot.z = rot,
                _ => {}
            }
        }
    }
    match axis {
        IMODV_CONTROL_XAXIS => state.last_x = -999.,
        IMODV_CONTROL_YAXIS => state.last_y = -999.,
        IMODV_CONTROL_ZAXIS => state.last_z = -999.,
        _ => {}
    }
    if let Some(imod) = unsafe { a.imod.as_ref() }.and_then(|m| m.view.first()) {
        imodv_new_model_angles(&imod.rot);
    }
    unsafe { imodv_draw() };
}

/// Original: `imodvControlRate`.
pub fn imodv_control_rate(a: &mut ImodvApp, n: &mut dyn ImodvControlNativeBoundary, value: i32) {
    a.delta_rot = value as f32;
    n.rotation_step_label(a.delta_rot / 10.);
}
/// Original: `imodvControlSpeed`.
pub fn imodv_control_speed(a: &mut ImodvApp, value: f32) {
    a.movie_speed = value;
}
/// Original: `imodvControlIncSpeed`.
pub fn imodv_control_inc_speed(
    a: &mut ImodvApp,
    state: &ImodvControlDialogState,
    n: &mut dyn ImodvControlNativeBoundary,
    step: i32,
) {
    a.movie_speed = if step > 0 {
        (a.movie_speed * IMODV_ROTATION_FACTOR).min(360.)
    } else {
        (a.movie_speed / IMODV_ROTATION_FACTOR).max(3.6)
    };
    if state.dialog_open {
        n.set_speed_text(a.movie_speed);
    }
}
/// Original: `imodvControlClosing`; Qt ownership boundary.
pub fn imodv_control_closing(
    state: &mut ImodvControlDialogState,
    n: &mut dyn ImodvControlNativeBoundary,
) {
    n.remove_dialog();
    state.dialog_open = false;
}
/// Original: `imodvControlQuit`; Qt ownership boundary.
pub fn imodv_control_quit(n: &mut dyn ImodvControlNativeBoundary) {
    n.close_dialog();
}
/// Original: `imodvControlChangeSteps`.
pub fn imodv_control_change_steps(
    a: &mut ImodvApp,
    state: &ImodvControlDialogState,
    n: &mut dyn ImodvControlNativeBoundary,
    delta: i32,
) {
    let mut newval = if delta > 0 {
        (a.delta_rot * IMODV_ROTATION_FACTOR).round()
    } else {
        (a.delta_rot / IMODV_ROTATION_FACTOR).round()
    };
    if newval == a.delta_rot {
        newval += delta as f32;
    }
    imodv_control_set_arot(a, state, n, newval.max(1.) as i32);
    imodv_control_inc_speed(a, state, n, delta);
}
/// Original: `imodvControlSetArot`.
pub fn imodv_control_set_arot(
    a: &mut ImodvApp,
    state: &ImodvControlDialogState,
    n: &mut dyn ImodvControlNativeBoundary,
    newval: i32,
) {
    imodv_control_rate(a, n, newval.min(ROTATION_MAX));
    if state.dialog_open {
        n.set_rotation_rate(a.delta_rot);
    }
}
/// Original: `imodvControlSetView`.
pub fn imodv_control_set_view(
    a: &mut ImodvApp,
    state: &ImodvControlDialogState,
    n: &mut dyn ImodvControlNativeBoundary,
) {
    if let Some(v) = unsafe { a.imod.as_ref() }.and_then(|m| m.view.first()) {
        a.fovy = v.fovy as i32;
        a.cnear = (v.cnear * 1000. + 0.5) as i32;
        a.cfar = (v.cfar * 1000. + 0.5) as i32;
        if !state.dialog_open {
            return;
        }
        n.set_view_slider(IMODV_CONTROL_NEAR, a.cnear);
        n.set_view_slider(IMODV_CONTROL_FAR, a.cfar);
        n.set_view_slider(IMODV_CONTROL_FOVY, a.fovy);
        n.set_view_slider(
            IMODV_CONTROL_ZSCALE,
            (unsafe { a.imod.as_ref() }.unwrap().zscale * 100. + 0.5) as i32,
        );
        n.set_kick_box((v.world & 0x20) != 0);
    }
}
/// Original: `imodvControlUpdate`; form update is owned by `formv_control.cpp`.
pub fn imodv_control_update(
    a: &mut ImodvApp,
    state: &mut ImodvControlDialogState,
    n: &mut dyn ImodvControlNativeBoundary,
) {
    if !state.dialog_open {
        return;
    }
    let Some(view) = unsafe { a.imod.as_ref() }.and_then(|m| m.view.first()) else {
        return;
    };
    for (value, last, axis) in [
        (view.rot.x, &mut state.last_x, IMODV_CONTROL_XAXIS),
        (view.rot.y, &mut state.last_y, IMODV_CONTROL_YAXIS),
        (view.rot.z, &mut state.last_z, IMODV_CONTROL_ZAXIS),
    ] {
        if *last != value {
            *last = value;
            n.set_axis_text(axis, value);
        }
    }
    let scale = 0.5 * a.winx.min(a.winy) as f32 / view.rad;
    if state.last_scale != scale {
        state.last_scale = scale;
        n.set_scale_text(scale);
    }
}
/// Original: `imodvControlLinkUpdate`; form update is owned by `formv_control.cpp`.
pub fn imodv_control_link_update(
    a: &mut ImodvApp,
    state: &ImodvControlDialogState,
    n: &mut dyn ImodvControlNativeBoundary,
) {
    if a.standalone == 0 && state.dialog_open {
        n.update_slicer_link(a.link_to_slicer);
    }
}
/// Original: `imodv_control`; creates, closes, or raises the Qt form.
pub fn imodv_control(
    a: &mut ImodvApp,
    dialog_state: &mut ImodvControlDialogState,
    n: &mut dyn ImodvControlNativeBoundary,
    state: i32,
) -> i32 {
    if state == 0 {
        if dialog_state.dialog_open {
            n.close_dialog();
        }
        -1
    } else if dialog_state.dialog_open {
        n.raise_dialog();
        -1
    } else if !n.create_dialog() {
        -1
    } else {
        dialog_state.dialog_open = true;
        dialog_state.last_x = -999.;
        dialog_state.last_y = -999.;
        dialog_state.last_z = -999.;
        dialog_state.last_scale = -999.;
        imodv_control_update(a, dialog_state, n);
        imodv_control_set_arot(a, dialog_state, n, a.delta_rot as i32);
        imodv_control_set_view(a, dialog_state, n);
        n.set_speed_text(a.movie_speed);
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::Imod;

    #[derive(Default)]
    struct Native {
        create: bool,
        closes: usize,
        raises: usize,
        model_edit_updates: usize,
        axes: Vec<(i32, f32)>,
        scale: Vec<f32>,
        rotation: Vec<f32>,
        speed: Vec<f32>,
        sliders: Vec<(i32, i32)>,
        kick: Vec<bool>,
    }
    impl ImodvControlNativeBoundary for Native {
        fn remove_dialog(&mut self) {}
        fn close_dialog(&mut self) {
            self.closes += 1;
        }
        fn model_edit_update(&mut self) {
            self.model_edit_updates += 1;
        }
        fn rotation_center_state(&mut self, _: bool) {}
        fn rotation_step_label(&mut self, _: f32) {}
        fn raise_dialog(&mut self) {
            self.raises += 1;
        }
        fn create_dialog(&mut self) -> bool {
            self.create
        }
        fn set_axis_text(&mut self, axis: i32, value: f32) {
            self.axes.push((axis, value));
        }
        fn set_scale_text(&mut self, value: f32) {
            self.scale.push(value);
        }
        fn set_rotation_rate(&mut self, value: f32) {
            self.rotation.push(value);
        }
        fn set_speed_text(&mut self, value: f32) {
            self.speed.push(value);
        }
        fn set_view_slider(&mut self, slider: i32, value: i32) {
            self.sliders.push((slider, value));
        }
        fn set_kick_box(&mut self, checked: bool) {
            self.kick.push(checked);
        }
        fn update_slicer_link(&mut self, _: i32) {}
    }

    #[test]
    fn speed_is_clamped_as_source() {
        let mut a = ImodvApp {
            movie_speed: 360.,
            ..Default::default()
        };
        imodv_control_inc_speed(
            &mut a,
            &ImodvControlDialogState::default(),
            &mut Native::default(),
            1,
        );
        assert_eq!(a.movie_speed, 360.);
    }
    #[test]
    fn clip_keeps_near_before_far() {
        let mut m = Imod::default();
        let p = &mut m as *mut _;
        let mut a = ImodvApp {
            imod: p,
            mod_: vec![p],
            num_mods: 1,
            cfar: 10,
            ..Default::default()
        };
        let mut state = ImodvControlDialogState::default();
        let mut native = Native::default();
        imodv_control_clip(
            &mut a,
            &mut state,
            &mut native,
            IMODV_CONTROL_NEAR,
            10,
            false,
        );
        assert_eq!(a.cfar, 11);
        assert_eq!(native.sliders, vec![(IMODV_CONTROL_FAR, 11)]);
    }
    #[test]
    fn opening_and_existing_dialog_follow_source_form_feedback() {
        let mut m = Imod::default();
        m.view[0].fovy = 23.;
        m.view[0].cnear = 0.012;
        m.view[0].cfar = 0.345;
        m.view[0].world = 0x20;
        m.zscale = 1.5;
        let p = &mut m as *mut _;
        let mut a = ImodvApp {
            imod: p,
            mod_: vec![p],
            num_mods: 1,
            winx: 400,
            winy: 200,
            delta_rot: 17.,
            movie_speed: 42.,
            ..Default::default()
        };
        let mut state = ImodvControlDialogState::default();
        let mut native = Native {
            create: true,
            ..Default::default()
        };
        assert_eq!(imodv_control(&mut a, &mut state, &mut native, 1), 0);
        assert!(state.dialog_open);
        assert_eq!(native.axes.len(), 3);
        assert_eq!(native.rotation, vec![17.]);
        assert_eq!(native.speed, vec![42.]);
        assert_eq!(native.sliders, vec![(1, 12), (2, 345), (3, 23), (4, 150)]);
        assert_eq!(native.kick, vec![true]);
        assert_eq!(imodv_control(&mut a, &mut state, &mut native, 1), -1);
        assert_eq!(native.raises, 1);
        assert_eq!(imodv_control(&mut a, &mut state, &mut native, 0), -1);
        assert_eq!(native.closes, 1);
    }
}
