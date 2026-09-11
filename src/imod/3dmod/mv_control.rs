//! Translation of `IMOD/3dmod/mv_control.cpp` and `mv_control.h`.
//!
//! The Qt form itself remains owned by the `formv_control.cpp` translation
//! closure.  This unit deliberately retains the source's state mutations;
//! callers pass the `ImodvApp` selected by the window/event boundary.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::imodel::Imod;
use crate::imod::three_dmod::imodv::{
    ImodvApp, imodv_draw, imodv_draw_imod_images, imodv_new_model_angles,
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

/// Original: `imodvControlZoom`.
pub fn imodv_control_zoom(a: &mut ImodvApp, zoom: i32) {
    imodv_zoomd(a, if zoom > 0 { 1.05 } else { 0.95238095 });
    unsafe { imodv_draw() };
}

/// Original: `imodvControlKickClips`.
pub fn imodv_control_kick_clips(a: &mut ImodvApp, state: bool) {
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
    for model in &a.mod_[first..last] {
        if let Some(view) = unsafe { model.as_mut() }.and_then(|m| m.view.first_mut()) {
            if state {
                view.world |= WORLD_KICKOUT_CLIPS;
            } else {
                view.world &= !WORLD_KICKOUT_CLIPS;
            }
        }
    }
    unsafe { imodv_draw() };
}

/// Original: `imodvControlClip`.
pub fn imodv_control_clip(a: &mut ImodvApp, plane: i32, value: i32, dragging: bool) {
    let Some(imod) = (unsafe { a.imod.as_mut() }) else {
        return;
    };
    let Some(view) = imod.view.first_mut() else {
        return;
    };
    match plane {
        IMODV_CONTROL_FAR => {
            a.cfar = value;
            if a.cnear >= a.cfar {
                a.cnear = a.cfar - 1;
            }
        }
        IMODV_CONTROL_NEAR => {
            a.cnear = value;
            if a.cfar <= a.cnear {
                a.cfar = a.cnear + 1;
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
pub fn imodv_control_zscale(a: &mut ImodvApp, value: i32, dragging: bool) {
    if let Some(model) = a
        .mod_
        .get(a.cur_mod.max(0) as usize)
        .and_then(|m| unsafe { m.as_mut() })
    {
        model.zscale = value as f32 / 100.;
    }
    unsafe { imodv_draw() };
    imodv_draw_imod_images(0);
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
pub fn imodv_control_start(a: &mut ImodvApp) {
    a.movie = if a.movie != 0 { 0 } else { 1 };
    a.xrot_movie = 0.;
    a.yrot_movie = 0.;
    a.zrot_movie = 0.;
    unsafe { imodv_draw() };
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
pub fn imodv_control_axis_text(a: &mut ImodvApp, axis: i32, rot: f32) {
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
    if let Some(imod) = unsafe { a.imod.as_ref() }.and_then(|m| m.view.first()) {
        imodv_new_model_angles(&imod.rot);
    }
    unsafe { imodv_draw() };
}

/// Original: `imodvControlRate`.
pub fn imodv_control_rate(a: &mut ImodvApp, value: i32) {
    a.delta_rot = value as f32;
}
/// Original: `imodvControlSpeed`.
pub fn imodv_control_speed(a: &mut ImodvApp, value: f32) {
    a.movie_speed = value;
}
/// Original: `imodvControlIncSpeed`.
pub fn imodv_control_inc_speed(a: &mut ImodvApp, step: i32) {
    a.movie_speed = if step > 0 {
        (a.movie_speed * IMODV_ROTATION_FACTOR).min(360.)
    } else {
        (a.movie_speed / IMODV_ROTATION_FACTOR).max(3.6)
    };
}
/// Original: `imodvControlClosing`; Qt ownership boundary.
pub fn imodv_control_closing() {}
/// Original: `imodvControlQuit`; Qt ownership boundary.
pub fn imodv_control_quit() {}
/// Original: `imodvControlChangeSteps`.
pub fn imodv_control_change_steps(a: &mut ImodvApp, delta: i32) {
    let mut n = if delta > 0 {
        (a.delta_rot * IMODV_ROTATION_FACTOR).round()
    } else {
        (a.delta_rot / IMODV_ROTATION_FACTOR).round()
    };
    if n == a.delta_rot {
        n += delta as f32;
    }
    imodv_control_set_arot(a, n.max(1.) as i32);
    imodv_control_inc_speed(a, delta);
}
/// Original: `imodvControlSetArot`.
pub fn imodv_control_set_arot(a: &mut ImodvApp, newval: i32) {
    imodv_control_rate(a, newval.min(ROTATION_MAX));
}
/// Original: `imodvControlSetView`.
pub fn imodv_control_set_view(a: &mut ImodvApp) {
    if let Some(v) = unsafe { a.imod.as_ref() }.and_then(|m| m.view.first()) {
        a.fovy = v.fovy as i32;
        a.cnear = (v.cnear * 1000. + 0.5) as i32;
        a.cfar = (v.cfar * 1000. + 0.5) as i32;
    }
}
/// Original: `imodvControlUpdate`; form update is owned by `formv_control.cpp`.
pub fn imodv_control_update(a: &mut ImodvApp) {}
/// Original: `imodvControlLinkUpdate`; form update is owned by `formv_control.cpp`.
pub fn imodv_control_link_update(a: &mut ImodvApp) {}
/// Original: `imodv_control`; creates/closes the Qt form in `formv_control.cpp`.
pub fn imodv_control(a: &mut ImodvApp, state: i32) -> i32 {
    if state == 0 {
        -1
    } else {
        imodv_control_set_view(a);
        imodv_control_set_arot(a, a.delta_rot as i32);
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::Imod;
    #[test]
    fn speed_is_clamped_as_source() {
        let mut a = ImodvApp {
            movie_speed: 360.,
            ..Default::default()
        };
        imodv_control_inc_speed(&mut a, 1);
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
        imodv_control_clip(&mut a, IMODV_CONTROL_NEAR, 10, false);
        assert_eq!(a.cfar, 11);
    }
}
