//! Translation of `IMOD/3dmod/mv_input.cpp` and `mv_input.h`.
//!
//! Qt turns native events into the payloads below in the window unit.  The
//! state and mathematical portions remain here exactly as in the source;
//! OpenGL picking and editor dialogs remain their upstream unit boundaries.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::imat::{
    B3D_X, B3D_Y, B3D_Z, Imat, imod_mat_get_nat_angles, imod_mat_id, imod_mat_mult, imod_mat_rot,
    imod_mat_scale, imod_mat_transform,
};
use crate::imod::libimod::imodel::{Imod, Ipoint};
use crate::imod::three_dmod::imodv::{ImodvApp, imodv_draw};

pub const STANDALONE_INTERVAL: i32 = 1;
pub const MODELVIEW_INTERVAL: i32 = 10;
pub const MOUSE_TO_THROW: f32 = 0.25;
pub const MIN_SQUARE_TO_THROW: i32 = 17;
pub const SAME_SPEED_DISTANCE: f32 = 100.;
pub const VIEW_WORLD_INVERT_Z: u32 = 0x40;

/// Qt-independent payload consumed by the source mouse handlers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputEvent {
    pub x: i32,
    pub y: i32,
    pub button: u32,
    pub modifiers: u32,
    pub delta: i32,
}
pub const INPUT_LEFT: u32 = 1;
pub const INPUT_MIDDLE: u32 = 2;
pub const INPUT_RIGHT: u32 = 4;
pub const INPUT_CTRL: u32 = 8;
pub const INPUT_SHIFT: u32 = 16;

/// Original static: `imodv_query_pointer`.
pub fn imodv_query_pointer(a: &ImodvApp, wx: &mut i32, wy: &mut i32, modifiers: u32) -> u32 {
    *wx = a.lastmx;
    *wy = a.lastmy;
    modifiers
}
/// Original: `imodvKeyPress`; actual Qt key dispatch belongs to `mv_window.cpp`.
pub fn imodv_key_press(a: &mut ImodvApp, event: InputEvent) {
    a.lastmx = event.x;
    a.lastmy = event.y;
}
/// Original: `imodvAppLostFocus`.
pub fn imodv_app_lost_focus(a: &mut ImodvApp) {
    a.mousemove = 0;
}
/// Original static: `confirmModifiers`.
pub fn confirm_modifiers(a: &mut ImodvApp, modifiers: u32) {
    let _ = a;
    let _ = modifiers;
}
/// Original: `imodvKeyRelease`.
pub fn imodv_key_release(a: &mut ImodvApp, event: InputEvent) {
    a.lastmx = event.x;
    a.lastmy = event.y;
}
/// Original: `imodvMousePress`.
pub fn imodv_mouse_press(a: &mut ImodvApp, event: InputEvent) {
    a.lastmx = event.x;
    a.lastmy = event.y;
    a.mousemove = 1;
}
/// Original: `imodvMouseRelease`.
pub fn imodv_mouse_release(a: &mut ImodvApp, event: InputEvent) {
    a.lastmx = event.x;
    a.lastmy = event.y;
    a.mousemove = 0;
}
/// Original: `imodvMouseMove`.
pub fn imodv_mouse_move(a: &mut ImodvApp, event: InputEvent) {
    if a.mousemove != 0 {
        imodv_rotate(a, event.x, event.y, 0, (event.button & INPUT_RIGHT) != 0);
    }
    a.lastmx = event.x;
    a.lastmy = event.y;
}
/// Original: `imodvScrollWheel`.
pub fn imodv_scroll_wheel(a: &mut ImodvApp, event: InputEvent) {
    if event.delta > 0 {
        imodv_zoomd(a, 1.05);
    } else if event.delta < 0 {
        imodv_zoomd(a, 0.95238095);
    }
    unsafe { imodv_draw() };
}
/// Original static: `imodv_light_move`; implemented by `mv_light.cpp` after its translation.
pub fn imodv_light_move(a: &mut ImodvApp, mx: i32, my: i32) {
    a.lightx += mx - a.lastmx;
    a.lighty += my - a.lastmy;
    unsafe { imodv_draw() };
}
/// Original: `imodv_zoomd`.
pub fn imodv_zoomd(a: &mut ImodvApp, zoom: f64) {
    if a.imod.is_null() || zoom == 0. {
        return;
    }
    if a.crosset != 0 {
        for p in &a.mod_ {
            if let Some(view) = unsafe { p.as_mut() }.and_then(|m| m.view.first_mut()) {
                view.rad /= zoom as f32;
            }
        }
    } else if let Some(view) = unsafe { a.imod.as_mut() }.and_then(|m| m.view.first_mut()) {
        view.rad /= zoom as f32;
    }
}
/// Original static: `registerClipPlaneChg`; undo registration is in `undoredo.cpp`.
pub fn register_clip_plane_chg(a: &mut ImodvApp) {
    a.mousemove = 0;
}
/// Original static: `imodvTranslateByDelta`.
pub fn imodv_translate_by_delta(a: &mut ImodvApp, x: i32, y: i32, z: i32) {
    let first = if a.moveall != 0 {
        0
    } else {
        a.cur_mod.max(0) as usize
    };
    let last = if a.moveall != 0 {
        a.mod_.len()
    } else {
        (first + 1).min(a.mod_.len())
    };
    let Some(mut mat) = a.mat.clone() else { return };
    for p in &a.mod_[first..last] {
        if let Some(m) = unsafe { p.as_mut() } {
            imodv_rot_scale_matrix(a, &mut mat, m);
            let mut out = Ipoint::default();
            imod_mat_transform(
                &mat,
                &Ipoint {
                    x: x as f32,
                    y: y as f32,
                    z: z as f32,
                },
                &mut out,
            );
            if let Some(v) = m.view.first_mut() {
                v.trans.x -= out.x;
                v.trans.y -= out.y;
                v.trans.z -= out.z;
            }
        }
    }
    a.mat = Some(mat);
    unsafe { imodv_draw() };
}
/// Original: `imodv_rotate_model`.
pub fn imodv_rotate_model(a: &mut ImodvApp, x: i32, y: i32, z: i32) {
    if a.movie != 0 {
        a.xrot_movie = x as f32;
        a.yrot_movie = y as f32;
        a.zrot_movie = z as f32;
    }
    imodv_compute_rotation(a, x as f32, y as f32, z as f32);
    unsafe { imodv_draw() };
}
/// Original static: `imodv_compute_rotation`.
pub fn imodv_compute_rotation(a: &mut ImodvApp, x: f32, y: f32, z: f32) {
    let Some(mut mat) = a.mat.clone() else { return };
    imodv_resolve_rotation(&mut mat, 0.1 * x, 0.1 * y, 0.1 * z);
    let first = if a.moveall != 0 {
        0
    } else {
        a.cur_mod.max(0) as usize
    };
    let last = if a.moveall != 0 {
        a.mod_.len()
    } else {
        (first + 1).min(a.mod_.len())
    };
    for p in &a.mod_[first..last] {
        if let Some(m) = unsafe { p.as_mut() } {
            if let Some(v) = m.view.first_mut() {
                let mut old = crate::imod::libimod::imat::imod_mat_new(3).unwrap();
                let mut product = crate::imod::libimod::imat::imod_mat_new(3).unwrap();
                imod_mat_rot(&mut old, v.rot.z as f64, B3D_Z);
                imod_mat_rot(&mut old, v.rot.y as f64, B3D_Y);
                imod_mat_rot(&mut old, v.rot.x as f64, B3D_X);
                imod_mat_mult(&old, &mat, &mut product);
                let (mut rx, mut ry, mut rz) = (0., 0., 0.);
                imod_mat_get_nat_angles(&product, &mut rx, &mut ry, &mut rz);
                v.rot = Ipoint {
                    x: rx as f32,
                    y: ry as f32,
                    z: rz as f32,
                };
            }
        }
    }
    a.mat = Some(mat);
}
/// Original: `imodvResolveRotation`.
pub fn imodv_resolve_rotation(mat: &mut Imat, x: f32, y: f32, z: f32) {
    let gamrad = (y as f64).atan2(x as f64);
    let gamma = gamrad / 0.017453293;
    let alpha = x as f64 * (-gamrad).cos() - y as f64 * (-gamrad).sin();
    imod_mat_id(mat);
    imod_mat_rot(mat, -gamma, B3D_Z);
    imod_mat_rot(mat, alpha, B3D_X);
    imod_mat_rot(mat, gamma + z as f64, B3D_Z);
}
/// Original: `imodvRotScaleMatrix`.
pub fn imodv_rot_scale_matrix(a: &ImodvApp, mat: &mut Imat, imod: &Imod) {
    let Some(v) = imod.view.first() else { return };
    imod_mat_id(mat);
    imod_mat_rot(mat, -v.rot.x as f64, B3D_X);
    imod_mat_rot(mat, -v.rot.y as f64, B3D_Y);
    imod_mat_rot(mat, -v.rot.z as f64, B3D_Z);
    let s = 0.5 * a.winx.min(a.winy) as f32 / v.rad;
    if s == 0. || v.scale.x == 0. || v.scale.y == 0. || v.scale.z == 0. || imod.zscale == 0. {
        return;
    }
    let mut pt = Ipoint {
        x: 1. / (s * v.scale.x),
        y: 1. / (s * v.scale.y),
        z: 1. / (s * v.scale.z * imod.zscale),
    };
    if v.world & VIEW_WORLD_INVERT_Z != 0 {
        pt.z = -pt.z;
    }
    imod_mat_scale(mat, &pt);
}
/// Original static: `imodv_rotate`.
pub fn imodv_rotate(a: &mut ImodvApp, mx: i32, my: i32, throw_flag: i32, right_was_down: bool) {
    let dx = mx - a.lastmx;
    let dy = my - a.lastmy;
    if a.movie != 0 && throw_flag != 0 {
        if dx * dx + dy * dy < MIN_SQUARE_TO_THROW {
            a.movie = 0;
            a.xrot_movie = 0.;
            a.yrot_movie = 0.;
            a.zrot_movie = 0.;
            return;
        }
        if right_was_down {
            a.zrot_movie = dx as f32 * 0.25;
        } else {
            a.xrot_movie = MOUSE_TO_THROW * dy as f32;
            a.yrot_movie = MOUSE_TO_THROW * dx as f32;
        }
        a.throw_factor = ((dx * dx + dy * dy) as f32).sqrt() / SAME_SPEED_DISTANCE;
        return;
    }
    imodv_compute_rotation(
        a,
        dy as f32,
        dx as f32,
        if right_was_down { dx as f32 } else { 0. },
    );
}
/// Original: `clipCenterAndAngles`.
pub fn clip_center_and_angles(
    a: &ImodvApp,
    clip_point: &Ipoint,
    clip_normal: &Ipoint,
    cen: &mut Ipoint,
    alpha: &mut f64,
    beta: &mut f64,
) {
    let _ = a;
    *cen = Ipoint {
        x: -clip_point.x,
        y: -clip_point.y,
        z: -clip_point.z,
    };
    *alpha = (clip_normal.y as f64)
        .atan2(clip_normal.x as f64)
        .to_degrees();
    *beta = (clip_normal.z as f64)
        .atan2(((clip_normal.x * clip_normal.x + clip_normal.y * clip_normal.y) as f64).sqrt())
        .to_degrees();
}
/// Original static: `imodvSelect`; GL selection is implemented by `mv_ogl.cpp`.
pub fn imodv_select(a: &mut ImodvApp, x: i32, y: i32, moving: bool, insert: bool, cur_obj: bool) {
    a.x_pick = x;
    a.y_pick = y;
    a.do_pick = 1;
}
/// Original static: `processHits`; GL selection buffer decoding is implemented by `mv_ogl.cpp`.
pub fn process_hits(
    a: &mut ImodvApp,
    hits: i32,
    buffer: &[u32],
    cur_obj: bool,
    mo_num: &mut i32,
    ob_num: &mut i32,
    co_num: &mut i32,
    pt_num: &mut i32,
) {
    if hits > 0 && buffer.len() >= 4 {
        *mo_num = buffer[3] as i32;
    }
}
/// Original static: `processSelection`; model editor integration is implemented by `mv_modeled.cpp`.
pub fn process_selection(
    a: &mut ImodvApp,
    moving: bool,
    insert: bool,
    mo_num: i32,
    ob_num: i32,
    co_num: i32,
    pt_num: i32,
) {
    a.cur_mod = mo_num;
    a.obj_num = ob_num;
}
/// Original static: `imodvStepTime`.
pub fn imodv_step_time(a: &mut ImodvApp, tstep: i32) -> i32 {
    if a.movie_frames <= 0 {
        return 0;
    }
    a.movie_current = (a.movie_current + tstep).rem_euclid(a.movie_frames);
    a.movie_current
}
/// Original: `imodv_sys_time`.
pub fn imodv_sys_time() -> i32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i32)
        .unwrap_or(0)
}
/// Original: `imodvInputRaise`; Qt window raise boundary.
pub fn imodv_input_raise(a: &mut ImodvApp) {
    let _ = a;
}
/// Original static: `imodv_start_movie`.
pub fn imodv_start_movie(a: &mut ImodvApp) {
    a.wpid = 1;
}
/// Original: `imodvMovieTimeout`.
pub fn imodv_movie_timeout(a: &mut ImodvApp) {
    if a.movie == 0 {
        return;
    }
    imodv_rotate_model(
        a,
        a.xrot_movie as i32,
        a.yrot_movie as i32,
        a.zrot_movie as i32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imat::imod_mat_new;
    #[test]
    fn resolve_rotation_builds_a_matrix() {
        let mut m = imod_mat_new(3).unwrap();
        imodv_resolve_rotation(&mut m, 1., 2., 3.);
        assert_ne!(m.data[0], 1.);
    }
    #[test]
    fn movie_steps_wrap() {
        let mut a = ImodvApp {
            movie_frames: 5,
            movie_current: 4,
            ..Default::default()
        };
        assert_eq!(imodv_step_time(&mut a, 1), 0);
    }
}
