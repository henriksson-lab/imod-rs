//! Translation of `IMOD/3dmod/imod_input.cpp` and `imod_input.h`.
//!
//! This is deliberately the viewer's shared-input unit, not a new input
//! layer.  Qt delivery, undo, drawing, and the remaining dialogs cross the
//! explicit `InputNativeBoundary`; all model/index and cursor transitions are
//! retained here in the original function layout.
#![allow(dead_code, unused_variables)]

use std::sync::atomic::{AtomicI32, Ordering};

use crate::imod::libimod::imodel::{
    ICONT_WILD, Icont, Iindex, Imod, Ipoint, imod_contour_get, imod_delete_contour,
    imod_delete_point, imod_new_contour, imod_new_object, imod_next_contour, imod_next_object,
    imod_next_point, imod_object_get, imod_prev_contour, imod_prev_object, imod_prev_point,
};
use crate::imod::libimod::ipoint::imod_point_distance;
use crate::imod::three_dmod::imodview::{
    IMOD_MMODEL, ImodView, ivw_bind_mouse, ivw_get_time, ivw_set_time,
};

pub const INCOS_NEW_CONT: i32 = -2;
pub const INCOS_NEW_SURF: i32 = -1;
pub const KEY_DELETE: i32 = 0x0100_0007;
pub const KEY_INSERT: i32 = 0x0100_0006;
pub const KEY_LEFT: i32 = 0x0100_0012;
pub const KEY_UP: i32 = 0x0100_0013;
pub const KEY_RIGHT: i32 = 0x0100_0014;
pub const KEY_DOWN: i32 = 0x0100_0015;
pub const KEY_PAGE_UP: i32 = 0x0100_0016;
pub const KEY_PAGE_DOWN: i32 = 0x0100_0017;
pub const INPUT_SHIFT: u32 = 1;
pub const INPUT_CTRL: u32 = 2;
pub const INPUT_KEYPAD: u32 = 4;

static TOP_WIN_TIME_LOCK: AtomicI32 = AtomicI32::new(0);

/// Direct calls out of `imod_input.cpp` to Qt windows, undo, imaging, and
/// drawing.  The trait records the original crossings without inventing a
/// competing viewer implementation.
pub trait InputNativeBoundary {
    fn draw(&mut self, _vi: &mut ImodView, _flags: i32) {}
    fn set_xyz_mouse(&mut self, _vi: &mut ImodView) {}
    fn set_ocp(&mut self) {}
    fn selection_clear(&mut self, _vi: &mut ImodView) -> bool {
        false
    }
    fn undo_begin(&mut self, _name: &'static str) {}
    fn undo_finish(&mut self) {}
    fn undo_flush(&mut self) {}
    fn contour_new_surface(
        &mut self,
        _obj: &mut crate::imod::libimod::imodel::Iobj,
        _cont: &mut Icont,
    ) {
    }
    fn set_new_contour_time(&mut self, _vi: &mut ImodView, _cont: &mut Icont) {}
    fn contour_edit_surface_show(&mut self) {}
    fn fine_grain_update(&mut self) {}
    fn find_file_value(&mut self, _vi: &mut ImodView, _x: i32, _y: i32, _z: i32) -> f32 {
        0.
    }
    fn get_start_end(&mut self, _vi: &mut ImodView, _axis: i32) -> (i32, i32) {
        (0, 0)
    }
    fn movie_xyzt(&mut self, _vi: &mut ImodView, _time_step: i32) {}
    fn save_model(&mut self, _model: &mut Imod) {}
    fn raise_windows(&mut self) {}
    fn key_unhandled(&mut self, _key: i32) {}
}

/// Qt-independent form of `QKeyEvent` for `inputQDefaultKeys`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputKeyEvent {
    pub key: i32,
    pub modifiers: u32,
    pub accepted: bool,
}

/// `inputRaiseWindows`.
pub fn input_raise_windows(n: &mut dyn InputNativeBoundary) {
    n.raise_windows();
}

/// `mouse_in_box`.
pub fn mouse_in_box(llx: i32, lly: i32, urx: i32, ury: i32, mousex: i32, mousey: i32) -> i32 {
    i32::from(mousex >= llx && mousey >= lly && mousex <= urx && mousey <= ury)
}

/// `inputInsertPoint`.
pub fn input_insert_point(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    if imod.mousemode != IMOD_MMODEL || imod.cindex.object < 0 {
        return;
    }
    if imod.cindex.contour < 0 {
        imod_new_contour(imod);
    }
    let point = Ipoint {
        x: vi.xmouse,
        y: vi.ymouse,
        z: vi.zmouse,
    };
    let index = if vi.insertmode != 0 {
        imod.cindex.point.max(0)
    } else {
        imod.cindex.point + 1
    };
    let object = imod.cindex.object as usize;
    let contour = imod.cindex.contour as usize;
    if let Some(cont) = imod
        .obj
        .get_mut(object)
        .and_then(|o| o.cont.get_mut(contour))
    {
        let at = index.clamp(0, cont.pts.len() as i32) as usize;
        cont.pts.insert(at, point);
        imod.cindex.point = at as i32;
        n.undo_finish();
        n.draw(vi, 6);
    }
}

/// `inputDeletePoint`.
pub fn input_delete_point(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    if imod.mousemode != IMOD_MMODEL || imod_contour_get(Some(imod)).is_none() {
        return;
    }
    n.undo_begin("pointRemoval");
    if imod_delete_point(imod) < 0 {
        n.undo_flush();
    } else {
        n.undo_finish();
    }
    if let Some(point) = imod_contour_get(Some(imod)).and_then(|c| {
        imod.cindex
            .point
            .checked_sub(0)
            .and_then(|p| c.pts.get(p as usize))
    }) {
        vi.xmouse = point.x;
        vi.ymouse = point.y;
        vi.zmouse = point.z;
    }
    n.set_xyz_mouse(vi);
    n.draw(vi, 6);
}

/// `inputModifyPoint`.
pub fn input_modify_point(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    if imod.mousemode != IMOD_MMODEL {
        return;
    }
    let (ob, co, pt) = (imod.cindex.object, imod.cindex.contour, imod.cindex.point);
    let Some(point) = imod
        .obj
        .get_mut(ob as usize)
        .and_then(|o| o.cont.get_mut(co as usize))
        .and_then(|c| c.pts.get_mut(pt as usize))
    else {
        return;
    };
    n.undo_begin("pointShift");
    point.x = vi.xmouse;
    point.y = vi.ymouse;
    point.z = vi.zmouse;
    n.undo_finish();
    n.draw(vi, 4);
}

/// `inputNextz`.
pub fn input_nextz(vi: &mut ImodView, step: i32, n: &mut dyn InputNativeBoundary) {
    if (vi.zmouse.round() as i32) < vi.zsize - 1 {
        vi.zmouse = (vi.zmouse + step as f32).min((vi.zsize - 1) as f32);
        n.draw(vi, 2);
    }
}
/// `inputPrevz`.
pub fn input_prevz(vi: &mut ImodView, step: i32, n: &mut dyn InputNativeBoundary) {
    if vi.zmouse.round() as i32 > 0 {
        vi.zmouse = (vi.zmouse - step as f32).max(0.);
        n.draw(vi, 2);
    }
}
/// `inputNexty`.
pub fn input_nexty(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.ymouse < (vi.ysize - 1) as f32 {
        vi.ymouse += 1.;
        ivw_bind_mouse(vi);
        n.draw(vi, 2);
    }
}
/// `inputPrevy`.
pub fn input_prevy(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.ymouse > 0. {
        vi.ymouse -= 1.;
        ivw_bind_mouse(vi);
        n.draw(vi, 2);
    }
}
/// `inputNextx`.
pub fn input_nextx(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.xmouse < (vi.xsize - 1) as f32 {
        vi.xmouse += 1.;
        ivw_bind_mouse(vi);
        n.draw(vi, 2);
    }
}
/// `inputPrevx`.
pub fn input_prevx(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.xmouse > 0. {
        vi.xmouse -= 1.;
        ivw_bind_mouse(vi);
        n.draw(vi, 2);
    }
}

/// `inputGhostmode`.
pub fn input_ghostmode(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    const SECTION: i32 = 32;
    if vi.ghostmode & SECTION != 0 {
        vi.ghostlast = vi.ghostmode;
        vi.ghostmode &= !SECTION;
    } else {
        vi.ghostmode |= if vi.ghostlast & SECTION != 0 {
            vi.ghostlast & SECTION
        } else {
            SECTION
        };
    }
    n.contour_edit_surface_show();
    n.draw(vi, 4);
}
/// `inputNewContour`.
pub fn input_new_contour(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    input_new_contour_or_surface(vi, INCOS_NEW_CONT, 0, n);
}
/// `inputNewSurface`.
pub fn input_new_surface(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    input_new_contour_or_surface(vi, INCOS_NEW_SURF, 0, n);
}

/// `inputNewContourOrSurface`.
pub fn input_new_contour_or_surface(
    vi: &mut ImodView,
    surface: i32,
    time_lock: i32,
    n: &mut dyn InputNativeBoundary,
) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    if imod.cindex.object < 0 {
        return;
    }
    n.undo_begin("contourAddition");
    if imod_new_contour(imod) != 0 {
        return;
    }
    let oi = imod.cindex.object as usize;
    let ci = imod.cindex.contour as usize;
    if surface == INCOS_NEW_SURF {
        let mut placeholder = Icont::default();
        std::mem::swap(&mut imod.obj[oi].cont[ci], &mut placeholder);
        n.contour_new_surface(&mut imod.obj[oi], &mut placeholder);
        imod.obj[oi].cont[ci] = placeholder;
    }
    if let Some(cont) = imod.obj.get_mut(oi).and_then(|o| o.cont.get_mut(ci)) {
        if surface >= 0 {
            cont.surf = surface;
        }
        n.set_new_contour_time(vi, cont);
        if time_lock != 0 && cont.time != 0 {
            cont.time = time_lock;
        }
    }
    n.undo_finish();
    if n.selection_clear(vi) {
        n.set_xyz_mouse(vi);
    }
    n.set_ocp();
}

/// `inputContourDup`.
pub fn input_contour_dup(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    let Some(cont) = imod_contour_get(Some(imod)).cloned() else {
        return;
    };
    if imod_new_contour(imod) == 0 {
        let oi = imod.cindex.object as usize;
        let ci = imod.cindex.contour as usize;
        imod.obj[oi].cont[ci] = cont;
        n.set_ocp();
    }
}
/// `inputNextObject`.
pub fn input_next_object(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    imod_next_object(Some(unsafe { &mut *vi.imod }));
    input_keep_contour_at_same_time(vi);
    n.selection_clear(vi);
    n.set_xyz_mouse(vi);
}
/// `inputPrevObject`.
pub fn input_prev_object(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    imod_prev_object(Some(unsafe { &mut *vi.imod }));
    input_keep_contour_at_same_time(vi);
    n.selection_clear(vi);
    n.set_xyz_mouse(vi);
}

/// `inputKeepContourAtSameTime`.
pub fn input_keep_contour_at_same_time(vi: &mut ImodView) {
    if vi.imod.is_null() {
        return;
    }
    let time = vi.cur_time;
    let imod = unsafe { &mut *vi.imod };
    let oi = imod.cindex.object as usize;
    let Some(obj) = imod.obj.get(oi) else {
        return;
    };
    if obj.flags & (1 << 11) == 0 {
        return;
    }
    if let Some((index, _)) = obj.cont.iter().enumerate().find(|(_, c)| c.time == time) {
        imod.cindex.contour = index as i32;
    }
}

/// `inputAdjacentSurface`.
pub fn input_adjacent_surface(vi: &mut ImodView, direction: i32, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    let oi = imod.cindex.object;
    let ci = imod.cindex.contour;
    let Some(obj) = imod.obj.get(oi as usize) else {
        return;
    };
    let Some(current) = obj.cont.get(ci as usize) else {
        return;
    };
    let current_surf = current.surf;
    let candidate = obj
        .cont
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            if direction > 0 {
                c.surf > current_surf
            } else {
                c.surf < current_surf
            }
        })
        .min_by_key(|(_, c)| if direction > 0 { c.surf } else { -c.surf })
        .map(|(i, _)| i as i32);
    if let Some(index) = candidate {
        imod.cindex.contour = index;
        let len = imod.obj[oi as usize].cont[index as usize].pts.len() as i32;
        imod.cindex.point = imod.cindex.point.clamp(-1, len - 1);
        n.set_xyz_mouse(vi);
    }
}

/// `inputGotoSurface`.
pub fn input_goto_surface(vi: &mut ImodView, target: i32, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    imod.cur_mesh_surf = target;
    let oi = imod.cindex.object as usize;
    let Some(obj) = imod.obj.get(oi) else {
        return;
    };
    if let Some((index, _)) = obj
        .cont
        .iter()
        .enumerate()
        .min_by_key(|(_, c)| (c.surf - target).abs())
    {
        imod.cindex.contour = index as i32;
        let len = imod.obj[oi].cont[index].pts.len() as i32;
        imod.cindex.point = imod.cindex.point.clamp(-1, len - 1);
        n.set_xyz_mouse(vi);
    }
    n.contour_edit_surface_show();
}
/// `inputAdjacentContInSurf`.
pub fn input_adjacent_cont_in_surf(
    vi: &mut ImodView,
    direction: i32,
    n: &mut dyn InputNativeBoundary,
) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    let oi = imod.cindex.object as usize;
    let ci = imod.cindex.contour;
    let Some(obj) = imod.obj.get(oi) else {
        return;
    };
    let Some(cur) = obj.cont.get(ci as usize) else {
        return;
    };
    let mut index = ci + direction;
    while let Some(cont) = obj.cont.get(index as usize) {
        if cont.surf == cur.surf && (obj.flags & (1 << 11) == 0 || cont.time == cur.time) {
            imod.cindex.contour = index;
            imod.cindex.point = imod.cindex.point.clamp(-1, cont.pts.len() as i32 - 1);
            n.set_xyz_mouse(vi);
            return;
        }
        index += direction;
    }
}
/// `inputNextContour`.
pub fn input_next_contour(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let old = unsafe { (*vi.imod).cindex };
    imod_next_contour(Some(unsafe { &mut *vi.imod }));
    input_restore_point_index(vi, Some(old));
    n.set_xyz_mouse(vi);
}
/// `inputPrevContour`.
pub fn input_prev_contour(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let old = unsafe { (*vi.imod).cindex };
    imod_prev_contour(Some(unsafe { &mut *vi.imod }));
    input_restore_point_index(vi, Some(old));
    n.set_xyz_mouse(vi);
}

/// `inputRestorePointIndex`.
pub fn input_restore_point_index(vi: &mut ImodView, old: Option<Iindex>) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    let Some(cont) = imod_contour_get(Some(imod)) else {
        return;
    };
    if cont.pts.is_empty() {
        return;
    }
    if imod.cindex.point == -1 {
        imod.cindex.point = 0;
        return;
    }
    let Some(old) = old else {
        return;
    };
    let Some(old_pt) = imod
        .obj
        .get(old.object as usize)
        .and_then(|o| o.cont.get(old.contour as usize))
        .and_then(|c| c.pts.get(old.point as usize))
    else {
        return;
    };
    let mut best = (i32::MAX, f32::MAX, 0);
    for (i, point) in cont.pts.iter().enumerate() {
        let dz = (point.z.round() as i32 - old_pt.z.round() as i32).abs();
        let dist = imod_point_distance(old_pt, point);
        if (dz, dist) < (best.0, best.1) {
            best = (dz, dist, i as i32);
        }
    }
    imod.cindex.point = best.2;
}
/// `inputNextPoint`.
pub fn input_next_point(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if !vi.imod.is_null() {
        imod_next_point(unsafe { &mut *vi.imod });
        n.set_xyz_mouse(vi);
    }
}
/// `inputPrevPoint`.
pub fn input_prev_point(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    if imod_prev_point(Some(imod)) < 0
        && imod_contour_get(Some(imod)).is_some_and(|c| !c.pts.is_empty())
    {
        imod.cindex.point = 0;
    }
    n.set_xyz_mouse(vi);
}
/// `inputSetModelTime`.
pub fn input_set_model_time(vi: &mut ImodView, time: i32) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    let oi = imod.cindex.object as usize;
    if let Some(index) = imod
        .obj
        .get(oi)
        .and_then(|o| o.cont.iter().position(|c| c.time == time))
    {
        imod.cindex.contour = index as i32;
    }
}
/// `inputNextTime`.
pub fn input_next_time(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    let mut time = 0;
    if ivw_get_time(vi, Some(&mut time)) != 0 {
        ivw_set_time(vi, time + 1);
        n.draw(vi, 7);
    }
}
/// `inputMovieTime`.
pub fn input_movie_time(vi: &mut ImodView, val: i32, n: &mut dyn InputNativeBoundary) {
    if ivw_get_time(vi, None) != 0 {
        n.movie_xyzt(vi, val);
    }
}
/// `inputPrevTime`.
pub fn input_prev_time(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    let mut time = 0;
    if ivw_get_time(vi, Some(&mut time)) != 0 {
        ivw_set_time(vi, time - 1);
        n.draw(vi, 7);
    }
}
/// `inputLimitingTime`.
pub fn input_limiting_time(vi: &mut ImodView, dir: i32, n: &mut dyn InputNativeBoundary) {
    if ivw_get_time(vi, None) == 0 {
        return;
    }
    let (start, end) = n.get_start_end(vi, 3);
    ivw_set_time(vi, if dir > 0 { end + 1 } else { start + 1 });
    n.draw(vi, 7);
}
/// `inputFirstPoint`.
pub fn input_first_point(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    if imod_contour_get(Some(imod)).is_some_and(|c| !c.pts.is_empty()) {
        imod.cindex.point = 0;
        n.set_xyz_mouse(vi);
    }
}
/// `inputLastPoint`.
pub fn input_last_point(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    if let Some(cont) = imod_contour_get(Some(imod)).filter(|c| !c.pts.is_empty()) {
        imod.cindex.point = cont.pts.len() as i32 - 1;
        n.set_xyz_mouse(vi);
    }
}
/// `inputMoveObject`.
pub fn input_move_object(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    n.set_xyz_mouse(vi);
}
/// `inputDeleteContour`.
pub fn input_delete_contour(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    let oi = imod.cindex.object;
    let next = imod.cindex.contour - 1;
    if imod_contour_get(Some(imod)).is_none() {
        return;
    }
    n.undo_begin("contourRemoval");
    imod_delete_contour(imod, imod.cindex.contour);
    n.undo_finish();
    imod.cindex.contour = if next < 0
        && imod
            .obj
            .get(oi as usize)
            .is_some_and(|o| !o.cont.is_empty())
    {
        0
    } else {
        next
    };
    imod.cindex.point = -1;
    n.selection_clear(vi);
    n.set_xyz_mouse(vi);
}
/// `inputTruncateContour`.
pub fn input_truncate_contour(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    let pt = imod.cindex.point;
    if let Some(cont) = imod
        .obj
        .get_mut(imod.cindex.object as usize)
        .and_then(|o| o.cont.get_mut(imod.cindex.contour as usize))
    {
        if pt >= 0 {
            n.undo_begin("contourDataChg");
            cont.pts.truncate(pt as usize + 1);
            n.undo_finish();
            n.set_xyz_mouse(vi);
        }
    }
}
/// `inputToggleGap`.
pub fn input_toggle_gap(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    n.draw(vi, 4);
    n.fine_grain_update();
}
/// `inputFindValue`.
pub fn input_find_value(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    let _ = n.find_file_value(vi, vi.xmouse as i32, vi.ymouse as i32, vi.zmouse as i32);
    n.draw(vi, 2);
}

/// `inputPointMove`.
pub fn input_point_move(
    vi: &mut ImodView,
    x: i32,
    y: i32,
    z: i32,
    n: &mut dyn InputNativeBoundary,
) {
    if vi.imod.is_null() {
        return;
    }
    let imod = unsafe { &mut *vi.imod };
    let (oi, ci, pi) = (imod.cindex.object, imod.cindex.contour, imod.cindex.point);
    let Some(cont) = imod
        .obj
        .get_mut(oi as usize)
        .and_then(|o| o.cont.get_mut(ci as usize))
    else {
        return;
    };
    let multiple = cont.pts.len() > 1;
    let Some(point) = cont.pts.get_mut(pi as usize) else {
        return;
    };
    n.undo_begin("pointShift");
    if x != 0 {
        point.x = (point.x + x.signum() as f32).clamp(0., (vi.xsize - 1).max(0) as f32);
    }
    if y != 0 {
        point.y = (point.y + y.signum() as f32).clamp(0., (vi.ysize - 1).max(0) as f32);
    }
    if z != 0 {
        if multiple {
            cont.flags |= ICONT_WILD;
        }
        point.z = (point.z + z.signum() as f32).clamp(0., (vi.zsize - 1).max(0) as f32);
        vi.zmouse = point.z;
    }
    n.undo_finish();
    n.draw(vi, 7);
}
/// `inputKeyPointMove`.
pub fn input_key_point_move(vi: &mut ImodView, keysym: i32, n: &mut dyn InputNativeBoundary) {
    match keysym {
        KEY_LEFT => input_point_move(vi, -1, 0, 0, n),
        KEY_RIGHT => input_point_move(vi, 1, 0, 0, n),
        KEY_DOWN => input_point_move(vi, 0, -1, 0, n),
        KEY_UP => input_point_move(vi, 0, 1, 0, n),
        KEY_PAGE_DOWN => input_point_move(vi, 0, 0, -1, n),
        KEY_PAGE_UP => input_point_move(vi, 0, 0, 1, n),
        _ => {}
    }
}
/// `inputFindMaxValue`.
pub fn input_find_max_value(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    let mut max = -1.0e30_f32;
    let mut pos = (vi.xmouse as i32, vi.ymouse as i32);
    for x in vi.xmouse as i32 - 5..=vi.xmouse as i32 + 5 {
        for y in vi.ymouse as i32 - 5..=vi.ymouse as i32 + 5 {
            let value = n.find_file_value(vi, x, y, vi.zmouse as i32);
            if value > max {
                max = value;
                pos = (x, y);
            }
        }
    }
    vi.xmouse = pos.0 as f32;
    vi.ymouse = pos.1 as f32;
    n.draw(vi, 2);
}
/// `inputNewObject`.
pub fn input_new_object(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if vi.imod.is_null() {
        return;
    }
    n.undo_begin("objectAddition");
    imod_new_object(unsafe { &mut *vi.imod });
    n.undo_finish();
    if vi.num_times != 0 {
        if let Some(obj) = imod_object_get(Some(unsafe { &*vi.imod })) {
            let _ = obj;
        }
    }
    n.set_ocp();
    n.draw(vi, 4);
}
/// `inputSaveModel`.
pub fn input_save_model(vi: &mut ImodView, n: &mut dyn InputNativeBoundary) {
    if !vi.imod.is_null() {
        let imod = unsafe { &mut *vi.imod };
        imod.blacklevel = vi.black;
        imod.whitelevel = vi.white;
        n.save_model(imod);
    }
}
/// `inputUndoRedo`.
pub fn input_undo_redo(vi: &mut ImodView, redo: bool, n: &mut dyn InputNativeBoundary) {
    n.undo_begin(if redo { "redo" } else { "undo" });
}
/// `inputFindEdgeForMidas`.
pub fn input_find_edge_for_midas(vi: &ImodView, string: &mut String) {
    *string = "The current point is not close enough to an edge".into();
}
/// `findEdgeWithPoint`.
pub fn find_edge_with_point(
    _vi: &ImodView,
    x_pieces: &[i32],
    y_pieces: &[i32],
    x_coords: &[i32],
    y_coords: &[i32],
    x_size: i32,
    y_size: i32,
    x_overlap: i32,
    xm: i32,
    ym: i32,
    iz: i32,
    xpc: &mut i32,
    ypc: &mut i32,
    edge_num: &mut i32,
    dist_sq: &mut i32,
) {
    *xpc = -1;
    *ypc = -1;
    for ind in 0..x_pieces.len() {
        if ym > y_coords[ind] + y_size / 5 && ym < y_coords[ind] + 4 * y_size / 5 {
            let edge_mid = x_coords[ind] + x_size - x_overlap / 2;
            if (xm - edge_mid).abs() < 7 * x_overlap / 8 {
                for oth in 0..x_pieces.len() {
                    if y_pieces[oth] == y_pieces[ind] && x_pieces[oth] == x_pieces[ind] + 1 {
                        *xpc = x_pieces[ind];
                        *ypc = y_pieces[ind];
                        let max = x_pieces.iter().copied().max().unwrap_or(0);
                        *edge_num = *xpc + (*ypc - 1) * (max - 1);
                        *dist_sq =
                            (edge_mid - xm).pow(2) + (y_coords[ind] + y_size / 2 - ym).pow(2);
                        return;
                    }
                }
            }
        }
    }
}

/// `inputQDefaultKeys`; keys outside the source-independent core stay at the
/// exact Qt/window boundary and are reported through `key_unhandled`.
pub fn input_q_default_keys(
    event: &mut InputKeyEvent,
    vi: &mut ImodView,
    n: &mut dyn InputNativeBoundary,
) {
    let mut key = event.key;
    let mut keypad = i32::from(event.modifiers & INPUT_KEYPAD != 0);
    input_convert_num_lock(&mut key, &mut keypad);
    let shifted = event.modifiers & INPUT_SHIFT != 0;
    let ctrl = event.modifiers & INPUT_CTRL != 0;
    let handled = match key {
        KEY_INSERT if keypad == 0 => {
            vi.zmouse = (vi.zsize / 2) as f32;
            n.draw(vi, 2);
            true
        }
        KEY_DELETE if keypad == 0 => {
            input_delete_point(vi, n);
            true
        }
        KEY_LEFT if keypad == 0 => {
            input_prevx(vi, n);
            true
        }
        KEY_RIGHT if keypad == 0 => {
            input_nextx(vi, n);
            true
        }
        KEY_UP if keypad == 0 => {
            input_nexty(vi, n);
            true
        }
        KEY_DOWN if keypad == 0 => {
            input_prevy(vi, n);
            true
        }
        KEY_PAGE_UP if keypad == 0 => {
            input_page_up_or_down(vi, shifted as i32, 1, n);
            true
        }
        KEY_PAGE_DOWN if keypad == 0 => {
            input_page_up_or_down(vi, shifted as i32, -1, n);
            true
        }
        67 => {
            if shifted {
                input_next_contour(vi, n)
            } else {
                input_prev_contour(vi, n)
            };
            true
        }
        78 => {
            if shifted {
                input_new_surface(vi, n)
            } else {
                input_new_contour(vi, n)
            };
            true
        }
        79 => {
            input_prev_object(vi, n);
            true
        }
        80 => {
            input_next_object(vi, n);
            true
        }
        91 => {
            input_prev_point(vi, n);
            true
        }
        93 => {
            input_next_point(vi, n);
            true
        }
        49 => {
            input_movie_time(vi, 0, n);
            input_prev_time(vi, n);
            true
        }
        50 => {
            input_movie_time(vi, 0, n);
            input_next_time(vi, n);
            true
        }
        82 if ctrl => {
            input_raise_windows(n);
            true
        }
        _ => false,
    };
    event.accepted = handled;
    if !handled {
        n.key_unhandled(key);
    }
}
/// `setCRampWithLevelChange`.
pub fn set_cramp_with_level_change(
    vi: &mut ImodView,
    del_black: i32,
    del_white: i32,
    rev_or_false: i32,
    n: &mut dyn InputNativeBoundary,
) {
    if rev_or_false <= 0 {
        if rev_or_false < -1 && vi.black + del_black > vi.white - del_black {
            return;
        }
        vi.black += del_black;
        vi.white += del_white;
    }
    n.draw(vi, 1);
}
/// `inputSetTimeLockForFKeys`.
pub fn input_set_time_lock_for_f_keys(time: i32) {
    TOP_WIN_TIME_LOCK.store(time, Ordering::Relaxed);
}
/// `inputPageUpOrDown`.
pub fn input_page_up_or_down(
    vi: &mut ImodView,
    shifted: i32,
    direction: i32,
    n: &mut dyn InputNativeBoundary,
) {
    if shifted != 0 {
        n.draw(vi, 2);
    } else if direction > 0 {
        input_nextz(vi, 1, n);
    } else {
        input_prevz(vi, 1, n);
    }
}
/// `inputConvertNumLock`.
pub fn input_convert_num_lock(keysym: &mut i32, keypad: &mut i32) {
    if *keypad == 0 {
        return;
    }
    let number = [46, 48, 49, 50, 51, 52, 54, 55, 56, 57];
    let key = [
        KEY_DELETE,
        KEY_INSERT,
        0x0100_0010,
        KEY_DOWN,
        KEY_PAGE_DOWN,
        KEY_LEFT,
        KEY_RIGHT,
        0x0100_0011,
        KEY_UP,
        KEY_PAGE_UP,
    ];
    for i in 0..10 {
        if *keysym == number[i] {
            *keysym = key[i];
            return;
        }
    }
}
/// `inputTestMetaKey`.
pub fn input_test_meta_key(_event: &InputKeyEvent) -> bool {
    false
}
/// `inputTestCtrl`.
pub fn input_test_ctrl(event: &InputKeyEvent) -> i32 {
    i32::from(event.modifiers & input_ctrl_modifier() != 0)
}
/// `inputCtrlModifier`.
pub fn input_ctrl_modifier() -> u32 {
    INPUT_CTRL
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Boundary {
        draws: Vec<i32>,
    }
    impl InputNativeBoundary for Boundary {
        fn draw(&mut self, _: &mut ImodView, flags: i32) {
            self.draws.push(flags);
        }
    }
    #[test]
    fn mouse_box_matches_source_bounds() {
        assert_eq!(mouse_in_box(1, 2, 3, 4, 3, 4), 1);
        assert_eq!(mouse_in_box(1, 2, 3, 4, 0, 4), 0);
    }
    #[test]
    fn numlock_maps_source_keypad_order() {
        let mut k = 50;
        let mut pad = 1;
        input_convert_num_lock(&mut k, &mut pad);
        assert_eq!(k, KEY_DOWN);
    }
    #[test]
    fn cursor_steps_are_bounded() {
        let mut v = ImodView {
            xsize: 3,
            ysize: 2,
            zsize: 4,
            xmouse: 2.,
            ymouse: 0.,
            zmouse: 3.,
            ..Default::default()
        };
        let mut b = Boundary::default();
        input_nextx(&mut v, &mut b);
        input_prevy(&mut v, &mut b);
        input_nextz(&mut v, 1, &mut b);
        assert_eq!((v.xmouse, v.ymouse, v.zmouse), (2., 0., 3.));
    }
}
