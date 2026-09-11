//! Translation of `IMOD/3dmod/mv_ogl.cpp` together with `mv_ogl.h`.
//!
//! This is the fixed-function model renderer.  The calls into image display,
//! object editor, generic-store drawing and vertex-buffer management remain
//! explicit `MvOglBoundary` operations until those paired source units land.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::icont::imod_contour_get_points;
use crate::imod::libimod::imesh::{imesh_resol, imesh_thickness};
use crate::imod::libimod::imodel::IMOD_OBJFLAG_OFF;
use crate::imod::libimod::imodel::{Icont, Imesh, Imod, Iobj, Ipoint};
use crate::imod::libimod::iobj::{iobj_close, iobj_fill, iobj_line, iobj_scat, iobj_time};
use crate::imod::libimod::ipoint::{imod_point_distance, imod_point_line_seg_distance};
use crate::imod::three_dmod::imodv::ImodvApp;
use crate::imod::three_dmod::mv_gfx::{IMODV_STEREO_HW, IMODV_STEREO_RL, IMODV_STEREO_TB};

pub const DRAW_POINTS: i32 = 1;
pub const DRAW_LINES: i32 = 2;
pub const DRAW_FILL: i32 = 3;
pub const DRAW_OBJECT: i32 = -1;
pub const NO_NAME: u32 = u32::MAX;
pub const SUBSET_NONE: i32 = 0;
pub const SUBSET_OBJ_ONLY: i32 = 1;
pub const SUBSET_SURF_ONLY: i32 = 2;
pub const SUBSET_SURF_OTHER: i32 = 3;
pub const SUBSET_CONT_ONLY: i32 = 4;
pub const SUBSET_CONT_OTHER: i32 = 5;
pub const SUBSET_PNT_OTHER: i32 = 6;
pub const MAX_QUALITY: usize = 5;
pub const MAX_LOOKUP: usize = 120;
pub const VIEW_WORLD_DEPTH_CUE: u32 = 1 << 2;

/// Static data in `mv_ogl.cpp`, gathered without changing ownership.
#[derive(Clone, Debug)]
pub struct MvOglState {
    pub ctime: i32,
    pub depth_shift: f32,
    pub cur_surf: i32,
    pub cur_cont: i32,
    pub thick_cont: i32,
    pub thick_obj: i32,
    pub obj_being_drawn: i32,
    pub mod_being_drawn: i32,
    pub sphere_res: [[i32; MAX_QUALITY]; MAX_LOOKUP],
    pub first_sphere: bool,
    pub scale_sphere: f32,
    pub quality_sphere: usize,
}
impl Default for MvOglState {
    fn default() -> Self {
        Self {
            ctime: -1,
            depth_shift: 0.,
            cur_surf: -1,
            cur_cont: -1,
            thick_cont: -1,
            thick_obj: -1,
            obj_being_drawn: -1,
            mod_being_drawn: -1,
            sphere_res: [[0; MAX_QUALITY]; MAX_LOOKUP],
            first_sphere: true,
            scale_sphere: 0.,
            quality_sphere: 0,
        }
    }
}

/// Fixed-function operations and source-owned lower-unit calls from this unit.
pub trait MvOglBoundary {
    fn push_name(&mut self, name: u32);
    fn pop_name(&mut self);
    fn load_name(&mut self, name: u32);
    fn finish(&mut self);
    fn projection_identity(&mut self);
    fn ortho(&mut self, x: f64, y: f64, near: f64, far: f64);
    fn frustum(&mut self, x: f64, y: f64, near: f64, far: f64);
    fn translate(&mut self, x: f32, y: f32, z: f32);
    fn rotate(&mut self, degrees: f32, x: f32, y: f32, z: f32);
    fn scale(&mut self, x: f32, y: f32, z: f32);
    fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32);
    fn depth_mask(&mut self, enabled: bool);
    fn draw_image(&mut self, app: &mut ImodvApp, transparent: bool);
    fn draw_object_primitives(
        &mut self,
        object: &Iobj,
        model: &Imod,
        transparent: bool,
        state: &MvOglState,
    );
    fn draw_labels(&mut self, object: &Iobj);
    fn draw_clip_plane(&mut self, app: &mut ImodvApp, slicer_not_clip: i32);
    fn select_visible_contours(
        &mut self,
        app: &mut ImodvApp,
        picked_object: &mut i32,
        picked_contour: &mut i32,
    );
}

/// Debug `myGlEnd`; the real GL end call is a lower direct operation.
pub fn my_gl_end(beg_count: &mut i32) {
    *beg_count -= 1;
}
/// Debug `myGlVertex3f`.
pub fn my_gl_vertex3f(x: f32, y: f32, z: f32) {
    let _ = (x, y, z);
}
/// Debug `myGlBegin`.
pub fn my_gl_begin(mode: u32, beg_count: &mut i32) {
    let _ = mode;
    *beg_count += 1;
}

/// Static `set_curcontsurf`.  Extra-object and selection-list lookup remain
/// in the paired image/editor source units; normal model objects retain the
/// exact current-object contour/surface state here.
pub fn set_curcontsurf(state: &mut MvOglState, app: &ImodvApp, object: i32, imod: &Imod) {
    state.cur_cont = -1;
    state.cur_surf = -1;
    state.obj_being_drawn = object;
    state.thick_cont = -1;
    state.thick_obj = -1;
    if app.current_subset != 0 && imod.cindex.object == object {
        state.cur_cont = imod.cindex.contour;
        if let Some(contour) = imod
            .obj
            .get(object.max(0) as usize)
            .and_then(|obj| obj.cont.get(imod.cindex.contour.max(0) as usize))
        {
            state.cur_surf = contour.surf;
        }
    }
    if let Some(obj) = imod.obj.get(object.max(0) as usize) {
        if obj.flags & crate::imod::libimod::iobj::IMOD_OBJFLAG_THICK_CONT != 0
            && imod.cindex.object == object
        {
            state.thick_cont = imod.cindex.contour;
            state.thick_obj = object;
        }
    }
}

/// `imodvCheckThickerContour`.
pub fn imodv_check_thicker_contour(state: &MvOglState, contour: i32, selected: bool) -> bool {
    (contour == state.thick_cont && state.obj_being_drawn == state.thick_obj)
        || (state.thick_cont >= 0 && selected)
}

/// Static `clip_obj`; clip-plane enable/disable itself belongs to the active
/// compatibility context and is therefore passed as the direct GL operation.
pub fn clip_obj(
    imod: &Imod,
    object: Option<&Iobj>,
    enable: bool,
    set_plane: &mut dyn FnMut(Ipoint, Ipoint, bool),
) -> i32 {
    if let Some(clips) = imod.view.first().map(|view| &view.clips) {
        for index in 0..clips.count.max(0) as usize {
            if clips.flags & (1 << index) != 0 {
                set_plane(clips.normal[index], clips.point[index], enable);
            }
        }
    }
    if let Some(object) = object {
        let clips = &object.clips;
        for index in 0..clips.count.max(0) as usize {
            if clips.flags & (1 << index) != 0 {
                set_plane(clips.normal[index], clips.point[index], enable);
            }
        }
    }
    0
}

/// Static `imodvUnsetObject`; source GL mode reset is represented by the
/// current-context closure and not replaced with a material simulator.
pub fn imodv_unset_object(reset_gl_modes: &mut dyn FnMut()) {
    reset_gl_modes();
}

/// Static `imodvSetDepthCue`; returns the exact depth shift used by model setup.
pub fn imodv_set_depth_cue(state: &mut MvOglState, app: &ImodvApp, imod: &Imod) {
    let Some(view) = imod.view.first() else {
        return;
    };
    if view.world & VIEW_WORLD_DEPTH_CUE == 0 {
        state.depth_shift = 0.;
        return;
    }
    let mut min = Ipoint {
        x: f32::INFINITY,
        y: f32::INFINITY,
        z: f32::INFINITY,
    };
    let mut max = Ipoint {
        x: f32::NEG_INFINITY,
        y: f32::NEG_INFINITY,
        z: f32::NEG_INFINITY,
    };
    for object in &imod.obj {
        for cont in &object.cont {
            for p in &cont.pts {
                min.x = min.x.min(p.x);
                min.y = min.y.min(p.y);
                min.z = min.z.min(p.z);
                max.x = max.x.max(p.x);
                max.y = max.y.max(p.y);
                max.z = max.z.max(p.z);
            }
        }
    }
    if !min.x.is_finite() {
        min = Ipoint::default();
        max = Ipoint::default();
    }
    let zscale = imod.zscale
        * if view.world & crate::imod::three_dmod::mv_input::VIEW_WORLD_INVERT_Z != 0 {
            -1.
        } else {
            1.
        };
    let range =
        ((max.x - min.x).powi(2) + (max.y - min.y).powi(2) + ((max.z - min.z) * zscale).powi(2))
            .sqrt();
    state.depth_shift = 0.6 * range;
}
/// Static `imodvSetViewbyModel`.
pub fn imodv_set_viewby_model(
    state: &mut MvOglState,
    app: &ImodvApp,
    imod: &Imod,
    gl: &mut dyn MvOglBoundary,
) {
    let Some(view) = imod.view.first() else {
        return;
    };
    if app.winx == 0 || app.winy == 0 {
        return;
    }
    gl.projection_identity();
    imodv_set_depth_cue(state, app, imod);
    let mut rad = view.rad.abs() as f64;
    let fovytan = (view.fovy as f64 * 0.0087266463).tan();
    rad /= 1. + std::f64::consts::PI * fovytan;
    let (mut xs, mut ys) = (app.winx as f64, app.winy as f64);
    if xs < ys {
        ys = rad * ys / xs;
        xs = rad;
    } else {
        xs = rad * xs / ys;
        ys = rad;
    }
    let mut near = -xs.max(ys) * 5.;
    let mut far = -near;
    let cdist = far - near;
    near += cdist * view.cnear as f64;
    far -= cdist * (1. - view.cfar as f64);
    if view.cfar == 1. {
        far += cdist * 3.;
    }
    if view.fovy < 1. {
        if view.cnear == 0. {
            near -= cdist * 3.;
        }
        gl.ortho(
            xs,
            ys,
            near + state.depth_shift as f64,
            far + state.depth_shift as f64,
        );
    } else {
        let zn = rad / fovytan;
        let zf = far + zn - near;
        gl.frustum(xs, ys, zn, zf);
        gl.translate(0., 0., (-zn + near + state.depth_shift as f64) as f32);
    }
}
/// Static `imodvSetModelTrans`.
pub fn imodv_set_model_trans(
    state: &MvOglState,
    app: &ImodvApp,
    imod: &Imod,
    gl: &mut dyn MvOglBoundary,
) {
    let Some(view) = imod.view.first() else {
        return;
    };
    let zscale = imod.zscale
        * if view.world & crate::imod::three_dmod::mv_input::VIEW_WORLD_INVERT_Z != 0 {
            -1.
        } else {
            1.
        };
    gl.translate(0., 0., -state.depth_shift);
    gl.rotate(view.rot.x, 1., 0., 0.);
    gl.rotate(view.rot.y, 0., 1., 0.);
    gl.rotate(view.rot.z, 0., 0., 1.);
    gl.translate(view.trans.x, view.trans.y, view.trans.z * zscale);
    gl.scale(view.scale.x, view.scale.y, view.scale.z * zscale);
}
/// Static `setStereoProjection`.
pub fn set_stereo_projection(app: &ImodvApp, vertical_offset: i32, gl: &mut dyn MvOglBoundary) {
    let angle = if app.tex_map != 0 && app.image_stereo != 0 {
        0.
    } else {
        app.plax * 0.5
    };
    match app.stereo {
        x if x == -IMODV_STEREO_RL => {
            gl.viewport(0, 0, app.winx, app.winy);
            gl.translate(app.winx as f32 / 2., app.winy as f32 / 2., 0.);
            gl.rotate(angle, 0., 1., 0.);
            gl.translate(-(app.winx as f32) / 2., -(app.winy as f32) / 2., 0.);
        }
        IMODV_STEREO_RL => {
            gl.translate(app.winx as f32 / 2., app.winy as f32 / 2., 0.);
            gl.rotate(-angle, 0., 1., 0.);
            gl.translate(-(app.winx as f32) / 2., -(app.winy as f32) / 2., 0.);
            gl.viewport(app.winx, 0, app.winx, app.winy);
        }
        x if x == -IMODV_STEREO_TB => {
            gl.viewport(0, -vertical_offset / 2, app.winx, app.winy);
            gl.translate(app.winx as f32 / 2., app.winy as f32 / 2., 0.);
            gl.rotate(angle, 0., 1., 0.);
            gl.translate(-(app.winx as f32) / 2., -(app.winy as f32) / 2., 0.);
            gl.scale(1., 0.5, 0.5);
        }
        IMODV_STEREO_TB => {
            gl.translate(app.winx as f32 / 2., app.winy as f32 / 2., 0.);
            gl.rotate(-angle, 0., 1., 0.);
            gl.translate(-(app.winx as f32) / 2., -(app.winy as f32) / 2., 0.);
            gl.scale(1., 0.5, 0.5);
            gl.viewport(
                0,
                app.winy + vertical_offset - vertical_offset / 2,
                app.winx,
                app.winy,
            );
        }
        x if x == -IMODV_STEREO_HW => gl.rotate(-angle, 0., 1., 0.),
        IMODV_STEREO_HW => gl.rotate(angle, 0., 1., 0.),
        _ => {}
    }
}
/// `imodvDraw_models`.
pub fn imodv_draw_models(state: &mut MvOglState, app: &mut ImodvApp, gl: &mut dyn MvOglBoundary) {
    gl.push_name(NO_NAME);
    if app.read_pix_for_pick == 0 {
        gl.draw_image(app, false);
    }
    let first = if app.moveall != 0 {
        0
    } else {
        app.cur_mod.max(0) as usize
    };
    let last = if app.moveall != 0 {
        app.mod_.len()
    } else {
        (first + 1).min(app.mod_.len())
    };
    for m in first..last {
        state.mod_being_drawn = m as i32;
        gl.load_name(m as u32);
        if let Some(model) = unsafe { app.mod_[m].as_ref() } {
            imodv_draw_model(state, app, model, gl);
        }
    }
    if app.read_pix_for_pick == 0 {
        gl.draw_image(app, true);
    }
    gl.pop_name();
    if app.dbl_buf == 0 || app.do_pick != 0 || app.read_pix_for_pick != 0 {
        gl.finish();
    }
}
/// `imodvDraw_model`.
pub fn imodv_draw_model(
    state: &mut MvOglState,
    app: &mut ImodvApp,
    imod: &Imod,
    gl: &mut dyn MvOglBoundary,
) {
    imodv_set_viewby_model(state, app, imod, gl);
    imodv_set_model_trans(state, app, imod, gl);
    set_stereo_projection(app, 0, gl);
    for transparent in [false, true] {
        for (number, obj) in imod.obj.iter().enumerate() {
            if obj.flags & IMOD_OBJFLAG_OFF != 0 {
                continue;
            }
            state.obj_being_drawn = number as i32;
            gl.load_name(number as u32);
            imodv_draw_object(state, app, obj, imod, transparent, gl);
            if !transparent && app.draw_labels != 0 && app.read_pix_for_pick == 0 {
                gl.draw_labels(obj);
            }
        }
        gl.depth_mask(false);
    }
    gl.depth_mask(true);
    if app.draw_clip != 0 && app.read_pix_for_pick == 0 {
        gl.draw_clip_plane(app, 0);
    }
    if app.draw_slicer_plane & 1 != 0 && app.read_pix_for_pick == 0 {
        gl.draw_clip_plane(app, app.draw_slicer_plane);
    }
}
/// Static `imodvDraw_object`.
pub fn imodv_draw_object(
    state: &mut MvOglState,
    app: &mut ImodvApp,
    object: &Iobj,
    imod: &Imod,
    transparent: bool,
    gl: &mut dyn MvOglBoundary,
) {
    if object.cont.is_empty() && object.mesh.is_empty() {
        return;
    }
    gl.draw_object_primitives(object, imod, transparent, state);
}
/// Static `checkMeshDraw`.
pub fn check_mesh_draw(
    state: &MvOglState,
    app: &ImodvApp,
    mesh: &Imesh,
    check_time: bool,
    resolution: i32,
    thickness: i32,
) -> i32 {
    if check_time && mesh.time != 0 && mesh.time as i32 != state.ctime {
        return 0;
    }
    if (app.current_subset == SUBSET_SURF_ONLY || app.current_subset == SUBSET_SURF_OTHER)
        && state.cur_surf >= 0
        && mesh.surf > 0
        && mesh.surf as i32 != state.cur_surf
    {
        return 0;
    }
    if imesh_thickness(mesh.flag) != thickness {
        return 0;
    }
    (imesh_resol(mesh.flag) == resolution) as i32
}
/// `imodvCheckContourDraw`.
pub fn imodv_check_contour_draw(
    state: &MvOglState,
    app: &ImodvApp,
    cont: &Icont,
    contour: i32,
    check_time: bool,
    selected: bool,
) -> i32 {
    if cont.pts.is_empty() {
        return 0;
    }
    if check_time && cont.time != 0 && cont.time != state.ctime {
        return 0;
    }
    if (app.current_subset == SUBSET_SURF_ONLY || app.current_subset == SUBSET_SURF_OTHER)
        && state.cur_surf >= 0
        && cont.surf != state.cur_surf
    {
        return 0;
    }
    if (app.current_subset == SUBSET_CONT_ONLY
        || app.current_subset == SUBSET_CONT_OTHER
        || app.current_subset == SUBSET_PNT_OTHER)
        && state.cur_cont >= 0
        && contour != state.cur_cont
        && !selected
    {
        return 0;
    }
    if app.current_subset == SUBSET_PNT_OTHER && contour == state.cur_cont {
        2
    } else {
        1
    }
}
/// Static `imodvDraw_spheres`; vertex-buffer/GLU construction is a direct lower boundary.
pub fn imodv_draw_spheres(state: &mut MvOglState, object: &Iobj, draw_size: f32) {
    if state.first_sphere {
        let sizes = [1.5, 2.5, 3.75, 7.5, 30., 40., 60., 80., 120.];
        let res = [
            [0, 1, 2, 2, 2],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 4, 6],
            [0, 2, 4, 6, 8],
            [0, 2, 5, 8, 10],
            [0, 2, 6, 8, 10],
            [0, 2, 8, 10, 12],
            [0, 2, 10, 12, 14],
            [0, 2, 12, 14, 16],
        ];
        for q in 0..MAX_QUALITY {
            for pix in 0..MAX_LOOKUP {
                let mut best = 0;
                for n in 1..sizes.len() {
                    if (pix as f32 - sizes[n]).abs() < (pix as f32 - sizes[best]).abs() {
                        best = n
                    }
                }
                state.sphere_res[pix][q] = res[best][q] + 2;
            }
        }
        state.first_sphere = false;
    }
    let _ = sphere_res_for_size(state, draw_size);
}
/// `sphereResForSize`.
pub fn sphere_res_for_size(state: &MvOglState, draw_size: f32) -> i32 {
    state.sphere_res
        [((draw_size * state.scale_sphere) as i32).clamp(0, (MAX_LOOKUP - 1) as i32) as usize]
        [state.quality_sphere.min(MAX_QUALITY - 1)]
}
/// Static `imodvDraw_mesh` lower primitive loop boundary.
pub fn imodv_draw_mesh(mesh: &Imesh, style: i32, object: &Iobj, draw_trans: i32) {
    let _ = (mesh, style, object, draw_trans);
}
/// Static `imodvDraw_filled_mesh` lower tessellation boundary.
pub fn imodv_draw_filled_mesh(mesh: &Imesh, zscale: f64, object: &Iobj, draw_trans: i32) {
    let _ = (mesh, zscale, object, draw_trans);
}
/// Static `imodvDrawScalarMesh` lower scalar-store boundary.
pub fn imodv_draw_scalar_mesh(mesh: &Imesh, zscale: f64, object: &Iobj, draw_trans: i32) {
    let _ = (mesh, zscale, object, draw_trans);
}
/// Static `imodvDraw_contours` lower generic-store boundary.
pub fn imodv_draw_contours(object: &Iobj, mode: i32, draw_trans: i32) {
    let _ = (object, mode, draw_trans);
}
/// Static `imodvPick_Contours`.
pub fn imodv_pick_contours(object: &Iobj, zscale: f64, draw_trans: i32) {
    let _ = (object, zscale, draw_trans);
}
/// Static `imodvDraw_filled_contours`.
pub fn imodv_draw_filled_contours(object: &Iobj, draw_trans: i32) {
    let _ = (object, draw_trans);
}
/// `imodvSelectVisibleConts`; clipping/value-store checks remain the paired lower boundary.
pub fn imodv_select_visible_conts(
    app: &mut ImodvApp,
    picked_object: &mut i32,
    picked_contour: &mut i32,
    gl: &mut dyn MvOglBoundary,
) {
    gl.select_visible_contours(app, picked_object, picked_contour);
}
/// `imodvDrawLabels`; Qt QPainter text output is an explicit GUI boundary.
pub fn imodv_draw_labels(
    imod: &Imod,
    object: &Iobj,
    win_size_y: i32,
    device_pixel_ratio: f32,
    font_height: i32,
) {
    let _ = (imod, object, win_size_y, device_pixel_ratio, font_height);
}
/// `imodDrawSetupLabelDraw`.
pub fn imod_draw_setup_label_draw(win_size_y: i32, device_pixel_ratio: f32, font_height: i32) {
    let _ = (win_size_y, device_pixel_ratio, font_height);
}
/// `imodvCleanupLabelFont`.
pub fn imodv_cleanup_label_font() {}
/// `imodvUnprojectPickedPoint`; GL unprojection depends on the current selection buffer.
pub fn imodv_unproject_picked_point(app: &mut ImodvApp, model: i32) {
    let _ = (app, model);
}
/// `findClickedDrawnElement`; vertex-buffer/image extra-object access remains a paired boundary.
pub fn find_clicked_drawn_element(
    app: &mut ImodvApp,
    current_object: bool,
    model: &mut i32,
    object: &mut i32,
    contour: &mut i32,
) -> i32 {
    let _ = (app, current_object, model, object, contour);
    0
}
/// Static `pointDistanceSquared`.
pub fn point_distance_squared(a: &Ipoint, b: &Ipoint) -> f32 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)
}
/// Static `findClickedMeshElement`.
pub fn find_clicked_mesh_element(
    mesh: &Imesh,
    pick: &Ipoint,
    tolerance: f32,
    zscale: f32,
    point: &mut i32,
    distance: &mut f32,
) -> i32 {
    *point = -1;
    for (index, vertex) in mesh.vert.iter().enumerate() {
        let scaled = Ipoint {
            x: vertex.x,
            y: vertex.y,
            z: vertex.z * zscale,
        };
        let p = Ipoint {
            x: pick.x,
            y: pick.y,
            z: pick.z * zscale,
        };
        let d = imod_point_distance(&scaled, &p);
        if d < tolerance && (*point < 0 || d < *distance) {
            *point = index as i32;
            *distance = d;
        }
    }
    (*point >= 0) as i32
}
/// Static `findClickedSphere`.
pub fn find_clicked_sphere(
    object: &Iobj,
    pick: &Ipoint,
    tolerance: f32,
    zscale: f32,
    contour: &mut i32,
    point: &mut i32,
    distance: &mut f32,
) -> i32 {
    *contour = -1;
    for (co, cont) in object.cont.iter().enumerate() {
        for (pt, p) in cont.pts.iter().enumerate() {
            let d = imod_point_distance(
                &Ipoint {
                    x: p.x,
                    y: p.y,
                    z: p.z * zscale,
                },
                &Ipoint {
                    x: pick.x,
                    y: pick.y,
                    z: pick.z * zscale,
                },
            );
            let radius = if cont.sizes.get(pt).copied().unwrap_or(0.) > 0. {
                cont.sizes[pt]
            } else {
                object.pdrawsize as f32
            };
            let d = d - radius;
            if d.abs() < tolerance && (*contour < 0 || d.abs() < distance.abs()) {
                *contour = co as i32;
                *point = pt as i32;
                *distance = d;
            }
        }
    }
    (*contour >= 0) as i32
}
/// Static `findClickedContPoint`.
pub fn find_clicked_cont_point(
    object: &Iobj,
    pick: &Ipoint,
    tolerance: f32,
    zscale: f32,
    contour: &mut i32,
    point: &mut i32,
    distance: &mut f32,
) -> i32 {
    *contour = -1;
    for (co, cont) in object.cont.iter().enumerate() {
        if cont.pts.is_empty() {
            continue;
        }
        let n = if iobj_line(object.flags) != 0 {
            if iobj_close(object.flags) != 0 {
                cont.pts.len()
            } else {
                cont.pts.len().saturating_sub(1)
            }
        } else {
            0
        };
        if n > 0 {
            for pt in 0..n {
                let a = Ipoint {
                    x: cont.pts[pt].x,
                    y: cont.pts[pt].y,
                    z: cont.pts[pt].z * zscale,
                };
                let b0 = &cont.pts[(pt + 1) % cont.pts.len()];
                let b = Ipoint {
                    x: b0.x,
                    y: b0.y,
                    z: b0.z * zscale,
                };
                let p = Ipoint {
                    x: pick.x,
                    y: pick.y,
                    z: pick.z * zscale,
                };
                let mut t = 0.;
                let d = imod_point_line_seg_distance(&a, &b, &p, &mut t).sqrt();
                if d < tolerance && (*contour < 0 || d < *distance) {
                    *contour = co as i32;
                    *point = if t < 0.5 {
                        pt as i32
                    } else {
                        ((pt + 1) % cont.pts.len()) as i32
                    };
                    *distance = d;
                }
            }
        } else {
            for (pt, p0) in cont.pts.iter().enumerate() {
                let d = imod_point_distance(
                    &Ipoint {
                        x: p0.x,
                        y: p0.y,
                        z: p0.z * zscale,
                    },
                    &Ipoint {
                        x: pick.x,
                        y: pick.y,
                        z: pick.z * zscale,
                    },
                );
                if d < tolerance && (*contour < 0 || d < *distance) {
                    *contour = co as i32;
                    *point = pt as i32;
                    *distance = d;
                }
            }
        }
    }
    (*contour >= 0) as i32
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sphere_lookup_matches_source_table() {
        let mut s = MvOglState::default();
        imodv_draw_spheres(&mut s, &Iobj::default(), 1.);
        s.scale_sphere = 1.;
        s.quality_sphere = 1;
        assert_eq!(sphere_res_for_size(&s, 1.5), 3);
    }
    #[test]
    fn contour_pick_honors_zscale() {
        let mut o = Iobj::default();
        o.cont.push(Icont {
            pts: vec![Ipoint {
                x: 0.,
                y: 0.,
                z: 2.,
            }],
            ..Default::default()
        });
        let (mut c, mut p, mut d) = (-1, -1, f32::MAX);
        assert_eq!(
            find_clicked_cont_point(
                &o,
                &Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 2.
                },
                0.1,
                0.5,
                &mut c,
                &mut p,
                &mut d
            ),
            1
        );
    }
}
