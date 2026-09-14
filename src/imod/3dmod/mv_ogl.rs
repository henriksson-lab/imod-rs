//! Translation of `IMOD/3dmod/mv_ogl.cpp` together with `mv_ogl.h`.
//!
//! This is the fixed-function model renderer.  The calls into image display,
//! object editor, generic-store drawing and vertex-buffer management remain
//! explicit `MvOglBoundary` operations until those paired source units land.
#![allow(dead_code, unused_variables)]

use std::sync::{LazyLock, Mutex};

use crate::imod::libimod::icont::{ICONT_STIPPLED, imod_contour_get_points};
use crate::imod::libimod::imat::{B3D_X, B3D_Y, imod_mat_rot, imod_mat_transform3d};
use crate::imod::libimod::imesh::{
    IMESH_FLAG_NMAG, IMOD_MESH_BGNBIGPOLY, IMOD_MESH_BGNPOLY, IMOD_MESH_BGNPOLYNORM,
    IMOD_MESH_BGNPOLYNORM2, IMOD_MESH_BGNTRI, IMOD_MESH_END, IMOD_MESH_ENDPOLY, IMOD_MESH_ENDTRI,
    IMOD_MESH_NORMAL, IMOD_MESH_SWAP, imesh_resol, imesh_thickness, imod_mesh_nearest_res,
    imod_mesh_poly_norm_factors,
};
use crate::imod::libimod::imodel::IMOD_OBJFLAG_OFF;
use crate::imod::libimod::imodel::Iplane;
use crate::imod::libimod::imodel::{
    ICONT_OPEN, ICONT_WILD, IMOD_CLIPSIZE, Iclip_planes, Icont, Iindex, Imesh, Imod, Iobj, Ipoint,
    Iview,
};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_ANTI_ALIAS, IMOD_OBJFLAG_DRAW_LABEL, IMOD_OBJFLAG_EXTRA_MODV, IMOD_OBJFLAG_FCOLOR,
    IMOD_OBJFLAG_FCOLOR_PNT, IMOD_OBJFLAG_PNT_NOMODV, IMOD_OBJFLAG_SCALAR, IMOD_OBJFLAG_TEMPUSE,
    IMOD_OBJFLAG_TWO_SIDE, iobj_close, iobj_fill, iobj_line, iobj_mesh, iobj_off, iobj_scat,
    iobj_time,
};
use crate::imod::libimod::iplane::{imod_plane_set_from_clips, imod_planes_clip};
use crate::imod::libimod::ipoint::{
    imod_point_distance, imod_point_get_size, imod_point_line_seg_distance,
};
use crate::imod::libimod::istore::{
    DrawProps, istore_first_change_index, istore_skip_to_index, istore_trans_state_matches,
};
use crate::imod::three_dmod::finegrain::{
    CHANGED_3DWIDTH, CHANGED_COLOR, FinegrainRenderBoundary, FinegrainValueState, HANDLE_3DWIDTH,
    HANDLE_MESH_COLOR, HANDLE_MESH_FCOLOR, HANDLE_TRANS, HANDLE_VALUE1, ifg_cont_trans_match,
    ifg_handle_color_trans, ifg_handle_cont_change, ifg_handle_mesh_change, ifg_handle_next_change,
    ifg_handle_surf_change, ifg_make_value_map, ifg_mesh_trans_match, ifg_setup_value_drawing,
};
use crate::imod::three_dmod::imod::{imod_print_stderr, imod_trace};
use crate::imod::three_dmod::imod_edit::{
    ImodEditSelection, imod_selection_list_add, imod_selection_list_clear,
    imod_selection_list_query,
};
use crate::imod::three_dmod::imodv::ImodvApp;
use crate::imod::three_dmod::imodview::ImodView;
use crate::imod::three_dmod::mv_gfx::{IMODV_STEREO_HW, IMODV_STEREO_RL, IMODV_STEREO_TB};
use crate::imod::three_dmod::mv_input::clip_center_and_angles;
use crate::imod::three_dmod::mv_objed::objed_object;

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
pub const MAX_MEASURES: usize = 9;
pub const VIEW_WORLD_DEPTH_CUE: u32 = 1 << 2;
/// `GEN_STORE_MINMAX1` (`istore.h:48`).
pub const GEN_STORE_MINMAX1: i16 = 11;
/// `WORLD_QUALITY_SHIFT` and `WORLD_QUALITY_BITS` (`imodel.h:214`).
pub const WORLD_QUALITY_SHIFT: u32 = 8;
pub const WORLD_QUALITY_BITS: u32 = 7 << WORLD_QUALITY_SHIFT;
/// `RADIANS_PER_DEGREE` (`b3dutil.h:68`).
pub const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// The `GL/gl.h` primitive-mode enumerants this unit names.  They are written
/// out because the boundary carries the source's own `glBegin` argument.
pub const GL_POINTS: u32 = 0x0000;
pub const GL_LINES: u32 = 0x0001;
pub const GL_LINE_LOOP: u32 = 0x0002;
pub const GL_LINE_STRIP: u32 = 0x0003;
pub const GL_TRIANGLES: u32 = 0x0004;
pub const GL_TRIANGLE_STRIP: u32 = 0x0005;
pub const GL_TRIANGLE_FAN: u32 = 0x0006;
pub const GL_QUADS: u32 = 0x0007;
pub const GL_QUAD_STRIP: u32 = 0x0008;
pub const GL_POLYGON: u32 = 0x0009;
/// `glBlendFunc` factors used here (`GL/gl.h`).
pub const GL_ZERO: u32 = 0;
pub const GL_ONE: u32 = 1;
pub const GL_SRC_ALPHA: u32 = 0x0302;
pub const GL_ONE_MINUS_SRC_ALPHA: u32 = 0x0303;
/// `gluQuadricDrawStyle` styles (`GL/glu.h`).
pub const GLU_POINT: u32 = 100010;
pub const GLU_LINE: u32 = 100011;
pub const GLU_FILL: u32 = 100012;

/// Opaque stand-in for the Qt font cached by label drawing.
#[derive(Clone, Debug, Default)]
pub struct LabelFont;

/// Opaque stand-in for the Qt painter cached by label drawing.
#[derive(Clone, Debug, Default)]
pub struct LabelPainter;

/// Static label-drawing state (`sLabelFont`, `sLabelPainter`, and `sFontSize`).
///
/// The concrete Qt resources remain at the GUI boundary, but their lifetime and
/// cached font-size state are retained here so cleanup follows the C++ source.
#[derive(Clone, Debug)]
pub struct LabelFontState {
    pub label_font: Option<LabelFont>,
    pub label_painter: Option<LabelPainter>,
    pub font_size: i32,
}

impl Default for LabelFontState {
    fn default() -> Self {
        Self {
            label_font: None,
            label_painter: None,
            font_size: -1,
        }
    }
}

pub static LABEL_FONT_STATE: LazyLock<Mutex<LabelFontState>> =
    LazyLock::new(|| Mutex::new(LabelFontState::default()));

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
    /// `finegrain.cpp`'s value-drawing statics, which every routine here sets
    /// up through `ifgSetupValueDrawing` before drawing.
    pub values: FinegrainValueState,
    /// `Imodv->vi->selectionList`, which `imodvCheckContourDraw`,
    /// `imodvCheckThickerContour` and `imodvSelectVisibleConts` all read.
    pub selection: ImodEditSelection,
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
            values: FinegrainValueState::default(),
            selection: ImodEditSelection::default(),
        }
    }
}

/// Fixed-function operations and source-owned lower-unit calls from this unit.
///
/// `FinegrainRenderBoundary` is a supertrait because `finegrain.cpp`'s
/// `ifgHandleContChange`, `ifgHandleNextChange`, `ifgHandleMeshChange` and
/// `ifgHandleColorTrans` all issue their colour, width and material calls on
/// the very context this unit is drawing into; the source has one GL state,
/// not two.
pub trait MvOglBoundary: FinegrainRenderBoundary {
    fn push_name(&mut self, name: u32);
    fn pop_name(&mut self);
    fn load_name(&mut self, name: u32);
    fn finish(&mut self);
    fn projection_identity(&mut self);
    /// `glMatrixMode(GL_PROJECTION)` alone, as `setStereoProjection` issues it
    /// before its viewport/rotation calls.
    fn projection_mode(&mut self);
    /// `glMatrixMode(GL_MODELVIEW)` plus `glLoadIdentity`, as
    /// `imodvSetModelTrans` issues them before its translate/rotate/scale.
    fn modelview_identity(&mut self);
    /// `glMatrixMode(GL_MODELVIEW)` alone.
    fn modelview_mode(&mut self);
    fn ortho(&mut self, x: f64, y: f64, near: f64, far: f64);
    fn frustum(&mut self, x: f64, y: f64, near: f64, far: f64);
    fn translate(&mut self, x: f32, y: f32, z: f32);
    fn rotate(&mut self, degrees: f32, x: f32, y: f32, z: f32);
    fn scale(&mut self, x: f32, y: f32, z: f32);
    fn push_matrix(&mut self);
    fn pop_matrix(&mut self);
    fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32);
    fn depth_mask(&mut self, enabled: bool);
    /// `glBegin`.
    fn begin(&mut self, mode: u32);
    /// `glEnd`.
    fn end(&mut self);
    /// `glVertex3f`, which is also what `glVertex3fv` on an `Ipoint` is.
    fn vertex3f(&mut self, x: f32, y: f32, z: f32);
    /// `glNormal3f`, which is also what `glNormal3fv` on an `Ipoint` is.
    fn normal3f(&mut self, x: f32, y: f32, z: f32);
    /// `glColor4ub`.
    fn color4ub(&mut self, red: u8, green: u8, blue: u8, alpha: u8);
    /// `glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)` or `GL_FILL`.
    fn polygon_mode_line(&mut self, line: bool);
    /// `glFrontFace(GL_CW)` or `GL_CCW`.
    fn front_face_cw(&mut self, cw: bool);
    /// `glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, value)`.
    fn light_model_two_side(&mut self, value: i32);
    /// `glEnable`/`glDisable(GL_BLEND)`.
    fn blend(&mut self, enabled: bool);
    /// `glBlendFunc`.
    fn blend_func(&mut self, source: u32, destination: u32);
    /// `glEnable`/`glDisable(GL_CULL_FACE)`.
    fn cull_face(&mut self, enabled: bool);
    /// `glEnable`/`glDisable(GL_LINE_SMOOTH)`.
    fn line_smooth(&mut self, enabled: bool);
    /// `glEnable`/`glDisable(GL_NORMALIZE)`.
    fn normalize(&mut self, enabled: bool);
    /// `light_on(obj, sModBeingDrawn)` (`mv_light.cpp`).  The source looks the
    /// view up as `Imodv->mod[modind]->view`, which is the view of the model
    /// `imodvSetObject` was handed, so it is passed straight through.
    fn light_on(&mut self, object: &Iobj, view: &Iview, model_zscale: f32);
    /// `light_off()` (`mv_light.cpp`).
    fn light_off(&mut self);
    /// `gluQuadricDrawStyle` on the file-static `qobj`.
    fn quadric_draw_style(&mut self, style: u32);
    /// `gluSphere` on the file-static `qobj`.
    fn sphere(&mut self, radius: f64, slices: i32, stacks: i32);
    /// `glGenLists`.
    fn gen_lists(&mut self, range: i32) -> u32;
    /// `glNewList(list, GL_COMPILE)`.
    fn new_list(&mut self, list: u32);
    /// `glEndList`.
    fn end_list(&mut self);
    /// `glCallList`.
    fn call_list(&mut self, list: u32);
    /// `glDeleteLists`.
    fn delete_lists(&mut self, list: u32, range: i32);
    /// The `gluNewTess`/`gluBeginPolygon`/`gluTessVertex`/`gluEndPolygon`/
    /// `gluDeleteTess` sequence of an `IMOD_MESH_BGNBIGPOLY` block, whose
    /// callbacks are `glBegin`, `glVertex3fv` and `glEnd`.
    fn tess_polygon(&mut self, vertices: &[Ipoint]);
    /// `setupFilledContTesselator` (`utilities.cpp:1182`).
    fn setup_filled_cont_tesselator(&mut self);
    /// `drawFilledPolygon` (`utilities.cpp:1194`).
    fn draw_filled_polygon(&mut self, contour: &Icont);
    /// `utilManagePairedMeshes` (`utilities.cpp:1255`).
    fn manage_paired_meshes(&mut self, object: &Iobj, object_number: i32) -> i32;
    /// `glGetIntegerv(GL_MAX_CLIP_PLANES, &maxPlanes)`.
    fn max_clip_planes(&mut self) -> i32;
    /// `glClipPlane(GL_CLIP_PLANE0 + index, params)`.
    fn clip_plane(&mut self, index: i32, params: [f64; 4]);
    /// `glEnable(GL_CLIP_PLANE0 + index)`.
    fn enable_clip_plane(&mut self, index: i32);
    /// `glDisable(GL_CLIP_PLANE0 + index)`.
    fn disable_clip_plane(&mut self, index: i32);
    /// `mvImageAnyClipping()` (`mv_image.cpp:328`).
    fn image_any_clipping(&mut self) -> bool;
    /// `mvImageGetClipPlanes()` (`mv_image.cpp:342`).
    fn image_clip_planes(&mut self) -> Option<Iclip_planes>;
    /// `imodvSetLight(imod->view)` (`mv_light.cpp:44`).
    fn set_light(&mut self, view: &mut Iview);
    /// `utilEnableStipple(Imodv->vi, cont)` (`utilities.cpp`).
    fn enable_stipple(&mut self, draw_stipple: i32, contour: &Icont);
    /// `utilDisableStipple(Imodv->vi, cont)` (`utilities.cpp`).
    fn disable_stipple(&mut self, draw_stipple: i32, contour: &Icont);
    /// `getTopSlicer` plus `getTopSlicerAngles` and `getNormalToPlane`
    /// (`sslice.cpp`), and the top slicer's `mWinx`, `mWiny` and `mZoom`.
    /// Returns non-zero when there is no top slicer or the angles are
    /// unavailable, which is the source's combined early return.
    fn top_slicer_plane(
        &mut self,
        angles: &mut [f32; 3],
        center: &mut Ipoint,
        time: &mut i32,
        normal: &mut Ipoint,
        winx: &mut i32,
        winy: &mut i32,
        zoom: &mut f32,
    ) -> i32;
    fn draw_image(&mut self, app: &mut ImodvApp, transparent: bool);
    /// The source passes the model it is drawing to `imodvDrawLabels`, and the
    /// window size and font metrics come from the window the boundary owns.
    fn draw_labels(&mut self, app: &ImodvApp, model: &Imod, object: &Iobj);
    /// `glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depths)`
    /// of `imodvUnprojectPickedPoint`.
    fn read_depth_pixels(&mut self, x: i32, y: i32, width: i32, height: i32, depths: &mut [f32]);
    /// `glGetDoublev(GL_MODELVIEW_MATRIX/GL_PROJECTION_MATRIX)`,
    /// `glGetIntegerv(GL_VIEWPORT)` and `gluUnProject` of
    /// `imodvUnprojectPickedPoint`.  The source ignores `gluUnProject`'s
    /// return value, so the object coordinates are the whole result.
    fn un_project(
        &mut self,
        winx: f64,
        winy: f64,
        winz: f64,
        objx: &mut f64,
        objy: &mut f64,
        objz: &mut f64,
    );
}
/// Debug `myGlEnd`; the real GL end call is a lower direct operation.
pub fn my_gl_end(beg_count: &mut i32) {
    *beg_count -= 1;
}
/// Debug `myGlVertex3f`.
///
/// The source prints the vertex and then issues the real `glVertex3f`; it is
/// reached only through the `#define glVertex3f myGlVertex3f` on line 69,
/// which is commented out, so nothing calls it.  The boundary argument is the
/// `glVertex3f` it ends with: there is no ambient current context here the way
/// there is in the C++.
pub fn my_gl_vertex3f(x: f32, y: f32, z: f32, gl: &mut dyn MvOglBoundary) {
    imod_print_stderr(&format!("Vertex {x:.1} {y:.1} {z:.1}\n"));
    gl.vertex3f(x, y, z);
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

/// Static `clip_obj`: set up clipping planes for the given object, or just the
/// global clip planes if `obj` is `None`.
pub fn clip_obj(imod: &Imod, obj: Option<&Iobj>, flag: i32, gl: &mut dyn MvOglBoundary) -> i32 {
    let mut params = [0f64; 4];
    let mut cpn = 0i32;
    let mut num_sets = 1;
    let mut clip_arr: [Option<Iclip_planes>; 3] = [None, None, None];
    clip_arr[0] = imod.view.first().map(|view| view.clips.clone());
    if let Some(obj) = obj {
        clip_arr[1] = Some(obj.clips.clone());
        num_sets = 2;
        if gl.image_any_clipping() {
            num_sets = 3;
            clip_arr[2] = gl.image_clip_planes();
        }
    }

    let max_planes = gl.max_clip_planes();

    /* Loop through global if not skip, object, and image view sets, loop through the
    planes in each set, and for each one that is on up to the limit, either turn on
    the plane or disable it based on the flag */
    for clip_set in 0..num_sets {
        let Some(clips) = clip_arr[clip_set].as_ref() else {
            continue;
        };
        for ip in 0..clips.count as usize {
            if clips.flags & (1 << ip) != 0 && cpn < max_planes {
                if flag != 0 {
                    params[0] = clips.normal[ip].x as f64;
                    params[1] = clips.normal[ip].y as f64;
                    params[2] = clips.normal[ip].z as f64;

                    /* DNM 10/13/05: this parameter is evidently supposed to be the
                    negative of this product for the plane equation Ax+By+Cz+D=0,
                    which is why clip points are all maintained as negative values */
                    params[3] = (clips.normal[ip].x * clips.point[ip].x) as f64
                        + (clips.normal[ip].y * clips.point[ip].y) as f64
                        + (clips.normal[ip].z * clips.point[ip].z) as f64;

                    gl.clip_plane(cpn, params);
                    gl.enable_clip_plane(cpn);
                } else {
                    gl.disable_clip_plane(cpn);
                }
                cpn += 1;
            }
        }
    }
    0
}

/// Static `imodvUnsetObject`.  Every mode `imodvSetObject` enables is disabled
/// here, including restoring polygon mode to fill for texture drawing.
pub fn imodv_unset_object(obj: &Iobj, gl: &mut dyn MvOglBoundary) {
    gl.light_off();
    gl.blend(false);
    gl.line_smooth(false);
    gl.cull_face(false);
    gl.polygon_mode_line(false);
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
    gl.modelview_identity();
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
    gl.projection_mode();
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
        let model = app.mod_[m];
        if !model.is_null() {
            unsafe { imodv_draw_model(state, app, model, gl) };
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
///
/// # Safety
/// `imod` must point at a live model, as `a->mod[m]` does in the source.
pub unsafe fn imodv_draw_model(
    state: &mut MvOglState,
    app: &mut ImodvApp,
    imod: *mut Imod,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        if imod.is_null() {
            return;
        }
        imodv_set_viewby_model(state, app, &*imod, gl);
        imodv_set_model_trans(state, app, &*imod, gl);
        set_stereo_projection(app, 0, gl);
        // `Imod::ctime` (`imodel.h:457`) is not carried by the translated
        // `Imod` in `libimod`, so `sCTime = imod->ctime` cannot be read here.
        // -1 is the file-static initial value, which makes the time checks
        // behave as they do for an untimed model.
        state.ctime = -1;
        if let Some(view) = (&mut (*imod).view).get_mut(0) {
            gl.set_light(view);
        }
        gl.push_name(NO_NAME);
        if let Some(mut mat) = app.mat.take() {
            crate::imod::three_dmod::mv_input::imodv_rot_scale_matrix(app, &mut mat, &*imod);
            app.mat = Some(mat);
        }

        // The source's second loop pass is over `a->vi->numExtraObj` extra
        // objects through `ivwGetAnExtraObject`; `ImodView` in `imodview.rs`
        // carries no extra-object list, so only the model's own objects are
        // drawn here.
        for transparent in [0, 1] {
            /* If displaying a current subset, set up object limits */
            let mut obstart = 0i32;
            let mut obend = (*imod).obj.len() as i32;
            if (app.current_subset == SUBSET_OBJ_ONLY
                || app.current_subset == SUBSET_SURF_ONLY
                || app.current_subset == SUBSET_CONT_ONLY)
                && (*imod).cindex.object >= 0
            {
                obstart = (*imod).cindex.object;
                obend = obstart + 1;
            }

            // If drawing only extra objects, set up to skip all regular ones
            if app.draw_extra_only != 0 {
                obend = obstart - 1;
            }

            let mut number = obstart;
            while number < obend {
                let obj: *mut Iobj = &mut (&mut (*imod).obj)[number as usize];
                if transparent == 0 {
                    (*obj).flags &= !IMOD_OBJFLAG_TEMPUSE;
                }
                if iobj_off((*obj).flags) == 0
                    && (transparent == 0 || (*obj).flags & IMOD_OBJFLAG_TEMPUSE != 0)
                {
                    set_curcontsurf(state, app, number, &*imod);
                    gl.load_name(number as u32);
                    clip_obj(&*imod, Some(&*obj), 1, gl);
                    imodv_draw_object(state, app, obj, imod, transparent, gl);
                    if transparent == 0 && app.draw_labels != 0 && app.read_pix_for_pick == 0 {
                        // Drawing after the first setObject gave garbled text in
                        // Qt 5, so just undo it, which leaves the colour set for
                        // the labels
                        imodv_set_object(state, app, imod, obj, DRAW_POINTS, 0, gl);
                        imodv_set_object(state, app, imod, obj, 0, 0, gl);
                        gl.draw_labels(app, &*imod, &*obj);
                    }
                    clip_obj(&*imod, Some(&*obj), 0, gl);
                }
                number += 1;
            }
            if app.read_pix_for_pick == 0 {
                gl.depth_mask(false);
            }
        }
        gl.depth_mask(true);
        imodv_cleanup_label_font();
        gl.pop_name();
        if app.draw_clip != 0 && imod == app.imod && app.read_pix_for_pick == 0 {
            draw_current_clip_plane(state, app, 0, gl);
        }
        if app.draw_slicer_plane & 1 != 0 && imod == app.imod && app.read_pix_for_pick == 0 {
            draw_current_clip_plane(state, app, app.draw_slicer_plane, gl);
        }
    }
}

/// Static `imodvSetObject`; sets the OpenGL modes the given object needs.
///
/// # Safety
/// `imod` and `obj` must point at a live model and one of its objects.
pub unsafe fn imodv_set_object(
    state: &MvOglState,
    app: &ImodvApp,
    imod: *mut Imod,
    obj: *mut Iobj,
    style: i32,
    mut draw_trans: i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let red: f32;
        let green: f32;
        let blue: f32;
        let trans: f32 = 1.0f32 - ((*obj).trans as f32 * 0.01f32);
        match style {
            0 => {
                imodv_unset_object(&*obj, gl);
                /* DNM 11/30/01: need to return, not break */
                return;
            }
            DRAW_POINTS | DRAW_LINES => {
                if style == DRAW_POINTS {
                    gl.point_size((*obj).linewidth as i32, &*obj);
                }
                if IMOD_OBJFLAG_ANTI_ALIAS & (*obj).flags != 0 {
                    gl.line_smooth(true);
                    draw_trans = 1;
                }
                gl.polygon_mode_line(true);
                gl.line_width((*obj).linewidth as i32, &*obj);
                gl.color4f((*obj).red, (*obj).green, (*obj).blue, trans);
            }
            DRAW_FILL => {
                gl.line_width((*obj).linewidth as i32, &*obj);
                gl.point_size((*obj).linewidth as i32, &*obj);
                if app.wireframe != 0 {
                    gl.polygon_mode_line(true);
                } else {
                    gl.polygon_mode_line(false);
                }
                if (*obj).flags & IMOD_OBJFLAG_TWO_SIDE != 0 {
                    gl.light_model_two_side(1);
                } else {
                    gl.light_model_two_side(0);
                }
                if let Some(view) = (*imod).view.first() {
                    if view.world & crate::imod::three_dmod::mv_input::VIEW_WORLD_INVERT_Z != 0 {
                        gl.front_face_cw(true);
                    } else {
                        gl.front_face_cw(false);
                    }
                }
                if (*obj).flags & IMOD_OBJFLAG_FCOLOR != 0 {
                    red = (*obj).fillred as f32 / 255.0f32;
                    green = (*obj).fillgreen as f32 / 255.0f32;
                    blue = (*obj).fillblue as f32 / 255.0f32;
                    gl.color4f(red, green, blue, trans);
                } else {
                    gl.color4f((*obj).red, (*obj).green, (*obj).blue, trans);
                }
                if app.lighting != 0 && app.wireframe == 0 {
                    if let Some(view) = (*imod).view.first() {
                        gl.light_on(&*obj, view, (*imod).zscale);
                    }
                } else {
                    gl.light_off();
                }
            }
            _ => {}
        }
        if draw_trans != 0 {
            gl.blend(true);
            gl.blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            if (*obj).flags & IMOD_OBJFLAG_TWO_SIDE == 0 && style == DRAW_FILL {
                gl.cull_face(true);
            }
        }
    }
}

/// Static `imodvDraw_object`: determine the types of draws to be done and call
/// the routines.
///
/// # Safety
/// `imod` and `obj` must point at a live model and one of its objects.
pub unsafe fn imodv_draw_object(
    state: &mut MvOglState,
    app: &ImodvApp,
    obj: *mut Iobj,
    imod: *mut Imod,
    draw_trans: i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let mut resol = 0i32;
        let mut flag_save: u32;
        let mut check_time = iobj_time((*obj).flags) as i32;
        let skip_spheres =
            (*obj).flags & IMOD_OBJFLAG_PNT_NOMODV != 0 && iobj_mesh((*obj).flags) != 0;
        let mut has_spheres = (iobj_scat((*obj).flags) != 0
            || ((*obj).pdrawsize > 0 && !skip_spheres))
            && !(*obj).cont.is_empty();

        if state.ctime == 0 {
            check_time = 0;
        }
        if obj.is_null() {
            return;
        }
        if iobj_off((*obj).flags) != 0 {
            return;
        }
        if (*obj).cont.is_empty() && (*obj).mesh.is_empty() {
            return;
        }

        // `zscale` is a double but the source's rescaled form is a float
        // expression: `imod->zscale * zbin` and the division by `xybin` are
        // both single precision before the assignment widens them.
        let mut zscale = (*imod).zscale as f64;
        if app.standalone == 0 {
            let vi = app.vi as *const ImodView;
            if !vi.is_null() {
                zscale = (((*imod).zscale * (*vi).zbin as f32) / (*vi).xybin as f32) as f64;
            }
        }

        // Check for individual point sizes if not scattered and no sphere size
        if !has_spheres && !skip_spheres {
            for co in 0..(*obj).cont.len() {
                if !(&(*obj).cont)[co].sizes.is_empty() {
                    has_spheres = true;
                    break;
                }
            }
        }

        if has_spheres {
            /* scattered points: if they are filled, draw as fill; then draw the
            lines on top if "Fill outline" is selected */
            if iobj_fill((*obj).flags) != 0 || iobj_scat((*obj).flags) == 0 {
                // If fill color for point flag is set, temporarily set fill color flag
                flag_save = (*obj).flags;
                if (*obj).flags & IMOD_OBJFLAG_FCOLOR_PNT != 0 {
                    (*obj).flags |= IMOD_OBJFLAG_FCOLOR;
                }
                imodv_set_object(state, app, imod, obj, DRAW_FILL, draw_trans, gl);
                imodv_draw_spheres(state, app, obj, zscale, DRAW_FILL, draw_trans, gl);
                flag_save |= (*obj).flags & IMOD_OBJFLAG_TEMPUSE;
                (*obj).flags = flag_save;
                if iobj_line((*obj).flags) != 0 && iobj_fill((*obj).flags) != 0 {
                    imodv_set_object(state, app, imod, obj, 0, 0, gl);
                    imodv_set_object(state, app, imod, obj, DRAW_LINES, draw_trans, gl);
                    imodv_draw_spheres(state, app, obj, zscale, DRAW_LINES, draw_trans, gl);
                }
            } else {
                /* or, just draw the lines if that is selected; otherwise draw points */
                if iobj_line((*obj).flags) != 0 {
                    imodv_set_object(state, app, imod, obj, DRAW_LINES, draw_trans, gl);
                    imodv_draw_spheres(state, app, obj, zscale, DRAW_LINES, draw_trans, gl);
                } else {
                    imodv_set_object(state, app, imod, obj, DRAW_POINTS, draw_trans, gl);
                    imodv_draw_spheres(state, app, obj, zscale, DRAW_POINTS, draw_trans, gl);
                }
            }
            imodv_set_object(state, app, imod, obj, 0, 0, gl);
            if iobj_scat((*obj).flags) != 0 {
                return;
            }
        }

        if app.do_pick != 0 {
            imodv_pick_contours(state, app, obj, zscale, draw_trans, gl);
            imodv_set_object(state, app, imod, obj, 0, 0, gl);
            return;
        }

        /*******************************************/
        /* Draw Mesh data instead of Contour data. */
        if iobj_mesh((*obj).flags) != 0 {
            if (*obj).mesh.is_empty() {
                return;
            }

            // Deal with a possible change in mesh thickness by making new mesh
            // pairs, or removing all pairs
            let mesh_thick = (*obj).mesh_thickness as i32;
            if gl.manage_paired_meshes(&*obj, state.obj_being_drawn) != 0 {
                return;
            }

            let mesh_count = (*obj).mesh.len() as i32;
            imod_mesh_nearest_res(&(*obj).mesh, mesh_count, app.lowres, &mut resol);

            /* Fill or fill outline: draw the filled mesh or scalar mesh */
            if iobj_fill((*obj).flags) != 0 {
                gl.modelview_mode();
                gl.push_matrix();
                // `1.0f / zscale` divides a float literal by a double.
                gl.scale(1.0f32, 1.0f32, (1.0f64 / zscale) as f32);
                imodv_set_object(state, app, imod, obj, DRAW_FILL, draw_trans, gl);
                for co in 0..(*obj).mesh.len() {
                    let mesh: *mut Imesh = &mut (&mut (*obj).mesh)[co];
                    if check_mesh_draw(state, app, &*mesh, check_time != 0, resol, mesh_thick) != 0
                    {
                        if (*obj).flags & IMOD_OBJFLAG_SCALAR != 0 {
                            imodv_draw_scalar_mesh(state, app, mesh, zscale, obj, draw_trans, gl);
                        } else {
                            imodv_draw_filled_mesh(state, app, mesh, zscale, obj, draw_trans, gl);
                        }
                    }
                }
                gl.pop_matrix();

                /* Fill outline: draw the mesh lines as well */
                if iobj_line((*obj).flags) != 0 {
                    imodv_set_object(state, app, imod, obj, 0, 0, gl);
                    imodv_set_object(state, app, imod, obj, DRAW_LINES, draw_trans, gl);
                    for co in 0..(*obj).mesh.len() {
                        let mesh: *mut Imesh = &mut (&mut (*obj).mesh)[co];
                        if check_mesh_draw(state, app, &*mesh, check_time != 0, resol, mesh_thick)
                            != 0
                        {
                            imodv_draw_mesh(state, app, mesh, DRAW_LINES, obj, draw_trans, gl);
                        }
                    }
                }
            } else {
                /* Mesh Lines: draw lines in scalar or regular mode */
                if iobj_line((*obj).flags) != 0 {
                    imodv_set_object(state, app, imod, obj, DRAW_LINES, draw_trans, gl);
                    for co in 0..(*obj).mesh.len() {
                        let mesh: *mut Imesh = &mut (&mut (*obj).mesh)[co];
                        if check_mesh_draw(state, app, &*mesh, check_time != 0, resol, mesh_thick)
                            != 0
                        {
                            if (*obj).flags & IMOD_OBJFLAG_SCALAR != 0 && app.read_pix_for_pick == 0
                            {
                                imodv_draw_scalar_mesh(
                                    state, app, mesh, zscale, obj, draw_trans, gl,
                                );
                            } else {
                                imodv_draw_mesh(state, app, mesh, DRAW_LINES, obj, draw_trans, gl);
                            }
                        }
                    }
                } else {
                    /* Mesh Points: draw points in scalar or regular mode */
                    imodv_set_object(state, app, imod, obj, DRAW_POINTS, draw_trans, gl);
                    for co in 0..(*obj).mesh.len() {
                        let mesh: *mut Imesh = &mut (&mut (*obj).mesh)[co];
                        if check_mesh_draw(state, app, &*mesh, check_time != 0, resol, mesh_thick)
                            != 0
                        {
                            if (*obj).flags & IMOD_OBJFLAG_SCALAR != 0 && app.read_pix_for_pick == 0
                            {
                                imodv_draw_scalar_mesh(
                                    state, app, mesh, zscale, obj, draw_trans, gl,
                                );
                            } else {
                                imodv_draw_mesh(state, app, mesh, DRAW_POINTS, obj, draw_trans, gl);
                            }
                        }
                    }
                }
            }
            imodv_set_object(state, app, imod, obj, 0, 0, gl);
            return;
        }

        /* Closed contours with Fill: draw the fill, then draw the outside lines
        if Fill Outline is selected */
        if iobj_close((*obj).flags) != 0 && iobj_fill((*obj).flags) != 0 {
            imodv_set_object(state, app, imod, obj, DRAW_FILL, draw_trans, gl);
            /* We have to either turn off the light or set the normal to
            something, to have display be independent of the state of the
            last object.  Also turn off back-face culling in case of trans */
            gl.light_off();
            gl.cull_face(false);
            imodv_draw_filled_contours(state, app, obj, draw_trans, gl);
            if iobj_line((*obj).flags) != 0 {
                imodv_set_object(state, app, imod, obj, 0, 0, gl);
                imodv_set_object(state, app, imod, obj, DRAW_LINES, draw_trans, gl);
                imodv_draw_contours(state, app, obj, GL_LINE_LOOP, draw_trans, gl);
            }
        } else {
            /* Contours as lines or points; draw as open or closed lines */
            if iobj_line((*obj).flags) != 0 {
                imodv_set_object(state, app, imod, obj, DRAW_LINES, draw_trans, gl);
                if iobj_close((*obj).flags) != 0 {
                    imodv_draw_contours(state, app, obj, GL_LINE_LOOP, draw_trans, gl);
                } else {
                    imodv_draw_contours(state, app, obj, GL_LINE_STRIP, draw_trans, gl);
                }
            } else {
                imodv_set_object(state, app, imod, obj, DRAW_POINTS, draw_trans, gl);
                imodv_draw_contours(state, app, obj, GL_POINTS, draw_trans, gl);
            }
        }

        imodv_set_object(state, app, imod, obj, 0, 0, gl);
    }
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
/***************************************************************************/
/// Static `imodvDraw_spheres`: draw point spheres.
///
/// # Safety
/// `obj` must point at a live object of the model being drawn.
pub unsafe fn imodv_draw_spheres(
    state: &mut MvOglState,
    app: &ImodvApp,
    obj: *mut Iobj,
    zscale: f64,
    style: i32,
    draw_trans: i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let z: f32 = zscale as f32;
        let mut check_time = iobj_time((*obj).flags) as i32;
        let mut drawsize: f32;
        let mut step_res: i32;
        let xybin: i32;
        let mut cont_draw: i32;
        let contsize = (*obj).cont.len();
        let mut handle_flags = (if app.read_pix_for_pick != 0 {
            0
        } else {
            HANDLE_3DWIDTH
        }) | HANDLE_TRANS
            | if style == DRAW_FILL && (*obj).flags & IMOD_OBJFLAG_FCOLOR != 0 {
                HANDLE_MESH_FCOLOR
            } else {
                HANDLE_MESH_COLOR
            };
        let measured_size: [f32; MAX_MEASURES] = [1.5, 2.5, 3.75, 7.5, 30., 40., 60., 80., 120.];
        let measured_res: [[i32; MAX_QUALITY]; MAX_MEASURES] = [
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

        /* first time, build lookup tables for sphere resolution versus size and
        quality */
        if state.first_sphere {
            for i in 0..MAX_QUALITY {
                for j in 0..MAX_LOOKUP {
                    let mut mindiff: f32 = 10000.;
                    let mut mink: usize = 0;
                    for k in 0..MAX_MEASURES {
                        let mut diff: f32 = j as f32 - measured_size[k];
                        if diff < 0. {
                            diff = -diff;
                        }
                        if diff < mindiff {
                            mindiff = diff;
                            mink = k;
                        }
                    }
                    state.sphere_res[j][i] = measured_res[mink][i] + 2;
                }
            }
            state.first_sphere = false;
        }

        if ifg_setup_value_drawing(
            &*obj,
            GEN_STORE_MINMAX1,
            -1,
            &mut state.values,
            &mut crate::imod::three_dmod::xcramp::xcramp_mapfalsecolor,
        ) != 0
        {
            handle_flags |= HANDLE_VALUE1;
        }

        xybin = if app.standalone != 0 {
            1
        } else {
            let vi = app.vi as *const ImodView;
            if vi.is_null() { 1 } else { (*vi).xybin }
        };

        /* Take maximum of quality from world flag setting and from object */
        let world = app
            .mod_
            .get(state.mod_being_drawn.max(0) as usize)
            .and_then(|m| m.as_ref())
            .and_then(|m| m.view.first())
            .map_or(0u32, |v| v.world);
        let rad = app
            .mod_
            .get(state.mod_being_drawn.max(0) as usize)
            .and_then(|m| m.as_ref())
            .and_then(|m| m.view.first())
            .map_or(1.0f32, |v| v.rad);
        let mut quality = (((world & WORLD_QUALITY_BITS) >> WORLD_QUALITY_SHIFT) + 1) as i32;
        if quality <= (*obj).quality as i32 {
            quality = (*obj).quality as i32 + 1;
        }
        if app.lowres != 0 {
            quality = 0;
        }
        if quality >= MAX_QUALITY as i32 {
            quality = MAX_QUALITY as i32 - 1;
        }
        state.quality_sphere = quality as usize;

        // `0.5 * int` is a double; the division by the float `rad` stays double
        // and only the assignment narrows it.
        state.scale_sphere = (0.5f64
            * (if app.winx > app.winy {
                app.winy
            } else {
                app.winx
            }) as f64
            / rad as f64) as f32;

        if state.ctime == 0 {
            check_time = 0;
        }

        // The vertex-buffer branch of this routine cannot be reached here: it
        // reads and writes `obj->vertBufSphere` (`vertexbuffer.h`), which the
        // translated `Iobj` in `libimod` does not carry, so `vbCleanupSphereVBD`,
        // `analyzeSpheres` and the `glDrawElements` draw have nothing to act on.
        // `vbd` is therefore always NULL, which is exactly the path the source
        // takes before any buffer has been built.

        // When drawing solids, if object has transparency, then check whether any
        // contour or point stores set transparency to 0 and if not, skip
        if draw_trans == 0 && (*obj).trans != 0 && istore_trans_state_matches(&(*obj).store, 0) == 0
        {
            let mut co = 0;
            while co < (*obj).cont.len() {
                if istore_trans_state_matches(&(&(*obj).cont)[co].store, 0) != 0 {
                    break;
                }
                co += 1;
            }
            if co >= (*obj).cont.len() {
                (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                return;
            }
        }

        match style {
            DRAW_POINTS => gl.quadric_draw_style(GLU_POINT),
            DRAW_LINES => gl.quadric_draw_style(GLU_LINE),
            _ => gl.quadric_draw_style(GLU_FILL),
        }

        gl.modelview_mode();
        gl.push_matrix();

        // `z` is a float, so this division is single precision, unlike the
        // `1.0f / zscale` of the filled-mesh setup in `imodvDraw_object`.
        gl.scale(1.0f32, 1.0f32, 1.0f32 / z);
        gl.push_name(NO_NAME);

        /* DNM: Get a display list to draw the default size. */
        // `obj->pdrawsize / xybin` is an integer division before the float
        // assignment; the per-point `drawsize /= xybin` below is not.
        drawsize = ((*obj).pdrawsize / xybin) as f32;
        step_res = sphere_res_for_size(state, drawsize);
        let list_index = gl.gen_lists(1);
        gl.new_list(list_index);
        gl.sphere(drawsize as f64, step_res * 2, step_res);
        gl.end_list();

        let cur_ob = if app.imod.is_null() {
            -1
        } else {
            (*app.imod).cindex.object
        };
        let cur_pt = if app.imod.is_null() {
            -1
        } else {
            (*app.imod).cindex.point
        };

        for co in 0..contsize {
            let cont: *mut Icont = &mut (&mut (*obj).cont)[co];
            gl.load_name(co as u32);
            let selected = imod_selection_list_query(&state.selection, cur_ob, co as i32) >= 0;
            cont_draw =
                imodv_check_contour_draw(state, app, &*cont, co as i32, check_time != 0, selected);
            if cont_draw == 0 {
                continue;
            }

            // Skip contour if not scattered and no sizes
            if iobj_scat((*obj).flags) == 0 && (*obj).pdrawsize == 0 && (*cont).sizes.is_empty() {
                continue;
            }

            let mut cont_props = DrawProps::default();
            let mut pt_props = DrawProps::default();
            let mut state_flags = 0;
            let mut change_flags = 0;
            let mut cursor = 0usize;
            let mut next_change = ifg_handle_cont_change(
                &*obj,
                co as i32,
                &mut cont_props,
                &mut pt_props,
                &mut state_flags,
                handle_flags,
                0,
                0,
                &state.values,
                gl,
            );
            if cont_props.gap != 0 {
                continue;
            }
            let mut pt: i32 = 0;
            if (if pt_props.trans != 0 { 1 } else { 0 }) != draw_trans {
                next_change = ifg_cont_trans_match(
                    &*obj,
                    &*cont,
                    &mut cursor,
                    &mut pt,
                    draw_trans,
                    &cont_props,
                    &mut pt_props,
                    &mut state_flags,
                    &mut change_flags,
                    handle_flags,
                    &state.values,
                    gl,
                );
                (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
            }

            // Set thicker line if this is the current contour, restore at end
            let thicker = imodv_check_thicker_contour(
                state,
                co as i32,
                imod_selection_list_query(&state.selection, state.obj_being_drawn, co as i32) > -2,
            );
            if thicker && app.read_pix_for_pick == 0 {
                gl.line_width((*obj).linewidth as i32 + 2, &*obj);
            }

            gl.push_name(NO_NAME);
            while (pt as usize) < (*cont).pts.len() {
                if next_change == pt {
                    next_change = ifg_handle_next_change(
                        &*obj,
                        &(*cont).store,
                        &mut cursor,
                        &cont_props,
                        &mut pt_props,
                        &mut state_flags,
                        &mut change_flags,
                        handle_flags,
                        0,
                        0,
                        &state.values,
                        gl,
                    );

                    // If trans state changes, seek point that restores it
                    if (if pt_props.trans != 0 { 1 } else { 0 }) != draw_trans {
                        next_change = ifg_cont_trans_match(
                            &*obj,
                            &*cont,
                            &mut cursor,
                            &mut pt,
                            draw_trans,
                            &cont_props,
                            &mut pt_props,
                            &mut state_flags,
                            &mut change_flags,
                            handle_flags,
                            &state.values,
                            gl,
                        );
                        (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                        if pt as usize >= (*cont).pts.len() {
                            break;
                        }
                    }
                }

                /* get the real point size, convert to number of pixels and
                look up step size based on current quality */
                drawsize = imod_point_get_size(&*obj, &*cont, pt);

                // Only draw zero-size points with scattered point objects
                if (iobj_scat((*obj).flags) == 0 && drawsize == 0.)
                    || (pt_props.gap != 0 && pt_props.valskip != 0)
                    || (cont_draw > 1 && pt != cur_pt)
                {
                    pt += 1;
                    continue;
                }

                gl.load_name(pt as u32);
                gl.push_matrix();
                let p = (&(*cont).pts)[pt as usize];
                gl.translate(p.x, p.y, p.z * z);

                if drawsize == (*obj).pdrawsize as f32 {
                    /* Use the display list if default size */
                    gl.call_list(list_index);
                } else {
                    drawsize /= xybin as f32;
                    step_res = sphere_res_for_size(state, drawsize);
                    gl.sphere(drawsize as f64, step_res * 2, step_res);
                }
                gl.pop_matrix();
                pt += 1;
            }
            gl.pop_name();
            if thicker && app.read_pix_for_pick == 0 {
                gl.line_width((*obj).linewidth as i32, &*obj);
            }
        }

        gl.delete_lists(list_index, 1);
        gl.pop_matrix();
        gl.pop_name();
    }
}
/// `sphereResForSize`.
pub fn sphere_res_for_size(state: &MvOglState, draw_size: f32) -> i32 {
    state.sphere_res
        [((draw_size * state.scale_sphere) as i32).clamp(0, (MAX_LOOKUP - 1) as i32) as usize]
        [state.quality_sphere.min(MAX_QUALITY - 1)]
}
/*****************************************************************************/
/*  Draw Mesh Data                                                           */
/*****************************************************************************/

/// Static `imodvDraw_mesh`.
///
/// # Safety
/// `mesh` and `obj` must point at a live mesh of the live object `obj`.
pub unsafe fn imodv_draw_mesh(
    state: &mut MvOglState,
    app: &ImodvApp,
    mesh: *mut Imesh,
    style: i32,
    obj: *mut Iobj,
    draw_trans: i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let poly_style: u32;
        let norm_style: u32;
        let mut def_props = DrawProps::default();
        let mut cur_props = DrawProps::default();
        let mut next_change: i32;
        let mut state_flags: i32;
        let mut change_flags = 0i32;
        let mut next_item_index: i32;
        let mut handle_flags = HANDLE_MESH_COLOR | HANDLE_TRANS;
        let skip_ends = if !((app.current_subset == SUBSET_SURF_ONLY
            || app.current_subset == SUBSET_SURF_OTHER)
            && state.cur_surf >= 0)
            || (*mesh).surf > 0
        {
            1
        } else {
            0
        };

        if mesh.is_null() || (*mesh).list.is_empty() {
            return;
        }

        if app.read_pix_for_pick == 0 {
            handle_flags |= HANDLE_3DWIDTH;
        }

        if ifg_setup_value_drawing(
            &*obj,
            GEN_STORE_MINMAX1,
            -1,
            &mut state.values,
            &mut crate::imod::three_dmod::xcramp::xcramp_mapfalsecolor,
        ) != 0
        {
            handle_flags |= HANDLE_VALUE1;
        }

        match style {
            DRAW_POINTS => {
                poly_style = GL_POINTS;
                norm_style = GL_POINTS;
            }
            DRAW_LINES => {
                poly_style = GL_LINE_STRIP;
                norm_style = GL_LINE_LOOP;
            }
            // The source leaves both uninitialised for any other style; it is
            // only ever called with DRAW_POINTS or DRAW_LINES.
            _ => {
                poly_style = GL_POINTS;
                norm_style = GL_POINTS;
            }
        }

        // The vertex-buffer branch cannot be reached here: it reads and writes
        // `mesh->vertBuf` and `obj->vertBufCont` (`vertexbuffer.h`), which the
        // translated `Imesh`/`Iobj` in `libimod` do not carry, so
        // `vbCleanupContVBD`, `vbCleanupVBD`, `analyzeMesh` and the
        // `glDrawElements` draw have nothing to act on.  `vbd` is therefore
        // always NULL, the source's own path before any buffer is built.

        state_flags = 0;
        ifg_handle_surf_change(
            &*obj,
            (*mesh).surf as i32,
            &mut def_props,
            &mut cur_props,
            &mut state_flags,
            handle_flags,
            &state.values,
            gl,
        );
        let def_trans = if def_props.trans != 0 { 1 } else { 0 };

        // First time in, if the trans state does not match the draw state, and
        // the storage list does not have a change to a matching state, return
        if draw_trans == 0 && def_trans != 0 && istore_trans_state_matches(&(*mesh).store, 0) == 0 {
            (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
            return;
        }

        state_flags = 0;
        let mut cursor = 0usize;
        next_change = istore_first_change_index(&(*mesh).store);
        next_item_index = next_change;

        let lsize = (*mesh).list.len() as i32;
        let vsize = (*mesh).vert.len() as i32;
        let mut i = 0i32;
        while i < lsize {
            let code = (&(*mesh).list)[i as usize];
            match code {
                IMOD_MESH_BGNPOLY | IMOD_MESH_BGNBIGPOLY => {
                    if skip_non_current_surface(state, app, mesh, &mut i, obj) != 0 {
                        i += 1;
                        continue;
                    }
                    gl.begin(poly_style);
                    i += 1;
                    while (&(*mesh).list)[i as usize] != IMOD_MESH_ENDPOLY {
                        let v = (&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize];
                        gl.vertex3f(v.x, v.y, v.z);
                        i += 1;
                    }
                    gl.end();
                }
                IMOD_MESH_BGNPOLYNORM => {
                    if skip_non_current_surface(state, app, mesh, &mut i, obj) != 0 {
                        i += 1;
                        continue;
                    }
                    i += 1;
                    while (&(*mesh).list)[i as usize] != IMOD_MESH_ENDPOLY {
                        gl.begin(norm_style);
                        i += 1;
                        let mut v = (&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize];
                        gl.vertex3f(v.x, v.y, v.z);
                        i += 2;
                        v = (&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize];
                        gl.vertex3f(v.x, v.y, v.z);
                        i += 2;
                        v = (&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize];
                        i += 1;
                        gl.vertex3f(v.x, v.y, v.z);
                        gl.end();
                    }
                }
                IMOD_MESH_BGNPOLYNORM2 => {
                    if skip_non_current_surface(state, app, mesh, &mut i, obj) != 0 {
                        next_change = istore_skip_to_index(&(*mesh).store, i);
                        next_item_index = next_change;
                        i += 1;
                        continue;
                    }
                    i += 1;

                    // Before starting loop, check if need to skip to matching trans state
                    if (if cur_props.trans != 0 { 1 } else { 0 }) != draw_trans
                        && (next_change < i || next_change > i + 2)
                    {
                        next_change = ifg_mesh_trans_match(
                            &*mesh,
                            &mut cursor,
                            def_trans,
                            draw_trans,
                            &mut i,
                            skip_ends,
                        );
                        next_item_index = next_change;
                        (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                    }
                    while (&(*mesh).list)[i as usize] != IMOD_MESH_ENDPOLY {
                        if next_change < i || next_change > i + 2 {
                            gl.begin(norm_style);

                            // This does not require Z scaling because it is in
                            // the transformation matrix
                            for _ in 0..3 {
                                let v = (&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize];
                                i += 1;
                                gl.vertex3f(v.x, v.y, v.z);
                            }
                            gl.end();
                        } else {
                            if state_flags != 0 || i == next_change {
                                next_change = ifg_handle_mesh_change(
                                    &*obj,
                                    &(*mesh).store,
                                    &mut cursor,
                                    &def_props,
                                    &mut cur_props,
                                    &mut next_item_index,
                                    i,
                                    &mut state_flags,
                                    &mut change_flags,
                                    handle_flags,
                                    &state.values,
                                    gl,
                                );
                                if (if cur_props.trans != 0 { 1 } else { 0 }) != draw_trans {
                                    if state_flags != 0 {
                                        ifg_handle_mesh_change(
                                            &*obj,
                                            &(*mesh).store,
                                            &mut cursor,
                                            &def_props,
                                            &mut cur_props,
                                            &mut next_item_index,
                                            0,
                                            &mut state_flags,
                                            &mut change_flags,
                                            handle_flags,
                                            &state.values,
                                            gl,
                                        );
                                    }
                                    next_change = ifg_mesh_trans_match(
                                        &*mesh,
                                        &mut cursor,
                                        def_trans,
                                        draw_trans,
                                        &mut i,
                                        skip_ends,
                                    );
                                    next_item_index = next_change;
                                    (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                                    continue;
                                }
                            }

                            if style == DRAW_POINTS {
                                // Points might as well be drawn singly
                                for j in 0..3 {
                                    if j != 0 && (state_flags != 0 || i == next_change) {
                                        next_change = ifg_handle_mesh_change(
                                            &*obj,
                                            &(*mesh).store,
                                            &mut cursor,
                                            &def_props,
                                            &mut cur_props,
                                            &mut next_item_index,
                                            i,
                                            &mut state_flags,
                                            &mut change_flags,
                                            handle_flags,
                                            &state.values,
                                            gl,
                                        );
                                    }
                                    gl.begin(norm_style);
                                    let v = (&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize];
                                    i += 1;
                                    gl.vertex3f(v.x, v.y, v.z);
                                    gl.end();
                                }
                            } else {
                                gl.begin(GL_LINE_STRIP);
                                let first_pt =
                                    (&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize];
                                i += 1;
                                gl.vertex3f(first_pt.x, first_pt.y, first_pt.z);
                                let first_red = cur_props.red;
                                let first_green = cur_props.green;
                                let first_blue = cur_props.blue;
                                let first_trans = cur_props.trans;

                                for _ in 0..2 {
                                    change_flags = 0;
                                    if state_flags != 0 || i == next_change {
                                        next_change = ifg_handle_mesh_change(
                                            &*obj,
                                            &(*mesh).store,
                                            &mut cursor,
                                            &def_props,
                                            &mut cur_props,
                                            &mut next_item_index,
                                            i,
                                            &mut state_flags,
                                            &mut change_flags,
                                            HANDLE_MESH_COLOR,
                                            &state.values,
                                            gl,
                                        );
                                    }
                                    let v = (&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize];
                                    i += 1;
                                    gl.vertex3f(v.x, v.y, v.z);
                                    if change_flags & CHANGED_3DWIDTH != 0
                                        && app.read_pix_for_pick == 0
                                    {
                                        gl.end();
                                        gl.line_width(cur_props.linewidth, &*obj);
                                        gl.begin(GL_LINE_STRIP);
                                        let v = (&(*mesh).vert)
                                            [(&(*mesh).list)[(i - 1) as usize] as usize];
                                        gl.vertex3f(v.x, v.y, v.z);
                                    }
                                }

                                // Reset color to first point
                                if first_red != cur_props.red
                                    || first_green != cur_props.green
                                    || first_blue != cur_props.blue
                                    || first_trans != cur_props.trans
                                {
                                    ifg_handle_color_trans(
                                        &*obj,
                                        first_red,
                                        first_green,
                                        first_blue,
                                        first_trans,
                                        gl,
                                    );
                                }
                                gl.vertex3f(first_pt.x, first_pt.y, first_pt.z);
                                gl.end();

                                // If reset color, better set it back to match curprops
                                if first_red != cur_props.red
                                    || first_green != cur_props.green
                                    || first_blue != cur_props.blue
                                    || first_trans != cur_props.trans
                                {
                                    ifg_handle_color_trans(
                                        &*obj,
                                        cur_props.red,
                                        cur_props.green,
                                        cur_props.blue,
                                        cur_props.trans,
                                        gl,
                                    );
                                }
                            }

                            // Reset if not in default state and the next positive
                            // change will not be in the next triangle
                            if state_flags != 0
                                && (next_item_index < i
                                    || next_item_index > i + 2
                                    || (&(*mesh).list)[i as usize] == IMOD_MESH_ENDPOLY)
                            {
                                next_change = ifg_handle_mesh_change(
                                    &*obj,
                                    &(*mesh).store,
                                    &mut cursor,
                                    &def_props,
                                    &mut cur_props,
                                    &mut next_item_index,
                                    i,
                                    &mut state_flags,
                                    &mut change_flags,
                                    handle_flags,
                                    &state.values,
                                    gl,
                                );
                                if (&(*mesh).list)[i as usize] != IMOD_MESH_ENDPOLY
                                    && (if cur_props.trans != 0 { 1 } else { 0 }) != draw_trans
                                {
                                    next_change = ifg_mesh_trans_match(
                                        &*mesh,
                                        &mut cursor,
                                        def_trans,
                                        draw_trans,
                                        &mut i,
                                        skip_ends,
                                    );
                                    next_item_index = next_change;
                                    (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                                }
                            }
                        }
                    }
                }
                IMOD_MESH_BGNTRI | IMOD_MESH_ENDTRI | IMOD_MESH_SWAP => {}
                IMOD_MESH_NORMAL => {
                    i += 1;
                }
                IMOD_MESH_END => return,
                _ => {
                    if code < vsize && code > -1 {
                        let v = (&(*mesh).vert)[code as usize];
                        gl.vertex3f(v.x, v.y, v.z);
                    }
                }
            }
            i += 1;
        }
    }
}
/// Static `imodvDraw_filled_mesh`: draws mesh with lighting model.
///
/// # Safety
/// `mesh` and `obj` must point at a live mesh of the live object `obj`.
pub unsafe fn imodv_draw_filled_mesh(
    state: &mut MvOglState,
    app: &ImodvApp,
    mesh: *mut Imesh,
    zscale: f64,
    obj: *mut Iobj,
    draw_trans: i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let z: f32 = zscale as f32;
        let mut def_props = DrawProps::default();
        let mut cur_props = DrawProps::default();
        let mut next_change: i32;
        let mut state_flags: i32;
        let mut change_flags = 0i32;
        let mut next_item_index: i32;
        let mut handle_flags = (if (*obj).flags & IMOD_OBJFLAG_FCOLOR != 0 {
            HANDLE_MESH_FCOLOR
        } else {
            HANDLE_MESH_COLOR
        }) | HANDLE_TRANS;
        // Skipends is 0 if current surface only is being drawn and it is not
        // done with surface numbers in mesh
        let skip_ends = if !((app.current_subset == SUBSET_SURF_ONLY
            || app.current_subset == SUBSET_SURF_OTHER)
            && state.cur_surf >= 0)
            || (*mesh).surf > 0
        {
            1
        } else {
            0
        };

        if mesh.is_null() || (*mesh).list.is_empty() {
            return;
        }

        if ifg_setup_value_drawing(
            &*obj,
            GEN_STORE_MINMAX1,
            -1,
            &mut state.values,
            &mut crate::imod::three_dmod::xcramp::xcramp_mapfalsecolor,
        ) != 0
        {
            handle_flags |= HANDLE_VALUE1;
        }

        // The vertex-buffer branch cannot be reached here for the same reason
        // as in `imodvDraw_mesh`: `mesh->vertBuf` is not a field of the
        // translated `Imesh`.

        /* Check to see if normals have magnitudes. */
        if (*mesh).flag & IMESH_FLAG_NMAG != 0 {
            gl.normalize(false);
        } else {
            gl.normalize(true);
        }

        state_flags = 0;
        ifg_handle_surf_change(
            &*obj,
            (*mesh).surf as i32,
            &mut def_props,
            &mut cur_props,
            &mut state_flags,
            handle_flags,
            &state.values,
            gl,
        );
        let def_trans = if def_props.trans != 0 { 1 } else { 0 };

        // First time in, if the trans state does not match the draw state, and
        // the storage list does not have a change to a matching state, return
        if draw_trans == 0 && def_trans != 0 && istore_trans_state_matches(&(*mesh).store, 0) == 0 {
            (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
            return;
        }

        state_flags = 0;
        let mut cursor = 0usize;
        next_change = istore_first_change_index(&(*mesh).store);
        next_item_index = next_change;

        let lsize = (*mesh).list.len() as i32;
        let vsize = (*mesh).vert.len() as i32;
        let mut i = 0i32;
        while i < lsize {
            let code = (&(*mesh).list)[i as usize];
            match code {
                IMOD_MESH_BGNTRI => gl.begin(GL_TRIANGLE_STRIP),
                IMOD_MESH_ENDTRI => gl.end(),
                IMOD_MESH_BGNPOLY => {
                    if skip_non_current_surface(state, app, mesh, &mut i, obj) != 0 {
                        i += 1;
                        continue;
                    }
                    gl.begin(GL_POLYGON);
                }
                IMOD_MESH_NORMAL => {
                    i += 1;
                    let ind = (&(*mesh).list)[i as usize];
                    if ind < vsize && ind > -1 {
                        let v = (&(*mesh).vert)[ind as usize];
                        gl.normal3f(v.x, v.y, v.z);
                    }
                }
                IMOD_MESH_BGNPOLYNORM => {
                    /* 6/19/01 note: using glVertex3fv with no z scaling increases
                    speed by 5% on PC, 0.2% on SGI */
                    if skip_non_current_surface(state, app, mesh, &mut i, obj) != 0 {
                        i += 1;
                        continue;
                    }
                    gl.begin(GL_TRIANGLES);
                    i += 1;
                    while (&(*mesh).list)[i as usize] != IMOD_MESH_ENDPOLY {
                        for _ in 0..3 {
                            let n = (&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize];
                            i += 1;
                            gl.normal3f(n.x, n.y, n.z);
                            let li = (&(*mesh).list)[i as usize] as usize;
                            i += 1;
                            let v = (&(*mesh).vert)[li];
                            gl.vertex3f(v.x, v.y, v.z * z);
                        }
                        if (i % 512) == 0 {
                            gl.finish();
                        }
                    }
                    gl.end();
                }
                IMOD_MESH_BGNPOLYNORM2 => {
                    if skip_non_current_surface(state, app, mesh, &mut i, obj) != 0 {
                        next_change = istore_skip_to_index(&(*mesh).store, i);
                        next_item_index = next_change;
                        i += 1;
                        continue;
                    }
                    gl.begin(GL_TRIANGLES);
                    i += 1;

                    // Before starting loop, check if need to skip to matching trans state
                    if (if cur_props.trans != 0 { 1 } else { 0 }) != draw_trans
                        && (next_change < i || next_change > i + 2)
                    {
                        next_change = ifg_mesh_trans_match(
                            &*mesh,
                            &mut cursor,
                            def_trans,
                            draw_trans,
                            &mut i,
                            skip_ends,
                        );
                        next_item_index = next_change;
                        (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                    }

                    while (&(*mesh).list)[i as usize] != IMOD_MESH_ENDPOLY {
                        if next_change < i || next_change > i + 2 {
                            for _ in 0..3 {
                                let li = (&(*mesh).list)[i as usize] as usize;
                                i += 1;
                                let n = (&(*mesh).vert)[li + 1];
                                gl.normal3f(n.x, n.y, n.z);
                                let v = (&(*mesh).vert)[li];
                                gl.vertex3f(v.x, v.y, v.z * z);
                            }
                        } else {
                            // Isolate a triangle with changes from other triangles
                            // with an End/begin pair regardless of type of change
                            gl.end();

                            // Get the next change for the first point
                            if state_flags != 0 || i == next_change {
                                next_change = ifg_handle_mesh_change(
                                    &*obj,
                                    &(*mesh).store,
                                    &mut cursor,
                                    &def_props,
                                    &mut cur_props,
                                    &mut next_item_index,
                                    i,
                                    &mut state_flags,
                                    &mut change_flags,
                                    handle_flags,
                                    &state.values,
                                    gl,
                                );

                                // If trans state does not match draw state, return
                                // to default and skip to next matching triangle
                                if (if cur_props.trans != 0 { 1 } else { 0 }) != draw_trans {
                                    if state_flags != 0 {
                                        ifg_handle_mesh_change(
                                            &*obj,
                                            &(*mesh).store,
                                            &mut cursor,
                                            &def_props,
                                            &mut cur_props,
                                            &mut next_item_index,
                                            0,
                                            &mut state_flags,
                                            &mut change_flags,
                                            handle_flags,
                                            &state.values,
                                            gl,
                                        );
                                    }
                                    next_change = ifg_mesh_trans_match(
                                        &*mesh,
                                        &mut cursor,
                                        def_trans,
                                        draw_trans,
                                        &mut i,
                                        skip_ends,
                                    );
                                    next_item_index = next_change;
                                    (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                                    gl.begin(GL_TRIANGLES);
                                    continue;
                                }
                            }

                            gl.begin(GL_TRIANGLES);
                            for j in 0..3 {
                                if j != 0 && (state_flags != 0 || i == next_change) {
                                    next_change = ifg_handle_mesh_change(
                                        &*obj,
                                        &(*mesh).store,
                                        &mut cursor,
                                        &def_props,
                                        &mut cur_props,
                                        &mut next_item_index,
                                        i,
                                        &mut state_flags,
                                        &mut change_flags,
                                        handle_flags,
                                        &state.values,
                                        gl,
                                    );
                                }
                                let li = (&(*mesh).list)[i as usize] as usize;
                                i += 1;
                                let n = (&(*mesh).vert)[li + 1];
                                gl.normal3f(n.x, n.y, n.z);
                                let v = (&(*mesh).vert)[li];
                                gl.vertex3f(v.x, v.y, v.z * z);
                            }

                            // Again, isolate this triangle from further ones.
                            gl.end();

                            // Reset if not in default state and the next positive
                            // change will not be in the next triangle
                            if state_flags != 0
                                && (next_item_index < i
                                    || next_item_index > i + 2
                                    || (&(*mesh).list)[i as usize] == IMOD_MESH_ENDPOLY)
                            {
                                next_change = ifg_handle_mesh_change(
                                    &*obj,
                                    &(*mesh).store,
                                    &mut cursor,
                                    &def_props,
                                    &mut cur_props,
                                    &mut next_item_index,
                                    i,
                                    &mut state_flags,
                                    &mut change_flags,
                                    handle_flags,
                                    &state.values,
                                    gl,
                                );
                                if (&(*mesh).list)[i as usize] != IMOD_MESH_ENDPOLY
                                    && (if cur_props.trans != 0 { 1 } else { 0 }) != draw_trans
                                {
                                    next_change = ifg_mesh_trans_match(
                                        &*mesh,
                                        &mut cursor,
                                        def_trans,
                                        draw_trans,
                                        &mut i,
                                        skip_ends,
                                    );
                                    next_item_index = next_change;
                                    (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                                }
                            }
                            gl.begin(GL_TRIANGLES);
                        }
                    }
                    gl.end();
                }
                IMOD_MESH_BGNBIGPOLY => {
                    if skip_non_current_surface(state, app, mesh, &mut i, obj) != 0 {
                        i += 1;
                        continue;
                    }
                    let mut verts: Vec<Ipoint> = Vec::new();
                    gl.push_matrix();
                    gl.scale(1.0f32, 1.0f32, z);
                    i += 1;
                    while (&(*mesh).list)[i as usize] != IMOD_MESH_ENDPOLY {
                        verts.push((&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize]);
                        i += 1;
                    }
                    gl.tess_polygon(&verts);
                    gl.pop_matrix();
                }
                IMOD_MESH_END => return,
                IMOD_MESH_SWAP => {
                    eprintln!("imodlib: old mesh");
                    return;
                }
                _ => {
                    if code < vsize && code > -1 {
                        let v = (&(*mesh).vert)[code as usize];
                        gl.vertex3f(v.x, v.y, v.z * z);
                    }
                }
            }
            i += 1;
        }
    }
}

/*
 * SCALAR MESH DRAWING ROUTINES
 */

/// Static `imodvDrawScalarMesh`.
///
/// # Safety
/// `mesh` and `obj` must point at a live mesh of the live object `obj`.
pub unsafe fn imodv_draw_scalar_mesh(
    state: &mut MvOglState,
    app: &ImodvApp,
    mesh: *mut Imesh,
    mut zscale: f64,
    obj: *mut Iobj,
    draw_trans: i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let z: f32 = zscale as f32;
        let trans: i32 = (2.55f32 * (100.0f32 - (*obj).trans as f32)) as i32;
        let mut list_inc = 0i32;
        let mut vert_base = 0i32;
        let mut norm_add = 0i32;
        let use_light = app.lighting;
        let poly_style: u32;

        if mesh.is_null() || (*mesh).list.is_empty() {
            return;
        }

        // Skip drawing if trans state does not match draw state
        if (if (*obj).trans != 0 { 1 } else { 0 }) != draw_trans {
            (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
            return;
        }

        if iobj_fill((*obj).flags) != 0 {
            poly_style = GL_POLYGON;
        } else if iobj_line((*obj).flags) != 0 {
            poly_style = GL_LINE_STRIP;
            zscale = 1.0;
        } else {
            poly_style = GL_POINTS;
            zscale = 1.0;
        }

        ifg_make_value_map(
            &*obj,
            &mut state.values,
            &mut crate::imod::three_dmod::xcramp::xcramp_mapfalsecolor,
        );
        let cmap = state.values.value_cmap;

        /*
         * Loop through mesh data and draw it.
         */
        let lsize = (*mesh).list.len() as i32;
        let vsize = (*mesh).vert.len() as i32;
        let mut i = 0i32;
        while i < lsize {
            let code = (&(*mesh).list)[i as usize];
            match code {
                IMOD_MESH_BGNTRI => gl.begin(GL_TRIANGLE_STRIP),
                IMOD_MESH_ENDTRI => gl.end(),
                IMOD_MESH_BGNPOLY => {
                    if skip_non_current_surface(state, app, mesh, &mut i, obj) != 0 {
                        i += 1;
                        continue;
                    }
                    gl.begin(GL_POLYGON);
                }
                IMOD_MESH_NORMAL => {
                    i += 1;
                    let ind = (&(*mesh).list)[i as usize];
                    if ind < vsize && ind > -1 {
                        let v = (&(*mesh).vert)[ind as usize];
                        gl.normal3f(v.x, v.y, v.z);
                    }
                }
                IMOD_MESH_BGNPOLYNORM | IMOD_MESH_BGNPOLYNORM2 => {
                    if skip_non_current_surface(state, app, mesh, &mut i, obj) != 0 {
                        i += 1;
                        continue;
                    }
                    imod_mesh_poly_norm_factors(
                        (&(*mesh).list)[i as usize],
                        &mut list_inc,
                        &mut vert_base,
                        &mut norm_add,
                    );
                    i += 1;
                    while (&(*mesh).list)[i as usize] != IMOD_MESH_ENDPOLY {
                        gl.begin(poly_style);

                        for _ in 0..3 {
                            let n =
                                (&(*mesh).vert)[((&(*mesh).list)[i as usize] + norm_add) as usize];
                            // `mag` is a float: the double product of 255.0
                            // and sqrt() is narrowed before the cast to byte.
                            let mag: f32 = (255.0f64
                                * (((n.x * n.x) + (n.y * n.y) + (n.z * n.z)) as f64).sqrt())
                                as f32;
                            let luv = mag as u8;

                            if use_light != 0 {
                                gl.light_adjust(
                                    &*obj,
                                    cmap[0][luv as usize] as f32 / 255.0f32,
                                    cmap[1][luv as usize] as f32 / 255.0f32,
                                    cmap[2][luv as usize] as f32 / 255.0f32,
                                    (*obj).trans as i32,
                                );
                            }

                            gl.color4ub(
                                cmap[0][luv as usize],
                                cmap[1][luv as usize],
                                cmap[2][luv as usize],
                                trans as u8,
                            );
                            gl.normal3f(n.x, n.y, n.z);
                            let v =
                                (&(*mesh).vert)[(&(*mesh).list)[(i + vert_base) as usize] as usize];
                            // `zscale` is a double here, so this product is
                            // double and only `glVertex3f` narrows it.
                            gl.vertex3f(v.x, v.y, (v.z as f64 * zscale) as f32);

                            i += list_inc;
                        }

                        gl.end();
                    }
                }
                IMOD_MESH_BGNBIGPOLY => {
                    if skip_non_current_surface(state, app, mesh, &mut i, obj) != 0 {
                        i += 1;
                        continue;
                    }
                    let mut verts: Vec<Ipoint> = Vec::new();
                    gl.push_matrix();
                    gl.scale(1.0f32, 1.0f32, z);
                    i += 1;
                    while (&(*mesh).list)[i as usize] != IMOD_MESH_ENDPOLY {
                        verts.push((&(*mesh).vert)[(&(*mesh).list)[i as usize] as usize]);
                        i += 1;
                    }
                    gl.tess_polygon(&verts);
                    gl.pop_matrix();
                }
                IMOD_MESH_END => return,
                IMOD_MESH_SWAP => {
                    eprintln!("imodlib: old mesh");
                    return;
                }
                _ => {
                    if code < vsize && code > -1 {
                        let v = (&(*mesh).vert)[code as usize];
                        gl.vertex3f(v.x, v.y, v.z * z);
                    }
                }
            }
            i += 1;
        }
    }
}

/// Static `skipNonCurrentSurface`: check if surface subset and if so, see if
/// mesh matches current surface.
///
/// # Safety
/// `mesh` and `obj` must point at a live mesh of the live object `obj`.
pub unsafe fn skip_non_current_surface(
    state: &MvOglState,
    app: &ImodvApp,
    mesh: *mut Imesh,
    ip: &mut i32,
    obj: *mut Iobj,
) -> i32 {
    unsafe {
        let lim_test = 9;

        // Test if surface subset on, it's also OK if the mesh surface is greater
        // than zero because a match was already tested for in checkMeshDraw
        if !((app.current_subset == SUBSET_SURF_ONLY || app.current_subset == SUBSET_SURF_OTHER)
            && state.cur_surf >= 0)
            || (*mesh).surf > 0
        {
            return 0;
        }

        // Set up indexes, offset and interval to check, ending code
        let mut i = *ip + 1;
        let mut list_skip = 1;
        let mut vert_offset = 0;
        let mut end_code = IMOD_MESH_ENDPOLY;

        if (&(*mesh).list)[(i - 1) as usize] == IMOD_MESH_BGNPOLYNORM {
            list_skip = 2;
            vert_offset = 1;
        }
        if (&(*mesh).list)[(i - 1) as usize] == IMOD_MESH_BGNPOLY {
            end_code = IMOD_MESH_END;
        }

        // Test up to the given limit of vertices in the mesh
        let mut num_test = 0;
        while (&(*mesh).list)[i as usize] != end_code && num_test < lim_test {
            let ind = (&(*mesh).list)[(i + vert_offset) as usize];
            let xx = (&(*mesh).vert)[ind as usize].x;
            let yy = (&(*mesh).vert)[ind as usize].y;
            let zz = (&(*mesh).vert)[ind as usize].z;
            for co in 0..(*obj).cont.len() {
                let cont = &(&(*obj).cont)[co];

                // If contour is not wild and Z is different, or if surface doesn't
                // match then skip the contour
                if cont.pts.is_empty() || (cont.flags & ICONT_WILD == 0 && zz != cont.pts[0].z) {
                    continue;
                }
                if cont.surf != state.cur_surf {
                    continue;
                }

                // Return upon an exact match
                for pt in 0..cont.pts.len() {
                    if cont.pts[pt].x == xx && cont.pts[pt].y == yy && cont.pts[pt].z == zz {
                        return 0;
                    }
                }
            }
            num_test += 1;
            i += list_skip;
        }

        // Most efficient to loop to the end code here
        while (&(*mesh).list)[*ip as usize] != end_code {
            *ip += 1;
        }

        1
    }
}
/****************************************************************************/
/* DRAW CONTOURS                                                            */

/// Static `imodvPick_Contours`.
///
/// # Safety
/// `obj` must point at a live object of the model being drawn.
pub unsafe fn imodv_pick_contours(
    state: &mut MvOglState,
    app: &ImodvApp,
    obj: *mut Iobj,
    zscale: f64,
    draw_trans: i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let mut npt: i32 = 0;
        let mut pmode = GL_POINTS;
        let mut do_lines = 0;
        let mut has_poly_norm2: bool;
        let mut cont_props = DrawProps::default();
        let mut pt_props = DrawProps::default();
        let mut state_flags = 0i32;
        let mut change_flags = 0i32;
        let mut handle_flags = HANDLE_MESH_COLOR | HANDLE_3DWIDTH;
        let mut check_time = iobj_time((*obj).flags) as i32;
        if state.ctime == 0 {
            check_time = 0;
        }

        if ifg_setup_value_drawing(
            &*obj,
            GEN_STORE_MINMAX1,
            -1,
            &mut state.values,
            &mut crate::imod::three_dmod::xcramp::xcramp_mapfalsecolor,
        ) != 0
        {
            handle_flags |= HANDLE_VALUE1;
        }

        // Skip drawing if trans state does not match draw state
        if (if (*obj).trans != 0 { 1 } else { 0 }) != draw_trans {
            (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
            return;
        }

        // Make sure there is a polynorm2 mesh before doing mesh drawing, so it
        // can fall back to contour drawing for old meshes
        has_poly_norm2 = false;
        let mut co = 0;
        while co < (*obj).mesh.len() && !has_poly_norm2 {
            let mesh = &(&(*obj).mesh)[co];
            for i in 0..mesh.list.len() {
                if mesh.list[i] == IMOD_MESH_BGNPOLYNORM2 {
                    has_poly_norm2 = true;
                    break;
                }
            }
            co += 1;
        }

        // If there is mesh drawing, draw the vertex points or the triangles
        gl.push_name(NO_NAME);
        if iobj_mesh((*obj).flags) != 0 && has_poly_norm2 {
            for co in 0..(*obj).mesh.len() {
                let mesh: *const Imesh = &(&(*obj).mesh)[co];
                if (*mesh).list.is_empty() || (*mesh).vert.is_empty() {
                    continue;
                }

                // Load the name as a number past the last contour
                gl.load_name((co + 1 + (*obj).cont.len()) as u32);
                gl.push_name(NO_NAME);

                // For ordinary meshes, better draw all the lines
                if !(*obj).cont.is_empty() {
                    // LINE/FILL give same time and seem to perform same on big triangles
                    gl.polygon_mode_line(false);
                    let mut i = 0usize;
                    while i < (*mesh).list.len() {
                        if (&(*mesh).list)[i] == IMOD_MESH_BGNPOLYNORM2 {
                            i += 1;
                            while (&(*mesh).list)[i] != IMOD_MESH_ENDPOLY {
                                let mut li = (&(*mesh).list)[i] as usize;
                                i += 1;

                                // The load name must occur outside begin-end sequence
                                gl.load_name(li as u32);
                                gl.begin(GL_TRIANGLES);
                                let v = (&(*mesh).vert)[li];
                                gl.vertex3f(v.x, v.y, v.z);
                                li = (&(*mesh).list)[i] as usize;
                                i += 1;
                                let v = (&(*mesh).vert)[li];
                                gl.vertex3f(v.x, v.y, v.z);
                                li = (&(*mesh).list)[i] as usize;
                                i += 1;
                                let v = (&(*mesh).vert)[li];
                                gl.vertex3f(v.x, v.y, v.z);
                                gl.end();
                            }
                        }
                        i += 1;
                    }
                } else {
                    // This is 5 times faster!  So do it for isosurfaces (no contours)
                    let mut li = 0usize;
                    while li < (*mesh).vert.len() {
                        gl.load_name(li as u32);
                        gl.begin(GL_POINTS);
                        let v = (&(*mesh).vert)[li];
                        gl.vertex3f(v.x, v.y, v.z);
                        gl.end();
                        li += 2;
                    }
                }
                gl.pop_name();
            }
            gl.pop_name();
            return;
        }

        if iobj_line((*obj).flags) != 0 {
            pmode = GL_LINES;
            do_lines = 1;
        }

        let cur_ob = if app.imod.is_null() {
            -1
        } else {
            (*app.imod).cindex.object
        };

        for co in 0..(*obj).cont.len() {
            let cont: *const Icont = &(&(*obj).cont)[co];
            let selected = imod_selection_list_query(&state.selection, cur_ob, co as i32) >= 0;
            if imodv_check_contour_draw(state, app, &*cont, co as i32, check_time != 0, selected)
                == 0
            {
                continue;
            }

            let mut cursor = 0usize;
            let mut next_change = ifg_handle_cont_change(
                &*obj,
                co as i32,
                &mut cont_props,
                &mut pt_props,
                &mut state_flags,
                handle_flags,
                0,
                0,
                &state.values,
                gl,
            );
            if cont_props.gap != 0 {
                continue;
            }

            gl.load_name(co as u32);

            gl.push_name(NO_NAME);
            let mut pt = 0i32;
            while (pt as usize) < (*cont).pts.len() {
                pt_props.gap = 0;
                if next_change == pt {
                    next_change = ifg_handle_next_change(
                        &*obj,
                        &(*cont).store,
                        &mut cursor,
                        &cont_props,
                        &mut pt_props,
                        &mut state_flags,
                        &mut change_flags,
                        handle_flags,
                        0,
                        0,
                        &state.values,
                        gl,
                    );
                }
                if do_lines != 0 {
                    if pt_props.gap != 0 {
                        pt += 1;
                        continue;
                    }
                    npt = pt + 1;
                    if npt as usize == (*cont).pts.len() {
                        if iobj_close((*obj).flags) == 0 || (*cont).flags & ICONT_OPEN != 0 {
                            break;
                        }
                        npt = 0;
                    }
                } else if pt_props.gap != 0 && pt_props.valskip != 0 {
                    pt += 1;
                    continue;
                }
                gl.load_name(pt as u32);
                gl.begin(pmode);
                let p = (&(*cont).pts)[pt as usize];
                gl.vertex3f(p.x, p.y, p.z);
                if do_lines != 0 {
                    let p = (&(*cont).pts)[npt as usize];
                    gl.vertex3f(p.x, p.y, p.z);
                }
                gl.end();
                pt += 1;
            }
            gl.pop_name();
        }

        gl.pop_name();
    }
}

/// Static `imodvDraw_contours`: draw lines or points in the contours of an object.
///
/// # Safety
/// `obj` must point at a live object of the model being drawn.
pub unsafe fn imodv_draw_contours(
    state: &mut MvOglState,
    app: &ImodvApp,
    obj: *mut Iobj,
    mode: u32,
    draw_trans: i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let mut thick_add = 0i32;
        let mut check_time = iobj_time((*obj).flags) as i32;
        let mut cont_props = DrawProps::default();
        let mut pt_props = DrawProps::default();
        let mut state_flags = 0i32;
        let mut change_flags = 0i32;
        let mut change_flags2 = 0i32;
        let mut handle_co_flgs = HANDLE_MESH_COLOR | HANDLE_TRANS;
        let mut handle_pt_flgs = HANDLE_TRANS;
        if state.ctime == 0 {
            check_time = 0;
        }

        if ifg_setup_value_drawing(
            &*obj,
            GEN_STORE_MINMAX1,
            -1,
            &mut state.values,
            &mut crate::imod::three_dmod::xcramp::xcramp_mapfalsecolor,
        ) != 0
        {
            handle_co_flgs |= HANDLE_VALUE1;
            handle_pt_flgs |= HANDLE_VALUE1;
        }

        // The vertex-buffer branch cannot be reached here: it reads and writes
        // `obj->vertBufCont` (`vertexbuffer.h`), which the translated `Iobj` in
        // `libimod` does not carry, so `vbCleanupMeshVBD`, `vbCleanupContVBD`,
        // `analyzeConts` and the `glDrawElements` draw have nothing to act on.
        // `vbd` is therefore always NULL, the source's own path before any
        // buffer is built.

        // First time in, if object has transparency, then check whether any
        // contour or point stores set transparency to 0 and if not, skip
        if draw_trans == 0 && (*obj).trans != 0 && istore_trans_state_matches(&(*obj).store, 0) == 0
        {
            let mut co = 0;
            while co < (*obj).cont.len() {
                if istore_trans_state_matches(&(&(*obj).cont)[co].store, 0) != 0 {
                    break;
                }
                co += 1;
            }
            if co >= (*obj).cont.len() {
                (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                return;
            }
        }

        let draw_stipple = if app.standalone != 0 {
            0
        } else {
            let vi = app.vi as *const ImodView;
            if vi.is_null() { 0 } else { (*vi).draw_stipple }
        };
        let cur_ob = if app.imod.is_null() {
            -1
        } else {
            (*app.imod).cindex.object
        };

        for co in 0..(*obj).cont.len() {
            let cont: *const Icont = &(&(*obj).cont)[co];
            let selected = imod_selection_list_query(&state.selection, cur_ob, co as i32) >= 0;
            if imodv_check_contour_draw(state, app, &*cont, co as i32, check_time != 0, selected)
                == 0
            {
                continue;
            }

            let mut cursor = 0usize;
            let mut next_change = ifg_handle_cont_change(
                &*obj,
                co as i32,
                &mut cont_props,
                &mut pt_props,
                &mut state_flags,
                handle_co_flgs,
                0,
                0,
                &state.values,
                gl,
            );
            if cont_props.gap != 0 {
                continue;
            }

            // Set thicker line if this is the current contour
            thick_add = if imodv_check_thicker_contour(
                state,
                co as i32,
                imod_selection_list_query(&state.selection, state.obj_being_drawn, co as i32) > -2,
            ) {
                2
            } else {
                0
            };

            let mut pt = 0i32;
            if (if pt_props.trans != 0 { 1 } else { 0 }) != draw_trans {
                next_change = ifg_cont_trans_match(
                    &*obj,
                    &*cont,
                    &mut cursor,
                    &mut pt,
                    draw_trans,
                    &cont_props,
                    &mut pt_props,
                    &mut state_flags,
                    &mut change_flags,
                    handle_co_flgs,
                    &state.values,
                    gl,
                );
                (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                if pt_props.gap != 0 {
                    pt += 1;
                }
                if pt as usize >= (*cont).pts.len() {
                    pt_props.gap = 1;
                }
            }

            if mode == GL_POINTS {
                // Set up to do points
                gl.point_size(
                    if app.read_pix_for_pick != 0 {
                        1
                    } else {
                        pt_props.linewidth + thick_add
                    },
                    &*obj,
                );
                gl.begin(GL_POINTS);
                while (pt as usize) < (*cont).pts.len() {
                    pt_props.gap = 0;

                    // For points, implement change before point is drawn
                    if next_change == pt {
                        next_change = ifg_handle_next_change(
                            &*obj,
                            &(*cont).store,
                            &mut cursor,
                            &cont_props,
                            &mut pt_props,
                            &mut state_flags,
                            &mut change_flags,
                            handle_pt_flgs,
                            0,
                            0,
                            &state.values,
                            gl,
                        );

                        // If trans state changes, seek point that restores it
                        if (if pt_props.trans != 0 { 1 } else { 0 }) != draw_trans {
                            gl.end();
                            next_change = ifg_cont_trans_match(
                                &*obj,
                                &*cont,
                                &mut cursor,
                                &mut pt,
                                draw_trans,
                                &cont_props,
                                &mut pt_props,
                                &mut state_flags,
                                &mut change_flags,
                                handle_co_flgs,
                                &state.values,
                                gl,
                            );
                            gl.begin(GL_POINTS);
                            (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                            if pt as usize >= (*cont).pts.len() {
                                break;
                            }
                        }

                        if change_flags & CHANGED_3DWIDTH != 0 && app.read_pix_for_pick == 0 {
                            gl.end();
                            gl.point_size(pt_props.linewidth + thick_add, &*obj);
                            gl.begin(GL_POINTS);
                        }
                        if change_flags & CHANGED_COLOR != 0 {
                            gl.end();
                            ifg_handle_color_trans(
                                &*obj,
                                pt_props.red,
                                pt_props.green,
                                pt_props.blue,
                                pt_props.trans,
                                gl,
                            );
                            gl.begin(GL_POINTS);
                        }
                    }
                    if pt_props.gap == 0 || pt_props.valskip == 0 {
                        let p = (&(*cont).pts)[pt as usize];
                        gl.vertex3f(p.x, p.y, p.z);
                    }
                    pt += 1;
                }
                gl.end();
            } else {
                // Set up to do lines
                if app.standalone == 0 {
                    gl.enable_stipple(draw_stipple, &*cont);
                }
                gl.line_width(
                    if app.read_pix_for_pick != 0 {
                        1
                    } else {
                        pt_props.linewidth + thick_add
                    },
                    &*obj,
                );
                gl.begin(GL_LINE_STRIP);
                while (pt as usize) < (*cont).pts.len() {
                    // Get change at point then add point so color changes during line
                    pt_props.gap = 0;
                    if next_change == pt {
                        next_change = ifg_handle_next_change(
                            &*obj,
                            &(*cont).store,
                            &mut cursor,
                            &cont_props,
                            &mut pt_props,
                            &mut state_flags,
                            &mut change_flags,
                            handle_pt_flgs,
                            0,
                            0,
                            &state.values,
                            gl,
                        );
                        let p = (&(*cont).pts)[pt as usize];
                        gl.vertex3f(p.x, p.y, p.z);

                        // Skip ahead if trans state changed
                        if (if pt_props.trans != 0 { 1 } else { 0 }) != draw_trans {
                            gl.end();
                            next_change = ifg_cont_trans_match(
                                &*obj,
                                &*cont,
                                &mut cursor,
                                &mut pt,
                                draw_trans,
                                &cont_props,
                                &mut pt_props,
                                &mut state_flags,
                                &mut change_flags2,
                                handle_pt_flgs,
                                &state.values,
                                gl,
                            );
                            (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;

                            // Set gap if skipping to end so connector is not drawn
                            if pt as usize >= (*cont).pts.len() {
                                pt_props.gap = 1;
                                break;
                            }
                            if (change_flags | change_flags2) & CHANGED_3DWIDTH != 0 {
                                gl.line_width(pt_props.linewidth + thick_add, &*obj);
                            }
                            if (change_flags | change_flags2) & CHANGED_COLOR != 0 {
                                ifg_handle_color_trans(
                                    &*obj,
                                    pt_props.red,
                                    pt_props.green,
                                    pt_props.blue,
                                    pt_props.trans,
                                    gl,
                                );
                            }
                            gl.begin(GL_LINE_STRIP);
                        } else {
                            if change_flags & CHANGED_3DWIDTH != 0 && app.read_pix_for_pick == 0 {
                                // Width change requires ending the strip and restarting it
                                gl.end();
                                gl.line_width(pt_props.linewidth + thick_add, &*obj);
                                gl.begin(GL_LINE_STRIP);
                            }

                            // So do color changes on nvidia/Linux
                            if change_flags & CHANGED_COLOR != 0 {
                                gl.end();
                                ifg_handle_color_trans(
                                    &*obj,
                                    pt_props.red,
                                    pt_props.green,
                                    pt_props.blue,
                                    pt_props.trans,
                                    gl,
                                );
                                gl.begin(GL_LINE_STRIP);
                            }
                        }
                    }
                    let p = (&(*cont).pts)[pt as usize];
                    gl.vertex3f(p.x, p.y, p.z);

                    if pt_props.gap != 0 {
                        gl.end();
                        gl.begin(GL_LINE_STRIP);
                    }
                    pt += 1;
                }
                if mode == GL_LINE_LOOP
                    && (*cont).flags & ICONT_OPEN == 0
                    && pt_props.gap == 0
                    && !(*cont).pts.is_empty()
                {
                    let p = (&(*cont).pts)[0];
                    gl.vertex3f(p.x, p.y, p.z);
                }
                gl.end();
                if app.standalone == 0 {
                    gl.disable_stipple(draw_stipple, &*cont);
                }
            }
        }
    }
}

/// Static `imodvDraw_filled_contours`: draw filled contours with polygon
/// tesselation.
///
/// # Safety
/// `obj` must point at a live object of the model being drawn.
pub unsafe fn imodv_draw_filled_contours(
    state: &mut MvOglState,
    app: &ImodvApp,
    obj: *mut Iobj,
    draw_trans: i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let mut cont_props = DrawProps::default();
        let mut pt_props = DrawProps::default();
        let mut state_flags = 0i32;
        let mut handle_flags = (if (*obj).flags & IMOD_OBJFLAG_FCOLOR != 0 {
            HANDLE_MESH_FCOLOR
        } else {
            HANDLE_MESH_COLOR
        }) | HANDLE_TRANS;
        let mut check_time = iobj_time((*obj).flags) as i32;
        if state.ctime == 0 {
            check_time = 0;
        }

        if (*obj).cont.is_empty() {
            return;
        }

        if ifg_setup_value_drawing(
            &*obj,
            GEN_STORE_MINMAX1,
            -1,
            &mut state.values,
            &mut crate::imod::three_dmod::xcramp::xcramp_mapfalsecolor,
        ) != 0
        {
            handle_flags |= HANDLE_VALUE1;
        }

        // Skip drawing first time in if object is trans and no contours become solid
        if draw_trans == 0 && (*obj).trans != 0 && istore_trans_state_matches(&(*obj).store, 0) == 0
        {
            (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
            return;
        }

        gl.setup_filled_cont_tesselator();

        let cur_ob = if app.imod.is_null() {
            -1
        } else {
            (*app.imod).cindex.object
        };

        gl.push_name(NO_NAME);
        for co in 0..(*obj).cont.len() {
            gl.load_name(co as u32);
            let cont: *const Icont = &(&(*obj).cont)[co];

            // 8/29/06: it was only checking time before (not even size)
            let selected = imod_selection_list_query(&state.selection, cur_ob, co as i32) >= 0;
            if imodv_check_contour_draw(state, app, &*cont, co as i32, check_time != 0, selected)
                == 0
            {
                continue;
            }

            ifg_handle_cont_change(
                &*obj,
                co as i32,
                &mut cont_props,
                &mut pt_props,
                &mut state_flags,
                handle_flags,
                0,
                0,
                &state.values,
                gl,
            );
            if cont_props.gap != 0 {
                continue;
            }
            if (if cont_props.trans != 0 { 1 } else { 0 }) != draw_trans {
                (*obj).flags |= IMOD_OBJFLAG_TEMPUSE;
                continue;
            }
            gl.draw_filled_polygon(&*cont);
        }
        gl.pop_name();
    }
}

/// `imodvSelectVisibleConts`.
///
/// # Safety
/// `app` must reference a live viewer with a live current model.
pub unsafe fn imodv_select_visible_conts(
    state: &mut MvOglState,
    app: &mut ImodvApp,
    picked_ob: &mut i32,
    picked_co: &mut i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let imod = app.imod;
        if imod.is_null() {
            return;
        }
        let ob = (*imod).cindex.object;
        let mut cont_props = DrawProps::default();
        let mut pt_props = DrawProps::default();
        let mut state_flags = 0i32;
        let mut handle_flags = 0i32;
        let mut num_sel = 0i32;
        let mut n_planes = 0i32;
        let mut plane = vec![Iplane::default(); 2 * IMOD_CLIPSIZE];

        if ob < 0 {
            return;
        }
        let obj: *mut Iobj = &mut (&mut (*imod).obj)[ob as usize];
        let mut check_time = iobj_time((*obj).flags) as i32;

        // `Imod::ctime` (`imodel.h:457`) is not carried by the translated `Imod`
        // in `libimod`, so `sCTime = imod->ctime` cannot be assigned here.
        if state.ctime == 0 {
            check_time = 0;
        }
        if iobj_off((*obj).flags) != 0 || iobj_line((*obj).flags) == 0 {
            return;
        }

        let max_planes = gl.max_clip_planes();
        imod_plane_set_from_clips(
            Some(&(*obj).clips),
            (*imod).view.first().map(|v| &v.clips),
            &mut plane,
            max_planes,
            &mut n_planes,
        );

        imod_selection_list_clear(&mut state.selection);

        set_curcontsurf(state, app, ob, &*imod);
        if ifg_setup_value_drawing(
            &*obj,
            GEN_STORE_MINMAX1,
            -1,
            &mut state.values,
            &mut crate::imod::three_dmod::xcramp::xcramp_mapfalsecolor,
        ) != 0
        {
            handle_flags |= HANDLE_VALUE1;
        }
        for co in 0..(*obj).cont.len() {
            let cont: *const Icont = &(&(*obj).cont)[co];
            let selected = imod_selection_list_query(&state.selection, ob, co as i32) >= 0;
            if imodv_check_contour_draw(state, app, &*cont, co as i32, check_time != 0, selected)
                == 0
            {
                continue;
            }
            ifg_handle_cont_change(
                &*obj,
                co as i32,
                &mut cont_props,
                &mut pt_props,
                &mut state_flags,
                handle_flags,
                0,
                0,
                &state.values,
                gl,
            );
            if cont_props.gap != 0 {
                continue;
            }

            // Check the clipping planes; if any point is visible, break and accept
            if n_planes > 0 {
                let mut pt = 0;
                while pt < (*cont).pts.len() {
                    if imod_planes_clip(&plane, n_planes, &(&(*cont).pts)[pt]) != 0 {
                        break;
                    }
                    pt += 1;
                }
                if pt >= (*cont).pts.len() {
                    continue;
                }
            }

            // The first time, just set the model index
            // The second time, add previous index to selection list
            // After the first time, add every index to the selection list
            if num_sel == 1 {
                imod_selection_list_add(&mut state.selection, (*imod).cindex);
            }
            let cindex = Iindex {
                object: ob,
                contour: co as i32,
                point: -1,
            };
            (*imod).cindex = cindex;
            if num_sel != 0 {
                imod_selection_list_add(&mut state.selection, cindex);
            }
            num_sel += 1;
            *picked_co = co as i32;
            *picked_ob = ob;
        }
    }
}

/// Static `drawCurrentClipPlane`: draw the current clip plane if it is on.
///
/// # Safety
/// `app` must reference a live viewer with a live current model.
pub unsafe fn draw_current_clip_plane(
    state: &mut MvOglState,
    app: &mut ImodvApp,
    slicer_not_clip: i32,
    gl: &mut dyn MvOglBoundary,
) {
    unsafe {
        let mut alpha = 0f64;
        let mut beta = 0f64;
        let Some(mut mat) = crate::imod::libimod::imat::imod_mat_new(3) else {
            return;
        };
        let mut radfrac: f32 = 0.95f32;
        let mut corner = Ipoint::default();
        let mut xcorn = Ipoint::default();
        let mut cen = Ipoint::default();
        let mut normal = Ipoint::default();
        let mut slcen = Ipoint::default();
        let mut vx = [0f32; 4];
        let mut vy = [0f32; 4];
        let mut vz = [0f32; 4];
        let mut angles = [0f32; 3];

        objed_object(app);
        if app.obj.is_null() || app.imod.is_null() {
            return;
        }
        let imod = app.imod;
        let Some(vw) = (*imod).view.first() else {
            return;
        };
        let rad = vw.rad;
        let zscale: f32 = if (*imod).zscale != 0. {
            (*imod).zscale
        } else {
            1.
        };

        if slicer_not_clip != 0 {
            let mut time = 0i32;
            let mut swinx = 0i32;
            let mut swiny = 0i32;
            let mut zoom = 1f32;
            if gl.top_slicer_plane(
                &mut angles,
                &mut cen,
                &mut time,
                &mut normal,
                &mut swinx,
                &mut swiny,
                &mut zoom,
            ) != 0
                || time != state.ctime
            {
                return;
            }
            // The source keeps drawing about the slicer centre `cen`; `slcen`
            // is the clip-frame centre the call returns and is not used again.
            clip_center_and_angles(app, &cen, &normal, &mut slcen, &mut alpha, &mut beta);
            if app.draw_slicer_plane & 2 == 0 {
                radfrac = (0.5 * ((swinx as f64 * swiny as f64).sqrt()) / zoom as f64 / rad as f64)
                    as f32;
            }
        } else {
            // `mv_ogl.cpp:3069`: `a->imod->editGlobalClip ? &a->imod->view->clips
            // : &a->obj->clips`.  `view` is the array and `view->clips` is
            // element 0's.
            let clips = if (*app.imod).edit_global_clip != 0 {
                &(&(*app.imod).view)[0].clips
            } else {
                &(*app.obj).clips
            };
            let ip = clips.plane as usize;
            if clips.flags & (1 << ip) == 0 {
                return;
            }
            let point = clips.point[ip];
            let norm = clips.normal[ip];
            clip_center_and_angles(app, &point, &norm, &mut cen, &mut alpha, &mut beta);
        }
        imod_mat_rot(
            &mut mat,
            -((alpha / RADIANS_PER_DEGREE) as f32) as f64,
            B3D_X,
        );
        imod_mat_rot(
            &mut mat,
            -((beta / RADIANS_PER_DEGREE) as f32) as f64,
            B3D_Y,
        );

        // Compute and draw 4 corner points.
        // It works best if you draw lines before plane
        gl.color4ub(255, 0, 0, 255);
        gl.begin(GL_LINE_LOOP);
        for ind in 0..4 {
            corner.x = (if ind == 1 || ind == 2 { -1. } else { 1. }) * radfrac * rad;
            corner.y = (if ind / 2 == 1 { -1. } else { 1. }) * radfrac * rad;
            corner.z = 0.;
            imod_mat_transform3d(&mat, &corner, &mut xcorn);
            vx[ind] = cen.x + xcorn.x;
            vy[ind] = cen.y + xcorn.y;
            vz[ind] = cen.z + xcorn.z / zscale;
            gl.vertex3f(vx[ind], vy[ind], vz[ind]);
        }
        gl.end();
        gl.blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        gl.blend(true);

        gl.color4ub(255, 0, 0, 96);
        gl.begin(GL_POLYGON);

        for ind in 0..4 {
            gl.vertex3f(vx[ind], vy[ind], vz[ind]);
        }
        gl.end();
        gl.blend_func(GL_ONE, GL_ZERO);
        gl.blend(false);

        crate::imod::libimod::imat::imod_mat_delete(&mut mat);
    }
}
/// `imodvDrawLabels` (`mv_ogl.cpp:1497`).
///
/// The source's early return on an object without `IMOD_OBJFLAG_DRAW_LABEL`
/// is reproduced exactly, so an object with labels off costs nothing and says
/// nothing.  Everything after that return ends in text output —
/// `QPainter::drawText` on the GL widget, or `QGLWidget::renderText` on the
/// pre-Qt5 branch — through a `QFont` sized from
/// `obj->extra[IOBJ_EX_LABEL_SIZE]`, positioned by `myProject` against the
/// current `GL_PROJECTION_MATRIX`/`GL_MODELVIEW_MATRIX`.  There is no glyph
/// rasteriser in this tree and no `sLabMat` transform reaches here, so the
/// call reports what it needs instead of drawing nothing silently.
pub fn imodv_draw_labels(
    imod: &Imod,
    object: &Iobj,
    win_size_y: i32,
    device_pixel_ratio: f32,
    font_height: i32,
) {
    let _ = (imod, win_size_y, device_pixel_ratio, font_height);
    if object.flags & IMOD_OBJFLAG_DRAW_LABEL == 0 {
        return;
    }
    static REPORTED: std::sync::Once = std::sync::Once::new();
    REPORTED.call_once(|| {
        imod_print_stderr(
            "3dmod: point labels are not drawn: imodvDrawLabels (mv_ogl.cpp:1497) needs a text \
             rasteriser for QPainter::drawText/QGLWidget::renderText and the sLabMat transform \
             that imodDrawSetupLabelDraw (model_draw.cpp:224) records\n",
        )
    });
}
/// `imodDrawSetupLabelDraw`, declared by `mv_ogl.h:31` and defined by
/// `model_draw.cpp:224`.
///
/// The parameters it records are the `model_draw.cpp` file statics, which the
/// paired translation already owns, so they are set there; the source's own
/// last statement, `imodvCleanupLabelFont()`, belongs to this unit and is
/// issued here.  `sLabMat` is not among the recorded fields of
/// `ModelDrawState`, which is the same gap `imodvDrawLabels` reports.
pub fn imod_draw_setup_label_draw(
    state: &mut crate::imod::three_dmod::model_draw::ModelDrawState,
    win_size_y: i32,
    device_pixel_ratio: f32,
    font_height: i32,
) {
    crate::imod::three_dmod::model_draw::imod_draw_setup_label_draw(
        state,
        win_size_y,
        device_pixel_ratio,
        font_height,
    );
    imodv_cleanup_label_font();
}
/// `imodvCleanupLabelFont`.
pub fn imodv_cleanup_label_font() {
    let mut state = LABEL_FONT_STATE.lock().expect("label font state poisoned");
    state.label_font = None;
    state.font_size = -1;
    state.label_painter = None;
}
/// `imodvUnprojectPickedPoint` (`mv_ogl.cpp:3114`).
pub fn imodv_unproject_picked_point(app: &mut ImodvApp, mo: i32, gl: &mut dyn MvOglBoundary) {
    let half_pick: i32;
    let x_start: i32;
    let y_start: i32;
    let mut min_x: i32;
    let mut min_y: i32;
    let mut new_x: i32;
    let mut new_y: i32;
    let cen_x: i32;
    let cen_y: i32;
    let mut dist: i32;
    let mut steps = 0;
    let max_steps = 2;
    let mut min_depth: f32;
    let mut depths = [0.0f32; 21 * 21];
    let mut upx = 0.;
    let mut upy = 0.;
    let mut upz = 0.;

    half_pick = app.w_pick / 2;
    x_start = 0.max(app.x_pick - half_pick);
    y_start = 0.max(app.y_pick - half_pick);
    cen_x = app.x_pick - x_start;
    min_x = cen_x;
    cen_y = app.y_pick - y_start;
    min_y = cen_y;

    gl.read_depth_pixels(x_start, y_start, app.w_pick, app.w_pick, &mut depths);

    // Find closest point to center that is not at 1.0
    min_depth = 1.0e10;
    for del in 0..app.w_pick {
        for iy in 0.max(cen_y - del)..=(app.w_pick - 1).min(cen_y + del) {
            for ix in 0.max(cen_x - del)..=(app.w_pick - 1).min(cen_x + del) {
                dist = (ix - cen_x) * (ix - cen_x) + (iy - cen_y) * (iy - cen_y);
                if dist as f32 > (del * del) as f32
                    || (del > 1 && (dist as f32) < (del as f32 - 1.5) * (del as f32 - 1.5))
                {
                    continue;
                }
                if depths[(ix + iy * app.w_pick) as usize] < min_depth {
                    min_depth = depths[(ix + iy * app.w_pick) as usize];
                    min_x = ix;
                    min_y = iy;
                }
            }
        }
        if min_depth < 1. {
            break;
        }
    }

    // Walk from there to a peak, but only a few paces to avoid hitting another structure
    if min_depth < 1. {
        new_x = min_x;
        new_y = min_y;
        loop {
            if steps >= max_steps {
                break;
            }
            for iy in 0.max(min_y - 1)..=(app.w_pick - 1).min(min_y + 1) {
                for ix in 0.max(min_x - 1)..=(app.w_pick - 1).min(min_x + 1) {
                    if depths[(ix + iy * app.w_pick) as usize] < min_depth {
                        min_depth = depths[(ix + iy * app.w_pick) as usize];
                        new_x = ix;
                        new_y = iy;
                        steps += 1;
                    }
                }
            }
            if new_y == min_y && new_x == min_x {
                break;
            }
            new_x = min_x;
            new_y = min_y;
        }
    }

    min_x += x_start;
    min_y += y_start;

    // NO examples show 0.5 added, but that is officially the coordinates of this pixel
    // Also, measurements of errors on points at 1:1 zoom validated the need for this,
    // but Y coordinates were still in error by ~-0.7 to -0.8 for points in Linux
    // so tried another 0.5. With this, 1:1 zoom line picking had mean errors near 0.25
    // on Linux and Windows, 0.1 on Mac.  Settle on adding only another 0.25
    gl.un_project(
        min_x as f64 + 0.5,
        min_y as f64 + 0.75,
        min_depth as f64,
        &mut upx,
        &mut upy,
        &mut upz,
    );
    if let Some(pick) = (mo >= 0)
        .then(|| app.mod_picks.get_mut(mo as usize))
        .flatten()
    {
        pick.x = upx as f32;
        pick.y = upy as f32;
        pick.z = upz as f32;
    }
    imod_trace(
        'p',
        &format!(
            "mod {mo} min {min_x} {min_y} {min_depth}   unproject {upx:.2}  {upy:.2}  {upz:.2}"
        ),
    );
}
/// `findClickedDrawnElement` (`mv_ogl.cpp:3202`).
pub unsafe fn find_clicked_drawn_element(
    state: &mut MvOglState,
    app: &mut ImodvApp,
    cur_obj: bool,
    mo_num: &mut i32,
    ob_num: &mut i32,
    co_num: &mut i32,
    pt_num: &mut i32,
) -> i32 {
    let mut planes = [Iplane::default(); 2 * IMOD_CLIPSIZE];
    let mut n_planes: i32;
    let mut mstart = 0;
    let mut mend = 0;
    let base_scan_tol = 2.5f32;
    let stop_tol = 0.5f32;
    let mut nloop;
    let mut obstart: i32;
    let mut obend: i32;
    let mut check_time: bool;
    let mut resol = 0;
    let mut co_sphere = -1;
    let mut pt_sphere = -1;
    let mut co_mesh_cont = -1;
    let mut pt_mesh_cont = -1;
    // `scanTol` is a function-scope float in the source, set inside the model
    // loop and read again by the final return.
    let mut scan_tol = 0.0f32;
    let mut zscale;
    let mut dist_sphere = 0.;
    let mut dist_mesh_cont = 0.;
    let mut best_dist = 1.0e10f32;
    let mut skip_spheres: bool;
    let mut has_spheres: bool;

    *co_num = -1;
    // `a->vi->numExtraObj`; the view is the opaque ownership pointer the
    // source dereferences directly.
    let vi = unsafe { (app.vi as *mut ImodView).as_mut() };
    nloop = if vi.as_ref().is_some_and(|vi| !vi.extra_obj.is_empty()) {
        2
    } else {
        1
    };
    let num_extra_obj = vi.as_ref().map_or(0, |vi| vi.extra_obj.len() as i32);

    if app.draw_extra_only == 0 {
        crate::imod::three_dmod::mv_modeled::imodv_model_draw_range(app, &mut mstart, &mut mend);
    }
    for mo_ind in mstart..=mend {
        if mo_ind < 0 || mo_ind as usize >= app.mod_.len() {
            continue;
        }
        let imod = app.mod_[mo_ind as usize];
        let Some(imod) = (unsafe { imod.as_mut() }) else {
            continue;
        };
        let Some(view) = imod.view.first().cloned() else {
            continue;
        };
        zscale = imod.zscale;
        if app.standalone == 0 {
            if let Some(vi) = unsafe { (app.vi as *const ImodView).as_ref() } {
                zscale = (imod.zscale * vi.zbin as f32) / vi.xybin as f32;
            }
        }
        let scale = 0.5 * app.winx.min(app.winy) as f32 / view.rad;
        scan_tol = base_scan_tol + 0.66 / scale;
        let pick = usize::try_from(mo_ind)
            .ok()
            .and_then(|index| app.mod_picks.get(index))
            .copied()
            .unwrap_or_default();

        /* If displaying a current subset or doing current obj, set up object limits */
        obstart = 0;
        obend = imod.obj.len() as i32;
        if (app.current_subset == SUBSET_OBJ_ONLY
            || app.current_subset == SUBSET_SURF_ONLY
            || app.current_subset == SUBSET_CONT_ONLY
            || cur_obj)
            && imod.cindex.object >= 0
        {
            obstart = imod.cindex.object;
            obend = obstart + 1;
            if cur_obj {
                nloop = 1;
            }
        }

        // If drawing only extra objects, set up to skip all regular ones
        if app.draw_extra_only != 0 {
            obend = obstart - 1;
        }

        // Loop on regular objects then on extra objects
        for _loop in 0..nloop {
            for ob_ind in obstart..obend {
                // The source takes `obj` as a pointer into either the model or
                // the extra-object list; both are owned elsewhere, so the
                // pointer is what is kept here too.
                let obj: *const Iobj = if ob_ind >= 0 {
                    &imod.obj[ob_ind as usize]
                } else {
                    let Some(vi) = (unsafe { (app.vi as *mut ImodView).as_mut() }) else {
                        continue;
                    };
                    let Some(extra) =
                        crate::imod::three_dmod::imodview::ivw_get_an_extra_object(vi, -1 - ob_ind)
                    else {
                        continue;
                    };
                    if extra.flags & IMOD_OBJFLAG_EXTRA_MODV == 0
                        || String::from_utf8_lossy(
                            &extra.name[..extra
                                .name
                                .iter()
                                .position(|byte| *byte == 0)
                                .unwrap_or(extra.name.len())],
                        )
                        .contains("point extra")
                    {
                        continue;
                    }
                    extra as *const Iobj
                };
                let obj = unsafe { &*obj };
                if iobj_off(obj.flags) != 0 || (obj.cont.is_empty() && obj.mesh.is_empty()) {
                    continue;
                }

                // If pick point is outside the clip for this object, skip
                n_planes = 0;
                imod_plane_set_from_clips(
                    Some(&obj.clips),
                    Some(&view.clips),
                    &mut planes,
                    2 * IMOD_CLIPSIZE as i32,
                    &mut n_planes,
                );
                if n_planes != 0 && imod_planes_clip(&planes, n_planes, &pick) == 0 {
                    continue;
                }

                set_curcontsurf(state, app, ob_ind, imod);
                check_time = iobj_time(obj.flags) != 0;
                if state.ctime == 0 {
                    check_time = false;
                }
                skip_spheres =
                    obj.flags & IMOD_OBJFLAG_PNT_NOMODV != 0 && iobj_mesh(obj.flags) != 0;
                has_spheres = (iobj_scat(obj.flags) != 0 || (obj.pdrawsize > 0 && !skip_spheres))
                    && !obj.cont.is_empty();

                // Check for individual point sizes if not scattered and no sphere size
                if !has_spheres && !skip_spheres {
                    for co in 0..obj.cont.len() {
                        if !obj.cont[co].sizes.is_empty() {
                            has_spheres = true;
                            break;
                        }
                    }
                }
                if has_spheres {
                    if find_clicked_sphere(
                        obj,
                        &pick,
                        scan_tol,
                        zscale,
                        &mut co_sphere,
                        &mut pt_sphere,
                        &mut dist_sphere,
                    ) != 0
                        && dist_sphere < best_dist
                    {
                        *co_num = co_sphere;
                        *pt_num = pt_sphere;
                        *mo_num = mo_ind;
                        *ob_num = ob_ind;
                        best_dist = dist_sphere;
                    }

                    // If this obj is scattered, test for distance small enough or distance
                    // negative when filled and stop, or go to next object
                    // Otherwise fall though to cont/mesh
                    if iobj_scat(obj.flags) != 0 {
                        if best_dist.abs() < stop_tol
                            || (iobj_fill(obj.flags) != 0 && best_dist < 0.)
                        {
                            return 1;
                        }
                        continue;
                    }
                }

                // See if mesh is to be drawn and search in it if so
                if iobj_mesh(obj.flags) != 0 {
                    if !obj.mesh.is_empty() {
                        imod_mesh_nearest_res(
                            &obj.mesh,
                            obj.mesh.len() as i32,
                            app.lowres,
                            &mut resol,
                        );
                        for co in 0..obj.mesh.len() {
                            let mesh = &obj.mesh[co];
                            if check_mesh_draw(
                                state,
                                app,
                                mesh,
                                check_time,
                                resol,
                                obj.mesh_thickness as i32,
                            ) != 0
                                && find_clicked_mesh_element(
                                    mesh,
                                    &pick,
                                    scan_tol,
                                    zscale,
                                    &mut pt_mesh_cont,
                                    &mut dist_mesh_cont,
                                ) != 0
                            {
                                // Keep track of mesh # by adding a + # of conts, like in
                                // old method
                                co_mesh_cont = co as i32 + 1 + obj.cont.len() as i32;
                            }
                        }
                    }
                } else {
                    // No mesh, look for contours instead
                    find_clicked_cont_point(
                        obj,
                        &pick,
                        scan_tol,
                        zscale,
                        &mut co_mesh_cont,
                        &mut pt_mesh_cont,
                        &mut dist_mesh_cont,
                    );
                }

                // Take this is best if it is better than last
                if co_mesh_cont >= 0 && dist_mesh_cont < best_dist.abs() {
                    *co_num = co_mesh_cont;
                    *pt_num = pt_mesh_cont;
                    *mo_num = mo_ind;
                    *ob_num = ob_ind;
                    best_dist = dist_mesh_cont;
                }
                if best_dist.abs() < stop_tol {
                    return 1;
                }
            }

            // Do extra objects on next loop
            obstart = -num_extra_obj;
            obend = 0;
        }
    }
    (best_dist < scan_tol) as i32
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

    /// Records the fixed-function calls the renderer makes, in order.
    #[derive(Default)]
    struct RecordingOglBoundary {
        calls: Vec<&'static str>,
        vertices: Vec<(f32, f32, f32)>,
        modes: Vec<u32>,
        /// What `glReadPixels(GL_DEPTH_COMPONENT)` hands back, laid out the
        /// way `imodvUnprojectPickedPoint` indexes it.
        depth_buffer: Vec<f32>,
        /// The window pixel and depth the unprojection was asked for.
        unproject_from: Vec<(f64, f64, f64)>,
        /// What `gluUnProject` answers with.
        unproject_to: (f64, f64, f64),
    }
    impl FinegrainRenderBoundary for RecordingOglBoundary {
        fn color3f(&mut self, _: f32, _: f32, _: f32) {
            self.calls.push("color3f")
        }
        fn color4f(&mut self, _: f32, _: f32, _: f32, _: f32) {
            self.calls.push("color4f")
        }
        fn line_width(&mut self, _: i32, _: &Iobj) {
            self.calls.push("line_width")
        }
        fn point_size(&mut self, _: i32, _: &Iobj) {
            self.calls.push("point_size")
        }
        fn light_adjust(&mut self, _: &Iobj, _: f32, _: f32, _: f32, _: i32) {
            self.calls.push("light_adjust")
        }
        fn rgba(&self) -> bool {
            true
        }
    }
    impl MvOglBoundary for RecordingOglBoundary {
        fn push_name(&mut self, _: u32) {
            self.calls.push("push_name")
        }
        fn pop_name(&mut self) {
            self.calls.push("pop_name")
        }
        fn load_name(&mut self, _: u32) {
            self.calls.push("load_name")
        }
        fn finish(&mut self) {
            self.calls.push("finish")
        }
        fn projection_identity(&mut self) {
            self.calls.push("projection_identity")
        }
        fn projection_mode(&mut self) {
            self.calls.push("projection_mode")
        }
        fn modelview_identity(&mut self) {
            self.calls.push("modelview_identity")
        }
        fn modelview_mode(&mut self) {
            self.calls.push("modelview_mode")
        }
        fn ortho(&mut self, _: f64, _: f64, _: f64, _: f64) {
            self.calls.push("ortho")
        }
        fn frustum(&mut self, _: f64, _: f64, _: f64, _: f64) {
            self.calls.push("frustum")
        }
        fn translate(&mut self, _: f32, _: f32, _: f32) {
            self.calls.push("translate")
        }
        fn rotate(&mut self, _: f32, _: f32, _: f32, _: f32) {
            self.calls.push("rotate")
        }
        fn scale(&mut self, _: f32, _: f32, _: f32) {
            self.calls.push("scale")
        }
        fn push_matrix(&mut self) {
            self.calls.push("push_matrix")
        }
        fn pop_matrix(&mut self) {
            self.calls.push("pop_matrix")
        }
        fn viewport(&mut self, _: i32, _: i32, _: i32, _: i32) {
            self.calls.push("viewport")
        }
        fn depth_mask(&mut self, _: bool) {
            self.calls.push("depth_mask")
        }
        fn begin(&mut self, mode: u32) {
            self.calls.push("begin");
            self.modes.push(mode);
        }
        fn end(&mut self) {
            self.calls.push("end")
        }
        fn vertex3f(&mut self, x: f32, y: f32, z: f32) {
            self.calls.push("vertex3f");
            self.vertices.push((x, y, z));
        }
        fn normal3f(&mut self, _: f32, _: f32, _: f32) {
            self.calls.push("normal3f")
        }
        fn color4ub(&mut self, _: u8, _: u8, _: u8, _: u8) {
            self.calls.push("color4ub")
        }
        fn polygon_mode_line(&mut self, _: bool) {
            self.calls.push("polygon_mode_line")
        }
        fn front_face_cw(&mut self, _: bool) {
            self.calls.push("front_face_cw")
        }
        fn light_model_two_side(&mut self, _: i32) {
            self.calls.push("light_model_two_side")
        }
        fn blend(&mut self, _: bool) {
            self.calls.push("blend")
        }
        fn blend_func(&mut self, _: u32, _: u32) {
            self.calls.push("blend_func")
        }
        fn cull_face(&mut self, _: bool) {
            self.calls.push("cull_face")
        }
        fn line_smooth(&mut self, _: bool) {
            self.calls.push("line_smooth")
        }
        fn normalize(&mut self, _: bool) {
            self.calls.push("normalize")
        }
        fn light_on(&mut self, _: &Iobj, _: &Iview, _: f32) {
            self.calls.push("light_on")
        }
        fn light_off(&mut self) {
            self.calls.push("light_off")
        }
        fn quadric_draw_style(&mut self, _: u32) {
            self.calls.push("quadric_draw_style")
        }
        fn sphere(&mut self, _: f64, _: i32, _: i32) {
            self.calls.push("sphere")
        }
        fn gen_lists(&mut self, _: i32) -> u32 {
            self.calls.push("gen_lists");
            1
        }
        fn new_list(&mut self, _: u32) {
            self.calls.push("new_list")
        }
        fn end_list(&mut self) {
            self.calls.push("end_list")
        }
        fn call_list(&mut self, _: u32) {
            self.calls.push("call_list")
        }
        fn delete_lists(&mut self, _: u32, _: i32) {
            self.calls.push("delete_lists")
        }
        fn tess_polygon(&mut self, _: &[Ipoint]) {
            self.calls.push("tess_polygon")
        }
        fn setup_filled_cont_tesselator(&mut self) {
            self.calls.push("setup_filled_cont_tesselator")
        }
        fn draw_filled_polygon(&mut self, _: &Icont) {
            self.calls.push("draw_filled_polygon")
        }
        fn manage_paired_meshes(&mut self, _: &Iobj, _: i32) -> i32 {
            self.calls.push("manage_paired_meshes");
            0
        }
        fn max_clip_planes(&mut self) -> i32 {
            self.calls.push("max_clip_planes");
            6
        }
        fn clip_plane(&mut self, _: i32, _: [f64; 4]) {
            self.calls.push("clip_plane")
        }
        fn enable_clip_plane(&mut self, _: i32) {
            self.calls.push("enable_clip_plane")
        }
        fn disable_clip_plane(&mut self, _: i32) {
            self.calls.push("disable_clip_plane")
        }
        fn image_any_clipping(&mut self) -> bool {
            false
        }
        fn image_clip_planes(&mut self) -> Option<Iclip_planes> {
            None
        }
        fn set_light(&mut self, _: &mut Iview) {
            self.calls.push("set_light")
        }
        fn enable_stipple(&mut self, _: i32, _: &Icont) {
            self.calls.push("enable_stipple")
        }
        fn disable_stipple(&mut self, _: i32, _: &Icont) {
            self.calls.push("disable_stipple")
        }
        fn top_slicer_plane(
            &mut self,
            _: &mut [f32; 3],
            _: &mut Ipoint,
            _: &mut i32,
            _: &mut Ipoint,
            _: &mut i32,
            _: &mut i32,
            _: &mut f32,
        ) -> i32 {
            self.calls.push("top_slicer_plane");
            1
        }
        fn draw_image(&mut self, _: &mut ImodvApp, _: bool) {
            self.calls.push("draw_image")
        }
        fn read_depth_pixels(&mut self, _: i32, _: i32, _: i32, _: i32, depths: &mut [f32]) {
            self.calls.push("read_depth_pixels");
            for (slot, value) in depths.iter_mut().zip(self.depth_buffer.iter()) {
                *slot = *value;
            }
        }
        fn un_project(
            &mut self,
            winx: f64,
            winy: f64,
            winz: f64,
            objx: &mut f64,
            objy: &mut f64,
            objz: &mut f64,
        ) {
            self.calls.push("un_project");
            self.unproject_from.push((winx, winy, winz));
            *objx = self.unproject_to.0;
            *objy = self.unproject_to.1;
            *objz = self.unproject_to.2;
        }
        fn draw_labels(&mut self, _: &ImodvApp, _: &Imod, _: &Iobj) {
            self.calls.push("draw_labels")
        }
    }

    /// One open contour object with `n` points on the X axis.
    fn line_object(points: &[Ipoint]) -> Iobj {
        let mut o = Iobj::default();
        o.linewidth = 1;
        o.cont.push(Icont {
            pts: points.to_vec(),
            ..Default::default()
        });
        o
    }

    #[test]
    fn imodv_set_model_trans_selects_the_modelview_matrix_first() {
        let state = MvOglState::default();
        let app = ImodvApp::default();
        let imod = Imod {
            view: vec![Iview::default()],
            ..Default::default()
        };
        let mut gl = RecordingOglBoundary::default();
        imodv_set_model_trans(&state, &app, &imod, &mut gl);
        assert_eq!(
            gl.calls,
            vec![
                "modelview_identity",
                "translate",
                "rotate",
                "rotate",
                "rotate",
                "translate",
                "scale",
            ]
        );
    }

    #[test]
    fn imodv_stereo_projection_selects_the_projection_matrix_first() {
        let app = ImodvApp::default();
        let mut gl = RecordingOglBoundary::default();
        set_stereo_projection(&app, 0, &mut gl);
        assert_eq!(gl.calls, vec!["projection_mode"]);
    }

    #[test]
    fn cleanup_label_font_discards_cached_font_and_painter() {
        {
            let mut state = LABEL_FONT_STATE.lock().unwrap();
            state.label_font = Some(LabelFont);
            state.label_painter = Some(LabelPainter);
            state.font_size = 14;
        }

        imodv_cleanup_label_font();

        let state = LABEL_FONT_STATE.lock().unwrap();
        assert!(state.label_font.is_none());
        assert!(state.label_painter.is_none());
        assert_eq!(state.font_size, -1);
    }

    #[test]
    fn sphere_lookup_matches_source_table() {
        let mut s = MvOglState::default();
        let app = ImodvApp::default();
        let mut o = Iobj::default();
        let mut gl = RecordingOglBoundary::default();
        unsafe { imodv_draw_spheres(&mut s, &app, &mut o, 1., DRAW_FILL, 0, &mut gl) };
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

    #[test]
    fn imodv_draw_contours_emits_one_line_strip_per_contour() {
        let mut state = MvOglState::default();
        let mut app = ImodvApp::default();
        app.standalone = 1;
        let mut obj = line_object(&[
            Ipoint {
                x: 1.,
                y: 2.,
                z: 3.,
            },
            Ipoint {
                x: 4.,
                y: 5.,
                z: 6.,
            },
        ]);
        let mut gl = RecordingOglBoundary::default();
        unsafe {
            imodv_draw_contours(&mut state, &app, &mut obj, GL_LINE_STRIP, 0, &mut gl);
        }
        // `ifgHandleContChange` issues the contour colour through
        // `ifgHandleColorTrans`, which is `glColor4f` plus `light_adjust`.
        assert_eq!(
            gl.calls,
            vec![
                "color4f",
                "light_adjust",
                "line_width",
                "begin",
                "vertex3f",
                "vertex3f",
                "end"
            ]
        );
        assert_eq!(gl.modes, vec![GL_LINE_STRIP]);
        assert_eq!(gl.vertices, vec![(1., 2., 3.), (4., 5., 6.)]);
    }

    #[test]
    fn imodv_draw_contours_closes_a_line_loop_with_the_first_point() {
        let mut state = MvOglState::default();
        let mut app = ImodvApp::default();
        app.standalone = 1;
        let mut obj = line_object(&[
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 1.,
                z: 0.,
            },
        ]);
        let mut gl = RecordingOglBoundary::default();
        unsafe {
            imodv_draw_contours(&mut state, &app, &mut obj, GL_LINE_LOOP, 0, &mut gl);
        }
        assert_eq!(gl.vertices.len(), 4);
        assert_eq!(gl.vertices[3], (0., 0., 0.));
    }

    #[test]
    fn imodv_draw_contours_points_mode_emits_a_point_batch() {
        let mut state = MvOglState::default();
        let mut app = ImodvApp::default();
        app.standalone = 1;
        let mut obj = line_object(&[Ipoint {
            x: 7.,
            y: 8.,
            z: 9.,
        }]);
        let mut gl = RecordingOglBoundary::default();
        unsafe {
            imodv_draw_contours(&mut state, &app, &mut obj, GL_POINTS, 0, &mut gl);
        }
        assert_eq!(
            gl.calls,
            vec![
                "color4f",
                "light_adjust",
                "point_size",
                "begin",
                "vertex3f",
                "end"
            ]
        );
        assert_eq!(gl.modes, vec![GL_POINTS]);
    }

    #[test]
    fn imodv_draw_filled_contours_tesselates_each_drawn_contour() {
        let mut state = MvOglState::default();
        let app = ImodvApp::default();
        let mut obj = line_object(&[
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 0.,
                y: 1.,
                z: 0.,
            },
        ]);
        let mut gl = RecordingOglBoundary::default();
        unsafe {
            imodv_draw_filled_contours(&mut state, &app, &mut obj, 0, &mut gl);
        }
        assert_eq!(
            gl.calls,
            vec![
                "setup_filled_cont_tesselator",
                "push_name",
                "load_name",
                "color4f",
                "light_adjust",
                "draw_filled_polygon",
                "pop_name",
            ]
        );
    }

    #[test]
    fn imodv_pick_contours_names_every_point_of_a_line_object() {
        let mut state = MvOglState::default();
        let app = ImodvApp::default();
        let mut obj = line_object(&[
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
        ]);
        // `iobjLine` is the inverted IMOD_OBJFLAG_NOLINE sense; a default
        // object is a line object, so pick draws GL_LINES segments.
        let mut gl = RecordingOglBoundary::default();
        unsafe {
            imodv_pick_contours(&mut state, &app, &mut obj, 1., 0, &mut gl);
        }
        // A default object is closed, so the last point's segment wraps back
        // to the first: two GL_LINES pairs, not one.
        assert_eq!(gl.modes, vec![GL_LINES, GL_LINES]);
        assert_eq!(
            gl.vertices,
            vec![(0., 0., 0.), (1., 1., 1.), (1., 1., 1.), (0., 0., 0.)]
        );
        assert_eq!(gl.calls.first(), Some(&"push_name"));
        assert_eq!(gl.calls.last(), Some(&"pop_name"));
    }

    #[test]
    fn imodv_draw_mesh_emits_a_triangle_loop_per_polynorm2_triangle() {
        let mut state = MvOglState::default();
        let app = ImodvApp::default();
        let mut obj = Iobj::default();
        obj.mesh.push(Imesh {
            vert: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 0.,
                },
            ],
            list: vec![
                IMOD_MESH_BGNPOLYNORM2,
                0,
                1,
                2,
                IMOD_MESH_ENDPOLY,
                IMOD_MESH_END,
            ],
            ..Default::default()
        });
        let mut gl = RecordingOglBoundary::default();
        let mesh: *mut Imesh = &mut obj.mesh[0];
        unsafe {
            imodv_draw_mesh(&mut state, &app, mesh, DRAW_LINES, &mut obj, 0, &mut gl);
        }
        assert_eq!(gl.modes, vec![GL_LINE_LOOP]);
        assert_eq!(gl.vertices, vec![(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)]);
    }

    #[test]
    fn imodv_draw_filled_mesh_scales_z_by_the_model_zscale() {
        let mut state = MvOglState::default();
        let app = ImodvApp::default();
        let mut obj = Iobj::default();
        // A BGNPOLYNORM2 list indexes the vertex and takes its normal from the
        // following entry, so each triangle corner needs a vertex/normal pair.
        obj.mesh.push(Imesh {
            vert: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 2.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 3.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
            ],
            list: vec![
                IMOD_MESH_BGNPOLYNORM2,
                0,
                2,
                4,
                IMOD_MESH_ENDPOLY,
                IMOD_MESH_END,
            ],
            ..Default::default()
        });
        let mut gl = RecordingOglBoundary::default();
        let mesh: *mut Imesh = &mut obj.mesh[0];
        unsafe {
            imodv_draw_filled_mesh(&mut state, &app, mesh, 2., &mut obj, 0, &mut gl);
        }
        assert_eq!(gl.modes, vec![GL_TRIANGLES]);
        assert_eq!(gl.vertices, vec![(0., 0., 2.), (1., 0., 4.), (0., 1., 6.)]);
    }

    #[test]
    fn imodv_draw_object_of_a_line_object_sets_the_object_then_draws_contours() {
        let mut state = MvOglState::default();
        let app = ImodvApp::default();
        let mut imod = Imod {
            view: vec![Iview::default()],
            ..Default::default()
        };
        imod.obj.push(line_object(&[
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
        ]));
        let mut gl = RecordingOglBoundary::default();
        let imod_ptr: *mut Imod = &mut imod;
        unsafe {
            let obj: *mut Iobj = &mut (&mut (*imod_ptr).obj)[0];
            imodv_draw_object(&mut state, &app, obj, imod_ptr, 0, &mut gl);
        }
        // imodvSetObject(DRAW_LINES) then the GL_LINE_LOOP contour draw then
        // imodvSetObject(0), which is imodvUnsetObject.
        assert_eq!(
            gl.calls,
            vec![
                "polygon_mode_line",
                "line_width",
                "color4f",
                "color4f",
                "light_adjust",
                "enable_stipple",
                "line_width",
                "begin",
                "vertex3f",
                "vertex3f",
                "vertex3f",
                "end",
                "disable_stipple",
                "light_off",
                "blend",
                "line_smooth",
                "cull_face",
                "polygon_mode_line",
            ]
        );
    }

    /// `imodvUnprojectPickedPoint` reads the depth square around the click,
    /// walks to the nearest pixel, and unprojects it with the source's own
    /// half-and-quarter-pixel offsets.
    #[test]
    fn imodv_unproject_picked_point_finds_the_nearest_depth_and_offsets_it() {
        let mut app = ImodvApp {
            w_pick: 5,
            h_pick: 5,
            x_pick: 10,
            y_pick: 20,
            mod_picks: vec![Ipoint::default(); 2],
            ..Default::default()
        };
        let mut gl = RecordingOglBoundary {
            depth_buffer: vec![1.0; 25],
            unproject_to: (3., 4., 5.),
            ..RecordingOglBoundary::default()
        };
        gl.depth_buffer[1 + 2 * 5] = 0.25;
        imodv_unproject_picked_point(&mut app, 1, &mut gl);
        assert_eq!(gl.calls, ["read_depth_pixels", "un_project"]);
        // xStart = 10 - 2, yStart = 20 - 2; the minimum is at (1, 2) in the
        // square, so (9, 20) in the window, plus 0.5 and 0.75.
        assert_eq!(gl.unproject_from, [(9.5, 20.75, 0.25)]);
        assert_eq!(
            (app.mod_picks[1].x, app.mod_picks[1].y, app.mod_picks[1].z),
            (3., 4., 5.)
        );
        assert_eq!(
            (app.mod_picks[0].x, app.mod_picks[0].y, app.mod_picks[0].z),
            (0., 0., 0.)
        );
    }

    /// With every depth at the far plane the source never breaks out of the
    /// search, and unprojects the click centre at depth 1.
    #[test]
    fn imodv_unproject_picked_point_uses_the_centre_when_nothing_is_nearer() {
        let mut app = ImodvApp {
            w_pick: 5,
            x_pick: 4,
            y_pick: 4,
            mod_picks: vec![Ipoint::default()],
            ..Default::default()
        };
        let mut gl = RecordingOglBoundary {
            depth_buffer: vec![1.0; 25],
            ..RecordingOglBoundary::default()
        };
        imodv_unproject_picked_point(&mut app, 0, &mut gl);
        assert_eq!(gl.unproject_from, [(4.5, 4.75, 1.0)]);
    }

    /// `findClickedDrawnElement` reaches the contour search and reports the
    /// contour and point nearest the unprojected pick.
    #[test]
    fn imodv_find_clicked_drawn_element_reports_the_nearest_contour_point() {
        let mut model = Imod::default();
        model.zscale = 1.;
        let mut view = Iview::default();
        view.rad = 100.;
        model.view.push(view);
        let mut object = Iobj::default();
        let mut contour = Icont::default();
        contour.pts.push(Ipoint {
            x: 0.,
            y: 0.,
            z: 0.,
        });
        contour.pts.push(Ipoint {
            x: 20.,
            y: 0.,
            z: 0.,
        });
        object.cont.push(contour);
        model.obj.push(object);
        model.cindex.object = -1;
        let mut app = ImodvApp {
            num_mods: 1,
            cur_mod: 0,
            winx: 512,
            winy: 512,
            mod_picks: vec![Ipoint {
                x: 20.,
                y: 0.,
                z: 0.,
            }],
            ..Default::default()
        };
        app.mod_.push(&mut model);
        app.imod = app.mod_[0];
        let mut state = MvOglState::default();
        let (mut mo, mut ob, mut co, mut pt) = (-1, -1, -1, -1);
        let found = unsafe {
            find_clicked_drawn_element(
                &mut state, &mut app, false, &mut mo, &mut ob, &mut co, &mut pt,
            )
        };
        assert_eq!(found, 1);
        assert_eq!((mo, ob, co, pt), (0, 0, 0, 1));
    }

    /// `imodDrawSetupLabelDraw` records the label-draw parameters in
    /// `model_draw.cpp`'s statics and ends by discarding the cached font.
    #[test]
    fn imodv_label_draw_setup_records_parameters_and_drops_the_font() {
        {
            let mut font = LABEL_FONT_STATE.lock().expect("label font state");
            font.font_size = 12;
            font.label_font = Some(LabelFont);
        }
        let mut draw = crate::imod::three_dmod::model_draw::ModelDrawState::default();
        imod_draw_setup_label_draw(&mut draw, 480, 2., 14);
        assert!(draw.label_draw_set);
        assert_eq!(
            (draw.win_size_y, draw.dev_pix_ratio, draw.font_height),
            (480, 2., 14)
        );
        let font = LABEL_FONT_STATE.lock().expect("label font state");
        assert_eq!(font.font_size, -1);
        assert!(font.label_font.is_none());
    }
}
