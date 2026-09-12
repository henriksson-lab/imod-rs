//! Translation of `IMOD/3dmod/model_draw.cpp`.
//!
//! This unit is the slicer-window model renderer.  It deliberately keeps the
//! compatibility-profile GL commands and the still-pointer-owned label
//! renderer at named boundaries; contour/property traversal remains native
//! and follows the upstream draw order.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::icont::{ICONT_CURSOR_LIKE, ICONT_DRAW_ALLZ, ICONT_MMODEL_ONLY};
use crate::imod::libimod::imesh::{
    IMOD_MESH_BGNBIGPOLY, IMOD_MESH_BGNPOLY, IMOD_MESH_BGNPOLYNORM, IMOD_MESH_BGNPOLYNORM2,
    IMOD_MESH_ENDPOLY, imesh_resol, imesh_thickness, imod_mesh_nearest_res,
};
use crate::imod::libimod::imodel::{ICONT_OPEN, Icont, Imod, Iobj, Ipoint};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_EXTRA_EDIT, IMOD_OBJFLAG_MODV_ONLY, IOBJ_EX_FLAGS, IOBJ_EXFLAG_MESH_ON_IMG,
    IOBJ_EXFLAG_SLICER_ONLY, IOBJ_SYM_CIRCLE, IOBJ_SYM_NONE, IOBJ_SYM_SQUARE, IOBJ_SYM_STAR,
    IOBJ_SYM_TRIANGLE, IOBJ_SYMF_ENDS, IOBJ_SYMF_FILL, iobj_off, iobj_open, iobj_scat,
};
use crate::imod::libimod::istore::DrawProps;
use crate::imod::three_dmod::finegrain::{
    CHANGED_COLOR, FinegrainRenderBoundary, FinegrainValueState, HANDLE_2DWIDTH, HANDLE_LINE_COLOR,
    HANDLE_VALUE1, ifg_handle_cont_change, ifg_handle_mesh_change, ifg_handle_next_change,
    ifg_handle_surf_change,
};
use crate::imod::three_dmod::imod_edit::{ImodEditSelection, imod_selection_list_query};
use crate::imod::three_dmod::imodview::{IMOD_MMODEL, ImodView};
use crate::imod::three_dmod::utilities::util_current_point_size;

/// OpenGL primitive modes used by the fixed-function source.
pub const GL_POINTS: u32 = 0;
pub const GL_LINES: u32 = 1;
pub const GL_LINE_LOOP: u32 = 2;
pub const GL_LINE_STRIP: u32 = 3;
pub const GL_POLYGON: u32 = 4;

/// The source's process-global label arguments plus its generic-store value
/// state.  A Rust-owned state replaces file statics without changing the
/// renderer's observable reset-at-end behaviour.
#[derive(Clone, Debug, Default)]
pub struct ModelDrawState {
    pub label_draw_set: bool,
    pub win_size_y: i32,
    pub dev_pix_ratio: f32,
    pub font_height: i32,
    pub values: FinegrainValueState,
    pub selection: ImodEditSelection,
    pub bgnpoint: i32,
    pub endpoint: i32,
    pub curpoint: i32,
    pub min_mod_size: i32,
    pub min_im_size: i32,
    pub slicer_thickness_scaling: i32,
}

/// Direct fixed-function GL, labels, and unresolved paired-object machinery.
pub trait ModelDrawBoundary: FinegrainRenderBoundary {
    fn set_object_color(&mut self, object: i32);
    fn color_index(&mut self, color: i32);
    fn begin(&mut self, mode: u32);
    fn end(&mut self);
    fn vertex3(&mut self, point: Ipoint);
    fn push_matrix(&mut self);
    fn pop_matrix(&mut self);
    fn translate(&mut self, point: Ipoint);
    fn scale(&mut self, x: f32, y: f32, z: f32);
    fn sphere(&mut self, radius: f64, slices: i32, loops: i32);
    fn disk(&mut self, inner: f64, outer: f64, slices: i32, loops: i32);
    fn set_stipple(&mut self, enabled: bool);
    fn time_mismatch(&self, view: &ImodView, object: &Iobj, contour: &Icont) -> bool;
    fn draw_labels(&mut self, model: &Imod, object: &Iobj, win_y: i32, ratio: f32, height: i32);
    fn cleanup_label_font(&mut self);
    /// `utilManagePairedMeshes`, which is still owned by the paired mesh UI.
    fn manage_paired_meshes(&mut self, object: &Iobj, object_number: i32) -> bool;
}

/// `imodDrawSetupLabelDraw`.
pub fn imod_draw_setup_label_draw(
    state: &mut ModelDrawState,
    win_size_y: i32,
    dev_pix_ratio: f32,
    font_height: i32,
) {
    state.label_draw_set = true;
    state.win_size_y = win_size_y;
    state.dev_pix_ratio = dev_pix_ratio;
    state.font_height = font_height;
}

/// `imodDrawModel`.
pub fn imod_draw_model(
    view: &ImodView,
    model: &Imod,
    draw_current: bool,
    zscale: f32,
    state: &mut ModelDrawState,
    render: &mut dyn ModelDrawBoundary,
) {
    if model.drawmode <= 0 {
        return;
    }
    let (curob, curco, curpt) = (
        model.cindex.object,
        model.cindex.contour,
        model.cindex.point,
    );
    let total = model.obj.len() + view.extra_obj.len();
    for ob in 0..total {
        let (obj, real) = if ob < model.obj.len() {
            (&model.obj[ob], true)
        } else {
            let n = ob - model.obj.len();
            if view.extra_obj_in_use.get(n).copied().unwrap_or(0) == 0 {
                continue;
            }
            let obj = &view.extra_obj[n];
            if obj.flags & IMOD_OBJFLAG_EXTRA_EDIT == 0 {
                continue;
            }
            (obj, false)
        };
        if iobj_off(obj.flags) != 0 {
            continue;
        }
        if real {
            render.set_object_color(ob as i32);
        }
        render.line_width(obj.linewidth2 as i32, obj);
        if obj.extra[IOBJ_EX_FLAGS] & IOBJ_EXFLAG_MESH_ON_IMG != 0 && !obj.mesh.is_empty() {
            imod_draw_mesh(obj, ob as i32, state, render);
        }
        if !real {
            continue;
        }
        let has_spheres = iobj_scat(obj.flags) != 0
            || obj.pdrawsize != 0
            || obj.cont.iter().any(|c| !c.sizes.is_empty());
        if has_spheres {
            imod_draw_spheres(view, obj, zscale, state, render);
        }
        if iobj_scat(obj.flags) == 0 {
            for (co, cont) in obj.cont.iter().enumerate() {
                if cont.pts.is_empty() || render.time_mismatch(view, obj, cont) {
                    continue;
                }
                let mode = if iobj_open(obj.flags) != 0
                    || cont.flags & ICONT_OPEN != 0
                    || (ob as i32 == curob && co as i32 == curco)
                {
                    GL_LINE_STRIP
                } else {
                    GL_LINE_LOOP
                };
                imod_draw_contour_lines(view, obj, ob as i32, co as i32, mode, state, render);
            }
        }
        imod_draw_object_symbols(view, obj, state, render);
        if state.label_draw_set {
            render.draw_labels(
                model,
                obj,
                state.win_size_y,
                state.dev_pix_ratio,
                state.font_height,
            );
        }
    }
    state.label_draw_set = false;
    render.cleanup_label_font();
    if !draw_current {
        return;
    }
    let Some(obj) = model.obj.get(curob.max(0) as usize) else {
        return;
    };
    let Some(cont) = obj.cont.get(curco.max(0) as usize) else {
        return;
    };
    let Some(point) = cont.pts.get(curpt.max(0) as usize) else {
        return;
    };
    let (mod_size, backup_size, _) =
        util_current_point_size(Some(obj), state.min_mod_size, state.min_im_size, view.xybin);
    if !cont.pts.is_empty() && !render.time_mismatch(view, obj, cont) && view.drawcursor != 0 {
        if cont.pts.len() > 1 {
            render.color_index(state.bgnpoint);
            imod_draw_symbol(
                &cont.pts[0],
                IOBJ_SYM_CIRCLE,
                mod_size,
                0,
                obj.linewidth2 as i32,
                render,
            );
            render.color_index(state.endpoint);
            imod_draw_symbol(
                cont.pts.last().unwrap(),
                IOBJ_SYM_CIRCLE,
                mod_size,
                0,
                obj.linewidth2 as i32,
                render,
            );
        }
        if model.mousemode == IMOD_MMODEL {
            render.color_index(state.curpoint);
            let at_end = cont.pts.len() > 1 && (curpt == 0 || curpt as usize == cont.pts.len() - 1);
            imod_draw_symbol(
                point,
                IOBJ_SYM_CIRCLE,
                if at_end { backup_size } else { mod_size },
                0,
                obj.linewidth2 as i32,
                render,
            );
        }
    }
}

/// `imodDrawExtraObjects`.
pub fn imod_draw_extra_objects(
    view: &ImodView,
    model: &Imod,
    zscale: f32,
    draw_all_z: i32,
    state: &mut ModelDrawState,
    render: &mut dyn ModelDrawBoundary,
) -> bool {
    if model.drawmode <= 0 && draw_all_z == 0 {
        return false;
    }
    let mut retval = false;
    for (ob, obj) in view.extra_obj.iter().enumerate() {
        if view.extra_obj_in_use.get(ob).copied().unwrap_or(0) == 0
            || obj.cont.is_empty()
            || iobj_off(obj.flags) != 0
            || obj.flags & IMOD_OBJFLAG_MODV_ONLY != 0
            || obj.extra[IOBJ_EX_FLAGS] & IOBJ_EXFLAG_SLICER_ONLY == 0
        {
            continue;
        }
        render.color3f(obj.red, obj.green, obj.blue);
        render.line_width(obj.linewidth2 as i32, obj);
        if draw_all_z == 0
            && (iobj_scat(obj.flags) != 0
                || obj.pdrawsize != 0
                || obj.cont.iter().any(|c| !c.sizes.is_empty()))
        {
            imod_draw_spheres(view, obj, zscale, state, render);
        }
        if iobj_scat(obj.flags) == 0 {
            for (co, cont) in obj.cont.iter().enumerate() {
                if cont.pts.is_empty()
                    || (cont.flags & ICONT_MMODEL_ONLY != 0 && model.mousemode != IMOD_MMODEL)
                {
                    continue;
                }
                let cursor_like = cont.flags & ICONT_CURSOR_LIKE != 0;
                let all_z = cont.flags & ICONT_DRAW_ALLZ != 0;
                if (draw_all_z == 0 && !cursor_like && !all_z)
                    || (draw_all_z == 1 && all_z && !cursor_like)
                    || (draw_all_z == 2 && (all_z || cursor_like))
                {
                    if cursor_like {
                        retval = true;
                    }
                    let mode = if iobj_open(obj.flags) != 0 || cont.flags & ICONT_OPEN != 0 {
                        GL_LINE_STRIP
                    } else {
                        GL_LINE_LOOP
                    };
                    imod_draw_contour_lines(
                        view,
                        obj,
                        -1 - ob as i32,
                        co as i32,
                        mode,
                        state,
                        render,
                    );
                }
            }
        }
        if draw_all_z == 0 && state.label_draw_set {
            render.draw_labels(
                model,
                obj,
                state.win_size_y,
                state.dev_pix_ratio,
                state.font_height,
            );
        }
    }
    state.label_draw_set = false;
    render.cleanup_label_font();
    retval
}

/// Static `imodDrawContourLines`.
pub fn imod_draw_contour_lines(
    view: &ImodView,
    obj: &Iobj,
    object_number: i32,
    contour_number: i32,
    mode: u32,
    state: &mut ModelDrawState,
    render: &mut dyn ModelDrawBoundary,
) {
    let cont = &obj.cont[contour_number as usize];
    let mut cont_props = DrawProps::default();
    let mut pt_props = DrawProps::default();
    let mut flags = 0;
    let mut handle = HANDLE_LINE_COLOR | HANDLE_2DWIDTH;
    if state.values.val_setup != 0 {
        handle |= HANDLE_VALUE1;
    }
    let selected =
        (imod_selection_list_query(&state.selection, object_number, contour_number) > -2) as i32;
    let mut next = ifg_handle_cont_change(
        obj,
        contour_number,
        &mut cont_props,
        &mut pt_props,
        &mut flags,
        handle,
        selected,
        state.slicer_thickness_scaling,
        &state.values,
        render,
    );
    if cont_props.gap != 0 {
        return;
    }
    let mut cursor = 0usize;
    let mut changes = 0;
    if view.draw_stipple != 0 && cont.flags & crate::imod::libimod::icont::ICONT_STIPPLED != 0 {
        render.set_stipple(true);
    }
    render.begin(GL_LINE_STRIP);
    for (pt, point) in cont.pts.iter().copied().enumerate() {
        render.vertex3(point);
        pt_props.gap = 0;
        if next == pt as i32 {
            next = ifg_handle_next_change(
                obj,
                &cont.store,
                &mut cursor,
                &cont_props,
                &mut pt_props,
                &mut flags,
                &mut changes,
                handle,
                selected,
                state.slicer_thickness_scaling,
                &state.values,
                render,
            );
        }
        if pt_props.gap != 0 {
            render.end();
            render.begin(GL_LINE_STRIP);
        }
    }
    if mode == GL_LINE_LOOP && pt_props.gap == 0 {
        render.vertex3(cont.pts[0]);
    }
    render.end();
    if view.draw_stipple != 0 && cont.flags & crate::imod::libimod::icont::ICONT_STIPPLED != 0 {
        render.set_stipple(false);
    }
}

/// Static `imodDrawObjectSymbols`.
pub fn imod_draw_object_symbols(
    view: &ImodView,
    obj: &Iobj,
    state: &mut ModelDrawState,
    render: &mut dyn ModelDrawBoundary,
) {
    for (co, cont) in obj.cont.iter().enumerate() {
        if render.time_mismatch(view, obj, cont) {
            continue;
        }
        let mut cp = DrawProps::default();
        let mut pp = DrawProps::default();
        let mut flags = 0;
        let mut handle = HANDLE_LINE_COLOR | HANDLE_2DWIDTH;
        if state.values.val_setup != 0 {
            handle |= HANDLE_VALUE1;
        }
        let mut next = ifg_handle_cont_change(
            obj,
            co as i32,
            &mut cp,
            &mut pp,
            &mut flags,
            handle,
            0,
            0,
            &state.values,
            render,
        );
        if cp.gap != 0 {
            continue;
        }
        let mut cursor = 0usize;
        let mut changes = 0;
        for (pt, point) in cont.pts.iter().enumerate() {
            pp.gap = 0;
            if next == pt as i32 {
                next = ifg_handle_next_change(
                    obj,
                    &cont.store,
                    &mut cursor,
                    &cp,
                    &mut pp,
                    &mut flags,
                    &mut changes,
                    handle,
                    0,
                    0,
                    &state.values,
                    render,
                );
            }
            if pp.symtype != IOBJ_SYM_NONE && !(pp.gap != 0 && pp.valskip != 0) {
                imod_draw_symbol(
                    point,
                    pp.symtype,
                    pp.symsize,
                    pp.symflags,
                    pp.linewidth2,
                    render,
                );
            }
        }
        if obj.symflags as u32 & IOBJ_SYMF_ENDS != 0 && !cont.pts.is_empty() {
            imod_draw_end_marker(
                &cont.pts[0],
                obj.symsize as i32,
                state.bgnpoint,
                cp.linewidth2,
                render,
            );
            imod_draw_end_marker(
                cont.pts.last().unwrap(),
                obj.symsize as i32,
                state.endpoint,
                cp.linewidth2,
                render,
            );
        }
    }
}

/// Static end-marker body embedded in `imodDrawObjectSymbols`.
pub fn imod_draw_end_marker(
    point: &Ipoint,
    size: i32,
    color: i32,
    width: i32,
    render: &mut dyn ModelDrawBoundary,
) {
    render.color_index(color);
    render.line_width(width, &Iobj::default());
    let mut a = *point;
    a.x -= size as f32 / 2.;
    a.y -= size as f32 / 2.;
    let mut b = a;
    b.x += size as f32;
    b.y += size as f32;
    render.begin(GL_LINES);
    render.vertex3(a);
    render.vertex3(b);
    render.end();
    a.x -= size as f32;
    b.x += size as f32;
    b.y -= size as f32;
    render.begin(GL_LINES);
    render.vertex3(a);
    render.vertex3(b);
    render.end();
}

/// Static `imodDrawSpheres`.
pub fn imod_draw_spheres(
    view: &ImodView,
    obj: &Iobj,
    zscale: f32,
    state: &mut ModelDrawState,
    render: &mut dyn ModelDrawBoundary,
) {
    let zinv = 1. / zscale.max(f32::MIN_POSITIVE);
    let mut handle = HANDLE_LINE_COLOR;
    if state.values.val_setup != 0 {
        handle |= HANDLE_VALUE1;
    }
    for (co, cont) in obj.cont.iter().enumerate() {
        if cont.pts.is_empty() || render.time_mismatch(view, obj, cont) {
            continue;
        }
        let mut cp = DrawProps::default();
        let mut pp = DrawProps::default();
        let mut flags = 0;
        let mut next = ifg_handle_cont_change(
            obj,
            co as i32,
            &mut cp,
            &mut pp,
            &mut flags,
            handle,
            0,
            0,
            &state.values,
            render,
        );
        if cp.gap != 0 {
            continue;
        }
        let mut cursor = 0usize;
        let mut changes = 0;
        for (pt, point) in cont.pts.iter().enumerate() {
            pp.gap = 0;
            if next == pt as i32 {
                next = ifg_handle_next_change(
                    obj,
                    &cont.store,
                    &mut cursor,
                    &cp,
                    &mut pp,
                    &mut flags,
                    &mut changes,
                    handle,
                    0,
                    0,
                    &state.values,
                    render,
                );
            }
            let drawsize = imod_point_get_size(obj, cont, pt) as f64 / view.xybin.max(1) as f64;
            if drawsize == 0. || (pp.gap != 0 && pp.valskip != 0) {
                continue;
            }
            let steps = if drawsize < 5. {
                drawsize as i32 + 4
            } else {
                8
            };
            render.push_matrix();
            render.translate(*point);
            render.scale(1., 1., zinv);
            render.sphere(drawsize, steps * 2, steps);
            render.pop_matrix();
        }
    }
}

/// `imodPointGetSize`, used directly by the source sphere loop.
pub fn imod_point_get_size(obj: &Iobj, cont: &Icont, point: usize) -> f32 {
    cont.sizes
        .get(point)
        .copied()
        .unwrap_or(obj.pdrawsize as f32)
}

/// Static `imodDrawMesh`.
pub fn imod_draw_mesh(
    obj: &Iobj,
    object_number: i32,
    state: &mut ModelDrawState,
    render: &mut dyn ModelDrawBoundary,
) {
    let mut resol = 0;
    imod_mesh_nearest_res(&obj.mesh, obj.mesh.len() as i32, 0, &mut resol);
    if render.manage_paired_meshes(obj, object_number) {
        return;
    }
    for mesh in &obj.mesh {
        if imesh_resol(mesh.flag) != resol
            || imesh_thickness(mesh.flag) != obj.mesh_thickness as i32
        {
            continue;
        }
        let mut def = DrawProps::default();
        let mut cur = DrawProps::default();
        let mut flags = 0;
        let mut changes = 0;
        let mut cursor = 0usize;
        let mut next_item = crate::imod::libimod::istore::istore_first_change_index(&mesh.store);
        let mut next = next_item;
        ifg_handle_surf_change(
            obj,
            mesh.surf as i32,
            &mut def,
            &mut cur,
            &mut flags,
            0,
            &state.values,
            render,
        );
        render.color3f(cur.red, cur.green, cur.blue);
        let mut i = 0usize;
        while i < mesh.list.len() {
            match mesh.list[i] {
                IMOD_MESH_BGNPOLY | IMOD_MESH_BGNBIGPOLY | IMOD_MESH_BGNPOLYNORM => {
                    while i < mesh.list.len() && mesh.list[i] != IMOD_MESH_ENDPOLY {
                        i += 1;
                    }
                }
                IMOD_MESH_BGNPOLYNORM2 => {
                    i += 1;
                    while i < mesh.list.len() && mesh.list[i] != IMOD_MESH_ENDPOLY {
                        if i + 2 >= mesh.list.len() {
                            break;
                        }
                        if next < i as i32 || next > i as i32 + 2 {
                            render.begin(GL_LINE_LOOP);
                            for _ in 0..3 {
                                if let Some(p) = mesh.vert.get(mesh.list[i] as usize) {
                                    render.vertex3(*p);
                                }
                                i += 1;
                            }
                            render.end();
                        } else {
                            let first = mesh.vert.get(mesh.list[i] as usize).copied();
                            render.begin(GL_LINE_STRIP);
                            for _ in 0..3 {
                                if flags != 0 || i as i32 == next {
                                    next = ifg_handle_mesh_change(
                                        obj,
                                        &mesh.store,
                                        &mut cursor,
                                        &def,
                                        &mut cur,
                                        &mut next_item,
                                        i as i32,
                                        &mut flags,
                                        &mut changes,
                                        0,
                                        &state.values,
                                        render,
                                    );
                                    if changes & CHANGED_COLOR != 0 {
                                        render.color3f(cur.red, cur.green, cur.blue);
                                    }
                                }
                                if let Some(p) = mesh.vert.get(mesh.list[i] as usize) {
                                    render.vertex3(*p);
                                }
                                i += 1;
                            }
                            if let Some(p) = first {
                                render.vertex3(p);
                            }
                            render.end();
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }
}

/// `imodDrawSymbol`.
pub fn imod_draw_symbol(
    point: &Ipoint,
    sym: i32,
    size: i32,
    flags: i32,
    linewidth: i32,
    render: &mut dyn ModelDrawBoundary,
) {
    match sym {
        IOBJ_SYM_CIRCLE => {
            let outer = size as f64;
            let inner = if flags as u32 & IOBJ_SYMF_FILL != 0 {
                0.
            } else {
                outer - linewidth as f64
            };
            render.push_matrix();
            render.translate(*point);
            render.disk(inner, outer, size + 4, 1);
            render.pop_matrix();
        }
        IOBJ_SYM_SQUARE => {
            let mut v = *point;
            v.x -= size as f32 / 2.;
            v.y -= size as f32 / 2.;
            render.begin(if flags as u32 & IOBJ_SYMF_FILL != 0 {
                GL_POLYGON
            } else {
                GL_LINE_LOOP
            });
            render.vertex3(v);
            v.x += size as f32;
            render.vertex3(v);
            v.y += size as f32;
            render.vertex3(v);
            v.x -= size as f32;
            render.vertex3(v);
            render.end();
        }
        IOBJ_SYM_TRIANGLE => {
            let mut v = *point;
            v.y += size as f32;
            render.begin(if flags as u32 & IOBJ_SYMF_FILL != 0 {
                GL_POLYGON
            } else {
                GL_LINE_LOOP
            });
            render.vertex3(v);
            v.x += size as f32;
            v.y -= (size + size / 2) as f32;
            render.vertex3(v);
            v.x -= (2 * size) as f32;
            render.vertex3(v);
            render.end();
        }
        IOBJ_SYM_NONE => {
            render.begin(GL_POINTS);
            render.vertex3(*point);
            render.end();
        }
        IOBJ_SYM_STAR => {}
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct R {
        verts: Vec<Ipoint>,
    }
    impl FinegrainRenderBoundary for R {
        fn color3f(&mut self, _: f32, _: f32, _: f32) {}
        fn color4f(&mut self, _: f32, _: f32, _: f32, _: f32) {}
        fn line_width(&mut self, _: i32, _: &Iobj) {}
        fn point_size(&mut self, _: i32, _: &Iobj) {}
        fn light_adjust(&mut self, _: &Iobj, _: f32, _: f32, _: f32, _: i32) {}
        fn rgba(&self) -> bool {
            true
        }
    }
    impl ModelDrawBoundary for R {
        fn set_object_color(&mut self, _: i32) {}
        fn color_index(&mut self, _: i32) {}
        fn begin(&mut self, _: u32) {}
        fn end(&mut self) {}
        fn vertex3(&mut self, p: Ipoint) {
            self.verts.push(p)
        }
        fn push_matrix(&mut self) {}
        fn pop_matrix(&mut self) {}
        fn translate(&mut self, _: Ipoint) {}
        fn scale(&mut self, _: f32, _: f32, _: f32) {}
        fn sphere(&mut self, _: f64, _: i32, _: i32) {}
        fn disk(&mut self, _: f64, _: f64, _: i32, _: i32) {}
        fn set_stipple(&mut self, _: bool) {}
        fn time_mismatch(&self, _: &ImodView, _: &Iobj, _: &Icont) -> bool {
            false
        }
        fn draw_labels(&mut self, _: &Imod, _: &Iobj, _: i32, _: f32, _: i32) {}
        fn cleanup_label_font(&mut self) {}
        fn manage_paired_meshes(&mut self, _: &Iobj, _: i32) -> bool {
            false
        }
    }
    #[test]
    fn contour_loop_repeats_first_vertex() {
        let mut obj = Iobj::default();
        obj.cont.push(Icont {
            pts: vec![
                Ipoint {
                    x: 1.,
                    ..Default::default()
                },
                Ipoint {
                    x: 2.,
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let mut r = R { verts: vec![] };
        imod_draw_contour_lines(
            &ImodView::default(),
            &obj,
            0,
            0,
            GL_LINE_LOOP,
            &mut ModelDrawState::default(),
            &mut r,
        );
        assert_eq!(r.verts.len(), 3);
        assert_eq!(r.verts[0], r.verts[2]);
    }
    #[test]
    fn point_size_prefers_contour_size() {
        let o = Iobj {
            pdrawsize: 5,
            ..Default::default()
        };
        let c = Icont {
            sizes: vec![7.],
            ..Default::default()
        };
        assert_eq!(imod_point_get_size(&o, &c, 0), 7.);
        assert_eq!(imod_point_get_size(&o, &c, 1), 5.);
    }
}
