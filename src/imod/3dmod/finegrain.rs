//! Translation of `IMOD/3dmod/finegrain.cpp` and `finegrain.h`.
//!
//! The general-store/property calculations are native Rust.  The two places
//! which touch the 3dmod editor and its compatibility OpenGL context are kept
//! as narrow, named boundaries, rather than being silently omitted.
#![allow(dead_code)]

use crate::imod::libimod::imodel::{Icont, Imesh, Iobj};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_FCOLOR, IMOD_OBJFLAG_MCOLOR, IMOD_OBJFLAG_USE_VALUE, MATFLAGS2_CONSTANT,
    MATFLAGS2_SKIP_HIGH, MATFLAGS2_SKIP_LOW,
};
use crate::imod::libimod::istore::{
    DrawProps, Istore, StoreUnion, istore_cont_surf_draw_props, istore_default_draw_props,
    istore_first_change_index, istore_get_min_max, istore_next_change,
};

pub const HANDLE_LINE_COLOR: i32 = 1;
pub const HANDLE_MESH_COLOR: i32 = 1 << 1;
pub const HANDLE_MESH_FCOLOR: i32 = 1 << 2;
pub const HANDLE_TRANS: i32 = 1 << 3;
pub const HANDLE_2DWIDTH: i32 = 1 << 4;
pub const HANDLE_3DWIDTH: i32 = 1 << 5;
pub const HANDLE_VALUE1: i32 = 1 << 6;
pub const GEN_STORE_COLOR: i16 = 1;
pub const GEN_STORE_FCOLOR: i16 = 2;
pub const GEN_STORE_TRANS: i16 = 3;
pub const GEN_STORE_GAP: i16 = 4;
pub const GEN_STORE_CONNECT: i16 = 5;
pub const GEN_STORE_3DWIDTH: i16 = 6;
pub const GEN_STORE_2DWIDTH: i16 = 7;
pub const GEN_STORE_SYMTYPE: i16 = 8;
pub const GEN_STORE_SYMSIZE: i16 = 9;
pub const GEN_STORE_VALUE1: i16 = 10;
pub const GEN_STORE_NO_CAP: i16 = 24;
pub const GEN_STORE_NOINDEX: u16 = 1 << 4;
pub const GEN_STORE_REVERT: u16 = 1 << 5;
pub const GEN_STORE_SURFACE: u16 = 1 << 6;
pub const GEN_STORE_ONEPOINT: u16 = 1 << 7;
pub const CHANGED_COLOR: i32 = 1;
pub const CHANGED_FCOLOR: i32 = 1 << 1;
pub const CHANGED_TRANS: i32 = 1 << 2;
pub const CHANGED_GAP: i32 = 1 << 3;
pub const CHANGED_CONNECT: i32 = 1 << 4;
pub const CHANGED_3DWIDTH: i32 = 1 << 5;
pub const CHANGED_2DWIDTH: i32 = 1 << 6;
pub const CHANGED_SYMTYPE: i32 = 1 << 7;
pub const CHANGED_SYMSIZE: i32 = 1 << 8;
pub const CHANGED_VALUE1: i32 = 1 << 9;

/// The source's GL and `light_adjust` calls.
pub trait FinegrainRenderBoundary {
    fn color3f(&mut self, red: f32, green: f32, blue: f32);
    fn color4f(&mut self, red: f32, green: f32, blue: f32, alpha: f32);
    fn line_width(&mut self, width: i32, object: &Iobj);
    fn point_size(&mut self, size: i32, object: &Iobj);
    fn light_adjust(&mut self, object: &Iobj, red: f32, green: f32, blue: f32, trans: i32);
    fn rgba(&self) -> bool;
}

/// File-static state `sValMin`, `sValSlope`, `sSkipLow`, `sSkipHigh`,
/// `sValueCmap`, `sValSetup`, and `sValConstant`.
#[derive(Clone, Debug)]
pub struct FinegrainValueState {
    pub val_min: f32,
    pub val_slope: f32,
    pub skip_low: i32,
    pub skip_high: i32,
    pub value_cmap: [[u8; 256]; 3],
    pub val_setup: i32,
    pub val_constant: i32,
    pub false_color_cmap: [[u8; 256]; 3],
    pub false_color_initialized: bool,
}
impl Default for FinegrainValueState {
    fn default() -> Self {
        Self {
            val_min: 0.,
            val_slope: 0.,
            skip_low: 0,
            skip_high: 255,
            value_cmap: [[0; 256]; 3],
            val_setup: 0,
            val_constant: 0,
            false_color_cmap: [[0; 256]; 3],
            false_color_initialized: false,
        }
    }
}

/// Original file-static `fgd` (`FgData`).  Upstream stores borrowed IMOD
/// pointers here; indices retain the same selection identity without making
/// Rust's model ownership unsafe.
#[derive(Clone, Debug)]
pub struct FgData {
    pub dialog_open: bool,
    pub pt_cont_surf: i32,
    pub pt_loaded: i32,
    pub cont_loaded: i32,
    pub obj_loaded: i32,
    pub surf_loaded: i32,
    pub state_flags: i32,
    pub range_end: i32,
    pub show_connects: i32,
    pub stipple_gaps: i32,
    pub change_all: i32,
    pub last_change_type: i32,
    pub cont_props: DrawProps,
    pub store: Istore,
}
impl Default for FgData {
    fn default() -> Self {
        Self {
            dialog_open: false,
            pt_cont_surf: 0,
            pt_loaded: -1,
            cont_loaded: -1,
            obj_loaded: -1,
            surf_loaded: -1,
            state_flags: 0,
            range_end: -1,
            show_connects: 0,
            stipple_gaps: 0,
            change_all: 0,
            last_change_type: -1,
            cont_props: DrawProps::default(),
            store: Istore::default(),
        }
    }
}

/// Editor/dialog, selection, undo, and redraw calls made directly by
/// `finegrain.cpp`.  They remain explicit until the editor's model ownership
/// is entirely native Rust.
pub trait FinegrainControllerBoundary {
    fn open_dialog(&mut self);
    fn raise_dialog(&mut self);
    fn close_dialog(&mut self);
    fn update_dialog(
        &mut self,
        pt_cont_surf: i32,
        enabled: i32,
        props: &DrawProps,
        flags: i32,
        next: bool,
        previous: bool,
    );
    fn apply_last_change(&mut self, change: i32) -> i32;
    fn goto_next_change(&mut self, previous: bool, mode: i32);
    fn mutate_store(&mut self, mode: i32, type_: i16, store: &Istore, clear: bool);
    fn redraw(&mut self);
    fn set_xyz_mouse(&mut self);
    fn dump_store(&mut self, mode: i32);
}

/// `fineGrainOpen`.
pub fn fine_grain_open(data: &mut FgData, boundary: &mut dyn FinegrainControllerBoundary) {
    if data.dialog_open {
        boundary.raise_dialog();
        return;
    }
    data.dialog_open = true;
    boundary.open_dialog();
    fine_grain_update(data, boundary);
}
/// `fineGrainUpdate`.  The upstream model lookup is represented by its named
/// controller boundary; retained state and enable calculation are identical.
pub fn fine_grain_update(data: &mut FgData, boundary: &mut dyn FinegrainControllerBoundary) {
    if !data.dialog_open {
        return;
    }
    let enabled = if data.pt_cont_surf == 2 && data.surf_loaded >= 0 {
        1
    } else if (data.pt_cont_surf == 0 && data.pt_loaded >= 0)
        || (data.pt_cont_surf != 0 && data.cont_loaded >= 0)
    {
        2
    } else {
        0
    };
    boundary.update_dialog(
        data.pt_cont_surf,
        enabled,
        &data.cont_props,
        data.state_flags,
        false,
        false,
    );
}
/// `fineGrainApplyLast`.
pub fn fine_grain_apply_last(data: &FgData, boundary: &mut dyn FinegrainControllerBoundary) -> i32 {
    if !data.dialog_open {
        0
    } else {
        boundary.apply_last_change(data.last_change_type)
    }
}
/// `ifgShowConnections`.
pub fn ifg_show_connections(data: &FgData) -> i32 {
    data.show_connects
}
/// `ifgStippleGaps`.
pub fn ifg_stipple_gaps(data: &FgData) -> i32 {
    data.stipple_gaps
}
/// `ifgGetChangeAll`.
pub fn ifg_get_change_all(data: &FgData) -> i32 {
    data.change_all
}
/// `ifgPtContSurfSelected`.
pub fn ifg_pt_cont_surf_selected(
    data: &mut FgData,
    which: i32,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.pt_cont_surf = which;
    fine_grain_update(data, boundary);
}
/// `ifgChangeAllToggled`.
pub fn ifg_change_all_toggled(data: &mut FgData, state: bool) {
    data.change_all = state as i32;
}
/// `ifgGotoNextChange`.
pub fn ifg_goto_next_change(
    data: &FgData,
    previous: bool,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    boundary.goto_next_change(previous, data.pt_cont_surf);
    boundary.set_xyz_mouse();
}
/// Static `insertAndUpdate`.
pub fn insert_and_update(
    data: &mut FgData,
    type_: i16,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.store.type_ = type_;
    if data.pt_cont_surf == 2 {
        data.store.flags |= GEN_STORE_SURFACE;
    }
    boundary.mutate_store(data.pt_cont_surf, type_, &data.store, false);
    boundary.set_xyz_mouse();
    fine_grain_update(data, boundary);
}
/// `ifgColorChanged`.
pub fn ifg_color_changed(
    data: &mut FgData,
    type_: i16,
    red: i32,
    green: i32,
    blue: i32,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.store.value = StoreUnion::from_b([red as u8, green as u8, blue as u8, 0]);
    data.store.flags = 3 << 2;
    insert_and_update(data, type_, boundary);
}
/// `ifgLineColorChanged`.
pub fn ifg_line_color_changed(
    data: &mut FgData,
    red: i32,
    green: i32,
    blue: i32,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.last_change_type = 0;
    ifg_color_changed(data, GEN_STORE_COLOR, red, green, blue, boundary);
}
/// `ifgFillColorChanged`.
pub fn ifg_fill_color_changed(
    data: &mut FgData,
    red: i32,
    green: i32,
    blue: i32,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.last_change_type = 1;
    ifg_color_changed(data, GEN_STORE_FCOLOR, red, green, blue, boundary);
}
/// `ifgIntChanged`.
pub fn ifg_int_changed(
    data: &mut FgData,
    type_: i16,
    value: i32,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.store.value = StoreUnion::from_i(value);
    data.store.flags = 0;
    insert_and_update(data, type_, boundary);
}
/// `ifgTransChanged`.
pub fn ifg_trans_changed(
    data: &mut FgData,
    value: i32,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.last_change_type = 2;
    ifg_int_changed(data, GEN_STORE_TRANS, value, boundary);
}
/// `ifgWidth2DChanged`.
pub fn ifg_width_2d_changed(
    data: &mut FgData,
    value: i32,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.last_change_type = 3;
    ifg_int_changed(data, GEN_STORE_2DWIDTH, value, boundary);
}
/// `ifgWidth3DChanged`.
pub fn ifg_width_3d_changed(
    data: &mut FgData,
    value: i32,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.last_change_type = 4;
    ifg_int_changed(data, GEN_STORE_3DWIDTH, value, boundary);
}
/// `ifgSymsizeChanged`.
pub fn ifg_symsize_changed(
    data: &mut FgData,
    value: i32,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.last_change_type = 5;
    ifg_int_changed(data, GEN_STORE_SYMSIZE, value, boundary);
}
/// `ifgSymtypeChanged`.
pub fn ifg_symtype_changed(
    data: &mut FgData,
    mut symtype: i32,
    filled: bool,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    if filled {
        symtype = -1 - symtype;
    }
    data.store.value = StoreUnion::from_i(symtype);
    data.store.flags = 0;
    data.last_change_type = 6;
    insert_and_update(data, GEN_STORE_SYMTYPE, boundary);
}
/// `ifgEndChange`.
pub fn ifg_end_change(
    data: &mut FgData,
    type_: i16,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    boundary.mutate_store(data.pt_cont_surf, type_, &data.store, false);
    boundary.set_xyz_mouse();
    fine_grain_update(data, boundary);
}
/// `ifgClearChange`.
pub fn ifg_clear_change(
    data: &mut FgData,
    type_: i16,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    boundary.mutate_store(data.pt_cont_surf, type_, &data.store, true);
    boundary.redraw();
    fine_grain_update(data, boundary);
}
/// `ifgGapChanged`.
pub fn ifg_gap_changed(
    data: &mut FgData,
    state: bool,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    if state {
        ifg_int_changed(data, GEN_STORE_GAP, 1, boundary);
    } else {
        ifg_clear_change(data, GEN_STORE_GAP, boundary);
    }
}
/// `ifgNoCapChanged`.
pub fn ifg_no_cap_changed(
    data: &mut FgData,
    state: bool,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    if state {
        ifg_int_changed(data, GEN_STORE_NO_CAP, 1, boundary);
    } else {
        ifg_clear_change(data, GEN_STORE_NO_CAP, boundary);
    }
}
/// `ifgConnectChanged`.
pub fn ifg_connect_changed(
    data: &mut FgData,
    value: i32,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    if value != 0 {
        ifg_int_changed(data, GEN_STORE_CONNECT, value, boundary);
    } else {
        ifg_clear_change(data, GEN_STORE_CONNECT, boundary);
    }
}
/// `ifgShowConnectChanged`.
pub fn ifg_show_connect_changed(
    data: &mut FgData,
    state: bool,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.show_connects = state as i32;
    boundary.redraw();
}
/// `ifgStippleGapsChanged`.
pub fn ifg_stipple_gaps_changed(
    data: &mut FgData,
    state: bool,
    boundary: &mut dyn FinegrainControllerBoundary,
) {
    data.stipple_gaps = state as i32;
    boundary.redraw();
}
/// `ifgDump`.
pub fn ifg_dump(data: &FgData, boundary: &mut dyn FinegrainControllerBoundary) {
    boundary.dump_store(data.pt_cont_surf);
}
/// `ifgClosing`.
pub fn ifg_closing(data: &mut FgData, boundary: &mut dyn FinegrainControllerBoundary) {
    if data.dialog_open {
        boundary.close_dialog();
    }
    data.dialog_open = false;
}

/// Static `getLoadedObjCont`.  Rubber-band selection and undo notifications
/// belong to the editor boundary; this is the source's model-index validity
/// portion, including the range sentinel reset.
pub fn get_loaded_obj_cont(data: &mut FgData, objects: &[Iobj], _add_type: i16) -> i32 {
    data.range_end = -1;
    let Some(object) = objects.get(data.obj_loaded.max(0) as usize) else {
        return 1;
    };
    if data.obj_loaded < 0 || data.cont_loaded >= object.cont.len() as i32 {
        return 1;
    }
    if data.pt_cont_surf == 2 {
        return 0;
    }
    let Some(contour) = object.cont.get(data.cont_loaded.max(0) as usize) else {
        return 1;
    };
    if data.cont_loaded < 0
        || (data.pt_cont_surf == 0 && data.pt_loaded >= contour.pts.len() as i32)
    {
        return 1;
    }
    0
}

/// `ifgSelectedLineWidth`.
pub fn ifg_selected_line_width(width: i32, selected: i32) -> i32 {
    if selected != 0 {
        if width < 3 { width + 2 } else { width / 2 }
    } else {
        width
    }
}

/// Static `ifgHandleColorTrans`.
pub fn ifg_handle_color_trans(
    object: &Iobj,
    red: f32,
    green: f32,
    blue: f32,
    trans: i32,
    render: &mut dyn FinegrainRenderBoundary,
) {
    render.color4f(red, green, blue, 1.0 - trans as f32 * 0.01);
    render.light_adjust(object, red, green, blue, trans);
}

/// Static `ifgHandleValue1`.
pub fn ifg_handle_value1(
    state: &FinegrainValueState,
    def_props: &DrawProps,
    cont_props: &mut DrawProps,
    state_flags: &mut i32,
    change_flags: &mut i32,
) {
    cont_props.valskip = 0;
    if state.val_constant != 0 {
        if *state_flags & CHANGED_VALUE1 != 0 {
            let index = (state.val_slope * (cont_props.value1 - state.val_min)) as i32;
            if index < state.skip_low || index > state.skip_high {
                cont_props.gap = 1;
                cont_props.valskip = 1;
            }
        }
        return;
    }
    if *state_flags & CHANGED_VALUE1 != 0 {
        let index = (state.val_slope * (cont_props.value1 - state.val_min)) as i32;
        let index = index.clamp(0, 255) as usize;
        if (index as i32) < state.skip_low || (index as i32) > state.skip_high {
            cont_props.gap = 1;
            cont_props.valskip = 1;
        }
        cont_props.red = state.value_cmap[0][index] as f32 / 255.;
        cont_props.green = state.value_cmap[1][index] as f32 / 255.;
        cont_props.blue = state.value_cmap[2][index] as f32 / 255.;
    } else {
        cont_props.red = def_props.red;
        cont_props.green = def_props.green;
        cont_props.blue = def_props.blue;
    }
    *change_flags |= CHANGED_COLOR;
}

/// Static `handleContChange`.
pub fn handle_cont_change(
    object: &Iobj,
    contour: i32,
    surface: i32,
    cont_props: &mut DrawProps,
    pt_props: &mut DrawProps,
    state_flags: &mut i32,
    handle_flags: i32,
    selected: i32,
    scale_thick: i32,
    value: &FinegrainValueState,
    render: &mut dyn FinegrainRenderBoundary,
) {
    let mut def_props = DrawProps::default();
    let mut cont_state = 0;
    let mut surf_state = 0;
    istore_default_draw_props(object, &mut def_props);
    *state_flags = istore_cont_surf_draw_props(
        &object.store,
        &def_props,
        cont_props,
        contour,
        surface,
        &mut cont_state,
        &mut surf_state,
    );
    if handle_flags & HANDLE_VALUE1 != 0 && *state_flags & CHANGED_VALUE1 != 0 {
        // C passes `stateFlags` for both mutable parameters.  Rust makes the
        // alias explicit while preserving the resulting OR operation.
        let mut changes = *state_flags;
        ifg_handle_value1(value, &def_props, cont_props, state_flags, &mut changes);
        *state_flags = changes;
    }
    *pt_props = *cont_props;
    pt_props.gap = 0;
    pt_props.valskip = 0;
    if handle_flags & HANDLE_LINE_COLOR != 0 && render.rgba() {
        render.color3f(pt_props.red, pt_props.green, pt_props.blue);
    }
    if handle_flags & HANDLE_MESH_COLOR != 0 {
        ifg_handle_color_trans(
            object,
            pt_props.red,
            pt_props.green,
            pt_props.blue,
            pt_props.trans,
            render,
        );
    }
    if handle_flags & HANDLE_MESH_FCOLOR != 0 {
        ifg_handle_color_trans(
            object,
            pt_props.fill_red,
            pt_props.fill_green,
            pt_props.fill_blue,
            pt_props.trans,
            render,
        );
    }
    if handle_flags & HANDLE_2DWIDTH != 0 {
        render.line_width(
            ifg_selected_line_width(scale_thick * pt_props.linewidth2, selected),
            object,
        );
    }
    if handle_flags & HANDLE_3DWIDTH != 0 {
        render.line_width(pt_props.linewidth, object);
        render.point_size(pt_props.linewidth, object);
    }
}

/// `ifgHandleSurfChange`.
pub fn ifg_handle_surf_change(
    object: &Iobj,
    surface: i32,
    cont_props: &mut DrawProps,
    pt_props: &mut DrawProps,
    state_flags: &mut i32,
    handle_flags: i32,
    value: &FinegrainValueState,
    render: &mut dyn FinegrainRenderBoundary,
) {
    handle_cont_change(
        object,
        -1,
        surface,
        cont_props,
        pt_props,
        state_flags,
        handle_flags,
        0,
        0,
        value,
        render,
    );
}

/// `ifgHandleContChange`.
pub fn ifg_handle_cont_change(
    object: &Iobj,
    contour: i32,
    cont_props: &mut DrawProps,
    pt_props: &mut DrawProps,
    state_flags: &mut i32,
    handle_flags: i32,
    selected: i32,
    scale_thick: i32,
    value: &FinegrainValueState,
    render: &mut dyn FinegrainRenderBoundary,
) -> i32 {
    let surface = object.cont[contour as usize].surf;
    handle_cont_change(
        object,
        contour,
        surface,
        cont_props,
        pt_props,
        state_flags,
        handle_flags,
        selected,
        scale_thick,
        value,
        render,
    );
    istore_first_change_index(&object.cont[contour as usize].store)
}

/// Static `ifgHandleStateChange`.
pub fn ifg_handle_state_change(
    object: &Iobj,
    def_props: &DrawProps,
    pt_props: &mut DrawProps,
    state_flags: &mut i32,
    change_flags: &mut i32,
    handle_flags: i32,
    selected: i32,
    scale_thick: i32,
    value: &FinegrainValueState,
    render: &mut dyn FinegrainRenderBoundary,
) {
    if handle_flags & HANDLE_VALUE1 != 0 && *change_flags & CHANGED_VALUE1 != 0 {
        ifg_handle_value1(value, def_props, pt_props, state_flags, change_flags);
    }
    if handle_flags & HANDLE_LINE_COLOR != 0 && *change_flags & CHANGED_COLOR != 0 && render.rgba()
    {
        render.color3f(pt_props.red, pt_props.green, pt_props.blue);
    }
    if handle_flags & HANDLE_MESH_COLOR != 0 && *change_flags & (CHANGED_COLOR | CHANGED_TRANS) != 0
    {
        ifg_handle_color_trans(
            object,
            pt_props.red,
            pt_props.green,
            pt_props.blue,
            pt_props.trans,
            render,
        );
    }
    if handle_flags & HANDLE_MESH_FCOLOR != 0
        && *change_flags & (CHANGED_FCOLOR | CHANGED_TRANS) != 0
    {
        ifg_handle_color_trans(
            object,
            pt_props.fill_red,
            pt_props.fill_green,
            pt_props.fill_blue,
            pt_props.trans,
            render,
        );
    }
    if handle_flags & HANDLE_3DWIDTH != 0 && *change_flags & CHANGED_3DWIDTH != 0 {
        render.line_width(pt_props.linewidth, object);
        render.point_size(pt_props.linewidth, object);
    }
    if handle_flags & HANDLE_2DWIDTH != 0 && *change_flags & CHANGED_2DWIDTH != 0 {
        render.line_width(
            ifg_selected_line_width(scale_thick * pt_props.linewidth2, selected),
            object,
        );
    }
}

/// `ifgHandleNextChange`.  `cursor` is the Rust ownership equivalent of the
/// upstream `Ilist::current` cursor.
pub fn ifg_handle_next_change(
    object: &Iobj,
    list: &[Istore],
    cursor: &mut usize,
    def_props: &DrawProps,
    pt_props: &mut DrawProps,
    state_flags: &mut i32,
    change_flags: &mut i32,
    handle_flags: i32,
    selected: i32,
    scale_thick: i32,
    value: &FinegrainValueState,
    render: &mut dyn FinegrainRenderBoundary,
) -> i32 {
    let next_change =
        istore_next_change(list, cursor, def_props, pt_props, state_flags, change_flags);
    ifg_handle_state_change(
        object,
        def_props,
        pt_props,
        state_flags,
        change_flags,
        handle_flags,
        selected,
        scale_thick,
        value,
        render,
    );
    next_change
}

/// `ifgHandleMeshChange`.
pub fn ifg_handle_mesh_change(
    object: &Iobj,
    list: &[Istore],
    cursor: &mut usize,
    def_props: &DrawProps,
    cur_props: &mut DrawProps,
    next_item_index: &mut i32,
    cur_index: i32,
    state_flags: &mut i32,
    change_flags: &mut i32,
    handle_flags: i32,
    value: &FinegrainValueState,
    render: &mut dyn FinegrainRenderBoundary,
) -> i32 {
    let need_change = *state_flags;
    *state_flags = 0;
    if cur_index == *next_item_index {
        let last_props = *cur_props;
        *cur_props = *def_props;
        *next_item_index = istore_next_change(
            list,
            cursor,
            def_props,
            cur_props,
            state_flags,
            change_flags,
        );
        if handle_flags & HANDLE_VALUE1 != 0 && *change_flags & CHANGED_VALUE1 != 0 {
            ifg_handle_value1(value, def_props, cur_props, state_flags, change_flags);
            *state_flags = *change_flags;
        }
        *change_flags |= need_change;
        if *state_flags & CHANGED_COLOR != 0
            && cur_props.red == last_props.red
            && cur_props.green == last_props.green
            && cur_props.blue == last_props.blue
        {
            *change_flags &= !CHANGED_COLOR;
        }
        if *state_flags & CHANGED_FCOLOR != 0
            && cur_props.fill_red == last_props.fill_red
            && cur_props.fill_green == last_props.fill_green
            && cur_props.fill_blue == last_props.fill_blue
        {
            *change_flags &= !CHANGED_FCOLOR;
        }
        if *state_flags & CHANGED_TRANS != 0 && cur_props.trans == last_props.trans {
            *change_flags &= !CHANGED_TRANS;
        }
        if *state_flags & CHANGED_3DWIDTH != 0 && cur_props.linewidth == last_props.linewidth {
            *change_flags &= !CHANGED_3DWIDTH;
        }
    } else {
        *change_flags = need_change;
        *cur_props = *def_props;
    }
    if handle_flags & HANDLE_MESH_COLOR != 0 && *change_flags & (CHANGED_COLOR | CHANGED_TRANS) != 0
    {
        ifg_handle_color_trans(
            object,
            cur_props.red,
            cur_props.green,
            cur_props.blue,
            cur_props.trans,
            render,
        );
    }
    if handle_flags & HANDLE_MESH_FCOLOR != 0
        && *change_flags & (CHANGED_FCOLOR | CHANGED_TRANS) != 0
    {
        ifg_handle_color_trans(
            object,
            cur_props.fill_red,
            cur_props.fill_green,
            cur_props.fill_blue,
            cur_props.trans,
            render,
        );
    }
    if handle_flags & HANDLE_3DWIDTH != 0 && *change_flags & CHANGED_3DWIDTH != 0 {
        render.line_width(cur_props.linewidth, object);
        render.point_size(cur_props.linewidth, object);
    }
    if *state_flags != 0 {
        cur_index + 1
    } else {
        *next_item_index
    }
}

/// `ifgContTransMatch`.
pub fn ifg_cont_trans_match(
    object: &Iobj,
    contour: &Icont,
    cursor: &mut usize,
    match_pt: &mut i32,
    draw_trans: i32,
    cont_props: &DrawProps,
    pt_props: &mut DrawProps,
    state_flags: &mut i32,
    all_changes: &mut i32,
    handle_flags: i32,
    value: &FinegrainValueState,
    render: &mut dyn FinegrainRenderBoundary,
) -> i32 {
    *match_pt = contour.pts.len() as i32;
    *all_changes = 0;
    if contour.store.is_empty() {
        return -1;
    }
    let current = *cursor;
    for store in contour.store.iter().skip(current) {
        if store.flags & (GEN_STORE_NOINDEX | 3) != 0
            || (store.index.i()) >= contour.pts.len() as i32
        {
            break;
        }
        let revert = store.flags & GEN_STORE_REVERT != 0;
        if store.type_ == GEN_STORE_TRANS
            && ((!revert && (((store.value.i()) != 0) as i32) == draw_trans)
                || (revert && ((cont_props.trans != 0) as i32) == draw_trans))
        {
            *match_pt = (store.index.i());
            *cursor = current;
            let mut next_change = 0;
            while next_change >= 0 && next_change <= *match_pt {
                let mut changes = 0;
                next_change = istore_next_change(
                    &contour.store,
                    cursor,
                    cont_props,
                    pt_props,
                    state_flags,
                    &mut changes,
                );
                *all_changes |= changes;
            }
            ifg_handle_state_change(
                object,
                cont_props,
                pt_props,
                state_flags,
                all_changes,
                handle_flags,
                0,
                0,
                value,
                render,
            );
            return next_change;
        }
    }
    -1
}

/// `ifgMeshTransMatch`.
pub fn ifg_mesh_trans_match(
    mesh: &Imesh,
    cursor: &mut usize,
    def_trans: i32,
    draw_trans: i32,
    mesh_ind: &mut i32,
    skip_ends: i32,
) -> i32 {
    if mesh.store.is_empty() {
        return -1;
    }
    let mut index = -1;
    let mut current = *cursor;
    if def_trans != draw_trans {
        for i in *cursor..mesh.store.len() {
            let st = &mesh.store[i];
            if st.flags & (GEN_STORE_NOINDEX | 3) != 0 {
                break;
            }
            if (st.index.i()) != index {
                index = (st.index.i());
                current = i;
            }
            if st.type_ == GEN_STORE_TRANS && (((st.value.i()) != 0) as i32) == draw_trans {
                *mesh_ind = index;
                *cursor = current;
                return index;
            }
        }
        *mesh_ind = mesh.list.len() as i32 - 2;
    } else {
        let mut trans_set = true;
        for i in *cursor..mesh.store.len() {
            let st = &mesh.store[i];
            if st.flags & (GEN_STORE_NOINDEX | 3) != 0 {
                break;
            }
            if (st.index.i()) != index {
                if !trans_set {
                    *cursor = current;
                    return index;
                }
                index = (st.index.i());
                current = i;
                trans_set = false;
                *mesh_ind += 1;
                while *mesh_ind < index && *mesh_ind < mesh.list.len() as i32 - 2 {
                    if mesh.list[*mesh_ind as usize] >= 0 || skip_ends == 0 {
                        return index;
                    }
                    *mesh_ind += 1;
                }
                if *mesh_ind >= mesh.list.len() as i32 - 2 {
                    *mesh_ind = mesh.list.len() as i32 - 2;
                    return -1;
                }
            }
            if st.type_ == GEN_STORE_TRANS {
                trans_set = true;
            }
            if st.type_ == GEN_STORE_TRANS && (((st.value.i()) != 0) as i32) == draw_trans {
                *cursor = current;
                return index;
            }
        }
        if !trans_set {
            *cursor = current;
            return index;
        }
        *mesh_ind += 1;
        if *mesh_ind >= mesh.list.len() as i32 - 2 {
            *mesh_ind = mesh.list.len() as i32 - 2;
        }
    };
    -1
}

/// `ifgMapFalseColor`.  `map` is the retained `xcramp_mapfalsecolor` boundary.
pub fn ifg_map_false_color(
    state: &mut FinegrainValueState,
    gray: i32,
    map: &mut dyn FnMut(i32) -> (i32, i32, i32),
) -> (i32, i32, i32) {
    if !state.false_color_initialized {
        for ind in 0..256 {
            let (r, g, b) = map(ind);
            state.false_color_cmap[0][ind as usize] = r as u8;
            state.false_color_cmap[1][ind as usize] = g as u8;
            state.false_color_cmap[2][ind as usize] = b as u8;
        }
        state.false_color_initialized = true;
    }
    let ind = gray as usize;
    (
        state.false_color_cmap[0][ind] as i32,
        state.false_color_cmap[1][ind] as i32,
        state.false_color_cmap[2][ind] as i32,
    )
}

/// `ifgMakeValueMap`.
pub fn ifg_make_value_map(
    object: &Iobj,
    state: &mut FinegrainValueState,
    map: &mut dyn FnMut(i32) -> (i32, i32, i32),
) {
    let mut black = object.valblack as i32;
    let mut white = object.valwhite as i32;
    let mut reverse = false;
    if black > white {
        std::mem::swap(&mut black, &mut white);
        reverse = true;
    }
    let rampsize = (white - black).max(1);
    for i in 0..black {
        state.value_cmap[0][i as usize] = 0;
    }
    for i in white..256 {
        state.value_cmap[0][i as usize] = 255;
    }
    let slope = 256. / rampsize as f32;
    for i in black..white {
        state.value_cmap[0][i as usize] = ((i - black) as f32 * slope) as u8;
    }
    if reverse {
        for i in 0..256 {
            state.value_cmap[0][i] = 255 - state.value_cmap[0][i];
        }
    }
    if object.flags & IMOD_OBJFLAG_MCOLOR != 0 {
        for i in 0..256 {
            let (r, g, b) = ifg_map_false_color(state, state.value_cmap[0][i] as i32, map);
            state.value_cmap[0][i] = r as u8;
            state.value_cmap[1][i] = g as u8;
            state.value_cmap[2][i] = b as u8;
        }
    } else {
        let (r, g, b) = if object.flags & IMOD_OBJFLAG_FCOLOR != 0 {
            (
                object.fillred as f32 / 255.,
                object.fillgreen as f32 / 255.,
                object.fillblue as f32 / 255.,
            )
        } else {
            (object.red, object.green, object.blue)
        };
        for i in 0..256 {
            let magnitude = state.value_cmap[0][i] as f32;
            state.value_cmap[0][i] = (r * magnitude) as u8;
            state.value_cmap[1][i] = (g * magnitude) as u8;
            state.value_cmap[2][i] = (b * magnitude) as u8;
        }
    }
}

/// `ifgSetupValueDrawing`.
pub fn ifg_setup_value_drawing(
    object: &Iobj,
    type_: i16,
    val_const: i32,
    state: &mut FinegrainValueState,
    map: &mut dyn FnMut(i32) -> (i32, i32, i32),
) -> i32 {
    state.val_setup = 0;
    if object.flags & IMOD_OBJFLAG_USE_VALUE == 0 {
        return 0;
    }
    let mut max = 0.;
    if istore_get_min_max(
        &object.store,
        object.cont.len() as i32,
        type_,
        &mut state.val_min,
        &mut max,
    ) == 0
        || max <= state.val_min
    {
        return 0;
    }
    state.val_slope = 255.9 / (max - state.val_min);
    state.val_constant = if val_const < 0 {
        ((object.matflags2 as u32 & MATFLAGS2_CONSTANT) != 0) as i32
    } else {
        val_const
    };
    if state.val_constant == 0 {
        ifg_make_value_map(object, state, map);
    }
    state.skip_low = if object.matflags2 as u32 & MATFLAGS2_SKIP_LOW != 0 {
        object.valblack.min(object.valwhite) as i32
    } else {
        0
    };
    state.skip_high = if object.matflags2 as u32 & MATFLAGS2_SKIP_HIGH != 0 {
        object.valblack.max(object.valwhite) as i32
    } else {
        255
    };
    state.val_setup = 1;
    1
}

/// `ifgGetValueSetupState`.
pub fn ifg_get_value_setup_state(state: &FinegrainValueState) -> i32 {
    state.val_setup
}
/// `ifgResetValueSetup`.
pub fn ifg_reset_value_setup(state: &mut FinegrainValueState) {
    state.val_setup = 0;
}

/// `ifgToggleGap`'s model-store operation, separated from source undo hooks.
pub fn ifg_toggle_gap(contour: &mut Icont, point_index: i32, enabled: bool) -> i32 {
    if enabled {
        let item = Istore {
            type_: GEN_STORE_GAP,
            flags: GEN_STORE_ONEPOINT,
            index: StoreUnion::from_i(point_index),
            value: StoreUnion::from_i(0),
        };
        crate::imod::libimod::istore::istore_add_one_index_item(&mut contour.store, item)
    } else {
        crate::imod::libimod::istore::istore_clear_one_index_item(
            &mut contour.store,
            GEN_STORE_GAP,
            point_index,
            0,
        )
    }
}

/// Static `findNextChange`.
pub fn find_next_change(list: &[Istore], index: i32, surf_flag: i32, previous: bool) -> i32 {
    if list.is_empty() {
        return -1;
    }
    let mut after = list.partition_point(|st| (st.index.i()) <= index);
    if previous && after > list.len() - 1 {
        after = list.len() - 1;
    }
    if previous {
        for st in list[..=after].iter().rev() {
            if (st.index.i()) >= index {
                continue;
            }
            if st.flags & (GEN_STORE_NOINDEX | 3) != 0 {
                return -1;
            }
            if (st.flags & GEN_STORE_SURFACE != 0) == (surf_flag != 0) {
                return (st.index.i());
            }
        }
    } else {
        for st in &list[after..] {
            if st.flags & (GEN_STORE_NOINDEX | 3) != 0 {
                return -1;
            }
            if (st.flags & GEN_STORE_SURFACE != 0) == (surf_flag != 0) {
                return (st.index.i());
            }
        }
    };
    -1
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_width_matches_source() {
        assert_eq!(ifg_selected_line_width(1, 1), 3);
        assert_eq!(ifg_selected_line_width(4, 1), 2);
        assert_eq!(ifg_selected_line_width(4, 0), 4);
    }
    #[test]
    fn value_setup_rejects_disabled_object() {
        let o = Iobj::default();
        let mut s = FinegrainValueState::default();
        assert_eq!(
            ifg_setup_value_drawing(&o, GEN_STORE_VALUE1, -1, &mut s, &mut |i| (i, i, i)),
            0
        );
    }
    #[test]
    fn false_color_initializes_once() {
        let mut s = FinegrainValueState::default();
        assert_eq!(
            ifg_map_false_color(&mut s, 7, &mut |i| (i, i + 1, i + 2)),
            (7, 8, 9)
        );
        assert_eq!(
            ifg_map_false_color(&mut s, 7, &mut |_| (0, 0, 0)),
            (7, 8, 9)
        );
    }
}
