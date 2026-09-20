//! Translation of `IMOD/3dmod/mv_objed.cpp` and `mv_objed.h`.
//!
//! The Qt form controls are data-bound at the `formv_objed.cpp` boundary.  The
//! selection and object mutations below retain the source control semantics so
//! they can be driven by either that form or the native event loop.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::imesh::{
    DEFAULT_VALUE, IMESH_CAP_ALL, IMESH_CAP_OFF, IMESH_MK_CAP_DOME, IMESH_MK_CAP_TUBE,
    IMESH_MK_SKIP, IMESH_MK_SURF, IMESH_MK_TUBE,
};
use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OFF, IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_SCAT, Imod, Iobj, Ipoint,
    imod_get_bounding_box,
};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_ANTI_ALIAS, IMOD_OBJFLAG_EXTRA_EDIT, IMOD_OBJFLAG_FCOLOR_PNT, IMOD_OBJFLAG_FILL,
    IMOD_OBJFLAG_MCOLOR, IMOD_OBJFLAG_MESH, IMOD_OBJFLAG_NOLINE, IMOD_OBJFLAG_PLANAR,
    IMOD_OBJFLAG_SCALAR, IMOD_OBJFLAG_SCALE_WDTH, IMOD_OBJFLAG_THICK_CONT, IMOD_OBJFLAG_TWO_SIDE,
    IMOD_OBJFLAG_USE_VALUE, MATFLAGS2_CONSTANT, MATFLAGS2_SKIP_HIGH, MATFLAGS2_SKIP_LOW,
    imod_object_get_bbox, iobj_close, iobj_open, iobj_scat,
};
use crate::imod::libimod::istore::{GEN_STORE_MINMAX1, istore_get_min_max};
use crate::imod::three_dmod::imodv::{
    ImodvApp, imodv_draw, imodv_finish_chg_unit, imodv_new_model_angles, imodv_register_model_chg,
    imodv_register_object_chg,
};
use crate::imod::three_dmod::imodview::ivw_get_an_extra_object;

pub const STYLE_POINTS: i32 = 0;
pub const STYLE_LINES: i32 = 1;
pub const STYLE_FILL: i32 = 2;
pub const STYLE_FILL_OUTLINE: i32 = 3;

const OBJTYPE_CLOSED: i32 = 1;
const OBJTYPE_OPEN: i32 = 2;
const OBJTYPE_SCAT: i32 = 4;
const WORLD_QUALITY_SHIFT: u32 = 8;
const WORLD_QUALITY_BITS: u32 = 7 << WORLD_QUALITY_SHIFT;

fn mesh_object_synchronously(object: &mut Iobj, scale: &Ipoint, low_resolution: bool) -> i32 {
    // `meshObject` (`mv_objed.cpp:2714`) sets the transient copy/no-warning
    // bits on the duplicate's own parameters, calls `analyzePrepSkinObj`, and
    // then stamps the resolution flag on every mesh it produced.
    if object.mesh_param.is_none() {
        object.mesh_param = Some(Default::default());
    }
    if let Some(params) = object.mesh_param.as_mut() {
        params.flags |= crate::imod::libimod::imesh::IMESH_MK_IS_COPY
            | crate::imod::libimod::imesh::IMESH_MK_NO_WARN;
    }
    let resol = i32::from(low_resolution);
    if crate::imod::libmesh::objprep::analyze_prep_skin_obj(object, resol, scale, None) != 0 {
        return 1;
    }
    for mesh in &mut object.mesh {
        mesh.flag |= (resol as u32) << crate::imod::libimod::imesh::IMESH_FLAG_RES_SHIFT;
    }
    0
}

/// Synchronous equivalent of native `meshOneObject` plus `finishMesh`:
/// mesh a contour-only duplicate, then replace just that resolution in the
/// live object.  A meshing error therefore cannot leave partial geometry in
/// the displayed model.
fn mesh_object_replacing_resolution(
    object: &mut Iobj,
    scale: &Ipoint,
    low_resolution: bool,
) -> i32 {
    let Some(mut duplicate) = crate::imod::libmesh::objprep::imesh_dup_marked_conts(object, 0)
    else {
        return -1;
    };
    duplicate.mesh_param = object
        .mesh_param
        .clone()
        .or_else(|| Some(Default::default()));
    let result = mesh_object_synchronously(&mut duplicate, scale, low_resolution);
    if result != 0 {
        return result;
    }
    let resolution = i32::from(low_resolution);
    let mut meshsize = object.mesh.len() as i32;
    let _ = crate::imod::libmesh::objprep::imod_meshes_delete_res(
        &mut object.mesh,
        &mut meshsize,
        resolution,
    );
    object.mesh.extend(duplicate.mesh);
    0
}

/// Original `ObjectEditField`; widget callbacks are the paired Qt form boundary.
pub struct ObjectEditField {
    pub label: &'static str,
    pub control_present: bool,
}
/// Original `objectEditFieldData` labels, in source order.
pub const OBJECT_EDIT_FIELD_DATA: &[ObjectEditField] = &[
    ObjectEditField {
        label: "Color",
        control_present: false,
    },
    ObjectEditField {
        label: "Fill",
        control_present: false,
    },
    ObjectEditField {
        label: "Material",
        control_present: false,
    },
    ObjectEditField {
        label: "Points",
        control_present: false,
    },
    ObjectEditField {
        label: "Lines",
        control_present: false,
    },
    ObjectEditField {
        label: "Scalar",
        control_present: false,
    },
    ObjectEditField {
        label: "Clip",
        control_present: false,
    },
    ObjectEditField {
        label: "Move",
        control_present: false,
    },
    ObjectEditField {
        label: "Subsets",
        control_present: false,
    },
    ObjectEditField {
        label: "Mesh draw",
        control_present: false,
    },
    ObjectEditField {
        label: "Meshing",
        control_present: false,
    },
];

/// Original `ImodvObjed`, minus QObject/QTimer ownership which belongs to Qt.
#[derive(Clone, Debug)]
pub struct ImodvObjed {
    pub timer_id: i32,
    pub current_panel_frame: i32,
    pub edit_all: i32,
    pub collect_changes: bool,
    pub ctrl_pressed: bool,
    pub multiple_color_ok: bool,
    pub dialog_open: bool,
    pub meshing_object: i32,
    pub mesh_busy: bool,
    pub make_low_res: bool,
    /// `meshMin` / `meshMax` and `meshRmin` / `meshRmax` from the Values
    /// panel.  They are control state, not part of the model.
    pub mesh_min: i32,
    pub mesh_max: i32,
    pub mesh_rmin: f32,
    pub mesh_rmax: f32,
}

impl Default for ImodvObjed {
    fn default() -> Self {
        Self {
            timer_id: 0,
            current_panel_frame: 0,
            edit_all: 0,
            collect_changes: false,
            ctrl_pressed: false,
            multiple_color_ok: false,
            dialog_open: false,
            meshing_object: -1,
            mesh_busy: false,
            make_low_res: false,
            mesh_min: 0,
            mesh_max: 255,
            mesh_rmin: 0.,
            mesh_rmax: 255.,
        }
    }
}
/// Original `MeshColorBar`; rendering is delegated to the active GL/Qt painter.
#[derive(Clone, Debug, Default)]
pub struct MeshColorBar {
    pub width: i32,
    pub height: i32,
}
/// Native constructor for C++ `MeshColorBar(QWidget *)`.
pub fn mesh_color_bar(width: i32, height: i32) -> MeshColorBar {
    MeshColorBar { width, height }
}

/// Original static `objedObject`.
pub fn objed_object(a: &mut ImodvApp) -> Option<&mut Iobj> {
    let count = num_editable_objects(a, a.cur_mod);
    if a.obj_num >= count {
        a.obj_num = (count - 1).max(0);
    }
    let Some(object) = editable_object(a, a.cur_mod, a.obj_num).map(|object| object as *mut Iobj)
    else {
        a.obj = std::ptr::null_mut();
        return None;
    };
    a.obj = object;
    // The cursor is retained by `ImodvApp`, exactly as `Imodv->obj` is in
    // the C implementation.  `editable_object` just produced this live
    // element from either that app's model or image-view extra-object store.
    unsafe { object.as_mut() }
}
/// Original static `numEditableObjects`.
pub fn num_editable_objects(a: &ImodvApp, model: i32) -> i32 {
    if model < 0 {
        return 0;
    }
    let Some(native_model) = (unsafe { a.mod_.get(model as usize).map(|m| m.as_ref()) }) else {
        return 0;
    };
    let base_count = native_model.obj.len() as i32;
    if model != a.cur_mod {
        return base_count;
    }
    let Some(view) = (unsafe { a.vi.as_ref() }) else {
        return base_count;
    };
    base_count
        + (0..view.num_extra_obj)
            .filter(|&index| {
                let index = index as usize;
                view.extra_obj_in_use.get(index).copied().unwrap_or(0) != 0
                    && view
                        .extra_obj
                        .get(index)
                        .is_some_and(|object| object.flags & IMOD_OBJFLAG_EXTRA_EDIT != 0)
            })
            .count() as i32
}
/// Original static `editableObject`.
pub fn editable_object(a: &mut ImodvApp, model: i32, object: i32) -> Option<&mut Iobj> {
    if model < 0 || object < 0 {
        return None;
    }
    let base_count = unsafe {
        a.mod_
            .get(model as usize)
            .map(|native_model| native_model.as_ref().obj.len() as i32)
    }?;
    if model != a.cur_mod || object < base_count {
        return unsafe { a.mod_.get_mut(model as usize).map(|m| m.as_mut()) }
            .and_then(|native_model| native_model.obj.get_mut(object as usize));
    }
    let mut editable_index = base_count;
    // Locate first through a shared view.  Calling `ivw_get_an_extra_object`
    // inside this loop would create a mutable borrow whose successful return
    // can escape, which is the C pointer walk but not a Rust borrow pattern.
    let extra_index = {
        let view = unsafe { a.vi.as_ref() }?;
        let mut found = None;
        for index in 0..view.num_extra_obj {
            let index_usize = index as usize;
            let is_editable = view.extra_obj_in_use.get(index_usize).copied().unwrap_or(0) != 0
                && view
                    .extra_obj
                    .get(index_usize)
                    .is_some_and(|extra| extra.flags & IMOD_OBJFLAG_EXTRA_EDIT != 0);
            if is_editable {
                if editable_index == object {
                    found = Some(index);
                    break;
                }
                editable_index += 1;
            }
        }
        found
    }?;
    let view = unsafe { a.vi.as_mut() }?;
    ivw_get_an_extra_object(view, extra_index)
}
/// Original static `setStartEndModel`.
pub fn set_start_end_model(a: &ImodvApp, multiple_ok: bool, mst: &mut i32, mnd: &mut i32) {
    *mst = a.cur_mod;
    *mnd = a.cur_mod;
    if multiple_ok && a.crosset != 0 {
        *mst = 0;
        *mnd = a.num_mods - 1;
    }
}
/// Original static `changeModelObject`.
pub fn change_model_object(a: &ImodvApp, model: i32, object: i32, multiple_ok: bool) -> bool {
    (model == a.cur_mod || (multiple_ok && a.crosset != 0))
        && object >= 0
        && object < num_editable_objects(a, model)
}
/// Original static `setObjFlag`.
pub fn set_obj_flag(
    a: &mut ImodvApp,
    flag: u32,
    state: i32,
    types: i32,
    registered: bool,
    extra_flag: bool,
) {
    if a.imod.is_null() || objed_object(a).is_none() {
        return;
    }
    let (mut first, mut last) = (0, 0);
    set_start_end_model(a, true, &mut first, &mut last);
    for m in first..=last {
        let count = num_editable_objects(a, m);
        for ob in 0..count {
            if change_model_object(a, m, ob, true) {
                if let Some(obj) = editable_object(a, m, ob) {
                    let object_flags = obj.flags;
                    if types != 0
                        && !((types & OBJTYPE_CLOSED != 0 && iobj_close(object_flags) != 0)
                            || (types & OBJTYPE_OPEN != 0 && iobj_open(object_flags) != 0)
                            || (types & OBJTYPE_SCAT != 0 && iobj_scat(object_flags) != 0))
                    {
                        continue;
                    }
                    if state != 0 {
                        obj.flags |= flag;
                    } else {
                        obj.flags &= !flag;
                    }
                }
            }
        }
    }
    if registered {
        imodv_register_object_chg(a.obj_num);
    }
}
/// Original static `setBitInMatFlags2`.
pub fn set_bit_in_mat_flags_2(a: &mut ImodvApp, flag: u32, state: i32) {
    if a.imod.is_null() || objed_object(a).is_none() {
        return;
    }
    let (mut first, mut last) = (0, 0);
    set_start_end_model(a, true, &mut first, &mut last);
    for model in first..=last {
        let count = num_editable_objects(a, model);
        for object in 0..count {
            if change_model_object(a, model, object, true) {
                if let Some(obj) = editable_object(a, model, object) {
                    imodv_register_object_chg(object);
                    if state != 0 {
                        obj.matflags2 |= flag as u8;
                    } else {
                        obj.matflags2 &= !(flag as u8);
                    }
                }
            }
        }
    }
}
/// Original static `optionSetFlags`.
pub fn option_set_flags(
    flag: &mut u32,
    on_test: u32,
    off_test: u32,
    pass_set: u32,
    pass_clear: u32,
    fail_set: u32,
    fail_clear: u32,
) {
    if (*flag & on_test) == on_test && (off_test == 0 || *flag & off_test == 0) {
        *flag = (*flag | pass_set) & !pass_clear;
    } else {
        *flag = (*flag | fail_set) & !fail_clear;
    }
}
/// Original `imodvObjedDrawData`.
pub fn imodv_objed_draw_data(a: &mut ImodvApp, option: i32, combined: bool) {
    match option {
        0 => set_obj_flag(a, IMOD_OBJFLAG_OFF, 1, 0, false, false),
        3 => set_obj_flag(a, IMOD_OBJFLAG_OFF, 0, 0, false, false),
        1 | 2 => {
            let (on, off, ps, pc, fs, fc) = if option == 1 {
                (
                    IMOD_OBJFLAG_MESH | IMOD_OBJFLAG_NOLINE | IMOD_OBJFLAG_FILL,
                    IMOD_OBJFLAG_OFF,
                    0,
                    IMOD_OBJFLAG_MESH | IMOD_OBJFLAG_NOLINE | IMOD_OBJFLAG_FILL,
                    0,
                    IMOD_OBJFLAG_MESH | IMOD_OBJFLAG_OFF,
                )
            } else {
                (
                    0,
                    IMOD_OBJFLAG_MESH | IMOD_OBJFLAG_NOLINE | IMOD_OBJFLAG_FILL | IMOD_OBJFLAG_OFF,
                    IMOD_OBJFLAG_MESH | IMOD_OBJFLAG_NOLINE | IMOD_OBJFLAG_FILL,
                    0,
                    IMOD_OBJFLAG_MESH,
                    IMOD_OBJFLAG_OFF,
                )
            };
            let (mut f, mut l) = (0, 0);
            set_start_end_model(a, true, &mut f, &mut l);
            for m in f..=l {
                for ob in 0..num_editable_objects(a, m) {
                    if let Some(o) = editable_object(a, m, ob) {
                        option_set_flags(&mut o.flags, on, off, ps, pc, fs, fc);
                    }
                }
            }
        }
        _ => {}
    }
    if !combined {
        finish_change_and_draw(a, false, true);
    }
}
/// Original `imodvObjedStyleData`.
pub fn imodv_objed_style_data(a: &mut ImodvApp, option: i32, combined: bool) {
    match option {
        STYLE_POINTS => {
            set_obj_flag(a, IMOD_OBJFLAG_NOLINE, 1, 0, combined, false);
            set_obj_flag(a, IMOD_OBJFLAG_FILL, 0, 0, true, false);
        }
        STYLE_LINES => set_obj_flag(
            a,
            IMOD_OBJFLAG_FILL | IMOD_OBJFLAG_NOLINE,
            0,
            0,
            combined,
            false,
        ),
        STYLE_FILL => set_obj_flag(
            a,
            IMOD_OBJFLAG_FILL | IMOD_OBJFLAG_NOLINE,
            1,
            0,
            combined,
            false,
        ),
        STYLE_FILL_OUTLINE => {
            set_obj_flag(a, IMOD_OBJFLAG_NOLINE, 0, 0, combined, false);
            set_obj_flag(a, IMOD_OBJFLAG_FILL, 1, 0, true, false);
        }
        _ => {}
    }
    finish_change_and_draw(a, false, false);
}
/// Original `imodvObjedSetDrawTypeAndStyle`.
pub fn imodv_objed_set_draw_type_and_style(a: &mut ImodvApp, new_data: i32) {
    imodv_objed_draw_data(a, new_data / 10, true);
    imodv_objed_style_data(a, new_data % 10, true);
}
/// Original `imodvObjedEditData`.
pub fn imodv_objed_edit_data(editor: &mut ImodvObjed, option: i32) {
    editor.edit_all = option;
}
/// Original `imodvObjedSelect`.
pub fn imodv_objed_select(a: &mut ImodvApp, which: i32) {
    a.obj_num = which - 1;
    if let Some(model) = unsafe { a.imod.as_mut() } {
        a.obj = model
            .obj
            .get_mut(a.obj_num.max(0) as usize)
            .map_or(std::ptr::null_mut(), |o| o);
    }
    unsafe { imodv_draw() };
}
/// Original `imodvObjedChangeObject`.
pub fn imodv_objed_change_object(a: &mut ImodvApp, dir: i32) {
    let next = a.obj_num + dir;
    if let Some(model) = unsafe { a.imod.as_ref() } {
        if next >= 0 && (next as usize) < model.obj.len() {
            imodv_objed_select(a, next + 1);
        }
    }
}
/// Original `imodvObjedName`.
pub fn imodv_objed_name(a: &mut ImodvApp, name: &str) {
    let index = a.obj_num;
    if let Some(obj) = objed_object(a) {
        obj.name = [0; 64];
        for (out, b) in obj.name.iter_mut().zip(name.bytes().take(63)) {
            *out = b;
        }
        imodv_register_object_chg(index);
        imodv_finish_chg_unit();
        unsafe { imodv_draw() };
    }
}
/// Original `objedToggleObj`.
pub fn objed_toggle_obj(a: &mut ImodvApp, ob: i32, state: bool) {
    let (mut f, mut l) = (0, 0);
    set_start_end_model(a, true, &mut f, &mut l);
    for m in f..=l {
        if let Some(o) = editable_object(a, m, ob) {
            if state {
                o.flags &= !IMOD_OBJFLAG_OFF;
                a.obj_num = ob;
            } else {
                o.flags |= IMOD_OBJFLAG_OFF;
            }
        }
    }
    finish_change_and_draw(a, true, true);
}
/// Original `imodvObjedCollectChanges`.
pub fn imodv_objed_collect_changes(a: &mut ImodvApp, editor: &mut ImodvObjed, value: bool) {
    editor.collect_changes = value;
    if !value {
        finish_change_and_draw(a, true, true);
    }
}
/// Original `imodvObjedFramePicked`.
pub fn imodv_objed_frame_picked(editor: &mut ImodvObjed, item: i32) {
    editor.current_panel_frame = item;
}
/// Original `imodvObjedNewView`.
pub fn imodv_objed_new_view(a: &mut ImodvApp) {
    if a.sync_objed_to_cur_obj != 0 {
        if let Some(m) = unsafe { a.imod.as_ref() } {
            if m.cindex.object >= 0 {
                a.obj_num = m.cindex.object;
            }
        }
    }
}
/// Original `object_edit_kill`.
pub fn object_edit_kill(editor: &mut ImodvObjed) -> i32 {
    if editor.dialog_open {
        editor.dialog_open = false;
        1
    } else {
        0
    }
}
/// Original `objed`.
pub fn objed(a: &mut ImodvApp, editor: &mut ImodvObjed) {
    if a.imod.is_null() {
        return;
    }
    editor.dialog_open = true;
    editor.edit_all = 0;
}
/// Original `imodvObjedDone`.
pub fn imodv_objed_done(editor: &mut ImodvObjed) {
    if !editor.mesh_busy {
        editor.dialog_open = false;
    }
}
/// Original `imodvObjedClosing`.
pub fn imodv_objed_closing(editor: &mut ImodvObjed) {
    editor.dialog_open = false;
}
/// Original `imodvObjedCtrlKey`.
pub fn imodv_objed_ctrl_key(editor: &mut ImodvObjed, pressed: bool) {
    editor.ctrl_pressed = pressed;
}
/// Original `imodvObjedSetCurFrame`.
pub fn imodv_objed_set_cur_frame(editor: &mut ImodvObjed, value: i32) {
    editor.current_panel_frame = value;
}
/// Original `imodvObjedGetCurFrame`.
pub fn imodv_objed_get_cur_frame(editor: &ImodvObjed) -> i32 {
    editor.current_panel_frame
}
/// Original `imodvObjedMakeOnOffs`.
pub fn imodv_objed_make_on_offs(a: &ImodvApp) -> usize {
    a.mod_
        .iter()
        .map(|m| unsafe { m.as_ref() })
        .map(|m| m.obj.len())
        .max()
        .unwrap_or(0)
        .min(100)
}

/// Native slot form of `ImodvObjed::toggleObjSlot`.
pub fn toggle_obj_slot(a: &mut ImodvApp, object: i32, checked: bool) {
    objed_toggle_obj(a, object, checked);
}
/// Native form of the Qt `objset` refresh: synchronize retained object-editor state.
pub fn objset(a: &mut ImodvApp, editor: &mut ImodvObjed) {
    objed(a, editor);
}
/// `setOnoffButtons`: source checked state for all editable objects.
pub fn set_onoff_buttons(a: &ImodvApp) -> Vec<bool> {
    unsafe { a.imod.as_ref() }
        .map(|model| {
            model
                .obj
                .iter()
                .map(|obj| obj.flags & IMOD_OBJFLAG_OFF == 0)
                .collect()
        })
        .unwrap_or_default()
}
/// `addOnoffButton`: append the source-default on state.
pub fn add_onoff_button(buttons: &mut Vec<bool>) {
    buttons.push(true);
}

/// `setLineColor_cb`: RGB plus transparency control values.
pub fn set_line_color_cb(obj: Option<&Iobj>) -> Option<[i32; 4]> {
    obj.map(|obj| {
        [
            (obj.red * 255.0) as i32,
            (obj.green * 255.0) as i32,
            (obj.blue * 255.0) as i32,
            obj.trans as i32,
        ]
    })
}
/// `mkLineColor_cb` native control labels.
pub fn mk_line_color_cb() -> [&'static str; 4] {
    ["Red", "Green", "Blue", "Transparency"]
}
/// `setFillColor_cb`: RGB fill control values.
pub fn set_fill_color_cb(obj: Option<&Iobj>) -> Option<[i32; 3]> {
    obj.map(|obj| {
        [
            obj.fillred as i32,
            obj.fillgreen as i32,
            obj.fillblue as i32,
        ]
    })
}
/// `mkFillColor_cb` native control labels.
pub fn mk_fill_color_cb() -> [&'static str; 3] {
    ["Red", "Green", "Blue"]
}
/// `setMaterial` source property assignment.
pub fn set_material(obj: &mut Iobj, which: i32, value: i32) {
    let value = value.clamp(0, 255) as u8;
    match which {
        0 => obj.ambient = value,
        1 => obj.diffuse = value,
        2 => obj.specular = value,
        3 => obj.shininess = value,
        4 => obj.valblack = value,
        5 => obj.valwhite = value,
        _ => {}
    }
}
/// `setMaterial_cb` current material control values.
pub fn set_material_cb(obj: Option<&Iobj>) -> Option<[i32; 4]> {
    obj.map(|obj| {
        [
            obj.ambient as i32,
            obj.diffuse as i32,
            obj.specular as i32,
            obj.shininess as i32,
        ]
    })
}
/// `mkMaterial_cb` native control labels.
pub fn mk_material_cb() -> [&'static str; 4] {
    ["Ambient", "Diffuse", "Specular", "Shininess"]
}
/// `setPoints_cb` retained point-size/quality controls.
pub fn set_points_cb(obj: Option<&Iobj>) -> Option<(i32, i32)> {
    obj.map(|obj| (obj.pdrawsize, obj.pdrawsize))
}
/// `mkPoints_cb` native point controls.
pub fn mk_points_cb() -> [&'static str; 2] {
    ["Size", "Quality"]
}
/// `setLines_cb` retained line-width controls.
pub fn set_lines_cb(obj: Option<&Iobj>) -> Option<i32> {
    obj.map(|obj| obj.linewidth as i32)
}
/// `mkLines_cb` native line-control labels.
pub fn mk_lines_cb() -> [&'static str; 5] {
    [
        "2D Line Width",
        "3D Line Width",
        "Scale for high DPI",
        "Anti-alias rendering",
        "Thicken current contour",
    ]
}
/// `mkScalar_cb` native scalar/mesh-value control labels.
pub fn mk_scalar_cb() -> [&'static str; 7] {
    [
        "No value drawing",
        "Show stored values",
        "Show normal magnitudes",
        "Black Level",
        "White Level",
        "False",
        "Fixed",
    ]
}
impl MeshColorBar {
    /// Native `MeshColorBar.paintEvent`: return whether a false-color bar should render.
    pub fn paint_event(&self, obj: Option<&Iobj>) -> bool {
        obj.is_some_and(|obj| obj.flags & IMOD_OBJFLAG_MCOLOR != 0)
            && self.width > 1
            && self.height > 0
    }
}
/// Native values read by `setClip_cb`.
pub fn set_clip_cb(
    draw_clip: bool,
    edit_global: bool,
    plane: i32,
    count: i32,
    enabled: bool,
) -> (bool, bool, i32, i32, bool) {
    (draw_clip, edit_global, plane, count, enabled)
}
/// `mkClip_cb` native control labels.
pub fn mk_clip_cb() -> [&'static str; 7] {
    [
        "Show current plane",
        "Object",
        "Global",
        "Skip global planes",
        "Plane #",
        "Clipping plane ON",
        "Adjust all ON planes",
    ]
}
/// `fixClip_cb`: native layout uses source button captions to calculate consistent widths.
pub fn fix_clip_cb() -> [&'static str; 4] {
    ["X", "Y", "Z", "Invert"]
}
/// `mkMove_cb` native move-control labels.
pub fn mk_move_cb() -> [&'static str; 13] {
    [
        "Center on Object",
        "Top",
        "Front",
        "Bottom",
        "Back",
        "Top",
        "Left",
        "Bottom",
        "Right",
        "Front",
        "Left",
        "Back",
        "Right",
    ]
}
/// `fixMove_cb`: all rotation buttons use the Bottom-label width.
pub fn fix_move_cb() -> &'static str {
    "Bottom"
}
/// `mkSubsets_cb` native subset selector labels.
pub fn mk_subsets_cb() -> [&'static str; 2] {
    ["Show all ON objects", "Current object only"]
}
/// Native `meshThickSlot` object thickness mutation.
pub fn mesh_thick_slot(obj: &mut Iobj, value: i32) {
    obj.mesh_thickness = value.clamp(0, u8::MAX as i32) as u8;
}
/// Native `meshOnImageSlot` state mutation.
pub fn mesh_on_image_slot(on_image: &mut bool, state: bool) {
    *on_image = state;
}
/// Native values read by `setMeshDraw_cb`.
pub fn set_mesh_draw_cb(obj: Option<&Iobj>) -> Option<(i32, bool)> {
    obj.map(|obj| {
        (
            obj.mesh_thickness as i32,
            obj.flags & IMOD_OBJFLAG_MESH != 0,
        )
    })
}
/// `mkMeshDraw_cb` native mesh-draw control labels.
pub fn mk_mesh_draw_cb() -> [&'static str; 2] {
    ["Surface thickness", "Draw mesh on image"]
}
/// Original static `finishChangeAndDraw`.
pub fn finish_change_and_draw(a: &mut ImodvApp, do_objset: bool, draw_images: bool) {
    imodv_finish_chg_unit();
    unsafe { imodv_draw() };
}

/// Source `setOrClearFlags` macro applied to a `MeshParams::flags` word.
fn set_or_clear_mesh_flag(
    params: &mut crate::imod::libimod::imesh::MeshParams,
    flag: u32,
    state: bool,
) {
    if state {
        params.flags |= flag;
    } else {
        params.flags &= !flag;
    }
}

impl ImodvObjed {
    /// Original `ImodvObjed::lineColorSlot`.
    pub fn line_color_slot(&mut self, a: &mut ImodvApp, color: i32, value: i32, dragging: bool) {
        if let Some(o) = objed_object(a) {
            match color {
                0 => o.red = value as f32 / 255.,
                1 => o.green = value as f32 / 255.,
                2 => o.blue = value as f32 / 255.,
                3 => o.trans = value.clamp(0, 255) as u8,
                _ => {}
            }
        }
        if !dragging {
            finish_change_and_draw(a, false, true);
        }
    }
    /// Original `ImodvObjed::multipleColorSlot`.
    pub fn multiple_color_slot(&mut self, state: bool) {
        self.multiple_color_ok = state;
    }
    /// Original `ImodvObjed::fillToggleSlot`.
    pub fn fill_toggle_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_obj_flag(a, IMOD_OBJFLAG_FILL, state as i32, 0, false, false);
        finish_change_and_draw(a, false, false);
    }
    /// Original `ImodvObjed::fillPntToggleSlot`.
    pub fn fill_pnt_toggle_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_obj_flag(a, IMOD_OBJFLAG_FCOLOR_PNT, state as i32, 0, false, false);
        finish_change_and_draw(a, true, false);
    }
    /// Original `ImodvObjed::fillColorSlot`.
    pub fn fill_color_slot(&mut self, a: &mut ImodvApp, color: i32, value: i32, dragging: bool) {
        if let Some(o) = objed_object(a) {
            match color {
                0 => o.fillred = value as u8,
                1 => o.fillgreen = value as u8,
                2 => o.fillblue = value as u8,
                _ => {}
            }
        }
        if !dragging {
            finish_change_and_draw(a, false, true);
        }
    }
    /// Original `ImodvObjed::materialSlot`.
    pub fn material_slot(&mut self, a: &mut ImodvApp, which: i32, value: i32, dragging: bool) {
        if let Some(o) = objed_object(a) {
            match which {
                0 => o.ambient = value as u8,
                1 => o.diffuse = value as u8,
                2 => o.specular = value as u8,
                3 => o.shininess = value as u8,
                4 => o.valblack = value as u8,
                5 => o.valwhite = value as u8,
                _ => {}
            }
        }
        if !dragging {
            finish_change_and_draw(a, false, true);
        }
    }
    /// Original `ImodvObjed::bothSidesSlot`.
    pub fn both_sides_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_obj_flag(a, IMOD_OBJFLAG_TWO_SIDE, state as i32, 0, false, false);
        finish_change_and_draw(a, false, false);
    }
    /// Original `ImodvObjed::pointSizeSlot`.
    pub fn point_size_slot(&mut self, a: &mut ImodvApp, value: i32) {
        if let Some(o) = objed_object(a) {
            o.pdrawsize = value;
        }
        finish_change_and_draw(a, false, true);
    }
    /// Original `ImodvObjed::pointQualitySlot`.
    pub fn point_quality_slot(&mut self, a: &mut ImodvApp, value: i32) {
        if let Some(o) = objed_object(a) {
            o.quality = value as u8;
        }
        finish_change_and_draw(a, false, true);
    }
    /// Original `ImodvObjed::globalQualitySlot`.
    pub fn global_quality_slot(&mut self, a: &mut ImodvApp, value: i32) {
        if a.imod.is_null() {
            return;
        }
        imodv_register_model_chg();
        let (mut first, mut last) = (0, 0);
        set_start_end_model(a, true, &mut first, &mut last);
        let quality = (value - 1) as u32;
        for model in first..=last {
            if let Some(view) = unsafe {
                a.mod_
                    .get(model.max(0) as usize)
                    .and_then(|m| m.as_ptr().as_mut())
                    .and_then(|m| m.view.first_mut())
            } {
                view.world = (view.world & !WORLD_QUALITY_BITS) | (quality << WORLD_QUALITY_SHIFT);
            }
        }
        finish_change_and_draw(a, false, false);
    }
    /// Original `ImodvObjed::pointNoDrawSlot`.
    pub fn point_no_draw_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_obj_flag(a, IMOD_OBJFLAG_SCAT, state as i32, 0, false, false);
        finish_change_and_draw(a, false, true);
    }
    /// Original `ImodvObjed::lineWidthSlot`.
    pub fn line_width_slot(&mut self, a: &mut ImodvApp, which: i32, value: i32, dragging: bool) {
        if let Some(o) = objed_object(a) {
            if which == 0 {
                o.linewidth = value as u8;
            } else {
                o.linewidth2 = value as u8;
            }
        }
        if !dragging {
            finish_change_and_draw(a, false, true);
        }
    }
    /// Original `ImodvObjed::scaleLineWidth`.
    pub fn scale_line_width(&mut self, a: &mut ImodvApp, state: bool) {
        set_obj_flag(a, IMOD_OBJFLAG_SCALE_WDTH, state as i32, 0, false, false);
        finish_change_and_draw(a, false, true);
    }
    /// Original `ImodvObjed::lineAliasSlot`.
    pub fn line_alias_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_obj_flag(a, IMOD_OBJFLAG_ANTI_ALIAS, state as i32, 0, false, false);
        finish_change_and_draw(a, false, true);
    }
    /// Original `ImodvObjed::lineThickenSlot`.
    pub fn line_thicken_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_obj_flag(a, IMOD_OBJFLAG_THICK_CONT, state as i32, 0, false, false);
        finish_change_and_draw(a, true, false);
    }
    /// Original `ImodvObjed::openObjectSlot`.
    pub fn open_object_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_obj_flag(
            a,
            IMOD_OBJFLAG_OPEN,
            state as i32,
            OBJTYPE_OPEN | OBJTYPE_CLOSED,
            false,
            false,
        );
        finish_change_and_draw(a, true, true);
    }
    /// Original `ImodvObjed::autoNewContSlot`.
    pub fn auto_new_cont_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_obj_flag(
            a,
            IMOD_OBJFLAG_PLANAR,
            state as i32,
            OBJTYPE_OPEN,
            false,
            false,
        );
        finish_change_and_draw(a, true, true);
    }
    /// Original `ImodvObjed::meshShowSlot`.
    pub fn mesh_show_slot(&mut self, a: &mut ImodvApp, value: i32) {
        set_obj_flag(a, IMOD_OBJFLAG_SCALAR, (value == 2) as i32, 0, false, false);
        set_obj_flag(
            a,
            IMOD_OBJFLAG_USE_VALUE,
            (value == 1) as i32,
            0,
            false,
            false,
        );
        finish_change_and_draw(a, true, true);
    }
    /// Original `ImodvObjed::meshFalseSlot`.
    pub fn mesh_false_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_obj_flag(a, IMOD_OBJFLAG_MCOLOR, state as i32, 0, false, false);
        finish_change_and_draw(a, true, true);
    }
    /// Original `ImodvObjed::meshConstantSlot`.
    pub fn mesh_constant_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_bit_in_mat_flags_2(a, MATFLAGS2_CONSTANT, state as i32);
        finish_change_and_draw(a, true, true);
    }
    /// Original `ImodvObjed::meshLevelSlot`.
    pub fn mesh_level_slot(&mut self, a: &mut ImodvApp, which: i32, value: i32, dragging: bool) {
        let span = self.mesh_max - self.mesh_min;
        if span <= 0 {
            return;
        }
        let scaled = (255. * (value - self.mesh_min) as f32 / span as f32 + 0.5).floor() as i32;
        self.material_slot(a, which + 4, scaled, dragging);
    }

    /// Original static `setScalar_cb`: refreshes the scalar control range for
    /// the current object.  Widget enablement/color-bar painting stay at the
    /// Rust GUI boundary, while this preserves the source range conversion.
    pub fn set_scalar(&mut self, a: &ImodvApp) {
        let Some(model) = (unsafe { a.imod.as_ref() }) else {
            return;
        };
        let Some(object) = model.obj.get(a.obj_num.max(0) as usize) else {
            return;
        };
        self.mesh_rmin = 0.;
        self.mesh_rmax = 255.;
        if object.flags & IMOD_OBJFLAG_USE_VALUE != 0 {
            istore_get_min_max(
                &object.store,
                object.cont.len() as i32,
                GEN_STORE_MINMAX1,
                &mut self.mesh_rmin,
                &mut self.mesh_rmax,
            );
        }
        let mut decimals = if self.mesh_rmax > self.mesh_rmin {
            -(((self.mesh_rmax - self.mesh_rmin) as f64 / 1500.).log10()) as i32
        } else {
            0
        };
        decimals = decimals.clamp(0, 6);
        let scale = 10_f32.powi(decimals);
        self.mesh_min = (self.mesh_rmin * scale + 0.5).floor() as i32;
        self.mesh_max = (self.mesh_rmax * scale + 0.5).floor() as i32;
        if self.mesh_min >= self.mesh_max {
            self.mesh_max = self.mesh_min + 1;
        }
    }
    /// Original `ImodvObjed::meshSkipLoSlot`.
    pub fn mesh_skip_lo_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_bit_in_mat_flags_2(a, MATFLAGS2_SKIP_LOW, state as i32);
        finish_change_and_draw(a, true, true);
    }
    /// Original `ImodvObjed::meshSkipHiSlot`.
    pub fn mesh_skip_hi_slot(&mut self, a: &mut ImodvApp, state: bool) {
        set_bit_in_mat_flags_2(a, MATFLAGS2_SKIP_HIGH, state as i32);
        finish_change_and_draw(a, true, true);
    }
    /// Original `ImodvObjed::clipShowSlot`.
    pub fn clip_show_slot(&mut self, a: &mut ImodvApp, state: bool) {
        a.draw_clip = state as i32;
        unsafe { imodv_draw() };
    }
    /// Original `ImodvObjed::clipGlobalSlot`.
    pub fn clip_global_slot(&mut self, a: &mut ImodvApp, value: i32) {
        if let Some(model) = unsafe { a.imod.as_mut() } {
            model.edit_global_clip = value;
        }
    }
    /// Original `ImodvObjed::clipSkipSlot`.
    pub fn clip_skip_slot(&mut self, a: &mut ImodvApp, state: bool) {
        let object_number = a.obj_num;
        if let Some(object) = objed_object(a) {
            imodv_register_object_chg(object_number);
            if state {
                object.clips.flags |= 1 << 7;
            } else {
                object.clips.flags &= !(1 << 7);
            }
            finish_change_and_draw(a, false, false);
        }
    }
    /// Original `ImodvObjed::clipPlaneSlot`.
    pub fn clip_plane_slot(&mut self, a: &mut ImodvApp, value: i32) {
        if let Some(model) = unsafe { a.imod.as_mut() } {
            if model.edit_global_clip != 0 {
                imodv_register_model_chg();
                model.view[0].clips.plane = (value - 1) as u8;
            } else if let Some(object) = model.obj.get_mut(a.obj_num.max(0) as usize) {
                imodv_register_object_chg(a.obj_num);
                object.clips.plane = (value - 1) as u8;
            }
        }
        if a.draw_clip != 0 {
            unsafe { imodv_draw() };
        }
    }
    /// Original `ImodvObjed::clipResetSlot`.
    pub fn clip_reset_slot(&mut self, a: &mut ImodvApp, which: i32) {
        self.clip_reset_plane_to_axis(a, which, -1, -1);
    }
    /// Original `ImodvObjed::clipResetPlaneToAxis`.
    pub fn clip_reset_plane_to_axis(
        &mut self,
        a: &mut ImodvApp,
        which: i32,
        do_global: i32,
        plane_num: i32,
    ) {
        let object_number = a.obj_num;
        let Some(model) = (unsafe { a.imod.as_mut() }) else {
            return;
        };
        let do_global = if do_global < 0 {
            model.edit_global_clip != 0
        } else {
            do_global != 0
        };
        let (mut min, mut max) = (Ipoint::default(), Ipoint::default());
        let clips = if do_global {
            imod_get_bounding_box(model, &mut min, &mut max);
            imodv_register_model_chg();
            &mut model.view[0].clips
        } else {
            let Some(object) = model.obj.get_mut(object_number.max(0) as usize) else {
                return;
            };
            imod_object_get_bbox(object, &mut min, &mut max);
            imodv_register_object_chg(object_number);
            &mut object.clips
        };
        let plane = if plane_num < 0 {
            clips.plane as usize
        } else {
            plane_num as usize
        };
        if plane >= clips.normal.len() {
            return;
        }
        clips.point[plane] = Ipoint {
            x: -(max.x + min.x) * 0.5,
            y: -(max.y + min.y) * 0.5,
            z: -(max.z + min.z) * 0.5,
        };
        clips.normal[plane] = Ipoint::default();
        if which == 2 {
            clips.normal[plane].z = -1.;
        } else if which == 1 {
            clips.normal[plane].y = -1.;
        } else {
            clips.normal[plane].x = -1.;
        }
        finish_change_and_draw(a, false, false);
    }
    /// Original `ImodvObjed::clipInvertSlot`.
    pub fn clip_invert_slot(&mut self, a: &mut ImodvApp) {
        const WORLD_MOVE_ALL_CLIP: u32 = 1 << 12;
        let object_number = a.obj_num;
        let Some(model) = (unsafe { a.imod.as_mut() }) else {
            return;
        };
        let move_all = model.view[0].world & WORLD_MOVE_ALL_CLIP != 0;
        let clips = if model.edit_global_clip != 0 {
            imodv_register_model_chg();
            &mut model.view[0].clips
        } else {
            let Some(object) = model.obj.get_mut(object_number.max(0) as usize) else {
                return;
            };
            imodv_register_object_chg(object_number);
            &mut object.clips
        };
        let (first, last) = if move_all && clips.count == 0 {
            (1, 0)
        } else if move_all {
            (0, clips.count as usize - 1)
        } else {
            let plane = clips.plane as usize;
            (plane, plane)
        };
        for index in first..=last.min(clips.normal.len() - 1) {
            if clips.flags & (1 << index) != 0 || first == last {
                clips.normal[index].x = -clips.normal[index].x;
                clips.normal[index].y = -clips.normal[index].y;
                clips.normal[index].z = -clips.normal[index].z;
            }
        }
        finish_change_and_draw(a, false, false);
    }
    /// Original `ImodvObjed::clipToggleSlot`.
    pub fn clip_toggle_slot(&mut self, a: &mut ImodvApp, state: bool) {
        self.toggle_clip_plane(a, state, -1, -1);
    }
    /// Original `ImodvObjed::toggleClipPlane`.
    pub fn toggle_clip_plane(
        &mut self,
        a: &mut ImodvApp,
        state: bool,
        do_global: i32,
        plane_num: i32,
    ) {
        const WORLD_MOVE_ALL_CLIP: u32 = 1 << 12;
        let object_number = a.obj_num;
        let Some(model) = (unsafe { a.imod.as_mut() }) else {
            return;
        };
        let do_global = if do_global < 0 {
            model.edit_global_clip != 0
        } else {
            do_global != 0
        };
        let move_all = plane_num < 0 && model.view[0].world & WORLD_MOVE_ALL_CLIP != 0;
        let clips = if do_global {
            imodv_register_model_chg();
            &mut model.view[0].clips
        } else {
            let Some(object) = model.obj.get_mut(a.obj_num.max(0) as usize) else {
                return;
            };
            imodv_register_object_chg(object_number);
            &mut object.clips
        };
        let index = if plane_num < 0 {
            clips.plane as usize
        } else {
            plane_num as usize
        };
        if index >= clips.normal.len() || index >= u8::BITS as usize {
            return;
        }
        let (first, last) = if move_all {
            (0, (clips.count as usize).saturating_sub(1).max(index))
        } else {
            (index, index)
        };
        let mut reset_plane = None;
        for index in first..=last.min(clips.normal.len() - 1) {
            let mask = 1u8 << index;
            if state {
                clips.flags |= mask;
                if index == clips.count as usize {
                    clips.count += 1;
                }
                if clips.point[index] == Ipoint::default()
                    && clips.normal[index]
                        == (Ipoint {
                            x: 0.,
                            y: 0.,
                            z: -1.,
                        })
                {
                    reset_plane = Some(index);
                    break;
                }
            } else {
                clips.flags &= !mask;
            }
        }
        if let Some(index) = reset_plane {
            self.clip_reset_plane_to_axis(a, 2, do_global as i32, index as i32);
            return;
        }
        finish_change_and_draw(a, true, false);
    }
    /// Original `ImodvObjed::clipMoveAllSlot`.
    pub fn clip_move_all_slot(&mut self, a: &mut ImodvApp, state: bool) {
        const WORLD_MOVE_ALL_CLIP: u32 = 1 << 12;
        imodv_register_model_chg();
        let (mut first, mut last) = (0, 0);
        set_start_end_model(a, true, &mut first, &mut last);
        for index in first..=last {
            if let Some(model) = unsafe {
                a.mod_
                    .get(index as usize)
                    .and_then(|model| model.as_ptr().as_mut())
            } {
                if state {
                    model.view[0].world |= WORLD_MOVE_ALL_CLIP;
                } else {
                    model.view[0].world &= !WORLD_MOVE_ALL_CLIP;
                }
            }
        }
        imodv_finish_chg_unit();
    }
    /// Original `ImodvObjed::moveCenterSlot`.
    pub fn move_center_slot(&mut self, a: &mut ImodvApp) {
        let Some(model) = (unsafe { a.imod.as_mut() }) else {
            return;
        };
        let Some(object) = model.obj.get(a.obj_num.max(0) as usize) else {
            return;
        };
        let (mut min, mut max) = (Ipoint::default(), Ipoint::default());
        if imod_object_get_bbox(object, &mut min, &mut max) < 0 {
            return;
        }
        let view = &mut model.view[0];
        view.trans.x = -((max.x + min.x) * 0.5);
        view.trans.y = -((max.y + min.y) * 0.5);
        view.trans.z = -((max.z + min.z) * 0.5);
        unsafe { imodv_draw() };
    }
    /// Original `ImodvObjed::moveAxisSlot`.
    pub fn move_axis_slot(&mut self, a: &mut ImodvApp, which: i32) {
        imodv_objed_move_to_axis(a, which);
    }
    /// Original `ImodvObjed::subsetSlot`.
    pub fn subset_slot(&mut self, a: &mut ImodvApp, which: i32) {
        a.current_subset = which;
        unsafe { imodv_draw() };
    }
    /// Original `ImodvObjed::makePassSlot` through `makeSpinChanged`.
    pub fn make_pass_slot(&mut self, a: &mut ImodvApp, value: i32) {
        self.make_spin_changed(a, 0, value as f64);
    }
    pub fn make_diam_slot(&mut self, a: &mut ImodvApp, value: f64) {
        self.make_spin_changed(a, 1, value);
    }
    pub fn make_tol_slot(&mut self, a: &mut ImodvApp, value: f64) {
        self.make_spin_changed(a, 2, value);
    }
    pub fn make_zinc_slot(&mut self, a: &mut ImodvApp, value: i32) {
        self.make_spin_changed(a, 3, value as f64);
    }
    pub fn make_flat_slot(&mut self, a: &mut ImodvApp, value: f64) {
        self.make_spin_changed(a, 4, value);
    }
    /// Original `ImodvObjed::makeSpinChanged`.
    pub fn make_spin_changed(&mut self, a: &mut ImodvApp, which: i32, value: f64) {
        if a.imod.is_null() || objed_object(a).is_none() {
            return;
        }
        let (mut first, mut last) = (0, 0);
        set_start_end_model(a, true, &mut first, &mut last);
        let mut any = false;
        for model in first..=last {
            let count = num_editable_objects(a, model);
            for object in 0..count {
                if !change_model_object(a, model, object, true) {
                    continue;
                }
                let Some(obj) = editable_object(a, model, object) else {
                    continue;
                };
                if iobj_scat(obj.flags) != 0 {
                    continue;
                }
                let param = obj.mesh_param.get_or_insert_with(Default::default);
                match which {
                    0 => param.passes = value.round() as i32,
                    1 => param.tube_diameter = value as f32,
                    2 if self.make_low_res => param.tol_low_res = value as f32,
                    2 => param.tol_high_res = value as f32,
                    3 if self.make_low_res => param.incz_low_res = value.round() as i32,
                    3 => param.incz_high_res = value.round() as i32,
                    4 => param.flat_crit = value as f32,
                    _ => continue,
                }
                any = true;
            }
        }
        if any {
            imodv_finish_chg_unit();
        }
    }
    /// Original `ImodvObjed::makeStateSlot`; `state` is supplied by the Rust
    /// UI event instead of fetched from the former Qt checkbox.
    pub fn make_state_slot(&mut self, a: &mut ImodvApp, which: i32, state: bool) {
        // MAKE_MESH_LOW
        if which == 0 {
            self.make_low_res = state;
            return;
        }
        if a.imod.is_null() || objed_object(a).is_none() {
            return;
        }
        let (mut first, mut last) = (0, 0);
        set_start_end_model(a, true, &mut first, &mut last);
        let mut any = false;
        for model in first..=last {
            for object in 0..num_editable_objects(a, model) {
                if !change_model_object(a, model, object, true) {
                    continue;
                }
                let Some(obj) = editable_object(a, model, object) else {
                    continue;
                };
                if iobj_scat(obj.flags) != 0 {
                    continue;
                }
                let open = iobj_open(obj.flags) != 0;
                let param = obj.mesh_param.get_or_insert_with(Default::default);
                let tube = param.flags & IMESH_MK_TUBE != 0 && open;
                match which {
                    1 if !tube => set_or_clear_mesh_flag(param, IMESH_MK_SKIP, state),
                    2 if !tube => set_or_clear_mesh_flag(param, IMESH_MK_SURF, state),
                    3 if open => set_or_clear_mesh_flag(param, IMESH_MK_TUBE, state),
                    4 if tube => set_or_clear_mesh_flag(param, IMESH_MK_CAP_TUBE, state),
                    4 => param.cap = if state { IMESH_CAP_ALL } else { IMESH_CAP_OFF },
                    5 if tube => set_or_clear_mesh_flag(param, IMESH_MK_CAP_DOME, state),
                    _ => continue,
                }
                any = true;
            }
        }
        if any {
            imodv_finish_chg_unit();
        }
    }
    /// Original `ImodvObjed::makeZeditSlot`; the Rust UI supplies the text.
    pub fn make_zedit_slot(&mut self, a: &mut ImodvApp, text: &str) {
        if a.imod.is_null() || objed_object(a).is_none() {
            return;
        }
        let numbers: Vec<i32> = text
            .split(',')
            .filter_map(|part| part.trim().parse::<i32>().ok())
            .collect();
        let (mut first, mut last) = (0, 0);
        set_start_end_model(a, true, &mut first, &mut last);
        let mut any = false;
        for model in first..=last {
            for object in 0..num_editable_objects(a, model) {
                if !change_model_object(a, model, object, true) {
                    continue;
                }
                let Some(obj) = editable_object(a, model, object) else {
                    continue;
                };
                if iobj_scat(obj.flags) != 0 {
                    continue;
                }
                let param = obj.mesh_param.get_or_insert_with(Default::default);
                param.minz = DEFAULT_VALUE;
                param.maxz = DEFAULT_VALUE;
                if let Some(&min) = numbers.first() {
                    let min = min - 1;
                    if numbers.len() == 1 {
                        param.minz = min;
                        param.maxz = min + 1;
                    } else {
                        let max = numbers[1] - 1;
                        param.minz = min.min(max);
                        param.maxz = (min.max(max)).max(param.minz + 1);
                    }
                }
                any = true;
            }
        }
        if any {
            imodv_finish_chg_unit();
        }
    }
    pub fn make_doit_slot(&mut self) {
        self.mesh_busy = true;
    }
    /// Runs the currently translated meshing branch for the selected object.
    /// A `-2` result means a closed-object Z section needs the remaining
    /// nesting dispatcher.
    pub fn make_doit_for_app(&mut self, a: &mut ImodvApp) -> i32 {
        let Some(model) = (unsafe { a.imod.as_mut() }) else {
            return -1;
        };
        let scale = Ipoint {
            x: if model.xscale == 0. { 1. } else { model.xscale },
            y: if model.yscale == 0. { 1. } else { model.yscale },
            z: if model.zscale == 0. { 1. } else { model.zscale },
        };
        let Some(object) = model.obj.get_mut(a.obj_num.max(0) as usize) else {
            return -1;
        };
        self.mesh_busy = true;
        self.update_meshing(a.obj_num);
        let result = mesh_object_replacing_resolution(object, &scale, self.make_low_res);
        self.mesh_busy = false;
        self.update_meshing(-1);
        if result == 0 {
            imodv_register_object_chg(a.obj_num);
        }
        result
    }
    pub fn make_do_all_slot(&mut self) {
        self.mesh_busy = true;
    }
    /// Synchronous Rust-native equivalent of `makeDoAllSlot`.
    pub fn make_do_all_for_app(&mut self, a: &mut ImodvApp) -> i32 {
        let Some(model) = (unsafe { a.imod.as_mut() }) else {
            return -1;
        };
        let scale = Ipoint {
            x: if model.xscale == 0. { 1. } else { model.xscale },
            y: if model.yscale == 0. { 1. } else { model.yscale },
            z: if model.zscale == 0. { 1. } else { model.zscale },
        };
        self.mesh_busy = true;
        let mut result = 0;
        for (index, object) in model.obj.iter_mut().enumerate() {
            if object.cont.is_empty() || object.flags & IMOD_OBJFLAG_SCAT != 0 {
                continue;
            }
            self.update_meshing(index as i32);
            let status = mesh_object_replacing_resolution(object, &scale, self.make_low_res);
            if status != 0 {
                result = status;
                break;
            }
            imodv_register_object_chg(index as i32);
        }
        self.mesh_busy = false;
        self.update_meshing(-1);
        result
    }
    /// `makeDoUpSlot`: step the saved Z range upward and remesh the current
    /// object.  Unlike the former placeholder, this takes the model-view
    /// application explicitly, which is the Rust owner of native `Imodv`.
    pub fn make_do_up_slot(&mut self, a: &mut ImodvApp) -> i32 {
        self.step_zand_mesh_one_for_app(a, 1)
    }
    /// `makeDoDownSlot`: step the saved Z range downward and remesh.
    pub fn make_do_down_slot(&mut self, a: &mut ImodvApp) -> i32 {
        self.step_zand_mesh_one_for_app(a, -1)
    }
    /// Synchronous Rust-native equivalent of `stepZandMeshOne`.
    pub fn step_zand_mesh_one_for_app(&mut self, a: &mut ImodvApp, direction: i32) -> i32 {
        let zmax = unsafe { a.imod.as_ref() }.map_or(-1, |model| model.zmax);
        if let Some(object) = objed_object(a) {
            if let Some(params) = object.mesh_param.as_mut() {
                // Native `stepZandMeshOne` requires `minz + dir > 0`, not
                // merely non-negative, before moving the saved mesh range.
                if params.minz + direction > 0 && params.maxz + direction <= zmax {
                    params.minz += direction;
                    params.maxz += direction;
                }
            }
        } else {
            return -1;
        }
        self.make_doit_for_app(a)
    }
    pub fn start_meshing_next(&mut self) -> i32 {
        (!self.mesh_busy) as i32
    }
    /// Source-compatible synchronous fallback for callers that own only an
    /// object.  The app-aware entry points use model scale; this boundary uses
    /// unit scale, as the original method's `Iobj *` signature has no model.
    pub fn mesh_one_object(&mut self, obj: &mut Iobj) -> i32 {
        self.mesh_busy = true;
        let result = mesh_object_replacing_resolution(
            obj,
            &Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            self.make_low_res,
        );
        self.mesh_busy = false;
        result
    }
    /// Source `updateMeshing`; the form renders `-1` as an empty label and
    /// non-negative values as its one-based object progress text.
    pub fn update_meshing(&mut self, ob: i32) {
        self.meshing_object = ob;
    }
    /// `stepZandMeshOne`.  Rust makes the native global `Imodv` an explicit
    /// argument so this slot can perform its source mutation and remesh.
    pub fn step_zand_mesh_one(&mut self, a: &mut ImodvApp, direction: i32) -> i32 {
        self.step_zand_mesh_one_for_app(a, direction)
    }
}
/// Original `imodvObjedDrawClipPlane`.
pub fn imodv_objed_draw_clip_plane(a: &mut ImodvApp, state: bool) {
    a.draw_clip = state as i32;
    unsafe { imodv_draw() };
}
/// Original `imodvObjedToggleClip`.
pub fn imodv_objed_toggle_clip(a: &mut ImodvApp, global: i32, plane: i32) {
    let Some(model) = (unsafe { a.imod.as_ref() }) else {
        return;
    };
    let do_global = if global < 0 {
        model.edit_global_clip != 0
    } else {
        global != 0
    };
    let Some(clips) = (if do_global {
        Some(&model.view[0].clips)
    } else {
        model
            .obj
            .get(a.obj_num.max(0) as usize)
            .map(|object| &object.clips)
    }) else {
        return;
    };
    let index = if plane < 0 {
        clips.plane as usize
    } else {
        plane as usize
    };
    if index >= clips.normal.len() || index >= u8::BITS as usize {
        return;
    }
    let state = clips.flags & (1u8 << index) == 0;
    let mut editor = ImodvObjed::default();
    editor.toggle_clip_plane(a, state, do_global as i32, plane);
}
/// Original `imodvObjedMoveToAxis`.
pub fn imodv_objed_move_to_axis(a: &mut ImodvApp, which: i32) {
    const MOVE_QUARTERS: [i32; 12] = [0, -1, 2, 1, 0, 1, 2, -1, 2, -1, 0, 1];
    let Some(&quarter) = MOVE_QUARTERS.get(which as usize) else {
        return;
    };
    let mut rot = Ipoint::default();
    match which / 4 {
        0 => rot.x = quarter as f32 * 90.,
        1 => rot.y = quarter as f32 * 90.,
        2 => {
            rot.x = 90.;
            rot.y = 180.;
            rot.z = quarter as f32 * 90.;
        }
        _ => return,
    }
    if let Some(model) = unsafe { a.imod.as_mut() } {
        model.view[0].rot = rot;
    } else {
        return;
    }
    imodv_new_model_angles(&rot);
    if a.moveall != 0 {
        for model in &a.mod_ {
            if let Some(view) =
                unsafe { model.as_ptr().as_mut() }.and_then(|model| model.view.first_mut())
            {
                view.rot = rot;
            }
        }
    }
    unsafe { imodv_draw() };
}
/// Original `imodvObjedFreeingExtraObj`.
pub fn imodv_objed_freeing_extra_obj(a: &mut ImodvApp, object: *mut Iobj) {
    // The C call is made only while `objed_dialog` exists.  The current Rust
    // menu boundary has no retained dialog object, so its invocation is the
    // equivalent lifecycle guard.  Form owners that have that state should
    // call the explicit variant below.
    imodv_objed_freeing_extra_obj_when_open(a, object, true);
}

/// `imodvObjedFreeingExtraObj`, with the source's `objed_dialog` guard made
/// explicit because Rust stores it in [`ImodvObjed`] instead of a global.
pub fn imodv_objed_freeing_extra_obj_when_open(
    a: &mut ImodvApp,
    object: *mut Iobj,
    dialog_open: bool,
) {
    if !dialog_open || object.is_null() {
        return;
    }
    let base_count = unsafe {
        a.mod_
            .get(a.cur_mod.max(0) as usize)
            .map(|model| model.as_ref().obj.len() as i32)
    }
    .unwrap_or(0);
    let count = num_editable_objects(a, a.cur_mod);
    for index in base_count..count {
        let candidate = editable_object(a, a.cur_mod, index)
            .map(|candidate| candidate as *mut Iobj)
            .unwrap_or(std::ptr::null_mut());
        if candidate == object {
            if index < a.obj_num {
                a.obj_num -= 1;
            }
            break;
        }
    }
}
/// Original `imodvObjedMeshObject`.
pub fn imodv_objed_mesh_object(editor: &mut ImodvObjed) {
    editor.mesh_busy = true;
}
/// Rust-native app-aware counterpart of `imodvObjedMeshObject`.
pub fn imodv_objed_mesh_object_for_app(editor: &mut ImodvObjed, a: &mut ImodvApp) -> i32 {
    editor.make_doit_for_app(a)
}
/// Original `meshingBusy`.
pub fn meshing_busy(editor: &ImodvObjed) -> bool {
    editor.mesh_busy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::Imod;
    use crate::imod::three_dmod::imodview::ImodView;

    #[test]
    fn editable_extra_objects_follow_model_objects_and_adjust_selection() {
        let mut model = Box::new(Imod::default());
        model.obj.extend([Iobj::default(), Iobj::default()]);
        let mut view = ImodView::default();
        view.extra_obj = vec![
            Iobj {
                flags: IMOD_OBJFLAG_EXTRA_EDIT,
                ..Default::default()
            },
            Iobj {
                flags: IMOD_OBJFLAG_EXTRA_EDIT,
                ..Default::default()
            },
        ];
        view.extra_obj_in_use = vec![1, 1];
        view.num_extra_obj = 2;
        let mut app = ImodvApp::default();
        app.imod = &mut *model;
        app.mod_.push(std::ptr::NonNull::from(&mut *model));
        app.num_mods = 1;
        app.vi = &mut view;

        assert_eq!(num_editable_objects(&app, 0), 4);
        app.obj_num = 3;
        let selected = objed_object(&mut app).map(|object| object as *mut Iobj);
        assert_eq!(selected, Some(&mut view.extra_obj[1] as *mut Iobj));
        assert_eq!(app.obj, selected.unwrap());

        let freed = &mut view.extra_obj[0] as *mut Iobj;
        imodv_objed_freeing_extra_obj_when_open(&mut app, freed, false);
        assert_eq!(app.obj_num, 3);
        imodv_objed_freeing_extra_obj_when_open(&mut app, freed, true);
        assert_eq!(app.obj_num, 2);
    }

    #[test]
    fn objed_object_rejects_invalid_indexes_and_clears_empty_cursor() {
        let mut model = Box::new(Imod::default());
        let mut app = ImodvApp::default();
        app.imod = &mut *model;
        app.mod_.push(std::ptr::NonNull::from(&mut *model));
        app.num_mods = 1;
        let mut stale = Iobj::default();
        app.obj = &mut stale;

        assert!(editable_object(&mut app, 0, -1).is_none());
        assert!(editable_object(&mut app, -1, 0).is_none());
        assert!(objed_object(&mut app).is_none());
        assert!(app.obj.is_null());
    }

    #[test]
    fn draw_data_changes_source_flags() {
        let mut m = Box::new(Imod::default());
        m.obj.push(Iobj::default());
        let mut a = ImodvApp::default();
        a.imod = &mut *m;
        a.mod_.push(std::ptr::NonNull::from(&mut *m));
        a.num_mods = 1;
        imodv_objed_draw_data(&mut a, 2, true);
        assert_ne!(m.obj[0].flags & IMOD_OBJFLAG_MESH, 0);
        imodv_objed_style_data(&mut a, STYLE_FILL, true);
        assert_ne!(m.obj[0].flags & IMOD_OBJFLAG_FILL, 0);
    }
    #[test]
    fn on_off_toggles_model_object() {
        let mut m = Box::new(Imod::default());
        m.obj.push(Iobj::default());
        let mut a = ImodvApp::default();
        a.imod = &mut *m;
        a.mod_.push(std::ptr::NonNull::from(&mut *m));
        a.num_mods = 1;
        objed_toggle_obj(&mut a, 0, false);
        assert_ne!(m.obj[0].flags & IMOD_OBJFLAG_OFF, 0);
    }

    #[test]
    fn simple_slots_update_their_state_and_object_flags() {
        let mut m = Box::new(Imod::default());
        m.obj
            .extend([Iobj::default(), Iobj::default(), Iobj::default()]);
        m.obj[1].flags |= IMOD_OBJFLAG_OPEN;
        m.obj[2].flags |= IMOD_OBJFLAG_SCAT;
        let mut a = ImodvApp::default();
        a.imod = &mut *m;
        a.mod_.push(std::ptr::NonNull::from(&mut *m));
        a.num_mods = 1;
        let mut editor = ImodvObjed {
            edit_all: 1,
            ..Default::default()
        };

        editor.multiple_color_slot(true);
        assert!(editor.multiple_color_ok);
        editor.fill_pnt_toggle_slot(&mut a, true);
        editor.line_thicken_slot(&mut a, true);
        assert_ne!(m.obj[0].flags & IMOD_OBJFLAG_FCOLOR_PNT, 0);
        assert_ne!(m.obj[0].flags & IMOD_OBJFLAG_THICK_CONT, 0);

        editor.auto_new_cont_slot(&mut a, true);
        assert_eq!(m.obj[0].flags & IMOD_OBJFLAG_PLANAR, 0);
        assert_ne!(m.obj[1].flags & IMOD_OBJFLAG_PLANAR, 0);
        assert_eq!(m.obj[2].flags & IMOD_OBJFLAG_PLANAR, 0);

        editor.open_object_slot(&mut a, true);
        assert_ne!(m.obj[0].flags & IMOD_OBJFLAG_OPEN, 0);
        assert_ne!(m.obj[1].flags & IMOD_OBJFLAG_OPEN, 0);
        assert_eq!(m.obj[2].flags & IMOD_OBJFLAG_OPEN, 0);

        editor.mesh_show_slot(&mut a, 1);
        assert_ne!(m.obj[0].flags & IMOD_OBJFLAG_USE_VALUE, 0);
        assert_eq!(m.obj[0].flags & IMOD_OBJFLAG_SCALAR, 0);
        editor.mesh_show_slot(&mut a, 2);
        assert_eq!(m.obj[0].flags & IMOD_OBJFLAG_USE_VALUE, 0);
        assert_ne!(m.obj[0].flags & IMOD_OBJFLAG_SCALAR, 0);
        editor.mesh_false_slot(&mut a, true);
        editor.mesh_constant_slot(&mut a, true);
        editor.mesh_skip_lo_slot(&mut a, true);
        editor.mesh_skip_hi_slot(&mut a, true);
        assert_ne!(m.obj[0].flags & IMOD_OBJFLAG_MCOLOR, 0);
        assert_eq!(
            m.obj[0].matflags2,
            (MATFLAGS2_CONSTANT | MATFLAGS2_SKIP_LOW | MATFLAGS2_SKIP_HIGH) as u8
        );
        editor.material_slot(&mut a, 4, 17, false);
        editor.material_slot(&mut a, 5, 233, false);
        assert_eq!((m.obj[0].valblack, m.obj[0].valwhite), (17, 233));
        editor.set_scalar(&a);
        editor.mesh_level_slot(&mut a, 0, 128, false);
        assert_eq!(m.obj[0].valblack, 128);
    }

    #[test]
    fn global_quality_slot_updates_quality_bits_in_selected_models() {
        let mut current = Box::new(Imod::default());
        let mut other = Box::new(Imod::default());
        current.obj.push(Iobj::default());
        other.obj.push(Iobj::default());
        current.view[0].world = 1 << 2;
        other.view[0].world = 1 << 3;
        let mut a = ImodvApp::default();
        a.imod = &mut *current;
        a.mod_.push(std::ptr::NonNull::from(&mut *current));
        a.mod_.push(std::ptr::NonNull::from(&mut *other));
        a.num_mods = 2;
        a.crosset = 1;
        let mut editor = ImodvObjed::default();

        editor.global_quality_slot(&mut a, 4);

        assert_eq!(current.view[0].world, (1 << 2) | (3 << WORLD_QUALITY_SHIFT));
        assert_eq!(other.view[0].world, (1 << 3) | (3 << WORLD_QUALITY_SHIFT));
    }

    #[test]
    fn move_controls_center_current_object_and_share_axis_rotation() {
        let mut one = Box::new(Imod::default());
        let mut object = Iobj::default();
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 2.,
                    y: 4.,
                    z: 6.,
                },
                Ipoint {
                    x: 6.,
                    y: 10.,
                    z: 14.,
                },
            ],
            ..Default::default()
        });
        one.obj.push(object);
        let mut two = Box::new(Imod::default());
        let mut a = ImodvApp {
            imod: &mut *one,
            mod_: vec![
                std::ptr::NonNull::from(&mut *one),
                std::ptr::NonNull::from(&mut *two),
            ],
            num_mods: 2,
            moveall: 1,
            ..Default::default()
        };
        let mut editor = ImodvObjed::default();

        editor.move_center_slot(&mut a);
        assert_eq!(one.view[0].trans.x, -4.);
        assert_eq!(one.view[0].trans.y, -7.);
        assert_eq!(one.view[0].trans.z, -10.);
        editor.move_axis_slot(&mut a, 5);
        assert_eq!(one.view[0].rot.y, 90.);
        assert_eq!(two.view[0].rot.y, 90.);
    }

    #[test]
    fn enabling_a_new_clip_plane_centers_it_on_the_object() {
        let mut model = Box::new(Imod::default());
        let mut object = Iobj::default();
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 2.,
                    y: 4.,
                    z: 6.,
                },
                Ipoint {
                    x: 6.,
                    y: 10.,
                    z: 14.,
                },
            ],
            ..Default::default()
        });
        model.obj.push(object);
        let mut a = ImodvApp {
            imod: &mut *model,
            mod_: vec![std::ptr::NonNull::from(&mut *model)],
            num_mods: 1,
            ..Default::default()
        };
        let mut editor = ImodvObjed::default();

        editor.clip_toggle_slot(&mut a, true);

        let clips = &model.obj[0].clips;
        assert_eq!(clips.count, 1);
        assert_eq!(clips.flags, 1);
        assert_eq!(
            clips.point[0],
            Ipoint {
                x: -4.,
                y: -7.,
                z: -10.
            }
        );
        assert_eq!(
            clips.normal[0],
            Ipoint {
                x: 0.,
                y: 0.,
                z: -1.
            }
        );
    }

    #[test]
    fn meshing_value_slots_create_and_update_object_parameters() {
        let mut model = Box::new(Imod::default());
        model.obj.push(Iobj::default());
        let mut a = ImodvApp {
            imod: &mut *model,
            mod_: vec![std::ptr::NonNull::from(&mut *model)],
            num_mods: 1,
            ..Default::default()
        };
        let mut editor = ImodvObjed::default();

        editor.make_pass_slot(&mut a, 4);
        editor.make_diam_slot(&mut a, 7.5);
        editor.make_tol_slot(&mut a, 0.25);
        editor.make_zinc_slot(&mut a, 3);
        editor.make_flat_slot(&mut a, 0.75);
        editor.make_low_res = true;
        editor.make_tol_slot(&mut a, 1.5);
        editor.make_zinc_slot(&mut a, 5);

        let params = model.obj[0].mesh_param.as_ref().unwrap();
        assert_eq!(params.passes, 4);
        assert_eq!(params.tube_diameter, 7.5);
        assert_eq!(params.tol_high_res, 0.25);
        assert_eq!(params.incz_high_res, 3);
        assert_eq!(params.flat_crit, 0.75);
        assert_eq!(params.tol_low_res, 1.5);
        assert_eq!(params.incz_low_res, 5);
    }

    #[test]
    fn meshing_option_slots_apply_native_open_object_rules() {
        let mut model = Box::new(Imod::default());
        let mut open = Iobj::default();
        open.flags |= IMOD_OBJFLAG_OPEN;
        model.obj.push(open);
        let mut a = ImodvApp {
            imod: &mut *model,
            mod_: vec![std::ptr::NonNull::from(&mut *model)],
            num_mods: 1,
            ..Default::default()
        };
        let mut editor = ImodvObjed::default();

        editor.make_state_slot(&mut a, 3, true); // tube
        editor.make_state_slot(&mut a, 4, true); // tube caps
        editor.make_state_slot(&mut a, 5, true); // tube domes
        editor.make_state_slot(&mut a, 1, true); // ignored for tube

        let params = model.obj[0].mesh_param.as_ref().unwrap();
        assert_ne!(params.flags & IMESH_MK_TUBE, 0);
        assert_ne!(params.flags & IMESH_MK_CAP_TUBE, 0);
        assert_ne!(params.flags & IMESH_MK_CAP_DOME, 0);
        assert_eq!(params.flags & IMESH_MK_SKIP, 0);
    }

    #[test]
    fn meshing_z_range_uses_native_one_based_input() {
        let mut model = Box::new(Imod::default());
        model.obj.push(Iobj::default());
        let mut a = ImodvApp {
            imod: &mut *model,
            mod_: vec![std::ptr::NonNull::from(&mut *model)],
            num_mods: 1,
            ..Default::default()
        };
        let mut editor = ImodvObjed::default();

        editor.make_zedit_slot(&mut a, "8,3");
        let params = model.obj[0].mesh_param.as_ref().unwrap();
        assert_eq!((params.minz, params.maxz), (2, 7));
        editor.make_zedit_slot(&mut a, "5");
        let params = model.obj[0].mesh_param.as_ref().unwrap();
        assert_eq!((params.minz, params.maxz), (4, 5));
    }

    #[test]
    fn make_doit_runs_selected_simple_contour_stack() {
        let mut model = Box::new(Imod::default());
        let mut object = Iobj::default();
        for z in [0., 1., 2.] {
            let mut contour = crate::imod::libimod::imodel::Icont::default();
            contour.pts = vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ];
            object.cont.push(contour);
        }
        model.obj.push(object);
        let mut a = ImodvApp {
            imod: &mut *model,
            mod_: vec![std::ptr::NonNull::from(&mut *model)],
            num_mods: 1,
            ..Default::default()
        };
        let mut editor = ImodvObjed {
            make_low_res: true,
            ..Default::default()
        };

        assert_eq!(editor.make_doit_for_app(&mut a), 0);
        assert_eq!(model.obj[0].mesh.len(), 1);
        assert_eq!(
            model.obj[0].mesh[0].list[0],
            crate::imod::libimod::imesh::IMOD_MESH_BGNPOLYNORM2
        );
        assert_eq!(
            crate::imod::libimod::imesh::imesh_resol(model.obj[0].mesh[0].flag),
            1
        );
        editor.make_low_res = false;
        assert_eq!(editor.make_doit_for_app(&mut a), 0);
        assert_eq!(model.obj[0].mesh.len(), 2);
        assert_eq!(
            model.obj[0]
                .mesh
                .iter()
                .filter(|mesh| crate::imod::libimod::imesh::imesh_resol(mesh.flag) == 1)
                .count(),
            1
        );
        assert_eq!(
            model.obj[0]
                .mesh
                .iter()
                .filter(|mesh| crate::imod::libimod::imesh::imesh_resol(mesh.flag) == 0)
                .count(),
            1
        );
        assert!(!editor.mesh_busy);
    }

    #[test]
    fn mesh_one_object_runs_synchronous_fallback() {
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = Iobj::default();
        object.cont = vec![contour(0.), contour(1.)];
        let mut editor = ImodvObjed::default();
        assert_eq!(editor.mesh_one_object(&mut object), 0);
        assert!(!object.mesh.is_empty());
        assert!(!editor.mesh_busy);
    }

    #[test]
    fn mesh_parameters_limit_triangles_to_configured_xy_bounds() {
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = Iobj::default();
        object.cont = vec![contour(0.), contour(1.)];
        let mut params = crate::imod::libimod::imesh::MeshParams::default();
        params.xmin = 10.;
        params.xmax = 20.;
        params.ymin = 10.;
        params.ymax = 20.;
        object.mesh_param = Some(params);
        let mut editor = ImodvObjed::default();
        assert_eq!(editor.mesh_one_object(&mut object), 0);
        assert!(
            object
                .mesh
                .iter()
                .all(|mesh| mesh.list.iter().all(|&index| index < 0))
        );
        crate::imod::libmesh::mkmesh::imesh_set_min_max(
            Ipoint {
                x: -1.0e30,
                y: -1.0e30,
                z: -1.0e30,
            },
            Ipoint {
                x: 1.0e30,
                y: 1.0e30,
                z: 1.0e30,
            },
        );
    }

    #[test]
    fn make_do_all_meshes_non_scattered_objects_only() {
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            ..Default::default()
        };
        let mut model = Box::new(Imod::default());
        let mut meshable = Iobj::default();
        meshable.cont = vec![contour(0.), contour(1.)];
        let mut scattered = Iobj::default();
        scattered.flags |= IMOD_OBJFLAG_SCAT;
        scattered.cont = vec![contour(0.), contour(1.)];
        model.obj = vec![meshable, scattered];
        let mut a = ImodvApp {
            imod: &mut *model,
            mod_: vec![std::ptr::NonNull::from(&mut *model)],
            num_mods: 1,
            ..Default::default()
        };
        let mut editor = ImodvObjed::default();

        assert_eq!(editor.make_do_all_for_app(&mut a), 0);
        assert_eq!(model.obj[0].mesh.len(), 1);
        assert!(model.obj[1].mesh.is_empty());
        assert!(!editor.mesh_busy);
    }

    #[test]
    fn step_z_and_mesh_updates_persisted_range_within_model_bounds() {
        let mut model = Box::new(Imod::default());
        model.zmax = 5;
        let mut object = Iobj::default();
        object.cont = vec![
            crate::imod::libimod::imodel::Icont {
                pts: vec![
                    Ipoint {
                        x: 0.,
                        y: 0.,
                        z: 2.,
                    },
                    Ipoint {
                        x: 1.,
                        y: 0.,
                        z: 2.,
                    },
                    Ipoint {
                        x: 0.,
                        y: 1.,
                        z: 2.,
                    },
                ],
                ..Default::default()
            },
            crate::imod::libimod::imodel::Icont {
                pts: vec![
                    Ipoint {
                        x: 0.,
                        y: 0.,
                        z: 3.,
                    },
                    Ipoint {
                        x: 1.,
                        y: 0.,
                        z: 3.,
                    },
                    Ipoint {
                        x: 0.,
                        y: 1.,
                        z: 3.,
                    },
                ],
                ..Default::default()
            },
        ];
        let mut params = crate::imod::libimod::imesh::MeshParams::default();
        params.minz = 1;
        params.maxz = 3;
        object.mesh_param = Some(params);
        model.obj.push(object);
        let mut a = ImodvApp {
            imod: &mut *model,
            mod_: vec![std::ptr::NonNull::from(&mut *model)],
            num_mods: 1,
            ..Default::default()
        };
        let mut editor = ImodvObjed::default();

        assert_eq!(editor.step_zand_mesh_one_for_app(&mut a, 1), 0);
        let params = model.obj[0].mesh_param.as_ref().unwrap();
        assert_eq!((params.minz, params.maxz), (2, 4));
        assert_eq!(model.obj[0].mesh.len(), 1);
        // Source requires the new minimum to be strictly positive, so this
        // attempted move to zero is rejected even though it is in array range.
        assert_eq!(editor.step_zand_mesh_one_for_app(&mut a, -2), 0);
        let params = model.obj[0].mesh_param.as_ref().unwrap();
        assert_eq!((params.minz, params.maxz), (2, 4));

        assert_eq!(editor.make_do_up_slot(&mut a), 0);
        assert_eq!(model.obj[0].mesh_param.as_ref().unwrap().minz, 3);
        assert_eq!(editor.make_do_down_slot(&mut a), 0);
        assert_eq!(model.obj[0].mesh_param.as_ref().unwrap().minz, 2);
    }
}
