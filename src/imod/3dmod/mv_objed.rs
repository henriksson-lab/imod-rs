//! Translation of `IMOD/3dmod/mv_objed.cpp` and `mv_objed.h`.
//!
//! The Qt form controls are data-bound at the `formv_objed.cpp` boundary.  The
//! selection and object mutations below retain the source control semantics so
//! they can be driven by either that form or the native event loop.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OFF, IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_SCAT, Imod, Iobj, Ipoint,
};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_ANTI_ALIAS, IMOD_OBJFLAG_FCOLOR_PNT, IMOD_OBJFLAG_FILL, IMOD_OBJFLAG_MESH,
    IMOD_OBJFLAG_NOLINE, IMOD_OBJFLAG_PLANAR, IMOD_OBJFLAG_SCALE_WDTH, IMOD_OBJFLAG_THICK_CONT,
    IMOD_OBJFLAG_TWO_SIDE, iobj_close, iobj_open, iobj_scat,
};
use crate::imod::three_dmod::imodv::{
    ImodvApp, imodv_draw, imodv_finish_chg_unit, imodv_register_model_chg,
    imodv_register_object_chg,
};

pub const STYLE_POINTS: i32 = 0;
pub const STYLE_LINES: i32 = 1;
pub const STYLE_FILL: i32 = 2;
pub const STYLE_FILL_OUTLINE: i32 = 3;

const OBJTYPE_CLOSED: i32 = 1;
const OBJTYPE_OPEN: i32 = 2;
const OBJTYPE_SCAT: i32 = 4;
const WORLD_QUALITY_SHIFT: u32 = 8;
const WORLD_QUALITY_BITS: u32 = 7 << WORLD_QUALITY_SHIFT;

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
#[derive(Clone, Debug, Default)]
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
}
/// Original `MeshColorBar`; rendering is delegated to the active GL/Qt painter.
#[derive(Clone, Debug, Default)]
pub struct MeshColorBar {
    pub width: i32,
    pub height: i32,
}

/// Original static `objedObject`.
pub fn objed_object(a: &mut ImodvApp) -> Option<&mut Iobj> {
    unsafe { a.imod.as_mut() }.and_then(|m| m.obj.get_mut(a.obj_num.max(0) as usize))
}
/// Original static `numEditableObjects`.
pub fn num_editable_objects(a: &ImodvApp, model: i32) -> i32 {
    unsafe { a.mod_.get(model.max(0) as usize).and_then(|m| m.as_ref()) }
        .map_or(0, |m| m.obj.len() as i32)
}
/// Original static `editableObject`.
pub fn editable_object(a: &mut ImodvApp, model: i32, object: i32) -> Option<&mut Iobj> {
    unsafe {
        a.mod_
            .get_mut(model.max(0) as usize)
            .and_then(|m| m.as_mut())
    }
    .and_then(|m| m.obj.get_mut(object.max(0) as usize))
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
            *out = b as i8;
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
        .filter_map(|m| unsafe { m.as_ref() })
        .map(|m| m.obj.len())
        .max()
        .unwrap_or(0)
        .min(100)
}
/// Original static `finishChangeAndDraw`.
pub fn finish_change_and_draw(a: &mut ImodvApp, do_objset: bool, draw_images: bool) {
    imodv_finish_chg_unit();
    unsafe { imodv_draw() };
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
                    .and_then(|m| m.as_mut())
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
        if let Some(o) = objed_object(a) {
            o.drawmode = value;
        }
        finish_change_and_draw(a, false, true);
    }
    /// Original `ImodvObjed::meshFalseSlot`.
    pub fn mesh_false_slot(&mut self, _a: &mut ImodvApp, _state: bool) {}
    /// Original `ImodvObjed::meshConstantSlot`.
    pub fn mesh_constant_slot(&mut self, _a: &mut ImodvApp, _state: bool) {}
    /// Original `ImodvObjed::meshLevelSlot`.
    pub fn mesh_level_slot(
        &mut self,
        _a: &mut ImodvApp,
        _which: i32,
        _value: i32,
        _dragging: bool,
    ) {
    }
    /// Original `ImodvObjed::meshSkipLoSlot`.
    pub fn mesh_skip_lo_slot(&mut self, _a: &mut ImodvApp, _state: bool) {}
    /// Original `ImodvObjed::meshSkipHiSlot`.
    pub fn mesh_skip_hi_slot(&mut self, _a: &mut ImodvApp, _state: bool) {}
    /// Original `ImodvObjed::clipShowSlot`.
    pub fn clip_show_slot(&mut self, a: &mut ImodvApp, state: bool) {
        a.draw_clip = state as i32;
        unsafe { imodv_draw() };
    }
    /// Original `ImodvObjed::clipGlobalSlot`.
    pub fn clip_global_slot(&mut self, _a: &mut ImodvApp, _value: i32) {}
    /// Original `ImodvObjed::clipSkipSlot`.
    pub fn clip_skip_slot(&mut self, _a: &mut ImodvApp, _state: bool) {}
    /// Original `ImodvObjed::clipPlaneSlot`.
    pub fn clip_plane_slot(&mut self, _a: &mut ImodvApp, _value: i32) {}
    /// Original `ImodvObjed::clipResetSlot`.
    pub fn clip_reset_slot(&mut self, _a: &mut ImodvApp, _which: i32) {}
    /// Original `ImodvObjed::clipInvertSlot`.
    pub fn clip_invert_slot(&mut self, _a: &mut ImodvApp) {}
    /// Original `ImodvObjed::clipToggleSlot`.
    pub fn clip_toggle_slot(&mut self, _a: &mut ImodvApp, _state: bool) {}
    /// Original `ImodvObjed::clipMoveAllSlot`.
    pub fn clip_move_all_slot(&mut self, _a: &mut ImodvApp, _state: bool) {}
    /// Original `ImodvObjed::moveCenterSlot`.
    pub fn move_center_slot(&mut self, _a: &mut ImodvApp) {}
    /// Original `ImodvObjed::moveAxisSlot`.
    pub fn move_axis_slot(&mut self, _a: &mut ImodvApp, _which: i32) {}
    /// Original `ImodvObjed::subsetSlot`.
    pub fn subset_slot(&mut self, a: &mut ImodvApp, which: i32) {
        a.current_subset = which;
        unsafe { imodv_draw() };
    }
    /// Original meshing slots: these preserve busy sequencing pending the paired mesh unit.
    pub fn make_pass_slot(&mut self, _value: i32) {}
    pub fn make_diam_slot(&mut self, _value: f64) {}
    pub fn make_tol_slot(&mut self, _value: f64) {}
    pub fn make_zinc_slot(&mut self, _value: i32) {}
    pub fn make_flat_slot(&mut self, _value: f64) {}
    pub fn make_spin_changed(&mut self, _which: i32, _value: f64) {}
    pub fn make_state_slot(&mut self, _which: i32) {}
    pub fn make_zedit_slot(&mut self) {}
    pub fn make_doit_slot(&mut self) {
        self.mesh_busy = true;
    }
    pub fn make_do_all_slot(&mut self) {
        self.mesh_busy = true;
    }
    pub fn make_do_up_slot(&mut self) {
        self.mesh_busy = true;
    }
    pub fn make_do_down_slot(&mut self) {
        self.mesh_busy = true;
    }
    pub fn start_meshing_next(&mut self) -> i32 {
        (!self.mesh_busy) as i32
    }
    pub fn mesh_one_object(&mut self, _obj: &mut Iobj) -> i32 {
        self.mesh_busy = true;
        0
    }
    pub fn update_meshing(&mut self, _ob: i32) {}
    pub fn step_zand_mesh_one(&mut self, _dir: i32) {}
}
/// Original `imodvObjedDrawClipPlane`.
pub fn imodv_objed_draw_clip_plane(a: &mut ImodvApp, state: bool) {
    a.draw_clip = state as i32;
    unsafe { imodv_draw() };
}
/// Original `imodvObjedToggleClip`.
pub fn imodv_objed_toggle_clip(_a: &mut ImodvApp, _global: i32, _plane: i32) {}
/// Original `imodvObjedMoveToAxis`.
pub fn imodv_objed_move_to_axis(_a: &mut ImodvApp, _which: i32) {}
/// Original `imodvObjedFreeingExtraObj`.
pub fn imodv_objed_freeing_extra_obj(_a: &mut ImodvApp, _obj: *mut Iobj) {}
/// Original `imodvObjedMeshObject`.
pub fn imodv_objed_mesh_object(editor: &mut ImodvObjed) {
    editor.mesh_busy = true;
}
/// Original `meshingBusy`.
pub fn meshing_busy(editor: &ImodvObjed) -> bool {
    editor.mesh_busy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::Imod;
    #[test]
    fn draw_data_changes_source_flags() {
        let mut m = Box::new(Imod::default());
        m.obj.push(Iobj::default());
        let mut a = ImodvApp::default();
        a.imod = &mut *m;
        a.mod_.push(&mut *m);
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
        a.mod_.push(&mut *m);
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
        a.mod_.push(&mut *m);
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
        a.mod_.push(&mut *current);
        a.mod_.push(&mut *other);
        a.num_mods = 2;
        a.crosset = 1;
        let mut editor = ImodvObjed::default();

        editor.global_quality_slot(&mut a, 4);

        assert_eq!(current.view[0].world, (1 << 2) | (3 << WORLD_QUALITY_SHIFT));
        assert_eq!(other.view[0].world, (1 << 3) | (3 << WORLD_QUALITY_SHIFT));
    }
}
