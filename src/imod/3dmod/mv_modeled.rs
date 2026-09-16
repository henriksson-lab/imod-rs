//! Translation of `IMOD/3dmod/mv_modeled.cpp` and `mv_modeled.h`.
//!
//! This unit owns model-selection and model-edit-dialog state.  Qt widgets are
//! deliberately represented as data here; their event boundary is in the
//! paired form/window translation.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::imodel::{imod_units, Imod, IMOD_STRSIZE};
use crate::imod::three_dmod::imodv::{
    imodv_draw, imodv_finish_chg_unit, imodv_register_model_chg, ImodvApp,
};
use crate::imod::three_dmod::model_edit::set_pixsize_and_units;
use crate::imod::three_dmod::utilities::imodw_given_name;

/// Original static `imodv_modeled`.
#[derive(Default, Clone, Debug)]
pub struct ImodvModeled {
    pub dialog_open: bool,
    pub model_number: i32,
    pub model_count: i32,
    pub file_name: String,
    pub internal_name: String,
    pub pixel_string: String,
    pub view_selection: i32,
    pub move_selection: i32,
    pub edit_selection: i32,
}

/// Original static `updateWorkArea`.
pub fn update_work_area(a: &ImodvApp, dialog: &mut ImodvModeled) {
    let Some(model) = (unsafe { a.imod.as_ref() }) else {
        return;
    };
    dialog.model_number = a.cur_mod + 1;
    dialog.model_count = a.num_mods;
    dialog.internal_name = model
        .name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8 as char)
        .collect();
    dialog.file_name = imodw_given_name(" ", model.file_name.as_deref()).unwrap_or_default();
    dialog.pixel_string = format!("{} {}", model.pixsize, imod_units(model));
}

/// Original `imodvModeledDone`.
pub fn imodv_modeled_done(dialog: &mut ImodvModeled) {
    dialog.dialog_open = false;
}
/// Original `imodvModeledClosing`.
pub fn imodv_modeled_closing(a: &mut ImodvApp, dialog: &mut ImodvModeled) {
    imodv_modeled_scale(a, dialog, false);
    dialog.dialog_open = false;
}
/// Original `imodvModelEditDialog`.
pub fn imodv_model_edit_dialog(a: &mut ImodvApp, state: i32, dialog: &mut ImodvModeled) {
    if state == 0 {
        dialog.dialog_open = false;
        return;
    }
    dialog.dialog_open = true;
    dialog.view_selection = a.drawall;
    dialog.move_selection = a.moveall;
    dialog.edit_selection = a.crosset;
    update_work_area(a, dialog);
}
/// Original `imodvSelectModel`.
pub fn imodv_select_model(a: &mut ImodvApp, ncm: i32) -> i32 {
    if a.num_mods <= 0 || a.mod_.is_empty() {
        return a.cur_mod;
    }
    let selected = ncm.clamp(0, a.num_mods - 1).min(a.mod_.len() as i32 - 1);
    a.cur_mod = selected;
    a.imod = a.mod_[selected as usize].as_ptr();
    if let Some(model) = unsafe { a.imod.as_mut() } {
        if a.obj_num < 0 || a.obj_num as usize >= model.obj.len() {
            a.obj_num = 0;
        }
        a.obj = model
            .obj
            .get_mut(a.obj_num as usize)
            .map_or(std::ptr::null_mut(), |o| o);
    }
    // `mv_modeled.cpp:114-120`: standalone 3dmodv has one fake `ImodView`
    // shared by the loaded models.  Its edit history and selection indexes
    // belong to the previously selected model and must not leak across this
    // switch.
    if a.standalone != 0 {
        if let Some(view) = unsafe { a.vi.as_mut() } {
            view.imod = a.imod;
            if let Some(undo) = view.undo.as_mut() {
                undo.clear_units();
            }
            view.selection_list.clear();
        }
    }
    // Native also refreshes the model widgets and caption here.  Those
    // presentation calls are routed by the Rust-native viewer host; the
    // source-owned bounding-box state belongs to this selection transition.
    crate::imod::three_dmod::mv_menu::imodv_add_bounding_box(a, -1);
    unsafe { imodv_draw() };
    a.cur_mod
}
/// Original `imodvModeledNumber`.
pub fn imodv_modeled_number(a: &mut ImodvApp, which: i32, dialog: &mut ImodvModeled) {
    imodv_select_model(a, which - 1);
    update_work_area(a, dialog);
}
/// Original `imodvModeledMove`.
pub fn imodv_modeled_move(a: &mut ImodvApp, item: i32) {
    a.moveall = item;
}
/// Original `imodvModeledView`.
pub fn imodv_modeled_view(a: &mut ImodvApp, item: i32) {
    a.drawall = item;
    unsafe { imodv_draw() };
}
/// Original `imodvModeledEdit`.
pub fn imodv_modeled_edit(a: &mut ImodvApp, item: i32) {
    a.crosset = item;
}
/// Original `imodvModeledSameScale`.
pub fn imodv_modeled_same_scale(a: &mut ImodvApp) {
    if a.standalone == 0 {
        return;
    }
    let rad = unsafe { a.imod.as_ref() }
        .and_then(|m| m.view.first())
        .map(|v| v.rad);
    if let Some(rad) = rad {
        for model in &a.mod_ {
            if let Some(view) =
                unsafe { model.as_ptr().as_mut() }.and_then(|model| model.view.first_mut())
            {
                view.rad = rad;
            }
        }
    }
    unsafe { imodv_draw() };
}
/// Original `imodvModeledName`.
pub fn imodv_modeled_name(a: &mut ImodvApp, name: &str, dialog: &mut ImodvModeled) {
    if let Some(model) = unsafe { a.imod.as_mut() } {
        let bytes = &name.as_bytes()[..name.len().min(IMOD_STRSIZE - 1)];
        model.name[..bytes.len()].copy_from_slice(bytes);
        // The C function writes only the terminating byte, leaving the
        // remainder of this fixed serialized field untouched.
        model.name[bytes.len()] = 0;
    }
    update_work_area(a, dialog);
}
/// Original `imodvModeledScale`.
pub fn imodv_modeled_scale(a: &mut ImodvApp, dialog: &mut ImodvModeled, update: bool) {
    if !dialog.pixel_string.is_empty() {
        if let Some(model) = unsafe { a.imod.as_mut() } {
            imodv_register_model_chg();
            imodv_finish_chg_unit();
            set_pixsize_and_units(model, &dialog.pixel_string);
        }
    }
    if update {
        update_work_area(a, dialog);
    }
}
/// Original `imeSetViewData`.
pub fn ime_set_view_data(dialog: &mut ImodvModeled, wi: i32) {
    if dialog.dialog_open {
        dialog.view_selection = wi;
    }
}
/// Original `imodvPixelChanged`.
pub fn imodv_pixel_changed(a: &ImodvApp, dialog: &mut ImodvModeled) {
    if dialog.dialog_open {
        update_work_area(a, dialog);
    }
}
/// Original `imodvModelDrawRange`.
pub fn imodv_model_draw_range(a: &ImodvApp, mstart: &mut i32, mend: &mut i32) {
    *mstart = a.cur_mod;
    *mend = a.cur_mod;
    match a.drawall {
        2 => *mend = (a.cur_mod + 1).min(a.num_mods - 1),
        1 => *mstart = (a.cur_mod - 1).max(0),
        3 => {
            *mstart = 0;
            *mend = a.num_mods - 1;
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::{Iindex, Iobj, IMOD_UNIT_NM};
    use crate::imod::three_dmod::imodview::ImodView;
    use crate::imod::three_dmod::undoredo::{UndoRedo, UndoState, UndoUnit};
    #[test]
    fn model_draw_range_matches_drawall_options() {
        let mut a = ImodvApp::default();
        a.cur_mod = 2;
        a.num_mods = 5;
        for (opt, want) in [(0, (2, 2)), (1, (1, 2)), (2, (2, 3)), (3, (0, 4))] {
            a.drawall = opt;
            let (mut s, mut e) = (0, 0);
            imodv_model_draw_range(&a, &mut s, &mut e);
            assert_eq!((s, e), want);
        }
    }
    #[test]
    fn selection_clamps_and_selects_object() {
        let mut one = Box::new(Imod::default());
        one.obj.push(Iobj::default());
        let mut two = Box::new(Imod::default());
        two.obj.push(Iobj::default());
        let mut a = ImodvApp::default();
        a.mod_ = vec![
            std::ptr::NonNull::from(&mut *one),
            std::ptr::NonNull::from(&mut *two),
        ];
        a.num_mods = 2;
        assert_eq!(imodv_select_model(&mut a, 99), 1);
        assert!(std::ptr::eq(a.imod, &mut *two));
    }

    #[test]
    fn standalone_selection_repoints_view_and_clears_edit_state() {
        let mut one = Box::new(Imod::default());
        let mut two = Box::new(Imod::default());
        let mut view = ImodView::default();
        let mut undo = UndoRedo::new();
        undo.unit_list.push(UndoUnit {
            before: UndoState {
                index: Iindex::default(),
                size: Iindex::default(),
            },
            after: UndoState {
                index: Iindex::default(),
                size: Iindex::default(),
            },
            changes: Vec::new(),
        });
        view.undo = Some(Box::new(undo));
        view.selection_list.push(Iindex {
            object: 2,
            contour: 3,
            point: 4,
        });
        let mut a = ImodvApp {
            mod_: vec![
                std::ptr::NonNull::from(&mut *one),
                std::ptr::NonNull::from(&mut *two),
            ],
            num_mods: 2,
            standalone: 1,
            vi: &mut view,
            ..Default::default()
        };

        assert_eq!(imodv_select_model(&mut a, 1), 1);
        assert!(std::ptr::eq(view.imod, &mut *two));
        assert!(view.selection_list.is_empty());
        assert!(view.undo.as_ref().unwrap().unit_list.is_empty());
    }

    #[test]
    fn model_editor_preserves_native_name_tail_and_pixel_units() {
        let mut model = Imod::default();
        model.name = [b'Z'; IMOD_STRSIZE];
        model.file_name = Some("models/example.mod".into());
        let mut app = ImodvApp {
            imod: &mut model,
            num_mods: 1,
            ..Default::default()
        };
        let mut dialog = ImodvModeled::default();

        imodv_modeled_name(&mut app, "A", &mut dialog);
        assert_eq!(&model.name[..3], b"A\0Z");
        dialog.pixel_string = "2.5 nm".into();
        imodv_modeled_scale(&mut app, &mut dialog, true);
        assert_eq!(model.pixsize, 2.5);
        assert_eq!(model.units, IMOD_UNIT_NM);
        assert_eq!(dialog.file_name, "  example.mod");
        assert_eq!(dialog.pixel_string, "2.5 nm");
    }
}
