//! Translation of `IMOD/3dmod/mv_modeled.cpp` and `mv_modeled.h`.
//!
//! This unit owns model-selection and model-edit-dialog state.  Qt widgets are
//! deliberately represented as data here; their event boundary is in the
//! paired form/window translation.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::imodel::{IMOD_STRSIZE, Imod};
use crate::imod::three_dmod::imodv::{
    ImodvApp, imodv_draw, imodv_finish_chg_unit, imodv_register_model_chg,
};

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
    dialog.pixel_string = format!("{}", model.pixsize);
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
        model.name = [0; IMOD_STRSIZE];
        for (out, byte) in model
            .name
            .iter_mut()
            .zip(name.as_bytes().iter().take(IMOD_STRSIZE - 1))
        {
            *out = *byte;
        }
    }
    update_work_area(a, dialog);
}
/// Original `imodvModeledScale`.
pub fn imodv_modeled_scale(a: &mut ImodvApp, dialog: &mut ImodvModeled, update: bool) {
    if let Ok(value) = dialog
        .pixel_string
        .split_whitespace()
        .next()
        .unwrap_or("")
        .parse::<f32>()
    {
        if let Some(model) = unsafe { a.imod.as_mut() } {
            imodv_register_model_chg();
            model.pixsize = value;
            imodv_finish_chg_unit();
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
    use crate::imod::libimod::imodel::Iobj;
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
}
