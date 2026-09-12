//! Translation of `IMOD/3dmod/mv_menu.cpp` and `mv_menu.h`.
//!
//! Qt file pickers, help pages, docks, colour selector widgets, the snapshot
//! encoder, and the active GL widget are deliberately parameters at this
//! boundary.  The state and menu dispatch below follow the source unit; no
//! synthetic replacement UI is introduced.
#![allow(dead_code, unused_variables)]

use std::fs::{File, remove_file, rename};
use std::path::Path;

use crate::imod::libcfshr::b3dutil::set_or_clear_flags;
use crate::imod::libimod::icont::{imod_contour_clear_points, imod_contour_new, imod_contours_new};
use crate::imod::libimod::imodel::{IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_SCAT, Imod, Ipoint};
use crate::imod::libimod::imodel_files::{imod_read, imod_write};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_ANTI_ALIAS, IMOD_OBJFLAG_EXTRA_EDIT, IMOD_OBJFLAG_EXTRA_MODV, IMOD_OBJFLAG_FILL,
    IMOD_OBJFLAG_MESH, IMOD_OBJFLAG_MODV_ONLY, IMOD_OBJFLAG_NOLINE, IMOD_OBJFLAG_WILD,
    imod_object_add_contour, imod_object_default, imod_object_get_bbox,
};
use crate::imod::libimod::ipoint::{imod_point_append_xyz, imod_point_set_size};
use crate::imod::libimod::iview::VIEW_WORLD_LIGHT;
use crate::imod::three_dmod::imodv::{
    ImodvApp, imodv_draw, imodv_finish_chg_unit, imodv_register_model_chg,
};
use crate::imod::three_dmod::imodview::{
    ImodView, ivw_free_extra_object, ivw_get_an_extra_object, ivw_get_free_extra_object_number,
};
use crate::imod::three_dmod::mv_modeled::imodv_select_model;
use crate::imod::three_dmod::mv_objed::{imodv_objed_freeing_extra_obj, imodv_objed_new_view};
use crate::imod::three_dmod::mv_views::{
    VIEW_WORLD_INVERT_Z, VIEW_WORLD_LABELS, VIEW_WORLD_LOWRES, VIEW_WORLD_WIREFRAME,
};
use crate::imod::three_dmod::mv_window::*;

/// The Qt `ColorSelector` state owned by `ImodvBkgColor`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImodvBkgColor {
    pub selector_open: bool,
    pub red: i32,
    pub green: i32,
    pub blue: i32,
}

/// `ImodvBkgColor::openDialog`.
pub fn imodv_bkg_color_open_dialog(color: &mut ImodvBkgColor, rgb: [i32; 3]) {
    color.selector_open = true;
    color.red = rgb[0];
    color.green = rgb[1];
    color.blue = rgb[2];
}
/// `ImodvBkgColor::ImodvBkgColor`.
pub fn imodv_bkg_color_new() -> ImodvBkgColor {
    ImodvBkgColor::default()
}
/// `ImodvBkgColor::newColorSlot`.
pub fn imodv_bkg_color_new_color_slot(color: &mut ImodvBkgColor, red: i32, green: i32, blue: i32) {
    color.red = red;
    color.green = green;
    color.blue = blue;
    unsafe { imodv_draw() };
}
/// `ImodvBkgColor::doneSlot`.
pub fn imodv_bkg_color_done_slot(color: &mut ImodvBkgColor) {
    if color.selector_open {
        color.selector_open = false;
    }
}
/// `ImodvBkgColor::closingSlot`.
pub fn imodv_bkg_color_closing_slot(color: &mut ImodvBkgColor) {
    color.selector_open = false;
}
/// `ImodvBkgColor::keyPressSlot`; the paired `mv_input` key dispatch is the caller boundary.
pub fn imodv_bkg_color_key_press_slot(_color: &mut ImodvBkgColor, event: KeyEvent) -> KeyEvent {
    event
}
/// `ImodvBkgColor::keyReleaseSlot`; the paired `mv_input` key dispatch is the caller boundary.
pub fn imodv_bkg_color_key_release_slot(_color: &mut ImodvBkgColor, event: KeyEvent) -> KeyEvent {
    event
}

/// `imodvMenuBgcolor`.
pub fn imodv_menu_bgcolor(state: i32, color: &mut ImodvBkgColor, rgb: [i32; 3]) {
    if state != 0 {
        if !color.selector_open {
            imodv_bkg_color_open_dialog(color, rgb);
        }
    } else {
        imodv_bkg_color_done_slot(color);
    }
}

/// `imodvEditMenu`.  Dialog/dock construction is an explicit Qt boundary;
/// the returned action names are the source calls that must be performed.
pub fn imodv_edit_menu(
    item: usize,
    color: &mut ImodvBkgColor,
    rgb: [i32; 3],
) -> Option<&'static str> {
    match item {
        VEDIT_MENU_OBJECTS => Some("objed"),
        VEDIT_MENU_CONTROLS => Some("imodv_control"),
        VEDIT_MENU_ROTATION => Some("open_rotation_tool"),
        VEDIT_MENU_OBJLIST => Some("imodv_object_list_dialog"),
        VEDIT_MENU_BKG => {
            imodv_menu_bgcolor(1, color, rgb);
            None
        }
        VEDIT_MENU_MODELS => Some("imodv_model_edit_dialog"),
        VEDIT_MENU_VIEWS => Some("imodv_view_edit_dialog"),
        VEDIT_MENU_IMAGE => Some("mv_image_edit_dialog"),
        VEDIT_MENU_ISOSURFACE => Some("imodv_isosurface_edit_dialog"),
        VEDIT_MENU_SAVE_DOCK => Some("save_stack_state_to_settings"),
        VEDIT_MENU_REOPEN_DOCK => Some("restore_stack_from_settings"),
        _ => None,
    }
}

/// `imodvHelpMenu`; browser/message-box work remains the native Qt boundary.
pub fn imodv_help_menu(item: usize) -> Option<&'static str> {
    match item {
        VHELP_MENU_MENUS => Some("modvMenus.html#TOP"),
        VHELP_MENU_KEYBOARD => Some("modvKeyboard.html#TOP"),
        VHELP_MENU_MOUSE => Some("modvMouse.html#TOP"),
        VHELP_MENU_ABOUT => Some("3dmodv Version"),
        _ => None,
    }
}

/// `imodvLoadModel`, after the source's Qt filename picker has selected `path`.
pub fn imodv_load_model(a: &mut ImodvApp, path: Option<&Path>) -> i32 {
    if a.standalone == 0 {
        return -1;
    }
    let Some(path) = path else {
        return 1;
    };
    let Ok(mut model) = imod_read(path) else {
        return -1;
    };
    model.cview = if model.cview != 0 { model.cview } else { 1 };
    let raw = Box::into_raw(Box::new(model));
    a.mod_.push(raw);
    a.num_mods += 1;
    imodv_select_model(a, a.num_mods - 1);
    0
}

/// `writeOpenedModelFile`, after the source's `fopen` has produced `file`.
pub fn write_opened_model_file(a: &mut ImodvApp, file: &mut File) -> i32 {
    let Some(model) = (unsafe { a.imod.as_ref() }) else {
        return 1;
    };
    imod_write(model, file).err().map_or(0, |_| 1)
}
/// `imodvFileSave`, after resolving the current source filename at the Qt/path boundary.
pub fn imodv_file_save(a: &mut ImodvApp, filename: &Path) -> i32 {
    let backup = filename.with_file_name(format!(
        "{}~",
        filename.file_name().unwrap_or_default().to_string_lossy()
    ));
    if filename.exists() {
        let _ = remove_file(&backup);
        if rename(filename, &backup).is_err() {
            return 1;
        }
    }
    let error = File::create(filename)
        .ok()
        .map_or(1, |mut f| write_opened_model_file(a, &mut f));
    if error != 0 {
        let _ = remove_file(filename);
        let _ = rename(&backup, filename);
    }
    error
}
/// `imodvSaveModelAs`, after the source's `imodPlugGetSaveName` Qt boundary.
pub fn imodv_save_model_as(a: &mut ImodvApp, filename: Option<&Path>) -> i32 {
    let Some(filename) = filename else {
        return 0;
    };
    imodv_file_save(a, filename)
}

/// `imodvFileMenu`.  File picker, snapshot encoding, directory chooser, movie
/// and close window calls are direct UI/backend boundaries represented by the returned action.
pub fn imodv_file_menu(item: usize) -> Option<&'static str> {
    match item {
        VFILE_MENU_LOAD => Some("imodv_load_model"),
        VFILE_MENU_SAVE => Some("imodv_file_save"),
        VFILE_MENU_SAVEAS => Some("imodv_save_model_as"),
        VFILE_MENU_SNAPRGB => Some("snapshot_rgb"),
        VFILE_MENU_SNAPTIFF => Some("snapshot_tiff"),
        VFILE_MENU_ZEROSNAP => Some("imodv_reset_snap"),
        VFILE_MENU_SNAPDIR => Some("b3d_set_snap_directory"),
        VFILE_MENU_MOVIE => Some("mv_movie_dialog"),
        VFILE_MENU_SEQUENCE => Some("mv_movie_sequence_dialog"),
        VFILE_MENU_QUIT => Some("close"),
        _ => None,
    }
}

/// `imodvViewMenu`.  Form-opening and GL-widget replacement cases are explicit
/// native boundaries; flag mutations and extra-object ownership are preserved here.
pub fn imodv_view_menu(
    a: &mut ImodvApp,
    item: usize,
    window: Option<&mut ImodvWindow>,
) -> Option<&'static str> {
    match item {
        VVIEW_MENU_DB => {
            a.dbl_buf = 1 - a.dbl_buf;
            if let Some(w) = window {
                w.set_enabled_menu_item(
                    VVIEW_MENU_TRANSBKGD,
                    a.dbl_buf != 0 && (a.enable_depth_dbal >= 0 || a.enable_depth_dbst_al >= 0),
                );
            }
            Some("imodv_setbuffer")
        }
        VVIEW_MENU_TRANSBKGD => {
            a.trans_bkgd = if a.trans_bkgd != 0 { 0 } else { a.alpha_visual };
            if let Some(w) = window {
                w.set_checkable_item(VVIEW_MENU_TRANSBKGD, a.trans_bkgd != 0);
                w.set_enabled_menu_item(
                    VVIEW_MENU_DB,
                    a.db_possible != 0 && a.enable_depth_sb >= 0 && a.trans_bkgd == 0,
                );
            }
            Some("imodv_setbuffer")
        }
        VVIEW_MENU_INVERTZ => {
            toggle_world_flag(
                a,
                &mut window.map(|w| w),
                VIEW_WORLD_INVERT_Z,
                VVIEW_MENU_INVERTZ,
                0,
            );
            None
        }
        VVIEW_MENU_LIGHTING => {
            toggle_world_flag(
                a,
                &mut window.map(|w| w),
                VIEW_WORLD_LIGHT,
                VVIEW_MENU_LIGHTING,
                1,
            );
            None
        }
        VVIEW_MENU_WIREFRAME => {
            toggle_world_flag(
                a,
                &mut window.map(|w| w),
                VIEW_WORLD_WIREFRAME,
                VVIEW_MENU_WIREFRAME,
                2,
            );
            None
        }
        VVIEW_MENU_LOWRES => {
            toggle_world_flag(
                a,
                &mut window.map(|w| w),
                VIEW_WORLD_LOWRES,
                VVIEW_MENU_LOWRES,
                3,
            );
            None
        }
        VVIEW_MENU_STEREO => Some("imodv_stereo_edit_dialog"),
        VVIEW_MENU_DEPTH => Some("imodv_depth_cue_edit_dialog"),
        VVIEW_MENU_SCALEBAR => Some("scale_bar_open"),
        VVIEW_MENU_RESIZE => Some("open_resize_tool"),
        VVIEW_MENU_BOUNDBOX | VVIEW_MENU_OBJBOUND => {
            let bound_for_object = item == VVIEW_MENU_OBJBOUND;
            let object = if bound_for_object { a.obj_num } else { -1 };
            if bound_for_object && a.obj_num < 0 {
                return None;
            }
            let mut extra_number = if bound_for_object {
                a.obj_bound_extra_obj
            } else {
                a.bound_box_extra_obj
            };
            let mut free_extra_object = true;
            if extra_number <= 0 && !a.vi.is_null() {
                extra_number =
                    ivw_get_free_extra_object_number(unsafe { &mut *(a.vi as *mut ImodView) });
                if extra_number <= 0 {
                    return None;
                }
                let mut initialized = false;
                if let Some(extra_object) =
                    ivw_get_an_extra_object(unsafe { &mut *(a.vi as *mut ImodView) }, extra_number)
                {
                    imod_object_default(extra_object);
                    let name: &[u8] = if bound_for_object {
                        b"Current object bounding box extra object\0"
                    } else {
                        b"Volume bounding box extra object\0"
                    };
                    for (out, input) in extra_object.name.iter_mut().zip(name) {
                        *out = *input as i8;
                    }
                    extra_object.cont = imod_contours_new(6).unwrap_or_default();
                    if extra_object.cont.len() == 6 {
                        extra_object.flags |= IMOD_OBJFLAG_OPEN
                            | IMOD_OBJFLAG_WILD
                            | IMOD_OBJFLAG_EXTRA_MODV
                            | IMOD_OBJFLAG_EXTRA_EDIT
                            | IMOD_OBJFLAG_ANTI_ALIAS
                            | IMOD_OBJFLAG_MODV_ONLY;
                        extra_object.red = 1.;
                        extra_object.green = if bound_for_object { 0.6 } else { 1. };
                        extra_object.blue = 0.;
                        extra_object.linewidth = 2;
                        initialized = true;
                    }
                }
                if initialized && imodv_add_bounding_box(a, object) == 0 {
                    free_extra_object = false;
                }
            }
            if free_extra_object && extra_number > 0 && !a.vi.is_null() {
                let vi = unsafe { &mut *(a.vi as *mut ImodView) };
                ivw_free_extra_object(vi, extra_number);
                extra_number = 0;
            }
            if bound_for_object {
                a.obj_bound_extra_obj = extra_number;
            } else {
                a.bound_box_extra_obj = extra_number;
            }
            if let Some(w) = window {
                w.set_checkable_item(item, !free_extra_object);
            }
            imodv_objed_new_view(a);
            unsafe { imodv_draw() };
            None
        }
        VVIEW_MENU_LABELS => {
            toggle_world_flag(
                a,
                &mut window.map(|w| w),
                VIEW_WORLD_LABELS,
                VVIEW_MENU_LABELS,
                4,
            );
            None
        }
        VVIEW_MENU_CURPNT => {
            let mut free_extra_object = true;
            if a.cur_point_extra_obj <= 0 && !a.vi.is_null() {
                let vi = unsafe { &mut *(a.vi as *mut ImodView) };
                a.cur_point_extra_obj = ivw_get_free_extra_object_number(vi);
                if a.cur_point_extra_obj <= 0 {
                    return None;
                }
                if let Some(extra_object) = ivw_get_an_extra_object(vi, a.cur_point_extra_obj) {
                    imod_object_default(extra_object);
                    for (out, input) in extra_object
                        .name
                        .iter_mut()
                        .zip(b"Current point extra object\0")
                    {
                        *out = *input as i8;
                    }
                    extra_object.flags |= IMOD_OBJFLAG_SCAT
                        | IMOD_OBJFLAG_MESH
                        | IMOD_OBJFLAG_NOLINE
                        | IMOD_OBJFLAG_FILL
                        | IMOD_OBJFLAG_EXTRA_MODV
                        | IMOD_OBJFLAG_EXTRA_EDIT
                        | IMOD_OBJFLAG_MODV_ONLY;
                    extra_object.pdrawsize = 7;
                    extra_object.red = 1.;
                    extra_object.green = 0.;
                    extra_object.blue = 0.;
                    extra_object.quality = 4;
                    if let Some(mut contour) = imod_contour_new() {
                        imod_point_append_xyz(&mut contour, 0., 0., 0.);
                        imod_point_set_size(&mut contour, 0, 5.);
                        if !contour.pts.is_empty()
                            && !contour.sizes.is_empty()
                            && imod_object_add_contour(extra_object, contour) >= 0
                        {
                            free_extra_object = false;
                        }
                    }
                }
            } else if a.cur_point_extra_obj > 0 && !a.vi.is_null() {
                let extra_object = ivw_get_an_extra_object(
                    unsafe { &mut *(a.vi as *mut ImodView) },
                    a.cur_point_extra_obj,
                )
                .map(|object| object as *mut _);
                if let Some(extra_object) = extra_object {
                    imodv_objed_freeing_extra_obj(a, extra_object);
                }
            }
            if free_extra_object && a.cur_point_extra_obj > 0 && !a.vi.is_null() {
                let vi = unsafe { &mut *(a.vi as *mut ImodView) };
                ivw_free_extra_object(vi, a.cur_point_extra_obj);
                a.cur_point_extra_obj = 0;
            }
            if let Some(w) = window {
                w.set_checkable_item(VVIEW_MENU_CURPNT, !free_extra_object);
            }
            imodv_objed_new_view(a);
            unsafe { imodv_draw() };
            None
        }
        _ => None,
    }
}

/// `toggleWorldFlag`.
pub fn toggle_world_flag(
    a: &mut ImodvApp,
    window: &mut Option<&mut ImodvWindow>,
    mask: u32,
    menu_id: usize,
    field: usize,
) {
    imodv_register_model_chg();
    imodv_finish_chg_unit();
    let value = match field {
        0 => {
            a.invert_z = 1 - a.invert_z;
            a.invert_z
        }
        1 => {
            a.lighting = 1 - a.lighting;
            a.lighting
        }
        2 => {
            a.wireframe = 1 - a.wireframe;
            a.wireframe
        }
        3 => {
            a.lowres = 1 - a.lowres;
            a.lowres
        }
        _ => {
            a.draw_labels = 1 - a.draw_labels;
            a.draw_labels
        }
    };
    if let Some(model) = unsafe { a.imod.as_mut() } {
        if let Some(view) = model.view.first_mut() {
            set_or_clear_flags(&mut view.world, mask, value);
        }
    }
    if let Some(w) = window {
        w.set_checkable_item(menu_id, value != 0);
    }
    unsafe { imodv_draw() };
}

/// `imodvMenuLight`.
pub fn imodv_menu_light(window: &mut ImodvWindow, value: i32) {
    window.set_checkable_item(VVIEW_MENU_LIGHTING, value != 0);
}
/// `imodvMenuLabels`.
pub fn imodv_menu_labels(window: &mut ImodvWindow, value: i32) {
    window.set_checkable_item(VVIEW_MENU_LABELS, value != 0);
}
/// `imodvMenuWireframe`.
pub fn imodv_menu_wireframe(window: &mut ImodvWindow, value: i32) {
    window.set_checkable_item(VVIEW_MENU_WIREFRAME, value != 0);
}
/// `imodvMenuLowres`.
pub fn imodv_menu_lowres(window: &mut ImodvWindow, value: i32) {
    window.set_checkable_item(VVIEW_MENU_LOWRES, value != 0);
}
/// `imodvMenuInvertZ`.
pub fn imodv_menu_invert_z(window: &mut ImodvWindow, value: i32) {
    window.set_checkable_item(VVIEW_MENU_INVERTZ, value != 0);
}

/// `imodvAddBoundingBox`.
pub fn imodv_add_bounding_box(a: &mut ImodvApp, obj_num: i32) -> i32 {
    let Some(model) = (unsafe { a.imod.as_ref() }) else {
        return 1;
    };
    if obj_num >= model.obj.len() as i32 {
        return 1;
    }
    let (min, max, extra) = if obj_num < 0 {
        (
            Ipoint {
                x: -1.,
                y: -1.,
                z: -1.,
            },
            Ipoint {
                x: model.xmax as f32,
                y: model.ymax as f32,
                z: model.zmax as f32,
            },
            a.bound_box_extra_obj,
        )
    } else {
        let mut min = Ipoint {
            x: -1.,
            y: -1.,
            z: -1.,
        };
        let mut max = Ipoint::default();
        imod_object_get_bbox(&model.obj[obj_num as usize], &mut min, &mut max);
        (min, max, a.obj_bound_extra_obj)
    };
    if extra <= 0 || a.vi.is_null() {
        return 1;
    }
    let vi = unsafe { &mut *(a.vi as *mut ImodView) };
    let Some(object) = ivw_get_an_extra_object(vi, extra) else {
        return 1;
    };
    if object.cont.len() < 6 {
        return 1;
    }
    for cont in &mut object.cont {
        imod_contour_clear_points(cont);
    }
    for i in 0..2 {
        let z = min.z + i as f32 * (max.z - min.z);
        for &(x, y) in &[
            (min.x, min.y),
            (max.x, min.y),
            (max.x, max.y),
            (min.x, max.y),
            (min.x, min.y),
        ] {
            imod_point_append_xyz(&mut object.cont[i], x, y, z);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let x = min.x + i as f32 * (max.x - min.x);
            let y = min.y + j as f32 * (max.y - min.y);
            let ind = 2 + i + 2 * j;
            imod_point_append_xyz(&mut object.cont[ind], x, y, min.z);
            imod_point_append_xyz(&mut object.cont[ind], x, y, max.z);
        }
    }
    0
}

/// `imodvOpenSelectedWindows`; native form construction is returned in source order.
pub fn imodv_open_selected_windows(keys: Option<&str>, standalone: i32) -> Vec<&'static str> {
    let Some(keys) = keys else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (key, action) in [
        ('C', "imodv_control"),
        ('O', "objed"),
        ('B', "imodv_menu_bgcolor"),
        ('L', "imodv_object_list_dialog"),
        ('V', "imodv_view_edit_dialog"),
        ('M', "imodv_model_edit_dialog"),
        ('m', "mv_movie_dialog"),
        ('N', "mv_movie_sequence_dialog"),
        ('S', "imodv_stereo_edit_dialog"),
        ('D', "imodv_depth_cue_edit_dialog"),
        ('R', "open_rotation_tool"),
    ] {
        if keys.contains(key) {
            out.push(action);
        }
    }
    if standalone == 0 {
        if keys.contains('I') {
            out.push("mv_image_edit_dialog");
        }
        if keys.contains('U') {
            out.push("imodv_isosurface_edit_dialog");
        }
    } else if keys.contains('e') {
        out.push("scale_bar_open");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::Imod;
    #[test]
    fn bounding_box_has_source_six_contours() {
        let mut model = Imod::default();
        model.xmax = 10;
        model.ymax = 20;
        model.zmax = 30;
        let mut vi = ImodView::default();
        let extra = ivw_get_free_extra_object_number(&mut vi);
        let mut app = ImodvApp {
            imod: &mut model,
            vi: (&mut vi as *mut ImodView).cast(),
            bound_box_extra_obj: extra,
            ..Default::default()
        };
        {
            let object = ivw_get_an_extra_object(&mut vi, extra).unwrap();
            object.cont = imod_contours_new(6).unwrap();
        }
        assert_eq!(imodv_add_bounding_box(&mut app, -1), 0);
        assert_eq!(
            vi.extra_obj[extra as usize]
                .cont
                .iter()
                .map(|c| c.pts.len())
                .collect::<Vec<_>>(),
            vec![5, 5, 2, 2, 2, 2]
        );
    }
    #[test]
    fn selected_windows_match_source_conditions() {
        assert_eq!(
            imodv_open_selected_windows(Some("CIUe"), 0),
            vec![
                "imodv_control",
                "mv_image_edit_dialog",
                "imodv_isosurface_edit_dialog"
            ]
        );
        assert_eq!(
            imodv_open_selected_windows(Some("eI"), 1),
            vec!["scale_bar_open"]
        );
    }
}
