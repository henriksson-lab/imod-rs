//! Translation of `IMOD/3dmod/xzap.cpp` and `xzap.h`.
//!
//! ZaP is the orthogonal image window and the centre of the normal `3dmod`
//! display host.  Everything the source computes itself — coordinate and zoom
//! arithmetic, the rubber band, the lasso and arrows, contour shifting and
//! transforming, the model/ghost/extra-object draw order, the montage
//! snapshot layout — is translated here.  Everything it reaches out to that
//! is a Qt widget, an OpenGL context, a dialog, a preferences object or
//! another unit's owned state is one method on [`ZapNativeBoundary`], named
//! after the call the source makes; each default body reports the missing
//! unit once and does nothing else.
#![allow(dead_code, unused_variables, unused_assignments, unused_mut)]

use std::cell::{Cell, RefCell};
use std::ptr;

use crate::imod::libcfshr::ilist::{Ilist, ilist_item, ilist_remove, ilist_size};
use crate::imod::libimod::icont::{
    ICONT_CURSOR_LIKE, ICONT_DRAW_ALLZ, ICONT_MMODEL_ONLY, ICONT_STIPPLED, Nesting,
};
use crate::imod::libimod::icont::{
    imod_contour_area, imod_contour_center_of_mass, imod_contour_check_nesting,
    imod_contour_delete, imod_contour_dup, imod_contour_free_nests, imod_contour_free_z_tables,
    imod_contour_get_bbox, imod_contour_inside_cont, imod_contour_join, imod_contour_length,
    imod_contour_make_z_tables, imod_contour_nest_levels, imod_contour_new, imodel_contour_scan,
};
use crate::imod::libimod::imesh::imod_mesh_nearest_res;
use crate::imod::libimod::imodel::{
    ICONT_OPEN, ICONT_WILD, IMOD_OBJFLAG_SCAT, Icont, Iindex, Imod, Iobj, Ipoint, imod_contour_get,
    imod_get_index, imod_insert_point, imod_object_get, imod_point_get, imod_units,
};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_DRAW_LABEL, IMOD_OBJFLAG_EXTRA_EDIT, IMOD_OBJFLAG_MODV_ONLY,
    IMOD_OBJFLAG_PNT_ON_SEC, IMOD_OBJFLAG_POLY_CONT, IOBJ_EX_2D_TRANS, imod_object_add_contour,
    imod_object_set_color, iobj_close, iobj_off, iobj_open, iobj_planar, iobj_scat,
};
use crate::imod::libimod::ipoint::{
    imod_point_append, imod_point_cont_distance, imod_point_delete, imod_point_distance,
    imod_point_intersect, imodel_point_dist,
};
use crate::imod::libimod::istore::DrawProps;
use crate::imod::libimod::istore::{Istore, StoreUnion, istore_count_items, istore_insert};
use crate::imod::three_dmod::b3dgfx::b3d_set_image_offset;
use crate::imod::three_dmod::control::{
    GRAPH_WINDOW_TYPE, MULTIZ_WINDOW_TYPE, ZAP_WINDOW_TYPE, ivw_control_active,
    ivw_control_priority,
};
use crate::imod::three_dmod::imod::{
    APP, imod_debug, imod_initial_zoom, imod_print_stderr, imod_puts, imod_trace, wprint,
};
use crate::imod::three_dmod::imod_edit::{X_SLICE_BOX, Y_SLICE_BOX, Z_SLICE_BOX};
use crate::imod::three_dmod::imodview::{
    ImodView, ivw_clear_an_extra_object, ivw_free_extra_object, ivw_get_an_extra_object,
    ivw_get_free_extra_object_number, ivw_get_image_padding, ivw_get_max_time,
    ivw_get_or_make_contour, ivw_get_z_section_time, ivw_make_line_pointers,
    ivw_register_insert_point, ivw_time_mismatch, ivw_window_time,
};
use crate::imod::three_dmod::mv_window::KeyEvent;
use crate::imod::three_dmod::undoredo::{
    undo_contour_data_chg, undo_contour_data_chg_cc, undo_finish_unit, undo_flush_unit,
    undo_point_addition_cc2, undo_point_removal, undo_point_shift_cp,
};
use crate::imod::three_dmod::utilities::{
    UtilitiesBoundary, imod_caption, util_analyze_band_edge, util_clear_window, util_close_key,
    util_current_point_size, util_disable_stipple, util_draw_symbol, util_enable_stipple,
    util_get_longest_time_string, util_is_band_committed, util_next_sec_with_cont,
    util_set_zoom_on_screen_change, util_test_band_move, util_unit_zoom_for_device_scaling,
};
use crate::imod::three_dmod::zap_classes::{
    MULTIZ_MAX_PANELS, ZAP_TOGGLE_ARROW, ZAP_TOGGLE_CENTER, ZAP_TOGGLE_INSERT, ZAP_TOGGLE_LASSO,
    ZAP_TOGGLE_RESOL, ZAP_TOGGLE_RUBBER, ZAP_TOGGLE_TIMELOCK, ZAP_TOGGLE_ZLOCK,
};

/// `IMOD_MMOVIE` / `IMOD_MMODEL` (`imodel.h`).
pub const IMOD_MMOVIE: i32 = 0;
/// See [`IMOD_MMOVIE`].
pub const IMOD_MMODEL: i32 = 1;
/// `IMOD_DRAW_*` (`imod.h:36-72`).
pub const IMOD_DRAW_IMAGE: i32 = 1;
/// See [`IMOD_DRAW_IMAGE`].
pub const IMOD_DRAW_XYZ: i32 = 1 << 1;
/// See [`IMOD_DRAW_IMAGE`].
pub const IMOD_DRAW_MOD: i32 = 1 << 2;
/// See [`IMOD_DRAW_IMAGE`].
pub const IMOD_DRAW_SLICE: i32 = 1 << 3;
/// See [`IMOD_DRAW_IMAGE`].
pub const IMOD_DRAW_COLORMAP: i32 = 1 << 11;
/// See [`IMOD_DRAW_IMAGE`].
pub const IMOD_DRAW_NOSYNC: i32 = 1 << 12;
/// See [`IMOD_DRAW_IMAGE`].
pub const IMOD_DRAW_RETHINK: i32 = 1 << 13;
/// See [`IMOD_DRAW_IMAGE`].
pub const IMOD_DRAW_ACTIVE: i32 = 1 << 14;
/// `IMOD_SELSIZE` (`imodP.h:285`).
pub const IMOD_SELSIZE: f32 = 15.;
/// `MOVIE_DEFAULT` (`imodP.h:360`).
pub const MOVIE_DEFAULT: i32 = 52965;
/// `AUTOX_ALTMOUSE_PAINT` (`autox.h:30`).
pub const AUTOX_ALTMOUSE_PAINT: i32 = 1;
/// `AUTOX_*` data flags (`autox.h`).
pub const AUTOX_FLOOD: u8 = 1;
/// See [`AUTOX_FLOOD`].
pub const AUTOX_WHITE: u8 = 2;
/// See [`AUTOX_FLOOD`].
pub const AUTOX_BLACK: u8 = 4;
/// `INCOS_NEW_CONT` (`imod_input.h:23`).
pub const INCOS_NEW_CONT: i32 = -2;
/// `HANDLE_LINE_COLOR` (`finegrain.h:12`).
pub const HANDLE_LINE_COLOR: i32 = 1;
/// `HANDLE_2DWIDTH` (`finegrain.h:16`).
pub const HANDLE_2DWIDTH: i32 = 1 << 4;
/// `HANDLE_VALUE1` (`finegrain.h:18`).
pub const HANDLE_VALUE1: i32 = 1 << 6;
/// `CHANGED_COLOR` (`istore.h:64`).
pub const CHANGED_COLOR: i32 = 1;
/// `GEN_STORE_MINMAX1` / `GEN_STORE_COLOR` / `GEN_STORE_CONNECT` / byte flag
/// (`istore.h`).
pub const GEN_STORE_MINMAX1: i16 = 3;
/// See [`GEN_STORE_MINMAX1`].
pub const GEN_STORE_COLOR: i16 = 1;
/// See [`GEN_STORE_MINMAX1`].
pub const GEN_STORE_CONNECT: i16 = 6;
/// See [`GEN_STORE_MINMAX1`].
pub const GEN_STORE_BYTE: u16 = 1;
/// `RADIANS_PER_DEGREE` (`b3dutil.h:68`).
pub const RADIANS_PER_DEGREE: f64 = 0.017_453_292_52;
/// `IMOD_GHOST_*` (`imod.h`).
pub const IMOD_GHOST_SECTION: i32 = 3;
/// See [`IMOD_GHOST_SECTION`].
pub const IMOD_GHOST_NEXTSEC: i32 = 1;
/// See [`IMOD_GHOST_SECTION`].
pub const IMOD_GHOST_PREVSEC: i32 = 2;
/// See [`IMOD_GHOST_SECTION`].
pub const IMOD_GHOST_SURFACE: i32 = 4;
/// See [`IMOD_GHOST_SECTION`].
pub const IMOD_GHOST_ALLOBJ: i32 = 8;
/// See [`IMOD_GHOST_SECTION`].
pub const IMOD_GHOST_LIGHTER: i32 = 16;
/// See [`IMOD_GHOST_SECTION`].
pub const IMOD_GHOST_ALLSCAT: i32 = 32;
/// See [`IMOD_GHOST_SECTION`].
pub const IMOD_GHOST_2SHADES: i32 = 128;
/// `IOBJ_SYM_*` (`iobj.h`).
pub const IOBJ_SYM_NONE: i32 = 0;
/// See [`IOBJ_SYM_NONE`].
pub const IOBJ_SYM_CIRCLE: i32 = 1;
/// See [`IOBJ_SYM_NONE`].
pub const IOBJ_SYM_TRIANGLE: i32 = 3;
/// `IOBJ_SYMF_ENDS` / `IOBJ_SYMF_ARROW` (`iobj.h`).
pub const IOBJ_SYMF_ENDS: u8 = 2;
/// See [`IOBJ_SYMF_ENDS`].
pub const IOBJ_SYMF_ARROW: u8 = 16;
/// `IOBJ_EX_PNT_LIMIT` / `IOBJ_EX_LABEL_SIZE` / `IOBJ_EX_FLAGS` /
/// `IOBJ_EX_LASSO_ID` (`iobj.h:119-123`); `IOBJ_EXSIZE` is 16.
pub const IOBJ_EX_PNT_LIMIT: usize = 0;
/// See [`IOBJ_EX_PNT_LIMIT`].
pub const IOBJ_EX_LABEL_SIZE: usize = 2;
/// See [`IOBJ_EX_PNT_LIMIT`].
pub const IOBJ_EX_FLAGS: usize = 3;
/// See [`IOBJ_EX_PNT_LIMIT`].
pub const IOBJ_EX_LASSO_ID: usize = 15;
/// `IOBJ_EXFLAG_SLICER_ONLY` / `IOBJ_EXFLAG_MESH_ON_IMG` (`iobj.h:127-128`).
pub const IOBJ_EXFLAG_SLICER_ONLY: u32 = 1 << 1;
/// See [`IOBJ_EXFLAG_SLICER_ONLY`].
pub const IOBJ_EXFLAG_MESH_ON_IMG: u32 = 1 << 2;
/// `IMOD_MESH_*` list codes (`imesh.h`).
pub const IMOD_MESH_BGNPOLY: i32 = 1;
/// See [`IMOD_MESH_BGNPOLY`].
pub const IMOD_MESH_ENDPOLY: i32 = 2;
/// See [`IMOD_MESH_BGNPOLY`].
pub const IMOD_MESH_BGNBIGPOLY: i32 = 13;
/// See [`IMOD_MESH_BGNPOLY`].
pub const IMOD_MESH_BGNPOLYNORM: i32 = 9;
/// See [`IMOD_MESH_BGNPOLY`].
pub const IMOD_MESH_BGNPOLYNORM2: i32 = 11;
/// `MRC_MODE_RGB` (`mrcfiles.h`).
pub const MRC_MODE_RGB: i32 = 16;

/// `BORDER_FRAC` (`xzap.cpp:1015`); `0.1` is a double literal in the source,
/// so `wsize * BORDER_FRAC` in `syncImage` is a double multiply.
pub const BORDER_FRAC: f64 = 0.1;
/// `BORDER_MIN` (`xzap.cpp:1016`).
pub const BORDER_MIN: i32 = 50;
/// `BORDER_MIN_MULTIZ` (`xzap.cpp:1017`).
pub const BORDER_MIN_MULTIZ: i32 = 20;
/// `BORDER_MAX` (`xzap.cpp:1018`).
pub const BORDER_MAX: i32 = 125;

/// Qt key codes reached by `keyInput`, `keyRelease` and `zapKey_cb`.
pub const KEY_ESCAPE: i32 = 0x0100_0000;
/// See [`KEY_ESCAPE`].
pub const KEY_HOME: i32 = 0x0100_0010;
/// See [`KEY_ESCAPE`].
pub const KEY_END: i32 = 0x0100_0011;
/// See [`KEY_ESCAPE`].
pub const KEY_LEFT: i32 = 0x0100_0012;
/// See [`KEY_ESCAPE`].
pub const KEY_UP: i32 = 0x0100_0013;
/// See [`KEY_ESCAPE`].
pub const KEY_RIGHT: i32 = 0x0100_0014;
/// See [`KEY_ESCAPE`].
pub const KEY_DOWN: i32 = 0x0100_0015;
/// See [`KEY_ESCAPE`].
pub const KEY_PAGE_UP: i32 = 0x0100_0016;
/// See [`KEY_ESCAPE`].
pub const KEY_PAGE_DOWN: i32 = 0x0100_0017;
/// See [`KEY_ESCAPE`].
pub const KEY_INSERT: i32 = 0x0100_0006;
/// See [`KEY_ESCAPE`].
pub const KEY_F1: i32 = 0x0100_0030;
/// See [`KEY_ESCAPE`].
pub const KEY_F8: i32 = 0x0100_0037;
/// See [`KEY_ESCAPE`].
pub const KEY_F11: i32 = 0x0100_003a;
/// See [`KEY_ESCAPE`].
pub const KEY_F12: i32 = 0x0100_003b;
/// See [`KEY_ESCAPE`].
pub const KEY_EXCLAM: i32 = 0x21;
/// See [`KEY_ESCAPE`].
pub const KEY_ASTERISK: i32 = 0x2a;
/// See [`KEY_ESCAPE`].
pub const KEY_PLUS: i32 = 0x2b;
/// See [`KEY_ESCAPE`].
pub const KEY_MINUS: i32 = 0x2d;
/// See [`KEY_ESCAPE`].
pub const KEY_SLASH: i32 = 0x2f;
/// See [`KEY_ESCAPE`].
pub const KEY_0: i32 = 0x30;
/// See [`KEY_ESCAPE`].
pub const KEY_1: i32 = 0x31;
/// See [`KEY_ESCAPE`].
pub const KEY_2: i32 = 0x32;
/// See [`KEY_ESCAPE`].
pub const KEY_EQUAL: i32 = 0x3d;
/// See [`KEY_ESCAPE`].
pub const KEY_AT: i32 = 0x40;
/// See [`KEY_ESCAPE`].
pub const KEY_A: i32 = 0x41;
/// See [`KEY_ESCAPE`].
pub const KEY_B: i32 = 0x42;
/// See [`KEY_ESCAPE`].
pub const KEY_F: i32 = 0x46;
/// See [`KEY_ESCAPE`].
pub const KEY_I: i32 = 0x49;
/// See [`KEY_ESCAPE`].
pub const KEY_K: i32 = 0x4b;
/// See [`KEY_ESCAPE`].
pub const KEY_P: i32 = 0x50;
/// See [`KEY_ESCAPE`].
pub const KEY_Q: i32 = 0x51;
/// See [`KEY_ESCAPE`].
pub const KEY_R: i32 = 0x52;
/// See [`KEY_ESCAPE`].
pub const KEY_S: i32 = 0x53;
/// See [`KEY_ESCAPE`].
pub const KEY_U: i32 = 0x55;
/// See [`KEY_ESCAPE`].
pub const KEY_Z: i32 = 0x5a;
/// `Qt::KeypadModifier`.
pub const KEYPAD_MODIFIER: i32 = 0x2000_0000;
/// `Qt::ShiftModifier`.
pub const SHIFT_MODIFIER: i32 = 0x0200_0000;
/// `Qt::ControlModifier`.
pub const CONTROL_MODIFIER: i32 = 0x0400_0000;
/// `QEvent::Enter`, `QEvent::Leave` and `QEvent::Wheel` type codes, the three
/// `generalEvent` tests.
pub const EVENT_ENTER: i32 = 10;
/// See [`EVENT_ENTER`].
pub const EVENT_LEAVE: i32 = 11;
/// See [`EVENT_ENTER`].
pub const EVENT_WHEEL: i32 = 31;

/// A `QEvent` as `generalEvent` reads it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZapGeneralEvent {
    /// `e->type()`.
    pub event_type: i32,
    /// A `QWheelEvent`'s `angleDelta().y()`, read only for `QEvent::Wheel`.
    pub wheel_delta: i32,
}

/// The reporting shape of a default boundary body: the missing unit is named
/// once per process, never silently skipped.  Same shape as `imodview.rs`.
fn report_once(call: &'static str, needs: &'static str) {
    use std::collections::HashSet;
    use std::sync::Mutex as StdMutex;
    static REPORTED: StdMutex<Option<HashSet<&'static str>>> = StdMutex::new(None);
    let mut guard = REPORTED.lock().unwrap();
    let set = guard.get_or_insert_with(HashSet::new);
    if set.insert(call) {
        eprintln!("3dmod: {call} needs {needs}, which has no native host yet");
    }
}

/// Everything `xzap.cpp` reaches that is a Qt widget, an OpenGL context, a
/// preferences object, the dialog manager, or another unit's owned state.
///
/// One method per call the source makes, named after it.  The default body
/// reports the unit that has no native host yet; a host overrides the ones it
/// can really perform.  `utilities.cpp`'s drawing primitives are reached
/// through the [`UtilitiesBoundary`] supertrait so that the translated
/// `utilDrawSymbol`, `utilEnableStipple`, `utilDisableStipple`,
/// `utilClearWindow` and `utilWheelChangePointSize` run for real.
pub trait ZapNativeBoundary: UtilitiesBoundary {
    // ---- App members that `ImodApp` (imodP.h:39) does not carry ------------
    /// `App->doublebuffer`.
    fn app_doublebuffer(&mut self) -> bool {
        report_once("App->doublebuffer", "the ImodApp members of imodP.h:39");
        false
    }
    /// `App->depth`.
    fn app_depth(&mut self) -> i32 {
        report_once("App->depth", "the ImodApp members of imodP.h:39");
        8
    }
    /// `App->qtEnableDepth`.
    fn app_qt_enable_depth(&mut self) -> bool {
        report_once("App->qtEnableDepth", "the ImodApp members of imodP.h:39");
        false
    }
    /// `App->devPixVaries`.
    fn app_dev_pix_varies(&mut self) -> bool {
        report_once("App->devPixVaries", "the ImodApp members of imodP.h:39");
        false
    }
    /// `App->minDevPixRatio`.
    fn app_min_dev_pix_ratio(&mut self) -> f32 {
        report_once("App->minDevPixRatio", "the ImodApp members of imodP.h:39");
        1.
    }
    /// `App->minDPRleftPos`.
    fn app_min_dpr_left_pos(&mut self) -> i32 {
        report_once("App->minDPRleftPos", "the ImodApp members of imodP.h:39");
        0
    }
    /// `App->minDPRtopPos`.
    fn app_min_dpr_top_pos(&mut self) -> i32 {
        report_once("App->minDPRtopPos", "the ImodApp members of imodP.h:39");
        0
    }
    /// `App->cvi`, read by `fillOverlayRGB` for `ushortStore`.
    fn app_cvi(&mut self) -> *mut ImodView {
        report_once("App->cvi", "the ImodApp members of imodP.h:39");
        ptr::null_mut()
    }

    // ---- ImodPrefs (preferences.cpp) --------------------------------------
    /// `if (!ImodPrefs)` in `zapReportBiggestMultiZ`.
    fn imod_prefs_exists(&mut self) -> bool {
        report_once("ImodPrefs", "the settings object of preferences.cpp");
        false
    }
    /// `ImodPrefs->startInHQ()`.
    fn prefs_start_in_hq(&mut self) -> bool {
        report_once(
            "ImodPrefs->startInHQ",
            "the settings object of preferences.cpp",
        );
        false
    }
    /// `ImodPrefs->arrowsScrollZap()`.
    fn prefs_arrows_scroll_zap(&mut self) -> bool {
        report_once(
            "ImodPrefs->arrowsScrollZap",
            "the settings object of preferences.cpp",
        );
        false
    }
    /// `ImodPrefs->minCurrentModPtSize()`, read by `utilCurrentPointSize`.
    fn prefs_min_current_mod_pt_size(&mut self) -> i32 {
        report_once(
            "ImodPrefs->minCurrentModPtSize",
            "the settings object of preferences.cpp",
        );
        4
    }
    /// `ImodPrefs->minCurrentImPtSize()`, read by `utilCurrentPointSize`.
    fn prefs_min_current_im_pt_size(&mut self) -> i32 {
        report_once(
            "ImodPrefs->minCurrentImPtSize",
            "the settings object of preferences.cpp",
        );
        4
    }
    /// `ImodPrefs->attachToOnObj()`, read by `imodAllObjNearest`.
    fn prefs_attach_to_on_obj(&mut self) -> bool {
        report_once(
            "ImodPrefs->attachToOnObj",
            "the settings object of preferences.cpp",
        );
        false
    }
    /// `ImodPrefs->actualButton(logical)`; the returned value is compared with
    /// `event->button()` and masked against `event->buttons()`, so it is a Qt
    /// button bit.  The default is the unswapped mapping
    /// (`Qt::LeftButton`, `Qt::MidButton`, `Qt::RightButton`).
    fn prefs_actual_button(&mut self, logical: i32) -> i32 {
        match logical {
            1 => 1,
            2 => 4,
            _ => 2,
        }
    }
    /// `ImodPrefs->getZapGeometry()`; `[x, y, width, height]` as `QRect`.
    fn prefs_get_zap_geometry(&mut self) -> [i32; 4] {
        report_once(
            "ImodPrefs->getZapGeometry",
            "the settings object of preferences.cpp",
        );
        [0; 4]
    }
    /// `ImodPrefs->getMultiZparams(numX, numY, zStep, drawCen, drawOther)`;
    /// returns the geometry and writes the five parameters back.
    fn prefs_get_multi_z_params(
        &mut self,
        num_x: &mut i32,
        num_y: &mut i32,
        z_step: &mut i32,
        draw_center: &mut i32,
        draw_others: &mut i32,
    ) -> [i32; 4] {
        report_once(
            "ImodPrefs->getMultiZparams",
            "the settings object of preferences.cpp",
        );
        [0; 4]
    }
    /// `ImodPrefs->recordMultiZparams(pos, numX, numY, zStep, cen, other)`.
    fn prefs_record_multi_z_params(
        &mut self,
        pos: [i32; 4],
        num_x: i32,
        num_y: i32,
        z_step: i32,
        draw_center: i32,
        draw_others: i32,
    ) {
        report_once(
            "ImodPrefs->recordMultiZparams",
            "the settings object of preferences.cpp",
        );
    }

    // ---- imodDialogManager (control.cpp) ----------------------------------
    /// `imodDialogManager.windowList(&objList, -1, type)`, already reduced to
    /// the `((ZapWindow *)objList.at(i))->mZap` the source takes from it.
    fn imod_dialog_manager_window_list(&mut self, window_type: i32) -> Vec<*mut ZapFuncs> {
        report_once(
            "imodDialogManager.windowList",
            "the dialog manager of control.cpp",
        );
        Vec::new()
    }
    /// `imodDialogManager.getTopWindow(withBand, withLasso, type, index)`.
    fn imod_dialog_manager_get_top_window(
        &mut self,
        with_band: bool,
        with_lasso: bool,
        window_type: i32,
        index: Option<&mut i32>,
    ) -> *mut ZapFuncs {
        report_once(
            "imodDialogManager.getTopWindow",
            "the dialog manager of control.cpp",
        );
        ptr::null_mut()
    }
    /// `imodDialogManager.windowCount(type)`.
    fn imod_dialog_manager_window_count(&mut self, window_type: i32) -> i32 {
        report_once(
            "imodDialogManager.windowCount",
            "the dialog manager of control.cpp",
        );
        0
    }
    /// `imodDialogManager.add((QWidget *)mQtWindow, IMOD_IMAGE, type, ctrl)`.
    fn imod_dialog_manager_add(&mut self, zap: *mut ZapFuncs, window_type: i32, ctrl: i32) {
        report_once("imodDialogManager.add", "the dialog manager of control.cpp");
    }
    /// `imodDialogManager.remove((QWidget *)mQtWindow)`.
    fn imod_dialog_manager_remove(&mut self, zap: *mut ZapFuncs) {
        report_once(
            "imodDialogManager.remove",
            "the dialog manager of control.cpp",
        );
    }
    /// The `GRAPH_WINDOW_TYPE` loop at the end of `paint`:
    /// `((GraphWindow *)objList.at(ob))->draw()`.
    fn graph_window_list_draw(&mut self) {
        report_once("GraphWindow::draw", "the graph windows of xgraph.cpp");
    }

    // ---- ZapWindow / ZapGL (zap_classes.cpp) ------------------------------
    /// `new ZapWindow(this, str, wintype != 0, ...)`; a null return is the
    /// source's `if (!mQtWindow)` failure.
    fn new_zap_window(
        &mut self,
        zap: *mut ZapFuncs,
        time_label: &str,
        wintype: bool,
    ) -> *mut crate::imod::three_dmod::zap_classes::ZapWindow {
        report_once("new ZapWindow", "the Zap window of zap_classes.cpp");
        ptr::null_mut()
    }
    /// `mQtWindow->close()`.
    fn zap_window_close(&mut self) {
        report_once("ZapWindow::close", "the Zap window of zap_classes.cpp");
    }
    /// `mQtWindow->setToggleState(index, state)`.
    fn zap_window_set_toggle_state(&mut self, index: usize, state: i32) {
        report_once(
            "ZapWindow::setToggleState",
            "the Zap toolbar of zap_classes.cpp",
        );
    }
    /// `mQtWindow->setLowHighSectionState(state)`.
    fn zap_window_set_low_high_section_state(&mut self, state: i32) {
        report_once(
            "ZapWindow::setLowHighSectionState",
            "the Zap toolbar of zap_classes.cpp",
        );
    }
    /// `mQtWindow->setZoomText(zoom)`.
    fn zap_window_set_zoom_text(&mut self, zoom: f32) {
        report_once(
            "ZapWindow::setZoomText",
            "the Zap toolbar of zap_classes.cpp",
        );
    }
    /// `mQtWindow->setSectionText(section)`.
    fn zap_window_set_section_text(&mut self, section: i32) {
        report_once(
            "ZapWindow::setSectionText",
            "the Zap toolbar of zap_classes.cpp",
        );
    }
    /// `mQtWindow->setSizeText(winx, winy)`.
    fn zap_window_set_size_text(&mut self, winx: i32, winy: i32) {
        report_once(
            "ZapWindow::setSizeText",
            "the Zap toolbar of zap_classes.cpp",
        );
    }
    /// `mQtWindow->setMaxZ(maxZ)`.
    fn zap_window_set_max_z(&mut self, max_z: i32) {
        report_once("ZapWindow::setMaxZ", "the Zap toolbar of zap_classes.cpp");
    }
    /// `mQtWindow->setTimeLabel(time, label)`.
    fn zap_window_set_time_label(&mut self, time: i32, label: &str) {
        report_once(
            "ZapWindow::setTimeLabel",
            "the Zap time toolbar of zap_classes.cpp",
        );
    }
    /// `mQtWindow->lowSection()`.
    fn zap_window_low_section(&mut self) -> String {
        report_once(
            "ZapWindow::lowSection",
            "the Zap toolbar of zap_classes.cpp",
        );
        String::new()
    }
    /// `mQtWindow->highSection()`.
    fn zap_window_high_section(&mut self) -> String {
        report_once(
            "ZapWindow::highSection",
            "the Zap toolbar of zap_classes.cpp",
        );
        String::new()
    }
    /// `mQtWindow->setFocus()`.
    fn zap_window_set_focus(&mut self) {
        report_once("ZapWindow::setFocus", "the Zap window of zap_classes.cpp");
    }
    /// `mQtWindow->setWindowTitle(...)` and the three toolbar titles.
    fn zap_window_set_window_title(&mut self, which: i32, title: &str) {
        report_once(
            "ZapWindow::setWindowTitle",
            "the Zap window of zap_classes.cpp",
        );
    }
    /// `mQtWindow->insertToolBarBreak(bar)`; 2 is `mToolBar2`, 3 is
    /// `mPanelBar`.
    fn zap_window_insert_toolbar_break(&mut self, which: i32) {
        report_once(
            "ZapWindow::insertToolBarBreak",
            "the Zap toolbars of zap_classes.cpp",
        );
    }
    /// `mQtWindow->mToolBar->sizeHint()` and friends, as `(width, height)`;
    /// 1 is `mToolBar`, 2 is `mToolBar2`, 3 is `mPanelBar`.
    fn zap_window_toolbar_size_hint(&mut self, which: i32) -> (i32, i32) {
        report_once(
            "ZapWindow::mToolBar->sizeHint",
            "the Zap toolbars of zap_classes.cpp",
        );
        (0, 0)
    }
    /// `mQtWindow->mToolBar2` / `mQtWindow->mPanelBar` non-null tests; 2 is
    /// `mToolBar2`, 3 is `mPanelBar`.
    fn zap_window_toolbar_exists(&mut self, which: i32) -> bool {
        report_once(
            "ZapWindow::mToolBar2",
            "the Zap toolbars of zap_classes.cpp",
        );
        false
    }
    /// `mQtWindow->mToolBar->height()` / `mPanelBar->height()`.
    fn zap_window_toolbar_height(&mut self, which: i32) -> i32 {
        report_once(
            "ZapWindow::mToolBar->height",
            "the Zap toolbars of zap_classes.cpp",
        );
        0
    }
    /// `diaSetSpinBox(mQtWindow->mColumnSpin/mRowSpin, n)`.
    fn zap_window_set_spin_boxes(&mut self, columns: i32, rows: i32) {
        report_once(
            "diaSetSpinBox on ZapWindow::mColumnSpin",
            "the Multi-Z toolbar of zap_classes.cpp",
        );
    }
    /// `mQtWindow->mZoomEdit->fontMetrics().height()`.
    fn zap_window_zoom_edit_font_height(&mut self) -> i32 {
        report_once(
            "ZapWindow::mZoomEdit->fontMetrics",
            "the Zap toolbar of zap_classes.cpp",
        );
        0
    }
    /// `mQtWindow->grabKeyboard()`.
    fn zap_window_grab_keyboard(&mut self) {
        report_once(
            "ZapWindow::grabKeyboard",
            "the Zap window of zap_classes.cpp",
        );
    }
    /// `mQtWindow->releaseKeyboard()`.
    fn zap_window_release_keyboard(&mut self) {
        report_once(
            "ZapWindow::releaseKeyboard",
            "the Zap window of zap_classes.cpp",
        );
    }
    /// `mQtWindow->resize(w, h)`.
    fn zap_window_resize(&mut self, width: i32, height: i32) {
        report_once("ZapWindow::resize", "the Zap window of zap_classes.cpp");
    }
    /// `mQtWindow->move(x, y)`.
    fn zap_window_move(&mut self, x: i32, y: i32) {
        report_once("ZapWindow::move", "the Zap window of zap_classes.cpp");
    }
    /// `mQtWindow->show()`.
    fn zap_window_show(&mut self) {
        report_once("ZapWindow::show", "the Zap window of zap_classes.cpp");
    }
    /// `mQtWindow->width()` and `height()`.
    fn zap_window_size(&mut self) -> (i32, i32) {
        report_once("ZapWindow::width", "the Zap window of zap_classes.cpp");
        (0, 0)
    }
    /// `mQtWindow->pos()`.
    fn zap_window_pos(&mut self) -> (i32, i32) {
        report_once("ZapWindow::pos", "the Zap window of zap_classes.cpp");
        (0, 0)
    }
    /// `mQtWindow->frameGeometry()` as `[x, y, width, height]`.
    fn zap_window_frame_geometry(&mut self) -> [i32; 4] {
        report_once(
            "ZapWindow::frameGeometry",
            "the Zap window of zap_classes.cpp",
        );
        [0; 4]
    }
    /// `ivwRestorableGeometry(mQtWindow)` as `[x, y, width, height]`.
    fn ivw_restorable_geometry(&mut self) -> [i32; 4] {
        report_once(
            "ivwRestorableGeometry",
            "the window geometry of control.cpp",
        );
        [0; 4]
    }
    /// `mGfx->updateGL()`.
    fn gfx_update_gl(&mut self) {
        report_once("ZapGL::updateGL", "the Zap GL widget of zap_classes.cpp");
    }
    /// `mGfx->swapBuffers()`.
    fn gfx_swap_buffers(&mut self) {
        report_once("ZapGL::swapBuffers", "the Zap GL widget of zap_classes.cpp");
    }
    /// `mGfx->setBufferSwapAuto(state)`.
    fn gfx_set_buffer_swap_auto(&mut self, state: bool) {
        report_once(
            "ZapGL::setBufferSwapAuto",
            "the Zap GL widget of zap_classes.cpp",
        );
    }
    /// `mGfx->grabMouse()`.
    fn gfx_grab_mouse(&mut self) {
        report_once("ZapGL::grabMouse", "the Zap GL widget of zap_classes.cpp");
    }
    /// `mGfx->releaseMouse()`.
    fn gfx_release_mouse(&mut self) {
        report_once(
            "ZapGL::releaseMouse",
            "the Zap GL widget of zap_classes.cpp",
        );
    }
    /// `mGfx->setMouseTracking(state)`.
    fn gfx_set_mouse_tracking(&mut self, state: bool) {
        report_once(
            "ZapGL::setMouseTracking",
            "the Zap GL widget of zap_classes.cpp",
        );
    }
    /// `mGfx->extraCursorInWindow()`.
    fn gfx_extra_cursor_in_window(&mut self) -> bool {
        report_once(
            "ZapGL::extraCursorInWindow",
            "the Zap GL widget of zap_classes.cpp",
        );
        false
    }
    /// `mGfx->scheduleRedraw(msec)`.
    fn gfx_schedule_redraw(&mut self, msec: i32) {
        report_once(
            "ZapGL::scheduleRedraw",
            "the Zap GL widget of zap_classes.cpp",
        );
    }
    /// `mGfx->cancelRedraw()`.
    fn gfx_cancel_redraw(&mut self) {
        report_once(
            "ZapGL::cancelRedraw",
            "the Zap GL widget of zap_classes.cpp",
        );
    }
    /// `mGfx->scheduleResize(msec)`.
    fn gfx_schedule_resize(&mut self, msec: i32) {
        report_once(
            "ZapGL::scheduleResize",
            "the Zap GL widget of zap_classes.cpp",
        );
    }
    /// `mGfx->getLastDrawMsec()`.
    fn gfx_get_last_draw_msec(&mut self) -> i32 {
        report_once(
            "ZapGL::getLastDrawMsec",
            "the Zap GL widget of zap_classes.cpp",
        );
        0
    }
    /// `mGfx->setMinimumSize(w, h)`.
    fn gfx_set_minimum_size(&mut self, width: i32, height: i32) {
        report_once(
            "ZapGL::setMinimumSize",
            "the Zap GL widget of zap_classes.cpp",
        );
    }
    /// `mGfx->mFirstDraw`.
    fn gfx_first_draw(&mut self) -> i32 {
        report_once("ZapGL::mFirstDraw", "the Zap GL widget of zap_classes.cpp");
        0
    }
    /// `mGfx->mInitWidth/mInitHeight/mInitLeft/mInitTop`.
    fn gfx_set_init_geometry(&mut self, width: i32, height: i32, left: i32, top: i32) {
        report_once("ZapGL::mInitWidth", "the Zap GL widget of zap_classes.cpp");
    }
    /// `mGfx->width()` and `height()`.
    fn gfx_size(&mut self) -> (i32, i32) {
        report_once("ZapGL::width", "the Zap GL widget of zap_classes.cpp");
        (0, 0)
    }
    /// `mGfx->setColormap(*(App->qColormap))`.
    fn gfx_set_colormap(&mut self) {
        report_once("ZapGL::setColormap", "the Qt colormap of imod.cpp");
    }
    /// `mGfx->setUpdateBehavior(PartialUpdate | NoPartialUpdate)`.
    fn gfx_set_update_behavior(&mut self, partial: bool) {
        report_once(
            "ZapGL::setUpdateBehavior",
            "the Zap GL widget of zap_classes.cpp",
        );
    }
    /// `mGfx->renderText(x, y, text, font)`.
    fn gfx_render_text(&mut self, x: i32, y: i32, text: &str) {
        report_once("ZapGL::renderText", "the Qt text rasteriser of QPainter");
    }
    /// `mGfx->mapFromGlobal(QCursor::pos())`, in widget pixels.
    fn gfx_map_from_global_cursor_pos(&mut self) -> (i32, i32) {
        report_once("QCursor::pos", "the Qt cursor of the Zap GL widget");
        (0, 0)
    }
    /// `QCursor::setPos(mGfx->mapToGlobal(QPoint(x, y)))`.
    fn gfx_set_cursor_pos(&mut self, x: i32, y: i32) {
        report_once("QCursor::setPos", "the Qt cursor of the Zap GL widget");
    }

    // ---- b3dgfx.cpp (its B3dGfxState and GL context are the host's) -------
    /// `b3dSetCurSize(winx, winy)`.
    fn b3d_set_cur_size(&mut self, winx: i32, winy: i32) {
        report_once("b3dSetCurSize", "the B3dGfxState of b3dgfx.cpp");
    }
    /// `b3dSetCurDevPixRatio(dpr)`.
    fn b3d_set_cur_dev_pix_ratio(&mut self, dpr: f32) {
        report_once("b3dSetCurDevPixRatio", "the B3dGfxState of b3dgfx.cpp");
    }
    /// `b3dGetCurXZoom()`.
    fn b3d_get_cur_x_zoom(&mut self) -> f32 {
        report_once("b3dGetCurXZoom", "the B3dGfxState of b3dgfx.cpp");
        0.
    }
    /// `b3dStepPixelZoom(zoom, step)`.
    fn b3d_step_pixel_zoom(&mut self, zoom: f64, step: i32) -> f64 {
        report_once("b3dStepPixelZoom", "the zoom list of preferences.cpp");
        zoom
    }
    /// `b3dZoomDownCrit()`.
    fn b3d_zoom_down_crit(&mut self) -> f32 {
        report_once("b3dZoomDownCrit", "the B3dGfxState of b3dgfx.cpp");
        0.
    }
    /// `b3dResizeViewportXY(winx, winy)`.
    fn b3d_resize_viewport_xy(&mut self, winx: i32, winy: i32) {
        report_once("b3dResizeViewportXY", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dSubareaViewport(x, y, width, height)`.
    fn b3d_subarea_viewport(&mut self, x: i32, y: i32, width: i32, height: i32) {
        report_once("b3dSubareaViewport", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dColorIndex(pix)`.
    fn b3d_color_index(&mut self, pix: i32) {
        report_once("b3dColorIndex", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dLineWidth(width)` and `b3dLineWidth(width, obj)`.
    fn b3d_line_width(&mut self, width: i32, obj: *mut Iobj) {
        report_once("b3dLineWidth", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dStippleNextLine(value)`.
    fn b3d_stipple_next_line(&mut self, value: bool) {
        report_once("b3dStippleNextLine", "the B3dGfxState of b3dgfx.cpp");
    }
    /// `b3dDrawLine(x1, y1, x2, y2)`.
    fn b3d_draw_line(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
        report_once("b3dDrawLine", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dDrawRectangle(x, y, width, height)`.
    fn b3d_draw_rectangle(&mut self, x: i32, y: i32, width: i32, height: i32) {
        report_once("b3dDrawRectangle", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dDrawFilledRectangle(x, y, width, height)`.
    fn b3d_draw_filled_rectangle(&mut self, x: i32, y: i32, width: i32, height: i32) {
        report_once("b3dDrawFilledRectangle", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dDrawCircle(x, y, radius)` and its `scaleForDev` overload.
    fn b3d_draw_circle(&mut self, x: i32, y: i32, radius: i32, scale_for_dev: bool) {
        report_once("b3dDrawCircle", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dDrawPlus(x, y, size)`.
    fn b3d_draw_plus(&mut self, x: i32, y: i32, size: i32) {
        report_once("b3dDrawPlus", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dDrawCross(x, y, size)`.
    fn b3d_draw_cross(&mut self, x: i32, y: i32, size: i32) {
        report_once("b3dDrawCross", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dDrawSquare(x, y, size)`.
    fn b3d_draw_square(&mut self, x: i32, y: i32, size: i32) {
        report_once("b3dDrawSquare", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dDrawTriangle(x, y, size)`.
    fn b3d_draw_triangle(&mut self, x: i32, y: i32, size: i32) {
        report_once("b3dDrawTriangle", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dDrawArrow(tailX, tailY, headX, headY)` and its five-argument form.
    fn b3d_draw_arrow(
        &mut self,
        tail_x: i32,
        tail_y: i32,
        head_x: i32,
        head_y: i32,
        tip_length: i32,
        thickness: i32,
        anti_alias: bool,
    ) {
        report_once("b3dDrawArrow", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dBeginLine()`.
    fn b3d_begin_line(&mut self) {
        report_once("b3dBeginLine", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dEndLine()`.
    fn b3d_end_line(&mut self) {
        report_once("b3dEndLine", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dVertex2i(x, y)`.
    fn b3d_vertex_2i(&mut self, x: i32, y: i32) {
        report_once("b3dVertex2i", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dDrawBoxout(llx, lly, urx, ury)`.
    fn b3d_draw_boxout(&mut self, llx: i32, lly: i32, urx: i32, ury: i32) {
        report_once("b3dDrawBoxout", "the OpenGL context of b3dgfx.cpp");
    }
    /// `b3dFlushImage(image)`; the slot is `-1` for `mImage` and the panel
    /// index for a member of `mImages`.
    fn b3d_flush_image(&mut self, slot: i32) {
        report_once("b3dFlushImage", "the image cache of b3dgfx.cpp");
    }
    /// `b3dFreeCIImage(image)`.
    fn b3d_free_ci_image(&mut self, slot: i32) {
        report_once("b3dFreeCIImage", "the image cache of b3dgfx.cpp");
    }
    /// `b3dGetNewCIImage(image, App->depth)`; a `false` return is the
    /// source's null, which makes the caller print its own memory message.
    fn b3d_get_new_ci_image(&mut self, slot: i32) -> bool {
        report_once("b3dGetNewCIImage", "the image cache of b3dgfx.cpp");
        false
    }
    /// `b3dBufferImage(newim)`.
    fn b3d_buffer_image(&mut self, slot: i32) {
        report_once("b3dBufferImage", "the image cache of b3dgfx.cpp");
    }
    /// `b3dDrawGreyScalePixelsHQ(...)`.
    #[allow(clippy::too_many_arguments)]
    fn b3d_draw_grey_scale_pixels_hq(
        &mut self,
        image_data: *mut *mut u8,
        xsize: i32,
        ysize: i32,
        xoffset: i32,
        yoffset: i32,
        wx: i32,
        wy: i32,
        width: i32,
        height: i32,
        slot: i32,
        base: i32,
        xzoom: f64,
        yzoom: f64,
        quality: i32,
        slice: i32,
        rgba: i32,
        time_ramp: i32,
    ) {
        report_once(
            "b3dDrawGreyScalePixelsHQ",
            "the OpenGL context of b3dgfx.cpp",
        );
    }
    /// `b3dKeySnapshot(name, shifted, ctrl, limits)`.
    fn b3d_key_snapshot(&mut self, name: &str, shifted: i32, ctrl: i32, limits: Option<[i32; 4]>) {
        report_once("b3dKeySnapshot", "the snapshot writer of b3dgfx.cpp");
    }
    /// `b3dNamedSnapshot(fname, "zap", format, limits, checkConvert)`.
    fn b3d_named_snapshot(
        &mut self,
        fname: &mut String,
        format: i32,
        limits: Option<[i32; 4]>,
        check_convert: bool,
    ) -> i32 {
        report_once("b3dNamedSnapshot", "the snapshot writer of b3dgfx.cpp");
        1
    }
    /// `b3dSetMovieSnapping(state)`.
    fn b3d_set_movie_snapping(&mut self, state: bool) {
        report_once("b3dSetMovieSnapping", "the snapshot writer of b3dgfx.cpp");
    }
    /// `b3dMilliSleep(msec)`.
    fn b3d_milli_sleep(&mut self, msec: i32) {
        report_once("b3dMilliSleep", "the Qt event loop of b3dutil.cpp");
    }

    // ---- raw OpenGL calls made by xzap.cpp itself -------------------------
    /// `glReadBuffer(App->newQtOpenGL ? GL_FRONT : GL_BACK)`.
    fn gl_read_buffer(&mut self, front: bool) {
        report_once("glReadBuffer", "the OpenGL context of the Zap GL widget");
    }
    /// `glReadPixels(0, 0, winx, winy, GL_RGBA, GL_UNSIGNED_BYTE, framePix)`.
    fn gl_read_pixels(&mut self, winx: i32, winy: i32, frame_pix: *mut u8) {
        report_once("glReadPixels", "the OpenGL context of the Zap GL widget");
    }
    /// `glFlush()`.
    fn gl_flush(&mut self) {
        report_once("glFlush", "the OpenGL context of the Zap GL widget");
    }
    /// `glFinish()`.
    fn gl_finish(&mut self) {
        report_once("glFinish", "the OpenGL context of the Zap GL widget");
    }
    /// `glEnable(GL_BLEND)` / `glDisable(GL_BLEND)`.
    fn gl_enable_blend(&mut self, enable: bool) {
        report_once(
            "glEnable(GL_BLEND)",
            "the OpenGL context of the Zap GL widget",
        );
    }
    /// `glBlendFunc(src, dst)`, with the source's two pairs distinguished by
    /// `src_alpha`: `true` is `GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA`, `false`
    /// is `GL_ONE, GL_ZERO`.
    fn gl_blend_func(&mut self, src_alpha: bool) {
        report_once("glBlendFunc", "the OpenGL context of the Zap GL widget");
    }
    /// `glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT)`.
    fn gl_clear_color_buffer(&mut self) {
        report_once("glClear", "the OpenGL context of the Zap GL widget");
    }
    /// `glColor3f(r, g, b)`.
    fn gl_color_3f(&mut self, red: f32, green: f32, blue: f32) {
        report_once("glColor3f", "the OpenGL context of the Zap GL widget");
    }
    /// `glColor4f(r, g, b, a)`.
    fn gl_color_4f(&mut self, red: f32, green: f32, blue: f32, alpha: f32) {
        report_once("glColor4f", "the OpenGL context of the Zap GL widget");
    }
    /// `glBegin(GL_LINES)`.
    fn gl_begin_lines(&mut self) {
        report_once(
            "glBegin(GL_LINES)",
            "the OpenGL context of the Zap GL widget",
        );
    }
    /// `glEnd()`.
    fn gl_end(&mut self) {
        report_once("glEnd", "the OpenGL context of the Zap GL widget");
    }
    /// `glVertex2i(x, y)`.
    fn gl_vertex_2i(&mut self, x: i32, y: i32) {
        report_once("glVertex2i", "the OpenGL context of the Zap GL widget");
    }
    /// `glLineStipple(factor, pattern)`.
    fn gl_line_stipple(&mut self, factor: i32, pattern: u16) {
        report_once("glLineStipple", "the OpenGL context of the Zap GL widget");
    }

    // ---- dia_qtutils.cpp --------------------------------------------------
    /// `diaMaximumWindowSize(maxWinx, maxWiny, widget)`.
    fn dia_maximum_window_size(&mut self) -> (i32, i32) {
        report_once(
            "diaMaximumWindowSize",
            "the Qt screen list of dia_qtutils.cpp",
        );
        (0, 0)
    }
    /// `diaMaxWinSizeAtPos(x, y, maxWinx, maxWiny, dpr)`.
    fn dia_max_win_size_at_pos(&mut self, x: i32, y: i32) -> (i32, i32, f32) {
        report_once(
            "diaMaxWinSizeAtPos",
            "the Qt screen list of dia_qtutils.cpp",
        );
        (0, 0, 0.)
    }
    /// `diaMinimumWindowPos(left, top, widget)`.
    fn dia_minimum_window_pos(&mut self) -> (i32, i32) {
        report_once(
            "diaMinimumWindowPos",
            "the Qt screen list of dia_qtutils.cpp",
        );
        (0, 0)
    }
    /// `diaLimitWindowPos(neww, newh, xleft, ytop, widget)`.
    fn dia_limit_window_pos(&mut self, neww: i32, newh: i32, xleft: &mut i32, ytop: &mut i32) {
        report_once("diaLimitWindowPos", "the Qt screen list of dia_qtutils.cpp");
    }
    /// `diaLimitWindowSize(neww, newh, widget)`.
    fn dia_limit_window_size(&mut self, neww: &mut i32, newh: &mut i32) {
        report_once(
            "diaLimitWindowSize",
            "the Qt screen list of dia_qtutils.cpp",
        );
    }
    /// `diaLimitWinSizeAtPos(x, y, neww, newh)`.
    fn dia_limit_win_size_at_pos(&mut self, x: i32, y: i32, neww: &mut i32, newh: &mut i32) {
        report_once(
            "diaLimitWinSizeAtPos",
            "the Qt screen list of dia_qtutils.cpp",
        );
    }
    /// `ImodInfoWin->raise()`.
    fn info_win_raise(&mut self) {
        report_once(
            "ImodInfoWin->raise",
            "the information window of info_setup.cpp",
        );
    }
    /// `ImodInfoWin->frameGeometry()` as `[x, y, width, height]`.
    fn info_win_frame_geometry(&mut self) -> [i32; 4] {
        report_once(
            "ImodInfoWin->frameGeometry",
            "the information window of info_setup.cpp",
        );
        [0; 4]
    }
    /// `ImodInfoWin->geometry()` as `[x, y, width, height]`.
    fn info_win_geometry(&mut self) -> [i32; 4] {
        report_once(
            "ImodInfoWin->geometry",
            "the information window of info_setup.cpp",
        );
        [0; 4]
    }
    /// `ImodInfoWin->width()` and `height()`.
    fn info_win_size(&mut self) -> (i32, i32) {
        report_once(
            "ImodInfoWin->width",
            "the information window of info_setup.cpp",
        );
        (0, 0)
    }
    /// `ImodInfoWin->move(x, y)`.
    fn info_win_move(&mut self, x: i32, y: i32) {
        report_once(
            "ImodInfoWin->move",
            "the information window of info_setup.cpp",
        );
    }

    // ---- utilities.cpp members with no translation yet --------------------
    /// `utilInitializeScreenChange(mQtWindow, mDevicePixelRatio)`.
    fn util_initialize_screen_change(&mut self) -> f32 {
        report_once(
            "utilInitializeScreenChange",
            "the screen-change tracking of utilities.cpp",
        );
        1.
    }
    /// `utilGetNewDevPixRatio(mQtWindow)`.
    fn util_get_new_dev_pix_ratio(&mut self) -> f32 {
        report_once(
            "utilGetNewDevPixRatio",
            "the screen-change tracking of utilities.cpp",
        );
        0.
    }
    /// `utilNeedToSetCursor()`.
    fn util_need_to_set_cursor(&mut self) -> bool {
        report_once(
            "utilNeedToSetCursor",
            "the Qt cursor state of utilities.cpp",
        );
        false
    }
    /// `utilSetCursor(mode, setAnyway, needSpecial, needSizeAll, dragging,
    /// needModel, mMousemode, mLastShape, mGfx)`; the two by-reference
    /// members travel back out.
    #[allow(clippy::too_many_arguments)]
    fn util_set_cursor(
        &mut self,
        mode: i32,
        set_anyway: bool,
        need_special: bool,
        need_size_all: bool,
        dragging: [i32; 4],
        need_model: bool,
        mousemode: &mut i32,
        last_shape: &mut i32,
    ) {
        report_once("utilSetCursor", "the Qt cursor set of utilities.cpp");
    }
    /// `utilRaiseIfNeeded(mQtWindow, event)`.
    fn util_raise_if_needed(&mut self, event: &KeyEvent) {
        report_once("utilRaiseIfNeeded", "the Qt window stack of utilities.cpp");
    }
    /// `utilWprintMeasure(str, imod, value, area)`.
    fn util_wprint_measure(&mut self, text: &str, imod: *mut Imod, value: f32, area: bool) {
        report_once("utilWprintMeasure", "the unit conversion of utilities.cpp");
    }
    /// `utilAutoNewContour(vi, cont, notInPlane, timeMismatch, timeLock,
    /// newSurf, "sections", "Z plane")`.
    #[allow(clippy::too_many_arguments)]
    fn util_auto_new_contour(
        &mut self,
        vi: *mut ImodView,
        cont: *mut Icont,
        not_in_plane: bool,
        time_mismatch: bool,
        time_lock: i32,
        new_surf: i32,
    ) -> *mut Icont {
        report_once(
            "utilAutoNewContour",
            "the new-contour helper of utilities.cpp",
        );
        ptr::null_mut()
    }
    /// `utilAssignSurfToCont(vi, obj, cont, newSurf)`.
    fn util_assign_surf_to_cont(
        &mut self,
        vi: *mut ImodView,
        obj: *mut Iobj,
        cont: *mut Icont,
        new_surf: i32,
    ) {
        report_once(
            "utilAssignSurfToCont",
            "the new-contour helper of utilities.cpp",
        );
    }
    /// `utilManagePairedMeshes(obj, ob)`.
    fn util_manage_paired_meshes(&mut self, obj: *mut Iobj, ob: i32) -> i32 {
        report_once(
            "utilManagePairedMeshes",
            "the paired-mesh handling of utilities.cpp",
        );
        0
    }
    /// `utilPreSnapChanges(vi)`.
    fn util_pre_snap_changes(&mut self, vi: *mut ImodView) {
        report_once(
            "utilPreSnapChanges",
            "the snapshot helpers of utilities.cpp",
        );
    }
    /// `utilRestoreSnapChanges(vi)`.
    fn util_restore_snap_changes(&mut self, vi: *mut ImodView) {
        report_once(
            "utilRestoreSnapChanges",
            "the snapshot helpers of utilities.cpp",
        );
    }
    /// `utilStartMontSnap(winx, winy, xFull, yFull, scaling, barSaved,
    /// numChunks, &framePix, &fullPix, &linePtrs)`.
    #[allow(clippy::too_many_arguments)]
    fn util_start_mont_snap(
        &mut self,
        winx: i32,
        winy: i32,
        x_full_size: i32,
        y_full_size: i32,
        scaling_factor: f32,
        num_chunks: &mut i32,
    ) -> i32 {
        report_once("utilStartMontSnap", "the montage snapshot of utilities.cpp");
        1
    }
    /// `utilMontSnapScaleBar(ix, iy, factor, winx - 4, winy - 4, zoom, draw)`.
    #[allow(clippy::too_many_arguments)]
    fn util_mont_snap_scale_bar(
        &mut self,
        ix: i32,
        iy: i32,
        factor: i32,
        base_x: i32,
        base_y: i32,
        zoom: f32,
        draw_bar: bool,
    ) {
        report_once(
            "utilMontSnapScaleBar",
            "the montage snapshot of utilities.cpp",
        );
    }
    /// `memLineCpy(linePtrs, framePix, xCopy, yCopy, 4, toX, toY, winx,
    /// fromX, fromY)`.
    #[allow(clippy::too_many_arguments)]
    fn util_mont_snap_copy_frame(
        &mut self,
        x_copy: i32,
        y_copy: i32,
        to_x: i32,
        to_y: i32,
        win_x: i32,
        from_x: i32,
        from_y: i32,
    ) {
        report_once(
            "memLineCpy",
            "the montage snapshot buffers of utilities.cpp",
        );
    }
    /// `utilFinishMontSnap(linePtrs, xFull, yFull, format, fileno, 3, factor,
    /// "zap", "3dmod: Saving zap")`.
    #[allow(clippy::too_many_arguments)]
    fn util_finish_mont_snap(
        &mut self,
        x_full_size: i32,
        y_full_size: i32,
        format: i32,
        fileno: &mut i32,
        digits: i32,
        zoom: f32,
    ) {
        report_once(
            "utilFinishMontSnap",
            "the montage snapshot of utilities.cpp",
        );
    }
    /// `utilFreeMontSnapArrays(fullPix, numChunks, framePix, linePtrs)`.
    fn util_free_mont_snap_arrays(&mut self, num_chunks: i32) {
        report_once(
            "utilFreeMontSnapArrays",
            "the montage snapshot of utilities.cpp",
        );
    }
    /// The frame buffer `utilStartMontSnap` allocated, handed to
    /// `glReadPixels`.
    fn util_mont_snap_frame_pix(&mut self) -> *mut u8 {
        report_once("utilStartMontSnap framePix", "the montage snapshot buffers");
        ptr::null_mut()
    }
    /// `setupFilledContTesselator()`.
    fn setup_filled_cont_tesselator(&mut self) {
        report_once(
            "setupFilledContTesselator",
            "the GLU tessellator of utilities.cpp",
        );
    }

    // ---- moviecon.cpp (its MovieConState is the host's) -------------------
    /// `imcGetSnapshot(vi)`.
    fn imc_get_snapshot(&mut self, vi: *mut ImodView) -> i32 {
        report_once("imcGetSnapshot", "the MovieConState of moviecon.cpp");
        0
    }
    /// `imcGetStarterID()`.
    fn imc_get_starter_id(&mut self) -> i32 {
        report_once("imcGetStarterID", "the MovieConState of moviecon.cpp");
        -1
    }
    /// `imcSetStarterID(ctrl)`.
    fn imc_set_starter_id(&mut self, ctrl: i32) {
        report_once("imcSetStarterID", "the MovieConState of moviecon.cpp");
    }
    /// `imcGetStartEnd(vi, axis, &start, &end)`.
    fn imc_get_start_end(&mut self, vi: *mut ImodView, axis: i32) -> (i32, i32) {
        report_once("imcGetStartEnd", "the MovieConState of moviecon.cpp");
        (0, 0)
    }
    /// `imcGetIncrement(vi, axis)`.
    fn imc_get_increment(&mut self, vi: *mut ImodView, axis: i32) -> i32 {
        report_once("imcGetIncrement", "the MovieConState of moviecon.cpp");
        1
    }
    /// `imcGetLoopMode(vi)`.
    fn imc_get_loop_mode(&mut self, vi: *mut ImodView) -> i32 {
        report_once("imcGetLoopMode", "the MovieConState of moviecon.cpp");
        0
    }
    /// `imcStartSnapHere(vi)`.
    fn imc_start_snap_here(&mut self, vi: *mut ImodView) -> i32 {
        report_once("imcStartSnapHere", "the MovieConState of moviecon.cpp");
        0
    }
    /// `imcGetSnapMontage(doing)`.
    fn imc_get_snap_montage(&mut self, doing: bool) -> bool {
        report_once("imcGetSnapMontage", "the MovieConState of moviecon.cpp");
        false
    }
    /// `imcGetMontageFactor()`.
    fn imc_get_montage_factor(&mut self) -> i32 {
        report_once("imcGetMontageFactor", "the MovieConState of moviecon.cpp");
        2
    }
    /// `imcGetSnapWholeMont()`.
    fn imc_get_snap_whole_mont(&mut self) -> i32 {
        report_once("imcGetSnapWholeMont", "the MovieConState of moviecon.cpp");
        0
    }
    /// `imcGetScaleSizes()`.
    fn imc_get_scale_sizes(&mut self) -> bool {
        report_once("imcGetScaleSizes", "the MovieConState of moviecon.cpp");
        false
    }
    /// `imcGetSizeScaling()`.
    fn imc_get_size_scaling(&mut self) -> i32 {
        report_once("imcGetSizeScaling", "the MovieConState of moviecon.cpp");
        1
    }
    /// `imodMovieXYZT(vi, x, y, z, t)`.
    fn imod_movie_xyzt(&mut self, vi: *mut ImodView, x: i32, y: i32, z: i32, t: i32) {
        report_once("imodMovieXYZT", "the movie timers of workprocs.cpp");
    }

    // ---- display.cpp / info_cb.cpp ----------------------------------------
    /// `imodDraw(vi, flag)`.
    fn imod_draw(&mut self, vi: *mut ImodView, flag: i32) {
        report_once("imodDraw", "the draw dispatcher of display.cpp");
    }
    /// `imodSetObjectColor(ob)`.
    fn imod_set_object_color(&mut self, ob: i32) {
        report_once("imodSetObjectColor", "the colour map of display.cpp");
    }
    /// `customGhostColor(red, green, blue)`.
    fn custom_ghost_color(&mut self, red: i32, green: i32, blue: i32) {
        report_once("customGhostColor", "the colour map of display.cpp");
    }
    /// `resetGhostColor()`.
    fn reset_ghost_color(&mut self) {
        report_once("resetGhostColor", "the colour map of display.cpp");
    }
    /// `imod_info_input()`.
    fn imod_info_input(&mut self) {
        report_once("imod_info_input", "the event dispatcher of info_cb.cpp");
    }
    /// `imod_setxyzmouse()`.
    fn imod_setxyzmouse(&mut self) {
        report_once("imod_setxyzmouse", "the information window of info_cb.cpp");
    }
    /// `imod_info_setxyz()`.
    fn imod_info_setxyz(&mut self) {
        report_once("imod_info_setxyz", "the information window of info_cb.cpp");
    }
    /// `imod_info_bwfloat(vi, section, time)`.
    fn imod_info_bwfloat(&mut self, vi: *mut ImodView, section: i32, time: i32) -> i32 {
        report_once("imod_info_bwfloat", "the InfoCbState of info_cb.cpp");
        0
    }
    /// `imodInfoTimeRampIndex(vi, timeLock)`.
    fn imod_info_time_ramp_index(&mut self, vi: *mut ImodView, time_lock: i32) -> i32 {
        report_once("imodInfoTimeRampIndex", "the InfoCbState of info_cb.cpp");
        -1
    }
    /// `imodInfoUpdateOnly(value)`.
    fn imod_info_update_only(&mut self, value: i32) {
        report_once("imodInfoUpdateOnly", "the InfoCbState of info_cb.cpp");
    }
    /// `imodShowHelpPage(page)`.
    fn imod_show_help_page(&mut self, page: &str) {
        report_once(
            "imodShowHelpPage",
            "the help assistant of imod_assistant.cpp",
        );
    }
    /// `locatorScheduleDraw(vi)`.
    fn locator_schedule_draw(&mut self, vi: *mut ImodView) {
        report_once("locatorScheduleDraw", "the locator window of locator.cpp");
    }

    // ---- imod_input.cpp ---------------------------------------------------
    /// `inputTestCtrl(event)`.
    fn input_test_ctrl(&mut self, modifiers: i32) -> i32 {
        i32::from(modifiers & CONTROL_MODIFIER != 0)
    }
    /// `inputTestMetaKey(event)`.
    fn input_test_meta_key(&mut self, event: &KeyEvent) -> i32 {
        0
    }
    /// `inputConvertNumLock(keysym, keypad)`.
    fn input_convert_num_lock(&mut self, keysym: &mut i32, keypad: &mut i32) {}
    /// `inputPageUpOrDown(vi, shifted, direction)`.
    fn input_page_up_or_down(&mut self, vi: *mut ImodView, shifted: i32, direction: i32) {
        report_once(
            "inputPageUpOrDown",
            "the input dispatcher of imod_input.cpp",
        );
    }
    /// `inputKeyPointMove(vi, keysym)`.
    fn input_key_point_move(&mut self, vi: *mut ImodView, keysym: i32) {
        report_once(
            "inputKeyPointMove",
            "the input dispatcher of imod_input.cpp",
        );
    }
    /// `inputNextz(vi, n)`.
    fn input_nextz(&mut self, vi: *mut ImodView, step: i32) {
        report_once("inputNextz", "the input dispatcher of imod_input.cpp");
    }
    /// `inputPrevz(vi, n)`.
    fn input_prevz(&mut self, vi: *mut ImodView, step: i32) {
        report_once("inputPrevz", "the input dispatcher of imod_input.cpp");
    }
    /// `inputSaveModel(vi)`.
    fn input_save_model(&mut self, vi: *mut ImodView) {
        report_once("inputSaveModel", "the model writer of imod_io.cpp");
    }
    /// `inputNextTime(vi)`.
    fn input_next_time(&mut self, vi: *mut ImodView) {
        report_once("inputNextTime", "the input dispatcher of imod_input.cpp");
    }
    /// `inputPrevTime(vi)`.
    fn input_prev_time(&mut self, vi: *mut ImodView) {
        report_once("inputPrevTime", "the input dispatcher of imod_input.cpp");
    }
    /// `inputQDefaultKeys(event, vi)`.
    fn input_q_default_keys(&mut self, event: &KeyEvent, vi: *mut ImodView) {
        report_once(
            "inputQDefaultKeys",
            "the input dispatcher of imod_input.cpp",
        );
    }
    /// `inputSetTimeLockForFKeys(timeLock)`.
    fn input_set_time_lock_for_f_keys(&mut self, time_lock: i32) {
        report_once(
            "inputSetTimeLockForFKeys",
            "the input dispatcher of imod_input.cpp",
        );
    }

    // ---- imodplug.cpp -----------------------------------------------------
    /// `imodPlugHandleKey(vi, event, type)`.
    fn imod_plug_handle_key(&mut self, vi: *mut ImodView, event: &KeyEvent, wtype: i32) -> i32 {
        report_once("imodPlugHandleKey", "the plugin list of imodplug.cpp");
        0
    }
    /// `imodPlugHandleMouse(vi, event, imx, imy, but1, but2, but3, type)`.
    #[allow(clippy::too_many_arguments)]
    fn imod_plug_handle_mouse(
        &mut self,
        vi: *mut ImodView,
        event: &KeyEvent,
        imx: f32,
        imy: f32,
        but1: i32,
        but2: i32,
        but3: i32,
        wtype: i32,
    ) -> i32 {
        report_once("imodPlugHandleMouse", "the plugin list of imodplug.cpp");
        0
    }
    /// `imodPlugHandleEvent(vi, e, imx, imy, type)`.
    fn imod_plug_handle_event(
        &mut self,
        vi: *mut ImodView,
        event: &ZapGeneralEvent,
        imx: f32,
        imy: f32,
        wtype: i32,
    ) -> i32 {
        report_once("imodPlugHandleEvent", "the plugin list of imodplug.cpp");
        0
    }

    // ---- other 3dmod units ------------------------------------------------
    /// `pvNewMousePosition(vi, imx, imy, imz)`.
    fn pv_new_mouse_position(&mut self, vi: *mut ImodView, imx: f32, imy: f32, imz: i32) {
        report_once(
            "pvNewMousePosition",
            "the pixel view window of pixelview.cpp",
        );
    }
    /// `iprocIsOpen()`.
    fn iproc_is_open(&mut self) -> bool {
        report_once("iprocIsOpen", "the image processing window of iproc.cpp");
        false
    }
    /// `iprocBusy()`.
    fn iproc_busy(&mut self) -> i32 {
        report_once("iprocBusy", "the image processing window of iproc.cpp");
        0
    }
    /// `iprocApply()`.
    fn iproc_apply(&mut self) {
        report_once("iprocApply", "the image processing window of iproc.cpp");
    }
    /// `iprocToggleFullFFT(vi)`.
    fn iproc_toggle_full_fft(&mut self, vi: *mut ImodView) {
        report_once(
            "iprocToggleFullFFT",
            "the image processing window of iproc.cpp",
        );
    }
    /// `autox_next(vi->ax)`.
    fn autox_next(&mut self, vi: *mut ImodView) {
        report_once("autox_next", "the auto contour window of autox.cpp");
    }
    /// `autox_smooth(vi->ax)`.
    fn autox_smooth(&mut self, vi: *mut ImodView) {
        report_once("autox_smooth", "the auto contour window of autox.cpp");
    }
    /// `autox_build(vi->ax)`.
    fn autox_build(&mut self, vi: *mut ImodView) {
        report_once("autox_build", "the auto contour window of autox.cpp");
    }
    /// `autox_fillmouse(vi, x, y)`.
    fn autox_fillmouse(&mut self, vi: *mut ImodView, x: i32, y: i32) {
        report_once("autox_fillmouse", "the auto contour window of autox.cpp");
    }
    /// `autox_sethigh(vi, x, y)`.
    fn autox_sethigh(&mut self, vi: *mut ImodView, x: i32, y: i32) {
        report_once("autox_sethigh", "the auto contour window of autox.cpp");
    }
    /// `autox_setlow(vi, x, y)`.
    fn autox_setlow(&mut self, vi: *mut ImodView, x: i32, y: i32) {
        report_once("autox_setlow", "the auto contour window of autox.cpp");
    }
    /// `vi->ax->altmouse`, `vi->ax->filled`, `vi->ax->cz` and `vi->ax->data`
    /// as the three reads `xzap.cpp` makes of them; `data` is the flag byte
    /// at the given index.
    fn autox_altmouse(&mut self, vi: *mut ImodView) -> i32 {
        report_once("vi->ax->altmouse", "the Autox state of autox.cpp");
        0
    }
    /// See [`ZapNativeBoundary::autox_altmouse`].
    fn autox_filled_at_section(&mut self, vi: *mut ImodView, section: i32) -> bool {
        report_once("vi->ax->filled", "the Autox state of autox.cpp");
        false
    }
    /// See [`ZapNativeBoundary::autox_altmouse`].
    fn autox_data(&mut self, vi: *mut ImodView, index: usize) -> u8 {
        report_once("vi->ax->data", "the Autox state of autox.cpp");
        0
    }
    /// `imodSelectionListQuery(vi, ob, co)`.
    fn imod_selection_list_query(&mut self, vi: *mut ImodView, ob: i32, co: i32) -> i32 {
        report_once(
            "imodSelectionListQuery",
            "the selection list of imod_edit.cpp",
        );
        -2
    }
    /// `imodSelectionListAdd(vi, index)`.
    fn imod_selection_list_add(&mut self, vi: *mut ImodView, index: Iindex) {
        report_once(
            "imodSelectionListAdd",
            "the selection list of imod_edit.cpp",
        );
    }
    /// `imodSelectionNewCurPoint(vi, imod, indSave, controlDown)`.
    fn imod_selection_new_cur_point(
        &mut self,
        vi: *mut ImodView,
        imod: *mut Imod,
        ind_save: Iindex,
        control_down: i32,
    ) {
        report_once(
            "imodSelectionNewCurPoint",
            "the selection list of imod_edit.cpp",
        );
    }
    /// `imodAllObjNearest(vi, &index, &pnt, selsize, time)`.
    fn imod_all_obj_nearest(
        &mut self,
        vi: *mut ImodView,
        index: &mut Iindex,
        pnt: &Ipoint,
        selsize: f32,
        time: i32,
    ) -> f32 {
        report_once("imodAllObjNearest", "the selection list of imod_edit.cpp");
        -1.
    }
    /// `imodvIsosurfaceUpdate(drawFlags)`.
    fn imodv_isosurface_update(&mut self, draw_flags: i32) -> bool {
        report_once(
            "imodvIsosurfaceUpdate",
            "the isosurface dialog of isosurface.cpp",
        );
        false
    }
    /// `imodv_draw()`.
    fn imodv_draw(&mut self) {
        report_once("imodv_draw", "the model view window of imodv.cpp");
    }
    /// `mvImageDrawingZplanes()`.
    fn mv_image_drawing_zplanes(&mut self) -> bool {
        report_once("mvImageDrawingZplanes", "the image state of mv_image.cpp");
        false
    }
    /// `iceGetWheelForSize()`.
    fn ice_get_wheel_for_size(&mut self) -> bool {
        report_once("iceGetWheelForSize", "the contour editor of cont_edit.cpp");
        false
    }
    /// `scaleBarDraw(winx, winy, zoom, 0, mGfx, dpr)`.
    fn scale_bar_draw(&mut self, winx: i32, winy: i32, zoom: f32, background: i32) -> f32 {
        report_once("scaleBarDraw", "the scale bar of scalebar.cpp");
        -1.
    }
    /// `scaleBarGetParams()->draw`, the only member `montageSnapshot` reads
    /// while the saved copy is in place.
    fn scale_bar_draw_flag(&mut self) -> bool {
        report_once("scaleBarGetParams", "the scale bar of scalebar.cpp");
        false
    }
    /// `barSaved = *barReal` and `*barReal = barSaved` around the montage
    /// snapshot: `true` saves, `false` restores.
    fn scale_bar_save_or_restore(&mut self, save: bool) {
        report_once("scaleBarGetParams", "the scale bar of scalebar.cpp");
    }
    /// `vi->pyrCache->getSectionArea(...)`; returns the line pointers.
    #[allow(clippy::too_many_arguments)]
    fn pyr_cache_get_section_area(
        &mut self,
        vi: *mut ImodView,
        section: i32,
        x_start: i32,
        y_start: i32,
        x_draw_size: i32,
        y_draw_size: i32,
        zoom: f32,
        async_load: bool,
        out_x_draw: &mut i32,
        out_y_draw: &mut i32,
        x_offset: &mut f32,
        y_offset: &mut f32,
        tile_scale: &mut i32,
        status: &mut i32,
    ) -> *mut *mut u8 {
        report_once(
            "PyramidCache::getSectionArea",
            "the tile cache host of pyramidcache.cpp",
        );
        ptr::null_mut()
    }
    /// `vi->pyrCache->zoomRequiresBigLoad(zoom, winx, winy)`.
    fn pyr_cache_zoom_requires_big_load(
        &mut self,
        vi: *mut ImodView,
        zoom: f64,
        winx: i32,
        winy: i32,
    ) -> bool {
        report_once(
            "PyramidCache::zoomRequiresBigLoad",
            "the tile cache host of pyramidcache.cpp",
        );
        false
    }

    // ---- finegrain.cpp (its FgData and value state are the host's) --------
    /// `ifgSetupValueDrawing(obj, GEN_STORE_MINMAX1)`.
    fn ifg_setup_value_drawing(&mut self, obj: *mut Iobj, store_type: i16) -> i32 {
        report_once("ifgSetupValueDrawing", "the FgData of finegrain.cpp");
        0
    }
    /// `ifgResetValueSetup()`.
    fn ifg_reset_value_setup(&mut self) {
        report_once("ifgResetValueSetup", "the FgData of finegrain.cpp");
    }
    /// `ifgGetValueSetupState()`.
    fn ifg_get_value_setup_state(&mut self) -> i32 {
        report_once("ifgGetValueSetupState", "the FgData of finegrain.cpp");
        0
    }
    /// `ifgStippleGaps()`.
    fn ifg_stipple_gaps(&mut self) -> bool {
        report_once("ifgStippleGaps", "the FgData of finegrain.cpp");
        false
    }
    /// `ifgShowConnections()`.
    fn ifg_show_connections(&mut self) -> bool {
        report_once("ifgShowConnections", "the FgData of finegrain.cpp");
        false
    }
    /// `ifgHandleContChange(obj, co, &contProps, &ptProps, &stateFlags,
    /// handleFlags, selected, scaleThick)`.
    #[allow(clippy::too_many_arguments)]
    fn ifg_handle_cont_change(
        &mut self,
        obj: *mut Iobj,
        co: i32,
        cont_props: &mut DrawProps,
        pt_props: &mut DrawProps,
        state_flags: &mut i32,
        handle_flags: i32,
        selected: i32,
        scale_thick: i32,
    ) -> i32 {
        report_once("ifgHandleContChange", "the FgData of finegrain.cpp");
        -1
    }
    /// `ifgHandleSurfChange(obj, surf, &defProps, &curProps, &stateFlags, 0)`.
    fn ifg_handle_surf_change(
        &mut self,
        obj: *mut Iobj,
        surf: i32,
        def_props: &mut DrawProps,
        cur_props: &mut DrawProps,
        state_flags: &mut i32,
        handle_flags: i32,
    ) {
        report_once("ifgHandleSurfChange", "the FgData of finegrain.cpp");
    }
    /// `ifgHandleNextChange(obj, store, &contProps, &ptProps, &stateFlags,
    /// &changeFlags, handleFlags, selected, scaleThick)`.
    #[allow(clippy::too_many_arguments)]
    fn ifg_handle_next_change(
        &mut self,
        obj: *mut Iobj,
        store: *const Vec<Istore>,
        cont_props: &mut DrawProps,
        pt_props: &mut DrawProps,
        state_flags: &mut i32,
        change_flags: &mut i32,
        handle_flags: i32,
        selected: i32,
        scale_thick: i32,
    ) -> i32 {
        report_once("ifgHandleNextChange", "the FgData of finegrain.cpp");
        -1
    }
    /// `ifgHandleMeshChange(obj, store, &defProps, &curProps, &nextItemIndex,
    /// i, &stateFlags, &changeFlags, 0)`.
    #[allow(clippy::too_many_arguments)]
    fn ifg_handle_mesh_change(
        &mut self,
        obj: *mut Iobj,
        store: *const Vec<Istore>,
        def_props: &mut DrawProps,
        cur_props: &mut DrawProps,
        next_item_index: &mut i32,
        cur_index: i32,
        state_flags: &mut i32,
        change_flags: &mut i32,
        handle_flags: i32,
    ) -> i32 {
        report_once("ifgHandleMeshChange", "the FgData of finegrain.cpp");
        -1
    }
    /// `istoreFirstChangeIndex(mesh->store)`.
    fn istore_first_change_index(&mut self, store: *const Vec<Istore>) -> i32 {
        crate::imod::libimod::istore::istore_first_change_index(unsafe { &*store })
    }
    /// `ImodvClosed` and `Imodv->lowres`, read by `drawMesh` when it picks a
    /// mesh resolution.
    fn imodv_lowres(&mut self) -> i32 {
        report_once("Imodv->lowres", "the model view state of imodv.cpp");
        0
    }
    /// The label font of `drawContour`: `new QFont(QApplication::font())`
    /// with its point size set, plus the `QPainter` when `App->newQtOpenGL`.
    fn make_label_font(&mut self, point_size: f32) {
        report_once("QFont(QApplication::font())", "the Qt font of QApplication");
    }
    /// `mLabelPainter->end(); delete mLabelPainter; delete mLabelFont;`.
    fn delete_label_font(&mut self) {
        report_once("delete mLabelFont", "the Qt font of QApplication");
    }
}

thread_local! {
    /// The source resolves these calls through process-global Qt objects,
    /// `App` and `ImodPrefs`; the Rust host supplies the same UI-thread
    /// ownership boundary.
    pub static ZAP_NATIVE_BOUNDARY: RefCell<Option<Box<dyn ZapNativeBoundary>>> =
        const { RefCell::new(None) };
}

/// The reporting boundary used whenever no host has been installed.
pub struct ZapReportingBoundary;
impl UtilitiesBoundary for ZapReportingBoundary {
    fn draw_symbol(&mut self, _x: i32, _y: i32, _symbol: i32, _size: i32, _filled: bool) {
        report_once("utilDrawSymbol", "the OpenGL context of b3dgfx.cpp");
    }
    fn set_stipple(&mut self, _enabled: bool) {
        report_once("utilEnableStipple", "the OpenGL context of b3dgfx.cpp");
    }
    fn clear_window(&mut self, _color_index: i32) {
        report_once("utilClearWindow", "the OpenGL context of b3dgfx.cpp");
    }
    fn redraw_model(&mut self) {
        report_once(
            "imodDraw for utilities",
            "the draw dispatcher of display.cpp",
        );
    }
    fn change_point_size(&mut self) {
        report_once("imodPointSetSize", "the contour editor of cont_edit.cpp");
    }
    fn finish_undo_unit(&mut self) {
        report_once("vi->undo->finishUnit", "the undo stack of undoredo.cpp");
    }
    fn message(&mut self, text: &str) {
        wprint(text);
    }
    fn flip_yz(&mut self, _imod: &mut Imod) {
        report_once("imodFlipYZ", "the model flip of imodel.cpp");
    }
    fn rotate_90_x(&mut self, _imod: &mut Imod, _inverse: bool) {
        report_once("imodRot90X", "the model rotation of imodel.cpp");
    }
    fn draw_filled_polygon(&mut self, _points: &[Ipoint]) {
        report_once("drawFilledPolygon", "the GLU tessellator of utilities.cpp");
    }
}
impl ZapNativeBoundary for ZapReportingBoundary {}

/// Runs `action` against the installed boundary, or against the reporting one.
fn with_boundary<T>(action: impl FnOnce(&mut dyn ZapNativeBoundary) -> T) -> T {
    ZAP_NATIVE_BOUNDARY.with(|slot| match slot.borrow_mut().as_deref_mut() {
        Some(boundary) => action(boundary),
        None => action(&mut ZapReportingBoundary),
    })
}

/// Installs the host for this thread; the previous one is returned.
pub fn set_zap_native_boundary(
    boundary: Option<Box<dyn ZapNativeBoundary>>,
) -> Option<Box<dyn ZapNativeBoundary>> {
    ZAP_NATIVE_BOUNDARY.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), boundary))
}

/// `QTIME` (`imodP.h:347`) is a `QElapsedTimer`; only `start`, `restart` and
/// `elapsed` are used, which `Instant` provides directly.
type QTime = std::time::Instant;

thread_local! {
    /// `sDragRegisterSize` (`xzap.cpp:68`).
    static S_DRAG_REGISTER_SIZE: Cell<i32> = const { Cell::new(10) };
    /// `sHqDrawTimeCrit` (`xzap.cpp:69`).
    static S_HQ_DRAW_TIME_CRIT: Cell<i32> = const { Cell::new(100) };
    /// `sInsertDown` (`xzap.cpp:72`).
    static S_INSERT_DOWN: Cell<i32> = const { Cell::new(0) };
    /// `sPixelViewOpen` (`xzap.cpp:73`).
    static S_PIXEL_VIEW_OPEN: Cell<bool> = const { Cell::new(false) };
    /// `sInsertTime` (`xzap.cpp:75`).
    static S_INSERT_TIME: RefCell<QTime> = RefCell::new(QTime::now());
    /// `sBut1downt` (`xzap.cpp:76`).
    static S_BUT1_DOWNT: RefCell<QTime> = RefCell::new(QTime::now());
    /// `sNumZapWindows` (`xzap.cpp:78`).
    static S_NUM_ZAP_WINDOWS: Cell<i32> = const { Cell::new(0) };
    /// `sSubStartX` (`xzap.cpp:79`).
    static S_SUB_START_X: Cell<i32> = const { Cell::new(0) };
    /// `sSubStartY` (`xzap.cpp:80`).
    static S_SUB_START_Y: Cell<i32> = const { Cell::new(0) };
    /// `sSubEndX` (`xzap.cpp:81`).
    static S_SUB_END_X: Cell<i32> = const { Cell::new(0) };
    /// `sSubEndY` (`xzap.cpp:82`).
    static S_SUB_END_Y: Cell<i32> = const { Cell::new(0) };
    /// `sFirstZapOpening` (`xzap.cpp:83`).
    static S_FIRST_ZAP_OPENING: Cell<bool> = const { Cell::new(true) };
    /// `sFirstDrag` (`xzap.cpp:85`).
    static S_FIRST_DRAG: Cell<i32> = const { Cell::new(0) };
    /// `sMoveBandLasso` (`xzap.cpp:86`).
    static S_MOVE_BAND_LASSO: Cell<i32> = const { Cell::new(0) };
    /// `sDragBandLasso` (`xzap.cpp:87`).
    static S_DRAG_BAND_LASSO: Cell<i32> = const { Cell::new(0) };
    /// `sDragging[4]` (`xzap.cpp:88`).
    static S_DRAGGING: Cell<[i32; 4]> = const { Cell::new([0; 4]) };
    /// `sFirstmx` (`xzap.cpp:89`).
    static S_FIRSTMX: Cell<i32> = const { Cell::new(0) };
    /// `sFirstmy` (`xzap.cpp:89`).
    static S_FIRSTMY: Cell<i32> = const { Cell::new(0) };
    /// `sMaxMultiZarea` (`xzap.cpp:90`).
    static S_MAX_MULTI_Z_AREA: Cell<i32> = const { Cell::new(0) };
    /// `sScaleSizes` (`xzap.cpp:91`).
    static S_SCALE_SIZES: Cell<i32> = const { Cell::new(1) };
    /// `sMousePressed` (`xzap.cpp:92`).
    static S_MOUSE_PRESSED: Cell<bool> = const { Cell::new(false) };
    /// `sNextOpenHQstate` (`xzap.cpp:93`).
    static S_NEXT_OPEN_HQ_STATE: Cell<i32> = const { Cell::new(-1) };
    /// `sMinXforRestraining` (`xzap.cpp:94`).
    static S_MIN_X_FOR_RESTRAINING: Cell<i32> = const { Cell::new(1050) };
    /// `sNextMultiZnumX` (`xzap.cpp:95`).
    static S_NEXT_MULTI_Z_NUM_X: Cell<i32> = const { Cell::new(0) };
    /// `sNextMultiZnumY` (`xzap.cpp:96`).
    static S_NEXT_MULTI_Z_NUM_Y: Cell<i32> = const { Cell::new(0) };
    /// `sNextMultiZXsize` (`xzap.cpp:97`).
    static S_NEXT_MULTI_Z_X_SIZE: Cell<i32> = const { Cell::new(0) };
    /// `sNextMultiZYsize` (`xzap.cpp:98`).
    static S_NEXT_MULTI_Z_Y_SIZE: Cell<i32> = const { Cell::new(0) };
    /// `sContShiftBase` (`xzap.cpp:3447`).
    static S_CONT_SHIFT_BASE: Cell<Ipoint> = const {
        Cell::new(Ipoint { x: 0., y: 0., z: 0. })
    };
    /// `static int trans = 5` (`xzap.cpp:1593`), the arrow-key translation.
    static S_KEY_TRANS: Cell<i32> = const { Cell::new(5) };
    /// `static int ... processing = 0` (`xzap.cpp:2378`) in `mouseMove`.
    static S_MOVE_PROCESSING: Cell<i32> = const { Cell::new(0) };
    /// `static int fileno = 0` (`xzap.cpp:4558`) in `montageSnapshot`.
    static S_MONT_FILENO: Cell<i32> = const { Cell::new(0) };
}

/// `ZapFuncs` (`xzap.h:30`), in declaration order: the public block first,
/// then the private one.
pub struct ZapFuncs {
    /// `mVi`.
    pub vi: *mut ImodView,
    /// `mQtWindow`; the widget itself belongs to the host, which hands back
    /// the pointer from `new ZapWindow` so the source's null tests hold.
    pub qt_window: *mut crate::imod::three_dmod::zap_classes::ZapWindow,
    /// `mGfx`; the GL sub-widget, likewise owned by the host.  It exists
    /// exactly when `mQtWindow` does, so the flag rides with it.
    pub gfx_created: bool,
    /// `mWinx`.
    pub winx: i32,
    /// `mWiny`.
    pub winy: i32,
    /// `mNumXpanels`.
    pub num_xpanels: i32,
    /// `mNumYpanels`.
    pub num_ypanels: i32,
    /// `mPanelZstep`.
    pub panel_zstep: i32,
    /// `mDrawInCenter`.
    pub draw_in_center: i32,
    /// `mDrawInOthers`.
    pub draw_in_others: i32,
    /// `mSection`.
    pub section: i32,
    /// `mRubberband`.
    pub rubberband: i32,
    /// `mRbImageX0`.
    pub rb_image_x0: f32,
    /// `mRbImageX1`.
    pub rb_image_x1: f32,
    /// `mRbImageY0`.
    pub rb_image_y0: f32,
    /// `mRbImageY1`.
    pub rb_image_y1: f32,
    /// `mRbMouseX0`.
    pub rb_mouse_x0: i32,
    /// `mRbMouseX1`.
    pub rb_mouse_x1: i32,
    /// `mRbMouseY0`.
    pub rb_mouse_y0: i32,
    /// `mRbMouseY1`.
    pub rb_mouse_y1: i32,
    /// `mBandChanged`.
    pub band_changed: i32,
    /// `mCtrl`.
    pub ctrl: i32,
    /// `mGinit`.
    pub ginit: i32,
    /// `mImages`; the `B3dCIImage *` array of a multi-Z window.  The images
    /// live in the host's `B3dGfxState`, so what the source's pointer array
    /// carries here is whether each slot is allocated.
    pub images: Vec<bool>,
    /// `mMovieSnapCount`.
    pub movie_snap_count: i32,
    /// `mPopup`.
    pub popup: i32,
    /// `mRecordSubarea`.
    pub record_subarea: i32,
    /// `mShowslice`.
    pub showslice: i16,
    /// `mStartingBand`.
    pub starting_band: i32,
    /// `mToolMaxZ`.
    pub tool_max_z: i32,
    /// `mXtrans`.
    pub xtrans: i32,
    /// `mYtrans`.
    pub ytrans: i32,
    /// `mZtrans`.
    pub ztrans: i32,
    /// `mZoom`.
    pub zoom: f32,
    /// `mNewScreenZoom`.
    pub new_screen_zoom: f32,
    /// `mLock`.
    pub lock: i32,
    /// `mTimeLock`.
    pub time_lock: i32,
    /// `mScaleBarSize`.
    pub scale_bar_size: f32,
    /// `mLassoOn`.
    pub lasso_on: bool,
    /// `mDrawingLasso`.
    pub drawing_lasso: bool,
    /// `mLassoObjNum`.
    pub lasso_obj_num: i32,
    /// `mArrowOn`.
    pub arrow_on: bool,
    /// `mDrawingArrow`.
    pub drawing_arrow: bool,
    /// `mArrowTail`.
    pub arrow_tail: Vec<Ipoint>,
    /// `mArrowHead`.
    pub arrow_head: Vec<Ipoint>,
    /// `mLabelFont`; the `QFont *` is the host's, so this records only
    /// whether one is currently allocated, which is the source's test.
    pub label_font: bool,
    /// `mLabelPainter`; likewise the `QPainter *`.
    pub label_painter: bool,
    /// `mDevicePixelRatio`.
    pub device_pixel_ratio: f32,

    // ---- private block ----------------------------------------------------
    /// `mXborder`.
    pub xborder: i32,
    /// `mYborder`.
    pub yborder: i32,
    /// `mXstart`.
    pub xstart: i32,
    /// `mYstart`.
    pub ystart: i32,
    /// `mXposStart`.
    pub xpos_start: i32,
    /// `mYposStart`.
    pub ypos_start: i32,
    /// `mXlastStart`.
    pub xlast_start: i32,
    /// `mYlastStart`.
    pub ylast_start: i32,
    /// `mXlastSize`.
    pub xlast_size: i32,
    /// `mYlastSize`.
    pub ylast_size: i32,
    /// `mLastStatus`.
    pub last_status: i32,
    /// `mXdrawsize`.
    pub xdrawsize: i32,
    /// `mYdrawsize`.
    pub ydrawsize: i32,
    /// `mLmx`.
    pub lmx: i32,
    /// `mLmy`.
    pub lmy: i32,
    /// `mTessCont`.
    pub tess_cont: *mut Icont,
    /// `mTessMaxPoints`.
    pub tess_max_points: i32,
    /// `mNestContMap`.
    pub nest_cont_map: Vec<i32>,
    /// `mContsAtCurZ`; the source keeps the borrowed row of `contAtZ`, which
    /// `imodContourFreeZTables` owns, so the copy is taken by value here.
    pub conts_at_cur_z: Option<Vec<i32>>,
    /// `mNumNests`.
    pub num_nests: i32,
    /// `mNestInd`.
    pub nest_ind: Vec<i32>,
    /// `mNests`.
    pub nests: Vec<Nesting>,
    /// `mHqgfx`.
    pub hqgfx: i32,
    /// `mHide`.
    pub hide: i32,
    /// `mHqgfxsave`.
    pub hqgfxsave: i32,
    /// `mDrawCurrentOnly`.
    pub draw_current_only: i32,
    /// `mLastHqDrawTime`.
    pub last_hq_draw_time: i32,
    /// `mShiftingCont`.
    pub shifting_cont: i32,
    /// `mXformCenter`.
    pub xform_center: Ipoint,
    /// `mXformFixedPt`.
    pub xform_fixed_pt: Ipoint,
    /// `mCenterDefined`.
    pub center_defined: i32,
    /// `mCenterMarked`.
    pub center_marked: i32,
    /// `mFixedPtDefined`.
    pub fixed_pt_defined: i32,
    /// `mShiftRegistered`.
    pub shift_registered: i32,
    /// `mShiftObjNum`.
    pub shift_obj_num: i32,
    /// `mDragAddCount`.
    pub drag_add_count: i32,
    /// `mDragAddIndex`.
    pub drag_add_index: Iindex,
    /// `mDragAddEnd`.
    pub drag_add_end: i32,
    /// `mDrewExtraCursor`.
    pub drew_extra_cursor: bool,
    /// `mXzoom`.
    pub xzoom: f32,
    /// `mData`; the source declares it and never assigns anything but NULL.
    pub data: *mut libc::c_char,
    /// `mImage`; see [`ZapFuncs::images`].
    pub image: bool,
    /// `mNumImages`.
    pub num_images: i32,
    /// `mSectionStep`.
    pub section_step: i32,
    /// `mTime`.
    pub time: i32,
    /// `mOverlay`.
    pub overlay: i32,
    /// `mKeepcentered`.
    pub keepcentered: i32,
    /// `mMousemode`.
    pub mousemode: i32,
    /// `mLastShape`.
    pub last_shape: i32,
    /// `mToolSection`.
    pub tool_section: i32,
    /// `mToolZoom`.
    pub tool_zoom: f32,
    /// `mToolTime`.
    pub tool_time: i32,
    /// `mToolSizeX`.
    pub tool_size_x: i32,
    /// `mToolSizeY`.
    pub tool_size_y: i32,
    /// `mInsertmode`.
    pub insertmode: i16,
    /// `mShowedSlice`.
    pub showed_slice: i32,
    /// `mDoingDraw`.
    pub doing_draw: bool,
    /// `mDoingMontage`.
    pub doing_montage: bool,
    /// `mPanelXborder`.
    pub panel_xborder: i32,
    /// `mPanelYborder`.
    pub panel_yborder: i32,
    /// `mPanelGutter`.
    pub panel_gutter: i32,
    /// `mPanelXsize`.
    pub panel_xsize: i32,
    /// `mPanelYsize`.
    pub panel_ysize: i32,
    /// `mToolstart`.
    pub toolstart: i32,
    /// `mScreenChanged`.
    pub screen_changed: bool,
    /// `mDeferScreenZoomChange`.
    pub defer_screen_zoom_change: bool,
    /// `mScreenResizeTime[2]`.
    pub screen_resize_time: [QTime; 2],
    /// `mLastXsizeChange`.
    pub last_xsize_change: f32,
    /// `mTwod`.
    pub twod: i32,
}

/// `imod_zap_open` (`xzap.cpp:104`); open the zap window.
///
/// The C++ `new`/`delete` pair becomes a `Box` that is leaked into the dialog
/// manager on success, which is where the source's ownership goes too.
pub fn imod_zap_open(vi: *mut ImodView, wintype: i32) -> i32 {
    let zap = ZapFuncs::new(vi, wintype);
    if zap.qt_window.is_null() {
        drop(zap);
        return -1;
    }
    Box::into_raw(zap);
    0
}

/// `zapReportBiggestMultiZ` (`xzap.cpp:119`); look through all multiZ windows
/// and report params of biggest one.
pub fn zap_report_biggest_multi_z() {
    if !with_boundary(|n| n.imod_prefs_exists()) {
        return;
    }
    let obj_list = with_boundary(|n| n.imod_dialog_manager_window_list(MULTIZ_WINDOW_TYPE));

    for &zap in &obj_list {
        let zap = unsafe { &mut *zap };
        if S_MAX_MULTI_Z_AREA.get() < zap.winx * zap.winy {
            S_MAX_MULTI_Z_AREA.set(zap.winx * zap.winy);
            let pos = with_boundary(|n| n.ivw_restorable_geometry());
            with_boundary(|n| {
                n.prefs_record_multi_z_params(
                    pos,
                    zap.num_xpanels,
                    zap.num_ypanels,
                    zap.panel_zstep,
                    zap.draw_in_center,
                    zap.draw_in_others,
                )
            });
        }
    }
}

/// `setNextMultiZpanelsAndSize` (`xzap.cpp:141`).
pub fn set_next_multi_z_panels_and_size(num_x: i32, num_y: i32, xsize: i32, ysize: i32) {
    S_NEXT_MULTI_Z_NUM_X.set(num_x);
    S_NEXT_MULTI_Z_NUM_Y.set(num_y);
    S_NEXT_MULTI_Z_X_SIZE.set(xsize);
    S_NEXT_MULTI_Z_Y_SIZE.set(ysize);
}

/// `getTopZapWindow` (`xzap.cpp:154`); find the first zap window of the given
/// type, with a rubberband and/or lasso if those flags are set.
pub fn get_top_zap_window(
    with_band: bool,
    with_lasso: bool,
    window_type: i32,
    index: Option<&mut i32>,
) -> *mut ZapFuncs {
    with_boundary(|n| {
        n.imod_dialog_manager_get_top_window(with_band, with_lasso, window_type, index)
    })
}

/// `getTopZapLassoContour` (`xzap.cpp:166`); return the lasso contour from
/// the top with lasso, but only above any rubberband if `above_band`.
pub fn get_top_zap_lasso_contour(above_band: bool) -> *mut Icont {
    let zap = get_top_zap_window(above_band, true, ZAP_WINDOW_TYPE, None);

    if zap.is_null() {
        return ptr::null_mut();
    }
    let zap = unsafe { &mut *zap };
    if !zap.lasso_on || zap.drawing_lasso {
        return ptr::null_mut();
    }
    zap.get_lasso_contour()
}

/// `zapReportRubberband` (`xzap.cpp:178`); report the rubberband coordinates
/// of the first zap window with a band.
pub fn zap_report_rubberband() {
    let mut low_section = 0;
    let mut high_section = 0;

    let zap = get_top_zap_window(true, false, ZAP_WINDOW_TYPE, None);
    if zap.is_null() {
        imod_print_stderr("ERROR: No Zap window has usable rubberband coordinates\n");
        return;
    }
    let zap = unsafe { &mut *zap };

    let bin = unsafe { (*zap.vi).xybin };
    let (mut ixl, mut ixr, mut iyb, mut iyt);
    if zap.rubberband != 0 {
        ixl = (zap.rb_image_x0 as f64 + 0.5).floor() as i32;
        ixr = (zap.rb_image_x1 as f64 - 0.5).floor() as i32;
        iyb = (zap.rb_image_y0 as f64 + 0.5).floor() as i32;
        iyt = (zap.rb_image_y1 as f64 - 0.5).floor() as i32;
    } else {
        // If band is just statring, report the full area
        ixl = 0;
        iyb = 0;
        ixr = unsafe { (*zap.vi).xsize } - 1;
        iyt = unsafe { (*zap.vi).ysize } - 1;
    }

    if ixl < 0 {
        ixl = 0;
    }
    if ixr >= unsafe { (*zap.vi).xsize } {
        ixr = unsafe { (*zap.vi).xsize } - 1;
    }
    if iyb < 0 {
        iyb = 0;
    }
    if iyt >= unsafe { (*zap.vi).ysize } {
        iyt = unsafe { (*zap.vi).ysize } - 1;
    }
    ixl *= bin;
    iyb *= bin;
    ixr = ixr * bin + bin - 1;
    iyt = iyt * bin + bin - 1;
    if zap.get_low_high_section(&mut low_section, &mut high_section) {
        imod_print_stderr(&format!(
            "Rubberband: {} {} {} {} {} {}\n",
            ixl + 1,
            iyb + 1,
            ixr + 1,
            iyt + 1,
            unsafe { (*zap.vi).zbin } * (low_section - 1) + 1,
            unsafe { (*zap.vi).zbin } * high_section
        ));
    } else {
        imod_print_stderr(&format!(
            "Rubberband: {} {} {} {}\n",
            ixl + 1,
            iyb + 1,
            ixr + 1,
            iyt + 1
        ));
    }
}

/// `zapRubberbandCoords` (`xzap.cpp:231`); return coordinates of first rubber
/// band, 1 if any and 0 if none.
pub fn zap_rubberband_coords(
    rb_x0: &mut f32,
    rb_x1: &mut f32,
    rb_y0: &mut f32,
    rb_y1: &mut f32,
) -> i32 {
    let obj_list = with_boundary(|n| n.imod_dialog_manager_window_list(ZAP_WINDOW_TYPE));

    for &zap in &obj_list {
        let zap = unsafe { &mut *zap };
        if zap.rubberband != 0 {
            *rb_x0 = zap.rb_image_x0;
            *rb_x1 = zap.rb_image_x1;
            *rb_y0 = zap.rb_image_y0;
            *rb_y1 = zap.rb_image_y1;
            return 1;
        }
    }
    0
}

/// `zapSetImageOrBandCenter` (`xzap.cpp:256`); reposition image center or
/// rubber band to the given absolute position, or by the given increments.
pub fn zap_set_image_or_band_center(mut imx: f32, mut imy: f32, incremental: bool) {
    let zap = get_top_zap_window(false, false, ZAP_WINDOW_TYPE, None);
    if zap.is_null() {
        return;
    }
    let zap = unsafe { &mut *zap };
    if zap.rubberband != 0 {
        // Rubberband: get desired shift if not incremental, try to do it
        if !incremental {
            imx -= ((zap.rb_image_x1 + zap.rb_image_x0) as f64 / 2.) as f32;
            imy -= ((zap.rb_image_y1 + zap.rb_image_y0) as f64 / 2.) as f32;
        }
        zap.shift_rubberband(imx, imy);

        // And center image on rubberband
        zap.xtrans = ((unsafe { (*zap.vi).xsize } as f64 / 2.
            - (zap.rb_image_x1 + zap.rb_image_x0) as f64 / 2.)
            + 0.5)
            .floor() as i32;
        zap.ytrans = ((unsafe { (*zap.vi).ysize } as f64 / 2.
            - (zap.rb_image_y1 + zap.rb_image_y0) as f64 / 2.)
            + 0.5)
            .floor() as i32;
        zap.band_changed = 1;
    } else {
        // Not rubberband: just adjust or set the translations, which will be
        // fixed when the draw is done
        if incremental {
            zap.xtrans -= (imx as f64 + 0.5).floor() as i32;
            zap.ytrans -= (imy as f64 + 0.5).floor() as i32;
        } else {
            zap.xtrans =
                ((unsafe { (*zap.vi).xsize } as f64 / 2. - imx as f64) + 0.5).floor() as i32;
            zap.ytrans =
                ((unsafe { (*zap.vi).ysize } as f64 / 2. - imy as f64) + 0.5).floor() as i32;
        }
    }
    zap.record_subarea = 1;
    zap.draw();
}

/// `zapPixelViewState` (`xzap.cpp:296`); the pixel view window has opened or
/// closed, set mouse tracking for all zaps.
pub fn zap_pixel_view_state(state: bool) {
    S_PIXEL_VIEW_OPEN.set(state);
    zap_set_mouse_tracking();
}

/// `zapSetMouseTracking` (`xzap.cpp:306`); the state of externally requested
/// tracking has changed somehow, set tracking for all zaps.
pub fn zap_set_mouse_tracking() {
    let obj_list = with_boundary(|n| n.imod_dialog_manager_window_list(ZAP_WINDOW_TYPE));

    for &zap in &obj_list {
        unsafe { (*zap).set_mouse_tracking() };
    }
}

/// `getTopZapMouse` (`xzap.cpp:322`); return the image coordinates of the
/// mouse in the top Zap.
pub fn get_top_zap_mouse(image_pt: &mut Ipoint) -> i32 {
    let mut iz = 0;
    let zap = get_top_zap_window(false, false, ZAP_WINDOW_TYPE, None);
    if zap.is_null() {
        return 1;
    }
    let zap = unsafe { &mut *zap };
    let (px, py) = with_boundary(|n| n.gfx_map_from_global_cursor_pos());
    let mx = (px as f64 * zap.device_pixel_ratio as f64 + 0.5).floor() as i32;
    let my = (py as f64 * zap.device_pixel_ratio as f64 + 0.5).floor() as i32;
    let (mut x, mut y) = (0., 0.);
    zap.getixy(mx, my, &mut x, &mut y, &mut iz);
    image_pt.x = x;
    image_pt.y = y;
    image_pt.z = iz as f32;
    0
}

/// `zapSubsetLimits` (`xzap.cpp:338`); return the subset limits from the
/// active window.
pub fn zap_subset_limits(
    vi: *mut ImodView,
    ix_start: &mut i32,
    iy_start: &mut i32,
    nx_use: &mut i32,
    ny_use: &mut i32,
) -> i32 {
    if S_NUM_ZAP_WINDOWS.get() <= 0
        || S_SUB_START_X.get() >= S_SUB_END_X.get()
        || S_SUB_START_Y.get() >= S_SUB_END_Y.get()
        || S_SUB_END_X.get() >= unsafe { (*vi).xsize }
        || S_SUB_END_Y.get() >= unsafe { (*vi).ysize }
    {
        return 1;
    }
    *ix_start = S_SUB_START_X.get();
    *nx_use = S_SUB_END_X.get() + 1 - S_SUB_START_X.get();
    *iy_start = S_SUB_START_Y.get();
    *ny_use = S_SUB_END_Y.get() + 1 - S_SUB_START_Y.get();
    0
}

/// `zapSetNextOpenHQstate` (`xzap.cpp:351`); set a flag for the HQ state of
/// the next opened window.
pub fn zap_set_next_open_hq_state(state: i32) {
    S_NEXT_OPEN_HQ_STATE.set(state);
}

/// Static `zapDraw_cb` (`xzap.cpp:359`); the external draw command from the
/// controller.
pub fn zap_draw_cb(vi: &mut ImodView, client: usize, drawflag: i32) {
    let vi: *mut ImodView = vi;
    let zap = client as *mut ZapFuncs;
    let snaptype = with_boundary(|n| n.imc_get_snapshot(unsafe { (*zap).vi }));
    let doing_snap = snaptype != 0
        && unsafe { (*(*zap).vi).zmovie } != 0
        && unsafe { (*zap).movie_snap_count } != 0
        && with_boundary(|n| n.imc_get_starter_id()) == unsafe { (*zap).ctrl };

    if imod_debug('z') {
        imod_print_stderr(&format!("Zap Draw  flags {drawflag:x}\n"));
    }

    if zap.is_null() {
        return;
    }
    let zap = unsafe { &mut *zap };
    if zap.popup == 0 || zap.ginit == 0 {
        imod_trace('z', "Canceled, not ready yet");
        return;
    }

    zap.set_cursor(unsafe { (*(*vi).imod).mousemode }, false);

    if drawflag & IMOD_DRAW_COLORMAP != 0 {
        with_boundary(|n| n.gfx_set_colormap());
        return;
    }

    // If the rubberband is enabled and a flip is happening, turn off the
    // rubberband.
    if (zap.rubberband != 0 || zap.starting_band != 0)
        && zap.tool_max_z != unsafe { (*zap.vi).zsize }
    {
        zap.toggle_rubberband(false);
    }
    // drawTools();

    if drawflag != 0 {
        if drawflag & IMOD_DRAW_SLICE != 0 {
            zap.showslice = 1;
        }

        if drawflag & IMOD_DRAW_IMAGE != 0 {
            zap.flush_image();
        }

        if drawflag & IMOD_DRAW_ACTIVE == 0 && drawflag & IMOD_DRAW_NOSYNC == 0 {
            zap.sync_image(false);
        }

        // Have to defer swap and read from back buffer to prevent overlaid
        // image on some systems (Quadro, maybe others).
        if doing_snap {
            if with_boundary(|n| n.app_doublebuffer()) {
                with_boundary(|n| n.gfx_set_buffer_swap_auto(false));
                let front = APP
                    .lock()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|a| a.new_qt_open_gl != 0);
                with_boundary(|n| n.gl_read_buffer(front));
            }
            let vi_ptr = zap.vi;
            with_boundary(|n| n.util_pre_snap_changes(vi_ptr));
        }
        zap.draw();
        with_boundary(|n| n.imod_info_input());

        /* DNM 3/8/01: add autosnapshot when movieing */
        // 3/8/07: make it take montages too
        if doing_snap {
            if with_boundary(|n| n.imc_get_snap_montage(true)) {
                zap.montage_snapshot(snaptype);
            } else {
                let mut limarr = [0; 4];
                let limits = zap.set_snapshot_limits(&mut limarr);
                let name = if zap.num_xpanels != 0 {
                    "multiz"
                } else {
                    "zap"
                };
                with_boundary(|n| n.b3d_key_snapshot(name, snaptype - 1, snaptype % 2, limits));
            }

            // Restore double buffering
            if with_boundary(|n| n.app_doublebuffer()) {
                with_boundary(|n| n.gfx_swap_buffers());
                with_boundary(|n| n.gfx_set_buffer_swap_auto(true));
            }
            let vi_ptr = zap.vi;
            with_boundary(|n| n.util_restore_snap_changes(vi_ptr));

            /* When count expires, stop movie */
            zap.movie_snap_count -= 1;
            if zap.movie_snap_count == 0 {
                unsafe { (*zap.vi).zmovie = 0 };
                with_boundary(|n| n.b3d_set_movie_snapping(false));
                if with_boundary(|n| n.imc_get_snap_montage(true)) {
                    zap.draw();
                }
            }
        }

        // If there is only one zap window, set flag to record the subarea
        if with_boundary(|n| n.imod_dialog_manager_window_count(ZAP_WINDOW_TYPE)) == 1
            && zap.num_xpanels == 0
        {
            zap.record_subarea = 1;
        }
    }
}

/// Static `zapClose_cb` (`xzap.cpp:460`); receives the close signal back from
/// the controller, tells the window to close, and sets the closing flag.
pub fn zap_close_cb(vi: &mut ImodView, client: usize, junk: i32) {
    let zap = unsafe { &mut *(client as *mut ZapFuncs) };
    if imod_debug('z') {
        imod_print_stderr("Sending zap window close.\n");
    }
    zap.popup = 0;
    with_boundary(|n| n.zap_window_close());
}

/// Static `zapKey_cb` (`xzap.cpp:472`); external key is passed on.
pub fn zap_key_cb(vi: &mut ImodView, client: usize, released: i32, e: &KeyEvent) {
    let zap = unsafe { &mut *(client as *mut ZapFuncs) };
    if e.keypad && (e.qt_key == KEY_INSERT || e.qt_key == KEY_0) {
        return;
    }
    if released != 0 {
        zap.key_release(e);
    } else {
        zap.key_input(e);
    }
}

impl ZapFuncs {
    /// `ZapFuncs::ZapFuncs` (`xzap.cpp:487`).
    ///
    /// The C++ object is allocated before the constructor body runs and the
    /// body takes `this` (for `ivwNewControl` and `imodDialogManager.add`),
    /// so the box is made first and the body then runs against it.  Members
    /// the source leaves uninitialised in a branch it does not take —
    /// `restrainSize`, `toolHeight`, `newWidth`, `newHeight`, `xleft`,
    /// `ytop` — are zeroed here; their C values are indeterminate and not
    /// reproducible (see CLAUDE.md on uninitialised memory).
    pub fn new(vi: *mut ImodView, wintype: i32) -> Box<Self> {
        let mut zap = Box::new(Self {
            vi: ptr::null_mut(),
            qt_window: ptr::null_mut(),
            gfx_created: false,
            winx: 0,
            winy: 0,
            num_xpanels: 0,
            num_ypanels: 0,
            panel_zstep: 0,
            draw_in_center: 0,
            draw_in_others: 0,
            section: 0,
            rubberband: 0,
            rb_image_x0: 0.,
            rb_image_x1: 0.,
            rb_image_y0: 0.,
            rb_image_y1: 0.,
            rb_mouse_x0: 0,
            rb_mouse_x1: 0,
            rb_mouse_y0: 0,
            rb_mouse_y1: 0,
            band_changed: 0,
            ctrl: 0,
            ginit: 0,
            images: Vec::new(),
            movie_snap_count: 0,
            popup: 0,
            record_subarea: 0,
            showslice: 0,
            starting_band: 0,
            tool_max_z: 0,
            xtrans: 0,
            ytrans: 0,
            ztrans: 0,
            zoom: 0.,
            new_screen_zoom: 0.,
            lock: 0,
            time_lock: 0,
            scale_bar_size: 0.,
            lasso_on: false,
            drawing_lasso: false,
            lasso_obj_num: 0,
            arrow_on: false,
            drawing_arrow: false,
            arrow_tail: Vec::new(),
            arrow_head: Vec::new(),
            label_font: false,
            label_painter: false,
            device_pixel_ratio: 0.,
            xborder: 0,
            yborder: 0,
            xstart: 0,
            ystart: 0,
            xpos_start: 0,
            ypos_start: 0,
            xlast_start: 0,
            ylast_start: 0,
            xlast_size: 0,
            ylast_size: 0,
            last_status: 0,
            xdrawsize: 0,
            ydrawsize: 0,
            lmx: 0,
            lmy: 0,
            tess_cont: ptr::null_mut(),
            tess_max_points: 0,
            nest_cont_map: Vec::new(),
            conts_at_cur_z: None,
            num_nests: 0,
            nest_ind: Vec::new(),
            nests: Vec::new(),
            hqgfx: 0,
            hide: 0,
            hqgfxsave: 0,
            draw_current_only: 0,
            last_hq_draw_time: 0,
            shifting_cont: 0,
            xform_center: Ipoint::default(),
            xform_fixed_pt: Ipoint::default(),
            center_defined: 0,
            center_marked: 0,
            fixed_pt_defined: 0,
            shift_registered: 0,
            shift_obj_num: 0,
            drag_add_count: 0,
            drag_add_index: Iindex {
                object: 0,
                contour: 0,
                point: 0,
            },
            drag_add_end: 0,
            drew_extra_cursor: false,
            xzoom: 0.,
            data: ptr::null_mut(),
            image: false,
            num_images: 0,
            section_step: 0,
            time: 0,
            overlay: 0,
            keepcentered: 0,
            mousemode: 0,
            last_shape: 0,
            tool_section: 0,
            tool_zoom: 0.,
            tool_time: 0,
            tool_size_x: 0,
            tool_size_y: 0,
            insertmode: 0,
            showed_slice: 0,
            doing_draw: false,
            doing_montage: false,
            panel_xborder: 0,
            panel_yborder: 0,
            panel_gutter: 0,
            panel_xsize: 0,
            panel_ysize: 0,
            toolstart: 0,
            screen_changed: false,
            defer_screen_zoom_change: false,
            screen_resize_time: [QTime::now(), QTime::now()],
            last_xsize_change: 0.,
            twod: 0,
        });
        let this = &raw mut *zap;

        // `QRect` values travel as [x, y, width, height].
        let mut old_geom = [0_i32; 4];
        let mut infoRect = [0_i32; 4];
        let (mut need_winx, mut need_winy) = (0, 0);
        let (mut max_winx, mut max_win_imx) = (0, 0);
        let (mut max_winy, mut max_win_imy) = (0, 0);
        let mut i;
        let (mut new_width, mut new_height, mut xleft, mut ytop) = (0, 0, 0, 0);
        let mut tool_height = 0;
        let mut usable_top = 0;
        let mut usable_left = 0;
        let (mut info_left, mut info_top, mut info_width, mut usable_right);
        let (mut new_out_width, mut new_out_height);
        let mut pos_dpr = 0.;
        let mut restrain_size = false;
        let mut new_zoom;
        let extra_top_bot = 20;

        zap.vi = vi;
        zap.ctrl = 0;
        zap.xtrans = 0;
        zap.ytrans = 0;
        zap.ztrans = 0;
        if S_NEXT_OPEN_HQ_STATE.get() >= 0 {
            zap.hqgfx = i32::from(S_NEXT_OPEN_HQ_STATE.get() != 0);
        } else if unsafe { (*vi).colormap_image } == 0 {
            zap.hqgfx = i32::from(with_boundary(|n| n.prefs_start_in_hq()));
        }
        S_NEXT_OPEN_HQ_STATE.set(-1);
        zap.last_hq_draw_time = 0;
        zap.hide = 0;
        zap.popup = 0;
        zap.data = ptr::null_mut();
        zap.image = false;
        zap.tess_cont = ptr::null_mut();
        zap.label_font = false;
        zap.label_painter = false;
        zap.nest_ind = Vec::new();
        zap.num_nests = 0;
        zap.nest_cont_map = Vec::new();
        zap.nests = Vec::new();
        zap.ginit = 0;
        zap.lock = 0;
        zap.keepcentered = 0;
        zap.insertmode = 0;
        zap.toolstart = 0;
        zap.showslice = 0;
        zap.showed_slice = 0;
        zap.time_lock = 0;
        zap.tool_section = -1;
        zap.tool_max_z = unsafe { (*vi).zsize };
        zap.tool_zoom = -1.0;
        zap.tool_time = 0;
        zap.tool_size_x = 0;
        zap.tool_size_y = 0;
        zap.twod = i32::from(unsafe { (*vi).dim } & 4 == 0);
        zap.scale_bar_size = -1.;
        zap.section_step = 0;
        zap.time = 0;
        zap.overlay = 0;
        zap.mousemode = 0;
        zap.last_shape = -1;
        zap.rubberband = 0;
        zap.starting_band = 0;
        zap.shifting_cont = 0;
        zap.band_changed = 0;
        zap.lasso_on = false;
        zap.drawing_lasso = false;
        zap.arrow_on = false;
        zap.drawing_arrow = false;
        zap.doing_draw = false;
        zap.doing_montage = false;
        zap.shift_registered = 0;
        zap.center_marked = 0;
        zap.xform_fixed_pt.x = 0.;
        zap.xform_fixed_pt.y = 0.;
        zap.movie_snap_count = 0;
        zap.draw_current_only = 0;
        zap.drag_add_count = 0;
        zap.drew_extra_cursor = false;
        zap.num_xpanels = if wintype != 0 { 5 } else { 0 };
        zap.num_ypanels = 1;
        zap.panel_zstep = 1;
        zap.draw_in_center = 1;
        zap.draw_in_others = 1;
        zap.panel_gutter = 8;
        zap.qt_window = ptr::null_mut();
        zap.last_xsize_change = 1.;
        zap.screen_changed = false;
        zap.defer_screen_zoom_change = false;
        zap.winx = 1;
        zap.device_pixel_ratio = 0.;
        zap.new_screen_zoom = 0.;

        if wintype != 0 {
            zap.images = vec![false; (MULTIZ_MAX_PANELS * MULTIZ_MAX_PANELS) as usize];
            zap.num_images = 0;
            let (mut nx, mut ny, mut zs, mut dc, mut do_) = (
                zap.num_xpanels,
                zap.num_ypanels,
                zap.panel_zstep,
                zap.draw_in_center,
                zap.draw_in_others,
            );
            old_geom = with_boundary(|n| {
                n.prefs_get_multi_z_params(&mut nx, &mut ny, &mut zs, &mut dc, &mut do_)
            });
            zap.num_xpanels = nx;
            zap.num_ypanels = ny;
            zap.panel_zstep = zs;
            zap.draw_in_center = dc;
            zap.draw_in_others = do_;
        }

        let str = with_boundary(|n| {
            let num_times = unsafe { (*vi).num_times };
            let mut labels = Vec::new();
            for t in 0..=num_times {
                let label =
                    unsafe { crate::imod::three_dmod::imodview::ivw_get_time_index_label(vi, t) };
                labels.push(if label.is_null() {
                    String::new()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(label) }
                        .to_string_lossy()
                        .into_owned()
                });
            }
            let _ = n;
            util_get_longest_time_string(num_times, &labels)
        });
        zap.qt_window = with_boundary(|n| n.new_zap_window(this, &str, wintype != 0));
        if zap.qt_window.is_null() {
            zap.images = Vec::new();
            wprint("\u{7}Error opening zap window.\n");
            return zap;
        }
        zap.gfx_created = true;
        if imod_debug('z') {
            imod_puts("Got a zap window");
        }
        let hqgfx = zap.hqgfx;
        with_boundary(|n| n.zap_window_set_toggle_state(ZAP_TOGGLE_RESOL, hqgfx));

        if APP.lock().unwrap().as_ref().is_some_and(|a| a.rgba == 0) {
            with_boundary(|n| n.gfx_set_colormap());
        }

        let title = imod_caption(
            if wintype != 0 {
                "3dmod Multi-Z Window"
            } else {
                "3dmod ZaP Window"
            },
            None,
        );
        with_boundary(|n| n.zap_window_set_window_title(0, &title));

        let bar_title = imod_caption("ZaP Toolbar", None);
        with_boundary(|n| n.zap_window_set_window_title(1, &bar_title));
        if with_boundary(|n| n.zap_window_toolbar_exists(2)) {
            let t = imod_caption("Time Toolbar", None);
            with_boundary(|n| n.zap_window_set_window_title(2, &t));
        }
        if with_boundary(|n| n.zap_window_toolbar_exists(3)) {
            let t = imod_caption("Multi-Z Toolbar", None);
            with_boundary(|n| n.zap_window_set_window_title(3, &t));
        }

        zap.ctrl = crate::imod::three_dmod::control::ivw_new_control(
            unsafe { &mut *vi },
            zap_draw_cb,
            zap_close_cb,
            Some(zap_key_cb),
            this as usize,
        );
        let ctrl = zap.ctrl;
        with_boundary(|n| {
            n.imod_dialog_manager_add(
                this,
                if wintype != 0 {
                    MULTIZ_WINDOW_TYPE
                } else {
                    ZAP_WINDOW_TYPE
                },
                ctrl,
            )
        });

        if wintype == 0 {
            old_geom = with_boundary(|n| n.prefs_get_zap_geometry());

            // Get the max size and DPR of the screen the window is supposed
            // to be on, otherwise just use current widget position
            let is_windows = APP
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|a| a.is_windows != 0);
            if old_geom[2] != 0 && is_windows {
                let (a, b, c) = with_boundary(|n| {
                    n.dia_max_win_size_at_pos(
                        old_geom[0] + old_geom[2] / 2,
                        old_geom[1] + old_geom[3] / 2,
                    )
                });
                max_winx = a;
                max_winy = b;
                pos_dpr = c;
            } else {
                let (a, b) = with_boundary(|n| n.dia_maximum_window_size());
                max_winx = a;
                max_winy = b;
            }
            let (wx, wy) = with_boundary(|n| n.zap_window_pos());
            imod_trace(
                'z',
                &format!("maxWinxy {max_winx} {max_winy} dpr {pos_dpr}  win l {wx} t {wy}"),
            );
            with_boundary(|n| n.zap_window_set_size_text(max_winx, max_winy));
            usable_right = max_winx;
        } else {
            usable_right = 0;
        }

        /* 1/28/03: this call is needed to get the toolbar size hint right */
        with_boundary(|n| n.imod_info_input());
        if with_boundary(|n| n.app_dev_pix_varies()) {
            // This gets the window onto the right screen, and if there was
            // not an old position (for Windows only), revise the max size
            with_boundary(|n| n.zap_window_show());
            with_boundary(|n| n.imod_info_input());
            let (wx, wy) = with_boundary(|n| n.zap_window_pos());
            imod_trace('z', &format!("after show win l {wx} t {wy}"));
            let is_windows = APP
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|a| a.is_windows != 0);
            if pos_dpr == 0. && is_windows {
                let (ww, wh) = with_boundary(|n| n.zap_window_size());
                let (a, b, c) =
                    with_boundary(|n| n.dia_max_win_size_at_pos(wx + ww / 2, wy + wh / 2));
                max_winx = a;
                max_winy = b;
                pos_dpr = c;
            }
        }
        let tool_size = with_boundary(|n| n.zap_window_toolbar_size_hint(1));
        let mut tool_size2 = (0, 0);
        let mut tool_size3 = (0, 0);
        if with_boundary(|n| n.zap_window_toolbar_exists(2)) {
            tool_size2 = with_boundary(|n| n.zap_window_toolbar_size_hint(2));
        }
        if wintype != 0 {
            tool_size3 = with_boundary(|n| n.zap_window_toolbar_size_hint(3));
        }
        tool_height = tool_size.1;
        if imod_debug('z') {
            let (ww, wh) = with_boundary(|n| n.zap_window_size());
            let (gw, gh) = with_boundary(|n| n.gfx_size());
            imod_print_stderr(&format!(
                "Toolsize {} {} win {} gfx {}\n",
                tool_size.0, tool_size.1, wh, gh
            ));
            let _ = (ww, gw);
        }
        zap.device_pixel_ratio = with_boundary(|n| n.util_initialize_screen_change());
        let pos = with_boundary(|n| n.ivw_restorable_geometry());
        imod_trace(
            'z',
            &format!(
                "mdpr {:.2}, posdpr {:.2} pos {} {}\n",
                zap.device_pixel_ratio, pos_dpr, pos[0], pos[1]
            ),
        );

        // Replace the DPR with the correct one for the screen before zoom
        let is_windows = APP
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|a| a.is_windows != 0);
        if pos_dpr > 0. && is_windows {
            zap.device_pixel_ratio = pos_dpr;
        }
        zap.zoom = util_unit_zoom_for_device_scaling(zap.device_pixel_ratio);

        if old_geom[2] == 0 {
            if wintype == 0 {
                // If no old geometry, adjust zoom if necessary to fit image
                let (ul, ut) = with_boundary(|n| n.dia_minimum_window_pos());
                usable_left = ul;
                usable_top = ut;
                with_boundary(|n| {
                    n.dia_limit_window_pos(100, 100, &mut usable_left, &mut usable_top)
                });
                usable_right += usable_left;
                infoRect = with_boundary(|n| n.info_win_frame_geometry());
                let dev_pix_varies = with_boundary(|n| n.app_dev_pix_varies());
                if max_winx > S_MIN_X_FOR_RESTRAINING.get() && !dev_pix_varies {
                    restrain_size = true;
                    max_winx -= infoRect[2];
                }
                max_win_imx =
                    (max_winx as f64 * zap.device_pixel_ratio as f64 + 0.5).floor() as i32;
                max_win_imy = ((max_winy - tool_height) as f64 * zap.device_pixel_ratio as f64
                    + 0.5)
                    .floor() as i32;
                i = 0;
                while i < 2 {
                    zap.zoom = imod_initial_zoom();
                    if zap.zoom == 0. {
                        zap.zoom = util_unit_zoom_for_device_scaling(zap.device_pixel_ratio);
                        while zap.zoom * unsafe { (*vi).xsize } as f32 > 1.1 * max_win_imx as f32
                            || zap.zoom * unsafe { (*vi).ysize } as f32 > 1.1 * max_win_imy as f32
                        {
                            new_zoom =
                                with_boundary(|n| n.b3d_step_pixel_zoom(zap.zoom as f64, -1));
                            if (new_zoom - zap.zoom as f64).abs() < 0.0001 {
                                break;
                            }
                            if !unsafe { (*zap.vi).pyr_cache }.is_null()
                                && with_boundary(|n| {
                                    n.pyr_cache_zoom_requires_big_load(
                                        vi,
                                        new_zoom,
                                        max_win_imx,
                                        max_win_imy,
                                    )
                                })
                            {
                                break;
                            }
                            zap.zoom = new_zoom as f32;
                        }
                    }

                    need_winx = ((zap.zoom * unsafe { (*vi).xsize } as f32) as f64
                        / zap.device_pixel_ratio as f64
                        + 0.5)
                        .floor() as i32;
                    if restrain_size {
                        need_winx = need_winx.min(max_winx);
                    }

                    // If Window is narrower than two toolbars, set up to
                    // stack the toolbars and increase the tool height
                    if i == 0 && need_winx < tool_size.0 + tool_size2.0 {
                        with_boundary(|n| n.zap_window_insert_toolbar_break(2));
                        tool_height += tool_size2.1;
                    }
                    i += 1;
                }

                need_winy = ((zap.zoom * unsafe { (*vi).ysize } as f32) as f64
                    / zap.device_pixel_ratio as f64
                    + 0.5)
                    .floor() as i32
                    + tool_height;
                with_boundary(|n| n.dia_limit_window_size(&mut need_winx, &mut need_winy));

                // Make the width big enough for the toolbar, and add the
                // difference between the window and image widget heights
                new_width = if tool_size.0 > need_winx {
                    tool_size.0
                } else {
                    need_winx
                };
                new_height = need_winy;
            } else {
                // For multiZ, just make it big enough for two bars, then if
                // necessary insert break and make it taller for the panel bar
                new_width = 640.max(tool_size.0 + tool_size2.0);
                new_height = 170;
                if new_width < tool_size.0 + tool_size2.0 + tool_size3.0 {
                    new_height += tool_size3.1;
                    with_boundary(|n| n.zap_window_insert_toolbar_break(3));
                }
                if S_NEXT_MULTI_Z_X_SIZE.get() > 0 && S_NEXT_MULTI_Z_Y_SIZE.get() > 0 {
                    new_width = (S_NEXT_MULTI_Z_X_SIZE.get() as f64 / zap.device_pixel_ratio as f64
                        + 0.5)
                        .floor() as i32;
                    new_height =
                        (S_NEXT_MULTI_Z_Y_SIZE.get() as f64 / zap.device_pixel_ratio as f64 + 0.5)
                            .floor() as i32;
                    if with_boundary(|n| n.zap_window_toolbar_exists(3))
                        && with_boundary(|n| n.zap_window_toolbar_exists(1))
                    {
                        new_height += with_boundary(|n| n.zap_window_toolbar_height(3))
                            + with_boundary(|n| n.zap_window_toolbar_height(1));
                    }
                    with_boundary(|n| n.dia_limit_window_size(&mut new_width, &mut new_height));
                    S_NEXT_MULTI_Z_X_SIZE.set(0);
                }
                if imod_initial_zoom() != 0. {
                    zap.zoom = imod_initial_zoom();
                }
            }

            // Would have to show first to get a good frame geometry but we
            // can add the borders from the info window to get the outside
            let pos = with_boundary(|n| n.zap_window_frame_geometry());
            xleft = pos[0];
            ytop = pos[1];
            let inside = with_boundary(|n| n.info_win_geometry());
            new_out_width = new_width + infoRect[2] - inside[2];
            new_out_height = new_height + infoRect[3] - inside[3];

            with_boundary(|n| n.dia_limit_window_pos(new_width, new_height, &mut xleft, &mut ytop));
            if imod_debug('z') {
                let (ww, wh) = with_boundary(|n| n.zap_window_size());
                let (gw, gh) = with_boundary(|n| n.gfx_size());
                imod_print_stderr(&format!(
                    "Sizes: zap {ww} {wh}, toolbar {} {}, GL {gw} {gh}: resize {new_width} {new_height}\n",
                    tool_size.0, tool_size.1
                ));
            }

            if wintype == 0 {
                info_top = infoRect[1];
                info_left = infoRect[0];
                info_width = infoRect[2];

                // For first zap, move Info to nearest corner with a bit more
                // allowance top/bottom
                if restrain_size && S_FIRST_ZAP_OPENING.get() {
                    if info_top - usable_top
                        > (usable_top + max_winy) - (infoRect[1] + infoRect[3] - 1)
                    {
                        info_top = (usable_top + max_winy) - (infoRect[3] + extra_top_bot);
                    } else {
                        info_top = usable_top + extra_top_bot;
                    }
                    if info_left - usable_left > usable_right - (info_left + info_width) {
                        info_left = usable_right - info_width;
                    } else {
                        info_left = usable_left;
                    }
                    with_boundary(|n| n.info_win_move(info_left, info_top));
                }

                // Regardless, now move zap to one side or another if it can
                // uncover some info win
                let dev_pix_varies = with_boundary(|n| n.app_dev_pix_varies());
                if !dev_pix_varies
                    && ((info_left - usable_left) + 20 < usable_right - new_out_width
                        || (info_left - usable_left) + info_width - 20 > new_out_width)
                {
                    if (info_left - usable_left) + 20 < usable_right - new_out_width {
                        xleft = (usable_right - new_out_width).min(info_left + info_width);
                    } else {
                        xleft = usable_left.max(info_left - new_out_width);
                    }

                    // And move it vertically to be adjacent - its original
                    // position may be meaningless
                    if info_top - usable_top
                        > (usable_top + max_winy) - (infoRect[1] + infoRect[3] - 1)
                    {
                        ytop = usable_top.max(info_top + infoRect[3] - new_out_height);
                    } else {
                        ytop = (usable_top + max_winy - new_out_height).min(info_top);
                    }
                }
            }
        } else {
            // Existing geometry - better fit it to current screen
            xleft = old_geom[0];
            ytop = old_geom[1];
            new_width = old_geom[2];
            new_height = old_geom[3];
            if wintype != 0 && S_NEXT_MULTI_Z_X_SIZE.get() > 0 && S_NEXT_MULTI_Z_Y_SIZE.get() > 0 {
                new_width = (S_NEXT_MULTI_Z_X_SIZE.get() as f64 / zap.device_pixel_ratio as f64
                    + 0.5)
                    .floor() as i32;
                new_height = (S_NEXT_MULTI_Z_Y_SIZE.get() as f64 / zap.device_pixel_ratio as f64
                    + 0.5)
                    .floor() as i32;
                if with_boundary(|n| n.zap_window_toolbar_exists(3))
                    && with_boundary(|n| n.zap_window_toolbar_exists(1))
                {
                    new_height += with_boundary(|n| n.zap_window_toolbar_height(3))
                        + with_boundary(|n| n.zap_window_toolbar_height(1));
                }
                S_NEXT_MULTI_Z_X_SIZE.set(0);
            }
            let dev_pix_varies = with_boundary(|n| n.app_dev_pix_varies());
            imod_trace(
                'z',
                &format!(
                    "oldGeom l {xleft} t {ytop}  w {new_width} h {new_height}  dpi vary {}",
                    i32::from(dev_pix_varies)
                ),
            );
            with_boundary(|n| {
                n.dia_limit_win_size_at_pos(
                    xleft + new_width / 2,
                    ytop + new_height / 2,
                    &mut new_width,
                    &mut new_height,
                )
            });
            imod_trace('z', &format!("limit size w {new_width} h {new_height}"));
            with_boundary(|n| n.dia_limit_window_pos(new_width, new_height, &mut xleft, &mut ytop));
            imod_trace('z', &format!("limit pos w {xleft} h {ytop}"));

            // Adjust the tool height: see if time bar fits on line and insert
            // break if not and add to height
            let mut tool_base = tool_size.0;
            if with_boundary(|n| n.zap_window_toolbar_exists(2)) {
                if new_width < tool_base + tool_size2.0 {
                    with_boundary(|n| n.zap_window_insert_toolbar_break(2));
                    tool_base = tool_size2.0;
                    tool_height += tool_size2.1;
                } else {
                    tool_base += tool_size2.0;
                }
            }

            // Then see if panel bar fits on line, insert break if not
            if wintype != 0 && new_width < tool_base + tool_size3.0 {
                with_boundary(|n| n.zap_window_insert_toolbar_break(3));
                tool_height += tool_size3.1;
            }
            if wintype != 0 && imod_initial_zoom() != 0. {
                zap.zoom = imod_initial_zoom();
            }

            if wintype == 0 {
                zap.zoom = imod_initial_zoom();
                if zap.zoom == 0. {
                    zap.zoom = 1.;

                    need_winx =
                        (new_width as f64 * zap.device_pixel_ratio as f64 + 0.5).floor() as i32;
                    need_winy = ((new_height - tool_height) as f64 * zap.device_pixel_ratio as f64
                        + 0.5)
                        .floor() as i32;

                    // If images are too big, zoom down until they almost fit
                    // If images are too small, start big and find first that
                    // fits.  Apply same overflow criterion so that reopened
                    // windows behave like when they were first opened
                    if unsafe { (*vi).xsize } < need_winx && unsafe { (*vi).ysize } < need_winy {
                        zap.zoom = ((2. * need_winx as f64) / unsafe { (*vi).xsize } as f64) as f32;
                    }

                    while zap.zoom * unsafe { (*vi).xsize } as f32 > 1.1 * need_winx as f32
                        || zap.zoom * unsafe { (*vi).ysize } as f32 > 1.1 * need_winy as f32
                    {
                        new_zoom = with_boundary(|n| n.b3d_step_pixel_zoom(zap.zoom as f64, -1));
                        // This test goes into infinite loop in Windows -
                        // Intel, 6/22/04:  if (newZoom == mZoom)
                        if (new_zoom - zap.zoom as f64).abs() < 0.0001 {
                            break;
                        }
                        if !unsafe { (*zap.vi).pyr_cache }.is_null()
                            && with_boundary(|n| {
                                n.pyr_cache_zoom_requires_big_load(
                                    vi, new_zoom, need_winx, need_winy,
                                )
                            })
                        {
                            break;
                        }
                        zap.zoom = new_zoom as f32;
                    }
                }
            }
        }

        // 9/23/03: changed setGeometry to resize/move and this allowed
        // elimination of setting again on the first real draw
        with_boundary(|n| n.gfx_set_init_geometry(new_width, new_height, xleft, ytop));
        zap.xzoom = zap.zoom;

        // On Windows, if DPR varies, do an initial move to the minimum DPR
        // screen and set the initial zoom to set when screen change comes in
        let dev_pix_varies = with_boundary(|n| n.app_dev_pix_varies());
        let is_windows = APP
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|a| a.is_windows != 0);
        if dev_pix_varies
            && is_windows
            && zap.device_pixel_ratio > with_boundary(|n| n.app_min_dev_pix_ratio())
        {
            zap.new_screen_zoom = zap.zoom;
            let (l, t) = (
                with_boundary(|n| n.app_min_dpr_left_pos()),
                with_boundary(|n| n.app_min_dpr_top_pos()),
            );
            with_boundary(|n| n.zap_window_move(l, t));
            with_boundary(|n| n.imod_info_input());
            with_boundary(|n| n.zap_window_show());
        }

        // Move and resize and show and resize and move!
        with_boundary(|n| n.zap_window_resize(new_width, new_height));
        with_boundary(|n| n.zap_window_move(xleft, ytop));
        zap.xzoom = zap.zoom;
        with_boundary(|n| n.zap_window_show());
        with_boundary(|n| n.imod_info_input());
        with_boundary(|n| n.zap_window_move(xleft, ytop));
        with_boundary(|n| n.zap_window_resize(new_width, new_height));
        zap.popup = 1;
        S_FIRST_ZAP_OPENING.set(false);
        zap.set_mouse_tracking();

        if imod_debug('z') {
            imod_puts("popup a zap dialog");
        }

        /* DNM: set cursor after window created so it has model mode cursor if
        an existing window put us in model mode */
        zap.set_cursor(unsafe { (*(*vi).imod).mousemode }, false);
        S_NUM_ZAP_WINDOWS.set(S_NUM_ZAP_WINDOWS.get() + 1);
        S_INSERT_TIME.with(|t| *t.borrow_mut() = QTime::now());
        if wintype != 0 && S_NEXT_MULTI_Z_NUM_X.get() > 0 {
            with_boundary(|n| n.imod_info_input());
            zap.set_multi_z_panels(S_NEXT_MULTI_Z_NUM_X.get(), S_NEXT_MULTI_Z_NUM_Y.get());
            S_NEXT_MULTI_Z_NUM_X.set(0);
        }
        zap
    }

    /// `ZapFuncs::help` (`xzap.cpp:927`).
    pub fn help(&mut self) {
        if self.num_xpanels != 0 {
            with_boundary(|n| n.imod_show_help_page("multizap.html#TOP"));
        } else {
            with_boundary(|n| n.imod_show_help_page("zap.html#TOP"));
        }
    }

    /// `ZapFuncs::closing` (`xzap.cpp:938`); receives a closing signal from
    /// the window.
    pub fn closing(&mut self) {
        if imod_debug('z') {
            imod_print_stderr("ZapClosing received.\n");
        }

        // Do cleanup
        self.popup = 0;
        crate::imod::three_dmod::control::ivw_remove_control(unsafe { &mut *self.vi }, self.ctrl);
        let this = &raw mut *self;
        with_boundary(|n| n.imod_dialog_manager_remove(this));
        if self.num_xpanels == 0 {
            S_NUM_ZAP_WINDOWS.set(S_NUM_ZAP_WINDOWS.get() - 1);
        }

        // What for?  flush any events that might refer to this zap
        with_boundary(|n| n.imod_info_input());

        if self.num_xpanels == 0 {
            with_boundary(|n| n.b3d_free_ci_image(-1));
        } else {
            if S_MAX_MULTI_Z_AREA.get() == 0 {
                let pos = with_boundary(|n| n.ivw_restorable_geometry());
                let (nx, ny, zs, dc, do_) = (
                    self.num_xpanels,
                    self.num_ypanels,
                    self.panel_zstep,
                    self.draw_in_center,
                    self.draw_in_others,
                );
                with_boundary(|n| n.prefs_record_multi_z_params(pos, nx, ny, zs, dc, do_));
            }
            for i in 0..self.num_images {
                with_boundary(|n| n.b3d_free_ci_image(i));
            }
            self.images = Vec::new();
        }
    }

    /// `ZapFuncs::startMovieCheckSnap` (`xzap.cpp:974`); start or stop movie
    /// and check for whether to start a movie snapshot sequence.
    pub fn start_movie_check_snap(&mut self, dir: i32) -> i32 {
        let vi = self.vi;

        with_boundary(|n| n.imod_movie_xyzt(vi, MOVIE_DEFAULT, MOVIE_DEFAULT, dir, MOVIE_DEFAULT));
        let ctrl = self.ctrl;
        with_boundary(|n| n.imc_set_starter_id(ctrl));

        self.movie_snap_count = 0;
        with_boundary(|n| n.b3d_set_movie_snapping(false));

        /* done if no movie, or if no snapshots are desired.  */
        if unsafe { (*vi).zmovie } == 0 || with_boundary(|n| n.imc_get_snapshot(vi)) == 0 {
            return 0;
        }

        /* Get start and end of loop, compute count */
        let (start, end) = with_boundary(|n| n.imc_get_start_end(vi, 2));
        self.movie_snap_count = (end - start) / with_boundary(|n| n.imc_get_increment(vi, 2)) + 1;
        if self.movie_snap_count < 1 {
            self.movie_snap_count = 1;
        }

        /* double count for normal mode, leave as is for one-way */
        if with_boundary(|n| n.imc_get_loop_mode(vi)) == 0 {
            self.movie_snap_count *= 2;
        }

        /* Set to start or end depending on which button was hit */
        if with_boundary(|n| n.imc_start_snap_here(vi)) == 0 {
            unsafe { (*vi).zmouse = if dir > 0 { start as f32 } else { end as f32 } };
        }

        // Inform autosnapshot not to check file numbers from 0
        with_boundary(|n| n.b3d_set_movie_snapping(true));

        /* draw - via imodDraw to get float done correctly */
        unsafe { (*vi).doing_snap_draw = 1 };
        with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_XYZ));
        unsafe { (*vi).doing_snap_draw = 0 };
        1
    }

    /// `ZapFuncs::syncImage` (`xzap.cpp:1020`); sync the pan position to the
    /// current model point.
    pub fn sync_image(&mut self, to_image_pt: bool) {
        let mut syncborder;
        let mut wposition;
        let mut wsize;
        let mut tripshift;
        let (mut trytrans, mut trydraws, mut tryborder, mut trystart);
        let border_min;
        let vi = self.vi;
        if self.lock == 0
            && ((unsafe { (*(*vi).imod).mousemode } == IMOD_MMODEL
                && unsafe { (*(*self.vi).imod).cindex.point } >= 0)
                || to_image_pt
                || (self.num_xpanels != 0 && self.keepcentered != 0))
        {
            border_min = if self.num_xpanels != 0 {
                BORDER_MIN_MULTIZ
            } else {
                BORDER_MIN
            };

            /* If the keepcentered flag is set, just do a shift to center */
            if self.keepcentered != 0 {
                tripshift = 1;
            } else {
                /* Otherwise, look at each axis independently.  First see if
                the position is within the borders for shifting */
                tripshift = 0;
                wsize = if self.num_xpanels != 0 {
                    self.panel_xsize
                } else {
                    self.winx
                };
                wposition = self.xpos(unsafe { (*vi).xmouse });
                syncborder = (wsize as f64 * BORDER_FRAC) as i32;
                syncborder = BORDER_MAX.min(border_min.max(syncborder));
                if wposition < syncborder || wposition > wsize - syncborder {
                    /* If close to a border, do an image offset computation to
                    see if the display would actually get moved if this axis
                    were centered on point */
                    trytrans = ((unsafe { (*vi).xsize } as f32 * 0.5f32) - unsafe { (*vi).xmouse }
                        + 0.5f32) as i32;
                    trydraws = self.xdrawsize;
                    tryborder = self.xborder;
                    trystart = self.xstart;
                    b3d_set_image_offset(
                        wsize,
                        unsafe { (*vi).xsize },
                        self.zoom as f64,
                        &mut trydraws,
                        &mut trytrans,
                        &mut tryborder,
                        &mut trystart,
                        1,
                    );
                    /* Can't use xtrans for a test, need to use the other two
                    values to see if change in display would occur */
                    if tryborder != self.xborder || trystart != self.xstart {
                        tripshift += 1;
                    }
                }

                /* Same for Y axis */
                wsize = if self.num_xpanels != 0 {
                    self.panel_ysize
                } else {
                    self.winy
                };
                wposition = self.ypos(unsafe { (*vi).ymouse });
                syncborder = (wsize as f64 * BORDER_FRAC) as i32;
                syncborder = BORDER_MAX.min(border_min.max(syncborder));
                if wposition < syncborder || wposition > wsize - syncborder {
                    trytrans = ((unsafe { (*vi).ysize } as f32 * 0.5f32) - unsafe { (*vi).ymouse }
                        + 0.5f32) as i32;
                    trydraws = self.ydrawsize;
                    tryborder = self.yborder;
                    trystart = self.ystart;
                    b3d_set_image_offset(
                        wsize,
                        unsafe { (*vi).ysize },
                        self.zoom as f64,
                        &mut trydraws,
                        &mut trytrans,
                        &mut tryborder,
                        &mut trystart,
                        1,
                    );
                    if tryborder != self.yborder || trystart != self.ystart {
                        tripshift += 2;
                    }
                }
            }

            if tripshift != 0 {
                self.xtrans = ((unsafe { (*vi).xsize } as f32 * 0.5f32) - unsafe { (*vi).xmouse }
                    + 0.5f32) as i32;
                self.ytrans = ((unsafe { (*vi).ysize } as f32 * 0.5f32) - unsafe { (*vi).ymouse }
                    + 0.5f32) as i32;
            }
        }
    }

    /// `ZapFuncs::getNewCIImage` (`xzap.cpp:1089`); `slot` is `-1` for
    /// `mImage` and the panel index for a member of `mImages`.
    pub fn get_new_ci_image(&mut self, slot: i32) -> bool {
        if self.ginit != 0 {
            with_boundary(|n| n.b3d_flush_image(slot));
        }

        let newim = with_boundary(|n| n.b3d_get_new_ci_image(slot));
        if !newim {
            wprint(
                "\u{7}Insufficient memory to run this Zap window.\nTry making it smaller or close it.\n",
            );
            return false;
        }

        with_boundary(|n| n.b3d_buffer_image(slot));
        true
    }

    /// `ZapFuncs::resize` (`xzap.cpp:1110`); receives the resize events which
    /// precede paint signals.
    pub fn resize(&mut self, mut winx: i32, mut winy: i32) {
        let old_screen_changed = self.screen_changed;
        let mut dpr;
        ivw_control_priority(unsafe { &mut *self.vi }, self.ctrl);

        if imod_debug('z') {
            imod_print_stderr("RESIZE: ");
        }

        if imod_debug('z') {
            let (ww, wh) = with_boundary(|n| n.zap_window_size());
            let th = with_boundary(|n| n.zap_window_toolbar_height(1));
            imod_print_stderr(&format!(
                "Size = {winx} x {winy}  win {ww} x {wh} tool {th} :"
            ));
            if self.ginit != 0 {
                imod_print_stderr(&format!("Old Size = {} x {} :", self.winx, self.winy));
            }
        }

        // The `Q_OS_MACX` deferred-resize arm is not compiled on this
        // platform (`xzap.cpp:1131-1137`).
        if self.winx > 1 && self.device_pixel_ratio > 0. {
            let new_qt_open_gl = APP
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|a| a.new_qt_open_gl != 0);
            if new_qt_open_gl {
                if with_boundary(|n| n.app_dev_pix_varies()) {
                    dpr = with_boundary(|n| n.util_get_new_dev_pix_ratio());
                    if dpr > 0. {
                        self.device_pixel_ratio = dpr;
                    }
                }

                winx = (winx as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
                winy = (winy as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
                if imod_debug('z') {
                    imod_print_stderr(&format!("scaled = {winx} x {winy}"));
                }
            }

            if self.defer_screen_zoom_change {
                self.screen_resize_time[0] = QTime::now();
            }
            self.defer_screen_zoom_change = false;
            let screen_elapsed = self.screen_resize_time[0].elapsed().as_millis() as i32;
            let resize_elapsed = self.screen_resize_time[1].elapsed().as_millis() as i32;
            let dev_pix_varies = with_boundary(|n| n.app_dev_pix_varies());
            let mut last = self.last_xsize_change;
            let mut changed = self.screen_changed;
            let mut zoom = self.zoom;
            util_set_zoom_on_screen_change(
                winx as f32 / self.winx as f32,
                &mut last,
                &mut changed,
                screen_elapsed,
                resize_elapsed,
                &mut zoom,
                dev_pix_varies,
            );
            self.last_xsize_change = last;
            self.screen_changed = changed;
            self.zoom = zoom;
            if old_screen_changed {
                with_boundary(|n| n.gfx_schedule_resize(250));
            }
        }
        self.winx = winx;
        self.winy = winy;
        with_boundary(|n| n.b3d_set_cur_size(winx, winy));
        with_boundary(|n| n.b3d_resize_viewport_xy(winx, winy));

        if self.num_xpanels != 0 {
            self.setup_panels();
        } else {
            self.image = self.get_new_ci_image(-1);
            if !self.image {
                return;
            }
            self.record_subarea = 1;
        }
        self.ginit = 1;
        if imod_debug('z') {
            imod_print_stderr("\n");
        }
    }

    /// `ZapFuncs::allocateToPanels` (`xzap.cpp:1180`).
    pub fn allocate_to_panels(
        &mut self,
        num: i32,
        win_size: i32,
        gutter: i32,
        panel_size: &mut i32,
        border: &mut i32,
    ) {
        let imarea = 0.max(win_size - (num - 1) * gutter);
        *panel_size = imarea / num;
        *border = (imarea % num) / 2;
    }

    /// `ZapFuncs::setupPanels` (`xzap.cpp:1188`).
    pub fn setup_panels(&mut self) -> i32 {
        let mut newnum = 0;
        let mut retval = 0;
        let numtot = self.num_xpanels * self.num_ypanels;
        let (num_x, winx, gutter) = (self.num_xpanels, self.winx, self.panel_gutter);
        let (mut size, mut border) = (0, 0);
        self.allocate_to_panels(num_x, winx, gutter, &mut size, &mut border);
        self.panel_xsize = size;
        self.panel_xborder = border;
        let (num_y, winy) = (self.num_ypanels, self.winy);
        self.allocate_to_panels(num_y, winy, gutter, &mut size, &mut border);
        self.panel_ysize = size;
        self.panel_yborder = border;

        // If panels are too small, set them to 1 as a signal
        if self.panel_xsize < 4 || self.panel_ysize < 4 {
            self.panel_xsize = 1;
            self.panel_ysize = 1;
            retval = 1;
        }

        // Set the minimum size regardless, so it won't get stuck at large
        let (w, h) = (
            (self.num_xpanels - 1) * (self.panel_gutter + 5) + 5,
            (self.num_ypanels - 1) * (self.panel_gutter + 5) + 5,
        );
        with_boundary(|n| n.gfx_set_minimum_size(w, h));

        // Allocate or resize the images, stop on a failure
        for i in 0..numtot {
            let ok = self.get_new_ci_image(i);
            self.images[i as usize] = ok;
            if !ok {
                retval = 2;
                break;
            }
            newnum += 1;
        }

        // Clear out unused images
        for i in newnum..self.num_images {
            with_boundary(|n| n.b3d_free_ci_image(i));
            self.images[i as usize] = false;
        }
        self.num_images = newnum;

        retval
    }

    /// `ZapFuncs::flushImage` (`xzap.cpp:1229`).
    pub fn flush_image(&mut self) {
        if self.num_xpanels != 0 {
            for ind in 0..self.num_images {
                with_boundary(|n| n.b3d_flush_image(ind));
            }
        } else {
            with_boundary(|n| n.b3d_flush_image(-1));
        }
    }

    /// `ZapFuncs::draw` (`xzap.cpp:1244`); the central drawing routine called
    /// from in the module or imodview.
    pub fn draw(&mut self) {
        let mut imz = 0;
        let app_closing = || APP.lock().unwrap().as_ref().is_some_and(|a| a.closing != 0);
        if app_closing() || self.doing_draw {
            return;
        }
        self.doing_draw = true;
        with_boundary(|n| n.gfx_update_gl());
        with_boundary(|n| n.imod_info_input());
        if app_closing() {
            return;
        }
        self.doing_draw = false;

        // Update pixel view if mouse is in this window
        if S_PIXEL_VIEW_OPEN.get() && self.num_xpanels == 0 {
            let (px, py) = with_boundary(|n| n.gfx_map_from_global_cursor_pos());
            let mx = (px as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
            let my = (py as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
            if mx >= 0 && mx < self.winx && my >= 0 && my < self.winy {
                let (mut imx, mut imy) = (0., 0.);
                self.getixy(mx, my, &mut imx, &mut imy, &mut imz);
                let vi = self.vi;
                with_boundary(|n| n.pv_new_mouse_position(vi, imx, imy, imz));
            }
        }
    }

    /// `ZapFuncs::paint` (`xzap.cpp:1271`); receives the paint events
    /// generated by the window manager.
    pub fn paint(&mut self) {
        let drawtime = QTime::now();
        if imod_debug('z') {
            imod_print_stderr(&format!("Paint  {}:", self.device_pixel_ratio));
        }

        // Use this to keep track of whether the first draw has happened
        if self.popup != 0 {
            self.popup = 2;
        }

        // The newer-Qt-on-Mac blend pre-enable (`xzap.cpp:1284-1295`) is not
        // compiled on this platform.

        with_boundary(|n| n.b3d_set_cur_size(self.winx, self.winy));
        with_boundary(|n| n.b3d_set_cur_dev_pix_ratio(self.device_pixel_ratio));

        if self.num_xpanels != 0 && self.panel_xsize < 4 {
            return;
        }

        self.auto_translate();
        self.drew_extra_cursor = false;

        // If the current only flag is set, swap the displayed buffer into the
        // drawing buffer and just draw the current contour.  Reset value
        // drawing since it has not been set up for this object
        if self.draw_current_only == 1 {
            if with_boundary(|n| n.app_doublebuffer()) {
                with_boundary(|n| n.gfx_swap_buffers());
            }
            with_boundary(|n| n.ifg_reset_value_setup());
            if self.drawing_lasso {
                self.draw_extra_object();
            } else {
                let ob = unsafe { (*(*self.vi).imod).cindex.object };
                with_boundary(|n| n.imod_set_object_color(ob));
                let obj = unsafe { (*(*self.vi).imod).obj.as_mut_ptr().add(ob as usize) };
                let width = unsafe { (*obj).linewidth2 } as i32;
                with_boundary(|n| n.b3d_line_width(width, obj));
                let co = unsafe { (*(*self.vi).imod).cindex.contour };
                self.draw_contour(co, ob);
            }
            self.draw_current_only = -1;
            return;
        }

        if self.draw_current_only > 1 {
            self.draw_current_only = 1;
        }

        if self.draw_current_only < 0 {
            self.set_draw_current_only(0);
        }

        /* DNM 1/29/03: no more skipping of further drawing */
        self.draw_graphics();

        if self.num_xpanels != 0 {
            // Multipanel draw: need to set borders and section for each panel
            let xborder_save = self.xborder;
            let yborder_save = self.yborder;
            let section_save = self.section;
            for ix in 0..self.num_xpanels {
                for iy in 0..self.num_ypanels {
                    let ind = ix + iy * self.num_xpanels;
                    let del_ind = ind - (self.num_xpanels * self.num_ypanels - 1) / 2;
                    self.section = section_save + self.panel_zstep * del_ind;
                    if self.section < 0 || self.section >= unsafe { (*self.vi).zsize } {
                        continue;
                    }
                    if (del_ind == 0 && self.draw_in_center == 0)
                        || (del_ind != 0 && self.draw_in_others == 0)
                    {
                        continue;
                    }
                    let panel_x = self.panel_xborder + ix * (self.panel_xsize + self.panel_gutter);
                    self.xborder = xborder_save + panel_x;
                    let panel_y = self.panel_yborder + iy * (self.panel_ysize + self.panel_gutter);
                    self.yborder = yborder_save + panel_y;
                    let (px, py, pw, ph) = (panel_x, panel_y, self.panel_xsize, self.panel_ysize);
                    with_boundary(|n| n.b3d_subarea_viewport(px, py, pw, ph));
                    self.draw_model();
                    self.draw_current_point();
                    self.draw_extra_object();
                }
            }
            self.xborder = xborder_save;
            self.yborder = yborder_save;
            self.section = section_save;
            let (wx, wy) = (self.winx, self.winy);
            with_boundary(|n| n.b3d_resize_viewport_xy(wx, wy));
        } else {
            // Normal draw
            self.draw_model();
            self.draw_current_point();
            self.draw_extra_object();
            self.draw_auto();
            if self.rubberband != 0 {
                with_boundary(|n| n.b3d_line_width(1, ptr::null_mut()));
                let endpoint = APP.lock().unwrap().as_ref().map_or(0, |a| a.endpoint);
                with_boundary(|n| n.b3d_color_index(endpoint));
                self.band_image_to_mouse(0);
                let (x, y, w, h) = (
                    self.rb_mouse_x0,
                    self.winy - 1 - self.rb_mouse_y1,
                    self.rb_mouse_x1 - self.rb_mouse_x0,
                    self.rb_mouse_y1 - self.rb_mouse_y0,
                );
                with_boundary(|n| n.b3d_draw_rectangle(x, y, w, h));
            }

            for ix in 0..self.arrow_head.len() {
                if (self.xpos(self.arrow_tail[ix].x) as f64
                    - self.xpos(self.arrow_head[ix].x) as f64)
                    .abs()
                    > 2.
                    || (self.ypos(self.arrow_tail[ix].y) as f64
                        - self.ypos(self.arrow_head[ix].y) as f64)
                        .abs()
                        > 2.
                {
                    let color = APP.lock().unwrap().as_ref().map_or(0, |a| a.arrow[ix % 4]);
                    with_boundary(|n| n.b3d_color_index(color));
                    let (tx, ty, hx, hy) = (
                        self.xpos(self.arrow_tail[ix].x),
                        self.ypos(self.arrow_tail[ix].y),
                        self.xpos(self.arrow_head[ix].x),
                        self.ypos(self.arrow_head[ix].y),
                    );
                    with_boundary(|n| n.b3d_draw_arrow(tx, ty, hx, hy, 16, 4, true));
                }
            }
        }
        self.draw_tools();
        let (wx, wy, zoom) = (self.winx, self.winy, self.zoom);
        self.scale_bar_size = with_boundary(|n| n.scale_bar_draw(wx, wy, zoom, 0));

        // Update graph windows if rubber band changed (this should be done by
        // control but that is not possible, it doesn't know types)
        if self.band_changed != 0 {
            with_boundary(|n| n.graph_window_list_draw());
            self.band_changed = 0;
        }

        if imod_debug('z') {
            imod_print_stderr("\n");
        }
        if self.hqgfx != 0 {
            self.last_hq_draw_time = drawtime.elapsed().as_millis() as i32;
        }
    }

    /// `ZapFuncs::stepZoom` (`xzap.cpp:1414`).
    pub fn step_zoom(&mut self, step: i32) {
        self.set_control_and_limits();
        self.zoom = with_boundary(|n| n.b3d_step_pixel_zoom(self.zoom as f64, step)) as f32;
        self.draw();
    }

    /// `ZapFuncs::enteredZoom` (`xzap.cpp:1421`).
    pub fn entered_zoom(&mut self, new_zoom: f32) {
        if self.popup == 0 {
            return;
        }
        self.set_control_and_limits();
        self.zoom = new_zoom;
        if self.zoom <= 0.001 {
            self.zoom = 0.001;
            self.tool_zoom = new_zoom;
        }
        self.draw();
        with_boundary(|n| n.zap_window_set_focus());
    }

    /// `ZapFuncs::stateToggled` (`xzap.cpp:1435`).
    pub fn state_toggled(&mut self, index: usize, state: i32) {
        let mut time = 0;
        self.set_control_and_limits();
        match index {
            ZAP_TOGGLE_RESOL => {
                self.hqgfx = state;
                self.draw();
            }
            ZAP_TOGGLE_ZLOCK => {
                self.lock = if state != 0 { 2 } else { 0 };
                if self.lock == 0 {
                    self.flush_image();
                    self.sync_image(false);
                    self.draw();
                }
            }
            ZAP_TOGGLE_CENTER => {
                self.keepcentered = state;
                if state != 0 {
                    self.flush_image();
                    self.sync_image(true);
                    self.draw();
                }
            }
            ZAP_TOGGLE_INSERT => {
                self.insertmode = state as i16;
                unsafe { (*self.vi).insertmode = self.insertmode as i32 };
                self.register_drag_additions();
            }
            ZAP_TOGGLE_RUBBER => self.toggle_rubberband(true),
            ZAP_TOGGLE_LASSO => self.toggle_lasso(true),
            ZAP_TOGGLE_ARROW => self.toggle_arrow(true),
            ZAP_TOGGLE_TIMELOCK => {
                unsafe {
                    crate::imod::three_dmod::imodview::ivw_get_time(self.vi, Some(&mut time))
                };
                self.time_lock = if state != 0 { time } else { 0 };
                if self.time_lock == 0 {
                    self.draw();
                }
            }
            _ => {}
        }
    }

    /// `ZapFuncs::enteredSection` (`xzap.cpp:1490`).
    pub fn entered_section(&mut self, sec: i32) {
        if self.popup == 0 {
            return;
        }
        self.set_control_and_limits();
        if self.lock != 2 {
            unsafe { (*self.vi).zmouse = (sec - 1) as f32 };
        }
        self.section = sec - 1;
        crate::imod::three_dmod::imodview::ivw_bind_mouse(unsafe { &mut *self.vi });
        let vi = self.vi;
        with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_XYZ));
        with_boundary(|n| n.zap_window_set_focus());
    }

    /// `ZapFuncs::stepTime` (`xzap.cpp:1503`).
    pub fn step_time(&mut self, step: i32) {
        self.set_control_and_limits();

        // if time locked, advance the time lock and draw this window
        // Does this make sense?
        if self.time_lock != 0 {
            self.time_lock += step;
            if self.time_lock <= 0 {
                self.time_lock = 1;
            }
            if self.time_lock > ivw_get_max_time(unsafe { &*self.vi }) {
                self.time_lock = ivw_get_max_time(unsafe { &*self.vi });
            }
            self.draw();
        } else {
            let vi = self.vi;
            with_boundary(|n| {
                n.imod_movie_xyzt(vi, MOVIE_DEFAULT, MOVIE_DEFAULT, MOVIE_DEFAULT, 0)
            });
            if step > 0 {
                with_boundary(|n| n.input_next_time(vi));
            } else {
                with_boundary(|n| n.input_prev_time(vi));
            }
        }
    }

    /// `ZapFuncs::screenChanged` (`xzap.cpp:1526`).
    pub fn screen_changed(&mut self, new_dpr: f32) {
        if new_dpr != 0. {
            let (px, py) = with_boundary(|n| n.zap_window_pos());
            imod_trace(
                'z',
                &format!("screen change Zap pos {px} {py}  DPR {new_dpr}"),
            );
            if self.popup != 0 {
                let screen_elapsed = self.screen_resize_time[0].elapsed().as_millis() as i32;
                let resize_elapsed = self.screen_resize_time[1].elapsed().as_millis() as i32;
                let dev_pix_varies = with_boundary(|n| n.app_dev_pix_varies());
                let mut last = self.last_xsize_change;
                let mut changed = self.screen_changed;
                let mut zoom = self.zoom;
                util_set_zoom_on_screen_change(
                    0.,
                    &mut last,
                    &mut changed,
                    screen_elapsed,
                    resize_elapsed,
                    &mut zoom,
                    dev_pix_varies,
                );
                self.last_xsize_change = last;
                self.screen_changed = changed;
                self.zoom = zoom;
                // `HANDLE_DPR_CHANGE_FOR_MAC` (`utilities.h:56`) is the plain
                // assignment off the Mac.
                self.device_pixel_ratio = new_dpr;
            }

            // For Windows, there is another resize after the screen change -
            // here use up setting it to the right zoom
            if self.new_screen_zoom > 0. && new_dpr > with_boundary(|n| n.app_min_dev_pix_ratio()) {
                self.zoom = self.new_screen_zoom;
                imod_trace('z', &format!("Zoom set to {:.2}\n", self.zoom));
                self.new_screen_zoom = 0.;
            }
        }
    }

    /// `ZapFuncs::autoTranslate` (`xzap.cpp:1547`).
    pub fn auto_translate(&mut self) {
        if self.lock == 2 {
            return;
        }

        self.section = (unsafe { (*self.vi).zmouse } + 0.5f32) as i32;

        self.draw_tools();

        if self.lock != 0 {}
    }

    /// `ZapFuncs::translate` (`xzap.cpp:1565`).
    pub fn translate(&mut self, x: i32, y: i32) {
        let vw = self.vi;
        self.xtrans += x;
        if self.xtrans > unsafe { (*vw).xsize } {
            self.xtrans = unsafe { (*vw).xsize };
        }
        if self.xtrans < -unsafe { (*vw).xsize } {
            self.xtrans = -unsafe { (*vw).xsize };
        }
        self.ytrans += y;
        if self.ytrans > unsafe { (*vw).ysize } {
            self.ytrans = unsafe { (*vw).ysize };
        }
        if self.ytrans < -unsafe { (*vw).ysize } {
            self.ytrans = -unsafe { (*vw).ysize };
        }
        self.draw();
    }

    /// `ZapFuncs::keyInput` (`xzap.cpp:1588`); respond to a key press.
    pub fn key_input(&mut self, event: &KeyEvent) {
        let vi = self.vi;
        let imod = unsafe { (*vi).imod };
        let mut keysym = event.qt_key;
        let mut limarr = [0; 4];
        let (mut rx, mut ix, mut iy, mut i, mut obst, mut obnd, mut ob, mut start, mut end);
        let mut keypad = i32::from(event.keypad);
        let shifted = i32::from(event.shift);
        let ctrl =
            with_boundary(|n| n.input_test_ctrl(if event.control { CONTROL_MODIFIER } else { 0 }));
        let mut handled = 0;
        let mut indadd = Iindex {
            object: 0,
            contour: 0,
            point: 0,
        };
        let mut selmin = Ipoint::default();
        let mut selmax = Ipoint::default();
        let mut add_pt = Ipoint::default();
        let mut lasso: *mut Icont;
        let mut cont: *mut Icont;
        let (mut cx, mut cy) = (0., 0.);
        let mut obj: *mut Iobj;

        if imod_debug('k') {
            imod_print_stderr(&format!("key {keysym:x}, state {:x}\n", event.qt_key));
        }
        if with_boundary(|n| n.input_test_meta_key(event)) != 0 {
            return;
        }

        if util_close_key(keysym) && with_boundary(|n| n.gfx_first_draw()) <= 0 {
            // For cocoa/Qt 4.5.0, need to prevent enter/leave events
            self.popup = 0;
            with_boundary(|n| n.zap_window_close());
            return;
        }

        with_boundary(|n| n.input_convert_num_lock(&mut keysym, &mut keypad));

        self.set_control_and_limits();
        ivw_control_active(unsafe { &mut *vi }, 0);

        let wtype = if self.num_xpanels != 0 {
            MULTIZ_WINDOW_TYPE
        } else {
            ZAP_WINDOW_TYPE
        };
        if with_boundary(|n| n.imod_plug_handle_key(vi, event, wtype)) != 0 {
            return;
        }
        ivw_control_active(unsafe { &mut *vi }, 1);

        /* DNM: set global insertmode from this zap's mode to get it to work
        right with Delete key */
        unsafe { (*vi).insertmode = self.insertmode as i32 };

        match keysym {
            KEY_UP | KEY_DOWN | KEY_RIGHT | KEY_LEFT => {
                let trans = S_KEY_TRANS.get();
                // If arrows scroll and it is not keypad, call the appropriate
                // routine for up/down
                if keypad == 0
                    && (keysym == KEY_DOWN || keysym == KEY_UP)
                    && with_boundary(|n| n.prefs_arrows_scroll_zap())
                {
                    if self.lock == 2 {
                        self.locked_page_up_or_down(
                            shifted,
                            if keysym == KEY_PAGE_UP { 1 } else { -1 },
                        );
                    } else {
                        ivw_control_active(unsafe { &mut *vi }, 0);
                        let dir = if keysym == KEY_UP { 1 } else { -1 };
                        with_boundary(|n| n.input_page_up_or_down(vi, shifted, dir));
                    }
                    handled = 1;

                // Translate with keypad in movie mode or regular arrows in
                // model mode
                } else if (keypad == 0 && unsafe { (*imod).mousemode } != IMOD_MMOVIE)
                    || (keypad != 0 && unsafe { (*imod).mousemode } == IMOD_MMOVIE)
                {
                    if keysym == KEY_LEFT {
                        self.translate(-trans, 0);
                    }
                    if keysym == KEY_RIGHT {
                        self.translate(trans, 0);
                    }
                    if keysym == KEY_DOWN {
                        self.translate(0, -trans);
                    }
                    if keysym == KEY_UP {
                        self.translate(0, trans);
                    }
                    handled = 1;

                // Move point with keypad in model mode
                } else if keypad != 0 && unsafe { (*imod).mousemode } != IMOD_MMOVIE {
                    with_boundary(|n| n.input_key_point_move(vi, keysym));
                    handled = 1;
                }
            }

            KEY_PAGE_UP | KEY_PAGE_DOWN => {
                let trans = S_KEY_TRANS.get();
                // With keypad, translate in movie mode or move point in model
                if keypad != 0 {
                    if unsafe { (*imod).mousemode } == IMOD_MMOVIE {
                        self.translate(
                            trans,
                            if keysym == KEY_PAGE_DOWN {
                                -trans
                            } else {
                                trans
                            },
                        );
                    } else {
                        with_boundary(|n| n.input_key_point_move(vi, keysym));
                    }
                    handled = 1;

                // with regular keys, handle specially if locked
                } else if keypad == 0 && self.lock == 2 {
                    self.locked_page_up_or_down(
                        shifted,
                        if keysym == KEY_PAGE_UP { 1 } else { -1 },
                    );
                    handled = 1;
                }
            }

            KEY_1 | KEY_2 => {
                if self.time_lock != 0 {
                    self.step_time(if keysym == KEY_1 { -1 } else { 1 });
                    handled = 1;
                }
            }

            KEY_AT | KEY_EXCLAM => {
                if self.time_lock != 0 {
                    let (s, e) = with_boundary(|n| n.imc_get_start_end(vi, 3));
                    start = s;
                    end = e;
                    self.time_lock = if keysym == KEY_AT { end + 1 } else { start + 1 };
                    self.draw();
                    handled = 1;
                }
            }

            KEY_HOME => {
                let trans = S_KEY_TRANS.get();
                if keypad != 0 && unsafe { (*imod).mousemode } == IMOD_MMOVIE {
                    self.translate(-trans, trans);
                    handled = 1;
                }
            }

            KEY_END => {
                let trans = S_KEY_TRANS.get();
                if keypad != 0 && unsafe { (*imod).mousemode } == IMOD_MMOVIE {
                    self.translate(-trans, -trans);
                    handled = 1;
                }
            }

            KEY_MINUS => {
                self.zoom = with_boundary(|n| n.b3d_step_pixel_zoom(self.zoom as f64, -1)) as f32;
                self.draw();
                handled = 1;
            }

            KEY_PLUS | KEY_EQUAL => {
                self.zoom = with_boundary(|n| n.b3d_step_pixel_zoom(self.zoom as f64, 1)) as f32;
                self.draw();
                handled = 1;
            }

            /* DNM: Keypad Insert key, alternative to middle mouse button */
            KEY_INSERT => {
                /* But skip out if in movie mode or already active */
                if keypad == 0
                    || unsafe { (*imod).mousemode } == IMOD_MMOVIE
                    || S_INSERT_DOWN.get() != 0
                {
                    // break
                } else {
                    // It wouldn't work going to a QPoint and accessing it, so
                    // do it in shot!
                    let (px, py) = with_boundary(|n| n.gfx_map_from_global_cursor_pos());
                    ix = (px as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
                    iy = (py as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;

                    // For multi-Z, add a point at mouse position, do not
                    // change current Z
                    let mut broke = false;
                    if self.num_xpanels != 0 {
                        i = 0;
                        let (mut ax, mut ay) = (0., 0.);
                        self.getixy(ix, iy, &mut ax, &mut ay, &mut i);
                        add_pt.x = ax;
                        add_pt.y = ay;
                        if i < 0 {
                            broke = true;
                        } else {
                            add_pt.z = i as f32;
                            obj = imod_object_get(unsafe { imod.as_ref() })
                                .map_or(ptr::null_mut(), |o| o as *const Iobj as *mut Iobj);
                            cont = ptr::null_mut();
                            if !obj.is_null() {
                                cont = unsafe { ivw_get_or_make_contour(vi, obj, self.time_lock) };
                            }
                            if obj.is_null()
                                || cont.is_null()
                                || unsafe { ivw_time_mismatch(vi, self.time_lock, obj, cont) }
                            {
                                broke = true;
                            } else {
                                end = unsafe { (*imod).cindex.point }
                                    + if self.insertmode != 0 { 0 } else { 1 };
                                if unsafe { (*imod).cindex.point } < 0 {
                                    end = if self.insertmode != 0 {
                                        0
                                    } else {
                                        unsafe { (&(*cont).pts).len() as i32 }
                                    };
                                }
                                if unsafe { !(&(*cont).pts).is_empty() }
                                    && (unsafe { (&(*cont).pts)[0].z } as f64 + 0.5).floor() as i32
                                        != add_pt.z as i32
                                {
                                    unsafe { (*cont).flags |= ICONT_WILD };
                                }
                                unsafe { ivw_register_insert_point(vi, cont, &mut add_pt, end) };
                                unsafe { (*vi).xmouse = add_pt.x };
                                unsafe { (*vi).ymouse = add_pt.y };
                                with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_MOD | IMOD_DRAW_XYZ));
                                handled = 1;
                                broke = true;
                            }
                        }
                    }

                    if !broke {
                        // Set a flag, set continuous tracking, grab keyboard
                        // and mouse
                        S_INSERT_DOWN.set(1);
                        self.set_mouse_tracking();
                        with_boundary(|n| n.zap_window_grab_keyboard());
                        with_boundary(|n| n.gfx_grab_mouse());

                        /* Use time since last event to determine whether to
                        treat like single click or drag */
                        rx = S_INSERT_TIME.with(|t| t.borrow().elapsed().as_millis() as i32);
                        S_INSERT_TIME.with(|t| *t.borrow_mut() = QTime::now());
                        if rx > 250 {
                            self.b2_click(ix, iy, 0);
                        } else {
                            self.b2_drag(ix, iy, 0);
                        }

                        self.lmx = ix;
                        self.lmy = iy;
                        handled = 1;
                    }
                }
            }

            // Step by # of panels displayed in Multi-Z
            KEY_SLASH => {
                if self.num_xpanels != 0 && keypad != 0 {
                    let step = self.num_xpanels * self.num_ypanels;
                    with_boundary(|n| n.input_prevz(vi, step));
                    handled = 1;
                }
            }

            KEY_ASTERISK => {
                if self.num_xpanels != 0 && keypad != 0 {
                    let step = self.num_xpanels * self.num_ypanels;
                    with_boundary(|n| n.input_nextz(vi, step));
                    handled = 1;
                }
            }

            /* DNM 12/13/01: add next and smooth hotkeys to autox */
            KEY_A => {
                if self.num_xpanels == 0 {
                    if ctrl != 0 {
                        // Select all contours in current object on section or
                        // in rubberband
                        if self.rubberband != 0 {
                            selmin.x = self.rb_image_x0;
                            selmax.x = self.rb_image_x1;
                            selmin.y = self.rb_image_y0;
                            selmax.y = self.rb_image_y1;
                        } else {
                            selmin.x = -unsafe { (*vi).xsize } as f32;
                            selmax.x = 2. * unsafe { (*vi).xsize } as f32;
                            selmin.y = -unsafe { (*vi).ysize } as f32;
                            selmax.y = 2. * unsafe { (*vi).ysize } as f32;
                        }
                        selmin.z = self.section as f32 - 0.5;
                        selmax.z = self.section as f32 + 0.5;
                        lasso = ptr::null_mut();
                        if self.lasso_on && !self.drawing_lasso {
                            lasso = self.get_lasso_contour();
                        }

                        // Look through selection list, remove any that do not
                        // fit constraints
                        let sel_list = unsafe { (*vi).selection_list };
                        i = unsafe { ilist_size(sel_list) } - 1;
                        while i >= 0 {
                            let indp = unsafe { ilist_item(sel_list, i) } as *mut Iindex;
                            let mut keep = false;
                            if unsafe { (*indp).object } < unsafe { (&(*imod).obj).len() as i32 } {
                                obj = unsafe {
                                    (&mut (*imod).obj).as_mut_ptr().add((*indp).object as usize)
                                };
                                if unsafe { (*indp).contour }
                                    < unsafe { (&(*obj).cont).len() as i32 }
                                {
                                    let co = unsafe { (*indp).contour } as usize;
                                    let cont_ref = unsafe { &(&(*obj).cont)[co] };
                                    if (!lasso.is_null()
                                        && crate::imod::three_dmod::imod_edit::imod_cont_inside_cont(
                                            unsafe { &*obj },
                                            cont_ref,
                                            unsafe { &*lasso },
                                            selmin.z,
                                            selmax.z,
                                        ) != 0)
                                        || (lasso.is_null()
                                            && crate::imod::three_dmod::imod_edit::imod_cont_in_select_area(
                                                unsafe { &*obj },
                                                cont_ref,
                                                selmin,
                                                selmax,
                                            ) != 0)
                                    {
                                        keep = true;
                                    }
                                }
                            }
                            if !keep {
                                unsafe { ilist_remove(sel_list, i) };
                            }
                            i -= 1;
                        }

                        obst = if shifted != 0 {
                            0
                        } else {
                            unsafe { (*imod).cindex.object }
                        };
                        obnd = if shifted != 0 {
                            (unsafe { (&(*imod).obj).len() }) as i32 - 1
                        } else {
                            unsafe { (*imod).cindex.object }
                        };

                        if obnd >= 0 {
                            // Set up an index to add, look for contours inside
                            // the bounding box, add them, make last one current
                            ob = obst;
                            while ob <= obnd {
                                obj = unsafe { (&mut (*imod).obj).as_mut_ptr().add(ob as usize) };
                                indadd.object = ob;
                                indadd.point = -1;
                                i = 0;
                                while i < unsafe { (&(*obj).cont).len() as i32 } {
                                    indadd.contour = i;
                                    let cont_ref = unsafe { &(&(*obj).cont)[i as usize] };
                                    if (!lasso.is_null()
                                        && crate::imod::three_dmod::imod_edit::imod_cont_inside_cont(
                                            unsafe { &*obj },
                                            cont_ref,
                                            unsafe { &*lasso },
                                            selmin.z,
                                            selmax.z,
                                        ) != 0)
                                        || (lasso.is_null()
                                            && crate::imod::three_dmod::imod_edit::imod_cont_in_select_area(
                                                unsafe { &*obj },
                                                cont_ref,
                                                selmin,
                                                selmax,
                                            ) != 0)
                                    {
                                        with_boundary(|n| n.imod_selection_list_add(vi, indadd));
                                        unsafe { (*imod).cindex = indadd };
                                    }
                                    i += 1;
                                }
                                ob += 1;
                            }
                            with_boundary(|n| n.imod_setxyzmouse());
                            handled = 1;
                        }
                    } else if shifted == 0 {
                        if !unsafe { (*vi).ax }.is_null() {
                            with_boundary(|n| n.autox_next(vi));
                        } else if with_boundary(|n| n.iproc_is_open())
                            && with_boundary(|n| n.iproc_busy()) == 0
                            && unsafe { (*vi).loading_image } == 0
                        {
                            with_boundary(|n| n.iproc_apply());
                        }
                        handled = 1;
                    }
                }
            }

            KEY_U => {
                if shifted == 0 && self.num_xpanels == 0 {
                    with_boundary(|n| n.autox_smooth(vi));
                    handled = 1;
                }
            }

            KEY_B => {
                if self.num_xpanels == 0 && ctrl == 0 {
                    if shifted != 0 {
                        self.toggle_rubberband(true);
                    } else {
                        with_boundary(|n| n.autox_build(vi));
                    }
                    handled = 1;
                }
            }

            KEY_P => {
                if self.num_xpanels == 0 && shifted != 0 {
                    self.toggle_contour_shift();
                    handled = 1;
                }
            }

            KEY_F => {
                if self.num_xpanels == 0 && unsafe { (*vi).loading_image } == 0 && shifted != 0 {
                    if with_boundary(|n| n.iproc_busy()) != 0 {
                        wprint("\u{7}Image processing is busy\n");
                    } else {
                        with_boundary(|n| n.iproc_toggle_full_fft(vi));
                    }
                    handled = 1;
                }
            }

            KEY_S => {
                if shifted != 0 || ctrl != 0 {
                    self.showslice = self.showed_slice as i16;

                    // Turn off double buffering and read from back buffer
                    if with_boundary(|n| n.app_doublebuffer()) {
                        with_boundary(|n| n.gfx_set_buffer_swap_auto(false));
                        let front = APP
                            .lock()
                            .unwrap()
                            .as_ref()
                            .is_some_and(|a| a.new_qt_open_gl != 0);
                        with_boundary(|n| n.gl_read_buffer(front));
                    }
                    with_boundary(|n| n.util_pre_snap_changes(vi));

                    // Take a montage snapshot if selected and no rubberband
                    if with_boundary(|n| n.imc_get_snap_montage(true)) && self.num_xpanels == 0 {
                        self.montage_snapshot(
                            (if ctrl != 0 { 1 } else { 0 }) + (if shifted != 0 { 2 } else { 0 }),
                        );
                    } else {
                        self.draw();
                        let limits = self.set_snapshot_limits(&mut limarr);
                        let name = if self.num_xpanels != 0 {
                            "multiz"
                        } else {
                            "zap"
                        };
                        with_boundary(|n| n.b3d_key_snapshot(name, shifted, ctrl, limits));
                    }
                    if with_boundary(|n| n.app_doublebuffer()) {
                        with_boundary(|n| n.gfx_swap_buffers());
                        with_boundary(|n| n.gfx_set_buffer_swap_auto(true));
                    }
                    with_boundary(|n| n.util_restore_snap_changes(vi));
                } else {
                    with_boundary(|n| n.input_save_model(vi));
                }
                handled = 1;
            }

            KEY_R => {
                if ctrl != 0 && shifted != 0 && self.num_xpanels == 0 && self.rubberband != 0 {
                    self.zoom = (self.winx as f64
                        / (self.rb_image_x1 as f64 + 1. - self.rb_image_x0 as f64))
                        .min(
                            self.winy as f64
                                / (self.rb_image_y1 as f64 + 1. - self.rb_image_y0 as f64),
                        ) as f32;
                    self.xtrans = (-(self.rb_image_x1 + self.rb_image_x0
                        - unsafe { (*self.vi).xsize } as f32)
                        / 2.) as i32;
                    self.ytrans = (-(self.rb_image_y1 + self.rb_image_y0
                        - unsafe { (*self.vi).ysize } as f32)
                        / 2.) as i32;
                    self.draw();
                    handled = 1;
                } else if shifted != 0 && self.num_xpanels == 0 {
                    self.resize_to_fit();
                    handled = 1;
                }
            }

            KEY_Z => {
                if shifted != 0 {
                    if self.num_xpanels == 0 {
                        if self.section_step != 0 {
                            self.section_step = 0;
                            wprint("Auto-section advance turned OFF\n");
                        } else {
                            self.section_step = 1;
                            wprint("\u{7}Auto-section advance turned ON\n");
                        }
                    }
                    handled = 1;
                }
            }

            KEY_I => {
                if self.num_xpanels == 0 {
                    if shifted != 0 {
                        self.print_info(true);
                    } else {
                        self.state_toggled(ZAP_TOGGLE_INSERT, 1 - self.insertmode as i32);
                        let mode = self.insertmode as i32;
                        with_boundary(|n| n.zap_window_set_toggle_state(ZAP_TOGGLE_INSERT, mode));
                        wprint("\u{7}Toggled modeling direction\n");
                    }
                    handled = 1;
                }
            }

            KEY_K => {
                if shifted == 0 && ctrl == 0 {
                    self.state_toggled(ZAP_TOGGLE_CENTER, 1 - self.keepcentered);
                    let kc = self.keepcentered;
                    with_boundary(|n| n.zap_window_set_toggle_state(ZAP_TOGGLE_CENTER, kc));
                    handled = 1;
                }
            }

            KEY_Q => {
                let (px, py) = with_boundary(|n| n.gfx_map_from_global_cursor_pos());
                ix = (px as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
                iy = (py as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
                i = 0;
                self.getixy(ix, iy, &mut cx, &mut cy, &mut i);
                let mut refx = unsafe { (*vi).xmouse };
                let mut refy = unsafe { (*vi).ymouse };
                let cur_pnt = imod_point_get(unsafe { &mut *imod })
                    .map_or(ptr::null(), |p| p as *const Ipoint);
                if !cur_pnt.is_null() && unsafe { (*imod).mousemode } == IMOD_MMODEL {
                    refx = unsafe { (*cur_pnt).x };
                    refy = unsafe { (*cur_pnt).y };
                }
                let dx = cx - refx;
                let dy = cy - refy;
                let dist2d =
                    (unsafe { (*vi).xybin } as f64 * ((dx * dx + dy * dy) as f64).sqrt()) as f32;
                wprint(&format!(
                    "From ({:.1}, {:.1}) to ({:.1}, {:.1}) =\n",
                    refx as f64 + 1.,
                    refy as f64 + 1.,
                    cx as f64 + 1.,
                    cy as f64 + 1.
                ));
                let str = format!(
                    "  {:.1} {}pixels",
                    dist2d,
                    if unsafe { (*vi).xybin } > 1 {
                        "unbinned "
                    } else {
                        ""
                    }
                );
                with_boundary(|n| n.util_wprint_measure(&str, imod, dist2d, false));
            }

            KEY_F1..=KEY_F8 | KEY_F11 | KEY_F12 => {
                if self.time_lock > 0 && self.time_lock != unsafe { (*vi).cur_time } {
                    let tl = self.time_lock;
                    with_boundary(|n| n.input_set_time_lock_for_f_keys(tl));
                }
            }

            _ => {}
        }

        // If event not handled, pass up to default processor
        if handled != 0 {
            // `event->accept()` is the Qt default; the portable payload has
            // no accepted flag to set.
        } else {
            // What does this mean? It is needed to get images to sync right
            ivw_control_active(unsafe { &mut *vi }, 0);
            with_boundary(|n| n.input_q_default_keys(event, vi));
            with_boundary(|n| n.input_set_time_lock_for_f_keys(0));
        }
    }

    /// `ZapFuncs::lockedPageUpOrDown` (`xzap.cpp:2085`).
    pub fn locked_page_up_or_down(&mut self, shifted: i32, direction: i32) {
        if shifted != 0 {
            let obj = imod_object_get(unsafe { (*self.vi).imod.as_ref() });
            self.section =
                util_next_sec_with_cont(unsafe { &*self.vi }, obj, self.section, direction);
        } else {
            self.section += direction;
        }
        self.section = self.section.clamp(0, unsafe { (*self.vi).zsize } - 1);
        self.draw();
    }

    /// `ZapFuncs::keyRelease` (`xzap.cpp:2101`); key is raised, finish up
    /// various tasks.
    pub fn key_release(&mut self, event: &KeyEvent) {
        if S_INSERT_DOWN.get() == 0
            || !event.keypad
            || (event.qt_key != KEY_INSERT && event.qt_key != KEY_0)
        {
            return;
        }
        S_INSERT_DOWN.set(0);
        self.register_drag_additions();
        self.set_mouse_tracking();
        with_boundary(|n| n.zap_window_release_keyboard());
        with_boundary(|n| n.gfx_release_mouse());

        // Note that unless the user turns off autorepeat on the key, there is
        // a series of key press - release events and it does full draws
        if self.draw_current_only != 0 {
            self.set_draw_current_only(0);
            let vi = self.vi;
            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_MOD | IMOD_DRAW_XYZ));
        }
    }

    /// `ZapFuncs::generalEvent` (`xzap.cpp:2122`); pass on various events to
    /// plugins.
    pub fn general_event(&mut self, e: &ZapGeneralEvent) {
        let mut iz = 0;
        let (mut imx, mut imy) = (0., 0.);
        let app_closing = APP.lock().unwrap().as_ref().is_some_and(|a| a.closing != 0);
        if self.num_xpanels != 0 || self.popup == 0 || app_closing {
            return;
        }
        let (px, py) = with_boundary(|n| n.gfx_map_from_global_cursor_pos());
        let ix = (px as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        let iy = (py as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        self.getixy(ix, iy, &mut imx, &mut imy, &mut iz);

        // Needed for Mac Qt 5.7/8
        if with_boundary(|n| n.util_need_to_set_cursor()) && e.event_type == EVENT_ENTER {
            let mode = unsafe { (*(*self.vi).imod).mousemode };
            self.set_cursor(mode, true);
        }
        let vi = self.vi;
        let ifdraw = with_boundary(|n| n.imod_plug_handle_event(vi, e, imx, imy, ZAP_WINDOW_TYPE));
        if ifdraw & 2 != 0 || (self.drew_extra_cursor && e.event_type == EVENT_LEAVE) {
            self.draw();
        }
        if ifdraw != 0 {
            return;
        }
        if e.event_type == EVENT_WHEEL && with_boundary(|n| n.ice_get_wheel_for_size()) {
            // `utilWheelChangePointSize(vi, mZoom, event)` operates on the
            // current contour's current point.
            let imod = unsafe { (*vi).imod };
            let point = unsafe { (*imod).cindex.point };
            let cont = imod_contour_get(unsafe { imod.as_ref() })
                .map_or(ptr::null_mut(), |c| c as *const Icont as *mut Icont);
            if !cont.is_null() && point >= 0 {
                let zoom = self.zoom;
                let delta = e.wheel_delta;
                with_boundary(|n| {
                    crate::imod::three_dmod::utilities::util_wheel_change_point_size(
                        n,
                        unsafe { &mut *cont },
                        point as usize,
                        zoom,
                        delta,
                    )
                });
            }
        }
    }

    /// `ZapFuncs::mousePress` (`xzap.cpp:2148`); respond to a mouse press.
    pub fn mouse_press(&mut self, event: &KeyEvent) {
        let mut ifdraw = 0;
        let mut drew = 0;
        let ctrl_down =
            with_boundary(|n| n.input_test_ctrl(if event.control { CONTROL_MODIFIER } else { 0 }));
        let mut dxll = 0;
        let mut cont: *mut Icont;
        let mut mpt = Ipoint::default();
        let rcrit = 10; /* Criterion for moving the whole band */
        let but1 = with_boundary(|n| n.prefs_actual_button(1)) as u32;
        let but2 = with_boundary(|n| n.prefs_actual_button(2)) as u32;
        let but3 = with_boundary(|n| n.prefs_actual_button(3)) as u32;
        let ebut1 = event.button == but1;
        let ebut2 = event.button == but2;

        let button1 = i32::from(event.buttons & but1 != 0);
        let button2 = i32::from(event.buttons & but2 != 0);
        let button3 = i32::from(event.buttons & but3 != 0);
        let x = (event.x as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        let y = (event.y as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        S_BUT1_DOWNT.with(|t| *t.borrow_mut() = QTime::now());
        S_FIRSTMX.set(x);
        S_FIRSTMY.set(y);
        self.lmx = x;
        self.lmy = y;
        S_MOUSE_PRESSED.set(true);
        with_boundary(|n| n.util_raise_if_needed(event));

        if imod_debug('m') {
            imod_print_stderr(&format!(
                "click at {x} {y}   buttons {button1} {button2} {button3}\n"
            ));
        }

        // Check for starting a band move before offering to plugin
        if ebut2 && button1 == 0 && button3 == 0 && self.shifting_cont == 0 {
            S_MOVE_BAND_LASSO.set(0);

            /* If rubber band is on and within criterion distance of any edge,
            set flag to move whole band and return */
            if self.rubberband != 0 {
                self.band_image_to_mouse(0);
                if util_test_band_move(
                    x,
                    y,
                    [
                        self.rb_mouse_x0,
                        self.rb_mouse_x1,
                        self.rb_mouse_y0,
                        self.rb_mouse_y1,
                    ],
                ) != 0
                {
                    S_MOVE_BAND_LASSO.set(1);
                    self.set_cursor(self.mousemode, false);
                    return;
                }
            }
        }

        // Also check if lasso is within criterion distance, with button 1 or 2
        if (ebut2 || ebut1) && button3 == 0 && self.shifting_cont == 0 {
            S_MOVE_BAND_LASSO.set(0);
            if self.lasso_on && !self.drawing_lasso {
                let (mut mx, mut my) = (0., 0.);
                self.getixy(x, y, &mut mx, &mut my, &mut dxll);
                mpt.x = mx;
                mpt.y = my;
                mpt.z = dxll as f32;
                cont = self.get_lasso_contour();
                if !cont.is_null()
                    && imod_point_cont_distance(unsafe { &*cont }, &mpt, 0, 0, &mut dxll)
                        * self.zoom
                        < rcrit as f32
                {
                    S_MOVE_BAND_LASSO.set(1);
                    self.set_cursor(self.mousemode, false);
                    return;
                }
            }
        }
        let set_anyway = with_boundary(|n| n.util_need_to_set_cursor());
        self.set_cursor(self.mousemode, set_anyway);

        // Now give the plugins a crack at it
        if !self.drawing_lasso && !self.drawing_arrow {
            ifdraw = self.check_plug_use_mouse(event, button1, button2, button3);
        }
        if ifdraw & 1 != 0 {
            return;
        }

        // Check for regular actions
        if ebut1 && !self.drawing_lasso {
            if self.shifting_cont != 0 {
                drew = self.start_shifting_contour(S_FIRSTMX.get(), S_FIRSTMY.get(), 1, ctrl_down);
            } else if self.starting_band != 0 || self.drawing_arrow {
                drew = self.b1_click(S_FIRSTMX.get(), S_FIRSTMY.get(), ctrl_down);
            } else {
                S_FIRST_DRAG.set(1);
            }
        } else if (self.drawing_lasso && (ebut1 || ebut2))
            || (ebut2 && button1 == 0 && button3 == 0)
        {
            if self.shifting_cont != 0 {
                drew = self.start_shifting_contour(x, y, 2, ctrl_down);
            } else {
                drew = self.b2_click(x, y, ctrl_down);
            }
        } else if event.button == but3 && button1 == 0 && button2 == 0 {
            if self.shifting_cont != 0 {
                drew = self.start_shifting_contour(x, y, 3, ctrl_down);
            } else {
                drew = self.b3_click(x, y, ctrl_down);
            }
        }
        if ifdraw != 0 && drew == 0 {
            self.draw();
        }
    }

    /// `ZapFuncs::mouseRelease` (`xzap.cpp:2241`).
    pub fn mouse_release(&mut self, event: &KeyEvent) {
        let mut imz = 0;
        let mut ifdraw = 0;
        let mut drew = 0;
        let but1 = with_boundary(|n| n.prefs_actual_button(1)) as u32;
        let but2 = with_boundary(|n| n.prefs_actual_button(2)) as u32;
        let but3 = with_boundary(|n| n.prefs_actual_button(3)) as u32;
        let button1 = i32::from(event.button == but1);
        let button2 = i32::from(event.button == but2);
        let button3 = i32::from(event.button == but3);
        S_MOUSE_PRESSED.set(false);
        let release_band = (((button2 != 0 && self.rubberband != 0)
            || ((button1 != 0 || button2 != 0) && self.lasso_on))
            && S_MOVE_BAND_LASSO.get() != 0)
            || (button1 != 0 && self.drawing_arrow);
        let ex = (event.x as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        let ey = (event.y as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;

        if imod_debug('m') {
            imod_print_stderr(&format!(
                "release at {ex} {ey}   buttons {button1} {button2} {button3}\n"
            ));
        }
        if self.shift_registered != 0 {
            self.shift_registered = 0;
            undo_finish_unit(unsafe { &mut *(*self.vi).undo }, unsafe {
                &*(*self.vi).imod
            });
        }
        let need_draw =
            self.drew_extra_cursor && !with_boundary(|n| n.gfx_extra_cursor_in_window());

        if !self.drawing_lasso && !release_band {
            ifdraw = self.check_plug_use_mouse(event, button1, button2, button3);
        }
        if ifdraw & 1 != 0 && ifdraw & 2 == 0 && need_draw {
            self.draw();

            // Defer the return so the band moving can be turned off, but then
            // only check other things below if this flag is off
        }

        if button1 != 0 && ifdraw & 1 == 0 && !self.drawing_lasso && !release_band {
            if S_DRAG_BAND_LASSO.get() != 0 {
                S_DRAG_BAND_LASSO.set(0);
                self.set_cursor(self.mousemode, false);
            }
            S_FIRST_DRAG.set(0);

            let elapsed = S_BUT1_DOWNT.with(|t| t.borrow().elapsed().as_millis() as i32);
            if imod_debug('m') {
                imod_print_stderr(&format!("Down time {elapsed} msec  {}\n", self.hqgfxsave));
            }
            if elapsed > 250 {
                if self.hqgfxsave != 0 || ifdraw != 0 {
                    self.draw();
                }
                self.hqgfxsave = 0;
                return; //IS THIS RIGHT?
            }
            let ctrl = with_boundary(|n| {
                n.input_test_ctrl(if event.control { CONTROL_MODIFIER } else { 0 })
            });
            drew = self.b1_click(ex, ey, ctrl);
        }

        // Button 2 and band moving, release the band
        if release_band {
            S_MOVE_BAND_LASSO.set(0);
            self.drawing_arrow = false;
            self.set_cursor(self.mousemode, false);

            // Do a full draw for lasso move because isosurface may need
            // updating
            if self.hqgfxsave != 0 || self.lasso_on || self.arrow_on {
                if self.lasso_on {
                    let vi = self.vi;
                    with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_MOD | IMOD_DRAW_XYZ));
                } else {
                    self.draw();
                }
                drew = 1;
            }
            self.hqgfxsave = 0;

        // Button 2 and doing a drag draw - draw for real.
        } else if (button2 != 0 || (button1 != 0 && self.drawing_lasso))
            && self.num_xpanels == 0
            && ifdraw & 1 == 0
            && (unsafe { (*(*self.vi).imod).mousemode } == IMOD_MMODEL || self.drawing_lasso)
        {
            if imod_debug('z') {
                let elapsed = S_BUT1_DOWNT.with(|t| t.borrow().elapsed().as_millis() as i32);
                imod_print_stderr(&format!("Down time {elapsed} msec\n"));
            }

            if self.drawing_lasso {
                let obj = ivw_get_an_extra_object(unsafe { &mut *self.vi }, self.lasso_obj_num)
                    .map_or(ptr::null_mut(), |o| o as *mut Iobj);
                if obj.is_null() || unsafe { (&(*obj).cont)[0].pts.len() } < 2 {
                    self.toggle_lasso(true);
                } else {
                    self.drawing_lasso = false;
                    let cont = unsafe { (&mut (*obj).cont).as_mut_ptr() };
                    crate::imod::libcfshr::b3dutil::set_or_clear_flags(
                        unsafe { &mut (*cont).flags },
                        ICONT_OPEN,
                        0,
                    );
                    crate::imod::three_dmod::imod_edit::imod_trim_contour_loops(
                        unsafe { &mut *cont },
                        0,
                    );
                    for ind in 0..2 {
                        let length = unsafe { (*self.vi).xybin } as f32
                            * imod_contour_length(unsafe { cont.as_ref() }, ind);
                        let str = format!(
                            "{} length {:.1} {}pixels",
                            if ind != 0 { "Closed" } else { "Open" },
                            length,
                            if unsafe { (*self.vi).xybin } > 1 {
                                "unbin "
                            } else {
                                ""
                            }
                        );
                        let imod = unsafe { (*self.vi).imod };
                        with_boundary(|n| n.util_wprint_measure(&str, imod, length, false));
                    }
                    let area = unsafe { (*self.vi).xybin } as f32
                        * unsafe { (*self.vi).xybin } as f32
                        * imod_contour_area(unsafe { cont.as_ref() });
                    let str = format!(
                        "Area {} {}pixels^2",
                        area,
                        if unsafe { (*self.vi).xybin } > 1 {
                            "unbin "
                        } else {
                            ""
                        }
                    );
                    let imod = unsafe { (*self.vi).imod };
                    with_boundary(|n| n.util_wprint_measure(&str, imod, area, true));

                    self.set_mouse_tracking();
                    self.set_cursor(self.mousemode, true);
                    if with_boundary(|n| n.imodv_isosurface_update(IMOD_DRAW_MOD)) {
                        with_boundary(|n| n.imodv_draw());
                    }
                }
            } else {
                self.register_drag_additions();
            }

            // Fix the mouse position and update the other windows finally
            // Why call imod_info_setxyz again on release of single point add?
            let (mut imx, mut imy) = (0., 0.);
            self.getixy(ex, ey, &mut imx, &mut imy, &mut imz);
            unsafe { (*self.vi).xmouse = imx };
            unsafe { (*self.vi).ymouse = imy };
            if self.draw_current_only != 0 {
                self.set_draw_current_only(0);
                let vi = self.vi;
                with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_MOD | IMOD_DRAW_XYZ));
                drew = 1;
            } else {
                with_boundary(|n| n.imod_info_setxyz());
            }
        }

        // Now return if plugin said to
        if ifdraw & 1 != 0 {
            return;
        }

        if self.center_marked != 0 && self.center_defined == 0 {
            ivw_clear_an_extra_object(unsafe { &mut *self.vi }, self.shift_obj_num);
            let vi = self.vi;
            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_MOD));
            self.center_marked = 0;
            drew = 1;
        }
        if (ifdraw != 0 || need_draw) && drew == 0 {
            self.draw();
        } else {
            self.set_area_limits();
        }
    }

    /// `ZapFuncs::mouseMove` (`xzap.cpp:2375`); respond to a mouse move event
    /// (mouse down).
    pub fn mouse_move(&mut self, event: &KeyEvent) {
        let mut imz = 0;
        let mut ifdraw = 0;
        let mut drew = 0;
        let cumthresh = 6 * 6;
        let dragthresh = 10 * 10;
        let moving_band_lasso = ((self.rubberband != 0 || self.lasso_on)
            && S_MOVE_BAND_LASSO.get() != 0)
            || self.drawing_arrow;

        // Record state of event and then return if eating move events
        let ctrl_down =
            with_boundary(|n| n.input_test_ctrl(if event.control { CONTROL_MODIFIER } else { 0 }));
        let shift_down = i32::from(event.shift);
        let ex = (event.x as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        let ey = (event.y as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        let but1 = with_boundary(|n| n.prefs_actual_button(1)) as u32;
        let but2 = with_boundary(|n| n.prefs_actual_button(2)) as u32;
        let but3 = with_boundary(|n| n.prefs_actual_button(3)) as u32;
        let mut button1 = i32::from(event.buttons & but1 != 0);
        let mut button2 = i32::from(event.buttons & but2 != 0);
        let mut button3 = i32::from(event.buttons & but3 != 0);
        if S_MOVE_PROCESSING.get() != 0 {
            S_MOVE_PROCESSING.set(S_MOVE_PROCESSING.get() + 1);
            return;
        }

        if S_PIXEL_VIEW_OPEN.get() {
            let (mut imx, mut imy) = (0., 0.);
            self.getixy(ex, ey, &mut imx, &mut imy, &mut imz);
            let vi = self.vi;
            with_boundary(|n| n.pv_new_mouse_position(vi, imx, imy, imz));
        }

        if !moving_band_lasso
            && (S_MOUSE_PRESSED.get()
                || S_INSERT_DOWN.get() != 0
                || unsafe { (*self.vi).track_mouse_for_plugs } != 0)
        {
            S_MOVE_PROCESSING.set(1);
            ifdraw = self.check_plug_use_mouse(event, button1, button2, button3);
            S_MOVE_PROCESSING.set(0);
            if ifdraw & 1 != 0 {
                return;
            }
        } else if S_MOUSE_PRESSED.get() || S_INSERT_DOWN.get() != 0 {
            self.set_control_and_limits();
        }

        if !(S_MOUSE_PRESSED.get() || S_INSERT_DOWN.get() != 0) {
            if (self.rubberband != 0 || (self.lasso_on && !self.drawing_lasso))
                && self.shifting_cont == 0
            {
                self.analyze_band_edge(ex, ey);
            }
            if ifdraw != 0 {
                self.draw();
            }
            return;
        }

        // For first button or band moving, eat any pending move events and use
        // latest position
        if (button1 != 0 && !self.drawing_lasso && button2 == 0 && button3 == 0)
            || moving_band_lasso
        {
            S_MOVE_PROCESSING.set(1);
            with_boundary(|n| n.imod_info_input());
            if imod_debug('m') && S_MOVE_PROCESSING.get() > 1 {
                imod_print_stderr(&format!(
                    "Flushed {} move events\n",
                    S_MOVE_PROCESSING.get() - 1
                ));
            }
            S_MOVE_PROCESSING.set(0);

            // If this processed a release, then turn off the buttons
            if !S_MOUSE_PRESSED.get() {
                button1 = 0;
                button2 = 0;
                button3 = 0;
            }
        }

        // Commit to using these values in case new values come in during
        // operations
        let use_x = ex;
        let use_y = ey;
        let cumdx = use_x - S_FIRSTMX.get();
        let cumdy = use_y - S_FIRSTMY.get();
        S_MOVE_PROCESSING.set(1);

        button2 = i32::from(button2 != 0 || S_INSERT_DOWN.get() != 0);
        if imod_debug('m') {
            imod_print_stderr(&format!(
                "move {use_x},{use_y}  mb  {button1}|{button2}|{button3}  c {ctrl_down:x} s {shift_down:x}\n"
            ));
        }

        if !moving_band_lasso
            && (button1 != 0 && !self.drawing_lasso)
            && button2 == 0
            && button3 == 0
        {
            if ctrl_down != 0 {
                drew = self.drag_select_conts_crossed(use_x, use_y);
            } else {
                /* DNM: wait for a bit of time or until enough distance moved,
                but if we do not replace original lmx, lmy, there is a
                disconcerting lurch */
                let elapsed = S_BUT1_DOWNT.with(|t| t.borrow().elapsed().as_millis() as i32);
                if elapsed > 250 || cumdx * cumdx + cumdy * cumdy > cumthresh {
                    drew = self.b1_drag(use_x, use_y);
                }
            }
        }

        // DNM 8/1/08: Reject small movements soon after the button press
        if (((self.drawing_lasso
            || self.drawing_arrow
            || (self.lasso_on && S_MOVE_BAND_LASSO.get() != 0))
            && (button1 != 0 || button2 != 0))
            || (button1 == 0 && button2 != 0))
            && button3 == 0
        {
            let elapsed = S_BUT1_DOWNT.with(|t| t.borrow().elapsed().as_millis() as i32);
            if elapsed > 150 || cumdx * cumdx + cumdy * cumdy > dragthresh {
                drew = self.b2_drag(use_x, use_y, ctrl_down);
            }
        }

        if button1 == 0 && button2 == 0 && button3 != 0 {
            drew = self.b3_drag(use_x, use_y, ctrl_down, shift_down);
        }

        self.lmx = use_x;
        self.lmy = use_y;
        if ifdraw != 0 && drew == 0 {
            self.draw();
        }
        S_MOVE_PROCESSING.set(0);
    }

    /// `ZapFuncs::checkPlugUseMouse` (`xzap.cpp:2482`); test for whether a
    /// plugin takes care of the mouse event, taking care of setting limits.
    pub fn check_plug_use_mouse(
        &mut self,
        event: &KeyEvent,
        but1: i32,
        but2: i32,
        but3: i32,
    ) -> i32 {
        let mut imz = 0;
        let (mut imx, mut imy) = (0., 0.);
        let set_control = S_MOUSE_PRESSED.get()
            || S_INSERT_DOWN.get() != 0
            || but1 != 0
            || but2 != 0
            || but3 != 0;
        if set_control {
            self.set_control_and_limits();
        }
        if self.num_xpanels != 0 {
            return 0;
        }
        if set_control {
            ivw_control_active(unsafe { &mut *self.vi }, 0);
        }
        let ex = (event.x as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        let ey = (event.y as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        self.getixy(ex, ey, &mut imx, &mut imy, &mut imz);
        let vi = self.vi;
        let ifdraw = with_boundary(|n| {
            n.imod_plug_handle_mouse(vi, event, imx, imy, but1, but2, but3, ZAP_WINDOW_TYPE)
        });
        if ifdraw & 1 != 0 {
            if ifdraw & 2 != 0 {
                self.draw();
            }
        } else if set_control {
            ivw_control_active(unsafe { &mut *self.vi }, 1);
        }
        ifdraw
    }

    /// `ZapFuncs::analyzeBandEdge` (`xzap.cpp:2509`); analyze for whether
    /// mouse is close to a corner or an edge and set flags for the cursor.
    pub fn analyze_band_edge(&mut self, ix: i32, iy: i32) {
        let rubbercrit = 10; /* Criterion distance for grabbing the band */
        let mut i = 0;
        let mut dxll = 0;
        let mut mpt = Ipoint::default();

        self.band_image_to_mouse(0);
        S_DRAG_BAND_LASSO.set(0);
        S_DRAGGING.set([0; 4]);

        if self.lasso_on {
            let cont = self.get_lasso_contour();
            let (mut mx, mut my) = (0., 0.);
            self.getixy(ix, iy, &mut mx, &mut my, &mut i);
            mpt.x = mx;
            mpt.y = my;
            mpt.z = i as f32;
            if !cont.is_null()
                && imod_point_cont_distance(unsafe { &*cont }, &mpt, 0, 0, &mut dxll) * self.zoom
                    < rubbercrit as f32
            {
                S_DRAG_BAND_LASSO.set(1);
            }
        } else {
            let mut drag_band = S_DRAG_BAND_LASSO.get();
            let mut dragging = S_DRAGGING.get();
            util_analyze_band_edge(
                ix,
                iy,
                [
                    self.rb_mouse_x0,
                    self.rb_mouse_x1,
                    self.rb_mouse_y0,
                    self.rb_mouse_y1,
                ],
                &mut drag_band,
                &mut dragging,
            );
            S_DRAG_BAND_LASSO.set(drag_band);
            S_DRAGGING.set(dragging);
        }
        self.set_cursor(self.mousemode, false);
    }

    /// `ZapFuncs::bandMinimum` (`xzap.cpp:2538`); adjust minimum size of
    /// rubberband down in case image is tiny.
    pub fn band_minimum(&self) -> i32 {
        let mut bandmin = 4.min((unsafe { (*self.vi).xsize } as f32 * self.xzoom) as i32 + 2);
        bandmin = bandmin.min((unsafe { (*self.vi).ysize } as f32 * self.zoom) as i32 + 2);
        bandmin
    }

    /// `ZapFuncs::b1Click` (`xzap.cpp:2549`); attach to nearest point in
    /// model mode, or just modify the current xmouse, ymouse values.
    pub fn b1_click(&mut self, x: i32, y: i32, control_down: i32) -> i32 {
        let vi = self.vi;
        let imod = unsafe { (*vi).imod };
        let mut pnt = Ipoint::default();
        let mut index = Iindex {
            object: 0,
            contour: 0,
            point: 0,
        };
        let mut iz = 0;
        let (mut ix, mut iy) = (0., 0.);
        let selsize = IMOD_SELSIZE / self.zoom;

        self.getixy(x, y, &mut ix, &mut iy, &mut iz);
        if iz < 0 {
            return 0;
        }

        // If starting rubber band, just record these coordinates
        if self.starting_band != 0 {
            self.rb_mouse_x0 = x;
            self.rb_mouse_y0 = y;
            return 0;
        }

        // If drawing an arrow, start the coordinates
        if self.drawing_arrow {
            iz = self.arrow_head.len() as i32 - 1;
            self.arrow_head[iz as usize].x = ix;
            self.arrow_tail[iz as usize].x = ix;
            self.arrow_head[iz as usize].y = iy;
            self.arrow_tail[iz as usize].y = iy;
            return 0;
        }

        if !unsafe { (*vi).ax }.is_null()
            && self.num_xpanels == 0
            && with_boundary(|n| n.autox_altmouse(vi)) == AUTOX_ALTMOUSE_PAINT
        {
            with_boundary(|n| n.autox_fillmouse(vi, ix as i32, iy as i32));
            return 1;
        }

        // In either mode, do a default modification of zmouse or mSection
        unsafe { (*vi).xmouse = ix };
        unsafe { (*vi).ymouse = iy };
        if self.lock != 0 {
            self.section = iz;
        } else {
            unsafe { (*vi).zmouse = iz as f32 };
        }

        if unsafe { (*(*vi).imod).mousemode } == IMOD_MMODEL {
            pnt.x = ix;
            pnt.y = iy;
            pnt.z = iz as f32;
            let ind_save = unsafe { (*(*vi).imod).cindex };

            let time = ivw_window_time(unsafe { &*vi }, self.time_lock);
            let distance =
                with_boundary(|n| n.imod_all_obj_nearest(vi, &mut index, &pnt, selsize, time));

            // If point found, manage selection list and even toggle this
            // contour off if appropriate
            if distance >= 0. {
                with_boundary(|n| n.imod_selection_new_cur_point(vi, imod, ind_save, control_down));
            }

            /* DNM: add the DRAW_XYZ flag to make it update info and Slicer */
            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_RETHINK | IMOD_DRAW_XYZ));
            return 1;
        }

        with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_XYZ));
        1
    }

    /// `ZapFuncs::b2Click` (`xzap.cpp:2619`); in model mode, add a model
    /// point, creating a new contour if necessary.
    pub fn b2_click(&mut self, x: i32, y: i32, control_down: i32) -> i32 {
        let vi = self.vi;
        let mut obj: *mut Iobj;
        let mut cont: *mut Icont;
        let mut point = Ipoint::default();
        let mut pt;
        let (mut ix, mut iy) = (0., 0.);
        let lastz;
        let mut iz = 0;
        let mut new_surf;
        let time = ivw_window_time(unsafe { &*vi }, self.time_lock);

        self.getixy(x, y, &mut ix, &mut iy, &mut iz);

        if !unsafe { (*vi).ax }.is_null()
            && self.num_xpanels == 0
            && with_boundary(|n| n.autox_altmouse(vi)) == AUTOX_ALTMOUSE_PAINT
        {
            /* DNM 2/1/01: need to call with int */
            with_boundary(|n| n.autox_sethigh(vi, ix as i32, iy as i32));
            return 1;
        }

        // 3/5/08: Moved band moving to mouse press routine so it would happen
        // before plugin action

        if unsafe { (*(*vi).imod).mousemode } == IMOD_MMODEL || self.drawing_lasso {
            if self.num_xpanels != 0 {
                return 0;
            }
            self.drag_add_count = 0;

            if self.drawing_lasso {
                obj = ivw_get_an_extra_object(unsafe { &mut *vi }, self.lasso_obj_num)
                    .map_or(ptr::null_mut(), |o| o as *mut Iobj);
                if obj.is_null() {
                    self.toggle_lasso(true);
                    return 0;
                }
                imod_object_set_color(unsafe { &mut *obj }, 1., 0., 0.);
                unsafe { (*obj).flags &= !IMOD_OBJFLAG_SCAT };
                unsafe { (*obj).pdrawsize = 0 };
                unsafe { (*obj).linewidth2 = 1 };
                unsafe { (*obj).extra[IOBJ_EX_LASSO_ID] = self.ctrl as u32 };
                let Some(new_cont) = imod_contour_new() else {
                    self.toggle_lasso(true);
                    return 0;
                };
                imod_object_add_contour(unsafe { &mut *obj }, new_cont);
                cont = unsafe { (&mut (*obj).cont).as_mut_ptr() };
                unsafe { (*cont).flags |= ICONT_DRAW_ALLZ | ICONT_STIPPLED | ICONT_OPEN };
            } else {
                obj = imod_object_get(unsafe { (*vi).imod.as_ref() })
                    .map_or(ptr::null_mut(), |o| o as *const Iobj as *mut Iobj);
                if obj.is_null() {
                    return 0;
                }

                // Get current contour; if there is none, start a new one
                // DNM 7/10/04: switch to calling routine; it now fixes time of
                // empty cont
                cont = unsafe { ivw_get_or_make_contour(vi, obj, self.time_lock) };
                if cont.is_null() {
                    return 0;
                }
            }

            point.x = ix;
            point.y = iy;
            point.z = self.section as f32;
            if self.twod != 0 && !cont.is_null() && unsafe { !(&(*cont).pts).is_empty() } {
                point.z = unsafe { (&(*cont).pts)[0].z };
            }
            unsafe { (*vi).xmouse = ix };
            unsafe { (*vi).ymouse = iy };

            if self.drawing_lasso {
                imod_point_append(unsafe { &mut *cont }, point);
            } else {
                // Get a new surface for planar modeling if the current surface
                // is planar in X or Y
                new_surf = INCOS_NEW_CONT;
                if iobj_planar(unsafe { (*obj).flags }) != 0
                    && (unsafe { (&(*cont).pts).is_empty() }
                        || unsafe { (*cont).flags } & ICONT_WILD != 0)
                    && unsafe { (*cont).surf } != 0
                    && (crate::imod::three_dmod::imod_edit::imod_surface_is_planar(
                        unsafe { &*obj },
                        unsafe { (*cont).surf },
                        time,
                        X_SLICE_BOX,
                    ) > 0
                        || crate::imod::three_dmod::imod_edit::imod_surface_is_planar(
                            unsafe { &*obj },
                            unsafe { (*cont).surf },
                            time,
                            Y_SLICE_BOX,
                        ) > 0)
                {
                    new_surf = crate::imod::three_dmod::imod_edit::imod_check_surf_for_new_cont(
                        unsafe { &*obj },
                        unsafe { cont.as_ref() },
                        time,
                        Z_SLICE_BOX,
                    );
                }

                /* If contours are closed and Z has changed, start a new
                contour.  Also check for a change in time, if time data are
                being modeled, and start new contour for any kind of contour */
                // DNM 7/10/04: just use first point instead of current point
                if unsafe { !(&(*cont).pts).is_empty() } {
                    let time_mismatch = unsafe { ivw_time_mismatch(vi, self.time_lock, obj, cont) };
                    let not_in_plane = iobj_planar(unsafe { (*obj).flags }) != 0
                        && unsafe { (*cont).flags } & ICONT_WILD == 0
                        && (unsafe { (&(*cont).pts)[0].z } as f64 + 0.5).floor() as i32
                            != point.z as i32;
                    if not_in_plane || time_mismatch || new_surf != INCOS_NEW_CONT {
                        let time_lock = self.time_lock;
                        cont = with_boundary(|n| {
                            n.util_auto_new_contour(
                                vi,
                                cont,
                                not_in_plane,
                                time_mismatch,
                                time_lock,
                                new_surf,
                            )
                        });
                        if cont.is_null() {
                            return 0;
                        }
                    }
                } else if new_surf != INCOS_NEW_CONT {
                    with_boundary(|n| n.util_assign_surf_to_cont(vi, obj, cont, new_surf));
                }

                /* Now if times still don't match refuse the point */
                if unsafe { ivw_time_mismatch(vi, self.time_lock, obj, cont) } {
                    wprint(
                        "\u{7}Contour time does not match current time.\nSet contour time to 0 to model across times.\n",
                    );
                    undo_finish_unit(unsafe { &mut *(*vi).undo }, unsafe { &*(*vi).imod });
                    return 0;
                }

                // DNM 11/17/04: Cleaned up adding point logic to set an
                // insertion point and just call InsertPoint with it
                // Set insertion point to next point and adjust it down if
                // going backwards
                pt = unsafe { (*(*vi).imod).cindex.point } + 1;
                if pt > 0 && unsafe { !(&(*cont).pts).is_empty() } {
                    lastz = unsafe { (&(*cont).pts)[(pt - 1) as usize].z };
                } else {
                    lastz = point.z;
                }

                if pt > 0 && self.insertmode != 0 {
                    pt -= 1;
                }

                unsafe { ivw_register_insert_point(vi, cont, &mut point, pt) };

                /* DNM: auto section advance is based on the direction of
                section change between last and just-inserted points */
                if self.section_step != 0 && point.z != lastz {
                    if point.z - lastz > 0.0 {
                        unsafe { (*vi).zmouse += 1.0 };
                    } else {
                        unsafe { (*vi).zmouse -= 1.0 };
                    }

                    if unsafe { (*vi).zmouse } < 0.0 {
                        unsafe { (*vi).zmouse = 0. };
                    }
                    if unsafe { (*vi).zmouse } > unsafe { (*vi).zsize } as f32 - 1. {
                        unsafe { (*vi).zmouse = (*vi).zsize as f32 - 1. };
                    }
                }
            }
            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_MOD | IMOD_DRAW_XYZ));
            /* DNM 5/22/03: sync all but the active window when single point
            added */
            return 1;
        }
        self.start_movie_check_snap(1)
    }

    /// `ZapFuncs::delUnderCursor` (`xzap.cpp:2773`); delete all points of
    /// current contour under the cursor.
    pub fn del_under_cursor(&mut self, x: i32, y: i32, cont: *mut Icont) -> i32 {
        let (mut ix, mut iy) = (0., 0.);
        let crit = 8. / self.zoom;
        let mut iz = 0;
        let mut deleted = 0;

        self.getixy(x, y, &mut ix, &mut iy, &mut iz);
        let critsq = crit * crit;
        let mut i = 0;
        while i < unsafe { (&(*cont).pts).len() as i32 } && unsafe { (&(*cont).pts).len() } > 1 {
            let lpt = unsafe { (&(*cont).pts)[i as usize] };
            if (lpt.z as f64 + 0.5).floor() as i32 == self.section {
                let dsq = (lpt.x - ix) * (lpt.x - ix) + (lpt.y - iy) * (lpt.y - iy);
                if dsq <= critsq {
                    undo_point_removal(
                        unsafe { &mut *(*self.vi).undo },
                        unsafe { &mut *(*self.vi).imod },
                        i,
                    );
                    imod_point_delete(unsafe { &mut *cont }, i);
                    unsafe {
                        (*(*self.vi).imod).cindex.point = ((&(*cont).pts).len() as i32 - 1)
                            .min((i + self.insertmode as i32 - 1).max(0))
                    };
                    deleted = 1;
                    continue;
                }
            }
            i += 1;
        }
        if deleted == 0 {
            return 0;
        }
        undo_finish_unit(unsafe { &mut *(*self.vi).undo }, unsafe {
            &*(*self.vi).imod
        });
        let vi = self.vi;
        with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_XYZ | IMOD_DRAW_MOD));
        1
    }

    /// `ZapFuncs::b3Click` (`xzap.cpp:2811`); in model mode, modify current
    /// point; otherwise run movie.
    pub fn b3_click(&mut self, x: i32, y: i32, control_down: i32) -> i32 {
        let vi = self.vi;
        let (mut ix, mut iy) = (0., 0.);
        let mut iz = 0;

        self.getixy(x, y, &mut ix, &mut iy, &mut iz);

        if !unsafe { (*vi).ax }.is_null()
            && self.num_xpanels == 0
            && with_boundary(|n| n.autox_altmouse(vi)) == AUTOX_ALTMOUSE_PAINT
        {
            /* DNM 2/1/01: need to call with int */
            with_boundary(|n| n.autox_setlow(vi, ix as i32, iy as i32));
            return 1;
        }

        if unsafe { (*(*vi).imod).mousemode } == IMOD_MMODEL {
            if self.num_xpanels != 0 {
                return 0;
            }
            let cont = imod_contour_get(unsafe { (*vi).imod.as_ref() })
                .map_or(ptr::null_mut(), |c| c as *const Icont as *mut Icont);
            let pt = unsafe { (*(*vi).imod).cindex.point };
            if cont.is_null() {
                return 0;
            }
            if pt < 0 {
                return 0;
            }

            let obj = imod_object_get(unsafe { (*vi).imod.as_ref() })
                .map_or(ptr::null_mut(), |o| o as *const Iobj as *mut Iobj);
            if unsafe { ivw_time_mismatch(vi, self.time_lock, obj, cont) } {
                return 0;
            }

            /* If the control key is down, delete points under the cursor */
            if control_down != 0 {
                return self.del_under_cursor(x, y, cont);
            }

            if self.point_visable(unsafe { &(&(*cont).pts)[pt as usize] }) == 0 {
                return 0;
            }

            undo_point_shift_cp(unsafe { &mut *(*vi).undo }, unsafe { &mut *(*vi).imod });
            unsafe { (&mut (*cont).pts)[pt as usize].x = ix };
            unsafe { (&mut (*cont).pts)[pt as usize].y = iy };
            undo_finish_unit(unsafe { &mut *(*vi).undo }, unsafe { &*(*vi).imod });

            unsafe { (*vi).xmouse = ix };
            unsafe { (*vi).ymouse = iy };

            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_RETHINK));
            return 1;
        }
        self.start_movie_check_snap(-1)
    }

    /// `ZapFuncs::b1Drag` (`xzap.cpp:2868`).
    pub fn b1_drag(&mut self, mut x: i32, mut y: i32) -> i32 {
        // For zooms less than one, move image along with mouse; for higher
        // zooms, translate 1 image pixel per mouse pixel (accelerated)
        let trans_fac: f64 = if self.zoom < 1. {
            1. / self.zoom as f64
        } else {
            1.
        };
        let cancel_hq = self.last_hq_draw_time > S_HQ_DRAW_TIME_CRIT.get();
        let bandmin = self.band_minimum();
        let (mut dx, mut dy);

        if self.shifting_cont != 0 {
            self.shift_contour(x, y, 1, 0);
            return 1;
        }

        // If we are still starting band, first find out if the mouse has moved
        // far enough to commit to a direction
        if self.starting_band != 0 {
            let mut rb = [
                self.rb_mouse_x0,
                self.rb_mouse_x1,
                self.rb_mouse_y0,
                self.rb_mouse_y1,
            ];
            let mut dragging = S_DRAGGING.get();
            let committed =
                util_is_band_committed(x, y, self.winx, self.winy, bandmin, &mut rb, &mut dragging);
            self.rb_mouse_x0 = rb[0];
            self.rb_mouse_x1 = rb[1];
            self.rb_mouse_y0 = rb[2];
            self.rb_mouse_y1 = rb[3];
            S_DRAGGING.set(dragging);
            if committed == 0 {
                return 0;
            }

            self.band_mouse_to_image(0);

            // Does image coord need to be moved?  Do so and move mouse coords
            dx = 0.;
            dy = 0.;
            if self.rb_image_x0 < 0. {
                dx = -self.rb_image_x0;
            }
            if self.rb_image_x1 > unsafe { (*self.vi).xsize } as f32 {
                dx = unsafe { (*self.vi).xsize } as f32 - self.rb_image_x1;
            }
            if self.rb_image_y0 < 0. {
                dy = -self.rb_image_y0;
            }
            if self.rb_image_y1 > unsafe { (*self.vi).ysize } as f32 {
                dy = unsafe { (*self.vi).ysize } as f32 - self.rb_image_y1;
            }
            if dx != 0. || dy != 0. {
                self.rb_image_x0 += dx;
                self.rb_image_x1 += dx;
                self.rb_image_y0 += dy;
                self.rb_image_y1 += dy;
                self.band_image_to_mouse(1);
                x = if S_DRAGGING.get()[0] != 0 {
                    self.rb_mouse_x0
                } else {
                    self.rb_mouse_x1
                };
                y = if S_DRAGGING.get()[2] != 0 {
                    self.rb_mouse_y0
                } else {
                    self.rb_mouse_y1
                };
                let cursx = (x as f64 / self.device_pixel_ratio as f64 + 0.5).floor() as i32;
                let cursy = (y as f64 / self.device_pixel_ratio as f64 + 0.5).floor() as i32;
                with_boundary(|n| n.gfx_set_cursor_pos(cursx, cursy));
                self.lmx = x;
                self.lmy = y;
            }

            // Set flags for the band being on and being dragged
            self.starting_band = 0;
            self.rubberband = 1;
            self.set_mouse_tracking();
            S_DRAG_BAND_LASSO.set(1);
            self.band_changed = 1;
            self.set_cursor(self.mousemode, false);
            self.draw();
            return 1;
        }

        // First time mouse moves, lock in the band drag position
        if self.rubberband != 0 && S_FIRST_DRAG.get() != 0 {
            self.analyze_band_edge(x, y);
        }
        S_FIRST_DRAG.set(0);

        if self.rubberband != 0 && S_DRAG_BAND_LASSO.get() != 0 {
            /* Move the rubber band */
            // Keep them within limit but swap coordinates and drag flag if
            // they cross
            let mut dragging = S_DRAGGING.get();
            let (mut ix0, mut ix1) = (self.rb_image_x0, self.rb_image_x1);
            let (mut d0, mut d1) = (dragging[0], dragging[1]);
            let xsize = unsafe { (*self.vi).xsize };
            self.drag_two_band_sides(&mut ix0, &mut ix1, &mut d0, &mut d1, x - self.lmx, xsize);
            self.rb_image_x0 = ix0;
            self.rb_image_x1 = ix1;
            dragging[0] = d0;
            dragging[1] = d1;
            let (mut iy0, mut iy1) = (self.rb_image_y0, self.rb_image_y1);
            let (mut d3, mut d2) = (dragging[3], dragging[2]);
            let ysize = unsafe { (*self.vi).ysize };
            self.drag_two_band_sides(&mut iy0, &mut iy1, &mut d3, &mut d2, self.lmy - y, ysize);
            self.rb_image_y0 = iy0;
            self.rb_image_y1 = iy1;
            dragging[3] = d3;
            dragging[2] = d2;
            S_DRAGGING.set(dragging);
            self.band_changed = 1;
            let set_anyway = with_boundary(|n| n.util_need_to_set_cursor());
            self.set_cursor(self.mousemode, set_anyway);
        } else {
            /* Move the image */
            if imod_debug('m') {
                imod_print_stderr(&format!(
                    "B1Drag: x,y {x},{y}  lmx,y {},{}  trans {},{}\n",
                    self.lmx, self.lmy, self.xtrans, self.ytrans
                ));
            }
            self.xtrans += (trans_fac * (x - self.lmx) as f64 + 0.5).floor() as i32;
            self.ytrans -= (trans_fac * (y - self.lmy) as f64 + 0.5).floor() as i32;
        }

        if cancel_hq {
            self.hqgfxsave = self.hqgfx;
            self.hqgfx = 0;
        }
        self.draw();
        if cancel_hq {
            self.hqgfx = self.hqgfxsave;
        }
        1
    }

    /// `ZapFuncs::dragTwoBandSides` (`xzap.cpp:2967`); operates on rubber band
    /// image coordinates in X or in Y, allowing the two sides to cross.
    pub fn drag_two_band_sides(
        &mut self,
        image0: &mut f32,
        image1: &mut f32,
        drag0: &mut i32,
        drag1: &mut i32,
        delta: i32,
        size: i32,
    ) {
        if *drag0 != 0 {
            *image0 += delta as f32 / self.xzoom;
            if (*image0 - *image1).abs() < 0.5 {
                if delta > 0 {
                    *image0 += if *image0 < size as f32 { 1. } else { -1. };
                } else {
                    *image0 += if *image0 > 0. { -1. } else { 1. };
                }
            }
            *image0 = image0.clamp(0., size as f32);
        }
        if *drag1 != 0 {
            *image1 += delta as f32 / self.xzoom;
            if (*image0 - *image1).abs() < 0.5 {
                if delta > 0 {
                    *image1 += if *image1 < size as f32 { 1. } else { -1. };
                } else {
                    *image1 += if *image1 > 0. { -1. } else { 1. };
                }
            }
            *image1 = image1.clamp(0., size as f32);
        }
        if *image0 > *image1 && (*drag0 != 0 || *drag1 != 0) {
            std::mem::swap(image0, image1);
            std::mem::swap(drag0, drag1);
        }
    }

    /// `ZapFuncs::dragSelectContsCrossed` (`xzap.cpp:3001`); select contours
    /// in the current object crossed by a mouse move.
    pub fn drag_select_conts_crossed(&mut self, x: i32, y: i32) -> i32 {
        let vi = self.vi;
        let imod = unsafe { (*vi).imod };
        let mut drew = 0;
        let mut pnt1 = Ipoint::default();
        let mut pnt2 = Ipoint::default();
        let obj = imod_object_get(unsafe { imod.as_ref() })
            .map_or(ptr::null_mut(), |o| o as *const Iobj as *mut Iobj);

        // Skip for movie mode
        if obj.is_null() || unsafe { (*imod).mousemode } == IMOD_MMOVIE {
            return 0;
        }

        // Get image positions of starting and current mouse positions
        let mut iz2 = 0;
        let mut iz1 = 0;
        let (mut p2x, mut p2y) = (0., 0.);
        self.getixy(x, y, &mut p2x, &mut p2y, &mut iz2);
        pnt2.x = p2x;
        pnt2.y = p2y;
        let (mut p1x, mut p1y) = (0., 0.);
        self.getixy(self.lmx, self.lmy, &mut p1x, &mut p1y, &mut iz1);
        pnt1.x = p1x;
        pnt1.y = p1y;
        if iz1 != iz2 || iz1 < 0 {
            return 0;
        }
        if imod_debug('z') {
            imod_print_stderr(&format!(
                "mouse segment {},{} to {},{}\n",
                pnt1.x, pnt1.y, pnt2.x, pnt2.y
            ));
        }

        // Loop on contours
        // Skip single point, ones already selected, or non-wild with Z not
        // matching
        for ob in 0..unsafe { (&(*imod).obj).len() as i32 } {
            let obj = unsafe { (&mut (*imod).obj).as_mut_ptr().add(ob as usize) };
            // Skip for scattered objects
            if iobj_scat(unsafe { (*obj).flags }) != 0 {
                continue;
            }
            for co in 0..unsafe { (&(*obj).cont).len() as i32 } {
                let cont = unsafe { (&mut (*obj).cont).as_mut_ptr().add(co as usize) };
                if unsafe { (&(*cont).pts).len() } < 2 {
                    continue;
                }
                if with_boundary(|n| n.imod_selection_list_query(vi, ob, co)) > -2
                    || (unsafe { (*imod).cindex.contour } == co
                        && unsafe { (*imod).cindex.object } == ob)
                {
                    continue;
                }
                if unsafe { (*cont).flags } & ICONT_WILD == 0
                    && (unsafe { (&(*cont).pts)[0].z } as f64 + 0.5).floor() as i32 != iz1
                {
                    continue;
                }

                // Set up to loop on second point in segment, starting at first
                // point in contour for closed contour
                let pt_start = if iobj_open(unsafe { (*obj).flags }) != 0
                    || unsafe { (*cont).flags } & ICONT_OPEN != 0
                {
                    1
                } else {
                    0
                };
                let mut last_z = iz1;
                if imod_debug('z') {
                    imod_print_stderr(&format!("Examining contour {co}\n"));
                }

                // Loop on points, look for segments on the section
                for pt in pt_start..unsafe { (&(*cont).pts).len() as i32 } {
                    let last_pt = if pt != 0 {
                        pt - 1
                    } else {
                        (unsafe { (&(*cont).pts).len() }) as i32 - 1
                    };
                    let this_z =
                        (unsafe { (&(*cont).pts)[pt as usize].z } as f64 + 0.5).floor() as i32;
                    if last_z == iz1 && this_z == iz1 {
                        if imod_debug('z') {
                            imod_print_stderr(&format!(
                                "{},{} to {},{}\n",
                                unsafe { (&(*cont).pts)[last_pt as usize].x },
                                unsafe { (&(*cont).pts)[last_pt as usize].y },
                                unsafe { (&(*cont).pts)[pt as usize].x },
                                unsafe { (&(*cont).pts)[pt as usize].y }
                            ));
                        }

                        if imod_point_intersect(
                            &pnt1,
                            &pnt2,
                            unsafe { &(&(*cont).pts)[last_pt as usize] },
                            unsafe { &(&(*cont).pts)[pt as usize] },
                        ) != 0
                        {
                            // Crosses.  Select this contour; add current
                            // contour if list empty
                            if unsafe { ilist_size((*vi).selection_list) } == 0
                                && unsafe { (*imod).cindex.contour } >= 0
                            {
                                let ind = unsafe { (*imod).cindex };
                                with_boundary(|n| n.imod_selection_list_add(vi, ind));
                            }
                            unsafe { (*imod).cindex.object = ob };
                            unsafe { (*imod).cindex.contour = co };
                            unsafe { (*imod).cindex.point = pt };
                            let ind = unsafe { (*imod).cindex };
                            with_boundary(|n| n.imod_selection_list_add(vi, ind));
                            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_RETHINK | IMOD_DRAW_XYZ));
                            drew = 1;
                            break;
                        }
                    }
                    last_z = this_z;
                }
            }
        }
        drew
    }

    /// `ZapFuncs::b2Drag` (`xzap.cpp:3083`).
    pub fn b2_drag(&mut self, x: i32, y: i32, control_down: i32) -> i32 {
        let vi = self.vi;
        let mut obj: *mut Iobj;
        let mut cont: *mut Icont;
        let lpt: *mut Ipoint;
        let mut cpt = Ipoint::default();
        let (mut ix, mut iy) = (0., 0.);
        let (mut idx, mut idy);
        let mut pt;
        let mut iz = 0;
        let cancel_hq = self.last_hq_draw_time > S_HQ_DRAW_TIME_CRIT.get();

        if self.num_xpanels != 0 {
            return 0;
        }

        if self.shifting_cont != 0 {
            self.shift_contour(x, y, 2, 0);
            return 1;
        }

        if !unsafe { (*vi).ax }.is_null()
            && with_boundary(|n| n.autox_altmouse(vi)) == AUTOX_ALTMOUSE_PAINT
        {
            self.getixy(x, y, &mut ix, &mut iy, &mut iz);
            /* DNM 2/1/01: need to call with int */
            with_boundary(|n| n.autox_sethigh(vi, ix as i32, iy as i32));
            return 1;
        }

        if (self.rubberband != 0 || self.lasso_on) && S_MOVE_BAND_LASSO.get() != 0 {
            /* Moving rubber band: get desired move and constrain it to keep
            band in the image */
            idx = (x - self.lmx) as f32 / self.xzoom;
            idy = (self.lmy - y) as f32 / self.zoom;
            if self.rubberband != 0 {
                self.shift_rubberband(idx, idy);
            } else {
                cont = self.get_lasso_contour();
                if !cont.is_null() {
                    self.limit_contour_shift(cont, &mut idx, &mut idy);
                    for pt in 0..unsafe { (&(*cont).pts).len() } {
                        unsafe { (&mut (*cont).pts)[pt].x += idx };
                        unsafe { (&mut (*cont).pts)[pt].y += idy };
                    }
                }
            }

            if cancel_hq {
                self.hqgfxsave = self.hqgfx;
                self.hqgfx = 0;
            }
            self.draw();
            if cancel_hq {
                self.hqgfx = self.hqgfxsave;
            }
            if self.rubberband != 0 {
                self.band_changed = 1;
            }
            return 1;
        }

        // Arrow: update the head coordinates
        self.getixy(x, y, &mut ix, &mut iy, &mut iz);
        if self.drawing_arrow {
            iz = self.arrow_head.len() as i32 - 1;
            self.arrow_head[iz as usize].x = ix;
            self.arrow_head[iz as usize].y = iy;
            self.draw();
            return 1;
        }

        if unsafe { (*(*vi).imod).mousemode } == IMOD_MMOVIE && !self.drawing_lasso {
            return 0;
        }

        if unsafe { (*(*vi).imod).cindex.point } < 0 && !self.drawing_lasso {
            return 0;
        }

        cpt.x = ix;
        cpt.y = iy;
        cpt.z = self.section as f32;

        if self.drawing_lasso {
            obj = ivw_get_an_extra_object(unsafe { &mut *vi }, self.lasso_obj_num)
                .map_or(ptr::null_mut(), |o| o as *mut Iobj);
            if obj.is_null() || unsafe { (&(*obj).cont).is_empty() } {
                self.toggle_lasso(true);
                return 0;
            }
            cont = unsafe { (&mut (*obj).cont).as_mut_ptr() };
            lpt = unsafe {
                (&mut (*cont).pts)
                    .as_mut_ptr()
                    .add((&(*cont).pts).len() - 1)
            };
        } else {
            obj = imod_object_get(unsafe { (*vi).imod.as_ref() })
                .map_or(ptr::null_mut(), |o| o as *const Iobj as *mut Iobj);
            if obj.is_null() {
                return 0;
            }

            cont = imod_contour_get(unsafe { (*vi).imod.as_ref() })
                .map_or(ptr::null_mut(), |c| c as *const Icont as *mut Icont);
            if cont.is_null() {
                return 0;
            }

            lpt = unsafe {
                (*cont)
                    .pts
                    .as_mut_ptr()
                    .add((*(*vi).imod).cindex.point as usize)
            };
            if self.twod != 0 {
                cpt.z = unsafe { (*lpt).z };
            }

            /* DNM 6/18/03: If Z or time has changed, treat it like a button
            click so new contour can be started */
            // DNM 6/30/04: change to start new for any kind of contour with
            // time change.  DNM 7/15/08: Start new contour at the point limit
            if (iobj_planar(unsafe { (*obj).flags }) != 0
                && unsafe { (*cont).flags } & ICONT_WILD == 0
                && (unsafe { (*lpt).z } as f64 + 0.5).floor() as i32 != cpt.z as i32)
                || unsafe { ivw_time_mismatch(vi, self.time_lock, obj, cont) }
                || (unsafe { (*obj).extra[IOBJ_EX_PNT_LIMIT] } != 0
                    && unsafe { (&(*cont).pts).len() as u32 }
                        >= unsafe { (*obj).extra[IOBJ_EX_PNT_LIMIT] })
            {
                self.register_drag_additions();
                return self.b2_click(x, y, 0);
            }

            if unsafe { ivw_time_mismatch(vi, self.time_lock, obj, cont) } {
                return 0;
            }
        }

        let dist = imodel_point_dist(unsafe { &*lpt }, &cpt);
        if dist
            > crate::imod::three_dmod::model_edit::scale_model_res(
                unsafe { (*(*vi).imod).res },
                self.zoom / self.device_pixel_ratio,
            ) as f64
        {
            if self.drawing_lasso {
                imod_point_append(unsafe { &mut *cont }, cpt);
                self.set_draw_current_only(1);
            } else {
                // Set insertion index to next point, or to current if drawing
                // backwards
                pt = unsafe { (*(*vi).imod).cindex.point } + 1;
                if pt > 0 && self.insertmode != 0 {
                    pt -= 1;
                }

                // Set flag for drawing current contour only if at end and
                // going forward
                if self.insertmode == 0 && pt == unsafe { (&(*cont).pts).len() as i32 } {
                    self.set_draw_current_only(1);
                }

                // Register previous additions if the count is up or if the
                // object or contour has changed
                if self.drag_add_count >= S_DRAG_REGISTER_SIZE.get()
                    || self.drag_add_index.object != unsafe { (*(*vi).imod).cindex.object }
                    || self.drag_add_index.contour != unsafe { (*(*vi).imod).cindex.contour }
                {
                    self.register_drag_additions();
                }

                // Start keeping track of delayed registrations by opening a
                // unit and saving the indices.  If general store exists, start
                // with whole data change to save the store.  Otherwise if
                // going backwards, need to increment registered first point
                if self.drag_add_count == 0 {
                    if unsafe { !(&(*cont).store).is_empty() } {
                        undo_contour_data_chg_cc(unsafe { &mut *(*vi).undo }, unsafe {
                            &mut *(*vi).imod
                        });
                    } else {
                        unsafe { (*(*vi).undo).get_open_unit(&*(*vi).imod) };
                    }
                    self.drag_add_index = unsafe { (*(*vi).imod).cindex };
                    self.drag_add_index.point = pt;
                } else if self.insertmode != 0 {
                    self.drag_add_index.point += 1;
                }

                // Always save last point not registered and increment count
                self.drag_add_end = pt;
                self.drag_add_count += 1;

                // Since we are not changing xyzmouse yet, drag draws can be
                // done without IMOD_DRAW_XYZ; this prevents some
                // time-consuming things in other windows
                imod_insert_point(unsafe { (*vi).imod.as_mut() }, Some(cpt), pt);
            }
            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_MOD | IMOD_DRAW_NOSYNC));
            return 1;
        }
        0
    }

    /// `ZapFuncs::b3Drag` (`xzap.cpp:3257`).
    pub fn b3_drag(&mut self, x: i32, y: i32, control_down: i32, shift_down: i32) -> i32 {
        let vi = self.vi;
        let mut lpt: *mut Ipoint;
        let mut pt = Ipoint::default();
        let (mut ix, mut iy) = (0., 0.);
        let mut iz = 0;

        if self.num_xpanels != 0 {
            return 0;
        }

        if self.shifting_cont != 0 {
            self.shift_contour(x, y, 3, shift_down);
            return 1;
        }

        if !unsafe { (*vi).ax }.is_null()
            && with_boundary(|n| n.autox_altmouse(vi)) == AUTOX_ALTMOUSE_PAINT
        {
            self.getixy(x, y, &mut ix, &mut iy, &mut iz);
            /* DNM 2/1/01: need to call with int */
            with_boundary(|n| n.autox_setlow(vi, ix as i32, iy as i32));
            return 1;
        }

        if unsafe { (*(*vi).imod).mousemode } == IMOD_MMOVIE {
            return 0;
        }

        if unsafe { (*(*vi).imod).cindex.point } < 0 {
            return 0;
        }

        let cont = imod_contour_get(unsafe { (*vi).imod.as_ref() })
            .map_or(ptr::null_mut(), |c| c as *const Icont as *mut Icont);
        if cont.is_null() {
            return 0;
        }

        /* DNM 11/13/02: do not allow operation on scattered points */
        let obj = imod_object_get(unsafe { (*vi).imod.as_ref() })
            .map_or(ptr::null_mut(), |o| o as *const Iobj as *mut Iobj);
        if iobj_scat(unsafe { (*obj).flags }) != 0 {
            return 0;
        }

        if unsafe { ivw_time_mismatch(vi, self.time_lock, obj, cont) } {
            return 0;
        }

        if control_down != 0 {
            return self.del_under_cursor(x, y, cont);
        }

        if unsafe { (*(*vi).imod).cindex.point } == unsafe { (&(*cont).pts).len() as i32 } - 1 {
            return 0;
        }

        /* DNM 11/13/02: need to test for both next and current points to
        prevent strange moves between sections */
        let cur = unsafe { (*(*vi).imod).cindex.point } as usize;
        if self.point_visable(unsafe { &(&(*cont).pts)[cur + 1] }) == 0
            || self.point_visable(unsafe { &(&(*cont).pts)[cur] }) == 0
        {
            return 0;
        }

        lpt = unsafe { (&mut (*cont).pts).as_mut_ptr().add(cur) };
        let (mut px, mut py) = (0., 0.);
        self.getixy(x, y, &mut px, &mut py, &mut iz);
        pt.x = px;
        pt.y = py;
        pt.z = unsafe { (*lpt).z };
        if imodel_point_dist(unsafe { &*lpt }, &pt)
            > crate::imod::three_dmod::model_edit::scale_model_res(
                unsafe { (*(*vi).imod).res },
                self.zoom / self.device_pixel_ratio,
            ) as f64
        {
            unsafe { (*(*vi).imod).cindex.point += 1 };
            undo_point_shift_cp(unsafe { &mut *(*vi).undo }, unsafe { &mut *(*vi).imod });
            lpt = unsafe {
                (*cont)
                    .pts
                    .as_mut_ptr()
                    .add((*(*vi).imod).cindex.point as usize)
            };
            unsafe { (*lpt).x = pt.x };
            unsafe { (*lpt).y = pt.y };
            unsafe { (*lpt).z = pt.z };
            undo_finish_unit(unsafe { &mut *(*vi).undo }, unsafe { &*(*vi).imod });
            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_XYZ | IMOD_DRAW_MOD));
            return 1;
        }
        0
    }

    /// `ZapFuncs::registerDragAdditions` (`xzap.cpp:3338`); register
    /// accumulated additions for undo.
    pub fn register_drag_additions(&mut self) {
        let index = unsafe { &raw mut (*(*self.vi).imod).cindex };
        if self.drag_add_count == 0 {
            return;
        }
        self.drag_add_count = 0;

        // If obj/cont don't match, forget it
        if self.drag_add_index.object != unsafe { (*index).object }
            || self.drag_add_index.contour != unsafe { (*index).contour }
        {
            undo_flush_unit(unsafe { &mut *(*self.vi).undo });
            return;
        }

        // Send out the additions
        undo_point_addition_cc2(
            unsafe { &mut *(*self.vi).undo },
            unsafe { &mut *(*self.vi).imod },
            self.drag_add_index.point.min(unsafe { (*index).point }),
            self.drag_add_index.point.max(unsafe { (*index).point }),
        );
        undo_finish_unit(unsafe { &mut *(*self.vi).undo }, unsafe {
            &*(*self.vi).imod
        });
    }

    /*
     * CONTOUR SHIFTING/TRANSFORMING
     */

    /// `ZapFuncs::toggleContourShift` (`xzap.cpp:3365`); single routine to
    /// toggle shift for contour move window to call.
    pub fn toggle_contour_shift(&mut self) {
        if self.shifting_cont != 0 {
            self.end_contour_shift();
        } else {
            self.setup_contour_shift();
        }
    }

    /// `ZapFuncs::endContourShift` (`xzap.cpp:3376`); turn off contour
    /// shifting and reset mouse.
    pub fn end_contour_shift(&mut self) {
        if self.shifting_cont == 0 {
            return;
        }
        self.shifting_cont = 0;
        ivw_free_extra_object(unsafe { &mut *self.vi }, self.shift_obj_num);
        if self.center_marked != 0 || self.lasso_on {
            let vi = self.vi;
            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_MOD));
        }
        self.set_cursor(self.mousemode, false);
    }

    /// `ZapFuncs::checkContourShift` (`xzap.cpp:3390`); check whether contour
    /// shifting is OK and return current contour.
    pub fn check_contour_shift(&mut self, pt: &mut i32, err: &mut i32) -> *mut Icont {
        let vi = self.vi;
        let obj = imod_object_get(unsafe { (*vi).imod.as_ref() })
            .map_or(ptr::null_mut(), |o| o as *const Iobj as *mut Iobj);
        let mut cont = imod_contour_get(unsafe { (*vi).imod.as_ref() })
            .map_or(ptr::null_mut(), |c| c as *const Icont as *mut Icont);
        *pt = unsafe { (*(*vi).imod).cindex.point };

        *err = 0;
        if self.lasso_on && cont.is_null() && !self.drawing_lasso {
            cont = self.get_lasso_contour();
            if !cont.is_null() {
                *pt = 0;
            } else {
                *err = 1;
            }
        } else {
            if unsafe { (*(*vi).imod).mousemode } != IMOD_MMODEL
                || obj.is_null()
                || cont.is_null()
                || unsafe { (&(*cont).pts).is_empty() }
            {
                *err = 1;
            } else if iobj_scat(unsafe { (*obj).flags }) != 0
                || (iobj_close(unsafe { (*obj).flags }) == 0
                    && unsafe { (*cont).flags } & ICONT_WILD != 0)
            {
                *err = -1;
            }

            // If no current point, just use first
            if *pt < 0 {
                *pt = 0;
            }
        }

        if *err != 0 {
            self.end_contour_shift();
        }
        cont
    }

    /// `ZapFuncs::setupContourShift` (`xzap.cpp:3424`); initiate contour
    /// shifting.
    pub fn setup_contour_shift(&mut self) {
        let mut pt = 0;
        let mut err = 0;
        self.check_contour_shift(&mut pt, &mut err);
        if err < 0 {
            wprint(
                "\u{7}You cannot shift scattered point or non-planar open contours.To shift this contour, temporarily make the object type be closed.\n",
            );
        }
        if err != 0 {
            return;
        }
        self.shifting_cont = 1;
        if self.starting_band != 0 {
            self.toggle_rubberband(true);
        }
        self.center_defined = 0;
        self.center_marked = 0;
        self.fixed_pt_defined = 0;
        self.shift_obj_num = ivw_get_free_extra_object_number(unsafe { &mut *self.vi });
        self.set_cursor(self.mousemode, false);
        if self.lasso_on {
            let vi = self.vi;
            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_MOD));
        }
    }

    /// `ZapFuncs::startShiftingContour` (`xzap.cpp:3452`); start the actual
    /// shift once the mouse goes down.
    pub fn start_shifting_contour(&mut self, x: i32, y: i32, button: i32, ctrl_down: i32) -> i32 {
        let mut pt = 0;
        let mut err = 0;
        let mut iz = 0;
        let (mut ix, mut iy) = (0., 0.);
        let cont = self.check_contour_shift(&mut pt, &mut err);
        if err != 0 {
            return 0;
        }

        self.getixy(x, y, &mut ix, &mut iy, &mut iz);

        // If button for marking center, save coordinates, set flag, show mark
        if button == 2 && ctrl_down != 0 {
            self.xform_center.x = ix;
            self.xform_center.y = iy;
            self.center_defined = 1;
            self.mark_xform_center(ix, iy);
            return 1;
        }

        // If button for 2nd fixed point, toggle it on or off
        if button == 3 && ctrl_down != 0 {
            if self.fixed_pt_defined != 0 {
                self.fixed_pt_defined = 0;
            } else {
                // Define center if it is not already defined
                if self.center_defined == 0 {
                    let (mut cx, mut cy) = (self.xform_center.x, self.xform_center.y);
                    let failed = self.default_xform_center(&mut cx, &mut cy);
                    self.xform_center.x = cx;
                    self.xform_center.y = cy;
                    if failed != 0 {
                        self.end_contour_shift();
                        return 0;
                    }
                }
                self.center_defined = 1;

                // Save coordinates and show both marks
                self.xform_fixed_pt.x = ix;
                self.xform_fixed_pt.y = iy;
                self.fixed_pt_defined = 1;
            }
            let (cx, cy) = (self.xform_center.x, self.xform_center.y);
            self.mark_xform_center(cx, cy);
            return 1;
        }

        if button == 1 {
            // Get base for shift as current point minus mouse position
            let base = Ipoint {
                x: unsafe { (&(*cont).pts)[pt as usize].x } - ix,
                y: unsafe { (&(*cont).pts)[pt as usize].y } - iy,
                z: S_CONT_SHIFT_BASE.get().z,
            };
            S_CONT_SHIFT_BASE.set(base);
        } else {
            // Use defined center if one was set
            if self.center_defined != 0 {
                let mut base = S_CONT_SHIFT_BASE.get();
                base.x = self.xform_center.x;
                base.y = self.xform_center.y;
                S_CONT_SHIFT_BASE.set(base);
            } else {
                // Otherwise get center for transforms as center of mass
                let mut base = S_CONT_SHIFT_BASE.get();
                let (mut bx, mut by) = (base.x, base.y);
                let failed = self.default_xform_center(&mut bx, &mut by);
                base.x = bx;
                base.y = by;
                S_CONT_SHIFT_BASE.set(base);
                if failed != 0 {
                    self.end_contour_shift();
                    return 0;
                }
            }
            let base = S_CONT_SHIFT_BASE.get();
            self.mark_xform_center(base.x, base.y);
            return 1;
        }
        0
    }

    /// `ZapFuncs::defaultXformCenter` (`xzap.cpp:3521`); get default center
    /// for transforms as center of mass.
    pub fn default_xform_center(&mut self, xcen: &mut f32, ycen: &mut f32) -> i32 {
        let mut pt = 0;
        let mut err = 0;
        let (mut curco, mut curob) = (0, 0);
        let mut cent = Ipoint::default();
        let mut cent_sum = Ipoint::default();
        let imod = unsafe { (*self.vi).imod };

        // Loop on contours and analyze current or selected ones
        imod_get_index(unsafe { &*imod }, &mut curob, &mut curco, &mut pt);
        if curco < 0 && self.lasso_on && !self.drawing_lasso {
            let cont = self.get_lasso_contour();
            if cont.is_null() {
                return 1;
            }
            imod_contour_center_of_mass(unsafe { cont.as_mut() }, &mut cent);
            *xcen = cent.x;
            *ycen = cent.y;
            return 0;
        }

        *xcen = 0.;
        *ycen = 0.;
        cent_sum.x = 0.;
        cent_sum.y = 0.;
        let mut area_sum = 0.;
        for ob in 0..unsafe { (&(*imod).obj).len() as i32 } {
            let obj = unsafe { (&mut (*imod).obj).as_mut_ptr().add(ob as usize) };
            if iobj_scat(unsafe { (*obj).flags }) != 0 {
                continue;
            }
            for co in 0..unsafe { (&(*obj).cont).len() as i32 } {
                let vi = self.vi;
                if (ob == curob && co == curco)
                    || with_boundary(|n| n.imod_selection_list_query(vi, ob, co)) > -2
                {
                    let cont = unsafe { (&mut (*obj).cont).as_mut_ptr().add(co as usize) };
                    if iobj_close(unsafe { (*obj).flags }) == 0
                        && unsafe { (*cont).flags } & ICONT_WILD != 0
                    {
                        continue;
                    }

                    // For each contour add centroid to straight sum,
                    // accumulate area-weighted sum also
                    imod_contour_center_of_mass(unsafe { cont.as_mut() }, &mut cent);
                    let area = imod_contour_area(unsafe { cont.as_ref() });
                    area_sum += area;
                    *xcen += cent.x;
                    *ycen += cent.y;
                    cent_sum.x += cent.x * area;
                    cent_sum.y += cent.y * area;
                    err += 1;
                }
            }
        }

        if err == 0 {
            return 1;
        }

        // Use plain sum if area small or if only one contour, otherwise
        // use an area-weighted sum
        if area_sum < 1. || err == 1 {
            *xcen /= err as f32;
            *ycen /= err as f32;
        } else if area_sum >= 1. && err > 1 {
            *xcen = cent_sum.x / area_sum;
            *ycen = cent_sum.y / area_sum;
        }
        0
    }

    /// `ZapFuncs::shiftContour` (`xzap.cpp:3589`); shift or transform contour
    /// upon mouse move.
    pub fn shift_contour(&mut self, x: i32, y: i32, button: i32, shift_down: i32) {
        let mut pt = 0;
        let mut err = 0;
        let (mut curco, mut curob) = (0, 0);
        let mut iz = 0;
        let (mut ix, mut iy) = (0., 0.);
        let mut mat = [[0.0f32; 2]; 2];
        let imod = unsafe { (*self.vi).imod };
        let mut cont = self.check_contour_shift(&mut pt, &mut err);
        if err != 0 {
            return;
        }

        if button == 1 {
            // Shift by change from original mouse pos minus change in current
            // point position
            self.getixy(x, y, &mut ix, &mut iy, &mut iz);
            ix += S_CONT_SHIFT_BASE.get().x - unsafe { (&(*cont).pts)[pt as usize].x };
            iy += S_CONT_SHIFT_BASE.get().y - unsafe { (&(*cont).pts)[pt as usize].y };
            self.limit_contour_shift(cont, &mut ix, &mut iy);
        } else {
            // Get transformation matrix if 2nd or 3rd button
            err = button + if button == 3 && shift_down != 0 { 1 } else { 0 };
            if self.mouse_xform_matrix(x, y, err, &mut mat) != 0 {
                return;
            }
        }

        imod_get_index(unsafe { &*imod }, &mut curob, &mut curco, &mut pt);
        if curco < 0 && self.lasso_on && !self.drawing_lasso {
            self.transform_contour(cont, &mat, ix, iy, button);
            let vi = self.vi;
            with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_XYZ | IMOD_DRAW_MOD));
            return;
        }

        // Loop on contours and act on current or selected ones
        for ob in 0..unsafe { (&(*imod).obj).len() as i32 } {
            let obj = unsafe { (&mut (*imod).obj).as_mut_ptr().add(ob as usize) };
            if iobj_scat(unsafe { (*obj).flags }) != 0 {
                continue;
            }
            for co in 0..unsafe { (&(*obj).cont).len() as i32 } {
                let vi = self.vi;
                if (ob == curob && co == curco)
                    || with_boundary(|n| n.imod_selection_list_query(vi, ob, co)) > -2
                {
                    cont = unsafe { (&mut (*obj).cont).as_mut_ptr().add(co as usize) };
                    if iobj_close(unsafe { (*obj).flags }) == 0
                        && unsafe { (*cont).flags } & ICONT_WILD != 0
                    {
                        continue;
                    }

                    // Register changes first time only
                    if self.shift_registered == 0 {
                        undo_contour_data_chg(
                            unsafe { &mut *(*self.vi).undo },
                            unsafe { &mut *(*self.vi).imod },
                            ob,
                            co,
                        );
                    }
                    self.transform_contour(cont, &mat, ix, iy, button);
                }
            }
        }

        self.shift_registered = 1;
        let vi = self.vi;
        with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_XYZ | IMOD_DRAW_MOD));
    }

    /// `ZapFuncs::limitContourShift` (`xzap.cpp:3651`); limit the shift to
    /// keep the contour intersecting with image.
    pub fn limit_contour_shift(&self, cont: *mut Icont, ix: &mut f32, iy: &mut f32) {
        let mut pmin = Ipoint::default();
        let mut pmax = Ipoint::default();
        imod_contour_get_bbox(unsafe { cont.as_ref() }, &mut pmin, &mut pmax);

        if pmin.x + *ix >= unsafe { (*self.vi).xsize } as f32 - 1. {
            *ix = unsafe { (*self.vi).xsize } as f32 - pmin.x - 1.;
        } else if pmax.x + *ix <= 0. {
            *ix = -pmax.x + 1.;
        }
        if pmin.y + *iy >= unsafe { (*self.vi).ysize } as f32 - 1. {
            *iy = unsafe { (*self.vi).ysize } as f32 - pmin.y - 1.;
        } else if pmax.y + *iy <= 0. {
            *iy = -pmax.y + 1.;
        }
    }

    /// `ZapFuncs::transformContour` (`xzap.cpp:3669`); transform one contour
    /// with shift of matrix.
    pub fn transform_contour(
        &mut self,
        cont: *mut Icont,
        mat: &[[f32; 2]; 2],
        mut ix: f32,
        mut iy: f32,
        button: i32,
    ) {
        if button == 1 {
            // Shift points
            for pt in 0..unsafe { (&(*cont).pts).len() } {
                unsafe { (&mut (*cont).pts)[pt].x += ix };
                unsafe { (&mut (*cont).pts)[pt].y += iy };
            }
        } else {
            // Transform points
            let base = S_CONT_SHIFT_BASE.get();
            for pt in 0..unsafe { (&(*cont).pts).len() } {
                ix = mat[0][0] * (unsafe { (&(*cont).pts)[pt].x } - base.x)
                    + mat[0][1] * (unsafe { (&(*cont).pts)[pt].y } - base.y)
                    + base.x;
                iy = mat[1][0] * (unsafe { (&(*cont).pts)[pt].x } - base.x)
                    + mat[1][1] * (unsafe { (&(*cont).pts)[pt].y } - base.y)
                    + base.y;
                unsafe { (&mut (*cont).pts)[pt].x = ix };
                unsafe { (&mut (*cont).pts)[pt].y = iy };
            }
        }
    }

    /// `ZapFuncs::mouseXformMatrix` (`xzap.cpp:3697`); compute transform
    /// matrix from the mouse move.
    pub fn mouse_xform_matrix(
        &mut self,
        x: i32,
        y: i32,
        type_: i32,
        mat: &mut [[f32; 2]; 2],
    ) -> i32 {
        let str_thresh = 0.001f32;
        let rot_thresh = 0.05f32;
        let del_crit = 20.0f32;
        let fix_pt_crit_sq = 1000.0f32;
        let (mut dxf, mut dyf, dxl, dyl, dxn): (f64, f64, f64, f64, f64);
        let (mut startang, mut endang, radst, radnd, mut delrad, mut drot, mut scale): (
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
        );
        let (dyn_, mut distsq, disxn, disyn, disxl, disyl, mut p2lsq, mut tmin): (
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
        );

        let base = S_CONT_SHIFT_BASE.get();
        let xcen = self.xpos(base.x);
        let ycen = self.ypos(base.y);

        // Compute starting/ending  angle and radii as in midas
        // l for last, n for new, f for fixed
        dxl = (self.lmx - xcen) as f64;
        dyl = (self.winy - 1 - self.lmy - ycen) as f64;
        if dxl > -(del_crit as f64)
            && dxl < del_crit as f64
            && dyl > -(del_crit as f64)
            && dyl < del_crit as f64
        {
            return 1;
        }
        radst = (dxl * dxl + dyl * dyl).sqrt();
        startang = dyl.atan2(dxl) / RADIANS_PER_DEGREE;

        dxn = (x - xcen) as f64;
        dyn_ = (self.winy - 1 - y - ycen) as f64;
        if dxn > -(del_crit as f64)
            && dxn < del_crit as f64
            && dyn_ > -(del_crit as f64)
            && dyn_ < del_crit as f64
        {
            return 1;
        }
        radnd = (dxn * dxn + dyn_ * dyn_).sqrt();
        endang = dyn_.atan2(dxn) / RADIANS_PER_DEGREE;

        drot = 0.;
        scale = 1.;
        delrad = 1.;
        if type_ == 3 && self.fixed_pt_defined != 0 {
            // Get fixed point in window coordinates and test if separation
            // from center is enough.  This code slavishly imitates midas but
            // all the names are changed
            let xfix = self.xpos(self.xform_fixed_pt.x);
            let yfix = self.ypos(self.xform_fixed_pt.y);
            dxf = (xfix - xcen) as f64;
            dyf = (yfix - ycen) as f64;
            // `xzap.cpp:3736` is `dxf * dxf + dyf + dyf`, not `dyf * dyf`.
            distsq = dxf * dxf + dyf + dyf;
            if distsq < fix_pt_crit_sq as f64 {
                return 1;
            }

            // Then make sure each point is far enough away from line
            tmin = (dxn * dxf + dyn_ * dyf) / distsq;
            disxn = tmin * dxf - dxn;
            disyn = tmin * dyf - dyn_;
            p2lsq = disxn * disxn + disyn * disyn;
            if p2lsq < fix_pt_crit_sq as f64 {
                return 1;
            }

            tmin = (dxl * dxf + dyl * dyf) / distsq;
            disxl = tmin * dxf - dxl;
            disyl = tmin * dyf - dyl;
            p2lsq = disxl * disxl + disyl * disyl;
            if p2lsq < fix_pt_crit_sq as f64 {
                return 1;
            }

            // If both points are on same side of line, do the change
            if disxl * disxn + disyl * disyn <= 0. {
                return 1;
            }
            distsq = dyl * dxf - dxl * dyf;
            if distsq.abs() < 1.0e-5 {
                return 1;
            }
            mat[0][0] = ((dxf * dyl - dxn * dyf) / distsq) as f32;
            mat[0][1] = ((dxf * dxn - dxl * dxf) / distsq) as f32;
            mat[1][0] = ((dyf * dyl - dyn_ * dyf) / distsq) as f32;
            mat[1][1] = ((dxf * dyn_ - dxl * dyf) / distsq) as f32;
            return 0;
        }

        if type_ == 2 {
            // Compute rotation from change in angle
            drot = endang - startang;
            if drot < -360. {
                drot += 360.;
            }
            if drot > 360. {
                drot -= 360.;
            }
            drot = rot_thresh as f64 * (drot / rot_thresh as f64 + 0.5).floor();
            endang = 0.;
            if drot == 0. {
                return 1;
            }
        } else {
            // Compute stretch from change in radius; set up as stretch or
            // scale
            delrad = (radnd - radst) / radst;
            delrad = str_thresh as f64 * (delrad / str_thresh as f64 + 0.5).floor();
            if delrad == 0. {
                return 1;
            }
            delrad += 1.;
            if type_ == 4 {
                scale = delrad;
                delrad = 1.;
                endang = 0.;
            }
        }

        // Compute matrix.  `amat_to_rotmagstr.rs`'s `rotmagstr_to_amat`
        // stores in the Fortran `(2, *)` order of the C wrapper
        // (`amat_to_rotmagstr.c:213`), so a11, a21, a12, a22, while the C
        // `rotmagstrToAmat` the source calls takes the four pointers in
        // a11, a12, a21, a22 order.
        let mut amat = [0.0f32; 4];
        crate::imod::libcfshr::amat_to_rotmagstr::rotmagstr_to_amat(
            drot as f32,
            scale as f32,
            delrad as f32,
            endang as f32,
            &mut amat,
        );
        mat[0][0] = amat[0];
        mat[0][1] = amat[2];
        mat[1][0] = amat[1];
        mat[1][1] = amat[3];
        0
    }

    /// `ZapFuncs::markXformCenter` (`xzap.cpp:3811`); mark the center of
    /// transformation with a star in extra object.
    pub fn mark_xform_center(&mut self, mut ix: f32, mut iy: f32) {
        let star_end: [f32; 6] = [0., 8., 7., 4., 7., -4.];
        let obj = ivw_get_an_extra_object(unsafe { &mut *self.vi }, self.shift_obj_num)
            .map_or(ptr::null_mut(), |o| o as *mut Iobj);
        let mut tpt = Ipoint::default();
        let mut store = Istore::default();

        if obj.is_null() {
            return;
        }

        // Clear out the object and set to yellow non scattered
        ivw_clear_an_extra_object(unsafe { &mut *self.vi }, self.shift_obj_num);
        imod_object_set_color(unsafe { &mut *obj }, 1., 1., 0.);
        unsafe { (*obj).flags &= !IMOD_OBJFLAG_SCAT };
        unsafe { (*obj).pdrawsize = 0 };
        unsafe { (*obj).linewidth2 = 1 };
        tpt.z = self.section as f32;

        // Add three contours for lines
        for which in 0..if self.fixed_pt_defined != 0 { 2 } else { 1 } {
            for i in 0..3 {
                let Some(mut cont) = imod_contour_new() else {
                    break;
                };
                tpt.x = ix + star_end[i * 2];
                tpt.y = iy + star_end[i * 2 + 1];
                imod_point_append(&mut cont, tpt);
                tpt.x = ix - star_end[i * 2];
                tpt.y = iy - star_end[i * 2 + 1];
                imod_point_append(&mut cont, tpt);
                imod_object_add_contour(unsafe { &mut *obj }, cont);
                if which != 0 {
                    store.type_ = GEN_STORE_COLOR;
                    store.flags = GEN_STORE_BYTE << 2;
                    store.index = StoreUnion { i: 3 + i as i32 };
                    store.value = StoreUnion { i: 0 };
                    unsafe { store.value.b[0] = 255 };
                    istore_insert(unsafe { &mut (*obj).store }, store);
                }
            }
            ix = self.xform_fixed_pt.x;
            iy = self.xform_fixed_pt.y;
        }
        self.center_marked = 1;
        let vi = self.vi;
        with_boundary(|n| n.imod_draw(vi, IMOD_DRAW_MOD));
    }

    /********************************************************
     * conversion functions between image and window cords. */

    /* DNM 9/15/03: use the possibly slightly different x zoom.  This might
    make a few tenths of a pixel difference */

    /// `ZapFuncs::xpos` (`xzap.cpp:3869`); return x pos in window for given
    /// image x cord.
    pub fn xpos(&self, x: f32) -> i32 {
        (((x - self.xpos_start as f32) * self.xzoom) + self.xborder as f32) as i32
    }

    /// `ZapFuncs::ypos` (`xzap.cpp:3875`); return y pos in window for given
    /// image y cord.
    pub fn ypos(&self, y: f32) -> i32 {
        (((y - self.ypos_start as f32) * self.zoom) + self.yborder as f32) as i32
    }

    /// `ZapFuncs::getixy` (`xzap.cpp:3881`); returns image coords in x, y, z,
    /// given mouse coords mx, my.
    pub fn getixy(&self, mx: i32, mut my: i32, x: &mut f32, y: &mut f32, z: &mut i32) {
        let mut mx = mx;
        let mut indx = 0;
        let mut indy = 0;

        // 10/31/04: winy - 1 maps to 0, not winy, so need a -1 here
        my = self.winy - 1 - my;

        if self.num_xpanels == 0 {
            *z = self.section;
        } else {
            self.panel_index_and_coord(
                self.panel_xsize,
                self.num_xpanels,
                self.panel_gutter,
                self.panel_xborder,
                &mut mx,
                &mut indx,
            );
            self.panel_index_and_coord(
                self.panel_ysize,
                self.num_ypanels,
                self.panel_gutter,
                self.panel_yborder,
                &mut my,
                &mut indy,
            );
            if indx < 0 || indy < 0 {
                *z = -1;
            } else {
                let indp = indx + indy * self.num_xpanels;
                let indmid = (self.num_xpanels * self.num_ypanels - 1) / 2;
                *z = self.section + (indp - indmid) * self.panel_zstep;
                if *z < 0 || *z >= unsafe { (*self.vi).zsize } {
                    *z = -1;
                }
            }
        }
        *x = (((mx as f64 + 0.5 - self.xborder as f64) as f32) / self.xzoom)
            + self.xpos_start as f32;
        *y = (((my as f64 + 0.5 - self.yborder as f64) as f32) / self.zoom)
            + self.ypos_start as f32;
    }

    /// `ZapFuncs::panelIndexAndCoord` (`xzap.cpp:3911`); determine which panel
    /// a point is in for one direction and get the position within it.
    pub fn panel_index_and_coord(
        &self,
        size: i32,
        num: i32,
        gutter: i32,
        border: i32,
        pos: &mut i32,
        panel_ind: &mut i32,
    ) {
        let spacing = size + gutter;
        *panel_ind = (*pos - border) / spacing;
        *pos -= border + *panel_ind * spacing;
        if *pos < 0 || *pos >= size || *panel_ind < 0 || *panel_ind >= num {
            *panel_ind = -1;
        }
    }

    /// `ZapFuncs::bandImageToMouse` (`xzap.cpp:3928`); convert image
    /// rubberband coordinates to outer window coordinates, with optional
    /// clipping to window limits.
    pub fn band_image_to_mouse(&mut self, ifclip: i32) {
        self.rb_mouse_x0 = self.xpos(self.rb_image_x0) - 1;
        self.rb_mouse_x1 = self.xpos(self.rb_image_x1);
        self.rb_mouse_y0 = self.winy - 1 - self.ypos(self.rb_image_y1);
        self.rb_mouse_y1 = self.winy - self.ypos(self.rb_image_y0);
        if ifclip != 0 {
            if self.rb_mouse_x0 < 0 {
                self.rb_mouse_x0 = 0;
            }
            if self.rb_mouse_x1 >= self.winx {
                self.rb_mouse_x1 = self.winx - 1;
            }
            if self.rb_mouse_y0 < 0 {
                self.rb_mouse_y0 = 0;
            }
            if self.rb_mouse_y1 >= self.winy {
                self.rb_mouse_y1 = self.winy - 1;
            }
        }
    }

    /// `ZapFuncs::bandMouseToImage` (`xzap.cpp:3953`); convert rubberband
    /// window coordinates to inside image coordinates, with optional clipping.
    pub fn band_mouse_to_image(&mut self, ifclip: i32) {
        let mut iz = 0;
        let (mut x0, mut y0) = (0., 0.);
        let (mut x1, mut y1) = (0., 0.);
        self.getixy(
            self.rb_mouse_x0 + 1,
            self.rb_mouse_y1 - 1,
            &mut x0,
            &mut y0,
            &mut iz,
        );
        self.rb_image_x0 = x0;
        self.rb_image_y0 = y0;
        self.getixy(
            self.rb_mouse_x1,
            self.rb_mouse_y0,
            &mut x1,
            &mut y1,
            &mut iz,
        );
        self.rb_image_x1 = x1;
        self.rb_image_y1 = y1;

        if ifclip != 0 {
            if self.rb_image_x0 < 0. {
                self.rb_image_x0 = 0.;
            }
            if self.rb_image_x1 > unsafe { (*self.vi).xsize } as f32 {
                self.rb_image_x1 = unsafe { (*self.vi).xsize } as f32;
            }
            if self.rb_image_y0 < 0. {
                self.rb_image_y0 = 0.;
            }
            if self.rb_image_y1 > unsafe { (*self.vi).ysize } as f32 {
                self.rb_image_y1 = unsafe { (*self.vi).ysize } as f32;
            }
        }
    }

    /// `ZapFuncs::setSnapshotLimits` (`xzap.cpp:3978`); sets the limits
    /// parameter to be used for snapshotting to null or to subarea limits
    /// that are placed in the supplied `limarr` array.
    pub fn set_snapshot_limits(&mut self, limarr: &mut [i32; 4]) -> Option<[i32; 4]> {
        if self.rubberband != 0 {
            self.band_image_to_mouse(1);
            limarr[0] = self.rb_mouse_x0 + 1;
            limarr[1] = self.winy - self.rb_mouse_y1;
            limarr[2] = self.rb_mouse_x1
                - (if self.device_pixel_ratio > 1. { 2 } else { 1 })
                - self.rb_mouse_x0;
            limarr[3] = self.rb_mouse_y1
                - (if self.device_pixel_ratio > 1. { 2 } else { 1 })
                - self.rb_mouse_y0;
            return Some(*limarr);
        }
        None
    }

    /// `ZapFuncs::getLowHighSection` (`xzap.cpp:3996`); return the correct low
    /// and high section values from the zap parameter.  Returns true if the
    /// rubberband is in use and at least one of the two values has been set.
    pub fn get_low_high_section(&mut self, low_section: &mut i32, high_section: &mut i32) -> bool {
        let mut low_high_section_set = true;
        // If rubberband is not enabled, set both ints to the current section
        // (original functionality) and return false.
        if self.rubberband + self.starting_band == 0 {
            low_high_section_set = false;
            *low_section = self.section + 1;
            *high_section = self.section + 1;
        } else {
            let low = with_boundary(|n| n.zap_window_low_section());
            let high = with_boundary(|n| n.zap_window_high_section());
            // If neither of the section values have been set, fall back to the
            // original functionality and return false.
            if low.is_empty() && high.is_empty() {
                low_high_section_set = false;
                *low_section = self.section + 1;
                *high_section = self.section + 1;
            } else {
                *low_section = low.trim().parse::<i32>().unwrap_or(0);
                *high_section = high.trim().parse::<i32>().unwrap_or(0);
                // If only one of the section values have been set, set the
                // other one to the max/min.
                if low.is_empty() {
                    *low_section = 1;
                } else if high.is_empty() {
                    *high_section = unsafe { (*self.vi).zsize };
                }
            }
        }
        // LowSection cannot be bigger then highSection.
        if *low_section > *high_section {
            std::mem::swap(low_section, high_section);
        }
        low_high_section_set
    }

    /// `ZapFuncs::printInfo` (`xzap.cpp:4043`); prints window size and image
    /// coordinates in Info Window and returns the partial trimvol command.
    pub fn print_info(&mut self, to_info_window: bool) -> String {
        let (mut xl, mut xr, mut yb, mut yt) = (0., 0., 0., 0.);
        let (mut ixl, mut ixr, mut iyb, mut iyt);
        let mut iz = 0;
        let (mut fx, mut fy, mut fz) = (0, 0, 0);
        let (mut llx, mut lly, mut llz) = (0, 0, 0);
        let (mut xpad, mut ypad, mut zpad) = (0, 0, 0);
        let mut itmp;
        let bin = unsafe { (*self.vi).xybin };
        let flipped = unsafe { (*(*self.vi).li).axis } == 2;

        ivw_control_priority(unsafe { &mut *self.vi }, self.ctrl);
        with_boundary(|n| n.info_win_raise());
        if self.rubberband != 0 {
            xl = self.rb_image_x0;
            yb = self.rb_image_y0;
            xr = self.rb_image_x1;
            yt = self.rb_image_y1;
        } else {
            self.getixy(0, -1, &mut xl, &mut yt, &mut iz);
            self.getixy(self.winx, self.winy - 1, &mut xr, &mut yb, &mut iz);
        }
        ixl = (xl as f64 + 0.5).floor() as i32;
        ixr = (xr as f64 - 0.5).floor() as i32;
        iyb = (yb as f64 + 0.5).floor() as i32;
        iyt = (yt as f64 - 0.5).floor() as i32;
        let ifpad = ixl < 0
            || iyb < 0
            || ixr >= unsafe { (*self.vi).xsize }
            || iyt >= unsafe { (*self.vi).ysize };
        let imx = bin * (ixr + 1 - ixl);
        let imy = bin * (iyt + 1 - iyb);
        let ixcen = bin * (ixr + 1 + ixl) / 2;
        let iycen = bin * (iyt + 1 + iyb) / 2;
        let ixofs = ixcen - (bin * unsafe { (*self.vi).xsize }) / 2;
        let iyofs = iycen - (bin * unsafe { (*self.vi).ysize }) / 2;
        ixl = 0.max(ixl);
        iyb = 0.max(iyb);
        ixr = ixr.min(unsafe { (*self.vi).xsize } - 1);
        iyt = iyt.min(unsafe { (*self.vi).ysize } - 1);

        let mut low_section = 0;
        let mut high_section = 0;
        self.get_low_high_section(&mut low_section, &mut high_section);
        if low_section < 1 || high_section < 1 || high_section > unsafe { (*self.vi).zsize } {
            wprint(&format!(
                "ERROR: {} is out of range.\n",
                if flipped { "-y" } else { "-z" }
            ));
            if !to_info_window {
                return String::new();
            }
        }

        // Get load offsets and padding
        let time = ivw_window_time(unsafe { &*self.vi }, self.time_lock);
        if unsafe {
            ivw_get_image_padding(
                self.vi,
                (iyb + iyt) / 2,
                (low_section + high_section) / 2,
                time,
                &mut llx,
                &mut xpad,
                &mut fx,
                &mut lly,
                &mut ypad,
                &mut fy,
                &mut llz,
                &mut zpad,
                &mut fz,
            )
        } < 0
        {
            llx = 0;
            lly = 0;
            llz = 0;
            xpad = 0;
            ypad = 0;
            zpad = 0;
        }

        // If flipped, adjust the section numbers and swap the load offsets,
        // not the pads
        if flipped {
            itmp = unsafe { (*self.vi).zsize } + 1 - low_section;
            low_section = unsafe { (*self.vi).zsize } + 1 - high_section;
            high_section = itmp;
            itmp = llz;
            llz = lly;
            lly = itmp;
        }

        // Adjust for load offsets/padding
        ixl += llx - xpad;
        ixr += llx - xpad;
        iyb += lly - ypad;
        iyt += lly - ypad;
        low_section += llz - zpad;
        high_section += llz - zpad;

        // Adjust for binning
        ixl *= bin;
        iyb *= bin;
        ixr = ixr * bin + bin - 1;
        iyt = iyt * bin + bin - 1;
        low_section = unsafe { (*self.vi).zbin } * (low_section - 1) + 1;
        high_section *= unsafe { (*self.vi).zbin };

        let trimvol = format!(
            "  trimvol -x {},{} {} {},{} {} {},{}",
            ixl + 1,
            ixr + 1,
            if flipped { "-z" } else { "-y" },
            iyb + 1,
            iyt + 1,
            if flipped { "-rx -y" } else { "-z" },
            low_section,
            high_section
        );

        if to_info_window {
            wprint(&format!(
                "{}enter ({},{}); offset {},{}\n",
                if bin > 1 { "Unbinned c" } else { "C" },
                ixcen + 1,
                iycen + 1,
                ixofs,
                iyofs
            ));
            wprint(&format!(
                "{}mage size: {} x {};   To excise:\n",
                if ifpad { "Padded i" } else { "I" },
                imx,
                imy
            ));
            wprint(&format!("{trimvol}\n"));
        }
        trimvol
    }

    /// `ZapFuncs::resizeToFit` (`xzap.cpp:4137`); resize window to fit either
    /// whole image or part in rubber band.
    pub fn resize_to_fit(&mut self) {
        let (mut neww, mut newh);
        let (mut newdx, mut newdy);
        let mut usable_top = 0;
        let mut usable_left = 0;
        let (mut xl, mut xr, mut yb, mut yt);
        let (ww, wh) = with_boundary(|n| n.zap_window_size());
        let mut width = (ww as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        let mut height = (wh as f64 * self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        let pos = with_boundary(|n| n.ivw_restorable_geometry());
        let info_rect = with_boundary(|n| n.info_win_frame_geometry());
        let (ul, ut) = with_boundary(|n| n.dia_minimum_window_pos());
        usable_left = ul;
        usable_top = ut;
        let (usable_width, mut dx) = with_boundary(|n| n.dia_maximum_window_size());

        dx = pos[0];
        let dy = pos[1];
        if self.rubberband != 0 {
            /* If rubberbanding, set size to size of band, and offset image by
            difference between band and window center */
            self.band_image_to_mouse(0);
            xl = self.rb_image_x0;
            yb = self.rb_image_y0;
            xr = self.rb_image_x1;
            yt = self.rb_image_y1;
            neww = self.rb_mouse_x1 - 1 - self.rb_mouse_x0 + width - self.winx;
            newh = self.rb_mouse_y1 - 1 - self.rb_mouse_y0 + height - self.winy;
            self.xtrans = (-(xr + xl - unsafe { (*self.vi).xsize } as f32) / 2.) as i32;
            self.ytrans = (-(yt + yb - unsafe { (*self.vi).ysize } as f32) / 2.) as i32;

            // 3/6/05: turn off through common function to keep synchronized
            self.toggle_rubberband(false);
        } else {
            /* Otherwise, make window the right size for the image */
            neww = (self.zoom * unsafe { (*self.vi).xsize } as f32 + (width - self.winx) as f32)
                as i32;
            newh = (self.zoom * unsafe { (*self.vi).ysize } as f32 + (height - self.winy) as f32)
                as i32;
        }
        neww = (neww as f64 / self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        newh = (newh as f64 / self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        width = (width as f64 / self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        height = (height as f64 / self.device_pixel_ratio as f64 + 0.5).floor() as i32;

        with_boundary(|n| n.dia_limit_window_size(&mut neww, &mut newh));
        newdx = dx + width / 2 - neww / 2;
        newdy = dy + height / 2 - newh / 2;
        with_boundary(|n| n.dia_limit_window_pos(neww, newh, &mut newdx, &mut newdy));

        // Restrain size if info is near edge and window is big enough
        let (info_w, _info_h) = with_boundary(|n| n.info_win_size());
        let frame_width = info_rect[2] - info_w;
        let info_left = info_rect[0];
        let info_right = info_rect[0] + info_rect[2] - 1;
        if usable_width > S_MIN_X_FOR_RESTRAINING.get()
            && (info_left < usable_left + 50 || info_right > usable_left + usable_width - 50)
        {
            if info_left < usable_left + 50 {
                neww = neww.min((usable_left + usable_width - info_right) - frame_width);
            } else {
                neww = neww.min((info_left - usable_left) - frame_width);
            }
        }

        // In any case, if window can fit beside info and it would occlude it,
        // shift it over
        let new_width = neww + frame_width;
        let usable_right = usable_left + usable_width;
        let new_right = newdx + new_width;
        if newdx < info_right && new_right >= info_left {
            // Get amount by which info would be occluded if on either side
            let occludes_left = new_width - (info_left - usable_left);
            let occludes_right = new_width - (usable_right - info_right);

            // If both sides work, keep it on the same side
            if occludes_left < 0 && occludes_right < 0 {
                if (newdx + new_right) / 2 < (info_left + info_right) / 2 {
                    newdx = info_left - new_width;
                } else {
                    newdx = info_right;
                }

            // Otherwise if one side works at all and is better, put it there
            } else if occludes_left < occludes_right && occludes_left < info_rect[2] {
                newdx = usable_left.max(info_left - new_width);
            } else if occludes_right < occludes_left && occludes_right < info_rect[2] {
                newdx = info_right.min(usable_right - new_width);
            }
        }

        if imod_debug('z') {
            imod_print_stderr("configuring widget...");
        }

        with_boundary(|n| n.zap_window_resize(neww, newh));
        with_boundary(|n| n.zap_window_move(newdx, newdy));

        /* DNM 9/12/03: remove the ZAP_EXPOSE_HACK, and a second set geometry
        that was needed temporarily with Qt 3.2.1 on Mac */

        if imod_debug('z') {
            imod_print_stderr("back\n");
        }
    }

    /// `ZapFuncs::setControlAndLimits` (`xzap.cpp:4234`); set the control
    /// priority and set flag to record subarea and do float.
    pub fn set_control_and_limits(&mut self) {
        ivw_control_priority(unsafe { &mut *self.vi }, self.ctrl);
        if self.num_xpanels == 0 {
            self.record_subarea = 1;
        }
    }

    /// `ZapFuncs::setAreaLimits` (`xzap.cpp:4244`); record the limits of the
    /// image displayed in the window or in the rubber band.
    pub fn set_area_limits(&mut self) {
        let mut iz = 0;
        let min_area = 16;
        let (mut xl, mut xr, mut yb, mut yt) = (0., 0., 0., 0.);
        let mut delta;
        if self.rubberband != 0 {
            xl = self.rb_image_x0;
            yb = self.rb_image_y0;
            xr = self.rb_image_x1;
            yt = self.rb_image_y1;

            // Enforce a minimum size so
            delta = (min_area as f32 - (xr - xl)) / 2.;
            if delta > 0. {
                xl -= delta;
                xr += delta;
            }
            delta = (min_area - (yt - yb) as i32) as f32 / 2.;
            if delta > 0. {
                yb -= delta;
                yt += delta;
            }
        } else {
            self.getixy(0, 0, &mut xl, &mut yt, &mut iz);
            self.getixy(self.winx, self.winy, &mut xr, &mut yb, &mut iz);
        }
        S_SUB_START_X.set(((xl as f64 + 0.5) as i32).max(0));
        S_SUB_END_X.set(((xr as f64 - 0.5) as i32).min(unsafe { (*self.vi).xsize } - 1));
        S_SUB_START_Y.set(((yb as f64 + 0.5) as i32).max(0));
        S_SUB_END_Y.set(((yt as f64 - 0.5) as i32).min(unsafe { (*self.vi).ysize } - 1));
        if imod_debug('z') {
            imod_print_stderr(&format!(
                "Set area {} {} {} {}\n",
                S_SUB_START_X.get(),
                S_SUB_END_X.get(),
                S_SUB_START_Y.get(),
                S_SUB_END_Y.get()
            ));
        }
    }

    /// `ZapFuncs::namedSnapshot` (`xzap.cpp:4282`); an external call for
    /// taking a snapshot of given format and given an optional name.
    pub fn named_snapshot(
        &mut self,
        fname: &mut String,
        format: i32,
        check_convert: bool,
        full_area: bool,
    ) -> i32 {
        let mut limits = None;
        let mut limarr = [0; 4];
        self.showslice = self.showed_slice as i16;
        self.draw();
        if !full_area {
            limits = self.set_snapshot_limits(&mut limarr);
        }
        with_boundary(|n| n.b3d_named_snapshot(fname, format, limits, check_convert))
    }

    /// `ZapFuncs::zoomedDownImage` (`xzap.cpp:4297`); returns size and limits
    /// within rubberband of the zoomed down image being displayed, and
    /// corresponding limits for the unzoomed stored image.  A `false` return
    /// is the source's NULL; `true` is `mImage`.
    #[allow(clippy::too_many_arguments)]
    pub fn zoomed_down_image(
        &mut self,
        subset: i32,
        nxim: &mut i32,
        nyim: &mut i32,
        ix_start: &mut i32,
        iy_start: &mut i32,
        nx_use: &mut i32,
        ny_use: &mut i32,
        uz_xstart: &mut i32,
        uz_ystart: &mut i32,
        uz_xuse: &mut i32,
        uz_yuse: &mut i32,
    ) -> bool {
        let (mut ll_x, mut left_xpad, mut right_xpad) = (0, 0, 0);
        let (mut ll_y, mut left_ypad, mut right_ypad) = (0, 0, 0);
        let (mut ll_z, mut left_zpad, mut right_zpad) = (0, 0, 0);
        let mut uz_xend = 0;
        let mut uz_yend;
        let (mut xl, mut xr, mut yb, mut yt) = (0., 0., 0., 0.);
        let time = ivw_window_time(unsafe { &*self.vi }, self.time_lock);

        let rgba = APP.lock().unwrap().as_ref().map_or(0, |a| a.rgba);
        if self.hqgfx == 0
            || self.zoom > with_boundary(|n| n.b3d_zoom_down_crit())
            || rgba == 0
            || !self.image
            || unsafe { (*(*self.vi).cramp).falsecolor } != 0
        {
            return false;
        }
        *ix_start = 1;
        *iy_start = 1;
        *nxim = (self.xdrawsize as f32 * self.zoom) as i32;
        *nyim = (self.ydrawsize as f32 * self.zoom) as i32;
        *nx_use = *nxim - 2;
        *ny_use = *nyim - 2;
        if subset != 0 && self.rubberband != 0 {
            self.band_image_to_mouse(1);
            *ix_start = 1.max(self.rb_mouse_x0 + 1 - self.xborder);
            *iy_start = 1.max(self.winy - self.rb_mouse_y1 + 1 - self.yborder);
            *nx_use = (*nx_use - *ix_start).min(self.rb_mouse_x1 - self.rb_mouse_x0 - 1);
            *ny_use = (*ny_use - *iy_start).min(self.rb_mouse_y1 - self.rb_mouse_y0 - 1);
        }

        // Get window limits of unzoomed image and limit by the possible rubber
        // band subarea
        self.getixy(0, 0, &mut xl, &mut yt, &mut uz_xend);
        self.getixy(self.winx, self.winy, &mut xr, &mut yb, &mut uz_xend);
        *uz_xstart = ((xl as f64 + 0.5) as i32).max(0);
        uz_xend = ((xr as f64 - 0.5) as i32).min(unsafe { (*self.vi).xsize } - 1);
        *uz_ystart = ((yb as f64 + 0.5) as i32).max(0);
        uz_yend = ((yt as f64 - 0.5) as i32).min(unsafe { (*self.vi).ysize } - 1);
        if subset != 0 {
            *uz_xstart = (*uz_xstart).max(S_SUB_START_X.get());
            uz_xend = uz_xend.min(S_SUB_END_X.get());
            *uz_ystart = (*uz_ystart).max(S_SUB_START_Y.get());
            uz_yend = uz_yend.min(S_SUB_END_Y.get());
        }
        *uz_xuse = uz_xend + 1 - *uz_xstart;
        *uz_yuse = uz_yend + 1 - *uz_ystart;

        // Get extent of padded region in image and use it to limit the subarea
        if unsafe {
            ivw_get_image_padding(
                self.vi,
                -1,
                self.section,
                time,
                &mut ll_x,
                &mut left_xpad,
                &mut right_xpad,
                &mut ll_y,
                &mut left_ypad,
                &mut right_ypad,
                &mut ll_z,
                &mut left_zpad,
                &mut right_zpad,
            )
        } != 0
        {
            return self.image;
        }

        if left_xpad != 0 || left_ypad != 0 || right_xpad != 0 || right_ypad != 0 {
            crate::imod::three_dmod::info_cb::imod_info_limit_subarea(
                self.xpos(left_xpad as f32) - self.xborder,
                self.xpos((unsafe { (*self.vi).xsize } - right_xpad) as f32) - self.xborder,
                self.ypos(left_ypad as f32) - self.yborder,
                self.ypos((unsafe { (*self.vi).ysize } - right_ypad) as f32) - self.yborder,
                ix_start,
                iy_start,
                nx_use,
                ny_use,
            );
            crate::imod::three_dmod::info_cb::imod_info_limit_subarea(
                left_xpad,
                unsafe { (*self.vi).xsize } - right_xpad,
                left_ypad,
                unsafe { (*self.vi).ysize } - right_ypad,
                uz_xstart,
                uz_ystart,
                uz_xuse,
                uz_yuse,
            );
        }
        self.image
    }

    /// `ZapFuncs::toggleRubberband` (`xzap.cpp:4357`).
    pub fn toggle_rubberband(&mut self, draw_win: bool) {
        if self.rubberband != 0 || self.starting_band != 0 {
            self.rubberband = 0;
            self.starting_band = 0;
            self.band_changed = 1;
            self.set_control_and_limits();
        } else {
            if self.lasso_on {
                self.toggle_lasso(false);
            }
            if self.drawing_arrow {
                self.toggle_arrow(false);
            }
            self.starting_band = 1;
            self.end_contour_shift();
            /* Eliminated old code for making initial band */
        }

        self.set_mouse_tracking();
        let state = self.rubberband + self.starting_band;
        with_boundary(|n| n.zap_window_set_low_high_section_state(state));

        // 3/6/05: synchronize the toolbar button
        with_boundary(|n| n.zap_window_set_toggle_state(ZAP_TOGGLE_RUBBER, state));
        let set_anyway = with_boundary(|n| n.util_need_to_set_cursor());
        self.set_cursor(self.mousemode, set_anyway);
        if draw_win {
            self.draw();
        }

        // 4/6/09: This was needed before and after the draw for Mac Qt 4.5.0
        let set_anyway = with_boundary(|n| n.util_need_to_set_cursor());
        self.set_cursor(self.mousemode, set_anyway);
    }

    /// `ZapFuncs::shiftRubberband` (`xzap.cpp:4391`); shift the rubberband by
    /// a desired amount to the extent possible.
    pub fn shift_rubberband(&mut self, mut idx: f32, mut idy: f32) {
        if self.rb_image_x0 + idx < 0. {
            idx = -self.rb_image_x0;
        }
        if self.rb_image_x1 + idx > unsafe { (*self.vi).xsize } as f32 {
            idx = unsafe { (*self.vi).xsize } as f32 - self.rb_image_x1;
        }
        if self.rb_image_y0 + idy < 0. {
            idy = -self.rb_image_y0;
        }
        if self.rb_image_y1 + idy > unsafe { (*self.vi).ysize } as f32 {
            idy = unsafe { (*self.vi).ysize } as f32 - self.rb_image_y1;
        }
        self.rb_image_x0 += idx;
        self.rb_image_x1 += idx;
        self.rb_image_y0 += idy;
        self.rb_image_y1 += idy;
    }

    /// `ZapFuncs::toggleLasso` (`xzap.cpp:4410`).
    pub fn toggle_lasso(&mut self, draw_win: bool) {
        if self.lasso_on {
            ivw_free_extra_object(unsafe { &mut *self.vi }, self.lasso_obj_num);
        } else {
            if self.rubberband != 0 || self.starting_band != 0 {
                self.toggle_rubberband(false);
            }
            if self.drawing_arrow {
                self.toggle_arrow(false);
            }
            self.end_contour_shift();
            self.lasso_obj_num = ivw_get_free_extra_object_number(unsafe { &mut *self.vi });
        }
        self.lasso_on = !self.lasso_on;
        self.drawing_lasso = self.lasso_on;
        self.set_mouse_tracking();
        let state = i32::from(self.lasso_on);
        with_boundary(|n| n.zap_window_set_toggle_state(ZAP_TOGGLE_LASSO, state));

        // Set it to a modeling cursor
        self.set_cursor(self.mousemode, true);
        if draw_win {
            self.draw();
        }
        if with_boundary(|n| n.imodv_isosurface_update(IMOD_DRAW_MOD)) {
            with_boundary(|n| n.imodv_draw());
        }
        self.set_cursor(self.mousemode, true);
    }

    /// `ZapFuncs::toggleArrow` (`xzap.cpp:4439`); toggle the arrow, turning
    /// off lasso or rubberband if they are starting.
    pub fn toggle_arrow(&mut self, draw_win: bool) {
        let zero = Ipoint {
            x: 0.,
            y: 0.,
            z: 0.,
        };
        if !self.arrow_on {
            if self.starting_band != 0 {
                self.toggle_rubberband(false);
            }
            if self.drawing_lasso {
                self.toggle_lasso(false);
            }
            self.end_contour_shift();
            self.arrow_head.push(zero);
            self.arrow_tail.push(zero);
        } else {
            self.arrow_head.pop();
            self.arrow_tail.pop();
        }
        self.arrow_on = !self.arrow_on;
        self.drawing_arrow = self.arrow_on;
        let state = i32::from(self.arrow_on);
        with_boundary(|n| n.zap_window_set_toggle_state(ZAP_TOGGLE_ARROW, state));

        self.set_cursor(self.mousemode, true);
        if draw_win {
            self.draw();
        }
        self.set_cursor(self.mousemode, true);
    }

    /// `ZapFuncs::clearArrows` (`xzap.cpp:4467`); clear out all arrows.
    pub fn clear_arrows(&mut self) {
        if self.arrow_on {
            self.toggle_arrow(false);
        }
        self.arrow_head.clear();
        self.arrow_tail.clear();
        self.draw();
    }

    /// `ZapFuncs::startAddedArrow` (`xzap.cpp:4479`); start a new arrow,
    /// keeping an existing one.
    ///
    /// `xzap.cpp:4481-4484` indexes `mArrowHead[ind]` with
    /// `ind = mArrowHead.size()`, one past the last element; the read is out
    /// of range in the source too, so the guarded read below stands in for it
    /// rather than reproducing the overrun.
    pub fn start_added_arrow(&mut self) {
        let ind = self.arrow_head.len();
        let head = self.arrow_head.get(ind).copied().unwrap_or_default();
        let tail = self.arrow_tail.get(ind).copied().unwrap_or_default();
        if self.drawing_arrow && head.x == 0. && tail.x == 0. && head.y == 0. && tail.y == 0. {
            return;
        }
        self.arrow_on = false;
        self.toggle_arrow(false);
    }

    /// `ZapFuncs::getLassoContour` (`xzap.cpp:4492`).
    pub fn get_lasso_contour(&mut self) -> *mut Icont {
        if !self.lasso_on {
            return ptr::null_mut();
        }
        let obj = ivw_get_an_extra_object(unsafe { &mut *self.vi }, self.lasso_obj_num);
        match obj {
            Some(obj) if !obj.cont.is_empty() => obj.cont.as_mut_ptr(),
            _ => ptr::null_mut(),
        }
    }

    /// `ZapFuncs::setMouseTracking` (`xzap.cpp:4504`); set mouse tracking
    /// based on state of all governing flags.
    pub fn set_mouse_tracking(&mut self) {
        let state = S_INSERT_DOWN.get() != 0
            || self.rubberband != 0
            || (self.lasso_on && !self.drawing_lasso)
            || S_PIXEL_VIEW_OPEN.get()
            || unsafe { (*self.vi).track_mouse_for_plugs } != 0;
        with_boundary(|n| n.gfx_set_mouse_tracking(state));
    }

    /// `ZapFuncs::externalSetSize` (`xzap.cpp:4513`); external call (like from
    /// `client_message`) to set window size, in device pixels.
    pub fn external_set_size(&mut self, width: i32, height: i32) {
        let mut width = (width as f64 / self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        let mut height = (height as f64 / self.device_pixel_ratio as f64 + 0.5).floor() as i32;
        if with_boundary(|n| n.zap_window_toolbar_exists(3))
            && with_boundary(|n| n.zap_window_toolbar_exists(1))
        {
            height += with_boundary(|n| n.zap_window_toolbar_height(3))
                + with_boundary(|n| n.zap_window_toolbar_height(1));
        }
        with_boundary(|n| n.dia_limit_window_size(&mut width, &mut height));
        let pos = with_boundary(|n| n.zap_window_frame_geometry());
        let mut xpos = pos[0];
        let mut ypos = pos[1];
        with_boundary(|n| n.dia_limit_window_pos(width, height, &mut xpos, &mut ypos));
        with_boundary(|n| n.zap_window_move(xpos, ypos));
        with_boundary(|n| n.zap_window_resize(width, height));
    }

    /// `ZapFuncs::setMultiZpanels` (`xzap.cpp:4531`); external call to set the
    /// number of panels in a multi-Z window.
    pub fn set_multi_z_panels(&mut self, num_x: i32, num_y: i32) {
        if self.num_xpanels == 0 || num_x < 1 || num_y < 1 || num_x * num_y < 2 {
            return;
        }
        self.num_xpanels = num_x;
        self.num_ypanels = num_y;
        if self.setup_panels() == 0 {
            if with_boundary(|n| n.zap_window_toolbar_exists(3)) {
                with_boundary(|n| n.zap_window_set_spin_boxes(num_x, num_y));
            }
            with_boundary(|n| n.gfx_update_gl());
        }
    }

    /// `ZapFuncs::montageSnapshot` (`xzap.cpp:4549`); take a snapshot by
    /// montaging at higher zoom.
    pub fn montage_snapshot(&mut self, snaptype: i32) {
        let (mut x_full_size, mut y_full_size) = (0, 0);
        let (mut x_trans_start, mut y_trans_start) = (0, 0);
        let (mut x_trans_delta, mut y_trans_delta) = (0, 0);
        let (mut x_copy_delta, mut y_copy_delta) = (0, 0);
        let mut num_chunks = 0;
        let (mut overhalf, mut x_copy, mut y_copy);
        let (mut from_xoff, mut from_yoff, mut to_xoff, mut to_yoff);
        let band_save = self.rubberband;
        let mut factor = with_boundary(|n| n.imc_get_montage_factor());
        let mut scaling_factor = factor as f32;
        let mut use_xsize;
        let mut use_ysize;
        let snap_whole_or_sub = with_boundary(|n| n.imc_get_snap_whole_mont());

        // Save translations and zoom
        let x_trans_save = self.xtrans;
        let y_trans_save = self.ytrans;
        let zoom_save = self.zoom as f64;
        let hq_save = self.hqgfx;

        use_xsize = unsafe { (*self.vi).xsize };
        use_ysize = unsafe { (*self.vi).ysize };
        if snap_whole_or_sub != 0 {
            if snap_whole_or_sub > 1 {
                self.set_area_limits();
                use_xsize = S_SUB_END_X.get() + 1 - S_SUB_START_X.get();
                use_ysize = S_SUB_END_Y.get() + 1 - S_SUB_START_Y.get();
            }
            factor = 1;
            while factor < 31 {
                if factor * self.winx >= use_xsize && factor * self.winy >= use_ysize {
                    break;
                }
                factor += 1;
            }
            if factor == 1 {
                wprint(&format!(
                    "\u{7}{} already fits in window at zoom 1, no montage needed.\n",
                    if snap_whole_or_sub > 1 {
                        "Subarea"
                    } else {
                        "Image"
                    }
                ));
                return;
            }
            if factor == 31 {
                wprint(&format!(
                    "\u{7}{} is too large for full montage snapshot in this window.\n",
                    if snap_whole_or_sub > 1 {
                        "Subarea"
                    } else {
                        "Image"
                    }
                ));
                return;
            }
            scaling_factor = (1. / zoom_save) as f32;
            self.zoom = (1. / factor as f64) as f32;
            if snap_whole_or_sub == 1 {
                self.draw();
            }
        }

        // Get coordinates and offsets and buffers
        self.set_area_limits();
        let (start_x, end_x) = if self.rubberband != 0 {
            (self.xstart.max(S_SUB_START_X.get()), S_SUB_END_X.get())
        } else {
            (self.xstart, -1)
        };
        let (start_y, end_y) = if self.rubberband != 0 {
            (self.ystart.max(S_SUB_START_Y.get()), S_SUB_END_Y.get())
        } else {
            (self.ystart, -1)
        };
        let xsize = unsafe { (*self.vi).xsize };
        let ysize = unsafe { (*self.vi).ysize };
        if self.get_montage_shifts(
            factor,
            start_x,
            self.xborder,
            xsize,
            self.winx,
            end_x,
            &mut x_trans_start,
            &mut x_trans_delta,
            &mut x_copy_delta,
            &mut x_full_size,
        ) != 0
            || self.get_montage_shifts(
                factor,
                start_y,
                self.yborder,
                ysize,
                self.winy,
                end_y,
                &mut y_trans_start,
                &mut y_trans_delta,
                &mut y_copy_delta,
                &mut y_full_size,
            ) != 0
        {
            wprint("\u{7}There is too much border around image for montage snapshot.\n");
            return;
        }
        with_boundary(|n| n.scale_bar_save_or_restore(true));
        let (wx, wy) = (self.winx, self.winy);
        if with_boundary(|n| {
            n.util_start_mont_snap(
                wx,
                wy,
                x_full_size,
                y_full_size,
                scaling_factor,
                &mut num_chunks,
            )
        }) != 0
        {
            wprint("\u{7}Failed to get memory for snapshot buffers.\n");
            return;
        }

        // 8/22/14: Move turning off autoswap to calling routines

        // Set up scaling
        if with_boundary(|n| n.imc_get_scale_sizes()) {
            S_SCALE_SIZES.set(with_boundary(|n| n.imc_get_size_scaling()));
            if S_SCALE_SIZES.get() == 1 {
                S_SCALE_SIZES.set(
                    (scaling_factor as f64 * (1.0f64).min(self.zoom as f64) + 0.5).floor() as i32,
                );
            }
        }

        // Loop on frames, getting pixels and copying them
        self.hqgfx = 1;
        self.zoom *= factor as f32;
        let show_slice = self.showslice;
        self.rubberband = 0;
        self.doing_montage = true;
        let bar_draw = with_boundary(|n| n.scale_bar_draw_flag());
        for iy in 0..factor {
            for ix in 0..factor {
                // Set up for scale bar if it is the right corner
                let (bx, by, zoom) = (self.winx - 4, self.winy - 4, self.zoom);
                with_boundary(|n| {
                    n.util_mont_snap_scale_bar(ix, iy, factor, bx, by, zoom, bar_draw)
                });

                self.xtrans = -(x_trans_start + ix * x_trans_delta);
                self.ytrans = -(y_trans_start + iy * y_trans_delta);
                self.draw();
                with_boundary(|n| n.gl_flush());
                with_boundary(|n| n.gl_finish());
                with_boundary(|n| n.imod_info_input());
                // The `QT_VERSION >= 0x060000` second draw (`xzap.cpp:4675`)
                // is not compiled against Qt 5.
                self.showslice = show_slice;
                let (wx, wy) = (self.winx, self.winy);
                with_boundary(|n| n.b3d_set_cur_size(wx, wy));

                // Print scale bar length if it was drawn
                if self.scale_bar_size > 0. {
                    imod_print_stderr(&format!(
                        "Scale bar for montage is {} {}\n",
                        self.scale_bar_size,
                        unsafe { std::ffi::CStr::from_ptr(imod_units(&*(*self.vi).imod)) }
                            .to_string_lossy()
                    ));
                }

                with_boundary(|n| n.gl_flush());
                with_boundary(|n| n.gl_finish());
                let frame_pix = with_boundary(|n| n.util_mont_snap_frame_pix());
                with_boundary(|n| n.gl_read_pixels(wx, wy, frame_pix));
                with_boundary(|n| n.gl_flush());
                with_boundary(|n| n.gl_finish());

                // set up copy parameters for full copy, and adjust to skip the
                // overlap after the first piece unless doing panel with bar
                x_copy = self.winx;
                y_copy = self.winy;
                to_xoff = ix * x_copy_delta;
                to_yoff = iy * y_copy_delta;
                from_xoff = 0;
                from_yoff = 0;
                if ix != 0 && !(bar_draw && ix == factor - 1) {
                    overhalf = self.winx - x_copy_delta - 2;
                    if overhalf > 2 && overhalf < x_copy {
                        from_xoff = overhalf;
                        to_xoff += overhalf;
                        x_copy -= overhalf;
                    }
                }
                if iy != 0 && !(bar_draw && iy == factor - 1) {
                    overhalf = self.winy - y_copy_delta - 2;
                    if overhalf > 2 && overhalf < y_copy {
                        from_yoff = overhalf;
                        to_yoff += overhalf;
                        y_copy -= overhalf;
                    }
                }

                with_boundary(|n| {
                    n.util_mont_snap_copy_frame(
                        x_copy, y_copy, to_xoff, to_yoff, wx, from_xoff, from_yoff,
                    )
                });
                if with_boundary(|n| n.app_doublebuffer()) {
                    with_boundary(|n| n.gfx_swap_buffers());
                }
            }
        }
        self.doing_montage = false;

        // Reset the file number to zero unless doing movie, then get name and
        // save
        if self.movie_snap_count == 0 {
            S_MONT_FILENO.set(0);
        }

        // Save the image then restore display
        let mut fileno = S_MONT_FILENO.get();
        with_boundary(|n| {
            n.util_finish_mont_snap(
                x_full_size,
                y_full_size,
                snaptype - 1,
                &mut fileno,
                3,
                factor as f32,
            )
        });
        S_MONT_FILENO.set(fileno);

        with_boundary(|n| n.scale_bar_save_or_restore(false));
        self.xtrans = x_trans_save;
        self.ytrans = y_trans_save;
        self.zoom = zoom_save as f32;
        self.rubberband = band_save;
        S_SCALE_SIZES.set(1);
        self.hqgfx = hq_save;
        self.set_area_limits();
        self.draw();
        with_boundary(|n| n.util_free_mont_snap_arrays(num_chunks));
    }

    /// `ZapFuncs::getMontageShifts` (`xzap.cpp:4725`); compute shifts and
    /// increments for the montage snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn get_montage_shifts(
        &mut self,
        factor: i32,
        im_start: i32,
        border: i32,
        im_size: i32,
        win_size: i32,
        band_end: i32,
        trans_start: &mut i32,
        trans_delta: &mut i32,
        copy_delta: &mut i32,
        full_size: &mut i32,
    ) -> i32 {
        let (mut wofftmp, mut dstmp, mut dofftmp) = (0, 0, 0);
        let mut im_end = (im_start + ((win_size - border) as f32 / self.zoom) as i32).min(im_size);
        if band_end > 0 {
            im_end = band_end;
        }
        let in_win = (win_size as f32 / (self.zoom * factor as f32)) as i32;
        if in_win >= im_size - factor {
            return 1;
        }

        // Get trans and back it off to avoid window offset on left
        let mut trans = -(im_start + (in_win - im_size) / 2);
        b3d_set_image_offset(
            win_size,
            im_size,
            (self.zoom * factor as f32) as f64,
            &mut dstmp,
            &mut trans,
            &mut wofftmp,
            &mut dofftmp,
            0,
        );
        if wofftmp > 0 {
            trans -= 1;
        }
        *trans_start = -trans;

        // Get overlap and delta, back off delta to avoid extra pixels on right
        let overlap = 0.max((factor * in_win + im_start - im_end) / (factor - 1));
        *trans_delta = in_win - overlap;
        *copy_delta = (self.zoom * factor as f32 * *trans_delta as f32) as i32;
        *full_size = (factor - 1) * *copy_delta + win_size;
        if *full_size > (self.zoom * factor as f32 * (im_size - im_start) as f32) as i32 {
            *trans_delta -= 1;
            *copy_delta = (self.zoom * factor as f32 * *trans_delta as f32) as i32;
            *full_size = (factor - 1) * *copy_delta + win_size;
        }

        if imod_debug('z') {
            imod_print_stderr(&format!(
                "im {im_start} - {im_end}  bord {border} win {win_size}  inwin {in_win} overlap {overlap} start {} delta {}  copy {}  full {}\n",
                *trans_start, *trans_delta, *copy_delta, *full_size
            ));
        }
        0
    }

    /****************************************************************************/
    /* drawing routines.                                                        */

    /// `ZapFuncs::drawGraphics` (`xzap.cpp:4768`); draws the image.
    pub fn draw_graphics(&mut self) {
        let vi = self.vi;
        let (mut bl, mut wh, mut ind, mut iz);
        let mut rgba = APP.lock().unwrap().as_ref().map_or(0, |a| a.rgba);
        let mut image_data: *mut *mut u8 = ptr::null_mut();
        let mut over_image: *mut u8 = ptr::null_mut();
        let mut overlay = 0;
        let other_sec = self.section + unsafe { (*vi).overlay_sec };
        let mut zoom;
        let (mut x_draw_size, mut y_draw_size, mut x_start, mut y_start) = (0, 0, 0, 0);
        let mut tile_scale = 1;
        let mut status = 0;
        let (mut im_xsize, mut im_ysize);
        let (mut x_offset, mut y_offset) = (0., 0.);
        let async_load = unsafe { (*self.vi).zmovie } == 0
            && with_boundary(|n| n.imod_dialog_manager_window_count(ZAP_WINDOW_TYPE)) == 1
            && !with_boundary(|n| n.mv_image_drawing_zplanes())
            && self.num_xpanels == 0
            && unsafe { (*self.vi).loading_image } == 0;

        zoom = self.zoom as f64;
        im_xsize = unsafe { (*vi).xsize };
        im_ysize = unsafe { (*vi).ysize };

        let win_x = if self.num_xpanels != 0 {
            self.panel_xsize
        } else {
            self.winx
        };
        let (mut xdrawsize, mut xtrans, mut xborder, mut xstart) =
            (self.xdrawsize, self.xtrans, self.xborder, self.xstart);
        b3d_set_image_offset(
            win_x,
            unsafe { (*vi).xsize },
            self.zoom as f64,
            &mut xdrawsize,
            &mut xtrans,
            &mut xborder,
            &mut xstart,
            1,
        );
        self.xdrawsize = xdrawsize;
        self.xtrans = xtrans;
        self.xborder = xborder;
        self.xstart = xstart;

        let win_y = if self.num_xpanels != 0 {
            self.panel_ysize
        } else {
            self.winy
        };
        let (mut ydrawsize, mut ytrans, mut yborder, mut ystart) =
            (self.ydrawsize, self.ytrans, self.yborder, self.ystart);
        b3d_set_image_offset(
            win_y,
            unsafe { (*vi).ysize },
            self.zoom as f64,
            &mut ydrawsize,
            &mut ytrans,
            &mut yborder,
            &mut ystart,
            1,
        );
        self.ydrawsize = ydrawsize;
        self.ytrans = ytrans;
        self.yborder = yborder;
        self.ystart = ystart;
        self.xpos_start = self.xstart;
        self.ypos_start = self.ystart;

        /* Get the time to display and flush if time is different. */
        let time = ivw_window_time(unsafe { &*vi }, self.time_lock);
        if time != self.time {
            self.flush_image();
        }

        // For tile cache, go ahead and get the section area even if doing
        // panels in order to get all the drawing parameters modified once
        if !unsafe { (*vi).pyr_cache }.is_null() {
            with_boundary(|n| n.gfx_cancel_redraw());
            let (section, xs, ys, xd, yd, zm) = (
                self.section,
                self.xstart,
                self.ystart,
                self.xdrawsize,
                self.ydrawsize,
                self.zoom,
            );
            image_data = with_boundary(|n| {
                n.pyr_cache_get_section_area(
                    vi,
                    section,
                    xs,
                    ys,
                    xd,
                    yd,
                    zm,
                    async_load,
                    &mut x_draw_size,
                    &mut y_draw_size,
                    &mut x_offset,
                    &mut y_offset,
                    &mut tile_scale,
                    &mut status,
                )
            });
            x_start = 0;
            y_start = 0;
            zoom = self.zoom as f64 * tile_scale as f64;
            im_xsize = x_draw_size;
            im_ysize = y_draw_size;
            self.xpos_start += x_offset as i32;
            self.ypos_start += y_offset as i32;
            if self.xlast_start != self.xstart
                || self.xlast_size != self.xdrawsize
                || self.ylast_start != self.ystart
                || self.ylast_size != self.ydrawsize
                || self.last_status > 0
            {
                self.flush_image();
            }
            self.xlast_start = self.xstart;
            self.xlast_size = self.xdrawsize;
            self.ylast_start = self.ystart;
            self.ylast_size = self.ydrawsize;
            self.last_status = status;
            // Old comment was "100 was way too often with debug version": now
            // make sure the redraws do not take more than 1/5 of the time away
            // from loading
            if status > 0 {
                let msec = 200.max(4 * with_boundary(|n| n.gfx_get_last_draw_msec()));
                with_boundary(|n| n.gfx_schedule_redraw(msec));
            }
        } else {
            x_start = self.xstart;
            x_draw_size = self.xdrawsize;
            y_start = self.ystart;
            y_draw_size = self.ydrawsize;
        }

        if self.num_xpanels == 0 {
            if unsafe { (*vi).pyr_cache }.is_null() {
                image_data = unsafe { ivw_get_z_section_time(vi, self.section, time) };
            }

            // If flag set, record the subarea size, clear flag, and do call
            // float to set the color map if necessary.  If the black/white
            // changes, flush image
            if self.record_subarea != 0 {
                // Set the X zoom to zoom if it hasn't been set yet or is not
                // close to zoom
                if self.xzoom == 0.
                    || (1. / self.xzoom as f64 - 1. / self.zoom as f64).abs() >= 0.002
                {
                    self.xzoom = self.zoom;
                }
                self.set_area_limits();
                let section = self.section;
                if !self.doing_montage
                    && with_boundary(|n| n.imod_info_bwfloat(vi, section, time)) != 0
                    && rgba != 0
                {
                    with_boundary(|n| n.b3d_flush_image(-1));
                }
            }

            let (bx, by, ex, ey) = (
                self.xborder,
                self.yborder,
                self.xborder + (x_draw_size as f64 * zoom) as i32,
                self.yborder + (y_draw_size as f64 * zoom) as i32,
            );
            with_boundary(|n| n.b3d_draw_boxout(bx, by, ex, ey));

            // If overlay section is set and legal, get an image buffer and
            // fill it with the color overlay
            if unsafe { (*vi).overlay_sec } != 0
                && rgba != 0
                && unsafe { (*vi).rgb_store } == 0
                && other_sec >= 0
                && other_sec < unsafe { (*vi).zsize }
                && unsafe { (*vi).pyr_cache }.is_null()
            {
                over_image = unsafe {
                    libc::malloc(3 * (*vi).xsize as usize * (*vi).ysize as usize) as *mut u8
                };
                if over_image.is_null() {
                    wprint("\u{7}Failed to get memory for overlay image.\n");
                } else {
                    overlay = unsafe { (*vi).overlay_sec };
                    rgba = 3;
                    let (nx, ny) = (unsafe { (*vi).xsize }, unsafe { (*vi).ysize });
                    if unsafe { (*vi).which_green } != 0 {
                        self.fill_overlay_rgb(image_data, nx, ny, 0, over_image);
                        self.fill_overlay_rgb(image_data, nx, ny, 2, over_image);
                    } else {
                        self.fill_overlay_rgb(image_data, nx, ny, 1, over_image);
                    }

                    image_data = unsafe { ivw_get_z_section_time(vi, other_sec, time) };
                    if unsafe { (*vi).which_green } != 0 {
                        self.fill_overlay_rgb(image_data, nx, ny, 1, over_image);
                    } else {
                        self.fill_overlay_rgb(image_data, nx, ny, 0, over_image);
                        self.fill_overlay_rgb(image_data, nx, ny, 2, over_image);
                    }
                    image_data =
                        unsafe { ivw_make_line_pointers(vi, over_image, nx, ny, MRC_MODE_RGB) };
                }
            }
            if overlay != self.overlay {
                with_boundary(|n| n.b3d_flush_image(-1));
            }
            self.overlay = overlay;

            let (bx, by) = (self.xborder, self.yborder);
            let rampbase = unsafe { (*vi).rampbase };
            let hq = self.hqgfx;
            let section = self.section;
            let time_ramp = with_boundary(|n| n.imod_info_time_ramp_index(vi, self.time_lock));
            with_boundary(|n| {
                n.b3d_draw_grey_scale_pixels_hq(
                    image_data,
                    im_xsize,
                    im_ysize,
                    x_start,
                    y_start,
                    bx,
                    by,
                    x_draw_size,
                    y_draw_size,
                    -1,
                    rampbase,
                    zoom,
                    zoom,
                    hq,
                    section,
                    rgba,
                    time_ramp,
                )
            });
        } else {
            // For panels, clear whole window then draw each panel if it is at
            // legal Z
            let background = APP.lock().unwrap().as_ref().map_or(0, |a| a.background);
            with_boundary(|n| util_clear_window(n, background));
            for ix in 0..self.num_xpanels {
                for iy in 0..self.num_ypanels {
                    ind = ix + iy * self.num_xpanels;
                    bl = ind - (self.num_xpanels * self.num_ypanels - 1) / 2;
                    iz = self.section + self.panel_zstep * bl;
                    if iz < 0 || iz >= unsafe { (*vi).zsize } {
                        continue;
                    }
                    if !unsafe { (*vi).pyr_cache }.is_null() {
                        let (xs, ys, xd, yd, zm) = (
                            self.xstart,
                            self.ystart,
                            self.xdrawsize,
                            self.ydrawsize,
                            self.zoom,
                        );
                        image_data = with_boundary(|n| {
                            n.pyr_cache_get_section_area(
                                vi,
                                iz,
                                xs,
                                ys,
                                xd,
                                yd,
                                zm,
                                async_load,
                                &mut x_draw_size,
                                &mut y_draw_size,
                                &mut x_offset,
                                &mut y_offset,
                                &mut tile_scale,
                                &mut status,
                            )
                        });
                    } else {
                        image_data = unsafe { ivw_get_z_section_time(vi, iz, time) };
                    }
                    bl = self.xborder
                        + self.panel_xborder
                        + ix * (self.panel_xsize + self.panel_gutter);
                    wh = self.yborder
                        + self.panel_yborder
                        + iy * (self.panel_ysize + self.panel_gutter);
                    let rampbase = unsafe { (*vi).rampbase };
                    let hq = self.hqgfx;
                    let time_ramp =
                        with_boundary(|n| n.imod_info_time_ramp_index(vi, self.time_lock));
                    with_boundary(|n| {
                        n.b3d_draw_grey_scale_pixels_hq(
                            image_data,
                            im_xsize,
                            im_ysize,
                            x_start,
                            y_start,
                            bl,
                            wh,
                            x_draw_size,
                            y_draw_size,
                            ind,
                            rampbase,
                            zoom,
                            zoom,
                            hq,
                            iz,
                            rgba,
                            time_ramp,
                        )
                    });
                }
            }
        }

        /* DNM 9/15/03: Get the X zoom, which might be slightly different */
        // Then get the subarea limits again with more correct zoom value
        self.xzoom = with_boundary(|n| n.b3d_get_cur_x_zoom()) / tile_scale as f32;
        self.time = time;
        if self.record_subarea != 0 {
            self.set_area_limits();
            with_boundary(|n| n.locator_schedule_draw(vi));
            self.record_subarea = 0;
        }
        if overlay != 0 {
            unsafe { libc::free(over_image as *mut libc::c_void) };
        }
    }

    /// `ZapFuncs::fillOverlayRGB` (`xzap.cpp:4960`).
    pub fn fill_overlay_rgb(
        &mut self,
        lines: *mut *mut u8,
        nx: i32,
        ny: i32,
        chan: i32,
        image: *mut u8,
    ) {
        let uslines = lines as *mut *mut u16;
        let mut image = unsafe { image.add(chan as usize) };

        let cvi = with_boundary(|n| n.app_cvi());
        let ushort_store = !cvi.is_null() && unsafe { (*cvi).ushort_store } != 0;
        for j in 0..ny as usize {
            if ushort_store {
                for i in 0..nx as usize {
                    unsafe { *image = (*(*uslines.add(j)).add(i) / 256) as u8 };
                    image = unsafe { image.add(3) };
                }
            } else {
                for i in 0..nx as usize {
                    unsafe { *image = *(*lines.add(j)).add(i) };
                    image = unsafe { image.add(3) };
                }
            }
        }
    }

    /// `ZapFuncs::drawModel` (`xzap.cpp:4982`).
    ///
    /// `scanCont` is a `Icont **` with NULL entries in the source; the
    /// translation keeps the contours in a `Vec<Icont>` with a parallel
    /// `scan_ok` flag standing for the pointer's non-nullness, because
    /// `imodel_contour_scan` returns an owned `Option<Icont>` rather than a
    /// pointer.  `mContsAtCurZ` likewise borrows a row of `contAtZ` in the
    /// source and is copied here, since `imodContourFreeZTables` owns it.
    pub fn draw_model(&mut self) {
        let vi = self.vi;
        let mut max_pts = 0;
        let mut surf = -1;
        let mut obj: *mut Iobj;
        let cont = imod_contour_get(unsafe { (*vi).imod.as_ref() });
        let mut scan_cont: Vec<Icont> = Vec::new();
        let mut scan_ok: Vec<bool> = Vec::new();
        let mut pmin: Vec<Ipoint> = Vec::new();
        let mut pmax: Vec<Ipoint> = Vec::new();
        let mut num_at_z: Vec<i32> = Vec::new();
        let mut cont_at_z: Vec<Vec<i32>> = Vec::new();
        let mut cont_z: Vec<i32> = Vec::new();
        let mut zlist: Vec<i32> = Vec::new();
        let (mut zl_size, mut num_max, mut error, mut zmin, mut zmax) = (0, 0, 0, 0, 0);
        let mut num_conts = 0;
        let mut num_warn = -1;
        let mut free_z_tables;

        if unsafe { (*(*vi).imod).drawmode } <= 0 {
            return;
        }

        self.draw_ghost();

        if let Some(cont) = cont {
            surf = cont.surf;
        }

        let objsize = unsafe { (*(*vi).imod).obj.len() as i32 };
        let num_extra = unsafe { (*vi).num_extra_obj };
        for ob in 0..objsize + num_extra {
            if ob < objsize {
                obj = unsafe { (*(*vi).imod).obj.as_mut_ptr().add(ob as usize) };
            } else {
                obj = ivw_get_an_extra_object(unsafe { &mut *vi }, ob - objsize)
                    .map_or(ptr::null_mut(), |o| o as *mut Iobj);
                if obj.is_null() || unsafe { (*obj).flags } & IMOD_OBJFLAG_EXTRA_EDIT == 0 {
                    continue;
                }
            }
            error = 0;
            free_z_tables = false;
            if iobj_off(unsafe { (*obj).flags }) != 0 {
                continue;
            }
            if ob < objsize {
                with_boundary(|n| n.imod_set_object_color(ob));
            }
            let width = S_SCALE_SIZES.get() * unsafe { (*obj).linewidth2 } as i32;
            with_boundary(|n| n.b3d_line_width(width, obj));
            with_boundary(|n| n.ifg_setup_value_drawing(obj, GEN_STORE_MINMAX1));

            if unsafe { (*obj).extra[IOBJ_EX_FLAGS] } & IOBJ_EXFLAG_MESH_ON_IMG != 0
                && unsafe { !(&(*obj).mesh).is_empty() }
            {
                self.draw_mesh(obj, ob);
            }

            if ob >= objsize {
                continue;
            }

            // Set up for filled contour tesselator drawing
            self.conts_at_cur_z = None;
            if unsafe { (*obj).extra[IOBJ_EX_2D_TRANS] } != 0 {
                error = imod_contour_make_z_tables(
                    unsafe { &mut (&mut (*(*vi).imod).obj)[ob as usize] },
                    1,
                    0,
                    &mut cont_z,
                    &mut zlist,
                    &mut num_at_z,
                    &mut cont_at_z,
                    &mut zmin,
                    &mut zmax,
                    &mut zl_size,
                    &mut num_max,
                );
                if error == 0 {
                    free_z_tables = true;
                }

                // Save the list of contours at this Z: it is the way to get
                // back to object conts
                if error == 0
                    && self.section >= zmin
                    && self.section <= zmax
                    && num_at_z[(self.section - zmin) as usize] > 0
                {
                    self.conts_at_cur_z = Some(cont_at_z[(self.section - zmin) as usize].clone());
                    num_conts = num_at_z[(self.section - zmin) as usize];
                }

                if self.conts_at_cur_z.is_some() && iobj_close(unsafe { (*obj).flags }) != 0 {
                    // Make arrays that are needed for nest analysis here and
                    // map for drawing
                    pmin = vec![Ipoint::default(); num_conts as usize];
                    pmax = vec![Ipoint::default(); num_conts as usize];
                    scan_cont = vec![Icont::default(); num_conts as usize];
                    scan_ok = vec![false; num_conts as usize];
                    self.nest_cont_map = vec![0; unsafe { (&(*obj).cont).len() }];
                    self.nest_ind = vec![0; num_conts as usize];

                    // Make map from contour number to # at this Z that will be
                    // in the analysis and get bounding boxes
                    for co in 0..unsafe { (&(*obj).cont).len() } {
                        self.nest_cont_map[co] = -1;
                    }
                    for zco in 0..num_conts as usize {
                        let co = self.conts_at_cur_z.as_ref().unwrap()[zco];
                        let zt_cont = unsafe { &(&(*obj).cont)[co as usize] };
                        imod_contour_get_bbox(Some(zt_cont), &mut pmin[zco], &mut pmax[zco]);
                        self.nest_cont_map[co as usize] = zco as i32;
                        self.nest_ind[zco] = -1;
                        scan_ok[zco] = false;
                    }
                    // Look for conts that might overlap based on bounding
                    // boxes, make scan conts for them, and check for nesting
                    let mut co = 0;
                    while co < num_conts - 1 && error == 0 {
                        let obco = self.conts_at_cur_z.as_ref().unwrap()[co as usize];
                        let mut zco = co + 1;
                        while zco < num_conts && error == 0 {
                            let obzco = self.conts_at_cur_z.as_ref().unwrap()[zco as usize];
                            if pmax[co as usize].x >= pmin[zco as usize].x
                                && pmax[zco as usize].x >= pmin[co as usize].x
                                && pmax[co as usize].y >= pmin[zco as usize].y
                                && pmax[zco as usize].y >= pmin[co as usize].y
                            {
                                if !scan_ok[co as usize] {
                                    if let Some(sc) = imodel_contour_scan(Some(unsafe {
                                        &(&(*obj).cont)[obco as usize]
                                    })) {
                                        scan_cont[co as usize] = sc;
                                        scan_ok[co as usize] = true;
                                    }
                                }
                                if !scan_ok[zco as usize] {
                                    if let Some(sc) = imodel_contour_scan(Some(unsafe {
                                        &(&(*obj).cont)[obzco as usize]
                                    })) {
                                        scan_cont[zco as usize] = sc;
                                        scan_ok[zco as usize] = true;
                                    }
                                }
                                if !scan_ok[co as usize]
                                    || !scan_ok[zco as usize]
                                    || imod_contour_check_nesting(
                                        co,
                                        zco,
                                        &mut scan_cont,
                                        &pmin,
                                        &pmax,
                                        &mut self.nests,
                                        &mut self.nest_ind,
                                        &mut self.num_nests,
                                        &mut num_warn,
                                    ) != 0
                                {
                                    error = 1;
                                }
                            }
                            zco += 1;
                        }
                        co += 1;
                    }

                    // Analyze the nesting
                    let num_nests = self.num_nests;
                    imod_contour_nest_levels(&mut self.nests, &self.nest_ind, num_nests);
                }

                // Now set up the tesselator contour if needed
                if error == 0 && self.conts_at_cur_z.is_some() {
                    with_boundary(|n| n.setup_filled_cont_tesselator());
                    for co in 0..num_conts as usize {
                        let ind = self.conts_at_cur_z.as_ref().unwrap()[co];
                        max_pts =
                            max_pts.max(unsafe { (&(*obj).cont)[ind as usize].pts.len() as i32 });
                    }
                    if let Some(mut tc) = imod_contour_new() {
                        tc.pts = vec![Ipoint::default(); max_pts as usize];
                        self.tess_max_points = max_pts;
                        self.tess_cont = Box::into_raw(Box::new(tc));
                    }
                }
            }

            // Draw the contours
            for co in 0..unsafe { (&(*obj).cont).len() as i32 } {
                if ob == unsafe { (*(*vi).imod).cindex.object } {
                    if co == unsafe { (*(*vi).imod).cindex.contour } {
                        self.draw_contour(co, ob);
                        continue;
                    }
                    if unsafe { (*vi).ghostmode } & IMOD_GHOST_SURFACE != 0
                        && surf >= 0
                        && surf != unsafe { (&(*obj).cont)[co as usize].surf }
                    {
                        let ghost = APP.lock().unwrap().as_ref().map_or(0, |a| a.ghost);
                        with_boundary(|n| n.b3d_color_index(ghost));
                        self.draw_contour(co, ob);
                        with_boundary(|n| n.imod_set_object_color(ob));
                        continue;
                    }
                }

                self.draw_contour(co, ob);
            }

            // Clean up the tesselator contour
            if !self.tess_cont.is_null() {
                unsafe { drop(Box::from_raw(self.tess_cont)) };
                self.tess_cont = ptr::null_mut();
                self.tess_max_points = 0;
            }
            if self.label_painter {
                with_boundary(|n| n.delete_label_font());
                self.label_painter = false;
            }
            self.label_font = false;

            // Clean up all the nesting analysis arrays
            if !scan_cont.is_empty() {
                scan_cont.clear();
                scan_ok.clear();
                let num_nests = self.num_nests;
                imod_contour_free_nests(&mut self.nests, num_nests);
            }
            if free_z_tables {
                imod_contour_free_z_tables(
                    &mut num_at_z,
                    &mut cont_at_z,
                    &mut cont_z,
                    &mut zlist,
                    zmin,
                    zmax,
                );
            }
            self.nest_ind = Vec::new();
            self.nest_cont_map = Vec::new();
            self.num_nests = 0;
            pmin = Vec::new();
            pmax = Vec::new();
        }
    }

    /// `ZapFuncs::drawMesh` (`xzap.cpp:5173`); draw mesh lines located on the
    /// current image.
    pub fn draw_mesh(&mut self, obj: *mut Iobj, ob: i32) {
        let mut def_props = DrawProps::default();
        let mut cur_props = DrawProps::default();
        let mut resol = 0;
        let mut start_of_poly;
        let (mut last_red, mut last_green, mut last_blue);
        let (mut first_red, mut first_green, mut first_blue);
        let (mut first_vis_r, mut first_vis_g, mut first_vis_b) = (0., 0., 0.);
        let (mut second_vis_r, mut second_vis_g, mut second_vis_b) = (0., 0., 0.);
        let (mut first_vis_ind, mut second_vis_ind);
        let (mut first_li, mut second_li) = (0usize, 0usize);

        let lowres = with_boundary(|n| n.imodv_lowres());
        imod_mesh_nearest_res(
            unsafe { &(&(*obj).mesh) },
            unsafe { (&(*obj).mesh).len() as i32 },
            lowres,
            &mut resol,
        );
        if with_boundary(|n| n.util_manage_paired_meshes(obj, ob)) != 0 {
            return;
        }

        for me in 0..unsafe { (&(*obj).mesh).len() } {
            let mesh = unsafe { (&mut (*obj).mesh).as_mut_ptr().add(me) };
            if crate::imod::libimod::imesh::imesh_resol(unsafe { (*mesh).flag }) != resol {
                continue;
            }
            if crate::imod::libimod::imesh::imesh_thickness(unsafe { (*mesh).flag })
                != unsafe { (*obj).mesh_thickness } as i32
            {
                continue;
            }

            let vert = unsafe { (&(*mesh).vert).as_ptr() };
            let mlist = unsafe { (&(*mesh).list).as_ptr() };

            // Initialize state and record initial color and set it
            let mut state_flags = 0;
            let mut change_flags = 0;
            let surf = unsafe { (*mesh).surf } as i32;
            with_boundary(|n| {
                n.ifg_handle_surf_change(
                    obj,
                    surf,
                    &mut def_props,
                    &mut cur_props,
                    &mut state_flags,
                    0,
                )
            });
            let store_ptr = unsafe { &raw const (*mesh).store };
            let mut next_item_index = with_boundary(|n| n.istore_first_change_index(store_ptr));
            let mut next_change = next_item_index;
            first_red = cur_props.red;
            last_red = first_red;
            first_green = cur_props.green;
            last_green = first_green;
            first_blue = cur_props.blue;
            last_blue = first_blue;
            with_boundary(|n| n.gl_color_3f(first_red, first_green, first_blue));

            let lsize = unsafe { (&(*mesh).list).len() as i32 };
            let mut i = 0;
            while i < lsize {
                match unsafe { *mlist.add(i as usize) } {
                    IMOD_MESH_BGNPOLY | IMOD_MESH_BGNBIGPOLY | IMOD_MESH_BGNPOLYNORM => {
                        while unsafe { *mlist.add(i as usize) } != IMOD_MESH_ENDPOLY {
                            i += 1;
                        }
                    }

                    IMOD_MESH_BGNPOLYNORM2 => {
                        i += 1;

                        // If no changes in whole mesh, process it simply.  Go
                        // through each triangle and see if two points are
                        // visible, if so draw line
                        if next_item_index < 0 {
                            while unsafe { *mlist.add(i as usize) } != IMOD_MESH_ENDPOLY {
                                first_vis_ind = -1;
                                second_vis_ind = -1;
                                for _j in 0..3 {
                                    let vi_ind = unsafe { *mlist.add(i as usize) };
                                    if self.point_visable(unsafe { &*vert.add(vi_ind as usize) })
                                        != 0
                                    {
                                        if first_vis_ind < 0 {
                                            first_vis_ind = vi_ind;
                                        } else {
                                            second_vis_ind = vi_ind;
                                        }
                                    }
                                    i += 1;
                                }
                                if second_vis_ind >= 0 {
                                    let (x1, y1, x2, y2) = (
                                        self.xpos(unsafe { (*vert.add(first_vis_ind as usize)).x }),
                                        self.ypos(unsafe { (*vert.add(first_vis_ind as usize)).y }),
                                        self.xpos(unsafe {
                                            (*vert.add(second_vis_ind as usize)).x
                                        }),
                                        self.ypos(unsafe {
                                            (*vert.add(second_vis_ind as usize)).y
                                        }),
                                    );
                                    with_boundary(|n| n.gl_begin_lines());
                                    with_boundary(|n| n.gl_vertex_2i(x1, y1));
                                    with_boundary(|n| n.gl_vertex_2i(x2, y2));
                                    with_boundary(|n| n.gl_end());
                                }
                            }
                        } else {
                            // Whole other loop when there is a store:
                            // Set the color to starting color at beginning of
                            // each triangle
                            start_of_poly = true;
                            while unsafe { *mlist.add(i as usize) } != IMOD_MESH_ENDPOLY {
                                cur_props.red = first_red;
                                cur_props.green = first_green;
                                cur_props.blue = first_blue;
                                first_vis_ind = -1;
                                second_vis_ind = -1;
                                for _j in 0..3 {
                                    // Get a change at appointed time or at
                                    // start of polygon
                                    if next_change == i || start_of_poly {
                                        let idx = i;
                                        next_change = with_boundary(|n| {
                                            n.ifg_handle_mesh_change(
                                                obj,
                                                store_ptr,
                                                &mut def_props,
                                                &mut cur_props,
                                                &mut next_item_index,
                                                idx,
                                                &mut state_flags,
                                                &mut change_flags,
                                                0,
                                            )
                                        });
                                        start_of_poly = false;
                                    }

                                    // When a visible point is found, record
                                    // its color too
                                    let vi_ind = unsafe { *mlist.add(i as usize) };
                                    if self.point_visable(unsafe { &*vert.add(vi_ind as usize) })
                                        != 0
                                    {
                                        if first_vis_ind < 0 {
                                            first_li = i as usize;
                                            first_vis_ind = vi_ind;
                                            first_vis_r = cur_props.red;
                                            first_vis_g = cur_props.green;
                                            first_vis_b = cur_props.blue;
                                        } else {
                                            second_li = i as usize;
                                            second_vis_ind = vi_ind;
                                            second_vis_r = cur_props.red;
                                            second_vis_g = cur_props.green;
                                            second_vis_b = cur_props.blue;
                                        }
                                    }
                                    i += 1;
                                }

                                // When two points found, set color as needed
                                // before each one
                                if second_vis_ind >= 0 {
                                    if last_red != first_vis_r
                                        || last_green != first_vis_g
                                        || last_blue != first_vis_b
                                    {
                                        with_boundary(|n| {
                                            n.gl_color_3f(first_vis_r, first_vis_g, first_vis_b)
                                        });
                                        last_red = first_vis_r;
                                        last_green = first_vis_g;
                                        last_blue = first_vis_b;
                                    }
                                    with_boundary(|n| n.gl_begin_lines());
                                    let (x1, y1) = (
                                        self.xpos(unsafe { (*vert.add(first_vis_ind as usize)).x }),
                                        self.ypos(unsafe { (*vert.add(first_vis_ind as usize)).y }),
                                    );
                                    with_boundary(|n| n.gl_vertex_2i(x1, y1));

                                    if last_red != second_vis_r
                                        || last_green != second_vis_g
                                        || last_blue != second_vis_b
                                    {
                                        with_boundary(|n| {
                                            n.gl_color_3f(second_vis_r, second_vis_g, second_vis_b)
                                        });
                                        last_red = second_vis_r;
                                        last_green = second_vis_g;
                                        last_blue = second_vis_b;
                                    }
                                    let (x2, y2) = (
                                        self.xpos(unsafe {
                                            (*vert.add(second_vis_ind as usize)).x
                                        }),
                                        self.ypos(unsafe {
                                            (*vert.add(second_vis_ind as usize)).y
                                        }),
                                    );
                                    with_boundary(|n| n.gl_vertex_2i(x2, y2));
                                    with_boundary(|n| n.gl_end());
                                }
                            }
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            let _ = (first_li, second_li);
        }
    }

    /// `ZapFuncs::drawExtraObject` (`xzap.cpp:5328`); a separate routine to
    /// draw the extra object(s) so that model - current point - extra object
    /// drawing could happen in the right order.
    pub fn draw_extra_object(&mut self) {
        let vi = self.vi;
        let mut stipple_save = 0;
        let cont_is_some = imod_contour_get(unsafe { (*vi).imod.as_ref() }).is_some();

        if unsafe { (*(*vi).imod).drawmode } <= 0 {
            return;
        }
        for ob in 0..unsafe { (*vi).num_extra_obj } {
            let xobj = ivw_get_an_extra_object(unsafe { &mut *vi }, ob)
                .map_or(ptr::null_mut(), |o| o as *mut Iobj);
            if xobj.is_null() || unsafe { (&(*xobj).cont).is_empty() } {
                continue;
            }
            if iobj_off(unsafe { (*xobj).flags }) != 0
                || unsafe { (*xobj).flags } & IMOD_OBJFLAG_MODV_ONLY != 0
                || unsafe { (*xobj).extra[IOBJ_EX_FLAGS] } & IOBJ_EXFLAG_SLICER_ONLY != 0
            {
                continue;
            }
            let lasso_ctrl = unsafe { (*xobj).extra[IOBJ_EX_LASSO_ID] } as i32;
            if lasso_ctrl != 0 && lasso_ctrl != self.ctrl {
                continue;
            }
            if lasso_ctrl != 0 {
                stipple_save = unsafe { (*self.vi).draw_stipple };
                unsafe { (*self.vi).draw_stipple = 1 };
                if self.shifting_cont != 0 && !cont_is_some && !self.drawing_lasso {
                    unsafe { (*xobj).linewidth2 = 2 };
                }
            }

            with_boundary(|n| n.ifg_reset_value_setup());

            // If there are contours in the extra object, set color and draw
            let (r, g, b) = (
                (255. * unsafe { (*xobj).red }) as i32,
                (255. * unsafe { (*xobj).green }) as i32,
                (255. * unsafe { (*xobj).blue }) as i32,
            );
            with_boundary(|n| n.custom_ghost_color(r, g, b));
            for co in 0..unsafe { (&(*xobj).cont).len() as i32 } {
                self.draw_contour(co, -1 - ob);
            }
            if lasso_ctrl != 0 {
                unsafe { (*self.vi).draw_stipple = stipple_save };
                unsafe { (*xobj).linewidth2 = 1 };
            }
        }
        with_boundary(|n| n.reset_ghost_color());
    }

    /// `ZapFuncs::drawContour` (`xzap.cpp:5372`); draw a contour, including
    /// contours in an extra object.
    pub fn draw_contour(&mut self, co: i32, ob: i32) {
        let vi = self.vi;
        let mut delz;
        let obj: *mut Iobj;
        let mut use_cont: *mut Icont;
        let mut in_cont: *mut Icont;
        let mut cont_props = DrawProps::default();
        let mut pt_props = DrawProps::default();
        let mut radius;
        let (mut last_x, mut last_y, mut this_x, mut this_y);
        let mut label_yoffset = 0;
        let (mut xlab, mut ylab);
        let mut nest;
        let (mut pt1, mut pt2) = (0usize, 0usize);
        let (mut dist, mut tdist);
        let mut drawsize;
        let mut next_change;
        let mut state_flags = 0;
        let mut change_flags = 0;
        let mut check_symbol = 0;
        let mut handle_flags = HANDLE_LINE_COLOR | HANDLE_2DWIDTH;
        let (mut last_visible, mut this_visible);
        let mut skip_outline = false;
        let current_cont = co == unsafe { (*(*vi).imod).cindex.contour }
            && ob == unsafe { (*(*vi).imod).cindex.object };

        if ob >= 0 {
            obj = unsafe { (*(*vi).imod).obj.as_mut_ptr().add(ob as usize) };
        } else {
            obj = ivw_get_an_extra_object(unsafe { &mut *vi }, -ob - 1)
                .map_or(ptr::null_mut(), |o| o as *mut Iobj);
        }

        if obj.is_null() {
            return;
        }
        let cont = unsafe { (&mut (*obj).cont).as_mut_ptr().add(co as usize) };
        if unsafe { (&(*cont).pts).is_empty() } {
            return;
        }

        let draw_all_z = unsafe { (*cont).flags } & (ICONT_CURSOR_LIKE | ICONT_DRAW_ALLZ) != 0;
        let draw_pnt_off_sec = unsafe { (*obj).flags } & IMOD_OBJFLAG_PNT_ON_SEC == 0;
        if unsafe { (*cont).flags } & ICONT_MMODEL_ONLY != 0
            && unsafe { (*(*vi).imod).mousemode } != IMOD_MMODEL
        {
            return;
        }

        if unsafe { (*cont).flags } & ICONT_CURSOR_LIKE != 0 {
            if !with_boundary(|n| n.gfx_extra_cursor_in_window()) || self.num_xpanels != 0 {
                return;
            }
            self.drew_extra_cursor = true;
        }

        if with_boundary(|n| n.ifg_get_value_setup_state()) != 0 {
            handle_flags |= HANDLE_VALUE1;
        }

        let zscale = ((if unsafe { (*(*vi).imod).zscale } != 0. {
            unsafe { (*(*vi).imod).zscale }
        } else {
            1.
        }) * unsafe { (*vi).zbin } as f32)
            / unsafe { (*vi).xybin } as f32;

        /* check for contours that contain time data. */
        /* Don't draw them if the time isn't right. */
        /* DNM 6/7/01: but draw contours with time 0 regardless of time */
        if unsafe { ivw_time_mismatch(vi, self.time_lock, obj, cont) } {
            return;
        }

        // get draw properties
        let selected = i32::from(with_boundary(|n| n.imod_selection_list_query(vi, ob, co)) > -2);
        let scale = S_SCALE_SIZES.get();
        next_change = with_boundary(|n| {
            n.ifg_handle_cont_change(
                obj,
                co,
                &mut cont_props,
                &mut pt_props,
                &mut state_flags,
                handle_flags,
                selected,
                scale,
            )
        });
        if cont_props.gap != 0 {
            return;
        }

        let draw_stipple = unsafe { (*vi).draw_stipple };
        let stipple_gaps =
            !with_boundary(|n| util_enable_stipple(n, draw_stipple, unsafe { &*cont }))
                && with_boundary(|n| n.ifg_stipple_gaps());

        /* Open or closed contour */
        // Skip if not wild and not on section
        last_visible = self.point_visable(unsafe { &(&(*cont).pts)[0] }) != 0 || draw_all_z;
        if iobj_scat(unsafe { (*obj).flags }) == 0
            && (unsafe { (*cont).flags } & ICONT_WILD != 0 || last_visible)
        {
            // Draw fill first if there is a 2d trans setting
            // First check for nesting if contour is on this Z level at all
            if unsafe { (*obj).extra[IOBJ_EX_2D_TRANS] } != 0
                && !self.tess_cont.is_null()
                && unsafe { (*cont).flags } & ICONT_WILD == 0
            {
                use_cont = cont;
                let mut use_cont_owned: Option<Box<Icont>> = None;
                if iobj_close(unsafe { (*obj).flags }) != 0
                    && self.nest_cont_map[co as usize] >= 0
                    && self.nest_ind[self.nest_cont_map[co as usize] as usize] >= 0
                {
                    nest = self.nest_ind[self.nest_cont_map[co as usize] as usize];

                    // If there is a nest, skip everything on even levels and
                    // suppress outline drawing for them as it becomes part of
                    // the fill
                    let level = self.nests[nest as usize].level;
                    if level % 2 == 0 {
                        use_cont = ptr::null_mut();
                        skip_outline = selected == 0
                            && unsafe { (*obj).flags } & IMOD_OBJFLAG_POLY_CONT == 0
                            && !current_cont;
                    } else {
                        // Loop on the inside ones and join ones at next level
                        // if they are truly inside the current useCont
                        for ind in 0..self.nests[nest as usize].ninside as usize {
                            let zco = self.nests[nest as usize].inside[ind];
                            if self.nests[self.nest_ind[zco as usize] as usize].level == level + 1 {
                                let inco = self.conts_at_cur_z.as_ref().unwrap()
                                    [self.nests[nest as usize].inside[ind] as usize];
                                in_cont =
                                    unsafe { (&mut (*obj).cont).as_mut_ptr().add(inco as usize) };
                                if imod_contour_inside_cont(unsafe { &*in_cont }, unsafe {
                                    &*use_cont
                                }) != 0
                                {
                                    // Find nearest points
                                    dist = 1.0e30f32;
                                    for i in 0..unsafe { (&(*use_cont).pts).len() } {
                                        for j in 0..unsafe { (&(*in_cont).pts).len() } {
                                            tdist = imod_point_distance(
                                                unsafe { &(&(*use_cont).pts)[i] },
                                                unsafe { &(&(*in_cont).pts)[j] },
                                            );
                                            if dist > tdist {
                                                dist = tdist;
                                                pt1 = i;
                                                pt2 = j;
                                            }
                                        }
                                    }

                                    // Join the contours, duplicating first
                                    // because they can get inverted in the
                                    // join which makes an unintended change to
                                    // the model
                                    let tmp_cont = imod_contour_dup(unsafe { &*in_cont })
                                        .unwrap_or_else(|| unsafe { (*in_cont).clone() });
                                    let mut tmp_cont = tmp_cont;
                                    let mut cur_copy = imod_contour_dup(unsafe { &*use_cont })
                                        .unwrap_or_else(|| unsafe { (*use_cont).clone() });
                                    if let Some(join_cont) = imod_contour_join(
                                        Some(&mut cur_copy),
                                        Some(&mut tmp_cont),
                                        pt1 as i32,
                                        pt2 as i32,
                                        2,
                                        1,
                                    ) {
                                        use_cont_owned = Some(Box::new(join_cont));
                                        use_cont = use_cont_owned.as_mut().unwrap().as_mut();
                                    }
                                }
                            }
                        }

                        // Reallocate the tess contour if necessary
                        if !use_cont.is_null()
                            && unsafe { (&(*use_cont).pts).len() as i32 } > self.tess_max_points
                        {
                            let need = unsafe { (&(*use_cont).pts).len() };
                            unsafe { (*self.tess_cont).pts = vec![Ipoint::default(); need] };
                            self.tess_max_points = need as i32;
                        }
                    }
                }

                // If there is a contour after all that, put it in the tess
                // cont and draw
                if !use_cont.is_null() {
                    with_boundary(|n| n.gl_enable_blend(true));
                    with_boundary(|n| n.gl_blend_func(true));
                    let alpha = 1. - unsafe { (*obj).extra[IOBJ_EX_2D_TRANS] } as f32 / 100.;
                    let (r, g, b) = (cont_props.red, cont_props.green, cont_props.blue);
                    with_boundary(|n| n.gl_color_4f(r, g, b, alpha));
                    for pt in 0..unsafe { (&(*use_cont).pts).len() } {
                        unsafe {
                            (&mut (*self.tess_cont).pts)[pt].x =
                                self.xpos((&(*use_cont).pts)[pt].x) as f32
                        };
                        unsafe {
                            (&mut (*self.tess_cont).pts)[pt].y =
                                self.ypos((&(*use_cont).pts)[pt].y) as f32
                        };
                    }
                    unsafe { (&mut (*self.tess_cont).pts).truncate((&(*use_cont).pts).len()) };
                    let pts = unsafe { (&(*self.tess_cont).pts).clone() };
                    with_boundary(|n| n.draw_filled_polygon(&pts));
                    with_boundary(|n| n.gl_enable_blend(false));
                    with_boundary(|n| n.gl_color_3f(r, g, b));
                    skip_outline = selected == 0
                        && unsafe { (*obj).flags } & IMOD_OBJFLAG_POLY_CONT == 0
                        && !current_cont;
                }
            }

            if !skip_outline && (unsafe { (*cont).flags } & ICONT_WILD != 0 || next_change >= 0) {
                if next_change == 0 {
                    let store = unsafe { &raw const (*cont).store };
                    next_change = with_boundary(|n| {
                        n.ifg_handle_next_change(
                            obj,
                            store,
                            &mut cont_props,
                            &mut pt_props,
                            &mut state_flags,
                            &mut change_flags,
                            handle_flags,
                            selected,
                            scale,
                        )
                    });
                }

                if stipple_gaps {
                    with_boundary(|n| n.gl_line_stipple(1, 0x0707));
                }
                if pt_props.gap != 0 && stipple_gaps {
                    with_boundary(|n| n.b3d_stipple_next_line(true));
                    pt_props.gap = 0;
                }

                // For wild contour, test every point and connect only pairs on
                // section
                last_x = self.xpos(unsafe { (&(*cont).pts)[0].x });
                last_y = self.ypos(unsafe { (&(*cont).pts)[0].y });
                for pt in 1..unsafe { (&(*cont).pts).len() as i32 } {
                    this_visible = self.point_visable(unsafe { &(&(*cont).pts)[pt as usize] }) != 0;
                    if this_visible {
                        this_x = self.xpos(unsafe { (&(*cont).pts)[pt as usize].x });
                        this_y = self.ypos(unsafe { (&(*cont).pts)[pt as usize].y });
                        if last_visible && pt_props.gap == 0 {
                            let (a, b, c, d) = (last_x, last_y, this_x, this_y);
                            with_boundary(|n| n.b3d_draw_line(a, b, c, d));
                        }
                        last_x = this_x;
                        last_y = this_y;
                    }
                    last_visible = this_visible;
                    pt_props.gap = 0;
                    if pt == next_change {
                        let store = unsafe { &raw const (*cont).store };
                        next_change = with_boundary(|n| {
                            n.ifg_handle_next_change(
                                obj,
                                store,
                                &mut cont_props,
                                &mut pt_props,
                                &mut state_flags,
                                &mut change_flags,
                                handle_flags,
                                selected,
                                scale,
                            )
                        });
                    }
                    if pt_props.gap != 0 && stipple_gaps {
                        with_boundary(|n| n.b3d_stipple_next_line(true));
                        pt_props.gap = 0;
                    }
                }

                // IF closed contour in closed object and not current, draw
                // closure as long as both points are visible
                if iobj_close(unsafe { (*obj).flags }) != 0
                    && unsafe { (*cont).flags } & ICONT_OPEN == 0
                    && pt_props.gap == 0
                    && !current_cont
                    && last_visible
                    && self.point_visable(unsafe { &(&(*cont).pts)[0] }) != 0
                {
                    let (a, b, c, d) = (
                        last_x,
                        last_y,
                        self.xpos(unsafe { (&(*cont).pts)[0].x }),
                        self.ypos(unsafe { (&(*cont).pts)[0].y }),
                    );
                    with_boundary(|n| n.b3d_draw_line(a, b, c, d));
                }

                if stipple_gaps {
                    with_boundary(|n| n.b3d_stipple_next_line(false));
                }
            } else if !skip_outline {
                // For non-wild contour with no changes, draw all points
                // without testing
                with_boundary(|n| n.b3d_begin_line());
                for pt in 0..unsafe { (&(*cont).pts).len() } {
                    let (x, y) = (
                        self.xpos(unsafe { (&(*cont).pts)[pt].x }),
                        self.ypos(unsafe { (&(*cont).pts)[pt].y }),
                    );
                    with_boundary(|n| n.b3d_vertex_2i(x, y));
                }

                // IF closed contour in closed object and not current, draw
                // closure
                if iobj_close(unsafe { (*obj).flags }) != 0
                    && unsafe { (*cont).flags } & ICONT_OPEN == 0
                    && !current_cont
                {
                    let (x, y) = (
                        self.xpos(unsafe { (&(*cont).pts)[0].x }),
                        self.ypos(unsafe { (&(*cont).pts)[0].y }),
                    );
                    with_boundary(|n| n.b3d_vertex_2i(x, y));
                }

                with_boundary(|n| n.b3d_end_line());
            }

            check_symbol = 1;
        }

        /* symbols */
        if unsafe { !(&(*cont).store).is_empty() } {
            next_change = with_boundary(|n| {
                n.ifg_handle_cont_change(
                    obj,
                    co,
                    &mut cont_props,
                    &mut pt_props,
                    &mut state_flags,
                    handle_flags,
                    selected,
                    scale,
                )
            });
        }
        if (iobj_scat(unsafe { (*obj).flags }) != 0 || check_symbol != 0)
            && (cont_props.symtype != IOBJ_SYM_NONE || next_change >= 0)
        {
            for pt in 0..unsafe { (&(*cont).pts).len() as i32 } {
                pt_props.gap = 0;
                if pt == next_change {
                    let store = unsafe { &raw const (*cont).store };
                    next_change = with_boundary(|n| {
                        n.ifg_handle_next_change(
                            obj,
                            store,
                            &mut cont_props,
                            &mut pt_props,
                            &mut state_flags,
                            &mut change_flags,
                            handle_flags,
                            selected,
                            scale,
                        )
                    });
                }

                if pt_props.symtype != IOBJ_SYM_NONE
                    && !(pt_props.gap != 0 && pt_props.valskip != 0)
                    && self.point_visable(unsafe { &(&(*cont).pts)[pt as usize] }) != 0
                {
                    let (x, y) = (
                        self.xpos(unsafe { (&(*cont).pts)[pt as usize].x }),
                        self.ypos(unsafe { (&(*cont).pts)[pt as usize].y }),
                    );
                    let (sym, size, flags) = (
                        pt_props.symtype,
                        pt_props.symsize * scale,
                        pt_props.symflags as u32,
                    );
                    with_boundary(|n| util_draw_symbol(n, x, y, sym, size, flags));
                }
            }
        }

        /* Any contour with point sizes set */
        if iobj_scat(unsafe { (*obj).flags }) != 0
            || unsafe { !(&(*cont).sizes).is_empty() }
            || unsafe { (*obj).pdrawsize } != 0
        {
            if unsafe { !(&(*cont).store).is_empty() } {
                next_change = with_boundary(|n| {
                    n.ifg_handle_cont_change(
                        obj,
                        co,
                        &mut cont_props,
                        &mut pt_props,
                        &mut state_flags,
                        handle_flags,
                        selected,
                        scale,
                    )
                });
            }
            for pt in 0..unsafe { (&(*cont).pts).len() as i32 } {
                pt_props.gap = 0;
                if pt == next_change {
                    let store = unsafe { &raw const (*cont).store };
                    next_change = with_boundary(|n| {
                        n.ifg_handle_next_change(
                            obj,
                            store,
                            &mut cont_props,
                            &mut pt_props,
                            &mut state_flags,
                            &mut change_flags,
                            handle_flags,
                            selected,
                            scale,
                        )
                    });
                }

                drawsize = crate::imod::three_dmod::model_draw::imod_point_get_size(
                    unsafe { &*obj },
                    unsafe { &*cont },
                    pt as usize,
                ) / unsafe { (*vi).xybin } as f32;
                if drawsize > 0. && !(pt_props.gap != 0 && pt_props.valskip != 0) {
                    if self.point_visable(unsafe { &(&(*cont).pts)[pt as usize] }) != 0 {
                        /* DNM: make the product cast to int, not drawsize */
                        let (x, y) = (
                            self.xpos(unsafe { (&(*cont).pts)[pt as usize].x }),
                            self.ypos(unsafe { (&(*cont).pts)[pt as usize].y }),
                        );
                        let r = (drawsize * self.zoom) as i32;
                        with_boundary(|n| n.b3d_draw_circle(x, y, r, false));
                        if drawsize > 3. && draw_pnt_off_sec {
                            with_boundary(|n| n.b3d_draw_plus(x, y, 3 * scale));
                        }
                    } else if drawsize > 1. && draw_pnt_off_sec {
                        /* DNM: fixed this at last, but let size round down so
                        circles get smaller */
                        /* draw a smaller circ if further away. */
                        delz = (unsafe { (&(*cont).pts)[pt as usize].z } - self.section as f32)
                            * zscale;
                        if delz < 0. {
                            delz = -delz;
                        }

                        if delz < drawsize - 0.01 {
                            radius = (((drawsize * drawsize - delz * delz) as f64).sqrt()
                                * self.zoom as f64) as i32;
                            let (x, y) = (
                                self.xpos(unsafe { (&(*cont).pts)[pt as usize].x }),
                                self.ypos(unsafe { (&(*cont).pts)[pt as usize].y }),
                            );
                            with_boundary(|n| n.b3d_draw_circle(x, y, radius, false));
                        }
                    }
                }
            }
        }

        with_boundary(|n| util_disable_stipple(n, draw_stipple, unsafe { &*cont }));

        // Draw labels if any
        if unsafe { (*cont).label.is_some() }
            && unsafe { (*obj).flags } & IMOD_OBJFLAG_DRAW_LABEL != 0
        {
            label_yoffset = ((with_boundary(|n| n.zap_window_zoom_edit_font_height()) / 3) as f64
                * self.device_pixel_ratio as f64
                + 0.5)
                .floor() as i32;
            if !self.label_font {
                let point_size = if unsafe { (*obj).extra[IOBJ_EX_LABEL_SIZE] } != 0 {
                    (scale * unsafe { (*obj).extra[IOBJ_EX_LABEL_SIZE] } as i32) as f32
                } else {
                    0.
                };
                with_boundary(|n| n.make_label_font(point_size));
                self.label_font = true;
                let new_qt_open_gl = APP
                    .lock()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|a| a.new_qt_open_gl != 0);
                if new_qt_open_gl {
                    self.label_painter = true;
                }
            }

            if unsafe { !(&(*cont).store).is_empty() } {
                next_change = with_boundary(|n| {
                    n.ifg_handle_cont_change(
                        obj,
                        co,
                        &mut cont_props,
                        &mut pt_props,
                        &mut state_flags,
                        handle_flags,
                        selected,
                        scale,
                    )
                });
            }
            for pt in 0..unsafe { (&(*cont).pts).len() as i32 } {
                pt_props.gap = 0;
                if pt == next_change {
                    let store = unsafe { &raw const (*cont).store };
                    next_change = with_boundary(|n| {
                        n.ifg_handle_next_change(
                            obj,
                            store,
                            &mut cont_props,
                            &mut pt_props,
                            &mut state_flags,
                            &mut change_flags,
                            handle_flags,
                            selected,
                            scale,
                        )
                    });
                }

                if !(pt_props.gap != 0 && pt_props.valskip != 0)
                    && self.point_visable(unsafe { &(&(*cont).pts)[pt as usize] }) != 0
                {
                    let pt_label = crate::imod::libimod::ilabel::imod_label_item_get(
                        unsafe { (*cont).label.as_ref() },
                        pt,
                    );
                    if let Some(pt_label) = pt_label {
                        drawsize = ((self.zoom
                            * crate::imod::three_dmod::model_draw::imod_point_get_size(
                                unsafe { &*obj },
                                unsafe { &*cont },
                                pt as usize,
                            )
                            / unsafe { (*vi).xybin } as f32)
                            as f64
                            + 0.5)
                            .floor() as f32;
                        drawsize = (((if (pt_props.symsize as f32) < drawsize {
                            drawsize
                        } else {
                            pt_props.symsize as f32
                        }) + 3.) as f64
                            * self.device_pixel_ratio as f64
                            + 0.5)
                            .floor() as f32;
                        xlab = ((self.xpos(unsafe { (&(*cont).pts)[pt as usize].x }) as f32
                            + drawsize) as f64
                            / self.device_pixel_ratio as f64
                            + 0.5)
                            .floor() as i32;
                        ylab = ((self.winy
                            - (self.ypos(unsafe { (&(*cont).pts)[pt as usize].y }) - label_yoffset))
                            as f64
                            / self.device_pixel_ratio as f64
                            + 0.5)
                            .floor() as i32;
                        let pt_label = String::from_utf8_lossy(pt_label).into_owned();
                        with_boundary(|n| n.gfx_render_text(xlab, ylab, &pt_label));
                    }
                }
            }
        }

        // Draw end markers with assigned colors or arrowhead with object color
        if unsafe { (*obj).symflags } & (IOBJ_SYMF_ENDS | IOBJ_SYMF_ARROW) != 0 {
            let psize = unsafe { (&(*cont).pts).len() };
            if psize > 1 && self.point_visable(unsafe { &(&(*cont).pts)[psize - 1] }) != 0 {
                if unsafe { (*obj).symflags } & IOBJ_SYMF_ARROW != 0 {
                    let (tx, ty, hx, hy) = (
                        self.xpos(unsafe { (&(*cont).pts)[psize - 2].x }),
                        self.ypos(unsafe { (&(*cont).pts)[psize - 2].y }),
                        self.xpos(unsafe { (&(*cont).pts)[psize - 1].x }),
                        self.ypos(unsafe { (&(*cont).pts)[psize - 1].y }),
                    );
                    let (tip, thick) =
                        (
                            scale * unsafe { (*obj).symsize } as i32,
                            unsafe { (*obj).linewidth2 } as i32,
                        );
                    with_boundary(|n| n.b3d_draw_arrow(tx, ty, hx, hy, tip, thick, false));
                } else if ob >= 0 {
                    let endpoint = APP.lock().unwrap().as_ref().map_or(0, |a| a.endpoint);
                    with_boundary(|n| n.b3d_color_index(endpoint));
                    let (x, y) = (
                        self.xpos(unsafe { (&(*cont).pts)[psize - 1].x }),
                        self.ypos(unsafe { (&(*cont).pts)[psize - 1].y }),
                    );
                    let s = scale * unsafe { (*obj).symsize } as i32 / 2;
                    with_boundary(|n| n.b3d_draw_cross(x, y, s));
                }
            }
            if ob >= 0
                && unsafe { (*obj).symflags } & IOBJ_SYMF_ENDS != 0
                && self.point_visable(unsafe { &(&(*cont).pts)[0] }) != 0
            {
                let bgnpoint = APP.lock().unwrap().as_ref().map_or(0, |a| a.bgnpoint);
                with_boundary(|n| n.b3d_color_index(bgnpoint));
                let (x, y) = (
                    self.xpos(unsafe { (&(*cont).pts)[0].x }),
                    self.ypos(unsafe { (&(*cont).pts)[0].y }),
                );
                let s = scale * unsafe { (*obj).symsize } as i32 / 2;
                with_boundary(|n| n.b3d_draw_cross(x, y, s));
            }
            with_boundary(|n| n.imod_set_object_color(ob));
        }

        // Draw connectors
        if with_boundary(|n| n.ifg_show_connections())
            && istore_count_items(unsafe { &(&(*cont).store) }, GEN_STORE_CONNECT, 0) != 0
        {
            let foreground = APP.lock().unwrap().as_ref().map_or(0, |a| a.foreground);
            with_boundary(|n| n.b3d_color_index(foreground));
            for st in 0..unsafe { (&(*cont).store).len() } {
                let stp = unsafe { &(&(*cont).store)[st] };
                if stp.type_ == GEN_STORE_CONNECT {
                    let pt = unsafe { stp.index.i };
                    if pt >= 0
                        && pt < unsafe { (&(*cont).pts).len() as i32 }
                        && self.point_visable(unsafe { &(&(*cont).pts)[pt as usize] }) != 0
                    {
                        let value = unsafe { stp.value.i };
                        let (x, y) = (
                            self.xpos(unsafe { (&(*cont).pts)[pt as usize].x }),
                            self.ypos(unsafe { (&(*cont).pts)[pt as usize].y }),
                        );
                        if value % 3 == 1 {
                            with_boundary(|n| n.b3d_draw_square(x, y, value + 5));
                        } else if value % 3 == 2 {
                            with_boundary(|n| n.b3d_draw_triangle(x, y, value + 3));
                        } else {
                            with_boundary(|n| n.b3d_draw_circle(x, y, value + 2, true));
                        }
                    }
                }
            }
            with_boundary(|n| n.imod_set_object_color(ob));
        }

        /* Removed drawing of size 3 circles at ends of current open contour if
        first two points visible or last point visible and next to last is
        not */

        if selected != 0 {
            let width = scale * unsafe { (*obj).linewidth2 } as i32;
            with_boundary(|n| n.b3d_line_width(width, obj));
        }
    }

    /// `ZapFuncs::drawCurrentPoint` (`xzap.cpp:5782`); draw the current point
    /// marker and contour end markers.
    pub fn draw_current_point(&mut self) {
        let vi = self.vi;
        let obj = imod_object_get(unsafe { (*vi).imod.as_ref() })
            .map_or(ptr::null_mut(), |o| o as *const Iobj as *mut Iobj);
        let cont = imod_contour_get(unsafe { (*vi).imod.as_ref() })
            .map_or(ptr::null_mut(), |c| c as *const Icont as *mut Icont);
        let pnt =
            imod_point_get(unsafe { &mut *(*vi).imod }).map_or(ptr::null(), |p| p as *const Ipoint);
        let (mut x, mut y);
        let mut symbol = IOBJ_SYM_CIRCLE;
        let flags = 0;
        let mut open_add = 0;

        if unsafe { (*vi).drawcursor } == 0 {
            return;
        }

        let min_mod = with_boundary(|n| n.prefs_min_current_mod_pt_size());
        let min_im = with_boundary(|n| n.prefs_min_current_im_pt_size());
        let (mod_pt_size, backup_size, im_pt_size) =
            util_current_point_size(unsafe { obj.as_ref() }, min_mod, min_im, unsafe {
                (*vi).xybin
            });

        // 11/11/04: Reset line width for slice lines or current image point,
        // set it below with object-specific thickness
        let scale = S_SCALE_SIZES.get();
        with_boundary(|n| n.b3d_line_width(scale, ptr::null_mut()));

        if unsafe { (*(*vi).imod).mousemode } == IMOD_MMOVIE || pnt.is_null() {
            x = self.xpos(((unsafe { (*vi).xmouse } as i32) as f64 + 0.5) as f32);
            y = self.ypos(((unsafe { (*vi).ymouse } as i32) as f64 + 0.5) as f32);
            let curpoint = APP.lock().unwrap().as_ref().map_or(0, |a| a.curpoint);
            with_boundary(|n| n.b3d_color_index(curpoint));
            let s = im_pt_size * scale;
            with_boundary(|n| n.b3d_draw_plus(x, y, s));
        } else if !cont.is_null() && unsafe { !(&(*cont).pts).is_empty() } && !pnt.is_null() {
            let width = scale * unsafe { (*obj).linewidth2 } as i32;
            with_boundary(|n| n.b3d_line_width(width, obj));
            let mut cur_size = mod_pt_size;
            let psize = unsafe { (&(*cont).pts).len() };
            if psize > 1
                && (std::ptr::eq(pnt, unsafe { (&(*cont).pts).as_ptr() })
                    || std::ptr::eq(pnt, unsafe { (&(*cont).pts).as_ptr().add(psize - 1) }))
            {
                cur_size = backup_size;
            }

            /* DNM 6/17/01: display off-time features as if off-section */
            x = self.xpos(unsafe { (*pnt).x });
            y = self.ypos(unsafe { (*pnt).y });
            if self.point_visable(unsafe { &*pnt }) != 0
                && !unsafe { ivw_time_mismatch(vi, self.time_lock, obj, cont) }
            {
                let curpoint = APP.lock().unwrap().as_ref().map_or(0, |a| a.curpoint);
                with_boundary(|n| n.b3d_color_index(curpoint));
            } else {
                let shadow = APP.lock().unwrap().as_ref().map_or(0, |a| a.shadow);
                with_boundary(|n| n.b3d_color_index(shadow));
            }
            let r = scale * cur_size;
            with_boundary(|n| n.b3d_draw_circle(x, y, r, false));
        }

        /* draw begin/end points for current contour */
        if !cont.is_null() {
            if unsafe { ivw_time_mismatch(vi, self.time_lock, obj, cont) } {
                return;
            }

            if iobj_close(unsafe { (*obj).flags }) != 0
                && unsafe { (*cont).flags } & ICONT_OPEN != 0
            {
                symbol = IOBJ_SYM_TRIANGLE;
                open_add = 1;
            }
            let width = scale * unsafe { (*obj).linewidth2 } as i32;
            with_boundary(|n| n.b3d_line_width(width, obj));
            let psize = unsafe { (&(*cont).pts).len() };
            if psize > 1 {
                if self.point_visable(unsafe { &(&(*cont).pts)[0] }) != 0 {
                    let bgnpoint = APP.lock().unwrap().as_ref().map_or(0, |a| a.bgnpoint);
                    with_boundary(|n| n.b3d_color_index(bgnpoint));
                    let (px, py) = (
                        self.xpos(unsafe { (&(*cont).pts)[0].x }),
                        self.ypos(unsafe { (&(*cont).pts)[0].y }),
                    );
                    let size = scale * (mod_pt_size + open_add);
                    with_boundary(|n| util_draw_symbol(n, px, py, symbol, size, flags));
                }
                if self.point_visable(unsafe { &(&(*cont).pts)[psize - 1] }) != 0 {
                    let endpoint = APP.lock().unwrap().as_ref().map_or(0, |a| a.endpoint);
                    with_boundary(|n| n.b3d_color_index(endpoint));
                    let (px, py) = (
                        self.xpos(unsafe { (&(*cont).pts)[psize - 1].x }),
                        self.ypos(unsafe { (&(*cont).pts)[psize - 1].y }),
                    );
                    let size = scale * (mod_pt_size + open_add);
                    with_boundary(|n| util_draw_symbol(n, px, py, symbol, size, flags));
                }
            }
        }

        with_boundary(|n| n.b3d_line_width(scale, ptr::null_mut()));
        self.showed_slice = self.showslice as i32;
        if self.showslice != 0 {
            let foreground = APP.lock().unwrap().as_ref().map_or(0, |a| a.foreground);
            with_boundary(|n| n.b3d_color_index(foreground));
            let (a, b, c, d) = (
                self.xpos(unsafe { (*vi).slice.zx1 } as f32 + 0.5f32),
                self.ypos(unsafe { (*vi).slice.zy1 } as f32 + 0.5f32),
                self.xpos(unsafe { (*vi).slice.zx2 } as f32 + 0.5f32),
                self.ypos(unsafe { (*vi).slice.zy2 } as f32 + 0.5f32),
            );
            with_boundary(|n| n.b3d_draw_line(a, b, c, d));
            self.showslice = 0;
        }
    }

    /// `ZapFuncs::drawGhost` (`xzap.cpp:5870`); draw ghost contours.
    pub fn draw_ghost(&mut self) {
        let mut shade;
        let mod_ = unsafe { (*self.vi).imod };
        let mut cont_props = DrawProps::default();
        let mut pt_props = DrawProps::default();
        let (mut nextz, mut prevz, mut iz);
        let stipple_gaps = with_boundary(|n| n.ifg_stipple_gaps());
        let (mut npt, mut last_x, mut last_y, mut this_x, mut this_y, mut delta_z, mut last_shade);
        let (mut delz, mut zscale, mut drawsize);
        let (mut scattered, mut use_point_color);
        let mut next_change;
        let mut state_flags = 0;
        let mut change_flags = 0;
        let mut handle_flags;

        if mod_.is_null() {
            return;
        }

        if unsafe { (*self.vi).ghostmode } & IMOD_GHOST_SECTION == 0 {
            return;
        }

        let scale = S_SCALE_SIZES.get();
        for ob in 0..unsafe { (&(*mod_).obj).len() as i32 } {
            let obj = unsafe { (&mut (*mod_).obj).as_mut_ptr().add(ob as usize) };
            scattered = iobj_scat(unsafe { (*obj).flags }) != 0;
            if ob != unsafe { (*mod_).cindex.object }
                && !((!scattered && unsafe { (*self.vi).ghostmode } & IMOD_GHOST_ALLOBJ != 0)
                    || (scattered && unsafe { (*self.vi).ghostmode } & IMOD_GHOST_ALLSCAT != 0))
            {
                continue;
            }

            handle_flags = HANDLE_2DWIDTH;
            if with_boundary(|n| n.ifg_setup_value_drawing(obj, GEN_STORE_MINMAX1)) != 0 {
                handle_flags |= HANDLE_VALUE1;
            }

            /* DNM 6/16/01: need to be based on mSection, not zmouse */
            zscale = ((if unsafe { (*(*self.vi).imod).zscale } != 0. {
                unsafe { (*(*self.vi).imod).zscale }
            } else {
                1.
            }) * unsafe { (*self.vi).zbin } as f32)
                / unsafe { (*self.vi).xybin } as f32;
            delta_z = unsafe { (*self.vi).ghostdist };
            if scattered {
                delta_z = (((1.max(unsafe { (*obj).pdrawsize })
                    * (2 + unsafe { (*self.vi).ghostdist })) as f32
                    / zscale
                    - 1.) as f64
                    + 0.5)
                    .floor() as i32;
            }
            if unsafe { (*self.vi).ghostdist } != 0 {
                nextz = self.section + delta_z;
                prevz = self.section - delta_z;
            } else {
                nextz = util_next_sec_with_cont(
                    unsafe { &*self.vi },
                    unsafe { obj.as_ref() },
                    self.section,
                    1,
                );
                prevz = util_next_sec_with_cont(
                    unsafe { &*self.vi },
                    unsafe { obj.as_ref() },
                    self.section,
                    -1,
                );
                if scattered {
                    nextz += delta_z;
                    prevz -= delta_z;
                }
            }
            let draw_prev = unsafe { (*self.vi).ghostmode } & IMOD_GHOST_PREVSEC != 0;
            let draw_next = unsafe { (*self.vi).ghostmode } & IMOD_GHOST_NEXTSEC != 0;

            for co in 0..unsafe { (&(*obj).cont).len() as i32 } {
                let cont = unsafe { (&mut (*obj).cont).as_mut_ptr().add(co as usize) };
                next_change = with_boundary(|n| {
                    n.ifg_handle_cont_change(
                        obj,
                        co,
                        &mut cont_props,
                        &mut pt_props,
                        &mut state_flags,
                        handle_flags,
                        0,
                        scale,
                    )
                });
                if cont_props.gap != 0 {
                    continue;
                }

                if scattered {
                    last_shade = 0;
                    use_point_color = false;
                    for pt in 0..unsafe { (&(*cont).pts).len() as i32 } {
                        pt_props.gap = 0;
                        if pt == next_change {
                            let store = unsafe { &raw const (*cont).store };
                            next_change = with_boundary(|n| {
                                n.ifg_handle_next_change(
                                    obj,
                                    store,
                                    &mut cont_props,
                                    &mut pt_props,
                                    &mut state_flags,
                                    &mut change_flags,
                                    handle_flags,
                                    0,
                                    scale,
                                )
                            });
                            last_shade = 0;
                            use_point_color = true;
                        }
                        drawsize = crate::imod::three_dmod::model_draw::imod_point_get_size(
                            unsafe { &*obj },
                            unsafe { &*cont },
                            pt as usize,
                        ) / unsafe { (*self.vi).xybin } as f32;
                        if drawsize > 0. && !(pt_props.gap != 0 && pt_props.valskip != 0) {
                            iz = (unsafe { (&(*cont).pts)[pt as usize].z } as f64 + 0.5).floor()
                                as i32;
                            delz = (unsafe { (&(*cont).pts)[pt as usize].z } - self.section as f32)
                                * zscale;
                            if (delz >= drawsize && draw_prev && iz <= nextz)
                                || (delz <= -drawsize && draw_next && iz >= prevz)
                            {
                                shade = -1;
                                if unsafe { (*self.vi).ghostmode } & IMOD_GHOST_2SHADES != 0
                                    && draw_prev
                                    && draw_next
                                    && iz > self.section
                                {
                                    shade = 1;
                                }
                                shade *=
                                    if unsafe { (*self.vi).ghostmode } & IMOD_GHOST_LIGHTER != 0 {
                                        -1
                                    } else {
                                        1
                                    };
                                if shade != last_shade {
                                    self.set_ghost_color(
                                        if use_point_color {
                                            pt_props.red
                                        } else {
                                            cont_props.red
                                        },
                                        if use_point_color {
                                            pt_props.green
                                        } else {
                                            cont_props.green
                                        },
                                        if use_point_color {
                                            pt_props.blue
                                        } else {
                                            cont_props.blue
                                        },
                                        shade,
                                    );
                                    last_shade = shade;
                                }
                                let (x, y) = (
                                    self.xpos(unsafe { (&(*cont).pts)[pt as usize].x }),
                                    self.ypos(unsafe { (&(*cont).pts)[pt as usize].y }),
                                );
                                let r = (drawsize * self.zoom) as i32;
                                with_boundary(|n| n.b3d_draw_circle(x, y, r, false));
                                if drawsize > 3. {
                                    with_boundary(|n| n.b3d_draw_plus(x, y, 3 * scale));
                                }
                            }
                        }
                    }

                /* DNM: don't display wild contours, only coplanar ones */
                /* By popular demand, display ghosts from lower and upper
                sections */
                } else if unsafe { !(&(*cont).pts).is_empty() }
                    && unsafe { (*cont).flags } & ICONT_WILD == 0
                {
                    iz = (unsafe { (&(*cont).pts)[0].z } as f64 + 0.5).floor() as i32;
                    if (iz > self.section
                        && draw_prev
                        && ((unsafe { (*self.vi).ghostdist } != 0 && iz <= nextz)
                            || (unsafe { (*self.vi).ghostdist } == 0 && iz == nextz)))
                        || (iz < self.section
                            && draw_next
                            && ((unsafe { (*self.vi).ghostdist } != 0 && iz >= prevz)
                                || (unsafe { (*self.vi).ghostdist } == 0 && iz == prevz)))
                    {
                        shade = -1;
                        if unsafe { (*self.vi).ghostmode } & IMOD_GHOST_2SHADES != 0
                            && draw_prev
                            && draw_next
                            && iz > self.section
                        {
                            shade = 1;
                        }
                        shade *= if unsafe { (*self.vi).ghostmode } & IMOD_GHOST_LIGHTER != 0 {
                            -1
                        } else {
                            1
                        };
                        self.set_ghost_color(
                            cont_props.red,
                            cont_props.green,
                            cont_props.blue,
                            shade,
                        );

                        if next_change < 0 {
                            with_boundary(|n| n.b3d_begin_line());
                            for i in 0..unsafe { (&(*cont).pts).len() } {
                                let (x, y) = (
                                    self.xpos(unsafe { (&(*cont).pts)[i].x }),
                                    self.ypos(unsafe { (&(*cont).pts)[i].y }),
                                );
                                with_boundary(|n| n.b3d_vertex_2i(x, y));
                            }

                            /* DNM: connect back to start only if closed */
                            if iobj_close(unsafe { (*obj).flags }) != 0
                                && unsafe { (*cont).flags } & ICONT_OPEN == 0
                            {
                                let (x, y) = (
                                    self.xpos(unsafe { (&(*cont).pts)[0].x }),
                                    self.ypos(unsafe { (&(*cont).pts)[0].y }),
                                );
                                with_boundary(|n| n.b3d_vertex_2i(x, y));
                            }
                            with_boundary(|n| n.b3d_end_line());
                        } else {
                            // If there are changes in contour, then draw only
                            // needed lines
                            if stipple_gaps {
                                with_boundary(|n| n.gl_line_stipple(1, 0x0707));
                            }
                            last_x = self.xpos(unsafe { (&(*cont).pts)[0].x });
                            last_y = self.ypos(unsafe { (&(*cont).pts)[0].y });
                            for pt in 0..unsafe { (&(*cont).pts).len() as i32 } {
                                pt_props.gap = 0;
                                if pt == next_change {
                                    let store = unsafe { &raw const (*cont).store };
                                    next_change = with_boundary(|n| {
                                        n.ifg_handle_next_change(
                                            obj,
                                            store,
                                            &mut cont_props,
                                            &mut pt_props,
                                            &mut state_flags,
                                            &mut change_flags,
                                            handle_flags,
                                            0,
                                            scale,
                                        )
                                    });
                                    if change_flags & CHANGED_COLOR != 0 {
                                        self.set_ghost_color(
                                            pt_props.red,
                                            pt_props.green,
                                            pt_props.blue,
                                            shade,
                                        );
                                    }
                                    if pt_props.gap != 0 && stipple_gaps {
                                        with_boundary(|n| n.b3d_stipple_next_line(true));
                                        pt_props.gap = 0;
                                    }
                                }

                                // Skip gap or last point if open
                                npt = (pt + 1) % unsafe { (&(*cont).pts).len() as i32 };
                                this_x = self.xpos(unsafe { (&(*cont).pts)[npt as usize].x });
                                this_y = self.ypos(unsafe { (&(*cont).pts)[npt as usize].y });
                                if (pt < unsafe { (&(*cont).pts).len() as i32 } - 1
                                    || (iobj_close(unsafe { (*obj).flags }) != 0
                                        && unsafe { (*cont).flags } & ICONT_OPEN == 0))
                                    && pt_props.gap == 0
                                {
                                    let (a, b, c, d) = (last_x, last_y, this_x, this_y);
                                    with_boundary(|n| n.b3d_draw_line(a, b, c, d));
                                }
                                last_x = this_x;
                                last_y = this_y;
                            }
                            if stipple_gaps {
                                with_boundary(|n| n.b3d_stipple_next_line(false));
                            }
                        }
                    }
                }
            }
        }
        with_boundary(|n| n.reset_ghost_color());
    }

    /// `ZapFuncs::setGhostColor` (`xzap.cpp:6040`).
    pub fn set_ghost_color(&mut self, obr: f32, obg: f32, obb: f32, shade: i32) {
        // Set base to 2 to make color get brighter instead of darker
        let base = if shade > 0 { 2 } else { 0 };
        let red = (((base as f32 + obr) as f64 * 255.0) / 3.0) as i32;
        let green = (((base as f32 + obg) as f64 * 255.0) / 3.0) as i32;
        let blue = (((base as f32 + obb) as f64 * 255.0) / 3.0) as i32;

        with_boundary(|n| n.custom_ghost_color(red, green, blue));
    }

    /// `ZapFuncs::drawAuto` (`xzap.cpp:6053`).
    pub fn draw_auto(&mut self) -> i32 {
        let vi = self.vi;
        let (mut x, mut y);
        let mut pixel;

        let xsize = unsafe { (*vi).xsize } as u32;
        let ysize = unsafe { (*vi).ysize } as u32;

        if unsafe { (*vi).ax }.is_null() {
            return -1;
        }

        if !with_boundary(|n| n.autox_filled_at_section(vi, self.section)) {
            return -1;
        }

        /* DNM 8/11/01: make rectangle size be nearest integer and not 0 */
        let rectsize = if self.zoom < 1. {
            1
        } else {
            (self.zoom as f64 + 0.5) as i32
        };
        for j in 0..ysize {
            y = self.ypos(j as f32);
            for i in 0..xsize {
                x = self.xpos(i as f32);
                let index = (i + (j * unsafe { (*vi).xsize } as u32)) as usize;
                let data = with_boundary(|n| n.autox_data(vi, index));
                /*DNM 2/1/01: pick a dark and light color to work in rgb mode */
                if data & AUTOX_BLACK != 0 {
                    pixel = APP.lock().unwrap().as_ref().map_or(0, |a| a.ghost);
                    with_boundary(|n| n.b3d_color_index(pixel));
                    with_boundary(|n| n.b3d_draw_filled_rectangle(x, y, rectsize, rectsize));
                    continue;
                }
                if data & AUTOX_FLOOD != 0 {
                    pixel = APP.lock().unwrap().as_ref().map_or(0, |a| a.endpoint);
                    with_boundary(|n| n.b3d_color_index(pixel));
                    with_boundary(|n| n.b3d_draw_filled_rectangle(x, y, rectsize, rectsize));
                    continue;
                }

                if data & AUTOX_WHITE != 0 {
                    pixel = APP.lock().unwrap().as_ref().map_or(0, |a| a.select);
                    with_boundary(|n| n.b3d_color_index(pixel));
                    with_boundary(|n| n.b3d_draw_filled_rectangle(x, y, rectsize, rectsize));
                }
            }
        }

        0
    }

    /// `ZapFuncs::drawTools` (`xzap.cpp:6107`); send a new value of section,
    /// zoom, or time label if it has changed.
    pub fn draw_tools(&mut self) {
        let (winx, winy);

        if self.tool_max_z != unsafe { (*self.vi).zsize } {
            self.tool_max_z = unsafe { (*self.vi).zsize };
            let max_z = self.tool_max_z;
            with_boundary(|n| n.zap_window_set_max_z(max_z));
        }

        // Workaround to Qt 4.5.0 cocoa bug, need to load these boxes 3 times
        if self.tool_section != self.section {
            if self.tool_zoom <= -4. || self.tool_zoom > -0.9 {
                self.tool_section = self.section;
            }
            let sec = self.section + 1;
            with_boundary(|n| n.zap_window_set_section_text(sec));
        }

        if self.tool_zoom != self.zoom {
            if self.tool_zoom < 0. {
                self.tool_zoom -= 1.;
            }
            if self.tool_zoom <= -4. || self.tool_zoom > -0.9 {
                self.tool_zoom = self.zoom;
            }
            let zoom = self.zoom;
            with_boundary(|n| n.zap_window_set_zoom_text(zoom));
        }

        if self.rubberband != 0 {
            self.band_image_to_mouse(0);
            winx = self.rb_mouse_x1 - 1 - self.rb_mouse_x0;
            winy = self.rb_mouse_y1 - 1 - self.rb_mouse_y0;
        } else {
            winx = self.winx;
            winy = self.winy;
        }
        if winx != self.tool_size_x || winy != self.tool_size_y {
            self.tool_size_x = winx;
            self.tool_size_y = winy;
            with_boundary(|n| n.zap_window_set_size_text(winx, winy));
        }

        if unsafe { (*self.vi).num_times } != 0 {
            let time = ivw_window_time(unsafe { &*self.vi }, self.time_lock);
            if self.tool_time != time {
                self.tool_time = time;
                let label = unsafe {
                    crate::imod::three_dmod::imodview::ivw_get_time_index_label(self.vi, time)
                };
                let label = if label.is_null() {
                    String::new()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(label) }
                        .to_string_lossy()
                        .into_owned()
                };
                with_boundary(|n| n.zap_window_set_time_label(time, &label));
            }
        }
    }

    /// `ZapFuncs::setCursor` (`xzap.cpp:6155`); set the cursor as appropriate
    /// for what is being drawn.
    pub fn set_cursor(&mut self, mode: i32, set_anyway: bool) {
        let need_special = ((self.rubberband != 0 || self.lasso_on)
            && (S_MOVE_BAND_LASSO.get() != 0 || S_DRAG_BAND_LASSO.get() != 0))
            || self.starting_band != 0
            || self.shifting_cont != 0
            || self.drawing_arrow;
        let need_size_all = self.starting_band != 0
            || S_MOVE_BAND_LASSO.get() != 0
            || self.shifting_cont != 0
            || (self.lasso_on && S_DRAG_BAND_LASSO.get() != 0)
            || self.drawing_arrow;
        let need_model = self.drawing_lasso;
        let dragging = S_DRAGGING.get();
        let mut mousemode = self.mousemode;
        let mut last_shape = self.last_shape;
        with_boundary(|n| {
            n.util_set_cursor(
                mode,
                set_anyway,
                need_special,
                need_size_all,
                dragging,
                need_model,
                &mut mousemode,
                &mut last_shape,
            )
        });
        self.mousemode = mousemode;
        self.last_shape = last_shape;
    }

    /// `ZapFuncs::pointVisable` (`xzap.cpp:6167`).
    pub fn point_visable(&self, pnt: &Ipoint) -> i32 {
        if self.twod != 0 {
            return 1;
        }

        /* DNM 11/30/02: replace +/- alternatives with standard nearest int */
        let cz = (pnt.z as f64 + 0.5).floor() as i32;

        if cz == self.section {
            return 1;
        }

        0
    }

    /// `ZapFuncs::setDrawCurrentOnly` (`xzap.cpp:6182`).
    ///
    /// The `QT_VERSION >= 0x060500` partial-update arm is not compiled
    /// against Qt 5, so only the assignment and `imodInfoUpdateOnly` run.
    pub fn set_draw_current_only(&mut self, value: i32) {
        self.draw_current_only = value;
        with_boundary(|n| n.imod_info_update_only(value));
    }

    /// `ZapFuncs::contInSelectArea` (`xzap.h:79`).
    ///
    /// The header declares it private but `xzap.cpp` contains no definition
    /// and nothing calls it; the member exists only as a declaration, so
    /// there is no body to translate.  `imodContInSelectArea`
    /// (`imod_edit.cpp:31`) is the routine the Ctrl-A path actually uses.
    pub fn cont_in_select_area(
        &self,
        _obj: *mut Iobj,
        _cont: *mut Icont,
        _selmin: Ipoint,
        _selmax: Ipoint,
    ) -> i32 {
        report_once(
            "ZapFuncs::contInSelectArea",
            "a definition, which xzap.cpp does not contain",
        );
        0
    }
}
