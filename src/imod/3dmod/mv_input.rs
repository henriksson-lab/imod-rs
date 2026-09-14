//! Translation of `IMOD/3dmod/mv_input.cpp` and `mv_input.h`.
//!
//! Qt turns native events into the payloads below in the window unit.  The
//! state and mathematical portions are retained here exactly as in the
//! source; every call this unit makes into a unit that is only reachable
//! through a widget, a dialog, a preferences object, or an OpenGL context
//! crosses [`MvInputNativeBoundary`], whose default bodies say which unit is
//! missing rather than silently dropping the action.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::b3dutil::{CArg, c_format, sprintf_arg};

use crate::imod::libimod::imat::{
    B3D_X, B3D_Y, B3D_Z, Imat, imod_mat_get_nat_angles, imod_mat_id, imod_mat_mult, imod_mat_new,
    imod_mat_rot, imod_mat_scale, imod_mat_transform,
};
use crate::imod::libimod::imodel::{
    Iindex, Imod, Iobj, Ipoint, imod_get_index, imod_object_get, imod_point_get,
    imod_set_cur_mesh_surf, imod_set_index,
};
use crate::imod::libimod::iobj::{iobj_mesh, iobj_scat};
use crate::imod::libimod::ipoint::{
    imod_point_dot, imod_point_get_size, imod_point_normalize, imod_point_set_size,
};
use crate::imod::three_dmod::imod::{
    imod_debug, imod_print_info, imod_print_stderr, imod_puts, wprint,
};
use crate::imod::three_dmod::imod_input::{
    InputKeyEvent, InputNativeBoundary, KEY_DELETE, KEY_DOWN, KEY_LEFT, KEY_PAGE_DOWN, KEY_PAGE_UP,
    KEY_RIGHT, KEY_UP, input_convert_num_lock, input_delete_contour, input_delete_point,
    input_insert_point, input_next_contour, input_next_point, input_next_time, input_prev_contour,
    input_prev_point, input_prev_time, input_test_ctrl, input_test_meta_key, input_undo_redo,
};
use crate::imod::three_dmod::imodv::{
    ImodvApp, MAX_MOVIE_TIMES, imodv_draw, imodv_draw_imod_images, imodv_finish_chg_unit,
    imodv_new_model_angles, imodv_register_model_chg, imodv_register_object_chg,
};
use crate::imod::three_dmod::imodview::{
    IMOD_MMODEL, ImodView, ivw_bind_mouse, ivw_get_an_extra_object,
};
use crate::imod::three_dmod::mv_image::{IMODV_DRAW_CX, IMODV_DRAW_CY, IMODV_DRAW_CZ};
use crate::imod::three_dmod::mv_modeled::imodv_select_model;
use crate::imod::three_dmod::mv_objed::{
    imodv_objed_change_object, imodv_objed_draw_clip_plane, imodv_objed_move_to_axis,
    imodv_objed_new_view, imodv_objed_set_draw_type_and_style, imodv_objed_toggle_clip,
    objed_object,
};
use crate::imod::three_dmod::mv_window::{
    VVIEW_MENU_CURPNT, VVIEW_MENU_LOWRES, VVIEW_MENU_TRANSBKGD,
};
use crate::imod::three_dmod::utilities::{
    util_close_key, util_mouse_zaxis_rotation, util_wheel_to_point_size_scaling,
};

/// Original: `STANDALONE_INTERVAL` (`mv_input.cpp:14`).
pub const STANDALONE_INTERVAL: i32 = 1;
/// Original: `MODELVIEW_INTERVAL` (`mv_input.cpp:15`).
pub const MODELVIEW_INTERVAL: i32 = 10;
/// Original: `MOUSE_TO_THROW` (`mv_input.cpp:1163`).
pub const MOUSE_TO_THROW: f32 = 0.25;
/// Original: `MIN_SQUARE_TO_THROW` (`mv_input.cpp:1164`).
pub const MIN_SQUARE_TO_THROW: i32 = 17;
/// Original: `SAME_SPEED_DISTANCE` (`mv_input.cpp:1165`).
pub const SAME_SPEED_DISTANCE: f64 = 100.;
/// Original: `SELECT_BUFSIZE` (`mv_input.cpp:1302`).
pub const SELECT_BUFSIZE: usize = 40960;
/// Original: `VIEW_WORLD_INVERT_Z` (`imodel.h:210`).
pub const VIEW_WORLD_INVERT_Z: u32 = 0x40;
/// Original: `VIEW_WORLD_ON` (`imodel.h:202`).
pub const VIEW_WORLD_ON: u32 = 1;
/// Original: `WORLD_QUALITY_SHIFT` (`imodel.h:214`).
pub const WORLD_QUALITY_SHIFT: u32 = 8;
/// Original: `WORLD_QUALITY_BITS` (`imodel.h:215`).
pub const WORLD_QUALITY_BITS: u32 = 7 << WORLD_QUALITY_SHIFT;
/// Original: `WORLD_MOVE_ALL_CLIP` (`imodel.h:218`).
pub const WORLD_MOVE_ALL_CLIP: u32 = 1 << 12;
/// Original: `SnapShot_RGB` (`b3dgfx.h`).
pub const SNAPSHOT_RGB: i32 = 0;
/// Original: `SnapShot_TIF` (`b3dgfx.h`).
pub const SNAPSHOT_TIF: i32 = 1;
/// Original: `IMODV_DIALOG` (`control.h`).
pub const IMODV_DIALOG: i32 = 0;
/// Original: `IMOD_DRAW_XYZ` (`imod.h`).
pub const IMOD_DRAW_XYZ: i32 = 3;

/// Qt-independent payload consumed by the source key and mouse handlers.
///
/// `key` carries the `QKeyEvent::key()` code in the same encoding that
/// `imod_input.rs` already uses for `inputQDefaultKeys` (ASCII for printable
/// keys, `0x0100_00xx` for the named ones), so `imodvKeyPress`'s switch is a
/// direct match on those values.  `modifiers` carries this unit's own mask
/// bits, which are the ones `imodv_query_pointer` combines with the button
/// state into `maskr`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputEvent {
    /// `QKeyEvent::key()`.
    pub key: i32,
    /// `QMouseEvent::x()`.
    pub x: i32,
    /// `QMouseEvent::y()`.
    pub y: i32,
    /// `QMouseEvent::button()` for press/release, `buttons()` for move.
    pub button: u32,
    /// `QMouseEvent::buttons()`, the state after a press or release.
    pub buttons: u32,
    /// `QInputEvent::modifiers()`.
    pub modifiers: u32,
    /// `QWheelEvent::angleDelta().y()`.
    pub delta: i32,
}
/// `Qt::LeftButton` in this unit's mask space.
pub const INPUT_LEFT: u32 = 1;
/// `MIDDLE_BUT` in this unit's mask space.
pub const INPUT_MIDDLE: u32 = 2;
/// `Qt::RightButton` in this unit's mask space.
pub const INPUT_RIGHT: u32 = 4;
/// `Qt::ControlModifier`, the value `inputCtrlModifier()` returns.
pub const INPUT_CTRL: u32 = 8;
/// `Qt::ShiftModifier`.
pub const INPUT_SHIFT: u32 = 16;
/// `Qt::KeypadModifier`.
pub const INPUT_KEYPAD: u32 = 32;

/// `Qt::Key_Escape`.
pub const KEY_ESCAPE: i32 = 0x0100_0000;
/// `Qt::Key_Enter` (the keypad return key).
pub const KEY_ENTER: i32 = 0x0100_0005;
/// `Qt::Key_Shift`.
pub const KEY_SHIFT: i32 = 0x0100_0020;
/// `Qt::Key_Control`.
pub const KEY_CONTROL: i32 = 0x0100_0021;
/// `Qt::Key_F1`.
pub const KEY_F1: i32 = 0x0100_0030;
/// `Qt::Key_F2`.
pub const KEY_F2: i32 = 0x0100_0031;
/// `Qt::Key_F3`.
pub const KEY_F3: i32 = 0x0100_0032;
/// `Qt::Key_F4`.
pub const KEY_F4: i32 = 0x0100_0033;
/// `Qt::Key_F11`.
pub const KEY_F11: i32 = 0x0100_003a;
/// `Qt::Key_Ampersand`.
pub const KEY_AMPERSAND: i32 = 0x26;
/// `Qt::Key_ParenLeft`.
pub const KEY_PAREN_LEFT: i32 = 0x28;
/// `Qt::Key_ParenRight`.
pub const KEY_PAREN_RIGHT: i32 = 0x29;
/// `Qt::Key_Plus`.
pub const KEY_PLUS: i32 = 0x2b;
/// `Qt::Key_Comma`.
pub const KEY_COMMA: i32 = 0x2c;
/// `Qt::Key_Minus`.
pub const KEY_MINUS: i32 = 0x2d;
/// `Qt::Key_Period`.
pub const KEY_PERIOD: i32 = 0x2e;
/// `Qt::Key_0`.
pub const KEY_0: i32 = 0x30;
/// `Qt::Key_1`.
pub const KEY_1: i32 = 0x31;
/// `Qt::Key_2`.
pub const KEY_2: i32 = 0x32;
/// `Qt::Key_3`.
pub const KEY_3: i32 = 0x33;
/// `Qt::Key_4`.
pub const KEY_4: i32 = 0x34;
/// `Qt::Key_5`.
pub const KEY_5: i32 = 0x35;
/// `Qt::Key_6`.
pub const KEY_6: i32 = 0x36;
/// `Qt::Key_7`.
pub const KEY_7: i32 = 0x37;
/// `Qt::Key_8`.
pub const KEY_8: i32 = 0x38;
/// `Qt::Key_9`.
pub const KEY_9: i32 = 0x39;
/// `Qt::Key_Equal`.
pub const KEY_EQUAL: i32 = 0x3d;
/// `Qt::Key_A` .. `Qt::Key_Z` are the ASCII upper-case codes.
pub const KEY_A: i32 = 0x41;
pub const KEY_B: i32 = 0x42;
pub const KEY_C: i32 = 0x43;
pub const KEY_D: i32 = 0x44;
pub const KEY_F: i32 = 0x46;
pub const KEY_G: i32 = 0x47;
pub const KEY_H: i32 = 0x48;
pub const KEY_I: i32 = 0x49;
pub const KEY_J: i32 = 0x4a;
pub const KEY_K: i32 = 0x4b;
pub const KEY_L: i32 = 0x4c;
pub const KEY_M: i32 = 0x4d;
pub const KEY_N: i32 = 0x4e;
pub const KEY_O: i32 = 0x4f;
pub const KEY_P: i32 = 0x50;
pub const KEY_Q: i32 = 0x51;
pub const KEY_R: i32 = 0x52;
pub const KEY_S: i32 = 0x53;
pub const KEY_T: i32 = 0x54;
pub const KEY_U: i32 = 0x55;
pub const KEY_V: i32 = 0x56;
pub const KEY_X: i32 = 0x58;
pub const KEY_Y: i32 = 0x59;
pub const KEY_Z: i32 = 0x5a;
/// `Qt::Key_BracketLeft`.
pub const KEY_BRACKET_LEFT: i32 = 0x5b;
/// `Qt::Key_BracketRight`.
pub const KEY_BRACKET_RIGHT: i32 = 0x5d;
/// `Qt::Key_Underscore`.
pub const KEY_UNDERSCORE: i32 = 0x5f;
/// `Qt::Key_BraceLeft`.
pub const KEY_BRACE_LEFT: i32 = 0x7b;
/// `Qt::Key_BraceRight`.
pub const KEY_BRACE_RIGHT: i32 = 0x7d;

/// Everything `mv_input.cpp` reaches that is a widget, a dialog, a
/// preferences object, an undo stack, or an OpenGL context.
///
/// Each method is one call the source makes, named after it.  The default
/// body reports the unit that has no native host yet and does nothing else;
/// an attached host overrides the ones it can really perform.  `mv_input.cpp`
/// also calls `imod_input.cpp` directly, so a host is an
/// [`InputNativeBoundary`] as well.
pub trait MvInputNativeBoundary: InputNativeBoundary {
    /// `a->mainWin->mCurGLw->mapFromGlobal(QCursor::pos())`
    /// (`mv_input.cpp:85-86`).  The source re-reads the pointer rather than
    /// using the event position; with no cursor host the last event position
    /// the handlers recorded is the closest available answer.
    fn query_pointer_position(&mut self, a: &ImodvApp) -> (i32, i32) {
        (a.lastmx, a.lastmy)
    }
    /// `ImodPrefs->actualModvButton(logicalButton)` (`preferences.cpp:1283`),
    /// returned in this unit's mask space.  The default is the unswapped
    /// mapping, which is what `ImodPreferences` returns with
    /// `modvSwapLeftMid` off.
    fn actual_modv_button(&mut self, logical_button: i32) -> u32 {
        match logical_button {
            1 => INPUT_LEFT,
            2 => INPUT_MIDDLE,
            _ => INPUT_RIGHT,
        }
    }
    /// `a->mainWin->close()` (`mv_input.cpp:117`); `imodv_close`
    /// (`imodv.cpp:696`) is that same call on that same window.
    fn main_win_close(&mut self) {
        crate::imod::three_dmod::imodv::imodv_close();
    }
    /// `ImodvClosed` (`imodv.h`), read by `imodvMovieTimeout`.  `imodv.rs`
    /// keeps the flag inside its `IMODV_STATE` thread-local with no reader,
    /// so a host that owns the window reports it instead.
    fn imodv_closed(&mut self) -> bool {
        false
    }
    /// `imodvMenuBgcolor(state)` (`mv_menu.cpp`).
    fn imodv_menu_bgcolor(&mut self, state: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvMenuBgcolor needs the background colour dialog of mv_menu.cpp, \
                 which has no native host yet"
            )
        });
    }
    /// `imodv_control(a, state)` (`mv_control.cpp`).
    fn imodv_control(&mut self, a: &mut ImodvApp, state: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodv_control needs the model view control dialog of mv_control.cpp, \
                 which has no native host yet"
            )
        });
    }
    /// `mvImageTogglePlane(flag)` (`mv_image.cpp`).
    fn mv_image_toggle_plane(&mut self, a: &mut ImodvApp, flag: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: mvImageTogglePlane needs the image state of mv_image.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `mvImageUpdate(a)` (`mv_image.cpp`).
    fn mv_image_update(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: mvImageUpdate needs the image state of mv_image.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `mvImageEditDialog(a, state)` (`mv_image.cpp`).
    fn mv_image_edit_dialog(&mut self, a: &mut ImodvApp, state: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: mvImageEditDialog needs the image dialog of mv_image.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `ImodPrefs->set2ndSnapFormat()` (`preferences.cpp`).
    fn set_2nd_snap_format(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: ImodPrefs->set2ndSnapFormat needs the settings object of \
                 preferences.cpp, which has no native host yet"
            )
        });
    }
    /// `ImodPrefs->restoreSnapFormat()` (`preferences.cpp`).
    fn restore_snap_format(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: ImodPrefs->restoreSnapFormat needs the settings object of \
                 preferences.cpp, which has no native host yet"
            )
        });
    }
    /// `imodv_auto_snapshot(name, format)` (`mv_gfx.cpp`).
    fn imodv_auto_snapshot(&mut self, a: &mut ImodvApp, name: &str, format_type: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodv_auto_snapshot needs the snapshot writer of mv_gfx.cpp and \
                 b3dgfx.cpp, which has no native host yet"
            )
        });
    }
    /// `imodvStereoToggle()` (`mv_stereo.cpp`).
    fn imodv_stereo_toggle(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvStereoToggle needs the stereo state of mv_stereo.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `imodvStereoUpdate()` (`mv_stereo.cpp`).
    fn imodv_stereo_update(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvStereoUpdate needs the stereo dialog of mv_stereo.cpp, which has \
                 no native host yet"
            )
        });
    }
    /// `imodvControlLinkUpdate(a)` (`mv_control.cpp`).
    fn imodv_control_link_update(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvControlLinkUpdate needs the control dialog of mv_control.cpp, \
                 which has no native host yet"
            )
        });
    }
    /// `imodvObjectListDialog(a, state)` (`mv_listobj.cpp`).
    fn imodv_object_list_dialog(&mut self, a: &mut ImodvApp, state: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvObjectListDialog needs the object list dialog of mv_listobj.cpp, \
                 which has no native host yet"
            )
        });
    }
    /// `imodvControlChangeSteps(a, delta)` (`mv_control.cpp`).
    fn imodv_control_change_steps(&mut self, a: &mut ImodvApp, delta: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvControlChangeSteps needs the control dialog of mv_control.cpp, \
                 which has no native host yet"
            )
        });
    }
    /// `imodvModelEditDialog(a, state)` (`mv_modeled.cpp`).
    fn imodv_model_edit_dialog(&mut self, a: &mut ImodvApp, state: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvModelEditDialog needs the model edit dialog of mv_modeled.cpp, \
                 which has no native host yet"
            )
        });
    }
    /// `imodvObjedMeshObject()` (`mv_objed.cpp`).
    fn imodv_objed_mesh_object(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvObjedMeshObject needs the object editor of mv_objed.cpp, which has \
                 no native host yet"
            )
        });
    }
    /// `mvMovieDialog(a, state)` (`mv_movie.cpp`).
    fn mv_movie_dialog(&mut self, a: &mut ImodvApp, state: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: mvMovieDialog needs the movie dialog of mv_movie.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `mvMovieSequenceDialog(a, state)` (`mv_movie.cpp`).
    fn mv_movie_sequence_dialog(&mut self, a: &mut ImodvApp, state: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: mvMovieSequenceDialog needs the movie sequence dialog of mv_movie.cpp, \
                 which has no native host yet"
            )
        });
    }
    /// `vbCleanupVBD(imod)` (`vertexbuffer.cpp`).
    fn vb_cleanup_vbd(&mut self, imod: *mut Imod) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: vbCleanupVBD needs the vertex buffer manager of vertexbuffer.cpp, which \
                 has no native host yet"
            )
        });
    }
    /// `imodvViewEditDialog(a, state)` (`mv_views.cpp`).
    fn imodv_view_edit_dialog(&mut self, a: &mut ImodvApp, state: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvViewEditDialog needs the view editor of mv_views.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `imeSetViewData(wi)` (`mv_modeled.cpp`).
    fn ime_set_view_data(&mut self, wi: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imeSetViewData needs the model edit dialog of mv_modeled.cpp, which has \
                 no native host yet"
            )
        });
    }
    /// `imodvIsosurfaceEditDialog(a, state)` (`isosurface.cpp`).
    fn imodv_isosurface_edit_dialog(&mut self, a: &mut ImodvApp, state: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvIsosurfaceEditDialog needs the isosurface dialog of \
                 isosurface.cpp, which has no native host yet"
            )
        });
    }
    /// `imodvIsosurfaceInvertThreshold()` (`isosurface.cpp`).
    fn imodv_isosurface_invert_threshold(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvIsosurfaceInvertThreshold needs the isosurface dialog of \
                 isosurface.cpp, which has no native host yet"
            )
        });
    }
    /// `imodvControlStart()` (`mv_control.cpp`).
    fn imodv_control_start(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvControlStart needs the rotation tool of mv_control.cpp and \
                 mv_window.cpp, which has no native host yet"
            )
        });
    }
    /// `a->mainWin->openRotationTool(a)` (`mv_window.cpp`).
    fn open_rotation_tool(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: openRotationTool needs the rotation tool window of rotationtool.cpp, \
                 which has no native host yet"
            )
        });
    }
    /// `imodvViewMenu(which)` (`mv_menu.cpp`).
    fn imodv_view_menu(&mut self, a: &mut ImodvApp, which: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvViewMenu needs the menu action set of mv_menu.cpp and \
                 mv_window.cpp, which has no native host yet"
            )
        });
    }
    /// `imodvMenuLowres(a->lowres)` (`mv_menu.cpp`).
    fn imodv_menu_lowres(&mut self, a: &mut ImodvApp, value: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvMenuLowres needs the menu action set of mv_menu.cpp and \
                 mv_window.cpp, which has no native host yet"
            )
        });
    }
    /// `fineGrainApplyLast()` (`finegrain.cpp`).
    fn fine_grain_apply_last(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: fineGrainApplyLast needs the fine grain dialog of finegrain.cpp, which \
                 has no native host yet"
            )
        });
    }
    /// `imodvSelectVisibleConts(a, pickedOb, pickedCo)` (`mv_ogl.h:24`); both
    /// indices are `int &` and come back updated.
    fn imodv_select_visible_conts(
        &mut self,
        a: &mut ImodvApp,
        picked_ob: &mut i32,
        picked_co: &mut i32,
    ) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvSelectVisibleConts needs the OpenGL draw state of mv_ogl.cpp, \
                 which has no native host yet"
            )
        });
    }
    /// `imod_setxyzmouse()` (`info_cb.cpp`).
    fn imod_setxyzmouse(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imod_setxyzmouse needs the information window of info_cb.cpp, which has \
                 no native host yet"
            )
        });
    }
    /// `imod_info_input()` (`info_cb.cpp`).
    fn imod_info_input(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imod_info_input needs the event dispatcher of info_cb.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `imodv_setbuffer(a, db, stereo, alpha)` (`mv_gfx.cpp`).
    fn imodv_setbuffer(&mut self, a: &mut ImodvApp, db: i32, stereo: i32, alpha: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodv_setbuffer needs the GL widget stack of mv_gfx.cpp and \
                 mv_window.cpp, which has no native host yet"
            )
        });
    }
    /// `a->mainWin->setEnabledMenuItem(id, state)` (`mv_window.cpp`).
    fn set_enabled_menu_item(&mut self, a: &mut ImodvApp, id: usize, state: bool) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: setEnabledMenuItem needs the menu bar of mv_window.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `objed(Imodv)` (`mv_objed.cpp`).
    fn objed(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: objed needs the object editor dialog of mv_objed.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `light_moveby(a->imod->view, x, y)` (`mv_light.cpp`).
    fn light_moveby(&mut self, a: &mut ImodvApp, x: i32, y: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: light_moveby needs the lighting state of mv_light.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `imodv_winset(a)` (`mv_gfx.cpp`).
    fn imodv_winset(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodv_winset needs the GL context of mv_gfx.cpp, which has no native \
                 host yet"
            )
        });
    }
    /// `glSelectBuffer(SELECT_BUFSIZE, buf)`: registers the buffer the
    /// selection-mode draw fills before it sets `a->pickHits`.
    fn gl_select_buffer(&mut self, a: &mut ImodvApp, buffer: &mut [u32]) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: legacy picking needs the GL selection buffer of mv_ogl.cpp, which has \
                 no native host yet"
            )
        });
    }
    /// `imodvModelDrawRange(a, mstart, mend)` and
    /// `imodvUnprojectPickedPoint(a, mend)` (`mv_modeled.cpp`, `mv_ogl.cpp`).
    fn imodv_unproject_picked_point(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvUnprojectPickedPoint needs the OpenGL projection state of \
                 mv_ogl.cpp, which has no native host yet"
            )
        });
    }
    /// `findClickedDrawnElement(a, curObj, moNum, obNum, coNum, ptNum)`
    /// (`mv_ogl.cpp`).
    fn find_clicked_drawn_element(
        &mut self,
        a: &mut ImodvApp,
        cur_obj: bool,
        mo_num: &mut i32,
        ob_num: &mut i32,
        co_num: &mut i32,
        pt_num: &mut i32,
    ) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: findClickedDrawnElement needs the OpenGL draw state of mv_ogl.cpp, \
                 which has no native host yet"
            )
        });
    }
    /// `imodSelectionListQuery(a->vi, ob, co)` (`imod_edit.cpp`).
    fn imod_selection_list_query(&mut self, a: &mut ImodvApp, ob: i32, co: i32) -> i32 {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodSelectionListQuery needs the selection list that imod_edit.cpp \
                 keeps on ImodView, which has no native host yet"
            )
        });
        -2
    }
    /// `imodSelectionNewCurPoint(a->vi, a->imod, indSave, ctrlDown)`
    /// (`imod_edit.cpp`).
    fn imod_selection_new_cur_point(&mut self, a: &mut ImodvApp, ind_save: Iindex, ctrl_down: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodSelectionNewCurPoint needs the selection list that imod_edit.cpp \
                 keeps on ImodView, which has no native host yet"
            )
        });
    }
    /// `ilistSize(a->vi->selectionList)` (`ilist.c`).
    ///
    /// Not an error path: with no selection list on `ImodView` the list is
    /// empty, and 0 is what `ilistSize` returns for it.  The queries beside
    /// this one do report.
    fn selection_list_size(&mut self, a: &mut ImodvApp) -> i32 {
        0
    }
    /// `imodDraw(a->vi, flag)` (`display.cpp`).
    fn imod_draw(&mut self, a: &mut ImodvApp, flag: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodDraw needs the image window set of display.cpp, which has no native \
                 host yet"
            )
        });
    }
    /// `a->vi->undo->contourDataChg()` (`undoredo.cpp`).
    fn undo_contour_data_chg(&mut self, a: &mut ImodvApp) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: undo->contourDataChg needs the undo stack of undoredo.cpp, which has no \
                 native host yet"
            )
        });
    }
    /// `Imodv->mainWin->isVisible()` / `raise()` / `activateWindow()`
    /// (`mv_input.cpp:1683-1690`).
    fn main_win_raise(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: raising the model view window needs the window manager calls of \
                 mv_window.cpp, which has no native host yet"
            )
        });
    }
    /// `imodvDialogManager.raise(IMODV_DIALOG)` (`control.cpp`).
    fn dialog_manager_raise(&mut self, which: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvDialogManager.raise has no model view dialogs to raise: none are \
                 registered with the DialogManager of control.cpp yet"
            )
        });
    }
    /// `a->mainWin->mTimer->start(interval)` followed by `timerId()`
    /// (`mv_input.cpp:1700-1701`).
    fn movie_timer_start(&mut self, interval: i32) -> i32 {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: the movie timer needs the QTimer of mv_window.cpp, which has no native \
                 host yet"
            )
        });
        0
    }
    /// `a->mainWin->mTimer->stop()` (`mv_input.cpp:1730`).
    fn movie_timer_stop(&mut self) {}
    /// `a->mainWin->releaseKeyboard()` (`mv_input.cpp:1053,1059`).
    ///
    /// Not an error path: the matching `grabKeyboard()` calls in
    /// `imodvKeyPress` are commented out in the source ("Grabs seem not to be
    /// needed and avoiding them saves a lot of trouble"), so there is no grab
    /// to release.
    fn release_keyboard(&mut self) {}
}

thread_local! {
    /// Original static: `sPickedObject` (`mv_input.cpp:71`).
    static S_PICKED_OBJECT: std::cell::Cell<i32> = const { std::cell::Cell::new(-1) };
    /// Original static: `sPickedContour` (`mv_input.cpp:72`).
    static S_PICKED_CONTOUR: std::cell::Cell<i32> = const { std::cell::Cell::new(-1) };
    /// Original static: `sCtrlDown` (`mv_input.cpp:74`).
    static S_CTRL_DOWN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Original static: `sShiftDown` (`mv_input.cpp:75`).
    static S_SHIFT_DOWN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Original static: `sLeftDown` (`mv_input.cpp:76`).
    static S_LEFT_DOWN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Original static: `sMidDown` (`mv_input.cpp:77`).
    static S_MID_DOWN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Original static: `sRightDown` (`mv_input.cpp:78`).
    static S_RIGHT_DOWN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Original static: `sFirstMove` (`mv_input.cpp:79`).
    static S_FIRST_MOVE: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    /// Original static: `sConfirmModifiers` (`mv_input.cpp:80`).
    static S_CONFIRM_MODIFIERS: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    /// Original static: `b2x` (`mv_input.cpp:91`).
    static B2X: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    /// Original static: `b2y` (`mv_input.cpp:92`).
    static B2Y: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    /// Static local of `imodvMouseMove` (`mv_input.cpp:757`).
    static S_PROCESSING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Static locals `ex`, `ey`, `shift` and `ctrl` of `imodvMouseMove`
    /// (`mv_input.cpp:756`).  They are shared, not per-call: an event that
    /// arrives while `imod_info_input` is processing overwrites them and
    /// returns, and the outer call then uses the newest position.
    static MOVE_EX: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    static MOVE_EY: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    static MOVE_SHIFT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static MOVE_CTRL: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
}

/// Original static: `imodv_query_pointer` (`mv_input.cpp:83`).
///
/// Look up current position of pointer and report last state of buttons/keys.
pub fn imodv_query_pointer(
    a: &ImodvApp,
    wx: &mut i32,
    wy: &mut i32,
    n: &mut dyn MvInputNativeBoundary,
) -> u32 {
    let maskr = S_LEFT_DOWN.get()
        | S_MID_DOWN.get()
        | S_RIGHT_DOWN.get()
        | S_CTRL_DOWN.get()
        | S_SHIFT_DOWN.get();
    let (x, y) = n.query_pointer_position(a);
    *wx = x;
    *wy = y;
    maskr
}

/// Original: `imodvKeyPress` (`mv_input.cpp:94`).
pub fn imodv_key_press(a: &mut ImodvApp, event: InputEvent, n: &mut dyn MvInputNativeBoundary) {
    let a_ptr = a as *mut ImodvApp;
    let mut keysym = event.key;
    let mut tstep = 1;
    let mut fastdraw: i32;
    let mut elapsed: f32;
    let mut ob = 0;
    let mut co = 0;
    let mut pt = 0;
    let imod = a.imod;
    let state = event.modifiers;
    let mut keypad = i32::from(event.modifiers & INPUT_KEYPAD != 0);
    let shifted = state & INPUT_SHIFT;
    let key_event = InputKeyEvent {
        key: event.key,
        modifiers: (if state & INPUT_SHIFT != 0 {
            crate::imod::three_dmod::imod_input::INPUT_SHIFT
        } else {
            0
        }) | (if state & INPUT_CTRL != 0 {
            crate::imod::three_dmod::imod_input::INPUT_CTRL
        } else {
            0
        }) | (if keypad != 0 {
            crate::imod::three_dmod::imod_input::INPUT_KEYPAD
        } else {
            0
        }),
        accepted: false,
    };
    let ctrl = input_test_ctrl(&key_event);
    let mut qstr = String::new();
    // `mv_input.cpp:110` declares `QString qstr, qstr2`; `SPRINTF(qstr2)(...)`
    // assigns the whole formatted string, so there is no buffer and no
    // truncation.
    let mut qstr2: String;
    let scattered_obj: bool;

    if input_test_meta_key(&key_event) {
        return;
    }

    if util_close_key(keysym) || keysym == KEY_Q {
        n.main_win_close();
        return;
    }
    input_convert_num_lock(&mut keysym, &mut keypad);
    confirm_modifiers(state, n);

    // Increase step size for shift, except not when moving a point
    if shifted != 0 && ctrl == 0 {
        tstep = 10;
    }
    if shifted != 0 && ctrl != 0 && (keysym == KEY_PAGE_DOWN || keysym == KEY_PAGE_UP) {
        // `a->imod->view->rad` is read before the `!Imodv->imod` test below.
        let rad = unsafe { a.imod.as_ref() }
            .and_then(|m| m.view.first())
            .map_or(1., |v| v.rad);
        tstep = (0.5 * a.winx.min(a.winy) as f32 / rad) as i32;
        tstep = 1.max(tstep);
    }

    if a.imod.is_null() {
        return;
    }
    let imod_ref = unsafe { &mut *imod };

    imod_get_index(imod_ref, &mut ob, &mut co, &mut pt);
    // `obj = &imod->obj[ob]` is a plain address computation in the source, so
    // `if (obj)` is always true even when `ob` is -1 and the read is past the
    // array.  A missing element is represented by a null pointer here.
    let obj: *mut Iobj = imod_ref
        .obj
        .get_mut(ob as usize)
        .map_or(std::ptr::null_mut(), |o| o as *mut Iobj);
    scattered_obj = if obj.is_null() {
        false
    } else {
        iobj_scat(unsafe { (*obj).flags }) != 0
    };

    if imod_debug('k') {
        imod_print_stderr(&format!("key {keysym:x}\n"));
    }

    match keysym {
        KEY_B => {
            if shifted != 0 {
                // DNM 12/1/02: call this so it can keep track of open/closed state
                n.imodv_menu_bgcolor(1);
            } else {
                imodv_objed_move_to_axis(a, 2);
            }
        }

        // Kludge, add clip data to model/object later.
        // print the current clipping plane parameters
        KEY_C => {
            if shifted != 0 && ctrl != 0 {
                if scattered_obj {
                    input_next_contour(unsafe { &mut *(a.vi as *mut ImodView) }, n);
                }
            } else if shifted != 0 {
                n.imodv_control(a, 1);
            } else if ctrl != 0 {
                imodv_objed_toggle_clip(a, -1, -1);
            } else {
                let obj = objed_object(unsafe { &mut *a_ptr })
                    .map_or(std::ptr::null_mut(), |o| o as *mut Iobj);
                if !obj.is_null() {
                    // `mv_input.cpp:167` `a->imod->editGlobalClip`.
                    let edit_global_clip = imod_ref.edit_global_clip;
                    let clips = if edit_global_clip != 0 {
                        &mut imod_ref.view[0].clips
                    } else {
                        unsafe { &mut (*obj).clips }
                    };
                    let ip = clips.plane as usize;

                    // DNM 7/31/01 remove pixsize from D
                    let zscale = imod_ref.zscale;
                    qstr2 = c_format(
                        "Current %s clip data = (A B C D) = %g %g %g %g.\n",
                        &[
                            CArg::Str(if edit_global_clip != 0 {
                                "Global"
                            } else {
                                "Object"
                            }),
                            CArg::Dbl(sprintf_arg(clips.normal[ip].x as f64)),
                            CArg::Dbl(sprintf_arg(clips.normal[ip].y as f64)),
                            CArg::Dbl(sprintf_arg((clips.normal[ip].z / zscale) as f64)),
                            CArg::Dbl(sprintf_arg(
                                ((clips.normal[ip].x * clips.point[ip].x)
                                    + (clips.normal[ip].y * clips.point[ip].y)
                                    + (clips.normal[ip].z * clips.point[ip].z))
                                    as f64,
                            )),
                        ],
                    );
                    qstr = qstr2;
                    imod_print_info(&qstr);
                }
            }
        }

        KEY_X => {
            if shifted != 0 && ctrl != 0 {
                if scattered_obj {
                    input_prev_contour(unsafe { &mut *(a.vi as *mut ImodView) }, n);
                }
            } else if shifted != 0
                && a.standalone == 0
                && unsafe { (*(a.vi as *mut ImodView)).fake_image } == 0
            {
                n.mv_image_toggle_plane(a, IMODV_DRAW_CX);
                n.mv_image_update(a);
            }
        }

        KEY_BRACE_LEFT | KEY_PAREN_LEFT => {
            if scattered_obj {
                input_prev_point(unsafe { &mut *(a.vi as *mut ImodView) }, n);
            }
        }

        KEY_BRACE_RIGHT | KEY_PAREN_RIGHT => {
            if scattered_obj {
                input_next_point(unsafe { &mut *(a.vi as *mut ImodView) }, n);
            }
        }

        // gooder (sic)
        KEY_G => {
            fastdraw =
                ((imod_ref.view[0].world & WORLD_QUALITY_BITS) >> WORLD_QUALITY_SHIFT) as i32;
            if shifted == 0 {
                fastdraw += 1;
            } else {
                fastdraw -= 1;
            }
            if fastdraw < 0 {
                fastdraw = 0;
            }
            if fastdraw > 3 {
                fastdraw = 3;
            }
            imod_print_stderr(&format!("Sphere draw quality {}\n", fastdraw + 1));
            imodv_register_model_chg();
            imodv_finish_chg_unit();
            imod_ref.view[0].world = (imod_ref.view[0].world & !WORLD_QUALITY_BITS)
                | ((fastdraw as u32) << WORLD_QUALITY_SHIFT);
            unsafe { imodv_draw() };
            imodv_objed_new_view(a);
        }

        KEY_MINUS => {
            imodv_zoomd(a, 0.95238095);
            unsafe { imodv_draw() };
        }

        KEY_UNDERSCORE => {
            imodv_zoomd(a, 0.5);
            unsafe { imodv_draw() };
        }

        KEY_EQUAL | KEY_PLUS => {
            if keypad != 0 || keysym == KEY_EQUAL {
                imodv_zoomd(a, 1.05);
            } else {
                imodv_zoomd(a, 2.0);
            }
            unsafe { imodv_draw() };
        }

        KEY_S => {
            if shifted != 0 {
                if ctrl != 0 {
                    n.set_2nd_snap_format();
                }
                n.imodv_auto_snapshot(a, "", SNAPSHOT_RGB);
                if ctrl != 0 {
                    n.restore_snap_format();
                }
            } else if ctrl != 0 {
                n.imodv_auto_snapshot(a, "", SNAPSHOT_TIF);
            } else {
                n.imodv_stereo_toggle(a);
            }
        }

        // '[' and ']' adjust stereo
        KEY_BRACKET_LEFT => {
            if ctrl != 0 {
                if scattered_obj {
                    input_prev_point(unsafe { &mut *(a.vi as *mut ImodView) }, n);
                }
            } else {
                a.plax -= 0.5f32;
                n.imodv_stereo_update(a);
                unsafe { imodv_draw() };
            }
        }

        KEY_BRACKET_RIGHT => {
            if ctrl != 0 {
                if scattered_obj {
                    input_next_point(unsafe { &mut *(a.vi as *mut ImodView) }, n);
                }
            } else {
                a.plax += 0.5f32;
                n.imodv_stereo_update(a);
                unsafe { imodv_draw() };
            }
        }

        KEY_L => {
            if ctrl != 0 && shifted != 0 {
                a.link_to_slicer = if a.link_to_slicer != 0 { 0 } else { 1 };
                n.imodv_control_link_update(a);
                imod_print_stderr(&format!(
                    "Linking to top slicer angles turned {}\n",
                    if a.link_to_slicer != 0 { "ON" } else { "OFF" }
                ));
            } else if shifted != 0 {
                n.imodv_object_list_dialog(a, 1);
            } else {
                imodv_objed_move_to_axis(a, 9);
            }
        }

        KEY_COMMA => n.imodv_control_change_steps(a, -1),

        KEY_PERIOD => n.imodv_control_change_steps(a, 1),

        KEY_M => {
            if shifted != 0 {
                n.imodv_model_edit_dialog(a, 1);
            } else if ctrl != 0 {
                n.imodv_objed_mesh_object(a);
            } else {
                n.mv_movie_dialog(a, 1);
            }
        }

        KEY_N => {
            if shifted != 0 {
                n.mv_movie_sequence_dialog(a, 1);
            }
        }

        KEY_V => {
            if shifted != 0 && ctrl != 0 && a.vert_buf_ok >= 0 {
                a.vert_buf_ok = 1 - a.vert_buf_ok;
                imod_print_stderr(&format!(
                    "Vertex buffers {}\n",
                    if a.vert_buf_ok != 0 { "ON" } else { "OFF" }
                ));
                if a.vert_buf_ok == 0 {
                    for m in 0..a.num_mods {
                        let model = a.mod_[m as usize];
                        n.vb_cleanup_vbd(model);
                    }
                }
                unsafe { imodv_draw() };
            } else if shifted != 0 {
                n.imodv_view_edit_dialog(a, 1);
            } else if ctrl != 0 {
                imodv_objed_draw_clip_plane(a, a.draw_clip == 0);
            }
        }

        KEY_1 => {
            if ctrl != 0 {
                imodv_objed_toggle_clip(a, 0, 0);
            } else {
                imodv_step_time(a, -1, n);
                unsafe { imodv_draw() };
            }
        }

        KEY_2 => {
            if ctrl != 0 {
                imodv_objed_toggle_clip(a, 0, 1);
            } else {
                imodv_step_time(a, 1, n);
                unsafe { imodv_draw() };
            }
        }

        KEY_8 => {
            if a.drawall != 0 {
                a.drawall = 0;
            } else {
                a.drawall = 3;
            }
            n.ime_set_view_data(a.drawall);
            unsafe { imodv_draw() };
        }

        KEY_9 => {
            if ctrl != 0 {
                if scattered_obj {
                    input_prev_point(unsafe { &mut *(a.vi as *mut ImodView) }, n);
                }
            } else {
                imodv_select_model(a, a.cur_mod - 1);
            }
        }

        KEY_0 => {
            if ctrl != 0 {
                if scattered_obj {
                    input_next_point(unsafe { &mut *(a.vi as *mut ImodView) }, n);
                }
            } else {
                imodv_select_model(a, a.cur_mod + 1);
            }
        }

        KEY_PAGE_DOWN => {
            if keypad != 0 {
                imodv_rotate_model(a, 0, 0, -a.delta_rot as i32, n);
            } else {
                imodv_translate_by_delta(a, 0, 0, tstep, n);
            }
        }
        KEY_PAGE_UP => {
            if keypad != 0 {
                imodv_rotate_model(a, 0, 0, a.delta_rot as i32, n);
            } else {
                imodv_translate_by_delta(a, 0, 0, -tstep, n);
            }
        }
        KEY_UP => {
            if keypad != 0 {
                imodv_rotate_model(a, -a.delta_rot as i32, 0, 0, n);
            } else {
                imodv_translate_by_delta(a, 0, -tstep, 0, n);
            }
        }
        KEY_DOWN => {
            if keypad != 0 {
                imodv_rotate_model(a, a.delta_rot as i32, 0, 0, n);
            } else {
                imodv_translate_by_delta(a, 0, tstep, 0, n);
            }
        }
        KEY_RIGHT => {
            if keypad != 0 {
                imodv_rotate_model(a, 0, a.delta_rot as i32, 0, n);
            } else {
                imodv_translate_by_delta(a, -tstep, 0, 0, n);
            }
        }
        KEY_LEFT => {
            if keypad != 0 {
                imodv_rotate_model(a, 0, -a.delta_rot as i32, 0, n);
            } else {
                imodv_translate_by_delta(a, tstep, 0, 0, n);
            }
        }
        KEY_5 | KEY_ENTER | KEY_U => {
            if shifted != 0 && keysym == KEY_U {
                if crate::imod::three_dmod::imodv::imodv_byte_images_exist() != 0 {
                    n.imodv_isosurface_edit_dialog(a, 1);
                }
            } else if !(keypad == 0 && keysym != KEY_U) {
                if ctrl != 0 && keysym == KEY_U {
                    imodv_rotate_model(a, 0, -a.delta_rot as i32, 0, n);
                } else {
                    n.imodv_control_start(a);
                }
            }
        }
        KEY_7 => imodv_rotate_model(a, -a.delta_rot as i32, 0, 0, n),
        KEY_J => imodv_rotate_model(a, a.delta_rot as i32, 0, 0, n),
        KEY_Y => {
            if ctrl != 0 {
                input_undo_redo(unsafe { &mut *(a.vi as *mut ImodView) }, true, n);
            } else if shifted != 0 {
                if a.standalone == 0 && unsafe { (*(a.vi as *mut ImodView)).fake_image } == 0 {
                    n.mv_image_toggle_plane(a, IMODV_DRAW_CY);
                    n.mv_image_update(a);
                }
            } else {
                imodv_rotate_model(a, 0, -a.delta_rot as i32, 0, n);
            }
        }
        KEY_I => {
            if shifted != 0 {
                if crate::imod::three_dmod::imodv::imodv_byte_images_exist() != 0 {
                    n.mv_image_edit_dialog(a, 1);
                }
            } else {
                imodv_rotate_model(a, 0, a.delta_rot as i32, 0, n);
            }
        }
        KEY_6 => imodv_rotate_model(a, 0, 0, -a.delta_rot as i32, n),
        KEY_H => imodv_rotate_model(a, 0, 0, a.delta_rot as i32, n),

        KEY_DELETE => {
            if keypad == 0
                && shifted != 0
                && ctrl != 0
                && a.standalone == 0
                && !obj.is_null()
                && imod_point_get(imod_ref).is_some()
            {
                if !scattered_obj {
                    wprint(
                        "\u{7}Object type must be scattered points to delete points in Model \
                         View\n",
                    );
                } else if imod_ref.mousemode != IMOD_MMODEL {
                    wprint("\u{7}You must be in Model Mode to delete points in Model View\n");
                } else {
                    input_delete_point(unsafe { &mut *(a.vi as *mut ImodView) }, n);
                }
            }
        }

        KEY_O => {
            if ctrl != 0 && shifted != 0 {
                imodv_objed_change_object(a, -1);
            } else if shifted != 0 {
                n.objed(a);
            } else {
                // output info
                if imod_ref.view[0].world & VIEW_WORLD_ON != 0 {
                    qstr += "Transformation matrix:";
                    tstep = 0;
                    while tstep < 16 {
                        if tstep % 4 == 0 {
                            qstr += "\n";
                        }
                        qstr2 = c_format(
                            "%7.3f ",
                            &[CArg::Dbl(sprintf_arg(
                                imod_ref.view[0].mat[tstep as usize] as f64,
                            ))],
                        );
                        qstr += &qstr2;
                        tstep += 1;
                    }
                    qstr += "\n";
                }
                qstr2 = c_format(
                    "Trans (x,y,z) = (%g, %g, %g)\n",
                    &[
                        CArg::Dbl(sprintf_arg(imod_ref.view[0].trans.x as f64)),
                        CArg::Dbl(sprintf_arg(imod_ref.view[0].trans.y as f64)),
                        CArg::Dbl(sprintf_arg(imod_ref.view[0].trans.z as f64)),
                    ],
                );
                qstr += &qstr2;
                qstr2 = c_format(
                    "Rotate (x,y,z) = (%g, %g, %g)\n",
                    &[
                        CArg::Dbl(sprintf_arg(imod_ref.view[0].rot.x as f64)),
                        CArg::Dbl(sprintf_arg(imod_ref.view[0].rot.y as f64)),
                        CArg::Dbl(sprintf_arg(imod_ref.view[0].rot.z as f64)),
                    ],
                );
                qstr += &qstr2;
                if a.movie_frames != 0 {
                    elapsed = (a.movie_current - a.movie_start) as f32 / 1000.0f32;
                    qstr2 = c_format(
                        "%d frames / %.3f sec = %.3f FPS\n",
                        &[
                            CArg::Int(a.movie_frames as i64),
                            CArg::Dbl(sprintf_arg(elapsed as f64)),
                            CArg::Dbl(sprintf_arg((a.movie_frames as f32 / elapsed) as f64)),
                        ],
                    );
                    qstr += &qstr2;
                }
                imod_print_info(&qstr);
            }
        }

        KEY_Z => {
            if ctrl != 0 {
                input_undo_redo(unsafe { &mut *(a.vi as *mut ImodView) }, false, n);
            } else if a.standalone == 0 && unsafe { (*(a.vi as *mut ImodView)).fake_image } == 0 {
                n.mv_image_toggle_plane(a, IMODV_DRAW_CZ);
                n.mv_image_update(a);
            }
        }

        KEY_R => {
            if ctrl != 0 {
                n.imodv_view_menu(a, VVIEW_MENU_LOWRES as i32);
                let lowres = a.lowres;
                n.imodv_menu_lowres(a, lowres);
            } else if shifted != 0 {
                n.open_rotation_tool(a);
            } else {
                imodv_objed_move_to_axis(a, 11);
            }
        }

        KEY_P => {
            if ctrl != 0 && shifted != 0 {
                imodv_objed_change_object(a, 1);
            } else if a.standalone == 0 {
                n.imodv_view_menu(a, VVIEW_MENU_CURPNT as i32);
            }
        }

        KEY_T => imodv_objed_move_to_axis(a, 0),

        KEY_F => {
            if ctrl != 0 {
                n.fine_grain_apply_last();
            } else {
                imodv_objed_move_to_axis(a, 1);
            }
        }

        KEY_K => {
            if ctrl != 0 && shifted != 0 {
                a.legacy_pick_mode = 1 - a.legacy_pick_mode;
                imod_print_stderr(&format!(
                    "{} method\n",
                    if a.legacy_pick_mode != 0 {
                        "Legacy selection"
                    } else {
                        "New picking"
                    }
                ));
            } else {
                imodv_objed_move_to_axis(a, 10);
            }
        }

        KEY_A => {
            // 8/29/06: Take accelerator for save as away so this will work
            if ctrl != 0 {
                let mut picked_ob = S_PICKED_OBJECT.get();
                let mut picked_co = S_PICKED_CONTOUR.get();
                n.imodv_select_visible_conts(a, &mut picked_ob, &mut picked_co);
                S_PICKED_OBJECT.set(picked_ob);
                S_PICKED_CONTOUR.set(picked_co);
                unsafe { imodv_draw() };
                if a.standalone == 0 {
                    n.imod_setxyzmouse();
                }
            } else if shifted == 0 && ctrl == 0 {
                a.plax *= -1.0f32;
                n.imodv_stereo_update(a);
                unsafe { imodv_draw() };
            }
        }

        KEY_D => {
            if shifted != 0
                && S_PICKED_CONTOUR.get() >= 0
                && ((imod_ref.cindex.object == S_PICKED_OBJECT.get()
                    && imod_ref.cindex.contour == S_PICKED_CONTOUR.get())
                    || n.imod_selection_list_query(
                        unsafe { &mut *a_ptr },
                        S_PICKED_OBJECT.get(),
                        S_PICKED_CONTOUR.get(),
                    ) > -2)
            {
                // This routine removes VBD's of all objects cont's removed from
                input_delete_contour(unsafe { &mut *(a.vi as *mut ImodView) }, n);
                S_PICKED_CONTOUR.set(-1);
            } else if shifted == 0 {
                let dbl_buf = a.dbl_buf;
                n.imodv_setbuffer(a, 1 - dbl_buf, -1, -1);
                let enabled = a.dbl_buf != 0
                    && a.trans_bkgd == 0
                    && (a.enable_depth_dbal >= 0 || a.enable_depth_dbst_al >= 0);
                n.set_enabled_menu_item(a, VVIEW_MENU_TRANSBKGD, enabled);
                unsafe { imodv_draw() };
            }
        }

        KEY_F11 | KEY_AMPERSAND => n.imodv_isosurface_invert_threshold(),

        KEY_3 | KEY_4 => {
            if ctrl != 0 {
                imodv_objed_toggle_clip(a, 11, keysym - KEY_3);
            }
        }

        KEY_F1 => imodv_objed_set_draw_type_and_style(a, 11),
        KEY_F2 => imodv_objed_set_draw_type_and_style(a, 22),
        KEY_F3 => imodv_objed_set_draw_type_and_style(a, 21),
        KEY_F4 => imodv_objed_set_draw_type_and_style(a, -1),

        // Grabs seem not to be needed and avoiding them saves a lot of trouble
        KEY_CONTROL => {
            S_CTRL_DOWN.set(INPUT_CTRL);
        }

        KEY_SHIFT => {
            S_SHIFT_DOWN.set(INPUT_SHIFT);
            if S_MID_DOWN.get() != 0 && a.draw_light == 0 {
                a.draw_light = 1;
                unsafe { imodv_draw() };
            }
        }

        _ => {}
    }
}

/// Original: `imodvAppLostFocus` (`mv_input.cpp:1023`).
///
/// When qApp loses focus (NULL new focused widget), clear the shift and ctrl
/// flags.
pub fn imodv_app_lost_focus() {
    S_CTRL_DOWN.set(0);
    S_SHIFT_DOWN.set(0);
    S_CONFIRM_MODIFIERS.set(1);
}

/// Original static: `confirmModifiers` (`mv_input.cpp:1031`).
///
/// For Mac, use modifiers unconditionally, other use only after losing focus.
pub fn confirm_modifiers(modifiers: u32, n: &mut dyn MvInputNativeBoundary) {
    // `#if !defined(Q_OS_MACX) || QT_VERSION < 0x060000` is the compiled arm
    // on this platform, so the early return is in force.
    if S_CONFIRM_MODIFIERS.get() == 0 {
        return;
    }
    S_CONFIRM_MODIFIERS.set(0);
    S_CTRL_DOWN.set(modifiers & INPUT_CTRL);
    S_SHIFT_DOWN.set(modifiers & INPUT_SHIFT);
}

/// Original: `imodvKeyRelease` (`mv_input.cpp:1042`).
pub fn imodv_key_release(a: &mut ImodvApp, event: InputEvent, n: &mut dyn MvInputNativeBoundary) {
    confirm_modifiers(event.modifiers, n);
    if event.key == KEY_CONTROL {
        S_CTRL_DOWN.set(0);
        if S_SHIFT_DOWN.get() == 0 {
            n.release_keyboard();
        }
    }
    if event.key == KEY_SHIFT {
        S_SHIFT_DOWN.set(0);
        if S_CTRL_DOWN.get() == 0 {
            n.release_keyboard();
        }
        if a.draw_light != 0 {
            a.draw_light = 0;
            unsafe { imodv_draw() };
        }
    }
}

/// Original: `imodvMousePress` (`mv_input.cpp:1061`).
///
/// Mouse press: start keeping track of movements.
pub fn imodv_mouse_press(a: &mut ImodvApp, event: InputEvent, n: &mut dyn MvInputNativeBoundary) {
    let shift = event.modifiers & INPUT_SHIFT;
    let key_event = InputKeyEvent {
        key: 0,
        modifiers: if event.modifiers & INPUT_CTRL != 0 {
            crate::imod::three_dmod::imod_input::INPUT_CTRL
        } else {
            0
        },
        accepted: false,
    };
    let ctrl = input_test_ctrl(&key_event);

    confirm_modifiers(event.modifiers, n);

    // Use state after in press and release to keep track of mouse state
    // `XY_PIXEL_TO_DEVICE` is `{ dx = px; dy = py; }` unless the Qt build
    // scales by the device pixel ratio (`imodP.h:333-341`).
    let (ex, ey) = (event.x, event.y);
    S_LEFT_DOWN.set(event.buttons & n.actual_modv_button(1));
    S_MID_DOWN.set(event.buttons & n.actual_modv_button(2));
    S_RIGHT_DOWN.set(event.buttons & n.actual_modv_button(3));
    // `utilRaiseIfNeeded` (`utilities.cpp:480`) has an empty body outside
    // `Q_OS_MACX`.

    // Set flag for any operation that needs to do something only on first move
    S_FIRST_MOVE.set(1);

    if event.button == n.actual_modv_button(1) {
        S_LEFT_DOWN.set(n.actual_modv_button(1));
        a.lastmx = ex;
        a.lastmy = ey;
        B2X.set(-10);
        B2Y.set(-10);
    /* DNM: why draw here? */
    /* imodvDraw(a); */
    } else if event.button == n.actual_modv_button(2)
        || (event.button == n.actual_modv_button(3) && shift != 0 && ctrl == 0)
    {
        a.lastmx = ex;
        B2X.set(ex);
        a.lastmy = ey;
        B2Y.set(ey);
        if event.button == n.actual_modv_button(2) && shift != 0 && ctrl == 0 {
            a.draw_light = 1;
            unsafe { imodv_draw() };
        }
        if event.button == n.actual_modv_button(2) && shift != 0 && ctrl != 0 {
            imodv_select(a, ex, ey, false, true, false, n);
        }
    } else if event.button == n.actual_modv_button(3) {
        imodv_select(a, ex, ey, false, false, shift != 0 && ctrl != 0, n);
    }
}

/// Original: `imodvMouseRelease` (`mv_input.cpp:1103`).
///
/// Mouse release: check for throwing.
pub fn imodv_mouse_release(a: &mut ImodvApp, event: InputEvent, n: &mut dyn MvInputNativeBoundary) {
    let right_was_down = event.button & n.actual_modv_button(3);
    let shift = event.modifiers & INPUT_SHIFT;
    let key_event = InputKeyEvent {
        key: 0,
        modifiers: if event.modifiers & INPUT_CTRL != 0 {
            crate::imod::three_dmod::imod_input::INPUT_CTRL
        } else {
            0
        },
        accepted: false,
    };
    let ctrl = input_test_ctrl(&key_event);

    confirm_modifiers(event.modifiers, n);
    let (ex, ey) = (event.x, event.y);
    S_LEFT_DOWN.set(event.buttons & n.actual_modv_button(1));
    S_MID_DOWN.set(event.buttons & n.actual_modv_button(2));
    S_RIGHT_DOWN.set(event.buttons & n.actual_modv_button(3));
    if ((event.button & n.actual_modv_button(2)) != 0 && shift == 0 && ctrl == 0)
        || (right_was_down != 0 && shift != 0)
    {
        imodv_rotate(a, ex, ey, 1, right_was_down as i32, n);
    }
    if a.draw_light != 0 {
        a.draw_light = 0;
        unsafe { imodv_draw() };
    }
}

/// Original: `imodvMouseMove` (`mv_input.cpp:1124`).
///
/// Mouse movement with button down.
pub fn imodv_mouse_move(a: &mut ImodvApp, event: InputEvent, n: &mut dyn MvInputNativeBoundary) {
    // `XY_PIXEL_TO_DEVICE` is `{ dx = px; dy = py; }` outside the scaled Qt
    // build (`imodP.h:341`).
    MOVE_EX.set(event.x);
    MOVE_EY.set(event.y);
    MOVE_SHIFT.set(event.modifiers & INPUT_SHIFT);
    let key_event = InputKeyEvent {
        key: 0,
        modifiers: if event.modifiers & INPUT_CTRL != 0 {
            crate::imod::three_dmod::imod_input::INPUT_CTRL
        } else {
            0
        },
        accepted: false,
    };
    MOVE_CTRL.set(input_test_ctrl(&key_event));
    confirm_modifiers(event.modifiers, n);

    // Use state in mouse move to keep track of button down
    S_LEFT_DOWN.set(event.buttons & n.actual_modv_button(1));
    S_MID_DOWN.set(event.buttons & n.actual_modv_button(2));
    S_RIGHT_DOWN.set(event.buttons & n.actual_modv_button(3));
    if imod_debug('m') {
        imod_print_stderr(&format!("Move ex,y {} {} ", MOVE_EX.get(), MOVE_EY.get()));
    }

    // Return after recording values if processing events, or process events
    // to stay up to date
    if S_PROCESSING.get() {
        return;
    }
    S_PROCESSING.set(true);
    n.imod_info_input();

    // Now save the values being used to set into lastmx/y at end, in case more events come
    // in during the draw
    let save_x = MOVE_EX.get();
    let save_y = MOVE_EY.get();

    if S_LEFT_DOWN.get() != 0 {
        imodv_translate_by_delta(
            a,
            -(MOVE_EX.get() - a.lastmx),
            MOVE_EY.get() - a.lastmy,
            0,
            n,
        );
    }
    if S_MID_DOWN.get() != 0 && MOVE_SHIFT.get() != 0 && MOVE_CTRL.get() == 0 {
        imodv_light_move(a, MOVE_EX.get(), MOVE_EY.get(), n);
    } else if (S_MID_DOWN.get() != 0 && MOVE_SHIFT.get() == 0)
        || (S_RIGHT_DOWN.get() != 0 && MOVE_SHIFT.get() != 0)
    {
        imodv_rotate(
            a,
            MOVE_EX.get(),
            MOVE_EY.get(),
            0,
            S_RIGHT_DOWN.get() as i32,
            n,
        );
    } else if S_RIGHT_DOWN.get() != 0 && MOVE_CTRL.get() != 0 {
        imodv_select(a, MOVE_EX.get(), MOVE_EY.get(), true, false, false, n);
    }
    a.lastmx = save_x;
    a.lastmy = save_y;
    if imod_debug('m') {
        imod_puts(" ");
    }
    S_PROCESSING.set(false);
}

/// Original: `imodvScrollWheel` (`mv_input.cpp:1177`).
///
/// A mouse wheel event either zooms or scales a scattered point size.
pub fn imodv_scroll_wheel(a: &mut ImodvApp, event: InputEvent, n: &mut dyn MvInputNativeBoundary) {
    let delta = event.delta;
    let power = -delta as f64 / 120.;
    let zoom = 1.05f64.powf(power);
    let scrn_scale: f32;
    let mut size: f32;
    let mut ob = 0;
    let mut co = 0;
    let mut pt = 0;
    let imod = a.imod;
    let key_event = InputKeyEvent {
        key: 0,
        modifiers: if event.modifiers & INPUT_CTRL != 0 {
            crate::imod::three_dmod::imod_input::INPUT_CTRL
        } else {
            0
        },
        accepted: false,
    };
    if (event.modifiers & INPUT_SHIFT) != 0 && input_test_ctrl(&key_event) != 0 {
        let Some(imod) = (unsafe { imod.as_mut() }) else {
            return;
        };
        imod_get_index(imod, &mut ob, &mut co, &mut pt);
        if pt < 0
            || ob < 0
            || ob as usize >= imod.obj.len()
            || iobj_scat(imod.obj[ob as usize].flags) == 0
        {
            return;
        }
        let (obj_index, cont_index, point_index) = (ob as usize, co as usize, pt);
        size = imod_point_get_size(
            &imod.obj[obj_index],
            &imod.obj[obj_index].cont[cont_index],
            point_index,
        );
        scrn_scale = (0.5 * a.winx.min(a.winy) as f64 / imod.view[0].rad as f64) as f32;
        size += delta as f32 * util_wheel_to_point_size_scaling(scrn_scale);
        size = if 0. > size { 0. } else { size };
        n.undo_contour_data_chg(a);
        let imod = unsafe { &mut *a.imod };
        imod_point_set_size(&mut imod.obj[obj_index].cont[cont_index], point_index, size);
        imodv_finish_chg_unit();
        imodv_draw_imod_images(0);
        unsafe { imodv_draw() };
    } else {
        imodv_zoomd(a, zoom);
        unsafe { imodv_draw() };
    }
}

/// Original static: `imodv_light_move` (`mv_input.cpp:1227`).
///
/// Move the light.
pub fn imodv_light_move(a: &mut ImodvApp, mx: i32, my: i32, n: &mut dyn MvInputNativeBoundary) {
    let a_ptr = a as *mut ImodvApp;
    let mut mxp = 0;
    let mut myp = 0;
    let maskr = imodv_query_pointer(a, &mut mxp, &mut myp, n);

    if (maskr & n.actual_modv_button(2)) != 0 && (maskr & INPUT_SHIFT) != 0 {
        if S_FIRST_MOVE.get() != 0 {
            imodv_register_model_chg();
            imodv_finish_chg_unit();
            S_FIRST_MOVE.set(0);
        }

        // 4/3/07: remove factor of 10 so sensitivity can be less
        let (dx, dy) = (mx - a.lastmx, my - a.lastmy);
        n.light_moveby(unsafe { &mut *a_ptr }, dx, dy);
    }
    unsafe { imodv_draw() };
}

/// Original: `imodv_zoomd` (`mv_input.cpp:1247`).
///
/// Change zoom by a factor.
pub fn imodv_zoomd(a: &mut ImodvApp, zoom: f64) {
    if a.imod.is_null() {
        return;
    }

    if a.crosset != 0 {
        for m in 0..a.num_mods {
            if let Some(model) = unsafe { a.mod_[m as usize].as_mut() } {
                let rad = model.view[0].rad;
                model.view[0].rad = (rad as f64 / zoom) as f32;
            }
        }
    } else {
        let model = unsafe { &mut *a.imod };
        let rad = model.view[0].rad;
        model.view[0].rad = (rad as f64 / zoom) as f32;
    }
}

/// Original static: `registerClipPlaneChg` (`mv_input.cpp:1261`).
///
/// Register a clip plane change for model or object on the first move.
pub fn register_clip_plane_chg(a: &mut ImodvApp, n: &mut dyn MvInputNativeBoundary) {
    let a_ptr = a as *mut ImodvApp;
    if S_FIRST_MOVE.get() != 0 {
        // `mv_input.cpp:864` `a->imod->editGlobalClip`.
        let edit_global_clip = unsafe { (*a.imod).edit_global_clip };
        if edit_global_clip != 0 {
            imodv_register_model_chg();
        } else {
            objed_object(unsafe { &mut *a_ptr });
            imodv_register_object_chg(a.obj_num);
        }
        imodv_finish_chg_unit();
        S_FIRST_MOVE.set(0);
    }
}

/// Original static: `imodvTranslateByDelta` (`mv_input.cpp:1277`).
///
/// Translate model or clipping plane or current point.
pub fn imodv_translate_by_delta(
    a: &mut ImodvApp,
    x: i32,
    y: i32,
    z: i32,
    n: &mut dyn MvInputNativeBoundary,
) {
    let a_ptr = a as *mut ImodvApp;
    let mut mx = 0;
    let mut my = 0;
    let maskr = imodv_query_pointer(a, &mut mx, &mut my, n);
    let ctrl = (maskr & INPUT_CTRL) != 0;
    let shift = (maskr & INPUT_SHIFT) != 0;

    let Some(mat) = a.mat.as_mut().map(|m| m as *mut Imat) else {
        return;
    };
    let mat = unsafe { &mut *mat };
    let mut ipt = Ipoint::default();
    let mut opt = Ipoint::default();
    let mut spt = Ipoint::default();
    let mut alpha = 0.;
    let mut beta = 0.;

    let (mstrt, mend) = if ctrl || a.moveall == 0 {
        (a.cur_mod, a.cur_mod + 1)
    } else {
        (0, a.num_mods)
    };

    /* DNM: changed to compute shift properly for each model, to take account
    of actual scale to window, and to shift by mouse move amount */
    for m in mstrt..mend {
        let Some(imod) = (unsafe {
            a.mod_
                .get(m as usize)
                .copied()
                .unwrap_or(std::ptr::null_mut())
                .as_mut()
        }) else {
            continue;
        };
        imodv_rot_scale_matrix(a, mat, imod);

        ipt.x = x as f32;
        ipt.y = y as f32;
        ipt.z = z as f32;
        imod_mat_transform(mat, &ipt, &mut opt);

        if ctrl && !shift {
            let obj = objed_object(unsafe { &mut *a_ptr })
                .map_or(std::ptr::null_mut(), |o| o as *mut Iobj);
            if !obj.is_null() {
                register_clip_plane_chg(unsafe { &mut *a_ptr }, n);
                // `mv_input.cpp:915` `a->imod->editGlobalClip`.
                let edit_global_clip = unsafe { (*a.imod).edit_global_clip };
                let clips = if edit_global_clip != 0 {
                    unsafe { &mut (&mut (*a.imod).view)[0].clips }
                } else {
                    unsafe { &mut (*obj).clips }
                };
                let mut ipst = clips.plane as i32;
                let mut ipnd = clips.plane as i32;
                if imod.view[0].world & WORLD_MOVE_ALL_CLIP != 0 {
                    ipst = 0;
                    ipnd = clips.count as i32 - 1;
                }
                for ip in ipst..=ipnd {
                    if clips.flags & (1 << ip) != 0 {
                        let ip = ip as usize;
                        clips.point[ip].x += opt.x;
                        clips.point[ip].y += opt.y;
                        clips.point[ip].z += opt.z;
                        let (point, normal) = (clips.point[ip], clips.normal[ip]);
                        clip_center_and_angles(
                            unsafe { &*a_ptr },
                            &point,
                            &normal,
                            &mut spt,
                            &mut alpha,
                            &mut beta,
                        );
                        clips.point[ip].x = -spt.x;
                        clips.point[ip].y = -spt.y;
                        clips.point[ip].z = -spt.z;
                    }
                }
            }
        } else if ctrl && shift {
            let mut ob = 0;
            let mut co = 0;
            let mut pt = 0;
            imod_get_index(imod, &mut ob, &mut co, &mut pt);
            if pt < 0
                || ob < 0
                || ob as usize >= imod.obj.len()
                || iobj_scat(imod.obj[ob as usize].flags) == 0
            {
                return;
            }
            if S_FIRST_MOVE.get() != 0 {
                n.undo_contour_data_chg(unsafe { &mut *a_ptr });
                imodv_finish_chg_unit();
                S_FIRST_MOVE.set(0);
            }
            let point = &mut imod.obj[ob as usize].cont[co as usize].pts[pt as usize];
            point.x -= opt.x;
            point.y -= opt.y;
            point.z -= opt.z;
            imodv_draw_imod_images(0);
        } else {
            imod.view[0].trans.x -= opt.x;
            imod.view[0].trans.y -= opt.y;
            imod.view[0].trans.z -= opt.z;
        }
    }

    unsafe { imodv_draw() };
}

/// Original: `imodv_rotate_model` (`mv_input.cpp:1358`).
///
/// Rotate the model by the actual angles x, y and z.  The angles are in 0.1
/// degree increments.  DNM: made this work properly.
pub fn imodv_rotate_model(
    a: &mut ImodvApp,
    x: i32,
    y: i32,
    z: i32,
    n: &mut dyn MvInputNativeBoundary,
) {
    /* IF movieing, save the current increments as ones to movie on */
    if a.movie != 0 {
        a.xrot_movie = x as f32;
        a.yrot_movie = y as f32;
        a.zrot_movie = z as f32;
    }
    imodv_compute_rotation(a, x as f32, y as f32, z as f32, n);
    unsafe { imodv_draw() };
}

/// Original static: `imodv_compute_rotation` (`mv_input.cpp:1369`).
pub fn imodv_compute_rotation(
    a: &mut ImodvApp,
    x: f32,
    y: f32,
    z: f32,
    n: &mut dyn MvInputNativeBoundary,
) {
    let a_ptr = a as *mut ImodvApp;
    let mut mx = 0;
    let mut my = 0;
    let maskr = imodv_query_pointer(a, &mut mx, &mut my, n);
    let mut alpha = 0.;
    let mut beta = 0.;
    let mut gamma = 0.;
    let Some(mat) = a.mat.as_mut().map(|m| m as *mut Imat) else {
        return;
    };
    let mat = unsafe { &mut *mat };
    let mut normal = Ipoint::default();
    let mut scale_point = Ipoint::default();
    let mut imod = a.imod;

    /* IF movieing, start the movie if necessary */
    if a.movie != 0 && a.wpid == 0 {
        a.throw_factor = 1.;
        imodv_start_movie(a, n);
        /*  return; */
    }

    let Some(mut mato) = imod_mat_new(3) else {
        return;
    };
    let Some(mut matp) = imod_mat_new(3) else {
        return;
    };

    imodv_resolve_rotation(mat, 0.1f32 * x, 0.1f32 * y, 0.1f32 * z);

    if (maskr & INPUT_CTRL) == 0 || a.movie != 0 {
        /* Regular rotation of one or all models */

        let (mstrt, mend) = if a.moveall == 0 {
            (a.cur_mod, a.cur_mod + 1)
        } else {
            (0, a.num_mods)
        };

        for m in mstrt..mend {
            imod = a
                .mod_
                .get(m as usize)
                .copied()
                .unwrap_or(std::ptr::null_mut());
            let Some(model) = (unsafe { imod.as_mut() }) else {
                continue;
            };

            /* Compute current rotation matrix */
            imod_mat_id(&mut mato);
            imod_mat_rot(&mut mato, model.view[0].rot.z as f64, B3D_Z);
            imod_mat_rot(&mut mato, model.view[0].rot.y as f64, B3D_Y);
            imod_mat_rot(&mut mato, model.view[0].rot.x as f64, B3D_X);

            /* Multiply by the new rotation, then get back to 3 angles */
            imod_mat_mult(&mato, mat, &mut matp);
            imod_mat_get_nat_angles(&matp, &mut alpha, &mut beta, &mut gamma);
            model.view[0].rot.x = alpha as f32;
            model.view[0].rot.y = beta as f32;
            model.view[0].rot.z = gamma as f32;
        }
        if let Some(model) = unsafe { imod.as_ref() } {
            let rot = model.view[0].rot;
            imodv_new_model_angles(&rot);
        }
    } else {
        let obj =
            objed_object(unsafe { &mut *a_ptr }).map_or(std::ptr::null_mut(), |o| o as *mut Iobj);
        // `mv_input.cpp:1031` `imod->editGlobalClip`.
        let edit_global_clip = unsafe { (*imod).edit_global_clip };
        let Some(model) = (unsafe { imod.as_mut() }) else {
            return;
        };
        if edit_global_clip != 0 || !obj.is_null() {
            // `clips` is the source's `IclipPlanes *`; it aliases the model
            // the loop below keeps reading, exactly as in the source.
            let clips: *mut crate::imod::libimod::imodel::Iclip_planes = if edit_global_clip != 0 {
                &mut model.view[0].clips
            } else {
                unsafe { &mut (*obj).clips }
            };
            let clips = unsafe { &mut *clips };
            let mut ipst = clips.plane as i32;
            let mut ipnd = clips.plane as i32;
            if model.view[0].world & WORLD_MOVE_ALL_CLIP != 0 {
                ipst = 0;
                ipnd = clips.count as i32 - 1;
            }
            for ip in ipst..=ipnd {
                if clips.flags & (1 << ip) != 0 {
                    let ip = ip as usize;

                    /* Clipping plane rotation: apply to current model only */

                    register_clip_plane_chg(unsafe { &mut *a_ptr }, n);

                    /* Find the normal in scaled model coordinates by scaling
                    each of the components appropriately */
                    let view_scale = model.view[0].scale;
                    let zscale = model.zscale;
                    scale_point.x = clips.normal[ip].x / view_scale.x;
                    scale_point.y = clips.normal[ip].y / view_scale.y;
                    scale_point.z = clips.normal[ip].z / (view_scale.z * zscale);

                    /* get current rotation transform into viewing space */
                    let rot = model.view[0].rot;
                    imod_mat_id(&mut mato);
                    imod_mat_rot(&mut mato, rot.z as f64, B3D_Z);
                    imod_mat_rot(&mut mato, rot.y as f64, B3D_Y);
                    imod_mat_rot(&mut mato, rot.x as f64, B3D_X);

                    /* Get product of that with screen-oriented rotation */
                    imod_mat_mult(&mato, mat, &mut matp);
                    imod_mat_transform(&matp, &scale_point, &mut normal);

                    /* Back-transform normal by inverse of current transform */

                    imod_mat_id(&mut mato);
                    imod_mat_rot(&mut mato, -(rot.x as f64), B3D_X);
                    imod_mat_rot(&mut mato, -(rot.y as f64), B3D_Y);
                    imod_mat_rot(&mut mato, -(rot.z as f64), B3D_Z);
                    imod_mat_transform(&mato, &normal, &mut scale_point);

                    /* Rescale components to get back to unscaled model normal */
                    clips.normal[ip].x = scale_point.x * view_scale.x;
                    clips.normal[ip].y = scale_point.y * view_scale.y;
                    clips.normal[ip].z = scale_point.z * (view_scale.z * zscale);
                    imod_point_normalize(&mut clips.normal[ip]);

                    // Reset the fixed point to point nearest center of field
                    let (point, plane_normal) = (clips.point[ip], clips.normal[ip]);
                    clip_center_and_angles(
                        unsafe { &*a_ptr },
                        &point,
                        &plane_normal,
                        &mut normal,
                        &mut alpha,
                        &mut beta,
                    );
                    clips.point[ip].x = -normal.x;
                    clips.point[ip].y = -normal.y;
                    clips.point[ip].z = -normal.z;
                }
            }
        }
    }
}

/// Original: `imodvResolveRotation` (`mv_input.cpp:1489`).
///
/// Compute a matrix that resolves X and Y rotations into rotation about a
/// single axis if both are present, and/or rotates about Z.
pub fn imodv_resolve_rotation(mat: &mut Imat, x: f32, y: f32, z: f32) {
    let gamrad = (y as f64).atan2(x as f64);
    let gamma = gamrad / 0.017453293;
    let alpha = x as f64 * (-gamrad).cos() - y as f64 * (-gamrad).sin();

    imod_mat_id(mat);
    imod_mat_rot(mat, -gamma, B3D_Z);
    imod_mat_rot(mat, alpha, B3D_X);
    imod_mat_rot(mat, gamma + z as f64, B3D_Z);
}

/// Original: `imodvRotScaleMatrix` (`mv_input.cpp:1504`).
///
/// Compute the current rotation and scaling matrix to apply to a screen shift
/// and get a model.
pub fn imodv_rot_scale_matrix(a: &ImodvApp, mat: &mut Imat, imod: &Imod) {
    let scrnscale: f32;
    let mut spt = Ipoint::default();
    let view = &imod.view[0];
    imod_mat_id(mat);
    imod_mat_rot(mat, -(view.rot.x as f64), B3D_X);
    imod_mat_rot(mat, -(view.rot.y as f64), B3D_Y);
    imod_mat_rot(mat, -(view.rot.z as f64), B3D_Z);

    scrnscale = (0.5 * a.winx.min(a.winy) as f64 / view.rad as f64) as f32;

    /* 11/17/18: add the division by the scale.xyz because it was done after all 3 calls
    to this.  */
    spt.x = 1.0f32 / (scrnscale * view.scale.x);
    spt.y = 1.0f32 / (scrnscale * view.scale.y);
    spt.z = 1.0f32 / (scrnscale * view.scale.z * imod.zscale);
    if view.world & VIEW_WORLD_INVERT_Z != 0 {
        spt.z = -spt.z;
    }
    imod_mat_scale(mat, &spt);
}

/// Original static: `imodv_rotate` (`mv_input.cpp:1167`).
///
/// Rotate the model or clipping planes by the amount of cursor movement.
pub fn imodv_rotate(
    a: &mut ImodvApp,
    mx: i32,
    my: i32,
    throw_flag: i32,
    right_was_down: i32,
    n: &mut dyn MvInputNativeBoundary,
) {
    let mut mxp = 0;
    let mut myp = 0;
    let mut idx = 0;
    let mut idy = 0;
    let mut idz = 0;
    let maskr = imodv_query_pointer(a, &mut mxp, &mut myp, n);
    let (dx, dy): (f32, f32);
    let angle_scale: f32;

    /* If movie on and not a Control rotation, then check the throw flag */

    if a.movie != 0 && (maskr & INPUT_CTRL) == 0 {
        if throw_flag != 0 {
            /* If throwing at end of movement, then turn off movie if
            movement is too small, otherwise set rotations to the total
            movement since the button was pressed */

            dx = (mx - B2X.get()) as f32;
            dy = (my - B2Y.get()) as f32;
            if dx * dx + dy * dy < MIN_SQUARE_TO_THROW as f32 {
                a.xrot_movie = 0.;
                a.yrot_movie = 0.;
                a.zrot_movie = 0.;
                a.movie = 0;
                return;
            }

            // Got to figure out which button it was!
            if right_was_down != 0 {
                idz = (10.
                    * util_mouse_zaxis_rotation(a.winx, mx, B2X.get(), a.winy, my, B2Y.get())
                        as f64
                    + 0.5)
                    .floor() as i32;
            } else {
                idx = ((MOUSE_TO_THROW * dy) as f64 + 0.5).floor() as i32;
                idy = ((MOUSE_TO_THROW * dx) as f64 + 0.5).floor() as i32;
            }
            if idx == 0 && idy == 0 && idz == 0 {
                a.movie = 0;
            }
            a.xrot_movie = idx as f32;
            a.yrot_movie = idy as f32;
            a.zrot_movie = idz as f32;

            a.throw_factor = (((dx * dx + dy * dy) as f64).sqrt() / SAME_SPEED_DISTANCE) as f32;
            /* Start movie if it is not already going */
            if a.movie != 0 && a.wpid == 0 {
                imodv_start_movie(a, n);
            }
        }
        return;
    }

    /* If the mouse button has been released, don't rotate. */
    if (maskr & (n.actual_modv_button(2) | n.actual_modv_button(3))) == 0 {
        return;
    }

    /* Turn off movie for all rotation axis. DNM add movie flag too */
    a.xrot_movie = 0.;
    a.yrot_movie = 0.;
    a.zrot_movie = 0.;
    a.movie = 0;

    if S_MID_DOWN.get() != 0 {
        /* Get the total x and y movement.  The scale factor will roll the surface
        of a sphere 0.8 times the size of window's smaller dimension at the
        same rate as the mouse */
        dx = (mx - a.lastmx) as f32;
        dy = (my - a.lastmy) as f32;
        angle_scale = (1800. / (3.142 * 0.4 * a.winx.min(a.winy) as f64)) as f32;
        idx = ((angle_scale * dy) as f64 + 0.5).floor() as i32;
        idy = ((angle_scale * dx) as f64 + 0.5).floor() as i32;
    } else {
        idz = (10. * util_mouse_zaxis_rotation(a.winx, mx, a.lastmx, a.winy, my, a.lastmy) as f64
            + 0.5)
            .floor() as i32;
    }
    if idx == 0 && idy == 0 && idz == 0 {
        return;
    }

    if imod_debug('m') {
        imod_print_stderr(&format!("mx,y {mx} {my}  lmx,y {} {}", a.lastmx, a.lastmy));
    }
    imodv_rotate_model(a, idx, idy, idz, n);

    /* This is uneeded, since the rotate_model has a draw */
    /* imodvDraw(a); */
}

/// Original: `clipCenterAndAngles` (`mv_input.cpp:1533`).
///
/// Compute the point on a clipping plane that is closest to the center of the
/// window view center, and the angles of rotation for the normal.
pub fn clip_center_and_angles(
    a: &ImodvApp,
    clip_point: &Ipoint,
    clip_normal: &Ipoint,
    cen: &mut Ipoint,
    alpha: &mut f64,
    beta: &mut f64,
) {
    let small_val: f32 = 1.0e-4;
    let zrot: f64;
    let Some(imod) = (unsafe { a.imod.as_ref() }) else {
        return;
    };
    let vw = &imod.view[0];
    let tt: f32;
    let zscale: f32 = if imod.zscale != 0. { imod.zscale } else { 1. };

    let mut normal = *clip_normal;
    normal.z /= zscale;
    imod_point_normalize(&mut normal);
    let mut trans = vw.trans;
    trans.z *= zscale;
    let mut point = *clip_point;
    point.z *= zscale;
    *alpha = 0.;
    *beta = 0.;
    if (normal.x as f64).abs() > small_val as f64 || (normal.z as f64).abs() > small_val as f64 {
        *beta = -(normal.x as f64).atan2(normal.z as f64);
    }
    zrot = normal.z as f64 * beta.cos() - normal.x as f64 * beta.sin();
    *alpha = -(zrot.atan2(normal.y as f64) - 1.570796);

    // Get a center point: point on plane closest to the center of the display
    tt = imod_point_dot(&normal, &trans) - imod_point_dot(&normal, &point);
    cen.x = normal.x * tt - trans.x;
    cen.y = normal.y * tt - trans.y;
    cen.z = normal.z * tt - trans.z;
    cen.z /= zscale;
}

/// Original static: `imodvSelect` (`mv_input.cpp:1362` of the picking block).
///
/// For select mode, set up for picking then call draw routine.
pub fn imodv_select(
    a: &mut ImodvApp,
    x: i32,
    y: i32,
    moving: bool,
    insert: bool,
    cur_obj: bool,
    n: &mut dyn MvInputNativeBoundary,
) {
    let a_ptr = a as *mut ImodvApp;
    // 5/29/08: This was static, but why?  It stays in scope while needed.
    let mut buf = vec![0u32; SELECT_BUFSIZE];
    let hits: i32;
    let obj =
        imod_object_get(unsafe { a.imod.as_ref() }).map_or(std::ptr::null(), |o| o as *const Iobj);
    let mut mo_num = 0;
    let mut ob_num = 0;
    let mut co_num = 0;
    let mut pt_num = 0;

    if insert {
        if a.standalone != 0 {
            return;
        }
        if obj.is_null() || iobj_scat(unsafe { (*obj).flags }) == 0 {
            wprint("\u{7}Object type must be scattered points to add points in Model View\n");
            return;
        }
        if unsafe { (*a.imod).mousemode } != IMOD_MMODEL {
            wprint("\u{7}You must be in Model Mode to add points in Model View\n");
            return;
        }
    }

    a.x_pick = x;
    a.y_pick = (a.winy - 1) - y;
    n.imodv_winset(unsafe { &mut *a_ptr });

    if a.stereo == 0 && a.legacy_pick_mode == 0 {
        // New picking mode: set up a slightly special draw then call routine ot find item
        let device_pixel_ratio =
            unsafe { a.main_win.as_ref() }.map_or(1., |w| w.device_pixel_ratio);
        a.w_pick = 2 * ((2.0f64.min(device_pixel_ratio as f64) * 5.) + 0.5).floor() as i32 + 1;
        if a.winy < a.w_pick || a.winx < a.w_pick {
            return;
        }
        if a.mod_picks.len() < a.num_mods as usize {
            if a.mod_picks
                .try_reserve_exact(a.num_mods as usize - a.mod_picks.len())
                .is_err()
            {
                return;
            }
            a.mod_picks.resize(a.num_mods as usize, Ipoint::default());
        }
        a.read_pix_for_pick = 1;
        unsafe { imodv_draw() };

        // `App->newQtOpenGL` is set on this build; the host performs
        // `imodvModelDrawRange` plus `imodvUnprojectPickedPoint`.
        n.imodv_unproject_picked_point(unsafe { &mut *a_ptr });
        n.find_clicked_drawn_element(
            unsafe { &mut *a_ptr },
            cur_obj,
            &mut mo_num,
            &mut ob_num,
            &mut co_num,
            &mut pt_num,
        );
    } else {
        // Legacy selection mode, stick with a 10x10 area, do special draw and process it
        n.gl_select_buffer(unsafe { &mut *a_ptr }, &mut buf);

        // Defer entering selection mode until inside the paint routine and context
        // is already set.  This avoid context-setting errors on some systems

        a.w_pick = 10;
        a.h_pick = 10;
        a.do_pick = 1;

        unsafe { imodv_draw() };

        a.do_pick = 0;
        hits = a.pick_hits;
        process_hits(
            a,
            hits,
            &buf,
            cur_obj,
            &mut mo_num,
            &mut ob_num,
            &mut co_num,
            &mut pt_num,
        );
    }

    // Either way, process the results
    process_selection(a, moving, insert, mo_num, ob_num, co_num, pt_num, n);
}

/// Original static: `processHits` (`mv_input.cpp:1477` of the picking block).
///
/// Go through the list of items hitting the pick box and find the nearest.
pub fn process_hits(
    a: &mut ImodvApp,
    mut hits: i32,
    buffer: &[u32],
    cur_obj: bool,
    mo_num: &mut i32,
    ob_num: &mut i32,
    co_num: &mut i32,
    pt_num: &mut i32,
) {
    let mut names: u32;
    // `z1`, `z2`, `zav`, `zmin` and `tmo`..`tpt` are uninitialised locals in
    // the source; `zmin` is only read once `ptNum` has been set.
    let mut zmin: u32 = 0;
    let mut tmo: i32 = 0;
    let mut tob: i32 = 0;
    let mut tco: i32 = 0;
    let mut tpt: i32 = 0;
    let obsave: i32;

    let Some(imod) = (unsafe { a.imod.as_ref() }) else {
        return;
    };
    imod_get_index(imod, ob_num, co_num, pt_num);
    obsave = *ob_num;
    *co_num = -1;

    if hits == 0 {
        return;
    }

    /* If it overflowed, process what's there */
    if hits < 0 {
        hits = SELECT_BUFSIZE as i32 / 3;
    }

    let mut ptr: usize = 0;
    *pt_num = -1;

    for _i in 0..hits {
        /* for each hit */
        if ptr >= SELECT_BUFSIZE - 7 {
            break;
        }
        names = buffer[ptr];
        ptr += 1;
        if ptr as u32 + names + 2 > SELECT_BUFSIZE as u32 {
            break;
        }

        let z1 = buffer[ptr];
        ptr += 1;
        let z2 = buffer[ptr];
        ptr += 1;
        let zav = z1 / 2 + z2 / 2;

        if imod_debug('p') {
            imod_print_stderr(&format!(" # names = {names}"));
            imod_print_stderr(&format!(";  z1 = {z1};"));
            imod_print_stderr(&format!(" z2 = {z2}; "));
            imod_print_stderr("   names are ");
        }

        for j in 0..names {
            /*  for each name */
            match j {
                0 => tmo = buffer[ptr] as i32,
                1 => tob = buffer[ptr] as i32,
                2 => tco = buffer[ptr] as i32,
                3 => tpt = buffer[ptr] as i32,
                _ => {}
            }

            if imod_debug('p') {
                imod_print_stderr(&format!("{} ", buffer[ptr]));
            }
            ptr += 1;
        }

        /* If it was a good hit (4 names) and its in front of any previous, take it */
        if names > 3 && (*pt_num == -1 || zav <= zmin) && (!cur_obj || tob == obsave) {
            zmin = zav;
            *mo_num = tmo;
            *ob_num = tob;
            *co_num = tco;
            *pt_num = tpt;
            if imod_debug('p') {
                imod_print_stderr(" *");
            }
        }
        if imod_debug('p') {
            imod_print_stderr("\n");
            imod_print_stderr(&format!("   zav = {zav}; zmin = {zmin}\n"));
        }
    }

    if imod_debug('p') {
        imod_print_stderr(&format!("hits = {hits}\n"));
    }
}

/// Original static: `processSelection` (`mv_input.cpp:1560` of the picking
/// block).
///
/// Analyze the result (the num values) from processing select mode or the
/// unprojection.
pub fn process_selection(
    a: &mut ImodvApp,
    moving: bool,
    insert: bool,
    mo_num: i32,
    ob_num: i32,
    mut co_num: i32,
    mut pt_num: i32,
    n: &mut dyn MvInputNativeBoundary,
) {
    let a_ptr = a as *mut ImodvApp;
    let mut ob_num = ob_num;
    let mut minco = 0;
    let mut cosave = 0;
    let mut obsave = 0;
    let mut minpt = 0;
    let mut tpt = 0;
    let mut tob;
    let mut tco;
    let ind_save: Iindex;
    let mut pickpt;
    let obj: *mut Iobj;
    let mut minsq: f32;
    let mut dsqr: f32;
    let (mut dx, mut dy, mut dz): (f32, f32, f32);
    // `#if defined(Q_OS_MACX) && QT_VERSION >= 0x060000` selects
    // `IMOD_DRAW_MOD`; this platform keeps `IMOD_DRAW_XYZ`.
    let draw_flag = IMOD_DRAW_XYZ;

    if co_num < 0
        || pt_num == -1
        || mo_num < 0
        || mo_num >= a.num_mods
        || ob_num >= unsafe { (*a.mod_[mo_num as usize]).obj.len() as i32 }
    {
        // `App->newQtOpenGL` is set on this build.
        unsafe { imodv_draw() };
        return;
    }

    imod_get_index(unsafe { &*a.imod }, &mut obsave, &mut cosave, &mut tpt);

    // 11/29/08: call central select function if changing model
    if a.cur_mod != mo_num {
        imodv_select_model(a, mo_num);
    }

    if ob_num >= 0 {
        obj = unsafe { &mut (&mut (*a.imod).obj)[ob_num as usize] as *mut Iobj };
    } else {
        let vi = a.vi as *mut ImodView;
        obj = if vi.is_null() {
            std::ptr::null_mut()
        } else {
            ivw_get_an_extra_object(unsafe { &mut *vi }, -1 - ob_num)
                .map_or(std::ptr::null_mut(), |o| o as *mut Iobj)
        };
        if obj.is_null() || a.standalone != 0 {
            return;
        }

        // For an extra object with contours, just set mouse from point position
        if !unsafe { (*obj).cont.is_empty() } {
            if (co_num as usize) < unsafe { (*obj).cont.len() } {
                let vi = unsafe { &mut *(a.vi as *mut ImodView) };
                let pt = unsafe { (&(*obj).cont)[co_num as usize].pts[pt_num as usize] };
                vi.xmouse = pt.x;
                vi.ymouse = pt.y;
                vi.zmouse = pt.z;
                ivw_bind_mouse(vi);
                if imod_debug('p') {
                    imod_print_stderr(&format!(
                        "Extra object, point at {:.1} {:.1} {:.1}   inserting {}\n",
                        vi.xmouse,
                        vi.ymouse,
                        vi.zmouse,
                        i32::from(insert)
                    ));
                }
                if insert {
                    input_insert_point(vi, n);
                } else {
                    // Detach from current model point in model mode so point shows up as cross
                    let imod = unsafe { &mut *a.imod };
                    if imod.mousemode == IMOD_MMODEL {
                        imod_set_index(imod, obsave, cosave, -1);
                    }
                    n.imod_draw(unsafe { &mut *a_ptr }, draw_flag);
                }
            }
            return;
        }
    }

    // If there is mesh drawing, get the position in the mesh
    ind_save = unsafe { (*a.imod).cindex };
    if co_num >= unsafe { (*obj).cont.len() as i32 } {
        co_num -= 1 + unsafe { (*obj).cont.len() as i32 };

        if iobj_mesh(unsafe { (*obj).flags }) == 0
            || co_num < 0
            || co_num >= unsafe { (*obj).mesh.len() as i32 }
            || pt_num >= unsafe { (&(*obj).mesh)[co_num as usize].vert.len() as i32 }
        {
            return;
        }
        pickpt = unsafe { (&(*obj).mesh)[co_num as usize].vert[pt_num as usize] };
        if unsafe { (*obj).cont.is_empty() } || insert {
            if a.standalone != 0 {
                return;
            }
            let vi = unsafe { &mut *(a.vi as *mut ImodView) };
            vi.xmouse = pickpt.x;
            vi.ymouse = pickpt.y;
            vi.zmouse = pickpt.z;
            ivw_bind_mouse(vi);
            if imod_debug('p') {
                imod_print_stderr(&format!(
                    "{}mesh, point at {:.1} {:.1} {:.1}  inserting {}\n",
                    if unsafe { (*obj).cont.is_empty() } {
                        "Contourless "
                    } else {
                        ""
                    },
                    vi.xmouse,
                    vi.ymouse,
                    vi.zmouse,
                    i32::from(insert)
                ));
            }
            if insert {
                input_insert_point(vi, n);
            } else {
                let contourless = unsafe { (*obj).cont.is_empty() };
                if unsafe { (*a.imod).mousemode } == IMOD_MMODEL || contourless {
                    // for a real object, set the object index and update the info window
                    if ob_num >= 0 {
                        let imod = unsafe { &mut *a.imod };
                        imod_set_index(imod, ob_num, -1, -1);
                        let surf = unsafe { (&(*obj).mesh)[co_num as usize].surf as i32 };
                        imod_set_cur_mesh_surf(imod, surf);
                        tpt = n.selection_list_size(unsafe { &mut *a_ptr });
                        if !moving
                            || n.imod_selection_list_query(unsafe { &mut *a_ptr }, ob_num, -1) < -1
                        {
                            n.imod_selection_new_cur_point(
                                unsafe { &mut *a_ptr },
                                ind_save,
                                S_CTRL_DOWN.get() as i32,
                            );
                        }
                        if a.standalone == 0 {
                            n.imod_setxyzmouse();
                            if n.selection_list_size(unsafe { &mut *a_ptr }) > 1
                                && n.selection_list_size(unsafe { &mut *a_ptr }) != tpt
                            {
                                wprint("Selected objs:");
                                tob = 0;
                                while tob < unsafe { (*a.imod).obj.len() as i32 } {
                                    if n.imod_selection_list_query(unsafe { &mut *a_ptr }, tob, -1)
                                        > -2
                                    {
                                        wprint(&format!(" {}", tob + 1));
                                    }
                                    tob += 1;
                                }
                                wprint("\n");
                            }
                        }
                    } else {
                        imod_set_index(unsafe { &mut *a.imod }, obsave, cosave, -1);
                    }
                }
                n.imod_draw(unsafe { &mut *a_ptr }, draw_flag);
            }
            return;
        }

        // Search contours for the point, first an exact hit
        co_num = 0;
        while co_num < unsafe { (*obj).cont.len() as i32 } {
            let pts = unsafe { &(&(*obj).cont)[co_num as usize].pts };
            pt_num = 0;
            while pt_num < pts.len() as i32 {
                let p = pts[pt_num as usize];
                if p.x == pickpt.x && p.y == pickpt.y && p.z == pickpt.z {
                    break;
                }
                pt_num += 1;
            }
            if pt_num < pts.len() as i32 {
                break;
            }
            co_num += 1;
        }
        if co_num < unsafe { (*obj).cont.len() as i32 } {
            if imod_debug('p') {
                imod_print_stderr("Mesh hit, exact match\n");
            }
        } else {
            // Now look for closest point
            minsq = 1.0e30;
            tco = 0;
            while tco < unsafe { (*obj).cont.len() as i32 } {
                let pts = unsafe { &(&(*obj).cont)[tco as usize].pts };
                tpt = 0;
                while tpt < pts.len() as i32 {
                    dx = pts[tpt as usize].x - pickpt.x;
                    dx *= dx;
                    if dx < minsq {
                        dy = pts[tpt as usize].y - pickpt.y;
                        dy *= dy;
                        if dy < minsq {
                            dz = pts[tpt as usize].z - pickpt.z;
                            dz *= dz;
                            if dz < minsq {
                                dsqr = dx + dy + dz;
                                if dsqr < minsq {
                                    minco = tco;
                                    minpt = tpt;
                                    minsq = dsqr;
                                }
                            }
                        }
                    }
                    tpt += 1;
                }
                tco += 1;
            }
            if minsq > 1.0e20 {
                return;
            }
            co_num = minco;
            pt_num = minpt;
            if imod_debug('p') {
                imod_print_stderr(&format!(
                    "Mesh hit, nearest point distance {:.3}\n",
                    (minsq as f64).sqrt()
                ));
            }
        }
    } else {
        pickpt =
            unsafe { (&(*a.imod).obj)[ob_num as usize].cont[co_num as usize].pts[pt_num as usize] };
    }

    // If inserting is requesting, always insert a point no matter how it was gotten
    if insert {
        let vi = unsafe { &mut *(a.vi as *mut ImodView) };
        vi.xmouse = pickpt.x;
        vi.ymouse = pickpt.y;
        vi.zmouse = pickpt.z;
        ivw_bind_mouse(vi);
        if imod_debug('p') {
            imod_print_stderr(&format!(
                "Inserting  {:.1} {:.1} {:.1}\n",
                vi.xmouse, vi.ymouse, vi.zmouse
            ));
        }
        input_insert_point(vi, n);
        return;
    }

    // Now process the indexable point whether from contour or mesh
    // Do not add to selection if current point is not picked here
    tpt = S_CTRL_DOWN.get() as i32;
    if S_PICKED_CONTOUR.get() != ind_save.contour || S_PICKED_OBJECT.get() != ind_save.object {
        tpt = 0;
    }
    imod_set_index(unsafe { &mut *a.imod }, ob_num, co_num, pt_num);
    if !moving || n.imod_selection_list_query(unsafe { &mut *a_ptr }, ob_num, co_num) < -1 {
        n.imod_selection_new_cur_point(unsafe { &mut *a_ptr }, ind_save, tpt);
    }
    if a.standalone == 0 {
        n.imod_setxyzmouse();
    } else {
        unsafe { imodv_draw() };
    }
    S_PICKED_CONTOUR.set(unsafe { (*a.imod).cindex.contour });
    S_PICKED_OBJECT.set(unsafe { (*a.imod).cindex.object });
    if imod_debug('p') {
        imod_print_stderr(&format!(
            "hit {ob_num} {co_num} {pt_num}  current picked {} {}\n",
            S_PICKED_OBJECT.get(),
            S_PICKED_CONTOUR.get()
        ));
    }
    let _ = &mut ob_num;
}

/// Original static: `imodvStepTime` (`mv_input.cpp:1663`).
pub fn imodv_step_time(a: &mut ImodvApp, tstep: i32, n: &mut dyn MvInputNativeBoundary) -> i32 {
    if a.standalone == 0 {
        let vi = unsafe { &mut *(a.vi as *mut ImodView) };
        if tstep > 0 {
            input_next_time(vi, n);
        }
        if tstep < 0 {
            input_prev_time(vi, n);
        }
        return 0;
    }

    // The standalone arm steps `a->imod->ctime` against `a->imod->tmax` and
    // the per-contour `time`.  The translated `Imod` carries neither `ctime`
    // nor `tmax` (`imodel.rs` deviation note above `imod_default`), so there
    // is no time index to advance and the loop cannot be entered.
    static REPORTED: std::sync::Once = std::sync::Once::new();
    REPORTED.call_once(|| {
        eprintln!(
            "3dmodv: model time stepping needs Imod::ctime and Imod::tmax, which the translated \
             Imod of libimod/imodel.rs does not carry"
        )
    });
    0
}

/// Original: `imodv_sys_time` (`mv_input.cpp:1690`).
///
/// DNM 2/27/03: replace unix times/clock with Qt time.
pub fn imodv_sys_time() -> i32 {
    // `QTime::currentTime()` is local time of day; only the epoch differs
    // here, and every use is a difference of two of these values.
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = (now.as_secs() % 86_400) as i32;
    let cur_hour = secs / 3600;
    let cur_minute = (secs % 3600) / 60;
    let cur_second = secs % 60;
    let cur_msec = now.subsec_millis() as i32;
    (((cur_hour * 60 + cur_minute) * 60) + cur_second) * 1000 + cur_msec
}

/// Original: `imodvInputRaise` (`mv_input.cpp:1697`).
pub fn imodv_input_raise(n: &mut dyn MvInputNativeBoundary) {
    n.main_win_raise();
    n.dialog_manager_raise(IMODV_DIALOG);
    // `#ifdef _WIN32` adds `activateWindow()`; not compiled on this platform.
}

/// Original static: `imodv_start_movie` (`mv_input.cpp:1713`).
///
/// DNM 11/5/00: changed logic from using interlocked time-outs and workprocs
/// to using just this workproc after starting the movie.
pub fn imodv_start_movie(a: &mut ImodvApp, n: &mut dyn MvInputNativeBoundary) {
    /* DNM: new workproc approach, start it here and go on */
    a.wpid = n.movie_timer_start(if a.standalone != 0 {
        STANDALONE_INTERVAL
    } else {
        MODELVIEW_INTERVAL
    });
    a.movie_frames = 0;
    a.movie_start = imodv_sys_time();
    for m in 0..MAX_MOVIE_TIMES {
        a.movie_times[m] = a.movie_start;
    }
}

/// Original: `imodvMovieTimeout` (`mv_input.cpp:1725`).
pub fn imodv_movie_timeout(a: &mut ImodvApp, n: &mut dyn MvInputNativeBoundary) {
    let index: usize;
    let nframes: i32;
    let rot: f32;
    let scale: f32;

    if a.wpid != 0
        && !n.imodv_closed()
        && a.movie != 0
        && (a.xrot_movie != 0. || a.yrot_movie != 0. || a.zrot_movie != 0.)
    {
        a.movie_frames += 1;
        a.movie_current = imodv_sys_time();
        index = (a.movie_frames as usize) % MAX_MOVIE_TIMES;
        nframes = if (a.movie_frames as usize) < MAX_MOVIE_TIMES {
            a.movie_frames
        } else {
            MAX_MOVIE_TIMES as i32
        };
        rot = ((a.xrot_movie * a.xrot_movie
            + a.yrot_movie * a.yrot_movie
            + a.zrot_movie * a.zrot_movie) as f64)
            .sqrt() as f32;
        scale = ((a.movie_speed * a.throw_factor * (a.movie_current - a.movie_times[index]) as f32)
            as f64
            / (100. * nframes as f64 * rot as f64)) as f32;
        a.movie_times[index] = a.movie_current;
        let (x, y, z) = (
            scale * a.xrot_movie,
            scale * a.yrot_movie,
            scale * a.zrot_movie,
        );
        imodv_compute_rotation(a, x, y, z, n);
        unsafe { imodv_draw() };
    } else {
        a.wpid = 0;
        n.movie_timer_stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imat::imod_mat_new;
    use crate::imod::libimod::imodel::{Imod, Iview};

    #[derive(Default)]
    struct Recorder {
        calls: Vec<String>,
        cursor: (i32, i32),
    }
    impl InputNativeBoundary for Recorder {}
    impl MvInputNativeBoundary for Recorder {
        fn query_pointer_position(&mut self, _a: &ImodvApp) -> (i32, i32) {
            self.cursor
        }
        fn imodv_menu_bgcolor(&mut self, state: i32) {
            self.calls.push(format!("bgcolor {state}"));
        }
        fn imodv_control_change_steps(&mut self, _a: &mut ImodvApp, delta: i32) {
            self.calls.push(format!("steps {delta}"));
        }
        fn imodv_stereo_update(&mut self, _a: &mut ImodvApp) {
            self.calls.push("stereo".into());
        }
        fn ime_set_view_data(&mut self, wi: i32) {
            self.calls.push(format!("view data {wi}"));
        }
        fn movie_timer_start(&mut self, interval: i32) -> i32 {
            self.calls.push(format!("timer {interval}"));
            7
        }
        fn movie_timer_stop(&mut self) {
            self.calls.push("timer stop".into());
        }
    }

    /// One model with one view, as `imodv_open` leaves `Imodv`.
    fn one_model_app() -> (ImodvApp, Box<Imod>) {
        let mut model = Box::new(Imod {
            view: vec![Iview::default()],
            ..Imod::default()
        });
        model.zscale = 1.;
        let ptr = &mut *model as *mut Imod;
        let mut a = ImodvApp {
            imod: ptr,
            num_mods: 1,
            winx: 512,
            winy: 512,
            delta_rot: 10.,
            mat: imod_mat_new(3),
            rmat: imod_mat_new(3),
            ..Default::default()
        };
        a.mod_.push(ptr);
        (a, model)
    }

    #[test]
    fn resolve_rotation_builds_a_matrix() {
        let mut m = imod_mat_new(3).unwrap();
        imodv_resolve_rotation(&mut m, 1., 2., 3.);
        assert_ne!(m.data[0], 1.);
    }

    #[test]
    fn minus_and_underscore_zoom_out_by_the_source_factors() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        a.imod = &*model as *const Imod as *mut Imod;
        let rad = unsafe { (&(*a.imod).view)[0].rad };
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_MINUS,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(
            unsafe { (&(*a.imod).view)[0].rad },
            (rad as f64 / 0.95238095) as f32
        );
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_UNDERSCORE,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(
            unsafe { (&(*a.imod).view)[0].rad },
            ((rad as f64 / 0.95238095) as f32 as f64 / 0.5) as f32
        );
        drop(model);
    }

    #[test]
    fn g_steps_the_sphere_quality_field_up_and_down() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_G,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(
            (unsafe { (&(*a.imod).view)[0].world } & WORLD_QUALITY_BITS) >> WORLD_QUALITY_SHIFT,
            1
        );
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_G,
                modifiers: INPUT_SHIFT,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(
            (unsafe { (&(*a.imod).view)[0].world } & WORLD_QUALITY_BITS) >> WORLD_QUALITY_SHIFT,
            0
        );
        drop(model);
    }

    #[test]
    fn eight_toggles_drawall_and_tells_the_view_dialog() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_8,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(a.drawall, 3);
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_8,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(a.drawall, 0);
        assert_eq!(n.calls, vec!["view data 3", "view data 0"]);
        drop(model);
    }

    #[test]
    fn comma_and_period_change_the_rotation_step() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_COMMA,
                ..Default::default()
            },
            &mut n,
        );
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_PERIOD,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(n.calls, vec!["steps -1", "steps 1"]);
        drop(model);
    }

    #[test]
    fn brackets_step_the_stereo_parallax_by_a_half() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        a.plax = 5.;
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_BRACKET_LEFT,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(a.plax, 4.5);
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_BRACKET_RIGHT,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(a.plax, 5.0);
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_A,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(a.plax, -5.0);
        drop(model);
    }

    #[test]
    fn keypad_arrows_rotate_by_delta_rot_and_plain_arrows_translate() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        a.moveall = 1;
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_UP,
                modifiers: INPUT_KEYPAD,
                ..Default::default()
            },
            &mut n,
        );
        assert!(unsafe { (&(*a.imod).view)[0].rot.x } != 0.);
        let rot = unsafe { (&(*a.imod).view)[0].rot };
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_UP,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(unsafe { (&(*a.imod).view)[0].rot }, rot);
        assert!(unsafe { (&(*a.imod).view)[0].trans.y } != 0.);
        drop(model);
    }

    #[test]
    fn k_toggles_the_picking_method_and_l_the_slicer_link() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_K,
                modifiers: INPUT_CTRL | INPUT_SHIFT,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(a.legacy_pick_mode, 1);
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_L,
                modifiers: INPUT_CTRL | INPUT_SHIFT,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(a.link_to_slicer, 1);
        drop(model);
    }

    #[test]
    fn shift_and_control_key_presses_track_the_modifier_statics() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        imodv_app_lost_focus();
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_CONTROL,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(S_CTRL_DOWN.get(), INPUT_CTRL);
        imodv_key_release(
            &mut a,
            InputEvent {
                key: KEY_CONTROL,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(S_CTRL_DOWN.get(), 0);
        imodv_key_press(
            &mut a,
            InputEvent {
                key: KEY_SHIFT,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(S_SHIFT_DOWN.get(), INPUT_SHIFT);
        imodv_key_release(
            &mut a,
            InputEvent {
                key: KEY_SHIFT,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(S_SHIFT_DOWN.get(), 0);
        drop(model);
    }

    #[test]
    fn a_middle_press_records_the_throw_origin_and_a_release_throws() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        a.movie = 1;
        imodv_mouse_press(
            &mut a,
            InputEvent {
                x: 100,
                y: 100,
                button: INPUT_MIDDLE,
                buttons: INPUT_MIDDLE,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!((B2X.get(), B2Y.get()), (100, 100));
        assert_eq!((a.lastmx, a.lastmy), (100, 100));
        n.cursor = (180, 100);
        imodv_mouse_release(
            &mut a,
            InputEvent {
                x: 180,
                y: 100,
                button: INPUT_MIDDLE,
                buttons: 0,
                ..Default::default()
            },
            &mut n,
        );
        // `MOUSE_TO_THROW * dx` with dx = 80 rounds to 20 about Y.
        assert_eq!(a.yrot_movie, 20.);
        assert_eq!(a.xrot_movie, 0.);
        assert!(n.calls.iter().any(|c| c.starts_with("timer ")));
        drop(model);
    }

    #[test]
    fn a_small_middle_throw_turns_the_movie_off() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        a.movie = 1;
        imodv_mouse_press(
            &mut a,
            InputEvent {
                x: 100,
                y: 100,
                button: INPUT_MIDDLE,
                buttons: INPUT_MIDDLE,
                ..Default::default()
            },
            &mut n,
        );
        imodv_mouse_release(
            &mut a,
            InputEvent {
                x: 102,
                y: 101,
                button: INPUT_MIDDLE,
                buttons: 0,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(a.movie, 0);
        assert_eq!(a.yrot_movie, 0.);
        drop(model);
    }

    #[test]
    fn the_wheel_zooms_by_a_power_of_the_source_factor() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        let rad = unsafe { (&(*a.imod).view)[0].rad };
        imodv_scroll_wheel(
            &mut a,
            InputEvent {
                delta: 120,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(
            unsafe { (&(*a.imod).view)[0].rad },
            (rad as f64 / 1.05f64.powf(-1.)) as f32
        );
        drop(model);
    }

    #[test]
    fn a_left_drag_translates_the_model_view() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        unsafe { (&mut (*a.imod).view)[0].rad = 100. };
        imodv_mouse_press(
            &mut a,
            InputEvent {
                x: 50,
                y: 50,
                button: INPUT_LEFT,
                buttons: INPUT_LEFT,
                ..Default::default()
            },
            &mut n,
        );
        imodv_mouse_move(
            &mut a,
            InputEvent {
                x: 60,
                y: 55,
                buttons: INPUT_LEFT,
                ..Default::default()
            },
            &mut n,
        );
        assert!(unsafe { (&(*a.imod).view)[0].trans.x } != 0.);
        assert_eq!((a.lastmx, a.lastmy), (60, 55));
        drop(model);
    }

    #[test]
    fn movie_steps_stop_the_timer_when_no_rotation_is_set() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        a.wpid = 3;
        a.movie = 1;
        imodv_movie_timeout(&mut a, &mut n);
        assert_eq!(a.wpid, 0);
        assert_eq!(n.calls, vec!["timer stop"]);
        drop(model);
    }

    #[test]
    fn a_started_movie_records_the_frame_ring() {
        let (mut a, model) = one_model_app();
        let mut n = Recorder::default();
        a.standalone = 1;
        imodv_start_movie(&mut a, &mut n);
        assert_eq!(a.wpid, 7);
        assert_eq!(a.movie_frames, 0);
        assert!(a.movie_times.iter().all(|t| *t == a.movie_start));
        assert_eq!(n.calls, vec!["timer 1"]);
        drop(model);
    }

    #[test]
    fn clip_center_and_angles_places_the_plane_point_at_the_view_center() {
        let (mut a, model) = one_model_app();
        let mut cen = Ipoint::default();
        let mut alpha = 0.;
        let mut beta = 0.;
        clip_center_and_angles(
            &a,
            &Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            &Ipoint {
                x: 0.,
                y: 0.,
                z: 1.,
            },
            &mut cen,
            &mut alpha,
            &mut beta,
        );
        // normal (0,0,1): beta is -atan2(0, 1) and alpha is
        // -(atan2(1, 0) - 1.570796), the residue of the source's literal.
        assert_eq!(beta, 0.);
        assert!(alpha.abs() < 1.0e-6);
        let _ = &mut a;
        drop(model);
    }

    #[test]
    fn query_pointer_reports_the_recorded_button_and_modifier_mask() {
        let (a, model) = one_model_app();
        let mut n = Recorder::default();
        n.cursor = (11, 22);
        S_MID_DOWN.set(INPUT_MIDDLE);
        S_SHIFT_DOWN.set(INPUT_SHIFT);
        let mut x = 0;
        let mut y = 0;
        let mask = imodv_query_pointer(&a, &mut x, &mut y, &mut n);
        assert_eq!((x, y), (11, 22));
        assert_eq!(mask, INPUT_MIDDLE | INPUT_SHIFT);
        S_MID_DOWN.set(0);
        S_SHIFT_DOWN.set(0);
        drop(model);
    }
}
