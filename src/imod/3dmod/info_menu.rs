//! Translation of `IMOD/3dmod/info_menu.cpp` together with the `InfoWindow`
//! declarations in `info_setup.h`.
//!
//! This is deliberately a menu-controller unit, not a replacement user
//! interface.  Qt dialogs, the information window, and the live viewer/model
//! services are represented by [`InfoMenuBoundary`].  The dispatch ordering,
//! guards, confirmation state, and menu identifiers mirror the C++ source.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::icont::imod_contour_break;
use crate::imod::libimod::imodel::{Icont, Iobj, Ipoint};

pub const FILE_MENU_NEW: i32 = 0;
pub const FILE_MENU_OPEN: i32 = 1;
pub const FILE_MENU_RELOAD: i32 = 2;
pub const FILE_MENU_SAVE: i32 = 3;
pub const FILE_MENU_SAVEAS: i32 = 4;
pub const FILE_MENU_SNAPDIR: i32 = 5;
pub const FILE_MENU_SNAPGRAY: i32 = 6;
pub const FILE_MENU_MOVIEMONT: i32 = 7;
pub const FILE_MENU_TIFF: i32 = 8;
pub const FILE_MENU_EXTRACT: i32 = 9;
pub const FILE_MENU_PROCESS: i32 = 10;
pub const FILE_MENU_SAVEINFO: i32 = 11;
pub const FILE_MENU_QUIT: i32 = 12;
pub const FWRITE_MENU_IMOD: i32 = 13;
pub const FWRITE_MENU_WIMP: i32 = 14;
pub const FWRITE_MENU_NFF: i32 = 15;
pub const FWRITE_MENU_SYNU: i32 = 16;
pub const EDIT_MENU_GRAIN: i32 = 17;
pub const EDIT_MENU_ANGLES: i32 = 18;
pub const EDIT_MENU_SCALEBAR: i32 = 19;
pub const EDIT_MENU_SAVE_DOCK: i32 = 20;
pub const EDIT_MENU_REOPEN_DOCK: i32 = 21;
pub const EDIT_MENU_PREFS: i32 = 22;
pub const EMODEL_MENU_HEADER: i32 = 23;
pub const EMODEL_MENU_OFFSETS: i32 = 24;
pub const EMODEL_MENU_CLEAN: i32 = 25;
pub const EOBJECT_MENU_NEW: i32 = 26;
pub const EOBJECT_MENU_DELETE: i32 = 27;
pub const EOBJECT_MENU_COLOR: i32 = 28;
pub const EOBJECT_MENU_TYPE: i32 = 29;
pub const EOBJECT_MENU_INFO: i32 = 30;
pub const EOBJECT_MENU_MOVE: i32 = 31;
pub const EOBJECT_MENU_CLEAN: i32 = 32;
pub const EOBJECT_MENU_FIXZ: i32 = 33;
pub const EOBJECT_MENU_FILLIN: i32 = 34;
pub const EOBJECT_MENU_FLATTEN: i32 = 35;
pub const EOBJECT_MENU_SORTDIST: i32 = 36;
pub const EOBJECT_MENU_RENUMBER: i32 = 37;
pub const EOBJECT_MENU_COMBINE: i32 = 38;
pub const EOBJECT_MENU_LIST_TO_SEL: i32 = 39;
pub const ESURFACE_MENU_NEW: i32 = 40;
pub const ESURFACE_MENU_GOTO: i32 = 41;
pub const ESURFACE_MENU_MOVE: i32 = 42;
pub const ESURFACE_MENU_DELETE: i32 = 43;
pub const ESURFACE_MENU_SORT: i32 = 44;
pub const ECONTOUR_MENU_NEW: i32 = 45;
pub const ECONTOUR_MENU_DELETE: i32 = 46;
pub const ECONTOUR_MENU_MOVE: i32 = 47;
pub const ECONTOUR_MENU_SORT: i32 = 48;
pub const ECONTOUR_MENU_AUTO: i32 = 49;
pub const ECONTOUR_MENU_TYPE: i32 = 50;
pub const ECONTOUR_MENU_INFO: i32 = 51;
pub const ECONTOUR_MENU_BREAK: i32 = 52;
pub const ECONTOUR_MENU_JOIN: i32 = 53;
pub const ECONTOUR_MENU_FIXZ: i32 = 54;
pub const ECONTOUR_MENU_INVERT: i32 = 55;
pub const ECONTOUR_MENU_COPY: i32 = 56;
pub const ECONTOUR_MENU_LOOPBACK: i32 = 57;
pub const ECONTOUR_MENU_FILLIN: i32 = 58;
pub const EPOINT_MENU_DELETE: i32 = 59;
pub const EPOINT_MENU_SORTZ: i32 = 60;
pub const EPOINT_MENU_SORTDIST: i32 = 61;
pub const EPOINT_MENU_DIST: i32 = 62;
pub const EPOINT_MENU_VALUE: i32 = 63;
pub const EPOINT_MENU_SIZE: i32 = 64;
pub const EIMAGE_MENU_PROCESS: i32 = 65;
pub const EIMAGE_MENU_COLORMAP: i32 = 66;
pub const EIMAGE_MENU_RELOAD: i32 = 67;
pub const EIMAGE_MENU_FLIP: i32 = 68;
pub const EIMAGE_MENU_FILLCACHE: i32 = 69;
pub const EIMAGE_MENU_FILLER: i32 = 70;
pub const IMAGE_MENU_GRAPH: i32 = 71;
pub const IMAGE_MENU_SLICER: i32 = 72;
pub const IMAGE_MENU_LINKSLICE: i32 = 73;
pub const IMAGE_MENU_TUMBLER: i32 = 74;
pub const IMAGE_MENU_MODV: i32 = 75;
pub const IMAGE_MENU_ZAP: i32 = 76;
pub const IMAGE_MENU_XYZ: i32 = 77;
pub const IMAGE_MENU_PIXEL: i32 = 78;
pub const IMAGE_MENU_LOCATOR: i32 = 79;
pub const IMAGE_MENU_MULTIZ: i32 = 80;
pub const IMAGE_MENU_ISOSURFACE: i32 = 81;
pub const HELP_MENU_CONTROLS: i32 = 82;
pub const HELP_MENU_MAN: i32 = 83;
pub const HELP_MENU_MENUS: i32 = 84;
pub const HELP_MENU_HOTKEY: i32 = 85;
pub const HELP_MENU_ABOUT: i32 = 86;
pub const LAST_MENU_ID: usize = 87;

/// Observable parts of `InfoWindow::mActions` used by the menu callbacks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InfoMenuAction {
    pub checked: bool,
}

/// Source-level viewer state queried by menu guards.  All live model/UI work
/// remains at the paired native boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InfoMenuState {
    pub forbid_level: i32,
    pub doing_initial_load: bool,
    pub model_changed: bool,
    pub rgb_store: bool,
    pub fake_image: bool,
    pub vm_size: i32,
    pub mouse_model: bool,
}

/// Direct Qt/viewer boundary for the source calls in this translation.
/// `call` has the original callee as its first argument; it is intentionally
/// not a policy layer or a synthetic GUI.
pub trait InfoMenuBoundary {
    fn state(&self) -> InfoMenuState;
    fn call(&mut self, callee: &'static str, args: &[i32]);
    fn ask_forever(&mut self, _: &str) -> i32 {
        1
    }
    fn input_integer(&mut self, _: i32, _: i32, _: i32, _: &str) -> Option<i32> {
        None
    }
    fn input_text(&mut self, _: &str) -> Option<String> {
        None
    }
}

/// `InfoWindow` fields owned by `info_setup.h` that are consumed here.
#[derive(Clone, Debug)]
pub struct InfoWindow {
    pub m_actions: [InfoMenuAction; LAST_MENU_ID],
    pub obj_moveto: i32,
    pub last_object_delete: i32,
    pub last_object_combine: i32,
    pub last_fixz: i32,
    pub last_flatten: i32,
    pub last_fillin: i32,
    pub last_sortdist: i32,
    pub last_surface_delete: i32,
}
impl Default for InfoWindow {
    fn default() -> Self {
        Self {
            m_actions: std::array::from_fn(|_| InfoMenuAction::default()),
            obj_moveto: 0,
            last_object_delete: 0,
            last_object_combine: 0,
            last_fixz: 0,
            last_flatten: 0,
            last_fillin: 0,
            last_sortdist: 0,
            last_surface_delete: 0,
        }
    }
}

impl InfoWindow {
    /// `InfoWindow::fileSlot`.
    pub fn file_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        let s = b.state();
        if s.forbid_level != 0 {
            if item == FILE_MENU_QUIT {
                b.call("imod_quit", &[]);
            }
            return;
        }
        match item {
            FILE_MENU_NEW => {
                if !(s.doing_initial_load && s.model_changed) {
                    b.call("undo.clearUnits", &[]);
                    b.call("createNewModel", &[]);
                }
            }
            FILE_MENU_RELOAD | FILE_MENU_OPEN => {
                if !(s.doing_initial_load && s.model_changed) {
                    b.call("imod_info_forbid", &[]);
                    b.call("imod_info_input", &[]);
                    b.call("releaseKeyboard", &[]);
                    b.call(
                        if item == FILE_MENU_OPEN {
                            "openModel"
                        } else {
                            "openModel.reload"
                        },
                        &[],
                    );
                    b.call("imod_info_enable", &[]);
                }
            }
            FILE_MENU_SAVE => {
                if !s.doing_initial_load {
                    b.call("imod_info_forbid", &[]);
                    b.call("imod_info_input", &[]);
                    b.call("releaseKeyboard", &[]);
                    b.call("SaveModel", &[]);
                    b.call("imod_info_enable", &[]);
                }
            }
            FILE_MENU_SAVEAS => {
                if !s.doing_initial_load {
                    b.call("imod_info_forbid", &[]);
                    b.call("imod_info_input", &[]);
                    b.call("releaseKeyboard", &[]);
                    b.call("SaveasModel", &[]);
                    b.call("imod_info_enable", &[]);
                }
            }
            FILE_MENU_MOVIEMONT => b.call("imodMovieConDialog", &[]),
            FILE_MENU_SNAPDIR => {
                b.call("imod_info_forbid", &[]);
                b.call("imod_info_input", &[]);
                b.call("releaseKeyboard", &[]);
                b.call("b3dSetSnapDirectory", &[]);
                b.call("imod_info_enable", &[]);
            }
            FILE_MENU_SNAPGRAY => {
                let a = &mut self.m_actions[FILE_MENU_SNAPGRAY as usize];
                a.checked = !a.checked;
                b.call("convertSnap", &[a.checked as i32]);
            }
            FILE_MENU_TIFF => {
                if s.rgb_store {
                    b.call("imod_info_forbid", &[]);
                    b.call("imod_info_input", &[]);
                    b.call("releaseKeyboard", &[]);
                    b.call("b3dSnapshot_TIF", &[]);
                    b.call("imod_info_enable", &[]);
                } else {
                    b.call("wprint.color_only", &[]);
                }
            }
            FILE_MENU_EXTRACT | FILE_MENU_PROCESS => {
                b.call("imod_info_forbid", &[]);
                b.call("imod_info_input", &[]);
                b.call("releaseKeyboard", &[]);
                b.call(
                    if item == FILE_MENU_EXTRACT {
                        "extract"
                    } else {
                        "processFile"
                    },
                    &[],
                );
                b.call("imod_info_enable", &[]);
            }
            FILE_MENU_SAVEINFO => b.call("wprintWriteFile", &[]),
            FILE_MENU_QUIT => b.call("imod_quit", &[]),
            _ => {}
        }
    }
    /// `InfoWindow::fileWriteSlot`.
    pub fn file_write_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        b.call("imod_info_forbid", &[]);
        b.call("imod_info_input", &[]);
        match item {
            FWRITE_MENU_IMOD => b.call("imodWrite", &[]),
            FWRITE_MENU_WIMP => b.call("imod_to_wmod", &[]),
            FWRITE_MENU_NFF => b.call("imod_to_nff", &[]),
            FWRITE_MENU_SYNU => b.call("imod_to_synu", &[]),
            _ => {}
        }
        b.call("imod_info_enable", &[]);
    }
    /// `InfoWindow::editSlot`.
    pub fn edit_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        match item {
            EDIT_MENU_GRAIN => b.call("fineGrainOpen", &[]),
            EDIT_MENU_ANGLES => b.call("slicerAnglesOpen", &[]),
            EDIT_MENU_SCALEBAR => b.call("scaleBarOpen", &[]),
            EDIT_MENU_SAVE_DOCK => b.call("imodDialogManager.saveStackStateToSettings", &[]),
            EDIT_MENU_REOPEN_DOCK => b.call("imodDialogManager.restoreStackFromSettings", &[]),
            EDIT_MENU_PREFS => b.call("ImodPrefs.editPrefs", &[]),
            _ => {}
        }
    }
    /// `InfoWindow::editModelSlot`.
    pub fn edit_model_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            EMODEL_MENU_HEADER => b.call("openModelEdit", &[]),
            EMODEL_MENU_OFFSETS => b.call("openModelOffset", &[]),
            EMODEL_MENU_CLEAN => {
                if b.ask_forever("Delete all empty objects?") != 0 {
                    b.call("vbCleanupVBD", &[]);
                    b.call("imodDeleteObject.empty_reverse", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodSelectionListClear", &[]);
                    b.call("imod_cmap", &[]);
                    b.call("imod_info_setobjcolor", &[]);
                    b.call("imodDraw.rethink", &[]);
                    b.call("imodvObjedNewView", &[]);
                }
            }
            _ => {}
        }
    }
    /// `InfoWindow::editObjectSlot`.
    pub fn edit_object_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            EOBJECT_MENU_NEW => {
                b.call("inputNewObject", &[]);
                b.call("imod_object_edit", &[]);
                b.call("imod_info_setobjcolor", &[]);
                b.call("imodvObjedNewView", &[]);
            }
            EOBJECT_MENU_DELETE => {
                if self.last_object_delete < 2 {
                    self.last_object_delete = b.ask_forever("Delete Object?");
                }
                if self.last_object_delete != 0 {
                    b.call("undo.objectRemoval.selected_reverse", &[]);
                    b.call("imodDeleteObject.selected_reverse", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodSelectionListClear", &[]);
                    b.call("ensureNewObject", &[]);
                    b.call("imodDraw.mod", &[]);
                    b.call("imod_cmap", &[]);
                    b.call("imod_info_setobjcolor", &[]);
                    b.call("imodvObjedNewView", &[]);
                }
            }
            EOBJECT_MENU_COLOR => {
                b.call("imod_info_forbid", &[]);
                b.call("imod_info_input", &[]);
                b.call("imod_info_enable", &[]);
                b.call("imod_object_color", &[]);
            }
            EOBJECT_MENU_TYPE => {
                b.call("imod_object_edit", &[]);
                b.call("imod_draw_window", &[]);
                b.call("imod_info_setobjcolor", &[]);
            }
            EOBJECT_MENU_MOVE => {
                b.call("imod_info_forbid", &[]);
                b.call("imod_info_input", &[]);
                b.call("imod_info_enable", &[]);
                if let Some(value) = b.input_integer(
                    self.obj_moveto,
                    1,
                    i32::MAX,
                    "Move all contours to selected object.",
                ) {
                    self.obj_moveto = value;
                    b.call("imodMoveAllContours", &[value - 1]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodSelectionListClear", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            EOBJECT_MENU_LIST_TO_SEL => {
                if let Some(text) =
                    b.input_text("List of objects to select (comma-separated ranges)")
                {
                    b.call("parselist", &[text.len() as i32]);
                    b.call("imodSelectionListClear", &[]);
                    b.call("imodSelectionListAdd.parsed", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            EOBJECT_MENU_COMBINE => {
                if self.last_object_combine < 2 {
                    self.last_object_combine = b.ask_forever("Combine selected objects?");
                }
                if self.last_object_combine != 0 {
                    b.call("imodMoveAllContours.selected", &[]);
                    b.call("imodDeleteObject.selected_reverse", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodSelectionListClear", &[]);
                    b.call("imodDraw.mod", &[]);
                    b.call("imod_cmap", &[]);
                    b.call("imod_info_setobjcolor", &[]);
                    b.call("imodvObjedNewView", &[]);
                }
            }
            EOBJECT_MENU_INFO => b.call("objectInfo", &[]),
            EOBJECT_MENU_CLEAN => {
                b.call("imodDeleteContour.empty", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodSelectionListClear", &[]);
                b.call("imodDraw.rethink", &[]);
            }
            EOBJECT_MENU_FIXZ => {
                if self.last_fixz < 2 {
                    self.last_fixz = b.ask_forever("Break all contours into Z planes?");
                }
                if self.last_fixz != 0 {
                    b.call("imodContourBreakByZ.all", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodSelectionListClear", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            EOBJECT_MENU_FLATTEN => {
                if self.last_flatten < 2 {
                    self.last_flatten = b.ask_forever("Flatten all contours?");
                }
                if self.last_flatten != 0 {
                    b.call("imodContourFlatten.all", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodSelectionListClear", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            EOBJECT_MENU_FILLIN => {
                if self.last_fillin < 2 {
                    self.last_fillin = b.ask_forever("Fill in Z in all contours?");
                }
                if self.last_fillin != 0 {
                    b.call("imodFillInContourZ.all", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            EOBJECT_MENU_SORTDIST => {
                if self.last_sortdist < 2 {
                    self.last_sortdist = b.ask_forever("Sort points by distance?");
                }
                if self.last_sortdist != 0 {
                    b.call("imodContourSort3D.all", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            EOBJECT_MENU_RENUMBER => {
                b.call("imod_info_forbid", &[]);
                b.call("imod_info_input", &[]);
                b.call("imod_info_enable", &[]);
                b.call("imodMoveObject", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodSelectionListClear", &[]);
                b.call("imodDraw.mod", &[]);
                b.call("imodvObjedNewView", &[]);
            }
            _ => {}
        }
    }
    /// `InfoWindow::editSurfaceSlot`.
    pub fn edit_surface_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            ESURFACE_MENU_NEW => b.call("inputNewSurface", &[]),
            ESURFACE_MENU_GOTO => b.call("imodContEditSurf", &[]),
            ESURFACE_MENU_MOVE => b.call("imodContEditMoveDialog.surface", &[]),
            ESURFACE_MENU_DELETE => {
                if self.last_surface_delete < 2 {
                    self.last_surface_delete = b.ask_forever("Delete contours in this surface?");
                }
                if self.last_surface_delete != 0 {
                    b.call("imodDeleteContour.surface_reverse", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodSelectionListClear", &[]);
                    b.call("imod_setxyzmouse", &[]);
                }
            }
            ESURFACE_MENU_SORT => self.sort_contours(b, 1),
            _ => {}
        }
    }
    /// `InfoWindow::editContourSlot`.
    pub fn edit_contour_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            ECONTOUR_MENU_NEW => b.call("inputNewContour", &[]),
            ECONTOUR_MENU_DELETE => b.call("inputDeleteContour", &[]),
            ECONTOUR_MENU_MOVE => b.call("imodContEditMoveDialog.contour", &[]),
            ECONTOUR_MENU_SORT => self.sort_contours(b, 0),
            ECONTOUR_MENU_AUTO => {
                b.call("autox_open", &[]);
                b.call("imod_info_setocp", &[]);
            }
            ECONTOUR_MENU_TYPE | EPOINT_MENU_SIZE => b.call("imodContEditSurf", &[]),
            ECONTOUR_MENU_INFO => b.call("contourStatistics", &[]),
            ECONTOUR_MENU_BREAK => b.call("imodContEditBreakOpen", &[]),
            ECONTOUR_MENU_FIXZ => {
                b.call("imodContourBreakByZ.current", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodSelectionListClear", &[]);
                b.call("imodDraw.mod", &[]);
            }
            ECONTOUR_MENU_JOIN => b.call("imodContEditJoinOpen", &[]),
            ECONTOUR_MENU_INVERT => {
                b.call("undo.contourDataChg", &[]);
                b.call("imodel_contour_invert", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodDraw.mod", &[]);
            }
            ECONTOUR_MENU_COPY => b.call("openContourCopyDialog", &[]),
            ECONTOUR_MENU_LOOPBACK => {
                b.call("undo.contourDataChg", &[]);
                b.call("imodPointAppend.loopback", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodDraw.mod", &[]);
            }
            ECONTOUR_MENU_FILLIN => {
                b.call("imodFillInContourZ", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodDraw.mod", &[]);
            }
            _ => {}
        }
    }
    /// `InfoWindow::editPointSlot`.
    pub fn edit_point_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            EPOINT_MENU_DELETE => b.call("inputDeletePoint", &[]),
            EPOINT_MENU_SORTDIST => {
                if b.state().mouse_model {
                    b.call("undo.contourDataChg", &[]);
                    b.call("imodContourSort3D", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            EPOINT_MENU_SORTZ => {
                if b.state().mouse_model {
                    b.call("undo.contourDataChg", &[]);
                    b.call("imodel_contour_sortz", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            EPOINT_MENU_DIST => b.call("pointDistanceReport", &[]),
            EPOINT_MENU_VALUE => b.call("ivwGetFileValue.report", &[]),
            EPOINT_MENU_SIZE => b.call("imodContEditSurf", &[]),
            _ => {}
        }
    }
    /// `InfoWindow::editImageSlot`.
    pub fn edit_image_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            EIMAGE_MENU_PROCESS => b.call("inputIProcOpen", &[]),
            EIMAGE_MENU_COLORMAP => b.call("imod_cmap.change", &[]),
            EIMAGE_MENU_RELOAD => b.call("imodImageScaleDialog", &[]),
            EIMAGE_MENU_FLIP => {
                b.call("undo.clearUnits", &[]);
                b.call("vbCleanupVBD", &[]);
                b.call("ivwFlip", &[]);
                b.call("ivwCheckWildFlag", &[]);
                b.call("imodDraw.image_xyz_mod", &[]);
            }
            EIMAGE_MENU_FILLCACHE => {
                if b.state().vm_size != 0 {
                    b.call("imodCacheFill", &[])
                } else {
                    b.call("wprint.cache_not_active", &[])
                }
            }
            EIMAGE_MENU_FILLER => {
                if b.state().vm_size != 0 {
                    b.call("imodCacheFillDialog", &[])
                } else {
                    b.call("wprint.cache_not_active", &[])
                }
            }
            _ => {}
        }
    }
    /// `InfoWindow::imageSlot`.
    pub fn image_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        let s = b.state();
        if s.forbid_level != 0 || s.doing_initial_load {
            return;
        }
        match item {
            IMAGE_MENU_GRAPH => {
                if !s.fake_image && !s.rgb_store {
                    b.call("xgraphOpen", &[])
                }
            }
            IMAGE_MENU_SLICER => b.call("slicerOpen", &[0]),
            IMAGE_MENU_LINKSLICE => b.call("setupLinkedSlicers", &[]),
            IMAGE_MENU_TUMBLER => {
                if !s.rgb_store {
                    b.call("xtumOpen", &[])
                }
            }
            IMAGE_MENU_LOCATOR => b.call("locatorOpen", &[]),
            IMAGE_MENU_MODV => {
                b.call("imod_autosave", &[]);
                b.call("imodv_open", &[]);
            }
            IMAGE_MENU_ZAP => b.call("imod_zap_open", &[0]),
            IMAGE_MENU_MULTIZ => b.call("imod_zap_open", &[1]),
            IMAGE_MENU_XYZ => b.call("xxyz_open", &[]),
            IMAGE_MENU_PIXEL => {
                if !s.fake_image {
                    b.call("open_pixelview", &[])
                }
            }
            IMAGE_MENU_ISOSURFACE => {
                if !s.fake_image && !s.rgb_store {
                    b.call("imodv_open", &[]);
                    b.call("imodvIsosurfaceEditDialog", &[1]);
                }
            }
            _ => {}
        }
    }
    /// `InfoWindow::helpSlot`.
    pub fn help_slot(&mut self, item: i32, b: &mut dyn InfoMenuBoundary) {
        match item {
            HELP_MENU_MAN => b.call("imodShowHelpPage.3dmod", &[]),
            HELP_MENU_MENUS => b.call("imodShowHelpPage.menus", &[]),
            HELP_MENU_CONTROLS => b.call("imodShowHelpPage.infowin", &[]),
            HELP_MENU_HOTKEY => b.call("imodShowHelpPage.keyboard", &[]),
            HELP_MENU_ABOUT => {
                b.call("imod_info_forbid", &[]);
                b.call("imod_info_input", &[]);
                b.call("imod_info_enable", &[]);
                b.call("dia_vasmsg.about", &[]);
            }
            _ => {}
        }
    }
    /// `InfoWindow::sortContours`.
    pub fn sort_contours(&mut self, b: &mut dyn InfoMenuBoundary, if_by_surf: i32) {
        if !b.state().mouse_model {
            b.call("wprint.must_be_model_mode", &[]);
        } else {
            b.call("undo.clearUnits", &[]);
            b.call("imodObjectSortBySurf", &[if_by_surf]);
            b.call("imodSelectionListClear", &[]);
            b.call("imod_info_setocp", &[]);
        }
    }
}

/// `imodContourBreakByZ`.  This retains the source's reverse scan, rounded-Z
/// comparison and contour append behavior; undo/store transfer is performed
/// by the supplied direct viewer boundary in the calling slot.
pub fn imod_contour_break_by_z(obj: &mut Iobj, co: usize) -> i32 {
    let Some(initial) = obj.cont.get(co) else {
        return 0;
    };
    let mut first = true;
    let mut point = initial.pts.len();
    while point > 1 {
        point -= 1;
        let Some(cont) = obj.cont.get_mut(co) else {
            return 0;
        };
        if (cont.pts[point].z + 0.5).floor() as i32 != (cont.pts[point - 1].z + 0.5).floor() as i32
        {
            let Some(new_cont) = imod_contour_break(cont, point as i32, -1) else {
                return 0;
            };
            obj.cont.push(new_cont);
            first = false;
        }
    }
    (!first) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native {
        state: InfoMenuState,
        calls: Vec<(&'static str, Vec<i32>)>,
        answer: i32,
    }
    impl InfoMenuBoundary for Native {
        fn state(&self) -> InfoMenuState {
            self.state
        }
        fn call(&mut self, n: &'static str, a: &[i32]) {
            self.calls.push((n, a.to_vec()));
        }
        fn ask_forever(&mut self, _: &str) -> i32 {
            self.answer
        }
    }
    #[test]
    fn file_save_keeps_source_forbid_sequence() {
        let mut w = InfoWindow::default();
        let mut n = Native::default();
        w.file_slot(FILE_MENU_SAVE, &mut n);
        assert_eq!(
            n.calls.iter().map(|x| x.0).collect::<Vec<_>>(),
            [
                "imod_info_forbid",
                "imod_info_input",
                "releaseKeyboard",
                "SaveModel",
                "imod_info_enable"
            ]
        );
    }
    #[test]
    fn image_guard_is_observed() {
        let mut w = InfoWindow::default();
        let mut n = Native {
            state: InfoMenuState {
                doing_initial_load: true,
                ..Default::default()
            },
            ..Default::default()
        };
        w.image_slot(IMAGE_MENU_SLICER, &mut n);
        assert!(n.calls.is_empty());
    }
    #[test]
    fn contour_breaks_on_rounded_z_transitions() {
        let mut obj = Iobj::default();
        obj.cont.push(Icont {
            pts: vec![
                Ipoint {
                    z: 0.,
                    ..Default::default()
                },
                Ipoint {
                    z: 1.,
                    ..Default::default()
                },
                Ipoint {
                    z: 1.,
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        assert_eq!(imod_contour_break_by_z(&mut obj, 0), 1);
        assert_eq!(obj.cont.len(), 2);
    }
}
