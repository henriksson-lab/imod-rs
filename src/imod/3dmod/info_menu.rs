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

/// `ECONTOUR_MENU_*` members of the menu-id enum (`info_setup.h:33-61`).
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContourMenuAction {
    New = 45,
    Delete = 46,
    Move = 47,
    Sort = 48,
    Auto = 49,
    Type = 50,
    Info = 51,
    Break = 52,
    Join = 53,
    FixZ = 54,
    Invert = 55,
    Copy = 56,
    LoopBack = 57,
    FillIn = 58,
}

impl ContourMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            45 => Some(Self::New),
            46 => Some(Self::Delete),
            47 => Some(Self::Move),
            48 => Some(Self::Sort),
            49 => Some(Self::Auto),
            50 => Some(Self::Type),
            51 => Some(Self::Info),
            52 => Some(Self::Break),
            53 => Some(Self::Join),
            54 => Some(Self::FixZ),
            55 => Some(Self::Invert),
            56 => Some(Self::Copy),
            57 => Some(Self::LoopBack),
            58 => Some(Self::FillIn),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

/// `EPOINT_MENU_*` members of the menu-id enum (`info_setup.h:33-61`).
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointMenuAction {
    Delete = 59,
    SortZ = 60,
    SortDist = 61,
    Dist = 62,
    Value = 63,
    Size = 64,
}

impl PointMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            59 => Some(Self::Delete),
            60 => Some(Self::SortZ),
            61 => Some(Self::SortDist),
            62 => Some(Self::Dist),
            63 => Some(Self::Value),
            64 => Some(Self::Size),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileMenuAction {
    New = 0,
    Open = 1,
    Reload = 2,
    Save = 3,
    SaveAs = 4,
    SnapDir = 5,
    SnapGray = 6,
    MovieMont = 7,
    Tiff = 8,
    Extract = 9,
    Process = 10,
    SaveInfo = 11,
    Quit = 12,
}

impl FileMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::New),
            1 => Some(Self::Open),
            2 => Some(Self::Reload),
            3 => Some(Self::Save),
            4 => Some(Self::SaveAs),
            5 => Some(Self::SnapDir),
            6 => Some(Self::SnapGray),
            7 => Some(Self::MovieMont),
            8 => Some(Self::Tiff),
            9 => Some(Self::Extract),
            10 => Some(Self::Process),
            11 => Some(Self::SaveInfo),
            12 => Some(Self::Quit),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WriteMenuAction {
    Imod = 13,
    Wimp = 14,
    Nff = 15,
    Synu = 16,
}

impl WriteMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            13 => Some(Self::Imod),
            14 => Some(Self::Wimp),
            15 => Some(Self::Nff),
            16 => Some(Self::Synu),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditMenuAction {
    Grain = 17,
    Angles = 18,
    ScaleBar = 19,
    SaveDock = 20,
    ReopenDock = 21,
    Prefs = 22,
}

impl EditMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            17 => Some(Self::Grain),
            18 => Some(Self::Angles),
            19 => Some(Self::ScaleBar),
            20 => Some(Self::SaveDock),
            21 => Some(Self::ReopenDock),
            22 => Some(Self::Prefs),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelMenuAction {
    Header = 23,
    Offsets = 24,
    Clean = 25,
}

impl ModelMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            23 => Some(Self::Header),
            24 => Some(Self::Offsets),
            25 => Some(Self::Clean),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectMenuAction {
    New = 26,
    Delete = 27,
    Color = 28,
    Type = 29,
    Info = 30,
    Move = 31,
    Clean = 32,
    Fixz = 33,
    FillIn = 34,
    Flatten = 35,
    SortDist = 36,
    Renumber = 37,
    Combine = 38,
    ListToSel = 39,
}

impl ObjectMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            26 => Some(Self::New),
            27 => Some(Self::Delete),
            28 => Some(Self::Color),
            29 => Some(Self::Type),
            30 => Some(Self::Info),
            31 => Some(Self::Move),
            32 => Some(Self::Clean),
            33 => Some(Self::Fixz),
            34 => Some(Self::FillIn),
            35 => Some(Self::Flatten),
            36 => Some(Self::SortDist),
            37 => Some(Self::Renumber),
            38 => Some(Self::Combine),
            39 => Some(Self::ListToSel),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceMenuAction {
    New = 40,
    Goto = 41,
    Move = 42,
    Delete = 43,
    Sort = 44,
}

impl SurfaceMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            40 => Some(Self::New),
            41 => Some(Self::Goto),
            42 => Some(Self::Move),
            43 => Some(Self::Delete),
            44 => Some(Self::Sort),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditImageMenuAction {
    Process = 65,
    Colormap = 66,
    Reload = 67,
    Flip = 68,
    FillCache = 69,
    Filler = 70,
}

impl EditImageMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            65 => Some(Self::Process),
            66 => Some(Self::Colormap),
            67 => Some(Self::Reload),
            68 => Some(Self::Flip),
            69 => Some(Self::FillCache),
            70 => Some(Self::Filler),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageMenuAction {
    Graph = 71,
    Slicer = 72,
    LinkSlice = 73,
    Tumbler = 74,
    Modv = 75,
    Zap = 76,
    Xyz = 77,
    Pixel = 78,
    Locator = 79,
    MultiZ = 80,
    Isosurface = 81,
}

impl ImageMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            71 => Some(Self::Graph),
            72 => Some(Self::Slicer),
            73 => Some(Self::LinkSlice),
            74 => Some(Self::Tumbler),
            75 => Some(Self::Modv),
            76 => Some(Self::Zap),
            77 => Some(Self::Xyz),
            78 => Some(Self::Pixel),
            79 => Some(Self::Locator),
            80 => Some(Self::MultiZ),
            81 => Some(Self::Isosurface),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HelpMenuAction {
    Controls = 82,
    Man = 83,
    Menus = 84,
    HotKey = 85,
    About = 86,
}

impl HelpMenuAction {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            82 => Some(Self::Controls),
            83 => Some(Self::Man),
            84 => Some(Self::Menus),
            85 => Some(Self::HotKey),
            86 => Some(Self::About),
            _ => None,
        }
    }

    pub const fn to_raw(self) -> i32 {
        self as i32
    }
}

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
    pub fn file_slot(&mut self, item: FileMenuAction, b: &mut dyn InfoMenuBoundary) {
        let s = b.state();
        if s.forbid_level != 0 {
            if item == FileMenuAction::Quit {
                b.call("imod_quit", &[]);
            }
            return;
        }
        match item {
            FileMenuAction::New => {
                if !(s.doing_initial_load && s.model_changed) {
                    b.call("undo.clearUnits", &[]);
                    b.call("createNewModel", &[]);
                }
            }
            FileMenuAction::Reload | FileMenuAction::Open => {
                if !(s.doing_initial_load && s.model_changed) {
                    b.call("imod_info_forbid", &[]);
                    b.call("imod_info_input", &[]);
                    b.call("releaseKeyboard", &[]);
                    b.call(
                        if item == FileMenuAction::Open {
                            "openModel"
                        } else {
                            "openModel.reload"
                        },
                        &[],
                    );
                    b.call("imod_info_enable", &[]);
                }
            }
            FileMenuAction::Save => {
                if !s.doing_initial_load {
                    b.call("imod_info_forbid", &[]);
                    b.call("imod_info_input", &[]);
                    b.call("releaseKeyboard", &[]);
                    b.call("SaveModel", &[]);
                    b.call("imod_info_enable", &[]);
                }
            }
            FileMenuAction::SaveAs => {
                if !s.doing_initial_load {
                    b.call("imod_info_forbid", &[]);
                    b.call("imod_info_input", &[]);
                    b.call("releaseKeyboard", &[]);
                    b.call("SaveasModel", &[]);
                    b.call("imod_info_enable", &[]);
                }
            }
            FileMenuAction::MovieMont => b.call("imodMovieConDialog", &[]),
            FileMenuAction::SnapDir => {
                b.call("imod_info_forbid", &[]);
                b.call("imod_info_input", &[]);
                b.call("releaseKeyboard", &[]);
                b.call("b3dSetSnapDirectory", &[]);
                b.call("imod_info_enable", &[]);
            }
            FileMenuAction::SnapGray => {
                let a = &mut self.m_actions[FileMenuAction::SnapGray.to_raw() as usize];
                a.checked = !a.checked;
                b.call("convertSnap", &[a.checked as i32]);
            }
            FileMenuAction::Tiff => {
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
            FileMenuAction::Extract | FileMenuAction::Process => {
                b.call("imod_info_forbid", &[]);
                b.call("imod_info_input", &[]);
                b.call("releaseKeyboard", &[]);
                b.call(
                    if item == FileMenuAction::Extract {
                        "extract"
                    } else {
                        "processFile"
                    },
                    &[],
                );
                b.call("imod_info_enable", &[]);
            }
            FileMenuAction::SaveInfo => b.call("wprintWriteFile", &[]),
            FileMenuAction::Quit => b.call("imod_quit", &[]),
            _ => {}
        }
    }
    /// `InfoWindow::fileWriteSlot`.
    pub fn file_write_slot(&mut self, item: WriteMenuAction, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        b.call("imod_info_forbid", &[]);
        b.call("imod_info_input", &[]);
        match item {
            WriteMenuAction::Imod => b.call("imodWrite", &[]),
            WriteMenuAction::Wimp => b.call("imod_to_wmod", &[]),
            WriteMenuAction::Nff => b.call("imod_to_nff", &[]),
            WriteMenuAction::Synu => b.call("imod_to_synu", &[]),
            _ => {}
        }
        b.call("imod_info_enable", &[]);
    }
    /// `InfoWindow::editSlot`.
    pub fn edit_slot(&mut self, item: EditMenuAction, b: &mut dyn InfoMenuBoundary) {
        match item {
            EditMenuAction::Grain => b.call("fineGrainOpen", &[]),
            EditMenuAction::Angles => b.call("slicerAnglesOpen", &[]),
            EditMenuAction::ScaleBar => b.call("scaleBarOpen", &[]),
            EditMenuAction::SaveDock => b.call("imodDialogManager.saveStackStateToSettings", &[]),
            EditMenuAction::ReopenDock => b.call("imodDialogManager.restoreStackFromSettings", &[]),
            EditMenuAction::Prefs => b.call("ImodPrefs.editPrefs", &[]),
            _ => {}
        }
    }
    /// `InfoWindow::editModelSlot`.
    pub fn edit_model_slot(&mut self, item: ModelMenuAction, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            ModelMenuAction::Header => b.call("openModelEdit", &[]),
            ModelMenuAction::Offsets => b.call("openModelOffset", &[]),
            ModelMenuAction::Clean => {
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
    pub fn edit_object_slot(&mut self, item: ObjectMenuAction, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            ObjectMenuAction::New => {
                b.call("inputNewObject", &[]);
                b.call("imod_object_edit", &[]);
                b.call("imod_info_setobjcolor", &[]);
                b.call("imodvObjedNewView", &[]);
            }
            ObjectMenuAction::Delete => {
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
            ObjectMenuAction::Color => {
                b.call("imod_info_forbid", &[]);
                b.call("imod_info_input", &[]);
                b.call("imod_info_enable", &[]);
                b.call("imod_object_color", &[]);
            }
            ObjectMenuAction::Type => {
                b.call("imod_object_edit", &[]);
                b.call("imod_draw_window", &[]);
                b.call("imod_info_setobjcolor", &[]);
            }
            ObjectMenuAction::Move => {
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
            ObjectMenuAction::ListToSel => {
                if let Some(text) =
                    b.input_text("List of objects to select (comma-separated ranges)")
                {
                    b.call("parselist", &[text.len() as i32]);
                    b.call("imodSelectionListClear", &[]);
                    b.call("imodSelectionListAdd.parsed", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            ObjectMenuAction::Combine => {
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
            ObjectMenuAction::Info => b.call("objectInfo", &[]),
            ObjectMenuAction::Clean => {
                b.call("imodDeleteContour.empty", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodSelectionListClear", &[]);
                b.call("imodDraw.rethink", &[]);
            }
            ObjectMenuAction::Fixz => {
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
            ObjectMenuAction::Flatten => {
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
            ObjectMenuAction::FillIn => {
                if self.last_fillin < 2 {
                    self.last_fillin = b.ask_forever("Fill in Z in all contours?");
                }
                if self.last_fillin != 0 {
                    b.call("imodFillInContourZ.all", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            ObjectMenuAction::SortDist => {
                if self.last_sortdist < 2 {
                    self.last_sortdist = b.ask_forever("Sort points by distance?");
                }
                if self.last_sortdist != 0 {
                    b.call("imodContourSort3D.all", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            ObjectMenuAction::Renumber => {
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
    pub fn edit_surface_slot(&mut self, item: SurfaceMenuAction, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            SurfaceMenuAction::New => b.call("inputNewSurface", &[]),
            SurfaceMenuAction::Goto => b.call("imodContEditSurf", &[]),
            SurfaceMenuAction::Move => b.call("imodContEditMoveDialog.surface", &[]),
            SurfaceMenuAction::Delete => {
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
            SurfaceMenuAction::Sort => self.sort_contours(b, 1),
            _ => {}
        }
    }
    /// `InfoWindow::editContourSlot`.
    pub fn edit_contour_slot(&mut self, item: ContourMenuAction, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            ContourMenuAction::New => b.call("inputNewContour", &[]),
            ContourMenuAction::Delete => b.call("inputDeleteContour", &[]),
            ContourMenuAction::Move => b.call("imodContEditMoveDialog.contour", &[]),
            ContourMenuAction::Sort => self.sort_contours(b, 0),
            ContourMenuAction::Auto => {
                b.call("autox_open", &[]);
                b.call("imod_info_setocp", &[]);
            }
            // The C handles `ECONTOUR_MENU_TYPE` in the *contour* switch
            // (info_menu.cpp:958) and `EPOINT_MENU_SIZE` in the *point* switch
            // (:1192).  They call the same function but live in different
            // switches, and the contour switch sends id 64 to `default: break`.
            // This arm had merged them, so the contour slot answered a point-menu
            // id the C ignores.  `edit_point_slot` already handles `Size`.
            ContourMenuAction::Type => b.call("imodContEditSurf", &[]),
            ContourMenuAction::Info => b.call("contourStatistics", &[]),
            ContourMenuAction::Break => b.call("imodContEditBreakOpen", &[]),
            ContourMenuAction::FixZ => {
                b.call("imodContourBreakByZ.current", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodSelectionListClear", &[]);
                b.call("imodDraw.mod", &[]);
            }
            ContourMenuAction::Join => b.call("imodContEditJoinOpen", &[]),
            ContourMenuAction::Invert => {
                b.call("undo.contourDataChg", &[]);
                b.call("imodel_contour_invert", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodDraw.mod", &[]);
            }
            ContourMenuAction::Copy => b.call("openContourCopyDialog", &[]),
            ContourMenuAction::LoopBack => {
                b.call("undo.contourDataChg", &[]);
                b.call("imodPointAppend.loopback", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodDraw.mod", &[]);
            }
            ContourMenuAction::FillIn => {
                b.call("imodFillInContourZ", &[]);
                b.call("undo.finishUnit", &[]);
                b.call("imodDraw.mod", &[]);
            }
            _ => {}
        }
    }
    /// `InfoWindow::editPointSlot`.
    pub fn edit_point_slot(&mut self, item: PointMenuAction, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            PointMenuAction::Delete => b.call("inputDeletePoint", &[]),
            PointMenuAction::SortDist => {
                if b.state().mouse_model {
                    b.call("undo.contourDataChg", &[]);
                    b.call("imodContourSort3D", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            PointMenuAction::SortZ => {
                if b.state().mouse_model {
                    b.call("undo.contourDataChg", &[]);
                    b.call("imodel_contour_sortz", &[]);
                    b.call("undo.finishUnit", &[]);
                    b.call("imodDraw.mod", &[]);
                }
            }
            PointMenuAction::Dist => b.call("pointDistanceReport", &[]),
            PointMenuAction::Value => b.call("ivwGetFileValue.report", &[]),
            PointMenuAction::Size => b.call("imodContEditSurf", &[]),
            _ => {}
        }
    }
    /// `InfoWindow::editImageSlot`.
    pub fn edit_image_slot(&mut self, item: EditImageMenuAction, b: &mut dyn InfoMenuBoundary) {
        if b.state().forbid_level != 0 {
            return;
        }
        match item {
            EditImageMenuAction::Process => b.call("inputIProcOpen", &[]),
            EditImageMenuAction::Colormap => b.call("imod_cmap.change", &[]),
            EditImageMenuAction::Reload => b.call("imodImageScaleDialog", &[]),
            EditImageMenuAction::Flip => {
                b.call("undo.clearUnits", &[]);
                b.call("vbCleanupVBD", &[]);
                b.call("ivwFlip", &[]);
                b.call("ivwCheckWildFlag", &[]);
                b.call("imodDraw.image_xyz_mod", &[]);
            }
            EditImageMenuAction::FillCache => {
                if b.state().vm_size != 0 {
                    b.call("imodCacheFill", &[])
                } else {
                    b.call("wprint.cache_not_active", &[])
                }
            }
            EditImageMenuAction::Filler => {
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
    pub fn image_slot(&mut self, item: ImageMenuAction, b: &mut dyn InfoMenuBoundary) {
        let s = b.state();
        if s.forbid_level != 0 || s.doing_initial_load {
            return;
        }
        match item {
            ImageMenuAction::Graph => {
                if !s.fake_image && !s.rgb_store {
                    b.call("xgraphOpen", &[])
                }
            }
            ImageMenuAction::Slicer => b.call("slicerOpen", &[0]),
            ImageMenuAction::LinkSlice => b.call("setupLinkedSlicers", &[]),
            ImageMenuAction::Tumbler => {
                if !s.rgb_store {
                    b.call("xtumOpen", &[])
                }
            }
            ImageMenuAction::Locator => b.call("locatorOpen", &[]),
            ImageMenuAction::Modv => {
                b.call("imod_autosave", &[]);
                b.call("imodv_open", &[]);
            }
            ImageMenuAction::Zap => b.call("imod_zap_open", &[0]),
            ImageMenuAction::MultiZ => b.call("imod_zap_open", &[1]),
            ImageMenuAction::Xyz => b.call("xxyz_open", &[]),
            ImageMenuAction::Pixel => {
                if !s.fake_image {
                    b.call("open_pixelview", &[])
                }
            }
            ImageMenuAction::Isosurface => {
                if !s.fake_image && !s.rgb_store {
                    b.call("imodv_open", &[]);
                    b.call("imodvIsosurfaceEditDialog", &[1]);
                }
            }
            _ => {}
        }
    }
    /// `InfoWindow::helpSlot`.
    pub fn help_slot(&mut self, item: HelpMenuAction, b: &mut dyn InfoMenuBoundary) {
        match item {
            HelpMenuAction::Man => b.call("imodShowHelpPage.3dmod", &[]),
            HelpMenuAction::Menus => b.call("imodShowHelpPage.menus", &[]),
            HelpMenuAction::Controls => b.call("imodShowHelpPage.infowin", &[]),
            HelpMenuAction::HotKey => b.call("imodShowHelpPage.keyboard", &[]),
            HelpMenuAction::About => {
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
        w.file_slot(FileMenuAction::Save, &mut n);
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
        w.image_slot(ImageMenuAction::Slicer, &mut n);
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
