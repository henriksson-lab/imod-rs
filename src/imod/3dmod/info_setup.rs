//! Translation of `IMOD/3dmod/info_setup.cpp` and `info_setup.h`.
//!
//! The original class owns the 3dmod information window.  Qt construction,
//! process launching, and calls into the separately paired `info_menu.cpp`
//! remain deliberately visible in [`InfoSetupNative`]; the source-owned menu,
//! resize, timer, process, and window state lives here.
#![allow(dead_code)]

use crate::imod::libimod::imodel::Imod;
use crate::imod::three_dmod::form_info::{InfoControls, InfoNativeBoundary};

pub const INFO_MIN_LINES: f32 = 4.;
pub const INFO_STARTING_LINES: f32 = 5.;

/// Items in the `info_setup.h` anonymous enum, preserving the source order.
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfoMenuId {
    FileNew,
    FileOpen,
    FileReload,
    FileSave,
    FileSaveas,
    FileSnapdir,
    FileSnapgray,
    FileMoviemont,
    FileTiff,
    FileExtract,
    FileProcess,
    FileSaveinfo,
    FileQuit,
    FwriteImod,
    FwriteWimp,
    FwriteNff,
    FwriteSynu,
    EditGrain,
    EditAngles,
    EditScalebar,
    EditSaveDock,
    EditReopenDock,
    EditPrefs,
    EmodelHeader,
    EmodelOffsets,
    EmodelClean,
    EobjectNew,
    EobjectDelete,
    EobjectColor,
    EobjectType,
    EobjectInfo,
    EobjectMove,
    EobjectClean,
    EobjectFixz,
    EobjectFillin,
    EobjectFlatten,
    EobjectSortdist,
    EobjectRenumber,
    EobjectCombine,
    EobjectListToSel,
    EsurfaceNew,
    EsurfaceGoto,
    EsurfaceMove,
    EsurfaceDelete,
    EsurfaceSort,
    EcontourNew,
    EcontourDelete,
    EcontourMove,
    EcontourSort,
    EcontourAuto,
    EcontourType,
    EcontourInfo,
    EcontourBreak,
    EcontourJoin,
    EcontourFixz,
    EcontourInvert,
    EcontourCopy,
    EcontourLoopback,
    EcontourFillin,
    EpointDelete,
    EpointSortz,
    EpointSortdist,
    EpointDist,
    EpointValue,
    EpointSize,
    EimageProcess,
    EimageColormap,
    EimageReload,
    EimageFlip,
    EimageFillcache,
    EimageFiller,
    ImageGraph,
    ImageSlicer,
    ImageLinkslice,
    ImageTumbler,
    ImageModv,
    ImageZap,
    ImageXyz,
    ImagePixel,
    ImageLocator,
    ImageMultiz,
    ImageIsosurface,
    HelpControls,
    HelpMan,
    HelpMenus,
    HelpHotkey,
    HelpAbout,
}
pub const LAST_MENU_ID: usize = InfoMenuId::HelpAbout as usize + 1;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InfoAction {
    pub text: String,
    pub enabled: bool,
    pub checkable: bool,
    pub checked: bool,
    pub shortcut: Option<String>,
}

/// View fields read by `InfoWindow::manageMenus` and `openSelectedWindows`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InfoViewState {
    pub fake_image: bool,
    pub rgb_store: bool,
    pub multi_file_z: i32,
    pub no_readable_image: bool,
    pub pyr_cache: bool,
    pub piece_list: bool,
    pub xy_bin: i32,
    pub vm_size: i32,
    pub num_times: i32,
    pub colormap_image: bool,
    pub reloadable: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProcessRequest {
    pub executable: String,
    pub arguments: Vec<String>,
    pub kind: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProcessError {
    #[default]
    Unknown,
    FailedToStart,
}

/// Direct Qt/viewer boundary.  The defaults are intentional for a headless
/// translation test; production UI adapters implement the actual callbacks.
pub trait InfoSetupNative: InfoNativeBoundary {
    fn view_state(&self) -> InfoViewState {
        InfoViewState::default()
    }
    fn iproc_is_open(&self) -> bool {
        false
    }
    fn iproc_busy(&self) -> bool {
        false
    }
    fn meshing_busy(&self) -> bool {
        false
    }
    fn write_status(&mut self, _: &str) {}
    fn open_plugin(&mut self, _: i32) {}
    fn open_plugin_by_name(&mut self, _: &str) {}
    fn open_all_external_plugins(&mut self) {}
    fn menu_slot(&mut self, _: &str, _: i32) {}
    fn hot_slider_key(&self) -> i32 {
        -1
    }
    fn hot_slider_enabled(&self) -> bool {
        false
    }
    fn info_ctrl_press(&mut self, _: i32) {}
    fn control_key(&mut self, _: bool, _: i32, _: bool) {}
    fn quit(&mut self) {}
    fn dialog_master_changed(&mut self) {}
    fn dialog_activated(&mut self) {}
    fn dialog_hide(&mut self) {}
    fn dialog_show(&mut self) {}
    fn dialog_manage_staying_on_top(&mut self) {}
    fn set_stay_on_top(&mut self, _: bool) {}
    fn model_view_menu_active(&mut self, _: bool) {}
    fn raise_window(&mut self) {}
    fn start_timer(&mut self, _: i32) -> i32 {
        1
    }
    fn kill_timer(&mut self, _: i32) {}
    fn resize_window(&mut self, _: i32, _: i32) {}
    fn move_window(&mut self, _: i32, _: i32) {}
    fn current_window_size(&self) -> (i32, i32) {
        (454, 500)
    }
    fn control_size(&self) -> (i32, i32) {
        (454, 376)
    }
    fn control_size_hint(&self) -> (i32, i32) {
        self.control_size()
    }
    fn status_height(&self) -> i32 {
        100
    }
    fn set_status_minimum_height(&mut self, _: i32) {}
    fn info_input(&mut self) {}
    fn top_zap_is_ready(&self) -> bool {
        true
    }
    fn start_process(&mut self, _: ProcessRequest) {}
    fn remove_file(&mut self, _: &str) {}
    fn current_image_path(&self, _: i32) -> String {
        String::new()
    }
    fn ifd_path(&self) -> String {
        String::new()
    }
    fn current_saved_model_file(&self) -> Option<String> {
        None
    }
    fn current_object_index(&self) -> i32 {
        -1
    }
}

/// Exact source-owned `InfoWindow` state, replacing Qt pointers with owned
/// Rust state and route identifiers.
#[derive(Clone, Debug)]
pub struct InfoWindow {
    pub actions: [InfoAction; LAST_MENU_ID],
    pub info_controls: InfoControls,
    pub status_text: String,
    pub minimized: bool,
    pub top_timer_id: i32,
    pub auto_timer_id: i32,
    pub info_timer_id: i32,
    pub old_font_height: i32,
    pub trimvol_process: Option<ProcessRequest>,
    pub trimvol_output: String,
    pub process_com_root: String,
    pub trimvol_type: i32,
    pub imodinfo_process: Option<ProcessRequest>,
    pub resized_height: i32,
    pub target_height: i32,
    pub target_move_x: i32,
    pub target_move_y: i32,
    pub staying_on_top: bool,
    pub device_pixel_ratio: f32,
    pub width: i32,
    pub height: i32,
    pub model_name: String,
    pub image_name: String,
    pub shown: bool,
}

impl InfoWindow {
    /// `InfoWindow::InfoWindow`.
    pub fn new(native: &mut dyn InfoSetupNative) -> Self {
        let mut out = Self {
            actions: std::array::from_fn(|_| InfoAction {
                enabled: true,
                ..InfoAction::default()
            }),
            info_controls: InfoControls::new(native),
            status_text: String::new(),
            minimized: false,
            top_timer_id: 0,
            auto_timer_id: 0,
            info_timer_id: 0,
            old_font_height: 16,
            trimvol_process: None,
            trimvol_output: String::new(),
            process_com_root: String::new(),
            trimvol_type: 0,
            imodinfo_process: None,
            resized_height: -1,
            target_height: 0,
            target_move_x: -30000,
            target_move_y: -30000,
            staying_on_top: false,
            device_pixel_ratio: 0.,
            width: 454,
            height: 500,
            model_name: String::new(),
            image_name: String::new(),
            shown: false,
        };
        out.install_actions();
        out.actions[InfoMenuId::FileSnapgray as usize].checkable = true;
        out
    }

    /// Source menu construction, condensed only in the repeated Qt action installation.
    pub fn install_actions(&mut self) {
        const NAMES: [&str; LAST_MENU_ID] = [
            "New Model",
            "Open Model",
            "Reload Model",
            "Save Model",
            "Save Model As...",
            "Set Snap Dir...",
            "Gray TIF Snaps",
            "Movie/Montage...",
            "Memory to TIF...",
            "Extract File...",
            "Process File...",
            "Save Info Text...",
            "Quit",
            "Imod",
            "Wimp",
            "NFF",
            "Synu",
            "Fine Grain...",
            "Angles...",
            "Scale Bar...",
            "Save Dock",
            "Reopen Dock",
            "Options...",
            "Header...",
            "Offsets...",
            "Clean",
            "New",
            "Delete",
            "Color...",
            "Type...",
            "Info",
            "Move...",
            "Clean",
            "Break by Z",
            "Fill in Z",
            "Flatten",
            "Sort by Dist",
            "Renumber...",
            "Combine",
            "Select by #...",
            "New",
            "Go To...",
            "Move...",
            "Delete",
            "Sort",
            "New",
            "Delete",
            "Move...",
            "Sort",
            "Auto...",
            "Type...",
            "Info",
            "Break...",
            "Join...",
            "Break by Z",
            "Invert",
            "Copy...",
            "Loopback",
            "Fill in Z",
            "Delete",
            "Sort by Z",
            "Sort by dist",
            "Distance",
            "Value",
            "Size...",
            "Process...",
            "Colormap",
            "Reload...",
            "Flip/Rotate",
            "Fill Cache",
            "Cache Filler...",
            "Graph",
            "Slicer",
            "Linked Slicers",
            "Tumbler",
            "Model View",
            "Zap",
            "XYZ",
            "Pixel View",
            "Locator",
            "Multi-Z",
            "Isosurface",
            "Controls",
            "Man Page",
            "Menus",
            "Hot Keys",
            "About",
        ];
        for (index, text) in NAMES.iter().enumerate() {
            self.actions[index].text = (*text).into();
        }
        self.actions[InfoMenuId::FileSave as usize].shortcut = Some("S".into());
        self.actions[InfoMenuId::EsurfaceNew as usize].shortcut = Some("Shift+N".into());
        self.actions[InfoMenuId::EcontourNew as usize].shortcut = Some("N".into());
        self.actions[InfoMenuId::EcontourDelete as usize].shortcut = Some("Shift+D".into());
        self.actions[InfoMenuId::EimageProcess as usize].shortcut = Some("Ctrl+P".into());
        self.actions[InfoMenuId::ImageZap as usize].shortcut = Some("Z".into());
        self.actions[InfoMenuId::ImageXyz as usize].shortcut = Some("Ctrl+X".into());
        self.actions[InfoMenuId::ImageSlicer as usize].shortcut = Some("\\".into());
        self.actions[InfoMenuId::ImageModv as usize].shortcut = Some("V".into());
        self.actions[InfoMenuId::ImageGraph as usize].shortcut = Some("Shift+G".into());
        self.actions[InfoMenuId::ImageIsosurface as usize].shortcut = Some("Shift+U".into());
    }

    /// `InfoWindow::setInitialHeights`.
    pub fn set_initial_heights(
        &mut self,
        for_screen_change: bool,
        native: &mut dyn InfoSetupNative,
    ) {
        native.info_input();
        let edit_height = native.status_height();
        let new_edit = (INFO_STARTING_LINES * self.old_font_height as f32).round() as i32;
        let hint = native.control_size_hint();
        let actual = native.control_size();
        let del_widget = (actual.1 - hint.1).max(0);
        native.set_status_minimum_height(
            (INFO_MIN_LINES * self.old_font_height as f32).round() as i32
        );
        self.info_timer_id = native.start_timer(if for_screen_change { 250 } else { 10 });
        self.resize_to_height(
            self.height - (edit_height - new_edit) - del_widget,
            0,
            native,
        );
    }

    /// `InfoWindow::resizeToHeight`.
    pub fn resize_to_height(
        &mut self,
        mut new_height: i32,
        trim_width: i32,
        native: &mut dyn InfoSetupNative,
    ) {
        let hint = native.control_size_hint();
        let actual = native.control_size();
        let edit_height = native.status_height();
        let new_edit = (INFO_STARTING_LINES * self.old_font_height as f32).round() as i32;
        let mut delh = (actual.1 - hint.1).max(0);
        if edit_height < new_edit {
            delh -= new_edit - edit_height;
        }
        new_height -= delh;
        if self.info_timer_id != 0 {
            self.target_height = new_height;
        } else {
            for _ in 0..5 {
                self.height = new_height;
                native.resize_window(100, new_height);
                native.info_input();
                if self.height == new_height {
                    break;
                }
            }
        }
        let _ = trim_width;
        self.resized_height = new_height;
    }
    /// `InfoWindow::doOrSetupMove`.
    pub fn do_or_setup_move(&mut self, x: i32, y: i32, native: &mut dyn InfoSetupNative) {
        if self.info_timer_id != 0 {
            self.target_move_x = x;
            self.target_move_y = y;
        } else {
            native.move_window(x, y);
            native.info_input();
        }
    }
    /// `InfoWindow::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self) {}
    /// `InfoWindow::getResizedHeight`.
    pub fn get_resized_height(&self) -> i32 {
        self.resized_height
    }
    /// `InfoWindow::getStayingOnTop`.
    pub fn get_staying_on_top(&self) -> bool {
        self.staying_on_top
    }
    /// `InfoWindow::extendInfoTimer`.
    pub fn extend_info_timer(&mut self, native: &mut dyn InfoSetupNative) {
        if self.info_timer_id != 0 {
            native.kill_timer(self.info_timer_id);
            self.info_timer_id = native.start_timer(250);
        }
    }

    /// `InfoWindow::changeEvent`, with Qt event dispatch exposed as arguments.
    pub fn change_event(
        &mut self,
        font_change: bool,
        activation: bool,
        closing: bool,
        native: &mut dyn InfoSetupNative,
    ) {
        if closing {
            return;
        }
        if activation {
            native.dialog_activated();
        }
        if !font_change {
            return;
        }
        self.info_controls.set_font_dependent_widths(native);
        self.set_font_dependent_widths();
        native.info_input();
        let (widget_height, hint_height) = (native.control_size().1, native.control_size_hint().1);
        self.resize_to_height(self.height + hint_height - widget_height, 4, native);
    }
    /// `InfoWindow::pluginSlot`.
    pub fn plugin_slot(&mut self, item: i32, native: &mut dyn InfoSetupNative) {
        native.open_plugin(item);
    }
    /// `InfoWindow::keyPressEvent`.
    pub fn key_press_event(&mut self, key: i32, control: bool, native: &mut dyn InfoSetupNative) {
        if key == native.hot_slider_key() && native.hot_slider_enabled() {
            native.info_ctrl_press(1);
        }
        if key != 27 {
            native.control_key(false, key, control);
        }
    }
    /// `InfoWindow::keyReleaseEvent`.
    pub fn key_release_event(&mut self, key: i32, control: bool, native: &mut dyn InfoSetupNative) {
        if key == native.hot_slider_key() {
            native.info_ctrl_press(0);
        }
        native.control_key(true, key, control);
    }
    /// `InfoWindow::closeEvent`.
    pub fn close_event(&mut self, native: &mut dyn InfoSetupNative) {
        native.quit();
    }
    /// `InfoWindow::resizeEvent`.
    pub fn resize_event(&mut self, info_initial_setup: bool, native: &mut dyn InfoSetupNative) {
        if !info_initial_setup {
            native.dialog_master_changed();
        }
    }
    /// `InfoWindow::moveEvent`.
    pub fn move_event(&mut self, info_initial_setup: bool, native: &mut dyn InfoSetupNative) {
        if !info_initial_setup {
            native.dialog_master_changed();
            self.extend_info_timer(native);
        }
    }
    /// `InfoWindow::event`.
    pub fn event(
        &mut self,
        minimized_or_hide: bool,
        normal_or_show: bool,
        activation: bool,
        native: &mut dyn InfoSetupNative,
    ) {
        if activation {
            native.model_view_menu_active(false);
        }
        if minimized_or_hide && !self.minimized {
            self.minimized = true;
            native.dialog_hide();
        } else if normal_or_show && self.minimized {
            self.minimized = false;
            native.dialog_show();
        }
    }

    /// `InfoWindow::manageMenus`.
    pub fn manage_menus(&mut self, native: &mut dyn InfoSetupNative) {
        let vi = native.view_state();
        let image_ok = !vi.fake_image && !vi.rgb_store;
        let extract_ok = image_ok
            && vi.multi_file_z <= 0
            && !vi.no_readable_image
            && !vi.pyr_cache
            && !vi.piece_list;
        self.actions[InfoMenuId::FileTiff as usize].enabled = vi.rgb_store;
        self.actions[InfoMenuId::FileExtract as usize].enabled = extract_ok;
        self.actions[InfoMenuId::FileProcess as usize].enabled =
            extract_ok && native.iproc_is_open() && vi.xy_bin == 1;
        self.actions[InfoMenuId::EimageFillcache as usize].enabled =
            vi.vm_size != 0 || vi.num_times > 0 || vi.pyr_cache;
        self.actions[InfoMenuId::EimageFiller as usize].enabled =
            (vi.vm_size != 0 || vi.num_times > 0) && !vi.pyr_cache;
        self.actions[InfoMenuId::ImageLinkslice as usize].enabled = vi.num_times > 0;
        self.actions[InfoMenuId::ImageIsosurface as usize].enabled = image_ok;
        self.actions[InfoMenuId::EcontourAuto as usize].enabled = image_ok;
        if !image_ok {
            self.actions[InfoMenuId::ImageGraph as usize].enabled = false;
        }
        self.actions[InfoMenuId::EimageProcess as usize].enabled =
            image_ok && !vi.colormap_image && !vi.pyr_cache;
        if !image_ok || vi.colormap_image {
            self.actions[InfoMenuId::ImageTumbler as usize].enabled = false;
            self.info_controls.set_float(-1);
        }
        self.actions[InfoMenuId::ImagePixel as usize].enabled = !vi.fake_image;
        self.actions[InfoMenuId::EimageFlip as usize].enabled =
            !native.iproc_busy() && !vi.colormap_image && !vi.pyr_cache;
        self.actions[InfoMenuId::EimageReload as usize].enabled = !native.iproc_busy()
            && image_ok
            && !vi.colormap_image
            && !vi.no_readable_image
            && !vi.pyr_cache;
        self.actions[InfoMenuId::FileReload as usize].enabled = vi.reloadable;
        self.actions[InfoMenuId::EobjectDelete as usize].enabled = !native.meshing_busy();
        self.actions[InfoMenuId::EobjectRenumber as usize].enabled = !native.meshing_busy();
    }

    /// `InfoWindow::extract`; Zap/Slicer command selection and the native save chooser remain direct callbacks.
    pub fn extract(
        &mut self,
        command: Option<(String, i32, bool)>,
        output: Option<String>,
        imod_dir: Option<&str>,
        native: &mut dyn InfoSetupNative,
    ) {
        let Some((command, time_lock, rotate_vol)) = command else {
            return;
        };
        let Some(output) = output else { return };
        let Some(dir) = imod_dir else {
            native.write_status("\x07Cannot run trimvol; IMOD_DIR not defined and IMOD is not installed in a standard location.\n");
            return;
        };
        self.trimvol_output = output;
        self.trimvol_type = i32::from(rotate_vol);
        self.actions[InfoMenuId::FileExtract as usize].enabled = false;
        self.actions[InfoMenuId::FileProcess as usize].enabled = false;
        let mut arguments: Vec<String> = command.split_whitespace().map(str::to_owned).collect();
        if !rotate_vol && !arguments.is_empty() {
            arguments.remove(0);
        }
        arguments.push(self.get_adjusted_file_path(time_lock, native));
        arguments.push(self.trimvol_output.clone());
        let executable = if rotate_vol {
            format!("{dir}/bin/rotatevol")
        } else {
            format!("{dir}/bin/trimvol")
        };
        let request = ProcessRequest {
            executable,
            arguments,
            kind: self.trimvol_type,
        };
        self.trimvol_process = Some(request.clone());
        native.start_process(request);
    }
    /// `InfoWindow::trimvolExited`.
    pub fn trimvol_exited(
        &mut self,
        exit_code: i32,
        normal_exit: bool,
        stdout: &[String],
        stderr: &[String],
        native: &mut dyn InfoSetupNative,
    ) {
        self.actions[InfoMenuId::FileExtract as usize].enabled = true;
        self.manage_menus(native);
        if self.trimvol_process.is_none() {
            return;
        }
        if exit_code == 0 && normal_exit {
            native.write_status(&format!("{} created.\n", self.trimvol_output));
            if self.trimvol_type > 1 {
                native.remove_file(&(self.process_com_root.clone() + "com"));
                native.remove_file(&(self.process_com_root.clone() + "log"));
            }
        } else {
            native.write_status(if self.trimvol_type > 1 {
                "\x07Image processing failed; check log for details\n"
            } else if self.trimvol_type != 0 {
                "\x07Rotatevol failed.\n"
            } else {
                "\x07Trimvol failed.\n"
            });
            for line in stdout {
                if line.starts_with("ERROR:") {
                    native.write_status(line);
                }
            }
        }
        if self.trimvol_type < 2 {
            for line in stderr {
                native.write_status(&format!("err:\n{line}\n"));
            }
        }
        self.trimvol_process = None;
    }
    /// `InfoWindow::trimvolError`.
    pub fn trimvol_error(&mut self, error: ProcessError, native: &mut dyn InfoSetupNative) {
        if error != ProcessError::FailedToStart {
            return;
        }
        native.write_status(if self.trimvol_type > 1 { "\x07Could not start submfg to process image - is IMOD fully installed and is python on the PATH?\n" } else if self.trimvol_type != 0 { "\x07Could not start rotatevol - is IMOD fully installed?\n" } else { "\x07Could not start trimvol - is python on the PATH?\n" });
        if self.trimvol_type > 1 {
            native.remove_file(&(self.process_com_root.clone() + "com"));
        }
        self.actions[InfoMenuId::FileExtract as usize].enabled = true;
        self.manage_menus(native);
    }
    /// `InfoWindow::processFile`; command-file writing is a direct native boundary, while request state is retained.
    pub fn process_file(
        &mut self,
        process_list: &[String],
        command: Option<(String, i32, bool)>,
        output: Option<String>,
        process_com_root: String,
        native: &mut dyn InfoSetupNative,
    ) {
        if process_list.is_empty() || process_list.iter().any(|entry| entry.starts_with("Cannot")) {
            native.write_status(
                "\x07There is no usable command list available from the processing dialog.\n",
            );
            return;
        }
        let Some(output) = output else { return };
        self.trimvol_output = output;
        self.process_com_root = process_com_root;
        let mut arguments = vec![self.process_com_root.clone() + "com"];
        if let Some((prefix, _, _)) = command {
            arguments.insert(0, prefix);
        }
        self.actions[InfoMenuId::FileExtract as usize].enabled = false;
        self.actions[InfoMenuId::FileProcess as usize].enabled = false;
        self.trimvol_type = 2;
        let request = ProcessRequest {
            executable: "submfg".into(),
            arguments,
            kind: 2,
        };
        self.trimvol_process = Some(request.clone());
        native.start_process(request);
    }
    /// `InfoWindow::putIMODandPythonOnPath`; environment mutation belongs to the Qt/process adapter.
    pub fn put_imod_and_python_on_path(&mut self, _: &str, _: i32) {}
    /// `InfoWindow::getAdjustedFilePath`.
    pub fn get_adjusted_file_path(&self, time_lock: i32, native: &dyn InfoSetupNative) -> String {
        let file_path = native.current_image_path(time_lock);
        let ifd = native.ifd_path();
        if ifd.is_empty() {
            file_path
        } else {
            format!("{}/{}", ifd.trim_end_matches('/'), file_path)
        }
    }
    /// `InfoWindow::objectInfo`.
    pub fn object_info(&mut self, native: &mut dyn InfoSetupNative) {
        let Some(filename) = native.current_saved_model_file() else {
            native.write_status("\x07Cannot run imodinfo - no current model file exists.\n");
            return;
        };
        let request = ProcessRequest {
            executable: "imodinfo".into(),
            arguments: vec![
                "-o".into(),
                (native.current_object_index() + 1).to_string(),
                filename,
            ],
            kind: 0,
        };
        self.imodinfo_process = Some(request.clone());
        self.actions[InfoMenuId::EobjectInfo as usize].enabled = false;
        native.start_process(request);
        native.write_status("Running imodinfo on current object\n");
    }
    /// `InfoWindow::imodinfoExited`.
    pub fn imodinfo_exited(
        &mut self,
        exit_code: i32,
        normal_exit: bool,
        output: &[String],
        stderr: &[String],
        native: &mut dyn InfoSetupNative,
    ) {
        self.actions[InfoMenuId::EobjectInfo as usize].enabled = true;
        if self.imodinfo_process.is_none() {
            return;
        }
        if exit_code != 0 || !normal_exit {
            native.write_status("\x07imodinfo failed.\n");
        }
        for line in output {
            let trimmed = line.trim();
            if !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.contains("Light")
                && !trimmed.contains("Color")
                && !trimmed.contains("Contours =")
                && !trimmed.contains("Shininess")
            {
                native.write_status(&(trimmed.to_owned() + "\n"));
            }
        }
        for line in stderr {
            native.write_status(&format!("err:\n{line}\n"));
        }
        self.imodinfo_process = None;
    }
    /// `InfoWindow::imodinfoError`.
    pub fn imodinfo_error(&mut self, error: ProcessError, native: &mut dyn InfoSetupNative) {
        if error == ProcessError::FailedToStart {
            native.write_status("\x07Could not start imodinfo.\n");
            self.actions[InfoMenuId::EobjectInfo as usize].enabled = true;
        }
    }
    /// `InfoWindow::setupAutoContrast`.
    pub fn setup_auto_contrast(&mut self, native: &mut dyn InfoSetupNative) {
        self.auto_timer_id = native.start_timer(10);
    }
    /// `InfoWindow::keepOnTop`.
    pub fn keep_on_top(&mut self, state: bool, native: &mut dyn InfoSetupNative) {
        self.staying_on_top = state;
        native.set_stay_on_top(state);
        native.dialog_manage_staying_on_top();
    }
    /// `InfoWindow::timerEvent`.
    pub fn timer_event(&mut self, timer_id: i32, native: &mut dyn InfoSetupNative) {
        if self.auto_timer_id != 0 && timer_id == self.auto_timer_id {
            if !native.top_zap_is_ready() {
                return;
            }
            native.kill_timer(self.auto_timer_id);
            native.info_input();
            let (mean, sd) = InfoNativeBoundary::auto_contrast_targets(native);
            InfoNativeBoundary::info_auto_contrast(native, mean, sd);
            self.auto_timer_id = 0;
        } else if self.info_timer_id != 0 && timer_id == self.info_timer_id {
            native.kill_timer(self.info_timer_id);
            self.info_timer_id = 0;
            if self.target_height > 0 {
                let height = self.target_height;
                self.resize_to_height(height, 0, native);
            }
            self.target_height = 0;
            if self.target_move_x >= -29000 && self.target_move_y >= -29000 {
                native.move_window(self.target_move_x, self.target_move_y);
            }
            self.target_move_x = -30000;
            self.target_move_y = -30000;
        } else {
            native.raise_window();
        }
    }
    /// `InfoWindow::openSelectedWindows`.
    pub fn open_selected_windows(
        &mut self,
        keys: Option<&str>,
        model_view_open: bool,
        native: &mut dyn InfoSetupNative,
    ) {
        let Some(keys) = keys else { return };
        let vi = native.view_state();
        let image_ok = !vi.fake_image && !vi.rgb_store;
        for key in keys.chars() {
            match key {
                'a' => native.menu_slot("editContour", InfoMenuId::EcontourAuto as i32),
                'b' => native.menu_slot("editContour", InfoMenuId::EcontourBreak as i32),
                'c' => native.menu_slot("editContour", InfoMenuId::EcontourCopy as i32),
                'e' => native.menu_slot("edit", InfoMenuId::EditScalebar as i32),
                'f' if vi.vm_size != 0 || vi.num_times > 0 => {
                    native.menu_slot("editImage", InfoMenuId::EimageFiller as i32)
                }
                'g' if image_ok => native.menu_slot("image", InfoMenuId::ImageGraph as i32),
                'h' => native.menu_slot("editModel", InfoMenuId::EmodelHeader as i32),
                'j' => native.menu_slot("editContour", InfoMenuId::EcontourJoin as i32),
                'l' => native.menu_slot("editObject", InfoMenuId::EobjectColor as i32),
                'm' if image_ok && !model_view_open => {
                    native.menu_slot("image", InfoMenuId::ImageLocator as i32)
                }
                'n' => native.menu_slot("file", InfoMenuId::FileMoviemont as i32),
                'o' => native.menu_slot("editModel", InfoMenuId::EmodelOffsets as i32),
                'p' if image_ok => native.menu_slot("editImage", InfoMenuId::EimageProcess as i32),
                'r' if image_ok => native.menu_slot("editImage", InfoMenuId::EimageReload as i32),
                's' => native.menu_slot("editSurface", InfoMenuId::EsurfaceGoto as i32),
                't' => native.menu_slot("editObject", InfoMenuId::EobjectType as i32),
                'u' if image_ok => native.menu_slot("image", InfoMenuId::ImageTumbler as i32),
                'v' => native.menu_slot("editContour", InfoMenuId::EcontourMove as i32),
                'x' if !vi.fake_image => native.menu_slot("image", InfoMenuId::ImagePixel as i32),
                'z' => native.menu_slot("image", InfoMenuId::ImageMultiz as i32),
                'A' => native.menu_slot("edit", InfoMenuId::EditAngles as i32),
                'F' => native.open_plugin_by_name("Bead Fixer"),
                'G' => native.menu_slot("edit", InfoMenuId::EditGrain as i32),
                'P' => native.open_all_external_plugins(),
                '4' if image_ok => native.open_plugin_by_name("Drawing Tools"),
                '5' => native.open_plugin_by_name("Interpolator"),
                '6' => native.open_plugin_by_name("Bead Helper"),
                'T' if image_ok => native.open_plugin_by_name("Line Track"),
                '9' => native.open_plugin_by_name("Bead Fixer2"),
                '1' => native.menu_slot("setMmode", 1),
                _ => {}
            }
        }
    }
}

/// `imod_info_open`; the source globals are explicit ownership in Rust.
pub fn imod_info_open(
    native: &mut dyn InfoSetupNative,
    image_name: Option<&str>,
    model_name: Option<&str>,
) -> InfoWindow {
    let mut window = InfoWindow::new(native);
    window.image_name = truncate_name(image_name.unwrap_or(" "), 23);
    window.info_controls.set_image_name(&window.image_name);
    window.shown = true;
    window.set_initial_heights(false, native);
    window.model_name = model_name.unwrap_or(" ").into();
    window.info_controls.set_model_name(&window.model_name);
    window
}

/// `MaintainModelName`.  `Imod` has no separate upstream `fileName` field in
/// this safe representation, so its NUL-terminated `name` storage is updated.
pub fn maintain_model_name(
    mod_: &mut Imod,
    imod_filename: &str,
    window: &mut InfoWindow,
    native: &mut dyn InfoSetupNative,
) {
    window.model_name = imod_filename.into();
    window.info_controls.set_model_name(imod_filename);
    mod_.name.fill(0);
    for (to, from) in mod_.name.iter_mut().zip(imod_filename.bytes()) {
        *to = from as i8;
    }
    window.manage_menus(native);
}

/// `truncate_name`.
pub fn truncate_name(name: &str, limit: usize) -> String {
    if name.len() <= limit {
        return name.into();
    }
    let prefix_end = name
        .char_indices()
        .nth(limit.saturating_sub(1))
        .map(|(at, _)| at)
        .unwrap_or(name.len());
    format!("{}...", &name[..prefix_end])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native {
        log: Vec<String>,
        view: InfoViewState,
        next_timer: i32,
    }
    impl InfoNativeBoundary for Native {
        fn image_state(&self) -> crate::imod::three_dmod::form_info::InfoImageState {
            Default::default()
        }
        fn get_float_flags(&self) -> (i32, i32, i32) {
            (0, 0, 0)
        }
        fn info_new_xyz(&mut self, _: [i32; 3]) {}
        fn info_new_ocp(&mut self, _: i32, _: i32, _: i32) {}
        fn info_new_bw(&mut self, _: i32, _: i32, _: i32) {}
        fn info_new_lh(&mut self, _: i32, _: i32, _: i32) {}
        fn info_mm_selected(&mut self, _: i32) {}
        fn info_float(&mut self, _: i32) {}
        fn input_raise_windows(&mut self) {}
        fn keep_on_top(&mut self, _: bool) {}
        fn info_subset(&mut self, _: i32) {}
        fn info_t_ramps(&mut self, _: i32) {}
        fn auto_contrast_targets(&self) -> (i32, i32) {
            (1, 2)
        }
        fn info_auto_contrast(&mut self, _: i32, _: i32) {}
        fn input_undo_redo(&mut self, _: bool) {}
        fn retranslate_ui(&mut self) {}
    }
    impl InfoSetupNative for Native {
        fn view_state(&self) -> InfoViewState {
            self.view
        }
        fn start_timer(&mut self, _: i32) -> i32 {
            self.next_timer += 1;
            self.next_timer
        }
        fn menu_slot(&mut self, kind: &str, item: i32) {
            self.log.push(format!("{kind}:{item}"))
        }
    }
    #[test]
    fn menu_enablement_follows_source_predicates() {
        let mut n = Native::default();
        n.view.rgb_store = true;
        let mut w = InfoWindow::new(&mut n);
        w.manage_menus(&mut n);
        assert!(w.actions[InfoMenuId::FileTiff as usize].enabled);
        assert!(!w.actions[InfoMenuId::ImageIsosurface as usize].enabled);
    }
    #[test]
    fn info_timer_defers_resize_and_move() {
        let mut n = Native::default();
        let mut w = InfoWindow::new(&mut n);
        w.info_timer_id = 7;
        w.resize_to_height(300, 0, &mut n);
        w.do_or_setup_move(2, 3, &mut n);
        w.timer_event(7, &mut n);
        assert_eq!(w.height, 300);
        assert_eq!((w.target_move_x, w.target_move_y), (-30000, -30000));
    }
    #[test]
    fn open_keys_routes_original_menu_id() {
        let mut n = Native::default();
        let mut w = InfoWindow::new(&mut n);
        w.open_selected_windows(Some("ah"), false, &mut n);
        assert_eq!(
            n.log,
            vec![
                format!("editContour:{}", InfoMenuId::EcontourAuto as i32),
                format!("editModel:{}", InfoMenuId::EmodelHeader as i32)
            ]
        );
    }
    #[test]
    fn truncation_has_source_length() {
        assert_eq!(truncate_name("abcdefghijklmnopqrstuvwxyz", 5), "abcd...");
    }
}
