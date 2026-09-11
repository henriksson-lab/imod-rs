//! `IMOD/Etomo/src/etomo/process/ImodProcess.java`.
//!
//! This is the source-shaped 3dmod command and message owner.  A live 3dmod child and
//! its Qt/X11 IPC are deliberately an explicit boundary: command construction and all
//! state/message encoding are implemented here, but [`ImodProcess::open`] records the
//! command and returns [`ImodProcessError::ViewerBoundary`] instead of claiming that a
//! viewer was opened.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::{BaseManager, get_imod_bin_path};
use crate::imod::etomo::r#type::axis_id::AxisID;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const MESSAGE_OPEN_MODEL: &str = "1";
pub const MESSAGE_SAVE_MODEL: &str = "2";
pub const MESSAGE_VIEW_MODEL: &str = "3";
pub const MESSAGE_CLOSE: &str = "4";
pub const MESSAGE_RAISE: &str = "5";
pub const MESSAGE_OPEN_MODEL_VIEW: &str = "3";
pub const MESSAGE_MODEL_MODE: &str = "6";
pub const MESSAGE_OPEN_KEEP_BW: &str = "7";
pub const MESSAGE_OPEN_BEADFIXER: &str = "8";
pub const MESSAGE_ONE_ZAP_OPEN: &str = "9";
pub const MESSAGE_RUBBERBAND: &str = "10";
pub const MESSAGE_OBJ_PROPERTIES: &str = "11";
pub const MESSAGE_NEWOBJ_PROPERTIES: &str = "12";
pub const MESSAGE_SLICER_ANGLES: &str = "13";
pub const MESSAGE_PLUGIN_MESSAGE: &str = "14";
pub const MESSAGE_MORE_OBJ_PROPERTIES: &str = "16";
pub const MESSAGE_INTERPOLATION: &str = "18";
pub const MESSAGE_OPEN_DIALOG: &str = "19";
pub const BEAD_FIXER_PLUGIN: &str = "Bead Fixer";
pub const BF_MESSAGE_OPEN_LOG: &str = "1";
pub const BF_MESSAGE_REREAD_LOG: &str = "2";
pub const BF_MESSAGE_NEW_CONTOURS: &str = "3";
pub const BF_MESSAGE_AUTO_CENTER: &str = "4";
pub const BF_MESSAGE_DIAMETER: &str = "5";
pub const BF_MESSAGE_MODE: &str = "6";
pub const BF_MESSAGE_SKIP_LIST: &str = "7";
pub const BF_MESSAGE_DELETE_ALL_SECTIONS: &str = "8";
pub const BF_MESSAGE_REMOVE_SKIP_LIST: &str = "9";
pub const MESSAGE_ON: &str = "1";
pub const MESSAGE_OFF: &str = "0";
pub const MESSAGE_STOP_LISTENING: &str = "\n";
pub const RUBBERBAND_RESULTS_STRING: &str = "Rubberband:";
pub const SLICER_ANGLES_RESULTS_STRING1: &str = "Slicer";
pub const SLICER_ANGLES_RESULTS_STRING2: &str = "angles:";
pub const IMOD_SEND_EVENT_STRING: &str = "imodsendevent returned:";
pub const CONTINUOUS_TAG: &str = "ETOMO INFO:";
pub const REQUEST_TAG: &str = "REQUEST";
pub const STOP_LISTENING_REQUEST: &str = "STOP LISTENING";
pub const DEFAULT_BINNING: i32 = 1;
const CIRCLE: i32 = 1;
const SURF_CONT_POINT_DIALOG: &str = "s";

/// Source dependency `Run3dmodMenuOptions`.  This value contains exactly the queries
/// ImodProcess makes; its owning source unit can replace it without changing command
/// construction.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Run3dmodMenuOptions {
    pub bin_by_2: bool,
    pub allow_binning_in_z: bool,
    pub startup_window: bool,
}
impl Run3dmodMenuOptions {
    pub fn is_bin_by_2(self) -> bool {
        self.bin_by_2
    }
    pub fn is_allow_binning_in_z(self) -> bool {
        self.allow_binning_in_z
    }
    pub fn is_startup_window(self) -> bool {
        self.startup_window
    }
}

/// Java `SystemProcessException` / `IOException` result at this unit's boundary.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ImodProcessError {
    ViewerBoundary { command: Vec<String> },
    NotRunning,
    NoWindowId,
    Io(String),
}
impl fmt::Display for ImodProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ViewerBoundary { command } => {
                write!(f, "3dmod execution boundary: {}", command.join(" "))
            }
            Self::NotRunning => f.write_str("3dmod is not running."),
            Self::NoWindowId => f.write_str("No window ID available for imod"),
            Self::Io(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for ImodProcessError {}

/// Java nested `WindowOpenOption`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowOpenOption {
    window_key: String,
    imodv: bool,
}
impl WindowOpenOption {
    pub const OPTION: &'static str = "-E";
    pub fn imodv_objects() -> Self {
        Self::new("O", true)
    }
    pub fn isosurface() -> Self {
        Self::new("U", true)
    }
    pub fn object_list() -> Self {
        Self::new("L", true)
    }
    pub fn model_edit() -> Self {
        Self::new("M", true)
    }
    pub fn new(window_key: impl Into<String>, imodv: bool) -> Self {
        Self {
            window_key: window_key.into(),
            imodv,
        }
    }
    pub fn is_imodv(&self) -> bool {
        self.imodv
    }
}
impl fmt::Display for WindowOpenOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.window_key)
    }
}

/// Java nested `BeadFixerMode` typesafe enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BeadFixerMode {
    SeedMode,
    GapMode,
    ResidualMode,
    PatchTrackingResidualMode,
}
impl BeadFixerMode {
    pub fn get_value(self) -> &'static str {
        match self {
            Self::SeedMode => "0",
            Self::GapMode => "1",
            Self::ResidualMode => "2",
            Self::PatchTrackingResidualMode => "3",
        }
    }
}

/// Java `ContinuousListenerTarget`; the callback preserves its listener contract.
pub trait ContinuousListenerTarget: Send + Sync {
    fn get_continuous_message(&self, message: &str, axis_id: Option<AxisID>);
}

/// Java private nested `Stderr`: the independent read cursors are represented by a
/// registration map and queue indices, just as in the source.
#[derive(Default)]
pub struct Stderr {
    registration: HashMap<i32, isize>,
    quick_listener_queue: Vec<String>,
    continuous_listener_queue: VecDeque<String>,
    request_queue: VecDeque<String>,
    received_interrupted_exception: bool,
    reg_id: i32,
}
impl Stderr {
    pub const EXPECTED_REGISTRANTS: usize = 2;
    pub const PURGE_SIZE: usize = 10;
    pub fn register(&mut self) -> i32 {
        self.reg_id += 1;
        self.registration.insert(self.reg_id, -1);
        self.reg_id
    }
    pub fn add(&mut self, input: impl Into<String>) {
        self.quick_listener_queue.push(input.into());
        self.purge_quick_listener_queue();
    }
    pub fn accept_stderr(&mut self, message: impl Into<String>) {
        let message = message.into();
        if message.starts_with(REQUEST_TAG) && message.contains(STOP_LISTENING_REQUEST) {
            self.request_queue.push_back(message);
        } else if message.starts_with(CONTINUOUS_TAG) {
            self.continuous_listener_queue.push_back(message);
        } else {
            self.add(message);
        }
    }
    pub fn get_quick_message(&mut self, reg_id: i32) -> Option<String> {
        let index = self.registration.get_mut(&reg_id)?;
        if *index < self.quick_listener_queue.len() as isize - 1 {
            *index += 1;
            self.quick_listener_queue.get(*index as usize).cloned()
        } else {
            None
        }
    }
    pub fn get_continuous_message(&mut self) -> Option<String> {
        self.continuous_listener_queue.pop_front()
    }
    pub fn get_request_message(&mut self) -> Option<String> {
        self.request_queue.pop_front()
    }
    pub fn purge_quick_listener_queue(&mut self) {
        if self.registration.len() < Self::EXPECTED_REGISTRANTS
            || self.quick_listener_queue.len() < Self::PURGE_SIZE
        {
            return;
        }
        let read_by_all = self.registration.values().copied().min().unwrap_or(-1);
        if read_by_all >= (Self::PURGE_SIZE / 2) as isize {
            let removed = read_by_all as usize + 1;
            self.quick_listener_queue.drain(..removed);
            for index in self.registration.values_mut() {
                *index -= removed as isize;
            }
        }
    }
}

/// Java package-private `QuickListenerQueueTestWrapper`.
pub struct QuickListenerQueueTestWrapper {
    stderr: Stderr,
}
impl QuickListenerQueueTestWrapper {
    pub fn new() -> Self {
        Self {
            stderr: Stderr::default(),
        }
    }
    pub fn get_expected_registrants(&self) -> usize {
        Stderr::EXPECTED_REGISTRANTS
    }
    pub fn register(&mut self) -> i32 {
        self.stderr.register()
    }
    pub fn get_purge_size(&self) -> usize {
        Stderr::PURGE_SIZE
    }
    pub fn add(&mut self, input: impl Into<String>) {
        self.stderr.add(input)
    }
    pub fn get_quick_message(&mut self, reg_id: i32) -> Option<String> {
        self.stderr.get_quick_message(reg_id)
    }
    pub fn purge(&mut self) {
        self.stderr.purge_quick_listener_queue()
    }
}
impl fmt::Display for QuickListenerQueueTestWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.stderr.quick_listener_queue)
    }
}

/// Java `ImodProcess` fields.  `last_command` is the explicit boundary record for the
/// untranslated `InteractiveSystemProgram` / child thread pair.
pub struct ImodProcess {
    dataset_name: String,
    model_name: String,
    window_id: String,
    swap_yz: bool,
    model_view: bool,
    use_modv: bool,
    output_window_id: bool,
    open_with_model: bool,
    working_directory: Option<PathBuf>,
    binning: i32,
    binning_xy: i32,
    send_arguments: Vec<String>,
    dataset_name_array: Option<Vec<String>>,
    frames: bool,
    piece_list_file_name: Option<String>,
    axis_id: Option<AxisID>,
    flip: bool,
    manager: Option<&'static dyn BaseManager>,
    beadfixer_diameter_set: bool,
    window_open_option_list: Option<Vec<WindowOpenOption>>,
    debug: bool,
    subdir_name: Option<String>,
    open_zap: bool,
    tilt_file: Option<String>,
    continuous_listener_target: Option<Arc<dyn ContinuousListenerTarget>>,
    file_list: Option<Vec<PathBuf>>,
    load_as_integers: bool,
    montage_separation: bool,
    model_name_list: Option<Vec<String>>,
    suppress_save_query: bool,
    listen_to_stdin: bool,
    stderr: Arc<Mutex<Stderr>>,
    message_sender_reg_id: i32,
    stderr_reg_id: i32,
    running: bool,
    last_command: Option<Vec<String>>,
}

impl ImodProcess {
    fn base(manager: Option<&'static dyn BaseManager>, axis_id: Option<AxisID>) -> Self {
        let mut stderr = Stderr::default();
        let message_sender_reg_id = stderr.register();
        let stderr_reg_id = stderr.register();
        Self {
            dataset_name: String::new(),
            model_name: String::new(),
            window_id: String::new(),
            swap_yz: false,
            model_view: false,
            use_modv: false,
            output_window_id: true,
            open_with_model: true,
            working_directory: None,
            binning: DEFAULT_BINNING,
            binning_xy: DEFAULT_BINNING,
            send_arguments: vec![],
            dataset_name_array: None,
            frames: false,
            piece_list_file_name: None,
            axis_id,
            flip: false,
            manager,
            beadfixer_diameter_set: false,
            window_open_option_list: None,
            debug: false,
            subdir_name: None,
            open_zap: false,
            tilt_file: None,
            continuous_listener_target: None,
            file_list: None,
            load_as_integers: false,
            montage_separation: false,
            model_name_list: None,
            suppress_save_query: false,
            listen_to_stdin: !cfg!(windows),
            stderr: Arc::new(Mutex::new(stderr)),
            message_sender_reg_id,
            stderr_reg_id,
            running: false,
            last_command: None,
        }
    }
    pub fn new(manager: Option<&'static dyn BaseManager>, axis_id: Option<AxisID>) -> Self {
        Self::base(manager, axis_id)
    }
    pub fn new_dataset_file_type(
        manager: Option<&'static dyn BaseManager>,
        dataset: impl Into<String>,
        axis_id: Option<AxisID>,
    ) -> Self {
        let mut value = Self::base(manager, axis_id);
        value.dataset_name = dataset.into();
        value
    }
    pub fn new_dataset_file(
        manager: Option<&'static dyn BaseManager>,
        dataset: impl Into<String>,
        axis_id: Option<AxisID>,
        _file: Option<&Path>,
    ) -> Self {
        Self::new_dataset_file_type(manager, dataset, axis_id)
    }
    pub fn new_file_name(
        manager: Option<&'static dyn BaseManager>,
        file_name: impl Into<String>,
        axis_id: Option<AxisID>,
    ) -> Self {
        Self::new_dataset_file_type(manager, file_name, axis_id)
    }
    pub fn new_dataset_model_file(
        manager: Option<&'static dyn BaseManager>,
        dataset: impl Into<String>,
        model: impl Into<String>,
        file: Option<&Path>,
    ) -> Self {
        let mut value = Self::new_dataset_file(manager, dataset, None, file);
        value.model_name = model.into();
        value
    }
    pub fn new_dataset_array_model_file(
        manager: Option<&'static dyn BaseManager>,
        dataset_array: Vec<String>,
        model: impl Into<String>,
        _file: Option<&Path>,
    ) -> Self {
        let mut value = Self::base(manager, None);
        value.dataset_name_array = Some(dataset_array);
        value.model_name = model.into();
        value
    }
    pub fn new_dataset_array(
        manager: Option<&'static dyn BaseManager>,
        dataset_array: Vec<String>,
    ) -> Self {
        let mut value = Self::base(manager, None);
        value.dataset_name_array = Some(dataset_array);
        value
    }
    pub fn new_file_list(
        manager: Option<&'static dyn BaseManager>,
        file_list: Vec<PathBuf>,
    ) -> Self {
        let mut value = Self::base(manager, None);
        value.file_list = Some(file_list);
        value
    }
    pub fn set_suppress_save_query(&mut self) {
        self.suppress_save_query = true
    }
    pub fn set_dataset_name(&mut self, input: impl Into<String>) {
        self.dataset_name = input.into()
    }
    pub fn set_subdir_name(&mut self, input: impl Into<String>) {
        self.subdir_name = Some(input.into())
    }
    pub fn get_subdir_name(&self) -> Option<&str> {
        self.subdir_name.as_deref()
    }
    pub fn set_frames(&mut self, input: bool) {
        self.frames = input
    }
    pub fn set_piece_list_file_name(&mut self, input: impl Into<String>) {
        self.piece_list_file_name = Some(input.into())
    }
    pub fn set_montage_separation(&mut self) {
        self.montage_separation = true
    }
    pub fn set_model_name(&mut self, input: impl Into<String>) {
        self.model_name = input.into()
    }
    pub fn set_model_name_list(&mut self, input: Option<Vec<String>>) {
        self.model_name_list = input
    }
    pub fn set_working_directory(&mut self, input: Option<PathBuf>) {
        self.working_directory = input
    }
    pub fn set_load_as_integers(&mut self) {
        self.load_as_integers = true
    }
    pub fn set_open_with_model(&mut self, input: bool) {
        self.open_with_model = input
    }
    pub fn set_listen_to_stdin(&mut self, input: bool) {
        self.listen_to_stdin = input
    }
    pub fn calc_current_binning(binning: i32, menu_options: Run3dmodMenuOptions) -> i32 {
        (if binning == DEFAULT_BINNING {
            0
        } else {
            binning
        }) + if menu_options.bin_by_2 { 2 } else { 0 }
    }
    /// Java `open`: records the exact constructed command, then stops at real child/UI boundary.
    pub fn open(&mut self, menu_options: Run3dmodMenuOptions) -> Result<(), ImodProcessError> {
        if self.is_running() {
            return self.raise_3dmod();
        }
        self.window_id.clear();
        let command = self.build_command(menu_options);
        self.last_command = Some(command.clone());
        Err(ImodProcessError::ViewerBoundary { command })
    }
    /// Command body of Java `open`, separately observable because launch is a boundary.
    pub fn build_command(&self, menu_options: Run3dmodMenuOptions) -> Vec<String> {
        let mut command = vec![
            std::env::var("IMOD_3DMOD")
                .unwrap_or_else(|_| format!("{}3dmod", get_imod_bin_path().unwrap_or_default())),
        ];
        if self.output_window_id {
            command.push("-W".to_string());
        }
        if self.listen_to_stdin {
            command.push("-L".to_string());
        }
        if self.swap_yz {
            command.push("-Y".to_string());
        }
        if self.frames {
            command.push("-f".to_string());
        }
        if self.montage_separation {
            // `etomo.comscript.Utilities.MONTAGE_SEPARATION` is "-10".
            command.extend(["-o".into(), "-10,-10".into()]);
        }
        if let Some(piece) = &self.piece_list_file_name {
            if !piece.trim().is_empty() {
                command.extend(["-p".into(), piece.clone()]);
            }
        }
        if self.model_view {
            command.push("-V".into());
        }
        if self.open_zap {
            command.push("-Z".into());
        }
        if self.load_as_integers {
            command.extend(["-I".into(), "1".into()]);
        }
        if let Some(tilt) = &self.tilt_file {
            command.extend(["-a".into(), tilt.clone()]);
        }
        if self.use_modv {
            command.push("-view".into());
        }
        if self.binning > DEFAULT_BINNING
            || (menu_options.bin_by_2 && menu_options.allow_binning_in_z)
        {
            command.extend([
                "-B".into(),
                Self::calc_current_binning(self.binning, menu_options).to_string(),
            ]);
        }
        if self.binning_xy > DEFAULT_BINNING
            || (menu_options.bin_by_2 && !menu_options.allow_binning_in_z)
        {
            command.extend([
                "-b".into(),
                Self::calc_current_binning(self.binning_xy, menu_options).to_string(),
            ]);
        }
        if menu_options.startup_window {
            command.push("-O".into());
        }
        if let Some(options) = &self.window_open_option_list {
            let mut value = options.iter().map(ToString::to_string).collect::<String>();
            if self.suppress_save_query {
                value.push('2');
            }
            command.extend([WindowOpenOption::OPTION.into(), value]);
        } else if self.suppress_save_query {
            command.extend([WindowOpenOption::OPTION.into(), "2".into()]);
        }
        if !self.dataset_name.is_empty() {
            command.push(self.dataset_name.clone());
        }
        if let Some(names) = &self.dataset_name_array {
            for name in names {
                command.push(
                    self.subdir_name
                        .as_ref()
                        .map(|dir| Path::new(dir).join(name).to_string_lossy().into_owned())
                        .unwrap_or_else(|| name.clone()),
                );
            }
        }
        if let Some(files) = &self.file_list {
            for file in files {
                let name = file.file_name().unwrap_or(file.as_os_str());
                command.push(
                    self.subdir_name
                        .as_ref()
                        .map(|dir| Path::new(dir).join(name).to_string_lossy().into_owned())
                        .unwrap_or_else(|| name.to_string_lossy().into_owned()),
                );
            }
        }
        if self.open_with_model {
            if !self.model_name.is_empty() {
                command.push(self.model_name.clone());
            }
            if let Some(models) = &self.model_name_list {
                command.extend(models.iter().cloned());
            }
        }
        command
    }
    pub fn last_command(&self) -> Option<&[String]> {
        self.last_command.as_deref()
    }
    pub fn quit(&mut self) -> Result<(), ImodProcessError> {
        if self.is_running() {
            self.send(&[MESSAGE_CLOSE])
        } else {
            Ok(())
        }
    }
    pub fn disconnect(&mut self) -> Result<(), ImodProcessError> {
        if self.listen_to_stdin && self.is_running() {
            self.send_commands_no_wait(&[MESSAGE_STOP_LISTENING])
        } else {
            Ok(())
        }
    }
    pub fn is_running(&self) -> bool {
        self.running
    }
    pub fn set_open_model_message(&mut self, model: impl Into<String>) {
        let model = model.into();
        self.model_name = model.clone();
        self.send_arguments
            .extend([MESSAGE_OPEN_MODEL.into(), model]);
    }
    pub fn open_model(
        &mut self,
        model: impl Into<String>,
        _model_mode: bool,
    ) -> Result<(), ImodProcessError> {
        let model = model.into();
        self.model_name = model.clone();
        self.send(&[MESSAGE_OPEN_MODEL, &model, MESSAGE_MODEL_MODE])
    }
    pub fn set_open_model_preserve_contrast_message(&mut self, model: impl Into<String>) {
        self.send_arguments
            .extend([MESSAGE_OPEN_KEEP_BW.into(), model.into()]);
    }
    pub fn open_model_preserve_contrast(
        &mut self,
        model: impl Into<String>,
    ) -> Result<(), ImodProcessError> {
        let model = model.into();
        self.send(&[MESSAGE_OPEN_KEEP_BW, &model])
    }
    pub fn save_model(&mut self) -> Result<(), ImodProcessError> {
        self.send(&[MESSAGE_SAVE_MODEL])
    }
    pub fn view_model(&mut self) -> Result<(), ImodProcessError> {
        self.send(&[MESSAGE_VIEW_MODEL])
    }
    pub fn set_new_contours_message(&mut self, open: bool) {
        self.set_new_object_message(0, open, CIRCLE, 7, 0)
    }
    pub fn set_point_limit_message(&mut self, point_limit: i32) {
        self.set_more_object_properties_message(1, point_limit, -1, -1)
    }
    pub fn set_start_new_contours_at_new_z(&mut self) {
        self.set_more_object_properties_message(1, -1, 1, -1)
    }
    pub fn set_interpolation(&mut self, input: bool) {
        self.send_arguments.extend([
            MESSAGE_INTERPOLATION.into(),
            if input { "1" } else { "0" }.into(),
        ]);
    }
    pub fn open_surf_cont_point(&mut self) {
        self.send_arguments
            .extend([MESSAGE_OPEN_DIALOG.into(), SURF_CONT_POINT_DIALOG.into()]);
    }
    pub fn set_new_object_message(
        &mut self,
        object: i32,
        open: bool,
        symbol: i32,
        size: i32,
        size_3d: i32,
    ) {
        self.send_arguments.extend([
            MESSAGE_NEWOBJ_PROPERTIES.into(),
            object.to_string(),
            if open { "1" } else { "0" }.into(),
            symbol.to_string(),
            size.to_string(),
            size_3d.to_string(),
        ]);
    }
    pub fn set_more_object_properties_message(
        &mut self,
        object: i32,
        point_limit: i32,
        new_contour_in_new_z: i32,
        sphere_in_central_only: i32,
    ) {
        self.send_arguments.extend([
            MESSAGE_MORE_OBJ_PROPERTIES.into(),
            object.to_string(),
            point_limit.to_string(),
            new_contour_in_new_z.to_string(),
            sphere_in_central_only.to_string(),
        ]);
    }
    pub fn set_model_mode_message(&mut self) {
        self.send_arguments
            .extend([MESSAGE_MODEL_MODE.into(), "1".into()]);
    }
    pub fn model_mode(&mut self) -> Result<(), ImodProcessError> {
        self.send(&[MESSAGE_MODEL_MODE])
    }
    pub fn set_movie_mode_message(&mut self) {
        self.send_arguments
            .extend([MESSAGE_MODEL_MODE.into(), "0".into()]);
    }
    pub fn movie_mode(&mut self) -> Result<(), ImodProcessError> {
        self.send(&[MESSAGE_MODEL_MODE, "0"])
    }
    pub fn set_raise_3dmod_message(&mut self) {
        self.send_arguments.push(MESSAGE_RAISE.into())
    }
    pub fn raise_3dmod(&mut self) -> Result<(), ImodProcessError> {
        self.send(&[MESSAGE_RAISE])
    }
    pub fn set_open_zap_window_message(&mut self) {
        self.send_arguments.push(MESSAGE_ONE_ZAP_OPEN.into())
    }
    pub fn open_zap_window(&mut self) -> Result<(), ImodProcessError> {
        self.send(&[MESSAGE_ONE_ZAP_OPEN])
    }
    pub fn set_open_bead_fixer_message(&mut self) {
        self.send_arguments.push(MESSAGE_OPEN_BEADFIXER.into())
    }
    pub fn set_open_model_view(&mut self) {
        self.send_arguments.push(MESSAGE_OPEN_MODEL_VIEW.into())
    }
    pub fn set_skip_list(&mut self, skip_list: Option<&str>) {
        if let Some(value) = skip_list {
            self.add_plugin_message_value(BEAD_FIXER_PLUGIN, BF_MESSAGE_SKIP_LIST, value)
        } else {
            self.add_plugin_message(BEAD_FIXER_PLUGIN, BF_MESSAGE_REMOVE_SKIP_LIST)
        }
    }
    pub fn set_delete_all_sections(&mut self, on: bool) {
        self.add_plugin_message_value(
            BEAD_FIXER_PLUGIN,
            BF_MESSAGE_DELETE_ALL_SECTIONS,
            if on { "1" } else { "0" },
        )
    }
    pub fn set_beadfixer_diameter(&mut self, diameter: Option<i32>) {
        if let Some(diameter) = diameter {
            self.beadfixer_diameter_set = true;
            self.add_plugin_message_value(
                BEAD_FIXER_PLUGIN,
                BF_MESSAGE_DIAMETER,
                &diameter.to_string(),
            )
        }
    }
    pub fn set_auto_center(&mut self, input: bool) {
        self.add_plugin_message_value(
            BEAD_FIXER_PLUGIN,
            BF_MESSAGE_AUTO_CENTER,
            if input { MESSAGE_ON } else { MESSAGE_OFF },
        )
    }
    pub fn set_new_contours(&mut self, input: bool) {
        self.add_plugin_message_value(
            BEAD_FIXER_PLUGIN,
            BF_MESSAGE_NEW_CONTOURS,
            if input { MESSAGE_ON } else { MESSAGE_OFF },
        )
    }
    pub fn set_beadfixer_mode(&mut self, input: BeadFixerMode) {
        self.add_plugin_message_value(BEAD_FIXER_PLUGIN, BF_MESSAGE_MODE, input.get_value())
    }
    pub fn reopen_log(&mut self) -> Result<(), ImodProcessError> {
        self.send_plugin_message(BEAD_FIXER_PLUGIN, BF_MESSAGE_REREAD_LOG)
    }
    pub fn set_open_log(&mut self, input: impl Into<String>) {
        self.add_plugin_message_value(BEAD_FIXER_PLUGIN, BF_MESSAGE_OPEN_LOG, &input.into())
    }
    pub fn open_bead_fixer(&mut self) -> Result<(), ImodProcessError> {
        self.send(&[MESSAGE_OPEN_BEADFIXER])
    }
    pub fn get_rubberband_coordinates(&mut self) -> Result<Vec<String>, ImodProcessError> {
        self.request(&[MESSAGE_RUBBERBAND])
    }
    pub fn get_slicer_angles(&mut self) -> Result<Vec<String>, ImodProcessError> {
        self.request(&[MESSAGE_SLICER_ANGLES])
    }
    fn send_plugin_message(&mut self, plugin: &str, message: &str) -> Result<(), ImodProcessError> {
        self.send(&[MESSAGE_PLUGIN_MESSAGE, plugin, message])
    }
    fn add_plugin_message_value(&mut self, plugin: &str, message: &str, value: &str) {
        self.send_arguments.extend([
            MESSAGE_PLUGIN_MESSAGE.into(),
            plugin.into(),
            message.into(),
            value.into(),
        ]);
    }
    fn add_plugin_message(&mut self, plugin: &str, message: &str) {
        self.send_arguments
            .extend([MESSAGE_PLUGIN_MESSAGE.into(), plugin.into(), message.into()]);
    }
    pub fn send_messages(&mut self) -> Result<(), ImodProcessError> {
        if self.send_arguments.is_empty() {
            return Ok(());
        }
        let values = std::mem::take(&mut self.send_arguments);
        self.send(&values.iter().map(String::as_str).collect::<Vec<_>>())
    }
    fn send(&mut self, args: &[&str]) -> Result<(), ImodProcessError> {
        if !self.running {
            return Err(ImodProcessError::NotRunning);
        }
        if self.listen_to_stdin {
            self.send_commands(args)
        } else {
            self.imod_send_event(args).map(|_| ())
        }
    }
    fn request(&mut self, args: &[&str]) -> Result<Vec<String>, ImodProcessError> {
        if !self.running {
            return Err(ImodProcessError::NotRunning);
        }
        if self.listen_to_stdin {
            self.send_request(args)
        } else {
            self.imod_send_and_receive(args)
        }
    }
    pub fn imod_send_and_receive(
        &mut self,
        args: &[&str],
    ) -> Result<Vec<String>, ImodProcessError> {
        self.imod_send_event(args)?;
        Ok(vec![])
    }
    pub fn parse_error(line: &str, error_message: &mut Vec<String>) -> bool {
        let index = line.find("ERROR:").or_else(|| line.find("WARNING:"));
        if let Some(index) = index {
            error_message.push(line[index..].to_string());
            true
        } else {
            false
        }
    }
    fn imod_send_event(&mut self, args: &[&str]) -> Result<Vec<String>, ImodProcessError> {
        if self.window_id.is_empty() {
            return Err(ImodProcessError::NoWindowId);
        }
        let mut command = vec!["imodsendevent".to_string(), self.window_id.clone()];
        command.extend(args.iter().map(|value| (*value).to_string()));
        self.last_command = Some(command.clone());
        Err(ImodProcessError::ViewerBoundary { command })
    }
    fn send_request(&mut self, args: &[&str]) -> Result<Vec<String>, ImodProcessError> {
        self.send_commands(args)?;
        Ok(vec![])
    }
    fn send_commands(&mut self, args: &[&str]) -> Result<(), ImodProcessError> {
        if !self.running {
            return Err(ImodProcessError::NotRunning);
        }
        self.last_command = Some(args.iter().map(|value| (*value).to_string()).collect());
        Err(ImodProcessError::ViewerBoundary {
            command: self.last_command.clone().unwrap_or_default(),
        })
    }
    fn send_commands_no_wait(&mut self, args: &[&str]) -> Result<(), ImodProcessError> {
        self.send_commands(args)
    }
    pub fn process_request(&mut self) {
        if self.is_request_received() {
            let _ = self.disconnect();
        }
    }
    fn is_request_received(&mut self) -> bool {
        self.stderr.lock().unwrap().get_request_message().is_some()
    }
    pub fn get_dataset_name(&self) -> &str {
        &self.dataset_name
    }
    pub fn get_model_name(&self) -> &str {
        &self.model_name
    }
    pub fn get_window_id(&self) -> &str {
        &self.window_id
    }
    pub fn get_swap_yz(&self) -> bool {
        self.swap_yz
    }
    pub fn set_swap_yz(&mut self, input: bool) {
        self.swap_yz = input
    }
    pub fn is_model_view(&self) -> bool {
        self.model_view
    }
    pub fn set_model_view(&mut self, input: bool) {
        self.model_view = input
    }
    pub fn set_open_zap(&mut self) {
        self.open_zap = true
    }
    pub fn set_tilt_file(&mut self, input: impl Into<String>) {
        self.tilt_file = Some(input.into())
    }
    pub fn reset_tilt_file(&mut self) {
        self.tilt_file = None
    }
    pub fn is_use_modv(&self) -> bool {
        self.use_modv
    }
    pub fn set_use_modv(&mut self, input: bool) {
        self.use_modv = input
    }
    pub fn is_output_window_id(&self) -> bool {
        self.output_window_id
    }
    pub fn set_output_window_id(&mut self, input: bool) {
        self.output_window_id = input
    }
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input
    }
    pub fn set_binning(&mut self, input: i32) {
        self.binning = input.max(DEFAULT_BINNING)
    }
    pub fn set_binning_xy(&mut self, input: i32) {
        self.binning_xy = input.max(DEFAULT_BINNING)
    }
    pub fn param_string(&self) -> String {
        format!(
            ",datasetName={}, modelName={}, windowID={}, swapYZ={}, modelView={}, useModv={}, outputWindowID={}, binning={}",
            self.dataset_name,
            self.model_name,
            self.window_id,
            self.swap_yz,
            self.model_view,
            self.use_modv,
            self.output_window_id,
            self.binning
        )
    }
    pub fn add_window_open_option(&mut self, option: WindowOpenOption) {
        self.window_open_option_list
            .get_or_insert_with(Vec::new)
            .push(option)
    }
    pub fn set_continuous_listener_target(
        &mut self,
        target: Option<Arc<dyn ContinuousListenerTarget>>,
    ) {
        self.continuous_listener_target = target
    }
    /// Input hook for the future `InteractiveSystemProgram` bridge.  It retains source
    /// queue routing and allows listener integration without creating a fake child.
    pub fn accept_stderr(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.stderr.lock().unwrap().accept_stderr(message.clone());
        if message.starts_with(CONTINUOUS_TAG) {
            if let Some(target) = &self.continuous_listener_target {
                target.get_continuous_message(&message, self.axis_id);
            }
        }
    }
}
impl fmt::Display for ImodProcess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}[{}]",
            std::any::type_name::<Self>(),
            self.param_string()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn command_tracks_java_option_order() {
        let mut process = ImodProcess::new_dataset_file_type(None, "raw.st", Some(AxisID::Only));
        process.set_swap_yz(true);
        process.set_frames(true);
        process.set_binning(4);
        process.set_model_name("fid.mod");
        assert_eq!(
            process.build_command(Run3dmodMenuOptions::default()),
            vec![
                "3dmod", "-W", "-L", "-Y", "-f", "-B", "4", "raw.st", "fid.mod"
            ]
        );
    }
    #[test]
    fn buffered_plugin_messages_match_protocol() {
        let mut process = ImodProcess::new(None, None);
        process.set_auto_center(true);
        process.set_skip_list(None);
        assert_eq!(
            process.send_arguments,
            vec!["14", "Bead Fixer", "4", "1", "14", "Bead Fixer", "9"]
        );
    }
    #[test]
    fn quick_listener_purges_after_all_read() {
        let mut queue = QuickListenerQueueTestWrapper::new();
        let one = queue.register();
        let two = queue.register();
        for value in 0..10 {
            queue.add(value.to_string());
        }
        for _ in 0..6 {
            let _ = queue.get_quick_message(one);
            let _ = queue.get_quick_message(two);
        }
        queue.purge();
        assert_eq!(queue.get_quick_message(one), Some("6".to_string()));
    }
}
