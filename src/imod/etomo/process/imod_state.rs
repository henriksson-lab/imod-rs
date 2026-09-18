//! `IMOD/Etomo/src/etomo/process/ImodState.java`.
//!
//! Persistent per-viewer configuration.  The source delegates opening, messages, and
//! event transport to `ImodProcess`; that 3dmod boundary is deliberately represented
//! by `Option<Infallible>` here.  State setters/getters retain their Java semantics,
//! while viewer actions report the unavailable boundary rather than claiming a viewer
//! was launched.
#![allow(dead_code)]

use crate::imod::etomo::process::imod_process::{ImodProcess, Run3dmodMenuOptions};
use crate::imod::etomo::r#type::axis_id::AxisID;
use std::convert::Infallible;
use std::path::{Path, PathBuf};

pub const MODEL_MODE: i32 = -1;
pub const MOVIE_MODE: i32 = -2;
pub const MODEL_VIEW: i32 = -3;
pub const MODV: i32 = -4;
const DEFAULT_OPEN_WITH_MODEL: bool = false;
const DEFAULT_PRESERVE_CONTRAST: bool = false;
const DEFAULT_OPEN_BEAD_FIXER: bool = false;
const DEFAULT_OPEN_CONTOURS: bool = false;
const DEFAULT_FRAMES: bool = false;
const DEFAULT_BINNING: i32 = 1;

/// Java final `ImodState`.
pub struct ImodState {
    model_view: bool,
    use_modv: bool,
    axis_id: AxisID,
    model_name: Option<String>,
    mode: i32,
    swap_yz: bool,
    preserve_contrast: bool,
    open_bead_fixer: bool,
    open_contours: bool,
    start_new_contours_at_new_z: bool,
    point_limit: i32,
    set_auto_center: bool,
    auto_center: bool,
    new_contours: bool,
    manage_new_contours: bool,
    beadfixer_mode: Option<Infallible>,
    skip_list: Option<String>,
    using_mode: bool,
    allow_menu_binning_in_z: bool,
    no_menu_options: bool,
    initial_model_name: String,
    initial_mode: i32,
    initial_swap_yz: bool,
    /// Java final `process`.  Child/window transport remains the explicit
    /// boundary of `ImodProcess`, but configuration now reaches that Rust
    /// implementation instead of being discarded in this state owner.
    process: ImodProcess,
    file_name_array: Option<Vec<String>>,
    file_list: Option<Vec<PathBuf>>,
    warned_stale_file: bool,
    initial_mode_set: bool,
    initial_swap_yz_set: bool,
    manager: Option<Infallible>,
    log_name: Option<String>,
    debug: bool,
    delete_all_sections: Option<bool>,
    file_name: Option<String>,
    interpolation: Option<bool>,
    model_name_list: Option<Vec<String>>,
    open_surf_cont_point: bool,
    dataset_name: Option<String>,
    subdir_name: Option<String>,
    binning: i32,
    binning_xy: i32,
    frames: bool,
    piece_list_file_name: Option<String>,
    montage_separation: bool,
    load_as_integers: bool,
    suppress_save_query: bool,
    open_zap: bool,
    working_directory: Option<PathBuf>,
    window_open_options: Vec<Option<Infallible>>,
}

impl ImodState {
    /// Java public `ImodState(BaseManager, AxisID)`.
    pub fn new(manager: Option<Infallible>, axis_id: AxisID) -> Self {
        let mut state = Self {
            model_view: false,
            use_modv: false,
            axis_id,
            model_name: None,
            mode: MOVIE_MODE,
            swap_yz: false,
            preserve_contrast: false,
            open_bead_fixer: false,
            open_contours: false,
            start_new_contours_at_new_z: false,
            point_limit: -1,
            set_auto_center: false,
            auto_center: false,
            new_contours: false,
            manage_new_contours: false,
            beadfixer_mode: None,
            skip_list: None,
            using_mode: false,
            allow_menu_binning_in_z: false,
            no_menu_options: false,
            initial_model_name: String::new(),
            initial_mode: MOVIE_MODE,
            initial_swap_yz: false,
            process: ImodProcess::new(None, Some(axis_id)),
            file_name_array: None,
            file_list: None,
            warned_stale_file: false,
            initial_mode_set: false,
            initial_swap_yz_set: false,
            manager,
            log_name: None,
            debug: false,
            delete_all_sections: None,
            file_name: None,
            interpolation: None,
            model_name_list: None,
            open_surf_cont_point: false,
            dataset_name: None,
            subdir_name: None,
            binning: DEFAULT_BINNING,
            binning_xy: DEFAULT_BINNING,
            frames: DEFAULT_FRAMES,
            piece_list_file_name: None,
            montage_separation: false,
            load_as_integers: false,
            suppress_save_query: false,
            open_zap: false,
            working_directory: None,
            window_open_options: Vec::new(),
        };
        state.reset();
        state
    }
    /// Java `ImodState(BaseManager,int,AxisID)`.
    pub fn new_with_model_view_type(
        manager: Option<Infallible>,
        model_view_type: i32,
        axis_id: AxisID,
    ) -> Self {
        let mut state = Self::new(manager, axis_id);
        state.set_model_view_type(model_view_type);
        state
    }
    /// Java `ImodState(BaseManager,String,AxisID)`.
    pub fn new_with_file_name(
        manager: Option<Infallible>,
        file_name: Option<&str>,
        axis_id: AxisID,
    ) -> Self {
        let mut state = Self::new(manager, axis_id);
        state.dataset_name = file_name.map(str::to_owned);
        state
    }
    /// Java dataset/file-type constructor (FileType remains an untranslated declared type).
    pub fn new_with_dataset_file_type(
        manager: Option<Infallible>,
        dataset: Option<&str>,
        axis_id: AxisID,
        file_type: Option<Infallible>,
    ) -> Self {
        let _ = file_type;
        Self::new_with_file_name(manager, dataset, axis_id)
    }
    /// Java dataset/model-view/file-type constructor.
    pub fn new_with_dataset_model_view_file_type(
        manager: Option<Infallible>,
        dataset: Option<&str>,
        model_view_type: i32,
        axis_id: AxisID,
        file_type: Option<Infallible>,
    ) -> Self {
        let _ = file_type;
        let mut state = Self::new_with_file_name(manager, dataset, axis_id);
        state.set_model_view_type(model_view_type);
        state
    }
    /// Java file/model-view constructor.
    pub fn new_with_file_model_view(
        manager: Option<Infallible>,
        file_name: Option<&str>,
        model_view_type: i32,
        axis_id: AxisID,
    ) -> Self {
        let mut state = Self::new_with_file_name(manager, file_name, axis_id);
        state.set_model_view_type(model_view_type);
        state
    }
    /// Java dataset/model-view/window option/file-type constructor.
    pub fn new_with_dataset_window_file_type(
        manager: Option<Infallible>,
        dataset: Option<&str>,
        model_view_type: i32,
        axis_id: AxisID,
        option: Option<Infallible>,
        file_type: Option<Infallible>,
    ) -> Self {
        let _ = file_type;
        let mut state = Self::new_with_file_model_view(manager, dataset, model_view_type, axis_id);
        state.add_window_open_option(option);
        state
    }
    /// Java dataset/model-view/window option/File constructor.
    pub fn new_with_dataset_window_file(
        manager: Option<Infallible>,
        dataset: Option<&str>,
        model_view_type: i32,
        axis_id: AxisID,
        option: Option<Infallible>,
        file: Option<&Path>,
    ) -> Self {
        let mut state = Self::new_with_file_model_view(manager, dataset, model_view_type, axis_id);
        state.add_window_open_option(option);
        state.file_list = file.map(|f| vec![f.to_owned()]);
        state
    }
    /// Java dataset/model constructor.
    pub fn new_with_dataset_model(
        manager: Option<Infallible>,
        dataset: Option<&str>,
        model: Option<&str>,
        axis_id: AxisID,
    ) -> Self {
        let mut state = Self::new_with_file_name(manager, dataset, axis_id);
        state.initial_model_name = model.unwrap_or_default().to_owned();
        state.reset();
        state
    }
    /// Java FileType constructor.
    pub fn new_with_file_type(
        manager: Option<Infallible>,
        file_type: Option<Infallible>,
        axis_id: AxisID,
    ) -> Self {
        let _ = file_type;
        Self::new(manager, axis_id)
    }
    /// Java File constructor.
    pub fn new_with_file(
        manager: Option<Infallible>,
        file: Option<&Path>,
        axis_id: AxisID,
    ) -> Self {
        let mut state = Self::new(manager, axis_id);
        state.file_list = file.map(|f| vec![f.to_owned()]);
        state.dataset_name = file.map(|f| f.to_string_lossy().into_owned());
        state
    }
    /// Java package-private `setFile`.
    pub fn set_file(&mut self, file: Option<&Path>) {
        self.dataset_name = file.map(|f| f.to_string_lossy().into_owned());
    }
    /// Java String-array constructor.
    pub fn new_with_file_name_array(
        manager: Option<Infallible>,
        files: Option<Vec<String>>,
        axis_id: AxisID,
    ) -> Self {
        let mut state = Self::new(manager, axis_id);
        state.file_name_array = files;
        state
    }
    /// Java String-array/subdir constructor.
    pub fn new_with_file_name_array_subdir(
        manager: Option<Infallible>,
        files: Option<Vec<String>>,
        axis_id: AxisID,
        subdir: Option<&str>,
    ) -> Self {
        let mut state = Self::new_with_file_name_array(manager, files, axis_id);
        state.subdir_name = subdir.map(str::to_owned);
        state
    }
    /// Java File-array constructor.
    pub fn new_with_file_list(
        manager: Option<Infallible>,
        files: Option<Vec<PathBuf>>,
        axis_id: AxisID,
    ) -> Self {
        let mut state = Self::new(manager, axis_id);
        state.file_list = files;
        state
    }
    /// Java complete file-name/FileType constructor.
    pub fn new_with_axis_file_name_file_type(
        manager: Option<Infallible>,
        axis_id: AxisID,
        file_name: Option<&str>,
        file_type: Option<Infallible>,
    ) -> Self {
        let _ = file_type;
        Self::new_with_file_name(manager, file_name, axis_id)
    }
    /// Java AxisID/FileType constructor.
    pub fn new_with_axis_file_type(
        manager: Option<Infallible>,
        axis_id: AxisID,
        file_type: Option<Infallible>,
    ) -> Self {
        let _ = file_type;
        Self::new(manager, axis_id)
    }
    /// Java three-dataset/FileType constructor.
    pub fn new_with_three_file_types(
        manager: Option<Infallible>,
        axis_id: AxisID,
        file_type1: Option<Infallible>,
        file_type2: Option<Infallible>,
        file_type3: Option<Infallible>,
        model_name: Option<&str>,
        model_ext: Option<&str>,
    ) -> Self {
        let _ = (file_type1, file_type2, file_type3);
        let mut state = Self::new(manager, axis_id);
        state.initial_model_name = format!(
            "{}{}{}",
            model_name.unwrap_or_default(),
            axis_id.get_extension(),
            model_ext.unwrap_or_default()
        );
        state.reset();
        state
    }
    /// Java `toString`.
    pub fn to_source_string(&self) -> String {
        format!("[process:{}]", self.process)
    }
    /// Java `processRequest`; ImodProcess boundary.
    pub fn process_request(&mut self) -> Result<(), String> {
        self.process.process_request();
        Ok(())
    }
    /// Java `open(Run3dmodMenuOptions)`.
    pub fn open(&mut self, menu_options: Option<Infallible>) -> Result<(), String> {
        let _ = menu_options;
        self.process.set_model_view(self.model_view);
        self.process.set_use_modv(self.use_modv);
        self.process.set_swap_yz(self.swap_yz);
        self.process.set_frames(self.frames);
        self.process.set_binning(self.binning);
        self.process.set_binning_xy(self.binning_xy);
        if let Some(model) = &self.model_name {
            self.process.set_model_name(model);
        }
        self.process
            .open(Run3dmodMenuOptions::default())
            .map_err(|error| error.to_string())
    }
    /// Java `setOpenSurfContPoint`.
    pub fn set_open_surf_cont_point(&mut self, input: bool) {
        self.open_surf_cont_point = input;
    }
    /// Java `open(FileType,...)`.
    pub fn open_file_type(
        &mut self,
        model: Option<Infallible>,
        menu: Option<Infallible>,
    ) -> Result<(), String> {
        let _ = model;
        self.open(menu)
    }
    /// Java `open(String,...)`.
    pub fn open_model_name(
        &mut self,
        model: Option<&str>,
        menu: Option<Infallible>,
    ) -> Result<(), String> {
        self.set_model_name(model);
        self.open(menu)
    }
    /// Java `open(List<String>,...)`.
    pub fn open_model_name_list(
        &mut self,
        models: Option<Vec<String>>,
        menu: Option<Infallible>,
    ) -> Result<(), String> {
        self.set_model_name_list(models);
        self.open(menu)
    }
    /// Java `open(String,boolean,...)`.
    pub fn open_model_name_mode(
        &mut self,
        model: Option<&str>,
        model_mode: bool,
        menu: Option<Infallible>,
    ) -> Result<(), String> {
        self.set_model_name(model);
        self.set_model_mode(model_mode);
        self.open(menu)
    }
    /// Java `getRubberbandCoordinates`.
    pub fn get_rubberband_coordinates(&mut self) -> Result<Vec<String>, String> {
        self.process
            .get_rubberband_coordinates()
            .map_err(|error| error.to_string())
    }
    /// Java `getSlicerAngles`.
    pub fn get_slicer_angles(&mut self) -> Result<Vec<String>, String> {
        self.process
            .get_slicer_angles()
            .map_err(|error| error.to_string())
    }
    /// Java `quit`.
    pub fn quit(&mut self) -> Result<(), String> {
        self.process.quit().map_err(|error| error.to_string())
    }
    /// Java `disconnect`.
    pub fn disconnect(&mut self) -> Result<(), String> {
        self.process.disconnect().map_err(|error| error.to_string())
    }
    /// Java `setModelViewType`.
    pub fn set_model_view_type(&mut self, t: i32) {
        self.model_view = t == MODEL_VIEW;
        self.use_modv = t == MODV;
        self.process.set_model_view(self.model_view);
        self.process.set_use_modv(self.use_modv);
    }
    /// Java `setOpenZap`.
    pub fn set_open_zap(&mut self) {
        self.open_zap = true;
        self.process.set_open_zap();
    }
    /// Java `setTiltFile`.
    pub fn set_tilt_file(&mut self, file: Option<&str>) {
        self.file_name = file.map(str::to_owned);
        if let Some(file) = file {
            self.process.set_tilt_file(file);
        }
    }
    /// Java `resetTiltFile`.
    pub fn reset_tilt_file(&mut self) {
        self.file_name = None;
        self.process.reset_tilt_file();
    }
    /// Java `addWindowOpenOption`.
    pub fn add_window_open_option(&mut self, option: Option<Infallible>) {
        self.window_open_options.push(option);
    }
    /// Java `reset`.
    pub fn reset(&mut self) {
        self.set_model_name(Some(&self.initial_model_name.clone()));
        self.mode = self.initial_mode;
        self.swap_yz = self.initial_swap_yz;
        self.preserve_contrast = DEFAULT_PRESERVE_CONTRAST;
        self.open_bead_fixer = DEFAULT_OPEN_BEAD_FIXER;
        self.open_contours = DEFAULT_OPEN_CONTOURS;
        self.binning = DEFAULT_BINNING;
        self.process.set_open_with_model(!self.preserve_contrast);
        self.process.set_binning(DEFAULT_BINNING);
        self.frames = DEFAULT_FRAMES;
        self.piece_list_file_name = None;
        self.manage_new_contours = false;
        self.point_limit = -1;
        self.start_new_contours_at_new_z = false;
    }
    /// Java `getModeString(int)`.
    pub fn get_mode_string_for(mode: i32) -> String {
        match mode {
            MOVIE_MODE => "MOVIE_MODE".into(),
            MODEL_MODE => "MODEL_MODE".into(),
            _ => format!("ERROR:{mode}"),
        }
    }
    /// Java `setDebug`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
        self.process.set_debug(input);
    }
    /// Java `equalsSubdirName`.
    pub fn equals_subdir_name(&self, input: Option<&str>) -> bool {
        self.subdir_name.as_deref() == input
    }
    /// Java `equalsFileNameArray`.
    pub fn equals_file_name_array(&self, input: Option<&[String]>) -> bool {
        self.file_name_array.as_deref() == input
    }
    /// Java `isModelView`.
    pub fn is_model_view(&self) -> bool {
        self.model_view
    }
    /// Java `isUseModv`.
    pub fn is_use_modv(&self) -> bool {
        self.use_modv
    }
    /// Java `getModelName`.
    pub fn get_model_name(&self) -> Option<String> {
        self.model_name.clone()
    }
    /// Java private `setModel(FileType)`.
    pub fn set_model(&mut self, model: Option<Infallible>) {
        let _ = model;
    }
    /// Java private `setModelName`.
    pub fn set_model_name(&mut self, model: Option<&str>) {
        self.model_name = model.map(str::to_owned);
        if let Some(model) = model {
            self.process.set_model_name(model);
        }
    }
    /// Java private `setModelNameList`.
    pub fn set_model_name_list(&mut self, models: Option<Vec<String>>) {
        self.model_name_list = models.clone();
        self.process.set_model_name_list(models);
    }
    /// Java `setLoadAsIntegers`.
    pub fn set_load_as_integers(&mut self) {
        self.load_as_integers = true;
        self.process.set_load_as_integers();
    }
    /// Java `setSuppressSaveQuery`.
    pub fn set_suppress_save_query(&mut self) {
        self.suppress_save_query = true;
        self.process.set_suppress_save_query();
    }
    /// Java `isUsingMode`.
    pub fn is_using_mode(&self) -> bool {
        self.using_mode
    }
    /// Java `setUsingMode`.
    pub fn set_using_mode(&mut self, input: bool) {
        self.using_mode = input;
    }
    /// Java `isOpenContours`.
    pub fn is_open_contours(&self) -> bool {
        self.open_contours
    }
    /// Java `setOpenContours`.
    pub fn set_open_contours(&mut self, input: bool) {
        self.open_contours = input;
    }
    /// Java `setStartNewContoursAtNewZ`.
    pub fn set_start_new_contours_at_new_z(&mut self, input: bool) {
        self.start_new_contours_at_new_z = input;
    }
    /// Java `setPointLimit`.
    pub fn set_point_limit(&mut self, input: i32) {
        self.point_limit = input;
    }
    /// Java final `getAxisID`.
    pub fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }
    /// Java `getMode`.
    pub fn get_mode(&self) -> i32 {
        self.mode
    }
    /// Java `getModeString()`.
    pub fn get_mode_string(&self) -> String {
        Self::get_mode_string_for(self.mode)
    }
    /// Java `setMode`.
    pub fn set_mode(&mut self, mode: i32) {
        self.using_mode = true;
        self.mode = mode;
    }
    /// Java private `setModelMode`.
    pub fn set_model_mode(&mut self, model_mode: bool) {
        self.using_mode = true;
        self.mode = if model_mode { MODEL_MODE } else { MOVIE_MODE };
    }
    /// Java `isSwapYZ`.
    pub fn is_swap_yz(&self) -> bool {
        self.swap_yz
    }
    /// Java `setSwapYZ`.
    pub fn set_swap_yz(&mut self, input: bool) {
        self.swap_yz = input;
        self.process.set_swap_yz(input);
    }
    /// Java `isPreserveContrast`.
    pub fn is_preserve_contrast(&self) -> bool {
        self.preserve_contrast
    }
    /// Java `setPreserveContrast`.
    pub fn set_preserve_contrast(&mut self, input: bool) {
        self.preserve_contrast = input;
    }
    /// Java `setFrames`.
    pub fn set_frames(&mut self, input: bool) {
        self.frames = input;
        self.process.set_frames(input);
    }
    /// Java `setPieceListFileName`.
    pub fn set_piece_list_file_name(&mut self, input: Option<&str>) {
        self.piece_list_file_name = input.map(str::to_owned);
        if let Some(input) = input {
            self.process.set_piece_list_file_name(input);
        }
    }
    /// Java `setMontageSeparation`.
    pub fn set_montage_separation(&mut self) {
        self.montage_separation = true;
        self.process.set_montage_separation();
    }
    /// Java `setInterpolation`.
    pub fn set_interpolation(&mut self, input: bool) {
        self.interpolation = Some(input);
    }
    /// Java `isOpenBeadFixer`.
    pub fn is_open_bead_fixer(&self) -> bool {
        self.open_bead_fixer
    }
    /// Java `setOpenBeadFixer`.
    pub fn set_open_bead_fixer(&mut self, input: bool) {
        self.open_bead_fixer = input;
        self.set_auto_center = false;
        self.auto_center = false;
        self.new_contours = false;
        self.manage_new_contours = false;
    }
    /// Java `setAutoCenter`.
    pub fn set_auto_center(&mut self, input: bool) {
        self.set_auto_center = true;
        self.auto_center = input;
    }
    /// Java `setSkipList`.
    pub fn set_skip_list(&mut self, input: Option<&str>) {
        self.skip_list = input.map(str::to_owned);
    }
    /// Java `setDeleteAllSections`.
    pub fn set_delete_all_sections(&mut self, input: bool) {
        self.delete_all_sections = Some(input);
    }
    /// Java `setBeadfixerMode`.
    pub fn set_beadfixer_mode(&mut self, input: Option<Infallible>) {
        self.beadfixer_mode = input;
    }
    /// Java `setOpenLogOff`.
    pub fn set_open_log_off(&mut self) {
        self.log_name = None;
    }
    /// Java `setOpenLog`.
    pub fn set_open_log(&mut self, open: bool, name: Option<&str>) {
        self.log_name = if open { name.map(str::to_owned) } else { None };
    }
    /// Java `setNewContours`.
    pub fn set_new_contours(&mut self, input: bool) {
        self.new_contours = input;
        self.manage_new_contours = true;
    }
    /// Java `getInitialModelName`.
    pub fn get_initial_model_name(&self) -> String {
        self.initial_model_name.clone()
    }
    /// Java `getDatasetName`.
    pub fn get_dataset_name(&self) -> Option<String> {
        self.dataset_name.clone()
    }
    /// Java `getInitialMode`.
    pub fn get_initial_mode(&self) -> i32 {
        self.initial_mode
    }
    /// Java `getInitialModeString`.
    pub fn get_initial_mode_string(&self) -> String {
        Self::get_mode_string_for(self.initial_mode)
    }
    /// Java `setInitialMode`.
    pub fn set_initial_mode(&mut self, input: i32) {
        if !self.initial_mode_set {
            self.initial_mode = input;
            self.set_mode(input);
            self.initial_mode_set = true;
        }
    }
    /// Java `isInitialSwapYZ`.
    pub fn is_initial_swap_yz(&self) -> bool {
        self.initial_swap_yz
    }
    /// Java public `setInitialSwapYZ`.
    pub fn set_initial_swap_yz(&mut self, input: bool) {
        if !self.initial_swap_yz_set {
            self.initial_swap_yz = input;
            self.set_swap_yz(input);
            self.initial_swap_yz_set = true;
        }
    }
    /// Java `isDefaultOpenWithModel`.
    pub fn is_default_open_with_model(&self) -> bool {
        DEFAULT_OPEN_WITH_MODEL
    }
    /// Java `isDefaultPreserveContrast`.
    pub fn is_default_preserve_contrast(&self) -> bool {
        DEFAULT_PRESERVE_CONTRAST
    }
    /// Java `isOpen`; untranslated process cannot report a fabricated running state.
    pub fn is_open(&self) -> bool {
        self.process.is_running()
    }
    /// Java final `setAllowMenuBinningInZ`.
    pub fn set_allow_menu_binning_in_z(&mut self, input: bool) {
        self.allow_menu_binning_in_z = input;
    }
    /// Java final `setNoMenuOptions`.
    pub fn set_no_menu_options(&mut self, input: bool) {
        self.no_menu_options = input;
    }
    /// Java `reopenLog`.
    pub fn reopen_log(&mut self) -> Result<(), String> {
        self.process.reopen_log().map_err(|error| error.to_string())
    }
    /// Java `openModel`.
    pub fn open_model(&mut self, model: Option<&str>, model_mode: bool) -> Result<(), String> {
        let Some(model) = model else {
            return Ok(());
        };
        self.process
            .open_model(model, model_mode)
            .map_err(|error| error.to_string())
    }
    /// Java `setBinning`.
    pub fn set_binning(&mut self, input: i32) {
        self.binning = input;
        self.process.set_binning(input);
    }
    /// Java `setBinningXY`.
    pub fn set_binning_xy(&mut self, input: i32) {
        self.binning_xy = input;
        self.process.set_binning_xy(input);
    }
    /// Java `setWorkingDirectory`.
    pub fn set_working_directory(&mut self, input: Option<&Path>) {
        self.working_directory = input.map(Path::to_owned);
        self.process
            .set_working_directory(self.working_directory.clone());
    }
    /// Java `setOpenModelView`.
    pub fn set_open_model_view(&mut self) -> Result<(), String> {
        self.process.set_open_model_view();
        Ok(())
    }
    /// Java `setContinuousListenerTarget`.
    pub fn set_continuous_listener_target(&self, input: Option<Infallible>) {
        let _ = input;
    }
    /// Java `isWarnedStaleFile`.
    pub fn is_warned_stale_file(&self) -> bool {
        self.warned_stale_file
    }
    /// Java `setWarnedStaleFile`.
    pub fn set_warned_stale_file(&mut self, input: bool) {
        self.warned_stale_file = input;
    }
    /// Java `paramString`.
    pub fn param_string(&self) -> String {
        format!(
            "[modelView={}, useModv={}, modelName={:?}, usingMode={}, mode={}, swapYZ={}, preserveContrast={}, openBeadFixer={}, initialModelName={}, initialMode={}, initialSwapYZ={}, defaultOpenWithModel={}, defaultPreserveContrast={}, process={}, warnedStaleFile={}, openContours={}]",
            self.model_view,
            self.use_modv,
            self.model_name,
            self.using_mode,
            self.get_mode_string(),
            self.swap_yz,
            self.preserve_contrast,
            self.open_bead_fixer,
            self.initial_model_name,
            self.get_initial_mode_string(),
            self.initial_swap_yz,
            DEFAULT_OPEN_WITH_MODEL,
            DEFAULT_PRESERVE_CONTRAST,
            self.process,
            self.warned_stale_file,
            self.open_contours
        )
    }
    /// Java `equalsInitialConfiguration`.
    pub fn equals_initial_configuration(&self, other: &Self) -> bool {
        self.model_view == other.model_view
            && self.use_modv == other.use_modv
            && self.initial_model_name == other.initial_model_name
            && self.initial_mode == other.initial_mode
            && self.initial_swap_yz == other.initial_swap_yz
    }
    /// Java `equalsCurrentConfiguration`.
    pub fn equals_current_configuration(&self, other: &Self) -> bool {
        self.model_view == other.model_view
            && self.use_modv == other.use_modv
            && self.model_name == other.model_name
            && self.using_mode == other.using_mode
            && self.open_contours == other.open_contours
            && self.mode == other.mode
            && self.swap_yz == other.swap_yz
            && self.preserve_contrast == other.preserve_contrast
            && self.open_bead_fixer == other.open_bead_fixer
            && self.start_new_contours_at_new_z == other.start_new_contours_at_new_z
    }
    /// Java `equals(ImodState)`.
    pub fn equals(&self, other: &Self) -> bool {
        self.equals_initial_configuration(other) && self.equals_current_configuration(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reset_preserves_initial_and_resets_current_state() {
        let mut state = ImodState::new(None, AxisID::First);
        state.set_initial_mode(MODEL_MODE);
        state.set_initial_swap_yz(true);
        state.set_preserve_contrast(true);
        state.set_open_contours(true);
        state.set_mode(MOVIE_MODE);
        state.reset();
        assert_eq!(state.get_mode(), MODEL_MODE);
        assert!(state.is_swap_yz());
        assert!(!state.is_preserve_contrast());
        assert!(!state.is_open_contours());
    }

    #[test]
    fn open_delegates_configured_state_to_imod_process_command() {
        let mut state = ImodState::new(None, AxisID::First);
        state.set_model_name(Some("model.mod"));
        state.set_model_view_type(MODV);
        state.set_swap_yz(true);
        state.set_frames(true);
        state.set_binning(2);
        let error = state.open(None).unwrap_err();
        assert!(error.contains("-view"));
        assert!(error.contains("-Y"));
        assert!(error.contains("-f"));
        assert!(error.contains("-B 2"));
        assert!(error.ends_with("model.mod"));
    }
}
