//! `IMOD/Etomo/src/etomo/process/ImodProcess.java`.
//!
//! ImodProcess opens an instance of imod with the specfied stack projection stack(s) and
//! possibly model files.  Model files can also be loaded and changed after the process
//! has started.
//!
//! **Shape.**  The Java object is reached from the event thread, from the thread running
//! 3dmod, from the `ContinuousListener` thread and from `MessageSender` threads, so every
//! mutable field sits behind a lock and every method takes `&self`; an instance lives in
//! an `Arc` (`this` is the `Weak` a `MessageSender` thread upgrades).  3dmod,
//! `imodsendevent` are started through `InteractiveSystemProgram`, which launches through
//! `system_program::runtime_exec`, so they resolve like every other IMOD program.
//!
//! **Monitors.**  The source synchronizes on two objects of `Stderr`: its instance
//! (`synchronized` methods) and its `quickListenerQueue` list (`synchronized
//! (stderr.quickListenerQueue)` blocks here).  They are two locks in [`Stderr`]:
//! `data` and `quick_listener_queue`.  `ContinuousListener` synchronizes `startThread`
//! and the whole of `run` on itself; that is its `monitor`.
//!
//! **Null manager.**  `manager` is `&'static dyn BaseManager`: every construction site
//! passes the manager that owns the `ImodManager`, so the source's `manager == null`
//! branches (`new EmergencyMonitor(null, axisID)`) are not reachable and are not kept.

use super::base_imod_manager::ImodManagerException;
use super::base_process_manager::SystemProcessException;
use super::continuous_listener_target::ContinuousListenerTarget;
use super::emergency_monitor::EmergencyMonitor;
use super::interactive_system_program::InteractiveSystemProgram;
use super::process_messages::{MessageType, ProcessMessages};
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::comscript::utilities as comscript_utilities;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::log_file::{Handle, LogFile};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::extension::{self, Extension};
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::os_type::OSType;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities;
use regex::Regex;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

/// The type the translated menus import from here; the Java class is
/// `etomo.type.Run3dmodMenuOptions`.
pub use crate::imod::etomo::r#type::run_3dmod_menu_options::Run3dmodMenuOptions;

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
pub const TRUE: &str = "1";
pub const FALSE: &str = "0";
pub const CIRCLE: i32 = 1;
pub const REQUEST_TAG: &str = "REQUEST";
pub const STOP_LISTENING_REQUEST: &str = "STOP LISTENING";
/// Java private static `defaultBinning`.
const DEFAULT_BINNING: i32 = 1;
pub const IMOD_SEND_EVENT_STRING: &str = "imodsendevent returned:";
// static final String CONTINUOUS_TAG = "ETOMO INFO:";
pub const CONTINUOUS_TAG: &str = "ETOMO INFO:";
/// Java private static `MESSAGE_OPEN_DIALOG`.
const MESSAGE_OPEN_DIALOG: &str = "19";
/// Java private static `SURF_CONT_POINT_DIALOG`.
const SURF_CONT_POINT_DIALOG: &str = "s";

/// The pattern of Java's `String.split("\\s+")`: `\s` is the ASCII class
/// `[ \t\n\x0B\f\r]`.
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?-u:\s)+").unwrap());

/// Java `AbstractCollection.toString()` of a `Vector` of strings.
fn vector_to_string(vector: &[String]) -> String {
    format!("[{}]", vector.join(", "))
}

/// The instance fields of `ImodProcess` other than the final ones.
struct ImodProcessFields {
    /// Java `datasetName`: the file name.  Can have an absolute path.
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
    /// Java `Vector sendArguments`.
    send_arguments: Vec<String>,
    dataset_name_array: Option<Vec<String>>,
    frames: bool,
    piece_list_file_name: Option<String>,
    flip: bool,
    beadfixer_diameter_set: bool,
    window_open_option_list: Option<Vec<WindowOpenOption>>,
    debug: bool,
    subdir_name: Option<String>,
    // Zap opens by default. OpenZap is only necessary when model view is in use.
    open_zap: bool,
    tilt_file: Option<String>,
    continuous_listener_target: Option<Arc<dyn ContinuousListenerTarget>>,
    file_list: Option<Vec<PathBuf>>,
    load_as_integers: bool,
    montage_separation: bool,
    model_name_list: Option<Vec<String>>,
    suppress_save_query: bool,
    first_file: Option<Arc<Handle>>,
}

impl ImodProcessFields {
    /// The Java field initialisers.
    fn initial() -> ImodProcessFields {
        ImodProcessFields {
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
            send_arguments: Vec::new(),
            dataset_name_array: None,
            frames: false,
            piece_list_file_name: None,
            flip: false,
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
            first_file: None,
        }
    }
}

/// Java `ImodProcess`.
pub struct ImodProcess {
    /// Java's `this`, for the `MessageSender` threads.
    this: Weak<ImodProcess>,
    /// Java `stderr`.  Get stderr messages only through this member variable.
    stderr: Arc<Stderr>,
    message_sender_reg_id: i32,
    stderr_reg_id: i32,
    continuous_listener: Arc<ContinuousListener>,
    fields: Mutex<ImodProcessFields>,
    /// Java package-private `imod`.
    imod: Mutex<Option<Arc<InteractiveSystemProgram>>>,
    /// Java `imodThread`.  `Thread.isAlive()` is `!JoinHandle::is_finished()`.
    imod_thread: Mutex<Option<Arc<JoinHandle<()>>>>,
    /// Java `axisID`; assigned only by the constructors that take it.
    axis_id: Option<AxisID>,
    manager: &'static dyn BaseManager,
    /// If true, run 3dmod with -L.  This means that imodsentevent will not be used - the
    /// MessageSender can be used instead.  In Windows the Stderr.requestQueue will
    /// receive requests to send MESSAGE_STOP_LISTENING.  The thread should be checking
    /// this queue (see ImodManager.ImodManager()).
    listen_to_stdin: bool,
}

impl ImodProcess {
    /// The Java field initialisers followed by `this.manager = manager` and
    /// `continuousListener = new ContinuousListener(stderr, axisID)`, which every
    /// constructor runs.
    fn construct(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        fields: ImodProcessFields,
    ) -> Arc<ImodProcess> {
        let stderr = Stderr::new();
        let message_sender_reg_id = stderr.register();
        let stderr_reg_id = stderr.register();
        let listen_to_stdin =
            !utilities::is_windows_os() || etomo_director::ARGUMENTS.lock().unwrap().is_listen();
        let continuous_listener = ContinuousListener::new(Arc::clone(&stderr), axis_id);
        Arc::new_cyclic(|this| ImodProcess {
            this: this.clone(),
            stderr,
            message_sender_reg_id,
            stderr_reg_id,
            continuous_listener,
            fields: Mutex::new(fields),
            imod: Mutex::new(None),
            imod_thread: Mutex::new(None),
            axis_id,
            manager,
            listen_to_stdin,
        })
    }

    /// Java `ImodProcess(BaseManager, AxisID)`.  Constructor for using imodv.
    pub fn new_base_manager_axis_id(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
    ) -> Arc<ImodProcess> {
        ImodProcess::construct(manager, axis_id, ImodProcessFields::initial())
    }

    /// Java `ImodProcess(BaseManager, String, AxisID, FileType)`.  Dataset only
    /// constructor.
    pub fn new_base_manager_string_axis_id_file_type(
        manager: &'static dyn BaseManager,
        dataset: &str,
        axis_id: Option<AxisID>,
        file_type: &FileType,
    ) -> Arc<ImodProcess> {
        let mut fields = ImodProcessFields::initial();
        fields.dataset_name = dataset.to_string();
        let process = ImodProcess::construct(manager, axis_id, fields);
        let first_file = ImodProcess::construct_first_file_base_manager_axis_id_file_type(
            manager, axis_id, file_type,
        );
        process.fields.lock().unwrap().first_file = first_file;
        process
    }

    /// Java `ImodProcess(BaseManager, String, AxisID, File)`.
    pub fn new_base_manager_string_axis_id_file(
        manager: &'static dyn BaseManager,
        dataset: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> Arc<ImodProcess> {
        let mut fields = ImodProcessFields::initial();
        fields.dataset_name = dataset.to_string();
        let process = ImodProcess::construct(manager, axis_id, fields);
        let first_file =
            ImodProcess::construct_first_file_base_manager_axis_id_file_emergency_monitor(
                manager, axis_id, file, None,
            );
        process.fields.lock().unwrap().first_file = first_file;
        process
    }

    /// Java `ImodProcess(BaseManager, String, AxisID)`.
    pub fn new_base_manager_string_axis_id(
        manager: &'static dyn BaseManager,
        file_name: &str,
        axis_id: Option<AxisID>,
    ) -> Arc<ImodProcess> {
        let mut fields = ImodProcessFields::initial();
        fields.dataset_name = file_name.to_string();
        let process = ImodProcess::construct(manager, axis_id, fields);
        let first_file = ImodProcess::construct_first_file_base_manager_axis_id_string(
            manager, axis_id, file_name,
        );
        process.fields.lock().unwrap().first_file = first_file;
        process
    }

    /// Java `ImodProcess(BaseManager, String, String, File, EmergencyMonitor)`.  Dataset
    /// and model file constructor.  The source never assigns `axisID` here, so it stays
    /// null, and it passes null rather than its `emergencyMonitor` parameter on.
    pub fn new_base_manager_string_string_file_emergency_monitor(
        manager: &'static dyn BaseManager,
        dataset: &str,
        model: &str,
        file: Option<&Path>,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Arc<ImodProcess> {
        let _ = emergency_monitor;
        let mut fields = ImodProcessFields::initial();
        fields.dataset_name = dataset.to_string();
        fields.model_name = model.to_string();
        let process = ImodProcess::construct(manager, None, fields);
        let first_file =
            ImodProcess::construct_first_file_base_manager_axis_id_file_emergency_monitor(
                manager, None, file, None,
            );
        process.fields.lock().unwrap().first_file = first_file;
        process
    }

    /// Java `ImodProcess(BaseManager, String[], String, File, EmergencyMonitor)`.
    /// Dataset and model file constructor.  `axisID` stays null.
    pub fn new_base_manager_string_array_string_file_emergency_monitor(
        manager: &'static dyn BaseManager,
        dataset_array: Option<Vec<String>>,
        model: &str,
        file: Option<&Path>,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Arc<ImodProcess> {
        let mut fields = ImodProcessFields::initial();
        fields.dataset_name_array = dataset_array;
        fields.model_name = model.to_string();
        let process = ImodProcess::construct(manager, None, fields);
        let first_file =
            ImodProcess::construct_first_file_base_manager_axis_id_file_emergency_monitor(
                manager,
                None,
                file,
                emergency_monitor,
            );
        process.fields.lock().unwrap().first_file = first_file;
        process
    }

    /// Java `ImodProcess(BaseManager, String[])`.  `axisID` stays null.
    pub fn new_base_manager_string_array(
        manager: &'static dyn BaseManager,
        dataset_array: Option<Vec<String>>,
    ) -> Arc<ImodProcess> {
        let mut fields = ImodProcessFields::initial();
        fields.dataset_name_array = dataset_array;
        ImodProcess::construct(manager, None, fields)
    }

    /// Java `ImodProcess(BaseManager, File[], EmergencyMonitor)`.  `axisID` stays null.
    pub fn new_base_manager_file_array_emergency_monitor(
        manager: &'static dyn BaseManager,
        file_list: Option<Vec<PathBuf>>,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Arc<ImodProcess> {
        let mut fields = ImodProcessFields::initial();
        fields.file_list = file_list.clone();
        let process = ImodProcess::construct(manager, None, fields);
        let first_file =
            ImodProcess::construct_first_file_base_manager_axis_id_file_array_emergency_monitor(
                manager,
                None,
                file_list.as_deref(),
                emergency_monitor,
            );
        process.fields.lock().unwrap().first_file = first_file;
        process
    }

    /// Java private static `constructFirstFile(BaseManager, AxisID, String)`.
    fn construct_first_file_base_manager_axis_id_string(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        file_name: &str,
    ) -> Option<Arc<Handle>> {
        // `new File(manager.getPropertyUserDir(), fileName)`.
        let file = match manager.get_property_user_dir() {
            None => PathBuf::from(file_name),
            Some(dir) => PathBuf::from(utilities::java_io_file_new(&dir, file_name)),
        };
        let emergency_monitor = manager.get_emergency_monitor(axis_id);
        match LogFile::get_instance_file(Some(&file), Some(emergency_monitor)) {
            Ok(handle) => Some(handle),
            Err(e) => {
                eprintln!("{}", e.get_message());
                eprintln!("{e}");
                None
            }
        }
    }

    /// Java private static `constructFirstFile(BaseManager, AxisID, FileType)`.
    fn construct_first_file_base_manager_axis_id_file_type(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        file_type: &FileType,
    ) -> Option<Arc<Handle>> {
        let emergency_monitor = manager.get_emergency_monitor(axis_id);
        match LogFile::get_instance_manager_file_type(
            Some(manager),
            axis_id,
            file_type,
            Some(emergency_monitor),
        ) {
            Ok(handle) => Some(handle),
            Err(e) => {
                eprintln!("{}", e.get_message());
                eprintln!("{e}");
                None
            }
        }
    }

    /// Java private static `constructFirstFile(BaseManager, AxisID, File[],
    /// EmergencyMonitor)`.
    fn construct_first_file_base_manager_axis_id_file_array_emergency_monitor(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        file_array: Option<&[PathBuf]>,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Option<Arc<Handle>> {
        let file_array = file_array?;
        // A Rust `PathBuf` cannot be null, so the first element is the first non-null
        // one the source's loop looks for.
        if let Some(file) = file_array.first() {
            return ImodProcess::construct_first_file_base_manager_axis_id_file_emergency_monitor(
                manager,
                axis_id,
                Some(file),
                emergency_monitor,
            );
        }
        None
    }

    /// Java private static `constructFirstFile(BaseManager, AxisID, File,
    /// EmergencyMonitor)`.
    fn construct_first_file_base_manager_axis_id_file_emergency_monitor(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Option<Arc<Handle>> {
        let file = file?;
        let emergency_monitor = match emergency_monitor {
            None => manager.get_emergency_monitor(axis_id),
            Some(emergency_monitor) => emergency_monitor,
        };
        match LogFile::get_instance_file(Some(file), Some(emergency_monitor)) {
            Ok(handle) => Some(handle),
            Err(e) => {
                eprintln!("{}", e.get_message());
                eprintln!("{e}");
                None
            }
        }
    }

    /// Java `setSuppressSaveQuery`.
    pub fn set_suppress_save_query(&self) {
        self.fields.lock().unwrap().suppress_save_query = true;
    }

    /// Java `setDatasetName`.  Change the dataset name.
    pub fn set_dataset_name(&self, dataset_name: &str) {
        self.fields.lock().unwrap().dataset_name = dataset_name.to_string();
    }

    /// Java `setSubdirName`.
    pub fn set_subdir_name(&self, input: Option<&str>) {
        self.fields.lock().unwrap().subdir_name = input.map(str::to_string);
    }

    /// Java `getSubdirName`.
    pub fn get_subdir_name(&self) -> Option<String> {
        self.fields.lock().unwrap().subdir_name.clone()
    }

    /// Java `setFrames`.  Sets the -f command line option.
    pub fn set_frames(&self, frames: bool) {
        self.fields.lock().unwrap().frames = frames;
    }

    /// Java `setPieceListFileName`.
    pub fn set_piece_list_file_name(&self, piece_list_file_name: Option<&str>) {
        self.fields.lock().unwrap().piece_list_file_name = piece_list_file_name.map(str::to_string);
    }

    /// Java `setMontageSeparation`.
    pub fn set_montage_separation(&self) {
        self.fields.lock().unwrap().montage_separation = true;
    }

    /// Java `setModelName`.  Specify or change the model name.
    ///
    /// `ImodState` can pass a null model name (its `modelName` starts null); the source
    /// then keeps null in this field and `open` dereferences it.  Fixed in translation:
    /// null is stored as "", the field's initial value, which `open` treats as "no
    /// model".
    pub fn set_model_name(&self, model_name: Option<&str>) {
        self.fields.lock().unwrap().model_name = model_name.unwrap_or("").to_string();
    }

    /// Java `setModelNameList`.
    pub fn set_model_name_list(&self, model_name_list: Option<Vec<String>>) {
        self.fields.lock().unwrap().model_name_list = model_name_list;
    }

    /// Java `setWorkingDirectory`.
    pub fn set_working_directory(&self, working_directory: Option<PathBuf>) {
        self.fields.lock().unwrap().working_directory = working_directory;
    }

    /// Java `setLoadAsIntegers`.
    pub fn set_load_as_integers(&self) {
        self.fields.lock().unwrap().load_as_integers = true;
    }

    /// Java `setOpenWithModel`.  When openWithModel is true 3dmod will open with a
    /// model, if a model is set.  The default for openWithModel is true.  Some open
    /// model options cannot be sent to 3dmod during open.  Turn off this option to
    /// prevent opening the model during open.  Example: MESSAGE_OPEN_KEEP_BW
    pub fn set_open_with_model(&self, open_with_model: bool) {
        self.fields.lock().unwrap().open_with_model = open_with_model;
    }

    /// Java private `calcCurrentBinning`.
    fn calc_current_binning(&self, binning: i32, menu_options: &Run3dmodMenuOptions) -> i32 {
        let mut current_binning;
        if binning == DEFAULT_BINNING {
            current_binning = 0;
        } else {
            current_binning = binning;
        }
        if menu_options.is_bin_by_2() {
            current_binning += 2;
        }
        current_binning
    }

    /// Java `open(Run3dmodMenuOptions)`.  Open the 3dmod process if is not already open.
    pub fn open(&self, menu_options: Run3dmodMenuOptions) -> Result<(), ImodManagerException> {
        if self.is_running() {
            self.raise_3dmod()?;
            return Ok(());
        }
        let test = etomo_director::ARGUMENTS.lock().unwrap().is_test();
        // Reset the window string
        self.fields.lock().unwrap().window_id = String::new();
        if etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_debug_level()
            .is_extra_verbose()
        {
            eprintln!("open 1:windowID:{}", self.fields.lock().unwrap().window_id);
        }
        let mut command_options: Vec<String> = Vec::new();
        command_options.push(base_manager::get_imod_bin_path().unwrap_or_default() + "3dmod");
        // Collect the command line options

        // 3/22/09
        // On Mac never run with -D, -W, and not -L. This will crash 3dmod by
        // copying the clipboard onto the message area. 3dmod will crash if there is
        // something big in the clipboard.
        let fields = self.fields.lock().unwrap();
        if etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_debug_level()
            .is_extra_verbose()
        {
            command_options.push("-DC".to_string());
            if OSType::get_instance() == OSType::Mac
                && fields.output_window_id
                && !self.listen_to_stdin
            {
                command_options.push("-L".to_string());
            }
        }

        if fields.output_window_id {
            command_options.push("-W".to_string());
        }
        if self.listen_to_stdin {
            command_options.push("-L".to_string());
        }

        if fields.swap_yz {
            command_options.push("-Y".to_string());
        }
        if fields.frames {
            command_options.push("-f".to_string());
        }

        if fields.montage_separation {
            command_options.push("-o".to_string());
            command_options.push(
                comscript_utilities::MONTAGE_SEPARATION.to_string()
                    + ","
                    + comscript_utilities::MONTAGE_SEPARATION,
            );
        }

        // `pieceListFileName.matches("\\S+")`: non-empty and no ASCII white space.
        if let Some(piece_list_file_name) = &fields.piece_list_file_name
            && !piece_list_file_name.is_empty()
            && !piece_list_file_name
                .chars()
                .any(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
        {
            command_options.push("-p".to_string());
            command_options.push(piece_list_file_name.clone());
        }

        if fields.model_view {
            command_options.push("-V".to_string());
        }

        if fields.open_zap {
            command_options.push("-Z".to_string());
        }

        if fields.load_as_integers {
            command_options.push("-I".to_string());
            command_options.push("1".to_string());
        }

        if let Some(tilt_file) = &fields.tilt_file {
            command_options.push("-a".to_string());
            command_options.push(tilt_file.clone());
            // If the raw tilt file is used, and this is a dose symmetric series, pass the
            // starting angle.
            // Java compares the singletons by identity; the registry holds copies, so
            // compare by value.
            if Extension::get_instance(tilt_file) == Some(&extension::CLASS.rawtlt) {
                // `manager instanceof ApplicationManager`.
                if let Some(application_manager) = crate::imod::etomo::application_manager::instance_of(self.manager) {
                    let meta_data = application_manager.get_meta_data();
                    // `isDoseSym`/`getDoseSym` test `axisID == AxisID.SECOND`, which a
                    // null axisID is not.
                    let axis_id = self.axis_id.unwrap_or(AxisID::Only);
                    if meta_data.is_dose_sym(axis_id) {
                        let dose_sym = meta_data.get_dose_sym(axis_id);
                        if !utilities::is_empty(Some(&dose_sym)) {
                            command_options.push("-it".to_string());
                            command_options.push(dose_sym);
                        }
                    }
                }
            }
        }

        if fields.use_modv {
            command_options.push("-view".to_string());
        }
        /* if (debug) { commandOptions.add("-DC"); } */
        if fields.binning > DEFAULT_BINNING
            || (menu_options.is_bin_by_2() && menu_options.is_allow_binning_in_z())
        {
            command_options.push("-B".to_string());
            command_options.push(
                self.calc_current_binning(fields.binning, &menu_options)
                    .to_string(),
            );
        }

        if fields.binning_xy > DEFAULT_BINNING
            || (menu_options.is_bin_by_2() && !menu_options.is_allow_binning_in_z())
        {
            command_options.push("-b".to_string());
            command_options.push(
                self.calc_current_binning(fields.binning_xy, &menu_options)
                    .to_string(),
            );
        }

        if menu_options.is_startup_window() {
            command_options.push("-O".to_string());
        }

        if let Some(window_open_option_list) = &fields.window_open_option_list {
            command_options.push(WindowOpenOption::OPTION.to_string());
            let mut buffer = window_open_option_list[0].to_string();
            for option in window_open_option_list.iter().skip(1) {
                buffer.push_str(&option.to_string());
            }
            if etomo_director::ARGUMENTS.lock().unwrap().is_test() || fields.suppress_save_query {
                buffer.push('2');
            }
            command_options.push(buffer);
        } else if etomo_director::ARGUMENTS.lock().unwrap().is_test() || fields.suppress_save_query
        {
            command_options.push(WindowOpenOption::OPTION.to_string());
            command_options.push("2".to_string());
        }

        if fields.dataset_name != "" {
            command_options.push(fields.dataset_name.clone());
        }

        if let Some(dataset_name_array) = &fields.dataset_name_array {
            for dataset_name in dataset_name_array {
                match &fields.subdir_name {
                    None => command_options.push(dataset_name.clone()),
                    Some(subdir_name) => {
                        command_options.push(utilities::java_io_file_new(subdir_name, dataset_name))
                    }
                }
            }
        }

        if let Some(file_list) = &fields.file_list {
            for file in file_list {
                // `File.getName()`.
                let name = file
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                match &fields.subdir_name {
                    None => command_options.push(name),
                    Some(subdir_name) => {
                        command_options.push(utilities::java_io_file_new(subdir_name, &name))
                    }
                }
            }
        }

        if fields.open_with_model {
            if fields.model_name != "" {
                command_options.push(fields.model_name.clone());
            }
            if let Some(model_name_list) = &fields.model_name_list {
                for model_name in model_name_list {
                    command_options.push(model_name.clone());
                }
            }
        }
        let debug = fields.debug;
        let working_directory = fields.working_directory.clone();
        let continuous_listener_target = fields.continuous_listener_target.clone();
        drop(fields);
        let command_array: Vec<String> = command_options;
        let mut printed = false;
        for command in &command_array {
            if etomo_director::ARGUMENTS.lock().unwrap().is_debug() || test {
                eprint!("{command} ");
                printed = true;
            } else if debug || test {
                eprint!("{command} ");
                printed = true;
            }
        }
        if printed {
            eprintln!();
        }
        // `InteractiveSystemProgram` needs an axis; a null one (see the constructors) is
        // only ever read back by a process manager, which this program has none of.
        let imod = Arc::new(InteractiveSystemProgram::new(
            self.manager,
            Some(command_array),
            self.axis_id.unwrap_or(AxisID::Only),
        ));
        *self.imod.lock().unwrap() = Some(Arc::clone(&imod));
        if test {
            imod.set_print_stderr();
        }
        self.stderr.set_imod(Some(Arc::clone(&imod)));

        if working_directory.is_some() {
            imod.set_working_directory(working_directory);
        }
        // Start the 3dmod program thread and wait for it to finish
        let imod_thread = {
            let imod = Arc::clone(&imod);
            Arc::new(std::thread::spawn(move || imod.run()))
        };
        *self.imod_thread.lock().unwrap() = Some(Arc::clone(&imod_thread));
        if let Some(continuous_listener_target) = continuous_listener_target {
            self.continuous_listener.start_thread(
                Some(Arc::clone(&imod_thread)),
                Some(continuous_listener_target),
            );
        }
        // Synchronized on stderr.quickListenerQueue to keep other threads from
        // from causing a response to appear on this queue before start up messages
        // are processed.
        let _quick_listener_queue = self.stderr.quick_listener_queue.lock().unwrap();
        // Check the stderr of the 3dmod process for the windowID and the
        if etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_debug_level()
            .is_verbose()
        {
            eprintln!(
                "ImodProcess:open {}",
                utilities::get_date_time_stamp_ms(true)
            );
        }
        while !imod_thread.is_finished() && self.fields.lock().unwrap().window_id == "" {
            while let Some(line) = self.stderr.get_quick_message(self.stderr_reg_id) {
                if line.contains("Window id = ") {
                    let words = utilities::java_lang_string_split(&line, &WHITESPACE);
                    if words.len() < 4 {
                        return Err(ImodManagerException::SystemProcess(SystemProcessException(
                            "Could not parse window ID from imod\n".to_string(),
                        )));
                    }
                    self.fields.lock().unwrap().window_id = words[3].clone();
                    if etomo_director::ARGUMENTS
                        .lock()
                        .unwrap()
                        .get_debug_level()
                        .is_extra_verbose()
                    {
                        eprintln!("Found windowID:{}", words[3]);
                    }
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        // If imod exited before getting the window report the problem to the user
        let (window_id_empty, output_window_id) = {
            let fields = self.fields.lock().unwrap();
            (fields.window_id == "", fields.output_window_id)
        };
        if window_id_empty && output_window_id {
            let mut message = format!(
                "Missing windowID.  3dmod returned: {}\n",
                imod.get_exit_value()
            );

            while let Some(line) = self.stderr.get_quick_message(self.stderr_reg_id) {
                eprintln!("{line}");
                message = message + "stderr: " + &line + "\n";
            }

            // The source reads a second line at the end of the loop body and discards
            // it, so every other stdout line is missing from the message.  Fixed in
            // translation: each line is read once.
            while let Some(line) = imod.read_stdout() {
                message = message + "stdout: " + &line + "\n";
            }

            return Err(ImodManagerException::SystemProcess(SystemProcessException(
                message,
            )));
        }
        Ok(())
    }

    /// Java `quit`.  Send the quit messsage to imod.
    pub fn quit(&self) -> Result<(), ImodManagerException> {
        if self.is_running() {
            let messages = vec![MESSAGE_CLOSE.to_string()];
            self.send(&messages)?;
        }
        Ok(())
    }

    /// Java `disconnect`.  When 3dmod is listening to stdin, it can't quit properly, so
    /// send it a command to stop listening to stdin.
    pub fn disconnect(&self) -> Result<(), ImodManagerException> {
        if self.listen_to_stdin && self.is_running() {
            let messages = vec![MESSAGE_STOP_LISTENING.to_string()];
            self.send_commands_no_wait(&messages)?;
            eprintln!(
                "Telling 3dmod {} to stop listening.",
                self.fields.lock().unwrap().dataset_name
            );
        }
        Ok(())
    }

    /// Java `isRunning`.  Check to see if this 3dmod process is running.
    pub fn is_running(&self) -> bool {
        match self.imod_thread.lock().unwrap().as_ref() {
            None => false,
            Some(imod_thread) => !imod_thread.is_finished(),
        }
    }

    /// Java `setOpenModelMessage`.  Places arguments to open a model on the argument
    /// list.
    pub fn set_open_model_message(&self, new_model_name: &str) {
        let mut fields = self.fields.lock().unwrap();
        fields.model_name = new_model_name.to_string();
        fields.send_arguments.push(MESSAGE_OPEN_MODEL.to_string());
        fields.send_arguments.push(new_model_name.to_string());
    }

    /// Java `openModel`.  Open a new model file.
    pub fn open_model(&self, model: &str, model_mode: bool) -> Result<(), ImodManagerException> {
        let _ = model_mode;
        self.fields.lock().unwrap().model_name = model.to_string();
        let args = vec![
            MESSAGE_OPEN_MODEL.to_string(),
            model.to_string(),
            MESSAGE_MODEL_MODE.to_string(),
        ];
        self.send(&args)
    }

    /// Java `setOpenModelPreserveContrastMessage`.  Places arguments to open a model and
    /// preserve contrast on the argument list.
    pub fn set_open_model_preserve_contrast_message(&self, new_model_name: &str) {
        let mut fields = self.fields.lock().unwrap();
        fields.send_arguments.push(MESSAGE_OPEN_KEEP_BW.to_string());
        fields.send_arguments.push(new_model_name.to_string());
    }

    /// Java `openModelPreserveContrast`.  Open a new model file, Preserve the constrast
    /// settings.
    pub fn open_model_preserve_contrast(
        &self,
        new_model_name: &str,
    ) -> Result<(), ImodManagerException> {
        let args = vec![MESSAGE_OPEN_KEEP_BW.to_string(), new_model_name.to_string()];
        self.send(&args)
    }

    /// Java `saveModel`.  Save the current model file.
    pub fn save_model(&self) -> Result<(), ImodManagerException> {
        let args = vec![MESSAGE_SAVE_MODEL.to_string()];
        self.send(&args)
    }

    /// Java `viewModel`.  View the current model file.
    pub fn view_model(&self) -> Result<(), ImodManagerException> {
        let args = vec![MESSAGE_VIEW_MODEL.to_string()];
        self.send(&args)
    }

    /// Java `setNewContoursMessage`.  Adds a message which sets new contours to be open
    /// Message description: 12 0 1 1 7 0 12 says to do it to a new (empty) contour only
    /// (11 would be unconditional) 0 is for object 1 1 sets it to open 1 sets it to
    /// display circles 7 makes circle size be 7 0 keeps 3D size at 0
    pub fn set_new_contours_message(&self, open: bool) {
        self.set_new_object_message(0, open, CIRCLE, 7, 0);
    }

    /// Java `setPointLimitMessage`.
    pub fn set_point_limit_message(&self, point_limit: i32) {
        self.set_more_object_properties_message(1, point_limit, -1, -1);
    }

    /// Java `setStartNewContoursAtNewZ`.
    pub fn set_start_new_contours_at_new_z(&self) {
        self.set_more_object_properties_message(1, -1, 1, -1);
    }

    /// Java `setInterpolation`.
    pub fn set_interpolation(&self, input: bool) {
        let mut fields = self.fields.lock().unwrap();
        fields
            .send_arguments
            .push(MESSAGE_INTERPOLATION.to_string());
        fields
            .send_arguments
            .push(if input { TRUE } else { FALSE }.to_string());
    }

    /// Java `openSurfContPoint`.
    pub fn open_surf_cont_point(&self) {
        let mut fields = self.fields.lock().unwrap();
        fields.send_arguments.push(MESSAGE_OPEN_DIALOG.to_string());
        fields
            .send_arguments
            .push(SURF_CONT_POINT_DIALOG.to_string());
    }

    /// Java `setNewObjectMessage`.
    pub fn set_new_object_message(
        &self,
        object: i32,
        open: bool,
        symbol: i32,
        size: i32,
        size_3d: i32,
    ) {
        let mut fields = self.fields.lock().unwrap();
        fields
            .send_arguments
            .push(MESSAGE_NEWOBJ_PROPERTIES.to_string());
        fields.send_arguments.push(object.to_string());
        fields
            .send_arguments
            .push(if open { TRUE } else { FALSE }.to_string());
        fields.send_arguments.push(symbol.to_string());
        fields.send_arguments.push(size.to_string());
        fields.send_arguments.push(size_3d.to_string());
    }

    /// Java `setMoreObjectPropertiesMessage`.
    pub fn set_more_object_properties_message(
        &self,
        object: i32,
        point_limit: i32,
        new_contour_in_new_z: i32,
        sphere_in_central_only: i32,
    ) {
        let mut fields = self.fields.lock().unwrap();
        fields
            .send_arguments
            .push(MESSAGE_MORE_OBJ_PROPERTIES.to_string());
        fields.send_arguments.push(object.to_string());
        fields.send_arguments.push(point_limit.to_string());
        fields.send_arguments.push(new_contour_in_new_z.to_string());
        fields
            .send_arguments
            .push(sphere_in_central_only.to_string());
    }

    /// Java `setModelModeMessage`.  Places arguments to set model mode on the argument
    /// list.
    pub fn set_model_mode_message(&self) {
        let mut fields = self.fields.lock().unwrap();
        fields.send_arguments.push(MESSAGE_MODEL_MODE.to_string());
        fields.send_arguments.push("1".to_string());
    }

    /// Java `modelMode`.  Switch the 3dmod process to model mode.
    pub fn model_mode(&self) -> Result<(), ImodManagerException> {
        let args = vec![MESSAGE_MODEL_MODE.to_string()];
        self.send(&args)
    }

    /// Java `setMovieModeMessage`.  Places arguments to set movie mode on the argument
    /// list.
    pub fn set_movie_mode_message(&self) {
        let mut fields = self.fields.lock().unwrap();
        fields.send_arguments.push(MESSAGE_MODEL_MODE.to_string());
        fields.send_arguments.push("0".to_string());
    }

    /// Java `movieMode`.  Switch the 3dmod process to movie mode.
    pub fn movie_mode(&self) -> Result<(), ImodManagerException> {
        let args = vec![MESSAGE_MODEL_MODE.to_string(), "0".to_string()];
        self.send(&args)
    }

    /// Java `setRaise3dmodMessage`.  Places arguments to raise 3dmod on the argument
    /// list.
    pub fn set_raise_3dmod_message(&self) {
        self.fields
            .lock()
            .unwrap()
            .send_arguments
            .push(MESSAGE_RAISE.to_string());
    }

    /// Java `raise3dmod`.  Raise the 3dmod window.
    pub fn raise_3dmod(&self) -> Result<(), ImodManagerException> {
        let args = vec![MESSAGE_RAISE.to_string()];
        self.send(&args)
    }

    /// Java `setOpenZapWindowMessage`.  Places arguments to open one zap window and raise
    /// 3dmod on the argument list.
    pub fn set_open_zap_window_message(&self) {
        self.fields
            .lock()
            .unwrap()
            .send_arguments
            .push(MESSAGE_ONE_ZAP_OPEN.to_string());
    }

    /// Java `openZapWindow`.  Open one zap window and raise 3dmod.
    pub fn open_zap_window(&self) -> Result<(), ImodManagerException> {
        let args = vec![MESSAGE_ONE_ZAP_OPEN.to_string()];
        self.send(&args)
    }

    /// Java `setOpenBeadFixerMessage`.  Places arguments to open the beadfixer dialog on
    /// the argument list.
    pub fn set_open_bead_fixer_message(&self) {
        self.fields
            .lock()
            .unwrap()
            .send_arguments
            .push(MESSAGE_OPEN_BEADFIXER.to_string());
    }

    /// Java `setOpenModelView`.
    pub fn set_open_model_view(&self) {
        self.fields
            .lock()
            .unwrap()
            .send_arguments
            .push(MESSAGE_OPEN_MODEL_VIEW.to_string());
    }

    /// Java `setSkipList`.
    pub fn set_skip_list(&self, skip_list: Option<&str>) {
        if let Some(skip_list) = skip_list {
            self.add_plugin_message_string_string_string(
                BEAD_FIXER_PLUGIN,
                BF_MESSAGE_SKIP_LIST,
                skip_list,
            );
        } else {
            self.add_plugin_message_string_string(BEAD_FIXER_PLUGIN, BF_MESSAGE_REMOVE_SKIP_LIST);
        }
    }

    /// Java `setDeleteAllSections`.
    pub fn set_delete_all_sections(&self, on: bool) {
        if on {
            self.add_plugin_message_string_string_string(
                BEAD_FIXER_PLUGIN,
                BF_MESSAGE_DELETE_ALL_SECTIONS,
                TRUE,
            );
        } else {
            self.add_plugin_message_string_string_string(
                BEAD_FIXER_PLUGIN,
                BF_MESSAGE_DELETE_ALL_SECTIONS,
                FALSE,
            );
        }
    }

    /// Java `setBeadfixerDiameter`.
    pub fn set_beadfixer_diameter(&self, beadfixer_diameter: Option<i32>) {
        if let Some(beadfixer_diameter) = beadfixer_diameter {
            self.fields.lock().unwrap().beadfixer_diameter_set = true;
            self.add_plugin_message_string_string_string(
                BEAD_FIXER_PLUGIN,
                BF_MESSAGE_DIAMETER,
                &beadfixer_diameter.to_string(),
            );
        }
    }

    /// Java `setAutoCenter`.
    pub fn set_auto_center(&self, auto_center: bool) {
        self.add_plugin_message_string_string_string(
            BEAD_FIXER_PLUGIN,
            BF_MESSAGE_AUTO_CENTER,
            if auto_center { MESSAGE_ON } else { MESSAGE_OFF },
        );
        // Make sure that beadfixer diameter is set when this is used.
    }

    /// Java `setNewContours`.
    pub fn set_new_contours(&self, new_contours: bool) {
        self.add_plugin_message_string_string_string(
            BEAD_FIXER_PLUGIN,
            BF_MESSAGE_NEW_CONTOURS,
            if new_contours {
                MESSAGE_ON
            } else {
                MESSAGE_OFF
            },
        );
    }

    /// Java `setBeadfixerMode`.
    pub fn set_beadfixer_mode(&self, beadfixer_mode: BeadFixerMode) {
        self.add_plugin_message_string_string_string(
            BEAD_FIXER_PLUGIN,
            BF_MESSAGE_MODE,
            beadfixer_mode.get_value(),
        );
    }

    /// Java `reopenLog`.
    pub fn reopen_log(&self) -> Result<(), ImodManagerException> {
        self.send_plugin_message(BEAD_FIXER_PLUGIN, BF_MESSAGE_REREAD_LOG)
    }

    /// Java `setOpenLog`.
    pub fn set_open_log(&self, log_name: &str) {
        self.add_plugin_message_string_string_string(
            BEAD_FIXER_PLUGIN,
            BF_MESSAGE_OPEN_LOG,
            log_name,
        );
    }

    /// Java `openBeadFixer`.  Open the beadfixer dialog.
    pub fn open_bead_fixer(&self) -> Result<(), ImodManagerException> {
        let args = vec![MESSAGE_OPEN_BEADFIXER.to_string()];
        self.send(&args)
    }

    /// Java `getRubberbandCoordinates`.  Sends message requesting rubberband
    /// coordinates.  Should not be used with sendMessages().  Returns rubberband
    /// coordinates and error messages.
    pub fn get_rubberband_coordinates(&self) -> Result<Option<Vec<String>>, ImodManagerException> {
        let args = vec![MESSAGE_RUBBERBAND.to_string()];
        self.request(&args)
    }

    /// Java `getSlicerAngles`.
    pub fn get_slicer_angles(&self) -> Result<Option<Vec<String>>, ImodManagerException> {
        let args = vec![MESSAGE_SLICER_ANGLES.to_string()];
        self.request(&args)
    }

    /// Java private `sendPluginMessage`.
    fn send_plugin_message(&self, plugin: &str, message: &str) -> Result<(), ImodManagerException> {
        self.send(&[
            MESSAGE_PLUGIN_MESSAGE.to_string(),
            plugin.to_string(),
            message.to_string(),
        ])
    }

    /// Java private `addPluginMessage(String, String, String)`.
    fn add_plugin_message_string_string_string(&self, plugin: &str, message: &str, value: &str) {
        let mut fields = self.fields.lock().unwrap();
        fields
            .send_arguments
            .push(MESSAGE_PLUGIN_MESSAGE.to_string());
        fields.send_arguments.push(plugin.to_string());
        fields.send_arguments.push(message.to_string());
        fields.send_arguments.push(value.to_string());
    }

    /// Java private `addPluginMessage(String, String)`.
    fn add_plugin_message_string_string(&self, plugin: &str, message: &str) {
        let mut fields = self.fields.lock().unwrap();
        fields
            .send_arguments
            .push(MESSAGE_PLUGIN_MESSAGE.to_string());
        fields.send_arguments.push(plugin.to_string());
        fields.send_arguments.push(message.to_string());
    }

    /// Java `getAxisID`.
    pub fn get_axis_id(&self) -> Option<AxisID> {
        self.axis_id
    }

    /// Java `sendMessages`.  Sends all messages collected in the argument list via
    /// imodSendEvent().  Clears the argument list.
    pub fn send_messages(&self) -> Result<(), ImodManagerException> {
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug()
            && etomo_director::ARGUMENTS.lock().unwrap().is_test()
        {
            eprintln!("sendMessages");
        }
        let send_arguments = self.fields.lock().unwrap().send_arguments.clone();
        if send_arguments.is_empty() {
            return Ok(());
        }
        if etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_debug_level()
            .is_extra_verbose()
        {
            eprint!("sendArguments: ");
            for argument in &send_arguments {
                eprint!("{argument} ");
            }
            if !send_arguments.is_empty() {
                eprintln!();
            }
        }
        let arg_array = send_arguments;
        if !self.listen_to_stdin {
            self.imod_send_event_string_array(&arg_array)?;
        } else {
            self.send_commands_string_array(&arg_array)?;
        }
        self.fields.lock().unwrap().send_arguments.clear();

        // The 3dmod process may have started without a continuous listener target.
        // If a target has been added and the continuous listener thread is not
        // running, start the continuous listener thread.
        let continuous_listener_target = self
            .fields
            .lock()
            .unwrap()
            .continuous_listener_target
            .clone();
        if self.is_running()
            && continuous_listener_target.is_some()
            && !self.continuous_listener.is_alive()
        {
            let imod_thread = self.imod_thread.lock().unwrap().clone();
            self.continuous_listener
                .start_thread(imod_thread, continuous_listener_target);
        }
        Ok(())
    }

    /// Java private `send`.
    fn send(&self, args: &[String]) -> Result<(), ImodManagerException> {
        if !self.listen_to_stdin {
            self.imod_send_event_string_array(args)
        } else {
            self.send_commands_string_array(args)
        }
    }

    /// Java private `request`.
    fn request(&self, args: &[String]) -> Result<Option<Vec<String>>, ImodManagerException> {
        if !self.listen_to_stdin {
            self.imod_send_and_receive(args)
        } else {
            if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                eprintln!("using stdin");
            }
            self.send_request(args).map(Some)
        }
    }

    /// Java `imodSendAndReceive`.  Sends a message and then records the results found in
    /// the error stream.
    pub fn imod_send_and_receive(
        &self,
        args: &[String],
    ) -> Result<Option<Vec<String>>, ImodManagerException> {
        let mut results: Vec<String> = Vec::new();
        if !self.is_running() {
            ui_harness::post_message_dialog(
                Some(self.manager),
                "3dmod is not running.".to_string(),
                "3dmod Warning".to_string(),
                self.axis_id,
            );
            return Ok(None);
        }
        self.imod_send_event_string_array_vector(args, Some(&mut results))?;
        // 3dmod sends the results before it returns
        // the exit value to imodSendEvent - no waiting
        let Some(imod) = self.imod.lock().unwrap().clone() else {
            return Ok(Some(results));
        };
        let Some(mut line) = imod.read_stderr() else {
            return Ok(Some(results));
        };
        // Currently assuming results can only be on one line.
        let mut found_error = false;
        loop {
            if !self.parse_error(&line, &mut results) {
                let words = utilities::java_lang_string_split(&line, &WHITESPACE);
                for word in words {
                    results.push(word);
                }
            } else {
                found_error = true;
            }
            match imod.read_stderr() {
                Some(next) => line = next,
                None => break,
            }
        }
        if found_error {
            ui_harness::post_message_dialog(
                Some(self.manager),
                vector_to_string(&results),
                "3dmod Message".to_string(),
                self.get_axis_id(),
            );
        }
        Ok(Some(results))
    }

    /// Java `parseError`.
    pub fn parse_error(&self, line: &str, error_message: &mut Vec<String>) -> bool {
        // Currently assuming that an error or warning message will be only one
        // line and contain ERROR_STRING or WARNING_STRING.
        if let Some(index) = ProcessMessages::get_error_index(line) {
            error_message.push(line[index..].to_string());
            return true;
        }
        if let Some(index) = line.find(MessageType::Warning.tag().unwrap()) {
            error_message.push(line[index..].to_string());
            return true;
        }
        false
    }

    /// Java private `imodSendEvent(String[])`.
    fn imod_send_event_string_array(&self, args: &[String]) -> Result<(), ImodManagerException> {
        self.imod_send_event_string_array_vector(args, None)
    }

    /// Java private `imodSendEvent(String[], Vector)`.  Send an event to 3dmod using the
    /// imodsendevent command.  Synchronized on stderr.quickListenerQueue to keep other
    /// threads from from causing a response to appear on this queue before this thread
    /// can read the response it generates.
    fn imod_send_event_string_array_vector(
        &self,
        args: &[String],
        messages: Option<&mut Vec<String>>,
    ) -> Result<(), ImodManagerException> {
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!("using imodsendevent");
        }
        let _quick_listener_queue = self.stderr.quick_listener_queue.lock().unwrap();
        let window_id = self.fields.lock().unwrap().window_id.clone();
        if window_id == "" {
            return Err(ImodManagerException::SystemProcess(SystemProcessException(
                "No window ID available for imod".to_string(),
            )));
        }
        let mut command: Vec<String> = Vec::with_capacity(2 + args.len());
        command.push(base_manager::get_imod_bin_path().unwrap_or_default() + "imodsendevent");
        if etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_debug_level()
            .is_extra_verbose()
        {
            eprintln!("imodSendEvent:windowID:{window_id}");
        }
        command.push(window_id);
        // String command = ApplicationManager.getIMODBinPath() + "imodsendevent "
        // + windowID + " ";
        for arg in args {
            command.push(arg.clone());
        }
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug()
            || etomo_director::ARGUMENTS.lock().unwrap().is_test()
        {
            // `System.err.print(command)` prints the array object; the elements are what
            // is recognisable here.
            eprint!("{command:?}");
        }
        let imod_send_event = Arc::new(InteractiveSystemProgram::new(
            self.manager,
            Some(command),
            self.axis_id.unwrap_or(AxisID::Only),
        ));

        // Start the imodSendEvent program thread and wait for it to finish
        let send_event_thread = {
            let imod_send_event = Arc::clone(&imod_send_event);
            std::thread::spawn(move || imod_send_event.run())
        };
        if let Err(except) = send_event_thread.join() {
            eprintln!("{except:?}");
        }
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!("...done");
        }

        // Check imodSendEvent's exit code, if it is not zero read in the
        // stderr/stdout stream and throw an exception describing why the file
        // was not loaded
        if imod_send_event.get_exit_value() != 0 {
            let mut message = format!(
                "{} {}\n",
                IMOD_SEND_EVENT_STRING,
                imod_send_event.get_exit_value()
            );

            let mut line = imod_send_event.read_stderr();
            while let Some(text) = line {
                message = message + "stderr: " + &text + "\n";
                line = imod_send_event.read_stderr();
            }
            line = imod_send_event.read_stdout();
            while let Some(text) = line {
                message = message + "stdout: " + &text + "\n";
                line = imod_send_event.read_stdout();
            }

            match messages {
                None => eprintln!("{message}"),
                Some(messages) => messages.push(message),
            }
        }
        Ok(())
    }

    /// Java private `sendRequest`.  Sends a request to 3dmod's stdin and returns the
    /// results.  Pops up error and warning messages from 3dmod that are directed at the
    /// user.  Synchronized on stderr.quickListenerQueue to keep other threads from from
    /// causing a response to appear on this queue before this thread can read the
    /// response it generates.  Returns the values received from 3dmod.
    fn send_request(&self, args: &[String]) -> Result<Vec<String>, ImodManagerException> {
        let _quick_listener_queue = self.stderr.quick_listener_queue.lock().unwrap();
        let imod_return_values = Vec::new();
        let imod_return_values =
            self.send_commands_string_array_vector_boolean(args, Some(imod_return_values), true)?;
        Ok(imod_return_values.unwrap_or_default())
    }

    /// Java private `sendCommands(String[])`.  Sends commands to 3dmod's stdin and
    /// process the results.  Pops up error and warning messages from 3dmod that are
    /// directed at the user.  Synchronized on stderr.quickListenerQueue.
    fn send_commands_string_array(&self, args: &[String]) -> Result<(), ImodManagerException> {
        let _quick_listener_queue = self.stderr.quick_listener_queue.lock().unwrap();
        self.send_commands_string_array_vector_boolean(args, None, true)?;
        Ok(())
    }

    /// Java private `sendCommandsNoWait`.
    fn send_commands_no_wait(&self, args: &[String]) -> Result<(), ImodManagerException> {
        self.send_commands_string_array_vector_boolean(args, None, false)?;
        Ok(())
    }

    /// Java private `sendCommands(String[], Vector, boolean)`.  Sends commands to
    /// 3dmod's stdin and process the results.  Pops up error and warning messages from
    /// 3dmod that are directed at the user.
    ///
    /// `imodReturnValues` is the optional return value vector to be used when expecting
    /// return values from 3dmod; Java fills the caller's vector, so the filled vector is
    /// handed back.
    fn send_commands_string_array_vector_boolean(
        &self,
        args: &[String],
        imod_return_values: Option<Vec<String>>,
        read_response: bool,
    ) -> Result<Option<Vec<String>>, ImodManagerException> {
        let process = self
            .this
            .upgrade()
            .expect("an ImodProcess is only reachable through its Arc");
        let mut message_sender =
            MessageSender::new(process, args.to_vec(), imod_return_values, read_response);
        /* //patch for quicklistener shared queue problem try { Thread.sleep(200); } catch
         * (InterruptedException e) { } */
        if message_sender.imod_return_values.is_none() {
            std::thread::spawn(move || message_sender.run());
            Ok(None)
        } else {
            // get return values
            message_sender.run();
            Ok(message_sender.imod_return_values)
        }
    }

    /// Java `processRequest`.
    pub fn process_request(&self) {
        if self.is_request_received()
            && let Err(e) = self.disconnect()
        {
            eprintln!("{e}");
        }
    }

    /// Java private `isRequestReceived`.
    fn is_request_received(&self) -> bool {
        if self.stderr.get_request_message().is_some() {
            return true;
        }
        false
    }

    /// Java `getDatasetName`.
    pub fn get_dataset_name(&self) -> String {
        self.fields.lock().unwrap().dataset_name.clone()
    }

    /// Java `getModelName`.
    pub fn get_model_name(&self) -> String {
        self.fields.lock().unwrap().model_name.clone()
    }

    /// Java `getWindowID`.
    pub fn get_window_id(&self) -> String {
        let window_id = self.fields.lock().unwrap().window_id.clone();
        if etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_debug_level()
            .is_extra_verbose()
        {
            eprintln!("getWindowID:windowID:{window_id}");
        }
        window_id
    }

    /// Java `getSwapYZ`.
    pub fn get_swap_yz(&self) -> bool {
        self.fields.lock().unwrap().swap_yz
    }

    /// Java `setSwapYZ`.
    pub fn set_swap_yz(&self, state: bool) {
        self.fields.lock().unwrap().swap_yz = state;
    }

    /// Java `isModelView`.
    pub fn is_model_view(&self) -> bool {
        self.fields.lock().unwrap().model_view
    }

    /// Java `setModelView`.
    pub fn set_model_view(&self, model_view: bool) {
        self.fields.lock().unwrap().model_view = model_view;
    }

    /// Java `setOpenZap`.
    pub fn set_open_zap(&self) {
        self.fields.lock().unwrap().open_zap = true;
    }

    /// Java `setTiltFile`.
    pub fn set_tilt_file(&self, input: Option<&str>) {
        self.fields.lock().unwrap().tilt_file = input.map(str::to_string);
    }

    /// Java `resetTiltFile`.
    pub fn reset_tilt_file(&self) {
        self.fields.lock().unwrap().tilt_file = None;
    }

    /// Java `isUseModv`.
    pub fn is_use_modv(&self) -> bool {
        self.fields.lock().unwrap().use_modv
    }

    /// Java `setUseModv`.
    pub fn set_use_modv(&self, b: bool) {
        self.fields.lock().unwrap().use_modv = b;
    }

    /// Java `isOutputWindowID`.
    pub fn is_output_window_id(&self) -> bool {
        self.fields.lock().unwrap().output_window_id
    }

    /// Java `setOutputWindowID`.
    pub fn set_output_window_id(&self, b: bool) {
        self.fields.lock().unwrap().output_window_id = b;
    }

    /// Java `setDebug`.
    pub fn set_debug(&self, input: bool) {
        self.fields.lock().unwrap().debug = input;
    }

    /// Java `setBinning`.
    pub fn set_binning(&self, binning: i32) {
        let mut fields = self.fields.lock().unwrap();
        if binning < DEFAULT_BINNING {
            fields.binning = DEFAULT_BINNING;
        } else {
            fields.binning = binning;
        }
    }

    /// Java `setBinningXY`.
    pub fn set_binning_xy(&self, binning_xy: i32) {
        let mut fields = self.fields.lock().unwrap();
        if binning_xy < DEFAULT_BINNING {
            fields.binning_xy = DEFAULT_BINNING;
        } else {
            fields.binning_xy = binning_xy;
        }
    }

    /// Java `paramString`.
    pub fn param_string(&self) -> String {
        let fields = self.fields.lock().unwrap();
        format!(
            ",datasetName={}, modelName={}, windowID={}, swapYZ={}, modelView={}, useModv={}, outputWindowID={}, binning={}",
            fields.dataset_name,
            fields.model_name,
            fields.window_id,
            fields.swap_yz,
            fields.model_view,
            fields.use_modv,
            fields.output_window_id,
            fields.binning
        )
    }

    /// Java `addWindowOpenOption`.
    pub fn add_window_open_option(&self, option: WindowOpenOption) {
        let mut fields = self.fields.lock().unwrap();
        if option.is_imodv() && !fields.model_view && !fields.use_modv {
            eprintln!(
                "WARNING:  Can't use 3dmod {} with {} because the Model View is not open.",
                WindowOpenOption::OPTION,
                option
            );
        }
        fields
            .window_open_option_list
            .get_or_insert_with(Vec::new)
            .push(option);
    }

    /// Java `setContinuousListenerTarget`.
    pub fn set_continuous_listener_target(
        &self,
        continuous_listener_target: Option<Arc<dyn ContinuousListenerTarget>>,
    ) {
        self.fields.lock().unwrap().continuous_listener_target = continuous_listener_target;
    }
}

/// Java `toString`: `getClass().getName() + "[" + paramString() + "]"`.
impl std::fmt::Display for ImodProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "etomo.process.ImodProcess[{}]", self.param_string())
    }
}

/// Java package-private static final nested `QuickListenerQueueTestWrapper`.  Class to
/// allow testing of the quick listener queue functionality in Stderr.
pub struct QuickListenerQueueTestWrapper {
    stderr: Arc<Stderr>,
}

impl QuickListenerQueueTestWrapper {
    /// Java `QuickListenerQueueTestWrapper()`.
    pub fn new() -> QuickListenerQueueTestWrapper {
        QuickListenerQueueTestWrapper {
            stderr: Stderr::new(),
        }
    }

    /// Java `getExpectedRegistrants`.
    pub fn get_expected_registrants(&self) -> i32 {
        Stderr::EXPECTED_REGISTRANTS
    }

    /// Java `register`.
    pub fn register(&self) -> i32 {
        self.stderr.register()
    }

    /// Java `getPurgeSize`.
    pub fn get_purge_size(&self) -> i32 {
        Stderr::PURGE_SIZE
    }

    /// Java `add`: `stderr.quickListenerQueue.add(input)`, with no purge.
    pub fn add(&self, input: &str) {
        self.stderr
            .data
            .lock()
            .unwrap()
            .quick_listener_queue
            .push(input.to_string());
    }

    /// Java `getQuickMessage`.
    pub fn get_quick_message(&self, reg_id: i32) -> Option<String> {
        self.stderr.get_quick_message(reg_id)
    }

    /// Java `purge`.
    pub fn purge(&self) {
        let mut data = self.stderr.data.lock().unwrap();
        Stderr::purge_quick_listener_queue(&mut data);
    }
}

impl Default for QuickListenerQueueTestWrapper {
    fn default() -> QuickListenerQueueTestWrapper {
        QuickListenerQueueTestWrapper::new()
    }
}

/// Java `toString`: `stderr.quickListenerQueue.toString()`.
impl std::fmt::Display for QuickListenerQueueTestWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&vector_to_string(
            &self.stderr.data.lock().unwrap().quick_listener_queue,
        ))
    }
}

/// The fields of `Stderr` guarded by its instance monitor.
struct StderrData {
    /// Contains an id and the last index used to read quickListenerQueue.
    registration: HashMap<i32, EtomoNumber>,
    /// Queue to hold returned data that was requested by etomo, and also error messages.
    /// Also contains miscellaneous messages that can be ignored.
    quick_listener_queue: Vec<String>,
    /// Queue to hold information from 3dmod.  These messages are requested by etomo but
    /// do not arrive instantly.
    continuous_listener_queue: VecDeque<String>,
    /// Queue to hold requests from 3dmod.  3dmod chooses when to send these messages.
    request_queue: VecDeque<String>,
    imod: Option<Arc<InteractiveSystemProgram>>,
    received_interrupted_exception: bool,
    reg_id: i32,
}

/// Java private static final nested `Stderr`.  Class to get messages from the stderr and
/// place them in queues.  This is only way that imod.stderr should be accessed.
struct Stderr {
    /// The Java instance monitor and the fields it guards.
    data: Mutex<StderrData>,
    /// The monitor of the `quickListenerQueue` object, which `ImodProcess` synchronizes
    /// on.  The list itself is in `data`.
    quick_listener_queue: Mutex<()>,
}

impl Stderr {
    const EXPECTED_REGISTRANTS: i32 = 2;
    const PURGE_SIZE: i32 = 10;

    /// Java private `Stderr()`.
    fn new() -> Arc<Stderr> {
        let stderr = Arc::new(Stderr {
            data: Mutex::new(StderrData {
                registration: HashMap::new(),
                quick_listener_queue: Vec::new(),
                continuous_listener_queue: VecDeque::new(),
                request_queue: VecDeque::new(),
                imod: None,
                received_interrupted_exception: false,
                reg_id: -1,
            }),
            quick_listener_queue: Mutex::new(()),
        });
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!(
                "Stderr:etomo.process.ImodProcess$Stderr@{:p}",
                Arc::as_ptr(&stderr)
            );
        }
        stderr
    }

    /// Java private `setImod`.
    fn set_imod(&self, imod: Option<Arc<InteractiveSystemProgram>>) {
        self.data.lock().unwrap().imod = imod;
    }

    /// Java private synchronized `register`.  Creates a new id and adds it to
    /// registration.  Returns the id.
    fn register(&self) -> i32 {
        let mut data = self.data.lock().unwrap();
        data.reg_id += 1;
        let id = data.reg_id;
        let mut index = EtomoNumber::new();
        index.set_int(-1);
        data.registration.insert(id, index);
        id
    }

    /// Java private synchronized `getQuickMessage`.  Returns one line from the
    /// quickListenerQueue, or null if the queue is empty.
    fn get_quick_message(&self, reg_id: i32) -> Option<String> {
        let mut data = self.data.lock().unwrap();
        let _queue_size = data.quick_listener_queue.len();
        Stderr::read_stderr(&mut data);
        let queue_len = data.quick_listener_queue.len() as i32;
        // An id that was never registered has no index; the source dereferences the
        // null.  Every caller passes an id `register` returned.
        let index = data.registration.get_mut(&reg_id)?;
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!("regId:{reg_id},index:{index}");
        }
        // Read a string from the queue, if there is anything left to read.
        if index.lt_int(queue_len - 1) {
            // Increment the index.
            index.add_int(1);
            let index = index.get_int() as usize;
            return data.quick_listener_queue.get(index).cloned();
        }
        None
    }

    /// Java private synchronized `purgeQuickListenerQueue`.  Don't let the quick listener
    /// queue grow too big.  Make sure that all interested parties are registered before
    /// doing any purging.  This function is not efficient at all but it isn't likely to
    /// be used very much.  To make it efficient, make in the quick listener queue a real
    /// link list.  Called with the instance monitor held.
    fn purge_quick_listener_queue(data: &mut StderrData) {
        if (data.registration.len() as i32) < Stderr::EXPECTED_REGISTRANTS
            || (data.quick_listener_queue.len() as i32) < Stderr::PURGE_SIZE
        {
            return;
        }
        // Get the lowest index, which is the last string that all the registrants have
        // read.
        let mut read_by_all_index = -1;
        let mut i = data.registration.values();
        if let Some(first) = i.next() {
            read_by_all_index = first.get_int();
            for next in i {
                read_by_all_index = read_by_all_index.min(next.get_int());
            }
        }
        // Purging is expensive - decide if its worth purging.
        if read_by_all_index >= Stderr::PURGE_SIZE / 2 {
            for _ in 0..=read_by_all_index {
                data.quick_listener_queue.remove(0);
            }
            // Now all the indexes are wrong - fix them.
            for index in data.registration.values_mut() {
                // Reduce the saved indices by the number of elements that where removed.
                let value = index.get_int() - read_by_all_index - 1;
                index.set_int(value);
            }
        }
    }

    /// Java private `getContinuousMessage`.  Removes and returns one message from the
    /// continuousListenerQueue, or null if the queue is empty.  Assuming that only one
    /// continuous listener exists per 3dmod instance.
    fn get_continuous_message(&self) -> Option<String> {
        let mut data = self.data.lock().unwrap();
        Stderr::read_stderr(&mut data);
        data.continuous_listener_queue.pop_front()
    }

    /// Java private `getRequestMessage`.  Removes and returns one message from the
    /// requestQueue, or null if the queue is empty.
    fn get_request_message(&self) -> Option<String> {
        let mut data = self.data.lock().unwrap();
        Stderr::read_stderr(&mut data);
        data.request_queue.pop_front()
    }

    /// Java private synchronized `readStderr`.  Sleeps and then moves all stderr messages
    /// found into a queue.  Messages that start with REQUEST_TAG go to the requeueQueue.
    /// Messages that start with CONTINOUS_TAG go to the continuousListenerQueue.  All
    /// other messages go to the quickListenerQueue.  This function should only be called
    /// by Stderr functions, with the instance monitor held (it sleeps holding it, as the
    /// source does).  A Rust sleep is not interruptible, so
    /// `receivedInterruptedException` stays false.
    fn read_stderr(data: &mut StderrData) {
        std::thread::sleep(Duration::from_millis(500));
        let Some(imod) = data.imod.clone() else {
            return;
        };
        while let Some(message) = imod.read_stderr() {
            if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                eprintln!("stderr:{message}");
            }
            if message.starts_with(REQUEST_TAG) && message.contains(STOP_LISTENING_REQUEST) {
                data.request_queue.push_back(message);
            } else if message.starts_with(CONTINUOUS_TAG) {
                data.continuous_listener_queue.push_back(message);
            } else {
                data.quick_listener_queue.push(message);
                Stderr::purge_quick_listener_queue(data);
            }
        }
    }
}

/// Java private static nested `ContinuousListener implements Runnable`.
struct ContinuousListener {
    /// Java's `this`, for `startThread`.
    this: Weak<ContinuousListener>,
    stderr: Arc<Stderr>,
    axis_id: Option<AxisID>,
    /// The instance monitor `startThread` and `run` are synchronized on.
    monitor: Mutex<()>,
    imod_thread: Mutex<Option<Arc<JoinHandle<()>>>>,
    continuous_listener_thread: Mutex<Option<JoinHandle<()>>>,
    target: Mutex<Option<Arc<dyn ContinuousListenerTarget>>>,
}

impl ContinuousListener {
    /// Java private `ContinuousListener(Stderr, AxisID)`.
    fn new(stderr: Arc<Stderr>, axis_id: Option<AxisID>) -> Arc<ContinuousListener> {
        Arc::new_cyclic(|this| ContinuousListener {
            this: this.clone(),
            stderr,
            axis_id,
            monitor: Mutex::new(()),
            imod_thread: Mutex::new(None),
            continuous_listener_thread: Mutex::new(None),
            target: Mutex::new(None),
        })
    }

    /// Java private synchronized `startThread`.  Set imodThread and target and run the
    /// run() function on a separate thread.
    fn start_thread(
        &self,
        imod_thread: Option<Arc<JoinHandle<()>>>,
        continuous_listener_target: Option<Arc<dyn ContinuousListenerTarget>>,
    ) {
        let _monitor = self.monitor.lock().unwrap();
        *self.imod_thread.lock().unwrap() = imod_thread;
        *self.target.lock().unwrap() = continuous_listener_target;
        // If thread has ended create and start a new thread
        let mut continuous_listener_thread = self.continuous_listener_thread.lock().unwrap();
        if continuous_listener_thread
            .as_ref()
            .is_none_or(|thread| thread.is_finished())
        {
            let this = self
                .this
                .upgrade()
                .expect("a ContinuousListener is only reachable through its Arc");
            *continuous_listener_thread = Some(std::thread::spawn(move || this.run()));
        }
    }

    /// Java private `isAlive`.
    fn is_alive(&self) -> bool {
        self.continuous_listener_thread
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
    }

    /// Java synchronized `run`.  Check stderr.continuousListenerQueue until imodThread is
    /// no longer alive or an interrupted exception is received.
    fn run(&self) {
        let _monitor = self.monitor.lock().unwrap();
        if etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_debug_level()
            .is_extra_verbose()
        {
            eprintln!(
                "ContinuousListener:run {}",
                utilities::get_date_time_stamp_ms(true)
            );
        }
        loop {
            std::thread::sleep(Duration::from_millis(500));
            let message = self.stderr.get_continuous_message();
            let target = self.target.lock().unwrap().clone();
            if let (Some(message), Some(target)) = (message, target) {
                target.get_continuous_message(&message, self.axis_id);
            }
            let alive = self
                .imod_thread
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|imod_thread| !imod_thread.is_finished());
            if !alive {
                break;
            }
        }
    }
}

/// Java private nested `MessageSender implements Runnable`.  Class to send a message to
/// 3dmod.  Can be run on a separate thread to avoid locking up the GUI.
struct MessageSender {
    /// The enclosing `ImodProcess`.
    process: Arc<ImodProcess>,
    args: Vec<String>,
    imod_return_values: Option<Vec<String>>,
    read_response: bool,
}

impl MessageSender {
    /// Java private `MessageSender(String[], Vector, boolean)`.
    fn new(
        process: Arc<ImodProcess>,
        args: Vec<String>,
        imod_return_values: Option<Vec<String>>,
        read_response: bool,
    ) -> MessageSender {
        MessageSender {
            process,
            args,
            imod_return_values,
            read_response,
        }
    }

    /// Java `run`.  Send the message and wait for a response.
    fn run(&mut self) {
        let process = Arc::clone(&self.process);
        // make sure that 3dmod is running
        let Some(imod) = process.imod.lock().unwrap().clone() else {
            if self.imod_return_values.is_some() {
                // unable to get return values
                ui_harness::post_message_dialog(
                    Some(process.manager),
                    "3dmod is not running.".to_string(),
                    "3dmod Warning".to_string(),
                    process.get_axis_id(),
                );
            }
            return;
        };
        // boolean responseReceived = false;
        // build a string to send
        let mut buffer = String::new();
        for arg in &self.args {
            buffer.push_str(&(arg.clone() + " "));
        }
        if !buffer.is_empty() {
            if etomo_director::ARGUMENTS.lock().unwrap().is_debug()
                || etomo_director::ARGUMENTS.lock().unwrap().is_test()
            {
                eprintln!(
                    "MessageSender:{:p},{}",
                    self as *const MessageSender, buffer
                );
            }
            // send the string to 3dmod's stdin
            if !process.is_running() {
                if self.imod_return_values.is_some() {
                    // unable to get return values
                    ui_harness::post_message_dialog(
                        Some(process.manager),
                        "3dmod is not running.".to_string(),
                        "3dmod Warning".to_string(),
                        process.get_axis_id(),
                    );
                }
                return;
            }
            if etomo_director::ARGUMENTS
                .lock()
                .unwrap()
                .get_debug_level()
                .is_extra_verbose()
            {
                eprintln!(
                    "ImodProcess:MessageSender:run:Setting stdin {}",
                    utilities::get_date_time_stamp_ms(true)
                );
            }
            if let Err(exception) = imod.set_current_std_input(&buffer) {
                // make sure that 3dmod is running
                if exception.to_string().to_lowercase().contains("broken pipe") {
                    if self.imod_return_values.is_some() {
                        // unable to get return values
                        ui_harness::post_message_dialog(
                            Some(process.manager),
                            "3dmod is not running.".to_string(),
                            "3dmod Warning".to_string(),
                            process.get_axis_id(),
                        );
                    }
                    return;
                } else {
                    eprintln!("{exception}");
                    ui_harness::post_message_dialog(
                        Some(process.manager),
                        exception.to_string(),
                        "3dmod Exception".to_string(),
                        process.get_axis_id(),
                    );
                }
            }
        }
        if self.read_response {
            // read the response from 3dmod
            self.read_response();
        }
    }

    /// Java `readResponse`.  Wait for a response to the message and pop up a message if
    /// there is a problem.
    fn read_response(&mut self) {
        let process = Arc::clone(&self.process);
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!(
                "MessageSender:{:p},readResponse",
                self as *const MessageSender
            );
        }
        let mut response_received = false;
        let mut user_message = String::new();
        // wait for the response for at most 5 seconds
        if etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_debug_level()
            .is_extra_verbose()
        {
            eprintln!(
                "ImodProcess:MessageSender:readResponse {}",
                utilities::get_date_time_stamp_ms(true)
            );
        }
        for _timeout in 0..30 {
            if response_received {
                if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                    eprintln!(
                        "MessageSender:{:p},responseReceived",
                        self as *const MessageSender
                    );
                }
                break;
            }
            // process response
            let failure = false;
            while let Some(response) = process
                .stderr
                .get_quick_message(process.message_sender_reg_id)
            {
                response_received = true;
                if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                    eprintln!(
                        "MessageSender:{:p},{}",
                        self as *const MessageSender, response
                    );
                }
                // `String.trim()` strips code units <= ' '.
                let response = response.trim_matches(|c: char| c <= ' ');
                if response == "OK" {
                    // OK is sent last, so this is done
                    break;
                }
                // if the response is not OK or an error message meant for the user
                // then it may be a requested return string. Otherwise it is some
                // 3dmod output that etomo can ignore.
                if !self.parse_user_messages(response, &mut user_message)
                    && self.imod_return_values.is_some()
                    && !failure
                    && !response.starts_with("imodExecuteMessage:")
                {
                    let words = utilities::java_lang_string_split(response, &WHITESPACE);
                    let imod_return_values = self.imod_return_values.as_mut().unwrap();
                    for word in words {
                        imod_return_values.push(word);
                    }
                }
            }
        }
        // pop up error and warning messages for the user
        if !user_message.is_empty() {
            ui_harness::post_message_dialog(
                Some(process.manager),
                user_message,
                "3dmod Message".to_string(),
                process.get_axis_id(),
            );
        }
        if !response_received {
            if process.is_running() {
                if (etomo_director::ARGUMENTS.lock().unwrap().is_debug()
                    || etomo_director::ARGUMENTS.lock().unwrap().is_test())
                    && process
                        .stderr
                        .data
                        .lock()
                        .unwrap()
                        .received_interrupted_exception
                {
                    eprintln!("\nsleep interrupted");
                }
                // no response received and 3dmod is running - "throw" exception
                let (dataset_name, model_name, working_directory) = {
                    let fields = process.fields.lock().unwrap();
                    (
                        fields.dataset_name.clone(),
                        fields.model_name.clone(),
                        fields.working_directory.clone(),
                    )
                };
                let exception = SystemProcessException(format!(
                    "MessageSender:{:p},No response received from 3dmod.  datasetName={},modelName={},workingDirectory={},axisID={}",
                    self as *const MessageSender,
                    dataset_name,
                    model_name,
                    working_directory
                        .map_or_else(|| "null".to_string(), |d| d.display().to_string()),
                    process
                        .axis_id
                        .map_or_else(|| "null".to_string(), |a| a.to_string())
                ));
                eprintln!("etomo.process.SystemProcessException: {exception}");
                ui_harness::post_message_dialog(
                    Some(process.manager),
                    exception.to_string(),
                    "3dmod Exception".to_string(),
                    process.get_axis_id(),
                );
            } else if self.imod_return_values.is_some() {
                // unable to get return values
                ui_harness::post_message_dialog(
                    Some(process.manager),
                    "3dmod is not running.".to_string(),
                    "3dmod Warning".to_string(),
                    process.get_axis_id(),
                );
            }
        }
    }

    /// Java private `parseUserMessages`.  Parse messages that are directed at the user -
    /// messages that contain ERROR_TAG or WARNING_TAG.  Returns true if an error or
    /// warning is found.
    fn parse_user_messages(&self, line: &str, user_messages: &mut String) -> bool {
        // Currently assuming that each user error or warning messages will be
        // only one
        // line and contain ERROR_STRING or WARNING_STRING.
        if ProcessMessages::get_error_index(line).is_some() {
            user_messages.push_str(&(line.to_string() + "\n"));
            return true;
        }
        if line.contains(MessageType::Warning.tag().unwrap()) {
            user_messages.push_str(&(line.to_string() + "\n"));
            return true;
        }
        false
    }
}

/// Java package-private static nested `WindowOpenOption`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowOpenOption {
    /// Java `IMODV_OBJECTS`, `new WindowOpenOption("O", true)`.
    ImodvObjects,
    /// Java `ISOSURFACE`, `new WindowOpenOption("U", true)`.
    Isosurface,
    /// Java `OBJECT_LIST`, `new WindowOpenOption("L", true)`.
    ObjectList,
    /// Java `MODEL_EDIT`, `new WindowOpenOption("M", true)`.
    ModelEdit,
}

impl WindowOpenOption {
    pub const OPTION: &'static str = "-E";

    /// Java field `windowKey`.
    fn window_key(self) -> &'static str {
        match self {
            WindowOpenOption::ImodvObjects => "O",
            WindowOpenOption::Isosurface => "U",
            WindowOpenOption::ObjectList => "L",
            WindowOpenOption::ModelEdit => "M",
        }
    }

    /// Java `isImodv`.
    pub fn is_imodv(self) -> bool {
        match self {
            WindowOpenOption::ImodvObjects
            | WindowOpenOption::Isosurface
            | WindowOpenOption::ObjectList
            | WindowOpenOption::ModelEdit => true,
        }
    }
}

/// Java `toString`.
impl std::fmt::Display for WindowOpenOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.window_key())
    }
}

/// Java public static final nested `BeadFixerMode`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BeadFixerMode {
    /// Java `SEED_MODE`, "0".
    SeedMode,
    /// Java `GAP_MODE`, "1".
    GapMode,
    /// Java `RESIDUAL_MODE`, "2".
    ResidualMode,
    /// Java `PATCH_TRACKING_RESIDUAL_MODE`, "3".
    PatchTrackingResidualMode,
}

impl BeadFixerMode {
    /// Java private `getValue`.
    fn get_value(self) -> &'static str {
        match self {
            BeadFixerMode::SeedMode => "0",
            BeadFixerMode::GapMode => "1",
            BeadFixerMode::ResidualMode => "2",
            BeadFixerMode::PatchTrackingResidualMode => "3",
        }
    }
}
