//! `IMOD/Etomo/src/etomo/process/ImodState.java`.
//!
//! ImodState constructs a private ImodProcess instance.  It should be used to perform
//! any function on the ImodProcess instance that is required by the ImodManager.
//!
//! ImodState preserves the original state configuration during multiple opens, raises
//! and re-opens of 3dmod, all of which are handle by the open() function.  It also
//! allows a new state configuration to temporarily override the original state
//! configuration for a single call to open().  Conflicting new configurations can be
//! created when calling open() several times, without any negative effect.  To allow
//! state information to be totally controlled by the user, don't change it in reset().
//!
//! Use:
//! Construction and initialization:
//! 1. Construct an ImodState.
//! 2. Use the setInitial functions to set the initial state.  ImodState resets to the
//!    initial state after each open call.
//! Opening and modeling:
//! 1. Use the set functions to set the current state.
//! 2. Call an open function.
//!
//! (The source's long "Upgrading" and "Rules for upgrading" notes describe how to add a
//! state variable; they are a maintenance guide and are not repeated here.)
//!
//! **Shape.**  An `ImodState` is held in `BaseImodManager`'s map and handed out as an
//! `Arc`, so it is used from whichever thread calls the manager; its mutable fields sit
//! behind one lock and every method takes `&self`.  `open` does not hold that lock while
//! `ImodProcess.open` waits for 3dmod to start.
//!
//! **Null manager.**  As in `ImodProcess`, `manager` is `&'static dyn BaseManager`; the
//! source's `manager == null` branches are not reachable and are not kept.

use super::base_imod_manager::ImodManagerException;
use super::continuous_listener_target::ContinuousListenerTarget;
use super::imod_process::{BeadFixerMode, ImodProcess, Run3dmodMenuOptions, WindowOpenOption};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::util::utilities;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// constants
// mode
pub const MODEL_MODE: i32 = -1;
pub const MOVIE_MODE: i32 = -2;
// modelView
pub const MODEL_VIEW: i32 = -3;
// useModv
pub const MODV: i32 = -4;

// default state information
const DEFAULT_OPEN_WITH_MODEL: bool = false;
const DEFAULT_PRESERVE_CONTRAST: bool = false;
const DEFAULT_OPEN_BEAD_FIXER: bool = false;
const DEFAULT_OPEN_CONTOURS: bool = false;
const DEFAULT_FRAMES: bool = false;
const DEFAULT_BINNING: i32 = 1;

/// The mutable instance fields of `ImodState`.
struct ImodStateFields {
    // unchanging state information
    model_view: bool,
    use_modv: bool,

    // current state information
    // reset to initial state
    model_name: Option<String>,
    mode: i32,
    swap_yz: bool,
    // reset to default state
    preserve_contrast: bool,
    open_bead_fixer: bool,
    open_contours: bool,
    start_new_contours_at_new_z: bool,
    point_limit: i32,

    // sent with open bead fixer
    set_auto_center: bool,
    auto_center: bool,
    new_contours: bool,
    manage_new_contours: bool,
    beadfixer_mode: Option<BeadFixerMode>,
    skip_list: Option<String>,

    // signals that a state variable has been changed at least once, so the
    // corrosponding message must always be sent
    using_mode: bool,

    // don't reset
    allow_menu_binning_in_z: bool,
    no_menu_options: bool,

    // reset values
    // initial state information
    initial_model_name: String,
    initial_mode: i32,
    initial_swap_yz: bool,

    // internal state information
    warned_stale_file: bool,
    // initial state information
    initial_mode_set: bool,
    initial_swap_yz_set: bool,

    log_name: Option<String>,
    debug: bool,
    // Should be turned off after each use. This is because it is rarely used and
    // should not be on for most situations. This way I don't have to keep track
    // of when it is on.
    delete_all_sections: Option<EtomoBoolean2>,
    file_name: Option<String>,
    interpolation: Option<EtomoBoolean2>,
    model_name_list: Option<Vec<String>>,

    open_surf_cont_point: bool,
}

impl ImodStateFields {
    /// The Java field initialisers.  `mode`, `swapYZ`, `preserveContrast`,
    /// `openBeadFixer` and `openContours` have none (Java zero values); every
    /// constructor ends in `reset()`, which sets them.
    fn initial() -> ImodStateFields {
        ImodStateFields {
            model_view: false,
            use_modv: false,
            model_name: None,
            mode: 0,
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
            warned_stale_file: false,
            initial_mode_set: false,
            initial_swap_yz_set: false,
            log_name: None,
            debug: false,
            delete_all_sections: None,
            file_name: None,
            interpolation: None,
            model_name_list: None,
            open_surf_cont_point: false,
        }
    }
}

/// Java final `ImodState`.
pub struct ImodState {
    /// Java `axisID`.  Nullable: `BaseImodManager` builds states for keys with no axis.
    axis_id: Option<AxisID>,
    /// Java final `process`.
    process: Arc<ImodProcess>,
    /// Java `fileNameArray`, assigned only by constructors.
    file_name_array: Option<Vec<String>>,
    /// Java `fileList`, assigned only by constructors.
    file_list: Option<Vec<PathBuf>>,
    /// Java final `manager`.
    manager: &'static dyn BaseManager,
    fields: Mutex<ImodStateFields>,
}

impl ImodState {
    /// The field initialisers and the assignments every constructor makes.
    fn construct(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        process: Arc<ImodProcess>,
        file_name_array: Option<Vec<String>>,
        file_list: Option<Vec<PathBuf>>,
        fields: ImodStateFields,
    ) -> ImodState {
        ImodState {
            axis_id,
            process,
            file_name_array,
            file_list,
            manager,
            fields: Mutex::new(fields),
        }
    }

    // constructors
    // they can set final state variables
    // they can also set initialModelName

    /// Java `ImodState(BaseManager, AxisID)`.  Use this constructor to create an
    /// instance of ImodProcess using ImodProcess().
    pub fn new_base_manager_axis_id(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let process = ImodProcess::new_base_manager_axis_id(manager, axis_id);
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, int, AxisID)`.  Use this constructor to create an
    /// instance of ImodProcess using ImodProcess() and set either model view or imodv.
    pub fn new_base_manager_int_axis_id(
        manager: &'static dyn BaseManager,
        model_view_type: i32,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let process = ImodProcess::new_base_manager_axis_id(manager, axis_id);
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.set_model_view_type(model_view_type);
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, String, AxisID)`.  Use this constructor to create an
    /// instance of ImodProcess using ImodProcess(String dataset).
    pub fn new_base_manager_string_axis_id(
        manager: &'static dyn BaseManager,
        file_name: &str,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let process = ImodProcess::new_base_manager_string_axis_id(manager, file_name, axis_id);
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, String, AxisID, FileType)`.  Use this constructor to
    /// create an instance of ImodProcess using ImodProcess(String dataset).
    pub fn new_base_manager_string_axis_id_file_type(
        manager: &'static dyn BaseManager,
        dataset_name: &str,
        axis_id: Option<AxisID>,
        file_type: &FileType,
    ) -> ImodState {
        let process = ImodProcess::new_base_manager_string_axis_id_file_type(
            manager,
            dataset_name,
            axis_id,
            file_type,
        );
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, String, int, AxisID, FileType)`.  Use this
    /// constructor to create an instance of ImodProcess using ImodProcess(String
    /// dataset) and set either model view or imodv.
    pub fn new_base_manager_string_int_axis_id_file_type(
        manager: &'static dyn BaseManager,
        dataset_name: &str,
        model_view_type: i32,
        axis_id: Option<AxisID>,
        file_type: &FileType,
    ) -> ImodState {
        let process = ImodProcess::new_base_manager_string_axis_id_file_type(
            manager,
            dataset_name,
            axis_id,
            file_type,
        );
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.set_model_view_type(model_view_type);
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, String, int, AxisID)`.  Use this constructor to
    /// create an instance of ImodProcess using ImodProcess(String dataset) and set either
    /// model view or imodv, and open a 3dmod window.
    pub fn new_base_manager_string_int_axis_id(
        manager: &'static dyn BaseManager,
        file_name: &str,
        model_view_type: i32,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let process = ImodProcess::new_base_manager_string_axis_id(manager, file_name, axis_id);
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.set_model_view_type(model_view_type);
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, String, int, AxisID, ImodProcess.WindowOpenOption,
    /// FileType)`.  Use this constructor to create an instance of ImodProcess using
    /// ImodProcess(String dataset) and set either model view or imodv, and open a 3dmod
    /// window.
    pub fn new_base_manager_string_int_axis_id_window_open_option_file_type(
        manager: &'static dyn BaseManager,
        dataset_name: &str,
        model_view_type: i32,
        axis_id: Option<AxisID>,
        option: WindowOpenOption,
        file_type: &FileType,
    ) -> ImodState {
        let process = ImodProcess::new_base_manager_string_axis_id_file_type(
            manager,
            dataset_name,
            axis_id,
            file_type,
        );
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.set_model_view_type(model_view_type);
        state.process.add_window_open_option(option);
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, String, int, AxisID, ImodProcess.WindowOpenOption,
    /// File)`.  Use this constructor to create an instance of ImodProcess using
    /// ImodProcess(String dataset) and set either model view or imodv, and open a 3dmod
    /// window.
    pub fn new_base_manager_string_int_axis_id_window_open_option_file(
        manager: &'static dyn BaseManager,
        dataset_name: &str,
        model_view_type: i32,
        axis_id: Option<AxisID>,
        option: WindowOpenOption,
        file: Option<&Path>,
    ) -> ImodState {
        let process =
            ImodProcess::new_base_manager_string_axis_id_file(manager, dataset_name, axis_id, file);
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.set_model_view_type(model_view_type);
        state.process.add_window_open_option(option);
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, String, String, AxisID)`.  Use this constructor to
    /// create an instance of ImodProcess using ImodProcess(String dataset, String model).
    pub fn new_base_manager_string_string_axis_id(
        manager: &'static dyn BaseManager,
        dataset_name: &str,
        model_name: &str,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let mut fields = ImodStateFields::initial();
        fields.initial_model_name = model_name.to_string();
        let emergency_monitor = manager.get_emergency_monitor(axis_id);
        // `new File(manager.getPropertyUserDir(), datasetName)`.
        let file = match manager.get_property_user_dir() {
            None => PathBuf::from(dataset_name),
            Some(dir) => PathBuf::from(utilities::java_io_file_new(&dir, dataset_name)),
        };

        let process = ImodProcess::new_base_manager_string_string_file_emergency_monitor(
            manager,
            dataset_name,
            model_name,
            Some(&file),
            Some(emergency_monitor),
        );
        let state = ImodState::construct(manager, axis_id, process, None, None, fields);
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, FileType, AxisID)`.
    ///
    /// `fileType.getFile(...)` can be null, where the source's `getAbsolutePath()`
    /// throws.  Fixed in translation: the dataset name is then empty (no file).
    pub fn new_base_manager_file_type_axis_id(
        manager: &'static dyn BaseManager,
        file_type: &FileType,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let dataset_name = file_type
            .get_file(Some(manager), axis_id)
            .map(|file| utilities::java_io_file_get_absolute_path(&file.to_string_lossy()))
            .unwrap_or_default();
        let process = ImodProcess::new_base_manager_string_axis_id_file_type(
            manager,
            &dataset_name,
            axis_id,
            file_type,
        );
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, File, AxisID)`.
    pub fn new_base_manager_file_axis_id(
        manager: &'static dyn BaseManager,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let process = match file {
            Some(file) => ImodProcess::new_base_manager_string_axis_id_file(
                manager,
                &utilities::java_io_file_get_absolute_path(&file.to_string_lossy()),
                axis_id,
                Some(file),
            ),
            None => ImodProcess::new_base_manager_axis_id(manager, axis_id),
        };
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.reset();
        state
    }

    /// Java `setFile`.
    pub fn set_file(&self, file: &Path) {
        self.process
            .set_dataset_name(&utilities::java_io_file_get_absolute_path(
                &file.to_string_lossy(),
            ));
    }

    /// Java `ImodState(BaseManager, String[], AxisID)`.
    pub fn new_base_manager_string_array_axis_id(
        manager: &'static dyn BaseManager,
        file_name_array: Option<Vec<String>>,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let process = ImodProcess::new_base_manager_string_array(manager, file_name_array.clone());
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            file_name_array,
            None,
            ImodStateFields::initial(),
        );
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, String[], AxisID, String)`.
    pub fn new_base_manager_string_array_axis_id_string(
        manager: &'static dyn BaseManager,
        file_name_array: Option<Vec<String>>,
        axis_id: Option<AxisID>,
        subdir_name: Option<&str>,
    ) -> ImodState {
        let process = ImodProcess::new_base_manager_string_array(manager, file_name_array.clone());
        process.set_subdir_name(subdir_name);
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            file_name_array,
            None,
            ImodStateFields::initial(),
        );
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, File[], AxisID)`.
    pub fn new_base_manager_file_array_axis_id(
        manager: &'static dyn BaseManager,
        file_list: Option<Vec<PathBuf>>,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let emergency_monitor = manager.get_emergency_monitor(axis_id);
        let process = ImodProcess::new_base_manager_file_array_emergency_monitor(
            manager,
            file_list.clone(),
            Some(emergency_monitor),
        );
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            file_list,
            ImodStateFields::initial(),
        );
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, AxisID, String, FileType)`.  Build ImodState with a
    /// complete file name.
    pub fn new_base_manager_axis_id_string_file_type(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        file_name: &str,
        file_type: &FileType,
    ) -> ImodState {
        let process = ImodProcess::new_base_manager_string_axis_id_file_type(
            manager, file_name, axis_id, file_type,
        );
        let state = ImodState::construct(
            manager,
            axis_id,
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, AxisID, FileType)`.
    ///
    /// The source dereferences `axisID` (`getExtension`); every caller passes one.
    /// `fileType.getFileName(...)` can be null, which `ImodProcess.open` would then
    /// dereference; fixed in translation as an empty dataset name.
    pub fn new_base_manager_axis_id_file_type(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        file_type: &FileType,
    ) -> ImodState {
        let axis_extension = axis_id.get_extension();
        if axis_extension == "ERROR" {
            // Unreachable: `AxisID` has exactly the three values that have extensions.
            unreachable!("{}", axis_id);
        }
        let file_name = file_type
            .get_file_name(Some(manager), Some(axis_id))
            .unwrap_or_default();
        let process = ImodProcess::new_base_manager_string_axis_id_file_type(
            manager,
            &file_name,
            Some(axis_id),
            file_type,
        );
        let state = ImodState::construct(
            manager,
            Some(axis_id),
            process,
            None,
            None,
            ImodStateFields::initial(),
        );
        state.reset();
        state
    }

    /// Java `ImodState(BaseManager, AxisID, FileType, FileType, FileType, String,
    /// String)`.  Use this constructor to insert the axis letter and create an instance
    /// of ImodProcess using ImodProcess(String dataset, String model).  This will work
    /// for any kind of AxisID.
    ///
    /// Example:
    /// ImodAssistant(axisID, "top", "mid", "bot" ,".rec", "tomopitch" ,".mod");
    /// will cause one of these calls:
    /// ImodProcess("top.rec mid.rec bot.rec", "tomopitch.mod");
    /// ImodProcess("topa.rec mida.rec bota.rec", "tomopitcha.mod");
    /// ImodProcess("topb.rec midb.rec botb.rec", "tomopitchb.mod");
    pub fn new_base_manager_axis_id_file_type_file_type_file_type_string_string(
        manager: &'static dyn BaseManager,
        temp_axis_id: AxisID,
        file_type1: &FileType,
        file_type2: &FileType,
        file_type3: &FileType,
        model_name: &str,
        model_ext: &str,
    ) -> ImodState {
        let axis_id = Some(temp_axis_id);
        let axis_extension = temp_axis_id.get_extension();
        if axis_extension == "ERROR" {
            // Unreachable: `AxisID` has exactly the three values that have extensions.
            unreachable!("{}", temp_axis_id);
        }
        // A null file name would print as "null" in the source's concatenation and be
        // passed on as a null array element; it is empty here.
        let dataset_name_array = vec![
            file_type1
                .get_file_name(Some(manager), axis_id)
                .unwrap_or_default(),
            file_type2
                .get_file_name(Some(manager), axis_id)
                .unwrap_or_default(),
            file_type3
                .get_file_name(Some(manager), axis_id)
                .unwrap_or_default(),
        ];
        let _dataset_name = dataset_name_array[0].clone()
            + " "
            + &dataset_name_array[1]
            + " "
            + &dataset_name_array[2];
        let mut fields = ImodStateFields::initial();
        fields.initial_model_name = model_name.to_string() + &axis_extension + model_ext;
        let emergency_monitor = manager.get_emergency_monitor(Some(temp_axis_id));
        let file = file_type1.get_file(Some(manager), Some(temp_axis_id));
        let process = ImodProcess::new_base_manager_string_array_string_file_emergency_monitor(
            manager,
            Some(dataset_name_array),
            model_name,
            file.as_deref(),
            Some(emergency_monitor),
        );
        let state = ImodState::construct(manager, axis_id, process, None, None, fields);
        state.reset();
        state
    }

    /// Java `processRequest`.
    pub fn process_request(&self) {
        self.process.process_request();
    }

    /// Java `open(Run3dmodMenuOptions)`.  Opens a process, opens a model.
    pub fn open_run3dmod_menu_options(
        &self,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let mut menu_options = match menu_options {
            None => Run3dmodMenuOptions::new(),
            Some(menu_options) => menu_options,
        };
        let (no_menu_options, allow_menu_binning_in_z) = {
            let fields = self.fields.lock().unwrap();
            (fields.no_menu_options, fields.allow_menu_binning_in_z)
        };
        menu_options.set_no_options(no_menu_options);
        menu_options.or_global_options();
        menu_options.set_allow_binning_in_z(allow_menu_binning_in_z);
        // process is not running
        if !self.process.is_running() {
            // open
            self.process.open(menu_options)?;
            let mut fields = self.fields.lock().unwrap();
            fields.warned_stale_file = false;
            // model will be opened
            // `modelName.matches("\\S+")`: non-empty and no ASCII white space.
            if let Some(model_name) = &fields.model_name
                && !model_name.is_empty()
                && !model_name
                    .chars()
                    .any(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
                && fields.preserve_contrast
            {
                self.process
                    .set_open_model_preserve_contrast_message(model_name);
            }
            // This message can only be sent after opening the model
            if fields.open_contours {
                self.process.set_new_contours_message(true);
            }
            if fields.start_new_contours_at_new_z {
                self.process.set_start_new_contours_at_new_z();
            }
            if let Some(interpolation) = &fields.interpolation {
                self.process.set_interpolation(interpolation.is());
                fields.interpolation = None;
            }
            if fields.point_limit != -1 {
                self.process.set_point_limit_message(fields.point_limit);
            }
            // open bead fixer
            if fields.open_bead_fixer {
                self.process.set_open_bead_fixer_message();
                if self.manager.is_beadfixer_diameter_available() {
                    self.process
                        .set_beadfixer_diameter(self.manager.get_beadfixer_diameter(self.axis_id));
                }
                if fields.set_auto_center {
                    self.process.set_auto_center(fields.auto_center);
                }
                if fields.manage_new_contours {
                    self.process.set_new_contours(fields.new_contours);
                }
                if let Some(beadfixer_mode) = fields.beadfixer_mode {
                    self.process.set_beadfixer_mode(beadfixer_mode);
                }
                if let Some(log_name) = &fields.log_name {
                    self.process.set_open_log(log_name);
                }
                if let Some(skip_list) = &fields.skip_list {
                    self.process.set_skip_list(Some(skip_list));
                }
                if let Some(delete_all_sections) = fields.delete_all_sections.as_mut() {
                    self.process
                        .set_delete_all_sections(delete_all_sections.is());
                    delete_all_sections.set_boolean(false);
                }
            }
        } else {
            let mut fields = self.fields.lock().unwrap();
            // process is running
            // raise 3dmod
            if !fields.model_view && !fields.use_modv {
                self.process.set_open_zap_window_message();
            }
            if let Some(interpolation) = &fields.interpolation {
                self.process.set_interpolation(interpolation.is());
                fields.interpolation = None;
            } else {
                self.process.set_raise_3dmod_message();
            }
            // reopen model
            if !fields.use_modv && !utilities::is_empty(fields.model_name.as_deref()) {
                let model_name = fields.model_name.clone().unwrap_or_default();
                if fields.preserve_contrast {
                    self.process
                        .set_open_model_preserve_contrast_message(&model_name);
                } else {
                    self.process.set_open_model_message(&model_name);
                }
            }
            // This message can only be sent after opening the model
            if fields.open_contours {
                self.process.set_new_contours_message(true);
            }
            // open bead fixer
            if fields.open_bead_fixer {
                self.process.set_open_bead_fixer_message();
                if self.manager.is_beadfixer_diameter_available() {
                    self.process
                        .set_beadfixer_diameter(self.manager.get_beadfixer_diameter(self.axis_id));
                }
                if fields.set_auto_center {
                    self.process.set_auto_center(fields.auto_center);
                }
                if fields.manage_new_contours {
                    self.process.set_new_contours(fields.new_contours);
                }
                if let Some(beadfixer_mode) = fields.beadfixer_mode {
                    self.process.set_beadfixer_mode(beadfixer_mode);
                }
                if let Some(log_name) = &fields.log_name {
                    self.process.set_open_log(log_name);
                }
                if let Some(delete_all_sections) = fields.delete_all_sections.as_mut() {
                    self.process
                        .set_delete_all_sections(delete_all_sections.is());
                    delete_all_sections.set_boolean(false);
                }
                self.process.set_skip_list(fields.skip_list.as_deref());
            }
        }
        {
            let mut fields = self.fields.lock().unwrap();
            if fields.open_surf_cont_point {
                self.process.open_surf_cont_point();
                fields.open_surf_cont_point = false;
            }
            // set mode
            if fields.using_mode {
                if fields.mode == MODEL_MODE {
                    self.process.set_model_mode_message();
                } else {
                    self.process.set_movie_mode_message();
                }
            }
            if (fields.model_view || fields.use_modv)
                && fields.interpolation.is_none()
                && fields.using_mode
                && fields.mode != MODEL_MODE
                && etomo_director::ARGUMENTS
                    .lock()
                    .unwrap()
                    .get_debug_level()
                    .is_extra_verbose()
            {
                eprintln!("ImodState:open: sendMessages");
                // `Thread.dumpStack()`.
                eprintln!("java.lang.Exception: Stack trace");
            }
        }
        self.process.send_messages()?;
        self.reset();
        Ok(())
    }

    /// Java `setOpenSurfContPoint`.
    pub fn set_open_surf_cont_point(&self, input: bool) {
        self.fields.lock().unwrap().open_surf_cont_point = input;
    }

    /// Java `open(FileType, Run3dmodMenuOptions)`.
    pub fn open_file_type_run3dmod_menu_options(
        &self,
        model_name: &FileType,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        self.set_model(model_name);
        self.open_run3dmod_menu_options(menu_options)
    }

    /// Java `open(String, Run3dmodMenuOptions)`.  Opens a process using the modelName
    /// parameter.  Ignores mode setting.
    ///
    /// Configuration functions you can use with this function:
    /// configureUseModv()
    pub fn open_string_run3dmod_menu_options(
        &self,
        model_name: Option<&str>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        self.set_model_name(model_name);
        self.open_run3dmod_menu_options(menu_options)
    }

    /// Java `open(List<String>, Run3dmodMenuOptions)`.
    pub fn open_list_run3dmod_menu_options(
        &self,
        model_name_list: Option<Vec<String>>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        self.set_model_name_list(model_name_list);
        self.open_run3dmod_menu_options(menu_options)
    }

    /// Java `open(String, boolean, Run3dmodMenuOptions)`.
    pub fn open_string_boolean_run3dmod_menu_options(
        &self,
        model_name: Option<&str>,
        model_mode: bool,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        self.set_model_name(model_name);
        self.set_model_mode(model_mode);
        self.open_run3dmod_menu_options(menu_options)
    }

    /// Java `getRubberbandCoordinates`.
    pub fn get_rubberband_coordinates(&self) -> Result<Option<Vec<String>>, ImodManagerException> {
        self.process.get_rubberband_coordinates()
    }

    /// Java `getSlicerAngles`.
    pub fn get_slicer_angles(&self) -> Result<Option<Vec<String>>, ImodManagerException> {
        self.process.get_slicer_angles()
    }

    /// Java `quit`.  Tells process to quit.
    pub fn quit(&self) -> Result<(), ImodManagerException> {
        self.process.quit()
    }

    /// Java `disconnect`.
    pub fn disconnect(&self) -> Result<(), ImodManagerException> {
        self.process.disconnect()
    }

    /// Java `setModelViewType`.  `modelViewType` is either MODEL_VIEW or MODV.
    pub fn set_model_view_type(&self, model_view_type: i32) {
        let mut fields = self.fields.lock().unwrap();
        if model_view_type == MODEL_VIEW {
            fields.model_view = true;
        } else if model_view_type == MODV {
            fields.use_modv = true;
        } else {
            fields.model_view = false;
            fields.use_modv = false;
        }
        self.process.set_model_view(fields.model_view);
        self.process.set_use_modv(fields.use_modv);
    }

    /// Java `setOpenZap`.  Zap opens by default.  OpenZap is only necessary when model
    /// view is used.
    pub fn set_open_zap(&self) {
        self.process.set_open_zap();
    }

    /// Java `setTiltFile`.
    pub fn set_tilt_file(&self, tilt_file: Option<&str>) {
        self.process.set_tilt_file(tilt_file);
    }

    /// Java `resetTiltFile`.
    pub fn reset_tilt_file(&self) {
        self.process.reset_tilt_file();
    }

    /// Java `addWindowOpenOption`.
    pub fn add_window_open_option(&self, option: WindowOpenOption) {
        self.process.add_window_open_option(option);
    }

    /// Java `reset`.
    pub fn reset(&self) {
        // reset to initial state
        let initial_model_name = self.fields.lock().unwrap().initial_model_name.clone();
        self.set_model_name(Some(&initial_model_name));
        let mut fields = self.fields.lock().unwrap();
        fields.mode = fields.initial_mode;
        fields.swap_yz = fields.initial_swap_yz;
        // reset to default state
        fields.preserve_contrast = DEFAULT_PRESERVE_CONTRAST;
        self.process.set_open_with_model(!fields.preserve_contrast);
        fields.open_bead_fixer = DEFAULT_OPEN_BEAD_FIXER;
        fields.open_contours = DEFAULT_OPEN_CONTOURS;
        self.process.set_binning(DEFAULT_BINNING);
        self.process.set_frames(DEFAULT_FRAMES);
        self.process.set_piece_list_file_name(None);
        fields.manage_new_contours = false;
        fields.point_limit = -1;
        fields.start_new_contours_at_new_z = false;
    }

    /// Java `getModeString(int)`.
    pub fn get_mode_string_int(&self, mode: i32) -> String {
        if mode == MOVIE_MODE {
            "MOVIE_MODE".to_string()
        } else if mode == MODEL_MODE {
            "MODEL_MODE".to_string()
        } else {
            "ERROR:".to_string() + &mode.to_string()
        }
    }

    /// Java `setDebug`.
    pub fn set_debug(&self, input: bool) {
        let mut fields = self.fields.lock().unwrap();
        fields.debug = input;
        self.process.set_debug(fields.debug);
    }

    /// Java `equalsSubdirName`.
    ///
    /// The source calls `process.getSubdirName().equals(input)`, which throws when this
    /// state has no subdirectory (a state first built by `open(String, String[],
    /// Run3dmodMenuOptions)`).  Fixed in translation: two null names are equal and a
    /// null name equals no non-null one.
    pub fn equals_subdir_name(&self, input: Option<&str>) -> bool {
        self.process.get_subdir_name().as_deref() == input
    }

    /// Java `equalsFileNameArray`.
    pub fn equals_file_name_array(&self, input: Option<&[String]>) -> bool {
        let file_name_array = self.file_name_array.as_deref();
        if file_name_array.is_none() && input.is_none() {
            return true;
        }
        let (Some(file_name_array), Some(input)) = (file_name_array, input) else {
            return false;
        };
        if file_name_array.len() != input.len() {
            return false;
        }
        for i in 0..file_name_array.len() {
            if file_name_array[i] != input[i] {
                return false;
            }
        }
        true
    }

    // unchanging state information

    /// Java `isModelView`.
    pub fn is_model_view(&self) -> bool {
        self.fields.lock().unwrap().model_view
    }

    /// Java `isUseModv`.
    pub fn is_use_modv(&self) -> bool {
        self.fields.lock().unwrap().use_modv
    }

    // current state information
    /// Java `getModelName`.
    pub fn get_model_name(&self) -> Option<String> {
        self.fields.lock().unwrap().model_name.clone()
    }

    /// Java private `setModel(FileType)`.
    ///
    /// A null `getFile` result, where the source's `getAbsolutePath()` throws, sets no
    /// model (fixed in translation).
    fn set_model(&self, model: &FileType) {
        let model_name = model
            .get_file(Some(self.manager), self.axis_id)
            .map(|file| utilities::java_io_file_get_absolute_path(&file.to_string_lossy()));
        self.set_model_name(model_name.as_deref());
    }

    /// Java private `setModelName`.
    fn set_model_name(&self, model_name: Option<&str>) {
        self.fields.lock().unwrap().model_name = model_name.map(str::to_string);
        self.process.set_model_name(model_name);
    }

    /// Java private `setModelNameList`.
    fn set_model_name_list(&self, model_name_list: Option<Vec<String>>) {
        self.fields.lock().unwrap().model_name_list = model_name_list.clone();
        self.process.set_model_name_list(model_name_list);
    }

    /// Java `setLoadAsIntegers`.
    pub fn set_load_as_integers(&self) {
        self.process.set_load_as_integers();
    }

    /// Java `setSuppressSaveQuery`.
    pub fn set_suppress_save_query(&self) {
        self.process.set_suppress_save_query();
    }

    /// Java `isUsingMode`.
    pub fn is_using_mode(&self) -> bool {
        self.fields.lock().unwrap().using_mode
    }

    /// Java `setUsingMode`.
    pub fn set_using_mode(&self, using_mode: bool) {
        self.fields.lock().unwrap().using_mode = using_mode;
    }

    /// Java `isOpenContours`.
    pub fn is_open_contours(&self) -> bool {
        self.fields.lock().unwrap().open_contours
    }

    /// Java `setOpenContours`.
    pub fn set_open_contours(&self, open_contours: bool) {
        self.fields.lock().unwrap().open_contours = open_contours;
    }

    /// Java `setStartNewContoursAtNewZ`.
    pub fn set_start_new_contours_at_new_z(&self, start_new_contours_at_new_z: bool) {
        self.fields.lock().unwrap().start_new_contours_at_new_z = start_new_contours_at_new_z;
    }

    /// Java `setPointLimit`.
    pub fn set_point_limit(&self, input: i32) {
        self.fields.lock().unwrap().point_limit = input;
    }

    /// Java final `getAxisID`.
    pub fn get_axis_id(&self) -> Option<AxisID> {
        self.axis_id
    }

    /// Java `getMode`.
    pub fn get_mode(&self) -> i32 {
        self.fields.lock().unwrap().mode
    }

    /// Java `getModeString()`.
    pub fn get_mode_string(&self) -> String {
        let mode = self.fields.lock().unwrap().mode;
        self.get_mode_string_int(mode)
    }

    /// Java `setMode`.  set the mode to model or movie.
    pub fn set_mode(&self, mode: i32) {
        let mut fields = self.fields.lock().unwrap();
        fields.using_mode = true;
        fields.mode = mode;
    }

    /// Java private `setModelMode`.  Sets the mode (model or movie).
    fn set_model_mode(&self, model_mode: bool) {
        let mut fields = self.fields.lock().unwrap();
        fields.using_mode = true;
        if model_mode {
            fields.mode = MODEL_MODE;
        } else {
            fields.mode = MOVIE_MODE;
        }
    }

    /// Java `isSwapYZ`.
    pub fn is_swap_yz(&self) -> bool {
        self.fields.lock().unwrap().swap_yz
    }

    /// Java `setSwapYZ`.
    pub fn set_swap_yz(&self, swap_yz: bool) {
        self.fields.lock().unwrap().swap_yz = swap_yz;
        self.process.set_swap_yz(swap_yz);
    }

    /// Java `isPreserveContrast`.
    pub fn is_preserve_contrast(&self) -> bool {
        self.fields.lock().unwrap().preserve_contrast
    }

    /// Java `setPreserveContrast`.
    pub fn set_preserve_contrast(&self, preserve_contrast: bool) {
        self.fields.lock().unwrap().preserve_contrast = preserve_contrast;
        self.process.set_open_with_model(!preserve_contrast);
    }

    /// Java `setFrames`.
    pub fn set_frames(&self, frames: bool) {
        self.process.set_frames(frames);
    }

    /// Java `setPieceListFileName`.
    pub fn set_piece_list_file_name(&self, piece_list_file_name: Option<&str>) {
        self.process.set_piece_list_file_name(piece_list_file_name);
    }

    /// Java `setMontageSeparation`.
    pub fn set_montage_separation(&self) {
        self.process.set_montage_separation();
    }

    /// Java `setInterpolation`.
    pub fn set_interpolation(&self, input: bool) {
        let mut fields = self.fields.lock().unwrap();
        let interpolation = fields.interpolation.get_or_insert_with(EtomoBoolean2::new);
        interpolation.set_boolean(input);
    }

    /// Java `isOpenBeadFixer`.
    pub fn is_open_bead_fixer(&self) -> bool {
        self.fields.lock().unwrap().open_bead_fixer
    }

    /// Java `setOpenBeadFixer`.
    pub fn set_open_bead_fixer(&self, open_bead_fixer: bool) {
        let mut fields = self.fields.lock().unwrap();
        fields.open_bead_fixer = open_bead_fixer;
        fields.set_auto_center = false;
        fields.auto_center = false;
        fields.new_contours = false;
        fields.manage_new_contours = false;
    }

    /// Java `setAutoCenter`.
    pub fn set_auto_center(&self, auto_center: bool) {
        let mut fields = self.fields.lock().unwrap();
        fields.set_auto_center = true;
        fields.auto_center = auto_center;
    }

    /// Java `setSkipList`.
    pub fn set_skip_list(&self, input: Option<&str>) {
        self.fields.lock().unwrap().skip_list = input.map(str::to_string);
    }

    /// Java `setDeleteAllSections`.
    pub fn set_delete_all_sections(&self, on: bool) {
        let mut fields = self.fields.lock().unwrap();
        let delete_all_sections = fields
            .delete_all_sections
            .get_or_insert_with(EtomoBoolean2::new);
        delete_all_sections.set_boolean(on);
    }

    /// Java `setBeadfixerMode`.
    pub fn set_beadfixer_mode(&self, mode: Option<BeadFixerMode>) {
        self.fields.lock().unwrap().beadfixer_mode = mode;
    }

    /// Java `setOpenLogOff`.
    pub fn set_open_log_off(&self) {
        self.fields.lock().unwrap().log_name = None;
    }

    /// Java `setOpenLog`.
    pub fn set_open_log(&self, open_log: bool, log_name: Option<&str>) {
        let mut fields = self.fields.lock().unwrap();
        if open_log {
            fields.log_name = log_name.map(str::to_string);
        } else {
            fields.log_name = None;
        }
    }

    /// Java `setNewContours`.
    pub fn set_new_contours(&self, new_contours: bool) {
        let mut fields = self.fields.lock().unwrap();
        fields.new_contours = new_contours;
        fields.manage_new_contours = true;
    }

    // initial state information
    /// Java `getInitialModelName`.
    pub fn get_initial_model_name(&self) -> String {
        self.fields.lock().unwrap().initial_model_name.clone()
    }

    /// Java `getDatasetName`.
    pub fn get_dataset_name(&self) -> String {
        self.process.get_dataset_name()
    }

    /// Java `getInitialMode`.
    pub fn get_initial_mode(&self) -> i32 {
        self.fields.lock().unwrap().initial_mode
    }

    /// Java `getInitialModeString`.
    pub fn get_initial_mode_string(&self) -> String {
        let initial_mode = self.fields.lock().unwrap().initial_mode;
        self.get_mode_string_int(initial_mode)
    }

    /// Java `setInitialMode`.
    pub fn set_initial_mode(&self, initial_mode: i32) {
        if self.fields.lock().unwrap().initial_mode_set {
            return;
        }
        self.fields.lock().unwrap().initial_mode = initial_mode;
        self.set_mode(initial_mode);
        self.fields.lock().unwrap().initial_mode_set = true;
    }

    /// Java `isInitialSwapYZ`.
    pub fn is_initial_swap_yz(&self) -> bool {
        self.fields.lock().unwrap().initial_swap_yz
    }

    /// Java `setInitialSwapYZ`.
    pub fn set_initial_swap_yz(&self, initial_swap_yz: bool) {
        if self.fields.lock().unwrap().initial_swap_yz_set {
            return;
        }
        self.fields.lock().unwrap().initial_swap_yz = initial_swap_yz;
        self.set_swap_yz(initial_swap_yz);
        self.fields.lock().unwrap().initial_swap_yz_set = true;
    }

    // default state information

    /// Java `isDefaultOpenWithModel`.
    pub fn is_default_open_with_model(&self) -> bool {
        DEFAULT_OPEN_WITH_MODEL
    }

    /// Java `isDefaultPreserveContrast`.
    pub fn is_default_preserve_contrast(&self) -> bool {
        DEFAULT_PRESERVE_CONTRAST
    }

    // user controlled state information - pass through to ImodProcess
    /// Java `isOpen`.  Returns true if process is running.
    pub fn is_open(&self) -> bool {
        self.process.is_running()
    }

    /// Java final `setAllowMenuBinningInZ`.
    pub fn set_allow_menu_binning_in_z(&self, allow_menu_binning_in_z: bool) {
        self.fields.lock().unwrap().allow_menu_binning_in_z = allow_menu_binning_in_z;
    }

    /// Java final `setNoMenuOptions`.
    pub fn set_no_menu_options(&self, no_menu_options: bool) {
        self.fields.lock().unwrap().no_menu_options = no_menu_options;
    }

    /// Java `reopenLog`.
    pub fn reopen_log(&self) -> Result<(), ImodManagerException> {
        self.process.reopen_log()
    }

    /// Java `openModel`.
    pub fn open_model(&self, model: &str, model_mode: bool) -> Result<(), ImodManagerException> {
        self.process.open_model(model, model_mode)
    }

    /// Java `setBinning`.
    pub fn set_binning(&self, binning: i32) {
        self.process.set_binning(binning);
    }

    /// Java `setBinningXY`.
    pub fn set_binning_xy(&self, binning: i32) {
        self.process.set_binning_xy(binning);
    }

    /// Java `setWorkingDirectory`.
    pub fn set_working_directory(&self, working_directory: Option<PathBuf>) {
        self.process.set_working_directory(working_directory);
    }

    /// Java `setOpenModelView`.
    pub fn set_open_model_view(&self) -> Result<(), ImodManagerException> {
        self.process.set_open_model_view();
        Ok(())
    }

    /// Java `setContinuousListenerTarget`.
    pub fn set_continuous_listener_target(
        &self,
        continuous_listener_target: Option<Arc<dyn ContinuousListenerTarget>>,
    ) {
        self.process
            .set_continuous_listener_target(continuous_listener_target);
    }

    // internal state sets and gets
    /// Java `isWarnedStaleFile`.
    pub fn is_warned_stale_file(&self) -> bool {
        self.fields.lock().unwrap().warned_stale_file
    }

    /// Java `setWarnedStaleFile`.
    pub fn set_warned_stale_file(&self, warned_stale_file: bool) {
        self.fields.lock().unwrap().warned_stale_file = warned_stale_file;
    }

    /// Java `paramString`.  `Vector.toString()` of the parameter strings.
    pub fn param_string(&self) -> String {
        let params = [
            format!("modelView={}", self.is_model_view()),
            format!("useModv={}", self.is_use_modv()),
            format!(
                "modelName={}",
                self.get_model_name().unwrap_or_else(|| "null".to_string())
            ),
            format!("usingMode={}", self.is_using_mode()),
            format!("mode={}", self.get_mode_string()),
            format!("swapYZ={}", self.is_swap_yz()),
            format!("preserveContrast={}", self.is_preserve_contrast()),
            format!("openBeadFixer={}", self.is_open_bead_fixer()),
            format!("initialModelName={}", self.get_initial_model_name()),
            format!("initialMode={}", self.get_initial_mode_string()),
            format!("initialSwapYZ={}", self.is_initial_swap_yz()),
            format!("defaultOpenWithModel={}", self.is_default_open_with_model()),
            format!(
                "defaultPreserveContrast={}",
                self.is_default_preserve_contrast()
            ),
            format!("process={}", self.process),
            format!("warnedStaleFile={}", self.is_warned_stale_file()),
            format!("openContours={}", self.is_open_contours()),
        ];
        format!("[{}]", params.join(", "))
    }

    /// Java `equalsInitialConfiguration`.  Returns true if unchanging and initial state
    /// information is the same.
    pub fn equals_initial_configuration(&self, imod_state: &ImodState) -> bool {
        let (model_view, use_modv, initial_model_name, initial_mode, initial_swap_yz) = {
            let fields = self.fields.lock().unwrap();
            (
                fields.model_view,
                fields.use_modv,
                fields.initial_model_name.clone(),
                fields.initial_mode,
                fields.initial_swap_yz,
            )
        };
        model_view == imod_state.is_model_view()
            && use_modv == imod_state.is_use_modv()
            && initial_model_name == imod_state.get_initial_model_name()
            && initial_mode == imod_state.get_initial_mode()
            && initial_swap_yz == imod_state.is_initial_swap_yz()
    }

    /// Java `equalsCurrentConfiguration`.  Returns true if unchanging and current state
    /// information is the same.
    ///
    /// The source's `modelName.equals(...)` throws on a null model name; two null names
    /// compare equal here (fixed in translation).
    pub fn equals_current_configuration(&self, imod_state: &ImodState) -> bool {
        let (
            model_view,
            use_modv,
            model_name,
            using_mode,
            open_contours,
            mode,
            swap_yz,
            preserve_contrast,
            open_bead_fixer,
            start_new_contours_at_new_z,
        ) = {
            let fields = self.fields.lock().unwrap();
            (
                fields.model_view,
                fields.use_modv,
                fields.model_name.clone(),
                fields.using_mode,
                fields.open_contours,
                fields.mode,
                fields.swap_yz,
                fields.preserve_contrast,
                fields.open_bead_fixer,
                fields.start_new_contours_at_new_z,
            )
        };
        model_view == imod_state.is_model_view()
            && use_modv == imod_state.is_use_modv()
            && model_name == imod_state.get_model_name()
            && using_mode == imod_state.is_using_mode()
            && open_contours == imod_state.is_open_contours()
            && mode == imod_state.get_mode()
            && swap_yz == imod_state.is_swap_yz()
            && preserve_contrast == imod_state.is_preserve_contrast()
            && open_bead_fixer == imod_state.is_open_bead_fixer()
            && start_new_contours_at_new_z
                == imod_state
                    .fields
                    .lock()
                    .unwrap()
                    .start_new_contours_at_new_z
    }

    /// Java `equals(ImodState)`.  Returns true if unchanging, initial, and current state
    /// information is the same.
    pub fn equals(&self, imod_state: &ImodState) -> bool {
        self.equals_initial_configuration(imod_state)
            && self.equals_current_configuration(imod_state)
    }
}

/// Java `toString`.
impl std::fmt::Display for ImodState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[process={}]", self.process)
    }
}
