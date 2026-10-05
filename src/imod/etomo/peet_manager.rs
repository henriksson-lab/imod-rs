//! `IMOD/Etomo/src/etomo/PeetManager.java`.
//!
//! The manager of the PEET (particle averaging) interface (`.epe` data files).  It
//! owns a `PeetMetaData`, a `PeetScreenState`, a `PeetState`, a
//! `PeetProcessManager`, the `MainPeetPanel`, the `PeetStartupDialog` while a new
//! project is being started, the `PeetDialog`, and the `MatlabParam` that reads and
//! writes the project's `.prm` file.
//!
//! **Threads.**  The manager is a process-lifetime singleton shared with process
//! threads (`Send + Sync`); the main panel, the dialogs and the `MatlabParam` are
//! event dispatch thread objects, held in `EdtCell`s and reached only on that
//! thread.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::comscript::average_all_param::{self, AverageAllParam};
use crate::imod::etomo::comscript::command_details::CommandDetails;
use crate::imod::etomo::comscript::peet_parser_param::PeetParserParam;
use crate::imod::etomo::comscript::processchunks_param::{OutputImageFileKey, ProcesschunksParam};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{FileFilter, JComponent};
use crate::imod::etomo::logic::peet_startup_data::PeetStartupData;
use crate::imod::etomo::logic::version_control;
use crate::imod::etomo::process::base_imod_manager::ImodManagerException;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::peet_process_manager::PeetProcessManager;
use crate::imod::etomo::process::process_interface::ProcessResultDisplayRef;
use crate::imod::etomo::process_series::{Process, ProcessSeries, ProcessSeriesHandle};
use crate::imod::etomo::storage::averaged_file_names::AveragedFileNames;
use crate::imod::etomo::storage::com_file_filter::ComFileFilter;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::storage::matlab_param::MatlabParam;
use crate::imod::etomo::storage::matlab_param_file_filter::MatlabParamFileFilter;
use crate::imod::etomo::storage::parameter_store::ParameterStore;
use crate::imod::etomo::storage::peet_file_filter::PeetFileFilter;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::base_state::BaseState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_key::{self, FileKey};
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::peet_meta_data::PeetMetaData;
use crate::imod::etomo::r#type::peet_screen_state::PeetScreenState;
use crate::imod::etomo::r#type::peet_state::PeetState;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::UiComponent;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::swing::abstract_parallel_dialog::AbstractParallelDialog;
use crate::imod::etomo::ui::swing::main_panel::MainPanelVirtual;
use crate::imod::etomo::ui::swing::main_peet_panel::MainPeetPanel;
use crate::imod::etomo::ui::swing::peet_dialog::{self, PeetDialog};
use crate::imod::etomo::ui::swing::peet_startup_dialog::PeetStartupDialog;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::environment_variable;
use crate::imod::etomo::util::event_queue::{EdtCell, EdtRef};
use crate::imod::etomo::util::utilities;

/// Java private static final `AXIS_ID`.
const AXIS_ID: AxisID = AxisID::Only;

/// Java private static final class `LoadState`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoadState {
    Success,
    Reload,
    Exit,
}

/// Java `public final class PeetManager extends BaseManager`.
pub struct PeetManager {
    /// Java superclass `BaseManager` state.
    base: BaseManagerBase,
    /// Java private final `screenState = new PeetScreenState(AXIS_ID,
    /// AxisType.SINGLE_AXIS)`.
    screen_state: PeetScreenState,
    /// Java private final `metaData` (needs `this`).
    meta_data: OnceLock<PeetMetaData>,
    /// Java private final `processMgr`.
    process_mgr: OnceLock<&'static PeetProcessManager>,
    /// Java private final `state`.
    state: PeetState,
    /// Java private `peetDialog`, initially null.
    peet_dialog: EdtCell<Rc<PeetDialog>>,
    /// Java private `matlabParam`, initially null.
    matlab_param: EdtCell<Rc<RefCell<MatlabParam>>>,
    /// Java private `mainPanel` (null in headless mode).
    main_panel: EdtCell<Rc<MainPeetPanel>>,
    /// Java private `peetStartupDialog`, initially null.
    peet_startup_dialog: EdtCell<Rc<PeetStartupDialog>>,
    /// Java private `loadState`, initially null.
    load_state: Mutex<Option<LoadState>>,
    /// Java private `valid = true`: for handling failure before the manager key is
    /// set in EtomoDirector.
    valid: AtomicBool,
}

/// Owns every `PeetManager` this module builds (Java's owner is
/// `EtomoDirector.managerList`; the translation hands out `&'static Self`).
static INSTANCES: Mutex<Vec<&'static PeetManager>> = Mutex::new(Vec::new());

impl PeetManager {
    /// Java private `PeetManager(String)`.  (`PeetManager()` is `this("")`.)
    fn new(param_file_name: Option<&str>) -> &'static PeetManager {
        let instance: &'static PeetManager = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            screen_state: PeetScreenState::new(AXIS_ID, AxisType::SingleAxis),
            meta_data: OnceLock::new(),
            process_mgr: OnceLock::new(),
            state: PeetState::new(),
            peet_dialog: EdtCell::new(),
            matlab_param: EdtCell::new(),
            main_panel: EdtCell::new(),
            peet_startup_dialog: EdtCell::new(),
            load_state: Mutex::new(None),
            valid: AtomicBool::new(true),
        }));
        INSTANCES.lock().unwrap().push(instance);
        // super()
        instance.base_manager();
        let _ = instance.meta_data.set(PeetMetaData::new(
            Some(instance),
            instance.get_log_properties(),
            param_file_name.is_none_or(str::is_empty),
        ));
        instance.create_state();
        let _ = instance.process_mgr.set(PeetProcessManager::new(instance));
        instance.initialize_ui_parameters_from_name(param_file_name, Some(AXIS_ID));
        if instance.loaded_param_file() {
            instance.set_matlab_param(false);
        }
        instance
    }

    /// Java static package-private `getInstance()`.
    pub fn get_instance() -> &'static PeetManager {
        let instance = PeetManager::new(Some(""));
        instance.open_dialog();
        instance
    }

    /// Java static package-private `getInstance(String)`.
    pub fn get_instance_string(param_file_name: Option<&str>) -> &'static PeetManager {
        let instance = PeetManager::new(param_file_name);
        if instance.load_state() != Some(LoadState::Exit) {
            instance.open_dialog();
        }
        instance
    }

    /// The constructed manager at its final address.
    fn this_static(&self) -> &'static PeetManager {
        INSTANCES
            .lock()
            .unwrap()
            .iter()
            .copied()
            .find(|manager| std::ptr::eq(*manager, self))
            .expect("constructed PeetManager")
    }

    fn meta_data(&self) -> &'static PeetMetaData {
        self.this_static()
            .meta_data
            .get()
            .expect("metaData is assigned by the constructor")
    }

    fn process_mgr(&self) -> &'static PeetProcessManager {
        self.process_mgr.get().expect("processMgr")
    }

    fn load_state(&self) -> Option<LoadState> {
        *self.load_state.lock().unwrap()
    }

    fn set_load_state(&self, load_state: LoadState) {
        *self.load_state.lock().unwrap() = Some(load_state);
    }

    fn loaded_param_file(&self) -> bool {
        *self.base().loaded_param_file.lock().unwrap()
    }

    fn param_file(&self) -> Option<PathBuf> {
        self.base().param_file.lock().unwrap().clone()
    }

    /// Java `uiHarness.openMessageDialog(this, message, title)`.
    fn open_message(&'static self, message: &str, title: &str) {
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string(Some(self), message, title)
        });
    }

    /// Java `uiHarness.openMessageDialog(this, message, title, axisID)`.
    fn open_message_axis(&'static self, message: &str, title: &str, axis_id: AxisID) {
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string_axis_id(
                Some(self),
                message,
                title,
                Some(axis_id),
            )
        });
    }

    /// Java `mainPanel.setStatusBarText(paramFile, metaData, logWindow)`.  (Java
    /// dereferences mainPanel, which is null in headless mode; it is skipped then.)
    fn set_status_bar_text(&self) {
        if let Some(main_panel) = self.main_panel.get() {
            let param_file = self.param_file();
            let log_window = self.base().log_window.get();
            MainPanelVirtual::set_status_bar_text(
                &*main_panel,
                param_file.as_deref(),
                Some(self.meta_data() as &dyn BaseMetaData),
                log_window.as_ref(),
            );
        }
    }

    // <p>Updates done</p>

    /// Java private `openDialog()`.
    fn open_dialog(&'static self) {
        if !version_control::is_compatible_peet(AxisID::Only) {
            self.valid.store(false, Ordering::SeqCst);
            return;
        }
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            self.open_processing_panel();
            self.set_status_bar_text();
            if self.loaded_param_file() {
                self.open_peet_dialog(None);
            } else {
                self.open_peet_startup_dialog();
            }
        }
    }

    /// Java package-private `display()`.
    pub fn display(&self) {
        if let Some(peet_startup_dialog) = self.peet_startup_dialog.get() {
            peet_startup_dialog.display();
        }
    }

    /// The code `EtomoDirector.openPeet` runs after `display()` returns.  Java's
    /// `display()` blocks in the modal startup dialog; here the code runs when the
    /// dialog is hidden (at once when there is none).
    pub fn after_display(&self, job: Box<dyn FnOnce()>) {
        match self.peet_startup_dialog.get() {
            Some(peet_startup_dialog) => peet_startup_dialog.after_display(job),
            None => job(),
        }
    }

    /// The showing startup dialog (driver and Slint bridge).
    pub fn get_peet_startup_dialog(&self) -> Option<Rc<PeetStartupDialog>> {
        self.peet_startup_dialog.get()
    }

    /// The PEET dialog (Slint bridge).
    pub fn get_peet_dialog(&self) -> Option<Rc<PeetDialog>> {
        self.peet_dialog.get()
    }

    /// Java static `isInterfaceAvailable()`.
    pub fn is_interface_available() -> bool {
        if !environment_variable::INSTANCE.exists(
            None,
            etomo_director::INSTANCE.get_original_user_dir().as_deref(),
            environment_variable::PARTICLE_DIR,
            Some(AxisID::Only),
        ) {
            ui_harness::with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string(
                    None,
                    "PEET is an optional package for particle averaging, which has not been installed and correctly configured.  See the PEET link under Other Programs at http://bio3d.colorado.edu/.",
                    "Interface Unavailable",
                )
            });
            return false;
        }
        true
    }

    /// Java `getState()`.
    pub fn get_state(&self) -> &PeetState {
        &self.state
    }

    /// Java `copyDataset(BaseManager, PeetStartupData)`.  Copy the .prm file, and the
    /// .epe file if requested.  Modify them, and load the dataset into the manager and
    /// the dialog.
    pub fn copy_dataset(
        &'static self,
        manager: &'static dyn BaseManager,
        startup_data: Option<&PeetStartupData>,
    ) {
        let Some(startup_data) = startup_data else {
            return;
        };
        let file = PathBuf::from(
            startup_data
                .get_copy_from()
                .unwrap_or_else(|| "null".to_owned()),
        );
        // Create and correct the .epe file the new dataset directory.
        // Get the root name for this dataset
        let fn_output = startup_data.get_base_name();
        let dest_dir = PathBuf::from(
            startup_data
                .get_directory()
                .unwrap_or_else(|| "null".to_owned()),
        );
        // Create the new file
        let dest_peet_file = startup_data.get_param_file().unwrap_or_default();
        let mut peet_file_copied = false;
        if PeetFileFilter::new().accept(&file) {
            // Copy the .epe file
            match utilities::copy_file(
                Some(self),
                Some(AxisID::Only),
                Some(&file),
                Some(&dest_peet_file),
                false,
                false,
                false,
            ) {
                Ok(()) => peet_file_copied = true,
                Err(e) => eprintln!("{e:?}"),
            }
        }
        // If .epe file not copied, then create an empty .epe
        if !peet_file_copied {
            BaseProcessManager::touch(
                &utilities::java_io_file_get_absolute_path(&dest_peet_file.to_string_lossy()),
                Some(self),
            );
        }
        // Completely load the properties structure from the copied file
        let loaded = (|| -> Result<(), LogFileError> {
            let Some(mut dest_parameter_store) = ParameterStore::get_instance_manager(
                Some(manager),
                Some(AxisID::Only),
                Some(dest_peet_file.clone()),
            )?
            else {
                return Ok(());
            };
            let dest_meta_data = PeetMetaData::new(Some(self), self.get_log_properties(), false);
            dest_parameter_store.load(&dest_meta_data);
            let screen_state = PeetScreenState::new(AxisID::Only, AxisType::SingleAxis);
            dest_parameter_store.load(&screen_state);
            let state = PeetState::new();
            dest_parameter_store.load(&state);
            // Modify the properties to work with the new dataset
            dest_meta_data.set_root_name(fn_output.as_deref());
            // Wipe out process data in case there is a process running in the source
            // dataset.
            let process_data = self.process_mgr().base.get_process_data(AxisID::Only);
            process_data.lock().unwrap().reset();
            // Save the properties back to the copied file
            dest_parameter_store.set_auto_store(true);
            dest_parameter_store.save(Some(&*process_data))?;
            dest_parameter_store.save(Some(&dest_meta_data))?;
            Ok(())
        })();
        let dest_peet_file_name = dest_peet_file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        match loaded {
            Ok(()) => {}
            Err(LogFileError::Lock(_)) => return,
            Err(LogFileError::Io(e)) => {
                eprintln!("{e}");
                self.open_message(
                    &format!("Unable to load {dest_peet_file_name}.  {e}"),
                    "Unable to Continue",
                );
                return;
            }
            Err(e) => {
                eprintln!("{e:?}");
                self.open_message(
                    &format!(
                        "Unable to load {dest_peet_file_name}.  Close this file if it is open.  {e}"
                    ),
                    "Unable to Continue",
                );
                return;
            }
        }

        // Create and correct the .prm file in the new dataset directory
        // Create the new file
        let dest_matlab_file = dest_dir.join(format!(
            "{}{}",
            fn_output.as_deref().unwrap_or("null"),
            dataset_files::MATLAB_PARAM_FILE_EXT
        ));
        let source_matlab_file = if MatlabParamFileFilter::new().accept(&file) {
            file.clone()
        } else {
            let source_peet_file_absolute_path =
                utilities::java_io_file_get_absolute_path(&file.to_string_lossy());
            let end = source_peet_file_absolute_path
                .rfind('.')
                .unwrap_or(source_peet_file_absolute_path.len());
            PathBuf::from(format!(
                "{}{}",
                &source_peet_file_absolute_path[..end],
                dataset_files::MATLAB_PARAM_FILE_EXT
            ))
        };
        // Attempt to copy the source .prm file to the directory of this dataset.
        if !source_matlab_file.exists() {
            self.open_message(
                &format!(
                    "Unable to copy .prm file. {} does not exist.",
                    utilities::java_io_file_get_absolute_path(
                        &source_matlab_file.to_string_lossy()
                    )
                ),
                "Entry Error",
            );
            return;
        }
        if utilities::copy_file(
            Some(self),
            Some(AxisID::Only),
            Some(&source_matlab_file),
            Some(&dest_matlab_file),
            false,
            false,
            false,
        )
        .is_err()
        {
            BaseProcessManager::touch(
                &utilities::java_io_file_get_absolute_path(&dest_matlab_file.to_string_lossy()),
                Some(self),
            );
        }
        // Load the .prm file
        let param = self.load_matlab_param(&dest_matlab_file, false);
        if self.load_state() == Some(LoadState::Success) {
            let mut param = param.borrow_mut();
            param.set_fn_output(fn_output.as_deref());
            param.set_file(&utilities::java_io_file_get_absolute_path(
                &dest_dir.to_string_lossy(),
            ));
            param.write(Some(self));
        } else if self.load_state() == Some(LoadState::Exit) {
            return;
        }

        // load the dateset into the manager and dialog
        self.initialize_ui_parameters_from_name(
            Some(&utilities::java_io_file_get_absolute_path(
                &dest_peet_file.to_string_lossy(),
            )),
            Some(AXIS_ID),
        );
        if !self.loaded_param_file() {
            self.open_message(
                &format!(
                    "Unable to load {}.",
                    utilities::java_io_file_get_absolute_path(&dest_peet_file.to_string_lossy())
                ),
                "Entry Error",
            );
            return;
        }
        if let Some(peet_dialog) = self.peet_dialog.get() {
            peet_dialog.set_fn_output(fn_output.as_deref());
            peet_dialog.set_directory(startup_data.get_directory().as_deref());
            if self.meta_data().is_valid() {
                peet_dialog.set_parameters_meta_data(self.meta_data());
            }
        }
        self.matlab_param.set(Some(param.clone()));
        // Always load from matlabParam after loading the data file because some
        // values are in both places (maskModelPts) and the .prm values take
        // precedence over the .epe values.
        if let Some(peet_dialog) = self.peet_dialog.get() {
            let import_dir = if peet_file_copied {
                None
            } else {
                source_matlab_file.parent().map(Path::to_path_buf)
            };
            peet_dialog.set_parameters_matlab_param(&mut param.borrow_mut(), import_dir.as_deref());
            peet_dialog.set_iteration_rows(&mut param.borrow_mut());
            peet_dialog.update_mode(true);
        }
        self.set_status_bar_text();
        etomo_director::INSTANCE.rename_current_manager(
            self.meta_data()
                .get_name()
                .unwrap_or_else(|| "null".to_owned()),
        );
        if let Some(peet_dialog) = self.peet_dialog.get() {
            peet_dialog.convert_copied_paths(&utilities::java_io_file_get_absolute_path(
                &file
                    .parent()
                    .map(|parent| parent.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            ));
            peet_dialog.check_incorrect_paths();
        }
    }

    /// Java `setParamFile(PeetStartupData)`.  Tries to set paramFile.  Returns true if
    /// able to set paramFile.  If paramFile is already set, returns true.  Updates the
    /// peet dialog display if paramFile was set successfully.
    pub fn set_param_file_startup_data(
        &'static self,
        startup_data: Option<&PeetStartupData>,
    ) -> bool {
        if self.loaded_param_file() {
            return true;
        }
        let Some(startup_data) = startup_data else {
            return false;
        };
        let name = startup_data.get_base_name();
        if let Some(peet_dialog) = self.peet_dialog.get() {
            peet_dialog.set_fn_output(name.as_deref());
        }
        let dir_name = startup_data.get_directory();
        if let Some(peet_dialog) = self.peet_dialog.get() {
            peet_dialog.set_directory(dir_name.as_deref());
        }
        let param_file = startup_data.get_param_file().unwrap_or_default();
        if !param_file.exists() {
            self.process_mgr()
                .base
                .create_new_file(&utilities::java_io_file_get_absolute_path(
                    &param_file.to_string_lossy(),
                ));
        }
        self.initialize_ui_parameters(Some(&param_file), Some(AXIS_ID), false);
        if !self.loaded_param_file() {
            return false;
        }
        self.meta_data().set_name(name.as_deref());
        if !self.meta_data().is_valid() {
            self.open_message(
                "Invalid data, unable to proceed.  Please exit and restart Etomo",
                "Fatal Error",
            );
            return false;
        }
        self.get_imod_manager()
            .set_meta_data_const_peet_meta_data(self.meta_data());
        self.set_status_bar_text();
        etomo_director::INSTANCE.rename_current_manager(
            self.meta_data()
                .get_name()
                .unwrap_or_else(|| "null".to_owned()),
        );
        if !self.matlab_param.is_some() {
            self.set_matlab_param(true);
            if self.load_state() == Some(LoadState::Exit) {
                return false;
            }
        }
        if let Some(peet_dialog) = self.peet_dialog.get() {
            peet_dialog.update_mode(true);
        }
        true
    }

    /// Java `imodAvgVol(Run3dmodMenuOptions)`.  Open the *AvgVol*.mrc files in 3dmod.
    pub fn imod_avg_vol(&'static self, menu_options: Option<Run3dmodMenuOptions>) {
        let _averaged_file_names = AveragedFileNames::new();
        let avg_vol_list = AveragedFileNames::new().get_list(
            self,
            AxisID::Only,
            &format!(
                "Must press either {} or {}.",
                peet_dialog::RUN_LABEL,
                peet_dialog::AVERAGE_ALL_LABEL
            ),
            "Process Not Run",
        );
        // Fixed in translation (PeetManager.java:483): `getList` returns null on a
        // file error (after its own message) and Java's buildFileNameArray throws
        // NullPointerException; nothing is opened.  (BUGS.md)
        let Some(avg_vol_list) = avg_vol_list else {
            return;
        };
        Self::print_imod_exception(
            self.get_imod_manager()
                .open_string_string_array_run3dmod_menu_options(
                    imod_manager::AVG_VOL_KEY,
                    Some(&Self::build_file_name_array(&avg_vol_list)),
                    menu_options,
                ),
        );
    }

    /// Java `imodRef(Run3dmodMenuOptions)`.  Open the *Ref*.mrc files in 3dmod.
    pub fn imod_ref(&'static self, menu_options: Option<Run3dmodMenuOptions>) {
        // build the list of files - they should be in order
        let iteration_list_size = self.state.get_iteration_list_size();
        let name = format!(
            "{}_Ref",
            self.meta_data()
                .get_name()
                .unwrap_or_else(|| "null".to_owned())
        );
        let mut file_name_list = Vec::new();
        let mut i = 1;
        while i <= iteration_list_size + 1 {
            file_name_list.push(format!("{name}{i}.mrc"));
            i += 1;
        }
        Self::print_imod_exception(
            self.get_imod_manager()
                .open_string_string_array_run3dmod_menu_options(
                    imod_manager::REF_KEY,
                    Some(&Self::build_file_name_array(&file_name_list)),
                    menu_options,
                ),
        );
    }

    /// The `catch (IOException | SystemProcessException | AxisTypeException e) {
    /// e.printStackTrace(); }` of the `imod` methods.
    fn print_imod_exception(result: Result<(), ImodManagerException>) {
        match result {
            Ok(()) => {}
            Err(ImodManagerException::SystemProcess(e)) => eprintln!("{e}"),
            Err(ImodManagerException::AxisType(e)) => eprintln!("{e}"),
            Err(ImodManagerException::Io(e)) => eprintln!("{e}"),
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }

    /// Java private `buildFileNameArray(List)`: the list as an array (three
    /// equivalent cases by size).
    fn build_file_name_array(file_name_list: &[String]) -> Vec<String> {
        file_name_list.to_vec()
    }

    /// Java `peetParser(ProcessSeries, DialogType, ProcessingMethod)`.
    pub fn peet_parser(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: Option<DialogType>,
        peet_processing_method: Option<ProcessingMethod>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, AxisID::Only, dialog_type, Some("peetParser")),
        };
        if self.process_mgr().base.in_use(AxisID::Only, None, true) {
            process_series.borrow().end_series();
            return;
        }
        if !self.save_peet_dialog(true, true) {
            process_series.borrow().end_series();
            return;
        }
        let Some(matlab_param) = self.matlab_param.get() else {
            self.open_message(
                &format!(
                    "Must set {} and {}",
                    peet_dialog::DIRECTORY_LABEL,
                    peet_dialog::FN_OUTPUT_LABEL
                ),
                "Entry Error",
            );
            process_series.borrow().end_series();
            return;
        };
        let mut param = PeetParserParam::new(self, matlab_param.borrow().get_file());
        param.set_parameters(&matlab_param.borrow());
        match LogFile::get_instance_file(
            Some(&param.get_log_file()),
            Some(self.get_emergency_monitor(Some(AxisID::Only))),
        )
        .and_then(|log| log.backup())
        {
            Ok(_) => {}
            Err(LogFileError::Lock(_)) => {}
            Err(e) => eprintln!("{e:?}"),
        }
        self.remove_com_files();
        let thread_name = match self.process_mgr().peet_parser(
            Arc::new(param),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                self.open_message(
                    &format!(
                        "Unable to run {}, SystemProcessException.\n{e}",
                        ProcessName::PEET_PARSER
                    ),
                    "Process Error",
                );
                return;
            }
        };
        process_series
            .borrow_mut()
            .set_next_process_output_file_keys(
                Some(&ProcessName::PROCESSCHUNKS.to_string()),
                Some(ProcessName::PEET_PARSER),
                Some(file_key::AVERAGED_VOLUMES.clone()),
                Some(file_key::REFERENCE_VOLUMES.clone()),
                peet_processing_method,
            );
        self.set_thread_name(Some(&thread_name), Some(AxisID::Only));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::PEET_PARSER)),
                AxisID::Only,
                Some(&ProcessName::PEET_PARSER),
            );
        }
    }

    /// Java `averageAll(ProcessSeries, DialogType)`.
    pub fn average_all(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: Option<DialogType>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, AxisID::Only, dialog_type, Some("averageAll")),
        };
        if !self.save_peet_dialog(true, true) {
            process_series.borrow().end_series();
            return;
        }
        let Some(matlab_param) = self.matlab_param.get() else {
            self.open_message(
                &format!(
                    "Must set {} and {}",
                    peet_dialog::DIRECTORY_LABEL,
                    peet_dialog::FN_OUTPUT_LABEL
                ),
                "Entry Error",
            );
            process_series.borrow().end_series();
            return;
        };
        let mut param = AverageAllParam::new(self, matlab_param.borrow().get_file());
        if let Some(peet_dialog) = self.peet_dialog.get() {
            peet_dialog.get_parameters_average_all_param(&mut param);
        }
        param.set_parameters(&matlab_param.borrow());
        match LogFile::get_instance_file(
            Some(&average_all_param::AverageAllParam::get_log_file()),
            Some(self.get_emergency_monitor(Some(AxisID::Only))),
        )
        .and_then(|log| log.backup())
        {
            Ok(_) => {}
            // `catch (LogFile.LockException e) {}` on the outer try: the process is
            // not started.
            Err(LogFileError::Lock(_)) => return,
            Err(e) => eprintln!("{e:?}"),
        }
        let thread_name = match self.process_mgr().average_all(
            Arc::new(param),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                self.open_message(
                    &format!(
                        "Unable to run {}, SystemProcessException.\n{e}",
                        ProcessName::AVERAGE_ALL
                    ),
                    "Process Error",
                );
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(AxisID::Only));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::AVERAGE_ALL)),
                AxisID::Only,
                Some(&ProcessName::AVERAGE_ALL),
            );
        }
    }

    /// Java private `removeComFiles()`.
    fn remove_com_files(&self) {
        let dir = PathBuf::from(self.get_property_user_dir().unwrap_or_default());
        let filter = ComFileFilter::new(self.meta_data().get_name().as_deref());
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if filter.accept(&path) && path.is_file() {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    /// Java private `setMatlabParam(boolean)`.  Initialize matlabParamFile.
    /// Dependent on metaData.
    fn set_matlab_param(&'static self, new_file: bool) {
        if !self.loaded_param_file() || self.matlab_param.is_some() || self.param_file().is_none() {
            return;
        }
        let param = self.load_matlab_param(
            &dataset_files::get_matlab_param_file_from_manager(self),
            new_file,
        );
        self.matlab_param.set(Some(param));
    }

    /// Java `loadMatlabParam(File, boolean)`.
    pub fn load_matlab_param(
        &'static self,
        matlab_file: &Path,
        new_file: bool,
    ) -> Rc<RefCell<MatlabParam>> {
        let mut param;
        loop {
            // Load the .prm file
            param = Rc::new(RefCell::new(MatlabParam::new(
                Some(self),
                AXIS_ID,
                matlab_file,
                new_file,
            )));
            let mut error_list: Vec<String> = Vec::new();
            let peet_dialog = self.peet_dialog.get();
            let read = param.borrow_mut().read(
                Some(self),
                &mut error_list,
                peet_dialog
                    .as_deref()
                    .map(|peet_dialog| peet_dialog as &dyn UIComponent),
            );
            if read {
                self.set_load_state(LoadState::Success);
            } else if !error_list.is_empty() {
                // Handle error in the .prm file
                let mut message = vec![
                    format!(
                        "Unable to successfully parse {}.",
                        utilities::java_io_file_get_absolute_path(&matlab_file.to_string_lossy())
                    ),
                    String::new(),
                    "Fix the .prm file and press Yes to reload.  Or press No to exit from this dataset."
                        .to_owned(),
                    String::new(),
                    "Error(s):".to_owned(),
                ];
                message.append(&mut error_list);
                if ui_harness::with(|harness| {
                    harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                        Some(self),
                        &message,
                        Some(AxisID::Only),
                    )
                }) {
                    // The .prm file has been fixed - must reload.
                    self.set_load_state(LoadState::Reload);
                } else {
                    // Must exit from etomo without saving to the .prm file, if the .prm
                    // can't be fixed.
                    self.set_load_state(LoadState::Exit);
                    etomo_director::INSTANCE.close_current_manager(Some(AxisID::Only), false);
                }
            }
            if self.load_state() != Some(LoadState::Reload) {
                break;
            }
        }
        param
    }

    /// Java private `createState()`: empty.
    fn create_state(&self) {}

    /// Java private `openProcessingPanel()`.  MUST run reconnect for all axis.
    fn open_processing_panel(&'static self) {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_processing_panel(AxisType::SingleAxis);
        }
        self.set_panel();
        self.reconnect(
            Some(
                self.get_axis_process_data()
                    .get_saved_process_data(AxisID::Only),
            ),
            Some(AxisID::Only),
            true,
            None,
        );
    }

    /// Java private `openPeetDialog(PeetStartupData)`.  Create (if necessary) and show
    /// the peet dialog.  Update data if the param file has been set.
    fn open_peet_dialog(&'static self, startup_data: Option<&PeetStartupData>) {
        if !self.loaded_param_file() && startup_data.is_none() {
            self.open_message_axis(
                "Failed to load the parameter file, unable to continue.",
                "Failed",
                AxisID::Only,
            );
            self.valid.store(false, Ordering::SeqCst);
            return;
        }
        if !self.peet_dialog.is_some() {
            self.peet_dialog
                .set(Some(PeetDialog::get_instance(self, AXIS_ID)));
        }
        self.set_peet_dialog_parameters(None, true);
        if let Some(main_panel) = self.main_panel.get()
            && let Some(peet_dialog) = self.peet_dialog.get()
        {
            main_panel.show_process(&peet_dialog.get_component(), AXIS_ID);
        }
        let action_message =
            utilities::prepare_dialog_action_message(Some(DialogType::Peet), AxisID::Only, None);
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java private `openPeetStartupDialog()`.
    fn open_peet_startup_dialog(&'static self) {
        if !self.peet_startup_dialog.is_some() {
            let action_message = utilities::prepare_dialog_action_message(
                Some(DialogType::PeetStartup),
                AxisID::Only,
                None,
            );
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_static_progress_bar(Some("Starting PEET interface"), AXIS_ID);
            }
            self.peet_startup_dialog
                .set(Some(PeetStartupDialog::get_instance(self, AXIS_ID)));
            if let Some(action_message) = action_message {
                eprintln!("{action_message}");
            }
        }
    }

    /// Java `cancelStartup()`.
    pub fn cancel_startup(&'static self) {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id_process_end_state(
                AXIS_ID,
                Some(ProcessEndState::Killed),
            );
        }
        etomo_director::INSTANCE.close_current_manager(Some(AxisID::Only), false);
    }

    /// Java `setStartupData(PeetStartupData)`.
    pub fn set_startup_data(&'static self, startup_data: &PeetStartupData) {
        self.peet_startup_dialog.set(None);
        self.open_peet_dialog(Some(startup_data));
        if startup_data.is_copy_from() {
            self.copy_dataset(self, Some(startup_data));
            if self.load_state() == Some(LoadState::Exit) {
                return;
            }
        } else {
            self.set_param_file_startup_data(Some(startup_data));
            if self.load_state() == Some(LoadState::Exit) {
                return;
            }
        }
        if !self.loaded_param_file() {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    AXIS_ID,
                    Some(ProcessEndState::Failed),
                );
            }
            self.open_message_axis(
                "Failed to load or create parameter file, unable to continue.",
                "Failed",
                AxisID::Only,
            );
            self.valid.store(false, Ordering::SeqCst);
            return;
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel
                .stop_progress_bar_axis_id_process_end_state(AXIS_ID, Some(ProcessEndState::Done));
        }
    }

    /// Java private `setPeetDialogParameters(File, boolean)`.
    fn set_peet_dialog_parameters(
        &'static self,
        import_dir: Option<&Path>,
        meta_data_loaded: bool,
    ) {
        let Some(peet_dialog) = self.peet_dialog.get() else {
            return;
        };
        if self.param_file().is_some() && self.meta_data().is_valid() {
            self.set_browsing_dir_string(self.meta_data().get_browsing_directory().as_deref());
            if let Some(matlab_param) = self.matlab_param.get() {
                peet_dialog.set_iteration_rows(&mut matlab_param.borrow_mut());
            }
            if meta_data_loaded {
                peet_dialog.set_parameters_meta_data(self.meta_data());
            }
            if self.meta_data().is_low_cutoff_backwards_compatibility() {
                // Fixed in translation (PeetManager.java:785): Java passes a null
                // matlabParam on, which IterationTable dereferences; with no .prm
                // loaded there is nothing to check.  (BUGS.md)
                if let Some(matlab_param) = self.matlab_param.get() {
                    peet_dialog
                        .check_low_cutoff_backwards_compatibility(&mut matlab_param.borrow_mut());
                }
            }
            // Always load from matlabParam after loading the data file because some
            // values are in both places (maskModelPts) and the .prm values take
            // precedence over the .epe values.
            if let Some(matlab_param) = self.matlab_param.get() {
                peet_dialog.set_parameters_matlab_param(&mut matlab_param.borrow_mut(), import_dir);
            }
            peet_dialog.set_directory(self.get_property_user_dir().as_deref());
            peet_dialog.check_incorrect_paths();
            peet_dialog.update_mode(self.loaded_param_file());
        }
    }

    /// Java private `savePeetDialog(boolean, boolean)`.
    fn save_peet_dialog(&'static self, for_run: bool, do_validation: bool) -> bool {
        let Some(peet_dialog) = self.peet_dialog.get() else {
            eprintln!("java.lang.Exception: Stack trace\n\tat etomo.PeetManager.savePeetDialog");
            return false;
        };
        if self.param_file().is_none() && !self.set_param_file() {
            eprintln!("java.lang.Exception: Stack trace\n\tat etomo.PeetManager.savePeetDialog");
            return false;
        }
        let Some(matlab_param) = self.matlab_param.get() else {
            eprintln!("java.lang.Exception: Stack trace\n\tat etomo.PeetManager.savePeetDialog");
            return false;
        };
        self.meta_data()
            .set_browsing_directory(self.get_browsing_dir().as_deref());
        peet_dialog.get_parameters_meta_data(self.meta_data());
        self.save_storables(Some(AXIS_ID));
        if !peet_dialog.get_parameters_matlab_param(
            &mut matlab_param.borrow_mut(),
            for_run,
            do_validation,
        ) {
            return false;
        }
        if self.load_state() != Some(LoadState::Exit) {
            matlab_param.borrow_mut().write(Some(self));
        }
        true
    }

    /// Java private `processchunks(ProcessSeries, FileKey, ProcessingMethod,
    /// DialogType)`.  Run processchunks.
    fn processchunks_peet(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        output_image_file_key: Option<FileKey>,
        processing_method: Option<ProcessingMethod>,
        dialog_type: Option<DialogType>,
    ) {
        let Some(peet_dialog) = self.peet_dialog.get() else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        // try { ... } catch (FieldValidationFailedException e) { endSeries; return; }
        let fn_output = match peet_dialog.get_fn_output(true) {
            Ok(fn_output) => fn_output,
            Err(_) => {
                if let Some(process_series) = &process_series {
                    process_series.borrow().end_series();
                }
                return;
            }
        };
        let mut param = ProcesschunksParam::get_instance(
            self,
            AxisID::Only,
            fn_output.as_deref(),
            output_image_file_key.map(OutputImageFileKey::FileKey),
        );
        let parallel_panel = self
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(AxisID::Only));
        let Some(parallel_panel) = parallel_panel else {
            self.open_message_axis(
                &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                "Unable to execute command",
                AxisID::Only,
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        AbstractParallelDialog::get_parameters(&*peet_dialog, &mut param);
        if !parallel_panel.get_parameters_processchunks_param_boolean(&param, true) {
            if let Some(main_panel) = self.get_main_panel() {
                main_panel
                    .main_panel()
                    .stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
            }
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        // param should never be set to resume
        parallel_panel
            .get_parallel_progress_display()
            .reset_results();
        <Self as BaseManager>::processchunks(
            self,
            Some(AxisID::Only),
            Some(Arc::new(param)),
            None,
            process_series,
            false,
            processing_method,
            false,
            dialog_type,
            None,
            None,
            None,
        );
    }
}

impl BaseManager for PeetManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    /// Java `isStartupPopupOpen()`.
    fn is_startup_popup_open(&self) -> bool {
        !self.loaded_param_file()
    }

    /// Java final package-private `isTomosnapshotThumbnail()`.
    fn is_tomosnapshot_thumbnail(&self) -> bool {
        true
    }

    /// Java `isValid()`.
    fn is_valid(&self) -> bool {
        self.valid.load(Ordering::SeqCst) && self.load_state() != Some(LoadState::Exit)
    }

    /// Java package-private `initializeUIParameters(String, AxisID)`.
    fn initialize_ui_parameters_from_name(
        &'static self,
        param_file_name: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        // super.initializeUIParameters(paramFileName, axisID)
        match param_file_name {
            None => self.initialize_ui_parameters(None, axis_id, false),
            Some(param_file_name) if param_file_name.is_empty() => {
                self.initialize_ui_parameters(None, axis_id, false)
            }
            Some(param_file_name) => {
                self.initialize_ui_parameters(Some(Path::new(param_file_name)), axis_id, false)
            }
        }
        if self.loaded_param_file() {
            // `System.setProperty("user.dir", propertyUserDir)`; `PWD` is this
            // translation's `user.dir`.
            match self.get_property_user_dir() {
                Some(property_user_dir) => unsafe { std::env::set_var("PWD", property_user_dir) },
                // System.setProperty with a null value throws NullPointerException;
                // a loaded param file always has a directory.
                None => {}
            }
        }
    }

    /// Java `pack()`.
    fn pack(&self) {
        if let Some(peet_dialog) = self.peet_dialog.get() {
            peet_dialog.pack();
        }
    }

    /// Java `getInterfaceType()`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::Peet)
    }

    /// Java `getBaseMetaData()`.
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        self.meta_data
            .get()
            .map(|meta_data| meta_data as &dyn BaseMetaData)
    }

    /// Java `getBaseScreenState(AxisID)`.
    fn get_base_screen_state(&self, _axis_id: Option<AxisID>) -> Option<&'static BaseScreenState> {
        Some(&self.this_static().screen_state)
    }

    /// Java `getMainPanel()`.
    fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.main_panel
            .get()
            .map(|main_panel| main_panel as Rc<dyn MainPanelVirtual>)
    }

    /// Java `getFocusComponent()`.
    fn get_focus_component(&self) -> Option<Rc<JComponent>> {
        let peet_dialog = self.peet_dialog.get()?;
        Some(peet_dialog.get_focus_component())
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        self.meta_data
            .get()
            .and_then(|meta_data| BaseMetaData::get_name(meta_data))
    }

    /// Java `setParamFile(File)`.
    fn set_param_file_from(&self, param_file: Option<&Path>) -> bool {
        let this = self.this_static();
        // Upstream bug fixed in translation: a null file throws NullPointerException
        // in Java; nothing is set.
        let Some(param_file) = param_file else {
            return false;
        };
        if !param_file.exists() {
            this.process_mgr()
                .base
                .create_new_file(&utilities::java_io_file_get_absolute_path(
                    &param_file.to_string_lossy(),
                ));
        }
        this.initialize_ui_parameters(Some(param_file), Some(AXIS_ID), false);
        if this.loaded_param_file() {
            let root_name = dataset_files::get_root_name(param_file);
            this.meta_data().set_name(Some(&root_name));
            this.get_imod_manager()
                .set_meta_data_const_peet_meta_data(this.meta_data());
            this.set_matlab_param(false);
            if this.load_state() == Some(LoadState::Exit) {
                return false;
            }
            if let Some(peet_dialog) = this.peet_dialog.get() {
                peet_dialog.set_directory(
                    param_file
                        .parent()
                        .map(|parent| parent.to_string_lossy().into_owned())
                        .as_deref(),
                );
                peet_dialog.set_fn_output(Some(&root_name));
                peet_dialog.update_mode(true);
            }
        }
        true
    }

    /// Java `exitProgram(AxisID)`.  Call BaseManager.exitProgram().  Call
    /// savePeetDialog.  To guarantee that etomo can always exit, catch all
    /// unrecognized Exceptions and Errors and return true.
    fn exit_program(&'static self, axis_id: Option<AxisID>) -> bool {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if self.exit_program_super(axis_id) {
                self.end_threads();
                if self.load_state() != Some(LoadState::Exit) {
                    self.save_param_file()?;
                }
                return Ok(true);
            }
            Ok::<bool, LogFileError>(false)
        }));
        match result {
            Ok(Ok(exit)) => exit,
            Ok(Err(e)) => {
                eprintln!("{e:?}");
                true
            }
            Err(_) => true,
        }
    }

    /// Java `save() throws LogFileException, IOException, LockException`.
    fn save(&'static self) -> Result<bool, LogFileError> {
        self.save_super()?;
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.done();
        }
        self.save_peet_dialog(false, false);
        Ok(true)
    }

    /// Java `getParallelProcessingDefaultNice()`.
    fn get_parallel_processing_default_nice(&self) -> i32 {
        18
    }

    /// Java package-private `isPopupChunkWarnings()`.
    fn is_popup_chunk_warnings(&self) -> bool {
        false
    }

    /// Java `getBaseState()`.
    fn get_base_state(&self) -> Option<&'static dyn BaseState> {
        Some(&self.this_static().state as &'static dyn BaseState)
    }

    /// Java package-private `createMainPanel()`.
    fn create_main_panel(&self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            let this = self.this_static();
            self.main_panel.set(Some(MainPeetPanel::new(this)));
        }
    }

    /// Java `getProcessManager()`.
    fn get_process_manager(&self) -> Option<&'static BaseProcessManager> {
        self.process_mgr.get().map(|process_mgr| &process_mgr.base)
    }

    /// Java package-private `getStorables(int)`.
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Option<&'static dyn Storable>>> {
        let this = self.this_static();
        let mut storables: Vec<Option<&'static dyn Storable>> = vec![None; (3 + offset) as usize];
        let mut index = offset as usize;
        storables[index] = this
            .meta_data
            .get()
            .map(|meta_data| meta_data as &'static dyn Storable);
        index += 1;
        storables[index] = Some(&this.screen_state as &'static dyn Storable);
        index += 1;
        storables[index] = Some(&this.state as &'static dyn Storable);
        Some(storables)
    }

    /// Java package-private `startNextProcess(...)`.  Start the next process specified
    /// by the nextProcess string; returns true if the process is recognized.
    fn start_next_process(
        &'static self,
        ui_component: Option<Rc<dyn UiComponent>>,
        axis_id: AxisID,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: &ProcessSeriesHandle,
        dialog_type: Option<DialogType>,
        display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool {
        if self.start_next_process_super(
            ui_component,
            axis_id,
            process,
            process_result_display,
            process_series,
            dialog_type,
            display,
        ) {
            return true;
        }
        if process.equals_string(Some(&ProcessName::PROCESSCHUNKS.to_string())) {
            self.processchunks_peet(
                Some(Rc::clone(process_series)),
                process.get_output_image_file_key().cloned(),
                process.get_processing_method(),
                dialog_type,
            );
            return true;
        }
        false
    }

    /// Java `resume(AxisID, ProcesschunksParam, ProcessResultDisplay, ProcessSeries,
    /// CommandDetails, boolean, ProcessingMethod, boolean, DialogType)`.
    fn resume(
        &'static self,
        axis_id: Option<AxisID>,
        param: Option<Arc<ProcesschunksParam>>,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        subcommand_details: Option<Arc<dyn CommandDetails + Send + Sync>>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        _multi_line_messages: bool,
        dialog_type: Option<DialogType>,
    ) {
        if !self.save_peet_dialog(true, true) {
            // Fixed in translation (PeetManager.java:917): Java dereferences a null
            // processSeries here; with none there is no series to end.
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        self.resume_super(
            axis_id,
            param,
            process_result_display,
            process_series,
            subcommand_details,
            popup_chunk_warnings,
            processing_method,
            false,
            dialog_type,
        );
    }
}
