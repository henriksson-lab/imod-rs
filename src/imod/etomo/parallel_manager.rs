//! `IMOD/Etomo/src/etomo/ParallelManager.java`.
//!
//! The manager of the generic parallel process ("Generic Parallel Process") and the
//! nonlinear anisotropic diffusion interfaces (`.epp` data files).  It owns a
//! `ParallelMetaData`, a `BaseScreenState`, a `ParallelState`, a
//! `ParallelProcessManager`, the `MainParallelPanel` and one of the
//! `ParallelChooser`, `ParallelDialog` and `AnisotropicDiffusionDialog`.
//!
//! **Threads.**  The manager is a process-lifetime singleton shared with process
//! threads (`Send + Sync`); the main panel and the dialogs are event dispatch thread
//! objects, held in `EdtCell`s and reached only on that thread.  The members the
//! process manager calls back (`getState`, `setChunkSetupOutputFile`,
//! `setParallelProcessName`) are posted to that thread where they touch a dialog.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::comscript::anisotropic_diffusion_param::{
    self, AnisotropicDiffusionParam, Mode,
};
use crate::imod::etomo::comscript::chunksetup_param::{self, ChunksetupParam};
use crate::imod::etomo::comscript::processchunks_param::{OutputImageFileKey, ProcesschunksParam};
use crate::imod::etomo::comscript::set_env_param::SetEnvParam;
use crate::imod::etomo::comscript::trimvol_param::{self, TrimvolParam};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::base_imod_manager::ImodManagerException;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::parallel_process_manager::ParallelProcessManager;
use crate::imod::etomo::process::process_interface::ProcessResultDisplayRef;
use crate::imod::etomo::process::process_output_strings;
use crate::imod::etomo::process_series::{Process, ProcessSeries, ProcessSeriesHandle};
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::base_state::BaseState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::imod_output_format;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::parallel_meta_data::ParallelMetaData;
use crate::imod::etomo::r#type::parallel_state::ParallelState;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::UiComponent;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::swing::abstract_parallel_dialog::AbstractParallelDialog;
use crate::imod::etomo::ui::swing::anisotropic_diffusion_dialog::{
    self, AnisotropicDiffusionDialog,
};
use crate::imod::etomo::ui::swing::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::ui::swing::etomo_menu;
use crate::imod::etomo::ui::swing::main_panel::MainPanelVirtual;
use crate::imod::etomo::ui::swing::main_parallel_panel::MainParallelPanel;
use crate::imod::etomo::ui::swing::parallel_chooser::ParallelChooser;
use crate::imod::etomo::ui::swing::parallel_dialog::ParallelDialog;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue::{EdtCell, EdtRef};
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;

/// Java private static final `AXIS_ID`.
const AXIS_ID: AxisID = AxisID::Only;

/// Java `public final class ParallelManager extends BaseManager`.
pub struct ParallelManager {
    /// Java superclass `BaseManager` state.
    base: BaseManagerBase,
    /// Java private final `screenState = new BaseScreenState(AXIS_ID,
    /// AxisType.SINGLE_AXIS)`.
    screen_state: BaseScreenState,
    /// Java private final `state = new ParallelState(this, AXIS_ID)`; a field
    /// initializer that needs `this`, so it is set after `super()`.
    state: OnceLock<ParallelState>,
    /// Java private final `processMgr`.
    process_mgr: OnceLock<&'static ParallelProcessManager>,
    /// Java private final `metaData`.
    meta_data: OnceLock<ParallelMetaData>,
    /// Java private `parallelDialog`, initially null.
    parallel_dialog: EdtCell<Rc<ParallelDialog>>,
    /// Java private `anisotropicDiffusionDialog`, initially null.
    anisotropic_diffusion_dialog: EdtCell<Rc<AnisotropicDiffusionDialog>>,
    /// Java private `mainPanel` (null in headless mode).
    main_panel: EdtCell<Rc<MainParallelPanel>>,
}

/// Owns every `ParallelManager` this module builds.  Java's owner is the collector,
/// by way of `EtomoDirector.managerList`, which keeps each manager for the run; the
/// translation hands out `&'static Self`, so without a root here the allocation is
/// unreachable the moment the constructor returns.
static INSTANCES: std::sync::Mutex<Vec<&'static ParallelManager>> =
    std::sync::Mutex::new(Vec::new());

impl ParallelManager {
    /// Java `ParallelManager()`: `this("", null)`.
    pub fn new() -> &'static ParallelManager {
        Self::new_with_parameters(Some(""), None)
    }

    /// Java `ParallelManager(DialogType)`: `this("", dialogType)`.
    pub fn new_with_dialog_type(dialog_type: DialogType) -> &'static ParallelManager {
        Self::new_with_parameters(Some(""), Some(dialog_type))
    }

    /// Java `ParallelManager(String)`: `this(paramFileName, null)`.
    pub fn new_with_param_file(param_file_name: Option<&str>) -> &'static ParallelManager {
        Self::new_with_parameters(param_file_name, None)
    }

    /// Java `ParallelManager(String, DialogType)`.
    pub fn new_with_parameters(
        param_file_name: Option<&str>,
        dialog_type: Option<DialogType>,
    ) -> &'static ParallelManager {
        let instance: &'static ParallelManager = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            screen_state: BaseScreenState::new(AXIS_ID, AxisType::SingleAxis),
            state: OnceLock::new(),
            process_mgr: OnceLock::new(),
            meta_data: OnceLock::new(),
            parallel_dialog: EdtCell::new(),
            anisotropic_diffusion_dialog: EdtCell::new(),
            main_panel: EdtCell::new(),
        }));
        INSTANCES.lock().unwrap().push(instance);
        // super()
        instance.base_manager();
        let _ = instance
            .state
            .set(ParallelState::new(Some(instance), AXIS_ID));
        let _ = instance.meta_data.set(ParallelMetaData::new(
            Some(instance),
            instance.get_log_properties(),
            dialog_type == Some(DialogType::Parallel),
            param_file_name.is_none_or(str::is_empty),
        ));
        instance.create_state();
        let _ = instance
            .process_mgr
            .set(ParallelProcessManager::new(instance));
        instance.initialize_ui_parameters_from_name(param_file_name, Some(AXIS_ID));
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            instance.open_processing_panel();
            instance.set_status_bar_text();
            if instance.param_file().is_none() {
                if dialog_type == Some(DialogType::Parallel) {
                    instance.open_parallel_dialog();
                } else if dialog_type == Some(DialogType::AnisotropicDiffusion) {
                    instance.open_anisotropic_diffusion_dialog();
                } else {
                    instance.open_parallel_chooser();
                }
            } else if instance.meta_data().get_dialog_type() == Some(DialogType::Parallel) {
                instance.open_parallel_dialog();
            } else if instance.meta_data().get_dialog_type()
                == Some(DialogType::AnisotropicDiffusion)
            {
                instance.open_anisotropic_diffusion_dialog();
            }
        }
        instance
    }

    /// The constructed manager at its final address (Java `this` inside the
    /// overrides that take `&self`).
    fn this_static(&self) -> &'static ParallelManager {
        INSTANCES
            .lock()
            .unwrap()
            .iter()
            .copied()
            .find(|manager| std::ptr::eq(*manager, self))
            .expect("constructed ParallelManager")
    }

    /// Java field read `metaData`.
    fn meta_data(&self) -> &ParallelMetaData {
        self.meta_data
            .get()
            .expect("metaData is assigned by the constructor")
    }

    /// Java field read `processMgr`.
    fn process_mgr(&self) -> &'static ParallelProcessManager {
        self.process_mgr.get().expect("processMgr")
    }

    /// Java field read `paramFile`.
    fn param_file(&self) -> Option<PathBuf> {
        self.base().param_file.lock().unwrap().clone()
    }

    /// Java field read `propertyUserDir`.
    fn property_user_dir(&self) -> Option<String> {
        self.base().property_user_dir.lock().unwrap().clone()
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

    /// Java `uiHarness.openMessageDialog(this, message, title, axisID)`.
    fn open_message(&'static self, message: &str, title: &str, axis_id: Option<AxisID>) {
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string_axis_id(
                Some(self),
                message,
                title,
                axis_id,
            )
        });
    }

    /// Java `uiHarness.openMessageDialog(this, String[], title, axisID)`.
    fn open_message_array(&'static self, message: &[String], title: &str, axis_id: AxisID) {
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_array_string_axis_id(
                Some(self),
                message,
                title,
                Some(axis_id),
            )
        });
    }

    /// Java `getState()`.
    pub fn get_state(&self) -> &ParallelState {
        self.state
            .get()
            .expect("state is assigned by the constructor")
    }

    /// Java `canSnapshot()`.
    pub fn can_snapshot(&self) -> bool {
        false
    }

    /// Java private `createState()`: empty.
    fn create_state(&self) {}

    /// Java `getMetaData()`.
    pub fn get_meta_data(&self) -> &ParallelMetaData {
        self.meta_data()
    }

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
            false,
            None,
        );
    }

    /// Java private `openParallelChooser()`.
    fn open_parallel_chooser(&'static self) {
        let parallel_chooser = ParallelChooser::get_instance(self);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&parallel_chooser.get_container(), AXIS_ID);
        }
    }

    /// Java `openParallelDialog()`.
    pub fn open_parallel_dialog(&'static self) {
        if !self.parallel_dialog.is_some() {
            self.parallel_dialog
                .set(Some(ParallelDialog::get_instance(self, AXIS_ID)));
        }
        let parallel_dialog = self.parallel_dialog.get().unwrap();
        parallel_dialog.set_parameters_screen_state(&self.screen_state);
        if self.param_file().is_some() && self.meta_data().is_valid() {
            parallel_dialog.set_parameters_meta_data(self.meta_data());
            parallel_dialog.set_setup_mode(false);
        }
        // Java dereferences mainPanel, which is null in headless mode; nothing is
        // shown then.
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&parallel_dialog.get_container(), AXIS_ID);
        }
        let action_message = utilities::prepare_dialog_action_message(
            Some(DialogType::Parallel),
            AxisID::Only,
            None,
        );
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java `openAnisotropicDiffusionDialog()`.
    pub fn open_anisotropic_diffusion_dialog(&'static self) {
        self.meta_data()
            .set_dialog_type(Some(DialogType::AnisotropicDiffusion));
        if !self.anisotropic_diffusion_dialog.is_some() {
            self.anisotropic_diffusion_dialog
                .set(Some(AnisotropicDiffusionDialog::get_instance(
                    self, AXIS_ID,
                )));
        }
        let anisotropic_diffusion_dialog = self.anisotropic_diffusion_dialog.get().unwrap();
        if self.param_file().is_some() && self.meta_data().is_valid() {
            anisotropic_diffusion_dialog.set_parameters(self.meta_data());
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&anisotropic_diffusion_dialog.get_container(), AXIS_ID);
        }
        let action_message = utilities::prepare_dialog_action_message(
            Some(DialogType::AnisotropicDiffusion),
            AxisID::Only,
            None,
        );
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java private `saveParallelDialog()`.
    fn save_parallel_dialog(&'static self) {
        let Some(parallel_dialog) = self.parallel_dialog.get() else {
            return;
        };
        parallel_dialog.get_parameters_screen_state(&self.screen_state);
        self.save_storables(Some(AXIS_ID));
    }

    /// Java private `saveAnisotropicDiffusionDialog()`.
    fn save_anisotropic_diffusion_dialog(&'static self) {
        let Some(anisotropic_diffusion_dialog) = self.anisotropic_diffusion_dialog.get() else {
            return;
        };
        anisotropic_diffusion_dialog.get_parameters_meta_data(self.meta_data());
        anisotropic_diffusion_dialog.get_parameters_for_trimvol(self.meta_data());
        self.save_storables(Some(AXIS_ID));
    }

    /// Java `processchunks(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, String, FileType, ProcessingMethod, DialogType)`.  Set the
    /// paramFile and change the state, if necessary; run BaseManager.processchunks.
    #[allow(clippy::too_many_arguments)]
    pub fn processchunks(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        root_name: Option<&str>,
        output_image_file_type: Option<Arc<FileType>>,
        processing_method: Option<ProcessingMethod>,
        dialog_type: Option<DialogType>,
    ) {
        let display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        let Some(parallel_dialog) = self.parallel_dialog.get() else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        if self.param_file().is_none() {
            if !self.set_new_param_file_void() {
                if let Some(process_series) = &process_series {
                    process_series.borrow().end_series();
                }
                return;
            }
            parallel_dialog.set_setup_mode(false);
        }
        self.send_msg_process_starting(display_ref.as_ref());
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, AXIS_ID, dialog_type, Some("processchunks")),
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        // (The source checks `parallelDialog == null` again here; it cannot have
        // become null.)
        let mut param = ProcesschunksParam::get_instance(
            self,
            AxisID::Only,
            root_name,
            output_image_file_type.map(OutputImageFileKey::FileType),
        );
        let parallel_panel = self
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(AxisID::Only));
        let Some(parallel_panel) = parallel_panel else {
            self.open_message(
                &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                "Unable to execute command",
                Some(AxisID::Only),
            );
            self.send_msg_process_failed_to_start(display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        AbstractParallelDialog::get_parameters(&*parallel_dialog, &mut param);
        if !parallel_panel.get_parameters_processchunks_param_boolean(&param, true) {
            if let Some(main_panel) = self.get_main_panel() {
                main_panel
                    .main_panel()
                    .stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
            }
            self.send_msg_process_failed_to_start(display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        parallel_panel
            .get_parallel_progress_display()
            .reset_results();
        <Self as BaseManager>::processchunks(
            self,
            Some(AxisID::Only),
            Some(Arc::new(param)),
            process_result_display,
            Some(process_series),
            true,
            processing_method,
            false,
            dialog_type,
            None,
            None,
            None,
        );
    }

    /// Java `setNewParamFile(File)`.
    pub fn set_new_param_file_file(&'static self, file: &Path) -> bool {
        if *self.base().loaded_param_file.lock().unwrap() {
            return true;
        }
        // set paramFile and propertyUserDir
        let parent_dir_path = utilities::java_io_file_get_absolute_path(
            &utilities::java_io_file_get_parent(&file.to_string_lossy()).unwrap_or_default(),
        );
        if parent_dir_path.ends_with(' ') {
            self.open_message(
                &format!(
                    "The directory, {parent_dir_path}, cannot be used because it ends with a space."
                ),
                "Unusable Directory Name",
                Some(AxisID::Only),
            );
            return false;
        }
        *self.base().property_user_dir.lock().unwrap() = Some(parent_dir_path.clone());
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!("\npropertyUserDir: {parent_dir_path}");
        }
        // Java dereferences the dialog unchecked; it is the caller.
        if let Some(anisotropic_diffusion_dialog) = self.anisotropic_diffusion_dialog.get() {
            anisotropic_diffusion_dialog.get_initial_parameters(self.meta_data());
        }
        let error_message = self.meta_data().validate();
        if let Some(error_message) = error_message {
            self.open_message(
                &error_message,
                "Anisotropic Diffusion Dialog error",
                Some(AXIS_ID),
            );
            return false;
        }
        self.get_imod_manager()
            .set_meta_data_parallel_meta_data(self.meta_data());
        let param_file = utilities::java_io_file_new(
            &parent_dir_path,
            self.meta_data()
                .get_meta_data_file_name()
                .as_deref()
                .unwrap_or("null"),
        );
        if !self.set_param_file_from(Some(Path::new(&param_file))) {
            return false;
        }
        *self.base().loaded_param_file.lock().unwrap() = true;
        etomo_director::INSTANCE.rename_current_manager(
            self.meta_data()
                .get_root_name()
                .unwrap_or_else(|| "null".to_owned()),
        );
        self.set_status_bar_text();
        true
    }

    /// Java `deleteSubdir(String)`.  Asks to close the files to be cleaned up, and
    /// then cleans up if the files are closed.
    pub fn delete_subdir(&'static self, subdir_name: &str) -> bool {
        if self.close_imods(
            Some(imod_manager::TEST_VOLUME_KEY),
            Some(imod_manager::VARYING_K_TEST_KEY),
            Some(imod_manager::VARYING_ITERATION_TEST_KEY),
            Some(AxisID::Only),
            Some("Temporary files"),
            Some(" must be closed before cleaning up."),
            Some("Should files be closed?"),
        ) {
            let subdir = PathBuf::from(utilities::java_io_file_new(
                &self
                    .get_property_user_dir()
                    .unwrap_or_else(|| "null".to_owned()),
                subdir_name,
            ));
            // Fixed in translation (ParallelManager.java:413): Java's `listFiles`
            // returns null for a missing directory and `.length` throws
            // NullPointerException; a missing directory has nothing to delete.
            if let Ok(entries) = std::fs::read_dir(&subdir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    // `File.delete()` removes a file or an empty directory.
                    if std::fs::remove_file(&path).is_err() {
                        let _ = std::fs::remove_dir(&path);
                    }
                }
            }
            let _ = std::fs::remove_dir(&subdir);
            return true;
        }
        false
    }

    /// The `catch (IOException | SystemProcessException | AxisTypeException e) {
    /// e.printStackTrace(); }` of the `imod` overloads.
    fn print_imod_exception(result: Result<(), ImodManagerException>) {
        match result {
            Ok(()) => {}
            Err(ImodManagerException::SystemProcess(e)) => eprintln!("{e}"),
            Err(ImodManagerException::AxisType(e)) => eprintln!("{e}"),
            Err(ImodManagerException::Io(e)) => eprintln!("{e}"),
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }

    /// Java `imod(FileType, Run3dmodMenuOptions, boolean)`.
    pub fn imod_file_type(
        &'static self,
        file_type: &FileType,
        menu_options: Option<Run3dmodMenuOptions>,
        flip: bool,
    ) {
        let file = file_type.get_file(Some(self), Some(AxisID::Only));
        if !file.as_deref().is_some_and(Path::exists) {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self),
                    "No file to open",
                    "Entry Error",
                )
            });
            return;
        }
        // Fixed in translation: a file type without a 3dmod key makes Java's
        // ImodManager throw NullPointerException; nothing is opened.
        let Some(key) = file_type.get_imod_manager_key() else {
            return;
        };
        Self::print_imod_exception(
            self.get_imod_manager()
                .open_string_file_run3dmod_menu_options_boolean(
                    key,
                    file.as_deref(),
                    menu_options,
                    flip,
                ),
        );
    }

    // Updates done

    /// Java `imod(String, File, Run3dmodMenuOptions, boolean)`.
    pub fn imod_string_file_run3dmod_menu_options_boolean(
        &'static self,
        key: &str,
        file: Option<&Path>,
        menu_options: Option<Run3dmodMenuOptions>,
        flip: bool,
    ) {
        let Some(file) = file else {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self),
                    "No file to open",
                    "Entry Error",
                )
            });
            return;
        };
        Self::print_imod_exception(
            self.get_imod_manager()
                .open_string_file_run3dmod_menu_options_boolean(
                    key,
                    Some(file),
                    menu_options,
                    flip,
                ),
        );
    }

    /// Java `imod(String, File, Run3dmodMenuOptions)`.
    pub fn imod_string_file_run3dmod_menu_options(
        &'static self,
        key: &str,
        file: Option<&Path>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let Some(file) = file else {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self),
                    "No file to open",
                    "Entry Error",
                )
            });
            return;
        };
        Self::print_imod_exception(
            self.get_imod_manager()
                .open_string_axis_id_file_run3dmod_menu_options(
                    key,
                    None,
                    Some(file),
                    menu_options,
                ),
        );
    }

    /// Java `imod(String, Run3dmodMenuOptions)`.
    pub fn imod_string_run3dmod_menu_options(
        &'static self,
        key: &str,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        Self::print_imod_exception(
            self.get_imod_manager()
                .open_string_run3dmod_menu_options(key, menu_options),
        );
    }

    /// Java `imodVaryingKValue(String, Run3dmodMenuOptions, String, String, boolean)`.
    pub fn imod_varying_k_value(
        &'static self,
        key: &str,
        menu_options: Option<Run3dmodMenuOptions>,
        subdir_name: Option<&str>,
        test_volume_name: Option<&str>,
        flip: bool,
    ) {
        let state = self.get_state();
        let file_name_list = anisotropic_diffusion_param::get_test_file_name_list_k(
            Some(self),
            &state.get_test_k_value_list(),
            &state.get_test_iteration(),
            test_volume_name,
        );
        self.imod_list(key, menu_options, subdir_name, &file_name_list, flip);
    }

    /// Java `imodVaryingIteration(String, Run3dmodMenuOptions, String, String,
    /// boolean)`.
    pub fn imod_varying_iteration(
        &'static self,
        key: &str,
        menu_options: Option<Run3dmodMenuOptions>,
        subdir_name: Option<&str>,
        test_volume_name: Option<&str>,
        flip: bool,
    ) {
        let state = self.get_state();
        let file_name_list = anisotropic_diffusion_param::get_test_file_name_list_iteration(
            &state.get_test_k_value(),
            &state.get_test_iteration_list(),
            test_volume_name,
        );
        self.imod_list(key, menu_options, subdir_name, &file_name_list, flip);
    }

    /// Java private `imod(String, Run3dmodMenuOptions, String, List, boolean)`.
    fn imod_list(
        &'static self,
        key: &str,
        menu_options: Option<Run3dmodMenuOptions>,
        subdir_name: Option<&str>,
        file_name_list: &[Option<String>],
        flip: bool,
    ) {
        // The source copies the list into a String[] in three equivalent ways by size.
        let file_name_array: Vec<String> = file_name_list
            .iter()
            .map(|name| name.clone().unwrap_or_else(|| "null".to_owned()))
            .collect();
        Self::print_imod_exception(
            self.get_imod_manager()
                .open_string_string_array_run3dmod_menu_options_string_boolean(
                    key,
                    Some(&file_name_array),
                    menu_options,
                    subdir_name,
                    flip,
                ),
        );
    }

    /// Java `makeSubdir(String)`.
    pub fn make_subdir(&'static self, subdir_name: &str) -> bool {
        if self.param_file().is_none() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self),
                    "Must pick a volume.",
                    "Entry Error",
                )
            });
            return false;
        }
        let subdir = utilities::java_io_file_new(
            &self
                .property_user_dir()
                .unwrap_or_else(|| "null".to_owned()),
            subdir_name,
        );
        if !Path::new(&subdir).exists() {
            let _ = std::fs::create_dir(&subdir);
        }
        true
    }

    /// Java private `updateTrimvolParam(boolean)`.
    fn update_trimvol_param(&'static self, do_validation: bool) -> Option<TrimvolParam> {
        // Get trimvol param data from dialog.
        let mut param = TrimvolParam::new(self, Some(trimvol_param::Mode::Nad));
        let anisotropic_diffusion_dialog = self.anisotropic_diffusion_dialog.get()?;
        if !anisotropic_diffusion_dialog.get_parameters_trimvol_param(&mut param, do_validation) {
            return None;
        }
        anisotropic_diffusion_dialog.get_parameters_for_trimvol(self.meta_data());
        param.set_old_flipped_coordinates(self.meta_data().is_new_style_z());
        Some(param)
    }

    /// `new SetEnvParam(ImodOutputFormat.ENV_VAR)` with the image output format, when
    /// the meta data does not use the old image filename style (the source's
    /// repeated block).
    fn set_env_command(&self) -> Option<String> {
        if !self.meta_data().base().is_old_image_filename_style() {
            // Keep the existing setting if it was set. If not then add it.
            let mut set_env_param = SetEnvParam::new(Some(imod_output_format::ENV_VAR));
            set_env_param.set_value(Some(
                &self
                    .meta_data()
                    .base()
                    .get_image_output_format()
                    .to_string(),
            ));
            return Some(set_env_param.get_command_line());
        }
        None
    }

    /// Java private `updateAnisotropicDiffusionParamForVaryingK(String, boolean) throws
    /// LogFileException, IOException, LockException`.
    fn update_anisotropic_diffusion_param_for_varying_k(
        &'static self,
        _subdir_name: Option<&str>,
        do_validation: bool,
    ) -> Result<Option<AnisotropicDiffusionParam>, LogFileError> {
        let set_env_command = self.set_env_command();
        let mut param =
            AnisotropicDiffusionParam::new(self, Mode::VaryingK, set_env_command.as_deref());
        let Some(anisotropic_diffusion_dialog) = self.anisotropic_diffusion_dialog.get() else {
            return Ok(None);
        };
        if !anisotropic_diffusion_dialog.get_parameters_for_varying_k(&mut param, do_validation) {
            return Ok(None);
        }
        param.delete_test_files();
        param.create_test_files()?;
        Ok(Some(param))
    }

    /// Java private `updateAnisotropicDiffusionParam(boolean) throws ...`.
    fn update_anisotropic_diffusion_param(
        &'static self,
        do_validation: bool,
    ) -> Result<Option<AnisotropicDiffusionParam>, LogFileError> {
        let set_env_command = self.set_env_command();
        let mut param =
            AnisotropicDiffusionParam::new(self, Mode::Full, set_env_command.as_deref());
        let Some(anisotropic_diffusion_dialog) = self.anisotropic_diffusion_dialog.get() else {
            return Ok(None);
        };
        if !anisotropic_diffusion_dialog
            .get_parameters_anisotropic_diffusion_param(&mut param, do_validation)
        {
            return Ok(None);
        }
        param.create_filter_full_file()?;
        Ok(Some(param))
    }

    /// Java private `updateChunksetupParam(DialogType)`.
    fn update_chunksetup_param(
        &'static self,
        dialog_type: Option<DialogType>,
    ) -> Option<ChunksetupParam> {
        let mut param = ChunksetupParam::new(dialog_type);
        if dialog_type == Some(DialogType::AnisotropicDiffusion)
            && let Some(anisotropic_diffusion_dialog) = self.anisotropic_diffusion_dialog.get()
        {
            anisotropic_diffusion_dialog.get_parameters_chunksetup_param(&mut param);
        } else if let Some(parallel_dialog) = self.parallel_dialog.get() {
            if !parallel_dialog.get_parameters_chunksetup_param(&mut param, true) {
                return None;
            }
        } else {
            return None;
        }
        Some(param)
    }

    /// Java `setupAnisotropicDiffusion(boolean)`.
    pub fn setup_anisotropic_diffusion(&'static self, do_validation: bool) -> bool {
        match self.update_anisotropic_diffusion_param(do_validation) {
            Ok(None) => return false,
            Ok(Some(_)) => {}
            Err(LogFileError::Lock(_)) => return false,
            Err(e) => {
                eprintln!("{e}");
                self.open_message(
                    "Anisotropic diffusion comscripts could not be created",
                    "Write Comscript Error",
                    Some(AxisID::Only),
                );
                return false;
            }
        }
        true
    }

    /// Java private `updateAnisotropicDiffusionParamForVaryingIteration(String,
    /// boolean)`.
    fn update_anisotropic_diffusion_param_for_varying_iteration(
        &'static self,
        _subdir_name: Option<&str>,
        do_validation: bool,
    ) -> Option<AnisotropicDiffusionParam> {
        let mut param = AnisotropicDiffusionParam::new(self, Mode::VaryingIterations, None);
        let anisotropic_diffusion_dialog = self.anisotropic_diffusion_dialog.get()?;
        if !anisotropic_diffusion_dialog
            .get_parameters_for_varying_iteration(&mut param, do_validation)
        {
            return None;
        }
        Some(param)
    }

    /// Java `chunksetup(ProcessSeries, Deferred3dmodButton, Run3dmodMenuOptions,
    /// DialogType, ProcessingMethod)`.
    pub fn chunksetup(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
        processing_method: Option<ProcessingMethod>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, AxisID::Only, dialog_type, Some("chunksetup")),
        };
        if dialog_type == Some(DialogType::AnisotropicDiffusion) {
            if !self.anisotropic_diffusion_dialog.is_some() {
                self.open_message(
                    "A Anisotropic diffusion dialog not open",
                    "Program logic error",
                    Some(AxisID::Only),
                );
                process_series.borrow().end_series();
                return;
            }
            if !self.setup_anisotropic_diffusion(true) {
                process_series.borrow().end_series();
                return;
            }
        } else if !self.parallel_dialog.is_some() {
            self.open_message(
                &format!("{} dialog not open", etomo_menu::GENERIC_LABEL),
                "Program logic error",
                Some(AxisID::Only),
            );
            process_series.borrow().end_series();
            return;
        }
        let Some(param) = self.update_chunksetup_param(dialog_type) else {
            process_series.borrow().end_series();
            return;
        };
        if dialog_type == Some(DialogType::AnisotropicDiffusion) {
            process_series.borrow_mut().set_next_process(
                Some(&ProcessName::ANISOTROPIC_DIFFUSION.to_string()),
                processing_method,
            );
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        let thread_name = match self.process_mgr().chunksetup(
            Arc::new(param),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                let message = [
                    format!("Can not execute {}command", ProcessName::CHUNKSETUP),
                    e.to_string(),
                ];
                self.open_message_array(&message, "Unable to execute command", AxisID::Only);
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(AxisID::Only));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&ProcessName::CHUNKSETUP.to_string()),
                AxisID::Only,
                Some(&ProcessName::CHUNKSETUP),
            );
        }
    }

    /// Java `setChunkSetupOutputFile(String[])`.  Finds the message containing the
    /// output file, parses it, and send the output file to the Parallel dialog.  Has
    /// no affect on the NAD dialog.
    pub fn set_chunk_setup_output_file(&self, stdout: Option<&[String]>) {
        let (Some(stdout), Some(parallel_dialog)) = (stdout, self.parallel_dialog.get()) else {
            return;
        };
        for line in stdout {
            if line.contains(process_output_strings::CHUNK_SETUP_OUTPUT_FILE) {
                let chars: Vec<char> = line.chars().collect();
                let start_index = line.find(':').map(|index| line[..index].chars().count());
                let Some(mut start_index) = start_index.filter(|index| index + 1 < chars.len())
                else {
                    eprintln!("Warning: unable to parse message:\n{line}");
                    return;
                };
                start_index += 1;
                let end_index = line
                    .rfind(process_output_strings::OPEN_BRACKET)
                    .map(|index| line[..index].chars().count());
                // `substring(startIndex + 1, ...)`: one character past the colon.
                let begin = start_index + 1;
                if let Some(end_index) = end_index.filter(|end_index| start_index < *end_index) {
                    let text: String = chars[begin.min(end_index)..end_index].iter().collect();
                    parallel_dialog.set_chunk_setup_output_file(Some(&text));
                } else {
                    let text: String = chars[begin.min(chars.len())..].iter().collect();
                    parallel_dialog.set_chunk_setup_output_file(Some(&text));
                }
            }
        }
    }

    /// Java `setParallelProcessName(String)`.
    pub fn set_parallel_process_name(&self, process_name: Option<&str>) {
        if let Some(parallel_dialog) = self.parallel_dialog.get() {
            parallel_dialog.set_process_name(
                Some(PathBuf::from(
                    self.property_user_dir()
                        .unwrap_or_else(|| "null".to_owned()),
                )),
                &format!(
                    "{}{}",
                    process_name.unwrap_or("null"),
                    chunksetup_param::RESULT_SUFFIX
                ),
            );
        }
    }

    /// Java `anisotropicDiffusionVaryingIteration(String, ProcessSeries,
    /// Deferred3dmodButton, Run3dmodMenuOptions, DialogType)`.
    pub fn anisotropic_diffusion_varying_iteration(
        &'static self,
        subdir_name: Option<&str>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(
                self,
                AxisID::Only,
                dialog_type,
                Some("anisotropicDiffusionVaryingIteration"),
            ),
        };
        if !self.anisotropic_diffusion_dialog.is_some() {
            self.open_message(
                "B Anisotropic diffusion dialog not open",
                "Program logic error",
                Some(AxisID::Only),
            );
            process_series.borrow().end_series();
            return;
        }
        let Some(param) =
            self.update_anisotropic_diffusion_param_for_varying_iteration(subdir_name, true)
        else {
            process_series.borrow().end_series();
            return;
        };
        if !self.validate_test_volume(&param) {
            process_series.borrow().end_series();
            return;
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        // Start the trimvol process
        let thread_name = match self.process_mgr().anisotropic_diffusion(
            Arc::new(param),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = [
                    format!(
                        "Can not execute {}command",
                        ProcessName::ANISOTROPIC_DIFFUSION
                    ),
                    e.to_string(),
                ];
                self.open_message_array(&message, "Unable to execute command", AxisID::Only);
                ProcessSeries::start_fail_process(&process_series, AxisID::Only);
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(AxisID::Only));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&ProcessName::ANISOTROPIC_DIFFUSION.to_string()),
                AxisID::Only,
                Some(&ProcessName::ANISOTROPIC_DIFFUSION),
            );
        }
    }

    /// Java `anisotropicDiffusion(ProcessSeries, ProcessingMethod, DialogType)`.
    pub fn anisotropic_diffusion(
        &'static self,
        process_series: Option<&ProcessSeriesHandle>,
        processing_method: Option<ProcessingMethod>,
        dialog_type: Option<DialogType>,
    ) -> bool {
        let Some(anisotropic_diffusion_dialog) = self.anisotropic_diffusion_dialog.get() else {
            self.open_message(
                "C Anisotropic diffusion dialog not open",
                "Program logic error",
                Some(AxisID::Only),
            );
            return false;
        };
        let mut param = ProcesschunksParam::get_instance_process_name(
            self,
            AxisID::Only,
            ProcessName::ANISOTROPIC_DIFFUSION,
            Some(OutputImageFileKey::FileType(
                file_type::CLASS.anisotropic_diffusion_output.clone(),
            )),
        );
        let parallel_panel = self
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(AxisID::Only));
        let Some(parallel_panel) = parallel_panel else {
            self.open_message(
                &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                "Unable to execute command",
                Some(AxisID::Only),
            );
            if let Some(main_panel) = self.get_main_panel() {
                main_panel
                    .main_panel()
                    .stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
            }
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return false;
        };
        AbstractParallelDialog::get_parameters(&*anisotropic_diffusion_dialog, &mut param);
        if !parallel_panel.get_parameters_processchunks_param_boolean(&param, true) {
            if let Some(main_panel) = self.get_main_panel() {
                main_panel
                    .main_panel()
                    .stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
            }
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return false;
        }
        parallel_panel.reset_results();
        <Self as BaseManager>::processchunks(
            self,
            Some(AxisID::Only),
            Some(Arc::new(param)),
            None,
            process_series.cloned(),
            true,
            processing_method,
            false,
            dialog_type,
            None,
            None,
            None,
        );
        true
    }

    /// Java private `validateTestVolume(AnisotropicDiffusionParam)`.  Attempts to
    /// validate the size of the test volume.  Returns false if the test volume is too
    /// big (larger then the "Memory per chunk" spinner divided by 36,
    /// ChunksetupParam.MEMORY_TO_VOXEL).  Returns true if the test volume is not too
    /// big or if the validation cannot be done.
    fn validate_test_volume(&'static self, param: &AnisotropicDiffusionParam) -> bool {
        let directory = utilities::java_io_file_get_absolute_path(&utilities::java_io_file_new(
            &self
                .property_user_dir()
                .unwrap_or_else(|| "null".to_owned()),
            param.get_subdir_name().as_deref().unwrap_or("null"),
        ));
        let Some(test_volume_header) = MRCHeader::get_instance_in_dir(
            Some(&directory),
            param.get_input_file_name().as_deref(),
            Some(AxisID::Only),
        ) else {
            return true;
        };
        let read = test_volume_header.borrow_mut().read_with_manager(self);
        match read {
            Err(e) => {
                eprintln!("{e}");
                eprintln!("Unable to validate test volume.");
            }
            Ok(_) => {
                let header = test_volume_header.borrow();
                let size = utilities::java_lang_math_round(
                    header.get_n_columns() as f64 / 1024.0
                        * header.get_n_rows() as f64
                        * header.get_n_sections() as f64
                        * chunksetup_param::MEMORY_TO_VOXEL as f64
                        / 1024.0,
                );
                let Some(anisotropic_diffusion_dialog) = self.anisotropic_diffusion_dialog.get()
                else {
                    return true;
                };
                let memory_per_chunk = anisotropic_diffusion_dialog.get_memory_per_chunk();
                if size > 0 && size > memory_per_chunk.int_value() as i64 {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self),
                            &format!(
                                "Processing this test volume would require {size} MB of memory, more than the limit of {memory_per_chunk} MB that you have set for chunks when processing the whole volume.  Make a smaller test volume or increase the memory limit for chunks in the \"{}\" spinner in the {} box.",
                                anisotropic_diffusion_dialog::MEMORY_PER_CHUNK_LABEL,
                                anisotropic_diffusion_dialog::FILTER_FULL_VOLUME_LABEL
                            ),
                            "Entry Error",
                            Some(AxisID::Only),
                        )
                    });
                    return false;
                }
            }
        }
        true
    }

    /// Java `anisotropicDiffusionVaryingK(String, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, ProcessingMethod)`.
    pub fn anisotropic_diffusion_varying_k(
        &'static self,
        subdir_name: Option<&str>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
        processing_method: Option<ProcessingMethod>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(
                self,
                AxisID::Only,
                dialog_type,
                Some("anisotropicDiffusionVaryingK"),
            ),
        };
        let Some(anisotropic_diffusion_dialog) = self.anisotropic_diffusion_dialog.get() else {
            self.open_message(
                "D Anisotropic diffusion dialog not open",
                "Program logic error",
                Some(AxisID::Only),
            );
            process_series.borrow().end_series();
            return;
        };
        let anisotropic_diffusion_param =
            match self.update_anisotropic_diffusion_param_for_varying_k(subdir_name, true) {
                Ok(Some(param)) => param,
                Ok(None) => {
                    process_series.borrow().end_series();
                    return;
                }
                Err(LogFileError::Lock(_)) => {
                    process_series.borrow().end_series();
                    return;
                }
                Err(e) => {
                    eprintln!("{e}");
                    self.open_message(
                        "Anisotropic diffusion comscripts could not be created",
                        "Write Comscript Error",
                        Some(AxisID::Only),
                    );
                    process_series.borrow().end_series();
                    return;
                }
            };
        if !self.validate_test_volume(&anisotropic_diffusion_param) {
            process_series.borrow().end_series();
            return;
        }
        let mut param = ProcesschunksParam::get_instance_process_name(
            self,
            AxisID::Only,
            ProcessName::ANISOTROPIC_DIFFUSION,
            None,
        );
        param.set_subcommand_details(Some(Arc::new(anisotropic_diffusion_param)));
        let parallel_panel = self
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(AxisID::Only));
        let Some(parallel_panel) = parallel_panel else {
            self.open_message(
                &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                "Unable to execute command",
                Some(AxisID::Only),
            );
            if let Some(main_panel) = self.get_main_panel() {
                main_panel
                    .main_panel()
                    .stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
            }
            process_series.borrow().end_series();
            return;
        };
        AbstractParallelDialog::get_parameters(&*anisotropic_diffusion_dialog, &mut param);
        if !parallel_panel.get_parameters_processchunks_param_boolean(&param, true) {
            if let Some(main_panel) = self.get_main_panel() {
                main_panel
                    .main_panel()
                    .stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
            }
            process_series.borrow().end_series();
            return;
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        parallel_panel.reset_results();
        <Self as BaseManager>::processchunks(
            self,
            Some(AxisID::Only),
            Some(Arc::new(param)),
            None,
            Some(process_series),
            true,
            processing_method,
            false,
            dialog_type,
            None,
            None,
            None,
        );
    }

    /// Java `trimVolume(ProcessSeries)`.  Execute trimvol.
    pub fn trim_volume(&'static self, process_series: Option<ProcessSeriesHandle>) {
        if !self.anisotropic_diffusion_dialog.is_some() {
            self.open_message(
                "E Anisotropic diffusion dialog not open",
                "Program logic error",
                Some(AxisID::Only),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let Some(param) = self.update_trimvol_param(true) else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        // Start the trimvol process
        let thread_name = match self.process_mgr().trim_volume(
            Arc::new(param),
            process_series.map(|process_series| Arc::new(EdtRef::new(process_series))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = ["Can not execute trimvol command".to_owned(), e.to_string()];
                self.open_message_array(&message, "Unable to execute command", AxisID::Only);
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(AxisID::Only));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Trimming volume"),
                AxisID::Only,
                Some(&ProcessName::TRIMVOL),
            );
        }
    }

    /// Java private `setNewParamFile()`.
    fn set_new_param_file_void(&'static self) -> bool {
        if *self.base().loaded_param_file.lock().unwrap() {
            return true;
        }
        let Some(parallel_dialog) = self.parallel_dialog.get() else {
            return false;
        };
        // set paramFile and propertyUserDir
        // Fixed in translation (ParallelManager.java:999): Java dereferences a null
        // working directory (no chunk comscript chosen); `ParallelDialog` stops that
        // action before it gets here, so this is not reached with one.
        let Some(working_dir) = parallel_dialog.get_working_dir() else {
            return false;
        };
        let working_dir = utilities::java_io_file_get_absolute_path(&working_dir.to_string_lossy());
        if working_dir.ends_with(' ') {
            self.open_message(
                &format!(
                    "The directory, {working_dir}, cannot be used because it ends with a space."
                ),
                "Unusable Directory Name",
                Some(AxisID::Only),
            );
            return false;
        }
        *self.base().property_user_dir.lock().unwrap() = Some(working_dir.clone());
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!("\npropertyUserDir: {working_dir}");
        }
        parallel_dialog.get_parameters_meta_data(self.meta_data());
        let error_message = self.meta_data().validate();
        if let Some(error_message) = error_message {
            self.open_message(&error_message, "Anisotropic Diffusion Error", Some(AXIS_ID));
            return false;
        }
        let param_file = utilities::java_io_file_new(
            &working_dir,
            self.meta_data()
                .get_meta_data_file_name()
                .as_deref()
                .unwrap_or("null"),
        );
        if !self.set_param_file_from(Some(Path::new(&param_file))) {
            return false;
        }
        eprintln!("paramFile: {param_file}");
        etomo_director::INSTANCE.rename_current_manager(
            self.meta_data()
                .get_root_name()
                .unwrap_or_else(|| "null".to_owned()),
        );
        self.set_status_bar_text();
        true
    }

    /// Java private `saveDialog()`.
    fn save_dialog(&'static self) {
        self.save_parallel_dialog();
        self.save_anisotropic_diffusion_dialog();
    }
}

impl BaseManager for ParallelManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    /// Java `getInterfaceType()`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::Pp)
    }

    /// Java package-private `createMainPanel()`.
    fn create_main_panel(&self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            let this = self.this_static();
            self.main_panel.set(Some(MainParallelPanel::new(this)));
        }
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

    /// Java `getBaseState()`.
    fn get_base_state(&self) -> Option<&'static dyn BaseState> {
        self.this_static()
            .state
            .get()
            .map(|state| state as &'static dyn BaseState)
    }

    /// Java `getMainPanel()`.
    fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.main_panel
            .get()
            .map(|main_panel| main_panel as Rc<dyn MainPanelVirtual>)
    }

    /// Java `getFileSubdirectoryName()`.
    fn get_file_subdirectory_name(&self) -> Option<String> {
        let anisotropic_diffusion_dialog = self.anisotropic_diffusion_dialog.get()?;
        anisotropic_diffusion_dialog.get_subdirectory()
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
        storables[index] = this.state.get().map(|state| state as &'static dyn Storable);
        Some(storables)
    }

    /// Java `getProcessManager()`.
    fn get_process_manager(&self) -> Option<&'static BaseProcessManager> {
        self.process_mgr.get().map(|process_mgr| &process_mgr.base)
    }

    /// Java `save() throws LogFileException, IOException, LockException`.
    fn save(&'static self) -> Result<bool, LogFileError> {
        self.save_super()?;
        // Java dereferences mainPanel, which is null in headless mode; it is skipped
        // here.
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.done();
        }
        self.save_dialog();
        Ok(true)
    }

    /// Java `exitProgram(AxisID)`.
    fn exit_program(&'static self, axis_id: Option<AxisID>) -> bool {
        // try { ... } catch (Throwable e) { e.printStackTrace(); return true; }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if self.exit_program_super(axis_id) {
                self.end_threads();
                self.save_param_file()?;
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

    /// Java package-private `startNextProcess(UIComponent, AxisID,
    /// ProcessSeries.Process, ProcessResultDisplay, ProcessSeries, DialogType,
    /// ProcessDisplay)`.  Returns true if the process is recognized.
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
        if process.equals_string(Some(&ProcessName::ANISOTROPIC_DIFFUSION.to_string())) {
            self.anisotropic_diffusion(
                Some(process_series),
                process.get_processing_method(),
                dialog_type,
            );
            return true;
        }
        false
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        self.meta_data
            .get()
            .and_then(|meta_data| meta_data.get_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::util::event_queue;

    #[test]
    fn retains_the_process_manager_and_reports_pp() {
        let manager = event_queue::invoke_and_wait(ParallelManager::new);
        assert!(std::ptr::eq(manager.process_mgr().get_manager(), manager));
        assert_eq!(manager.get_interface_type(), Some(InterfaceType::Pp));
        assert!(!manager.can_snapshot());
    }
}
