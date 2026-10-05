//! `IMOD/Etomo/src/etomo/SerialSectionsManager.java`.
//!
//! The manager of the Serial Sections interface (`.ess`): the startup dialog, the
//! serial sections dialog, the preblend/blend/newst comscripts and the process
//! manager that runs extractpieces, blendmont, xftoxg and newst.
//!
//! **Representation.**  `SerialSectionsManager extends BaseManager`, so - as
//! `etomo/base_manager.rs` sets out - the superclass state is the `base` field and the
//! superclass methods are the `BaseManager` trait, which this struct implements; the
//! `@Override` members are the trait implementations and everything else is an
//! inherent method.  Java's managers are created by `EtomoDirector` and never
//! collected, so the constructor leaks its allocation and hands back
//! `&'static SerialSectionsManager`.
//!
//! **Threads.**  The manager is shared with process threads (`Send + Sync`); the main
//! panel and the dialogs are event dispatch thread objects, held in `EdtCell`s and
//! reached only on that thread.
//!
//! **Headless.**  The constructor builds neither the main panel nor the dialogs when
//! etomo runs headless, and Java then dereferences the null `mainPanel` in the members
//! that reach it (NullPointerException).  Those members skip the missing panel here
//! ("fixed in translation", `BUGS.md`).
//!
//! **The modal startup dialog.**  Java's `display()` blocks in the startup dialog's
//! modal `setVisible(true)`; the `jdk` dialog does not block, so what `EtomoDirector`
//! runs after `display()` is handed to [`SerialSectionsManager::display_then`].

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

use crate::imod::etomo::auto_alignment_controller::AutoAlignmentController;
use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::comscript::blendmont_param::{self, BlendmontParam};
use crate::imod::etomo::comscript::extractpieces_param::{self, ExtractpiecesParam};
use crate::imod::etomo::comscript::midas_param::{self, MidasParam};
use crate::imod::etomo::comscript::newst_param::{self, NewstParam, SetSizeToOutputInXandYError};
use crate::imod::etomo::comscript::serial_sections_com_script_manager::SerialSectionsComScriptManager;
use crate::imod::etomo::comscript::set_env_param::SetEnvParam;
use crate::imod::etomo::comscript::tomodataplots_param;
use crate::imod::etomo::comscript::xfalign_param::XfalignParam;
use crate::imod::etomo::comscript::xftoxg_param::{self, XftoxgParam};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::logic::serial_sections_startup_data::SerialSectionsStartupData;
use crate::imod::etomo::process::base_imod_manager::ImodManagerException;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::process_interface::{ProcessResultDisplayRef, ProcessSeriesRef};
use crate::imod::etomo::process::serial_sections_process_manager::SerialSectionsProcessManager;
use crate::imod::etomo::process_series::{Process, ProcessSeries, ProcessSeriesHandle};
use crate::imod::etomo::storage::blendmont_log::BlendmontLog;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::auto_alignment_meta_data::AutoAlignmentMetaData;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::const_serial_sections_meta_data::ConstSerialSectionsMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::imod_output_format;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::serial_sections_meta_data::SerialSectionsMetaData;
use crate::imod::etomo::r#type::serial_sections_state::SerialSectionsState;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::UiComponent;
use crate::imod::etomo::ui::auto_alignment_display::AutoAlignmentDisplay;
use crate::imod::etomo::ui::swing::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::ui::swing::main_panel::MainPanelVirtual;
use crate::imod::etomo::ui::swing::main_serial_sections_panel::MainSerialSectionsPanel;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::ui::swing::serial_sections_dialog::SerialSectionsDialog;
use crate::imod::etomo::ui::swing::serial_sections_startup_dialog::SerialSectionsStartupDialog;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::event_queue::{EdtCell, EdtRef};
use crate::imod::etomo::util::utilities;

/// Java private static final `AXIS_ID`.
const AXIS_ID: AxisID = AxisID::Only;

/// Java `public final class SerialSectionsManager extends BaseManager`.
pub struct SerialSectionsManager {
    /// Java superclass `BaseManager` state.
    base: BaseManagerBase,
    /// Java private final `state = new SerialSectionsState()`.
    state: SerialSectionsState,
    /// Java private final `metaData`, assigned in the constructor body after `super()`.
    meta_data: OnceLock<SerialSectionsMetaData>,
    /// Java private final `processMgr`.
    process_mgr: OnceLock<&'static SerialSectionsProcessManager>,
    /// Java private `startupDialog`, initially null.
    startup_dialog: EdtCell<Rc<SerialSectionsStartupDialog>>,
    /// Java private `dialog`, initially null.
    dialog: EdtCell<Rc<SerialSectionsDialog>>,
    /// Java private `autoAlignmentController`, initially null.
    auto_alignment_controller: Mutex<Option<&'static AutoAlignmentController>>,
    /// Java private `valid`, initially true.  Valid is for handling failure before
    /// the manager key is set in EtomoDirector.
    valid: Mutex<bool>,
    /// Java private `origUserDir`, initially null.
    orig_user_dir: Mutex<Option<String>>,
    /// Java private `mainPanel`, initialized during the parent constructor.
    main_panel: EdtCell<Rc<MainSerialSectionsPanel>>,
    /// Java private `comScriptMgr`, initialized during the parent constructor.
    com_script_mgr: OnceLock<&'static SerialSectionsComScriptManager>,
}

/// Owns every `SerialSectionsManager` this module builds.  Java's owner is the
/// collector, by way of `EtomoDirector.managerList`, which keeps each manager for the
/// run; the translation hands out `&'static Self`, so without a root here the
/// allocation is unreachable the moment the constructor returns.
static INSTANCES: Mutex<Vec<&'static SerialSectionsManager>> = Mutex::new(Vec::new());

/// Owns every `SerialSectionsComScriptManager` this module builds, for the same reason.
static COM_SCRIPT_ROOTS: Mutex<Vec<&'static SerialSectionsComScriptManager>> =
    Mutex::new(Vec::new());

/// The `Process` handle of a process series as the process managers take it.
fn series_ref(process_series: &ProcessSeriesHandle) -> ProcessSeriesRef {
    Arc::new(EdtRef::new(Rc::clone(process_series)))
}

impl SerialSectionsManager {
    /// Java private `SerialSectionsManager()`: `this("")`.
    fn new() -> &'static SerialSectionsManager {
        Self::new_with_param_file_name(Some(""))
    }

    /// Java package-private `SerialSectionsManager(String)`.
    fn new_with_param_file_name(param_file_name: Option<&str>) -> &'static SerialSectionsManager {
        let instance: &'static SerialSectionsManager = Box::leak(Box::new(SerialSectionsManager {
            base: BaseManagerBase::initial(),
            state: SerialSectionsState::new(),
            meta_data: OnceLock::new(),
            process_mgr: OnceLock::new(),
            startup_dialog: EdtCell::new(),
            dialog: EdtCell::new(),
            auto_alignment_controller: Mutex::new(None),
            valid: Mutex::new(true),
            orig_user_dir: Mutex::new(None),
            main_panel: EdtCell::new(),
            com_script_mgr: OnceLock::new(),
        }));
        INSTANCES.lock().unwrap().push(instance);
        // Java `super()`.
        instance.base_manager();
        let _ = instance.meta_data.set(SerialSectionsMetaData::new(
            instance,
            instance.get_log_properties(),
            param_file_name.is_none_or(str::is_empty),
        ));
        let _ = instance
            .process_mgr
            .set(SerialSectionsProcessManager::new(instance));
        instance.initialize_ui_parameters_from_name(param_file_name, Some(AXIS_ID));
        if *instance.base.loaded_param_file.lock().unwrap() {
            instance
                .get_imod_manager()
                .set_meta_data_serial_sections_meta_data(instance.meta_data());
        }
        instance
    }

    /// The constructed manager at its final address (Java `this` inside the
    /// overrides, which take `&self`).
    fn this_static(&self) -> &'static SerialSectionsManager {
        INSTANCES
            .lock()
            .unwrap()
            .iter()
            .copied()
            .find(|manager| std::ptr::eq(*manager, self))
            .expect("constructed SerialSectionsManager")
    }

    /// Java field read `metaData`.
    fn meta_data(&self) -> &SerialSectionsMetaData {
        self.meta_data
            .get()
            .expect("metaData is assigned by the constructor")
    }

    /// Java field read `processMgr`.
    fn process_mgr(&self) -> &'static SerialSectionsProcessManager {
        self.process_mgr.get().expect("processMgr")
    }

    /// Java field read `comScriptMgr`.
    fn com_script_mgr(&self) -> &'static SerialSectionsComScriptManager {
        self.com_script_mgr.get().expect("comScriptMgr")
    }

    /// Java field read `propertyUserDir`.
    fn property_user_dir(&self) -> Option<String> {
        self.base.property_user_dir.lock().unwrap().clone()
    }

    /// Java field read `loadedParamFile`.
    fn loaded_param_file(&self) -> bool {
        *self.base.loaded_param_file.lock().unwrap()
    }

    /// Java `uiHarness.openMessageDialog(this, message, title, axisID)`.
    fn open_message(&self, message: &str, title: &str, axis_id: Option<AxisID>) {
        let this = self.this_static();
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string_axis_id(
                Some(this),
                message,
                title,
                axis_id,
            )
        });
    }

    /// Java `uiHarness.openMessageDialog(this, String[], title, axisID)`.
    fn open_message_array(&self, message: &[String], title: &str, axis_id: Option<AxisID>) {
        let this = self.this_static();
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_array_string_axis_id(
                Some(this),
                message,
                title,
                axis_id,
            )
        });
    }

    /// Java `mainPanel.setStatusBarText(paramFile, metaData, logWindow)`.
    fn set_main_panel_status_bar_text(&self) {
        if let Some(main_panel) = self.main_panel.get() {
            let param_file = self.base.param_file.lock().unwrap().clone();
            let log_window = self.get_log_window();
            main_panel.set_status_bar_text(
                param_file.as_deref(),
                Some(self.meta_data() as &dyn BaseMetaData),
                log_window.as_ref(),
            );
        }
    }

    /// Java static package-private `getInstance()`.
    pub fn get_instance() -> &'static SerialSectionsManager {
        let instance = Self::new();
        instance.open_dialog();
        instance
    }

    /// Java static package-private `getInstance(String)`.
    pub fn get_instance_with_param_file_name(
        param_file_name: Option<&str>,
    ) -> &'static SerialSectionsManager {
        let instance = Self::new_with_param_file_name(param_file_name);
        instance.open_dialog();
        instance
    }

    /// Java private `openDialog()`.
    fn open_dialog(&'static self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            self.open_processing_panel();
            self.set_main_panel_status_bar_text();
            if self.loaded_param_file() {
                // catch (final LockException e) {}
                self.open_serial_sections_dialog(None);
            } else {
                self.open_serial_sections_startup_dialog();
            }
        }
    }

    /// Java package-private `display()`.
    pub fn display(&self) {
        if let Some(startup_dialog) = self.startup_dialog.get() {
            startup_dialog.display();
        }
    }

    /// What `EtomoDirector` runs after the blocking `display()` (see the module
    /// comment): at once when no modal startup dialog is showing, else once it is
    /// hidden.  Rust-only plumbing for the modal dialog.
    pub fn display_then(&self, job: Box<dyn FnOnce()>) {
        match self.startup_dialog.get() {
            Some(startup_dialog) => startup_dialog.get_dialog().after_modal_return(job),
            None => job(),
        }
    }

    /// Java private `openProcessingPanel()`.  MUST run reconnect for all axes.
    fn open_processing_panel(&'static self) {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_processing_panel(AxisType::SingleAxis);
        }
        self.set_panel();
        self.reconnect(
            Some(self.get_axis_process_data().get_saved_process_data(AXIS_ID)),
            Some(AXIS_ID),
            true,
            None,
        );
    }

    /// Java private `openSerialSectionsStartupDialog()`.  Create (if necessary) and
    /// show the serial sections startup dialog.
    fn open_serial_sections_startup_dialog(&'static self) {
        if !self.startup_dialog.is_some() {
            let action_message = utilities::prepare_dialog_action_message(
                Some(DialogType::SerialSectionsStartup),
                AxisID::Only,
                None,
            );
            if let Some(main_panel) = self.main_panel.get() {
                main_panel
                    .set_static_progress_bar(Some("Starting Serial Sections interface"), AXIS_ID);
            }
            self.startup_dialog
                .set(Some(SerialSectionsStartupDialog::get_instance(
                    self, AXIS_ID,
                )));
            if let Some(action_message) = action_message {
                eprintln!("{action_message}");
            }
        }
    }

    /// Java `setStartupData(SerialSectionsStartupData)`.
    pub fn set_startup_data(&'static self, startup_data: Option<&SerialSectionsStartupData>) {
        self.startup_dialog.set(None);
        self.set_param_file_from_startup_data(startup_data);
        // catch (final LockException e) {}
        self.open_serial_sections_dialog(startup_data);
        if !self.loaded_param_file() {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    AXIS_ID,
                    Some(ProcessEndState::Failed),
                );
            }
            self.open_message(
                "Failed to load or create parameter file, unable to continue.",
                "Failed",
                Some(AxisID::Only),
            );
            *self.valid.lock().unwrap() = false;
            return;
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel
                .stop_progress_bar_axis_id_process_end_state(AXIS_ID, Some(ProcessEndState::Done));
        }
    }

    /// Java `setParamFile(SerialSectionsStartupData)`.  Tries to set paramFile.
    /// Returns true if able to set paramFile.  If paramFile is already set, returns
    /// true.  Returns false if unable to set paramFile.  Updates the serial sections
    /// dialog display if paramFile was set successfully.
    pub fn set_param_file_from_startup_data(
        &'static self,
        startup_data: Option<&SerialSectionsStartupData>,
    ) -> bool {
        if self.loaded_param_file() {
            return true;
        }
        let Some(startup_data) = startup_data else {
            return false;
        };
        let name = startup_data.get_root_name();
        let _ = name;
        // `getParamFile()` is null only without a stack, which `validate()` refused.
        let Some(param_file) = startup_data.get_param_file() else {
            return false;
        };
        if !param_file.exists() {
            self.process_mgr()
                .create_new_file(&utilities::java_io_file_get_absolute_path(
                    &param_file.to_string_lossy(),
                ));
        }
        self.initialize_ui_parameters(Some(&param_file), Some(AXIS_ID), false);
        if !self.loaded_param_file() {
            return false;
        }
        self.meta_data().set_startup_data(startup_data);
        if !self.meta_data().is_valid() {
            self.open_message(
                "Invalid data, unable to proceed.  Please exit and restart Etomo",
                "Fatal Error",
                None,
            );
            return false;
        }
        self.get_imod_manager()
            .set_meta_data_serial_sections_meta_data(self.meta_data());
        self.set_main_panel_status_bar_text();
        etomo_director::INSTANCE
            .rename_current_manager(BaseMetaData::get_name(self.meta_data()).unwrap_or_default());
        true
    }

    /// Java `cancelStartup()`.
    pub fn cancel_startup(&self) {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id_process_end_state(
                AXIS_ID,
                Some(ProcessEndState::Killed),
            );
        }
        etomo_director::INSTANCE.close_current_manager(Some(AxisID::Only), false);
    }

    /// Java private `openSerialSectionsDialog(SerialSectionsStartupData) throws
    /// LockException`.  Create (if necessary) and show the serial sections dialog.
    /// Update data if the param file has been set.
    fn open_serial_sections_dialog(
        &'static self,
        startup_data: Option<&SerialSectionsStartupData>,
    ) {
        if !self.loaded_param_file() && startup_data.is_none() {
            self.open_message(
                "Failed to load the parameter file, unable to continue.",
                "Failed",
                Some(AxisID::Only),
            );
            *self.valid.lock().unwrap() = false;
            return;
        }
        if !self.dialog.is_some() {
            self.dialog
                .set(Some(SerialSectionsDialog::get_instance(self, AXIS_ID)));
        }
        let dialog = self.dialog.get().expect("dialog");
        let stack = ConstSerialSectionsMetaData::get_stack(self.meta_data());
        let auto_alignment_controller = AutoAlignmentController::new(
            self,
            dialog.clone() as Rc<dyn AutoAlignmentDisplay>,
            self.get_imod_manager(),
            Some(&stack),
        );
        *self.auto_alignment_controller.lock().unwrap() = Some(auto_alignment_controller);
        dialog.set_auto_alignment_controller(auto_alignment_controller);
        if self.loaded_param_file() {
            // The declared LockException is caught by the callers (empty catch).
            let _ = auto_alignment_controller.create_empty_xf_file();
        }
        self.set_serial_sections_dialog_parameters();
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&dialog.get_root_container(), AXIS_ID);
        }
        let action_message = utilities::prepare_dialog_action_message(
            Some(DialogType::SerialSections),
            AxisID::Only,
            None,
        );
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java `completeStartup(UIComponent, AxisID)`.
    pub fn complete_startup(
        &'static self,
        _ui_component: Option<Rc<dyn UiComponent>>,
        axis_id: AxisID,
    ) {
        let process_series = ProcessSeries::new(
            self,
            axis_id,
            Some(DialogType::SerialSectionsStartup),
            Some("completeStartup"),
        );
        {
            let mut series = process_series.borrow_mut();
            series.set_next_process_task(Rc::new(Task::ChangeDirectory));
            series.add_process(Rc::new(Task::CreateComscripts));
            series.add_process(Rc::new(Task::AddOutputFormat));
            series.add_process_force(Rc::new(Task::CopyDistortionFieldFile), true);
            series.add_process(Rc::new(Task::DoneStartupDialog));
            series.add_process(Rc::new(Task::ExtractPieces));
            series.set_fail_process(Rc::new(Task::ResetStartupState));
        }
        ProcessSeries::start_next_process(&process_series, axis_id);
    }

    /// Java `preblend(ProcessSeries, ProcessResultDisplay, Deferred3dmodButton, AxisID,
    /// Run3dmodMenuOptions, DialogType)`.
    pub fn preblend(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        process_result_display: Option<ProcessResultDisplayRef>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        axis_id: AxisID,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, dialog_type, Some("preblend")),
        };
        if self.get_view_type() != ViewType::Montage || !self.dialog.is_some() {
            ProcessSeries::start_fail_process(&process_series, axis_id);
            return;
        }
        let Some(param) = self.update_preblend_comscript(axis_id, true) else {
            ProcessSeries::start_fail_process(&process_series, axis_id);
            return;
        };
        let thread_name = match self.process_mgr().blend(
            Arc::new(param),
            process_result_display,
            axis_id,
            Some(series_ref(&process_series)),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                self.open_message(
                    &format!("Unable to run preblend.\n{}", e),
                    "Process Failed",
                    Some(axis_id),
                );
                ProcessSeries::start_fail_process(&process_series, axis_id);
                return;
            }
        };
        let stack = self.get_stack();
        if let Some(stack) = stack
            && !dataset_tool::is_one_by(
                self.get_property_user_dir().as_deref(),
                Some(&utilities::java_io_file_get_name(&stack.to_string_lossy())),
                self,
                axis_id,
            )
        {
            process_series
                .borrow_mut()
                .add_process(Rc::new(tomodataplots_param::Task::SerialSectionsMeanMax));
        }
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
    }

    /// Java `midasFixEdges(AxisID, ConstProcessSeries)`.  Run fix edges in Midas.
    pub fn midas_fix_edges(
        &'static self,
        axis_id: AxisID,
        process_series: Option<&ProcessSeriesHandle>,
    ) {
        let mut param = MidasParam::new(self, axis_id, midas_param::Mode::FixEdges);
        self.get_parameters_midas(&mut param);
        if let Some(dialog) = self.dialog.get() {
            dialog.get_parameters_midas(&mut param);
        }
        if let Err(_e) = self.process_mgr().base.midas(Arc::new(param)) {
            self.open_message(
                &format!(
                    "Unable open midas on {}.  ",
                    ConstSerialSectionsMetaData::get_stack(self.meta_data())
                ),
                "Unable to Run Process",
                Some(axis_id),
            );
            if let Some(process_series) = process_series {
                ProcessSeries::start_fail_process(process_series, axis_id);
            }
            return;
        }
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process(process_series, axis_id);
        }
    }

    /// Java `align(AxisID, ProcessResultDisplay, Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.
    pub fn align(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let process_series = ProcessSeries::new(
            self,
            axis_id,
            Some(DialogType::SerialSectionsStartup),
            Some("align"),
        );
        {
            let mut series = process_series.borrow_mut();
            series.add_process(Rc::new(Task::Xftoxg));
            series.set_last_process_task(Rc::new(Task::Align));
            series.set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        }
        ProcessSeries::start_next_process_display(&process_series, axis_id, process_result_display);
    }

    /// Java private `changeDirectory(AxisID, ConstProcessSeries)`.
    fn change_directory(&self, axis_id: AxisID, process_series: Option<&ProcessSeriesHandle>) {
        let Some(stack) = self.get_stack() else {
            if let Some(process_series) = process_series {
                ProcessSeries::start_fail_process(process_series, axis_id);
            }
            return;
        };
        let property_user_dir = utilities::java_io_file_get_parent(&stack.to_string_lossy());
        *self.base.property_user_dir.lock().unwrap() = property_user_dir.clone();
        // origUserDir = System.setProperty("user.dir", propertyUserDir)
        *self.orig_user_dir.lock().unwrap() = std::env::var("PWD").ok();
        match &property_user_dir {
            Some(property_user_dir) => unsafe { std::env::set_var("PWD", property_user_dir) },
            // System.setProperty with a null value throws NullPointerException (fixed
            // in translation: a stack with no parent leaves user.dir alone).
            None => {}
        }
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process(process_series, axis_id);
        }
    }

    /// Java private `extractpieces(UIComponent, AxisID, ProcessSeries)`.  Runs
    /// extractpieces.  Starts a failure process if it fails.  Starts the next process
    /// if it didn't spawn a process thread.
    ///
    /// The Java's last branch (a montage whose piece list exists) neither starts a
    /// process nor the next one, which ends the series silently; that is kept.
    fn extractpieces(&'static self, axis_id: AxisID, process_series: Option<&ProcessSeriesHandle>) {
        let view_type = self.get_view_type_option();
        let Some(view_type) = view_type else {
            if let Some(process_series) = process_series {
                ProcessSeries::start_fail_process(process_series, axis_id);
            }
            return;
        };
        if view_type != ViewType::Montage {
            if let Some(process_series) = process_series {
                ProcessSeries::start_next_process(process_series, axis_id);
            }
            return;
        }
        let piece_list_file = dataset_files::get_piece_list_file(self, Some(axis_id));
        if !piece_list_file.exists() {
            let stack_name = self
                .get_stack()
                .map(|stack| utilities::java_io_file_get_name(&stack.to_string_lossy()));
            let name = BaseManager::get_name(self);
            let mut param = ExtractpiecesParam::new_with_raw_stack(
                stack_name.as_deref(),
                name.as_deref(),
                Some(AxisType::SingleAxis),
                self,
                axis_id,
            );
            param
                .set_mdoc_metadata_file_enabled(self.meta_data().isextract_pl_mdoc_metadata_file());
            let thread_name = match self.process_mgr().extractpieces(
                &mut param,
                axis_id,
                process_series.map(series_ref),
            ) {
                Ok(thread_name) => thread_name,
                Err(e) => {
                    eprintln!("{e}");
                    let this = self.this_static();
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_ui_component_string_string_axis_id(
                            Some(this),
                            None,
                            &format!(
                                "Can not execute {}\n{}",
                                extractpieces_param::COMMAND_NAME,
                                e
                            ),
                            "Unable to execute command",
                            Some(axis_id),
                        )
                    });
                    if let Some(process_series) = process_series {
                        ProcessSeries::start_fail_process(process_series, axis_id);
                    }
                    return;
                }
            };
            self.set_thread_name(Some(&thread_name), Some(axis_id));
            if let Some(main_panel) = self.get_main_panel() {
                main_panel
                    .main_panel()
                    .start_progress_bar_string_axis_id_process_name(
                        Some(&format!("Running {}", extractpieces_param::COMMAND_NAME)),
                        axis_id,
                        Some(&ProcessName::EXTRACTPIECES),
                    );
            }
        }
    }

    /// Java private `createComscripts(UIComponent, AxisID, ConstProcessSeries)`.
    /// Creates blend or newst comscripts.
    fn create_comscripts(
        &'static self,
        axis_id: AxisID,
        process_series: Option<&ProcessSeriesHandle>,
    ) {
        let startup_data = self
            .startup_dialog
            .get()
            .and_then(|startup_dialog| startup_dialog.get_startup_data());
        let Some(startup_data) = startup_data else {
            if let Some(process_series) = process_series {
                ProcessSeries::start_fail_process(process_series, axis_id);
            }
            return;
        };
        let name = BaseManager::get_name(self);
        if self.get_view_type_option() == Some(ViewType::Montage) {
            // preblend
            if !file_type::CLASS
                .preblend_comscript
                .get_file(Some(self), Some(axis_id))
                .is_some_and(|file| file.exists())
            {
                if let Err(e) = utilities::copy_file_file_types(
                    &file_type::CLASS.sloppy_blend_comscript,
                    &file_type::CLASS.preblend_comscript,
                    Some(self),
                    Some(axis_id),
                    false,
                    false,
                    false,
                ) {
                    eprintln!("{e:?}");
                    let this = self.this_static();
                    let message = format!(
                        "Unable to copy {} to {}",
                        file_type::CLASS
                            .sloppy_blend_comscript
                            .get_file(Some(self), Some(axis_id))
                            .map(|file| utilities::java_io_file_get_absolute_path(
                                &file.to_string_lossy()
                            ))
                            .unwrap_or_else(|| "null".to_string()),
                        file_type::CLASS
                            .preblend_comscript
                            .get_file_name(Some(self), Some(axis_id))
                            .unwrap_or_else(|| "null".to_string())
                    );
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_ui_component_string_string_axis_id(
                            Some(this),
                            None,
                            &message,
                            "Unable to Create Comscripts",
                            Some(axis_id),
                        )
                    });
                    if let Some(process_series) = process_series {
                        ProcessSeries::start_fail_process(process_series, axis_id);
                    }
                    return;
                }
            }
            let com_script_mgr = self.com_script_mgr();
            com_script_mgr.load_preblend(axis_id);
            let mut blendmont_param =
                com_script_mgr.get_blendmont_param_from_preblend(axis_id, name.as_deref());
            startup_data.get_preblend_parameters(&mut blendmont_param, self);
            com_script_mgr.save_preblend(&blendmont_param, axis_id);
            // blend
            if !file_type::CLASS
                .blend_comscript
                .get_file(Some(self), Some(axis_id))
                .is_some_and(|file| file.exists())
            {
                BaseProcessManager::touch(
                    &file_type::CLASS
                        .blend_comscript
                        .get_file(Some(self), Some(axis_id))
                        .map(|file| {
                            utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                        })
                        .unwrap_or_default(),
                    Some(self),
                );
            }
            com_script_mgr.load_blend(axis_id);
            let mut blendmont_param =
                com_script_mgr.get_blendmont_param_from_blend(axis_id, name.as_deref());
            startup_data.get_blend_parameters(&mut blendmont_param, self);
            com_script_mgr.save_blend(&blendmont_param, axis_id);
        } else {
            // newst
            if !file_type::CLASS
                .newst_comscript
                .get_file(Some(self), Some(axis_id))
                .is_some_and(|file| file.exists())
            {
                BaseProcessManager::touch(
                    &file_type::CLASS
                        .newst_comscript
                        .get_file(Some(self), Some(axis_id))
                        .map(|file| {
                            utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                        })
                        .unwrap_or_default(),
                    Some(self),
                );
            }
            let com_script_mgr = self.com_script_mgr();
            com_script_mgr.load_newst(axis_id);
            let mut param = com_script_mgr.get_newstack_param(axis_id, name.as_deref());
            startup_data.get_parameters_newst(&mut param, self);
            com_script_mgr.save_newst(&param, axis_id);
        }
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process(process_series, axis_id);
        }
    }

    // Updates done

    /// Java private `addOutputFormat(ProcessSeries, AxisID)`.  Add setenv
    /// IMOD_OUTPUT_FORMAT to preblend, blend, or newst comscripts where it is
    /// missing.  Comscripts must exist.  If the setenv command is already there, do
    /// not change it because it reflects the settings when the dataset was created.
    fn add_output_format(&self, process_series: Option<&ProcessSeriesHandle>, axis_id: AxisID) {
        let com_script_mgr = self.com_script_mgr();
        let new_param = || {
            let mut param = SetEnvParam::new(Some(imod_output_format::ENV_VAR));
            param.set_value(Some(
                &self
                    .meta_data()
                    .base()
                    .get_image_output_format()
                    .to_string(),
            ));
            param
        };
        if self.get_view_type_option() == Some(ViewType::Montage) {
            // preblend
            com_script_mgr.load_preblend(axis_id);
            let param = com_script_mgr
                .get_set_env_param_from_preblend(axis_id, imod_output_format::ENV_VAR);
            if param.is_none() {
                // Keep the existing setting if it was set. If not then add it.
                com_script_mgr.save_preblend_set_env(
                    &new_param(),
                    axis_id,
                    imod_output_format::ENV_VAR,
                );
            }
            // blend
            com_script_mgr.load_blend(axis_id);
            let param =
                com_script_mgr.get_set_env_param_from_blend(axis_id, imod_output_format::ENV_VAR);
            if param.is_none() {
                // Keep the existing setting if it was set. If not then add it.
                com_script_mgr.save_blend_set_env(
                    &new_param(),
                    axis_id,
                    imod_output_format::ENV_VAR,
                );
            }
        } else {
            // newst
            com_script_mgr.load_newst(axis_id);
            let param =
                com_script_mgr.get_set_env_param_from_newst(axis_id, imod_output_format::ENV_VAR);
            if param.is_none() {
                // Keep the existing setting if it was set. If not then add it.
                com_script_mgr.save_newst_set_env(
                    &new_param(),
                    axis_id,
                    imod_output_format::ENV_VAR,
                );
            }
        }
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process(process_series, axis_id);
        }
    }

    /// Java private `xftoxg(ProcessSeries, ProcessResultDisplay, AxisID)`.
    fn xftoxg(
        &'static self,
        process_series: &ProcessSeriesHandle,
        process_result_display: Option<ProcessResultDisplayRef>,
        axis_id: AxisID,
    ) {
        let Some(dialog) = self.dialog.get() else {
            ProcessSeries::start_fail_process(process_series, axis_id);
            return;
        };
        let controller = *self.auto_alignment_controller.lock().unwrap();
        if let Some(controller) = controller {
            controller.copy_most_recent_xf_file("Align tab");
        }
        let mut param = XftoxgParam::new(self);
        param.set_xf_file_name(
            &file_type::CLASS
                .local_transformation_list
                .get_file_name(Some(self), Some(axis_id))
                .unwrap_or_else(|| "null".to_string()),
        );
        param.set_xg_file_name(
            &file_type::CLASS
                .global_transformation_list
                .get_file_name(Some(self), Some(axis_id))
                .unwrap_or_else(|| "null".to_string()),
        );
        dialog.get_parameters_xftoxg(&mut param);
        let thread_name = match self.process_mgr().xftoxg(
            Arc::new(param),
            process_result_display,
            axis_id,
            Some(series_ref(process_series)),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = ["Can not execute xftoxg.".to_string(), e.to_string()];
                self.open_message_array(&message, "Unable to execute process", Some(axis_id));
                ProcessSeries::start_fail_process(process_series, axis_id);
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&xftoxg_param::command_name()),
                AxisID::Only,
                Some(&ProcessName::XFTOXG),
            );
        }
    }

    /// Java private `align(ProcessSeries, ProcessResultDisplay, AxisID)`.
    fn align_process_series(
        &'static self,
        process_series: &ProcessSeriesHandle,
        process_result_display: Option<ProcessResultDisplayRef>,
        axis_id: AxisID,
    ) {
        if !self.dialog.is_some() {
            ProcessSeries::start_fail_process(process_series, axis_id);
            return;
        }
        let thread_name;
        if ConstSerialSectionsMetaData::get_view_type(self.meta_data()) == Some(ViewType::Montage) {
            let Some(param) = self.update_blend_comscript(axis_id, true) else {
                ProcessSeries::start_fail_process(process_series, axis_id);
                return;
            };
            match self.process_mgr().blend(
                Arc::new(param),
                process_result_display,
                axis_id,
                Some(series_ref(process_series)),
            ) {
                Ok(name) => thread_name = name,
                Err(e) => {
                    eprintln!("{e}");
                    let message = [
                        format!("Can not execute newst{}.com", axis_id.get_extension()),
                        e.to_string(),
                    ];
                    self.open_message_array(
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    );
                    ProcessSeries::start_fail_process(process_series, axis_id);
                    return;
                }
            }
        } else {
            let Some(param) = self.update_newst_com(axis_id, true) else {
                ProcessSeries::start_fail_process(process_series, axis_id);
                return;
            };
            match self.process_mgr().newst(
                Arc::new(param),
                process_result_display,
                axis_id,
                Some(series_ref(process_series)),
            ) {
                Ok(name) => thread_name = name,
                Err(e) => {
                    eprintln!("{e}");
                    let message = [
                        format!("Can not execute newst{}.com", axis_id.get_extension()),
                        e.to_string(),
                    ];
                    self.open_message_array(
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    );
                    ProcessSeries::start_fail_process(process_series, axis_id);
                    return;
                }
            }
        }
        self.set_thread_name(Some(&thread_name), Some(axis_id));
    }

    /// Java private `updateNewstCom(AxisID, boolean)`.
    fn update_newst_com(&'static self, axis_id: AxisID, do_validation: bool) -> Option<NewstParam> {
        let dialog = self.dialog.get()?;
        let com_script_mgr = self.com_script_mgr();
        com_script_mgr.load_newst(axis_id);
        let name = BaseManager::get_name(self);
        let mut param = com_script_mgr.get_newstack_param(axis_id, name.as_deref());
        param.set_cnverbose(true);
        param.set_command_mode(Some(newst_param::Mode::FullAlignedStack));
        param.set_output_image_file_key(Some(
            crate::imod::etomo::r#type::file_key::FileKey::clone(
                &file_type::CLASS.aligned_stack_mrc,
            ),
        ));
        param.set_transform_file(
            file_type::CLASS
                .global_transformation_list
                .get_file_name(Some(self), Some(axis_id))
                .as_deref(),
        );
        match dialog.get_parameters_newst(&mut param, do_validation) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(SetSizeToOutputInXandYError::FortranInputSyntax(except)) => {
                let error_message = [
                    "newst Parameter Syntax Error".to_string(),
                    format!("Axis: {}", axis_id.get_extension()),
                    except.to_string(),
                ];
                self.open_message_array(
                    &error_message,
                    "Newst Parameter Syntax Error",
                    Some(axis_id),
                );
                return None;
            }
            Err(SetSizeToOutputInXandYError::HeaderRead(e)) => {
                eprintln!("{e}");
                self.open_message(
                    &format!("Unable to update newst com:  {}", e),
                    "Etomo Error",
                    Some(axis_id),
                );
                return None;
            }
        }
        com_script_mgr.save_newst(&param, axis_id);
        Some(param)
    }

    /// Java `msgPreblendSucceeded()`.
    pub fn msg_preblend_succeeded(&self) {
        let Some(dialog) = self.dialog.get() else {
            return;
        };
        // Bug# 2051: Don't want to try to watch the .ecd file. Instead check
        // "use existing edge displayment file" when blendmont finishes successfully.
        dialog.set_preblend_read_in_xcorrs(true);
    }

    /// Java private `updatePreblendComscript(AxisID, boolean)`.
    fn update_preblend_comscript(
        &'static self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<BlendmontParam> {
        let com_script_mgr = self.com_script_mgr();
        com_script_mgr.load_preblend(axis_id);
        let name = BaseManager::get_name(self);
        let mut param = com_script_mgr.get_blendmont_param_from_preblend(axis_id, name.as_deref());
        param.set_mode(blendmont_param::Mode::SerialSectionPreblend);
        // Upstream bug fixed in translation (SerialSectionsManager.java:730): Java
        // dereferences a null dialog (a save before the dialog exists).
        let dialog = self.dialog.get()?;
        if !dialog.get_preblend_parameters(&mut param, do_validation) {
            return None;
        }
        param.set_blendmont_state_recreated(
            &self.state.get_invalid_edge_functions(),
            self.get_state_for_edge_functions_recreated(),
        );
        com_script_mgr.save_preblend(&param, axis_id);
        Some(param)
    }

    /// Java private `getStateForEdgeFunctionsRecreated()`.
    fn get_state_for_edge_functions_recreated(&self) -> bool {
        let Some(dialog) = self.dialog.get() else {
            return false;
        };
        (!self.state.equals_preblend_robust_fitting(
            dialog.is_preblend_robust_fitting(),
            dialog.get_preblend_robust_fitting().as_deref(),
        )) || (!self.state.equals_preblend_fix_intensity_from_edges(
            dialog.is_fix_intensity_from_edges(),
            dialog.get_fix_intensity_from_edges(),
        )) || (!self.state.equals_preblend_sum_pieces_for_gradient(
            dialog.is_sum_pieces_for_gradient(),
            dialog.get_sum_pieces_for_gradient(),
        )) || (!self.state.equals_preblend_other_sum_gradient_file(
            dialog.is_other_sum_gradient_file(),
            dialog.get_other_sum_gradient_file().as_deref(),
        ))
    }

    /// Java `getState()`.
    pub fn get_state(&self) -> &SerialSectionsState {
        &self.state
    }

    /// Java private `updateBlendComscript(AxisID, boolean)`.
    fn update_blend_comscript(
        &'static self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<BlendmontParam> {
        let com_script_mgr = self.com_script_mgr();
        com_script_mgr.load_blend(axis_id);
        let name = BaseManager::get_name(self);
        let mut param = com_script_mgr.get_blendmont_param_from_blend(axis_id, name.as_deref());
        param.set_mode(blendmont_param::Mode::SerialSectionBlend);
        param.set_image_input_file(Some(&ConstSerialSectionsMetaData::get_stack(
            self.meta_data(),
        )));
        param.set_transform_file(
            file_type::CLASS
                .global_transformation_list
                .get_file_name(Some(self), Some(axis_id))
                .as_deref(),
        );
        let mut log = BlendmontLog::new();
        let log_file = LogFile::get_instance_file(
            file_type::CLASS
                .preblend_log
                .get_file(Some(self), Some(axis_id))
                .as_deref(),
            Some(self.get_emergency_monitor(Some(axis_id))),
        );
        match log_file {
            Ok(log_file) => {
                if log.find_unaligned_starting_xand_y(Some(&log_file)) {
                    let unaligned = log.get_unaligned_starting_xand_y();
                    let unaligned: Option<Vec<Option<String>>> =
                        unaligned.map(|array| array.into_iter().map(Some).collect());
                    param.set_unaligned_starting_xand_y(unaligned.as_deref());
                }
            }
            Err(e) => eprintln!("{e:?}"),
        }
        // Upstream bug fixed in translation (SerialSectionsManager.java:772): Java
        // dereferences a null dialog (a save before the dialog exists).
        let dialog = self.dialog.get()?;
        if !dialog.get_blend_parameters(&mut param, do_validation) {
            return None;
        }
        param.set_blendmont_state(&self.state.get_invalid_edge_functions());
        com_script_mgr.save_blend(&param, axis_id);
        Some(param)
    }

    /// Java private `copyDistortionFieldFile(ConstProcessSeries, UIComponent, AxisID)`.
    /// Copies the distortion field file to the directory containing the stack.  Always
    /// tries to start the next process.
    fn copy_distortion_field_file(
        &'static self,
        process_series: Option<&ProcessSeriesHandle>,
        axis_id: AxisID,
    ) {
        let distortion_field = self.get_distortion_field();
        if let Some(distortion_field) = distortion_field {
            let stack = self.get_stack();
            if let Some(stack) = stack {
                let distortion_field_name = distortion_field.to_string_lossy().into_owned();
                let destination = PathBuf::from(utilities::java_io_file_new(
                    &utilities::java_io_file_get_parent(&stack.to_string_lossy())
                        .unwrap_or_else(|| "null".to_string()),
                    &utilities::java_io_file_get_name(&distortion_field_name),
                ));
                if utilities::copy_file(
                    Some(self),
                    Some(axis_id),
                    Some(&distortion_field),
                    Some(&destination),
                    false,
                    false,
                    false,
                )
                .is_err()
                {
                    let this = self.this_static();
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_ui_component_string_string_axis_id(
                            Some(this),
                            None,
                            &format!(
                                "Unable to copy {}.  Please copy this file by hand.",
                                utilities::java_io_file_get_absolute_path(&distortion_field_name)
                            ),
                            "Unable to Copy File",
                            Some(axis_id),
                        )
                    });
                    if let Some(process_series) = process_series {
                        ProcessSeries::start_fail_process(process_series, axis_id);
                    }
                    return;
                }
            }
        }
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process(process_series, axis_id);
        }
    }

    /// Java private `doneStartupDialog(ConstProcessSeries, AxisID)`.  Attempts to
    /// close the startup dialog.
    fn done_startup_dialog(&self, process_series: Option<&ProcessSeriesHandle>, axis_id: AxisID) {
        // Upstream bug fixed in translation (SerialSectionsManager.java:820): Java
        // dereferences a null startupDialog; the dialog is set whenever this task runs.
        if let Some(startup_dialog) = self.startup_dialog.get() {
            startup_dialog.done();
        }
        if let Some(log_window) = self.get_log_window() {
            log_window.show();
        }
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process(process_series, axis_id);
        }
    }

    /// Java private `resetStartupState(ConstProcessSeries, AxisID)`.  Attempts to
    /// reset the saved state of dialogType dialog.
    fn reset_startup_state(&self, process_series: Option<&ProcessSeriesHandle>, axis_id: AxisID) {
        let orig_user_dir = self.orig_user_dir.lock().unwrap().take();
        if let Some(orig_user_dir) = orig_user_dir {
            *self.base.property_user_dir.lock().unwrap() = Some(orig_user_dir.clone());
            // System.setProperty("user.dir", propertyUserDir)
            unsafe { std::env::set_var("PWD", &orig_user_dir) };
        }
        if let Some(startup_dialog) = self.startup_dialog.get() {
            startup_dialog.reset_saved_state();
        }
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process(process_series, axis_id);
        }
    }

    /// The three `catch` arms the `imod*` members share (Java writes them out in
    /// each).
    fn imod_error(&self, result: Result<(), ImodManagerException>, axis_id: AxisID) {
        match result {
            Ok(()) => {}
            Err(except @ ImodManagerException::AxisType(_)) => {
                eprintln!("{except}");
                self.open_message(&except.to_string(), "AxisType problem", Some(axis_id));
            }
            Err(except @ ImodManagerException::SystemProcess(_)) => {
                eprintln!("{except}");
                self.open_message(
                    &except.to_string(),
                    "Can't open 3dmod with the tomogram",
                    Some(axis_id),
                );
            }
            Err(e @ ImodManagerException::Io(_)) => {
                eprintln!("{e}");
                self.open_message(&e.to_string(), "IO Exception", Some(axis_id));
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }

    /// Java `imodRaw(AxisID, Run3dmodMenuOptions)`.
    pub fn imod_raw(&'static self, axis_id: AxisID, menu_options: Option<Run3dmodMenuOptions>) {
        let imod_manager = self.get_imod_manager();
        let result = (|| -> Result<(), ImodManagerException> {
            if file_type::CLASS
                .piece_list
                .exists(Some(self), Some(axis_id))
            {
                imod_manager.set_piece_list_file_name_string_axis_id_string(
                    imod_manager::RAW_STACK_KEY,
                    Some(axis_id),
                    file_type::CLASS
                        .piece_list
                        .get_file_name(Some(self), Some(axis_id))
                        .as_deref(),
                )?;
            }
            let file = PathBuf::from(utilities::java_io_file_new(
                &self
                    .property_user_dir()
                    .unwrap_or_else(|| "null".to_string()),
                &ConstSerialSectionsMetaData::get_stack(self.meta_data()),
            ));
            imod_manager.open_string_axis_id_file_run3dmod_menu_options(
                imod_manager::RAW_STACK_KEY,
                Some(axis_id),
                Some(&file),
                menu_options,
            )
        })();
        self.imod_error(result, axis_id);
    }

    /// Java `imodPreblend(AxisID, Run3dmodMenuOptions)`.
    pub fn imod_preblend(
        &'static self,
        axis_id: AxisID,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let result = self
            .get_imod_manager()
            .open_string_axis_id_run3dmod_menu_options(
                imod_manager::PREBLEND_KEY,
                Some(axis_id),
                menu_options,
            );
        self.imod_error(result, axis_id);
    }

    /// Java `imodPrealign(AxisID, Run3dmodMenuOptions)`.
    pub fn imod_prealign(
        &'static self,
        axis_id: AxisID,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if ConstSerialSectionsMetaData::get_view_type(self.meta_data()) == Some(ViewType::Montage) {
            self.imod_preblend(axis_id, menu_options);
        } else {
            self.imod_raw(axis_id, menu_options);
        }
    }

    /// Java `imodAlign(AxisID, Run3dmodMenuOptions)`.
    pub fn imod_align(&'static self, axis_id: AxisID, menu_options: Option<Run3dmodMenuOptions>) {
        let result = self
            .get_imod_manager()
            .open_string_axis_id_run3dmod_menu_options(
                imod_manager::ALIGNED_STACK_KEY,
                Some(axis_id),
                menu_options,
            );
        self.imod_error(result, axis_id);
    }

    /// Java private `saveSerialSectionsDialog(boolean)`.
    fn save_serial_sections_dialog(&'static self, _for_run: bool) -> bool {
        let Some(dialog) = self.dialog.get() else {
            return false;
        };
        if self.base.param_file.lock().unwrap().is_none() && !self.set_param_file() {
            return false;
        }
        dialog.get_parameters_meta_data(self.meta_data(), false);
        if self.get_view_type_option() == Some(ViewType::Montage) {
            self.update_preblend_comscript(AXIS_ID, false);
            self.update_blend_comscript(AXIS_ID, false);
        } else {
            self.update_newst_com(AXIS_ID, false);
        }
        self.save_storables(Some(AXIS_ID));
        true
    }

    /// Java private `getDistortionField()`.  Attempts to get the distortion field
    /// file.  Returns null if unable to get the file.
    fn get_distortion_field(&self) -> Option<PathBuf> {
        if self.loaded_param_file() {
            return Some(PathBuf::from(self.meta_data().get_distortion_field()));
        }
        if let Some(startup_dialog) = self.startup_dialog.get() {
            return startup_dialog.get_distortion_field();
        }
        None
    }

    /// Java `getStack()`.  Attempts to get the stack file.  Returns null if unable to
    /// get the file.
    pub fn get_stack(&self) -> Option<PathBuf> {
        if self.loaded_param_file() {
            return Some(PathBuf::from(utilities::java_io_file_new(
                &self
                    .property_user_dir()
                    .unwrap_or_else(|| "null".to_string()),
                &ConstSerialSectionsMetaData::get_stack(self.meta_data()),
            )));
        }
        if let Some(startup_dialog) = self.startup_dialog.get() {
            return startup_dialog.get_stack();
        }
        None
    }

    /// Java `getViewType()` with its null return.  Attempts to get the view type.
    /// Returns null if unable to get it.
    pub fn get_view_type_option(&self) -> Option<ViewType> {
        if self.loaded_param_file() {
            return ConstSerialSectionsMetaData::get_view_type(self.meta_data());
        }
        if let Some(startup_dialog) = self.startup_dialog.get() {
            return startup_dialog.get_view_type();
        }
        None
    }

    /// Java `getMetaData()`, declared `ConstSerialSectionsMetaData`.
    pub fn get_meta_data(&'static self) -> &'static SerialSectionsMetaData {
        self.meta_data()
    }

    /// Java private `setSerialSectionsDialogParameters()`.
    fn set_serial_sections_dialog_parameters(&'static self) {
        let Some(dialog) = self.dialog.get() else {
            return;
        };
        if self.loaded_param_file()
            && self.base.param_file.lock().unwrap().is_some()
            && self.meta_data().is_valid()
        {
            dialog.set_parameters_meta_data(self.meta_data());
            let com_script_mgr = self.com_script_mgr();
            let name = BaseManager::get_name(self);
            if self.get_view_type_option() == Some(ViewType::Montage) {
                com_script_mgr.load_preblend(AXIS_ID);
                let param =
                    com_script_mgr.get_blendmont_param_from_preblend(AXIS_ID, name.as_deref());
                dialog.set_preblend_parameters(&param);
                com_script_mgr.load_blend(AXIS_ID);
                let param = com_script_mgr.get_blendmont_param_from_blend(AXIS_ID, name.as_deref());
                dialog.set_blend_parameters(&param);
            } else {
                com_script_mgr.load_newst(AXIS_ID);
                let param = com_script_mgr.get_newstack_param(AXIS_ID, name.as_deref());
                dialog.set_parameters_newst(&param);
            }
        }
    }

    /// Java `getParameters(MidasParam)`.
    pub fn get_parameters_midas(&self, param: &mut MidasParam) {
        param.set_input_file_name(Some(&ConstSerialSectionsMetaData::get_stack(
            self.meta_data(),
        )));
    }

    /// Java `getAutoAlignmentParameters(MidasParam, AxisID)`.
    pub fn get_auto_alignment_parameters_midas(
        &'static self,
        param: &mut MidasParam,
        axis_id: AxisID,
    ) {
        if self.get_view_type_option() == Some(ViewType::Montage) {
            param.set_input_file_name(
                file_type::CLASS
                    .preblend_output_mrc
                    .get_file_name(Some(self), Some(axis_id))
                    .as_deref(),
            );
        } else {
            param.set_input_file_name(Some(&ConstSerialSectionsMetaData::get_stack(
                self.meta_data(),
            )));
        }
    }

    /// Java `getAutoAlignmentParameters(XfalignParam, AxisID)`.
    pub fn get_auto_alignment_parameters_xfalign(
        &'static self,
        param: &mut XfalignParam,
        axis_id: AxisID,
    ) {
        if self.get_view_type_option() == Some(ViewType::Montage) {
            param.set_input_file_name(
                file_type::CLASS
                    .preblend_output_mrc
                    .get_file_name(Some(self), Some(axis_id))
                    .as_deref(),
            );
        } else {
            param.set_input_file_name(Some(&ConstSerialSectionsMetaData::get_stack(
                self.meta_data(),
            )));
        }
    }
}

impl BaseManager for SerialSectionsManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    /// Java package-private override `initializeUIParameters(String, AxisID)`.
    fn initialize_ui_parameters_from_name(
        &'static self,
        param_file_name: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        match param_file_name {
            Some(param_file_name) if !param_file_name.is_empty() => {
                self.initialize_ui_parameters(Some(Path::new(param_file_name)), axis_id, false)
            }
            _ => self.initialize_ui_parameters(None, axis_id, false),
        }
        if self.loaded_param_file()
            && let Some(property_user_dir) = self.property_user_dir()
        {
            // System.setProperty("user.dir", propertyUserDir)
            unsafe { std::env::set_var("PWD", &property_user_dir) };
        }
    }

    /// Java `isStartupPopupOpen()`.
    fn is_startup_popup_open(&self) -> bool {
        !self.loaded_param_file()
    }

    /// Java override `startNextProcess(...)`.  Returns true if the process is
    /// recognized.
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
            process_result_display.clone(),
            process_series,
            dialog_type,
            display,
        ) {
            return true;
        }
        if process.equals_task(&Task::ChangeDirectory) {
            self.change_directory(axis_id, Some(process_series));
            return true;
        }
        if process.equals_task(&Task::ExtractPieces) {
            self.extractpieces(axis_id, Some(process_series));
            return true;
        }
        if process.equals_task(&Task::CreateComscripts) {
            self.create_comscripts(axis_id, Some(process_series));
            return true;
        }
        if process.equals_task(&Task::CopyDistortionFieldFile) {
            self.copy_distortion_field_file(Some(process_series), axis_id);
            return true;
        }
        if process.equals_task(&Task::DoneStartupDialog) {
            self.done_startup_dialog(Some(process_series), axis_id);
            return true;
        }
        if process.equals_task(&Task::ResetStartupState) {
            self.reset_startup_state(Some(process_series), axis_id);
            return true;
        }
        if process.equals_task(&Task::Xftoxg) {
            self.xftoxg(process_series, process_result_display, axis_id);
            return true;
        }
        if process.equals_task(&Task::Align) {
            self.align_process_series(process_series, process_result_display, axis_id);
            return true;
        }
        if process.equals_task(&Task::AddOutputFormat) {
            self.add_output_format(Some(process_series), axis_id);
            return true;
        }
        false
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

    /// Java `save() throws LogFileException, IOException, LockException`.
    fn save(&'static self) -> Result<bool, LogFileError> {
        self.save_super()?;
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.done();
        }
        self.save_serial_sections_dialog(false);
        Ok(true)
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        if self.loaded_param_file() {
            return BaseMetaData::get_name(self.meta_data());
        }
        let mut name: Option<String> = None;
        if let Some(startup_dialog) = self.startup_dialog.get() {
            name = startup_dialog.get_root_name();
        }
        if name.is_some() {
            return name;
        }
        // The constructor body assigns metaData after `super()`, which already asks
        // for the name (the log window title); Java's field is null then and
        // `metaData.getName()` throws NullPointerException.  The new-dataset title is
        // the name until the meta data exists.
        match self.meta_data.get() {
            Some(meta_data) => BaseMetaData::get_name(meta_data),
            None => {
                Some(crate::imod::etomo::r#type::serial_sections_meta_data::NEW_TITLE.to_string())
            }
        }
    }

    /// Java `getViewType()`.  A null view type (no data and no startup dialog) reads
    /// as the default here; `get_view_type_option` keeps the null.
    fn get_view_type(&self) -> ViewType {
        self.get_view_type_option().unwrap_or(ViewType::DEFAULT)
    }

    /// Java package-private `createComScriptManager()`.
    fn create_com_script_manager(&self) {
        let this = self.this_static();
        let com_script_mgr: &'static SerialSectionsComScriptManager =
            Box::leak(Box::new(SerialSectionsComScriptManager::new(this)));
        COM_SCRIPT_ROOTS.lock().unwrap().push(com_script_mgr);
        let _ = self.com_script_mgr.set(com_script_mgr);
    }

    /// Java package-private `createMainPanel()`.
    fn create_main_panel(&self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            let this = self.this_static();
            self.main_panel
                .set(Some(MainSerialSectionsPanel::new(this)));
        }
    }

    /// Java `getBaseMetaData()`.
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        self.meta_data
            .get()
            .map(|meta_data| meta_data as &dyn BaseMetaData)
    }

    /// Java `getInterfaceType()`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::SerialSections)
    }

    /// Java `getMainPanel()`.
    fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.main_panel
            .get()
            .map(|main_panel| main_panel as Rc<dyn MainPanelVirtual>)
    }

    /// Java `getProcessManager()`.
    fn get_process_manager(&self) -> Option<&'static BaseProcessManager> {
        self.process_mgr.get().map(|process_mgr| &process_mgr.base)
    }

    /// Java package-private `getAutoAlignmentMetaData()`.
    fn get_auto_alignment_meta_data(&self) -> Option<&'static Mutex<AutoAlignmentMetaData>> {
        let this = self.this_static();
        Some(ConstSerialSectionsMetaData::get_auto_alignment_meta_data(
            this.meta_data(),
        ))
    }

    /// Java `updateMetaData(DialogType, AxisID, boolean)`.
    fn update_meta_data(
        &self,
        _dialog_type: Option<DialogType>,
        _axis_id: Option<AxisID>,
        do_validation: bool,
    ) -> bool {
        // Upstream bug fixed in translation (SerialSectionsManager.java:1108): Java
        // dereferences a null dialog (still on the startup dialog); nothing is read
        // then and the update fails.
        let Some(dialog) = self.dialog.get() else {
            return false;
        };
        if !dialog.get_parameters_meta_data(self.meta_data(), do_validation) {
            return false;
        }
        true
    }

    /// Java package-private `getStorables(int)`.
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Option<&'static dyn Storable>>> {
        let this = self.this_static();
        let mut storables: Vec<Option<&'static dyn Storable>> =
            vec![None; (2 + offset).max(0) as usize];
        let mut index = offset.max(0) as usize;
        storables[index] = this
            .meta_data
            .get()
            .map(|meta_data| meta_data as &dyn Storable);
        index += 1;
        storables[index] = Some(&this.state as &dyn Storable);
        Some(storables)
    }

    /// Java `isValid()` (`valid` is for handling failure before the manager key is set
    /// in EtomoDirector).
    fn is_valid(&self) -> bool {
        *self.valid.lock().unwrap()
    }
}

/// Java public static final nested `Task implements TaskInterface`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    /// `CHANGE_DIRECTORY = new Task("change directory")`.
    ChangeDirectory,
    /// `EXTRACT_PIECES = new Task("extract pieces")`.
    ExtractPieces,
    /// `CREATE_COMSCRIPTS = new Task("create comscripts")`.
    CreateComscripts,
    /// `COPY_DISTORTION_FIELD_FILE = new Task("copy distortion field file")`.
    CopyDistortionFieldFile,
    /// `DONE_STARTUP_DIALOG = new Task(true, "done startup dialog")`.
    DoneStartupDialog,
    /// `RESET_STARTUP_STATE = new Task(true, "reset startup state")`.
    ResetStartupState,
    /// `XFTOXG = new Task("xftoxg")`.
    Xftoxg,
    /// `ALIGN = new Task("align")`.
    Align,
    /// `ADD_OUTPUT_FORMAT = new Task("add output format")`.
    AddOutputFormat,
}

impl TaskInterface for Task {
    /// Java `getDescr()`.
    fn get_descr(&self) -> Option<String> {
        Some(
            match self {
                Self::ChangeDirectory => "change directory",
                Self::ExtractPieces => "extract pieces",
                Self::CreateComscripts => "create comscripts",
                Self::CopyDistortionFieldFile => "copy distortion field file",
                Self::DoneStartupDialog => "done startup dialog",
                Self::ResetStartupState => "reset startup state",
                Self::Xftoxg => "xftoxg",
                Self::Align => "align",
                Self::AddOutputFormat => "add output format",
            }
            .to_string(),
        )
    }

    /// Java `okToDrop()`.
    fn ok_to_drop(&self) -> bool {
        matches!(self, Self::DoneStartupDialog | Self::ResetStartupState)
    }
}
