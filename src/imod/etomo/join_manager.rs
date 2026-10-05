//! `IMOD/Etomo/src/etomo/JoinManager.java`.
//!
//! The manager of the Join interface: it owns the join dialog, the join comscripts and
//! the join process manager, and runs `makejoincom`, `startjoin`, `xfjointomo`,
//! `xfmodel`, `xftoxg`, `finishjoin` and the refine chain.
//!
//! **Representation.**  `JoinManager extends BaseManager`, so - as
//! `etomo/base_manager.rs` sets out - the superclass state is the `base` field and the
//! superclass methods are the `BaseManager` trait, which this struct implements; the
//! `@Override` members are the trait implementations and everything else is an inherent
//! method.  Java's managers are created by `EtomoDirector` and never collected, so the
//! constructor leaks its allocation and hands back `&'static JoinManager`.
//!
//! **Threads.**  The manager is shared with process threads (`Send + Sync`); the main
//! panel and the dialog are event dispatch thread objects, held in `EdtCell`s and
//! reached only on that thread (the process manager posts its callbacks there).
//!
//! **Headless.**  The constructor builds neither the main panel nor the dialog when
//! etomo runs headless, and Java then dereferences the null fields in every member
//! that reaches them (NullPointerException).  Those members skip the missing panel or
//! dialog here ("fixed in translation", `BUGS.md`).

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

use crate::imod::etomo::auto_alignment_controller::AutoAlignmentController;
use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::comscript::clip_param::ClipParam;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::finishjoin_param::{self, FinishjoinParam};
use crate::imod::etomo::comscript::join_comscript_manager::JoinComscriptManager;
use crate::imod::etomo::comscript::joinwarp2model_param::{self, Joinwarp2modelParam};
use crate::imod::etomo::comscript::makejoincom_param::{self, MakejoincomParam};
use crate::imod::etomo::comscript::midas_param::MidasParam;
use crate::imod::etomo::comscript::process_details::ProcessDetails;
use crate::imod::etomo::comscript::remapmodel_param::{self, RemapmodelParam};
use crate::imod::etomo::comscript::start_join_param::StartJoinParam;
use crate::imod::etomo::comscript::xfalign_param::XfalignParam;
use crate::imod::etomo::comscript::xfjointomo_param::XfjointomoParam;
use crate::imod::etomo::comscript::xfmodel_param::{self, XfmodelParam};
use crate::imod::etomo::comscript::xftoxg_param::{self, XftoxgParam};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::base_imod_manager::ImodManagerException;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::{self, Run3dmodMenuOptions};
use crate::imod::etomo::process::join_process_manager::JoinProcessManager;
use crate::imod::etomo::process::process_interface::ProcessResultDisplayRef;
use crate::imod::etomo::process::process_messages::{MessageType, ProcessMessages};
use crate::imod::etomo::process_series::{Process, ProcessSeries, ProcessSeriesHandle};
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::auto_alignment_meta_data::AutoAlignmentMetaData;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::base_state::BaseState;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::const_join_meta_data::ConstJoinMetaData;
use crate::imod::etomo::r#type::const_join_state::ConstJoinState;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::join_meta_data::JoinMetaData;
use crate::imod::etomo::r#type::join_screen_state::JoinScreenState;
use crate::imod::etomo::r#type::join_state::{self, JoinState};
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::slicer_angles::SlicerAngles;
use crate::imod::etomo::ui::UiComponent;
use crate::imod::etomo::ui::auto_alignment_display::AutoAlignmentDisplay;
use crate::imod::etomo::ui::swing::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::ui::swing::join_dialog::{self, JoinDialog};
use crate::imod::etomo::ui::swing::main_join_panel::MainJoinPanel;
use crate::imod::etomo::ui::swing::main_panel::MainPanelVirtual;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::event_queue::{EdtCell, EdtRef};
use crate::imod::etomo::util::utilities;

/// Java `public final class JoinManager extends BaseManager`.
pub struct JoinManager {
    /// Java superclass `BaseManager` state.
    base: BaseManagerBase,
    /// Java private `joinDialog`, initially null.
    join_dialog: EdtCell<Rc<JoinDialog>>,
    /// Java private `autoAlignmentController`, initially null.
    auto_alignment_controller: Mutex<Option<&'static AutoAlignmentController>>,
    /// Java private `mainPanel` (null in headless mode).
    main_panel: EdtCell<Rc<MainJoinPanel>>,
    /// Java private final `metaData`, assigned in the constructor body after `super()`.
    meta_data: OnceLock<JoinMetaData>,
    /// Java private `processMgr`.
    process_mgr: OnceLock<&'static JoinProcessManager>,
    /// Java private `state`, set by `createState`.
    state: OnceLock<&'static JoinState>,
    /// Java private `startJoinParam`, initially null.
    start_join_param: Mutex<Option<StartJoinParam>>,
    /// Java private final `screenState = new JoinScreenState(AxisID.ONLY,
    /// AxisType.SINGLE_AXIS)`.
    screen_state: JoinScreenState,
    /// Java private `debug`, initially false.
    debug: Mutex<bool>,
    /// Java private `comScriptMgr`.
    com_script_mgr: OnceLock<&'static JoinComscriptManager>,
}

/// Owns every `JoinManager` this module builds.  Java's owner is the collector, by
/// way of `EtomoDirector.managerList`, which keeps each manager for the run;
/// the translation hands out `&'static Self`, so without a root here the
/// allocation is unreachable the moment the constructor returns.
static INSTANCES: Mutex<Vec<&'static JoinManager>> = Mutex::new(Vec::new());

/// Owns every `JoinState` `createState` builds, for the same reason as `INSTANCES`.
static STATE_ROOTS: Mutex<Vec<&'static JoinState>> = Mutex::new(Vec::new());

impl JoinManager {
    /// Java package-private `JoinManager(String, AxisID)`.
    pub fn new(param_file_name: Option<&str>, axis_id: Option<AxisID>) -> &'static JoinManager {
        let instance: &'static JoinManager = Box::leak(Box::new(JoinManager {
            base: BaseManagerBase::initial(),
            join_dialog: EdtCell::new(),
            auto_alignment_controller: Mutex::new(None),
            main_panel: EdtCell::new(),
            meta_data: OnceLock::new(),
            process_mgr: OnceLock::new(),
            state: OnceLock::new(),
            start_join_param: Mutex::new(None),
            screen_state: JoinScreenState::new(AxisID::Only, AxisType::SingleAxis),
            debug: Mutex::new(false),
            com_script_mgr: OnceLock::new(),
        }));
        INSTANCES.lock().unwrap().push(instance);
        // Java `super()`.
        instance.base_manager();
        let _ = instance.meta_data.set(JoinMetaData::new(
            instance,
            instance.get_log_properties(),
            param_file_name.is_none_or(str::is_empty),
        ));
        instance.create_state();
        let _ = instance
            .process_mgr
            .set(JoinProcessManager::new(instance, instance.get_state()));
        instance.initialize_ui_parameters_from_name(param_file_name, axis_id);
        // Upstream bug fixed in translation (JoinManager.java:98): Java's
        // `paramFileName.equals("")` throws NullPointerException for a null name; a
        // null name is an empty one here.
        if !param_file_name.unwrap_or("").is_empty()
            && *instance.base.loaded_param_file.lock().unwrap()
        {
            instance
                .get_imod_manager()
                .set_meta_data_join_meta_data(instance.meta_data());
            instance.set_main_panel_status_bar_text();
        }
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            instance.open_join_dialog();
            instance.set_mode();
        }
        let com_script_mgr: &'static JoinComscriptManager =
            Box::leak(Box::new(JoinComscriptManager::new(instance)));
        let _ = instance.com_script_mgr.set(com_script_mgr);
        instance
    }

    /// The constructed manager at its final address (Java `this` inside the
    /// overrides, which take `&self`).
    fn this_static(&self) -> &'static JoinManager {
        INSTANCES
            .lock()
            .unwrap()
            .iter()
            .copied()
            .find(|manager| std::ptr::eq(*manager, self))
            .expect("constructed JoinManager")
    }

    /// Java field read `metaData`.
    fn meta_data(&self) -> &JoinMetaData {
        self.meta_data
            .get()
            .expect("metaData is assigned by the constructor")
    }

    /// Java field read `processMgr`.
    fn process_mgr(&self) -> &'static JoinProcessManager {
        self.process_mgr.get().expect("processMgr")
    }

    /// Java field read `comScriptMgr`.
    fn com_script_mgr(&self) -> &'static JoinComscriptManager {
        self.com_script_mgr.get().expect("comScriptMgr")
    }

    /// Java field read `propertyUserDir`.
    fn property_user_dir(&self) -> Option<String> {
        self.base.property_user_dir.lock().unwrap().clone()
    }

    /// Java `mainPanel.setStatusBarText(paramFile, metaData, logWindow)`, which four
    /// members write out in full.
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

    /// Java `openJoinDialog()`.  Open the join dialog.
    pub fn open_join_dialog(&'static self) -> bool {
        let mut retval = true;
        self.open_processing_panel();
        if !self.join_dialog.is_some() {
            let join_dialog = if *self.base.loaded_param_file.lock().unwrap() {
                JoinDialog::get_instance_working_dir(
                    self,
                    self.property_user_dir().as_deref(),
                    self.meta_data(),
                    self.get_state(),
                )
            } else {
                JoinDialog::get_instance_meta_data(self, self.meta_data(), self.get_state())
            };
            self.join_dialog.set(Some(join_dialog.clone()));
            let auto_alignment_controller = AutoAlignmentController::new(
                self,
                join_dialog.clone(),
                self.get_imod_manager(),
                None,
            );
            *self.auto_alignment_controller.lock().unwrap() = Some(auto_alignment_controller);
            join_dialog.set_auto_alignment_controller(auto_alignment_controller);
        }
        if *self.base.loaded_param_file.lock().unwrap() {
            let controller = *self.auto_alignment_controller.lock().unwrap();
            if let Some(controller) = controller
                && let Err(LogFileError::Lock(_)) = controller.create_empty_xf_file()
            {
                retval = false;
            }
        }
        if let (Some(main_panel), Some(join_dialog)) =
            (self.main_panel.get(), self.join_dialog.get())
        {
            main_panel.show_process(&join_dialog.get_container(), AxisID::Only);
        }
        let action_message =
            utilities::prepare_dialog_action_message(Some(DialogType::Join), AxisID::Only, None);
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
        retval
    }

    /// Java `getParameters(MidasParam, AxisID)`.
    pub fn get_parameters_midas(&'static self, param: &mut MidasParam, axis_id: AxisID) {
        param.set_input_file_name(
            file_type::CLASS
                .join_sample
                .get_file_name(Some(self), Some(axis_id))
                .as_deref(),
        );
        param.set_section_table_row_data(self.meta_data().get_section_table_data());
    }

    /// Java `getParameters(XfalignParam, AxisID)`.
    pub fn get_parameters_xfalign(&'static self, param: &mut XfalignParam, axis_id: AxisID) {
        param.set_input_file_name(
            file_type::CLASS
                .join_sample_averages
                .get_file_name(Some(self), Some(axis_id))
                .as_deref(),
        );
    }

    /// Java private `doneJoinDialog()`.
    fn done_join_dialog(&'static self) -> bool {
        let Some(join_dialog) = self.join_dialog.get() else {
            return false;
        };
        let working_dir = join_dialog.get_working_dir_name();
        let loaded_param_file = *self.base.loaded_param_file.lock().unwrap();
        if !loaded_param_file && let Some(working_dir) = working_dir.as_deref().filter(|dir| {
            !crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace(
                dir,
            )
        }) {
            if working_dir.ends_with(' ') {
                self.open_message(
                    &format!(
                        "The directory, {}, cannot be used because it ends with a space.",
                        working_dir
                    ),
                    "Unusable Directory Name",
                    Some(AxisID::Only),
                );
                return false;
            }
            *self.base.property_user_dir.lock().unwrap() = Some(working_dir.to_string());
        }
        let Some(root_name) = join_dialog.get_root_name() else {
            return false;
        };
        if !*self.base.loaded_param_file.lock().unwrap()
            && !crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace(
                &root_name,
            )
        {
            let param_file =
                Path::new(&self.property_user_dir().unwrap_or_default()).join(format!(
                    "{}{}",
                    root_name,
                    self.meta_data().get_file_extension().unwrap_or_default()
                ));
            *self.base.param_file.lock().unwrap() = Some(param_file.clone());
            if !param_file.exists() {
                self.process_mgr()
                    .create_new_file(&utilities::java_io_file_get_absolute_path(
                        &param_file.to_string_lossy(),
                    ));
            }
            self.initialize_ui_parameters(Some(&param_file), Some(AxisID::Only), false);
            if *self.base.loaded_param_file.lock().unwrap() {
                self.get_imod_manager()
                    .set_meta_data_join_meta_data(self.meta_data());
                self.set_main_panel_status_bar_text();
            }
        }
        join_dialog.get_meta_data(self.meta_data(), false);
        join_dialog.get_screen_state(&self.screen_state);
        self.get_state().set_done_mode(join_dialog.get_mode());
        self.save_storables(Some(AxisID::Only));
        true
    }

    /// Java `getScreenState()`.
    pub fn get_screen_state(&'static self) -> &'static JoinScreenState {
        &self.screen_state
    }

    /// Java `imodOpen(String, int, Run3dmodMenuOptions)`.  Open 3dmod with binning.
    pub fn imod_open_binning(
        &'static self,
        imod_key: Option<&str>,
        binning: i32,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let key = imod_key.unwrap_or_default();
        let imod_manager = self.get_imod_manager();
        let result = imod_manager
            .set_binning_xy_string_int(key, binning)
            .and_then(|()| imod_manager.open_string_run3dmod_menu_options(key, menu_options));
        self.imod_open_error(imod_key, result);
    }

    /// The three `catch` arms the `imodOpen` overloads share (Java writes them out in
    /// each).
    fn imod_open_error(&self, imod_key: Option<&str>, result: Result<(), ImodManagerException>) {
        match result {
            Ok(()) => {}
            Err(except @ ImodManagerException::AxisType(_)) => {
                eprintln!("{except}");
                self.open_message(&except.to_string(), "AxisType problem", Some(AxisID::Only));
            }
            Err(except @ ImodManagerException::SystemProcess(_)) => {
                eprintln!("{except}");
                self.open_message(
                    &except.to_string(),
                    &format!("Can't open {} in 3dmod ", imod_key.unwrap_or("null")),
                    Some(AxisID::Only),
                );
            }
            Err(e @ ImodManagerException::Io(_)) => {
                eprintln!("{e}");
                self.open_message(&e.to_string(), "IO Exception", Some(AxisID::Only));
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }

    /// Java `imodOpen(String, int, String, Run3dmodMenuOptions)`.
    pub fn imod_open_binning_model(
        &'static self,
        imod_key: Option<&str>,
        binning: i32,
        model_name: Option<&str>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if *self.debug.lock().unwrap() {
            eprintln!("imodOpen:modelName={}", model_name.unwrap_or("null"));
        }
        let key = imod_key.unwrap_or_default();
        let imod_manager = self.get_imod_manager();
        let result = imod_manager
            .set_binning_xy_string_int(key, binning)
            .and_then(|()| {
                imod_manager.open_string_string_run3dmod_menu_options(key, model_name, menu_options)
            });
        self.imod_open_error(imod_key, result);
    }

    /// Java `imodOpen(ProcessSeries, String)`.
    pub fn imod_open_process_series(
        &'static self,
        process_series: Option<&ProcessSeriesHandle>,
        imod_key: Option<&str>,
    ) {
        let result = self
            .get_imod_manager()
            .open_string_axis_id(imod_key.unwrap_or_default(), Some(AxisID::Only));
        self.imod_open_error(imod_key, result);
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process(process_series, AxisID::Only);
        }
    }

    /// Java `isImodOpen(String)`.
    pub fn is_imod_open(&'static self, imod_key: Option<&str>) -> bool {
        match self
            .get_imod_manager()
            .is_open_string(imod_key.unwrap_or_default())
        {
            Ok(open) => open,
            Err(e @ ImodManagerException::AxisType(_)) => {
                self.open_message(&e.to_string(), "AxisType problem", Some(AxisID::Only));
                false
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(e) => {
                eprintln!("{e}");
                false
            }
        }
    }

    /// Java `imodRemove(String, int)`.  Remove a specific 3dmod.
    pub fn imod_remove(&'static self, imod_key: Option<&str>, imod_index: i32) {
        if imod_index == -1 {
            return;
        }
        match self
            .get_imod_manager()
            .delete(imod_key.unwrap_or_default(), imod_index)
        {
            Ok(()) => {}
            Err(except @ ImodManagerException::AxisType(_)) => {
                eprintln!("{except}");
                self.open_message(&except.to_string(), "AxisType problem", Some(AxisID::Only));
            }
            Err(e @ ImodManagerException::Io(_)) => {
                eprintln!("{e}");
                self.open_message(&e.to_string(), "IO Exception", Some(AxisID::Only));
            }
            Err(e @ ImodManagerException::SystemProcess(_)) => {
                eprintln!("{e}");
                self.open_message(
                    &e.to_string(),
                    "System Process Exception",
                    Some(AxisID::Only),
                );
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }

    /// Java `imodGetSlicerAngles(String, int)`.
    pub fn imod_get_slicer_angles(
        &'static self,
        imod_key: Option<&str>,
        imod_index: i32,
    ) -> Option<SlicerAngles> {
        let mut results: Option<Vec<String>> = None;
        if imod_index == -1 {
            self.open_message(
                &format!(
                    "The is no open {} 3dmod for the highlighted row.",
                    imod_key.unwrap_or("null")
                ),
                "No 3dmod",
                Some(AxisID::Only),
            );
        }
        match self
            .get_imod_manager()
            .get_slicer_angles(imod_key.unwrap_or_default(), imod_index)
        {
            Ok(value) => results = value,
            Err(except @ ImodManagerException::AxisType(_)) => {
                eprintln!("{except}");
                self.open_message(&except.to_string(), "AxisType problem", Some(AxisID::Only));
            }
            Err(e @ ImodManagerException::Io(_)) => {
                eprintln!("{e}");
                self.open_message(&e.to_string(), "IO Exception", Some(AxisID::Only));
            }
            Err(e @ ImodManagerException::SystemProcess(_)) => {
                eprintln!("{e}");
                self.open_message(
                    &e.to_string(),
                    "System Process Exception",
                    Some(AxisID::Only),
                );
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
        let mut message_array: Vec<String> = Vec::new();
        let mut slicer_angles: Option<SlicerAngles> = None;
        match results {
            None => {
                message_array.push("Unable to retrieve slicer angles.".to_string());
                message_array.push(format!(
                    "The {} may not be open in 3dmod.",
                    imod_key.unwrap_or("null")
                ));
            }
            Some(results) => {
                let mut angles = SlicerAngles::new();
                let mut found_result_line1 = false;
                let mut found_result = false;
                for result in &results {
                    if ProcessMessages::get_error_index(result).is_some()
                        || result.contains(MessageType::Warning.tag().unwrap_or_default())
                    {
                        message_array.push(result.clone());
                    } else if !found_result_line1
                        && !found_result
                        && result == imod_process::SLICER_ANGLES_RESULTS_STRING1
                    {
                        found_result_line1 = true;
                    } else if found_result_line1
                        && !found_result
                        && result == imod_process::SLICER_ANGLES_RESULTS_STRING2
                    {
                        found_result = true;
                    } else if found_result && !angles.is_complete() {
                        angles.add(Some(result));
                    } else {
                        message_array.push(result.clone());
                    }
                }
                if !angles.is_complete() {
                    message_array.push(format!(
                        "Unable to retrieve slicer angles from {} 3dmod.",
                        imod_key.unwrap_or("null")
                    ));
                    if !angles.is_empty() {
                        message_array.push(format!("slicerAngles={}", angles));
                    }
                }
                slicer_angles = Some(angles);
            }
        }
        if !message_array.is_empty() {
            let this = self.this_static();
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_array_string_axis_id(
                    Some(this),
                    &message_array,
                    "Slicer Angles",
                    Some(AxisID::Only),
                )
            });
        }
        slicer_angles
    }

    /// Java `makejoincom(ProcessSeries, Deferred3dmodButton, Run3dmodMenuOptions,
    /// DialogType)`.
    pub fn makejoincom(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, AxisID::Only, dialog_type, Some("makejoincom")),
        };
        let Some(join_dialog) = self.join_dialog.get() else {
            process_series.borrow().end_series();
            return;
        };
        if !join_dialog.get_meta_data(self.meta_data(), true) {
            process_series.borrow().end_series();
            return;
        }
        if !self
            .meta_data()
            .is_valid_string(join_dialog.get_working_dir_name().as_deref())
        {
            self.open_message(
                &self.meta_data().get_invalid_reason(),
                "Invalid Data",
                Some(AxisID::Only),
            );
            process_series.borrow().end_series();
            return;
        }
        if !join_dialog.validate_makejoincom() {
            process_series.borrow().end_series();
            return;
        }
        let root_name = ConstJoinMetaData::get_dataset_name(self.meta_data());
        etomo_director::INSTANCE.rename_current_manager(root_name);
        let controller = *self.auto_alignment_controller.lock().unwrap();
        if let Some(controller) = controller {
            // catch (final LockException e) {}; Java declares no other exception.
            let _ = controller.create_empty_xf_file();
        }
        let makejoincom_param = MakejoincomParam::new(self.meta_data(), self.get_state(), self);
        if self.base.param_file.lock().unwrap().is_none() {
            self.end_setup_mode();
        }
        if !join_dialog.get_meta_data(self.meta_data(), true) {
            process_series.borrow().end_series();
            return;
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        match self.process_mgr().makejoincom(
            Arc::new(makejoincom_param),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => *self.base.thread_name_a.lock().unwrap() = thread_name,
            Err(except) => {
                eprintln!("{except}");
                self.open_message(
                    &format!("Can't run makejoincom\n{}", except),
                    "SystemProcessException",
                    Some(AxisID::Only),
                );
                return;
            }
        }
        process_series
            .borrow_mut()
            .set_next_process(Some("startjoin"), None);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Makejoincom"),
                AxisID::Only,
                Some(&ProcessName::MAKEJOINCOM),
            );
        }
    }

    /// Java `postProcess(String, ProcessDetails)`.  Post processing after a successful
    /// process.
    pub fn post_process(
        &'static self,
        command_name: Option<&str>,
        command_details: Option<&Arc<dyn Command + Send + Sync>>,
    ) {
        let process_details: Option<&dyn ProcessDetails> =
            command_details.and_then(|details| details.get_process_details());
        let Some(join_dialog) = self.join_dialog.get() else {
            return;
        };
        let result = (|| -> Result<(), LogFileError> {
            if command_name == Some(ProcessName::MAKEJOINCOM.to_string().as_str()) {
                if let Some(process_details) = process_details
                    && process_details.get_boolean_value(&makejoincom_param::Fields::Rotate)
                        == Some(true)
                {
                    let mut param = self.new_start_join_param();
                    param.set_rotate(true);
                    // Java's `getIntValue` returns an int; a field the details do not
                    // carry would throw IllegalArgumentException, and MakejoincomParam
                    // carries this one.
                    param.set_total_rows(
                        process_details
                            .get_int_value(&makejoincom_param::Fields::TotalRows)
                            .unwrap_or_default(),
                    );
                    param.set_rotation_angles_list(
                        process_details
                            .get_hashtable(&makejoincom_param::Fields::RotationAnglesList),
                    );
                    *self.start_join_param.lock().unwrap() = Some(param);
                }
                join_dialog.set_inverted()?;
            } else if command_name == Some(ProcessName::XFJOINTOMO.to_string().as_str()) {
                join_dialog.set_xfjointomo_result()?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(LogFileError::Lock(_)) => {}
            Err(e) => {
                eprintln!("{e:?}");
                let this = self.this_static();
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        Some(this),
                        &format!(
                            "Unable to read {}.\n{}",
                            dataset_files::XFJOINTOMO_LOG,
                            e.get_message()
                        ),
                        "Read Error",
                    )
                });
            }
        }
    }

    /// Java `endSetupMode()`.  If paramFile is not set, attempts to end setup mode and
    /// set the param file name.
    pub fn end_setup_mode(&'static self) -> bool {
        if self.base.param_file.lock().unwrap().is_some() {
            return self.set_mode();
        }
        let working_dir_name = self
            .join_dialog
            .get()
            .and_then(|join_dialog| join_dialog.get_working_dir_name());
        if !self.set_mode_with_dir(working_dir_name.as_deref()) {
            return false;
        }
        // setMode(String) is false for a null or blank directory, so the name is set.
        let working_dir_name = working_dir_name.unwrap_or_default();
        if working_dir_name.ends_with(' ') {
            self.open_message(
                &format!(
                    "The directory, {}, cannot be used because it ends with a space.",
                    working_dir_name
                ),
                "Unusable Directory Name",
                Some(AxisID::Only),
            );
            return false;
        }
        *self.base.property_user_dir.lock().unwrap() = Some(working_dir_name.clone());
        self.get_imod_manager()
            .set_meta_data_join_meta_data(self.meta_data());
        let param_file = Path::new(&working_dir_name).join(format!(
            "{}{}",
            ConstJoinMetaData::get_dataset_name(self.meta_data()),
            self.meta_data().get_file_extension().unwrap_or_default()
        ));
        *self.base.param_file.lock().unwrap() = Some(param_file.clone());
        if !param_file.exists() {
            self.process_mgr()
                .create_new_file(&utilities::java_io_file_get_absolute_path(
                    &param_file.to_string_lossy(),
                ));
        }
        *self.base.loaded_param_file.lock().unwrap() = true;
        // initializeUIParameters(paramFile, AxisID.ONLY, false);
        // if (loadedParamFile) {
        // imodManager.setMetaData(metaData);
        // mainPanel.setStatusBarText(paramFile, metaData, logPanel);
        // }
        self.set_main_panel_status_bar_text();
        true
    }

    /// Java private `copyMostRecentXfFile(String)`.
    fn copy_most_recent_xf_file(&'static self, command_description: &str) -> bool {
        let root_name = ConstJoinMetaData::get_dataset_name(self.meta_data());
        let xf_file_name = format!("{}.xf", root_name);
        let property_user_dir = self.property_user_dir().unwrap_or_default();
        let new_xf_file = utilities::most_recent_file(
            &property_user_dir,
            Some(&xf_file_name),
            Some(&format!(
                "{}{}",
                root_name,
                MidasParam::get_output_file_extension()
            )),
            Some(&format!(
                "{}{}",
                root_name,
                XfalignParam::get_output_file_extension()
            )),
            Some(&format!("{}_empty.xf", root_name)),
        );
        let Some(new_xf_file) = new_xf_file else {
            return true;
        };
        // If the most recent .xf file is not root.xf, copy it to root.xf
        let new_xf_file_name = utilities::java_io_file_get_name(&new_xf_file.to_string_lossy());
        if new_xf_file_name != xf_file_name {
            let xf_file = Path::new(&property_user_dir).join(&xf_file_name);
            if let Err(e) = utilities::copy_file(
                Some(self),
                Some(AxisID::Only),
                Some(&new_xf_file),
                Some(&xf_file),
                false,
                false,
                false,
            ) {
                eprintln!("{e:?}");
                let message = [
                    format!(
                        "Unable to copy {} to {}.",
                        utilities::java_io_file_get_absolute_path(&new_xf_file.to_string_lossy()),
                        xf_file_name
                    ),
                    format!("Copy {} to {}", new_xf_file_name, xf_file_name),
                    format!(" and then rerun {}.", command_description),
                ];
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Cannot Run Command",
                        Some(AxisID::Only),
                    )
                });
                return false;
            }
        }
        true
    }

    /// Java `setMode(String)`.  Sets the mode in joinDialog based on whether the
    /// working directory and root name are entered and whether a sample is saved.
    pub fn set_mode_with_dir(&'static self, working_dir_name: Option<&str>) -> bool {
        let state = self.get_state();
        // get a non-shared copy of doneMode
        let done_mode = state.get_done_mode();
        // only check done mode when we first set the mode
        state.clear_done_mode();
        let join_dialog = self.join_dialog.get();
        if !self.meta_data().is_valid_string(working_dir_name) {
            if let Some(join_dialog) = join_dialog {
                join_dialog.set_mode(join_dialog::SETUP_MODE);
            }
            return false;
        }
        if !state.is_sample_produced() || done_mode == join_dialog::CHANGING_SAMPLE_MODE {
            // either the sample was not produced, or the user had been changing the
            // sample when they exited the join dialog. If the done mode is
            // CHANGING_SAMPLE_MODE, then the sample values are not valid and the
            // original sample values have been lost.
            if let Some(join_dialog) = join_dialog {
                join_dialog.set_mode(join_dialog::SAMPLE_NOT_PRODUCED_MODE);
            }
        } else if let Some(join_dialog) = join_dialog {
            join_dialog.set_mode(join_dialog::SAMPLE_PRODUCED_MODE);
        }
        true
    }

    /// Java `setMode()`.
    pub fn set_mode(&'static self) -> bool {
        self.set_mode_with_dir(self.property_user_dir().as_deref())
    }

    /// Java `startjoin(ProcessSeries)`.
    pub fn startjoin(&'static self, process_series: Option<&ProcessSeriesHandle>) {
        let working_dir = self
            .join_dialog
            .get()
            .and_then(|join_dialog| join_dialog.get_working_dir());
        if !self.meta_data().is_valid_file(working_dir.as_deref()) {
            self.open_message(
                &self.meta_data().get_invalid_reason(),
                "Invalid Data",
                Some(AxisID::Only),
            );
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let param = {
            let mut start_join_param = self.start_join_param.lock().unwrap();
            start_join_param
                .take()
                .unwrap_or_else(|| StartJoinParam::new(AxisID::Only))
        };
        match self.process_mgr().startjoin(
            Arc::new(param),
            process_series.map(|series| Arc::new(EdtRef::new(Rc::clone(series)))),
        ) {
            Ok(thread_name) => {
                *self.base.thread_name_a.lock().unwrap() = thread_name;
                // startJoinParam = null (taken above).
            }
            Err(except) => {
                eprintln!("{except}");
                self.open_message(
                    &format!("Can't run startjoin.com\n{}", except),
                    "SystemProcessException",
                    Some(AxisID::Only),
                );
                return;
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Startjoin"),
                AxisID::Only,
                Some(&ProcessName::STARTJOIN),
            );
        }
    }

    /// Java `xfjointomo(ProcessSeries)`.
    pub fn xfjointomo(&'static self, process_series: Option<&ProcessSeriesHandle>) {
        let mut xfjointomo_param =
            XfjointomoParam::new(self, self.get_state().get_refine_trial().is());
        let valid = self.join_dialog.get().is_some_and(|join_dialog| {
            join_dialog.get_parameters_xfjointomo_param(&mut xfjointomo_param, true)
        });
        if !valid {
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        match self.process_mgr().xfjointomo(
            &mut xfjointomo_param,
            process_series.map(|series| Arc::new(EdtRef::new(Rc::clone(series)))),
        ) {
            Ok(thread_name) => *self.base.thread_name_a.lock().unwrap() = thread_name,
            Err(except) => {
                eprintln!("{except}");
                self.open_message(
                    &format!("Can't run {}\n{}", ProcessName::XFJOINTOMO, except),
                    "SystemProcessException",
                    Some(AxisID::Only),
                );
                return;
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&ProcessName::XFJOINTOMO.to_string()),
                AxisID::Only,
                Some(&ProcessName::XFJOINTOMO),
            );
        }
    }

    /// Java private `remapmodel(ProcessSeries)`.
    fn remapmodel(&'static self, process_series: &ProcessSeriesHandle) {
        let param = RemapmodelParam::new(self);
        match self.process_mgr().remapmodel(
            Arc::new(param),
            Some(Arc::new(EdtRef::new(Rc::clone(process_series)))),
        ) {
            Ok(thread_name) => *self.base.thread_name_a.lock().unwrap() = thread_name,
            Err(except) => {
                eprintln!("{except}");
                self.open_message(
                    &format!("Can't run {}\n{}", *remapmodel_param::COMMAND_NAME, except),
                    "SystemProcessException",
                    Some(AxisID::Only),
                );
                return;
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&remapmodel_param::COMMAND_NAME),
                AxisID::Only,
                Some(&ProcessName::REMAPMODEL),
            );
        }
    }

    /// Java `xfmodel(String, String, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType)`.
    pub fn xfmodel_with_files(
        &'static self,
        input_file: Option<&str>,
        output_file: Option<&str>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, AxisID::Only, dialog_type, Some("xfmodel")),
        };
        let mut param = XfmodelParam::new_join(self);
        param.set_input_file(input_file);
        param.set_output_file(output_file);
        if param.is_valid() {
            process_series
                .borrow_mut()
                .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
            self.xfmodel_with_param(param, Some(process_series), dialog_type);
        }
    }

    /// Java private `xfmodel(ProcessSeries, DialogType)`.
    fn xfmodel(
        &'static self,
        process_series: &ProcessSeriesHandle,
        dialog_type: Option<DialogType>,
    ) {
        self.xfmodel_with_param(
            XfmodelParam::new_join(self),
            Some(Rc::clone(process_series)),
            dialog_type,
        );
    }

    /// Java private `xfmodel(XfmodelParam, ProcessSeries, DialogType)`.
    fn xfmodel_with_param(
        &'static self,
        param: XfmodelParam,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: Option<DialogType>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, AxisID::Only, dialog_type, Some("xfmodel")),
        };
        let state = self.get_state();
        if *self.debug.lock().unwrap() {
            eprintln!("xfmodel:gapExist={}", state.is_gaps_exist());
        }
        if state.is_gaps_exist() {
            process_series
                .borrow_mut()
                .set_next_process(Some(&ProcessName::REMAPMODEL.to_string()), None);
        }
        match self.process_mgr().xfmodel(
            Arc::new(param),
            AxisID::Only,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => *self.base.thread_name_a.lock().unwrap() = thread_name,
            Err(except) => {
                eprintln!("{except}");
                self.open_message(
                    &format!("Can't run {}\n{}", *xfmodel_param::COMMAND_NAME, except),
                    "SystemProcessException",
                    Some(AxisID::Only),
                );
                return;
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&xfmodel_param::COMMAND_NAME),
                AxisID::Only,
                Some(&ProcessName::XFMODEL),
            );
        }
    }

    /// Java private `xftoxg(ProcessSeries, DialogType)`.
    fn xftoxg(
        &'static self,
        process_series: &ProcessSeriesHandle,
        _dialog_type: Option<DialogType>,
    ) {
        process_series
            .borrow_mut()
            .set_next_process(Some(&ProcessName::XFMODEL.to_string()), None);
        let mut param = XftoxgParam::new(self);
        let state = self.get_state();
        let ref_section = state.get_join_alignment_ref_section(state.get_refine_trial().is());
        if !ref_section.is_null() {
            param.set_reference_section_const_etomo_number(Some(&ref_section));
        }
        param.set_number_to_fit_int(0);
        param.set_xf_file_name(&dataset_files::get_refine_xf_file_name(self));
        param.set_xg_file_name(&dataset_files::get_refine_xg_file_name(self));
        match self.process_mgr().xftoxg(
            Arc::new(param),
            Some(Arc::new(EdtRef::new(Rc::clone(process_series)))),
        ) {
            Ok(thread_name) => *self.base.thread_name_a.lock().unwrap() = thread_name,
            Err(except) => {
                eprintln!("{except}");
                self.open_message(
                    &format!("Can't run {}\n{}", xftoxg_param::command_name(), except),
                    "SystemProcessException",
                    Some(AxisID::Only),
                );
                return;
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&xftoxg_param::command_name()),
                AxisID::Only,
                Some(&ProcessName::XFTOXG),
            );
        }
    }

    /// Java `updateJoinDialogDisplay()`.
    pub fn update_join_dialog_display(&self) {
        if let Some(join_dialog) = self.join_dialog.get() {
            join_dialog.update_display();
        }
    }

    /// Java `finishjoin(FinishjoinParam.Mode, String, ProcessSeries,
    /// Deferred3dmodButton, Run3dmodMenuOptions, DialogType)`.  Runs finishjoin in the
    /// mode specified.  If the mode is SUPPRESS_EXECUTION then finishjoin is not
    /// executed, and only used for placing data into JoinState.
    pub fn finishjoin(
        &'static self,
        mode: finishjoin_param::Mode,
        button_text: &str,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, AxisID::Only, dialog_type, Some("finishjoin")),
        };
        if !self.update_meta_data_from_join_dialog(AxisID::Only, true) {
            process_series.borrow().end_series();
            return;
        }
        let param = FinishjoinParam::new(self, mode);
        if !self.copy_most_recent_xf_file(button_text) {
            process_series.borrow().end_series();
            return;
        }
        if !self
            .join_dialog
            .get()
            .is_some_and(|join_dialog| join_dialog.validate_finishjoin())
        {
            process_series.borrow().end_series();
            return;
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        if mode == finishjoin_param::Mode::Rejoin
            || mode == finishjoin_param::Mode::SuppressExecution
        {
            process_series
                .borrow_mut()
                .set_next_process(Some(&ProcessName::XFTOXG.to_string()), None);
        }
        let series_ref = Some(Arc::new(EdtRef::new(Rc::clone(&process_series))));
        if mode == finishjoin_param::Mode::SuppressExecution {
            process_series
                .borrow_mut()
                .set_last_process(Some(imod_manager::TRANSFORMED_MODEL_KEY));
            self.process_mgr()
                .save_finishjoin_state(Arc::new(param), series_ref);
            ProcessSeries::start_next_process(&process_series, AxisID::Only);
        } else {
            match self.process_mgr().finishjoin(Arc::new(param), series_ref) {
                Ok(thread_name) => *self.base.thread_name_a.lock().unwrap() = thread_name,
                Err(except) => {
                    eprintln!("{except}");
                    self.open_message(
                        &format!("Can't run {}\n{}", button_text, except),
                        "SystemProcessException",
                        Some(AxisID::Only),
                    );
                    return;
                }
            }
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.start_progress_bar_string_axis_id_process_name(
                    Some(&format!("Finishjoin: {}", button_text)),
                    AxisID::Only,
                    Some(&ProcessName::FINISHJOIN),
                );
            }
        }
    }

    /// Java private `updateMetaDataFromJoinDialog(AxisID, boolean)`.
    fn update_meta_data_from_join_dialog(
        &'static self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> bool {
        if !self
            .join_dialog
            .get()
            .is_some_and(|join_dialog| join_dialog.get_meta_data(self.meta_data(), do_validation))
        {
            return false;
        }
        if !self
            .meta_data()
            .is_valid_string(self.property_user_dir().as_deref())
        {
            self.open_message(
                &self.meta_data().get_invalid_reason(),
                "Invalid Data",
                Some(axis_id),
            );
            return false;
        }
        let result = (|| -> Result<bool, LogFileError> {
            let Some(parameter_store) = self.get_parameter_store(Some(axis_id))? else {
                return Ok(false);
            };
            parameter_store
                .lock()
                .unwrap()
                .save(Some(self.meta_data() as &dyn Storable))?;
            Ok(true)
        })();
        match result {
            Ok(false) => return false,
            Ok(true) => {}
            Err(LogFileError::Lock(_)) => {}
            Err(e) => {
                let this = self.this_static();
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        Some(this),
                        &format!("Cannot save or write to metaData.\n{}", e.get_message()),
                        "Etomo Error",
                    )
                });
            }
        }
        true
    }

    /// Java `setSize(String, String)`.
    pub fn set_size(&self, size_in_x_string: Option<&str>, size_in_y_string: Option<&str>) {
        let mut size_in_x = EtomoNumber::new_with_type(Some(Type::Integer));
        let mut size_in_y = EtomoNumber::new_with_type(Some(Type::Integer));
        size_in_x.set_string(size_in_x_string);
        let join_dialog = self.join_dialog.get();
        if let Some(join_dialog) = &join_dialog {
            join_dialog.set_size_in_x(&size_in_x);
        }
        size_in_y.set_string(size_in_y_string);
        if let Some(join_dialog) = &join_dialog {
            join_dialog.set_size_in_y(&size_in_y);
        }
    }

    /// Java `setShift(int, int)`.
    pub fn set_shift(&self, shift_in_x: i32, shift_in_y: i32) {
        if let Some(join_dialog) = self.join_dialog.get() {
            join_dialog.set_shift_in_x(shift_in_x);
            join_dialog.set_shift_in_y(shift_in_y);
        }
    }

    /// Java `rotx(File, File, ProcessSeries)`.
    pub fn rotx(
        &'static self,
        tomogram: Option<&Path>,
        working_dir: Option<&Path>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        // Upstream bug fixed in translation (JoinManager.java:873): a null tomogram or
        // working directory throws NullPointerException in ClipParam; nothing runs.
        let (Some(tomogram), Some(working_dir)) = (tomogram, working_dir) else {
            return;
        };
        let clip_param = ClipParam::get_rotx_instance(self, AxisID::Only, tomogram, working_dir);
        match self.process_mgr().rotx(
            Arc::new(clip_param),
            process_series.map(|series| Arc::new(EdtRef::new(series))),
        ) {
            Ok(thread_name) => *self.base.thread_name_a.lock().unwrap() = thread_name,
            Err(except) => {
                if let Some(join_dialog) = self.join_dialog.get() {
                    join_dialog.abort_add_section();
                }
                eprintln!("{except}");
                self.open_message(
                    &format!("Can't run clip rotx\n{}", except),
                    "SystemProcessException",
                    Some(AxisID::Only),
                );
                return;
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id(
                Some(&format!(
                    "rotating {}",
                    utilities::java_io_file_get_name(&tomogram.to_string_lossy())
                )),
                AxisID::Only,
            );
        }
    }

    /// Java `abortAddSection()`.
    pub fn abort_add_section(&self) {
        if let Some(join_dialog) = self.join_dialog.get() {
            join_dialog.abort_add_section();
        }
    }

    /// Java `addSection(File)`.
    pub fn add_section(&self, tomogram: Option<&Path>) {
        // Upstream bug fixed in translation (JoinManager.java:892): a null file would
        // reach `new SectionTableRow(... null ...)` and throw; nothing is added.
        if let (Some(join_dialog), Some(tomogram)) = (self.join_dialog.get(), tomogram) {
            join_dialog.add_section(tomogram);
        }
    }

    /// Java private `openProcessingPanel()`.  Open the main window in processing mode.
    /// MUST run reconnect for all axis.
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

    /// Java `getConstMetaData()`.
    pub fn get_const_meta_data(&self) -> &JoinMetaData {
        self.meta_data()
    }

    /// Java `getJoinMetaData()`.
    pub fn get_join_meta_data(&'static self) -> &'static JoinMetaData {
        self.meta_data()
    }

    /// Java package-private `createState()`.  The state lives for the run, so the
    /// allocation is leaked and rooted in `STATE_ROOTS`.
    pub(crate) fn create_state(&'static self) {
        let state: &'static JoinState = Box::leak(Box::new(JoinState::new(self)));
        STATE_ROOTS.lock().unwrap().push(state);
        let _ = self.state.set(state);
    }

    /// Java `getState()`, declared `ConstJoinState`.  The constructor calls
    /// `createState`, so the field is set for every constructed manager.
    pub fn get_state(&self) -> &'static JoinState {
        self.state.get().expect("state")
    }

    /// Java `newStartJoinParam()`.  The caller fills the returned parameter and puts
    /// it back in `startJoinParam` (Java returns the field's object itself).
    pub fn new_start_join_param(&self) -> StartJoinParam {
        StartJoinParam::new(AxisID::Only)
    }

    /// Java public final `packDialogs(AxisID)`: empty.
    pub fn pack_dialogs_with_axis(&self, _axis_id: Option<AxisID>) {}

    /// Java public final `packDialogs()`: empty.
    pub fn pack_dialogs(&self) {}

    /// Java private `updateJoinwarp2modelParam(boolean)`.
    fn update_joinwarp2model_param(
        &'static self,
        do_validation: bool,
    ) -> Option<Joinwarp2modelParam> {
        self.com_script_mgr().load_join_warp_2_model(AxisID::Only);
        let mut param = self
            .com_script_mgr()
            .get_joinwarp2model_param(AxisID::Only)?;
        let refine_model_filename = dataset_files::get_refine_model_file_name(self);
        param.set_refine_model_filename(Some(&refine_model_filename));
        let modeled_join_filename = file_type::CLASS
            .modeled_join
            .get_file_name(Some(self), Some(AxisID::Only));
        param.set_modeled_join_filename(modeled_join_filename.as_deref());
        let input_warp_file_name = file_type::CLASS
            .local_transformation_list
            .get_file_name(Some(self), Some(AxisID::Only));
        param.set_input_warp_filename(input_warp_file_name.as_deref());
        let applied_transform_filename = file_type::CLASS
            .warp_xg
            .get_file_name(Some(self), Some(AxisID::Only));
        param.set_applied_transform_filename(applied_transform_filename.as_deref());

        if !self.join_dialog.get().is_some_and(|join_dialog| {
            join_dialog.get_parameters_joinwarp2model_param(&mut param, do_validation)
        }) {
            return None;
        }
        self.com_script_mgr()
            .save_joinwarp2model(&param, AxisID::Only);
        Some(param)
    }

    /// Java `startRefine()`.  Copy the join or trial join file to _modeled.join.  Also
    /// try to make sure that the data about the join or trial join file is accurate.
    pub fn start_refine(&'static self) {
        let Some(join_dialog) = self.join_dialog.get() else {
            return;
        };
        let process_series = ProcessSeries::new(
            self,
            AxisID::Only,
            Some(DialogType::Join),
            Some("startRefine"),
        );
        let use_trial = join_dialog.is_refine_with_trial();
        let join_file_type = if use_trial {
            &file_type::CLASS.trial_join
        } else {
            &file_type::CLASS.join
        };
        let button_name = join_dialog.get_button_name(use_trial); // ask join dialog
        let dialog_axis_id = join_dialog.get_axis_id();
        let join_file_name = join_file_type
            .get_file_name(Some(self), Some(dialog_axis_id))
            .unwrap_or_else(|| "null".to_string());
        // make sure the file to be moved exists
        if !dataset_files::get_join_file(use_trial, self).exists() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self),
                    &format!(
                        "{} does not exist.  Press {} to create it.",
                        join_file_name, button_name
                    ),
                    "Failed File Move",
                )
            });
            process_series.borrow().end_series();
            return;
        }
        // Make sure that there is data available about the .join file which will be
        // used in the refine. If there is not, ask for the user's assurance that
        // data on the screen corresponds to the .join file.
        let mut convert_version = false;
        let state = self.get_state();
        if !state.is_join_version_ge(use_trial, Some(&join_state::MIN_REFINE_VERSION)) {
            join_dialog.set_refine_data_highlight(true);
            convert_version = ui_harness::with(|harness| {
                harness.open_yes_no_dialog_base_manager_string_axis_id(
                    Some(self),
                    &format!(
                        "IMPORTANT!!  Information about the {} file has not been saved.\nRefine \
                         cannot proceed without complete information on the join file.  If the \
                         highlighted data on the screen has NOT been changed since the {} file \
                         was created, press Yes.  Otherwise press no and then press {} to \
                         recreate the file.",
                        join_file_name, join_file_name, button_name
                    ),
                    Some(AxisID::Only),
                )
            });
            join_dialog.set_refine_data_highlight(false);
            if !convert_version {
                process_series.borrow().end_series();
                return;
            }
        }
        // The _modeled.join file already exists and there is a model that may be
        // associated with it. Warn the user before overwriting it.
        if file_type::CLASS
            .modeled_join
            .get_file(Some(self), Some(AxisID::Only))
            .is_some_and(|file| file.exists())
            && dataset_files::get_refine_model_file(self).exists()
            && !ui_harness::with(|harness| {
                harness.open_yes_no_dialog_base_manager_string_axis_id(
                    Some(self),
                    "The modeled join file and the refine model already exist.\nPress Yes to \
                     overwrite the modeled join file.  IMPORTANT:  If the binning of the \
                     modeled join file will be different, you must open the model with the \
                     new modeled join file and save it at least once.",
                    Some(AxisID::Only),
                )
            })
        {
            process_series.borrow().end_series();
            return;
        }
        let imod_key = if use_trial {
            imod_manager::TRIAL_JOIN_KEY
        } else {
            imod_manager::JOIN_KEY
        };
        if !self.close_imod(
            Some(imod_key),
            Some(dialog_axis_id),
            Some(&format!("{} file in 3dmod", join_file_name)),
            false,
        ) {
            // Java passes a fourth message argument, "This file will be moved to
            // <modeled join>", to its five-argument closeImod.
            process_series.borrow().end_series();
            return;
        }
        let result = (|| -> Result<(), LogFileError> {
            // move the join file (or trial join) to the _modeled.join file
            println!("backup {}", file_type::CLASS.modeled_join);
            self.backup_image_file(Some(&file_type::CLASS.modeled_join), Some(dialog_axis_id))?;
            println!(
                "rename {},{}",
                join_file_type,
                file_type::CLASS.modeled_join
            );
            utilities::rename_file(
                Some(self),
                Some(dialog_axis_id),
                join_file_type
                    .get_file(Some(self), Some(dialog_axis_id))
                    .as_deref(),
                file_type::CLASS
                    .modeled_join
                    .get_file(Some(self), Some(dialog_axis_id))
                    .as_deref(),
                false,
                false,
                false,
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(e @ LogFileError::Lock(_)) => {
                eprintln!("{e:?}");
                process_series.borrow().end_series();
                return;
            }
            Err(e) => {
                eprintln!("{e:?}");
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        Some(self),
                        &format!("Unable to move join file.\n{}", e.get_message()),
                        "Failed File Move",
                    )
                });
                process_series.borrow().end_series();
                return;
            }
        }
        // convertVersion is true when the .ejf file is an older version and the user
        // wishes to use screen values to convert it.
        if convert_version {
            // ask join dialog
            if !join_dialog.get_section_table_meta_data() {
                process_series.borrow().end_series();
                return;
            }
            state.set_join_version1_0(use_trial, self.get_join_meta_data());
        }
        let model_file = Path::new(&self.property_user_dir().unwrap_or_default())
            .join(dataset_files::get_refine_model_file_name(self));
        // Joinwarp2model
        process_series
            .borrow_mut()
            .set_next_process(Some(&ProcessName::JOIN_WARP_2_MODEL.to_string()), None);
        if !model_file.exists()
            && file_type::CLASS
                .warp_xg
                .exists(Some(self), Some(AxisID::Only))
        {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.start_progress_bar_string_axis_id_process_name(
                    Some("Start Refine"),
                    dialog_axis_id,
                    Some(&ProcessName::JOIN_WARP_2_MODEL),
                );
            }
            BaseProcessManager::touch(
                &file_type::CLASS
                    .join_warp_2_model_comscript
                    .get_file(Some(self), Some(AxisID::Only))
                    .map(|file| utilities::java_io_file_get_absolute_path(&file.to_string_lossy()))
                    .unwrap_or_default(),
                Some(self),
            );
            let Some(param) = self.update_joinwarp2model_param(true) else {
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        dialog_axis_id,
                        Some(ProcessEndState::Failed),
                    );
                }
                process_series.borrow().end_series();
                return;
            };
            let thread_name = match self.process_mgr().joinwarp2model(
                &param,
                Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
            ) {
                Ok(thread_name) => thread_name,
                Err(e) => {
                    eprintln!("{e}");
                    self.open_message(
                        &format!(
                            "Can not execute {}\n{}",
                            joinwarp2model_param::COMMAND_NAME,
                            e
                        ),
                        "Unable to execute command",
                        Some(AxisID::Only),
                    );
                    return;
                }
            };
            self.set_thread_name(Some(&thread_name), Some(AxisID::Only));
        } else {
            ProcessSeries::start_next_process(&process_series, AxisID::Only);
        }
    }

    /// Java private `refineJoin(ProcessSeries)`.
    fn refine_join(&'static self, process_series: Option<&ProcessSeriesHandle>) {
        let Some(join_dialog) = self.join_dialog.get() else {
            return;
        };
        let dialog_axis_id = join_dialog.get_axis_id();
        // Refine join
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id(Some("Refine Join"), dialog_axis_id);
        }
        let use_trial = join_dialog.is_refine_with_trial();
        join_dialog.set_refining_join(true);
        self.get_state().set_refine_trial(use_trial);
        join_dialog.update_display();
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id_process_end_state(
                dialog_axis_id,
                Some(ProcessEndState::Done),
            );
        }
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process(process_series, AxisID::Only);
        }
    }
}

impl BaseManager for JoinManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    /// Java `getInterfaceType()`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::Join)
    }

    /// Java `saveParamFile()`.
    fn save_param_file(&'static self) -> Result<bool, LogFileError> {
        let retval = self.save_param_file_super()?;
        if retval {
            self.end_setup_mode();
        }
        Ok(retval)
    }

    /// Java `setParamFile()`.
    fn set_param_file(&self) -> bool {
        let this = self.this_static();
        if !*self.base.loaded_param_file.lock().unwrap()
            && let Some(join_dialog) = self.join_dialog.get()
        {
            let dir = join_dialog.get_working_dir_name();
            let root = join_dialog.get_root_name();
            if let (Some(dir), Some(root)) = (dir, root)
                && !crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace(&dir)
                && !crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace(&root)
            {
                let file = Path::new(&dir).join(format!("{}{}", root, DataFileType::Join.extension().unwrap_or_default()));
                if !file.exists() {
                    self.process_mgr()
                        .create_new_file(&utilities::java_io_file_get_absolute_path(
                            &file.to_string_lossy(),
                        ));
                }
                this.initialize_ui_parameters(Some(&file), Some(AxisID::Only), false);
                if *self.base.loaded_param_file.lock().unwrap() {
                    self.get_imod_manager()
                        .set_meta_data_join_meta_data(self.meta_data());
                    self.set_main_panel_status_bar_text();
                }
            }
        }
        *self.base.loaded_param_file.lock().unwrap()
    }

    /// Java `getFocusComponent()`.
    fn get_focus_component(&self) -> Option<Rc<crate::imod::etomo::jdk::JComponent>> {
        self.join_dialog
            .get()
            .and_then(|join_dialog| join_dialog.get_focus_component())
    }

    /// Java package-private `paramString()`.
    fn param_string(&self) -> Option<String> {
        let join_dialog = if crate::imod::etomo::util::event_queue::is_dispatch_thread() {
            self.join_dialog
                .get()
                .map(|join_dialog| join_dialog.to_string())
        } else {
            None
        };
        Some(format!(
            "joinDialog={},metaData={},\nprocessMgr={},state={},\nsuper[{}]",
            join_dialog.unwrap_or_else(|| "null".to_string()),
            self.meta_data
                .get()
                .map(|meta_data| meta_data.to_string())
                .unwrap_or_else(|| "null".to_string()),
            self.process_mgr
                .get()
                .map(|_| "etomo.process.JoinProcessManager".to_string())
                .unwrap_or_else(|| "null".to_string()),
            self.state
                .get()
                .map(|state| state.to_string())
                .unwrap_or_else(|| "null".to_string()),
            self.param_string_super().unwrap_or_default()
        ))
    }

    /// Java `getParamFile()`.  Return the test parameter file as a File object.
    fn get_param_file(&self) -> Option<PathBuf> {
        if self.base.param_file.lock().unwrap().is_none() && !self.this_static().done_join_dialog()
        {
            return None;
        }
        self.base.param_file.lock().unwrap().clone()
    }

    /// Java package-private `createMainPanel()`.
    fn create_main_panel(&self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            let this = self.this_static();
            self.main_panel.set(Some(MainJoinPanel::new(this)));
        }
    }

    /// Java `setDebug(boolean)`.
    fn set_debug(&self, debug: bool) {
        self.set_debug_super(debug);
        *self.debug.lock().unwrap() = debug;
    }

    /// Java package-private final `isTomosnapshotThumbnail()`.
    fn is_tomosnapshot_thumbnail(&self) -> bool {
        true
    }

    /// Java `setParamFile(File)`.  Set the data set parameter file. This also updates
    /// the mainframe data parameters.
    fn set_param_file_from(&self, param_file: Option<&Path>) -> bool {
        if !self.set_param_file_from_super(param_file) {
            return false;
        }
        // Update main window information and status bar
        self.set_main_panel_status_bar_text();
        true
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
        if *self.debug.lock().unwrap() {
            eprintln!(
                "startNextProcess:axisID={},nextProcess={}",
                axis_id,
                process.to_source_string()
            );
        }
        if process.equals_string(Some("startjoin")) {
            self.startjoin(Some(process_series));
            return true;
        }
        if process.equals_string(Some(&ProcessName::XFTOXG.to_string())) {
            self.xftoxg(process_series, dialog_type);
            return true;
        }
        if process.equals_string(Some(&ProcessName::XFMODEL.to_string())) {
            self.xfmodel(process_series, dialog_type);
            return true;
        }
        if process.equals_string(Some(&ProcessName::REMAPMODEL.to_string())) {
            self.remapmodel(process_series);
            return true;
        }
        if process.equals_string(Some(imod_manager::TRANSFORMED_MODEL_KEY)) {
            self.imod_open_process_series(
                Some(process_series),
                Some(imod_manager::TRANSFORMED_MODEL_KEY),
            );
            return true;
        }
        if process.equals_string(Some(&ProcessName::JOIN_WARP_2_MODEL.to_string())) {
            self.refine_join(Some(process_series));
            return true;
        }
        false
    }

    /// Java `getBaseMetaData()`.
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        self.meta_data
            .get()
            .map(|meta_data| meta_data as &dyn BaseMetaData)
    }

    /// Java `getMainPanel()`.
    fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.main_panel
            .get()
            .map(|main_panel| main_panel as Rc<dyn MainPanelVirtual>)
    }

    /// Java `getBaseState()`.
    fn get_base_state(&self) -> Option<&'static dyn BaseState> {
        self.state
            .get()
            .map(|state| *state as &'static dyn BaseState)
    }

    /// Java package-private `getAutoAlignmentMetaData()`.
    fn get_auto_alignment_meta_data(&self) -> Option<&'static Mutex<AutoAlignmentMetaData>> {
        let this = self.this_static();
        Some(this.meta_data().get_auto_alignment_meta_data())
    }

    /// Java `updateMetaData(DialogType, AxisID, boolean)`.
    fn update_meta_data(
        &self,
        _dialog_type: Option<DialogType>,
        _axis_id: Option<AxisID>,
        do_validation: bool,
    ) -> bool {
        self.join_dialog
            .get()
            .is_some_and(|join_dialog| join_dialog.get_meta_data(self.meta_data(), do_validation))
    }

    /// Java `pause(AxisID)`.  Interrupt the currently running thread for this axis.
    ///
    /// Java throws `IllegalStateException("pause is not available in join")`, which
    /// Swing's handler prints; it is printed here and nothing is paused.
    fn pause(&self, _axis_id: Option<AxisID>) -> bool {
        eprintln!("java.lang.IllegalStateException: pause is not available in join");
        false
    }

    /// Java `getProcessManager()`.
    fn get_process_manager(&self) -> Option<&'static BaseProcessManager> {
        self.process_mgr.get().map(|process_mgr| &process_mgr.base)
    }

    /// Java package-private final `getStorables(int)`.
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Option<&'static dyn Storable>>> {
        let this = self.this_static();
        let mut storable: Vec<Option<&'static dyn Storable>> =
            vec![None; (3 + offset).max(0) as usize];
        let mut index = offset.max(0) as usize;
        storable[index] = this
            .meta_data
            .get()
            .map(|meta_data| meta_data as &dyn Storable);
        index += 1;
        storable[index] = this.state.get().map(|state| *state as &dyn Storable);
        index += 1;
        storable[index] = Some(&this.screen_state as &dyn Storable);
        Some(storable)
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
        self.done_join_dialog();
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.done();
        }
        Ok(true)
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        self.meta_data
            .get()
            .map(|meta_data| ConstJoinMetaData::get_name(meta_data))
    }
}

/// Java `toString()`.
impl std::fmt::Display for JoinManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "etomo.JoinManager[{}]",
            self.param_string().unwrap_or_default()
        )
    }
}
