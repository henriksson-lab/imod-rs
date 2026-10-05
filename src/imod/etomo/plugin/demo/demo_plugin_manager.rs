//! `IMOD/Etomo/src/etomo/plugin/demo/DemoPluginManager.java`.
//!
//! This demo may not be ideal for copying and modifying, but it does contain up to date
//! ways of using etomo's functions and classes.  (The rest of the Java class comment -
//! advice on relocating an extension to a separate package - is not repeated here.)
//!
//! The manager for the plugin.  There is a separate instance for each axis.  This
//! plugin creates a panel which is extremely integrated with etomo.
//!
//! An event-dispatch-thread object created as `Rc<Self>` by [`DemoPluginManager::new`]
//! (Java: the service loader's default constructor); every method takes `&self` and
//! the fields the Java assigns after construction sit in `Cell`/`RefCell`s.  It is the
//! `NextProcessTarget` of the process series it starts, so it keeps its own weak
//! handle.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::demo_com_script_manager::DemoComScriptManager;
use super::demo_file_type;
use super::demo_imod_manager::{self, DemoImodManager};
use super::demo_panel::DemoPanel;
use super::demo_process_manager::DemoProcessManager;
use super::demo_process_name;
use super::demo_process_result_display_factory::DemoProcessResultDisplayFactory;
use super::demo_screen_state::DemoScreenState;
use super::demo_tilt_panel::DemoTiltPanel;
use super::etomo_plugin_demo_param::EtomoPluginDemoParam;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::plugin::plugin::Plugin;
use crate::imod::etomo::plugin::plugin_panel::PluginPanel;
use crate::imod::etomo::plugin::tomo_gen_method_plugin::TomoGenMethodPlugin;
use crate::imod::etomo::process::base_imod_manager::ImodManagerException;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::process_interface::ProcessResultDisplayRef;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::process_series::{
    NextProcessTarget, Process, ProcessSeries, ProcessSeriesHandle,
};
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::process_result::ProcessResult;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::ui::UiComponent;
use crate::imod::etomo::ui::swing::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::ui::swing::global_expand_button::GlobalExpandButton;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::ui::swing::tilt_panel::TiltPanelVirtual;
use crate::imod::etomo::ui::swing::tomogram_generation_dialog::TomogramGenerationDialog;
use crate::imod::etomo::ui::swing::tomogram_generation_expert::TomogramGenerationExpert;
use crate::imod::etomo::ui::swing::tomogram_generation_parent::TomogramGenerationParent;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::event_queue::EdtRef;

/// Java `public class DemoPluginManager implements TomoGenMethodPlugin,
/// NextProcessTarget`.
pub struct DemoPluginManager {
    /// Rust-only: Java `this`.
    self_ref: Weak<DemoPluginManager>,
    /// Java private `manager`, initialised to null.
    manager: Cell<Option<&'static ApplicationManager>>,
    /// Java private `axisID`, initialised to null.
    axis_id: Cell<Option<AxisID>>,
    /// Java private `axisType`, initialised to null.  The Java never assigns it (`init`
    /// does not store its `axisType` parameter), so it stays null; kept native.
    axis_type: Cell<Option<AxisType>>,
    /// Java private `comScriptMgr`, initialised to null.
    com_script_mgr: RefCell<Option<Rc<DemoComScriptManager>>>,
    /// Java private `processMgr`, initialised to null.  A separate process manager that
    /// shares axis blocking information with the manager's process manager.
    process_mgr: Cell<Option<&'static DemoProcessManager>>,
    /// Java private `dialogType`, initialised to null.
    dialog_type: Cell<Option<DialogType>>,
    /// Java private `expert`, initialised to null (the expert owns this plugin).
    expert: RefCell<Weak<TomogramGenerationExpert>>,
    /// Java private `screenState`, initialised to null.
    screen_state: RefCell<Option<Arc<DemoScreenState>>>,
    /// Java private `processResultDisplayFactory`, initialised to null.
    process_result_display_factory: RefCell<Option<Rc<DemoProcessResultDisplayFactory>>>,
    /// Java private `panel`, initialised to null.
    panel: RefCell<Option<Rc<DemoPanel>>>,
    /// Java private `imodManager`, initialised to null.
    imod_manager: Cell<Option<&'static DemoImodManager>>,
    /// Java private `autodoc`, initialised to null.  The autodoc factory keeps every
    /// autodoc it returns for the life of the process.
    autodoc: Cell<*const Autodoc>,
}

impl DemoPluginManager {
    /// Java public `DemoPluginManager()`.  Generic constructor required by service
    /// loader.
    pub fn new() -> Rc<DemoPluginManager> {
        Rc::new_cyclic(|this: &Weak<DemoPluginManager>| DemoPluginManager {
            self_ref: this.clone(),
            manager: Cell::new(None),
            axis_id: Cell::new(None),
            axis_type: Cell::new(None),
            com_script_mgr: RefCell::new(None),
            process_mgr: Cell::new(None),
            dialog_type: Cell::new(None),
            expert: RefCell::new(Weak::new()),
            screen_state: RefCell::new(None),
            process_result_display_factory: RefCell::new(None),
            panel: RefCell::new(None),
            imod_manager: Cell::new(None),
            autodoc: Cell::new(std::ptr::null()),
        })
    }

    /// Rust-only: Java `this` as a strong reference.
    fn this(&self) -> Rc<DemoPluginManager> {
        self.self_ref
            .upgrade()
            .expect("DemoPluginManager used after it was dropped")
    }

    /// Rust-only: the Java `axisID` field read where the Java uses it as a non-null
    /// axis (`init` always sets it).
    fn axis(&self) -> AxisID {
        self.axis_id.get().unwrap_or(AxisID::Only)
    }

    /// Rust-only: the Java `panel` field read (null is `None`), cloned out so no borrow
    /// is held.
    fn panel(&self) -> Option<Rc<DemoPanel>> {
        self.panel.borrow().clone()
    }

    /// Java package-private `getAutodoc() throws LogFileException, IOException,
    /// LockException`.
    pub fn get_autodoc(&self) -> Result<Option<&'static dyn ReadOnlyAutodoc>, LogFileError> {
        if self.autodoc.get().is_null() {
            let manager: Option<&'static dyn BaseManager> = self
                .manager
                .get()
                .map(|manager| manager as &'static dyn BaseManager);
            // SAFETY: the factory keeps every autodoc it returns (and its sections) for
            // the life of the process.
            let autodoc = unsafe {
                autodoc_factory::get_unmanaged_instance_file_type(
                    manager,
                    self.axis(),
                    Some(&demo_file_type::ETOMO_DEMO_PLUGIN_AUTODOC),
                )
            }?;
            self.autodoc.set(autodoc as *const Autodoc);
        }
        let autodoc = self.autodoc.get();
        if autodoc.is_null() {
            return Ok(None);
        }
        // SAFETY: see above.
        Ok(Some(unsafe { &*autodoc } as &'static dyn ReadOnlyAutodoc))
    }

    /// Java package-private `getDemoScreenState()`.
    pub fn get_demo_screen_state(&self) -> Option<Arc<DemoScreenState>> {
        self.screen_state.borrow().clone()
    }

    // Updates done

    /// Java package-private `getProcessResultDisplayFactory()`.
    pub fn get_process_result_display_factory(&self) -> Rc<DemoProcessResultDisplayFactory> {
        let existing = self.process_result_display_factory.borrow().clone();
        if let Some(factory) = existing {
            return factory;
        }
        // `manager` is set by `initTomoGenMethod`, which runs before any panel asks for
        // the factory.
        let manager = self
            .manager
            .get()
            .expect("initTomoGenMethod sets the manager before the panel is built");
        let factory = DemoProcessResultDisplayFactory::get_instance(
            manager,
            self.axis_id.get(),
            self.axis_type.get(),
        );
        *self.process_result_display_factory.borrow_mut() = Some(factory.clone());
        factory
    }

    /// Java package-private `updateEtomoPluginDemoCom(boolean, boolean)`.  Updates
    /// demosetup.com.  Created demosetup.com if it is not there, and required is true.
    pub fn update_etomo_plugin_demo_com(
        &self,
        required: bool,
        do_validation: bool,
    ) -> Option<EtomoPluginDemoParam> {
        let manager = self.manager.get()?;
        let panel = self.panel()?;
        let com_script_mgr = self.com_script_mgr.borrow().clone()?;
        let axis_id = self.axis();

        let demo_setup_com_file =
            demo_file_type::DEMO_COMSCRIPT.get_file(Some(manager), Some(axis_id))?;
        if !demo_setup_com_file.exists() {
            if !required {
                return None;
            }
            // Calling makecomfile is unnecessary. Just create the file.
            BaseProcessManager::touch(
                &crate::imod::etomo::util::utilities::java_io_file_get_absolute_path(
                    &demo_setup_com_file.to_string_lossy(),
                ),
                Some(manager),
            );
            com_script_mgr.load_demo();
        }
        let mut param = com_script_mgr.get_etomo_demo_plugin_param();
        if !panel.get_parameters_param(&mut param, do_validation) {
            return None;
        }
        com_script_mgr.save_demo(&param, axis_id);
        Some(param)
    }

    /// Java package-private `updateValues(int)`.
    pub fn update_values(&self, sleep_time: i32) {
        if let Some(panel) = self.panel() {
            panel.set_sleep_time_used(sleep_time);
            // How to get the number of CPUs selected from the parallel panel
            // Instead of using a getParameters function, just get the number of CPUs.
            let parallel_panel = self
                .manager
                .get()
                .and_then(|manager| manager.get_main_panel())
                .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(self.axis()));
            if let Some(parallel_panel) = parallel_panel {
                panel.set_cpus(parallel_panel.get_number_of_processors(false).as_deref());
            }
        }
    }

    /// Java package-private `unselectDependencies(ProcessResultDisplay,
    /// ProcessSeries)`.
    pub fn unselect_dependencies(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        let Some(manager) = self.manager.get() else {
            return;
        };
        let axis_id = self.axis();
        let process_result_display = match process_result_display {
            Some(process_result_display) => process_result_display,
            None => self
                .get_process_result_display_factory()
                .get_unselect_dependencies(),
        };
        manager.start_progress_bar(Some("Unsetting dependents"), Some(axis_id), None);
        // Must tell the button that the process is starting.
        process_result_display.msg_process_starting();
        // This causes dependent buttons to be unselected.
        process_result_display.msg_process_result(ProcessResult::Succeeded);
        manager.stop_progress_bar(Some(axis_id));
        if let Some(process_series) = process_series {
            // This continues the process series.
            ProcessSeries::start_next_process(&process_series, axis_id);
        }
    }

    /// Java package-private `etomoPluginDemo(ProcessResultDisplay, ProcessSeries,
    /// Deferred3dmodButton, Run3dmodMenuOptions)`.  Updates an etomo com file.  Updates
    /// a demo com file.  Runs the demo com file.  Adds processes to a process series.
    pub fn etomo_plugin_demo(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let Some(manager) = self.manager.get() else {
            return;
        };
        let axis_id = self.axis();
        let dialog_type = self
            .dialog_type
            .get()
            .unwrap_or(DialogType::TomogramGeneration);
        if let Some(process_result_display) = &process_result_display {
            process_result_display.msg_process_starting();
        }
        // Demonstration of using manager's functionality
        let tilt_display = self
            .expert
            .borrow()
            .upgrade()
            .and_then(|expert| expert.get_tilt_display());
        if manager
            .update_tilt_com_tilt_display_axis_id_boolean(tilt_display.as_deref(), axis_id, true)
            .is_none()
        {
            // The Java reports the failure and carries on (no return); kept native.
            if let Some(process_result_display) = &process_result_display {
                process_result_display.msg_process_result(ProcessResult::FailedToStart);
            }
        }
        let Some(mut param) = self.update_etomo_plugin_demo_com(true, true) else {
            if let Some(process_result_display) = &process_result_display {
                process_result_display.msg_process_result(ProcessResult::FailedToStart);
            }
            return;
        };
        manager.set_process_state(ProcessState::InProgress, axis_id, dialog_type);
        let mut sleep_time = EtomoNumber::new();
        sleep_time.set_string(
            self.panel()
                .and_then(|panel| panel.get_sleep_time_used())
                .as_deref(),
        );
        let mut start_series = false;
        // The initiator of the sequence will create the processSeries.
        let process_series = match process_series {
            None => {
                start_series = true;
                ProcessSeries::new(manager, axis_id, Some(dialog_type), Some("etomoPluginDemo"))
            }
            Some(process_series) => {
                if !sleep_time.is_null() && sleep_time.gt_int(1) {
                    // Reduce sleep time each time etomoPluginDemo is run
                    param.set_sleep_time_int(sleep_time.get_int() / 2);
                } else {
                    // shouldn't happen
                    return;
                }
                process_series
            }
        };
        let message = param.get_message();
        let sleep_time_gt_1 = param.is_sleep_time_gt(1);
        let process_mgr = self
            .process_mgr
            .get()
            .expect("init creates the process manager");
        let thread_name = match process_mgr.etomo_plugin_demo(
            process_result_display
                .clone()
                .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef),
            Some(Arc::new(EdtRef::new(process_series.clone()))),
            param,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let panel = self.panel();
                ui_harness::INSTANCE.with(|harness| {
                    harness.open_message_dialog_base_manager_ui_component_string_string_axis_id(
                        Some(manager),
                        panel.as_deref().map(|panel| panel as &dyn UIComponent),
                        &format!(
                            "Can not execute {}{}",
                            *demo_process_name::ETOMO_PLUGIN_DEMO,
                            e
                        ),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                if let Some(process_result_display) = &process_result_display {
                    process_result_display.msg_process_result(ProcessResult::FailedToStart);
                }
                return;
            }
        };
        // Use processSeries to run multiple commands in sequence.
        if start_series && message.to_lowercase().contains("unselect") {
            // ProcessSeries will use the manager's startNextProcess function, unless
            // this instance is set as the next process target.
            process_series.borrow_mut().set_last_process_target_task(
                self.this() as Rc<dyn NextProcessTarget>,
                Rc::new(Task::UnselectButtons),
            );
        }
        if sleep_time_gt_1 {
            process_series.borrow_mut().set_next_process_target_task(
                self.this() as Rc<dyn NextProcessTarget>,
                Rc::new(Task::EtomoPluginDemo),
            );
        }
        // This will open a file using 3dmod at the end of the process series. This is
        // done with the right-click menu in the etomoPluginDemo button.
        if deferred_3dmod_button.is_some() {
            process_series
                .borrow_mut()
                .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        }
        manager.set_thread_name(Some(&thread_name), Some(axis_id));
        manager.start_progress_bar(
            Some(&format!(
                "Running {}",
                *demo_process_name::ETOMO_PLUGIN_DEMO
            )),
            Some(axis_id),
            Some(*demo_process_name::ETOMO_PLUGIN_DEMO),
        );
    }

    /// Java package-private `open3dmod(File, boolean, Run3dmodMenuOptions)`.
    pub fn open_3dmod(
        &self,
        file: Option<&Path>,
        swap_yz: bool,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let Some(file) = file else {
            return;
        };
        let (Some(manager), Some(imod_manager)) = (self.manager.get(), self.imod_manager.get())
        else {
            return;
        };
        let axis_id = self.axis();
        let result = (|| -> Result<(), ImodManagerException> {
            imod_manager.set_swap_yz_string_axis_id_boolean(
                demo_imod_manager::DEMO_KEY,
                Some(axis_id),
                swap_yz,
            )?;
            imod_manager.set_file_string_axis_id_file(
                demo_imod_manager::DEMO_KEY,
                Some(axis_id),
                Some(file),
            )?;
            imod_manager.open_string_run3dmod_menu_options(
                demo_imod_manager::DEMO_KEY,
                run_3dmod_menu_options,
            )
        })();
        match result {
            Ok(()) => {}
            // catch (SystemProcessException | AxisTypeException | IOException e)
            Err(
                e @ (ImodManagerException::SystemProcess(_)
                | ImodManagerException::AxisType(_)
                | ImodManagerException::Io(_)),
            ) => {
                eprintln!("{e}");
                let panel = self.panel();
                ui_harness::INSTANCE.with(|harness| {
                    harness.open_message_dialog_base_manager_ui_component_string_string_axis_id(
                        Some(manager),
                        panel.as_deref().map(|panel| panel as &dyn UIComponent),
                        &format!(
                            "{}\nCan't open {}",
                            e,
                            crate::imod::etomo::util::utilities::java_io_file_get_absolute_path(
                                &file.to_string_lossy()
                            )
                        ),
                        "Cannot Open 3dmod",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }
}

impl Plugin for DemoPluginManager {
    /// Java `getKey()`.
    fn get_key(&self) -> Option<String> {
        Some("etomo.plugin.demo.DemoPluginManager".to_string())
    }

    /// Java `getTitle()`.
    fn get_title(&self) -> Option<String> {
        Some("Etomo Demo of TomoGenMethodPlugin".to_string())
    }

    /// Java `getVersion()`.
    fn get_version(&self) -> Option<String> {
        Some("1.0".to_string())
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some("Demo of TomoGenMethodPlugin interface".to_string())
    }

    /// Java `init(BaseManager, AxisID, AxisType, DialogType)`.
    fn init(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        axis_type: Option<AxisType>,
        dialog_type: Option<DialogType>,
    ) {
        if axis_type.is_none()
            || axis_type == Some(AxisType::NotSet)
            || axis_type == Some(AxisType::SingleAxis)
        {
            self.axis_id.set(Some(AxisID::Only));
        } else {
            self.axis_id.set(axis_id);
        }
        match dialog_type {
            None => self.dialog_type.set(Some(DialogType::TomogramGeneration)),
            Some(dialog_type) => self.dialog_type.set(Some(dialog_type)),
        }
        if dialog_type != Some(DialogType::TomogramGeneration) {
            eprintln!("Error: this plugin only works with the tomogram generation dialog");
            // Thread.dumpStack(): no Rust counterpart for a Java stack dump.
        }
        // The Java passes the parameter `axisID` here (not the field).
        *self.com_script_mgr.borrow_mut() = Some(Rc::new(DemoComScriptManager::new(
            manager, axis_id, axis_type,
        )));
        self.process_mgr.set(Some(DemoProcessManager::new(
            manager,
            axis_id.unwrap_or(AxisID::Only),
            Arc::new(EdtRef::new(self.this())),
        )));
        self.imod_manager.set(Some(DemoImodManager::new(manager)));
    }

    /// Java `setParameters()`.
    fn set_parameters(&self) {
        let com_script_mgr = self.com_script_mgr.borrow().clone();
        let (Some(com_script_mgr), Some(panel)) = (com_script_mgr, self.panel()) else {
            return;
        };
        let param = if com_script_mgr.load_demo() {
            com_script_mgr.get_etomo_demo_plugin_param()
        } else {
            // create an empty param to get the defaults.
            EtomoPluginDemoParam::new(self.axis())
        };
        panel.set_parameters_param(&param);
    }

    /// Java `save()`.
    fn save(&self) {
        self.update_etomo_plugin_demo_com(false, false);
    }
}

impl TomoGenMethodPlugin for DemoPluginManager {
    /// Java `initTomoGenMethod(ApplicationManager, TomogramGenerationExpert)`.
    fn init_tomo_gen_method(
        &self,
        manager: &'static ApplicationManager,
        expert: Weak<TomogramGenerationExpert>,
    ) {
        self.manager.set(Some(manager));
        *self.expert.borrow_mut() = expert;
        *self.screen_state.borrow_mut() = Some(DemoScreenState::get_instance(manager, self.axis()));
    }

    /// Java `getPanel(TomogramGenerationDialog, GlobalExpandButton)`.
    fn get_panel(
        &self,
        parent: Weak<TomogramGenerationDialog>,
        btn_advanced_dialog: &Rc<GlobalExpandButton>,
    ) -> Option<Rc<dyn PluginPanel>> {
        let manager = self.manager.get()?;
        let panel = DemoPanel::get_instance(
            &self.this(),
            manager,
            self.axis(),
            self.dialog_type.get(),
            btn_advanced_dialog,
            parent,
        );
        *self.panel.borrow_mut() = Some(panel.clone());
        Some(panel as Rc<dyn PluginPanel>)
    }

    /// Java `hasCustomTiltPanel()`.
    fn has_custom_tilt_panel(&self) -> bool {
        true
    }

    /// Java `getTiltPanel(TomogramGenerationParent, GlobalExpandButton)`.
    fn get_tilt_panel(
        &self,
        parent: Weak<dyn TomogramGenerationParent>,
        btn_advanced_dialog: &Rc<GlobalExpandButton>,
    ) -> Option<Rc<dyn TiltPanelVirtual>> {
        let manager = self.manager.get()?;
        Some(DemoTiltPanel::get_instance(
            manager,
            self.axis(),
            self.dialog_type
                .get()
                .unwrap_or(DialogType::TomogramGeneration),
            btn_advanced_dialog,
            parent,
        ) as Rc<dyn TiltPanelVirtual>)
    }
}

impl NextProcessTarget for DemoPluginManager {
    /// Java `startNextProcess(UIComponent, AxisID, ProcessSeries.Process,
    /// ProcessResultDisplay, ProcessSeries, DialogType, ProcessDisplay)`.
    fn start_next_process(
        &self,
        _ui_component: Option<Rc<dyn UiComponent>>,
        _axis_id: AxisID,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: &ProcessSeriesHandle,
        _dialog_type: Option<DialogType>,
        _display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool {
        if self.panel().is_none() {
            return false;
        }
        if process.equals_task(&Task::EtomoPluginDemo) {
            self.etomo_plugin_demo(
                process_result_display.map(|display| display.get().clone()),
                Some(process_series.clone()),
                None,
                None,
            );
            return true;
        }
        if process.equals_task(&Task::UnselectButtons) {
            self.unselect_dependencies(None, Some(process_series.clone()));
            return true;
        }
        false
    }
}

/// Java private static final class `Task implements TaskInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Task {
    /// Java `UNSELECT_BUTTONS = new Task("UNSELECT_BUTTONS")`.
    UnselectButtons,
    /// Java `ETOMO_PLUGIN_DEMO = new Task("ETOMO_PLUGIN_DEMO")`.
    EtomoPluginDemo,
}

impl Task {
    /// Java private final field `string`.
    fn string(self) -> &'static str {
        match self {
            Task::UnselectButtons => "UNSELECT_BUTTONS",
            Task::EtomoPluginDemo => "ETOMO_PLUGIN_DEMO",
        }
    }
}

impl TaskInterface for Task {
    /// Java `getDescr()`.
    fn get_descr(&self) -> Option<String> {
        Some(self.string().to_string())
    }

    /// Java `okToDrop()`.
    fn ok_to_drop(&self) -> bool {
        false
    }
}

/// Java `toString()`.
impl std::fmt::Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.string())
    }
}
