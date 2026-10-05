//! `IMOD/Etomo/src/etomo/ui/swing/TomogramGenerationExpert.java`.
//!
//! Java `public final class TomogramGenerationExpert extends ReconUIExpert`.
//! The superclass is the embedded [`ReconUIExpert`] `base` (reached through
//! `Deref`); the abstract methods are [`ReconUIExpertVirtual`] and the
//! interface is [`UIExpert`].  The expert lives on the event dispatch thread
//! as an `Rc`; every method takes `&self`, the `dialog` field is a
//! `RefCell<Option<Rc<..>>>` cloned out before each use, and no borrow is
//! held across a call to the manager or back into the dialog.
//!
//! Java private final `comScriptMgr` (`manager.getComScriptManager()`) is not
//! stored: the manager hands out a guard that must only be held for the
//! statement that uses it, so each Java `comScriptMgr.x(...)` is
//! `self.manager.get_com_script_manager().x(...)`.
//!
//! The `*Display` values the dialog hands out (`getTiltDisplay` and the
//! others used by `saveDialog`) are `Rc<dyn ...Display>`; the manager takes
//! them as `&dyn ...Display`.

use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::main_tomogram_panel::MainTomogramPanel;
use super::process_dialog::{DialogExitState, ProcessDialogVirtual};
use super::process_display::ProcessDisplay;
use super::recon_ui_expert::{ReconUIExpert, ReconUIExpertVirtual};
use super::tilt_display::TiltDisplay;
use super::tomogram_generation_dialog::TomogramGenerationDialog;
use super::ui_expert::UIExpert;
use super::ui_expert_utilities::UIExpertUtilities;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;
use crate::imod::etomo::comscript::processchunks_param::OutputImageFileKey;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::plugin::plugin_factory::PluginFactory;
use crate::imod::etomo::plugin::tomo_gen_method_plugin::TomoGenMethodPlugin;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process_series::{Process, ProcessSeriesHandle};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::process_track::ProcessTrack;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `public static final String SIRT_DONE`.
pub const SIRT_DONE: &str = "sirtDone";

/// Java `public final class TomogramGenerationExpert extends ReconUIExpert`.
pub struct TomogramGenerationExpert {
    /// The `ReconUIExpert` superclass.
    base: ReconUIExpert,
    /// The Java `this` handed to `TomogramGenerationDialog.getInstance`.
    this: Weak<TomogramGenerationExpert>,
    /// Java private final `state`.
    state: &'static TomogramState,
    /// Java private final `screenState`.
    screen_state: &'static ReconScreenState,
    /// Java private `dialog`, initialised to null.
    dialog: RefCell<Option<Rc<TomogramGenerationDialog>>>,
    /// Java private `advanced`, initialised to false.
    advanced: Cell<bool>,
    /// Java private `getBinningFromNewst`, initialised to true (never read in
    /// the Java).
    get_binning_from_newst: Cell<bool>,
    /// Java private `methodPlugin`, initialised to null.
    method_plugin: RefCell<Option<Rc<dyn TomoGenMethodPlugin>>>,
}

impl Deref for TomogramGenerationExpert {
    type Target = ReconUIExpert;
    fn deref(&self) -> &ReconUIExpert {
        &self.base
    }
}

impl TomogramGenerationExpert {
    /// Java `TomogramGenerationExpert(ApplicationManager, MainTomogramPanel,
    /// ProcessTrack, AxisID)`.
    pub fn new(
        manager: &'static ApplicationManager,
        main_panel: Option<Rc<MainTomogramPanel>>,
        process_track: Option<&'static ProcessTrack>,
        axis_id: AxisID,
    ) -> Rc<TomogramGenerationExpert> {
        let instance = Rc::new_cyclic(|this| TomogramGenerationExpert {
            // super(manager, mainPanel, processTrack, axisID,
            // DialogType.TOMOGRAM_GENERATION)
            base: ReconUIExpert::new(
                manager,
                main_panel,
                process_track,
                axis_id,
                DialogType::TomogramGeneration,
            ),
            this: this.clone(),
            // comScriptMgr = manager.getComScriptManager(): see the module docs.
            state: manager.get_state(),
            screen_state: manager.get_screen_state(axis_id),
            dialog: RefCell::new(None),
            advanced: Cell::new(false),
            get_binning_from_newst: Cell::new(true),
            method_plugin: RefCell::new(None),
        });
        let this: Weak<dyn ReconUIExpertVirtual> =
            Rc::downgrade(&instance) as Weak<dyn ReconUIExpertVirtual>;
        instance.base.set_this(this);
        instance
    }

    /// The Java `dialog` field read (null is `None`), cloned out so no borrow
    /// of the field is held.
    fn dialog(&self) -> Option<Rc<TomogramGenerationDialog>> {
        self.dialog.borrow().clone()
    }

    /// Java public `msgSirtsetupSucceeded()`.
    pub fn msg_sirtsetup_succeeded(&self) {
        self.sirt_checkpoint();
    }

    /// Java private `sirtCheckpoint()`.
    fn sirt_checkpoint(&self) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        self.manager
            .get_com_script_manager()
            .load_tilt_for_sirt(self.axis_id);
        let tilt_param = self
            .manager
            .get_com_script_manager()
            .get_tilt_param_from_tilt_for_sirt(self.axis_id);
        dialog.sirt_checkpoint(&tilt_param, self.state);
    }

    /// Java public `msgSirtSucceeded(ProcessSeries)`.
    pub fn msg_sirt_succeeded(&self, process_series: Option<ProcessSeriesHandle>) {
        if let Some(dialog) = self.dialog() {
            dialog.msg_sirt_succeeded();
        }
        if let Some(process_series) = &process_series {
            process_series.borrow().end_series();
        }
    }

    /// Java public `reconnectTilt(ProcessName)`.
    pub fn reconnect_tilt(&self, process_name: ProcessName) -> bool {
        let display: ProcessResultDisplayHandle = self
            .manager
            .get_process_result_display_factory(self.axis_id)
            .get_tilt(DialogType::TomogramGeneration);
        self.send_msg_process_starting(Some(&display));
        self.manager
            .reconnect_tilt(self.axis_id, process_name, Some(display))
    }

    /// Java public `setTiltState()`.
    pub fn set_tilt_state(&self) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.set_tilt_state(self.state, self.meta_data);
    }

    /// Java private `setParameters(ConstMetaData)`.
    fn set_parameters_const_meta_data(
        &self,
        meta_data: &dyn crate::imod::etomo::r#type::const_meta_data::ConstMetaData,
    ) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.set_parameters_const_meta_data(meta_data);
    }

    /// Java private final `setParameters(ReconScreenState)`.
    fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.set_parameters_recon_screen_state(screen_state);
    }

    /// Java private `getParameters(ReconScreenState)`.
    fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.get_parameters_recon_screen_state(screen_state);
    }

    /// Java private `getParameters(MetaData) throws
    /// FortranInputSyntaxException`.
    fn get_parameters_meta_data(
        &self,
        meta_data: &crate::imod::etomo::r#type::meta_data::MetaData,
    ) -> Result<
        (),
        crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException,
    > {
        let Some(dialog) = self.dialog() else {
            return Ok(());
        };
        dialog.get_parameters_meta_data(meta_data)
    }

    /// Java private `setParameters(ConstTiltParam, boolean)`.  Set the UI
    /// parameters with the specified tiltParam values.  WARNING: be sure the
    /// setNewstParam is called first so the binning value for the stack is
    /// known.  The thickness, first and last slice, width and x,y,z offsets
    /// are scaled so that they are represented to the user in unbinned
    /// dimensions.
    fn set_parameters_const_tilt_param_boolean(
        &self,
        tilt_param: &dyn crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam,
        initialize: bool,
    ) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.set_parameters_const_tilt_param_boolean(tilt_param, initialize);
    }

    /// Java private `setParameters(MultifiltSetupParam)`.
    fn set_parameters_multifilt_setup_param(
        &self,
        param: &crate::imod::etomo::comscript::multifilt_setup_param::MultifiltSetupParam,
    ) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.set_parameters_multifilt_setup_param(param);
    }

    /// Java private `setParameters(Ctf3dSetupParam)`.
    fn set_parameters_ctf3d_setup_param(
        &self,
        param: &crate::imod::etomo::comscript::ctf3d_setup_param::Ctf3dSetupParam,
    ) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.set_parameters_ctf3d_setup_param(param);
    }

    /// Java private `setParameters(SirtsetupParam)`.
    fn set_parameters_sirtsetup_param(
        &self,
        param: &crate::imod::etomo::comscript::sirtsetup_param::SirtsetupParam,
    ) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.set_parameters_sirtsetup_param(param);
    }

    /// Java public `getTiltDisplay()`; `None` is Java null.
    pub fn get_tilt_display(&self) -> Option<Rc<dyn TiltDisplay>> {
        let dialog = self.dialog()?;
        Some(dialog.get_tilt_display())
    }
}

impl ReconUIExpertVirtual for TomogramGenerationExpert {
    /// Java package-private override `doneDialog()`.
    fn done_dialog_void(&self) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        let exit_state = dialog.get_exit_state();
        if exit_state == DialogExitState::Execute {
            self.manager.close_imods(
                Some(imod_manager::TRIAL_TOMOGRAM_KEY),
                self.axis_id,
                Some("Trial tomogram"),
            );
        }
        if exit_state != DialogExitState::Cancel {
            self.save_dialog_void();
        }
        // Clean up the existing dialog
        self.leave_dialog(exit_state);
        // Hold onto the finished dialog in case anything is running that needs it or
        // there are next processes that need it.
    }

    /// Java package-private override `saveDialog()`.
    fn save_dialog_void(&self) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        self.advanced.set(dialog.is_advanced());
        // Get the user input data from the dialog box
        self.get_parameters_recon_screen_state(self.screen_state);
        let tilt_display = dialog.get_tilt_display();
        self.manager.update_tilt_com_tilt_display_axis_id_boolean(
            Some(&*tilt_display),
            self.axis_id,
            false,
        );
        let multifilt_setup_display = dialog.get_multifilt_setup_display();
        self.manager.update_multifilt_setup_com(
            Some(&*multifilt_setup_display),
            self.axis_id,
            false,
        );
        let ctf3d_setup_display = dialog.get_ctf3d_setup_display();
        self.manager
            .update_ctf3d_setup_com(ctf3d_setup_display.as_deref(), self.axis_id, false);
        if let Err(e) = self.get_parameters_meta_data(self.meta_data) {
            let manager: &'static dyn BaseManager = self.manager;
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(manager),
                    e.get_message().unwrap_or("null"),
                    "Data File Error",
                )
            });
        }
        let sirtsetup_display = dialog.get_sirtsetup_display();
        self.manager
            .update_sirt_setup_com(self.axis_id, &*sirtsetup_display, false);
        let method_plugin = self.method_plugin.borrow().clone();
        if let Some(method_plugin) = method_plugin {
            method_plugin.save();
        }
        self.manager.save_storables(Some(self.axis_id));
    }

    /// Java package-private override `getDialog()`.
    fn get_dialog(&self) -> Option<Rc<dyn ProcessDialogVirtual>> {
        self.dialog()
            .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>)
    }
}

impl UIExpert for TomogramGenerationExpert {
    /// Java public override `openDialog()`.  Open the tomogram generation
    /// dialog.
    fn open_dialog(&self) {
        if !self.can_show_dialog() {
            return;
        }
        let manager = self.manager;
        let axis_id = self.axis_id;
        let meta_data = self.meta_data;
        let action_message =
            BaseManager::set_current_dialog_type(manager, Some(self.dialog_type), Some(axis_id));
        let existing = self.dialog();
        if self.show_dialog(
            existing.as_ref().map(|dialog| dialog.process_dialog()),
            action_message.as_deref(),
        ) {
            return;
        }
        // Create the dialog and show it.
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("TomogramGenerationDialog"),
            Some(utilities::STARTED_STATUS),
        );
        if self.method_plugin.borrow().is_none() {
            let main_panel = manager.get_main_panel();
            let ui_component: Option<&dyn UIComponent> = main_panel
                .as_ref()
                .map(|main_panel| &**main_panel.main_panel() as &dyn UIComponent);
            let method_plugin = if etomo_director::ARGUMENTS.lock().unwrap().is_plugin() {
                PluginFactory::load_demo_plugin(ui_component)
            } else {
                PluginFactory::load_tomo_gen_method_plugin(ui_component)
            };
            if let Some(method_plugin) = &method_plugin {
                method_plugin.init(
                    manager,
                    Some(axis_id),
                    manager
                        .get_base_meta_data()
                        .map(|meta_data| meta_data.base().get_axis_type()),
                    Some(self.dialog_type),
                );
                method_plugin.init_tomo_gen_method(manager, self.this.clone());
            }
            *self.method_plugin.borrow_mut() = method_plugin;
        }
        let method_plugin = self.method_plugin.borrow().clone();
        let dialog = TomogramGenerationDialog::get_instance(
            manager,
            self.this.clone(),
            axis_id,
            method_plugin.as_ref(),
        );
        *self.dialog.borrow_mut() = Some(dialog.clone());
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("TomogramGenerationDialog"),
            Some(utilities::FINISHED_STATUS),
        );
        // no longer managing image size
        self.set_parameters_const_meta_data(meta_data);
        self.set_parameters_recon_screen_state(self.screen_state);
        // load SIRT first because the resume radio buttons disable tilt fields
        if !manager.get_com_script_manager().load_sirtsetup(axis_id) {
            let mut makecom_file_param = MakecomfileParam::new(
                manager,
                axis_id,
                file_type::CLASS.sirtsetup_comscript.clone(),
            );
            manager.makecomfile(axis_id, &mut makecom_file_param);
            manager.get_com_script_manager().load_sirtsetup(axis_id);
        }
        let sirtsetup_param = manager
            .get_com_script_manager()
            .get_sirtsetup_param(axis_id);
        self.set_parameters_sirtsetup_param(&sirtsetup_param);
        if let Some(method_plugin) = &method_plugin {
            method_plugin.set_parameters();
        }
        // Read in the tilt{|a|b}.com parameters and display the dialog panel
        manager.get_com_script_manager().load_tilt(axis_id);
        let mut tilt_param = manager.get_com_script_manager().get_tilt_param(axis_id);
        tilt_param.set_fiducialess(meta_data.is_fiducialess(axis_id));
        // If this is a montage, then binning can only be 1, so no need to upgrade
        if meta_data.get_view_type() != ViewType::Montage {
            // upgrade and save param to comscript
            UIExpertUtilities::INSTANCE.upgrade_old_tilt_com(manager, axis_id, &mut tilt_param);
        }
        let gen_exists = meta_data.is_gen_exists(axis_id);
        self.set_parameters_const_tilt_param_boolean(&tilt_param, !gen_exists);
        self.sirt_checkpoint();
        // Set the fidcialess state and tilt axis angle
        self.set_tilt_state();
        meta_data.set_gen_exists(axis_id, true);
        // MultifiltSetup
        if !manager
            .get_com_script_manager()
            .load_multifilt_setup(axis_id, false)
        {
            // `FileType.MULTIFILT_SETUP_COMSCRIPT.getFile(manager, axisID)
            // .getAbsolutePath()`.
            if let Some(file) = file_type::CLASS
                .multifilt_setup_comscript
                .get_file(Some(manager), Some(axis_id))
            {
                let path = std::path::absolute(&file).unwrap_or(file);
                BaseProcessManager::touch(&path.to_string_lossy(), Some(manager));
            }
            manager
                .get_com_script_manager()
                .load_multifilt_setup(axis_id, true);
        }
        let multifilt_setup_param = manager
            .get_com_script_manager()
            .get_multifilt_setup_param(axis_id);
        self.set_parameters_multifilt_setup_param(&multifilt_setup_param);
        // comfiles used by ctf3d
        manager
            .get_com_script_manager()
            .load_ctf_correction(axis_id, false);
        manager
            .get_com_script_manager()
            .load_gold_eraser(axis_id, false);
        manager.get_com_script_manager().load_mtf_filter(axis_id);
        // Ctf3dSetup
        if !manager
            .get_com_script_manager()
            .load_ctf3d_setup(axis_id, false)
        {
            let mut param = MakecomfileParam::new(
                manager,
                axis_id,
                file_type::CLASS.ctf_3d_setup_comscript.clone(),
            );
            if file_type::CLASS
                .tilt_comscript
                .exists(Some(manager), Some(axis_id))
            {
                param.set_input_file(
                    file_type::CLASS
                        .tilt_comscript
                        .get_file_name(Some(manager), Some(axis_id))
                        .as_deref(),
                );
                manager.makecomfile(axis_id, &mut param);
            } else if let Some(file) = file_type::CLASS
                .ctf_3d_setup_comscript
                .get_file(Some(manager), Some(axis_id))
            {
                let path = std::path::absolute(&file).unwrap_or(file);
                BaseProcessManager::touch(&path.to_string_lossy(), Some(manager));
            }
            manager
                .get_com_script_manager()
                .load_ctf3d_setup(axis_id, true);
        }
        let ctf3d_setup_param = manager
            .get_com_script_manager()
            .get_ctf3d_setup_param(axis_id);
        self.set_parameters_ctf3d_setup_param(&ctf3d_setup_param);
        self.open_dialog_process_dialog_string(dialog.process_dialog(), action_message.as_deref());
    }

    /// Java public override `startNextProcess(ProcessSeries.Process,
    /// ProcessResultDisplay, ProcessSeries, DialogType, ProcessDisplay)`.
    /// Start the next process specified by the nextProcess string.  Returns
    /// true if the process is recognized.
    fn start_next_process(
        &self,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        _dialog_type: Option<DialogType>,
        _display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool {
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        if process.equals_string(Some(&ProcessName::PROCESSCHUNKS.to_string())) {
            // `process.getOutputImageFileKey()`.
            let output_image_file_key = process
                .get_output_image_file_key()
                .cloned()
                .map(OutputImageFileKey::FileKey);
            // `processchunks(manager, dialog, ...)`: the dialog is passed as the
            // `AbstractParallelDialog` its `ProcessDialog` superclass is.
            let dialog = self.dialog();
            let parallel_dialog: Option<&dyn AbstractParallelDialog> = dialog
                .as_ref()
                .map(|dialog| dialog.process_dialog() as &dyn AbstractParallelDialog);
            if process.get_subprocess_name() == Some(ProcessName::TILT) {
                self.processchunks(
                    manager,
                    parallel_dialog,
                    process_result_display,
                    process_series,
                    &format!("{}{}", ProcessName::TILT, axis_id.get_extension()),
                    output_image_file_key,
                    process.get_processing_method(),
                    false,
                );
                return true;
            }
            if process.get_subprocess_name() == Some(ProcessName::TILT_MULTIFILT) {
                self.processchunks(
                    manager,
                    parallel_dialog,
                    process_result_display,
                    process_series,
                    &format!("{}{}", ProcessName::TILT_MULTIFILT, axis_id.get_extension()),
                    output_image_file_key,
                    process.get_processing_method(),
                    false,
                );
                return true;
            }
            if process.get_subprocess_name() == Some(ProcessName::CTF_3D) {
                self.processchunks(
                    manager,
                    parallel_dialog,
                    process_result_display,
                    process_series,
                    &format!("{}{}", ProcessName::CTF_3D, axis_id.get_extension()),
                    output_image_file_key,
                    process.get_processing_method(),
                    false,
                );
                return true;
            }
            self.processchunks(
                manager,
                parallel_dialog,
                process_result_display,
                process_series,
                &format!("{}{}_sirt", ProcessName::TILT, axis_id.get_extension()),
                output_image_file_key,
                process.get_processing_method(),
                false,
            );
            return true;
        }
        if process.equals_string(Some(SIRT_DONE)) {
            self.msg_sirt_succeeded(process_series);
            return true;
        }
        false
    }

    /// Java inherited final `ReconUIExpert.saveAction()`.
    fn save_action(&self) {
        self.base.save_action();
    }

    /// Java inherited final `ReconUIExpert.saveDialog(DialogExitState)`.
    fn save_dialog(&self, exit_state: DialogExitState) {
        self.base.save_dialog_dialog_exit_state(exit_state);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
