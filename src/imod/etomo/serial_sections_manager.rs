//! `IMOD/Etomo/src/etomo/SerialSectionsManager.java`.
//!
//! Manager for the Serial Sections interface.  This is deliberately source-shaped:
//! methods retain the Java control-flow boundaries and Java overloads have descriptive
//! Rust suffixes.  The process, comscript and Swing classes named by this unit have not
//! yet been translated, so their nullable Java references use `Option<Infallible>`.
//! That represents only Java `null`; it does not fabricate an executable substitute.
#![allow(dead_code)]

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::view_type::ViewType;

/// Java private static final `AXIS_ID`.
const AXIS_ID: AxisID = AxisID::Only;

/// Java `SerialSectionsManager`.
pub struct SerialSectionsManager {
    /// Java superclass `BaseManager` state.
    base: BaseManagerBase,
    /// `SerialSectionsMetaData.getName()` until its typed metadata unit is
    /// translated; required immediately by EtomoDirector manager registration.
    name: String,
    /// Java final `state = new SerialSectionsState()`.
    // TODO(unit): etomo/type/SerialSectionsState.java.
    state: Option<Infallible>,
    /// Java final `metaData`.
    // TODO(unit): etomo/type/SerialSectionsMetaData.java.
    meta_data: Option<Infallible>,
    /// Java final `processMgr`.
    // TODO(unit): etomo/process/SerialSectionsProcessManager.java.
    process_mgr: Option<Infallible>,
    /// Java `startupDialog`, initially null.
    // TODO(unit): etomo/ui/swing/SerialSectionsStartupDialog.java.
    startup_dialog: Mutex<Option<Infallible>>,
    /// Java `dialog`, initially null.
    // TODO(unit): etomo/ui/swing/SerialSectionsDialog.java.
    dialog: Mutex<Option<Infallible>>,
    /// Java `autoAlignmentController`, initially null.
    // TODO(unit): etomo/AutoAlignmentController.java.
    auto_alignment_controller: Mutex<Option<Infallible>>,
    /// Java `valid`, initially true.
    valid: Mutex<bool>,
    /// Java `origUserDir`, initially null.
    orig_user_dir: Mutex<Option<String>>,
    /// Java `mainPanel`, initialized in `createMainPanel`.
    // TODO(unit): etomo/ui/swing/MainSerialSectionsPanel.java.
    main_panel: Mutex<Option<Infallible>>,
    /// Java `comScriptMgr`, initialized in `createComScriptManager`.
    // TODO(unit): etomo/comscript/SerialSectionsComScriptManager.java.
    com_script_mgr: Mutex<Option<Infallible>>,
}

impl SerialSectionsManager {
    /// Java private `SerialSectionsManager()`.
    pub fn new() -> &'static SerialSectionsManager {
        Self::new_with_param_file_name(Some(""))
    }

    /// Java package-private `SerialSectionsManager(String)`.
    pub fn new_with_param_file_name(
        param_file_name: Option<&str>,
    ) -> &'static SerialSectionsManager {
        let instance = Box::leak(Box::new(SerialSectionsManager {
            base: BaseManagerBase::initial(),
            name: param_file_name.unwrap_or("Serial Sections").to_owned(),
            state: None,
            meta_data: None,
            process_mgr: None,
            startup_dialog: Mutex::new(None),
            dialog: Mutex::new(None),
            auto_alignment_controller: Mutex::new(None),
            valid: Mutex::new(true),
            orig_user_dir: Mutex::new(None),
            main_panel: Mutex::new(None),
            com_script_mgr: Mutex::new(None),
        }));
        instance.base_manager();
        instance.initialize_ui_parameters_from_name(param_file_name, Some(AXIS_ID));
        instance
    }

    /// Java static `getInstance()`.
    pub fn get_instance() -> &'static SerialSectionsManager {
        let instance = Self::new();
        instance.open_dialog();
        instance
    }

    /// Java static `getInstance(String)`.
    pub fn get_instance_with_param_file_name(
        param_file_name: Option<&str>,
    ) -> &'static SerialSectionsManager {
        let instance = Self::new_with_param_file_name(param_file_name);
        instance.open_dialog();
        instance
    }

    /// Java override `initializeUIParameters(String, AxisID)`.
    pub fn initialize_ui_parameters_from_name(
        &self,
        param_file_name: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        BaseManager::initialize_ui_parameters_from_name(self, param_file_name, axis_id);
        // Java sets System user.dir after a loaded file.  Rust deliberately does not
        // mutate process-global current-directory state; `propertyUserDir` remains the
        // manager-local source equivalent until process execution is translated.
    }

    /// Java private `openDialog`.
    fn open_dialog(&self) {
        // TODO(unit): MainSerialSectionsPanel / dialogs.  Headless branching is held by
        // BaseManager; no dialog can be materialized while these source units are absent.
    }

    /// Java `isStartupPopupOpen`.
    pub fn is_startup_popup_open(&self) -> bool {
        !*self.base.loaded_param_file.lock().unwrap()
    }

    /// Java package-private `display`.
    pub fn display(&self) {
        drop(self.startup_dialog.lock().unwrap());
    }

    /// Java private `openProcessingPanel`.
    fn open_processing_panel(&self) {}

    /// Java private `openSerialSectionsStartupDialog`.
    fn open_serial_sections_startup_dialog(&self) {}

    /// Java `setStartupData(SerialSectionsStartupData)`.
    // TODO(unit): etomo/logic/SerialSectionsStartupData.java and startup dialog.
    pub fn set_startup_data(&self, startup_data: Option<Infallible>) {
        let _ = startup_data;
    }

    /// Java `setParamFile(SerialSectionsStartupData)`.
    // TODO(unit): SerialSectionsStartupData / SerialSectionsMetaData / process manager.
    pub fn set_param_file_from_startup_data(&self, startup_data: Option<Infallible>) -> bool {
        if *self.base.loaded_param_file.lock().unwrap() {
            return true;
        }
        startup_data.is_some() && false
    }

    /// Java `cancelStartup`.
    pub fn cancel_startup(&self) { /* TODO(unit): main panel and EtomoDirector close */
    }

    /// Java private `openSerialSectionsDialog(SerialSectionsStartupData)`.
    // TODO(unit): SerialSectionsDialog / AutoAlignmentController / UI panel.
    fn open_serial_sections_dialog(&self, startup_data: Option<Infallible>) -> Result<(), ()> {
        let _ = startup_data;
        Err(())
    }

    /// Java `completeStartup`.
    // TODO(unit): ProcessSeries / UIComponent.
    pub fn complete_startup(&self, ui_component: Option<Infallible>, axis_id: Option<AxisID>) {
        let _ = (ui_component, axis_id);
    }

    /// Java `preblend`.
    // TODO(unit): ProcessSeries, ProcessResultDisplay, Deferred3dmodButton, Run3dmodMenuOptions.
    pub fn preblend(
        &self,
        process_series: Option<Infallible>,
        process_result_display: Option<Infallible>,
        deferred_3dmod_button: Option<Infallible>,
        axis_id: Option<AxisID>,
        run_3dmod_menu_options: Option<Infallible>,
        dialog_type: Option<DialogType>,
    ) {
        let _ = (
            process_series,
            process_result_display,
            deferred_3dmod_button,
            axis_id,
            run_3dmod_menu_options,
            dialog_type,
        );
    }

    /// Java `midasFixEdges`.
    // TODO(unit): MidasParam, ConstProcessSeries, SerialSectionsProcessManager.
    pub fn midas_fix_edges(&self, axis_id: Option<AxisID>, process_series: Option<Infallible>) {
        let _ = (axis_id, process_series);
    }

    /// Java public `align(AxisID, ProcessResultDisplay, Deferred3dmodButton, Run3dmodMenuOptions)`.
    // TODO(unit): ProcessResultDisplay, Deferred3dmodButton, Run3dmodMenuOptions.
    pub fn align(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        deferred_3dmod_button: Option<Infallible>,
        run_3dmod_menu_options: Option<Infallible>,
    ) {
        let _ = (
            axis_id,
            process_result_display,
            deferred_3dmod_button,
            run_3dmod_menu_options,
        );
    }

    /// Java private `changeDirectory`.
    fn change_directory(&self, axis_id: Option<AxisID>, process_series: Option<Infallible>) {
        let _ = (axis_id, process_series);
    }
    /// Java private `extractpieces`.
    fn extractpieces(
        &self,
        ui_component: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_series: Option<Infallible>,
    ) {
        let _ = (ui_component, axis_id, process_series);
    }
    /// Java private `createComscripts`.
    fn create_comscripts(
        &self,
        ui_component: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_series: Option<Infallible>,
    ) {
        let _ = (ui_component, axis_id, process_series);
    }
    /// Java private `addOutputFormat`.
    fn add_output_format(&self, process_series: Option<Infallible>, axis_id: Option<AxisID>) {
        let _ = (process_series, axis_id);
    }
    /// Java private `xftoxg`.
    fn xftoxg(
        &self,
        process_series: Option<Infallible>,
        process_result_display: Option<Infallible>,
        axis_id: Option<AxisID>,
    ) {
        let _ = (process_series, process_result_display, axis_id);
    }
    /// Java private `align(ProcessSeries, ProcessResultDisplay, AxisID)`.
    fn align_process_series(
        &self,
        process_series: Option<Infallible>,
        process_result_display: Option<Infallible>,
        axis_id: Option<AxisID>,
    ) {
        let _ = (process_series, process_result_display, axis_id);
    }

    /// Java private `updateNewstCom`.
    // TODO(unit): ConstNewstParam / NewstParam / SerialSectionsDialog.
    fn update_newst_com(&self, axis_id: Option<AxisID>, do_validation: bool) -> Option<Infallible> {
        let _ = (axis_id, do_validation);
        None
    }
    /// Java `msgPreblendSucceeded`.
    pub fn msg_preblend_succeeded(&self) {
        drop(self.dialog.lock().unwrap());
    }
    /// Java private `updatePreblendComscript`.
    fn update_preblend_comscript(
        &self,
        axis_id: Option<AxisID>,
        do_validation: bool,
    ) -> Option<Infallible> {
        let _ = (axis_id, do_validation);
        None
    }
    /// Java private `getStateForEdgeFunctionsRecreated`.
    fn get_state_for_edge_functions_recreated(&self) -> bool {
        false
    }
    /// Java `getState`.
    // TODO(unit): SerialSectionsState.
    pub fn get_state(&self) -> Option<Infallible> {
        self.state
    }
    /// Java private `updateBlendComscript`.
    fn update_blend_comscript(
        &self,
        axis_id: Option<AxisID>,
        do_validation: bool,
    ) -> Option<Infallible> {
        let _ = (axis_id, do_validation);
        None
    }
    /// Java private `copyDistortionFieldFile`.
    fn copy_distortion_field_file(
        &self,
        process_series: Option<Infallible>,
        ui_component: Option<Infallible>,
        axis_id: Option<AxisID>,
    ) {
        let _ = (process_series, ui_component, axis_id);
    }
    /// Java private `doneStartupDialog`.
    fn done_startup_dialog(&self, process_series: Option<Infallible>, axis_id: Option<AxisID>) {
        let _ = (process_series, axis_id);
    }
    /// Java private `resetStartupState`.
    fn reset_startup_state(&self, process_series: Option<Infallible>, axis_id: Option<AxisID>) {
        let _ = (process_series, axis_id);
    }

    /// Java `imodRaw`.
    // TODO(unit): BaseImodManager / Run3dmodMenuOptions.
    pub fn imod_raw(&self, axis_id: Option<AxisID>, menu_options: Option<Infallible>) {
        let _ = (axis_id, menu_options);
    }
    /// Java `imodPreblend`.
    pub fn imod_preblend(&self, axis_id: Option<AxisID>, menu_options: Option<Infallible>) {
        let _ = (axis_id, menu_options);
    }
    /// Java `imodPrealign`.
    pub fn imod_prealign(&self, axis_id: Option<AxisID>, menu_options: Option<Infallible>) {
        if BaseManager::get_view_type(self) == ViewType::Montage {
            self.imod_preblend(axis_id, menu_options);
        } else {
            self.imod_raw(axis_id, menu_options);
        }
    }
    /// Java `imodAlign`.
    pub fn imod_align(&self, axis_id: Option<AxisID>, menu_options: Option<Infallible>) {
        let _ = (axis_id, menu_options);
    }

    /// Java private `saveSerialSectionsDialog`.
    fn save_serial_sections_dialog(&self, for_run: bool) -> bool {
        let _ = for_run;
        self.dialog.lock().unwrap().is_some()
    }

    /// Java private `getDistortionField`.
    fn get_distortion_field(&self) -> Option<PathBuf> {
        None
    }
    /// Java `getStack`.
    pub fn get_stack(&self) -> Option<PathBuf> {
        None
    }
    /// Java `getMetaData`.
    // TODO(unit): ConstSerialSectionsMetaData.
    pub fn get_meta_data(&self) -> Option<Infallible> {
        self.meta_data
    }
    /// Java private `setSerialSectionsDialogParameters`.
    fn set_serial_sections_dialog_parameters(&self) {}
    /// Java `getParameters(MidasParam)`.
    pub fn get_parameters_midas(&self, param: Option<Infallible>) {
        let _ = param;
    }
    /// Java `getAutoAlignmentParameters(MidasParam, AxisID)`.
    pub fn get_auto_alignment_parameters_midas(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
    ) {
        let _ = (param, axis_id);
    }
    /// Java `getAutoAlignmentParameters(XfalignParam, AxisID)`.
    pub fn get_auto_alignment_parameters_xfalign(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
    ) {
        let _ = (param, axis_id);
    }
}

impl BaseManager for SerialSectionsManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }
    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }
    /// Java `createComScriptManager`.
    fn create_com_script_manager(&self) { /* TODO(unit): SerialSectionsComScriptManager */
    }
    /// Java `createMainPanel`.
    fn create_main_panel(&self) { /* TODO(unit): MainSerialSectionsPanel */
    }
    /// Java `getBaseMetaData`.
    fn get_base_meta_data(
        &self,
    ) -> Option<&dyn crate::imod::etomo::r#type::base_meta_data::BaseMetaData> {
        None
    }
    /// Java `getInterfaceType`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::SerialSections)
    }
    /// Java `getMainPanel`.
    fn get_main_panel(&self) -> Option<Infallible> {
        *self.main_panel.lock().unwrap()
    }
    /// Java `getProcessManager`.
    fn get_process_manager(&self) -> Option<Infallible> {
        self.process_mgr
    }
    /// Java override `startNextProcess`.
    // TODO(unit): ProcessSeries, ProcessResultDisplay, ProcessDisplay, UIComponent.
    fn start_next_process(
        &self,
        ui_component: Option<Infallible>,
        axis_id: Option<AxisID>,
        process: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        dialog_type: Option<Infallible>,
        display: Option<Infallible>,
    ) -> bool {
        let _ = (
            ui_component,
            axis_id,
            process,
            process_result_display,
            process_series,
            dialog_type,
            display,
        );
        false
    }
    /// Java `getAutoAlignmentMetaData`.
    fn get_auto_alignment_meta_data(&self) -> Option<Infallible> {
        None
    }
    /// Java `updateMetaData`.
    fn update_meta_data(
        &self,
        dialog_type: Option<DialogType>,
        axis_id: Option<AxisID>,
        do_validation: bool,
    ) -> bool {
        let _ = (dialog_type, axis_id, do_validation);
        false
    }
    /// Java `getStorables(int)`.
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Box<dyn Storable>>> {
        let _ = offset;
        None
    }
    /// Java `exitProgram`.
    fn exit_program(&self, axis_id: Option<AxisID>) -> bool {
        if self.exit_program_super(axis_id) {
            self.end_threads();
            self.save_param_file();
            true
        } else {
            false
        }
    }
    /// Java `save`.
    fn save(&self) -> bool {
        self.save_super() && self.save_serial_sections_dialog(false)
    }
    /// Java `getName`.
    fn get_name(&self) -> Option<String> {
        Some(self.name.clone())
    }
    /// Java `getViewType`.
    fn get_view_type(&self) -> ViewType {
        ViewType::DEFAULT
    }
    /// Java `isValid`.
    fn is_valid(&self) -> bool {
        *self.valid.lock().unwrap()
    }
}

/// Java public static final nested `Task`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    ChangeDirectory,
    ExtractPieces,
    CreateComscripts,
    CopyDistortionFieldFile,
    DoneStartupDialog,
    ResetStartupState,
    Xftoxg,
    Align,
    AddOutputFormat,
}
impl TaskInterface for Task {
    /// Java `getDescr`.
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
    /// Java `okToDrop`.
    fn ok_to_drop(&self) -> bool {
        matches!(self, Self::DoneStartupDialog | Self::ResetStartupState)
    }
}
