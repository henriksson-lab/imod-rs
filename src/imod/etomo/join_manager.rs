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
//! constructor leaks its allocation and hands back `&'static JoinManager`, the reference
//! type every `BaseManager` parameter in the tree uses.
//!
//! **Frontier.**  This class names 6 Swing types (`JoinDialog`, `MainJoinPanel`,
//! `MainPanel`, `Deferred3dmodButton`, `ProcessDisplay`, `UIHarness`), the join process
//! boundary (`JoinProcessManager`, `BaseProcessManager`, `ProcessSeries`), fourteen
//! comscript parameter classes and the join metadata/state classes.  None of those has a
//! module, so every member that dereferences one carries a `// TODO(unit):` marker
//! naming the exact file, and a field or parameter whose declared type has no module is
//! `Option<std::convert::Infallible>` - the Rust type with exactly the one inhabitant
//! Java's `null` has, the `etomo/process/emergency_monitor.rs` precedent - so no stub
//! type is invented.  What this module does carry is the class's shape: which members
//! exist, which are overrides, and what each returns.
#![allow(dead_code)]

use std::convert::Infallible;
use std::path::{Path, PathBuf};

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java `JoinManager`.
pub struct JoinManager {
    /// Java superclass `BaseManager` state.
    base: BaseManagerBase,
    /// `JoinMetaData.getName()` until its full typed metadata model is
    /// translated.  The constructor's file name is already a source-owned
    /// identity and is needed by EtomoDirector's manager list immediately.
    name: String,
    /// Java private field `joinDialog`, a process dialog reference which defaults to
    /// null.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java - the field's declared type.
    join_dialog: Option<Infallible>,
    /// Java private field `autoAlignmentController`, which defaults to null.
    // TODO(unit): needs etomo/AutoAlignmentController.java - the field's declared
    // type.
    auto_alignment_controller: Option<Infallible>,
    /// Java private field `mainPanel`, cast from the base class variable and initialized
    /// in the create function.
    // TODO(unit): needs etomo/ui/swing/MainJoinPanel.java - the field's declared type.
    main_panel: Option<Infallible>,
    /// Java private final field `metaData`.
    // TODO(unit): needs etomo/type/JoinMetaData.java - the field's declared type.
    meta_data: Option<Infallible>,
    /// Java private field `processMgr`.
    // TODO(unit): needs etomo/process/JoinProcessManager.java - the field's declared
    // type.
    process_mgr: Option<Infallible>,
    /// Java private field `state`.
    // TODO(unit): needs etomo/type/JoinState.java - the field's declared type.
    state: Option<Infallible>,
    /// Java private field `startJoinParam`, which defaults to null.
    // TODO(unit): needs etomo/comscript/StartJoinParam.java - the field's declared type.
    start_join_param: Option<Infallible>,
    /// Java private final field `screenState`, initialised to
    /// `new JoinScreenState(AxisID.ONLY, AxisType.SINGLE_AXIS)`.
    // TODO(unit): needs etomo/type/JoinScreenState.java - the field's declared type.
    screen_state: Option<Infallible>,
    /// Java private field `debug`, which defaults to false.
    debug: std::sync::Mutex<bool>,
    /// Java private field `comScriptMgr`.
    // TODO(unit): needs etomo/comscript/JoinComscriptManager.java - the field's declared
    // type.
    com_script_mgr: Option<Infallible>,
}

impl JoinManager {
    /// Java package-private `JoinManager(String, AxisID)`.  Java's managers are created
    /// by `EtomoDirector` and live for the run, so the allocation is leaked; `super()`
    /// is the `BaseManager` trait's `base_manager`, which needs the allocation to exist
    /// because it passes `this` on.
    pub fn new(param_file_name: Option<&str>, axis_id: Option<AxisID>) -> &'static JoinManager {
        let instance: &'static JoinManager = Box::leak(Box::new(JoinManager {
            base: BaseManagerBase::initial(),
            name: param_file_name.unwrap_or("New Join").to_owned(),
            join_dialog: None,
            auto_alignment_controller: None,
            main_panel: None,
            meta_data: None,
            process_mgr: None,
            state: None,
            start_join_param: None,
            screen_state: None,
            debug: std::sync::Mutex::new(false),
            com_script_mgr: None,
        }));
        // Java `super()`.
        instance.base_manager();
        // TODO(unit): needs etomo/type/JoinMetaData.java,
        // etomo/process/JoinProcessManager.java, etomo/process/BaseImodManager.java,
        // etomo/ui/swing/MainJoinPanel.java and etomo/comscript/JoinComscriptManager.java
        // - the rest of the constructor builds `metaData`, calls `createState()`, builds
        // `processMgr`, calls `initializeUIParameters(paramFileName, axisID)`, and on a
        // loaded param file calls `imodManager.setMetaData(metaData)` and
        // `mainPanel.setStatusBarText(paramFile, metaData, logWindow)`; when not
        // headless it calls `openJoinDialog()` and `setMode()`, and it ends by building
        // `comScriptMgr`.
        let _ = (param_file_name, axis_id);
        instance
    }

    /// Java `openJoinDialog`.  Open the join dialog.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java,
    // etomo/AutoAlignmentController.java and etomo/ui/swing/MainJoinPanel.java -
    // the body builds the dialog and the controller, calls
    // `autoAlignmentController.createEmptyXfFile()` and `mainPanel.showProcess(...)`,
    // and then prints `Utilities.prepareDialogActionMessage(DialogType.JOIN, AxisID.ONLY,
    // null)` - the one part of it with a module, and the reason this class is what
    // `prepareDialogActionMessage` was blocked on.
    pub fn open_join_dialog(&self) -> bool {
        true
    }

    /// Java `getParameters(MidasParam, AxisID)`.
    // TODO(unit): needs etomo/comscript/MidasParam.java and etomo/type/JoinMetaData.java
    // - the body is `param.setInputFileName(FileType.JOIN_SAMPLE.getFileName(this,
    // axisID))` and `param.setSectionTableRowData(metaData.getSectionTableData())`.
    pub fn get_parameters_midas(&self, param: Option<Infallible>, axis_id: Option<AxisID>) {
        let _ = (param, axis_id);
    }

    /// Java `getParameters(XfalignParam, AxisID)`.
    // TODO(unit): needs etomo/comscript/XfalignParam.java - the body is
    // `param.setInputFileName(FileType.JOIN_SAMPLE_AVERAGES.getFileName(this, axisID))`.
    pub fn get_parameters_xfalign(&self, param: Option<Infallible>, axis_id: Option<AxisID>) {
        let _ = (param, axis_id);
    }

    /// Java private `doneJoinDialog`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java and etomo/ui/swing/UIHarness.java
    // - the body reads the dialog's working directory and root name and opens message
    // dialogs about them.
    fn done_join_dialog(&self) -> bool {
        false
    }

    /// Java `getScreenState`.
    // TODO(unit): needs etomo/type/JoinScreenState.java - the return type.
    pub fn get_screen_state(&self) -> Option<Infallible> {
        self.screen_state
    }

    /// Java `imodOpen(String, int, Run3dmodMenuOptions)`.
    // TODO(unit): needs etomo/process/BaseImodManager.java through `imodManager.open` and
    // etomo/type/Run3dmodMenuOptions.java.
    pub fn imod_open(
        &self,
        imod_key: Option<&str>,
        binning: i32,
        menu_options: Option<Infallible>,
    ) {
        let _ = (imod_key, binning, menu_options);
    }

    /// Java `imodOpen(String, int, String, Run3dmodMenuOptions)`.
    // TODO(unit): needs etomo/process/BaseImodManager.java through `imodManager.open` and
    // etomo/type/Run3dmodMenuOptions.java.
    pub fn imod_open_with_model(
        &self,
        imod_key: Option<&str>,
        binning: i32,
        model_name: Option<&str>,
        menu_options: Option<Infallible>,
    ) {
        let _ = (imod_key, binning, model_name, menu_options);
    }

    /// Java `imodOpen(ProcessSeries, String)`.
    // TODO(unit): needs etomo/process/BaseImodManager.java and
    // etomo/ProcessSeries.java.
    pub fn imod_open_process_series(
        &self,
        process_series: Option<Infallible>,
        imod_key: Option<&str>,
    ) {
        let _ = (process_series, imod_key);
    }

    /// Java `isImodOpen`.
    // TODO(unit): needs etomo/process/BaseImodManager.java - the body is
    // `imodManager.isOpen(imodKey)`.
    pub fn is_imod_open(&self, imod_key: Option<&str>) -> bool {
        let _ = imod_key;
        false
    }

    /// Java `imodRemove`.
    // TODO(unit): needs etomo/process/BaseImodManager.java - the body is
    // `imodManager.delete(imodKey, imodIndex)`.
    pub fn imod_remove(&self, imod_key: Option<&str>, imod_index: i32) {
        let _ = (imod_key, imod_index);
    }

    /// Java `imodGetSlicerAngles`.
    // TODO(unit): needs etomo/process/BaseImodManager.java and
    // etomo/type/SlicerAngles.java.
    pub fn imod_get_slicer_angles(
        &self,
        imod_key: Option<&str>,
        imod_index: i32,
    ) -> Option<Infallible> {
        let _ = (imod_key, imod_index);
        None
    }

    /// Java `makejoincom`.
    // TODO(unit): needs etomo/comscript/MakejoincomParam.java,
    // etomo/process/JoinProcessManager.java, etomo/ProcessSeries.java and
    // etomo/ui/swing/Deferred3dmodButton.java.
    pub fn makejoincom(
        &self,
        process_series: Option<Infallible>,
        deferred_3dmod_button: Option<Infallible>,
        run_3dmod_menu_options: Option<Infallible>,
        dialog_type: Option<DialogType>,
    ) {
        let _ = (
            process_series,
            deferred_3dmod_button,
            run_3dmod_menu_options,
            dialog_type,
        );
    }

    /// Java `postProcess`.
    // TODO(unit): needs etomo/comscript/ProcessDetails.java and
    // etomo/type/JoinState.java.
    pub fn post_process(&self, command_name: Option<&str>, process_details: Option<Infallible>) {
        let _ = (command_name, process_details);
    }

    /// Java `endSetupMode`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java - the body is
    // `joinDialog.setMode(JoinDialog.SAMPLE_NOT_PRODUCED_MODE)`.
    pub fn end_setup_mode(&self) -> bool {
        false
    }

    /// Java private `copyMostRecentXfFile`.
    // TODO(unit): needs etomo/AutoAlignmentController.java.
    fn copy_most_recent_xf_file(&self, command_description: Option<&str>) -> bool {
        let _ = command_description;
        false
    }

    /// Java `setMode(String)`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java.
    pub fn set_mode_with_dir(&self, working_dir_name: Option<&str>) -> bool {
        let _ = working_dir_name;
        false
    }

    /// Java `setMode()`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java.
    pub fn set_mode(&self) -> bool {
        false
    }

    /// Java `startjoin`.
    // TODO(unit): needs etomo/comscript/StartJoinParam.java and
    // etomo/process/JoinProcessManager.java.
    pub fn startjoin(&self, process_series: Option<Infallible>) {
        let _ = process_series;
    }

    /// Java `xfjointomo`.
    // TODO(unit): needs etomo/comscript/XfjointomoParam.java and
    // etomo/process/JoinProcessManager.java.
    pub fn xfjointomo(&self, process_series: Option<Infallible>) {
        let _ = process_series;
    }

    /// Java private `remapmodel`.
    // TODO(unit): needs etomo/comscript/RemapmodelParam.java and
    // etomo/process/JoinProcessManager.java.
    fn remapmodel(&self, process_series: Option<Infallible>) {
        let _ = process_series;
    }

    /// Java `xfmodel(String, String, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType)`.
    // TODO(unit): needs etomo/comscript/XfmodelParam.java and
    // etomo/ui/swing/Deferred3dmodButton.java.
    pub fn xfmodel_with_files(
        &self,
        input_file: Option<&str>,
        output_file: Option<&str>,
        process_series: Option<Infallible>,
        deferred_3dmod_button: Option<Infallible>,
        run_3dmod_menu_options: Option<Infallible>,
        dialog_type: Option<DialogType>,
    ) {
        let _ = (
            input_file,
            output_file,
            process_series,
            deferred_3dmod_button,
            run_3dmod_menu_options,
            dialog_type,
        );
    }

    /// Java private `xfmodel(ProcessSeries, DialogType)`.
    // TODO(unit): needs etomo/comscript/XfmodelParam.java.
    fn xfmodel(&self, process_series: Option<Infallible>, dialog_type: Option<DialogType>) {
        let _ = (process_series, dialog_type);
    }

    /// Java private `xfmodel(XfmodelParam, ProcessSeries, DialogType)`.
    // TODO(unit): needs etomo/comscript/XfmodelParam.java and
    // etomo/process/JoinProcessManager.java.
    fn xfmodel_with_param(
        &self,
        param: Option<Infallible>,
        process_series: Option<Infallible>,
        dialog_type: Option<DialogType>,
    ) {
        let _ = (param, process_series, dialog_type);
    }

    /// Java private `xftoxg`.
    // TODO(unit): needs etomo/comscript/XftoxgParam.java and
    // etomo/process/JoinProcessManager.java.
    fn xftoxg(&self, process_series: Option<Infallible>, dialog_type: Option<DialogType>) {
        let _ = (process_series, dialog_type);
    }

    /// Java `updateJoinDialogDisplay`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java.
    pub fn update_join_dialog_display(&self) {}

    /// Java `finishjoin`.
    // TODO(unit): needs etomo/comscript/FinishjoinParam.java,
    // etomo/process/JoinProcessManager.java and etomo/ui/swing/Deferred3dmodButton.java.
    pub fn finishjoin(
        &self,
        mode: Option<Infallible>,
        button_text: Option<&str>,
        process_series: Option<Infallible>,
        deferred_3dmod_button: Option<Infallible>,
        run_3dmod_menu_options: Option<Infallible>,
        dialog_type: Option<DialogType>,
    ) {
        let _ = (
            mode,
            button_text,
            process_series,
            deferred_3dmod_button,
            run_3dmod_menu_options,
            dialog_type,
        );
    }

    /// Java private `updateMetaDataFromJoinDialog`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java and etomo/type/JoinMetaData.java.
    fn update_meta_data_from_join_dialog(
        &self,
        axis_id: Option<AxisID>,
        do_validation: bool,
    ) -> bool {
        let _ = (axis_id, do_validation);
        false
    }

    /// Java `setSize`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java.
    pub fn set_size(&self, size_in_x_string: Option<&str>, size_in_y_string: Option<&str>) {
        let _ = (size_in_x_string, size_in_y_string);
    }

    /// Java `setShift`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java.
    pub fn set_shift(&self, shift_in_x: i32, shift_in_y: i32) {
        let _ = (shift_in_x, shift_in_y);
    }

    /// Java `rotx`.
    // TODO(unit): needs etomo/comscript/ClipParam.java and
    // etomo/process/JoinProcessManager.java.
    pub fn rotx(
        &self,
        tomogram: Option<&Path>,
        working_dir: Option<&Path>,
        process_series: Option<Infallible>,
    ) {
        let _ = (tomogram, working_dir, process_series);
    }

    /// Java `abortAddSection`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java.
    pub fn abort_add_section(&self) {}

    /// Java `addSection`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java and etomo/type/JoinMetaData.java.
    pub fn add_section(&self, tomogram: Option<&Path>) {
        let _ = tomogram;
    }

    /// Java private `openProcessingPanel`.
    // TODO(unit): needs etomo/ui/swing/MainJoinPanel.java - the body is
    // `mainPanel.showProcessingPanel(AxisType.SINGLE_AXIS)`, `setDividerLocation()` and
    // `mainPanel.setStatusBarText(...)`.
    fn open_processing_panel(&self) {}

    /// Java `getConstMetaData`.
    // TODO(unit): needs etomo/type/ConstJoinMetaData.java - the return type.
    pub fn get_const_meta_data(&self) -> Option<Infallible> {
        self.meta_data
    }

    /// Java `getJoinMetaData`.
    // TODO(unit): needs etomo/type/JoinMetaData.java - the return type.
    pub fn get_join_meta_data(&self) -> Option<Infallible> {
        self.meta_data
    }

    /// Java package-private `createState`.
    // TODO(unit): needs etomo/type/JoinState.java - the body is
    // `state = new JoinState(this)`.
    pub(crate) fn create_state(&self) {}

    /// Java `getState`.
    // TODO(unit): needs etomo/type/ConstJoinState.java - the return type.
    pub fn get_state(&self) -> Option<Infallible> {
        self.state
    }

    /// Java `newStartJoinParam`.
    // TODO(unit): needs etomo/comscript/StartJoinParam.java - the body is
    // `startJoinParam = new StartJoinParam(AxisID.ONLY)`.
    pub fn new_start_join_param(&self) -> Option<Infallible> {
        self.start_join_param
    }

    /// Java `packDialogs(AxisID)`, whose body is empty.
    pub fn pack_dialogs_with_axis(&self, axis_id: Option<AxisID>) {
        let _ = axis_id;
    }

    /// Java `packDialogs()`, whose body is empty.
    pub fn pack_dialogs(&self) {}

    /// Java private `updateJoinwarp2modelParam`.
    // TODO(unit): needs etomo/comscript/Joinwarp2modelParam.java and
    // etomo/ui/swing/JoinDialog.java.
    fn update_joinwarp2model_param(&self, do_validation: bool) -> Option<Infallible> {
        let _ = do_validation;
        None
    }

    /// Java `startRefine`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java and
    // etomo/ProcessSeries.java.
    pub fn start_refine(&self) {}

    /// Java private `refineJoin`.
    // TODO(unit): needs etomo/comscript/Joinwarp2modelParam.java and
    // etomo/process/JoinProcessManager.java.
    fn refine_join(&self, process_series: Option<Infallible>) {
        let _ = process_series;
    }
}

impl BaseManager for JoinManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    /// Java `getInterfaceType`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::Join)
    }

    /// Java `saveParamFile`.
    fn save_param_file(&self) -> bool {
        let retval = self.save_param_file_super();
        if retval {
            self.end_setup_mode();
        }
        retval
    }

    /// Java `setParamFile()`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java,
    // etomo/process/JoinProcessManager.java, etomo/process/BaseImodManager.java and
    // etomo/ui/swing/MainJoinPanel.java - the body reads the dialog's working directory
    // and root name, creates the .ejf file through `processMgr.createNewFile`, and on
    // success calls `imodManager.setMetaData(metaData)` and
    // `mainPanel.setStatusBarText(...)`.
    fn set_param_file(&self) -> bool {
        *self.base.loaded_param_file.lock().unwrap()
    }

    /// Java `getFocusComponent`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java - the body is
    // `joinDialog.getFocusComponent()`, and `java.awt.Component` is the return type.
    fn get_focus_component(&self) -> Option<Infallible> {
        if self.join_dialog.is_none() {
            return None;
        }
        self.join_dialog
    }

    /// Java package-private `paramString`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java, etomo/type/JoinMetaData.java,
    // etomo/process/JoinProcessManager.java and etomo/type/JoinState.java - the body
    // prints all four fields before `super.paramString()`.
    fn param_string(&self) -> Option<String> {
        Some(format!(
            "joinDialog=null,metaData=null,\nprocessMgr=null,state=null,\nsuper[{}]",
            self.param_string_super().unwrap_or_default()
        ))
    }

    /// Java `getParamFile`.  Return the test parameter file as a File object.
    fn get_param_file(&self) -> Option<PathBuf> {
        if self.base.param_file.lock().unwrap().is_none() && !self.done_join_dialog() {
            return None;
        }
        self.base.param_file.lock().unwrap().clone()
    }

    /// Java package-private `createMainPanel`.
    // TODO(unit): needs etomo/ui/swing/MainJoinPanel.java - the body is
    // `mainPanel = new MainJoinPanel(this)` unless the program is headless.
    fn create_main_panel(&self) {}

    /// Java `setDebug`.
    fn set_debug(&self, debug: bool) {
        // Java calls `super.setDebug(debug)` and then stores its own copy.
        self.set_debug_super(debug);
        *self.debug.lock().unwrap() = debug;
    }

    /// Java package-private `isTomosnapshotThumbnail`.
    fn is_tomosnapshot_thumbnail(&self) -> bool {
        true
    }

    /// Java `setParamFile(File)`.
    // TODO(unit): needs etomo/ui/swing/MainJoinPanel.java and
    // etomo/process/BaseImodManager.java - the body calls
    // `initializeUIParameters(paramFile, AxisID.ONLY, false)` and, when that loads,
    // `imodManager.setMetaData(metaData)` and `mainPanel.setStatusBarText(...)`.
    fn set_param_file_from(&self, param_file: Option<&Path>) -> bool {
        let _ = param_file;
        *self.base.loaded_param_file.lock().unwrap()
    }

    /// Java package-private `startNextProcess`.
    // TODO(unit): needs etomo/ProcessSeries.java,
    // etomo/type/ProcessResultDisplay.java, etomo/ui/swing/ProcessDisplay.java and
    // etomo/ui/UIComponent.java - the body dispatches on the process name to
    // `remapmodel`, `xfmodel`, `xftoxg`, `finishjoin` and `refineJoin`.
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

    /// Java `getBaseMetaData`.
    // TODO(unit): needs etomo/type/JoinMetaData.java - the body returns `metaData`,
    // whose declared type has no module.
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        None
    }

    /// Java `getMainPanel`.
    // TODO(unit): needs etomo/ui/swing/MainJoinPanel.java - the body returns `mainPanel`.
    fn get_main_panel(&self) -> Option<Infallible> {
        self.main_panel
    }

    /// Java `getBaseState`.
    // TODO(unit): needs etomo/type/JoinState.java - the body returns `state`.
    fn get_base_state(&self) -> Option<Infallible> {
        self.state
    }

    /// Java package-private `getAutoAlignmentMetaData`.
    // TODO(unit): needs etomo/type/JoinMetaData.java - the body is
    // `metaData.getAutoAlignmentMetaData()`.
    fn get_auto_alignment_meta_data(&self) -> Option<Infallible> {
        None
    }

    /// Java `updateMetaData`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java and etomo/type/JoinMetaData.java -
    // the body is `joinDialog.getMetaData(metaData, doValidation)`; neither the
    // `dialogType` nor the `axisID` parameter is read.
    fn update_meta_data(
        &self,
        dialog_type: Option<DialogType>,
        axis_id: Option<AxisID>,
        do_validation: bool,
    ) -> bool {
        let _ = (dialog_type, axis_id, do_validation);
        false
    }

    /// Java `pause`.
    // TODO(unit): needs etomo/process/JoinProcessManager.java - the body is
    // `processMgr.pause(axisID)`.
    fn pause(&self, axis_id: Option<AxisID>) -> bool {
        let _ = axis_id;
        false
    }

    /// Java `getProcessManager`.
    // TODO(unit): needs etomo/process/JoinProcessManager.java - the body returns
    // `processMgr`.
    fn get_process_manager(&self) -> Option<Infallible> {
        self.process_mgr
    }

    /// Java package-private `getStorables(int)`.
    // TODO(unit): needs etomo/type/JoinMetaData.java, etomo/type/JoinState.java and
    // etomo/type/JoinScreenState.java - the array the source fills is
    // `{ metaData, state, screenState }` at the given offset.
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Box<dyn Storable>>> {
        let _ = offset;
        None
    }

    /// Java `exitProgram`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java - the body calls
    // `super.exitProgram(axisID)` and then saves the dialog's state.
    fn exit_program(&self, axis_id: Option<AxisID>) -> bool {
        self.exit_program_super(axis_id)
    }

    /// Java `save`.
    // TODO(unit): needs etomo/ui/swing/JoinDialog.java - the body calls `super.save()`
    // and then `joinDialog.getParameters(...)`/`saveParamFile()`.
    fn save(&self) -> bool {
        self.save_super()
    }

    /// Java `getName`.
    // TODO(unit): needs etomo/type/JoinMetaData.java - the body is `metaData.getName()`.
    fn get_name(&self) -> Option<String> {
        Some(self.name.clone())
    }
}

/// Java `toString`.
impl std::fmt::Display for JoinManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "etomo.JoinManager[{}]",
            self.param_string().unwrap_or_default()
        )
    }
}
