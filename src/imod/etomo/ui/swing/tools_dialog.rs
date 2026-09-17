//! `IMOD/Etomo/src/etomo/ui/swing/ToolsDialog.java`.
//!
//! `JPanel`, `JTextArea`, `JScrollPane`, the three tool-panel static factories,
//! and `EtomoLogger`'s Swing `invokeLater` queue are native/UI-source-unit
//! boundaries.  This module retains the state and dispatch that
//! `ToolsDialog.java` owns; it does not invent a second dialog toolkit or a
//! process implementation.
#![allow(dead_code)]

use std::path::Path;

use super::abstract_frame::ComponentState;
use super::etomo_menu::ToolType;
use super::log_interface::FileReaderRef;
use super::panel::Panel;
use super::tool_panel::ToolPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Direct declared-type boundary for `etomo.comscript.ConstWarpVolParam`.
/// Its complete source unit owns the parameter fields.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConstWarpVolParam;

/// Direct declared-type boundary for `etomo.comscript.GpuTiltTestParam`.
/// Its complete source unit owns validation and parameter fields.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GpuTiltTestParam {
    /// Java `GpuTiltTestParam.nMinutes` at the declared-parameter boundary.
    pub n_minutes: Option<String>,
    /// Java `GpuTiltTestParam.gpuNumber` at the declared-parameter boundary.
    pub gpu_number: Option<i32>,
}

impl super::gpu_tilt_test_panel::GpuTiltTestParameters for GpuTiltTestParam {
    fn set_n_minutes(&mut self, input: String) {
        self.n_minutes = Some(input);
    }
    fn set_gpu_number(&mut self, input: i32) {
        self.gpu_number = Some(input);
    }
}

/// Rust surface of the source's direct `FlattenVolumePanel` cast.
pub trait FlattenVolumePanel: ToolPanel {
    /// Java `FlattenVolumePanel.setParameters(ConstWarpVolParam)`.
    fn set_parameters(&mut self, param: &ConstWarpVolParam);
}

/// Rust surface of the source's direct `GpuTiltTestPanel` cast.
pub trait GpuTiltTestPanel: ToolPanel {
    /// Java `GpuTiltTestPanel.getParameters(GpuTiltTestParam, boolean)`.
    fn get_parameters(&mut self, param: &mut GpuTiltTestParam, do_validation: bool) -> bool;
}

/// `AlignFramesPanel` has no additional method called by this source unit.
pub trait AlignFramesPanel: ToolPanel {}

/// One result of the three static panel factories called by the Java
/// constructor.  The enum is the Rust replacement for Java's downcasts.
pub enum ToolsToolPanel {
    FlattenVolume(Box<dyn FlattenVolumePanel>),
    GpuTiltTest(Box<dyn GpuTiltTestPanel>),
    AlignFrames(Box<dyn AlignFramesPanel>),
}

impl ToolsToolPanel {
    /// Java `toolPanel.getComponent()` at each source call site.
    fn get_component(&self) -> &ComponentState {
        match self {
            Self::FlattenVolume(panel) => panel.get_component(),
            Self::GpuTiltTest(panel) => panel.get_component(),
            Self::AlignFrames(panel) => panel.get_component(),
        }
    }
}

/// Boundary for the three source static factory calls.  The corresponding
/// panel source units own their component construction and process wiring.
pub trait ToolsDialogPanelFactory {
    /// Java `FlattenVolumePanel.getToolsInstance(manager, axisID, dialogType)`.
    fn flatten_volume_panel(
        &mut self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Box<dyn FlattenVolumePanel>;

    /// Java `GpuTiltTestPanel.getInstance(manager, axisID)`.
    fn gpu_tilt_test_panel(
        &mut self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Box<dyn GpuTiltTestPanel>;

    /// Java `AlignFramesPanel.getToolsInstance(manager, axisID, dialogType)`.
    fn align_frames_panel(
        &mut self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Box<dyn AlignFramesPanel>;
}

/// Swing `JTextArea` source-observable state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TaskLog {
    pub text: String,
    pub editable: bool,
    pub rows: i32,
    pub line_wrap: bool,
    pub wrap_style_word: bool,
}

/// Swing `JScrollPane` state used by this class.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TaskLogScrollPane {
    pub view_is_task_log: bool,
}

/// Fields and all source-owned behavior of Java's final `ToolsDialog`.
pub struct ToolsDialog {
    pub pnl_root: Panel,
    pub root_children: Vec<ComponentState>,
    pub ta_task_log: TaskLog,
    pub scr_task_log: TaskLogScrollPane,
    /// Java `EtomoLogger`; its timestamp/file/secondary-log execution remains
    /// at the `EtomoLogger.java` and native event-queue boundary.
    pub allow_primary_logging: bool,
    pub tool_panel: Option<ToolsToolPanel>,
    pub tool_type: ToolType,
    /// `ToolsManager` is a `BaseManager` subclass.  Until its complete source
    /// unit is translated, the inherited declared surface is exact here.
    pub manager: &'static dyn BaseManager,
    pub axis_id: AxisID,
}

impl ToolsDialog {
    /// Java private `ToolsDialog(ToolsManager, AxisID, DialogType, ToolType)`.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        tool_type: ToolType,
        panel_factory: &mut dyn ToolsDialogPanelFactory,
    ) -> Self {
        let tool_panel = match tool_type {
            ToolType::FlattenVolume => Some(ToolsToolPanel::FlattenVolume(
                panel_factory.flatten_volume_panel(manager, axis_id, dialog_type),
            )),
            ToolType::GpuTiltTest => Some(ToolsToolPanel::GpuTiltTest(
                panel_factory.gpu_tilt_test_panel(manager, axis_id),
            )),
            ToolType::AlignFrames => Some(ToolsToolPanel::AlignFrames(
                panel_factory.align_frames_panel(manager, axis_id, dialog_type),
            )),
        };
        Self {
            pnl_root: Panel::default(),
            root_children: vec![],
            ta_task_log: TaskLog::default(),
            scr_task_log: TaskLogScrollPane {
                view_is_task_log: true,
            },
            allow_primary_logging: true,
            tool_panel,
            tool_type,
            manager,
            axis_id,
        }
    }

    /// Java `getInstance(ToolsManager, AxisID, DialogType, ToolType)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        tool_type: ToolType,
        panel_factory: &mut dyn ToolsDialogPanelFactory,
    ) -> Self {
        let mut instance = Self::new(manager, axis_id, dialog_type, tool_type, panel_factory);
        instance.create_panel();
        instance
    }

    /// Java `getManager()`.
    pub fn get_manager(&self) -> &'static dyn BaseManager {
        self.manager
    }

    /// Java `getAxisID()`.
    pub fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `setParameters(ConstWarpVolParam)`.  A wrong panel type throws
    /// Java's `ClassCastException`; Rust makes that source precondition panic.
    pub fn set_parameters(&mut self, param: &ConstWarpVolParam) {
        match self.tool_panel.as_mut() {
            Some(ToolsToolPanel::FlattenVolume(panel)) => panel.set_parameters(param),
            _ => panic!("ToolsDialog.setParameters requires FlattenVolumePanel"),
        }
    }

    /// Java `getParameters(GpuTiltTestParam, boolean)`.
    pub fn get_parameters(&mut self, param: &mut GpuTiltTestParam, do_validation: bool) -> bool {
        match self.tool_panel.as_mut() {
            Some(ToolsToolPanel::GpuTiltTest(panel)) => panel.get_parameters(param, do_validation),
            _ => panic!("ToolsDialog.getParameters requires GpuTiltTestPanel"),
        }
    }

    /// Java `isAllowPrimaryLogging()`.
    pub fn is_allow_primary_logging(&self) -> bool {
        self.allow_primary_logging
    }

    /// Java `setAllowPrimaryLogging(boolean)`.
    pub fn set_allow_primary_logging(&mut self, input: bool) {
        self.allow_primary_logging = input;
    }

    /// Java `logMessage(String)` after the `EtomoLogger` event-queue boundary.
    pub fn log_message(&mut self, message: &str) {
        if self.allow_primary_logging {
            self.append(message);
        }
    }

    /// Java `logMessage(String, boolean, boolean, FileWriter)`.  FileWriter is
    /// an untranslated storage boundary; the source-visible primary-log part
    /// is retained and `timestamp`/secondary output are deliberately not guessed.
    pub fn log_message_with_secondary_log(
        &mut self,
        message: &str,
        _timestamp: bool,
        _newline: bool,
    ) {
        self.log_message(message);
    }

    /// Java `logMessage(File, FileWriter)` at the file-reader boundary.
    pub fn log_message_file(&mut self, file: &Path) {
        if self.allow_primary_logging {
            self.append(&file.to_string_lossy());
        }
    }

    /// Java `logMessage(File, boolean, FileWriter)` at the file-reader boundary.
    pub fn log_message_file_with_newline(&mut self, file: &Path, _newline: bool) {
        self.log_message_file(file);
    }

    /// Java `logMessagePrimaryLog(FileReader)`.  The Java implementation
    /// delegates to `EtomoLogger`; this Rust log surface consumes the same
    /// reader boundary directly.
    pub fn log_message_primary_log(&mut self, reader: Option<FileReaderRef>) {
        let Some(reader) = reader else { return };
        while reader.borrow().is_readable() {
            let Some(line) = reader.borrow_mut().read_line() else {
                break;
            };
            self.append(&line);
            if !line.ends_with('\n') {
                self.append("\n");
            }
        }
    }

    /// Java `save()`.
    pub fn save(&mut self) {}

    /// Java `msgChanged()`.
    pub fn msg_changed(&mut self) {}

    /// Java `append(String)`.
    pub fn append(&mut self, line: &str) {
        self.ta_task_log.text.push_str(line);
    }

    /// Java `getPrevLineEndOffset()`.  `BadLocationException` cannot arise
    /// for the owned Rust text state, so this returns the calculated offset.
    pub fn get_prev_line_end_offset(&self) -> usize {
        self.ta_task_log
            .text
            .rfind('\n')
            .map_or(self.ta_task_log.text.len(), |offset| offset + 1)
    }

    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        self.ta_task_log.editable = false;
        self.ta_task_log.rows = 5;
        self.ta_task_log.line_wrap = true;
        self.ta_task_log.wrap_style_word = true;
        if let Some(tool_panel) = &self.tool_panel {
            self.root_children.push(tool_panel.get_component().clone());
        }
        // Java adds the JScrollPane after the tool component.  The real native
        // JScrollPane remains a GUI boundary, represented by its task-log view.
        self.root_children.push(ComponentState::default());
        self.manager.pack();
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> &Panel {
        &self.pnl_root
    }

    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::base_manager::BaseManagerBase;
    use crate::imod::etomo::storage::storable::Storable;
    use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
    use crate::imod::etomo::r#type::interface_type::InterfaceType;
    use std::convert::Infallible;

    struct Manager(BaseManagerBase);
    impl BaseManager for Manager {
        fn base(&self) -> &BaseManagerBase {
            &self.0
        }
        fn this(&'static self) -> &'static dyn BaseManager {
            self
        }
        fn get_interface_type(&self) -> Option<InterfaceType> {
            None
        }
        fn create_main_panel(&self) {}
        fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
            None
        }
        fn get_main_panel(&self) -> Option<Infallible> {
            None
        }
        fn get_process_manager(&self) -> Option<Infallible> {
            None
        }
        fn get_storables_with_offset(&self, _: i32) -> Option<Vec<Box<dyn Storable>>> {
            None
        }
        fn get_name(&self) -> Option<String> {
            None
        }
    }

    #[derive(Default)]
    struct Flatten {
        component: ComponentState,
        got_parameters: bool,
    }
    impl ToolPanel for Flatten {
        fn get_component(&self) -> &ComponentState {
            &self.component
        }
    }
    impl FlattenVolumePanel for Flatten {
        fn set_parameters(&mut self, _: &ConstWarpVolParam) {
            self.got_parameters = true;
        }
    }
    #[derive(Default)]
    struct Gpu {
        component: ComponentState,
    }
    impl ToolPanel for Gpu {
        fn get_component(&self) -> &ComponentState {
            &self.component
        }
    }
    impl GpuTiltTestPanel for Gpu {
        fn get_parameters(&mut self, _: &mut GpuTiltTestParam, do_validation: bool) -> bool {
            do_validation
        }
    }
    #[derive(Default)]
    struct Align {
        component: ComponentState,
    }
    impl ToolPanel for Align {
        fn get_component(&self) -> &ComponentState {
            &self.component
        }
    }
    impl AlignFramesPanel for Align {}
    struct Factory;
    impl ToolsDialogPanelFactory for Factory {
        fn flatten_volume_panel(
            &mut self,
            _: &'static dyn BaseManager,
            _: AxisID,
            _: DialogType,
        ) -> Box<dyn FlattenVolumePanel> {
            Box::new(Flatten::default())
        }
        fn gpu_tilt_test_panel(
            &mut self,
            _: &'static dyn BaseManager,
            _: AxisID,
        ) -> Box<dyn GpuTiltTestPanel> {
            Box::new(Gpu::default())
        }
        fn align_frames_panel(
            &mut self,
            _: &'static dyn BaseManager,
            _: AxisID,
            _: DialogType,
        ) -> Box<dyn AlignFramesPanel> {
            Box::new(Align::default())
        }
    }

    #[test]
    fn get_instance_selects_source_factory_and_adds_tool_before_log() {
        let manager: &'static Manager = Box::leak(Box::new(Manager(BaseManagerBase::default())));
        let mut factory = Factory;
        let dialog = ToolsDialog::get_instance(
            manager,
            AxisID::First,
            DialogType::Tools,
            ToolType::AlignFrames,
            &mut factory,
        );
        assert_eq!(dialog.ta_task_log.rows, 5);
        assert!(dialog.ta_task_log.line_wrap && dialog.ta_task_log.wrap_style_word);
        assert_eq!(dialog.root_children.len(), 2);
        assert!(dialog.scr_task_log.view_is_task_log);
    }

    #[test]
    fn gpu_parameter_delegation_preserves_validation_argument() {
        let manager: &'static Manager = Box::leak(Box::new(Manager(BaseManagerBase::default())));
        let mut factory = Factory;
        let mut dialog = ToolsDialog::get_instance(
            manager,
            AxisID::Only,
            DialogType::Tools,
            ToolType::GpuTiltTest,
            &mut factory,
        );
        assert!(dialog.get_parameters(&mut GpuTiltTestParam::default(), true));
        assert!(!dialog.get_parameters(&mut GpuTiltTestParam::default(), false));
    }

    #[test]
    fn append_and_previous_line_end_follow_jtext_area_offsets() {
        let manager: &'static Manager = Box::leak(Box::new(Manager(BaseManagerBase::default())));
        let mut factory = Factory;
        let mut dialog = ToolsDialog::get_instance(
            manager,
            AxisID::Only,
            DialogType::Tools,
            ToolType::FlattenVolume,
            &mut factory,
        );
        dialog.append("one\ntwo");
        assert_eq!(dialog.get_prev_line_end_offset(), 4);
    }
}
