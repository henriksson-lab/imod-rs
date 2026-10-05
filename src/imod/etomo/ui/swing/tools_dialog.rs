//! `IMOD/Etomo/src/etomo/ui/swing/ToolsDialog.java`.
//!
//! The Tools interface's one dialog: the tool's panel (`FlattenVolumePanel`,
//! `GpuTiltTestPanel` or `AlignFramesPanel`, chosen by the `ToolType`) above a
//! task log.  It is also the manager's `LogInterface`.  An
//! event-dispatch-thread object (`Rc`, `&self` methods).

use std::path::Path;
use std::rc::{Rc, Weak};

use super::align_frames_panel::AlignFramesPanel;
use super::context_menu::ContextMenu;
use super::etomo_logger::EtomoLogger;
use super::flatten_volume_panel::FlattenVolumePanel;
use super::gpu_tilt_test_panel::GpuTiltTestPanel;
use super::log_interface::{BadLocationException, LogInterface};
use super::tool_panel::ToolPanel;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_warp_vol_param::ConstWarpVolParam;
use crate::imod::etomo::comscript::gpu_tilt_test_param::GpuTiltTestParam;
use crate::imod::etomo::jdk::{JComponent, MouseEvent};
use crate::imod::etomo::storage::file_reader::FileReaderRef;
use crate::imod::etomo::storage::file_writer::FileWriterRef;
use crate::imod::etomo::storage::loggable::Loggable;
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::swing::etomo_menu::ToolType;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `private final ToolPanel toolPanel`, keeping the concrete class the
/// source casts it back to (`(FlattenVolumePanel) toolPanel`,
/// `(GpuTiltTestPanel) toolPanel`).
#[derive(Clone)]
pub enum ToolPanelInstance {
    FlattenVolume(Rc<FlattenVolumePanel>),
    GpuTiltTest(Rc<GpuTiltTestPanel>),
    AlignFrames(Rc<AlignFramesPanel>),
}

impl ToolPanelInstance {
    /// The value as the Java declared type, `ToolPanel`.
    fn as_tool_panel(&self) -> &dyn ToolPanel {
        match self {
            ToolPanelInstance::FlattenVolume(panel) => &**panel,
            ToolPanelInstance::GpuTiltTest(panel) => &**panel,
            ToolPanelInstance::AlignFrames(panel) => &**panel,
        }
    }
}

/// Java `public final class ToolsDialog implements ContextMenu, LogInterface`.
pub struct ToolsDialog {
    /// Java `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java `taTaskLog = new JTextArea()`.
    ta_task_log: Rc<JComponent>,
    /// Java `scrTaskLog = new JScrollPane(taTaskLog)`.
    scr_task_log: Rc<JComponent>,

    /// Java private final `logger`.
    logger: EtomoLogger,

    /// Java private final `toolPanel`.
    tool_panel: Option<ToolPanelInstance>,
    /// Java private final `toolType`.
    tool_type: ToolType,
    /// Java private final `manager`.
    manager: &'static ToolsManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
}

impl ToolsDialog {
    /// Java private `ToolsDialog(ToolsManager, AxisID, DialogType, ToolType)`.
    fn new(
        manager: &'static ToolsManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        tool_type: ToolType,
    ) -> Rc<ToolsDialog> {
        // The tool panel is built before the dialog object exists here; the
        // Java builds it in the constructor body after `logger`, and nothing
        // in either constructor reads the other.
        let tool_panel = if tool_type == ToolType::FlattenVolume {
            Some(ToolPanelInstance::FlattenVolume(
                FlattenVolumePanel::get_tools_instance(manager, axis_id, dialog_type),
            ))
        } else if tool_type == ToolType::GpuTiltTest {
            Some(ToolPanelInstance::GpuTiltTest(
                GpuTiltTestPanel::get_instance(manager, axis_id),
            ))
        } else if tool_type == ToolType::AlignFrames {
            Some(ToolPanelInstance::AlignFrames(
                AlignFramesPanel::get_tools_instance(manager, axis_id, dialog_type),
            ))
        } else {
            None
        };
        let ta_task_log = JComponent::new_text_area();
        let scr_task_log = JComponent::new_scroll_pane(Some(&ta_task_log));
        Rc::new_cyclic(|self_ref: &Weak<ToolsDialog>| {
            let primary_log: Weak<dyn LogInterface> = self_ref.clone();
            ToolsDialog {
                pnl_root: JComponent::new_panel(),
                ta_task_log,
                scr_task_log,
                // logger = new EtomoLogger(this)
                logger: EtomoLogger::new(primary_log),
                tool_panel,
                tool_type,
                manager,
                axis_id,
            }
        })
    }

    /// Java static `getInstance(ToolsManager, AxisID, DialogType, ToolType)`.
    pub fn get_instance(
        manager: &'static ToolsManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        tool_type: ToolType,
    ) -> Rc<ToolsDialog> {
        let instance = ToolsDialog::new(manager, axis_id, dialog_type, tool_type);
        instance.create_panel();
        instance
    }

    /// Java `setParameters(ConstWarpVolParam)`.
    pub fn set_parameters(&self, param: &dyn ConstWarpVolParam) {
        // Upstream bug fixed in translation (ToolsDialog.java:79): Java casts
        // toolPanel to FlattenVolumePanel unchecked (ClassCastException for
        // another tool, NullPointerException for none); only a flatten panel
        // is set here.
        if let Some(ToolPanelInstance::FlattenVolume(panel)) = &self.tool_panel {
            panel.set_parameters_const_warp_vol_param(param);
        }
    }

    /// Java `getParameters(GpuTiltTestParam, boolean)`.
    pub fn get_parameters(&self, param: &mut GpuTiltTestParam, do_validation: bool) -> bool {
        // Upstream bug fixed in translation (ToolsDialog.java:83): Java casts
        // toolPanel to GpuTiltTestPanel unchecked; any other panel gets no
        // parameters (false) here.
        match &self.tool_panel {
            Some(ToolPanelInstance::GpuTiltTest(panel)) => {
                panel.get_parameters(param, do_validation)
            }
            _ => false,
        }
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // initialize
        self.ta_task_log.set_editable(false);
        // Swing layout: taTaskLog.setRows(5); taTaskLog.setLineWrap(true);
        // taTaskLog.setWrapStyleWord(true).
        // Root panel
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.Y_AXIS)).
        if let Some(tool_panel) = &self.tool_panel {
            self.pnl_root
                .add(&tool_panel.as_tool_panel().get_component());
        }
        self.pnl_root.add(&self.scr_task_log);
        ui_harness::INSTANCE.with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        Rc::clone(&self.pnl_root)
    }

    /// Java private final `toolType` (no getter in the source).
    pub fn tool_type(&self) -> ToolType {
        self.tool_type
    }

    /// The Java `toolPanel` field (no getter in the source; for the bridge
    /// and test driver, which find widgets by name below `getContainer`).
    pub fn tool_panel(&self) -> Option<ToolPanelInstance> {
        self.tool_panel.clone()
    }
}

impl LogInterface for ToolsDialog {
    /// Java `getManager()`.
    fn get_manager(&self) -> Option<&'static dyn BaseManager> {
        Some(self.manager)
    }

    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> Option<AxisID> {
        Some(self.axis_id)
    }

    /// Java `logMessage(String, AxisID, String[], String)`.
    fn log_message_string_axis_id_string_array_string(
        &self,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        message: Option<&[String]>,
        msg_id: Option<&str>,
    ) -> bool {
        self.logger
            .log_message_string_axis_id_string_array_string(title, axis_id, message, msg_id)
    }

    /// Java `logMessage(String, AxisID, ArrayList<String>)`.
    fn log_message_string_axis_id_array_list(
        &self,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        message: Option<&[String]>,
    ) {
        self.logger
            .log_message_string_axis_id_array_list(title, axis_id, message);
    }

    /// Java `logMessage(AxisID, ArrayList<String>)`.
    fn log_message_axis_id_array_list(&self, axis_id: Option<AxisID>, message: Option<&[String]>) {
        self.logger.log_message_axis_id_array_list(axis_id, message);
    }

    /// Java `logMessage(Loggable, AxisID)`.
    fn log_message_loggable_axis_id(
        &self,
        loggable: Option<&dyn Loggable>,
        axis_id: Option<AxisID>,
    ) {
        self.logger.log_message_loggable_axis_id(loggable, axis_id);
    }

    /// Java `logMessage(String, AxisID)`.
    fn log_message_string_axis_id(&self, title: Option<&str>, axis_id: Option<AxisID>) {
        self.logger.log_message_string_axis_id(title, axis_id);
    }

    /// Java `logMessage(String)`.
    fn log_message_string(&self, message: Option<&str>) {
        self.logger.log_message_string(message);
    }

    /// Java `logMessage(String, boolean, boolean, FileWriter)`.
    fn log_message_string_boolean_boolean_file_writer(
        &self,
        message: Option<&str>,
        timestamp: bool,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    ) {
        self.logger.log_message_string_boolean_boolean_file_writer(
            message,
            timestamp,
            newline,
            secondary_log,
        );
    }

    /// Java `logMessage(File, FileWriter)`.
    fn log_message_file_file_writer(
        &self,
        file: Option<&Path>,
        secondary_log: Option<FileWriterRef>,
    ) {
        self.logger
            .log_message_file_file_writer(file, secondary_log);
    }

    /// Java `logMessage(File, boolean, FileWriter)`.
    fn log_message_file_boolean_file_writer(
        &self,
        file: Option<&Path>,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    ) {
        self.logger
            .log_message_file_boolean_file_writer(file, newline, secondary_log);
    }

    /// Java `logMessagePrimaryLog(FileReader)`.
    fn log_message_primary_log(&self, reader: Option<FileReaderRef>) {
        self.logger.log_message_primary_log(reader);
    }

    /// Java `save()`: empty.
    fn save(&self) {}

    /// Java `setAllowPrimaryLogging(boolean)`.
    fn set_allow_primary_logging(&self, input: bool) {
        self.logger.set_allow_primary_logging(input);
    }

    /// Java `isAllowPrimaryLogging()`.
    fn is_allow_primary_logging(&self) -> bool {
        self.logger.is_allow_primary_logging()
    }

    /// Java `append(String)`.
    fn append(&self, line: &str) {
        self.ta_task_log.append(line);
    }

    /// Java `msgChanged()`: empty.
    fn msg_changed(&self) {}

    /// Java `getPrevLineEndOffset()`:
    /// `taTaskLog.getLineEndOffset(taTaskLog.getLineCount() - 1)`.  The end
    /// offset of the last line is the document length; the line count is
    /// never 0, so the BadLocationException is never thrown (as in
    /// `LogWindow`).
    fn get_prev_line_end_offset(&self) -> Result<usize, BadLocationException> {
        Ok(self.ta_task_log.get_text().encode_utf16().count())
    }
}

impl ContextMenu for ToolsDialog {
    /// Java `popUpContextMenu(MouseEvent)`: empty.
    fn pop_up_context_menu(&self, _mouse_event: &MouseEvent) {}
}
