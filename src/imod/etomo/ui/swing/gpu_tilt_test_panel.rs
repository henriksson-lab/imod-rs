//! `IMOD/Etomo/src/etomo/ui/swing/GpuTiltTestPanel.java`.
//!
//! Swing widget attachment, `DatasetTool.validateDatasetName`, and the
//! `ToolsManager.gpuTiltTest` launch are explicit native GUI/application
//! boundaries.  This unit retains the Java panel's controls, construction
//! order, validation order, popup contents, and action dispatch.
#![allow(dead_code)]

use std::path::Path;

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::abstract_frame::ComponentState;
use super::context_popup::{ContextPopup, MouseEvent, TOMO_GUIDE};
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::tool_panel::ToolPanel;

/// Java private static final `DATASET_ROOT`.
pub const DATASET_ROOT: &str = "gputest";

/// The two writes made to `GpuTiltTestParam` by `getParameters`.
pub trait GpuTiltTestParameters {
    fn set_n_minutes(&mut self, input: String);
    fn set_gpu_number(&mut self, input: i32);
}

/// Direct `DatasetTool` and `ToolsManager` calls made by `action`.
pub trait GpuTiltTestPanelManager {
    fn property_user_dir(&self) -> &Path;
    fn validate_dataset_name(
        &mut self,
        axis_id: AxisID,
        user_dir: &Path,
        dataset_root: &str,
        data_file_type: DataFileType,
        dataset_name: bool,
    ) -> bool;
    fn gpu_tilt_test(&mut self, axis_id: AxisID);
}

/// Source-visible `JPanel` structure created by Java `createPanel`.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuTiltTestPanelLayout {
    pub root_box_layout_y_axis: bool,
    pub fields_box_layout_x_axis: bool,
    pub root_component_order: Vec<String>,
    pub fields_component_order: Vec<String>,
    pub run_test_alignment_x: f32,
    pub mouse_listener_count: usize,
    pub action_listener_count: usize,
    pub tooltip_initialized: bool,
}

impl Default for GpuTiltTestPanelLayout {
    fn default() -> Self {
        Self {
            root_box_layout_y_axis: false,
            fields_box_layout_x_axis: false,
            root_component_order: Vec::new(),
            fields_component_order: Vec::new(),
            run_test_alignment_x: 0.5,
            mouse_listener_count: 0,
            action_listener_count: 0,
            tooltip_initialized: false,
        }
    }
}

/// Java `GpuTiltTestPanel`.
pub struct GpuTiltTestPanel {
    pub pnl_root: ComponentState,
    pub layout: GpuTiltTestPanelLayout,
    pub ltf_n_minutes: LabeledTextField,
    pub sp_gpu_number: LabeledSpinner,
    pub btn_run_test: MultiLineButton,
    pub axis_id: AxisID,
    /// The most recently constructed Java `ContextPopup`.
    pub last_context_popup: Option<ContextPopup>,
}

impl GpuTiltTestPanel {
    /// Java private `GpuTiltTestPanel(ToolsManager, AxisID)`.
    pub fn new(axis_id: AxisID) -> Self {
        Self {
            pnl_root: ComponentState::default(),
            layout: GpuTiltTestPanelLayout::default(),
            ltf_n_minutes: LabeledTextField::new(FieldType::FloatingPoint, "# of minutes: "),
            sp_gpu_number: LabeledSpinner::get_instance("GPU #: ", 0, 0, 8, 1),
            btn_run_test: MultiLineButton::new_with_label(Some("Run GPU Test")),
            axis_id,
            last_context_popup: None,
        }
    }

    /// Java static `getInstance(ToolsManager, AxisID)`.
    pub fn get_instance(axis_id: AxisID) -> Self {
        let mut instance = Self::new(axis_id);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel`.
    pub fn create_panel(&mut self) {
        self.ltf_n_minutes.set_text("1");
        self.ltf_n_minutes.set_preferred_width(80, None);
        self.btn_run_test.set_alignment_x(0.5);
        self.layout.root_box_layout_y_axis = true;
        self.layout.fields_box_layout_x_axis = true;
        self.layout.run_test_alignment_x = 0.5;
        self.layout.root_component_order = vec![
            "FixedDim.x0_y5".into(),
            "pnlFields".into(),
            "FixedDim.x0_y23".into(),
            "btnRunTest".into(),
            "FixedDim.x0_y15".into(),
        ];
        self.layout.fields_component_order = vec![
            "FixedDim.x10_y0".into(),
            "ltfNMinutes".into(),
            "FixedDim.x15_y0".into(),
            "spGpuNumber".into(),
            "FixedDim.x10_y0".into(),
        ];
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.layout.mouse_listener_count += 1;
        self.btn_run_test.add_action_listener();
        self.layout.action_listener_count += 1;
    }

    /// Java `getParameters(GpuTiltTestParam, boolean)`.
    pub fn get_parameters<P: GpuTiltTestParameters>(
        &self,
        param: &mut P,
        do_validation: bool,
    ) -> bool {
        match self.ltf_n_minutes.get_text_validated(do_validation) {
            Ok(n_minutes) => {
                param.set_n_minutes(n_minutes);
                param.set_gpu_number(self.sp_gpu_number.get_value());
                true
            }
            Err(_) => false,
        }
    }

    /// Java private `action`.
    pub fn action<M: GpuTiltTestPanelManager>(&self, manager: &mut M) {
        let property_user_dir = manager.property_user_dir().to_owned();
        if !manager.validate_dataset_name(
            self.axis_id,
            &property_user_dir,
            DATASET_ROOT,
            DataFileType::Tools,
            true,
        ) {
            return;
        }
        manager.gpu_tilt_test(self.axis_id);
    }

    /// Java `getComponent`.
    pub fn get_component(&self) -> &ComponentState {
        &self.pnl_root
    }

    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        let man_page_label = ["gputilttest".into()];
        let man_page = ["gputilttest.html".into()];
        let log_file_label = ["GPU test".into()];
        let log_file = [format!("{DATASET_ROOT}.log")];
        self.last_context_popup = ContextPopup::new_log_files(
            mouse_event,
            Some("GPU Test"),
            TOMO_GUIDE,
            &man_page_label,
            &man_page,
            &log_file_label,
            &log_file,
            self.axis_id,
            None,
        )
        .ok();
    }

    /// Java private `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.ltf_n_minutes
            .set_tool_tip_text(Some("The number of minutes to run the test"));
        self.sp_gpu_number.set_tool_tip_text(Some(
            "The GPU number, numbered from 1.  When 0 is selected, the fastest GPU will be used.",
        ));
        self.btn_run_test.set_tool_tip_text(Some(
            "Test the reliability of the GPU by using gputilttest to run the Tilt program repeatedly.",
        ));
        self.layout.tooltip_initialized = true;
    }

    /// Java inner `GpuTiltTestActionListener.actionPerformed(ActionEvent)`.
    pub fn gpu_tilt_test_action_listener_action_performed<M: GpuTiltTestPanelManager>(
        &self,
        manager: &mut M,
    ) {
        self.action(manager);
    }
}

impl ToolPanel for GpuTiltTestPanel {
    fn get_component(&self) -> &ComponentState {
        self.get_component()
    }
}

impl super::tools_dialog::GpuTiltTestPanel for GpuTiltTestPanel {
    fn get_parameters(
        &mut self,
        param: &mut super::tools_dialog::GpuTiltTestParam,
        do_validation: bool,
    ) -> bool {
        self.get_parameters(param, do_validation)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    #[derive(Default)]
    struct Param {
        n_minutes: Option<String>,
        gpu_number: Option<i32>,
    }
    impl GpuTiltTestParameters for Param {
        fn set_n_minutes(&mut self, input: String) {
            self.n_minutes = Some(input);
        }
        fn set_gpu_number(&mut self, input: i32) {
            self.gpu_number = Some(input);
        }
    }

    struct Manager {
        directory: PathBuf,
        valid: bool,
        launches: Vec<AxisID>,
    }
    impl GpuTiltTestPanelManager for Manager {
        fn property_user_dir(&self) -> &Path {
            &self.directory
        }
        fn validate_dataset_name(
            &mut self,
            _axis_id: AxisID,
            _user_dir: &Path,
            dataset_root: &str,
            data_file_type: DataFileType,
            dataset_name: bool,
        ) -> bool {
            assert_eq!(dataset_root, DATASET_ROOT);
            assert_eq!(data_file_type, DataFileType::Tools);
            assert!(dataset_name);
            self.valid
        }
        fn gpu_tilt_test(&mut self, axis_id: AxisID) {
            self.launches.push(axis_id);
        }
    }

    #[test]
    fn source_construction_and_parameter_order_are_preserved() {
        let mut panel = GpuTiltTestPanel::get_instance(AxisID::First);
        assert_eq!(panel.ltf_n_minutes.text, "1");
        assert_eq!(panel.layout.mouse_listener_count, 1);
        assert_eq!(panel.layout.action_listener_count, 1);
        panel.sp_gpu_number.set_value_int(4);
        let mut param = Param::default();
        assert!(panel.get_parameters(&mut param, true));
        assert_eq!(param.n_minutes.as_deref(), Some("1"));
        assert_eq!(param.gpu_number, Some(4));
    }

    #[test]
    fn action_validates_dataset_before_launching() {
        let panel = GpuTiltTestPanel::get_instance(AxisID::Second);
        let mut manager = Manager {
            directory: PathBuf::from("."),
            valid: false,
            launches: Vec::new(),
        };
        panel.action(&mut manager);
        assert!(manager.launches.is_empty());
        manager.valid = true;
        panel.gpu_tilt_test_action_listener_action_performed(&mut manager);
        assert_eq!(manager.launches, vec![AxisID::Second]);
    }

    #[test]
    fn context_popup_keeps_the_source_manual_and_log_file() {
        let mut panel = GpuTiltTestPanel::get_instance(AxisID::Only);
        panel.pop_up_context_menu(MouseEvent {
            x: 3,
            y: 4,
            right_mouse_button: false,
        });
        let popup = panel.last_context_popup.as_ref().unwrap();
        assert_eq!(popup.anchor.as_deref(), Some("GPU Test"));
        assert_eq!(
            popup.man_page_name.as_ref().unwrap(),
            &["gputilttest.html#TOP"]
        );
        assert_eq!(popup.log_file_name.as_ref().unwrap(), &["gputest.log"]);
    }
}
