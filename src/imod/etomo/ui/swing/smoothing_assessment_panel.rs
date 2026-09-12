//! `IMOD/Etomo/src/etomo/ui/swing/SmoothingAssessmentPanel.java`.
//!
//! Swing construction plus `ApplicationManager`, `ToolsManager`, `UIHarness`,
//! and autodoc loading are direct boundaries.  The panel retains the source's
//! field/default, parameter routing, validation order, and panel-id action
//! dispatch without creating a second flattenwarp controller.
#![allow(dead_code)]

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::multi_line_button::MultiLineButton;
use super::tilt_panel::Deferred3dmodButton;

pub const LAMBDA_FOR_SMOOTHING_LABEL: &str = "Smoothing factors to try";
pub const FLATTEN_WARP_LABEL: &str = "Run Flattenwarp to Assess Smoothing";
pub const LAMBDA_FOR_SMOOTHING_ASSESSMENT_DEFAULT: &str = "1,1.5,2,2.5,3";
pub const SMOOTHING_ASSESSMENT_OUTPUT_MODEL: &str = "_checkflat.mod";

/// Java `PanelId.POST_FLATTEN_VOLUME` and `PanelId.TOOLS_FLATTEN_VOLUME`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SmoothingAssessmentPanelId {
    PostFlattenVolume,
    ToolsFlattenVolume,
}

/// Java `SmoothingAssessmentParent`.
pub trait SmoothingAssessmentParent {
    fn is_one_surface(&self) -> bool;
    fn get_warp_spacing_x(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException>;
    fn get_warp_spacing_y(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException>;
}

/// The `FlattenWarpParam` methods used by this source unit.  Its command-line
/// construction belongs to `FlattenWarpParam.java`, still an explicit comscript boundary.
pub trait SmoothingAssessmentFlattenWarpParam {
    fn set_lambda_for_smoothing(&mut self, value: String) -> Option<String>;
    fn set_middle_contour_file(&mut self, value: String);
    fn set_one_surface(&mut self, value: bool);
    fn set_warp_spacing_x(&mut self, value: String) -> Option<String>;
    fn set_warp_spacing_y(&mut self, value: String) -> Option<String>;
}

/// `ConstMetaData` and `MetaData` calls made by this source unit.
pub trait SmoothingAssessmentConstMetaData {
    fn is_lambda_for_smoothing_list_empty(&self) -> bool;
    fn lambda_for_smoothing_list(&self) -> String;
}
pub trait SmoothingAssessmentMetaData {
    fn set_lambda_for_smoothing_list(&mut self, value: String);
}

/// The common direct calls made by `ApplicationManager` and `ToolsManager`.
/// The selected manager is determined solely by `panel_id`, as in Java.
pub trait SmoothingAssessmentManager {
    fn flatten_warp(
        &mut self,
        button: &MultiLineButton,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        axis_id: AxisID,
    );
    fn imod_view_model(&mut self, axis_id: AxisID, file_name: &str);
    fn open_message_dialog(&mut self, message: String, title: &str, axis_id: AxisID);
}

/// Source-visible Swing layout/listener state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SmoothingAssessmentPanelLayout {
    pub root_box_layout_y_axis: bool,
    pub root_border_title: Option<String>,
    pub root_component_order: Vec<String>,
    pub buttons_box_layout_x_axis: bool,
    pub buttons_component_order: Vec<String>,
    pub flatten_warp_deferred_3dmod_button: bool,
    pub action_listener_registered: bool,
}

/// Java final `SmoothingAssessmentPanel`.
pub struct SmoothingAssessmentPanel<P: SmoothingAssessmentParent> {
    pub pnl_root: SmoothingAssessmentPanelLayout,
    pub ltf_lambda_for_smoothing: LabeledTextField,
    pub btn_3dmod: MultiLineButton,
    pub btn_flatten_warp: MultiLineButton,
    pub axis_id: AxisID,
    pub parent: P,
    pub dialog_type: DialogType,
    pub panel_id: SmoothingAssessmentPanelId,
    /// `AutodocFactory`/`EtomoAutodoc.getTooltip` result at the direct storage boundary.
    pub flatten_warp_autodoc_tooltip: Option<String>,
}

impl<P: SmoothingAssessmentParent> SmoothingAssessmentPanel<P> {
    /// Java private `SmoothingAssessmentPanel(ApplicationManager, ...)` constructor.
    pub fn new_post(
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: P,
        flatten_warp_button: MultiLineButton,
    ) -> Self {
        Self::new(
            axis_id,
            dialog_type,
            parent,
            SmoothingAssessmentPanelId::PostFlattenVolume,
            flatten_warp_button,
        )
    }

    /// Java private `SmoothingAssessmentPanel(ToolsManager, ...)` constructor.
    pub fn new_tools(axis_id: AxisID, dialog_type: DialogType, parent: P) -> Self {
        Self::new(
            axis_id,
            dialog_type,
            parent,
            SmoothingAssessmentPanelId::ToolsFlattenVolume,
            MultiLineButton::new_with_label(Some(FLATTEN_WARP_LABEL)),
        )
    }

    fn new(
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: P,
        panel_id: SmoothingAssessmentPanelId,
        mut btn_flatten_warp: MultiLineButton,
    ) -> Self {
        btn_flatten_warp.set_action_command(Some(FLATTEN_WARP_LABEL));
        let mut btn_3dmod = MultiLineButton::new_with_label(Some("Open Assessment in 3dmod"));
        btn_3dmod.set_action_command(Some("Open Assessment in 3dmod"));
        Self {
            pnl_root: SmoothingAssessmentPanelLayout::default(),
            ltf_lambda_for_smoothing: LabeledTextField::new(
                FieldType::FloatingPointArray,
                &format!("{LAMBDA_FOR_SMOOTHING_LABEL}: "),
            ),
            btn_3dmod,
            btn_flatten_warp,
            axis_id,
            parent,
            dialog_type,
            panel_id,
            flatten_warp_autodoc_tooltip: None,
        }
    }

    /// Java static `getPostInstance`.
    pub fn get_post_instance(
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: P,
        flatten_warp_button: MultiLineButton,
    ) -> Self {
        let mut instance = Self::new_post(axis_id, dialog_type, parent, flatten_warp_button);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java static `getToolsInstance`.
    pub fn get_tools_instance(axis_id: AxisID, dialog_type: DialogType, parent: P) -> Self {
        let mut instance = Self::new_tools(axis_id, dialog_type, parent);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_flatten_warp.add_action_listener();
        self.btn_3dmod.add_action_listener();
        self.pnl_root.action_listener_registered = true;
    }

    /// Java `done`.
    pub fn done(&mut self) {
        self.btn_flatten_warp.remove_action_listener();
    }

    /// Java private `createPanel`.
    pub fn create_panel(&mut self) {
        self.pnl_root.flatten_warp_deferred_3dmod_button = true;
        self.ltf_lambda_for_smoothing
            .set_text(LAMBDA_FOR_SMOOTHING_ASSESSMENT_DEFAULT);
        self.pnl_root.root_box_layout_y_axis = true;
        self.pnl_root.root_border_title = Some("Smoothing Assessment".into());
        self.pnl_root.root_component_order = vec![
            "ltfLambdaForSmoothing".into(),
            "FixedDim.x0_y5".into(),
            "pnlButtons".into(),
        ];
        self.pnl_root.buttons_box_layout_x_axis = true;
        self.pnl_root.buttons_component_order = vec![
            "btnFlattenWarp".into(),
            "FixedDim.x5_y0".into(),
            "btn3dmod".into(),
        ];
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters<M: SmoothingAssessmentConstMetaData>(&mut self, meta_data: &M) {
        if !meta_data.is_lambda_for_smoothing_list_empty() {
            self.ltf_lambda_for_smoothing
                .set_text(&meta_data.lambda_for_smoothing_list());
        }
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_meta<M: SmoothingAssessmentMetaData>(&self, meta_data: &mut M) {
        meta_data.set_lambda_for_smoothing_list(self.ltf_lambda_for_smoothing.get_text());
    }

    /// Java private `validateFlattenWarp`.
    pub fn validate_flatten_warp<M: SmoothingAssessmentManager>(&self, manager: &mut M) -> bool {
        if self.ltf_lambda_for_smoothing.is_empty() {
            manager.open_message_dialog(
                format!("{LAMBDA_FOR_SMOOTHING_LABEL} is required."),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        true
    }

    /// Java `getParameters(FlattenWarpParam, boolean)`.
    pub fn get_parameters<F: SmoothingAssessmentFlattenWarpParam, M: SmoothingAssessmentManager>(
        &self,
        param: &mut F,
        do_validation: bool,
        manager: &mut M,
        smoothing_assessment_output_model: &str,
    ) -> bool {
        let lambda_for_smoothing = match self
            .ltf_lambda_for_smoothing
            .get_text_validated(do_validation)
        {
            Ok(value) => value,
            Err(_) => return false,
        };
        if let Some(error_message) = param.set_lambda_for_smoothing(lambda_for_smoothing) {
            manager.open_message_dialog(
                format!("Error in {LAMBDA_FOR_SMOOTHING_LABEL}:  {error_message}"),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        param.set_middle_contour_file(smoothing_assessment_output_model.into());
        param.set_one_surface(self.parent.is_one_surface());
        let warp_spacing_x = match self.parent.get_warp_spacing_x(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        if let Some(error_message) = param.set_warp_spacing_x(warp_spacing_x) {
            manager.open_message_dialog(
                format!("Error in Warp spacing X:  {error_message}"),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        let warp_spacing_y = match self.parent.get_warp_spacing_y(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        if let Some(error_message) = param.set_warp_spacing_y(warp_spacing_y) {
            manager.open_message_dialog(
                format!("Error in Warp spacing Y:  {error_message}"),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        true
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: SmoothingAssessmentManager>(
        &self,
        command: &str,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
        manager: &mut M,
    ) {
        let flatten_warp_command = self
            .btn_flatten_warp
            .get_action_command()
            .unwrap_or_default();
        let btn_3dmod_command = self.btn_3dmod.get_action_command().unwrap_or_default();
        match self.panel_id {
            SmoothingAssessmentPanelId::PostFlattenVolume
            | SmoothingAssessmentPanelId::ToolsFlattenVolume
                if command == flatten_warp_command =>
            {
                if self.validate_flatten_warp(manager) {
                    manager.flatten_warp(
                        &self.btn_flatten_warp,
                        deferred_3dmod_button,
                        options,
                        self.dialog_type,
                        self.axis_id,
                    );
                }
            }
            SmoothingAssessmentPanelId::PostFlattenVolume
            | SmoothingAssessmentPanelId::ToolsFlattenVolume
                if command == btn_3dmod_command =>
            {
                manager.imod_view_model(self.axis_id, SMOOTHING_ASSESSMENT_OUTPUT_MODEL);
            }
            _ => panic!("Unknown command {command}"),
        }
    }

    /// Java `setTooltips`.  Autodoc retrieval is external; if its source string
    /// is supplied through the retained boundary field, preserve Java's concatenation.
    pub fn set_tooltips(&mut self) {
        self.btn_flatten_warp
            .set_tool_tip_text(Some("Run flattenwarp with different smoothing factors."));
        self.btn_3dmod
            .set_tool_tip_text(Some("Open model created by flattenwarp."));
        if let Some(autodoc_tooltip) = &self.flatten_warp_autodoc_tooltip {
            self.ltf_lambda_for_smoothing
                .set_tool_tip_text(Some(&format!(
                    "A list of different LambdaForSmoothing values.  {autodoc_tooltip}"
                )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Parent {
        one_surface: bool,
        x: String,
        y: String,
    }
    impl SmoothingAssessmentParent for Parent {
        fn is_one_surface(&self) -> bool {
            self.one_surface
        }
        fn get_warp_spacing_x(&self, _: bool) -> Result<String, FieldValidationFailedException> {
            Ok(self.x.clone())
        }
        fn get_warp_spacing_y(&self, _: bool) -> Result<String, FieldValidationFailedException> {
            Ok(self.y.clone())
        }
    }
    #[derive(Default)]
    struct Param {
        lambda: String,
        middle: String,
        one: bool,
        x: String,
        y: String,
    }
    impl SmoothingAssessmentFlattenWarpParam for Param {
        fn set_lambda_for_smoothing(&mut self, value: String) -> Option<String> {
            self.lambda = value;
            None
        }
        fn set_middle_contour_file(&mut self, value: String) {
            self.middle = value
        }
        fn set_one_surface(&mut self, value: bool) {
            self.one = value
        }
        fn set_warp_spacing_x(&mut self, value: String) -> Option<String> {
            self.x = value;
            None
        }
        fn set_warp_spacing_y(&mut self, value: String) -> Option<String> {
            self.y = value;
            None
        }
    }
    #[derive(Default)]
    struct Manager {
        flatten: usize,
        model: usize,
        messages: Vec<String>,
    }
    impl SmoothingAssessmentManager for Manager {
        fn flatten_warp(
            &mut self,
            _: &MultiLineButton,
            _: Option<&Deferred3dmodButton>,
            _: Option<Run3dmodMenuOptions>,
            _: DialogType,
            _: AxisID,
        ) {
            self.flatten += 1
        }
        fn imod_view_model(&mut self, _: AxisID, _: &str) {
            self.model += 1
        }
        fn open_message_dialog(&mut self, message: String, _: &str, _: AxisID) {
            self.messages.push(message)
        }
    }
    fn panel() -> SmoothingAssessmentPanel<Parent> {
        SmoothingAssessmentPanel::get_tools_instance(
            AxisID::Only,
            DialogType::PostProcessing,
            Parent {
                one_surface: true,
                x: "4".into(),
                y: "5".into(),
            },
        )
    }

    #[test]
    fn construction_preserves_default_layout_and_listeners() {
        let panel = panel();
        assert_eq!(
            panel.ltf_lambda_for_smoothing.get_text(),
            LAMBDA_FOR_SMOOTHING_ASSESSMENT_DEFAULT
        );
        assert!(panel.pnl_root.flatten_warp_deferred_3dmod_button);
        assert_eq!(panel.pnl_root.root_component_order.len(), 3);
        assert_eq!(panel.btn_flatten_warp.button.action_listener_count, 1);
    }
    #[test]
    fn parameters_follow_source_order() {
        let panel = panel();
        let mut param = Param::default();
        let mut manager = Manager::default();
        assert!(panel.get_parameters(&mut param, true, &mut manager, "dataseta_checkflat.mod"));
        assert_eq!(param.lambda, LAMBDA_FOR_SMOOTHING_ASSESSMENT_DEFAULT);
        assert_eq!(param.middle, "dataseta_checkflat.mod");
        assert!(param.one);
        assert_eq!((param.x.as_str(), param.y.as_str()), ("4", "5"));
    }
    #[test]
    fn action_rejects_blank_then_routes_each_known_command() {
        let mut panel = panel();
        let mut manager = Manager::default();
        panel.ltf_lambda_for_smoothing.set_text(" ");
        panel.action(FLATTEN_WARP_LABEL, None, None, &mut manager);
        assert_eq!(manager.flatten, 0);
        assert_eq!(manager.messages.len(), 1);
        panel.ltf_lambda_for_smoothing.set_text("1");
        panel.action(FLATTEN_WARP_LABEL, None, None, &mut manager);
        panel.action("Open Assessment in 3dmod", None, None, &mut manager);
        assert_eq!((manager.flatten, manager.model), (1, 1));
    }
}
