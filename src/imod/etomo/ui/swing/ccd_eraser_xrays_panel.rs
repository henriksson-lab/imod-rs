//! `IMOD/Etomo/src/etomo/ui/swing/CcdEraserXRaysPanel.java`.
//!
//! Swing construction, autodoc loading, file naming, and the concrete
//! `ApplicationManager` are direct GUI/application boundaries.  This module
//! retains the Java panel's control state, validation order, enablement, and
//! action routing without introducing another process controller.
#![allow(dead_code)]

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::check_box::CheckBox;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::multi_line_button::MultiLineButton;
use super::tilt_panel::Deferred3dmodButton;

pub const ERASE_LABEL: &str = "Create Fixed Stack";
pub const USE_FIXED_STACK_LABEL: &str = "Use Fixed Stack";

/// Java `ConstCCDEraserParam` reads performed by `setParameters`.
pub trait ConstCcdEraserParam {
    fn find_peaks(&self) -> bool;
    fn value(&self, key: &str) -> String;
    fn model_file(&self) -> String;
    fn include_adjacent_points(&self) -> bool;
}

/// Java `CCDEraserParam` writes performed by `getParameters`.
pub trait CcdEraserParam: ConstCcdEraserParam {
    fn set_find_peaks(&mut self, value: bool);
    fn set_value(&mut self, key: &str, value: String);
    fn set_model_file(&mut self, value: String);
    fn set_include_adjacent_points(&mut self, value: bool);
}

/// Java `BaseScreenState` use by the manual-replacement `PanelHeader`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CcdEraserScreenState {
    pub manual_replacement_advanced: bool,
}

/// The direct `ApplicationManager` dispatch boundary from this Java source unit.
pub trait CcdEraserXRaysApplicationManager {
    fn find_xrays(
        &mut self,
        axis: AxisID,
        options: Option<Run3dmodMenuOptions>,
        dialog: DialogType,
    );
    fn pre_eraser(
        &mut self,
        axis: AxisID,
        options: Option<Run3dmodMenuOptions>,
        dialog: DialogType,
    );
    fn replace_raw_stack(&mut self, axis: AxisID, dialog: DialogType);
    fn imod_xray_model(&mut self, axis: AxisID, options: Option<Run3dmodMenuOptions>);
    fn imod_manual_erase(
        &mut self,
        axis: AxisID,
        options: Option<Run3dmodMenuOptions>,
        dialog: DialogType,
    );
    fn imod_erased_stack(&mut self, axis: AxisID, options: Option<Run3dmodMenuOptions>);
    fn clip_stats(&mut self, axis: AxisID, fixed: bool, dialog: DialogType);
    fn pack(&mut self, axis: AxisID);
}

/// Source-shaped UI hierarchy state; actual Swing/Slint widget installation stays
/// outside this panel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CcdEraserXRaysPanelLayout {
    pub automatic_component_order: Vec<&'static str>,
    pub manual_component_order: Vec<&'static str>,
    pub root_component_order: Vec<&'static str>,
    pub manual_body_visible: bool,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub last_context_menu: Option<String>,
}

/// Java final `CcdEraserXRaysPanel` source-owned state.
#[derive(Clone, Debug)]
pub struct CcdEraserXRaysPanel {
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub layout: CcdEraserXRaysPanelLayout,
    pub cb_xray_replacement: CheckBox,
    pub ltf_peak_criterion: LabeledTextField,
    pub ltf_diff_criterion: LabeledTextField,
    pub ltf_grow_criterion: LabeledTextField,
    pub ltf_edge_exclusion: LabeledTextField,
    pub ltf_maximum_radius: LabeledTextField,
    pub ltf_annulus_width: LabeledTextField,
    pub ltf_scan_region_size: LabeledTextField,
    pub ltf_scan_criterion: LabeledTextField,
    pub btn_find_xrays: MultiLineButton,
    pub btn_view_xray_model: MultiLineButton,
    pub cb_manual_replacement: CheckBox,
    pub ltf_global_replacement_list: LabeledTextField,
    pub ltf_local_replacement_list: LabeledTextField,
    pub ltf_boundary_replacement_list: LabeledTextField,
    pub btn_create_model: MultiLineButton,
    pub ltf_border_pixels: LabeledTextField,
    pub ltf_polynomial_order: LabeledTextField,
    pub cb_include_adjacent_points: CheckBox,
    pub btn_view_erased: MultiLineButton,
    pub btn_clip_stats_raw: MultiLineButton,
    pub btn_clip_stats_fixed: MultiLineButton,
    pub ltf_giant_criterion: LabeledTextField,
    pub ltf_big_diff_criterion: LabeledTextField,
    pub ltf_extra_large_radius: LabeledTextField,
    pub btn_erase: MultiLineButton,
    pub btn_replace_raw_stack: MultiLineButton,
    /// `FileType.MANUAL_REPLACEMENT_MODEL.getFileName(applicationManager, axisID)`.
    /// FileType/application dataset expansion is retained as a direct boundary.
    pub manual_replacement_model_file: String,
    pub actions_attached: bool,
}

impl CcdEraserXRaysPanel {
    /// Java private constructor `CcdEraserXRaysPanel(...)`.
    pub fn new(axis_id: AxisID, dialog_type: DialogType, global_advanced: bool) -> Self {
        let mut panel = Self {
            axis_id,
            dialog_type,
            layout: CcdEraserXRaysPanelLayout {
                automatic_component_order: vec![
                    "xray-replacement",
                    "peak",
                    "difference",
                    "maximum-radius",
                    "giant",
                    "big-difference",
                    "extra-large-radius",
                    "grow",
                    "edge-exclusion",
                    "annulus-width",
                    "xy-scan-size",
                    "scan-criterion",
                    "buttons",
                ],
                manual_component_order: vec![
                    "manual-replacement",
                    "all-sections",
                    "line",
                    "boundary",
                    "create-model",
                ],
                root_component_order: vec![
                    "automatic",
                    "manual",
                    "border-pixels",
                    "polynomial-order",
                    "adjacent-points",
                    "erase-buttons",
                ],
                manual_body_visible: global_advanced,
                ..Default::default()
            },
            cb_xray_replacement: CheckBox::new_with_text("Automatic x-ray replacement"),
            ltf_peak_criterion: LabeledTextField::new(FieldType::FloatingPoint, "Peak criterion:"),
            ltf_diff_criterion: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Difference criterion:",
            ),
            ltf_grow_criterion: LabeledTextField::new(FieldType::FloatingPoint, "Grow criterion:"),
            ltf_edge_exclusion: LabeledTextField::new(FieldType::Integer, "Edge exclusion:"),
            ltf_maximum_radius: LabeledTextField::new(FieldType::FloatingPoint, "Maximum radius:"),
            ltf_annulus_width: LabeledTextField::new(FieldType::FloatingPoint, "Annulus width:"),
            ltf_scan_region_size: LabeledTextField::new(FieldType::Integer, "XY scan size:"),
            ltf_scan_criterion: LabeledTextField::new(FieldType::FloatingPoint, "Scan criterion:"),
            btn_find_xrays: MultiLineButton::new_with_label(Some("Find X-rays")),
            btn_view_xray_model: MultiLineButton::new_with_label(Some("View X-ray Model")),
            cb_manual_replacement: CheckBox::new_with_text("Manual replacement"),
            ltf_global_replacement_list: LabeledTextField::new(
                FieldType::IntegerList,
                "All section replacement list: ",
            ),
            ltf_local_replacement_list: LabeledTextField::new(
                FieldType::IntegerList,
                "Line replacement list: ",
            ),
            ltf_boundary_replacement_list: LabeledTextField::new(
                FieldType::IntegerList,
                "Boundary replacement list: ",
            ),
            btn_create_model: MultiLineButton::new_with_label(Some(
                "Create Manual Replacement Model",
            )),
            ltf_border_pixels: LabeledTextField::new(FieldType::Integer, "Border pixels: "),
            ltf_polynomial_order: LabeledTextField::new(FieldType::Integer, "Polynomial order: "),
            cb_include_adjacent_points: CheckBox::new_with_text("Include adjacent points"),
            btn_view_erased: MultiLineButton::new_with_label(Some("View Fixed Stack")),
            btn_clip_stats_raw: MultiLineButton::new_with_label(Some("Show Min/Max for Raw Stack")),
            btn_clip_stats_fixed: MultiLineButton::new_with_label(Some(
                "Show Min/Max for Fixed Stack",
            )),
            ltf_giant_criterion: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Extra-large peak criterion:",
            ),
            ltf_big_diff_criterion: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Extra-large difference criterion:",
            ),
            ltf_extra_large_radius: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Maximum radius of extra-large peak:",
            ),
            btn_erase: MultiLineButton::new_with_label(Some(ERASE_LABEL)),
            btn_replace_raw_stack: MultiLineButton::new_with_label(Some(USE_FIXED_STACK_LABEL)),
            manual_replacement_model_file: String::new(),
            actions_attached: false,
        };
        panel.set_tool_tip_text();
        panel.enable_xray_replacement();
        panel.enable_manual_replacement();
        panel.update_manual_replacement_advanced(global_advanced);
        panel
    }

    /// Java static `getInstance`.
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType, global_advanced: bool) -> Self {
        let mut instance = Self::new(axis_id, dialog_type, global_advanced);
        instance.add_listeners();
        instance
    }

    /// Java `addListeners`.
    pub fn add_listeners(&mut self) {
        self.layout.listener_count += 10;
        for button in [
            &mut self.btn_find_xrays,
            &mut self.btn_view_xray_model,
            &mut self.btn_create_model,
            &mut self.btn_erase,
            &mut self.btn_view_erased,
            &mut self.btn_replace_raw_stack,
            &mut self.btn_clip_stats_raw,
            &mut self.btn_clip_stats_fixed,
        ] {
            button.add_action_listener();
        }
        self.cb_xray_replacement.add_action_listener();
        self.cb_manual_replacement.add_action_listener();
        self.actions_attached = true;
    }

    /// Java `setParameters(ConstCCDEraserParam)`.
    pub fn set_parameters<P: ConstCcdEraserParam>(&mut self, param: &P) {
        self.cb_xray_replacement.set_selected(param.find_peaks());
        self.ltf_peak_criterion
            .set_text(&param.value("peak-criterion"));
        self.ltf_diff_criterion
            .set_text(&param.value("diff-criterion"));
        self.ltf_grow_criterion
            .set_text(&param.value("grow-criterion"));
        self.ltf_scan_criterion
            .set_text(&param.value("scan-criterion"));
        self.ltf_maximum_radius
            .set_text(&param.value("maximum-radius"));
        self.ltf_annulus_width
            .set_text(&param.value("annulus-width"));
        self.ltf_scan_region_size
            .set_text(&param.value("xy-scan-size"));
        self.ltf_edge_exclusion
            .set_text(&param.value("edge-exclusion"));
        self.ltf_global_replacement_list
            .set_text(&param.value("global-replacement-list"));
        self.ltf_local_replacement_list
            .set_text(&param.value("local-replacement-list"));
        self.ltf_boundary_replacement_list
            .set_text(&param.value("boundary-replacement-list"));
        self.ltf_border_pixels
            .set_text(&param.value("border-pixels"));
        self.ltf_polynomial_order
            .set_text(&param.value("polynomial-order"));
        self.ltf_giant_criterion
            .set_text(&param.value("giant-criterion"));
        self.ltf_big_diff_criterion
            .set_text(&param.value("big-diff-criterion"));
        self.ltf_extra_large_radius
            .set_text(&param.value("extra-large-radius"));
        self.cb_manual_replacement
            .set_selected(!param.model_file().is_empty());
        self.cb_include_adjacent_points
            .set_selected(param.include_adjacent_points());
        self.enable_xray_replacement();
        self.enable_manual_replacement();
    }

    /// Java `done`.
    pub fn done(&mut self) {
        self.actions_attached = false;
    }
    /// Java `setParameters(BaseScreenState)`.
    pub fn set_screen_state(&mut self, state: &CcdEraserScreenState) {
        self.layout.manual_body_visible = state.manual_replacement_advanced;
    }
    /// Java `getParameters(BaseScreenState)`.
    pub fn get_screen_state(&self, state: &mut CcdEraserScreenState) {
        state.manual_replacement_advanced = self.layout.manual_body_visible;
    }

    /// Java `getParameters(CCDEraserParam, boolean)`.
    pub fn get_parameters<P: CcdEraserParam>(&self, param: &mut P, do_validation: bool) -> bool {
        param.set_find_peaks(self.cb_xray_replacement.is_selected());
        let value = match self.ltf_peak_criterion.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("peak-criterion", value);
        let value = match self.ltf_diff_criterion.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("diff-criterion", value);
        let value = match self.ltf_grow_criterion.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("grow-criterion", value);
        let value = match self.ltf_scan_criterion.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("scan-criterion", value);
        let value = match self.ltf_maximum_radius.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("maximum-radius", value);
        let value = match self.ltf_annulus_width.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("annulus-width", value);
        let value = match self.ltf_scan_region_size.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("xy-scan-size", value);
        let value = match self.ltf_edge_exclusion.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("edge-exclusion", value);
        let value = match self
            .ltf_global_replacement_list
            .get_text_validated(do_validation)
        {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("global-replacement-list", value);
        let value = match self
            .ltf_local_replacement_list
            .get_text_validated(do_validation)
        {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("local-replacement-list", value);
        let value = match self
            .ltf_boundary_replacement_list
            .get_text_validated(do_validation)
        {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("boundary-replacement-list", value);
        let value = match self.ltf_border_pixels.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("border-pixels", value);
        let value = match self.ltf_polynomial_order.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("polynomial-order", value);
        let value = match self.ltf_giant_criterion.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("giant-criterion", value);
        let value = match self
            .ltf_big_diff_criterion
            .get_text_validated(do_validation)
        {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("big-diff-criterion", value);
        let value = match self
            .ltf_extra_large_radius
            .get_text_validated(do_validation)
        {
            Ok(value) => value,
            Err(_) => return false,
        };
        param.set_value("extra-large-radius", value);
        param.set_include_adjacent_points(self.cb_include_adjacent_points.is_selected());
        param.set_model_file(if self.cb_manual_replacement.is_selected() {
            self.manual_replacement_model_file.clone()
        } else {
            String::new()
        });
        true
    }
    /// Java `getParameters(MakecomfileParam, boolean)`.
    pub fn get_makecomfile_parameters(&self, _do_validation: bool) -> bool {
        true
    }
    /// Java `getContainer` native-widget boundary.
    pub fn get_container(&self) -> &CcdEraserXRaysPanelLayout {
        &self.layout
    }

    /// Java `updateAdvanced`.
    pub fn update_advanced(&mut self, state: bool) {
        self.cb_xray_replacement.set_visible(state);
        self.ltf_grow_criterion.set_visible(state);
        self.ltf_edge_exclusion.set_visible(state);
        self.ltf_annulus_width.set_visible(state);
        self.ltf_scan_region_size.set_visible(state);
        self.ltf_scan_criterion.set_visible(state);
        self.ltf_border_pixels.set_visible(state);
        self.ltf_polynomial_order.set_visible(state);
        self.cb_include_adjacent_points.set_visible(state);
        self.ltf_giant_criterion.set_visible(state);
        self.ltf_extra_large_radius.set_visible(state);
    }
    /// Java `updateManualReplacementAdvanced`.
    pub fn update_manual_replacement_advanced(&mut self, advanced: bool) {
        self.layout.manual_body_visible = advanced;
    }
    /// Java `expand(GlobalExpandButton)`.
    pub fn expand_global(&mut self, expanded: bool) {
        self.update_advanced(expanded);
    }
    /// Java `expand(ExpandButton)`.
    pub fn expand_manual_replacement(&mut self, expanded: bool) {
        self.update_manual_replacement_advanced(expanded);
    }

    /// Java `action`; deferred button is intentionally only a GUI launch boundary.
    pub fn action<M: CcdEraserXRaysApplicationManager>(
        &mut self,
        manager: &mut M,
        command: &str,
        _deferred: Option<&Deferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
    ) {
        match command {
            "Find X-rays" => manager.find_xrays(self.axis_id, options, self.dialog_type),
            ERASE_LABEL => manager.pre_eraser(self.axis_id, options, self.dialog_type),
            USE_FIXED_STACK_LABEL => manager.replace_raw_stack(self.axis_id, self.dialog_type),
            "Automatic x-ray replacement" => self.enable_xray_replacement(),
            "Manual replacement" => self.enable_manual_replacement(),
            "View X-ray Model" => manager.imod_xray_model(self.axis_id, options),
            "Create Manual Replacement Model" => {
                manager.imod_manual_erase(self.axis_id, options, self.dialog_type)
            }
            "View Fixed Stack" => manager.imod_erased_stack(self.axis_id, options),
            "Show Min/Max for Raw Stack" => {
                manager.clip_stats(self.axis_id, false, self.dialog_type)
            }
            "Show Min/Max for Fixed Stack" => {
                manager.clip_stats(self.axis_id, true, self.dialog_type)
            }
            _ => {}
        }
    }

    #[allow(non_snake_case)]
    /// Native-name adapter for the Java listener's `actionPerformed`.
    pub fn actionPerformed<M: CcdEraserXRaysApplicationManager>(
        &mut self,
        manager: &mut M,
        command: &str,
    ) {
        self.action(manager, command, None, None);
    }
    /// Java `popUpContextMenu`.
    pub fn pop_up_context_menu(&mut self) {
        self.layout.last_context_menu = Some(format!(
            "CCDEraser|ccderaser.html|eraser{}.log|MIN_MAX|FIXED_MIN_MAX",
            self.axis_id.get_extension()
        ));
    }

    /// Java private `enableXRayReplacement`.
    pub fn enable_xray_replacement(&mut self) {
        let on = self.cb_xray_replacement.is_selected();
        self.ltf_peak_criterion.set_enabled(on);
        self.ltf_diff_criterion.set_enabled(on);
        self.ltf_grow_criterion.set_enabled(on);
        self.ltf_edge_exclusion.set_enabled(on);
        self.ltf_maximum_radius.set_enabled(on);
        self.ltf_annulus_width.set_enabled(on);
        self.ltf_scan_region_size.set_enabled(on);
        self.ltf_scan_criterion.set_enabled(on);
        self.ltf_giant_criterion.set_enabled(on);
        self.ltf_big_diff_criterion.set_enabled(on);
        self.ltf_extra_large_radius.set_enabled(on);
        for b in [&mut self.btn_find_xrays, &mut self.btn_view_xray_model] {
            b.set_enabled(on);
        }
    }
    /// Java private `enableManualReplacement`.
    pub fn enable_manual_replacement(&mut self) {
        let on = self.cb_manual_replacement.is_selected();
        for f in [
            &mut self.ltf_global_replacement_list,
            &mut self.ltf_local_replacement_list,
            &mut self.ltf_boundary_replacement_list,
        ] {
            f.set_enabled(on);
        }
        self.btn_create_model.set_enabled(on);
    }
    /// Java private `setToolTipText`; autodoc lookup itself remains an explicit boundary.
    pub fn set_tool_tip_text(&mut self) {
        self.layout.tooltip_initialized = true;
        self.btn_view_xray_model
            .set_tool_tip_text(Some("View the x-ray model on the raw stack in 3dmod."));
        self.btn_create_model
            .set_tool_tip_text(Some("Create a manual replacement model using 3dmod."));
        self.btn_view_erased
            .set_tool_tip_text(Some("View the erased stack in 3dmod."));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Param {
        find: bool,
        values: std::collections::BTreeMap<String, String>,
        model: String,
        adjacent: bool,
    }
    impl ConstCcdEraserParam for Param {
        fn find_peaks(&self) -> bool {
            self.find
        }
        fn value(&self, k: &str) -> String {
            self.values.get(k).cloned().unwrap_or_default()
        }
        fn model_file(&self) -> String {
            self.model.clone()
        }
        fn include_adjacent_points(&self) -> bool {
            self.adjacent
        }
    }
    impl CcdEraserParam for Param {
        fn set_find_peaks(&mut self, v: bool) {
            self.find = v
        }
        fn set_value(&mut self, k: &str, v: String) {
            self.values.insert(k.into(), v);
        }
        fn set_model_file(&mut self, v: String) {
            self.model = v
        }
        fn set_include_adjacent_points(&mut self, v: bool) {
            self.adjacent = v
        }
    }
    #[test]
    fn selection_controls_source_enablement() {
        let mut panel = CcdEraserXRaysPanel::new(AxisID::First, DialogType::PreProcessing, false);
        assert!(!panel.btn_find_xrays.is_enabled());
        panel.cb_xray_replacement.set_selected(true);
        panel.enable_xray_replacement();
        assert!(panel.btn_find_xrays.is_enabled());
        panel.cb_manual_replacement.set_selected(true);
        panel.enable_manual_replacement();
        assert!(panel.btn_create_model.is_enabled());
    }
    #[test]
    fn parameter_round_trip_preserves_manual_model_decision() {
        let mut panel = CcdEraserXRaysPanel::new(AxisID::First, DialogType::PreProcessing, true);
        panel.cb_manual_replacement.set_selected(true);
        panel.manual_replacement_model_file = "dataseta.erase".into();
        panel.ltf_peak_criterion.set_text("2.5");
        let mut param = Param::default();
        assert!(panel.get_parameters(&mut param, true));
        assert_eq!(param.values["peak-criterion"], "2.5");
        assert_eq!(param.model, "dataseta.erase");
    }
    #[test]
    fn advanced_visibility_and_context_match_source() {
        let mut panel = CcdEraserXRaysPanel::new(AxisID::Second, DialogType::PreProcessing, true);
        panel.update_advanced(false);
        assert!(!panel.ltf_grow_criterion.visible);
        panel.pop_up_context_menu();
        assert_eq!(
            panel.layout.last_context_menu.as_deref(),
            Some("CCDEraser|ccderaser.html|eraserb.log|MIN_MAX|FIXED_MIN_MAX")
        );
    }
}
