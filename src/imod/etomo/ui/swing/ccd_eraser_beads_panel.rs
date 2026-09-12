//! `IMOD/Etomo/src/etomo/ui/swing/CcdEraserBeadsPanel.java`.
//!
//! Swing construction, autodoc lookup, and concrete `ApplicationManager` calls
//! stay at direct boundaries.  The Rust panel retains its source control order,
//! diameter conversion, parameter routing, button state, and focus update.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::{BaseScreenState, MultiLineButton};
use super::radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup};
use super::tilt_panel::Deferred3dmodButton;

pub const CCD_ERASER_LABEL: &str = "Erase Beads";
pub const USE_ERASED_STACK_LABEL: &str = "Use Erased Stack";
pub const FIDUCIAL_DIAMETER_LABEL: &str = "Diameter to erase";

/// Java `CheckBoxSpinner` at the source dependency boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckBoxSpinner {
    pub label: String,
    pub value: i32,
    pub minimum: i32,
    pub maximum: i32,
    pub selected: bool,
    pub tooltip: Option<String>,
}
impl CheckBoxSpinner {
    pub fn get_instance(label: &str, value: i32, minimum: i32, maximum: i32) -> Self {
        Self {
            label: label.into(),
            value,
            minimum,
            maximum,
            selected: false,
            tooltip: None,
        }
    }
}

/// Java nested `PolynomialOrder` enum singleton values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PolynomialOrder {
    #[default]
    UseMean,
    FitAPlane,
    FillWithNoise,
}
impl PolynomialOrder {
    pub const fn get_instance(value: i32) -> Self {
        match value {
            1 => Self::FitAPlane,
            -1 => Self::FillWithNoise,
            _ => Self::UseMean,
        }
    }
    pub fn get_instance_string(value: &str) -> Self {
        value.parse().map(Self::get_instance).unwrap_or_default()
    }
    pub const fn get_value(self) -> i32 {
        match self {
            Self::UseMean => 0,
            Self::FitAPlane => 1,
            Self::FillWithNoise => -1,
        }
    }
    pub const fn is_default(self) -> bool {
        matches!(self, Self::UseMean)
    }
}
impl std::fmt::Display for PolynomialOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.get_value().fmt(f)
    }
}

/// Calls on Java `CCDEraserParam` made by this source unit.
pub trait CcdEraserParam {
    fn set_input_file(&mut self, value: String);
    fn set_model_file(&mut self, value: String);
    fn set_output_file(&mut self, value: String);
    fn set_better_radius(&mut self, value: f64);
    fn set_expand_circle_iterations(&mut self, value: i32);
    fn reset_expand_circle_iterations(&mut self);
    fn set_polynomial_order(&mut self, value: String);
    fn validate(&self) -> bool;
    fn is_better_radius_set(&self) -> bool;
    fn better_radius(&self) -> f64;
    fn is_expand_circle_iterations_set(&self) -> bool;
    fn expand_circle_iterations(&self) -> i32;
    fn polynomial_order(&self) -> i32;
}
/// Calls on Java `MakecomfileParam` made by this source unit.
pub trait MakecomfileParam {
    fn set_bead_size(&mut self, value: String);
}
/// Java `ConstMetaData` reads used by this source unit.
pub trait CcdEraserBeadsMetaData {
    fn ctf3d_setup_slab_thickness_in_nm_set(&self) -> bool;
    fn final_stack_fiducial_diameter(&self, axis_id: AxisID) -> Option<String>;
    fn final_stack_better_radius(&self, axis_id: AxisID) -> Option<f64>;
    fn fiducial_diameter(&self) -> f64;
    fn pixel_size(&self) -> f64;
    fn final_stack_polynomial_order(&self, axis_id: AxisID) -> i32;
    fn use_final_stack_expand_circle_iterations(&self, axis_id: AxisID) -> bool;
    fn final_stack_expand_circle_iterations(&self, axis_id: AxisID) -> Option<i32>;
}
/// Java `MetaData` writes used by this source unit.
pub trait CcdEraserBeadsMetaDataMut {
    fn set_final_stack_fiducial_diameter(&mut self, axis_id: AxisID, value: String);
    fn set_final_stack_expand_circle_iterations(&mut self, axis_id: AxisID, value: i32);
    fn set_use_final_stack_expand_circle_iterations(&mut self, axis_id: AxisID, value: bool);
    fn set_final_stack_polynomial_order(&mut self, axis_id: AxisID, value: String);
}
/// Concrete Java manager/FileType calls made by this source unit.
pub trait CcdEraserBeadsApplicationManager {
    fn calc_binned_bead_diameter_pixels(&self, axis_id: AxisID) -> String;
    fn aligned_stack_exists(&self, axis_id: AxisID) -> bool;
    fn aligned_stack_binning(&self, axis_id: AxisID) -> Result<i32, String>;
    fn aligned_stack_file_name(&self, axis_id: AxisID) -> String;
    fn ccd_eraser_beads_input_model_file_name(&self, axis_id: AxisID) -> String;
    fn gold_eraser(
        &mut self,
        axis_id: AxisID,
        dialog_type: DialogType,
        options: Option<Run3dmodMenuOptions>,
    );
    fn imod_erased_fiducials(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn use_ccd_eraser(&mut self, axis_id: AxisID, dialog_type: DialogType, label: &str);
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CcdEraserBeadsPanelLayout {
    pub root_border: Option<String>,
    pub root_component_order: Vec<String>,
    pub parameter_component_order: Vec<String>,
    pub polynomial_order_component_order: Vec<String>,
    pub button_component_order: Vec<String>,
    pub ctf_visible: bool,
    pub listeners_added: bool,
    pub tooltips_set: bool,
}

/// Java final `CcdEraserBeadsPanel`.
pub struct CcdEraserBeadsPanel {
    pub pnl_root: CcdEraserBeadsPanelLayout,
    pub ltf_fiducial_diameter: LabeledTextField,
    pub l_fiducial_diameter_1: String,
    pub l_fiducial_diameter_2: String,
    pub cbsp_expand_circle_iterations: CheckBoxSpinner,
    pub bg_polynomial_order: Rc<RefCell<RadioButtonGroup>>,
    pub rb_polynomial_order_use_mean: RadioButton,
    pub rb_polynomial_order_fill_noise: RadioButton,
    pub rb_polynomial_order_fit_a_plane: RadioButton,
    pub btn_ccd_eraser: MultiLineButton,
    pub btn_3dmod_ccd_eraser: MultiLineButton,
    pub btn_use_ccd_eraser: MultiLineButton,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub aligned_stack_binning: Option<i32>,
}

impl CcdEraserBeadsPanel {
    pub fn new(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut make_radio = |label: &str, order: PolynomialOrder, default: bool| {
            RadioButton::new_with_enumerated_type(
                Some(label.into()),
                EnumeratedTypeBoundary {
                    label: order.to_string(),
                    default,
                    value: Some(order.to_string()),
                },
                Some(group.clone()),
            )
        };
        let result = Self {
            pnl_root: CcdEraserBeadsPanelLayout::default(),
            ltf_fiducial_diameter: LabeledTextField::new(
                FieldType::FloatingPoint,
                &format!("{FIDUCIAL_DIAMETER_LABEL} (binned pixels): "),
            ),
            l_fiducial_diameter_1: FIDUCIAL_DIAMETER_LABEL.into(),
            l_fiducial_diameter_2: FIDUCIAL_DIAMETER_LABEL.into(),
            cbsp_expand_circle_iterations: CheckBoxSpinner::get_instance(
                "Iterations to grow circular areas:",
                2,
                1,
                5,
            ),
            bg_polynomial_order: group.clone(),
            rb_polynomial_order_use_mean: make_radio(
                "Use mean of surrounding points",
                PolynomialOrder::UseMean,
                true,
            ),
            rb_polynomial_order_fill_noise: make_radio(
                "Fill pixels with noise values",
                PolynomialOrder::FillWithNoise,
                false,
            ),
            rb_polynomial_order_fit_a_plane: make_radio(
                "Fit a plane to surrounding points",
                PolynomialOrder::FitAPlane,
                false,
            ),
            btn_ccd_eraser: MultiLineButton::new_with_label(Some(CCD_ERASER_LABEL)),
            btn_3dmod_ccd_eraser: MultiLineButton::new_with_label(Some("View Erased Stack")),
            btn_use_ccd_eraser: MultiLineButton::new_with_label(Some(USE_ERASED_STACK_LABEL)),
            axis_id,
            dialog_type,
            aligned_stack_binning: None,
        };
        result
    }
    pub fn get_instance<M: CcdEraserBeadsApplicationManager>(
        manager: &M,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Self {
        let mut instance = Self::new(axis_id, dialog_type);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance.update_aligned_stack_binning(manager);
        instance
    }
    pub fn add_listeners(&mut self) {
        self.btn_ccd_eraser.add_action_listener();
        self.btn_3dmod_ccd_eraser.add_action_listener();
        self.btn_use_ccd_eraser.add_action_listener();
        self.ltf_fiducial_diameter.add_focus_listener();
        self.pnl_root.listeners_added = true;
    }
    pub fn initialize<M: CcdEraserBeadsApplicationManager>(&mut self, manager: &M) {
        if self.ltf_fiducial_diameter.is_empty() {
            self.ltf_fiducial_diameter
                .set_text(&manager.calc_binned_bead_diameter_pixels(self.axis_id));
        }
    }
    pub fn create_panel(&mut self) {
        self.pnl_root.root_border = Some(CCD_ERASER_LABEL.into());
        self.pnl_root.root_component_order = vec![
            "ccdEraserParameterPanel".into(),
            "verticalStrut(3)".into(),
            "pnlCtf3d".into(),
            "verticalStrut(5)".into(),
            "ccdEraserButtonPanel".into(),
        ];
        self.pnl_root.parameter_component_order = vec![
            "fiducialDiameterPanel".into(),
            "polynomialOrderPanel".into(),
        ];
        self.pnl_root.polynomial_order_component_order =
            vec!["useMean".into(), "fillNoise".into(), "fitAPlane".into()];
        self.pnl_root.button_component_order = vec![
            "btnCcdEraser".into(),
            "btn3dmodCcdEraser".into(),
            "btnUseCcdEraser".into(),
        ];
    }
    pub fn get_component(&self) -> &CcdEraserBeadsPanelLayout {
        &self.pnl_root
    }
    pub fn get_polynomial_order(&self) -> String {
        if self.rb_polynomial_order_fill_noise.is_selected() {
            PolynomialOrder::FillWithNoise.to_string()
        } else if self.rb_polynomial_order_fit_a_plane.is_selected() {
            PolynomialOrder::FitAPlane.to_string()
        } else {
            PolynomialOrder::UseMean.to_string()
        }
    }
    pub fn set_polynomial_order(&mut self, order: PolynomialOrder) {
        match order {
            PolynomialOrder::UseMean => self.rb_polynomial_order_use_mean.set_selected(true),
            PolynomialOrder::FillWithNoise => {
                self.rb_polynomial_order_fill_noise.set_selected(true)
            }
            PolynomialOrder::FitAPlane => self.rb_polynomial_order_fit_a_plane.set_selected(true),
        }
    }
    pub fn get_parameters<M: CcdEraserBeadsMetaDataMut>(&self, data: &mut M) {
        data.set_final_stack_fiducial_diameter(self.axis_id, self.ltf_fiducial_diameter.get_text());
        data.set_final_stack_expand_circle_iterations(
            self.axis_id,
            self.cbsp_expand_circle_iterations.value,
        );
        data.set_use_final_stack_expand_circle_iterations(
            self.axis_id,
            self.cbsp_expand_circle_iterations.selected,
        );
        data.set_final_stack_polynomial_order(self.axis_id, self.get_polynomial_order());
    }
    pub fn set_parameters<M: CcdEraserBeadsMetaData>(&mut self, data: &M) {
        self.pnl_root.ctf_visible = data.ctf3d_setup_slab_thickness_in_nm_set();
        if let Some(value) = data.final_stack_fiducial_diameter(self.axis_id) {
            self.ltf_fiducial_diameter.set_text(&value);
        } else if let Some(radius) = data.final_stack_better_radius(self.axis_id) {
            self.ltf_fiducial_diameter
                .set_text_number((radius * 2.0 * 10.0).round() / 10.0);
        } else {
            self.ltf_fiducial_diameter.set_text_number(
                (data.fiducial_diameter() / data.pixel_size() * 10.0).round() / 10.0,
            );
        }
        self.set_polynomial_order(PolynomialOrder::get_instance(
            data.final_stack_polynomial_order(self.axis_id),
        ));
        self.cbsp_expand_circle_iterations.selected =
            data.use_final_stack_expand_circle_iterations(self.axis_id);
        if let Some(value) = data.final_stack_expand_circle_iterations(self.axis_id) {
            self.cbsp_expand_circle_iterations.value = value;
        }
        self.focus_lost();
    }
    pub fn get_parameters_ccd_eraser<M: CcdEraserBeadsApplicationManager, P: CcdEraserParam>(
        &self,
        manager: &M,
        param: &mut P,
        do_validation: bool,
    ) -> bool {
        let Ok(text) = self.ltf_fiducial_diameter.get_text_validated(do_validation) else {
            return false;
        };
        let Ok(diameter) = text.parse::<f64>() else {
            return false;
        };
        param.set_input_file(manager.aligned_stack_file_name(self.axis_id));
        param.set_model_file(manager.ccd_eraser_beads_input_model_file_name(self.axis_id));
        param.set_output_file("erased_beads_stack".into());
        param.set_better_radius(diameter / 2.0);
        if self.cbsp_expand_circle_iterations.selected {
            param.set_expand_circle_iterations(self.cbsp_expand_circle_iterations.value);
        } else {
            param.reset_expand_circle_iterations();
        }
        param.set_polynomial_order(self.get_polynomial_order());
        param.validate()
    }
    pub fn set_parameters_ccd_eraser<P: CcdEraserParam>(&mut self, param: &P) {
        if param.is_better_radius_set() {
            self.ltf_fiducial_diameter
                .set_text_number(param.better_radius() * 2.0);
        }
        self.cbsp_expand_circle_iterations.selected = param.is_expand_circle_iterations_set();
        if self.cbsp_expand_circle_iterations.selected {
            self.cbsp_expand_circle_iterations.value = param.expand_circle_iterations();
        }
        self.set_polynomial_order(PolynomialOrder::get_instance(param.polynomial_order()));
        self.focus_lost();
    }
    pub fn get_parameters_makecomfile<P: MakecomfileParam>(
        &self,
        param: &mut P,
        do_validation: bool,
    ) -> bool {
        match self.ltf_fiducial_diameter.get_text_validated(do_validation) {
            Ok(value) => {
                param.set_bead_size(value);
                true
            }
            Err(_) => false,
        }
    }
    pub fn update_aligned_stack_binning<M: CcdEraserBeadsApplicationManager>(
        &mut self,
        manager: &M,
    ) {
        self.aligned_stack_binning = if manager.aligned_stack_exists(self.axis_id) {
            match manager.aligned_stack_binning(self.axis_id) {
                Ok(value) => Some(value),
                Err(_) => return,
            }
        } else {
            None
        };
        self.focus_lost();
    }
    pub fn focus_gained(&mut self) {}
    pub fn focus_lost(&mut self) {
        if let Some(binning) = self.aligned_stack_binning {
            self.l_fiducial_diameter_1.clear();
            self.l_fiducial_diameter_2.clear();
            let value = self.ltf_fiducial_diameter.get_text();
            if let Ok(diameter) = value.parse::<f64>() {
                let unbinned = (diameter * f64::from(binning) * 10.0).round() / 10.0;
                self.l_fiducial_diameter_1 =
                    format!("(corresponds to unbinned diameter of {unbinned}");
                self.l_fiducial_diameter_2 = format!("with current binning of {binning})");
            }
        } else {
            self.l_fiducial_diameter_1 = "Enter diameter in unbinned pixels".into();
            self.l_fiducial_diameter_2.clear();
        }
    }
    pub fn action<M: CcdEraserBeadsApplicationManager>(
        &mut self,
        manager: &mut M,
        command: &str,
        _deferred: Option<Deferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
    ) {
        if self.btn_ccd_eraser.button.action_command.as_deref() == Some(command) {
            manager.gold_eraser(self.axis_id, self.dialog_type, options);
        } else if self.btn_3dmod_ccd_eraser.button.action_command.as_deref() == Some(command) {
            manager.imod_erased_fiducials(self.axis_id, options);
        } else if self.btn_use_ccd_eraser.button.action_command.as_deref() == Some(command) {
            manager.use_ccd_eraser(self.axis_id, self.dialog_type, CCD_ERASER_LABEL);
        }
    }
    pub fn done(&mut self) {
        self.btn_ccd_eraser.button.action_listener_count = self
            .btn_ccd_eraser
            .button
            .action_listener_count
            .saturating_sub(1);
        self.btn_use_ccd_eraser.button.action_listener_count = self
            .btn_use_ccd_eraser
            .button
            .action_listener_count
            .saturating_sub(1);
    }
    pub fn set_parameters_screen_state(&mut self, screen_state: &BaseScreenState) {
        let ccd_key = self.btn_ccd_eraser.get_button_state_key();
        self.btn_ccd_eraser
            .set_button_state(screen_state.get_button_state(ccd_key.as_deref()));
        let use_key = self.btn_use_ccd_eraser.get_button_state_key();
        self.btn_use_ccd_eraser
            .set_button_state(screen_state.get_button_state(use_key.as_deref()));
    }
    pub fn set_tool_tip_text(&mut self) {
        self.ltf_fiducial_diameter.set_tool_tip_text(Some(
            "The diameter, in pixels of the aligned stack, that will be erased around each point.",
        ));
        self.cbsp_expand_circle_iterations.tooltip = Some("Expand circle iterations".into());
        self.rb_polynomial_order_use_mean.set_tool_tip_text(Some(
            "Fill the erased pixels with the mean of surrounding pixels.",
        ));
        self.rb_polynomial_order_fill_noise.set_tool_tip_text(Some(
            "Fill the erased pixels with noise matching the SD of surrounding pixels.",
        ));
        self.rb_polynomial_order_fit_a_plane.set_tool_tip_text(Some(
            "Fill the erased pixels with a gradient based on plane fit to surrounding pixels.",
        ));
        self.btn_ccd_eraser.set_tool_tip_text(Some(
            "Run Ccderaser on the aligned stack to erase around model points.",
        ));
        self.btn_3dmod_ccd_eraser.set_tool_tip_text(Some("View the results of running Ccderaser on the aligned stack along with the _erase.fid model."));
        self.btn_use_ccd_eraser.set_tool_tip_text(Some(
            "Replace the full aligned stack (.ali) with the erased stack (_erase.ali).",
        ));
        self.pnl_root.tooltips_set = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn focus_lost_converts_binned_diameter() {
        let mut panel = CcdEraserBeadsPanel::new(AxisID::Only, DialogType::SetupRecon);
        panel.ltf_fiducial_diameter.set_text("6.25");
        panel.aligned_stack_binning = Some(2);
        panel.focus_lost();
        assert_eq!(
            panel.l_fiducial_diameter_1,
            "(corresponds to unbinned diameter of 12.5"
        );
        assert_eq!(panel.l_fiducial_diameter_2, "with current binning of 2)");
    }
    #[test]
    fn polynomial_order_defaults_as_java() {
        assert_eq!(PolynomialOrder::get_instance(4), PolynomialOrder::UseMean);
        assert_eq!(
            PolynomialOrder::get_instance_string("-1"),
            PolynomialOrder::FillWithNoise
        );
    }
}
