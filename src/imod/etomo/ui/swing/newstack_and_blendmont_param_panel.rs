//! `IMOD/Etomo/src/etomo/ui/swing/NewstackAndBlendmontParamPanel.java`.
//!
//! Swing painting and the application/com-script/autodoc implementations are
//! explicit boundaries.  The panel retains the controls, their construction
//! order, listeners, and the exact parameter transfers owned by the Java unit.
#![allow(dead_code)]

use super::check_box::CheckBox;
use super::fiducialess_params::FiducialessParams;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::newstack_or_blendmont_panel::{BlendmontParam, MetaData, NewstParam};
use super::spaced_panel::{SpacedPanel, Y_AXIS};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field_type::FieldType;

pub const SIZE_TO_OUTPUT_IN_X_AND_Y_LABEL: &str = "Size to output";
pub const BINNING_LABEL: &str = "Aligned image stack binning";

/// Direct `AutodocFactory` lookup made by Java `setToolTipText`.
pub trait NewstackAndBlendmontParamPanelAutodoc {
    fn newstack_size_to_output_tooltip(&self, axis_id: AxisID) -> Option<String>;
}

/// GUI-boundary state for the two CTF notices and their insertion order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ctf3dLabel {
    pub text: &'static str,
    pub complete_foreground: bool,
    pub visible: bool,
}

/// Java final `NewstackAndBlendmontParamPanel`.
#[derive(Clone, Debug)]
pub struct NewstackAndBlendmontParamPanel {
    pub pnl_root: SpacedPanel,
    pub spin_binning: LabeledSpinner,
    pub ltf_size_to_output_in_x_and_y: LabeledTextField,
    pub ltf_rotation: LabeledTextField,
    pub cb_fiducialess: CheckBox,
    pub cb_use_linear_interpolation: CheckBox,
    pub l_ctf3d1: Ctf3dLabel,
    pub l_ctf3d2: Ctf3dLabel,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    /// The sole state exposed by Java `updateAdvanced`.
    pub advanced: bool,
    /// Absent precisely for montage data, as in the Java constructor.
    pub cb_antialias_filter: Option<CheckBox>,
    pub antialias_filter_value: Option<f64>,
    pub action_listener_present: bool,
    pub binning_change_listener_present: bool,
}

impl PartialEq for NewstackAndBlendmontParamPanel {
    fn eq(&self, other: &Self) -> bool {
        self.axis_id == other.axis_id
            && self.dialog_type == other.dialog_type
            && self.advanced == other.advanced
            && self.spin_binning.get_value() == other.spin_binning.get_value()
            && self.ltf_size_to_output_in_x_and_y.get_text()
                == other.ltf_size_to_output_in_x_and_y.get_text()
            && self.ltf_rotation.get_text() == other.ltf_rotation.get_text()
            && self.cb_use_linear_interpolation.is_selected()
                == other.cb_use_linear_interpolation.is_selected()
            && self.cb_antialias_filter.as_ref().map(CheckBox::is_selected)
                == other
                    .cb_antialias_filter
                    .as_ref()
                    .map(CheckBox::is_selected)
            && self.antialias_filter_value == other.antialias_filter_value
    }
}

impl NewstackAndBlendmontParamPanel {
    /// Java private constructor; manager metadata is reduced to its exact view
    /// type read at construction.
    pub fn new(axis_id: AxisID, dialog_type: DialogType, view_type: ViewType) -> Self {
        Self {
            pnl_root: SpacedPanel::get_instance_y_axis_padding(true),
            spin_binning: LabeledSpinner::get_instance(&format!("{BINNING_LABEL}: "), 1, 1, 8, 1),
            ltf_size_to_output_in_x_and_y: LabeledTextField::new(
                FieldType::IntegerPair,
                &format!("{SIZE_TO_OUTPUT_IN_X_AND_Y_LABEL} (X,Y - unbinned): "),
            ),
            ltf_rotation: LabeledTextField::new(FieldType::FloatingPoint, "Tilt axis rotation: "),
            cb_fiducialess: CheckBox::new_with_text("Coarse alignment only"),
            cb_use_linear_interpolation: CheckBox::new_with_text("Use linear interpolation"),
            l_ctf3d1: Ctf3dLabel {
                text: "No need to make stack if doing 3D CTF and using raw images - ",
                complete_foreground: true,
                visible: false,
            },
            l_ctf3d2: Ctf3dLabel {
                text: "unless you want to check gold erasing with an aligned stack.",
                complete_foreground: true,
                visible: false,
            },
            axis_id,
            dialog_type,
            advanced: false,
            cb_antialias_filter: (view_type != ViewType::Montage)
                .then(|| CheckBox::new_with_text("Reduce size with antialiasing filter")),
            antialias_filter_value: None,
            action_listener_present: false,
            binning_change_listener_present: false,
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance<A: NewstackAndBlendmontParamPanelAutodoc>(
        autodoc: &A,
        axis_id: AxisID,
        dialog_type: DialogType,
        view_type: ViewType,
    ) -> Self {
        let mut instance = Self::new(axis_id, dialog_type, view_type);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text(autodoc);
        instance
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.cb_fiducialess.add_action_listener();
        self.spin_binning.add_change_listener();
        self.action_listener_present = true;
        self.binning_change_listener_present = true;
    }

    /// Java `getComponent`.
    pub fn get_component(&self) -> &super::spaced_panel::JPanel {
        self.pnl_root.get_container()
    }

    /// Java private `createPanel`.  Widget insertion itself stays at the
    /// native Swing boundary; all source layout and ordering decisions remain.
    pub fn create_panel(&mut self) {
        self.pnl_root.set_box_layout(Y_AXIS);
        self.pnl_root.set_component_alignment_x(0.0);
        self.update_fiducialess();
    }

    /// Java `setVisible`.
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }

    /// Java `setParameters(BlendmontParam)`.
    pub fn set_blendmont_parameters(&mut self, blendmont_param: &BlendmontParam) {
        self.cb_use_linear_interpolation
            .set_selected(blendmont_param.linear_interpolation);
    }

    /// Java `setParameters(ConstNewstParam)`.
    pub fn set_newst_parameters(&mut self, newst_param: &NewstParam) {
        self.cb_use_linear_interpolation
            .set_selected(newst_param.linear_interpolation);
        if let Some(check_box) = &mut self.cb_antialias_filter {
            let antialias_filter = newst_param.antialias_filter.is_some();
            check_box.set_selected(antialias_filter);
            if antialias_filter {
                self.antialias_filter_value = newst_param.antialias_filter;
            }
        }
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_meta_data_parameters(&self, meta_data: &mut MetaData) {
        let index = self.axis_id.get_axis_of_extension() as usize;
        meta_data.size_to_output_in_x_and_y[index] = self.ltf_size_to_output_in_x_and_y.get_text();
        meta_data.stack_binning[index] = self.get_binning();
        meta_data.antialias_filter[index] = self.antialias_filter_value;
    }

    /// Java `getParameters(BlendmontParam, boolean)`.  Com-script conversion
    /// is represented by the fields transferred into its direct boundary.
    pub fn get_blendmont_parameters(
        &self,
        blendmont_param: &mut BlendmontParam,
        do_validation: bool,
    ) -> bool {
        let Ok(size) = self
            .ltf_size_to_output_in_x_and_y
            .get_text_validated(do_validation)
        else {
            return false;
        };
        blendmont_param.bin_by_factor = Some(self.get_binning());
        blendmont_param.linear_interpolation = self.cb_use_linear_interpolation.is_selected();
        blendmont_param.size_to_output_in_x_and_y = Some(size);
        blendmont_param.fiducialess = self.cb_fiducialess.is_selected();
        true
    }

    /// Java `getParameters(NewstParam, boolean)`.
    pub fn get_newst_parameters(&self, newst_param: &mut NewstParam, do_validation: bool) -> bool {
        let Ok(size) = self
            .ltf_size_to_output_in_x_and_y
            .get_text_validated(do_validation)
        else {
            return false;
        };
        let binning = self.get_binning();
        newst_param.bin_by_factor = (binning > 1).then_some(binning);
        if let Some(check_box) = &self.cb_antialias_filter {
            newst_param.antialias_filter = check_box
                .is_selected()
                .then_some(self.antialias_filter_value.unwrap_or_default());
        }
        newst_param.linear_interpolation = self.cb_use_linear_interpolation.is_selected();
        newst_param.size_to_output_in_x_and_y = Some(size);
        true
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_meta_data_parameters(&mut self, meta_data: &MetaData) {
        let index = self.axis_id.get_axis_of_extension() as usize;
        self.l_ctf3d1.visible = meta_data.ctf3d_setup_slab_thickness_in_nm_set;
        self.l_ctf3d2.visible = meta_data.ctf3d_setup_slab_thickness_in_nm_set;
        self.spin_binning
            .set_value_int(meta_data.stack_binning[index]);
        if meta_data.antialias_filter[index].is_some() {
            self.antialias_filter_value = meta_data.antialias_filter[index];
        }
        self.ltf_size_to_output_in_x_and_y
            .set_text(&meta_data.size_to_output_in_x_and_y[index]);
        self.update_fiducialess();
        self.update_enabled();
    }

    /// Java `setFiducialessAlignment`.
    pub fn set_fiducialess_alignment(&mut self, input: bool) {
        self.cb_fiducialess.set_selected(input);
        self.update_fiducialess();
    }

    /// Java `setImageRotation`.
    pub fn set_image_rotation(&mut self, input: impl Into<String>) {
        self.ltf_rotation.set_text(&input.into());
    }

    /// Java `setBinning(ConstEtomoNumber)` at its number-value boundary.
    pub fn set_binning(&mut self, binning: i32) {
        self.spin_binning.set_value_int(binning);
        self.update_enabled();
    }

    /// Java private `getBinning`.
    pub fn get_binning(&self) -> i32 {
        self.spin_binning.get_value()
    }

    /// Java private `updateEnabled`.
    pub fn update_enabled(&mut self) {
        if let Some(check_box) = &mut self.cb_antialias_filter {
            check_box.set_enabled(self.spin_binning.get_value() > 1);
        }
    }

    /// Java interface `isFiducialess`.
    pub fn is_fiducialess(&self) -> bool {
        self.cb_fiducialess.is_selected()
    }

    /// Java private `updateFiducialess`.
    pub fn update_fiducialess(&mut self) {
        self.ltf_rotation
            .set_enabled(self.cb_fiducialess.is_selected());
    }

    /// Java `updateAdvanced`.
    pub fn update_advanced(&mut self, advanced: bool) {
        self.advanced = advanced;
        self.ltf_size_to_output_in_x_and_y.set_visible(advanced);
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action(&mut self, command: &str) {
        if self.cb_fiducialess.get_action_command() == Some(command) {
            self.update_fiducialess();
        }
    }

    /// Native callback endpoint for Java
    /// `NewstackAndBlendmontParamPanelActionListener.actionPerformed`.
    #[allow(non_snake_case)]
    pub fn actionPerformed(&mut self, command: &str) {
        self.action(command);
    }

    /// Native callback endpoint for Java
    /// `NewstackAndBlendmontBinningChangeListener.stateChanged`.
    #[allow(non_snake_case)]
    pub fn stateChanged(&mut self) {
        self.update_enabled();
    }

    /// Java private `setToolTipText` with autodoc I/O retained as a direct boundary.
    pub fn set_tool_tip_text<A: NewstackAndBlendmontParamPanelAutodoc>(&mut self, autodoc: &A) {
        self.ltf_size_to_output_in_x_and_y.set_tool_tip_text(
            autodoc
                .newstack_size_to_output_tooltip(self.axis_id)
                .as_deref(),
        );
        self.cb_use_linear_interpolation.set_tool_tip_text(Some(
            "Make aligned stack with linear instead of cubic interpolation to  reduce noise.",
        ));
        self.spin_binning.set_tool_tip_text(Some("Set the binning for the aligned image stack and tomogram.  With a binned tomogram, all of the thickness, position, and size parameters in Tomogram Generation are still entered in unbinned pixels."));
        self.cb_fiducialess
            .set_tool_tip_text(Some("Use cross-correlation alignment only."));
        self.ltf_rotation.set_tool_tip_text(Some("Rotation angle of tilt axis for generating aligned stack from cross-correlation alignment only."));
        if let Some(check_box) = &mut self.cb_antialias_filter {
            check_box.set_tool_tip_text(Some("Use antialiased image reduction instead binning with the default filter in Newstack; useful for data from direct detection cameras."));
        }
    }
}

impl FiducialessParams for NewstackAndBlendmontParamPanel {
    fn is_fiducialess(&self) -> bool {
        Self::is_fiducialess(self)
    }

    fn get_image_rotation(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_rotation.get_text_validated(do_validation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Autodoc;
    impl NewstackAndBlendmontParamPanelAutodoc for Autodoc {
        fn newstack_size_to_output_tooltip(&self, _: AxisID) -> Option<String> {
            Some("size tooltip".into())
        }
    }

    #[test]
    fn construction_and_listeners_follow_java_branches() {
        let panel = NewstackAndBlendmontParamPanel::get_instance(
            &Autodoc,
            AxisID::First,
            DialogType::FinalAlignedStack,
            ViewType::SingleView,
        );
        assert!(panel.cb_antialias_filter.is_some());
        assert!(panel.action_listener_present && panel.binning_change_listener_present);
        assert!(!panel.ltf_rotation.enabled);
        let montage = NewstackAndBlendmontParamPanel::new(
            AxisID::Only,
            DialogType::FinalAlignedStack,
            ViewType::Montage,
        );
        assert!(montage.cb_antialias_filter.is_none());
    }

    #[test]
    fn parameter_and_control_transfers_preserve_source_rules() {
        let mut panel = NewstackAndBlendmontParamPanel::new(
            AxisID::First,
            DialogType::FinalAlignedStack,
            ViewType::SingleView,
        );
        panel.spin_binning.set_value_int(2);
        panel.stateChanged();
        panel.cb_use_linear_interpolation.set_selected(true);
        panel.cb_fiducialess.set_selected(true);
        let command = panel
            .cb_fiducialess
            .get_action_command()
            .unwrap()
            .to_owned();
        panel.actionPerformed(&command);
        panel.ltf_size_to_output_in_x_and_y.set_text("100,200");
        let mut newst = NewstParam::default();
        assert!(panel.get_newst_parameters(&mut newst, true));
        assert_eq!(newst.bin_by_factor, Some(2));
        assert!(newst.linear_interpolation);
        assert!(panel.ltf_rotation.enabled);
        assert!(
            panel
                .cb_antialias_filter
                .as_ref()
                .is_some_and(|value| value.check_box.enabled)
        );
    }

    #[test]
    fn implements_canonical_fiducialess_params_without_duplicate_state() {
        let mut panel = NewstackAndBlendmontParamPanel::new(
            AxisID::First,
            DialogType::FinalAlignedStack,
            ViewType::SingleView,
        );
        panel.set_fiducialess_alignment(true);
        panel.set_image_rotation("-3.5");
        let params: &dyn FiducialessParams = &panel;
        assert!(params.is_fiducialess());
        assert_eq!(params.get_image_rotation(true).unwrap(), "-3.5");
    }
}
