//! `IMOD/Etomo/src/etomo/ui/swing/MissingWedgeCompensationPanel.java`.
//!
//! The Swing containers, `ButtonGroup`, and action-listener installation stay at
//! the GUI boundary.  The panel's source-owned field, compatibility-checkbox,
//! parameter, validation, and parent-dispatch behaviour is represented here.
#![allow(dead_code)]

use std::{cell::RefCell, rc::Rc};

use super::{
    check_box::CheckBox,
    labeled_text_field::{FieldValidationFailedException, LabeledTextField},
    radio_button::{RadioButton, RadioButtonGroup},
    spinner::Spinner,
};
use crate::imod::etomo::ui::field_type::FieldType;

pub const VOLUME_SIZE_LABEL: &str = "Volume Size";
pub const MISSING_WEDGE_COMPENSATION_LABEL: &str = "Missing Wedge Compensation";
pub const EDGE_SHIFT_DEFAULT: i32 = 1;
pub const EDGE_SHIFT_MIN: i32 = 0;
pub const EDGE_SHIFT_MAX: i32 = 3;
pub const N_WEIGHT_GROUP_DEFAULT: i32 = 8;
pub const N_WEIGHT_GROUP_OFF: i32 = 0;
pub const N_WEIGHT_GROUP_MIN: i32 = 0;
pub const N_WEIGHT_GROUP_MAX: i32 = 32;

/// `MissingWedgeCompensationParent` calls owned by the enclosing PEET panel.
pub trait MissingWedgeCompensationParent {
    fn is_reference_particle_selected(&self) -> bool;
    fn is_volume_table_empty(&self) -> bool;
    fn update_display(&mut self, init: bool);
}

/// `PeetMetaData` fields accessed by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PeetMetaData {
    pub edge_shift: i32,
    pub n_weight_group: i32,
    pub tilt_range: bool,
    pub flg_wedge_weight: bool,
    pub tilt_range_multi_axes: bool,
}
pub type ConstPeetMetaData = PeetMetaData;

/// `MatlabParam` fields and empty-tilt-range state accessed by this unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MatlabParam {
    pub sz_vol_x: String,
    pub sz_vol_y: String,
    pub sz_vol_z: String,
    pub tilt_range_empty: bool,
    pub flg_wedge_weight: bool,
    pub tilt_range_multi_axes: bool,
    pub edge_shift: i32,
    pub n_weight_group: i32,
}

/// Source-visible Swing hierarchy state retained at the native-widget boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MissingWedgeCompensationPanelLayout {
    pub root_components: Vec<String>,
    pub volume_size_components: Vec<String>,
    pub missing_wedge_components: Vec<String>,
    pub enabled_components: Vec<String>,
    pub tilt_range_components: Vec<String>,
    pub april_fools_background: bool,
}

pub struct MissingWedgeCompensationPanel<P: MissingWedgeCompensationParent> {
    pub parent: P,
    pub ltf_volume_size_x: LabeledTextField,
    pub ltf_volume_size_y: LabeledTextField,
    pub ltf_volume_size_z: LabeledTextField,
    pub cb_missing_wedge_compensation: CheckBox,
    pub s_edge_shift: Spinner,
    pub s_n_weight_group: Spinner,
    pub bg_tilt_range: Rc<RefCell<RadioButtonGroup>>,
    pub rb_tilt_range_single: RadioButton,
    pub rb_tilt_range_multi: RadioButton,
    pub l_tilt_range_enabled: bool,
    /// Deprecated Java `cbTiltRange`, retained for old `.prm` files.
    pub cb_tilt_range: CheckBox,
    /// Deprecated Java `cbFlgWedgeWeight`, retained for old `.prm` files.
    pub cb_flg_wedge_weight: CheckBox,
    pub layout: MissingWedgeCompensationPanelLayout,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
}

impl<P: MissingWedgeCompensationParent> MissingWedgeCompensationPanel<P> {
    /// Java private constructor plus `getInstance` construction sequence.
    pub fn get_instance(parent: P, april_fools: bool) -> Self {
        let bg_tilt_range = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut instance = Self {
            parent,
            ltf_volume_size_x: LabeledTextField::new_with_location(
                FieldType::Integer,
                "X: ",
                "the Setup tab",
            ),
            ltf_volume_size_y: LabeledTextField::new_with_location(
                FieldType::Integer,
                "Y: ",
                "the Setup tab",
            ),
            ltf_volume_size_z: LabeledTextField::new_with_location(
                FieldType::Integer,
                "Z: ",
                "the Setup tab",
            ),
            cb_missing_wedge_compensation: CheckBox::new_with_text("Enabled"),
            s_edge_shift: Spinner::get_labeled_instance(
                "Edge shift: ",
                EDGE_SHIFT_DEFAULT,
                EDGE_SHIFT_MIN,
                EDGE_SHIFT_MAX,
                1,
            ),
            s_n_weight_group: Spinner::get_labeled_instance(
                "Weight groups: ",
                N_WEIGHT_GROUP_DEFAULT,
                N_WEIGHT_GROUP_MIN,
                N_WEIGHT_GROUP_MAX,
                1,
            ),
            bg_tilt_range: bg_tilt_range.clone(),
            rb_tilt_range_single: RadioButton::new_in_group("1", bg_tilt_range.clone()),
            rb_tilt_range_multi: RadioButton::new_in_group("2 or more", bg_tilt_range),
            l_tilt_range_enabled: true,
            cb_tilt_range: CheckBox::new_with_text("Use tilt range in averaging"),
            cb_flg_wedge_weight: CheckBox::new_with_text("Use tilt range in alignment"),
            layout: MissingWedgeCompensationPanelLayout {
                april_fools_background: april_fools,
                ..Default::default()
            },
            listener_count: 0,
            tooltip_initialized: false,
        };
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java `addListeners`; dispatch remains a native widget callback boundary.
    pub fn add_listeners(&mut self) {
        self.listener_count = 5;
    }

    /// Java `createPanel` component hierarchy and initial compatibility state.
    pub fn create_panel(&mut self) {
        self.rb_tilt_range_single.set_selected(true);
        self.cb_tilt_range.set_visible(false);
        self.cb_flg_wedge_weight.set_visible(false);
        self.layout.root_components = vec!["volume-size".into(), "missing-wedge".into()];
        self.layout.volume_size_components = vec!["X".into(), "Y".into(), "Z".into()];
        self.layout.missing_wedge_components = vec![
            "enabled".into(),
            "tilt-range".into(),
            "deprecated-tilt-range".into(),
            "deprecated-flg-wedge-weight".into(),
        ];
        self.layout.enabled_components = vec![
            "Enabled".into(),
            "Edge shift".into(),
            "Weight groups".into(),
        ];
        self.layout.tilt_range_components =
            vec!["Number of Tilt Axes".into(), "1".into(), "2 or more".into()];
    }

    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn get_component(&self) -> &Self {
        self
    }

    /// Java `getParameters(PeetMetaData)`.
    pub fn get_peet_parameters(&self, meta_data: &mut PeetMetaData) {
        meta_data.edge_shift = self.s_edge_shift.get_value();
        meta_data.n_weight_group = self.s_n_weight_group.get_value();
        if self.cb_tilt_range.is_visible() {
            meta_data.tilt_range = self.cb_tilt_range.is_selected();
            meta_data.flg_wedge_weight = self.cb_flg_wedge_weight.is_selected();
        } else {
            meta_data.tilt_range = self.cb_missing_wedge_compensation.is_selected();
            meta_data.flg_wedge_weight = self.cb_missing_wedge_compensation.is_selected();
            meta_data.tilt_range_multi_axes = self.rb_tilt_range_multi.is_selected();
        }
    }

    /// Java `setParameters(ConstPeetMetaData)`.
    pub fn set_peet_parameters(&mut self, meta_data: &ConstPeetMetaData) {
        self.cb_tilt_range.set_selected(meta_data.tilt_range);
        self.cb_flg_wedge_weight
            .set_selected(meta_data.flg_wedge_weight);
        self.s_edge_shift.set_value(meta_data.edge_shift);
        self.s_n_weight_group.set_value(meta_data.n_weight_group);
        if meta_data.tilt_range_multi_axes {
            self.rb_tilt_range_multi.set_selected(true);
        } else {
            self.rb_tilt_range_single.set_selected(true);
        }
    }

    /// Java `setParameters(MatlabParam)`.
    pub fn set_matlab_parameters(&mut self, matlab_param: &MatlabParam) {
        self.ltf_volume_size_x.set_text(&matlab_param.sz_vol_x);
        self.ltf_volume_size_y.set_text(&matlab_param.sz_vol_y);
        self.ltf_volume_size_z.set_text(&matlab_param.sz_vol_z);
        if !matlab_param.tilt_range_empty {
            self.cb_tilt_range.set_selected(true);
            self.cb_flg_wedge_weight
                .set_selected(matlab_param.flg_wedge_weight);
        }
        let missing_wedge_compensation =
            self.cb_tilt_range.is_selected() && self.cb_flg_wedge_weight.is_selected();
        self.cb_missing_wedge_compensation
            .set_selected(missing_wedge_compensation);
        if !missing_wedge_compensation {
            if self.cb_tilt_range.is_selected() || self.cb_flg_wedge_weight.is_selected() {
                self.cb_tilt_range.set_visible(true);
                self.cb_flg_wedge_weight.set_visible(true);
            }
        } else if matlab_param.tilt_range_multi_axes {
            self.rb_tilt_range_multi.set_selected(true);
        } else {
            self.rb_tilt_range_single.set_selected(true);
        }
        if missing_wedge_compensation
            || (self.cb_tilt_range.is_visible() && self.cb_tilt_range.is_selected())
        {
            self.s_edge_shift.set_value(matlab_param.edge_shift);
        }
        if missing_wedge_compensation
            || (self.cb_tilt_range.is_visible() && self.cb_flg_wedge_weight.is_selected())
        {
            self.s_n_weight_group.set_value(matlab_param.n_weight_group);
        }
        self.update_display();
    }

    pub fn is_tilt_range_required(&self) -> bool {
        self.cb_missing_wedge_compensation.is_selected()
            || (self.cb_tilt_range.is_visible() && self.cb_tilt_range.is_selected())
    }
    pub fn is_tilt_range_multi_axes(&self) -> bool {
        self.rb_tilt_range_multi.is_selected()
    }

    /// Java `getParameters(MatlabParam, boolean)`; the overload is named for its Rust target.
    pub fn get_matlab_parameters(
        &self,
        matlab_param: &mut MatlabParam,
        do_validation: bool,
    ) -> bool {
        let values: Result<(String, String, String), FieldValidationFailedException> = (|| {
            Ok((
                self.ltf_volume_size_x.get_text_validated(do_validation)?,
                self.ltf_volume_size_y.get_text_validated(do_validation)?,
                self.ltf_volume_size_z.get_text_validated(do_validation)?,
            ))
        })();
        let Ok((x, y, z)) = values else {
            return false;
        };
        matlab_param.sz_vol_x = x;
        matlab_param.sz_vol_y = y;
        matlab_param.sz_vol_z = z;
        if !self.cb_missing_wedge_compensation.is_selected()
            && (!self.cb_tilt_range.is_visible() || !self.cb_tilt_range.is_selected())
        {
            matlab_param.tilt_range_empty = true;
        }
        matlab_param.flg_wedge_weight = self.cb_missing_wedge_compensation.is_selected()
            || (self.cb_tilt_range.is_visible()
                && self.cb_tilt_range.is_selected()
                && self.cb_flg_wedge_weight.is_selected());
        matlab_param.tilt_range_multi_axes = self.rb_tilt_range_multi.is_selected();
        if self.s_edge_shift.is_enabled() {
            matlab_param.edge_shift = self.s_edge_shift.get_value();
        }
        if self.s_n_weight_group.is_enabled() {
            matlab_param.n_weight_group = self.s_n_weight_group.get_value();
        } else {
            matlab_param.n_weight_group = N_WEIGHT_GROUP_OFF;
        }
        true
    }

    /// Java `updateDisplay`.
    pub fn update_display(&mut self) {
        let missing_wedge_compensation = self.cb_missing_wedge_compensation.is_selected();
        self.l_tilt_range_enabled = missing_wedge_compensation;
        self.rb_tilt_range_multi
            .set_enabled(missing_wedge_compensation);
        self.rb_tilt_range_single
            .set_enabled(missing_wedge_compensation);
        if missing_wedge_compensation {
            self.cb_tilt_range.set_selected(true);
            self.cb_flg_wedge_weight.set_selected(true);
        } else if self.cb_tilt_range.is_selected() && self.cb_flg_wedge_weight.is_selected() {
            self.cb_tilt_range.set_selected(false);
            self.cb_flg_wedge_weight.set_selected(false);
        }
        self.cb_flg_wedge_weight
            .set_enabled(self.cb_tilt_range.is_selected());
        self.s_edge_shift.set_enabled(
            missing_wedge_compensation
                || (self.cb_tilt_range.is_visible() && self.cb_tilt_range.is_selected()),
        );
        self.s_n_weight_group.set_enabled(
            missing_wedge_compensation
                || (self.cb_tilt_range.is_visible()
                    && self.cb_tilt_range.is_selected()
                    && self.cb_flg_wedge_weight.is_visible()
                    && self.cb_flg_wedge_weight.is_enabled()
                    && self.cb_flg_wedge_weight.is_selected()
                    && self.parent.is_reference_particle_selected()
                    && !self.parent.is_volume_table_empty()),
        );
    }

    /// Java `updateDisplayBackwardCompatibility(boolean)`.
    pub fn update_display_backward_compatibility(&mut self, init: bool) {
        self.cb_missing_wedge_compensation.set_selected(
            self.cb_tilt_range.is_visible()
                && self.cb_tilt_range.is_selected()
                && self.cb_flg_wedge_weight.is_selected(),
        );
        self.parent.update_display(init);
    }

    /// Java `validateRun`.
    pub fn validate_run(&self) -> Option<String> {
        for field in [
            &self.ltf_volume_size_x,
            &self.ltf_volume_size_y,
            &self.ltf_volume_size_z,
        ] {
            if field.is_empty() {
                return Some(format!(
                    "In {VOLUME_SIZE_LABEL}, {} is required.",
                    field.get_label()
                ));
            }
        }
        None
    }

    /// Java private `action(String)` dispatched by the native listener boundary.
    pub fn action(&mut self, action_command: &str) {
        if action_command
            == self
                .cb_missing_wedge_compensation
                .get_action_command()
                .unwrap_or_default()
            || action_command == self.rb_tilt_range_single.get_action_command()
            || action_command == self.rb_tilt_range_multi.get_action_command()
        {
            self.parent.update_display(false);
        }
        if action_command == self.cb_tilt_range.get_action_command().unwrap_or_default()
            || action_command
                == self
                    .cb_flg_wedge_weight
                    .get_action_command()
                    .unwrap_or_default()
        {
            self.update_display_backward_compatibility(false);
        }
    }

    /// Java `reset`.
    pub fn reset(&mut self) {
        self.ltf_volume_size_x.clear();
        self.ltf_volume_size_y.clear();
        self.ltf_volume_size_z.clear();
        self.cb_missing_wedge_compensation.set_selected(false);
        self.cb_tilt_range.set_selected(false);
        self.s_edge_shift.reset();
        self.cb_flg_wedge_weight.set_selected(false);
        self.s_n_weight_group.reset();
        self.rb_tilt_range_single.set_selected(true);
    }
    /// Java `setDefaults`.
    pub fn set_defaults(&mut self) {
        self.s_edge_shift.set_value(EDGE_SHIFT_DEFAULT);
        self.s_n_weight_group.set_value(N_WEIGHT_GROUP_DEFAULT);
    }
    /// Java `setTooltips`.
    pub fn set_tooltips(&mut self) {
        let tooltip = "The size of the volume around each particle to excise and average.";
        self.ltf_volume_size_x.set_tool_tip_text(Some(tooltip));
        self.ltf_volume_size_y.set_tool_tip_text(Some(tooltip));
        self.ltf_volume_size_z.set_tool_tip_text(Some(tooltip));
        self.cb_tilt_range.set_tool_tip_text(Some("Use the tilt range(s) specified in the volume table for missing wedge compensation during averaging."));
        self.cb_flg_wedge_weight.set_tool_tip_text(Some("Use the tilt range(s) specified in the volume table for missing wedge compensation during alignment."));
        self.cb_missing_wedge_compensation.set_tool_tip_text(Some("Use the tilt range(s) or wedge masks specified in the volume table for missing wedge compensation during alignment and averaging."));
        self.s_n_weight_group.set_tool_tip_text(Some("Number of groups to use for equalizing median cross-correlation coefficient between groups.  Set to 0 or 1 to turn off."));
        self.s_edge_shift.set_tool_tip_text(Some("Number of pixels to shift the edge of the wedge mask to include frequency information just inside the missing wedge."));
        self.rb_tilt_range_single.set_tool_tip_text(Some(
            "Single-axis tilt with range Tilt Range around the tomogram Y axis",
        ));
        self.rb_tilt_range_multi.set_tool_tip_text(Some("Missing / valid data regions for each tomogram specified by a binary missing wedge mask file. Please see the PEET, dualAxisMask, and multiTiltMask man pages for more details."));
        self.tooltip_initialized = true;
    }
}

/// Java private static `MissingWedgeCompensationActionListener`.
pub struct MissingWedgeCompensationActionListener;
impl MissingWedgeCompensationActionListener {
    pub fn action_performed<P: MissingWedgeCompensationParent>(
        panel: &mut MissingWedgeCompensationPanel<P>,
        action_command: &str,
    ) {
        panel.action(action_command);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Parent {
        reference: bool,
        empty: bool,
        updates: Vec<bool>,
    }
    impl MissingWedgeCompensationParent for Parent {
        fn is_reference_particle_selected(&self) -> bool {
            self.reference
        }
        fn is_volume_table_empty(&self) -> bool {
            self.empty
        }
        fn update_display(&mut self, init: bool) {
            self.updates.push(init);
        }
    }
    #[test]
    fn matlab_parameter_loading_keeps_old_checkbox_compatibility() {
        let mut panel = MissingWedgeCompensationPanel::get_instance(
            Parent {
                reference: true,
                ..Default::default()
            },
            false,
        );
        panel.set_matlab_parameters(&MatlabParam {
            sz_vol_x: "20".into(),
            sz_vol_y: "21".into(),
            sz_vol_z: "22".into(),
            tilt_range_empty: false,
            flg_wedge_weight: true,
            tilt_range_multi_axes: true,
            edge_shift: 2,
            n_weight_group: 7,
        });
        assert!(panel.cb_missing_wedge_compensation.is_selected());
        assert!(panel.rb_tilt_range_multi.is_selected());
        assert_eq!(panel.s_edge_shift.get_value(), 2);
        assert!(panel.s_n_weight_group.is_enabled());
    }
    #[test]
    fn validation_and_disabled_weight_group_match_source() {
        let mut panel = MissingWedgeCompensationPanel::get_instance(Parent::default(), false);
        assert_eq!(
            panel.validate_run().as_deref(),
            Some("In Volume Size, X:  is required.")
        );
        panel.ltf_volume_size_x.set_text("1");
        panel.ltf_volume_size_y.set_text("2");
        panel.ltf_volume_size_z.set_text("3");
        panel.update_display();
        let mut parameter = MatlabParam::default();
        assert!(panel.get_matlab_parameters(&mut parameter, true));
        assert_eq!(parameter.n_weight_group, N_WEIGHT_GROUP_OFF);
    }
    #[test]
    fn old_checkbox_action_syncs_new_checkbox_and_parent() {
        let mut panel = MissingWedgeCompensationPanel::get_instance(Parent::default(), false);
        panel.cb_tilt_range.set_visible(true);
        panel.cb_flg_wedge_weight.set_visible(true);
        panel.cb_tilt_range.set_selected(true);
        panel.cb_flg_wedge_weight.set_selected(true);
        let action = panel.cb_tilt_range.get_action_command().unwrap().to_owned();
        MissingWedgeCompensationActionListener::action_performed(&mut panel, &action);
        assert!(panel.cb_missing_wedge_compensation.is_selected());
        assert_eq!(panel.parent.updates, vec![false]);
    }
}
