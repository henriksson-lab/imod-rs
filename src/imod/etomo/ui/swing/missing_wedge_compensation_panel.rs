//! `IMOD/Etomo/src/etomo/ui/swing/MissingWedgeCompensationPanel.java`.
//!
//! The PEET dialog's "Volume Size" and "Missing Wedge Compensation" boxes.  An event
//! dispatch thread object, created as `Rc<Self>` by
//! [`MissingWedgeCompensationPanel::get_instance`]; it keeps a weak reference to its
//! parent.

use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::etched_border::EtchedBorder;
use super::labeled_text_field::LabeledTextField;
use super::missing_wedge_compensation_parent::MissingWedgeCompensationParent;
use super::peet_dialog;
use super::radio_button::RadioButton;
use super::spaced_panel::{self, SpacedPanel};
use super::spinner::Spinner;
use super::swing_component::SwingComponent;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::storage::matlab_param::{self, MatlabParam};
use crate::imod::etomo::r#type::const_etomo_number::Number;
use crate::imod::etomo::r#type::const_peet_meta_data::ConstPeetMetaData;
use crate::imod::etomo::r#type::peet_meta_data::PeetMetaData;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java private static final `VOLUME_SIZE_LABEL`.
const VOLUME_SIZE_LABEL: &str = "Volume Size";
/// Java private static final `MISSING_WEDGE_COMPENSATION_LABEL`.
const MISSING_WEDGE_COMPENSATION_LABEL: &str = "Missing Wedge Compensation";

/// Java package-private `final class MissingWedgeCompensationPanel implements
/// UIComponent, SwingComponent`.
pub struct MissingWedgeCompensationPanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `ltfVolumeSizeX`.
    ltf_volume_size_x: Rc<LabeledTextField>,
    /// Java private final `ltfVolumeSizeY`.
    ltf_volume_size_y: Rc<LabeledTextField>,
    /// Java private final `ltfVolumeSizeZ`.
    ltf_volume_size_z: Rc<LabeledTextField>,
    /// Java private final `cbMissingWedgeCompensation`.
    cb_missing_wedge_compensation: Rc<CheckBox>,
    /// Java private final `sEdgeShift`.
    s_edge_shift: Rc<Spinner>,
    /// Java private final `sNWeightGroup`.
    s_n_weight_group: Rc<Spinner>,
    /// Java private `bgTiltRange`.
    bg_tilt_range: Rc<ButtonGroup>,
    /// Java private `rbTiltRangeSingle`.
    rb_tilt_range_single: Rc<RadioButton>,
    /// Java private `rbTiltRangeMulti`.
    rb_tilt_range_multi: Rc<RadioButton>,
    /// Java private `lTiltRange`.
    l_tilt_range: Rc<JComponent>,
    /// Java private final `cbTiltRange` (deprecated: replaced by
    /// cbMissingWedgeCompensation).
    cb_tilt_range: Rc<CheckBox>,
    /// Java private final `cbFlgWedgeWeight` (deprecated: replaced by
    /// cbMissingWedgeCompensation).
    cb_flg_wedge_weight: Rc<CheckBox>,
    /// Java private final `parent`.
    parent: Weak<dyn MissingWedgeCompensationParent>,
    /// Java private final `fieldDisplayer`.
    field_displayer: Option<Rc<dyn FieldDisplayer>>,
    /// Java `this`.
    self_ref: Weak<MissingWedgeCompensationPanel>,
}

impl MissingWedgeCompensationPanel {
    /// Java private `MissingWedgeCompensationPanel(MissingWedgeCompensationParent,
    /// FieldDisplayer)`, with the field initializers.
    fn new(
        parent: Weak<dyn MissingWedgeCompensationParent>,
        field_displayer: Option<Rc<dyn FieldDisplayer>>,
    ) -> Rc<MissingWedgeCompensationPanel> {
        let bg_tilt_range = ButtonGroup::new();
        Rc::new_cyclic(|self_ref: &Weak<MissingWedgeCompensationPanel>| {
            MissingWedgeCompensationPanel {
                pnl_root: SpacedPanel::get_instance_void(),
                ltf_volume_size_x: LabeledTextField::new_field_type_string_string(
                    FieldType::Integer,
                    Some("X: "),
                    Some(peet_dialog::SETUP_LOCATION_DESCR),
                ),
                ltf_volume_size_y: LabeledTextField::new_field_type_string_string(
                    FieldType::Integer,
                    Some("Y: "),
                    Some(peet_dialog::SETUP_LOCATION_DESCR),
                ),
                ltf_volume_size_z: LabeledTextField::new_field_type_string_string(
                    FieldType::Integer,
                    Some("Z: "),
                    Some(peet_dialog::SETUP_LOCATION_DESCR),
                ),
                cb_missing_wedge_compensation: CheckBox::new_string(Some("Enabled")),
                s_edge_shift: Spinner::get_labeled_instance_string_int_int_int(
                    Some(&format!("{}: ", shared_strings::EDGE_SHIFT_LABEL)),
                    matlab_param::EDGE_SHIFT_DEFAULT,
                    matlab_param::EDGE_SHIFT_MIN,
                    matlab_param::EDGE_SHIFT_MAX,
                ),
                s_n_weight_group: Spinner::get_labeled_instance_string_int_int_int(
                    Some(&format!("{}: ", shared_strings::N_WEIGHT_GROUP_LABEL)),
                    matlab_param::N_WEIGHT_GROUP_DEFAULT,
                    matlab_param::N_WEIGHT_GROUP_MIN,
                    matlab_param::N_WEIGHT_GROUP_MAX,
                ),
                rb_tilt_range_single: RadioButton::new_string_button_group(
                    Some("1"),
                    Some(&bg_tilt_range),
                ),
                rb_tilt_range_multi: RadioButton::new_string_button_group(
                    Some("2 or more"),
                    Some(&bg_tilt_range),
                ),
                bg_tilt_range,
                l_tilt_range: JComponent::new_label("Number of Tilt Axes: "),
                cb_tilt_range: CheckBox::new_string(Some("Use tilt range in averaging")),
                cb_flg_wedge_weight: CheckBox::new_string(Some("Use tilt range in alignment")),
                parent,
                field_displayer,
                self_ref: self_ref.clone(),
            }
        })
    }

    /// Java static `getInstance(MissingWedgeCompensationParent, FieldDisplayer)`.
    pub fn get_instance(
        parent: Weak<dyn MissingWedgeCompensationParent>,
        field_displayer: Option<Rc<dyn FieldDisplayer>>,
    ) -> Rc<MissingWedgeCompensationPanel> {
        let instance = MissingWedgeCompensationPanel::new(parent, field_displayer);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java field read `parent`.
    fn parent(&self) -> Rc<dyn MissingWedgeCompensationParent> {
        self.parent
            .upgrade()
            .expect("the PEET dialog owns its missing wedge compensation panel")
    }

    /// Java private `addListeners()` with `MissingWedgeCompensationActionListener`.
    fn add_listeners(&self) {
        let adaptee = self.self_ref.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(panel) = adaptee.upgrade() {
                panel.action(event.get_action_command().unwrap_or(""));
            }
        });
        self.cb_missing_wedge_compensation
            .add_action_listener(Some(action_listener.clone()));
        self.rb_tilt_range_single
            .add_action_listener(action_listener.clone());
        self.rb_tilt_range_multi
            .add_action_listener(action_listener.clone());
        self.cb_tilt_range
            .add_action_listener(Some(action_listener.clone()));
        self.cb_flg_wedge_weight
            .add_action_listener(Some(action_listener));
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        self.rb_tilt_range_single.set_selected_boolean(true);
        self.cb_tilt_range.set_visible(false);
        self.cb_flg_wedge_weight.set_visible(false);
        self.ltf_volume_size_x
            .set_overridable_field_displayers(None, self.field_displayer.clone());
        self.ltf_volume_size_y
            .set_overridable_field_displayers(None, self.field_displayer.clone());
        self.ltf_volume_size_z
            .set_overridable_field_displayers(None, self.field_displayer.clone());
        // local panels
        let pnl_volume_size = SpacedPanel::get_instance_void();
        let pnl_missing_wedge_compensation = SpacedPanel::get_instance_void();
        let pnl_enabled = SpacedPanel::get_instance_void();
        let pnl_tilt_range = SpacedPanel::get_instance_void();
        // Root panel
        self.pnl_root.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_root.set_component_alignment_x(0.0);
        self.pnl_root.add_spaced_panel(&pnl_volume_size);
        self.pnl_root
            .add_spaced_panel(&pnl_missing_wedge_compensation);
        // pnlVolumeSize (rigid areas x5_y0 / x40_y0 between)
        pnl_volume_size.set_box_layout(spaced_panel::X_AXIS);
        pnl_volume_size.set_border(
            &EtchedBorder::new(Some(&format!("{VOLUME_SIZE_LABEL} (Voxels)"))).get_border(),
        );
        pnl_volume_size.add_container(&self.ltf_volume_size_x.get_container());
        pnl_volume_size.add_container(&self.ltf_volume_size_y.get_container());
        pnl_volume_size.add_container(&self.ltf_volume_size_z.get_container());
        // pnlMissingWedgeCompensation
        pnl_missing_wedge_compensation.set_box_layout(spaced_panel::Y_AXIS);
        pnl_missing_wedge_compensation
            .set_border(&EtchedBorder::new(Some(MISSING_WEDGE_COMPENSATION_LABEL)).get_border());
        pnl_missing_wedge_compensation.set_component_alignment_x(1.0);
        pnl_missing_wedge_compensation.add_spaced_panel(&pnl_enabled);
        pnl_missing_wedge_compensation.add_spaced_panel(&pnl_tilt_range);
        pnl_missing_wedge_compensation.add_check_box(&self.cb_tilt_range);
        pnl_missing_wedge_compensation.add_check_box(&self.cb_flg_wedge_weight);
        // enabled
        pnl_enabled.set_box_layout(spaced_panel::X_AXIS);
        pnl_enabled.set_component_alignment_x(0.0);
        pnl_enabled.add_check_box(&self.cb_missing_wedge_compensation);
        pnl_enabled.add_rigid_area_dimension((3, 0));
        pnl_enabled.add_container(&self.s_edge_shift.get_container());
        pnl_enabled.add_rigid_area_dimension((3, 0));
        pnl_enabled.add_container(&self.s_n_weight_group.get_container());
        // TiltRange
        pnl_tilt_range.set_box_layout(spaced_panel::X_AXIS);
        pnl_tilt_range.set_component_alignment_x(1.0);
        pnl_tilt_range.add_j_label(&self.l_tilt_range);
        pnl_tilt_range.add_radio_button(&self.rb_tilt_range_single);
        pnl_tilt_range.add_rigid_area_dimension((15, 0));
        pnl_tilt_range.add_radio_button(&self.rb_tilt_range_multi);
        pnl_tilt_range.add_rigid_area_dimension((45, 0));
    }

    /// Java package-private `getParameters(PeetMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &PeetMetaData) {
        meta_data.set_edge_shift(Some(self.s_edge_shift.get_value()));
        meta_data.set_n_weight_group(Some(self.s_n_weight_group.get_value()));
        if self.cb_tilt_range.is_visible() {
            meta_data.set_tilt_range(self.cb_tilt_range.is_selected());
            meta_data.set_flg_wedge_weight(self.cb_flg_wedge_weight.is_selected());
        } else {
            meta_data.set_tilt_range(self.cb_missing_wedge_compensation.is_selected());
            meta_data.set_flg_wedge_weight(self.cb_missing_wedge_compensation.is_selected());
            meta_data.set_tilt_range_multi_axes(self.rb_tilt_range_multi.is_selected());
        }
    }

    /// Java package-private `setParameters(ConstPeetMetaData)`.
    pub fn set_parameters_meta_data(&self, meta_data: &dyn ConstPeetMetaData) {
        self.cb_tilt_range
            .set_selected_boolean(meta_data.is_tilt_range());
        self.cb_flg_wedge_weight
            .set_selected_boolean(meta_data.is_flg_wedge_weight());
        self.s_edge_shift
            .set_value_const_etomo_number(&meta_data.get_edge_shift());
        self.s_n_weight_group
            .set_value_const_etomo_number(&meta_data.get_n_weight_group());
        if meta_data.is_tilt_range_multi_axes() {
            self.rb_tilt_range_multi.set_selected_boolean(true);
        } else {
            self.rb_tilt_range_single.set_selected_boolean(true);
        }
    }

    /// Java package-private `setParameters(MatlabParam)`.  Load data from
    /// MatlabParamFile.
    pub fn set_parameters_matlab_param(&self, matlab_param: &MatlabParam) {
        self.ltf_volume_size_x
            .set_text_string(matlab_param.get_sz_vol_x().as_deref());
        self.ltf_volume_size_y
            .set_text_string(matlab_param.get_sz_vol_y().as_deref());
        self.ltf_volume_size_z
            .set_text_string(matlab_param.get_sz_vol_z().as_deref());
        // the new checkbox is selected when when tiltRange and flgWedgeWeight are both
        // selected. Expose the tiltRange and flgWedgeWeight checkboxes if one of them is
        // selected.
        if !matlab_param.is_tilt_range_empty() {
            self.cb_tilt_range.set_selected_boolean(true);
            self.cb_flg_wedge_weight
                .set_selected_boolean(matlab_param.is_flg_wedge_weight());
        }
        let missing_wedge_compensation =
            self.cb_tilt_range.is_selected() && self.cb_flg_wedge_weight.is_selected();
        self.cb_missing_wedge_compensation
            .set_selected_boolean(missing_wedge_compensation);
        if !missing_wedge_compensation {
            if self.cb_tilt_range.is_selected() || self.cb_flg_wedge_weight.is_selected() {
                self.cb_tilt_range.set_visible(true);
                self.cb_flg_wedge_weight.set_visible(true);
            }
        } else if matlab_param.is_tilt_range_multi_axes() {
            self.rb_tilt_range_multi.set_selected_boolean(true);
        } else {
            self.rb_tilt_range_single.set_selected_boolean(true);
        }
        if missing_wedge_compensation
            || (self.cb_tilt_range.is_visible() && self.cb_tilt_range.is_selected())
        {
            self.s_edge_shift
                .set_value_parsed_element(Some(matlab_param.get_edge_shift()));
        }
        if missing_wedge_compensation
            || (self.cb_tilt_range.is_visible() && self.cb_flg_wedge_weight.is_selected())
        {
            self.s_n_weight_group
                .set_value_parsed_element(Some(matlab_param.get_n_weight_group()));
        }
        self.update_display();
    }

    /// Java package-private `isTiltRangeRequired()`.
    pub fn is_tilt_range_required(&self) -> bool {
        self.cb_missing_wedge_compensation.is_selected()
            || (self.cb_tilt_range.is_visible() && self.cb_tilt_range.is_selected())
    }

    /// Java package-private `isTiltRangeMultiAxes()`.
    pub fn is_tilt_range_multi_axes(&self) -> bool {
        self.rb_tilt_range_multi.is_selected()
    }

    /// Java package-private `getParameters(MatlabParam, boolean)`.
    pub fn get_parameters_matlab_param(
        &self,
        matlab_param: &mut MatlabParam,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let Ok(x) = self.ltf_volume_size_x.get_text_boolean(do_validation) else {
            return false;
        };
        matlab_param.set_sz_vol_x(x.as_deref());
        let Ok(y) = self.ltf_volume_size_y.get_text_boolean(do_validation) else {
            return false;
        };
        matlab_param.set_sz_vol_y(y.as_deref());
        let Ok(z) = self.ltf_volume_size_z.get_text_boolean(do_validation) else {
            return false;
        };
        matlab_param.set_sz_vol_z(z.as_deref());
        // If cbTiltRange is off, this overrides what was set in the volumeTable.
        if !self.cb_missing_wedge_compensation.is_selected()
            && (!self.cb_tilt_range.is_visible() || !self.cb_tilt_range.is_selected())
        {
            matlab_param.set_tilt_range_empty();
        }
        matlab_param.set_flg_wedge_weight(
            self.cb_missing_wedge_compensation.is_selected()
                || (self.cb_tilt_range.is_visible()
                    && self.cb_tilt_range.is_selected()
                    && self.cb_flg_wedge_weight.is_selected()),
        );
        matlab_param.set_tilt_range_multi_axes(self.rb_tilt_range_multi.is_selected());
        if self.s_edge_shift.is_enabled() {
            matlab_param.set_edge_shift(Some(self.s_edge_shift.get_value()));
        }
        if self.s_n_weight_group.is_enabled() {
            matlab_param.set_n_weight_group(Some(self.s_n_weight_group.get_value()));
        } else {
            matlab_param
                .set_n_weight_group(Some(Number::Integer(matlab_param::N_WEIGHT_GROUP_OFF)));
        }
        true
    }

    /// Java package-private `updateDisplay()`.
    pub fn update_display(&self) {
        let missing_wedge_compensation = self.cb_missing_wedge_compensation.is_selected();
        self.l_tilt_range.set_enabled(missing_wedge_compensation);
        self.rb_tilt_range_multi
            .set_enabled(missing_wedge_compensation);
        self.rb_tilt_range_single
            .set_enabled(missing_wedge_compensation);
        // Get the new checkbox and the old backward compatibility checkboxes to sync to
        // each other.
        if missing_wedge_compensation {
            self.cb_tilt_range.set_selected_boolean(true);
            self.cb_flg_wedge_weight.set_selected_boolean(true);
        } else if self.cb_tilt_range.is_selected() && self.cb_flg_wedge_weight.is_selected() {
            self.cb_tilt_range.set_selected_boolean(false);
            self.cb_flg_wedge_weight.set_selected_boolean(false);
        }
        self.cb_flg_wedge_weight
            .set_enabled(self.cb_tilt_range.is_selected());
        self.s_edge_shift.set_enabled(
            missing_wedge_compensation
                || (self.cb_tilt_range.is_visible() && self.cb_tilt_range.is_selected()),
        );
        let parent = self.parent();
        self.s_n_weight_group.set_enabled(
            missing_wedge_compensation
                || (self.cb_tilt_range.is_visible()
                    && self.cb_tilt_range.is_selected()
                    && self.cb_flg_wedge_weight.is_visible()
                    && self.cb_flg_wedge_weight.is_enabled()
                    && self.cb_flg_wedge_weight.is_selected()
                    && parent.is_reference_particle_selected()
                    && !parent.is_volume_table_empty()),
        );
    }

    /// Java package-private `updateDisplayBackwardCompatibility(boolean)`.  Called when
    /// the old checkboxes are changed.
    pub fn update_display_backward_compatibility(&self, init: bool) {
        self.cb_missing_wedge_compensation.set_selected_boolean(
            self.cb_tilt_range.is_visible()
                && self.cb_tilt_range.is_selected()
                && self.cb_flg_wedge_weight.is_selected(),
        );
        self.parent().update_display(init);
    }

    /// Java package-private `validateRun()`.
    pub fn validate_run(&self) -> Option<String> {
        // particle volume
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

    /// Java private `action(String)`.
    fn action(&self, action_command: &str) {
        let action_command = Some(action_command);
        if action_command
            == self
                .cb_missing_wedge_compensation
                .get_action_command()
                .as_deref()
            || action_command == self.rb_tilt_range_single.get_action_command().as_deref()
            || action_command == self.rb_tilt_range_multi.get_action_command().as_deref()
        {
            self.parent().update_display(false);
        }
        if action_command == self.cb_tilt_range.get_action_command().as_deref()
            || action_command == self.cb_flg_wedge_weight.get_action_command().as_deref()
        {
            self.update_display_backward_compatibility(false);
        }
    }

    /// Java package-private `reset()`.  Reset values and set defaults.
    pub fn reset(&self) {
        self.ltf_volume_size_x.clear();
        self.ltf_volume_size_y.clear();
        self.ltf_volume_size_z.clear();
        self.cb_missing_wedge_compensation
            .set_selected_boolean(false);
        self.cb_tilt_range.set_selected_boolean(false);
        self.s_edge_shift.reset();
        self.cb_flg_wedge_weight.set_selected_boolean(false);
        self.s_n_weight_group.reset();
        self.rb_tilt_range_single.set_selected_boolean(true);
    }

    /// Java package-private `setDefaults()`.
    pub fn set_defaults(&self) {
        self.s_edge_shift
            .set_value_int(matlab_param::EDGE_SHIFT_DEFAULT);
        self.s_n_weight_group
            .set_value_int(matlab_param::N_WEIGHT_GROUP_DEFAULT);
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        let tooltip = "The size of the volume around each particle to excise and average.";
        self.ltf_volume_size_x.set_tool_tip_text(Some(tooltip));
        self.ltf_volume_size_y.set_tool_tip_text(Some(tooltip));
        self.ltf_volume_size_z.set_tool_tip_text(Some(tooltip));
        self.cb_tilt_range.set_tool_tip_text_string(Some(
            "Use the tilt range(s) specified in the volume table for missing wedge compensation during averaging.",
        ));
        self.cb_flg_wedge_weight.set_tool_tip_text_string(Some(
            "Use the tilt range(s) specified in the volume table for missing wedge compensation during alignment.",
        ));
        self.cb_missing_wedge_compensation.set_tool_tip_text_string(Some(
            "Use the tilt range(s) or wedge masks specified in the volume table for missing wedge compensation during alignment and averaging.",
        ));
        self.s_n_weight_group.set_tool_tip_text(Some(
            "Number of groups to use for equalizing median cross-correlation coefficient between groups.  Set to 0 or 1 to turn off.",
        ));
        self.s_edge_shift.set_tool_tip_text(Some(
            "Number of pixels to shift the edge of the wedge mask to include frequency information just inside the missing wedge.",
        ));
        self.rb_tilt_range_single.set_tool_tip_text_string(Some(
            "Single-axis tilt with range Tilt Range around the tomogram Y axis",
        ));
        self.rb_tilt_range_multi.set_tool_tip_text_string(Some(
            "Missing / valid data regions for each tomogram specified by a binary missing wedge mask file. Please see the PEET, dualAxisMask, and multiTiltMask man pages for more details.",
        ));
    }
}

impl SwingComponent for MissingWedgeCompensationPanel {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }
}

impl UIComponent for MissingWedgeCompensationPanel {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }
}
