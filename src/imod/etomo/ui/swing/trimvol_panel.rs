//! `IMOD/Etomo/src/etomo/ui/swing/TrimvolPanel.java`.
//!
//! Java `final class TrimvolPanel implements Run3dmodButtonContainer,
//! RubberbandContainer, TrimvolDisplay, ContextMenu`: the "Trim vol" tab of
//! the Post Processing dialog - volume range, byte scaling, reorientation, and
//! the Trim Volume / 3dmod buttons.
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by [`TrimvolPanel::new`];
//! every method takes `&self`.  The listener classes `ScalingListener` and
//! `ButtonListener` are closures holding a weak reference to the panel.
//!
//! The panel's `setParameters`/`getParameters` overloads carry the
//! parameter-type suffix (`ui.md` naming rule).

use std::rc::{Rc, Weak};

use super::beveled_border::BeveledBorder;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::process_control_panel;
use super::radio_button::RadioButton;
use super::rubberband_container::RubberbandContainer;
use super::rubberband_panel::RubberbandPanel;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::{self, SpacedPanel};
use super::trimvol_display::TrimvolDisplay;
use super::ui_harness;
use super::volume_range_panel::VolumeRangePanel;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::trimvol_param::TrimvolParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent, MouseEvent};
use crate::imod::etomo::logic::trimvol_input_file_state::TrimvolInputFileState;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java public static final `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static final `SCALING_ERROR_TITLE`.
const SCALING_ERROR_TITLE: &str = "Scaling Panel Error";
/// Java private static final `FIXED_SCALE_MIN_LABEL`.
const FIXED_SCALE_MIN_LABEL: &str = "black: ";
/// Java private static final `FIXED_SCALE_MAX_LABEL`.
const FIXED_SCALE_MAX_LABEL: &str = " white: ";
/// Java private static final `SECTION_SCALE_MIN_LABEL`.
const SECTION_SCALE_MIN_LABEL: &str = "Z min: ";
/// Java private static final `SECTION_SCALE_MAX_LABEL`.
const SECTION_SCALE_MAX_LABEL: &str = " Z max: ";
/// Java static final `SWAP_YZ_LABEL`.
pub const SWAP_YZ_LABEL: &str = "Swap Y and Z dimensions";
/// Java static final `REORIENTATION_GROUP_LABEL`.
pub const REORIENTATION_GROUP_LABEL: &str = "Reorientation:";

/// Java `final class TrimvolPanel`.
pub struct TrimvolPanel {
    /// Java private `applicationManager`.
    application_manager: &'static ApplicationManager,

    /// Java private `pnlTrimvol = new EtomoPanel()`.
    pnl_trimvol: Rc<EtomoPanel>,

    /// Java private `pnlScale = SpacedPanel.getInstance()`.
    pnl_scale: Rc<SpacedPanel>,
    /// Java private `pnlScaleFixed = new JPanel()`.
    pnl_scale_fixed: Rc<JComponent>,
    /// Java private `cbConvertToBytes`.
    cb_convert_to_bytes: Rc<CheckBox>,
    /// Java private `rbScaleFixed`.
    rb_scale_fixed: Rc<RadioButton>,
    /// Java private `ltfFixedScaleMin`.
    ltf_fixed_scale_min: Rc<LabeledTextField>,
    /// Java private `ltfFixedScaleMax`.
    ltf_fixed_scale_max: Rc<LabeledTextField>,

    /// Java private `rbScaleSection`.
    rb_scale_section: Rc<RadioButton>,
    /// Java private `pnlScaleSection = new JPanel()`.
    pnl_scale_section: Rc<JComponent>,
    /// Java private `ltfSectionScaleMin`.
    ltf_section_scale_min: Rc<LabeledTextField>,
    /// Java private `ltfSectionScaleMax`.
    ltf_section_scale_max: Rc<LabeledTextField>,
    /// Rust-only: the constructor's local `ButtonGroup bgScale`.  Swing's
    /// button models keep the group alive; the stand-in buttons hold it
    /// weakly, so the panel keeps it.
    bg_scale: Rc<ButtonGroup>,

    /// Java private final `pnlReorientationChoices = new EtomoPanel()`.
    pnl_reorientation_choices: Rc<EtomoPanel>,
    /// Java private final `bgReorientation = new ButtonGroup()`.
    bg_reorientation: Rc<ButtonGroup>,
    /// Java private final `rbNone`.
    rb_none: Rc<RadioButton>,
    /// Java private final `rbSwapYZ`.
    rb_swap_yz: Rc<RadioButton>,
    /// Java private final `rbRotateX`.
    rb_rotate_x: Rc<RadioButton>,
    /// Java private final `lWarning1`.
    l_warning1: Rc<JComponent>,
    /// Java private final `lWarning2`.
    l_warning2: Rc<JComponent>,
    /// Java private final `lWarning3`.
    l_warning3: Rc<JComponent>,
    /// Java private final `lWarning4`.
    l_warning4: Rc<JComponent>,
    /// Java private final `lWarning5`.
    l_warning5: Rc<JComponent>,
    /// Java private final `warning = new JLabel()`.
    warning: Rc<JComponent>,

    /// Java private `pnlButton = new JPanel()`.
    pnl_button: Rc<JComponent>,
    /// Java private `btnImodFull`.
    btn_imod_full: Rc<Run3dmodButton>,
    /// Java private final `btnTrimvol`.
    btn_trimvol: Rc<Run3dmodButton>,
    /// Java private `btnImodTrim`.
    btn_imod_trim: Rc<Run3dmodButton>,
    /// Java private `btnGetCoordinates`.
    btn_get_coordinates: Rc<MultiLineButton>,
    /// Java private `pnlImodFull = new JPanel()`.
    pnl_imod_full: Rc<JComponent>,
    /// Java private final `volumeRangePanel`.
    volume_range_panel: Rc<VolumeRangePanel>,

    /// Java private final `buttonActonListener` (sic).
    button_acton_listener: ActionListener,
    /// Java private final `pnlScaleRubberband`.
    pnl_scale_rubberband: Rc<RubberbandPanel>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `trimvolInputFileMissing`.
    trimvol_input_file_missing: bool,
}

impl TrimvolPanel {
    /// Java package-private constructor `TrimvolPanel(ApplicationManager,
    /// AxisID, DialogType, boolean)`.  Default constructor.
    pub fn new(
        app_mgr: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        trimvol_input_file_missing: bool,
    ) -> Rc<TrimvolPanel> {
        let instance = Rc::new_cyclic(|self_ref: &Weak<TrimvolPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = self_ref.clone();
            // Field initializers.
            let pnl_trimvol = EtomoPanel::new();
            let pnl_scale = SpacedPanel::get_instance_void();
            let pnl_scale_fixed = JComponent::new_panel();
            let cb_convert_to_bytes = CheckBox::new_string(Some("Convert to bytes"));
            let rb_scale_fixed = RadioButton::new_string(Some("Scale to match contrast  "));
            let ltf_fixed_scale_min = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(FIXED_SCALE_MIN_LABEL),
            );
            let ltf_fixed_scale_max = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(FIXED_SCALE_MAX_LABEL),
            );
            let rb_scale_section = RadioButton::new_string(Some("Find scaling from sections  "));
            let pnl_scale_section = JComponent::new_panel();
            let ltf_section_scale_min = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(SECTION_SCALE_MIN_LABEL),
            );
            let ltf_section_scale_max = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(SECTION_SCALE_MAX_LABEL),
            );
            let pnl_reorientation_choices = EtomoPanel::new();
            let bg_reorientation = ButtonGroup::new();
            let rb_none = RadioButton::new_string(Some("None"));
            let rb_swap_yz = RadioButton::new_string(Some(SWAP_YZ_LABEL));
            let rb_rotate_x = RadioButton::new_string(Some("Rotate around X axis"));
            let l_warning1 = JComponent::new_label("Warning:");
            let l_warning2 = JComponent::new_label("For serial joins, use");
            let l_warning3 = JComponent::new_label("the same reorientation");
            let l_warning4 = JComponent::new_label("method for each");
            let l_warning5 = JComponent::new_label("section.");
            let warning = JComponent::new_label("");
            let pnl_button = JComponent::new_panel();
            let btn_imod_full =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("3dmod Full Volume"),
                    Some(container.clone()),
                );
            let btn_imod_trim =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("3dmod Trimmed Volume"),
                    Some(container.clone()),
                );
            let btn_get_coordinates =
                MultiLineButton::new_string(Some("Get XYZ Volume Range From 3dmod"));
            let pnl_imod_full = JComponent::new_panel();

            // Constructor body.
            let volume_range_panel = VolumeRangePanel::get_instance(trimvol_input_file_missing);
            // panels
            let rubberband_container: Weak<dyn RubberbandContainer> = self_ref.clone();
            let pnl_scale_rubberband = RubberbandPanel::get_no_button_instance(
                app_mgr,
                Some(rubberband_container),
                Some(imod_manager::COMBINED_TOMOGRAM_KEY),
                Some("Scaling from sub-area:"),
                Some("Get XYZ Sub-Area From 3dmod"),
                Some("Minimum X coordinate on the left side to analyze for contrast range."),
                Some("Maximum X coordinate on the right side to analyze for contrast range."),
                Some("The lower Y coordinate to analyze for contrast range."),
                Some("The upper Y coordinate to analyze for contrast range."),
                trimvol_input_file_missing,
            );
            let btn_trimvol = app_mgr
                .get_process_result_display_factory(AxisID::Only)
                .get_trim_volume();
            btn_trimvol.set_container(Some(container.clone()));
            btn_trimvol.set_deferred_3dmod_button_deferred_3dmod_button(Some(
                btn_imod_trim.clone() as Rc<dyn Deferred3dmodButton>,
            ));

            // Java `buttonActonListener = new ButtonListener(this)`.
            let listenee = self_ref.clone();
            let button_acton_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(listenee) = listenee.upgrade() {
                    listenee.action(event.get_action_command().unwrap_or(""), None, None);
                }
            });

            TrimvolPanel {
                application_manager: app_mgr,
                pnl_trimvol,
                pnl_scale,
                pnl_scale_fixed,
                cb_convert_to_bytes,
                rb_scale_fixed,
                ltf_fixed_scale_min,
                ltf_fixed_scale_max,
                rb_scale_section,
                pnl_scale_section,
                ltf_section_scale_min,
                ltf_section_scale_max,
                bg_scale: ButtonGroup::new(),
                pnl_reorientation_choices,
                bg_reorientation,
                rb_none,
                rb_swap_yz,
                rb_rotate_x,
                l_warning1,
                l_warning2,
                l_warning3,
                l_warning4,
                l_warning5,
                warning,
                pnl_button,
                btn_imod_full,
                btn_trimvol,
                btn_imod_trim,
                btn_get_coordinates,
                pnl_imod_full,
                volume_range_panel,
                button_acton_listener,
                pnl_scale_rubberband,
                axis_id,
                dialog_type,
                trimvol_input_file_missing,
            }
        });
        // The rest of the Java constructor body, which needs the finished
        // object (`this`).
        // init
        instance
            .btn_trimvol
            .set_enabled(!instance.trimvol_input_file_missing);

        // Layout the scale panel
        // Swing layout: pnlScaleFixed.setLayout(new BoxLayout(pnlScaleFixed,
        // BoxLayout.X_AXIS)).

        instance
            .pnl_scale_fixed
            .add(&instance.rb_scale_fixed.get_component());
        instance
            .pnl_scale_fixed
            .add(&instance.ltf_fixed_scale_min.get_container());
        instance
            .pnl_scale_fixed
            .add(&instance.ltf_fixed_scale_max.get_container());

        // Swing layout: pnlScaleSection X_AXIS BoxLayout.
        instance
            .pnl_scale_section
            .add(&instance.rb_scale_section.get_component());
        // Swing layout: ltfSectionScaleMin/Max.setTextPreferredWidth(
        // UIParameters.getInstance().getFourDigitWidth()).
        instance
            .pnl_scale_section
            .add(&instance.ltf_section_scale_min.get_container());
        instance
            .pnl_scale_section
            .add(&instance.ltf_section_scale_max.get_container());

        // ButtonGroup bgScale = new ButtonGroup();
        instance
            .bg_scale
            .add(&instance.rb_scale_fixed.get_abstract_button());
        instance
            .bg_scale
            .add(&instance.rb_scale_section.get_abstract_button());

        instance.pnl_scale.set_box_layout(spaced_panel::Y_AXIS);
        instance
            .pnl_scale
            .set_border(&EtchedBorder::new(Some("Scaling")).get_border());

        // Swing layout: cbConvertToBytes.setAlignmentX(Component.RIGHT_ALIGNMENT).
        let pnl_convert_to_bytes = JComponent::new_panel();
        // Swing layout: pnlConvertToBytes X_AXIS BoxLayout, CENTER_ALIGNMENT.
        pnl_convert_to_bytes.add(&instance.cb_convert_to_bytes.get_component());
        // Swing layout: pnlConvertToBytes.add(Box.createHorizontalGlue()).
        instance.pnl_scale.add_j_panel(&pnl_convert_to_bytes);
        instance.pnl_scale.add_j_panel(&instance.pnl_scale_fixed);
        instance.pnl_scale.add_j_panel(&instance.pnl_scale_section);
        instance
            .pnl_scale
            .add_component(&instance.pnl_scale_rubberband.get_component());
        instance.pnl_scale.add_component(
            &instance
                .pnl_scale_rubberband
                .get_rubberband_button_component(),
        );

        // Swing layout: pnlButton X_AXIS BoxLayout, horizontal glue around and
        // between the buttons.
        instance
            .pnl_button
            .add(&instance.btn_trimvol.get_component());
        instance
            .pnl_button
            .add(&instance.btn_imod_trim.get_component());

        // Swing layout: pnlTrimvol.setLayout(new BoxLayout(pnlTrimvol,
        // BoxLayout.Y_AXIS)).
        instance
            .pnl_trimvol
            .set_border(&BeveledBorder::new(Some("Volume Trimming")).get_border());
        instance
            .warning
            .set_foreground(Some(process_control_panel::COLOR_NOT_STARTED));
        // Swing layout: warning.setAlignmentX(Component.CENTER_ALIGNMENT).
        let pnl_trimvol = instance.pnl_trimvol.get_component();
        pnl_trimvol.add(&instance.warning);
        // Swing layout: pnlImodFull X_AXIS BoxLayout, horizontal glue around and
        // between the buttons.
        instance
            .pnl_imod_full
            .add(&instance.btn_imod_full.get_component());
        instance
            .pnl_imod_full
            .add(&instance.btn_get_coordinates.get_component());
        pnl_trimvol.add(&instance.pnl_imod_full);
        pnl_trimvol.add(&instance.volume_range_panel.get_component());
        // Swing layout: pnlTrimvol.add(Box.createRigidArea(FixedDim.x0_y10)).
        pnl_trimvol.add(&instance.pnl_scale.get_container());
        // Swing layout: pnlTrimvol.add(Box.createRigidArea(FixedDim.x0_y10)).
        let pnl_reorientation = SpacedPanel::get_instance_void();
        pnl_reorientation.set_box_layout(spaced_panel::X_AXIS);
        // Swing layout: pnlReorientationChoices Y_AXIS BoxLayout.
        instance
            .pnl_reorientation_choices
            .set_border(&EtchedBorder::new(Some(REORIENTATION_GROUP_LABEL)).get_border());
        // Swing layout: pnlReorientationChoices.setAlignmentX(RIGHT_ALIGNMENT).
        instance
            .bg_reorientation
            .add(&instance.rb_none.get_abstract_button());
        instance
            .bg_reorientation
            .add(&instance.rb_swap_yz.get_abstract_button());
        instance
            .bg_reorientation
            .add(&instance.rb_rotate_x.get_abstract_button());
        // Swing layout: rbNone/rbSwapYZ/rbRotateX.setAlignmentX(LEFT_ALIGNMENT).
        let pnl_reorientation_choices = instance.pnl_reorientation_choices.get_component();
        pnl_reorientation_choices.add(&instance.rb_none.get_component());
        pnl_reorientation_choices.add(&instance.rb_swap_yz.get_component());
        pnl_reorientation_choices.add(&instance.rb_rotate_x.get_component());
        pnl_reorientation.add_j_panel(&pnl_reorientation_choices);
        // reorientation warning panel
        let pnl_reorientation_warning = JComponent::new_panel();
        // Swing layout: pnlReorientationWarning Y_AXIS BoxLayout.
        pnl_reorientation_warning.add(&instance.l_warning1);
        pnl_reorientation_warning.add(&instance.l_warning2);
        pnl_reorientation_warning.add(&instance.l_warning3);
        pnl_reorientation_warning.add(&instance.l_warning4);
        pnl_reorientation_warning.add(&instance.l_warning5);
        pnl_reorientation.add_j_panel(&pnl_reorientation_warning);
        // trimvol panel
        pnl_trimvol.add(&pnl_reorientation.get_container());
        // Swing layout: pnlTrimvol.add(Box.createRigidArea(FixedDim.x0_y10)).
        pnl_trimvol.add(&instance.pnl_button);
        // Swing layout: pnlTrimvol.add(Box.createRigidArea(FixedDim.x0_y10)).

        instance.set_tool_tip_text();

        // Java `ScalingListener ScalingListener = new ScalingListener(this)`.
        let listenee = Rc::downgrade(&instance);
        let scaling_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(listenee) = listenee.upgrade() {
                listenee.scale_action(event);
            }
        });
        instance
            .rb_scale_fixed
            .add_action_listener(scaling_listener.clone());
        instance
            .rb_scale_section
            .add_action_listener(scaling_listener.clone());
        instance
            .cb_convert_to_bytes
            .add_action_listener(Some(scaling_listener));

        instance
            .btn_imod_full
            .add_action_listener(instance.button_acton_listener.clone());
        instance
            .btn_trimvol
            .add_action_listener(instance.button_acton_listener.clone());
        instance
            .btn_imod_trim
            .add_action_listener(instance.button_acton_listener.clone());
        instance
            .btn_get_coordinates
            .add_action_listener(instance.button_acton_listener.clone());

        // pnlTrimvol.addMouseListener(new GenericMouseAdapter(this)).
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(&instance) as Weak<dyn ContextMenu>;
        instance
            .pnl_trimvol
            .get_component()
            .add_mouse_listener(GenericMouseAdapter::new(context_menu));
        instance
    }

    /// Java package-private `getContainer()`.  Return the container of the
    /// panel.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_trimvol.get_component()
    }

    /// Java package-private `initParameters(TrimvolParam)`.  The param is
    /// `&mut` because `RubberbandPanel.initScaleParameters` reads it through
    /// `TrimvolParam.getScaleXYParam()`, which hands out the mutable member.
    pub fn init_parameters(&self, param: &mut TrimvolParam) {
        if self.trimvol_input_file_missing {
            return;
        }
        // volumeRangePanel.setParameters(param);
        if param.is_swap_yz() {
            self.rb_swap_yz.set_selected_boolean(true);
        } else if param.is_rotate_x() {
            self.rb_rotate_x.set_selected_boolean(true);
        } else {
            self.rb_none.set_selected_boolean(true);
        }
        // ConvertToBytes default is coming from metadata where is can be
        // overridden by batchruntomo.
        if param.is_fixed_scaling() {
            self.rb_scale_fixed.set_selected_boolean(true);
        } else {
            self.ltf_section_scale_min
                .set_text_const_etomo_number(Some(param.get_section_scale_min()));
            self.ltf_section_scale_max
                .set_text_const_etomo_number(Some(param.get_section_scale_max()));
            self.rb_scale_section.set_selected_boolean(true);
        }
        self.volume_range_panel.init_parameters(param);
        self.pnl_scale_rubberband.init_scale_parameters(param);
        self.set_scale_state();
    }

    /// Java package-private `setParameters(ConstMetaData, boolean)`.  Set the
    /// panel values with the specified parameters.
    pub fn set_parameters_const_meta_data_boolean(
        &self,
        meta_data: &dyn ConstMetaData,
        dialog_exists: bool,
    ) {
        if !dialog_exists || self.trimvol_input_file_missing {
            // TrimvolParam can calculate the initial values, while metaData would
            // have nothing from this panel if the dialog hadn't been created yet.
            return;
        }
        self.volume_range_panel.set_parameters(meta_data);
        if meta_data.is_post_trimvol_swap_yz() {
            self.rb_swap_yz.set_selected_boolean(true);
        } else if meta_data.is_post_trimvol_rotate_x() {
            self.rb_rotate_x.set_selected_boolean(true);
        } else {
            self.rb_none.set_selected_boolean(true);
        }

        self.cb_convert_to_bytes
            .set_selected_boolean(meta_data.is_post_trimvol_convert_to_bytes());
        if meta_data.is_post_trimvol_fixed_scaling() {
            self.ltf_fixed_scale_min
                .set_text_string(Some(&meta_data.get_post_trimvol_fixed_scale_min()));
            self.ltf_fixed_scale_max
                .set_text_string(Some(&meta_data.get_post_trimvol_fixed_scale_max()));
            self.rb_scale_fixed.set_selected_boolean(true);
        } else {
            self.ltf_section_scale_min
                .set_text_string(Some(&meta_data.get_post_trimvol_section_scale_min()));
            self.ltf_section_scale_max
                .set_text_string(Some(&meta_data.get_post_trimvol_section_scale_max()));
            self.rb_scale_section.set_selected_boolean(true);
        }
        self.set_scale_state();
        self.pnl_scale_rubberband
            .set_parameters_const_meta_data(meta_data);
    }

    /// Java package-private `setStartupWarnings(TrimvolInputFileState)`.
    /// Returns true if a warning that says "...values have been restored to
    /// defaults" has been displayed.
    pub fn set_startup_warnings(&self, input_file_state: &TrimvolInputFileState) -> bool {
        // set warning
        if input_file_state.is_n_columns_changed() || input_file_state.is_n_rows_changed() {
            if input_file_state.is_n_sections_changed() {
                self.warning
                    .set_text("Min and max values have been restored to defaults");
            } else {
                self.warning
                    .set_text("X,Y values have been restored to defaults");
            }
            return true;
        }
        if input_file_state.is_n_sections_changed() {
            self.warning
                .set_text("Z values have been restored to defaults");
            return true;
        }
        self.warning.set_visible(false);
        false
    }

    /// Java package-private `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        if self.trimvol_input_file_missing {
            return;
        }
        self.volume_range_panel.get_parameters_meta_data(meta_data);
        meta_data.set_post_trimvol_swap_yz(self.rb_swap_yz.is_selected());
        meta_data.set_post_trimvol_rotate_x(self.rb_rotate_x.is_selected());
        meta_data.set_post_trimvol_convert_to_bytes(self.cb_convert_to_bytes.is_selected());
        meta_data.set_post_trimvol_fixed_scaling(self.rb_scale_fixed.is_selected());
        meta_data
            .set_post_trimvol_fixed_scale_min(self.ltf_fixed_scale_min.get_text_void().as_deref());
        meta_data
            .set_post_trimvol_fixed_scale_max(self.ltf_fixed_scale_max.get_text_void().as_deref());
        meta_data.set_post_trimvol_section_scale_min(
            self.ltf_section_scale_min.get_text_void().as_deref(),
        );
        meta_data.set_post_trimvol_section_scale_max(
            self.ltf_section_scale_max.get_text_void().as_deref(),
        );
        // get the xyParam and set the values in it
        self.pnl_scale_rubberband
            .get_parameters_meta_data(meta_data);
    }

    /// Java package-private `getParametersForTrimvol(MetaData)`.
    pub fn get_parameters_for_trimvol(&self, meta_data: &MetaData) {
        if self.trimvol_input_file_missing {
            return;
        }
        self.volume_range_panel
            .get_parameters_for_trimvol(meta_data);
        meta_data.set_post_trimvol_scaling_new_style_z(
            self.ltf_section_scale_min.get_text_void().as_deref(),
            self.ltf_section_scale_max.get_text_void().as_deref(),
        );
    }

    /// Java package-private `getParameters(TrimvolParam, boolean)`.  Get the
    /// parameter values from the panel.
    pub fn get_parameters_trimvol_param_boolean(
        &self,
        trimvol_param: &mut TrimvolParam,
        do_validation: bool,
    ) -> bool {
        if self.trimvol_input_file_missing {
            return true;
        }
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            if !self
                .volume_range_panel
                .get_parameters_trimvol_param_boolean(trimvol_param, do_validation)
            {
                return Ok(false);
            }
            // Assume volume is flipped and set flipped - this means that Y and Z
            // don't have to be swapped when setting them.
            trimvol_param.set_flipped_volume(true);
            trimvol_param.set_swap_yz(self.rb_swap_yz.is_selected());
            trimvol_param.set_rotate_x(self.rb_rotate_x.is_selected());
            trimvol_param.set_format_of_output_file(Some(
                self.application_manager
                    .get_meta_data()
                    .base()
                    .get_image_output_format(),
            ));

            trimvol_param.set_convert_to_bytes(self.cb_convert_to_bytes.is_selected());
            let manager: &'static dyn BaseManager = self.application_manager;
            // Java `throw new InvalidEtomoNumberException(errorMessage)` after
            // each message dialog, caught right below by `catch
            // (InvalidEtomoNumberException e) { return false; }`: each is a
            // `return Ok(false)` here.
            let mut error_message: Option<String>;
            if self.rb_scale_fixed.is_selected() {
                trimvol_param.set_fixed_scaling(true);

                error_message = trimvol_param
                    .set_fixed_scale_min(
                        self.ltf_fixed_scale_min
                            .get_text_boolean(do_validation)?
                            .as_deref(),
                    )
                    .validate(Some(FIXED_SCALE_MIN_LABEL));
                if let Some(message) = &error_message {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(manager),
                            message,
                            SCALING_ERROR_TITLE,
                            Some(self.axis_id),
                        )
                    });
                    return Ok(false);
                }
                error_message = trimvol_param
                    .set_fixed_scale_max(
                        self.ltf_fixed_scale_max
                            .get_text_boolean(do_validation)?
                            .as_deref(),
                    )
                    .validate(Some(FIXED_SCALE_MAX_LABEL));
                if let Some(message) = &error_message {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(manager),
                            message,
                            SCALING_ERROR_TITLE,
                            Some(self.axis_id),
                        )
                    });
                    return Ok(false);
                }
            } else {
                trimvol_param.set_fixed_scaling(false);
                error_message = trimvol_param
                    .set_section_scale_min(
                        self.ltf_section_scale_min
                            .get_text_boolean(do_validation)?
                            .as_deref(),
                    )
                    .validate(Some(SECTION_SCALE_MIN_LABEL));
                if let Some(message) = &error_message {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(manager),
                            message,
                            SCALING_ERROR_TITLE,
                            Some(self.axis_id),
                        )
                    });
                    return Ok(false);
                }
                error_message = trimvol_param
                    .set_section_scale_max(
                        self.ltf_section_scale_max
                            .get_text_boolean(do_validation)?
                            .as_deref(),
                    )
                    .validate(Some(SECTION_SCALE_MAX_LABEL));
                if let Some(message) = &error_message {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(manager),
                            message,
                            SCALING_ERROR_TITLE,
                            Some(self.axis_id),
                        )
                    });
                    return Ok(false);
                }
            }
            let _ = error_message;
            // get the xyParam and set the values in it
            if !self
                .pnl_scale_rubberband
                .get_scale_parameters(trimvol_param, do_validation)
            {
                return Ok(false);
            }
            Ok(true)
        })();
        result.unwrap_or(false)
    }

    /// Java package-private `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        if self.trimvol_input_file_missing {
            return;
        }
        self.btn_trimvol.set_button_state(
            screen_state.get_button_state(self.btn_trimvol.get_button_state_key().as_deref()),
        );
    }

    /// Java private `setScaleState()`.  Enable/disable the appropriate text
    /// fields for the scale section.
    fn set_scale_state(&self) {
        self.rb_scale_fixed
            .set_enabled(self.cb_convert_to_bytes.is_selected());
        self.rb_scale_section
            .set_enabled(self.cb_convert_to_bytes.is_selected());
        let fixed_state =
            self.cb_convert_to_bytes.is_selected() && self.rb_scale_fixed.is_selected();
        self.ltf_fixed_scale_min.set_enabled(fixed_state);
        self.ltf_fixed_scale_max.set_enabled(fixed_state);
        let scale_state =
            self.cb_convert_to_bytes.is_selected() && self.rb_scale_section.is_selected();
        self.ltf_section_scale_min.set_enabled(scale_state);
        self.ltf_section_scale_max.set_enabled(scale_state);
        self.pnl_scale_rubberband.set_enabled(scale_state);
    }

    /// Java package-private `scaleAction(ActionEvent)`.  Call setScaleState
    /// when the radio buttons change.
    pub fn scale_action(&self, _event: &ActionEvent) {
        self.set_scale_state();
    }

    /// Java package-private `done()`.
    pub fn done(&self) {
        self.btn_trimvol
            .remove_action_listener(&self.button_acton_listener);
    }

    /// Java private `cbConvertToBytesAction(ActionEvent)`.  Never called in
    /// the source (no listener is registered for it).
    #[allow(dead_code)]
    fn cb_convert_to_bytes_action(&self, _event: &ActionEvent) {
        let state = self.cb_convert_to_bytes.is_selected();
        self.rb_scale_fixed.set_enabled(state);
        self.ltf_fixed_scale_max.set_enabled(state);
        self.ltf_fixed_scale_min.set_enabled(state);

        self.rb_scale_section.set_enabled(state);
        self.ltf_section_scale_min.set_enabled(state);
        self.ltf_section_scale_max.set_enabled(state);
    }

    /// Java private `setToolTipText()`.  Initialize the tooltip text.
    fn set_tool_tip_text(&self) {
        self.cb_convert_to_bytes.set_tool_tip_text_string(Some(
            "Scale densities to bytes with extreme densities truncated.",
        ));
        self.rb_scale_fixed.set_tool_tip_text_string(Some(
            "Set the scaling to match the contrast in a 3dmod display.",
        ));
        self.ltf_fixed_scale_min.set_tool_tip_text(Some(
            "Enter the black contrast slider setting (0-254) that gives the desired contrast.",
        ));
        self.ltf_fixed_scale_max.set_tool_tip_text(Some(
            "Enter the white contrast slider setting (1-255) that gives the desired contrast.",
        ));
        self.rb_scale_section.set_tool_tip_text_string(Some(
            "Set the scaling based on the range of contrast in a subset of sections and XY \
             volume.  Exclude areas with extreme densities that can be truncated (gold \
             particles).",
        ));
        self.ltf_section_scale_min.set_tool_tip_text(Some(
            "Minimum Z section of the subset to analyze for contrast range.",
        ));
        self.ltf_section_scale_max.set_tool_tip_text(Some(
            "Maximum Z section of the subset to analyze for contrast range.",
        ));
        self.pnl_reorientation_choices
            .get_component()
            .set_tool_tip_text(Some(
                "If the output volume is not reoriented, the file will need to be flipped \
                 when loaded into 3dmod.",
            ));
        self.rb_none.set_tool_tip_text_string(Some(
            "Do not change the orientation of the output volume.  The file will need to be \
             flipped when loaded into 3dmod.",
        ));
        self.rb_swap_yz.set_tool_tip_text_string(Some(
            "Flip Y and Z in the output volume so that the file does not need to be flipped \
             when loaded into 3dmod.",
        ));
        self.rb_rotate_x.set_tool_tip_text_string(Some(
            "Rotate the output volume by -90 degrees around the X axis, by first creating a \
             temporary trimmed volume with newstack then running \"clip rotx\" on this volume \
             to create the final output file.  The slices will look the same as with the -yz \
             option but rotating instead of flipping will preserve the handedness of \
             structures.",
        ));
        self.btn_imod_full
            .set_tool_tip_text(Some("View the original, untrimmed volume in 3dmod."));
        self.btn_get_coordinates.set_tool_tip_text(Some(
            "After pressing the 3dmod Full Volume button, press shift-B in the ZaP window.  \
             Create a rubberband around the volume range.  Then press this button to \
             retrieve X and Y coordinates.",
        ));
        self.btn_trimvol.set_tool_tip_text(Some(
            "Trim the original volume with the parameters given above.",
        ));
        self.btn_imod_trim
            .set_tool_tip_text(Some("View the trimmed volume."));
    }
}

impl Run3dmodButtonContainer for TrimvolPanel {
    /// Java public `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // Java passes a null Run3dmodMenuOptions from the ButtonListener; the
        // ApplicationManager methods take the options by value, and
        // ImodState.open replaces null with a new Run3dmodMenuOptions() (the
        // default value).
        if Some(command) == self.btn_trimvol.get_action_command().as_deref() {
            self.application_manager.trim_volume(
                Some(self.btn_trimvol.clone() as ProcessResultDisplayHandle),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options.unwrap_or_default(),
                self.dialog_type,
            );
        } else if Some(command) == self.btn_get_coordinates.get_action_command().as_deref() {
            self.volume_range_panel.set_xy_min_and_max(
                self.application_manager
                    .imod_get_rubberband_coordinates(
                        Some(imod_manager::COMBINED_TOMOGRAM_KEY),
                        Some(AxisID::Only),
                    )
                    .as_deref(),
            );
        } else if Some(command) == self.btn_imod_full.get_action_command().as_deref() {
            self.application_manager
                .imod_combined_tomogram(run_3dmod_menu_options.unwrap_or_default());
        } else if Some(command) == self.btn_imod_trim.get_action_command().as_deref() {
            self.application_manager
                .imod_trimmed_volume(run_3dmod_menu_options.unwrap_or_default(), self.axis_id);
        }
    }
}

impl RubberbandContainer for TrimvolPanel {
    /// Java public `setRubberbandContainerZMin(String)`.  Set scale section Min
    /// if selected.  From the subarea scale.
    fn set_rubberband_container_z_min(&self, section_scale_min: Option<&str>) {
        if self.rb_scale_section.is_selected() {
            self.ltf_section_scale_min
                .set_text_string(section_scale_min);
        }
    }

    /// Java public `setRubberbandContainerZMax(String)`.  Set scale section Max
    /// if selected.  From the subarea scale.
    fn set_rubberband_container_z_max(&self, section_scale_max: Option<&str>) {
        if self.rb_scale_section.is_selected() {
            self.ltf_section_scale_max
                .set_text_string(section_scale_max);
        }
    }
}

impl TrimvolDisplay for TrimvolPanel {
    /// Java public `setSwapYZ(boolean)`.
    fn set_swap_yz(&self, input: bool) {
        if input {
            self.rb_swap_yz.set_selected_boolean(input);
        }
    }

    /// Java public `setRotateX(boolean)`.
    fn set_rotate_x(&self, input: bool) {
        if input {
            self.rb_rotate_x.set_selected_boolean(input);
        }
    }

    /// Java public `setConvertToBytes(boolean)`.
    fn set_convert_to_bytes(&self, input: bool) {
        self.cb_convert_to_bytes.set_selected_boolean(input);
    }

    /// Java public `setSectionScaleMin(String)`.
    fn set_section_scale_min(&self, input: &str) {
        self.ltf_section_scale_min.set_text_string(Some(input));
    }

    /// Java public `setSectionScaleMax(String)`.
    fn set_section_scale_max(&self, input: &str) {
        self.ltf_section_scale_max.set_text_string(Some(input));
    }

    /// Java public `setXMin(String)`.
    fn set_x_min(&self, input: &str) {
        self.volume_range_panel.set_x_min(Some(input));
    }

    /// Java public `setXMax(String)`.
    fn set_x_max(&self, input: &str) {
        self.volume_range_panel.set_x_max(Some(input));
    }

    /// Java public `setYMin(String)`.
    fn set_y_min(&self, input: &str) {
        self.volume_range_panel.set_y_min(Some(input));
    }

    /// Java public `setYMax(String)`.
    fn set_y_max(&self, input: &str) {
        self.volume_range_panel.set_y_max(Some(input));
    }

    /// Java public `setZMin(String)`.
    fn set_z_min(&self, input: &str) {
        self.volume_range_panel.set_z_min(Some(input));
    }

    /// Java public `setZMax(String)`.
    fn set_z_max(&self, input: &str) {
        self.volume_range_panel.set_z_max(Some(input));
    }

    /// Java public `setScaleXMin(String)`.
    fn set_scale_x_min(&self, input: &str) {
        self.pnl_scale_rubberband.set_x_min(Some(input));
    }

    /// Java public `setScaleYMin(String)`.
    fn set_scale_y_min(&self, input: &str) {
        self.pnl_scale_rubberband.set_y_min(Some(input));
    }

    /// Java public `setScaleXMax(String)`.
    fn set_scale_x_max(&self, input: &str) {
        self.pnl_scale_rubberband.set_x_max(Some(input));
    }

    /// Java public `setScaleYMax(String)`.
    fn set_scale_y_max(&self, input: &str) {
        self.pnl_scale_rubberband.set_y_max(Some(input));
    }
}

impl ContextMenu for TrimvolPanel {
    /// Java public `popUpContextMenu(MouseEvent)`.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = ["Trimvol".to_string()];
        let man_page = ["trimvol.html".to_string()];

        let manager: &'static dyn BaseManager = self.application_manager;
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_base_manager_axis_id(
            &self.pnl_trimvol.get_component(),
            mouse_event,
            Some("POST-PROCESSING"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            manager,
            self.axis_id,
        );
    }
}
