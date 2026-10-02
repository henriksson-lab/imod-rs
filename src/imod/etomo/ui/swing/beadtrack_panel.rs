//! `IMOD/Etomo/src/etomo/ui/swing/BeadtrackPanel.java`.
//!
//! Java `public final class BeadtrackPanel implements Expandable,
//! Run3dmodButtonContainer, BeadTrackDisplay`.  An EDT object: created as
//! `Rc<Self>` by [`BeadtrackPanel::get_instance`], every method takes `&self`,
//! mutable state lives in `Cell`s.  `this` (handed to the two `PanelHeader`s
//! and to `btnFixModel.setContainer`) is the weak self reference made by
//! `Rc::new_cyclic`.

use crate::imod::etomo::ui::field::Field;
use std::cell::Cell;
use std::rc::{Rc, Weak};

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::beadtrack_param::{self, BeadtrackParam};
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::jdk::{ActionListener, JComponent};
use crate::imod::etomo::process::imod_process::{BeadFixerMode, Run3dmodMenuOptions};
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::const_etomo_number::{self, Type};
use crate::imod::etomo::r#type::const_panel_header_settings::ConstPanelHeaderSettings;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::invalid_etomo_number_exception::InvalidEtomoNumberException;
use crate::imod::etomo::ui::field_type::FieldType;

use super::bead_track_display::{BeadTrackDisplay, BeadTrackDisplayException};
use super::check_box::CheckBox;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::text_efield::TextEfield;
use super::ui_expert_utilities::UIExpertUtilities;
use super::ui_harness;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplay;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java `public static final TRACK_LABEL`.
pub const TRACK_LABEL: &str = "Track Seed Model";
/// Java `public static final USE_MODEL_LABEL`.
pub const USE_MODEL_LABEL: &str = "Track with Fiducial Model as Seed";
/// Java private static final `VIEW_SKIP_LIST_LABEL`.
const VIEW_SKIP_LIST_LABEL: &str = "View skip list";
/// Java package-private static final `LIGHT_BEADS_LABEL`.
pub const LIGHT_BEADS_LABEL: &str = "Light fiducial markers";

/// Java `public final class BeadtrackPanel implements Expandable,
/// Run3dmodButtonContainer, BeadTrackDisplay`.
pub struct BeadtrackPanel {
    /// Java private final `panelBeadtrackX`.
    panel_beadtrack_x: Rc<EtomoPanel>,
    /// Java private final `panelBeadtrack`.
    panel_beadtrack: Rc<EtomoPanel>,
    /// Java private final `panelBeadtrackBody`.
    panel_beadtrack_body: Rc<JComponent>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `btnFixModel`.
    btn_fix_model: Rc<Run3dmodButton>,

    ltf_view_skip_list: Rc<LabeledTextField>,
    ltf_additional_view_sets: Rc<LabeledTextField>,
    ltf_tilt_angle_group_size: Rc<LabeledTextField>,
    ltf_tilt_angle_groups: Rc<LabeledTextField>,
    ltf_magnification_group_size: Rc<LabeledTextField>,
    ltf_magnification_groups: Rc<LabeledTextField>,
    ltf_n_min_views: Rc<LabeledTextField>,
    ltf_bead_diameter: Rc<LabeledTextField>,
    cb_light_beads: Rc<CheckBox>,
    /// Java package-private (non-final) `cbFillGaps`.
    pub(crate) cb_fill_gaps: Rc<CheckBox>,
    ltf_max_gap: Rc<LabeledTextField>,
    ltf_min_tilt_range_to_find_axis: Rc<LabeledTextField>,
    ltf_min_tilt_range_to_find_angle: Rc<LabeledTextField>,
    ltf_search_box_pixels: Rc<LabeledTextField>,
    ltf_max_fiducials_avg: Rc<LabeledTextField>,
    ltf_fiducial_extrapolation_params: Rc<LabeledTextField>,
    ltf_rescue_attempt_params: Rc<LabeledTextField>,
    ltf_min_rescue_distance: Rc<LabeledTextField>,
    ltf_rescue_relaxtion_params: Rc<LabeledTextField>,
    ltf_residual_distance_limit: Rc<LabeledTextField>,
    ltf_mean_resid_change_limits: Rc<LabeledTextField>,
    ltf_deletion_params: Rc<LabeledTextField>,
    ltf_density_relaxation_post_fit: Rc<LabeledTextField>,
    ltf_max_rescue_distance: Rc<LabeledTextField>,

    cb_local_area_tracking: Rc<CheckBox>,
    ltf_local_area_target_size: Rc<LabeledTextField>,
    ltf_min_beads_in_area: Rc<LabeledTextField>,
    ltf_min_overlap_beads: Rc<LabeledTextField>,
    ltf_max_views_in_align: Rc<LabeledTextField>,
    ltf_rounds_of_tracking: Rc<LabeledTextField>,
    cb_sobel_filter_centering: Rc<CheckBox>,
    /// ScalableSigmaForSobel replaced KernelSigmaForSobel.
    ltf_scalable_sigma_for_sobel: Rc<LabeledTextField>,
    tf_low_pass_cutoff_inverse_nm: Rc<TextEfield>,

    pnl_checkbox: Rc<JComponent>,
    pnl_light_beads: Rc<JComponent>,
    pnl_local_area_tracking: Rc<JComponent>,
    pnl_expert_parameters: Rc<EtomoPanel>,
    pnl_expert_parameters_body: Rc<JComponent>,

    /// Java private final `expertParametersHeader`.
    expert_parameters_header: Rc<PanelHeader>,
    /// Java private final `header`.
    header: Rc<PanelHeader>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `btnTrack`.
    btn_track: Rc<MultiLineButton>,
    /// Java private final `btnUseModel`.
    btn_use_model: Rc<MultiLineButton>,
    /// Java private final `actionListener` (`BeadtrackPanelActionListener`).
    action_listener: ActionListener,
    pnl_fill_gaps: Rc<JComponent>,
    pnl_track: Rc<JComponent>,

    /// Java private final `dialogType`.
    dialog_type: DialogType,

    /// Java private `autofidseedMode`.
    autofidseed_mode: Cell<bool>,
}

impl BeadtrackPanel {
    /// Java private constructor `BeadtrackPanel(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton)`: construct a new beadtrack panel.
    fn new(
        manager: &'static ApplicationManager,
        id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<BeadtrackPanel> {
        Rc::new_cyclic(|this: &Weak<BeadtrackPanel>| {
            // Field initializers, in declaration order.
            let panel_beadtrack_x = EtomoPanel::new();
            let panel_beadtrack = EtomoPanel::new();
            let panel_beadtrack_body = JComponent::new_panel();
            let ltf_view_skip_list = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some(&format!("{}: ", VIEW_SKIP_LIST_LABEL)),
            );
            let ltf_additional_view_sets = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Separate view groups: "),
            );
            let ltf_tilt_angle_group_size = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Tilt angle group size: "),
            );
            let ltf_tilt_angle_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("Non-default tilt angle groups: "),
            );
            let ltf_magnification_group_size = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Magnification group size: "),
            );
            let ltf_magnification_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("Non-default magnification groups: "),
            );
            let ltf_n_min_views = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Minimum # of views for tilt alignment: "),
            );
            let ltf_bead_diameter = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Unbinned bead diameter: "),
            );
            let cb_light_beads = CheckBox::new_string(Some(LIGHT_BEADS_LABEL));
            let cb_fill_gaps = CheckBox::new_string(Some("Fill seed model gaps"));
            let ltf_max_gap = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Maximum gap size: "),
            );
            let ltf_min_tilt_range_to_find_axis = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Minimum tilt range for finding axis: "),
            );
            let ltf_min_tilt_range_to_find_angle = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Minimum tilt range for finding angles: "),
            );
            let ltf_search_box_pixels = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("Search box size (pixels): "),
            );
            let ltf_max_fiducials_avg = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Maximum # of views for fiducial avg.: "),
            );
            let ltf_fiducial_extrapolation_params = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("Fiducial extrapolation limits: "),
            );
            let ltf_rescue_attempt_params = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointPair,
                Some("Rescue attempt criteria: "),
            );
            let ltf_min_rescue_distance = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Distance criterion for rescue (pixels): "),
            );
            let ltf_rescue_relaxtion_params = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointPair,
                Some("Rescue relaxation factors: "),
            );
            let ltf_residual_distance_limit = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("First pass residual limit for deletion: "),
            );
            let ltf_mean_resid_change_limits = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("Residual change limits: "),
            );
            let ltf_deletion_params = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointPair,
                Some("Deletion residual parameters: "),
            );
            let ltf_density_relaxation_post_fit = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Second pass density relaxation: "),
            );
            let ltf_max_rescue_distance = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Second pass maximum rescue distance: "),
            );

            let cb_local_area_tracking = CheckBox::new_string(Some("Local tracking"));
            let ltf_local_area_target_size = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Local area size: "),
            );
            let ltf_min_beads_in_area = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Minimum beads in area: "),
            );
            let ltf_min_overlap_beads = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Minimum beads overlapping: "),
            );
            let ltf_max_views_in_align = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Max. # views to include in align: "),
            );
            let ltf_rounds_of_tracking = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Rounds of tracking: "),
            );
            let cb_sobel_filter_centering =
                CheckBox::new_string(Some("Refine center with Sobel filter"));
            // ScalableSigmaForSobel replaced KernelSigmaForSobel.
            let ltf_scalable_sigma_for_sobel = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Sobel sigma relative to bead size: "),
            );
            let tf_low_pass_cutoff_inverse_nm = TextEfield::get_labeled_instance(
                Some("Overall low-pass filter cutoff (/nm): "),
                Some(FieldType::FloatingPoint),
            );

            let pnl_checkbox = JComponent::new_panel();
            let pnl_light_beads = JComponent::new_panel();
            let pnl_local_area_tracking = JComponent::new_panel();
            let pnl_expert_parameters = EtomoPanel::new();
            let pnl_expert_parameters_body = JComponent::new_panel();

            let btn_use_model = MultiLineButton::new_string(Some(USE_MODEL_LABEL));
            // BeadtrackPanelActionListener
            let action_listener: ActionListener = {
                let adaptee = this.clone();
                Rc::new(move |event| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                    }
                })
            };
            let pnl_fill_gaps = JComponent::new_panel();
            let pnl_track = JComponent::new_panel();

            // Constructor body.
            let axis_id = id;
            let btn_track = manager
                .get_process_result_display_factory(axis_id)
                .get_track_fiducials();
            let btn_fix_model = manager
                .get_process_result_display_factory(axis_id)
                .get_fix_fiducial_model();
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            btn_fix_model.set_container(Some(container));
            let expandable: Weak<dyn Expandable> = this.clone();
            let expert_parameters_header = PanelHeader::get_instance(
                Some("Expert Parameters"),
                Some(expandable.clone()),
                Some(dialog_type),
            );
            let header =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Beadtracker"),
                    Some(expandable),
                    Some(dialog_type),
                    Some(global_advanced_button.clone()),
                );
            tf_low_pass_cutoff_inverse_nm.set_preferred_width(70);
            ltf_scalable_sigma_for_sobel.set_max_decimal_places(3);

            let pnl_sobel_filter_centering = JComponent::new_panel();

            // Swing layout: panelBeadtrackBody BoxLayout Y_AXIS; rigid area x0_y5.
            panel_beadtrack_body.add(&ltf_view_skip_list.get_container());
            panel_beadtrack_body.add(&ltf_additional_view_sets.get_container());
            panel_beadtrack_body.add(&ltf_tilt_angle_group_size.get_container());
            panel_beadtrack_body.add(&ltf_tilt_angle_groups.get_container());
            panel_beadtrack_body.add(&ltf_magnification_group_size.get_container());
            panel_beadtrack_body.add(&ltf_magnification_groups.get_container());
            panel_beadtrack_body.add(&ltf_n_min_views.get_container());
            panel_beadtrack_body.add(&ltf_bead_diameter.get_container());

            // Swing layout: pnlLightBeads BoxLayout Y_AXIS, CENTER_ALIGNMENT.
            pnl_light_beads.add(&cb_light_beads.get_component());
            // Swing layout: horizontal glue.
            panel_beadtrack_body.add(&pnl_light_beads);
            // Swing layout: rigid area x0_y2.
            panel_beadtrack_body.add(&pnl_sobel_filter_centering);
            panel_beadtrack_body.add(&ltf_scalable_sigma_for_sobel.get_container());
            panel_beadtrack_body.add(&tf_low_pass_cutoff_inverse_nm.get_component());
            // Swing layout: rigid area x0_y2; pnlCheckbox BoxLayout Y_AXIS,
            // CENTER_ALIGNMENT; pnlFillGaps BoxLayout X_AXIS, CENTER_ALIGNMENT.
            pnl_fill_gaps.add(&cb_fill_gaps.get_component());
            // Swing layout: horizontal glue.
            pnl_checkbox.add(&pnl_fill_gaps);

            panel_beadtrack_body.add(&pnl_checkbox);
            panel_beadtrack_body.add(&ltf_max_gap.get_container());

            // Swing layout: pnlLocalAreaTracking BoxLayout Y_AXIS, CENTER_ALIGNMENT.
            pnl_local_area_tracking.add(&cb_local_area_tracking.get_component());
            // Swing layout: horizontal glue.
            panel_beadtrack_body.add(&pnl_local_area_tracking);

            panel_beadtrack_body.add(&ltf_local_area_target_size.get_container());
            panel_beadtrack_body.add(&ltf_min_beads_in_area.get_container());
            panel_beadtrack_body.add(&ltf_min_overlap_beads.get_container());
            panel_beadtrack_body.add(&ltf_max_views_in_align.get_container());
            panel_beadtrack_body.add(&ltf_rounds_of_tracking.get_container());

            panel_beadtrack_body.add(&ltf_min_tilt_range_to_find_axis.get_container());
            panel_beadtrack_body.add(&ltf_min_tilt_range_to_find_angle.get_container());
            panel_beadtrack_body.add(&ltf_search_box_pixels.get_container());
            // Swing layout: rigid area x0_y5.

            // SobelFilterCentering
            // Swing layout: pnlSobelFilterCentering BoxLayout X_AXIS.
            pnl_sobel_filter_centering.add(&cb_sobel_filter_centering.get_component());
            // Swing layout: horizontal glue.

            // Swing layout: pnlExpertParametersBody BoxLayout Y_AXIS; rigid area x0_y5.
            pnl_expert_parameters_body.add(&ltf_max_fiducials_avg.get_container());
            pnl_expert_parameters_body.add(&ltf_fiducial_extrapolation_params.get_container());
            pnl_expert_parameters_body.add(&ltf_rescue_attempt_params.get_container());
            pnl_expert_parameters_body.add(&ltf_min_rescue_distance.get_container());
            pnl_expert_parameters_body.add(&ltf_rescue_relaxtion_params.get_container());
            pnl_expert_parameters_body.add(&ltf_residual_distance_limit.get_container());
            pnl_expert_parameters_body.add(&ltf_density_relaxation_post_fit.get_container());
            pnl_expert_parameters_body.add(&ltf_max_rescue_distance.get_container());
            pnl_expert_parameters_body.add(&ltf_mean_resid_change_limits.get_container());
            pnl_expert_parameters_body.add(&ltf_deletion_params.get_container());

            // Swing layout: pnlExpertParameters BoxLayout Y_AXIS, etched border.
            pnl_expert_parameters.add(&expert_parameters_header);
            pnl_expert_parameters
                .get_component()
                .add(&pnl_expert_parameters_body);
            panel_beadtrack_body.add(&pnl_expert_parameters.get_component());

            // Swing layout: btnTrack CENTER_ALIGNMENT.
            panel_beadtrack_body.add(&btn_track.get_component());

            // Swing layout: pnlTrack BoxLayout X_AXIS, CENTER_ALIGNMENT;
            // btnFixModel CENTER_ALIGNMENT.
            pnl_track.add(&btn_fix_model.get_component());
            // Swing layout: rigid area x5_y0.
            pnl_track.add(&btn_use_model.get_component());
            // Swing layout: rigid area x0_y5.
            panel_beadtrack_body.add(&pnl_track);
            // Swing layout: rigid area x0_y5.

            // Swing layout: panelBeadtrack BoxLayout Y_AXIS, etched border.
            panel_beadtrack.add(&header);
            panel_beadtrack.get_component().add(&panel_beadtrack_body);

            // Swing layout: panelBeadtrackX BoxLayout X_AXIS.
            panel_beadtrack_x
                .get_component()
                .add(&panel_beadtrack.get_component());

            BeadtrackPanel {
                panel_beadtrack_x,
                panel_beadtrack,
                panel_beadtrack_body,
                axis_id,
                btn_fix_model,
                ltf_view_skip_list,
                ltf_additional_view_sets,
                ltf_tilt_angle_group_size,
                ltf_tilt_angle_groups,
                ltf_magnification_group_size,
                ltf_magnification_groups,
                ltf_n_min_views,
                ltf_bead_diameter,
                cb_light_beads,
                cb_fill_gaps,
                ltf_max_gap,
                ltf_min_tilt_range_to_find_axis,
                ltf_min_tilt_range_to_find_angle,
                ltf_search_box_pixels,
                ltf_max_fiducials_avg,
                ltf_fiducial_extrapolation_params,
                ltf_rescue_attempt_params,
                ltf_min_rescue_distance,
                ltf_rescue_relaxtion_params,
                ltf_residual_distance_limit,
                ltf_mean_resid_change_limits,
                ltf_deletion_params,
                ltf_density_relaxation_post_fit,
                ltf_max_rescue_distance,
                cb_local_area_tracking,
                ltf_local_area_target_size,
                ltf_min_beads_in_area,
                ltf_min_overlap_beads,
                ltf_max_views_in_align,
                ltf_rounds_of_tracking,
                cb_sobel_filter_centering,
                ltf_scalable_sigma_for_sobel,
                tf_low_pass_cutoff_inverse_nm,
                pnl_checkbox,
                pnl_light_beads,
                pnl_local_area_tracking,
                pnl_expert_parameters,
                pnl_expert_parameters_body,
                expert_parameters_header,
                header,
                manager,
                btn_track,
                btn_use_model,
                action_listener,
                pnl_fill_gaps,
                pnl_track,
                dialog_type,
                autofidseed_mode: Cell::new(false),
            }
        })
    }

    /// Java `updateAutofidseed(boolean)`: changes the display to/from
    /// autofidseedMode.  Does nothing if autofidseedMode would be unchanged.
    pub fn update_autofidseed(&self, input: bool) {
        if input == self.autofidseed_mode.get() {
            return;
        }
        self.autofidseed_mode.set(input);
        // Change the padding
        self.panel_beadtrack_x.get_component().remove_all();
        if self.autofidseed_mode.get() {
            // Swing layout: rigid area x197_y0.
        }
        self.panel_beadtrack_x
            .get_component()
            .add(&self.panel_beadtrack.get_component());
        if self.autofidseed_mode.get() {
            // Swing layout: rigid area x197_y0.
        }
        // Change visibility of fields
        let autofidseed_mode = self.autofidseed_mode.get();
        self.ltf_tilt_angle_group_size
            .set_visible(!autofidseed_mode);
        self.ltf_tilt_angle_groups.set_visible(!autofidseed_mode);
        self.ltf_magnification_group_size
            .set_visible(!autofidseed_mode);
        self.ltf_magnification_groups.set_visible(!autofidseed_mode);
        self.ltf_n_min_views.set_visible(!autofidseed_mode);
        self.ltf_bead_diameter.set_visible(!autofidseed_mode);
        self.pnl_checkbox.set_visible(!autofidseed_mode);
        self.ltf_max_gap.set_visible(!autofidseed_mode);
        self.pnl_fill_gaps.set_visible(!autofidseed_mode);
        self.pnl_local_area_tracking.set_visible(!autofidseed_mode);
        self.ltf_local_area_target_size
            .set_visible(!autofidseed_mode);
        self.ltf_min_beads_in_area.set_visible(!autofidseed_mode);
        self.ltf_min_overlap_beads.set_visible(!autofidseed_mode);
        self.ltf_max_views_in_align.set_visible(!autofidseed_mode);
        self.ltf_rounds_of_tracking.set_visible(!autofidseed_mode);
        self.ltf_min_tilt_range_to_find_axis
            .set_visible(!autofidseed_mode);
        self.ltf_min_tilt_range_to_find_angle
            .set_visible(!autofidseed_mode);
        self.ltf_search_box_pixels.set_visible(!autofidseed_mode);
        self.pnl_expert_parameters
            .get_component()
            .set_visible(!autofidseed_mode);
        self.btn_track.set_visible(!autofidseed_mode);
        self.pnl_track.set_visible(!autofidseed_mode);
        self.update_advanced(self.header.is_advanced());
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType,
    /// GlobalExpandButton)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<BeadtrackPanel> {
        let instance = BeadtrackPanel::new(manager, id, dialog_type, global_advanced_button);
        // The Java constructor ends with setToolTipText(); it reads the
        // constructed fields, so it runs here, before addListeners() as in Java.
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.cb_local_area_tracking
            .add_action_listener(Some(self.action_listener.clone()));
        self.btn_track
            .add_action_listener(self.action_listener.clone());
        self.btn_use_model
            .add_action_listener(self.action_listener.clone());
        self.btn_fix_model
            .add_action_listener(self.action_listener.clone());
        self.cb_sobel_filter_centering
            .add_action_listener(Some(self.action_listener.clone()));
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.panel_beadtrack_x.get_component().set_visible(visible);
    }

    /// Java `setParameters(BaseScreenState)`.
    pub fn set_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.expert_parameters_header
            .set_button_states_base_screen_state_boolean(Some(screen_state), false);
        self.header
            .set_button_states_base_screen_state(Some(screen_state));
        // btnFixModel.setButtonState(screenState.getButtonState(btnFixModel
        // .getButtonStateKey()));
        // btnTrack.setButtonState(screenState.getButtonState(btnTrack
        // .getButtonStateKey()));
    }

    /// Java `getParameters(BaseScreenState)`.
    pub fn get_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.expert_parameters_header
            .get_button_states(Some(screen_state));
        self.header.get_button_states(Some(screen_state));
    }

    /// Java `setParameters(BeadtrackParam, boolean)`: set the field values for
    /// the panel from the ConstBeadtrackParam object.
    pub fn set_parameters_beadtrack_param_boolean(
        &self,
        beadtrack_params: &mut BeadtrackParam,
        for_transfer_fid: bool,
    ) {
        self.cb_light_beads
            .set_selected_boolean(beadtrack_params.get_light_beads().is());
        self.cb_sobel_filter_centering
            .set_selected_boolean(beadtrack_params.is_sobel_filter_centering());
        self.ltf_scalable_sigma_for_sobel
            .set_text_string(Some(&beadtrack_params.get_scalable_sigma_for_sobel()));
        if self.ltf_scalable_sigma_for_sobel.is_empty() {
            let mut kernel_sigma_for_sobel = EtomoNumber::new_with_type(Some(Type::Double));
            kernel_sigma_for_sobel.set_string(Some(&beadtrack_params.get_kernel_sigma_for_sobel()));
            if !kernel_sigma_for_sobel.is_null() {
                // Convert kernelSigmaForSobel into ScalableSigmaForSobel.
                beadtrack_params.convert_kernel_sigma_for_sobel_to_scalable_sigma_for_sobel(
                    self.manager.get_meta_data().get_fiducial_diameter(),
                    Some(&file_type::CLASS.prealigned_stack),
                );
                self.ltf_scalable_sigma_for_sobel
                    .set_text_string(Some(&beadtrack_params.get_scalable_sigma_for_sobel()));
            }
        }
        self.tf_low_pass_cutoff_inverse_nm
            .set_text_string(Some(&beadtrack_params.get_low_pass_cutoff_inverse_nm()));
        // Java `ConstEtomoNumber field = null;` (never read).
        if !for_transfer_fid {
            self.ltf_view_skip_list
                .set_text_string(Some(&beadtrack_params.get_skip_views()));
            self.ltf_additional_view_sets
                .set_text_string(Some(&beadtrack_params.get_additional_view_groups()));
            self.ltf_tilt_angle_group_size.set_text_string(Some(
                &beadtrack_params.get_tilt_default_grouping().to_string(),
            ));
            self.ltf_tilt_angle_groups
                .set_text_string(Some(&beadtrack_params.get_tilt_angle_groups()));
            self.ltf_magnification_group_size
                .set_text_int(beadtrack_params.get_magnification_group_size());
            self.ltf_magnification_groups
                .set_text_string(Some(&beadtrack_params.get_magnification_groups()));
            self.ltf_n_min_views.set_text_string(Some(
                &beadtrack_params.get_min_views_for_tiltalign().to_string(),
            ));
            self.ltf_bead_diameter
                .set_text_string(Some(&beadtrack_params.get_bead_diameter().to_string()));
            self.cb_fill_gaps
                .set_selected_boolean(beadtrack_params.get_fill_gaps());
            self.ltf_max_gap
                .set_text_string(Some(&beadtrack_params.get_max_gap_size().to_string()));
            self.ltf_min_tilt_range_to_find_axis.set_text_string(Some(
                &beadtrack_params
                    .get_min_tilt_range_to_find_axis()
                    .to_string(),
            ));
            self.ltf_min_tilt_range_to_find_angle.set_text_string(Some(
                &beadtrack_params
                    .get_min_tilt_range_to_find_angles()
                    .to_string(),
            ));
            self.ltf_search_box_pixels
                .set_text_string(Some(&beadtrack_params.get_search_box_pixels()));
            self.ltf_max_fiducials_avg.set_text_string(Some(
                &beadtrack_params.get_max_beads_to_average().to_string(),
            ));
            self.ltf_fiducial_extrapolation_params
                .set_text_string(Some(&beadtrack_params.get_fiducial_extrapolation_params()));
            self.ltf_rescue_attempt_params
                .set_text_string(Some(&beadtrack_params.get_rescue_attempt_params()));
            self.ltf_min_rescue_distance.set_text_string(Some(
                &beadtrack_params.get_distance_rescue_criterion().to_string(),
            ));
            self.ltf_rescue_relaxtion_params
                .set_text_string(Some(&beadtrack_params.get_rescue_relaxation_params()));
            self.ltf_residual_distance_limit.set_text_string(Some(
                &beadtrack_params.get_post_fit_rescue_residual().to_string(),
            ));
            self.ltf_density_relaxation_post_fit.set_text_string(Some(
                &beadtrack_params
                    .get_density_relaxation_post_fit()
                    .to_string(),
            ));
            self.ltf_max_rescue_distance.set_text_string(Some(
                &beadtrack_params.get_max_rescue_distance().to_string(),
            ));
            self.ltf_mean_resid_change_limits
                .set_text_string(Some(&beadtrack_params.get_mean_resid_change_limits()));
            self.ltf_deletion_params
                .set_text_string(Some(&beadtrack_params.get_deletion_params()));
            self.cb_local_area_tracking
                .set_selected_boolean(beadtrack_params.get_local_area_tracking().is());
            self.ltf_local_area_target_size.set_text_string(Some(
                &beadtrack_params.get_local_area_target_size().to_string(),
            ));
            self.ltf_min_beads_in_area
                .set_text_string(Some(&beadtrack_params.get_min_beads_in_area().to_string()));
            self.ltf_min_overlap_beads
                .set_text_string(Some(&beadtrack_params.get_min_overlap_beads().to_string()));
            self.ltf_max_views_in_align
                .set_text_string(Some(&beadtrack_params.get_max_views_in_align().to_string()));
            self.ltf_rounds_of_tracking
                .set_text_string(Some(&beadtrack_params.get_rounds_of_tracking().to_string()));
        }
        self.set_enabled();
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.panel_beadtrack_x.get_component()
    }

    /// Java private `setEnabled()`.
    fn set_enabled(&self) {
        self.ltf_local_area_target_size
            .set_enabled(self.cb_local_area_tracking.is_selected());
        self.ltf_min_beads_in_area
            .set_enabled(self.cb_local_area_tracking.is_selected());
        self.ltf_min_overlap_beads
            .set_enabled(self.cb_local_area_tracking.is_selected());
        self.ltf_scalable_sigma_for_sobel
            .set_enabled(self.cb_sobel_filter_centering.is_selected());
    }

    /// Java `updateAdvanced(boolean)`: makes the advanced components visible
    /// or invisible.
    pub fn update_advanced(&self, state: bool) {
        self.cb_light_beads.set_visible(state);
        if self.autofidseed_mode.get() {
            return;
        }
        self.ltf_tilt_angle_group_size.set_visible(state);
        self.ltf_tilt_angle_groups.set_visible(state);
        self.ltf_magnification_group_size.set_visible(state);
        self.ltf_magnification_groups.set_visible(state);
        self.ltf_n_min_views.set_visible(state);
        self.ltf_bead_diameter.set_visible(state);
        self.ltf_max_gap.set_visible(state);
        self.ltf_min_tilt_range_to_find_axis.set_visible(state);
        self.ltf_min_tilt_range_to_find_angle.set_visible(state);
        self.ltf_search_box_pixels.set_visible(state);
        self.pnl_expert_parameters
            .get_component()
            .set_visible(state);
        self.ltf_min_beads_in_area.set_visible(state);
        self.ltf_min_overlap_beads.set_visible(state);
        self.ltf_rounds_of_tracking.set_visible(state);
    }

    /// Java `done()`.
    pub fn done(&self) {
        self.btn_track.remove_action_listener(&self.action_listener);
        self.btn_fix_model
            .remove_action_listener(&self.action_listener);
    }

    /// Java private `setToolTipText()`: ToolTip string setup.
    fn set_tool_tip_text(&self) {
        // Java `String text;` (never read).
        let mut autodoc: Option<*mut Autodoc> = None;
        // SAFETY: `AutodocFactory` owns every autodoc it returns for the life
        // of the process (the Java GC-owned singletons), so the pointer stays
        // valid for this method.
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::BEADTRACK),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = Some(instance),
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except)
            Err(except) => eprintln!("{except}"),
        }
        let Some(autodoc) = autodoc else {
            return;
        };
        // SAFETY: see above; the autodoc outlives this method.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = Some(unsafe { &*autodoc });
        self.tf_low_pass_cutoff_inverse_nm.set_tooltip(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::LOW_PASS_CUTOFF_INVERSE_NM))
                .as_deref(),
        );
        self.ltf_view_skip_list.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::SKIP_VIEW_LIST_KEY))
                .as_deref(),
        );
        self.ltf_additional_view_sets.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::ADDITIONAL_VIEW_GROUPS_KEY))
                .as_deref(),
        );
        self.ltf_tilt_angle_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::TILT_ANGLE_GROUP_PARAMS_KEY))
                .as_deref(),
        );
        self.ltf_tilt_angle_groups.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::TILT_ANGLE_GROUPS_KEY))
                .as_deref(),
        );
        self.ltf_magnification_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(beadtrack_param::MAGNIFICATION_GROUP_PARAMS_KEY),
            )
            .as_deref(),
        );
        self.ltf_magnification_groups.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::MAGNIFICATION_GROUPS_KEY))
                .as_deref(),
        );
        self.ltf_n_min_views.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::N_MIN_VIEWS_KEY)).as_deref(),
        );
        self.ltf_bead_diameter.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::BEAD_DIAMETER_KEY))
                .as_deref(),
        );
        self.cb_light_beads.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::LIGHT_BEADS_KEY)).as_deref(),
        );
        self.cb_fill_gaps.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::FILL_GAPS_KEY)).as_deref(),
        );
        self.ltf_max_gap.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::MAX_GAP_KEY)).as_deref(),
        );
        self.ltf_min_tilt_range_to_find_axis.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(beadtrack_param::MIN_TILT_RANGE_TO_FIND_AXIS_KEY),
            )
            .as_deref(),
        );
        self.ltf_min_tilt_range_to_find_angle.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(beadtrack_param::MIN_TILT_RANGE_TO_FIND_ANGLES_KEY),
            )
            .as_deref(),
        );
        self.ltf_search_box_pixels.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::SEARCH_BOX_PIXELS_KEY))
                .as_deref(),
        );
        self.ltf_max_fiducials_avg.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::MAX_FIDUCIALS_AVG_KEY))
                .as_deref(),
        );
        self.ltf_fiducial_extrapolation_params.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(beadtrack_param::FIDUCIAL_EXTRAPOLATION_PARAMS_KEY),
            )
            .as_deref(),
        );
        self.ltf_rescue_attempt_params.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::RESCUE_ATTEMPT_PARAMS_KEY))
                .as_deref(),
        );
        self.ltf_min_rescue_distance.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::MIN_RESCUE_DISTANCE_KEY))
                .as_deref(),
        );
        self.ltf_rescue_relaxtion_params.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(beadtrack_param::RESCUE_RELAXATION_PARAMS_KEY),
            )
            .as_deref(),
        );
        self.ltf_residual_distance_limit.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::RESIDUAL_DISTANCE_LIMIT_KEY))
                .as_deref(),
        );
        self.ltf_density_relaxation_post_fit.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(beadtrack_param::DENSITY_RELAXATION_POST_FIT_KEY),
            )
            .as_deref(),
        );
        self.ltf_max_rescue_distance.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::MAX_RESCUE_DISTANCE_KEY))
                .as_deref(),
        );
        self.ltf_mean_resid_change_limits.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(beadtrack_param::MEAN_RESID_CHANGE_LIMITS_KEY),
            )
            .as_deref(),
        );
        self.ltf_deletion_params.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::DELETION_PARAMS_KEY))
                .as_deref(),
        );

        self.cb_local_area_tracking.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::LOCAL_AREA_TRACKING_KEY))
                .as_deref(),
        );
        self.ltf_local_area_target_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::LOCAL_AREA_TARGET_SIZE_KEY))
                .as_deref(),
        );
        self.ltf_min_beads_in_area.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::MIN_BEADS_IN_AREA_KEY))
                .as_deref(),
        );
        self.ltf_min_overlap_beads.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::MIN_OVERLAP_BEADS_KEY))
                .as_deref(),
        );
        self.ltf_max_views_in_align.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::MAX_VIEWS_IN_ALIGN_KEY))
                .as_deref(),
        );
        self.ltf_rounds_of_tracking.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::ROUNDS_OF_TRACKING_KEY))
                .as_deref(),
        );
        self.cb_sobel_filter_centering.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(beadtrack_param::SOBEL_FILTER_CENTERING_KEY))
                .as_deref(),
        );
        self.ltf_scalable_sigma_for_sobel.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(beadtrack_param::SCALABLE_SIGMA_FOR_SOBEL_KEY),
            )
            .as_deref(),
        );
        self.btn_track.set_tool_tip_text(Some(
            "Run Beadtrack to produce fiducial model from seed model.",
        ));
        self.btn_fix_model
            .set_tool_tip_text(Some("Load fiducial model into 3dmod."));
        self.btn_use_model.set_tool_tip_text(Some(&format!(
            "{}{}",
            "Turn the output of Beadtrack (fiducial model) into a new seed model and then track.  ",
            "Your original seed model will be moved into an _orig.seed file."
        )));
    }
}

/// Java `Expandable`.
impl Expandable for BeadtrackPanel {
    /// Java `expand(GlobalExpandButton)` (empty in the Java).
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.expert_parameters_header.equals_open_close(button) {
            self.pnl_expert_parameters_body
                .set_visible(button.is_expanded());
        } else if self.header.equals_open_close(button) {
            self.panel_beadtrack_body.set_visible(button.is_expanded());
        } else if self.header.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
        });
    }
}

/// Java `BeadTrackDisplay`.
impl BeadTrackDisplay for BeadtrackPanel {
    /// Java `getParameters(BeadtrackParam, boolean) throws
    /// FortranInputSyntaxException, InvalidEtomoNumberException`: get the field
    /// values from the panel filling in the BeadtrackParam object.
    fn get_parameters(
        &self,
        beadtrack_params: &mut BeadtrackParam,
        do_validation: bool,
    ) -> Result<bool, BeadTrackDisplayException> {
        /// The exceptions the Java `try` blocks of this method see.
        enum Thrown {
            FieldValidationFailed(FieldValidationFailedException),
            FortranInputSyntax(FortranInputSyntaxException),
            InvalidEtomoNumber(InvalidEtomoNumberException),
        }
        let manager = self.manager;
        let axis_id = self.axis_id;
        // Java outer try { ... } catch (FieldValidationFailedException e) {
        // return false; }
        let outer = (|| -> Result<(), Thrown> {
            beadtrack_params.set_skip_views(
                self.ltf_view_skip_list
                    .get_text_boolean(do_validation)
                    .map_err(Thrown::FieldValidationFailed)?
                    .as_deref(),
            );
            beadtrack_params.set_additional_view_groups(
                self.ltf_additional_view_sets
                    .get_text_boolean(do_validation)
                    .map_err(Thrown::FieldValidationFailed)?
                    .as_deref(),
            );
            beadtrack_params.set_light_beads(self.cb_light_beads.is_selected());
            beadtrack_params
                .set_sobel_filter_centering(self.cb_sobel_filter_centering.is_selected());
            beadtrack_params.set_scalable_sigma_for_sobel(
                self.ltf_scalable_sigma_for_sobel
                    .get_text_boolean(do_validation)
                    .map_err(Thrown::FieldValidationFailed)?
                    .as_deref(),
            );
            beadtrack_params.set_low_pass_cutoff_inverse_nm(
                self.tf_low_pass_cutoff_inverse_nm
                    .get_text_boolean(do_validation)
                    .map_err(Thrown::FieldValidationFailed)?
                    .as_deref(),
            );
            beadtrack_params.set_images_are_binned(
                UIExpertUtilities::INSTANCE.get_stack_binning_base_manager_axis_id_file_type(
                    manager,
                    axis_id,
                    &file_type::CLASS.prealigned_stack,
                ),
            );

            // Beadtrack only fields.
            beadtrack_params.set_fill_gaps(self.cb_fill_gaps.is_selected());

            let error_title = "FieldInterface Error";
            let mut bad_parameter = String::new();
            // handle field that throw FortranInputSyntaxException
            // Java middle try { ... } catch (FortranInputSyntaxException except).
            let middle = (|| -> Result<(), Thrown> {
                bad_parameter = self.ltf_tilt_angle_groups.get_label();
                beadtrack_params
                    .set_tilt_angle_groups(
                        self.ltf_tilt_angle_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_magnification_groups.get_label();
                beadtrack_params
                    .set_magnification_groups(
                        self.ltf_magnification_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_search_box_pixels.get_label();
                beadtrack_params
                    .set_search_box_pixels(
                        self.ltf_search_box_pixels
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_fiducial_extrapolation_params.get_label();
                beadtrack_params
                    .set_fiducial_extrapolation_params(
                        self.ltf_fiducial_extrapolation_params
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_rescue_attempt_params.get_label();
                beadtrack_params
                    .set_rescue_attempt_params(
                        self.ltf_rescue_attempt_params
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_rescue_relaxtion_params.get_label();
                beadtrack_params
                    .set_rescue_relaxation_params(
                        self.ltf_rescue_relaxtion_params
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_mean_resid_change_limits.get_label();
                beadtrack_params
                    .set_mean_resid_change_limits(
                        self.ltf_mean_resid_change_limits
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_deletion_params.get_label();
                beadtrack_params
                    .set_deletion_params(
                        self.ltf_deletion_params
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                // handle fields that display their own messages and throw
                // InvalidEtomoNumberException
                // Java inner try { ... } catch (InvalidEtomoNumberException e)
                // { throw e; } -- a rethrow, so the `?`s below propagate as is.
                {
                    // Java: `if (errorMessage != null) { UIHarness.INSTANCE
                    // .openMessageDialog(manager, errorMessage, errorTitle,
                    // axisID); throw new InvalidEtomoNumberException(errorMessage); }`
                    // after each validate.  Written out at every site, as in Java.
                    bad_parameter = self.ltf_tilt_angle_group_size.get_label();
                    let text = self
                        .ltf_tilt_angle_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_tilt_default_grouping(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_magnification_group_size.get_label();
                    let text = self
                        .ltf_magnification_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_mag_default_grouping(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_n_min_views.get_label();
                    let text = self
                        .ltf_n_min_views
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_min_views_for_tiltalign(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_max_gap.get_label();
                    let text = self
                        .ltf_max_gap
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_max_gap_size(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_max_fiducials_avg.get_label();
                    let text = self
                        .ltf_max_fiducials_avg
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_max_beads_to_average(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_min_rescue_distance.get_label();
                    let text = self
                        .ltf_min_rescue_distance
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_distance_rescue_criterion(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_residual_distance_limit.get_label();
                    let text = self
                        .ltf_residual_distance_limit
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_post_fit_rescue_residual(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_density_relaxation_post_fit.get_label();
                    let text = self
                        .ltf_density_relaxation_post_fit
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_density_relaxation_post_fit(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_max_rescue_distance.get_label();
                    let text = self
                        .ltf_max_rescue_distance
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_max_rescue_distance(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_min_tilt_range_to_find_axis.get_label();
                    let text = self
                        .ltf_min_tilt_range_to_find_axis
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_min_tilt_range_to_find_axis(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_min_tilt_range_to_find_angle.get_label();
                    let text = self
                        .ltf_min_tilt_range_to_find_angle
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_min_tilt_range_to_find_angles(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_bead_diameter.get_label();
                    let text = self
                        .ltf_bead_diameter
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_bead_diameter(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    // Java string concatenation renders null as "null".
                    bad_parameter = self
                        .cb_local_area_tracking
                        .get_text_void()
                        .unwrap_or_else(|| "null".to_string());
                    let error_message = beadtrack_params
                        .set_local_area_tracking(self.cb_local_area_tracking.is_selected())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    // Upstream bug fixed (BeadtrackPanel.java:602, 612, 622,
                    // 632, 642): Java sets `badParameter =
                    // ltfX.getText(doValidation)` -- the field's value -- for
                    // these five fields, where every other site uses
                    // `getLabel()`, so the validation message and the
                    // FortranInputSyntaxException prefix named the entered
                    // value instead of the field.  We use `getLabel()`.  (The
                    // extra getText call could only throw the same
                    // FieldValidationFailedException the next line throws.)
                    bad_parameter = self.ltf_local_area_target_size.get_label();
                    let text = self
                        .ltf_local_area_target_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_local_area_target_size(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_min_beads_in_area.get_label();
                    let text = self
                        .ltf_min_beads_in_area
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_min_beads_in_area(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_min_overlap_beads.get_label();
                    let text = self
                        .ltf_min_overlap_beads
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_min_overlap_beads(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_max_views_in_align.get_label();
                    let text = self
                        .ltf_max_views_in_align
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_max_views_in_align(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }

                    bad_parameter = self.ltf_rounds_of_tracking.get_label();
                    let text = self
                        .ltf_rounds_of_tracking
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?;
                    let error_message = beadtrack_params
                        .set_rounds_of_tracking(text.as_deref())
                        .validate(Some(bad_parameter.as_str()));
                    if let Some(error_message) = error_message {
                        ui_harness::INSTANCE.with(|harness| {
                            harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(manager),
                                &error_message,
                                error_title,
                                Some(axis_id),
                            )
                        });
                        return Err(Thrown::InvalidEtomoNumber(
                            InvalidEtomoNumberException::new(&error_message),
                        ));
                    }
                }
                Ok(())
            })();
            match middle {
                // catch (FortranInputSyntaxException except) {
                //   String message = badParameter + " " + except.getMessage();
                //   throw new FortranInputSyntaxException(message); }
                Err(Thrown::FortranInputSyntax(except)) => {
                    let message = format!(
                        "{} {}",
                        bad_parameter,
                        except.get_message().unwrap_or("null")
                    );
                    Err(Thrown::FortranInputSyntax(
                        FortranInputSyntaxException::new(&message),
                    ))
                }
                other => other,
            }
        })();
        match outer {
            Ok(()) => Ok(true),
            // catch (FieldValidationFailedException e) { return false; }
            Err(Thrown::FieldValidationFailed(_)) => Ok(false),
            Err(Thrown::FortranInputSyntax(except)) => Err(
                BeadTrackDisplayException::FortranInputSyntaxException(except),
            ),
            Err(Thrown::InvalidEtomoNumber(except)) => Err(
                BeadTrackDisplayException::InvalidEtomoNumberException(except),
            ),
        }
    }
}

/// Java `Run3dmodButtonContainer`.
impl Run3dmodButtonContainer for BeadtrackPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // Java try { ... } catch (FieldValidationFailedException e) { return; }
        if Some(command) == self.btn_track.get_action_command().as_deref() {
            let display: Rc<dyn ProcessResultDisplay> = self.btn_track.clone();
            self.manager.fiducial_model_track(
                self.axis_id,
                Some(display),
                None,
                self.dialog_type,
                self,
            );
        } else if Some(command) == self.btn_use_model.get_action_command().as_deref() {
            if self.manager.make_fiducial_model_seed_model(self.axis_id) {
                let display: Rc<dyn ProcessResultDisplay> = self.btn_use_model.clone();
                self.manager.fiducial_model_track(
                    self.axis_id,
                    Some(display),
                    None,
                    self.dialog_type,
                    self,
                );
            }
        } else if Some(command) == self.cb_local_area_tracking.get_text_void().as_deref()
            || Some(command) == self.cb_sobel_filter_centering.get_text_void().as_deref()
        {
            self.set_enabled();
        } else if Some(command) == self.btn_fix_model.get_action_command().as_deref() {
            // Validate skipList
            let skip_list_text = match self.ltf_view_skip_list.get_text_boolean(true) {
                Ok(text) => text,
                Err(_) => return,
            };
            let mut skip_list: Option<String> = Some(
                const_etomo_number::java_lang_string_trim(
                    skip_list_text.as_deref().unwrap_or_default(),
                )
                .to_string(),
            );
            if skip_list.as_deref().is_some_and(|s| !s.is_empty()) {
                // skipList.matches(".*\\s+.*"): Java `\s` is [ \t\n\x0B\f\r].
                if skip_list
                    .as_deref()
                    .unwrap()
                    .chars()
                    .any(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
                {
                    ui_harness::INSTANCE.with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.manager),
                            &format!("{} cannot contain embedded spaces.", VIEW_SKIP_LIST_LABEL),
                            "Entry Error",
                            Some(self.axis_id),
                        )
                    });
                    return;
                }
            } else {
                skip_list = None;
            }
            let display: Rc<dyn ProcessResultDisplay> = self.btn_fix_model.clone();
            // The manager takes the options by value; a Java null is the
            // default (no options set).
            self.manager.imod_fix_fiducials(
                self.axis_id,
                run_3dmod_menu_options.unwrap_or_default(),
                Some(display),
                BeadFixerMode::GapMode,
                skip_list.as_deref(),
            );
        }
    }
}
