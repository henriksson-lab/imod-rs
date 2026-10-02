//! `IMOD/Etomo/src/etomo/ui/swing/TiltxcorrPanel.java`.
//!
//! The tiltxcorr panel, used by the coarse alignment dialog (cross
//! correlation) and the fiducial model dialog (patch tracking).

use crate::imod::etomo::ui::field::Field;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::check_text_field::CheckTextField;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::process_display::ProcessDisplay;
use super::radio_button::RadioButton;
use super::radio_text_field::RadioTextField;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use super::spinner::Spinner;
use super::tiltxcorr_display::TiltXcorrDisplay;
use super::ui_expert_utilities::UIExpertUtilities;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::const_tiltxcorr_param::ConstTiltxcorrParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::imodchopconts_param::{self, ImodchopcontsParam};
use crate::imod::etomo::comscript::tiltxcorr_param::{self, TiltxcorrParam};
use crate::imod::etomo::jdk::MouseEvent;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java `final class TiltxcorrPanel implements Expandable, TiltXcorrDisplay,
/// Run3dmodButtonContainer, ContextMenu`.
pub struct TiltxcorrPanel {
    /// Java `this`, handed to the manager as the `TiltXcorrDisplay`.
    this: Weak<TiltxcorrPanel>,
    pnl_root: Rc<SpacedPanel>,
    pnl_body: Rc<JComponent>,
    pnl_advanced: Rc<JComponent>,
    pnl_advanced2: Rc<JComponent>,
    pnl_x_min_and_max: Rc<JComponent>,
    pnl_y_min_and_max: Rc<JComponent>,

    cb_exclude_central_peak: Rc<CheckBox>,

    ltf_test_output: Rc<LabeledTextField>,
    ltf_filter_sigma1: Rc<LabeledTextField>,
    ltf_filter_radius2: Rc<LabeledTextField>,
    ltf_filter_sigma2: Rc<LabeledTextField>,
    ltf_trim: Rc<LabeledTextField>,
    ltf_x_min: Rc<LabeledTextField>,
    ltf_x_max: Rc<LabeledTextField>,
    ltf_y_min: Rc<LabeledTextField>,
    ltf_y_max: Rc<LabeledTextField>,
    ltf_pad_percent: Rc<LabeledTextField>,
    ltf_taper_percent: Rc<LabeledTextField>,
    cb_cumulative_correlation: Rc<CheckBox>,
    cb_absolute_cosine_stretch: Rc<CheckBox>,
    cb_no_cosine_stretch: Rc<CheckBox>,
    ltf_view_range: Rc<LabeledTextField>,
    ltf_angle_offset: Rc<LabeledTextField>,
    /// Java `actionListener` (a `CrossCorrelationActionListener`).
    action_listener: ActionListener,
    ltf_skip_views: Rc<LabeledTextField>,

    // Patch tracking
    ltf_size_of_patches_x_and_y: Rc<LabeledTextField>,
    bg_patch_layout: Rc<ButtonGroup>,
    rtf_overlap_of_patches_x_and_y: Rc<RadioTextField>,
    rtf_number_of_patches_x_and_y: Rc<RadioTextField>,
    sp_iterate_correlations: Rc<Spinner>,
    ltf_shift_limits_x_and_y: Rc<LabeledTextField>,
    ctf_length_of_pieces_minimum_overlap: Rc<CheckTextField>,
    cb_boundary_model: Rc<CheckBox>,
    btn_3dmod_boundary_model: Rc<Run3dmodButton>,
    bg_length_of_pieces: Rc<ButtonGroup>,
    rb_length_of_pieces_default: Rc<RadioButton>,
    rtf_length_of_pieces: Rc<RadioTextField>,

    axis_id: AxisID,
    dialog_type: DialogType,
    /// Java `btnTiltxcorr`, declared `MultiLineButton`; the factory builds a
    /// plain `MultiLineButton` for coarse alignment and a `Run3dmodButton` for
    /// patch tracking, and returns null for any other dialog type.
    btn_tiltxcorr: Option<ProcessResultDisplayHandle>,
    /// `btnTiltxcorr` when the factory built a plain `MultiLineButton`.
    btn_tiltxcorr_multi_line: Option<Rc<MultiLineButton>>,
    /// `btnTiltxcorr` when the factory built a `Run3dmodButton`.
    btn_tiltxcorr_run_3dmod: Option<Rc<Run3dmodButton>>,
    /// Java `btnImodchopconts`, declared `MultiLineButton`; the factory builds a
    /// `Run3dmodButton`.
    btn_imodchopconts: Rc<Run3dmodButton>,
    btn_3dmod_patch_tracking: Rc<Run3dmodButton>,
    header: Rc<PanelHeader>,
    application_manager: &'static ApplicationManager,
    panel_id: PanelId,
    /// Java `skipViews`, assigned nowhere but its declaration.
    skip_views: RefCell<Option<String>>,
    ctf_mag_changes: Option<Rc<CheckTextField>>,
    mag_changes_mode: bool,

    /// Java package-visible `contextMenu`.
    pub context_menu: Option<Weak<dyn ContextMenu>>,
}

impl TiltxcorrPanel {
    /// Java private constructor `TiltxcorrPanel(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton, PanelId, ContextMenu, boolean)`
    /// (TiltxcorrPanel.java:137-162).
    fn new(
        application_manager: &'static ApplicationManager,
        id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
        panel_id: PanelId,
        context_menu: Option<Weak<dyn ContextMenu>>,
        mag_changes_mode: bool,
    ) -> Rc<TiltxcorrPanel> {
        Rc::new_cyclic(|weak: &Weak<TiltxcorrPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = weak.clone();
            // Field initializers (TiltxcorrPanel.java:51-135).
            let pnl_root = SpacedPanel::get_instance_boolean(true);
            let pnl_body = JComponent::new_panel();
            let pnl_advanced = JComponent::new_panel();
            let pnl_advanced2 = JComponent::new_panel();
            let pnl_x_min_and_max = JComponent::new_panel();
            let pnl_y_min_and_max = JComponent::new_panel();
            let cb_exclude_central_peak =
                CheckBox::new_string(Some("Exclude central peak due to fixed pattern noise"));
            let ltf_test_output =
                LabeledTextField::new_field_type_string(FieldType::String, Some("Test output: "));
            let ltf_filter_sigma1 = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Low frequency rolloff sigma: "),
            );
            let ltf_filter_radius2 = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("High frequency cutoff radius: "),
            );
            let ltf_filter_sigma2 = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("High frequency rolloff sigma: "),
            );
            let ltf_trim = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("Pixels to trim (x,y): "),
            );
            let ltf_x_min =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("X axis min "));
            let ltf_x_max =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Max "));
            let ltf_y_min =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y axis min "));
            let ltf_y_max =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Max "));
            let ltf_pad_percent = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("Pixels to pad (x,y): "),
            );
            let ltf_taper_percent = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("Pixels to taper (x,y): "),
            );
            let cb_cumulative_correlation = CheckBox::new_string(Some("Cumulative correlation"));
            let cb_absolute_cosine_stretch = CheckBox::new_string(Some("Absolute Cosine Stretch"));
            let cb_no_cosine_stretch = CheckBox::new_string(Some("No Cosine Stretch"));
            let ltf_view_range = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("View range (start,end): "),
            );
            let ltf_angle_offset = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Tilt angle offset: "),
            );
            // Java `new CrossCorrelationActionListener(this)`; the static nested
            // class is at TiltxcorrPanel.java:738-749.
            let adaptee = weak.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            });
            let ltf_skip_views = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Views to skip: "),
            );

            // Patch tracking
            let ltf_size_of_patches_x_and_y = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("Size of patches (X,Y): "),
            );
            let bg_patch_layout = ButtonGroup::new();
            let rtf_overlap_of_patches_x_and_y =
                RadioTextField::get_instance_field_type_string_button_group(
                    FieldType::FloatingPointPair,
                    Some("Fractional overlap of patches (X,Y): "),
                    Some(&bg_patch_layout),
                );
            let rtf_number_of_patches_x_and_y =
                RadioTextField::get_instance_field_type_string_button_group(
                    FieldType::IntegerPair,
                    Some("Number of patches (X,Y): "),
                    Some(&bg_patch_layout),
                );
            let sp_iterate_correlations = Spinner::get_labeled_instance_string_int_int_int(
                Some("Iterations to increase subpixel accuracy: "),
                tiltxcorr_param::ITERATE_CORRELATIONS_DEFAULT,
                tiltxcorr_param::ITERATE_CORRELATIONS_MIN,
                tiltxcorr_param::ITERATE_CORRELATIONS_MAX,
            );
            let ltf_shift_limits_x_and_y = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("Limits on shifts from correlation (X,Y): "),
            );
            let ctf_length_of_pieces_minimum_overlap = CheckTextField::get_instance(
                FieldType::Integer,
                "Break contours into pieces with overlap: ",
            );
            let cb_boundary_model = CheckBox::new_string(Some("Use boundary model"));
            let btn_3dmod_boundary_model =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Create Boundary Model"),
                    Some(container.clone()),
                );
            let bg_length_of_pieces = ButtonGroup::new();
            let rb_length_of_pieces_default = RadioButton::new_string_button_group(
                Some("Use default length"),
                Some(&bg_length_of_pieces),
            );
            let rtf_length_of_pieces = RadioTextField::get_instance_field_type_string_button_group(
                FieldType::Integer,
                Some("Use length"),
                Some(&bg_length_of_pieces),
            );
            let btn_3dmod_patch_tracking =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Tracked Patches"),
                    Some(container.clone()),
                );

            // Constructor body (TiltxcorrPanel.java:140-161).
            let expandable: Weak<dyn Expandable> = weak.clone();
            let header =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Tiltxcorr"),
                    Some(expandable),
                    Some(dialog_type),
                    Some(global_advanced_button.clone()),
                );

            let factory = application_manager.get_process_result_display_factory(id);
            let btn_tiltxcorr = factory.get_tiltxcorr(dialog_type);
            // Java `(MultiLineButton) factory.getTiltxcorr(dialogType)`.
            let btn_tiltxcorr_run_3dmod = btn_tiltxcorr
                .clone()
                .and_then(|button| button.as_any_rc().downcast::<Run3dmodButton>().ok());
            let btn_tiltxcorr_multi_line = btn_tiltxcorr
                .clone()
                .and_then(|button| button.as_any_rc().downcast::<MultiLineButton>().ok());
            let btn_imodchopconts = factory.get_imodchopconts();
            let ctf_mag_changes;
            if panel_id == PanelId::PatchTracking {
                ctf_mag_changes = None;
                // Java `((Run3dmodButton) btnTiltxcorr)`: for the fiducial model
                // dialog the factory's tiltxcorr button is a Run3dmodButton.  A
                // null button would be a NullPointerException here
                // (TiltxcorrPanel.java:155); it is skipped.
                if let Some(button) = &btn_tiltxcorr_run_3dmod {
                    button.set_deferred_3dmod_button_deferred_3dmod_button(Some(
                        btn_3dmod_patch_tracking.clone() as Rc<dyn Deferred3dmodButton>,
                    ));
                    button.set_container(Some(container.clone()));
                }
            } else {
                ctf_mag_changes = Some(CheckTextField::get_instance(
                    FieldType::IntegerList,
                    "Find mag change at view(s):",
                ));
            }

            TiltxcorrPanel {
                this: weak.clone(),
                pnl_root,
                pnl_body,
                pnl_advanced,
                pnl_advanced2,
                pnl_x_min_and_max,
                pnl_y_min_and_max,
                cb_exclude_central_peak,
                ltf_test_output,
                ltf_filter_sigma1,
                ltf_filter_radius2,
                ltf_filter_sigma2,
                ltf_trim,
                ltf_x_min,
                ltf_x_max,
                ltf_y_min,
                ltf_y_max,
                ltf_pad_percent,
                ltf_taper_percent,
                cb_cumulative_correlation,
                cb_absolute_cosine_stretch,
                cb_no_cosine_stretch,
                ltf_view_range,
                ltf_angle_offset,
                action_listener,
                ltf_skip_views,
                ltf_size_of_patches_x_and_y,
                bg_patch_layout,
                rtf_overlap_of_patches_x_and_y,
                rtf_number_of_patches_x_and_y,
                sp_iterate_correlations,
                ltf_shift_limits_x_and_y,
                ctf_length_of_pieces_minimum_overlap,
                cb_boundary_model,
                btn_3dmod_boundary_model,
                bg_length_of_pieces,
                rb_length_of_pieces_default,
                rtf_length_of_pieces,
                axis_id: id,
                dialog_type,
                btn_tiltxcorr,
                btn_tiltxcorr_multi_line,
                btn_tiltxcorr_run_3dmod,
                btn_imodchopconts,
                btn_3dmod_patch_tracking,
                header,
                application_manager,
                panel_id,
                skip_views: RefCell::new(None),
                ctf_mag_changes,
                mag_changes_mode,
                context_menu,
            }
        })
    }

    /// Java `static getCrossCorrelationInstance(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton, ContextMenu, boolean)`
    /// (TiltxcorrPanel.java:164-174).
    pub fn get_cross_correlation_instance(
        application_manager: &'static ApplicationManager,
        id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
        context_menu: Option<Weak<dyn ContextMenu>>,
        mag_changes_mode: bool,
    ) -> Rc<TiltxcorrPanel> {
        let instance = TiltxcorrPanel::new(
            application_manager,
            id,
            dialog_type,
            global_advanced_button,
            PanelId::CrossCorrelation,
            context_menu,
            mag_changes_mode,
        );
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java `static getPatchTrackingInstance(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton)` (TiltxcorrPanel.java:176-185).
    pub fn get_patch_tracking_instance(
        application_manager: &'static ApplicationManager,
        id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<TiltxcorrPanel> {
        let instance = TiltxcorrPanel::new(
            application_manager,
            id,
            dialog_type,
            global_advanced_button,
            PanelId::PatchTracking,
            None,
            false,
        );
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()` (TiltxcorrPanel.java:187-327).
    fn create_panel(&self) {
        // initialize
        self.rb_length_of_pieces_default.set_selected_boolean(true);
        self.rb_length_of_pieces_default.set_text(Some(
            &("Use default length (".to_string()
                + &ImodchopcontsParam::get_length_of_pieces_default(
                    self.application_manager,
                    self.axis_id,
                    &file_type::CLASS.prealigned_stack,
                )
                + ")"),
        ));
        // root panel
        // Swing layout: pnlRoot.setBoxLayout(BoxLayout.Y_AXIS).
        // Construct the min and max subpanels
        // Swing layout: pnlXMinAndMax BoxLayout X_AXIS.  Each
        // `UIUtilities.addWithXSpace(p, c)` is `p.add(c)` plus a rigid area
        // (UIUtilities.java:471-474); likewise addWithYSpace (:481-484).
        self.pnl_x_min_and_max.add(&self.ltf_x_min.get_container());
        self.pnl_x_min_and_max.add(&self.ltf_x_max.get_container());

        // Swing layout: pnlYMinAndMax BoxLayout X_AXIS.
        self.pnl_y_min_and_max.add(&self.ltf_y_min.get_container());
        self.pnl_y_min_and_max.add(&self.ltf_y_max.get_container());
        // advanced panel
        // Swing layout: pnlAdvanced and pnlAdvanced2 BoxLayout Y_AXIS.
        if self.panel_id == PanelId::CrossCorrelation {
            // Construct the advanced panel
            self.pnl_advanced
                .add(&self.ltf_angle_offset.get_container());
            self.pnl_advanced
                .add(&self.ltf_filter_sigma1.get_container());
            self.pnl_advanced
                .add(&self.ltf_filter_radius2.get_container());
            self.pnl_advanced
                .add(&self.ltf_filter_sigma2.get_container());
            self.pnl_advanced.add(&self.ltf_trim.get_container());
            self.pnl_advanced.add(&self.pnl_x_min_and_max);
            self.pnl_advanced.add(&self.pnl_y_min_and_max);
            self.pnl_advanced.add(&self.ltf_pad_percent.get_container());
            self.pnl_advanced
                .add(&self.ltf_taper_percent.get_container());

            self.pnl_advanced2
                .add(&self.cb_cumulative_correlation.get_component());
            self.pnl_advanced2
                .add(&self.cb_absolute_cosine_stretch.get_component());
            self.pnl_advanced2
                .add(&self.cb_no_cosine_stretch.get_component());
            self.pnl_advanced2
                .add(&self.cb_exclude_central_peak.get_component());
            self.pnl_advanced2
                .add(&self.ltf_test_output.get_container());
            self.pnl_advanced2.add(&self.ltf_view_range.get_container());
            self.pnl_advanced2.add(&self.ltf_skip_views.get_container());

            // Swing layout: pnlBody BoxLayout Y_AXIS; rigid area x0_y5.
            self.pnl_body.add(&self.pnl_advanced);
            if let Some(ctf_mag_changes) = &self.ctf_mag_changes {
                self.pnl_body.add(&ctf_mag_changes.get_component());
            }
            self.pnl_body.add(&self.pnl_advanced2);
            // Swing layout: rigid area x0_y5.
            if let Some(btn_tiltxcorr) = self.btn_tiltxcorr_button() {
                self.pnl_body.add(&btn_tiltxcorr.get_component());
            }
            // Swing layout: rigid area x0_y5; center align pnlBody's components,
            // left align pnlAdvanced's and pnlAdvanced2's.

            // Swing layout: pnlRoot untitled etched border.
            self.pnl_root.add_panel_header(&self.header);
            self.pnl_root.add_j_panel(&self.pnl_body);
        } else if self.panel_id == PanelId::PatchTracking {
            // initialize
            self.rtf_overlap_of_patches_x_and_y
                .set_text_string(Some(tiltxcorr_param::OVERLAP_OF_PATCHES_X_AND_Y_DEFAULT));
            // Swing layout: ctfLengthOfPiecesMinimumOverlap.setTextPreferredWidth(30).
            self.rtf_overlap_of_patches_x_and_y
                .set_selected_boolean(true);
            self.ltf_trim
                .set_text_string(Some(&*TiltxcorrParam::get_borders_in_x_and_y_default(
                    self.application_manager,
                    self.axis_id,
                    &file_type::CLASS.prealigned_stack,
                )));
            self.ltf_filter_sigma1
                .set_text_string(Some(tiltxcorr_param::FILTER_SIGMA_1_DEFAULT));
            self.ltf_filter_radius2
                .set_text_string(Some(tiltxcorr_param::FILTER_RADIUS_2_DEFAULT));
            self.ltf_filter_sigma2
                .set_text_string(Some(tiltxcorr_param::FILTER_SIGMA_2_DEFAULT));
            // local panels
            let pnl_patch_layout = JComponent::new_panel();
            let pnl_boundary_model = JComponent::new_panel();
            let pnl_buttons = JComponent::new_panel();
            let pnl_imodchopconts = JComponent::new_panel();
            let pnl_length_of_pieces = JComponent::new_panel();
            // root panel
            self.pnl_root
                .set_border(&EtchedBorder::new(Some("Patch Tracking")).get_border());
            self.pnl_root
                .add_container(&self.ltf_size_of_patches_x_and_y.get_container());
            self.pnl_root.add_j_panel(&pnl_patch_layout);
            self.pnl_root.add_j_panel(&pnl_boundary_model);
            let pnl_iterate_correlations = JComponent::new_panel();
            // Swing layout: pnlIterateCorrelations BoxLayout X_AXIS, center
            // aligned, horizontal glue after the spinner.
            pnl_iterate_correlations.add(&self.sp_iterate_correlations.get_container());
            self.pnl_root.add_j_panel(&pnl_iterate_correlations);
            self.pnl_root
                .add_container(&self.ltf_shift_limits_x_and_y.get_container());
            self.pnl_root.add_component(
                &self
                    .ctf_length_of_pieces_minimum_overlap
                    .get_root_component(),
            );
            self.pnl_root.add_j_panel(&pnl_length_of_pieces);
            self.pnl_root.add_labeled_text_field(&self.ltf_angle_offset);
            self.pnl_root.add_j_panel(&self.pnl_advanced2);
            self.pnl_root.add_labeled_text_field(&self.ltf_trim);
            self.pnl_root.add_j_panel(&self.pnl_x_min_and_max);
            self.pnl_root.add_j_panel(&self.pnl_y_min_and_max);
            self.pnl_root.add_j_panel(&self.pnl_advanced);
            self.pnl_root.add_j_panel(&pnl_buttons);
            self.pnl_root.add_j_panel(&pnl_imodchopconts);
            // patch layout panel
            // Swing layout: pnlPatchLayout BoxLayout Y_AXIS.
            // Java `pnlPatchLayout.setBorder(new EtchedBorder("Patch Layout").getBorder())`.
            pnl_patch_layout.set_border_title(Some("Patch Layout"));
            pnl_patch_layout.add(&self.rtf_overlap_of_patches_x_and_y.get_container());
            pnl_patch_layout.add(&self.rtf_number_of_patches_x_and_y.get_container());
            // boundary model panel
            // Swing layout: pnlBoundaryModel BoxLayout X_AXIS with glue.
            pnl_boundary_model.add(&self.cb_boundary_model.get_component());
            pnl_boundary_model.add(&self.btn_3dmod_boundary_model.get_component());
            // LengthOfPieces
            // Swing layout: pnlLengthOfPieces BoxLayout X_AXIS; rigid area x10_y0.
            pnl_length_of_pieces.add(&self.rb_length_of_pieces_default.get_component());
            pnl_length_of_pieces.add(&self.rtf_length_of_pieces.get_container());
            // advanced 2 panel
            self.pnl_advanced2
                .add(&self.ltf_filter_sigma1.get_container());
            self.pnl_advanced2
                .add(&self.ltf_filter_radius2.get_container());
            self.pnl_advanced2
                .add(&self.ltf_filter_sigma2.get_container());
            // advanced panel
            self.pnl_advanced.add(&self.ltf_pad_percent.get_container());
            self.pnl_advanced
                .add(&self.ltf_taper_percent.get_container());
            self.pnl_advanced.add(&self.ltf_test_output.get_container());
            self.pnl_advanced.add(&self.ltf_view_range.get_container());
            self.pnl_advanced.add(&self.ltf_skip_views.get_container());
            // button panel
            // Swing layout: pnlButtons BoxLayout X_AXIS with glue.
            if let Some(btn_tiltxcorr) = self.btn_tiltxcorr_button() {
                pnl_buttons.add(&btn_tiltxcorr.get_component());
            }
            pnl_buttons.add(&self.btn_3dmod_patch_tracking.get_component());
            // Imodchopconts
            // Swing layout: pnlImodchopconts BoxLayout X_AXIS with glue.
            pnl_imodchopconts.add(&self.btn_imodchopconts.get_component());
            self.update_panel();
        }
    }

    /// Java private `addListeners()` (TiltxcorrPanel.java:329-344).
    fn add_listeners(&self) {
        // `pnlRoot.addMouseListener(new GenericMouseAdapter(this))`: mouse events
        // are not modelled.
        let listener = &self.action_listener;
        self.cb_cumulative_correlation
            .add_action_listener(Some(listener.clone()));
        self.cb_no_cosine_stretch
            .add_action_listener(Some(listener.clone()));
        if let Some(btn_tiltxcorr) = self.btn_tiltxcorr_button() {
            btn_tiltxcorr.add_action_listener(listener.clone());
        }
        self.btn_3dmod_patch_tracking
            .add_action_listener(listener.clone());
        self.cb_boundary_model
            .add_action_listener(Some(listener.clone()));
        self.btn_3dmod_boundary_model
            .add_action_listener(listener.clone());
        if let Some(ctf_mag_changes) = &self.ctf_mag_changes {
            ctf_mag_changes.add_action_listener(listener.clone());
        }
        self.btn_imodchopconts.add_action_listener(listener.clone());
        self.ctf_length_of_pieces_minimum_overlap
            .add_action_listener(listener.clone());
    }

    /// Java field `btnTiltxcorr` as its declared type `MultiLineButton`.  The
    /// factory's button is one of two Rust types; a `Run3dmodButton` is a
    /// `MultiLineButton` through its `base`.
    fn btn_tiltxcorr_button(&self) -> Option<&MultiLineButton> {
        if let Some(button) = &self.btn_tiltxcorr_run_3dmod {
            return Some(&button.base);
        }
        self.btn_tiltxcorr_multi_line.as_deref()
    }

    /// Java `done()` (TiltxcorrPanel.java:372-375).
    pub fn done(&self) {
        // Java dereferences btnTiltxcorr unconditionally
        // (TiltxcorrPanel.java:373), a NullPointerException when the factory
        // returned null; fixed in translation by skipping a null button, as the
        // rest of this class does.
        if let Some(btn_tiltxcorr) = self.btn_tiltxcorr_button() {
            btn_tiltxcorr.remove_action_listener(&self.action_listener);
        }
        self.btn_imodchopconts
            .remove_action_listener(&self.action_listener);
    }

    /// Java `updateAdvanced(boolean)` (TiltxcorrPanel.java:377-388).
    pub fn update_advanced(&self, state: bool) {
        self.pnl_advanced.set_visible(state);
        self.pnl_advanced2.set_visible(state);
        if self.panel_id == PanelId::PatchTracking {
            self.ltf_angle_offset.set_visible(state);
            self.ltf_shift_limits_x_and_y.set_visible(state);
        }
        // If magChangesMode is true, then the mag changes fields are not advanced fields.
        if !self.mag_changes_mode {
            if let Some(ctf_mag_changes) = &self.ctf_mag_changes {
                ctf_mag_changes.set_visible(state);
            }
        }
    }

    /// Java `getPanel()` (TiltxcorrPanel.java:407-409).
    pub fn get_panel(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java `setParameters(ImodchopcontsParam)` (TiltxcorrPanel.java:411-427).
    pub fn set_parameters_imodchopconts_param(&self, param: &ImodchopcontsParam) {
        if self.panel_id == PanelId::PatchTracking {
            self.ctf_length_of_pieces_minimum_overlap
                .set_text_string(Some(&*param.get_minimum_overlap()));
            let length_of_pieces_set = !param.is_length_of_pieces_null();
            self.ctf_length_of_pieces_minimum_overlap
                .set_selected_boolean(length_of_pieces_set);
            if length_of_pieces_set {
                if param.is_length_of_pieces_default() {
                    self.rb_length_of_pieces_default.set_selected_boolean(true);
                } else {
                    self.rtf_length_of_pieces.set_selected_boolean(true);
                    self.rtf_length_of_pieces
                        .set_text_string(param.get_length_of_pieces().as_deref());
                }
            }
            self.update_panel();
        }
    }

    /// Java `setParameters(ConstTiltxcorrParam)` (TiltxcorrPanel.java:432-481).
    /// Set the field values for the panel from the ConstTiltxcorrParam object.
    pub fn set_parameters_const_tiltxcorr_param(
        &self,
        tilt_xcorr_params: &dyn ConstTiltxcorrParam,
    ) {
        self.ltf_angle_offset
            .set_text_string(Some(&*tilt_xcorr_params.get_angle_offset()));
        // Avoid overriding the default
        if tilt_xcorr_params.is_borders_in_x_and_y_set() {
            self.ltf_trim
                .set_text_string(Some(&*tilt_xcorr_params.get_borders_in_x_and_y()));
        }
        self.ltf_x_min
            .set_text_string(Some(&*tilt_xcorr_params.get_x_min_string()));
        self.ltf_x_max
            .set_text_string(Some(&*tilt_xcorr_params.get_x_max_string()));
        self.ltf_y_min
            .set_text_string(Some(&*tilt_xcorr_params.get_y_min_string()));
        self.ltf_y_max
            .set_text_string(Some(&*tilt_xcorr_params.get_y_max_string()));
        self.ltf_pad_percent
            .set_text_string(Some(&*tilt_xcorr_params.get_pads_in_x_and_y_string()));
        self.ltf_taper_percent
            .set_text_string(Some(&*tilt_xcorr_params.get_taper_percent_string()));
        self.ltf_test_output
            .set_text_string(tilt_xcorr_params.get_test_output().as_deref());
        self.ltf_view_range
            .set_text_string(Some(&*tilt_xcorr_params.get_starting_ending_views()));
        self.ltf_skip_views
            .set_text_string(Some(&*tilt_xcorr_params.get_skip_views()));
        if tilt_xcorr_params.is_filter_sigma1_set() {
            self.ltf_filter_sigma1
                .set_text_string(Some(&*tilt_xcorr_params.get_filter_sigma1_string()));
        }
        if tilt_xcorr_params.is_filter_radius2_set() {
            self.ltf_filter_radius2
                .set_text_string(Some(&*tilt_xcorr_params.get_filter_radius2_string()));
        }
        if tilt_xcorr_params.is_filter_sigma2_set() {
            self.ltf_filter_sigma2
                .set_text_string(Some(&*tilt_xcorr_params.get_filter_sigma2_string()));
        }
        if self.panel_id == PanelId::CrossCorrelation {
            self.cb_exclude_central_peak
                .set_selected_boolean(tilt_xcorr_params.get_exclude_central_peak());
            self.cb_cumulative_correlation
                .set_selected_boolean(tilt_xcorr_params.is_cumulative_correlation());
            self.cb_absolute_cosine_stretch
                .set_selected_boolean(tilt_xcorr_params.is_absolute_cosine_stretch());
            self.cb_no_cosine_stretch
                .set_selected_boolean(tilt_xcorr_params.is_no_cosine_stretch());
        } else if self.panel_id == PanelId::PatchTracking {
            self.ltf_size_of_patches_x_and_y
                .set_text_string(Some(&*tilt_xcorr_params.get_size_of_patches_x_and_y()));
            if tilt_xcorr_params.is_overlap_of_patches_x_and_y_set() {
                self.rtf_overlap_of_patches_x_and_y
                    .set_selected_boolean(true);
                self.rtf_overlap_of_patches_x_and_y
                    .set_text_string(Some(&*tilt_xcorr_params.get_overlap_of_patches_x_and_y()));
            }
            if tilt_xcorr_params.is_number_of_patches_x_and_y_set() {
                self.rtf_number_of_patches_x_and_y
                    .set_selected_boolean(true);
                self.rtf_number_of_patches_x_and_y
                    .set_text_string(Some(&*tilt_xcorr_params.get_number_of_patches_x_and_y()));
            }
            self.sp_iterate_correlations
                .set_value_int(tilt_xcorr_params.get_iterate_correlations());
            self.ltf_shift_limits_x_and_y
                .set_text_string(Some(&*tilt_xcorr_params.get_shift_limits_x_and_y()));
            self.cb_boundary_model
                .set_selected_boolean(tilt_xcorr_params.is_boundary_model_set());
        }
        if let Some(ctf_mag_changes) = &self.ctf_mag_changes {
            ctf_mag_changes.set_selected_boolean(tilt_xcorr_params.is_search_mag_changes());
            ctf_mag_changes.set_text_string(Some(&*tilt_xcorr_params.get_views_with_mag_changes()));
        }
        self.update_panel();
    }

    /// Java `setParameters(ConstMetaData)` (TiltxcorrPanel.java:487-500).  Load
    /// parameters which can be inactivated before loading from TiltxcorrParam.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        if self.panel_id == PanelId::PatchTracking {
            // Don't override defaults unless there is a value in meta data
            if meta_data.is_track_overlap_of_patches_x_and_y_set(self.axis_id) {
                self.rtf_overlap_of_patches_x_and_y.set_text_string(Some(
                    &meta_data.get_track_overlap_of_patches_x_and_y(self.axis_id),
                ));
            }
            self.rtf_number_of_patches_x_and_y.set_text_string(Some(
                &meta_data.get_track_number_of_patches_x_and_y(self.axis_id),
            ));
            // Backwards compatibility
            if meta_data.is_track_length_and_overlap_set(self.axis_id) {
                self.ctf_length_of_pieces_minimum_overlap
                    .set_text_string(Some(&*meta_data.get_minimum_overlap(self.axis_id)));
            }
            self.rtf_length_of_pieces
                .set_text_string(Some(&*meta_data.get_length_of_pieces(self.axis_id)));
        }
    }

    /// Java `getParameters(MetaData)` (TiltxcorrPanel.java:506-515).  Save
    /// parameters which can be inactivated to meta data.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        if self.panel_id == PanelId::PatchTracking {
            meta_data.set_track_overlap_of_patches_x_and_y(
                self.axis_id,
                self.rtf_overlap_of_patches_x_and_y
                    .get_text_void()
                    .as_deref(),
            );
            meta_data.set_track_number_of_patches_x_and_y(
                self.axis_id,
                self.rtf_number_of_patches_x_and_y
                    .get_text_void()
                    .as_deref(),
            );
            meta_data.set_length_of_pieces(
                self.axis_id,
                self.rtf_length_of_pieces.get_text_void().as_deref(),
            );
            // MinimumOverlap does not have to be saved because it can be placed in the
            // comscript even when it is disabled. It is in meta data for backwards
            // compatibility, and is loaded from trackLengthAndOverlap.
        }
    }

    /// Java `setParameters(BaseScreenState)` (TiltxcorrPanel.java:517-521).
    pub fn set_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        // btnCrossCorrelate.setButtonState(screenState
        // .getButtonState(btnCrossCorrelate.getButtonStateKey()));
        self.header
            .set_button_states_base_screen_state(Some(screen_state));
    }

    /// Java `getParameters(BaseScreenState)` (TiltxcorrPanel.java:523-525).
    pub fn get_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.header.get_button_states(Some(screen_state));
    }

    /// Java `setVisible(boolean)` (TiltxcorrPanel.java:670-672).
    pub fn set_visible(&self, state: bool) {
        self.pnl_root.set_visible(state);
    }

    /// Java `updatePanel()` (TiltxcorrPanel.java:674-692).
    pub fn update_panel(&self) {
        if self.cb_cumulative_correlation.is_selected() && !self.cb_no_cosine_stretch.is_selected()
        {
            self.cb_absolute_cosine_stretch.set_enabled(true);
        } else {
            self.cb_absolute_cosine_stretch.set_selected_boolean(false);
            self.cb_absolute_cosine_stretch.set_enabled(false);
        }
        self.btn_3dmod_boundary_model
            .set_enabled(self.cb_boundary_model.is_selected());
        if let Some(ctf_mag_changes) = &self.ctf_mag_changes {
            self.cb_cumulative_correlation
                .set_enabled(!ctf_mag_changes.is_selected() || !ctf_mag_changes.is_enabled());
            ctf_mag_changes.set_enabled(
                !self.cb_cumulative_correlation.is_selected()
                    || !self.cb_cumulative_correlation.is_enabled(),
            );
        }
        let enable = self.ctf_length_of_pieces_minimum_overlap.is_selected();
        self.rb_length_of_pieces_default.set_enabled(enable);
        self.rtf_length_of_pieces.set_enabled(enable);
    }

    /// Java private `validate()` (TiltxcorrPanel.java:694-703).
    fn validate(&self) -> bool {
        if self.panel_id == PanelId::PatchTracking {
            if self.ltf_size_of_patches_x_and_y.is_empty() {
                ui_harness::INSTANCE.with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.application_manager),
                        &(self.ltf_size_of_patches_x_and_y.get_label() + " is required."),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return false;
            }
        }
        true
    }

    /// Java private `setToolTipText()` (TiltxcorrPanel.java:754-832).  Tooltip
    /// string initialization.
    fn set_tool_tip_text(&self) {
        let mut text: Option<String>;
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();

        match unsafe {
            autodoc_factory::get_instance(
                Some(self.application_manager),
                Some(autodoc_factory::TILTXCORR),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = instance as *const Autodoc,
            // `catch (final LockException except) {}`.
            Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException except)`:
            // `except.printStackTrace()`.
            Err(except) => eprintln!("{}", except),
        }
        // SAFETY: `autodoc` is null or an autodoc the factory keeps for the
        // life of the process.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*autodoc })
        };
        let tooltip = |key: &str| etomo_autodoc::get_tooltip(autodoc, Some(key));
        self.ltf_test_output
            .set_tool_tip_text(tooltip("TestOutput").as_deref());
        self.ltf_filter_sigma1
            .set_tool_tip_text(tooltip(tiltxcorr_param::FILTER_SIGMA1_KEY).as_deref());
        self.ltf_filter_radius2
            .set_tool_tip_text(tooltip("FilterRadius2").as_deref());
        self.ltf_filter_sigma2
            .set_tool_tip_text(tooltip("FilterSigma2").as_deref());
        self.ltf_trim
            .set_tool_tip_text(tooltip("BordersInXandY").as_deref());
        if let Some(ctf_mag_changes) = &self.ctf_mag_changes {
            ctf_mag_changes.set_check_box_tool_tip_text(
                tooltip(tiltxcorr_param::SEARCH_MAG_CHANGES_KEY).as_deref(),
            );
            // Java sets the check box tooltip a second time here with the
            // ViewsWithMagChanges text (TiltxcorrPanel.java:775-776), which
            // overwrites the SearchMagChanges tooltip and leaves the field
            // without one.  Fixed in translation: the ViewsWithMagChanges text
            // goes on the field, as the pairing of keys to parts intends.
            ctf_mag_changes.set_field_tool_tip_text(
                tooltip(tiltxcorr_param::VIEWS_WITH_MAG_CHANGES_KEY).as_deref(),
            );
        }
        text = tooltip("XMinAndMax");
        if let Some(text) = &text {
            self.pnl_x_min_and_max
                .set_tool_tip_text(Some(text.as_str()));
            self.ltf_x_min.set_tool_tip_text(Some(text.as_str()));
            self.ltf_x_max.set_tool_tip_text(Some(text.as_str()));
        }
        text = tooltip("YMinAndMax");
        if let Some(text) = &text {
            self.pnl_y_min_and_max
                .set_tool_tip_text(Some(text.as_str()));
            self.ltf_y_min.set_tool_tip_text(Some(text.as_str()));
            self.ltf_y_max.set_tool_tip_text(Some(text.as_str()));
        }
        self.ltf_pad_percent
            .set_tool_tip_text(tooltip("PadsInXandY").as_deref());
        self.ltf_taper_percent
            .set_tool_tip_text(tooltip("TapersInXandY").as_deref());
        self.cb_cumulative_correlation
            .set_tool_tip_text_string(tooltip("CumulativeCorrelation").as_deref());
        self.cb_absolute_cosine_stretch
            .set_tool_tip_text_string(tooltip("AbsoluteCosineStretch").as_deref());
        self.cb_no_cosine_stretch
            .set_tool_tip_text_string(tooltip("NoCosineStretch").as_deref());
        self.ltf_view_range
            .set_tool_tip_text(tooltip("StartingEndingViews").as_deref());
        self.ltf_skip_views
            .set_tool_tip_text(tooltip(tiltxcorr_param::SKIP_VIEWS_KEY).as_deref());
        self.cb_exclude_central_peak
            .set_tool_tip_text_string(tooltip("ExcludeCentralPeak").as_deref());
        self.ltf_angle_offset
            .set_tool_tip_text(tooltip("AngleOffset").as_deref());
        if let Some(btn_tiltxcorr) = self.btn_tiltxcorr_button() {
            btn_tiltxcorr.set_tool_tip_text(Some(
                &*("Find alignment transformations between successive ".to_string()
                    + "images by cross-correlation."),
            ));
        }
        self.btn_3dmod_patch_tracking.set_tool_tip_text(Some(
            &*("Open the pre-aligned stack with the patch tracking ".to_string()
                + "fiducial model."),
        ));
        self.ltf_size_of_patches_x_and_y
            .set_tool_tip_text(tooltip(tiltxcorr_param::SIZE_OF_PATCHES_X_AND_Y_KEY).as_deref());
        self.rtf_overlap_of_patches_x_and_y
            .set_tool_tip_text(tooltip(tiltxcorr_param::OVERLAP_OF_PATCHES_X_AND_Y_KEY).as_deref());
        self.rtf_number_of_patches_x_and_y
            .set_tool_tip_text(tooltip(tiltxcorr_param::NUMBER_OF_PATCHES_X_AND_Y_KEY).as_deref());
        self.sp_iterate_correlations
            .set_tool_tip_text(tooltip(tiltxcorr_param::ITERATE_CORRELATIONS_KEY).as_deref());
        self.ltf_shift_limits_x_and_y
            .set_tool_tip_text(tooltip(tiltxcorr_param::SHIFT_LIMITS_X_AND_Y_KEY).as_deref());
        let tooltip_text = tooltip(tiltxcorr_param::BOUNDARY_MODEL_KEY);
        self.cb_boundary_model
            .set_tool_tip_text_string(tooltip_text.as_deref());
        self.btn_3dmod_boundary_model
            .set_tool_tip_text(tooltip_text.as_deref());
        self.ctf_length_of_pieces_minimum_overlap
            .set_check_box_tool_tip_text(
                tooltip(imodchopconts_param::LENGTH_OF_PIECES_KEY).as_deref(),
            );
        self.ctf_length_of_pieces_minimum_overlap
            .set_field_tool_tip_text(tooltip(imodchopconts_param::MINIMUM_OVERLAP_KEY).as_deref());
        self.rb_length_of_pieces_default.set_tool_tip_text_string(
            tooltip(imodchopconts_param::LENGTH_OF_PIECES_KEY).as_deref(),
        );
        self.rtf_length_of_pieces
            .set_tool_tip_text(tooltip(imodchopconts_param::LENGTH_OF_PIECES_KEY).as_deref());
        self.btn_imodchopconts.set_tool_tip_text(Some(
            "Changes the contour pieces without rerunning tiltaxcorr.",
        ));
    }
}

impl ProcessDisplay for TiltxcorrPanel {
    fn as_tilt_xcorr_display(&self) -> Option<&dyn TiltXcorrDisplay> {
        Some(self)
    }
}

impl TiltXcorrDisplay for TiltxcorrPanel {
    /// Java `getParameters(TiltxcorrParam, boolean) throws
    /// FortranInputSyntaxException` (TiltxcorrPanel.java:556-668).  Get the
    /// field values from the panel filling in the TiltxcorrParam object.
    /// Returns false if there was a field error.
    fn get_parameters(
        &self,
        tilt_xcorr_params: &mut TiltxcorrParam,
        do_validation: bool,
    ) -> Result<bool, FortranInputSyntaxException> {
        // The outer `try { ... } catch (FieldValidationFailedException e) {
        // return false; }`: `?` on a field is the throw, `Err` below the catch.
        let outer = (|| -> Result<Result<bool, FortranInputSyntaxException>, FieldValidationFailedException> {
            // A text field's text is never null.
            tilt_xcorr_params.set_test_output(
                &self
                    .ltf_test_output
                    .get_text_boolean(do_validation)?
                    .unwrap_or_default(),
            );
            if self.panel_id == PanelId::CrossCorrelation {
                tilt_xcorr_params
                    .set_exclude_central_peak(self.cb_exclude_central_peak.is_selected());
            } else if self.panel_id == PanelId::PatchTracking {
                let error_message = tilt_xcorr_params
                    .set_iterate_correlations(Some(self.sp_iterate_correlations.get_value()));
                if let Some(error_message) = error_message {
                    ui_harness::INSTANCE.with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.application_manager),
                            &(self
                                .sp_iterate_correlations
                                .get_label()
                                .unwrap_or_else(|| "null".to_string())
                                + ": "
                                + &error_message),
                            "Entry Error",
                            Some(self.axis_id),
                        )
                    });
                    return Ok(Ok(false));
                }
                tilt_xcorr_params.set_input_file(
                    file_type::CLASS
                        .prealigned_stack
                        .get_file_name(Some(self.application_manager), Some(self.axis_id))
                        .as_deref(),
                );
                tilt_xcorr_params.set_output_file(
                    file_type::CLASS
                        .fiducial_patch_tracking_model
                        .get_file_name(Some(self.application_manager), Some(self.axis_id))
                        .as_deref(),
                );
                if self.cb_boundary_model.is_selected() {
                    tilt_xcorr_params.set_boundary_model(
                        file_type::CLASS
                            .patch_tracking_boundary_model
                            .get_file_name(Some(self.application_manager), Some(self.axis_id))
                            .as_deref(),
                    );
                } else {
                    tilt_xcorr_params.reset_boundary_model();
                }
                let meta_data = self.application_manager.get_meta_data();
                let tilt_angle_spec = meta_data.get_tilt_angle_spec(self.axis_id);
                tilt_xcorr_params.set_tilt_angle_spec(&tilt_angle_spec);
                tilt_xcorr_params
                    .set_rotation_angle(meta_data.get_image_rotation(self.axis_id).get_double());
            }
            let mut current_param = String::from("unknown");
            // The inner `try { ... } catch (FortranInputSyntaxException except)`:
            // `Ok(Err(except))` is the throw, `Ok(Ok(Some(false)))` an early
            // `return false`, `Ok(Ok(None))` falling out of the try block.
            //
            // `ParamUtilities` setters that parse a number return `Err(String)`
            // for Java's unchecked NumberFormatException, which Java does not
            // catch here: it escapes getParameters and aborts the Swing action
            // with a stack trace.  Fixed in translation: the exception is
            // reported on standard error and getParameters returns false (a
            // field error), so the caller stops as it does for a bad field.
            let inner = (|| -> Result<Result<Option<bool>, FortranInputSyntaxException>, FieldValidationFailedException> {
                current_param = self.ltf_angle_offset.get_label();
                tilt_xcorr_params
                    .set_angle_offset(self.ltf_angle_offset.get_text_boolean(do_validation)?.as_deref());
                current_param = self.ltf_trim.get_label();
                if let Err(except) = tilt_xcorr_params
                    .set_borders_in_x_and_y(self.ltf_trim.get_text_boolean(do_validation)?.as_deref())
                {
                    return Ok(Err(except));
                }
                current_param = "X".to_string() + &self.ltf_x_min.get_label();
                if let Err(number_format_exception) =
                    tilt_xcorr_params.set_x_min(self.ltf_x_min.get_text_boolean(do_validation)?.as_deref())
                {
                    eprintln!("java.lang.NumberFormatException: {}", number_format_exception);
                    return Ok(Ok(Some(false)));
                }
                current_param = "X".to_string() + &self.ltf_x_max.get_label();
                if let Err(number_format_exception) =
                    tilt_xcorr_params.set_x_max(self.ltf_x_max.get_text_boolean(do_validation)?.as_deref())
                {
                    eprintln!("java.lang.NumberFormatException: {}", number_format_exception);
                    return Ok(Ok(Some(false)));
                }
                current_param = "Y".to_string() + &self.ltf_y_min.get_label();
                if let Err(number_format_exception) =
                    tilt_xcorr_params.set_y_min(self.ltf_y_min.get_text_boolean(do_validation)?.as_deref())
                {
                    eprintln!("java.lang.NumberFormatException: {}", number_format_exception);
                    return Ok(Ok(Some(false)));
                }
                current_param = "Y".to_string() + &self.ltf_y_max.get_label();
                if let Err(number_format_exception) =
                    tilt_xcorr_params.set_y_max(self.ltf_y_max.get_text_boolean(do_validation)?.as_deref())
                {
                    eprintln!("java.lang.NumberFormatException: {}", number_format_exception);
                    return Ok(Ok(Some(false)));
                }
                current_param = self.ltf_pad_percent.get_label();
                if let Err(except) = tilt_xcorr_params
                    .set_pads_in_x_and_y(self.ltf_pad_percent.get_text_boolean(do_validation)?.as_deref())
                {
                    return Ok(Err(except));
                }
                current_param = self.ltf_taper_percent.get_label();
                if let Err(except) = tilt_xcorr_params.set_tapers_in_x_and_y(self.ltf_taper_percent.get_text_boolean(do_validation)?.as_deref()) {
                    return Ok(Err(except));
                }
                current_param = self.ltf_view_range.get_label();
                if let Err(except) = tilt_xcorr_params.set_starting_ending_views(self.ltf_view_range.get_text_boolean(do_validation)?.as_deref()) {
                    return Ok(Err(except));
                }
                current_param = self.ltf_skip_views.get_label();
                tilt_xcorr_params
                    .set_skip_views(self.ltf_skip_views.get_text_boolean(do_validation)?.as_deref());
                current_param = self.ltf_filter_sigma1.get_label();
                tilt_xcorr_params
                    .set_filter_sigma1(self.ltf_filter_sigma1.get_text_boolean(do_validation)?.as_deref());
                current_param = self.ltf_filter_radius2.get_label();
                tilt_xcorr_params.set_filter_radius2(self.ltf_filter_radius2.get_text_boolean(do_validation)?.as_deref());
                current_param = self.ltf_filter_sigma2.get_label();
                tilt_xcorr_params
                    .set_filter_sigma2(self.ltf_filter_sigma2.get_text_boolean(do_validation)?.as_deref());
                if self.panel_id == PanelId::CrossCorrelation {
                    current_param = self.cb_cumulative_correlation.get_text_void().unwrap_or_else(|| "null".to_string());
                    tilt_xcorr_params
                        .set_cumulative_correlation(self.cb_cumulative_correlation.is_selected());
                    current_param = self.cb_absolute_cosine_stretch.get_text_void().unwrap_or_else(|| "null".to_string());
                    tilt_xcorr_params
                        .set_absolute_cosine_stretch(self.cb_absolute_cosine_stretch.is_selected());
                    current_param = self.cb_no_cosine_stretch.get_text_void().unwrap_or_else(|| "null".to_string());
                    tilt_xcorr_params.set_no_cosine_stretch(self.cb_no_cosine_stretch.is_selected());
                } else if self.panel_id == PanelId::PatchTracking {
                    current_param = self.ltf_size_of_patches_x_and_y.get_label();
                    match tilt_xcorr_params.set_size_of_patches_x_and_y(
                        self.ltf_size_of_patches_x_and_y.get_text_boolean(do_validation)?.as_deref(),
                        &self.ltf_size_of_patches_x_and_y.get_label(),
                    ) {
                        Err(except) => return Ok(Err(except)),
                        Ok(false) => return Ok(Ok(Some(false))),
                        Ok(true) => {}
                    }
                    current_param = self.rtf_overlap_of_patches_x_and_y.get_label().unwrap_or_else(|| "null".to_string());
                    if self.rtf_overlap_of_patches_x_and_y.is_selected() {
                        if let Err(except) = tilt_xcorr_params.set_overlap_of_patches_x_and_y(self.rtf_overlap_of_patches_x_and_y.get_text_boolean(do_validation)?.as_deref()) {
                            return Ok(Err(except));
                        }
                    } else {
                        tilt_xcorr_params.reset_overlap_of_patches_x_and_y();
                    }
                    current_param = self.rtf_number_of_patches_x_and_y.get_label().unwrap_or_else(|| "null".to_string());
                    if self.rtf_number_of_patches_x_and_y.is_selected() {
                        if let Err(except) = tilt_xcorr_params.set_number_of_patches_x_and_y(self.rtf_number_of_patches_x_and_y.get_text_boolean(do_validation)?.as_deref()) {
                            return Ok(Err(except));
                        }
                    } else {
                        tilt_xcorr_params.reset_number_of_patches_x_and_y();
                    }
                    current_param = self.ltf_shift_limits_x_and_y.get_label();
                    if let Err(except) = tilt_xcorr_params.set_shift_limits_x_and_y(self.ltf_shift_limits_x_and_y.get_text_boolean(do_validation)?.as_deref()) {
                        return Ok(Err(except));
                    }
                    tilt_xcorr_params.set_prealignment_transform_file_default();
                    tilt_xcorr_params.set_images_are_binned(
                        UIExpertUtilities::INSTANCE.get_stack_binning_base_manager_axis_id_file_type(
                            self.application_manager,
                            self.axis_id,
                            &file_type::CLASS.prealigned_stack,
                        ),
                    );
                }
                if let Some(ctf_mag_changes) = &self.ctf_mag_changes {
                    tilt_xcorr_params.set_search_mag_changes(ctf_mag_changes.is_selected());
                    tilt_xcorr_params.set_views_with_mag_changes(ctf_mag_changes.get_text_boolean(do_validation)?.as_deref());
                }
                Ok(Ok(None))
            })()?;
            match inner {
                Ok(Some(value)) => return Ok(Ok(value)),
                Ok(None) => {}
                Err(except) => {
                    let message = current_param + except.get_message().unwrap_or("null");
                    return Ok(Err(FortranInputSyntaxException::new(&message)));
                }
            }
            Ok(Ok(true))
        })();
        match outer {
            Ok(result) => result,
            Err(_field_validation_failed_exception) => Ok(false),
        }
    }

    /// Java `getPanelId()` (TiltxcorrPanel.java:368-370).
    fn get_panel_id(&self) -> PanelId {
        self.panel_id
    }

    /// Java `getParameters(ImodchopcontsParam, boolean)` (TiltxcorrPanel.java:527-550).
    fn get_parameters_imodchopconts(
        &self,
        param: &mut ImodchopcontsParam,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<(), FieldValidationFailedException> {
            if self.panel_id == PanelId::PatchTracking {
                param.set_minimum_overlap(
                    self.ctf_length_of_pieces_minimum_overlap
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
                if self.ctf_length_of_pieces_minimum_overlap.is_selected() {
                    if self.rb_length_of_pieces_default.is_selected() {
                        param.set_length_of_pieces_default();
                    } else {
                        param.set_length_of_pieces(
                            self.rtf_length_of_pieces
                                .get_text_boolean(do_validation)?
                                .as_deref(),
                        );
                    }
                } else {
                    param.reset_length_of_pieces();
                }
            }
            Ok(())
        })();
        result.is_ok()
    }
}

impl Expandable for TiltxcorrPanel {
    /// Java `expand(GlobalExpandButton)` (TiltxcorrPanel.java:393-394).  All
    /// expansion is done through the header.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java `expand(ExpandButton)` (TiltxcorrPanel.java:396-405).
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.header.equals_open_close(button) {
            self.pnl_body.set_visible(button.is_expanded());
        } else if self.header.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.application_manager))
        });
    }
}

impl Run3dmodButtonContainer for TiltxcorrPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`
    /// (TiltxcorrPanel.java:705-736).
    fn action(
        &self,
        action_command: &str,
        deferred3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let run_tiltxcorr = match self.btn_tiltxcorr_button() {
            Some(btn_tiltxcorr) => {
                Some(action_command) == btn_tiltxcorr.get_action_command().as_deref()
            }
            None => false,
        };
        if run_tiltxcorr && self.panel_id == PanelId::CrossCorrelation {
            self.application_manager.pre_cross_correlate(
                self.axis_id,
                self.btn_tiltxcorr.clone(),
                None,
                self.dialog_type,
                self.this.upgrade().expect("TiltxcorrPanel dropped") as Rc<dyn TiltXcorrDisplay>,
            );
        } else if (run_tiltxcorr && self.panel_id == PanelId::PatchTracking)
            || Some(action_command) == self.btn_imodchopconts.get_action_command().as_deref()
        {
            // The validations do not cover imodchopconts fields
            if run_tiltxcorr && !self.validate() {
                return;
            }
            self.application_manager.tiltxcorr(
                self.axis_id,
                if run_tiltxcorr {
                    self.btn_tiltxcorr.clone()
                } else {
                    Some(self.btn_imodchopconts.clone() as ProcessResultDisplayHandle)
                },
                deferred3dmod_button,
                run3dmod_menu_options.unwrap_or_default(),
                None,
                Some(self.dialog_type),
                self,
                false,
                ProcessName::XCORR_PT, /*was: FileType.PATCH_TRACKING_COMSCRIPT*/
                run_tiltxcorr,
                self.ctf_length_of_pieces_minimum_overlap.is_selected(),
            );
        } else if Some(action_command)
            == self
                .btn_3dmod_patch_tracking
                .get_action_command()
                .as_deref()
        {
            self.application_manager.imod_model(
                &file_type::CLASS.prealigned_stack,
                &file_type::CLASS.fiducial_model,
                self.axis_id,
                run3dmod_menu_options.unwrap_or_default(),
                false,
                true,
            );
        } else if Some(action_command)
            == self
                .btn_3dmod_boundary_model
                .get_action_command()
                .as_deref()
        {
            self.application_manager.imod_model(
                &file_type::CLASS.prealigned_stack,
                &file_type::CLASS.patch_tracking_boundary_model,
                self.axis_id,
                run3dmod_menu_options.unwrap_or_default(),
                false,
                true,
            );
        } else {
            self.update_panel();
        }
    }
}

impl ContextMenu for TiltxcorrPanel {
    /// Java `popUpContextMenu(MouseEvent)` (TiltxcorrPanel.java:349-365).  Right
    /// mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        if self.panel_id != PanelId::PatchTracking {
            if let Some(context_menu) = self.context_menu.as_ref().and_then(Weak::upgrade) {
                context_menu.pop_up_context_menu(mouse_event);
            }
            return;
        }
        let man_pagelabel = [
            "Tiltxcorr".to_string(),
            "Imodchopconts".to_string(),
            "3dmod".to_string(),
        ];
        let man_page = [
            "tiltxcorr.html".to_string(),
            "imodchopconts.html".to_string(),
            "3dmod.html".to_string(),
        ];

        let log_file_label = ["Xcorr_pt".to_string()];
        let log_file = ["xcorr_pt".to_string() + &self.axis_id.get_extension() + ".log"];
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root.get_container(),
            mouse_event,
            Some("PatchTracking"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label[..]),
            Some(&log_file[..]),
            self.application_manager,
            self.axis_id,
        );
    }
}
