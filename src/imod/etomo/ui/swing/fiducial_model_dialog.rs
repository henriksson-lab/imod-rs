//! `IMOD/Etomo/src/etomo/ui/swing/FiducialModelDialog.java`.
//!
//! Java `public final class FiducialModelDialog extends ProcessDialog
//! implements ContextMenu, Run3dmodButtonContainer, Expandable`: the dialog
//! box for creating the fiducial model(s).  An EDT object: created as
//! `Rc<Self>` by [`FiducialModelDialog::get_instance`]; every method takes
//! `&self`; the `ProcessDialog` superclass is the embedded `base` (reached
//! through `Deref`), and the overridden `done()` is
//! `ProcessDialogVirtual::done`.
//!
//! Inner classes: the listener classes `FiducialModelActionListener`,
//! `SeedAndTrackTabChangeListener` and `RunRaptorTabChangeListener` are
//! closures holding a weak reference to the dialog; the enumerated type
//! `SeedModelEnumeratedType` is [`SeedModelEnumeratedType`]; the private tab
//! classes `SeedAndTrackTab` and `RunRaptorTab` are private structs below.

use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::bead_track_display::BeadTrackDisplay;
use super::beadtrack_panel::BeadtrackPanel;
use super::beveled_border::BeveledBorder;
use super::check_box::CheckBox;
use super::check_box_spinner::CheckBoxSpinner;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::radio_text_field::RadioTextField;
use super::raptor_panel::{self, RaptorPanel};
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::tabbed_pane::TabbedPane;
use super::tiltxcorr_display::TiltXcorrDisplay;
use super::tiltxcorr_panel::TiltxcorrPanel;
use super::transferfid_panel::TransferfidPanel;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::autofidseed_param::{self, AutofidseedParam};
use crate::imod::etomo::comscript::beadtrack_param::BeadtrackParam;
use crate::imod::etomo::comscript::const_tiltxcorr_param::ConstTiltxcorrParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::imodchopconts_param::ImodchopcontsParam;
use crate::imod::etomo::comscript::runraptor_param::RunraptorParam;
use crate::imod::etomo::comscript::transferfid_param::TransferfidParam;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, ChangeEvent, JComponent, MouseEvent, MouseListener,
};
use crate::imod::etomo::logic::tracking_method::{self, TrackingMethod};
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autofidseed_init_file_filter;
use crate::imod::etomo::storage::autofidseed_log::AutofidseedLog;
use crate::imod::etomo::storage::autofidseed_selection_and_sorting;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::util::dataset_files;

/// Java package-private static final `SEEDING_NOT_DONE_LABEL`.
pub const SEEDING_NOT_DONE_LABEL: &str = "Seed Fiducial Model";
/// Java private static final `SEEDING_DONE_LABEL`.
const SEEDING_DONE_LABEL: &str = "View Seed Model";
/// Java private static final `DIALOG_TYPE`.
const DIALOG_TYPE: DialogType = DialogType::FiducialModel;
/// Java package-private static final `AUTOFIDSEED_NEW_MODEL_LABEL`.
pub const AUTOFIDSEED_NEW_MODEL_LABEL: &str = "Generate Seed Model";
/// Java private static final `AUTOFIDSEED_APPEND_LABEL`.
const AUTOFIDSEED_APPEND_LABEL: &str = "Add Points to Seed Model";
/// Java private static final `AUTOFIDSEED_NEW_MODEL_TITLE`.
const AUTOFIDSEED_NEW_MODEL_TITLE: &str = "Generate seed model automatically";
/// Java private static final `AUTOFIDSEED_APPEND_TITLE`.
const AUTOFIDSEED_APPEND_TITLE: &str = "Add points to seed model automatically";

/// Java `public final class FiducialModelDialog extends ProcessDialog
/// implements ContextMenu, Run3dmodButtonContainer, Expandable`.
pub struct FiducialModelDialog {
    /// The `ProcessDialog` superclass.
    base: Rc<ProcessDialog>,
    /// Java `this`, for the manager calls that pass the dialog itself.
    this: Weak<FiducialModelDialog>,

    /// Java private final `pnlMain = new JPanel()`.
    pnl_main: Rc<JComponent>,
    /// Java private final `actionListener` (a `FiducialModelActionListener`).
    action_listener: ActionListener,
    /// Java private final `bgMethod = new ButtonGroup()`.
    bg_method: Rc<ButtonGroup>,
    /// Java private final `rbMethodSeed`.
    rb_method_seed: Rc<RadioButton>,
    /// Java private final `rbMethodPatchTracking`.
    rb_method_patch_tracking: Rc<RadioButton>,
    /// Java private final `rbMethodRaptor`.
    rb_method_raptor: Rc<RadioButton>,
    /// Java private final `pnlMethodArray = new JPanel[TrackingMethod.NUM]`
    /// (filled by `createPanel`).
    pnl_method_array: RefCell<Vec<Rc<JComponent>>>,
    /// Java private `tpSeedAndTrack = new TabbedPane()`.
    tp_seed_and_track: Rc<TabbedPane>,
    /// Java private final `pnlSeedAndTrackArray`.
    pnl_seed_and_track_array: RefCell<Vec<Rc<JComponent>>>,
    /// Java private final `pnlSeedAndTrackBodyArray`.
    pnl_seed_and_track_body_array: RefCell<Vec<Rc<JComponent>>>,
    /// Java private `tpRunRaptor = new TabbedPane()`.
    tp_run_raptor: Rc<TabbedPane>,
    /// Java private final `pnlRunRaptorArray`.
    pnl_run_raptor_array: RefCell<Vec<Rc<JComponent>>>,
    /// Java private final `pnlRunRaptorBodyArray`.
    pnl_run_raptor_body_array: RefCell<Vec<Rc<JComponent>>>,
    /// Java private final `bgSeedModel = new ButtonGroup()`.
    bg_seed_model: Rc<ButtonGroup>,
    /// Java private final `rbSeedModelManual`.
    rb_seed_model_manual: Rc<RadioButton>,
    /// Java private final `rbSeedModelAuto`.
    rb_seed_model_auto: Rc<RadioButton>,
    /// Java private final `rbSeedModelTransfer`.
    rb_seed_model_transfer: Rc<RadioButton>,
    /// Java private final `pnlSeedModelArray`.
    pnl_seed_model_array: RefCell<Vec<Rc<JComponent>>>,
    /// Java private final `pnlManualSeedModel`.
    pnl_manual_seed_model: Rc<JComponent>,
    /// Java private final `pnlAutoSeedModel`.
    pnl_auto_seed_model: Rc<JComponent>,
    /// Java private final `pnlAutofidseed`.
    pnl_autofidseed: Rc<JComponent>,
    /// Java private final `cbBoundaryModel`.
    cb_boundary_model: Rc<CheckBox>,
    /// Java private final `btnBoundaryModel`.
    btn_boundary_model: Rc<Run3dmodButton>,
    /// Java private final `cbExcludeInsideAreas`.
    cb_exclude_inside_areas: Rc<CheckBox>,
    /// Java private final `ltfBordersInXandY`.
    ltf_borders_in_xand_y: Rc<LabeledTextField>,
    /// Java private final `ltfMinGuessNumBeads`.
    ltf_min_guess_num_beads: Rc<LabeledTextField>,
    /// Java private final `ltfMinSpacing`.
    ltf_min_spacing: Rc<LabeledTextField>,
    /// Java private final `ltfPeakStorageFraction`.
    ltf_peak_storage_fraction: Rc<LabeledTextField>,
    /// Java private final `bgTarget = new ButtonGroup()`.
    #[allow(dead_code)]
    bg_target: Rc<ButtonGroup>,
    /// Java private final `rtfTargetNumberOfBeads`.
    rtf_target_number_of_beads: Rc<RadioTextField>,
    /// Java private final `rtfTargetDensityOfBeads`.
    rtf_target_density_of_beads: Rc<RadioTextField>,
    /// Java private final `cbTwoSurfaces`.
    cb_two_surfaces: Rc<CheckBox>,
    /// Java private final `cbAppendToSeedModel`.
    cb_append_to_seed_model: Rc<CheckBox>,
    /// Java private final `ltfIgnoreSurfaceData`.
    ltf_ignore_surface_data: Rc<LabeledTextField>,
    /// Java private final `ltfDropTracks`.
    ltf_drop_tracks: Rc<LabeledTextField>,
    /// Java private final `ltfMaxMajorToMinorRatio`.
    ltf_max_major_to_minor_ratio: Rc<LabeledTextField>,
    /// Java private final `cbClusteredPointsAllowedClustered`.
    cb_clustered_points_allowed_clustered: Rc<CheckBox>,
    /// Java private final `cbsElongatedPointsAllowed`.
    cbs_elongated_points_allowed: Rc<CheckBoxSpinner>,
    /// Java private final `ltfLowerTargetForClustered`.
    ltf_lower_target_for_clustered: Rc<LabeledTextField>,
    /// Java private final `btn3dmodAutofidseed`.
    btn_3dmod_autofidseed: Rc<Run3dmodButton>,
    /// Java private final `btn3dmodInitialBeadFinding`.
    btn_3dmod_initial_bead_finding: Rc<Run3dmodButton>,
    /// Java private final `btn3dmodBeadSelectionAndSorting`.
    btn_3dmod_bead_selection_and_sorting: Rc<Run3dmodButton>,
    /// Java private final `btn3dmodClusteredElongatedModel`.
    btn_3dmod_clustered_elongated_model: Rc<Run3dmodButton>,
    /// Java private final `btnCleanup`.
    btn_cleanup: Rc<MultiLineButton>,
    /// Java private final `cbAdjustSizes`.
    cb_adjust_sizes: Rc<CheckBox>,
    /// Java private final `ltfJustFindShiftsNearZero`.
    ltf_just_find_shifts_near_zero: Rc<LabeledTextField>,
    /// Java private final `pnlAllowClusteredElongated`.
    pnl_allow_clustered_elongated: Rc<JComponent>,

    /// Java private final `btnSeed`.
    btn_seed: Rc<Run3dmodButton>,
    /// Java private final `pnlBeadtrack`.
    pnl_beadtrack: Rc<BeadtrackPanel>,
    /// Java private final `pnlTransferfid` (null unless dual axis).
    pnl_transferfid: Option<Rc<TransferfidPanel>>,
    /// Java private final `tiltxcorrPanel`.
    tiltxcorr_panel: Rc<TiltxcorrPanel>,
    /// Java private final `raptorPanel` (null for the second axis of a dual
    /// axis dataset).
    raptor_panel: Option<Rc<RaptorPanel>>,
    /// Java private final `axisType` (stored, never read after construction).
    #[allow(dead_code)]
    axis_type: AxisType,
    /// Java private final `btnAutofidseed`.
    btn_autofidseed: Rc<Run3dmodButton>,
    /// Java private final `btnUseAdjustedTrackCom`.
    btn_use_adjusted_track_com: Rc<MultiLineButton>,
    /// Java private final `btnJustFindShiftsNearZero`.
    btn_just_find_shifts_near_zero: Rc<MultiLineButton>,

    /// Java private `transferfidEnabled = false`.
    transferfid_enabled: Cell<bool>,
    /// Java private `curMethodIndex = -1`.
    cur_method_index: Cell<i32>,
    /// Java private `curSeedAndTrackTab = null`.
    cur_seed_and_track_tab: Cell<Option<SeedAndTrackTab>>,
    /// Java private `curSeedModelIndex = -1`.
    cur_seed_model_index: Cell<i32>,
    /// Java private `curRunRaptorTab = null`.
    cur_run_raptor_tab: Cell<Option<RunRaptorTab>>,
}

impl Deref for FiducialModelDialog {
    type Target = ProcessDialog;
    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

impl FiducialModelDialog {
    /// Java private constructor `FiducialModelDialog(ApplicationManager,
    /// AxisID, AxisType)`, with the field initializers.
    fn new(
        app_mgr: &'static ApplicationManager,
        axis_id: AxisID,
        axis_type: AxisType,
    ) -> Rc<FiducialModelDialog> {
        let dialog = Rc::new_cyclic(|this: &Weak<FiducialModelDialog>| {
            // super(appMgr, axisID, DIALOG_TYPE)
            let base = ProcessDialog::new_application_manager_axis_id_dialog_type(
                app_mgr,
                axis_id,
                DIALOG_TYPE,
            );
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();

            // Field initializers, in declaration order.
            let pnl_main = JComponent::new_panel();
            // Java `new FiducialModelActionListener(this)`: its `actionPerformed`
            // calls `adaptee.action(event.getActionCommand(), null, null)`.
            let adaptee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            });
            let bg_method = ButtonGroup::new();
            let rb_method_seed = RadioButton::new_string_enumerated_type_button_group(
                Some("Make seed and track"),
                Some(EnumeratedTypeRef::new(tracking_method::SEED)),
                Some(&bg_method),
            );
            let rb_method_patch_tracking = RadioButton::new_string_enumerated_type_button_group(
                Some("Use patch tracking to make fiducial model"),
                Some(EnumeratedTypeRef::new(tracking_method::PATCH_TRACKING)),
                Some(&bg_method),
            );
            let rb_method_raptor = RadioButton::new_string_enumerated_type_button_group(
                Some("Run RAPTOR and fix"),
                Some(EnumeratedTypeRef::new(tracking_method::RAPTOR)),
                Some(&bg_method),
            );
            let tp_seed_and_track = TabbedPane::new();
            let tp_run_raptor = TabbedPane::new();
            let bg_seed_model = ButtonGroup::new();
            let rb_seed_model_manual = RadioButton::new_string_enumerated_type_button_group(
                Some("Make seed model manually"),
                Some(EnumeratedTypeRef::new(SeedModelEnumeratedType::MANUAL)),
                Some(&bg_seed_model),
            );
            let rb_seed_model_auto = RadioButton::new_string_enumerated_type_button_group(
                Some(AUTOFIDSEED_NEW_MODEL_TITLE),
                Some(EnumeratedTypeRef::new(SeedModelEnumeratedType::AUTO)),
                Some(&bg_seed_model),
            );
            let rb_seed_model_transfer = RadioButton::new_string_enumerated_type_button_group(
                Some("Transfer seed model from the other axis"),
                Some(EnumeratedTypeRef::new(SeedModelEnumeratedType::TRANSFER)),
                Some(&bg_seed_model),
            );
            let pnl_manual_seed_model = JComponent::new_panel();
            let pnl_auto_seed_model = JComponent::new_panel();
            let pnl_autofidseed = JComponent::new_panel();
            let cb_boundary_model = CheckBox::new_string(Some("Use boundary model"));
            let btn_boundary_model =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Create/Edit Boundary Model"),
                    Some(container.clone()),
                );
            let cb_exclude_inside_areas =
                CheckBox::new_string(Some("Exclude inside boundary contours"));
            let ltf_borders_in_xand_y = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("Borders in X & Y: "),
            );
            let ltf_min_guess_num_beads = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Estimated number of beads in sample: "),
            );
            let ltf_min_spacing = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Minimum spacing: "),
            );
            let ltf_peak_storage_fraction = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Fraction of peaks to store: "),
            );
            let bg_target = ButtonGroup::new();
            let rtf_target_number_of_beads =
                RadioTextField::get_instance_field_type_string_button_group(
                    FieldType::Integer,
                    Some("Total number:"),
                    Some(&bg_target),
                );
            let rtf_target_density_of_beads =
                RadioTextField::get_instance_field_type_string_button_group(
                    FieldType::FloatingPoint,
                    Some("Density (per megapixel):"),
                    Some(&bg_target),
                );
            let cb_two_surfaces = CheckBox::new_string(Some("Select beads on two surfaces"));
            let cb_append_to_seed_model = CheckBox::new_string(Some("Add beads to existing model"));
            let ltf_ignore_surface_data = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Ignore sorting in tracked models: "),
            );
            let ltf_drop_tracks = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Drop tracked models: "),
            );
            let ltf_max_major_to_minor_ratio = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Maximum ratio between surfaces: "),
            );
            let cb_clustered_points_allowed_clustered =
                CheckBox::new_string(Some("Allow clustered beads"));
            let cbs_elongated_points_allowed = CheckBoxSpinner::get_instance_string_int_int_int(
                Some("Allow elongated beads of severity: "),
                1,
                1,
                3,
            );
            let ltf_lower_target_for_clustered = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Lower target number for allowing "),
            );
            let btn_3dmod_autofidseed =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Seed Model"),
                    Some(container.clone()),
                );
            let btn_3dmod_initial_bead_finding =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Initial Bead Model"),
                    Some(container.clone()),
                );
            let btn_3dmod_bead_selection_and_sorting =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Sorted 3D Models"),
                    Some(container.clone()),
                );
            let btn_3dmod_clustered_elongated_model =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Clustered / Elongated Model"),
                    Some(container.clone()),
                );
            let btn_cleanup = MultiLineButton::new_string(Some("Clean Up Temporary Files"));
            let cb_adjust_sizes = CheckBox::new_string(Some("Find and adjust bead size"));
            let ltf_just_find_shifts_near_zero = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Estimated number of beads in sample"),
            );
            let pnl_allow_clustered_elongated = JComponent::new_panel();

            // Constructor body.
            let factory = app_mgr.get_process_result_display_factory(axis_id);
            // Java casts `(Run3dmodButton)` / `(MultiLineButton)` the factory's
            // displays; the factory returns the concrete buttons.
            let btn_seed = factory.get_seed_fiducial_model();
            let btn_autofidseed = factory.get_autofidseed();
            let btn_use_adjusted_track_com = factory.get_use_adjusted_track_com();
            btn_use_adjusted_track_com.set_enabled(false);
            let btn_just_find_shifts_near_zero = factory.get_just_find_shifts_near_zero();
            let raptor_panel = if axis_type != AxisType::DualAxis || axis_id != AxisID::Second {
                Some(RaptorPanel::get_instance(
                    app_mgr,
                    axis_id,
                    base.dialog_type,
                ))
            } else {
                None
            };
            let pnl_beadtrack = BeadtrackPanel::get_instance(
                app_mgr,
                axis_id,
                base.dialog_type,
                &base.btn_advanced,
            );
            let tiltxcorr_panel = TiltxcorrPanel::get_patch_tracking_instance(
                app_mgr,
                axis_id,
                DIALOG_TYPE,
                &base.btn_advanced,
            );
            let pnl_transferfid = if base.application_manager.is_dual_axis() {
                Some(TransferfidPanel::get_instance(
                    base.application_manager,
                    axis_id,
                    base.dialog_type,
                    &base.btn_advanced,
                ))
            } else {
                None
            };

            FiducialModelDialog {
                base,
                this: this.clone(),
                pnl_main,
                action_listener,
                bg_method,
                rb_method_seed,
                rb_method_patch_tracking,
                rb_method_raptor,
                pnl_method_array: RefCell::new(Vec::new()),
                tp_seed_and_track,
                pnl_seed_and_track_array: RefCell::new(Vec::new()),
                pnl_seed_and_track_body_array: RefCell::new(Vec::new()),
                tp_run_raptor,
                pnl_run_raptor_array: RefCell::new(Vec::new()),
                pnl_run_raptor_body_array: RefCell::new(Vec::new()),
                bg_seed_model,
                rb_seed_model_manual,
                rb_seed_model_auto,
                rb_seed_model_transfer,
                pnl_seed_model_array: RefCell::new(Vec::new()),
                pnl_manual_seed_model,
                pnl_auto_seed_model,
                pnl_autofidseed,
                cb_boundary_model,
                btn_boundary_model,
                cb_exclude_inside_areas,
                ltf_borders_in_xand_y,
                ltf_min_guess_num_beads,
                ltf_min_spacing,
                ltf_peak_storage_fraction,
                bg_target,
                rtf_target_number_of_beads,
                rtf_target_density_of_beads,
                cb_two_surfaces,
                cb_append_to_seed_model,
                ltf_ignore_surface_data,
                ltf_drop_tracks,
                ltf_max_major_to_minor_ratio,
                cb_clustered_points_allowed_clustered,
                cbs_elongated_points_allowed,
                ltf_lower_target_for_clustered,
                btn_3dmod_autofidseed,
                btn_3dmod_initial_bead_finding,
                btn_3dmod_bead_selection_and_sorting,
                btn_3dmod_clustered_elongated_model,
                btn_cleanup,
                cb_adjust_sizes,
                ltf_just_find_shifts_near_zero,
                pnl_allow_clustered_elongated,
                btn_seed,
                pnl_beadtrack,
                pnl_transferfid,
                tiltxcorr_panel,
                raptor_panel,
                axis_type,
                btn_autofidseed,
                btn_use_adjusted_track_com,
                btn_just_find_shifts_near_zero,
                transferfid_enabled: Cell::new(false),
                cur_method_index: Cell::new(-1),
                cur_seed_and_track_tab: Cell::new(None),
                cur_seed_model_index: Cell::new(-1),
                cur_run_raptor_tab: Cell::new(None),
            }
        });
        // Java `this` as the ProcessDialog subclass (for the virtual `done()`).
        let this: Weak<dyn ProcessDialogVirtual> =
            Rc::downgrade(&dialog) as Weak<dyn ProcessDialogVirtual>;
        dialog.base.set_this(this);
        dialog
    }

    /// Java public static `getInstance(ApplicationManager, AxisID, AxisType)`.
    pub fn get_instance(
        app_mgr: &'static ApplicationManager,
        axis_id: AxisID,
        axis_type: AxisType,
    ) -> Rc<FiducialModelDialog> {
        let instance = FiducialModelDialog::new(app_mgr, axis_id, axis_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // Local panels
        let pnl_method = JComponent::new_panel();
        let pnl_method_x = JComponent::new_panel();
        let pnl_seed_model = JComponent::new_panel();
        let pnl_seed_model_x = JComponent::new_panel();
        let pnl_autofidseed_param = JComponent::new_panel();
        let pnl_initial_bead_finding = JComponent::new_panel();
        let pnl_boundary_model = JComponent::new_panel();
        let pnl_3dmod_initial_bead_finding = JComponent::new_panel();
        let pnl_bead_searching_and_sorting = JComponent::new_panel();
        let pnl_exclude_inside_areas = JComponent::new_panel();
        let pnl_target = JComponent::new_panel();
        let pnl_two_surfaces = JComponent::new_panel();
        let pnl_append_to_seed_model = JComponent::new_panel();
        let pnl_allow_clustered = JComponent::new_panel();
        let pnl_allow_elongated = JComponent::new_panel();
        let pnl_allow_only_if_number = JComponent::new_panel();
        let pnl_3dmod_bead_sorting_and_searching = JComponent::new_panel();
        let pnl_buttons = JComponent::new_panel();
        let pnl_3dmod_clustered_elongated_model = JComponent::new_panel();
        let pnl_adjust_sizes = JComponent::new_panel();
        let pnl_just_find_shifts_near_zero = JComponent::new_panel();
        let pnl_seed = JComponent::new_panel();
        let pnl_just_find_shifts_near_zero_button = JComponent::new_panel();
        // Init
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_3dmod_autofidseed.clone();
        self.btn_autofidseed
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        self.btn_3dmod_initial_bead_finding.set_enabled(false);
        self.btn_3dmod_bead_selection_and_sorting.set_enabled(false);
        self.ltf_ignore_surface_data.set_enabled(false);
        self.btn_3dmod_clustered_elongated_model.set_enabled(false);
        self.btn_execute.set_text(Some("Done"));
        self.rtf_target_number_of_beads.set_selected_boolean(true);
        self.rtf_target_number_of_beads.set_required(true);
        self.rtf_target_density_of_beads.set_required(true);

        self.ltf_just_find_shifts_near_zero.set_columns(3);
        // This is only used when the JustFindShiftsNearZero button is pressed. It is
        // currently not being read at all in other situations. If it needs to be read in
        // other situations, then its require setting will need to be turned on and off.
        self.ltf_just_find_shifts_near_zero.set_required(true);
        self.ltf_just_find_shifts_near_zero
            .set_number_must_be_positive(true);

        // Root
        // Swing layout: rootPanel BoxLayout Y_AXIS.
        self.root_panel
            .set_border(&BeveledBorder::new(Some("Fiducial Model Generation")).get_border());
        self.root_panel.get_component().add(&self.pnl_main);
        self.add_exit_buttons();
        // Main
        // Swing layout: pnlMain BoxLayout Y_AXIS.
        self.pnl_main.add(&pnl_method_x);
        // Method
        // center the radio button panel
        // Swing layout: pnlMethodX BoxLayout X_AXIS, CENTER_ALIGNMENT,
        // horizontal glue around pnlMethod.
        pnl_method_x.add(&pnl_method);
        // radio button panel
        // Swing layout: pnlMethod BoxLayout Y_AXIS, etched border.
        pnl_method.add(&self.rb_method_seed.get_component());
        pnl_method.add(&self.rb_method_patch_tracking.get_component());
        if self.raptor_panel.is_some() {
            pnl_method.add(&self.rb_method_raptor.get_component());
        }
        // panels to switch when the radio buttons change
        {
            let mut pnl_method_array = self.pnl_method_array.borrow_mut();
            for _ in 0..tracking_method::NUM {
                pnl_method_array.push(JComponent::new_panel());
            }
        }
        let mut i = tracking_method::SEED.get_value().get_int();
        self.pnl_method(i)
            .add(&self.tp_seed_and_track.get_component());
        i = tracking_method::PATCH_TRACKING.get_value().get_int();
        self.pnl_method(i).add(&self.tiltxcorr_panel.get_panel());
        i = tracking_method::RAPTOR.get_value().get_int();
        self.pnl_method(i).add(&self.tp_run_raptor.get_component());
        // Seed and track
        // panels to switch when the tab changes
        i = 0;
        while i < SeedAndTrackTab::NUM_TABS {
            let pnl = JComponent::new_panel();
            self.pnl_seed_and_track_array.borrow_mut().push(pnl.clone());
            self.tp_seed_and_track
                .add_tab_string_component(SeedAndTrackTab::get_instance(i).title, &pnl);
            self.pnl_seed_and_track_body_array
                .borrow_mut()
                .push(JComponent::new_panel());
            // Swing layout: pnlSeedAndTrackBodyArray[i] BoxLayout Y_AXIS.
            i += 1;
        }
        i = SeedAndTrackTab::SEED.index;
        self.pnl_seed_and_track_body(i).add(&pnl_seed_model_x);
        // Run raptor
        // panels to switch when the tab changes
        i = 0;
        while i < RunRaptorTab::NUM_TABS {
            let pnl = JComponent::new_panel();
            self.pnl_run_raptor_array.borrow_mut().push(pnl.clone());
            self.tp_run_raptor
                .add_tab_string_component(RunRaptorTab::get_instance(i).title, &pnl);
            self.pnl_run_raptor_body_array
                .borrow_mut()
                .push(JComponent::new_panel());
            i += 1;
        }
        i = RunRaptorTab::RAPTOR.index;
        if let Some(raptor_panel) = &self.raptor_panel {
            self.pnl_run_raptor_body(i)
                .add(&raptor_panel.get_component());
        }
        // Seed model
        // center the radio button panel
        // Swing layout: pnlSeedModelX BoxLayout X_AXIS, CENTER_ALIGNMENT,
        // horizontal glue around pnlSeedModel.
        pnl_seed_model_x.add(&pnl_seed_model);
        // radio button panel
        // Swing layout: pnlSeedModel BoxLayout Y_AXIS, etched border.
        pnl_seed_model.add(&self.rb_seed_model_manual.get_component());
        pnl_seed_model.add(&self.rb_seed_model_auto.get_component());
        if self.pnl_transferfid.is_some() {
            pnl_seed_model.add(&self.rb_seed_model_transfer.get_component());
        }
        // panels to switch when the radio buttons change
        {
            let mut pnl_seed_model_array = self.pnl_seed_model_array.borrow_mut();
            for _ in 0..SeedModelEnumeratedType::NUM {
                pnl_seed_model_array.push(JComponent::new_panel());
            }
        }
        i = SeedModelEnumeratedType::MANUAL.value;
        self.pnl_seed_model(i).add(&self.pnl_manual_seed_model);
        i = SeedModelEnumeratedType::AUTO.value;
        self.pnl_seed_model(i).add(&self.pnl_auto_seed_model);
        i = SeedModelEnumeratedType::TRANSFER.value;
        if let Some(pnl_transferfid) = &self.pnl_transferfid {
            self.pnl_seed_model(i).add(&pnl_transferfid.get_container());
        }
        // ManualSeedModel
        // Swing layout: pnlManualSeedModel BoxLayout Y_AXIS.
        let _ = self.btn_seed.get_component();
        self.pnl_manual_seed_model.add(&pnl_seed);
        // Swing layout: Box.createRigidArea(FixedDim.x0_y5).
        self.pnl_manual_seed_model
            .add(&pnl_just_find_shifts_near_zero);
        // Seed
        // Swing layout: pnlSeed BoxLayout X_AXIS, horizontal glue around btnSeed.
        pnl_seed.add(&self.btn_seed.get_component());
        // JustFindShiftsNearZeroButton
        // Swing layout: BoxLayout X_AXIS, horizontal glue around the button.
        pnl_just_find_shifts_near_zero_button
            .add(&self.btn_just_find_shifts_near_zero.get_component());
        // JustFindShiftsNearZero
        // Swing layout: pnlJustFindShiftsNearZero BoxLayout Y_AXIS.
        pnl_just_find_shifts_near_zero.set_border_title(
            EtchedBorder::new(Some("Find shifts near zero tilt"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_just_find_shifts_near_zero.add(&self.ltf_just_find_shifts_near_zero.get_component());
        pnl_just_find_shifts_near_zero.add(&pnl_just_find_shifts_near_zero_button);
        // Auto seed model
        // Swing layout: pnlAutoSeedModel BoxLayout Y_AXIS.
        // Autofidseed - goes on the AutoSeedModel panel under the shared beadtrack panel
        // Swing layout: pnlAutofidseed BoxLayout Y_AXIS; rigid area x0_y5.
        self.pnl_autofidseed.add(&pnl_autofidseed_param);
        // Swing layout: Box.createRigidArea(FixedDim.x0_y10).
        self.pnl_autofidseed.add(&pnl_buttons);
        // AutofidseedParam
        // Autofidseed - goes on the AutoSeedModel panel under the shared beadtrack panel
        // Swing layout: pnlAutofidseedParam BoxLayout X_AXIS.
        pnl_autofidseed_param.add(&pnl_initial_bead_finding);
        // Swing layout: Box.createRigidArea(FixedDim.x5_y0).
        pnl_autofidseed_param.add(&pnl_bead_searching_and_sorting);
        // Initial bead finding
        // Swing layout: pnlInitialBeadFinding BoxLayout Y_AXIS, TOP_ALIGNMENT.
        pnl_initial_bead_finding.set_border_title(
            EtchedBorder::new(Some("Initial Bead Finding Parameters"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_initial_bead_finding.add(&pnl_boundary_model);
        pnl_initial_bead_finding.add(&pnl_exclude_inside_areas);
        pnl_initial_bead_finding.add(&pnl_adjust_sizes);
        pnl_initial_bead_finding.add(&self.ltf_borders_in_xand_y.get_component());
        pnl_initial_bead_finding.add(&self.ltf_min_guess_num_beads.get_component());
        pnl_initial_bead_finding.add(&self.ltf_min_spacing.get_component());
        pnl_initial_bead_finding.add(&self.ltf_peak_storage_fraction.get_component());
        // Swing layout: Box.createRigidArea(FixedDim.x0_y3).
        pnl_initial_bead_finding.add(&pnl_3dmod_initial_bead_finding);
        // Boundary model
        // Swing layout: pnlBoundaryModel BoxLayout X_AXIS.
        pnl_boundary_model.add(&self.cb_boundary_model.get_component());
        pnl_boundary_model.add(&self.btn_boundary_model.get_component());
        // ExcludeInsideAreas
        // Swing layout: BoxLayout X_AXIS, trailing horizontal glue.
        pnl_exclude_inside_areas.add(&self.cb_exclude_inside_areas.get_component());
        // AdjustSizes
        // Swing layout: BoxLayout X_AXIS, trailing horizontal glue.
        pnl_adjust_sizes.add(&self.cb_adjust_sizes.get_component());
        // 3dmodInitialBeadFinding
        // Swing layout: BoxLayout X_AXIS, horizontal glue around the button.
        pnl_3dmod_initial_bead_finding.add(&self.btn_3dmod_initial_bead_finding.get_component());
        // Bead sorting and searching
        // Swing layout: pnlBeadSearchingAndSorting BoxLayout Y_AXIS, TOP_ALIGNMENT.
        pnl_bead_searching_and_sorting.set_border_title(
            EtchedBorder::new(Some("Selection and Sorting Parameters"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_bead_searching_and_sorting.add(&pnl_target);
        pnl_bead_searching_and_sorting.add(&pnl_two_surfaces);
        pnl_bead_searching_and_sorting.add(&pnl_append_to_seed_model);
        pnl_bead_searching_and_sorting.add(&self.ltf_ignore_surface_data.get_component());
        pnl_bead_searching_and_sorting.add(&self.ltf_drop_tracks.get_component());
        pnl_bead_searching_and_sorting.add(&self.ltf_max_major_to_minor_ratio.get_component());
        pnl_bead_searching_and_sorting.add(&self.pnl_allow_clustered_elongated);
        // Swing layout: Box.createRigidArea(FixedDim.x0_y3).
        pnl_bead_searching_and_sorting.add(&pnl_3dmod_bead_sorting_and_searching);
        // Swing layout: Box.createRigidArea(FixedDim.x0_y3).
        pnl_bead_searching_and_sorting.add(&pnl_3dmod_clustered_elongated_model);
        // Target
        // Swing layout: pnlTarget BoxLayout Y_AXIS.
        pnl_target.set_border_title(
            EtchedBorder::new(Some("Seed Points to Select"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_target.add(&self.rtf_target_number_of_beads.get_container());
        pnl_target.add(&self.rtf_target_density_of_beads.get_container());
        // TwoSurfaces
        // Swing layout: BoxLayout X_AXIS, trailing horizontal glue.
        pnl_two_surfaces.add(&self.cb_two_surfaces.get_component());
        // AppendToSeedModel
        // Swing layout: BoxLayout X_AXIS, trailing horizontal glue.
        pnl_append_to_seed_model.add(&self.cb_append_to_seed_model.get_component());
        // Allow Clustered/Elongated
        // Swing layout: pnlAllowClusteredElongated BoxLayout Y_AXIS.
        self.pnl_allow_clustered_elongated.set_border_title(
            EtchedBorder::new(Some("Clustered/Elongated Beads"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_allow_clustered_elongated.add(&pnl_allow_clustered);
        self.pnl_allow_clustered_elongated.add(&pnl_allow_elongated);
        self.pnl_allow_clustered_elongated
            .add(&pnl_allow_only_if_number);
        // AllowClustered
        // Swing layout: BoxLayout X_AXIS, rigid area x3_y0, trailing glue.
        pnl_allow_clustered.add(&self.cb_clustered_points_allowed_clustered.get_component());
        // AlowElongated
        // Swing layout: BoxLayout X_AXIS, trailing horizontal glue.
        pnl_allow_elongated.add(&self.cbs_elongated_points_allowed.get_container());
        // AllowOnlyIfNumber<
        // Swing layout: BoxLayout X_AXIS, rigid area x3_y0, trailing glue.
        pnl_allow_only_if_number.add(&self.ltf_lower_target_for_clustered.get_component());
        // 3dmodBeadSortingAndSearching
        // Swing layout: BoxLayout X_AXIS, horizontal glue around the button.
        pnl_3dmod_bead_sorting_and_searching
            .add(&self.btn_3dmod_bead_selection_and_sorting.get_component());
        // 3dmodClusteredElongatedModel
        // Swing layout: BoxLayout X_AXIS, horizontal glue around the button.
        pnl_3dmod_clustered_elongated_model
            .add(&self.btn_3dmod_clustered_elongated_model.get_component());
        // Buttons
        // Swing layout: horizontal glue between the buttons.
        pnl_buttons.add(&self.btn_autofidseed.get_component());
        pnl_buttons.add(&self.btn_3dmod_autofidseed.get_component());
        pnl_buttons.add(&self.btn_use_adjusted_track_com.get_component());
        pnl_buttons.add(&self.btn_cleanup.get_component());

        // update
        self.update_advanced_void();
        self.update_enabled();
        self.update_method();
        self.change_seed_and_track_tab();
        self.change_run_raptor_tab();
        self.update_seed_model();
        self.update_display();
    }

    /// Java `pnlMethodArray[i]`.
    fn pnl_method(&self, i: i32) -> Rc<JComponent> {
        self.pnl_method_array.borrow()[i as usize].clone()
    }

    /// Java `pnlSeedAndTrackArray[i]`.
    fn pnl_seed_and_track(&self, i: i32) -> Rc<JComponent> {
        self.pnl_seed_and_track_array.borrow()[i as usize].clone()
    }

    /// Java `pnlSeedAndTrackBodyArray[i]`.
    fn pnl_seed_and_track_body(&self, i: i32) -> Rc<JComponent> {
        self.pnl_seed_and_track_body_array.borrow()[i as usize].clone()
    }

    /// Java `pnlRunRaptorArray[i]`.
    fn pnl_run_raptor(&self, i: i32) -> Rc<JComponent> {
        self.pnl_run_raptor_array.borrow()[i as usize].clone()
    }

    /// Java `pnlRunRaptorBodyArray[i]`.
    fn pnl_run_raptor_body(&self, i: i32) -> Rc<JComponent> {
        self.pnl_run_raptor_body_array.borrow()[i as usize].clone()
    }

    /// Java `pnlSeedModelArray[i]`.
    fn pnl_seed_model(&self, i: i32) -> Rc<JComponent> {
        self.pnl_seed_model_array.borrow()[i as usize].clone()
    }

    /// Java `((RadioButton.RadioButtonModel) group.getSelection())
    /// .getEnumeratedType().getValue().getInt()`.  Java throws
    /// NullPointerException with nothing selected; both groups this dialog
    /// reads always have a selection (their default member selects itself).
    fn selected_value(group: &ButtonGroup) -> i32 {
        (group.get_selection())
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<RadioButtonModel>()
                    .and_then(|model| model.get_enumerated_type())
            })
            .expect("button group has a selected RadioButtonModel")
            .get_value()
            .get_int()
    }

    /// Java private `updateMethod()`.  Responds to the method radio buttons.
    /// Places the panel which corresponds to the currently selected radio
    /// button in the main panel.
    fn update_method(&self) {
        // Changed the panel
        if self.cur_method_index.get() != -1 {
            self.pnl_main
                .remove(&self.pnl_method(self.cur_method_index.get()));
        }
        self.cur_method_index
            .set(Self::selected_value(&self.bg_method));
        self.pnl_main
            .add(&self.pnl_method(self.cur_method_index.get()));
        // Refresh the tabs if necessary
        if self.cur_method_index.get() == tracking_method::SEED.get_value().get_int() {
            self.change_seed_and_track_tab();
        } else if self.cur_method_index.get() == tracking_method::RAPTOR.get_value().get_int() {
            self.change_run_raptor_tab();
        } else {
            self.pack();
        }
    }

    /// Java `UIHarness.INSTANCE.pack(axisID, applicationManager)`.
    fn pack(&self) {
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }

    /// Java private `changeSeedAndTrackTab()`.  Responds to the make seed and
    /// track tabs.  Places the body panel into the panel in the tab which the
    /// user selected.
    ///
    /// Upstream bug fixed in translation (FiducialModelDialog.java:518-519):
    /// with no current tab and no selected tab (`newIndex == -1`) the Java
    /// condition evaluates `!curSeedAndTrackTab.equals(newIndex)` on null and
    /// throws NullPointerException.  Here that case is "no change".
    fn change_seed_and_track_tab(&self) {
        let new_index = self.tp_seed_and_track.get_component().get_selected_tab();
        // Change the tab body panel
        let changed = match self.cur_seed_and_track_tab.get() {
            None => new_index != -1,
            Some(cur) => !cur.equals(new_index),
        };
        if changed {
            // If the tab has changed:
            // Remove the body from previous tab so that the size of the tabbed pane won't
            // reflect it.
            if let Some(cur) = self.cur_seed_and_track_tab.get() {
                self.pnl_seed_and_track(cur.index)
                    .remove(&self.pnl_seed_and_track_body(cur.index));
            }
            // Update the current tab and add the body to the tabbed pane
            let cur = SeedAndTrackTab::get_instance(new_index);
            self.cur_seed_and_track_tab.set(Some(cur));
            self.pnl_seed_and_track(cur.index)
                .add(&self.pnl_seed_and_track_body(cur.index));
        }
        // Replace shared panels
        if self.cur_seed_and_track_tab.get() == Some(SeedAndTrackTab::TRACK) {
            let index = SeedAndTrackTab::TRACK.index;
            self.pnl_seed_and_track_body(index)
                .add(&self.pnl_beadtrack.get_container());
            self.pnl_beadtrack.update_autofidseed(false);
        }
        // Refresh the auto seed panel if necessary
        if self.cur_seed_and_track_tab.get() == Some(SeedAndTrackTab::SEED)
            && self.cur_seed_model_index.get() == SeedModelEnumeratedType::AUTO.value
        {
            self.update_seed_model();
        } else {
            self.pack();
        }
        ui_harness::with(|harness| harness.move_sub_frame());
    }

    /// Java private `changeRunRaptorTab()`.  Responds to the make run raptor
    /// tabs.  Places the body panel into the panel in the tab which the user
    /// selected.
    ///
    /// Upstream bug fixed in translation (FiducialModelDialog.java:555-556):
    /// the same null dereference as `changeSeedAndTrackTab`; see there.
    fn change_run_raptor_tab(&self) {
        let new_index = self.tp_run_raptor.get_component().get_selected_tab();
        // Change the tab body panel
        let changed = match self.cur_run_raptor_tab.get() {
            None => new_index != -1,
            Some(cur) => !cur.equals(new_index),
        };
        if changed {
            // If the tab has changed:
            // Remove the body from previous tab so that the size of the tabbed pane won't
            // reflect it.
            if let Some(cur) = self.cur_run_raptor_tab.get() {
                self.pnl_run_raptor(cur.index)
                    .remove(&self.pnl_run_raptor_body(cur.index));
            }
            let cur = RunRaptorTab::get_instance(new_index);
            self.cur_run_raptor_tab.set(Some(cur));
            self.pnl_run_raptor(cur.index)
                .add(&self.pnl_run_raptor_body(cur.index));
        }
        // Replace shared panels
        if self.cur_run_raptor_tab.get() == Some(RunRaptorTab::TRACK) {
            let index = RunRaptorTab::TRACK.index;
            self.pnl_run_raptor_body(index)
                .add(&self.pnl_beadtrack.get_container());
            self.pnl_beadtrack.update_autofidseed(false);
        }
        self.pack();
        ui_harness::with(|harness| harness.move_sub_frame());
    }

    /// Java private `updateSeedModel()`.  Responds to the seed model radio
    /// buttons.  Places the panel which corresponds to the currently selected
    /// radio button in the seed tab's body panel.
    fn update_seed_model(&self) {
        let new_index = Self::selected_value(&self.bg_seed_model);
        // Change the panel
        let cur_seed_model_index = self.cur_seed_model_index.get();
        if (cur_seed_model_index == -1 && new_index != -1) || cur_seed_model_index != new_index {
            // If a different radio button has been selected:
            if cur_seed_model_index != -1 {
                self.pnl_seed_and_track_body(SeedAndTrackTab::SEED.index)
                    .remove(&self.pnl_seed_model(cur_seed_model_index));
            }
            self.cur_seed_model_index.set(new_index);
            self.pnl_seed_and_track_body(SeedAndTrackTab::SEED.index)
                .add(&self.pnl_seed_model(new_index));
        }
        // Replace shared panels
        if self.cur_seed_and_track_tab.get() == Some(SeedAndTrackTab::SEED)
            && self.cur_seed_model_index.get() == SeedModelEnumeratedType::AUTO.value
        {
            self.pnl_auto_seed_model.remove(&self.pnl_autofidseed);
            self.pnl_auto_seed_model
                .add(&self.pnl_beadtrack.get_container());
            self.pnl_auto_seed_model.add(&self.pnl_autofidseed);
            self.pnl_beadtrack.update_autofidseed(true);
        }
        self.pack();
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_seed.set_container(Some(container.clone()));
        self.btn_autofidseed.set_container(Some(container));
        let context_menu: Weak<dyn ContextMenu> = self.this.clone();
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.root_panel
            .get_component()
            .add_mouse_listener(mouse_adapter.clone());
        let mut i = 0;
        while i < SeedAndTrackTab::NUM_TABS {
            self.pnl_seed_and_track(i)
                .add_mouse_listener(mouse_adapter.clone());
            self.pnl_seed_and_track_body(i)
                .add_mouse_listener(mouse_adapter.clone());
            i += 1;
        }
        let mut i = 0;
        while i < RunRaptorTab::NUM_TABS {
            self.pnl_run_raptor(i)
                .add_mouse_listener(mouse_adapter.clone());
            self.pnl_run_raptor_body(i)
                .add_mouse_listener(mouse_adapter.clone());
            i += 1;
        }
        // Java `new SeedAndTrackTabChangeListener(this)`: `stateChanged` calls
        // `dialog.changeSeedAndTrackTab()`.
        let dialog = self.this.clone();
        self.tp_seed_and_track
            .get_component()
            .add_change_listener(Rc::new(move |_change_event: &ChangeEvent| {
                if let Some(dialog) = dialog.upgrade() {
                    dialog.change_seed_and_track_tab();
                }
            }));
        // Java `new RunRaptorTabChangeListener(this)`: `stateChanged` calls
        // `dialog.changeRunRaptorTab()`.
        let dialog = self.this.clone();
        self.tp_run_raptor
            .get_component()
            .add_change_listener(Rc::new(move |_change_event: &ChangeEvent| {
                if let Some(dialog) = dialog.upgrade() {
                    dialog.change_run_raptor_tab();
                }
            }));
        self.btn_seed
            .add_action_listener(self.action_listener.clone());
        self.btn_autofidseed
            .add_action_listener(self.action_listener.clone());
        self.btn_use_adjusted_track_com
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_autofidseed
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_initial_bead_finding
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_bead_selection_and_sorting
            .add_action_listener(self.action_listener.clone());
        self.btn_cleanup
            .add_action_listener(self.action_listener.clone());
        let expandable: Weak<dyn Expandable> = self.this.clone();
        self.btn_advanced.register_expandable(expandable);
        self.rb_method_seed
            .add_action_listener(self.action_listener.clone());
        self.rb_method_patch_tracking
            .add_action_listener(self.action_listener.clone());
        self.rb_method_raptor
            .add_action_listener(self.action_listener.clone());
        self.rb_seed_model_manual
            .add_action_listener(self.action_listener.clone());
        self.rb_seed_model_auto
            .add_action_listener(self.action_listener.clone());
        self.rb_seed_model_transfer
            .add_action_listener(self.action_listener.clone());
        self.cb_boundary_model
            .add_action_listener(Some(self.action_listener.clone()));
        self.btn_boundary_model
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_clustered_elongated_model
            .add_action_listener(self.action_listener.clone());
        self.cb_append_to_seed_model
            .add_action_listener(Some(self.action_listener.clone()));
        self.rtf_target_number_of_beads
            .add_action_listener(self.action_listener.clone());
        self.rtf_target_density_of_beads
            .add_action_listener(self.action_listener.clone());
        self.cb_clustered_points_allowed_clustered
            .add_action_listener(Some(self.action_listener.clone()));
        self.cbs_elongated_points_allowed
            .add_action_listener(Some(self.action_listener.clone()));
        self.btn_just_find_shifts_near_zero
            .add_action_listener(self.action_listener.clone());
    }

    /// Java public static `getUseRaptorResultLabel()`.
    pub fn get_use_raptor_result_label() -> &'static str {
        raptor_panel::USE_RAPTOR_RESULT_LABEL
    }

    /// Java `updateDisplay()`.
    pub fn update_display(&self) {
        if self
            .application_manager
            .get_state()
            .is_seeding_done(self.axis_id)
        {
            self.btn_seed.set_text(Some(SEEDING_DONE_LABEL));
        } else {
            self.btn_seed.set_text(Some(SEEDING_NOT_DONE_LABEL));
        }
        let selected = self.cb_boundary_model.is_selected();
        self.btn_boundary_model.set_enabled(selected);
        self.cb_exclude_inside_areas.set_enabled(selected);
        if self.cb_append_to_seed_model.is_selected() {
            self.btn_autofidseed
                .set_text(Some(AUTOFIDSEED_APPEND_LABEL));
            self.rb_seed_model_auto
                .set_text(Some(AUTOFIDSEED_APPEND_TITLE));
        } else {
            self.btn_autofidseed
                .set_text(Some(AUTOFIDSEED_NEW_MODEL_LABEL));
            self.rb_seed_model_auto
                .set_text(Some(AUTOFIDSEED_NEW_MODEL_TITLE));
        }
        self.ltf_lower_target_for_clustered.set_enabled(
            (self.cb_clustered_points_allowed_clustered.is_selected()
                || self.cbs_elongated_points_allowed.is_selected())
                && !self.rtf_target_density_of_beads.is_selected(),
        );
    }

    /// Java private `updateAdvanced()`.  Set the advanced state for the
    /// dialog box.
    fn update_advanced_void(&self) {
        self.update_advanced_boolean(self.is_advanced());
    }

    /// Java private `updateAdvanced(boolean)`.
    fn update_advanced_boolean(&self, advanced: bool) {
        self.pnl_beadtrack.update_advanced(advanced);
        if let Some(pnl_transferfid) = &self.pnl_transferfid {
            pnl_transferfid.update_advanced(advanced);
        }
        self.tiltxcorr_panel.update_advanced(advanced);

        self.cb_adjust_sizes.set_visible(advanced);
        self.ltf_borders_in_xand_y.set_visible(advanced);
        self.ltf_min_guess_num_beads.set_visible(advanced);
        self.ltf_min_spacing.set_visible(advanced);
        self.ltf_peak_storage_fraction.set_visible(advanced);
        self.ltf_ignore_surface_data.set_visible(advanced);
        self.ltf_drop_tracks.set_visible(advanced);
        self.ltf_max_major_to_minor_ratio.set_visible(advanced);
        self.pnl_allow_clustered_elongated.set_visible(advanced);
        self.cb_clustered_points_allowed_clustered
            .set_visible(advanced);
        self.cbs_elongated_points_allowed.set_visible(advanced);
        self.ltf_lower_target_for_clustered.set_visible(advanced);
        self.btn_3dmod_clustered_elongated_model
            .set_visible(advanced);

        self.pack();
    }

    /// Java `updateEnabled()`.
    pub fn update_enabled(&self) {
        let transferfid_enabled = self.transferfid_enabled.get();
        if let Some(pnl_transferfid) = &self.pnl_transferfid {
            pnl_transferfid.set_enabled(transferfid_enabled);
        }
        if !transferfid_enabled && self.rb_seed_model_transfer.is_enabled() {
            self.rb_seed_model_transfer.set_enabled(false);
            self.rb_seed_model_auto.set_selected_boolean(true);
            self.update_method();
        } else if transferfid_enabled {
            self.rb_seed_model_transfer.set_enabled(true);
        }
        let manager: &'static dyn BaseManager = self.application_manager;
        self.btn_3dmod_initial_bead_finding
            .set_enabled(autofidseed_init_file_filter::exists(manager, self.axis_id));
        let models_exist = autofidseed_selection_and_sorting::exists(manager, self.axis_id);
        self.btn_3dmod_bead_selection_and_sorting
            .set_enabled(models_exist);
        self.ltf_ignore_surface_data.set_enabled(models_exist);
        let exists = file_type::CLASS
            .clustered_elongated_model
            .exists(Some(manager), Some(self.axis_id));
        self.btn_3dmod_clustered_elongated_model.set_enabled(exists);
        // If track_adjusted.com is present along with an message about adjusting the track
        // comfile.
        // Java evaluates `FileType.TRACK_ADJUSTED_COMSCRIPT.exists(...) && ...` with
        // short-circuit: the log is only read when the com file exists.
        let result: Result<bool, LogFileError> = if file_type::CLASS
            .track_adjusted_comscript
            .exists(Some(manager), Some(self.axis_id))
        {
            AutofidseedLog::get_instance(
                manager,
                self.axis_id,
                self.application_manager.get_property_user_dir().as_deref(),
            )
            .is_tracking_adjusted()
        } else {
            Ok(false)
        };
        match result {
            Ok(enabled) => self.btn_use_adjusted_track_com.set_enabled(enabled),
            Err(LogFileError::Lock(_)) => {
                self.btn_use_adjusted_track_com.set_enabled(true);
            }
            Err(e) => {
                eprintln!("{e}");
                // Something went wrong. Enabling button in case the functionality should be
                // available.
                self.btn_use_adjusted_track_com.set_enabled(true);
            }
        }
    }

    /// Java `setBeadtrackParams(BeadtrackParam, boolean)`.  Set the parameters
    /// for the specified beadtrack panel.
    pub fn set_beadtrack_params(
        &self,
        beadtrack_params: &mut BeadtrackParam,
        for_transfer_fid: bool,
    ) {
        if !for_transfer_fid {
            if let Some(raptor_panel) = &self.raptor_panel {
                raptor_panel.set_beadtrack_params(beadtrack_params);
            }
        }
        self.pnl_beadtrack
            .set_parameters_beadtrack_param_boolean(beadtrack_params, for_transfer_fid);
    }

    /// Java `setTransferFidParams()`.
    pub fn set_transfer_fid_params(&self) {
        if let Some(pnl_transferfid) = &self.pnl_transferfid {
            pnl_transferfid.set_parameters_void();
        }
    }

    /// Java `getBeadTrackDisplay()`.
    pub fn get_bead_track_display(&self) -> Rc<dyn BeadTrackDisplay> {
        self.pnl_beadtrack.clone()
    }

    /// Java `getTiltxcorrDisplay()`.
    pub fn get_tiltxcorr_display(&self) -> Rc<dyn TiltXcorrDisplay> {
        self.tiltxcorr_panel.clone()
    }

    /// Java `getParameters(BaseScreenState)`.
    pub fn get_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.pnl_beadtrack
            .get_parameters_base_screen_state(screen_state);
        if let Some(pnl_transferfid) = &self.pnl_transferfid {
            pnl_transferfid.get_parameters_base_screen_state(screen_state);
        }
    }

    /// Java `getParameters(RunraptorParam, boolean)`.
    pub fn get_parameters_runraptor_param_boolean(
        &self,
        param: &mut RunraptorParam,
        do_validation: bool,
    ) -> bool {
        if let Some(raptor_panel) = &self.raptor_panel {
            return raptor_panel.get_parameters_runraptor_param_boolean(param, do_validation);
        }
        true
    }

    /// Java `getParameters(MetaData)`.
    ///
    /// Upstream bug fixed in translation (FiducialModelDialog.java:810-811):
    /// Java dereferences `curSeedAndTrackTab` / `curRunRaptorTab`
    /// unconditionally; they are always set once `createPanel` has run, and a
    /// null tab here is skipped instead of throwing NullPointerException.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        if let Some(raptor_panel) = &self.raptor_panel {
            raptor_panel.get_parameters_meta_data(meta_data);
        }
        let enumerated_type = (self.bg_method.get_selection())
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<RadioButtonModel>()
                    .and_then(|model| model.get_enumerated_type())
            })
            .expect("bgMethod has a selected RadioButtonModel");
        meta_data.set_track_method(self.axis_id, Some(&(*enumerated_type).to_string()));
        self.tiltxcorr_panel.get_parameters_meta_data(meta_data);
        meta_data
            .set_track_seed_model_manual(self.rb_seed_model_manual.is_selected(), self.axis_id);
        meta_data.set_track_seed_model_auto(self.rb_seed_model_auto.is_selected(), self.axis_id);
        meta_data
            .set_track_seed_model_transfer(self.rb_seed_model_transfer.is_selected(), self.axis_id);
        meta_data.set_track_exclude_inside_areas(
            self.cb_exclude_inside_areas.is_selected(),
            self.axis_id,
        );
        meta_data.set_track_just_find_shifts_near_zero(
            self.ltf_just_find_shifts_near_zero
                .get_text_void()
                .as_deref(),
            self.axis_id,
        );
        meta_data.set_track_target_number_of_beads(
            self.rtf_target_number_of_beads.get_text_void().as_deref(),
            self.axis_id,
        );
        meta_data.set_track_target_density_of_beads(
            self.rtf_target_density_of_beads.get_text_void().as_deref(),
            self.axis_id,
        );
        meta_data.set_track_elongated_points_allowed(
            self.axis_id,
            Some(self.cbs_elongated_points_allowed.get_value()),
        );
        meta_data.set_track_lower_target_for_clustered(
            self.axis_id,
            self.ltf_lower_target_for_clustered
                .get_text_void()
                .as_deref(),
        );
        meta_data.set_track_advanced(self.btn_advanced.is_expanded(), self.axis_id);
        if let Some(cur) = self.cur_seed_and_track_tab.get() {
            meta_data.set_seed_and_track_tab(self.axis_id, cur.index);
        }
        if let Some(cur) = self.cur_run_raptor_tab.get() {
            meta_data.set_raptor_tab(self.axis_id, cur.index);
        }
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        let method: Option<TrackingMethod> =
            TrackingMethod::get_instance(Some(&meta_data.get_track_method(self.axis_id)));
        if method == Some(tracking_method::SEED) {
            self.rb_method_seed.set_selected_boolean(true);
        } else if method == Some(tracking_method::PATCH_TRACKING) {
            self.rb_method_patch_tracking.set_selected_boolean(true);
        } else if self.axis_id != AxisID::Second && method == Some(tracking_method::RAPTOR) {
            self.rb_method_raptor.set_selected_boolean(true);
        }
        if let Some(raptor_panel) = &self.raptor_panel {
            raptor_panel.set_parameters(meta_data);
        }
        self.tiltxcorr_panel
            .set_parameters_const_meta_data(meta_data);
        if meta_data.is_track_seed_model_manual(self.axis_id) {
            self.rb_seed_model_manual.set_selected_boolean(true);
        }
        if meta_data.is_track_seed_model_auto(self.axis_id) {
            self.rb_seed_model_auto.set_selected_boolean(true);
        }
        if meta_data.is_track_seed_model_transfer(self.axis_id) {
            self.rb_seed_model_transfer.set_selected_boolean(true);
        }
        self.ltf_just_find_shifts_near_zero.set_text_string(Some(
            &meta_data.get_track_just_find_shifts_near_zero(self.axis_id),
        ));
        self.cb_exclude_inside_areas
            .set_selected_boolean(meta_data.is_track_exclude_inside_areas(self.axis_id));
        self.rtf_target_number_of_beads.set_text_string(Some(
            &meta_data.get_track_target_number_of_beads(self.axis_id),
        ));
        self.rtf_target_density_of_beads.set_text_string(Some(
            &meta_data.get_track_target_density_of_beads(self.axis_id),
        ));
        // backwards compatibility
        self.cbs_elongated_points_allowed
            .set_selected(meta_data.is_track_clustered_points_allowed_elongated(self.axis_id));
        self.cbs_elongated_points_allowed.set_value_int(
            meta_data.get_track_clustered_points_allowed_elongated_value(self.axis_id),
        );
        if !meta_data.is_track_elongated_points_allowed_null(self.axis_id) {
            self.cbs_elongated_points_allowed
                .set_value_const_etomo_number(
                    &meta_data.get_track_elongated_points_allowed(self.axis_id),
                );
        }
        self.ltf_lower_target_for_clustered.set_text_string(Some(
            &meta_data.get_track_lower_target_for_clustered(self.axis_id),
        ));
        self.btn_advanced
            .change_state(meta_data.is_track_advanced(self.axis_id));
        // SeedAndTrackTab.getInstance never returns null.
        let seed_and_track_tab =
            SeedAndTrackTab::get_instance(meta_data.get_seed_and_track_tab(self.axis_id));
        self.tp_seed_and_track
            .get_component()
            .set_selected_tab(seed_and_track_tab.index);
        self.change_seed_and_track_tab();
        let run_raptor_tab = RunRaptorTab::get_instance(meta_data.get_raptor_tab(self.axis_id));
        self.tp_run_raptor
            .get_component()
            .set_selected_tab(run_raptor_tab.index);
        self.change_run_raptor_tab();
        self.update_method();
        self.update_seed_model();
        self.update_advanced_void();
    }

    /// Java `setParameters(ConstTiltxcorrParam)`.
    pub fn set_parameters_const_tiltxcorr_param(
        &self,
        tilt_xcorr_params: &dyn ConstTiltxcorrParam,
    ) {
        self.tiltxcorr_panel
            .set_parameters_const_tiltxcorr_param(tilt_xcorr_params);
    }

    /// Java `setParameters(ImodchopcontsParam)`.
    pub fn set_parameters_imodchopconts_param(&self, param: &ImodchopcontsParam) {
        self.tiltxcorr_panel
            .set_parameters_imodchopconts_param(param);
    }

    /// Java `setParameters(AutofidseedParam, boolean)`.
    pub fn set_parameters_autofidseed_param_boolean(
        &self,
        param: &AutofidseedParam,
        for_transfer_fid: bool,
    ) {
        self.cb_adjust_sizes
            .set_selected_boolean(param.is_adjust_sizes());
        self.ltf_min_guess_num_beads
            .set_text_string(Some(&param.get_min_guess_num_beads()));
        self.ltf_min_spacing
            .set_text_string(Some(&param.get_min_spacing()));
        self.ltf_peak_storage_fraction
            .set_text_string(Some(&param.get_peak_storage_fraction()));
        if param.is_target_number_of_beads() {
            self.rtf_target_number_of_beads.set_selected_boolean(true);
            self.rtf_target_number_of_beads
                .set_text_string(Some(&param.get_target_number_of_beads()));
        }
        if param.is_target_density_of_beads() {
            self.rtf_target_density_of_beads.set_selected_boolean(true);
            self.rtf_target_density_of_beads
                .set_text_string(Some(&param.get_target_density_of_beads()));
        }
        self.cb_two_surfaces
            .set_selected_boolean(param.is_two_surfaces());
        self.ltf_max_major_to_minor_ratio
            .set_text_string(Some(&param.get_max_major_to_minor_ratio()));
        self.cb_clustered_points_allowed_clustered
            .set_selected_boolean(param.is_clustered_points_allowed());
        if self.cb_clustered_points_allowed_clustered.is_selected() {
            let cpa = param.get_clustered_points_allowed();
            // Backwards compatibility
            if let Some(cpa) = cpa
                && cpa.is_elongated()
            {
                self.cbs_elongated_points_allowed.set_selected(true);
                self.cbs_elongated_points_allowed
                    .set_value_int(cpa.convert_to_display_value());
            }
        }
        if param.is_elongated_points_allowed_set() {
            self.cbs_elongated_points_allowed.set_selected(true);
            self.cbs_elongated_points_allowed
                .set_value_const_etomo_number(param.get_elongated_points_allowed());
        } else {
            self.cbs_elongated_points_allowed.set_selected(false);
        }
        if param.is_lower_target_for_clustered() {
            Field::set_value_string(
                &*self.ltf_lower_target_for_clustered,
                Some(&param.get_lower_target_for_clustered()),
            );
        }
        if !for_transfer_fid {
            self.cb_boundary_model
                .set_selected_boolean(param.is_boundary_model());
            if self.cb_exclude_inside_areas.is_enabled() {
                self.cb_exclude_inside_areas
                    .set_selected_boolean(param.is_exclude_inside_areas());
            }
            self.ltf_borders_in_xand_y
                .set_text_string(Some(&param.get_borders_in_xand_y()));
            self.cb_append_to_seed_model
                .set_selected_boolean(param.is_append_to_seed_model());
            self.ltf_ignore_surface_data
                .set_text_string(Some(&param.get_ignore_surface_data()));
            self.ltf_drop_tracks
                .set_text_string(Some(&param.get_drop_tracks()));
        }
        self.update_display();
        self.update_enabled();
    }

    /// Java `getParameters(AutofidseedParam, boolean, boolean) throws
    /// FortranInputSyntaxException`.
    pub fn get_parameters_autofidseed_param_boolean_boolean(
        &self,
        param: &mut AutofidseedParam,
        just_find_shifts_near_zero: bool,
        do_validation: bool,
    ) -> Result<bool, FortranInputSyntaxException> {
        /// The exceptions the Java `try` block sees.
        enum Thrown {
            FieldValidationFailed,
            FortranInputSyntax(FortranInputSyntaxException),
        }
        impl From<FieldValidationFailedException> for Thrown {
            fn from(_: FieldValidationFailedException) -> Thrown {
                Thrown::FieldValidationFailed
            }
        }
        impl From<FortranInputSyntaxException> for Thrown {
            fn from(except: FortranInputSyntaxException) -> Thrown {
                Thrown::FortranInputSyntax(except)
            }
        }
        // Java try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| -> Result<bool, Thrown> {
            // Just find shifts near zero tilt
            if just_find_shifts_near_zero {
                param.set_just_find_shifts_near_zero(
                    self.ltf_just_find_shifts_near_zero
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                param.reset_just_find_shifts_near_zero();
            }

            // Generate seed model automatically
            param.set_min_guess_num_beads(
                self.ltf_min_guess_num_beads
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            param.set_min_spacing(
                self.ltf_min_spacing
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            param.set_peak_storage_fraction(
                self.ltf_peak_storage_fraction
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            param.set_boundary_model(self.cb_boundary_model.is_selected());
            param.set_exclude_inside_areas(
                self.cb_exclude_inside_areas.is_enabled()
                    && self.cb_exclude_inside_areas.is_selected(),
            );
            param.set_adjust_sizes(self.cb_adjust_sizes.is_selected());
            param.set_borders_in_xand_y(
                self.ltf_borders_in_xand_y
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            )?;
            param.set_two_surfaces(self.cb_two_surfaces.is_selected());
            param.set_append_to_seed_model(self.cb_append_to_seed_model.is_selected());
            if self.rtf_target_number_of_beads.is_selected() {
                param.set_target_number_of_beads(
                    self.rtf_target_number_of_beads
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                param.reset_target_number_of_beads();
            }
            if self.rtf_target_density_of_beads.is_selected() {
                param.set_target_density_of_beads(
                    self.rtf_target_density_of_beads
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                param.reset_target_density_of_beads();
            }
            param.set_max_major_to_minor_ratio(
                self.ltf_max_major_to_minor_ratio
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            param.set_clustered_points_allowed(
                self.cb_clustered_points_allowed_clustered.is_selected(),
            );
            if self.cbs_elongated_points_allowed.is_selected() {
                param.set_elongated_points_allowed(Some(
                    self.cbs_elongated_points_allowed.get_value(),
                ));
            } else {
                param.reset_elongated_points_allowed();
            }
            if self.ltf_lower_target_for_clustered.is_enabled() {
                param.set_lower_target_for_clustered(
                    self.ltf_lower_target_for_clustered
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                param.reset_lower_target_for_clustered();
            }
            param.set_ignore_surface_data(
                self.ltf_ignore_surface_data
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            param.set_drop_tracks(
                self.ltf_drop_tracks
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );

            Ok(true)
        })();
        match result {
            Ok(value) => Ok(value),
            Err(Thrown::FieldValidationFailed) => Ok(false),
            Err(Thrown::FortranInputSyntax(except)) => Err(except),
        }
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        if let Some(pnl_transferfid) = &self.pnl_transferfid {
            pnl_transferfid.set_parameters_recon_screen_state(screen_state);
        }
        self.pnl_beadtrack
            .set_parameters_base_screen_state(screen_state);
    }

    /// Java `getTransferFidParams(boolean)`.
    pub fn get_transfer_fid_params_boolean(&self, do_validation: bool) -> bool {
        if let Some(pnl_transferfid) = &self.pnl_transferfid {
            return pnl_transferfid.get_parameters_boolean(do_validation);
        }
        true
    }

    /// Java `getTransferFidParams(TransferfidParam, boolean)`.
    pub fn get_transfer_fid_params_transferfid_param_boolean(
        &self,
        transfer_fid_param: &mut TransferfidParam,
        do_validation: bool,
    ) -> bool {
        if let Some(pnl_transferfid) = &self.pnl_transferfid {
            return pnl_transferfid
                .get_parameters_transferfid_param_boolean(transfer_fid_param, do_validation);
        }
        true
    }

    /// Java `setTransferfidEnabled(boolean)`.
    pub fn set_transferfid_enabled(&self, file_exists: bool) {
        self.transferfid_enabled.set(file_exists);
    }

    /// Java private `setToolTipText()`.  Tooltip string initialization.
    fn set_tool_tip_text(&self) {
        self.btn_use_adjusted_track_com.set_tool_tip_text(Some(
            "For tracking this seed, use the com file with an adjusted bead size or information on large shifts between views.",
        ));
        self.rb_method_seed.set_tool_tip_text_string(Some(
            "Create a seed model and use beadtracker to generate the fiducial model.",
        ));
        self.rb_method_patch_tracking
            .set_tool_tip_text_string(Some("Create the fiducial model with patch tracking."));
        self.rb_method_raptor
            .set_tool_tip_text_string(Some("Use RAPTOR to create the fiducial model."));
        self.btn_seed
            .set_tool_tip_text(Some("Open new or existing seed model in 3dmod."));
        self.rb_seed_model_manual.set_tool_tip_text_string(Some(
            "Open 3dmod to create a seed model by selecting fiducials manually.",
        ));
        self.rb_seed_model_auto.set_tool_tip_text_string(Some(
            "Use Autofidseed to select a fiducial seed model automatically.",
        ));
        self.rb_seed_model_transfer.set_tool_tip_text_string(Some(
            "Create a seed mode by transferring the fiducial selection from the other axis.",
        ));
        self.btn_3dmod_autofidseed.set_tool_tip_text(Some(
            "Open the model of seed points selected by Autofidseed.",
        ));
        self.btn_3dmod_initial_bead_finding.set_tool_tip_text(Some(
            "Open model of beads found by Imodfindbeads on all of the views analyzed.",
        ));
        self.btn_3dmod_bead_selection_and_sorting
            .set_tool_tip_text(Some(
                "Open 3D models of beads sorted onto two surfaces in the different Beadtrack runs.",
            ));
        self.btn_cleanup
            .set_tool_tip_text(Some("Delete the temporary directory."));
        self.btn_autofidseed.set_tool_tip_text(Some(
            "Run Autofidseed to find beads, track them through 11 views, and select a seed model.",
        ));
        self.cbs_elongated_points_allowed.set_spinner_tool_tip_text(Some(
            "Select 1, 2, or 3 to include beads identified as elongated in up to 1/3, up to 2/3, or all of the Beadtrack runs, respectively",
        ));
        self.btn_3dmod_clustered_elongated_model.set_tool_tip_text(Some(
            "Open a model with all beads that are candidates for selection, color-coded by whether they are clustered or elongated",
        ));
        let mut autodoc: Option<*mut Autodoc> = None;
        // SAFETY: `AutodocFactory` owns every autodoc it returns for the life
        // of the process (the Java singletons), so the pointer stays valid for
        // this method.
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.application_manager),
                Some(autodoc_factory::AUTOFIDSEED),
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
        // SAFETY: see above; the autodoc outlives this method.
        let autodoc: Option<&dyn ReadOnlyAutodoc> =
            autodoc.map(|autodoc| unsafe { &*autodoc } as &dyn ReadOnlyAutodoc);
        let tooltip =
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::BOUNDARY_MODEL_KEY));
        self.cb_boundary_model
            .set_tool_tip_text_string(tooltip.as_deref());
        self.btn_boundary_model.set_tool_tip_text(Some(
            "Open 3dmod to create or edit a model with contours around areas to include or exclude",
        ));
        self.cb_exclude_inside_areas.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::EXCLUDE_INSIDE_AREAS_KEY))
                .as_deref(),
        );
        self.cb_adjust_sizes.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::ADJUST_SIZES_KEY))
                .as_deref(),
        );
        self.ltf_borders_in_xand_y.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::BORDERS_IN_X_AND_Y_KEY))
                .as_deref(),
        );
        self.ltf_min_guess_num_beads.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::MIN_GUESS_NUM_BEADS_KEY))
                .as_deref(),
        );
        self.ltf_min_spacing.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::MIN_SPACING_KEY))
                .as_deref(),
        );
        self.ltf_peak_storage_fraction.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::PEAK_STORAGE_FRACTION_KEY))
                .as_deref(),
        );
        Field::set_tool_tip_text(
            &*self.rtf_target_number_of_beads,
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(autofidseed_param::TARGET_NUMBER_OF_BEADS_KEY),
            )
            .as_deref(),
        );
        Field::set_tool_tip_text(
            &*self.rtf_target_density_of_beads,
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(autofidseed_param::TARGET_DENSITY_OF_BEADS_KEY),
            )
            .as_deref(),
        );
        self.cb_two_surfaces.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::TWO_SURFACES_KEY))
                .as_deref(),
        );
        self.cb_append_to_seed_model.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::APPEND_TO_SEED_MODEL_KEY))
                .as_deref(),
        );
        self.ltf_ignore_surface_data.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::IGNORE_SURFACE_DATA_KEY))
                .as_deref(),
        );
        self.ltf_drop_tracks.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(autofidseed_param::DROP_TRACKS_KEY))
                .as_deref(),
        );
        self.ltf_max_major_to_minor_ratio.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(autofidseed_param::MAX_MAJOR_TO_MINOR_RATIO_KEY),
            )
            .as_deref(),
        );
        self.cb_clustered_points_allowed_clustered
            .set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(autofidseed_param::CLUSTERED_POINTS_ALLOWED_KEY),
                )
                .as_deref(),
            );
        self.cbs_elongated_points_allowed
            .set_check_box_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(autofidseed_param::ELONGATED_POINTS_ALLOWED_KEY),
                )
                .as_deref(),
            );
        self.ltf_lower_target_for_clustered.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(autofidseed_param::LOWER_TARGET_FOR_CLUSTERED_KEY),
            )
            .as_deref(),
        );
        self.ltf_just_find_shifts_near_zero.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(autofidseed_param::JUST_FIND_SHIFTS_NEAR_ZERO_KEY),
            )
            .as_deref(),
        );
    }
}

impl ProcessDialogVirtual for FiducialModelDialog {
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java `done()`.
    fn done(&self) {
        self.application_manager
            .done_fiducial_model_dialog(self.axis_id);
        if let Some(pnl_transferfid) = &self.pnl_transferfid {
            pnl_transferfid.done();
        }
        if let Some(raptor_panel) = &self.raptor_panel {
            raptor_panel.done();
        }
        self.btn_seed.remove_action_listener(&self.action_listener);
        self.btn_autofidseed
            .remove_action_listener(&self.action_listener);
        self.btn_use_adjusted_track_com
            .remove_action_listener(&self.action_listener);
        self.btn_just_find_shifts_near_zero
            .remove_action_listener(&self.action_listener);
        self.pnl_beadtrack.done();
        self.tiltxcorr_panel.done();
        self.set_displayed(false);
    }
}

impl Expandable for FiducialModelDialog {
    /// Java `expand(ExpandButton)`: empty.
    fn expand_expand_button(&self, _button: &Rc<ExpandButton>) {}

    /// Java `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.update_advanced_boolean(button.is_expanded());
    }
}

impl ContextMenu for FiducialModelDialog {
    /// Java `popUpContextMenu(MouseEvent)`: right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel: Vec<String>;
        let man_page: Vec<String>;
        let autofidseed = self.cur_method_index.get()
            == tracking_method::SEED.get_value().get_int()
            && self.cur_seed_and_track_tab.get() == Some(SeedAndTrackTab::SEED)
            && self.cur_seed_model_index.get() == SeedModelEnumeratedType::AUTO.value;
        if autofidseed {
            man_pagelabel = [
                "Autofidseed",
                "Imodfindbeads",
                "Beadtrack",
                "Sortbeadsurfs",
                "Pickbestseed",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            man_page = [
                "autofidseed.html",
                "imodfindbeads.html",
                "beadtrack.html",
                "sortbeadsurfs.html",
                "pickbestseed.html",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
        } else {
            man_pagelabel = ["Autofidseed", "Beadtrack", "Transferfid", "3dmod"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            man_page = [
                "autofidseed.html",
                "beadtrack.html",
                "transferfid.html",
                "3dmod.html",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
        }

        let log_file_label: Vec<String> = ["Autofidseed", "Track", "Transferfid"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut log_file: Vec<String> = vec![String::new(); 3];
        log_file[0] = format!("autofidseed{}.log", self.axis_id.get_extension());
        log_file[1] = format!("track{}.log", self.axis_id.get_extension());
        log_file[2] = "transferfid.log".to_string();

        let anchor = if autofidseed {
            "AutomaticSeed"
        } else {
            "GETTING FIDUCIAL"
        };
        let manager: &'static dyn BaseManager = self.application_manager;
        // Java `new ContextPopup(...)`; its constructor throws
        // IllegalArgumentException on mismatched array lengths, which these
        // arrays cannot have.
        if let Err(message) = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.root_panel.get_component(),
            mouse_event,
            Some(anchor),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(log_file_label.as_slice()),
            Some(log_file.as_slice()),
            manager,
            self.axis_id,
        ) {
            eprintln!("java.lang.IllegalArgumentException: {message}");
        }
    }
}

impl Run3dmodButtonContainer for FiducialModelDialog {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`: action
    /// function for buttons.
    ///
    /// The action listener passes a null `Run3dmodMenuOptions`; the manager
    /// takes the options by value, and a default instance (every option off)
    /// is what the 3dmod layer makes of null.
    fn action(
        &self,
        command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let menu_options = run_3dmod_menu_options.unwrap_or_default();
        let manager: &'static dyn BaseManager = self.application_manager;
        let is = |action_command: Option<String>| Some(command) == action_command.as_deref();
        if is(self.rb_method_seed.get_action_command())
            || is(self.rb_method_patch_tracking.get_action_command())
            || is(self.rb_method_raptor.get_action_command())
        {
            self.update_method();
        } else if is(self.rb_seed_model_manual.get_action_command())
            || is(self.rb_seed_model_auto.get_action_command())
            || is(self.rb_seed_model_transfer.get_action_command())
        {
            self.update_seed_model();
        } else if is(self.btn_seed.get_action_command())
            || is(self.btn_3dmod_autofidseed.get_action_command())
        {
            let seed_file_name = dataset_files::get_seed_file_name(manager, Some(self.axis_id));
            let raw_tilt_file = dataset_files::get_raw_tilt_file(manager, Some(self.axis_id));
            self.application_manager.imod_seed_model(
                self.axis_id,
                menu_options,
                Some(self.btn_seed.clone() as ProcessResultDisplayHandle),
                imod_manager::COARSE_ALIGNED_KEY,
                seed_file_name.as_deref(),
                Some(&raw_tilt_file),
                self.dialog_type,
            );
        } else if is(self.cb_boundary_model.get_action_command())
            || is(self.cb_append_to_seed_model.get_action_command())
        {
            self.update_display();
        } else if is(self.btn_autofidseed.get_action_command()) {
            if let Some(this) = self.this.upgrade() {
                // Java passes null options.
                self.application_manager.autofidseed(
                    self.axis_id,
                    Some(self.btn_autofidseed.clone() as ProcessResultDisplayHandle),
                    None,
                    Run3dmodMenuOptions::default(),
                    None,
                    DIALOG_TYPE,
                    &this,
                    false,
                );
            }
        } else if is(self.btn_just_find_shifts_near_zero.get_action_command()) {
            if let Some(this) = self.this.upgrade() {
                // Java passes null options.
                self.application_manager.autofidseed(
                    self.axis_id,
                    Some(self.btn_just_find_shifts_near_zero.clone() as ProcessResultDisplayHandle),
                    None,
                    Run3dmodMenuOptions::default(),
                    None,
                    DIALOG_TYPE,
                    &this,
                    true,
                );
            }
        } else if is(self.btn_3dmod_initial_bead_finding.get_action_command()) {
            let file_name = autofidseed_init_file_filter::get_file_name(manager, self.axis_id);
            if let Some(file_name) = file_name {
                self.application_manager.imod_coarse_align(
                    self.axis_id,
                    menu_options,
                    Some(&file_name),
                    true,
                );
            }
        } else if is(self
            .btn_3dmod_bead_selection_and_sorting
            .get_action_command())
        {
            let file_name_list =
                autofidseed_selection_and_sorting::get_file_name_list(manager, self.axis_id);
            if let Some(file_name_list) = file_name_list {
                self.application_manager.imod_sorted_models(
                    self.axis_id,
                    menu_options,
                    &file_name_list,
                );
            }
        } else if is(self
            .btn_3dmod_clustered_elongated_model
            .get_action_command())
        {
            self.application_manager
                .imod_clustered_elongated_model(self.axis_id, menu_options);
        } else if is(self.btn_cleanup.get_action_command()) {
            self.application_manager.cleanup_autofidseed(self.axis_id);
            self.update_enabled();
        } else if is(self.btn_boundary_model.get_action_command()) {
            self.application_manager.imod_model(
                &file_type::CLASS.prealigned_stack,
                &file_type::CLASS.autofidseed_boundary_model,
                self.axis_id,
                menu_options,
                true,
                true,
            );
        } else if is(self.btn_use_adjusted_track_com.get_action_command()) {
            self.application_manager.use_track_adjusted_comfile(
                self.axis_id,
                Some(self.btn_use_adjusted_track_com.clone() as ProcessResultDisplayHandle),
            );
            self.update_enabled();
        }
        self.update_display();
    }
}

/// Java `public static final class SeedModelEnumeratedType implements
/// EnumeratedType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedModelEnumeratedType {
    /// Java private final `isDefault`.
    is_default: bool,
    /// Java private final `value` (an `EtomoNumber` set to this int).
    value: i32,
    /// Java private final `string`.
    string: &'static str,
}

impl SeedModelEnumeratedType {
    /// Java private static final `MANUAL`.
    const MANUAL: SeedModelEnumeratedType = SeedModelEnumeratedType {
        is_default: true,
        value: 0,
        string: "Manual",
    };
    /// Java private static final `AUTO`.
    const AUTO: SeedModelEnumeratedType = SeedModelEnumeratedType {
        is_default: false,
        value: 1,
        string: "Auto",
    };
    /// Java public static final `TRANSFER`.
    pub const TRANSFER: SeedModelEnumeratedType = SeedModelEnumeratedType {
        is_default: false,
        value: 2,
        string: "Transfer",
    };

    /// Java private static final `NUM`.
    const NUM: i32 = 3;

    /// Java private static `getInstance(String)` (unused in the source).
    #[allow(dead_code)]
    fn get_instance(string: Option<&str>) -> Option<SeedModelEnumeratedType> {
        let string = string?;
        if string == Self::MANUAL.string {
            return Some(Self::MANUAL);
        }
        if string == Self::AUTO.string {
            return Some(Self::AUTO);
        }
        if string == Self::TRANSFER.string {
            return Some(Self::TRANSFER);
        }
        None
    }
}

impl EnumeratedType for SeedModelEnumeratedType {
    /// Java `isDefault()`.
    fn is_default(&self) -> bool {
        self.is_default
    }

    /// Java `getValue()`.
    fn get_value(&self) -> ConstEtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(self.value);
        value.base
    }

    /// Java `getLabel()`.
    fn get_label(&self) -> Option<String> {
        None
    }
}

/// Java `toString()`.
impl std::fmt::Display for SeedModelEnumeratedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.string)
    }
}

/// Java private static final class `SeedAndTrackTab`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SeedAndTrackTab {
    /// Java private final `index`.
    index: i32,
    /// Java private final `title`.
    title: &'static str,
}

impl SeedAndTrackTab {
    /// Java private static final `SEED`.
    const SEED: SeedAndTrackTab = SeedAndTrackTab {
        index: 0,
        title: "Seed Model",
    };
    /// Java private static final `TRACK`.
    const TRACK: SeedAndTrackTab = SeedAndTrackTab {
        index: 1,
        title: "Track Beads",
    };

    /// Java private static final `DEFAULT = SEED`.
    const DEFAULT: SeedAndTrackTab = Self::SEED;

    /// Java private static final `NUM_TABS`.
    const NUM_TABS: i32 = 2;

    /// Java private static `getInstance(int)`.
    fn get_instance(index: i32) -> SeedAndTrackTab {
        if index == Self::SEED.index {
            return Self::SEED;
        }
        if index == Self::TRACK.index {
            return Self::TRACK;
        }
        Self::DEFAULT
    }

    /// Java `equals(int)`.
    fn equals(&self, index: i32) -> bool {
        index == self.index
    }
}

/// Java `toString()`.
impl std::fmt::Display for SeedAndTrackTab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[index:{},title:{}]", self.index, self.title)
    }
}

/// Java private static final class `RunRaptorTab`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunRaptorTab {
    /// Java private final `index`.
    index: i32,
    /// Java private final `title`.
    title: &'static str,
}

impl RunRaptorTab {
    /// Java private static final `RAPTOR`.
    const RAPTOR: RunRaptorTab = RunRaptorTab {
        index: 0,
        title: "Run RAPTOR",
    };
    /// Java private static final `TRACK`.
    const TRACK: RunRaptorTab = RunRaptorTab {
        index: 1,
        title: "Track Beads",
    };

    /// Java private static final `DEFAULT = RAPTOR`.
    const DEFAULT: RunRaptorTab = Self::RAPTOR;

    /// Java private static final `NUM_TABS`.
    const NUM_TABS: i32 = 2;

    /// Java private static `getInstance(int)`.
    fn get_instance(index: i32) -> RunRaptorTab {
        if index == Self::RAPTOR.index {
            return Self::RAPTOR;
        }
        if index == Self::TRACK.index {
            return Self::TRACK;
        }
        Self::DEFAULT
    }

    /// Java `equals(int)`.
    fn equals(&self, index: i32) -> bool {
        index == self.index
    }
}
