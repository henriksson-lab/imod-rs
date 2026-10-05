//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoDatasetDialog.java`.
//!
//! The dataset values of the batchruntomo interface: the global instance is the body of
//! the dialog's "Dataset" tab; a row instance is a separate window ("<dir>/<name>
//! Dataset Values") opened from a row of the dataset table.  The "Advanced" button
//! replaces the basic fields with a `DirectivesDialog`.  An event dispatch thread
//! object, created as `Rc<Self>` by the `get_*instance` functions.
//!
//! The private `makeList(String, String)` has no caller in the Java and is not
//! translated (DEAD_CODE.md).
//!
//! **Windows.**  The row instance's `JFrame` (the private static class `Frame`, which
//! hides the window on `WINDOW_CLOSING`) is a non-modal `JDialog` of the Swing stand-in
//! with that close behaviour; its menu bar's "View" menu is the first child of the
//! window's content pane.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::batch_run_tomo_dialog::BatchRunTomoDialog;
use super::batch_run_tomo_row::BatchRunTomoRow;
use super::check_box::CheckBox;
use super::check_text_field::CheckTextField;
use super::directives_dialog::DirectivesDialog;
use super::ebutton::Ebutton;
use super::etched_border::EtchedBorder;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::file_text_field2::FileTextField2;
use super::fixed_dim;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::LabeledTextField;
use super::panel_header::PanelHeader;
use super::popup::Popup;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::radio_text_field::RadioTextField;
use super::single_line_button::SingleLineButton;
use super::spacer::Spacer;
use super::swing_component::SwingComponent;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent, JDialog, WindowListener};
use crate::imod::etomo::logic::autodoc_attribute_retriever;
use crate::imod::etomo::logic::batch_tool::{self, TemplateValues};
use crate::imod::etomo::logic::config_tool;
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::logic::seeding_method::{self, SeedingMethod};
use crate::imod::etomo::logic::tracking_method::{self, TrackingMethod};
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::storage::directive_file_interface::DirectiveFileInterface;
use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::file_text_field_interface::FileTextFieldInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::batch_run_tomo_dataset_meta_data::{self, BatchRunTomoDatasetMetaData};
use crate::imod::etomo::r#type::batch_run_tomo_status::{self, BatchRunTomoStatus};
use crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::erase_gold::EraseGold;
use crate::imod::etomo::r#type::const_etomo_number::Type as EtomoNumberType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::frame_status::FrameStatus;
use crate::imod::etomo::r#type::sample_type::SampleType;
use crate::imod::etomo::r#type::status::StatusRef;
use crate::imod::etomo::r#type::status_change_boolean_event::StatusChangeBooleanEvent;
use crate::imod::etomo::r#type::status_change_event::StatusChangeEvent;
use crate::imod::etomo::ui::batch_run_tomo_tab::BatchRunTomoTab;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::table_listener::TableListener;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java private static final `LENGTH_OF_PIECES_DEFAULT`.
const LENGTH_OF_PIECES_DEFAULT: &str = "-1";
/// Java private static final `SCALE_TO_INTEGER_VALUE`.
const SCALE_TO_INTEGER_VALUE: &str = "-20000,20000";
/// Java private static final `CTF_RANGE_DEFAULT`.
const CTF_RANGE_DEFAULT: &str = "0.0";
/// Java private static final `CTF_STEP_DEFAULT`.
const CTF_STEP_DEFAULT: &str = "0.0";
/// Java private static final `AXIS_ID`.
const AXIS_ID: AxisID = AxisID::Only;
/// Java private static final `TARGET_NUMBER_OF_BEADS_LABEL`.
const TARGET_NUMBER_OF_BEADS_LABEL: &str = "Target number of beads: ";

/// Java private static final `CTF_RANGE_FIT_EVERY_IMAGE` (a double `EtomoNumber` every
/// constructor sets to 1).
fn ctf_range_fit_every_image() -> EtomoNumber {
    let mut number = EtomoNumber::new_with_type(Some(EtomoNumberType::Double));
    number.set_int(1);
    number
}

/// Java private static final `CTF_STEP_FIT_EVERY_IMAGE` (a double `EtomoNumber` every
/// constructor sets to 0).
fn ctf_step_fit_every_image() -> EtomoNumber {
    let mut number = EtomoNumber::new_with_type(Some(EtomoNumberType::Double));
    number.set_int(0);
    number
}

/// Java `final class BatchRunTomoDatasetDialog implements ActionListener, Expandable,
/// TableListener, UIComponent, SwingComponent, FieldDisplayer`.
pub struct BatchRunTomoDatasetDialog {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `cbRemoveXrays`.
    cb_remove_xrays: Rc<CheckBox>,
    /// Java private final `cbEnableStretching`.
    cb_enable_stretching: Rc<CheckBox>,
    /// Java private final `cbLocalAlignments`.
    cb_local_alignments: Rc<CheckBox>,
    /// Java private final `btnModelFile`.
    btn_model_file: Rc<SingleLineButton>,
    /// Java private final `bgTrackingMethod`.
    bg_tracking_method: Rc<ButtonGroup>,
    /// Java private final `rbTrackingMethodSeed`.
    rb_tracking_method_seed: Rc<RadioButton>,
    /// Java private final `rbTrackingMethodRaptor`.
    rb_tracking_method_raptor: Rc<RadioButton>,
    /// Java private final `rbTrackingMethodPatchTracking`.
    rb_tracking_method_patch_tracking: Rc<RadioButton>,
    /// Java private final `rbFiducialless`.
    rb_fiducialless: Rc<RadioButton>,
    /// Java private final `ltfGold`.
    ltf_gold: Rc<LabeledTextField>,
    /// Java private final `ltfTargetNumberOfBeads`.
    ltf_target_number_of_beads: Rc<LabeledTextField>,
    /// Java private final `ltfNumberOfMarkers`.
    ltf_number_of_markers: Rc<LabeledTextField>,
    /// Java private final `ltfSizeOfPatchesXandY`.
    ltf_size_of_patches_x_and_y: Rc<LabeledTextField>,
    /// Java private final `cbLengthOfPieces`.
    cb_length_of_pieces: Rc<CheckBox>,
    /// Java private final `lsBinByFactor`.
    ls_bin_by_factor: Rc<LabeledSpinner>,
    /// Java private final `cbCorrectCTF`.
    cb_correct_ctf: Rc<CheckBox>,
    /// Java private final `ltfScanDefocusRange`.
    ltf_scan_defocus_range: Rc<LabeledTextField>,
    /// Java private final `ltfDefocus`.
    ltf_defocus: Rc<LabeledTextField>,
    /// Java private final `bgAutofit`.
    #[allow(dead_code)]
    bg_autofit: Rc<ButtonGroup>,
    /// Java private final `rbFitEveryImage`.
    rb_fit_every_image: Rc<RadioButton>,
    /// Java private final `rtfAutoFitRangeAndStep`.
    rtf_auto_fit_range_and_step: Rc<RadioTextField>,
    /// Java private final `ltfAutoFitStep`.
    ltf_auto_fit_step: Rc<LabeledTextField>,
    /// Java private final `cbDoBackprojAlso`.
    cb_do_backproj_also: Rc<CheckBox>,
    /// Java private final `cbFakeSIRTiterations`.
    cb_fake_sirt_iterations: Rc<CheckBox>,
    /// Java private final `ltfFakeSIRTiterations`.
    ltf_fake_sirt_iterations: Rc<LabeledTextField>,
    /// Java private final `cbUseSirt`.
    cb_use_sirt: Rc<CheckBox>,
    /// Java private final `ltfLeaveIterations`.
    ltf_leave_iterations: Rc<LabeledTextField>,
    /// Java private final `cbScaleToInteger`.
    cb_scale_to_integer: Rc<CheckBox>,
    /// Java private final `bgThickness`.
    #[allow(dead_code)]
    bg_thickness: Rc<ButtonGroup>,
    /// Java private final `rtfThickness`.
    rtf_thickness: Rc<RadioTextField>,
    /// Java private final `rtfBinnedThickness`.
    rtf_binned_thickness: Rc<RadioTextField>,
    /// Java private final `rbFallbackAndExtraThickness`.
    rb_fallback_and_extra_thickness: Rc<RadioButton>,
    /// Java private final `ltfExtraThickness`.
    ltf_extra_thickness: Rc<LabeledTextField>,
    /// Java private final `ltfFallbackThickness`.
    ltf_fallback_thickness: Rc<LabeledTextField>,
    /// Java private final `fieldList`.
    field_list: RefCell<Vec<Rc<dyn Field>>>,
    /// Java private final `pnlRootBody`.
    pnl_root_body: Rc<JComponent>,
    /// Java private final `spaceModelFile`.
    space_model_file: Spacer,
    /// Java private final `lsPrenewstBinByFactor`.
    ls_prenewst_bin_by_factor: Rc<LabeledSpinner>,
    /// Java private final `lsPreblendBinByFactor`.
    ls_preblend_bin_by_factor: Rc<LabeledSpinner>,
    /// Java private final `cbSampleType`.
    cb_sample_type: Rc<CheckBox>,
    /// Java private final `cbDoTrimvol`.
    cb_do_trimvol: Rc<CheckBox>,
    /// Java private final `pnlPostprocessingBody`.
    pnl_postprocessing_body: Rc<JComponent>,
    /// Java private final `ctfFindSecAddThickness`.
    ctf_find_sec_add_thickness: Rc<CheckTextField>,
    /// Java private final `ctfScaleFromZ`.
    ctf_scale_from_z: Rc<CheckTextField>,
    /// Java private final `cbEraseGold`.
    cb_erase_gold: Rc<CheckBox>,
    /// Java private final `bgEraseGold`.
    bg_erase_gold: Rc<ButtonGroup>,
    /// Java private final `rbEraseGoldFid`.
    rb_erase_gold_fid: Rc<RadioButton>,
    /// Java private final `rbEraseGold3d`.
    rb_erase_gold_3d: Rc<RadioButton>,
    /// Java private final `ltfGoldErasingThickness`.
    ltf_gold_erasing_thickness: Rc<LabeledTextField>,
    /// Java private final `bgSampleType`.
    bg_sample_type: Rc<ButtonGroup>,
    /// Java private final `rbSampleTypePlasticSection`.
    rb_sample_type_plastic_section: Rc<RadioButton>,
    /// Java private final `rbSampleTypeCryo`.
    rb_sample_type_cryo: Rc<RadioButton>,
    /// Java private final `ltfPositioningThickness`.
    ltf_positioning_thickness: Rc<LabeledTextField>,
    /// Java private final `cbHasGoldBeads`.
    cb_has_gold_beads: Rc<CheckBox>,
    /// Java private final `ltfPositioningGold`.
    ltf_positioning_gold: Rc<LabeledTextField>,
    /// Java private final `sDistortion`.
    s_distortion: Rc<JComponent>,
    /// Java private final `btnAdvanced`.
    btn_advanced: Rc<Ebutton>,
    /// Java private final `btnBasic`.
    btn_basic: Rc<Ebutton>,
    /// Java private final `pnlBasic`.
    pnl_basic: Rc<JComponent>,
    /// Java private final `phRoot`.
    ph_root: Rc<PanelHeader>,
    /// Java private final `advancedFieldDisplayer`.
    advanced_field_displayer: Rc<AdvancedFieldDisplayer>,
    /// Java private final `cbTuneFittingAndSampling`.
    cb_tune_fitting_and_sampling: Rc<CheckBox>,
    /// Java private final `pnlGold`.
    pnl_gold: Rc<JComponent>,

    /// Java private final `ftfDistort`.
    ftf_distort: Option<Rc<FileTextField2>>,
    /// Java private final `ftfGradient`.
    ftf_gradient: Option<Rc<FileTextField2>>,
    /// Java private final `ftfModelFile`.
    ftf_model_file: Rc<FileTextField2>,
    /// Java private final `frame` (not used in the global instance).
    frame: Option<Rc<JDialog>>,
    /// Java private final `menuFitWindow`.
    menu_fit_window: Option<Rc<JComponent>>,
    /// Java private final `manager`.
    manager: &'static BatchRunTomoManager,
    /// Java private final `datasetFile`.
    dataset_file: Option<PathBuf>,
    /// Java private final `parent`.
    parent: Option<Weak<BatchRunTomoDialog>>,
    /// Java private final `btnOk`.
    btn_ok: Option<Rc<SingleLineButton>>,
    /// Java private final `btnRevertToGlobal`.
    btn_revert_to_global: Option<Rc<SingleLineButton>>,
    /// Java private final `phPostprocessing`.
    ph_postprocessing: Rc<PanelHeader>,
    /// Java private final `templateValues`.
    template_values: Option<Rc<RefCell<TemplateValues>>>,
    /// Java private final `browsingDir`.
    browsing_dir: Option<Weak<dyn BrowsingDirectory>>,
    /// Java private final `stackID`.
    stack_id: Option<String>,
    /// Java private final `basicDirectives`.
    basic_directives: Option<Rc<RefCell<HashSet<DirectiveDef>>>>,
    /// Java private final `global`.
    global: bool,
    /// Java private final `fromSaved`.
    from_saved: bool,

    /// Java private `lengthOfPieces`, initially null.
    length_of_pieces: RefCell<Option<String>>,
    /// Java private `status`, initially `BatchRunTomoStatus.DEFAULT`.
    status: Cell<Option<BatchRunTomoStatus>>,
    /// Java private `directivesDialog`, initially null.
    directives_dialog: RefCell<Option<Rc<DirectivesDialog>>>,

    /// Java private `row`.
    row: RefCell<Option<Weak<BatchRunTomoRow>>>,
    /// Java private `emptyTable`.
    empty_table: Cell<bool>,
    /// Java private `fiducialModelMode`, initially true.
    fiducial_model_mode: Cell<bool>,
    /// Java private `shiftBasicButton`, initially false.
    shift_basic_button: Cell<bool>,
    /// Java private `advanced`, initially false.
    advanced: Cell<bool>,
    /// Java `this`.
    this: Weak<BatchRunTomoDatasetDialog>,
}

/// Java private final inner class `AdvancedFieldDisplayer implements FieldDisplayer`.
pub struct AdvancedFieldDisplayer {
    /// Java private final `dialog`.
    dialog: Weak<BatchRunTomoDatasetDialog>,
}

impl FieldDisplayer for AdvancedFieldDisplayer {
    /// Java `display()`.
    fn display_void(&self) {
        if let Some(dialog) = self.dialog.upgrade() {
            dialog.display_advanced(true);
        }
    }

    /// Java `display(UIComponent)`.
    fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {
        self.display_void();
    }
}

/// The private static class `Frame extends JFrame`'s `processWindowEvent` override:
/// `WINDOW_CLOSING` hides the dialog.
struct FrameCloser(Weak<BatchRunTomoDatasetDialog>);

impl WindowListener for FrameCloser {
    fn window_closing(&self) {
        if let Some(dialog) = self.0.upgrade() {
            dialog.set_visible(false);
        }
    }
}

impl BatchRunTomoDatasetDialog {
    /// Java private `BatchRunTomoDatasetDialog(BatchRunTomoManager, File, boolean,
    /// BatchRunTomoRow, boolean, BatchRunTomoDialog, Map<DirectiveDef, String>,
    /// Set<DirectiveDef>, BrowsingDirectory, String, boolean)`, with the field
    /// initialisers.
    #[allow(clippy::too_many_arguments)]
    fn new(
        manager: &'static BatchRunTomoManager,
        dataset_file: Option<PathBuf>,
        global: bool,
        row: Option<Weak<BatchRunTomoRow>>,
        empty_table: bool,
        parent: Option<Weak<BatchRunTomoDialog>>,
        template_values: Option<Rc<RefCell<TemplateValues>>>,
        basic_directives: Option<Rc<RefCell<HashSet<DirectiveDef>>>>,
        browsing_dir: Option<Weak<dyn BrowsingDirectory>>,
        stack_id: Option<&str>,
        from_saved: bool,
    ) -> Rc<BatchRunTomoDatasetDialog> {
        let base_manager: &'static dyn BaseManager = manager;
        let bg_tracking_method = ButtonGroup::new();
        let bg_autofit = ButtonGroup::new();
        let bg_thickness = ButtonGroup::new();
        let bg_erase_gold = ButtonGroup::new();
        let bg_sample_type = ButtonGroup::new();
        // CTF_RANGE_FIT_EVERY_IMAGE.set(1); CTF_STEP_FIT_EVERY_IMAGE.set(0): see
        // ctf_range_fit_every_image / ctf_step_fit_every_image.
        let distortion_dir = dataset_files::get_distortion_dir(
            Some(base_manager),
            manager.get_property_user_dir().as_deref(),
            Some(AXIS_ID),
        );
        let mut shift_basic_button = false;
        let (ftf_distort, ftf_gradient) =
            if distortion_dir.as_ref().is_some_and(|dir| dir.exists()) {
                (
                    Some(FileTextField2::get_alt_layout_instance(Some(base_manager), Some("Image distortion file: "))),
                    Some(FileTextField2::get_alt_layout_instance(Some(base_manager), Some("Mag gradient file: "))),
                )
            } else {
                shift_basic_button = true;
                (None, None)
            };
        let ftf_model_file =
            FileTextField2::get_alt_layout_instance(Some(base_manager), Some("Manual replacement model: "));
        Rc::new_cyclic(|this: &Weak<BatchRunTomoDatasetDialog>| {
            let expandable: Weak<dyn Expandable> = this.clone();
            let ph_root = PanelHeader::get_instance(
                Some("Global Dataset Values"),
                Some(expandable.clone()),
                Some(DialogType::BatchRunTomo),
            );
            let ph_postprocessing = PanelHeader::get_instance(
                Some("Postprocessing"),
                Some(expandable),
                Some(DialogType::BatchRunTomo),
            );
            let (frame, menu_fit_window, btn_ok, btn_revert_to_global) = if global {
                (None, None, None, None)
            } else {
                let frame = JDialog::new("", false);
                // `setDefaultCloseOperation(DO_NOTHING_ON_CLOSE)`.
                frame.set_default_close_operation(crate::imod::etomo::jdk::DO_NOTHING_ON_CLOSE);
                frame.add_window_listener(Rc::new(FrameCloser(this.clone())));
                let menu_fit_window = JComponent::new_menu_item("Fit Window");
                ph_root.set_text_string(Some(" Dataset Values"));
                (
                    Some(frame),
                    Some(menu_fit_window),
                    Some(SingleLineButton::new_string(Some("OK"))),
                    Some(SingleLineButton::new_string(Some("Revert to Global"))),
                )
            };
            BatchRunTomoDatasetDialog {
                pnl_root: JComponent::new_panel(),
                cb_remove_xrays: CheckBox::new_string(Some("Remove X-rays")),
                cb_enable_stretching: CheckBox::new_string(Some(
                    "Enable distortion (stretching) in alignment",
                )),
                cb_local_alignments: CheckBox::new_string(Some("Use local alignments")),
                btn_model_file: SingleLineButton::new_string(Some("Make in 3dmod")),
                rb_tracking_method_seed: RadioButton::new_string_enumerated_type_button_group(
                    Some("Autoseed and track"),
                    Some(EnumeratedTypeRef::new(tracking_method::SEED)),
                    Some(&bg_tracking_method),
                ),
                rb_tracking_method_raptor: RadioButton::new_string_enumerated_type_button_group(
                    Some("Raptor and track"),
                    Some(EnumeratedTypeRef::new(tracking_method::RAPTOR)),
                    Some(&bg_tracking_method),
                ),
                rb_tracking_method_patch_tracking:
                    RadioButton::new_string_enumerated_type_button_group(
                        Some("Patch tracking"),
                        Some(EnumeratedTypeRef::new(tracking_method::PATCH_TRACKING)),
                        Some(&bg_tracking_method),
                    ),
                rb_fiducialless: RadioButton::new_string_button_group(
                    Some("Coarse alignment only"),
                    Some(&bg_tracking_method),
                ),
                bg_tracking_method: bg_tracking_method.clone(),
                ltf_gold: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Bead size (nm): "),
                ),
                ltf_target_number_of_beads: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(TARGET_NUMBER_OF_BEADS_LABEL),
                ),
                ltf_number_of_markers: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(TARGET_NUMBER_OF_BEADS_LABEL),
                ),
                ltf_size_of_patches_x_and_y: LabeledTextField::new_field_type_string(
                    FieldType::IntegerPair,
                    Some("Patch tracking size: "),
                ),
                cb_length_of_pieces: CheckBox::new_string(Some("Break contours into pieces")),
                ls_bin_by_factor: LabeledSpinner::get_instance_string_int_int_int_int(
                    Some("Aligned stack binning: "),
                    1,
                    1,
                    8,
                    1,
                ),
                cb_correct_ctf: CheckBox::new_string(Some("Correct CTF")),
                ltf_scan_defocus_range: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPointPair,
                    Some("Defocus range to scan: "),
                ),
                ltf_defocus: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Defocus: "),
                ),
                rb_fit_every_image: RadioButton::new_string_button_group(
                    Some("Fit every image"),
                    Some(&bg_autofit),
                ),
                rtf_auto_fit_range_and_step: RadioTextField::get_instance_field_type_string_button_group(
                    FieldType::FloatingPoint,
                    Some("Autofit range "),
                    Some(&bg_autofit),
                ),
                bg_autofit: bg_autofit.clone(),
                ltf_auto_fit_step: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some(" and step "),
                ),
                cb_do_backproj_also: CheckBox::new_string(Some("R-weighted backprojection")),
                cb_fake_sirt_iterations: CheckBox::new_string(Some("SIRT-like filter")),
                ltf_fake_sirt_iterations: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Equivalent iterations: "),
                ),
                cb_use_sirt: CheckBox::new_string(Some("SIRT")),
                ltf_leave_iterations: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some("Leave iterations: "),
                ),
                cb_scale_to_integer: CheckBox::new_string(Some("Scale to integers")),
                rtf_thickness: RadioTextField::get_instance_field_type_string_button_group(
                    FieldType::Integer,
                    Some("Thickness total (unbinned pixels): "),
                    Some(&bg_thickness),
                ),
                rtf_binned_thickness: RadioTextField::get_instance_field_type_string_button_group(
                    FieldType::Integer,
                    Some("Thickness total (binned pixels): "),
                    Some(&bg_thickness),
                ),
                rb_fallback_and_extra_thickness: RadioButton::new_string_button_group(
                    Some("Calculated thickness (unbinned pixels):"),
                    Some(&bg_thickness),
                ),
                bg_thickness: bg_thickness.clone(),
                ltf_extra_thickness: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("          Plus (optional): "),
                ),
                ltf_fallback_thickness: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("          With fallback: "),
                ),
                field_list: RefCell::new(Vec::new()),
                pnl_root_body: JComponent::new_panel(),
                space_model_file: Spacer::new(fixed_dim::x5_y0),
                ls_prenewst_bin_by_factor: LabeledSpinner::get_instance_string_int_int_int_int(
                    Some("Coarse aligned stack binning - single frame: "),
                    1,
                    1,
                    8,
                    1,
                ),
                ls_preblend_bin_by_factor: LabeledSpinner::get_instance_string_int_int_int_int(
                    Some("Montage: "),
                    1,
                    1,
                    8,
                    1,
                ),
                cb_sample_type: CheckBox::new_string(Some("Do positioning for:")),
                cb_do_trimvol: CheckBox::new_string(Some("Postprocess with trimvol")),
                pnl_postprocessing_body: JComponent::new_panel(),
                ctf_find_sec_add_thickness: CheckTextField::get_instance(
                    FieldType::FloatingPoint,
                    "Find plastic section limits and add: ",
                ),
                ctf_scale_from_z: CheckTextField::get_instance(
                    FieldType::FloatingPoint,
                    "Fraction of Z slices to analyze:",
                ),
                cb_erase_gold: CheckBox::new_string(Some("Erase gold")),
                rb_erase_gold_fid: RadioButton::new_string_enumerated_type_button_group(
                    Some("Use fiducial model"),
                    Some(EnumeratedTypeRef::new(EraseGold::Fid)),
                    Some(&bg_erase_gold),
                ),
                rb_erase_gold_3d: RadioButton::new_string_enumerated_type_button_group(
                    Some("Find beads in 3D"),
                    Some(EnumeratedTypeRef::new(EraseGold::Find3d)),
                    Some(&bg_erase_gold),
                ),
                bg_erase_gold: bg_erase_gold.clone(),
                ltf_gold_erasing_thickness: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Tomogram thickness (pixels): "),
                ),
                rb_sample_type_plastic_section:
                    RadioButton::new_string_enumerated_type_button_group(
                        Some("Plastic section"),
                        Some(EnumeratedTypeRef::new(SampleType::PlasticSection)),
                        Some(&bg_sample_type),
                    ),
                rb_sample_type_cryo: RadioButton::new_string_enumerated_type_button_group(
                    Some("Cryo sample"),
                    Some(EnumeratedTypeRef::new(SampleType::Cryo)),
                    Some(&bg_sample_type),
                ),
                bg_sample_type: bg_sample_type.clone(),
                ltf_positioning_thickness: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Tomogram thickness: "),
                ),
                cb_has_gold_beads: CheckBox::new_string(Some("Sample has gold beads")),
                ltf_positioning_gold: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Bead size (nm): "),
                ),
                s_distortion: JComponent::new_other(),
                btn_advanced: Ebutton::get_single_line_instance(Some("Advanced")),
                btn_basic: Ebutton::get_single_line_instance(Some("Basic")),
                pnl_basic: JComponent::new_panel(),
                ph_root,
                advanced_field_displayer: Rc::new(AdvancedFieldDisplayer {
                    dialog: this.clone(),
                }),
                cb_tune_fitting_and_sampling: CheckBox::new_string(Some("Autotune")),
                pnl_gold: JComponent::new_panel(),
                ftf_distort,
                ftf_gradient,
                ftf_model_file,
                frame,
                menu_fit_window,
                manager,
                dataset_file,
                parent,
                btn_ok,
                btn_revert_to_global,
                ph_postprocessing,
                template_values,
                browsing_dir,
                stack_id: stack_id.map(str::to_owned),
                basic_directives,
                global,
                from_saved,
                length_of_pieces: RefCell::new(None),
                status: Cell::new(Some(batch_run_tomo_status::DEFAULT)),
                directives_dialog: RefCell::new(None),
                row: RefCell::new(row),
                empty_table: Cell::new(empty_table),
                fiducial_model_mode: Cell::new(true),
                shift_basic_button: Cell::new(shift_basic_button),
                advanced: Cell::new(false),
                this: this.clone(),
            }
        })
    }

    /// Java package-private static `getGlobalInstance(BatchRunTomoManager,
    /// BatchRunTomoDialog, Map<DirectiveDef, String>, Set<DirectiveDef>,
    /// BrowsingDirectory)`.
    pub fn get_global_instance(
        manager: &'static BatchRunTomoManager,
        parent: Weak<BatchRunTomoDialog>,
        template_values: Option<Rc<RefCell<TemplateValues>>>,
        basic_directives: Option<Rc<RefCell<HashSet<DirectiveDef>>>>,
        browsing_dir: Option<Weak<dyn BrowsingDirectory>>,
    ) -> Rc<BatchRunTomoDatasetDialog> {
        utilities::timestamp_full(
            Some("new"),
            Some("BatchRunTomoDatasetDialog getGlobalInstance"),
            None,
            Some(utilities::STARTED_STATUS),
        );
        let instance = BatchRunTomoDatasetDialog::new(
            manager,
            None,
            true,
            None,
            true,
            Some(parent),
            template_values,
            basic_directives,
            browsing_dir,
            None,
            true,
        );
        instance.create_panel(true);
        instance.set_tooltips();
        instance.add_listeners();
        utilities::timestamp_full(
            Some("new"),
            Some("BatchRunTomoDatasetDialog getGlobalInstance"),
            None,
            Some(utilities::FINISHED_STATUS),
        );
        instance
    }

    /// Java package-private static `getRowInstance(BatchRunTomoManager, File,
    /// BatchRunTomoRow, Map<DirectiveDef, String>, Set<DirectiveDef>, BrowsingDirectory,
    /// String)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_row_instance(
        manager: &'static BatchRunTomoManager,
        dataset_file: Option<PathBuf>,
        row: Weak<BatchRunTomoRow>,
        template_values: Option<Rc<RefCell<TemplateValues>>>,
        basic_directives: Option<Rc<RefCell<HashSet<DirectiveDef>>>>,
        browsing_dir: Option<Weak<dyn BrowsingDirectory>>,
        stack_id: Option<&str>,
    ) -> Rc<BatchRunTomoDatasetDialog> {
        utilities::timestamp_full(
            Some("new"),
            Some("BatchRunTomoDatasetDialog getRowInstance"),
            None,
            Some(utilities::STARTED_STATUS),
        );
        let instance = BatchRunTomoDatasetDialog::new(
            manager,
            dataset_file,
            false,
            Some(row),
            false,
            None,
            template_values,
            basic_directives,
            browsing_dir,
            stack_id,
            false,
        );
        instance.create_panel(false);
        instance.set_tooltips();
        instance.add_listeners();
        instance.set_visible(true);
        utilities::timestamp_full(
            Some("new"),
            Some("BatchRunTomoDatasetDialog getRowInstance"),
            None,
            Some(utilities::FINISHED_STATUS),
        );
        instance
    }

    /// Java package-private static `getSavedRowInstance(BatchRunTomoManager, File,
    /// BatchRunTomoRow, Map<DirectiveDef, String>, Set<DirectiveDef>, BrowsingDirectory,
    /// String)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_saved_row_instance(
        manager: &'static BatchRunTomoManager,
        dataset_file: Option<PathBuf>,
        row: Weak<BatchRunTomoRow>,
        template_values: Option<Rc<RefCell<TemplateValues>>>,
        basic_directives: Option<Rc<RefCell<HashSet<DirectiveDef>>>>,
        browsing_dir: Option<Weak<dyn BrowsingDirectory>>,
        stack_id: Option<&str>,
    ) -> Rc<BatchRunTomoDatasetDialog> {
        let instance = BatchRunTomoDatasetDialog::new(
            manager,
            dataset_file,
            false,
            Some(row),
            false,
            None,
            template_values,
            basic_directives,
            browsing_dir,
            stack_id,
            true,
        );
        instance.create_panel(false);
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    fn base_manager(&self) -> &'static dyn BaseManager {
        self.manager
    }

    fn parent(&self) -> Option<Rc<BatchRunTomoDialog>> {
        self.parent.as_ref().and_then(Weak::upgrade)
    }

    fn row(&self) -> Option<Rc<BatchRunTomoRow>> {
        self.row.borrow().as_ref().and_then(Weak::upgrade)
    }

    /// This dialog as the `FieldDisplayer` Java passes as `this`.
    fn field_displayer(&self) -> Option<Rc<dyn FieldDisplayer>> {
        self.this
            .upgrade()
            .map(|this| this as Rc<dyn FieldDisplayer>)
    }

    /// Java package-private `isFindSecAddThicknessSet()`.
    pub fn is_find_sec_add_thickness_set(&self) -> bool {
        self.ctf_find_sec_add_thickness.is_selected() && !self.ctf_find_sec_add_thickness.is_empty()
    }

    /// Java package-private `isScaleFromZSet()`.
    pub fn is_scale_from_z_set(&self) -> bool {
        self.ctf_scale_from_z.is_selected() && !self.ctf_scale_from_z.is_empty()
    }

    /// Java package-private `hasDual()`.
    pub fn has_dual(&self) -> bool {
        if let Some(row) = self.row() {
            return row.is_dual();
        }
        self.parent().is_some_and(|parent| parent.has_dual())
    }

    /// Java private `createPanel(boolean)`.
    fn create_panel(&self, _global: bool) {
        // local panels
        let pnl_model_file = JComponent::new_panel();
        let pnl_tracking_method = JComponent::new_panel();
        let pnl_size_of_patches_x_and_y = JComponent::new_panel();
        let pnl_bin_by_factor = JComponent::new_panel();
        let pnl_aligned_stack = JComponent::new_panel();
        let pnl_reconstruction = JComponent::new_panel();
        let pnl_reconstruction_type = JComponent::new_panel();
        let pnl_thickness = JComponent::new_panel();
        let pnl_auto_fit_range_and_step = JComponent::new_panel();
        let pnl_defocus = JComponent::new_panel();
        let pnl_buttons = self.frame.as_ref().map(|_| JComponent::new_panel());
        let pnl_remove_xrays = JComponent::new_panel();
        let pnl_enable_stretching = JComponent::new_panel();
        let pnl_local_alignments = JComponent::new_panel();
        let pnl_pre_bin_by_factor = JComponent::new_panel();
        let pnl_postprocessing = JComponent::new_panel();
        let pnl_do_trimvol = JComponent::new_panel();
        let pnl_trimvol = JComponent::new_panel();
        let pnl_sample_type = JComponent::new_panel();
        let pnl_sample_type_chooser = JComponent::new_panel();
        let pnl_sample_type_data = JComponent::new_panel();
        let pnl_distort = self.ftf_distort.as_ref().map(|_| JComponent::new_panel());
        // init
        self.rb_tracking_method_seed.set_selected_boolean(true);
        // Swing layout: btnModelFile, btnRevertToGlobal and btnOk setToPreferredSize().
        if let Some(ftf_distort) = &self.ftf_distort {
            ftf_distort.set_origin_string(
                config_tool::get_distortion_dir(self.base_manager(), None).as_deref(),
            );
            ftf_distort.set_absolute_path(true);
        }
        if let Some(ftf_gradient) = &self.ftf_gradient {
            ftf_gradient.set_preferred_width(272.0);
            ftf_gradient.set_origin_string(
                config_tool::get_distortion_dir(self.base_manager(), None).as_deref(),
            );
            ftf_gradient.set_absolute_path(true);
        }
        self.ftf_model_file.set_absolute_path(true);
        self.rtf_auto_fit_range_and_step.set_required(true);
        self.ltf_auto_fit_step.set_required(true);
        self.ltf_gold.set_required(true);
        self.ltf_positioning_gold.set_required(true);
        self.ltf_positioning_thickness.set_columns(3);
        self.ltf_target_number_of_beads.set_required(true);
        self.ltf_number_of_markers.set_required(true);
        self.ltf_size_of_patches_x_and_y.set_required(true);
        self.ltf_fake_sirt_iterations.set_required(true);
        self.ltf_leave_iterations.set_required(true);
        self.ltf_fallback_thickness.set_required(true);
        self.rtf_thickness.set_required(true);
        self.rtf_binned_thickness.set_required(true);
        self.ctf_scale_from_z
            .set_text_string(Some(batch_run_tomo_dataset_meta_data::SCALE_FROM_Z_DEFAULT));
        self.ph_postprocessing.set_open(false);
        self.ctf_find_sec_add_thickness.set_required(true);
        self.rb_erase_gold_3d.set_selected_boolean(true);
        self.rtf_auto_fit_range_and_step.set_selected_boolean(true);
        self.cb_do_backproj_also.set_selected_boolean(true);
        // Set directive defs where there is a one-to-one correspondence between field
        // and directive. The directive def is used for setting the default in comparam
        // directives, and saving fields to autodoc files.
        //
        // IMPORTANT: Code currently assumes that all defaults (for comparam directive
        // fields) can be set generically.
        self.setup_field(self.ftf_distort.clone().map(|f| f as Rc<dyn Field>), Some(DirectiveDef::DISTORT), None);
        self.setup_field(self.ftf_gradient.clone().map(|f| f as Rc<dyn Field>), Some(DirectiveDef::GRADIENT), None);
        self.setup_field(Some(self.cb_remove_xrays.clone()), Some(DirectiveDef::REMOVE_XRAYS), None);
        self.setup_field(Some(self.ftf_model_file.clone()), Some(DirectiveDef::MODEL_FILE), None);
        // manual - also affects seedingMethod
        self.setup_field(
            Some(self.rb_tracking_method_seed.clone()),
            Some(DirectiveDef::TRACKING_METHOD),
            Some(DirectiveDef::SEEDING_METHOD),
        );
        self.setup_field(Some(self.rb_tracking_method_raptor.clone()), Some(DirectiveDef::TRACKING_METHOD), None);
        self.setup_field(
            Some(self.rb_tracking_method_patch_tracking.clone()),
            Some(DirectiveDef::TRACKING_METHOD),
            None,
        );
        self.setup_field(Some(self.rb_fiducialless.clone()), Some(DirectiveDef::FIDUCIALLESS), None);
        self.setup_field(Some(self.ltf_gold.clone()), Some(DirectiveDef::GOLD), None);
        // Hiding targetDensityOfBeads
        self.setup_field(
            Some(self.ltf_target_number_of_beads.clone()),
            Some(DirectiveDef::TARGET_NUMBER_OF_BEADS),
            Some(DirectiveDef::TARGET_DENSITY_OF_BEADS),
        );
        self.setup_field(Some(self.ltf_number_of_markers.clone()), Some(DirectiveDef::NUMBER_OF_MARKERS), None);
        self.setup_field(
            Some(self.ltf_size_of_patches_x_and_y.clone()),
            Some(DirectiveDef::SIZE_OF_PATCHES_X_AND_Y),
            None,
        );
        self.setup_field(Some(self.cb_length_of_pieces.clone()), Some(DirectiveDef::LENGTH_OF_PIECES), None);
        self.setup_field(Some(self.cb_enable_stretching.clone()), Some(DirectiveDef::ENABLE_STRETCHING), None);
        self.setup_field(Some(self.cb_local_alignments.clone()), Some(DirectiveDef::LOCAL_ALIGNMENTS), None);
        self.setup_field(
            Some(self.ls_bin_by_factor.clone()),
            Some(DirectiveDef::BIN_BY_FACTOR_FOR_ALIGNED_STACK),
            None,
        );
        self.setup_field(Some(self.cb_correct_ctf.clone()), Some(DirectiveDef::CORRECT_CTF), None);
        self.setup_field(Some(self.ltf_scan_defocus_range.clone()), Some(DirectiveDef::SCAN_DEFOCUS_RANGE), None);
        self.setup_field(Some(self.ltf_defocus.clone()), Some(DirectiveDef::DEFOCUS), None);
        self.setup_field(
            Some(self.rtf_auto_fit_range_and_step.clone()),
            Some(DirectiveDef::AUTO_FIT_RANGE_AND_STEP),
            None,
        ); // manual
        self.setup_field(Some(self.ltf_auto_fit_step.clone()), Some(DirectiveDef::AUTO_FIT_RANGE_AND_STEP), None); // manual
        self.setup_field(Some(self.rb_fit_every_image.clone()), Some(DirectiveDef::AUTO_FIT_RANGE_AND_STEP), None); // manual
        self.setup_field(Some(self.cb_sample_type.clone()), Some(DirectiveDef::SAMPLE_TYPE), None);
        self.setup_field(Some(self.cb_do_backproj_also.clone()), Some(DirectiveDef::DO_BACKPROJ_ALSO), None);
        self.setup_field(Some(self.cb_fake_sirt_iterations.clone()), Some(DirectiveDef::FAKE_SIRT_ITERATIONS), None);
        self.setup_field(Some(self.ltf_fake_sirt_iterations.clone()), Some(DirectiveDef::FAKE_SIRT_ITERATIONS), None);
        self.setup_field(Some(self.cb_use_sirt.clone()), Some(DirectiveDef::USE_SIRT), None);
        self.setup_field(Some(self.ltf_leave_iterations.clone()), Some(DirectiveDef::LEAVE_ITERATIONS), None);
        self.setup_field(Some(self.cb_scale_to_integer.clone()), Some(DirectiveDef::SCALE_TO_INTEGER), None);
        self.setup_field(Some(self.rtf_thickness.clone()), Some(DirectiveDef::THICKNESS_FOR_TILT), None); // manual
        self.setup_field(Some(self.rtf_binned_thickness.clone()), Some(DirectiveDef::BINNED_THICKNESS), None); // manual
        self.setup_field(
            Some(self.rb_fallback_and_extra_thickness.clone()),
            Some(DirectiveDef::FALLBACK_THICKNESS),
            Some(DirectiveDef::EXTRA_THICKNESS),
        );
        self.setup_field(Some(self.ltf_extra_thickness.clone()), Some(DirectiveDef::EXTRA_THICKNESS), None); // manual
        self.setup_field(Some(self.ltf_fallback_thickness.clone()), Some(DirectiveDef::FALLBACK_THICKNESS), None); // manual
        self.setup_field(
            Some(self.ls_prenewst_bin_by_factor.clone()),
            Some(DirectiveDef::BIN_BY_FACTOR_FOR_PRENEWST),
            None,
        );
        self.setup_field(
            Some(self.ls_preblend_bin_by_factor.clone()),
            Some(DirectiveDef::BIN_BY_FACTOR_FOR_PREBLEND),
            None,
        );
        self.setup_field(Some(self.cb_do_trimvol.clone()), Some(DirectiveDef::DO_TRIMVOL), None);
        self.setup_field(
            Some(self.ctf_find_sec_add_thickness.clone()),
            Some(DirectiveDef::FIND_SEC_ADD_THICKNESS),
            None,
        );
        self.setup_field(Some(self.ctf_scale_from_z.clone()), Some(DirectiveDef::SCALE_FROM_Z), None);
        self.setup_field(Some(self.cb_erase_gold.clone()), Some(DirectiveDef::ERASE_GOLD), None);
        self.setup_field(Some(self.rb_erase_gold_fid.clone()), Some(DirectiveDef::ERASE_GOLD), None);
        self.setup_field(Some(self.rb_erase_gold_3d.clone()), Some(DirectiveDef::ERASE_GOLD), None);
        self.setup_field(
            Some(self.ltf_gold_erasing_thickness.clone()),
            Some(DirectiveDef::THICKNESS_FOR_GOLD_ERASING),
            None,
        );
        self.setup_field(Some(self.rb_sample_type_plastic_section.clone()), Some(DirectiveDef::SAMPLE_TYPE), None);
        self.setup_field(Some(self.rb_sample_type_cryo.clone()), Some(DirectiveDef::SAMPLE_TYPE), None);
        self.setup_field(
            Some(self.ltf_positioning_thickness.clone()),
            Some(DirectiveDef::THICKNESS_FOR_POSITIONING),
            None,
        );
        self.setup_field(Some(self.cb_has_gold_beads.clone()), Some(DirectiveDef::HAS_GOLD_BEADS), None);
        self.setup_field(Some(self.ltf_positioning_gold.clone()), Some(DirectiveDef::GOLD), None);
        self.setup_field(
            Some(self.cb_tune_fitting_and_sampling.clone()),
            Some(DirectiveDef::TUNE_FITTING_AND_SAMPLING),
            None,
        );
        // defaults
        self.set_individual_defaults();
        if let Some(frame) = &self.frame {
            // `new JScrollPane(pnlRoot)`; `getVerticalScrollBar().setUnitIncrement(16)`.
            let scroll_pane = JComponent::new_scroll_pane(Some(&self.pnl_root));
            // construct: `new JMenuBar()`, `new Menu("View")`.
            let menu_bar = JComponent::new_panel();
            let menu = JComponent::new_menu("View");
            // frame
            frame.get_content_pane().add(&menu_bar);
            frame.get_content_pane().add(&scroll_pane);
            // menuBar
            menu_bar.add(&menu);
            // menu: `setMnemonic(KeyEvent.VK_V)`.
            if let Some(menu_fit_window) = &self.menu_fit_window {
                menu.add(menu_fit_window);
                // menuFitWindow: `setAccelerator(ctrl F)`.
            }
            // title
            if let Some(dataset_file) = &self.dataset_file {
                let file = dataset_file
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned());
                let mut name = String::new();
                if let Some(file) = &file {
                    match file.rfind('.') {
                        Some(index) => name = file[..index].to_owned(),
                        None => name = file.clone(),
                    }
                }
                let dir_name = dataset_file
                    .parent()
                    .and_then(|dir| dir.file_name())
                    .map(|dir_name| dir_name.to_string_lossy().into_owned());
                frame.set_title(&format!(
                    "{}{}",
                    match dir_name {
                        Some(dir_name) => format!("{}{}", dir_name, std::path::MAIN_SEPARATOR),
                        None => String::new(),
                    },
                    name
                ));
            }
        }
        // root
        self.pnl_root.set_border_title(None);
        self.pnl_root.add(&self.ph_root.get_container());
        self.pnl_root.add(&self.pnl_root_body);
        // root body
        self.pnl_root_body.add(&self.pnl_basic);
        // Basic
        if let Some(pnl_distort) = &pnl_distort {
            self.pnl_basic.add(pnl_distort);
        }
        if let Some(ftf_gradient) = &self.ftf_gradient {
            self.pnl_basic.add(&ftf_gradient.get_root_panel());
        }
        self.pnl_basic.add(&self.s_distortion);
        self.pnl_basic.add(&pnl_remove_xrays);
        self.pnl_basic.add(&pnl_model_file);
        self.pnl_basic.add(&JComponent::new_other());
        self.pnl_basic.add(&pnl_pre_bin_by_factor);
        self.pnl_basic.add(&pnl_tracking_method);
        self.pnl_basic.add(&self.pnl_gold);
        self.pnl_basic.add(&pnl_size_of_patches_x_and_y);
        self.pnl_basic.add(&pnl_enable_stretching);
        self.pnl_basic.add(&pnl_local_alignments);
        self.pnl_basic.add(&JComponent::new_other());
        self.pnl_basic.add(&pnl_sample_type);
        self.pnl_basic.add(&JComponent::new_other());
        self.pnl_basic.add(&pnl_bin_by_factor);
        self.pnl_basic.add(&pnl_aligned_stack);
        self.pnl_basic.add(&pnl_reconstruction);
        self.pnl_basic.add(&pnl_postprocessing);
        if let Some(pnl_buttons) = &pnl_buttons {
            self.pnl_basic.add(pnl_buttons);
        }
        // Distort
        if let (Some(pnl_distort), Some(ftf_distort)) = (&pnl_distort, &self.ftf_distort) {
            pnl_distort.add(&ftf_distort.get_root_panel());
            pnl_distort.add(&self.btn_advanced.get_component());
        }
        // RemoveXrays
        pnl_remove_xrays.add(&self.cb_remove_xrays.get_component());
        // Alternate place to put the advanced button.
        if pnl_distort.is_none() {
            pnl_remove_xrays.add(&self.btn_advanced.get_component());
        }
        // ModelFile
        pnl_model_file.add(&self.ftf_model_file.get_root_panel());
        pnl_model_file.add(&self.space_model_file.get_component());
        pnl_model_file.add(&SwingComponent::get_component(&*self.btn_model_file));
        // PreBinByFactor
        pnl_pre_bin_by_factor.add(&self.ls_prenewst_bin_by_factor.get_container());
        pnl_pre_bin_by_factor.add(&self.ls_preblend_bin_by_factor.get_container());
        // TrackingMethod
        pnl_tracking_method.set_border_title(
            EtchedBorder::new(Some("Alignment Method"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_tracking_method.add(&self.rb_tracking_method_seed.get_component());
        pnl_tracking_method.add(&self.rb_tracking_method_patch_tracking.get_component());
        pnl_tracking_method.add(&self.rb_tracking_method_raptor.get_component());
        pnl_tracking_method.add(&self.rb_fiducialless.get_component());
        // Gold
        self.pnl_gold.add(&self.ltf_gold.get_component());
        self.update_gold_panel();
        // SizeOfPatchesXandY
        pnl_size_of_patches_x_and_y.add(&self.ltf_size_of_patches_x_and_y.get_component());
        pnl_size_of_patches_x_and_y.add(&self.cb_length_of_pieces.get_component());
        // EnableStretching
        pnl_enable_stretching.add(&self.cb_enable_stretching.get_component());
        // LocalAlignments
        pnl_local_alignments.add(&self.cb_local_alignments.get_component());
        // SampleType
        pnl_sample_type.add(&pnl_sample_type_chooser);
        pnl_sample_type.add(&pnl_sample_type_data);
        // SampleTypeChooser
        pnl_sample_type_chooser.add(&self.cb_sample_type.get_component());
        pnl_sample_type_chooser.add(&self.rb_sample_type_plastic_section.get_component());
        pnl_sample_type_chooser.add(&self.rb_sample_type_cryo.get_component());
        // SampleTypeData
        pnl_sample_type_data.add(&self.ltf_positioning_thickness.get_component());
        pnl_sample_type_data.add(&self.cb_has_gold_beads.get_component());
        pnl_sample_type_data.add(&self.ltf_positioning_gold.get_component());
        // BinByFactor
        pnl_bin_by_factor.add(&self.ls_bin_by_factor.get_container());
        // AlignedStack
        pnl_aligned_stack.add(&self.cb_correct_ctf.get_component());
        pnl_aligned_stack.add(&self.cb_erase_gold.get_component());
        pnl_aligned_stack.add(&pnl_defocus);
        pnl_aligned_stack.add(&self.rb_erase_gold_fid.get_component());
        pnl_aligned_stack.add(&self.ltf_scan_defocus_range.get_component());
        pnl_aligned_stack.add(&self.rb_erase_gold_3d.get_component());
        pnl_aligned_stack.add(&pnl_auto_fit_range_and_step);
        pnl_aligned_stack.add(&self.ltf_gold_erasing_thickness.get_component());
        pnl_aligned_stack.add(&self.rb_fit_every_image.get_component());
        // Defocus
        pnl_defocus.add(&self.ltf_defocus.get_component());
        pnl_defocus.add(&self.cb_tune_fitting_and_sampling.get_component());
        // AutoFitRangeAndStep
        pnl_auto_fit_range_and_step.add(&self.rtf_auto_fit_range_and_step.get_container());
        pnl_auto_fit_range_and_step.add(&self.ltf_auto_fit_step.get_component());
        // Reconstruction
        pnl_reconstruction.set_border_title(
            EtchedBorder::new(Some("Reconstruction"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_reconstruction.add(&pnl_reconstruction_type);
        pnl_reconstruction.add(&pnl_thickness);
        // ReconstructionType
        pnl_reconstruction_type.add(&self.cb_do_backproj_also.get_component());
        pnl_reconstruction_type.add(&self.cb_fake_sirt_iterations.get_component());
        pnl_reconstruction_type.add(&self.cb_use_sirt.get_component());
        pnl_reconstruction_type.add(&self.ltf_fake_sirt_iterations.get_component());
        pnl_reconstruction_type.add(&self.ltf_leave_iterations.get_component());
        pnl_reconstruction_type.add(&self.cb_scale_to_integer.get_component());
        // Thickness
        pnl_thickness.add(&self.rtf_thickness.get_container());
        pnl_thickness.add(&self.rtf_binned_thickness.get_container());
        pnl_thickness.add(&self.rb_fallback_and_extra_thickness.get_component());
        pnl_thickness.add(&self.ltf_fallback_thickness.get_component());
        pnl_thickness.add(&self.ltf_extra_thickness.get_component());
        // Postprocessing
        pnl_postprocessing.add(&self.ph_postprocessing.get_container());
        pnl_postprocessing.add(&self.pnl_postprocessing_body);
        // PostprocessingBody
        self.pnl_postprocessing_body.add(&pnl_do_trimvol);
        self.pnl_postprocessing_body.add(&pnl_trimvol);
        // DoTrimvol
        pnl_do_trimvol.add(&self.cb_do_trimvol.get_component());
        // Trimvol
        pnl_trimvol.add(&self.ctf_find_sec_add_thickness.get_component());
        pnl_trimvol.add(&self.ctf_scale_from_z.get_component());
        // buttons
        if let Some(pnl_buttons) = &pnl_buttons {
            if let Some(btn_ok) = &self.btn_ok {
                pnl_buttons.add(&SwingComponent::get_component(&**btn_ok));
            }
            if let Some(btn_revert_to_global) = &self.btn_revert_to_global {
                pnl_buttons.add(&SwingComponent::get_component(&**btn_revert_to_global));
            }
        }
        // update
        self.update_display();
        self.update_advanced(false, self.from_saved);
        self.status_changed_status(self.status.get().map(StatusRef::BatchRunTomoStatus));
        // display
        self.pack();
    }

    /// Java private `updateGoldPanel()`.
    fn update_gold_panel(&self) {
        if self.rb_tracking_method_seed.is_selected() {
            self.pnl_gold
                .remove(&self.ltf_number_of_markers.get_component());
            self.pnl_gold
                .add(&self.ltf_target_number_of_beads.get_component());
            self.pack();
            self.update_display();
        } else if self.rb_tracking_method_raptor.is_selected() {
            self.pnl_gold
                .remove(&self.ltf_target_number_of_beads.get_component());
            self.pnl_gold.add(&self.ltf_number_of_markers.get_component());
            self.pack();
            self.update_display();
        }
    }

    /// Java private `setupField(Field, DirectiveDef, DirectiveDef)` (and the
    /// two-argument overload).
    fn setup_field(
        &self,
        field: Option<Rc<dyn Field>>,
        directive_def: Option<DirectiveDef>,
        directive_def2: Option<DirectiveDef>,
    ) {
        let Some(field) = field else {
            return;
        };
        field.set_directive_def(directive_def);
        self.field_list.borrow_mut().push(field);
        if let Some(basic_directives) = &self.basic_directives {
            let mut basic_directives = basic_directives.borrow_mut();
            if let Some(directive_def) = directive_def
                && !basic_directives.contains(&directive_def)
            {
                basic_directives.insert(directive_def);
            }
            if let Some(directive_def2) = directive_def2
                && !basic_directives.contains(&directive_def2)
            {
                basic_directives.insert(directive_def2);
            }
        }
    }

    /// Java package-private `isAdvancedDialogExists()`.
    pub fn is_advanced_dialog_exists(&self) -> bool {
        self.directives_dialog.borrow().is_some()
    }

    /// Java package-private `isTrackingMethodSeed()`.
    pub fn is_tracking_method_seed(&self) -> bool {
        self.rb_tracking_method_seed.is_selected()
    }

    /// Java package-private `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        self.ftf_model_file.get_preferred_width()
            + self.space_model_file.get_preferred_width()
            + super::ui_utilities::get_preferred_width_abstract_button_string(
                &SwingComponent::get_component(&*self.btn_model_file),
                Some("Make in 3dmod"),
            )
    }

    /// Java package-private `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        if let Some(frame) = &self.frame {
            frame.set_visible(visible);
        }
    }

    /// This dialog as the `ActionListener` Java registers as `this`.
    fn action_listener(&self) -> ActionListener {
        let this = self.this.clone();
        Rc::new(move |event: &ActionEvent| {
            if let Some(dialog) = this.upgrade() {
                dialog.action_performed(Some(event));
            }
        })
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let listener = self.action_listener();
        self.cb_remove_xrays.add_action_listener(Some(listener.clone()));
        self.rb_tracking_method_seed.add_action_listener(listener.clone());
        self.rb_tracking_method_raptor.add_action_listener(listener.clone());
        self.rb_tracking_method_patch_tracking
            .add_action_listener(listener.clone());
        self.rb_fiducialless.add_action_listener(listener.clone());
        self.cb_correct_ctf.add_action_listener(Some(listener.clone()));
        self.rtf_auto_fit_range_and_step
            .add_action_listener(listener.clone());
        self.rb_fit_every_image.add_action_listener(listener.clone());
        self.cb_do_backproj_also
            .add_action_listener(Some(listener.clone()));
        self.cb_fake_sirt_iterations
            .add_action_listener(Some(listener.clone()));
        self.cb_use_sirt.add_action_listener(Some(listener.clone()));
        self.rtf_thickness.add_action_listener(listener.clone());
        self.rtf_binned_thickness.add_action_listener(listener.clone());
        self.rb_fallback_and_extra_thickness
            .add_action_listener(listener.clone());
        if let Some(btn_ok) = &self.btn_ok {
            btn_ok.add_action_listener(listener.clone());
        }
        if let Some(btn_revert_to_global) = &self.btn_revert_to_global {
            btn_revert_to_global.add_action_listener(listener.clone());
        }
        self.btn_model_file.add_action_listener(listener.clone());
        self.cb_do_trimvol.add_action_listener(Some(listener.clone()));
        self.cb_erase_gold.add_action_listener(Some(listener.clone()));
        self.rb_erase_gold_fid.add_action_listener(listener.clone());
        self.rb_erase_gold_3d.add_action_listener(listener.clone());
        self.cb_sample_type.add_action_listener(Some(listener.clone()));
        self.rb_sample_type_plastic_section
            .add_action_listener(listener.clone());
        self.rb_sample_type_cryo.add_action_listener(listener.clone());
        self.cb_has_gold_beads.add_action_listener(Some(listener.clone()));
        self.btn_advanced
            .add_action_listener_action_listener(Some(listener.clone()));
        self.btn_basic
            .add_action_listener_action_listener(Some(listener.clone()));
        if let Some(menu_fit_window) = &self.menu_fit_window {
            menu_fit_window.add_action_listener(listener);
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `pack()`.
    fn pack(&self) {
        match &self.frame {
            None => {
                ui_harness::with(|harness| harness.pack_base_manager(Some(self.base_manager())))
            }
            Some(frame) => frame.pack(),
        }
    }

    /// Java package-private `validate()`.
    pub fn validate(&self) -> bool {
        if self.rb_sample_type_cryo.is_selected() && self.ltf_positioning_thickness.is_empty() {
            self.display_void();
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_ui_component_string_string(
                    Some(self.base_manager()),
                    Some(&*self.ltf_positioning_thickness as &dyn UIComponent),
                    &format!(
                        "{} is required.",
                        Field::get_quoted_label(&*self.ltf_positioning_thickness)
                            .unwrap_or_else(|| "null".to_owned())
                    ),
                    "Required Field (1)",
                )
            });
            return false;
        }
        let directives_dialog = self.directives_dialog.borrow().clone();
        if let Some(directives_dialog) = directives_dialog {
            return directives_dialog.validate(Some(
                self.advanced_field_displayer.clone() as Rc<dyn FieldDisplayer>
            ));
        }
        true
    }

    /// Java package-private `validate(boolean)`.
    pub fn validate_boolean(&self, surfaces_to_analyze2: bool) -> bool {
        if !surfaces_to_analyze2
            && self.ltf_gold_erasing_thickness.is_enabled()
            && self.ltf_gold_erasing_thickness.is_empty()
        {
            self.display_void();
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_ui_component_string_string(
                    Some(self.base_manager()),
                    Some(&*self.ltf_gold_erasing_thickness as &dyn UIComponent),
                    &format!(
                        "{} is required.",
                        Field::get_quoted_label(&*self.ltf_gold_erasing_thickness)
                            .unwrap_or_else(|| "null".to_owned())
                    ),
                    "Required Field (2)",
                )
            });
            return false;
        }
        true
    }

    /// Java package-private `isParallelProcessing()`.
    pub fn is_parallel_processing(&self) -> bool {
        if let Some(parent) = self.parent() {
            return parent.is_parallel_processing();
        }
        // dataset level
        if let Some(row) = self.row() {
            return row.is_parallel_processing();
        }
        false
    }

    /// Java private `updateAdvanced(boolean, boolean)`.
    fn update_advanced(&self, advanced: bool, init: bool) {
        // A new row-level advanced dialog inherits the global state.
        // TODO This is probably from Bug# 2442. Shouldn't it inherit the global state
        // when it's a basic dialog?
        if !init && advanced && self.stack_id.is_some() {
            self.manager.save_batch_run_tomo_dialog(
                None,
                false,
                init,
                None,
                false,
                self.is_parallel_processing(),
                false,
            );
        }
        // When loading the saved instance load all the directives dialogs.
        if self.directives_dialog.borrow().is_none()
            && ((init
                && (self.stack_id.is_none()
                    || self
                        .manager
                        .is_dataset_dialog(self.stack_id.as_deref().unwrap_or_default())))
                || (!init && advanced && self.stack_id.is_some()))
        {
            utilities::timestamp_full(
                Some("new"),
                Some("DirectivesDialog"),
                None,
                Some(utilities::STARTED_STATUS),
            );
            let directives_dialog = DirectivesDialog::get_instance(
                self.manager,
                self.this.clone(),
                self.template_values.clone(),
                self.basic_directives.clone(),
                Some(&self.btn_basic),
                self.shift_basic_button.get(),
                self.browsing_dir.clone(),
            );
            *self.directives_dialog.borrow_mut() = Some(directives_dialog);
            if self.stack_id.is_none() || (!init && advanced && self.stack_id.is_some()) {
                self.manager.init_dialog(self.stack_id.as_deref(), true);
            }
        }
        if advanced {
            self.pnl_root_body.remove_all();
            // Java dereferences directivesDialog, which the block above always created
            // for an advanced request.
            if let Some(directives_dialog) = self.directives_dialog.borrow().as_ref() {
                self.pnl_root_body.add(&directives_dialog.get_component());
            }
        } else {
            self.pnl_root_body.remove_all();
            self.pnl_root_body.add(&self.pnl_basic);
        }
        self.advanced.set(advanced);
        if let Some(parent) = self.parent() {
            parent.set_dataset_table_visible(!advanced);
        }
        self.pack();
        utilities::timestamp_full(
            Some("new"),
            Some("DirectivesDialog"),
            None,
            Some(utilities::FINISHED_STATUS),
        );
    }

    /// Java package-private `getDirectivesDialog()`.
    pub fn get_directives_dialog(&self) -> Option<Rc<DirectivesDialog>> {
        self.directives_dialog.borrow().clone()
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        let remove_xrays = self.cb_remove_xrays.is_selected();
        self.ftf_model_file.set_enabled(remove_xrays);
        self.btn_model_file
            .set_enabled(remove_xrays && !self.empty_table.get());
        let fiducialless = self.rb_fiducialless.is_selected();
        self.cb_enable_stretching.set_enabled(!fiducialless);
        self.cb_local_alignments.set_enabled(!fiducialless);
        let autofidseed = self.rb_tracking_method_seed.is_selected();
        let raptor = self.rb_tracking_method_raptor.is_selected();
        let bead_tracking = autofidseed || raptor;
        self.ltf_gold.set_enabled(bead_tracking);
        self.ltf_target_number_of_beads.set_enabled(autofidseed);
        self.ltf_number_of_markers.set_enabled(raptor);
        let sample_type = self.cb_sample_type.is_selected();
        self.rb_sample_type_plastic_section.set_enabled(sample_type);
        self.rb_sample_type_cryo.set_enabled(sample_type);
        let sample_type_cryo = sample_type && self.rb_sample_type_cryo.is_selected();
        self.cb_has_gold_beads
            .set_enabled(!bead_tracking && sample_type_cryo);
        self.ltf_positioning_gold.set_enabled(
            !bead_tracking && sample_type_cryo && self.cb_has_gold_beads.is_selected(),
        );
        // Set gold and positioning values
        if bead_tracking {
            self.cb_has_gold_beads.set_selected_boolean(true);
        }
        // Transfer gold size from active field to inactive field. ltfGold is active when
        // fiducialModelMode is true. Positioning gold is active when it is false. Once the
        // gold is transfered, reset the mode.
        if self.fiducial_model_mode.get() {
            self.ltf_positioning_gold
                .set_text_string(Field::get_text_void(&*self.ltf_gold).as_deref());
        } else {
            self.ltf_gold
                .set_text_string(Field::get_text_void(&*self.ltf_positioning_gold).as_deref());
        }
        self.fiducial_model_mode.set(
            self.rb_tracking_method_seed.is_selected() || self.rb_tracking_method_raptor.is_selected(),
        );
        let patch_tracking = self.rb_tracking_method_patch_tracking.is_selected();
        self.ltf_size_of_patches_x_and_y.set_enabled(patch_tracking);
        self.cb_length_of_pieces.set_enabled(patch_tracking);
        let ctf = self.cb_correct_ctf.is_selected();
        self.ltf_scan_defocus_range.set_enabled(ctf);
        self.ltf_defocus.set_enabled(ctf);
        self.cb_tune_fitting_and_sampling.set_enabled(ctf);
        self.rtf_auto_fit_range_and_step.set_enabled(ctf);
        self.rb_fit_every_image.set_enabled(ctf);
        let auto_fit_range_and_step = ctf && self.rtf_auto_fit_range_and_step.is_selected();
        self.ltf_auto_fit_step.set_enabled(auto_fit_range_and_step);
        self.ltf_fake_sirt_iterations
            .set_enabled(self.cb_fake_sirt_iterations.is_selected());
        let use_sirt = self.cb_use_sirt.is_selected();
        self.ltf_fake_sirt_iterations.set_visible(!use_sirt);
        self.ltf_leave_iterations.set_visible(use_sirt);
        self.ltf_leave_iterations.set_enabled(use_sirt);
        self.cb_scale_to_integer.set_enabled(use_sirt);
        let fallback_and_extra_thickness = self.rb_fallback_and_extra_thickness.is_selected();
        self.ltf_extra_thickness
            .set_enabled(fallback_and_extra_thickness);
        self.ltf_fallback_thickness
            .set_enabled(fallback_and_extra_thickness);
        let do_trimvol = self.cb_do_trimvol.is_selected();
        self.ctf_find_sec_add_thickness.set_enabled(do_trimvol);
        self.ctf_scale_from_z.set_enabled(do_trimvol);
        let erase_gold = self.cb_erase_gold.is_selected();
        self.rb_erase_gold_fid
            .set_enabled(erase_gold && !fiducialless && !patch_tracking);
        self.rb_erase_gold_3d.set_enabled(erase_gold);
        if erase_gold && (fiducialless || patch_tracking) {
            self.rb_erase_gold_3d.set_selected_boolean(true);
        }
        self.ltf_gold_erasing_thickness
            .set_enabled(erase_gold && self.rb_erase_gold_3d.is_selected());
    }

    /// Java package-private `statusChanged(BatchRunTomoStatus)`.
    pub fn status_changed_status(&self, status: Option<StatusRef>) {
        let status = match status {
            Some(StatusRef::BatchRunTomoStatus(status)) => Some(status),
            _ => None,
        };
        self.status.set(status);
        // Don't open the dataset for STOPPED status. Nothing should be changed when
        // running from a stopping point.
        let open = status.is_none()
            || status == Some(BatchRunTomoStatus::Open)
            || status == Some(BatchRunTomoStatus::Done)
            || status == Some(BatchRunTomoStatus::Failed);
        self.cb_remove_xrays.set_editable(open);
        self.rb_tracking_method_seed.set_editable(open);
        self.rb_tracking_method_raptor.set_editable(open);
        self.rb_tracking_method_patch_tracking.set_editable(open);
        self.rb_fiducialless.set_editable(open);
        self.ls_bin_by_factor.set_editable(open);
        self.cb_correct_ctf.set_editable(open);
        self.cb_do_backproj_also.set_editable(open);
        self.cb_fake_sirt_iterations.set_editable(open);
        self.ltf_fake_sirt_iterations.set_editable(open);
        self.cb_use_sirt.set_editable(open);
        self.rtf_thickness.set_editable(open);
        self.rtf_binned_thickness.set_editable(open);
        self.rb_fallback_and_extra_thickness.set_editable(open);
        if let Some(btn_ok) = &self.btn_ok {
            btn_ok.set_editable(open);
        }
        if let Some(btn_revert_to_global) = &self.btn_revert_to_global {
            btn_revert_to_global.set_editable(open);
        }
        if let Some(ftf_distort) = &self.ftf_distort {
            ftf_distort.set_editable(open);
        }
        if let Some(ftf_gradient) = &self.ftf_gradient {
            ftf_gradient.set_editable(open);
        }
        self.ftf_model_file.set_editable(open);
        self.btn_model_file.set_editable(open);
        self.cb_enable_stretching.set_editable(open);
        self.cb_local_alignments.set_editable(open);
        self.ltf_gold.set_editable(open);
        self.ltf_target_number_of_beads.set_editable(open);
        self.ltf_number_of_markers.set_editable(open);
        self.ltf_size_of_patches_x_and_y.set_editable(open);
        self.cb_length_of_pieces.set_editable(open);
        self.ltf_scan_defocus_range.set_editable(open);
        self.ltf_defocus.set_editable(open);
        self.rtf_auto_fit_range_and_step.set_editable(open);
        self.rb_fit_every_image.set_editable(open);
        self.cb_sample_type.set_editable(open);
        self.ltf_auto_fit_step.set_editable(open);
        self.ltf_leave_iterations.set_editable(open);
        self.cb_scale_to_integer.set_editable(open);
        self.ltf_extra_thickness.set_editable(open);
        self.ltf_fallback_thickness.set_editable(open);
        self.ls_prenewst_bin_by_factor.set_editable(open);
        self.ls_preblend_bin_by_factor.set_editable(open);
        self.cb_do_trimvol.set_editable(open);
        self.ctf_find_sec_add_thickness.set_editable(open);
        self.ctf_scale_from_z.set_editable(open);
        self.cb_erase_gold.set_editable(open);
        self.rb_erase_gold_fid.set_editable(open);
        self.rb_erase_gold_3d.set_editable(open);
        self.ltf_gold_erasing_thickness.set_editable(open);
        self.rb_sample_type_plastic_section.set_editable(open);
        self.rb_sample_type_cryo.set_editable(open);
        self.ltf_positioning_thickness.set_editable(open);
        self.cb_has_gold_beads.set_editable(open);
        self.ltf_positioning_gold.set_editable(open);
        self.cb_tune_fitting_and_sampling.set_editable(open);
        let directives_dialog = self.directives_dialog.borrow().clone();
        if let Some(directives_dialog) = directives_dialog {
            directives_dialog.status_changed(self.status.get());
        }
    }

    /// Java package-private `statusChanged(StatusChangeEvent)`.
    pub fn status_changed_event(&self, status_change_event: Option<&dyn StatusChangeEvent>) {
        let Some(status_change_event) = status_change_event else {
            return;
        };
        let Some(event) = status_change_event
            .as_any()
            .downcast_ref::<StatusChangeBooleanEvent>()
        else {
            return;
        };
        let status = event.get_status();
        let Some(StatusRef::FrameStatus(frame_status)) = status else {
            return;
        };
        if frame_status == FrameStatus::Single {
            self.ls_prenewst_bin_by_factor.set_enabled(event.is());
        } else if frame_status == FrameStatus::Montage {
            self.ls_preblend_bin_by_factor.set_enabled(event.is());
        }
    }

    /// Java package-private `backupIfChanged(boolean)`.  Check isDifferentFromCheckpoint
    /// on all data entry fields; returns true if any field's isDifferentFromCheckpoint
    /// function returned true.
    pub fn backup_if_changed(&self, only_advanced_dataset_dialog: bool) -> bool {
        let mut changed = false;
        if !only_advanced_dataset_dialog {
            let field_list = self.field_list.borrow().clone();
            for field in &field_list {
                if field.is_different_from_checkpoint(true) {
                    field.backup();
                    changed = true;
                }
            }
        }
        if only_advanced_dataset_dialog
            && let Some(directives_dialog) = self.directives_dialog.borrow().clone()
        {
            directives_dialog.backup_if_changed();
        }
        changed
    }

    /// Java package-private `applyValues(boolean, boolean, DirectiveFileCollection,
    /// boolean)`.  Called when the template has changed.
    pub fn apply_values(
        &self,
        init: bool,
        retain_user_values: bool,
        directive_file_collection: &DirectiveFileCollection,
        only_advanced_dataset_dialog: bool,
    ) {
        let field_list = self.field_list.borrow().clone();
        // to apply values and highlights, start with a clean slate
        if !only_advanced_dataset_dialog && !init {
            for field in &field_list {
                field.clear_field_highlight();
                field.clear();
            }
        }
        let directives_dialog = self.directives_dialog.borrow().clone();
        if let Some(directives_dialog) = &directives_dialog {
            directives_dialog.clear_template_values();
            directives_dialog.clear();
        }
        // Apply default values
        if !only_advanced_dataset_dialog {
            self.set_individual_defaults();
        }
        if !only_advanced_dataset_dialog {
            for field in &field_list {
                field.use_default_value();
            }
        }
        // No settings values to apply
        // Apply the directive collection values
        self.set_values(
            directive_file_collection,
            false,
            only_advanced_dataset_dialog,
            false,
        );
        // Checkpoint and restore from backup
        if !only_advanced_dataset_dialog {
            for field in &field_list {
                // checkpoint
                field.checkpoint();
                // If the user wants to retain their values, apply backed up values and
                // then delete them.
                if retain_user_values {
                    field.restore_from_backup();
                }
            }
        }
        if let Some(directives_dialog) = &directives_dialog {
            directives_dialog.checkpoint_and_restore_from_backup(retain_user_values);
        }
        // Set new highlight values - batch directive file must be ignored
        self.set_values(
            directive_file_collection,
            true,
            only_advanced_dataset_dialog,
            false,
        );
    }

    /// Java package-private `setMontage(boolean)`.  Sets coarse alignment binning based
    /// on a single row.
    pub fn set_montage(&self, montage: bool) {
        self.ls_prenewst_bin_by_factor.set_enabled(!montage);
        self.ls_preblend_bin_by_factor.set_enabled(montage);
    }

    /// Java private `setIndividualDefaults()`.
    fn set_individual_defaults(&self) {
        self.rtf_auto_fit_range_and_step.set_selected_boolean(true);
    }

    /// Java package-private `setParameters(BatchRunTomoDatasetMetaData)`.
    pub fn set_parameters_dataset_meta_data(&self, meta_data: &BatchRunTomoDatasetMetaData) {
        let header = meta_data.get_header();
        self.ph_root.set(
            header
                .as_ref()
                .map(|header| header as &dyn crate::imod::etomo::r#type::const_panel_header_settings::ConstPanelHeaderSettings),
        );
        let postprocessing_header = meta_data.get_postprocessing_header();
        self.ph_postprocessing.set(Some(&postprocessing_header));
        self.ftf_model_file
            .set_text_string_boolean(Some(&meta_data.get_model_file()), false);
        self.cb_enable_stretching
            .set_selected_const_etomo_number_boolean(Some(&meta_data.get_enable_stretching()), false);
        self.cb_local_alignments
            .set_selected_const_etomo_number_boolean(Some(&meta_data.get_local_alignments()), false);
        self.ltf_gold
            .set_text_string_boolean(Some(&meta_data.get_gold()), false);
        self.ltf_target_number_of_beads
            .set_text_string_boolean(Some(&meta_data.get_target_number_of_beads()), false);
        self.ltf_number_of_markers
            .set_text_string_boolean(Some(&meta_data.get_number_of_markers()), false);
        self.ltf_size_of_patches_x_and_y
            .set_text_string_boolean(Some(&meta_data.get_size_of_patches_x_and_y()), false);
        self.cb_length_of_pieces
            .set_selected_const_etomo_number_boolean(Some(&meta_data.get_length_of_pieces()), false);
        self.ltf_scan_defocus_range
            .set_text_string_boolean(Some(&meta_data.get_scan_defocus_range()), false);
        self.ltf_defocus
            .set_text_string_boolean(Some(&meta_data.get_defocus()), false);
        self.rtf_auto_fit_range_and_step.set_selected_const_etomo_number_boolean(
            Some(&meta_data.get_auto_fit_range_and_step()),
            false,
        );
        self.rtf_auto_fit_range_and_step
            .set_text_string_boolean(Some(&meta_data.get_auto_fit_range()), false);
        self.rb_fit_every_image
            .set_selected_const_etomo_number_boolean(Some(&meta_data.get_fit_every_image()), false);
        self.ltf_auto_fit_step
            .set_text_string_boolean(Some(&meta_data.get_auto_fit_step()), false);
        self.cb_fake_sirt_iterations
            .set_selected_boolean(meta_data.is_use_fake_sirt_iterations());
        self.ltf_fake_sirt_iterations
            .set_text_string_boolean(Some(&meta_data.get_fake_sirt_iterations()), false);
        self.ltf_leave_iterations
            .set_text_string_boolean(Some(&meta_data.get_leave_iterations()), false);
        self.cb_scale_to_integer
            .set_selected_const_etomo_number_boolean(Some(&meta_data.get_scale_to_integer()), false);
        self.rtf_thickness
            .set_text_string_boolean(Some(&meta_data.get_thickness()), false);
        self.rtf_binned_thickness
            .set_text_string_boolean(Some(&meta_data.get_binned_thickness()), false);
        self.ltf_extra_thickness
            .set_text_string_boolean(Some(&meta_data.get_extra_thickness()), false);
        self.ltf_fallback_thickness
            .set_text_string_boolean(Some(&meta_data.get_fallback_thickness()), false);
        self.ls_prenewst_bin_by_factor
            .set_value_string_boolean(Some(&meta_data.get_prenewst_bin_by_factor()), false);
        self.ls_preblend_bin_by_factor
            .set_value_string_boolean(Some(&meta_data.get_preblend_bin_by_factor()), false);
        self.ctf_find_sec_add_thickness.set_selected_const_etomo_number_boolean(
            Some(&meta_data.get_use_find_sec_add_thickness()),
            false,
        );
        self.ctf_find_sec_add_thickness
            .set_text_string_boolean(Some(&meta_data.get_find_sec_add_thickness()), true);
        self.ctf_scale_from_z
            .set_selected_const_etomo_number_boolean(Some(&meta_data.get_use_scale_from_z()), false);
        self.ctf_scale_from_z
            .set_text_string_boolean(Some(&meta_data.get_scale_from_z()), false);
        self.rb_erase_gold_fid
            .set_selected_const_etomo_number_boolean(Some(&meta_data.get_erase_gold_fid()), false);
        self.rb_erase_gold_3d
            .set_selected_const_etomo_number_boolean(Some(&meta_data.get_erase_gold_3d()), false);
        self.ltf_gold_erasing_thickness
            .set_text_string_boolean(Some(&meta_data.get_gold_erasing_thickness()), false);
        self.rb_sample_type_plastic_section.set_selected_const_etomo_number_boolean(
            Some(&meta_data.get_sample_type_plastic_section()),
            false,
        );
        self.rb_sample_type_cryo
            .set_selected_const_etomo_number_boolean(Some(&meta_data.get_sample_type_cryo()), false);
        self.ltf_positioning_thickness
            .set_text_string_boolean(Some(&meta_data.get_positioning_thickness()), false);
        self.cb_has_gold_beads
            .set_selected_const_etomo_number_boolean(Some(&meta_data.get_has_gold_beads()), false);
        self.ltf_positioning_gold
            .set_text_string_boolean(Some(&meta_data.get_positioning_gold()), false);
        self.cb_tune_fitting_and_sampling.set_selected_const_etomo_number_boolean(
            Some(&meta_data.get_tune_fitting_and_sampling()),
            false,
        );
        self.update_display();
        self.status_changed_status(self.status.get().map(StatusRef::BatchRunTomoStatus));
    }

    /// Java package-private `getParameters(BatchRunTomoDatasetMetaData)`.
    pub fn get_parameters_dataset_meta_data(&self, meta_data: &BatchRunTomoDatasetMetaData) {
        meta_data.set_header(Some(&*self.ph_root));
        meta_data.set_postprocessing_header(&*self.ph_postprocessing);
        meta_data.set_model_file(self.ftf_model_file.get_file().as_deref());
        meta_data.set_enable_stretching(self.cb_enable_stretching.is_selected());
        meta_data.set_local_alignments(self.cb_local_alignments.is_selected());
        meta_data.set_gold(Field::get_text_void(&*self.ltf_gold).as_deref());
        meta_data.set_target_number_of_beads(
            Field::get_text_void(&*self.ltf_target_number_of_beads).as_deref(),
        );
        meta_data.set_number_of_markers(Field::get_text_void(&*self.ltf_number_of_markers).as_deref());
        meta_data.set_size_of_patches_x_and_y(
            Field::get_text_void(&*self.ltf_size_of_patches_x_and_y).as_deref(),
        );
        meta_data.set_length_of_pieces(self.cb_length_of_pieces.is_selected());
        meta_data
            .set_scan_defocus_range(Field::get_text_void(&*self.ltf_scan_defocus_range).as_deref());
        meta_data.set_defocus(Field::get_text_void(&*self.ltf_defocus).as_deref());
        meta_data.set_auto_fit_range_and_step(self.rtf_auto_fit_range_and_step.is_selected());
        meta_data.set_auto_fit_range(Field::get_text_void(&*self.rtf_auto_fit_range_and_step).as_deref());
        meta_data.set_fit_every_image(self.rb_fit_every_image.is_selected());
        meta_data.set_auto_fit_step(Field::get_text_void(&*self.ltf_auto_fit_step).as_deref());
        meta_data.set_use_fake_sirt_iterations(self.cb_fake_sirt_iterations.is_selected());
        meta_data.set_fake_sirt_iterations(
            Field::get_text_void(&*self.ltf_fake_sirt_iterations).as_deref(),
        );
        meta_data.set_leave_iterations(Field::get_text_void(&*self.ltf_leave_iterations).as_deref());
        meta_data.set_scale_to_integer(self.cb_scale_to_integer.is_selected());
        meta_data.set_thickness(Field::get_text_void(&*self.rtf_thickness).as_deref());
        meta_data.set_binned_thickness(Field::get_text_void(&*self.rtf_binned_thickness).as_deref());
        meta_data.set_extra_thickness(Field::get_text_void(&*self.ltf_extra_thickness).as_deref());
        meta_data
            .set_fallback_thickness(Field::get_text_void(&*self.ltf_fallback_thickness).as_deref());
        meta_data.set_prenewst_bin_by_factor(Some(self.ls_prenewst_bin_by_factor.get_value()));
        meta_data.set_preblend_bin_by_factor(Some(self.ls_preblend_bin_by_factor.get_value()));
        meta_data.set_use_find_sec_add_thickness(self.ctf_find_sec_add_thickness.is_selected());
        meta_data.set_find_sec_add_thickness(
            Field::get_text_void(&*self.ctf_find_sec_add_thickness).as_deref(),
        );
        meta_data.set_use_scale_from_z(self.ctf_scale_from_z.is_selected());
        meta_data.set_scale_from_z(Field::get_text_void(&*self.ctf_scale_from_z).as_deref());
        meta_data.set_erase_gold_fid(self.rb_erase_gold_fid.is_selected());
        meta_data.set_erase_gold_3d(self.rb_erase_gold_3d.is_selected());
        meta_data.set_gold_erasing_thickness(
            Field::get_text_void(&*self.ltf_gold_erasing_thickness).as_deref(),
        );
        meta_data.set_sample_type_plastic_section(self.rb_sample_type_plastic_section.is_selected());
        meta_data.set_sample_type_cryo(self.rb_sample_type_cryo.is_selected());
        meta_data.set_positioning_thickness(
            Field::get_text_void(&*self.ltf_positioning_thickness).as_deref(),
        );
        meta_data.set_has_gold_beads(self.cb_has_gold_beads.is_selected());
        meta_data.set_positioning_gold(Field::get_text_void(&*self.ltf_positioning_gold).as_deref());
        meta_data.set_tune_fitting_and_sampling(self.cb_tune_fitting_and_sampling.is_selected());
    }

    /// Java `display()`.
    pub fn display_void(&self) {
        self.display_advanced(false);
    }

    /// Java private `display(boolean)`.
    fn display_advanced(&self, advanced: bool) {
        if let Some(frame) = &self.frame {
            if !frame.is_visible() {
                frame.set_visible(true);
            }
            // `frame.setState(Frame.NORMAL)`; `frame.toFront()`.
        } else if let Some(parent) = self.parent() {
            parent.display_tab(Some(BatchRunTomoTab::Dataset));
        }
        self.set_advanced(advanced);
    }

    /// Java private `setAdvanced(boolean)`.
    fn set_advanced(&self, advanced: bool) {
        if self.advanced.get() == advanced {
            return;
        }
        // There are two buttons for advanced/basic rather then a toggle button.
        if !self.advanced.get() {
            self.btn_advanced.do_click();
        } else {
            self.btn_basic.do_click();
        }
    }

    /// The selected button's enumerated type, Java
    /// `((RadioButton.RadioButtonModel) group.getSelection()).getEnumeratedType()`.
    fn selection_enumerated_type(group: &ButtonGroup) -> Option<EnumeratedTypeRef> {
        let selection = group.get_selection()?;
        let model = selection.get_model()?;
        let model = model.as_any().downcast_ref::<RadioButtonModel>()?;
        AbstractRadioButtonModel::get_enumerated_type(model)
    }

    /// The selected button's model, Java
    /// `(RadioButton.RadioButtonModel) group.getSelection()`.
    fn selection_model(group: &ButtonGroup) -> Option<Rc<dyn crate::imod::etomo::jdk::ButtonModel>> {
        group.get_selection()?.get_model()
    }

    /// Java package-private `saveAutodoc(WritableAutodoc, boolean, boolean)`.
    pub fn save_autodoc(&self, autodoc: *mut Autodoc, do_validation: bool, validate_only: bool) -> bool {
        if (validate_only && !do_validation) || autodoc.is_null() {
            return true;
        }
        let template_values = self
            .template_values
            .as_ref()
            .map(|template_values| template_values.borrow().clone());
        let template_values = template_values.as_ref();
        let displayer = self.field_displayer();
        // try
        let result: Result<bool, FieldValidationFailedException> = (|| {
            // Phasing out BatchTool
            batch_tool::save_text_field_to_autodoc(
                self.ftf_distort.as_deref().map(|f| f as &dyn crate::imod::etomo::ui::text_field_interface::TextFieldInterface),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                self.ftf_gradient.as_deref().map(|f| f as &dyn crate::imod::etomo::ui::text_field_interface::TextFieldInterface),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cb_remove_xrays),
                autodoc,
                template_values,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ftf_model_file),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ls_prenewst_bin_by_factor),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ls_preblend_bin_by_factor),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            // Tracking method and fiducialess
            // Get the radio button model and save the value of the enumerated type to
            // the autodoc.
            let tracking_model = Self::selection_model(&self.bg_tracking_method);
            let tracking_model = tracking_model
                .as_ref()
                .and_then(|model| model.as_any().downcast_ref::<RadioButtonModel>());
            if !batch_tool::save_radio_button_model_to_autodoc(
                tracking_model,
                autodoc,
                template_values,
                validate_only,
            )? {
                // No enumerated type was found, so fidless was selected
                batch_tool::save_boolean_to_autodoc(
                    Some(&*self.rb_fiducialless),
                    autodoc,
                    template_values,
                    validate_only,
                )?;
                // Also save an empty tracking method
                batch_tool::override_in_autodoc(
                    DirectiveDef::TRACKING_METHOD,
                    autodoc,
                    template_values,
                    validate_only,
                )?;
                batch_tool::override_in_autodoc(
                    DirectiveDef::SEEDING_METHOD,
                    autodoc,
                    template_values,
                    validate_only,
                )?;
            } else if self.rb_tracking_method_seed.is_selected() {
                // autodoc and track also sets the Both seeding method:
                // `saveTextToAutodoc(boolean, DirectiveDef, EnumeratedType, ...)` with
                // `SeedingMethod.BOTH`, whose text is `getValue().toString()`.
                let both: SeedingMethod = seeding_method::BOTH;
                let value = both.get_value();
                let text = if value.is_null() {
                    None
                } else {
                    Some(value.to_string())
                };
                batch_tool::save_text_to_autodoc(
                    self.rb_tracking_method_seed.is_enabled(),
                    Some(DirectiveDef::SEEDING_METHOD),
                    text.as_deref(),
                    None,
                    autodoc,
                    template_values,
                    validate_only,
                )?;
            }
            // Gold
            // Both gold fields use the same directive. Use positioning gold if it is
            // enabled. To avoid the value in these fields being overridden with the
            // batchDefaults value, save the value even if it is disabled.
            let gold = if self.fiducial_model_mode.get() {
                Field::get_text_boolean_field_displayer(&*self.ltf_gold, do_validation, displayer.clone())?
            } else if self.ltf_positioning_gold.is_enabled() {
                Field::get_text_boolean_field_displayer(
                    &*self.ltf_positioning_gold,
                    do_validation,
                    displayer.clone(),
                )?
            } else {
                Some("0".to_owned())
            };
            batch_tool::save_text_to_autodoc(
                true,
                Some(DirectiveDef::GOLD),
                gold.as_deref(),
                None,
                autodoc,
                template_values,
                validate_only,
            )?;
            //
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ltf_target_number_of_beads),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ltf_number_of_markers),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ltf_size_of_patches_x_and_y),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            // lengthOfPieces or default length of pieces should be saved when the
            // checkbox is true.
            batch_tool::save_boolean_text_to_autodoc_default(
                Some(&*self.cb_length_of_pieces),
                self.length_of_pieces.borrow().as_deref(),
                Some(LENGTH_OF_PIECES_DEFAULT),
                autodoc,
                template_values,
                validate_only,
            )?;
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cb_enable_stretching),
                autodoc,
                template_values,
                validate_only,
            )?;
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cb_local_alignments),
                autodoc,
                template_values,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ls_bin_by_factor),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cb_correct_ctf),
                autodoc,
                template_values,
                validate_only,
            )?;
            //
            // Defocus
            // At least one defocus field must be filled in if CTF correction is selected.
            if do_validation
                && self.cb_correct_ctf.is_selected()
                && self.ltf_scan_defocus_range.is_empty()
                && self.ltf_defocus.is_empty()
            {
                Popup::get_error_instance(
                    Some(self as &dyn UIComponent),
                    Some("Missing Defocus Value"),
                    Some(&format!(
                        "Fill in either {} or {}.",
                        utilities::quote_label(Some(&self.ltf_scan_defocus_range.get_label()))
                            .unwrap_or_else(|| "null".to_owned()),
                        utilities::quote_label(Some(&self.ltf_defocus.get_label()))
                            .unwrap_or_else(|| "null".to_owned())
                    )),
                    displayer.clone(),
                    None,
                )
                .open();
                return Ok(false);
            }
            // scanDefocusRange
            let double_array = converter::to_double_array(
                Field::get_text_boolean_field_displayer(
                    &*self.ltf_scan_defocus_range,
                    do_validation,
                    displayer.clone(),
                )?
                .as_deref(),
                Some(self.ltf_scan_defocus_range.get_field_type()),
            );
            let mut string_builder: Option<String> = None;
            if let Some(double_array) = &double_array {
                let mut builder = String::new();
                for (i, value) in double_array.iter().enumerate() {
                    if i > 0 {
                        builder.push(',');
                    }
                    if let Some(value) = value {
                        builder.push_str(&java_lang_double_to_string(value * 1000.0));
                    }
                }
                string_builder = Some(builder);
            }
            batch_tool::save_text_to_autodoc_field(
                Some(&*self.ltf_scan_defocus_range),
                string_builder.as_deref(),
                autodoc,
                template_values,
                validate_only,
            )?;
            // defocus
            let mut defocus = EtomoNumber::new_with_type(Some(EtomoNumberType::Double));
            let mut directive_value =
                Field::get_text_boolean_field_displayer(&*self.ltf_defocus, do_validation, displayer.clone())?;
            defocus.set_string(directive_value.as_deref());
            if !defocus.is_null() {
                directive_value = Some(java_lang_double_to_string(defocus.get_double() * 1000.0));
            }
            batch_tool::save_text_to_autodoc_field(
                Some(&*self.ltf_defocus),
                directive_value.as_deref(),
                autodoc,
                template_values,
                validate_only,
            )?;
            //
            // AUTO_FIT_RANGE_AND_STEP
            let mut range = Some(CTF_RANGE_DEFAULT.to_owned());
            let mut separator = ",";
            let mut step = Some(CTF_STEP_DEFAULT.to_owned());
            if self.rb_fit_every_image.is_selected() {
                range = Some(ctf_range_fit_every_image().to_string());
                step = Some(ctf_step_fit_every_image().to_string());
            } else {
                range = Field::get_text_boolean_field_displayer(
                    &*self.rtf_auto_fit_range_and_step,
                    do_validation,
                    displayer.clone(),
                )?;
                step = Field::get_text_boolean_field_displayer(
                    &*self.ltf_auto_fit_step,
                    do_validation,
                    displayer.clone(),
                )?;
                if step.as_deref().is_none_or(|step| {
                    step.chars()
                        .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
                }) {
                    step = Some(String::new());
                    separator = "";
                }
            }
            batch_tool::save_text_to_autodoc(
                self.cb_correct_ctf.is_selected(),
                Field::get_directive_def(&*self.rtf_auto_fit_range_and_step),
                Some(&format!(
                    "{}{}{}",
                    range.as_deref().unwrap_or("null"),
                    separator,
                    step.as_deref().unwrap_or("null")
                )),
                None,
                autodoc,
                template_values,
                validate_only,
            )?;
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cb_tune_fitting_and_sampling),
                autodoc,
                template_values,
                validate_only,
            )?;
            // back projection, sirt-like, and sirt
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cb_do_backproj_also),
                autodoc,
                template_values,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ltf_fake_sirt_iterations),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cb_use_sirt),
                autodoc,
                template_values,
                validate_only,
            )?;
            //
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ltf_leave_iterations),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            // Replacing user scale to integer value with recommended value.
            batch_tool::save_boolean_text_to_autodoc_default(
                Some(&*self.cb_scale_to_integer),
                None,
                Some(SCALE_TO_INTEGER_VALUE),
                autodoc,
                template_values,
                validate_only,
            )?;
            //
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ltf_fallback_thickness),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ltf_extra_thickness),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_boolean_text_field_to_autodoc(
                Some(&*self.rtf_binned_thickness),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_boolean_text_field_to_autodoc(
                Some(&*self.rtf_thickness),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            // If scaleFromZ is enabled & unchecked, override the scaleFromZ directive.
            if self.ctf_scale_from_z.is_enabled() && !self.ctf_scale_from_z.is_selected() {
                batch_tool::override_in_autodoc(
                    DirectiveDef::SCALE_FROM_Z,
                    autodoc,
                    template_values,
                    validate_only,
                )?;
            } else {
                batch_tool::save_boolean_text_field_to_autodoc(
                    Some(&*self.ctf_scale_from_z),
                    autodoc,
                    do_validation,
                    displayer.clone(),
                    template_values,
                    validate_only,
                )?;
            }
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cb_do_trimvol),
                autodoc,
                template_values,
                validate_only,
            )?;
            batch_tool::save_boolean_text_field_to_autodoc(
                Some(&*self.ctf_find_sec_add_thickness),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            // erase gold
            let mut enumerated_type = None;
            if self.cb_erase_gold.is_selected() {
                enumerated_type = Self::selection_enumerated_type(&self.bg_erase_gold);
            }
            batch_tool::save_enumerated_type_to_autodoc(
                self.cb_erase_gold.is_enabled(),
                Field::get_directive_def(&*self.cb_erase_gold),
                enumerated_type.as_ref(),
                autodoc,
                template_values,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ltf_gold_erasing_thickness),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            // SampleType
            if self.cb_sample_type.is_selected() {
                batch_tool::save_enumerated_type_to_autodoc(
                    self.cb_sample_type.is_enabled(),
                    Field::get_directive_def(&*self.cb_sample_type),
                    Self::selection_enumerated_type(&self.bg_sample_type).as_ref(),
                    autodoc,
                    template_values,
                    validate_only,
                )?;
            } else {
                // Save sampleType = 0, if necessary
                batch_tool::save_not_text_to_autodoc(
                    DirectiveDef::SAMPLE_TYPE,
                    Some(&EnumeratedTypeRef::new(SampleType::None)),
                    autodoc,
                    template_values,
                    validate_only,
                )?;
            }
            //
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.ltf_positioning_thickness),
                autodoc,
                do_validation,
                displayer.clone(),
                template_values,
                validate_only,
            )?;
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cb_has_gold_beads),
                autodoc,
                template_values,
                validate_only,
            )?;
            let directives_dialog = self.directives_dialog.borrow().clone();
            if let Some(directives_dialog) = directives_dialog {
                return Ok(directives_dialog.save_autodoc(
                    autodoc,
                    do_validation,
                    Some(&*self.advanced_field_displayer as &dyn FieldDisplayer),
                    validate_only,
                ));
            }
            Ok(true)
        })();
        // catch (FieldValidationFailedException e)
        result.unwrap_or(false)
    }

    /// Java package-private `setValues(DirectiveFileInterface, boolean, boolean,
    /// boolean)`.  Set values from the directive file collection.  Only change fields
    /// that exist in directive file collection.  Make sure the template values are set
    /// for each field when setFieldHighlightValue is on.
    pub fn set_values(
        &self,
        directive_files: &dyn DirectiveFileInterface,
        set_field_highlight_value: bool,
        only_advanced_dataset_dialog: bool,
        loading_directive_file: bool,
    ) {
        let mut template_values_guard = self
            .template_values
            .as_ref()
            .map(|template_values| template_values.borrow_mut());
        if !only_advanced_dataset_dialog {
            // `templateValues` for each BatchTool call.
            macro_rules! tv {
                () => {
                    template_values_guard.as_deref_mut()
                };
            }
            // Phasing out BatchTool
            batch_tool::set_text_value(
                self.ftf_distort.as_deref().map(|f| f as &dyn crate::imod::etomo::ui::text_field_interface::TextFieldInterface),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_text_value(
                self.ftf_gradient.as_deref().map(|f| f as &dyn crate::imod::etomo::ui::text_field_interface::TextFieldInterface),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_boolean_value(
                Some(&*self.cb_remove_xrays),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_text_value(
                Some(&*self.ftf_model_file),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_text_value(
                Some(&*self.ls_prenewst_bin_by_factor),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_text_value(
                Some(&*self.ls_preblend_bin_by_factor),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_boolean_value(
                Some(&*self.rb_fiducialless),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            // Tracking, seeding radio buttons
            // Tracking method
            let mut directive_def = DirectiveDef::TRACKING_METHOD;
            let mut contains_tracking_method = false;
            let mut tracking_method: Option<TrackingMethod> = None;
            if directive_files.contains_template(Some(directive_def), set_field_highlight_value) {
                contains_tracking_method = true;
                let directive_value = directive_files.get_value(Some(directive_def));
                if set_field_highlight_value && let Some(template_values) = tv!() {
                    template_values.insert(directive_def, directive_value.clone());
                }
                tracking_method = TrackingMethod::get_instance(directive_value.as_deref());
            }
            // Seeding method
            directive_def = DirectiveDef::SEEDING_METHOD;
            let mut contains_seeding_method = false;
            let mut seeding_method_value: Option<SeedingMethod> = None;
            if directive_files.contains_template(Some(directive_def), set_field_highlight_value) {
                contains_seeding_method = true;
                let directive_value = directive_files.get_value(Some(directive_def));
                if set_field_highlight_value && let Some(template_values) = tv!() {
                    template_values.insert(directive_def, directive_value.clone());
                }
                seeding_method_value = SeedingMethod::get_instance(directive_value.as_deref());
            }
            if let Some(tracking_method) = tracking_method {
                // Don't highlight auto seed unless both seeding method and tracking method
                // are present.
                if tracking_method == tracking_method::SEED
                    && (seeding_method_value == Some(seeding_method::AUTO_FID_SEED)
                        || seeding_method_value == Some(seeding_method::BOTH))
                {
                    batch_tool::set_boolean_value_selected_highlight(
                        Some(&*self.rb_tracking_method_seed),
                        true,
                        contains_tracking_method && contains_seeding_method,
                        set_field_highlight_value,
                    );
                } else if tracking_method == tracking_method::RAPTOR {
                    batch_tool::set_boolean_value_selected(
                        Some(&*self.rb_tracking_method_raptor),
                        true,
                        set_field_highlight_value,
                    );
                } else if tracking_method == tracking_method::PATCH_TRACKING {
                    batch_tool::set_boolean_value_selected(
                        Some(&*self.rb_tracking_method_patch_tracking),
                        true,
                        set_field_highlight_value,
                    );
                }
                self.update_gold_panel();
            }
            // Gold
            let directive_value = directive_files
                .get_value_template(Some(DirectiveDef::GOLD), set_field_highlight_value);
            batch_tool::set_text_value_string(
                Some(&*self.ltf_gold),
                directive_value.as_deref(),
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_text_value_string(
                Some(&*self.ltf_positioning_gold),
                directive_value.as_deref(),
                set_field_highlight_value,
                tv!(),
            );
            //
            batch_tool::set_text_value(
                Some(&*self.ltf_target_number_of_beads),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_text_value(
                Some(&*self.ltf_number_of_markers),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_text_value(
                Some(&*self.ltf_size_of_patches_x_and_y),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            // length of pieces
            let value = batch_tool::set_boolean_value_from_text(
                Some(&*self.cb_length_of_pieces),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            // Save length of pieces
            if !set_field_highlight_value {
                *self.length_of_pieces.borrow_mut() = value;
            }
            //
            batch_tool::set_boolean_value(
                Some(&*self.cb_enable_stretching),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_boolean_value(
                Some(&*self.cb_local_alignments),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_text_value(
                Some(&*self.ls_bin_by_factor),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_boolean_value(
                Some(&*self.cb_correct_ctf),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_boolean_value(
                Some(&*self.cb_tune_fitting_and_sampling),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            // ScanDefocusRange
            let directive_def = Field::get_directive_def(&*self.ltf_scan_defocus_range);
            let mut directive_value = None;
            let mut calculated_value = None;
            if directive_files.contains_template(directive_def, set_field_highlight_value) {
                directive_value =
                    directive_files.get_value_template(directive_def, set_field_highlight_value);
                let double_array = converter::to_double_array(
                    directive_value.as_deref(),
                    Some(self.ltf_scan_defocus_range.get_field_type()),
                );
                if let Some(double_array) = &double_array {
                    let mut builder = String::new();
                    for (i, value) in double_array.iter().enumerate() {
                        if i > 0 {
                            builder.push(',');
                        }
                        if let Some(value) = value {
                            builder.push_str(&java_lang_double_to_string(value / 1000.0));
                        }
                    }
                    calculated_value = Some(builder);
                }
            }
            batch_tool::set_text_value_derived(
                Some(&*self.ltf_scan_defocus_range),
                directive_value.as_deref(),
                calculated_value.as_deref(),
                set_field_highlight_value,
                tv!(),
            );
            // Defocus
            let directive_def = Field::get_directive_def(&*self.ltf_defocus);
            let mut directive_value = None;
            let mut calculated_value = None;
            if directive_files.contains_template(directive_def, set_field_highlight_value) {
                directive_value =
                    directive_files.get_value_template(directive_def, set_field_highlight_value);
                let mut defocus = EtomoNumber::new_with_type(Some(EtomoNumberType::Double));
                defocus.set_string(directive_value.as_deref());
                if !defocus.is_null() && defocus.is_valid() {
                    calculated_value =
                        Some(java_lang_double_to_string(defocus.get_double() / 1000.0));
                }
            }
            batch_tool::set_text_value_derived(
                Some(&*self.ltf_defocus),
                directive_value.as_deref(),
                calculated_value.as_deref(),
                set_field_highlight_value,
                tv!(),
            );
            // Three AUTO_FIT_RANGE_AND_STEP fields
            if batch_tool::set_text_values(
                Some(&*self.rtf_auto_fit_range_and_step),
                &*self.ltf_auto_fit_step,
                directive_files,
                set_field_highlight_value,
                tv!(),
            ) && !batch_tool::set_boolean_value_if_equals(
                Some(&*self.rb_fit_every_image),
                Field::get_text_void(&*self.ltf_auto_fit_step).as_deref(),
                0,
                set_field_highlight_value,
            ) && !set_field_highlight_value
            {
                batch_tool::set_boolean_value_selected(
                    Some(&*self.rtf_auto_fit_range_and_step),
                    true,
                    set_field_highlight_value,
                );
            }
            if self.cb_correct_ctf.is_selected() {
                if ctf_range_fit_every_image()
                    .equals_string(Field::get_text_void(&*self.rtf_auto_fit_range_and_step).as_deref())
                    && ctf_step_fit_every_image()
                        .equals_string(Field::get_text_void(&*self.ltf_auto_fit_step).as_deref())
                {
                    self.rb_fit_every_image.set_selected_boolean(true);
                } else {
                    self.rtf_auto_fit_range_and_step.set_selected_boolean(true);
                }
            }
            // back-projection, sirt-like, and sirt
            batch_tool::set_boolean_value_from_text(
                Some(&*self.cb_fake_sirt_iterations),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_text_value(
                Some(&*self.ltf_fake_sirt_iterations),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_boolean_value(
                Some(&*self.cb_use_sirt),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            // doBackprojAlso
            batch_tool::set_boolean_value_missing(
                Some(&*self.cb_do_backproj_also),
                directive_files,
                set_field_highlight_value,
                loading_directive_file,
                tv!(),
            );
            // DoBackprojAlso is also the default.
            if !self.cb_fake_sirt_iterations.is_selected() && !self.cb_use_sirt.is_selected() {
                self.cb_do_backproj_also.set_selected_boolean(true);
            }
            batch_tool::set_text_value(
                Some(&*self.ltf_leave_iterations),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            // For ScaleToInteger, only using the recommended value, so don't save the
            // directive value.
            batch_tool::set_boolean_value_from_text(
                Some(&*self.cb_scale_to_integer),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            // Priority of directives
            // 1. THICKNESS
            // 2. binnedThickness
            // 3. fallbackThickness (default)
            // Initialize them in reverse order of priority.
            let mut thickness_set = false;
            if batch_tool::set_boolean_and_text_value(
                Some(&*self.rb_fallback_and_extra_thickness),
                &*self.ltf_fallback_thickness,
                directive_files,
                set_field_highlight_value,
                tv!(),
            ) {
                thickness_set = true;
            }
            // Highlight radio button for Fallback and extra thickness when either fallback
            // or extra is set.
            if batch_tool::set_text_value(
                Some(&*self.ltf_extra_thickness),
                directive_files,
                set_field_highlight_value,
                tv!(),
            ) && set_field_highlight_value
            {
                self.rb_fallback_and_extra_thickness
                    .set_field_highlight_boolean(true);
            }
            if batch_tool::set_boolean_text_value(
                Some(&*self.rtf_binned_thickness),
                directive_files,
                set_field_highlight_value,
                tv!(),
            ) {
                thickness_set = true;
            }
            if batch_tool::set_boolean_text_value(
                Some(&*self.rtf_thickness),
                directive_files,
                set_field_highlight_value,
                tv!(),
            ) {
                thickness_set = true;
            }
            // set fallback setting
            if !set_field_highlight_value && !thickness_set {
                self.rb_fallback_and_extra_thickness
                    .set_selected_boolean(true);
            }
            // trimvol and scaling
            let contains_scale_from_x = directive_files
                .contains_template(Some(DirectiveDef::SCALE_FROM_X), set_field_highlight_value);
            let contains_scale_from_y = directive_files
                .contains_template(Some(DirectiveDef::SCALE_FROM_Y), set_field_highlight_value);
            let contains_scale_from_z = batch_tool::set_boolean_text_value(
                Some(&*self.ctf_scale_from_z),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            // Scale from Z is checked and set to a default when either scale from X or Y
            // is set.
            if !contains_scale_from_z && (contains_scale_from_x || contains_scale_from_y) {
                batch_tool::set_boolean_text_value_string(
                    Some(&*self.ctf_scale_from_z),
                    Some(batch_run_tomo_dataset_meta_data::SCALE_FROM_Z_DEFAULT),
                    set_field_highlight_value,
                );
            }
            // Turn on trimvol if any trimvol values are set.
            if !batch_tool::set_boolean_value(
                Some(&*self.cb_do_trimvol),
                directive_files,
                set_field_highlight_value,
                tv!(),
            ) && (contains_scale_from_x
                || contains_scale_from_y
                || contains_scale_from_z
                // Backwards compatibility - older trimvol values.
                || directive_files.contains_template(
                    Some(DirectiveDef::FIND_SEC_ADD_THICKNESS),
                    set_field_highlight_value,
                )
                || directive_files
                    .contains_template(Some(DirectiveDef::REORIENT), set_field_highlight_value)
                || directive_files.contains_template(
                    Some(DirectiveDef::THICKNESS_FOR_TRIMVOL),
                    set_field_highlight_value,
                )
                || directive_files
                    .contains_template(Some(DirectiveDef::SIZE_IN_X), set_field_highlight_value)
                || directive_files
                    .contains_template(Some(DirectiveDef::SIZE_IN_Y), set_field_highlight_value))
            {
                batch_tool::set_boolean_value_selected(
                    Some(&*self.cb_do_trimvol),
                    true,
                    set_field_highlight_value,
                );
            }
            //
            batch_tool::set_boolean_text_value(
                Some(&*self.ctf_find_sec_add_thickness),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            let erase_gold = EraseGold::get_instance(
                batch_tool::set_boolean_value_from_text(
                    Some(&*self.cb_erase_gold),
                    directive_files,
                    set_field_highlight_value,
                    tv!(),
                )
                .as_deref(),
            );
            if erase_gold == Some(EraseGold::Fid) {
                batch_tool::set_boolean_value_selected(
                    Some(&*self.rb_erase_gold_fid),
                    true,
                    set_field_highlight_value,
                );
            } else if erase_gold == Some(EraseGold::Find3d) {
                batch_tool::set_boolean_value_selected(
                    Some(&*self.rb_erase_gold_3d),
                    true,
                    set_field_highlight_value,
                );
            }
            batch_tool::set_text_value(
                Some(&*self.ltf_gold_erasing_thickness),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            let sample_type = SampleType::get_instance_from_string(
                batch_tool::set_boolean_value_from_unselected_text(
                    Some(&*self.cb_sample_type),
                    &SampleType::None.to_string(),
                    directive_files,
                    set_field_highlight_value,
                    tv!(),
                )
                .as_deref(),
            );
            if sample_type == Some(SampleType::PlasticSection) {
                batch_tool::set_boolean_value_selected(
                    Some(&*self.rb_sample_type_plastic_section),
                    true,
                    set_field_highlight_value,
                );
            } else if sample_type == Some(SampleType::Cryo) {
                batch_tool::set_boolean_value_selected(
                    Some(&*self.rb_sample_type_cryo),
                    true,
                    set_field_highlight_value,
                );
            }
            batch_tool::set_text_value(
                Some(&*self.ltf_positioning_thickness),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            batch_tool::set_boolean_value(
                Some(&*self.cb_has_gold_beads),
                directive_files,
                set_field_highlight_value,
                tv!(),
            );
            self.update_display();
        }
        drop(template_values_guard);
        let directives_dialog = self.directives_dialog.borrow().clone();
        if let Some(directives_dialog) = directives_dialog {
            directives_dialog.set_values(directive_files, set_field_highlight_value);
        }
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: Option<&ActionEvent>) {
        let Some(event) = event else {
            return;
        };
        let Some(action_command) = event.get_action_command() else {
            return;
        };
        let action_command = Some(action_command.to_owned());
        let row = self.row();
        if action_command == self.btn_model_file.get_action_command() && row.is_some() {
            let row = row.unwrap();
            // One 3dmod instance is created for each stack file, and the row controls
            // them.
            let model_file = self.ftf_model_file.get_file();
            if let Some(model_file) = &model_file {
                row.imod_stack_model_file(Some(model_file));
            } else {
                self.ftf_model_file.set_file(
                    row.imod_stack_file_type(Some(&file_type::CLASS.manual_replacement_model)),
                );
            }
        } else if self
            .btn_ok
            .as_ref()
            .is_some_and(|btn_ok| action_command == btn_ok.get_action_command())
        {
            self.set_visible(false);
        } else if self.btn_revert_to_global.as_ref().is_some_and(|btn_revert_to_global| {
            action_command == btn_revert_to_global.get_action_command()
        }) {
            if ui_harness::with(|harness| {
                harness.open_yes_no_dialog_base_manager_ui_component_string(
                    Some(self.base_manager()),
                    Some(self as &dyn UIComponent),
                    "Data in this window will be lost.  Revert to global dataset data for this stack?",
                )
            }) {
                self.set_visible(false);
                if let Some(row) = self.row() {
                    row.delete_dataset();
                }
            }
        } else if action_command == self.btn_advanced.get_action_command() {
            self.update_advanced(true, false);
        } else if action_command == self.btn_basic.get_action_command() {
            self.update_advanced(false, false);
        }
        if action_command == self.rb_tracking_method_seed.get_action_command() {
            self.update_gold_panel();
        }
        if action_command == self.rb_tracking_method_raptor.get_action_command() {
            self.update_gold_panel();
        } else if self.menu_fit_window.as_ref().is_some_and(|menu_fit_window| {
            action_command == menu_fit_window.get_action_command()
        }) {
            self.pack();
        } else if action_command == self.cb_do_backproj_also.get_action_command() {
            if !self.cb_fake_sirt_iterations.is_selected() && !self.cb_use_sirt.is_selected() {
                // If neither SIRT or SIRT-like is used, back project is done
                // automatically.
                self.cb_do_backproj_also.set_selected_boolean(true);
            }
            self.update_display();
        } else if action_command == self.cb_fake_sirt_iterations.get_action_command() {
            if self.cb_fake_sirt_iterations.is_selected() {
                self.cb_use_sirt.set_selected_boolean(false);
            } else if !self.cb_use_sirt.is_selected() {
                // If neither SIRT or SIRT-like is used, back project is done
                // automatically.
                self.cb_do_backproj_also.set_selected_boolean(true);
            }
            self.update_display();
        } else if action_command == self.cb_use_sirt.get_action_command() {
            if self.cb_use_sirt.is_selected() {
                self.cb_fake_sirt_iterations.set_selected_boolean(false);
            } else if !self.cb_fake_sirt_iterations.is_selected() {
                // If neither SIRT or SIRT-like is used, back project is done
                // automatically.
                self.cb_do_backproj_also.set_selected_boolean(true);
            }
            self.update_display();
        } else {
            self.update_display();
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        // override generated tooltips
        self.ls_prenewst_bin_by_factor.set_unformatted_tooltip(Some(
            "Image reduction used to make coarse aligned stack for single-frame datasets",
        ));
        self.ls_preblend_bin_by_factor.set_unformatted_tooltip(Some(
            "Image reduction used to make coarse aligned stack for montaged datasets",
        ));
        self.rb_tracking_method_seed.set_unformatted_tooltip(Some(
            "Make fiducial model of beads by finding seed points and tracking them with Beadtrack.",
        ));
        self.rb_tracking_method_raptor.set_unformatted_tooltip(Some(
            "Make fiducial model wih Raptor and run Beadtrack to complete model.",
        ));
        self.rb_tracking_method_patch_tracking.set_unformatted_tooltip(Some(
            "Align images without fiducials by tracking local patches from one view to the next.",
        ));
        self.rb_fiducialless
            .set_unformatted_tooltip(Some("Make tomogram from coarse alignment only."));
        self.cb_length_of_pieces.set_unformatted_tooltip(Some(
            "Divide contours from patch tracking into multiple overlapping pieces.",
        ));
        self.cb_local_alignments.set_unformatted_tooltip(Some(
            "Compute alignments in a local areas if there are enough beads.",
        ));
        self.cb_sample_type.set_unformatted_tooltip(Some(
            "Find specimen surfaces and do positioning for chosen sample type.",
        ));
        self.rb_sample_type_plastic_section.set_unformatted_tooltip(Some(
            "Find surfaces of stained, sectioned material and use for positioning.",
        ));
        self.rb_sample_type_cryo
            .set_unformatted_tooltip(Some("Find best orientation and Z shift of a cryospecimen."));
        Field::set_unformatted_tooltip(&*self.ltf_positioning_thickness, Some(
            "Unbinned thickness of tomogram used for positioning (required for cryo, optional for plastic)",
        ));
        self.cb_has_gold_beads.set_unformatted_tooltip(Some(
            "Indicate whether there are any gold beads; this information is needed for cryopositioning.",
        ));
        self.ls_bin_by_factor
            .set_unformatted_tooltip(Some("Image reduction when creating the aligned stack"));
        self.cb_correct_ctf.set_unformatted_tooltip(Some(
            "Find defocus by fitting to power spectrum of groups of views in Ctfplotter.",
        ));
        Field::set_unformatted_tooltip(
            &*self.rtf_auto_fit_range_and_step,
            Some("Tilt angle range for groups of views to fit"),
        );
        Field::set_unformatted_tooltip(
            &*self.ltf_auto_fit_step,
            Some("Tilt angle step between groups of views to fit"),
        );
        self.rb_fit_every_image.set_unformatted_tooltip(Some(
            "Find defocus by fitting to power spectrum of each image in Ctfplotter.",
        ));

        Field::set_unformatted_tooltip(
            &*self.ltf_defocus,
            Some("Nominal defocus in microns, underfocus positive."),
        );
        self.cb_erase_gold
            .set_unformatted_tooltip(Some("Erase gold by the selected method."));
        self.rb_erase_gold_fid.set_unformatted_tooltip(Some(
            "Erase gold selected as fiducial markers using a filled-in model.",
        ));
        self.rb_erase_gold_3d
            .set_unformatted_tooltip(Some("Find all gold in tomogram and erase from aligned stack."));
        Field::set_unformatted_tooltip(
            &*self.ltf_gold_erasing_thickness,
            Some("Unbinned thickness of tomogram used to find beads"),
        );

        self.cb_use_sirt
            .set_unformatted_tooltip(Some("Use the SIRT-like filter for reconstruction."));
        self.cb_use_sirt
            .set_unformatted_tooltip(Some("Use SIRT for reconstruction."));
        Field::set_unformatted_tooltip(
            &*self.rtf_thickness,
            Some("Use a fixed unbinned thickness for reconstruction."),
        );
        self.rtf_thickness
            .set_text_field_unformatted_tooltip(Some("Thickness of reconstruction in unbinned pixels"));
        Field::set_unformatted_tooltip(
            &*self.rtf_binned_thickness,
            Some("Use a fixed binned thickness for reconstruction."),
        );
        self.rtf_binned_thickness
            .set_text_field_unformatted_tooltip(Some("Thickness of reconstruction in binned pixels"));

        Field::set_unformatted_tooltip(&*self.ltf_extra_thickness, Some(
            "Unbinned thickness to add to a thickness computed from positioning or fiducial positions",
        ));
        self.ctf_find_sec_add_thickness.set_check_box_unformatted_tooltip(Some(
            "Use Findsection to find the limits in Z of the specimen; suitable for plastic sections only.",
        ));
        self.ctf_find_sec_add_thickness.set_field_unformatted_tooltip(Some(
            "Number of binned pixels to add to the extent found by Findsection; a fraction of the Z extent can also be entered",
        ));
        self.ctf_scale_from_z.set_check_box_unformatted_tooltip(Some(
            "Scale to bytes based on densities in the specified central fraction of the Z slices",
        ));
        self.ctf_scale_from_z.set_field_unformatted_tooltip(Some(
            "Fraction of Z slice to analyze to find scaling to bytes; the number of slices can also be entered",
        ));

        let field_list = self.field_list.borrow().clone();
        let parent = self.parent();
        for (i, field) in field_list.iter().enumerate() {
            if parent.is_none() || self.frame.is_none() || self.global {
                autodoc_attribute_retriever::INSTANCE.set_tool_tip_text(Some(&**field));
            } else if let Some(parent) = &parent {
                // Get tooltips from the global instance.
                let global = parent.get_dataset_dialog();
                let global_field = global.field_list.borrow().get(i).cloned();
                field.set_tooltip(global_field.as_deref());
            }
        }

        // constant tooltips
        self.btn_model_file.set_tool_tip_text(Some(
            "Open raw stack in 3dmod to draw points, lines, or contours to be replaced on all sections in objects 1, 2 or 3.",
        ));
        self.cb_do_backproj_also
            .set_tool_tip_text_string(Some("Use backprojection for reconstruction."));
        self.rb_fallback_and_extra_thickness.set_tool_tip_text_string(Some(
            "Use thickness computed from positioning, or from the distance between two layers of fiducials.",
        ));
        if let Some(btn_ok) = &self.btn_ok {
            btn_ok.set_tool_tip_text(Some("Exit dialog, keeping the dataset-specific values."));
        }
        if let Some(btn_revert_to_global) = &self.btn_revert_to_global {
            btn_revert_to_global.set_tool_tip_text(Some("Do not keep the dataset-specfic values."));
        }
    }
}

impl Expandable for BatchRunTomoDatasetDialog {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        let expanded = button.is_expanded();
        if self.ph_root.equals_open_close(button) {
            self.pnl_root_body.set_visible(expanded);
        } else if self.ph_postprocessing.equals_open_close(button) {
            self.pnl_postprocessing_body.set_visible(expanded);
        }
        self.pack();
    }

    /// Java `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}

impl TableListener for BatchRunTomoDatasetDialog {
    /// Java `lastRowDeleted(EventObject)`.  Only the global instance exists when there
    /// is an empty table.  So this function should only be called in the global
    /// instance.
    fn last_row_deleted(&self, _event: Option<&()>) {
        self.empty_table.set(true);
        *self.row.borrow_mut() = None;
        self.update_display();
    }

    /// Java `firstRowAdded(EventObject)`.  Only the global instance exists when there is
    /// an empty table.  So this function should only be called in the global instance.
    fn first_row_added(&self, _event: Option<&()>) {
        self.empty_table.set(false);
        if let Some(parent) = self.parent() {
            // The global instance is attached to the first row for the purpose of
            // opening 3dmod.
            *self.row.borrow_mut() = parent.get_first_row().map(|row| Rc::downgrade(&row));
        }
        self.update_display();
    }
}

impl SwingComponent for BatchRunTomoDatasetDialog {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }
}

impl UIComponent for BatchRunTomoDatasetDialog {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }
}

impl FieldDisplayer for BatchRunTomoDatasetDialog {
    /// Java `display()`.
    fn display_void(&self) {
        BatchRunTomoDatasetDialog::display_void(self);
    }

    /// Java `display(UIComponent)`.
    fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {
        BatchRunTomoDatasetDialog::display_void(self);
    }
}

/// `Path` is used by the window title code.
#[allow(dead_code)]
fn _path(_path: &Path) {}
