//! `IMOD/Etomo/src/etomo/ui/swing/FinalAlignedStackDialog.java`.
//!
//! Java `public final class FinalAlignedStackDialog extends ProcessDialog
//! implements Expandable, Run3dmodButtonContainer, ContextMenu,
//! ProcessInterface` (the Final Aligned Stack dialog: Create / Correct CTF /
//! Erase Gold / 2D Filter tabs).
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by
//! [`FinalAlignedStackDialog::get_instance`]; every method takes `&self`; the
//! `ProcessDialog` superclass is the embedded `base` (reached through `Deref`)
//! and the overridden `done()` is `ProcessDialogVirtual::done`.  The listener
//! inner classes (`ButtonListener`, `MtfFileActionListener`,
//! `StartingAndEndingZKeyListener`, `TabChangeListener`) are closures holding a
//! weak reference to the dialog; the static inner classes `Tab` and
//! `TypeOfDoseFile` are the enums [`Tab`] and [`TypeOfDoseFile`].
//!
//! The expert is held weakly: the expert owns the dialog (Java field
//! `FinalAlignedStackExpert.dialog`), and the Java `expert` field is only used
//! to call back into it.
//!
//! Java field `newstackOrBlendmontPanel` has the abstract type
//! `NewstackOrBlendmontPanel`; the constructor stores a `BlendmontPanel` or a
//! `NewstackPanel`.  It is [`NewstackOrBlendmontPanelRef`] here: the concrete
//! object, dereferencing to the shared superclass part.

use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use std::cell::Cell;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::blendmont_param::BlendmontParam;
use crate::imod::etomo::comscript::const_ccd_eraser_param::ConstCCDEraserParam;
use crate::imod::etomo::comscript::const_ctf_phase_flip_param::ConstCtfPhaseFlipParam;
use crate::imod::etomo::comscript::const_find_beads3d_param::ConstFindBeads3dParam;
use crate::imod::etomo::comscript::const_mtf_filter_param::ConstMTFFilterParam;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::const_tiltalign_param::ConstTiltalignParam;
use crate::imod::etomo::comscript::ctf_phase_flip_param::{self, CtfPhaseFlipParam};
use crate::imod::etomo::comscript::ctf_plotter_param;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::mtf_filter_param::{self, MTFFilterParam};
use crate::imod::etomo::comscript::newst_param::NewstParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent, MouseEvent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::autodoc::section::Section;
use crate::imod::etomo::storage::dose_weighting_file_filter::DoseWeightingFileFilter;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::mtf_file_filter::MtfFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_header_state::PanelHeaderState;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::beveled_border::BeveledBorder;
use super::blendmont_display::BlendmontDisplay;
use super::blendmont_panel::BlendmontPanel;
use super::button_component::ButtonComponent;
use super::button_control_text_efield::ButtonControlTextEfield;
use super::ccd_eraser_beads_panel;
use super::ccd_eraser_display::CcdEraserDisplay;
use super::check_box::CheckBox;
use super::check_box_efield::CheckBoxEfield;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::control_target::ControlTarget;
use super::cpu_gpu_panel::CpuGpuPanel;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::erase_gold_panel::{self, EraseGoldPanel};
use super::etched_border::EtchedBorder;
use super::etomo_button_group::EtomoButtonGroup;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::fiducialess_params::FiducialessParams;
use super::file_chooser::{self, FileChooser};
use super::file_text_field2::FileTextField2;
use super::final_aligned_stack_expert::FinalAlignedStackExpert;
use super::find_beads3d_display::FindBeads3dDisplay;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::global_expand_button::GlobalExpandButton;
use super::label::Label;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::newstack_display::NewstackDisplay;
use super::newstack_or_blendmont_panel::NewstackOrBlendmontPanel;
use super::newstack_panel::NewstackPanel;
use super::panel_header::PanelHeader;
use super::process_control_panel;
use super::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use super::process_interface::ProcessInterface;
use super::radio_button_interface::EnumeratedTypeRef;
use super::radio_ebutton::{RadioEButtonModel, RadioEbutton};
use super::recon_ui_expert::ReconUIExpertVirtual;
use super::reproject_model_panel;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::scaled_image::{self, ScaledImage};
use super::simple_button::SimpleButton;
use super::spaced_panel::{self, SpacedPanel};
use super::spaced_text_field::SpacedTextField;
use super::tabbed_pane::TabbedPane;
use super::text_efield::TextEfield;
use super::tilt_display::TiltDisplay;
use super::tilt3d_find_panel;
use super::tooltip_formatter;
use super::ui_harness;
use super::ui_utilities;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java private static final `MTF_FILE_LABEL`.
const MTF_FILE_LABEL: &str = "MTF file: ";
/// Java public static final `USE_CTF_CORRECTION_LABEL`.
pub const USE_CTF_CORRECTION_LABEL: &str = "Use CTF Correction";
/// Java public static final `CTF_TAB_LABEL`.
pub const CTF_TAB_LABEL: &str = "Correct CTF";
/// Java public static final `USE_FILTERED_STACK_LABEL`.
pub const USE_FILTERED_STACK_LABEL: &str = "Use Filtered Stack";
/// Java public static final `MTF_FILTER_TAB_LABEL = SharedStrings._2D_FILTER_LABEL`.
pub const MTF_FILTER_TAB_LABEL: &str = shared_strings::_2D_FILTER_LABEL;
/// Java public static final `FINAL_ALIGNED_STACK_TAB_LABEL`.
pub const FINAL_ALIGNED_STACK_TAB_LABEL: &str = "Create";

/// Java private static final `DIALOG_TYPE`.
const DIALOG_TYPE: DialogType = DialogType::FinalAlignedStack;
/// Java public static final `CTF_CORRECTION_LABEL`.
pub const CTF_CORRECTION_LABEL: &str = "Correct CTF";
/// Java private static final `FIXED_IMAGE_DOSE_LABEL`.
const FIXED_IMAGE_DOSE_LABEL: &str = "No dose file; use fixed dose per image: ";

/// The Java field `newstackOrBlendmontPanel` (declared type: the abstract
/// class `NewstackOrBlendmontPanel`): whichever subclass the constructor
/// built.  Dereferences to the superclass part, as a Java call through the
/// abstract type would resolve.
pub enum NewstackOrBlendmontPanelRef {
    /// `BlendmontPanel.getInstance(...)` (montage datasets).
    Blendmont(Rc<BlendmontPanel>),
    /// `NewstackPanel.getInstance(...)` (single-frame datasets).
    Newstack(Rc<NewstackPanel>),
}

impl Deref for NewstackOrBlendmontPanelRef {
    type Target = NewstackOrBlendmontPanel;
    fn deref(&self) -> &NewstackOrBlendmontPanel {
        match self {
            NewstackOrBlendmontPanelRef::Blendmont(panel) => panel,
            NewstackOrBlendmontPanelRef::Newstack(panel) => panel,
        }
    }
}

impl NewstackOrBlendmontPanelRef {
    /// Java `(BlendmontDisplay) newstackOrBlendmontPanel`: the superclass
    /// implements `BlendmontDisplay`, so the cast succeeds for either subclass.
    fn as_blendmont_display(&self) -> Rc<dyn BlendmontDisplay> {
        match self {
            NewstackOrBlendmontPanelRef::Blendmont(panel) => panel.clone(),
            NewstackOrBlendmontPanelRef::Newstack(panel) => panel.clone(),
        }
    }

    /// Java `(NewstackDisplay) newstackOrBlendmontPanel`: the superclass
    /// implements `NewstackDisplay`, so the cast succeeds for either subclass.
    fn as_newstack_display(&self) -> Rc<dyn NewstackDisplay> {
        match self {
            NewstackOrBlendmontPanelRef::Blendmont(panel) => panel.clone(),
            NewstackOrBlendmontPanelRef::Newstack(panel) => panel.clone(),
        }
    }
}

/// Java `public final class FinalAlignedStackDialog extends ProcessDialog
/// implements Expandable, Run3dmodButtonContainer, ContextMenu,
/// ProcessInterface`.
pub struct FinalAlignedStackDialog {
    /// The `ProcessDialog` superclass.
    base: Rc<ProcessDialog>,
    /// Java `this` (handed to the mediator and the listeners).
    this: Weak<FinalAlignedStackDialog>,

    /// Java private final `newstackOrBlendmontPanel`.
    newstack_or_blendmont_panel: NewstackOrBlendmontPanelRef,

    /// Java private final `pnlFinalAlignedStack` (never laid out).
    pnl_final_aligned_stack: Rc<EtomoPanel>,

    // MTF Filter objects
    /// Java private final `ltfLowPassRadiusSigma`.
    ltf_low_pass_radius_sigma: Rc<LabeledTextField>,
    /// Java private final `scaledImage`.
    scaled_image: &'static ScaledImage,
    /// Java private final `ltfMtfFile`.
    ltf_mtf_file: Rc<LabeledTextField>,
    /// Java private final `btnMtfFile`.
    btn_mtf_file: Rc<SimpleButton>,
    /// Java private final `ltfMaximumInverse`.
    ltf_maximum_inverse: Rc<LabeledTextField>,
    /// Java private final `ltfInverseRolloffRadiusSigma`.
    ltf_inverse_rolloff_radius_sigma: Rc<LabeledTextField>,
    /// Java private final `btnFilter`.
    btn_filter: Rc<Run3dmodButton>,
    /// Java private final `btnViewFilter`.
    btn_view_filter: Rc<Run3dmodButton>,
    /// Java private final `btnUseFilter`.
    btn_use_filter: Rc<MultiLineButton>,
    /// Java private final `ltfStartingAndEndingZ`.
    ltf_starting_and_ending_z: Rc<SpacedTextField>,
    /// Java private final `filterHeader`.
    filter_header: Rc<PanelHeader>,

    // panels that are changed in setAdvanced()
    /// Java private final `inverseParamsPanel`.
    inverse_params_panel: Rc<SpacedPanel>,
    /// Java private final `filterBodyPanel`.
    filter_body_panel: Rc<JComponent>,

    // backward compatibility functionality - if the metadata binning is missing
    // get binning from newst
    /// Java private final `screenState`.
    screen_state: &'static ReconScreenState,

    /// Java private final `finalAlignedStackListener` (`ButtonListener`).
    final_aligned_stack_listener: ActionListener,

    /// Java private final `expert` (held weakly; see the module docs).
    expert: Weak<FinalAlignedStackExpert>,
    /// Java private final `mediator`.
    mediator: Option<Rc<ProcessingMethodMediator>>,

    // ctf correction
    /// Java private final `ctfCorrectionHeader`.
    ctf_correction_header: Rc<PanelHeader>,
    /// Java private final `ctfCorrectionBodyPanel`.
    ctf_correction_body_panel: Rc<SpacedPanel>,
    /// Java private final `ltfVoltage`.
    ltf_voltage: Rc<LabeledTextField>,
    /// Java private final `ltfSphericalAberration`.
    ltf_spherical_aberration: Rc<LabeledTextField>,
    /// Java private final `cbInvertTiltAngles`.
    cb_invert_tilt_angles: Rc<CheckBox>,
    /// Java private final `ltfAmplitudeContrast`.
    ltf_amplitude_contrast: Rc<LabeledTextField>,
    /// Java private final `ltfScanDefocusRange`.
    ltf_scan_defocus_range: Rc<LabeledTextField>,
    /// Java private final `ltfExpectedDefocus`.
    ltf_expected_defocus: Rc<LabeledTextField>,
    /// Java private final `tfPhaseShiftInDegrees`.
    tf_phase_shift_in_degrees: Rc<TextEfield>,
    /// Java private final `ltfOffsetToAdd`.
    ltf_offset_to_add: Rc<LabeledTextField>,
    /// Java private final `ltfInterpolationWidth`.
    ltf_interpolation_width: Rc<LabeledTextField>,
    /// Java private final `ltfDefocusTol`.
    ltf_defocus_tol: Rc<LabeledTextField>,
    /// Java private final `btnCtfPlotter`.
    btn_ctf_plotter: Rc<MultiLineButton>,
    /// Java private final `btnCtfCorrection`.
    btn_ctf_correction: Rc<Run3dmodButton>,
    /// Java private final `btnImodCtfCorrection`.
    btn_imod_ctf_correction: Rc<Run3dmodButton>,
    /// Java private final `btnUseCtfCorrection`.
    btn_use_ctf_correction: Rc<MultiLineButton>,
    /// Java private final `cbUseExpectedDefocus`.
    cb_use_expected_defocus: Rc<CheckBox>,

    /// Java private final `tabbedPane`.
    tabbed_pane: Rc<TabbedPane>,
    /// Java private final `ctfCorrectionMainPanel`.
    ctf_correction_main_panel: Rc<EtomoPanel>,
    /// Java private final `filterPanel`.
    filter_panel: Rc<JComponent>,

    /// Java private final `ftfConfigFile`.
    ftf_config_file: Rc<FileTextField2>,

    // Dose Weighting
    /// Java private final `cbDoseWeightFiltering`.
    cb_dose_weight_filtering: Rc<CheckBoxEfield>,
    /// Java private final `pnlUniformFiltering`.
    pnl_uniform_filtering: Rc<JComponent>,
    /// Java private final `cbFixedImageDose`.
    cb_fixed_image_dose: Rc<CheckBoxEfield>,
    /// Java private final `tfFixedImageDose`.
    tf_fixed_image_dose: Rc<TextEfield>,
    /// Java private final `lFixedImageDose` (a `JLabel`).
    l_fixed_image_dose: Rc<JComponent>,
    /// Java private final `bctfDoseWeightingFile`.
    bctf_dose_weighting_file: Rc<ButtonControlTextEfield>,
    /// Java private final `pnlTypeOfDoseFile`.
    pnl_type_of_dose_file: Rc<JComponent>,
    /// Java private final `bgTypeOfDoseFile`.
    bg_type_of_dose_file: Rc<EtomoButtonGroup>,
    /// Java private final `rbTypeOfDoseFileMdocFile`.
    rb_type_of_dose_file_mdoc_file: Rc<RadioEbutton>,
    /// Java private final `rbTypeOfDoseFileImageDose`.
    rb_type_of_dose_file_image_dose: Rc<RadioEbutton>,
    /// Java private final `rbTypeOfDoseFileAccumulatedAndImageDose`.
    rb_type_of_dose_file_accumulated_and_image_dose: Rc<RadioEbutton>,
    /// Java private final `rbTypeOfDoseFilePriorAndCumulativeDose`.
    rb_type_of_dose_file_prior_and_cumulative_dose: Rc<RadioEbutton>,
    /// Java private final `cbVoltage200`.
    cb_voltage200: Rc<CheckBoxEfield>,
    /// Java private final `tfOptimalDoseScaling`.
    tf_optimal_dose_scaling: Rc<TextEfield>,
    /// Java private final `tfBidirectionalNumViews`.
    tf_bidirectional_num_views: Rc<TextEfield>,
    /// Java private final `tfCtfPhaseFlipXAxisTilt`.
    tf_ctf_phase_flip_x_axis_tilt: Rc<TextEfield>,
    /// Java private final `cbCtfPhaseFlipXAxisTilt`.
    cb_ctf_phase_flip_x_axis_tilt: Rc<CheckBoxEfield>,
    /// Java private final `tfScaleByCtfPower`.
    tf_scale_by_ctf_power: Rc<TextEfield>,
    /// Java private final `cbScaleByCtfPower`.
    cb_scale_by_ctf_power: Rc<CheckBoxEfield>,
    /// Java private final `tfMinimumZeroSpacing`.
    tf_minimum_zero_spacing: Rc<TextEfield>,
    /// Java private final `lCtf3dCorrectCtf` (`new Label(...)`).
    l_ctf3d_correct_ctf: Rc<Label>,
    /// Java private final `lCtf3d2dFilter1`.
    l_ctf3d_2d_filter1: Rc<Label>,
    /// Java private final `lCtf3d2dFilter2`.
    l_ctf3d_2d_filter2: Rc<Label>,

    /// Java private final `eraseGoldPanel`.
    erase_gold_panel: Rc<EraseGoldPanel>,
    /// Java private final `cpuGpuPanel`.
    cpu_gpu_panel: Rc<CpuGpuPanel>,

    /// Java private `trialTilt` (never read).
    trial_tilt: Cell<bool>,
    /// Java private `curTab`.
    cur_tab: Cell<Tab>,
    /// Java private `processingMethodLocked`.
    processing_method_locked: Cell<bool>,
    /// Java private `validAutodoc` (never read).
    valid_autodoc: Cell<bool>,
    /// Java private `eraseBeadsInitialized`.
    erase_beads_initialized: Cell<bool>,
    // DialogSaved has no functionality right now. Tilt3dfind has been created when the
    // dialog is first opened, so it's fine for figuring out whether initialize should be
    // on.aswhhas better. But it is moved to dataset creation, there's no other way to tell
    // if this dialog has been opened before. This metadata parameter will provide backwards
    // compatibility and functionality going forwards.
    /// Java private `dialogSaved`.
    dialog_saved: Cell<bool>,
}

impl Deref for FinalAlignedStackDialog {
    type Target = ProcessDialog;
    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

/// Error of Java `getParameters(MTFFilterParam, boolean) throws
/// FortranInputSyntaxException`: the declared exception, plus the unchecked
/// `NumberFormatException` that `MTFFilterParam.setMaximumInverse` throws and
/// the expert catches (`FinalAlignedStackExpert.updateMTFFilterCom`).
#[derive(Debug)]
pub enum MTFFilterParametersException {
    /// Java `FortranInputSyntaxException`.
    FortranInputSyntax(FortranInputSyntaxException),
    /// Java `NumberFormatException`, with its message.
    NumberFormat(String),
}

impl std::fmt::Display for MTFFilterParametersException {
    /// Java `getMessage()` of the exception.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MTFFilterParametersException::FortranInputSyntax(e) => write!(f, "{e}"),
            MTFFilterParametersException::NumberFormat(message) => f.write_str(message),
        }
    }
}

impl FinalAlignedStackDialog {
    /// Java private constructor `FinalAlignedStackDialog(ApplicationManager,
    /// FinalAlignedStackExpert, AxisID)` (FinalAlignedStackDialog.java:229).
    fn new(
        app_mgr: &'static ApplicationManager,
        expert: Weak<FinalAlignedStackExpert>,
        axis_id: AxisID,
    ) -> Rc<FinalAlignedStackDialog> {
        let instance = Rc::new_cyclic(|this: &Weak<FinalAlignedStackDialog>| {
            // super(appMgr, axisID, DIALOG_TYPE)
            let base = ProcessDialog::new_application_manager_axis_id_dialog_type(
                app_mgr,
                axis_id,
                DIALOG_TYPE,
            );
            let expandable: Weak<dyn Expandable> = this.clone();
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            // Field initializers, in declaration order.
            let pnl_final_aligned_stack = EtomoPanel::new();
            // MTF Filter objects
            let ltf_low_pass_radius_sigma = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointPair,
                Some("Low pass (cutoff,sigma): "),
            );
            let scaled_image: &'static ScaledImage = if !*utilities::APRIL_FOOLS {
                &scaled_image::OPEN_FILE
            } else {
                &scaled_image::OPEN_FILE_FOOL
            };
            let ltf_mtf_file =
                LabeledTextField::new_field_type_string(FieldType::String, Some(MTF_FILE_LABEL));
            let btn_mtf_file = SimpleButton::new_scaled_image(Some(scaled_image));
            let ltf_maximum_inverse = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Maximum Inverse: "),
            );
            let ltf_inverse_rolloff_radius_sigma = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointPair,
                Some("Rolloff (radius,sigma): "),
            );
            let btn_view_filter =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Filtered Stack"),
                    Some(container.clone()),
                );
            let ltf_starting_and_ending_z =
                SpacedTextField::new(FieldType::IntegerPair, "Starting and ending views: ");
            let filter_header = PanelHeader::get_advanced_basic_only_instance(
                Some("2D Filtering (optional)"),
                Some(expandable.clone()),
                Some(DIALOG_TYPE),
                Some(base.btn_advanced.clone()),
                false,
            );
            // ButtonListener
            let final_aligned_stack_listener: ActionListener = {
                let adaptee = this.clone();
                Rc::new(move |event: &ActionEvent| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                    }
                })
            };
            // ctf correction
            let ctf_correction_header = PanelHeader::get_advanced_basic_only_instance(
                Some("CTF Correction"),
                Some(expandable.clone()),
                Some(DIALOG_TYPE),
                Some(base.btn_advanced.clone()),
                false,
            );
            let ctf_correction_body_panel = SpacedPanel::get_instance_boolean(true);
            let ltf_voltage =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Voltage (KV): "));
            let ltf_spherical_aberration = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Spherical Aberration (mm): "),
            );
            let cb_invert_tilt_angles = CheckBox::new_string(Some("Invert sign of tilt angles"));
            let ltf_amplitude_contrast = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Amplitude contrast: "),
            );
            let ltf_scan_defocus_range = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointPair,
                Some("Defocus range to scan (low, high in microns): "),
            );
            let ltf_expected_defocus = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Expected defocus (microns): "),
            );
            let tf_phase_shift_in_degrees = TextEfield::get_labeled_instance(
                Some("Expected phase shift (degrees): "),
                Some(FieldType::FloatingPoint),
            );
            let ltf_offset_to_add = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Offset to add to image values: "),
            );
            let ltf_interpolation_width = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Interpolation width (pixels): "),
            );
            let ltf_defocus_tol = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Defocus tolerance (nm): "),
            );
            let btn_ctf_plotter = MultiLineButton::new_string(Some(&format!(
                "Run {}",
                shared_strings::CTF_PLOTTER_LABEL
            )));
            let btn_imod_ctf_correction =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View CTF Correction"),
                    Some(container.clone()),
                );
            let cb_use_expected_defocus = CheckBox::new_string(Some(&format!(
                "Use {} instead of ctfplotter output",
                shared_strings::EXPECTED_DEFOCUS_LABEL
            )));
            let tabbed_pane = TabbedPane::new();
            let ctf_correction_main_panel = EtomoPanel::new();
            let filter_panel = JComponent::new_panel();
            // Dose Weighting
            let cb_dose_weight_filtering =
                CheckBoxEfield::get_instance(Some("Apply dose weighting"));
            let pnl_uniform_filtering = JComponent::new_panel();
            let cb_fixed_image_dose = CheckBoxEfield::get_instance(Some(FIXED_IMAGE_DOSE_LABEL));
            let tf_fixed_image_dose = TextEfield::get_instance(
                Some(FIXED_IMAGE_DOSE_LABEL),
                Some(FieldType::FloatingPoint),
            );
            let l_fixed_image_dose = JComponent::new_label(" e/sq A");
            let bctf_dose_weighting_file =
                ButtonControlTextEfield::get_labeled_file_instance_string(Some(
                    "Dose information file: ",
                ));
            let pnl_type_of_dose_file = JComponent::new_panel();
            let bg_type_of_dose_file = EtomoButtonGroup::new();
            let rb_type_of_dose_file_mdoc_file = RadioEbutton::get_enum_instance(
                Some(EnumeratedTypeRef::new(TypeOfDoseFile::MdocFile)),
                Some(&*bg_type_of_dose_file),
            );
            let rb_type_of_dose_file_image_dose = RadioEbutton::get_enum_instance(
                Some(EnumeratedTypeRef::new(TypeOfDoseFile::ImageDose)),
                Some(&*bg_type_of_dose_file),
            );
            let rb_type_of_dose_file_accumulated_and_image_dose = RadioEbutton::get_enum_instance(
                Some(EnumeratedTypeRef::new(
                    TypeOfDoseFile::AccumulatedAndImageDose,
                )),
                Some(&*bg_type_of_dose_file),
            );
            let rb_type_of_dose_file_prior_and_cumulative_dose = RadioEbutton::get_enum_instance(
                Some(EnumeratedTypeRef::new(
                    TypeOfDoseFile::PriorAndCumulativeDose,
                )),
                Some(&*bg_type_of_dose_file),
            );
            let cb_voltage200 = CheckBoxEfield::get_instance(Some("Microscope voltage is 200 KV"));
            let tf_optimal_dose_scaling = TextEfield::get_labeled_instance(
                Some("Optimal dose scaling"),
                Some(FieldType::FloatingPoint),
            );
            let tf_bidirectional_num_views = TextEfield::get_labeled_instance(
                Some(&format!(
                    "{}{}",
                    "\n", "# of views in first half of bidirectional series"
                )),
                Some(FieldType::Integer),
            );
            let tf_ctf_phase_flip_x_axis_tilt = TextEfield::get_disabled_instance(
                Some("Correct for X axis tilt of"),
                Some(FieldType::FloatingPoint),
            );
            let cb_ctf_phase_flip_x_axis_tilt = CheckBoxEfield::get_enable_control_instance(Some(
                Rc::downgrade(&tf_ctf_phase_flip_x_axis_tilt) as Weak<dyn ControlTarget>,
            ));
            let tf_scale_by_ctf_power = TextEfield::get_disabled_instance(
                Some("Scale by CTF to the power:"),
                Some(FieldType::FloatingPoint),
            );
            let cb_scale_by_ctf_power = CheckBoxEfield::get_enable_control_instance(Some(
                Rc::downgrade(&tf_scale_by_ctf_power) as Weak<dyn ControlTarget>,
            ));
            let tf_minimum_zero_spacing = TextEfield::get_labeled_instance(
                Some("Minimum zero spacing in strip FFTs:"),
                Some(FieldType::FloatingPoint),
            );
            let l_ctf3d_correct_ctf = Label::new_string(Some(
                "If doing 3D CTF, set correction parameters here; no need to run \"Correct CTF\".",
            ));
            let l_ctf3d_2d_filter1 = Label::new_string(Some(
                "If doing 3D CTF, set filtering parameters here; no need to",
            ));
            let l_ctf3d_2d_filter2 = Label::new_string(Some(
                "run \"Filter\" if \"Apply 2D filter\" option set in 3D CTF.",
            ));

            // Constructor body.
            // this.expert = expert;
            let mediator = app_mgr.get_processing_method_mediator(Some(axis_id));
            let display_factory = app_mgr.get_process_result_display_factory(axis_id);
            let erase_gold_panel = EraseGoldPanel::get_instance(
                app_mgr,
                this.clone(),
                axis_id,
                base.dialog_type,
                &base.btn_advanced,
            );
            let cpu_gpu_panel = CpuGpuPanel::get_instance(
                app_mgr,
                axis_id,
                PanelId::CtfPhaseFlip,
                -1,
                false,
                spaced_panel::X_AXIS,
            );
            let newstack_or_blendmont_panel =
                if app_mgr.get_meta_data().get_view_type() == ViewType::Montage {
                    NewstackOrBlendmontPanelRef::Blendmont(BlendmontPanel::get_instance(
                        app_mgr,
                        axis_id,
                        DIALOG_TYPE,
                        &base.btn_advanced,
                    ))
                } else {
                    NewstackOrBlendmontPanelRef::Newstack(NewstackPanel::get_instance(
                        app_mgr,
                        axis_id,
                        DIALOG_TYPE,
                        &base.btn_advanced,
                    ))
                };
            let screen_state = app_mgr.get_screen_state(axis_id);
            // Java casts `(Run3dmodButton) displayFactory.getFilter()`; the
            // factory returns the concrete button.
            let btn_filter = display_factory.get_filter();
            // init
            bctf_dose_weighting_file.set_limit_displayed_file_path(50);
            if ConstMetaData::get_image_filename_style(app_mgr.get_meta_data())
                != ImageFilenameStyle::Hdf
            {
                bctf_dose_weighting_file.set_required(true);
                let manager: &'static dyn BaseManager = app_mgr;
                bctf_dose_weighting_file.set_text_file(
                    file_type::CLASS
                        .mtf_filter_mdoc
                        .get_file(Some(manager), Some(axis_id))
                        .as_deref(),
                );
            }
            btn_filter.set_container(Some(container.clone()));
            let deferred: Rc<dyn Deferred3dmodButton> = btn_view_filter.clone();
            btn_filter.set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
            let btn_use_filter = display_factory.get_use_filtered_stack();
            // Swing layout: rootPanel.setLayout(new BoxLayout(rootPanel, BoxLayout.Y_AXIS)).
            base.btn_execute.set_text(Some("Done"));
            btn_mtf_file.set_name(Some(MTF_FILE_LABEL));
            // Layout the main panel (and sub panels) and add it to the root panel
            base.root_panel
                .set_border(&BeveledBorder::new(Some("Final Aligned Stack")).get_border());
            let filter_body_panel = JComponent::new_panel();
            let inverse_params_panel = SpacedPanel::get_instance_boolean(true);

            let btn_ctf_correction = display_factory.get_ctf_correction();
            btn_ctf_correction.set_container(Some(container.clone()));
            let deferred: Rc<dyn Deferred3dmodButton> = btn_imod_ctf_correction.clone();
            btn_ctf_correction.set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
            let btn_use_ctf_correction = display_factory.get_use_ctf_correction();
            let ftf_config_file =
                FileTextField2::get_instance(Some(app_mgr), Some("Config file: "));
            // `expert.getConfigDir()`: the expert creates this dialog, so it is
            // alive here.
            let config_dir = expert.upgrade().map(|expert| expert.get_config_dir());
            ftf_config_file.set_origin_file(config_dir.as_deref());
            ftf_config_file.set_absolute_path(true);

            FinalAlignedStackDialog {
                base,
                this: this.clone(),
                newstack_or_blendmont_panel,
                pnl_final_aligned_stack,
                ltf_low_pass_radius_sigma,
                scaled_image,
                ltf_mtf_file,
                btn_mtf_file,
                ltf_maximum_inverse,
                ltf_inverse_rolloff_radius_sigma,
                btn_filter,
                btn_view_filter,
                btn_use_filter,
                ltf_starting_and_ending_z,
                filter_header,
                inverse_params_panel,
                filter_body_panel,
                screen_state,
                final_aligned_stack_listener,
                expert,
                mediator,
                ctf_correction_header,
                ctf_correction_body_panel,
                ltf_voltage,
                ltf_spherical_aberration,
                cb_invert_tilt_angles,
                ltf_amplitude_contrast,
                ltf_scan_defocus_range,
                ltf_expected_defocus,
                tf_phase_shift_in_degrees,
                ltf_offset_to_add,
                ltf_interpolation_width,
                ltf_defocus_tol,
                btn_ctf_plotter,
                btn_ctf_correction,
                btn_imod_ctf_correction,
                btn_use_ctf_correction,
                cb_use_expected_defocus,
                tabbed_pane,
                ctf_correction_main_panel,
                filter_panel,
                ftf_config_file,
                cb_dose_weight_filtering,
                pnl_uniform_filtering,
                cb_fixed_image_dose,
                tf_fixed_image_dose,
                l_fixed_image_dose,
                bctf_dose_weighting_file,
                pnl_type_of_dose_file,
                bg_type_of_dose_file,
                rb_type_of_dose_file_mdoc_file,
                rb_type_of_dose_file_image_dose,
                rb_type_of_dose_file_accumulated_and_image_dose,
                rb_type_of_dose_file_prior_and_cumulative_dose,
                cb_voltage200,
                tf_optimal_dose_scaling,
                tf_bidirectional_num_views,
                tf_ctf_phase_flip_x_axis_tilt,
                cb_ctf_phase_flip_x_axis_tilt,
                tf_scale_by_ctf_power,
                cb_scale_by_ctf_power,
                tf_minimum_zero_spacing,
                l_ctf3d_correct_ctf,
                l_ctf3d_2d_filter1,
                l_ctf3d_2d_filter2,
                erase_gold_panel,
                cpu_gpu_panel,
                trial_tilt: Cell::new(false),
                cur_tab: Cell::new(Tab::DEFAULT),
                processing_method_locked: Cell::new(false),
                valid_autodoc: Cell::new(false),
                erase_beads_initialized: Cell::new(false),
                dialog_saved: Cell::new(false),
            }
        });
        // Java `this` as the ProcessDialog subclass (for the virtual `done()`).
        let this: Weak<dyn ProcessDialogVirtual> =
            Rc::downgrade(&instance) as Weak<dyn ProcessDialogVirtual>;
        instance.base.set_this(this);
        // The rest of the Java constructor needs the constructed fields.
        // field instantiation
        instance.layout_newst_panel();
        instance.layout_ctf_correction_panel();
        instance.layout_ccd_eraser();
        instance.layout_filter_panel();
        instance
            .base
            .root_panel
            .get_component()
            .add(&instance.tabbed_pane.get_component());
        instance.base.add_exit_buttons();
        // Set the default advanced dialog state
        instance.update_advanced();
        instance.set_tool_tip_text();
        instance.reregister_processing_method_mediator();
        instance
    }

    /// Java static package-private `getInstance(ApplicationManager,
    /// FinalAlignedStackExpert, AxisID, Tab)` (FinalAlignedStackDialog.java:289).
    pub fn get_instance(
        app_mgr: &'static ApplicationManager,
        expert: Weak<FinalAlignedStackExpert>,
        axis_id: AxisID,
        cur_tab: Tab,
    ) -> Rc<FinalAlignedStackDialog> {
        let instance = FinalAlignedStackDialog::new(app_mgr, expert, axis_id);
        instance.add_listeners();
        instance
            .tabbed_pane
            .get_component()
            .set_selected_tab(cur_tab.to_int());
        instance
    }

    /// Java private `addListeners()` (FinalAlignedStackDialog.java:298).
    fn add_listeners(&self) {
        // Bind the buttons to the action listener
        self.btn_filter
            .add_action_listener(self.final_aligned_stack_listener.clone());
        self.btn_view_filter
            .add_action_listener(self.final_aligned_stack_listener.clone());
        self.btn_use_filter
            .add_action_listener(self.final_aligned_stack_listener.clone());
        // MtfFileActionListener
        let adaptee = self.this.clone();
        self.btn_mtf_file
            .get_component()
            .add_action_listener(Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.btn_mtf_file_action(event);
                }
            }));
        // StartingAndEndingZKeyListener: key events are not modelled by the
        // Swing stand-in; the listener's keyReleased body is
        // `adaptee.startingAndEndingZKeyReleased(event)`.
        let adaptee = self.this.clone();
        self.ltf_starting_and_ending_z
            .add_key_listener(Rc::new(move || {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.starting_and_ending_z_key_released();
                }
            }));
        self.cb_use_expected_defocus
            .add_action_listener(Some(self.final_aligned_stack_listener.clone()));
        self.btn_ctf_plotter
            .add_action_listener(self.final_aligned_stack_listener.clone());
        self.btn_ctf_correction
            .add_action_listener(self.final_aligned_stack_listener.clone());
        self.btn_imod_ctf_correction
            .add_action_listener(self.final_aligned_stack_listener.clone());
        self.btn_use_ctf_correction
            .add_action_listener(self.final_aligned_stack_listener.clone());
        // TabChangeListener
        let adaptee = self.this.clone();
        self.tabbed_pane
            .get_component()
            .add_change_listener(Rc::new(move |_event| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.change_tab();
                }
            }));
        self.cb_dose_weight_filtering
            .add_action_listener(self.final_aligned_stack_listener.clone());
        self.cb_fixed_image_dose
            .add_action_listener(self.final_aligned_stack_listener.clone());
        self.rb_type_of_dose_file_mdoc_file
            .add_action_listener(self.final_aligned_stack_listener.clone());
        self.rb_type_of_dose_file_image_dose
            .add_action_listener(self.final_aligned_stack_listener.clone());
        self.rb_type_of_dose_file_accumulated_and_image_dose
            .add_action_listener(self.final_aligned_stack_listener.clone());
        self.rb_type_of_dose_file_prior_and_cumulative_dose
            .add_action_listener(self.final_aligned_stack_listener.clone());
        // Mouse adapter for context menu
        let context_menu: Weak<dyn ContextMenu> = self.this.clone();
        let mouse_adapter: Rc<dyn crate::imod::etomo::jdk::MouseListener> =
            GenericMouseAdapter::new(context_menu);
        self.base
            .root_panel
            .get_component()
            .add_mouse_listener(mouse_adapter.clone());
        self.tabbed_pane
            .get_component()
            .add_mouse_listener(mouse_adapter);
    }

    /// Java public static `getTilt3dFindButtonLabel()`.
    pub fn get_tilt3d_find_button_label() -> &'static str {
        tilt3d_find_panel::TILT_3D_FIND_LABEL
    }

    /// Java public static `getReprojectModelButtonLabel()`.
    pub fn get_reproject_model_button_label() -> &'static str {
        reproject_model_panel::REPROJECT_MODEL_LABEL
    }

    /// Java static package-private `getUseErasedStackLabel()`.
    pub fn get_use_erased_stack_label() -> &'static str {
        ccd_eraser_beads_panel::USE_ERASED_STACK_LABEL
    }

    /// Java static package-private `getErasedStackTabLabel()`.
    pub fn get_erased_stack_tab_label() -> &'static str {
        erase_gold_panel::ERASE_GOLD_TAB_LABEL
    }

    /// Java `isFiducialess()`.
    pub fn is_fiducialess(&self) -> bool {
        NewstackDisplay::is_fiducialess(&*self.newstack_or_blendmont_panel)
    }

    /// Java `setFilterButtonEnabled(boolean)`.
    pub fn set_filter_button_enabled(&self, enable: bool) {
        self.btn_filter.set_enabled(enable);
    }

    /// Java `setFilterButtonState(ReconScreenState)`.
    pub fn set_filter_button_state(&self, screen_state: &ReconScreenState) {
        self.btn_filter.set_button_state(
            screen_state.get_button_state(self.btn_filter.get_button_state_key().as_deref()),
        );
    }

    /// Java `setCtfCorrectionButtonState(ReconScreenState)`.
    pub fn set_ctf_correction_button_state(&self, screen_state: &ReconScreenState) {
        self.btn_ctf_correction.set_button_state(
            screen_state
                .get_button_state(self.btn_ctf_correction.get_button_state_key().as_deref()),
        );
    }

    /// Java `setViewFilterButtonEnabled(boolean)`.
    pub fn set_view_filter_button_enabled(&self, enable: bool) {
        self.btn_view_filter.set_enabled(enable);
    }

    /// Java `setVoltage(ConstEtomoNumber)`.
    pub fn set_voltage(&self, input: &ConstEtomoNumber) {
        self.ltf_voltage.set_text_const_etomo_number(Some(input));
    }

    /// Java `getConfigFile()`.
    pub fn get_config_file(&self) -> Option<String> {
        Field::get_text_void(&*self.ftf_config_file)
    }

    /// Java `setConfigFile(String)`.
    pub fn set_config_file(&self, input: Option<&str>) {
        self.ftf_config_file.set_text_string(input);
    }

    /// Java `setSphericalAberration(ConstEtomoNumber)`.
    pub fn set_spherical_aberration(&self, input: &ConstEtomoNumber) {
        self.ltf_spherical_aberration
            .set_text_const_etomo_number(Some(input));
    }

    /// Java `setInvertTiltAngles(boolean)`.
    pub fn set_invert_tilt_angles(&self, input: bool) {
        self.cb_invert_tilt_angles.set_selected_boolean(input);
    }

    /// Java `setAmplitudeContrast(ConstEtomoNumber)`.
    pub fn set_amplitude_contrast(&self, input: &ConstEtomoNumber) {
        self.ltf_amplitude_contrast
            .set_text_const_etomo_number(Some(input));
    }

    /// Java `setDefocusTol(ConstEtomoNumber)`.
    pub fn set_defocus_tol(&self, input: &ConstEtomoNumber) {
        self.ltf_defocus_tol
            .set_text_const_etomo_number(Some(input));
    }

    /// Java `setScanDefocusRange(String)`.
    pub fn set_scan_defocus_range(&self, input: Option<&str>) {
        self.ltf_scan_defocus_range.set_text_string(input);
    }

    /// Java `setExpectedDefocus(ConstEtomoNumber)`.
    pub fn set_expected_defocus(&self, input: &ConstEtomoNumber) {
        self.ltf_expected_defocus
            .set_text_const_etomo_number(Some(input));
    }

    /// Java `setPhaseShiftInDegrees(String)`.
    pub fn set_phase_shift_in_degrees(&self, input: Option<&str>) {
        self.tf_phase_shift_in_degrees.set_text_string(input);
    }

    /// Java `setOffsetToAdd(ConstEtomoNumber)`.
    pub fn set_offset_to_add(&self, input: &ConstEtomoNumber) {
        self.ltf_offset_to_add
            .set_text_const_etomo_number(Some(input));
    }

    /// Java `getDefocusTol(boolean) throws FieldValidationFailedException`.
    pub fn get_defocus_tol(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_defocus_tol.get_text_boolean(do_validation)
    }

    /// Java `setTiltState(TomogramState, ConstMetaData)`.
    pub fn set_tilt_state(&self, state: &TomogramState, meta_data: &dyn ConstMetaData) {
        self.erase_gold_panel.set_tilt_state(state, meta_data);
    }

    /// Java `getScanDefocusRange(boolean) throws FieldValidationFailedException`.
    pub fn get_scan_defocus_range(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_scan_defocus_range.get_text_boolean(do_validation)
    }

    /// Java `getExpectedDefocus(boolean) throws FieldValidationFailedException`.
    pub fn get_expected_defocus(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_expected_defocus.get_text_boolean(do_validation)
    }

    /// Java `getPhaseShiftInDegrees(boolean) throws FieldValidationFailedException`.
    pub fn get_phase_shift_in_degrees(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.tf_phase_shift_in_degrees
            .get_text_boolean(do_validation)
    }

    /// Java `getOffsetToAdd(boolean) throws FieldValidationFailedException`.
    pub fn get_offset_to_add(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_offset_to_add.get_text_boolean(do_validation)
    }

    /// Java `setUseFilterEnabled(boolean)`.
    pub fn set_use_filter_enabled(&self, enable: bool) {
        self.btn_use_filter.set_enabled(enable);
    }

    /// Java `setInterpolationWidth(ConstEtomoNumber)`.
    pub fn set_interpolation_width(&self, input: &ConstEtomoNumber) {
        self.ltf_interpolation_width
            .set_text_const_etomo_number(Some(input));
    }

    /// Java `setFiducialessAlignment(boolean)`.
    pub fn set_fiducialess_alignment(&self, input: bool) {
        self.newstack_or_blendmont_panel
            .set_fiducialess_alignment(input);
    }

    /// Java `setImageRotation(String)`.
    pub fn set_image_rotation(&self, input: Option<&str>) {
        self.newstack_or_blendmont_panel.set_image_rotation(input);
    }

    /// Java `setFilterHeaderState(PanelHeaderState)`.
    pub fn set_filter_header_state(&self, state: &PanelHeaderState) {
        self.filter_header.set_state(Some(state));
    }

    /// Java `setCtfCorrectionHeaderState(PanelHeaderState)`.
    pub fn set_ctf_correction_header_state(&self, state: &PanelHeaderState) {
        self.ctf_correction_header.set_state(Some(state));
    }

    /// Java `getCurTab()`.
    pub fn get_cur_tab(&self) -> Tab {
        self.cur_tab.get()
    }

    /// Java `getInterpolationWidth(boolean) throws FieldValidationFailedException`.
    pub fn get_interpolation_width(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_interpolation_width.get_text_boolean(do_validation)
    }

    /// Java `getStartingAndEndingZ()`.
    pub fn get_starting_and_ending_z(&self) -> String {
        self.ltf_starting_and_ending_z.get_text_void()
    }

    /// Java `getCtfPhaseFlipXAxisTilt(boolean) throws
    /// FieldValidationFailedException`.
    pub fn get_ctf_phase_flip_x_axis_tilt(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        if self.tf_ctf_phase_flip_x_axis_tilt.is_enabled() {
            return self
                .tf_ctf_phase_flip_x_axis_tilt
                .get_text_boolean(do_validation);
        }
        Ok(None)
    }

    /// Java `getScaleByCtfPower(boolean) throws FieldValidationFailedException`.
    pub fn get_scale_by_ctf_power(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        if self.tf_scale_by_ctf_power.is_enabled() {
            return self.tf_scale_by_ctf_power.get_text_boolean(do_validation);
        }
        Ok(None)
    }

    /// Java `getMinimumZeroSpacing(boolean) throws FieldValidationFailedException`.
    pub fn get_minimum_zero_spacing(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.tf_minimum_zero_spacing.get_text_boolean(do_validation)
    }

    /// Java `setCtfPhaseFlipXAxisTilt(String, boolean)`.
    pub fn set_ctf_phase_flip_x_axis_tilt(&self, x_axis_tilt: Option<&str>, from_tilt_com: bool) {
        let x_axis_tilt_set = x_axis_tilt.is_some_and(|x_axis_tilt| !x_axis_tilt.is_empty());
        // For CTF correction correcting for X axis tilt takes a long time, so don't switch it
        // on unless the setting is coming from the CTF correction com file.
        if !from_tilt_com && x_axis_tilt_set {
            self.cb_ctf_phase_flip_x_axis_tilt.set_selected(true);
        }
        if x_axis_tilt_set {
            self.tf_ctf_phase_flip_x_axis_tilt
                .set_text_string(x_axis_tilt);
        }
    }

    /// Java `setScaleByCtfPower(String)`.
    pub fn set_scale_by_ctf_power(&self, scale_by_ctf_power: Option<&str>) {
        self.cb_scale_by_ctf_power.set_selected(
            scale_by_ctf_power.is_some_and(|scale_by_ctf_power| !scale_by_ctf_power.is_empty()),
        );
        if self.cb_scale_by_ctf_power.is_selected() {
            self.tf_scale_by_ctf_power
                .set_text_string(scale_by_ctf_power);
        }
    }

    /// Java `setMinimumZeroSpacing(String)`.
    pub fn set_minimum_zero_spacing(&self, minimum_zero_spacing: Option<&str>) {
        self.tf_minimum_zero_spacing
            .set_text_string(minimum_zero_spacing);
    }

    /// Java `setUseExpectedDefocus(boolean)`.
    pub fn set_use_expected_defocus(&self, input: bool) {
        self.cb_use_expected_defocus.set_selected_boolean(input);
    }

    /// Java `setUseFilterButtonState(ReconScreenState)`.
    pub fn set_use_filter_button_state(&self, screen_state: &ReconScreenState) {
        self.btn_use_filter.set_button_state(
            screen_state.get_button_state(self.btn_use_filter.get_button_state_key().as_deref()),
        );
    }

    /// Java `setUseCtfCorrectionButtonState(ReconScreenState)`.
    pub fn set_use_ctf_correction_button_state(&self, screen_state: &ReconScreenState) {
        self.btn_use_ctf_correction.set_button_state(
            screen_state.get_button_state(
                self.btn_use_ctf_correction
                    .get_button_state_key()
                    .as_deref(),
            ),
        );
    }

    /// Java `getVoltage(boolean) throws FieldValidationFailedException`.
    pub fn get_voltage(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_voltage.get_text_boolean(do_validation)
    }

    /// Java `getSphericalAberration(boolean) throws FieldValidationFailedException`.
    pub fn get_spherical_aberration(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_spherical_aberration
            .get_text_boolean(do_validation)
    }

    /// Java `getInvertTiltAngles()`.
    pub fn get_invert_tilt_angles(&self) -> bool {
        self.cb_invert_tilt_angles.is_selected()
    }

    /// Java `getParameters(MetaData) throws FortranInputSyntaxException`.  The
    /// Metadata values that are from the setup dialog should not be overrided
    /// by this dialog unless the Metadata values are empty.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        meta_data.set_final_aligned_stack_dialog_saved(self.axis_id, true);
        self.dialog_saved.set(true);
        meta_data.set_erase_beads_initialized(self.erase_beads_initialized.get());
        meta_data.set_final_stack_ctf_correction_parallel_boolean(
            self.axis_id,
            self.is_parallel_process(),
        );
        meta_data.set_use_stack_ctf_phase_flip_x_axis_tilt(
            self.axis_id,
            self.cb_ctf_phase_flip_x_axis_tilt.is_selected(),
        );
        meta_data.set_stack_ctf_phase_flip_x_axis_tilt(
            self.axis_id,
            self.tf_ctf_phase_flip_x_axis_tilt
                .get_text_void()
                .as_deref(),
        );
        meta_data.set_stack_ctf_phase_flip_scale_by_ctf_power(
            self.axis_id,
            self.tf_scale_by_ctf_power.get_text_void().as_deref(),
        );
        meta_data.set_stack_mtf_filter_low_pass_radius_sigma(
            self.axis_id,
            Field::get_text_void(&*self.ltf_low_pass_radius_sigma).as_deref(),
        );
        meta_data.set_stack_mtf_filter_mtf_file(
            self.axis_id,
            Field::get_text_void(&*self.ltf_mtf_file).as_deref(),
        );
        meta_data.set_stack_mtf_filter_maximum_inverse(
            self.axis_id,
            Field::get_text_void(&*self.ltf_maximum_inverse).as_deref(),
        );
        meta_data.set_stack_mtf_filter_inverse_rolloff_radius_sigma(
            self.axis_id,
            Field::get_text_void(&*self.ltf_inverse_rolloff_radius_sigma).as_deref(),
        );
        meta_data.set_use_stack_mtf_filter_fixed_image_dose(
            self.axis_id,
            self.cb_fixed_image_dose.is_selected(),
        );
        meta_data.set_stack_mtf_filter_fixed_image_dose(
            self.axis_id,
            self.tf_fixed_image_dose.get_text_void().as_deref(),
        );
        meta_data.set_stack_mtf_filter_dose_weighting_file(
            self.axis_id,
            self.bctf_dose_weighting_file.get_text_void().as_deref(),
        );
        // ((AbstractRadioButtonModel) bgTypeOfDoseFile.getSelection()).getEnumeratedType()
        let enumerated_type: Option<EnumeratedTypeRef> = self
            .bg_type_of_dose_file
            .get_selection()
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<RadioEButtonModel>()
                    .and_then(|model| AbstractRadioButtonModel::get_enumerated_type(model))
            });
        meta_data.set_stack_mtf_filter_type_of_dose_file(self.axis_id, enumerated_type.as_deref());
        meta_data.set_stack_mtf_filter_voltage_200(self.axis_id, self.cb_voltage200.is_selected());
        meta_data.set_stack_mtf_filter_optimal_dose_scaling(
            self.axis_id,
            self.tf_optimal_dose_scaling.get_text_void().as_deref(),
        );
        meta_data.set_stack_mtf_filter_bidirectional_num_views(
            self.axis_id,
            self.tf_bidirectional_num_views.get_text_void().as_deref(),
        );
        self.newstack_or_blendmont_panel
            .get_parameters_meta_data(meta_data)?;
        self.erase_gold_panel.get_parameters_meta_data(meta_data)?;
        Ok(())
    }

    /// Java `getBlendmontDisplay()`.
    pub fn get_blendmont_display(&self) -> Option<Rc<dyn BlendmontDisplay>> {
        if self.application_manager.get_meta_data().get_view_type() == ViewType::Montage {
            return Some(self.newstack_or_blendmont_panel.as_blendmont_display());
        }
        None
    }

    /// Java `getBlendmont3dFindDisplay()`.
    pub fn get_blendmont3d_find_display(&self) -> Option<Rc<dyn BlendmontDisplay>> {
        self.erase_gold_panel.get_blendmont3d_find_display()
    }

    /// Java `getNewstackDisplay()`.
    pub fn get_newstack_display(&self) -> Option<Rc<dyn NewstackDisplay>> {
        if self.application_manager.get_meta_data().get_view_type() != ViewType::Montage {
            return Some(self.newstack_or_blendmont_panel.as_newstack_display());
        }
        None
    }

    /// Java `getNewstack3dFindDisplay()`.
    pub fn get_newstack3d_find_display(&self) -> Option<Rc<dyn NewstackDisplay>> {
        self.erase_gold_panel.get_newstack3d_find_display()
    }

    /// Java `getTilt3dFindDisplay()`.
    pub fn get_tilt3d_find_display(&self) -> Option<Rc<dyn TiltDisplay>> {
        self.erase_gold_panel.get_tilt3d_find_display()
    }

    /// Java `getFindBeads3dDisplay()`.
    pub fn get_find_beads3d_display(&self) -> Option<Rc<dyn FindBeads3dDisplay>> {
        self.erase_gold_panel.get_find_beads3d_display()
    }

    /// Java `getCcdEraserBeadsDisplay()`.
    pub fn get_ccd_eraser_beads_display(&self) -> Option<Rc<dyn CcdEraserDisplay>> {
        self.erase_gold_panel.get_ccd_eraser_beads_display()
    }

    /// Java `getFiducialessParams()`.
    pub fn get_fiducialess_params(&self) -> Rc<dyn FiducialessParams> {
        self.newstack_or_blendmont_panel.get_fiducialess_params()
    }

    /// Java `getAmplitudeContrast(boolean) throws FieldValidationFailedException`.
    pub fn get_amplitude_contrast(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_amplitude_contrast.get_text_boolean(do_validation)
    }

    /// Java `getFilterHeaderState(PanelHeaderState)`.
    pub fn get_filter_header_state(&self, state: &PanelHeaderState) {
        self.filter_header.get_state(Some(state));
    }

    /// Java `getCtfCorrectionHeaderState(PanelHeaderState)`.
    pub fn get_ctf_correction_header_state(&self, state: &PanelHeaderState) {
        self.ctf_correction_header.get_state(Some(state));
    }

    /// Java `getParameters(CtfPhaseFlipParam)`.
    pub fn get_parameters_ctf_phase_flip_param(&self, param: &mut CtfPhaseFlipParam) {
        self.cpu_gpu_panel
            .get_parameters_ctf_phase_flip_param(param);
    }

    /// Java `setParameters(CtfPhaseFlipParam, boolean)`.  (The Java class
    /// declares it twice, for `CtfPhaseFlipParam` and `ConstCtfPhaseFlipParam`;
    /// both forward to the panel's `ConstCtfPhaseFlipParam` overload.)
    pub fn set_parameters_ctf_phase_flip_param_boolean(
        &self,
        param: &CtfPhaseFlipParam,
        initialize: bool,
    ) {
        self.cpu_gpu_panel
            .set_parameters_const_ctf_phase_flip_param_boolean(param, initialize);
    }

    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.newstack_or_blendmont_panel
            .get_parameters_recon_screen_state(screen_state);
        self.erase_gold_panel
            .get_parameters_recon_screen_state(screen_state);
        self.get_filter_header_state(screen_state.get_stack_mtffilter_header_state());
        self.get_ctf_correction_header_state(screen_state.get_stack_ctf_correction_header_state());
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.newstack_or_blendmont_panel
            .set_parameters_recon_screen_state(screen_state);
        self.erase_gold_panel
            .set_parameters_recon_screen_state(screen_state);
        self.set_filter_header_state(screen_state.get_stack_mtffilter_header_state());
        self.set_ctf_correction_header_state(screen_state.get_stack_ctf_correction_header_state());
        self.set_use_filter_button_state(screen_state);
        self.set_filter_button_state(screen_state);
        self.set_ctf_correction_button_state(screen_state);
    }

    /// Java `setStartingAndEndingZ(String)`.
    pub fn set_starting_and_ending_z(&self, starting_and_ending_z: Option<&str>) {
        self.ltf_starting_and_ending_z
            .set_text_string(starting_and_ending_z);
    }

    /// Java private `updateAdvancedFilter(boolean)`.
    fn update_advanced_filter(&self, advanced: bool) {
        self.ltf_starting_and_ending_z.set_visible(advanced);
        self.inverse_params_panel.set_visible(advanced);
        self.tf_optimal_dose_scaling.set_visible(advanced);
        self.tf_bidirectional_num_views.set_visible(advanced);
    }

    /// Java private `updateAdvancedCtfCorrection(boolean)`.
    fn update_advanced_ctf_correction(&self, advanced: bool) {
        self.ltf_amplitude_contrast.set_visible(advanced);
        self.ltf_defocus_tol.set_visible(advanced);
        self.tf_minimum_zero_spacing.set_visible(advanced);
        self.cb_invert_tilt_angles.set_visible(advanced);
        self.ltf_offset_to_add.set_visible(advanced);
    }

    /// Java `updateAlignedStackBinning()`.
    pub fn update_aligned_stack_binning(&self) {
        self.erase_gold_panel.update_aligned_stack_binning();
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &MetaData) {
        let axis_id = self.axis_id;
        self.dialog_saved
            .set(meta_data.is_final_aligned_stack_dialog_saved(axis_id));
        let ctf3d = meta_data.is_ctf_3d_setup_slab_thickness_in_nm_set();
        self.l_ctf3d_correct_ctf.set_visible(ctf3d);
        self.l_ctf3d_2d_filter1.set_visible(ctf3d);
        self.l_ctf3d_2d_filter2.set_visible(ctf3d);
        self.erase_beads_initialized
            .set(meta_data.is_erase_beads_initialized());
        self.cb_ctf_phase_flip_x_axis_tilt
            .set_selected(meta_data.is_use_stack_ctf_phase_flip_x_axis_tilt(axis_id));
        self.tf_ctf_phase_flip_x_axis_tilt.set_text_string(Some(
            &meta_data.get_stack_ctf_phase_flip_x_axis_tilt(axis_id),
        ));
        self.tf_scale_by_ctf_power.set_text_string(Some(
            &meta_data.get_stack_ctf_phase_flip_scale_by_ctf_power(axis_id),
        ));
        self.ltf_low_pass_radius_sigma.set_text_string(Some(
            &meta_data.get_stack_mtf_filter_low_pass_radius_sigma(axis_id),
        ));
        self.ltf_mtf_file
            .set_text_string(Some(&meta_data.get_stack_mtf_filter_mtf_file(axis_id)));
        self.ltf_maximum_inverse.set_text_string(Some(
            &meta_data.get_stack_mtf_filter_maximum_inverse(axis_id),
        ));
        self.ltf_inverse_rolloff_radius_sigma.set_text_string(Some(
            &meta_data.get_stack_mtf_filter_inverse_rolloff_radius_sigma(axis_id),
        ));
        self.cb_fixed_image_dose
            .set_selected(meta_data.is_use_stack_mtf_filter_fixed_image_dose(axis_id));
        self.tf_fixed_image_dose.set_text_string(Some(
            &meta_data.get_stack_mtf_filter_fixed_image_dose(axis_id),
        ));
        let dose_weighting_file = meta_data.get_stack_mtf_filter_dose_weighting_file(axis_id);
        if !dose_weighting_file.is_empty() && !dose_weighting_file.starts_with('.') {
            self.bctf_dose_weighting_file
                .set_file(Some(&dose_weighting_file));
        }
        let type_of_dose_file = TypeOfDoseFile::get_instance(Some(
            &meta_data.get_stack_mtf_filter_type_of_dose_file(axis_id),
        ));
        self.bg_type_of_dose_file
            .set_selected(Some(&EnumeratedTypeRef::new(type_of_dose_file)));
        self.cb_voltage200
            .set_selected(meta_data.is_stack_mtf_filter_voltage_200(axis_id));
        self.tf_optimal_dose_scaling.set_text_string(Some(
            &meta_data.get_stack_mtf_filter_optimal_dose_scaling(axis_id),
        ));
        self.tf_bidirectional_num_views.set_text_string(Some(
            &meta_data.get_stack_mtf_filter_bidirectional_num_views(axis_id),
        ));

        let parallel_process = meta_data.get_final_stack_ctf_correction_parallel(axis_id);
        self.cpu_gpu_panel
            .set_parameters_const_meta_data_const_etomo_number(
                meta_data,
                parallel_process
                    .as_ref()
                    .map(|parallel_process| -> &ConstEtomoNumber { parallel_process }),
            );
        self.newstack_or_blendmont_panel
            .set_parameters_const_meta_data(meta_data);
        self.erase_gold_panel
            .set_parameters_const_meta_data(meta_data);
        if let (Some(mediator), Some(this)) = (&self.mediator, self.this.upgrade()) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.set_method_process_interface_processing_method(
                &origin,
                ProcessInterface::get_processing_method(self),
            );
        }
    }

    /// Java `setParameters(BlendmontParam)`.
    pub fn set_parameters_blendmont_param(&self, param: &BlendmontParam) {
        if self.application_manager.get_meta_data().get_view_type() == ViewType::Montage {
            self.newstack_or_blendmont_panel
                .as_blendmont_display()
                .set_parameters(param);
        }
    }

    /// Java `setEraseGoldParameters(BlendmontParam)`.
    pub fn set_erase_gold_parameters_blendmont_param(&self, param: &BlendmontParam) {
        self.erase_gold_panel.set_parameters_blendmont_param(param);
    }

    /// Java `setEraseGoldParameters(NewstParam)`.
    pub fn set_erase_gold_parameters_newst_param(&self, param: &NewstParam) {
        self.erase_gold_panel.set_parameters_newst_param(param);
    }

    /// Java `setParameters(ConstTiltParam, boolean) throws
    /// FileNotFoundException, IOException`.
    pub fn set_parameters_const_tilt_param_boolean(
        &self,
        param: &dyn ConstTiltParam,
        calculate_values: bool,
    ) -> Result<(), std::io::Error> {
        self.erase_gold_panel
            .set_parameters_const_tilt_param_boolean(param, calculate_values)
    }

    /// Java `setParameters(ConstFindBeads3dParam, boolean)`.
    pub fn set_parameters_const_find_beads3d_param_boolean(
        &self,
        param: &dyn ConstFindBeads3dParam,
        initialize: bool,
    ) {
        self.erase_gold_panel
            .set_parameters_const_find_beads3d_param_boolean(param, initialize);
    }

    // <p>updates done</p>

    /// Java `setParameters(ConstCtfPhaseFlipParam, boolean)`.
    pub fn set_parameters_const_ctf_phase_flip_param_boolean(
        &self,
        param: &dyn ConstCtfPhaseFlipParam,
        initialize: bool,
    ) {
        self.cpu_gpu_panel
            .set_parameters_const_ctf_phase_flip_param_boolean(param, initialize);
    }

    /// Java `initialize()`.
    pub fn initialize(&self) {
        self.erase_gold_panel.initialize();
    }

    /// Java `setParameters(ConstTiltalignParam, boolean)`.
    pub fn set_parameters_const_tiltalign_param_boolean(
        &self,
        param: &ConstTiltalignParam,
        initialize: bool,
    ) {
        self.erase_gold_panel
            .set_parameters_const_tiltalign_param_boolean(param, initialize);
    }

    /// Java `setParameters(ConstCCDEraserParam)`.
    pub fn set_parameters_const_ccd_eraser_param(&self, param: &ConstCCDEraserParam) {
        self.erase_gold_panel
            .set_parameters_const_ccd_eraser_param(param);
    }

    /// Java `setOverrideParameters(ConstMetaData)`.
    pub fn set_override_parameters(&self, meta_data: &dyn ConstMetaData) {
        self.erase_gold_panel.set_override_parameters(meta_data);
    }

    /// Java `setParameters(ConstNewstParam)`.
    pub fn set_parameters_const_newst_param(&self, param: &dyn ConstNewstParam) {
        if self.application_manager.get_meta_data().get_view_type() != ViewType::Montage {
            self.newstack_or_blendmont_panel
                .as_newstack_display()
                .set_parameters(param);
        }
    }

    /// Java private `updateAdvanced()`: update the dialog with the current
    /// advanced state.
    fn update_advanced(&self) {
        let advanced = self.is_advanced();
        self.newstack_or_blendmont_panel.update_advanced(advanced);
        self.erase_gold_panel.update_advanced(advanced);
        self.update_advanced_filter(advanced);
        self.update_advanced_ctf_correction(advanced);
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }

    /// Java `isParallelProcess()`.
    pub fn is_parallel_process(&self) -> bool {
        self.cpu_gpu_panel.is_parallel_process()
    }

    /// Java `isUseExpectedDefocus()`.
    pub fn is_use_expected_defocus(&self) -> bool {
        self.cb_use_expected_defocus.is_selected()
    }

    /// Java `isCtfPhaseFlipXAxisTiltEmpty()`.
    pub fn is_ctf_phase_flip_x_axis_tilt_empty(&self) -> bool {
        self.tf_ctf_phase_flip_x_axis_tilt.is_empty()
    }

    /// Java `setTiltComParameters(ConstTiltParam)`.
    pub fn set_tilt_com_parameters(&self, param: &dyn ConstTiltParam) {
        self.set_ctf_phase_flip_x_axis_tilt(Some(&param.get_x_axis_tilt_string()), true);
    }

    /// Java private `layoutCcdEraser()`.
    fn layout_ccd_eraser(&self) {
        // panel
        let ccd_eraser_root = JComponent::new_panel();
        self.tabbed_pane
            .add_tab_string_component(erase_gold_panel::ERASE_GOLD_TAB_LABEL, &ccd_eraser_root);
    }

    /// Java private `layoutCtfCorrectionPanel()`.
    fn layout_ctf_correction_panel(&self) {
        // panels
        let ctf_correction_root = JComponent::new_panel();
        let pnl_ctf_phase_flip_x_axis_tilt = JComponent::new_panel();
        let pnl_scale_by_ctf_power = JComponent::new_panel();
        let pnl_ctf3d_correct_ctf = JComponent::new_panel();
        // init
        self.l_ctf3d_correct_ctf
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_ctf3d_correct_ctf.set_visible(false);
        //
        self.tabbed_pane
            .add_tab_string_component(CTF_TAB_LABEL, &ctf_correction_root);
        // Swing layout: ctfCorrectionMainPanel BoxLayout Y_AXIS; untitled
        // etched border.
        let ctf_correction_main_panel = self.ctf_correction_main_panel.get_component();
        ctf_correction_main_panel.add(&self.ctf_correction_header.get_container());
        ctf_correction_main_panel.add(&self.ctf_correction_body_panel.get_container());
        // body
        self.ctf_correction_body_panel
            .set_box_layout(spaced_panel::Y_AXIS);
        self.ctf_correction_body_panel
            .add_labeled_text_field(&self.ltf_voltage);
        self.ctf_correction_body_panel
            .add_labeled_text_field(&self.ltf_spherical_aberration);
        self.ctf_correction_body_panel
            .add_labeled_text_field(&self.ltf_amplitude_contrast);
        let pnl_invert_tilt_angles = JComponent::new_panel();
        // Swing layout: pnlInvertTiltAngles BoxLayout X_AXIS, CENTER_ALIGNMENT.
        pnl_invert_tilt_angles.add(&self.cb_invert_tilt_angles.get_component());
        // Swing layout: pnlInvertTiltAngles.add(Box.createHorizontalGlue()).
        self.ctf_correction_body_panel
            .add_j_panel(&pnl_invert_tilt_angles);
        // ctf plotter
        let ctf_plotter_panel = SpacedPanel::get_instance_void();
        ctf_plotter_panel.set_box_layout(spaced_panel::Y_AXIS);
        ctf_plotter_panel.set_border(&EtchedBorder::new(Some("CTF Plotter")).get_border());
        // Component.CENTER_ALIGNMENT
        ctf_plotter_panel.set_component_alignment_x(0.5);
        self.ctf_correction_body_panel
            .add_spaced_panel(&ctf_plotter_panel);
        ctf_plotter_panel.add_component(&UIComponent::get_component(&*self.ftf_config_file));
        ctf_plotter_panel.add_labeled_text_field(&self.ltf_scan_defocus_range);
        ctf_plotter_panel.add_labeled_text_field(&self.ltf_expected_defocus);
        ctf_plotter_panel.add_component(&self.tf_phase_shift_in_degrees.get_component());
        ctf_plotter_panel.add_labeled_text_field(&self.ltf_offset_to_add);
        ctf_plotter_panel.add_multi_line_button(&self.btn_ctf_plotter);
        // ctf phase flip
        let ctf_correction_panel = SpacedPanel::get_instance_void();
        ctf_correction_panel.set_box_layout(spaced_panel::Y_AXIS);
        ctf_correction_panel.set_border(
            &EtchedBorder::new(Some(shared_strings::CTF_CORRECTION_LABEL)).get_border(),
        );
        self.ctf_correction_body_panel
            .add_spaced_panel(&ctf_correction_panel);
        // use expected defocus
        let use_expected_defocus_panel = JComponent::new_panel();
        // Swing layout: useExpectedDefocusPanel BoxLayout X_AXIS, CENTER_ALIGNMENT.
        ctf_correction_panel.add_j_panel(&use_expected_defocus_panel);
        use_expected_defocus_panel.add(&self.cb_use_expected_defocus.get_component());
        // Swing layout: useExpectedDefocusPanel.add(Box.createHorizontalGlue()).

        ctf_correction_panel.add_component(&self.cpu_gpu_panel.get_component());
        ctf_correction_panel.add_labeled_text_field(&self.ltf_interpolation_width);
        ctf_correction_panel.add_j_panel(&pnl_ctf_phase_flip_x_axis_tilt);
        ctf_correction_panel.add_component(&pnl_scale_by_ctf_power);
        ctf_correction_panel.add_labeled_text_field(&self.ltf_defocus_tol);
        ctf_correction_panel.add_component(&self.tf_minimum_zero_spacing.get_component());
        // Swing layout: ctfCorrectionPanel.add(Box.createVerticalStrut(3)).
        ctf_correction_panel.add_j_panel(&pnl_ctf3d_correct_ctf);
        // Swing layout: ctfCorrectionPanel.add(Box.createVerticalStrut(5)).
        // Ctf3dCorrectCtf
        // Swing layout: pnlCtf3dCorrectCtf BoxLayout X_AXIS.
        pnl_ctf3d_correct_ctf.add(&self.l_ctf3d_correct_ctf.get_component());
        // CtfPhaseFlipXAxisTilt
        // Swing layout: pnlCtfPhaseFlipXAxisTilt BoxLayout X_AXIS.
        pnl_ctf_phase_flip_x_axis_tilt.add(&self.cb_ctf_phase_flip_x_axis_tilt.get_component());
        pnl_ctf_phase_flip_x_axis_tilt.add(&self.tf_ctf_phase_flip_x_axis_tilt.get_component());
        // TODO 2220
        pnl_ctf_phase_flip_x_axis_tilt.add(&JComponent::new_label(" (takes much longer)"));
        // ScaleByCtfPower
        // Swing layout: pnlScaleByCtfPower BoxLayout X_AXIS.
        pnl_scale_by_ctf_power.add(&self.cb_scale_by_ctf_power.get_component());
        pnl_scale_by_ctf_power.add(&self.tf_scale_by_ctf_power.get_component());
        // buttons
        let button_panel = SpacedPanel::get_instance_void();
        button_panel.set_box_layout(spaced_panel::X_AXIS);
        ctf_correction_panel.add_spaced_panel(&button_panel);
        button_panel.add_horizontal_glue();
        button_panel.add_multi_line_button(&self.btn_ctf_correction);
        button_panel.add_multi_line_button(&self.btn_imod_ctf_correction);
        button_panel.add_multi_line_button(&self.btn_use_ctf_correction);
        button_panel.add_horizontal_glue();
        // init
        self.ctf_correction_header.set_open(false);
    }

    /// Java private `layoutNewstPanel()`: layout the newstack panel.
    fn layout_newst_panel(&self) {
        // panels
        let newst_root = JComponent::new_panel();
        self.tabbed_pane
            .add_tab_string_component(FINAL_ALIGNED_STACK_TAB_LABEL, &newst_root);
        newst_root.add(&self.newstack_or_blendmont_panel.get_component());
    }

    /// Java private `layoutFilterPanel()`: layout the MTF filter panel.
    fn layout_filter_panel(&self) {
        // panels
        let filter_root = JComponent::new_panel();
        let pnl_dose_weight_filtering = JComponent::new_panel();
        let pnl_dose_weighting = JComponent::new_panel();
        let pnl_fixed_image_dose = JComponent::new_panel();
        let pnl_voltage200 = JComponent::new_panel();
        let pnl_glue_type_of_dose_file = JComponent::new_panel();
        let pnl_ctf3d_2d_filter = JComponent::new_panel();
        // init
        self.tabbed_pane
            .add_tab_string_component(MTF_FILTER_TAB_LABEL, &filter_root);
        self.bctf_dose_weighting_file
            .set_file_filter(Some(Rc::new(DoseWeightingFileFilter::new())));
        self.bctf_dose_weighting_file
            .set_select_file_dir(self.application_manager.get_property_user_dir().as_deref());
        self.tf_fixed_image_dose.set_must_be_positive(true);
        self.l_ctf3d_2d_filter1
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_ctf3d_2d_filter2
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_ctf3d_2d_filter1.set_visible(false);
        self.l_ctf3d_2d_filter2.set_visible(false);
        // filter
        // Swing layout: filterPanel BoxLayout Y_AXIS; untitled etched border.
        self.filter_panel.add(&self.filter_header.get_container());
        self.filter_panel.add(&self.filter_body_panel);
        // Swing layout: filterBodyPanel BoxLayout Y_AXIS.
        self.filter_body_panel.add(&pnl_dose_weight_filtering);
        self.filter_body_panel.add(&self.pnl_uniform_filtering);
        self.filter_body_panel.add(&pnl_dose_weighting);
        // DoseWeightFiltering
        // Swing layout: pnlDoseWeightFiltering BoxLayout X_AXIS.
        pnl_dose_weight_filtering.add(&self.cb_dose_weight_filtering.get_component());
        // Swing layout: pnlDoseWeightFiltering.add(Box.createHorizontalGlue()).
        // UniformFiltering
        self.pnl_uniform_filtering.set_border_title(
            EtchedBorder::new(Some("Uniform Filtering"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        // Swing layout: pnlUniformFiltering BoxLayout Y_AXIS.
        self.pnl_uniform_filtering
            .add(&self.ltf_low_pass_radius_sigma.get_container());
        self.pnl_uniform_filtering
            .add(&self.inverse_params_panel.get_container());
        // DoseWeighting
        pnl_dose_weighting.set_border_title(
            EtchedBorder::new(Some(shared_strings::DOSE_WEIGHTING_LABEL))
                .get_border()
                .get_title()
                .as_deref(),
        );
        // Swing layout: pnlDoseWeighting BoxLayout Y_AXIS.
        pnl_dose_weighting.add(&pnl_fixed_image_dose);
        pnl_dose_weighting.add(&pnl_glue_type_of_dose_file);
        // Swing layout: pnlDoseWeighting.add(Box.createRigidArea(FixedDim.x0_y1)).
        pnl_dose_weighting.add(&self.bctf_dose_weighting_file.get_component());
        pnl_dose_weighting.add(&pnl_voltage200);
        pnl_dose_weighting.add(&self.tf_optimal_dose_scaling.get_component());
        // Swing layout: pnlDoseWeighting.add(Box.createRigidArea(FixedDim.x0_y1)).
        pnl_dose_weighting.add(&self.tf_bidirectional_num_views.get_component());
        // GlueTypeOfDoseFile
        // Swing layout: pnlGlueTypeOfDoseFile BoxLayout X_AXIS, horizontal glue
        // on both sides of pnlTypeOfDoseFile.
        pnl_glue_type_of_dose_file.add(&self.pnl_type_of_dose_file);
        // TypeOfDoseFile
        self.pnl_type_of_dose_file.set_border_title(
            EtchedBorder::new(Some("Type of Dose File"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        // Swing layout: pnlTypeOfDoseFile BoxLayout Y_AXIS.
        self.pnl_type_of_dose_file
            .add(&self.rb_type_of_dose_file_mdoc_file.get_component());
        self.pnl_type_of_dose_file
            .add(&self.rb_type_of_dose_file_image_dose.get_component());
        self.pnl_type_of_dose_file.add(
            &self
                .rb_type_of_dose_file_accumulated_and_image_dose
                .get_component(),
        );
        self.pnl_type_of_dose_file.add(
            &self
                .rb_type_of_dose_file_prior_and_cumulative_dose
                .get_component(),
        );
        // Voltage200
        // Swing layout: pnlVoltage200 BoxLayout X_AXIS.
        pnl_voltage200.add(&self.cb_voltage200.get_component());
        // Swing layout: pnlVoltage200.add(Box.createHorizontalGlue()).
        // inverseParams
        self.inverse_params_panel
            .set_box_layout(spaced_panel::Y_AXIS);
        self.inverse_params_panel
            .set_border(&EtchedBorder::new(Some("Inverse Filtering Parameters: ")).get_border());
        // FixedImageDose
        // Swing layout: pnlFixedImageDose BoxLayout X_AXIS.
        pnl_fixed_image_dose.add(&self.cb_fixed_image_dose.get_component());
        pnl_fixed_image_dose.add(&self.tf_fixed_image_dose.get_component());
        pnl_fixed_image_dose.add(&self.l_fixed_image_dose);
        let mtf_file_panel = SpacedPanel::get_instance_void();
        mtf_file_panel.set_box_layout(spaced_panel::X_AXIS);
        let inverse_panel = SpacedPanel::get_instance_void();
        inverse_panel.set_box_layout(spaced_panel::X_AXIS);
        let button_panel = SpacedPanel::get_instance_boolean(true);
        button_panel.set_box_layout(spaced_panel::X_AXIS);
        // buttonPanel
        button_panel.add_multi_line_button(&self.btn_filter);
        button_panel.add_multi_line_button(&self.btn_view_filter);
        button_panel.add_multi_line_button(&self.btn_use_filter);
        // inversePanel
        inverse_panel.add_labeled_text_field(&self.ltf_maximum_inverse);
        inverse_panel.add_labeled_text_field(&self.ltf_inverse_rolloff_radius_sigma);
        // mtfFilePanel
        mtf_file_panel.add_labeled_text_field(&self.ltf_mtf_file);
        mtf_file_panel.add_j_button(&self.btn_mtf_file.get_component());
        // inverseParamsPanel
        self.inverse_params_panel.add_spaced_panel(&mtf_file_panel);
        self.inverse_params_panel.add_spaced_panel(&inverse_panel);
        // filterBodyPanel
        // Swing layout: filterBodyPanel.add(Box.createRigidArea(FixedDim.x0_y5)).
        self.filter_body_panel
            .add(&self.ltf_starting_and_ending_z.get_container());
        // Swing layout: filterBodyPanel.add(Box.createVerticalStrut(5)).
        self.filter_body_panel.add(&pnl_ctf3d_2d_filter);
        // Swing layout: filterBodyPanel.add(Box.createVerticalStrut(7)).
        self.filter_body_panel.add(&button_panel.get_container());
        // Ctf3d2dFilter
        // Swing layout: pnlCtf3d2dFilter BoxLayout Y_AXIS.
        pnl_ctf3d_2d_filter.add(&self.l_ctf3d_2d_filter1.get_component());
        pnl_ctf3d_2d_filter.add(&self.l_ctf3d_2d_filter2.get_component());
        // Component.CENTER_ALIGNMENT
        ui_utilities::align_components_x(&pnl_ctf3d_2d_filter, 0.5);
        self.update_display();
    }

    /// Java `btnMtfFileAction(ActionEvent)`.
    pub fn btn_mtf_file_action(&self, _event: &ActionEvent) {
        // Open up the file chooser in the $IMOD_CALIB_DIR/Camera, if available,
        // otherwise open in the working directory
        let mut current_mtf_directory: Option<String>;
        // try { ... } catch (FieldValidationFailedException e) { e.printStackTrace(); }
        match self.ltf_mtf_file.get_text_boolean(true) {
            Err(e) => {
                eprintln!("{e}");
            }
            Ok(text) => {
                current_mtf_directory = text;
                // Java `currentMtfDirectory.equals("")`; the field's text is never
                // null in Java, so a missing value is treated as empty.
                if current_mtf_directory.as_deref().unwrap_or("").is_empty() {
                    let calibration_dir: Option<PathBuf> =
                        etomo_director::INSTANCE.get_imod_calib_directory();
                    // Upstream bug fixed in translation
                    // (FinalAlignedStackDialog.java:1058): Java calls
                    // `calibrationDir.getAbsolutePath()` on a null calibration
                    // directory (IMOD_CALIB_DIR unset) and throws a
                    // NullPointerException out of the button handler.  Here a
                    // missing calibration directory is treated as a camera
                    // directory that does not exist, so the working directory is
                    // used, as the Java does when the directory is absent.
                    let camera_dir: Option<PathBuf> = calibration_dir.map(|calibration_dir| {
                        PathBuf::from(utilities::java_io_file_get_absolute_path(
                            &calibration_dir.to_string_lossy(),
                        ))
                        .join("Camera")
                    });
                    if let Some(camera_dir) = camera_dir.filter(|camera_dir| camera_dir.exists()) {
                        current_mtf_directory = Some(utilities::java_io_file_get_absolute_path(
                            &camera_dir.to_string_lossy(),
                        ));
                    } else {
                        current_mtf_directory = self.application_manager.get_property_user_dir();
                    }
                }
                let manager: &'static dyn BaseManager = self.application_manager;
                let chooser: Rc<FileChooser> = FileChooser::new_base_manager_string(
                    Some(manager),
                    current_mtf_directory.as_deref(),
                );
                let mtf_file_filter = MtfFileFilter::new();
                chooser.set_file_filter(Some(Rc::new(mtf_file_filter)));
                // Swing layout: chooser.setPreferredSize(FixedDim.fileChooser).
                chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
                let return_val =
                    chooser.show_open_dialog(Some(&self.base.root_panel.get_component()));
                if return_val == file_chooser::APPROVE_OPTION {
                    let mtf_file = chooser.get_selected_file();
                    // try { ... } catch (Exception excep) { excep.printStackTrace(); }
                    if let Some(mtf_file) = mtf_file {
                        self.ltf_mtf_file.set_text_string(Some(
                            &utilities::java_io_file_get_absolute_path(&mtf_file.to_string_lossy()),
                        ));
                    }
                }
            }
        }
    }

    /// Java `startingAndEndingZKeyReleased(KeyEvent)`.
    pub fn starting_and_ending_z_key_released(&self) {
        if let Some(expert) = self.expert.upgrade() {
            expert.enable_use_filter();
        }
    }

    /// Java `updateCtfPlotter()`.
    pub fn update_ctf_plotter(&self) {
        let enable = !self.cb_use_expected_defocus.is_selected();
        self.ftf_config_file.set_enabled(enable);
        self.btn_ctf_plotter.set_enabled(enable);
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        let dose_weight_filtering = self.cb_dose_weight_filtering.is_selected();
        self.pnl_uniform_filtering
            .set_enabled(!dose_weight_filtering);
        self.ltf_low_pass_radius_sigma
            .set_enabled(!dose_weight_filtering);
        self.inverse_params_panel
            .set_enabled(!dose_weight_filtering);
        self.ltf_mtf_file.set_enabled(!dose_weight_filtering);
        self.btn_mtf_file
            .get_component()
            .set_enabled(!dose_weight_filtering);
        self.ltf_maximum_inverse.set_enabled(!dose_weight_filtering);
        self.ltf_inverse_rolloff_radius_sigma
            .set_enabled(!dose_weight_filtering);
        self.cb_fixed_image_dose.set_enabled(dose_weight_filtering);
        //
        let fixed_image_dose = self.cb_fixed_image_dose.is_selected();
        self.tf_fixed_image_dose
            .set_enabled(dose_weight_filtering && fixed_image_dose);
        self.l_fixed_image_dose.set_enabled(dose_weight_filtering);
        self.bctf_dose_weighting_file
            .set_enabled(dose_weight_filtering && !fixed_image_dose);
        self.pnl_type_of_dose_file
            .set_enabled(dose_weight_filtering && !fixed_image_dose);
        self.rb_type_of_dose_file_mdoc_file
            .set_enabled(dose_weight_filtering && !fixed_image_dose);
        self.rb_type_of_dose_file_image_dose
            .set_enabled(dose_weight_filtering && !fixed_image_dose);
        self.rb_type_of_dose_file_accumulated_and_image_dose
            .set_enabled(dose_weight_filtering && !fixed_image_dose);
        self.rb_type_of_dose_file_prior_and_cumulative_dose
            .set_enabled(dose_weight_filtering && !fixed_image_dose);
        self.cb_voltage200.set_enabled(dose_weight_filtering);
        self.tf_optimal_dose_scaling
            .set_enabled(dose_weight_filtering);
        // ((TypeOfDoseFile) ((AbstractRadioButtonModel) bgTypeOfDoseFile.getSelection())
        // .getEnumeratedType()).enableBidirectionalNumViews - evaluated only when
        // doseWeightFiltering is true (Java's &&).  The group always has a
        // selection (TypeOfDoseFile.DEFAULT selects itself when its radio button
        // is constructed); a missing one would be a NullPointerException in Java
        // and reads as false here.
        self.tf_bidirectional_num_views.set_enabled(
            dose_weight_filtering
                && self
                    .bg_type_of_dose_file
                    .get_selection()
                    .and_then(|button| button.get_model())
                    .and_then(|model| {
                        model
                            .as_any()
                            .downcast_ref::<RadioEButtonModel>()
                            .and_then(|model| AbstractRadioButtonModel::get_enumerated_type(model))
                    })
                    .and_then(|enumerated_type| {
                        enumerated_type
                            .downcast_ref::<TypeOfDoseFile>()
                            .map(|type_of_dose_file| {
                                type_of_dose_file.enable_bidirectional_num_views()
                            })
                    })
                    .unwrap_or(false),
        );
    }

    /// Java `getParameters(MTFFilterParam, boolean) throws
    /// FortranInputSyntaxException`.
    pub fn get_parameters_mtf_filter_param_boolean(
        &self,
        param: &mut MTFFilterParam,
        do_validation: bool,
    ) -> Result<bool, MTFFilterParametersException> {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        macro_rules! field {
            ($e:expr) => {
                match $e {
                    Ok(value) => value,
                    Err(_) => return Ok(false),
                }
            };
        }
        if Field::is_enabled(&*self.ltf_low_pass_radius_sigma) {
            param
                .set_low_pass_radius_sigma(
                    field!(
                        self.ltf_low_pass_radius_sigma
                            .get_text_boolean(do_validation)
                    )
                    .as_deref(),
                )
                .map_err(MTFFilterParametersException::FortranInputSyntax)?;
        } else {
            param.reset_low_pass_radius_sigma();
        }
        if Field::is_enabled(&*self.ltf_mtf_file) {
            param
                .set_mtf_file(field!(self.ltf_mtf_file.get_text_boolean(do_validation)).as_deref());
        } else {
            param.reset_mtf_file();
        }
        if Field::is_enabled(&*self.ltf_maximum_inverse) {
            param
                .set_maximum_inverse(
                    field!(self.ltf_maximum_inverse.get_text_boolean(do_validation)).as_deref(),
                )
                .map_err(MTFFilterParametersException::NumberFormat)?;
        } else {
            param.reset_maximum_inverse();
        }
        if Field::is_enabled(&*self.ltf_inverse_rolloff_radius_sigma) {
            param
                .set_inverse_rolloff_radius_sigma(
                    field!(
                        self.ltf_inverse_rolloff_radius_sigma
                            .get_text_boolean(do_validation)
                    )
                    .as_deref(),
                )
                .map_err(MTFFilterParametersException::FortranInputSyntax)?;
        } else {
            param.reset_inverse_rolloff_radius_sigma();
        }
        if self.tf_fixed_image_dose.is_enabled() {
            param.set_fixed_image_dose(
                field!(self.tf_fixed_image_dose.get_text_boolean(do_validation)).as_deref(),
            );
        } else {
            param.reset_fixed_image_dose();
        }
        let type_of_dose_file: Option<EnumeratedTypeRef>;
        if self.pnl_type_of_dose_file.is_enabled() {
            type_of_dose_file = self
                .bg_type_of_dose_file
                .get_selection()
                .and_then(|button| button.get_model())
                .and_then(|model| {
                    model
                        .as_any()
                        .downcast_ref::<RadioEButtonModel>()
                        .and_then(|model| AbstractRadioButtonModel::get_enumerated_type(model))
                });
            param.set_type_of_dose_file(type_of_dose_file.as_deref());
        } else {
            param.reset_type_of_dose_file();
        }
        if self.cb_fixed_image_dose.is_enabled()
            && !self.cb_fixed_image_dose.is_selected()
            && self.bctf_dose_weighting_file.is_enabled()
        {
            param.set_dose_weighting_file(
                field!(
                    self.bctf_dose_weighting_file
                        .get_text_boolean(do_validation)
                )
                .as_deref(),
            );
        } else {
            param.reset_dose_weighting_file();
        }
        if self.cb_voltage200.is_enabled() {
            param.set_voltage200(self.cb_voltage200.is_selected());
        } else {
            param.reset_voltage();
        }
        if self.tf_optimal_dose_scaling.is_enabled() {
            param.set_optimal_dose_scaling(
                field!(self.tf_optimal_dose_scaling.get_text_boolean(do_validation)).as_deref(),
            );
        } else {
            param.reset_optimal_dose_scaling();
        }
        if self.tf_bidirectional_num_views.is_enabled() {
            param.set_bidirectional_num_views(
                field!(
                    self.tf_bidirectional_num_views
                        .get_text_boolean(do_validation)
                )
                .as_deref(),
            );
        } else {
            param.reset_bidirectional_num_views();
        }
        param
            .set_starting_and_ending_z(
                field!(
                    self.ltf_starting_and_ending_z
                        .get_text_boolean(do_validation)
                )
                .as_deref(),
            )
            .map_err(MTFFilterParametersException::FortranInputSyntax)?;
        Ok(true)
    }

    /// Java `setParameters(ConstMTFFilterParam)`.
    pub fn set_parameters_const_mtf_filter_param(&self, param: &dyn ConstMTFFilterParam) {
        self.cb_dose_weight_filtering
            .set_selected(param.is_type_of_dose_file_set() || param.is_fixed_image_dose_set());
        if param.is_mtf_file_set() {
            self.ltf_mtf_file
                .set_text_string(param.get_mtf_file().as_deref());
        }
        if param.is_maximum_inverse_set() {
            self.ltf_maximum_inverse
                .set_text_string(Some(&param.get_maximum_inverse_string()));
        }
        if param.is_low_pass_radius_sigma_set() {
            self.ltf_low_pass_radius_sigma
                .set_text_string(Some(&param.get_low_pass_radius_sigma_string()));
        }
        if param.is_inverse_rolloff_radius_sigma_set() {
            self.ltf_inverse_rolloff_radius_sigma
                .set_text_string(Some(&param.get_inverse_rolloff_radius_sigma_string()));
        }
        self.cb_fixed_image_dose
            .set_selected(param.is_fixed_image_dose_set());
        if self.cb_fixed_image_dose.is_selected() {
            self.tf_fixed_image_dose
                .set_text_string(Some(&param.get_fixed_image_dose()));
        }
        if param.is_dose_weighting_file_set() {
            let dose_weighting_file = param.get_dose_weighting_file();
            // Treat an extension (like .mrc or .st) as a string.
            if !dose_weighting_file.is_empty() && !dose_weighting_file.starts_with('.') {
                self.bctf_dose_weighting_file
                    .set_file(Some(&dose_weighting_file));
            }
        }
        let type_of_dose_file = TypeOfDoseFile::get_instance(Some(&param.get_type_of_dose_file()));
        self.bg_type_of_dose_file
            .set_selected(Some(&EnumeratedTypeRef::new(type_of_dose_file)));
        self.cb_voltage200.set_selected(param.is_voltage200());
        if param.is_optimal_dose_scaling_set() {
            self.tf_optimal_dose_scaling
                .set_text_string(Some(&param.get_optimal_dose_scaling()));
        }
        if param.is_bidirectional_num_views_set() {
            self.tf_bidirectional_num_views
                .set_text_string(Some(&param.get_bidirectional_num_views()));
        }
        self.ltf_starting_and_ending_z
            .set_text_string(Some(&param.get_starting_and_ending_z_string()));
        self.update_display();
    }

    /// Java private `reregisterProcessingMethodMediator()`: see
    /// `getProcessingMethod()`.
    fn reregister_processing_method_mediator(&self) {
        let cur_tab = self.cur_tab.get();
        if cur_tab == Tab::CcdEraser {
            self.erase_gold_panel
                .reregister_processing_method_mediator();
        } else if cur_tab == Tab::CtfCorrection {
            self.cpu_gpu_panel.reregister_processing_method_mediator();
        } else if let (Some(mediator), Some(this)) = (&self.mediator, self.this.upgrade()) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.register_process_interface(origin.clone());
            mediator.set_method_process_interface_processing_method(
                &origin,
                ProcessInterface::get_processing_method(self),
            );
        }
    }

    /// Java private `changeTab()`.
    fn change_tab(&self) {
        let prev_tab = self.cur_tab.get();
        self.cur_tab.set(Tab::get_instance(
            self.tabbed_pane.get_component().get_selected_tab(),
        ));
        let cur_tab = self.cur_tab.get();
        if prev_tab == cur_tab {
            // Tab didn't change - nothing to do
            return;
        }
        if let Some(prev_panel) = self
            .tabbed_pane
            .get_component()
            .get_component_at(prev_tab.to_int() as usize)
        {
            prev_panel.remove_all();
        }
        if prev_tab != Tab::CcdEraser
            && cur_tab == Tab::CcdEraser
            && !self.erase_beads_initialized.get()
        {
            self.erase_gold_panel.initialize_beads();
            self.erase_beads_initialized.set(true);
        }
        self.reregister_processing_method_mediator();
        // Add panel for new tab
        let tabbed_pane = self.tabbed_pane.get_component();
        let panel = tabbed_pane.get_component_at(tabbed_pane.get_selected_tab() as usize);
        if let Some(panel) = panel {
            if cur_tab == Tab::Newst {
                panel.add(&self.newstack_or_blendmont_panel.get_component());
            } else if cur_tab == Tab::CtfCorrection {
                panel.add(&self.ctf_correction_main_panel.get_component());
            } else if cur_tab == Tab::CcdEraser {
                panel.add(&self.erase_gold_panel.get_component());
            } else if cur_tab == Tab::MtfFilter {
                panel.add(&self.filter_panel);
            }
        }
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
        // Warning caused by leaving the previous tab
        if cur_tab != prev_tab {
            let state = self.application_manager.get_state();
            if prev_tab == Tab::CtfCorrection && state.is_use_ctf_correction_warning(self.axis_id) {
                // The use button wasn't pressed and the user is moving on to the next
                // dialog. Don't put this message in the log.
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        None,
                        &format!(
                            "To use the CTF correction go back to the {} tab and press the \"{}\" button.",
                            CTF_TAB_LABEL, USE_CTF_CORRECTION_LABEL
                        ),
                        "Entry Warning",
                        Some(self.axis_id),
                    )
                });
                // Only warn once.
                state.set_use_ctf_correction_warning(self.axis_id, false);
            } else if prev_tab == Tab::CcdEraser && state.is_use_erased_stack_warning(self.axis_id)
            {
                // The use button wasn't pressed and the user is moving on to the next
                // dialog. Don't put this message in the log.
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        None,
                        &format!(
                            "To use the stack with the erased beads go back to the {} tab and press the \"{}\" button.",
                            erase_gold_panel::ERASE_GOLD_TAB_LABEL,
                            ccd_eraser_beads_panel::USE_ERASED_STACK_LABEL
                        ),
                        "Entry Warning",
                        Some(self.axis_id),
                    )
                });
                // Only warn once.
                state.set_use_erased_stack_warning(self.axis_id, false);
            } else if prev_tab == Tab::MtfFilter
                && state.is_use_filtered_stack_warning(self.axis_id)
            {
                // The use button wasn't pressed and the user is moving on to the next
                // dialog. Don't put this message in the log.
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        None,
                        &format!(
                            "To use the MTF filtered stack go back to the {} tab and press the \"{}\" button.",
                            MTF_FILTER_TAB_LABEL, USE_FILTERED_STACK_LABEL
                        ),
                        "Entry Warning",
                        Some(self.axis_id),
                    )
                });
                // Only warn once.
                state.set_use_filtered_stack_warning(self.axis_id, false);
            }
        }
        ui_harness::with(|harness| harness.move_sub_frame());
    }

    /// Java private `setToolTipText()`: initialize the tooltip text for the
    /// axis panel objects.
    fn set_tool_tip_text(&self) {
        let manager: &'static dyn BaseManager = self.application_manager;
        let mut autodoc: Option<*mut Autodoc> = None;
        // SAFETY (all `unsafe` below): `AutodocFactory` owns every autodoc it
        // returns, and each autodoc owns its sections, for the life of the
        // process (the Java GC-owned singletons), so the pointers stay valid
        // for this method.
        match unsafe {
            autodoc_factory::get_instance(
                Some(manager),
                Some(autodoc_factory::MTF_FILTER),
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
        if let Some(autodoc_ptr) = autodoc {
            let autodoc_ref: &dyn ReadOnlyAutodoc = unsafe { &*autodoc_ptr };
            self.ltf_starting_and_ending_z.set_tool_tip_text(
                etomo_autodoc::get_tooltip(Some(autodoc_ref), Some("StartingAndEndingZ"))
                    .as_deref(),
            );
            Field::set_tool_tip_text(
                &*self.ltf_low_pass_radius_sigma,
                etomo_autodoc::get_tooltip(Some(autodoc_ref), Some("LowPassRadiusSigma"))
                    .as_deref(),
            );
            let text = etomo_autodoc::get_tooltip(Some(autodoc_ref), Some("MtfFile"));
            if let Some(text) = &text {
                Field::set_tool_tip_text(&*self.ltf_mtf_file, Some(text));
                self.btn_mtf_file
                    .get_component()
                    .set_tool_tip_text(Some(text));
            }
            Field::set_tool_tip_text(
                &*self.ltf_maximum_inverse,
                etomo_autodoc::get_tooltip(Some(autodoc_ref), Some("MaximumInverse")).as_deref(),
            );
            Field::set_tool_tip_text(
                &*self.ltf_inverse_rolloff_radius_sigma,
                etomo_autodoc::get_tooltip(Some(autodoc_ref), Some("InverseRolloffRadiusSigma"))
                    .as_deref(),
            );
            self.bctf_dose_weighting_file.set_tooltip_string(
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(mtf_filter_param::DOSE_WEIGHTING_FILE_KEY),
                )
                .as_deref(),
            );
            self.pnl_type_of_dose_file.set_tool_tip_text(
                tooltip_formatter::INSTANCE
                    .format(
                        etomo_autodoc::get_tooltip(
                            Some(autodoc_ref),
                            Some(mtf_filter_param::TYPE_OF_DOSE_FILE_KEY),
                        )
                        .as_deref(),
                    )
                    .as_deref(),
            );
            let section: *mut Section = unsafe {
                (*autodoc_ptr).get_section(
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(mtf_filter_param::TYPE_OF_DOSE_FILE_KEY),
                )
            };
            if !section.is_null() {
                let section: &dyn ReadOnlySection = unsafe { &*section };
                self.rb_type_of_dose_file_image_dose
                    .set_tooltip_string_read_only_section(
                        Some(autodoc_factory::MTF_FILTER),
                        section,
                    );
                self.rb_type_of_dose_file_accumulated_and_image_dose
                    .set_tooltip_string_read_only_section(
                        Some(autodoc_factory::MTF_FILTER),
                        section,
                    );
                self.rb_type_of_dose_file_prior_and_cumulative_dose
                    .set_tooltip_string_read_only_section(
                        Some(autodoc_factory::MTF_FILTER),
                        section,
                    );
            } else {
                // Java `setTooltip(String, ReadOnlySection)` with a null section:
                // `EtomoAutodoc.getTooltip` answers null for a null section, so
                // each radio button's tooltip is set to null.
                self.rb_type_of_dose_file_image_dose
                    .set_tooltip_string(None);
                self.rb_type_of_dose_file_accumulated_and_image_dose
                    .set_tooltip_string(None);
                self.rb_type_of_dose_file_prior_and_cumulative_dose
                    .set_tooltip_string(None);
            }
            self.tf_optimal_dose_scaling.set_tooltip(
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(mtf_filter_param::OPTIMAL_DOSE_SCALING_KEY),
                )
                .as_deref(),
            );
            self.tf_bidirectional_num_views.set_tooltip(
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(mtf_filter_param::BIDIRECTIONAL_NUM_VIEWS_KEY),
                )
                .as_deref(),
            );
        }
        self.rb_type_of_dose_file_mdoc_file.set_tooltip_string(Some(
            "Use dose information from a SerialEM .mdoc file named setname.mrc.mdoc and in the \
             current directory",
        ));
        self.tf_fixed_image_dose.set_tooltip(Some(
            "Use the same dose for each image instead of getting dose information from a file.",
        ));
        self.cb_dose_weight_filtering.set_tooltip(Some(
            "Enables 'dose weighting', which filters out high frequencies as a function of the \
             dose already applied to a cryo-specimen.",
        ));
        self.cb_voltage200.set_tooltip(Some(
            "Set microscope voltage in kV to 200; the default is 300 KV.",
        ));
        self.btn_filter
            .set_tool_tip_text(Some("Run mtffilter on the full aligned stack."));
        self.btn_view_filter.set_tool_tip_text(Some(&format!(
            "{}{}",
            "View the results of running mtffilter on the full ", "aligned stack."
        )));
        self.btn_use_filter.set_tool_tip_text(Some(&format!(
            "{}{}",
            "Use the results of running mtffilter as the new full ", "aligned stack."
        )));
        match unsafe {
            autodoc_factory::get_instance(
                Some(manager),
                Some(autodoc_factory::CTF_PLOTTER),
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
        // `autodoc` is only reassigned on success, so a failed lookup keeps the
        // previous autodoc, as in the source.
        if let Some(autodoc_ptr) = autodoc {
            let autodoc_ref: &dyn ReadOnlyAutodoc = unsafe { &*autodoc_ptr };
            Field::set_tool_tip_text(
                &*self.ftf_config_file,
                etomo_autodoc::get_tooltip(Some(autodoc_ref), Some("ConfigFile")).as_deref(),
            );
            Field::set_tool_tip_text(
                &*self.ltf_voltage,
                Some(&format!(
                    "{}  Also used in {}.",
                    etomo_autodoc::get_tooltip(
                        Some(autodoc_ref),
                        Some(ctf_phase_flip_param::VOLTAGE_OPTION)
                    )
                    .as_deref()
                    .unwrap_or("null"),
                    ctf_phase_flip_param::COMMAND
                )),
            );
            Field::set_tool_tip_text(
                &*self.ltf_spherical_aberration,
                Some(&format!(
                    "{}  Also used in {}.",
                    etomo_autodoc::get_tooltip(
                        Some(autodoc_ref),
                        Some(ctf_phase_flip_param::SPHERICAL_ABERRATION_OPTION)
                    )
                    .as_deref()
                    .unwrap_or("null"),
                    ctf_phase_flip_param::COMMAND
                )),
            );
            self.cb_invert_tilt_angles
                .set_tool_tip_text_string(Some(&format!(
                    "{}  Also used in {}.",
                    etomo_autodoc::get_tooltip(
                        Some(autodoc_ref),
                        Some(ctf_phase_flip_param::INVERT_TILT_ANGLES_OPTION)
                    )
                    .as_deref()
                    .unwrap_or("null"),
                    ctf_phase_flip_param::COMMAND
                )));
            Field::set_tool_tip_text(
                &*self.ltf_amplitude_contrast,
                Some(&format!(
                    "{}  Also used in {}.",
                    etomo_autodoc::get_tooltip(
                        Some(autodoc_ref),
                        Some(ctf_phase_flip_param::AMPLITUDE_CONTRAST_OPTION)
                    )
                    .as_deref()
                    .unwrap_or("null"),
                    ctf_phase_flip_param::COMMAND
                )),
            );
            Field::set_tool_tip_text(
                &*self.ltf_scan_defocus_range,
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(ctf_plotter_param::SCAN_DEFOCUS_RANGE_OPTION),
                )
                .as_deref(),
            );
            Field::set_tool_tip_text(
                &*self.ltf_expected_defocus,
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(ctf_plotter_param::EXPECTED_DEFOCUS_OPTION),
                )
                .as_deref(),
            );
            self.tf_phase_shift_in_degrees.set_tooltip(
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(ctf_plotter_param::PHASE_SHIFT_IN_DEGREES_OPTION),
                )
                .as_deref(),
            );
            Field::set_tool_tip_text(
                &*self.ltf_offset_to_add,
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(ctf_plotter_param::OFFSET_TO_ADD_OPTION),
                )
                .as_deref(),
            );
        }
        self.btn_ctf_plotter
            .set_tool_tip_text(Some("Run ctfplotter"));
        match unsafe {
            autodoc_factory::get_instance(
                Some(manager),
                Some(autodoc_factory::CTF_PHASE_FLIP),
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
        if let Some(autodoc_ptr) = autodoc {
            let autodoc_ref: &dyn ReadOnlyAutodoc = unsafe { &*autodoc_ptr };
            Field::set_tool_tip_text(
                &*self.ltf_interpolation_width,
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(ctf_phase_flip_param::INTERPOLATION_WIDTH_OPTION),
                )
                .as_deref(),
            );
            self.cb_ctf_phase_flip_x_axis_tilt.set_tooltip(Some(
                "Correct for X-axis tilt, which may take roughly 10 times longer.  A few degrees \
                 of X axis tilt is insignificant except for the highest resolution work.",
            ));
            self.tf_ctf_phase_flip_x_axis_tilt.set_tooltip(
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(ctf_phase_flip_param::X_AXIS_TILT_OPTION),
                )
                .as_deref(),
            );
            let tooltip = etomo_autodoc::get_tooltip(
                Some(autodoc_ref),
                Some(ctf_phase_flip_param::SCALE_BY_CTF_POWER_OPTION),
            );
            self.cb_scale_by_ctf_power.set_tooltip(tooltip.as_deref());
            self.tf_scale_by_ctf_power.set_tooltip(tooltip.as_deref());
            self.tf_minimum_zero_spacing.set_tooltip(
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(ctf_phase_flip_param::MINIMUM_ZERO_SPACING_OPTION),
                )
                .as_deref(),
            );
            Field::set_tool_tip_text(
                &*self.ltf_defocus_tol,
                etomo_autodoc::get_tooltip(
                    Some(autodoc_ref),
                    Some(ctf_phase_flip_param::DEFOCUS_TOL_OPTION),
                )
                .as_deref(),
            );
        }
        self.cb_use_expected_defocus
            .set_tool_tip_text_string(Some(&format!(
                "Instead of using the CTF plotter output ({}) use a one line file containing the \
                 expected defocus ({}).  Etomo will create the {} file when this checkbox is \
                 checked.",
                dataset_files::CTF_PLOTTER_EXT,
                dataset_files::SIMPLE_DEFOCUS_EXT,
                dataset_files::SIMPLE_DEFOCUS_EXT
            )));
        self.btn_ctf_correction.set_tool_tip_text(Some(&format!(
            "Run {}{}, which calls {}.",
            ProcessName::CTF_CORRECTION,
            dataset_files::COMSCRIPT_EXT,
            ctf_phase_flip_param::COMMAND
        )));
        let ctf_corrected_file_name = file_type::CLASS
            .ctf_corrected_stack
            .get_file_name(Some(manager), Some(self.axis_id))
            .unwrap_or_else(|| "null".to_string());
        self.btn_imod_ctf_correction
            .set_tool_tip_text(Some(&format!(
                "Open CTF corrected stack ({ctf_corrected_file_name})."
            )));
        self.btn_use_ctf_correction.set_tool_tip_text(Some(&format!(
            "Replace full aligned stack ({}) with CTF corrected stack ({}).",
            file_type::CLASS
                .aligned_stack
                .get_file_name(Some(manager), Some(self.axis_id))
                .unwrap_or_else(|| "null".to_string()),
            ctf_corrected_file_name
        )));
    }
}

impl ProcessDialogVirtual for FinalAlignedStackDialog {
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java override `done()`.
    fn done(&self) {
        if let Some(expert) = self.expert.upgrade() {
            expert.done_dialog_void();
        }
        self.newstack_or_blendmont_panel.done();
        self.erase_gold_panel.done();
        self.btn_use_filter
            .remove_action_listener(&self.final_aligned_stack_listener);
        self.btn_filter
            .remove_action_listener(&self.final_aligned_stack_listener);
        self.btn_ctf_correction
            .remove_action_listener(&self.final_aligned_stack_listener);
        self.btn_use_ctf_correction
            .remove_action_listener(&self.final_aligned_stack_listener);
        self.base.set_displayed(false);
        if let (Some(mediator), Some(this)) = (&self.mediator, self.this.upgrade()) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.deregister_process_interface(&origin);
        }
    }
}

impl Expandable for FinalAlignedStackDialog {
    /// Java override `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java override `expand(ExpandButton)`.  Expands the appropriate panel.
    /// Also called when advanced/basic buttons are pressed in child classes.
    /// Responsible for changing the state of the big advanced/basic button at
    /// the bottom of the dialog.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.filter_header.equals_advanced_basic(button) {
            self.update_advanced_filter(button.is_expanded());
        } else if self.ctf_correction_header.equals_advanced_basic(button) {
            self.update_advanced_ctf_correction(button.is_expanded());
        }
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }
}

impl ContextMenu for FinalAlignedStackDialog {
    /// Java override `popUpContextMenu(MouseEvent)`: right mouse button
    /// context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let align_manpage_label: &str;
        let align_manpage: &str;
        let align_logfile_label: &str;
        let align_logfile: &str;
        let manager: &'static dyn BaseManager = self.application_manager;
        let axis_id = self.axis_id;
        let cur_tab = self.cur_tab.get();
        let to_strings =
            |array: &[&str]| -> Vec<String> { array.iter().map(|s| s.to_string()).collect() };
        if cur_tab == Tab::CtfCorrection {
            let man_pagelabel = to_strings(&["Ctfplotter", "Ctfphaseflip", "3dmod"]);
            let man_page = to_strings(&["ctfplotter.html", "ctfphaseflip.html", "3dmod.html"]);
            let log_file_label = to_strings(&["Ctfplotter", "Ctfcorrection"]);
            let mut log_file: Vec<String> = vec![String::new(); 2];
            log_file[0] = format!("ctfplotter{}.log", axis_id.get_extension());
            log_file[1] = format!("ctfcorrection{}.log", axis_id.get_extension());
            let _context_popup =
                ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
                    &self.base.root_panel.get_component(),
                    mouse_event,
                    Some("CorrectingCTF"),
                    Some(context_popup::TOMO_GUIDE),
                    &man_pagelabel,
                    &man_page,
                    Some(&log_file_label),
                    Some(&log_file),
                    manager,
                    axis_id,
                );
        } else if cur_tab == Tab::MtfFilter {
            let man_pagelabel = to_strings(&["Mtffilter", "3dmod"]);
            let man_page = to_strings(&["mtffilter.html", "3dmod.html"]);
            let log_file_label = to_strings(&["Mtffilter"]);
            let mut log_file: Vec<String> = vec![String::new(); 1];
            log_file[0] = format!("mtffilter{}.log", axis_id.get_extension());
            let _context_popup =
                ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
                    &self.base.root_panel.get_component(),
                    mouse_event,
                    Some("Filtering2D"),
                    Some(context_popup::TOMO_GUIDE),
                    &man_pagelabel,
                    &man_page,
                    Some(&log_file_label),
                    Some(&log_file),
                    manager,
                    axis_id,
                );
        } else {
            if self.application_manager.get_meta_data().get_view_type() == ViewType::Montage {
                align_manpage_label = "Blendmont";
                align_manpage = "blendmont";
                align_logfile_label = "Blend";
                align_logfile = "blend";
            } else {
                align_manpage_label = "Newstack";
                align_manpage = "newstack";
                align_logfile_label = "Newst";
                align_logfile = "newst";
            }
            let man_pagelabel = to_strings(&[align_manpage_label, "3dmod"]);
            let man_page = vec![format!("{align_manpage}.html"), "3dmod.html".to_string()];
            let log_file_label = to_strings(&[align_logfile_label]);
            let mut log_file: Vec<String> = vec![String::new(); 1];
            log_file[0] = format!("{}{}.log", align_logfile, axis_id.get_extension());
            let _context_popup =
                ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
                    &self.base.root_panel.get_component(),
                    mouse_event,
                    Some("FinalAligned"),
                    Some(context_popup::TOMO_GUIDE),
                    &man_pagelabel,
                    &man_page,
                    Some(&log_file_label),
                    Some(&log_file),
                    manager,
                    axis_id,
                );
        }
    }
}

impl Run3dmodButtonContainer for FinalAlignedStackDialog {
    /// Java override `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    /// Executes the action associated with command.  Deferred3dmodButton is
    /// null if it comes from the dialog's ActionListener.  Otherwise is comes
    /// from a Run3dmodButton which called action(Run3dmodButton,
    /// Run3dmoMenuOptions).  In that case it will be null unless it was set in
    /// the Run3dmodButton.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let expert = self.expert.upgrade();
        if Some(command) == self.btn_filter.get_action_command().as_deref() {
            if let Some(expert) = &expert {
                let display: ProcessResultDisplayHandle = self.btn_filter.clone();
                expert.mtffilter(
                    Some(display),
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                );
            }
        } else if Some(command) == self.btn_use_filter.get_action_command().as_deref() {
            if let Some(expert) = &expert {
                let display: ProcessResultDisplayHandle = self.btn_use_filter.clone();
                expert.use_mtf_filter(Some(display));
            }
        } else if Some(command) == self.btn_view_filter.get_action_command().as_deref() {
            self.application_manager
                .imod_mtf_filter(self.axis_id, run_3dmod_menu_options.unwrap_or_default());
        } else if Some(command) == self.cb_use_expected_defocus.get_action_command().as_deref() {
            self.update_ctf_plotter();
        } else if Some(command) == self.btn_ctf_plotter.get_action_command().as_deref() {
            if let Some(expert) = &expert {
                let display: ProcessResultDisplayHandle = self.btn_ctf_plotter.clone();
                expert.ctf_plotter(Some(display));
            }
        } else if Some(command) == self.btn_ctf_correction.get_action_command().as_deref() {
            if let Some(expert) = &expert {
                let display: ProcessResultDisplayHandle = self.btn_ctf_correction.clone();
                expert
                    .ctf_correction_process_result_display_process_series_deferred_3dmod_button_run_3dmod_menu_options_processing_method(
                        Some(display),
                        None,
                        deferred_3dmod_button,
                        run_3dmod_menu_options,
                        self.cpu_gpu_panel.get_run_method_for_process_interface(),
                    );
            }
        } else if Some(command) == self.btn_imod_ctf_correction.get_action_command().as_deref() {
            self.application_manager
                .imod_ctf_correction(self.axis_id, run_3dmod_menu_options.unwrap_or_default());
        } else if Some(command) == self.btn_use_ctf_correction.get_action_command().as_deref() {
            if let Some(expert) = &expert {
                let display: ProcessResultDisplayHandle = self.btn_use_ctf_correction.clone();
                expert.use_ctf_correction(Some(display));
            }
        } else {
            self.update_display();
        }
    }
}

impl QueueTableListener for FinalAlignedStackDialog {
    /// Java inherited `ProcessDialog.queueTableEventAction(QueueTableEvent)`:
    /// empty.
    fn queue_table_event_action(&self, event: &QueueTableEvent) {
        self.base.queue_table_event_action(event);
    }
}

impl ProcessInterface for FinalAlignedStackDialog {
    /// Java override `updateGpu(boolean)`: empty.
    fn update_gpu(&self, _disable_gpu: bool) {}

    /// Java override `getProcessingMethod()`: return local processing method.
    /// Ctf correction and bead erasing are handled by different instances of
    /// the CpuGpuPanel.  See `reregisterProcessingMethodMediator`.
    fn get_processing_method(&self) -> ProcessingMethod {
        ProcessingMethod::LocalCpu
    }

    /// Java override `getSecondaryProcessingMethod()`.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }

    /// Java override `lockProcessingMethod(boolean)`.
    fn lock_processing_method(&self, lock: bool) {
        self.processing_method_locked.set(lock);
        self.cpu_gpu_panel.lock_processing_method(lock);
    }

    /// Java override `setMethod(ProcessingMethod)`.
    fn set_method(&self, processing_method: ProcessingMethod) {
        if let (Some(mediator), Some(this)) = (&self.mediator, self.this.upgrade()) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.set_method_process_interface_processing_method(&origin, processing_method);
        }
    }

    /// Java override `isUseGpu()`.
    fn is_use_gpu(&self) -> bool {
        false
    }

    /// Java override `setUseQueueCheckBox(ButtonComponent)`: empty.
    fn set_use_queue_check_box(&self, _use_queue_checkbox: Option<Rc<dyn ButtonComponent>>) {}

    /// Java inherited `ProcessDialog.addQueueTableListener(QueueTableListener)`.
    fn add_queue_table_listener(&self, listener: Rc<dyn QueueTableListener>) {
        self.base.add_queue_table_listener(listener);
    }

    /// Java inherited `ProcessDialog.removeQueueTableListener(QueueTableListener)`.
    fn remove_queue_table_listener(&self, listener: &Rc<dyn QueueTableListener>) {
        self.base.remove_queue_table_listener(listener);
    }
}

/// Java `static final class Tab` (FinalAlignedStackDialog.java:1590).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    /// Java `NEWST = new Tab(0, "NEWST")`.
    Newst,
    /// Java `CTF_CORRECTION = new Tab(1, "CTF_CORRECTION")`.
    CtfCorrection,
    /// Java `CCD_ERASER = new Tab(2, "CCD_ERASER")`.
    CcdEraser,
    /// Java `MTF_FILTER = new Tab(3, "MTF_FILTER")`.
    MtfFilter,
}

impl Tab {
    /// Java static final `DEFAULT = NEWST`.
    pub const DEFAULT: Tab = Tab::Newst;

    /// Java private field `index`.
    fn index(self) -> i32 {
        match self {
            Tab::Newst => 0,
            Tab::CtfCorrection => 1,
            Tab::CcdEraser => 2,
            Tab::MtfFilter => 3,
        }
    }

    /// Java private static `getInstance(int)`.
    fn get_instance(index: i32) -> Tab {
        if index == Tab::Newst.index() {
            return Tab::Newst;
        }
        if index == Tab::CtfCorrection.index() {
            return Tab::CtfCorrection;
        }
        if index == Tab::CcdEraser.index() {
            return Tab::CcdEraser;
        }
        if index == Tab::MtfFilter.index() {
            return Tab::MtfFilter;
        }
        Tab::DEFAULT
    }

    /// Java private `isDefault()`.
    fn is_default(self) -> bool {
        self == Tab::DEFAULT
    }

    /// Java private `toInt()`.
    fn to_int(self) -> i32 {
        self.index()
    }
}

impl std::fmt::Display for Tab {
    /// Java `toString()`: the field `string`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Tab::Newst => "NEWST",
            Tab::CtfCorrection => "CTF_CORRECTION",
            Tab::CcdEraser => "CCD_ERASER",
            Tab::MtfFilter => "MTF_FILTER",
        })
    }
}

/// Java `private static final class TypeOfDoseFile implements EnumeratedType`
/// (FinalAlignedStackDialog.java:1646).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypeOfDoseFile {
    /// Java `IMAGE_DOSE = new TypeOfDoseFile(1, "Dose for each image", true)`.
    ImageDose,
    /// Java `ACCUMULATED_AND_IMAGE_DOSE = new TypeOfDoseFile(2, "Prior dose and
    /// dose of image", false)`.
    AccumulatedAndImageDose,
    /// Java `PRIOR_AND_CUMULATIVE_DOSE = new TypeOfDoseFile(3, "Cumulative dose
    /// before and after image", false)`.
    PriorAndCumulativeDose,
    /// Java `MDOC_FILE = new TypeOfDoseFile(4, "Metadata in .mdoc file", true)`.
    MdocFile,
}

impl TypeOfDoseFile {
    /// Java private static final `DEFAULT = MDOC_FILE`.
    const DEFAULT: TypeOfDoseFile = TypeOfDoseFile::MdocFile;

    /// Java private final field `value` (`new EtomoNumber()` then
    /// `value.set(int)` in the constructor).
    fn value(self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(match self {
            TypeOfDoseFile::ImageDose => 1,
            TypeOfDoseFile::AccumulatedAndImageDose => 2,
            TypeOfDoseFile::PriorAndCumulativeDose => 3,
            TypeOfDoseFile::MdocFile => 4,
        });
        value
    }

    /// Java private final field `label`.
    fn label(self) -> &'static str {
        match self {
            TypeOfDoseFile::ImageDose => "Dose for each image",
            TypeOfDoseFile::AccumulatedAndImageDose => "Prior dose and dose of image",
            TypeOfDoseFile::PriorAndCumulativeDose => "Cumulative dose before and after image",
            TypeOfDoseFile::MdocFile => "Metadata in .mdoc file",
        }
    }

    /// Java private final field `enableBidirectionalNumViews`.
    fn enable_bidirectional_num_views(self) -> bool {
        match self {
            TypeOfDoseFile::ImageDose => true,
            TypeOfDoseFile::AccumulatedAndImageDose => false,
            TypeOfDoseFile::PriorAndCumulativeDose => false,
            TypeOfDoseFile::MdocFile => true,
        }
    }

    /// Java private static `getInstance(String)`.
    fn get_instance(value: Option<&str>) -> TypeOfDoseFile {
        let Some(value) = value else {
            return TypeOfDoseFile::DEFAULT;
        };
        if TypeOfDoseFile::ImageDose.value().equals_string(Some(value)) {
            return TypeOfDoseFile::ImageDose;
        }
        if TypeOfDoseFile::AccumulatedAndImageDose
            .value()
            .equals_string(Some(value))
        {
            return TypeOfDoseFile::AccumulatedAndImageDose;
        }
        if TypeOfDoseFile::PriorAndCumulativeDose
            .value()
            .equals_string(Some(value))
        {
            return TypeOfDoseFile::PriorAndCumulativeDose;
        }
        if TypeOfDoseFile::MdocFile.value().equals_string(Some(value)) {
            return TypeOfDoseFile::MdocFile;
        }
        TypeOfDoseFile::DEFAULT
    }
}

impl EnumeratedType for TypeOfDoseFile {
    /// Java override `isDefault()`.
    fn is_default(&self) -> bool {
        *self == TypeOfDoseFile::DEFAULT
    }

    /// Java override `getValue()`.
    fn get_value(&self) -> ConstEtomoNumber {
        self.value().base
    }

    /// Java override `getLabel()`.
    fn get_label(&self) -> Option<String> {
        Some(self.label().to_string())
    }
}

impl std::fmt::Display for TypeOfDoseFile {
    /// Java override `toString()`: `value.toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tab_and_type_of_dose_file_lookups() {
        assert_eq!(Tab::get_instance(2), Tab::CcdEraser);
        assert_eq!(Tab::get_instance(7), Tab::DEFAULT);
        assert!(Tab::Newst.is_default());
        assert_eq!(Tab::MtfFilter.to_string(), "MTF_FILTER");
        assert_eq!(TypeOfDoseFile::get_instance(None), TypeOfDoseFile::MdocFile);
        assert_eq!(
            TypeOfDoseFile::get_instance(Some("3")),
            TypeOfDoseFile::PriorAndCumulativeDose
        );
        assert_eq!(TypeOfDoseFile::ImageDose.to_string(), "1");
        assert!(TypeOfDoseFile::MdocFile.is_default());
        assert!(!TypeOfDoseFile::AccumulatedAndImageDose.enable_bidirectional_num_views());
    }
}
