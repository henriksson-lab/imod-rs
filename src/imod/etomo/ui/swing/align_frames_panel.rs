//! `IMOD/Etomo/src/etomo/ui/swing/AlignFramesPanel.java`.
//!
//! Java `public final class AlignFramesPanel implements ToolPanel, Expandable,
//! ContextMenu, ActionListener, ControlTarget, ControlListener,
//! AlignFramesDisplay, BrowsingDirectory, FocusListener, ChangeListener,
//! Run3dmodButtonContainer, FieldDisplayer`: the Align Frames tool of the Tools
//! dialog (alignframes / sorttiltframes).
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by
//! [`AlignFramesPanel::get_tools_instance`]; every method takes `&self`.  The
//! panel is its own `ActionListener` and `ChangeListener`; those listeners are
//! closures holding a weak reference.  It hands itself to its file fields as
//! their `BrowsingDirectory`, `ControlListener` and `FieldDisplayer`, which the
//! Rust widgets hold as strong `Rc`s: a reference cycle, like the Java object
//! graph, released only with the process (the Tools dialog lives as long as its
//! manager).  Because those widgets need a strong handle, the constructor's
//! `setAltBrowsingDirectory(this)` calls and `setFieldDisplayer()` run right
//! after the panel's `Rc` exists (still inside construction, before
//! `createPanel`), in the Java order.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::align_frames_display::AlignFramesDisplay;
use super::beveled_border::BeveledBorder;
use super::button_control_text_efield::ButtonControlTextEfield;
use super::check_box::CheckBox;
use super::combo_box::ComboBox;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::control_listener::ControlListener;
use super::control_state::ControlState;
use super::control_target::ControlTarget;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::ebutton::Ebutton;
use super::eer_super_res_z_sum_padding_panel::EERSuperResZSumPaddingPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::file_chooser::{self, FileChooser};
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel::Panel;
use super::panel_header::PanelHeader;
use super::process_display::ProcessDisplay;
use super::radio_button::RadioButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::select_file_extension::SelectFileExtension;
use super::spaced_panel::{self, SpacedPanel};
use super::spinner::Spinner;
use super::tabbed_pane::TabbedPane;
use super::text_area::TextArea;
use super::text_field::TextField;
use super::tool_panel::ToolPanel;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::align_frames_param::{self, AlignFramesParam};
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::sort_tilt_frames_param::{self, SortTiltFramesParam};
use crate::imod::etomo::comscript::tomodataplots_param::Task;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, ChangeEvent, ChangeListener, Dimension, FileFilter,
    FocusEvent, FocusListener, JComponent, MouseEvent, MouseListener,
};
use crate::imod::etomo::local_arguments::LocalArguments;
use crate::imod::etomo::logic::validation_set::ValidationSet;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::com_file_file_filter::ComFileFileFilter;
use crate::imod::etomo::storage::frame_file_filter::FrameFileFilter;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::mdoc_file_filter::MdocFileFilter;
use crate::imod::etomo::storage::text_file_file_filter::TextFileFileFilter;
use crate::imod::etomo::storage::tilt_extension_file_filter::TiltExtensionFileFilter;
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_double_to_string, java_lang_double_value_of,
};
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::extension::{self, EXTENSION_DIVIDER};
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java private static final `DIRECTORY_LABEL`.
const DIRECTORY_LABEL: &str = "Directory";
/// Java private static final `INPUT_FILES_LABEL`.
const INPUT_FILES_LABEL: &str = "Input File(s):";
/// Java private static final `ROOTNAME_OUTPUT_FILES_LABEL`.
const ROOTNAME_OUTPUT_FILES_LABEL: &str = " Root name for output files";
/// Java private static final `PATH_TO_FRAMES_IN_MDOC_LABEL`.
const PATH_TO_FRAMES_IN_MDOC_LABEL: &str = "Other directory with frames: ";
/// Java private static final `CORRESPONDING_STACK_LABEL`.
const CORRESPONDING_STACK_LABEL: &str = "Matching tilt series file: ";
/// Java private static final `TILT_ANGLE_FILE_LABEL`.
const TILT_ANGLE_FILE_LABEL: &str = "Text file with tilt angles:";
/// Java private static final `ANGLES_IN_FILENAMES_LABEL`.
const ANGLES_IN_FILENAMES_LABEL: &str = "Tilt angles in filenames";
/// Java private static final `DELIMITER_OPEN_LABEL`.
const DELIMITER_OPEN_LABEL: &str = "Delimiting character at start ";
/// Java private static final `DELIMITER_CLOSE_LABEL`.
const DELIMITER_CLOSE_LABEL: &str = "and end ";
/// Java private static final `AXIS_ROTATION_ANGLE_LABEL`.
const AXIS_ROTATION_ANGLE_LABEL: &str = "Tilt axis rotation";
/// Java private static final `REF_AND_DEFECT_FROM_TITLES_LABEL`.
const REF_AND_DEFECT_FROM_TITLES_LABEL: &str =
    "Gain normalize from reference and defect files in frame file header";
/// Java private static final `ROTATION_AND_FLIP_LABEL`.
const ROTATION_AND_FLIP_LABEL: &str = "Reference rotation and flip";
/// Java private static final `INPUT_COUNTS_LABEL`.
const INPUT_COUNTS_LABEL: &str = "input counts";
/// Java private static final `SDS_FROM_MEAN_LABEL`.
const SDS_FROM_MEAN_LABEL: &str = "SDs from mean";
/// Java private static final `TEST_BINNINGS_LABEL`.
const TEST_BINNINGS_LABEL: &str = "Test multiple binnings:";
/// Java private static final `SHIFT_LIMIT_LABEL`.
const SHIFT_LIMIT_LABEL: &str = "Maximum shift between frames: ";
/// Java private static final `STARTING_ENDING_FRAMES_FIRST_LABEL`.
const STARTING_ENDING_FRAMES_FIRST_LABEL: &str = "Align subsets of frames from";
/// Java private static final `STARTING_ENDING_FRAMES_SECOND_LABEL`.
const STARTING_ENDING_FRAMES_SECOND_LABEL: &str = " to";
/// Java private static final `FIXED_TOTAL_DOSE_LABEL`.
const FIXED_TOTAL_DOSE_LABEL: &str = "Dose weighting with fixed dose/image of ";
/// Java private static final `OPTIMAL_DOSE_SCALING_LABEL`.
const OPTIMAL_DOSE_SCALING_LABEL: &str = " Optimal dose scaling: ";
/// Java private static final `SCALING_OF_SUM_LABEL`.
const SCALING_OF_SUM_LABEL: &str = "Scale output by";
/// Java private static final `SCALING_OF_SUM_FACTOR_LABEL`.
const SCALING_OF_SUM_FACTOR_LABEL: &str = "Factor:";
/// Java private static final `SUM_ROTATION_AND_FLIP_LABEL`.
const SUM_ROTATION_AND_FLIP_LABEL: &str = "Rotation/flip for output ";
/// Java private static final `ALIGN_AND_SUM_BINNING_VALUE2_LABEL`.
const ALIGN_AND_SUM_BINNING_VALUE2_LABEL: &str = "Reduce output size by ";
/// Java private static final `OUTPUT_IMAGE_FILE_LABEL`.
const OUTPUT_IMAGE_FILE_LABEL: &str = "Output file name";
/// Java private static final `ABOVE_LABEL`.
const ABOVE_LABEL: &str = "Above";

/// Java `public final class AlignFramesPanel`.
pub struct AlignFramesPanel {
    /// Rust-only: Java `this`.
    self_ref: Weak<AlignFramesPanel>,

    /// Java private final `pnlTabArray = new JPanel[Tab.NUM_TABS]`; the
    /// elements are created by `createPanel`.
    pnl_tab_array: RefCell<Vec<Rc<JComponent>>>,
    /// Java private final `pnlTabBodyArray = new JPanel[Tab.NUM_TABS]`; the
    /// elements are created by `createPanel`.
    pnl_tab_body_array: RefCell<Vec<Rc<JComponent>>>,
    /// Java private `curTab = null`.
    cur_tab: Cell<Option<&'static Tab>>,
    /// Java private final `tabPane = new TabbedPane()`.
    tab_pane: Rc<TabbedPane>,

    /// Java private final `pnlOuterPanel = new Panel()`.
    pnl_outer_panel: Rc<Panel>,
    /// Java private final `pnlBtnAdvancedBasic = new JPanel()` (never used).
    #[allow(dead_code)]
    pnl_btn_advanced_basic: Rc<JComponent>,
    /// Java private final `pnlBtnLoadComFile`.
    pnl_btn_load_com_file: Rc<JComponent>,
    /// Java private final `pnlRootnameOutputFiles`.
    pnl_rootname_output_files: Rc<JComponent>,
    /// Java private final `pnlOuterPathToFramesInMdoc`.
    pnl_outer_path_to_frames_in_mdoc: Rc<JComponent>,
    /// Java private final `pnlPathToFramesInMdoc = new Panel()`.
    pnl_path_to_frames_in_mdoc: Rc<Panel>,
    /// Java private final `pnlOuterGainReference`.
    pnl_outer_gain_reference: Rc<JComponent>,
    /// Java private final `pnlGainReference` (shadowed by a local in
    /// `getPnlGainReference`; never used).
    #[allow(dead_code)]
    pnl_gain_reference: Rc<JComponent>,
    /// Java private final `pnlGainReferenceFile = new Panel()`.
    pnl_gain_reference_file: Rc<Panel>,
    /// Java private final `pnlRotationAndFlip = new Panel()`.
    pnl_rotation_and_flip: Rc<Panel>,
    /// Java private final `pnlOuterCameraDefectFile`.
    pnl_outer_camera_defect_file: Rc<JComponent>,
    /// Java private final `pnlCameraDefectFile = new Panel()`.
    pnl_camera_defect_file: Rc<Panel>,
    /// Java private final `pnlTruncateValues`.
    pnl_truncate_values: Rc<JComponent>,
    /// Java private final `pnlFitPairwiseShifts`.
    pnl_fit_pairwise_shifts: Rc<JComponent>,
    /// Java private final `pnlBinningForAlignment`.
    pnl_binning_for_alignment: Rc<JComponent>,
    /// Java private final `pnlTestBinnings`.
    pnl_test_binnings: Rc<JComponent>,
    /// Java private final `pnlVaryFilter`.
    pnl_vary_filter: Rc<JComponent>,
    /// Java private final `pnlFilterCutoffs`.
    pnl_filter_cutoffs: Rc<JComponent>,
    /// Java private final `pnlShiftLimit`.
    pnl_shift_limit: Rc<JComponent>,
    /// Java private final `pnlGroupFrames`.
    pnl_group_frames: Rc<JComponent>,
    /// Java private final `pnlRefinement`.
    pnl_refinement: Rc<JComponent>,
    /// Java private final `pnlRefineWithGroupSums`.
    pnl_refine_with_group_sums: Rc<JComponent>,
    /// Java private final `pnlStopIterationsAtShift`.
    pnl_stop_iterations_at_shift: Rc<JComponent>,
    /// Java private final `pnlMinForSplineSmoothing`.
    pnl_min_for_spline_smoothing: Rc<JComponent>,
    /// Java private final `pnlStartingEndingFrames`.
    pnl_starting_ending_frames: Rc<JComponent>,
    /// Java private final `pnlDoDoseWeighting`.
    pnl_do_dose_weighting: Rc<JComponent>,
    /// Java private final `pnlDoseWeighting`.
    pnl_dose_weighting: Rc<JComponent>,
    /// Java private final `pnlDoseWeightingBody`.
    pnl_dose_weighting_body: Rc<JComponent>,
    /// Java private final `pnlFixedTotalDose`.
    pnl_fixed_total_dose: Rc<JComponent>,
    /// Java private final `pnlScalingOfSum`.
    pnl_scaling_of_sum: Rc<JComponent>,
    /// Java private final `pnlModeToOutout` (the source's spelling).
    pnl_mode_to_outout: Rc<JComponent>,
    /// Java private final `pnlSumRotationAndFlip`.
    pnl_sum_rotation_and_flip: Rc<JComponent>,
    /// Java private final `pnlAlignAndSumBinning`.
    pnl_align_and_sum_binning: Rc<JComponent>,
    /// Java private final `pnlOutputImageFile`.
    pnl_output_image_file: Rc<JComponent>,
    /// Java private final `pnlUseGPU`.
    pnl_use_gpu: Rc<JComponent>,
    /// Java private final `pnlRunAlignFramesButtons`.
    pnl_run_align_frames_buttons: Rc<JComponent>,
    /// Java private final `pnlStartReconButton`.
    pnl_start_recon_button: Rc<JComponent>,
    /// Java private final `pnlInputFileSpecRight`.
    pnl_input_file_spec_right: Rc<JComponent>,
    /// Java private final `pnlOtherSourceOfMetadataBody = new Panel()`.
    pnl_other_source_of_metadata_body: Rc<Panel>,

    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `tempPanel = new JPanel()` (never used).
    #[allow(dead_code)]
    temp_panel: Rc<JComponent>,
    /// Java private final `bgInputFileSpec`.
    #[allow(dead_code)]
    bg_input_file_spec: Rc<ButtonGroup>,
    /// Java private final `rbMetadataFile`.
    rb_metadata_file: Rc<RadioButton>,
    /// Java private final `rbListOfInputFiles`.
    rb_list_of_input_files: Rc<RadioButton>,
    /// Java private final `rbSelectedFiles`.
    rb_selected_files: Rc<RadioButton>,
    /// Java private final `ltfDirectory`.
    ltf_directory: Rc<LabeledTextField>,
    /// Java private final `textAreaInputFiles = new TextArea(INPUT_FILES_LABEL, 7,
    /// 15)`.
    text_area_input_files: Rc<TextArea>,
    /// Java private final `scrollPaneInputFiles = new
    /// JScrollPane(textAreaInputFiles)`.
    scroll_pane_input_files: Rc<JComponent>,
    /// Java private final `ltfRootnameOutputFiles`.
    ltf_rootname_output_files: Rc<LabeledTextField>,

    /// Java private final `cbCorrespondingStack`.
    cb_corresponding_stack: Rc<CheckBox>,
    /// Java private final `cbTiltAngleFile`.
    cb_tilt_angle_file: Rc<CheckBox>,
    /// Java private final `cbAnglesInFilenames`.
    cb_angles_in_filenames: Rc<CheckBox>,
    /// Java private final `ltfDelimitersOpen`.
    ltf_delimiters_open: Rc<LabeledTextField>,
    /// Java private final `ltfDelimitersClose`.
    ltf_delimiters_close: Rc<LabeledTextField>,
    /// Java private final `ltfAxisRotationAngle`.
    ltf_axis_rotation_angle: Rc<LabeledTextField>,
    /// Java private final `cbRefAndDefectFromTitles`.
    cb_ref_and_defect_from_titles: Rc<CheckBox>,
    /// Java private final `strRotationAndFlip` (a `JLabel`).
    str_rotation_and_flip: Rc<JComponent>,
    /// Java private final `bgRotationAndFlip`.
    #[allow(dead_code)]
    bg_rotation_and_flip: Rc<ButtonGroup>,
    /// Java private final `rbRotationAndFlip`.
    rb_rotation_and_flip: Rc<RadioButton>,
    /// Java private final `rbRotationAndFlip2`.
    rb_rotation_and_flip2: Rc<RadioButton>,
    /// Java private final `spRotationAndFlip`.
    sp_rotation_and_flip: Rc<Spinner>,
    /// Java private final `bgTruncateValues`.
    #[allow(dead_code)]
    bg_truncate_values: Rc<ButtonGroup>,
    /// Java private final `rbTruncateAboveNone`.
    rb_truncate_above_none: Rc<RadioButton>,
    /// Java private final `rbTruncateAboveInputCounts`.
    rb_truncate_above_input_counts: Rc<RadioButton>,
    /// Java private final `tfTruncateAboveInputCounts`.
    tf_truncate_above_input_counts: Rc<TextField>,
    /// Java private final `strTruncateAboveInputCounts` (a `JLabel`).
    str_truncate_above_input_counts: Rc<JComponent>,
    /// Java private final `rbTruncateAboveSDs`.
    rb_truncate_above_sds: Rc<RadioButton>,
    /// Java private final `tfTruncateAboveSDs`.
    tf_truncate_above_sds: Rc<TextField>,
    /// Java private final `strTruncateAboveSDs` (a `JLabel`).
    str_truncate_above_sds: Rc<JComponent>,
    /// Java private final `bgPairwiseFrames`.
    #[allow(dead_code)]
    bg_pairwise_frames: Rc<ButtonGroup>,
    /// Java private final `rbCustomPairwiseFrames`.
    rb_custom_pairwise_frames: Rc<RadioButton>,
    /// Java private final `spPairwiseFrames`.
    sp_pairwise_frames: Rc<Spinner>,
    /// Java private final `strCustomPairwiseFrames` (a `JLabel`).
    str_custom_pairwise_frames: Rc<JComponent>,
    /// Java private final `rbHalfPairwiseFrames`.
    rb_half_pairwise_frames: Rc<RadioButton>,
    /// Java private final `rbAllPairwiseFrames`.
    rb_all_pairwise_frames: Rc<RadioButton>,
    /// Java private final `bgBinningForAlignment`.
    #[allow(dead_code)]
    bg_binning_for_alignment: Rc<ButtonGroup>,
    /// Java private final `rbBinningDefault`.
    rb_binning_default: Rc<RadioButton>,
    /// Java private final `rbReduceBy`.
    rb_reduce_by: Rc<RadioButton>,
    /// Java private final `spReduceBy`.
    sp_reduce_by: Rc<Spinner>,
    /// Java private final `rbTargetAlignSize`.
    rb_target_align_size: Rc<RadioButton>,
    /// Java private final `spTargetAlignSize`.
    sp_target_align_size: Rc<Spinner>,
    /// Java private final `strTargetAlignSize` (a `JLabel`).
    str_target_align_size: Rc<JComponent>,
    /// Java private final `rbTestBinnings`.
    rb_test_binnings: Rc<RadioButton>,
    /// Java private final `tfTestBinnings`.
    tf_test_binnings: Rc<TextField>,
    /// Java private final `ltfFilterCutoffs`.
    ltf_filter_cutoffs: Rc<LabeledTextField>,
    /// Java private final `cbUseHybridShifts`.
    cb_use_hybrid_shifts: Rc<CheckBox>,
    /// Java private final `ltfShiftLimit`.
    ltf_shift_limit: Rc<LabeledTextField>,
    /// Java private final `strShiftLimit` (a `JLabel`).
    str_shift_limit: Rc<JComponent>,
    /// Java private final `cbGroupFrames`.
    cb_group_frames: Rc<CheckBox>,
    /// Java private final `spGroupFrames`.
    sp_group_frames: Rc<Spinner>,
    /// Java private final `cbRefineAlignment`.
    cb_refine_alignment: Rc<CheckBox>,
    /// Java private final `spRefineAlignment`.
    sp_refine_alignment: Rc<Spinner>,
    /// Java private final `strRefineAlignment` (a `JLabel`).
    str_refine_alignment: Rc<JComponent>,
    /// Java private final `cbRefineWithGroupSums`.
    cb_refine_with_group_sums: Rc<CheckBox>,
    /// Java private final `ltfRefineRadius2`.
    ltf_refine_radius2: Rc<LabeledTextField>,
    /// Java private final `ltfStopIterationsAtShift`.
    ltf_stop_iterations_at_shift: Rc<LabeledTextField>,
    /// Java private final `strStopIterationsAtShift` (a `JLabel`).
    str_stop_iterations_at_shift: Rc<JComponent>,
    /// Java private final `cbMinForSplineSmoothing`.
    cb_min_for_spline_smoothing: Rc<CheckBox>,
    /// Java private final `spMinForSplineSmoothing`.
    sp_min_for_spline_smoothing: Rc<Spinner>,
    /// Java private final `strMinForSplineSmoothing` (a `JLabel`).
    str_min_for_spline_smoothing: Rc<JComponent>,
    /// Java private final `ltfStartingEndingFramesFirst`.
    ltf_starting_ending_frames_first: Rc<LabeledTextField>,
    /// Java private final `ltfStartingEndingFramesSecond`.
    ltf_starting_ending_frames_second: Rc<LabeledTextField>,
    /// Java private final `cbDoDoseWeighting`.
    cb_do_dose_weighting: Rc<CheckBox>,
    /// Java private final `bgDoseWeighting`.
    #[allow(dead_code)]
    bg_dose_weighting: Rc<ButtonGroup>,
    /// Java private final `rbFixedTotalDose`.
    rb_fixed_total_dose: Rc<RadioButton>,
    /// Java private final `tfFixedTotalDose`.
    tf_fixed_total_dose: Rc<TextField>,
    /// Java private final `strFixedTotalDoseUnit` (a `JLabel`).
    str_fixed_total_dose_unit: Rc<JComponent>,
    /// Java private final `rbDoseWeightingFile`.
    rb_dose_weighting_file: Rc<RadioButton>,
    /// Java private final `cbNormalizeDoseWeighting`.
    cb_normalize_dose_weighting: Rc<CheckBox>,
    /// Java private final `cbVoltage`.
    cb_voltage: Rc<CheckBox>,
    /// Java private final `ltfOptimalDoseScaling`.
    ltf_optimal_dose_scaling: Rc<LabeledTextField>,
    /// Java private final `cbUnweightedOutputFile`.
    cb_unweighted_output_file: Rc<CheckBox>,
    /// Java private final `strScalingOfSum` (a `JLabel`).
    str_scaling_of_sum: Rc<JComponent>,
    /// Java private `bgScalingOfSum` (never reassigned).
    #[allow(dead_code)]
    bg_scaling_of_sum: Rc<ButtonGroup>,
    /// Java private final `rbScalingOfSumDefault`.
    rb_scaling_of_sum_default: Rc<RadioButton>,
    /// Java private final `rbScalingOfSumFactor`.
    rb_scaling_of_sum_factor: Rc<RadioButton>,
    /// Java private final `tfScalingOfSumFactor`.
    tf_scaling_of_sum_factor: Rc<TextField>,
    /// Java private final `strModeToOutput` (a `JLabel`).
    str_mode_to_output: Rc<JComponent>,
    /// Java private final `bgModeToOutput`.
    #[allow(dead_code)]
    bg_mode_to_output: Rc<ButtonGroup>,
    /// Java private final `rbModeToOutput16bitInt`.
    rb_mode_to_output_16bit_int: Rc<RadioButton>,
    /// Java private final `rbModeToOutputFloat`.
    rb_mode_to_output_float: Rc<RadioButton>,
    /// Java private final `strSumRotationAndFlip` (a `JLabel`).
    str_sum_rotation_and_flip: Rc<JComponent>,
    /// Java private final `bgSumRotationAndFlip`.
    #[allow(dead_code)]
    bg_sum_rotation_and_flip: Rc<ButtonGroup>,
    /// Java private final `rbSumRotationAndFlip`.
    rb_sum_rotation_and_flip: Rc<RadioButton>,
    /// Java private final `rbSumRotationAndFlip2`.
    rb_sum_rotation_and_flip2: Rc<RadioButton>,
    /// Java private final `spSumRotationAndFlip`.
    sp_sum_rotation_and_flip: Rc<Spinner>,
    /// Java private final `spAlignAndSumBinningValue2`.
    sp_align_and_sum_binning_value2: Rc<Spinner>,
    /// Java private final `ltfOutputImageFile`.
    ltf_output_image_file: Rc<LabeledTextField>,
    /// Java private final `cbUseGPU`.
    cb_use_gpu: Rc<CheckBox>,
    /// Java private final `btnAdvanced = GlobalExpandButton.getInstance("Advanced",
    /// "Basic")`.
    btn_advanced: Rc<GlobalExpandButton>,

    /// Java private final `panelId`.
    #[allow(dead_code)]
    panel_id: PanelId,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static ToolsManager,
    /// Java private final `dialogType`.
    #[allow(dead_code)]
    dialog_type: DialogType,
    /// Java private final `btnLoadStartingComFile`.
    btn_load_starting_com_file: Rc<MultiLineButton>,
    /// Java private final `btnRunAlignFrames`.
    btn_run_align_frames: Rc<MultiLineButton>,
    /// Java private final `btnPlotAllResults`.
    btn_plot_all_results: Rc<MultiLineButton>,
    /// Java private final `btnOpenOutputTiltSeries`.
    btn_open_output_tilt_series: Rc<Run3dmodButton>,
    /// Java private final `btnSetupReconstruction`.
    btn_setup_reconstruction: Rc<MultiLineButton>,
    /// Java private final `bctfMetadataFile`.
    bctf_metadata_file: Rc<ButtonControlTextEfield>,
    /// Java private final `bctfListOfInputFiles`.
    bctf_list_of_input_files: Rc<ButtonControlTextEfield>,
    /// Java private final `btnSelectedFiles`.
    btn_selected_files: Rc<Ebutton>,
    /// Java private final `bctfCorrespondingStack`.
    bctf_corresponding_stack: Rc<ButtonControlTextEfield>,
    /// Java private final `bctfTiltAngleFile`.
    bctf_tilt_angle_file: Rc<ButtonControlTextEfield>,
    /// Java private final `bctfGainReferenceFile`.
    bctf_gain_reference_file: Rc<ButtonControlTextEfield>,
    /// Java private final `bctfCameraDefectFile`.
    bctf_camera_defect_file: Rc<ButtonControlTextEfield>,
    /// Java private final `bctfPathToFramesInMdoc`.
    bctf_path_to_frames_in_mdoc: Rc<ButtonControlTextEfield>,
    /// Java private final `cmbRotationAndFlipTranslation`.
    cmb_rotation_and_flip_translation: Rc<ComboBox>,
    /// Java private final `cmbSumRotationAndFlipTranslation`.
    cmb_sum_rotation_and_flip_translation: Rc<ComboBox>,
    /// Java private final `rotationAndFlipStringList`.
    rotation_and_flip_string_list: Vec<String>,
    /// Java private final `fcLocalArgumentsDir = new JFileChooser()`.
    fc_local_arguments_dir: Rc<FileChooser>,
    /// Java private final `phDoseWeighting`.
    ph_dose_weighting: Rc<PanelHeader>,
    /// Java private final `phOtherSourceOfMetadata`.
    ph_other_source_of_metadata: Rc<PanelHeader>,
    /// Java private final `eerSuperResZSumPaddingPanel`.
    eer_super_res_z_sum_padding_panel: Rc<EERSuperResZSumPaddingPanel>,

    /// Java private `localArgumentsDir = ""`.
    local_arguments_dir: RefCell<String>,

    /// Java private `alignFramesBrowsingDir`.
    align_frames_browsing_dir: RefCell<Option<PathBuf>>,
    /// Java private `doseWeightingAdvanced`.
    dose_weighting_advanced: Cell<bool>,

    /// Rust-only: Java `this` as the `ActionListener` it registers.
    action_listener: ActionListener,
    /// Rust-only: Java `this` as the `ChangeListener` it registers.
    change_listener: ChangeListener,
}

impl AlignFramesPanel {
    /// Java private constructor `AlignFramesPanel(ToolsManager, AxisID,
    /// DialogType)`, with the field initializers.
    fn new(
        manager: &'static ToolsManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<AlignFramesPanel> {
        let base_manager: &'static dyn BaseManager = manager;
        let instance = Rc::new_cyclic(|self_ref: &Weak<AlignFramesPanel>| {
            // Field initializers, in declaration order.
            let bg_input_file_spec = ButtonGroup::new();
            let rb_metadata_file = RadioButton::new_string_button_group(
                Some("Metadata (.mdoc) file:"),
                Some(&bg_input_file_spec),
            );
            let rb_list_of_input_files = RadioButton::new_string_button_group(
                Some("Text file with list of files:"),
                Some(&bg_input_file_spec),
            );
            let rb_selected_files = RadioButton::new_string_button_group(
                Some("Selected Files"),
                Some(&bg_input_file_spec),
            );
            let ltf_directory = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some(&format!("{DIRECTORY_LABEL}: ")),
            );
            let text_area_input_files = TextArea::new(Some(INPUT_FILES_LABEL), 7, 15);
            let scroll_pane_input_files =
                JComponent::new_scroll_pane(Some(&text_area_input_files.get_component()));
            let ltf_rootname_output_files = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some(&format!("{ROOTNAME_OUTPUT_FILES_LABEL}: ")),
            );
            let cb_corresponding_stack = CheckBox::new_string(Some(CORRESPONDING_STACK_LABEL));
            let cb_tilt_angle_file = CheckBox::new_string(Some(TILT_ANGLE_FILE_LABEL));
            let cb_angles_in_filenames = CheckBox::new_string(Some(ANGLES_IN_FILENAMES_LABEL));
            let ltf_delimiters_open = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some(DELIMITER_OPEN_LABEL),
            );
            let ltf_delimiters_close = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some(DELIMITER_CLOSE_LABEL),
            );
            let ltf_axis_rotation_angle = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some(&format!("{AXIS_ROTATION_ANGLE_LABEL}: ")),
            );
            let cb_ref_and_defect_from_titles =
                CheckBox::new_string(Some(REF_AND_DEFECT_FROM_TITLES_LABEL));
            let str_rotation_and_flip = JComponent::new_label(ROTATION_AND_FLIP_LABEL);
            let bg_rotation_and_flip = ButtonGroup::new();
            let rb_rotation_and_flip = RadioButton::new_string_button_group(
                Some("Value in title"),
                Some(&bg_rotation_and_flip),
            );
            let rb_rotation_and_flip2 = RadioButton::new_string_string_button_group(
                Some(""),
                Some("rotation and flip spinner"),
                Some(&bg_rotation_and_flip),
            );
            let sp_rotation_and_flip = Spinner::get_instance_string_int_int_int(
                Some(ROTATION_AND_FLIP_LABEL),
                align_frames_param::ROTATION_AND_FLIP_SPINNER_DEFAULT,
                align_frames_param::ROTATION_AND_FLIP_SPINNER_MIN,
                align_frames_param::ROTATION_AND_FLIP_SPINNER_MAX,
            );
            let bg_truncate_values = ButtonGroup::new();
            let rb_truncate_above_none =
                RadioButton::new_string_button_group(Some("None"), Some(&bg_truncate_values));
            let rb_truncate_above_input_counts =
                RadioButton::new_string_button_group(Some(ABOVE_LABEL), Some(&bg_truncate_values));
            let tf_truncate_above_input_counts =
                TextField::new(FieldType::Integer, Some(ABOVE_LABEL), None);
            let str_truncate_above_input_counts = JComponent::new_label(INPUT_COUNTS_LABEL);
            let rb_truncate_above_sds =
                RadioButton::new_string_button_group(Some(ABOVE_LABEL), Some(&bg_truncate_values));
            let tf_truncate_above_sds = TextField::new(FieldType::Integer, Some(ABOVE_LABEL), None);
            let str_truncate_above_sds = JComponent::new_label(SDS_FROM_MEAN_LABEL);
            let bg_pairwise_frames = ButtonGroup::new();
            let rb_custom_pairwise_frames = RadioButton::new_string_button_group(
                Some("Fit to sets of"),
                Some(&bg_pairwise_frames),
            );
            let sp_pairwise_frames = Spinner::get_instance_string_int_int_int(
                Some("Fit to sets of"),
                align_frames_param::PAIRWISE_FRAMES_SPINNER_DEFAULT,
                align_frames_param::PAIRWISE_FRAMES_SPINNER_MIN,
                align_frames_param::PAIRWISE_FRAMES_SPINNER_MAX,
            );
            let str_custom_pairwise_frames = JComponent::new_label(" frames ");
            let rb_half_pairwise_frames = RadioButton::new_string_button_group(
                Some("Fit to sets of half the frames"),
                Some(&bg_pairwise_frames),
            );
            let rb_all_pairwise_frames = RadioButton::new_string_button_group(
                Some("One fit to all frames"),
                Some(&bg_pairwise_frames),
            );
            let bg_binning_for_alignment = ButtonGroup::new();
            let rb_binning_default = RadioButton::new_string_button_group(
                Some("Default"),
                Some(&bg_binning_for_alignment),
            );
            let rb_reduce_by = RadioButton::new_string_button_group(
                Some("Reduce by"),
                Some(&bg_binning_for_alignment),
            );
            let sp_reduce_by = Spinner::get_instance_string_int_int_int(
                Some("Reduce by"),
                align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER_DEFAULT,
                align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER_MIN,
                align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER_MAX,
            );
            let rb_target_align_size = RadioButton::new_string_button_group(
                Some("Reduce to about"),
                Some(&bg_binning_for_alignment),
            );
            let sp_target_align_size = Spinner::get_instance_string_int_int_int_int(
                Some("Reduce to about"),
                align_frames_param::TARGET_ALIGN_SIZE_SPINNER_DEFAULT,
                align_frames_param::TARGET_ALIGN_SIZE_SPINNER_MIN,
                align_frames_param::TARGET_ALIGN_SIZE_SPINNER_MAX,
                align_frames_param::TARGET_ALIGN_SIZE_SPINNER_STEP_SIZE,
            );
            let str_target_align_size = JComponent::new_label("  pixels  ");
            let rb_test_binnings = RadioButton::new_string_button_group(
                Some(TEST_BINNINGS_LABEL),
                Some(&bg_binning_for_alignment),
            );
            let tf_test_binnings =
                TextField::new(FieldType::String, Some(TEST_BINNINGS_LABEL), None);
            let ltf_filter_cutoffs = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some("Filter cutoffs: "),
            );
            let cb_use_hybrid_shifts = CheckBox::new_string(Some("Use hybrid shifts"));
            let ltf_shift_limit = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(SHIFT_LIMIT_LABEL),
            );
            let str_shift_limit = JComponent::new_label(" unbinned pixels");
            let cb_group_frames = CheckBox::new_string(Some("Group frames by"));
            let sp_group_frames = Spinner::get_instance_string_int_int_int(
                Some("Group frames by"),
                align_frames_param::GROUP_FRAMES_SPINNER_DEFAULT,
                align_frames_param::GROUP_FRAMES_SPINNER_MIN,
                align_frames_param::GROUP_FRAMES_SPINNER_MAX,
            );
            let cb_refine_alignment = CheckBox::new_string(Some("Refine alignment with up to"));
            // The Java uses ALIGN_AND_SUM_BINNING_SPINNER_MAX (16) as this spinner's
            // maximum, not REFINE_ALIGNMENT_SPINNER_MAX (10); setParameters still
            // only accepts values up to 10.  Kept: it may be intended.
            let sp_refine_alignment = Spinner::get_instance_string_int_int_int(
                Some("Refine alignment with up to"),
                align_frames_param::REFINE_ALIGNMENT_SPINNER_DEFAULT,
                align_frames_param::REFINE_ALIGNMENT_SPINNER_MIN,
                align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER_MAX,
            );
            let str_refine_alignment = JComponent::new_label(" iterations ");
            let cb_refine_with_group_sums = CheckBox::new_string(Some("Refine in groups"));
            let ltf_refine_radius2 = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Filter cutoff for refining "),
            );
            let ltf_stop_iterations_at_shift = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Refine until changes are below "),
            );
            let str_stop_iterations_at_shift = JComponent::new_label(" unbinned pixels ");
            let cb_min_for_spline_smoothing =
                CheckBox::new_string(Some("Spline smoothing of shifts if more than"));
            let sp_min_for_spline_smoothing = Spinner::get_instance_string_int_int_int(
                Some("Spline smoothing of shifts if more than"),
                align_frames_param::MIN_FOR_SPLINE_SMOOTHING_SPINNER_DEFAULT,
                align_frames_param::MIN_FOR_SPLINE_SMOOTHING_SPINNER_MIN,
                align_frames_param::MIN_FOR_SPLINE_SMOOTHING_SPINNER_MAX,
            );
            let str_min_for_spline_smoothing = JComponent::new_label(" frames ");
            let ltf_starting_ending_frames_first = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(&format!("{STARTING_ENDING_FRAMES_FIRST_LABEL} ")),
            );
            let ltf_starting_ending_frames_second = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(&format!("{STARTING_ENDING_FRAMES_SECOND_LABEL} ")),
            );
            let cb_do_dose_weighting = CheckBox::new_string(Some("Do dose weighting"));
            let bg_dose_weighting = ButtonGroup::new();
            let rb_fixed_total_dose = RadioButton::new_string_button_group(
                Some(FIXED_TOTAL_DOSE_LABEL),
                Some(&bg_dose_weighting),
            );
            let tf_fixed_total_dose =
                TextField::new(FieldType::FloatingPoint, Some(FIXED_TOTAL_DOSE_LABEL), None);
            let str_fixed_total_dose_unit = JComponent::new_label("e/A2");
            let rb_dose_weighting_file = RadioButton::new_string_button_group(
                Some("Dose weighting with doses from .mdoc file"),
                Some(&bg_dose_weighting),
            );
            let cb_normalize_dose_weighting =
                CheckBox::new_string(Some("Normalize within each set of frames"));
            let cb_voltage = CheckBox::new_string(Some("Microscope voltage is 200 kV"));
            let ltf_optimal_dose_scaling = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(OPTIMAL_DOSE_SCALING_LABEL),
            );
            let cb_unweighted_output_file =
                CheckBox::new_string(Some("Non-dose weighted output also"));
            let str_scaling_of_sum = JComponent::new_label(SCALING_OF_SUM_LABEL);
            let bg_scaling_of_sum = ButtonGroup::new();
            let rb_scaling_of_sum_default =
                RadioButton::new_string_button_group(Some("Default"), Some(&bg_scaling_of_sum));
            let rb_scaling_of_sum_factor = RadioButton::new_string_button_group(
                Some(SCALING_OF_SUM_FACTOR_LABEL),
                Some(&bg_scaling_of_sum),
            );
            let tf_scaling_of_sum_factor = TextField::new(
                FieldType::FloatingPoint,
                Some(&format!(
                    "{SCALING_OF_SUM_LABEL} {SCALING_OF_SUM_FACTOR_LABEL}"
                )),
                None,
            );
            let str_mode_to_output = JComponent::new_label("Output file mode");
            let bg_mode_to_output = ButtonGroup::new();
            let rb_mode_to_output_16bit_int = RadioButton::new_string_button_group(
                Some("16-bit integers"),
                Some(&bg_mode_to_output),
            );
            let rb_mode_to_output_float = RadioButton::new_string_button_group(
                Some("Floating point"),
                Some(&bg_mode_to_output),
            );
            let str_sum_rotation_and_flip = JComponent::new_label(SUM_ROTATION_AND_FLIP_LABEL);
            let bg_sum_rotation_and_flip = ButtonGroup::new();
            let rb_sum_rotation_and_flip = RadioButton::new_string_button_group(
                Some("Value in title"),
                Some(&bg_sum_rotation_and_flip),
            );
            let rb_sum_rotation_and_flip2 = RadioButton::new_string_string_button_group(
                Some(""),
                Some("sum rotation and flip spinner"),
                Some(&bg_sum_rotation_and_flip),
            );
            let sp_sum_rotation_and_flip = Spinner::get_instance_string_int_int_int(
                Some(SUM_ROTATION_AND_FLIP_LABEL),
                align_frames_param::SUM_ROTATION_AND_FLIP_SPINNER_DEFAULT,
                align_frames_param::SUM_ROTATION_AND_FLIP_SPINNER_MIN,
                align_frames_param::SUM_ROTATION_AND_FLIP_SPINNER_MAX,
            );
            let sp_align_and_sum_binning_value2 = Spinner::get_labeled_instance_string_int_int_int(
                Some(ALIGN_AND_SUM_BINNING_VALUE2_LABEL),
                align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER2_DEFAULT,
                align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER2_MIN,
                align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER2_MAX,
            );
            let ltf_output_image_file = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some(&format!("{OUTPUT_IMAGE_FILE_LABEL}: ")),
            );
            let cb_use_gpu = CheckBox::new_string(Some("Use the GPU"));
            let btn_advanced = GlobalExpandButton::get_instance(Some("Advanced"), Some("Basic"));
            // Java `this` as the ActionListener and ChangeListener.
            let adaptee = self_ref.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action_performed(Some(event));
                }
            });
            let adaptee = self_ref.clone();
            let change_listener: ChangeListener = Rc::new(move |event: &ChangeEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.state_changed(event);
                }
            });

            // Constructor body.
            let container: Weak<dyn Run3dmodButtonContainer> = self_ref.clone();
            let control_target: Weak<dyn ControlTarget> = self_ref.clone();
            let expandable: Weak<dyn Expandable> = self_ref.clone();
            let panel_id = PanelId::AlignFrames;

            let btn_load_starting_com_file =
                MultiLineButton::new_string(Some("Load Starting Com File"));
            let btn_run_align_frames = MultiLineButton::new_string(Some("Run Alignframes"));
            let btn_plot_all_results = MultiLineButton::new_string(Some("Plot All Results"));
            let btn_open_output_tilt_series =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Output Tilt Series"),
                    Some(container),
                );
            let btn_setup_reconstruction =
                MultiLineButton::new_string(Some("Setup Reconstruction"));
            // Java `new JFileChooser()`.
            let fc_local_arguments_dir = FileChooser::new_void();
            fc_local_arguments_dir.set_dialog_title(Some("Select Reconstruction Directory"));
            fc_local_arguments_dir.set_file_selection_mode(file_chooser::DIRECTORIES_ONLY);
            // Java `fcLocalArgumentsDir.setCurrentDirectory(getBrowsingDir())`: the
            // browsing directory is still null at this point of the constructor.
            fc_local_arguments_dir.set_current_directory(None);
            let metadata_file_extension = SelectFileExtension::new();
            metadata_file_extension.set_file_chooser_title(Some("Select metadata file"));
            let bctf_metadata_file =
                ButtonControlTextEfield::get_file_instance_string_select_file_extension_boolean_boolean(
                    Some("Metadata file"),
                    Some(metadata_file_extension),
                    false,
                    false,
                );
            bctf_metadata_file.set_file_filter(Some(Rc::new(MdocFileFilter::get_instance(
                Some(base_manager),
                true,
            )) as Rc<dyn FileFilter>));
            bctf_metadata_file.set_limit_displayed_file_path(20);
            // bctfMetadataFile.setAltBrowsingDirectory(this): after the Rc exists
            // (see the module docs).
            let list_of_input_files_extension = SelectFileExtension::new();
            list_of_input_files_extension
                .set_file_chooser_title(Some("Select text file with list of input files"));
            let bctf_list_of_input_files =
                ButtonControlTextEfield::get_file_instance_string_select_file_extension_boolean_boolean(
                    Some("Text file with list of files"),
                    Some(list_of_input_files_extension),
                    false,
                    false,
                );
            bctf_list_of_input_files.set_file_filter(Some(
                Rc::new(TextFileFileFilter::get_instance(Some(base_manager), true))
                    as Rc<dyn FileFilter>,
            ));
            bctf_list_of_input_files.set_limit_displayed_file_path(20);
            // bctfListOfInputFiles.setAltBrowsingDirectory(this): see above.
            let selected_files_extension = SelectFileExtension::new();
            selected_files_extension.set_file_chooser_title(Some("Select multiple files"));
            let btn_selected_files =
                Ebutton::get_select_multiple_files_instance_string_control_target_select_file_extension(
                    Some("Selected files"),
                    Some(control_target),
                    Some(selected_files_extension),
                );
            btn_selected_files.set_file_filter(Some(Rc::new(FrameFileFilter::get_instance(
                Some(base_manager),
                true,
            )) as Rc<dyn FileFilter>));
            // btnSelectedFiles.setAltBrowsingDirectory(this): see above.
            let bctf_path_to_frames_in_mdoc =
                ButtonControlTextEfield::get_labeled_file_instance_string_boolean_boolean(
                    Some(PATH_TO_FRAMES_IN_MDOC_LABEL),
                    true,
                    false,
                );
            bctf_path_to_frames_in_mdoc.set_file_selection_mode(file_chooser::DIRECTORIES_ONLY);
            // bctfPathToFramesInMdoc.setAltBrowsingDirectory(this): see above.
            let bctf_corresponding_stack = ButtonControlTextEfield::get_file_instance_string(Some(
                "Matching tilt series file",
            ));
            // bctfCorrespondingStack.setAltBrowsingDirectory(this): see above.
            let bctf_tilt_angle_file = ButtonControlTextEfield::get_file_instance_string(Some(
                "Text file with tilt angles",
            ));
            bctf_tilt_angle_file.set_file_filter(Some(Rc::new(
                TiltExtensionFileFilter::get_instance(Some(base_manager), true),
            ) as Rc<dyn FileFilter>));
            // bctfTiltAngleFile.setAltBrowsingDirectory(this): see above.
            let gain_reference_file_extension = SelectFileExtension::new();
            gain_reference_file_extension
                .set_file_chooser_title(Some("Select gain reference file"));
            let bctf_gain_reference_file =
                ButtonControlTextEfield::get_labeled_file_instance_string_boolean_boolean_select_file_extension(
                    Some("Gain reference file: "),
                    true,
                    false,
                    Some(gain_reference_file_extension),
                );
            // bctfGainReferenceFile.setAltBrowsingDirectory(this): see above.
            let bctf_camera_defect_file =
                ButtonControlTextEfield::get_labeled_file_instance_string_boolean_boolean(
                    Some("Defect file: "),
                    true,
                    false,
                );
            // bctfCameraDefectFile.setAltBrowsingDirectory(this): see above.
            btn_advanced.register_expandable(expandable.clone());
            let ph_dose_weighting =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Dose Weighting"),
                    Some(expandable.clone()),
                    Some(dialog_type),
                    Some(btn_advanced.clone()),
                );
            let ph_other_source_of_metadata = PanelHeader::get_instance(
                Some("Other sources of metadata"),
                Some(expandable),
                Some(dialog_type),
            );
            let eer_super_res_z_sum_padding_panel =
                EERSuperResZSumPaddingPanel::get_instance(base_manager, axis_id, dialog_type);
            let dose_weighting_advanced = false;
            ph_dose_weighting.set_open(false);
            let cmb_rotation_and_flip_translation =
                ComboBox::get_unlabeled_instance(Some("SpinnerRotationAndFlip"));
            let cmb_sum_rotation_and_flip_translation =
                ComboBox::get_unlabeled_instance(Some("SpinnerSumRotationAndFlip"));
            let rotation_and_flip_string_list: Vec<String> = [
                "0 degrees, No flip",
                "90 degrees, No flip",
                "180 degrees, No flip",
                "270 degrees, No flip",
                "0 degrees, Flip",
                "90 degrees, Flip",
                "180 degrees, Flip",
                "270 degrees, Flip",
            ]
            .iter()
            .map(|label| label.to_string())
            .collect();

            AlignFramesPanel {
                self_ref: self_ref.clone(),
                pnl_tab_array: RefCell::new(Vec::new()),
                pnl_tab_body_array: RefCell::new(Vec::new()),
                cur_tab: Cell::new(None),
                tab_pane: TabbedPane::new(),
                pnl_outer_panel: Panel::new(),
                pnl_btn_advanced_basic: JComponent::new_panel(),
                pnl_btn_load_com_file: JComponent::new_panel(),
                pnl_rootname_output_files: JComponent::new_panel(),
                pnl_outer_path_to_frames_in_mdoc: JComponent::new_panel(),
                pnl_path_to_frames_in_mdoc: Panel::new(),
                pnl_outer_gain_reference: JComponent::new_panel(),
                pnl_gain_reference: JComponent::new_panel(),
                pnl_gain_reference_file: Panel::new(),
                pnl_rotation_and_flip: Panel::new(),
                pnl_outer_camera_defect_file: JComponent::new_panel(),
                pnl_camera_defect_file: Panel::new(),
                pnl_truncate_values: JComponent::new_panel(),
                pnl_fit_pairwise_shifts: JComponent::new_panel(),
                pnl_binning_for_alignment: JComponent::new_panel(),
                pnl_test_binnings: JComponent::new_panel(),
                pnl_vary_filter: JComponent::new_panel(),
                pnl_filter_cutoffs: JComponent::new_panel(),
                pnl_shift_limit: JComponent::new_panel(),
                pnl_group_frames: JComponent::new_panel(),
                pnl_refinement: JComponent::new_panel(),
                pnl_refine_with_group_sums: JComponent::new_panel(),
                pnl_stop_iterations_at_shift: JComponent::new_panel(),
                pnl_min_for_spline_smoothing: JComponent::new_panel(),
                pnl_starting_ending_frames: JComponent::new_panel(),
                pnl_do_dose_weighting: JComponent::new_panel(),
                pnl_dose_weighting: JComponent::new_panel(),
                pnl_dose_weighting_body: JComponent::new_panel(),
                pnl_fixed_total_dose: JComponent::new_panel(),
                pnl_scaling_of_sum: JComponent::new_panel(),
                pnl_mode_to_outout: JComponent::new_panel(),
                pnl_sum_rotation_and_flip: JComponent::new_panel(),
                pnl_align_and_sum_binning: JComponent::new_panel(),
                pnl_output_image_file: JComponent::new_panel(),
                pnl_use_gpu: JComponent::new_panel(),
                pnl_run_align_frames_buttons: JComponent::new_panel(),
                pnl_start_recon_button: JComponent::new_panel(),
                pnl_input_file_spec_right: JComponent::new_panel(),
                pnl_other_source_of_metadata_body: Panel::new(),
                pnl_root: SpacedPanel::get_instance_void(),
                temp_panel: JComponent::new_panel(),
                bg_input_file_spec,
                rb_metadata_file,
                rb_list_of_input_files,
                rb_selected_files,
                ltf_directory,
                text_area_input_files,
                scroll_pane_input_files,
                ltf_rootname_output_files,
                cb_corresponding_stack,
                cb_tilt_angle_file,
                cb_angles_in_filenames,
                ltf_delimiters_open,
                ltf_delimiters_close,
                ltf_axis_rotation_angle,
                cb_ref_and_defect_from_titles,
                str_rotation_and_flip,
                bg_rotation_and_flip,
                rb_rotation_and_flip,
                rb_rotation_and_flip2,
                sp_rotation_and_flip,
                bg_truncate_values,
                rb_truncate_above_none,
                rb_truncate_above_input_counts,
                tf_truncate_above_input_counts,
                str_truncate_above_input_counts,
                rb_truncate_above_sds,
                tf_truncate_above_sds,
                str_truncate_above_sds,
                bg_pairwise_frames,
                rb_custom_pairwise_frames,
                sp_pairwise_frames,
                str_custom_pairwise_frames,
                rb_half_pairwise_frames,
                rb_all_pairwise_frames,
                bg_binning_for_alignment,
                rb_binning_default,
                rb_reduce_by,
                sp_reduce_by,
                rb_target_align_size,
                sp_target_align_size,
                str_target_align_size,
                rb_test_binnings,
                tf_test_binnings,
                ltf_filter_cutoffs,
                cb_use_hybrid_shifts,
                ltf_shift_limit,
                str_shift_limit,
                cb_group_frames,
                sp_group_frames,
                cb_refine_alignment,
                sp_refine_alignment,
                str_refine_alignment,
                cb_refine_with_group_sums,
                ltf_refine_radius2,
                ltf_stop_iterations_at_shift,
                str_stop_iterations_at_shift,
                cb_min_for_spline_smoothing,
                sp_min_for_spline_smoothing,
                str_min_for_spline_smoothing,
                ltf_starting_ending_frames_first,
                ltf_starting_ending_frames_second,
                cb_do_dose_weighting,
                bg_dose_weighting,
                rb_fixed_total_dose,
                tf_fixed_total_dose,
                str_fixed_total_dose_unit,
                rb_dose_weighting_file,
                cb_normalize_dose_weighting,
                cb_voltage,
                ltf_optimal_dose_scaling,
                cb_unweighted_output_file,
                str_scaling_of_sum,
                bg_scaling_of_sum,
                rb_scaling_of_sum_default,
                rb_scaling_of_sum_factor,
                tf_scaling_of_sum_factor,
                str_mode_to_output,
                bg_mode_to_output,
                rb_mode_to_output_16bit_int,
                rb_mode_to_output_float,
                str_sum_rotation_and_flip,
                bg_sum_rotation_and_flip,
                rb_sum_rotation_and_flip,
                rb_sum_rotation_and_flip2,
                sp_sum_rotation_and_flip,
                sp_align_and_sum_binning_value2,
                ltf_output_image_file,
                cb_use_gpu,
                btn_advanced,
                panel_id,
                axis_id,
                manager,
                dialog_type,
                btn_load_starting_com_file,
                btn_run_align_frames,
                btn_plot_all_results,
                btn_open_output_tilt_series,
                btn_setup_reconstruction,
                bctf_metadata_file,
                bctf_list_of_input_files,
                btn_selected_files,
                bctf_corresponding_stack,
                bctf_tilt_angle_file,
                bctf_gain_reference_file,
                bctf_camera_defect_file,
                bctf_path_to_frames_in_mdoc,
                cmb_rotation_and_flip_translation,
                cmb_sum_rotation_and_flip_translation,
                rotation_and_flip_string_list,
                fc_local_arguments_dir,
                ph_dose_weighting,
                ph_other_source_of_metadata,
                eer_super_res_z_sum_padding_panel,
                local_arguments_dir: RefCell::new(String::new()),
                align_frames_browsing_dir: RefCell::new(None),
                dose_weighting_advanced: Cell::new(dose_weighting_advanced),
                action_listener,
                change_listener,
            }
        });
        // The constructor's setAltBrowsingDirectory(this) calls (see the module
        // docs).
        let browsing_directory: Rc<dyn BrowsingDirectory> = instance.clone();
        instance
            .bctf_metadata_file
            .set_alt_browsing_directory(Some(browsing_directory.clone()));
        instance
            .bctf_list_of_input_files
            .set_alt_browsing_directory(Some(browsing_directory.clone()));
        instance
            .btn_selected_files
            .set_alt_browsing_directory(Some(browsing_directory.clone()));
        instance
            .bctf_path_to_frames_in_mdoc
            .set_alt_browsing_directory(Some(browsing_directory.clone()));
        instance
            .bctf_corresponding_stack
            .set_alt_browsing_directory(Some(browsing_directory.clone()));
        instance
            .bctf_tilt_angle_file
            .set_alt_browsing_directory(Some(browsing_directory.clone()));
        instance
            .bctf_gain_reference_file
            .set_alt_browsing_directory(Some(browsing_directory.clone()));
        instance
            .bctf_camera_defect_file
            .set_alt_browsing_directory(Some(browsing_directory));
        instance.set_field_displayer();
        instance.set_validation_set();
        instance.set_required_fields();
        instance.set_number_must_be_positive();
        // Java `alignFramesBrowsingDir = new File(manager.getPropertyUserDir())`.
        // Upstream bug fixed in translation (AlignFramesPanel.java:429): `new
        // File(null)` throws NullPointerException; a missing directory leaves the
        // browsing directory null.
        *instance.align_frames_browsing_dir.borrow_mut() =
            base_manager.get_property_user_dir().map(PathBuf::from);
        instance
    }

    /// Java package-private static `getToolsInstance(ToolsManager, AxisID,
    /// DialogType)`.
    pub fn get_tools_instance(
        manager: &'static ToolsManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<AlignFramesPanel> {
        let instance = AlignFramesPanel::new(manager, axis_id, dialog_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// This panel as a `FieldDisplayer` (Java `this` passed to `getText(true,
    /// this)`).
    fn field_displayer(&self) -> Option<Rc<dyn FieldDisplayer>> {
        self.self_ref
            .upgrade()
            .map(|this| this as Rc<dyn FieldDisplayer>)
    }

    /// Java private `fillComboBox(ComboBox, String[])`.
    fn fill_combo_box(&self, combo_box: Option<&ComboBox>, string_list: Option<&[String]>) {
        let Some(combo_box) = combo_box else {
            return;
        };
        let len = string_list.map_or(0, |string_list| string_list.len());
        // comboBox.setPlaceholder(EMPTY_OPTION, NO_SELECTION1 + len + NO_SELECTION2);
        if len > 0 {
            let string_list = string_list.unwrap_or(&[]);
            for item in string_list.iter().take(len) {
                combo_box.add_item(Some(item));
            }
        }
        combo_box.unselect();
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        let manager: &'static dyn BaseManager = self.manager;
        // Swing layout: FlowLayout LEADING / TRAILING / CENTER layouts.

        // init
        for i in 0..Tab::NUM_TABS {
            let pnl_tab = JComponent::new_panel();
            let pnl_tab_body = JComponent::new_panel();
            self.pnl_tab_array.borrow_mut().push(pnl_tab.clone());
            self.pnl_tab_body_array.borrow_mut().push(pnl_tab_body);
            // Java `tabPane.add(Tab.getInstance(i).toString(), pnlTabArray[i])`
            // (JTabbedPane.add(String, Component) is addTab).
            let title = Tab::get_instance(i as i32)
                .map(|tab| tab.to_string())
                .unwrap_or_else(|| "null".to_string());
            self.tab_pane.add_tab_string_component(&title, &pnl_tab);
        }
        self.tab_pane
            .get_component()
            .set_enabled_at(Tab::INPUT_AND_PREPROCESSING.index as usize, true);

        // initialize
        self.ltf_directory.set_editable(false);
        self.ltf_output_image_file.set_editable(false);
        // set defaults
        self.ltf_delimiters_open
            .set_text_string(Some(sort_tilt_frames_param::DELIMITER_OPEN_DEFAULT));
        self.ltf_delimiters_close
            .set_text_string(Some(sort_tilt_frames_param::DELIMITER_CLOSE_DEFAULT));
        self.ltf_filter_cutoffs
            .set_text_string(Some(align_frames_param::VARY_FILTER_DEFAULT));
        self.ltf_shift_limit
            .set_text_int(align_frames_param::SHIFT_LIMIT_DEFAULT);
        self.ltf_stop_iterations_at_shift
            .set_text_double(align_frames_param::STOP_ITERATIONS_AT_SHIFT_DEFAULT);
        self.rb_metadata_file.set_selected_boolean(true);
        self.rb_truncate_above_none.set_selected_boolean(true);
        self.rb_all_pairwise_frames.set_selected_boolean(true);
        self.rb_binning_default.set_selected_boolean(true);
        self.cb_use_hybrid_shifts.set_selected_boolean(true);
        self.cb_refine_alignment.set_selected_boolean(true);
        self.cb_min_for_spline_smoothing.set_selected_boolean(true);
        self.rb_fixed_total_dose.set_selected_boolean(true);
        self.cb_normalize_dose_weighting.set_selected_boolean(true);
        self.rb_scaling_of_sum_default.set_selected_boolean(true);
        self.rb_mode_to_output_16bit_int.set_selected_boolean(true);
        self.rb_rotation_and_flip.set_selected_boolean(true);
        self.rb_sum_rotation_and_flip.set_selected_boolean(true);
        // Root panel
        self.pnl_root.set_box_layout(spaced_panel::X_AXIS);
        // Swing layout: pnlRoot.getJPanel().setLayout(flowLayout).
        self.pnl_root
            .add_j_panel(&self.pnl_outer_panel.get_component());
        // Swing layout: pnlOuterPanel BoxLayout Y_AXIS.
        self.pnl_outer_panel.get_component().set_border_title(
            BeveledBorder::new(Some("Align Frames"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        // pnlOuterPanel.add(Box.createRigidArea(FixedDim.x0_y2));
        self.pnl_outer_panel
            .get_component()
            .add(&self.tab_pane.get_component());
        // Swing layout: pnlOuterPanel.add(Box.createHorizontalGlue()).
        // input and pre-processing tab
        let tab_body = self.pnl_tab_body_array.borrow().clone();
        let mut index = Tab::INPUT_AND_PREPROCESSING.index as usize;
        // Swing layout: pnlTabBodyArray[index] BoxLayout Y_AXIS.
        tab_body[index].add(&self.pnl_btn_load_com_file);
        tab_body[index].add(&self.get_pnl_input_file_specification());
        tab_body[index].add(&self.pnl_rootname_output_files);
        tab_body[index].add(&self.get_pnl_path_to_frames_in_mdoc());
        tab_body[index].add(&self.get_pnl_other_source_of_metadata());
        // Swing layout: Box.createVerticalStrut(2).
        tab_body[index].add(&self.eer_super_res_z_sum_padding_panel.get_component());
        tab_body[index].add(&self.get_pnl_gain_reference());
        tab_body[index].add(&self.get_pnl_camera_defect_file());
        tab_body[index].add(&self.pnl_truncate_values);
        // alignment tab
        index = Tab::ALIGNMENT.index as usize;
        // Swing layout: pnlTabBodyArray[index] BoxLayout Y_AXIS.
        tab_body[index].add(&self.pnl_fit_pairwise_shifts);
        tab_body[index].add(&self.pnl_binning_for_alignment);
        tab_body[index].add(&self.pnl_vary_filter);
        tab_body[index].add(&self.pnl_shift_limit);
        tab_body[index].add(&self.pnl_group_frames);
        tab_body[index].add(&self.pnl_refinement);
        tab_body[index].add(&self.pnl_min_for_spline_smoothing);
        tab_body[index].add(&self.pnl_starting_ending_frames);
        tab_body[index].add(&self.pnl_do_dose_weighting);
        tab_body[index].add(&self.pnl_dose_weighting);
        tab_body[index].add(&self.pnl_scaling_of_sum);
        tab_body[index].add(&self.pnl_mode_to_outout);
        tab_body[index].add(&self.pnl_sum_rotation_and_flip);
        tab_body[index].add(&self.pnl_align_and_sum_binning);
        tab_body[index].add(&self.pnl_output_image_file);
        tab_body[index].add(&self.pnl_use_gpu);
        self.pnl_outer_panel
            .get_component()
            .add(&self.pnl_run_align_frames_buttons);
        self.pnl_outer_panel
            .get_component()
            .add(&self.pnl_start_recon_button);
        // Load starting com file button
        // Swing layout: pnlBtnLoadComFile FlowLayout CENTER.
        self.pnl_btn_load_com_file
            .add(&self.btn_load_starting_com_file.get_component());
        // Rootname output files
        self.ltf_rootname_output_files.set_preferred_width(300);
        // Swing layout: pnlRootnameOutputFiles BoxLayout X_AXIS.
        self.pnl_rootname_output_files
            .add(&self.ltf_rootname_output_files.get_container());
        // Truncate values
        let pnl_truncate_above_input_counts = JComponent::new_panel();
        let pnl_truncate_above_sds = JComponent::new_panel();
        self.tf_truncate_above_input_counts.set_preferred_width(100);
        self.tf_truncate_above_sds.set_preferred_width(100);
        // Swing layout: pnlTruncateValues BoxLayout Y_AXIS.
        self.pnl_truncate_values.set_border_title(
            BeveledBorder::new(Some("Truncate values"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_truncate_values
            .add(&self.rb_truncate_above_none.get_component());
        self.pnl_truncate_values
            .add(&pnl_truncate_above_input_counts);
        self.pnl_truncate_values.add(&pnl_truncate_above_sds);
        // Swing layout: pnlTruncateAboveInputCounts BoxLayout X_AXIS; rigid area
        // x2_y0 before the label.
        pnl_truncate_above_input_counts.add(&self.rb_truncate_above_input_counts.get_component());
        pnl_truncate_above_input_counts.add(&self.tf_truncate_above_input_counts.get_component());
        pnl_truncate_above_input_counts.add(&self.str_truncate_above_input_counts);
        // Swing layout: pnlTruncateAboveSDs BoxLayout X_AXIS; rigid area x2_y0
        // before the label.
        pnl_truncate_above_sds.add(&self.rb_truncate_above_sds.get_component());
        pnl_truncate_above_sds.add(&self.tf_truncate_above_sds.get_component());
        pnl_truncate_above_sds.add(&self.str_truncate_above_sds);
        // Fir pairwise shifts
        let pnl_custom_pairwise_frames = JComponent::new_panel();
        self.sp_pairwise_frames.set_maximum_width_int(50);
        // Swing layout: pnlFitPairwiseShifts BoxLayout Y_AXIS.
        self.pnl_fit_pairwise_shifts.set_border_title(
            BeveledBorder::new(Some("Fit pairwise shifts"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_fit_pairwise_shifts
            .add(&pnl_custom_pairwise_frames);
        self.pnl_fit_pairwise_shifts
            .add(&self.rb_half_pairwise_frames.get_component());
        self.pnl_fit_pairwise_shifts
            .add(&self.rb_all_pairwise_frames.get_component());
        // Swing layout: pnlCustomPairwiseFrames BoxLayout X_AXIS.
        pnl_custom_pairwise_frames.add(&self.rb_custom_pairwise_frames.get_component());
        pnl_custom_pairwise_frames.add(&self.sp_pairwise_frames.get_container());
        pnl_custom_pairwise_frames.add(&self.str_custom_pairwise_frames);
        // Binning (reduction) for alignment
        let pnl_reduce_by = JComponent::new_panel();
        let pnl_target_align_size = JComponent::new_panel();
        self.sp_reduce_by.set_maximum_width_int(50);
        self.sp_target_align_size.set_maximum_width_int(70);
        self.tf_test_binnings.set_preferred_width(300);
        // Swing layout: pnlBinningForAlignment BoxLayout Y_AXIS.
        self.pnl_binning_for_alignment.set_border_title(
            BeveledBorder::new(Some("Binning (reduction) for alignment"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_binning_for_alignment
            .add(&self.rb_binning_default.get_component());
        self.pnl_binning_for_alignment.add(&pnl_reduce_by);
        self.pnl_binning_for_alignment.add(&pnl_target_align_size);
        self.pnl_binning_for_alignment.add(&self.pnl_test_binnings);
        // Swing layout: pnlReduceBy BoxLayout X_AXIS.
        pnl_reduce_by.add(&self.rb_reduce_by.get_component());
        pnl_reduce_by.add(&self.sp_reduce_by.get_container());
        // Swing layout: pnlTargetAlignSize BoxLayout X_AXIS; rigid areas x2_y0
        // (font smaller than default) before and x20_y0 (font larger than
        // default) after the label (UIUtilities.isFontLess/GreaterThanDefaultSize).
        pnl_target_align_size.add(&self.rb_target_align_size.get_component());
        pnl_target_align_size.add(&self.sp_target_align_size.get_container());
        pnl_target_align_size.add(&self.str_target_align_size);
        // Swing layout: pnlTestBinnings BoxLayout X_AXIS.
        self.pnl_test_binnings
            .add(&self.rb_test_binnings.get_component());
        self.pnl_test_binnings
            .add(&self.tf_test_binnings.get_component());
        // Cutoffs
        self.ltf_filter_cutoffs.set_preferred_width(300);
        // Swing layout: pnlVaryFilter BoxLayout Y_AXIS.
        self.pnl_vary_filter.add(&self.pnl_filter_cutoffs);
        // Swing layout: pnlFilterCutoffs BoxLayout X_AXIS.
        self.pnl_filter_cutoffs
            .add(&self.ltf_filter_cutoffs.get_container());
        self.pnl_filter_cutoffs
            .add(&self.cb_use_hybrid_shifts.get_component());
        // Shift limit
        self.ltf_shift_limit.set_preferred_width(70);
        // Swing layout: pnlShiftLimit BoxLayout X_AXIS.
        self.pnl_shift_limit
            .add(&self.ltf_shift_limit.get_container());
        self.pnl_shift_limit.add(&self.str_shift_limit);
        // Group frames checkbox
        self.sp_group_frames.set_maximum_width_int(50);
        // Swing layout: pnlGroupFrames BoxLayout X_AXIS; horizontal glue at the
        // end.
        self.pnl_group_frames
            .add(&self.cb_group_frames.get_component());
        self.pnl_group_frames
            .add(&self.sp_group_frames.get_container());
        // Refinement
        let pnl_refine_alignment = JComponent::new_panel();
        self.sp_refine_alignment.set_maximum_width_int(50);
        self.ltf_refine_radius2.set_preferred_width(100);
        self.ltf_stop_iterations_at_shift.set_preferred_width(100);
        // Swing layout: pnlRefinement BoxLayout Y_AXIS.
        self.pnl_refinement.set_border_title(
            BeveledBorder::new(Some("Refinement"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_refinement.add(&pnl_refine_alignment);
        self.pnl_refinement.add(&self.pnl_refine_with_group_sums);
        self.pnl_refinement.add(&self.pnl_stop_iterations_at_shift);
        // Swing layout: pnlRefineAlignment BoxLayout X_AXIS.
        pnl_refine_alignment.add(&self.cb_refine_alignment.get_component());
        pnl_refine_alignment.add(&self.sp_refine_alignment.get_component());
        pnl_refine_alignment.add(&self.str_refine_alignment);
        // Swing layout: pnlRefineWithGroupSums BoxLayout X_AXIS; rigid area x30_y0
        // between the check box and the field.
        self.pnl_refine_with_group_sums
            .add(&self.cb_refine_with_group_sums.get_component());
        self.pnl_refine_with_group_sums
            .add(&self.ltf_refine_radius2.get_container());
        // Swing layout: pnlStopIterationsAtShift BoxLayout X_AXIS; rigid area
        // x2_y0 first.
        self.pnl_stop_iterations_at_shift
            .add(&self.ltf_stop_iterations_at_shift.get_container());
        self.pnl_stop_iterations_at_shift
            .add(&self.str_stop_iterations_at_shift);
        // Swing layout: UIUtilities.alignComponentsX(pnlRefinement, LEFT_ALIGNMENT).
        // Min for spline smoothing
        self.sp_min_for_spline_smoothing.set_maximum_width_int(50);
        // Swing layout: pnlMinForSplineSmoothing BoxLayout X_AXIS.
        self.pnl_min_for_spline_smoothing
            .add(&self.cb_min_for_spline_smoothing.get_component());
        self.pnl_min_for_spline_smoothing
            .add(&self.sp_min_for_spline_smoothing.get_container());
        self.pnl_min_for_spline_smoothing
            .add(&self.str_min_for_spline_smoothing);
        // Starting ending frames
        self.ltf_starting_ending_frames_first
            .set_preferred_width(100);
        self.ltf_starting_ending_frames_second
            .set_preferred_width(100);
        // Swing layout: pnlStartingEndingFrames BoxLayout X_AXIS.
        self.pnl_starting_ending_frames
            .add(&self.ltf_starting_ending_frames_first.get_container());
        self.pnl_starting_ending_frames
            .add(&self.ltf_starting_ending_frames_second.get_container());
        // Do Dose Weighting checkbox
        // Swing layout: pnlDoDoseWeighting BoxLayout X_AXIS.
        self.pnl_do_dose_weighting
            .add(&self.cb_do_dose_weighting.get_component());
        // Dose weighting
        self.tf_fixed_total_dose.set_preferred_width(100);
        self.ltf_optimal_dose_scaling.set_preferred_width(100);
        // Swing layout: pnlDoseWeighting BoxLayout Y_AXIS; etched border
        // (untitled).
        self.pnl_dose_weighting
            .add(&self.ph_dose_weighting.get_container());
        self.pnl_dose_weighting.add(&self.pnl_dose_weighting_body);
        // Swing layout: pnlDoseWeightingBody BoxLayout Y_AXIS.
        // pnlDoseWeightingBody.add(Box.createRigidArea(FixedDim.x0_y5));
        self.pnl_dose_weighting_body.add(&self.pnl_fixed_total_dose);
        self.pnl_dose_weighting_body
            .add(&self.rb_dose_weighting_file.get_component());
        self.pnl_dose_weighting_body
            .add(&self.cb_normalize_dose_weighting.get_component());
        self.pnl_dose_weighting_body
            .add(&self.cb_voltage.get_component());
        self.pnl_dose_weighting_body
            .add(&self.ltf_optimal_dose_scaling.get_container());
        self.pnl_dose_weighting_body
            .add(&self.cb_unweighted_output_file.get_component());
        // Swing layout: pnlFixedTotalDose BoxLayout X_AXIS; rigid areas x2_y0
        // before and after the unit label.
        self.pnl_fixed_total_dose
            .add(&self.rb_fixed_total_dose.get_component());
        self.pnl_fixed_total_dose
            .add(&self.tf_fixed_total_dose.get_component());
        self.pnl_fixed_total_dose
            .add(&self.str_fixed_total_dose_unit);
        // Scale out by
        self.tf_scaling_of_sum_factor.set_preferred_width(100);
        // Swing layout: pnlScalingOfSum BoxLayout X_AXIS.
        self.pnl_scaling_of_sum.add(&self.str_scaling_of_sum);
        self.pnl_scaling_of_sum
            .add(&self.rb_scaling_of_sum_default.get_component());
        self.pnl_scaling_of_sum
            .add(&self.rb_scaling_of_sum_factor.get_component());
        self.pnl_scaling_of_sum
            .add(&self.tf_scaling_of_sum_factor.get_component());
        // Output file mode
        // Swing layout: pnlModeToOutout BoxLayout X_AXIS.
        self.pnl_mode_to_outout.add(&self.str_mode_to_output);
        self.pnl_mode_to_outout
            .add(&self.rb_mode_to_output_16bit_int.get_component());
        self.pnl_mode_to_outout
            .add(&self.rb_mode_to_output_float.get_component());
        // Sum rotation and flip
        self.cmb_sum_rotation_and_flip_translation
            .set_maximum_width(180);
        self.fill_combo_box(
            Some(&self.cmb_sum_rotation_and_flip_translation),
            Some(&self.rotation_and_flip_string_list),
        );
        self.cmb_sum_rotation_and_flip_translation
            .set_selected_index(self.sp_sum_rotation_and_flip.get_value().int_value());
        self.sp_sum_rotation_and_flip.set_maximum_width_int(50);
        // Swing layout: pnlSumRotationAndFlip BoxLayout X_AXIS; rigid area x5_y0
        // before the combo box, and x20_y0 after it when the font is larger than
        // default.
        self.pnl_sum_rotation_and_flip
            .add(&self.str_sum_rotation_and_flip);
        self.pnl_sum_rotation_and_flip
            .add(&self.rb_sum_rotation_and_flip.get_component());
        self.pnl_sum_rotation_and_flip
            .add(&self.rb_sum_rotation_and_flip2.get_component());
        self.pnl_sum_rotation_and_flip
            .add(&self.sp_sum_rotation_and_flip.get_container());
        self.pnl_sum_rotation_and_flip
            .add(&self.cmb_sum_rotation_and_flip_translation.get_component());
        // Align and sum binning
        self.sp_align_and_sum_binning_value2
            .set_maximum_width_int(50);
        // Swing layout: pnlAlignAndSumBinning BoxLayout X_AXIS.
        self.pnl_align_and_sum_binning
            .add(&self.sp_align_and_sum_binning_value2.get_container());
        // Output image file
        self.ltf_output_image_file.set_preferred_width(300);
        // Swing layout: pnlOutputImageFile BoxLayout X_AXIS.
        self.pnl_output_image_file
            .add(&self.ltf_output_image_file.get_container());
        // Use GPU
        // Swing layout: pnlUseGPU BoxLayout X_AXIS.
        self.pnl_use_gpu.add(&self.cb_use_gpu.get_component());
        // Run align frame buttons
        // Swing layout: pnlRunAlignFramesButtons FlowLayout CENTER; rigid areas
        // x5_y0 between the buttons.
        self.pnl_run_align_frames_buttons
            .add(&self.btn_run_align_frames.get_component());
        self.pnl_run_align_frames_buttons
            .add(&self.btn_plot_all_results.get_component());
        self.pnl_run_align_frames_buttons
            .add(&self.btn_open_output_tilt_series.get_component());
        // Swing layout: pnlStartReconButton FlowLayout CENTER; rigid areas
        // x175_y0 before the button and x70_y0 between the buttons.
        self.pnl_start_recon_button
            .add(&self.btn_setup_reconstruction.get_component());
        self.pnl_start_recon_button
            .add(&self.btn_advanced.get_component());

        self.pnl_outer_panel.set_maximum_size(Some(&mut Dimension {
            width: 900,
            height: 10000,
        }));
        // Swing layout: pnlOuterPanel.add(Box.createHorizontalGlue()); the
        // UIUtilities.alignAllComponentsX(..., LEFT_ALIGNMENT) calls on pnlRoot,
        // pnlOuterPanel, tabPane and the two tab bodies.

        // Display
        self.cur_tab.set(Tab::get_instance(
            self.tab_pane.get_component().get_selected_tab(),
        ));
        if let Some(cur_tab) = self.cur_tab.get() {
            let pnl_tab = self.pnl_tab_array.borrow()[cur_tab.index as usize].clone();
            pnl_tab.add(&tab_body[cur_tab.index as usize]);
        }
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
        ui_harness::with(|harness| harness.move_sub_frame());
        self.update_display();
        self.update_advanced(self.is_advanced());
    }

    /// Java private `getPnlInputFileSpecification()`.
    fn get_pnl_input_file_specification(&self) -> Rc<JComponent> {
        let pnl_outer_panel = JComponent::new_panel();
        let pnl_input_file_specification = Panel::new();
        let pnl_input_file_spec_left = JComponent::new_panel();
        let pnl_metadata_file = JComponent::new_panel();
        let pnl_list_of_input_files = JComponent::new_panel();
        let pnl_selected_files = JComponent::new_panel();
        let pnl_directory = JComponent::new_panel();
        // Swing layout: FlowLayout LEADING.
        self.text_area_input_files
            .get_component()
            .set_editable(false);
        // Swing layout: scrollPaneInputFiles VERTICAL_SCROLLBAR_ALWAYS,
        // HORIZONTAL_SCROLLBAR_AS_NEEDED, setBounds(0, 0, 10, 20).

        // Swing layout: pnlOuterPanel BoxLayout X_AXIS; horizontal glue after the
        // specification panel.
        pnl_outer_panel.add(&pnl_input_file_specification.get_component());

        self.bctf_metadata_file.set_preferred_width(167);
        self.bctf_list_of_input_files.set_preferred_width(150);
        self.ltf_directory.set_preferred_width(295);

        // Input file specification
        // Swing layout: pnlInputFileSpecification FlowLayout LEADING.
        pnl_input_file_specification
            .get_component()
            .set_border_title(
                BeveledBorder::new(Some("Input file specification"))
                    .get_border()
                    .get_title()
                    .as_deref(),
            );
        pnl_input_file_specification
            .get_component()
            .add(&pnl_input_file_spec_left);
        pnl_input_file_specification
            .get_component()
            .add(&self.pnl_input_file_spec_right);
        // Input File Spec Left
        // Swing layout: pnlInputFileSpecLeft BoxLayout Y_AXIS.
        pnl_input_file_spec_left.add(&pnl_metadata_file);
        pnl_input_file_spec_left.add(&pnl_list_of_input_files);
        pnl_input_file_spec_left.add(&pnl_selected_files);
        pnl_input_file_spec_left.add(&pnl_directory);
        // Input File Spec Right
        // Swing layout: pnlInputFileSpecRight BoxLayout Y_AXIS.
        self.pnl_input_file_spec_right
            .add(&JComponent::new_label(INPUT_FILES_LABEL));
        // pnlInputFileSpecRight.add(Box.createRigidArea(FixedDim.x0_y2));
        self.pnl_input_file_spec_right
            .add(&self.scroll_pane_input_files);
        // Metadata file
        // Swing layout: pnlMetadataFile FlowLayout LEADING; horizontal glue at
        // the end.
        pnl_metadata_file.add(&self.rb_metadata_file.get_component());
        pnl_metadata_file.add(&self.bctf_metadata_file.get_component());
        // List of input files
        // Swing layout: pnlListOfInputFiles FlowLayout LEADING; horizontal glue at
        // the end.
        pnl_list_of_input_files.add(&self.rb_list_of_input_files.get_component());
        pnl_list_of_input_files.add(&self.bctf_list_of_input_files.get_component());
        // Selected files
        // Swing layout: pnlSelectedFiles FlowLayout LEADING; horizontal glue at the
        // end.
        pnl_selected_files.add(&self.rb_selected_files.get_component());
        pnl_selected_files.add(&self.btn_selected_files.get_component());
        // Directory
        // Swing layout: pnlDirectory FlowLayout LEADING; rigid area x5_y0 first.
        pnl_directory.add(&self.ltf_directory.get_component());
        pnl_input_file_specification.set_maximum_size(Some(&mut Dimension {
            width: 800,
            height: 250,
        }));
        pnl_outer_panel
    }

    /// Java private `getPnlOtherSourceOfMetadata()`.
    fn get_pnl_other_source_of_metadata(&self) -> Rc<JComponent> {
        let pnl_other_source_of_metadata = JComponent::new_panel();
        let pnl_corresponding_stack = Panel::new();
        let pnl_tilt_angle_file = Panel::new();
        let pnl_angles_in_filenames = Panel::new();
        let pnl_delimiter_open_close = Panel::new();
        let pnl_axis_rotation_angle = Panel::new();
        // Swing layout: FlowLayout LEADING.

        // init
        self.bctf_corresponding_stack.set_preferred_width(250);
        self.bctf_tilt_angle_file.set_preferred_width(250);
        self.ltf_axis_rotation_angle.set_preferred_width(350);
        self.ltf_delimiters_open.set_preferred_width(100);
        self.ltf_delimiters_close.set_preferred_width(100);

        // pnlOtherSourceOfMetadata
        // Swing layout: BoxLayout Y_AXIS; etched border (untitled).
        pnl_other_source_of_metadata.add(&self.ph_other_source_of_metadata.get_component());
        pnl_other_source_of_metadata.add(&self.pnl_other_source_of_metadata_body.get_component());

        // Other source of metadata body
        // Swing layout: BoxLayout Y_AXIS.
        let body = self.pnl_other_source_of_metadata_body.get_component();
        body.add(&pnl_corresponding_stack.get_component());
        body.add(&pnl_tilt_angle_file.get_component());
        body.add(&pnl_angles_in_filenames.get_component());
        body.add(&pnl_delimiter_open_close.get_component());
        body.add(&pnl_axis_rotation_angle.get_component());
        // Corresponding stack
        // Swing layout: FlowLayout LEADING; horizontal glue at the end.
        pnl_corresponding_stack
            .get_component()
            .add(&self.cb_corresponding_stack.get_component());
        pnl_corresponding_stack
            .get_component()
            .add(&self.bctf_corresponding_stack.get_component());
        pnl_corresponding_stack.set_maximum_size(Some(&mut Dimension {
            width: 850,
            height: 30,
        }));
        // Tilt angle file
        // Swing layout: FlowLayout LEADING; horizontal glue at the end.
        pnl_tilt_angle_file
            .get_component()
            .add(&self.cb_tilt_angle_file.get_component());
        pnl_tilt_angle_file
            .get_component()
            .add(&self.bctf_tilt_angle_file.get_component());
        pnl_tilt_angle_file.set_maximum_size(Some(&mut Dimension {
            width: 850,
            height: 30,
        }));
        // Angles in filenames
        // Swing layout: FlowLayout LEADING; horizontal glue at the end.
        pnl_angles_in_filenames
            .get_component()
            .add(&self.cb_angles_in_filenames.get_component());
        pnl_angles_in_filenames.set_maximum_size(Some(&mut Dimension {
            width: 850,
            height: 30,
        }));
        // Delimiter open-close
        // Swing layout: FlowLayout LEADING; horizontal glue at the end.
        pnl_delimiter_open_close
            .get_component()
            .add(&self.ltf_delimiters_open.get_container());
        pnl_delimiter_open_close
            .get_component()
            .add(&self.ltf_delimiters_close.get_container());
        pnl_delimiter_open_close.set_maximum_size(Some(&mut Dimension {
            width: 850,
            height: 30,
        }));
        // Axis rotation angle
        // Swing layout: FlowLayout LEADING; horizontal glue at the end.
        pnl_axis_rotation_angle
            .get_component()
            .add(&self.ltf_axis_rotation_angle.get_container());
        pnl_axis_rotation_angle.set_maximum_size(Some(&mut Dimension {
            width: 850,
            height: 30,
        }));

        self.pnl_other_source_of_metadata_body
            .set_maximum_size(Some(&mut Dimension {
                width: 850,
                height: 180,
            }));

        pnl_other_source_of_metadata
    }

    /// Java private `getPnlGainReference()`.
    fn get_pnl_gain_reference(&self) -> Rc<JComponent> {
        let pnl_gain_reference = Panel::new();
        let pnl_ref_and_defect = JComponent::new_panel();
        // Swing layout: FlowLayout LEADING.

        // Swing layout: pnlOuterGainReference BoxLayout X_AXIS; horizontal glue
        // after the gain reference panel.
        self.pnl_outer_gain_reference
            .add(&pnl_gain_reference.get_component());
        self.sp_rotation_and_flip.set_maximum_width_int(50);
        self.bctf_gain_reference_file.set_preferred_width(307);
        self.fill_combo_box(
            Some(&self.cmb_rotation_and_flip_translation),
            Some(&self.rotation_and_flip_string_list),
        );
        self.cmb_rotation_and_flip_translation
            .set_selected_index(self.sp_rotation_and_flip.get_value().int_value());
        // Gain reference
        // Swing layout: pnlGainReference BoxLayout Y_AXIS.
        pnl_gain_reference.get_component().add(&pnl_ref_and_defect);
        pnl_gain_reference
            .get_component()
            .add(&self.pnl_gain_reference_file.get_component());
        pnl_gain_reference
            .get_component()
            .add(&self.pnl_rotation_and_flip.get_component());
        // Reference and defect from titles
        // Swing layout: pnlRefAndDefect FlowLayout LEADING.
        pnl_ref_and_defect.add(&self.cb_ref_and_defect_from_titles.get_component());
        // Gain reference file
        // Swing layout: pnlGainReferenceFile FlowLayout LEADING.
        self.pnl_gain_reference_file
            .get_component()
            .add(&self.bctf_gain_reference_file.get_component());
        self.pnl_gain_reference_file
            .set_maximum_size(Some(&mut Dimension {
                width: 900,
                height: 30,
            }));
        // Rotation and flip
        // Swing layout: pnlRotationAndFlip FlowLayout LEADING.
        let pnl_rotation_and_flip = self.pnl_rotation_and_flip.get_component();
        pnl_rotation_and_flip.add(&self.str_rotation_and_flip);
        pnl_rotation_and_flip.add(&self.rb_rotation_and_flip.get_component());
        pnl_rotation_and_flip.add(&self.rb_rotation_and_flip2.get_component());
        pnl_rotation_and_flip.add(&self.sp_rotation_and_flip.get_component());
        pnl_rotation_and_flip.add(&self.cmb_rotation_and_flip_translation.get_component());
        self.pnl_rotation_and_flip
            .set_maximum_size(Some(&mut Dimension {
                width: 900,
                height: 30,
            }));
        self.pnl_outer_gain_reference.clone()
    }

    /// Java private `getPnlPathToFramesInMdoc()`.
    fn get_pnl_path_to_frames_in_mdoc(&self) -> Rc<JComponent> {
        // Swing layout: FlowLayout LEADING.

        // Swing layout: pnlOuterPathToFramesInMdoc BoxLayout X_AXIS; horizontal
        // glue after the panel.
        self.pnl_outer_path_to_frames_in_mdoc
            .add(&self.pnl_path_to_frames_in_mdoc.get_component());

        // Path to frames in mdoc
        self.bctf_path_to_frames_in_mdoc.set_preferred_width(245);
        // Swing layout: pnlPathToFramesInMdoc FlowLayout LEADING.
        self.pnl_path_to_frames_in_mdoc
            .get_component()
            .add(&self.bctf_path_to_frames_in_mdoc.get_component());
        self.pnl_path_to_frames_in_mdoc
            .set_maximum_size(Some(&mut Dimension {
                width: 850,
                height: 30,
            }));

        self.pnl_outer_path_to_frames_in_mdoc.clone()
    }

    /// Java private `getPnlCameraDefectFile()`.
    fn get_pnl_camera_defect_file(&self) -> Rc<JComponent> {
        // Swing layout: FlowLayout LEADING.

        // Swing layout: pnlOuterCameraDefectFile BoxLayout X_AXIS; horizontal glue
        // after the panel.
        self.pnl_outer_camera_defect_file
            .add(&self.pnl_camera_defect_file.get_component());
        // Camera defect file
        self.bctf_camera_defect_file.set_preferred_width(370);
        // Swing layout: pnlCameraDefectFile FlowLayout LEADING.
        self.pnl_camera_defect_file
            .get_component()
            .add(&self.bctf_camera_defect_file.get_component());
        self.pnl_camera_defect_file
            .set_maximum_size(Some(&mut Dimension {
                width: 900,
                height: 30,
            }));

        self.pnl_outer_camera_defect_file.clone()
    }

    /// Java package-private `addListeners()`.
    fn add_listeners(&self) {
        let context_menu: Weak<dyn ContextMenu> = self.self_ref.clone();
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.tab_pane
            .get_component()
            .add_mouse_listener(mouse_adapter);
        // Java `tabPane.addChangeListener(new TabChangeListener(this))`.
        let tab_change_listener = TabChangeListener::new(self.self_ref.clone());
        self.tab_pane.get_component().add_change_listener(Rc::new(
            move |change_event: &ChangeEvent| tab_change_listener.state_changed(change_event),
        ));
        let action_listener = || self.action_listener.clone();
        let control_listener: Option<Rc<dyn ControlListener>> = self
            .self_ref
            .upgrade()
            .map(|this| this as Rc<dyn ControlListener>);
        self.btn_load_starting_com_file
            .add_action_listener(action_listener());
        self.cb_do_dose_weighting
            .add_action_listener(Some(action_listener()));
        self.rb_metadata_file.add_action_listener(action_listener());
        self.bctf_metadata_file
            .add_control_listener(control_listener.clone());
        self.rb_list_of_input_files
            .add_action_listener(action_listener());
        self.bctf_list_of_input_files
            .add_control_listener(control_listener.clone());
        self.rb_selected_files
            .add_action_listener(action_listener());
        let focus_listener: FocusListener = {
            let adaptee = self.self_ref.clone();
            Rc::new(move |event: &FocusEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    if event.gained {
                        adaptee.focus_gained();
                    } else {
                        adaptee.focus_lost();
                    }
                }
            })
        };
        self.ltf_rootname_output_files
            .add_focus_listener(focus_listener.clone());
        self.cb_corresponding_stack
            .add_action_listener(Some(action_listener()));
        self.cb_tilt_angle_file
            .add_action_listener(Some(action_listener()));
        self.cb_angles_in_filenames
            .add_action_listener(Some(action_listener()));
        self.cb_ref_and_defect_from_titles
            .add_action_listener(Some(action_listener()));
        self.bctf_gain_reference_file
            .add_control_listener(control_listener);
        self.rb_rotation_and_flip
            .add_action_listener(action_listener());
        self.rb_rotation_and_flip2
            .add_action_listener(action_listener());
        self.sp_rotation_and_flip
            .add_change_listener_change_listener(Some(self.change_listener.clone()));
        self.cmb_rotation_and_flip_translation
            .add_action_listener(action_listener());
        self.rb_truncate_above_none
            .add_action_listener(action_listener());
        self.rb_truncate_above_input_counts
            .add_action_listener(action_listener());
        self.rb_truncate_above_sds
            .add_action_listener(action_listener());
        self.rb_custom_pairwise_frames
            .add_action_listener(action_listener());
        self.rb_half_pairwise_frames
            .add_action_listener(action_listener());
        self.rb_all_pairwise_frames
            .add_action_listener(action_listener());
        self.rb_binning_default
            .add_action_listener(action_listener());
        self.rb_reduce_by.add_action_listener(action_listener());
        self.rb_target_align_size
            .add_action_listener(action_listener());
        self.rb_test_binnings.add_action_listener(action_listener());
        self.ltf_filter_cutoffs.add_focus_listener(focus_listener);
        self.cb_group_frames
            .add_action_listener(Some(action_listener()));
        self.cb_refine_alignment
            .add_action_listener(Some(action_listener()));
        self.cb_min_for_spline_smoothing
            .add_action_listener(Some(action_listener()));
        self.rb_fixed_total_dose
            .add_action_listener(action_listener());
        self.rb_dose_weighting_file
            .add_action_listener(action_listener());
        self.cb_normalize_dose_weighting
            .add_action_listener(Some(action_listener()));
        self.cb_unweighted_output_file
            .add_action_listener(Some(action_listener()));
        self.rb_scaling_of_sum_default
            .add_action_listener(action_listener());
        self.rb_scaling_of_sum_factor
            .add_action_listener(action_listener());
        self.rb_mode_to_output_16bit_int
            .add_action_listener(action_listener());
        self.rb_mode_to_output_float
            .add_action_listener(action_listener());
        self.rb_sum_rotation_and_flip
            .add_action_listener(action_listener());
        self.rb_sum_rotation_and_flip2
            .add_action_listener(action_listener());
        self.sp_sum_rotation_and_flip
            .add_change_listener_change_listener(Some(self.change_listener.clone()));
        self.cmb_sum_rotation_and_flip_translation
            .add_action_listener(action_listener());
        self.cb_use_gpu.add_action_listener(Some(action_listener()));
        self.btn_run_align_frames
            .add_action_listener(action_listener());
        self.btn_plot_all_results
            .add_action_listener(action_listener());
        self.btn_open_output_tilt_series
            .add_action_listener(action_listener());
        self.btn_setup_reconstruction
            .add_action_listener(action_listener());
    }

    /// Java private `action(ActionEvent)`.
    fn action_event(&self, event: &ActionEvent) {
        let command = event.get_action_command();
        // Java `command.equals(...)`: a null command matches nothing here.
        if command.is_some()
            && command
                == self
                    .cmb_rotation_and_flip_translation
                    .get_action_command()
                    .as_deref()
        {
            if self
                .cmb_rotation_and_flip_translation
                .verify_combo_box_source(event.get_source())
            {
                self.sp_rotation_and_flip
                    .set_value_int(self.cmb_rotation_and_flip_translation.get_selected_index());
            } else if self
                .cmb_sum_rotation_and_flip_translation
                .verify_combo_box_source(event.get_source())
            {
                self.sp_sum_rotation_and_flip.set_value_int(
                    self.cmb_sum_rotation_and_flip_translation
                        .get_selected_index(),
                );
            }
        } else {
            self.action_option(event.get_action_command(), None, None);
        }
    }

    /// Java private `isAdvanced()`.
    fn is_advanced(&self) -> bool {
        self.btn_advanced.is_expanded()
    }

    /// Java protected `setDoseWeightingAdvanced(boolean)`.
    fn set_dose_weighting_advanced(&self, advanced: bool) {
        self.dose_weighting_advanced.set(advanced);
        if self.pnl_dose_weighting_body.is_visible() {
            self.ltf_optimal_dose_scaling.set_visible(advanced);
            self.cb_unweighted_output_file.set_visible(advanced);
        }
    }

    /// Java private `isDoseWeightingAdvanced()`.
    fn is_dose_weighting_advanced(&self) -> bool {
        self.dose_weighting_advanced.get()
    }

    /// Java private `updateAdvanced(boolean)`.
    fn update_advanced(&self, advanced: bool) {
        self.pnl_outer_path_to_frames_in_mdoc.set_visible(advanced);
        self.pnl_gain_reference_file
            .get_component()
            .set_visible(advanced);
        self.pnl_rotation_and_flip
            .get_component()
            .set_visible(advanced);
        self.pnl_outer_camera_defect_file.set_visible(advanced);
        self.pnl_test_binnings.set_visible(advanced);
        self.pnl_shift_limit.set_visible(advanced);
        self.pnl_refine_with_group_sums.set_visible(advanced);
        self.pnl_stop_iterations_at_shift.set_visible(advanced);
        self.pnl_starting_ending_frames.set_visible(advanced);
        self.pnl_scaling_of_sum.set_visible(advanced);
        self.pnl_mode_to_outout.set_visible(advanced);
        self.btn_advanced.change_state(advanced);
        self.set_dose_weighting_advanced(self.is_dose_weighting_advanced());
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        self.bctf_metadata_file
            .set_enabled(self.rb_metadata_file.is_selected());
        self.bctf_metadata_file
            .set_editable(self.rb_metadata_file.is_selected());
        self.bctf_list_of_input_files
            .set_enabled(self.rb_list_of_input_files.is_selected());
        self.btn_selected_files
            .set_enabled(self.rb_selected_files.is_selected());
        self.bctf_path_to_frames_in_mdoc
            .set_enabled(self.rb_metadata_file.is_selected());
        self.cb_corresponding_stack
            .set_enabled(!self.rb_metadata_file.is_selected());
        self.cb_tilt_angle_file
            .set_enabled(!self.rb_metadata_file.is_selected());
        self.cb_angles_in_filenames
            .set_enabled(!self.rb_metadata_file.is_selected());
        self.ltf_axis_rotation_angle
            .set_enabled(!self.rb_metadata_file.is_selected());
        self.bctf_corresponding_stack.set_enabled(
            self.cb_corresponding_stack.is_enabled() && self.cb_corresponding_stack.is_selected(),
        );
        self.bctf_tilt_angle_file.set_enabled(
            self.cb_tilt_angle_file.is_enabled() && self.cb_tilt_angle_file.is_selected(),
        );
        self.ltf_delimiters_open.set_enabled(
            self.cb_angles_in_filenames.is_enabled() && self.cb_angles_in_filenames.is_selected(),
        );
        self.ltf_delimiters_close.set_enabled(
            self.cb_angles_in_filenames.is_enabled() && self.cb_angles_in_filenames.is_selected(),
        );
        self.str_rotation_and_flip.set_enabled(
            self.cb_ref_and_defect_from_titles.is_selected()
                || self.bctf_gain_reference_file.get_file().is_some(),
        );
        self.rb_rotation_and_flip.set_enabled(
            self.cb_ref_and_defect_from_titles.is_selected()
                || self.bctf_gain_reference_file.get_file().is_some(),
        );
        self.rb_rotation_and_flip2.set_enabled(
            self.cb_ref_and_defect_from_titles.is_selected()
                || self.bctf_gain_reference_file.get_file().is_some(),
        );
        self.sp_rotation_and_flip.set_enabled(
            (self.cb_ref_and_defect_from_titles.is_selected()
                || self.bctf_gain_reference_file.get_file().is_some())
                && self.rb_rotation_and_flip2.is_selected(),
        );
        self.cmb_rotation_and_flip_translation.set_enabled(
            (self.cb_ref_and_defect_from_titles.is_selected()
                || self.bctf_gain_reference_file.get_file().is_some())
                && self.rb_rotation_and_flip2.is_selected(),
        );
        self.tf_truncate_above_input_counts
            .set_enabled(self.rb_truncate_above_input_counts.is_selected());
        self.tf_truncate_above_sds
            .set_enabled(self.rb_truncate_above_sds.is_selected());
        self.sp_pairwise_frames
            .set_enabled(self.rb_custom_pairwise_frames.is_selected());
        self.sp_reduce_by
            .set_enabled(self.rb_reduce_by.is_selected());
        self.sp_target_align_size
            .set_enabled(self.rb_target_align_size.is_selected());
        self.tf_test_binnings
            .set_enabled(self.rb_test_binnings.is_selected());
        self.cb_use_hybrid_shifts.set_enabled(
            !self.rb_all_pairwise_frames.is_selected() && self.is_multiple_filter_cutoffs(),
        );
        self.sp_group_frames
            .set_enabled(self.cb_group_frames.is_selected());
        self.sp_refine_alignment
            .set_enabled(self.cb_refine_alignment.is_selected());
        self.str_refine_alignment
            .set_enabled(self.cb_refine_alignment.is_selected());
        self.cb_refine_with_group_sums.set_enabled(
            self.cb_group_frames.is_selected() && self.cb_refine_alignment.is_selected(),
        );
        self.ltf_refine_radius2
            .set_enabled(self.cb_refine_alignment.is_selected());
        self.ltf_stop_iterations_at_shift
            .set_enabled(self.cb_refine_alignment.is_selected());
        self.str_stop_iterations_at_shift
            .set_enabled(self.cb_refine_alignment.is_selected());
        self.sp_min_for_spline_smoothing
            .set_enabled(self.cb_min_for_spline_smoothing.is_selected());
        if self.rb_metadata_file.is_selected() {
            self.cb_do_dose_weighting.set_enabled(true);
        } else if self.cb_tilt_angle_file.is_selected() || self.cb_angles_in_filenames.is_selected()
        {
            self.cb_do_dose_weighting.set_enabled(true);
        } else {
            self.cb_do_dose_weighting.set_enabled(false);
        }
        self.rb_fixed_total_dose.set_enabled(
            self.cb_do_dose_weighting.is_enabled() && self.cb_do_dose_weighting.is_selected(),
        );
        self.tf_fixed_total_dose.set_enabled(
            self.rb_fixed_total_dose.is_enabled() && self.rb_fixed_total_dose.is_selected(),
        );
        self.str_fixed_total_dose_unit.set_enabled(
            self.rb_fixed_total_dose.is_enabled() && self.rb_fixed_total_dose.is_selected(),
        );
        self.rb_dose_weighting_file.set_enabled(
            self.rb_metadata_file.is_selected()
                && self.cb_do_dose_weighting.is_enabled()
                && self.cb_do_dose_weighting.is_selected(),
        );
        self.cb_normalize_dose_weighting.set_enabled(
            self.cb_do_dose_weighting.is_enabled() && self.cb_do_dose_weighting.is_selected(),
        );
        self.cb_voltage.set_enabled(
            self.cb_do_dose_weighting.is_enabled() && self.cb_do_dose_weighting.is_selected(),
        );
        self.ltf_optimal_dose_scaling.set_enabled(
            self.cb_do_dose_weighting.is_enabled() && self.cb_do_dose_weighting.is_selected(),
        );
        self.cb_unweighted_output_file.set_enabled(
            self.cb_do_dose_weighting.is_enabled()
                && self.cb_do_dose_weighting.is_selected()
                && !self.cb_normalize_dose_weighting.is_selected(),
        );
        self.tf_scaling_of_sum_factor
            .set_enabled(self.rb_scaling_of_sum_factor.is_selected());
        self.sp_sum_rotation_and_flip
            .set_enabled(self.rb_sum_rotation_and_flip2.is_selected());
        self.cmb_sum_rotation_and_flip_translation
            .set_enabled(self.rb_sum_rotation_and_flip2.is_selected());
    }

    /// Java private `changeTab(int)`.
    fn change_tab_int(&self, new_tab_index: i32) {
        self.tab_pane
            .get_component()
            .set_selected_tab(new_tab_index);
        self.change_tab_void();
    }

    /// Java private `changeTab()`.
    fn change_tab_void(&self) {
        let new_tab = Tab::get_instance(self.tab_pane.get_component().get_selected_tab());
        let same = match (new_tab, self.cur_tab.get()) {
            (Some(new_tab), Some(cur_tab)) => std::ptr::eq(new_tab, cur_tab),
            (None, None) => true,
            _ => false,
        };
        if same {
            return;
        }
        let pnl_tab_array = self.pnl_tab_array.borrow().clone();
        let pnl_tab_body_array = self.pnl_tab_body_array.borrow().clone();
        if let Some(cur_tab) = self.cur_tab.get() {
            pnl_tab_array[cur_tab.index as usize]
                .remove(&pnl_tab_body_array[cur_tab.index as usize]);
        }
        self.cur_tab.set(new_tab);
        // Upstream bug fixed in translation (AlignFramesPanel.java:1300): Java
        // dereferences a null curTab (a selected index that is no tab, e.g. -1);
        // nothing is shown then.
        if let Some(cur_tab) = new_tab {
            pnl_tab_array[cur_tab.index as usize].add(&pnl_tab_body_array[cur_tab.index as usize]);
        }
        self.update_display();
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
        ui_harness::with(|harness| harness.move_sub_frame());
    }

    /// Java public `display()`: empty.
    pub fn display_void(&self) {}

    /// Java public `display(UIComponent)`.
    pub fn display_ui_component(&self, ui_component: Option<&dyn UIComponent>) {
        let Some(ui_component) = ui_component else {
            self.display_void();
            return;
        };
        // Java `uiComponent == field`: the same Swing component.
        let component = ui_component.get_component();
        let is = |field: &dyn UIComponent| Rc::ptr_eq(&component, &field.get_component());
        if is(&*self.ltf_axis_rotation_angle) {
            self.change_tab_int(Tab::INPUT_AND_PREPROCESSING.index);
        } else if is(&*self.tf_truncate_above_input_counts) {
            self.change_tab_int(Tab::INPUT_AND_PREPROCESSING.index);
        } else if is(&*self.tf_truncate_above_sds) {
            self.change_tab_int(Tab::INPUT_AND_PREPROCESSING.index);
        } else if is(&*self.tf_test_binnings) {
            self.change_tab_int(Tab::ALIGNMENT.index);
            self.update_advanced(true);
        } else if is(&*self.ltf_filter_cutoffs) {
            self.change_tab_int(Tab::ALIGNMENT.index);
        } else if is(&*self.ltf_shift_limit) {
            self.change_tab_int(Tab::ALIGNMENT.index);
            self.update_advanced(true);
        } else if is(&*self.ltf_refine_radius2) {
            self.change_tab_int(Tab::ALIGNMENT.index);
            self.update_advanced(true);
        } else if is(&*self.ltf_stop_iterations_at_shift) {
            self.change_tab_int(Tab::ALIGNMENT.index);
            self.update_advanced(true);
        } else if is(&*self.ltf_starting_ending_frames_first) {
            self.change_tab_int(Tab::ALIGNMENT.index);
            self.update_advanced(true);
        } else if is(&*self.ltf_starting_ending_frames_second) {
            self.change_tab_int(Tab::ALIGNMENT.index);
            self.update_advanced(true);
        } else if is(&*self.tf_fixed_total_dose) {
            self.change_tab_int(Tab::ALIGNMENT.index);
            self.ph_dose_weighting.set_open(true);
        } else if is(&*self.ltf_optimal_dose_scaling) {
            self.change_tab_int(Tab::ALIGNMENT.index);
            self.ph_dose_weighting.set_open(true);
            self.set_dose_weighting_advanced(true);
            self.update_advanced(true);
        } else if is(&*self.tf_scaling_of_sum_factor) {
            self.change_tab_int(Tab::ALIGNMENT.index);
            self.update_advanced(true);
        } else if is(&*self.ltf_output_image_file) {
            self.change_tab_int(Tab::ALIGNMENT.index);
        }
    }

    /// Java private `setRequiredFields()`.
    fn set_required_fields(&self) {
        self.ltf_rootname_output_files.set_required(true);
        self.bctf_corresponding_stack.set_required(true);
        self.bctf_tilt_angle_file.set_required(true);
        self.ltf_delimiters_open.set_required(true);
        self.ltf_delimiters_close.set_required(true);
        self.bctf_gain_reference_file.set_required(true);
        self.bctf_camera_defect_file.set_required(true);
        self.tf_truncate_above_input_counts.set_required(true);
        self.tf_truncate_above_sds.set_required(true);
        self.tf_test_binnings.set_required(true);
        self.tf_fixed_total_dose.set_required(true);
        self.tf_scaling_of_sum_factor.set_required(true);
    }

    /// Java private `setNumberMustBePositive()`.
    fn set_number_must_be_positive(&self) {
        self.tf_truncate_above_input_counts
            .set_number_must_be_positive(true);
        self.tf_truncate_above_sds.set_number_must_be_positive(true);
        self.tf_test_binnings.set_number_must_be_positive(true);
        self.ltf_filter_cutoffs.set_number_must_be_positive(true);
        self.ltf_shift_limit.set_number_must_be_positive(true);
        self.ltf_refine_radius2.set_number_must_be_positive(true);
        self.ltf_stop_iterations_at_shift
            .set_number_must_be_positive(true);
        self.ltf_starting_ending_frames_first
            .set_number_must_be_positive(true);
        self.ltf_starting_ending_frames_second
            .set_number_must_be_positive(true);
        self.ltf_optimal_dose_scaling
            .set_number_must_be_positive(true);
        self.tf_scaling_of_sum_factor
            .set_number_must_be_positive(true);
    }

    /// Java private `setFieldDisplayer()`.
    fn set_field_displayer(&self) {
        let this = self.field_displayer();
        self.ltf_axis_rotation_angle
            .set_overridable_field_displayers(None, this.clone());
        self.tf_truncate_above_input_counts
            .set_overridable_field_displayers(None, this.clone());
        self.tf_truncate_above_sds
            .set_overridable_field_displayers(None, this.clone());
        self.tf_test_binnings
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_filter_cutoffs
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_shift_limit
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_refine_radius2
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_stop_iterations_at_shift
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_starting_ending_frames_first
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_starting_ending_frames_second
            .set_overridable_field_displayers(None, this.clone());
        self.tf_fixed_total_dose
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_optimal_dose_scaling
            .set_overridable_field_displayers(None, this.clone());
        self.tf_scaling_of_sum_factor
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_output_image_file
            .set_overridable_field_displayers(None, this);
    }

    /// Java private `setValidationSet()`.
    fn set_validation_set(&self) {
        let mut val_set_optimal_dose_scaling =
            ValidationSet::new(Some(FieldType::FloatingPoint), None);
        val_set_optimal_dose_scaling
            .set_minimum(align_frames_param::OPTIMAL_DOSE_SCALING_VALIDATE_MIN);
        val_set_optimal_dose_scaling
            .set_maximum(align_frames_param::OPTIMAL_DOSE_SCALING_VALIDATE_MAX);
        self.ltf_optimal_dose_scaling
            .set_validation_set(Some(&val_set_optimal_dose_scaling));

        let mut val_set_scaling_of_sum_factor =
            ValidationSet::new(Some(FieldType::FloatingPoint), None);
        val_set_scaling_of_sum_factor
            .set_minimum(align_frames_param::SCALING_OF_SUM_VALIDATE_MIN as f64);
        self.tf_scaling_of_sum_factor
            .set_validation_set(Some(&val_set_scaling_of_sum_factor));
    }

    /// Java public `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: Option<&ActionEvent>) {
        if let Some(event) = event {
            self.action_event(event);
        }
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        // Java passes a null AxisID; ALIGN_FRAMES is not a per-axis autodoc, so
        // `AxisID::Only` stands in for it.
        // SAFETY: the factory keeps every autodoc it returns (and its sections)
        // for the life of the process.
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::ALIGN_FRAMES),
                AxisID::Only,
                false,
            )
        } {
            Ok(instance) => autodoc = instance as *const Autodoc,
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except):
            // except.printStackTrace().
            Err(except) => eprintln!("{except}"),
        }
        if autodoc.is_null() {
            return;
        }
        // SAFETY: see above.
        let autodoc: &dyn ReadOnlyAutodoc = unsafe { &*autodoc };
        let autodoc_name = autodoc.get_autodoc_name();
        // Java `EtomoAutodoc.getTooltip(autodoc, fieldName)`.
        let tooltip =
            |field_name: &str| etomo_autodoc::get_tooltip(Some(autodoc), Some(field_name));
        // Java `autodoc.getSection(EtomoAutodoc.FIELD_SECTION_NAME, name)`.
        // SAFETY: see above.
        let get_section = |name: &str| unsafe {
            autodoc.get_section(Some(etomo_autodoc::FIELD_SECTION_NAME), Some(name))
        };
        // Java `EtomoAutodoc.getTooltip(String, ReadOnlySection, String)`: a null
        // section is caught (NullPointerException) and answers null.
        let enum_tooltip =
            |section: *const dyn ReadOnlySection, enum_value_name: &str| -> Option<String> {
                if section.is_null() {
                    return None;
                }
                // SAFETY: see above.
                let section: &dyn ReadOnlySection = unsafe { &*section };
                etomo_autodoc::get_tooltip_enum_value_name(
                    Some(&autodoc_name),
                    section,
                    Some(enum_value_name),
                )
            };
        self.btn_load_starting_com_file.set_tool_tip_text(Some(
            "Read an existing alignframes .com or .pcm file and use options not specific to \
             the frame file to initialize values here",
        ));
        self.rb_metadata_file.set_tool_tip_text_string(Some(
            "Read frame filenames and other metadata from a SerialEM .mdoc file",
        ));
        self.bctf_metadata_file
            .set_tooltip_string(tooltip("MetadataFile").as_deref());
        self.rb_list_of_input_files.set_tool_tip_text_string(Some(
            "Use a text file with a list of the frame files to process, one per line",
        ));
        self.bctf_list_of_input_files
            .set_tooltip_string(tooltip("ListOfInputFiles").as_deref());
        self.rb_selected_files.set_tool_tip_text_string(Some(
            "Enter all the frame files to process through a file chooser",
        ));
        self.ltf_directory.set_tool_tip_text(Some(
            "Working directory where the output files will be made, based on location of the \
             metadata file, list file, or selected frame files",
        ));
        self.ltf_rootname_output_files
            .set_tool_tip_text(Some("The base name for all of the output files"));
        self.bctf_path_to_frames_in_mdoc
            .set_tooltip_string(tooltip("PathToFramesInMdoc").as_deref());
        self.cb_corresponding_stack
            .set_tool_tip_text_string(tooltip("CorrespondingStack").as_deref());
        self.bctf_corresponding_stack
            .set_tooltip_string(tooltip("CorrespondingStack").as_deref());
        self.cb_tilt_angle_file
            .set_tool_tip_text_string(tooltip("TiltAngleFile").as_deref());
        self.bctf_tilt_angle_file
            .set_tooltip_string(tooltip("TiltAngleFile").as_deref());
        self.cb_angles_in_filenames.set_tool_tip_text_string(Some(
            "Extract tilt angles from the filenames, looking for numbers between the starting \
             and ending delimiting characters",
        ));
        self.ltf_delimiters_open
            .set_tool_tip_text(Some("Character just before tilt angle in filenames"));
        self.ltf_delimiters_close
            .set_tool_tip_text(Some("Character just after tilt angle in filenames"));
        self.ltf_axis_rotation_angle
            .set_tool_tip_text(tooltip("AxisRotationAngle").as_deref());
        self.cb_ref_and_defect_from_titles
            .set_tool_tip_text_string(tooltip("RefAndDefectFromTitles").as_deref());
        self.bctf_gain_reference_file
            .set_tooltip_string(tooltip("GainReferenceFile").as_deref());
        self.rb_rotation_and_flip.set_tool_tip_text_string(Some(
            "Use the r/f value in the frame file header to determine the rotation and flip \
             operation that needs to be applied to the gain reference",
        ));
        self.rb_rotation_and_flip2
            .set_tool_tip_text_string(tooltip("RotationAndFlip").as_deref());
        self.sp_rotation_and_flip
            .set_tool_tip_text(tooltip("RotationAndFlip").as_deref());
        self.cmb_rotation_and_flip_translation
            .set_tool_tip_text(tooltip("RotationAndFlip").as_deref());
        self.bctf_camera_defect_file
            .set_tooltip_string(tooltip("CameraDefectFile").as_deref());
        let section_truncate_above = get_section("TruncateAbove");
        self.rb_truncate_above_none
            .set_tool_tip_text_string(enum_tooltip(section_truncate_above, "none").as_deref());
        self.rb_truncate_above_input_counts
            .set_tool_tip_text_string(enum_tooltip(section_truncate_above, "pos").as_deref());
        self.tf_truncate_above_input_counts
            .set_tool_tip_text(enum_tooltip(section_truncate_above, "pos").as_deref());
        self.rb_truncate_above_sds
            .set_tool_tip_text_string(enum_tooltip(section_truncate_above, "neg").as_deref());
        self.tf_truncate_above_sds
            .set_tool_tip_text(enum_tooltip(section_truncate_above, "neg").as_deref());
        let section_pairwise_frames = get_section("PairwiseFrames");
        self.rb_custom_pairwise_frames
            .set_tool_tip_text_string(enum_tooltip(section_pairwise_frames, "custom").as_deref());
        self.sp_pairwise_frames
            .set_tool_tip_text(enum_tooltip(section_pairwise_frames, "num").as_deref());
        self.rb_half_pairwise_frames
            .set_tool_tip_text_string(enum_tooltip(section_pairwise_frames, "half").as_deref());
        self.rb_all_pairwise_frames
            .set_tool_tip_text_string(enum_tooltip(section_pairwise_frames, "all").as_deref());
        let section_align_and_sum_binning = get_section("AlignAndSumBinning");
        self.rb_binning_default.set_tool_tip_text_string(
            enum_tooltip(section_align_and_sum_binning, "default").as_deref(),
        );
        self.rb_reduce_by.set_tool_tip_text_string(
            enum_tooltip(section_align_and_sum_binning, "set").as_deref(),
        );
        self.sp_reduce_by
            .set_tool_tip_text(enum_tooltip(section_align_and_sum_binning, "num").as_deref());
        self.rb_target_align_size.set_tool_tip_text_string(Some(
            "Specify a target size to which images will be reduced for aligning",
        ));
        self.sp_target_align_size
            .set_tool_tip_text(tooltip("TargetAlignSize").as_deref());
        self.rb_test_binnings.set_tool_tip_text_string(Some(
            "Test a set of reduction values to determine which is best",
        ));
        self.tf_test_binnings
            .set_tool_tip_text(tooltip("TestBinnings").as_deref());
        self.ltf_filter_cutoffs
            .set_tool_tip_text(tooltip("VaryFilter").as_deref());
        self.cb_use_hybrid_shifts
            .set_tool_tip_text_string(tooltip("UseHybridShifts").as_deref());
        self.ltf_shift_limit
            .set_tool_tip_text(tooltip("ShiftLimit").as_deref());
        self.cb_group_frames.set_tool_tip_text_string(Some(
            "Sum consecutive frames to reduce noise for the pairwise alignments; shifts will \
             generally still be derived for each individual frame",
        ));
        self.sp_group_frames
            .set_tool_tip_text(tooltip("GroupSize").as_deref());
        self.cb_refine_alignment.set_tool_tip_text_string(Some(
            "Refine alignment based on pairwise shifts by aligning each frame to the aligned \
             sum of the rest",
        ));
        self.sp_refine_alignment
            .set_tool_tip_text(tooltip("RefineAlignment").as_deref());
        self.cb_refine_with_group_sums
            .set_tool_tip_text_string(tooltip("RefineWithGroupSums").as_deref());
        self.ltf_refine_radius2
            .set_tool_tip_text(tooltip("RefineRadius2").as_deref());
        self.ltf_stop_iterations_at_shift
            .set_tool_tip_text(tooltip("StopIterationsAtShift").as_deref());
        self.cb_min_for_spline_smoothing
            .set_tool_tip_text_string(Some(
                "Smooth the shifts with a spline curve if number of frames is high enough; do not \
             use with less than 15 frames",
            ));
        self.sp_min_for_spline_smoothing
            .set_tool_tip_text(tooltip("MinForSplineSmoothing").as_deref());
        self.ltf_starting_ending_frames_first
            .set_tool_tip_text(tooltip("StartingEndingFrames").as_deref());
        self.ltf_starting_ending_frames_second
            .set_tool_tip_text(tooltip("StartingEndingFrames").as_deref());
        self.cb_do_dose_weighting.set_tool_tip_text_string(Some(
            "Filter out high frequencies as a function of dose to the specimen",
        ));
        self.rb_fixed_total_dose.set_tool_tip_text_string(Some(
            "Apply dose weighting based on the entered value for the dose applied during each \
             set of frames (tilt image)",
        ));
        self.tf_fixed_total_dose
            .set_tool_tip_text(tooltip("FixedTotalDose").as_deref());
        self.rb_dose_weighting_file.set_tool_tip_text_string(Some(
            "Apply dose weighting based on the the dose values in the .mdoc file",
        ));
        self.cb_normalize_dose_weighting
            .set_tool_tip_text_string(tooltip("NormalizeDoseWeighting").as_deref());
        self.cb_voltage
            .set_tool_tip_text_string(tooltip("Voltage").as_deref());
        self.ltf_optimal_dose_scaling
            .set_tool_tip_text(tooltip("OptimalDoseScaling").as_deref());
        self.cb_unweighted_output_file
            .set_tool_tip_text_string(tooltip("UnweightedOutputFile").as_deref());
        let section_scaling_of_sum = get_section("ScalingOfSum");
        self.rb_scaling_of_sum_default
            .set_tool_tip_text_string(enum_tooltip(section_scaling_of_sum, "def").as_deref());
        self.rb_scaling_of_sum_factor
            .set_tool_tip_text_string(enum_tooltip(section_scaling_of_sum, "fac").as_deref());
        self.tf_scaling_of_sum_factor
            .set_tool_tip_text(enum_tooltip(section_scaling_of_sum, "val").as_deref());
        self.rb_mode_to_output_16bit_int
            .set_tool_tip_text_string(tooltip("ModeToOutput").as_deref());
        self.rb_mode_to_output_float
            .set_tool_tip_text_string(tooltip("ModeToOutput").as_deref());
        self.rb_sum_rotation_and_flip.set_tool_tip_text_string(Some(
            "Use the \"need\" value in the frame file header, if any, to determine the rotation \
             and flip operation to apply to the output sums",
        ));
        self.rb_sum_rotation_and_flip2
            .set_tool_tip_text_string(tooltip("SumRotationAndFlip").as_deref());
        self.sp_sum_rotation_and_flip
            .set_tool_tip_text(tooltip("SumRotationAndFlip").as_deref());
        self.cmb_sum_rotation_and_flip_translation
            .set_tool_tip_text(tooltip("SumRotationAndFlip").as_deref());
        self.sp_align_and_sum_binning_value2
            .set_tool_tip_text(enum_tooltip(section_align_and_sum_binning, "output").as_deref());
        self.ltf_output_image_file
            .set_tool_tip_text(tooltip("OutputImageFile").as_deref());
        self.cb_use_gpu
            .set_tool_tip_text_string(tooltip("UseGPU").as_deref());
        self.btn_run_align_frames
            .set_tool_tip_text(Some("Run Aligframes with all of the current parameters"));
        self.btn_plot_all_results.set_tool_tip_text(Some(
            "Open graphs for shift distances, mean weighted residuals, and maximum of the \
             maximum residual from the fits",
        ));
        self.btn_open_output_tilt_series.set_tool_tip_text(Some(
            "Open stack of aligned sums; right click to open binned by 2 or with startup window",
        ));
        self.btn_setup_reconstruction.set_tool_tip_text(Some(
            "Open interface to Setup Tomogram reconstruction, moving file to new directory if \
             desired",
        ));
    }

    /// Java public `setText(File)`.  setText is handled by control event for
    /// MetadataFile and ListOfInputFiles option.
    pub fn set_text_file(&self, file: Option<&Path>) {
        let Some(file) = file else {
            return;
        };
        let manager: &'static dyn BaseManager = self.manager;
        // Java `file.getParent()` / `file.getName()`.
        let parent = file
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned());
        let name = file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        // Java `name.split("\\.(?=[^\\.]+$)")`: split at the last '.' that is
        // followed by at least one character and no other '.'.
        let split_name = |name: &str| -> Vec<String> {
            match name.rfind('.') {
                Some(index) if index + 1 < name.len() => {
                    vec![name[..index].to_string(), name[index + 1..].to_string()]
                }
                _ => vec![name.to_string()],
            }
        };
        // `in = new BufferedReader(new FileReader(file))`.
        let reader = match std::fs::read(file) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                // catch (final FileNotFoundException e): e.printStackTrace().
                eprintln!("{e}");
                None
            }
        };
        if reader.is_some() {
            self.bctf_corresponding_stack.clear();
            self.bctf_tilt_angle_file.clear();
            self.bctf_path_to_frames_in_mdoc.clear();
            self.text_area_input_files.get_component().set_text("");
            self.ltf_directory.set_text_string(parent.as_deref());
            manager.set_property_user_dir(parent.as_deref());
            let split_base_and_extension = split_name(&name);
            if split_base_and_extension.len() > 1
                && split_base_and_extension[1] == extension::CLASS.mdoc.to_string()
            {
                let split_base_and_extension2 = split_name(&split_base_and_extension[0]);
                self.ltf_rootname_output_files
                    .set_text_string(Some(&split_base_and_extension2[0]));
                self.set_output_image_file();
                self.manager.set_rootname(
                    Field::get_text_void(&*self.ltf_rootname_output_files).as_deref(),
                );
                // in.close()
                return;
            } else {
                self.ltf_rootname_output_files
                    .set_text_string(Some(&split_base_and_extension[0]));
                self.set_output_image_file();
                self.manager.set_rootname(
                    Field::get_text_void(&*self.ltf_rootname_output_files).as_deref(),
                );
            }
        }
        // Upstream bug fixed in translation (AlignFramesPanel.java:2117): after a
        // FileNotFoundException Java reads from the null reader
        // (NullPointerException); nothing is read here.
        let Some(bytes) = reader else {
            return;
        };
        // `BufferedReader.readLine()` loop: each line without its terminator,
        // followed by "\n".
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let mut rest: &str = &text;
        while !rest.is_empty() {
            let (line, next) = match rest.find(['\n', '\r']) {
                Some(index) => {
                    let terminator_len = if rest[index..].starts_with("\r\n") {
                        2
                    } else {
                        1
                    };
                    (&rest[..index], &rest[index + terminator_len..])
                }
                None => (rest, ""),
            };
            self.text_area_input_files.get_component().append(line);
            rest = next;
            self.text_area_input_files.get_component().append("\n");
        }
    }

    /// Java public `setText(File[])`.  setText is handled by control event for
    /// SelectedFiles option.
    pub fn set_text_file_array(&self, files: Option<&[PathBuf]>) {
        // Java `files.length` on a null array throws NullPointerException; no files
        // is treated as an empty selection (upstream bug fixed in translation,
        // AlignFramesPanel.java:2178).
        let Some(files) = files else {
            return;
        };
        if files.is_empty() {
            return;
        }
        let manager: &'static dyn BaseManager = self.manager;
        let name_of = |file: &PathBuf| {
            file.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        self.bctf_corresponding_stack.clear();
        self.bctf_tilt_angle_file.clear();
        self.bctf_path_to_frames_in_mdoc.clear();
        self.text_area_input_files.get_component().set_text("");
        let parent = files[0]
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned());
        self.ltf_directory.set_text_string(parent.as_deref());
        manager.set_property_user_dir(parent.as_deref());
        for i in 0..files.len() {
            self.text_area_input_files
                .get_component()
                .append(&name_of(&files[i]));
            if i < files.len() - 1 {
                self.text_area_input_files.get_component().append("\n");
            }
        }
        let rootname_for_selected_files = self.get_rootname_for_selected_files(files);
        self.ltf_rootname_output_files
            .set_text_string(Some(&rootname_for_selected_files));
        // Java `rootnameForSelectedFiles.split("-|\\_")`: trailing empty strings
        // are dropped (an empty input gives one empty string).
        let mut split_base_and_extension: Vec<&str> =
            rootname_for_selected_files.split(['-', '_']).collect();
        if !rootname_for_selected_files.is_empty() {
            while split_base_and_extension.last() == Some(&"") {
                split_base_and_extension.pop();
            }
        }
        for i in 0..split_base_and_extension.len() {
            let segment: Vec<char> = split_base_and_extension[i].chars().collect();
            let mut count_zeros = 0;
            for j in 0..segment.len() {
                if segment[j] == '0' {
                    count_zeros += 1;
                }
            }
            if count_zeros == segment.len() {
                let mut calc_sub_str_length: i32 = 0;
                for k in 0..i {
                    calc_sub_str_length += split_base_and_extension[k].chars().count() as i32;
                }
                calc_sub_str_length += i as i32 - 1;
                // Upstream bug fixed in translation (AlignFramesPanel.java:2207): when
                // the first segment is all zeros (or empty) Java computes a length of
                // -1 and `substring(0, -1)` throws StringIndexOutOfBoundsException; the
                // root name before that segment is empty.
                let calc_sub_str_length = calc_sub_str_length.max(0) as usize;
                let rootname: String = rootname_for_selected_files
                    .chars()
                    .take(calc_sub_str_length)
                    .collect();
                self.ltf_rootname_output_files
                    .set_text_string(Some(&rootname));
                self.set_output_image_file();
                self.manager.set_rootname(
                    Field::get_text_void(&*self.ltf_rootname_output_files).as_deref(),
                );
                return;
            }
        }
        self.set_output_image_file();
        self.manager
            .set_rootname(Field::get_text_void(&*self.ltf_rootname_output_files).as_deref());
    }

    /// Java public `getRootnameForSelectedFiles(File[])`.
    pub fn get_rootname_for_selected_files(&self, files: &[PathBuf]) -> String {
        let name_of = |file: &PathBuf| {
            file.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let filename1: Vec<char> = name_of(&files[0]).chars().collect();
        let mut prev_sub_string = String::new();
        if files.len() == 1 {
            // Java `files[0].getName().split("\\.(?=[^\\.]+$)")[0]`.
            let name = name_of(&files[0]);
            prev_sub_string = match name.rfind('.') {
                Some(index) if index + 1 < name.len() => name[..index].to_string(),
                _ => name,
            };
        } else {
            for i in 0..filename1.len() {
                let curr_sub_string: String = filename1[..i].iter().collect();
                for j in 0..files.len() {
                    if !name_of(&files[j]).starts_with(&curr_sub_string) {
                        return prev_sub_string;
                    }
                }
                prev_sub_string = curr_sub_string;
            }
        }
        prev_sub_string
    }

    /// Java public `setOutputImageFile()`.
    pub fn set_output_image_file(&self) {
        let rootname = Field::get_text_void(&*self.ltf_rootname_output_files)
            .unwrap_or_else(|| "null".to_string());
        if self.cb_do_dose_weighting.is_selected()
            && !self.cb_normalize_dose_weighting.is_selected()
        {
            self.ltf_output_image_file.set_text_string(Some(&format!(
                "{rootname}{}{EXTENSION_DIVIDER}{}",
                align_frames_param::OUTPUT_IMAGE_FILE_AF_DW,
                extension::CLASS.mrc
            )));
        } else {
            self.ltf_output_image_file.set_text_string(Some(&format!(
                "{rootname}{}{EXTENSION_DIVIDER}{}",
                align_frames_param::OUTPUT_IMAGE_FILE_AF,
                extension::CLASS.mrc
            )));
        }
    }

    /// Java `focusGained(FocusEvent)`: empty.
    pub fn focus_gained(&self) {}

    /// Java `focusLost(FocusEvent)`.
    pub fn focus_lost(&self) {
        self.manager
            .set_rootname(Field::get_text_void(&*self.ltf_rootname_output_files).as_deref());
        self.set_output_image_file();

        self.update_display();
    }

    /// Java `stateChanged(ChangeEvent)`.
    pub fn state_changed(&self, event: &ChangeEvent) {
        if self
            .sp_rotation_and_flip
            .verify_spinner_source(&event.source)
        {
            self.cmb_rotation_and_flip_translation
                .set_selected_index(self.sp_rotation_and_flip.get_value().int_value());
        } else if self
            .sp_sum_rotation_and_flip
            .verify_spinner_source(&event.source)
        {
            self.cmb_sum_rotation_and_flip_translation
                .set_selected_index(self.sp_sum_rotation_and_flip.get_value().int_value());
        }
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)` with a
    /// possibly-null command (Java `actionCommand.equals(...)` throws on null; a
    /// null command matches no branch here).
    fn action_option(
        &self,
        action_command: Option<&str>,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let manager: &'static dyn BaseManager = self.manager;
        let matches = |command: Option<String>| {
            action_command.is_some() && action_command == command.as_deref()
        };
        if matches(self.cb_do_dose_weighting.get_action_command()) {
            self.ph_dose_weighting
                .set_open(self.cb_do_dose_weighting.is_selected());
            if self.cb_do_dose_weighting.is_selected() {
                self.pnl_dose_weighting_body.set_visible(true);
            }
            self.set_output_image_file();
        } else if matches(self.cb_normalize_dose_weighting.get_action_command()) {
            self.set_output_image_file();
        } else if matches(self.btn_load_starting_com_file.get_action_command()) {
            let chooser = FileChooser::new_base_manager(Some(manager));
            chooser.set_dialog_title(Some("Select a stack for each dataset"));
            chooser.set_file_filter(Some(Rc::new(ComFileFileFilter::get_instance(
                Some(manager),
                true,
            )) as Rc<dyn FileFilter>));
            // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
            // .getFileChooserDimension()).
            let return_val =
                chooser.show_open_dialog(Some(&self.btn_load_starting_com_file.get_component()));
            if return_val == file_chooser::APPROVE_OPTION {
                let com_file = chooser.get_selected_file();
                // Java passes btnLoadStartingComFile as the UIComponent; the
                // manager does not read it.
                if self
                    .manager
                    .open_align_frames_input_com_file(self, com_file.as_deref(), None)
                {
                    ui_harness::with(|harness| harness.pack_base_manager(Some(manager)));
                }
            }
        } else if matches(self.btn_run_align_frames.get_action_command()) {
            if self.rb_metadata_file.is_selected() || !self.cb_angles_in_filenames.is_selected() {
                self.manager.set_align_frames_tilt_angle_file_created(false);
                self.manager.align_frames(None, self);
            } else if let Some(this) = self.self_ref.upgrade() {
                self.manager
                    .sorttiltframes(this as Rc<dyn AlignFramesDisplay>);
            }
        } else if matches(self.btn_plot_all_results.get_action_command()) {
            self.manager.plot_all_results(self);
        } else if matches(self.btn_open_output_tilt_series.get_action_command()) {
            self.manager
                .open_output_tilt_series(run_3dmod_menu_options, self);
        } else if matches(self.btn_setup_reconstruction.get_action_command()) {
            // Pop-up filechooser (directories only) and use it for setDir
            let return_val = self.fc_local_arguments_dir.show_open_dialog(None);
            if return_val == file_chooser::APPROVE_OPTION {
                // Upstream bug fixed in translation (AlignFramesPanel.java:2331): an
                // approved chooser with no selected file makes Java throw
                // NullPointerException; nothing is opened then.
                if let Some(selected_file) = self.fc_local_arguments_dir.get_selected_file() {
                    *self.local_arguments_dir.borrow_mut() =
                        crate::imod::etomo::util::utilities::java_io_file_get_absolute_path(
                            &selected_file.to_string_lossy(),
                        );
                    self.manager.open_tomogram(self);
                }
            }
        }

        self.update_display();
    }

    /// Java public `isMultipleFilterCutoffs()`.
    pub fn is_multiple_filter_cutoffs(&self) -> bool {
        let mut split_filter_cutoffs: Vec<String> = Vec::new();
        match Field::get_text_boolean_field_displayer(
            &*self.ltf_filter_cutoffs,
            true,
            self.field_displayer(),
        ) {
            Ok(text) => {
                // Java `String.split(",")`: trailing empty strings are dropped (an
                // empty input gives one empty string).
                let text = text.unwrap_or_else(|| "null".to_string());
                split_filter_cutoffs = text.split(',').map(str::to_string).collect();
                if !text.is_empty() {
                    while split_filter_cutoffs.last().is_some_and(String::is_empty) {
                        split_filter_cutoffs.pop();
                    }
                }
            }
            // catch (FieldValidationFailedException e): e.printStackTrace().
            Err(e) => eprintln!("{e}"),
        }
        split_filter_cutoffs.len() > 1
    }
}

impl ToolPanel for AlignFramesPanel {
    /// Java public `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }
}

impl Expandable for AlignFramesPanel {
    /// Java public `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        // Java `if (phDoseWeighting != null)`: always set by the constructor.
        if self.ph_dose_weighting.equals_open_close(button) {
            self.pnl_dose_weighting_body
                .set_visible(button.is_expanded());
            self.set_dose_weighting_advanced(self.is_dose_weighting_advanced());
        } else if self.ph_dose_weighting.equals_advanced_basic(button) {
            self.set_dose_weighting_advanced(button.is_expanded());
        }
        if self.ph_other_source_of_metadata.equals_open_close(button) {
            self.pnl_other_source_of_metadata_body
                .get_component()
                .set_visible(button.is_expanded());
        }
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }

    /// Java public `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.update_advanced(button.is_expanded());
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }
}

impl ContextMenu for AlignFramesPanel {
    /// Java public `popUpContextMenu(MouseEvent)`.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let manager: &'static dyn BaseManager = self.manager;
        let label: Vec<String> = vec!["Alignframes".to_string()];
        let man_pagelabel: Vec<String> = vec!["Alignframes".to_string()];
        let man_page: Vec<String> = vec!["alignframes.html".to_string()];
        // Java declares manPagelabel and passes `label` in its place.
        let _ = man_pagelabel;

        let log_file_label: Vec<String> = vec!["Alignframes".to_string()];
        let rootname = AlignFramesDisplay::get_rootname_output_files(self);
        let property_user_dir = manager.get_property_user_dir();
        // Java `FileType.ALIGN_FRAMES_LOG.getFile(...).getName()`.
        let log_file: Vec<String> = vec![
            file_type::CLASS
                .align_frames_log
                .get_file_with_property_user_dir(
                    Some(manager),
                    Some(&rootname),
                    None,
                    None,
                    property_user_dir.as_deref(),
                )
                .and_then(|file| {
                    file.file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                })
                .unwrap_or_else(|| "null".to_string()),
        ];

        let graph: Vec<Task> = vec![
            Task::AlignFramesShifts,
            Task::AlignFramesMeanResiduals,
            Task::AlignFramesMaxofmaxResiduals,
        ];

        let directory = Field::get_text_void(&*self.ltf_directory);
        let graph_input_file: Vec<Option<PathBuf>> = (0..3)
            .map(|_| {
                file_type::CLASS
                    .align_frames_log
                    .get_file_with_property_user_dir(
                        Some(manager),
                        Some(&rootname),
                        None,
                        None,
                        directory.as_deref(),
                    )
            })
            .collect();

        if let Err(except) = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_task_array_file_array_base_manager_axis_id(
            &self.tab_pane.get_component(),
            mouse_event,
            Some("ALIGN_FRAMES"),
            Some(context_popup::ALIGNFRAMES_GUIDE),
            &label,
            &man_page,
            &log_file_label,
            &log_file,
            Some(&graph),
            Some(&graph_input_file),
            manager,
            self.axis_id,
        ) {
            // An exception thrown by the ContextPopup constructor propagates to
            // the Swing event dispatch thread, which prints it.
            eprintln!("{except}");
        }
    }
}

impl ProcessDisplay for AlignFramesPanel {
    /// Java cast `(AlignFramesDisplay) display`.
    fn as_align_frames_display(&self) -> Option<&dyn AlignFramesDisplay> {
        Some(self)
    }
}

impl AlignFramesDisplay for AlignFramesPanel {
    /// Java `getParameters(SortTiltFramesParam)`.
    fn get_parameters_sort_tilt_frames_param(
        &self,
        param: &mut SortTiltFramesParam,
    ) -> Result<bool, FieldValidationFailedException> {
        let field_displayer: &dyn FieldDisplayer = self;
        // catch (LogFileException | IOException e): e.printStackTrace();
        // catch (LockException e) {}.
        let report = |e: LogFileError| match e {
            LogFileError::Lock(_) => {}
            e => eprintln!("{e}"),
        };
        let result = (|| -> Result<(), FieldValidationFailedException> {
            // build sorttiltframes options
            if self.is_angles_in_filenames_selected() {
                if self.is_corresponding_stack_selected() || self.is_tilt_angle_file_selected() {
                    // NOT just angles in filenames
                    if self.is_corresponding_stack_selected() {
                        param.set_tilt_series_file(
                            self.bctf_corresponding_stack
                                .get_text_boolean_field_displayer(true, Some(field_displayer))?
                                .as_deref(),
                        );
                    }
                    if self.is_tilt_angle_file_selected() {
                        param.set_tilt_angle_file(
                            self.bctf_tilt_angle_file
                                .get_text_boolean_field_displayer(true, Some(field_displayer))?
                                .as_deref(),
                        );
                    }
                    let subdir =
                        PathBuf::from(self.manager.get_property_user_dir().unwrap_or_default());
                    if self.is_list_of_input_files_selected() {
                        param.set_list_of_input_files(
                            self.bctf_list_of_input_files
                                .get_text_boolean_field_displayer(true, Some(field_displayer))?
                                .as_deref(),
                        );
                    } else if self.is_selected_files_selected() {
                        match self.manager.get_list_of_input_files(self, &subdir, false) {
                            Ok(list) => param.set_list_of_input_files(Some(&list)),
                            Err(e) => report(e),
                        }
                    }
                    self.manager.set_align_frames_tilt_angle_file_created(false);
                    match self.manager.get_output_file_list(self, &subdir) {
                        Ok(list) => param.set_output_file_list(Some(&list)),
                        Err(e) => report(e),
                    }
                } else {
                    // Just angles in filenames
                    let subdir =
                        PathBuf::from(self.manager.get_property_user_dir().unwrap_or_default());
                    if self.is_list_of_input_files_selected() || self.is_selected_files_selected() {
                        match self.manager.get_list_of_input_files(self, &subdir, true) {
                            Ok(list) => param.set_list_of_input_files(Some(&list)),
                            Err(e) => report(e),
                        }
                    }
                    param.set_unsorted_output(true);
                    param.set_output_tilt_angle_file(Some(&format!(
                        "{}{}{EXTENSION_DIVIDER}{}",
                        Field::get_text_boolean_field_displayer(
                            &*self.ltf_rootname_output_files,
                            true,
                            self.field_displayer(),
                        )?
                        .unwrap_or_else(|| "null".to_string()),
                        sort_tilt_frames_param::OUTPUT_TILT_ANGLE_FILE_MATCHING,
                        extension::CLASS.tlt
                    )));
                    self.manager.set_align_frames_tilt_angle_file_created(true);
                }

                param.set_delimiters(Some(&format!(
                    "{}{}",
                    Field::get_text_boolean_field_displayer(
                        &*self.ltf_delimiters_open,
                        true,
                        self.field_displayer(),
                    )?
                    .unwrap_or_else(|| "null".to_string()),
                    Field::get_text_boolean_field_displayer(
                        &*self.ltf_delimiters_close,
                        true,
                        self.field_displayer(),
                    )?
                    .unwrap_or_else(|| "null".to_string())
                )));
                if self.is_fixed_total_dose_selected() && self.is_do_dose_weighting_selected() {
                    param.set_fixed_image_dose(
                        Field::get_text_boolean_field_displayer(
                            &*self.tf_fixed_total_dose,
                            true,
                            self.field_displayer(),
                        )?
                        .as_deref(),
                    );
                    param.set_dose_output_file(Some(&format!(
                        "{}{}{EXTENSION_DIVIDER}{}",
                        Field::get_text_boolean_field_displayer(
                            &*self.ltf_rootname_output_files,
                            true,
                            self.field_displayer(),
                        )?
                        .unwrap_or_else(|| "null".to_string()),
                        sort_tilt_frames_param::DOSE_OUTPUT_FILE_DOSE,
                        extension::CLASS.txt
                    )));
                }
            }
            Ok(())
        })();
        // catch (final FieldValidationFailedException e) { return false; }
        if result.is_err() {
            return Ok(false);
        }
        Ok(true)
    }

    /// Java `getParameters(AlignFramesParam)`.
    fn get_parameters_align_frames_param(
        &self,
        param: &mut AlignFramesParam,
    ) -> Result<bool, FortranInputSyntaxException> {
        /// The two exceptions the Java's nested try blocks catch.
        enum Thrown {
            FortranInputSyntax(FortranInputSyntaxException),
            FieldValidationFailed(FieldValidationFailedException),
        }
        let fd = || self.field_displayer();
        let report = |e: LogFileError| match e {
            LogFileError::Lock(_) => {}
            e => eprintln!("{e}"),
        };
        let mut bad_parameter = String::new();
        let result = (|| -> Result<bool, Thrown> {
            let syntax = Thrown::FortranInputSyntax;
            let validation = Thrown::FieldValidationFailed;
            if self.rb_metadata_file.is_selected() {
                param.set_metadata_file(self.bctf_metadata_file.get_text_void().as_deref());
                param.set_adjust_and_write_mdoc(Some(
                    &align_frames_param::ADJUST_AND_WRITE_MDOC_VALUE.to_string(),
                ));
            } else {
                let subdir =
                    PathBuf::from(self.manager.get_property_user_dir().unwrap_or_default());
                if !self.is_angles_in_filenames_selected() {
                    match self.manager.get_list_of_input_files(self, &subdir, false) {
                        Ok(list) => param.set_list_of_input_files(Some(&list)),
                        Err(e) => report(e),
                    }
                } else {
                    param.set_list_of_input_files(Some(&format!(
                        "{}{}{EXTENSION_DIVIDER}{}",
                        Field::get_text_void(&*self.ltf_rootname_output_files)
                            .unwrap_or_else(|| "null".to_string()),
                        sort_tilt_frames_param::OUTPUT_FILE_LIST_INLIST,
                        extension::CLASS.txt
                    )));
                }
            }
            param.set_path_to_frames_in_mdoc(
                self.bctf_path_to_frames_in_mdoc.get_text_void().as_deref(),
            );
            if self.cb_corresponding_stack.is_enabled() && self.cb_corresponding_stack.is_selected()
            {
                param.set_corresponding_stack(
                    self.bctf_corresponding_stack.get_text_void().as_deref(),
                );
            }
            if self.cb_tilt_angle_file.is_enabled() && self.cb_tilt_angle_file.is_selected() {
                param.set_tilt_angle_file(self.bctf_tilt_angle_file.get_text_void().as_deref());
            }
            bad_parameter = self.ltf_axis_rotation_angle.get_label();
            param.set_axis_rotation_angle(
                Field::get_text_boolean_field_displayer(&*self.ltf_axis_rotation_angle, true, fd())
                    .map_err(validation)?
                    .as_deref(),
            );
            self.eer_super_res_z_sum_padding_panel.get_parameters(param);
            param.set_ref_and_defect_from_titles(self.cb_ref_and_defect_from_titles.is_selected());
            param.set_gain_reference_file(self.bctf_gain_reference_file.get_text_void().as_deref());
            if self.rb_rotation_and_flip.is_enabled() {
                if self.rb_rotation_and_flip.is_selected() {
                    param.set_rotation_and_flip(Some(
                        &align_frames_param::ROTATION_AND_FLIP_DEFAULT.to_string(),
                    ));
                } else if self.rb_rotation_and_flip2.is_selected() {
                    param.set_rotation_and_flip(Some(
                        &self.sp_rotation_and_flip.get_value().to_string(),
                    ));
                }
            } else {
                param.reset_rotation_and_flip();
            }
            param.set_camera_defect_file(self.bctf_camera_defect_file.get_text_void().as_deref());
            if self.rb_truncate_above_input_counts.is_selected() {
                bad_parameter = Field::get_quoted_label(&*self.tf_truncate_above_input_counts)
                    .unwrap_or_else(|| "null".to_string());
                param.set_truncate_above(
                    Field::get_text_boolean_field_displayer(
                        &*self.tf_truncate_above_input_counts,
                        true,
                        fd(),
                    )
                    .map_err(validation)?
                    .as_deref(),
                );
            } else if self.rb_truncate_above_sds.is_selected() {
                bad_parameter = Field::get_quoted_label(&*self.tf_truncate_above_sds)
                    .unwrap_or_else(|| "null".to_string());
                param.set_truncate_above(Some(&format!(
                    "-{}",
                    Field::get_text_boolean_field_displayer(
                        &*self.tf_truncate_above_sds,
                        true,
                        fd()
                    )
                    .map_err(validation)?
                    .unwrap_or_else(|| "null".to_string())
                )));
            }
            if self.rb_custom_pairwise_frames.is_selected() {
                param.set_pairwise_frames(Some(&self.sp_pairwise_frames.get_value().to_string()));
            } else if self.rb_half_pairwise_frames.is_selected() {
                param.set_pairwise_frames(Some(
                    &align_frames_param::HALF_PAIRWISE_FRAMES.to_string(),
                ));
            } else if self.rb_all_pairwise_frames.is_selected() {
                param.set_pairwise_frames(Some(
                    &align_frames_param::ALL_PAIRWISE_FRAMES.to_string(),
                ));
            }

            if self.rb_reduce_by.is_selected() {
                param
                    .set_align_sum_and_binning(
                        Some(&self.sp_reduce_by.get_value().to_string()),
                        Some(&self.sp_align_and_sum_binning_value2.get_value().to_string()),
                    )
                    .map_err(syntax)?;
            } else {
                param
                    .set_align_sum_and_binning(
                        Some(&align_frames_param::ALIGN_AND_SUM_BINNING_DEFAULT.to_string()),
                        Some(&self.sp_align_and_sum_binning_value2.get_value().to_string()),
                    )
                    .map_err(syntax)?;
            }
            if self.rb_target_align_size.is_selected() {
                param.set_target_align_size(Some(
                    &self.sp_target_align_size.get_value().to_string(),
                ));
            }
            if self.rb_test_binnings.is_selected() {
                bad_parameter = Field::get_quoted_label(&*self.tf_test_binnings)
                    .unwrap_or_else(|| "null".to_string());
                param
                    .set_test_binnings(
                        Field::get_text_boolean_field_displayer(
                            &*self.tf_test_binnings,
                            true,
                            fd(),
                        )
                        .map_err(validation)?
                        .as_deref(),
                    )
                    .map_err(syntax)?;
            }
            // try { ... } catch (final FortranInputSyntaxException e) {
            // e.printStackTrace(); }
            bad_parameter = self.ltf_filter_cutoffs.get_label();
            let vary_filter =
                Field::get_text_boolean_field_displayer(&*self.ltf_filter_cutoffs, true, fd())
                    .map_err(validation)?;
            if let Err(e) = param.set_vary_filter(vary_filter.as_deref()) {
                eprintln!("{e}");
            }
            param.set_use_hybrid_shifts(self.cb_use_hybrid_shifts.is_selected());
            bad_parameter = self.ltf_shift_limit.get_label();
            param.set_shift_limit(
                Field::get_text_boolean_field_displayer(&*self.ltf_shift_limit, true, fd())
                    .map_err(validation)?
                    .as_deref(),
            );
            if self.cb_group_frames.is_selected() {
                param.set_group_size(Some(&self.sp_group_frames.get_value().to_string()));
            }
            if self.cb_refine_alignment.is_selected() {
                param.set_refine_alignment(Some(&self.sp_refine_alignment.get_value().to_string()));
            }
            param.set_refine_with_group_sums(
                self.cb_refine_with_group_sums.is_enabled()
                    && self.cb_refine_with_group_sums.is_selected(),
            );
            if self.cb_refine_alignment.is_selected() {
                bad_parameter = self.ltf_refine_radius2.get_label();
                param.set_refine_radius2(
                    Field::get_text_boolean_field_displayer(&*self.ltf_refine_radius2, true, fd())
                        .map_err(validation)?
                        .as_deref(),
                );
                bad_parameter = self.ltf_stop_iterations_at_shift.get_label();
                param.set_stop_iterations_at_shift(
                    Field::get_text_boolean_field_displayer(
                        &*self.ltf_stop_iterations_at_shift,
                        true,
                        fd(),
                    )
                    .map_err(validation)?
                    .as_deref(),
                );
            }
            if self.cb_min_for_spline_smoothing.is_selected() {
                param.set_min_for_spline_smoothing(Some(
                    &self.sp_min_for_spline_smoothing.get_value().to_string(),
                ));
            } else {
                param.set_min_for_spline_smoothing(Some(
                    &align_frames_param::PREVENT_MIN_FOR_SPLINE_SMOOTHING.to_string(),
                ));
            }
            bad_parameter = self.ltf_starting_ending_frames_first.get_label();
            bad_parameter = self.ltf_starting_ending_frames_second.get_label();
            let first = Field::get_text_boolean_field_displayer(
                &*self.ltf_starting_ending_frames_first,
                true,
                fd(),
            )
            .map_err(validation)?;
            let second = Field::get_text_boolean_field_displayer(
                &*self.ltf_starting_ending_frames_second,
                true,
                fd(),
            )
            .map_err(validation)?;
            param
                .set_starting_ending_frames(first.as_deref(), second.as_deref())
                .map_err(syntax)?;
            if self.cb_do_dose_weighting.is_enabled() && self.cb_do_dose_weighting.is_selected() {
                if self.rb_fixed_total_dose.is_selected() {
                    if self.cb_tilt_angle_file.is_selected()
                        || self.cb_angles_in_filenames.is_selected()
                    {
                        param.set_type_of_dose_file(Some(
                            &align_frames_param::TYPE_OF_DOSE_FILE_VAL_2.to_string(),
                        ));

                        param.set_dose_weighting_file(Some(&format!(
                            "{}{}{EXTENSION_DIVIDER}{}",
                            AlignFramesDisplay::get_rootname_output_files(self),
                            sort_tilt_frames_param::DOSE_OUTPUT_FILE_DOSE,
                            extension::CLASS.txt
                        )));
                    } else {
                        bad_parameter = Field::get_quoted_label(&*self.tf_fixed_total_dose)
                            .unwrap_or_else(|| "null".to_string());
                        param.set_fixed_total_dose(
                            Field::get_text_boolean_field_displayer(
                                &*self.tf_fixed_total_dose,
                                true,
                                fd(),
                            )
                            .map_err(validation)?
                            .as_deref(),
                        );
                    }
                } else if self.rb_dose_weighting_file.is_selected() {
                    param.set_type_of_dose_file(Some(
                        &align_frames_param::TYPE_OF_DOSE_FILE_VAL_4.to_string(),
                    ));
                }
                param.set_normalize_dose_weighting(self.cb_normalize_dose_weighting.is_selected());
                if self.cb_voltage.is_selected() {
                    param.set_voltage(Some(&align_frames_param::VOLTAGE_DEFAULT.to_string()));
                }
                if !Field::is_empty(&*self.ltf_optimal_dose_scaling) {
                    bad_parameter = self.ltf_optimal_dose_scaling.get_label();
                    param
                        .set_optimal_dose_scaling(
                            Field::get_text_boolean_field_displayer(
                                &*self.ltf_optimal_dose_scaling,
                                true,
                                fd(),
                            )
                            .map_err(validation)?
                            .as_deref(),
                        )
                        .map_err(syntax)?;
                }
                if self.cb_unweighted_output_file.is_selected()
                    && !self.cb_normalize_dose_weighting.is_selected()
                {
                    let unweighted_output_file = format!(
                        "{}{}{EXTENSION_DIVIDER}{}",
                        Field::get_text_void(&*self.ltf_rootname_output_files)
                            .unwrap_or_else(|| "null".to_string()),
                        align_frames_param::UNWEIGHTED_OUTPUT_FILENAME,
                        extension::CLASS.mrc
                    );
                    param.set_unweighted_output_file(Some(&unweighted_output_file));
                }
            }
            if self.rb_scaling_of_sum_factor.is_selected() {
                bad_parameter = Field::get_quoted_label(&*self.tf_scaling_of_sum_factor)
                    .unwrap_or_else(|| "null".to_string());
                param.set_scaling_of_sum(
                    Field::get_text_boolean_field_displayer(
                        &*self.tf_scaling_of_sum_factor,
                        true,
                        fd(),
                    )
                    .map_err(validation)?
                    .as_deref(),
                );
            }
            if self.rb_mode_to_output_16bit_int.is_selected() {
                param.set_mode_to_output(Some(
                    &align_frames_param::MODE_TO_OUTPUT_16BIT_INT.to_string(),
                ));
            } else if self.rb_mode_to_output_float.is_selected() {
                param.set_mode_to_output(Some(
                    &align_frames_param::MODE_TO_OUTPUT_FLOAT.to_string(),
                ));
            }
            if self.rb_sum_rotation_and_flip.is_selected() {
                param.set_sum_rotation_and_flip(Some(
                    &align_frames_param::SUM_ROTATION_AND_FLIP_DEFAULT.to_string(),
                ));
            } else if self.rb_sum_rotation_and_flip2.is_selected() {
                param.set_sum_rotation_and_flip(Some(
                    &self.sp_sum_rotation_and_flip.get_value().to_string(),
                ));
            }
            bad_parameter = self.ltf_output_image_file.get_label();
            param.set_output_image_file(
                Field::get_text_boolean_field_displayer(&*self.ltf_output_image_file, true, fd())
                    .map_err(validation)?
                    .as_deref(),
            );
            if self.cb_use_gpu.is_selected() {
                param.set_use_gpu(Some(&align_frames_param::USE_GPU_VALUE.to_string()));
            }
            Ok(true)
        })();
        match result {
            Ok(value) => Ok(value),
            // catch (final FortranInputSyntaxException except): rethrown with the
            // bad parameter's label in front.
            Err(Thrown::FortranInputSyntax(except)) => {
                let message = format!(
                    "{} {}",
                    bad_parameter,
                    except.get_message().unwrap_or("null")
                );
                Err(FortranInputSyntaxException::new(&message))
            }
            // catch (final FieldValidationFailedException e) { return false; }
            Err(Thrown::FieldValidationFailed(_)) => Ok(false),
        }
    }

    /// Java `setParameters(AlignFramesParam)`.
    ///
    /// Upstream bug fixed in translation (AlignFramesPanel.java:1714-1897): the
    /// Java parses the com file's values with `Integer.parseInt` /
    /// `Double.parseDouble` and lets a NumberFormatException escape (the rest of
    /// the parameters are then never shown); here an unparsable value skips only
    /// its own field.
    fn set_parameters(&self, param: &AlignFramesParam) {
        let parse_int = |value: String| value.parse::<i32>().ok();
        let parse_double = |value: String| java_lang_double_value_of(&value).ok();
        self.eer_super_res_z_sum_padding_panel.set_parameters(param);
        Field::set_value_string(
            &*self.ltf_axis_rotation_angle,
            Some(&param.get_axis_rotation_angle()),
        );
        self.cb_ref_and_defect_from_titles
            .set_selected_boolean(param.get_ref_and_defect_from_titles());
        if param.is_rotation_and_flip() {
            if let Some(rotation_and_flip) = parse_int(param.get_rotation_and_flip()) {
                if rotation_and_flip == align_frames_param::ROTATION_AND_FLIP_DEFAULT {
                    self.rb_rotation_and_flip.set_selected_boolean(true);
                } else {
                    self.rb_rotation_and_flip2.set_selected_boolean(true);
                    self.sp_rotation_and_flip
                        .set_value_string(Some(&param.get_rotation_and_flip()));
                }
            }
        }
        if param.is_truncate_above() {
            if let Some(truncate_above) = parse_double(param.get_truncate_above()) {
                if truncate_above >= 0.0 {
                    self.rb_truncate_above_input_counts
                        .set_selected_boolean(true);
                    Field::set_value_string(
                        &*self.tf_truncate_above_input_counts,
                        Some(&param.get_truncate_above()),
                    );
                } else {
                    self.rb_truncate_above_sds.set_selected_boolean(true);
                    let truncate_above_sds = truncate_above * -1.0;
                    Field::set_value_string(
                        &*self.tf_truncate_above_sds,
                        Some(&java_lang_double_to_string(truncate_above_sds)),
                    );
                }
            }
        } else {
            self.rb_truncate_above_none.set_selected_boolean(true);
        }
        if param.is_pairwise_frames() {
            if let Some(pairwise_frames_value) = parse_int(param.get_pairwise_frames()) {
                if pairwise_frames_value >= align_frames_param::PAIRWISE_FRAMES_SPINNER_MIN
                    && pairwise_frames_value <= align_frames_param::PAIRWISE_FRAMES_SPINNER_MAX
                {
                    self.rb_custom_pairwise_frames.set_selected_boolean(true);
                    self.sp_pairwise_frames.set_value_int(pairwise_frames_value);
                } else if pairwise_frames_value == align_frames_param::HALF_PAIRWISE_FRAMES {
                    self.rb_half_pairwise_frames.set_selected_boolean(true);
                } else if pairwise_frames_value == align_frames_param::ALL_PAIRWISE_FRAMES {
                    self.rb_all_pairwise_frames.set_selected_boolean(true);
                }
            }
        } else {
            // Because it is optional to load a .com file, some defaults have to be
            // revised.
            self.rb_custom_pairwise_frames.set_selected_boolean(true);
        }
        if param.is_target_align_size() {
            self.rb_target_align_size.set_selected_boolean(true);
            self.sp_target_align_size
                .set_value_string(Some(&param.get_target_align_size()));
        } else if param.is_test_binnings() {
            self.rb_test_binnings.set_selected_boolean(true);
            Field::set_value_string(&*self.tf_test_binnings, Some(&param.get_test_binnings()));
        } else if param.is_reduce_by() {
            let reduce_by_val = param.get_reduce_by_value();
            if reduce_by_val >= align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER_MIN
                && reduce_by_val <= align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER_MAX
            {
                self.rb_reduce_by.set_selected_boolean(true);
                self.sp_reduce_by.set_value_int(reduce_by_val);
            } else if reduce_by_val == align_frames_param::ALIGN_AND_SUM_BINNING_DEFAULT {
                self.rb_binning_default.set_selected_boolean(true);
            }

            // AlignAndSumBinning second spinner value
            let align_and_sum_binning_val2 = param.get_align_and_sum_binning_val2();
            if align_and_sum_binning_val2 >= align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER2_MIN
                && align_and_sum_binning_val2
                    <= align_frames_param::ALIGN_AND_SUM_BINNING_SPINNER2_MAX
            {
                self.sp_align_and_sum_binning_value2
                    .set_value_int(param.get_align_and_sum_binning_val2());
            }
        }
        self.ltf_filter_cutoffs
            .set_text_string(Some(&param.get_vary_filter_cutoff()));
        self.cb_use_hybrid_shifts
            .set_selected_boolean(param.is_use_hybrid_shifts());
        if param.is_shift_limit() {
            Field::set_value_string(&*self.ltf_shift_limit, Some(&param.get_shift_limit()));
        }
        if param.is_group_size() {
            if let Some(group_frames_value) = parse_int(param.get_group_size()) {
                if group_frames_value >= align_frames_param::GROUP_FRAMES_SPINNER_MIN
                    && group_frames_value <= align_frames_param::GROUP_FRAMES_SPINNER_MAX
                {
                    self.cb_group_frames.set_selected_boolean(true);
                    self.sp_group_frames.set_value_int(group_frames_value);
                }
            }
        }
        if param.is_refine_alignment() {
            if let Some(refine_alignment_value) = parse_int(param.get_refine_alignment()) {
                if refine_alignment_value >= align_frames_param::REFINE_ALIGNMENT_SPINNER_MIN
                    && refine_alignment_value <= align_frames_param::REFINE_ALIGNMENT_SPINNER_MAX
                {
                    self.cb_refine_alignment.set_selected_boolean(true);
                    self.sp_refine_alignment
                        .set_value_int(refine_alignment_value);
                }
            }
        } else {
            // Because it is optional to load a .com file, some defaults have to be
            // revised.
            self.cb_refine_alignment.set_selected_boolean(false);
        }
        self.cb_refine_with_group_sums
            .set_selected_boolean(param.is_refine_with_group_sums());
        if param.is_refine_radius2() {
            Field::set_value_string(&*self.ltf_refine_radius2, Some(&param.get_refine_radius2()));
        }
        if param.is_stop_iterations_at_shift() {
            Field::set_value_string(
                &*self.ltf_stop_iterations_at_shift,
                Some(&param.get_stop_iterations_at_shift()),
            );
        }
        if param.is_min_for_spline_smoothing_set() {
            if let Some(min_for_spline_smoothing_value) =
                parse_int(param.get_min_for_spline_smoothing())
            {
                if min_for_spline_smoothing_value
                    >= align_frames_param::MIN_FOR_SPLINE_SMOOTHING_SPINNER_MIN
                    && min_for_spline_smoothing_value
                        <= align_frames_param::MIN_FOR_SPLINE_SMOOTHING_SPINNER_MAX
                {
                    self.cb_min_for_spline_smoothing.set_selected_boolean(true);
                    self.sp_min_for_spline_smoothing
                        .set_value_int(min_for_spline_smoothing_value);
                }
                if min_for_spline_smoothing_value
                    == align_frames_param::PREVENT_MIN_FOR_SPLINE_SMOOTHING
                {
                    self.cb_min_for_spline_smoothing.set_selected_boolean(false);
                }
            }
        }
        if param.is_starting_ending_frames_first() {
            Field::set_value_string(
                &*self.ltf_starting_ending_frames_first,
                Some(&param.get_starting_ending_frames_first().to_string()),
            );
        }
        if param.is_starting_ending_frames_second() {
            Field::set_value_string(
                &*self.ltf_starting_ending_frames_second,
                Some(&param.get_starting_ending_frames_second().to_string()),
            );
        }
        let is_fixed_total_dose = param.is_fixed_total_dose();
        let is_type_of_dose_file = param.is_type_of_dose_file();
        if is_fixed_total_dose || is_type_of_dose_file {
            self.cb_do_dose_weighting.set_selected_boolean(true);
            self.ph_dose_weighting
                .set_open(self.cb_do_dose_weighting.is_selected());
            self.pnl_dose_weighting_body.set_visible(true);
            self.set_output_image_file();
        }
        if is_fixed_total_dose {
            self.rb_fixed_total_dose.set_selected_boolean(true);
            Field::set_value_string(
                &*self.tf_fixed_total_dose,
                Some(&param.get_fixed_total_dose()),
            );
        }
        if is_type_of_dose_file {
            if parse_int(param.get_type_of_dose_file())
                == Some(align_frames_param::TYPE_OF_DOSE_FILE_VAL_4)
            {
                self.rb_dose_weighting_file.set_selected_boolean(true);
            }
        }
        // Because it is optional to load a .com file, some defaults have to be
        // revised.
        if self.cb_do_dose_weighting.is_selected() {
            self.cb_normalize_dose_weighting
                .set_selected_boolean(param.is_normalize_dose_weighting());
        }
        if param.is_voltage() {
            if parse_int(param.get_voltage()) == Some(align_frames_param::VOLTAGE_DEFAULT) {
                self.cb_voltage.set_selected_boolean(true);
            }
        }
        if param.is_optimal_dose_scaling() {
            // DO NOT validate here. Set values as they are
            if let Some(optimal_dose_scaling_value) = parse_double(param.get_optimal_dose_scaling())
            {
                if optimal_dose_scaling_value
                    > align_frames_param::OPTIMAL_DOSE_SCALING_VALIDATE_MIN
                    && optimal_dose_scaling_value
                        < align_frames_param::OPTIMAL_DOSE_SCALING_VALIDATE_MAX
                {
                    Field::set_value_string(
                        &*self.ltf_optimal_dose_scaling,
                        Some(&param.get_optimal_dose_scaling()),
                    );
                }
            }
        }
        if param.is_unweighted_output_file() {
            self.cb_unweighted_output_file.set_selected_boolean(true);
        }
        if param.is_scaling_of_sum() {
            self.rb_scaling_of_sum_factor.set_selected_boolean(true);
            self.tf_scaling_of_sum_factor
                .set_text_string(Some(&param.get_scaling_of_sum()));
        }
        if param.is_mode_to_output() {
            if let Some(mode_to_output_value) = parse_int(param.get_mode_to_output()) {
                if mode_to_output_value == align_frames_param::MODE_TO_OUTPUT_16BIT_INT {
                    self.rb_mode_to_output_16bit_int.set_selected_boolean(true);
                } else if mode_to_output_value == align_frames_param::MODE_TO_OUTPUT_FLOAT {
                    self.rb_mode_to_output_float.set_selected_boolean(true);
                }
            }
        }
        if param.is_sum_rotation_and_flip() {
            if let Some(sum_rotation_and_flip) = parse_int(param.get_sum_rotation_and_flip()) {
                if sum_rotation_and_flip == align_frames_param::SUM_ROTATION_AND_FLIP_DEFAULT {
                    self.rb_sum_rotation_and_flip.set_selected_boolean(true);
                } else {
                    self.rb_sum_rotation_and_flip2.set_selected_boolean(true);
                    self.sp_sum_rotation_and_flip
                        .set_value_string(Some(&param.get_sum_rotation_and_flip()));
                }
            }
        }
        self.cb_use_gpu.set_selected_boolean(param.is_use_gpu());
        self.update_display();
    }

    /// Java `isListOfInputFilesSelected()`.
    fn is_list_of_input_files_selected(&self) -> bool {
        self.rb_list_of_input_files.is_selected()
    }

    /// Java `isSelectedFilesSelected()`.
    fn is_selected_files_selected(&self) -> bool {
        self.rb_selected_files.is_selected()
    }

    /// Java `isAnglesInFilenamesSelected()`.
    fn is_angles_in_filenames_selected(&self) -> bool {
        self.cb_angles_in_filenames.is_selected() && self.cb_angles_in_filenames.is_enabled()
    }

    /// Java `isCorrespondingStackSelected()`.
    fn is_corresponding_stack_selected(&self) -> bool {
        self.cb_corresponding_stack.is_selected() && self.cb_corresponding_stack.is_enabled()
    }

    /// Java `isTiltAngleFileSelected()`.
    fn is_tilt_angle_file_selected(&self) -> bool {
        self.cb_tilt_angle_file.is_selected() && self.cb_tilt_angle_file.is_enabled()
    }

    /// Java `isFixedTotalDoseSelected()`.
    fn is_fixed_total_dose_selected(&self) -> bool {
        self.rb_fixed_total_dose.is_selected()
    }

    /// Java `isDoDoseWeightingSelected()`.
    fn is_do_dose_weighting_selected(&self) -> bool {
        self.cb_do_dose_weighting.is_selected()
    }

    /// Java `getTextAreaInputFiles()`.
    fn get_text_area_input_files(&self) -> String {
        self.text_area_input_files.get_component().get_text()
    }

    /// Java `getRootnameOutputFiles()`.
    fn get_rootname_output_files(&self) -> String {
        Field::get_text_void(&*self.ltf_rootname_output_files).unwrap_or_default()
    }

    /// Java `getOutputImageFileName()`.
    fn get_output_image_file_name(&self) -> String {
        Field::get_text_void(&*self.ltf_output_image_file).unwrap_or_default()
    }

    /// Java `setupLocalArguments()`.
    fn setup_local_arguments(&self) -> LocalArguments {
        let mut local_arguments = LocalArguments::default();
        local_arguments.set_raw_image_stack(
            Field::get_text_void(&*self.ltf_output_image_file).unwrap_or_default(),
        );
        local_arguments.set_dir(self.local_arguments_dir.borrow().clone());

        local_arguments
    }

    /// Java `isMetadataFileSelected()`.
    fn is_metadata_file_selected(&self) -> bool {
        self.rb_metadata_file.is_selected()
    }

    /// Java `getNewMdocFileName()`.
    fn get_new_mdoc_file_name(&self) -> String {
        format!(
            "{}{EXTENSION_DIVIDER}{}",
            Field::get_text_void(&*self.ltf_output_image_file)
                .unwrap_or_else(|| "null".to_string()),
            extension::CLASS.mdoc
        )
    }
}

impl ControlTarget for AlignFramesPanel {
    /// Java `clear()`: empty.
    fn clear(&self) {}

    /// Java `setText(File)`.
    fn set_text_file(&self, file: Option<&Path>) {
        AlignFramesPanel::set_text_file(self, file);
    }

    /// Java `setText(File[])`.
    fn set_text_file_array(&self, files: Option<&[PathBuf]>) {
        AlignFramesPanel::set_text_file_array(self, files);
    }

    /// Java `getLabel()`: null.
    fn get_label(&self) -> Option<String> {
        None
    }

    /// Java `setComponentControl(boolean, ControlState)`: empty.
    fn set_component_control(&self, _control: bool, _state: Option<&'static ControlState>) {}

    /// Java `setEnableControl(boolean, ControlState)`: empty.
    fn set_enable_control(&self, _control: bool, _state: Option<&'static ControlState>) {}

    /// Java `sendControlEvent()`: empty.
    fn send_control_event(&self) {}

    /// Java `isLocalDir(String)`: false.
    fn is_local_dir(&self, _current_directory: Option<&str>) -> bool {
        false
    }
}

impl ControlListener for AlignFramesPanel {
    /// Java `controlEvent()`.
    fn control_event(&self) {
        if self.bctf_metadata_file.is_enabled() {
            self.set_text_file(self.bctf_metadata_file.get_file().as_deref());
        } else if self.bctf_list_of_input_files.is_enabled() {
            self.set_text_file(self.bctf_list_of_input_files.get_file().as_deref());
        }

        self.update_display();
    }
}

impl BrowsingDirectory for AlignFramesPanel {
    /// Java `getBrowsingDir()`.
    fn get_browsing_dir(&self) -> Option<PathBuf> {
        self.align_frames_browsing_dir.borrow().clone()
    }

    /// Java `setBrowsingDir(File)`.
    fn set_browsing_dir(&self, file: Option<&Path>) {
        *self.align_frames_browsing_dir.borrow_mut() = file.map(Path::to_path_buf);
    }
}

impl FieldDisplayer for AlignFramesPanel {
    /// Java `display()`.
    fn display_void(&self) {
        AlignFramesPanel::display_void(self);
    }

    /// Java `display(UIComponent)`.
    fn display_ui_component(&self, ui_component: Option<&dyn UIComponent>) {
        AlignFramesPanel::display_ui_component(self, ui_component);
    }
}

impl Run3dmodButtonContainer for AlignFramesPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        self.action_option(
            Some(action_command),
            deferred_3dmod_button,
            run_3dmod_menu_options,
        );
    }
}

/// Java private static final nested class `Tab`.
pub struct Tab {
    /// Java private final `index`.
    index: i32,
    /// Java private final `title`.
    title: &'static str,
}

impl Tab {
    /// Java private static final `INPUT_AND_PREPROCESSING`.
    const INPUT_AND_PREPROCESSING: &'static Tab = &Tab {
        index: 0,
        title: "Input and Pre-processing",
    };
    /// Java private static final `ALIGNMENT`.
    const ALIGNMENT: &'static Tab = &Tab {
        index: 1,
        title: "Alignment",
    };

    /// Java private static final `NUM_TABS`.
    const NUM_TABS: usize = 2;

    /// Java private static `getInstance(int)`.
    fn get_instance(index: i32) -> Option<&'static Tab> {
        if index == Tab::INPUT_AND_PREPROCESSING.index {
            return Some(Tab::INPUT_AND_PREPROCESSING);
        }
        if index == Tab::ALIGNMENT.index {
            return Some(Tab::ALIGNMENT);
        }
        None
    }

    /// Java private static `getDefaultInstance(ViewType)` (never called).
    #[allow(dead_code)]
    fn get_default_instance(view_type: ViewType) -> &'static Tab {
        if view_type == ViewType::Montage {
            return Tab::INPUT_AND_PREPROCESSING;
        }
        Tab::ALIGNMENT
    }

    /// Java `equals(int)`.
    #[allow(dead_code)]
    pub fn equals(&self, index: i32) -> bool {
        self.index == index
    }
}

/// Java `toString()`: the title.
impl std::fmt::Display for Tab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.title)
    }
}

/// Java private static final nested class `TabChangeListener implements
/// ChangeListener`.
struct TabChangeListener {
    /// Java private `alignFramePanel`.
    align_frame_panel: Weak<AlignFramesPanel>,
}

impl TabChangeListener {
    /// Java `TabChangeListener(AlignFramesPanel)`.
    fn new(tiltalign_panel: Weak<AlignFramesPanel>) -> TabChangeListener {
        TabChangeListener {
            align_frame_panel: tiltalign_panel,
        }
    }

    /// Java `stateChanged(ChangeEvent)`.
    fn state_changed(&self, _change_event: &ChangeEvent) {
        if let Some(align_frame_panel) = self.align_frame_panel.upgrade() {
            align_frame_panel.change_tab_void();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::jdk::named_components;
    use crate::imod::etomo::ui::swing::etomo_menu::ToolType;
    use crate::imod::etomo::util::event_queue;

    #[test]
    fn tools_instance_builds_named_tree_and_derives_output_name() {
        let (names, output) = event_queue::invoke_and_wait(|| {
            let manager = ToolsManager::new(ToolType::AlignFrames);
            let panel =
                AlignFramesPanel::get_tools_instance(manager, AxisID::Only, DialogType::Tools);
            panel
                .ltf_rootname_output_files
                .set_text_string(Some("TS_01"));
            panel.set_output_image_file();
            let names: Vec<String> = named_components(&ToolPanel::get_component(&*panel))
                .into_iter()
                .map(|(name, _)| name)
                .collect();
            (
                names,
                AlignFramesDisplay::get_output_image_file_name(&*panel),
            )
        });
        assert_eq!(output, "TS_01_af.mrc");
        assert!(names.iter().any(|name| name.starts_with("tb.")));
        assert!(names.iter().any(|name| name.starts_with("ta.")));
    }
}
