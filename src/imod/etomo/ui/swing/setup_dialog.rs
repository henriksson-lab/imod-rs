//! `IMOD/Etomo/src/etomo/ui/swing/SetupDialog.java`.
//!
//! Description: Setup dialog for tomogram reconstruction.
//!
//! Java `final class SetupDialog extends ProcessDialog implements ContextMenu,
//! Run3dmodButtonContainer, Expandable, SetupReconInterface, FocusListener,
//! ControlListener, ActionListener`.  Following the translation's inheritance
//! convention the `ProcessDialog` superclass is embedded as `base` (with
//! `Deref`), and the methods `ProcessDialog` calls back into (`done`,
//! `buttonExecuteAction`) are this class's `ProcessDialogVirtual` impl.  The
//! dialog is an event-dispatch-thread object: it is created as `Rc<Self>`
//! (with `Rc::new_cyclic`, because field initialisers hand `this` to the
//! `Run3dmodButton`s), every method takes `&self`, and its mutable fields are
//! `Cell`/`RefCell`s borrowed only for one statement.
//!
//! The dialog holds its `SetupDialogExpert` as a `Weak`: the expert owns the
//! dialog, and the Java's two-way reference would otherwise be an `Rc` cycle.
//!
//! Layout, sizes, fonts, borders other than titles, and mouse/focus plumbing
//! are not modelled (see `jdk.rs`); each such Java statement is a
//! `// Swing layout:` comment in place.

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::copy_tomo_coms;
use crate::imod::etomo::comscript::exclude_views_param::ExcludeViewsParam;
use crate::imod::etomo::jdk::MouseEvent;
use crate::imod::etomo::jdk::{ActionEvent, ButtonGroup, FocusEvent, JComponent};
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::logic::validation_set::ValidationSet;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_double_to_string, java_lang_string_matches_whitespace, java_lang_string_trim,
};
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::extension::{EXTENSION_DIVIDER, Extension};
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::setup_recon_interface::{
    DirectiveFileCollectionHandle, SetupReconInterface,
};
use crate::imod::etomo::ui::swing::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::ui::swing::beveled_border::BeveledBorder;
use crate::imod::etomo::ui::swing::button_control_text_efield::ButtonControlTextEfield;
use crate::imod::etomo::ui::swing::check_box::CheckBox;
use crate::imod::etomo::ui::swing::context_menu::ContextMenu;
use crate::imod::etomo::ui::swing::context_popup::ContextPopup;
use crate::imod::etomo::ui::swing::control_listener::ControlListener;
use crate::imod::etomo::ui::swing::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::ui::swing::etched_border::EtchedBorder;
use crate::imod::etomo::ui::swing::etomo_panel::EtomoPanel;
use crate::imod::etomo::ui::swing::expand_button::ExpandButton;
use crate::imod::etomo::ui::swing::expandable::Expandable;
use crate::imod::etomo::ui::swing::file_chooser::{self, FileChooser};
use crate::imod::etomo::ui::swing::file_text_field::FileTextField;
use crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface;
use crate::imod::etomo::ui::swing::generic_mouse_adapter::GenericMouseAdapter;
use crate::imod::etomo::ui::swing::global_expand_button::GlobalExpandButton;
use crate::imod::etomo::ui::swing::label::Label;
use crate::imod::etomo::ui::swing::labeled_text_field::LabeledTextField;
use crate::imod::etomo::ui::swing::multi_line_button::MultiLineButton;
use crate::imod::etomo::ui::swing::process_control_panel;
use crate::imod::etomo::ui::swing::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use crate::imod::etomo::ui::swing::radio_button::RadioButton;
use crate::imod::etomo::ui::swing::run_3dmod_button::Run3dmodButton;
use crate::imod::etomo::ui::swing::run_3dmod_button_container::Run3dmodButtonContainer;
use crate::imod::etomo::ui::swing::setup_dialog_expert::SetupDialogExpert;
use crate::imod::etomo::ui::swing::template_panel::TemplatePanel;
use crate::imod::etomo::ui::swing::text_field::TextField;
use crate::imod::etomo::ui::swing::tilt_angle_panel::TiltAnglePanel;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities;
use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::Arc;

// private static final String RAW_IMAGE_STACK_LABEL = "Raw Image Stack: ";
/// Java package-private static final `FIDUCIAL_DIAMETER_LABEL`.
pub const FIDUCIAL_DIAMETER_LABEL: &str = "Fiducial diameter (nm): ";
/// Java package-private static final `AXIS_TYPE_LABEL`.
pub const AXIS_TYPE_LABEL: &str = "Axis Type";
/// Java package-private static final `FRAME_TYPE_LABEL`.
pub const FRAME_TYPE_LABEL: &str = "Frame Type";
/// Java package-private static final `SINGLE_AXIS_LABEL`.
pub const SINGLE_AXIS_LABEL: &str = "Single axis";
/// Java package-private static final `MONTAGE_LABEL`.
pub const MONTAGE_LABEL: &str = "Montage";
/// Java package-private static final `SINGLE_FRAME_LABEL`.
pub const SINGLE_FRAME_LABEL: &str = "Single frame";
/// Java private final (instance, not static) `BACKUP_DIRECTORY_LABEL`; a
/// constant string either way.
const BACKUP_DIRECTORY_LABEL: &str = "Backup directory: ";
/// Java private static final `TWODIR_LABEL_1` (unused in the Java).
#[allow(dead_code)]
const TWODIR_LABEL_1: &str = "Series was bidirectional from ";
/// Java private static final `TWODIR_LABEL_2`.
const TWODIR_LABEL_2: &str = " degrees";
/// Java private static final `VIEW_RAW_STACK_LABEL`.
const VIEW_RAW_STACK_LABEL: &str = "View Raw Image Stack";
/// Java private static final `REMOVE_EXCLUDE_VIEW_MSG`.
const REMOVE_EXCLUDE_VIEW_MSG: &str = "Excluded views have been removed";

/// Java `final class SetupDialog`.
pub struct SetupDialog {
    /// Java superclass `ProcessDialog`.
    base: Rc<ProcessDialog>,
    /// Java private final `pnlDataParameters`.
    pnl_data_parameters: Rc<JComponent>,
    // Dataset GUI objects
    /// Java private final `pnlDataset`.
    pnl_dataset: Rc<JComponent>,
    /// Java private final `bctfRawImageStack`.
    bctf_raw_image_stack: Rc<ButtonControlTextEfield>,
    /// Java private final `ftfBackupDirectory`.
    ftf_backup_directory: Rc<FileTextField>,
    // Data type GUI objects
    /// Java private final `pnlPerAxisInfo`.
    pnl_per_axis_info: Rc<JComponent>,
    /// Java private final `pnlAxisInfoA`.
    pnl_axis_info_a: Rc<EtomoPanel>,
    /// Java private final `pnlDataType`.
    pnl_data_type: Rc<EtomoPanel>,
    /// Java private final `pnlAxisType`.
    pnl_axis_type: Rc<EtomoPanel>,
    /// Java private final `rbSingleAxis`.
    rb_single_axis: Rc<RadioButton>,
    /// Java private final `rbDualAxis`.
    rb_dual_axis: Rc<RadioButton>,
    /// Java private final `pnlViewType`.
    pnl_view_type: Rc<EtomoPanel>,
    /// Java private final `rbSingleView`.
    rb_single_view: Rc<RadioButton>,
    /// Java private final `rbMontage`.
    rb_montage: Rc<RadioButton>,
    /// Java private final `btnViewRawStackA`.
    btn_view_raw_stack_a: Rc<Run3dmodButton>,
    /// Java private final `btnViewRawStackB`.
    btn_view_raw_stack_b: Rc<Run3dmodButton>,
    // Image parameter objects
    /// Java private final `pnlImageParams`.
    pnl_image_params: Rc<JComponent>,
    /// Java private final `btnScanHeader`.
    btn_scan_header: Rc<MultiLineButton>,
    /// Java private final `pnlImageRows`.
    pnl_image_rows: Rc<JComponent>,
    /// Java private final `pnlStackInfo`.
    pnl_stack_info: Rc<JComponent>,
    /// Java private final `ltfPixelSize`.
    ltf_pixel_size: Rc<LabeledTextField>,
    /// Java private final `ltfFiducialDiameter`.
    ltf_fiducial_diameter: Rc<LabeledTextField>,
    /// Java private final `ltfImageRotation`.
    ltf_image_rotation: Rc<LabeledTextField>,
    /// Java private final `pnlDistortionInfo`.
    pnl_distortion_info: Rc<JComponent>,
    /// Java private final `ftfDistortionFile`.
    ftf_distortion_file: Rc<FileTextField>,
    /// Java private final `ltfBinning`.
    ltf_binning: Rc<LabeledTextField>,
    /// Java private final `pnlMagGradientInfo`.
    pnl_mag_gradient_info: Rc<JComponent>,
    /// Java private final `ftfMagGradientFile`.
    ftf_mag_gradient_file: Rc<FileTextField>,
    /// Java private final `cbParallelProcess`.
    cb_parallel_process: Rc<CheckBox>,
    /// Java private final `cbGpuProcessing`.
    cb_gpu_processing: Rc<CheckBox>,
    // Tilt angle GUI objects
    /// Java private final `ltfExcludeListA`.
    ltf_exclude_list_a: Rc<LabeledTextField>,
    /// Java private final `pnlAdjustedFocusA`.
    pnl_adjusted_focus_a: Rc<JComponent>,
    /// Java private final `cbAdjustedFocusA`.
    cb_adjusted_focus_a: Rc<CheckBox>,
    /// Java private final `borderAxisInfoB`.
    border_axis_info_b: Rc<BeveledBorder>,
    /// Java private final `ltfExcludeListB`.
    ltf_exclude_list_b: Rc<LabeledTextField>,
    /// Java private final `pnlAdjustedFocusB`.
    pnl_adjusted_focus_b: Rc<JComponent>,
    /// Java private final `cbAdjustedFocusB`.
    cb_adjusted_focus_b: Rc<CheckBox>,
    /// Java private final `cbTfTwodir`.
    cb_tf_twodir: Rc<CheckBox>,
    /// Java private (non-final, never reassigned) `bgTfTwoDir`.
    #[allow(dead_code)]
    bg_tf_two_dir: Rc<ButtonGroup>,
    /// Java private final `rbBidirectional`.
    rb_bidirectional: Rc<RadioButton>,
    /// Java private final `rbDoseSymmetric`.
    rb_dose_symmetric: Rc<RadioButton>,
    /// Java private final `tfTwodir`.
    tf_twodir: Rc<TextField>,
    /// Java private final `tfDoseSym`.
    tf_dose_sym: Rc<TextField>,
    /// Java private final `cbTfBtwodir`.
    cb_tf_btwodir: Rc<CheckBox>,
    /// Java private (non-final, never reassigned) `bgTfBtwoDir`.
    #[allow(dead_code)]
    bg_tf_btwo_dir: Rc<ButtonGroup>,
    /// Java private final `rbBBidirectional`.
    rb_b_bidirectional: Rc<RadioButton>,
    /// Java private final `rbBDoseSymmetric`.
    rb_b_dose_symmetric: Rc<RadioButton>,
    /// Java private final `tfBtwodir`.
    tf_btwodir: Rc<TextField>,
    /// Java private final `tfBDoseSym`.
    tf_b_dose_sym: Rc<TextField>,
    /// Java private final `lTwodirfrom`.
    l_twodirfrom: Rc<JComponent>,
    /// Java private final `lBtwodirfrom`.
    l_btwodirfrom: Rc<JComponent>,
    /// Java private final `lTwodir`.
    l_twodir: Rc<JComponent>,
    /// Java private final `lBtwodir`.
    l_btwodir: Rc<JComponent>,
    /// Java private final `cbRemoveExcludedViews`.
    cb_remove_excluded_views: Rc<CheckBox>,
    /// Java private final `cbDeleteOldFiles`.
    cb_delete_old_files: Rc<CheckBox>,
    /// Java private final `lRemoveExcludeViewsMsgA`.
    l_remove_exclude_views_msg_a: Rc<JComponent>,
    /// Java private final `lRemoveExcludeViewsMsgB`.
    l_remove_exclude_views_msg_b: Rc<JComponent>,
    // HalfFloatModeOutput
    /// Java private final `lHalfFloatModeOutput` (declared `JLabel`, built as
    /// a `Label`).
    l_half_float_mode_output: Rc<Label>,
    /// Java private final `cbHalfFloatModeOutput`.
    cb_half_float_mode_output: Rc<CheckBox>,
    /// Java private final `cbHalfFloatModeOutputIfFloat`.
    cb_half_float_mode_output_if_float: Rc<CheckBox>,
    /// Java private final `lHalfFloatModeOutputActive` (declared `JLabel`,
    /// built as a `Label`).
    l_half_float_mode_output_active: Rc<Label>,

    /// Java private final `expert`.  Weak: the expert owns this dialog.
    expert: Weak<SetupDialogExpert>,
    /// Java private final `calibrationAvailable`.
    calibration_available: bool,
    /// Java private final `listener`.
    listener: Rc<SetupDialogActionListener>,
    /// Java private final `templatePanel`.
    template_panel: Rc<TemplatePanel>,
    /// Java private final `progressPanel`.
    progress_panel: Rc<AxisProgressPanel>,
    /// Java private final `tiltAnglesA`.
    tilt_angles_a: Rc<TiltAnglePanel>,
    /// Java private final `tiltAnglesB`.
    tilt_angles_b: Rc<TiltAnglePanel>,

    /// Java private `directiveFileCollection` (never read or written after its
    /// initialiser in the Java).
    #[allow(dead_code)]
    directive_file_collection: RefCell<Option<DirectiveFileCollectionHandle>>,
    /// Java private `excludeViewsSucceededA`.
    exclude_views_succeeded_a: Cell<bool>,
    /// Java private `excludeViewsSucceededB`.
    exclude_views_succeeded_b: Cell<bool>,
    /// Rust-only: Java's `this`, handed to listeners and to the expandable
    /// registration.
    this: Weak<SetupDialog>,
}

impl Deref for SetupDialog {
    type Target = ProcessDialog;

    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

impl SetupDialog {
    /// Java private `SetupDialog(SetupDialogExpert, ApplicationManager, AxisID,
    /// DialogType, boolean, ValidationSet, AxisProgressPanel)`: construct the
    /// setup dialog.
    ///
    /// Java runs `super(...)`, then the field initialisers in declaration
    /// order, then the body; the statements of the body that only assign
    /// fields run inside `Rc::new_cyclic` in their Java order, and the rest of
    /// the body runs right after, on the finished `Rc`, also in Java order.
    #[allow(clippy::too_many_arguments)]
    fn new(
        expert: &Rc<SetupDialogExpert>,
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        calibration_available: bool,
        binning_validation_set: Arc<ValidationSet>,
        progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<SetupDialog> {
        let instance = Rc::new_cyclic(|this: &Weak<SetupDialog>| {
            let base = ProcessDialog::new_application_manager_axis_id_dialog_type_boolean(
                manager,
                axis_id,
                dialog_type,
                false,
            );
            // Field initialisers.
            let pnl_data_parameters = JComponent::new_panel();
            let pnl_dataset = JComponent::new_panel();
            let bctf_raw_image_stack = ButtonControlTextEfield::get_labeled_file_instance_string(
                Some("Raw Image Stack: "),
            );
            let ftf_backup_directory = FileTextField::new(BACKUP_DIRECTORY_LABEL);
            let pnl_per_axis_info = JComponent::new_panel();
            let pnl_axis_info_a = EtomoPanel::new();
            let pnl_data_type = EtomoPanel::new();
            let pnl_axis_type = EtomoPanel::new();
            let rb_single_axis = RadioButton::new_string(Some(SINGLE_AXIS_LABEL));
            let rb_dual_axis = RadioButton::new_string(Some("Dual axis"));
            let pnl_view_type = EtomoPanel::new();
            let rb_single_view = RadioButton::new_string(Some(SINGLE_FRAME_LABEL));
            let rb_montage = RadioButton::new_string(Some(MONTAGE_LABEL));
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_view_raw_stack_a =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some(VIEW_RAW_STACK_LABEL),
                    Some(container.clone()),
                );
            let btn_view_raw_stack_b =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some(VIEW_RAW_STACK_LABEL),
                    Some(container),
                );
            let pnl_image_params = JComponent::new_panel();
            let btn_scan_header = MultiLineButton::new_string(Some("Scan Header"));
            let pnl_image_rows = JComponent::new_panel();
            let pnl_stack_info = JComponent::new_panel();
            let ltf_pixel_size = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Pixel size (nm): "),
            );
            let ltf_fiducial_diameter = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(FIDUCIAL_DIAMETER_LABEL),
            );
            let ltf_image_rotation = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Image rotation (degrees): "),
            );
            let pnl_distortion_info = JComponent::new_panel();
            let ftf_distortion_file = FileTextField::new("Image distortion field file: ");
            let ltf_binning = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Binning: "),
            );
            let pnl_mag_gradient_info = JComponent::new_panel();
            let ftf_mag_gradient_file = FileTextField::new("Mag gradients correction: ");
            let cb_parallel_process = CheckBox::new_string(Some("Parallel Processing"));
            let cb_gpu_processing = CheckBox::new_string(Some("Graphics card processing"));
            let ltf_exclude_list_a = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Exclude views: "),
            );
            let pnl_adjusted_focus_a = JComponent::new_panel();
            let cb_adjusted_focus_a =
                CheckBox::new_string(Some("Focus was adjusted between montage frames"));
            let border_axis_info_b = Rc::new(BeveledBorder::new(Some("Axis B: ")));
            let ltf_exclude_list_b = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Exclude views: "),
            );
            let pnl_adjusted_focus_b = JComponent::new_panel();
            let cb_adjusted_focus_b =
                CheckBox::new_string(Some("Focus was adjusted between montage frames"));
            let cb_tf_twodir = CheckBox::new_string(Some("Series was"));
            let bg_tf_two_dir = ButtonGroup::new();
            let rb_bidirectional =
                RadioButton::new_string_button_group(Some("bidirectional"), Some(&bg_tf_two_dir));
            let rb_dose_symmetric =
                RadioButton::new_string_button_group(Some("dose-symmetric"), Some(&bg_tf_two_dir));
            let tf_twodir = TextField::new(FieldType::FloatingPoint, Some("bidirectional"), None);
            let tf_dose_sym =
                TextField::new(FieldType::FloatingPoint, Some("dose-symmetric"), None);
            let cb_tf_btwodir = CheckBox::new_string(Some("Series was"));
            let bg_tf_btwo_dir = ButtonGroup::new();
            let rb_b_bidirectional =
                RadioButton::new_string_button_group(Some("bidirectional"), Some(&bg_tf_btwo_dir));
            let rb_b_dose_symmetric =
                RadioButton::new_string_button_group(Some("dose-symmetric"), Some(&bg_tf_btwo_dir));
            let tf_btwodir = TextField::new(FieldType::FloatingPoint, Some("bidirectional"), None);
            let tf_b_dose_sym =
                TextField::new(FieldType::FloatingPoint, Some("b-dose-symmetric"), None);
            let l_twodirfrom = JComponent::new_label("from");
            let l_btwodirfrom = JComponent::new_label("from");
            let l_twodir = JComponent::new_label(TWODIR_LABEL_2);
            let l_btwodir = JComponent::new_label(TWODIR_LABEL_2);
            let cb_remove_excluded_views = CheckBox::new_string(Some("Remove excluded views"));
            let cb_delete_old_files = CheckBox::new_string(Some("Delete original files"));
            let l_remove_exclude_views_msg_a = JComponent::new_label("");
            let l_remove_exclude_views_msg_b = JComponent::new_label("");
            // HalfFloatModeOutput
            let l_half_float_mode_output = Label::new_string(Some("Output as half-size floats: "));
            let cb_half_float_mode_output = CheckBox::new_string(Some("Unconditionally"));
            let cb_half_float_mode_output_if_float =
                CheckBox::new_string(Some("Only if raw stack is floating point"));
            let l_half_float_mode_output_active = Label::new_string(Some("Using half-size floats"));

            // Constructor body: the field assignments.
            let tilt_angles_a = expert
                .get_tilt_angles_panel_expert(AxisID::First)
                .get_panel();
            tilt_angles_a.set_parent(this.clone());
            let tilt_angles_b = expert
                .get_tilt_angles_panel_expert(AxisID::Second)
                .get_panel();
            tilt_angles_b.set_parent(this.clone());
            ltf_binning.set_validation_set(Some(&*binning_validation_set));
            let listener = Rc::new(SetupDialogActionListener::new(Rc::downgrade(expert)));
            let template_listener = listener.clone();
            let template_panel = TemplatePanel::get_instance(
                manager,
                axis_id,
                Rc::new(move |event: &ActionEvent| template_listener.action_performed(event)),
                Some("Templates"),
                None,
                false,
            );
            // progressPanel = AxisProgressPanel.getInstance(axisID, manager);
            SetupDialog {
                base,
                pnl_data_parameters,
                pnl_dataset,
                bctf_raw_image_stack,
                ftf_backup_directory,
                pnl_per_axis_info,
                pnl_axis_info_a,
                pnl_data_type,
                pnl_axis_type,
                rb_single_axis,
                rb_dual_axis,
                pnl_view_type,
                rb_single_view,
                rb_montage,
                btn_view_raw_stack_a,
                btn_view_raw_stack_b,
                pnl_image_params,
                btn_scan_header,
                pnl_image_rows,
                pnl_stack_info,
                ltf_pixel_size,
                ltf_fiducial_diameter,
                ltf_image_rotation,
                pnl_distortion_info,
                ftf_distortion_file,
                ltf_binning,
                pnl_mag_gradient_info,
                ftf_mag_gradient_file,
                cb_parallel_process,
                cb_gpu_processing,
                ltf_exclude_list_a,
                pnl_adjusted_focus_a,
                cb_adjusted_focus_a,
                border_axis_info_b,
                ltf_exclude_list_b,
                pnl_adjusted_focus_b,
                cb_adjusted_focus_b,
                cb_tf_twodir,
                bg_tf_two_dir,
                rb_bidirectional,
                rb_dose_symmetric,
                tf_twodir,
                tf_dose_sym,
                cb_tf_btwodir,
                bg_tf_btwo_dir,
                rb_b_bidirectional,
                rb_b_dose_symmetric,
                tf_btwodir,
                tf_b_dose_sym,
                l_twodirfrom,
                l_btwodirfrom,
                l_twodir,
                l_btwodir,
                cb_remove_excluded_views,
                cb_delete_old_files,
                l_remove_exclude_views_msg_a,
                l_remove_exclude_views_msg_b,
                l_half_float_mode_output,
                cb_half_float_mode_output,
                cb_half_float_mode_output_if_float,
                l_half_float_mode_output_active,
                expert: Rc::downgrade(expert),
                calibration_available,
                listener,
                template_panel,
                progress_panel,
                tilt_angles_a,
                tilt_angles_b,
                directive_file_collection: RefCell::new(None),
                exclude_views_succeeded_a: Cell::new(false),
                exclude_views_succeeded_b: Cell::new(false),
                this: this.clone(),
            }
        });
        // The superclass dispatches `done` / `buttonExecuteAction` to this
        // subclass.
        instance
            .base
            .set_this(Rc::downgrade(&instance) as Weak<dyn ProcessDialogVirtual>);

        // Constructor body, continued.
        instance.progress_panel.set_visible(false);
        // Swing layout: rootPanel BoxLayout Y_AXIS.
        instance
            .bctf_raw_image_stack
            .set_file_selection_mode(file_chooser::FILES_ONLY);
        let expert_dataset_dir = expert.get_dataset_dir();
        instance
            .bctf_raw_image_stack
            .set_select_file_dir(expert_dataset_dir.as_deref());
        // Defaults to absolute path
        let base_manager: &'static dyn BaseManager = manager;
        instance.bctf_raw_image_stack.set_file_filter(Some(Rc::new(
            crate::imod::etomo::storage::stack_file_filter::StackFileFilter::get_instance(
                Some(base_manager),
                false,
            ),
        )
            as Rc<dyn crate::imod::etomo::jdk::FileFilter>));
        instance
            .bctf_raw_image_stack
            .set_limit_displayed_file_path(50);
        instance.create_dataset_panel();
        instance.create_data_type_panel();
        instance.create_per_axis_info_panel();
        instance
            .btn_view_raw_stack_a
            .set_action_command(Some(&format!(
                "{}{}",
                VIEW_RAW_STACK_LABEL,
                AxisID::First.get_extension()
            )));
        instance
            .btn_view_raw_stack_b
            .set_action_command(Some(&format!(
                "{}{}",
                VIEW_RAW_STACK_LABEL,
                AxisID::Second.get_extension()
            )));
        instance.btn_execute.set_text(Some("Create Com Scripts"));

        if calibration_available {
            // There are no advanced settings for this dialog, remove the
            // advanced button
            instance
                .pnl_exit_buttons
                .remove(&instance.btn_advanced.get_component());
        }

        // Add the panes to the dialog box
        let root_panel = instance.root_panel.get_component();
        root_panel.add(&instance.pnl_data_parameters);
        // Swing layout: vertical glue, rigid area x0_y10.
        root_panel.add(&instance.pnl_per_axis_info);
        // Swing layout: vertical glue.
        instance.add_exit_buttons();
        // Swing layout: UIUtilities.alignComponentsX(rootPanel, CENTER_ALIGNMENT).

        // Resize the standard panel buttons
        // Swing layout: UIUtilities.setButtonSizeAll(pnlExitButtons, button dimension).
        if !calibration_available {
            instance.update_advanced(instance.btn_advanced.is_expanded());
            instance
                .btn_advanced
                .register_expandable(Rc::downgrade(&instance) as Weak<dyn Expandable>);
        }
        // Calcute the necessary window size
        instance.pack_axis();
        instance
    }

    /// Java package-private static `getInstance(SetupDialogExpert,
    /// ApplicationManager, AxisID, DialogType, boolean, ValidationSet,
    /// AxisProgressPanel)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance(
        expert: &Rc<SetupDialogExpert>,
        manger: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        calibration_available: bool,
        binning_validation_set: Arc<ValidationSet>,
        progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<SetupDialog> {
        let instance = SetupDialog::new(
            expert,
            manger,
            axis_id,
            dialog_type,
            calibration_available,
            binning_validation_set,
            progress_panel,
        );
        instance.add_listeners();
        instance
    }

    /// Java `UIHarness.INSTANCE.pack(axisID, applicationManager)`.
    fn pack_axis(&self) {
        let manager: &'static dyn BaseManager = self.application_manager;
        let axis_id = self.axis_id;
        ui_harness::INSTANCE
            .with(|harness| harness.pack_axis_id_base_manager(Some(axis_id), Some(manager)));
    }

    /// Java `UIHarness.INSTANCE.pack(applicationManager)`.
    fn pack(&self) {
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::INSTANCE.with(|harness| harness.pack_base_manager(Some(manager)));
    }

    fn expert(&self) -> Option<Rc<SetupDialogExpert>> {
        self.expert.upgrade()
    }

    /// Java package-private `msgSetupReconFailed()`.
    pub fn msg_setup_recon_failed(&self) {
        if self.exclude_views_succeeded_a.get() || self.exclude_views_succeeded_b.get() {
            // `openInfoMessageDialog(manager, cbRemoveExcludedViews, ...)`: the
            // component only places the popup.
            ui_harness::open_info_message_dialog_from_process(
                Some(self.application_manager as &'static dyn BaseManager),
                "Excludeviews has run successfully.  Views have been removed.",
                "Views Excluded",
                None,
            );
        }
        self.update_display(false, true);
    }

    /// Java private `isRemoveExcludedViews()`.
    fn is_remove_excluded_views(&self) -> bool {
        self.cb_remove_excluded_views.is_enabled() && self.cb_remove_excluded_views.is_selected()
    }

    /// Java package-private `showProgressPanel()`.
    pub fn show_progress_panel(&self) {
        self.progress_panel.set_visible(true);
        self.update_display(true, false);
    }

    /// Java package-private `updateDisplay(boolean, boolean)`: enable/disable
    /// fields.  `process_done` is set when a process from this dialog has
    /// completed.
    pub fn update_display(&self, process_running: bool, process_done: bool) {
        let is_float_mode_input = || {
            self.expert()
                .is_some_and(|expert| expert.is_float_mode_input())
        };
        self.l_half_float_mode_output_active.set_visible(
            self.l_half_float_mode_output.get_component().is_enabled()
                && (self.cb_half_float_mode_output.is_selected()
                    || (self.cb_half_float_mode_output_if_float.is_selected()
                        && is_float_mode_input())),
        );

        self.cb_delete_old_files
            .set_enabled(self.cb_remove_excluded_views.is_selected());
        let enabled;
        if process_running {
            enabled = false;
        } else if process_done {
            enabled = true;
        } else {
            enabled = self.progress_panel.is_stopped();
        }
        self.btn_execute.set_enabled(enabled);
        self.btn_cancel.set_enabled(enabled);
        self.tilt_angles_a.update_display();
        self.tilt_angles_b.update_display();
        // AxisA
        self.rb_bidirectional
            .set_enabled(self.cb_tf_twodir.is_selected());
        self.rb_dose_symmetric
            .set_enabled(self.cb_tf_twodir.is_selected());
        self.tf_twodir
            .set_enabled(self.cb_tf_twodir.is_selected() && self.rb_bidirectional.is_selected());
        self.tf_twodir
            .set_visible(self.rb_bidirectional.is_selected());
        self.tf_dose_sym
            .set_enabled(self.cb_tf_twodir.is_selected() && self.rb_dose_symmetric.is_selected());
        self.tf_dose_sym
            .set_visible(self.rb_dose_symmetric.is_selected());
        self.l_twodir.set_enabled(self.cb_tf_twodir.is_selected());
        self.l_twodirfrom
            .set_enabled(self.cb_tf_twodir.is_selected());
        // AxisB
        if self.rb_dual_axis.is_selected() {
            self.rb_b_bidirectional
                .set_enabled(self.cb_tf_btwodir.is_selected());
            self.rb_b_dose_symmetric
                .set_enabled(self.cb_tf_btwodir.is_selected());
            self.tf_btwodir.set_enabled(
                self.cb_tf_btwodir.is_selected() && self.rb_b_bidirectional.is_selected(),
            );
            self.tf_btwodir
                .set_visible(self.rb_b_bidirectional.is_selected());
            self.tf_b_dose_sym.set_enabled(
                self.cb_tf_btwodir.is_selected() && self.rb_b_dose_symmetric.is_selected(),
            );
            self.tf_b_dose_sym
                .set_visible(self.rb_b_dose_symmetric.is_selected());
            self.l_btwodir.set_enabled(self.cb_tf_btwodir.is_selected());
            self.l_btwodirfrom
                .set_enabled(self.cb_tf_btwodir.is_selected());
        }

        self.pack();
    }

    /// Java package-private `getDatasetName()`: derive the dataset name from
    /// whatever is in ftfDataset (which make be a file name, a file with path,
    /// or a dataset name).
    pub fn get_dataset_name(&self) -> Option<String> {
        let dataset = self.bctf_raw_image_stack.get_text_void()?;
        let dataset = java_lang_string_trim(&dataset).to_owned();
        if java_lang_string_matches_whitespace(&dataset) {
            return None;
        }
        let mut dataset_name: Vec<char> =
            utilities::java_io_file_get_name(&dataset).chars().collect();
        // Remove the extension.
        let index = dataset_name.iter().rposition(|&c| c == '.');
        if let Some(index) = index
            && index > 1
        {
            dataset_name.truncate(index);
        }
        let length = dataset_name.len();
        let mut dataset_name: String = dataset_name.into_iter().collect();
        // Check for and remove axis extension.
        let axis_extension_len = AxisID::get_extension_length() as usize;
        // Upstream bug fixed in translation (SetupDialog.java:392-394): the
        // source's parentheses close around the first `endsWith` only, so
        // `dual && long enough && endsWith(a) || endsWith(b)` strips a
        // trailing "b" from a single-axis name, and from a name no longer than
        // the axis extension.  The test here is the one the parentheses and
        // `DatasetTool.getDatasetName` show was meant: dual axis, long enough,
        // and ending in either axis extension.
        if self.rb_dual_axis.is_selected()
            && length > axis_extension_len
            && (dataset_name.ends_with(&AxisID::First.get_extension())
                || dataset_name.ends_with(&AxisID::Second.get_extension()))
        {
            dataset_name = dataset_name
                .chars()
                .take(length - axis_extension_len)
                .collect();
        }
        Some(dataset_name)
    }

    /// Java package-private `getDirectory()`: the parent of the raw image
    /// stack's path (`File.getParentFile()`), or null.
    pub fn get_directory(&self) -> Option<String> {
        let raw_image_stack = self.bctf_raw_image_stack.get_text_void()?;
        if raw_image_stack.is_empty() {
            return None;
        }
        utilities::java_io_file_get_parent(&raw_image_stack)
    }

    /// Java `@Override actionPerformed(ActionEvent)` (ActionListener, for the
    /// two half-float check boxes).
    pub fn action_performed(&self, event: Option<&ActionEvent>) {
        let Some(event) = event else {
            return;
        };
        let action_command = event.get_action_command();
        if let Some(action_command) = action_command {
            // The two halfFloatModeOutput check box are exclusive like radio
            // buttons
            if Some(action_command)
                == self
                    .cb_half_float_mode_output
                    .get_action_command()
                    .as_deref()
                && self.cb_half_float_mode_output.is_selected()
                && self.cb_half_float_mode_output_if_float.is_selected()
            {
                self.cb_half_float_mode_output_if_float
                    .set_selected_boolean(false);
            } else if Some(action_command)
                == self
                    .cb_half_float_mode_output_if_float
                    .get_action_command()
                    .as_deref()
                && self.cb_half_float_mode_output_if_float.is_selected()
            {
                if let Some(expert) = self.expert() {
                    expert.load_header();
                }
                if self.cb_half_float_mode_output.is_selected() {
                    self.cb_half_float_mode_output.set_selected_boolean(false);
                }
            }
        }
        self.update_display(false, false);
    }

    /// Java `@Override focusGained(FocusEvent)`: listening only to the dataset
    /// focus.  Need to see if the dataset has changed.
    pub fn focus_gained(&self) {}

    /// Java `@Override focusLost(FocusEvent)`: listening only to the dataset
    /// focus.  Find out if the dataset has changed.
    pub fn focus_lost(&self) {
        self.control_event();
    }

    /// Java package-private `isRemoveExcludeViewsMsg(AxisID)`.
    pub fn is_remove_exclude_views_msg(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.l_remove_exclude_views_msg_b.get_text() != "";
        }
        self.l_remove_exclude_views_msg_a.get_text() != ""
    }

    /// Java package-private `setParameters(UserConfiguration)`.
    pub fn set_parameters(&self, user_config: &UserConfiguration) {
        self.template_panel
            .set_parameters_user_configuration(user_config);
        self.cb_remove_excluded_views
            .set_selected_boolean(user_config.is_remove_excluded_views());
        self.update_display(false, false);
    }

    /// Java package-private `checkpoint()`.
    pub fn checkpoint(&self) {
        self.bctf_raw_image_stack.checkpoint();
        self.ftf_backup_directory.checkpoint();
        self.rb_single_axis.checkpoint_void();
        self.rb_dual_axis.checkpoint_void();
        self.rb_single_view.checkpoint_void();
        self.rb_montage.checkpoint_void();
        self.ltf_pixel_size.checkpoint_void();
        self.ltf_fiducial_diameter.checkpoint_void();
        self.ltf_image_rotation.checkpoint_void();
        self.cb_half_float_mode_output.checkpoint_void();
        self.cb_half_float_mode_output_if_float.checkpoint_void();
        self.ftf_distortion_file.checkpoint();
        self.ltf_binning.checkpoint_void();
        self.ftf_mag_gradient_file.checkpoint();
        self.cb_parallel_process.checkpoint_void();
        self.cb_gpu_processing.checkpoint_void();
        self.ltf_exclude_list_a.checkpoint_void();
        self.cb_adjusted_focus_a.checkpoint_void();
        self.ltf_exclude_list_b.checkpoint_void();
        self.cb_adjusted_focus_b.checkpoint_void();
        self.tf_twodir.checkpoint_void();
        self.tf_dose_sym.checkpoint_void();
        self.tf_btwodir.checkpoint_void();
        self.tf_b_dose_sym.checkpoint_void();
    }

    /// Java package-private `updateTemplateValues()`.
    pub fn update_template_values(&self) {
        // The template panel always holds a collection in the Java (it is
        // dereferenced unchecked); with none there is nothing to apply.
        let Some(directive_file_collection) = self.template_panel.get_directive_file_collection()
        else {
            return;
        };
        let dfc = directive_file_collection.borrow();
        // Handle dual differently because the dual is the default.
        if dfc.contains(Some(DirectiveDef::DUAL)) {
            if dfc.is_value(Some(DirectiveDef::DUAL)) {
                self.rb_dual_axis.set_selected_boolean(true);
            } else {
                self.rb_single_axis.set_selected_boolean(true);
            }
        }
        if dfc.contains(Some(DirectiveDef::MONTAGE)) {
            if dfc.is_value(Some(DirectiveDef::MONTAGE)) {
                self.rb_montage.set_selected_boolean(true);
            } else {
                self.rb_single_view.set_selected_boolean(true);
            }
        }
        if dfc.contains(Some(DirectiveDef::PIXEL)) {
            self.ltf_pixel_size
                .set_text_string(dfc.get_pixel_size(false).as_deref());
        }
        let twodir_a_set;
        let mut twodir_a: Option<String> = None;
        twodir_a_set = dfc.is_twodir(Some(AxisID::First));
        if twodir_a_set {
            self.cb_tf_twodir.set_selected_boolean(true);
            twodir_a = dfc.get_twodir(Some(AxisID::First), false);
            self.tf_twodir.set_text_string(twodir_a.as_deref());
        }
        if dfc.is_twodir(Some(AxisID::Second)) {
            self.cb_tf_btwodir.set_selected_boolean(true);
            self.tf_btwodir
                .set_text_string(dfc.get_twodir(Some(AxisID::Second), false).as_deref());
        } else if twodir_a_set {
            // Use the twodir for axis B.
            self.cb_tf_btwodir.set_selected_boolean(true);
            self.tf_btwodir.set_text_string(twodir_a.as_deref());
        }
        let dose_sym_a_set;
        let mut dose_sym_a: Option<String> = None;
        dose_sym_a_set = dfc.is_dose_sym(Some(AxisID::First));
        if dose_sym_a_set {
            self.cb_tf_twodir.set_selected_boolean(true);
            dose_sym_a = dfc.get_dose_sym(Some(AxisID::First), false);
            // Upstream bug fixed in translation (SetupDialog.java:602): the
            // source sets the dose-symmetric starting angle to `twodirA`, the
            // bidirectional value read above (null unless axis A is also
            // bidirectional), instead of the `doseSymA` it has just read.  The
            // dose-symmetric value is used.
            self.tf_dose_sym.set_text_string(dose_sym_a.as_deref());
        }
        if dfc.is_dose_sym(Some(AxisID::Second)) {
            self.cb_tf_btwodir.set_selected_boolean(true);
            self.tf_b_dose_sym
                .set_text_string(dfc.get_dose_sym(Some(AxisID::Second), false).as_deref());
        } else if dose_sym_a_set {
            // Use the twodir for axis A.
            self.cb_tf_btwodir.set_selected_boolean(true);
            self.tf_b_dose_sym.set_text_string(dose_sym_a.as_deref());
        }
        if dfc.contains(Some(DirectiveDef::GOLD)) {
            self.ltf_fiducial_diameter
                .set_text_string(dfc.get_fiducial_diameter(false).as_deref());
        }
        if dfc.contains(Some(DirectiveDef::ROTATION)) {
            self.ltf_image_rotation.set_text_string(
                dfc.get_image_rotation(Some(AxisID::First), false)
                    .as_deref(),
            );
        }
        if dfc.contains(Some(DirectiveDef::HALF_FLOAT)) {
            let half_float_mode_output = dfc.get_half_float_mode_output();
            if Some(copy_tomo_coms::HALF_FLOAT) == half_float_mode_output {
                self.cb_half_float_mode_output.set_selected_boolean(true);
            } else if Some(copy_tomo_coms::HALF_FLOAT_IF_FLOAT) == half_float_mode_output {
                self.cb_half_float_mode_output_if_float
                    .set_selected_boolean(true);
            }
        }
        if let Some(expert) = self.expert() {
            expert.update_tilt_angle_panel_template_values(&dfc);
        }
        if dfc.contains(Some(DirectiveDef::DISTORT)) {
            self.ftf_distortion_file
                .set_text_string(dfc.get_distortion_file().as_deref());
        }
        if dfc.contains(Some(DirectiveDef::BINNING)) {
            self.ltf_binning
                .set_text_string(dfc.get_binning().as_deref());
        }
        if dfc.contains(Some(DirectiveDef::GRADIENT)) {
            self.ftf_mag_gradient_file
                .set_text_string(dfc.get_mag_gradient_file().as_deref());
        }
        if dfc.contains_axis(Some(DirectiveDef::FOCUS), Some(AxisID::First)) {
            self.cb_adjusted_focus_a
                .set_selected_boolean(dfc.is_adjusted_focus_selected(Some(AxisID::First)));
        }
        if dfc.contains_axis(Some(DirectiveDef::FOCUS), Some(AxisID::Second)) {
            self.cb_adjusted_focus_b
                .set_selected_boolean(dfc.is_adjusted_focus_selected(Some(AxisID::Second)));
        }
        if dfc.contains(Some(DirectiveDef::REMOVE_EXCLUDED_VIEWS)) {
            self.cb_remove_excluded_views.set_selected_boolean(
                dfc.is_value_template(Some(DirectiveDef::REMOVE_EXCLUDED_VIEWS), true),
            );
        }
        if dfc.contains(Some(DirectiveDef::DELETE_OLD_FILES)) {
            self.cb_delete_old_files.set_selected_boolean(
                dfc.is_value_template(Some(DirectiveDef::DELETE_OLD_FILES), true),
            );
        }
        drop(dfc);
        if let Some(expert) = self.expert() {
            expert.set_tilt_angle_panel_enabled(AxisID::Second, self.rb_dual_axis.is_selected());
        }
        self.update_display(false, false);
    }

    /// Java private `viewRawStackA()`.
    fn view_raw_stack_a(&self) {
        self.action(
            self.btn_view_raw_stack_a
                .get_action_command()
                .as_deref()
                .unwrap_or_default(),
            None,
            None,
        );
    }

    /// Java private `viewRawStackB()`.
    fn view_raw_stack_b(&self) {
        self.action(
            self.btn_view_raw_stack_b
                .get_action_command()
                .as_deref()
                .unwrap_or_default(),
            None,
            None,
        );
    }

    /// Java package-private `setRawImageStack(String)`.
    pub fn set_raw_image_stack(&self, input: Option<&str>) {
        self.bctf_raw_image_stack.set_text_string(input);
    }

    /// Java package-private `setParallelProcess(boolean)`.
    pub fn set_parallel_process(&self, input: bool) {
        self.cb_parallel_process.set_selected_boolean(input);
    }

    /// Java package-private `setGpuProcessingEnabled(boolean)`.
    pub fn set_gpu_processing_enabled(&self, input: bool) {
        self.cb_gpu_processing.set_enabled(input);
    }

    /// Java package-private `setGpuProcessing(boolean)`.
    pub fn set_gpu_processing(&self, input: bool) {
        self.cb_gpu_processing.set_selected_boolean(input);
    }

    /// Java package-private `setBackupDirectory(String)`.
    pub fn set_backup_directory(&self, input: Option<&str>) {
        self.ftf_backup_directory.set_text_string(input);
    }

    /// Java package-private `setDistortionFile(String)`.
    pub fn set_distortion_file(&self, input: Option<&str>) {
        self.ftf_distortion_file.set_text_string(input);
    }

    /// Java package-private `setMagGradientFile(String)`.
    pub fn set_mag_gradient_file(&self, input: Option<&str>) {
        self.ftf_mag_gradient_file.set_text_string(input);
    }

    /// Java package-private `setAdjustedFocus(AxisID, boolean)`.
    pub fn set_adjusted_focus(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.cb_adjusted_focus_b.set_selected_boolean(input);
        } else {
            self.cb_adjusted_focus_a.set_selected_boolean(input);
        }
    }

    /// Java package-private `setAxisTypeTooltip(String)`.
    pub fn set_axis_type_tooltip(&self, tooltip: &str) {
        self.pnl_axis_type
            .get_component()
            .set_tool_tip_text(Some(tooltip));
        self.rb_single_axis.set_tool_tip_text_string(Some(tooltip));
        self.rb_dual_axis.set_tool_tip_text_string(Some(tooltip));
    }

    /// Java package-private `setDistortionFileTooltip(String)`.
    pub fn set_distortion_file_tooltip(&self, tooltip: &str) {
        self.ftf_distortion_file
            .set_field_tool_tip_text(Some(tooltip));
        self.ftf_distortion_file
            .set_button_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setViewTypeTooltip(String)`.
    pub fn set_view_type_tooltip(&self, tooltip: &str) {
        self.pnl_view_type
            .get_component()
            .set_tool_tip_text(Some(tooltip));
        self.rb_single_view.set_tool_tip_text_string(Some(tooltip));
        self.rb_montage.set_tool_tip_text_string(Some(tooltip));
    }

    /// Java package-private `setPixelSizeTooltip(String)`.
    pub fn set_pixel_size_tooltip(&self, tooltip: &str) {
        self.ltf_pixel_size.set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setFiducialDiameterTooltip(String)`.
    pub fn set_fiducial_diameter_tooltip(&self, tooltip: &str) {
        self.ltf_fiducial_diameter.set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setImageRotationTooltip(String)`.
    pub fn set_image_rotation_tooltip(&self, tooltip: &str) {
        self.ltf_image_rotation.set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setHalfFloatModeOutputTooltip(String, String)`.
    pub fn set_half_float_mode_output_tooltip(&self, tooltip: &str, if_float_tooltip: &str) {
        self.cb_half_float_mode_output
            .set_tool_tip_text_string(Some(tooltip));
        self.cb_half_float_mode_output_if_float
            .set_tool_tip_text_string(Some(if_float_tooltip));
    }

    /// Java package-private `setBinningTooltip(String)`.
    pub fn set_binning_tooltip(&self, tooltip: &str) {
        self.ltf_binning.set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setViewRawStackTooltip(String)`.
    pub fn set_view_raw_stack_tooltip(&self, tooltip: &str) {
        self.btn_view_raw_stack_a.set_tool_tip_text(Some(tooltip));
        self.btn_view_raw_stack_b.set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setAdjustedFocusTooltip(String)`.
    pub fn set_adjusted_focus_tooltip(&self, tooltip: &str) {
        self.cb_adjusted_focus_a
            .set_tool_tip_text_string(Some(tooltip));
        self.cb_adjusted_focus_b
            .set_tool_tip_text_string(Some(tooltip));
    }

    /// Java package-private `setExcludeListTooltip(String)`.
    pub fn set_exclude_list_tooltip(&self, tooltip: &str) {
        self.ltf_exclude_list_a.set_tool_tip_text(Some(tooltip));
        self.ltf_exclude_list_b.set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setTwodirTooltip()`.
    pub fn set_twodir_tooltip(&self) {
        self.cb_tf_twodir.set_tool_tip_text_string(Some(
            "Tilt series was bidirectional or dose-symmetric from the given starting angle",
        ));
        self.cb_tf_btwodir.set_tool_tip_text_string(Some(
            "Tilt series was bidirectional or dose-symmetric from the given starting angle",
        ));
        self.rb_bidirectional
            .set_tool_tip_text_string(Some("Select this option for a bidirectional tilt series"));
        self.rb_b_bidirectional
            .set_tool_tip_text_string(Some("Select this option for a bidirectional tilt series"));
        self.rb_dose_symmetric
            .set_tool_tip_text_string(Some("Select this option for dose-symmetric series"));
        self.rb_b_dose_symmetric
            .set_tool_tip_text_string(Some("Select this option for dose-symmetric series"));
        self.tf_twodir.set_tool_tip_text(Some(
            "Starting angle of the tilt series; the break in the series \
             is assumed to be after the first image in the stack at that angle.",
        ));
        self.tf_btwodir.set_tool_tip_text(Some(
            "Starting angle of the tilt series; the break in the series \
             is assumed to be after the first image in the stack at that angle.",
        ));
        self.tf_dose_sym
            .set_tool_tip_text(Some("Starting angle of the tilt series"));
        self.tf_b_dose_sym
            .set_tool_tip_text(Some("Starting angle of the tilt series"));
    }

    /// Java package-private `setExecuteTooltip(String)`.
    pub fn set_execute_tooltip(&self, tooltip: &str) {
        self.btn_execute.set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setParallelProcessTooltip(String)`.
    pub fn set_parallel_process_tooltip(&self, tooltip: &str) {
        self.cb_parallel_process
            .set_tool_tip_text_string(Some(tooltip));
    }

    /// Java package-private `setGpuProcessingTooltip(String)`.
    pub fn set_gpu_processing_tooltip(&self, tooltip: &str) {
        self.cb_gpu_processing
            .set_tool_tip_text_string(Some(tooltip));
    }

    /// Java package-private `setMagGradientFileTooltip(String)`.
    pub fn set_mag_gradient_file_tooltip(&self, tooltip: &str) {
        self.ftf_mag_gradient_file
            .set_field_tool_tip_text(Some(tooltip));
        self.ftf_mag_gradient_file
            .set_button_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setTooltips()`.
    pub fn set_tooltips(&self) {
        self.cb_remove_excluded_views.set_tool_tip_text_string(Some(
            "Make a new stack with excluded views removed before running copytomocoms",
        ));
        self.cb_delete_old_files.set_tool_tip_text_string(Some(
            "delete original file and keep excluded views, which can be used to restore the \
             original file.",
        ));
        self.template_panel.set_scope_tooltip(Some(
            "Select the first system-wide template file from which parameters will be set.",
        ));
        self.template_panel.set_system_tooltip(Some(
            "Select the second system-wide template file from which parameters will be set.",
        ));
        self.template_panel.set_user_tooltip(Some(
            "Select a personal template file from which parameters will be set.",
        ));
    }

    /// Java package-private `setSingleAxis(boolean)`.
    pub fn set_single_axis(&self, input: bool) {
        self.rb_single_axis.set_selected_boolean(input);
    }

    /// Java package-private `getAxisType()`.
    pub fn get_axis_type(&self) -> AxisType {
        if self.rb_dual_axis.is_selected() {
            AxisType::DualAxis
        } else {
            AxisType::SingleAxis
        }
    }

    /// Java package-private `setDualAxis(boolean)`.
    pub fn set_dual_axis(&self, input: bool) {
        self.rb_dual_axis.set_selected_boolean(input);
    }

    /// Java package-private `setFiducialDiameter(double)`.
    pub fn set_fiducial_diameter(&self, input: f64) {
        self.ltf_fiducial_diameter.set_text_double(input);
    }

    /// Java package-private `setImageRotation(double)`.  The class also has
    /// `setImageRotation(String)` (`SetupReconInterface`), so both carry
    /// their parameter type.
    pub fn set_image_rotation_double(&self, input: f64) {
        self.ltf_image_rotation.set_text_double(input);
    }

    /// Java package-private `setExcludeList(AxisID, String)`.
    pub fn set_exclude_list(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.ltf_exclude_list_b.set_text_string(input);
        } else {
            self.ltf_exclude_list_a.set_text_string(input);
        }
    }

    /// Java package-private `setTwodir(AxisID, boolean)`.
    pub fn set_twodir_axis_id_boolean(&self, axis_id: AxisID, selected: bool) {
        if axis_id == AxisID::Second {
            self.cb_tf_btwodir.set_selected_boolean(selected);
            self.rb_b_bidirectional.set_selected_boolean(selected);
        } else {
            self.cb_tf_twodir.set_selected_boolean(selected);
            self.rb_bidirectional.set_selected_boolean(selected);
        }
        self.update_display(false, false);
    }

    /// Java package-private `setTwodir(AxisID, String)`.
    pub fn set_twodir_axis_id_string(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.tf_btwodir.set_text_string(input);
        } else {
            self.tf_twodir.set_text_string(input);
        }
    }

    /// Java package-private `setDoseSym(AxisID, String)`.
    pub fn set_dose_sym_axis_id_string(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.tf_b_dose_sym.set_text_string(input);
        } else {
            self.tf_dose_sym.set_text_string(input);
        }
    }

    /// Java package-private `setDoseSym(AxisID, boolean)`.
    pub fn set_dose_sym_axis_id_boolean(&self, axis_id: AxisID, selected: bool) {
        if axis_id == AxisID::Second {
            self.cb_tf_btwodir.set_selected_boolean(selected);
            self.rb_b_dose_symmetric.set_selected_boolean(selected);
        } else {
            self.cb_tf_twodir.set_selected_boolean(selected);
            self.rb_dose_symmetric.set_selected_boolean(selected);
        }
        self.update_display(false, false);
    }

    /// Java public `setTwodir(AxisID, double)` (`SetupReconInterface`).
    pub fn set_twodir_axis_id_double(&self, axis_id: AxisID, input: f64) {
        if axis_id == AxisID::Second {
            self.cb_tf_btwodir.set_selected_boolean(true);
            self.rb_b_bidirectional.set_selected_boolean(true);
            self.tf_btwodir
                .set_text_string(Some(&java_lang_double_to_string(input)));
        } else {
            self.cb_tf_twodir.set_selected_boolean(true);
            self.rb_bidirectional.set_selected_boolean(true);
            self.tf_twodir
                .set_text_string(Some(&java_lang_double_to_string(input)));
        }
        self.update_display(false, false);
    }

    /// Java package-private `setExcludeListEnabled(AxisID, boolean)`.
    pub fn set_exclude_list_enabled(&self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.ltf_exclude_list_b.set_enabled(enable);
        } else {
            self.ltf_exclude_list_a.set_enabled(enable);
        }
    }

    /// Java package-private `setTwodirEnabled(AxisID, boolean)`.
    pub fn set_twodir_enabled(&self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.cb_tf_btwodir.set_enabled(enable);
            self.rb_b_bidirectional.set_enabled(enable);
            self.l_btwodirfrom.set_enabled(enable);
            self.tf_btwodir.set_enabled(enable);
            self.l_btwodir.set_enabled(enable);
        } else {
            self.cb_tf_twodir.set_enabled(enable);
            self.rb_bidirectional.set_enabled(enable);
            self.l_twodirfrom.set_enabled(enable);
            self.tf_twodir.set_enabled(enable);
            self.l_twodir.set_enabled(enable);
        }
    }

    /// Java package-private `setDoseSymEnabled(AxisID, boolean)`.
    pub fn set_dose_sym_enabled(&self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.rb_b_dose_symmetric.set_enabled(enable);
            self.tf_b_dose_sym.set_enabled(enable);
        } else {
            self.rb_dose_symmetric.set_enabled(enable);
            self.tf_dose_sym.set_enabled(enable);
        }
    }

    /// Java package-private `setViewRawStackEnabled(AxisID, boolean)`.
    pub fn set_view_raw_stack_enabled(&self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.btn_view_raw_stack_b.set_enabled(enable);
        } else {
            self.btn_view_raw_stack_a.set_enabled(enable);
        }
    }

    /// Java package-private `setSingleView(boolean)`.
    pub fn set_single_view(&self, input: bool) {
        self.rb_single_view.set_selected_boolean(input);
    }

    /// Java package-private `setMontage(boolean)`.
    pub fn set_montage(&self, input: bool) {
        self.rb_montage.set_selected_boolean(input);
    }

    /// Java package-private `getViewsToSkip(AxisID, boolean)`.
    pub fn get_views_to_skip(&self, axis_id: AxisID, do_validation: bool) -> Option<String> {
        let result = if axis_id == AxisID::Second {
            self.ltf_exclude_list_b.get_text_boolean(do_validation)
        } else {
            self.ltf_exclude_list_a.get_text_boolean(do_validation)
        };
        match result {
            Ok(text) => text,
            // catch (final FieldValidationFailedException e)
            Err(_) => None,
        }
    }

    /// Java package-private `equalsSingleAxisActionCommand(String)`.
    pub fn equals_single_axis_action_command(&self, action_command: &str) -> bool {
        Some(action_command) == self.rb_single_axis.get_action_command().as_deref()
    }

    /// Java package-private `equalsDualAxisActionCommand(String)`.
    pub fn equals_dual_axis_action_command(&self, action_command: &str) -> bool {
        Some(action_command) == self.rb_dual_axis.get_action_command().as_deref()
    }

    /// Java package-private `equalsSingleViewActionCommand(String)`.
    pub fn equals_single_view_action_command(&self, action_command: &str) -> bool {
        Some(action_command) == self.rb_single_view.get_action_command().as_deref()
    }

    /// Java package-private `setAdjustedFocusEnabled(AxisID, boolean)`.
    pub fn set_adjusted_focus_enabled(&self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.cb_adjusted_focus_b.set_enabled(enable);
        } else {
            self.cb_adjusted_focus_a.set_enabled(enable);
        }
    }

    /// Java package-private `equalsMontageActionCommand(String)`.
    pub fn equals_montage_action_command(&self, action_command: &str) -> bool {
        Some(action_command) == self.rb_montage.get_action_command().as_deref()
    }

    /// Java package-private `equalsScanHeaderActionCommand(String)`.
    pub fn equals_scan_header_action_command(&self, action_command: &str) -> bool {
        Some(action_command) == self.btn_scan_header.get_action_command().as_deref()
    }

    /// Java package-private `equalsTemplateActionCommand(String)`.
    pub fn equals_template_action_command(&self, action_command: &str) -> bool {
        self.template_panel
            .equals_action_command(Some(action_command))
    }

    /// Java package-private `updateAdvanced(boolean)`.
    pub fn update_advanced(&self, advanced: bool) {
        // `pnlDistortionInfo == null` can never be true (a final field); only
        // `calibrationAvailable` is tested.
        if self.calibration_available {
            return;
        }
        self.ftf_distortion_file.set_visible(advanced);
        self.ltf_binning.set_visible(advanced);
        self.pnl_mag_gradient_info.set_visible(advanced);
    }

    /// Java package-private `setMagGradientInfoVisible(boolean)` (empty in the
    /// Java).
    pub fn set_mag_gradient_info_visible(&self, _visible: bool) {}

    /// Java package-private `getFile(String, FileFilter, int)`: runs a file
    /// chooser in `dir` and returns the chosen file, or null.
    pub fn get_file(
        &self,
        dir: Option<&str>,
        file_filter: Option<Rc<dyn crate::imod::etomo::jdk::FileFilter>>,
        selection_mode: i32,
    ) -> Option<PathBuf> {
        let chooser = FileChooser::new_base_manager_string(
            Some(self.application_manager as &'static dyn BaseManager),
            dir,
        );
        if let Some(file_filter) = file_filter {
            chooser.set_file_filter(Some(file_filter));
        }
        // Swing layout: chooser.setPreferredSize(FixedDim.fileChooser).
        chooser.set_file_selection_mode(selection_mode);
        let return_val = chooser.show_open_dialog(Some(&self.root_panel.get_component()));
        if return_val == file_chooser::APPROVE_OPTION {
            return chooser.get_selected_file();
        }
        None
    }

    /// Java package-private `backupDirectoryAction()`.
    pub fn backup_directory_action(&self) {
        // try { ... } catch (final Exception excep) { excep.printStackTrace(); }
        let current_backup_directory = self
            .expert()
            .and_then(|expert| expert.get_current_backup_directory());
        let file = self.get_file(
            current_backup_directory.as_deref(),
            None,
            file_chooser::DIRECTORIES_ONLY,
        );
        if let Some(file) = file {
            // `File.getCanonicalPath` throws `IOException`, caught above.
            match std::fs::canonicalize(&file) {
                Ok(canonical_path) => self
                    .ftf_backup_directory
                    .set_text_string(Some(&canonical_path.to_string_lossy())),
                Err(excep) => eprintln!("{excep}"),
            }
        }
    }

    /// Java package-private `distortionFileAction()`.
    pub fn distortion_file_action(&self) {
        // try { ... } catch (final Exception excep) { excep.printStackTrace(); }
        let distortion_dir = crate::imod::etomo::logic::config_tool::get_distortion_dir(
            self.application_manager,
            self.ftf_distortion_file.get_file().as_deref(),
        );
        let file = self.get_file(
            distortion_dir.as_deref(),
            Some(Rc::new(
                crate::imod::etomo::storage::distortion_file_filter::DistortionFileFilter::new(),
            ) as Rc<dyn crate::imod::etomo::jdk::FileFilter>),
            file_chooser::FILES_ONLY,
        );
        if let Some(file) = file {
            self.ftf_distortion_file.set_text_string(Some(
                &utilities::java_io_file_get_absolute_path(&file.to_string_lossy()),
            ));
        }
    }

    /// Java package-private `magGradientFileAction()`: lets the user choose
    /// the mag gradients correction file.
    pub fn mag_gradient_file_action(&self) {
        // try { ... } catch (final Exception excep) { excep.printStackTrace(); }
        let current_mag_gradient_dir = self
            .expert()
            .and_then(|expert| expert.get_current_mag_gradient_dir());
        let file = self.get_file(
            current_mag_gradient_dir.as_deref(),
            Some(Rc::new(
                crate::imod::etomo::storage::mag_gradient_file_filter::MagGradientFileFilter::new(),
            ) as Rc<dyn crate::imod::etomo::jdk::FileFilter>),
            file_chooser::FILES_ONLY,
        );
        if let Some(file) = file {
            self.ftf_mag_gradient_file.set_text_string(Some(
                &utilities::java_io_file_get_absolute_path(&file.to_string_lossy()),
            ));
        }
    }

    /// Java package-private `setRawImageStackTooltip(String, String)`.
    pub fn set_raw_image_stack_tooltip(&self, field_tooltip: &str, button_tooltip: &str) {
        self.bctf_raw_image_stack
            .set_tooltip_string_string(Some(field_tooltip), Some(button_tooltip));
    }

    /// Java package-private `setBackupDirectoryTooltip(String, String)`.
    pub fn set_backup_directory_tooltip(&self, field_tooltip: &str, button_tooltip: &str) {
        self.ftf_backup_directory
            .set_field_tool_tip_text(Some(field_tooltip));
        self.ftf_backup_directory
            .set_button_tool_tip_text(Some(button_tooltip));
    }

    /// Java package-private `setScanHeaderTooltip(String)`.
    pub fn set_scan_header_tooltip(&self, tooltip: &str) {
        self.btn_scan_header.set_tool_tip_text(Some(tooltip));
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // Mouse adapter for context menu
        let context_menu: Weak<dyn ContextMenu> = self.this.clone();
        self.root_panel
            .get_component()
            .add_mouse_listener(GenericMouseAdapter::new(context_menu));
        let this = self.this.clone();
        // BackupDirectoryActionListener
        self.ftf_backup_directory
            .add_action_listener(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = this.upgrade() {
                    adaptee.backup_directory_action();
                }
            }));
        let this = self.this.clone();
        // DistortionFileActionListener
        self.ftf_distortion_file
            .add_action_listener(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = this.upgrade() {
                    adaptee.distortion_file_action();
                }
            }));
        let this = self.this.clone();
        // MagGradientFileActionListener
        self.ftf_mag_gradient_file
            .add_action_listener(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = this.upgrade() {
                    adaptee.mag_gradient_file_action();
                }
            }));
        let this = self.this.clone();
        // ViewRawStackAActionListener
        self.btn_view_raw_stack_a
            .add_action_listener(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = this.upgrade() {
                    adaptee.view_raw_stack_a();
                }
            }));
        let this = self.this.clone();
        // ViewRawStackBActionListener
        self.btn_view_raw_stack_b
            .add_action_listener(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = this.upgrade() {
                    adaptee.view_raw_stack_b();
                }
            }));
        // `listener` (SetupDialogActionListener) on the radio buttons, the scan
        // header button and the check boxes.
        let listener = &self.listener;
        self.rb_single_axis
            .add_action_listener(listener.as_action_listener());
        self.rb_dual_axis
            .add_action_listener(listener.as_action_listener());
        self.rb_single_view
            .add_action_listener(listener.as_action_listener());
        self.rb_montage
            .add_action_listener(listener.as_action_listener());
        self.btn_scan_header
            .add_action_listener(listener.as_action_listener());
        self.cb_remove_excluded_views
            .add_action_listener(Some(listener.as_action_listener()));
        self.bctf_raw_image_stack.add_control_listener(
            self.this
                .upgrade()
                .map(|this| this as Rc<dyn ControlListener>),
        );
        let this = self.this.clone();
        self.bctf_raw_image_stack
            .add_focus_listener(Rc::new(move |event: &FocusEvent| {
                if let Some(this) = this.upgrade() {
                    if event.gained {
                        this.focus_gained();
                    } else {
                        this.focus_lost();
                    }
                }
            }));
        self.cb_tf_twodir
            .add_action_listener(Some(listener.as_action_listener()));
        self.cb_tf_btwodir
            .add_action_listener(Some(listener.as_action_listener()));
        self.rb_bidirectional
            .add_action_listener(listener.as_action_listener());
        self.rb_b_bidirectional
            .add_action_listener(listener.as_action_listener());
        self.rb_dose_symmetric
            .add_action_listener(listener.as_action_listener());
        self.rb_b_dose_symmetric
            .add_action_listener(listener.as_action_listener());
        // `addActionListener(this)` for the two half-float check boxes.
        let this = self.this.clone();
        self.cb_half_float_mode_output
            .add_action_listener(Some(Rc::new(move |event: &ActionEvent| {
                if let Some(this) = this.upgrade() {
                    this.action_performed(Some(event));
                }
            })));
        let this = self.this.clone();
        self.cb_half_float_mode_output_if_float
            .add_action_listener(Some(Rc::new(move |event: &ActionEvent| {
                if let Some(this) = this.upgrade() {
                    this.action_performed(Some(event));
                }
            })));
    }

    /// Java private `createDatasetPanel()`.
    fn create_dataset_panel(&self) {
        // Swing layout: pnlDataset BoxLayout X_AXIS.

        // Bind the buttons to their adapters

        // Add the GUI objects to the pnl
        // Swing layout: rigid area x5_y0.

        self.pnl_dataset
            .add(&self.bctf_raw_image_stack.get_component());
        // Swing layout: rigid area x10_y0.

        self.pnl_dataset
            .add(&self.ftf_backup_directory.get_container());
        // Swing layout: rigid area x5_y0.
    }

    /// Java private `createDataTypePanel()`.
    fn create_data_type_panel(&self) {
        let pnl_row2 = JComponent::new_panel();
        let pnl_remove_excluded_views = JComponent::new_panel();
        let pnl_half_float_mode_output = JComponent::new_panel();
        // init
        // Swing layout: ftfDistortionFile / ftfMagGradientFile
        // setTextPreferredWidth(505).
        // Datatype subpnls: DataSource AxisType Viewtype
        // Swing layout: dimDataTypePref = 150 x 80 scaled by the font size
        // adjustment.

        let enable_half_float_mode_output = self
            .application_manager
            .get_meta_data()
            .get_image_filename_style()
            != ImageFilenameStyle::Hdf;
        self.l_half_float_mode_output
            .get_component()
            .set_enabled(enable_half_float_mode_output);
        self.cb_half_float_mode_output
            .set_enabled(enable_half_float_mode_output);
        self.cb_half_float_mode_output_if_float
            .set_enabled(enable_half_float_mode_output);
        self.l_half_float_mode_output_active
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_half_float_mode_output_active.set_visible(false);

        let bg_axis_type = ButtonGroup::new();
        bg_axis_type.add(&self.rb_single_axis.get_abstract_button());
        bg_axis_type.add(&self.rb_dual_axis.get_abstract_button());
        // Swing layout: pnlAxisType BoxLayout Y_AXIS, preferred size dimDataTypePref.
        self.pnl_axis_type
            .set_border(&EtchedBorder::new(Some(AXIS_TYPE_LABEL)).get_border());
        let pnl_axis_type = self.pnl_axis_type.get_component();
        pnl_axis_type.add(&self.rb_single_axis.get_component());
        pnl_axis_type.add(&self.rb_dual_axis.get_component());
        let bg_view_type = ButtonGroup::new();
        bg_view_type.add(&self.rb_single_view.get_abstract_button());
        bg_view_type.add(&self.rb_montage.get_abstract_button());
        // Swing layout: pnlViewType BoxLayout Y_AXIS, preferred size dimDataTypePref.
        self.pnl_view_type
            .set_border(&EtchedBorder::new(Some(FRAME_TYPE_LABEL)).get_border());
        let pnl_view_type = self.pnl_view_type.get_component();
        pnl_view_type.add(&self.rb_single_view.get_component());
        pnl_view_type.add(&self.rb_montage.get_component());

        // Datatype panel
        // Swing layout: pnlDataType BoxLayout X_AXIS.
        self.pnl_data_type
            .set_border(&EtchedBorder::new(Some("Data Type")).get_border());
        let pnl_data_type = self.pnl_data_type.get_component();
        pnl_data_type.add(&pnl_axis_type);
        // Swing layout: horizontal glue.
        pnl_data_type.add(&pnl_view_type);
        // Swing layout: horizontal glue.

        // Pixel & Alignment panel
        // Swing layout: ltfPixelSize.setColumns(8).
        self.ltf_pixel_size.set_required(true);
        // Swing layout: ltfFiducialDiameter.setColumns(5), ltfImageRotation.setColumns(5).
        // ltfBinning.setTextMaxmimumSize(UIParameters.getInstance().getSpinnerDimension());

        // Swing layout: pnlStackInfo BoxLayout X_AXIS; btnScanHeader
        // setAlignmentY(CENTER_ALIGNMENT); rigid area x5_y0, horizontal glue.
        self.pnl_stack_info
            .add(&self.btn_scan_header.get_component());
        // Swing layout: rigid areas x10_y0, x10_y0.
        self.pnl_stack_info
            .add(&self.ltf_pixel_size.get_container());
        // Swing layout: rigid area x10_y0, horizontal glue.
        self.pnl_stack_info
            .add(&self.ltf_fiducial_diameter.get_container());
        // Swing layout: rigid area x10_y0, horizontal glue.
        self.pnl_stack_info
            .add(&self.ltf_image_rotation.get_container());
        // Swing layout: rigid area x10_y0, horizontal glue, rigid area x5_y0.

        // Swing layout: pnlDistortionInfo BoxLayout X_AXIS; rigid area x10_y0.
        self.pnl_distortion_info
            .add(&self.ftf_distortion_file.get_container());
        // Swing layout: rigid area x10_y0.
        self.pnl_distortion_info
            .add(&self.ltf_binning.get_container());
        // Swing layout: rigid area x5_y0.

        // Swing layout: pnlMagGradientInfo BoxLayout X_AXIS; rigid area x10_y0.
        self.pnl_mag_gradient_info
            .add(&self.ftf_mag_gradient_file.get_container());
        // Swing layout: rigid area x119_y0.

        let pnl_parallel_process = JComponent::new_panel();
        // Swing layout: pnlParallelProcess BoxLayout X_AXIS; rigid area x5_y0.
        pnl_parallel_process.add(&self.cb_parallel_process.get_component());
        // Swing layout: rigid area x15_y0.
        pnl_parallel_process.add(&self.cb_gpu_processing.get_component());

        // Swing layout: pnlImageRows BoxLayout Y_AXIS.
        self.pnl_image_rows.add(&self.pnl_stack_info);
        // Swing layout: rigid area x0_y15.
        self.pnl_image_rows.add(&pnl_half_float_mode_output);
        // Swing layout: rigid area x0_y15.
        self.pnl_image_rows.add(&pnl_parallel_process);
        // Swing layout: rigid area x0_y5.
        self.pnl_image_rows.add(&self.pnl_distortion_info);
        // Swing layout: rigid area x0_y5.
        self.pnl_image_rows.add(&self.pnl_mag_gradient_info);
        self.pnl_image_rows.add(&pnl_remove_excluded_views);
        // Swing layout: UIUtilities.alignComponentsX(pnlImageRows, LEFT_ALIGNMENT).

        // HalfFloatModeOutput
        // Swing layout: pnlHalfFloatModeOutput BoxLayout X_AXIS; horizontal strut 8.
        pnl_half_float_mode_output.add(&self.l_half_float_mode_output.get_component());
        // Swing layout: horizontal strut 2.
        pnl_half_float_mode_output.add(&self.cb_half_float_mode_output.get_component());
        // Swing layout: horizontal strut 9.
        pnl_half_float_mode_output.add(&self.cb_half_float_mode_output_if_float.get_component());
        // Swing layout: horizontal strut 14.
        pnl_half_float_mode_output.add(&self.l_half_float_mode_output_active.get_component());

        // Swing layout: pnlImageRows setAlignmentY(CENTER_ALIGNMENT);
        // pnlImageParams BoxLayout X_AXIS.

        self.pnl_image_params.add(&self.pnl_image_rows);
        // Swing layout: horizontal glue, rigid area x5_y0.

        // Swing layout: pnlRow2 BoxLayout X_AXIS.
        pnl_row2.add(&self.template_panel.get_component());
        // Swing layout: rigid area x2_y0.
        pnl_row2.add(&pnl_data_type);

        // Create Data Parameters panel
        // Swing layout: pnlDataParameters BoxLayout Y_AXIS.
        self.pnl_data_parameters
            .add(&self.progress_panel.get_component());
        // Swing layout: rigid area x0_y10.
        self.pnl_data_parameters.add(&self.pnl_dataset);
        // Swing layout: rigid area x0_y10.
        self.pnl_data_parameters.add(&pnl_row2);
        // Swing layout: rigid area x0_y10.
        self.pnl_data_parameters.add(&self.pnl_image_params);
        // Swing layout: rigid area x0_y10.

        // RemoveExcludedViews
        // Swing layout: pnlRemoveExcludedViews BoxLayout X_AXIS.
        pnl_remove_excluded_views.add(&self.cb_remove_excluded_views.get_component());
        // Swing layout: rigid area x10_y0.
        pnl_remove_excluded_views.add(&self.cb_delete_old_files.get_component());
    }

    /// Java private `createPerAxisInfoPanel()`.
    fn create_per_axis_info_panel(&self) {
        // constructors
        let pnl_twodir = JComponent::new_panel();
        let pnl_btwodir = JComponent::new_panel();
        // init
        self.ltf_fiducial_diameter.set_required(true);
        self.tf_twodir.set_columns_void();
        self.tf_twodir.set_required(true);
        self.tf_twodir.set_enabled(false);
        self.tf_dose_sym.set_columns_void();
        self.tf_dose_sym.set_required(true);
        self.tf_dose_sym.set_enabled(false);
        self.tf_btwodir.set_columns_void();
        self.tf_btwodir.set_required(true);
        self.tf_btwodir.set_enabled(false);
        self.tf_b_dose_sym.set_columns_void();
        self.tf_b_dose_sym.set_required(true);
        self.tf_b_dose_sym.set_enabled(false);
        self.cb_tf_twodir.set_selected_boolean(false);
        self.cb_tf_btwodir.set_selected_boolean(false);
        self.rb_bidirectional.set_selected_boolean(true);
        self.rb_bidirectional.set_enabled(false);
        self.rb_b_bidirectional.set_selected_boolean(true);
        self.rb_b_bidirectional.set_enabled(false);
        self.rb_dose_symmetric.set_enabled(false);
        self.rb_dose_symmetric.set_selected_boolean(false);
        self.rb_b_dose_symmetric.set_enabled(false);
        self.rb_b_dose_symmetric.set_selected_boolean(false);
        self.tf_dose_sym.set_visible(false);
        self.tf_b_dose_sym.set_visible(false);
        self.l_remove_exclude_views_msg_a
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_remove_exclude_views_msg_b
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        // Tilt angle specification panels
        self.pnl_axis_info_a
            .set_border(&BeveledBorder::new(Some("Axis A: ")).get_border());
        // Swing layout: pnlAxisInfoA BoxLayout Y_AXIS; ltfExcludeListA,
        // btnViewRawStackA and lRemoveExcludeViewsMsgA
        // setAlignmentX(CENTER_ALIGNMENT).

        let pnl_axis_info_a = self.pnl_axis_info_a.get_component();
        pnl_axis_info_a.add(&self.tilt_angles_a.get_component());
        // Swing layout: rigid area x0_y5.
        pnl_axis_info_a.add(&pnl_twodir);
        // Swing layout: rigid area x0_y3.
        pnl_axis_info_a.add(&self.ltf_exclude_list_a.get_container());
        pnl_axis_info_a.add(&self.l_remove_exclude_views_msg_a);
        // Swing layout: rigid area x0_y10.
        pnl_axis_info_a.add(&self.btn_view_raw_stack_a.get_component());
        // Add adjusted focus checkbox
        // Swing layout: pnlAdjustedFocusA BoxLayout X_AXIS, setAlignmentX(CENTER_ALIGNMENT).
        self.pnl_adjusted_focus_a
            .add(&self.cb_adjusted_focus_a.get_component());
        // Swing layout: horizontal glue; cbAdjustedFocusA setAlignmentX(RIGHT_ALIGNMENT).
        self.cb_adjusted_focus_a.set_enabled(false);
        pnl_axis_info_a.add(&self.pnl_adjusted_focus_a);

        let pnl_axis_info_b_panel = EtomoPanel::new();
        pnl_axis_info_b_panel.set_border(&self.border_axis_info_b.get_border());
        // Swing layout: pnlAxisInfoB BoxLayout Y_AXIS; ltfExcludeListB,
        // btnViewRawStackB and lRemoveExcludeViewsMsgB
        // setAlignmentX(CENTER_ALIGNMENT).
        let pnl_axis_info_b = pnl_axis_info_b_panel.get_component();
        pnl_axis_info_b.add(&self.tilt_angles_b.get_component());
        // Swing layout: rigid area x0_y5.
        pnl_axis_info_b.add(&pnl_btwodir);
        // Swing layout: rigid area x0_y3.
        pnl_axis_info_b.add(&self.ltf_exclude_list_b.get_container());
        pnl_axis_info_b.add(&self.l_remove_exclude_views_msg_b);
        // Swing layout: rigid area x0_y10.
        pnl_axis_info_b.add(&self.btn_view_raw_stack_b.get_component());
        // Swing layout: cbAdjustedFocusB setAlignmentX(RIGHT_ALIGNMENT).
        // Add adjusted focus checkbox
        // Swing layout: pnlAdjustedFocusB BoxLayout X_AXIS, setAlignmentX(CENTER_ALIGNMENT).
        self.pnl_adjusted_focus_b
            .add(&self.cb_adjusted_focus_b.get_component());
        // Swing layout: horizontal glue; cbAdjustedFocusB setAlignmentX(RIGHT_ALIGNMENT).
        self.cb_adjusted_focus_b.set_enabled(false);
        pnl_axis_info_b.add(&self.pnl_adjusted_focus_b);

        // Swing layout: pnlPerAxisInfo BoxLayout X_AXIS.
        self.pnl_per_axis_info.add(&pnl_axis_info_a);
        self.pnl_per_axis_info.add(&pnl_axis_info_b);
        // twodir
        // Swing layout: pnlTwodir BoxLayout X_AXIS.
        pnl_twodir.add(&self.cb_tf_twodir.get_component());
        pnl_twodir.add(&self.rb_bidirectional.get_component());
        pnl_twodir.add(&self.rb_dose_symmetric.get_component());
        // Swing layout: rigid area x10_y0.
        pnl_twodir.add(&self.l_twodirfrom);
        // Swing layout: rigid area x5_y0.
        pnl_twodir.add(&self.tf_twodir.get_component());
        pnl_twodir.add(&self.tf_dose_sym.get_component());
        pnl_twodir.add(&self.l_twodir);
        // Swing layout: pnlBtwodir BoxLayout X_AXIS.
        pnl_btwodir.add(&self.cb_tf_btwodir.get_component());
        pnl_btwodir.add(&self.rb_b_bidirectional.get_component());
        pnl_btwodir.add(&self.rb_b_dose_symmetric.get_component());
        // Swing layout: rigid area x10_y0.
        pnl_btwodir.add(&self.l_btwodirfrom);
        // Swing layout: rigid area x5_y0.
        pnl_btwodir.add(&self.tf_btwodir.get_component());
        pnl_btwodir.add(&self.tf_b_dose_sym.get_component());
        pnl_btwodir.add(&self.l_btwodir);

        self.update_display(false, false);
    }
}

/// Java private static final `SetupDialogActionListener implements
/// TemplateActionListener` (an `ActionListener`).
pub struct SetupDialogActionListener {
    /// Java private final `adaptee`.  Weak: the expert owns the dialog that
    /// owns this listener.
    adaptee: Weak<SetupDialogExpert>,
}

impl SetupDialogActionListener {
    /// Java private `SetupDialogActionListener(SetupDialogExpert)`.
    fn new(adaptee: Weak<SetupDialogExpert>) -> SetupDialogActionListener {
        SetupDialogActionListener { adaptee }
    }

    /// Java `@Override actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: &ActionEvent) {
        let Some(adaptee) = self.adaptee.upgrade() else {
            return;
        };
        adaptee.action(event.get_action_command().unwrap_or_default());
    }

    /// Rust-only: this one listener object registered on a component, as the
    /// Java registers the same instance on many (`jdk::ActionListener` is a
    /// closure).
    fn as_action_listener(self: &Rc<Self>) -> crate::imod::etomo::jdk::ActionListener {
        let listener = self.clone();
        Rc::new(move |event: &ActionEvent| listener.action_performed(event))
    }
}

impl ProcessDialogVirtual for SetupDialog {
    /// Rust-only: the `ProcessDialog` part of this dialog.
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java `@Override done()`.
    fn done(&self) {
        let mut dataset_dir_string: Option<String> = None;
        let dataset_dir = self.bctf_raw_image_stack.get_file();
        if let Some(dataset_dir) = dataset_dir {
            dataset_dir_string = utilities::java_io_file_get_parent(&dataset_dir.to_string_lossy());
        }
        let remove_excluded_views = self.is_remove_excluded_views();
        self.exclude_views_succeeded_a.set(false);
        self.exclude_views_succeeded_b.set(false);

        self.application_manager.done_setup_dialog(
            remove_excluded_views,
            remove_excluded_views,
            dataset_dir_string.as_deref(),
            self.rb_dual_axis.is_selected(),
            Some(self.progress_panel.clone()),
        );
    }

    /// Java `@Override buttonExecuteAction()`.
    fn button_execute_action(&self) -> bool {
        // Java `getText().indexOf(...)` would throw on null; a null text is
        // treated as empty.
        let raw_image_stack = self
            .bctf_raw_image_stack
            .get_text_void()
            .unwrap_or_default();
        let axis_type = self.expert().and_then(|expert| expert.get_axis_type());
        if raw_image_stack.contains(std::path::MAIN_SEPARATOR) {
            let file = self.bctf_raw_image_stack.get_file();
            if !dataset_tool::validate_dataset_name_input_file(
                self.application_manager,
                AxisID::Only,
                file.as_deref(),
                DataFileType::Recon,
                axis_type,
            ) {
                return false;
            }
        } else {
            let dataset_name = self
                .bctf_raw_image_stack
                .get_text_void()
                .unwrap_or_default();
            let property_user_dir = self
                .expert()
                .and_then(|expert| expert.get_property_user_dir());
            // `getBaseMetaData().getRawImageStackExtension().toString()`.
            // Upstream bug fixed in translation (SetupDialog.java:425): the
            // source throws a NullPointerException when no raw image stack
            // extension is set.  A name cannot end with an extension that is
            // not set, so the entry is treated as a dataset name.
            // The base metadata of an ApplicationManager is its `MetaData`,
            // whose `getRawImageStackExtension` override is the one Java
            // dispatches to.
            let raw_image_stack_extension = self
                .application_manager
                .get_meta_data()
                .get_raw_image_stack_extension()
                .map(|extension| extension.to_string());
            let is_dataset_name = !raw_image_stack_extension
                .is_some_and(|extension| dataset_name.ends_with(&extension));
            // `new File(expert.getPropertyUserDir())`: a null directory would
            // throw in `new File`; "null" is used as Java string conversion
            // would print it.
            let directory = property_user_dir.unwrap_or_else(|| "null".to_owned());
            if !dataset_tool::validate_dataset_name(
                self.application_manager,
                None,
                AxisID::Only,
                Path::new(&directory),
                Some(dataset_name.as_str()),
                DataFileType::Recon,
                axis_type,
                is_dataset_name,
            ) {
                return false;
            }
        }
        self.base.button_execute_action_super()
    }
}

impl ContextMenu for SetupDialog {
    /// Java `@Override popUpContextMenu(MouseEvent)`: right mouse button
    /// context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let _context_popup = ContextPopup::new_component_mouse_event_string_base_manager_axis_id(
            &self.root_panel.get_component(),
            mouse_event,
            Some("INITIAL STEPS"),
            self.application_manager,
            self.axis_id,
        );
    }
}

impl Run3dmodButtonContainer for SetupDialog {
    /// Java `@Override action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let raw_image_stack = self
            .bctf_raw_image_stack
            .get_text_void()
            .unwrap_or_default();
        if raw_image_stack.is_empty() {
            ui_harness::open_message_dialog_from_process(
                Some(self.application_manager as &'static dyn BaseManager),
                "Raw image stack has not been entered",
                "Raw Image Stack",
                Some(AxisID::Only),
            );
            return;
        }
        let Some(extension) = Extension::get_instance(&raw_image_stack) else {
            return;
        };
        let file_extension = format!("{EXTENSION_DIVIDER}{extension}");
        let Some(expert) = self.expert() else {
            return;
        };
        if self.btn_view_raw_stack_a.get_action_command().as_deref() == Some(action_command) {
            expert.view_raw_stack(&file_extension, AxisID::First, run_3dmod_menu_options);
        } else if self.btn_view_raw_stack_b.get_action_command().as_deref() == Some(action_command)
        {
            expert.view_raw_stack(&file_extension, AxisID::Second, run_3dmod_menu_options);
        }
    }
}

impl Expandable for SetupDialog {
    /// Java `@Override expand(ExpandButton)` (empty).
    fn expand_expand_button(&self, _button: &Rc<ExpandButton>) {}

    /// Java `@Override expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.update_advanced(button.is_expanded());
        self.pack_axis();
    }
}

impl ControlListener for SetupDialog {
    /// Java `@Override controlEvent()`.
    fn control_event(&self) {
        self.update_display(false, false);
        let Some(raw_image_stack) = self.bctf_raw_image_stack.get_file() else {
            self.l_remove_exclude_views_msg_a.set_text("");
            self.l_remove_exclude_views_msg_b.set_text("");
            return;
        };
        let raw_image_stack = raw_image_stack.to_string_lossy().into_owned();
        let dual_axis = self.rb_dual_axis.is_selected();
        let dataset_name = dataset_tool::get_dataset_name(
            Some(utilities::java_io_file_get_name(&raw_image_stack).as_str()),
            dual_axis,
        );
        let mut axis_id = AxisID::Only;
        if dual_axis {
            axis_id = AxisID::First;
        }
        let file_name_filter =
            crate::imod::etomo::storage::exclude_views_info_file_name_filter::ExcludeViewsInfoFileNameFilter::new(
                dataset_name.as_deref(),
            );
        file_name_filter.set_axis_id(axis_id);
        // `File dir = rawImageStack.getParentFile(); dir.list(fileNameFilter)`.
        // Upstream bug fixed in translation (SetupDialog.java:513-514): a raw
        // image stack path with no parent directory makes `dir` null and
        // `dir.list` throw a NullPointerException; such a path has no directory
        // to search, so no previous excludeviews run is found (as when
        // `File.list` returns null).
        let dir = utilities::java_io_file_get_parent(&raw_image_stack);
        let list = |filter: &crate::imod::etomo::storage::exclude_views_info_file_name_filter::ExcludeViewsInfoFileNameFilter| -> Option<Vec<String>> {
            let dir = dir.as_deref()?;
            let entries = std::fs::read_dir(dir).ok()?;
            Some(
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .filter(|name| filter.accept(Path::new(dir), Some(name.as_str())))
                    .collect(),
            )
        };
        let file_name_list = list(&file_name_filter);
        if file_name_list.is_some_and(|file_name_list| !file_name_list.is_empty()) {
            // Excludeviews was run for this dataset name previously.
            self.l_remove_exclude_views_msg_a
                .set_text(REMOVE_EXCLUDE_VIEW_MSG);
        } else {
            self.l_remove_exclude_views_msg_a.set_text("");
        }
        if dual_axis {
            file_name_filter.set_axis_id(AxisID::Second);
            let file_name_list = list(&file_name_filter);
            if file_name_list.is_some_and(|file_name_list| !file_name_list.is_empty()) {
                // Excludeviews was run for this dataset name previously.
                self.l_remove_exclude_views_msg_b
                    .set_text(REMOVE_EXCLUDE_VIEW_MSG);
            } else {
                self.l_remove_exclude_views_msg_b.set_text("");
            }
        }
        if self.cb_half_float_mode_output_if_float.is_selected() {
            if let Some(expert) = self.expert() {
                expert.load_header();
            }
        }
        self.update_display(false, false);
        // `UIHarness.INSTANCE.pack(axisID, applicationManager)` with the local
        // `axisID` computed above, which shadows the field.
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::INSTANCE
            .with(|harness| harness.pack_axis_id_base_manager(Some(axis_id), Some(manager)));
    }
}

impl SetupReconInterface for SetupDialog {
    /// Java `@Override setBinning(String)`.
    fn set_binning(&self, input: Option<&str>) {
        self.ltf_binning.set_text_string(input);
    }

    /// Java `@Override setImageRotation(String)`.
    fn set_image_rotation(&self, input: Option<&str>) {
        self.ltf_image_rotation.set_text_string(input);
    }

    /// Java `@Override setPixelSize(double)`.
    fn set_pixel_size(&self, input: f64) {
        self.ltf_pixel_size.set_text_double(input);
    }

    /// Java `@Override setHalfFloatModeOutput(Integer)`.  The Java compares
    /// the `Integer`s with `==` (reference identity); both constants are small
    /// autoboxed values from the `Integer` cache, so identity is value
    /// equality.
    fn set_half_float_mode_output(&self, input: Option<i32>) {
        if Some(copy_tomo_coms::HALF_FLOAT) == input {
            self.cb_half_float_mode_output.set_selected_boolean(true);
        } else if Some(copy_tomo_coms::HALF_FLOAT_IF_FLOAT) == input {
            self.cb_half_float_mode_output_if_float
                .set_selected_boolean(true);
        }
    }

    /// Java `@Override getDataset()` (deprecated 4/8/2019, replaced by
    /// getRawImageStack).
    #[allow(deprecated)]
    fn get_dataset(&self) -> Option<String> {
        self.get_raw_image_stack()
    }

    /// Java `@Override getRawImageStack()`.
    fn get_raw_image_stack(&self) -> Option<String> {
        self.bctf_raw_image_stack.get_text_void()
    }

    /// Java `@Override isDualAxisSelected()`.
    fn is_dual_axis_selected(&self) -> bool {
        self.rb_dual_axis.is_selected()
    }

    /// Java `@Override getDistortionFile()`.
    fn get_distortion_file(&self) -> Option<String> {
        self.ftf_distortion_file.get_text()
    }

    /// Java `@Override getMagGradientFile()`.
    fn get_mag_gradient_file(&self) -> Option<String> {
        self.ftf_mag_gradient_file.get_text()
    }

    /// Java `@Override validateTiltAngle(AxisID, String)`.
    fn validate_tilt_angle(&self, axis_id: AxisID, error_title: &str) -> bool {
        self.expert()
            .is_some_and(|expert| expert.validate_tilt_angle(axis_id, error_title))
    }

    /// Java `@Override isSingleViewSelected()`.
    fn is_single_view_selected(&self) -> bool {
        self.rb_single_view.is_selected()
    }

    /// Java `@Override getBackupDirectory()`.
    fn get_backup_directory(&self) -> Option<String> {
        self.ftf_backup_directory.get_text()
    }

    /// Java `@Override getBinning(boolean)`.
    fn get_binning(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_binning.get_text_boolean(do_validation)
    }

    /// Java `@Override getExcludeList(AxisID, boolean)`.
    fn get_exclude_list(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        if axis_id == AxisID::Second {
            return self.ltf_exclude_list_b.get_text_boolean(do_validation);
        }
        self.ltf_exclude_list_a.get_text_boolean(do_validation)
    }

    /// Java `@Override getTwodir(AxisID, boolean)`.
    fn get_twodir(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        if axis_id == AxisID::Second {
            return self.tf_btwodir.get_text_boolean(do_validation);
        }
        self.tf_twodir.get_text_boolean(do_validation)
    }

    /// Java `@Override isTwodir(AxisID)`.
    fn is_twodir(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.cb_tf_btwodir.is_selected() && self.rb_b_bidirectional.is_selected();
        }
        self.cb_tf_twodir.is_selected() && self.rb_bidirectional.is_selected()
    }

    /// Java `@Override setTwodir(AxisID, double)`.
    fn set_twodir(&self, axis_id: AxisID, input: f64) {
        self.set_twodir_axis_id_double(axis_id, input);
    }

    /// Java `@Override getFiducialDiameter(boolean)`.
    fn get_fiducial_diameter(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_fiducial_diameter.get_text_boolean(do_validation)
    }

    /// Java `@Override getImageRotation(AxisID, boolean)`; `axisID` has no
    /// effect.
    fn get_image_rotation(
        &self,
        _axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_image_rotation.get_text_boolean(do_validation)
    }

    /// Java `@Override getPixelSize(boolean)`.
    fn get_pixel_size(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_pixel_size.get_text_boolean(do_validation)
    }

    /// Java `@Override getHalfFloatModeOutput()`.
    fn get_half_float_mode_output(&self) -> Option<i32> {
        if self.cb_half_float_mode_output.is_selected() {
            return Some(copy_tomo_coms::HALF_FLOAT);
        }
        if self.cb_half_float_mode_output_if_float.is_selected() {
            return Some(copy_tomo_coms::HALF_FLOAT_IF_FLOAT);
        }
        None
    }

    /// Java `@Override isAdjustedFocusSelected(AxisID)`.
    fn is_adjusted_focus_selected(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.cb_adjusted_focus_b.is_selected();
        }
        self.cb_adjusted_focus_a.is_selected()
    }

    /// Java `@Override isSingleAxisSelected()`.
    fn is_single_axis_selected(&self) -> bool {
        self.rb_single_axis.is_selected()
    }

    /// Java `@Override isGpuProcessingSelected(String)`.
    fn is_gpu_processing_selected(&self, _property_user_dir: Option<&str>) -> bool {
        self.cb_gpu_processing.is_selected()
    }

    /// Java `@Override isParallelProcessSelected(String)`.
    fn is_parallel_process_selected(&self, _property_user_dir: Option<&str>) -> bool {
        self.cb_parallel_process.is_selected()
    }

    /// Java `@Override getTiltAngleFields(AxisID, TiltAngleSpec, boolean)`.
    fn get_tilt_angle_fields(
        &self,
        axis_id: AxisID,
        tilt_angle_spec: &mut TiltAngleSpec,
        do_validation: bool,
    ) -> Result<bool, String> {
        match self.expert() {
            Some(expert) => expert.get_tilt_angle_fields(axis_id, tilt_angle_spec, do_validation),
            None => Ok(false),
        }
    }

    /// Java `@Override getDirectiveFileCollection()`.
    fn get_directive_file_collection(&self) -> Option<DirectiveFileCollectionHandle> {
        self.template_panel.get_directive_file_collection()
    }

    /// Java `@Override initTiltAngleFields(AxisID, TiltAngleSpec,
    /// UserConfiguration)`.
    fn init_tilt_angle_fields(
        &self,
        axis_id: AxisID,
        tilt_angle_spec: &TiltAngleSpec,
        user_configuration: &UserConfiguration,
    ) {
        if let Some(expert) = self.expert() {
            expert.set_tilt_angle_fields(axis_id, tilt_angle_spec, user_configuration);
        }
    }

    /// Java `@Override getParameters(ExcludeViewsParam, AxisID, boolean,
    /// boolean)`.
    fn get_parameters(
        &self,
        param: &mut ExcludeViewsParam,
        axis_id: AxisID,
        _dual_axis: bool,
        _do_validation: bool,
    ) -> bool {
        let raw_image_stack = self.bctf_raw_image_stack.get_file();
        if let Some(raw_image_stack) = raw_image_stack {
            let dual = self.rb_dual_axis.is_selected();
            let root_name = dataset_tool::get_dataset_name(
                Some(&utilities::java_io_file_get_name(
                    &raw_image_stack.to_string_lossy(),
                )),
                dual,
            );
            if root_name.is_some() {
                param.set_stack_name(
                    file_type::CLASS
                        .raw_stack
                        .get_file_name(
                            Some(self.application_manager as &'static dyn BaseManager),
                            Some(axis_id),
                        )
                        .as_deref(),
                );
            }
        }
        param.set_montaged_images(self.rb_montage.is_selected());
        param.set_delete_old_files(
            self.cb_delete_old_files.is_enabled() && self.cb_delete_old_files.is_selected(),
        );
        true
    }

    /// Java `@Override msgExcludeViewsSucceeded(AxisID, boolean, boolean)`.
    fn msg_exclude_views_succeeded(
        &self,
        axis_id: AxisID,
        process_running: bool,
        process_done: bool,
    ) {
        if axis_id == AxisID::Second {
            self.exclude_views_succeeded_b.set(true);
            self.ltf_exclude_list_b.set_text_string(Some(""));
            self.l_remove_exclude_views_msg_b
                .set_text(REMOVE_EXCLUDE_VIEW_MSG);
            self.tilt_angles_b.msg_exclude_views_succeeded();
            if self.ltf_exclude_list_a.is_empty() {
                self.cb_remove_excluded_views.set_selected_boolean(false);
            }
        } else {
            self.exclude_views_succeeded_a.set(true);
            self.ltf_exclude_list_a.set_text_string(Some(""));
            self.l_remove_exclude_views_msg_a
                .set_text(REMOVE_EXCLUDE_VIEW_MSG);
            self.tilt_angles_a.msg_exclude_views_succeeded();
            if !self.ltf_exclude_list_b.is_enabled() || self.ltf_exclude_list_b.is_empty() {
                self.cb_remove_excluded_views.set_selected_boolean(false);
            }
        }
        self.update_display(process_running, process_done);
    }

    /// Java `@Override getDoseSym(AxisID, boolean)`.
    fn get_dose_sym(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        if axis_id == AxisID::Second {
            return self.tf_b_dose_sym.get_text_boolean(do_validation);
        }
        self.tf_dose_sym.get_text_boolean(do_validation)
    }

    /// Java `@Override isDoseSym(AxisID)`.
    fn is_dose_sym(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.cb_tf_btwodir.is_selected() && self.rb_b_dose_symmetric.is_selected();
        }
        self.cb_tf_twodir.is_selected() && self.rb_dose_symmetric.is_selected()
    }

    /// Java `@Override setDoseSym(AxisID, double)`.
    fn set_dose_sym(&self, axis_id: AxisID, input: f64) {
        if axis_id == AxisID::Second {
            self.cb_tf_btwodir.set_selected_boolean(true);
            self.rb_b_dose_symmetric.set_selected_boolean(true);
            self.tf_b_dose_sym
                .set_text_string(Some(&java_lang_double_to_string(input)));
        } else {
            self.cb_tf_twodir.set_selected_boolean(true);
            self.rb_dose_symmetric.set_selected_boolean(true);
            self.tf_dose_sym
                .set_text_string(Some(&java_lang_double_to_string(input)));
        }
        self.update_display(false, false);
    }
}
