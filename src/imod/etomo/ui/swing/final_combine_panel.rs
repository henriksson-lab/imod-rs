//! `IMOD/Etomo/src/etomo/ui/swing/FinalCombinePanel.java`.
//!
//! The Final Match tab of `TomogramCombinationDialog`: the patch region
//! model, the Patchcorr (corrsearch3d) parameters, the Matchorwarp
//! parameters, the Volcombine parameters and the restart / 3dmod buttons.
//!
//! Java `class FinalCombinePanel implements ContextMenu, FinalCombineFields,
//! Run3dmodButtonContainer, Expandable`: an EDT object created as `Rc<Self>`
//! by [`FinalCombinePanel::new`] (the Java builds everything in its
//! constructor); every method takes `&self`.  The dialog is held weakly (it
//! owns its panels).  The inner class `ButtonActionListener` is a closure
//! holding a weak reference to the panel, kept in `action_listener` so
//! `removeListeners` can remove it.
//!
//! The Java fields that are never reassigned are plain fields here; the
//! Java declares most of them non-final but only assigns them once.

use std::fmt;
use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::final_combine_fields::FinalCombineFields;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::process_interface::ProcessInterface;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::{self, SpacedPanel};
use super::text_field::TextField;
use super::tomogram_combination_dialog::{self, TomogramCombinationDialog};
use super::ui_harness;
use super::ui_utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_combine_params::ConstCombineParams;
use crate::imod::etomo::comscript::const_matchorwarp_param::ConstMatchorwarpParam;
use crate::imod::etomo::comscript::const_patchcrawl3d_param::{self, ConstPatchcrawl3DParam};
use crate::imod::etomo::comscript::matchorwarp_param::MatchorwarpParam;
use crate::imod::etomo::comscript::patchcrawl3d_param::{self, Patchcrawl3DParam};
use crate::imod::etomo::comscript::set_param::SetParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent, MouseEvent, MouseListener};
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_integer_parse_int;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::util::dataset_files;

/// Java package-private static final `NO_VOLCOMBINE_TITLE`.
pub const NO_VOLCOMBINE_TITLE: &str = "Stop before running volcombine";
/// Java package-private static final `VOLCOMBINE_PARALLEL_PROCESSING_TOOL_TIP`.
pub const VOLCOMBINE_PARALLEL_PROCESSING_TOOL_TIP: &str =
    "Check to distribute the volcombine process across multiple computers.";
/// Java private static final `KERNEL_SIGMA_LABEL`.
const KERNEL_SIGMA_LABEL: &str = "Kernel filtering with sigma: ";

/// Java `class FinalCombinePanel implements ContextMenu, FinalCombineFields,
/// Run3dmodButtonContainer, Expandable`.
pub struct FinalCombinePanel {
    /// Java private `tomogramCombinationDialog` (held weakly).
    tomogram_combination_dialog: Weak<TomogramCombinationDialog>,
    /// Java private `applicationManager`.
    application_manager: &'static ApplicationManager,

    /// Java private `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,

    /// Java private `pnlPatchcorr = new EtomoPanel()`.
    pnl_patchcorr: Rc<EtomoPanel>,
    /// Java private `pnlPatchcorrBody = SpacedPanel.getInstance(true)`.
    pnl_patchcorr_body: Rc<SpacedPanel>,

    /// Java private `pnlPatchsize = new JPanel()`.
    pnl_patchsize: Rc<JComponent>,
    /// Java private `pnlPatchsizeEdit = new JPanel()`.
    pnl_patchsize_edit: Rc<JComponent>,
    /// Java private `ltfXPatchSize` ("X patch size :").
    ltf_x_patch_size: Rc<LabeledTextField>,
    /// Java private `ltfYPatchSize` (labelled "Z patch size :": Y and Z are
    /// flipped).
    ltf_y_patch_size: Rc<LabeledTextField>,
    /// Java private `ltfZPatchSize` (labelled "Y patch size :").
    ltf_z_patch_size: Rc<LabeledTextField>,
    /// Java private `pnlPatchsizeButtons = new JPanel()`.
    pnl_patchsize_buttons: Rc<JComponent>,
    /// Java private `btnPatchsizeIncrease`.
    btn_patchsize_increase: Rc<MultiLineButton>,
    /// Java private `btnPatchsizeDecrease`.
    btn_patchsize_decrease: Rc<MultiLineButton>,

    /// Java private `ltfXNPatches`.
    ltf_x_n_patches: Rc<LabeledTextField>,
    /// Java private `ltfYNPatches` ("Number of Z patches :").
    ltf_y_n_patches: Rc<LabeledTextField>,
    /// Java private `ltfZNPatches` ("Number of Y patches :").
    ltf_z_n_patches: Rc<LabeledTextField>,
    /// Java private `cbKernelSigma`.
    cb_kernel_sigma: Rc<CheckBox>,
    /// Java private `tfKernelSigma`.
    tf_kernel_sigma: Rc<TextField>,

    /// Java private final `pnlBoundary = new JPanel()`.
    pnl_boundary: Rc<JComponent>,
    /// Java private `ltfXLow`.
    ltf_x_low: Rc<LabeledTextField>,
    /// Java private `ltfXHigh`.
    ltf_x_high: Rc<LabeledTextField>,
    /// Java private `ltfYLow` ("Z Low :").
    ltf_y_low: Rc<LabeledTextField>,
    /// Java private `ltfYHigh` ("Z high :").
    ltf_y_high: Rc<LabeledTextField>,
    /// Java private `ltfZLow` ("Y Low :").
    ltf_z_low: Rc<LabeledTextField>,
    /// Java private `ltfZHigh` ("Y high :").
    ltf_z_high: Rc<LabeledTextField>,
    /// Java private final `actionListener = new ButtonActionListener(this)`.
    action_listener: ActionListener,

    /// Java private final `btnPatchcorrRestart`.
    btn_patchcorr_restart: Rc<Run3dmodButton>,

    /// Java private `pnlMatchorwarp = new EtomoPanel()`.
    pnl_matchorwarp: Rc<EtomoPanel>,
    /// Java private `pnlMatchorwarpBody = new JPanel()`.
    pnl_matchorwarp_body: Rc<JComponent>,
    /// Java private `pnlPatchRegionModel = new EtomoPanel()`.
    pnl_patch_region_model: Rc<EtomoPanel>,
    /// Java private `pnlPatchRegionModelBody = SpacedPanel.getInstance(true)`.
    pnl_patch_region_model_body: Rc<SpacedPanel>,
    /// Java private `cbUsePatchRegionModel`.
    cb_use_patch_region_model: Rc<CheckBox>,
    /// Java private `btnPatchRegionModel`.
    btn_patch_region_model: Rc<Run3dmodButton>,
    /// Java private `ltfWarpLimit`.
    ltf_warp_limit: Rc<LabeledTextField>,
    /// Java private `ltfRefineLimit`.
    ltf_refine_limit: Rc<LabeledTextField>,

    /// Java private `ltfXLowerExclude`.
    ltf_x_lower_exclude: Rc<LabeledTextField>,
    /// Java private `ltfXUpperExclude`.
    ltf_x_upper_exclude: Rc<LabeledTextField>,
    /// Java private `ltfZLowerExclude`.
    ltf_z_lower_exclude: Rc<LabeledTextField>,
    /// Java private `ltfZUpperExclude`.
    ltf_z_upper_exclude: Rc<LabeledTextField>,
    /// Java private `cbUseLinearInterpolation`.
    cb_use_linear_interpolation: Rc<CheckBox>,
    /// Java private `pnlMatchorwarpButtons = new JPanel()`.
    pnl_matchorwarp_buttons: Rc<JComponent>,
    /// Java private final `btnMatchorwarpRestart`.
    btn_matchorwarp_restart: Rc<Run3dmodButton>,
    /// Java private `btnMatchorwarpTrial`.
    btn_matchorwarp_trial: Rc<MultiLineButton>,
    /// Java private `pnlVolcombine = new EtomoPanel()`.
    pnl_volcombine: Rc<EtomoPanel>,
    /// Java private `pnlVolcombineBody = new JPanel()`.
    pnl_volcombine_body: Rc<JComponent>,
    /// Java private final `btnVolcombineRestart`.
    btn_volcombine_restart: Rc<Run3dmodButton>,
    /// Java private `pnlButton = new JPanel()`.
    pnl_button: Rc<JComponent>,
    /// Java private `btnPatchVectorModel`.
    btn_patch_vector_model: Rc<MultiLineButton>,
    /// Java private `btnReplacePatchOut`.
    btn_replace_patch_out: Rc<MultiLineButton>,
    /// Java private `btnImodMatchedTo`.
    btn_imod_matched_to: Rc<Run3dmodButton>,
    /// Java private final `btnImodCombined`.
    btn_imod_combined: Rc<Run3dmodButton>,
    /// Java private `cbNoVolcombine`.
    cb_no_volcombine: Rc<CheckBox>,
    /// Java private `ltfReductionFactor`.
    ltf_reduction_factor: Rc<LabeledTextField>,
    /// Java private `ltfLowFromBothRadius`.
    ltf_low_from_both_radius: Rc<LabeledTextField>,
    /// Java private `cbParallelProcess` (assigned in the constructor).
    cb_parallel_process: Rc<CheckBox>,
    /// Java private final `patchRegionModelHeader`.
    patch_region_model_header: Rc<PanelHeader>,
    /// Java private final `patchcorrHeader`.
    patchcorr_header: Rc<PanelHeader>,
    /// Java private final `matchorwarpHeader`.
    matchorwarp_header: Rc<PanelHeader>,
    /// Java private final `volcombineHeader`.
    volcombine_header: Rc<PanelHeader>,
    /// Java private final `pnlKernelSigma = SpacedPanel.getInstance()`.
    pnl_kernel_sigma: Rc<SpacedPanel>,
    /// Java private final `ltfInitialShiftX`.
    ltf_initial_shift_x: Rc<LabeledTextField>,
    /// Java private final `ltfInitialShiftY` (labelled "Z:").
    ltf_initial_shift_y: Rc<LabeledTextField>,
    /// Java private final `ltfInitialShiftZ` (labelled "Y:").
    ltf_initial_shift_z: Rc<LabeledTextField>,
    /// Java private final `pnlInitialShiftXYZ = SpacedPanel.getInstance()`.
    pnl_initial_shift_xyz: Rc<SpacedPanel>,
    /// Java private `btnPatchVectorCCCModel`.
    btn_patch_vector_ccc_model: Rc<MultiLineButton>,
    /// Java private final `dialogType`.
    dialog_type: DialogType,

    /// Java `this`.
    this: Weak<FinalCombinePanel>,
}

impl fmt::Display for FinalCombinePanel {
    /// Java public override `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // getClass().getName() + "[" + paramString() + "]\n"
        write!(
            f,
            "etomo.ui.swing.FinalCombinePanel[{}]\n",
            self.param_string()
        )
    }
}

impl FinalCombinePanel {
    /// Java package-private `paramString()`.
    pub fn param_string(&self) -> String {
        format!(
            "ltfXPatchSize={},\nltfYPatchSize={},\nltfZPatchSize={},\nltfXNPatches={},\n\
             ltfYNPatches={},\nltfZNPatches={},\nltfXLow={},\nltfXHigh={},\nltfYLow={},\n\
             ltfYHigh={},\nltfZLow={},\nltfZHigh={},\ncbUsePatchRegionModel={},\n\
             ltfWarpLimit={},\nltfRefineLimit={},\nltfXLowerExclude={},\n\
             ltfXUpperExclude={},\nltfZLowerExclude={},\nltfZUpperExclude={},\n\
             cbUseLinearInterpolation={},\ncbNoVolcombine={},\nltfReductionFactor={},\n\
             cbParallelProcess={}",
            self.ltf_x_patch_size,
            self.ltf_y_patch_size,
            self.ltf_z_patch_size,
            self.ltf_x_n_patches,
            self.ltf_y_n_patches,
            self.ltf_z_n_patches,
            self.ltf_x_low,
            self.ltf_x_high,
            self.ltf_y_low,
            self.ltf_y_high,
            self.ltf_z_low,
            self.ltf_z_high,
            self.cb_use_patch_region_model,
            self.ltf_warp_limit,
            self.ltf_refine_limit,
            self.ltf_x_lower_exclude,
            self.ltf_x_upper_exclude,
            self.ltf_z_lower_exclude,
            self.ltf_z_upper_exclude,
            self.cb_use_linear_interpolation,
            self.cb_no_volcombine,
            self.ltf_reduction_factor,
            self.cb_parallel_process
        )
    }

    /// Java package-private constructor `FinalCombinePanel(
    /// TomogramCombinationDialog, ApplicationManager, DialogType,
    /// GlobalExpandButton)`.  Default constructor.
    ///
    /// The fields (the Java field initializers and the fields the constructor
    /// assigns) are created first, inside `Rc::new_cyclic` because several of
    /// them take `this`; the rest of the constructor body then runs on the
    /// finished object in the Java order.
    pub fn new(
        parent: Weak<TomogramCombinationDialog>,
        app_mgr: &'static ApplicationManager,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<FinalCombinePanel> {
        let instance = Rc::new_cyclic(|this: &Weak<FinalCombinePanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let expandable: Weak<dyn Expandable> = this.clone();
            // Field initializers, in declaration order.
            let pnl_root = JComponent::new_panel();
            let pnl_patchcorr = EtomoPanel::new();
            let pnl_patchcorr_body = SpacedPanel::get_instance_boolean(true);
            let pnl_patchsize = JComponent::new_panel();
            let pnl_patchsize_edit = JComponent::new_panel();
            let ltf_x_patch_size =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("X patch size :"));
            let ltf_y_patch_size =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Z patch size :"));
            let ltf_z_patch_size =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y patch size :"));
            let pnl_patchsize_buttons = JComponent::new_panel();
            let btn_patchsize_increase = MultiLineButton::new_string(Some("Patch Size +20%"));
            let btn_patchsize_decrease = MultiLineButton::new_string(Some("Patch Size -20%"));
            let ltf_x_n_patches = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Number of X patches :"),
            );
            let ltf_y_n_patches = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Number of Z patches :"),
            );
            let ltf_z_n_patches = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Number of Y patches :"),
            );
            let cb_kernel_sigma = CheckBox::new_string(Some(KERNEL_SIGMA_LABEL));
            let tf_kernel_sigma =
                TextField::new(FieldType::FloatingPoint, Some(KERNEL_SIGMA_LABEL), None);
            let pnl_boundary = JComponent::new_panel();
            let ltf_x_low =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("X Low :"));
            let ltf_x_high =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("X high :"));
            let ltf_y_low =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Z Low :"));
            let ltf_y_high =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Z high :"));
            let ltf_z_low =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y Low :"));
            let ltf_z_high =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y high :"));
            // Java `new ButtonActionListener(this)`.
            let listenee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(listenee) = listenee.upgrade() {
                    listenee.action(event.get_action_command().unwrap_or(""), None, None);
                }
            });
            let pnl_matchorwarp = EtomoPanel::new();
            let pnl_matchorwarp_body = JComponent::new_panel();
            let pnl_patch_region_model = EtomoPanel::new();
            let pnl_patch_region_model_body = SpacedPanel::get_instance_boolean(true);
            let cb_use_patch_region_model = CheckBox::new_string(Some("Use patch region model"));
            let btn_patch_region_model =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Create/Edit Patch Region Model"),
                    Some(container.clone()),
                );
            let ltf_warp_limit = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some("Warping residual limits: "),
            );
            let ltf_refine_limit = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Residual limit for single transform: "),
            );
            let ltf_x_lower_exclude = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Number of columns to exclude on left (in X): "),
            );
            let ltf_x_upper_exclude = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Number of columns to exclude on right (in X): "),
            );
            let ltf_z_lower_exclude = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Number of rows to exclude on bottom (in Y): "),
            );
            let ltf_z_upper_exclude = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Number of rows to exclude on top (in Y): "),
            );
            let cb_use_linear_interpolation =
                CheckBox::new_string(Some("Use linear interpolation"));
            let pnl_matchorwarp_buttons = JComponent::new_panel();
            let btn_matchorwarp_trial = MultiLineButton::new_string(Some("Matchorwarp Trial Run"));
            let pnl_volcombine = EtomoPanel::new();
            let pnl_volcombine_body = JComponent::new_panel();
            let pnl_button = JComponent::new_panel();
            let btn_patch_vector_model =
                MultiLineButton::new_string(Some("Examine Patch Vector Model"));
            let btn_replace_patch_out = MultiLineButton::new_string(Some("Replace Patch Vectors"));
            let btn_imod_matched_to =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Volume Being Matched To"),
                    Some(container.clone()),
                );
            let btn_imod_combined =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Combined Volume"),
                    Some(container.clone()),
                );
            let cb_no_volcombine = CheckBox::new_string(Some(NO_VOLCOMBINE_TITLE));
            let ltf_reduction_factor = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Reduction factor for matching amplitudes in combined FFT: "),
            );
            let ltf_low_from_both_radius = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Radius below which to average components from both tomograms: "),
            );
            let pnl_kernel_sigma = SpacedPanel::get_instance_void();
            let ltf_initial_shift_x = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Initial shift in X:"),
            );
            let ltf_initial_shift_y =
                LabeledTextField::new_field_type_string(FieldType::FloatingPoint, Some("Z:"));
            let ltf_initial_shift_z =
                LabeledTextField::new_field_type_string(FieldType::FloatingPoint, Some("Y:"));
            let pnl_initial_shift_xyz = SpacedPanel::get_instance_void();
            let btn_patch_vector_ccc_model =
                MultiLineButton::new_string(Some("Open Vector Model with Correlations"));

            // Constructor body: the field assignments (the rest runs below).
            // this.dialogType = dialogType; tomogramCombinationDialog = parent;
            let factory = app_mgr.get_process_result_display_factory(AxisID::Only);
            // (Run3dmodButton) appMgr.getProcessResultDisplayFactory(AxisID.ONLY)
            // .getRestartPatchcorr(), .getRestartMatchorwarp(),
            // .getRestartVolcombine()
            let btn_patchcorr_restart = factory.get_restart_patchcorr();
            let btn_matchorwarp_restart = factory.get_restart_matchorwarp();
            let btn_volcombine_restart = factory.get_restart_volcombine();
            // applicationManager = appMgr;
            let patch_region_model_header = PanelHeader::get_instance(
                Some("Patch Region Model"),
                Some(expandable.clone()),
                Some(dialog_type),
            );
            let patchcorr_header =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Patchcorr Parameters"),
                    Some(expandable.clone()),
                    Some(dialog_type),
                    Some(global_advanced_button.clone()),
                );
            let matchorwarp_header =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Matchorwarp Parameters"),
                    Some(expandable.clone()),
                    Some(dialog_type),
                    Some(global_advanced_button.clone()),
                );
            // cbParallelProcess =
            // new CheckBox(tomogramCombinationDialog.parallelProcessCheckBoxText);
            // The dialog exists (it is constructing this panel); a gone dialog
            // (Java NullPointerException) gives an unlabelled check box.
            let parallel_process_check_box_text = parent
                .upgrade()
                .map(|parent| parent.parallel_process_check_box_text.clone())
                .unwrap_or_default();
            let cb_parallel_process = CheckBox::new_string(Some(&parallel_process_check_box_text));
            let volcombine_header =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Volcombine Parameters"),
                    Some(expandable),
                    Some(dialog_type),
                    Some(global_advanced_button.clone()),
                );
            FinalCombinePanel {
                tomogram_combination_dialog: parent,
                application_manager: app_mgr,
                pnl_root,
                pnl_patchcorr,
                pnl_patchcorr_body,
                pnl_patchsize,
                pnl_patchsize_edit,
                ltf_x_patch_size,
                ltf_y_patch_size,
                ltf_z_patch_size,
                pnl_patchsize_buttons,
                btn_patchsize_increase,
                btn_patchsize_decrease,
                ltf_x_n_patches,
                ltf_y_n_patches,
                ltf_z_n_patches,
                cb_kernel_sigma,
                tf_kernel_sigma,
                pnl_boundary,
                ltf_x_low,
                ltf_x_high,
                ltf_y_low,
                ltf_y_high,
                ltf_z_low,
                ltf_z_high,
                action_listener,
                btn_patchcorr_restart,
                pnl_matchorwarp,
                pnl_matchorwarp_body,
                pnl_patch_region_model,
                pnl_patch_region_model_body,
                cb_use_patch_region_model,
                btn_patch_region_model,
                ltf_warp_limit,
                ltf_refine_limit,
                ltf_x_lower_exclude,
                ltf_x_upper_exclude,
                ltf_z_lower_exclude,
                ltf_z_upper_exclude,
                cb_use_linear_interpolation,
                pnl_matchorwarp_buttons,
                btn_matchorwarp_restart,
                btn_matchorwarp_trial,
                pnl_volcombine,
                pnl_volcombine_body,
                btn_volcombine_restart,
                pnl_button,
                btn_patch_vector_model,
                btn_replace_patch_out,
                btn_imod_matched_to,
                btn_imod_combined,
                cb_no_volcombine,
                ltf_reduction_factor,
                ltf_low_from_both_radius,
                cb_parallel_process,
                patch_region_model_header,
                patchcorr_header,
                matchorwarp_header,
                volcombine_header,
                pnl_kernel_sigma,
                ltf_initial_shift_x,
                ltf_initial_shift_y,
                ltf_initial_shift_z,
                pnl_initial_shift_xyz,
                btn_patch_vector_ccc_model,
                dialog_type,
                this: this.clone(),
            }
        });
        let this = &instance;
        let container: Weak<dyn Run3dmodButtonContainer> = Rc::downgrade(this) as Weak<_>;
        let combined: Rc<dyn Deferred3dmodButton> = this.btn_imod_combined.clone();
        // Constructor body, in the Java order.
        this.btn_patchcorr_restart
            .set_container(Some(container.clone()));
        this.btn_patchcorr_restart
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(combined.clone()));
        this.btn_matchorwarp_restart
            .set_container(Some(container.clone()));
        this.btn_matchorwarp_restart
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(combined.clone()));
        this.btn_volcombine_restart.set_container(Some(container));
        this.btn_volcombine_restart
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(combined));
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.Y_AXIS)).

        // Layout Patch region model panel
        this.pnl_patch_region_model_body
            .set_box_layout(spaced_panel::X_AXIS);
        this.pnl_patch_region_model_body
            .add_check_box(&this.cb_use_patch_region_model);
        this.pnl_patch_region_model_body
            .add_multi_line_button(&this.btn_patch_region_model);
        this.pnl_patch_region_model_body.add_horizontal_glue();
        // btnPatchRegionModel.setSize();

        // Swing layout: pnlPatchRegionModel.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)); setBorder(BorderFactory.createEtchedBorder()).
        this.pnl_patch_region_model
            .add(&this.patch_region_model_header);
        this.pnl_patch_region_model
            .get_component()
            .add(&this.pnl_patch_region_model_body.get_container());

        // Layout the Patchcorr panel
        this.pnl_patchcorr_body.set_box_layout(spaced_panel::Y_AXIS);

        // Swing layout: pnlPatchsizeButtons.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)); a rigid area (FixedDim.x0_y5) between the buttons.
        this.pnl_patchsize_buttons
            .add(&this.btn_patchsize_increase.get_component());
        this.pnl_patchsize_buttons
            .add(&this.btn_patchsize_decrease.get_component());
        // Swing layout: UIUtilities.setButtonSizeAll(pnlPatchsizeButtons,
        // UIParameters.getInstance().getButtonDimension()).

        // Swing layout: pnlPatchsizeEdit.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)); a rigid area (FixedDim.x0_y5) after each field.
        this.pnl_patchsize_edit
            .add(&this.ltf_x_patch_size.get_container());
        this.pnl_patchsize_edit
            .add(&this.ltf_z_patch_size.get_container());
        this.pnl_patchsize_edit
            .add(&this.ltf_y_patch_size.get_container());

        // Swing layout: pnlPatchsize.setLayout(new BoxLayout(pnlPatchsize,
        // BoxLayout.X_AXIS)); a rigid area (FixedDim.x10_y0) between the panels.
        this.pnl_patchsize.add(&this.pnl_patchsize_edit);
        this.pnl_patchsize.add(&this.pnl_patchsize_buttons);
        this.pnl_patchcorr_body.add_j_panel(&this.pnl_patchsize);

        // Swing layout: pnlBoundary.setLayout(new GridLayout(3, 3, 5, 5)).
        this.pnl_boundary.add(&this.ltf_x_n_patches.get_container());
        this.pnl_boundary.add(&this.ltf_x_low.get_container());
        this.pnl_boundary.add(&this.ltf_x_high.get_container());
        this.pnl_boundary.add(&this.ltf_z_n_patches.get_container());
        this.pnl_boundary.add(&this.ltf_z_low.get_container());
        this.pnl_boundary.add(&this.ltf_z_high.get_container());
        this.pnl_boundary.add(&this.ltf_y_n_patches.get_container());
        this.pnl_boundary.add(&this.ltf_y_low.get_container());
        this.pnl_boundary.add(&this.ltf_y_high.get_container());
        this.pnl_patchcorr_body.add_j_panel(&this.pnl_boundary);

        this.pnl_initial_shift_xyz
            .set_box_layout(spaced_panel::X_AXIS);
        this.pnl_initial_shift_xyz
            .add_labeled_text_field(&this.ltf_initial_shift_x);
        this.pnl_initial_shift_xyz
            .add_labeled_text_field(&this.ltf_initial_shift_z);
        this.pnl_initial_shift_xyz
            .add_labeled_text_field(&this.ltf_initial_shift_y);
        this.pnl_patchcorr_body
            .add_spaced_panel(&this.pnl_initial_shift_xyz);

        this.pnl_kernel_sigma.set_box_layout(spaced_panel::X_AXIS);
        this.pnl_kernel_sigma.add_check_box(&this.cb_kernel_sigma);
        this.pnl_kernel_sigma.add_text_field(&this.tf_kernel_sigma);
        this.tf_kernel_sigma.set_enabled(false);
        this.pnl_patchcorr_body
            .add_spaced_panel(&this.pnl_kernel_sigma);

        // Swing layout: btnPatchcorrRestart.setAlignmentX(Component.CENTER_ALIGNMENT).
        let pnl_patchcorr_buttons = JComponent::new_panel();
        // Swing layout: pnlPatchcorrButtons.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); horizontal glue around and between the buttons.
        pnl_patchcorr_buttons.add(&this.btn_patchcorr_restart.get_component());
        pnl_patchcorr_buttons.add(&this.btn_patch_vector_ccc_model.get_component());
        this.pnl_patchcorr_body.add_j_panel(&pnl_patchcorr_buttons);

        // Swing layout: pnlPatchsizeButtons.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)) (again).

        // Swing layout: pnlPatchcorr.setLayout(new BoxLayout(pnlPatchcorr,
        // BoxLayout.Y_AXIS)); setBorder(BorderFactory.createEtchedBorder()).
        // patchcorrHeader = PanelHeader.getAdvancedBasicInstance(...) (created
        // with the fields).
        this.pnl_patchcorr.add(&this.patchcorr_header);
        this.pnl_patchcorr
            .get_component()
            .add(&this.pnl_patchcorr_body.get_container());

        // Layout the Matchorwarp panel
        // Swing layout: pnlMatchorwarpBody.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)); rigid areas (FixedDim.x0_y10 / x0_y5) between the
        // components.
        this.pnl_matchorwarp_body
            .add(&this.ltf_refine_limit.get_container());
        this.pnl_matchorwarp_body
            .add(&this.ltf_warp_limit.get_container());

        this.pnl_matchorwarp_body
            .add(&this.ltf_x_lower_exclude.get_container());
        this.pnl_matchorwarp_body
            .add(&this.ltf_x_upper_exclude.get_container());
        this.pnl_matchorwarp_body
            .add(&this.ltf_z_lower_exclude.get_container());
        this.pnl_matchorwarp_body
            .add(&this.ltf_z_upper_exclude.get_container());
        this.pnl_matchorwarp_body
            .add(&this.cb_use_linear_interpolation.get_component());

        // Swing layout: pnlMatchorwarpButtons.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); horizontal glue around and between the buttons.
        this.pnl_matchorwarp_buttons
            .add(&this.btn_matchorwarp_restart.get_component());
        this.pnl_matchorwarp_buttons
            .add(&this.btn_matchorwarp_trial.get_component());
        // Swing layout: UIUtilities.setButtonSizeAll(pnlMatchorwarpButtons,
        // UIParameters.getInstance().getButtonDimension()).

        this.pnl_matchorwarp_body.add(&this.pnl_matchorwarp_buttons);
        // Swing layout: pnlMatchorwarpBody.add(Box.createRigidArea(FixedDim.x0_y5)).

        // Swing layout: pnlMatchorwarp.setLayout(new BoxLayout(pnlMatchorwarp,
        // BoxLayout.Y_AXIS)); setBorder(BorderFactory.createEtchedBorder()).
        this.pnl_matchorwarp.add(&this.matchorwarp_header);
        this.pnl_matchorwarp
            .get_component()
            .add(&this.pnl_matchorwarp_body);

        // Swing layout: pnlVolcombineBody.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)).
        // cbParallelProcess = new CheckBox(...) (created with the fields).
        let pnl_parallel_process = JComponent::new_panel();
        // Swing layout: pnlParallelProcess.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); setAlignmentX(Component.CENTER_ALIGNMENT);
        // horizontal glue after the check box.
        pnl_parallel_process.add(&this.cb_parallel_process.get_component());
        this.pnl_volcombine_body.add(&pnl_parallel_process);
        let pnl_no_volcombine = JComponent::new_panel();
        // Swing layout: pnlNoVolcombine.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); setAlignmentX(Component.CENTER_ALIGNMENT);
        // horizontal glue after the check box.
        pnl_no_volcombine.add(&this.cb_no_volcombine.get_component());
        this.pnl_volcombine_body.add(&pnl_no_volcombine);
        this.pnl_volcombine_body
            .add(&this.ltf_reduction_factor.get_container());
        this.pnl_volcombine_body
            .add(&this.ltf_low_from_both_radius.get_container());
        // Swing layout: pnlVolcombineBody.add(Box.createRigidArea(FixedDim.x0_y5)).
        this.pnl_volcombine_body
            .add(&this.btn_volcombine_restart.get_component());
        // Swing layout: cbNoVolcombine and btnVolcombineRestart
        // .setAlignmentX(Component.CENTER_ALIGNMENT);
        // UIUtilities.setButtonSizeAll(pnlVolcombineBody,
        // UIParameters.getInstance().getButtonDimension()).
        ui_utilities::align_components_x(&this.pnl_volcombine_body, 0.5);

        // Swing layout: pnlVolcombine.setLayout(new BoxLayout(pnlVolcombine,
        // BoxLayout.Y_AXIS)); setBorder(BorderFactory.createEtchedBorder()).
        this.pnl_volcombine.add(&this.volcombine_header);
        this.pnl_volcombine
            .get_component()
            .add(&this.pnl_volcombine_body);

        // Create the button panel
        // Swing layout: pnlButton.setLayout(new BoxLayout(pnlButton,
        // BoxLayout.X_AXIS)); horizontal glue around and between the buttons.
        this.pnl_button
            .add(&this.btn_patch_vector_model.get_component());
        this.pnl_button
            .add(&this.btn_replace_patch_out.get_component());
        this.pnl_button
            .add(&this.btn_imod_matched_to.get_component());
        this.pnl_button.add(&this.btn_imod_combined.get_component());
        // Swing layout: UIUtilities.setButtonSizeAll(pnlButton,
        // UIParameters.getInstance().getButtonDimension()).

        // Root panel layout
        this.pnl_root
            .add(&this.pnl_patch_region_model.get_component());
        this.pnl_root.add(&this.pnl_patchcorr.get_component());
        this.pnl_root.add(&this.pnl_matchorwarp.get_component());
        // Swing layout: pnlRoot.add(Box.createRigidArea(FixedDim.x0_y5)).
        this.pnl_root.add(&this.pnl_volcombine.get_component());
        // Swing layout: pnlRoot.add(Box.createRigidArea(FixedDim.x0_y5));
        // pnlRoot.add(Box.createVerticalGlue()).
        this.pnl_root.add(&this.pnl_button);

        // Bind the buttons to action listener
        let action_listener = this.action_listener.clone();
        this.btn_patchcorr_restart
            .add_action_listener(action_listener.clone());
        this.btn_patchsize_increase
            .add_action_listener(action_listener.clone());
        this.btn_patchsize_decrease
            .add_action_listener(action_listener.clone());
        this.btn_patch_region_model
            .add_action_listener(action_listener.clone());
        this.btn_matchorwarp_restart
            .add_action_listener(action_listener.clone());
        this.btn_matchorwarp_trial
            .add_action_listener(action_listener.clone());
        this.btn_volcombine_restart
            .add_action_listener(action_listener.clone());
        this.btn_patch_vector_model
            .add_action_listener(action_listener.clone());
        this.btn_replace_patch_out
            .add_action_listener(action_listener.clone());
        this.btn_imod_matched_to
            .add_action_listener(action_listener.clone());
        this.btn_imod_combined
            .add_action_listener(action_listener.clone());
        this.cb_parallel_process
            .add_action_listener(Some(action_listener.clone()));
        this.cb_kernel_sigma
            .add_action_listener(Some(action_listener.clone()));
        this.btn_patch_vector_ccc_model
            .add_action_listener(action_listener);

        // Mouse listener for context menu
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(this) as Weak<_>;
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        this.pnl_root.add_mouse_listener(mouse_adapter);
        this.set_tool_tip_text();
        instance
    }

    /// Java package-private `removeListeners()`.
    pub fn remove_listeners(&self) {
        self.btn_patchcorr_restart
            .remove_action_listener(&self.action_listener);
        self.btn_matchorwarp_restart
            .remove_action_listener(&self.action_listener);
        self.btn_volcombine_restart
            .remove_action_listener(&self.action_listener);
    }

    /// Java package-private `updateAdvanced(boolean)`.
    pub fn update_advanced(&self, state: bool) {
        self.update_advanced_patchcorr(state);
        self.update_advanced_matchorwarp(state);
        self.update_advanced_volcombine(state);
    }

    /// Java package-private final `updateAdvancedPatchcorr(boolean)`.
    pub fn update_advanced_patchcorr(&self, state: bool) {
        self.pnl_boundary.set_visible(state);
        self.pnl_initial_shift_xyz.set_visible(state);
        self.pnl_kernel_sigma.set_visible(state);
    }

    /// Java package-private final `updateAdvancedMatchorwarp(boolean)`.
    pub fn update_advanced_matchorwarp(&self, state: bool) {
        self.ltf_refine_limit.set_visible(state);
        self.cb_use_linear_interpolation.set_visible(state);
    }

    /// Java package-private final `updateAdvancedVolcombine(boolean)`.
    pub fn update_advanced_volcombine(&self, state: bool) {
        self.ltf_reduction_factor.set_visible(state);
        self.ltf_low_from_both_radius.set_visible(state);
    }

    /// Java package-private `getPatchcorrProcessResultDisplay()`.
    pub fn get_patchcorr_process_result_display(&self) -> ProcessResultDisplayHandle {
        self.btn_patchcorr_restart.clone()
    }

    /// Java package-private `getImodCombinedButton()`.
    pub fn get_imod_combined_button(&self) -> Rc<Run3dmodButton> {
        self.btn_imod_combined.clone()
    }

    /// Java package-private `getMatchorwarpProcessResultDisplay()`.
    pub fn get_matchorwarp_process_result_display(&self) -> ProcessResultDisplayHandle {
        self.btn_matchorwarp_restart.clone()
    }

    /// Java package-private `getVolcombineProcessResultDisplay()`.
    pub fn get_volcombine_process_result_display(&self) -> ProcessResultDisplayHandle {
        self.btn_volcombine_restart.clone()
    }

    /// Java package-private `getContainer()`.  Return the pnlRoot reference.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java package-private `isRunVolcombine()`.
    pub fn is_run_volcombine(&self) -> bool {
        !self.cb_no_volcombine.is_selected()
    }

    /// Java package-private `setRunVolcombine(boolean)`.
    pub fn set_run_volcombine(&self, run_volcombine: bool) {
        self.cb_no_volcombine.set_selected_boolean(!run_volcombine);
    }

    /// Java package-private `setParameters(ConstCombineParams)`.
    pub fn set_parameters_const_combine_params(&self, combine_params: &dyn ConstCombineParams) {
        self.ltf_reduction_factor
            .set_text_string(combine_params.get_wedge_reduction_fraction().as_deref());
        self.ltf_low_from_both_radius
            .set_text_string(combine_params.get_low_from_both_radius().as_deref());
    }

    /// Java package-private final `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.patch_region_model_header.set_state(Some(
            screen_state.get_combine_final_patch_region_header_state(),
        ));
        self.patchcorr_header.set_state(Some(
            screen_state.get_combine_final_patchcorr_header_state(),
        ));
        // Upstream bug fixed in translation (FinalCombinePanel.java:565): Java
        // restores matchorwarpHeader from getCombineFinalPatchcorrHeaderState(),
        // the Patchcorr header's state (a copy-paste slip; ReconScreenState
        // keeps a separate CombineFinalMatchorwarpHeaderState that nothing else
        // reads or writes), so the Matchorwarp panel always opened and closed
        // with the Patchcorr panel.  We use the Matchorwarp header's own state.
        self.matchorwarp_header.set_state(Some(
            screen_state.get_combine_final_matchorwarp_header_state(),
        ));
        self.volcombine_header.set_state(Some(
            screen_state.get_combine_final_volcombine_header_state(),
        ));
        self.btn_patchcorr_restart.set_button_state(
            screen_state
                .get_button_state(self.btn_patchcorr_restart.get_button_state_key().as_deref()),
        );
        self.btn_matchorwarp_restart.set_button_state(
            screen_state.get_button_state(
                self.btn_matchorwarp_restart
                    .get_button_state_key()
                    .as_deref(),
            ),
        );
        self.btn_volcombine_restart.set_button_state(
            screen_state.get_button_state(
                self.btn_volcombine_restart
                    .get_button_state_key()
                    .as_deref(),
            ),
        );
        // if the kernal sigma value isn't coming from the comscript, get it from the
        // .edf, if it exists
        if !self.cb_kernel_sigma.is_selected() {
            let kernel_sigma = screen_state.get_patchcorr_kernel_sigma();
            if let Some(kernel_sigma) = kernel_sigma {
                self.tf_kernel_sigma
                    .set_text_string(Some(&kernel_sigma.to_string()));
            }
        }
    }

    /// Java package-private final `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.patch_region_model_header.get_state(Some(
            screen_state.get_combine_final_patch_region_header_state(),
        ));
        self.patchcorr_header.get_state(Some(
            screen_state.get_combine_final_patchcorr_header_state(),
        ));
        // Upstream bug fixed in translation (FinalCombinePanel.java:586): see
        // set_parameters_recon_screen_state; Java saved matchorwarpHeader into the
        // Patchcorr header's state, overwriting what patchcorrHeader had just
        // saved.
        self.matchorwarp_header.get_state(Some(
            screen_state.get_combine_final_matchorwarp_header_state(),
        ));
        self.volcombine_header.get_state(Some(
            screen_state.get_combine_final_volcombine_header_state(),
        ));
        screen_state.set_patchcorr_kernel_sigma(self.tf_kernel_sigma.get_text_void().as_deref());
    }

    /// Java package-private final `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_patch_region_model
            .get_component()
            .set_visible(visible);
        self.pnl_patchcorr.get_component().set_visible(visible);
        self.pnl_matchorwarp.get_component().set_visible(visible);
        self.pnl_volcombine.get_component().set_visible(visible);
        self.update_patch_vector_model_display();
    }

    /// Java package-private final `getVolcombineButtonName()`.
    pub fn get_volcombine_button_name(&self) -> String {
        ProcessName::VOLCOMBINE.to_string()
    }

    /// Java package-private `setPatchcrawl3DParams(ConstPatchcrawl3DParam)`.
    /// Set the values of the patchcrawl3D UI objects from the
    /// ConstPatchcrawl3DParam object.
    pub fn set_patchcrawl3_d_params(&self, patchrawl_param: &ConstPatchcrawl3DParam) {
        self.cb_use_patch_region_model
            .set_selected_boolean(patchrawl_param.is_use_boundary_model());
        self.ltf_x_patch_size
            .set_text_int(patchrawl_param.get_x_patch_size());
        self.ltf_y_patch_size
            .set_text_int(patchrawl_param.get_y_patch_size());
        self.ltf_z_patch_size
            .set_text_int(patchrawl_param.get_z_patch_size());
        self.ltf_x_n_patches.set_text_int(patchrawl_param.get_nx());
        self.ltf_y_n_patches.set_text_int(patchrawl_param.get_ny());
        self.ltf_z_n_patches.set_text_int(patchrawl_param.get_nz());
        self.ltf_x_low.set_text_int(patchrawl_param.get_x_low());
        self.ltf_x_high.set_text_int(patchrawl_param.get_x_high());
        self.ltf_y_low.set_text_int(patchrawl_param.get_y_low());
        self.ltf_y_high.set_text_int(patchrawl_param.get_y_high());
        self.ltf_z_low.set_text_int(patchrawl_param.get_z_low());
        self.ltf_z_high.set_text_int(patchrawl_param.get_z_high());
        self.ltf_initial_shift_x
            .set_text_string(Some(&patchrawl_param.get_initial_shift_x()));
        self.ltf_initial_shift_y
            .set_text_string(Some(&patchrawl_param.get_initial_shift_y()));
        self.ltf_initial_shift_z
            .set_text_string(Some(&patchrawl_param.get_initial_shift_z()));
        self.cb_kernel_sigma
            .set_selected_boolean(patchrawl_param.is_kernel_sigma_active());
        self.tf_kernel_sigma
            .set_text_string(Some(&patchrawl_param.get_kernel_sigma().to_string()));
        self.update_kernel_sigma();
    }

    /// Java package-private `setReductionFactorParams(ConstSetParam)`.
    pub fn set_reduction_factor_params(&self, set_param: Option<&SetParam>) {
        let Some(set_param) = set_param.filter(|set_param| set_param.is_valid()) else {
            return;
        };
        self.ltf_reduction_factor
            .set_text_string(set_param.get_value().as_deref());
    }

    /// Java package-private `setLowFromBothRadiusParams(ConstSetParam)`.
    pub fn set_low_from_both_radius_params(&self, set_param: Option<&SetParam>) {
        let Some(set_param) = set_param.filter(|set_param| set_param.is_valid()) else {
            return;
        };
        self.ltf_low_from_both_radius
            .set_text_string(set_param.get_value().as_deref());
    }

    /// Java package-private `getReductionFactorParam(SetParam, boolean)`.
    pub fn get_reduction_factor_param(
        &self,
        param: Option<&mut SetParam>,
        do_validation: bool,
    ) -> bool {
        let Some(param) = param else {
            return false;
        };
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        match self.ltf_reduction_factor.get_text_boolean(do_validation) {
            Ok(text) => {
                param.set_value(text.as_deref());
                true
            }
            Err(_) => false,
        }
    }

    /// Java package-private `getLowFromBothRadiusParam(SetParam, boolean)`.
    pub fn get_low_from_both_radius_param(
        &self,
        param: Option<&mut SetParam>,
        do_validation: bool,
    ) -> bool {
        let Some(param) = param else {
            return false;
        };
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        match self
            .ltf_low_from_both_radius
            .get_text_boolean(do_validation)
        {
            Ok(text) => {
                param.set_value(text.as_deref());
                true
            }
            Err(_) => false,
        }
    }

    /// Java package-private `enableReductionFactor(boolean)`.
    pub fn enable_reduction_factor(&self, enable: bool) {
        self.ltf_reduction_factor.set_enabled(enable);
    }

    /// Java package-private `enableLowFromBothRadius(boolean)`.
    pub fn enable_low_from_both_radius(&self, enable: bool) {
        self.ltf_low_from_both_radius.set_enabled(enable);
    }

    /// Java package-private `getPatchcrawl3DParams(Patchcrawl3DParam, boolean)
    /// throws NumberFormatException`.  Set the Patchcrawl3DParam object values
    /// from the UI values.
    ///
    /// `Err` is the rethrown NumberFormatException's message
    /// (`badParameter + " " + except.getMessage()`), which Java lets propagate
    /// to `ApplicationManager`.
    pub fn get_patchcrawl3_d_params(
        &self,
        patchcrawl3_d_param: &mut Patchcrawl3DParam,
        do_validation: bool,
    ) -> Result<bool, String> {
        let mut bad_parameter: String = String::new();
        // try { ... }
        // catch (NumberFormatException except) { rethrow with badParameter }
        // catch (FieldValidationFailedException e) { return false; }
        let result: Result<Result<(), String>, FieldValidationFailedException> = (|| {
            // Integer.parseInt(text); getText never returns null for a text field.
            macro_rules! parse_int {
                ($text:expr) => {
                    match java_lang_integer_parse_int($text.as_deref().unwrap_or("")) {
                        Ok(value) => value,
                        Err(message) => return Ok(Err(message)),
                    }
                };
            }
            bad_parameter = self
                .cb_use_patch_region_model
                .get_text_void()
                .unwrap_or_else(|| "null".to_string());
            patchcrawl3_d_param
                .set_use_boundary_model(self.cb_use_patch_region_model.is_selected());
            bad_parameter = self.ltf_x_patch_size.get_label();
            patchcrawl3_d_param.set_x_patch_size(parse_int!(
                self.ltf_x_patch_size.get_text_boolean(do_validation)?
            ));
            bad_parameter = self.ltf_y_patch_size.get_label();
            patchcrawl3_d_param.set_y_patch_size(parse_int!(
                self.ltf_y_patch_size.get_text_boolean(do_validation)?
            ));
            bad_parameter = self.ltf_z_patch_size.get_label();
            patchcrawl3_d_param.set_z_patch_size(parse_int!(
                self.ltf_z_patch_size.get_text_boolean(do_validation)?
            ));
            bad_parameter = self.ltf_x_n_patches.get_label();
            patchcrawl3_d_param.set_nx(parse_int!(
                self.ltf_x_n_patches.get_text_boolean(do_validation)?
            ));
            bad_parameter = self.ltf_y_n_patches.get_label();
            patchcrawl3_d_param.set_ny(parse_int!(
                self.ltf_y_n_patches.get_text_boolean(do_validation)?
            ));
            bad_parameter = self.ltf_z_n_patches.get_label();
            patchcrawl3_d_param.set_nz(parse_int!(
                self.ltf_z_n_patches.get_text_boolean(do_validation)?
            ));
            bad_parameter = self.ltf_x_low.get_label();
            patchcrawl3_d_param
                .set_x_low(parse_int!(self.ltf_x_low.get_text_boolean(do_validation)?));
            bad_parameter = self.ltf_x_high.get_label();
            patchcrawl3_d_param
                .set_x_high(parse_int!(self.ltf_x_high.get_text_boolean(do_validation)?));
            bad_parameter = self.ltf_y_low.get_label();
            patchcrawl3_d_param
                .set_y_low(parse_int!(self.ltf_y_low.get_text_boolean(do_validation)?));
            bad_parameter = self.ltf_y_high.get_label();
            patchcrawl3_d_param
                .set_y_high(parse_int!(self.ltf_y_high.get_text_boolean(do_validation)?));
            bad_parameter = self.ltf_z_low.get_label();
            patchcrawl3_d_param
                .set_z_low(parse_int!(self.ltf_z_low.get_text_boolean(do_validation)?));
            bad_parameter = self.ltf_z_high.get_label();
            patchcrawl3_d_param
                .set_z_high(parse_int!(self.ltf_z_high.get_text_boolean(do_validation)?));
            bad_parameter = self.ltf_initial_shift_x.get_label();
            patchcrawl3_d_param.set_initial_shift_x(
                self.ltf_initial_shift_x
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            bad_parameter = self.ltf_initial_shift_y.get_label();
            patchcrawl3_d_param.set_initial_shift_y(
                self.ltf_initial_shift_y
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            bad_parameter = self.ltf_initial_shift_z.get_label();
            patchcrawl3_d_param.set_initial_shift_z(
                self.ltf_initial_shift_z
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            bad_parameter = self
                .cb_kernel_sigma
                .get_text_void()
                .unwrap_or_else(|| "null".to_string());
            patchcrawl3_d_param.set_kernel_sigma(
                self.cb_kernel_sigma.is_selected(),
                self.tf_kernel_sigma
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            Ok(Ok(()))
        })();
        match result {
            Ok(Ok(())) => Ok(true),
            Ok(Err(message)) => Err(format!("{bad_parameter} {message}")),
            Err(_) => Ok(false),
        }
    }

    /// Java package-private `setMatchorwarpParams(ConstMatchorwarpParam)`.  Set
    /// the values of the matchorwarp UI objects from the ConstMatchorwarpParam
    /// object.
    pub fn set_matchorwarp_params(&self, matchorwarp_param: &dyn ConstMatchorwarpParam) {
        self.ltf_warp_limit
            .set_text_string(Some(&matchorwarp_param.get_warp_limits()));
        self.ltf_refine_limit
            .set_text_string(Some(&matchorwarp_param.get_refine_limit()));

        if matchorwarp_param.is_x_lower_exclude_set() {
            self.ltf_x_lower_exclude
                .set_text_int(matchorwarp_param.get_x_lower_exclude());
        }
        if matchorwarp_param.is_x_upper_exclude_set() {
            self.ltf_x_upper_exclude
                .set_text_int(matchorwarp_param.get_x_upper_exclude());
        }

        if matchorwarp_param.is_z_lower_exclude_set() {
            self.ltf_z_lower_exclude
                .set_text_int(matchorwarp_param.get_z_lower_exclude());
        }

        if matchorwarp_param.is_z_upper_exclude_set() {
            self.ltf_z_upper_exclude
                .set_text_int(matchorwarp_param.get_z_upper_exclude());
        }

        self.cb_use_linear_interpolation
            .set_selected_boolean(matchorwarp_param.is_linear_interpolation());

        // when loading into the dialog, matchorwarp takes precidence over patchcorr
        self.cb_use_patch_region_model
            .set_selected_boolean(matchorwarp_param.is_use_model_file());
    }

    /// Java package-private `getMatchorwarpParams(MatchorwarpParam, boolean)
    /// throws NumberFormatException`.  Set the MatchorwarpParam object values
    /// from the UI values.
    ///
    /// The declared NumberFormatException has no source here: none of the
    /// `MatchorwarpParam` setters called parses a number (the Java rethrow
    /// with `badParameter` is unreachable), so the result is a plain `bool`.
    pub fn get_matchorwarp_params(
        &self,
        matchorwarp_param: &mut MatchorwarpParam,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let result: Result<(), FieldValidationFailedException> = (|| {
            let mut _bad_parameter: String;

            _bad_parameter = self
                .cb_use_patch_region_model
                .get_text_void()
                .unwrap_or_else(|| "null".to_string());
            if self.cb_use_patch_region_model.is_selected() {
                matchorwarp_param.set_default_model_file();
            } else {
                matchorwarp_param.set_model_file(Some(""));
            }

            _bad_parameter = self.ltf_warp_limit.get_label();
            matchorwarp_param.set_warp_limits(
                self.ltf_warp_limit
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );

            _bad_parameter = self.ltf_refine_limit.get_label();
            matchorwarp_param.set_refine_limit(
                self.ltf_refine_limit
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );

            _bad_parameter = self.ltf_x_lower_exclude.get_label();
            if !self.ltf_x_lower_exclude.is_empty() {
                matchorwarp_param.set_x_lower_exclude_string(
                    self.ltf_x_lower_exclude
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                matchorwarp_param.reset_x_lower_exclude();
            }
            _bad_parameter = self.ltf_x_upper_exclude.get_label();
            if !self.ltf_x_upper_exclude.is_empty() {
                matchorwarp_param.set_x_upper_exclude_string(
                    self.ltf_x_upper_exclude
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                matchorwarp_param.reset_x_upper_exclude();
            }
            _bad_parameter = self.ltf_z_lower_exclude.get_label();
            if !self.ltf_z_lower_exclude.is_empty() {
                matchorwarp_param.set_z_lower_exclude_string(
                    self.ltf_z_lower_exclude
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                matchorwarp_param.reset_z_lower_exclude();
            }
            _bad_parameter = self.ltf_z_upper_exclude.get_label();
            if !self.ltf_z_upper_exclude.is_empty() {
                matchorwarp_param.set_z_upper_exclude_string(
                    self.ltf_z_upper_exclude
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                matchorwarp_param.reset_z_upper_exclude();
            }
            _bad_parameter = self
                .cb_use_linear_interpolation
                .get_text_void()
                .unwrap_or_else(|| "null".to_string());
            matchorwarp_param
                .set_linear_interpolation(self.cb_use_linear_interpolation.is_selected());
            Ok(())
        })();
        result.is_ok()
    }

    /// Java package-private `getProcessingMethod()`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        if self.cb_parallel_process.is_enabled() && self.cb_parallel_process.is_selected() {
            return ProcessingMethod::PpCpu;
        }
        ProcessingMethod::LocalCpu
    }

    /// Java private `sendProcessingMethodMessage()`.
    fn send_processing_method_message(&self) {
        let Some(dialog) = self.tomogram_combination_dialog.upgrade() else {
            return;
        };
        let origin: Rc<dyn ProcessInterface> = dialog;
        if let Some(mediator) = self
            .application_manager
            .get_processing_method_mediator(Some(AxisID::First))
        {
            mediator.set_method_process_interface_processing_method(
                &origin,
                self.get_processing_method(),
            );
        }
    }

    /// Java private `updateKernelSigma()`.
    fn update_kernel_sigma(&self) {
        self.tf_kernel_sigma
            .set_enabled(self.cb_kernel_sigma.is_selected());
    }

    /// Java package-private `updatePatchVectorModelDisplay()`.
    pub fn update_patch_vector_model_display(&self) {
        let manager: &'static dyn BaseManager = self.application_manager;
        let enable = dataset_files::get_patch_vector_model(manager).exists();
        self.btn_patch_vector_model.set_enabled(enable);
        self.btn_replace_patch_out.set_enabled(enable);
    }

    /// Java private `setToolTipText()`.  Initialize the tooltip text.
    fn set_tool_tip_text(&self) {
        let mut text: String;
        let mut adoc_combine_fft: Option<*mut Autodoc> = None;
        let mut adoc_corrsearch3d: Option<*mut Autodoc> = None;
        let manager: &'static dyn BaseManager = self.application_manager;
        // SAFETY (all `unsafe` below): `AutodocFactory` owns every autodoc it
        // returns for the life of the process (the Java GC-owned singletons), so
        // the pointers stay valid for this method.
        //
        // Java's one try block: an exception from the first getInstance skips
        // the second.
        let result = (|| -> Result<(), LogFileError> {
            adoc_combine_fft = Some(unsafe {
                autodoc_factory::get_instance(
                    Some(manager),
                    Some(autodoc_factory::COMBINE_FFT),
                    AxisID::Only,
                    false,
                )
            }?);
            adoc_corrsearch3d = Some(unsafe {
                autodoc_factory::get_instance(
                    Some(manager),
                    Some(autodoc_factory::CORR_SEARCH_3D),
                    AxisID::Only,
                    false,
                )
            }?);
            Ok(())
        })();
        match result {
            Ok(()) => {}
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except)
            // { except.printStackTrace(); }
            Err(except) => eprintln!("{except}"),
        }
        let adoc_combine_fft: Option<&dyn ReadOnlyAutodoc> =
            adoc_combine_fft.map(|autodoc| unsafe { &*autodoc } as &dyn ReadOnlyAutodoc);
        let adoc_corrsearch3d: Option<&dyn ReadOnlyAutodoc> =
            adoc_corrsearch3d.map(|autodoc| unsafe { &*autodoc } as &dyn ReadOnlyAutodoc);
        self.ltf_x_patch_size
            .set_tool_tip_text(Some("Size of correlation patches in X."));
        self.ltf_y_patch_size
            .set_tool_tip_text(Some("Size of correlation patches in Y."));
        self.ltf_z_patch_size
            .set_tool_tip_text(Some("Size of correlation patches in Z."));
        self.btn_patchsize_increase
            .set_tool_tip_text(Some("Increase all patch dimensions by 20%."));
        self.btn_patchsize_decrease
            .set_tool_tip_text(Some("Decrease all patch dimensions by 20%."));
        self.ltf_x_n_patches
            .set_tool_tip_text(Some("Number of patches to correlate in the X dimension."));
        self.ltf_y_n_patches
            .set_tool_tip_text(Some("Number of patches to correlate in the Y dimension."));
        self.ltf_z_n_patches
            .set_tool_tip_text(Some("Number of patches to correlate in the Z dimension."));
        self.ltf_x_low.set_tool_tip_text(Some(
            "Minimum X coordinate for left edge of correlation patches.",
        ));
        self.ltf_y_low.set_tool_tip_text(Some(
            "Minimum Y coordinate for upper edge of correlation patches.",
        ));
        self.ltf_z_low.set_tool_tip_text(Some(
            "Minimum Z coordinate for top edge of correlation patches.",
        ));
        self.ltf_x_high.set_tool_tip_text(Some(
            "Maximum X coordinate for right edge of correlation patches.",
        ));
        self.ltf_y_high.set_tool_tip_text(Some(
            "Maximum Y coordinate for lower edge of correlation patches.",
        ));
        self.ltf_z_high.set_tool_tip_text(Some(
            "Maximum Z coordinate for bottom edge of correlation patches.",
        ));
        self.btn_patch_vector_ccc_model.set_tool_tip_text(Some(
            "Open a patch vector model containing cross-correlation coefficients.  \
             In 3dmodv Objects, click on Values, and select on Show stored values.",
        ));
        self.btn_patchcorr_restart.set_tool_tip_text(Some(
            "Compute new displacements between patches by cross-correlation.",
        ));
        self.cb_use_patch_region_model
            .set_tool_tip_text_string(Some(
                "Use a model with contours around the areas where patches should be \
                 correlated to prevent bad patches outside those areas.",
            ));
        self.btn_patch_region_model.set_tool_tip_text(Some(
            "Open the volume being matched to and create the patch region model.",
        ));
        // Odd but kept (not clearly a slip): the two tooltips below describe
        // each other's fields.
        self.ltf_refine_limit.set_tool_tip_text(Some(
            "Enter a comma-separate series of mean residual limits to try in \
             succession when fitting warping transformations to the patch \
             displacements.",
        ));
        self.ltf_warp_limit.set_tool_tip_text(Some(
            "The mean residual limit for fit all patch displacements to a single \
             linear transformation.",
        ));
        self.ltf_x_lower_exclude.set_tool_tip_text(Some(
            "Exclude columns of patches on the left from the fits. Number of columns \
             of patches on the left to exclude from the fits.",
        ));
        self.ltf_x_upper_exclude.set_tool_tip_text(Some(
            "Exclude columns of patches on the right from the fits. Number of columns \
             of patches on the right to exclude from the fits.",
        ));
        self.ltf_z_lower_exclude.set_tool_tip_text(Some(
            "Exclude rows of patches on the bottom from the fits. Number of rows of \
             patches on the bottom in Y to exclude from the fits.",
        ));
        self.ltf_z_upper_exclude.set_tool_tip_text(Some(
            "Exclude rows of patches on the top from the fits. Number of rows of \
             patches on the top in Y to exclude from the fits.",
        ));

        self.cb_use_linear_interpolation
            .set_tool_tip_text_string(Some(
                "Uses linear instead of quadratic interpolation for transforming\
                 the volume with Matchvol or Warpvol.",
            ));
        self.btn_matchorwarp_restart.set_tool_tip_text(Some(
            "Restart the combine operation at Matchorwarp, which tries to fit \
             transformations to the patch displacements.",
        ));
        self.btn_volcombine_restart.set_tool_tip_text(Some(
            "Restart the combine operation at Volcombine, which combines volumes.",
        ));
        self.btn_matchorwarp_trial.set_tool_tip_text(Some(
            "Run Matchorwarp in trial mode; find transformations then stop.",
        ));
        self.btn_patch_vector_model.set_tool_tip_text(Some(
            "View the patch displacement vectors in and possibly \
             delete bad vectors.  To see the residual values, click on Values in \
             3dmodv Objects, and select on Show stored values.",
        ));
        self.btn_replace_patch_out.set_tool_tip_text(Some(
            "Replace the patch displacements with the vectors from the edited model.",
        ));
        self.btn_imod_matched_to
            .set_tool_tip_text(Some("View the volume being matched to in 3dmod."));
        text = "View the final combined volume.".to_string();
        self.btn_imod_combined
            .set_tool_tip_text(Some("View the final combined volume."));
        self.cb_no_volcombine.set_tool_tip_text_string(Some(
            "Stop after running Matchorwarp.  Use the \"Restart at Volcombine\" button to \
             continue.",
        ));

        text = "Filter by convolving in real space with a Gaussian kernel.  The \
                amount of filtering is controlled by the sigma of the Gaussian, in \
                pixels.  Higher sigma filters more.  Kernel filtering increases \
                execution time ~30% for sigma under 1.5 and ~2-fold for sigma 1.5 or \
                higher."
            .to_string();
        self.cb_kernel_sigma.set_tool_tip_text_string(Some(&text));
        self.tf_kernel_sigma.set_tool_tip_text(Some(&text));

        self.cb_parallel_process
            .set_tool_tip_text_string(Some(VOLCOMBINE_PARALLEL_PROCESSING_TOOL_TIP));

        if adoc_combine_fft.is_some() {
            self.ltf_reduction_factor.set_tool_tip_text(
                etomo_autodoc::get_tooltip(adoc_combine_fft, Some("ReductionFraction")).as_deref(),
            );
            self.ltf_low_from_both_radius.set_tool_tip_text(
                etomo_autodoc::get_tooltip(adoc_combine_fft, Some("LowFromBothRadius")).as_deref(),
            );
        }

        // A null autodoc gives a null tooltip (EtomoAutodoc.getTooltip).
        let initial_shift_text = etomo_autodoc::get_tooltip(
            adoc_corrsearch3d,
            Some(const_patchcrawl3d_param::INITIAL_SHIFT_XYZ_KEY),
        );
        self.ltf_initial_shift_x
            .set_tool_tip_text(initial_shift_text.as_deref());
        self.ltf_initial_shift_y
            .set_tool_tip_text(initial_shift_text.as_deref());
        self.ltf_initial_shift_z
            .set_tool_tip_text(initial_shift_text.as_deref());
    }
}

impl FinalCombineFields for FinalCombinePanel {
    /// Java public override `setUsePatchRegionModel(boolean)`.
    fn set_use_patch_region_model(&self, use_patch_region_model: bool) {
        self.cb_use_patch_region_model
            .set_selected_boolean(use_patch_region_model);
    }

    /// Java public override `isUsePatchRegionModel()`.
    fn is_use_patch_region_model(&self) -> bool {
        self.cb_use_patch_region_model.is_selected()
    }

    /// Java public override `isParallel()`.
    fn is_parallel(&self) -> bool {
        self.cb_parallel_process.is_selected()
    }

    /// Java public override `isParallelEnabled()`.
    fn is_parallel_enabled(&self) -> bool {
        self.cb_parallel_process.is_enabled()
    }

    /// Java public override `setXMin(String)`.
    fn set_x_min(&self, x_min: Option<&str>) {
        self.ltf_x_low.set_text_string(x_min);
    }

    /// Java public override `getXMin()`.
    fn get_x_min(&self) -> Option<String> {
        self.ltf_x_low.get_text_void()
    }

    /// Java public override `setXMax(String)`.
    fn set_x_max(&self, x_max: Option<&str>) {
        self.ltf_x_high.set_text_string(x_max);
    }

    /// Java public override `isEnabled()`.
    fn is_enabled(&self) -> bool {
        match self.tomogram_combination_dialog.upgrade() {
            Some(dialog) => dialog.is_tab_enabled(tomogram_combination_dialog::LBL_FINAL),
            None => false,
        }
    }

    /// Java public override `getXMax()`.
    fn get_x_max(&self) -> Option<String> {
        self.ltf_x_high.get_text_void()
    }

    /// Java public override `setYMin(String)`.
    fn set_y_min(&self, y_min: Option<&str>) {
        self.ltf_z_low.set_text_string(y_min);
    }

    /// Java public override `getYMin()`.
    fn get_y_min(&self) -> Option<String> {
        self.ltf_z_low.get_text_void()
    }

    /// Java public override `setYMax(String)`.
    fn set_y_max(&self, y_max: Option<&str>) {
        self.ltf_z_high.set_text_string(y_max);
    }

    /// Java public override `getYMax()`.
    fn get_y_max(&self) -> Option<String> {
        self.ltf_z_high.get_text_void()
    }

    /// Java public override `setZMin(String)`.
    fn set_z_min(&self, z_min: Option<&str>) {
        self.ltf_y_low.set_text_string(z_min);
    }

    /// Java public override `getZMin()`.
    fn get_z_min(&self) -> Option<String> {
        self.ltf_y_low.get_text_void()
    }

    /// Java public override `setZMax(String)`.
    fn set_z_max(&self, z_max: Option<&str>) {
        self.ltf_y_high.set_text_string(z_max);
    }

    /// Java public override `getZMax()`.
    fn get_z_max(&self) -> Option<String> {
        self.ltf_y_high.get_text_void()
    }

    /// Java public final override `setNoVolcombine(boolean)`.
    fn set_no_volcombine(&self, no_volcombine: bool) {
        self.cb_no_volcombine.set_selected_boolean(no_volcombine);
    }

    /// Java public final override `isNoVolcombine()`.
    fn is_no_volcombine(&self) -> bool {
        self.cb_no_volcombine.is_selected()
    }

    /// Java public final override `setParallel(boolean)`.
    fn set_parallel(&self, parallel: bool) {
        self.cb_parallel_process.set_selected_boolean(parallel);
        // Used for synchronization - don't send message to mediator
    }

    /// Java public final override `setParallelEnabled(boolean)`.
    fn set_parallel_enabled(&self, parallel_enabled: bool) {
        self.cb_parallel_process.set_enabled(parallel_enabled);
    }
}

impl Expandable for FinalCombinePanel {
    /// Java public override `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java public override `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.patch_region_model_header.equals_open_close(button) {
            self.pnl_patch_region_model_body
                .set_visible(button.is_expanded());
        } else if self.patchcorr_header.equals_open_close(button) {
            self.pnl_patchcorr_body.set_visible(button.is_expanded());
        } else if self.patchcorr_header.equals_advanced_basic(button) {
            self.update_advanced_patchcorr(button.is_expanded());
        } else if self.matchorwarp_header.equals_open_close(button) {
            self.pnl_matchorwarp_body.set_visible(button.is_expanded());
        } else if self.matchorwarp_header.equals_advanced_basic(button) {
            self.update_advanced_matchorwarp(button.is_expanded());
        } else if self.volcombine_header.equals_open_close(button) {
            self.pnl_volcombine_body.set_visible(button.is_expanded());
        } else if self.volcombine_header.equals_advanced_basic(button) {
            self.update_advanced_volcombine(button.is_expanded());
        }
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(AxisID::Only), Some(manager))
        });
    }
}

impl ContextMenu for FinalCombinePanel {
    /// Java public override `popUpContextMenu(MouseEvent)`.  Right mouse button
    /// context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = [Patchcrawl3DParam::get_title(), "Matchorwarp".to_string()];
        let man_page = [
            format!("{}.html", patchcrawl3d_param::COMMAND),
            "matchorwarp.html".to_string(),
        ];
        let log_file_label = [
            "Patchcorr".to_string(),
            "Matchorwarp".to_string(),
            "Volcombine".to_string(),
        ];
        let log_file = [
            "patchcorr.log".to_string(),
            "matchorwarp.log".to_string(),
            "volcombine.log".to_string(),
        ];
        let manager: &'static dyn BaseManager = self.application_manager;
        // The Java constructor's IllegalArgumentException (mismatched arrays)
        // cannot occur: the arrays are built in label/value pairs.
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root,
            mouse_event,
            Some("Patch Problems in Combining"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label),
            Some(&log_file),
            manager,
            AxisID::Only,
        );
    }
}

impl Run3dmodButtonContainer for FinalCombinePanel {
    /// Java public override `action(String, Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // The dialog owns this panel, so it is alive while the panel is.
        let Some(dialog) = self.tomogram_combination_dialog.upgrade() else {
            return;
        };
        // Synchronize this panel with the others
        dialog.synchronize(tomogram_combination_dialog::LBL_FINAL, true);
        // A null Run3dmodMenuOptions becomes `new Run3dmodMenuOptions()` where the
        // 3dmod state opens (ImodState.open).
        let menu_options = run_3dmod_menu_options.unwrap_or_default();
        // Decrease patch sizes by 20%
        // and then round to ints
        // since they are in
        // pixels
        //
        // try { ... } catch (FieldValidationFailedException e) {}
        //
        // Upstream bug fixed in translation (FinalCombinePanel.java:915-931):
        // `Integer.parseInt` on a patch size that is empty or not an integer
        // (validation of an INTEGER field lets an empty one through) throws a
        // NumberFormatException the method does not catch; Swing prints it on
        // the event thread and abandons the action part way (the X size may
        // already be changed).  We report the same message on stderr and stop
        // at the same point, without unwinding the UI thread.
        let _: Result<(), FieldValidationFailedException> = (|| {
            macro_rules! parse_int {
                ($text:expr) => {
                    match java_lang_integer_parse_int($text.as_deref().unwrap_or("")) {
                        Ok(value) => value,
                        Err(message) => {
                            eprintln!("java.lang.NumberFormatException: {message}");
                            return Ok(());
                        }
                    }
                };
            }
            if Some(command) == self.btn_patchsize_decrease.get_action_command().as_deref() {
                let value = parse_int!(self.ltf_x_patch_size.get_text_boolean(true)?);
                self.ltf_x_patch_size
                    .set_text_int(ui_utilities::java_math_round_f32(value as f32 / 1.2f32));
                let value = parse_int!(self.ltf_y_patch_size.get_text_boolean(true)?);
                self.ltf_y_patch_size
                    .set_text_int(ui_utilities::java_math_round_f32(value as f32 / 1.2f32));
                let value = parse_int!(self.ltf_z_patch_size.get_text_boolean(true)?);
                self.ltf_z_patch_size
                    .set_text_int(ui_utilities::java_math_round_f32(value as f32 / 1.2f32));
            }
            // Increase patch sizes by 20% and then round to ints since they are
            // in
            // pixels
            else if Some(command) == self.btn_patchsize_increase.get_action_command().as_deref() {
                let value = parse_int!(self.ltf_x_patch_size.get_text_boolean(true)?);
                self.ltf_x_patch_size
                    .set_text_int(ui_utilities::java_math_round_f32(value as f32 * 1.2f32));
                let value = parse_int!(self.ltf_y_patch_size.get_text_boolean(true)?);
                self.ltf_y_patch_size
                    .set_text_int(ui_utilities::java_math_round_f32(value as f32 * 1.2f32));
                let value = parse_int!(self.ltf_z_patch_size.get_text_boolean(true)?);
                self.ltf_z_patch_size
                    .set_text_int(ui_utilities::java_math_round_f32(value as f32 * 1.2f32));
            } else if Some(command) == self.btn_patchcorr_restart.get_action_command().as_deref() {
                let display: ProcessResultDisplayHandle = self.btn_patchcorr_restart.clone();
                self.application_manager.patchcorr_combine(
                    Some(display),
                    None,
                    deferred_3dmod_button,
                    menu_options,
                    self.dialog_type,
                    dialog.get_run_processing_method(),
                    self.is_parallel(),
                    !self.is_run_volcombine(),
                );
            } else if Some(command) == self.btn_matchorwarp_restart.get_action_command().as_deref()
            {
                let display: ProcessResultDisplayHandle = self.btn_matchorwarp_restart.clone();
                self.application_manager.matchorwarp_combine(
                    Some(display),
                    None,
                    deferred_3dmod_button,
                    menu_options,
                    self.dialog_type,
                    dialog.get_run_processing_method(),
                    self.is_parallel(),
                    !self.is_run_volcombine(),
                );
            } else if Some(command) == self.btn_matchorwarp_trial.get_action_command().as_deref() {
                self.application_manager.matchorwarp_trial(None);
            } else if Some(command) == self.btn_volcombine_restart.get_action_command().as_deref() {
                if self.cb_parallel_process.is_selected() {
                    self.application_manager
                        .splitcombine_process_series_deferred3dmod_button_run3dmod_menu_options_dialog_type_processing_method_boolean_boolean(
                            None,
                            deferred_3dmod_button,
                            run_3dmod_menu_options,
                            Some(self.dialog_type),
                            Some(dialog.get_run_processing_method()),
                            self.is_parallel(),
                            !self.is_run_volcombine(),
                        );
                } else {
                    let display: ProcessResultDisplayHandle = self.btn_volcombine_restart.clone();
                    self.application_manager.volcombine(
                        Some(display),
                        None,
                        deferred_3dmod_button,
                        menu_options,
                        self.dialog_type,
                    );
                }
            } else if Some(command) == self.btn_patch_vector_model.get_action_command().as_deref() {
                self.application_manager
                    .imod_patch_vector_model(imod_manager::PATCH_VECTOR_MODEL_KEY);
            } else if Some(command)
                == self
                    .btn_patch_vector_ccc_model
                    .get_action_command()
                    .as_deref()
            {
                self.application_manager
                    .imod_patch_vector_model(imod_manager::PATCH_VECTOR_CCC_MODEL_KEY);
            } else if Some(command) == self.btn_replace_patch_out.get_action_command().as_deref() {
                self.application_manager.model_to_patch();
            } else if Some(command) == self.cb_parallel_process.get_action_command().as_deref() {
                self.send_processing_method_message();
            } else if Some(command) == self.cb_kernel_sigma.get_action_command().as_deref() {
                self.update_kernel_sigma();
            } else if Some(command) == self.btn_patch_region_model.get_action_command().as_deref() {
                self.application_manager
                    .imod_patch_region_model(menu_options);
            } else if Some(command) == self.btn_imod_matched_to.get_action_command().as_deref() {
                self.application_manager
                    .imod_matched_to_tomogram(menu_options);
            } else if Some(command) == self.btn_imod_combined.get_action_command().as_deref() {
                self.application_manager
                    .imod_combined_tomogram(menu_options);
            }
            Ok(())
        })();
    }
}
