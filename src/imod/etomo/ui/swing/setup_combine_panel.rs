//! `IMOD/Etomo/src/etomo/ui/swing/SetupCombinePanel.java`.
//!
//! The Setup tab of `TomogramCombinationDialog`: the matching direction, the
//! Setup copy of `SolvematchPanel`, the patch parameters (patch sizes, patch
//! region model, X/Y/Z min and max), the volcombine controls, the temporary
//! directory, and the Create Combine Scripts / Start Combine buttons.
//!
//! Java `final class SetupCombinePanel implements ContextMenu,
//! InitialCombineFields, FinalCombineFields, Run3dmodButtonContainer,
//! Expandable`: an EDT object created as `Rc<Self>` by
//! [`SetupCombinePanel::get_instance`]; every method takes `&self`.  The
//! dialog is held weakly (it owns its panels).  The listener classes
//! (`SetupCombineActionListener`, `RBMatchToListener`, `CBPatchListener`) are
//! closures holding a weak reference to the panel; the first is kept in
//! `action_listener` so `removeListeners` can remove it.

use std::cell::Cell;
use std::rc::{Rc, Weak};

use super::beveled_border::BeveledBorder;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::final_combine_fields::FinalCombineFields;
use super::final_combine_panel;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::global_expand_button::GlobalExpandButton;
use super::initial_combine_fields::InitialCombineFields;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::patch_size_panel::PatchSizePanel;
use super::process_control_panel;
use super::process_interface::ProcessInterface;
use super::radio_button::RadioButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::solvematch_panel::SolvematchPanel;
use super::tomogram_combination_dialog::{self, TomogramCombinationDialog};
use super::ui_harness;
use super::ui_utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::combine_params::{self, CombineParams};
use crate::imod::etomo::comscript::const_combine_params::ConstCombineParams;
use crate::imod::etomo::comscript::const_patchcrawl3d_param::ConstPatchcrawl3DParam;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, JComponent, MouseEvent, MouseListener,
};
use crate::imod::etomo::logic::tomogram_tool::TomogramTool;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_integer_parse_int;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::match_mode::MatchMode;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::{self, ReconScreenState};
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::util::mrc_header::MRCHeader;

/// Java private static final `TOMOGRAM_SIZE_CHANGED_STRING`.
const TOMOGRAM_SIZE_CHANGED_STRING: &str = "THE TOMOGRAM HAS CHANGED - check min and max values";

/// Java `final class SetupCombinePanel implements ContextMenu,
/// InitialCombineFields, FinalCombineFields, Run3dmodButtonContainer,
/// Expandable`.
pub struct SetupCombinePanel {
    /// Java private final `pnlRoot = new EtomoPanel()`.
    pnl_root: Rc<EtomoPanel>,
    /// Java private final `pnlToSelector = new EtomoPanel()`.
    pnl_to_selector: Rc<EtomoPanel>,
    /// Java private final `pnlRBToSelector = new JPanel()`.
    pnl_rb_to_selector: Rc<JComponent>,
    /// Java private final `lblEffectWarning`.
    lbl_effect_warning: Rc<JComponent>,
    /// Java private final `bgToSelector = new ButtonGroup()`.
    #[allow(dead_code)]
    bg_to_selector: Rc<ButtonGroup>,
    /// Java private final `rbBtoA`.
    rb_bto_a: Rc<RadioButton>,
    /// Java private final `rbAtoB`.
    rb_ato_b: Rc<RadioButton>,
    /// Java private final `pnlPatchAndMinMax = new EtomoPanel()`.
    pnl_patch_and_min_max: Rc<EtomoPanel>,
    /// Java private final `pnlPatchAndMinMaxBody = new JPanel()`.
    pnl_patch_and_min_max_body: Rc<JComponent>,
    /// Java private final `cbPatchRegionModel`.
    cb_patch_region_model: Rc<CheckBox>,
    /// Java private final `btnPatchRegionModel`.
    btn_patch_region_model: Rc<Run3dmodButton>,
    /// Java private final `pnlVolcombineControls = new EtomoPanel()`.
    pnl_volcombine_controls: Rc<EtomoPanel>,
    /// Java private final `pnlVolcombineControlsBody = new JPanel()`.
    pnl_volcombine_controls_body: Rc<JComponent>,
    /// Java private final `ltfXMin`.
    ltf_x_min: Rc<LabeledTextField>,
    /// Java private final `ltfXMax`.
    ltf_x_max: Rc<LabeledTextField>,
    /// Java private final `ltfYMin`.
    ltf_y_min: Rc<LabeledTextField>,
    /// Java private final `ltfYMax`.
    ltf_y_max: Rc<LabeledTextField>,
    /// Java private final `ltfZMin`.
    ltf_z_min: Rc<LabeledTextField>,
    /// Java private final `ltfZMax`.
    ltf_z_max: Rc<LabeledTextField>,
    /// Java private final `pnlTempDirectory = new EtomoPanel()`.
    pnl_temp_directory: Rc<EtomoPanel>,
    /// Java private final `pnlTempDirectoryBody = new JPanel()`.
    pnl_temp_directory_body: Rc<JComponent>,
    /// Java private final `ltfTempDirectory`.
    ltf_temp_directory: Rc<LabeledTextField>,
    /// Java private final `cbManualCleanup`.
    cb_manual_cleanup: Rc<CheckBox>,
    /// Java private final `btnImodVolumeA`.
    btn_imod_volume_a: Rc<Run3dmodButton>,
    /// Java private final `btnImodVolumeB`.
    btn_imod_volume_b: Rc<Run3dmodButton>,
    /// Java private final `lTomogramSizeWarning = new JLabel()`.
    l_tomogram_size_warning: Rc<JComponent>,
    /// Java private final `btnDefaults`.
    btn_defaults: Rc<MultiLineButton>,
    /// Java private final `binningWarning = new JLabel()`.
    binning_warning: Rc<JComponent>,
    /// Java private final `cbNoVolcombine`.
    cb_no_volcombine: Rc<CheckBox>,
    /// Java private final `actionListener = new SetupCombineActionListener(this)`.
    action_listener: ActionListener,
    /// Java private final `cbAutoPatchFinalSize`.
    cb_auto_patch_final_size: Rc<CheckBox>,
    /// Java private final `ltfExtraResidualTargets`.
    ltf_extra_residual_targets: Rc<LabeledTextField>,
    /// Java private final `pspPatchTypeOrXYZ = PatchSizePanel.getInstance(false)`.
    psp_patch_type_or_xyz: Rc<PatchSizePanel>,
    /// Java private final `pspAutoPatchFinalSize = PatchSizePanel.getInstance(true)`.
    psp_auto_patch_final_size: Rc<PatchSizePanel>,

    /// Java private final `btnCreate`.
    btn_create: Rc<MultiLineButton>,
    /// Java private final `btnCombine`.
    btn_combine: Rc<Run3dmodButton>,
    /// Java private final `toSelectorHeader`.
    to_selector_header: Rc<PanelHeader>,
    /// Java private final `phPatchAndMinMax`.
    ph_patch_and_min_max: Rc<PanelHeader>,
    /// Java private final `tempDirectoryHeader`.
    temp_directory_header: Rc<PanelHeader>,
    /// Java private final `volcombineHeader`.
    volcombine_header: Rc<PanelHeader>,
    /// Java private final `cbParallelProcess`.
    cb_parallel_process: Rc<CheckBox>,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `tomogramCombinationDialog` (held weakly).
    tomogram_combination_dialog: Weak<TomogramCombinationDialog>,
    /// Java private final `applicationManager`.
    application_manager: &'static ApplicationManager,
    /// Java private final `pnlSolvematch`.
    pnl_solvematch: Rc<SolvematchPanel>,

    /// Java private `maxZMax = 0`.
    max_z_max: Cell<i32>,
    /// Java private `processingMethodLocked = false`.
    processing_method_locked: Cell<bool>,

    /// Java private `matchBtoA` (default false).
    match_bto_a: Cell<bool>,

    /// Java `this`.
    this: Weak<SetupCombinePanel>,
}

impl SetupCombinePanel {
    /// Java private constructor `SetupCombinePanel(TomogramCombinationDialog,
    /// ApplicationManager, DialogType)`.  Default constructor.
    fn new(
        parent: Weak<TomogramCombinationDialog>,
        app_mgr: &'static ApplicationManager,
        dialog_type: DialogType,
    ) -> Rc<SetupCombinePanel> {
        Rc::new_cyclic(|this: &Weak<SetupCombinePanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let expandable: Weak<dyn Expandable> = this.clone();
            // Field initializers, in declaration order.
            let pnl_root = EtomoPanel::new();
            let pnl_to_selector = EtomoPanel::new();
            let pnl_rb_to_selector = JComponent::new_panel();
            let lbl_effect_warning = JComponent::new_label(
                "You must create new combine scripts for some changes in these parameters to \
                 take effect.",
            );
            let bg_to_selector = ButtonGroup::new();
            let rb_bto_a = RadioButton::new_string_button_group(
                Some("Match the B tomogram to A"),
                Some(&bg_to_selector),
            );
            let rb_ato_b = RadioButton::new_string_button_group(
                Some("Match the A tomogram to B"),
                Some(&bg_to_selector),
            );
            let pnl_patch_and_min_max = EtomoPanel::new();
            let pnl_patch_and_min_max_body = JComponent::new_panel();
            let cb_patch_region_model = CheckBox::new_string(Some("Use patch region model"));
            let btn_patch_region_model =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Create/Edit Patch Region Model"),
                    Some(container.clone()),
                );
            let pnl_volcombine_controls = EtomoPanel::new();
            let pnl_volcombine_controls_body = JComponent::new_panel();
            let ltf_x_min =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("X axis min: "));
            let ltf_x_max =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("X axis max: "));
            let ltf_y_min =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y axis min: "));
            let ltf_y_max =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y axis max: "));
            let ltf_z_min = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(&format!("{}: ", combine_params::PATCH_Z_MIN_LABEL)),
            );
            let ltf_z_max = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(&format!("{}: ", combine_params::PATCH_Z_MAX_LABEL)),
            );
            let pnl_temp_directory = EtomoPanel::new();
            let pnl_temp_directory_body = JComponent::new_panel();
            let ltf_temp_directory = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some("Temporary directory: "),
            );
            let cb_manual_cleanup = CheckBox::new_string(Some("Manual cleanup"));
            let btn_imod_volume_a =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("3dmod Volume A"),
                    Some(container.clone()),
                );
            let btn_imod_volume_b =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("3dmod Volume B"),
                    Some(container),
                );
            let l_tomogram_size_warning = JComponent::new_label("");
            let btn_defaults = MultiLineButton::new_string(Some("Defaults"));
            let binning_warning = JComponent::new_label("");
            let cb_no_volcombine =
                CheckBox::new_string(Some(final_combine_panel::NO_VOLCOMBINE_TITLE));
            // Java `new SetupCombineActionListener(this)`.
            let adaptee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                }
            });
            let cb_auto_patch_final_size =
                CheckBox::new_string(Some("Use Automatic Patch Fitting"));
            let ltf_extra_residual_targets = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some("Extra warping limits: "),
            );
            let psp_patch_type_or_xyz = PatchSizePanel::get_instance(false);
            let psp_auto_patch_final_size = PatchSizePanel::get_instance(true);
            // Constructor body.
            // tomogramCombinationDialog = parent; applicationManager = appMgr;
            // this.dialogType = dialogType;
            let factory = app_mgr.get_process_result_display_factory(AxisID::Only);
            // (MultiLineButton) ...getCreateCombine(); (Run3dmodButton) ...getCombine()
            let btn_create = factory.get_create_combine();
            let btn_combine = factory.get_combine();
            let to_selector_header = PanelHeader::get_instance(
                Some("Tomogram Matching Relationship"),
                Some(expandable.clone()),
                Some(dialog_type),
            );
            // Create the solvematch panel
            let pnl_solvematch = SolvematchPanel::get_instance(
                parent.clone(),
                tomogram_combination_dialog::LBL_SETUP,
                app_mgr,
                recon_screen_state::COMBINE_SETUP_SOLVEMATCH_HEADER_GROUP.as_str(),
                dialog_type,
                false,
                None,
            );
            let ph_patch_and_min_max = PanelHeader::get_instance(
                Some("Patch Parameters for Refining Alignment"),
                Some(expandable.clone()),
                Some(dialog_type),
            );
            let temp_directory_header = PanelHeader::get_instance(
                Some("Intermediate Data Storage"),
                Some(expandable.clone()),
                Some(dialog_type),
            );
            // new CheckBox(tomogramCombinationDialog.parallelProcessCheckBoxText).
            // The dialog exists (it is constructing this panel); a gone dialog
            // (Java NullPointerException) gives an unlabelled check box.
            let parallel_process_check_box_text = parent
                .upgrade()
                .map(|parent| parent.parallel_process_check_box_text.clone())
                .unwrap_or_default();
            let cb_parallel_process = CheckBox::new_string(Some(&parallel_process_check_box_text));
            let volcombine_header = PanelHeader::get_instance(
                Some("Volcombine Controls"),
                Some(expandable),
                Some(dialog_type),
            );
            SetupCombinePanel {
                pnl_root,
                pnl_to_selector,
                pnl_rb_to_selector,
                lbl_effect_warning,
                bg_to_selector,
                rb_bto_a,
                rb_ato_b,
                pnl_patch_and_min_max,
                pnl_patch_and_min_max_body,
                cb_patch_region_model,
                btn_patch_region_model,
                pnl_volcombine_controls,
                pnl_volcombine_controls_body,
                ltf_x_min,
                ltf_x_max,
                ltf_y_min,
                ltf_y_max,
                ltf_z_min,
                ltf_z_max,
                pnl_temp_directory,
                pnl_temp_directory_body,
                ltf_temp_directory,
                cb_manual_cleanup,
                btn_imod_volume_a,
                btn_imod_volume_b,
                l_tomogram_size_warning,
                btn_defaults,
                binning_warning,
                cb_no_volcombine,
                action_listener,
                cb_auto_patch_final_size,
                ltf_extra_residual_targets,
                psp_patch_type_or_xyz,
                psp_auto_patch_final_size,
                btn_create,
                btn_combine,
                to_selector_header,
                ph_patch_and_min_max,
                temp_directory_header,
                volcombine_header,
                cb_parallel_process,
                dialog_type,
                tomogram_combination_dialog: parent,
                application_manager: app_mgr,
                pnl_solvematch,
                max_z_max: Cell::new(0),
                processing_method_locked: Cell::new(false),
                match_bto_a: Cell::new(false),
                this: this.clone(),
            }
        })
    }

    /// Java package-private static `getInstance(TomogramCombinationDialog,
    /// ApplicationManager, DialogType)`.
    pub fn get_instance(
        parent: Weak<TomogramCombinationDialog>,
        app_mgr: &'static ApplicationManager,
        dialog_type: DialogType,
    ) -> Rc<SetupCombinePanel> {
        let instance = SetupCombinePanel::new(parent, app_mgr, dialog_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        self.ltf_x_min.set_required(true);
        self.ltf_x_max.set_required(true);
        self.ltf_y_min.set_required(true);
        self.ltf_y_max.set_required(true);
        self.ltf_z_min.set_required(true);
        self.ltf_z_max.set_required(true);
        self.ltf_x_min.set_number_must_be_positive(true);
        self.ltf_x_max.set_number_must_be_positive(true);
        self.ltf_y_min.set_number_must_be_positive(true);
        self.ltf_y_max.set_number_must_be_positive(true);
        self.ltf_z_min.set_number_must_be_positive(true);
        self.ltf_z_max.set_number_must_be_positive(true);
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_combine.set_container(Some(container));
        self.l_tomogram_size_warning
            .set_foreground(Some(process_control_panel::COLOR_NOT_STARTED));
        self.l_tomogram_size_warning.set_visible(false);
        // Swing layout: lTomogramSizeWarning, lblEffectWarning and binningWarning
        // .setAlignmentX(Component.CENTER_ALIGNMENT); rbAtoB and rbBtoA
        // .setAlignmentX(Component.LEFT_ALIGNMENT).
        self.temp_directory_header.set_open(false);
        self.cb_auto_patch_final_size.set_selected_boolean(true);
        // panels
        let pnl_min_max = JComponent::new_panel();
        let pnl_bto_a = JComponent::new_panel();
        let pnl_ato_b = JComponent::new_panel();
        let pnl_parallel_process = JComponent::new_panel();
        let pnl_no_volcombine = JComponent::new_panel();
        let pnl_manual_cleanup = JComponent::new_panel();
        let pnl_patch = JComponent::new_panel();
        let pnl_auto_patch = JComponent::new_panel();
        let pnl_patch_size = JComponent::new_panel();
        let pnl_xyz = JComponent::new_panel();
        let pnl_patch_region_model = JComponent::new_panel();
        let pnl_button = JComponent::new_panel();
        let pnl_auto_patch_final_size_check_box = JComponent::new_panel();
        let _pnl_initial_matching = JComponent::new_panel();
        // Root
        let pnl_root = self.pnl_root.get_component();
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.Y_AXIS)).
        self.pnl_root
            .set_border(&BeveledBorder::new(Some("Combination Parameters")).get_border());
        pnl_root.add(&self.lbl_effect_warning);
        pnl_root.add(&self.pnl_to_selector.get_component());
        pnl_root.add(&self.pnl_solvematch.get_container());
        pnl_root.add(&self.pnl_patch_and_min_max.get_component());
        pnl_root.add(&self.pnl_volcombine_controls.get_component());
        pnl_root.add(&self.pnl_temp_directory.get_component());
        // Swing layout: pnlRoot.add(Box.createVerticalGlue()).
        ui_utilities::add_with_y_space(&pnl_root, &pnl_button);
        // ToSelector
        // Swing layout: pnlToSelector.setBorder(BorderFactory.createEtchedBorder());
        // setLayout(new BoxLayout(pnlToSelector, BoxLayout.Y_AXIS));
        // setAlignmentX(Component.CENTER_ALIGNMENT).
        self.pnl_to_selector.add(&self.to_selector_header);
        self.pnl_to_selector
            .get_component()
            .add(&self.pnl_rb_to_selector);
        // Swing layout: pnlToSelector.add(Box.createHorizontalGlue()).
        // RBToSelector
        // Swing layout: pnlRBToSelector.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)).
        self.pnl_rb_to_selector.add(&pnl_bto_a);
        self.pnl_rb_to_selector.add(&pnl_ato_b);
        // BtoA
        // Swing layout: pnlBtoA.setLayout(new BoxLayout(pnlBtoA, BoxLayout.X_AXIS));
        // setAlignmentX(Component.CENTER_ALIGNMENT); horizontal glue after.
        pnl_bto_a.add(&self.rb_bto_a.get_component());
        // AtoB
        // Swing layout: pnlAtoB.setLayout(new BoxLayout(pnlAtoB, BoxLayout.X_AXIS));
        // setAlignmentX(Component.CENTER_ALIGNMENT); horizontal glue after.
        pnl_ato_b.add(&self.rb_ato_b.get_component());
        // PatchAndMinMax
        // Swing layout: pnlPatchAndMinMax.setBorder(
        // BorderFactory.createEtchedBorder()); setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)).
        self.pnl_patch_and_min_max.add(&self.ph_patch_and_min_max);
        self.pnl_patch_and_min_max
            .get_component()
            .add(&self.pnl_patch_and_min_max_body);
        // PatchAndMinMaxBody
        // Swing layout: pnlPatchAndMinMaxBody.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)); rigid areas (FixedDim.x0_y5, x0_y20, x0_y1).
        self.pnl_patch_and_min_max_body
            .add(&self.l_tomogram_size_warning);
        self.pnl_patch_and_min_max_body.add(&pnl_patch);
        self.pnl_patch_and_min_max_body.add(&self.binning_warning);
        self.pnl_patch_and_min_max_body.add(&pnl_min_max);
        // Patch
        // Swing layout: pnlPatch.setLayout(new BoxLayout(pnlPatch, BoxLayout.X_AXIS));
        // a rigid area (FixedDim.x20_y0) between the panels.
        pnl_patch.add(&pnl_patch_size);
        pnl_patch.add(&pnl_auto_patch);
        // PatchSize
        // Swing layout: pnlPatchSize.setLayout(new BoxLayout(pnlPatchSize,
        // BoxLayout.Y_AXIS)).
        pnl_patch_size.add(&self.psp_patch_type_or_xyz.get_component());
        pnl_patch_size.add(&pnl_patch_region_model);
        // PatchRegionModel
        // Swing layout: pnlPatchRegionModel.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)).
        pnl_patch_region_model.add(&self.cb_patch_region_model.get_component());
        pnl_patch_region_model.add(&self.btn_patch_region_model.get_component());
        // AutoPatch
        // Swing layout: pnlAutoPatch.setLayout(new BoxLayout(pnlAutoPatch,
        // BoxLayout.Y_AXIS)); rigid areas (FixedDim.x0_y5, x0_y10, x0_y10).
        pnl_auto_patch.add(&pnl_auto_patch_final_size_check_box);
        pnl_auto_patch.add(&self.psp_auto_patch_final_size.get_component());
        pnl_auto_patch.add(&self.ltf_extra_residual_targets.get_component());
        // AutoPatchFinalSizeCheckBox
        // Swing layout: pnlAutoPatchFinalSizeCheckBox.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); horizontal glue after the check box.
        pnl_auto_patch_final_size_check_box.add(&self.cb_auto_patch_final_size.get_component());
        // MinMax
        // Swing layout: pnlMinMax.setLayout(new BoxLayout(pnlMinMax,
        // BoxLayout.X_AXIS)); a rigid area (FixedDim.x10_y0) between.
        pnl_min_max.add(&pnl_xyz);
        pnl_min_max.add(&self.btn_defaults.get_component());
        // XYZ
        // Swing layout: pnlXYZ.setLayout(new GridLayout(2, 3, 10, 10)).
        pnl_xyz.add(&self.ltf_x_min.get_container());
        pnl_xyz.add(&self.ltf_y_min.get_container());
        pnl_xyz.add(&self.ltf_z_min.get_container());
        pnl_xyz.add(&self.ltf_x_max.get_container());
        pnl_xyz.add(&self.ltf_y_max.get_container());
        pnl_xyz.add(&self.ltf_z_max.get_container());
        // VolcombineControls
        // Swing layout: pnlVolcombineControls.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)); setBorder(BorderFactory.createEtchedBorder()).
        self.pnl_volcombine_controls.add(&self.volcombine_header);
        self.pnl_volcombine_controls
            .get_component()
            .add(&self.pnl_volcombine_controls_body);
        // VolcombineControlsBody
        // Swing layout: pnlVolcombineControlsBody.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)).
        self.pnl_volcombine_controls_body.add(&pnl_parallel_process);
        self.pnl_volcombine_controls_body.add(&pnl_no_volcombine);
        // ParallelProcess
        // Swing layout: pnlParallelProcess.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); setAlignmentX(Box.CENTER_ALIGNMENT); horizontal glue.
        pnl_parallel_process.add(&self.cb_parallel_process.get_component());
        // NoVolcombine
        // Swing layout: pnlNoVolcombine.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); setAlignmentX(Box.CENTER_ALIGNMENT); horizontal glue.
        pnl_no_volcombine.add(&self.cb_no_volcombine.get_component());
        // TempDirectory
        // Swing layout: pnlTempDirectory.setBorder(
        // BorderFactory.createEtchedBorder()); setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)).
        self.pnl_temp_directory.add(&self.temp_directory_header);
        self.pnl_temp_directory
            .get_component()
            .add(&self.pnl_temp_directory_body);
        // TempDirectoryBody
        // Swing layout: pnlTempDirectoryBody.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)); a rigid area (FixedDim.x0_y5) first.
        self.pnl_temp_directory_body
            .add(&self.ltf_temp_directory.get_container());
        self.pnl_temp_directory_body.add(&pnl_manual_cleanup);
        // ManualCleanup
        // Swing layout: pnlManualCleanup.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); setAlignmentX(Component.CENTER_ALIGNMENT);
        // horizontal glue after the check box.
        pnl_manual_cleanup.add(&self.cb_manual_cleanup.get_component());
        // Button
        // Swing layout: pnlButton.setLayout(new BoxLayout(pnlButton,
        // BoxLayout.X_AXIS)); horizontal glue around and between the buttons.
        pnl_button.add(&self.btn_imod_volume_a.get_component());
        pnl_button.add(&self.btn_imod_volume_b.get_component());
        pnl_button.add(&self.btn_create.get_component());
        pnl_button.add(&self.btn_combine.get_component());
        // modify panels
        ui_utilities::align_components_x(&self.pnl_volcombine_controls_body, 0.5);
        // Swing layout: UIUtilities.setButtonSizeAll(pnlButton,
        // UIParameters.getInstance().getButtonDimension()).
        // update display
        self.update_patch_region_model();
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // Bind the buttons to the action listener
        self.btn_patch_region_model
            .add_action_listener(self.action_listener.clone());
        self.btn_imod_volume_a
            .add_action_listener(self.action_listener.clone());
        self.btn_imod_volume_b
            .add_action_listener(self.action_listener.clone());
        self.btn_create
            .add_action_listener(self.action_listener.clone());
        self.btn_combine
            .add_action_listener(self.action_listener.clone());
        self.btn_defaults
            .add_action_listener(self.action_listener.clone());
        self.cb_parallel_process
            .add_action_listener(Some(self.action_listener.clone()));
        self.cb_auto_patch_final_size
            .add_action_listener(Some(self.action_listener.clone()));
        // Bind the radio buttons to the action listener
        // Java `RBMatchToListener rbMatchToListener = new RBMatchToListener(this)`.
        let adaptee = self.this.clone();
        let rb_match_to_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.rb_match_to_action(event);
            }
        });
        self.rb_ato_b
            .add_action_listener(rb_match_to_listener.clone());
        self.rb_bto_a.add_action_listener(rb_match_to_listener);
        // Bind the patch region model check box to its action listener
        // Java `new CBPatchListener(this)`.
        let adaptee = self.this.clone();
        let cb_patch_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.cb_patch_region_action(event);
            }
        });
        self.cb_patch_region_model
            .add_action_listener(Some(cb_patch_listener));
        // Mouse listener for context menu
        let context_menu: Weak<dyn ContextMenu> = self.this.clone();
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.pnl_root
            .get_component()
            .add_mouse_listener(mouse_adapter);
    }

    /// Java package-private `removeListeners()`.
    pub fn remove_listeners(&self) {
        self.btn_create
            .remove_action_listener(&self.action_listener);
        self.btn_combine
            .remove_action_listener(&self.action_listener);
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_root.get_component()
    }

    /// Java package-private `show(boolean)`.
    pub fn show(&self, enable_combine: bool) {
        self.pnl_solvematch.show();
        self.update_tomogram_size_warning(enable_combine);
    }

    /// Java package-private `setDeferred3dmodButtons()`.
    pub fn set_deferred_3dmod_buttons(&self) {
        if let Some(dialog) = self.tomogram_combination_dialog.upgrade() {
            let deferred: Rc<dyn Deferred3dmodButton> = dialog.get_imod_combined_button();
            self.btn_combine
                .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        }
        self.pnl_solvematch.set_deferred_3dmod_buttons();
    }

    /// Java private `updateTomogramSizeWarning(boolean)`.  The parameter is not
    /// read.
    fn update_tomogram_size_warning(&self, _enable_combine: bool) {
        let changed = TomogramTool::is_tomogram_size_changed(
            self.application_manager,
            self.match_bto_a.get(),
            AxisID::Only,
        );
        self.l_tomogram_size_warning.set_visible(changed);
        if changed {
            self.l_tomogram_size_warning
                .set_text(TOMOGRAM_SIZE_CHANGED_STRING);
        }
    }

    /// Java package-private `getCombineResultDisplay()`.
    pub fn get_combine_result_display(&self) -> ProcessResultDisplayHandle {
        self.btn_combine.clone()
    }

    /// Java package-private `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data.set_combine_volcombine_parallel_boolean(self.cb_parallel_process.is_selected());
    }

    /// Java package-private `updateDisplay(boolean)`.
    pub fn update_display(&self, enable_combine: bool) {
        self.btn_combine.set_enabled(enable_combine);
        self.update_tomogram_size_warning(enable_combine);
        let auto_patch_final_size = self.cb_auto_patch_final_size.is_selected();
        self.psp_auto_patch_final_size
            .set_enabled(auto_patch_final_size);
        self.ltf_extra_residual_targets
            .set_enabled(auto_patch_final_size);
        self.pnl_solvematch.update_display();
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        // Parallel processing is optional in tomogram reconstruction, so only use it
        // if the user set it up.
        let combine_volcombine_parallel = meta_data.get_combine_volcombine_parallel();
        self.cb_parallel_process
            .set_enabled(!self.processing_method_locked.get());
        match combine_volcombine_parallel {
            None => {
                self.cb_parallel_process
                    .set_selected_boolean(meta_data.is_default_parallel());
            }
            Some(combine_volcombine_parallel) => {
                self.cb_parallel_process
                    .set_selected_boolean(combine_volcombine_parallel.is());
            }
        }
        self.send_processing_method_message();
    }

    /// Java package-private `lockProcessingMethod(boolean)`.
    pub fn lock_processing_method(&self, lock: bool) {
        self.processing_method_locked.set(lock);
        self.cb_parallel_process
            .set_enabled(!self.processing_method_locked.get());
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

    /// Java package-private `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.pnl_solvematch
            .get_parameters_recon_screen_state(screen_state);
        self.to_selector_header.get_state(Some(
            screen_state.get_combine_setup_to_selector_header_state(),
        ));
        self.ph_patch_and_min_max.get_state(Some(
            screen_state.get_combine_setup_patchcorr_header_state(),
        ));
        self.volcombine_header.get_state(Some(
            screen_state.get_combine_setup_volcombine_header_state(),
        ));
        self.temp_directory_header
            .get_state(Some(screen_state.get_combine_setup_temp_dir_header_state()));
    }

    /// Java package-private `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.pnl_solvematch
            .set_parameters_recon_screen_state(screen_state);
        self.to_selector_header.set_state(Some(
            screen_state.get_combine_setup_to_selector_header_state(),
        ));
        self.ph_patch_and_min_max.set_state(Some(
            screen_state.get_combine_setup_patchcorr_header_state(),
        ));
        self.volcombine_header.set_state(Some(
            screen_state.get_combine_setup_volcombine_header_state(),
        ));
        self.temp_directory_header
            .set_state(Some(screen_state.get_combine_setup_temp_dir_header_state()));
        self.btn_create.set_button_state(
            screen_state.get_button_state(self.btn_create.get_button_state_key().as_deref()),
        );
        self.btn_combine.set_button_state(
            screen_state.get_button_state(self.btn_combine.get_button_state_key().as_deref()),
        );
    }

    /// Java package-private `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.lbl_effect_warning.set_visible(visible);
        self.pnl_to_selector.get_component().set_visible(visible);
        self.pnl_solvematch.set_visible(visible);
        self.pnl_patch_and_min_max
            .get_component()
            .set_visible(visible);
        self.pnl_volcombine_controls
            .get_component()
            .set_visible(visible);
        self.pnl_temp_directory.get_component().set_visible(visible);
    }

    /// Java private `setBtoA(MatchMode)`.
    fn set_bto_a(&self, match_mode: Option<MatchMode>) {
        if match_mode.is_none() || match_mode == Some(MatchMode::BToA) {
            self.rb_bto_a.set_selected_boolean(true);
            self.match_bto_a.set(true);
        } else {
            self.rb_ato_b.set_selected_boolean(true);
            self.match_bto_a.set(false);
        }
    }

    /// Java package-private `setParameters(ConstCombineParams, boolean)`.  Set
    /// the parameters of the panel using the combineParams object.
    pub fn set_parameters_const_combine_params_boolean(
        &self,
        combine_params: &dyn ConstCombineParams,
        init: bool,
    ) {
        let match_mode = combine_params.get_match_mode();
        self.set_bto_a(match_mode);
        self.pnl_solvematch
            .set_parameters_const_combine_params_boolean(combine_params, init);
        self.psp_patch_type_or_xyz
            .set_parameters_const_combine_params(combine_params);
        self.psp_auto_patch_final_size
            .set_parameters_const_combine_params(combine_params);
        self.cb_patch_region_model
            .set_selected_boolean(combine_params.use_patch_region_model());
        self.ltf_x_min
            .set_text_int(combine_params.get_patch_x_min());
        self.ltf_x_max
            .set_text_int(combine_params.get_patch_x_max());
        self.ltf_y_min
            .set_text_int(combine_params.get_patch_y_min());
        self.ltf_y_max
            .set_text_int(combine_params.get_patch_y_max());
        self.ltf_z_min
            .set_text_const_etomo_number(Some(combine_params.get_patch_z_min()));
        self.ltf_z_max
            .set_text_const_etomo_number(Some(combine_params.get_patch_z_max()));
        self.max_z_max.set(combine_params.get_max_patch_z_max());
        self.ltf_temp_directory
            .set_text_string(Some(&combine_params.get_temp_directory()));
        self.cb_manual_cleanup
            .set_selected_boolean(combine_params.get_manual_cleanup());
        if combine_params.is_extra_residual_targets_set() {
            self.ltf_extra_residual_targets
                .set_text_string(combine_params.get_extra_residual_targets().as_deref());
        }
        // update
        self.set_auto_patch_z();
        self.update_patch_region_model();
    }

    /// Java package-private `setParameters(ConstPatchcrawl3DParam)`.
    pub fn set_parameters_const_patchcrawl3d_param(
        &self,
        patchrawl_param: &ConstPatchcrawl3DParam,
    ) {
        self.psp_patch_type_or_xyz
            .set_parameters_const_patchcrawl3d_param(patchrawl_param);
        self.ltf_x_min.set_text_int(patchrawl_param.get_x_low());
        self.ltf_x_max.set_text_int(patchrawl_param.get_x_high());
        // Assuming flipped
        self.ltf_y_min.set_text_int(patchrawl_param.get_z_low());
        self.ltf_y_max.set_text_int(patchrawl_param.get_z_high());
        self.ltf_z_min.set_text_int(patchrawl_param.get_y_low());
        self.ltf_z_max.set_text_int(patchrawl_param.get_y_high());
        self.set_auto_patch_z();
    }

    /// Java package-private `getParameters(CombineParams, boolean) throws
    /// NumberFormatException`.  Get the cobineParams from the panel.
    ///
    /// `Err` is the rethrown NumberFormatException's message
    /// (`badParameter + " " + except.getMessage()`), which Java lets propagate
    /// to `ApplicationManager.updateCombineParams` ("Number format error").
    pub fn get_parameters_combine_params_boolean(
        &self,
        combine_params: &mut CombineParams,
        do_validation: bool,
    ) -> Result<bool, String> {
        if !self
            .psp_patch_type_or_xyz
            .get_parameters(combine_params, do_validation)
        {
            return Ok(false);
        }
        if !self
            .psp_auto_patch_final_size
            .get_parameters(combine_params, do_validation)
        {
            return Ok(false);
        }
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        // The inner Result is the NumberFormatException's message; the outer
        // `Ok(false)` is an early `return false`.
        let result: Result<Result<bool, String>, FieldValidationFailedException> = (|| {
            if self.ltf_extra_residual_targets.is_enabled() {
                combine_params.set_extra_residual_targets(
                    self.ltf_extra_residual_targets
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                combine_params.reset_extra_residual_targets();
            }
            let mut bad_parameter: String = "unknown".to_string();
            // try { ... } catch (NumberFormatException except) {
            //   throw new NumberFormatException(badParameter + " " + message); }
            macro_rules! parse_int {
                ($text:expr) => {
                    match java_lang_integer_parse_int($text.as_deref().unwrap_or("")) {
                        Ok(value) => value,
                        Err(message) => return Ok(Err(format!("{bad_parameter} {message}"))),
                    }
                };
            }
            combine_params.set_match_mode_boolean(self.rb_bto_a.is_selected());
            if !self
                .pnl_solvematch
                .get_parameters_combine_params_boolean(combine_params, do_validation)
            {
                return Ok(Ok(false));
            }

            if self.cb_patch_region_model.is_selected() {
                combine_params.set_default_patch_region_model();
            } else {
                combine_params.set_patch_region_model("");
            }

            bad_parameter = self.ltf_x_min.get_label();
            combine_params
                .set_patch_x_min(parse_int!(self.ltf_x_min.get_text_boolean(do_validation)?));
            bad_parameter = self.ltf_x_max.get_label();
            combine_params
                .set_patch_x_max(parse_int!(self.ltf_x_max.get_text_boolean(do_validation)?));
            bad_parameter = self.ltf_y_min.get_label();
            combine_params
                .set_patch_y_min(parse_int!(self.ltf_y_min.get_text_boolean(do_validation)?));
            bad_parameter = self.ltf_y_max.get_label();
            combine_params
                .set_patch_y_max(parse_int!(self.ltf_y_max.get_text_boolean(do_validation)?));
            bad_parameter = self.ltf_z_min.get_label();
            combine_params
                .set_patch_z_min(self.ltf_z_min.get_text_boolean(do_validation)?.as_deref());
            bad_parameter = self.ltf_z_max.get_label();
            combine_params
                .set_patch_z_max(self.ltf_z_max.get_text_boolean(do_validation)?.as_deref());
            combine_params.set_max_patch_z_max(self.max_z_max.get());
            bad_parameter = "unknown".to_string();

            combine_params.set_temp_directory(
                &self
                    .ltf_temp_directory
                    .get_text_boolean(do_validation)?
                    .unwrap_or_default(),
            );

            combine_params.set_manual_cleanup(self.cb_manual_cleanup.is_selected());
            Ok(Ok(true))
        })();
        match result {
            Ok(result) => result,
            Err(_) => Ok(false),
        }
    }

    /// Java package-private `getCombineProcessResultDisplay()`.
    pub fn get_combine_process_result_display(&self) -> ProcessResultDisplayHandle {
        self.btn_combine.clone()
    }

    /// Java package-private `setBinningWarning(boolean)`.
    pub fn set_binning_warning(&self, binning_warning: bool) {
        if binning_warning {
            self.binning_warning
                .set_text("WARNING:  Coordinates must be selected from an unbinned 3dmod");
        } else {
            self.binning_warning.set_text("");
        }
    }

    /// Java private `setAutoPatchZ()`.  Sets empty Zmin/max fields to 1/Z if
    /// autoPatchFinalSize is selected.
    fn set_auto_patch_z(&self) {
        if self.cb_auto_patch_final_size.is_selected() {
            if self.ltf_z_min.is_empty() {
                self.ltf_z_min.set_text_string(Some("1"));
            }
            if self.ltf_z_max.is_empty() {
                let manager: &'static dyn BaseManager = self.application_manager;
                // MRCHeader.getInstance(BaseManager, AxisID, FileType) never
                // returns null.
                let Some(header) = MRCHeader::get_instance_from_file_type(
                    manager,
                    Some(AxisID::First),
                    &file_type::CLASS.tilt_output,
                ) else {
                    return;
                };
                // try { header.read(applicationManager); ltfZMax.setText(...); }
                // catch (InvalidParameterException | IOException e)
                // { e.printStackTrace(); }
                let read = header.borrow_mut().read_with_manager(manager);
                match read {
                    Ok(_) => {
                        // flipped
                        let n_rows = header.borrow().get_n_rows();
                        self.ltf_z_max.set_text_int(n_rows);
                    }
                    Err(e) => eprintln!("{e}"),
                }
            }
        }
    }

    /// Java private `resetXandY()`.
    fn reset_xand_y(&self) {
        let to_axis_id = if self.match_bto_a.get() {
            AxisID::First
        } else {
            AxisID::Second
        };
        let manager: &'static dyn BaseManager = self.application_manager;
        let Some(mrc_header) = MRCHeader::get_instance_in_dir(
            self.application_manager.get_property_user_dir().as_deref(),
            file_type::CLASS
                .tilt_output
                .get_file_name(Some(manager), Some(to_axis_id))
                .as_deref(), /*was DatasetFiles.getTomogramName(applicationManager, toAxisID)*/
            Some(AxisID::Only),
        ) else {
            return;
        };
        // try { ... } catch (IOException e) {} catch (InvalidParameterException e) {}
        let read = mrc_header.borrow_mut().read_with_manager(manager);
        match read {
            Ok(true) => {}
            Ok(false) | Err(_) => return,
        }
        let (xyborder, n_columns, n_sections) = {
            let mrc_header = mrc_header.borrow();
            (
                CombineParams::get_xy_border(&mrc_header),
                mrc_header.get_n_columns(),
                mrc_header.get_n_sections(),
            )
        };
        self.ltf_x_min.set_text_int(xyborder);
        self.ltf_x_max.set_text_int(n_columns - xyborder);
        self.ltf_y_min.set_text_int(xyborder);
        self.ltf_y_max.set_text_int(n_sections - xyborder);
        if let Some(dialog) = self.tomogram_combination_dialog.upgrade() {
            dialog.synchronize_from_current_tab();
        }
    }

    /// Java package-private `rbMatchToAction(ActionEvent)`.  Manage radio button
    /// action events.
    pub fn rb_match_to_action(&self, _event: &ActionEvent) {
        self.update_match_to();
        if let Some(dialog) = self.tomogram_combination_dialog.upgrade() {
            dialog.update_display();
        }
    }

    /// Java package-private `isChanged(TomogramState)`.
    pub fn is_changed(&self, state: &TomogramState) -> bool {
        if !state.get_combine_scripts_created().is() {
            return true;
        }
        let script_match_mode = state.get_combine_match_mode();
        script_match_mode.is_none()
            || (script_match_mode == Some(MatchMode::AToB) && !self.rb_ato_b.is_selected())
            || (script_match_mode == Some(MatchMode::BToA) && !self.rb_bto_a.is_selected())
            || TomogramTool::is_tomogram_size_changed(
                self.application_manager,
                self.match_bto_a.get(),
                AxisID::Only,
            )
    }

    /// Java private `updateMatchTo()`.
    fn update_match_to(&self) {
        // Swap the X and Y values if the matching state changes
        if (self.match_bto_a.get() && self.rb_ato_b.is_selected())
            || (!self.match_bto_a.get() && self.rb_bto_a.is_selected())
        {
            let mut temp = self.ltf_x_min.get_text_void();
            self.ltf_x_min
                .set_text_string(self.ltf_y_min.get_text_void().as_deref());
            self.ltf_y_min.set_text_string(temp.as_deref());
            temp = self.ltf_x_max.get_text_void();
            self.ltf_x_max
                .set_text_string(self.ltf_y_max.get_text_void().as_deref());
            self.ltf_y_max.set_text_string(temp.as_deref());
        }

        if self.rb_ato_b.is_selected() {
            self.match_bto_a.set(false);
        } else {
            self.match_bto_a.set(true);
        }
    }

    /// Java private `cbPatchRegionAction(ActionEvent)`.  Manage patch region
    /// check box actions.
    fn cb_patch_region_action(&self, _event: &ActionEvent) {
        self.update_patch_region_model();
    }

    /// Java private `updatePatchRegionModel()`.  Enable/disable the patch region
    /// model button.
    fn update_patch_region_model(&self) {
        self.btn_patch_region_model
            .set_enabled(self.cb_patch_region_model.is_selected());
    }

    // Java: a commented-out `updateStartCombine()`; nothing to translate.

    /// Java private `setToolTipText()`.  Initialize the tooltip text.
    fn set_tool_tip_text(&self) {
        self.rb_bto_a.set_tool_tip_text_string(Some(
            "Transform the B tomogram into the same orientation as the A tomogram.",
        ));
        self.rb_ato_b.set_tool_tip_text_string(Some(
            "Transform the A tomogram into the same orientation as the B tomogram.",
        ));
        self.cb_patch_region_model.set_tool_tip_text_string(Some(
            "Use a model with contours around the areas where patches should be \
             correlated to prevent bad patches outside those areas.",
        ));
        self.btn_patch_region_model.set_tool_tip_text(Some(
            "Open the volume being matched to and create the patch region model.",
        ));
        self.ltf_x_min.set_tool_tip_text(Some(
            "Minimum X coordinate for left edge of correlation patches.",
        ));
        self.ltf_x_max.set_tool_tip_text(Some(
            "Maximum X coordinate for right edge of correlation patches.",
        ));
        self.ltf_y_min.set_tool_tip_text(Some(
            "Minimum Y coordinate for upper edge of correlation patches.",
        ));
        self.ltf_y_max.set_tool_tip_text(Some(
            "Maximum Y coordinate for lower edge of correlation patches.",
        ));
        self.ltf_z_min.set_tool_tip_text(Some(
            "Minimum Z coordinate for top edge of correlation patches.",
        ));
        self.ltf_z_max.set_tool_tip_text(Some(
            "Maximum Z coordinate for bottom edge of correlation patches.",
        ));
        self.ltf_temp_directory.set_tool_tip_text(Some(
            "Specify a directory on local disk (e.g., /usr/tmp, or /scratch/myarea) \
             to avoid writing temporary files over a network.",
        ));
        self.cb_manual_cleanup.set_tool_tip_text_string(Some(
            "If using a temporary directory, select this option if you will want to \
             examine the *.mat file that will be left in it.",
        ));
        self.btn_imod_volume_a
            .set_tool_tip_text(Some("Display tomogram from axis A"));
        self.btn_imod_volume_b
            .set_tool_tip_text(Some("Display tomogram from axis B"));
        self.btn_create.set_tool_tip_text(Some(
            "Run setupcombine to create the com scripts for combining, using the \
             current parameters.",
        ));
        self.btn_combine.set_tool_tip_text(Some(
            "Start running the combine operation from the beginning.",
        ));
        self.cb_no_volcombine.set_tool_tip_text_string(Some(
            "Stop after running Matchorwarp.  Use the \"Restart at Volcombine\" button to \
             continue.",
        ));
        self.cb_parallel_process.set_tool_tip_text_string(Some(
            final_combine_panel::VOLCOMBINE_PARALLEL_PROCESSING_TOOL_TIP,
        ));
        self.cb_auto_patch_final_size.set_tool_tip_text_string(Some(
            "Use Autopatchfit to try a sequence of increasing patch sizes and more closely \
             spaced patches until target residual is achieved",
        ));
        let mut tooltip_start = "Use ";
        let mut tooltip_finish = " patches for refining the alignment with correlation, and \
                                  as starting size in automatic patch fitting.  This is \
                                  appropriate for feature-rich tomogram from binned CCD \
                                  camera images or from film.";
        self.psp_patch_type_or_xyz
            .set_small_tooltip(&format!("{tooltip_start}small{tooltip_finish}"));
        self.psp_patch_type_or_xyz
            .set_medium_tooltip(&format!("{tooltip_start}medium{tooltip_finish}"));
        self.psp_patch_type_or_xyz
            .set_large_tooltip(&format!("{tooltip_start}large{tooltip_finish}"));
        self.psp_patch_type_or_xyz.set_custom_tooltip(
            "Set size of patches to use for refining the alignment with correlation, and as \
             starting size in automatic patch fitting.",
        );
        tooltip_start = "Size in ";
        tooltip_finish = " of patches to use, or to start with in automatic patch fitting";
        self.psp_patch_type_or_xyz
            .set_x_tooltip(&format!("{tooltip_start}X{tooltip_finish}"));
        self.psp_patch_type_or_xyz
            .set_y_tooltip(&format!("{tooltip_start}Y{tooltip_finish}"));
        self.psp_patch_type_or_xyz
            .set_z_tooltip(&format!("{tooltip_start}Z{tooltip_finish}"));
        tooltip_start = "Use ";
        tooltip_finish = " as the biggest patches to try in automatic patch fitting.";
        self.psp_auto_patch_final_size
            .set_medium_tooltip(&format!("{tooltip_start}medium-sized{tooltip_finish}"));
        self.psp_auto_patch_final_size
            .set_large_tooltip(&format!("{tooltip_start}large{tooltip_finish}"));
        self.psp_auto_patch_final_size
            .set_extra_large_tooltip(&format!("{tooltip_start}extra large{tooltip_finish}"));
        self.psp_auto_patch_final_size
            .set_custom_tooltip("Set size of biggest patches to try in automatic patch fitting.");
        tooltip_start = "Maximum size in ";
        tooltip_finish = " of patches to use in automatic patch fitting";
        self.psp_auto_patch_final_size
            .set_x_tooltip(&format!("{tooltip_start}X{tooltip_finish}"));
        self.psp_auto_patch_final_size
            .set_y_tooltip(&format!("{tooltip_start}Y{tooltip_finish}"));
        self.psp_auto_patch_final_size
            .set_z_tooltip(&format!("{tooltip_start}Z{tooltip_finish}"));
        self.btn_defaults
            .set_tool_tip_text(Some("Reset X/Y/Z min and max values to initial defaults."));
        self.ltf_extra_residual_targets.set_tool_tip_text(Some(
            "Additional residual warping limits to try to reach on the last attempt with \
             automatic patch fitting",
        ));
    }
}

impl FinalCombineFields for SetupCombinePanel {
    /// Java public override `setNoVolcombine(boolean)`.
    fn set_no_volcombine(&self, no_volcombine: bool) {
        self.cb_no_volcombine.set_selected_boolean(no_volcombine);
    }

    /// Java public override `isNoVolcombine()`.
    fn is_no_volcombine(&self) -> bool {
        self.cb_no_volcombine.is_selected()
    }

    /// Java public override `setParallel(boolean)`.
    fn set_parallel(&self, parallel: bool) {
        self.cb_parallel_process.set_selected_boolean(parallel);
        // Used for synchronization - don't send message to mediator
    }

    /// Java public override `setParallelEnabled(boolean)`.
    fn set_parallel_enabled(&self, parallel_enabled: bool) {
        self.cb_parallel_process.set_enabled(parallel_enabled);
    }

    /// Java public override `isParallel()`.
    fn is_parallel(&self) -> bool {
        self.cb_parallel_process.is_selected()
    }

    /// Java public override `isParallelEnabled()`.
    fn is_parallel_enabled(&self) -> bool {
        self.cb_parallel_process.is_enabled()
    }

    /// Java public override `isEnabled()`.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Java public override `setUsePatchRegionModel(boolean)`.
    fn set_use_patch_region_model(&self, use_patch_region_model: bool) {
        self.cb_patch_region_model
            .set_selected_boolean(use_patch_region_model);
        self.update_patch_region_model();
    }

    /// Java public override `isUsePatchRegionModel()`.
    fn is_use_patch_region_model(&self) -> bool {
        self.cb_patch_region_model.is_selected()
    }

    /// Java public override `setXMin(String)`.
    fn set_x_min(&self, x_min: Option<&str>) {
        self.ltf_x_min.set_text_string(x_min);
    }

    /// Java public override `getXMin()`.
    fn get_x_min(&self) -> Option<String> {
        self.ltf_x_min.get_text_void()
    }

    /// Java public override `setXMax(String)`.
    fn set_x_max(&self, x_max: Option<&str>) {
        self.ltf_x_max.set_text_string(x_max);
    }

    /// Java public override `getXMax()`.
    fn get_x_max(&self) -> Option<String> {
        self.ltf_x_max.get_text_void()
    }

    /// Java public override `setYMin(String)`.
    fn set_y_min(&self, y_min: Option<&str>) {
        self.ltf_y_min.set_text_string(y_min);
    }

    /// Java public override `getYMin()`.
    fn get_y_min(&self) -> Option<String> {
        self.ltf_y_min.get_text_void()
    }

    /// Java public override `setYMax(String)`.
    fn set_y_max(&self, y_max: Option<&str>) {
        self.ltf_y_max.set_text_string(y_max);
    }

    /// Java public override `getYMax()`.
    fn get_y_max(&self) -> Option<String> {
        self.ltf_y_max.get_text_void()
    }

    /// Java public override `setZMin(String)`.
    fn set_z_min(&self, z_min: Option<&str>) {
        self.ltf_z_min.set_text_string(z_min);
        self.set_auto_patch_z();
    }

    /// Java public override `getZMin()`.
    fn get_z_min(&self) -> Option<String> {
        self.ltf_z_min.get_text_void()
    }

    /// Java public override `setZMax(String)`.
    fn set_z_max(&self, z_max: Option<&str>) {
        self.ltf_z_max.set_text_string(z_max);
        self.set_auto_patch_z();
    }

    /// Java public override `getZMax()`.
    fn get_z_max(&self) -> Option<String> {
        self.ltf_z_max.get_text_void()
    }
}

impl InitialCombineFields for SetupCombinePanel {
    /// Java public override `isUseCorrespondingPoints()`.
    fn is_use_corresponding_points(&self) -> bool {
        self.pnl_solvematch.is_use_corresponding_points()
    }

    /// Java public override `setUseCorrespondingPoints(boolean)`.
    fn set_use_corresponding_points(&self, use_: bool) {
        self.pnl_solvematch.set_use_corresponding_points(use_);
    }

    /// Java public override `isEnabled()`.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Java public override `isInitialVolumeMatching()`.
    fn is_initial_volume_matching(&self) -> bool {
        self.pnl_solvematch.is_initial_volume_matching()
    }

    /// Java public override `setInitialVolumeMatching(boolean)`.
    fn set_initial_volume_matching(&self, input: bool) {
        self.pnl_solvematch.set_initial_volume_matching(input);
    }

    /// Java public override `getMatchMode()`.
    fn get_match_mode(&self) -> Option<MatchMode> {
        if self.rb_bto_a.is_selected() {
            return Some(MatchMode::BToA);
        }
        Some(MatchMode::AToB)
    }

    /// Java public override `setMatchMode(MatchMode)`.
    fn set_match_mode(&self, match_mode: Option<MatchMode>) {
        if match_mode.is_none() {
            return;
        }
        self.set_bto_a(match_mode);
    }

    // InitialiCombineFields interface pass-thru
    /// Java public override `getSurfacesOrModels()`.
    fn get_surfaces_or_models(&self) -> FiducialMatch {
        self.pnl_solvematch.get_surfaces_or_models()
    }

    /// Java public override `setSurfacesOrModels(FiducialMatch)`.
    fn set_surfaces_or_models(&self, state: FiducialMatch) {
        self.pnl_solvematch.set_surfaces_or_models(state);
    }

    /// Java public override `isBinBy2()`.
    fn is_bin_by2(&self) -> bool {
        self.pnl_solvematch.is_bin_by2()
    }

    /// Java public override `setBinBy2(boolean)`.
    fn set_bin_by2(&self, state: bool) {
        self.pnl_solvematch.set_bin_by2(state);
    }

    /// Java public override `setFiducialMatchListA(String)`.
    fn set_fiducial_match_list_a(&self, fiducial_match_list_a: Option<&str>) {
        self.pnl_solvematch
            .set_fiducial_match_list_a(fiducial_match_list_a);
    }

    /// Java public override `setUseList(String)`.
    fn set_use_list(&self, use_list: Option<&str>) {
        self.pnl_solvematch.set_use_list(use_list);
    }

    /// Java public override `getUseList(boolean)`.
    fn get_use_list_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.pnl_solvematch.get_use_list_boolean(do_validation)
    }

    /// Java public override `getUseList()`.
    fn get_use_list_void(&self) -> Option<String> {
        self.pnl_solvematch.get_use_list_void()
    }

    /// Java public override `getFiducialMatchListA(boolean)`.
    fn get_fiducial_match_list_a_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.pnl_solvematch
            .get_fiducial_match_list_a_boolean(do_validation)
    }

    /// Java public override `getFiducialMatchListA()`.
    fn get_fiducial_match_list_a_void(&self) -> Option<String> {
        self.pnl_solvematch.get_fiducial_match_list_a_void()
    }

    /// Java public override `setFiducialMatchListB(String)`.
    fn set_fiducial_match_list_b(&self, fiducial_match_list_b: Option<&str>) {
        self.pnl_solvematch
            .set_fiducial_match_list_b(fiducial_match_list_b);
    }

    /// Java public override `getFiducialMatchListB(boolean)`.
    fn get_fiducial_match_list_b_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.pnl_solvematch
            .get_fiducial_match_list_b_boolean(do_validation)
    }

    /// Java public override `getFiducialMatchListB()`.
    fn get_fiducial_match_list_b_void(&self) -> Option<String> {
        self.pnl_solvematch.get_fiducial_match_list_b_void()
    }
}

impl Expandable for SetupCombinePanel {
    /// Java public override `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java public override `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.to_selector_header.equals_open_close(button) {
            self.pnl_rb_to_selector.set_visible(button.is_expanded());
        } else if self.ph_patch_and_min_max.equals_open_close(button) {
            self.pnl_patch_and_min_max_body
                .set_visible(button.is_expanded());
        } else if self.volcombine_header.equals_open_close(button) {
            self.pnl_volcombine_controls_body
                .set_visible(button.is_expanded());
        } else if self.temp_directory_header.equals_open_close(button) {
            self.pnl_temp_directory_body
                .set_visible(button.is_expanded());
        }
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(AxisID::Only), Some(manager))
        });
    }
}

impl ContextMenu for SetupCombinePanel {
    /// Java public override `popUpContextMenu(MouseEvent)`.  Right mouse btn
    /// context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = [
            "Solvematch".to_string(),
            "Matchshifts".to_string(),
            "Patchcrawl3d".to_string(),
            "Matchorwarp".to_string(),
        ];
        let man_page = [
            "solvematch.html".to_string(),
            "matchshifts.html".to_string(),
            "patchcrawl3d.html".to_string(),
            "matchorwarp.html".to_string(),
        ];
        let log_file_label = [
            "Transferfid".to_string(),
            "Solvematch".to_string(),
            "Patchcorr".to_string(),
            "Matchorwarp".to_string(),
            "Volcombine".to_string(),
        ];
        let log_file = [
            "transferfid.log".to_string(),
            "solvematch.log".to_string(),
            "patchcorr.log".to_string(),
            "matchorwarp.log".to_string(),
            "volcombine.log".to_string(),
        ];

        let manager: &'static dyn BaseManager = self.application_manager;
        // The Java constructor's IllegalArgumentException (mismatched arrays)
        // cannot occur: the arrays are built in label/value pairs.
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root.get_component(),
            mouse_event,
            Some("TOMOGRAM COMBINATION"),
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

impl Run3dmodButtonContainer for SetupCombinePanel {
    /// Java public override `action(String, Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.  Executes the action associated with command.
    /// Deferred3dmodButton is null if it comes from dialog's ActionListener.
    /// Otherwise is comes from a Run3dmodButton which called
    /// action(Run3dmodButton, Run3dmoMenuOptions).  In that case it will be null
    /// unless it was set in the Run3dmodButton.
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
        dialog.synchronize(tomogram_combination_dialog::LBL_SETUP, true);
        // A null Run3dmodMenuOptions becomes `new Run3dmodMenuOptions()` where the
        // 3dmod state opens (ImodState.open).
        let menu_options = run_3dmod_menu_options.unwrap_or_default();
        if Some(command) == self.btn_create.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_create.clone();
            self.update_tomogram_size_warning(
                self.application_manager
                    .create_combine_scripts(Some(display)),
            );
            dialog.update_display();
        } else if Some(command) == self.btn_combine.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_combine.clone();
            self.application_manager.combine(
                Some(display),
                None,
                deferred_3dmod_button,
                menu_options,
                self.dialog_type,
                dialog.get_run_processing_method(),
                self.pnl_solvematch.is_initial_volume_matching(),
                self.is_parallel(),
                self.is_no_volcombine(),
            );
        } else if Some(command) == self.cb_parallel_process.get_action_command().as_deref() {
            self.send_processing_method_message();
        } else if Some(command) == self.btn_defaults.get_action_command().as_deref() {
            self.reset_xand_y();
        } else if Some(command) == self.btn_patch_region_model.get_action_command().as_deref() {
            self.application_manager
                .imod_patch_region_model(menu_options);
        } else if Some(command) == self.btn_imod_volume_a.get_action_command().as_deref() {
            self.application_manager
                .imod_full_volume(AxisID::First, menu_options);
        } else if Some(command) == self.btn_imod_volume_b.get_action_command().as_deref() {
            self.application_manager
                .imod_full_volume(AxisID::Second, menu_options);
        } else if Some(command)
            == self
                .cb_auto_patch_final_size
                .get_action_command()
                .as_deref()
        {
            self.set_auto_patch_z();
            dialog.update_display();
        } else {
            dialog.update_display();
        }
    }
}
