//! `IMOD/Etomo/src/etomo/ui/swing/SolvematchPanel.java`.
//!
//! Handles solvematch and dualvolmatch (the Java notes that the name should be
//! InitialMatchPanel).  One instance sits on the Setup tab and one on the
//! Initial Match tab of `TomogramCombinationDialog`; the Initial Match copy
//! (`parentTitle == TomogramCombinationDialog.lblInitial`) adds the restart
//! button, the solvematch residual and center shift limits and the
//! dualvolmatch parameters.
//!
//! Java `final class SolvematchPanel implements Run3dmodButtonContainer,
//! Expandable, ActionListener`: an EDT object created as `Rc<Self>` by
//! [`SolvematchPanel::get_instance`]; every method takes `&self`.  The panel
//! is its own action listener (a closure holding a weak reference to it), and
//! it holds the dialog weakly (the dialog owns its panels).

use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use std::cell::Cell;
use std::rc::{Rc, Weak};
use std::sync::LazyLock;

use regex::Regex;

use super::check_box::CheckBox;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::fixed_dim;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::panel_header::PanelHeader;
use super::radio_button::RadioButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::tomogram_combination_dialog::{self, TomogramCombinationDialog};
use super::ui_harness;
use super::ui_utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::combine_params::CombineParams;
use crate::imod::etomo::comscript::const_combine_params::ConstCombineParams;
use crate::imod::etomo::comscript::const_solvematch_param::{self, ConstSolvematchParam};
use crate::imod::etomo::comscript::dualvolmatch_param::{self, DualvolmatchParam};
use crate::imod::etomo::comscript::solvematch_param::SolvematchParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::logic::combine_tool::CombineTool;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::autodoc::section::Section;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_panel_header_settings::ConstPanelHeaderSettings;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java `"\\s*/\\s*"` (`String.matches` anchors the whole string; Java `\s` is
/// ASCII whitespace).
static SLASH_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?-u:\s)*/(?-u:\s)*$").unwrap());

/// Java private static final `INITIAL_MATCH_LABEL`.
const INITIAL_MATCH_LABEL: &str = "Initial Matching Parameters";

/// Java `final class SolvematchPanel implements Run3dmodButtonContainer,
/// Expandable, ActionListener`.
pub struct SolvematchPanel {
    /// Java private final `pnlRoot = new EtomoPanel()`.
    pnl_root: Rc<EtomoPanel>,
    /// Java private final `pnlFiducialRadio = new JPanel()`.
    pnl_fiducial_radio: Rc<JComponent>,
    /// Java private final `pnlFiducialSelect = new JPanel()`.
    pnl_fiducial_select: Rc<JComponent>,
    /// Java private final `bgFiducialParams = new ButtonGroup()`.
    #[allow(dead_code)]
    bg_fiducial_params: Rc<ButtonGroup>,
    /// Java private final `rbBothSides`.
    rb_both_sides: Rc<RadioButton>,
    /// Java private final `rbOneSide`.
    rb_one_side: Rc<RadioButton>,
    /// Java private final `rbOneSideInverted` (deprecated).
    rb_one_side_inverted: Rc<RadioButton>,
    /// Java private final `rbUseModel` (deprecated).
    rb_use_model: Rc<RadioButton>,
    /// Java private final `rbUseModelOnly`.
    rb_use_model_only: Rc<RadioButton>,
    /// Java private final `pnlImodMatchModels = new JPanel()`.
    pnl_imod_match_models: Rc<JComponent>,
    /// Java private final `cbBinBy2`.
    cb_bin_by2: Rc<CheckBox>,
    /// Java private final `btnImodMatchModels`.
    btn_imod_match_models: Rc<Run3dmodButton>,
    /// Java private final `ltfFiducialMatchListA`.
    ltf_fiducial_match_list_a: Rc<LabeledTextField>,
    /// Java private final `ltfFiducialMatchListB`.
    ltf_fiducial_match_list_b: Rc<LabeledTextField>,
    /// Java private final `ltfUseList`.
    ltf_use_list: Rc<LabeledTextField>,
    /// Java private final `cbUseCorrespondingPoints`.
    cb_use_corresponding_points: Rc<CheckBox>,
    /// Java private final `cbInitialVolumeMatching`.
    cb_initial_volume_matching: Rc<CheckBox>,
    /// Java private final `pnlRootBody = new JPanel()`.
    pnl_root_body: Rc<JComponent>,

    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `headerGroup` (stored, not read).
    #[allow(dead_code)]
    header_group: String,
    /// Java private final `parent` (held weakly: the dialog owns its panels).
    parent: Weak<TomogramCombinationDialog>,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `phInitialMatch`.
    ph_initial_match: Rc<PanelHeader>,
    /// Java private final `parentTitle`.
    parent_title: String,
    /// Java private final `btnRestart`; null on the Setup tab.
    btn_restart: Option<Rc<Run3dmodButton>>,
    /// Java private final `ltfSolvematchMaximumResidual`; null on the Setup tab.
    ltf_solvematch_maximum_residual: Option<Rc<LabeledTextField>>,
    /// Java private final `ltfSolvematchCenterShiftLimit`; null on the Setup tab.
    ltf_solvematch_center_shift_limit: Option<Rc<LabeledTextField>>,
    /// Java private final `ltfDualvolmatchMaximumResidual`; null on the Setup
    /// tab.
    ltf_dualvolmatch_maximum_residual: Option<Rc<LabeledTextField>>,
    /// Java private final `ltfDualvolmatchCenterShiftLimit`; null on the Setup
    /// tab.
    ltf_dualvolmatch_center_shift_limit: Option<Rc<LabeledTextField>>,

    /// Java private `binningWarning = false`.
    binning_warning: Cell<bool>,
    // initial tab only
    /// Java private `useCorrespondingPointsChanged = false`.
    use_corresponding_points_changed: Cell<bool>,
    /// Java private `debug = false`.
    #[allow(dead_code)]
    debug: Cell<bool>,

    /// Java `this` (the button container, expandable and action listener).
    this: Weak<SolvematchPanel>,
}

impl SolvematchPanel {
    /// Java private constructor `SolvematchPanel(TomogramCombinationDialog,
    /// String, ApplicationManager, String, DialogType, boolean,
    /// GlobalExpandButton)`.
    fn new(
        parent: Weak<TomogramCombinationDialog>,
        parent_title: &str,
        manager: &'static ApplicationManager,
        header_group: &str,
        dialog_type: DialogType,
        debug: bool,
        global_advanced_button: Option<&Rc<GlobalExpandButton>>,
    ) -> Rc<SolvematchPanel> {
        let instance = Rc::new_cyclic(|this: &Weak<SolvematchPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let expandable: Weak<dyn Expandable> = this.clone();
            // Field initializers, in declaration order.
            let pnl_root = EtomoPanel::new();
            let pnl_fiducial_radio = JComponent::new_panel();
            let pnl_fiducial_select = JComponent::new_panel();
            let bg_fiducial_params = ButtonGroup::new();
            let rb_both_sides = RadioButton::new_string_button_group(
                Some("Fiducials on both sides"),
                Some(&bg_fiducial_params),
            );
            let rb_one_side = RadioButton::new_string_button_group(
                Some("Fiducials on one side"),
                Some(&bg_fiducial_params),
            );
            let rb_one_side_inverted = RadioButton::new_string_button_group(
                Some("Fiducials on one side, inverted"),
                Some(&bg_fiducial_params),
            );
            let rb_use_model = RadioButton::new_string_button_group(
                Some("Use matching models and fiducials"),
                Some(&bg_fiducial_params),
            );
            let rb_use_model_only = RadioButton::new_string_button_group(
                Some("Use matching models only"),
                Some(&bg_fiducial_params),
            );
            let pnl_imod_match_models = JComponent::new_panel();
            let cb_bin_by2 = CheckBox::new_string(Some("Load binned by 2"));
            let btn_imod_match_models =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Create Matching Models in 3dmod"),
                    Some(container.clone()),
                );
            let ltf_fiducial_match_list_a = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Corresponding fiducial list A: "),
            );
            let ltf_fiducial_match_list_b = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Corresponding fiducial list B: "),
            );
            let ltf_use_list = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Starting points to use from A: "),
            );
            let cb_use_corresponding_points = CheckBox::new_string(Some(
                "Specify corresponding points instead of using coordinate file",
            ));
            let cb_initial_volume_matching = CheckBox::new_string(Some(
                "Use image correlations instead of Solvematch for initial match",
            ));
            let pnl_root_body = JComponent::new_panel();
            // Constructor body.
            let ph_initial_match;
            let btn_restart;
            let ltf_solvematch_maximum_residual;
            let ltf_solvematch_center_shift_limit;
            let ltf_dualvolmatch_maximum_residual;
            let ltf_dualvolmatch_center_shift_limit;
            if parent_title == tomogram_combination_dialog::LBL_INITIAL {
                ph_initial_match =
                    PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                        Some(INITIAL_MATCH_LABEL),
                        Some(expandable.clone()),
                        Some(dialog_type),
                        global_advanced_button.cloned(),
                    );
                // (Run3dmodButton) manager.getProcessResultDisplayFactory(AxisID.ONLY)
                // .getRestartCombine()
                btn_restart = Some(
                    manager
                        .get_process_result_display_factory(AxisID::Only)
                        .get_restart_combine(),
                );
                ltf_solvematch_maximum_residual = Some(LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Limit on maximum residual: "),
                ));
                ltf_solvematch_center_shift_limit = Some(LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Limit on center shift: "),
                ));
                ltf_dualvolmatch_maximum_residual = Some(LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Limit on mean residual in patch correlations: "),
                ));
                ltf_dualvolmatch_center_shift_limit =
                    Some(LabeledTextField::new_field_type_string(
                        FieldType::FloatingPoint,
                        Some("Limit on center shift: "),
                    ));
            } else {
                ph_initial_match = PanelHeader::get_instance(
                    Some(INITIAL_MATCH_LABEL),
                    Some(expandable.clone()),
                    Some(dialog_type),
                );
                btn_restart = None;
                ltf_solvematch_maximum_residual = None;
                ltf_solvematch_center_shift_limit = None;
                ltf_dualvolmatch_maximum_residual = None;
                ltf_dualvolmatch_center_shift_limit = None;
            }
            if let Some(global_advanced_button) = global_advanced_button {
                global_advanced_button.register_expandable(expandable);
            }
            SolvematchPanel {
                pnl_root,
                pnl_fiducial_radio,
                pnl_fiducial_select,
                bg_fiducial_params,
                rb_both_sides,
                rb_one_side,
                rb_one_side_inverted,
                rb_use_model,
                rb_use_model_only,
                pnl_imod_match_models,
                cb_bin_by2,
                btn_imod_match_models,
                ltf_fiducial_match_list_a,
                ltf_fiducial_match_list_b,
                ltf_use_list,
                cb_use_corresponding_points,
                cb_initial_volume_matching,
                pnl_root_body,
                manager,
                header_group: header_group.to_string(),
                parent,
                dialog_type,
                ph_initial_match,
                parent_title: parent_title.to_string(),
                btn_restart,
                ltf_solvematch_maximum_residual,
                ltf_solvematch_center_shift_limit,
                ltf_dualvolmatch_maximum_residual,
                ltf_dualvolmatch_center_shift_limit,
                binning_warning: Cell::new(false),
                use_corresponding_points_changed: Cell::new(false),
                debug: Cell::new(debug),
                this: this.clone(),
            }
        });
        instance
    }

    /// Java package-private static `getInstance(TomogramCombinationDialog,
    /// String, ApplicationManager, String, DialogType, boolean,
    /// GlobalExpandButton)`.
    pub fn get_instance(
        parent: Weak<TomogramCombinationDialog>,
        parent_title: &str,
        manager: &'static ApplicationManager,
        header_group: &str,
        dialog_type: DialogType,
        debug: bool,
        global_advanced_button: Option<&Rc<GlobalExpandButton>>,
    ) -> Rc<SolvematchPanel> {
        let instance = SolvematchPanel::new(
            parent,
            parent_title,
            manager,
            header_group,
            dialog_type,
            debug,
            global_advanced_button,
        );
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.show();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // locals
        let pnl_solve_match = JComponent::new_panel();
        let pnl_fiducial_radio_outer = JComponent::new_panel();
        let pnl_use_corresponding_points = JComponent::new_panel();
        let mut pnl_restart: Option<Rc<JComponent>> = None;
        if self.btn_restart.is_some() {
            pnl_restart = Some(JComponent::new_panel());
        }
        let pnl_initial_volume_matching = JComponent::new_panel();
        let mut pnl_dualvolmatch: Option<Rc<JComponent>> = None;
        if self.ltf_dualvolmatch_maximum_residual.is_some() {
            pnl_dualvolmatch = Some(JComponent::new_panel());
        }
        // init
        // Swing layout: rbBothSides, rbOneSide, rbOneSideInverted and rbUseModel
        // .setAlignmentX(Component.LEFT_ALIGNMENT);
        // pnlFiducialRadioOuter.setAlignmentX(Component.CENTER_ALIGNMENT).
        if let Some(btn_restart) = &self.btn_restart {
            let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
            btn_restart.set_container(Some(container));
        }
        self.cb_initial_volume_matching.set_selected_boolean(
            CombineTool::get_initial_volume_matching_init_value(self.manager),
        );
        self.rb_one_side_inverted.set_visible(false);
        self.rb_use_model.set_visible(false);
        // Root
        // Swing layout: pnlRoot.setBorder(BorderFactory.createEtchedBorder());
        // pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.Y_AXIS)).
        self.pnl_root.add(&self.ph_initial_match);
        self.pnl_root.get_component().add(&self.pnl_root_body);
        // RootBody
        // Swing layout: pnlRootBody.setLayout(new BoxLayout(pnlRootBody,
        // BoxLayout.Y_AXIS)); rigid areas (FixedDim.x0_y5) before
        // pnlInitialVolumeMatching and before pnlSolveMatch.
        self.pnl_root_body.add(&pnl_initial_volume_matching);
        self.pnl_root_body.add(&pnl_solve_match);
        if let Some(pnl_dualvolmatch) = &pnl_dualvolmatch {
            self.pnl_root_body.add(pnl_dualvolmatch);
        }
        if let Some(pnl_restart) = &pnl_restart {
            ui_utilities::add_with_y_space(&self.pnl_root_body, pnl_restart);
        }
        // InitialVolumeMatching
        // Swing layout: pnlInitialVolumeMatching.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); horizontal glue after the check box.
        pnl_initial_volume_matching.add(&self.cb_initial_volume_matching.get_component());
        // SolveMatch
        pnl_solve_match.set_border_title(
            EtchedBorder::new(Some("Solvematch Parameters"))
                .get_title()
                .as_deref(),
        );
        // Swing layout: pnlSolveMatch.setLayout(new BoxLayout(pnlSolveMatch,
        // BoxLayout.Y_AXIS)).
        ui_utilities::add_with_space(
            &pnl_solve_match,
            &self.pnl_fiducial_select,
            fixed_dim::x0_y10,
        );
        ui_utilities::add_with_y_space(&pnl_solve_match, &pnl_use_corresponding_points);
        ui_utilities::add_with_y_space(&pnl_solve_match, &self.ltf_use_list.get_container());
        ui_utilities::add_with_y_space(
            &pnl_solve_match,
            &self.ltf_fiducial_match_list_a.get_container(),
        );
        ui_utilities::add_with_y_space(
            &pnl_solve_match,
            &self.ltf_fiducial_match_list_b.get_container(),
        );
        if let Some(ltf) = &self.ltf_solvematch_maximum_residual {
            ui_utilities::add_with_y_space(&pnl_solve_match, &ltf.get_container());
        }
        if let Some(ltf) = &self.ltf_solvematch_center_shift_limit {
            ui_utilities::add_with_y_space(&pnl_solve_match, &ltf.get_container());
        }
        // FiducialSelect
        // Swing layout: pnlFiducialSelect.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); horizontal glue at the end.
        ui_utilities::add_with_space(
            &self.pnl_fiducial_select,
            &pnl_fiducial_radio_outer,
            fixed_dim::x20_y0,
        );
        self.pnl_fiducial_select.add(&self.pnl_imod_match_models);
        // FiducialRadioOuter
        // Swing layout: pnlFiducialRadioOuter.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); horizontal glue after pnlFiducialRadio.
        pnl_fiducial_radio_outer.add(&self.pnl_fiducial_radio);
        // FiducialRadio
        // Swing layout: pnlFiducialRadio.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)).
        self.pnl_fiducial_radio
            .add(&self.rb_both_sides.get_component());
        self.pnl_fiducial_radio
            .add(&self.rb_one_side.get_component());
        self.pnl_fiducial_radio
            .add(&self.rb_one_side_inverted.get_component());
        self.pnl_fiducial_radio
            .add(&self.rb_use_model.get_component());
        self.pnl_fiducial_radio
            .add(&self.rb_use_model_only.get_component());
        // ImodMatchModels
        // Swing layout: pnlImodMatchModels.setLayout(new BoxLayout(...,
        // BoxLayout.Y_AXIS)).
        self.pnl_imod_match_models
            .add(&self.cb_bin_by2.get_component());
        self.pnl_imod_match_models
            .add(&self.btn_imod_match_models.get_component());
        // UseCorrespondingPoints
        // Swing layout: pnlUseCorrespondingPoints.setLayout(new BoxLayout(...,
        // BoxLayout.X_AXIS)); setAlignmentX(Component.CENTER_ALIGNMENT);
        // horizontal glue after the check box.
        pnl_use_corresponding_points.add(&self.cb_use_corresponding_points.get_component());
        // Dualvolmatch
        if let Some(pnl_dualvolmatch) = &pnl_dualvolmatch {
            pnl_dualvolmatch.set_border_title(
                EtchedBorder::new(Some("Dualvolmatch Parameters"))
                    .get_title()
                    .as_deref(),
            );
            // Swing layout: pnlDualvolmatch.setLayout(new BoxLayout(...,
            // BoxLayout.Y_AXIS)).
            if let Some(ltf) = &self.ltf_dualvolmatch_maximum_residual {
                pnl_dualvolmatch.add(&ltf.get_component());
            }
            if let Some(ltf) = &self.ltf_dualvolmatch_center_shift_limit {
                pnl_dualvolmatch.add(&ltf.get_component());
            }
        }
        // Restart
        if let Some(pnl_restart) = &pnl_restart {
            // Swing layout: pnlRestart.setLayout(new BoxLayout(pnlRestart,
            // BoxLayout.X_AXIS)); setAlignmentX(Component.CENTER_ALIGNMENT);
            // horizontal glue around the button.
            if let Some(btn_restart) = &self.btn_restart {
                pnl_restart.add(&btn_restart.get_component());
            }
        }
        // adjust
        // Swing layout: UIUtilities.setButtonSizeAll(pnlImodMatchModels,
        // UIParameters.getInstance().getButtonDimension()).
        // update
        self.update_display();
        self.update_advanced(self.ph_initial_match.is_advanced());
    }

    /// Java package-private `show()`.
    pub fn show(&self) {
        if self.manager.coord_file_exists() {
            self.cb_use_corresponding_points.set_selected_boolean(false);
            self.cb_use_corresponding_points.set_visible(true);
            self.ltf_use_list.set_visible(true);
        } else {
            self.cb_use_corresponding_points.set_selected_boolean(true);
            self.cb_use_corresponding_points.set_visible(false);
            self.ltf_use_list.set_visible(false);
        }
        self.update_display();
    }

    /// Java package-private `setDeferred3dmodButtons()`.
    pub fn set_deferred_3dmod_buttons(&self) {
        if let Some(btn_restart) = &self.btn_restart {
            let Some(parent) = self.parent.upgrade() else {
                return;
            };
            let deferred: Rc<dyn Deferred3dmodButton> = parent.get_imod_combined_button();
            btn_restart.set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        }
    }

    /// Java private `addListeners()`: the panel itself is the action listener.
    fn add_listeners(&self) {
        let adaptee = self.this.clone();
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action_performed(event);
            }
        });
        // Bind the ui elements to their listeners
        if let Some(btn_restart) = &self.btn_restart {
            btn_restart.add_action_listener(listener.clone());
        }
        self.btn_imod_match_models
            .add_action_listener(listener.clone());
        self.cb_bin_by2.add_action_listener(Some(listener.clone()));
        self.cb_use_corresponding_points
            .add_action_listener(Some(listener.clone()));
        self.rb_both_sides.add_action_listener(listener.clone());
        self.rb_one_side.add_action_listener(listener.clone());
        self.rb_one_side_inverted
            .add_action_listener(listener.clone());
        self.rb_use_model.add_action_listener(listener.clone());
        self.rb_use_model_only.add_action_listener(listener.clone());
        self.cb_initial_volume_matching
            .add_action_listener(Some(listener));
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_root.get_component()
    }

    // FIXME there are current two ways to get the parameters into and out of the
    // panel. Does this need to be the case? It seem redundant.
    /// Java package-private `setParameters(ConstCombineParams, boolean)`.
    pub fn set_parameters_const_combine_params_boolean(
        &self,
        combine_params: &dyn ConstCombineParams,
        init: bool,
    ) {
        let match_ = combine_params.get_fiducial_match();
        if match_ == Some(FiducialMatch::BothSides) {
            self.rb_both_sides.set_selected_boolean(true);
        } else if match_ == Some(FiducialMatch::OneSide) {
            self.rb_one_side.set_selected_boolean(true);
        }
        // backwards compatibility
        else if match_ == Some(FiducialMatch::OneSideInverted) {
            self.rb_one_side_inverted.set_selected_boolean(true);
            self.rb_one_side_inverted.set_visible(true);
        } else if match_ == Some(FiducialMatch::UseModel) {
            self.rb_use_model.set_selected_boolean(true);
            self.rb_use_model.set_visible(true);
        } else if match_ == Some(FiducialMatch::UseModelOnly) {
            self.rb_use_model_only.set_selected_boolean(true);
        }
        self.ltf_fiducial_match_list_a
            .set_text_string(Some(&combine_params.get_fiducial_match_list_a()));
        self.ltf_fiducial_match_list_b
            .set_text_string(Some(&combine_params.get_fiducial_match_list_b()));
        self.ltf_use_list
            .set_text_string(Some(&combine_params.get_use_list()));
        if self.cb_use_corresponding_points.is_visible() {
            self.cb_use_corresponding_points
                .set_selected_boolean(!combine_params.is_transfer());
            self.update_display();
        }
        let initial_volume_matching = combine_params.is_initial_volume_matching();
        if !init || initial_volume_matching {
            self.cb_initial_volume_matching
                .set_selected_boolean(initial_volume_matching);
        }
    }

    /// Java package-private `setParameters(DualvolmatchParam)`.
    pub fn set_parameters_dualvolmatch_param(&self, param: &DualvolmatchParam) {
        if let Some(ltf) = &self.ltf_dualvolmatch_maximum_residual {
            ltf.set_text_string(Some(&param.get_maximum_residual()));
        }
        if let Some(ltf) = &self.ltf_dualvolmatch_center_shift_limit {
            ltf.set_text_string(Some(&param.get_center_shift_limit()));
        }
    }

    /// Java package-private `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        if self.btn_restart.is_some() {
            // initial tab
            self.ph_initial_match.get_state(Some(
                screen_state.get_combine_initial_solvematch_header_state(),
            ));
        } else {
            // setup tab
            self.ph_initial_match.get_state(Some(
                screen_state.get_combine_setup_solvematch_header_state(),
            ));
        }
    }

    /// Java package-private final `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        if let Some(btn_restart) = &self.btn_restart {
            // initial tab
            self.ph_initial_match.set_state(Some(
                screen_state.get_combine_initial_solvematch_header_state(),
            ));
            let dialog_type = self.parent.upgrade().map(|parent| parent.dialog_type);
            btn_restart.set_button_state(
                screen_state
                    .get_button_state(btn_restart.create_button_state_key(dialog_type).as_deref()),
            );
            btn_restart.set_button_state(
                screen_state.get_button_state(btn_restart.get_button_state_key().as_deref()),
            );
        } else {
            // setup tab
            self.ph_initial_match.set_state(Some(
                screen_state.get_combine_setup_solvematch_header_state(),
            ));
        }
    }

    /// Java package-private `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_root.get_component().set_visible(visible);
    }

    /// Java package-private `getParameters(CombineParams, boolean)`.  Get the
    /// parameters from the ui and filling in the appropriate fields in the
    /// CombineParams object.
    pub fn get_parameters_combine_params_boolean(
        &self,
        combine_params: &mut CombineParams,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let result: Result<(), FieldValidationFailedException> = (|| {
            if self.rb_both_sides.is_selected() && self.rb_both_sides.is_enabled() {
                combine_params.set_fiducial_match(Some(FiducialMatch::BothSides));
            }
            if self.rb_one_side.is_selected() && self.rb_one_side.is_enabled() {
                combine_params.set_fiducial_match(Some(FiducialMatch::OneSide));
            }
            if self.rb_one_side_inverted.is_selected()
                && self.rb_one_side_inverted.is_enabled()
                && self.rb_one_side_inverted.is_visible()
            {
                combine_params.set_fiducial_match(Some(FiducialMatch::OneSideInverted));
            }
            if self.rb_use_model.is_selected()
                && self.rb_use_model.is_enabled()
                && self.rb_use_model.is_visible()
            {
                combine_params.set_fiducial_match(Some(FiducialMatch::UseModel));
            }
            if self.rb_use_model_only.is_selected() && self.rb_use_model_only.is_enabled() {
                combine_params.set_fiducial_match(Some(FiducialMatch::UseModelOnly));
            }
            combine_params.set_transfer(!self.cb_use_corresponding_points.is_selected());
            combine_params.set_fiducial_match_list_a(
                self.ltf_fiducial_match_list_a
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            combine_params.set_fiducial_match_list_b(
                self.ltf_fiducial_match_list_b
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            // ltfUseList.getText().matches("\\s*/\\s*")
            let use_list_text = self.ltf_use_list.get_text_void().unwrap_or_default();
            if SLASH_PATTERN.is_match(&use_list_text) {
                combine_params.set_use_list(Some(""));
            } else {
                combine_params.set_use_list(
                    self.ltf_use_list
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            }
            combine_params
                .set_initial_volume_matching(self.cb_initial_volume_matching.is_selected());
            Ok(())
        })();
        result.is_ok()
    }

    /// Java package-private `setParameters(ConstSolvematchParam)`.
    pub fn set_parameters_const_solvematch_param(&self, solvematch_param: &ConstSolvematchParam) {
        self.set_surfaces_or_models(solvematch_param.get_surfaces_or_model());
        if solvematch_param.is_match_b_to_a() {
            self.ltf_fiducial_match_list_a.set_text_string(Some(
                &solvematch_param.get_to_correspondence_list().to_string(),
            ));
            self.ltf_fiducial_match_list_b.set_text_string(Some(
                &solvematch_param.get_from_correspondence_list().to_string(),
            ));
        } else {
            self.ltf_fiducial_match_list_b.set_text_string(Some(
                &solvematch_param.get_to_correspondence_list().to_string(),
            ));
            self.ltf_fiducial_match_list_a.set_text_string(Some(
                &solvematch_param.get_from_correspondence_list().to_string(),
            ));
        }
        if let Some(ltf) = &self.ltf_solvematch_maximum_residual {
            ltf.set_text_double(solvematch_param.get_maximum_residual());
        }
        if let Some(ltf) = &self.ltf_solvematch_center_shift_limit {
            ltf.set_text_const_etomo_number(Some(solvematch_param.get_center_shift_limit()));
        }
        self.ltf_use_list
            .set_text_string(Some(&solvematch_param.get_use_points().to_string()));
    }

    /// Java package-private `getParameters(DualvolmatchParam, boolean)`.
    pub fn get_parameters_dualvolmatch_param_boolean(
        &self,
        param: &mut DualvolmatchParam,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let result: Result<(), FieldValidationFailedException> = (|| {
            if let Some(ltf) = &self.ltf_dualvolmatch_maximum_residual {
                param.set_maximum_residual(ltf.get_text_boolean(do_validation)?.as_deref());
            }
            if let Some(ltf) = &self.ltf_dualvolmatch_center_shift_limit {
                param.set_center_shift_limit(ltf.get_text_boolean(do_validation)?.as_deref());
            }
            Ok(())
        })();
        result.is_ok()
    }

    /// Java package-private `getParameters(SolvematchParam, boolean)`.  Get the
    /// parameters from the ui and filling in the appropriate fields in the
    /// SolvematchParam object.
    ///
    /// `Err` carries the message of the unchecked NumberFormatException that
    /// `SolvematchParam.setMaximumResidual(String)` throws on an unparsable
    /// value; Java lets it propagate to `ApplicationManager` (the declared
    /// `throws NumberFormatException` of
    /// `TomogramCombinationDialog.getSolvematchParams`).
    pub fn get_parameters_solvematch_param_boolean(
        &self,
        solvematch_param: &mut SolvematchParam,
        do_validation: bool,
    ) -> Result<bool, String> {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let result: Result<Result<(), String>, FieldValidationFailedException> = (|| {
            solvematch_param.set_surfaces_or_model(self.get_surfaces_or_models());
            if solvematch_param.is_match_b_to_a() {
                solvematch_param.set_to_correspondence_list(
                    self.ltf_fiducial_match_list_a
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
                solvematch_param.set_from_correspondence_list(
                    self.ltf_fiducial_match_list_b
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                solvematch_param.set_from_correspondence_list(
                    self.ltf_fiducial_match_list_a
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
                solvematch_param.set_to_correspondence_list(
                    self.ltf_fiducial_match_list_b
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            }
            if let Some(ltf) = &self.ltf_solvematch_maximum_residual {
                let text = ltf.get_text_boolean(do_validation)?;
                // NumberFormatException propagates out of the method.
                if let Err(message) = solvematch_param.set_maximum_residual_string(text.as_deref())
                {
                    return Ok(Err(message));
                }
            }
            if let Some(ltf) = &self.ltf_solvematch_center_shift_limit {
                solvematch_param
                    .set_center_shift_limit(ltf.get_text_boolean(do_validation)?.as_deref());
            }

            solvematch_param
                .set_transfer_coordinate_file(self.cb_use_corresponding_points.is_selected());
            solvematch_param.set_use_points(
                self.ltf_use_list
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            Ok(Ok(()))
        })();
        match result {
            Ok(Ok(())) => Ok(true),
            Ok(Err(message)) => Err(message),
            Err(_) => Ok(false),
        }
    }

    /// Java package-private `getSurfacesOrModels()`.
    pub fn get_surfaces_or_models(&self) -> FiducialMatch {
        if self.rb_both_sides.is_selected() && self.rb_both_sides.is_enabled() {
            return FiducialMatch::BothSides;
        }
        if self.rb_one_side.is_selected() && self.rb_one_side.is_enabled() {
            return FiducialMatch::OneSide;
        }
        if self.rb_one_side_inverted.is_selected()
            && self.rb_one_side_inverted.is_enabled()
            && self.rb_one_side_inverted.is_visible()
        {
            return FiducialMatch::OneSideInverted;
        }
        if self.rb_use_model.is_selected()
            && self.rb_use_model.is_enabled()
            && self.rb_use_model.is_visible()
        {
            return FiducialMatch::UseModel;
        }
        if self.rb_use_model_only.is_selected() && self.rb_use_model_only.is_enabled() {
            return FiducialMatch::UseModelOnly;
        }
        FiducialMatch::NotSet
    }

    /// Java package-private `setSurfacesOrModels(FiducialMatch)`.
    pub fn set_surfaces_or_models(&self, value: FiducialMatch) {
        if value == FiducialMatch::UseModelOnly {
            self.rb_use_model_only.set_selected_boolean(true);
        }
        // backwards compatibility
        if value == FiducialMatch::OneSideInverted {
            self.rb_one_side_inverted.set_selected_boolean(true);
            self.rb_one_side_inverted.set_visible(true);
        }
        if value == FiducialMatch::UseModel {
            self.rb_use_model.set_selected_boolean(true);
            self.rb_use_model.set_visible(true);
        }
        if value == FiducialMatch::OneSide {
            self.rb_one_side.set_selected_boolean(true);
        }
        if value == FiducialMatch::BothSides {
            self.rb_both_sides.set_selected_boolean(true);
        }
        self.update_display();
    }

    /// Java package-private `isBinBy2()`.
    pub fn is_bin_by2(&self) -> bool {
        self.cb_bin_by2.is_selected()
    }

    /// Java package-private `setBinBy2(boolean)`.
    pub fn set_bin_by2(&self, state: bool) {
        self.cb_bin_by2.set_selected_boolean(state);
    }

    /// Java package-private `setUseList(String)`.
    pub fn set_use_list(&self, use_list: Option<&str>) {
        self.ltf_use_list.set_text_string(use_list);
    }

    /// Java package-private `setFiducialMatchListA(String)`.
    pub fn set_fiducial_match_list_a(&self, fiducial_match_list_a: Option<&str>) {
        self.ltf_fiducial_match_list_a
            .set_text_string(fiducial_match_list_a);
    }

    /// Java package-private `getUseList(boolean) throws
    /// FieldValidationFailedException`.
    pub fn get_use_list_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_use_list.get_text_boolean(do_validation)
    }

    /// Java package-private `getUseList()`.
    pub fn get_use_list_void(&self) -> Option<String> {
        self.ltf_use_list.get_text_void()
    }

    /// Java package-private `getFiducialMatchListA(boolean) throws
    /// FieldValidationFailedException`.
    pub fn get_fiducial_match_list_a_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_fiducial_match_list_a
            .get_text_boolean(do_validation)
    }

    /// Java package-private `getFiducialMatchListA()`.
    pub fn get_fiducial_match_list_a_void(&self) -> Option<String> {
        self.ltf_fiducial_match_list_a.get_text_void()
    }

    /// Java package-private `setFiducialMatchListB(String)`.
    pub fn set_fiducial_match_list_b(&self, fiducial_match_list_b: Option<&str>) {
        self.ltf_fiducial_match_list_b
            .set_text_string(fiducial_match_list_b);
    }

    /// Java package-private `getFiducialMatchListB(boolean) throws
    /// FieldValidationFailedException`.
    pub fn get_fiducial_match_list_b_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_fiducial_match_list_b
            .get_text_boolean(do_validation)
    }

    /// Java package-private `getFiducialMatchListB()`.
    pub fn get_fiducial_match_list_b_void(&self) -> Option<String> {
        self.ltf_fiducial_match_list_b.get_text_void()
    }

    /// Java public override `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: &ActionEvent) {
        // Java `action(event.getActionCommand(), null, null)`; a button's action
        // command is never null (it defaults to the button text).
        self.action(event.get_action_command().unwrap_or(""), None, None);
    }

    /// Java private `rbFiducialAction(ActionEvent)`.  Manage fiducial radio
    /// button action.  (Unused in the Java.)
    #[allow(dead_code)]
    fn rb_fiducial_action(&self, _event: &ActionEvent) {
        self.update_display();
    }

    /// Java private `updateAdvanced(boolean)`.
    fn update_advanced(&self, advanced: bool) {
        if let Some(ltf) = &self.ltf_solvematch_center_shift_limit {
            ltf.set_visible(advanced);
        }
        if let Some(ltf) = &self.ltf_dualvolmatch_center_shift_limit {
            ltf.set_visible(advanced);
        }
    }

    /// Java package-private `updateDisplay()`.
    pub fn update_display(&self) {
        let initial_volume_matching = self.cb_initial_volume_matching.is_selected();
        if let Some(ltf) = &self.ltf_solvematch_center_shift_limit {
            ltf.set_enabled(!initial_volume_matching);
        }
        if let Some(ltf) = &self.ltf_solvematch_maximum_residual {
            ltf.set_enabled(!initial_volume_matching);
        }
        self.rb_both_sides.set_enabled(!initial_volume_matching);
        self.rb_one_side.set_enabled(!initial_volume_matching);
        self.rb_use_model_only.set_enabled(!initial_volume_matching);
        let fiducial_mode = self.rb_use_model.is_selected() || self.rb_use_model_only.is_selected();
        self.btn_imod_match_models
            .set_enabled(fiducial_mode && !initial_volume_matching);
        self.cb_bin_by2
            .set_enabled(fiducial_mode && !initial_volume_matching);
        self.cb_use_corresponding_points
            .set_enabled(!initial_volume_matching);
        self.ltf_fiducial_match_list_a
            .set_enabled(!initial_volume_matching);
        self.ltf_fiducial_match_list_b
            .set_enabled(!initial_volume_matching);
        self.ltf_use_list.set_enabled(!initial_volume_matching);
        if let Some(ltf) = &self.ltf_dualvolmatch_maximum_residual {
            ltf.set_enabled(initial_volume_matching);
        }
        if let Some(ltf) = &self.ltf_dualvolmatch_center_shift_limit {
            ltf.set_enabled(initial_volume_matching);
        }
        if self.cb_use_corresponding_points.is_selected() {
            self.ltf_fiducial_match_list_a.set_visible(true);
            self.ltf_fiducial_match_list_b.set_visible(true);
            self.ltf_use_list.set_visible(false);
        } else {
            self.ltf_fiducial_match_list_a.set_visible(false);
            self.ltf_fiducial_match_list_b.set_visible(false);
            self.ltf_use_list.set_visible(true);
        }
        if self.use_corresponding_points_changed.get() {
            self.use_corresponding_points_changed.set(false);
            if let Some(parent) = self.parent.upgrade() {
                parent.update_display();
            }
        }
    }

    /// Java package-private `isUseCorrespondingPoints()`.
    pub fn is_use_corresponding_points(&self) -> bool {
        self.cb_use_corresponding_points.is_selected()
    }

    /// Java package-private `isInitialVolumeMatching()`.
    pub fn is_initial_volume_matching(&self) -> bool {
        self.cb_initial_volume_matching.is_selected()
    }

    /// Java package-private `setInitialVolumeMatching(boolean)`.
    pub fn set_initial_volume_matching(&self, input: bool) {
        self.cb_initial_volume_matching.set_selected_boolean(input);
    }

    /// Java package-private `setUseCorrespondingPoints(boolean)`.
    pub fn set_use_corresponding_points(&self, selected: bool) {
        self.cb_use_corresponding_points
            .set_selected_boolean(selected);
        self.update_display();
    }

    /// Java private `setToolTipText()`.  Initialize the tooltip text.
    fn set_tool_tip_text(&self) {
        let solvematch_section: *mut Section;
        let mut solvematch_autodoc: Option<*mut Autodoc> = None;
        let mut dualvolmatch_autodoc: Option<*mut Autodoc> = None;
        let manager: &'static dyn BaseManager = self.manager;
        // SAFETY (all `unsafe` below): `AutodocFactory` owns every autodoc it
        // returns, and each autodoc owns its sections, for the life of the
        // process (the Java GC-owned singletons), so the pointers stay valid
        // for this method.
        //
        // Java's one try block: an exception from the first getInstance skips
        // the second.
        let result = (|| -> Result<(), LogFileError> {
            solvematch_autodoc = Some(unsafe {
                autodoc_factory::get_instance(
                    Some(manager),
                    Some(autodoc_factory::SOLVEMATCH),
                    AxisID::Only,
                    false,
                )
            }?);
            dualvolmatch_autodoc = Some(unsafe {
                autodoc_factory::get_instance(
                    Some(manager),
                    Some(autodoc_factory::DUALVOLMATCH),
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
        let solvematch_autodoc_ref: Option<&dyn ReadOnlyAutodoc> =
            solvematch_autodoc.map(|autodoc| unsafe { &*autodoc } as &dyn ReadOnlyAutodoc);
        let dualvolmatch_autodoc_ref: Option<&dyn ReadOnlyAutodoc> =
            dualvolmatch_autodoc.map(|autodoc| unsafe { &*autodoc } as &dyn ReadOnlyAutodoc);
        // Upstream bug fixed in translation (SolvematchPanel.java:743-745): Java
        // calls `solvematchAutodoc.getAutodocName()` and `getSection(...)` without
        // a null check, so a missing or locked solvematch autodoc throws a
        // NullPointerException out of the constructor and the Tomogram
        // Combination dialog cannot be built.  We treat a null autodoc as having
        // no name and no sections: every autodoc tooltip is then null (as
        // `EtomoAutodoc.getTooltip` returns for a null autodoc or section) and the
        // fixed tooltips are still set.
        let autodoc_name: Option<String> =
            solvematch_autodoc_ref.map(|autodoc| autodoc.get_autodoc_name());
        let autodoc_name = autodoc_name.as_deref();
        solvematch_section = match solvematch_autodoc {
            Some(autodoc) => unsafe {
                (*autodoc).get_section(
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(const_solvematch_param::SURFACE_OR_USE_MODELS),
                )
            },
            None => std::ptr::null_mut(),
        };
        self.cb_use_corresponding_points
            .set_tool_tip_text_string(Some(
                "Check to use the points in A and B in the transferfid log file.  \
                 Leave unchecked to use transferfid.coord.",
            ));
        if !solvematch_section.is_null() {
            let solvematch_section: &dyn ReadOnlySection = unsafe { &*solvematch_section };
            self.rb_both_sides.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    solvematch_section,
                    const_solvematch_param::BOTH_SIDES_OPTION,
                )
                .as_deref(),
            );
            self.rb_one_side_inverted.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    solvematch_section,
                    const_solvematch_param::ONE_SIDE_INVERTED_OPTION,
                )
                .as_deref(),
            );
            self.rb_one_side.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    solvematch_section,
                    const_solvematch_param::ONE_SIDE_OPTION,
                )
                .as_deref(),
            );
            self.rb_use_model.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    solvematch_section,
                    const_solvematch_param::USE_MODEL_OPTION,
                )
                .as_deref(),
            );
            self.rb_use_model_only.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    solvematch_section,
                    const_solvematch_param::USE_MODEL_ONLY_OPTION,
                )
                .as_deref(),
            );
            if let Some(btn_restart) = &self.btn_restart {
                btn_restart.set_tool_tip_text(Some(
                    "Restart the combine operation from the beginning with the parameters \
                     specified here.",
                ));
            }
        }
        self.cb_bin_by2.set_tool_tip_text_string(Some(
            "Use binning by 2 when opening matching models to allow the two 3dmods \
             to fit into the computer's memory.",
        ));
        self.btn_imod_match_models
            .set_tool_tip_text(Some("Create models of corresponding points."));
        self.ltf_fiducial_match_list_a.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                solvematch_autodoc_ref,
                Some(const_solvematch_param::TO_CORRESPONDENCE_LIST),
            )
            .as_deref(),
        );
        self.ltf_fiducial_match_list_b.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                solvematch_autodoc_ref,
                Some(const_solvematch_param::FROM_CORRESPONDENCE_LIST),
            )
            .as_deref(),
        );
        self.ltf_use_list.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                solvematch_autodoc_ref,
                Some(const_solvematch_param::USE_POINTS),
            )
            .as_deref(),
        );
        self.cb_initial_volume_matching
            .set_tool_tip_text_string(Some(
                "Use Dualvolmatch when there are no or too few matching fiducials to use \
                 Solvematch.",
            ));
        if let Some(ltf) = &self.ltf_solvematch_maximum_residual {
            ltf.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    solvematch_autodoc_ref,
                    Some(const_solvematch_param::MAXIMUM_RESIDUAL),
                )
                .as_deref(),
            );
        }
        if let Some(ltf) = &self.ltf_solvematch_center_shift_limit {
            ltf.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    solvematch_autodoc_ref,
                    Some(const_solvematch_param::CENTER_SHIFT_LIMIT_KEY),
                )
                .as_deref(),
            );
        }
        if let Some(ltf) = &self.ltf_dualvolmatch_maximum_residual {
            ltf.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    dualvolmatch_autodoc_ref,
                    Some(dualvolmatch_param::MAXIMUM_RESIDUAL),
                )
                .as_deref(),
            );
        }
        if let Some(ltf) = &self.ltf_dualvolmatch_center_shift_limit {
            ltf.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    dualvolmatch_autodoc_ref,
                    Some(dualvolmatch_param::CENTER_SHIFT_LIMIT),
                )
                .as_deref(),
            );
        }
    }
}

impl Expandable for SolvematchPanel {
    /// Java public override `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.ph_initial_match.equals_open_close(button) {
            self.pnl_root_body.set_visible(button.is_expanded());
        } else if self.ph_initial_match.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
    }

    /// Java public override `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.update_advanced(button.is_expanded());
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(AxisID::Only), Some(manager))
        });
    }
}

impl Run3dmodButtonContainer for SolvematchPanel {
    /// Java public override `action(String, Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command)
            == self
                .cb_use_corresponding_points
                .get_action_command()
                .as_deref()
        {
            self.use_corresponding_points_changed.set(true);
            self.update_display();
        } else if Some(command) == self.rb_both_sides.get_action_command().as_deref()
            || Some(command) == self.rb_one_side.get_action_command().as_deref()
            || Some(command) == self.rb_use_model_only.get_action_command().as_deref()
            || Some(command)
                == self
                    .cb_initial_volume_matching
                    .get_action_command()
                    .as_deref()
        {
            self.update_display();
        } else {
            // The dialog owns this panel, so it is alive while the panel is.
            let Some(parent) = self.parent.upgrade() else {
                return;
            };
            // Synchronize this panel with the others
            parent.synchronize(&self.parent_title, true);
            if Some(command) == self.cb_bin_by2.get_action_command().as_deref() {
                if !self.binning_warning.get() && self.cb_bin_by2.is_selected() {
                    parent.set_binning_warning(true);
                    self.binning_warning.set(true);
                }
            } else if let Some(btn_restart) = self
                .btn_restart
                .as_ref()
                .filter(|btn_restart| Some(command) == btn_restart.get_action_command().as_deref())
            {
                let display: ProcessResultDisplayHandle = btn_restart.clone();
                // A null Run3dmodMenuOptions becomes `new Run3dmodMenuOptions()`
                // where the 3dmod state opens (ImodState.open).
                self.manager.combine(
                    Some(display),
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options.unwrap_or_default(),
                    self.dialog_type,
                    parent.get_run_processing_method(),
                    self.cb_initial_volume_matching.is_selected(),
                    parent.is_parallel(),
                    !parent.is_run_volcombine(),
                );
            } else if Some(command) == self.btn_imod_match_models.get_action_command().as_deref() {
                self.manager.imod_matching_model(
                    self.cb_bin_by2.is_selected(),
                    run_3dmod_menu_options.unwrap_or_default(),
                );
            }
        }
    }
}
