//! `IMOD/Etomo/src/etomo/ui/swing/SolvematchPanel.java`.
//!
//! Swing layout, autodoc I/O, `TomogramCombinationDialog`, and the concrete
//! `ApplicationManager` remain explicit boundaries.  The panel's selected
//! matching mode, fields, enablement/visibility transitions, and dispatch
//! arguments are retained as source-owned Rust state.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::dialog_type::DialogType;
pub use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::ui::field_type::FieldType;

use super::check_box::CheckBox;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::multi_line_button::MultiLineButton;
use super::panel_header::{ExpandButton, PanelHeader};
use super::radio_button::{RadioButton, RadioButtonGroup};

pub const INITIAL_MATCH_LABEL: &str = "Initial Matching Parameters";

/// `CombineParams` calls made by `SolvematchPanel.java`.
pub trait CombineParameters {
    fn fiducial_match(&self) -> FiducialMatch;
    fn set_fiducial_match(&mut self, value: FiducialMatch);
    fn fiducial_match_list_a(&self) -> String;
    fn set_fiducial_match_list_a(&mut self, value: String);
    fn fiducial_match_list_b(&self) -> String;
    fn set_fiducial_match_list_b(&mut self, value: String);
    fn use_list(&self) -> String;
    fn set_use_list(&mut self, value: String);
    fn transfer(&self) -> bool;
    fn set_transfer(&mut self, value: bool);
    fn initial_volume_matching(&self) -> bool;
    fn set_initial_volume_matching(&mut self, value: bool);
}

/// `SolvematchParam` calls made by this source unit.
pub trait SolvematchParameters {
    fn surfaces_or_model(&self) -> FiducialMatch;
    fn set_surfaces_or_model(&mut self, value: FiducialMatch);
    fn match_b_to_a(&self) -> bool;
    fn to_correspondence_list(&self) -> String;
    fn set_to_correspondence_list(&mut self, value: String);
    fn from_correspondence_list(&self) -> String;
    fn set_from_correspondence_list(&mut self, value: String);
    fn maximum_residual(&self) -> String;
    fn set_maximum_residual(&mut self, value: String);
    fn center_shift_limit(&self) -> String;
    fn set_center_shift_limit(&mut self, value: String);
    fn use_points(&self) -> String;
    fn set_use_points(&mut self, value: String);
    fn set_transfer_coordinate_file(&mut self, value: bool);
}

/// `DualvolmatchParam` calls made by this source unit.
pub trait DualvolmatchParameters {
    fn maximum_residual(&self) -> String;
    fn set_maximum_residual(&mut self, value: String);
    fn center_shift_limit(&self) -> String;
    fn set_center_shift_limit(&mut self, value: String);
}

/// The two `ReconScreenState` header slots addressed by this Java unit.
pub trait SolvematchPanelScreenState {
    fn combine_initial_solvematch_header_state(&self) -> Option<&str>;
    fn combine_setup_solvematch_header_state(&self) -> Option<&str>;
    fn set_combine_initial_solvematch_header_state(&mut self, state: Option<String>);
    fn set_combine_setup_solvematch_header_state(&mut self, state: Option<String>);
}

/// Direct `TomogramCombinationDialog` calls made by this panel.
pub trait SolvematchPanelParent {
    fn synchronize(&mut self, title: &str, include_this: bool);
    fn set_binning_warning(&mut self, value: bool);
    fn update_display(&mut self);
    fn run_processing_method(&self) -> String;
    fn is_parallel(&self) -> bool;
    fn is_run_volcombine(&self) -> bool;
}

/// Direct `ApplicationManager` calls made by this panel.
pub trait SolvematchPanelApplicationManager {
    fn combine(
        &mut self,
        button: &MultiLineButton,
        deferred_3dmod_button: Option<&MultiLineButton>,
        options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        run_processing_method: String,
        initial_volume_matching: bool,
        parallel: bool,
        not_run_volcombine: bool,
    );
    fn imod_matching_model(&mut self, bin_by_2: bool, options: Option<Run3dmodMenuOptions>);
}

/// Source-visible component hierarchy/renderer boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SolvematchPanelLayout {
    pub root_visible: bool,
    pub root_body_visible: bool,
    pub restart_present: bool,
    pub dualvolmatch_present: bool,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub body_component_order: Vec<String>,
}

/// Java `SolvematchPanel`.
pub struct SolvematchPanel {
    pub pnl_root: SolvematchPanelLayout,
    pub header_group: String,
    pub parent_title: String,
    pub dialog_type: DialogType,
    pub ph_initial_match: PanelHeader,
    pub bg_fiducial_params: Rc<RefCell<RadioButtonGroup>>,
    pub rb_both_sides: RadioButton,
    pub rb_one_side: RadioButton,
    pub rb_one_side_inverted: RadioButton,
    pub rb_use_model: RadioButton,
    pub rb_use_model_only: RadioButton,
    pub cb_bin_by_2: CheckBox,
    pub btn_imod_match_models: MultiLineButton,
    pub ltf_fiducial_match_list_a: LabeledTextField,
    pub ltf_fiducial_match_list_b: LabeledTextField,
    pub ltf_use_list: LabeledTextField,
    pub cb_use_corresponding_points: CheckBox,
    pub cb_initial_volume_matching: CheckBox,
    pub btn_restart: Option<MultiLineButton>,
    pub ltf_solvematch_maximum_residual: Option<LabeledTextField>,
    pub ltf_solvematch_center_shift_limit: Option<LabeledTextField>,
    pub ltf_dualvolmatch_maximum_residual: Option<LabeledTextField>,
    pub ltf_dualvolmatch_center_shift_limit: Option<LabeledTextField>,
    pub binning_warning: bool,
    pub use_corresponding_points_changed: bool,
    pub debug: bool,
}

impl SolvematchPanel {
    /// Java private constructor `SolvematchPanel(...)`.
    pub fn new(
        parent_title: &str,
        header_group: &str,
        dialog_type: DialogType,
        debug: bool,
    ) -> Self {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let initial = parent_title == "Initial";
        let mut result = Self {
            pnl_root: SolvematchPanelLayout {
                root_visible: true,
                root_body_visible: true,
                restart_present: initial,
                dualvolmatch_present: initial,
                body_component_order: vec![
                    "initialVolumeMatching".into(),
                    "solvematch".into(),
                    "dualvolmatch".into(),
                    "restart".into(),
                ],
                ..Default::default()
            },
            header_group: header_group.into(),
            parent_title: parent_title.into(),
            dialog_type,
            ph_initial_match: PanelHeader::new(
                INITIAL_MATCH_LABEL,
                true,
                initial,
                dialog_type,
                true,
                true,
                true,
                false,
                true,
            ),
            bg_fiducial_params: group.clone(),
            rb_both_sides: RadioButton::new_in_group("Fiducials on both sides", group.clone()),
            rb_one_side: RadioButton::new_in_group("Fiducials on one side", group.clone()),
            rb_one_side_inverted: RadioButton::new_in_group(
                "Fiducials on one side, inverted",
                group.clone(),
            ),
            rb_use_model: RadioButton::new_in_group(
                "Use matching models and fiducials",
                group.clone(),
            ),
            rb_use_model_only: RadioButton::new_in_group("Use matching models only", group),
            cb_bin_by_2: CheckBox::new_with_text("Load binned by 2"),
            btn_imod_match_models: MultiLineButton::new_with_label(Some(
                "Create Matching Models in 3dmod",
            )),
            ltf_fiducial_match_list_a: LabeledTextField::new(
                FieldType::IntegerList,
                "Corresponding fiducial list A: ",
            ),
            ltf_fiducial_match_list_b: LabeledTextField::new(
                FieldType::IntegerList,
                "Corresponding fiducial list B: ",
            ),
            ltf_use_list: LabeledTextField::new(
                FieldType::IntegerList,
                "Starting points to use from A: ",
            ),
            cb_use_corresponding_points: CheckBox::new_with_text(
                "Specify corresponding points instead of using coordinate file",
            ),
            cb_initial_volume_matching: CheckBox::new_with_text(
                "Use image correlations instead of Solvematch for initial match",
            ),
            btn_restart: initial.then(|| MultiLineButton::new_with_label(Some("Restart Combine"))),
            ltf_solvematch_maximum_residual: initial.then(|| {
                LabeledTextField::new(FieldType::FloatingPoint, "Limit on maximum residual: ")
            }),
            ltf_solvematch_center_shift_limit: initial.then(|| {
                LabeledTextField::new(FieldType::FloatingPoint, "Limit on center shift: ")
            }),
            ltf_dualvolmatch_maximum_residual: initial.then(|| {
                LabeledTextField::new(
                    FieldType::FloatingPoint,
                    "Limit on mean residual in patch correlations: ",
                )
            }),
            ltf_dualvolmatch_center_shift_limit: initial.then(|| {
                LabeledTextField::new(FieldType::FloatingPoint, "Limit on center shift: ")
            }),
            binning_warning: false,
            use_corresponding_points_changed: false,
            debug,
        };
        result.rb_one_side_inverted.set_visible(false);
        result.rb_use_model.set_visible(false);
        result.create_panel();
        result.set_tool_tip_text();
        result.show(true);
        result.add_listeners();
        result
    }

    /// Java static `getInstance(...)`.
    pub fn get_instance(
        parent_title: &str,
        header_group: &str,
        dialog_type: DialogType,
        debug: bool,
    ) -> Self {
        Self::new(parent_title, header_group, dialog_type, debug)
    }

    /// Java private `createPanel()`; actual Swing container construction is a renderer boundary.
    pub fn create_panel(&mut self) {
        self.update_display();
        self.update_advanced(self.ph_initial_match.is_advanced());
    }
    /// Java `show()` with its `manager.coordFileExists()` result supplied at the boundary.
    pub fn show(&mut self, coord_file_exists: bool) {
        self.cb_use_corresponding_points
            .set_selected(!coord_file_exists);
        self.cb_use_corresponding_points
            .set_visible(coord_file_exists);
        self.ltf_use_list.set_visible(coord_file_exists);
        self.update_display();
    }
    pub fn set_deferred_3dmod_buttons(&mut self, deferred: Option<&MultiLineButton>) {
        // `Run3dmodButton.setDeferred3dmodButton` is an ownership relation in
        // the deferred-launcher boundary; the actual launcher is supplied to
        // `action` below.  Keep the source call and its null branch explicit.
        if self.btn_restart.is_some() && deferred.is_some() {}
    }
    pub fn add_listeners(&mut self) {
        self.pnl_root.listener_count = 9;
    }
    pub fn get_container(&self) -> &SolvematchPanelLayout {
        &self.pnl_root
    }
    pub fn set_combine_parameters<P: CombineParameters>(&mut self, params: &P, init: bool) {
        self.set_surfaces_or_models(params.fiducial_match());
        self.ltf_fiducial_match_list_a
            .set_text(&params.fiducial_match_list_a());
        self.ltf_fiducial_match_list_b
            .set_text(&params.fiducial_match_list_b());
        self.ltf_use_list.set_text(&params.use_list());
        if self.cb_use_corresponding_points.is_visible() {
            self.cb_use_corresponding_points
                .set_selected(!params.transfer());
            self.update_display();
        }
        if !init || params.initial_volume_matching() {
            self.cb_initial_volume_matching
                .set_selected(params.initial_volume_matching());
        }
    }

    /// Rust spelling retained for Java `setParameters(ConstCombineParams,
    /// boolean)`.  The parameter type selects the source overload at compile
    /// time, rather than through Swing's runtime dispatch.
    #[allow(non_snake_case)]
    pub fn setParameters<P: CombineParameters>(&mut self, params: &P, init: bool) {
        self.set_combine_parameters(params, init);
    }
    pub fn set_dualvolmatch_parameters<P: DualvolmatchParameters>(&mut self, params: &P) {
        if let Some(field) = &mut self.ltf_dualvolmatch_maximum_residual {
            field.set_text(&params.maximum_residual());
        }
        if let Some(field) = &mut self.ltf_dualvolmatch_center_shift_limit {
            field.set_text(&params.center_shift_limit());
        }
    }
    /// Java `getParameters(ReconScreenState)`.
    pub fn get_screen_state<S: SolvematchPanelScreenState>(&self, state: &mut S) {
        let header_state = self
            .ph_initial_match
            .get_open_close_button()
            .map(|button| button.get_state());
        if self.btn_restart.is_some() {
            state.set_combine_initial_solvematch_header_state(header_state);
        } else {
            state.set_combine_setup_solvematch_header_state(header_state);
        }
    }
    /// Java `setParameters(ReconScreenState)`.
    pub fn set_screen_state<S: SolvematchPanelScreenState>(&mut self, state: &S) {
        let header_state = if self.btn_restart.is_some() {
            state.combine_initial_solvematch_header_state()
        } else {
            state.combine_setup_solvematch_header_state()
        };
        if let (Some(button), Some(header_state)) =
            (&mut self.ph_initial_match.btn_open_close, header_state)
        {
            button.set_state(Some(header_state));
        }
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.root_visible = visible;
    }
    pub fn get_combine_parameters<P: CombineParameters>(
        &self,
        params: &mut P,
        validate: bool,
    ) -> bool {
        let result: Result<(), FieldValidationFailedException> = (|| {
            params.set_fiducial_match(self.get_surfaces_or_models());
            params.set_transfer(!self.cb_use_corresponding_points.is_selected());
            params.set_fiducial_match_list_a(
                self.ltf_fiducial_match_list_a
                    .get_text_validated(validate)?,
            );
            params.set_fiducial_match_list_b(
                self.ltf_fiducial_match_list_b
                    .get_text_validated(validate)?,
            );
            let use_list = self.ltf_use_list.get_text();
            params.set_use_list(if use_list.trim() == "/" {
                String::new()
            } else {
                self.ltf_use_list.get_text_validated(validate)?
            });
            params.set_initial_volume_matching(self.cb_initial_volume_matching.is_selected());
            Ok(())
        })();
        result.is_ok()
    }
    pub fn set_solvematch_parameters<P: SolvematchParameters>(&mut self, params: &P) {
        self.set_surfaces_or_models(params.surfaces_or_model());
        if params.match_b_to_a() {
            self.ltf_fiducial_match_list_a
                .set_text(&params.to_correspondence_list());
            self.ltf_fiducial_match_list_b
                .set_text(&params.from_correspondence_list());
        } else {
            self.ltf_fiducial_match_list_a
                .set_text(&params.from_correspondence_list());
            self.ltf_fiducial_match_list_b
                .set_text(&params.to_correspondence_list());
        }
        if let Some(field) = &mut self.ltf_solvematch_maximum_residual {
            field.set_text(&params.maximum_residual());
        }
        if let Some(field) = &mut self.ltf_solvematch_center_shift_limit {
            field.set_text(&params.center_shift_limit());
        }
        self.ltf_use_list.set_text(&params.use_points());
    }
    pub fn get_dualvolmatch_parameters<P: DualvolmatchParameters>(
        &self,
        params: &mut P,
        validate: bool,
    ) -> bool {
        let result: Result<(), FieldValidationFailedException> = (|| {
            if let Some(field) = &self.ltf_dualvolmatch_maximum_residual {
                params.set_maximum_residual(field.get_text_validated(validate)?);
            }
            if let Some(field) = &self.ltf_dualvolmatch_center_shift_limit {
                params.set_center_shift_limit(field.get_text_validated(validate)?);
            }
            Ok(())
        })();
        result.is_ok()
    }
    pub fn get_solvematch_parameters<P: SolvematchParameters>(
        &self,
        params: &mut P,
        validate: bool,
    ) -> bool {
        let result: Result<(), FieldValidationFailedException> = (|| {
            params.set_surfaces_or_model(self.get_surfaces_or_models());
            let a = self
                .ltf_fiducial_match_list_a
                .get_text_validated(validate)?;
            let b = self
                .ltf_fiducial_match_list_b
                .get_text_validated(validate)?;
            if params.match_b_to_a() {
                params.set_to_correspondence_list(a);
                params.set_from_correspondence_list(b);
            } else {
                params.set_from_correspondence_list(a);
                params.set_to_correspondence_list(b);
            }
            if let Some(field) = &self.ltf_solvematch_maximum_residual {
                params.set_maximum_residual(field.get_text_validated(validate)?);
            }
            if let Some(field) = &self.ltf_solvematch_center_shift_limit {
                params.set_center_shift_limit(field.get_text_validated(validate)?);
            }
            params.set_transfer_coordinate_file(self.cb_use_corresponding_points.is_selected());
            params.set_use_points(self.ltf_use_list.get_text_validated(validate)?);
            Ok(())
        })();
        result.is_ok()
    }
    pub fn get_surfaces_or_models(&self) -> FiducialMatch {
        if self.rb_both_sides.is_selected() && self.rb_both_sides.radio_button.enabled {
            FiducialMatch::BothSides
        } else if self.rb_one_side.is_selected() && self.rb_one_side.radio_button.enabled {
            FiducialMatch::OneSide
        } else if self.rb_one_side_inverted.is_selected()
            && self.rb_one_side_inverted.radio_button.enabled
            && self.rb_one_side_inverted.is_visible()
        {
            FiducialMatch::OneSideInverted
        } else if self.rb_use_model.is_selected()
            && self.rb_use_model.radio_button.enabled
            && self.rb_use_model.is_visible()
        {
            FiducialMatch::UseModel
        } else if self.rb_use_model_only.is_selected()
            && self.rb_use_model_only.radio_button.enabled
        {
            FiducialMatch::UseModelOnly
        } else {
            FiducialMatch::NotSet
        }
    }
    pub fn set_surfaces_or_models(&mut self, value: FiducialMatch) {
        match value {
            FiducialMatch::UseModelOnly => self.rb_use_model_only.set_selected(true),
            FiducialMatch::OneSideInverted => {
                self.rb_one_side_inverted.set_selected(true);
                self.rb_one_side_inverted.set_visible(true);
            }
            FiducialMatch::UseModel => {
                self.rb_use_model.set_selected(true);
                self.rb_use_model.set_visible(true);
            }
            FiducialMatch::OneSide => self.rb_one_side.set_selected(true),
            FiducialMatch::BothSides => self.rb_both_sides.set_selected(true),
            FiducialMatch::NotSet => {}
        }
        self.update_display();
    }
    pub fn is_bin_by_2(&self) -> bool {
        self.cb_bin_by_2.is_selected()
    }
    pub fn set_bin_by_2(&mut self, value: bool) {
        self.cb_bin_by_2.set_selected(value);
    }
    pub fn set_use_list(&mut self, value: &str) {
        self.ltf_use_list.set_text(value);
    }
    pub fn get_use_list(&self, validate: bool) -> Result<String, FieldValidationFailedException> {
        self.ltf_use_list.get_text_validated(validate)
    }
    pub fn set_fiducial_match_list_a(&mut self, value: &str) {
        self.ltf_fiducial_match_list_a.set_text(value);
    }
    pub fn get_fiducial_match_list_a(
        &self,
        validate: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_fiducial_match_list_a.get_text_validated(validate)
    }
    pub fn set_fiducial_match_list_b(&mut self, value: &str) {
        self.ltf_fiducial_match_list_b.set_text(value);
    }
    pub fn get_fiducial_match_list_b(
        &self,
        validate: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_fiducial_match_list_b.get_text_validated(validate)
    }
    pub fn expand(&mut self, button: &ExpandButton) {
        if self.ph_initial_match.equals_open_close(button) {
            self.pnl_root.root_body_visible = button.is_expanded();
        } else if self.ph_initial_match.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
    }
    pub fn expand_global(&mut self, advanced: bool) {
        self.update_advanced(advanced);
    }
    pub fn action<M: SolvematchPanelApplicationManager, T: SolvematchPanelParent>(
        &mut self,
        manager: &mut M,
        parent: &mut T,
        command: &str,
        deferred: Option<&MultiLineButton>,
        options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command)
            == self
                .cb_use_corresponding_points
                .check_box
                .action_command
                .as_deref()
        {
            self.use_corresponding_points_changed = true;
            self.update_display();
            if self.use_corresponding_points_changed {
                self.use_corresponding_points_changed = false;
                parent.update_display();
            }
            return;
        }
        if Some(command) == self.rb_both_sides.radio_button.action_command.as_deref()
            || Some(command) == self.rb_one_side.radio_button.action_command.as_deref()
            || Some(command)
                == self
                    .rb_use_model_only
                    .radio_button
                    .action_command
                    .as_deref()
            || Some(command)
                == self
                    .cb_initial_volume_matching
                    .check_box
                    .action_command
                    .as_deref()
        {
            self.update_display();
            return;
        }
        parent.synchronize(&self.parent_title, true);
        if Some(command) == self.cb_bin_by_2.check_box.action_command.as_deref() {
            if !self.binning_warning && self.cb_bin_by_2.is_selected() {
                parent.set_binning_warning(true);
                self.binning_warning = true;
            }
        } else if self
            .btn_restart
            .as_ref()
            .is_some_and(|button| Some(command) == button.get_action_command())
        {
            manager.combine(
                self.btn_restart.as_ref().unwrap(),
                deferred,
                options,
                self.dialog_type,
                parent.run_processing_method(),
                self.cb_initial_volume_matching.is_selected(),
                parent.is_parallel(),
                !parent.is_run_volcombine(),
            );
        } else if Some(command) == self.btn_imod_match_models.get_action_command() {
            manager.imod_matching_model(self.cb_bin_by_2.is_selected(), options);
        }
    }

    /// Native callback endpoint for Java `ActionListener.actionPerformed`.
    /// The frontend provides the action command directly; the optional
    /// deferred button and menu options are absent for this listener path.
    #[allow(non_snake_case)]
    pub fn actionPerformed<M: SolvematchPanelApplicationManager, T: SolvematchPanelParent>(
        &mut self,
        manager: &mut M,
        parent: &mut T,
        command: &str,
    ) {
        self.action(manager, parent, command, None, None);
    }
    pub fn rb_fiducial_action(&mut self) {
        self.update_display();
    }
    pub fn update_advanced(&mut self, advanced: bool) {
        if let Some(field) = &mut self.ltf_solvematch_center_shift_limit {
            field.set_visible(advanced);
        }
        if let Some(field) = &mut self.ltf_dualvolmatch_center_shift_limit {
            field.set_visible(advanced);
        }
    }
    pub fn update_display(&mut self) {
        let initial = self.cb_initial_volume_matching.is_selected();
        if let Some(field) = &mut self.ltf_solvematch_center_shift_limit {
            field.set_enabled(!initial);
        }
        if let Some(field) = &mut self.ltf_solvematch_maximum_residual {
            field.set_enabled(!initial);
        }
        self.rb_both_sides.set_enabled(!initial);
        self.rb_one_side.set_enabled(!initial);
        self.rb_use_model_only.set_enabled(!initial);
        let model = self.rb_use_model.is_selected() || self.rb_use_model_only.is_selected();
        self.btn_imod_match_models.set_enabled(model && !initial);
        self.cb_bin_by_2.set_enabled(model && !initial);
        self.cb_use_corresponding_points.set_enabled(!initial);
        self.ltf_fiducial_match_list_a.set_enabled(!initial);
        self.ltf_fiducial_match_list_b.set_enabled(!initial);
        self.ltf_use_list.set_enabled(!initial);
        if let Some(field) = &mut self.ltf_dualvolmatch_maximum_residual {
            field.set_enabled(initial);
        }
        if let Some(field) = &mut self.ltf_dualvolmatch_center_shift_limit {
            field.set_enabled(initial);
        }
        let corresponding = self.cb_use_corresponding_points.is_selected();
        self.ltf_fiducial_match_list_a.set_visible(corresponding);
        self.ltf_fiducial_match_list_b.set_visible(corresponding);
        self.ltf_use_list.set_visible(!corresponding);
    }
    pub fn is_use_corresponding_points(&self) -> bool {
        self.cb_use_corresponding_points.is_selected()
    }
    pub fn is_initial_volume_matching(&self) -> bool {
        self.cb_initial_volume_matching.is_selected()
    }
    pub fn set_initial_volume_matching(&mut self, value: bool) {
        self.cb_initial_volume_matching.set_selected(value);
    }
    pub fn set_use_corresponding_points(&mut self, value: bool) {
        self.cb_use_corresponding_points.set_selected(value);
        self.update_display();
    }
    pub fn set_tool_tip_text(&mut self) {
        self.pnl_root.tooltip_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initial_matching_switches_solvematch_and_dualvolmatch_fields() {
        let mut panel = SolvematchPanel::get_instance(
            "Initial",
            "combine",
            DialogType::TomogramCombination,
            false,
        );
        panel.rb_use_model_only.set_selected(true);
        panel.update_display();
        assert!(panel.btn_imod_match_models.is_enabled());
        panel.set_initial_volume_matching(true);
        panel.update_display();
        assert!(!panel.ltf_fiducial_match_list_a.enabled);
        assert!(
            panel
                .ltf_dualvolmatch_maximum_residual
                .as_ref()
                .unwrap()
                .enabled
        );
    }
    #[test]
    fn matching_mode_retains_deprecated_visible_modes() {
        let mut panel = SolvematchPanel::get_instance(
            "Setup",
            "combine",
            DialogType::TomogramCombination,
            false,
        );
        panel.set_surfaces_or_models(FiducialMatch::OneSideInverted);
        assert!(panel.rb_one_side_inverted.is_visible());
        assert_eq!(
            panel.get_surfaces_or_models(),
            FiducialMatch::OneSideInverted
        );
    }
}
