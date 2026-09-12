//! `IMOD/Etomo/src/etomo/ui/swing/RaptorPanel.java`.
//!
//! Swing component construction and `ApplicationManager` calls are direct
//! boundaries.  The source unit's controls, parameter transfer, validation
//! order, and four-way action dispatch remain owned by this panel.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::context_popup::{ContextPopup, MouseEvent, TOMO_GUIDE};
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::multi_line_button::MultiLineButton;
use super::radio_button::{RadioButton, RadioButtonGroup};
use super::tilt_panel::Deferred3dmodButton;

pub const MARK_LABEL: &str = "# of beads to choose";
pub const DIAM_LABEL: &str = "Unbinned Bead diameter";
pub const RUN_RAPTOR_LABEL: &str = "Run RAPTOR";
pub const USE_RAPTOR_RESULT_LABEL: &str = "Use RAPTOR Result as Fiducial Model";

/// The `BeadtrackParam` read performed by `setBeadtrackParams`.
pub trait RaptorBeadtrackParam {
    fn bead_diameter(&self) -> Option<f64>;
}

/// The `RunraptorParam` write boundary used by `getParameters`.
pub trait RaptorRunraptorParam {
    fn set_use_raw_stack(&mut self, use_raw_stack: bool);
    fn set_mark(&mut self, mark: String) -> Option<String>;
    fn set_diam(&mut self, diameter: String, preali_stack: bool) -> Option<String>;
}

/// The `ConstMetaData` and `MetaData` fields touched by this source unit.
pub trait RaptorMetaData {
    fn track_raptor_use_raw_stack(&self) -> bool;
    fn track_raptor_mark(&self) -> String;
    fn track_raptor_diam(&self) -> Option<String>;
    fn set_track_raptor_use_raw_stack(&mut self, value: bool);
    fn set_track_raptor_mark(&mut self, value: String);
    fn set_track_raptor_diam(&mut self, value: String);
}

/// The direct `ApplicationManager`/`UIHarness` calls in `RaptorPanel.java`.
pub trait RaptorPanelApplicationManager {
    fn view_type(&self) -> ViewType;
    fn imod_raw_stack(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn imod_coarse_align(
        &mut self,
        axis_id: AxisID,
        options: Option<Run3dmodMenuOptions>,
        model: Option<&str>,
        something: bool,
    );
    fn runraptor(
        &mut self,
        button: &MultiLineButton,
        process_series: Option<&str>,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        axis_id: AxisID,
    );
    fn imod_runraptor_result(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn use_runraptor_result(
        &mut self,
        button: &MultiLineButton,
        axis_id: AxisID,
        dialog_type: DialogType,
    );
    fn open_message_dialog(&mut self, message: String, title: &str, axis_id: AxisID);
}

/// Source-visible Swing layout/listener state of `RaptorPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RaptorPanelLayout {
    pub root_box_layout_y_axis: bool,
    pub root_border_title: Option<String>,
    pub root_component_order: Vec<String>,
    pub input_box_layout_y_axis: bool,
    pub input_border_etched: bool,
    pub input_component_order: Vec<String>,
    pub raptor_buttons_box_layout_x_axis: bool,
    pub raptor_buttons_component_order: Vec<String>,
    pub root_visible: bool,
    pub mouse_listener_count: usize,
    pub tooltip_initialized: bool,
}

/// Java `RaptorPanel`.
pub struct RaptorPanel {
    pub pnl_root: RaptorPanelLayout,
    pub btn_open_stack: MultiLineButton,
    pub ltf_mark: LabeledTextField,
    pub ltf_diam: LabeledTextField,
    pub bg_input: Rc<RefCell<RadioButtonGroup>>,
    pub rb_input_preali: RadioButton,
    pub rb_input_raw: RadioButton,
    pub btn_raptor: MultiLineButton,
    pub btn_open_raptor_result: MultiLineButton,
    pub btn_use_raptor_result: MultiLineButton,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    /// Java `btnRaptor.setDeferred3dmodButton(btnOpenRaptorResult)` boundary.
    pub raptor_deferred_result_button: bool,
    /// Java's private inner listener is represented by its registered target.
    pub action_listener_registered: bool,
    pub last_context_popup: Option<ContextPopup>,
}

impl RaptorPanel {
    /// Java private `RaptorPanel(ApplicationManager, AxisID, DialogType)`.
    pub fn new(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let bg_input = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut result = Self {
            pnl_root: RaptorPanelLayout {
                root_visible: true,
                ..Default::default()
            },
            btn_open_stack: MultiLineButton::new_with_label(Some("Open Stack in 3dmod")),
            ltf_mark: LabeledTextField::new(FieldType::Integer, "# of beads to choose: "),
            ltf_diam: LabeledTextField::new(
                FieldType::Integer,
                "Unbinned Bead diameter (in pixels): ",
            ),
            rb_input_preali: RadioButton::new_in_group(
                "Run against the coarse aligned stack",
                bg_input.clone(),
            ),
            rb_input_raw: RadioButton::new_in_group("Run against the raw stack", bg_input.clone()),
            bg_input,
            btn_raptor: MultiLineButton::new_with_label(Some(RUN_RAPTOR_LABEL)),
            btn_open_raptor_result: MultiLineButton::new_with_label(Some(
                "Open RAPTOR Model in 3dmod",
            )),
            btn_use_raptor_result: MultiLineButton::new_with_label(Some(USE_RAPTOR_RESULT_LABEL)),
            axis_id,
            dialog_type,
            raptor_deferred_result_button: false,
            action_listener_registered: false,
            last_context_popup: None,
        };
        // Swing `AbstractButton` defaults its action command to its text.  The
        // generic `MultiLineButton` boundary has no native event model, so retain
        // that source-visible default here for this panel's listener dispatch.
        result
            .btn_open_stack
            .set_action_command(Some("Open Stack in 3dmod"));
        result.btn_raptor.set_action_command(Some(RUN_RAPTOR_LABEL));
        result
            .btn_open_raptor_result
            .set_action_command(Some("Open RAPTOR Model in 3dmod"));
        result
            .btn_use_raptor_result
            .set_action_command(Some(USE_RAPTOR_RESULT_LABEL));
        result
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType)`.
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut instance = Self::new(axis_id, dialog_type);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.pnl_root.mouse_listener_count += 1;
        self.btn_open_stack.add_action_listener();
        self.btn_raptor.add_action_listener();
        self.btn_open_raptor_result.add_action_listener();
        self.btn_use_raptor_result.add_action_listener();
        self.action_listener_registered = true;
    }

    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        let man_page_label = ["Raptor".into(), "Beadtrack".into(), "3dmod".into()];
        let man_page = [
            "raptor.html".into(),
            "beadtrack.html".into(),
            "3dmod.html".into(),
        ];
        let log_file_label = ["Track".into()];
        let log_file = [format!("track{}.log", self.axis_id.get_extension())];
        self.last_context_popup = ContextPopup::new_log_files(
            mouse_event,
            Some("UsingRaptor"),
            TOMO_GUIDE,
            &man_page_label,
            &man_page,
            &log_file_label,
            &log_file,
            self.axis_id,
            None,
        )
        .ok();
    }

    /// Java private `createPanel`.
    pub fn create_panel(&mut self) {
        self.pnl_root.root_box_layout_y_axis = true;
        self.pnl_root.root_border_title = Some(RUN_RAPTOR_LABEL.into());
        self.pnl_root.root_component_order = vec![
            "pnlInput".into(),
            "btnOpenStack".into(),
            "ltfMark".into(),
            "ltfDiam".into(),
            "pnlRaptorButtons".into(),
        ];
        self.pnl_root.input_box_layout_y_axis = true;
        self.pnl_root.input_border_etched = true;
        self.pnl_root.input_component_order = vec!["rbInputPreali".into(), "rbInputRaw".into()];
        self.pnl_root.raptor_buttons_box_layout_x_axis = true;
        self.pnl_root.raptor_buttons_component_order = vec![
            "btnRaptor".into(),
            "btnOpenRaptorResult".into(),
            "btnUseRaptorResult".into(),
        ];
        self.rb_input_preali.set_selected(true);
        self.btn_open_stack.set_alignment_x(0.5);
        self.raptor_deferred_result_button = true;
    }

    /// Java `done`.
    pub fn done(&mut self) {
        self.btn_raptor.remove_action_listener();
        self.btn_open_raptor_result.remove_action_listener();
        self.btn_use_raptor_result.remove_action_listener();
    }

    /// Java `setBeadtrackParams(BeadtrackParam)`.
    pub fn set_beadtrack_params<P: RaptorBeadtrackParam>(&mut self, beadtrack_params: &P) {
        if let Some(bead_diameter) = beadtrack_params.bead_diameter() {
            self.ltf_diam.set_text_number(bead_diameter.round() as i64);
        }
    }

    /// Java `getParameters(RunraptorParam, boolean)`.
    pub fn get_parameters<P: RaptorRunraptorParam, M: RaptorPanelApplicationManager>(
        &mut self,
        param: &mut P,
        do_validation: bool,
        manager: &mut M,
    ) -> bool {
        param.set_use_raw_stack(self.rb_input_raw.is_selected());
        let mark = match self.ltf_mark.get_text_validated(do_validation) {
            Ok(mark) => mark,
            Err(FieldValidationFailedException(_)) => return false,
        };
        if let Some(error_message) = param.set_mark(mark) {
            manager.open_message_dialog(
                format!("Error in {MARK_LABEL}: {error_message}"),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        let diameter = match self.ltf_diam.get_text_validated(do_validation) {
            Ok(diameter) => diameter,
            Err(FieldValidationFailedException(_)) => return false,
        };
        if let Some(error_message) = param.set_diam(diameter, self.rb_input_preali.is_selected()) {
            manager.open_message_dialog(
                format!("Error in {DIAM_LABEL}: {error_message}"),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        true
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_metadata_parameters<M: RaptorMetaData>(&self, meta_data: &mut M) {
        if self.axis_id != AxisID::Second {
            meta_data.set_track_raptor_use_raw_stack(self.rb_input_raw.is_selected());
            meta_data.set_track_raptor_mark(self.ltf_mark.get_text());
            meta_data.set_track_raptor_diam(self.ltf_diam.get_text());
        }
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters<M: RaptorMetaData, A: RaptorPanelApplicationManager>(
        &mut self,
        meta_data: &M,
        manager: &A,
    ) {
        if self.axis_id != AxisID::Second {
            if meta_data.track_raptor_use_raw_stack() {
                self.rb_input_raw.set_selected(true);
            } else {
                self.rb_input_preali.set_selected(true);
            }
            self.ltf_mark.set_text(&meta_data.track_raptor_mark());
            if let Some(diameter) = meta_data.track_raptor_diam() {
                self.ltf_diam.set_text(&diameter);
            }
        }
        if manager.view_type() == ViewType::Montage {
            self.rb_input_preali.set_selected(true);
            self.rb_input_raw.set_enabled(false);
        }
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: RaptorPanelApplicationManager>(
        &mut self,
        command: &str,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        manager: &mut M,
    ) {
        if self.btn_open_stack.get_action_command() == Some(command) {
            if self.rb_input_raw.is_selected() {
                manager.imod_raw_stack(self.axis_id, run_3dmod_menu_options);
            } else {
                manager.imod_coarse_align(self.axis_id, run_3dmod_menu_options, None, false);
            }
        } else if self.btn_raptor.get_action_command() == Some(command) {
            manager.runraptor(
                &self.btn_raptor,
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                DialogType::FiducialModel,
                self.axis_id,
            );
        } else if self.btn_open_raptor_result.get_action_command() == Some(command) {
            manager.imod_runraptor_result(self.axis_id, run_3dmod_menu_options);
        } else if self.btn_use_raptor_result.get_action_command() == Some(command) {
            manager.use_runraptor_result(
                &self.btn_use_raptor_result,
                self.axis_id,
                DialogType::FiducialModel,
            );
        }
    }

    /// Java `getComponent`; concrete component is a GUI boundary.
    pub fn get_component(&self) -> &RaptorPanelLayout {
        &self.pnl_root
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.root_visible = visible;
    }

    /// Java private `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.rb_input_preali
            .set_tool_tip_text(Some("Run RAPTOR against the coarsely aligned stack."));
        self.rb_input_raw
            .set_tool_tip_text(Some("Run RAPTOR against the raw stack."));
        self.btn_open_stack
            .set_tool_tip_text(Some("Opens the file that RAPTOR will be run against."));
        self.ltf_mark
            .set_tool_tip_text(Some("Number of markers to track."));
        self.ltf_diam
            .set_tool_tip_text(Some("Bead diameter in pixels."));
        self.btn_raptor
            .set_tool_tip_text(Some("Runs the runraptor script"));
        self.btn_open_raptor_result.set_tool_tip_text(Some(
            "Opens the model generated by RAPTOR and the file that RAPTOR was run against.",
        ));
        self.btn_use_raptor_result.set_tool_tip_text(Some(
            "Copies the model generated by RAPTOR to the .fid file.",
        ));
        self.pnl_root.tooltip_initialized = true;
    }

    /// Java inner `RaptorPanelActionListener.actionPerformed(ActionEvent)`.
    pub fn action_performed<M: RaptorPanelApplicationManager>(
        &mut self,
        command: &str,
        manager: &mut M,
    ) {
        self.action(command, None, None, manager);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Param {
        raw: bool,
        mark: String,
        diam: String,
    }
    impl RaptorRunraptorParam for Param {
        fn set_use_raw_stack(&mut self, value: bool) {
            self.raw = value;
        }
        fn set_mark(&mut self, value: String) -> Option<String> {
            self.mark = value;
            None
        }
        fn set_diam(&mut self, value: String, _: bool) -> Option<String> {
            self.diam = value;
            None
        }
    }
    struct Manager {
        calls: Vec<&'static str>,
        view: ViewType,
    }
    impl Default for Manager {
        fn default() -> Self {
            Self {
                calls: Vec::new(),
                view: ViewType::SingleView,
            }
        }
    }
    impl RaptorPanelApplicationManager for Manager {
        fn view_type(&self) -> ViewType {
            self.view
        }
        fn imod_raw_stack(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {
            self.calls.push("raw");
        }
        fn imod_coarse_align(
            &mut self,
            _: AxisID,
            _: Option<Run3dmodMenuOptions>,
            _: Option<&str>,
            _: bool,
        ) {
            self.calls.push("coarse");
        }
        fn runraptor(
            &mut self,
            _: &MultiLineButton,
            _: Option<&str>,
            _: Option<&Deferred3dmodButton>,
            _: Option<Run3dmodMenuOptions>,
            _: DialogType,
            _: AxisID,
        ) {
            self.calls.push("raptor");
        }
        fn imod_runraptor_result(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {
            self.calls.push("result");
        }
        fn use_runraptor_result(&mut self, _: &MultiLineButton, _: AxisID, _: DialogType) {
            self.calls.push("use");
        }
        fn open_message_dialog(&mut self, _: String, _: &str, _: AxisID) {
            self.calls.push("message");
        }
    }
    #[test]
    fn create_panel_retains_source_initial_selection_and_layout() {
        let panel = RaptorPanel::get_instance(AxisID::First, DialogType::FiducialModel);
        assert!(panel.rb_input_preali.is_selected());
        assert!(panel.raptor_deferred_result_button);
        assert_eq!(panel.pnl_root.root_component_order.len(), 5);
    }
    #[test]
    fn parameters_validate_and_transfer_in_source_order() {
        let mut panel = RaptorPanel::get_instance(AxisID::First, DialogType::FiducialModel);
        panel.ltf_mark.set_text("12");
        panel.ltf_diam.set_text("10");
        let mut param = Param::default();
        let mut manager = Manager::default();
        assert!(panel.get_parameters(&mut param, true, &mut manager));
        assert_eq!((param.mark.as_str(), param.diam.as_str()), ("12", "10"));
    }
    #[test]
    fn action_dispatches_all_source_commands() {
        let mut panel = RaptorPanel::get_instance(AxisID::First, DialogType::FiducialModel);
        let mut manager = Manager::default();
        for command in [
            "Open Stack in 3dmod",
            RUN_RAPTOR_LABEL,
            "Open RAPTOR Model in 3dmod",
            USE_RAPTOR_RESULT_LABEL,
        ] {
            panel.action(command, None, None, &mut manager);
        }
        assert_eq!(manager.calls, ["coarse", "raptor", "result", "use"]);
    }
}
