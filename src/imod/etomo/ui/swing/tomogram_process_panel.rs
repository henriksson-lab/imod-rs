//! `IMOD/Etomo/src/etomo/ui/swing/TomogramProcessPanel.java`.
//!
//! The source panel's Swing controls are retained as source-visible form state.
//! `ApplicationManager`, `UIHarness`, and the three expert dialog targets are
//! explicit call boundaries: their concrete implementations are supplied to
//! event methods instead of being replaced with invented application logic.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::process_control_panel::ProcessControlPanel;
use super::simple_button::SimpleButton;
use super::ui_utilities::UiUtilities;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::util::utilities;

pub const BOTH_AXIS_LABEL: &str = "Both";
pub const AXIS_B_LABEL: &str = "Axis B";
const AXIS_A_LABEL: &str = "Axis A";

/// Direct `UIHarness` calls in this Java source unit.
pub trait TomogramProcessPanelUiHarness {
    fn show_both_axis(&mut self);
    fn show_axis_a(&mut self);
    fn show_axis_b(&mut self);
    fn move_sub_frame(&mut self);
}

/// Direct `ApplicationManager` and expert calls in this Java source unit.
pub trait TomogramProcessPanelApplicationManager {
    fn save_current_dialog(&mut self, axis_id: AxisID);
    fn open_pre_proc_dialog(&mut self, axis_id: AxisID);
    fn open_coarse_align_dialog(&mut self, axis_id: AxisID);
    fn open_fiducial_model_dialog(&mut self, axis_id: AxisID);
    fn open_fine_alignment_dialog(&mut self, axis_id: AxisID);
    fn open_tomogram_positioning_dialog(&mut self, axis_id: AxisID);
    fn open_final_aligned_stack_dialog(&mut self, axis_id: AxisID);
    fn open_tomogram_generation_dialog(&mut self, axis_id: AxisID);
    fn open_tomogram_combination_dialog(&mut self);
    fn open_post_processing_dialog(&mut self);
    fn open_clean_up_dialog(&mut self);
}

/// Java `JPanel axisButtonPanel` fields used by this unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AxisButtonPanelState {
    pub visible: bool,
    pub background: Option<String>,
    pub vertical_layout: bool,
    pub component_order: Vec<String>,
}

/// Java package-private final `TomogramProcessPanel` including its superclass.
pub struct TomogramProcessPanel {
    pub axis_process_panel: AxisProcessPanel,
    pub proc_ctl_pre_proc: ProcessControlPanel,
    pub proc_ctl_coarse_align: ProcessControlPanel,
    pub proc_ctl_fiducial_model: ProcessControlPanel,
    pub proc_ctl_fine_alignment: ProcessControlPanel,
    pub proc_ctl_tomogram_positioning: ProcessControlPanel,
    pub proc_ctl_final_aligned_stack: ProcessControlPanel,
    pub proc_ctl_tomogram_generation: ProcessControlPanel,
    pub proc_ctl_tomogram_combination: ProcessControlPanel,
    pub proc_ctl_post_processing: ProcessControlPanel,
    pub proc_ctl_clean_up: ProcessControlPanel,
    pub axis_button_1: SimpleButton,
    pub axis_button_2: SimpleButton,
    pub axis_button_panel: AxisButtonPanelState,
    pub both_axis_tooltip: Option<String>,
    pub axis_a_tooltip: Option<String>,
    pub axis_b_tooltip: Option<String>,
    /// Java `busyStatusMediator`; its concrete direct type is not yet a Rust unit.
    pub busy_status_mediator_present: bool,
    /// Source order of Swing additions to `panelProcessSelect`.
    pub process_select_component_order: Vec<String>,
}

impl TomogramProcessPanel {
    /// `TomogramProcessPanel(ApplicationManager, AxisID, AxisProgressPanel)`.
    /// `compact_display` is read from `EtomoDirector.INSTANCE` by Java and is
    /// explicit here because the director is a separate global boundary.
    pub fn new(
        application_manager: &'static dyn BaseManager,
        axis: AxisID,
        axis_progress_panel: AxisProgressPanel,
        compact_display: bool,
    ) -> Self {
        let axis_process_panel = AxisProcessPanel::new(
            axis,
            application_manager,
            true,
            true,
            InterfaceType::Recon,
            false,
            axis_progress_panel,
        );
        let mut instance = Self {
            axis_process_panel,
            proc_ctl_pre_proc: ProcessControlPanel::new(DialogType::PreProcessing, compact_display),
            proc_ctl_coarse_align: ProcessControlPanel::new(
                DialogType::CoarseAlignment,
                compact_display,
            ),
            proc_ctl_fiducial_model: ProcessControlPanel::new(
                DialogType::FiducialModel,
                compact_display,
            ),
            proc_ctl_fine_alignment: ProcessControlPanel::new(
                DialogType::FineAlignment,
                compact_display,
            ),
            proc_ctl_tomogram_positioning: ProcessControlPanel::new(
                DialogType::TomogramPositioning,
                compact_display,
            ),
            proc_ctl_final_aligned_stack: ProcessControlPanel::new(
                DialogType::FinalAlignedStack,
                compact_display,
            ),
            proc_ctl_tomogram_generation: ProcessControlPanel::new(
                DialogType::TomogramGeneration,
                compact_display,
            ),
            proc_ctl_tomogram_combination: ProcessControlPanel::new(
                DialogType::TomogramCombination,
                compact_display,
            ),
            proc_ctl_post_processing: ProcessControlPanel::new(
                DialogType::PostProcessing,
                compact_display,
            ),
            proc_ctl_clean_up: ProcessControlPanel::new(DialogType::CleanUp, compact_display),
            axis_button_1: SimpleButton::new(),
            axis_button_2: SimpleButton::new(),
            axis_button_panel: AxisButtonPanelState::default(),
            both_axis_tooltip: None,
            axis_a_tooltip: None,
            axis_b_tooltip: None,
            busy_status_mediator_present: true,
            process_select_component_order: Vec::new(),
        };
        instance.create_process_control_panel(compact_display);
        instance.axis_process_panel.initialize_panels(true);
        instance
    }

    /// `buttonAxisAction(ActionEvent)`.
    pub fn button_axis_action<U: TomogramProcessPanelUiHarness>(
        &mut self,
        command: &str,
        ui_harness: &mut U,
    ) {
        if command == BOTH_AXIS_LABEL {
            ui_harness.show_both_axis();
        } else if command == AXIS_A_LABEL {
            ui_harness.show_axis_a();
        } else if command == AXIS_B_LABEL {
            ui_harness.show_axis_b();
        }
    }

    /// `buttonProcessAction(ActionEvent)`.
    pub fn button_process_action<
        M: TomogramProcessPanelApplicationManager,
        U: TomogramProcessPanelUiHarness,
    >(
        &mut self,
        command: &str,
        application_manager: &mut M,
        ui_harness: &mut U,
    ) {
        utilities::button_timestamp(Some(command));
        application_manager.save_current_dialog(self.axis_process_panel.axis_id);
        if command == self.proc_ctl_pre_proc.get_command() {
            application_manager.open_pre_proc_dialog(self.axis_process_panel.axis_id);
        } else if command == self.proc_ctl_coarse_align.get_command() {
            application_manager.open_coarse_align_dialog(self.axis_process_panel.axis_id);
        } else if command == self.proc_ctl_fiducial_model.get_command() {
            application_manager.open_fiducial_model_dialog(self.axis_process_panel.axis_id);
        } else if command == self.proc_ctl_fine_alignment.get_command() {
            application_manager.open_fine_alignment_dialog(self.axis_process_panel.axis_id);
        } else if command == self.proc_ctl_tomogram_positioning.get_command() {
            application_manager.open_tomogram_positioning_dialog(self.axis_process_panel.axis_id);
        } else if command == self.proc_ctl_final_aligned_stack.get_command() {
            application_manager.open_final_aligned_stack_dialog(self.axis_process_panel.axis_id);
        } else if command == self.proc_ctl_tomogram_generation.get_command() {
            application_manager.open_tomogram_generation_dialog(self.axis_process_panel.axis_id);
        } else if command == self.proc_ctl_tomogram_combination.get_command() {
            application_manager.open_tomogram_combination_dialog();
        } else if command == self.proc_ctl_post_processing.get_command() {
            application_manager.open_post_processing_dialog();
        } else if command == self.proc_ctl_clean_up.get_command() {
            application_manager.open_clean_up_dialog();
        }
        if self.axis_process_panel.axis_id != AxisID::Second {
            ui_harness.move_sub_frame();
        }
    }

    pub fn set_pre_proc_state(&mut self, state: ProcessState) {
        self.proc_ctl_pre_proc.set_state(state);
    }
    pub fn set_coarse_align_state(&mut self, state: ProcessState) {
        self.proc_ctl_coarse_align.set_state(state);
    }
    pub fn set_fiducial_model_state(&mut self, state: ProcessState) {
        self.proc_ctl_fiducial_model.set_state(state);
    }
    pub fn set_fine_alignment_state(&mut self, state: ProcessState) {
        self.proc_ctl_fine_alignment.set_state(state);
    }
    pub fn set_tomogram_positioning_state(&mut self, state: ProcessState) {
        self.proc_ctl_tomogram_positioning.set_state(state);
    }
    pub fn set_final_aligned_stack_state(&mut self, state: ProcessState) {
        self.proc_ctl_final_aligned_stack.set_state(state);
    }
    pub fn set_tomogram_generation_state(&mut self, state: ProcessState) {
        self.proc_ctl_tomogram_generation.set_state(state);
    }
    pub fn set_tomogram_combination_state(&mut self, state: ProcessState) {
        self.proc_ctl_tomogram_combination.set_state(state);
    }
    pub fn set_post_processing_state(&mut self, state: ProcessState) {
        self.proc_ctl_post_processing.set_state(state);
    }
    pub fn set_clean_up_state(&mut self, state: ProcessState) {
        self.proc_ctl_clean_up.set_state(state);
    }

    /// Override `showBothAxis()`.
    pub fn show_both_axis(&mut self) {
        if self.axis_process_panel.axis_id == AxisID::First {
            self.show_axis_a_private(true);
        } else if self.axis_process_panel.axis_id == AxisID::Second {
            self.show_axis_b_private(true);
        }
    }
    /// Java private `showAxisOnly`.
    pub fn show_axis_only(&mut self) {
        if *utilities::APRIL_FOOLS {
            self.set_background("rgb(163,214,247)");
        } else {
            self.set_background("rgb(173,199,224)");
        }
    }
    /// Java public/package `showAxisA`.
    pub fn show_axis_a(&mut self) {
        self.show_axis_a_private(false);
    }
    /// Java private `showAxisA(boolean)`.
    pub fn show_axis_a_private(&mut self, showing_both_axis: bool) {
        assert_eq!(
            self.axis_process_panel.axis_id,
            AxisID::First,
            "Function should only be called for A axis panel."
        );
        if *utilities::APRIL_FOOLS {
            self.set_background("rgb(163,214,247)");
        } else {
            self.set_background("rgb(173,199,224)");
        }
        if showing_both_axis {
            Self::set_button(
                &mut self.axis_button_1,
                AXIS_A_LABEL,
                self.axis_a_tooltip.clone(),
            );
            Self::set_button(
                &mut self.axis_button_2,
                AXIS_B_LABEL,
                self.axis_b_tooltip.clone(),
            );
        } else {
            Self::set_button(
                &mut self.axis_button_1,
                AXIS_B_LABEL,
                self.axis_b_tooltip.clone(),
            );
            Self::set_button(
                &mut self.axis_button_2,
                BOTH_AXIS_LABEL,
                self.both_axis_tooltip.clone(),
            );
        }
        self.axis_button_panel.visible = true;
    }
    /// Java package `showAxisB`.
    pub fn show_axis_b(&mut self) {
        self.show_axis_b_private(false);
    }
    /// Java private `showAxisB(boolean)`.
    pub fn show_axis_b_private(&mut self, showing_both_axis: bool) {
        assert_eq!(
            self.axis_process_panel.axis_id,
            AxisID::Second,
            "Function should only be called for B axis panel."
        );
        if *utilities::APRIL_FOOLS {
            self.set_background("rgb(255,216,141)");
        } else {
            self.set_background("rgb(173,224,199)");
        }
        if showing_both_axis {
            self.axis_button_panel.visible = false;
        } else {
            Self::set_button(
                &mut self.axis_button_1,
                AXIS_A_LABEL,
                self.axis_a_tooltip.clone(),
            );
            Self::set_button(
                &mut self.axis_button_2,
                BOTH_AXIS_LABEL,
                self.both_axis_tooltip.clone(),
            );
            self.axis_button_panel.visible = true;
        }
    }
    /// Override `setBackground(Color)`.
    pub fn set_background(&mut self, color: impl Into<String>) {
        let color = color.into();
        self.axis_process_panel.set_background(color.clone());
        self.axis_button_panel.background = Some(color);
    }
    /// Java private `setButton(SimpleButton, String, String)`.
    pub fn set_button(button: &mut SimpleButton, label: &str, tooltip: Option<String>) {
        button.set_text(Some(label));
        button.button.tooltip = tooltip;
    }

    /// Override `createProcessControlPanel()`.
    pub fn create_process_control_panel(&mut self, compact_display: bool) {
        self.axis_process_panel.create_process_control_panel();
        let labels = [
            AXIS_B_LABEL.to_owned(),
            AXIS_A_LABEL.to_owned(),
            BOTH_AXIS_LABEL.to_owned(),
        ];
        let index = UiUtilities::get_max_width_index(
            &self.axis_button_1.button.abstract_button,
            Some(&labels),
        );
        let widest_label = (index != -1)
            .then(|| labels[index as usize].as_str())
            .unwrap_or(AXIS_B_LABEL);
        self.axis_button_1.set_text(Some(widest_label));
        self.axis_button_1.set_to_preferred_size();
        self.axis_button_2.set_text(Some(widest_label));
        self.axis_button_2.set_to_preferred_size();
        self.set_tool_tip_text();
        self.process_select_component_order
            .push("rigid:x0_y5".into());
        self.axis_button_panel.vertical_layout = compact_display;
        if self.axis_process_panel.axis_id == AxisID::Only {
            self.show_axis_only();
        } else {
            self.axis_button_1.button.action_listener_count += 1;
            self.axis_button_2.button.action_listener_count += 1;
            self.axis_button_panel
                .component_order
                .push("axisButton1".into());
            self.axis_button_panel.component_order.push(
                if compact_display {
                    "rigid:x0_y5"
                } else {
                    "rigid:x40_y0"
                }
                .into(),
            );
            self.axis_button_panel
                .component_order
                .push("axisButton2".into());
            self.process_select_component_order
                .push("axisButtonPanel".into());
            if self.axis_process_panel.axis_id == AxisID::First {
                self.show_axis_a();
            }
        }
        self.process_select_component_order
            .push("rigid:x0_y10".into());
        self.process_select_component_order
            .push("procCtlPreProc".into());
        self.process_select_component_order
            .push("rigid:x0_y10".into());
        self.process_select_component_order
            .push("procCtlCoarseAlign".into());
        self.process_select_component_order
            .push("rigid:x0_y10".into());
        self.process_select_component_order
            .push("procCtlFiducialModel".into());
        self.process_select_component_order
            .push("rigid:x0_y10".into());
        self.process_select_component_order
            .push("procCtlFineAlignment".into());
        self.process_select_component_order
            .push("rigid:x0_y10".into());
        self.process_select_component_order
            .push("procCtlTomogramPositioning".into());
        self.process_select_component_order
            .push("rigid:x0_y10".into());
        self.process_select_component_order
            .push("procCtlFinalAlignedStack".into());
        self.process_select_component_order
            .push("rigid:x0_y10".into());
        self.process_select_component_order
            .push("procCtlTomogramGeneration".into());
        if self.axis_process_panel.axis_id == AxisID::First {
            self.process_select_component_order
                .push("rigid:x0_y10".into());
            self.process_select_component_order
                .push("procCtlTomogramCombination".into());
        }
        if self.axis_process_panel.axis_id != AxisID::Second {
            self.process_select_component_order
                .push("rigid:x0_y10".into());
            self.process_select_component_order
                .push("procCtlPostProcessing".into());
            self.process_select_component_order
                .push("rigid:x0_y10".into());
            self.process_select_component_order
                .push("procCtlCleanUp".into());
        }
        self.process_select_component_order
            .push("rigid:x0_y10".into());
        self.proc_ctl_pre_proc.set_button_action_listener();
        self.proc_ctl_pre_proc.add_mouse_listener();
        self.proc_ctl_coarse_align.set_button_action_listener();
        self.proc_ctl_coarse_align.add_mouse_listener();
        self.proc_ctl_fiducial_model.set_button_action_listener();
        self.proc_ctl_fiducial_model.add_mouse_listener();
        self.proc_ctl_fine_alignment.set_button_action_listener();
        self.proc_ctl_fine_alignment.add_mouse_listener();
        self.proc_ctl_tomogram_positioning
            .set_button_action_listener();
        self.proc_ctl_tomogram_positioning.add_mouse_listener();
        self.proc_ctl_final_aligned_stack
            .set_button_action_listener();
        self.proc_ctl_final_aligned_stack.add_mouse_listener();
        self.proc_ctl_tomogram_generation
            .set_button_action_listener();
        self.proc_ctl_tomogram_generation.add_mouse_listener();
        self.proc_ctl_tomogram_combination
            .set_button_action_listener();
        self.proc_ctl_tomogram_combination.add_mouse_listener();
        self.proc_ctl_post_processing.set_button_action_listener();
        self.proc_ctl_post_processing.add_mouse_listener();
        self.proc_ctl_clean_up.set_button_action_listener();
        self.proc_ctl_clean_up.add_mouse_listener();
    }
    /// `selectButton(String)`.
    pub fn select_button(&mut self, name: &str) {
        self.un_select_all();
        if name == self.proc_ctl_pre_proc.get_command() {
            self.proc_ctl_pre_proc.set_selected(true);
            return;
        }
        if name == self.proc_ctl_coarse_align.get_command() {
            self.proc_ctl_coarse_align.set_selected(true);
            return;
        }
        if name == self.proc_ctl_fiducial_model.get_command() {
            self.proc_ctl_fiducial_model.set_selected(true);
            return;
        }
        if name == self.proc_ctl_fine_alignment.get_command() {
            self.proc_ctl_fine_alignment.set_selected(true);
            return;
        }
        if name == self.proc_ctl_tomogram_positioning.get_command() {
            self.proc_ctl_tomogram_positioning.set_selected(true);
            return;
        }
        if name == self.proc_ctl_final_aligned_stack.get_command() {
            self.proc_ctl_final_aligned_stack.set_selected(true);
            return;
        }
        if name == self.proc_ctl_tomogram_generation.get_command() {
            self.proc_ctl_tomogram_generation.set_selected(true);
            return;
        }
        if name == self.proc_ctl_tomogram_combination.get_command() {
            self.proc_ctl_tomogram_combination.set_selected(true);
            return;
        }
        if name == self.proc_ctl_post_processing.get_command() {
            self.proc_ctl_post_processing.set_selected(true);
            return;
        }
        if name == self.proc_ctl_clean_up.get_command() {
            self.proc_ctl_clean_up.set_selected(true);
        }
    }
    /// Java private `unSelectAll`.
    pub fn un_select_all(&mut self) {
        self.proc_ctl_pre_proc.set_selected(false);
        self.proc_ctl_coarse_align.set_selected(false);
        self.proc_ctl_fiducial_model.set_selected(false);
        self.proc_ctl_fine_alignment.set_selected(false);
        self.proc_ctl_tomogram_positioning.set_selected(false);
        self.proc_ctl_final_aligned_stack.set_selected(false);
        self.proc_ctl_tomogram_generation.set_selected(false);
        self.proc_ctl_tomogram_combination.set_selected(false);
        self.proc_ctl_post_processing.set_selected(false);
        self.proc_ctl_clean_up.set_selected(false);
    }
    /// Java private `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.both_axis_tooltip = Some("See Axis A and B.".into());
        self.axis_a_tooltip = Some("See Axis A only.".into());
        self.axis_b_tooltip = Some("See Axis B only.".into());
        self.proc_ctl_pre_proc.set_tool_tip_text("Open the Pre-processing panel to erase x-rays, bad pixels and/or bad CCD rows from the raw projection stack.");
        self.proc_ctl_coarse_align.set_tool_tip_text("Open the Coarse Alignment panel to generate a coarsely aligned stack using cross correlation and to fix coarse alignment problems with Midas.");
        self.proc_ctl_fiducial_model.set_tool_tip_text("Open the Fiducial Model Generation panel to create a fiducial model to be used in the fine alignment step.");
        self.proc_ctl_fine_alignment.set_tool_tip_text("Open the Fine Alignment panel to use the generated fiducial model to sub-pixel align the project sequence.");
        self.proc_ctl_tomogram_positioning.set_tool_tip_text("Open the Tomogram Position panel to optimally adjust the 3D location and size of the reconstruction volume.");
        self.proc_ctl_final_aligned_stack.set_tool_tip_text(
            "Open the Final Aligned Stack panel to generate the final aligned stack.",
        );
        self.proc_ctl_tomogram_generation.set_tool_tip_text(
            "Open the Tomogram Generation panel to calcuate the tomographic reconstruction.",
        );
        self.proc_ctl_tomogram_combination.set_tool_tip_text("Open the Tomogram Combination panel to combine the tomograms generated from the A and B axes into a single dual axis reconstruction.");
        self.proc_ctl_post_processing.set_tool_tip_text("Open the Post Processing panel to trim the final reconstruction to size and squeeze the final reconstruction volume.");
        self.proc_ctl_clean_up
            .set_tool_tip_text("Open the Clean Up panel to delete the intermediate files.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    struct Manager {
        calls: Vec<String>,
    }
    impl TomogramProcessPanelApplicationManager for Manager {
        fn save_current_dialog(&mut self, _: AxisID) {
            self.calls.push("save".into());
        }
        fn open_pre_proc_dialog(&mut self, _: AxisID) {
            self.calls.push("pre".into());
        }
        fn open_coarse_align_dialog(&mut self, _: AxisID) {
            self.calls.push("coarse".into());
        }
        fn open_fiducial_model_dialog(&mut self, _: AxisID) {
            self.calls.push("fiducial".into());
        }
        fn open_fine_alignment_dialog(&mut self, _: AxisID) {
            self.calls.push("fine".into());
        }
        fn open_tomogram_positioning_dialog(&mut self, _: AxisID) {
            self.calls.push("position".into());
        }
        fn open_final_aligned_stack_dialog(&mut self, _: AxisID) {
            self.calls.push("stack".into());
        }
        fn open_tomogram_generation_dialog(&mut self, _: AxisID) {
            self.calls.push("generation".into());
        }
        fn open_tomogram_combination_dialog(&mut self) {
            self.calls.push("combination".into());
        }
        fn open_post_processing_dialog(&mut self) {
            self.calls.push("post".into());
        }
        fn open_clean_up_dialog(&mut self) {
            self.calls.push("cleanup".into());
        }
    }
    #[derive(Default)]
    struct Ui {
        calls: Vec<String>,
    }
    impl TomogramProcessPanelUiHarness for Ui {
        fn show_both_axis(&mut self) {
            self.calls.push("both".into());
        }
        fn show_axis_a(&mut self) {
            self.calls.push("a".into());
        }
        fn show_axis_b(&mut self) {
            self.calls.push("b".into());
        }
        fn move_sub_frame(&mut self) {
            self.calls.push("move".into());
        }
    }
    fn panel(axis: AxisID) -> TomogramProcessPanel {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        TomogramProcessPanel::new(
            manager,
            axis,
            AxisProgressPanel::get_instance(Some(axis), manager),
            false,
        )
    }
    #[test]
    fn first_axis_has_source_controls_and_switch_labels() {
        let mut p = panel(AxisID::First);
        assert!(
            p.process_select_component_order
                .contains(&"procCtlTomogramCombination".into())
        );
        assert_eq!(p.axis_button_1.button.text.as_deref(), Some(AXIS_B_LABEL));
        p.show_both_axis();
        assert_eq!(p.axis_button_1.button.text.as_deref(), Some(AXIS_A_LABEL));
    }
    #[test]
    fn second_axis_hides_buttons_when_both_axes_show() {
        let mut p = panel(AxisID::Second);
        p.show_both_axis();
        assert!(!p.axis_button_panel.visible);
        assert!(
            !p.process_select_component_order
                .contains(&"procCtlPostProcessing".into())
        );
    }
    #[test]
    fn selection_is_exclusive() {
        let mut p = panel(AxisID::Only);
        let command = p.proc_ctl_fine_alignment.get_command();
        p.select_button(&command);
        assert!(p.proc_ctl_fine_alignment.button_run.button.selected);
        assert!(!p.proc_ctl_pre_proc.button_run.button.selected);
    }
    #[test]
    fn process_dispatch_saves_then_opens_and_moves_non_b_axis_frame() {
        let mut p = panel(AxisID::First);
        let command = p.proc_ctl_tomogram_generation.get_command();
        let mut manager = Manager { calls: vec![] };
        let mut ui = Ui::default();
        p.button_process_action(&command, &mut manager, &mut ui);
        assert_eq!(manager.calls, ["save", "generation"]);
        assert_eq!(ui.calls, ["move"]);
    }
    #[test]
    fn axis_button_dispatch_uses_source_commands() {
        let mut p = panel(AxisID::Only);
        let mut ui = Ui::default();
        p.button_axis_action(BOTH_AXIS_LABEL, &mut ui);
        p.button_axis_action(AXIS_B_LABEL, &mut ui);
        assert_eq!(ui.calls, ["both", "b"]);
    }
}
