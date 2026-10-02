//! `IMOD/Etomo/src/etomo/ui/swing/TomogramProcessPanel.java`.
//!
//! The axis process panel of the reconstruction interface: the column of
//! process buttons (Pre-processing ... Clean Up) and, for a dual axis data
//! set, the axis buttons ("Axis A", "Axis B", "Both").
//!
//! Extends [`AxisProcessPanel`] (held as `base`, dereffed to) and overrides
//! `showBothAxis`, `setBackground` and `createProcessControlPanel` through
//! [`AxisProcessPanelVirtual`].

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::axis_process_panel::{AxisProcessPanel, AxisProcessPanelVirtual};
use super::axis_progress_panel::AxisProgressPanel;
use super::context_menu::ContextMenu;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::process_control_panel::ProcessControlPanel;
use super::simple_button::SimpleButton;
use super::tooltip_formatter;
use super::ui_expert::UIExpert;
use super::ui_harness;
use super::ui_utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusMediator;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::util::utilities;

/// Java `BOTH_AXIS_LABEL`.
pub const BOTH_AXIS_LABEL: &str = "Both";
/// Java package-private `AXIS_B_LABEL`.
pub const AXIS_B_LABEL: &str = "Axis B";
/// Java private `AXIS_A_LABEL`.
const AXIS_A_LABEL: &str = "Axis A";

/// Java final class `TomogramProcessPanel extends AxisProcessPanel`.
pub struct TomogramProcessPanel {
    /// The Java superclass part.
    base: Rc<AxisProcessPanel>,
    /// This panel's own handle (Java `this`, for the listeners).
    self_ref: Weak<TomogramProcessPanel>,

    proc_ctl_pre_proc: Rc<ProcessControlPanel>,
    proc_ctl_coarse_align: Rc<ProcessControlPanel>,
    proc_ctl_fiducial_model: Rc<ProcessControlPanel>,
    proc_ctl_fine_alignment: Rc<ProcessControlPanel>,
    proc_ctl_tomogram_positioning: Rc<ProcessControlPanel>,
    proc_ctl_final_aligned_stack: Rc<ProcessControlPanel>,
    proc_ctl_tomogram_generation: Rc<ProcessControlPanel>,
    proc_ctl_tomogram_combination: Rc<ProcessControlPanel>,
    proc_ctl_post_processing: Rc<ProcessControlPanel>,
    proc_ctl_clean_up: Rc<ProcessControlPanel>,
    /// Java `axisButton1 = new SimpleButton()`.
    axis_button1: Rc<SimpleButton>,
    /// Java `axisButton2 = new SimpleButton()`.
    axis_button2: Rc<SimpleButton>,
    /// Java `axisButtonPanel = new JPanel()`.
    axis_button_panel: Rc<JComponent>,

    both_axis_tooltip: RefCell<Option<String>>,
    axis_a_tooltip: RefCell<Option<String>>,
    axis_b_tooltip: RefCell<Option<String>>,

    // Java `private final UIHarness uiHarness = UIHarness.INSTANCE;` - the
    // translation reaches the thread-local `ui_harness::INSTANCE` at each use.
    application_manager: &'static ApplicationManager,
    #[allow(dead_code)]
    busy_status_mediator: Arc<BusyStatusMediator>,
}

impl Deref for TomogramProcessPanel {
    type Target = AxisProcessPanel;
    fn deref(&self) -> &AxisProcessPanel {
        &self.base
    }
}

impl TomogramProcessPanel {
    /// Java constructor `TomogramProcessPanel(ApplicationManager, AxisID,
    /// AxisProgressPanel)`.
    pub fn new(
        app_manager: &'static ApplicationManager,
        axis: AxisID,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<TomogramProcessPanel> {
        let this = Rc::new_cyclic(|self_ref| {
            let base = AxisProcessPanel::new(
                axis,
                app_manager,
                true,
                true,
                InterfaceType::Recon,
                false,
                axis_progress_panel,
            );
            TomogramProcessPanel {
                base,
                self_ref: self_ref.clone(),
                proc_ctl_pre_proc: ProcessControlPanel::new(DialogType::PreProcessing),
                proc_ctl_coarse_align: ProcessControlPanel::new(DialogType::CoarseAlignment),
                proc_ctl_fiducial_model: ProcessControlPanel::new(DialogType::FiducialModel),
                proc_ctl_fine_alignment: ProcessControlPanel::new(DialogType::FineAlignment),
                proc_ctl_tomogram_positioning: ProcessControlPanel::new(
                    DialogType::TomogramPositioning,
                ),
                proc_ctl_final_aligned_stack: ProcessControlPanel::new(
                    DialogType::FinalAlignedStack,
                ),
                proc_ctl_tomogram_generation: ProcessControlPanel::new(
                    DialogType::TomogramGeneration,
                ),
                proc_ctl_tomogram_combination: ProcessControlPanel::new(
                    DialogType::TomogramCombination,
                ),
                proc_ctl_post_processing: ProcessControlPanel::new(DialogType::PostProcessing),
                proc_ctl_clean_up: ProcessControlPanel::new(DialogType::CleanUp),
                axis_button1: SimpleButton::new_void(),
                axis_button2: SimpleButton::new_void(),
                axis_button_panel: JComponent::new_panel(),
                both_axis_tooltip: RefCell::new(None),
                axis_a_tooltip: RefCell::new(None),
                axis_b_tooltip: RefCell::new(None),
                // Java `applicationManager = (ApplicationManager) manager`.
                application_manager: app_manager,
                busy_status_mediator: app_manager.get_busy_status_mediator(),
            }
        });
        this.base
            .set_this(Rc::downgrade(&this) as Weak<dyn AxisProcessPanelVirtual>);
        // Create the process control panel
        this.create_process_control_panel();
        this.base.initialize_panels();
        this
    }

    /// Java `buttonAxisAction(ActionEvent)`.
    pub fn button_axis_action(&self, event: &ActionEvent) {
        let command = event.get_action_command().unwrap_or("").to_owned();
        if command == BOTH_AXIS_LABEL {
            ui_harness::INSTANCE.with(|harness| harness.show_both_axis());
        } else if command == AXIS_A_LABEL {
            ui_harness::INSTANCE.with(|harness| harness.show_axis_a());
        } else if command == AXIS_B_LABEL {
            ui_harness::INSTANCE.with(|harness| harness.show_axis_b());
        }
    }

    /// Java `buttonProcessAction(ActionEvent)`.  Invoke the appropriate
    /// ApplicationManager method for the button press.
    pub fn button_process_action(&self, event: &ActionEvent) {
        let command = event.get_action_command().unwrap_or("").to_owned();
        utilities::button_timestamp(Some(&command));
        let axis_id = self.base.axis_id;
        self.application_manager.save_current_dialog(axis_id);
        if command == self.proc_ctl_pre_proc.get_command() {
            self.application_manager.open_pre_proc_dialog(axis_id);
        } else if command == self.proc_ctl_coarse_align.get_command() {
            self.application_manager.open_coarse_align_dialog(axis_id);
        } else if command == self.proc_ctl_fiducial_model.get_command() {
            self.application_manager.open_fiducial_model_dialog(axis_id);
        } else if command == self.proc_ctl_fine_alignment.get_command() {
            self.application_manager.open_fine_alignment_dialog(axis_id);
        } else if command == self.proc_ctl_tomogram_positioning.get_command() {
            // Java casts to TomogramPositioningExpert; openDialog is declared by
            // the UIExpert interface.
            if let Some(expert) = self
                .application_manager
                .get_ui_expert(Some(DialogType::TomogramPositioning), axis_id)
            {
                expert.open_dialog();
            }
        } else if command == self.proc_ctl_final_aligned_stack.get_command() {
            if let Some(expert) = self
                .application_manager
                .get_ui_expert(Some(DialogType::FinalAlignedStack), axis_id)
            {
                expert.open_dialog();
            }
        } else if command == self.proc_ctl_tomogram_generation.get_command() {
            if let Some(expert) = self
                .application_manager
                .get_ui_expert(Some(DialogType::TomogramGeneration), axis_id)
            {
                expert.open_dialog();
            }
        } else if command == self.proc_ctl_tomogram_combination.get_command() {
            self.application_manager.open_tomogram_combination_dialog();
        } else if command == self.proc_ctl_post_processing.get_command() {
            self.application_manager.open_post_processing_dialog();
        } else if command == self.proc_ctl_clean_up.get_command() {
            self.application_manager.open_clean_up_dialog();
        }
        if axis_id != AxisID::Second {
            ui_harness::INSTANCE.with(|harness| harness.move_sub_frame());
        }
    }

    /// Java `setPreProcState(ProcessState)`.  Pre-processing panel state control
    pub fn set_pre_proc_state(&self, state: ProcessState) {
        self.proc_ctl_pre_proc.set_state(state);
    }

    /// Java `setCoarseAlignState(ProcessState)`.
    pub fn set_coarse_align_state(&self, state: ProcessState) {
        self.proc_ctl_coarse_align.set_state(state);
    }

    /// Java `setFiducialModelState(ProcessState)`.
    pub fn set_fiducial_model_state(&self, state: ProcessState) {
        self.proc_ctl_fiducial_model.set_state(state);
    }

    /// Java `setFineAlignmentState(ProcessState)`.
    pub fn set_fine_alignment_state(&self, state: ProcessState) {
        self.proc_ctl_fine_alignment.set_state(state);
    }

    /// Java `setTomogramPositioningState(ProcessState)`.
    pub fn set_tomogram_positioning_state(&self, state: ProcessState) {
        self.proc_ctl_tomogram_positioning.set_state(state);
    }

    /// Java `setFinalAlignedStackState(ProcessState)`.
    pub fn set_final_aligned_stack_state(&self, state: ProcessState) {
        self.proc_ctl_final_aligned_stack.set_state(state);
    }

    /// Java `setTomogramGenerationState(ProcessState)`.
    pub fn set_tomogram_generation_state(&self, state: ProcessState) {
        self.proc_ctl_tomogram_generation.set_state(state);
    }

    /// Java `setTomogramCombinationState(ProcessState)`.
    pub fn set_tomogram_combination_state(&self, state: ProcessState) {
        self.proc_ctl_tomogram_combination.set_state(state);
    }

    /// Java `setPostProcessingState(ProcessState)`.
    pub fn set_post_processing_state(&self, state: ProcessState) {
        self.proc_ctl_post_processing.set_state(state);
    }

    /// Java `setCleanUpState(ProcessState)`.
    pub fn set_clean_up_state(&self, state: ProcessState) {
        self.proc_ctl_clean_up.set_state(state);
    }

    /// Java private `showAxisOnly()`.
    fn show_axis_only(&self) {
        // Swing painting: setBackground(Colors.getBackgroundA()).
    }

    /// Java private `showAxisA(boolean)`.
    fn show_axis_a_boolean(&self, showing_both_axis: bool) {
        if self.base.axis_id != AxisID::First {
            // Java `throw new IllegalStateException(...)`, an unchecked
            // exception that no caller catches.
            panic!("Function should only be called for A axis panel.");
        }
        // Swing painting: setBackground(Colors.getBackgroundA()).
        if showing_both_axis {
            self.set_button(
                &self.axis_button1,
                AXIS_A_LABEL,
                self.axis_a_tooltip.borrow().clone(),
            );
            self.set_button(
                &self.axis_button2,
                AXIS_B_LABEL,
                self.axis_b_tooltip.borrow().clone(),
            );
        } else {
            self.set_button(
                &self.axis_button1,
                AXIS_B_LABEL,
                self.axis_b_tooltip.borrow().clone(),
            );
            self.set_button(
                &self.axis_button2,
                BOTH_AXIS_LABEL,
                self.both_axis_tooltip.borrow().clone(),
            );
        }
        self.axis_button_panel.set_visible(true);
    }

    /// Java `showAxisA()`.
    pub fn show_axis_a_void(&self) {
        self.show_axis_a_boolean(false);
    }

    /// Java private `showAxisB(boolean)`.
    fn show_axis_b_boolean(&self, showing_both_axis: bool) {
        if self.base.axis_id != AxisID::Second {
            // Java `throw new IllegalStateException(...)`.
            panic!("Function should only be called for B axis panel.");
        }
        // Swing painting: setBackground(Colors.getBackgroundB()).
        if showing_both_axis {
            self.axis_button_panel.set_visible(false);
        } else {
            self.set_button(
                &self.axis_button1,
                AXIS_A_LABEL,
                self.axis_a_tooltip.borrow().clone(),
            );
            self.set_button(
                &self.axis_button2,
                BOTH_AXIS_LABEL,
                self.both_axis_tooltip.borrow().clone(),
            );
            self.axis_button_panel.set_visible(true);
        }
    }

    /// Java `showAxisB()`.
    pub fn show_axis_b_void(&self) {
        self.show_axis_b_boolean(false);
    }

    // Java `setBackground(Color)` override: super.setBackground(color);
    // axisButtonPanel.setBackground(color) - Swing painting, not modelled.

    /// Java private `setButton(SimpleButton, String, String)`.
    fn set_button(&self, button: &Rc<SimpleButton>, label: &str, tooltip: Option<String>) {
        button.set_text(Some(label));
        button.get_component().set_tool_tip_text(tooltip.as_deref());
    }

    /// Java `selectButton(String)`.  Select the requested button.
    pub fn select_button(&self, name: &str) {
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

    /// Java private `unSelectAll()`.
    fn un_select_all(&self) {
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

    /// Java private `setToolTipText()`.  Initialize the tooltip text for the
    /// axis panel objects.
    fn set_tool_tip_text(&self) {
        *self.both_axis_tooltip.borrow_mut() =
            tooltip_formatter::INSTANCE.format(Some("See Axis A and B."));
        *self.axis_a_tooltip.borrow_mut() =
            tooltip_formatter::INSTANCE.format(Some("See Axis A only."));
        *self.axis_b_tooltip.borrow_mut() =
            tooltip_formatter::INSTANCE.format(Some("See Axis B only."));
        self.proc_ctl_pre_proc.set_tool_tip_text(
            "Open the Pre-processing panel to erase x-rays, bad pixels and/or bad CCD rows from the raw projection stack.",
        );
        self.proc_ctl_coarse_align.set_tool_tip_text(
            "Open the Coarse Alignment panel to generate a coarsely aligned stack using cross correlation and to fix coarse alignment problems with Midas.",
        );
        self.proc_ctl_fiducial_model.set_tool_tip_text(
            "Open the Fiducial Model Generation panel to create a fiducial model to be used in the fine alignment step.",
        );
        self.proc_ctl_fine_alignment.set_tool_tip_text(
            "Open the Fine Alignment panel to use the generated fiducial model to sub-pixel align the project sequence.",
        );
        self.proc_ctl_tomogram_positioning.set_tool_tip_text(
            "Open the Tomogram Position panel to optimally adjust the 3D location and size of the reconstruction volume.",
        );
        self.proc_ctl_final_aligned_stack.set_tool_tip_text(
            "Open the Final Aligned Stack panel to generate the final aligned stack.",
        );
        self.proc_ctl_tomogram_generation.set_tool_tip_text(
            "Open the Tomogram Generation panel to calcuate the tomographic reconstruction.",
        );
        self.proc_ctl_tomogram_combination.set_tool_tip_text(
            "Open the Tomogram Combination panel to combine the tomograms generated from the A and B axes into a single dual axis reconstruction.",
        );
        self.proc_ctl_post_processing.set_tool_tip_text(
            "Open the Post Processing panel to trim the final reconstruction to size and squeeze the final reconstruction volume.",
        );
        self.proc_ctl_clean_up
            .set_tool_tip_text("Open the Clean Up panel to delete the intermediate files.");
    }
}

impl AxisProcessPanelVirtual for TomogramProcessPanel {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        &self.base
    }

    /// Java `showBothAxis()` override.
    fn show_both_axis(&self) {
        if self.base.axis_id == AxisID::First {
            self.show_axis_a_boolean(true);
        } else if self.base.axis_id == AxisID::Second {
            self.show_axis_b_boolean(true);
        }
    }

    /// Java `createProcessControlPanel()` override.
    fn create_process_control_panel(&self) {
        self.base.create_process_control_panel_super();
        // Init
        // Make sure axis buttons have a stable size and always show their whole label.
        let axis_label_array: [Option<String>; 3] = [
            Some(AXIS_B_LABEL.to_owned()),
            Some(AXIS_A_LABEL.to_owned()),
            Some(BOTH_AXIS_LABEL.to_owned()),
        ];
        let index = ui_utilities::get_max_width_index(
            &self.axis_button1.get_component(),
            Some(&axis_label_array),
        );
        let mut widest_label = AXIS_B_LABEL;
        if index != -1 {
            widest_label = axis_label_array[index as usize].as_deref().unwrap();
        }
        self.axis_button1.set_text(Some(widest_label));
        // Swing layout: axisButton1.setToPreferredSize() (preferred and maximum
        // size set to the preferred size).
        self.axis_button2.set_text(Some(widest_label));
        // Swing layout: axisButton2.setToPreferredSize().
        // Bind each button to action listener and the generic mouse listener
        let mouse_adapter =
            GenericMouseAdapter::new(Rc::downgrade(&self.base) as Weak<dyn ContextMenu>);
        // Java `new ProcessButtonActionListener(this)`.
        let weak = self.self_ref.clone();
        let button_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = weak.upgrade() {
                adaptee.button_process_action(event);
            }
        });
        // Java `new AxisButtonActionListener(this)`.
        let weak = self.self_ref.clone();
        let axis_button_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = weak.upgrade() {
                adaptee.button_axis_action(event);
            }
        });
        self.set_tool_tip_text();
        // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y5)).
        let compact_display =
            etomo_director::INSTANCE.with_user_configuration(|c| c.get_compact_display());
        // Swing layout: axisButtonPanel.setLayout(new BoxLayout(axisButtonPanel,
        // compactDisplay ? BoxLayout.Y_AXIS : BoxLayout.X_AXIS)).
        let _ = compact_display;
        if self.base.axis_id == AxisID::Only {
            self.show_axis_only();
        } else {
            self.axis_button1
                .get_component()
                .add_action_listener(axis_button_listener.clone());
            self.axis_button2
                .get_component()
                .add_action_listener(axis_button_listener);
            self.axis_button_panel
                .add(&self.axis_button1.get_component());
            // Swing layout: axisButtonPanel.add(Box.createRigidArea(compactDisplay
            // ? FixedDim.x0_y5 : FixedDim.x40_y0)).
            self.axis_button_panel
                .add(&self.axis_button2.get_component());
            // Swing layout: axisButtonPanel.setAlignmentX(Container.CENTER_ALIGNMENT).
            self.base.panel_process_select.add(&self.axis_button_panel);
            if self.base.axis_id == AxisID::First {
                self.show_axis_a_void();
            }
        }
        // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
        self.proc_ctl_pre_proc
            .set_button_action_listener(button_listener.clone());
        self.proc_ctl_pre_proc.add_mouse_listener(&mouse_adapter);
        // Swing layout: proc_ctl_pre_proc.getContainer().setAlignmentX(Container.CENTER_ALIGNMENT).
        self.base
            .panel_process_select
            .add(&self.proc_ctl_pre_proc.get_container());

        // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
        self.proc_ctl_coarse_align
            .add_mouse_listener(&mouse_adapter);
        self.proc_ctl_coarse_align
            .set_button_action_listener(button_listener.clone());
        // Swing layout: proc_ctl_coarse_align.getContainer().setAlignmentX(Container.CENTER_ALIGNMENT).
        self.base
            .panel_process_select
            .add(&self.proc_ctl_coarse_align.get_container());

        // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
        self.proc_ctl_fiducial_model
            .add_mouse_listener(&mouse_adapter);
        self.proc_ctl_fiducial_model
            .set_button_action_listener(button_listener.clone());
        // Swing layout: proc_ctl_fiducial_model.getContainer().setAlignmentX(Container.CENTER_ALIGNMENT).
        self.base
            .panel_process_select
            .add(&self.proc_ctl_fiducial_model.get_container());

        // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
        self.proc_ctl_fine_alignment
            .add_mouse_listener(&mouse_adapter);
        self.proc_ctl_fine_alignment
            .set_button_action_listener(button_listener.clone());
        // Swing layout: proc_ctl_fine_alignment.getContainer().setAlignmentX(Container.CENTER_ALIGNMENT).
        self.base
            .panel_process_select
            .add(&self.proc_ctl_fine_alignment.get_container());

        // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
        self.proc_ctl_tomogram_positioning
            .add_mouse_listener(&mouse_adapter);
        self.proc_ctl_tomogram_positioning
            .set_button_action_listener(button_listener.clone());
        // Swing layout: proc_ctl_tomogram_positioning.getContainer().setAlignmentX(Container.CENTER_ALIGNMENT).
        self.base
            .panel_process_select
            .add(&self.proc_ctl_tomogram_positioning.get_container());

        // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
        self.proc_ctl_final_aligned_stack
            .add_mouse_listener(&mouse_adapter);
        self.proc_ctl_final_aligned_stack
            .set_button_action_listener(button_listener.clone());
        // Swing layout: proc_ctl_final_aligned_stack.getContainer().setAlignmentX(Container.CENTER_ALIGNMENT).
        self.base
            .panel_process_select
            .add(&self.proc_ctl_final_aligned_stack.get_container());

        // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
        self.proc_ctl_tomogram_generation
            .add_mouse_listener(&mouse_adapter);
        self.proc_ctl_tomogram_generation
            .set_button_action_listener(button_listener.clone());
        // Swing layout: proc_ctl_tomogram_generation.getContainer().setAlignmentX(Container.CENTER_ALIGNMENT).
        self.base
            .panel_process_select
            .add(&self.proc_ctl_tomogram_generation.get_container());

        if self.base.axis_id == AxisID::First {
            // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
            self.proc_ctl_tomogram_combination
                .add_mouse_listener(&mouse_adapter);
            self.proc_ctl_tomogram_combination
                .set_button_action_listener(button_listener.clone());
            // Swing layout: proc_ctl_tomogram_combination.getContainer().setAlignmentX(Container.CENTER_ALIGNMENT).
            self.base
                .panel_process_select
                .add(&self.proc_ctl_tomogram_combination.get_container());
        }
        if self.base.axis_id != AxisID::Second {
            // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
            self.proc_ctl_post_processing
                .add_mouse_listener(&mouse_adapter);
            self.proc_ctl_post_processing
                .set_button_action_listener(button_listener.clone());
            // Swing layout: proc_ctl_post_processing.getContainer().setAlignmentX(Container.CENTER_ALIGNMENT).
            self.base
                .panel_process_select
                .add(&self.proc_ctl_post_processing.get_container());

            // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
            self.proc_ctl_clean_up.add_mouse_listener(&mouse_adapter);
            self.proc_ctl_clean_up
                .set_button_action_listener(button_listener.clone());
            // Swing layout: proc_ctl_clean_up.getContainer().setAlignmentX(Container.CENTER_ALIGNMENT).
            self.base
                .panel_process_select
                .add(&self.proc_ctl_clean_up.get_container());
        }
        // Swing layout: panelProcessSelect.add(Box.createRigidArea(FixedDim.x0_y10)).
    }
}
