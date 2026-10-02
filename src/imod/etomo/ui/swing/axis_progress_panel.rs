//! `IMOD/Etomo/src/etomo/ui/swing/AxisProgressPanel.java`.
//!
//! The progress panel of one axis plus its "Kill Process" button.  The
//! button's `ActionListener` is this class itself (`actionPerformed` kills the
//! axis's process).

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use super::progress_panel::ProgressPanel;
use super::simple_button::SimpleButton;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::logic::busy_status_mediator::{self, BusyStatusMediator};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;

/// Java `KILL_BUTTON_LABEL`.
pub const KILL_BUTTON_LABEL: &str = "Kill Process";

/// Java public final class `AxisProgressPanel implements ActionListener`.
pub struct AxisProgressPanel {
    /// Java `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java `buttonKillProcess = new SimpleButton(KILL_BUTTON_LABEL)`.
    button_kill_process: Rc<SimpleButton>,
    manager: &'static dyn BaseManager,
    progress_panel: Rc<ProgressPanel>,
    busy_status_mediator: Arc<BusyStatusMediator>,
    /// Java `axisID`, initialised to `AxisID.ONLY`.
    axis_id: Cell<AxisID>,
    /// Java `processCanBeKilled`, initialised to true.
    process_can_be_killed: Cell<bool>,
}

impl AxisProgressPanel {
    /// Java private constructor `AxisProgressPanel(AxisID, BaseManager)`.
    fn new(axis_id: Option<AxisID>, manager: &'static dyn BaseManager) -> AxisProgressPanel {
        let this_axis_id = Cell::new(AxisID::Only);
        if let Some(axis_id) = axis_id {
            this_axis_id.set(axis_id);
        }
        let progress_panel = ProgressPanel::get_instance(Some("No process"), manager, axis_id);
        let busy_status_mediator = manager.get_busy_status_mediator();
        AxisProgressPanel {
            pnl_root: JComponent::new_panel(),
            button_kill_process: SimpleButton::new_string(Some(KILL_BUTTON_LABEL)),
            manager,
            progress_panel,
            busy_status_mediator,
            axis_id: this_axis_id,
            process_can_be_killed: Cell::new(true),
        }
    }

    /// Java static `getInstance(AxisID, BaseManager)`.
    pub fn get_instance(
        axis_id: Option<AxisID>,
        manager: &'static dyn BaseManager,
    ) -> Rc<AxisProgressPanel> {
        let instance = Rc::new(AxisProgressPanel::new(axis_id, manager));
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java `correctAxisID(AxisType)`.
    pub fn correct_axis_id(&self, axis_type: AxisType) {
        if self.axis_id.get() == AxisID::Second {
            // Always correct.
            return;
        }
        if axis_type == AxisType::DualAxis {
            self.axis_id.set(AxisID::First);
        } else {
            self.axis_id.set(AxisID::Only);
        }
    }

    /// Java `isStopped()`.
    pub fn is_stopped(&self) -> bool {
        self.progress_panel.is_stopped()
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        self.button_kill_process.get_component().set_enabled(false);
        self.busy_status_mediator
            .msg_kill_process_button(self.axis_id.get(), false);
        // Swing layout: buttonKillProcess.setAlignmentY(Component.BOTTOM_ALIGNMENT).
        // root
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.X_AXIS));
        // pnlRoot.add(Box.createRigidArea(FixedDim.x5_y0)).
        self.pnl_root.add(&self.progress_panel.get_container());
        // Swing layout: pnlRoot.add(Box.createRigidArea(FixedDim.x5_y0)).
        self.pnl_root.add(&self.button_kill_process.get_component());
        // Swing layout: pnlRoot.add(Box.createRigidArea(FixedDim.x0_y5)).
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, input: bool) {
        self.pnl_root.set_visible(input);
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        self.button_kill_process
            .get_component()
            .set_tool_tip_text(Some("Press to end the current process."));
    }

    /// Java private `addListeners()`: `buttonKillProcess.addActionListener(this)`.
    fn add_listeners(self: &Rc<Self>) {
        let weak = Rc::downgrade(self);
        self.button_kill_process
            .get_component()
            .add_action_listener(Rc::new(move |event: &ActionEvent| {
                if let Some(this) = weak.upgrade() {
                    this.action_performed(event);
                }
            }));
    }

    // Java `setBackground(Color)`: pnlRoot.setBackground(color);
    // progressPanel.setBackground(color) - Swing painting, not modelled.

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        self.manager.kill(Some(self.axis_id.get()));
    }

    /// Java `setProcessCanBeKilled(boolean)`.
    pub fn set_process_can_be_killed(&self, input: bool) {
        self.process_can_be_killed.set(input);
    }

    /// Java private `setButtonKillProcessEnabled(boolean)`.
    fn set_button_kill_process_enabled(&self, enabled: bool) {
        // Java `synchronized (busyStatusMediator) { ... }`: holds the mediator's
        // monitor, which its own synchronized methods also take, across the
        // whole block.
        let _monitor = self.busy_status_mediator.synchronized();
        if enabled {
            self.busy_status_mediator
                .msg_kill_process_button(self.axis_id.get(), enabled);
        }
        if !enabled {
            std::thread::sleep(std::time::Duration::from_millis(
                busy_status_mediator::DELAY,
            ));
        }
        self.button_kill_process
            .get_component()
            .set_enabled(enabled);
        if !enabled {
            self.busy_status_mediator
                .msg_kill_process_button(self.axis_id.get(), enabled);
        }
    }

    /// Java `setProgressBar(String, int, boolean)`.  Setup the progress bar for
    /// a determinate.
    pub fn set_progress_bar_string_int_boolean(
        &self,
        label: Option<&str>,
        n_steps: i32,
        indeterminate_mode: bool,
    ) {
        self.progress_panel.set_label(label);
        self.progress_panel.set_minimum(0);
        self.progress_panel.set_maximum(n_steps, indeterminate_mode);
        self.set_button_kill_process_enabled(self.process_can_be_killed.get());
    }

    /// Java `setStaticProgressBar(String)`.  Setup the progress bar for a
    /// state, not a process (with no moving bar, or percentage, and without the
    /// kill button being enabled.
    pub fn set_static_progress_bar(&self, label: Option<&str>) {
        self.progress_panel.set_label(label);
    }

    /// Java `setProgressBarValue(int)`.
    pub fn set_progress_bar_value_int(&self, n: i32) {
        self.progress_panel.set_value_int(n);
    }

    /// Java `setProgressBarValue(int, StandardBarString, String, boolean)`.
    pub fn set_progress_bar_value_int_standard_bar_string_string_boolean(
        &self,
        n: i32,
        standard_bar_string: Option<StandardBarString>,
        bar_string: Option<&str>,
        print: bool,
    ) {
        self.progress_panel
            .set_value_int_standard_bar_string_string_boolean(
                n,
                standard_bar_string,
                bar_string,
                print,
            );
    }

    /// Java `setEmergencyMonitorBarString(StandardBarString, String)`.
    pub fn set_emergency_monitor_bar_string(
        &self,
        standard_bar_string: Option<StandardBarString>,
        bar_string: Option<&str>,
    ) {
        self.progress_panel
            .set_emergency_monitor_bar_string(standard_bar_string, bar_string);
    }

    /// Java `isProgressBarStopped(AxisID)`.
    pub fn is_progress_bar_stopped(&self, _axis_id: AxisID) -> bool {
        self.progress_panel.is_progress_bar_stopped()
    }

    /// Java `startProgressBar(String, ProcessName)`.
    pub fn start_progress_bar_string_process_name(
        &self,
        label: Option<&str>,
        _process_name: Option<&ProcessName>,
    ) {
        self.progress_panel.set_label(label);
        self.progress_panel.start();
        self.set_button_kill_process_enabled(self.process_can_be_killed.get());
    }

    /// Java `startProgressBar(int, String, boolean)`.
    pub fn start_progress_bar_int_string_boolean(
        &self,
        n: i32,
        bar_string: Option<&str>,
        print: bool,
    ) {
        self.progress_panel.start_indeterminate_mode();
        self.progress_panel
            .set_value_int_standard_bar_string_string_boolean(n, None, bar_string, print);
        self.set_button_kill_process_enabled(self.process_can_be_killed.get());
    }

    /// Java `setProgressBar(String, boolean)`.
    pub fn set_progress_bar_string_boolean(&self, label: Option<&str>, indeterminate_mode: bool) {
        self.progress_panel.set_label(label);
        if indeterminate_mode {
            self.progress_panel.start_indeterminate_mode();
        }
    }

    /// Java `stopProgressBar(ProcessEndState, String)`.
    pub fn stop_progress_bar(
        &self,
        process_end_state: Option<ProcessEndState>,
        status_string: Option<&str>,
    ) {
        self.progress_panel.stop(process_end_state, status_string);
        self.process_can_be_killed.set(true);
        self.set_button_kill_process_enabled(false);
    }
}
