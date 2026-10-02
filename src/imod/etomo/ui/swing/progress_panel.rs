//! `IMOD/Etomo/src/etomo/ui/swing/ProgressPanel.java`.
//!
//! The task label and progress bar of one axis.  Every public mutator records
//! the new value and posts the Swing update to the event dispatch thread
//! (`SwingUtilities.invokeLater`), as the Java does; the posted jobs are the
//! Java's `Runnable` inner classes, translated as the private `*_later`
//! functions below.  A job carries the panel as an [`EdtRef`], which is
//! `Send` but only dereferenced on the event dispatch thread.
//!
//! A `javax.swing.Timer` (one second) drives the elapsed-time display while
//! the bar is indeterminate.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use super::ui_harness;
use super::ui_utilities;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{JComponent, Timer};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;
use crate::imod::etomo::util::event_queue::{self, EdtRef};
use crate::imod::etomo::util::utilities;

/// Java `NAME`.
pub const NAME: &str = "the-progress-bar";
/// Java `LABEL_NAME = NAME + "-label"`.
pub const LABEL_NAME: &str = "the-progress-bar-label";
/// Java private `MAX_PACK`.
const MAX_PACK: i32 = 5;

/// Java public final class `ProgressPanel`.
pub struct ProgressPanel {
    /// Java `panel = new JPanel()`.
    panel: Rc<JComponent>,
    /// Java `progressPanel = new JPanel()`: declared, never used by the Java.
    #[allow(dead_code)]
    progress_panel: Rc<JComponent>,
    /// Java `taskLabel = new JLabel()`.
    task_label: Rc<JComponent>,
    /// Java `progressBar = new JProgressBar()`.
    progress_bar: Rc<JComponent>,
    manager: &'static dyn BaseManager,
    /// Java `axisID`; the Java constructor accepts null.
    axis_id: Option<AxisID>,

    // Keep these around so that SwingUtilities.invokeLater can update the
    // the UI status
    counter: Cell<i32>,
    value: Cell<i32>,
    maximum: Cell<i32>,
    minimum: Cell<i32>,
    start_time: Cell<i64>,
    cur_standard_bar_string: Cell<Option<StandardBarString>>,
    bar_string: RefCell<Option<String>>,
    emergency_monitor_alert: Cell<bool>,
    label: RefCell<Option<String>>,
    // stopped: IMPORTANT: The stop action should turn this boolean on, all other
    // actions, except increment should turn this off.
    stopped: Cell<bool>,
    n_packed: Cell<i32>,
    /// Java `debugLevel = EtomoDirector.INSTANCE.getArguments().getDebugLevel()`.
    #[allow(dead_code)]
    debug_level: DebugLevel,

    // required - instantiate once
    progress_timer_action_listener: RefCell<Option<Rc<ProgressTimerActionListener>>>,
    timer: RefCell<Option<Rc<Timer>>>,
}

impl ProgressPanel {
    /// Java private constructor `ProgressPanel(String, BaseManager, AxisID)`.
    fn new(
        new_label: Option<&str>,
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
    ) -> ProgressPanel {
        let panel = JComponent::new_panel();
        let task_label = JComponent::new_label("");
        let progress_bar = JComponent::new_progress_bar();
        let debug_level = etomo_director::ARGUMENTS.lock().unwrap().get_debug_level();
        let this = ProgressPanel {
            panel,
            progress_panel: JComponent::new_panel(),
            task_label,
            progress_bar,
            manager,
            axis_id,
            counter: Cell::new(0),
            value: Cell::new(0),
            maximum: Cell::new(0),
            minimum: Cell::new(0),
            start_time: Cell::new(0),
            cur_standard_bar_string: Cell::new(None),
            bar_string: RefCell::new(None),
            emergency_monitor_alert: Cell::new(false),
            label: RefCell::new(None),
            stopped: Cell::new(true),
            n_packed: Cell::new(0),
            debug_level,
            progress_timer_action_listener: RefCell::new(None),
            timer: RefCell::new(None),
        };
        if let Some(new_label) = new_label {
            this.task_label.set_text(new_label);
        } else {
            this.task_label.set_text("");
        }
        // Swing layout: panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS)).
        this.panel.add(&this.task_label);
        // Swing layout: panel.add(Box.createRigidArea(FixedDim.x0_y5)).
        this.panel.add(&this.progress_bar);
        // Swing layout: panel.setAlignmentY(Component.BOTTOM_ALIGNMENT).
        this.progress_bar.set_name(Some(NAME));
        this.task_label.set_name(Some(LABEL_NAME));
        this
    }

    /// Java static `getInstance(String, BaseManager, AxisID)`.
    pub fn get_instance(
        new_label: Option<&str>,
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
    ) -> Rc<ProgressPanel> {
        let instance = Rc::new(ProgressPanel::new(new_label, manager, axis_id));
        instance.add_listeners();
        instance
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.panel.set_visible(visible);
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        let listener = ProgressTimerActionListener::new(Rc::downgrade(self));
        *self.progress_timer_action_listener.borrow_mut() = Some(listener.clone());
        let timer = Timer::new(
            1000,
            Rc::new(move || {
                listener.action_performed();
            }),
        );
        *self.timer.borrow_mut() = Some(timer);
    }

    /// Java private `pack()`.  Pack the dialog the first few times it is
    /// changed, so that scroll bars aren't displayed the first time a process
    /// runs.
    fn pack(&self) {
        if self.n_packed.get() >= MAX_PACK {
            return;
        }
        self.n_packed.set(self.n_packed.get() + 1);
        let manager = self.manager;
        let axis_id = self.axis_id;
        ui_harness::with(|harness| harness.pack_axis_id_base_manager(axis_id, Some(manager)));
    }

    /// Java `isStopped()`.
    pub fn is_stopped(&self) -> bool {
        self.stopped.get()
    }

    // Java `setBackground(Color bg) { panel.setBackground(bg); }` - Swing
    // painting (background colour), not modelled.

    /// Java `setLabel(String)`.
    pub fn set_label(self: &Rc<Self>, new_label: Option<&str>) {
        self.stopped.set(false);
        *self.label.borrow_mut() = new_label.map(str::to_owned);
        let this = EdtRef::new(self.clone());
        event_queue::invoke_later(move || this.get().set_label_later());
    }

    /// Java inner class `SetLabelLater.run()`.
    fn set_label_later(&self) {
        self.set_task_label();
        self.revalidate();
        self.repaint();
        self.pack();
    }

    /// Java `start()`.
    pub fn start(self: &Rc<Self>) {
        self.stopped.set(false);
        // Setting the progress bar indeterminate causes it to move on its own
        self.counter.set(0);
        *self.bar_string.borrow_mut() = None;
        self.start_time
            .set(utilities::java_lang_system_current_time_millis());
        let this = EdtRef::new(self.clone());
        event_queue::invoke_later(move || this.get().start_later());
    }

    /// Java inner class `StartLater.run()`.
    fn start_later(&self) {
        self.emergency_monitor_alert.set(false);
        let progress_bar = self.get_progress_bar();
        progress_bar.set_indeterminate(true);
        self.cur_standard_bar_string.set(None);
        Self::set_bar_string_and_prefer(Some(&progress_bar), Some(""), true);
        self.start_timer();
        self.pack();
    }

    /// Java `startIndeterminateMode()`.
    pub fn start_indeterminate_mode(self: &Rc<Self>) {
        self.stopped.set(false);
        // Setting the progress bar indeterminate causes it to move on its own
        self.counter.set(0);
        self.start_time
            .set(utilities::java_lang_system_current_time_millis());
        let this = EdtRef::new(self.clone());
        event_queue::invoke_later(move || this.get().start_indeterminate_mode_later());
    }

    /// Java inner class `startIndeterminateModeLater.run()`.
    fn start_indeterminate_mode_later(&self) {
        self.emergency_monitor_alert.set(false);
        let progress_bar = self.get_progress_bar();
        progress_bar.set_indeterminate(true);
        self.cur_standard_bar_string.set(None);
        Self::set_bar_string_and_prefer(Some(&progress_bar), Some(""), true);
        self.revalidate();
        self.repaint();
        self.pack();
    }

    /// Java `stop(ProcessEndState, String)`.
    pub fn stop(self: &Rc<Self>, state: Option<ProcessEndState>, status_string: Option<&str>) {
        let mut state = state;
        self.stopped.set(true);
        self.counter.set(0);
        // File lock information should be preserved.
        if state != Some(ProcessEndState::FileLockFailure) {
            *self.bar_string.borrow_mut() = None;
        }
        if state.is_none() {
            state = Some(ProcessEndState::Done);
        }
        let this = EdtRef::new(self.clone());
        let status_string = status_string.map(str::to_owned);
        event_queue::invoke_later(move || this.get().stop_later(state, status_string.as_deref()));
    }

    /// Java inner class `StopLater.run()`.
    fn stop_later(&self, state: Option<ProcessEndState>, status_string: Option<&str>) {
        self.stop_timer();
        let progress_bar = self.get_progress_bar();
        self.set_progress_bar_counter();
        progress_bar.set_indeterminate(false);
        // If it's failed, don't override emergency monitor bar strings.
        // if (state == ProcessEndState.FILE_LOCK_FAILURE
        // || (emergencyMonitorAlert && state == ProcessEndState.FAILED)) {
        // return;
        // }
        self.emergency_monitor_alert.set(false);
        let mut new_bar_string = String::new();
        let cur_bar_string = progress_bar.get_string();
        // Keep the current bar string if its still in force.
        if let Some(cur_standard_bar_string) = self.cur_standard_bar_string.get() {
            if cur_standard_bar_string.matches(cur_bar_string.as_deref()) {
                // Java StringBuilder.append(null) appends "null"; matches() is
                // false for a null string, so it cannot be null here.
                new_bar_string.push_str(cur_bar_string.as_deref().unwrap_or("null"));
            }
        }
        if let Some(state) = state {
            if !new_bar_string.is_empty() {
                new_bar_string.push(' ');
            }
            new_bar_string.push_str(state.get_bar_string());
        }
        if !utilities::is_empty(status_string) {
            if !new_bar_string.is_empty() {
                new_bar_string.push_str(":  ");
            }
            new_bar_string.push_str(status_string.unwrap());
        }
        Self::set_bar_string_and_prefer(Some(&progress_bar), Some(&new_bar_string), true);
        self.pack();
    }

    /// Java private static `setBarStringAndPrefer(JProgressBar, String, boolean)`.
    /// Sets the progress bar string and increases its preferred width if
    /// necessary.  Returns true if the width changed.
    fn set_bar_string_and_prefer(
        progress_bar: Option<&Rc<JComponent>>,
        bar_string: Option<&str>,
        paint: bool,
    ) -> bool {
        let width_changed = ui_utilities::set_string_and_prefer(progress_bar, bar_string, true);
        if progress_bar.is_some() && paint {
            // Swing painting: progressBar.setStringPainted(true).
        }
        width_changed
    }

    /// Java private `increment()`.
    fn increment(self: &Rc<Self>) {
        let this = EdtRef::new(self.clone());
        let stopped = self.stopped.get();
        event_queue::invoke_later(move || this.get().increment_later(stopped));
    }

    /// Java inner class `IncrementLater.run()`, with its `stopped` field.
    fn increment_later(&self, stopped: bool) {
        // Fixing a bug during kill process where the timer doesn't stop: the
        // progress bar goes to determinate mode and increments based on the timer.
        // If the progress bar is stopped this call should never happen.
        // If the timer did not stop before it generated the event that caused
        // increment to be called, then the timer will never stop.
        // Tell the timer to stop each time this function is called incorrectly.
        if stopped {
            return;
        }
        self.set_progress_bar_value();
        // Put the elapsed time into the progress bar string
        let bar_string = self.bar_string.borrow().clone();
        match bar_string {
            None => {
                Self::set_bar_string_and_prefer(
                    Some(&self.get_progress_bar()),
                    Some(&format!(
                        "Elapsed time: {}",
                        utilities::millis_to_min_and_secs(
                            (utilities::java_lang_system_current_time_millis()
                                - self.get_start_time()) as f64
                        )
                    )),
                    true,
                );
            }
            Some(bar_string) => {
                Self::set_bar_string_and_prefer(
                    Some(&self.get_progress_bar()),
                    Some(&format!(
                        "{} : {}",
                        bar_string,
                        utilities::millis_to_min_and_secs(
                            (utilities::java_lang_system_current_time_millis()
                                - self.get_start_time()) as f64
                        )
                    )),
                    true,
                );
            }
        }
        self.validate();
        self.repaint();
        self.increment_counter();
        self.restart_timer();
        self.pack();
    }

    /// Java `setMaximum(int, boolean)`.
    pub fn set_maximum(self: &Rc<Self>, n: i32, indeterminate_mode: bool) {
        self.stopped.set(false);
        self.maximum.set(n);
        let this = EdtRef::new(self.clone());
        event_queue::invoke_later(move || this.get().set_maximum_later(indeterminate_mode));
    }

    /// Java inner class `SetMaximumLater.run()`.
    fn set_maximum_later(&self, indeterminate_mode: bool) {
        self.set_progress_bar_maximum();
        self.get_progress_bar()
            .set_indeterminate(indeterminate_mode);
        // Swing painting: getProgressBar().setStringPainted(true).
        if indeterminate_mode {
            self.revalidate();
            self.repaint();
        }
        self.pack();
    }

    /// Java `setMinimum(int)`.
    pub fn set_minimum(self: &Rc<Self>, n: i32) {
        self.stopped.set(false);
        self.minimum.set(n);
        let this = EdtRef::new(self.clone());
        event_queue::invoke_later(move || this.get().set_minimum_later());
    }

    /// Java inner class `SetMinimumLater.run()`.
    fn set_minimum_later(&self) {
        self.set_progress_bar_minimum();
    }

    /// Java `setValue(int)`.
    pub fn set_value_int(self: &Rc<Self>, n: i32) {
        self.stopped.set(false);
        self.value.set(n);
        let this = EdtRef::new(self.clone());
        event_queue::invoke_later(move || this.get().set_value_later());
    }

    /// Java inner class `SetValueLater.run()`.
    fn set_value_later(&self) {
        self.set_progress_bar_value();
    }

    /// Java synchronized `setEmergencyMonitorBarString(StandardBarString, String)`.
    /// The panel is confined to the event dispatch thread, so `synchronized`
    /// has no Rust counterpart.
    pub fn set_emergency_monitor_bar_string(
        self: &Rc<Self>,
        standard_bar_string: Option<StandardBarString>,
        bar_string: Option<&str>,
    ) {
        // Java constructor of SetEmergencyMonitorBarStringLater, run here.
        self.emergency_monitor_alert.set(true);
        let this = EdtRef::new(self.clone());
        let bar_string = bar_string.map(str::to_owned);
        event_queue::invoke_later(move || {
            this.get()
                .set_emergency_monitor_bar_string_later(standard_bar_string, bar_string.as_deref())
        });
    }

    /// Java inner class `SetEmergencyMonitorBarStringLater.run()`.
    fn set_emergency_monitor_bar_string_later(
        &self,
        standard_bar_string: Option<StandardBarString>,
        bar_string: Option<&str>,
    ) {
        self.set_progress_bar_string(standard_bar_string, bar_string);
    }

    /// Java `isProgressBarStopped()`.
    pub fn is_progress_bar_stopped(&self) -> bool {
        self.stopped.get()
    }

    /// Java synchronized `setValue(int, StandardBarString, String, boolean)`.
    pub fn set_value_int_standard_bar_string_string_boolean(
        self: &Rc<Self>,
        n: i32,
        standard_bar_string: Option<StandardBarString>,
        bar_string: Option<&str>,
        print: bool,
    ) {
        self.stopped.set(false);
        self.value.set(n);
        // this.barString = barString;
        let this = EdtRef::new(self.clone());
        let bar_string = bar_string.map(str::to_owned);
        event_queue::invoke_later(move || {
            this.get()
                .set_value_and_string_later(standard_bar_string, bar_string.as_deref(), print)
        });
    }

    /// Java inner class `SetValueAndStringLater.run()`; its `print` field is
    /// not read by `run`.
    fn set_value_and_string_later(
        &self,
        standard_bar_string: Option<StandardBarString>,
        bar_string: Option<&str>,
        _print: bool,
    ) {
        self.set_progress_bar_value();
        self.set_progress_bar_string(standard_bar_string, bar_string);
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.panel.clone()
    }

    /// Java `getMaximum()`.
    pub fn get_maximum(&self) -> i32 {
        self.progress_bar.get_maximum()
    }

    /// Java `getMinimum()`.
    pub fn get_minimum(&self) -> i32 {
        self.progress_bar.get_minimum()
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> i32 {
        self.progress_bar.get_value()
    }

    /// Java private `setTaskLabel()`.
    fn set_task_label(&self) {
        // The label starts a new process.
        self.cur_standard_bar_string.set(None);
        let label = self.label.borrow().clone();
        if let Some(label) = label {
            self.task_label.set_text(&label);
        } else {
            self.task_label.set_text("");
        }
    }

    /// Java private `revalidate()`.
    fn revalidate(&self) {
        // Swing layout: panel.revalidate().
    }

    /// Java private `repaint()`.
    fn repaint(&self) {
        // Swing painting: panel.repaint().
    }

    /// Java private `validate()`.
    fn validate(&self) {
        // Swing layout: panel.validate().
    }

    /// Java private `getProgressBar()`.
    fn get_progress_bar(&self) -> Rc<JComponent> {
        self.progress_bar.clone()
    }

    /// Java private `restartTimer()`.
    fn restart_timer(&self) {
        let timer = self.timer.borrow().clone();
        if let Some(timer) = timer {
            timer.restart();
        }
    }

    /// Java private `startTimer()`.
    fn start_timer(&self) {
        let timer = self.timer.borrow().clone();
        if let Some(timer) = timer {
            timer.start();
        }
    }

    /// Java private `stopTimer()`.
    fn stop_timer(&self) {
        let timer = self.timer.borrow().clone();
        if let Some(timer) = timer {
            timer.stop();
        }
    }

    /// Java private `setProgressBarCounter()`.
    fn set_progress_bar_counter(&self) {
        self.progress_bar.set_value(self.counter.get());
    }

    /// Java private `setProgressBarValue()`.
    fn set_progress_bar_value(&self) {
        self.progress_bar.set_value(self.value.get());
    }

    /// Java private `incrementCounter()`.
    fn increment_counter(&self) {
        self.counter.set(self.counter.get() + 1);
    }

    /// Java private `setProgressBarMaximum()`.
    fn set_progress_bar_maximum(&self) {
        self.progress_bar.set_maximum(self.maximum.get());
    }

    /// Java private `setProgressBarMinimum()`.
    fn set_progress_bar_minimum(&self) {
        self.progress_bar.set_minimum(self.minimum.get());
    }

    /// Java private `setProgressBarString(StandardBarString, String)`.
    fn set_progress_bar_string(
        &self,
        standard_bar_string: Option<StandardBarString>,
        bar_string: Option<&str>,
    ) {
        self.cur_standard_bar_string.set(standard_bar_string);
        *self.bar_string.borrow_mut() = bar_string.map(str::to_owned);
        if bar_string.is_some() {
            if Self::set_bar_string_and_prefer(Some(&self.progress_bar), bar_string, true) {
                self.pack();
            }
        } else if Self::set_bar_string_and_prefer(Some(&self.progress_bar), Some(""), true) {
            self.pack();
        }
    }

    /// Java private `getStartTime()`.
    fn get_start_time(&self) -> i64 {
        self.start_time.get()
    }
}

/// Java private static final class `ProgressTimerActionListener implements
/// ActionListener`.  Holds its panel weakly (the panel owns the timer that
/// owns this listener).
struct ProgressTimerActionListener {
    panel: std::rc::Weak<ProgressPanel>,
}

impl ProgressTimerActionListener {
    /// Java private constructor `ProgressTimerActionListener(ProgressPanel)`.
    fn new(panel: std::rc::Weak<ProgressPanel>) -> Rc<ProgressTimerActionListener> {
        Rc::new(ProgressTimerActionListener { panel })
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self) {
        let Some(panel) = self.panel.upgrade() else {
            return;
        };
        panel.increment();
    }
}
