//! `IMOD/Etomo/src/etomo/ui/swing/MainPanel.java`.
//!
//! Abstract base of every manager's main panel: the centre area holding the
//! axis process panels (each inside a scroll pane) or the setup dialog, and
//! the status bar with the busy indicator.  It also carries the progress-bar
//! API the process monitors drive (`setProgressBar`, `setProgressBarValue`,
//! `startProgressBar`, `stopProgressBar`, ...), which it forwards to the
//! axis's `AxisProgressPanel`.
//!
//! Object model (see `ui.md`): [`MainPanel`] is created as an `Rc` (the
//! constructor registers it as the manager's `BusyStatusListener`); a
//! subclass holds it as `base: Rc<MainPanel>` and derefs to it.  The abstract
//! and overridden methods are the trait [`MainPanelVirtual`]; the base stores
//! the subclass as `this` and dispatches through it.  All methods take
//! `&self`: the panel lives on the event dispatch thread, and process
//! monitors reach it through `BaseManager::post_main_panel`.
//!
//! Component tree: `get_component()` is the root panel; the scroll panes of
//! the axis panels (and `MainTomogramPanel`'s setup dialog) are added under
//! `panelCenter` exactly where the Java adds them, so a driver searching the
//! frame's root finds the current dialog's components by name.

use std::cell::{Cell, RefCell, RefMut};
use std::fmt::Display;
use std::ops::Deref;
use std::path::Path;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanelVirtual;
use super::axis_progress_panel::AxisProgressPanel;
use super::busy_status_panel;
use super::etomo_panel::EtomoPanel;
use super::log_window::LogWindow;
use super::parallel_panel::ParallelPanel;
use super::single_line_button::SingleLineButton;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{FileFilter, JComponent};
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusListener;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;
use crate::imod::etomo::util::event_queue::{self, EdtRef};

/// Java private `STATUS_BAR_EMPTY_TITLE`.
const STATUS_BAR_EMPTY_TITLE: &str = "No data set loaded";
/// Java private `STATUS_BAR_BASE_TITLE`.
const STATUS_BAR_BASE_TITLE: &str = "Data file: ";

// Java private statics `estimatedMenuHeight = 60`, `extraScreenWidthMultiplier
// = 2` and `frameBorder = FixedDim.frameBorder` - Swing layout sizes, unused.

/// The `MainPanel` methods a subclass implements or overrides, dispatched
/// virtually.  Defaults are the `MainPanel` bodies.
pub trait MainPanelVirtual {
    /// The embedded `MainPanel` (the Java superclass part).
    fn main_panel(&self) -> &MainPanel;

    /// Java abstract `createAxisPanelA(AxisID, AxisProgressPanel)`.
    fn create_axis_panel_a(&self, axis_id: AxisID, axis_progress_panel: Rc<AxisProgressPanel>);
    /// Java abstract `createAxisPanelB(AxisProgressPanel)`.
    fn create_axis_panel_b(&self, axis_progress_panel: Rc<AxisProgressPanel>);
    /// Java abstract `resetAxisPanels()`.
    fn reset_axis_panels(&self);
    /// Java abstract `addAxisPanelA()`.
    fn add_axis_panel_a(&self);
    /// Java abstract `addAxisPanelB()`.
    fn add_axis_panel_b(&self);
    /// Java abstract `isAxisPanelANull()`.
    fn is_axis_panel_a_null(&self) -> bool;
    /// Java abstract `isAxisPanelBNull()`.
    fn is_axis_panel_b_null(&self) -> bool;
    /// Java abstract `hideAxisPanelA()`.
    fn hide_axis_panel_a(&self) -> bool;
    /// Java abstract `hideAxisPanelB()`.
    fn hide_axis_panel_b(&self) -> bool;
    /// Java abstract `showAxisPanelA()`.
    fn show_axis_panel_a(&self);
    /// Java abstract `showAxisPanelB()`.
    fn show_axis_panel_b(&self);
    /// Java abstract `mapBaseAxisProcessPanel(AxisID)`; Java may return null.
    fn map_base_axis_process_panel(
        &self,
        axis_id: AxisID,
    ) -> Option<Rc<dyn AxisProcessPanelVirtual>>;
    /// Java abstract `getDataFileFilter()`.
    fn get_data_file_filter(&self) -> Option<Rc<dyn FileFilter>>;
    /// Java abstract `saveDisplayState()`.
    fn save_display_state(&self);
    /// Java abstract `getAxisPanelA()`.
    fn get_axis_panel_a(&self) -> Option<Rc<dyn AxisProcessPanelVirtual>>;
    /// Java abstract `getAxisPanelB()`.
    fn get_axis_panel_b(&self) -> Option<Rc<dyn AxisProcessPanelVirtual>>;
    /// Java abstract `setState(ProcessState, AxisID, AbstractParallelDialog)`.
    fn set_state(
        &self,
        process_state: ProcessState,
        axis_id: AxisID,
        parallel_dialog: &dyn AbstractParallelDialog,
    );
    /// Java abstract `mapAxisProgressPanel(AxisID)`; Java may return null.
    fn map_axis_progress_panel(&self, axis_id: AxisID) -> Option<Rc<AxisProgressPanel>>;

    /// Java `setStatusBarText(File, BaseMetaData, LogWindow)`.
    fn set_status_bar_text(
        &self,
        param_file: Option<&Path>,
        meta_data: Option<&dyn BaseMetaData>,
        log_window: Option<&Rc<LogWindow>>,
    ) {
        self.main_panel()
            .set_status_bar_text_super(param_file, meta_data, log_window);
    }

    /// Java `showBlankProcess(AxisID)`.
    fn show_blank_process(&self, axis_id: AxisID) {
        self.main_panel().show_blank_process_super(axis_id);
    }

    /// Java `stopProgressBar(AxisID, ProcessEndState, String)`.
    fn stop_progress_bar_axis_id_process_end_state_string(
        &self,
        axis_id: AxisID,
        process_end_state: Option<ProcessEndState>,
        status_string: Option<&str>,
    ) {
        self.main_panel()
            .stop_progress_bar_super(axis_id, process_end_state, status_string);
    }

    /// Java `showProcessingPanel(AxisType)`.
    fn show_processing_panel(&self, axis_type: AxisType) {
        self.main_panel().show_processing_panel_super(axis_type);
    }

    /// Java `showBothAxis()`; returns the B axis scroll pane or null.
    fn show_both_axis(&self) -> Option<Rc<JComponent>> {
        self.main_panel().show_both_axis_super()
    }

    /// Java `showAxisA()`.
    fn show_axis_a(&self) {
        self.main_panel().show_axis_a_super();
    }

    /// Java `showAxisB()`.
    fn show_axis_b(&self) {
        self.main_panel().show_axis_b_super();
    }
}

/// Java public abstract class `MainPanel extends EtomoPanel implements
/// BusyStatusListener`.
pub struct MainPanel {
    /// The Java superclass part (`EtomoPanel extends JPanel`).
    base: Rc<EtomoPanel>,
    /// This panel's own handle (Java `this`).
    self_ref: Weak<MainPanel>,
    /// The subclass object, for virtual dispatch.
    this: RefCell<Weak<dyn MainPanelVirtual>>,

    /// Java `statusBar = new JLabel(STATUS_BAR_EMPTY_TITLE)`.
    pub status_bar: Rc<JComponent>,
    /// Java `panelCenter = new JPanel()`.
    pub panel_center: Rc<JComponent>,
    progress_panel_state_a: RefCell<ProgressPanelState>,

    // These panels get instantiated as needed
    /// Java `scrollA` (a `ScrollPanel`, which is a `JPanel` whose `Scrollable`
    /// overrides are layout only).
    scroll_a: RefCell<Option<Rc<JComponent>>>,
    /// Java `scrollPaneA`.
    scroll_pane_a: RefCell<Option<Rc<JComponent>>>,
    scroll_b: RefCell<Option<Rc<JComponent>>>,
    scroll_pane_b: RefCell<Option<Rc<JComponent>>>,

    /// Java package-private `manager`.
    pub manager: &'static dyn BaseManager,

    axis_progress_panel_a: RefCell<Option<Rc<AxisProgressPanel>>>,
    axis_progress_panel_b: RefCell<Option<Rc<AxisProgressPanel>>>,

    showing_both_axis: Cell<bool>,
    showing_axis_a: Cell<bool>,
    showing_setup: Cell<bool>,
    /// Java package-private `axisType`, initialised to `AxisType.NOT_SET`.
    pub axis_type: Cell<AxisType>,

    /// Java `pnlBusyStatus = new JPanel()`.
    pnl_busy_status: Rc<JComponent>,
    /// Java `lBusyStatusA = createBusyStatusLabel()`.
    l_busy_status_a: Rc<JComponent>,

    l_busy_status_b: RefCell<Option<Rc<JComponent>>>,
    progress_panel_state_b: RefCell<Option<ProgressPanelState>>,
    debug: Cell<bool>,
}

impl Deref for MainPanel {
    type Target = EtomoPanel;
    fn deref(&self) -> &EtomoPanel {
        &self.base
    }
}

impl MainPanel {
    /// Java constructor `MainPanel(BaseManager)`.  Main window constructor.
    /// This sets up the menus and status line.  The subclass must call
    /// [`MainPanel::set_this`] right after it has been created.
    pub fn new(manager: &'static dyn BaseManager) -> Rc<MainPanel> {
        let this = Rc::new_cyclic(|self_ref| MainPanel {
            base: EtomoPanel::new(),
            self_ref: self_ref.clone(),
            this: RefCell::new(Weak::<NoSubclass>::new() as Weak<dyn MainPanelVirtual>),
            status_bar: JComponent::new_label(STATUS_BAR_EMPTY_TITLE),
            panel_center: JComponent::new_panel(),
            progress_panel_state_a: RefCell::new(ProgressPanelState::new()),
            scroll_a: RefCell::new(None),
            scroll_pane_a: RefCell::new(None),
            scroll_b: RefCell::new(None),
            scroll_pane_b: RefCell::new(None),
            manager,
            axis_progress_panel_a: RefCell::new(None),
            axis_progress_panel_b: RefCell::new(None),
            showing_both_axis: Cell::new(false),
            showing_axis_a: Cell::new(true),
            showing_setup: Cell::new(false),
            axis_type: Cell::new(AxisType::NotSet),
            pnl_busy_status: JComponent::new_panel(),
            l_busy_status_a: MainPanel::create_busy_status_label(),
            l_busy_status_b: RefCell::new(None),
            progress_panel_state_b: RefCell::new(None),
            debug: Cell::new(false),
        });
        manager.add_busy_status_listener(Some(Arc::new(EdtRef::new(
            this.clone() as Rc<dyn BusyStatusListener>
        ))));
        // AWT events (not modelled): enableEvents(AWTEvent.WINDOW_EVENT_MASK).
        this.l_busy_status_a.set_enabled(false);
        // Swing layout: setLayout(new BorderLayout()).
        // Construct the main frame panel layout
        // Swing layout: panelCenter.setLayout(new BoxLayout(panelCenter, BoxLayout.X_AXIS)).
        let root = this.base.get_component();
        // BorderLayout.CENTER
        root.add(&this.panel_center);
        let pnl_status = JComponent::new_panel();
        // BorderLayout.SOUTH
        root.add(&pnl_status);
        // Status
        // Swing layout: pnlStatus.setLayout(createStatusBorderLayout()).
        // BorderLayout.WEST
        pnl_status.add(&this.status_bar);
        // BorderLayout.EAST
        pnl_status.add(&this.pnl_busy_status);
        // BusyStatus
        // Swing layout: pnlBusyStatus.setLayout(new BorderLayout()).
        // BorderLayout.EAST
        this.pnl_busy_status.add(&this.l_busy_status_a);
        this
    }

    /// Installs the subclass object for virtual dispatch (Rust-only; the Java
    /// `this` is the subclass object from the start).
    pub fn set_this(&self, this: Weak<dyn MainPanelVirtual>) {
        *self.this.borrow_mut() = this;
    }

    /// The subclass object (Java `this` seen through a virtual call).
    pub fn this(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.this.borrow().upgrade()
    }

    /// The root panel (the Java `MainPanel` object as a `Component`).
    pub fn get_component(&self) -> Rc<JComponent> {
        self.base.get_component()
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
    }

    // Java static `createStatusBorderLayout()`: returns a BorderLayout with an
    // hgap of 2 - Swing layout, not modelled.

    /// Java final `getStatusBarText()`.
    pub fn get_status_bar_text(&self) -> String {
        self.status_bar.get_text()
    }

    /// Java final `getStatus()`.
    pub fn get_status(&self) -> String {
        let status = self.status_bar.get_text();
        if status == STATUS_BAR_EMPTY_TITLE {
            return String::new();
        }
        if let Some(rest) = status.strip_prefix(STATUS_BAR_BASE_TITLE) {
            return rest.to_owned();
        }
        status
    }

    /// Java final `repaint()` (overrides `JComponent.repaint`).
    pub fn repaint(&self) {
        // Swing painting: super.repaint().
        let focus_component = self.manager.get_focus_component();
        if focus_component.is_some() {
            // Swing focus (not modelled): focusComponent.requestFocus().
        }
    }

    /// Java `getScrollA()`.
    pub fn get_scroll_a(&self) -> Option<Rc<JComponent>> {
        self.scroll_a.borrow().clone()
    }

    /// Java `getScrollB()`.
    pub fn get_scroll_b(&self) -> Option<Rc<JComponent>> {
        self.scroll_b.borrow().clone()
    }

    /// Java `setStatusBarText(File, BaseMetaData, LogWindow)`, dispatched to
    /// the subclass.
    pub fn set_status_bar_text(
        &self,
        param_file: Option<&Path>,
        meta_data: Option<&dyn BaseMetaData>,
        log_window: Option<&Rc<LogWindow>>,
    ) {
        match self.this() {
            Some(this) => this.set_status_bar_text(param_file, meta_data, log_window),
            None => self.set_status_bar_text_super(param_file, meta_data, log_window),
        }
    }

    /// The `MainPanel` body of Java `setStatusBarText(File, BaseMetaData,
    /// LogWindow)`.
    pub fn set_status_bar_text_super(
        &self,
        param_file: Option<&Path>,
        meta_data: Option<&dyn BaseMetaData>,
        log_window: Option<&Rc<LogWindow>>,
    ) {
        // Set the title of log panel. SetStatusBarText is used by all of the
        // interfaces so this is good place to do it.
        if let Some(log_window) = log_window {
            log_window.set_title(
                param_file,
                meta_data,
                self.manager.get_property_user_dir().as_deref(),
            );
        }
        // Set the status bar
        let max_title_length: usize = 60;
        if meta_data.is_none() {
            self.status_bar.set_text(STATUS_BAR_EMPTY_TITLE);
        } else if let Some(param_file) = param_file {
            // Java `File.getAbsolutePath()`.
            let mut dataset_name = std::path::absolute(param_file)
                .unwrap_or_else(|_| param_file.to_path_buf())
                .to_string_lossy()
                .into_owned();
            let length = dataset_name.chars().count();
            if STATUS_BAR_BASE_TITLE.len() + length > max_title_length {
                // Shorten the dataset name
                let keep = max_title_length - STATUS_BAR_BASE_TITLE.len() - 3;
                let tail: String = dataset_name.chars().skip(length - keep).collect();
                dataset_name = format!("...{tail}");
            }
            let title = format!("{STATUS_BAR_BASE_TITLE}{dataset_name}");
            self.status_bar.set_text(&title);
        } else {
            self.status_bar
                .set_text(&format!("{STATUS_BAR_BASE_TITLE}NOT SAVED"));
        }
    }

    /// Java final `setStatusBarTextToDirectory(String, int)`.
    pub fn set_status_bar_text_to_directory(
        &self,
        directory: Option<&str>,
        max_title_length: usize,
    ) {
        // Set the status bar
        // Java `directory.matches("\\s*")`: every character is Java whitespace.
        let blank = directory.is_none_or(|directory| {
            directory
                .chars()
                .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
        });
        if blank {
            self.status_bar.set_text("");
        } else {
            let mut directory = directory.unwrap().to_owned();
            let length = directory.chars().count();
            if length > max_title_length {
                // Shorten the dataset name
                let tail: String = directory.chars().skip(length - max_title_length).collect();
                directory = format!("...{tail}");
            }
            let title = directory;
            self.status_bar.set_text(&title);
        }
    }

    /// Java final `setDividerLocation(double)`.  Empty in the Java.
    pub fn set_divider_location(&self, _value: f64) {}

    /// Java `showBlankProcess(AxisID)`, dispatched to the subclass.
    pub fn show_blank_process(&self, axis_id: AxisID) {
        match self.this() {
            Some(this) => this.show_blank_process(axis_id),
            None => self.show_blank_process_super(axis_id),
        }
    }

    /// The `MainPanel` body of Java `showBlankProcess(AxisID)`.  Show a blank
    /// processing panel.
    pub fn show_blank_process_super(&self, axis_id: AxisID) {
        // Upstream bug fixed in translation (MainPanel.java:221): the Java
        // dereferences mapBaseAxisProcessPanel's result, which is null before the
        // processing panel exists (NullPointerException); nothing is erased then.
        if let Some(axis_panel) = self.map_base_axis_process_panel(axis_id) {
            axis_panel.axis_process_panel().erase_dialog_panel();
        }
    }

    /// Java final `showProcess(Container, AxisID)`.  Show the specified
    /// processing panel.
    pub fn show_process(&self, process_panel: &Rc<JComponent>, axis_id: AxisID) {
        let axis_panel = self.map_base_axis_process_panel(axis_id);
        // Upstream bug fixed in translation (MainPanel.java:228-229): a null axis
        // panel throws a NullPointerException in the Java; nothing is shown then.
        if let Some(axis_panel) = axis_panel {
            axis_panel
                .axis_process_panel()
                .replace_dialog_panel(process_panel);
        }
        ui_harness::INSTANCE
            .with(|harness| harness.pack_axis_id_base_manager(Some(axis_id), Some(self.manager)));
    }

    /// Java final `setProgressBar(String, int, boolean, AxisID)`.
    pub fn set_progress_bar_string_int_boolean_axis_id(
        &self,
        label: Option<&str>,
        n_steps: i32,
        indeterminate_mode: bool,
        axis_id: AxisID,
    ) {
        self.set_progress_bar_string_int_boolean_axis_id_boolean(
            label,
            n_steps,
            indeterminate_mode,
            axis_id,
            false,
        );
    }

    /// Java `mapProgressPanelState(AxisID)`.
    fn map_progress_panel_state(&self, axis_id: AxisID) -> RefMut<'_, ProgressPanelState> {
        if axis_id == AxisID::Second {
            // Java `synchronized (this)`: the panel is confined to the event
            // dispatch thread.
            return RefMut::map(self.progress_panel_state_b.borrow_mut(), |state| {
                state.get_or_insert_with(ProgressPanelState::new)
            });
        }
        self.progress_panel_state_a.borrow_mut()
    }

    /// Java final `setProcessCanBeKilled(boolean, AxisID)`.
    pub fn set_process_can_be_killed(&self, killable: bool, axis_id: AxisID) {
        self.map_progress_panel_state(axis_id).print_killable_state(
            "setProcessCanBeKilled",
            false,
            killable,
        );
        let panel = self.map_axis_progress_panel(axis_id);
        if let Some(panel) = panel {
            panel.set_process_can_be_killed(killable);
        }
    }

    /// Java final `setProgressBar(String, AxisID, boolean)`.  Just give the
    /// progress bar a label and an optional moving bar.
    pub fn set_progress_bar_string_axis_id_boolean(
        &self,
        label: Option<&str>,
        axis_id: AxisID,
        indeterminate_mode: bool,
    ) {
        self.map_progress_panel_state(axis_id)
            .print_state_string_boolean_string("setProgressBar", false, label);
        let panel = self.map_axis_progress_panel(axis_id);
        if let Some(panel) = panel {
            panel.set_progress_bar_string_boolean(label, indeterminate_mode);
        }
    }

    /// Java final `setProgressBar(String, int, boolean, AxisID, boolean)`.  Set
    /// the progress bar to the beginning of determinant sequence.  Returns
    /// whether the values have changed.
    pub fn set_progress_bar_string_int_boolean_axis_id_boolean(
        &self,
        label: Option<&str>,
        n_steps: i32,
        indeterminate_mode: bool,
        axis_id: AxisID,
        pause_enabled: bool,
    ) -> bool {
        let retval = self
            .map_progress_panel_state(axis_id)
            .print_state_string_boolean_string_int_boolean(
                "setProgressBar",
                false,
                label,
                n_steps,
                pause_enabled,
            );
        let panel = self.map_axis_progress_panel(axis_id);
        if let Some(panel) = panel {
            panel.set_progress_bar_string_int_boolean(label, n_steps, indeterminate_mode);
            panel.set_progress_bar_value_int(0);
        }
        let axis_panel = self.map_base_axis_process_panel(axis_id);
        if let Some(axis_panel) = axis_panel {
            axis_panel
                .axis_process_panel()
                .set_pause_enabled(pause_enabled);
        }
        retval
    }

    /// Java final `setStaticProgressBar(String, AxisID)`.
    pub fn set_static_progress_bar(&self, label: Option<&str>, axis_id: AxisID) {
        self.map_progress_panel_state(axis_id)
            .print_state_string_boolean_string("setStaticProgressBar", false, label);
        let panel = self.map_axis_progress_panel(axis_id);
        if let Some(panel) = panel {
            panel.set_static_progress_bar(label);
        }
    }

    /// Java `createParallelPanel(AxisID)`.
    pub fn create_parallel_panel(&self, axis_id: AxisID) {
        let axis_panel = self.map_base_axis_process_panel(axis_id);
        if let Some(axis_panel) = axis_panel {
            axis_panel.axis_process_panel().build_parallel_panel();
        }
    }

    /// Java `getParallelPanel(AxisID)`.
    pub fn get_parallel_panel(&self, axis_id: AxisID) -> Option<Rc<ParallelPanel>> {
        let axis_panel = self.map_base_axis_process_panel(axis_id);
        if let Some(axis_panel) = axis_panel {
            return axis_panel.axis_process_panel().get_parallel_panel();
        }
        None
    }

    /// Java final `getParallelPauseButton(AxisID)`.
    pub fn get_parallel_pause_button(&self, axis_id: AxisID) -> Option<Rc<SingleLineButton>> {
        let parallel_panel = self.get_parallel_panel(axis_id);
        if let Some(parallel_panel) = parallel_panel {
            return parallel_panel.get_parallel_pause_button();
        }
        None
    }

    /// Java final `getParallelResumeButton(AxisID)`.
    pub fn get_parallel_resume_button(&self, axis_id: AxisID) -> Option<Rc<SingleLineButton>> {
        let parallel_panel = self.get_parallel_panel(axis_id);
        if let Some(parallel_panel) = parallel_panel {
            return parallel_panel.get_parallel_resume_button();
        }
        None
    }

    /// Java final `getParallelStatusPanel(AxisID)`.
    pub fn get_parallel_status_panel(&self, axis_id: AxisID) -> Option<Rc<JComponent>> {
        let axis_panel = self.map_base_axis_process_panel(axis_id);
        if let Some(axis_panel) = axis_panel {
            return Some(axis_panel.axis_process_panel().get_parallel_status_panel());
        }
        None
    }

    /// Java final `done()`.
    pub fn done(&self) {
        let axis_panel = self.map_base_axis_process_panel(AxisID::First);
        if let Some(axis_panel) = axis_panel {
            axis_panel.axis_process_panel().done();
        }
        let axis_panel = self.map_base_axis_process_panel(AxisID::Second);
        if let Some(axis_panel) = axis_panel {
            axis_panel.axis_process_panel().done();
        }
    }

    /// Java final `setProgressBarValue(int, AxisID)`.  Set the progress bar to
    /// the specified value.
    pub fn set_progress_bar_value_int_axis_id(&self, value: i32, axis_id: AxisID) {
        self.map_progress_panel_state(axis_id)
            .print_state_string_boolean_int("setProgressBarValue", false, value);
        // Upstream bug fixed in translation (MainPanel.java:360): the Java
        // dereferences mapAxisProgressPanel's result without the null check the
        // other progress methods make; a null panel is skipped here.
        if let Some(panel) = self.map_axis_progress_panel(axis_id) {
            panel.set_progress_bar_value_int(value);
        }
    }

    /// Java final `setProgressBarValue(int, StandardBarString, AxisID)`.
    pub fn set_progress_bar_value_int_standard_bar_string_axis_id(
        &self,
        value: i32,
        standard_bar_string: Option<StandardBarString>,
        axis_id: AxisID,
    ) -> bool {
        self.set_progress_bar_value_int_standard_bar_string_string_boolean_boolean_axis_id(
            value,
            standard_bar_string,
            None,
            false,
            false,
            axis_id,
        )
    }

    /// Java final `setProgressBarValue(int, StandardBarString, String, boolean,
    /// boolean, AxisID)`.
    pub fn set_progress_bar_value_int_standard_bar_string_string_boolean_boolean_axis_id(
        &self,
        value: i32,
        standard_bar_string: Option<StandardBarString>,
        file_name: Option<&str>,
        renamed: bool,
        failed: bool,
        axis_id: AxisID,
    ) -> bool {
        self.set_progress_bar_value_int_standard_bar_string_string_string_boolean_boolean_axis_id(
            value,
            standard_bar_string,
            file_name,
            None,
            renamed,
            failed,
            axis_id,
        )
    }

    /// Java final `setProgressBarValue(int, StandardBarString, File, File,
    /// AxisID)`.
    pub fn set_progress_bar_value_int_standard_bar_string_file_file_axis_id(
        &self,
        value: i32,
        standard_bar_string: Option<StandardBarString>,
        from_file: Option<&Path>,
        to_file: Option<&Path>,
        axis_id: AxisID,
    ) -> bool {
        let from_name = from_file.map(file_get_name);
        let to_name = to_file.map(file_get_name);
        self.set_progress_bar_value_int_standard_bar_string_string_string_boolean_boolean_axis_id(
            value,
            standard_bar_string,
            from_name.as_deref(),
            to_name.as_deref(),
            false,
            false,
            axis_id,
        )
    }

    /// Java final `setProgressBarValue(int, StandardBarString, File, File,
    /// boolean, boolean, AxisID)`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_progress_bar_value_int_standard_bar_string_file_file_boolean_boolean_axis_id(
        &self,
        value: i32,
        standard_bar_string: Option<StandardBarString>,
        from_file: Option<&Path>,
        to_file: Option<&Path>,
        renamed: bool,
        failed: bool,
        axis_id: AxisID,
    ) -> bool {
        let from_name = from_file.map(file_get_name);
        let to_name = to_file.map(file_get_name);
        self.set_progress_bar_value_int_standard_bar_string_string_string_boolean_boolean_axis_id(
            value,
            standard_bar_string,
            from_name.as_deref(),
            to_name.as_deref(),
            renamed,
            failed,
            axis_id,
        )
    }

    /// Java final `setProgressBarValue(int, StandardBarString, String, String,
    /// boolean, boolean, AxisID)`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_progress_bar_value_int_standard_bar_string_string_string_boolean_boolean_axis_id(
        &self,
        value: i32,
        standard_bar_string: Option<StandardBarString>,
        from_file_name: Option<&str>,
        to_file_name: Option<&str>,
        renamed: bool,
        failed: bool,
        axis_id: AxisID,
    ) -> bool {
        let bar_string = StandardBarString::build_bar_string_static(
            standard_bar_string,
            from_file_name,
            to_file_name,
            renamed,
            failed,
        );
        self.set_progress_bar_value_int_standard_bar_string_string_axis_id(
            value,
            standard_bar_string,
            Some(&bar_string),
            axis_id,
        )
    }

    /// Java final `setEmergencyMonitorBarString(StandardBarString, String,
    /// String, boolean, boolean, AxisID)`.
    pub fn set_emergency_monitor_bar_string_standard_bar_string_string_string_boolean_boolean_axis_id(
        &self,
        standard_bar_string: Option<StandardBarString>,
        from_file_name: Option<&str>,
        to_file_name: Option<&str>,
        renamed: bool,
        failed: bool,
        axis_id: AxisID,
    ) {
        let bar_string = StandardBarString::build_bar_string_static(
            standard_bar_string,
            from_file_name,
            to_file_name,
            renamed,
            failed,
        );
        self.set_emergency_monitor_bar_string_standard_bar_string_string_axis_id(
            standard_bar_string,
            Some(&bar_string),
            axis_id,
        );
    }

    /// Java final `setProgressBarValue(int, String, AxisID)`.  Set the progress
    /// bar to the specified value and update the string.
    pub fn set_progress_bar_value_int_string_axis_id(
        &self,
        value: i32,
        bar_string: Option<&str>,
        axis_id: AxisID,
    ) -> bool {
        let print = self
            .map_progress_panel_state(axis_id)
            .print_state_string_boolean_int_string("setProgressBarValue", false, value, bar_string);
        // Upstream bug fixed in translation (MainPanel.java:417): unchecked
        // dereference of mapAxisProgressPanel's result; a null panel is skipped.
        if let Some(panel) = self.map_axis_progress_panel(axis_id) {
            panel.set_progress_bar_value_int_standard_bar_string_string_boolean(
                value, None, bar_string, print,
            );
        }
        print
    }

    /// Java final `setProgressBarValue(int, StandardBarString, String, AxisID)`.
    pub fn set_progress_bar_value_int_standard_bar_string_string_axis_id(
        &self,
        value: i32,
        standard_bar_string: Option<StandardBarString>,
        bar_string: Option<&str>,
        axis_id: AxisID,
    ) -> bool {
        let print = self
            .map_progress_panel_state(axis_id)
            .print_state_string_boolean_int_string("setProgressBarValue", false, value, bar_string);
        // Upstream bug fixed in translation (MainPanel.java:426): unchecked
        // dereference of mapAxisProgressPanel's result; a null panel is skipped.
        if let Some(panel) = self.map_axis_progress_panel(axis_id) {
            panel.set_progress_bar_value_int_standard_bar_string_string_boolean(
                value,
                standard_bar_string,
                bar_string,
                print,
            );
        }
        print
    }

    /// Java final `setEmergencyMonitorBarString(StandardBarString, String,
    /// AxisID)`.
    pub fn set_emergency_monitor_bar_string_standard_bar_string_string_axis_id(
        &self,
        standard_bar_string: Option<StandardBarString>,
        bar_string: Option<&str>,
        axis_id: AxisID,
    ) {
        // Upstream bug fixed in translation (MainPanel.java:434): unchecked
        // dereference of mapAxisProgressPanel's result; a null panel is skipped.
        if let Some(panel) = self.map_axis_progress_panel(axis_id) {
            panel.set_emergency_monitor_bar_string(standard_bar_string, bar_string);
        }
    }

    /// Java final `startProgressBar(String, AxisID)`.
    pub fn start_progress_bar_string_axis_id(&self, label: Option<&str>, axis_id: AxisID) {
        self.start_progress_bar_string_axis_id_process_name(label, axis_id, None);
    }

    /// Java final `isProgressBarStopped(AxisID)`.
    pub fn is_progress_bar_stopped(&self, axis_id: AxisID) -> bool {
        // Upstream bug fixed in translation (MainPanel.java:447): unchecked
        // dereference of mapAxisProgressPanel's result; a missing panel reads
        // as stopped.
        match self.map_axis_progress_panel(axis_id) {
            Some(panel) => panel.is_progress_bar_stopped(axis_id),
            None => true,
        }
    }

    /// Java final `startProgressBar(String, AxisID, ProcessName)`.  Start the
    /// indeterminate progress bar on the specified axis.
    pub fn start_progress_bar_string_axis_id_process_name(
        &self,
        label: Option<&str>,
        axis_id: AxisID,
        process_name: Option<&ProcessName>,
    ) {
        self.map_progress_panel_state(axis_id)
            .print_state_string_boolean_string_process_name(
                "startProgressBar",
                false,
                label,
                process_name.cloned(),
            );
        // Upstream bug fixed in translation (MainPanel.java:458): unchecked
        // dereference of mapAxisProgressPanel's result; a null panel is skipped.
        if let Some(panel) = self.map_axis_progress_panel(axis_id) {
            panel.start_progress_bar_string_process_name(label, process_name);
        }
    }

    // Java: three commented-out startProgressBar(int, ...) overloads.

    /// Java final `stopProgressBar(AxisID)`.
    pub fn stop_progress_bar_axis_id(&self, axis_id: AxisID) {
        self.stop_progress_bar_axis_id_process_end_state_string(
            axis_id,
            Some(ProcessEndState::Done),
            None,
        );
    }

    /// Java final `stopProgressBar(AxisID, ProcessEndState)`.
    pub fn stop_progress_bar_axis_id_process_end_state(
        &self,
        axis_id: AxisID,
        process_end_state: Option<ProcessEndState>,
    ) {
        self.stop_progress_bar_axis_id_process_end_state_string(axis_id, process_end_state, None);
    }

    /// Java `stopProgressBar(AxisID, ProcessEndState, String)`, dispatched to
    /// the subclass.
    pub fn stop_progress_bar_axis_id_process_end_state_string(
        &self,
        axis_id: AxisID,
        process_end_state: Option<ProcessEndState>,
        status_string: Option<&str>,
    ) {
        match self.this() {
            Some(this) => this.stop_progress_bar_axis_id_process_end_state_string(
                axis_id,
                process_end_state,
                status_string,
            ),
            None => self.stop_progress_bar_super(axis_id, process_end_state, status_string),
        }
    }

    /// The `MainPanel` body of Java `stopProgressBar(AxisID, ProcessEndState,
    /// String)`.  Stop the specified progress bar.
    pub fn stop_progress_bar_super(
        &self,
        axis_id: AxisID,
        process_end_state: Option<ProcessEndState>,
        status_string: Option<&str>,
    ) {
        self.map_progress_panel_state(axis_id)
            .print_state_string_boolean_process_end_state_string(
                "stopProgressBar",
                false,
                process_end_state,
                status_string,
            );
        if self.debug.get() {
            eprintln!(
                "stopProgressBar:processEndState:{},statusString:{}",
                process_end_state.map_or("null".to_owned(), |state| state.to_string()),
                status_string.unwrap_or("null")
            );
        }
        let panel = self.map_axis_progress_panel(axis_id);
        if let Some(panel) = panel {
            panel.stop_progress_bar(process_end_state, status_string);
        }
        let axis_panel = self.map_base_axis_process_panel(axis_id);
        if let Some(axis_panel) = axis_panel {
            axis_panel.axis_process_panel().set_pause_enabled(false);
        }
    }

    /// Java final `getProgressPanel(AxisID)`.  Returns an axis progress panel,
    /// never null; `axis_id` is the axis the panel is created with if it is
    /// null.
    pub fn get_progress_panel(&self, axis_id: AxisID) -> Rc<AxisProgressPanel> {
        if axis_id == AxisID::Second {
            if self.axis_progress_panel_b.borrow().is_none() {
                *self.axis_progress_panel_b.borrow_mut() =
                    Some(AxisProgressPanel::get_instance(Some(axis_id), self.manager));
            }
            return self.axis_progress_panel_b.borrow().clone().unwrap();
        }
        if self.axis_progress_panel_a.borrow().is_none() {
            *self.axis_progress_panel_a.borrow_mut() =
                Some(AxisProgressPanel::get_instance(Some(axis_id), self.manager));
        }
        self.axis_progress_panel_a.borrow().clone().unwrap()
    }

    /// Java `showProcessingPanel(AxisType)`, dispatched to the subclass.
    pub fn show_processing_panel(&self, axis_type: AxisType) {
        match self.this() {
            Some(this) => this.show_processing_panel(axis_type),
            None => self.show_processing_panel_super(axis_type),
        }
    }

    /// The `MainPanel` body of Java `showProcessingPanel(AxisType)`.  Show the
    /// processing panel for the requested AxisType.  Corrects the
    /// axisProgressPanel AxisID based on axisType.
    pub fn show_processing_panel_super(&self, axis_type: AxisType) {
        let Some(this) = self.this() else {
            return;
        };
        // Delete any existing panels
        this.reset_axis_panels();
        self.axis_type.set(axis_type);
        self.panel_center.remove_all();
        let axis_progress_panel_a = self.axis_progress_panel_a.borrow().clone();
        if let Some(axis_progress_panel_a) = axis_progress_panel_a {
            axis_progress_panel_a.correct_axis_id(axis_type);
        }
        if axis_type == AxisType::SingleAxis {
            let axis_id = AxisID::Only;
            if self.axis_progress_panel_a.borrow().is_none() {
                *self.axis_progress_panel_a.borrow_mut() =
                    Some(AxisProgressPanel::get_instance(Some(axis_id), self.manager));
            }
            let axis_progress_panel_a = self.axis_progress_panel_a.borrow().clone().unwrap();
            this.create_axis_panel_a(axis_id, axis_progress_panel_a);
            *self.scroll_a.borrow_mut() = Some(JComponent::new_panel());
            this.add_axis_panel_a();
            let scroll_pane_a = JComponent::new_scroll_pane(self.scroll_a.borrow().as_ref());
            *self.scroll_pane_a.borrow_mut() = Some(scroll_pane_a.clone());
            // Swing layout: setScrollBarIncrements(scrollPaneA.getVerticalScrollBar());
            // setScrollBarIncrements(scrollPaneA.getHorizontalScrollBar()).
            self.panel_center.add(&scroll_pane_a);
        } else {
            let mut axis_id = AxisID::First;
            if self.axis_progress_panel_a.borrow().is_none() {
                *self.axis_progress_panel_a.borrow_mut() =
                    Some(AxisProgressPanel::get_instance(Some(axis_id), self.manager));
            }
            let axis_progress_panel_a = self.axis_progress_panel_a.borrow().clone().unwrap();
            this.create_axis_panel_a(axis_id, axis_progress_panel_a);
            *self.scroll_a.borrow_mut() = Some(JComponent::new_panel());
            this.add_axis_panel_a();
            let scroll_pane_a = JComponent::new_scroll_pane(self.scroll_a.borrow().as_ref());
            *self.scroll_pane_a.borrow_mut() = Some(scroll_pane_a);
            // Swing layout: setScrollBarIncrements(scrollPaneA.getVerticalScrollBar());
            // setScrollBarIncrements(scrollPaneA.getHorizontalScrollBar()).
            axis_id = AxisID::Second;
            if self.axis_progress_panel_b.borrow().is_none() {
                *self.axis_progress_panel_b.borrow_mut() =
                    Some(AxisProgressPanel::get_instance(Some(axis_id), self.manager));
            }
            let axis_progress_panel_b = self.axis_progress_panel_b.borrow().clone().unwrap();
            this.create_axis_panel_b(axis_progress_panel_b);
            *self.scroll_b.borrow_mut() = Some(JComponent::new_panel());
            this.add_axis_panel_b();
            let scroll_pane_b = JComponent::new_scroll_pane(self.scroll_b.borrow().as_ref());
            *self.scroll_pane_b.borrow_mut() = Some(scroll_pane_b);
            // Swing layout: setScrollBarIncrements(scrollPaneB.getVerticalScrollBar());
            // setScrollBarIncrements(scrollPaneB.getHorizontalScrollBar()).
            self.set_axis_a();
        }
    }

    /// Java `getVerticalScrollBarValue(AxisID)`.
    pub fn get_vertical_scroll_bar_value(&self, axis_id: AxisID) -> Option<i32> {
        // Swing layout: the scroll panes' vertical scroll bars (their
        // visibility and value) are not modelled, so no scroll bar is found
        // and the Java's null is returned.
        let _ = axis_id;
        None
    }

    /// Java `setVerticalScrollBarValue(AxisID, Integer)`.
    pub fn set_vertical_scroll_bar_value(&self, axis_id: AxisID, value: Option<i32>) {
        std::thread::sleep(std::time::Duration::from_millis(7));
        if value.is_none() {
            return;
        }
        // Swing layout: (axisID == SECOND ? scrollPaneB : scrollPaneA)
        // .getVerticalScrollBar().setValue(value) - not modelled.  (The Java
        // throws a NullPointerException when that scroll pane is null.)
        let _ = axis_id;
    }

    // Java private final `setScrollBarIncrements(JScrollBar)`: unit increment
    // 10, block increment 50 - Swing layout, not modelled.

    /// Java `showBothAxis()`, dispatched to the subclass.
    pub fn show_both_axis(&self) -> Option<Rc<JComponent>> {
        match self.this() {
            Some(this) => this.show_both_axis(),
            None => self.show_both_axis_super(),
        }
    }

    /// The `MainPanel` body of Java `showBothAxis()`.
    pub fn show_both_axis_super(&self) -> Option<Rc<JComponent>> {
        if self.axis_type.get() != AxisType::DualAxis || self.showing_both_axis.get() {
            return None;
        }
        self.showing_both_axis.set(true);
        self.showing_axis_a.set(true);
        let this = self.this()?;
        let axis_panel = this.get_axis_panel_b();
        if let Some(axis_panel) = axis_panel {
            axis_panel.show_both_axis();
        }
        // Upstream: `getAxisPanelA().showBothAxis()` is unchecked in the Java.
        if let Some(axis_panel_a) = this.get_axis_panel_a() {
            axis_panel_a.show_both_axis();
        }
        self.scroll_pane_b.borrow().clone()
    }

    /// Java final `isShowingBothAxis()`.
    pub fn is_showing_both_axis(&self) -> bool {
        self.showing_both_axis.get()
    }

    /// Java final `isShowingAxisA()`.
    pub fn is_showing_axis_a(&self) -> bool {
        self.showing_axis_a.get()
    }

    /// Java private inner class `SetBusyStatus.run()`, with its `axisID` and
    /// `enabled` fields (the constructor's `gui` argument is unused).
    fn set_busy_status_run(&self, axis_id: AxisID, enabled: bool) {
        if axis_id == AxisID::Second {
            if self.l_busy_status_b.borrow().is_none() {
                *self.l_busy_status_b.borrow_mut() = Some(MainPanel::create_busy_status_label());
            }
            self.l_busy_status_b
                .borrow()
                .as_ref()
                .unwrap()
                .set_enabled(enabled);
        } else {
            self.l_busy_status_a.set_enabled(enabled);
        }
    }

    /// Java static `createBusyStatusLabel()`.
    pub fn create_busy_status_label() -> Rc<JComponent> {
        // Java `new JLabel(BusyStatusPanel.ICON)`; the icon is not modelled.
        let l_busy_icon = JComponent::new_label("");
        l_busy_icon.set_name(Some(busy_status_panel::LABEL));
        l_busy_icon
    }

    /// Java `showAxisA()`, dispatched to the subclass.
    pub fn show_axis_a(&self) {
        match self.this() {
            Some(this) => this.show_axis_a(),
            None => self.show_axis_a_super(),
        }
    }

    /// The `MainPanel` body of Java `showAxisA()`.
    pub fn show_axis_a_super(&self) {
        self.panel_center.remove_all();
        self.pnl_busy_status.remove_all();
        self.set_axis_a();
    }

    /// Java private final `setAxisA()`.
    fn set_axis_a(&self) {
        self.showing_both_axis.set(false);
        self.showing_axis_a.set(true);
        if self.manager.is_valid() {
            // Upstream: Java adds a null scrollPaneA (NullPointerException) before
            // showProcessingPanel; nothing is added then.
            if let Some(scroll_pane_a) = self.scroll_pane_a.borrow().clone() {
                self.panel_center.add(&scroll_pane_a);
            }
        }
        // BorderLayout.EAST
        self.pnl_busy_status.add(&self.l_busy_status_a);
    }

    /// Java `showAxisB()`, dispatched to the subclass.
    pub fn show_axis_b(&self) {
        match self.this() {
            Some(this) => this.show_axis_b(),
            None => self.show_axis_b_super(),
        }
    }

    /// The `MainPanel` body of Java `showAxisB()`.
    pub fn show_axis_b_super(&self) {
        self.showing_both_axis.set(false);
        self.showing_axis_a.set(false);
        self.panel_center.remove_all();
        self.pnl_busy_status.remove_all();
        // Upstream: Java adds a null scrollPaneB (NullPointerException) on a
        // single axis data set; nothing is added then.
        if let Some(scroll_pane_b) = self.scroll_pane_b.borrow().clone() {
            self.panel_center.add(&scroll_pane_b);
        }
        if self.l_busy_status_b.borrow().is_none() {
            let l_busy_status_b = MainPanel::create_busy_status_label();
            l_busy_status_b.set_enabled(false);
            *self.l_busy_status_b.borrow_mut() = Some(l_busy_status_b);
        }
        let l_busy_status_b = self.l_busy_status_b.borrow().clone().unwrap();
        // BorderLayout.EAST
        self.pnl_busy_status.add(&l_busy_status_b);
    }

    /// Java final `setVerticalScrollBarPolicy(boolean)`.
    pub fn set_vertical_scroll_bar_policy(&self, always: bool) {
        // Swing layout: scrollPaneA/scrollPaneB.setVerticalScrollBarPolicy(always
        // ? VERTICAL_SCROLLBAR_ALWAYS : VERTICAL_SCROLLBAR_AS_NEEDED).
        let _ = always;
    }

    // TODO Need a way to repaint the existing font
    /// Java final `repaintWindow()`.
    pub fn repaint_window(&self) {
        // Swing painting: repaintContainer(this) repaints every descendant.
        self.repaint();
    }

    /// Java final `getAxisType()`.
    pub fn get_axis_type(&self) -> AxisType {
        self.axis_type.get()
    }

    /// Java final `isShowingSetup()`.
    pub fn is_showing_setup(&self) -> bool {
        self.showing_setup.get()
    }

    /// Java final `setShowingSetup(boolean)`.
    pub fn set_showing_setup(&self, input: bool) {
        self.showing_setup.set(input);
    }

    /// Java abstract `mapBaseAxisProcessPanel(AxisID)`, dispatched.
    pub fn map_base_axis_process_panel(
        &self,
        axis_id: AxisID,
    ) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        self.this()?.map_base_axis_process_panel(axis_id)
    }

    /// Java abstract `mapAxisProgressPanel(AxisID)`, dispatched.
    pub fn map_axis_progress_panel(&self, axis_id: AxisID) -> Option<Rc<AxisProgressPanel>> {
        self.this()?.map_axis_progress_panel(axis_id)
    }

    /// Java abstract `getDataFileFilter()`, dispatched.
    pub fn get_data_file_filter(&self) -> Option<Rc<dyn FileFilter>> {
        self.this()?.get_data_file_filter()
    }

    /// Java abstract `saveDisplayState()`, dispatched.
    pub fn save_display_state(&self) {
        if let Some(this) = self.this() {
            this.save_display_state();
        }
    }
}

impl Display for MainPanel {
    /// Java final `toString()`.  `super.toString()` (the `JPanel`'s class name
    /// and layout parameters) is not modelled and is left out.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{},]",
            self.manager
                .get_property_user_dir()
                .unwrap_or_else(|| "null".to_owned())
        )
    }
}

impl BusyStatusListener for MainPanel {
    /// Java `msgBusyStatusChanged(AxisID, boolean)`.
    fn msg_busy_status_changed(&self, axis_id: AxisID, process_status: bool) {
        let Some(this) = self.self_ref.upgrade() else {
            return;
        };
        let this = EdtRef::new(this);
        event_queue::invoke_later(move || this.get().set_busy_status_run(axis_id, process_status));
    }
}

/// Java `File.getName()`: the last name segment of the path.
fn file_get_name(file: &Path) -> String {
    file.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Java private static final class `StateVariable<V>`.  Currently only used
/// for diagnostics.
struct StateVariable<V> {
    printed: bool,
    cur_value: Option<V>,
    changed: bool,
}

impl<V: Clone + PartialEq + Display> StateVariable<V> {
    /// Java private constructor `StateVariable()`.
    fn new() -> Self {
        StateVariable {
            printed: false,
            cur_value: None,
            changed: false,
        }
    }

    /// Java `set(V)`.
    fn set_v(&mut self, new_value: Option<V>) {
        self.set_v_boolean(new_value, false);
    }

    /// Java `set(V, boolean)`.  Decide if the newValue is different and save
    /// it.
    fn set_v_boolean(&mut self, new_value: Option<V>, print: bool) {
        self.changed = !self.printed
            || (self.cur_value.is_none() && new_value.is_some())
            || (self.cur_value.is_some() && self.cur_value != new_value);
        if self.changed && print {
            println!(
                "MainPanel:StateVariable:set:printed:{},curValue:{},newValue:{}",
                self.printed,
                self.cur_value
                    .as_ref()
                    .map_or("null".to_owned(), |value| value.to_string()),
                new_value
                    .as_ref()
                    .map_or("null".to_owned(), |value| value.to_string())
            );
        }
        self.cur_value = new_value;
    }

    /// Java `isChanged()`.
    fn is_changed(&self) -> bool {
        self.changed
    }

    /// Java `toString()`.  Assume a print was done.
    fn to_string(&mut self) -> Option<String> {
        self.printed = true;
        self.cur_value.as_ref().map(|value| value.to_string())
    }
}

/// Java `Thread.dumpStack()`.
fn dump_stack() {
    eprintln!(
        "java.lang.Exception: Stack trace\n{}",
        std::backtrace::Backtrace::force_capture()
    );
}

/// Formats Java string concatenation of a possibly null value.
fn or_null(value: Option<String>) -> String {
    value.unwrap_or_else(|| "null".to_owned())
}

/// Java private static final class `ProgressPanelState`.  Currently only used
/// for diagnostics.
struct ProgressPanelState {
    // Set diagnostics to true to print state information when it changes.
    // Diagnostics prints values the first time they are set and each time they
    // change.
    diagnostics: bool,

    killable: StateVariable<bool>,
    label: StateVariable<String>,
    value: StateVariable<i32>,
    n_steps: StateVariable<i32>,
    pause_enabled: StateVariable<bool>,
    bar_string: StateVariable<String>,
    stopped: StateVariable<bool>,
    process_name: StateVariable<ProcessName>,
    process_end_state: StateVariable<ProcessEndState>,
    status_string: StateVariable<String>,
}

impl ProgressPanelState {
    /// Java private constructor `ProgressPanelState()`.
    fn new() -> Self {
        ProgressPanelState {
            diagnostics: false,
            killable: StateVariable::new(),
            label: StateVariable::new(),
            value: StateVariable::new(),
            n_steps: StateVariable::new(),
            pause_enabled: StateVariable::new(),
            bar_string: StateVariable::new(),
            stopped: StateVariable::new(),
            process_name: StateVariable::new(),
            process_end_state: StateVariable::new(),
            status_string: StateVariable::new(),
        }
    }

    /// Java `printKillableState(String, boolean, boolean)`.
    fn print_killable_state(&mut self, descr: &str, dump_stack: bool, killable: bool) -> bool {
        self.killable.set_v(Some(killable));
        if !self.diagnostics || !self.killable.is_changed() {
            return false;
        }
        println!("{descr}:killable:{}", or_null(self.killable.to_string()));
        if dump_stack {
            self::dump_stack();
        }
        true
    }

    /// Java `printState(String, boolean, String, int, boolean)`.
    fn print_state_string_boolean_string_int_boolean(
        &mut self,
        descr: &str,
        dump_stack: bool,
        label: Option<&str>,
        n_steps: i32,
        pause_enabled: bool,
    ) -> bool {
        self.label.set_v(label.map(str::to_owned));
        self.n_steps.set_v(Some(n_steps));
        self.pause_enabled.set_v(Some(pause_enabled));
        if !self.diagnostics
            || (!self.label.is_changed()
                && !self.n_steps.is_changed()
                && !self.pause_enabled.is_changed())
        {
            return false;
        }
        let label = or_null(self.label.to_string());
        let n_steps = or_null(self.n_steps.to_string());
        let pause_enabled = or_null(self.pause_enabled.to_string());
        println!("{descr}:label:{label},nSteps:{n_steps},pauseEnabled:{pause_enabled}");
        if dump_stack {
            self::dump_stack();
        }
        true
    }

    /// Java `printState(String, boolean, String)`.
    fn print_state_string_boolean_string(
        &mut self,
        descr: &str,
        dump_stack: bool,
        label: Option<&str>,
    ) -> bool {
        self.label.set_v(label.map(str::to_owned));
        if !self.diagnostics || !self.label.is_changed() {
            return false;
        }
        println!("{descr}:label:{}", or_null(self.label.to_string()));
        if dump_stack {
            self::dump_stack();
        }
        true
    }

    /// Java `printState(String, boolean, int)`.
    fn print_state_string_boolean_int(
        &mut self,
        descr: &str,
        dump_stack: bool,
        value: i32,
    ) -> bool {
        self.value.set_v(Some(value));
        if !self.diagnostics || !self.value.is_changed() {
            return false;
        }
        println!("{descr}:value:{}", or_null(self.value.to_string()));
        if dump_stack {
            self::dump_stack();
        }
        true
    }

    /// Java `printState(String, boolean, int, String)`.
    fn print_state_string_boolean_int_string(
        &mut self,
        descr: &str,
        dump_stack: bool,
        value: i32,
        bar_string: Option<&str>,
    ) -> bool {
        self.value.set_v_boolean(Some(value), dump_stack);
        self.bar_string
            .set_v_boolean(bar_string.map(str::to_owned), dump_stack);
        if !self.diagnostics || (!self.value.is_changed() && !self.bar_string.is_changed()) {
            return false;
        }
        let value = or_null(self.value.to_string());
        let bar_string = or_null(self.bar_string.to_string());
        println!("{descr}:value:{value},barString:{bar_string}");
        if dump_stack {
            self::dump_stack();
        }
        true
    }

    /// Java `printStoppedState(String, boolean, boolean)`.  Uncalled in the
    /// Java.
    #[allow(dead_code)]
    fn print_stopped_state(&mut self, descr: &str, dump_stack: bool, stopped: bool) -> bool {
        self.stopped.set_v(Some(stopped));
        if !self.diagnostics || !self.stopped.is_changed() {
            return false;
        }
        println!("{descr}:stopped:{}", or_null(self.stopped.to_string()));
        if dump_stack {
            self::dump_stack();
        }
        true
    }

    /// Java `printState(String, boolean, String, ProcessName)`.
    fn print_state_string_boolean_string_process_name(
        &mut self,
        descr: &str,
        dump_stack: bool,
        label: Option<&str>,
        process_name: Option<ProcessName>,
    ) -> bool {
        self.label.set_v(label.map(str::to_owned));
        self.process_name.set_v(process_name);
        if !self.diagnostics || (!self.label.is_changed() && !self.process_name.is_changed()) {
            return false;
        }
        let label = or_null(self.label.to_string());
        let process_name = or_null(self.process_name.to_string());
        println!("{descr}:label:{label},processName:{process_name}");
        if dump_stack {
            self::dump_stack();
        }
        true
    }

    /// Java `printState(String, boolean, ProcessEndState)`.  Uncalled in the
    /// Java.
    #[allow(dead_code)]
    fn print_state_string_boolean_process_end_state(
        &mut self,
        descr: &str,
        dump_stack: bool,
        process_end_state: Option<ProcessEndState>,
    ) -> bool {
        self.process_end_state.set_v(process_end_state);
        if !self.diagnostics || !self.process_end_state.is_changed() {
            return false;
        }
        println!(
            "{descr}:processEndState:{}",
            or_null(self.process_end_state.to_string())
        );
        if dump_stack {
            self::dump_stack();
        }
        true
    }

    /// Java `printState(String, boolean, ProcessEndState, String)`.
    fn print_state_string_boolean_process_end_state_string(
        &mut self,
        descr: &str,
        dump_stack: bool,
        process_end_state: Option<ProcessEndState>,
        status_string: Option<&str>,
    ) -> bool {
        self.process_end_state.set_v(process_end_state);
        self.status_string.set_v(status_string.map(str::to_owned));
        if !self.diagnostics
            || (!self.process_end_state.is_changed() && !self.status_string.is_changed())
        {
            return false;
        }
        let process_end_state = or_null(self.process_end_state.to_string());
        let status_string = or_null(self.status_string.to_string());
        println!("{descr}:processEndState:{process_end_state},statusString:{status_string}");
        if dump_stack {
            self::dump_stack();
        }
        true
    }
}

/// Placeholder type for the empty `Weak<dyn MainPanelVirtual>` held before
/// the subclass installs itself.
struct NoSubclass;

impl MainPanelVirtual for NoSubclass {
    fn main_panel(&self) -> &MainPanel {
        unreachable!("an empty Weak never upgrades")
    }
    fn create_axis_panel_a(&self, _: AxisID, _: Rc<AxisProgressPanel>) {}
    fn create_axis_panel_b(&self, _: Rc<AxisProgressPanel>) {}
    fn reset_axis_panels(&self) {}
    fn add_axis_panel_a(&self) {}
    fn add_axis_panel_b(&self) {}
    fn is_axis_panel_a_null(&self) -> bool {
        true
    }
    fn is_axis_panel_b_null(&self) -> bool {
        true
    }
    fn hide_axis_panel_a(&self) -> bool {
        false
    }
    fn hide_axis_panel_b(&self) -> bool {
        false
    }
    fn show_axis_panel_a(&self) {}
    fn show_axis_panel_b(&self) {}
    fn map_base_axis_process_panel(&self, _: AxisID) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        None
    }
    fn get_data_file_filter(&self) -> Option<Rc<dyn FileFilter>> {
        unreachable!("an empty Weak never upgrades")
    }
    fn save_display_state(&self) {}
    fn get_axis_panel_a(&self) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        None
    }
    fn get_axis_panel_b(&self) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        None
    }
    fn set_state(&self, _: ProcessState, _: AxisID, _: &dyn AbstractParallelDialog) {}
    fn map_axis_progress_panel(&self, _: AxisID) -> Option<Rc<AxisProgressPanel>> {
        None
    }
}
