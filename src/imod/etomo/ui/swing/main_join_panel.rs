//! `IMOD/Etomo/src/etomo/ui/swing/MainJoinPanel.java`.
//!
//! The main panel of the Join interface (`JoinManager`): one
//! `JoinProcessPanel`, into which `JoinManager.openJoinDialog` shows the
//! `JoinDialog`.  Extends [`MainPanel`] (held as `base`, dereffed to) and
//! implements [`MainPanelVirtual`].

use std::cell::RefCell;
use std::ops::Deref;
use std::path::Path;
use std::rc::{Rc, Weak};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanelVirtual;
use super::axis_progress_panel::AxisProgressPanel;
use super::join_process_panel::JoinProcessPanel;
use super::log_window::LogWindow;
use super::main_panel::{MainPanel, MainPanelVirtual};
use super::ui_harness;
use crate::imod::etomo::jdk::{FileFilter, JComponent};
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::join_file_filter::JoinFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public class MainJoinPanel extends MainPanel`.
pub struct MainJoinPanel {
    /// The Java superclass part.
    base: Rc<MainPanel>,
    /// Java superclass field `manager`, read through the `(JoinManager)` cast in
    /// `createAxisPanelA`.
    manager: &'static JoinManager,
    /// Java private `axisPanelA`.
    axis_panel_a: RefCell<Option<Rc<JoinProcessPanel>>>,
    /// Java private `axisPanelB`, which this class never constructs.
    axis_panel_b: RefCell<Option<Rc<JoinProcessPanel>>>,
}

impl Deref for MainJoinPanel {
    type Target = MainPanel;
    fn deref(&self) -> &MainPanel {
        &self.base
    }
}

impl MainJoinPanel {
    /// Java `MainJoinPanel(JoinManager)`.
    pub fn new(join_manager: &'static JoinManager) -> Rc<MainJoinPanel> {
        let this = Rc::new(MainJoinPanel {
            // super(joinManager)
            base: MainPanel::new(join_manager),
            manager: join_manager,
            axis_panel_a: RefCell::new(None),
            axis_panel_b: RefCell::new(None),
        });
        this.base
            .set_this(Rc::downgrade(&this) as Weak<dyn MainPanelVirtual>);
        this
    }

    /// Java package-private `openPanel(JPanel)`.  Open the setup panel.
    pub fn open_panel(&self, panel: &Rc<JComponent>) {
        // Upstream bug fixed in translation (MainJoinPanel.java:156): Java
        // dereferences getScrollA() unchecked; a null scroll panel is skipped.
        if let Some(scroll_a) = self.base.get_scroll_a() {
            scroll_a.add(panel);
        }
        // Swing layout: revalidate().
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }
}

impl MainPanelVirtual for MainJoinPanel {
    fn main_panel(&self) -> &MainPanel {
        &self.base
    }

    /// Java `saveDisplayState()`: empty.
    fn save_display_state(&self) {}

    /// Java package-private `getDataFileFilter()`.
    fn get_data_file_filter(&self) -> Option<Rc<dyn FileFilter>> {
        Some(Rc::new(JoinFileFilter::new()) as Rc<dyn FileFilter>)
    }

    /// Java package-private `createAxisPanelA(AxisID, AxisProgressPanel)`.
    fn create_axis_panel_a(&self, axis_id: AxisID, axis_progress_panel: Rc<AxisProgressPanel>) {
        let panel = JoinProcessPanel::new(self.manager, axis_id, axis_progress_panel);
        *self.axis_panel_a.borrow_mut() = Some(panel);
    }

    /// Java package-private `createAxisPanelB(AxisProgressPanel)`: empty.
    fn create_axis_panel_b(&self, _axis_progress_panel: Rc<AxisProgressPanel>) {}

    /// Java package-private `getAxisPanelA()`.
    fn get_axis_panel_a(&self) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        self.axis_panel_a
            .borrow()
            .clone()
            .map(|panel| panel as Rc<dyn AxisProcessPanelVirtual>)
    }

    /// Java package-private `getAxisPanelB()`: null.
    fn get_axis_panel_b(&self) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        None
    }

    /// Java public final `setStatusBarText(File, BaseMetaData, LogWindow)`.  Set the
    /// status bar with the file name of the data parameter file.
    fn set_status_bar_text(
        &self,
        param_file: Option<&Path>,
        meta_data: Option<&dyn BaseMetaData>,
        log_window: Option<&Rc<LogWindow>>,
    ) {
        let buffer = String::new();
        match meta_data {
            Some(meta_data) if meta_data.is_valid() => {
                self.base
                    .set_status_bar_text_super(param_file, Some(meta_data), log_window);
            }
            _ => self.base.status_bar.set_text(&buffer),
        }
    }

    /// Java package-private `resetAxisPanels()`.
    fn reset_axis_panels(&self) {
        *self.axis_panel_a.borrow_mut() = None;
        *self.axis_panel_b.borrow_mut() = None;
    }

    /// Java package-private `addAxisPanelA()`.
    fn add_axis_panel_a(&self) {
        // Upstream bug fixed in translation (MainJoinPanel.java:182): Java
        // dereferences getScrollA() and axisPanelA unchecked; a null one is
        // skipped here.
        let axis_panel_a = self.axis_panel_a.borrow().clone();
        if let (Some(scroll_a), Some(axis_panel_a)) = (self.base.get_scroll_a(), axis_panel_a) {
            scroll_a.add(&axis_panel_a.get_container());
        }
    }

    /// Java package-private `addAxisPanelB()`.
    fn add_axis_panel_b(&self) {
        // Upstream bug fixed in translation (MainJoinPanel.java:187): axisPanelB is
        // never constructed, so Java's dereference throws NullPointerException; a
        // null panel is skipped here.
        let axis_panel_b = self.axis_panel_b.borrow().clone();
        if let (Some(scroll_b), Some(axis_panel_b)) = (self.base.get_scroll_b(), axis_panel_b) {
            scroll_b.add(&axis_panel_b.get_container());
        }
    }

    /// Java package-private `isAxisPanelANull()`.
    fn is_axis_panel_a_null(&self) -> bool {
        self.axis_panel_a.borrow().is_none()
    }

    /// Java package-private `isAxisPanelBNull()`.
    fn is_axis_panel_b_null(&self) -> bool {
        self.axis_panel_b.borrow().is_none()
    }

    /// Java package-private `hideAxisPanelA()`.
    fn hide_axis_panel_a(&self) -> bool {
        // Upstream bug fixed in translation (MainJoinPanel.java:202): a null panel
        // is not hidden (false) here instead of a NullPointerException.
        self.axis_panel_a
            .borrow()
            .clone()
            .is_some_and(|panel| panel.hide())
    }

    /// Java package-private `hideAxisPanelB()`.
    fn hide_axis_panel_b(&self) -> bool {
        // Upstream bug fixed in translation (MainJoinPanel.java:207): axisPanelB is
        // always null; it is not hidden (false) here.
        self.axis_panel_b
            .borrow()
            .clone()
            .is_some_and(|panel| panel.hide())
    }

    /// Java package-private `showAxisPanelA()`.
    fn show_axis_panel_a(&self) {
        if let Some(panel) = self.axis_panel_a.borrow().clone() {
            panel.show();
        }
    }

    /// Java package-private `showAxisPanelB()`.
    fn show_axis_panel_b(&self) {
        if let Some(panel) = self.axis_panel_b.borrow().clone() {
            panel.show();
        }
    }

    /// Java package-private `mapBaseAxisProcessPanel(AxisID)`.
    fn map_base_axis_process_panel(
        &self,
        axis_id: AxisID,
    ) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        if axis_id == AxisID::Second {
            return self
                .axis_panel_b
                .borrow()
                .clone()
                .map(|panel| panel as Rc<dyn AxisProcessPanelVirtual>);
        }
        self.axis_panel_a
            .borrow()
            .clone()
            .map(|panel| panel as Rc<dyn AxisProcessPanelVirtual>)
    }

    /// Java package-private `mapAxisProgressPanel(AxisID)`.
    fn map_axis_progress_panel(&self, axis_id: AxisID) -> Option<Rc<AxisProgressPanel>> {
        if axis_id == AxisID::Second {
            return None;
        }
        Some(self.base.get_progress_panel(axis_id))
    }

    /// Java package-private final `setState(ProcessState, AxisID,
    /// AbstractParallelDialog)`: empty.
    fn set_state(
        &self,
        _process_state: ProcessState,
        _axis_id: AxisID,
        _parallel_dialog: &dyn AbstractParallelDialog,
    ) {
    }
}
