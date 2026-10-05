//! `IMOD/Etomo/src/etomo/ui/swing/MainBatchRunTomoPanel.java`.
//!
//! The main panel of the batchruntomo interface (`BatchRunTomoManager`): one
//! `BatchRunTomoProcessPanel`, into which the manager shows the
//! `BatchRunTomoDialog`.  Extends [`MainPanel`] (held as `base`, dereffed to) and
//! implements [`MainPanelVirtual`].

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanelVirtual;
use super::axis_progress_panel::AxisProgressPanel;
use super::main_panel::{MainPanel, MainPanelVirtual};
use super::batch_run_tomo_process_panel::BatchRunTomoProcessPanel;
use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java `public final class MainBatchRunTomoPanel extends MainPanel`.
pub struct MainBatchRunTomoPanel {
    /// The Java superclass part.
    base: Rc<MainPanel>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private `axisPanelA`, initialised to null.
    axis_panel_a: RefCell<Option<Rc<BatchRunTomoProcessPanel>>>,
}

impl Deref for MainBatchRunTomoPanel {
    type Target = MainPanel;
    fn deref(&self) -> &MainPanel {
        &self.base
    }
}

impl MainBatchRunTomoPanel {
    /// Java `MainBatchRunTomoPanel(ParallelManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> Rc<MainBatchRunTomoPanel> {
        let this = Rc::new(MainBatchRunTomoPanel {
            // super(manager)
            base: MainPanel::new(manager),
            manager,
            axis_panel_a: RefCell::new(None),
        });
        this.base
            .set_this(Rc::downgrade(&this) as Weak<dyn MainPanelVirtual>);
        this
    }
}

impl MainPanelVirtual for MainBatchRunTomoPanel {
    fn main_panel(&self) -> &MainPanel {
        &self.base
    }

    /// Java package-private `addAxisPanelA()`.
    fn add_axis_panel_a(&self) {
        // Upstream bug fixed in translation (MainBatchRunTomoPanel.java:32): Java
        // dereferences getScrollA() and axisPanelA unchecked; a null one is
        // skipped here.
        let axis_panel_a = self.axis_panel_a.borrow().clone();
        if let (Some(scroll_a), Some(axis_panel_a)) = (self.base.get_scroll_a(), axis_panel_a) {
            scroll_a.add(&axis_panel_a.get_container());
        }
    }

    /// Java package-private `addAxisPanelB()`: empty.
    fn add_axis_panel_b(&self) {}

    /// Java package-private `isAxisPanelANull()`.
    fn is_axis_panel_a_null(&self) -> bool {
        self.axis_panel_a.borrow().is_none()
    }

    /// Java package-private `isAxisPanelBNull()`.
    fn is_axis_panel_b_null(&self) -> bool {
        true
    }

    /// Java package-private `createAxisPanelA(AxisID, AxisProgressPanel)`.
    fn create_axis_panel_a(&self, _axis_id: AxisID, axis_progress_panel: Rc<AxisProgressPanel>) {
        let panel = BatchRunTomoProcessPanel::new(
            self.manager,
            InterfaceType::BatchRunTomo,
            axis_progress_panel,
        );
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

    /// Java package-private `getDataFileFilter()`: null.
    fn get_data_file_filter(&self) -> Option<Rc<dyn FileFilter>> {
        None
    }

    /// Java package-private `hideAxisPanelA()`.
    fn hide_axis_panel_a(&self) -> bool {
        // Upstream bug fixed in translation (MainBatchRunTomoPanel.java:68): Java
        // dereferences axisPanelA unchecked; a null panel is not hidden
        // (false) here.
        self.axis_panel_a
            .borrow()
            .clone()
            .is_some_and(|panel| panel.hide())
    }

    /// Java package-private `hideAxisPanelB()`.
    fn hide_axis_panel_b(&self) -> bool {
        true
    }

    /// Java package-private `mapBaseAxisProcessPanel(AxisID)`.
    fn map_base_axis_process_panel(
        &self,
        axis_id: AxisID,
    ) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        if axis_id == AxisID::Second {
            return None;
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

    /// Java package-private `resetAxisPanels()`.
    fn reset_axis_panels(&self) {
        *self.axis_panel_a.borrow_mut() = None;
    }

    /// Java `saveDisplayState()`: empty.
    fn save_display_state(&self) {}

    /// Java `setState(ProcessState, AxisID, AbstractParallelDialog)`: empty.
    fn set_state(
        &self,
        _process_state: ProcessState,
        _axis_id: AxisID,
        _parallel_dialog: &dyn AbstractParallelDialog,
    ) {
    }

    /// Java package-private `showAxisPanelA()`.
    fn show_axis_panel_a(&self) {
        // Upstream bug fixed in translation (MainBatchRunTomoPanel.java:112): Java
        // dereferences axisPanelA unchecked; a null panel is skipped here.
        if let Some(panel) = self.axis_panel_a.borrow().clone() {
            panel.show();
        }
    }

    /// Java package-private `showAxisPanelB()`: empty.
    fn show_axis_panel_b(&self) {}
}
