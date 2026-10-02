//! `IMOD/Etomo/src/etomo/ui/swing/MainFrontPagePanel.java`.
//!
//! The main panel of the front page (`FrontPageManager`): one
//! `FrontPageProcessPanel`, into which `FrontPageDialog.show` puts the front
//! page's buttons.  Extends [`MainPanel`] (held as `base`, dereffed to) and
//! implements [`MainPanelVirtual`].  Required class for FrontPageManager.

use std::cell::RefCell;
use std::ops::Deref;
use std::path::Path;
use std::rc::{Rc, Weak};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanelVirtual;
use super::axis_progress_panel::AxisProgressPanel;
use super::front_page_process_panel::FrontPageProcessPanel;
use super::log_window::LogWindow;
use super::main_panel::{MainPanel, MainPanelVirtual};
use crate::imod::etomo::front_page_manager::FrontPageManager;
use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java public final class `MainFrontPagePanel extends MainPanel`.
pub struct MainFrontPagePanel {
    /// The Java superclass part.
    base: Rc<MainPanel>,
    /// Java private final `manager`.
    manager: &'static FrontPageManager,
    /// Java private `axisPanelA`, initialised to null.
    axis_panel_a: RefCell<Option<Rc<FrontPageProcessPanel>>>,
}

impl Deref for MainFrontPagePanel {
    type Target = MainPanel;
    fn deref(&self) -> &MainPanel {
        &self.base
    }
}

impl MainFrontPagePanel {
    /// Java constructor `MainFrontPagePanel(FrontPageManager)`.
    pub fn new(manager: &'static FrontPageManager) -> Rc<MainFrontPagePanel> {
        let this = Rc::new(MainFrontPagePanel {
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

impl MainPanelVirtual for MainFrontPagePanel {
    fn main_panel(&self) -> &MainPanel {
        &self.base
    }

    /// Java package-private `addAxisPanelA()`.
    fn add_axis_panel_a(&self) {
        // Upstream bug fixed in translation (MainFrontPagePanel.java:47): Java
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
        let panel = FrontPageProcessPanel::new(self.manager, axis_progress_panel);
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
        // Upstream bug fixed in translation (MainFrontPagePanel.java:82): Java
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
        // Upstream bug fixed in translation (MainFrontPagePanel.java:119): Java
        // dereferences axisPanelA unchecked; a null panel is skipped here.
        if let Some(panel) = self.axis_panel_a.borrow().clone() {
            panel.show();
        }
    }

    /// Java package-private `showAxisPanelB()`: empty.
    fn show_axis_panel_b(&self) {}

    /// Java public final `setStatusBarText(File, BaseMetaData, LogWindow)`.
    fn set_status_bar_text(
        &self,
        _param_file: Option<&Path>,
        meta_data: Option<&dyn BaseMetaData>,
        _log_window: Option<&Rc<LogWindow>>,
    ) {
        // Upstream bug fixed in translation (MainFrontPagePanel.java:131): Java
        // dereferences metaData unchecked; a null one leaves the status bar
        // unchanged here.
        let Some(meta_data) = meta_data else {
            return;
        };
        // JLabel.setText(null) shows nothing, as "" does.
        self.base
            .status_bar
            .set_text(&meta_data.get_name().unwrap_or_default());
    }
}
