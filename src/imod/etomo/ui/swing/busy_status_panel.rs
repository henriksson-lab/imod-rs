//! `IMOD/Etomo/src/etomo/ui/swing/BusyStatusPanel.java`.
//!
//! The native `JPanel`/`JLabel` is presentation-owned.  This source unit owns
//! the busy icon identity, one-axis filtering, and the queued status update.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java `LABEL`.
pub const LABEL: &str = "lb.busy";
/// Java `ICON = CompleteIcon.createIcon("busy.png")`.
pub const ICON: &str = "busy.png";

/// Java's private inner `SetBusyStatus` runnable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SetBusyStatus {
    pub enabled: bool,
}
impl SetBusyStatus {
    /// `SetBusyStatus(boolean)`.
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }
    /// `run()`; the Slint event loop is its corresponding dispatch boundary.
    pub fn run(self, panel: &mut BusyStatusPanel) {
        panel.busy_status_enabled = self.enabled;
    }
}

/// Fields and operations of Java's final `BusyStatusPanel`.
pub struct BusyStatusPanel {
    pub pnl_root_present: bool,
    pub busy_status_name: Option<String>,
    pub busy_status_enabled: bool,
    pub axis_id: AxisID,
    pub registered: bool,
    /// Swing `invokeLater` queue, retained so callers/event loop own execution
    /// timing instead of applying the update synchronously.
    pub queued_status: Vec<SetBusyStatus>,
}
impl BusyStatusPanel {
    /// `BusyStatusPanel(AxisID)`.
    fn new(axis_id: AxisID) -> Self {
        Self {
            pnl_root_present: true,
            busy_status_name: None,
            busy_status_enabled: true,
            axis_id,
            registered: false,
            queued_status: vec![],
        }
    }
    /// `getInstance(BaseManager, AxisID)`.
    pub fn get_instance(manager: &'static dyn BaseManager, axis_id: AxisID) -> Self {
        let mut instance = Self::new(axis_id);
        instance.create_panel();
        instance.add_listeners(manager);
        instance
    }
    /// `createPanel()`.
    fn create_panel(&mut self) {
        self.busy_status_name = Some(LABEL.into());
        self.busy_status_enabled = false;
    }
    /// `addListeners(BaseManager)`. BaseManager's current BusyStatusMediator
    /// declaration is still an explicit boundary, but registration ownership is
    /// faithfully retained here.
    pub fn add_listeners(&mut self, manager: &'static dyn BaseManager) {
        manager.add_busy_status_listener(None);
        self.registered = true;
    }
    /// `removeListeners(BaseManager)`.
    pub fn remove_listeners(&mut self, manager: &'static dyn BaseManager) {
        manager.remove_busy_status_listener(None);
        self.registered = false;
    }
    /// `getComponent()`.
    pub fn get_component(&self) -> bool {
        self.pnl_root_present
    }
    /// `msgBusyStatusChanged(AxisID, boolean)`.
    pub fn msg_busy_status_changed(&mut self, axis_id: Option<AxisID>, process_status: bool) {
        if self.axis_id.is_same_axis(axis_id) {
            self.queued_status.push(SetBusyStatus::new(process_status));
        }
    }
    /// Runs exactly one queued Java `SwingUtilities.invokeLater` Runnable.
    pub fn run_next_set_busy_status(&mut self) {
        if let Some(update) = self.queued_status.first().copied() {
            self.queued_status.remove(0);
            update.run(self);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    fn panel() -> BusyStatusPanel {
        BusyStatusPanel::get_instance(
            DirectiveEditorManager::new(None, None, None, None),
            AxisID::Second,
        )
    }
    #[test]
    fn source_initialization_names_and_disables_icon() {
        let p = panel();
        assert_eq!(p.busy_status_name.as_deref(), Some(LABEL));
        assert!(!p.busy_status_enabled);
    }
    #[test]
    fn only_matching_axis_enqueues_update() {
        let mut p = panel();
        p.msg_busy_status_changed(Some(AxisID::First), true);
        assert!(p.queued_status.is_empty());
        p.msg_busy_status_changed(Some(AxisID::Second), true);
        assert_eq!(p.queued_status.len(), 1);
        p.run_next_set_busy_status();
        assert!(p.busy_status_enabled);
    }
    #[test]
    fn listener_lifetime_is_source_owned() {
        let mut p = panel();
        let m = DirectiveEditorManager::new(None, None, None, None);
        p.remove_listeners(m);
        assert!(!p.registered);
        p.add_listeners(m);
        assert!(p.registered);
    }
}
