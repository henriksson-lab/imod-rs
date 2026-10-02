//! `IMOD/Etomo/src/etomo/ui/swing/ButtonComponent.java`.
//!
//! The button-like members a panel reads from a check box or radio button it
//! is handed (e.g. `setUseQueueCheckBox`).  Handles are
//! `Rc<dyn ButtonComponent>`.

use crate::imod::etomo::jdk::ActionListener;

/// Java `ButtonComponent`.
pub trait ButtonComponent {
    /// Java `addActionListener(ActionListener)`.
    fn add_action_listener(&self, listener: ActionListener);

    /// Java `isSelected()`.
    fn is_selected(&self) -> bool;

    /// Java `getActionCommand()`.
    fn get_action_command(&self) -> Option<String>;

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;
}
