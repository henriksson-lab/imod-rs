//! `IMOD/Etomo/src/etomo/ui/swing/ToggleCell.java`.
//!
//! Java package-private `interface ToggleCell`: a table cell holding a toggle button
//! (`CheckBoxCell`, `RadioButtonCell`).  Objects passed as a `ToggleCell` are
//! `Rc<dyn ToggleCell>`.

use std::rc::Rc;

use crate::imod::etomo::jdk::{ActionListener, ChangeListener, JComponent};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ToggleCell`.
pub trait ToggleCell {
    /// Java `getLabel()`.
    fn get_label(&self) -> Option<String>;

    /// Java `setLabel(String)`.
    fn set_label(&self, label: Option<&str>);

    /// Java `setSelected(boolean)`.
    fn set_selected(&self, selected: bool);

    /// Java `addActionListener(ActionListener)`.
    fn add_action_listener(&self, action_listener: ActionListener);

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)`.  The layout and
    /// constraints are layout only and are not modelled.
    fn add(&self, panel: &Rc<JComponent>);

    /// Java `isSelected()`.
    fn is_selected(&self) -> bool;

    /// Java `getHeight()`.
    fn get_height(&self) -> i32;

    /// Java `getWidth()`.
    fn get_width(&self) -> i32;

    /// Java `setWarning(boolean)`.
    fn set_warning(&self, warning: bool);

    /// Java `addChangeListener(ChangeListener)`.
    fn add_change_listener(&self, listener: ChangeListener);

    /// Java `setEnabled(boolean)`.
    fn set_enabled(&self, enabled: bool);

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;
}
