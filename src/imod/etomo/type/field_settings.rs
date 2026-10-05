//! `IMOD/Etomo/src/etomo/type/FieldSettings.java`.
//!
//! The implementors are event dispatch thread objects (`Rc`, `&self` methods).

/// Java `public interface FieldSettings`.
pub trait FieldSettings {
    /// Java `setSelected(boolean)`.
    fn set_selected(&self, selected: bool);

    /// Java `isSelected()`.
    fn is_selected(&self) -> bool;

    /*
     * Convert as needed.  The lock setting used to be the editable setting.  The
     * editable setting is a real setting only in text fields.  It can mean either
     * editable or locked in etomo text fields.  Editable means that field is
     * participating but can't be changed.  Locked usually means that the field can't
     * be changed until a process is completed.  If non text fields, editable is or
     * was most likely used only for locking.  Convert fields to locking as separate
     * from editable when implementing this interface.  Lock is equivalent to
     * !editable.
     *
     * For InputCell classes, hook isLocked() up to !InputCell.isEditable().  May
     * eventually be able to eliminate the confusing fake editable functionality being
     * used for locking.
     */

    /// Java `isEditable()`.
    fn is_editable(&self) -> bool;

    /// Java `setEditable(boolean)`.
    fn set_editable(&self, editable: bool);

    /// Java `setEnabled(boolean)`.
    fn set_enabled(&self, enabled: bool);

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;
}
