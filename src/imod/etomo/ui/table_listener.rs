//! `IMOD/Etomo/src/etomo/ui/TableListener.java`.
//!
//! Listener for table events.  The implementors are event dispatch thread objects;
//! Java's `EventObject` argument (whose source is the table) is not read by any
//! implementor and is carried as `()`.

/// Java `public interface TableListener extends EventListener`.
pub trait TableListener {
    /// Java `lastRowDeleted(EventObject)`.
    fn last_row_deleted(&self, event: Option<&()>);

    /// Java `firstRowAdded(EventObject)`.
    fn first_row_added(&self, event: Option<&()>);
}
