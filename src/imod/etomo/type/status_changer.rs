//! `IMOD/Etomo/src/etomo/type/StatusChanger.java`.
//!
//! `addStatusChangeListener` is called on the event dispatch thread, where the
//! listeners live.  A changer that runs on another thread (the batchruntomo
//! monitors) keeps the listener in an `EdtRef` and delivers to it on the event
//! dispatch thread, as the Java does with `SwingUtilities.invokeLater`.

use std::rc::Rc;

use super::status_change_listener::StatusChangeListener;

/// Java `public interface StatusChanger`.
pub trait StatusChanger {
    /// Java `addStatusChangeListener(StatusChangeListener)`.
    fn add_status_change_listener(&self, listener: Option<Rc<dyn StatusChangeListener>>);
}
