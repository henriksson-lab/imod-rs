//! `IMOD/Etomo/src/etomo/ui/FlagOriginListener.java`.
//!
//! An interface to allow `FlagExtension` to listen for a flagged state.

use crate::imod::etomo::jdk::ItemEvent;

/// Java `FlagOriginListener extends ItemListener, FocusListener`.  The two Swing
/// listener interfaces contribute their methods here.  No implementer reads the
/// `FocusEvent`, so the focus methods take no event.
pub trait FlagOriginListener {
    /// Java `ItemListener.itemStateChanged(ItemEvent)`.
    fn item_state_changed(&self, event: &ItemEvent);

    /// Java `FocusListener.focusGained(FocusEvent)`.
    fn focus_gained(&self);

    /// Java `FocusListener.focusLost(FocusEvent)`.
    fn focus_lost(&self);
}
