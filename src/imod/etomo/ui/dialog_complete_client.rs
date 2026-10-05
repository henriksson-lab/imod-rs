//! `IMOD/Etomo/src/etomo/ui/DialogCompleteClient.java`.
//!
//! Client that DialogCompleteListener will contact.  The one implementor
//! (`BatchRunTomoMetaData`) lives as long as its manager, so a client is a
//! `&'static` reference.

/// Java `public interface DialogCompleteClient`.
pub trait DialogCompleteClient: Send + Sync {
    /// Java `msgDialogComplete(String)`.
    fn msg_dialog_complete(&self, string: Option<&str>);
}
