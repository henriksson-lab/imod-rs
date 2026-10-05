//! `IMOD/Etomo/src/etomo/ui/DialogCompleteListener.java`.
//!
//! Listener to be responded to when a dialog is complete.

use super::dialog_complete_client::DialogCompleteClient;

/// Java `public final class DialogCompleteListener`.
pub struct DialogCompleteListener {
    /// Java private final `client`.
    client: Option<&'static dyn DialogCompleteClient>,
    /// Java private final `string`.
    string: Option<String>,
}

impl DialogCompleteListener {
    /// Java `DialogCompleteListener(DialogCompleteClient, String)`.  `client` is the
    /// instance to be informed when msgDialogComplete is called; `string` is whatever
    /// information needs to be passed back to the client.
    pub fn new(
        client: Option<&'static dyn DialogCompleteClient>,
        string: Option<&str>,
    ) -> DialogCompleteListener {
        DialogCompleteListener {
            client,
            string: string.map(str::to_owned),
        }
    }

    /// Java `msgDialogComplete()`.
    pub fn msg_dialog_complete(&self) {
        if let Some(client) = self.client {
            client.msg_dialog_complete(self.string.as_deref());
        }
    }
}
