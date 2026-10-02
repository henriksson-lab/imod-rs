//! `IMOD/Etomo/src/etomo/ui/swing/Run3dmodButtonContainer.java`.
//!
//! The dialog or panel that owns 3dmod buttons and runs their actions.
//! Buttons hold their container as a `Weak<dyn Run3dmodButtonContainer>`:
//! the container owns the buttons, and usually hands itself over while it is
//! still being constructed.

use std::rc::Rc;

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;

use super::deferred_3dmod_button::Deferred3dmodButton;

/// Java `Run3dmodButtonContainer`.
pub trait Run3dmodButtonContainer {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );
}
