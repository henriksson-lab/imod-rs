//! `IMOD/Etomo/src/etomo/ui/swing/Run3dmodButtonContainer.java`.
#![allow(dead_code)]
use super::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;

/// Java `Run3dmodButtonContainer`.
pub trait Run3dmodButtonContainer {
    fn action(
        &mut self,
        action_command: &str,
        deferred_3dmod_button: Option<&mut dyn Deferred3dmodButton>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
    );
}
