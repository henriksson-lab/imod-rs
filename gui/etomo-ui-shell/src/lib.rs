//! The application-shell dialogs of eTomo's Swing UI, expressed in Slint.
//!
//! One `ui/<Name>.slint` per Java source under
//! `IMOD/Etomo/src/etomo/ui/swing`:
//!
//! * `FrontPageDialog.java` — the default display, buttons for choosing one of
//!   the interfaces.
//! * `SettingsDialog.java` — the "Etomo Settings" dialog.
//! * `ToolsDialog.java` — the top-level dialog of `ToolsManager`.
//! * `AnisotropicDiffusionDialog.java` — the NAD (`nad_eed_3d`) dialog.
//! * `ParallelDialog.java` — the generic parallel process dialog.
//! * `UtilitiesDialog.java` — an empty stub class upstream.
//!
//! Appearance only: this crate carries no behaviour, no callbacks, no process
//! launching and no manager state.  Every field holds a static default.
//! Wiring comes later.
#![allow(clippy::all)]

slint::include_modules!();
