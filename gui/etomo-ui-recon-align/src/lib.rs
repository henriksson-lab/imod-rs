//! The four reconstruction alignment/positioning dialogs of eTomo's Swing UI,
//! expressed in Slint.  Each `ui/<Name>.slint` mirrors the Java class of the
//! same base name under `IMOD/Etomo/src/etomo/ui/swing`:
//!
//! * `FiducialModelDialog.java`
//! * `AlignmentEstimationDialog.java`
//! * `FinalAlignedStackDialog.java`
//! * `TomogramPositioningDialog.java`
//!
//! Two supporting files carry constructs those dialogs need that the shared
//! `etomo-ui-common` vocabulary does not have: `ui/widgets_local.slint`
//! (`EtchedBorder.java`, `RadioTextField.java`, `CheckBoxSpinner.java`,
//! `CheckTextField.java`, `FileTextField2.java`, `SingleLineButton.java` and
//! the exit-button row of `ProcessDialog.java`) and `ui/TiltalignPanel.slint`
//! (`TiltalignPanel.java`, which is the entire body of
//! `AlignmentEstimationDialog`).
//!
//! Appearance only: no callbacks, no process launching, no manager state and
//! no file I/O.  Every field carries a static default value.  Wiring comes
//! later.
#![allow(clippy::all)]

slint::include_modules!();
