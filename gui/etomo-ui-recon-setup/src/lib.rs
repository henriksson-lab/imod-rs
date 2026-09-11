//! The first three dialogs of eTomo's reconstruction flow, rendered in Slint.
//!
//! One `.slint` file per Swing dialog class, mirroring the Java source:
//!
//! * `ui/SetupDialog.slint`         <- `IMOD/Etomo/src/etomo/ui/swing/SetupDialog.java`
//! * `ui/PreProcessingDialog.slint` <- `IMOD/Etomo/src/etomo/ui/swing/PreProcessingDialog.java`
//! * `ui/CoarseAlignDialog.slint`   <- `IMOD/Etomo/src/etomo/ui/swing/CoarseAlignDialog.java`
//!
//! The dialogs each delegate part of their layout to a panel class, which is
//! translated inline in the dialog's own file, as the Java composes it:
//! `PreProcessingDialog` embeds `CcdEraserXRaysPanel`; `CoarseAlignDialog`
//! embeds `TiltxcorrPanel` (its cross-correlation instance) and
//! `PrenewstPanel`; `SetupDialog` embeds `TemplatePanel`, `TiltAnglePanel` and
//! `AxisProgressPanel`.  The exit-button row of all three comes from
//! `ProcessDialog.java`.
//!
//! Appearance only: this crate carries no behaviour, no callbacks, no process
//! launching and no manager state.  Every field holds a static default.
//! Wiring comes later.
#![allow(clippy::all)]

slint::include_modules!();
