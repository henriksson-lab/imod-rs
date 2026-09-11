//! Slint reproductions of the eTomo Swing dialogs for the Join and PEET
//! interfaces:
//!
//! * `IMOD/Etomo/src/etomo/ui/swing/JoinDialog.java`
//! * `IMOD/Etomo/src/etomo/ui/swing/PeetDialog.java`
//! * `IMOD/Etomo/src/etomo/ui/swing/PeetStartupDialog.java`
//!
//! Appearance only: no callbacks, no process launching, no manager or state
//! objects, no file I/O.  Every field carries a static default.  Wiring comes
//! later.
#![allow(clippy::all)]

slint::include_modules!();
