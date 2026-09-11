//! The shared widget vocabulary of eTomo's Swing UI
//! (`IMOD/Etomo/src/etomo/ui/swing`), expressed once in Slint so that every
//! per-manager dialog crate spells a labelled field, a check box or a panel
//! header the same way the Java does.
//!
//! Appearance only: this crate carries no behaviour, no process launching and
//! no manager state.  Wiring comes later.
#![allow(clippy::all)]

slint::include_modules!();
