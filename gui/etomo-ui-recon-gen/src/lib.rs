//! Appearance-only Slint mirrors of eTomo's reconstruction back-end Swing
//! dialogs, one `ui/<Name>.slint` per Java source:
//!
//! * `IMOD/Etomo/src/etomo/ui/swing/TomogramGenerationDialog.java`
//! * `IMOD/Etomo/src/etomo/ui/swing/TomogramCombinationDialog.java`
//! * `IMOD/Etomo/src/etomo/ui/swing/PostProcessingDialog.java`
//! * `IMOD/Etomo/src/etomo/ui/swing/CleanUpDialog.java`
//!
//! The panels each dialog nests are mirrored alongside them, from their own
//! Java classes: `TiltPanel`, `MultifiltPanel`, `SirtPanel`, `Ctf3dPanel`
//! (generation); `SetupCombinePanel`, `InitialCombinePanel`,
//! `FinalCombinePanel` (combination); `TrimvolPanel`, `FlattenVolumePanel`,
//! `SqueezeVolPanel`, `AltStackPanel`, `SubtomogramsPanel` (post processing);
//! and `CleanupPanel` (clean up).
//!
//! Appearance only: no callbacks, no process launching, no manager state and
//! no file I/O.  Every field carries a static default.  Wiring comes later.
#![allow(clippy::all)]

slint::include_modules!();
