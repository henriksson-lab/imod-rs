//! The eTomo batchruntomo and serial-sections dialogs, redrawn in Slint.
//!
//! One `ui/<DialogName>.slint` per Swing dialog under
//! `IMOD/Etomo/src/etomo/ui/swing`:
//!
//! * `BatchRunTomoDialog.java`        -> `ui/BatchRunTomoDialog.slint`
//! * `BatchRunTomoDatasetDialog.java` -> `ui/BatchRunTomoDatasetDialog.slint`
//! * `SerialSectionsDialog.java`      -> `ui/SerialSectionsDialog.slint`
//! * `SerialSectionsStartupDialog.java` -> `ui/SerialSectionsStartupDialog.slint`
//!
//! The panels those dialogs embed from neighbouring Swing classes
//! (`AutoAlignmentPanel`, `TransformChooserPanel`, `BatchRunTomoTable`,
//! `BatchRunTomoStepPanel`, `TemplatePanel`) are drawn inline in the file of
//! the dialog that owns them, since only these four dialogs are in scope here.
//! `SeriesWatcherPanel.java` is the exception: it gets its own
//! `ui/SeriesWatcherPanel.slint`, imported by `ui/BatchRunTomoDialog.slint`.
//!
//! Widgets the Java constructs but hides or conditionally inserts at
//! construction time are kept in the layout behind an `in property <bool>`
//! whose default is that construction-time visibility, so no label string or
//! layout slot is lost.
//!
//! Appearance only: no callbacks, no manager or state objects, no process
//! launching and no file I/O.  Every field carries the static default the Java
//! constructor sets.  Wiring comes later.
#![allow(clippy::all)]

slint::include_modules!();
