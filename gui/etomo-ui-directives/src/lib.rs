//! The two directive-editor dialogs of eTomo, rendered in Slint.
//!
//! One `.slint` file per Swing class, mirroring the Java source:
//!
//! * `ui/DirectiveEditorDialog.slint`  <- `IMOD/Etomo/src/etomo/ui/swing/DirectiveEditorDialog.java`
//! * `ui/DirectivesDialog.slint`       <- `IMOD/Etomo/src/etomo/ui/swing/DirectivesDialog.java`
//!
//! The panel classes the two dialogs construct are translated in their own
//! files, named after the Java class, rather than stood in for by a box:
//!
//! * `ui/DirectiveSectionPanel.slint`    <- `DirectiveSectionPanel.java`   (built by `DirectiveEditorDialog.createPanel`)
//! * `ui/DirectivePanel.slint`           <- `DirectivePanel.java`          (built by `DirectiveSectionPanel.createPanel`)
//! * `ui/DirectivesTable.slint`          <- `DirectivesTable.java`         (built by the `DirectivesDialog` constructor)
//! * `ui/DirectivesSectionRow.slint`     <- `DirectivesSectionRow.java`    (built by `DirectivesTable.RowList.createPanel`)
//! * `ui/DirectivesDirectiveRow.slint`   <- `DirectivesDirectiveRow.java`  (built by `DirectivesTable.RowList.createPanel`)
//!
//! `ui/widgets_local.slint` carries the Swing widget classes these dialogs use
//! that `gui/etomo-ui-common` does not: `EtchedBorder`, `SingleLineButton`,
//! `Ebutton`, `ToggleEbutton`, `CheckBoxEfield`, `TextField`, `ComboBox`,
//! `SimpleButton`, `TextEfield`, `ComboBoxEfield`, `BooleanComboBoxEfield` and
//! `ButtonControlTextEfield`.
//!
//! Appearance only: this crate carries no behaviour, no callbacks, no process
//! launching and no manager state.  Every field holds a static default.  The
//! directive lists themselves come from `IMOD/com/directives.csv` through
//! `DirectiveDescrFile` at run time, so the row models default to empty; the
//! row and section *shapes* are translated in full.  Wiring comes later.
#![allow(clippy::all)]

slint::include_modules!();
