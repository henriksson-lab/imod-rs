//! eTomo's main application window — the frame that hosts every dialog —
//! rendered in Slint.
//!
//! One `ui/<Name>.slint` per Java source under
//! `IMOD/Etomo/src/etomo/ui/swing`:
//!
//! The frame and the two shared panel bases:
//!
//! * `MainFrame.java`        — the application window: the `EtomoMenu` menu
//!   bar, the `WindowSwitch` tab strip and the root panel that holds the
//!   current manager's `MainPanel`.  `EtomoMenu.java`, `EtomoFrame.java`'s
//!   `initialize`/`getMenus` and `WindowSwitch.java` are translated inline in
//!   that file, with the head comment naming them.
//! * `MainPanel.java`        — the abstract base: the axis scroll pane, the
//!   status bar and the busy-status strip.
//! * `AxisProcessPanel.java` — the abstract base of one axis column: the
//!   process-button column, the progress panel, the parallel-status slot and
//!   the dialog slot.
//!
//! The per-manager main panels, each a `MainPanel` subclass:
//! `MainTomogramPanel.java`, `MainJoinPanel.java`, `MainParallelPanel.java`,
//! `MainFrontPagePanel.java`, `MainPeetPanel.java`, `MainToolsPanel.java`,
//! `MainDirectiveEditorPanel.java`, `MainSerialSectionsPanel.java`,
//! `MainBatchRunTomoPanel.java`.
//!
//! The process panels, each an `AxisProcessPanel` subclass:
//! `TomogramProcessPanel.java`, `JoinProcessPanel.java`,
//! `ParallelProcessPanel.java`, `ToolsProcessPanel.java`,
//! `PeetProcessPanel.java`, `FrontPageProcessPanel.java`,
//! `SerialSectionsProcessPanel.java`, `BatchRunTomoProcessPanel.java`,
//! `DirectiveEditorProcessPanel.java`, `DirectivesProcessPanel.java`.
//!
//! A subclass instantiates its base component and supplies the properties and
//! extra children its Java constructor passes to `super` and appends
//! afterwards, so the Java inheritance is mirrored rather than copy-pasted.
//!
//! `ui/MainWindowWidgets.slint` carries the small Swing widget classes this
//! family needs that `gui/etomo-ui-common` does not: `ScrollPanel`,
//! `ProgressPanel`, `AxisProgressPanel`, `ProcessControlPanel` (with
//! `ColoredStateText`), `SimpleButton`, `SimpleToggleButton`, `Menu`,
//! `MenuItem`, `CheckBoxMenuItem`, the `JMenuBar`, and the `Colors` palette.
//! `Panel.java`, `PagingPanel.java` and `ToolPanel.java` are not referenced by
//! any source in this family and are not translated here.
//!
//! The dialog area that `AxisProcessPanel.panelDialog` hosts is the one
//! legitimately empty slot: those dialogs live in `gui/etomo-ui-recon-setup`,
//! `gui/etomo-ui-recon-align`, `gui/etomo-ui-recon-gen`,
//! `gui/etomo-ui-join-peet`, `gui/etomo-ui-batch-serial` and
//! `gui/etomo-ui-shell`.
//!
//! Appearance only: this crate carries no behaviour, no callbacks, no process
//! launching and no manager state.  Every field holds a static default.
//! Wiring comes later.
#![allow(clippy::all)]

slint::include_modules!();
