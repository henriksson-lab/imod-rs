//! eTomo's secondary windows and popups — the frames and menus it opens
//! beside the main window — rendered in Slint.
//!
//! One `ui/<Name>.slint` per Java source under
//! `IMOD/Etomo/src/etomo/ui/swing`:
//!
//! * `ContextPopup.java`       — the right-mouse-button help menu.  Its eleven
//!   constructors build eleven different menu shapes out of the same six
//!   blocks; the component reproduces each one, selected by `shape`.
//! * `LogWindow.java`          — the "Project Log" frame: a menu bar and a
//!   bordered scrolling text area.  Its `LogFrame` inner class is the frame
//!   itself and is translated in that file.
//! * `TabbedTextWindow.java`   — the tabbed log-file frame `ContextPopup`
//!   opens for a log file *set*.
//! * `TextPageWindow.java`     — a plain monospaced text page in a frame.
//! * `HTMLPageWindow.java`     — the same frame with an `HTMLEditorKit`.
//! * `MainFrame_AboutBox.java` — the "About" dialog, plus its
//!   `AboutActionListener` inner class.
//! * `Popup.java`              — the configurable `JOptionPane` dialog, in
//!   each of the four shapes its factory methods produce.
//!
//! `ui/WindowWidgets.slint` carries the widget vocabulary this family needs
//! that `gui/etomo-ui-common` does not: `EtomoPanel`, `EtchedBorder`, `Menu`,
//! `MenuItem`, and the plain Swing containers `JMenuBar`, `JPopupMenu`,
//! `JPopupMenuSeparator`, `JScrollPane`, `JEditorPane`, `JTextArea`,
//! `JTabbedPane` and `JOptionPane`.
//!
//! Appearance only: this crate carries no behaviour, no callbacks, no process
//! launching and no manager state.  Every field holds a static default.
//! Wiring comes later.
#![allow(clippy::all)]

slint::include_modules!();
