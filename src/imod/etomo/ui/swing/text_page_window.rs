//! `IMOD/Etomo/src/etomo/ui/swing/TextPageWindow.java`.
//!
//! The JFrame, editor pane, scroll pane, and `JOptionPane` calls remain
//! source-observable UI state.  The source file-reading control flow is kept
//! in this unit.
#![allow(dead_code)]

use std::io;
use std::path::Path;

/// Java `WindowConstants.DISPOSE_ON_CLOSE` selected by the constructor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextPageWindowCloseOperation {
    DisposeOnClose,
}

/// Java `JEditorPane` fields touched by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextPageWindowEditorPane {
    pub font_family: String,
    pub font_size: i32,
    pub text: String,
    pub editable: bool,
}

/// Java `JScrollPane(editorPane)` relationship.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextPageWindowScrollPane {
    pub view_is_editor_pane: bool,
}

/// Java `JOptionPane.showMessageDialog` state produced by `setFile()`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextPageWindowMessageDialog {
    pub messages: Vec<String>,
    pub title: String,
}

/// Java `TextPageWindow`, including inherited JFrame state used by its source.
pub struct TextPageWindow {
    pub rcsid: &'static str,
    pub main_panel_has_scroll_pane_in_center: bool,
    pub filename: Option<String>,
    pub editor_pane: TextPageWindowEditorPane,
    pub scroll_pane: TextPageWindowScrollPane,
    pub reader_filename: Option<String>,
    pub size: (i32, i32),
    pub title: String,
    pub default_close_operation: TextPageWindowCloseOperation,
    pub message_dialogs: Vec<TextPageWindowMessageDialog>,
}

impl TextPageWindow {
    /// Java `TextPageWindow()`.
    pub fn new(current_font_size: i32) -> Self {
        Self {
            rcsid: "$Id$",
            main_panel_has_scroll_pane_in_center: true,
            filename: None,
            editor_pane: TextPageWindowEditorPane {
                font_family: "monospaced".into(),
                font_size: current_font_size,
                text: String::new(),
                editable: true,
            },
            scroll_pane: TextPageWindowScrollPane {
                view_is_editor_pane: true,
            },
            reader_filename: None,
            size: (current_font_size * 625 / 12, current_font_size * 800 / 12),
            title: String::new(),
            default_close_operation: TextPageWindowCloseOperation::DisposeOnClose,
            message_dialogs: Vec::new(),
        }
    }

    /// Java `setFile(File)`.
    pub fn set_file_from_file(&mut self, file: &Path) -> bool {
        self.filename = Some(file.to_string_lossy().into_owned());
        self.title = file
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .into();
        self.set_file()
    }

    /// Java package-private `setFile(String)`.
    pub fn set_file_from_file_name(&mut self, file_name: String) -> bool {
        self.filename = Some(file_name.clone());
        self.title = file_name;
        self.set_file()
    }

    /// Java package-private `setFile()`.
    pub fn set_file(&mut self) -> bool {
        let filename = self.filename.clone().unwrap_or_default();
        self.reader_filename = Some(filename.clone());
        match std::fs::read_to_string(&filename) {
            Ok(text) => {
                self.editor_pane.text = text;
                self.editor_pane.editable = false;
                true
            }
            Err(except) if except.kind() == io::ErrorKind::NotFound => {
                self.message_dialogs.push(TextPageWindowMessageDialog {
                    messages: vec![
                        except.to_string(),
                        format!("Make sure that {filename} is available"),
                    ],
                    title: format!("{filename} not found"),
                });
                false
            }
            Err(except) => {
                self.message_dialogs.push(TextPageWindowMessageDialog {
                    messages: vec![except.to_string()],
                    title: format!("{filename} IO Exception"),
                });
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_matches_swing_setup() {
        let window = TextPageWindow::new(12);
        assert_eq!(window.size, (625, 800));
        assert!(window.main_panel_has_scroll_pane_in_center);
        assert!(window.scroll_pane.view_is_editor_pane);
        assert_eq!(window.editor_pane.font_family, "monospaced");
        assert!(window.editor_pane.editable);
    }

    #[test]
    fn set_file_reads_text_and_makes_editor_uneditable() {
        let file = std::env::temp_dir().join(format!("text-page-window-{}", std::process::id()));
        std::fs::write(&file, "Etomo text page").unwrap();
        let mut window = TextPageWindow::new(12);
        assert!(window.set_file_from_file(&file));
        std::fs::remove_file(file).unwrap();
        assert_eq!(window.editor_pane.text, "Etomo text page");
        assert!(!window.editor_pane.editable);
    }

    #[test]
    fn missing_file_displays_source_error_message() {
        let mut window = TextPageWindow::new(12);
        assert!(!window.set_file_from_file_name("not-present-etomo-page".into()));
        assert_eq!(
            window.message_dialogs[0].title,
            "not-present-etomo-page not found"
        );
        assert_eq!(window.message_dialogs[0].messages.len(), 2);
    }
}
