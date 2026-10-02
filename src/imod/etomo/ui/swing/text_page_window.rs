//! `IMOD/Etomo/src/etomo/ui/swing/TextPageWindow.java`.
//!
//! A frame showing one text file (a log) read-only.  `TextPageWindow extends
//! JFrame`; `jdk.rs` has no frame, so the `JFrame` state this class sets and
//! its callers read back is modelled here, as `LogWindow` models its own
//! `LogFrame`: the content pane (a `JComponent` panel holding the scroll pane
//! and editor pane, so the bridge and test driver can reach the text), the
//! title, the size, the visibility, the default close operation, and
//! `Window.getWindows()` (the JDK's list of top-level windows, here the
//! `TextPageWindow`s shown and not disposed, which is how a presentation finds
//! a window a caller made visible).  Fonts and painting are not modelled (`jdk.rs`).
//!
//! An event-dispatch-thread object (`Rc`, `&self` methods).
//!
//! **Constructor.**  Java's no-argument constructor takes the font size of
//! the first `FontUIResource` in `UIManager.getDefaults()`; the look and feel
//! is not modelled, so the caller passes that size (`UIParameters`' default,
//! which is what the look and feel is set from).

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;

use super::ui_harness;
use crate::imod::etomo::jdk::{Dimension, JComponent};
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// `javax.swing.WindowConstants.DISPOSE_ON_CLOSE`.
pub const DISPOSE_ON_CLOSE: i32 = 2;

thread_local! {
    /// `Window.getWindows()`: the windows shown and not yet disposed.
    static WINDOWS: RefCell<Vec<Rc<TextPageWindow>>> = const { RefCell::new(Vec::new()) };
}

/// Java `public class TextPageWindow extends JFrame`.
pub struct TextPageWindow {
    /// Java package-private `mainPanel` (the frame's content pane).
    main_panel: Rc<JComponent>,
    /// Java package-private `filename`.
    filename: RefCell<Option<String>>,
    /// Java package-private `editorPane = new JEditorPane()`.
    editor_pane: Rc<JComponent>,
    /// Java package-private `scrollPane = new JScrollPane(editorPane)`.
    scroll_pane: Rc<JComponent>,

    // --- the JFrame state this translation models ---
    /// `Frame.setTitle` / `getTitle`.
    title: RefCell<String>,
    /// `Window.isVisible`.
    visible: Cell<bool>,
    /// `Component.setSize` / `getSize`.
    size: Cell<Dimension>,
    /// `JFrame.setDefaultCloseOperation`.
    default_close_operation: Cell<i32>,
    /// `Window.dispose` has run.
    disposed: Cell<bool>,
}

impl TextPageWindow {
    /// Java `TextPageWindow()`; `current_font_size` is the font size Java
    /// reads from `UIManager` (see the module comment).
    pub fn new(current_font_size: i32) -> Rc<TextPageWindow> {
        let editor_pane = JComponent::new_text_area();
        let scroll_pane = JComponent::new_scroll_pane(Some(&editor_pane));
        // java.util.Enumeration keys = UIManager.getDefaults().keys(); ... the
        // first FontUIResource's family and size: the size is the argument.
        // Swing painting: editorPane.setFont(new Font("monospaced", Font.PLAIN,
        // currentFontSize)).
        // editorPane.setEditorKit(new StyledEditorKit());
        let main_panel = JComponent::new_panel();
        // A JFrame is created invisible.
        main_panel.set_visible(false);
        let this = Rc::new(TextPageWindow {
            main_panel,
            filename: RefCell::new(None),
            editor_pane,
            scroll_pane,
            title: RefCell::new(String::new()),
            visible: Cell::new(false),
            size: Cell::new(Dimension {
                width: 0,
                height: 0,
            }),
            default_close_operation: Cell::new(1),
            disposed: Cell::new(false),
        });
        // mainPanel = getContentPane(); mainPanel.add(scrollPane, BorderLayout.CENTER)
        this.main_panel.add(&this.scroll_pane);
        this.set_size(current_font_size * 625 / 12, current_font_size * 800 / 12);
        this.set_default_close_operation(DISPOSE_ON_CLOSE);
        this
    }

    /// Java `setFile(File)`.
    pub fn set_file_from_file(&self, file: &Path) -> bool {
        *self.filename.borrow_mut() = Some(utilities::java_io_file_get_absolute_path(
            &file.to_string_lossy(),
        ));
        self.set_title(&utilities::java_io_file_get_name(&file.to_string_lossy()));
        self.set_file()
    }

    /// Java package-private `setFile(String)`.
    pub fn set_file_from_file_name(&self, file_name: String) -> bool {
        self.set_title(&file_name);
        *self.filename.borrow_mut() = Some(file_name);
        self.set_file()
    }

    /// Java package-private `setFile()`.
    pub fn set_file(&self) -> bool {
        let filename = self
            .filename
            .borrow()
            .clone()
            .unwrap_or_else(|| "null".to_owned());
        // reader = new FileReader(filename); editorPane.read(reader, filename)
        match std::fs::read(&filename) {
            Ok(bytes) => {
                // FileReader decodes with the platform charset; a malformed
                // sequence becomes U+FFFD, as Java's decoder does.
                self.editor_pane.set_text(&String::from_utf8_lossy(&bytes));
                self.editor_pane.set_editable(false);
                // reader.close()
            }
            Err(except) if except.kind() == std::io::ErrorKind::NotFound => {
                // catch (FileNotFoundException except)
                let messages = [
                    format!("{filename} (No such file or directory)"),
                    format!("Make sure that {filename} is available"),
                ];
                ui_harness::open_message_dialog_from_process(
                    None,
                    &messages.join("\n"),
                    &format!("{filename} not found"),
                    None,
                );
                return false;
            }
            Err(except) => {
                // catch (IOException except)
                ui_harness::open_message_dialog_from_process(
                    None,
                    &except.to_string(),
                    &format!("{filename} IO Exception"),
                    None,
                );
                return false;
            }
        }
        true
    }

    /// `Window.setVisible(boolean)` (JFrame).  A shown window is held by
    /// the window list (as AWT holds a displayable frame) until it is
    /// disposed, so a caller may drop its reference.
    pub fn set_visible(self: &Rc<Self>, visible: bool) {
        if visible {
            self.disposed.set(false);
            WINDOWS.with(|windows| {
                let mut windows = windows.borrow_mut();
                if !windows.iter().any(|window| Rc::ptr_eq(window, self)) {
                    windows.push(Rc::clone(self));
                }
            });
        }
        self.visible.set(visible);
        self.main_panel.set_visible(visible);
    }

    /// `Window.isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.visible.get()
    }

    /// `Frame.setTitle(String)`.
    pub fn set_title(&self, title: &str) {
        *self.title.borrow_mut() = title.to_owned();
    }

    /// `Frame.getTitle()`.
    pub fn get_title(&self) -> String {
        self.title.borrow().clone()
    }

    /// `Component.setSize(int, int)`.
    pub fn set_size(&self, width: i32, height: i32) {
        self.size.set(Dimension { width, height });
    }

    /// `Component.getSize()`.
    pub fn get_size(&self) -> Dimension {
        self.size.get()
    }

    /// `JFrame.setDefaultCloseOperation(int)`.
    pub fn set_default_close_operation(&self, operation: i32) {
        self.default_close_operation.set(operation);
    }

    /// `JFrame.getDefaultCloseOperation()`.
    pub fn get_default_close_operation(&self) -> i32 {
        self.default_close_operation.get()
    }

    /// `JFrame.getContentPane()` (Java's `mainPanel`).
    pub fn get_content_pane(&self) -> Rc<JComponent> {
        Rc::clone(&self.main_panel)
    }

    /// The text the editor pane shows (`JEditorPane.getText()`).
    pub fn get_text(&self) -> String {
        self.editor_pane.get_text()
    }

    /// `Window.dispose()`: hides the window and drops it from
    /// [`get_windows`].
    pub fn dispose(&self) {
        self.visible.set(false);
        self.main_panel.set_visible(false);
        self.disposed.set(true);
        WINDOWS.with(|windows| {
            windows
                .borrow_mut()
                .retain(|window| !std::ptr::eq(Rc::as_ptr(window), self))
        });
    }

    /// The user closing the frame (`WindowEvent.WINDOW_CLOSING`): with
    /// `DISPOSE_ON_CLOSE` the frame is disposed, with `HIDE_ON_CLOSE` (1) it
    /// is hidden, with `DO_NOTHING_ON_CLOSE` (0) nothing happens.
    pub fn process_window_closing(&self) {
        match self.default_close_operation.get() {
            DISPOSE_ON_CLOSE => self.dispose(),
            1 => {
                self.visible.set(false);
                self.main_panel.set_visible(false);
            }
            _ => {}
        }
    }
}

/// `Window.getWindows()`, restricted to this class: every `TextPageWindow` on
/// this thread shown and not yet disposed.
pub fn get_windows() -> Vec<Rc<TextPageWindow>> {
    WINDOWS.with(|windows| windows.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_file_reads_text_and_shows_frame_state() {
        let file = std::env::temp_dir().join(format!("text-page-window-{}", std::process::id()));
        std::fs::write(&file, "Etomo text page").unwrap();
        let window = TextPageWindow::new(12);
        assert_eq!(
            window.get_size(),
            Dimension {
                width: 625,
                height: 800
            }
        );
        assert!(window.set_file_from_file(&file));
        window.set_visible(true);
        std::fs::remove_file(&file).unwrap();
        assert_eq!(window.get_text(), "Etomo text page");
        assert!(window.is_visible());
        assert!(get_windows().iter().any(|w| Rc::ptr_eq(w, &window)));
        window.process_window_closing();
        assert!(!window.is_visible());
        assert!(!get_windows().iter().any(|w| Rc::ptr_eq(w, &window)));
    }
}
