//! `IMOD/Etomo/src/etomo/ui/swing/TabbedTextWindow.java`.
//!
//! `final class TabbedTextWindow extends JFrame`: the frame's content pane is
//! the `frame` component, the `JTabbedPane` and each tab's `JEditorPane` are
//! `jdk::JComponent`s (a text area stands in for the editor pane), and the
//! frame's visible/displayable state is held here, as in the other frames of
//! this translation.  An EDT object: created as `Rc<Self>`, `&self` methods,
//! mutable fields in `Cell`s.

use std::cell::{Cell, RefCell};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java final `TabbedTextWindow extends JFrame`.
pub struct TabbedTextWindow {
    /// Java private final `axisID` (stored, not read again).
    axis_id: AxisID,
    /// Java private final `label`.
    label: String,

    /// Java package-private `displayWholeLog = true`.
    pub display_whole_log: Cell<bool>,
    /// Java package-private `displayResiduals = true`.
    pub display_residuals: Cell<bool>,
    /// Java package-private `displaySolutions = true`.
    pub display_solutions: Cell<bool>,
    /// Java package-private `displayEverythingElse = true`.
    pub display_everything_else: Cell<bool>,

    /// The `JFrame`'s content pane (`getContentPane()`).
    content_pane: Rc<JComponent>,
    /// `JFrame` title (`setTitle`).
    title: RefCell<Option<String>>,
    /// `JFrame` visibility (`setVisible`).
    frame_visible: Cell<bool>,
    /// `JFrame` displayability (cleared by `dispose`).
    displayable: Cell<bool>,
}

thread_local! {
    /// Every displayable frame, as `Window.getWindows()` lists them: AWT keeps
    /// a frame until it is disposed.  The Slint bridge draws the visible ones.
    static WINDOWS: RefCell<Vec<Rc<TabbedTextWindow>>> = const { RefCell::new(Vec::new()) };
}

/// The displayable frames, oldest first.
pub fn get_windows() -> Vec<Rc<TabbedTextWindow>> {
    WINDOWS.with(|windows| windows.borrow().clone())
}

impl TabbedTextWindow {
    /// Java package-private `TabbedTextWindow(String, AxisID)`.
    pub fn new(label: String, axis_id: AxisID) -> Rc<TabbedTextWindow> {
        let window = Self::construct(label, axis_id);
        WINDOWS.with(|windows| windows.borrow_mut().push(window.clone()));
        window
    }

    fn construct(label: String, axis_id: AxisID) -> Rc<TabbedTextWindow> {
        Rc::new(TabbedTextWindow {
            axis_id,
            label,
            display_whole_log: Cell::new(true),
            display_residuals: Cell::new(true),
            display_solutions: Cell::new(true),
            display_everything_else: Cell::new(true),
            content_pane: JComponent::new_panel(),
            title: RefCell::new(None),
            frame_visible: Cell::new(false),
            displayable: Cell::new(true),
        })
    }

    /// The frame's content pane, for the test driver and the Slint bridge.
    pub fn get_content_pane(&self) -> Rc<JComponent> {
        self.content_pane.clone()
    }

    /// Java `Frame.getTitle()`.
    pub fn get_title(&self) -> Option<String> {
        self.title.borrow().clone()
    }

    /// Java `Component.setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.frame_visible.set(visible);
        self.content_pane.set_visible(visible);
    }

    /// Java `Component.isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.frame_visible.get()
    }

    /// Java `Window.dispose()`.
    pub fn dispose(&self) {
        WINDOWS.with(|windows| {
            windows
                .borrow_mut()
                .retain(|window| !std::ptr::eq(window.as_ref(), self))
        });
        self.frame_visible.set(false);
        self.content_pane.set_visible(false);
        self.displayable.set(false);
    }

    /// Java `Component.isDisplayable()`.
    pub fn is_displayable(&self) -> bool {
        self.displayable.get()
    }

    /// Java package-private `openFiles(BaseManager, String[], String[], AxisID)
    /// throws IOException, FileNotFoundException`.  Open the array of files.
    /// A `FileNotFoundException` is an `io::Error` of kind `NotFound`.
    pub fn open_files(
        &self,
        manager: Option<&'static dyn BaseManager>,
        files: &[String],
        labels: &[String],
        axis_id: AxisID,
    ) -> io::Result<bool> {
        self.check_size(files);
        let mut error: Option<String> = None;
        let mut tab_pane: Option<Rc<JComponent>> = None;

        // Swing fonts: the loop over UIManager.getDefaults().keys() finds the
        // first FontUIResource's family and size (currentFontFamily,
        // currentFontSize); fonts and sizes are not modelled.

        // DisplayEverythingElse is the last boolean to be turned off if there is a
        // memory limitation, so use to decide if a tabbed pane should be displayed.
        if self.display_everything_else.get() {
            let new_tab_pane = JComponent::new_tabbed_pane();
            let main_panel = &self.content_pane;
            main_panel.add(&new_tab_pane);
            tab_pane = Some(new_tab_pane);
            // setTitle(label)
            *self.title.borrow_mut() = Some(self.label.clone());
            // Swing layout: setSize(currentFontSize * 625 / 12, currentFontSize * 800
            // / 12); setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE).
        } else {
            let mut new_error = String::new();
            new_error.push_str("Unable to display log files:  ");
            error = Some(new_error);
        }

        for i in 0..files.len() {
            let file = PathBuf::from(&files[i]);
            let file_name = utilities::java_io_file_get_name(&files[i]);
            if self.display_everything_else.get() {
                let editor_pane = JComponent::new_text_area();
                // Swing fonts: editorPane.setFont(new Font("monospaced", Font.PLAIN,
                // currentFontSize)).
                let scroll_pane = JComponent::new_scroll_pane(Some(&editor_pane));
                // try { ... } catch (OutOfMemoryError e): the JVM heap limit is
                // not modelled (checkSize has already sized the display against
                // the available memory), so there is no counterpart to the
                // catch's "Out of Memory" dialog and rethrow.
                if let Some(tab_pane) = &tab_pane {
                    tab_pane.add_tab(&labels[i], &scroll_pane);
                }
                if file_name.starts_with("align") {
                    self.display(self.display_whole_log.get(), &editor_pane, &file)?;
                } else if file_name.starts_with("taResiduals") {
                    self.display(self.display_residuals.get(), &editor_pane, &file)?;
                } else if file_name.starts_with("taSolution") {
                    self.display(self.display_solutions.get(), &editor_pane, &file)?;
                } else {
                    self.display(true, &editor_pane, &file)?;
                }
            } else if let Some(error) = error.as_mut() {
                error.push_str(&file_name);
                if i + 1 < files.len() {
                    error.push_str(", ");
                }
            }
        }
        if !self.display_everything_else.get() {
            let mut error = error.unwrap_or_default();
            error.push_str(".  Not enough available memory.  Close unnecessary windows.");
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    manager,
                    &error,
                    "Memory Limitation",
                    Some(axis_id),
                )
            });
        }
        Ok(self.display_everything_else.get())
    }

    /// Java private `checkSize(String[])`.  Decide which files to display.  Do
    /// not display the whole log if it is over 200K.  Do not display the
    /// residual log if it is over 1MB.  Do not display the solutions log if it
    /// is too big to fit into the available memory.  Do not display all the
    /// other logs if they are too big to fit into the available memory.
    fn check_size(&self, files: &[String]) {
        let mut whole_log_size: i64 = 0;
        let mut residuals_size: i64 = 0;
        let mut solutions_size: i64 = 0;
        for i in 0..files.len() {
            let file = Path::new(&files[i]);
            let file_name = utilities::java_io_file_get_name(&files[i]);
            // `File.length()`: 0 when the file does not exist.
            let length = || {
                file.metadata()
                    .map(|metadata| metadata.len() as i64)
                    .unwrap_or(0)
            };
            if file_name.starts_with("align") {
                whole_log_size = length();
                if whole_log_size <= 200 * 1024 {
                    return;
                }
            } else if file_name.starts_with("taResiduals") {
                residuals_size = length();
            } else if file_name.starts_with("taSolution") {
                solutions_size = length();
            }
            if whole_log_size > 0 && residuals_size > 0 && solutions_size > 0 {
                break;
            }
        }
        // Whole log is bigger then 200k
        self.display_whole_log.set(false);
        let everything_else_size = whole_log_size - residuals_size - solutions_size;
        if residuals_size as f64 > 1.0 * 1024.0 * 1024.0 {
            // Residuals log is bigger then 1MB
            self.display_residuals.set(false);
            residuals_size = 0;
        }
        let memory = etomo_director::INSTANCE.get_available_memory();
        // Available is available memory - padding.
        // In linux java begins to get unreliable when it has between 14MB -17MB
        // memory available.
        let danger_area: i64 = 15 * 1024 * 1024;
        let mut available = memory - std::cmp::max(danger_area, everything_else_size * 3);
        // AverageFactor was calculated for bug# 1099 by opening log files and taking
        // the average.
        let overhead: i32 = 8;
        if available >= overhead as i64 * (residuals_size + solutions_size + everything_else_size) {
            return;
        }
        // Not enough space. Turn off residuals if they are still being displayed.
        if self.display_residuals.get() {
            self.display_residuals.set(false);
            if available >= overhead as i64 * (solutions_size + everything_else_size) {
                return;
            }
        }
        // Still not enough space. Turn off solutions.
        self.display_solutions.set(false);
        // Reduce the padding for minimal log file display.
        available = memory - danger_area;
        if available >= overhead as i64 * everything_else_size {
            return;
        }
        // Not enough space. Turn off everything.
        self.display_everything_else.set(false);
    }

    /// Java private `display(boolean, JEditorPane, File) throws IOException,
    /// FileNotFoundException`.  Read a file into the editor pane.  If
    /// displayFile is false, place an error message in the editor pane
    /// instead.
    fn display(
        &self,
        display_file: bool,
        editor_pane: &Rc<JComponent>,
        file: &Path,
    ) -> io::Result<()> {
        if display_file {
            let mut reader = std::fs::File::open(file)?;
            // editorPane.read(reader, file)
            let mut text = String::new();
            reader.read_to_string(&mut text)?;
            editor_pane.set_text(&text);
            // reader.close()
        } else {
            editor_pane.set_text(&format!(
                "{} is too large to display",
                utilities::java_io_file_get_name(&file.to_string_lossy())
            ));
        }
        editor_pane.set_editable(false);
        Ok(())
    }
}
