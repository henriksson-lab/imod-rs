//! `IMOD/Etomo/src/etomo/ui/swing/TabbedTextWindow.java`.
//!
//! `JFrame`, `JTabbedPane`, `JEditorPane`, and `UIHarness` are retained as
//! named boundaries.  File selection and the source memory policy remain in
//! this source-shaped unit.
#![allow(dead_code)]

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use crate::imod::etomo::r#type::axis_id::AxisID;

/// Direct `EtomoDirector.INSTANCE.getAvailableMemory()` boundary.
pub trait EtomoDirectorBoundary {
    fn get_available_memory(&self) -> i64;
}

/// Direct `UIHarness.INSTANCE.openMessageDialog(...)` boundary.
pub trait TabbedTextWindowManager {
    fn open_message_dialog(&mut self, message: String, title: &str);

    fn open_message_dialog_with_axis(&mut self, message: String, title: &str, axis_id: AxisID);
}

/// Source-observable `JFrame`/`JTabbedPane`/`JEditorPane` state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TabbedTextWindowBoundary {
    pub title: String,
    pub size: (i32, i32),
    pub dispose_on_close: bool,
    pub tabs: Vec<(String, String, bool)>,
}

/// Java final `TabbedTextWindow`.
pub struct TabbedTextWindow {
    pub axis_id: AxisID,
    pub label: String,
    pub display_whole_log: bool,
    pub display_residuals: bool,
    pub display_solutions: bool,
    pub display_everything_else: bool,
    pub boundary: TabbedTextWindowBoundary,
}

impl TabbedTextWindow {
    /// Java `TabbedTextWindow(String, AxisID)`.
    pub fn new(label: String, axis_id: AxisID) -> Self {
        Self {
            axis_id,
            label,
            display_whole_log: true,
            display_residuals: true,
            display_solutions: true,
            display_everything_else: true,
            boundary: TabbedTextWindowBoundary::default(),
        }
    }

    /// Java `openFiles(BaseManager, String[], String[], AxisID)`.
    pub fn open_files<M: TabbedTextWindowManager, D: EtomoDirectorBoundary>(
        &mut self,
        manager: &mut M,
        files: &[impl AsRef<Path>],
        labels: &[String],
        axis_id: AxisID,
        etomo_director: &D,
        current_font_size: i32,
    ) -> io::Result<bool> {
        self.check_size(files, etomo_director);
        let mut error = String::new();

        if self.display_everything_else {
            self.boundary.title = self.label.clone();
            self.boundary.size = (current_font_size * 625 / 12, current_font_size * 800 / 12);
            self.boundary.dispose_on_close = true;
        } else {
            error.push_str("Unable to display log files:  ");
        }

        for (i, file) in files.iter().enumerate() {
            let file = file.as_ref();
            let file_name = file
                .file_name()
                .and_then(|file_name| file_name.to_str())
                .unwrap_or_default();
            if self.display_everything_else {
                let display_file = if file_name.starts_with("align") {
                    self.display_whole_log
                } else if file_name.starts_with("taResiduals") {
                    self.display_residuals
                } else if file_name.starts_with("taSolution") {
                    self.display_solutions
                } else {
                    true
                };
                let mut editor_pane = String::new();
                self.display(display_file, &mut editor_pane, file)?;
                self.boundary
                    .tabs
                    .push((labels[i].clone(), editor_pane, false));
            } else {
                error.push_str(file_name);
                if i < files.len() - 1 {
                    error.push_str(", ");
                }
            }
        }
        if !self.display_everything_else {
            error.push_str(".  Not enough available memory.  Close unnecessary windows.");
            manager.open_message_dialog_with_axis(error, "Memory Limitation", axis_id);
        }
        Ok(self.display_everything_else)
    }

    /// Java private `checkSize(String[])`.
    pub fn check_size<D: EtomoDirectorBoundary>(
        &mut self,
        files: &[impl AsRef<Path>],
        etomo_director: &D,
    ) {
        let mut whole_log_size = 0_i64;
        let mut residuals_size = 0_i64;
        let mut solutions_size = 0_i64;
        for file in files {
            let file = file.as_ref();
            let file_name = file
                .file_name()
                .and_then(|file_name| file_name.to_str())
                .unwrap_or_default();
            if file_name.starts_with("align") {
                whole_log_size = file
                    .metadata()
                    .map(|metadata| metadata.len() as i64)
                    .unwrap_or(0);
                if whole_log_size <= 200 * 1024 {
                    return;
                }
            } else if file_name.starts_with("taResiduals") {
                residuals_size = file
                    .metadata()
                    .map(|metadata| metadata.len() as i64)
                    .unwrap_or(0);
            } else if file_name.starts_with("taSolution") {
                solutions_size = file
                    .metadata()
                    .map(|metadata| metadata.len() as i64)
                    .unwrap_or(0);
            }
            if whole_log_size > 0 && residuals_size > 0 && solutions_size > 0 {
                break;
            }
        }
        self.display_whole_log = false;
        let everything_else_size = whole_log_size - residuals_size - solutions_size;
        if residuals_size > 1024 * 1024 {
            self.display_residuals = false;
            residuals_size = 0;
        }
        let memory = etomo_director.get_available_memory();
        let danger_area = 15 * 1024 * 1024;
        let mut available = memory - std::cmp::max(danger_area, everything_else_size * 3);
        let overhead = 8;
        if available >= overhead * (residuals_size + solutions_size + everything_else_size) {
            return;
        }
        if self.display_residuals {
            self.display_residuals = false;
            if available >= overhead * (solutions_size + everything_else_size) {
                return;
            }
        }
        self.display_solutions = false;
        available = memory - danger_area;
        if available >= overhead * everything_else_size {
            return;
        }
        self.display_everything_else = false;
    }

    /// Java private `display(boolean, JEditorPane, File)`.
    pub fn display(
        &mut self,
        display_file: bool,
        editor_pane: &mut String,
        file: &Path,
    ) -> io::Result<()> {
        if display_file {
            let mut reader = File::open(file)?;
            reader.read_to_string(editor_pane)?;
        } else {
            *editor_pane = format!(
                "{} is too large to display",
                file.file_name()
                    .and_then(|file_name| file_name.to_str())
                    .unwrap_or_default()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Director(i64);

    impl EtomoDirectorBoundary for Director {
        fn get_available_memory(&self) -> i64 {
            self.0
        }
    }

    #[derive(Default)]
    struct Manager(Vec<(String, String, AxisID)>);

    impl TabbedTextWindowManager for Manager {
        fn open_message_dialog(&mut self, _: String, _: &str) {}

        fn open_message_dialog_with_axis(&mut self, message: String, title: &str, axis_id: AxisID) {
            self.0.push((message, title.into(), axis_id));
        }
    }

    #[test]
    fn check_size_turns_off_every_display_group_when_memory_is_too_low() {
        let mut window = TabbedTextWindow::new("Logs".into(), AxisID::Only);
        let file =
            std::env::temp_dir().join(format!("align-tabbed-text-window-{}", std::process::id()));
        File::create(&file)
            .unwrap()
            .set_len(200 * 1024 + 1)
            .unwrap();
        window.check_size(&[file.as_path()], &Director(0));
        std::fs::remove_file(file).unwrap();
        assert!(!window.display_whole_log);
        assert!(!window.display_residuals);
        assert!(!window.display_solutions);
        assert!(!window.display_everything_else);
    }

    #[test]
    fn open_files_reports_memory_limitation_when_everything_is_disabled() {
        let mut window = TabbedTextWindow::new("Logs".into(), AxisID::Only);
        window.display_everything_else = false;
        let mut manager = Manager::default();
        assert!(
            !window
                .open_files(
                    &mut manager,
                    &[Path::new("align.log")],
                    &["Align".into()],
                    AxisID::Only,
                    &Director(0),
                    12,
                )
                .unwrap()
        );
        assert_eq!(manager.0[0].1, "Memory Limitation");
    }
}
