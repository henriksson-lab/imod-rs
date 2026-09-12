//! `IMOD/Etomo/src/etomo/ui/swing/PeetStartupDialog.java`.
//!
//! Native file chooser, dataset-name validation, busy status, and manager
//! startup are explicit boundaries; construction and validation ordering stay
//! with the source dialog.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Control {
    pub text: String,
    pub selected: bool,
    pub enabled: bool,
    pub tooltip: Option<String>,
    pub listener_count: usize,
}
impl Control {
    fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            selected: false,
            enabled: true,
            tooltip: None,
            listener_count: 0,
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileTextField {
    pub label: String,
    pub text: String,
    pub file: Option<PathBuf>,
    pub enabled: bool,
    pub directories_only: bool,
    pub absolute_path: bool,
    pub origin_etomo_run_dir: bool,
    pub tooltip: Option<String>,
    pub result_listener_count: usize,
}
impl FileTextField {
    fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            text: String::new(),
            file: None,
            enabled: true,
            directories_only: false,
            absolute_path: false,
            origin_etomo_run_dir: false,
            tooltip: None,
            result_listener_count: 0,
        }
    }
    fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
    fn quoted_label(&self) -> String {
        format!("'{}'", self.label.trim_end_matches(':'))
    }
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PeetStartupData {
    pub directory: String,
    pub copy_from: Option<String>,
    pub base_name: String,
}
impl PeetStartupData {
    pub fn validate(&self) -> Option<String> {
        if self.base_name.trim().is_empty() {
            Some("Base name is required.".into())
        } else {
            None
        }
    }
}
/// Java `PeetManager` calls from this source unit.
pub trait PeetStartupManager {
    fn validate_dataset_name(&mut self, directory: &Path, base_name: &str) -> bool;
    fn set_startup_data(&mut self, data: PeetStartupData);
    fn cancel_startup(&mut self);
}

/// Java final `PeetStartupDialog`.
pub struct PeetStartupDialog<M: PeetStartupManager> {
    pub pnl_root_name: String,
    pub cb_copy_from: Control,
    pub ltf_base_name: FileTextField,
    pub btn_ok: Control,
    pub btn_cancel: Control,
    pub ftf_directory: FileTextField,
    pub ftf_copy_from: FileTextField,
    pub manager: M,
    pub busy_status_listener_removed: bool,
    pub dialog_visible: bool,
    pub dialog_disposed: bool,
    pub dialog_packed: bool,
    pub window_listener_count: usize,
    pub last_error: Option<String>,
}
impl<M: PeetStartupManager> PeetStartupDialog<M> {
    pub const COPY_FROM_LABEL: &'static str = "Copy project from ";
    pub const NAME: &'static str = "Starting PEET";
    fn new(manager: M) -> Self {
        Self {
            pnl_root_name: "panel.start-peet".into(),
            cb_copy_from: Control::new(Self::COPY_FROM_LABEL),
            ltf_base_name: FileTextField::new("Base name: "),
            btn_ok: Control::new("OK"),
            btn_cancel: Control::new("Cancel"),
            ftf_directory: FileTextField::new("Directory: "),
            ftf_copy_from: FileTextField::new(Self::COPY_FROM_LABEL),
            manager,
            busy_status_listener_removed: false,
            dialog_visible: false,
            dialog_disposed: false,
            dialog_packed: false,
            window_listener_count: 0,
            last_error: None,
        }
    }
    pub fn get_instance(manager: M) -> Self {
        let mut v = Self::new(manager);
        v.create_panel();
        v.set_tooltips();
        v.add_listeners();
        v
    }
    pub fn create_panel(&mut self) {
        self.ftf_directory.directories_only = true;
        self.ftf_directory.absolute_path = true;
        self.ftf_directory.origin_etomo_run_dir = true;
        self.ftf_directory.file = Some(PathBuf::new());
        self.ftf_copy_from.absolute_path = true;
        self.ftf_copy_from.origin_etomo_run_dir = true;
        self.update_display();
    }
    pub fn add_listeners(&mut self) {
        self.window_listener_count += 1;
        self.ftf_copy_from.result_listener_count += 1;
        self.cb_copy_from.listener_count += 1;
        self.btn_ok.listener_count += 1;
        self.btn_cancel.listener_count += 1;
    }
    pub fn display(&mut self) {
        self.dialog_packed = true;
        self.dialog_visible = true;
    }
    pub fn dispose(&mut self) {
        self.dialog_visible = false;
        self.dialog_disposed = true;
        self.busy_status_listener_removed = true;
    }
    pub fn get_startup_data(&self) -> PeetStartupData {
        PeetStartupData {
            directory: self.ftf_directory.text.clone(),
            copy_from: self
                .cb_copy_from
                .selected
                .then(|| self.ftf_copy_from.text.clone()),
            base_name: self.ltf_base_name.text.clone(),
        }
    }
    /// Java `validate()`: caller supplies file-system facts through actual paths.
    pub fn validate(&mut self) -> bool {
        let directory = match self.ftf_directory.file.as_ref() {
            Some(path) if !self.ftf_directory.is_empty() => path,
            _ => {
                self.last_error = Some(format!(
                    "{} is required.",
                    self.ftf_directory.quoted_label()
                ));
                return false;
            }
        };
        if !directory.exists() || !directory.is_dir() {
            self.last_error = Some(format!(
                "{} must contain a directory which exists.",
                self.ftf_directory.quoted_label()
            ));
            return false;
        }
        if self.cb_copy_from.selected {
            let copy = match self.ftf_copy_from.file.as_ref() {
                Some(path) if !self.ftf_copy_from.is_empty() => path,
                _ => {
                    self.last_error = Some(format!(
                        "{} is required.",
                        self.ftf_copy_from.quoted_label()
                    ));
                    return false;
                }
            };
            if !copy.exists() || !copy.is_file() {
                self.last_error = Some(format!(
                    "{} must contain a file which exists.",
                    self.ftf_copy_from.quoted_label()
                ));
                return false;
            }
            if self.ltf_base_name.is_empty() {
                self.last_error = Some(format!(
                    "{} is required.",
                    self.ltf_base_name.quoted_label()
                ));
                return false;
            }
        }
        if !self
            .manager
            .validate_dataset_name(directory, &self.ltf_base_name.text)
        {
            return false;
        }
        true
    }
    pub fn update_display(&mut self) {
        self.ftf_copy_from.enabled = self.cb_copy_from.selected;
    }
    pub fn action(&mut self, command: &str) {
        if command == self.cb_copy_from.text {
            self.update_display()
        } else if command == self.btn_ok.text {
            if !self.validate() {
                return;
            }
            let data = self.get_startup_data();
            if let Some(error) = data.validate() {
                self.last_error = Some(error);
                return;
            }
            self.dispose();
            self.manager.set_startup_data(data)
        } else if command == self.btn_cancel.text {
            self.dispose();
            self.manager.cancel_startup();
        }
    }
    pub fn window_closing(&mut self) {
        self.dispose();
        self.manager.cancel_startup();
    }
    pub fn process_result_copy_from(&mut self) {
        if !self.ltf_base_name.is_empty() {
            return;
        }
        if let Some(path) = self.ftf_copy_from.file.as_ref() {
            self.ltf_base_name.text = path
                .file_stem()
                .map_or_else(String::new, |v| v.to_string_lossy().into_owned());
        }
    }
    pub fn set_tooltips(&mut self) {
        self.ftf_directory.tooltip=Some("The directory which will contain the parameter and project files, logs, intermediate files, and results.".into());
        self.ltf_base_name.tooltip=Some("The base name of the output files for the average volumes, the reference volumes, and the transformation parameters.".into());
        let text="Check and fill in an .epe or .prm file to create a new PEET project from an existing parameter or project file.".to_owned();
        self.cb_copy_from.tooltip = Some(text.clone());
        self.ftf_copy_from.tooltip = Some(text);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Manager {
        startup: Option<PeetStartupData>,
        cancelled: bool,
    }
    impl PeetStartupManager for Manager {
        fn validate_dataset_name(&mut self, _: &Path, _: &str) -> bool {
            true
        }
        fn set_startup_data(&mut self, d: PeetStartupData) {
            self.startup = Some(d)
        }
        fn cancel_startup(&mut self) {
            self.cancelled = true
        }
    }
    #[test]
    fn copy_toggle_controls_copy_file_field() {
        let mut d = PeetStartupDialog::get_instance(Manager::default());
        assert!(!d.ftf_copy_from.enabled);
        d.cb_copy_from.selected = true;
        d.action("Copy project from ");
        assert!(d.ftf_copy_from.enabled)
    }
    #[test]
    fn copy_result_derives_base_name_only_if_empty() {
        let mut d = PeetStartupDialog::get_instance(Manager::default());
        d.ftf_copy_from.file = Some(PathBuf::from("old.epe"));
        d.process_result_copy_from();
        assert_eq!(d.ltf_base_name.text, "old");
        d.ltf_base_name.text = "given".into();
        d.process_result_copy_from();
        assert_eq!(d.ltf_base_name.text, "given")
    }
}
