//! `IMOD/Etomo/src/etomo/ui/swing/CleanupPanel.java`.
//!
//! The Java file chooser and `ApplicationManager` message-dialog dispatch are
//! retained as explicit frontend/application boundaries.  The filesystem walk,
//! deletion order, filter construction decisions, and button routing are the
//! source unit's own logic and run here.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::util::utilities;

use super::etomo_panel::{EtomoPanel, TitledBorder};
use super::multi_line_button::MultiLineButton;

/// Direct `ApplicationManager`, `MetaData`, `FileType`, and `UIHarness`
/// operations performed by this source unit.
pub trait CleanupApplicationManager {
    fn property_user_dir(&self) -> &Path;
    fn dataset_name(&self) -> &str;
    fn image_filename_style(&self) -> ImageFilenameStyle;
    fn axis_type(&self) -> AxisType;
    fn trim_vol_output_file_name(&self, axis_id: AxisID) -> String;
    fn open_message_dialog(&mut self, message: String, title: &str, axis_id: AxisID);
}

/// Java `FileFilterCollection`, `IntermediateFileFilter`, and
/// `SirtOutputFileFilter` state as installed in `JFileChooser`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CleanupFileFilterCollection {
    pub dataset_name: String,
    pub image_filename_style: ImageFilenameStyle,
    pub accept_pretrimmed_tomograms: bool,
    pub sirt_axes: Vec<AxisID>,
    pub sirt_filter_arguments: (bool, bool, bool),
}

/// State at the direct Swing `JFileChooser` boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileChooserBoundary {
    pub custom_dialog: bool,
    pub multi_selection_enabled: bool,
    pub control_buttons_shown: bool,
    pub current_directory: PathBuf,
    pub selected_files: Vec<PathBuf>,
    pub selected_file: Option<PathBuf>,
    pub tooltip: Option<String>,
    pub rescan_count: usize,
    pub file_filter_collection: CleanupFileFilterCollection,
}

/// Java final `CleanupPanel` state.
#[derive(Clone, Debug, PartialEq)]
pub struct CleanupPanel {
    pub pnl_cleanup: EtomoPanel,
    pub instructions: String,
    pub button_component_order: Vec<&'static str>,
    pub btn_delete: MultiLineButton,
    pub btn_rescan_dir: MultiLineButton,
    pub file_chooser: FileChooserBoundary,
    pub dir_size_label: String,
    pub listeners_attached: bool,
}

impl CleanupPanel {
    /// Java private `CleanupPanel(ApplicationManager)` constructor.
    pub fn new<M: CleanupApplicationManager>(application_manager: &M) -> Self {
        let property_user_dir = application_manager.property_user_dir();
        let trimmed_tomogram =
            property_user_dir.join(application_manager.trim_vol_output_file_name(AxisID::Only));
        let axis_type = application_manager.axis_type();
        let mut pnl_cleanup = EtomoPanel::default();
        pnl_cleanup.set_border(TitledBorder {
            title: "Intermediate File Cleanup".to_owned(),
        });
        let mut panel = Self {
            pnl_cleanup,
            instructions: "Select files to be deleted then press the \"Delete Selected\" button. Ctrl-A selects all displayed files.".to_owned(),
            button_component_order: vec!["horizontal-glue", "delete", "horizontal-glue", "rescan-directory", "horizontal-glue"],
            btn_delete: MultiLineButton::new_with_label(Some("Delete Selected")),
            btn_rescan_dir: MultiLineButton::new_with_label(Some("Rescan Directory")),
            file_chooser: FileChooserBoundary {
                custom_dialog: true,
                multi_selection_enabled: true,
                control_buttons_shown: false,
                current_directory: property_user_dir.to_path_buf(),
                selected_files: Vec::new(),
                selected_file: None,
                tooltip: None,
                rescan_count: 0,
                file_filter_collection: CleanupFileFilterCollection {
                    dataset_name: application_manager.dataset_name().to_owned(),
                    image_filename_style: application_manager.image_filename_style(),
                    accept_pretrimmed_tomograms: trimmed_tomogram.exists(),
                    sirt_axes: if axis_type == AxisType::DualAxis {
                        vec![AxisID::First, AxisID::Second]
                    } else {
                        vec![AxisID::Only]
                    },
                    sirt_filter_arguments: (true, true, true),
                },
            },
            dir_size_label: String::new(),
            listeners_attached: false,
        };
        panel.set_tool_tip_text();
        panel
    }

    /// Java static `getInstance(ApplicationManager)`.
    pub fn get_instance<M: CleanupApplicationManager>(application_manager: &M) -> Self {
        let mut instance = Self::new(application_manager);
        instance.set_dir_size();
        instance.add_listeners();
        instance
    }

    /// Java `getContainer()`; native layout installation is the GUI boundary.
    pub fn get_container(&self) -> &EtomoPanel {
        &self.pnl_cleanup
    }

    /// Java private `addListeners()`.
    pub fn add_listeners(&mut self) {
        self.btn_delete.add_action_listener();
        self.btn_rescan_dir.add_action_listener();
        self.listeners_attached = true;
    }

    /// Java private `setDirSize()`.
    pub fn set_dir_size(&mut self) {
        let mut dir_size = 0_u64;
        if let Ok(file_list) = fs::read_dir(&self.file_chooser.current_directory) {
            for file in file_list.flatten() {
                if let Ok(file_type) = file.file_type() {
                    if file_type.is_file() {
                        if let Ok(metadata) = file.metadata() {
                            dir_size += metadata.len();
                        }
                    }
                }
            }
        }
        self.dir_size_label = format!("Directory size (MB): {}", dir_size / 1_000_000);
    }

    /// Java private `deleteSelected()`.
    pub fn delete_selected<M: CleanupApplicationManager>(&mut self, application_manager: &mut M) {
        let mut deleted_all = true;
        for file in &self.file_chooser.selected_files {
            if fs::remove_file(file).is_err() {
                deleted_all = false;
            }
        }
        self.file_chooser.rescan_count += 1;
        self.file_chooser.selected_file = Some(PathBuf::new());
        if !deleted_all {
            let mut message = "Unable to delete file(s).  Check file permissions.".to_owned();
            if utilities::is_windows_os() {
                message.push_str("\nIf the files are open in 3dmod, close 3dmod.");
            }
            application_manager.open_message_dialog(
                message,
                "Unable to delete intermediate file",
                AxisID::Only,
            );
        }
        self.set_dir_size();
    }

    /// Java protected `buttonAction(ActionEvent)`.
    pub fn button_action<M: CleanupApplicationManager>(
        &mut self,
        action_command: Option<&str>,
        application_manager: &mut M,
    ) {
        if action_command == self.btn_delete.get_action_command() {
            self.delete_selected(application_manager);
        }
        if action_command == self.btn_rescan_dir.get_action_command() {
            self.file_chooser.rescan_count += 1;
        }
    }

    /// Java private `setToolTipText()`.
    pub fn set_tool_tip_text(&mut self) {
        self.file_chooser.tooltip =
            Some("The list of files in this text box will be deleted.".to_owned());
        self.btn_delete.set_tool_tip_text(Some(
            "Delete the files listed in the \"File name\" text box.",
        ));
        self.btn_rescan_dir.set_tool_tip_text(Some(
            "Read the directory again to update the list in the file selection box.",
        ));
    }
}

/// Java private static final `ButtonActonListener`.  Event delivery remains at
/// the frontend boundary; its body is `CleanupPanel.buttonAction`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ButtonActonListener;

impl ButtonActonListener {
    /// Java `ButtonActonListener(CleanupPanel)`.
    pub fn new() -> Self {
        Self
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed<M: CleanupApplicationManager>(
        &self,
        listenee: &mut CleanupPanel,
        action_command: Option<&str>,
        application_manager: &mut M,
    ) {
        listenee.button_action(action_command, application_manager);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Manager {
        directory: PathBuf,
        axis_type: AxisType,
        dialogs: Vec<(String, String, AxisID)>,
    }

    impl CleanupApplicationManager for Manager {
        fn property_user_dir(&self) -> &Path {
            &self.directory
        }
        fn dataset_name(&self) -> &str {
            "set"
        }
        fn image_filename_style(&self) -> ImageFilenameStyle {
            ImageFilenameStyle::Mrc
        }
        fn axis_type(&self) -> AxisType {
            self.axis_type
        }
        fn trim_vol_output_file_name(&self, _: AxisID) -> String {
            "trim.rec".into()
        }
        fn open_message_dialog(&mut self, message: String, title: &str, axis: AxisID) {
            self.dialogs.push((message, title.into(), axis));
        }
    }

    #[test]
    fn get_instance_constructs_single_axis_filters_and_directory_size() {
        let directory = std::env::temp_dir().join(format!(
            "imod-cleanup-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("a"), vec![0_u8; 1_000_001]).unwrap();
        let manager = Manager {
            directory: directory.clone(),
            axis_type: AxisType::SingleAxis,
            dialogs: vec![],
        };
        let panel = CleanupPanel::get_instance(&manager);
        assert_eq!(
            panel.file_chooser.file_filter_collection.sirt_axes,
            vec![AxisID::Only]
        );
        assert_eq!(panel.dir_size_label, "Directory size (MB): 1");
        assert!(panel.listeners_attached);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn delete_selected_rescans_clears_selection_and_reports_failure() {
        let directory = std::env::temp_dir().join(format!(
            "imod-cleanup-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&directory).unwrap();
        let remove = directory.join("remove");
        fs::write(&remove, b"x").unwrap();
        let missing = directory.join("missing");
        let mut manager = Manager {
            directory: directory.clone(),
            axis_type: AxisType::DualAxis,
            dialogs: vec![],
        };
        let mut panel = CleanupPanel::get_instance(&manager);
        panel.file_chooser.selected_files = vec![remove.clone(), missing];
        panel.delete_selected(&mut manager);
        assert!(!remove.exists());
        assert_eq!(panel.file_chooser.rescan_count, 1);
        assert_eq!(panel.file_chooser.selected_file, Some(PathBuf::new()));
        assert_eq!(manager.dialogs.len(), 1);
        assert_eq!(
            panel.file_chooser.file_filter_collection.sirt_axes,
            vec![AxisID::First, AxisID::Second]
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
