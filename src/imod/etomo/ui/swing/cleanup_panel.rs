//! `IMOD/Etomo/src/etomo/ui/swing/CleanupPanel.java`.
//!
//! Java `final class CleanupPanel`: the Intermediate File Cleanup box of the
//! Clean Up dialog - an embedded file chooser listing the dataset's
//! intermediate files, the directory size, and the Delete Selected / Rescan
//! Directory buttons.
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by
//! [`CleanupPanel::get_instance`]; every method takes `&self`.  The listener
//! class `ButtonActonListener` is a closure holding a weak reference to the
//! panel.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::file_chooser;
use super::multi_line_button::MultiLineButton;
use super::tooltip_formatter;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, FileFilter, JComponent, JFileChooser};
use crate::imod::etomo::storage::file_filter_collection::FileFilterCollection;
use crate::imod::etomo::storage::intermediate_file_filter::IntermediateFileFilter;
use crate::imod::etomo::storage::sirt_output_file_filter::SirtOutputFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::util::utilities;

/// Java public static final `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `final class CleanupPanel`.
pub struct CleanupPanel {
    /// Java private final `pnlCleanup = new EtomoPanel()`.
    pnl_cleanup: Rc<EtomoPanel>,
    /// Java private final `instructions`.
    instructions: Rc<JComponent>,
    /// Java private final `pnlButton = new JPanel()`.
    pnl_button: Rc<JComponent>,
    /// Java private final `btnDelete`.
    btn_delete: Rc<MultiLineButton>,
    /// Java private final `btnRescanDir`.
    btn_rescan_dir: Rc<MultiLineButton>,
    /// Java private final `fileChooser = new JFileChooser()`.
    file_chooser: Rc<JFileChooser>,
    /// Java private final `lDirSize = new JLabel()`.
    l_dir_size: Rc<JComponent>,
    /// Java private final `applicationManager`.
    application_manager: &'static ApplicationManager,
}

impl CleanupPanel {
    /// Java private constructor `CleanupPanel(ApplicationManager)`, with the
    /// field initializers.
    fn new(app_mgr: &'static ApplicationManager) -> Rc<CleanupPanel> {
        let manager: &'static dyn BaseManager = app_mgr;
        let this = Rc::new(CleanupPanel {
            pnl_cleanup: EtomoPanel::new(),
            instructions: JComponent::new_label(
                "Select files to be deleted then press the \"Delete Selected\" button. Ctrl-A selects all displayed files.",
            ),
            pnl_button: JComponent::new_panel(),
            btn_delete: MultiLineButton::new_string(Some("Delete Selected")),
            btn_rescan_dir: MultiLineButton::new_string(Some("Rescan Directory")),
            // Java `new JFileChooser()`: the stand-in is the eTomo FileChooser.
            file_chooser: JFileChooser::new_void(),
            l_dir_size: JComponent::new_label(""),
            application_manager: app_mgr,
        });
        // Create the filechooser
        let meta_data = this.application_manager.get_meta_data();
        let dataset_name = meta_data.get_dataset_name();
        // Collect the file filters
        let file_filter_collection = FileFilterCollection::new();
        let image_filename_style = ConstMetaData::get_image_filename_style(meta_data);
        let intermediate_file_filter =
            IntermediateFileFilter::get_instance(app_mgr, Some(&dataset_name));
        let trimmed_tomogram = Path::new(
            &this
                .application_manager
                .get_property_user_dir()
                .unwrap_or_else(|| "null".to_string()),
        )
        .join(
            file_type::CLASS
                .trim_vol_output
                .get_file_name(Some(manager), Some(AxisID::Only))
                .unwrap_or_else(|| "null".to_string()), /*was: datasetName + ".rec"*/
        );
        if trimmed_tomogram.exists() {
            intermediate_file_filter.set_accept_pretrimmed_tomograms();
        }
        file_filter_collection.add_file_filter(intermediate_file_filter as Rc<dyn FileFilter>);
        if ConstMetaData::get_axis_type(app_mgr.get_meta_data()) == AxisType::DualAxis {
            file_filter_collection.add_file_filter(SirtOutputFileFilter::get_instance(
                manager,
                Some(image_filename_style),
                AxisID::First,
                true,
                true,
                true,
            ) as Rc<dyn FileFilter>);
            file_filter_collection.add_file_filter(SirtOutputFileFilter::get_instance(
                manager,
                Some(image_filename_style),
                AxisID::Second,
                true,
                true,
                true,
            ) as Rc<dyn FileFilter>);
        } else {
            file_filter_collection.add_file_filter(SirtOutputFileFilter::get_instance(
                manager,
                Some(image_filename_style),
                AxisID::Only,
                true,
                true,
                true,
            ) as Rc<dyn FileFilter>);
        }
        // Setup the file chooser
        this.file_chooser
            .set_dialog_type(file_chooser::CUSTOM_DIALOG);
        this.file_chooser
            .set_file_filter(Some(file_filter_collection as Rc<dyn FileFilter>));
        this.file_chooser.set_multi_selection_enabled(true);
        this.file_chooser.set_control_buttons_are_shown(false);
        this.file_chooser.set_current_directory(Some(Path::new(
            &this
                .application_manager
                .get_property_user_dir()
                .unwrap_or_else(|| "null".to_string()),
        )));

        // Swing layout: pnlCleanup.setLayout(new BoxLayout(pnlCleanup,
        // BoxLayout.Y_AXIS)).
        this.pnl_cleanup
            .set_border(&EtchedBorder::new(Some("Intermediate File Cleanup")).get_border());
        // Swing layout: instructions.setAlignmentX(Component.CENTER_ALIGNMENT).
        let pnl_cleanup = this.pnl_cleanup.get_component();
        pnl_cleanup.add(&this.instructions);
        // Swing layout: pnlCleanup.add(Box.createRigidArea(FixedDim.x0_y5)).
        // Swing layout: lDirSize.setAlignmentX(Component.CENTER_ALIGNMENT).
        pnl_cleanup.add(&this.l_dir_size);
        // Swing layout: pnlCleanup.add(Box.createRigidArea(FixedDim.x0_y10)).
        pnl_cleanup.add(&this.file_chooser.get_component());

        // Swing layout: pnlButton X_AXIS BoxLayout, horizontal glue around and
        // between the buttons.
        this.pnl_button.add(&this.btn_delete.get_component());
        this.pnl_button.add(&this.btn_rescan_dir.get_component());
        // Swing layout: pnlCleanup.add(Box.createRigidArea(FixedDim.x0_y10)).
        pnl_cleanup.add(&this.pnl_button);
        // Swing layout: pnlCleanup.add(Box.createRigidArea(FixedDim.x0_y10)).
        this.set_tool_tip_text();
        this
    }

    /// Java package-private static `getInstance(ApplicationManager)`.
    pub fn get_instance(manager: &'static ApplicationManager) -> Rc<CleanupPanel> {
        let instance = CleanupPanel::new(manager);
        instance.set_dir_size();
        instance.add_listeners();
        instance
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_cleanup.get_component()
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        // Java `new ButtonActonListener(this)`.
        let listenee = Rc::downgrade(self);
        let button_action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(listenee) = listenee.upgrade() {
                listenee.button_action(event);
            }
        });
        self.btn_delete
            .add_action_listener(button_action_listener.clone());
        self.btn_rescan_dir
            .add_action_listener(button_action_listener);
    }

    /// Java private `setDirSize()`.
    fn set_dir_size(&self) {
        // `new File(propertyUserDir).listFiles()`: null when the directory
        // cannot be read.
        let file_list: Option<Vec<PathBuf>> = std::fs::read_dir(
            self.application_manager
                .get_property_user_dir()
                .unwrap_or_else(|| "null".to_string()),
        )
        .ok()
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .collect()
        });
        let mut dir_size: i64 = 0;
        if let Some(file_list) = &file_list {
            for file in file_list {
                if file.is_file() {
                    // `File.length()`: 0 when unavailable.
                    dir_size += std::fs::metadata(file)
                        .map(|metadata| metadata.len() as i64)
                        .unwrap_or(0);
                }
            }
        }
        self.l_dir_size
            .set_text(&format!("Directory size (MB): {}", dir_size / 1000000));
    }

    /// Java private `deleteSelected()`.
    fn delete_selected(&self) {
        let mut deleted_all = true;
        let delete_list = self.file_chooser.get_selected_files();
        for file in &delete_list {
            // `File.delete()` removes a file or an empty directory.
            let deleted = if file.is_dir() {
                std::fs::remove_dir(file).is_ok()
            } else {
                std::fs::remove_file(file).is_ok()
            };
            if !deleted {
                deleted_all = false;
            }
        }
        // if (deletedAll) {
        self.file_chooser.rescan_current_directory();
        self.file_chooser.set_selected_file(Some(Path::new("")));
        if !deleted_all {
            let mut message = String::from("Unable to delete file(s).  Check file permissions.");
            if utilities::is_windows_os() {
                message.push_str("\nIf the files are open in 3dmod, close 3dmod.");
            }
            let manager: &'static dyn BaseManager = self.application_manager;
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(manager),
                    &message,
                    "Unable to delete intermediate file",
                    Some(AxisID::Only),
                )
            });
        }
        self.set_dir_size();
    }

    /// Java private `buttonAction(ActionEvent)`.
    fn button_action(&self, event: &ActionEvent) {
        if event.get_action_command() == self.btn_delete.get_action_command().as_deref() {
            self.delete_selected();
        }
        if event.get_action_command() == self.btn_rescan_dir.get_action_command().as_deref() {
            self.file_chooser.rescan_current_directory();
        }
    }

    /// Java private `setToolTipText()`.  Initialize the tooltip text.
    fn set_tool_tip_text(&self) {
        self.file_chooser.set_tool_tip_text(
            tooltip_formatter::INSTANCE
                .format(Some("The list of files in this text box will be deleted."))
                .as_deref(),
        );
        self.btn_delete.set_tool_tip_text(Some(
            "Delete the files listed in the \"File name\" text box.",
        ));
        self.btn_rescan_dir.set_tool_tip_text(Some(
            "Read the directory again to update the list in the file selection box.",
        ));
    }
}
