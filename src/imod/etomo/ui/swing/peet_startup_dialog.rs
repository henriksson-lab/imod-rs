//! `IMOD/Etomo/src/etomo/ui/swing/PeetStartupDialog.java`.
//!
//! Modal dialog for collecting the values needed to create the PEET dialog: the
//! project directory, an optional `.epe`/`.prm` file to copy the project from, and
//! the base name.  The `JDialog` is `jdk::JDialog`; its `setVisible(true)` does not
//! block (see that type), so the code the Java runs after `display()` returns is
//! handed to [`PeetStartupDialog::after_display`].  Swing layout (`BoxLayout`,
//! `GridLayout`, rigid areas, preferred widths) is recorded as `// Swing layout:`
//! comments.

use std::any::Any;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use super::busy_status_panel::BusyStatusPanel;
use super::check_box::CheckBox;
use super::file_chooser;
use super::file_text_field_interface::FileTextFieldInterface;
use super::file_text_field2::FileTextField2;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::result_listener::ResultListener;
use super::swing_component::SwingComponent;
use super::ui_harness;
use crate::imod::etomo::jdk::{
    self, ActionEvent, ActionListener, FileFilter, JComponent, JDialog, WindowListener,
};
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::logic::peet_startup_data::PeetStartupData;
use crate::imod::etomo::peet_manager::PeetManager;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::storage::peet_and_matlab_param_file_filter::PeetAndMatlabParamFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java private `COPY_FROM_LABEL`.
const COPY_FROM_LABEL: &str = "Copy project from ";
/// Java private `NAME`.
pub const NAME: &str = "Starting PEET";

/// Java `public final class PeetStartupDialog implements UIComponent, SwingComponent`.
pub struct PeetStartupDialog {
    this: Weak<PeetStartupDialog>,
    pnl_root: Rc<JComponent>,
    cb_copy_from: Rc<CheckBox>,
    ltf_base_name: Rc<LabeledTextField>,
    btn_ok: Rc<MultiLineButton>,
    btn_cancel: Rc<MultiLineButton>,
    ftf_directory: Rc<FileTextField2>,
    ftf_copy_from: Rc<FileTextField2>,
    dialog: Rc<JDialog>,
    axis_id: AxisID,
    manager: &'static PeetManager,
    busy_status_panel: Rc<BusyStatusPanel>,
}

impl PeetStartupDialog {
    /// Java private `PeetStartupDialog(PeetManager, AxisID)`.
    fn new(manager: &'static PeetManager, axis_id: AxisID) -> Rc<PeetStartupDialog> {
        let pnl_root = JComponent::new_panel();
        let busy_status_panel = BusyStatusPanel::get_instance(manager, AxisID::Only);
        let ftf_directory =
            FileTextField2::get_peet_instance(Some(manager), Some(axis_id), Some("Directory: "));
        let ftf_copy_from =
            FileTextField2::get_unlabeled_peet_instance(Some(manager), Some(COPY_FROM_LABEL));
        // `new JDialog(UIHarness.INSTANCE.getFrame(manager), NAME, true)`.
        let dialog = JDialog::new(NAME, true);
        dialog.set_default_close_operation(jdk::DO_NOTHING_ON_CLOSE);
        let field_type = UITestFieldType::PANEL;
        let name = utilities::convert_label_to_name(Some(NAME), field_type.is_unlimited_segments());
        pnl_root.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        Rc::new_cyclic(|this| PeetStartupDialog {
            this: this.clone(),
            pnl_root,
            cb_copy_from: CheckBox::new_string(Some(COPY_FROM_LABEL)),
            ltf_base_name: LabeledTextField::new_field_type_string(
                FieldType::String,
                Some("Base name: "),
            ),
            btn_ok: MultiLineButton::new_string(Some("OK")),
            btn_cancel: MultiLineButton::new_string(Some("Cancel")),
            ftf_directory,
            ftf_copy_from,
            dialog,
            axis_id,
            manager,
            busy_status_panel,
        })
    }

    /// Java static `getInstance(PeetManager, AxisID)`.
    pub fn get_instance(manager: &'static PeetManager, axis_id: AxisID) -> Rc<PeetStartupDialog> {
        let instance = PeetStartupDialog::new(manager, axis_id);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        let pnl_label = JComponent::new_panel();
        let pnl_label_x = JComponent::new_panel();
        let pnl_data = JComponent::new_panel();
        let pnl_data_x = JComponent::new_panel();
        let pnl_directory = JComponent::new_panel();
        let pnl_copy_from = JComponent::new_panel();
        let pnl_buttons = JComponent::new_panel();
        let pnl_status = JComponent::new_panel();
        self.ltf_base_name.set_preferred_width(125);
        self.ftf_directory.set_adjusted_field_width(175.0);
        self.ftf_directory
            .set_file_selection_mode(file_chooser::DIRECTORIES_ONLY);
        self.ftf_directory.set_absolute_path(true);
        self.ftf_directory.set_origin_etomo_run_dir(true);
        self.ftf_directory.set_file(Some(PathBuf::from("")));
        self.ftf_copy_from.set_adjusted_field_width(225.0);
        self.ftf_copy_from
            .set_file_filter(Some(Rc::new(PeetAndMatlabParamFileFilter::new())));
        self.ftf_copy_from.set_absolute_path(true);
        self.ftf_copy_from.set_origin_etomo_run_dir(true);
        self.update_display();
        // dialog
        self.dialog.get_content_pane().add(&self.pnl_root);
        // root panel (BoxLayout Y_AXIS; rigid areas 20, 20, 40)
        self.pnl_root.add(&pnl_label_x);
        self.pnl_root.add(&pnl_data_x);
        self.pnl_root.add(&pnl_buttons);
        self.pnl_root.add(&pnl_status);
        // X direction label panel (BoxLayout X_AXIS, a 15 pixel rigid area first)
        pnl_label_x.add(&pnl_label);
        // label panel (GridLayout 2 x 1)
        pnl_label.add(&JComponent::new_label(
            "Each PEET project must reside in its own directory.",
        ));
        pnl_label.add(&JComponent::new_label(
            "Please choose a directory and a base name for output files.",
        ));
        // X direction data panel (BoxLayout X_AXIS, a 15 pixel rigid area first)
        pnl_data_x.add(&pnl_data);
        // data panel (GridLayout 3 x 1, vgap 15)
        pnl_data.add(&pnl_directory);
        pnl_data.add(&pnl_copy_from);
        pnl_data.add(&self.ltf_base_name.get_container());
        // directory panel (BoxLayout X_AXIS, a 142 pixel rigid area after)
        pnl_directory.add(&self.ftf_directory.get_root_panel());
        // copy from panel (BoxLayout X_AXIS, a 15 pixel rigid area after)
        pnl_copy_from.add(&self.cb_copy_from.get_component());
        pnl_copy_from.add(&self.ftf_copy_from.get_root_panel());
        // button panel (BoxLayout X_AXIS, rigid areas 200, 15, 40)
        pnl_buttons.add(&self.btn_ok.get_component());
        pnl_buttons.add(&self.btn_cancel.get_component());
        // Status (BorderLayout, the busy status panel EAST)
        pnl_status.add(&self.busy_status_panel.get_component());
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.dialog
            .add_window_listener(Rc::new(PeetStartupWindowListener {
                dialog: self.this.clone(),
            }));
        let result_listener: Rc<RefCell<dyn ResultListener>> =
            Rc::new(RefCell::new(PeetStartupResultListener {
                dialog: self.this.clone(),
            }));
        self.ftf_copy_from
            .add_result_listener(Some(result_listener));
        let adaptee = self.this.clone();
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action(event.get_action_command().unwrap_or(""));
            }
        });
        self.cb_copy_from
            .add_action_listener(Some(listener.clone()));
        self.btn_ok.add_action_listener(listener.clone());
        self.btn_cancel.add_action_listener(listener);
    }

    /// Java `display()`.
    pub fn display(&self) {
        self.dialog.pack();
        self.dialog.set_visible(true);
    }

    /// The code the Java runs after `display()` returns (the modal
    /// `setVisible(true)` blocks there until the dialog is hidden).
    pub fn after_display(&self, job: Box<dyn FnOnce()>) {
        self.dialog.after_modal_return(job);
    }

    /// The dialog (driver and Slint bridge).
    pub fn get_dialog(&self) -> Rc<JDialog> {
        self.dialog.clone()
    }

    /// Java private `dispose()`.
    fn dispose(&self) {
        self.dialog.set_visible(false);
        self.dialog.dispose();
        self.busy_status_panel.remove_listeners(self.manager);
    }

    /// Java private `getStartupData()`.  Returns filled peet startup data instance.
    /// Does not do validation.
    fn get_startup_data(&self) -> PeetStartupData {
        let mut startup_data = PeetStartupData::new();
        startup_data.set_directory(self.ftf_directory.get_text_void().as_deref());
        if self.cb_copy_from.is_selected() {
            startup_data.set_copy_from(self.ftf_copy_from.get_text_void().as_deref());
        }
        startup_data.set_base_name(self.ltf_base_name.get_text_void().as_deref());
        startup_data
    }

    /// Java private `validate()`.  Dialog validation.  Pops up an error message if
    /// validation fails.
    fn validate(&self) -> bool {
        let quoted = |field: &dyn Field| {
            field
                .get_quoted_label()
                .unwrap_or_else(|| "null".to_owned())
        };
        let mut error_message: Option<String> = None;
        // Directory
        if self.ftf_directory.is_empty() {
            error_message = Some(format!("{} is required.", quoted(&*self.ftf_directory)));
        } else {
            let file = self.ftf_directory.get_file().unwrap_or_default();
            let metadata = std::fs::metadata(&file);
            if metadata.is_err() {
                error_message = Some(format!(
                    "{} must contain a directory which exists.",
                    quoted(&*self.ftf_directory)
                ));
            } else if !file.is_dir() {
                error_message = Some(format!(
                    "{} must contain a directory.",
                    quoted(&*self.ftf_directory)
                ));
            } else if !can_read(&file) {
                error_message = Some(format!(
                    "{} must contain a readable directory.",
                    quoted(&*self.ftf_directory)
                ));
            } else if !can_write(&file) {
                error_message = Some(format!(
                    "{} must contain a writable directory.",
                    quoted(&*self.ftf_directory)
                ));
            }
            // CopyFrom
            else if self.cb_copy_from.is_selected() {
                if self.ftf_copy_from.is_empty() {
                    error_message = Some(format!("{} is required.", quoted(&*self.ftf_copy_from)));
                } else {
                    let file = self.ftf_copy_from.get_file().unwrap_or_default();
                    let filter = PeetAndMatlabParamFileFilter::new();
                    let description = filter
                        .get_description()
                        .unwrap_or_else(|| "null".to_owned());
                    if !file.exists() {
                        error_message = Some(format!(
                            "{} must contain a file which exists: {description}.",
                            quoted(&*self.ftf_copy_from)
                        ));
                    } else if !file.is_file() {
                        error_message = Some(format!(
                            "{} must contain a file: {description}.",
                            quoted(&*self.ftf_copy_from)
                        ));
                    } else if !filter.accept(&file) {
                        error_message = Some(format!(
                            "{} must contain the correct file type: {description}.",
                            quoted(&*self.ftf_copy_from)
                        ));
                    } else if !can_read(&file) {
                        error_message = Some(format!(
                            "{} must contain a readable file.",
                            quoted(&*self.ftf_copy_from)
                        ));
                    }
                    // BaseName
                    else if self.ltf_base_name.is_empty() {
                        error_message =
                            Some(format!("{} is required.", quoted(&*self.ltf_base_name)));
                    }
                }
            }
        }
        let Some(error_message) = error_message else {
            return dataset_tool::validate_dataset_name(
                self.manager,
                Some(self as &dyn crate::imod::etomo::ui::ui_component::UIComponent),
                Some(self.axis_id),
                &self.ftf_directory.get_file().unwrap_or_default(),
                self.ltf_base_name.get_text_void().as_deref(),
                DataFileType::Peet,
                None,
                true,
            );
        };
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_ui_component_string_string_axis_id(
                Some(self.manager),
                Some(self as &dyn UIComponent),
                &error_message,
                "Entry Error",
                Some(self.axis_id),
            )
        });
        false
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        self.ftf_copy_from
            .set_enabled(self.cb_copy_from.is_selected());
    }

    /// Java private `action(String)`.
    fn action(&self, command: &str) {
        if self.cb_copy_from.get_action_command().as_deref() == Some(command) {
            self.update_display();
        } else if self.btn_ok.get_action_command().as_deref() == Some(command) {
            if !self.validate() {
                return;
            }
            let startup_data = self.get_startup_data();
            if let Some(error_message) = startup_data.validate() {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_ui_component_string_string_axis_id(
                        Some(self.manager),
                        Some(self as &dyn UIComponent),
                        &error_message,
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return;
            }
            self.dispose();
            self.manager.set_startup_data(&startup_data);
        } else if self.btn_cancel.get_action_command().as_deref() == Some(command) {
            self.dispose();
            self.manager.cancel_startup();
        }
    }

    /// Java private `windowClosing()`.
    fn window_closing(&self) {
        self.dispose();
        self.manager.cancel_startup();
    }

    /// Java private `processResult(Object)`.
    fn process_result(&self, result_origin: &dyn Any) {
        let is_copy_from = result_origin
            .downcast_ref::<FileTextField2>()
            .is_some_and(|origin| std::ptr::eq(origin, &*self.ftf_copy_from));
        if is_copy_from {
            if !self.ltf_base_name.is_empty() {
                return;
            }
            let from_file = self.ftf_copy_from.get_file();
            self.ltf_base_name.set_text_string(
                utilities::get_stripped_file_name_file(from_file.as_deref()).as_deref(),
            );
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        self.ftf_directory.set_tool_tip_text(Some(
            "The directory which will contain the parameter and project files, logs, intermediate files, and results. Data files can also be located in this directory, but are not required to be.",
        ));
        self.ltf_base_name.set_tool_tip_text(Some(
            "The base name of the output files for the average volumes, the reference volumes, and the transformation parameters.",
        ));
        let tooltip = "Check and fill in an .epe or .prm file to create a new PEET project from an existing parameter or project file, duplicating all parameters except root name and location.";
        self.cb_copy_from.set_tool_tip_text_string(Some(tooltip));
        self.ftf_copy_from.set_tool_tip_text(Some(tooltip));
    }
}

/// `java.io.File.canRead()`.
fn can_read(file: &std::path::Path) -> bool {
    std::fs::metadata(file).is_ok_and(|metadata| {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o444 != 0
    }) && (std::fs::read_dir(file).is_ok() || std::fs::File::open(file).is_ok())
}

/// `java.io.File.canWrite()`.
fn can_write(file: &std::path::Path) -> bool {
    std::fs::metadata(file).is_ok_and(|metadata| !metadata.permissions().readonly())
}

impl SwingComponent for PeetStartupDialog {
    /// Java `getComponent()`: the dialog; its content pane here.
    fn get_component(&self) -> Rc<JComponent> {
        self.dialog.get_content_pane()
    }
}

impl UIComponent for PeetStartupDialog {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    fn get_component(&self) -> Rc<JComponent> {
        self.dialog.get_content_pane()
    }
}

/// Java private static final class `PeetStartupResultListener`.
struct PeetStartupResultListener {
    dialog: Weak<PeetStartupDialog>,
}

impl ResultListener for PeetStartupResultListener {
    fn process_result(&mut self, result_origin: &dyn Any, _init: bool) {
        if let Some(dialog) = self.dialog.upgrade() {
            dialog.process_result(result_origin);
        }
    }
}

/// Java private static final class `PeetStartupWindowListener`.
struct PeetStartupWindowListener {
    dialog: Weak<PeetStartupDialog>,
}

impl WindowListener for PeetStartupWindowListener {
    fn window_closing(&self) {
        if let Some(dialog) = self.dialog.upgrade() {
            dialog.window_closing();
        }
    }
}
