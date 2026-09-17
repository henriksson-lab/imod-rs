//! `IMOD/Etomo/src/etomo/ui/swing/LogWindow.java`.
//!
//! Swing's `JFrame`, `JTextArea`, menu bar, and event queue are represented by
//! source-observable state here.  Painting and native dispatch stay at that UI
//! boundary; the project-log file, menu actions, persistence, and logging
//! semantics use the translated storage and interface units directly.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::etched_border::EtchedBorder;
use super::log_interface::{
    BadLocationException, FileReaderRef, FileWriterRef, LogInterface, Loggable,
};
use super::menu::Menu;
use super::menu_item::MenuItem;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::{Handle, LogFile};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::ui::log_properties::LogProperties;
use crate::imod::etomo::util::utilities;

/// Java static `TITLE`.
pub const TITLE: &str = "Project Log";
/// Java static `VISIBLE_DEFAULT`.
pub const VISIBLE_DEFAULT: bool = true;
/// Java static `PREPEND`.
pub const PREPEND: &str = "ProjectLog";

/// Direct `JFrame`/`Container`/`JTextArea`/`JScrollPane` boundary declared by
/// `LogWindow`.  Its fields are the state source methods can observe.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LogWindowBoundary {
    pub title: String,
    pub visible: bool,
    pub location: Option<(i32, i32)>,
    pub location_by_platform: bool,
    pub size: (i32, i32),
    pub repaint_count: usize,
    pub pack_count: usize,
    pub root_panel_added: bool,
    pub root_y_axis_layout: bool,
    pub scroll_pane_added: bool,
    pub text_rows: i32,
    pub text_columns: i32,
    pub text: String,
}

/// Java final `LogWindow`.
pub struct LogWindow {
    pub frame_size_width_property: EtomoNumber,
    pub frame_size_height_property: EtomoNumber,
    pub frame_location_x_property: EtomoNumber,
    pub frame_location_y_property: EtomoNumber,
    pub visible_property: EtomoBoolean2,
    pub menu_bar: Vec<String>,
    pub menu_file: Menu,
    pub menu_save: MenuItem,
    pub menu_view: Menu,
    pub menu_hide: MenuItem,
    pub menu_fit_window: MenuItem,
    pub border: EtchedBorder,
    pub boundary: LogWindowBoundary,
    pub manager: Option<&'static dyn BaseManager>,
    pub file: Option<Arc<Handle>>,
    pub user_dir: Option<PathBuf>,
    pub dataset_name: Option<String>,
    pub changed: bool,
    pub file_failed: bool,
    pub write_failed: bool,
    pub displayed: bool,
    pub allow_primary_logging: bool,
    /// Direct `UIHarness.openMessageDialog` boundary; source keeps displaying
    /// only the first access/write error, and this records those calls.
    pub messages: Vec<(String, String)>,
}

impl LogWindow {
    /// Java private `LogWindow(BaseManager)`.
    pub fn new(manager: Option<&'static dyn BaseManager>) -> Self {
        let mut frame_size_width_property = EtomoNumber::new_with_name("FrameSize.Width");
        frame_size_width_property.set_display_value_int(683);
        let mut frame_size_height_property = EtomoNumber::new_with_name("FrameSize.Height");
        frame_size_height_property.set_display_value_int(230);
        let mut visible_property = EtomoBoolean2::new_with_name("Visible");
        visible_property.set_boolean(VISIBLE_DEFAULT);
        Self {
            frame_size_width_property,
            frame_size_height_property,
            frame_location_x_property: EtomoNumber::new_with_name("FrameLocation.X"),
            frame_location_y_property: EtomoNumber::new_with_name("FrameLocation.Y"),
            visible_property,
            menu_bar: Vec::new(),
            menu_file: Menu::new("File"),
            menu_save: MenuItem::with_mnemonic("Save Log", 83),
            menu_view: Menu::new("View"),
            menu_hide: MenuItem::with_mnemonic("Hide Log Window", 72),
            menu_fit_window: MenuItem::with_mnemonic("Fit Log Window", 70),
            border: EtchedBorder::new(TITLE),
            boundary: LogWindowBoundary {
                text_rows: 10,
                text_columns: 60,
                ..Default::default()
            },
            manager,
            file: None,
            user_dir: None,
            dataset_name: None,
            changed: false,
            file_failed: false,
            write_failed: false,
            displayed: false,
            allow_primary_logging: true,
            messages: Vec::new(),
        }
    }

    /// Java static `getInstance(BaseManager)`.  The caller supplies the
    /// translated director's headless result because a native window cannot be
    /// constructed in that mode.
    pub fn get_instance(manager: Option<&'static dyn BaseManager>, headless: bool) -> Option<Self> {
        if headless {
            return None;
        }
        let mut instance = Self::new(manager);
        instance.create_window();
        instance.add_listeners();
        Some(instance)
    }

    /// Java `getManager()`.
    pub fn get_manager(&self) -> Option<&'static dyn BaseManager> {
        self.manager
    }

    /// Java `getAxisID()`.
    pub fn get_axis_id(&self) -> Option<AxisID> {
        None
    }

    /// Java private `createWindow()`.
    pub fn create_window(&mut self) {
        self.boundary.visible = false;
        self.boundary.title = self.border.get_title().to_owned();
        self.menu_bar = vec!["File".into(), "View".into()];
        self.boundary.root_panel_added = true;
        self.boundary.root_y_axis_layout = true;
        self.boundary.scroll_pane_added = true;
    }

    /// Java private `addListeners()`.
    pub fn add_listeners(&mut self) {
        self.menu_file.menu.name = Some("mn.file".into());
        self.menu_view.menu.name = Some("mn.view".into());
        self.menu_hide.action_listener_count += 1;
        self.menu_save.action_listener_count += 1;
        self.menu_fit_window.action_listener_count += 1;
    }

    /// Java private `action(String)`.
    pub fn action(&mut self, command: &str) {
        if self.menu_save.action_command == command {
            self.menu_save();
        } else if self.menu_hide.action_command == command {
            self.hide();
            self.visible_property.set_boolean(false);
        }
        if self.menu_fit_window.action_command == command {
            self.fit();
        }
    }

    /// Java private `fit()`.
    pub fn fit(&mut self) {
        self.boundary.repaint_count += 1;
        self.boundary.pack_count += 1;
    }

    /// Java final `msgCurrentManagerChanged(boolean, boolean)`.  `test` is the
    /// direct translated `EtomoDirector.INSTANCE.getArguments().isTest()` input.
    pub fn msg_current_manager_changed(
        &mut self,
        current: bool,
        startup_popup_open: bool,
        test: bool,
    ) {
        if current && !startup_popup_open && !test {
            if self.visible_property.is() && !self.boundary.visible {
                self.show();
            }
        } else if self.boundary.visible {
            self.boundary.visible = false;
        }
    }

    /// Java `show()`.
    pub fn show(&mut self) {
        if !self.displayed {
            self.displayed = true;
            if !self.frame_location_x_property.is_null() {
                self.boundary.location = Some((
                    self.frame_location_x_property.get_int(),
                    self.frame_location_y_property.get_int(),
                ));
            } else {
                self.boundary.location_by_platform = true;
            }
            self.boundary.size = (
                self.frame_size_width_property.get_int(),
                self.frame_size_height_property.get_int(),
            );
        }
        self.boundary.visible = true;
        self.visible_property.set_boolean(true);
    }

    /// Java private `hide()`.
    pub fn hide(&mut self) {
        self.boundary.visible = false;
        self.visible_property.set_boolean(false);
    }

    /// Java `showHide()`.
    pub fn show_hide(&mut self) {
        if !self.boundary.visible {
            self.show();
        } else {
            self.hide();
        }
    }

    /// Java synchronized `setTitle(File, BaseMetaData, String)`.
    pub fn set_title(
        &mut self,
        param_file: Option<&Path>,
        meta_data: Option<&dyn BaseMetaData>,
        property_user_dir: Option<&Path>,
    ) {
        if let (Some(_), Some(meta_data), Some(property_user_dir)) =
            (param_file, meta_data, property_user_dir)
        {
            self.dataset_name = meta_data.get_name();
            let dataset_name = self.dataset_name.clone().unwrap_or_default();
            self.border.set_title(&format!("{dataset_name} {TITLE}"));
            self.user_dir = Some(property_user_dir.to_path_buf());
            if self.open_file(None) {
                if self.file.as_ref().is_some_and(|file| file.exists()) {
                    if let Some(file) = &self.file {
                        match file.open_reader() {
                            Ok(Some(reader_id)) => {
                                let mut line_list = Vec::new();
                                loop {
                                    match file.read_line(&reader_id) {
                                        Ok(Some(line)) => line_list.push(line),
                                        Ok(None) => break,
                                        Err(_) => {
                                            self.messages.push((
                                                format!(
                                                    "Unabled to load {}",
                                                    file.get_absolute_path()
                                                ),
                                                "System Error".into(),
                                            ));
                                            break;
                                        }
                                    }
                                }
                                file.close_id(Some(&*reader_id));
                                for line in line_list {
                                    self.append(&(line + "\n"));
                                }
                            }
                            Ok(None) => {}
                            Err(_) => self.messages.push((
                                format!("Unabled to load {}", file.get_absolute_path()),
                                "System Error".into(),
                            )),
                        }
                    }
                } else {
                    if self.allow_primary_logging {
                        self.append(&(utilities::get_date_time_stamp() + "\n"));
                        self.append(&(dataset_name + "\n"));
                        if let Some(user_dir) = &self.user_dir {
                            self.append(&(user_dir.display().to_string() + "\n"));
                        }
                    }
                }
            }
        } else {
            self.dataset_name = None;
            self.border.set_title(TITLE);
            self.file = None;
        }
        self.boundary.title = self.border.get_title().to_owned();
    }

    /// Java package-private `menuSave()`.
    pub fn menu_save(&mut self) {
        self.save_force(true);
    }

    /// Java private synchronized `save(boolean)`.
    pub fn save_force(&mut self, force: bool) {
        if !force && !self.changed {
            return;
        }
        if !self.open_file(Some(AxisID::Only)) {
            return;
        }
        let Some(file) = self.file.clone() else {
            return;
        };
        if !file.is_backedup() {
            let _ = file.double_backup_once();
        } else {
            let _ = file.backup();
        }
        match file.open_writer() {
            Ok(writer_id) => {
                self.changed = false;
                let text = self.boundary.text.trim_end_matches('\n');
                let line_array: Vec<&str> = if text.is_empty() {
                    vec![text]
                } else {
                    text.split('\n').collect()
                };
                for line in line_array {
                    if !line.is_empty() {
                        if line.ends_with('\r') {
                            if line.len() > 1 {
                                let _ = file.write(Some(&line[..line.len() - 2]), &writer_id);
                            }
                        } else {
                            let _ = file.write(Some(line), &writer_id);
                        }
                    }
                    let _ = file.new_line(&writer_id);
                }
                file.close_id(Some(&*writer_id));
            }
            Err(_) => {
                if !self.write_failed {
                    self.write_failed = true;
                    self.messages.push((
                        format!("Unabled to write to file {}", file.get_absolute_path()),
                        "System Error".into(),
                    ));
                }
            }
        }
    }

    /// Java private `openFile(AxisID)`.
    pub fn open_file(&mut self, axis_id: Option<AxisID>) -> bool {
        let (Some(dataset_name), Some(user_dir)) = (&self.dataset_name, &self.user_dir) else {
            return false;
        };
        if self.file.is_none() {
            let file_name = format!("{dataset_name}_project.log");
            let emergency_monitor = self
                .manager
                .map(|manager| manager.get_emergency_monitor(axis_id));
            match LogFile::get_instance_dir(user_dir, &file_name, emergency_monitor) {
                Ok(file) => self.file = Some(file),
                Err(_) => {
                    if !self.file_failed {
                        self.file_failed = true;
                        self.messages.push((
                            format!(
                                "Unabled to access file {file_name} in {}",
                                user_dir.display()
                            ),
                            "System Error".into(),
                        ));
                    }
                    return false;
                }
            }
        }
        true
    }

    /// Java nested `LogWindowKeyListener.keyTyped(KeyEvent)`.
    pub fn key_typed(&mut self) {
        self.msg_changed();
    }

    /// Java nested `LogFrame.processWindowEvent(WindowEvent)`.
    pub fn process_window_event(&mut self, closing: bool) {
        if closing {
            self.action(&self.menu_hide.action_command.clone());
        }
    }

    /// Java nested `MenuActionListener.actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self, command: &str) {
        self.action(command);
    }

    /// Java package-private `getRootPanel()` at the native widget boundary.
    pub fn get_root_panel(&self) -> &LogWindowBoundary {
        &self.boundary
    }
}

/// Native adapter for Java `LogWindowKeyListener`; press/release deliberately
/// have no source side effect, while typing is handled by `key_typed`.
pub struct LogWindowKeyListener;

impl LogWindowKeyListener {
    #[allow(non_snake_case)]
    pub fn keyPressed() {}

    #[allow(non_snake_case)]
    pub fn keyReleased() {}
}

impl LogProperties for LogWindow {
    /// Java `store(Properties, String)`.
    fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let prepend = self.create_prepend(prepend);
        let (width, height) = self.boundary.size;
        if width <= 0 || height <= 0 {
            return;
        }
        let mut width_property = self.frame_size_width_property.clone();
        width_property.set_int(width);
        let mut height_property = self.frame_size_height_property.clone();
        height_property.set_int(height);
        let mut x_property = self.frame_location_x_property.clone();
        x_property.set_int(self.boundary.location.unwrap_or_default().0);
        let mut y_property = self.frame_location_y_property.clone();
        y_property.set_int(self.boundary.location.unwrap_or_default().1);
        width_property.store_with_prepend(props, Some(&prepend));
        height_property.store_with_prepend(props, Some(&prepend));
        x_property.store_with_prepend(props, Some(&prepend));
        y_property.store_with_prepend(props, Some(&prepend));
        self.visible_property
            .store_with_prepend(props, Some(&prepend));
    }

    /// Java `load(Properties, String)`.
    fn load(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        self.frame_size_width_property.reset();
        self.frame_size_height_property.reset();
        self.frame_location_x_property.reset();
        self.frame_location_y_property.reset();
        self.visible_property.set_boolean(VISIBLE_DEFAULT);
        let prepend = self.create_prepend(prepend);
        self.frame_size_width_property
            .load_with_prepend(props, Some(&prepend));
        self.frame_size_height_property
            .load_with_prepend(props, Some(&prepend));
        self.frame_location_x_property
            .load_with_prepend(props, Some(&prepend));
        self.frame_location_y_property
            .load_with_prepend(props, Some(&prepend));
        self.visible_property
            .load_with_prepend(props, Some(&prepend));
    }
}

impl LogWindow {
    /// Java private `createPrepend(String)`.
    pub fn create_prepend(&self, prepend: Option<&str>) -> String {
        match prepend.filter(|value| !value.trim().is_empty()) {
            Some(value) => format!("{value}.{PREPEND}"),
            None => PREPEND.into(),
        }
    }
}

impl LogInterface for LogWindow {
    fn get_manager(&self) -> Option<&'static dyn BaseManager> {
        self.get_manager()
    }
    fn get_axis_id(&self) -> Option<AxisID> {
        self.get_axis_id()
    }
    fn log_message_array(
        &mut self,
        title: Option<&str>,
        _: Option<AxisID>,
        message: Option<&[Option<String>]>,
        msg_id: Option<&str>,
    ) -> bool {
        if self.allow_primary_logging {
            self.append(&(utilities::get_date_time_stamp() + "\n"));
            if let Some(title) = title {
                self.append(&(title.to_owned() + "\n"));
            }
            if let Some(message) = message {
                for line in message.iter().flatten() {
                    self.append(&(line.clone() + "\n"));
                }
            }
        }
        message.is_some_and(|message| {
            msg_id.is_some_and(|id| message.iter().flatten().any(|line| line.contains(id)))
        })
    }
    fn log_message_list(
        &mut self,
        title: Option<&str>,
        _: Option<AxisID>,
        message: Option<&[Option<String>]>,
    ) {
        if self.allow_primary_logging {
            self.append(&(utilities::get_date_time_stamp() + "\n"));
            if let Some(title) = title {
                self.append(&(title.to_owned() + "\n"));
            }
            if let Some(message) = message {
                for line in message.iter().flatten() {
                    self.append(&(line.clone() + "\n"));
                }
            }
        }
    }
    fn log_message_axis_list(&mut self, _: Option<AxisID>, message: Option<&[Option<String>]>) {
        if self.allow_primary_logging {
            self.append(&(utilities::get_date_time_stamp() + "\n"));
            if let Some(message) = message {
                for line in message.iter().flatten() {
                    self.append(&(line.clone() + "\n"));
                }
            }
        }
    }
    fn log_message_loggable(&mut self, loggable: Option<&dyn Loggable>, axis_id: Option<AxisID>) {
        if let Some(loggable) = loggable {
            match loggable.get_log_message() {
                Ok(message) => {
                    self.log_message_list(Some(&loggable.get_name()), axis_id, Some(&message))
                }
                Err(error) => {
                    if self.allow_primary_logging {
                        self.append("Unable to log message:\n");
                        self.append(&(format!("{error:?}") + "\n"));
                    }
                }
            }
        }
    }
    fn log_message_title_axis(&mut self, title: Option<&str>, _: Option<AxisID>) {
        if self.allow_primary_logging {
            self.append(&(utilities::get_date_time_stamp() + "\n"));
            if let Some(title) = title {
                self.append(&(title.to_owned() + "\n"));
            }
        }
    }
    fn log_message(&mut self, message: Option<&str>) {
        if self.allow_primary_logging {
            self.append(&(utilities::get_date_time_stamp() + "\n"));
            if let Some(message) = message {
                self.append(&(message.to_owned() + "\n"));
            }
        }
    }
    fn log_message_secondary(
        &mut self,
        message: Option<&str>,
        timestamp: bool,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    ) {
        if timestamp {
            self.append(&(utilities::get_date_time_stamp() + "\n"));
        }
        if let Some(message) = message {
            self.append(message);
            if newline {
                self.append("\n");
            }
        }
        if let Some(secondary_log) = secondary_log {
            if let Some(message) = message {
                secondary_log.borrow_mut().append(message);
                if newline {
                    secondary_log.borrow_mut().append("\n");
                }
            }
        }
    }
    fn log_message_file(&mut self, file: Option<&Path>, secondary_log: Option<FileWriterRef>) {
        self.log_message_file_newline(file, true, secondary_log);
    }
    fn log_message_file_newline(
        &mut self,
        file: Option<&Path>,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    ) {
        if let Some(file) = file {
            match std::fs::read_to_string(file) {
                Ok(contents) => {
                    self.append(&contents);
                    if newline && !contents.ends_with('\n') {
                        self.append("\n");
                    }
                    if let Some(secondary_log) = secondary_log {
                        secondary_log.borrow_mut().append(&contents);
                    }
                }
                Err(_) => {}
            }
        }
    }
    fn log_message_primary_log(&mut self, reader: Option<FileReaderRef>) {
        if let Some(reader) = reader {
            let mut reader = reader.borrow_mut();
            while reader.is_readable() {
                let Some(line) = reader.read_line() else {
                    break;
                };
                self.append(&(line + "\n"));
            }
        }
    }
    fn save(&mut self) {
        self.save_force(false);
    }
    fn set_allow_primary_logging(&mut self, input: bool) {
        self.allow_primary_logging = input;
    }
    fn is_allow_primary_logging(&self) -> bool {
        self.allow_primary_logging
    }
    fn append(&mut self, line: &str) {
        self.boundary.text.push_str(line);
    }
    fn msg_changed(&mut self) {
        self.changed = true;
    }
    fn get_prev_line_end_offset(&self) -> Result<usize, BadLocationException> {
        self.boundary
            .text
            .rfind('\n')
            .map(|offset| offset + 1)
            .ok_or_else(|| BadLocationException {
                message: "No line end".into(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_and_window_setup_match_source_defaults() {
        let log = LogWindow::get_instance(None, false).unwrap();
        assert_eq!(log.boundary.title, TITLE);
        assert_eq!(log.boundary.size, (0, 0));
        assert_eq!(log.menu_bar, ["File", "View"]);
        assert!(log.boundary.root_y_axis_layout);
        assert_eq!(log.menu_save.action_listener_count, 1);
    }

    #[test]
    fn show_hide_and_manager_transition_preserve_visible_property_rule() {
        let mut log = LogWindow::new(None);
        log.show();
        assert_eq!(log.boundary.size, (683, 230));
        assert!(log.visible_property.is());
        log.msg_current_manager_changed(false, false, false);
        assert!(!log.boundary.visible);
        assert!(log.visible_property.is());
        log.msg_current_manager_changed(true, false, false);
        assert!(log.boundary.visible);
        log.process_window_event(true);
        assert!(!log.visible_property.is());
    }

    #[test]
    fn properties_use_source_prepend_and_skip_an_unshown_frame() {
        let mut log = LogWindow::new(None);
        let mut props = BTreeMap::new();
        log.store(&mut props, Some("manager"));
        assert!(props.is_empty());
        log.show();
        log.boundary.location = Some((4, 9));
        log.store(&mut props, Some("manager"));
        assert_eq!(
            props
                .get("manager.ProjectLog.FrameSize.Width")
                .map(String::as_str),
            Some("683")
        );
        assert_eq!(
            props
                .get("manager.ProjectLog.FrameLocation.Y")
                .map(String::as_str),
            Some("9")
        );
    }

    #[test]
    fn action_commands_match_source_hide_and_fit_order() {
        let mut log = LogWindow::new(None);
        log.show();
        log.action("Fit Log Window");
        assert_eq!(log.boundary.repaint_count, 1);
        log.action("Hide Log Window");
        assert!(!log.boundary.visible);
        assert!(!log.visible_property.is());
    }

    #[test]
    fn logger_interface_keeps_primary_logging_gate_and_text_area_change_state() {
        let mut log = LogWindow::new(None);
        log.set_allow_primary_logging(false);
        log.log_message(Some("not logged"));
        assert!(log.boundary.text.is_empty());
        log.set_allow_primary_logging(true);
        log.log_message_array(
            Some("title"),
            None,
            Some(&[Some("first".into()), None, Some("second".into())]),
            Some("second"),
        );
        assert!(log.boundary.text.contains("title\nfirst\nsecond\n"));
        log.key_typed();
        assert!(log.changed);
    }
}
