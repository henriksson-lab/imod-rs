//! `IMOD/Etomo/src/etomo/ui/swing/LogWindow.java`.
//!
//! Text of a project log.  Associated with one manager.  Do not pass the
//! manager when popping up messages.  This could lead to an infinite loop
//! because every message with a manager gets logged.
//!
//! `LogWindow` is an EDT object (`Rc`, `&self` methods).  Its private
//! `LogFrame extends JFrame` is modelled, as `AbstractFrame` models its own
//! `JFrame`, by the frame state the class reads back: visibility, title, menu
//! bar, content pane, location and size.  Layout, painting, key strokes and
//! mnemonics are not modelled (`jdk.rs`); those statements are kept as
//! `// Swing layout:` comments.
//!
//! Java `synchronized` on `setTitle` and `save(boolean)` has no counterpart:
//! both run on the event dispatch thread only.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::path::Path;
use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::time::Duration;

use super::etched_border::EtchedBorder;
use super::etomo_logger::EtomoLogger;
use super::etomo_panel::EtomoPanel;
use super::log_interface::{BadLocationException, LogInterface};
use super::menu::Menu;
use super::menu_item::MenuItem;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, Dimension, JComponent, Point};
use crate::imod::etomo::storage::file_reader::FileReaderRef;
use crate::imod::etomo::storage::file_writer::FileWriterRef;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, WriterId};
use crate::imod::etomo::storage::loggable::Loggable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::ui::log_properties::LogProperties;

/// Java package-private static final `TITLE`.
pub const TITLE: &str = "Project Log";
/// Java private static final `VISIBLE_DEFAULT`.
const VISIBLE_DEFAULT: bool = true;
/// Java private static final `PREPEND`.
const PREPEND: &str = "ProjectLog";

/// `java.awt.event.KeyEvent.VK_S`.
const VK_S: i32 = 83;
/// `java.awt.event.KeyEvent.VK_H`.
const VK_H: i32 = 72;
/// `java.awt.event.KeyEvent.VK_F`.
const VK_F: i32 = 70;

/// `java.awt.event.WindowEvent.WINDOW_CLOSING`.
pub const WINDOW_CLOSING: i32 = 201;

/// Java `public final class LogWindow implements LogInterface, LogProperties`.
pub struct LogWindow {
    /// Java `frameSizeWidthProperty = new EtomoNumber("FrameSize.Width")`.
    frame_size_width_property: RefCell<EtomoNumber>,
    /// Java `frameSizeHeightProperty = new EtomoNumber("FrameSize.Height")`.
    frame_size_height_property: RefCell<EtomoNumber>,
    /// Java `frameLocationXProperty = new EtomoNumber("FrameLocation.X")`.
    frame_location_x_property: RefCell<EtomoNumber>,
    /// Java `frameLocationYProperty = new EtomoNumber("FrameLocation.Y")`.
    frame_location_y_property: RefCell<EtomoNumber>,
    /// Java `visibleProperty = new EtomoBoolean2("Visible")`.
    visible_property: RefCell<EtomoBoolean2>,
    /// Java `menuBar = new JMenuBar()`.
    menu_bar: Rc<JComponent>,
    /// Java `menuFile = new Menu("File")`.
    menu_file: Rc<Menu>,
    /// Java `menuSave = new MenuItem("Save Log", KeyEvent.VK_S)`.
    menu_save: Rc<MenuItem>,
    /// Java `menuView = new Menu("View")`.
    menu_view: Rc<Menu>,
    /// Java `menuHide = new MenuItem("Hide Log Window", KeyEvent.VK_H)`.
    menu_hide: Rc<MenuItem>,
    /// Java `menuFitWindow = new MenuItem("Fit Log Window", KeyEvent.VK_F)`.
    menu_fit_window: Rc<MenuItem>,
    /// Java `rootPanel = new EtomoPanel()`.
    root_panel: Rc<EtomoPanel>,
    /// Java `border = new EtchedBorder(TITLE)`.
    border: EtchedBorder,
    /// Java `textArea = new JTextArea(10, 60)`.
    text_area: Rc<JComponent>,
    /// Java `scrollPane = new JScrollPane(textArea)`.
    scroll_pane: Rc<JComponent>,
    /// Java `logger = new EtomoLogger(this)`.
    logger: EtomoLogger,
    /// Java `frame = new LogFrame(menuHide)`.
    frame: LogFrame,

    /// Java `private final BaseManager manager`.
    manager: Option<&'static dyn BaseManager>,

    /// Java `private LogFile.Handle file = null`.
    file: RefCell<Option<Arc<Handle>>>,
    /// Java `private String userDir = null`.
    user_dir: RefCell<Option<String>>,
    /// Java `private String datasetName = null`.
    dataset_name: RefCell<Option<String>>,
    /// Java `private boolean changed = false`.
    changed: Cell<bool>,
    /// Java `private boolean fileFailed = false`.
    file_failed: Cell<bool>,
    /// Java `private boolean writeFailed = false`.
    write_failed: Cell<bool>,
    /// Java `private boolean displayed = false`.
    displayed: Cell<bool>,
}

impl LogWindow {
    /// Java private `LogWindow(BaseManager)`, with the field initialisers.
    fn new(manager: Option<&'static dyn BaseManager>) -> Rc<LogWindow> {
        let text_area = JComponent::new_text_area();
        // Swing layout: JTextArea(10, 60) rows and columns.
        let scroll_pane = JComponent::new_scroll_pane(Some(&text_area));
        let menu_hide = MenuItem::new_string_int("Hide Log Window", VK_H);
        let instance = Rc::new_cyclic(|this: &Weak<LogWindow>| {
            let primary_log: Weak<dyn LogInterface> = this.clone();
            LogWindow {
                frame_size_width_property: RefCell::new(EtomoNumber::new_with_name(
                    "FrameSize.Width",
                )),
                frame_size_height_property: RefCell::new(EtomoNumber::new_with_name(
                    "FrameSize.Height",
                )),
                frame_location_x_property: RefCell::new(EtomoNumber::new_with_name(
                    "FrameLocation.X",
                )),
                frame_location_y_property: RefCell::new(EtomoNumber::new_with_name(
                    "FrameLocation.Y",
                )),
                visible_property: RefCell::new(EtomoBoolean2::new_with_name("Visible")),
                menu_bar: JComponent::new_other(),
                menu_file: Menu::new("File"),
                menu_save: MenuItem::new_string_int("Save Log", VK_S),
                menu_view: Menu::new("View"),
                menu_hide: menu_hide.clone(),
                menu_fit_window: MenuItem::new_string_int("Fit Log Window", VK_F),
                root_panel: EtomoPanel::new(),
                border: EtchedBorder::new(Some(TITLE)),
                text_area,
                scroll_pane,
                logger: EtomoLogger::new(primary_log),
                frame: LogFrame::new(menu_hide),
                manager,
                file: RefCell::new(None),
                user_dir: RefCell::new(None),
                dataset_name: RefCell::new(None),
                changed: Cell::new(false),
                file_failed: Cell::new(false),
                write_failed: Cell::new(false),
                displayed: Cell::new(false),
            }
        });
        instance
            .frame_size_width_property
            .borrow_mut()
            .set_display_value_int(683);
        instance
            .frame_size_height_property
            .borrow_mut()
            .set_display_value_int(230);
        instance
            .visible_property
            .borrow_mut()
            .set_boolean(VISIBLE_DEFAULT);
        instance
    }

    /// Java static `getInstance(BaseManager)`.
    pub fn get_instance(manager: Option<&'static dyn BaseManager>) -> Option<Rc<LogWindow>> {
        if etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            return None;
        }
        let instance = LogWindow::new(manager);
        instance.create_window();
        instance.add_listeners(&instance);
        Some(instance)
    }

    /// Java private `createWindow()`.
    fn create_window(&self) {
        // init
        self.frame.set_visible(false);
        // frame
        self.frame.set_title(self.border.get_title().as_deref());
        self.frame.set_j_menu_bar(Some(self.menu_bar.clone()));
        // menu
        self.menu_bar.add(&self.menu_file.get_component());
        self.menu_bar.add(&self.menu_view.get_component());
        self.menu_file
            .get_component()
            .add(&self.menu_save.get_component());
        self.menu_view
            .get_component()
            .add(&self.menu_hide.get_component());
        self.menu_view
            .get_component()
            .add(&self.menu_fit_window.get_component());
        // content pane
        let content_pane = self.frame.get_content_pane();
        content_pane.add(&self.root_panel.get_component());
        // root panel
        // Swing layout: rootPanel.setLayout(new BoxLayout(rootPanel, BoxLayout.Y_AXIS)).
        self.root_panel.set_border(&self.border.get_border());
        self.root_panel.get_component().add(&self.scroll_pane);
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self, this: &Rc<LogWindow>) {
        // Mnemonics for the main menu bar
        // Swing key binding: menuFile.setMnemonic(KeyEvent.VK_F);
        // menuView.setMnemonic(KeyEvent.VK_V).
        // Accelerators
        // Swing key binding: menuHide.setAccelerator(CTRL+L);
        // menuFitWindow.setAccelerator(CTRL+F).
        // Bind the menu items to their listeners
        // MenuActionListener actionListener = new MenuActionListener(this);
        let window = Rc::downgrade(this);
        let action_listener: crate::imod::etomo::jdk::ActionListener =
            Rc::new(move |event: &ActionEvent| {
                // MenuActionListener.actionPerformed(ActionEvent)
                if let Some(window) = window.upgrade() {
                    window.action(event.get_action_command());
                }
            });
        self.menu_save.add_action_listener(action_listener.clone());
        self.menu_hide.add_action_listener(action_listener.clone());
        self.menu_fit_window.add_action_listener(action_listener);
        // text area
        // textArea.addKeyListener(new LogWindowKeyListener(this)): key events are
        // not modelled by jdk.rs; `key_typed` below is the listener's only
        // non-empty method, for a frontend to call.
    }

    /// Java `LogWindowKeyListener.keyTyped(KeyEvent)`: `adaptee.msgChanged()`.
    /// (`keyPressed` and `keyReleased` are empty.)
    pub fn key_typed(&self) {
        self.msg_changed();
    }

    /// Java private `action(String)`.
    fn action(&self, command: Option<&str>) {
        if self.menu_save.get_action_command().as_deref() == command {
            self.menu_save();
        } else if self.menu_hide.get_action_command().as_deref() == command {
            self.hide();
            self.visible_property.borrow_mut().set_boolean(false);
        }
        if self.menu_fit_window.get_action_command().as_deref() == command {
            self.fit();
        }
    }

    /// Java private `fit()`.
    fn fit(&self) {
        // Swing layout: frame.repaint(); frame.pack().
        self.frame.pack();
    }

    /// Java public final `msgCurrentManagerChanged(boolean, boolean)`.  Called
    /// when the manager's interface is either displayed or hidden.  Hide the log
    /// window when the interface is hidden.  Remember whether is was visible or
    /// not.  When the interface is made current, show it if it was visible the
    /// last time the interface was displayed.
    pub fn msg_current_manager_changed(&self, current: bool, startup_popup_open: bool) {
        let test = etomo_director::ARGUMENTS.lock().unwrap().is_test();
        if current && !startup_popup_open && !test {
            let visible = self.visible_property.borrow().is();
            if visible && !self.frame.is_visible() {
                self.show();
            }
        } else if self.frame.is_visible() {
            // LogProperties.visible are being kept up to date, so just hide the frame.
            // When the manager is made visible again, the log window will be made
            // visible too.

            // For test don't update the visibility property.
            self.frame.set_visible(false);
        }
    }

    /// Java `show()`.  Makes the window visible.  Use this function instead of
    /// calling JFrame.setVisible(true) directly.  Updates LogProperties.visible.
    /// Sets the location and size the first time it is made visible.
    pub fn show(&self) {
        // try { Thread.sleep(50); } catch (InterruptedException e) {}
        std::thread::sleep(Duration::from_millis(50));
        if !self.displayed.get() {
            self.displayed.set(true);
            let location_x_null = self.frame_location_x_property.borrow().is_null();
            if !location_x_null {
                let x = self.frame_location_x_property.borrow().get_int();
                let y = self.frame_location_y_property.borrow().get_int();
                self.frame.set_location(x, y);
            } else {
                self.frame.set_location_by_platform(true);
            }
            let width = self.frame_size_width_property.borrow().get_int();
            let height = self.frame_size_height_property.borrow().get_int();
            self.frame.set_size(Dimension { width, height });
        }
        self.frame.set_visible(true);
        self.visible_property.borrow_mut().set_boolean(true);
    }

    /// Java private `hide()`.  Makes the window invisible.  Updates
    /// LogProperties.visible.  Do not use this when hiding the log because the
    /// manager interface is being hidden, because the properties should not be
    /// updated in that case.
    fn hide(&self) {
        self.frame.set_visible(false);
        self.visible_property.borrow_mut().set_boolean(false);
    }

    /// Java `showHide()`.
    pub fn show_hide(&self) {
        if !self.frame.is_visible() {
            self.show();
        } else {
            self.hide();
        }
    }

    /// Java package-private synchronized `setTitle(File, BaseMetaData, String)`.
    /// Sets the title.  If paramFile and metaData are set then also sets ready to
    /// true.  Sends a message to the LogFrame.  Reads the file into the text
    /// area.  Must be run with a paramFile and metaData before save() can be run.
    /// If this is a new dataset, sets up a header for the log.
    pub fn set_title(
        &self,
        param_file: Option<&Path>,
        meta_data: Option<&dyn BaseMetaData>,
        property_user_dir: Option<&str>,
    ) {
        if let (Some(meta_data), Some(_)) = (meta_data, param_file) {
            let dataset_name = meta_data.get_name();
            *self.dataset_name.borrow_mut() = dataset_name.clone();
            self.border.set_title(Some(&format!(
                "{} {}",
                dataset_name.as_deref().unwrap_or("null"),
                TITLE
            )));
            *self.user_dir.borrow_mut() = property_user_dir.map(str::to_owned);
            if self.open_file(None) {
                let file = self.file.borrow().clone();
                let file = file.expect("openFile returned true");
                if file.exists() {
                    // `if (file != null)`: always true here.
                    let mut reader_id = None;
                    let result: Result<(), LogFileError> = 'try_block: {
                        reader_id = match file.open_reader() {
                            Ok(reader_id) => reader_id,
                            Err(e) => break 'try_block Err(e),
                        };
                        // Upstream bug fixed in translation (LogWindow.java:257):
                        // openReader can return null, which readLine then
                        // dereferences; a null reader loads nothing.
                        let Some(reader_id) = &reader_id else {
                            break 'try_block Ok(());
                        };
                        let mut line = match file.read_line(reader_id) {
                            Ok(line) => line,
                            Err(e) => break 'try_block Err(e),
                        };
                        let mut line_list: Vec<Option<String>> = Vec::new();
                        while let Some(current) = line {
                            line_list.push(Some(current));
                            line = match file.read_line(reader_id) {
                                Ok(line) => line,
                                Err(e) => break 'try_block Err(e),
                            };
                        }
                        self.logger.load_init_messages(Some(line_list));
                        file.close_id(Some(&**reader_id));
                        Ok(())
                    };
                    match result {
                        Ok(()) => {}
                        Err(LogFileError::Lock(_)) => {}
                        Err(e) => {
                            // e.printStackTrace();
                            eprintln!("{}", e);
                            ui_harness::with(|harness| {
                                harness.open_message_dialog_base_manager_string_string(
                                    None,
                                    &format!("Unabled to load {}", file.get_absolute_path()),
                                    "System Error",
                                )
                            });
                        }
                    }
                    let _ = reader_id;
                } else {
                    let dataset_name = self.dataset_name.borrow().clone();
                    let user_dir = self.user_dir.borrow().clone();
                    self.logger
                        .log_message_string_string(dataset_name.as_deref(), user_dir.as_deref());
                }
            }
        } else {
            *self.dataset_name.borrow_mut() = None;
            self.border.set_title(Some(TITLE));
            *self.file.borrow_mut() = None;
        }
        self.frame.set_title(self.border.get_title().as_deref());
    }

    /// Java package-private `menuSave()`.  Always save.
    pub fn menu_save(&self) {
        self.save_boolean(true);
    }

    /// Java private synchronized `save(boolean)`.  Save textArea to file.
    /// `force` - when true, save even if changed is false.
    fn save_boolean(&self, force: bool) {
        if !force && !self.changed.get() {
            return;
        }
        if !self.open_file(Some(AxisID::Only)) {
            return;
        }
        let file = self.file.borrow().clone().expect("openFile returned true");
        let backup_result = if !file.is_backedup() {
            file.double_backup_once().map(|_| ())
        } else {
            file.backup().map(|_| ())
        };
        match backup_result {
            Ok(()) => {}
            Err(LogFileError::Lock(_)) => {}
            Err(e) => {
                // e.printStackTrace();
                eprintln!("{}", e);
            }
        }
        let mut writer_id: Option<WriterId> = None;
        let result: Result<(), LogFileError> = 'try_block: {
            writer_id = match file.open_writer() {
                Ok(writer_id) => Some(writer_id),
                Err(e) => break 'try_block Err(e),
            };
            let writer = writer_id.as_ref().unwrap();
            self.changed.set(false);
            let text = self.text_area.get_text();
            // LogFile.newLine() will put the appropriate line endings in the file.
            // Strip the line endings by breaking up the text by \n. Strip windows
            // line endings (\r\n) by removing the \r which, if it is in use, will
            // now be at the end of each line. This also adds a new line to the end
            // of the file, if there is not one already there. This should also
            // preserve empty lines.
            // `text.split("\n")`: Java's String.split drops trailing empty
            // strings, and returns the whole string when there is no match.
            let mut line_array: Vec<&str> = if text.contains('\n') {
                text.split('\n').collect()
            } else {
                vec![text.as_str()]
            };
            if text.contains('\n') {
                while line_array.last().is_some_and(|line| line.is_empty()) {
                    line_array.pop();
                }
            }
            for line in &line_array {
                // Preserve an empty line by calling newLine.
                if !line.is_empty() {
                    // Look for Windows line ending.
                    if line.ends_with('\r') {
                        // Preserve an empty line by calling newLine.
                        if line.len() > 1 {
                            // Write a line which has a Windows line ending (strip \r).
                            // Upstream bug fixed in translation
                            // (LogWindow.java:334): Java writes
                            // `substring(0, length() - 2)`, which drops the last
                            // character of the line along with the \r.  Only the
                            // \r is stripped here.
                            if let Err(e) = file.write(Some(&line[..line.len() - 1]), writer) {
                                break 'try_block Err(e);
                            }
                        }
                    } else {
                        // Write a line which has a Linux line ending.
                        if let Err(e) = file.write(Some(line), writer) {
                            break 'try_block Err(e);
                        }
                    }
                }
                if let Err(e) = file.new_line(writer) {
                    break 'try_block Err(e);
                }
            }
            file.close_id(Some(&**writer));
            Ok(())
        };
        match result {
            Ok(()) => {}
            Err(LogFileError::Lock(_)) => {
                if !self.write_failed.get() {
                    self.write_failed.set(true);
                }
            }
            Err(e) => {
                // e.printStackTrace();
                eprintln!("{}", e);
                if !self.write_failed.get() {
                    self.write_failed.set(true);
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string(
                            None,
                            &format!("Unabled to write to file {}", file.get_absolute_path()),
                            "System Error",
                        )
                    });
                    if let Some(writer_id) = &writer_id
                        && !writer_id.is_empty()
                    {
                        file.close_id(Some(&**writer_id));
                    }
                }
            }
        }
    }

    /// Java private `openFile(AxisID)`.  Set file if necessary.  Returns true if
    /// file can be used.
    fn open_file(&self, axis_id: Option<AxisID>) -> bool {
        // Can't open log until the dataset and userDir is set
        let dataset_name = self.dataset_name.borrow().clone();
        let user_dir = self.user_dir.borrow().clone();
        let (Some(dataset_name), Some(user_dir)) = (dataset_name, user_dir) else {
            return false;
        };
        if self.file.borrow().is_none() {
            let file_name = format!("{}_project.log", dataset_name);
            let emergency_monitor = self
                .manager
                .map(|manager| manager.get_emergency_monitor(axis_id));
            match LogFile::get_instance_user_dir(
                &user_dir,
                &format!("{}_project.log", dataset_name),
                emergency_monitor,
            ) {
                Ok(file) => *self.file.borrow_mut() = Some(file),
                Err(e) => {
                    // catch (final FileException | IOException e)
                    // e.printStackTrace();
                    eprintln!("{}", e);
                    if !self.file_failed.get() {
                        self.file_failed.set(true);
                        ui_harness::with(|harness| {
                            harness.open_message_dialog_base_manager_string_string(
                                None,
                                &format!("Unabled to access file {} in {}", file_name, user_dir),
                                "System Error",
                            )
                        });
                    }
                    return false;
                }
            }
        }
        true
    }

    /// Java package-private `getRootPanel()`.
    pub fn get_root_panel(&self) -> Rc<JComponent> {
        self.root_panel.get_component()
    }

    /// Java `store(Properties, String)` (`LogProperties`).
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let prepend = self.create_prepend(prepend);
        let size = self.frame.get_size();
        // A panel that has never been displayed will have a height and width of 0.
        if size.width <= 0 || size.height <= 0 {
            return;
        }
        // update properties
        self.frame_size_width_property
            .borrow_mut()
            .set_int(size.width);
        self.frame_size_height_property
            .borrow_mut()
            .set_int(size.height);
        let point = self.frame.get_location();
        self.frame_location_x_property.borrow_mut().set_int(point.x);
        self.frame_location_y_property.borrow_mut().set_int(point.y);
        // store
        self.frame_size_width_property
            .borrow()
            .store_with_prepend(props, Some(&prepend));
        self.frame_size_height_property
            .borrow()
            .store_with_prepend(props, Some(&prepend));
        self.frame_location_x_property
            .borrow()
            .store_with_prepend(props, Some(&prepend));
        self.frame_location_y_property
            .borrow()
            .store_with_prepend(props, Some(&prepend));
        self.visible_property
            .borrow()
            .store_with_prepend(props, Some(&prepend));
    }

    /// Java `load(Properties, String)` (`LogProperties`).
    pub fn load(&self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        // reset
        self.frame_size_width_property.borrow_mut().reset();
        self.frame_size_height_property.borrow_mut().reset();
        self.frame_location_x_property.borrow_mut().reset();
        self.frame_location_y_property.borrow_mut().reset();
        self.visible_property
            .borrow_mut()
            .set_boolean(VISIBLE_DEFAULT);
        // load
        let prepend = self.create_prepend(prepend);
        self.frame_size_width_property
            .borrow_mut()
            .load_with_prepend(props, Some(&prepend));
        self.frame_size_height_property
            .borrow_mut()
            .load_with_prepend(props, Some(&prepend));
        self.frame_location_x_property
            .borrow_mut()
            .load_with_prepend(props, Some(&prepend));
        self.frame_location_y_property
            .borrow_mut()
            .load_with_prepend(props, Some(&prepend));
        self.visible_property
            .borrow_mut()
            .load_with_default_boolean(props, Some(&prepend), VISIBLE_DEFAULT);
    }

    /// Java private `createPrepend(String)`.
    fn create_prepend(&self, prepend: Option<&str>) -> String {
        match prepend {
            // `prepend.matches("\\s*")`
            None => PREPEND.to_owned(),
            Some(prepend)
                if crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace(
                    prepend,
                ) =>
            {
                PREPEND.to_owned()
            }
            Some(prepend) => format!("{}.{}", prepend, PREPEND),
        }
    }

    /// The `LogFrame`'s `processWindowEvent(WindowEvent)`, for a frontend to
    /// deliver window events (Rust-only entry point to the private frame).
    pub fn process_window_event(&self, event_id: i32) {
        self.frame.process_window_event(event_id);
    }

    /// The frame's visibility (`frame.isVisible()`), for a frontend
    /// (Rust-only accessor to the private frame).
    pub fn is_frame_visible(&self) -> bool {
        self.frame.is_visible()
    }

    /// The frame's title (`frame.getTitle()`), for a frontend (Rust-only
    /// accessor to the private frame).
    pub fn get_frame_title(&self) -> String {
        self.frame.get_title()
    }

    /// The frame's content pane, the root of its component tree, for a frontend
    /// or a uitest search (Rust-only accessor to the private frame).
    pub fn get_frame_content_pane(&self) -> Rc<JComponent> {
        self.frame.get_content_pane()
    }

    /// The frame's menu bar (`frame.getJMenuBar()`), for a uitest search
    /// (Rust-only accessor to the private frame).
    pub fn get_frame_j_menu_bar(&self) -> Option<Rc<JComponent>> {
        self.frame.j_menu_bar.borrow().clone()
    }
}

impl LogInterface for LogWindow {
    /// Java `getManager()`.
    fn get_manager(&self) -> Option<&'static dyn BaseManager> {
        self.manager
    }

    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> Option<AxisID> {
        None
    }

    /// Java `logMessage(String, AxisID, String[], String)`.
    fn log_message_string_axis_id_string_array_string(
        &self,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        message: Option<&[String]>,
        msg_id: Option<&str>,
    ) -> bool {
        self.logger
            .log_message_string_axis_id_string_array_string(title, axis_id, message, msg_id)
    }

    /// Java `logMessage(String, AxisID, ArrayList<String>)`.
    fn log_message_string_axis_id_array_list(
        &self,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        message: Option<&[String]>,
    ) {
        self.logger
            .log_message_string_axis_id_array_list(title, axis_id, message);
    }

    /// Java `logMessage(AxisID, ArrayList<String>)`.
    fn log_message_axis_id_array_list(&self, axis_id: Option<AxisID>, message: Option<&[String]>) {
        self.logger.log_message_axis_id_array_list(axis_id, message);
    }

    /// Java `logMessage(Loggable, AxisID)`.
    fn log_message_loggable_axis_id(
        &self,
        loggable: Option<&dyn Loggable>,
        axis_id: Option<AxisID>,
    ) {
        self.logger.log_message_loggable_axis_id(loggable, axis_id);
    }

    /// Java `logMessage(String, AxisID)`.
    fn log_message_string_axis_id(&self, title: Option<&str>, axis_id: Option<AxisID>) {
        self.logger.log_message_string_axis_id(title, axis_id);
    }

    /// Java `logMessage(String)`.
    fn log_message_string(&self, message: Option<&str>) {
        self.logger.log_message_string(message);
    }

    /// Java `logMessage(String, boolean, boolean, FileWriter)`.
    fn log_message_string_boolean_boolean_file_writer(
        &self,
        message: Option<&str>,
        timestamp: bool,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    ) {
        self.logger.log_message_string_boolean_boolean_file_writer(
            message,
            timestamp,
            newline,
            secondary_log,
        );
    }

    /// Java `logMessage(File, FileWriter)`.
    fn log_message_file_file_writer(
        &self,
        file: Option<&Path>,
        secondary_log: Option<FileWriterRef>,
    ) {
        self.logger
            .log_message_file_file_writer(file, secondary_log);
    }

    /// Java `logMessage(File, boolean, FileWriter)`.
    fn log_message_file_boolean_file_writer(
        &self,
        file: Option<&Path>,
        newline: bool,
        secondary_log: Option<FileWriterRef>,
    ) {
        self.logger
            .log_message_file_boolean_file_writer(file, newline, secondary_log);
    }

    /// Java `logMessagePrimaryLog(FileReader)`.
    fn log_message_primary_log(&self, reader: Option<FileReaderRef>) {
        self.logger.log_message_primary_log(reader);
    }

    /// Java `save()`.  Save if something has changed.
    fn save(&self) {
        self.save_boolean(false);
    }

    /// Java `setAllowPrimaryLogging(boolean)`.
    fn set_allow_primary_logging(&self, input: bool) {
        self.logger.set_allow_primary_logging(input);
    }

    /// Java `isAllowPrimaryLogging()`.
    fn is_allow_primary_logging(&self) -> bool {
        self.logger.is_allow_primary_logging()
    }

    /// Java `append(String)`.
    fn append(&self, line: &str) {
        self.text_area.append(line);
    }

    /// Java `msgChanged()`.
    fn msg_changed(&self) {
        self.changed.set(true);
    }

    /// Java `getPrevLineEndOffset()`:
    /// `textArea.getLineEndOffset(textArea.getLineCount() - 1)`.  The end offset
    /// of the last line is the document length (JTextArea hides the implicit
    /// break at the end of the document); the line count is never 0, so the
    /// BadLocationException is never thrown.
    fn get_prev_line_end_offset(&self) -> Result<usize, BadLocationException> {
        Ok(self.text_area.get_text().encode_utf16().count())
    }
}

impl LogProperties for LogWindow {
    /// Java `store(Properties, String)`.
    fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        LogWindow::store(self, props, prepend);
    }

    /// Java `load(Properties, String)`.
    fn load(&self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        LogWindow::load(self, props, prepend);
    }
}

/// Java private static final `LogFrame extends JFrame`.  Only the `JFrame`
/// state `LogWindow` reads back is modelled (see the module comment).
struct LogFrame {
    /// Java `private final JMenuItem menuHide`.
    menu_hide: Rc<MenuItem>,

    // --- the JFrame state this translation models ---
    /// `JFrame.getContentPane()`.
    content_pane: Rc<JComponent>,
    /// `Frame.setTitle` / `getTitle`.
    title: RefCell<String>,
    /// `Window.isVisible`.
    visible: Cell<bool>,
    /// `JFrame.setJMenuBar`.
    j_menu_bar: RefCell<Option<Rc<JComponent>>>,
    /// `Component.getLocation` / `setLocation`.
    location: Cell<Point>,
    /// `Window.setLocationByPlatform`.
    location_by_platform: Cell<bool>,
    /// `Component.getSize` / `setSize`.
    size: Cell<Dimension>,
}

impl LogFrame {
    /// Java private `LogFrame(JMenuItem)`.
    fn new(menu_hide: Rc<MenuItem>) -> LogFrame {
        let content_pane = JComponent::new_panel();
        // A JFrame is created invisible.
        content_pane.set_visible(false);
        // setDefaultCloseOperation(DO_NOTHING_ON_CLOSE): closing is handled by
        // processWindowEvent.
        LogFrame {
            menu_hide,
            content_pane,
            title: RefCell::new(String::new()),
            visible: Cell::new(false),
            j_menu_bar: RefCell::new(None),
            location: Cell::new(Point { x: 0, y: 0 }),
            location_by_platform: Cell::new(false),
            size: Cell::new(Dimension {
                width: 0,
                height: 0,
            }),
        }
    }

    /// Java `processWindowEvent(WindowEvent)`: overridden so we can hide when
    /// window is closed.
    fn process_window_event(&self, event_id: i32) {
        // super.processWindowEvent(event): window listeners are not modelled.
        if event_id == WINDOW_CLOSING {
            self.menu_hide.do_click();
        }
    }

    /// `Window.setVisible(boolean)`.  The content pane mirrors the frame's
    /// visibility so a component search limited to showing components skips a
    /// hidden frame.
    fn set_visible(&self, visible: bool) {
        self.visible.set(visible);
        self.content_pane.set_visible(visible);
    }

    /// `Window.isVisible()`.
    fn is_visible(&self) -> bool {
        self.visible.get()
    }

    /// `Frame.setTitle(String)`.
    fn set_title(&self, title: Option<&str>) {
        *self.title.borrow_mut() = title.unwrap_or("").to_owned();
    }

    /// `Frame.getTitle()`.
    fn get_title(&self) -> String {
        self.title.borrow().clone()
    }

    /// `JFrame.setJMenuBar(JMenuBar)`.
    fn set_j_menu_bar(&self, menu_bar: Option<Rc<JComponent>>) {
        *self.j_menu_bar.borrow_mut() = menu_bar;
    }

    /// `JFrame.getContentPane()`.
    fn get_content_pane(&self) -> Rc<JComponent> {
        self.content_pane.clone()
    }

    /// `Component.setLocation(int, int)`.
    fn set_location(&self, x: i32, y: i32) {
        self.location.set(Point { x, y });
    }

    /// `Window.setLocationByPlatform(boolean)`.
    fn set_location_by_platform(&self, location_by_platform: bool) {
        self.location_by_platform.set(location_by_platform);
    }

    /// `Component.getLocation()`.
    fn get_location(&self) -> Point {
        self.location.get()
    }

    /// `Window.setSize(Dimension)`.
    fn set_size(&self, size: Dimension) {
        self.size.set(size);
    }

    /// `Component.getSize()`.
    fn get_size(&self) -> Dimension {
        self.size.get()
    }

    /// `Window.pack()`: sizes the frame to its preferred size, which is Swing
    /// layout and not modelled; the frame keeps its size.
    fn pack(&self) {}
}
