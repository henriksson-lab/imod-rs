//! `IMOD/Etomo/src/etomo/ui/swing/UIHarness.java`.
//!
//! This is the narrow application-frame part of `UIHarness`: its `initialized`,
//! `headless`, and `mainFrame` state plus `createMainFrame`, `isHead`, and
//! `setVisible`.  The other methods dispatch manager, dialog, chooser, and
//! window-switch callbacks; they cannot be translated truthfully until their
//! corresponding Java targets are present.

use std::cell::RefCell;
use std::rc::Rc;

use etomo_ui_main_window::MainFrameWindow;
use slint::ComponentHandle;

use super::etomo_menu::EtomoMenu;
use super::file_chooser::FileChooser;
use super::panel::Dimension;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;

/// Java private `MessageType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageType {
    Info,
    Standard,
    Warning,
    Error,
}

/// Java `JOptionPane` result values used by the yes/no source methods.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DialogReturnValue {
    Yes,
    No,
    Cancel,
}

/// Direct `EtomoDirector.exitProgram`, `stopMaintainEtomo`, and process-exit boundary.
pub trait UiHarnessExitBoundary {
    fn exit_program(&mut self, axis_id: AxisID) -> bool;
    fn stop_maintain_etomo(&mut self);
    fn exit(&mut self, exit_value: i32);
}

pub trait UiHarnessMainFrameBoundary {
    fn set_title(&mut self, title: &str);
    fn move_sub_frame(&mut self);
    fn show_axis_a(&mut self);
    fn show_axis_b(&mut self);
    fn show_both_axis(&mut self);
    fn to_front(&mut self, manager: Option<&str>);
    fn pack(&mut self, manager: Option<&str>, axis_id: Option<AxisID>, force: Option<bool>);
    fn cancel(&mut self, manager: Option<&str>);
    fn save_manager(&mut self, manager: Option<&str>, axis_id: AxisID);
    fn do_layout(&mut self, manager: Option<&str>);
    fn validate(&mut self, manager: Option<&str>);
    fn set_visible(&mut self, manager: Option<&str>, visible: bool);
    fn set_enabled_new_tomogram_menu_item(&mut self, enabled: bool);
    fn set_enabled_log_window_menu_item(&mut self, enabled: bool);
    fn set_enabled_new_join_menu_item(&mut self, enabled: bool);
    fn set_enabled_new_generic_parallel_menu_item(&mut self, enabled: bool);
    fn set_enabled_new_anisotropic_diffusion_menu_item(&mut self, enabled: bool);
    fn set_enabled_new_batch_run_tomo_menu_item(&mut self, enabled: bool);
    fn set_enabled_new_peet_menu_item(&mut self, enabled: bool);
    fn set_enabled_new_serial_sections_menu_item(&mut self, enabled: bool);
    fn set_current_manager(
        &mut self,
        manager: Option<&str>,
        manager_key: &str,
        new_window: Option<bool>,
        manager_stamp: Option<bool>,
    );
    fn update_frame(&mut self, manager: Option<&str>);
    fn save(&mut self, axis_id: AxisID);
    fn select_window_menu_item(&mut self, manager_key: &str, new_window: Option<bool>);
    fn add_window(&mut self, manager: &str, axis_id: AxisID, manager_key: &str);
    fn remove_window(&mut self, manager_key: &str);
    fn rename_window(&mut self, old_manager_key: &str, new_manager_key: &str);
    fn repaint_window(&mut self, manager: Option<&str>, axis_id: AxisID);
    fn save_location(&mut self);
}

pub trait UiHarnessPopupBoundary {
    fn open(&mut self);
    fn log(&mut self);
}

pub trait UiHarnessManagerBoundary {
    fn get_vertical_scroll_bar_value(&self, axis_id: Option<AxisID>) -> Option<i32>;
    fn set_vertical_scroll_bar_value(&mut self, axis_id: Option<AxisID>, value: i32);
}

pub trait UiHarnessCloseManagerBoundary {
    fn close_manager(&mut self, axis_id: AxisID, manager_key: &str);
}

pub struct ScrollBarUtil<'a, M: UiHarnessManagerBoundary> {
    pub manager: &'a mut M,
    pub axis_id: Option<AxisID>,
    pub value: Option<i32>,
}

impl<'a, M: UiHarnessManagerBoundary> ScrollBarUtil<'a, M> {
    pub fn new(manager: &'a mut M, axis_id: Option<AxisID>, value: Option<i32>) -> Self {
        Self {
            manager,
            axis_id,
            value,
        }
    }
    pub fn run(&mut self) {
        if let Some(value) = self.value {
            self.manager
                .set_vertical_scroll_bar_value(self.axis_id, value);
        }
    }
}

pub struct CloseActionListener {
    pub axis_id: AxisID,
    pub manager_key: String,
}

impl CloseActionListener {
    pub fn new(axis_id: AxisID, manager_key: String) -> Self {
        Self {
            axis_id,
            manager_key,
        }
    }
    pub fn action_performed<D: UiHarnessCloseManagerBoundary>(&self, director: &mut D) {
        director.close_manager(self.axis_id, &self.manager_key);
    }
}

/// Direct `ProcessMessages` printing boundary.
pub trait ProcessMessagesBoundary {
    fn print(&self, message_type: Option<MessageType>) -> Vec<String>;
}

/// Direct `UIComponent.getComponent()` boundary.
pub trait UiComponentBoundary {
    fn has_component(&self) -> bool;
}

/// Direct `FieldDisplayer.display(UIComponent)` boundary.
pub trait FieldDisplayerBoundary {
    fn display(&mut self, ui_component: Option<&dyn UiComponentBoundary>);
}

/// Direct `AbstractFrame` dialog-presentation boundary.
pub trait UiHarnessPresentation {
    fn popup_message(
        &mut self,
        message_type: MessageType,
        manager: Option<&str>,
        has_component: bool,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&dyn ProcessMessagesBoundary>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        initial_value: Option<&str>,
    );

    fn open_yes_no(
        &mut self,
        manager: Option<&str>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        default_no: bool,
        warning: bool,
        delete: bool,
    ) -> DialogReturnValue;
}

/// Java private inner `MessageDisplayer`.
struct MessageDisplayer<'a> {
    message_type: MessageType,
    manager: Option<&'a str>,
    ui_component: Option<&'a dyn UiComponentBoundary>,
    message: Option<&'a str>,
    message_array: Option<&'a [String]>,
    process_messages: Option<&'a dyn ProcessMessagesBoundary>,
    title: Option<&'a str>,
    axis_id: Option<AxisID>,
}

impl<'a> MessageDisplayer<'a> {
    /// Java `MessageDisplayer(... String, String, AxisID)`.
    fn new_message(
        message_type: MessageType,
        manager: Option<&'a str>,
        ui_component: Option<&'a dyn UiComponentBoundary>,
        message: &'a str,
        title: &'a str,
        axis_id: Option<AxisID>,
    ) -> Self {
        Self {
            message_type,
            manager,
            ui_component,
            message: Some(message),
            message_array: None,
            process_messages: None,
            title: Some(title),
            axis_id,
        }
    }

    /// Java `MessageDisplayer(... String[], String, AxisID)`.
    fn new_message_array(
        message_type: MessageType,
        manager: Option<&'a str>,
        ui_component: Option<&'a dyn UiComponentBoundary>,
        message_array: &'a [String],
        title: &'a str,
        axis_id: Option<AxisID>,
    ) -> Self {
        Self {
            message_type,
            manager,
            ui_component,
            message: None,
            message_array: Some(message_array),
            process_messages: None,
            title: Some(title),
            axis_id,
        }
    }

    /// Java `MessageDisplayer(... ProcessMessages, String, AxisID)`.
    fn new_process_messages(
        message_type: MessageType,
        manager: Option<&'a str>,
        ui_component: Option<&'a dyn UiComponentBoundary>,
        process_messages: &'a dyn ProcessMessagesBoundary,
        title: &'a str,
        axis_id: Option<AxisID>,
    ) -> Self {
        Self {
            message_type,
            manager,
            ui_component,
            message: None,
            message_array: None,
            process_messages: Some(process_messages),
            title: Some(title),
            axis_id,
        }
    }

    /// Java `open()`.
    fn open(self, harness: &mut UiHarness, presentation: Option<&mut dyn UiHarnessPresentation>) {
        self.display(harness, presentation);
    }

    /// Java `display()`.
    fn display(
        self,
        harness: &mut UiHarness,
        presentation: Option<&mut dyn UiHarnessPresentation>,
    ) {
        harness.popup_message(
            presentation,
            self.message_type,
            self.manager,
            self.ui_component,
            self.message,
            self.message_array,
            self.process_messages,
            self.title,
            self.axis_id,
            None,
        );
    }
}

/// Java `UIHarness.INSTANCE`.
///
/// Swing confines UI objects to its event thread.  Slint has the same rule, so
/// this is thread-local instead of a `Mutex` static: putting a Slint component
/// in `EtomoDirector.INSTANCE` would falsely claim that it is thread-safe.
thread_local! {
    pub static INSTANCE: RefCell<UiHarness> = RefCell::new(UiHarness::default());
}

/// One `ActionEvent` that crossed the translated `EtomoMenu` listener
/// boundary.  Java uses the menu item's action-command string and, for its
/// two `JCheckBoxMenuItem`s, the selected state held by the component.
///
/// This intentionally records rather than performs a manager action.  The
/// corresponding `AbstractFrame`/`EtomoDirector` targets have to be
/// translated before an action can faithfully be executed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MenuEvent {
    pub command: String,
    pub checked: Option<bool>,
}

/// Java `UIHarness` fields used on the `EtomoDirector.initialize` startup path.
///
pub struct UiHarness {
    /// Java final `managerFrameTable`; manager-frame routing remains a native frame boundary.
    pub manager_frame_table: std::collections::HashMap<String, String>,
    /// Java field `initialized`, initialised to false.
    initialized: bool,
    /// Java field `headless`, initialised by `initialize()`.
    headless: bool,
    /// Java `verbose`, initialised to false.
    pub verbose: bool,
    /// Java `frameSize`, initialised to null.
    pub frame_size: Option<(i32, i32)>,
    /// Headless/test-failed Java stderr messages, retained as observable output.
    pub logged_messages: Vec<String>,
    /// Direct `EtomoDirector.isTestFailed()` state used by message source methods.
    pub test_failed: bool,
    /// Java `fileChooser`, initialised to null.
    pub file_chooser: Option<FileChooser>,
    /// Java field `mainFrame`, initialised to null.
    main_frame: Option<MainFrameWindow>,
    pub main_frame_boundary: Option<Rc<RefCell<dyn UiHarnessMainFrameBoundary>>>,
    /// Events supplied by `MainFrame` callbacks, in UI event-loop order.
    menu_events: Rc<RefCell<Vec<MenuEvent>>>,
    /// Java `MainFrame` owns one `EtomoMenu`; this is the authoritative
    /// command/checkbox/MRU state behind the Slint presentation.
    menu: Rc<RefCell<EtomoMenu>>,
}

impl Default for UiHarness {
    fn default() -> Self {
        Self {
            manager_frame_table: std::collections::HashMap::new(),
            initialized: false,
            headless: false,
            verbose: false,
            frame_size: None,
            logged_messages: Vec::new(),
            test_failed: false,
            file_chooser: None,
            main_frame: None,
            main_frame_boundary: None,
            menu_events: Rc::new(RefCell::new(Vec::new())),
            menu: Rc::new(RefCell::new(EtomoMenu::get_instance(false))),
        }
    }
}

impl UiHarness {
    pub fn bind_main_frame_boundary(
        &mut self,
        main_frame_boundary: Rc<RefCell<dyn UiHarnessMainFrameBoundary>>,
    ) {
        self.main_frame_boundary = Some(main_frame_boundary);
    }
    /// Java private `initialize()`, with `EtomoDirector.getArguments().isHeadless()`
    /// supplied by the translated director rather than accessed through Java's global.
    pub fn initialize(&mut self, headless: bool) {
        self.initialized = true;
        self.headless = headless;
    }

    /// Java `createMainFrame()` (`UIHarness.java:1250-1257`).
    pub fn create_main_frame(&mut self, headless: bool) -> Result<(), slint::PlatformError> {
        if !self.initialized {
            self.initialize(headless);
        }
        if !self.headless && self.main_frame.is_none() {
            let main_frame = MainFrameWindow::new()?;
            let menu_events = Rc::clone(&self.menu_events);
            main_frame.on_menu_command(move |command| {
                menu_events.borrow_mut().push(MenuEvent {
                    command: command.to_string(),
                    checked: None,
                });
            });
            let menu_events = Rc::clone(&self.menu_events);
            let menu = Rc::clone(&self.menu);
            main_frame.on_menu_toggled(move |command, checked| {
                let mut menu = menu.borrow_mut();
                // `CheckBoxMenuItem` uses the text as its Swing action
                // command.  The Slint surface carries the source-mapped,
                // stable menu path instead, so update the same Java field at
                // this presentation boundary.
                if command == "options.3dmod-startup-window" {
                    menu.set_menu_3dmod_startup_window(checked);
                }
                if command == "options.3dmod-bin-by-2" {
                    menu.set_menu_3dmod_bin_by_2(checked);
                }
                menu_events.borrow_mut().push(MenuEvent {
                    command: command.to_string(),
                    checked: Some(checked),
                });
            });
            self.main_frame = Some(main_frame);
        }
        Ok(())
    }

    /// Java `isHead()` (`UIHarness.java:1282-1287`).
    pub fn is_head(&mut self, headless: bool) -> bool {
        if !self.initialized {
            self.initialize(headless);
        }
        self.main_frame.is_some() || self.main_frame_boundary.is_some()
    }

    /// Java `getMainFrame()` (`UIHarness.java:823-825`).
    pub fn get_main_frame(&self) -> Option<&MainFrameWindow> {
        self.main_frame.as_ref()
    }

    /// Rust dispatch endpoint corresponding to Java's listener calls in
    /// `EtomoMenu.addListeners` (`EtomoMenu.java:268-324`).  Callers take
    /// events explicitly so no unported manager action is silently invented.
    pub fn take_menu_events(&self) -> Vec<MenuEvent> {
        std::mem::take(&mut *self.menu_events.borrow_mut())
    }

    /// `MainFrame.getEtomoMenu()` ownership path.
    pub fn menu(&self) -> Rc<RefCell<EtomoMenu>> {
        Rc::clone(&self.menu)
    }

    /// Java `setVisible(BaseManager, boolean)` (`UIHarness.java:985-994`) for
    /// the initial main-frame call.  Manager-frame selection is intentionally
    /// not represented until `BaseManager`/`WindowSwitch` are wired.
    pub fn set_visible(&self, visible: bool) -> Result<(), slint::PlatformError> {
        if let Some(main_frame) = &self.main_frame {
            if visible {
                main_frame.show()?;
            } else {
                main_frame.hide()?;
            }
        }
        Ok(())
    }

    /// Rust event-loop boundary after Java's `setVisible` schedules the main
    /// frame.  Slint requires this explicit call; Swing owns it internally.
    pub fn run(&self) -> Result<(), slint::PlatformError> {
        if let Some(main_frame) = &self.main_frame {
            main_frame.run()?;
        }
        Ok(())
    }

    /// Java private `getFrame(BaseManager, UIComponent)`.
    fn get_frame(
        &mut self,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
    ) -> Option<&MainFrameWindow> {
        let _ = (manager, ui_component);
        if self.is_head(self.headless) && !self.test_failed {
            self.main_frame.as_ref()
        } else {
            None
        }
    }

    /// Java private `popupMessage(...)`.
    #[allow(clippy::too_many_arguments)]
    fn popup_message(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        message_type: MessageType,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&dyn ProcessMessagesBoundary>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        initial_value: Option<&str>,
    ) {
        let head = self.is_head(self.headless);
        if head && presentation.is_some() {
            presentation.unwrap().popup_message(
                message_type,
                manager,
                ui_component.is_some_and(UiComponentBoundary::has_component),
                message,
                message_array,
                process_messages,
                title,
                axis_id,
                initial_value,
            );
        } else {
            self.log_message(
                message_type,
                message,
                message_array,
                process_messages,
                title,
                axis_id,
            );
        }
    }

    /// Java private `logMessage(MessageType, String, String[], ProcessMessages, String, AxisID)`.
    fn log_message(
        &mut self,
        message_type: MessageType,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&dyn ProcessMessagesBoundary>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        if let Some(message) = message {
            self.log_message_string(message, title, axis_id);
        } else if let Some(message_array) = message_array {
            self.log_message_array(message_array, title, axis_id);
        } else if let Some(process_messages) = process_messages {
            match message_type {
                MessageType::Info => self.log_info(process_messages, title, axis_id),
                MessageType::Warning => self.log_warning(process_messages, title, axis_id),
                MessageType::Error => self.log_error(process_messages, title, axis_id),
                MessageType::Standard => self.log_process_message(process_messages, title, axis_id),
            }
        } else {
            self.logged_messages
                .push("No message.  Unable to log.".into());
        }
    }

    /// Java `openMessageDialog(BaseManager, String, String, AxisID)`.
    pub fn open_message_dialog_manager_message_title_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &str,
        title: &str,
        axis_id: AxisID,
    ) {
        MessageDisplayer::new_message(
            MessageType::Standard,
            manager,
            None,
            message,
            title,
            Some(axis_id),
        )
        .open(self, presentation);
    }

    /// Java `openMessageDialog(BaseManager, UIComponent, String, String, AxisID)`.
    pub fn open_message_dialog_manager_component_message_title_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
        message: &str,
        title: &str,
        axis_id: AxisID,
    ) {
        MessageDisplayer::new_message(
            MessageType::Standard,
            manager,
            ui_component,
            message,
            title,
            Some(axis_id),
        )
        .open(self, presentation);
    }

    /// Java `openMessageDialog(BaseManager, UIComponent, String, String)`.
    pub fn open_message_dialog_manager_component_message_title(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
        message: &str,
        title: &str,
    ) {
        MessageDisplayer::new_message(
            MessageType::Standard,
            manager,
            ui_component,
            message,
            title,
            None,
        )
        .open(self, presentation);
    }

    /// Java `openMessageDialog(... FieldDisplayer)`.
    pub fn open_message_dialog_with_field_displayer(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
        message: &str,
        title: &str,
        field_displayer: Option<&mut dyn FieldDisplayerBoundary>,
    ) {
        if self.is_head(self.headless) && !self.test_failed {
            if let Some(field_displayer) = field_displayer {
                field_displayer.display(ui_component);
            }
        }
        MessageDisplayer::new_message(MessageType::Standard, manager, None, message, title, None)
            .open(self, presentation);
    }

    /// Java `openMessageDialog(... FieldDisplayer, FieldDisplayer)`.
    pub fn open_message_dialog_with_field_displayers(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
        message: &str,
        title: &str,
        field_displayer1: Option<&mut dyn FieldDisplayerBoundary>,
        field_displayer2: Option<&mut dyn FieldDisplayerBoundary>,
    ) {
        if self.is_head(self.headless) && !self.test_failed {
            if let Some(field_displayer) = field_displayer1 {
                field_displayer.display(ui_component);
            }
            if let Some(field_displayer) = field_displayer2 {
                field_displayer.display(ui_component);
            }
        }
        MessageDisplayer::new_message(
            MessageType::Standard,
            manager,
            ui_component,
            message,
            title,
            None,
        )
        .open(self, presentation);
    }

    /// Java `openWarningMessageDialog(BaseManager, UIComponent, String, String)`.
    pub fn open_warning_message_dialog_manager_component_message_title(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
        message: &str,
        title: &str,
    ) {
        MessageDisplayer::new_message(
            MessageType::Warning,
            manager,
            ui_component,
            message,
            title,
            None,
        )
        .open(self, presentation);
    }

    /// Java `openMessageDialog(BaseManager, UIComponent, String[], String)`.
    pub fn open_message_dialog_manager_component_message_array_title(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
        message: &[String],
        title: &str,
    ) {
        MessageDisplayer::new_message_array(
            MessageType::Standard,
            manager,
            ui_component,
            message,
            title,
            None,
        )
        .open(self, presentation);
    }

    /// Java `openInfoMessageDialog(BaseManager, String, String, AxisID)`.
    pub fn open_info_message_dialog_manager_message_title_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &str,
        title: &str,
        axis_id: AxisID,
    ) {
        self.open_info_message_dialog_manager_component_message_title_axis(
            presentation,
            manager,
            None,
            message,
            title,
            axis_id,
        );
    }

    /// Java `openInfoMessageDialog(BaseManager, UIComponent, String, String, AxisID)`.
    pub fn open_info_message_dialog_manager_component_message_title_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
        message: &str,
        title: &str,
        axis_id: AxisID,
    ) {
        MessageDisplayer::new_message(
            MessageType::Info,
            manager,
            ui_component,
            message,
            title,
            Some(axis_id),
        )
        .open(self, presentation);
    }

    /// Java `openErrorMessageDialog(BaseManager, ProcessMessages, String, AxisID)`.
    pub fn open_error_message_dialog_manager_process_messages_title_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        messages: &dyn ProcessMessagesBoundary,
        title: &str,
        axis_id: AxisID,
    ) {
        MessageDisplayer::new_process_messages(
            MessageType::Error,
            manager,
            None,
            messages,
            title,
            Some(axis_id),
        )
        .open(self, presentation);
    }

    /// Java `openMessageDialog(BaseManager, ProcessMessages, String, AxisID)`.
    pub fn open_message_dialog_manager_process_messages_title_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        messages: &dyn ProcessMessagesBoundary,
        title: &str,
        axis_id: AxisID,
    ) {
        MessageDisplayer::new_process_messages(
            MessageType::Standard,
            manager,
            None,
            messages,
            title,
            Some(axis_id),
        )
        .open(self, presentation);
    }

    /// Java `openWarningMessageDialog(BaseManager, ProcessMessages, String, AxisID)`.
    pub fn open_warning_message_dialog_manager_process_messages_title_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        messages: &dyn ProcessMessagesBoundary,
        title: &str,
        axis_id: AxisID,
    ) {
        MessageDisplayer::new_process_messages(
            MessageType::Warning,
            manager,
            None,
            messages,
            title,
            Some(axis_id),
        )
        .open(self, presentation);
    }

    /// Java `openWarningMessageDialog(BaseManager, String, String, AxisID)`.
    pub fn open_warning_message_dialog_manager_message_title_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &str,
        title: &str,
        axis_id: AxisID,
    ) {
        MessageDisplayer::new_message(
            MessageType::Warning,
            manager,
            None,
            message,
            title,
            Some(axis_id),
        )
        .open(self, presentation);
    }

    /// Java `openMessageDialog(BaseManager, String, String)`.
    pub fn open_message_dialog_manager_message_title(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &str,
        title: &str,
    ) {
        MessageDisplayer::new_message(MessageType::Standard, manager, None, message, title, None)
            .open(self, presentation);
    }

    /// Java `openMessageDialog(String[], LinkedHashSet<String>)`.
    pub fn open_message_dialog_message_array_additional_messages(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        message_array: Option<&[String]>,
        additional_messages: Option<&std::collections::BTreeSet<String>>,
    ) {
        if message_array.is_none_or(|value| value.is_empty())
            && additional_messages.is_none_or(|value| value.is_empty())
        {
            return;
        }
        let title = message_array
            .and_then(|value| value.first())
            .filter(|value| !value.is_empty())
            .map_or("Alert", String::as_str);
        let mut values: Vec<&str> = message_array.map_or_else(Vec::new, |value| {
            value[1..]
                .iter()
                .filter(|value| !value.is_empty())
                .map(String::as_str)
                .collect()
        });
        if let Some(additional_messages) = additional_messages {
            values.extend(additional_messages.iter().map(String::as_str));
        }
        let message = values.join("  ");
        MessageDisplayer::new_message(MessageType::Standard, None, None, &message, title, None)
            .open(self, presentation);
    }

    /// Java `openMessageDialog(BaseManager, String[], String, AxisID)`.
    pub fn open_message_dialog_manager_message_array_title_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &[String],
        title: &str,
        axis_id: AxisID,
    ) {
        MessageDisplayer::new_message_array(
            MessageType::Standard,
            manager,
            None,
            message,
            title,
            Some(axis_id),
        )
        .open(self, presentation);
    }

    /// Java `openProblemValueMessageDialog(...)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_problem_value_message_dialog(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
        problem: &str,
        param_name: &str,
        param_descr: Option<&str>,
        field_label: Option<&str>,
        problem_value: &str,
        replacement_value: Option<&str>,
        replacement_value_descr: Option<&str>,
    ) {
        let mut message = format!("{problem} '{param_name}' parameter ");
        if let Some(param_descr) = param_descr {
            message.push_str(param_descr);
        }
        message.push_str(&format!(" value '{problem_value}'."));
        if let Some(replacement_value) = replacement_value {
            message.push_str(&format!(
                "  The {} value will be replaced with ",
                problem.to_lowercase()
            ));
            if let Some(replacement_value_descr) = replacement_value_descr {
                message.push_str(&format!("'{replacement_value_descr}': "));
            }
            message.push_str(&format!(" '{replacement_value}'."));
        }
        if let Some(field_label) = field_label {
            message.push_str(&format!("  See the '{field_label}' field."));
        }
        self.open_warning_message_dialog_manager_component_message_title(
            presentation,
            manager,
            ui_component,
            &message,
            &format!("{problem} Value"),
        );
    }

    /// Java `openYesNoDialog(BaseManager, UIComponent, String)`.
    pub fn open_yes_no_dialog_manager_component_message(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        ui_component: Option<&dyn UiComponentBoundary>,
        message: &str,
    ) -> bool {
        let headless = self.headless;
        if self.is_head(headless) && !self.test_failed {
            if let Some(presentation) = presentation {
                return presentation.open_yes_no(
                    manager,
                    Some(message),
                    None,
                    None,
                    None,
                    false,
                    false,
                    false,
                ) == DialogReturnValue::Yes;
            }
        }
        let _ = ui_component;
        self.log_message_string(message, None, None);
        self.test_failed
    }

    /// Java `openYesNoCancelDialog(BaseManager, String, AxisID)`.
    pub fn open_yes_no_cancel_dialog_manager_message_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &str,
        axis_id: AxisID,
    ) -> Option<EtomoBoolean2> {
        let headless = self.headless;
        let result = if self.is_head(headless) && !self.test_failed {
            presentation.map_or(DialogReturnValue::No, |presentation| {
                presentation.open_yes_no(
                    manager,
                    Some(message),
                    None,
                    None,
                    Some(axis_id),
                    false,
                    false,
                    false,
                )
            })
        } else {
            self.log_message_string(message, None, Some(axis_id));
            if self.test_failed {
                DialogReturnValue::Yes
            } else {
                DialogReturnValue::No
            }
        };
        if result == DialogReturnValue::Cancel {
            return None;
        }
        let mut value = EtomoBoolean2::new();
        value.set_boolean(result == DialogReturnValue::Yes);
        Some(value)
    }

    /// Java `openYesNoDialog(BaseManager, String, AxisID)`.
    pub fn open_yes_no_dialog_manager_message_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &str,
        axis_id: AxisID,
    ) -> bool {
        self.open_yes_no_dialog(
            presentation,
            manager,
            Some(message),
            None,
            None,
            Some(axis_id),
            false,
            false,
            false,
        )
    }
    /// Java `openYesNoDialogWithDefaultNo(BaseManager, String, String, AxisID)`.
    pub fn open_yes_no_dialog_with_default_no(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &str,
        title: &str,
        axis_id: AxisID,
    ) -> bool {
        self.open_yes_no_dialog(
            presentation,
            manager,
            Some(message),
            None,
            Some(title),
            Some(axis_id),
            true,
            false,
            false,
        )
    }
    /// Java `openDeleteDialog(BaseManager, String[], AxisID)`.
    pub fn open_delete_dialog(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &[String],
        axis_id: AxisID,
    ) -> bool {
        self.open_yes_no_dialog(
            presentation,
            manager,
            None,
            Some(message),
            None,
            Some(axis_id),
            false,
            false,
            true,
        )
    }
    /// Java `openYesNoWarningDialog(BaseManager, String, AxisID)`.
    pub fn open_yes_no_warning_dialog(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &str,
        axis_id: AxisID,
    ) -> bool {
        self.open_yes_no_dialog(
            presentation,
            manager,
            Some(message),
            None,
            None,
            Some(axis_id),
            true,
            true,
            false,
        )
    }
    /// Java `openYesNoDialog(BaseManager, String[], AxisID)`.
    pub fn open_yes_no_dialog_manager_message_array_axis(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: &[String],
        axis_id: AxisID,
    ) -> bool {
        self.open_yes_no_dialog(
            presentation,
            manager,
            None,
            Some(message),
            None,
            Some(axis_id),
            false,
            false,
            false,
        )
    }

    /// Shared native dialog boundary for the source yes/no overloads.
    #[allow(clippy::too_many_arguments)]
    fn open_yes_no_dialog(
        &mut self,
        presentation: Option<&mut dyn UiHarnessPresentation>,
        manager: Option<&str>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        default_no: bool,
        warning: bool,
        delete: bool,
    ) -> bool {
        let headless = self.headless;
        if self.is_head(headless) && !self.test_failed {
            if let Some(presentation) = presentation {
                return presentation.open_yes_no(
                    manager,
                    message,
                    message_array,
                    title,
                    axis_id,
                    default_no,
                    warning,
                    delete,
                ) == DialogReturnValue::Yes;
            }
        }
        if let Some(message) = message {
            self.log_message_string(message, None, axis_id);
        } else if let Some(message_array) = message_array {
            self.log_message_array(message_array, None, axis_id);
        }
        self.test_failed
    }

    /// Java private `log(String, String, AxisID)`.
    fn log_message_string(&mut self, message: &str, title: Option<&str>, axis_id: Option<AxisID>) {
        self.log_header(title, axis_id);
        self.logged_messages.push(message.into());
        if !message.is_empty() {
            self.logged_messages.push(String::new());
        }
    }
    /// Java private `log(String[], String, AxisID)`.
    fn log_message_array(
        &mut self,
        message: &[String],
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_header(title, axis_id);
        self.logged_messages.extend(message.iter().cloned());
        if !message.is_empty() {
            self.logged_messages.push(String::new());
        }
    }
    /// Java private `logError(ProcessMessages, String, AxisID)`.
    fn log_error(
        &mut self,
        messages: &dyn ProcessMessagesBoundary,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_process_messages(messages, Some(MessageType::Error), title, axis_id);
    }
    /// Java private `logMessage(ProcessMessages, String, AxisID)`.
    fn log_process_message(
        &mut self,
        messages: &dyn ProcessMessagesBoundary,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_process_messages(messages, None, title, axis_id);
    }
    /// Java private `logInfo(ProcessMessages, String, AxisID)`.
    fn log_info(
        &mut self,
        messages: &dyn ProcessMessagesBoundary,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_process_messages(messages, Some(MessageType::Info), title, axis_id);
    }
    /// Java private `logWarning(ProcessMessages, String, AxisID)`.
    fn log_warning(
        &mut self,
        messages: &dyn ProcessMessagesBoundary,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_process_messages(messages, Some(MessageType::Warning), title, axis_id);
    }
    /// Source ProcessMessages print body selected by the four Java log methods.
    fn log_process_messages(
        &mut self,
        messages: &dyn ProcessMessagesBoundary,
        message_type: Option<MessageType>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_header(title, axis_id);
        let messages = messages.print(message_type);
        let printed = !messages.is_empty();
        self.logged_messages.extend(messages);
        if printed {
            self.logged_messages.push(String::new());
        }
    }

    /// Java `showAxisA()`.
    pub fn show_axis_a(&mut self) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().show_axis_a();
            }
        }
    }
    /// Java `showAxisB()`.
    pub fn show_axis_b(&mut self) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().show_axis_b();
            }
        }
    }
    /// Java `showBothAxis()`.
    pub fn show_both_axis(&mut self) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().show_both_axis();
            }
        }
    }
    /// Java `toFront(BaseManager)`.
    pub fn to_front(&mut self, manager: Option<&str>) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().to_front(manager);
            }
        }
    }
    /// Java `pack(BaseManager)`.
    pub fn pack_manager(&mut self, manager: Option<&str>) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().pack(manager, None, None);
            }
        }
    }
    /// Java `cancel(BaseManager)`.
    pub fn cancel(&mut self, manager: Option<&str>) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().cancel(manager);
            }
        }
    }
    /// Java `save(BaseManager, AxisID)`.
    pub fn save_manager_axis(&mut self, manager: Option<&str>, axis_id: AxisID) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().save_manager(manager, axis_id);
            }
        }
    }
    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> Option<&MainFrameWindow> {
        self.get_main_frame()
    }
    /// Java `getComponent()`.
    pub fn get_component(&self) -> Option<&MainFrameWindow> {
        self.get_main_frame()
    }
    /// Java `getMessageFrame(BaseManager, UIComponent)`.
    pub fn get_message_frame(
        &self,
        manager: Option<&str>,
        ui_component_present: bool,
    ) -> Option<&MainFrameWindow> {
        let _ = (manager, ui_component_present);
        self.get_main_frame()
    }
    /// Java public `getFrame(BaseManager)`.
    pub fn get_frame_manager(&self, manager: Option<&str>) -> Option<&MainFrameWindow> {
        let _ = manager;
        self.get_main_frame()
    }
    /// Java `pack(boolean, BaseManager)`.
    pub fn pack_force_manager(&mut self, force: bool, manager: Option<&str>) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().pack(manager, None, Some(force));
            }
        }
    }
    /// Java `pack(AxisID, BaseManager)`.
    pub fn pack_axis_manager(&mut self, axis_id: AxisID, manager: Option<&str>) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().pack(manager, Some(axis_id), None);
            }
        }
    }
    /// Java `pack(AxisID, boolean, BaseManager)`.
    pub fn pack_axis_force_manager(&mut self, axis_id: AxisID, force: bool, manager: Option<&str>) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame
                    .borrow_mut()
                    .pack(manager, Some(axis_id), Some(force));
            }
        }
    }
    /// Java `setEnabledNewTomogramMenuItem(boolean)`.
    pub fn set_enabled_new_tomogram_menu_item(&mut self, enable: bool) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame
                    .borrow_mut()
                    .set_enabled_new_tomogram_menu_item(enable);
            }
        }
    }
    /// Java `setMRUFileLabels(String[])`.
    pub fn set_mru_file_labels(&mut self, mru_list: &[String]) {
        let headless = self.headless;
        if self.is_head(headless) {
            self.menu.borrow_mut().set_mru_file_labels(mru_list);
        }
    }
    /// Java `is3dmodStartupWindow()`.
    pub fn is_3dmod_startup_window(&mut self) -> bool {
        let headless = self.headless;
        self.is_head(headless) && self.menu.borrow().is_menu_3dmod_startup_window()
    }
    /// Java `is3dmodBinBy2()`.
    pub fn is_3dmod_bin_by_2(&mut self) -> bool {
        let headless = self.headless;
        self.is_head(headless) && self.menu.borrow().is_menu_3dmod_bin_by_2()
    }
    /// Java `doLayout(BaseManager)`.
    pub fn do_layout(&mut self, manager: Option<&str>) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().do_layout(manager);
            }
        }
    }
    /// Java `validate(BaseManager)`.
    pub fn validate(&mut self, manager: Option<&str>) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().validate(manager);
            }
        }
    }
    /// Java `setVisible(BaseManager, boolean)`.
    pub fn set_visible_manager(&mut self, manager: Option<&str>, visible: bool) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().set_visible(manager, visible);
            }
        }
    }
    /// Java `getSize(BaseManager)`.
    pub fn get_size(&mut self, manager: Option<&str>) -> Dimension {
        let _ = manager;
        self.frame_size
            .map(|(width, height)| Dimension { width, height })
            .unwrap_or(Dimension {
                width: 0,
                height: 0,
            })
    }
    /// Java `getLocation(BaseManager)`.
    pub fn get_location(&mut self, manager: Option<&str>) -> (i32, i32) {
        let _ = manager;
        (0, 0)
    }
    /// Java `getFileChooser()`.
    pub fn get_file_chooser(
        &mut self,
        file_chooser_dimension: Dimension,
    ) -> Option<&mut FileChooser> {
        let headless = self.headless;
        if !self.is_head(headless) {
            return None;
        }
        if self.file_chooser.is_none() {
            let mut chooser = FileChooser::new();
            chooser.set_preferred_size(file_chooser_dimension);
            self.file_chooser = Some(chooser);
        }
        self.file_chooser.as_mut()
    }
    /// Java `setCurrentManager(BaseManager, UniqueKey, boolean, boolean)`.
    pub fn set_current_manager(
        &mut self,
        current_manager: Option<&str>,
        manager_key: &str,
        new_window: bool,
        manager_stamp: bool,
    ) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame.borrow_mut().set_current_manager(
                current_manager,
                manager_key,
                Some(new_window),
                Some(manager_stamp),
            );
        }
    }
    /// Java `updateFrame(BaseManager)`.
    pub fn update_frame(&mut self, current_manager: Option<&str>) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame.borrow_mut().update_frame(current_manager);
        }
    }
    /// Java `setCurrentManager(BaseManager, UniqueKey)`.
    pub fn set_current_manager_key(&mut self, current_manager: Option<&str>, manager_key: &str) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .set_current_manager(current_manager, manager_key, None, None);
        }
    }
    /// Java `save(AxisID)`.
    pub fn save_axis(&mut self, axis_id: AxisID) {
        let headless = self.headless;
        if self.is_head(headless) {
            if let Some(main_frame) = &self.main_frame_boundary {
                main_frame.borrow_mut().save(axis_id);
            }
        }
    }
    /// Java `selectWindowMenuItem(UniqueKey)`.
    pub fn select_window_menu_item(&mut self, current_key: &str) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .select_window_menu_item(current_key, None);
        }
    }
    /// Java `selectWindowMenuItem(UniqueKey, boolean)`.
    pub fn select_window_menu_item_new_window(&mut self, current_key: &str, new_window: bool) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .select_window_menu_item(current_key, Some(new_window));
        }
    }
    /// Java `setEnabledLogWindowMenuItem(boolean)`.
    pub fn set_enabled_log_window_menu_item(&mut self, enable: bool) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .set_enabled_log_window_menu_item(enable);
        }
    }
    /// Java `setEnabledNewJoinMenuItem(boolean)`.
    pub fn set_enabled_new_join_menu_item(&mut self, enable: bool) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .set_enabled_new_join_menu_item(enable);
        }
    }
    /// Java `setEnabledNewGenericParallelMenuItem(boolean)`.
    pub fn set_enabled_new_generic_parallel_menu_item(&mut self, enable: bool) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .set_enabled_new_generic_parallel_menu_item(enable);
        }
    }
    /// Java `setEnabledNewAnisotropicDiffusionMenuItem(boolean)`.
    pub fn set_enabled_new_anisotropic_diffusion_menu_item(&mut self, enable: bool) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .set_enabled_new_anisotropic_diffusion_menu_item(enable);
        }
    }
    /// Java `setEnabledNewBatchRunTomoMenuItem(boolean)`.
    pub fn set_enabled_new_batch_run_tomo_menu_item(&mut self, enable: bool) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .set_enabled_new_batch_run_tomo_menu_item(enable);
        }
    }
    /// Java `setEnabledNewPeetMenuItem(boolean)`.
    pub fn set_enabled_new_peet_menu_item(&mut self, enable: bool) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .set_enabled_new_peet_menu_item(enable);
        }
    }
    /// Java `setEnabledNewSerialSectionsMenuItem(boolean)`.
    pub fn set_enabled_new_serial_sections_menu_item(&mut self, enable: bool) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .set_enabled_new_serial_sections_menu_item(enable);
        }
    }
    /// Java `addFrame(BaseManager, boolean)`.
    pub fn add_frame(&mut self, manager: &str, savable: bool) {
        self.manager_frame_table
            .insert(manager.into(), format!("savable:{savable}"));
    }
    /// Java `addWindow(BaseManager, AxisID, UniqueKey)`.
    pub fn add_window(&mut self, manager: &str, axis_id: AxisID, manager_key: &str) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .add_window(manager, axis_id, manager_key);
        }
    }
    /// Java `removeWindow(UniqueKey)`.
    pub fn remove_window(&mut self, manager_key: &str) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame.borrow_mut().remove_window(manager_key);
        }
    }
    /// Java `renameWindow(UniqueKey, UniqueKey)`.
    pub fn rename_window(&mut self, old_manager_key: &str, new_manager_key: &str) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame
                .borrow_mut()
                .rename_window(old_manager_key, new_manager_key);
        }
    }
    /// Java `repaintWindow(BaseManager, AxisID)`.
    pub fn repaint_window(&mut self, manager: Option<&str>, axis_id: AxisID) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame.borrow_mut().repaint_window(manager, axis_id);
        }
    }
    /// Java `exit(AxisID, int)`.
    pub fn exit<D: UiHarnessExitBoundary>(
        &mut self,
        director: &mut D,
        axis_id: AxisID,
        exit_value: i32,
    ) {
        if let Some(main_frame) = &self.main_frame_boundary {
            main_frame.borrow_mut().save_location();
        }
        if director.exit_program(axis_id) {
            director.stop_maintain_etomo();
            director.exit(exit_value);
        }
    }
    pub fn set_title<F: UiHarnessMainFrameBoundary>(&mut self, main_frame: &mut F, title: &str) {
        let headless = self.headless;
        if self.is_head(headless) {
            main_frame.set_title(title);
        }
    }
    pub fn move_sub_frame<F: UiHarnessMainFrameBoundary>(
        &mut self,
        main_frame: &mut F,
        move_b: bool,
    ) {
        let headless = self.headless;
        if self.is_head(headless) && move_b {
            main_frame.move_sub_frame();
        }
    }
    pub fn open_popup<P: UiHarnessPopupBoundary>(&mut self, popup: &mut P) {
        let headless = self.headless;
        if self.is_head(headless) && !self.test_failed {
            popup.open();
        } else {
            popup.log();
        }
    }
    pub fn get_close_action_listener(
        &self,
        axis_id: AxisID,
        manager_key: String,
    ) -> CloseActionListener {
        CloseActionListener::new(axis_id, manager_key)
    }
    /// Java private `logHeader(String, AxisID)`.
    fn log_header(&mut self, title: Option<&str>, axis_id: Option<AxisID>) {
        let axis = axis_id
            .filter(|axis| *axis != AxisID::Only)
            .map(|axis| format!("({})", axis.key()))
            .unwrap_or_default();
        self.logged_messages
            .push(format!("LOG: {}{axis}:", title.unwrap_or_default()));
    }
}

#[cfg(test)]
mod tests {
    use super::{MenuEvent, UiHarness, UiHarnessMainFrameBoundary};
    use crate::imod::etomo::r#type::axis_id::AxisID;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct MainFrameMock(Vec<String>);
    impl UiHarnessMainFrameBoundary for MainFrameMock {
        fn set_title(&mut self, _: &str) {}
        fn move_sub_frame(&mut self) {}
        fn show_axis_a(&mut self) {
            self.0.push("show-axis-a".into());
        }
        fn show_axis_b(&mut self) {}
        fn show_both_axis(&mut self) {}
        fn to_front(&mut self, _: Option<&str>) {}
        fn pack(&mut self, _: Option<&str>, _: Option<AxisID>, _: Option<bool>) {
            self.0.push("pack".into());
        }
        fn cancel(&mut self, _: Option<&str>) {}
        fn save_manager(&mut self, _: Option<&str>, _: AxisID) {}
        fn do_layout(&mut self, _: Option<&str>) {}
        fn validate(&mut self, _: Option<&str>) {}
        fn set_visible(&mut self, _: Option<&str>, _: bool) {}
        fn set_enabled_new_tomogram_menu_item(&mut self, _: bool) {}
        fn set_enabled_log_window_menu_item(&mut self, _: bool) {}
        fn set_enabled_new_join_menu_item(&mut self, _: bool) {
            self.0.push("new-join".into());
        }
        fn set_enabled_new_generic_parallel_menu_item(&mut self, _: bool) {}
        fn set_enabled_new_anisotropic_diffusion_menu_item(&mut self, _: bool) {}
        fn set_enabled_new_batch_run_tomo_menu_item(&mut self, _: bool) {}
        fn set_enabled_new_peet_menu_item(&mut self, _: bool) {}
        fn set_enabled_new_serial_sections_menu_item(&mut self, _: bool) {}
        fn set_current_manager(
            &mut self,
            _: Option<&str>,
            _: &str,
            _: Option<bool>,
            _: Option<bool>,
        ) {
            self.0.push("current-manager".into());
        }
        fn update_frame(&mut self, _: Option<&str>) {}
        fn save(&mut self, _: AxisID) {}
        fn select_window_menu_item(&mut self, _: &str, _: Option<bool>) {}
        fn add_window(&mut self, _: &str, _: AxisID, _: &str) {
            self.0.push("add-window".into());
        }
        fn remove_window(&mut self, _: &str) {}
        fn rename_window(&mut self, _: &str, _: &str) {}
        fn repaint_window(&mut self, _: Option<&str>, _: AxisID) {}
        fn save_location(&mut self) {}
    }

    #[test]
    fn create_main_frame_keeps_java_headless_path_empty() {
        let mut harness = UiHarness::default();
        harness.create_main_frame(true).unwrap();
        assert!(!harness.is_head(true));
        assert!(harness.get_main_frame().is_none());
    }

    #[test]
    fn menu_events_are_fifo_and_do_not_invoke_manager_actions() {
        let harness = UiHarness::default();
        harness.menu_events.borrow_mut().extend([
            MenuEvent {
                command: "file.save".to_owned(),
                checked: None,
            },
            MenuEvent {
                command: "options.3dmod-bin-by-2".to_owned(),
                checked: Some(true),
            },
        ]);
        assert_eq!(
            harness.take_menu_events(),
            vec![
                MenuEvent {
                    command: "file.save".to_owned(),
                    checked: None,
                },
                MenuEvent {
                    command: "options.3dmod-bin-by-2".to_owned(),
                    checked: Some(true),
                },
            ]
        );
        assert!(harness.take_menu_events().is_empty());
    }

    #[test]
    fn headless_message_dialog_uses_the_source_logging_path() {
        let mut harness = UiHarness::default();
        harness.initialize(true);
        harness.open_message_dialog_manager_message_title_axis(
            None,
            Some("manager"),
            "No graphical dialog",
            "Alert",
            AxisID::First,
        );
        assert_eq!(
            harness.logged_messages,
            vec![
                "LOG: Alert(First):".to_owned(),
                "No graphical dialog".to_owned(),
                String::new(),
            ]
        );
    }

    #[test]
    fn migrated_window_methods_invoke_concrete_main_frame_boundary() {
        let mock = Rc::new(RefCell::new(MainFrameMock::default()));
        let boundary: Rc<RefCell<dyn UiHarnessMainFrameBoundary>> = mock.clone();
        let mut harness = UiHarness::default();
        harness.main_frame_boundary = Some(boundary);
        harness.show_axis_a();
        harness.pack_manager(Some("manager"));
        harness.set_current_manager(Some("manager"), "key", true, false);
        harness.set_enabled_new_join_menu_item(true);
        harness.add_window("manager", AxisID::First, "key");
        assert_eq!(
            mock.borrow().0,
            [
                "show-axis-a",
                "pack",
                "current-manager",
                "new-join",
                "add-window"
            ]
        );
    }
}
