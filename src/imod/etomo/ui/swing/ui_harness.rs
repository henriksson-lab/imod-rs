//! `IMOD/Etomo/src/etomo/ui/swing/UIHarness.java`.
//!
//! The public interface to the application frames: every popup message and
//! question, frame fitting, the main frame's menu state, and the manager
//! frames.  Headless (`--headless`) there is no main frame, and every message
//! is logged to standard error (`LOG: <title>(<axis>):` then the text) and
//! every question answered with its default.
//!
//! `UIHarness` is an EDT object: [`INSTANCE`] is a `thread_local!` holding
//! `Rc<UIHarness>` on the event dispatch thread, reached as
//! `ui_harness::INSTANCE.with(|h| h.method(...))` or [`with`].  Process and
//! monitor threads, which the Java lets call `UIHarness` directly, go through
//! the `post_*` / `*_from_process` functions at the end of this file, which
//! run the call on the EDT (`util/event_queue.rs`).
//!
//! **Popups.**  Where the Java shows a modal `JOptionPane` (through
//! `AbstractFrame.showOptionDialog`), the translation calls
//! [`present_popup`] with a [`PopupRequest`].  A presentation hook installed
//! with [`UIHarness::set_popup_hook`] answers every popup (a test driver
//! installs one, logs the request and picks a button).  Without a hook the
//! popup is written to standard error in the harness's `LOG:` form and
//! answered `CLOSED_OPTION`, which every caller treats as the default (no,
//! not deleted, dismissed).  With the `gui` feature and a Slint main window,
//! the popup is shown as a modal dialog (`slint_bridge::present_popup`).
//!
//! The Slint main window (`MainFrameWindow`, `gui` feature) stays the visual
//! of the main frame; the translated [`MainFrame`] holds the component tree
//! ([`UIHarness::get_main_frame_root`]).

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;

#[cfg(feature = "gui")]
use etomo_ui_main_window::MainFrameWindow;
#[cfg(feature = "gui")]
use slint::ComponentHandle;
/// Without the `gui` feature there is no Slint main window.
#[cfg(not(feature = "gui"))]
pub type MainFrameWindow = std::convert::Infallible;

use super::abstract_frame::{self, AbstractFrameVirtual};
use super::etomo_frame::EtomoFrameVirtual;
use super::file_chooser::{self, FileChooser};
use super::fixed_dim;
use super::main_frame::MainFrame;
use super::manager_frame::ManagerFrame;
use super::popup::Popup;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, Dimension, JComponent, Point};
use crate::imod::etomo::process::process_messages::{self, ProcessMessages};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::event_queue;
use crate::imod::etomo::util::unique_key::UniqueKey;
use crate::imod::etomo::util::utilities;

/// Java public static final `LOG_TAG`.
pub const LOG_TAG: &str = "LOG";

// Java private static final `DEBUG = EtomoDirector.INSTANCE.getArguments()
// .isDebug()`: never read in this class.

/// Java private static final `MOVE_B = EtomoDirector.INSTANCE.getArguments()
/// .isMoveB()`, read when used.
fn move_b() -> bool {
    etomo_director::ARGUMENTS.lock().unwrap().is_move_b()
}

/// Java private static final `PACK_SLEEP` (milliseconds).
const PACK_SLEEP: u64 = 1;

thread_local! {
    /// Java public static final `INSTANCE`.  Lives on the event dispatch
    /// thread; see the module documentation.
    pub static INSTANCE: Rc<UIHarness> = Rc::new(UIHarness::new());
}

/// Runs `f` on [`INSTANCE`] (the calling thread's harness; the EDT's is the
/// real one).
pub fn with<R>(f: impl FnOnce(&UIHarness) -> R) -> R {
    INSTANCE.with(|harness| f(harness))
}

// ---------------------------------------------------------------------------
// Rust-only popup presentation hook
// ---------------------------------------------------------------------------

/// Everything a modal `JOptionPane` popup was built from.
#[derive(Clone)]
pub struct PopupRequest {
    /// The uitest popup name, `Utilities.convertLabelToName(title, true)`
    /// (Java `pane.setName(name)`).
    pub name: Option<String>,
    /// The dialog title.
    pub title: Option<String>,
    /// The wrapped message lines.
    pub message: Vec<String>,
    /// The labels of the buttons the pane shows, in order.
    pub options: Vec<String>,
    /// `JOptionPane` option type (`abstract_frame::DEFAULT_OPTION`, ...).
    pub option_type: i32,
    /// `JOptionPane` message type after the icon adjustment
    /// (`abstract_frame::ERROR_MESSAGE`, ...).
    pub message_type: i32,
    /// The label of the initially selected button, where one is given.
    pub initial_value: Option<String>,
    /// The axis the popup belongs to.
    pub axis_id: Option<AxisID>,
    /// The parent component (the frame's content pane or a field).
    pub parent_component: Option<Rc<JComponent>>,
}

/// The answer to a popup: the index of the button pressed in
/// [`PopupRequest::options`], or the dialog was closed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PopupAnswer {
    /// Java `CLOSED_OPTION` (a null pane value).
    Closed,
    /// The button at this index was pressed.
    Selected(usize),
}

/// A popup presentation hook.
pub type PopupHook = Box<dyn Fn(&PopupRequest) -> PopupAnswer>;

/// Shows one popup and returns the answer (Java `dialog.setVisible(true)` on
/// a modal `JOptionPane`).  See the module documentation.
pub fn present_popup(request: &PopupRequest) -> PopupAnswer {
    let hook = with(|harness| harness.popup_hook.borrow().clone());
    if let Some(hook) = hook {
        return hook(request);
    }
    #[cfg(feature = "gui")]
    {
        let has_window = with(|harness| harness.main_frame_window.borrow().is_some());
        if has_window && let Some(answer) = super::slint_bridge::present_popup(request) {
            return answer;
        }
    }
    // No presentation: log it as UIHarness.log does and answer with the
    // default (closed).
    with(|harness| {
        harness.log_string_array_string_axis_id(
            Some(&request.message),
            request.title.as_deref(),
            request.axis_id,
        )
    });
    PopupAnswer::Closed
}

/// One `ActionEvent` from the Slint main window's menu, recorded until the
/// Slint menu is bridged to the translated `EtomoMenu` items.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MenuEvent {
    pub command: String,
    pub checked: Option<bool>,
}

// ---------------------------------------------------------------------------
// Boundary traits kept for the process layer
// ---------------------------------------------------------------------------

/// Java private `enum MessageType { INFO, STANDARD, WARNING, ERROR }`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageType {
    Info,
    Standard,
    Warning,
    Error,
}

/// `ProcessMessages.print(MessageType)` as a list of lines, for callers that
/// hold a `ProcessMessages` through this interface.
pub trait ProcessMessagesBoundary {
    fn print(&self, message_type: Option<MessageType>) -> Vec<String>;
}

/// A caller-side stand-in for a `UIComponent` parent that cannot cross to
/// the event dispatch thread (`logic/dataset_tool.rs`).
pub trait UiComponentBoundary {
    fn has_component(&self) -> bool;
}

// ---------------------------------------------------------------------------
// UIHarness
// ---------------------------------------------------------------------------

/// Java `public final class UIHarness implements UIComponent`.
pub struct UIHarness {
    /// Java private final `Hashtable<BaseManager, ManagerFrame>
    /// managerFrameTable`, keyed by manager identity.
    manager_frame_table: RefCell<Vec<(&'static dyn BaseManager, Rc<ManagerFrame>)>>,
    /// Java private `boolean initialized = false`.
    initialized: Cell<bool>,
    /// Java private `boolean headless = false`.
    headless: Cell<bool>,
    /// Java private `MainFrame mainFrame = null`.
    main_frame: RefCell<Option<Rc<MainFrame>>>,
    /// Java private `boolean verbose = false`.
    verbose: Cell<bool>,
    /// Java private `JFileChooser fileChooser = null`.
    file_chooser: RefCell<Option<Rc<FileChooser>>>,
    /// Java private `Dimension frameSize = null`.
    frame_size: Cell<Option<Dimension>>,

    // --- Rust-only presentation state ---
    /// The popup presentation hook.
    popup_hook: RefCell<Option<Rc<dyn Fn(&PopupRequest) -> PopupAnswer>>>,
    /// The Slint main window drawing the main frame.
    #[cfg(feature = "gui")]
    main_frame_window: RefCell<Option<MainFrameWindow>>,
    /// The refresh timer of the Slint bridge (`slint_bridge.rs`), kept alive
    /// with the window.
    #[cfg(feature = "gui")]
    bridge_timer: RefCell<Option<slint::Timer>>,
    /// Menu events from the Slint main window.
    menu_events: Rc<RefCell<Vec<MenuEvent>>>,
}

/// Java private inner class `MessageDisplayer`.
struct MessageDisplayer<'a> {
    /// Java private final `AbstractFrame frame`.
    frame: Option<Rc<dyn AbstractFrameVirtual>>,
    /// Java private final `MessageType type`.
    type_: Option<MessageType>,
    /// Java private final `BaseManager manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java private final `UIComponent uiComponent`.
    ui_component: Option<&'a dyn UIComponent>,
    /// Java private final `String message`.
    message: Option<&'a str>,
    /// Java private final `String[] messageArray`.
    message_array: Option<&'a [String]>,
    /// Java private final `ProcessMessages processMessages`.
    process_messages: Option<&'a ProcessMessages>,
    /// Java private final `String title`.
    title: Option<&'a str>,
    /// Java private final `AxisID axisID`.
    axis_id: Option<AxisID>,
}

impl<'a> MessageDisplayer<'a> {
    /// Java private `MessageDisplayer(AbstractFrame, MessageType, BaseManager,
    /// UIComponent, String, String, AxisID)`.
    fn new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
        frame: Option<Rc<dyn AbstractFrameVirtual>>,
        type_: Option<MessageType>,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&'a dyn UIComponent>,
        message: Option<&'a str>,
        title: Option<&'a str>,
        axis_id: Option<AxisID>,
    ) -> MessageDisplayer<'a> {
        MessageDisplayer {
            frame,
            type_,
            manager,
            ui_component,
            message,
            message_array: None,
            process_messages: None,
            title,
            axis_id,
        }
    }

    /// Java private `MessageDisplayer(AbstractFrame, MessageType, BaseManager,
    /// UIComponent, String[], String, AxisID)`.
    fn new_abstract_frame_message_type_base_manager_ui_component_string_array_string_axis_id(
        frame: Option<Rc<dyn AbstractFrameVirtual>>,
        type_: Option<MessageType>,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&'a dyn UIComponent>,
        message_array: Option<&'a [String]>,
        title: Option<&'a str>,
        axis_id: Option<AxisID>,
    ) -> MessageDisplayer<'a> {
        MessageDisplayer {
            frame,
            type_,
            manager,
            ui_component,
            message: None,
            message_array,
            process_messages: None,
            title,
            axis_id,
        }
    }

    /// Java private `MessageDisplayer(AbstractFrame, MessageType, BaseManager,
    /// UIComponent, ProcessMessages, String, AxisID)`.
    fn new_abstract_frame_message_type_base_manager_ui_component_process_messages_string_axis_id(
        frame: Option<Rc<dyn AbstractFrameVirtual>>,
        type_: Option<MessageType>,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&'a dyn UIComponent>,
        process_messages: Option<&'a ProcessMessages>,
        title: Option<&'a str>,
        axis_id: Option<AxisID>,
    ) -> MessageDisplayer<'a> {
        MessageDisplayer {
            frame,
            type_,
            manager,
            ui_component,
            message: None,
            message_array: None,
            process_messages,
            title,
            axis_id,
        }
    }

    /// Java private `open()`.
    fn open(&self, harness: &UIHarness) {
        // StackTrace stackTrace = new StackTrace(): only read by the
        // commented-out invokeLater branch below.
        // if (type == null || frame == null || !isHead() || stackTrace.isStarting()
        // || stackTrace.isExiting()) {
        self.display(harness);
        // }
        // else {
        // SwingUtilities.invokeLater(new Runnable() {
        // @Override
        // public void run() {
        // display();
        // }
        // });
        // }
    }

    /// Java private `display()`.
    fn display(&self, harness: &UIHarness) {
        harness.popup_message(
            self.frame.clone(),
            self.type_,
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

/// Java private final inner class `ScrollBarUtil implements Runnable`.  Made
/// to be used with the invokeLater function.
pub struct ScrollBarUtil {
    /// Java private final `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java private final `axisID`.
    axis_id: Option<AxisID>,
    /// Java private final `Integer value`.
    value: Option<i32>,
}

impl ScrollBarUtil {
    /// Java private `ScrollBarUtil(BaseManager, AxisID, Integer)`.
    fn new(
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        value: Option<i32>,
    ) -> ScrollBarUtil {
        ScrollBarUtil {
            manager,
            axis_id,
            value,
        }
    }

    /// Java `run()`.
    pub fn run(&self) {
        if let (Some(manager), Some(value)) = (self.manager, self.value) {
            manager.set_vertical_scroll_bar_value(self.axis_id, Some(value));
        }
    }
}

/// Java package-private static final `CloseActionListener implements
/// ActionListener`.
pub struct CloseActionListener {
    /// Java private final `axisID`.
    axis_id: Option<AxisID>,
    /// Java private final `managerKey`.
    manager_key: UniqueKey,
}

impl CloseActionListener {
    /// Java package-private `CloseActionListener(AxisID, UniqueKey)`.
    pub fn new(axis_id: Option<AxisID>, manager_key: UniqueKey) -> CloseActionListener {
        CloseActionListener {
            axis_id,
            manager_key,
        }
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: &ActionEvent) {
        let _ = event;
        etomo_director::INSTANCE.close_manager(self.axis_id, Some(&self.manager_key));
    }
}

impl UIHarness {
    /// Java private `UIHarness()`.
    fn new() -> UIHarness {
        UIHarness {
            manager_frame_table: RefCell::new(Vec::new()),
            initialized: Cell::new(false),
            headless: Cell::new(false),
            main_frame: RefCell::new(None),
            verbose: Cell::new(false),
            file_chooser: RefCell::new(None),
            frame_size: Cell::new(None),
            popup_hook: RefCell::new(None),
            #[cfg(feature = "gui")]
            main_frame_window: RefCell::new(None),
            #[cfg(feature = "gui")]
            bridge_timer: RefCell::new(None),
            menu_events: Rc::new(RefCell::new(Vec::new())),
        }
    }

    // --- Rust-only presentation members ---

    /// Installs (or, with `None`, removes) the popup presentation hook that
    /// answers every popup.
    pub fn set_popup_hook(&self, hook: Option<PopupHook>) {
        *self.popup_hook.borrow_mut() =
            hook.map(|hook| Rc::from(hook) as Rc<dyn Fn(&PopupRequest) -> PopupAnswer>);
    }

    /// The main frame's content tree root (`MainFrame.rootPanel`), if there is
    /// a main frame.  A driver finds components under it by name
    /// (`jdk::find_component`).
    /// The main frame's menu bar.  Rust-only accessor.
    pub fn get_main_frame_menu_bar(&self) -> Option<Rc<JComponent>> {
        self.main_frame
            .borrow()
            .as_ref()
            .and_then(|main_frame| main_frame.get_menu_bar_component())
    }

    pub fn get_main_frame_root(&self) -> Option<Rc<JComponent>> {
        self.main_frame
            .borrow()
            .as_ref()
            .map(|main_frame| main_frame.get_root_component())
    }

    /// Menu events recorded from the Slint main window, in event-loop order.
    pub fn take_menu_events(&self) -> Vec<MenuEvent> {
        std::mem::take(&mut *self.menu_events.borrow_mut())
    }

    /// Runs the Slint event loop on the main window (Swing owns its event
    /// loop; Slint needs this explicit call).
    #[cfg(feature = "gui")]
    pub fn run(&self) -> Result<(), slint::PlatformError> {
        let window = self
            .main_frame_window
            .borrow()
            .as_ref()
            .map(|w| w.clone_strong());
        if let Some(window) = window {
            window.run()?;
        }
        Ok(())
    }

    // --- UIHarness ---

    /// Java private `getFrame(BaseManager, UIComponent)`.
    fn get_frame_base_manager_ui_component(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
    ) -> Option<Rc<dyn AbstractFrameVirtual>> {
        if self.is_head() && !etomo_director::is_test_failed() {
            return self.get_message_frame(manager, ui_component);
        }
        None
    }

    // TODO
    // Switch to swing do later thread. Make the popup dialog non model. This
    // can't be done with QUESTION_MESSAGE, because they need to block.

    /// Java private `popupMessage(AbstractFrame, MessageType, BaseManager,
    /// UIComponent, String, String[], ProcessMessages, String, AxisID,
    /// String)`.  Pop up all types of messages.
    #[allow(clippy::too_many_arguments)]
    fn popup_message(
        &self,
        frame: Option<Rc<dyn AbstractFrameVirtual>>,
        type_: Option<MessageType>,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&ProcessMessages>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
        initial_value: Option<&str>,
    ) {
        let _ = initial_value;
        if self.is_head()
            && let Some(frame) = frame
        {
            let mut component = None;
            if ui_component.is_some() {
                component = self.get_component_ui_component(ui_component);
            }
            // Pop up the message.
            let modal = Some(true);
            let frame = frame.abstract_frame();
            if type_.is_none() || type_ == Some(MessageType::Info) {
                frame.open_info_message_dialog_base_manager_component_axis_id_string_string_array_process_messages_string_boolean(
                    manager,
                    component.as_ref(),
                    axis_id,
                    message,
                    message_array,
                    process_messages,
                    title,
                    modal,
                );
            } else if type_.is_none() || type_ == Some(MessageType::Standard) {
                frame.open_message_dialog_base_manager_component_axis_id_string_string_array_process_messages_string_boolean(
                    manager,
                    component.as_ref(),
                    axis_id,
                    message,
                    message_array,
                    process_messages,
                    title,
                    modal,
                );
            } else if type_ == Some(MessageType::Warning) {
                frame.open_warning_message_dialog_base_manager_component_axis_id_string_string_array_process_messages_string_boolean(
                    manager,
                    component.as_ref(),
                    axis_id,
                    message,
                    message_array,
                    process_messages,
                    title,
                    modal,
                );
            } else if type_ == Some(MessageType::Error) {
                frame.open_error_message_dialog_base_manager_component_axis_id_string_string_array_process_messages_string_boolean(
                    manager,
                    component.as_ref(),
                    axis_id,
                    message,
                    message_array,
                    process_messages,
                    title,
                    modal,
                );
            }
        } else {
            // Log the message if etomo is running headless, or this is a failed
            // test.
            self.log_message_message_type_string_string_array_process_messages_string_axis_id(
                type_,
                message,
                message_array,
                process_messages,
                title,
                axis_id,
            );
        }
    }

    /// Java private `logMessage(MessageType, String, String[],
    /// ProcessMessages, String, AxisID)`.  Log all types of messages.
    fn log_message_message_type_string_string_array_process_messages_string_axis_id(
        &self,
        type_: Option<MessageType>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&ProcessMessages>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        if let Some(message) = message {
            self.log_string_string_axis_id(Some(message), title, axis_id);
        } else if let Some(message_array) = message_array {
            self.log_string_array_string_axis_id(Some(message_array), title, axis_id);
        } else if let Some(process_messages) = process_messages {
            if type_ == Some(MessageType::Info) {
                self.log_info(process_messages, title, axis_id);
            } else if type_ == Some(MessageType::Warning) {
                self.log_warning(process_messages, title, axis_id);
            } else if type_ == Some(MessageType::Error) {
                self.log_error(process_messages, title, axis_id);
            } else {
                self.log_message_process_messages_string_axis_id(process_messages, title, axis_id);
            }
        } else {
            // new Exception("No message.  Unable to log.").printStackTrace()
            eprintln!("java.lang.Exception: No message.  Unable to log.");
        }
    }

    /// Java `setTitle(BaseManager, String)`.
    pub fn set_title(&self, manager: Option<&'static dyn BaseManager>, title: &str) {
        if self.is_head() {
            let title = title.to_owned();
            event_queue::invoke_later(move || {
                with(|harness| {
                    // Upstream bug fixed (UIHarness.java:238): a manager whose
                    // ManagerFrame has not been added yet has no frame, and Java
                    // throws NullPointerException; nothing is done then.
                    if let Some(frame) = harness.get_frame_base_manager(manager) {
                        frame.abstract_frame().set_title(Some(&title));
                    }
                })
            });
        }
    }

    /// Java `moveSubFrame()`.
    pub fn move_sub_frame(&self) {
        if self.is_head() && move_b() {
            event_queue::invoke_later(move || {
                if etomo_director::INSTANCE.is_test() {
                    eprintln!("Moving subframe.");
                }
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.move_sub_frame();
                    }
                })
            });
        }
    }

    /// Java synchronized `openMessageDialog(BaseManager, String, String,
    /// AxisID)`.  Open a message dialog.
    pub fn open_message_dialog_base_manager_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &str,
        title: &str,
        axis_id: Option<AxisID>,
    ) {
        // getMessageFrame(manager, null).displayMessage(manager, message, title, axisID);
        // log(message, title, axisID);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
            self.get_frame_base_manager_ui_component(manager, None),
            Some(MessageType::Standard),
            manager,
            None,
            Some(message),
            Some(title),
            axis_id,
        )
        .open(self);
    }

    /// Java private `getComponent(UIComponent)`.
    fn get_component_ui_component(
        &self,
        ui_component: Option<&dyn UIComponent>,
    ) -> Option<Rc<JComponent>> {
        if let Some(ui_component) = ui_component {
            return Some(ui_component.get_component());
        }
        None
    }

    /// Java synchronized `openMessageDialog(BaseManager, UIComponent, String,
    /// String, AxisID)`.  Open a message dialog.
    pub fn open_message_dialog_base_manager_ui_component_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
        message: &str,
        title: &str,
        axis_id: Option<AxisID>,
    ) {
        // getMessageFrame(manager, uiComponent).displayMessage(manager,
        // getComponent(uiComponent), message, title, axisID);
        // log(message, title, axisID);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
            self.get_message_frame(manager, ui_component),
            Some(MessageType::Standard),
            manager,
            ui_component,
            Some(message),
            Some(title),
            axis_id,
        )
        .open(self);
    }

    /// Java synchronized `openMessageDialog(BaseManager, UIComponent, String,
    /// String)`.
    pub fn open_message_dialog_base_manager_ui_component_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
        message: &str,
        title: &str,
    ) {
        // getMessageFrame(manager, uiComponent).displayMessage(manager,
        // getComponent(uiComponent), message, title, null);
        // log(message, title, null);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
            self.get_message_frame(manager, ui_component),
            Some(MessageType::Standard),
            manager,
            ui_component,
            Some(message),
            Some(title),
            None,
        )
        .open(self);
    }

    /// Java synchronized `openMessageDialog(BaseManager, UIComponent, String,
    /// String, FieldDisplayer)`.
    ///
    /// Java posts `fieldDisplayer.display(uiComponent)` with invokeLater, so it
    /// runs while the modal popup is up; the borrowed displayer cannot be
    /// posted, so it runs just before the popup opens, which shows the same
    /// field behind the same popup.
    pub fn open_message_dialog_base_manager_ui_component_string_string_field_displayer(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
        message: &str,
        title: &str,
        field_displayer: Option<&dyn FieldDisplayer>,
    ) {
        if self.is_head() && !etomo_director::is_test_failed() {
            if let Some(field_displayer) = field_displayer {
                field_displayer.display_ui_component(ui_component);
            }
        }
        // getMessageFrame(manager, uiComponent).displayMessage(manager,
        // getComponent(uiComponent), message, title, null);
        // log(message, title, null);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
            self.get_message_frame(manager, ui_component),
            Some(MessageType::Standard),
            manager,
            None,
            Some(message),
            Some(title),
            None,
        )
        .open(self);
    }

    /// Java synchronized `openMessageDialog(BaseManager, UIComponent, String,
    /// String, FieldDisplayer, FieldDisplayer)`.  See the one-displayer
    /// overload for when the displayers run.
    #[allow(clippy::too_many_arguments)]
    pub fn open_message_dialog_base_manager_ui_component_string_string_field_displayer_field_displayer(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
        message: &str,
        title: &str,
        field_displayer1: Option<&dyn FieldDisplayer>,
        field_displayer2: Option<&dyn FieldDisplayer>,
    ) {
        if self.is_head() && !etomo_director::is_test_failed() {
            if let Some(field_displayer1) = field_displayer1 {
                field_displayer1.display_ui_component(ui_component);
            }
            if let Some(field_displayer2) = field_displayer2 {
                field_displayer2.display_ui_component(ui_component);
            }
        }
        // getMessageFrame(manager, uiComponent).displayMessage(manager,
        // getComponent(uiComponent), message, title, null);
        // log(message, title, null);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
            self.get_message_frame(manager, ui_component),
            Some(MessageType::Standard),
            manager,
            ui_component,
            Some(message),
            Some(title),
            None,
        )
        .open(self);
    }

    /// Java synchronized `openYesNoDialog(BaseManager, UIComponent, String)`.
    pub fn open_yes_no_dialog_base_manager_ui_component_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
        message: &str,
    ) -> bool {
        if self.is_head() && !etomo_director::is_test_failed() {
            // Upstream bug fixed (UIHarness.java:369): a manager in a
            // ManagerFrame not yet added has no message frame, and Java throws
            // NullPointerException; such a question is logged and answered
            // like a headless one instead.
            if let Some(frame) = self.get_message_frame(manager, ui_component) {
                let component = self.get_component_ui_component(ui_component);
                return frame.display_yes_no_message_base_manager_component_string_axis_id(
                    manager,
                    component.as_ref(),
                    Some(message),
                    None,
                );
            }
        }
        self.log_string_axis_id(Some(message), None);
        if etomo_director::is_test_failed() {
            return true;
        }
        false
    }

    /// Java `openProblemValueMessageDialog(BaseManager, UIComponent, String,
    /// String, String, String, String, String, String)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_problem_value_message_dialog(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
        problem: &str,
        param_name: Option<&str>,
        param_descr: Option<&str>,
        field_label: Option<&str>,
        problem_value: Option<&str>,
        replacement_value: Option<&str>,
        replacement_value_descr: Option<&str>,
    ) {
        // Build message
        let mut message = String::new();
        message.push_str(problem);
        message.push_str(&format!(" '{}' parameter ", param_name.unwrap_or("null")));
        if let Some(param_descr) = param_descr {
            message.push_str(param_descr);
        }
        message.push_str(" value '");
        message.push_str(problem_value.unwrap_or("null"));
        message.push_str("'.");
        if let Some(replacement_value) = replacement_value {
            message.push_str("  The ");
            message.push_str(&problem.to_lowercase());
            message.push_str(" value will be replaced with ");
            if let Some(replacement_value_descr) = replacement_value_descr {
                message.push('\'');
                message.push_str(replacement_value_descr);
                message.push_str("': ");
            }
            message.push_str(" '");
            message.push_str(replacement_value);
            message.push_str("'.");
        }
        if let Some(field_label) = field_label {
            message.push_str("  See the '");
            message.push_str(field_label);
            message.push_str("' field.");
        }

        self.open_warning_message_dialog_base_manager_ui_component_string_string(
            manager,
            ui_component,
            &message,
            &format!("{} Value", problem),
        );
    }

    /// Java `openPopup(Popup)`.
    pub fn open_popup(&self, popup: &Popup) {
        if self.is_head() && !etomo_director::is_test_failed() {
            popup.open();
        } else {
            popup.log();
        }
    }

    /// Java synchronized `openWarningMessageDialog(BaseManager, UIComponent,
    /// String, String)`.
    pub fn open_warning_message_dialog_base_manager_ui_component_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
        message: &str,
        title: &str,
    ) {
        // getMessageFrame(manager, uiComponent).displayWarningMessage(null,
        // getComponent(uiComponent), message, title, null);
        // log(message, title, null);
        // The source passes a null manager to the displayer (the popup text is
        // then logged to standard error rather than to the manager's log).
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
            self.get_message_frame(manager, ui_component),
            Some(MessageType::Warning),
            None,
            ui_component,
            Some(message),
            Some(title),
            None,
        )
        .open(self);
    }

    /// Java synchronized `openMessageDialog(BaseManager, UIComponent, String[],
    /// String)`.
    pub fn open_message_dialog_base_manager_ui_component_string_array_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
        message: &[String],
        title: &str,
    ) {
        // getMessageFrame(manager, uiComponent).displayMessage(manager,
        // getComponent(uiComponent), message, title, null);
        // log(message, title, null);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_array_string_axis_id(
            self.get_message_frame(manager, ui_component),
            Some(MessageType::Standard),
            manager,
            ui_component,
            Some(message),
            Some(title),
            None,
        )
        .open(self);
    }

    /// Java synchronized `openInfoMessageDialog(BaseManager, String, String,
    /// AxisID)`.
    pub fn open_info_message_dialog_base_manager_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &str,
        title: &str,
        axis_id: Option<AxisID>,
    ) {
        self.open_info_message_dialog_base_manager_ui_component_string_string_axis_id(
            manager, None, message, title, axis_id,
        );
    }

    /// Java synchronized `openInfoMessageDialog(BaseManager, UIComponent,
    /// String, String, AxisID)`.
    pub fn open_info_message_dialog_base_manager_ui_component_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
        message: &str,
        title: &str,
        axis_id: Option<AxisID>,
    ) {
        // getMessageFrame(manager, uiComponent).displayInfoMessage(manager,
        // getComponent(uiComponent), message, title, axisID);
        // log(message, title, axisID);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
            self.get_message_frame(manager, ui_component),
            Some(MessageType::Info),
            manager,
            ui_component,
            Some(message),
            Some(title),
            axis_id,
        )
        .open(self);
    }

    /// Java synchronized `openErrorMessageDialog(BaseManager, ProcessMessages,
    /// String, AxisID)`.  Open one dialog and display all error messages in
    /// messages.
    pub fn open_error_message_dialog(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &ProcessMessages,
        title: &str,
        axis_id: Option<AxisID>,
    ) {
        // getMessageFrame(manager, null).displayErrorMessage(manager, message,
        // title, axisID);
        // logError(message, title, axisID);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_process_messages_string_axis_id(
            self.get_message_frame(manager, None),
            Some(MessageType::Error),
            manager,
            None,
            Some(message),
            Some(title),
            axis_id,
        )
        .open(self);
    }

    /// Java synchronized `openMessageDialog(BaseManager, ProcessMessages,
    /// String, AxisID)`.  Open one dialog and display all messages in process
    /// messages.
    pub fn open_message_dialog_base_manager_process_messages_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &ProcessMessages,
        title: &str,
        axis_id: Option<AxisID>,
    ) {
        // getMessageFrame(manager, null).displayMessage(manager, message, title,
        // axisID);
        // logMessage(message, title, axisID);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_process_messages_string_axis_id(
            self.get_message_frame(manager, None),
            Some(MessageType::Standard),
            manager,
            None,
            Some(message),
            Some(title),
            axis_id,
        )
        .open(self);
    }

    // TODO

    /// Java synchronized `openWarningMessageDialog(BaseManager,
    /// ProcessMessages, String, AxisID)`.  Open one dialog and display all
    /// warning messages in messages.
    pub fn open_warning_message_dialog_base_manager_process_messages_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        messages: &ProcessMessages,
        title: &str,
        axis_id: Option<AxisID>,
    ) {
        // getMessageFrame(manager, null).displayWarningMessage(manager, messages,
        // title, axisID);
        // logWarning(messages, title, axisID);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_process_messages_string_axis_id(
            self.get_message_frame(manager, None),
            Some(MessageType::Warning),
            manager,
            None,
            Some(messages),
            Some(title),
            axis_id,
        )
        .open(self);
    }

    /// Java synchronized `openWarningMessageDialog(BaseManager, String,
    /// String, AxisID)`.
    pub fn open_warning_message_dialog_base_manager_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &str,
        title: &str,
        axis_id: Option<AxisID>,
    ) {
        // getMessageFrame(manager, null).displayWarningMessage(manager, messages,
        // title, axisID);
        // logWarning(messages, title, axisID);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
            self.get_message_frame(manager, None),
            Some(MessageType::Warning),
            manager,
            None,
            Some(message),
            Some(title),
            axis_id,
        )
        .open(self);
    }

    /// Java synchronized `openMessageDialog(BaseManager, String, String)`.
    /// Open a message dialog.
    pub fn open_message_dialog_base_manager_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &str,
        title: &str,
    ) {
        // getMessageFrame(manager, null).displayMessage(manager, message, title);
        // log(message, title);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
            self.get_message_frame(manager, None),
            Some(MessageType::Standard),
            manager,
            None,
            Some(message),
            Some(title),
            None,
        )
        .open(self);
    }

    /// Java synchronized `openMessageDialog(String[], LinkedHashSet<String>)`.
    /// Puts together and pops up a message from a bunch of strings.
    /// `message_array` is the title followed by messages; `additional_messages`
    /// is the list of extra messages to add to the popup (the
    /// `LinkedHashSet`'s elements in insertion order).
    pub fn open_message_dialog_string_array_linked_hash_set(
        &self,
        message_array: Option<&[String]>,
        additional_messages: Option<&[String]>,
    ) {
        if message_array.is_none_or(|array| array.is_empty())
            && additional_messages.is_none_or(|set| set.is_empty())
        {
            // No message to display.
            return;
        }
        // The first element is assumed to be the title.
        let title: String = match message_array {
            Some(array) if !array.is_empty() => array[0].clone(),
            _ => "Alert".to_owned(),
        };
        // Build the message.
        let mut builder: Option<String> = None;
        if let Some(message_array) = message_array {
            for item in message_array.iter().skip(1) {
                match builder.as_mut() {
                    None => builder = Some(String::new()),
                    Some(builder) => builder.push_str("  "),
                }
                builder.as_mut().unwrap().push_str(item);
            }
        }
        if let Some(additional_messages) = additional_messages {
            for message in additional_messages {
                match builder.as_mut() {
                    None => builder = Some(String::new()),
                    Some(builder) => builder.push_str("  "),
                }
                builder.as_mut().unwrap().push_str(message);
            }
        }
        let message: Option<String> = builder;
        // getMessageFrame(null, null).displayMessage(null, message, title);
        // log(message, title);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_string_axis_id(
            self.get_message_frame(None, None),
            Some(MessageType::Standard),
            None,
            None,
            message.as_deref(),
            Some(&title),
            None,
        )
        .open(self);
    }

    /// Java synchronized `openMessageDialog(BaseManager, String[], String,
    /// AxisID)`.  Open a message dialog.
    pub fn open_message_dialog_base_manager_string_array_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        title: &str,
        axis_id: Option<AxisID>,
    ) {
        // getMessageFrame(manager, null).displayMessage(manager, message, title,
        // axisID);
        // log(message, title, axisID);
        MessageDisplayer::new_abstract_frame_message_type_base_manager_ui_component_string_array_string_axis_id(
            self.get_frame_base_manager_ui_component(manager, None),
            Some(MessageType::Standard),
            manager,
            None,
            Some(message),
            Some(title),
            axis_id,
        )
        .open(self);
    }

    /// Java synchronized `openYesNoCancelDialog(BaseManager, String, AxisID)`.
    /// Returns null for Cancel.
    pub fn open_yes_no_cancel_dialog(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &str,
        axis_id: Option<AxisID>,
    ) -> Option<EtomoBoolean2> {
        let mut retval: Option<EtomoBoolean2>;
        if self.is_head() && !etomo_director::is_test_failed() {
            // Upstream bug fixed (UIHarness.java:639): Java throws
            // NullPointerException without a message frame; see
            // open_yes_no_dialog_base_manager_ui_component_string.
            if let Some(frame) = self.get_message_frame(manager, None) {
                let dialog_ret_value =
                    frame.display_yes_no_cancel_message(manager, Some(message), axis_id);
                if dialog_ret_value == abstract_frame::CANCEL_OPTION {
                    return None;
                }
                let mut value = EtomoBoolean2::new();
                if dialog_ret_value == abstract_frame::YES_OPTION {
                    value.set_boolean(true);
                } else if dialog_ret_value == abstract_frame::NO_OPTION {
                    // Upstream bug fixed (UIHarness.java:646): Java tests
                    // YES_OPTION a second time here, so a No answer left the
                    // value unset (null) instead of false.
                    value.set_boolean(false);
                }
                retval = Some(value);
                return retval;
            }
        }
        self.log_string_axis_id(Some(message), axis_id);

        let mut value = EtomoBoolean2::new();
        if etomo_director::is_test_failed() {
            value.set_boolean(true);
        } else {
            value.set_boolean(false);
        }
        retval = Some(value);
        retval
    }

    /// Java synchronized `openYesNoDialog(BaseManager, String, AxisID)`.
    pub fn open_yes_no_dialog_base_manager_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &str,
        axis_id: Option<AxisID>,
    ) -> bool {
        if self.is_head() && !etomo_director::is_test_failed() {
            // Upstream bug fixed (UIHarness.java:670): NullPointerException
            // without a message frame.
            if let Some(frame) = self.get_message_frame(manager, None) {
                return frame.display_yes_no_message_base_manager_string_axis_id(
                    manager,
                    Some(message),
                    axis_id,
                );
            }
        }
        self.log_string_axis_id(Some(message), axis_id);
        if etomo_director::is_test_failed() {
            return true;
        }
        false
    }

    /// Java synchronized `openYesNoDialogWithDefaultNo(BaseManager, String,
    /// String, AxisID)`.
    pub fn open_yes_no_dialog_with_default_no(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &str,
        title: &str,
        axis_id: Option<AxisID>,
    ) -> bool {
        if self.is_head() && !etomo_director::is_test_failed() {
            // Upstream bug fixed (UIHarness.java:682): NullPointerException
            // without a message frame.
            if let Some(frame) = self.get_message_frame(manager, None) {
                return frame.open_yes_no_dialog_with_default_no(
                    manager,
                    Some(message),
                    Some(title),
                    axis_id,
                );
            }
        }
        self.log_string_axis_id(Some(message), axis_id);
        if etomo_director::is_test_failed() {
            return true;
        }
        false
    }

    /// Java synchronized `openDeleteDialog(BaseManager, String[], AxisID)`.
    pub fn open_delete_dialog(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        axis_id: Option<AxisID>,
    ) -> bool {
        if self.is_head() && !etomo_director::is_test_failed() {
            // Upstream bug fixed (UIHarness.java:694): NullPointerException
            // without a message frame.
            if let Some(frame) = self.get_message_frame(manager, None) {
                return frame.display_delete_message(manager, message, axis_id);
            }
        }
        self.log_string_array_axis_id(Some(message), axis_id);
        if etomo_director::is_test_failed() {
            return true;
        }
        false
    }

    /// Java synchronized `openYesNoWarningDialog(BaseManager, String,
    /// AxisID)`.
    pub fn open_yes_no_warning_dialog(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &str,
        axis_id: Option<AxisID>,
    ) -> bool {
        if self.is_head() && !etomo_director::is_test_failed() {
            // Upstream bug fixed (UIHarness.java:707): NullPointerException
            // without a message frame.
            if let Some(frame) = self.get_message_frame(manager, None) {
                return frame.display_yes_no_warning_dialog(manager, Some(message), axis_id);
            }
        }
        self.log_string_axis_id(Some(message), axis_id);
        if etomo_director::is_test_failed() {
            return true;
        }
        false
    }

    /// Java synchronized `openYesNoDialog(BaseManager, String[], AxisID)`.
    pub fn open_yes_no_dialog_base_manager_string_array_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        axis_id: Option<AxisID>,
    ) -> bool {
        if self.is_head() && !etomo_director::is_test_failed() {
            // Upstream bug fixed (UIHarness.java:720): NullPointerException
            // without a message frame.
            if let Some(frame) = self.get_message_frame(manager, None) {
                return frame.display_yes_no_message_base_manager_string_array_axis_id(
                    manager, message, axis_id,
                );
            }
        }
        self.log_string_array_axis_id(Some(message), axis_id);
        if etomo_director::is_test_failed() {
            return true;
        }
        false
    }

    /// Java `showAxisA()`.
    pub fn show_axis_a(&self) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.show_axis_a();
                    }
                })
            });
        }
    }

    /// Java `showAxisB()`.
    pub fn show_axis_b(&self) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.show_axis_b();
                    }
                })
            });
        }
    }

    /// Java `showBothAxis()`.
    pub fn show_both_axis(&self) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.show_both_axis();
                    }
                })
            });
        }
    }

    /// Java `toFront(BaseManager)`.
    pub fn to_front(&self, manager: Option<&'static dyn BaseManager>) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    // Upstream bug fixed (UIHarness.java:765): NullPointerException
                    // when the manager's frame is not registered yet.
                    if let Some(frame) = harness.get_frame_base_manager(manager) {
                        frame.abstract_frame().to_front();
                    }
                })
            });
        }
    }

    /// Java `pack(BaseManager)`.
    pub fn pack_base_manager(&self, manager: Option<&'static dyn BaseManager>) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                std::thread::sleep(std::time::Duration::from_millis(PACK_SLEEP));
                let mut value = None;
                if let Some(manager) = manager {
                    value = manager.get_vertical_scroll_bar_value(None);
                    manager.pack();
                }
                with(|harness| {
                    // Upstream bug fixed (UIHarness.java:786): a manager whose
                    // frame is not registered yet makes Java throw
                    // NullPointerException; the pack is skipped.
                    let Some(abstract_frame) = harness.get_frame_base_manager(manager) else {
                        return;
                    };
                    // Swing painting: abstractFrame.repaint().
                    abstract_frame.pack_void();
                    if let Some(manager) = manager {
                        let focus_component = manager.get_focus_component();
                        if focus_component.is_some() {
                            // Swing focus: focusComponent.requestFocus().
                        }
                        if value.is_some() {
                            ScrollBarUtil::new(Some(manager), None, value).run();
                        }
                    }
                    let size = abstract_frame.abstract_frame().get_size();
                    harness.frame_size.set(Some(size));
                })
            });
        }
    }

    /// Java `cancel(BaseManager)`.
    pub fn cancel(&self, manager: Option<&'static dyn BaseManager>) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    let frame = harness.get_frame_base_manager(manager);
                    if let Some(frame) = frame {
                        frame.cancel();
                    }
                })
            });
        }
    }

    /// Java `save(BaseManager, AxisID)`.
    pub fn save_base_manager_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) {
        if self.is_head() {
            let frame = self.get_frame_base_manager(manager);
            if let Some(frame) = frame {
                frame.save(axis_id);
            }
        }
    }

    /// Java `getUIComponent()` (`UIComponent`): the main frame.
    pub fn get_ui_component(&self) -> Option<Rc<MainFrame>> {
        self.get_main_frame()
    }

    /// Java `getComponent()` (`UIComponent`): the main frame as a component.
    pub fn get_component_void(&self) -> Option<Rc<JComponent>> {
        self.get_main_frame()
            .map(|main_frame| main_frame.get_content_pane())
    }

    /// Java `getMessageFrame(BaseManager, UIComponent)`.
    pub fn get_message_frame(
        &self,
        manager: Option<&'static dyn BaseManager>,
        ui_component: Option<&dyn UIComponent>,
    ) -> Option<Rc<dyn AbstractFrameVirtual>> {
        // The uiComponent is the prefered parent for the popup message. No need
        // to look for a specific frame if the uiComponent is present.
        if ui_component.is_some()
            || manager.is_none()
            || !manager.is_some_and(|manager| manager.is_in_manager_frame())
        {
            return self
                .main_frame
                .borrow()
                .clone()
                .map(|frame| frame as Rc<dyn AbstractFrameVirtual>);
        }
        let manager = manager.unwrap();
        self.manager_frame_table
            .borrow()
            .iter()
            .find(|(key, _)| {
                std::ptr::addr_eq(
                    *key as *const dyn BaseManager,
                    manager as *const dyn BaseManager,
                )
            })
            .map(|(_, frame)| frame.clone() as Rc<dyn AbstractFrameVirtual>)
    }

    /// Java `getFrame(BaseManager)`.
    pub fn get_frame_base_manager(
        &self,
        manager: Option<&'static dyn BaseManager>,
    ) -> Option<Rc<dyn AbstractFrameVirtual>> {
        if manager.is_none() || !manager.is_some_and(|manager| manager.is_in_manager_frame()) {
            return self
                .main_frame
                .borrow()
                .clone()
                .map(|frame| frame as Rc<dyn AbstractFrameVirtual>);
        }
        let manager = manager.unwrap();
        self.manager_frame_table
            .borrow()
            .iter()
            .find(|(key, _)| {
                std::ptr::addr_eq(
                    *key as *const dyn BaseManager,
                    manager as *const dyn BaseManager,
                )
            })
            .map(|(_, frame)| frame.clone() as Rc<dyn AbstractFrameVirtual>)
    }

    /// Java `getMainFrame()`.
    pub fn get_main_frame(&self) -> Option<Rc<MainFrame>> {
        self.main_frame.borrow().clone()
    }

    /// Java `pack(boolean, BaseManager)`.
    pub fn pack_boolean_base_manager(
        &self,
        force: bool,
        manager: Option<&'static dyn BaseManager>,
    ) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                std::thread::sleep(std::time::Duration::from_millis(PACK_SLEEP));
                let mut value = None;
                if let Some(manager) = manager {
                    value = manager.get_vertical_scroll_bar_value(None);
                    manager.pack();
                }
                with(|harness| {
                    // Upstream bug fixed (UIHarness.java:863): see pack(BaseManager).
                    let Some(abstract_frame) = harness.get_frame_base_manager(manager) else {
                        return;
                    };
                    // Swing painting: abstractFrame.repaint().
                    abstract_frame.pack_boolean(force);
                    if let Some(manager) = manager {
                        let focus_component = manager.get_focus_component();
                        if focus_component.is_some() {
                            // Swing focus: focusComponent.requestFocus().
                        }
                        if value.is_some() {
                            ScrollBarUtil::new(Some(manager), None, value).run();
                        }
                    }
                    let size = abstract_frame.abstract_frame().get_size();
                    harness.frame_size.set(Some(size));
                })
            });
        }
    }

    /// Java `pack(AxisID, BaseManager)`.
    pub fn pack_axis_id_base_manager(
        &self,
        axis_id: Option<AxisID>,
        manager: Option<&'static dyn BaseManager>,
    ) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                std::thread::sleep(std::time::Duration::from_millis(PACK_SLEEP));
                let mut value = None;
                if let Some(manager) = manager {
                    value = manager.get_vertical_scroll_bar_value(axis_id);
                    manager.pack();
                }
                with(|harness| {
                    // Upstream bug fixed (UIHarness.java:894): see pack(BaseManager).
                    let Some(abstract_frame) = harness.get_frame_base_manager(manager) else {
                        return;
                    };
                    abstract_frame.repaint(axis_id);
                    abstract_frame.pack_axis_id(axis_id);
                    if let Some(manager) = manager {
                        let focus_component = manager.get_focus_component();
                        if focus_component.is_some() {
                            // Swing focus: focusComponent.requestFocus().
                        }
                        if value.is_some() {
                            ScrollBarUtil::new(Some(manager), axis_id, value).run();
                        }
                    }
                    let size = abstract_frame.abstract_frame().get_size();
                    harness.frame_size.set(Some(size));
                })
            });
        }
    }

    /// Java `pack(AxisID, boolean, BaseManager)`.
    pub fn pack_axis_id_boolean_base_manager(
        &self,
        axis_id: Option<AxisID>,
        force: bool,
        manager: Option<&'static dyn BaseManager>,
    ) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                std::thread::sleep(std::time::Duration::from_millis(PACK_SLEEP));
                let mut value = None;
                if let Some(manager) = manager {
                    value = manager.get_vertical_scroll_bar_value(axis_id);
                    manager.pack();
                }
                with(|harness| {
                    // Upstream bug fixed (UIHarness.java:926): see pack(BaseManager).
                    let Some(abstract_frame) = harness.get_frame_base_manager(manager) else {
                        return;
                    };
                    abstract_frame.repaint(axis_id);
                    abstract_frame.pack_axis_id_boolean(axis_id, force);
                    if let Some(manager) = manager {
                        let focus_component = manager.get_focus_component();
                        if focus_component.is_some() {
                            // Swing focus: focusComponent.requestFocus().
                        }
                        if value.is_some() {
                            ScrollBarUtil::new(Some(manager), axis_id, value).run();
                        }
                    }
                    let size = abstract_frame.abstract_frame().get_size();
                    harness.frame_size.set(Some(size));
                })
            });
        }
    }

    /// Java `setEnabledNewTomogramMenuItem(boolean)`.
    pub fn set_enabled_new_tomogram_menu_item(&self, enable: bool) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_enabled_new_tomogram_menu_item(enable);
                    }
                })
            });
        }
    }

    /// Java `setMRUFileLabels(String[])`.
    pub fn set_mru_file_labels(&self, m_ru_list: &[String]) {
        if self.is_head() {
            let m_ru_list = m_ru_list.to_vec();
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_mru_file_labels(&m_ru_list);
                    }
                })
            });
        }
    }

    /// Java `is3dmodStartupWindow()`.
    pub fn is_3dmod_startup_window(&self) -> bool {
        if !self.is_head() {
            return false;
        }
        let main_frame = self.main_frame.borrow().clone();
        main_frame.is_some_and(|main_frame| main_frame.is_menu_3dmod_startup_window())
    }

    /// Java `is3dmodBinBy2()`.
    pub fn is_3dmod_bin_by_2(&self) -> bool {
        if !self.is_head() {
            return false;
        }
        let main_frame = self.main_frame.borrow().clone();
        main_frame.is_some_and(|main_frame| main_frame.is_menu_3dmod_bin_by_2())
    }

    /// Java `doLayout(BaseManager)`.
    pub fn do_layout(&self, manager: Option<&'static dyn BaseManager>) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    // Upstream bug fixed (UIHarness.java:977): NullPointerException
                    // when the manager's frame is not registered yet.
                    if let Some(frame) = harness.get_frame_base_manager(manager) {
                        frame.abstract_frame().do_layout();
                    }
                })
            });
        }
    }

    /// Java `validate(BaseManager)`.
    pub fn validate(&self, manager: Option<&'static dyn BaseManager>) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    // Upstream bug fixed (UIHarness.java:988): NullPointerException
                    // when the manager's frame is not registered yet.
                    if let Some(frame) = harness.get_frame_base_manager(manager) {
                        frame.abstract_frame().validate();
                    }
                })
            });
        }
    }

    /// Java `setVisible(BaseManager, boolean)`.
    pub fn set_visible(&self, manager: Option<&'static dyn BaseManager>, b: bool) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    // Upstream bug fixed (UIHarness.java:999): NullPointerException
                    // when the manager's frame is not registered yet.
                    let Some(frame) = harness.get_frame_base_manager(manager) else {
                        return;
                    };
                    frame.set_visible(b);
                    // The Slint window draws the main frame: show or hide it
                    // with the frame (Rust-only presentation).
                    #[cfg(feature = "gui")]
                    {
                        let is_main_frame =
                            harness
                                .main_frame
                                .borrow()
                                .as_ref()
                                .is_some_and(|main_frame| {
                                    std::ptr::addr_eq(
                                        Rc::as_ptr(main_frame),
                                        Rc::as_ptr(&frame) as *const dyn AbstractFrameVirtual,
                                    )
                                });
                        if is_main_frame {
                            if let Some(window) = harness.main_frame_window.borrow().as_ref() {
                                let _ = if b { window.show() } else { window.hide() };
                            }
                        }
                    }
                })
            });
        }
    }

    /// Java `getSize(BaseManager)`.
    pub fn get_size(&self, manager: Option<&'static dyn BaseManager>) -> Dimension {
        if self.is_head() {
            // Upstream bug fixed (UIHarness.java:1007): NullPointerException
            // when the manager's frame is not registered yet; the headless
            // value is returned then.
            if let Some(frame) = self.get_frame_base_manager(manager) {
                return frame.abstract_frame().get_size();
            }
        }
        fixed_dim::x0_y0
    }

    /// Java `getLocation(BaseManager)`.
    pub fn get_location(&self, manager: Option<&'static dyn BaseManager>) -> Point {
        if self.is_head() {
            // Upstream bug fixed (UIHarness.java:1014): see getSize.
            if let Some(frame) = self.get_frame_base_manager(manager) {
                return frame.abstract_frame().get_location();
            }
        }
        Point { x: 0, y: 0 }
    }

    /// Java `getFileChooser()`.  Returns the existing file chooser.
    /// Everything in the file chooser is reset except current directory.
    /// Returns a file chooser if the application has a GUI, otherwise null.
    pub fn get_file_chooser(&self) -> Option<Rc<FileChooser>> {
        if self.is_head() {
            let existing = self.file_chooser.borrow().clone();
            match existing {
                None => {
                    let file_chooser = FileChooser::new_void();
                    // Swing layout: fileChooser.setPreferredSize(UIParameters
                    // .getInstance().getFileChooserDimension()).
                    *self.file_chooser.borrow_mut() = Some(file_chooser);
                }
                Some(file_chooser) => {
                    // restore to defaults
                    file_chooser.reset_choosable_file_filters();
                    file_chooser.set_dialog_title(Some(file_chooser::DEFAULT_TITLE));
                    file_chooser.set_dialog_type(file_chooser::OPEN_DIALOG);
                    file_chooser.set_file_filter(None);
                    file_chooser.set_file_hiding_enabled(true);
                    file_chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
                    file_chooser.set_multi_selection_enabled(false);
                    file_chooser.set_selected_file(Some(Path::new("")));
                }
            }
            return self.file_chooser.borrow().clone();
        }
        None
    }

    /// Java `setCurrentManager(BaseManager, UniqueKey, boolean, boolean)`.
    pub fn set_current_manager_base_manager_unique_key_boolean_boolean(
        &self,
        current_manager: Option<&'static dyn BaseManager>,
        manager_key: Option<&UniqueKey>,
        new_window: bool,
        manager_stamp: bool,
    ) {
        if self.is_head() {
            let manager_key = manager_key.cloned();
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_current_manager_base_manager_unique_key_boolean_boolean(
                            current_manager,
                            manager_key.as_ref(),
                            new_window,
                            manager_stamp,
                        );
                    }
                })
            });
        }
    }

    /// Java `updateFrame(BaseManager)`.
    pub fn update_frame(&self, current_manager: Option<&'static dyn BaseManager>) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.update_frame(current_manager);
                    }
                })
            });
        }
    }

    /// Java `setCurrentManager(BaseManager, UniqueKey)`.
    pub fn set_current_manager_base_manager_unique_key(
        &self,
        current_manager: Option<&'static dyn BaseManager>,
        manager_key: Option<&UniqueKey>,
    ) {
        if self.is_head() {
            let manager_key = manager_key.cloned();
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_current_manager_base_manager_unique_key(
                            current_manager,
                            manager_key.as_ref(),
                        );
                    }
                })
            });
        }
    }

    /// Java `save(AxisID)`.
    pub fn save_axis_id(&self, axis_id: Option<AxisID>) {
        if self.is_head() {
            let main_frame = self.main_frame.borrow().clone();
            if let Some(main_frame) = main_frame {
                if main_frame.is_menu_save_enabled() {
                    AbstractFrameVirtual::save(&*main_frame, axis_id);
                }
            }
        }
    }

    /// Java `selectWindowMenuItem(UniqueKey)`.
    pub fn select_window_menu_item_unique_key(&self, current_key: Option<&UniqueKey>) {
        if self.is_head() {
            let current_key = current_key.cloned();
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.select_window_menu_item_unique_key(current_key.as_ref());
                    }
                })
            });
        }
    }

    /// Java `selectWindowMenuItem(UniqueKey, boolean)`.  If there is a head,
    /// tells mainFrame to select a window menu item based on
    /// currentManagerKey.
    pub fn select_window_menu_item_unique_key_boolean(
        &self,
        current_manager_key: Option<&UniqueKey>,
        new_window: bool,
    ) {
        if self.is_head() {
            let current_manager_key = current_manager_key.cloned();
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.select_window_menu_item_unique_key_boolean(
                            current_manager_key.as_ref(),
                            new_window,
                        );
                    }
                })
            });
        }
    }

    /// Java `setEnabledLogWindowMenuItem(boolean)`.
    pub fn set_enabled_log_window_menu_item(&self, enable: bool) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_enabled_log_window_menu_item(enable);
                    }
                })
            });
        }
    }

    /// Java `setEnabledNewJoinMenuItem(boolean)`.
    pub fn set_enabled_new_join_menu_item(&self, enable: bool) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_enabled_new_join_menu_item(enable);
                    }
                })
            });
        }
    }

    /// Java `setEnabledNewGenericParallelMenuItem(boolean)`.
    pub fn set_enabled_new_generic_parallel_menu_item(&self, enable: bool) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_enabled_new_generic_parallel_menu_item(enable);
                    }
                })
            });
        }
    }

    /// Java `setEnabledNewAnisotropicDiffusionMenuItem(boolean)`.
    pub fn set_enabled_new_anisotropic_diffusion_menu_item(&self, enable: bool) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_enabled_new_anisotropic_diffusion_menu_item(enable);
                    }
                })
            });
        }
    }

    /// Java `setEnabledNewBatchRunTomoMenuItem(boolean)`.
    pub fn set_enabled_new_batch_run_tomo_menu_item(&self, enable: bool) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_enabled_new_batch_run_tomo_menu_item(enable);
                    }
                })
            });
        }
    }

    /// Java `setEnabledNewPeetMenuItem(boolean)`.
    pub fn set_enabled_new_peet_menu_item(&self, enable: bool) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_enabled_new_peet_menu_item(enable);
                    }
                })
            });
        }
    }

    /// Java `setEnabledNewSerialSectionsMenuItem(boolean)`.
    pub fn set_enabled_new_serial_sections_menu_item(&self, enable: bool) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.set_enabled_new_serial_sections_menu_item(enable);
                    }
                })
            });
        }
    }

    /// Java `addFrame(BaseManager, boolean)`.
    pub fn add_frame(&self, manager: &'static dyn BaseManager, savable: bool) {
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    let manager_frame = ManagerFrame::get_instance(manager, savable);
                    // Hashtable.put: replaces the value for an existing key.
                    let mut table = harness.manager_frame_table.borrow_mut();
                    table.retain(|(key, _)| {
                        !std::ptr::addr_eq(
                            *key as *const dyn BaseManager,
                            manager as *const dyn BaseManager,
                        )
                    });
                    table.push((manager, manager_frame));
                })
            });
        }
    }

    /// Java `addWindow(BaseManager, AxisID, UniqueKey)`.
    pub fn add_window(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        manager_key: Option<&UniqueKey>,
    ) {
        if self.is_head() {
            let manager_key = manager_key.cloned();
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.add_window(manager, axis_id, manager_key.as_ref());
                    }
                })
            });
        }
    }

    /// Java `removeWindow(UniqueKey)`.
    pub fn remove_window(&self, manager_key: Option<&UniqueKey>) {
        if self.is_head() {
            let manager_key = manager_key.cloned();
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame.remove_window(manager_key.as_ref());
                    }
                })
            });
        }
    }

    /// Java `renameWindow(UniqueKey, UniqueKey)`.
    pub fn rename_window(
        &self,
        old_manager_key: Option<&UniqueKey>,
        new_manager_key: Option<&UniqueKey>,
    ) {
        if self.is_head() {
            let old_manager_key = old_manager_key.cloned();
            let new_manager_key = new_manager_key.cloned();
            event_queue::invoke_later(move || {
                with(|harness| {
                    if let Some(main_frame) = harness.main_frame.borrow().clone() {
                        main_frame
                            .rename_window(old_manager_key.as_ref(), new_manager_key.as_ref());
                    }
                })
            });
        }
    }

    /// Java `repaintWindow(BaseManager, AxisID)`.
    pub fn repaint_window(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) {
        let _ = axis_id;
        if self.is_head() {
            event_queue::invoke_later(move || {
                with(|harness| {
                    // Upstream bug fixed (UIHarness.java:1233): NullPointerException
                    // when the manager's frame is not registered yet.
                    if let Some(frame) = harness.get_frame_base_manager(manager) {
                        frame.abstract_frame().repaint_window();
                    }
                })
            });
        }
    }

    /// Java `createMainFrame()`.  Initialize if necessary.  Instantiate
    /// mainFrame if headless is false.
    ///
    /// Must run on the event dispatch thread.  With the `gui` feature this is
    /// also where the Slint main window is created and the calling thread,
    /// which then runs the Slint event loop ([`UIHarness::run`]), becomes the
    /// EDT; a Slint failure is reported and leaves the translated main frame
    /// without a visual.
    pub fn create_main_frame(&self) {
        if !self.initialized.get() {
            self.initialize();
        }
        if !self.headless.get() && self.main_frame.borrow().is_none() {
            #[cfg(feature = "gui")]
            {
                match MainFrameWindow::new() {
                    Ok(window) => {
                        let menu_events = Rc::clone(&self.menu_events);
                        // The Slint menu items also report through the
                        // bridge (kinds "mn"/"cbmn"), which clicks the
                        // translated `EtomoMenu` item of the same text; the
                        // command names are recorded here.
                        window.on_menu_command(move |command| {
                            menu_events.borrow_mut().push(MenuEvent {
                                command: command.to_string(),
                                checked: None,
                            });
                        });
                        let menu_events = Rc::clone(&self.menu_events);
                        window.on_menu_toggled(move |command, checked| {
                            menu_events.borrow_mut().push(MenuEvent {
                                command: command.to_string(),
                                checked: Some(checked),
                            });
                        });
                        // The Slint event loop's thread is the event dispatch
                        // thread.
                        event_queue::install_slint_edt();
                        // Rust-only: connect the Slint widgets to the
                        // translated component tree.
                        *self.bridge_timer.borrow_mut() =
                            Some(super::slint_bridge::install(&window));
                        *self.main_frame_window.borrow_mut() = Some(window);
                    }
                    Err(error) => {
                        eprintln!("Unable to create the Slint main window: {}", error);
                    }
                }
            }
            let main_frame = MainFrame::new();
            main_frame.set_verbose(self.verbose.get());
            *self.main_frame.borrow_mut() = Some(main_frame);
        }
    }

    /// Java `exit(AxisID, int)`.
    pub fn exit(&self, axis_id: Option<AxisID>, exit_value: i32) {
        // Store the current location of the frame in case etomo exits.
        if self.is_head() {
            let main_frame = self.main_frame.borrow().clone();
            if let Some(main_frame) = main_frame {
                main_frame.save_location();
                let sub_frame = main_frame.get_other_frame();
                if let Some(sub_frame) = sub_frame {
                    sub_frame.etomo_frame().save_location();
                }
            }
        }
        // Check to see if etomo can exit, save data, and then exit.
        let exit_program = etomo_director::INSTANCE.exit_program(axis_id);
        if exit_program {
            eprintln!("exitValue:{}", exit_value);
            etomo_director::INSTANCE.stop_maintain_etomo();
            std::process::exit(exit_value);
        }
    }

    /// Java `isHead()`.  Initialize if necessary.  Returns true if mainFrame is
    /// not null.
    pub fn is_head(&self) -> bool {
        if !self.initialized.get() {
            self.initialize();
        }
        self.main_frame.borrow().is_some()
    }

    /// Java private `initialize()`.  Initialize headless, testLog, and
    /// logWriter.
    fn initialize(&self) {
        self.initialized.set(true);
        let headless = etomo_director::ARGUMENTS.lock().unwrap().is_headless();
        self.headless.set(headless);
    }

    /// Java private `log(String, AxisID)`.  Log the parameters.
    fn log_string_axis_id(&self, message: Option<&str>, axis_id: Option<AxisID>) {
        self.log_string_string_axis_id(message, None, axis_id);
    }

    /// Java private `log(String, String)`.  Log the parameters.
    #[allow(dead_code)]
    fn log_string_string(&self, message: Option<&str>, title: Option<&str>) {
        self.log_string_string_axis_id(message, title, Some(AxisID::Only));
    }

    /// Java private `log(String, String, AxisID)`.  Log the parameters.
    fn log_string_string_axis_id(
        &self,
        message: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_header(title, axis_id);
        eprintln!("{}", message.unwrap_or("null"));
        if !utilities::is_empty(message) {
            eprintln!();
        }
    }

    /// Java private `logError(ProcessMessages, String, AxisID)`.
    fn log_error(
        &self,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_header(title, axis_id);
        // Java `ProcessMessages.print(MessageType)`: prints the list to stderr only
        // when DEBUG (`--debug`) is set.
        let printed = match process_messages.print(process_messages::MessageType::Error) {
            Some(text) if etomo_director::ARGUMENTS.lock().unwrap().is_debug() => {
                eprintln!("{text}");
                true
            }
            _ => false,
        };
        if printed {
            eprintln!();
        }
    }

    /// Java private `logMessage(ProcessMessages, String, AxisID)`.
    fn log_message_process_messages_string_axis_id(
        &self,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_header(title, axis_id);
        let mut list_type_iterator = process_messages.list_type_iterator();
        let mut printed = false;
        while list_type_iterator.has_next() {
            let list_type = list_type_iterator.next();
            // Upstream bug fixed (UIHarness.java:1383): Java writes `printed =
            // printed || processMessages.print(...)`, which stops printing the
            // remaining lists as soon as one list printed; every list is
            // printed here.
            let printed_list = list_type.is_some_and(|list_type| {
                // Java `ProcessMessages.print(ListType)`: prints the list to
                // stderr only when DEBUG (`--debug`) is set.
                match process_messages.print_list(list_type) {
                    Some(text) if etomo_director::ARGUMENTS.lock().unwrap().is_debug() => {
                        eprintln!("{text}");
                        true
                    }
                    _ => false,
                }
            });
            printed = printed_list || printed;
        }
        if printed {
            eprintln!();
        }
    }

    /// Java private `logInfo(ProcessMessages, String, AxisID)`.
    fn log_info(
        &self,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_header(title, axis_id);
        // Java `ProcessMessages.print(MessageType)`: prints the list to stderr only
        // when DEBUG (`--debug`) is set.
        let printed = match process_messages.print(process_messages::MessageType::Info) {
            Some(text) if etomo_director::ARGUMENTS.lock().unwrap().is_debug() => {
                eprintln!("{text}");
                true
            }
            _ => false,
        };
        if printed {
            eprintln!();
        }
    }

    /// Java private `logWarning(ProcessMessages, String, AxisID)`.
    fn log_warning(
        &self,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_header(title, axis_id);
        // Java `ProcessMessages.print(MessageType)`: prints the list to stderr only
        // when DEBUG (`--debug`) is set.
        let printed = match process_messages.print(process_messages::MessageType::Warning) {
            Some(text) if etomo_director::ARGUMENTS.lock().unwrap().is_debug() => {
                eprintln!("{text}");
                true
            }
            _ => false,
        };
        if printed {
            eprintln!();
        }
    }

    /// Java private `log(String[], AxisID)`.  Log the parameters.
    fn log_string_array_axis_id(&self, message: Option<&[String]>, axis_id: Option<AxisID>) {
        self.log_string_array_string_axis_id(message, None, axis_id);
    }

    /// Java private `log(String[], String, AxisID)`.  Log the parameters.
    fn log_string_array_string_axis_id(
        &self,
        message: Option<&[String]>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.log_header(title, axis_id);
        let mut printed = false;
        if let Some(message) = message {
            for line in message {
                eprintln!("{}", line);
                printed = true;
            }
        }
        if printed {
            eprintln!();
        }
    }

    /// Java private `logHeader(String, AxisID)`.
    fn log_header(&self, title: Option<&str>, axis_id: Option<AxisID>) {
        eprintln!(
            "{}: {}{}:",
            LOG_TAG,
            title.unwrap_or(""),
            match axis_id {
                None | Some(AxisID::Only) => String::new(),
                Some(axis_id) => format!("({})", axis_id),
            }
        );
    }

    /// Java package-private `getCloseActionListener(AxisID, UniqueKey)`.
    pub fn get_close_action_listener(
        &self,
        axis_id: Option<AxisID>,
        manager_key: UniqueKey,
    ) -> CloseActionListener {
        CloseActionListener::new(axis_id, manager_key)
    }
}

// ---------------------------------------------------------------------------
// Process-thread entry points (kept with their existing signatures)
// ---------------------------------------------------------------------------

/// `ProcessMessages.print(MessageType)` as `UIHarness`'s logging reads it.
impl ProcessMessagesBoundary for ProcessMessages {
    fn print(&self, message_type: Option<MessageType>) -> Vec<String> {
        use crate::imod::etomo::process::process_messages::MessageType as PmType;
        let types: &[PmType] = match message_type {
            Some(MessageType::Error) => &[PmType::Error],
            Some(MessageType::Warning) => &[PmType::Warning],
            Some(MessageType::Info) => &[PmType::Info],
            _ => &[PmType::Error, PmType::Warning, PmType::Info],
        };
        let mut lines = Vec::new();
        for ty in types {
            for index in 0..self.size(*ty) {
                if let Some(line) = self.get(*ty, index) {
                    lines.push(line.to_owned());
                }
            }
        }
        lines
    }
}

/// Runs `call` on the event dispatch thread's harness: directly when already
/// there, otherwise with the calling thread's own harness, which has no main
/// frame and so logs (the Java calls `UIHarness` from process threads
/// directly; the posting variants below are the way to reach the EDT's).
fn open_from_process(
    call: impl FnOnce(&UIHarness, Option<&'static dyn BaseManager>),
    manager: Option<&'static dyn BaseManager>,
) {
    with(|harness| call(harness, manager));
}

/// `UIHarness.INSTANCE.openMessageDialog(BaseManager, String, String[,
/// AxisID])` from any thread.  Off the event dispatch thread the call is
/// posted to it.
pub fn open_message_dialog_from_process(
    manager: Option<&'static dyn BaseManager>,
    message: &str,
    title: &str,
    axis_id: Option<AxisID>,
) {
    if !event_queue::is_dispatch_thread() {
        post_message_dialog(manager, message.to_owned(), title.to_owned(), axis_id);
        return;
    }
    open_from_process(
        |harness, manager| match axis_id {
            Some(axis_id) => harness.open_message_dialog_base_manager_string_string_axis_id(
                manager,
                message,
                title,
                Some(axis_id),
            ),
            None => harness.open_message_dialog_base_manager_string_string(manager, message, title),
        },
        manager,
    );
}

/// `UIHarness.INSTANCE.openMessageDialog(BaseManager, UIComponent, String,
/// String[, AxisID])` from any thread.  The component only parents the
/// popup (and, for a manager in its own frame, puts the popup on the main
/// frame); it cannot cross threads, so the popup is parented by the manager's
/// frame.
pub fn open_message_dialog_with_component_from_process(
    manager: Option<&'static dyn BaseManager>,
    ui_component: Option<&dyn UiComponentBoundary>,
    message: &str,
    title: &str,
    axis_id: Option<AxisID>,
) {
    let _ = ui_component;
    open_message_dialog_from_process(manager, message, title, axis_id);
}

/// `UIHarness.INSTANCE.openInfoMessageDialog(BaseManager, String, String,
/// AxisID)` from any thread.  Off the event dispatch thread the call is
/// posted to it.
pub fn open_info_message_dialog_from_process(
    manager: Option<&'static dyn BaseManager>,
    message: &str,
    title: &str,
    axis_id: Option<AxisID>,
) {
    if !event_queue::is_dispatch_thread() {
        let message = message.to_owned();
        let title = title.to_owned();
        event_queue::invoke_later(move || {
            open_info_message_dialog_from_process(manager, &message, &title, axis_id);
        });
        return;
    }
    open_from_process(
        |harness, manager| {
            harness.open_info_message_dialog_base_manager_string_string_axis_id(
                manager, message, title, axis_id,
            )
        },
        manager,
    );
}

/// `UIHarness.INSTANCE.openErrorMessageDialog(BaseManager, ProcessMessages,
/// String, AxisID)`; run on the event dispatch thread (see
/// [`post_error_message_dialog`]).
pub fn open_error_message_dialog_from_process(
    manager: Option<&'static dyn BaseManager>,
    messages: &ProcessMessages,
    title: &str,
    axis_id: AxisID,
) {
    open_from_process(
        |harness, manager| {
            harness.open_error_message_dialog(manager, messages, title, Some(axis_id))
        },
        manager,
    );
}

/// `UIHarness.INSTANCE.openWarningMessageDialog(BaseManager, ProcessMessages,
/// String, AxisID)`; run on the event dispatch thread (see
/// [`post_warning_message_dialog`]).
pub fn open_warning_message_dialog_from_process(
    manager: Option<&'static dyn BaseManager>,
    messages: &ProcessMessages,
    title: &str,
    axis_id: AxisID,
) {
    open_from_process(
        |harness, manager| {
            harness.open_warning_message_dialog_base_manager_process_messages_string_axis_id(
                manager,
                messages,
                title,
                Some(axis_id),
            )
        },
        manager,
    );
}

/// Posts [`open_message_dialog_from_process`] to the event dispatch thread.
pub fn post_message_dialog(
    manager: Option<&'static dyn BaseManager>,
    message: String,
    title: String,
    axis_id: Option<AxisID>,
) {
    event_queue::invoke_later(move || {
        open_message_dialog_from_process(manager, &message, &title, axis_id);
    });
}

/// Posts [`open_error_message_dialog_from_process`] to the event dispatch
/// thread.
pub fn post_error_message_dialog(
    manager: Option<&'static dyn BaseManager>,
    messages: ProcessMessages,
    title: String,
    axis_id: AxisID,
) {
    event_queue::invoke_later(move || {
        open_error_message_dialog_from_process(manager, &messages, &title, axis_id);
    });
}

/// Posts [`open_warning_message_dialog_from_process`] to the event dispatch
/// thread.
pub fn post_warning_message_dialog(
    manager: Option<&'static dyn BaseManager>,
    messages: ProcessMessages,
    title: String,
    axis_id: AxisID,
) {
    event_queue::invoke_later(move || {
        open_warning_message_dialog_from_process(manager, &messages, &title, axis_id);
    });
}

/// A yes/no question from any thread: run on the event dispatch thread and
/// wait for the answer, as the Java's modal dialog does.  `delete` asks
/// `openDeleteDialog`, `warning` `openYesNoWarningDialog`, `default_no`
/// `openYesNoDialogWithDefaultNo`, otherwise `openYesNoDialog` with the
/// message or message array.
#[allow(clippy::too_many_arguments)]
fn yes_no_from_process(
    manager: Option<&'static dyn BaseManager>,
    message: Option<String>,
    message_array: Option<Vec<String>>,
    axis_id: Option<AxisID>,
    default_no: bool,
    warning: bool,
    delete: bool,
) -> bool {
    event_queue::invoke_and_wait(move || {
        with(|harness| {
            let message_array = message_array.unwrap_or_default();
            let message = message.unwrap_or_default();
            if delete {
                harness.open_delete_dialog(manager, &message_array, axis_id)
            } else if warning {
                harness.open_yes_no_warning_dialog(manager, &message, axis_id)
            } else if default_no {
                harness.open_yes_no_dialog_with_default_no(manager, &message, "", axis_id)
            } else if !message_array.is_empty() {
                harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                    manager,
                    &message_array,
                    axis_id,
                )
            } else {
                harness.open_yes_no_dialog_base_manager_string_axis_id(manager, &message, axis_id)
            }
        })
    })
}

/// `UIHarness.INSTANCE.openDeleteDialog(BaseManager, String[], AxisID)` from
/// any thread.
pub fn open_delete_dialog_from_process(
    manager: Option<&'static dyn BaseManager>,
    message: &[String],
    axis_id: Option<AxisID>,
) -> bool {
    yes_no_from_process(
        manager,
        None,
        Some(message.to_vec()),
        axis_id,
        false,
        false,
        true,
    )
}

/// `UIHarness.INSTANCE.openYesNoDialog(BaseManager, String, AxisID)` from any
/// thread.
pub fn open_yes_no_dialog_from_process(
    manager: Option<&'static dyn BaseManager>,
    message: &str,
    axis_id: Option<AxisID>,
) -> bool {
    yes_no_from_process(
        manager,
        Some(message.to_owned()),
        None,
        axis_id,
        false,
        false,
        false,
    )
}

/// `UIHarness.INSTANCE.openMessageDialog(BaseManager, String[], String,
/// AxisID)` from any thread (posted to the event dispatch thread).
pub fn open_message_dialog_array_from_process(
    manager: Option<&'static dyn BaseManager>,
    message: &[String],
    title: &str,
    axis_id: Option<AxisID>,
) {
    let message = message.to_vec();
    let title = title.to_owned();
    event_queue::invoke_later(move || {
        open_from_process(
            |harness, manager| {
                harness.open_message_dialog_base_manager_string_array_string_axis_id(
                    manager,
                    &message,
                    &title,
                    Some(axis_id.unwrap_or(AxisID::Only)),
                )
            },
            manager,
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headless_harness_has_no_main_frame_root_and_logs_questions() {
        // A fresh thread has its own, uninitialised harness.
        std::thread::spawn(|| {
            with(|harness| {
                harness.initialized.set(true);
                harness.headless.set(true);
                assert!(!harness.is_head());
                assert!(harness.get_main_frame_root().is_none());
                assert!(!harness.open_yes_no_dialog_base_manager_string_axis_id(
                    None,
                    "Question?",
                    Some(AxisID::First)
                ));
            });
        })
        .join()
        .unwrap();
    }
}
