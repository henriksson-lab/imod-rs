//! `IMOD/Etomo/src/etomo/ui/swing/AbstractFrame.java`.
//!
//! The base of eTomo's top-level windows (`MainFrame`, `SubFrame`,
//! `ManagerFrame`).  It standardises every popup message and question dialog:
//! all of them end in `showOptionDialog`, which logs the message, picks the
//! icon, names the popup for uitest (`printName`) and shows a modal
//! `JOptionPane`.
//!
//! The `JFrame` itself is modelled only by the state the translation reads
//! back: its content pane (the root of the frame's component tree, searched by
//! name), title, visibility, displayability, location and size, its menu bar,
//! and its window-focus listeners.  Layout, painting and icons are not
//! modelled (see `jdk.rs`); those statements are kept as `// Swing layout:`
//! comments.
//!
//! The modal `JOptionPane` is replaced by the Rust-only presentation hook in
//! `ui_harness.rs` ([`ui_harness::present_popup`]): the dialog request carries
//! everything the Java pane was built from (title, wrapped message lines,
//! button labels, message and option type, initial value and the uitest
//! popup name), and the answer is the index of the button pressed, or
//! `CLOSED_OPTION`, exactly what the Java pane's value reduces to.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::etomo_frame::FrameType;
use super::swing_component::SwingComponent;
use super::ui_harness::{self, PopupAnswer, PopupRequest};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, Dimension, JComponent, Point};
use crate::imod::etomo::logic::popup_tool;
use crate::imod::etomo::process::process_messages::{self, ProcessMessages};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

// `javax.swing.JOptionPane` constants used by this class and its callers.
// TODO(unit): jdk.rs has no JOptionPane; these are the JDK's values.
/// `JOptionPane.DEFAULT_OPTION`.
pub const DEFAULT_OPTION: i32 = -1;
/// `JOptionPane.YES_NO_OPTION`.
pub const YES_NO_OPTION: i32 = 0;
/// `JOptionPane.YES_NO_CANCEL_OPTION`.
pub const YES_NO_CANCEL_OPTION: i32 = 1;
/// `JOptionPane.OK_CANCEL_OPTION`.
pub const OK_CANCEL_OPTION: i32 = 2;
/// `JOptionPane.YES_OPTION`.
pub const YES_OPTION: i32 = 0;
/// `JOptionPane.NO_OPTION`.
pub const NO_OPTION: i32 = 1;
/// `JOptionPane.CANCEL_OPTION`.
pub const CANCEL_OPTION: i32 = 2;
/// `JOptionPane.OK_OPTION`.
pub const OK_OPTION: i32 = 0;
/// `JOptionPane.CLOSED_OPTION`.
pub const CLOSED_OPTION: i32 = -1;
/// `JOptionPane.ERROR_MESSAGE`.
pub const ERROR_MESSAGE: i32 = 0;
/// `JOptionPane.INFORMATION_MESSAGE`.
pub const INFORMATION_MESSAGE: i32 = 1;
/// `JOptionPane.WARNING_MESSAGE`.
pub const WARNING_MESSAGE: i32 = 2;
/// `JOptionPane.QUESTION_MESSAGE`.
pub const QUESTION_MESSAGE: i32 = 3;
/// `JOptionPane.PLAIN_MESSAGE`.
pub const PLAIN_MESSAGE: i32 = -1;

/// `java.awt.event.WindowEvent.WINDOW_CLOSING`.
pub const WINDOW_CLOSING: i32 = 201;

/// Java private static final `OK`.
const OK: &str = "OK";
/// Java private static final `ETOMO_QUESTION`.
const ETOMO_QUESTION: &str = "Etomo question";
/// Java private static final `YES`.
const YES: &str = "Yes";
/// Java private static final `NO`.
const NO: &str = "No";
/// Java private static final `CANCEL`.
const CANCEL: &str = "Cancel";
/// Java private static final `YES_NO_LABEL_ARRAY`.
const YES_NO_LABEL_ARRAY: [&str; 2] = [YES, NO];
/// Java private static final `NO_INDEX`.
const NO_INDEX: usize = 1;
/// Java private static final `OK_LABEL_ARRAY`.
const OK_LABEL_ARRAY: [&str; 1] = [OK];
/// Java private static final `DELETE_NO_LABEL_ARRAY`.
const DELETE_NO_LABEL_ARRAY: [&str; 2] = ["Delete", NO];
/// Java public static final `DELETE_OPTION`.
pub const DELETE_OPTION: i32 = YES_OPTION;
/// Java private static final `YES_NO_CANCEL_LABEL_ARRAY`.
const YES_NO_CANCEL_LABEL_ARRAY: [&str; 3] = [YES, NO, CANCEL];

/// Java `PRINT_NAMES = EtomoDirector.INSTANCE.getArguments().isPrintNames()`,
/// read when used (the Java static is read once at class initialisation,
/// after the arguments are parsed).
fn print_names() -> bool {
    etomo_director::ARGUMENTS.lock().unwrap().is_print_names()
}

/// Java `WindowFocusListener`: the frame's focus events.  Rust-only stand-in
/// for the AWT listener interface (`jdk.rs` models no window events).
pub trait WindowFocusListener {
    /// Java `windowGainedFocus(WindowEvent)`.
    fn window_gained_focus(&self);
    /// Java `windowLostFocus(WindowEvent)`.
    fn window_lost_focus(&self);
}

/// The members `AbstractFrame` declares abstract, or declares and a subclass
/// overrides, dispatched through the subclass object.  Every default method
/// holds the `AbstractFrame` body.
pub trait AbstractFrameVirtual {
    /// The `AbstractFrame` part of the object.
    fn abstract_frame(&self) -> &AbstractFrame;

    /// Java abstract `menuFileAction(ActionEvent)`.
    fn menu_file_action(&self, action_event: &ActionEvent);
    /// Java abstract `menuToolsAction(ActionEvent)`.
    fn menu_tools_action(&self, action_event: &ActionEvent);
    /// Java abstract `menuViewAction(ActionEvent)`.
    fn menu_view_action(&self, action_event: &ActionEvent);
    /// Java abstract `menuOptionsAction(ActionEvent)`.
    fn menu_options_action(&self, action_event: &ActionEvent);
    /// Java abstract `menuHelpAction(ActionEvent)`.
    fn menu_help_action(&self, action_event: &ActionEvent);
    /// Java abstract `getFrameType()`.  `ManagerFrame` returns null.
    fn get_frame_type(&self) -> Option<FrameType>;
    /// Java abstract `cancel()`.
    fn cancel(&self);
    /// Java abstract `save(AxisID)`.
    fn save(&self, axis_id: Option<AxisID>);
    /// Java abstract `saveAs()`.
    fn save_as(&self);
    /// Java abstract `close()`.
    fn close(&self);

    /// Java `setVisible(boolean)` (overrides `Window.setVisible`).
    fn set_visible(&self, visible: bool) {
        let frame = self.abstract_frame();
        if visible {
            let director = &*etomo_director::INSTANCE;
            let frame_type = self.get_frame_type();
            let location = frame_type.and_then(|frame_type| {
                director.with_user_configuration(|user_configuration| {
                    if user_configuration.is_last_location_set(frame_type) {
                        Some((
                            user_configuration.get_last_location_x(frame_type),
                            user_configuration.get_last_location_y(frame_type),
                        ))
                    } else {
                        None
                    }
                })
            });
            if !director.get_arguments().is_ignore_loc()
                && let Some((x, y)) = location
            {
                frame.set_location(x, y);
            }
        }
        frame.set_visible_super(visible);
    }

    /// Java package-private `getAxisID()`.
    fn get_axis_id(&self) -> Option<AxisID> {
        Some(AxisID::Only)
    }

    /// Java package-private `pack(boolean)`.
    fn pack_boolean(&self, force: bool) {
        let auto_fit = etomo_director::INSTANCE.with_user_configuration(|c| c.is_auto_fit());
        if !force && !auto_fit {
            self.set_visible(true);
        } else {
            let frame = self.abstract_frame();
            let mut bounds = frame.get_size();
            bounds.height += 1;
            bounds.width += 1;
            frame.set_size(bounds);
            // `try { super.pack(); } catch (NullPointerException e) {
            // e.printStackTrace(); }` - Swing layout: Window.pack().
        }
    }

    /// Java package-private `repaint(AxisID)`.
    fn repaint(&self, axis_id: Option<AxisID>) {
        let _ = axis_id;
        // Swing painting: repaint().
    }

    /// Java package-private `pack(AxisID)`.
    fn pack_axis_id(&self, axis_id: Option<AxisID>) {
        let _ = axis_id;
        self.pack_void();
    }

    /// Java package-private `pack(AxisID, boolean)`.
    fn pack_axis_id_boolean(&self, axis_id: Option<AxisID>, force: bool) {
        let _ = axis_id;
        self.pack_boolean(force);
    }

    /// Java `Window.pack()`, which `EtomoFrame` overrides.  In a frame that
    /// does not override it, it is Swing layout only.
    fn pack_void(&self) {
        // Swing layout: Window.pack().
    }

    /// Java package-private `menuFileMRUListAction(ActionEvent)`: empty.
    fn menu_file_mru_list_action(&self, event: &ActionEvent) {
        let _ = event;
    }

    // --- display* functions (EtomoFrame overrides some of them) ---

    /// Java `displayMessage(BaseManager, String, String, AxisID)`.  Open a
    /// message dialog.
    fn display_message_base_manager_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.abstract_frame()
            .open_message_dialog_base_manager_axis_id_string_string(
                manager, axis_id, message, title,
            );
    }

    /// Java `displayMessage(BaseManager, Component, String, String, AxisID)`.
    fn display_message_base_manager_component_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        message: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.abstract_frame()
            .open_message_dialog_base_manager_component_axis_id_string_string(
                manager,
                parent_component,
                axis_id,
                message,
                title,
            );
    }

    /// Java `displayYesNoMessage(BaseManager, Component, String, AxisID)`.
    fn display_yes_no_message_base_manager_component_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        message: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.abstract_frame()
            .open_yes_no_dialog_base_manager_component_axis_id_string(
                manager,
                parent_component,
                axis_id,
                message,
            )
    }

    /// Java `displayWarningMessage(BaseManager, Component, String, String,
    /// AxisID)`.
    fn display_warning_message_base_manager_component_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        message: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.abstract_frame()
            .open_warning_message_dialog_base_manager_component_axis_id_string_string(
                manager,
                parent_component,
                axis_id,
                message,
                title,
            );
    }

    /// Java `displayMessage(BaseManager, Component, String[], String, AxisID)`.
    fn display_message_base_manager_component_string_array_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        message: &[String],
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.abstract_frame()
            .open_message_dialog_base_manager_component_axis_id_string_array_string(
                manager,
                parent_component,
                axis_id,
                message,
                title,
            );
    }

    /// Java `displayMessage(BaseManager, String, String)`.  Open a message
    /// dialog.
    fn display_message_base_manager_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        title: Option<&str>,
    ) {
        self.abstract_frame()
            .open_message_dialog_base_manager_axis_id_string_string(
                manager,
                Some(AxisID::Only),
                message,
                title,
            );
    }

    /// Java `displayInfoMessage(BaseManager, Component, String, String,
    /// AxisID)`.
    fn display_info_message(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        message: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.abstract_frame()
            .open_info_message_dialog_base_manager_component_axis_id_string_string(
                manager,
                parent_component,
                axis_id,
                message,
                title,
            );
    }

    /// Java `displayYesNoCancelMessage(BaseManager, String, AxisID)`.
    fn display_yes_no_cancel_message(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> i32 {
        self.abstract_frame()
            .open_yes_no_cancel_dialog_base_manager_axis_id_string(manager, axis_id, message)
    }

    /// Java `displayYesNoMessage(BaseManager, String[], AxisID)`.
    fn display_yes_no_message_base_manager_string_array_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        axis_id: Option<AxisID>,
    ) -> bool {
        self.abstract_frame()
            .open_yes_no_dialog_base_manager_axis_id_string_array(manager, axis_id, message)
    }

    /// Java `displayYesNoMessage(BaseManager, String, AxisID)`.
    fn display_yes_no_message_base_manager_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.abstract_frame()
            .open_yes_no_dialog_base_manager_axis_id_string(manager, axis_id, message)
    }

    /// Java `openYesNoDialogWithDefaultNo(BaseManager, String, String,
    /// AxisID)`.
    fn open_yes_no_dialog_with_default_no(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.abstract_frame()
            .open_yes_no_dialog_base_manager_axis_id_string_string_int_boolean(
                manager, axis_id, message, title, NO_INDEX, true,
            )
    }

    /// Java `displayDeleteMessage(BaseManager, String[], AxisID)`.
    fn display_delete_message(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        axis_id: Option<AxisID>,
    ) -> bool {
        self.abstract_frame()
            .open_delete_dialog_base_manager_axis_id_string_array(manager, axis_id, message)
    }

    /// Java `displayMessage(BaseManager, String[], String, AxisID)`.  Open a
    /// message dialog.
    fn display_message_base_manager_string_array_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.abstract_frame()
            .open_message_dialog_base_manager_axis_id_string_array_string(
                manager, axis_id, message, title,
            );
    }

    /// Java `displayErrorMessage(BaseManager, ProcessMessages, String,
    /// AxisID)`.
    fn display_error_message(
        &self,
        manager: Option<&'static dyn BaseManager>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.abstract_frame()
            .open_error_message_dialog_base_manager_axis_id_process_messages_string(
                manager,
                axis_id,
                process_messages,
                title,
            );
    }

    /// Java `displayMessage(BaseManager, ProcessMessages, String, AxisID)`.
    fn display_message_base_manager_process_messages_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.abstract_frame()
            .open_message_dialog_base_manager_axis_id_process_messages_string(
                manager,
                axis_id,
                process_messages,
                title,
            );
    }

    /// Java `displayYesNoWarningDialog(BaseManager, String, AxisID)`.
    fn display_yes_no_warning_dialog(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.abstract_frame()
            .open_yes_no_warning_dialog_base_manager_axis_id_string(manager, axis_id, message)
    }

    /// Java `displayWarningMessage(BaseManager, ProcessMessages, String,
    /// AxisID)`.
    fn display_warning_message_base_manager_process_messages_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.abstract_frame()
            .open_warning_message_dialog_base_manager_axis_id_process_messages_string(
                manager,
                axis_id,
                process_messages,
                title,
            );
    }
}

/// Java package-private `abstract class AbstractFrame extends JFrame
/// implements UIComponent, SwingComponent`.
pub struct AbstractFrame {
    /// Java `private boolean verbose = false`.
    verbose: Cell<bool>,

    // --- the JFrame state this translation models ---
    /// `JFrame.getContentPane()`: the root of the frame's component tree.
    content_pane: Rc<JComponent>,
    /// `Frame.setTitle` / `getTitle`.
    title: RefCell<String>,
    /// `Window.isVisible`.
    frame_visible: Cell<bool>,
    /// `Component.isDisplayable`: true from construction until `dispose()`.
    displayable: Cell<bool>,
    /// `Component.getLocation` / `setLocation`.
    location: Cell<Point>,
    /// `Component.getSize` / `setBounds`.
    size: Cell<Dimension>,
    /// `JFrame.setJMenuBar`.
    j_menu_bar: RefCell<Option<Rc<JComponent>>>,
    /// `Window.addWindowFocusListener`.
    window_focus_listeners: RefCell<Vec<Rc<dyn WindowFocusListener>>>,

    /// The subclass object, for virtual dispatch (Java `this`).
    this: RefCell<Weak<dyn AbstractFrameVirtual>>,
}

/// Placeholder type for an unset `this`.
struct NoSubclass;

impl AbstractFrameVirtual for NoSubclass {
    fn abstract_frame(&self) -> &AbstractFrame {
        unreachable!("AbstractFrame used before its subclass was installed")
    }
    fn menu_file_action(&self, _action_event: &ActionEvent) {}
    fn menu_tools_action(&self, _action_event: &ActionEvent) {}
    fn menu_view_action(&self, _action_event: &ActionEvent) {}
    fn menu_options_action(&self, _action_event: &ActionEvent) {}
    fn menu_help_action(&self, _action_event: &ActionEvent) {}
    fn get_frame_type(&self) -> Option<FrameType> {
        None
    }
    fn cancel(&self) {}
    fn save(&self, _axis_id: Option<AxisID>) {}
    fn save_as(&self) {}
    fn close(&self) {}
}

impl AbstractFrame {
    /// Java implicit constructor (`JFrame()`): a new, not yet visible frame
    /// with an empty content pane.  The subclass must call
    /// [`AbstractFrame::set_this`] once it has been created.
    pub fn new() -> AbstractFrame {
        let content_pane = JComponent::new_panel();
        // A JFrame is created invisible.
        content_pane.set_visible(false);
        AbstractFrame {
            verbose: Cell::new(false),
            content_pane,
            title: RefCell::new(String::new()),
            frame_visible: Cell::new(false),
            displayable: Cell::new(true),
            location: Cell::new(Point { x: 0, y: 0 }),
            size: Cell::new(Dimension {
                width: 0,
                height: 0,
            }),
            j_menu_bar: RefCell::new(None),
            window_focus_listeners: RefCell::new(Vec::new()),
            this: RefCell::new(Weak::<NoSubclass>::new() as Weak<dyn AbstractFrameVirtual>),
        }
    }

    /// Installs the subclass object for virtual dispatch (Rust-only).
    pub fn set_this(&self, this: Weak<dyn AbstractFrameVirtual>) {
        *self.this.borrow_mut() = this;
    }

    /// The subclass object as a `Weak` (Java `this`).
    pub fn this_weak(&self) -> Weak<dyn AbstractFrameVirtual> {
        self.this.borrow().clone()
    }

    /// The subclass object (Java `this` seen through a virtual call).
    pub fn this(&self) -> Option<Rc<dyn AbstractFrameVirtual>> {
        self.this.borrow().upgrade()
    }

    // --- javax.swing.JFrame / java.awt.Window members ---

    /// Java `JFrame.getContentPane()`.
    pub fn get_content_pane(&self) -> Rc<JComponent> {
        self.content_pane.clone()
    }

    /// Java `Frame.setTitle(String)`.
    pub fn set_title(&self, title: Option<&str>) {
        *self.title.borrow_mut() = title.unwrap_or("").to_owned();
    }

    /// Java `Frame.getTitle()`.
    pub fn get_title(&self) -> String {
        self.title.borrow().clone()
    }

    /// Java `super.setVisible(boolean)` (`Window.setVisible`).  The content
    /// pane mirrors the frame's visibility so that a component search limited
    /// to showing components skips a hidden frame, as `isShowing` does.
    pub fn set_visible_super(&self, visible: bool) {
        self.frame_visible.set(visible);
        self.content_pane.set_visible(visible);
        if visible {
            self.displayable.set(true);
        }
    }

    /// Java `Window.isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.frame_visible.get()
    }

    /// Java `Component.isDisplayable()`.
    pub fn is_displayable(&self) -> bool {
        self.displayable.get()
    }

    /// Java `Window.dispose()`.
    pub fn dispose(&self) {
        self.frame_visible.set(false);
        self.content_pane.set_visible(false);
        self.displayable.set(false);
    }

    /// Java `Component.setLocation(int, int)`.
    pub fn set_location(&self, x: i32, y: i32) {
        self.location.set(Point { x, y });
    }

    /// Java `Component.getLocation()`.
    pub fn get_location(&self) -> Point {
        self.location.get()
    }

    /// Java `Component.getSize()`.
    pub fn get_size(&self) -> Dimension {
        self.size.get()
    }

    /// Java `Component.setBounds(Rectangle)`, size part.
    pub fn set_size(&self, size: Dimension) {
        self.size.set(size);
    }

    /// Java `JFrame.setJMenuBar(JMenuBar)`.
    pub fn set_j_menu_bar(&self, menu_bar: Option<Rc<JComponent>>) {
        *self.j_menu_bar.borrow_mut() = menu_bar;
    }

    /// Java `JFrame.getJMenuBar()`.
    pub fn get_j_menu_bar(&self) -> Option<Rc<JComponent>> {
        self.j_menu_bar.borrow().clone()
    }

    /// Java `Window.toFront()`.
    pub fn to_front(&self) {
        // Swing window stacking: toFront().
    }

    /// Java `Container.doLayout()`.
    pub fn do_layout(&self) {
        // Swing layout: doLayout().
    }

    /// Java `Container.validate()`.
    pub fn validate(&self) {
        // Swing layout: validate().
    }

    /// Java `Window.addWindowFocusListener(WindowFocusListener)`.
    pub fn add_window_focus_listener(&self, listener: Rc<dyn WindowFocusListener>) {
        self.window_focus_listeners.borrow_mut().push(listener);
    }

    /// Delivers a window focus event to the focus listeners, as AWT does when
    /// the frame gains or loses focus (Rust-only entry point for the
    /// presentation layer).
    pub fn process_window_focus_event(&self, gained: bool) {
        let listeners: Vec<_> = self.window_focus_listeners.borrow().clone();
        for listener in listeners {
            if gained {
                listener.window_gained_focus();
            } else {
                listener.window_lost_focus();
            }
        }
    }

    // --- AbstractFrame ---

    /// Java final `setVerbose(boolean)`.
    pub fn set_verbose(&self, verbose: bool) {
        self.verbose.set(verbose);
    }

    /// Java package-private `repaintWindow()`.
    pub fn repaint_window(&self) {
        self.repaint_container(&self.content_pane);
        // Swing painting: this.repaint().
    }

    /// Java private `repaintContainer(Container)`.
    fn repaint_container(&self, container: &Rc<JComponent>) {
        let comps = container.get_components();
        for comp in &comps {
            // Every Swing component is a Container.
            self.repaint_container(comp);
            // Swing painting: comps[i].repaint().
        }
    }

    // Standardize default dialogs.

    /// Java `openInfoMessageDialog(BaseManager, Component, AxisID, String,
    /// String[], ProcessMessages, String, Boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_info_message_dialog_base_manager_component_axis_id_string_string_array_process_messages_string_boolean(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&ProcessMessages>,
        title: Option<&str>,
        modal: Option<bool>,
    ) {
        self.show_option_pane_base_manager_component_axis_id_string_string_array_process_messages_message_type_string_int_boolean(
            manager,
            parent_component,
            axis_id,
            message,
            message_array,
            process_messages,
            Some(process_messages::MessageType::Info),
            title,
            INFORMATION_MESSAGE,
            modal,
        );
    }

    /// Java `openMessageDialog(BaseManager, Component, AxisID, String,
    /// String[], ProcessMessages, String, Boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_message_dialog_base_manager_component_axis_id_string_string_array_process_messages_string_boolean(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&ProcessMessages>,
        title: Option<&str>,
        modal: Option<bool>,
    ) {
        self.show_option_pane_base_manager_component_axis_id_string_string_array_process_messages_message_type_string_int_boolean(
            manager,
            parent_component,
            axis_id,
            message,
            message_array,
            process_messages,
            None,
            title,
            ERROR_MESSAGE,
            modal,
        );
    }

    /// Java `openWarningMessageDialog(BaseManager, Component, AxisID, String,
    /// String[], ProcessMessages, String, Boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_warning_message_dialog_base_manager_component_axis_id_string_string_array_process_messages_string_boolean(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&ProcessMessages>,
        title: Option<&str>,
        modal: Option<bool>,
    ) {
        self.show_option_pane_base_manager_component_axis_id_string_string_array_process_messages_message_type_string_int_boolean(
            manager,
            parent_component,
            axis_id,
            message,
            message_array,
            process_messages,
            Some(process_messages::MessageType::Warning),
            title,
            WARNING_MESSAGE,
            modal,
        );
    }

    /// Java `openErrorMessageDialog(BaseManager, Component, AxisID, String,
    /// String[], ProcessMessages, String, Boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_error_message_dialog_base_manager_component_axis_id_string_string_array_process_messages_string_boolean(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&ProcessMessages>,
        title: Option<&str>,
        modal: Option<bool>,
    ) {
        self.show_option_pane_base_manager_component_axis_id_string_string_array_process_messages_message_type_string_int_boolean(
            manager,
            parent_component,
            axis_id,
            message,
            message_array,
            process_messages,
            Some(process_messages::MessageType::Error),
            title,
            ERROR_MESSAGE,
            modal,
        );
    }

    /// Java private `showOptionPane(BaseManager, Component, AxisID, String,
    /// String[], ProcessMessages, ProcessMessages.MessageType, String, int,
    /// Boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn show_option_pane_base_manager_component_axis_id_string_string_array_process_messages_message_type_string_int_boolean(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&ProcessMessages>,
        process_message_type: Option<process_messages::MessageType>,
        title: Option<&str>,
        message_type: i32,
        modal: Option<bool>,
    ) {
        let ok_labels: Vec<String> = OK_LABEL_ARRAY.iter().map(|s| s.to_string()).collect();
        let wrapped = self.wrap_factory(
            message,
            message_array,
            process_messages,
            process_message_type,
        );
        self.show_option_pane_base_manager_component_axis_id_string_array_string_int_int_object_boolean_string_array_boolean(
            manager,
            parent_component,
            axis_id,
            wrapped,
            title,
            DEFAULT_OPTION,
            message_type,
            None,
            false,
            Some(&ok_labels),
            modal,
        );
    }

    // Standardize question dialogs.

    /// Java `openYesNoDialog(BaseManager, Component, AxisID, String, String[],
    /// String, String)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_yes_no_dialog_base_manager_component_axis_id_string_string_array_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        initial_value: Option<&str>,
    ) -> Option<i32> {
        let labels: Vec<String> = YES_NO_LABEL_ARRAY.iter().map(|s| s.to_string()).collect();
        Some(self.show_option_confirm_pane_base_manager_component_axis_id_string_string_array_string_integer_integer_string_string_array(
            manager,
            parent_component,
            axis_id,
            message,
            message_array,
            title,
            Some(YES_NO_OPTION),
            None,
            initial_value,
            Some(&labels),
        ))
    }

    /// Java `openYesNoCancelDialog(BaseManager, Component, AxisID, String,
    /// String[], String, String)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_yes_no_cancel_dialog_base_manager_component_axis_id_string_string_array_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        initial_value: Option<&str>,
    ) -> i32 {
        let labels: Vec<String> = YES_NO_CANCEL_LABEL_ARRAY
            .iter()
            .map(|s| s.to_string())
            .collect();
        self.show_option_confirm_pane_base_manager_component_axis_id_string_string_array_string_integer_integer_string_string_array(
            manager,
            parent_component,
            axis_id,
            message,
            message_array,
            title,
            Some(YES_NO_CANCEL_OPTION),
            None,
            initial_value,
            Some(&labels),
        )
    }

    /// Java `openDeleteDialog(BaseManager, Component, AxisID, String,
    /// String[], String, String)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_delete_dialog_base_manager_component_axis_id_string_string_array_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        initial_value: Option<&str>,
    ) -> i32 {
        let labels: Vec<String> = DELETE_NO_LABEL_ARRAY
            .iter()
            .map(|s| s.to_string())
            .collect();
        self.show_option_confirm_pane_base_manager_component_axis_id_string_string_array_string_integer_integer_string_string_array(
            manager,
            parent_component,
            axis_id,
            message,
            message_array,
            Some(title.unwrap_or("Delete File?")),
            None,
            None,
            initial_value,
            Some(&labels),
        )
    }

    /// Java `openYesNoWarningDialog(BaseManager, Component, AxisID, String,
    /// String[], String, String)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_yes_no_warning_dialog_base_manager_component_axis_id_string_string_array_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        initial_value: Option<&str>,
    ) -> i32 {
        let labels: Vec<String> = YES_NO_LABEL_ARRAY.iter().map(|s| s.to_string()).collect();
        self.show_option_confirm_pane_base_manager_component_axis_id_string_string_array_string_integer_integer_string_string_array(
            manager,
            parent_component,
            axis_id,
            message,
            message_array,
            Some(title.unwrap_or("Etomo Warning")),
            Some(YES_NO_OPTION),
            Some(WARNING_MESSAGE),
            Some(initial_value.unwrap_or(YES_NO_LABEL_ARRAY[NO_INDEX])),
            Some(&labels),
        )
    }

    /// Java private `showOptionConfirmPane(BaseManager, Component, AxisID,
    /// String, String[], String, Integer, Integer, String, String[])`.
    #[allow(clippy::too_many_arguments)]
    fn show_option_confirm_pane_base_manager_component_axis_id_string_string_array_string_integer_integer_string_string_array(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        option_type: Option<i32>,
        message_type: Option<i32>,
        initial_value: Option<&str>,
        option_strings: Option<&[String]>,
    ) -> i32 {
        let wrapped = self.wrap_factory(message, message_array, None, None);
        self.show_option_pane_base_manager_component_axis_id_string_array_string_int_int_object_boolean_string_array_boolean(
            manager,
            parent_component,
            axis_id,
            wrapped,
            Some(title.unwrap_or(ETOMO_QUESTION)),
            option_type.unwrap_or(DEFAULT_OPTION),
            message_type.unwrap_or(QUESTION_MESSAGE),
            initial_value,
            option_strings.is_some(),
            option_strings,
            None,
        )
    }

    //

    // The `display*` members are virtual (see `AbstractFrameVirtual`); the
    // remaining `AbstractFrame` members follow.

    /// Java `openMessageDialog(BaseManager, AxisID, String, String)`.  Open a
    /// message dialog with a wrapped message with the dataset appended.
    pub fn open_message_dialog_base_manager_axis_id_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_string(message);
        self.show_option_pane_base_manager_axis_id_string_array_string_int(
            manager,
            axis_id,
            wrapped,
            title,
            ERROR_MESSAGE,
        );
    }

    /// Java `openMessageDialog(BaseManager, Component, AxisID, String,
    /// String)`.
    pub fn open_message_dialog_base_manager_component_axis_id_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_string(message);
        self.show_option_pane_base_manager_component_axis_id_string_array_string_int_boolean(
            manager,
            parent_component,
            axis_id,
            wrapped,
            title,
            ERROR_MESSAGE,
            None,
        );
    }

    /// Java `openYesNoDialog(BaseManager, Component, AxisID, String)`.  Open a
    /// Yes or No question dialog.
    pub fn open_yes_no_dialog_base_manager_component_axis_id_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
    ) -> bool {
        let labels: Vec<String> = YES_NO_LABEL_ARRAY.iter().map(|s| s.to_string()).collect();
        let wrapped = self.wrap_string(message);
        let result = self
            .show_option_confirm_pane_base_manager_component_axis_id_string_array_string_int_string_array(
                manager,
                parent_component,
                axis_id,
                wrapped,
                Some(ETOMO_QUESTION),
                YES_NO_OPTION,
                Some(&labels),
            );
        result == YES_OPTION
    }

    /// Java `openWarningMessageDialog(BaseManager, Component, AxisID, String,
    /// String)`.
    pub fn open_warning_message_dialog_base_manager_component_axis_id_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parentc_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_string(message);
        self.show_option_pane_base_manager_component_axis_id_string_array_string_int_boolean(
            manager,
            parentc_component,
            axis_id,
            wrapped,
            title,
            WARNING_MESSAGE,
            None,
        );
    }

    /// Java `openMessageDialog(BaseManager, Component, AxisID, String[],
    /// String)`.
    pub fn open_message_dialog_base_manager_component_axis_id_string_array_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parentc_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: &[String],
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_string_array(message);
        self.show_option_pane_base_manager_component_axis_id_string_array_string_int_boolean(
            manager,
            parentc_component,
            axis_id,
            wrapped,
            title,
            ERROR_MESSAGE,
            None,
        );
    }

    /// Java `openWarningMessageDialog(BaseManager, AxisID, ProcessMessages,
    /// String)`.
    pub fn open_warning_message_dialog_base_manager_axis_id_process_messages_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_warning(process_messages);
        // ERROR_MESSAGE as the source passes it; showOptionDialog switches the
        // icon to a warning when the title or text says "warning".
        self.show_option_pane_base_manager_axis_id_string_array_string_int(
            manager,
            axis_id,
            wrapped,
            title,
            ERROR_MESSAGE,
        );
    }

    /// Java `openErrorMessageDialog(BaseManager, AxisID, ProcessMessages,
    /// String)`.
    pub fn open_error_message_dialog_base_manager_axis_id_process_messages_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_error(process_messages);
        self.show_option_pane_base_manager_axis_id_string_array_string_int(
            manager,
            axis_id,
            wrapped,
            title,
            ERROR_MESSAGE,
        );
    }

    /// Java `openMessageDialog(BaseManager, AxisID, ProcessMessages, String)`.
    pub fn open_message_dialog_base_manager_axis_id_process_messages_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_messages_process_messages(process_messages);
        self.show_option_pane_base_manager_axis_id_string_array_string_int(
            manager,
            axis_id,
            wrapped,
            title,
            ERROR_MESSAGE,
        );
    }

    /// Java `openYesNoWarningDialog(BaseManager, AxisID, String)`.
    pub fn open_yes_no_warning_dialog_base_manager_axis_id_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
    ) -> bool {
        let labels: Vec<String> = YES_NO_LABEL_ARRAY.iter().map(|s| s.to_string()).collect();
        let wrapped = self.wrap_string(message);
        let result = self
            .show_option_pane_base_manager_axis_id_string_array_string_int_int_object_boolean_string_array(
                manager,
                axis_id,
                wrapped,
                Some("Etomo Warning"),
                YES_NO_OPTION,
                WARNING_MESSAGE,
                Some(YES_NO_LABEL_ARRAY[NO_INDEX]),
                false,
                Some(&labels),
            );
        result == 0
    }

    /// Java `openYesNoDialog(BaseManager, AxisID, String)`.  Open a Yes or No
    /// question dialog.
    pub fn open_yes_no_dialog_base_manager_axis_id_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
    ) -> bool {
        let labels: Vec<String> = YES_NO_LABEL_ARRAY.iter().map(|s| s.to_string()).collect();
        let wrapped = self.wrap_string(message);
        let result = self
            .show_option_confirm_pane_base_manager_axis_id_string_array_string_int_string_array(
                manager,
                axis_id,
                wrapped,
                Some(ETOMO_QUESTION),
                YES_NO_OPTION,
                Some(&labels),
            );
        result == YES_OPTION
    }

    /// Java `openYesNoDialog(BaseManager, AxisID, String, String, int,
    /// boolean)`.  Open a Yes or No question dialog.  Control which option is
    /// the default.
    pub fn open_yes_no_dialog_base_manager_axis_id_string_string_int_boolean(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        title: Option<&str>,
        initial_value_index: usize,
        override_default_labels: bool,
    ) -> bool {
        let labels: Vec<String> = YES_NO_LABEL_ARRAY.iter().map(|s| s.to_string()).collect();
        let wrapped = self.wrap_string(message);
        let result = self
            .show_option_confirm_pane_base_manager_axis_id_string_array_string_int_string_boolean_string_array(
                manager,
                axis_id,
                wrapped,
                title,
                YES_NO_OPTION,
                Some(YES_NO_LABEL_ARRAY[initial_value_index]),
                override_default_labels,
                Some(&labels),
            );
        result == YES_OPTION
    }

    /// Java `openDeleteDialog(BaseManager, AxisID, String[])`.  Open a Yes or
    /// No question dialog.
    pub fn open_delete_dialog_base_manager_axis_id_string_array(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: &[String],
    ) -> bool {
        let labels: Vec<String> = vec!["Delete".to_owned(), NO.to_owned()];
        let wrapped = self.wrap_string_array(message);
        let result = self
            .show_option_pane_base_manager_axis_id_string_array_string_int_int_object_boolean_string_array(
                manager,
                axis_id,
                wrapped,
                Some("Delete File?"),
                DEFAULT_OPTION,
                QUESTION_MESSAGE,
                None,
                true,
                Some(&labels),
            );
        result == 0
    }

    /// Java `openYesNoDialog(BaseManager, AxisID, String[])`.  Open a Yes or No
    /// question dialog.
    pub fn open_yes_no_dialog_base_manager_axis_id_string_array(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: &[String],
    ) -> bool {
        let labels: Vec<String> = YES_NO_LABEL_ARRAY.iter().map(|s| s.to_string()).collect();
        let wrapped = self.wrap_string_array(message);
        let result = self
            .show_option_confirm_pane_base_manager_axis_id_string_array_string_int_string_array(
                manager,
                axis_id,
                wrapped,
                Some(ETOMO_QUESTION),
                YES_NO_OPTION,
                Some(&labels),
            );
        result == YES_OPTION
    }

    /// Java `openInfoMessageDialog(BaseManager, Component, AxisID, String,
    /// String)`.
    pub fn open_info_message_dialog_base_manager_component_axis_id_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_string(message);
        self.show_option_pane_base_manager_component_axis_id_string_array_string_int_boolean(
            manager,
            parent_component,
            axis_id,
            wrapped,
            title,
            INFORMATION_MESSAGE,
            None,
        );
    }

    /// Java `openMessageDialog(BaseManager, AxisID, String[], String)`.  Open
    /// a message dialog.
    pub fn open_message_dialog_base_manager_axis_id_string_array_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: &[String],
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_string_array(message);
        self.show_option_pane_base_manager_axis_id_string_array_string_int(
            manager,
            axis_id,
            wrapped,
            title,
            ERROR_MESSAGE,
        );
    }

    /// Java `openYesNoCancelDialog(BaseManager, AxisID, String)`.  Open a Yes,
    /// No or Cancel question dialog; returns the state of the user's
    /// selection.
    pub fn open_yes_no_cancel_dialog_base_manager_axis_id_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
    ) -> i32 {
        let labels: Vec<String> = vec![YES.to_owned(), NO.to_owned(), CANCEL.to_owned()];
        let wrapped = self.wrap_string(message);
        self.show_option_confirm_pane_base_manager_axis_id_string_array_string_int_string_array(
            manager,
            axis_id,
            wrapped,
            Some(ETOMO_QUESTION),
            YES_NO_CANCEL_OPTION,
            Some(&labels),
        )
    }

    /// Java private `showOptionConfirmPane(BaseManager, AxisID, String[],
    /// String, int, String[])`.
    fn show_option_confirm_pane_base_manager_axis_id_string_array_string_int_string_array(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: Option<Vec<String>>,
        title: Option<&str>,
        option_type: i32,
        option_strings: Option<&[String]>,
    ) -> i32 {
        self.show_option_pane_base_manager_axis_id_string_array_string_int_int_object_boolean_string_array(
            manager,
            axis_id,
            message,
            title,
            option_type,
            QUESTION_MESSAGE,
            None,
            false,
            option_strings,
        )
    }

    /// Java private `showOptionConfirmPane(BaseManager, AxisID, String[],
    /// String, int, String, boolean, String[])`.
    #[allow(clippy::too_many_arguments)]
    fn show_option_confirm_pane_base_manager_axis_id_string_array_string_int_string_boolean_string_array(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: Option<Vec<String>>,
        title: Option<&str>,
        option_type: i32,
        initial_value: Option<&str>,
        override_default_labels: bool,
        option_strings: Option<&[String]>,
    ) -> i32 {
        self.show_option_pane_base_manager_axis_id_string_array_string_int_int_object_boolean_string_array(
            manager,
            axis_id,
            message,
            title,
            option_type,
            QUESTION_MESSAGE,
            initial_value,
            override_default_labels,
            option_strings,
        )
    }

    /// Java private `wrapFactory(String, String[], ProcessMessages,
    /// ProcessMessages.MessageType)`.
    fn wrap_factory(
        &self,
        message: Option<&str>,
        message_array: Option<&[String]>,
        process_messages: Option<&ProcessMessages>,
        process_message_type: Option<process_messages::MessageType>,
    ) -> Option<Vec<String>> {
        if message.is_some() {
            return self.wrap_string(message);
        }
        if let Some(message_array) = message_array {
            return self.wrap_string_array(message_array);
        }
        if let Some(process_messages) = process_messages {
            self.wrap_messages_process_messages_message_type(process_messages, process_message_type)
        } else {
            // new Exception("Failed popup.  No message").printStackTrace()
            eprintln!("java.lang.Exception: Failed popup.  No message");
            None
        }
    }

    /// Java private final `wrapMessages(ProcessMessages,
    /// ProcessMessages.MessageType)`.
    fn wrap_messages_process_messages_message_type(
        &self,
        process_messages: &ProcessMessages,
        process_message_type: Option<process_messages::MessageType>,
    ) -> Option<Vec<String>> {
        let Some(process_message_type) = process_message_type else {
            return self.wrap_messages_process_messages(process_messages);
        };
        let mut message_array: Option<Vec<String>> = None;
        for i in 0..process_messages.size(process_message_type) {
            message_array = Some(popup_tool::wrap_message(
                process_messages.get(process_message_type, i),
                message_array,
            ));
        }
        self.to_string_array(message_array)
    }

    /// Java private final `wrapWarning(ProcessMessages)`.
    fn wrap_warning(&self, process_messages: &ProcessMessages) -> Option<Vec<String>> {
        let mut message_array: Option<Vec<String>> = None;
        for i in 0..process_messages.size(process_messages::MessageType::Warning) {
            message_array = Some(popup_tool::wrap_message(
                process_messages.get(process_messages::MessageType::Warning, i),
                message_array,
            ));
        }
        self.to_string_array(message_array)
    }

    /// Java private final `wrapError(ProcessMessages)`.  Add the current
    /// dataset name to the message and wrap.
    fn wrap_error(&self, process_messages: &ProcessMessages) -> Option<Vec<String>> {
        let mut message_array: Option<Vec<String>> = None;
        for i in 0..process_messages.size(process_messages::MessageType::Error) {
            message_array = Some(popup_tool::wrap_message(
                process_messages.get(process_messages::MessageType::Error, i),
                message_array,
            ));
        }
        self.to_string_array(message_array)
    }

    /// Java private final `wrapMessages(ProcessMessages)`.  Add the current
    /// dataset name to the message and wrap.
    fn wrap_messages_process_messages(
        &self,
        process_messages: &ProcessMessages,
    ) -> Option<Vec<String>> {
        let mut message_array: Option<Vec<String>> = None;
        let mut list_type_iterator = process_messages.list_type_iterator();
        while list_type_iterator.has_next() {
            let list_type = list_type_iterator.next();
            if let Some(iterator) = list_type.and_then(|lt| process_messages.iterator(lt)) {
                for message in iterator {
                    message_array = Some(popup_tool::wrap_message(
                        Some(message.as_str()),
                        message_array,
                    ));
                }
            }
            message_array = Some(popup_tool::linefeed(message_array));
        }
        self.to_string_array(message_array)
    }

    /// Java private `wrap(String)`.  Add the current dataset name to the
    /// message and wrap.
    fn wrap_string(&self, message: Option<&str>) -> Option<Vec<String>> {
        let message_array = popup_tool::wrap_message(message, None);
        self.to_string_array(Some(message_array))
    }

    /// Java private `wrap(String[])`.  Add the current dataset name to the
    /// message and wrap.
    fn wrap_string_array(&self, message: &[String]) -> Option<Vec<String>> {
        let mut message_array: Option<Vec<String>> = None;
        for item in message {
            message_array = Some(popup_tool::wrap_message(Some(item.as_str()), message_array));
        }
        self.to_string_array(message_array)
    }

    /// Java private final `toStringArray(ArrayList)`.
    fn to_string_array(&self, array_list: Option<Vec<String>>) -> Option<Vec<String>> {
        // Java copies the one-element list into a new array; both arms return
        // the list's elements.
        array_list
    }

    /// Java private `showOptionPane(BaseManager, AxisID, String[], String,
    /// int)`.
    fn show_option_pane_base_manager_axis_id_string_array_string_int(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: Option<Vec<String>>,
        title: Option<&str>,
        message_type: i32,
    ) {
        let ok_labels: Vec<String> = OK_LABEL_ARRAY.iter().map(|s| s.to_string()).collect();
        self.show_option_pane_base_manager_axis_id_string_array_string_int_int_object_boolean_string_array(
            manager,
            axis_id,
            message,
            title,
            DEFAULT_OPTION,
            message_type,
            None,
            false,
            Some(&ok_labels),
        );
    }

    /// Java private `showOptionPane(BaseManager, Component, AxisID, String[],
    /// String, int, Boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn show_option_pane_base_manager_component_axis_id_string_array_string_int_boolean(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<Vec<String>>,
        title: Option<&str>,
        message_type: i32,
        modal: Option<bool>,
    ) {
        let ok_labels: Vec<String> = OK_LABEL_ARRAY.iter().map(|s| s.to_string()).collect();
        self.show_option_pane_base_manager_component_axis_id_string_array_string_int_int_object_boolean_string_array_boolean(
            manager,
            parent_component,
            axis_id,
            message,
            title,
            DEFAULT_OPTION,
            message_type,
            None,
            false,
            Some(&ok_labels),
            modal,
        );
    }

    /// Java private `showOptionConfirmPane(BaseManager, Component, AxisID,
    /// String[], String, int, String[])`.
    #[allow(clippy::too_many_arguments)]
    fn show_option_confirm_pane_base_manager_component_axis_id_string_array_string_int_string_array(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<Vec<String>>,
        title: Option<&str>,
        option_type: i32,
        option_strings: Option<&[String]>,
    ) -> i32 {
        self.show_option_pane_base_manager_component_axis_id_string_array_string_int_int_object_boolean_string_array_boolean(
            manager,
            parent_component,
            axis_id,
            message,
            title,
            option_type,
            QUESTION_MESSAGE,
            None,
            false,
            option_strings,
            None,
        )
    }

    /// Java private `showOptionPane(BaseManager, AxisID, String[], String,
    /// int, int, Object, boolean, String[])`.
    #[allow(clippy::too_many_arguments)]
    fn show_option_pane_base_manager_axis_id_string_array_string_int_int_object_boolean_string_array(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        message: Option<Vec<String>>,
        title: Option<&str>,
        option_type: i32,
        message_type: i32,
        initial_value: Option<&str>,
        override_default_labels: bool,
        option_labels: Option<&[String]>,
    ) -> i32 {
        // `this` (the frame) is the parent component.
        let this_component = self.content_pane.clone();
        let result = self.show_option_dialog(
            manager,
            axis_id,
            Some(&this_component),
            message,
            title,
            option_type,
            message_type,
            initial_value,
            override_default_labels,
            option_labels,
            None,
        );
        result
    }

    /// Java private `showOptionPane(BaseManager, Component, AxisID, String[],
    /// String, int, int, Object, boolean, String[], Boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn show_option_pane_base_manager_component_axis_id_string_array_string_int_int_object_boolean_string_array_boolean(
        &self,
        manager: Option<&'static dyn BaseManager>,
        parent_component: Option<&Rc<JComponent>>,
        axis_id: Option<AxisID>,
        message: Option<Vec<String>>,
        title: Option<&str>,
        option_type: i32,
        message_type: i32,
        initial_value: Option<&str>,
        override_default_labels: bool,
        option_strings: Option<&[String]>,
        modal: Option<bool>,
    ) -> i32 {
        let result = self.show_option_dialog(
            manager,
            axis_id,
            parent_component,
            message,
            title,
            option_type,
            message_type,
            initial_value,
            override_default_labels,
            option_strings,
            modal,
        );
        result
    }

    /// Java private `showOptionDialog(BaseManager, AxisID, Component,
    /// String[], String, int, int, Icon, Object, boolean, String[], Boolean)`.
    /// Shows all pop up message dialogs.  The `Icon` parameter is always null
    /// in the source and is not modelled.
    ///
    /// The modal `JOptionPane` is shown through
    /// [`ui_harness::present_popup`]; its answer is the index of the button
    /// pressed among the buttons the pane displays, or `CLOSED_OPTION`.
    #[allow(clippy::too_many_arguments)]
    fn show_option_dialog(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        parent_component: Option<&Rc<JComponent>>,
        message: Option<Vec<String>>,
        title: Option<&str>,
        option_type: i32,
        mut message_type: i32,
        initial_value: Option<&str>,
        override_defaults: bool,
        options: Option<&[String]>,
        modal: Option<bool>,
    ) -> i32 {
        let _ = modal;
        let Some(message) = message else {
            return CLOSED_OPTION;
        };
        if let Some(manager) = manager {
            manager.log_message_array(Some(message.as_slice()), title, None, axis_id);
        } else {
            eprintln!(
                "{}\n{} - {} axis:",
                utilities::get_date_time_stamp(),
                title.unwrap_or("null"),
                axis_id.map_or_else(|| "null".to_owned(), |axis_id| axis_id.to_string())
            );
            for line in &message {
                eprintln!("{}", line);
            }
        }
        // Change the message icon to match the message.
        // (`icon == null` always holds.)
        if message_type == ERROR_MESSAGE {
            let mut error_message = false;
            let mut warning_message = false;
            if let Some(title) = title {
                let lc_title = title.to_lowercase();
                // Change the icon if the message contains "warning", and does not
                // contain "error".
                if lc_title.contains("error") {
                    error_message = true;
                } else if lc_title.contains("warning") {
                    warning_message = true;
                }
                if !error_message {
                    // Check the first three lines of the message.
                    for line in message.iter().take(3) {
                        let lc_message = line.to_lowercase();
                        if lc_message.contains("error:") {
                            error_message = true;
                            break;
                        }
                        if lc_message.contains("warning:") {
                            warning_message = true;
                        }
                    }
                }
                if !error_message && warning_message {
                    message_type = WARNING_MESSAGE;
                }
            }
        }
        // Decide whether to pass an array of button labels (to override the
        // defaults) or null.  Without an override the pane shows the look and
        // feel's default buttons for the option type.
        let buttons: Vec<String> = if override_defaults && options.is_some() {
            options.unwrap().to_vec()
        } else {
            match option_type {
                YES_NO_OPTION => vec![YES.to_owned(), NO.to_owned()],
                YES_NO_CANCEL_OPTION => vec![YES.to_owned(), NO.to_owned(), CANCEL.to_owned()],
                OK_CANCEL_OPTION => vec![OK.to_owned(), CANCEL.to_owned()],
                _ => vec![OK.to_owned()],
            }
        };
        // Swing: pane.setInitialValue(initialValue); pane.setComponentOrientation(...);
        // JDialog dialog = pane.createDialog(parentComponent, title).
        // A popup with a parent component and no axis is most likely connected
        // to a field.
        if parent_component.is_some() && axis_id.is_none() {
            // Swing layout: dialog location adjusted with
            // PopupTool.adjustLocationY(location.y, parentComponent.getHeight(),
            // dialog.getHeight()).
        }
        // Swing focus: pane.selectInitialValue().
        let name = utilities::convert_label_to_name(title, true);
        // pane.setName(name): carried by the request.
        self.print_name(name.as_deref(), options, title, Some(&message));
        let request = PopupRequest {
            name: name.clone(),
            title: title.map(str::to_owned),
            message: message.clone(),
            options: buttons.clone(),
            option_type,
            message_type,
            initial_value: initial_value.map(str::to_owned),
            axis_id,
            parent_component: parent_component.cloned(),
        };
        // dialog.setVisible(true); dialog.dispose(); Object selectedValue =
        // pane.getValue();
        let answer = ui_harness::present_popup(&request);
        let selected = match answer {
            PopupAnswer::Closed => return CLOSED_OPTION,
            PopupAnswer::Selected(index) => index,
        };
        // If a null array of options was passed, pane returns an integer: the
        // index of the default button.  If an array of options was passed, pane
        // returns the label of the button selected, which is matched against the
        // options; both reduce to the index of the button pressed.
        if selected < buttons.len() {
            if !override_defaults || options.is_none() {
                return selected as i32;
            }
            let options = options.unwrap();
            for (counter, option) in options.iter().enumerate() {
                if *option == buttons[selected] {
                    return counter as i32;
                }
            }
        }
        CLOSED_OPTION
    }

    /// Java private synchronized final `printName(String, String[], String,
    /// String[])`.
    fn print_name(
        &self,
        name: Option<&str>,
        options: Option<&[String]>,
        title: Option<&str>,
        message: Option<&[String]>,
    ) {
        if print_names() {
            // print popup name/value pair
            let mut buffer = format!(
                "{}{}{} {} ",
                UITestFieldType::POPUP,
                SEPARATOR_CHAR,
                name.unwrap_or("null"),
                DEFAULT_DELIMITER
            );
            // if there are options, then print a popup name/value pair
            if let Some(options) = options
                && !options.is_empty()
            {
                let mut appended = false;
                // Starts at 1, as the source does (the first option is not
                // listed).
                for option in options.iter().skip(1) {
                    if appended {
                        buffer.push(',');
                    }
                    buffer.push_str(option);
                    appended = true;
                }
                println!("{}", buffer);
            }
        }
        if self.verbose.get() {
            // if verbose then print the popup title and message
            eprintln!("Popup:");
            eprintln!("{}", title.unwrap_or("null"));
            if let Some(message) = message {
                for line in message {
                    eprintln!("{}", line);
                }
            }
        }
    }
}

impl Default for AbstractFrame {
    fn default() -> Self {
        AbstractFrame::new()
    }
}

impl SwingComponent for AbstractFrame {
    /// Java `getComponent()`: the frame itself; its content pane stands for
    /// it as a component.
    fn get_component(&self) -> Rc<JComponent> {
        self.content_pane.clone()
    }
}

impl UIComponent for AbstractFrame {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.content_pane.clone()
    }
}
