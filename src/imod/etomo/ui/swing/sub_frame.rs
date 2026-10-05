//! `IMOD/Etomo/src/etomo/ui/swing/SubFrame.java`.
//!
//! The second eTomo window: shows axis B (the main panel's axis B scroll
//! pane) and its own status bar when both axes are displayed.  It shares the
//! main frame's `MainPanel`.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::abstract_frame::{AbstractFrame, AbstractFrameVirtual, WINDOW_CLOSING};
use super::etomo_frame::{self, EtomoFrame, EtomoFrameVirtual};
use super::main_frame::MainFrame;
use super::main_panel::MainPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusListener;
use crate::imod::etomo::process::process_messages::ProcessMessages;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::frame_type::FrameType;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::util::event_queue::{self, EdtRef};
use crate::imod::etomo::util::utilities;

/// Java package-private static final `NAME`.
pub const NAME: &str = "sub-frame";

/// Java package-private `final class SubFrame extends EtomoFrame implements
/// BusyStatusListener`.
pub struct SubFrame {
    /// The `EtomoFrame` superclass part.
    base: EtomoFrame,
    /// This frame's own handle (Java `this` as a `BusyStatusListener`).
    self_ref: Weak<SubFrame>,
    /// Java private final `pnlStatus = new JPanel()`.
    pnl_status: Rc<JComponent>,
    /// Java private final `pnlBusyStatus = new JPanel()`.
    pnl_busy_status: Rc<JComponent>,
    /// Java private final `lBusyStatusB = MainPanel.createBusyStatusLabel()`.
    l_busy_status_b: Rc<JComponent>,
    /// Java private final `MainFrame mainFrame`.
    main_frame: Weak<MainFrame>,
    /// Java private `JPanel rootPanel`.
    root_panel: RefCell<Option<Rc<JComponent>>>,
    /// Java private `JLabel statusBar`.
    status_bar: RefCell<Option<Rc<JComponent>>>,
    /// `this` as registered with the manager's `BusyStatusMediator`, kept so
    /// the same handle can be removed (Java removes by identity).
    busy_status_listener: RefCell<Option<Arc<EdtRef<dyn BusyStatusListener>>>>,
}

impl Deref for SubFrame {
    type Target = EtomoFrame;
    fn deref(&self) -> &EtomoFrame {
        &self.base
    }
}

impl SubFrame {
    /// Java package-private `SubFrame(MainFrame)`.
    pub fn new(main_frame: Weak<MainFrame>) -> Rc<SubFrame> {
        let this = Rc::new_cyclic(|self_ref| SubFrame {
            base: EtomoFrame::new_void(),
            self_ref: self_ref.clone(),
            pnl_status: JComponent::new_panel(),
            pnl_busy_status: JComponent::new_panel(),
            l_busy_status_b: MainPanel::create_busy_status_label(),
            main_frame,
            root_panel: RefCell::new(None),
            status_bar: RefCell::new(None),
            busy_status_listener: RefCell::new(None),
        });
        let weak: Weak<SubFrame> = Rc::downgrade(&this);
        this.base.set_this(weak as Weak<dyn EtomoFrameVirtual>);
        this.clone().register();
        // Swing layout: pnlBusyStatus.setLayout(new BorderLayout()).
        this.l_busy_status_b.set_enabled(false);
        // Status
        // Swing layout: pnlStatus.setLayout(MainPanel.createStatusBorderLayout()).
        // BorderLayout.EAST
        this.pnl_status.add(&this.pnl_busy_status);
        // BorderLayout.EAST
        this.pnl_busy_status.add(&this.l_busy_status_b);
        this
    }

    /// `this` as a `BusyStatusListener` handle, created once.
    fn busy_status_listener(&self) -> Option<Arc<EdtRef<dyn BusyStatusListener>>> {
        if self.busy_status_listener.borrow().is_none() {
            let this = self.self_ref.upgrade()?;
            *self.busy_status_listener.borrow_mut() =
                Some(Arc::new(EdtRef::new(this as Rc<dyn BusyStatusListener>)));
        }
        self.busy_status_listener.borrow().clone()
    }

    /// Java package-private `initialize(String, BaseManager, String[])`.
    /// One-time initialization.  Sets the mainPanel, creates the menus, and
    /// displays the axis panel.
    pub fn initialize(
        &self,
        title: &str,
        current_manager: Option<&'static dyn BaseManager>,
        m_ru_list: Option<&[String]>,
    ) {
        self.base.initialize();
        if let Some(previous) = self.base.current_manager.get() {
            previous.remove_busy_status_listener(self.busy_status_listener().as_ref());
        }
        self.base.current_manager.set(current_manager);
        // Upstream bug fixed (SubFrame.java:127): Java throws
        // NullPointerException when there is no current manager; the listener
        // is not registered then.
        if let Some(current_manager) = current_manager {
            current_manager.add_busy_status_listener(self.busy_status_listener());
        }
        self.base.set_title(Some(title));
        let main_panel = self
            .main_frame
            .upgrade()
            .and_then(|main_frame| main_frame.get_main_panel());
        *self.base.main_panel.borrow_mut() = main_panel.clone();
        let root_panel = self.base.get_content_pane();
        *self.root_panel.borrow_mut() = Some(root_panel.clone());
        // Swing layout: rootPanel.setLayout(new BorderLayout()).
        // set name
        let field_type = UITestFieldType::PANEL;
        let name = utilities::convert_label_to_name(Some(NAME), field_type.is_unlimited_segments());
        root_panel.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if etomo_director::ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{}{}{} {} ",
                field_type,
                SEPARATOR_CHAR,
                name.as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
        let status_bar = JComponent::new_label(
            &main_panel
                .as_ref()
                .map(|main_panel| main_panel.main_panel().get_status_bar_text())
                .unwrap_or_default(),
        );
        *self.status_bar.borrow_mut() = Some(status_bar.clone());
        // BorderLayout.WEST
        self.pnl_status.add(&status_bar);
        // menu.setEnabled(currentManager);
        let menu = self.base.menu.borrow().clone();
        if let Some(menu) = menu {
            if let Some(other_frame) = self.base.get_other_frame() {
                if let Some(other_menu) = other_frame.etomo_frame().menu.borrow().clone() {
                    menu.set_enabled_etomo_menu(&other_menu);
                }
            }
            // Upstream bug fixed (SubFrame.java:147): EtomoMenu.setMRUFileLabels
            // reads mRUList.length, so a MainFrame whose MRU list was never set
            // makes Java throw NullPointerException; the labels are left alone.
            if let Some(m_ru_list) = m_ru_list {
                menu.set_mru_file_labels(m_ru_list);
            }
            if let Some(main_frame) = self.main_frame.upgrade() {
                if let Some(main_menu) = main_frame.menu.borrow().clone() {
                    menu.set_menu_3dmod_startup_window(main_menu.is_menu_3dmod_startup_window());
                    menu.set_menu_3dmod_bin_by_2(main_menu.is_menu_3dmod_bin_by_2());
                }
            }
        }
        AbstractFrameVirtual::set_visible(self, true);
    }

    /// Java `processWindowEvent(WindowEvent)` (override).  Overridden so we
    /// can exit when window is closed.
    pub fn process_window_event(&self, event_id: i32) {
        // Swing: super.processWindowEvent(event).
        if event_id == WINDOW_CLOSING {
            if let Some(main_frame) = self.main_frame.upgrade() {
                main_frame.show_axis_a();
            }
        }
    }

    /// Java package-private `setMainPanel(String, BaseManager)`.  Set the main
    /// panel when switching managers.
    pub fn set_main_panel(&self, title: &str, current_manager: Option<&'static dyn BaseManager>) {
        if let Some(previous) = self.base.current_manager.get() {
            previous.remove_busy_status_listener(self.busy_status_listener().as_ref());
        }
        self.base.current_manager.set(current_manager);
        // Upstream bug fixed (SubFrame.java:199): NullPointerException without
        // a current manager.
        if let Some(current_manager) = current_manager {
            current_manager.add_busy_status_listener(self.busy_status_listener());
        }
        let main_panel = self
            .main_frame
            .upgrade()
            .and_then(|main_frame| main_frame.get_main_panel());
        *self.base.main_panel.borrow_mut() = main_panel.clone();
        self.base.set_title(Some(title));
        if let Some(status_bar) = self.status_bar.borrow().as_ref() {
            status_bar.set_text(
                &main_panel
                    .as_ref()
                    .map(|main_panel| main_panel.main_panel().get_status_bar_text())
                    .unwrap_or_default(),
            );
        }
    }

    /// Java `menuViewAction(ActionEvent)` (override).  Override superclass to
    /// call mainFrame for command which require switching axis or handling
    /// the log window.
    pub fn menu_view_action(&self, event: &ActionEvent) {
        let menu = self.base.menu.borrow().clone();
        let Some(menu) = menu else {
            return;
        };
        if menu.equals_axis_a(event)
            || menu.equals_axis_b(event)
            || menu.equals_axis_both(event)
            || menu.equals_log_window(event)
        {
            if let Some(main_frame) = self.main_frame.upgrade() {
                main_frame.menu_view_action(event);
            }
        } else {
            self.base.menu_view_action(event);
        }
    }

    /// Java private `setAxis()`.  Refresh or add the axis scroll panel and the
    /// status bar, and set the location.
    fn set_axis(&self) {
        let Some(root_panel) = self.root_panel.borrow().clone() else {
            return;
        };
        root_panel.remove_all();
        let axis = self
            .base
            .main_panel
            .borrow()
            .clone()
            .and_then(|main_panel| main_panel.main_panel().show_both_axis());
        if let Some(axis) = axis {
            // BorderLayout.CENTER
            root_panel.add(&axis);
        }
        // BorderLayout.SOUTH
        root_panel.add(&self.pnl_status);
        self.base.validate();
    }

    /// Java `moveSubFrame()` (override).
    pub fn move_sub_frame(&self) {
        if !self.base.is_visible() {
            return;
        }
        let Some(main_frame) = self.main_frame.upgrade() else {
            return;
        };
        // Java: mainFrameBounds = mainFrame.getBounds(); deviceBounds =
        // mainFrame.getGraphicsConfiguration().getBounds().  The screen's
        // device bounds are not modelled; the origin with no size limit stands
        // for them.
        let main_frame_location = main_frame.get_location();
        let main_frame_size = main_frame.get_size();
        let device_x = 0;
        let device_y = 0;
        let x_location = device_x + main_frame_location.x + main_frame_size.width;
        // `if (xLocation > deviceBounds.x + deviceBounds.width) xLocation =
        // (deviceBounds.x + deviceBounds.width) / 2;` - screen size not modelled.
        self.base
            .set_location(x_location, device_y + main_frame_location.y);
    }
}

/// Java private final inner `SetBusyStatus implements Runnable`.
struct SetBusyStatus {
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `enabled`.
    enabled: bool,
}

impl SetBusyStatus {
    /// Java private `SetBusyStatus(AxisID, boolean)`.
    fn new(axis_id: AxisID, enabled: bool) -> SetBusyStatus {
        SetBusyStatus { axis_id, enabled }
    }

    /// Java `run()`.  The enclosing `SubFrame` is the registered sub frame
    /// (only one may exist).
    fn run(&self) {
        if self.axis_id != AxisID::Second {
            return;
        }
        if let Some(sub_frame) = etomo_frame::sub_frame() {
            sub_frame.l_busy_status_b.set_enabled(self.enabled);
        }
    }
}

impl BusyStatusListener for SubFrame {
    /// Java `msgBusyStatusChanged(AxisID, boolean)`.
    fn msg_busy_status_changed(&self, axis_id: AxisID, busy_status: bool) {
        let set_busy_status = SetBusyStatus::new(axis_id, busy_status);
        event_queue::invoke_later(move || set_busy_status.run());
    }
}

impl AbstractFrameVirtual for SubFrame {
    fn abstract_frame(&self) -> &AbstractFrame {
        &self.base
    }
    fn menu_file_action(&self, action_event: &ActionEvent) {
        self.base.menu_file_action(action_event);
    }
    fn menu_tools_action(&self, action_event: &ActionEvent) {
        self.base.menu_tools_action(action_event);
    }
    fn menu_view_action(&self, action_event: &ActionEvent) {
        SubFrame::menu_view_action(self, action_event);
    }
    fn menu_options_action(&self, action_event: &ActionEvent) {
        self.base.menu_options_action(action_event);
    }
    fn menu_help_action(&self, action_event: &ActionEvent) {
        self.base.menu_help_action(action_event);
    }
    /// Java final `getFrameType()`.
    fn get_frame_type(&self) -> Option<FrameType> {
        Some(FrameType::Sub)
    }
    fn cancel(&self) {
        self.base.cancel();
    }
    fn save(&self, axis_id: Option<AxisID>) {
        self.base.save(axis_id);
    }
    fn save_as(&self) {
        self.base.save_as();
    }
    fn close(&self) {
        self.base.close();
    }
    fn get_axis_id(&self) -> Option<AxisID> {
        self.base.get_axis_id()
    }
    /// Java `setVisible(boolean)` (override).
    fn set_visible(&self, visible: bool) {
        if visible {
            self.set_axis();
        }
        // super.setVisible(visible): the AbstractFrame body.
        let frame = &self.base;
        if visible {
            let director = &*etomo_director::INSTANCE;
            let location = director.with_user_configuration(|user_configuration| {
                if user_configuration.is_last_location_set(FrameType::Sub) {
                    Some((
                        user_configuration.get_last_location_x(FrameType::Sub),
                        user_configuration.get_last_location_y(FrameType::Sub),
                    ))
                } else {
                    None
                }
            });
            if !director.get_arguments().is_ignore_loc()
                && let Some((x, y)) = location
            {
                frame.set_location(x, y);
            }
        }
        frame.set_visible_super(visible);
    }
    fn repaint(&self, axis_id: Option<AxisID>) {
        self.base.repaint(axis_id);
    }
    fn pack_axis_id(&self, axis_id: Option<AxisID>) {
        self.base.pack_axis_id(axis_id);
    }
    fn pack_axis_id_boolean(&self, axis_id: Option<AxisID>, force: bool) {
        self.base.pack_axis_id_boolean(axis_id, force);
    }
    fn pack_void(&self) {
        self.base.pack_void();
    }
    fn menu_file_mru_list_action(&self, event: &ActionEvent) {
        self.base.menu_file_mru_list_action(event);
    }
    fn display_message_base_manager_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.base
            .display_message_base_manager_string_string_axis_id(manager, message, title, axis_id);
    }
    fn display_message_base_manager_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        title: Option<&str>,
    ) {
        self.base
            .display_message_base_manager_string_string(manager, message, title);
    }
    fn display_message_base_manager_string_array_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.base
            .display_message_base_manager_string_array_string_axis_id(
                manager, message, title, axis_id,
            );
    }
    fn display_error_message(
        &self,
        manager: Option<&'static dyn BaseManager>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.base
            .display_error_message(manager, process_messages, title, axis_id);
    }
    fn display_warning_message_base_manager_process_messages_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.base
            .display_warning_message_base_manager_process_messages_string_axis_id(
                manager,
                process_messages,
                title,
                axis_id,
            );
    }
    fn display_yes_no_cancel_message(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> i32 {
        self.base
            .display_yes_no_cancel_message(manager, message, axis_id)
    }
    fn display_yes_no_message_base_manager_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.base
            .display_yes_no_message_base_manager_string_axis_id(manager, message, axis_id)
    }
    fn display_delete_message(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        axis_id: Option<AxisID>,
    ) -> bool {
        self.base.display_delete_message(manager, message, axis_id)
    }
    fn display_yes_no_warning_dialog(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.base
            .display_yes_no_warning_dialog(manager, message, axis_id)
    }
    fn display_yes_no_message_base_manager_string_array_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        axis_id: Option<AxisID>,
    ) -> bool {
        self.base
            .display_yes_no_message_base_manager_string_array_axis_id(manager, message, axis_id)
    }
}

impl EtomoFrameVirtual for SubFrame {
    fn etomo_frame(&self) -> &EtomoFrame {
        &self.base
    }

    /// Java final synchronized `register()`.
    fn register(self: Rc<Self>) {
        if etomo_frame::sub_frame().is_some() {
            // Java throws IllegalStateException; MainFrame.showBothAxis, the
            // only constructor caller, clears a stale registration first.
            panic!("Only one instance of SubFrame is allowed.");
        }
        self.base.main.set(false);
        etomo_frame::set_sub_frame(Some(self));
    }

    /// Java `moveSubFrame()` (override).
    fn move_sub_frame(&self) {
        SubFrame::move_sub_frame(self);
    }
}
