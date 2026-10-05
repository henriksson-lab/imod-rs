//! `IMOD/Etomo/src/etomo/ui/swing/MainFrame.java`.
//!
//! The main eTomo window.  Its content pane (`rootPanel`) holds the current
//! manager's main panel, or the `WindowSwitch` tabbed pane when several
//! managers are open; [`MainFrame::get_root_component`] exposes it so a test
//! driver or the Slint bridge can find components by name
//! (`jdk::find_component`).  Axis B is shown in the `SubFrame` when both axes
//! are displayed.

use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::abstract_frame::{AbstractFrame, AbstractFrameVirtual, WINDOW_CLOSING};
use super::context_menu::ContextMenu;
use super::context_popup::ContextPopup;
use super::etomo_frame::{self, EtomoFrame, EtomoFrameVirtual};
use super::fixed_dim;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::main_panel::MainPanelVirtual;
use super::sub_frame::SubFrame;
use super::ui_harness;
use super::window_switch::WindowSwitch;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, Dimension, JComponent, MouseEvent};
use crate::imod::etomo::process::process_messages::ProcessMessages;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::frame_type::FrameType;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::util::unique_key::UniqueKey;
use crate::imod::etomo::util::utilities;

/// Java public static final `rcsid`.
pub const RCSID: &str = "$Id$";
/// Java package-private static final `EXTRA_SCREEN_WIDTH_MULTIPLIER`.
pub const EXTRA_SCREEN_WIDTH_MULTIPLIER: i32 = 2;
/// Java package-private static final `FRAME_BORDER = FixedDim.frameBorder`.
pub const FRAME_BORDER: Dimension = fixed_dim::frameBorder;
/// Java public static final `ETOMO_TITLE`.
pub const ETOMO_TITLE: &str = "Etomo";
/// Java public static final `NAME`.
pub const NAME: &str = "main-frame";
/// Java private static final `aAxisTitle`.
const A_AXIS_TITLE: &str = "A Axis - ";
/// Java private static final `bAxisTitle`.
const B_AXIS_TITLE: &str = "B Axis - ";

/// Java `public final class MainFrame extends EtomoFrame implements
/// ContextMenu`.
pub struct MainFrame {
    /// The `EtomoFrame` superclass part.
    base: EtomoFrame,
    /// This frame's own handle (Java `this`, passed to `new SubFrame(this)`).
    self_ref: Weak<MainFrame>,
    /// Java private final `rootPanel = (JPanel) getContentPane()`.
    root_panel: Rc<JComponent>,
    /// Java package-private `GenericMouseAdapter mouseAdapter = null`.
    pub mouse_adapter: RefCell<Option<Rc<GenericMouseAdapter>>>,
    /// Java package-private `WindowSwitch windowSwitch = new WindowSwitch()`.
    pub window_switch: Rc<WindowSwitch>,
    /// Java private `String title`.
    title: RefCell<Option<String>>,
    /// Java private `String[] mRUList`.
    m_ru_list: RefCell<Option<Vec<String>>>,
    /// Java private `boolean registered = false`.
    registered: Cell<bool>,
}

impl Deref for MainFrame {
    type Target = EtomoFrame;
    fn deref(&self) -> &EtomoFrame {
        &self.base
    }
}

impl MainFrame {
    /// Java package-private `MainFrame()`.  Main window constructor.  This
    /// sets up the menus and status line.
    pub fn new() -> Rc<MainFrame> {
        let base = EtomoFrame::new_void();
        let root_panel = base.get_content_pane();
        let this = Rc::new_cyclic(|self_ref| MainFrame {
            base,
            self_ref: self_ref.clone(),
            root_panel,
            mouse_adapter: RefCell::new(None),
            window_switch: WindowSwitch::new(),
            title: RefCell::new(None),
            m_ru_list: RefCell::new(None),
            registered: Cell::new(false),
        });
        let weak: Weak<MainFrame> = Rc::downgrade(&this);
        this.base.set_this(weak as Weak<dyn EtomoFrameVirtual>);
        this.clone().register();
        // AWT events: enableEvents(AWTEvent.WINDOW_EVENT_MASK).
        // Swing layout: screenSize = UIUtilities.getScreenSize(); screenSize.width *=
        // EXTRA_SCREEN_WIDTH_MULTIPLIER; rootPanelSize = screenSize less
        // FRAME_BORDER.
        // Swing layout: rootPanel.setLayout(new BorderLayout());
        // rootPanel.setMaximumSize(rootPanelSize).
        // set name
        let field_type = UITestFieldType::PANEL;
        let name = utilities::convert_label_to_name(Some(NAME), field_type.is_unlimited_segments());
        this.root_panel.set_name(Some(&format!(
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

        // rootPanel.setLayout(new BoxLayout(rootPanel, BoxLayout.PAGE_AXIS));

        // add the context menu to all of the main window objects
        // Swing: setDefaultCloseOperation(DO_NOTHING_ON_CLOSE).
        this.base.initialize();
        this
    }

    /// The frame's content tree root (Java `rootPanel`), for finding
    /// components by name.  Rust-only accessor.
    pub fn get_root_component(&self) -> Rc<JComponent> {
        self.root_panel.clone()
    }

    /// The frame's menu bar (Java `getJMenuBar()`), for finding menu items by
    /// text.  Rust-only accessor.
    pub fn get_menu_bar_component(&self) -> Option<Rc<JComponent>> {
        self.base.menu_bar.borrow().clone()
    }

    /// Java package-private `updateFrame(BaseManager)`.
    pub fn update_frame(&self, current_manager: Option<&'static dyn BaseManager>) {
        self.base.set_enabled_base_manager(current_manager);
    }

    /// Java package-private `setCurrentManager(BaseManager, UniqueKey,
    /// boolean, boolean)`.
    pub fn set_current_manager_base_manager_unique_key_boolean_boolean(
        &self,
        current_manager: Option<&'static dyn BaseManager>,
        manager_key: Option<&UniqueKey>,
        new_window: bool,
        manager_stamp: bool,
    ) {
        // Hide log window from previous manager
        if let Some(previous) = self.base.current_manager.get() {
            previous.msg_current_manager_changed(false);
        }
        self.base.set_enabled_base_manager(current_manager);
        self.base.current_manager.set(current_manager);
        match current_manager {
            None => {
                let _ = etomo_director::INSTANCE.make_original_dir_local();
            }
            Some(current_manager) => {
                current_manager.msg_current_manager_changed(true);
                if manager_stamp {
                    utilities::manager_stamp_new_window(
                        current_manager.get_property_user_dir().as_deref(),
                        current_manager.get_name().as_deref(),
                        new_window,
                    );
                }
            }
        }
        // Remove everything from rootPanel if the main panel has been set from
        // the previous manager.
        if self.base.main_panel.borrow().is_some() {
            self.root_panel.remove_all();
        }
        match current_manager {
            None => {
                *self.title.borrow_mut() = Some(ETOMO_TITLE.to_owned());
                self.hide_axis_b();
            }
            Some(current_manager) => {
                *self.base.main_panel.borrow_mut() = current_manager.get_main_panel();
                *self.title.borrow_mut() = Some(format!(
                    "{} - {}",
                    current_manager
                        .get_name()
                        .unwrap_or_else(|| "null".to_owned()),
                    ETOMO_TITLE
                ));
                // Upstream bug fixed (MainFrame.java:370): Java adds a null
                // panel (NullPointerException) when the window switch has no
                // entry for the key; nothing is added instead.
                if let Some(panel) = self.window_switch.get_panel(manager_key) {
                    self.root_panel.add(&panel);
                }
                self.base.to_front();
                let main_panel = self.base.main_panel.borrow().clone();
                // Upstream bug fixed (MainFrame.java:372): a manager without a
                // main panel makes Java throw NullPointerException here; the
                // panel calls are skipped instead.
                if let Some(main_panel) = &main_panel {
                    // mainPanel.addMouseListener(mouseAdapter): mouseAdapter is
                    // always null, which Swing ignores.
                    main_panel.main_panel().repaint();
                }

                if let Some(sub_frame) = etomo_frame::sub_frame() {
                    let title = self.title.borrow().clone().unwrap_or_default();
                    sub_frame.set_main_panel(
                        &format!("{}{} ", B_AXIS_TITLE, title),
                        Some(current_manager),
                    );
                }
                if new_window {
                    self.show_axis_a();
                } else if let Some(main_panel) = main_panel {
                    if main_panel.main_panel().is_showing_both_axis() {
                        self.show_both_axis();
                    } else if main_panel.main_panel().is_showing_axis_a() {
                        self.show_axis_a();
                    } else {
                        self.show_axis_b();
                    }
                }
            }
        }
    }

    /// Java package-private `showHideLog()`.
    pub fn show_hide_log(&self) {
        if let Some(current_manager) = self.base.current_manager.get() {
            current_manager.show_hide_log();
        }
    }

    /// Java package-private `getMainPanel()`.
    pub fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.base.main_panel.borrow().clone()
    }

    /// Java package-private `setCurrentManager(BaseManager, UniqueKey)`.
    pub fn set_current_manager_base_manager_unique_key(
        &self,
        current_manager: Option<&'static dyn BaseManager>,
        manager_key: Option<&UniqueKey>,
    ) {
        self.set_current_manager_base_manager_unique_key_boolean_boolean(
            current_manager,
            manager_key,
            false,
            true,
        );
    }

    /// Java `setMRUFileLabels(String[])` (override).
    pub fn set_mru_file_labels(&self, m_ru_list: &[String]) {
        *self.m_ru_list.borrow_mut() = Some(m_ru_list.to_vec());
        self.base.set_mru_file_labels(m_ru_list);
    }

    /// Java package-private `addWindow(BaseManager, AxisID, UniqueKey)`.
    pub fn add_window(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        manager_key: Option<&UniqueKey>,
    ) {
        self.window_switch.add(manager, axis_id, manager_key);
    }

    /// Java package-private `removeWindow(UniqueKey)`.
    pub fn remove_window(&self, manager_key: Option<&UniqueKey>) {
        self.window_switch.remove(manager_key);
    }

    /// Java package-private `renameWindow(UniqueKey, UniqueKey)`.
    pub fn rename_window(
        &self,
        old_manager_key: Option<&UniqueKey>,
        new_manager_key: Option<&UniqueKey>,
    ) {
        self.window_switch.rename(old_manager_key, new_manager_key);
    }

    /// Java package-private `selectWindowMenuItem(UniqueKey)`.
    pub fn select_window_menu_item_unique_key(&self, current_manager_key: Option<&UniqueKey>) {
        self.select_window_menu_item_unique_key_boolean(current_manager_key, false);
    }

    /// Java package-private `selectWindowMenuItem(UniqueKey, boolean)`.
    pub fn select_window_menu_item_unique_key_boolean(
        &self,
        current_manager_key: Option<&UniqueKey>,
        new_window: bool,
    ) {
        self.window_switch
            .select_window(current_manager_key, new_window);
    }

    /// Java `menuViewAction(ActionEvent)` (override).  Handle the view menu
    /// events.
    pub fn menu_view_action(&self, event: &ActionEvent) {
        let menu = self.base.menu.borrow().clone();
        let Some(menu) = menu else {
            return;
        };
        if menu.equals_axis_a(event) {
            self.show_axis_a();
        } else if menu.equals_axis_b(event) {
            self.show_axis_b();
        } else if menu.equals_axis_both(event) {
            self.show_both_axis();
        } else if menu.equals_log_window(event) {
            self.show_hide_log();
        } else {
            self.base.menu_view_action(event);
        }
    }

    /// Java private `setTitle(AxisID)`.
    fn set_title_axis_id(&self, axis_id: Option<AxisID>) {
        let title = self.title.borrow().clone();
        let title_text = title.as_deref().unwrap_or("null");
        let dual = self
            .base
            .main_panel
            .borrow()
            .as_ref()
            .is_some_and(|main_panel| {
                main_panel.main_panel().get_axis_type() == AxisType::DualAxis
            });
        if dual {
            if axis_id == Some(AxisID::First) {
                self.base
                    .set_title(Some(&format!("{}{}", A_AXIS_TITLE, title_text)));
            } else if axis_id == Some(AxisID::Second) {
                self.base
                    .set_title(Some(&format!("{}{}", B_AXIS_TITLE, title_text)));
            } else {
                self.base.set_title(title.as_deref());
            }
        } else {
            self.base.set_title(title.as_deref());
        }
    }

    /// Java package-private `hideAxisB()`.
    pub fn hide_axis_b(&self) {
        let title = self.title.borrow().clone();
        self.base.set_title(title.as_deref());
        if let Some(sub_frame) = etomo_frame::sub_frame() {
            sub_frame.set_visible(false);
        }
        self.base.pack_void();
    }

    /// Java package-private `showAxisA()`.
    pub fn show_axis_a(&self) {
        self.set_title_axis_id(Some(AxisID::First));
        if let Some(sub_frame) = etomo_frame::sub_frame() {
            sub_frame.set_visible(false);
        }
        // Upstream bug fixed (MainFrame.java:514): with no main panel (no
        // current manager) Java throws NullPointerException here.
        if let Some(main_panel) = self.get_main_panel() {
            main_panel.main_panel().show_axis_a();
        }
        let current_manager = self.base.current_manager.get();
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(AxisID::First), current_manager)
        });
    }

    /// Java package-private `showAxisB()`.
    pub fn show_axis_b(&self) {
        self.set_title_axis_id(Some(AxisID::Second));
        if let Some(sub_frame) = etomo_frame::sub_frame() {
            sub_frame.set_visible(false);
        }
        // Upstream bug fixed (MainFrame.java:523): NullPointerException without
        // a main panel.
        if let Some(main_panel) = self.get_main_panel() {
            main_panel.main_panel().show_axis_b();
        }
        let current_manager = self.base.current_manager.get();
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(AxisID::Second), current_manager)
        });
    }

    /// Java package-private `showBothAxis()`.
    pub fn show_both_axis(&self) {
        let title = self
            .title
            .borrow()
            .clone()
            .unwrap_or_else(|| "null".to_owned());
        self.base
            .set_title(Some(&format!("{}{}", A_AXIS_TITLE, title)));
        // Upstream bug fixed (MainFrame.java:529): NullPointerException without
        // a main panel.
        let Some(main_panel) = self.get_main_panel() else {
            return;
        };
        main_panel.main_panel().show_axis_a();
        let existing = etomo_frame::sub_frame();
        let sub_frame = match existing {
            Some(sub_frame) if sub_frame.is_displayable() => {
                sub_frame.set_visible(true);
                sub_frame
            }
            existing => {
                // Upstream bug fixed (MainFrame.java:531-532): replacing a
                // disposed sub frame calls SubFrame.register(), which throws
                // IllegalStateException while the old one is still registered;
                // the old registration is cleared first.
                if existing.is_some() {
                    etomo_frame::set_sub_frame(None);
                }
                let sub_frame = SubFrame::new(self.self_ref.clone());
                let m_ru_list = self.m_ru_list.borrow().clone();
                sub_frame.initialize(
                    &format!("{}{} ", B_AXIS_TITLE, title),
                    self.base.current_manager.get(),
                    m_ru_list.as_deref(),
                );
                sub_frame
            }
        };
        let current_manager = self.base.current_manager.get();
        ui_harness::with(|harness| harness.pack_base_manager(current_manager));
        sub_frame.pack_void();
    }

    /// Java `processWindowEvent(WindowEvent)` (override).  Overridden so we
    /// can exit when window is closed.
    pub fn process_window_event(&self, event_id: i32) {
        // Swing: super.processWindowEvent(event).
        let is_test = etomo_director::ARGUMENTS.lock().unwrap().is_test();
        if event_id == WINDOW_CLOSING && !is_test {
            if let Some(menu) = self.base.menu.borrow().clone() {
                menu.do_click_file_exit();
            }
        }
    }
}

impl ContextMenu for MainFrame {
    /// Java `popUpContextMenu(MouseEvent)`.  Right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let main_panel = self.get_main_panel();
        // Upstream bug fixed (MainFrame.java:420): ContextPopup dereferences
        // the main panel and the manager; without a current manager Java
        // throws NullPointerException.  No popup is shown then.
        let (Some(main_panel), Some(current_manager)) =
            (main_panel, self.base.current_manager.get())
        else {
            return;
        };
        let _context_popup = ContextPopup::new_component_mouse_event_string_base_manager_axis_id(
            &main_panel.main_panel().get_component(),
            mouse_event,
            Some(""),
            current_manager,
            self.base.get_axis_id().unwrap_or(AxisID::Only),
        );
    }
}

impl AbstractFrameVirtual for MainFrame {
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
        MainFrame::menu_view_action(self, action_event);
    }
    fn menu_options_action(&self, action_event: &ActionEvent) {
        self.base.menu_options_action(action_event);
    }
    fn menu_help_action(&self, action_event: &ActionEvent) {
        self.base.menu_help_action(action_event);
    }
    /// Java final `getFrameType()`.
    fn get_frame_type(&self) -> Option<FrameType> {
        Some(FrameType::Main)
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

impl EtomoFrameVirtual for MainFrame {
    fn etomo_frame(&self) -> &EtomoFrame {
        &self.base
    }

    /// Java final synchronized `register()`.
    fn register(self: Rc<Self>) {
        if self.registered.get() {
            // Java throws IllegalStateException; `registered` is per instance
            // and set only here, so this cannot happen.
            panic!("Only one instance of MainFrame is allowed.");
        }
        self.registered.set(true);
        self.base.main.set(true);
        etomo_frame::set_main_frame(Some(self));
    }
}
