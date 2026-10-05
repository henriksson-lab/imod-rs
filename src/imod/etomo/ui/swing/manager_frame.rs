//! `IMOD/Etomo/src/etomo/ui/swing/ManagerFrame.java`.
//!
//! An independent frame associated with a single manager (tools, directive
//! editor): its content pane holds the manager's main panel.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::abstract_frame::{
    AbstractFrame, AbstractFrameVirtual, WINDOW_CLOSING, WindowFocusListener,
};
use super::etomo_menu::EtomoMenu;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::frame_type::FrameType;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::util::utilities;

/// Java public static final `NAME`.
pub const NAME: &str = "manager-frame";

/// Java `public final class ManagerFrame extends AbstractFrame`.
pub struct ManagerFrame {
    /// The `AbstractFrame` superclass part.
    base: AbstractFrame,
    /// Java private final `EtomoMenu menu`.
    menu: Rc<EtomoMenu>,
    /// Java private final `BaseManager manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `JPanel rootPanel = (JPanel) getContentPane()`.
    root_panel: Rc<JComponent>,
    /// Java private final `boolean savable`.
    savable: bool,
}

impl Deref for ManagerFrame {
    type Target = AbstractFrame;
    fn deref(&self) -> &AbstractFrame {
        &self.base
    }
}

impl ManagerFrame {
    /// Java private `ManagerFrame(BaseManager, boolean)`.
    fn new(manager: &'static dyn BaseManager, savable: bool) -> Rc<ManagerFrame> {
        Rc::new_cyclic(|self_ref: &Weak<ManagerFrame>| {
            let base = AbstractFrame::new();
            base.set_this(self_ref.clone() as Weak<dyn AbstractFrameVirtual>);
            let root_panel = base.get_content_pane();
            let menu = EtomoMenu::get_instance_manager_frame_boolean(
                self_ref.clone() as Weak<dyn AbstractFrameVirtual>,
                savable,
            );
            ManagerFrame {
                base,
                menu,
                manager,
                root_panel,
                savable,
            }
        })
    }

    /// Java package-private static `getInstance(BaseManager, boolean)`.
    ///
    /// Java throws `NullPointerException("manager is null")` in `initialize`
    /// for a null manager; the manager is not optional here.
    pub fn get_instance(manager: &'static dyn BaseManager, savable: bool) -> Rc<ManagerFrame> {
        let instance = ManagerFrame::new(manager, savable);
        instance.initialize();
        instance.add_listeners();
        instance
    }

    /// Java private `initialize()`.
    fn initialize(&self) {
        // Swing: setDefaultCloseOperation(DO_NOTHING_ON_CLOSE).
        // (`manager == null` cannot happen: see get_instance.)
        // set name
        let field_type = UITestFieldType::PANEL;
        let name = utilities::convert_label_to_name(Some(NAME), field_type.is_unlimited_segments());
        self.root_panel.set_name(Some(&format!(
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
        // Swing: ImageIcon iconEtomo = new ImageIcon(ClassLoader.getSystemResource(
        // "images/etomo.png")); setIconImage(iconEtomo.getImage()).
        self.base.set_j_menu_bar(Some(self.menu.get_menu_bar()));
        self.base.set_title(self.manager.get_name().as_deref());
        // Upstream bug fixed (ManagerFrame.java:78): a manager without a main
        // panel makes Java add null (NullPointerException); nothing is added.
        if let Some(main_panel) = self.manager.get_main_panel() {
            self.root_panel
                .add(&main_panel.main_panel().get_component());
        }
        // Swing painting: rootPanel.repaint().
        AbstractFrameVirtual::set_visible(self, true);
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.base
            .add_window_focus_listener(Rc::new(ManagerWindowFocusListener::new(self.manager)));
    }

    /// Java `processWindowEvent(WindowEvent)` (override).
    pub fn process_window_event(&self, event_id: i32) {
        // Swing: super.processWindowEvent(event).
        let is_test = etomo_director::ARGUMENTS.lock().unwrap().is_test();
        if event_id == WINDOW_CLOSING && !is_test {
            AbstractFrameVirtual::close(self);
        }
    }

    /// Java `pack(boolean)` (override).
    pub fn pack(&self, force: bool) {
        let auto_fit = etomo_director::INSTANCE.with_user_configuration(|c| c.is_auto_fit());
        if !force && !auto_fit {
            AbstractFrameVirtual::set_visible(self, true);
        } else {
            let mut bounds = self.base.get_size();
            bounds.height += 1;
            bounds.width += 1;
            self.base.set_size(bounds);
            // `try { super.pack(); } catch (NullPointerException e) {
            // e.printStackTrace(); }` - Swing layout: Window.pack().
        }
    }
}

impl AbstractFrameVirtual for ManagerFrame {
    fn abstract_frame(&self) -> &AbstractFrame {
        &self.base
    }

    /// Java `getFrameType()`.
    fn get_frame_type(&self) -> Option<FrameType> {
        None
    }

    /// Java `menuFileAction(ActionEvent)`.
    fn menu_file_action(&self, event: &ActionEvent) {
        self.menu.menu_file_action(event);
    }

    /// Java `save(AxisID)`.
    fn save(&self, axis_id: Option<AxisID>) {
        let _ = axis_id;
        if self.savable {
            self.manager.save_to_file();
        }
    }

    /// Java `saveAs()`.
    fn save_as(&self) {
        if self.savable {
            self.manager.save_as_to_file();
        }
    }

    /// Java `cancel()`.
    fn cancel(&self) {
        AbstractFrameVirtual::set_visible(self, false);
        self.base.dispose();
    }

    /// Java `close()`.
    fn close(&self) {
        if self.manager.close_frame() {
            AbstractFrameVirtual::set_visible(self, false);
            self.base.dispose();
        }
    }

    /// Java `menuViewAction(ActionEvent)`.  Handle some of the view menu
    /// events.
    fn menu_view_action(&self, event: &ActionEvent) {
        // Run fitWindow on both frames.
        if self.menu.equals_fit_window(event) {
            let manager = self.manager;
            ui_harness::with(|harness| harness.pack_boolean_base_manager(true, Some(manager)));
        } else {
            // IllegalStateException on the EDT: printed, event dropped.
            eprintln!(
                "Exception in thread \"AWT-EventQueue-0\" java.lang.IllegalStateException: Cannot handled menu command in this class.  command={}",
                event.get_action_command().unwrap_or("null")
            );
        }
    }

    /// Java `pack(boolean)` (override).
    fn pack_boolean(&self, force: bool) {
        ManagerFrame::pack(self, force);
    }

    /// Java `menuOptionsAction(ActionEvent)`.  Handle some of the options menu
    /// events.
    fn menu_options_action(&self, event: &ActionEvent) {
        if self.menu.equals_settings(event) {
            let _ = etomo_director::INSTANCE.open_settings_dialog();
        } else {
            // IllegalStateException on the EDT: printed, event dropped.
            eprintln!(
                "Exception in thread \"AWT-EventQueue-0\" java.lang.IllegalStateException: Cannot handled menu command in this class.  command={}",
                event.get_action_command().unwrap_or("null")
            );
        }
    }

    /// Java `menuToolsAction(ActionEvent)`.
    fn menu_tools_action(&self, event: &ActionEvent) {
        self.menu.menu_tools_action(AxisID::Only, event);
    }

    /// Java `menuHelpAction(ActionEvent)`.  Handle help menu actions.
    fn menu_help_action(&self, event: &ActionEvent) {
        let frame = self.base.get_content_pane();
        self.menu
            .menu_help_action(Some(self.manager), AxisID::Only, &frame, event);
    }
}

/// Java private static final `ManagerWindowFocusListener implements
/// WindowFocusListener`.
struct ManagerWindowFocusListener {
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
}

impl ManagerWindowFocusListener {
    /// Java private `ManagerWindowFocusListener(BaseManager)`.
    fn new(manager: &'static dyn BaseManager) -> ManagerWindowFocusListener {
        ManagerWindowFocusListener { manager }
    }
}

impl WindowFocusListener for ManagerWindowFocusListener {
    /// Java `windowGainedFocus(WindowEvent)`.
    fn window_gained_focus(&self) {
        self.manager.make_property_user_dir_local();
    }

    /// Java `windowLostFocus(WindowEvent)`: empty.
    fn window_lost_focus(&self) {}
}
