//! `IMOD/Etomo/src/etomo/ui/swing/ManagerFrame.java`.
//!
//! This is the independent frame Java associates with exactly one
//! `BaseManager`.  Its native widget is owned by the optional GUI harness; the
//! source-visible JFrame, root-panel, and menu state live here.
#![allow(dead_code)]

use super::abstract_frame::AbstractFrame;
use super::etomo_frame::{ActionEvent, FrameType};
use super::etomo_menu::{EtomoMenu, MenuTarget};
use super::settings_dialog::SettingsDialog;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::utilities;

/// Java `ManagerFrame.NAME`.
pub const NAME: &str = "manager-frame";

/// Java `private static final class ManagerWindowFocusListener`.
///
/// The listener keeps the manager reference exactly as its Java counterpart;
/// it has no state of its own.
pub struct ManagerWindowFocusListener {
    pub manager: &'static dyn BaseManager,
}

impl ManagerWindowFocusListener {
    /// `ManagerWindowFocusListener(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> Self {
        Self { manager }
    }

    /// `windowGainedFocus(WindowEvent)`.
    pub fn window_gained_focus(&self) {
        self.manager.make_property_user_dir_local();
    }

    /// `windowLostFocus(WindowEvent)`, intentionally empty in Java.
    pub fn window_lost_focus(&self) {}
}

/// Fields of Java's final `ManagerFrame` class, including inherited
/// `AbstractFrame` state.
pub struct ManagerFrame {
    pub abstract_frame: AbstractFrame,
    pub menu: EtomoMenu,
    pub manager: &'static dyn BaseManager,
    /// Swing's `rootPanel`, represented at the presentation boundary by the
    /// source-generated UI-test name and whether its manager panel was added.
    pub root_panel_name: String,
    pub root_panel_has_manager_panel: bool,
    pub savable: bool,
    pub disposed: bool,
    pub focus_listener: ManagerWindowFocusListener,
    /// Java's director-owned SettingsDialog is represented at the frame endpoint
    /// once the GUI backend supplies its available font families.
    pub settings_dialog: Option<SettingsDialog>,
}

impl ManagerFrame {
    /// `ManagerFrame(BaseManager, boolean)`.
    fn new(manager: &'static dyn BaseManager, savable: bool) -> Self {
        Self {
            abstract_frame: AbstractFrame::new(),
            menu: EtomoMenu::get_manager_instance(savable, false),
            manager,
            root_panel_name: String::new(),
            root_panel_has_manager_panel: false,
            savable,
            disposed: false,
            focus_listener: ManagerWindowFocusListener::new(manager),
            settings_dialog: None,
        }
    }

    /// `getInstance(BaseManager, boolean)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        savable: bool,
    ) -> Result<Self, String> {
        let manager = manager.ok_or_else(|| "manager is null".to_string())?;
        let mut instance = Self::new(manager, savable);
        instance.initialize();
        instance.add_listeners();
        Ok(instance)
    }

    /// `initialize()`.
    fn initialize(&mut self) {
        // `DO_NOTHING_ON_CLOSE`: process_window_event owns the close action.
        let name = utilities::convert_label_to_name(Some(NAME), true).unwrap_or_default();
        self.root_panel_name = format!("pnl{SEPARATOR_CHAR}{name}");
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!("pnl{SEPARATOR_CHAR}{name} {DEFAULT_DELIMITER} ");
        }
        // `setIconImage` and `setJMenuBar` are native-widget operations.  The
        // harness reads this frame's presentation/menu state rather than
        // duplicating it.
        self.abstract_frame.presentation.title = self.manager.get_name().unwrap_or_default();
        // `getMainPanel()` has declared Swing type `MainPanel`; that full unit
        // is not translated, so BaseManager faithfully exposes only null.
        self.root_panel_has_manager_panel = self.manager.get_main_panel().is_some();
        self.abstract_frame.repaint(AxisID::Only);
        self.abstract_frame.set_visible(true);
    }

    /// `addListeners()`.
    fn add_listeners(&mut self) {
        // The listener value is installed at construction.  The GUI harness
        // calls `window_gained_focus` when its native window gains focus.
    }

    /// `getFrameType()`: Java deliberately returns null for manager frames.
    pub fn get_frame_type(&self) -> Option<FrameType> {
        None
    }

    /// `menuFileAction(ActionEvent)`.
    pub fn menu_file_action(&mut self, event: &ActionEvent) -> MenuTarget {
        self.menu
            .menu_file_action(&event.action_command)
            .unwrap_or_else(|target| target)
    }

    /// `save(AxisID)`.
    pub fn save(&mut self, _axis_id: AxisID) {
        if self.savable {
            self.manager.save_to_file();
        }
    }

    /// `saveAs()`.
    pub fn save_as(&mut self) {
        if self.savable {
            self.manager.save_as_to_file();
        }
    }

    /// `cancel()`.
    pub fn cancel(&mut self) {
        self.abstract_frame.set_visible(false);
        self.disposed = true;
    }

    /// `close()`.
    pub fn close(&mut self) {
        if self.manager.close_frame() {
            self.abstract_frame.set_visible(false);
            self.disposed = true;
        }
    }

    /// `processWindowEvent(WindowEvent)`.  `closing` is Java's
    /// `event.getID() == WindowEvent.WINDOW_CLOSING`.
    pub fn process_window_event(&mut self, closing: bool) {
        if closing && !ARGUMENTS.lock().unwrap().is_test() {
            self.close();
        }
    }

    /// `menuViewAction(ActionEvent)`.
    pub fn menu_view_action(&mut self, event: &ActionEvent, auto_fit: bool) -> Result<(), String> {
        if event.action_command == self.menu.menu_fit_window.action_command {
            // Java calls UIHarness.INSTANCE.pack(true, manager).  That method
            // routes back to this frame once the ManagerFrame table is present;
            // this direct call is its source-owned frame endpoint.
            self.pack(true, auto_fit);
            Ok(())
        } else {
            Err(format!(
                "Cannot handled menu command in this class.  command={}",
                event.action_command
            ))
        }
    }

    /// `pack(boolean)`.  `auto_fit` is the direct value of the still-unported
    /// `UserConfiguration.isAutoFit()` call.
    pub fn pack(&mut self, force: bool, auto_fit: bool) {
        if !force && !auto_fit {
            self.abstract_frame.set_visible(true);
        } else {
            self.abstract_frame.component.height += 1;
            self.abstract_frame.presentation.packed = true;
        }
    }

    /// `menuOptionsAction(ActionEvent)`.
    pub fn menu_options_action(&mut self, event: &ActionEvent) -> Result<(), String> {
        if event.action_command == self.menu.menu_settings.action_command {
            Err("EtomoDirector.openSettingsDialog requires SettingsDialog.java, which is not yet translated".into())
        } else {
            Err(format!(
                "Cannot handled menu command in this class.  command={}",
                event.action_command
            ))
        }
    }

    /// The concrete endpoint of Java `EtomoDirector.openSettingsDialog()` for
    /// this manager frame. GraphicsEnvironment/CpuAdoc inputs belong to their
    /// native/source boundaries and are supplied by the GUI launcher.
    pub fn open_settings_dialog(&mut self, available_fonts: &[String], cpu_adoc_viable: bool) {
        self.settings_dialog = Some(SettingsDialog::get_instance(
            self.manager,
            self.manager.get_property_user_dir().unwrap_or_default(),
            available_fonts,
            cpu_adoc_viable,
        ));
    }

    /// `menuToolsAction(ActionEvent)`.
    pub fn menu_tools_action(&mut self, event: &ActionEvent) -> MenuTarget {
        self.menu.menu_tools_action(&event.action_command)
    }

    /// `menuHelpAction(ActionEvent)`.
    pub fn menu_help_action(&mut self, event: &ActionEvent) -> MenuTarget {
        self.menu.menu_help_action(&event.action_command)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;

    fn manager() -> &'static dyn BaseManager {
        DirectiveEditorManager::new(None, None, None, None)
    }

    #[test]
    fn independent_frame_keeps_source_manager_menu_shape() {
        let frame = ManagerFrame::get_instance(Some(manager()), true).unwrap();
        assert!(!frame.menu.dataset);
        assert!(frame.menu.savable);
        assert!(frame.abstract_frame.presentation.visible);
        assert_eq!(frame.get_frame_type(), None);
    }

    #[test]
    fn fit_window_packs_and_other_view_commands_are_errors() {
        let mut frame = ManagerFrame::get_instance(Some(manager()), true).unwrap();
        frame
            .menu_view_action(&ActionEvent::new("Fit Window"), false)
            .unwrap();
        assert!(frame.abstract_frame.presentation.packed);
        assert!(
            frame
                .menu_view_action(&ActionEvent::new("Axis A"), false)
                .is_err()
        );
    }

    #[test]
    fn focus_listener_uses_its_manager() {
        let frame = ManagerFrame::get_instance(Some(manager()), false).unwrap();
        frame.focus_listener.window_gained_focus();
    }
}
