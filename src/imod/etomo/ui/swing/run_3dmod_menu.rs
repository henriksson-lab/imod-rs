//! `IMOD/Etomo/src/etomo/ui/swing/Run3dmodMenu.java`.
//!
//! The right-click menu of a 3dmod button or of a process button that opens 3dmod
//! afterwards: open with the startup window, binned by 2, or (process buttons) plain.
//!
//! The target (the button) owns this menu, so it is held as a `Weak`.  The popup is
//! shown by the Swing stand-in's popup layer (`JComponent::show`).

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::menu_item::MenuItem;
use crate::imod::etomo::jdk::{ActionEvent, JComponent, MouseEvent};
use crate::imod::etomo::r#type::run_3dmod_menu_options::Run3dmodMenuOptions;
use crate::imod::etomo::ui::run_3dmod_menu_target::Run3dmodMenuTarget;

/// Java private static final `DEFAULT_DESCR`.
const DEFAULT_DESCR: &str = "3dmod";

/// Java package-private `final class Run3dmodMenu implements ActionListener`.
pub struct Run3dmodMenu {
    /// This object, for `addActionListener(this)`.
    self_ref: RefCell<Weak<Run3dmodMenu>>,
    /// Java final `contextMenu` (`new JPopupMenu("3dmod Options")`).
    context_menu: Rc<JComponent>,
    /// Java final `target`.
    target: Weak<dyn Run3dmodMenuTarget>,
    /// Java final `startupWindow`.
    startup_window: Rc<MenuItem>,
    /// Java final `binBy2`.
    bin_by_2: Rc<MenuItem>,
    /// Java final `run3dmod` (null unless a process button).
    run_3dmod: Option<Rc<MenuItem>>,
    /// Java final `processButton`.
    process_button: bool,
    /// Java `fileToOpenKnown`.
    file_to_open_known: Cell<bool>,
    /// Java `buttonActionCommand` (only read in commented-out code in the Java).
    #[allow(dead_code)]
    button_action_command: RefCell<Option<String>>,
}

impl Run3dmodMenu {
    /// Java private `Run3dmodMenu(Run3dmodMenuTarget, String, boolean)`.
    fn new(
        target: Weak<dyn Run3dmodMenuTarget>,
        open_string: &str,
        process_button: bool,
    ) -> Rc<Run3dmodMenu> {
        let startup_window = MenuItem::new_string(&format!("{open_string} with startup window"));
        let bin_by_2 = MenuItem::new_string(&format!("{open_string} binned by 2"));
        let run_3dmod = if process_button {
            Some(MenuItem::new_string(open_string))
        } else {
            None
        };
        let instance = Rc::new(Run3dmodMenu {
            self_ref: RefCell::new(Weak::new()),
            context_menu: JComponent::new_popup_menu("3dmod Options"),
            target,
            startup_window,
            bin_by_2,
            run_3dmod,
            process_button,
            file_to_open_known: Cell::new(true),
            button_action_command: RefCell::new(None),
        });
        *instance.self_ref.borrow_mut() = Rc::downgrade(&instance);
        instance
    }

    /// Java static `get3dmodButtonInstance(Run3dmodMenuTarget, String)`.
    pub fn get_3dmod_button_instance(
        button: Rc<dyn Run3dmodMenuTarget>,
        descr: Option<&str>,
    ) -> Rc<Run3dmodMenu> {
        let open_string = match descr {
            None => "Open".to_owned(),
            Some(descr) => format!("Open {descr}"),
        };
        let instance = Run3dmodMenu::new(Rc::downgrade(&button), &open_string, false);
        instance.create_menu();
        instance.add_listeners();
        instance
    }

    /// Java static `getProcessButtonInstance(Run3dmodMenuTarget, String)`.
    pub fn get_process_button_instance(
        button: Rc<dyn Run3dmodMenuTarget>,
        descr: Option<&str>,
    ) -> Rc<Run3dmodMenu> {
        let descr = descr.unwrap_or(DEFAULT_DESCR);
        let instance =
            Run3dmodMenu::new(Rc::downgrade(&button), &format!("And open {descr}"), true);
        instance.create_menu();
        instance.add_listeners();
        instance
    }

    /// Java private `createMenu()`.
    fn create_menu(&self) {
        if let Some(run_3dmod) = &self.run_3dmod {
            self.context_menu.add(&run_3dmod.get_component());
        }
        self.context_menu.add(&self.startup_window.get_component());
        self.context_menu.add(&self.bin_by_2.get_component());
    }

    /// Java `setFileToOpenKnown(boolean)`.  Adjust the menus so that they still make
    /// sense when the file to open is not known.
    pub fn set_file_to_open_known(&self, file_to_open_known: bool) {
        self.file_to_open_known.set(file_to_open_known);
        if self.process_button {
            // When the file to open is not known, there are two choices for the process
            // button: don't run 3dmod, or run it with the startup window.
            if let Some(run_3dmod) = &self.run_3dmod {
                run_3dmod.get_component().set_enabled(file_to_open_known);
            }
            self.bin_by_2
                .get_component()
                .set_enabled(file_to_open_known);
        }
    }

    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let Some(target) = self.target.upgrade() else {
            return;
        };
        // When the file to open is not known, the only choice for a 3dmod button is
        // running with the startup window. So that becomes and default and the menu is
        // unnecessary.
        if !target.is_enabled() || !self.file_to_open_known.get() && !self.process_button {
            return;
        }
        self.context_menu
            .show(&target.get_component(), mouse_event.x, mouse_event.y);
        self.context_menu.set_visible(true);
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let items: Vec<&Rc<MenuItem>> = self
            .run_3dmod
            .iter()
            .chain([&self.startup_window, &self.bin_by_2])
            .collect();
        for item in items {
            // item.addActionListener(this)
            let this = self.self_ref.borrow().clone();
            item.get_component()
                .add_action_listener(Rc::new(move |event| {
                    if let Some(this) = this.upgrade() {
                        this.action_performed(event);
                    }
                }));
        }
    }

    /// Java `@Override actionPerformed(ActionEvent)`.
    ///
    /// Upstream bug fixed in translation (`Run3dmodMenu.java:125`): the Java tests
    /// `actionCommand.equals(run3dmod)`, comparing the command string with the
    /// `JMenuItem` itself, which is never true, so choosing the plain "And open ..."
    /// item never set `noOptions`.  The evident intent, as in the two branches above
    /// it, is the item's text; that is what this compares.
    pub fn action_performed(&self, event: &ActionEvent) {
        let Some(action_command) = event.get_action_command() else {
            return;
        };
        // MenuOptions holds the current menu choice.
        let mut menu_options = Run3dmodMenuOptions::new();
        if action_command == self.startup_window.get_component().get_text() {
            menu_options.set_startup_window(true);
        } else if action_command == self.bin_by_2.get_component().get_text() {
            menu_options.set_bin_by_2(true);
        } else if self
            .run_3dmod
            .as_ref()
            .is_some_and(|run_3dmod| action_command == run_3dmod.get_component().get_text())
        {
            menu_options.set_no_options(true);
        }
        // (A commented-out branch in the Java set the startup window for a 3dmod
        // button whose file to open is unknown when the command is
        // buttonActionCommand.)
        if let Some(target) = self.target.upgrade() {
            target.menu_action(menu_options);
        }
    }
}
