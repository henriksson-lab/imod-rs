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
    /// Java field `initialized`, initialised to false.
    initialized: bool,
    /// Java field `headless`, initialised by `initialize()`.
    headless: bool,
    /// Java field `mainFrame`, initialised to null.
    main_frame: Option<MainFrameWindow>,
    /// Events supplied by `MainFrame` callbacks, in UI event-loop order.
    menu_events: Rc<RefCell<Vec<MenuEvent>>>,
    /// Java `MainFrame` owns one `EtomoMenu`; this is the authoritative
    /// command/checkbox/MRU state behind the Slint presentation.
    menu: Rc<RefCell<EtomoMenu>>,
}

impl Default for UiHarness {
    fn default() -> Self {
        Self {
            initialized: false,
            headless: false,
            main_frame: None,
            menu_events: Rc::new(RefCell::new(Vec::new())),
            menu: Rc::new(RefCell::new(EtomoMenu::get_instance(false))),
        }
    }
}

impl UiHarness {
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
        self.main_frame.is_some()
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
}

#[cfg(test)]
mod tests {
    use super::{MenuEvent, UiHarness};

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
}
