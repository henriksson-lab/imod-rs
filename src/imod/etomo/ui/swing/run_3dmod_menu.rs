//! `IMOD/Etomo/src/etomo/ui/swing/Run3dmodMenu.java`.
#![allow(dead_code)]
use super::context_menu::MouseEvent;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;

pub const DEFAULT_DESCR: &str = "3dmod";
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Run3dmodMenuItem {
    pub text: String,
    pub enabled: bool,
}
/// Java final `Run3dmodMenu`; native popup presentation remains a GUI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Run3dmodMenu {
    pub startup_window: Run3dmodMenuItem,
    pub bin_by_2: Run3dmodMenuItem,
    pub run_3dmod: Option<Run3dmodMenuItem>,
    pub process_button: bool,
    pub file_to_open_known: bool,
    pub popup_events: Vec<MouseEvent>,
    pub menu_order: Vec<String>,
    pub listeners_attached: bool,
}
impl Run3dmodMenu {
    fn new(open_string: &str, process_button: bool) -> Self {
        Self {
            startup_window: Run3dmodMenuItem {
                text: format!("{open_string} with startup window"),
                enabled: true,
            },
            bin_by_2: Run3dmodMenuItem {
                text: format!("{open_string} binned by 2"),
                enabled: true,
            },
            run_3dmod: process_button.then(|| Run3dmodMenuItem {
                text: open_string.into(),
                enabled: true,
            }),
            process_button,
            file_to_open_known: true,
            popup_events: Vec::new(),
            menu_order: Vec::new(),
            listeners_attached: false,
        }
    }
    pub fn get_3dmod_button_instance(description: Option<&str>) -> Self {
        let mut value = Self::new(
            &description
                .map(|v| format!("Open {v}"))
                .unwrap_or_else(|| "Open".into()),
            false,
        );
        value.createMenu();
        value.addListeners();
        value
    }
    pub fn get_process_button_instance(description: Option<&str>) -> Self {
        let mut value = Self::new(
            &format!("And open {}", description.unwrap_or(DEFAULT_DESCR)),
            true,
        );
        value.createMenu();
        value.addListeners();
        value
    }
    #[allow(non_snake_case)]
    /// Java private `createMenu`, represented by native popup item order.
    pub fn createMenu(&mut self) {
        self.menu_order.clear();
        if let Some(item) = &self.run_3dmod {
            self.menu_order.push(item.text.clone());
        }
        self.menu_order.push(self.startup_window.text.clone());
        self.menu_order.push(self.bin_by_2.text.clone());
    }
    #[allow(non_snake_case)]
    /// Java private `addListeners` at the Rust popup-event boundary.
    pub fn addListeners(&mut self) {
        self.listeners_attached = true;
    }
    pub fn set_file_to_open_known(&mut self, file_to_open_known: bool) {
        self.file_to_open_known = file_to_open_known;
        if self.process_button {
            if let Some(item) = &mut self.run_3dmod {
                item.enabled = file_to_open_known;
            }
            self.bin_by_2.enabled = file_to_open_known;
        }
    }
    pub fn pop_up_context_menu(&mut self, enabled: bool, mouse_event: MouseEvent) {
        if enabled && (self.file_to_open_known || self.process_button) {
            self.popup_events.push(mouse_event);
        }
    }
    pub fn action_performed(&self, action_command: Option<&str>) -> Option<Run3dmodMenuOptions> {
        let command = action_command?;
        let mut options = Run3dmodMenuOptions::default();
        if command == self.startup_window.text {
            options.startup_window = true;
        } else if command == self.bin_by_2.text {
            options.bin_by_2 = true;
        } else if self
            .run_3dmod
            .as_ref()
            .is_some_and(|item| command == item.text)
        {
        } else {
            return None;
        }
        Some(options)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn process_menu_disables_file_dependent_choices_and_maps_actions() {
        let mut menu = Run3dmodMenu::get_process_button_instance(None);
        assert_eq!(menu.run_3dmod.as_ref().unwrap().text, "And open 3dmod");
        menu.set_file_to_open_known(false);
        assert!(!menu.bin_by_2.enabled);
        assert!(!menu.run_3dmod.as_ref().unwrap().enabled);
        assert!(
            menu.action_performed(Some(&menu.startup_window.text))
                .unwrap()
                .startup_window
        );
    }
}
