//! `IMOD/Etomo/src/etomo/ui/swing/Run3dmodButton.java`.
#![allow(dead_code)]
use super::{
    context_menu::{ContextMenu, MouseEvent},
    deferred_3dmod_button::Deferred3dmodButton,
    multi_line_button::MultiLineButton,
    run_3dmod_menu::Run3dmodMenu,
};
use crate::imod::etomo::{
    process::imod_process::Run3dmodMenuOptions,
    r#type::{dialog_type::DialogType, file_key::FileKey},
};

/// Java final `Run3dmodButton`.  Container invocation is left explicit because Java
/// stores an interface reference while Rust cannot retain an arbitrary mutable borrow.
#[derive(Clone, Debug, PartialEq)]
pub struct Run3dmodButton {
    pub multi_line_button: MultiLineButton,
    pub deferred: bool,
    pub run_3dmod_menu: Run3dmodMenu,
    pub container_attached: bool,
    pub deferred_3dmod_button_attached: bool,
    pub process_output_file_known: bool,
    pub last_action: Option<Run3dmodMenuOptions>,
}
impl Run3dmodButton {
    fn new(
        label: &str,
        container_attached: bool,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
        deferred: bool,
        description: Option<&str>,
        output_image_file_key: Option<FileKey>,
    ) -> Self {
        let mut multi_line_button = MultiLineButton::new_full(
            Some(label),
            toggle_button,
            dialog_type,
            false,
            false,
            false,
            output_image_file_key,
        );
        multi_line_button.set_action_command(Some(label));
        multi_line_button.add_mouse_listener();
        Self {
            multi_line_button,
            deferred,
            run_3dmod_menu: if deferred {
                Run3dmodMenu::get_process_button_instance(description)
            } else {
                Run3dmodMenu::get_3dmod_button_instance(description)
            },
            container_attached,
            deferred_3dmod_button_attached: false,
            process_output_file_known: true,
            last_action: None,
        }
    }
    pub fn get_3dmod_instance(label: &str, container_attached: bool) -> Self {
        Self::new(label, container_attached, false, None, false, None, None)
    }
    pub fn get_3dmod_instance_with_output_image_file_key(
        label: &str,
        container_attached: bool,
        output_image_file_key: Option<FileKey>,
    ) -> Self {
        Self::new(
            label,
            container_attached,
            false,
            None,
            false,
            None,
            output_image_file_key,
        )
    }
    pub fn get_toggle_3dmod_instance(label: &str, dialog_type: Option<DialogType>) -> Self {
        Self::new(label, false, true, dialog_type, false, None, None)
    }
    pub fn get_deferred_3dmod_instance(label: &str, container_attached: bool) -> Self {
        Self::new(label, container_attached, false, None, true, None, None)
    }
    pub fn get_deferred_3dmod_instance_with_description(
        label: &str,
        container_attached: bool,
        description: Option<&str>,
    ) -> Self {
        Self::new(
            label,
            container_attached,
            false,
            None,
            true,
            description,
            None,
        )
    }
    pub fn get_deferred_toggle_3dmod_instance(
        label: &str,
        dialog_type: Option<DialogType>,
    ) -> Self {
        Self::new(label, false, true, dialog_type, true, None, None)
    }
    pub fn set_deferred_3dmod_button(
        &mut self,
        input_present: bool,
        output_image_file_key: Option<FileKey>,
    ) {
        if !input_present && self.deferred {
            panic!("A deferred instance needs to have a deferred3dmodButton.");
        }
        self.deferred_3dmod_button_attached = input_present;
        if input_present {
            self.multi_line_button
                .set_output_image_file_key(output_image_file_key);
        }
    }
    pub fn set_file_to_open_known(&mut self, known: bool) {
        self.process_output_file_known = known;
        self.run_3dmod_menu.set_file_to_open_known(known);
    }
    pub fn set_container(&mut self, attached: bool) {
        self.container_attached = attached;
    }
    pub fn get_component(&self) -> &MultiLineButton {
        &self.multi_line_button
    }
    pub fn set_tool_tip_text(&mut self, text: &str) {
        self.multi_line_button.set_tool_tip_text(Some(text));
    }
    pub fn add_action_listener(&mut self) {
        self.multi_line_button.add_action_listener();
    }
    /// Java private `addListeners`: the native button retains its generic
    /// mouse adapter as a concrete listener count.
    #[allow(non_snake_case)]
    pub fn addListeners(&mut self) {
        if self.multi_line_button.button.mouse_listener_count == 0 {
            self.multi_line_button.add_mouse_listener();
        }
    }
    /// Java `getDeferred3dmodButton` availability boundary.  The borrowed
    /// deferred endpoint is represented by the owning Rust action path.
    #[allow(non_snake_case)]
    pub fn getDeferred3dmodButton(&self) -> bool {
        self.deferred_3dmod_button_attached
    }
    pub fn get_action_command(&self) -> Option<&str> {
        self.multi_line_button.get_action_command()
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.multi_line_button.set_enabled(enabled);
    }
    pub fn menu_action(&mut self, options: Run3dmodMenuOptions) {
        self.action(options);
        if self.multi_line_button.toggle_button {
            self.multi_line_button.set_selected(true);
        }
    }
    pub fn action(&mut self, options: Run3dmodMenuOptions) {
        if self.container_attached {
            self.last_action = Some(options);
        }
    }
}
impl Deferred3dmodButton for Run3dmodButton {
    fn action(&mut self, menu_options: Run3dmodMenuOptions) {
        Run3dmodButton::action(self, menu_options)
    }
    fn get_output_image_file_key(&self) -> Option<&FileKey> {
        self.multi_line_button.get_output_image_file_key()
    }
}
impl ContextMenu for Run3dmodButton {
    fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        self.run_3dmod_menu
            .pop_up_context_menu(self.multi_line_button.is_enabled(), mouse_event);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deferred_menu_action_records_container_dispatch_and_selects_toggle() {
        let mut button = Run3dmodButton::get_deferred_toggle_3dmod_instance("View", None);
        button.set_container(true);
        let options = Run3dmodMenuOptions {
            bin_by_2: true,
            ..Default::default()
        };
        button.menu_action(options);
        assert_eq!(button.last_action, Some(options));
        assert!(button.multi_line_button.button.selected);
        assert_eq!(button.multi_line_button.button.mouse_listener_count, 1);
    }
}
