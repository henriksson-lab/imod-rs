//! `IMOD/Etomo/src/etomo/ui/swing/Run3dmodSingleLineButton.java`.
#![allow(dead_code)]
use super::{
    context_menu::{ContextMenu, MouseEvent},
    deferred_3dmod_button::Deferred3dmodButton,
    run_3dmod_menu::Run3dmodMenu,
    single_line_button::SingleLineButton,
};
use crate::imod::etomo::{
    process::imod_process::Run3dmodMenuOptions,
    r#type::{dialog_type::DialogType, file_key::FileKey},
};
#[derive(Clone, Debug, PartialEq)]
pub struct Run3dmodSingleLineButton {
    pub single_line_button: SingleLineButton,
    pub deferred: bool,
    pub run_3dmod_menu: Run3dmodMenu,
    pub container_attached: bool,
    pub deferred_3dmod_button_attached: bool,
    pub last_action: Option<Run3dmodMenuOptions>,
}
impl Run3dmodSingleLineButton {
    fn new(
        label: &str,
        container_attached: bool,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
        deferred: bool,
        description: Option<&str>,
    ) -> Self {
        let mut single_line_button =
            SingleLineButton::new_with_toggle_button(Some(label), toggle_button, dialog_type);
        single_line_button
            .multi_line_button
            .set_action_command(Some(label));
        single_line_button.multi_line_button.add_mouse_listener();
        Self {
            single_line_button,
            deferred,
            run_3dmod_menu: if deferred {
                Run3dmodMenu::get_process_button_instance(description)
            } else {
                Run3dmodMenu::get_3dmod_button_instance(description)
            },
            container_attached,
            deferred_3dmod_button_attached: false,
            last_action: None,
        }
    }
    pub fn get_3dmod_instance(label: &str, container_attached: bool) -> Self {
        Self::new(label, container_attached, false, None, false, None)
    }
    pub fn get_deferred_3dmod_instance(label: &str, container_attached: bool) -> Self {
        Self::new(label, container_attached, false, None, true, None)
    }
    /// Rust form of the Java frontend listener hookup.
    pub fn add_listeners(&mut self) {
        self.single_line_button
            .multi_line_button
            .add_mouse_listener();
    }
    #[allow(non_snake_case)]
    /// Native-name adapter for Java `addListeners`.
    pub fn addListeners(&mut self) {
        self.add_listeners();
    }
    pub fn menu_action(&mut self, options: Run3dmodMenuOptions) {
        self.action(options);
        if self.single_line_button.multi_line_button.toggle_button {
            self.single_line_button.multi_line_button.set_selected(true);
        }
    }
    pub fn action(&mut self, options: Run3dmodMenuOptions) {
        if self.container_attached {
            self.last_action = Some(options);
        }
    }
}
impl Deferred3dmodButton for Run3dmodSingleLineButton {
    fn action(&mut self, options: Run3dmodMenuOptions) {
        Run3dmodSingleLineButton::action(self, options)
    }
    fn get_output_image_file_key(&self) -> Option<&FileKey> {
        self.single_line_button
            .multi_line_button
            .get_output_image_file_key()
    }
}
impl ContextMenu for Run3dmodSingleLineButton {
    fn pop_up_context_menu(&mut self, event: MouseEvent) {
        self.run_3dmod_menu.pop_up_context_menu(
            self.single_line_button.multi_line_button.is_enabled(),
            event,
        );
    }
}
