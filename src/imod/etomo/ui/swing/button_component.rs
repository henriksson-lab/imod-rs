//! `IMOD/Etomo/src/etomo/ui/swing/ButtonComponent.java`.
//!
//! `ActionListener` registration and dispatch belong to the GUI toolkit.  This
//! module keeps the four-method source interface as a Rust trait, with an
//! explicit listener boundary rather than manufacturing a second event loop.
#![allow(dead_code)]

/// Boundary for Java `java.awt.event.ActionListener`.
///
/// The frontend owns event construction and listener lifetime.  It invokes
/// this method when Swing would invoke `actionPerformed(ActionEvent)`.
pub trait ActionListenerBoundary {
    /// Java `ActionListener.actionPerformed(ActionEvent)`.
    fn action_performed(&mut self, action_command: Option<&str>);
}

/// Java `ButtonComponent`.
///
/// Java permits a null action command despite the declared `String` return
/// type, so `Option<&str>` preserves that observable value.  Listener
/// registration is intentionally a GUI boundary; implementations pass the
/// listener to the frontend instead of storing a non-owning Rust reference.
pub trait ButtonComponent {
    /// Java `addActionListener(ActionListener)`.
    fn add_action_listener(&mut self, listener: &mut dyn ActionListenerBoundary);

    /// Java `isSelected`.
    fn is_selected(&self) -> bool;

    /// Java `getActionCommand`.
    fn get_action_command(&self) -> Option<&str>;

    /// Java `isEnabled`.
    fn is_enabled(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::{ActionListenerBoundary, ButtonComponent};
    use crate::imod::etomo::ui::swing::check_box::CheckBox;
    use crate::imod::etomo::ui::swing::radio_button::RadioButton;

    struct TestListener {
        command: Option<String>,
    }

    impl ActionListenerBoundary for TestListener {
        fn action_performed(&mut self, action_command: Option<&str>) {
            self.command = action_command.map(str::to_owned);
        }
    }

    struct TestButtonComponent {
        selected: bool,
        action_command: Option<String>,
        enabled: bool,
        listener_count: usize,
    }

    impl ButtonComponent for TestButtonComponent {
        fn add_action_listener(&mut self, _listener: &mut dyn ActionListenerBoundary) {
            self.listener_count += 1;
        }

        fn is_selected(&self) -> bool {
            self.selected
        }

        fn get_action_command(&self) -> Option<&str> {
            self.action_command.as_deref()
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }
    }

    #[test]
    fn source_interface_retains_button_state_and_listener_registration() {
        let mut button = TestButtonComponent {
            selected: true,
            action_command: Some("use-gpus".to_owned()),
            enabled: false,
            listener_count: 0,
        };
        let mut listener = TestListener { command: None };

        button.add_action_listener(&mut listener);

        assert_eq!(button.listener_count, 1);
        assert!(button.is_selected());
        assert_eq!(button.get_action_command(), Some("use-gpus"));
        assert!(!button.is_enabled());
    }

    #[test]
    fn nullable_java_action_command_is_preserved() {
        let button = TestButtonComponent {
            selected: false,
            action_command: None,
            enabled: true,
            listener_count: 0,
        };

        assert_eq!(button.get_action_command(), None);
    }

    #[test]
    fn original_checkbox_and_radio_button_implement_the_contract() {
        let mut listener = TestListener { command: None };
        let mut check_box = CheckBox::new();
        let mut radio_button = RadioButton::new("CPUs only");

        ButtonComponent::add_action_listener(&mut check_box, &mut listener);
        ButtonComponent::add_action_listener(&mut radio_button, &mut listener);

        assert_eq!(ButtonComponent::get_action_command(&check_box), None);
        assert_eq!(
            ButtonComponent::get_action_command(&radio_button),
            Some("CPUs only")
        );
        assert!(ButtonComponent::is_enabled(&check_box));
        assert!(ButtonComponent::is_enabled(&radio_button));
    }
}
