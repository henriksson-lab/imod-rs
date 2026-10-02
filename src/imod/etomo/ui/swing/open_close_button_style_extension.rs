//! `IMOD/Etomo/src/etomo/ui/swing/OpenCloseButtonStyleExtension.java`.
//!
//! A button style that adds a plus/minus icon to a button.
//!
//! A stateless singleton subclass of `ButtonStyleExtension`: it embeds that
//! struct as `base` and implements [`ButtonStyleExtensionVirtual`], overriding
//! `setup`.  Java's lazily created `private static INSTANCE` is a thread-local
//! on the EDT, where every button lives.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

use super::button_style_extension::{ButtonStyleExtension, ButtonStyleExtensionVirtual};
use super::complete_icon::CompleteIcon;
use crate::imod::etomo::jdk::JComponent;

thread_local! {
    /// Java `private static OpenCloseButtonStyleExtension INSTANCE = null`.
    static INSTANCE: RefCell<Option<Rc<OpenCloseButtonStyleExtension>>> =
        const { RefCell::new(None) };
}

/// Java `OpenCloseButtonStyleExtension extends ButtonStyleExtension`.
pub struct OpenCloseButtonStyleExtension {
    base: ButtonStyleExtension,
}

impl Deref for OpenCloseButtonStyleExtension {
    type Target = ButtonStyleExtension;
    fn deref(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ButtonStyleExtensionVirtual for OpenCloseButtonStyleExtension {
    fn get_button_style_extension(&self) -> &ButtonStyleExtension {
        &self.base
    }

    /// Java `@Override setup(AbstractButton, String, boolean)`.
    fn setup(&self, button: Option<&Rc<JComponent>>, label: Option<&str>, debug: bool) {
        self.base.setup(button, label, debug);
        if button.is_none() {
            return;
        }
        // Swing layout: button.setMargin(new Insets(0, 8, 0, 8)).
    }
}

impl OpenCloseButtonStyleExtension {
    /// Java private `OpenCloseButtonStyleExtension()`.
    fn new() -> OpenCloseButtonStyleExtension {
        OpenCloseButtonStyleExtension {
            base: ButtonStyleExtension::new(
                true,
                Some(Rc::new(CompleteIcon::new_string_string_string_string(
                    Some("openSmall.png"),
                    Some("closeSmall.png"),
                    None,
                    None,
                ))),
                None,
                Some(Rc::new(CompleteIcon::new_string_string_string_string(
                    Some("openRedSmall.png"),
                    Some("closeRedSmall.png"),
                    None,
                    None,
                ))),
                None,
                false,
            ),
        }
    }

    /// Java `static getInstance()`.
    pub fn get_instance() -> Rc<OpenCloseButtonStyleExtension> {
        INSTANCE.with(|instance| {
            if instance.borrow().is_none() {
                *instance.borrow_mut() = Some(Rc::new(OpenCloseButtonStyleExtension::new()));
            }
            instance.borrow().clone().unwrap()
        })
    }
}
