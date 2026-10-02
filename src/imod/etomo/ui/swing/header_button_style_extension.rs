//! `IMOD/Etomo/src/etomo/ui/swing/HeaderButtonStyleExtension.java`.
//!
//! A button style for table header buttons: the label gets a trailing ": ", and the
//! button is unfocusable, transparent and etched.
//!
//! A stateless singleton subclass of `ButtonStyleExtension`: it embeds that struct as
//! `base` and implements [`ButtonStyleExtensionVirtual`], overriding `setup` (which
//! does not call the superclass's).  Java's lazily created `private static INSTANCE`
//! is a thread-local on the EDT, where every button lives.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

use super::button_style_extension::{ButtonStyleExtension, ButtonStyleExtensionVirtual};
use crate::imod::etomo::jdk::JComponent;

thread_local! {
    /// Java `private static HeaderButtonStyleExtension INSTANCE = null`.
    static INSTANCE: RefCell<Option<Rc<HeaderButtonStyleExtension>>> =
        const { RefCell::new(None) };
}

/// Java `final class HeaderButtonStyleExtension extends ButtonStyleExtension`.
pub struct HeaderButtonStyleExtension {
    base: ButtonStyleExtension,
}

impl Deref for HeaderButtonStyleExtension {
    type Target = ButtonStyleExtension;
    fn deref(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ButtonStyleExtensionVirtual for HeaderButtonStyleExtension {
    fn get_button_style_extension(&self) -> &ButtonStyleExtension {
        &self.base
    }

    /// Java final `@Override setup(AbstractButton, String, boolean)`.
    fn setup(&self, button: Option<&Rc<JComponent>>, label: Option<&str>, _debug: bool) {
        let Some(button) = button else {
            return;
        };
        if let Some(label) = label {
            button.set_text(&format!("{label}: "));
        }
        // Swing layout: button.setFocusable(false); button.setContentAreaFilled(false);
        // button.setBorder(BorderFactory.createEtchedBorder()).
    }
}

impl HeaderButtonStyleExtension {
    /// Java private `HeaderButtonStyleExtension()`.
    fn new() -> HeaderButtonStyleExtension {
        HeaderButtonStyleExtension {
            base: ButtonStyleExtension::new(false, None, None, None, None, false),
        }
    }

    /// Java static `getInstance()`.
    pub fn get_instance() -> Rc<HeaderButtonStyleExtension> {
        INSTANCE.with(|instance| {
            if instance.borrow().is_none() {
                *instance.borrow_mut() = Some(Rc::new(HeaderButtonStyleExtension::new()));
            }
            instance.borrow().clone().unwrap()
        })
    }
}
