//! `IMOD/Etomo/src/etomo/ui/swing/XButtonStyleExtension.java`.
//!
//! A button style that adds an X icon.
//!
//! A stateless singleton subclass of `ButtonStyleExtension`: it embeds that
//! struct as `base` and implements [`ButtonStyleExtensionVirtual`] with the
//! inherited bodies.  Java's lazily created `private static INSTANCE` is a
//! thread-local on the EDT, where every button lives.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

use super::button_style_extension::{ButtonStyleExtension, ButtonStyleExtensionVirtual};
use super::complete_icon::CompleteIcon;

thread_local! {
    /// Java `private static XButtonStyleExtension INSTANCE = null`.
    static INSTANCE: RefCell<Option<Rc<XButtonStyleExtension>>> = const { RefCell::new(None) };
}

/// Java `XButtonStyleExtension extends ButtonStyleExtension`.
pub struct XButtonStyleExtension {
    base: ButtonStyleExtension,
}

impl Deref for XButtonStyleExtension {
    type Target = ButtonStyleExtension;
    fn deref(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ButtonStyleExtensionVirtual for XButtonStyleExtension {
    fn get_button_style_extension(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl XButtonStyleExtension {
    /// Java private `XButtonStyleExtension()`.
    fn new() -> XButtonStyleExtension {
        XButtonStyleExtension {
            base: ButtonStyleExtension::new(
                false,
                Some(Rc::new(CompleteIcon::new_string_string_string_string(
                    Some("x.png"),
                    None,
                    Some("x-pressed.png"),
                    None,
                ))),
                Some(Rc::new(CompleteIcon::new_string_string_string_string(
                    Some("xBlue.png"),
                    None,
                    Some("xBlue-pressed.png"),
                    None,
                ))),
                Some(Rc::new(CompleteIcon::new_string_string_string_string(
                    Some("xRed.png"),
                    None,
                    Some("xRed-pressed.png"),
                    None,
                ))),
                None,
                false,
            ),
        }
    }

    /// Java `static getInstance()`.
    pub fn get_instance() -> Rc<XButtonStyleExtension> {
        INSTANCE.with(|instance| {
            if instance.borrow().is_none() {
                *instance.borrow_mut() = Some(Rc::new(XButtonStyleExtension::new()));
            }
            instance.borrow().clone().unwrap()
        })
    }
}
