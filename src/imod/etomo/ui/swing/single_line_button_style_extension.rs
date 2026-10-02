//! `IMOD/Etomo/src/etomo/ui/swing/SingleLineButtonStyleExtension.java`.
//!
//! A button style that gives a button the standard single-line button size.
//!
//! A stateless singleton subclass of `ButtonStyleExtension`: it embeds that struct as
//! `base` and implements [`ButtonStyleExtensionVirtual`] with the inherited bodies.
//! Its own one-argument `setup(AbstractButton)` is an overload, not an override: the
//! three-argument `setup` a button calls is the inherited one.  Java's lazily created
//! `private static INSTANCE` is a thread-local on the EDT, where every button lives.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

use super::button_style_extension::{ButtonStyleExtension, ButtonStyleExtensionVirtual};
use crate::imod::etomo::jdk::JComponent;

thread_local! {
    /// Java `private static SingleLineButtonStyleExtension INSTANCE = null`.
    static INSTANCE: RefCell<Option<Rc<SingleLineButtonStyleExtension>>> =
        const { RefCell::new(None) };
}

/// Java `final class SingleLineButtonStyleExtension extends ButtonStyleExtension`.
pub struct SingleLineButtonStyleExtension {
    base: ButtonStyleExtension,
}

impl Deref for SingleLineButtonStyleExtension {
    type Target = ButtonStyleExtension;
    fn deref(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ButtonStyleExtensionVirtual for SingleLineButtonStyleExtension {
    fn get_button_style_extension(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl SingleLineButtonStyleExtension {
    /// Java private `SingleLineButtonStyleExtension()`.
    fn new() -> SingleLineButtonStyleExtension {
        SingleLineButtonStyleExtension {
            base: ButtonStyleExtension::new(false, None, None, None, None, false),
        }
    }

    /// Java static `getInstance()`.
    pub fn get_instance() -> Rc<SingleLineButtonStyleExtension> {
        INSTANCE.with(|instance| {
            if instance.borrow().is_none() {
                *instance.borrow_mut() = Some(Rc::new(SingleLineButtonStyleExtension::new()));
            }
            instance.borrow().clone().unwrap()
        })
    }

    /// Java `setup(AbstractButton)`.  Does not call `ButtonStyleExtension.setup`.
    pub fn setup(&self, _button: &Rc<JComponent>) {
        // Swing layout: size = UIParameters.getInstance(UIUtilities.getFontMetrics(
        // button)).getButtonSingleLineDimension(); button.setPreferredSize(size);
        // button.setMaximumSize(size).
    }
}
