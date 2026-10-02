//! `IMOD/Etomo/src/etomo/ui/swing/CloseButtonStyleExtension.java`.
//!
//! A tiny button for closing things.  Has an x image with a red
//! rollover.
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
use super::scaled_image;
use crate::imod::etomo::jdk::JComponent;

thread_local! {
    /// Java `private static CloseButtonStyleExtension INSTANCE = null`.
    static INSTANCE: RefCell<Option<Rc<CloseButtonStyleExtension>>> = const { RefCell::new(None) };
}

/// Java `CloseButtonStyleExtension extends ButtonStyleExtension`.
pub struct CloseButtonStyleExtension {
    base: ButtonStyleExtension,
}

impl Deref for CloseButtonStyleExtension {
    type Target = ButtonStyleExtension;
    fn deref(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ButtonStyleExtensionVirtual for CloseButtonStyleExtension {
    fn get_button_style_extension(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl CloseButtonStyleExtension {
    /// Java private `CloseButtonStyleExtension(ImageObserver)`.
    fn new(image_observer: Option<&Rc<JComponent>>) -> CloseButtonStyleExtension {
        CloseButtonStyleExtension {
            base: ButtonStyleExtension::new(
                false,
                Some(Rc::new(
                CompleteIcon::new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
                    Some(&scaled_image::SMALL_X), None, Some(&scaled_image::SMALL_X_PRESSED), Some(&scaled_image::SMALL_X_ROLLOVER),
                    image_observer,
                    false,
                ),
            )),
                None,
                None,
                None,
                true,
            ),
        }
    }

    /// Java `static getInstance(ImageObserver)`.
    pub fn get_instance(image_observer: &Rc<JComponent>) -> Rc<CloseButtonStyleExtension> {
        INSTANCE.with(|instance| {
            if instance.borrow().is_none() {
                *instance.borrow_mut() = Some(Rc::new(CloseButtonStyleExtension::new(Some(
                    image_observer,
                ))));
            }
            instance.borrow().clone().unwrap()
        })
    }
}
