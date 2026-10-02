//! `IMOD/Etomo/src/etomo/ui/swing/EtomoButtonStyleExtension.java`.
//!
//! Button with an icon that matches the etomo icon.
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
    /// Java `private static EtomoButtonStyleExtension INSTANCE = null`.
    static INSTANCE: RefCell<Option<Rc<EtomoButtonStyleExtension>>> = const { RefCell::new(None) };
}

/// Java `EtomoButtonStyleExtension extends ButtonStyleExtension`.
pub struct EtomoButtonStyleExtension {
    base: ButtonStyleExtension,
}

impl Deref for EtomoButtonStyleExtension {
    type Target = ButtonStyleExtension;
    fn deref(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ButtonStyleExtensionVirtual for EtomoButtonStyleExtension {
    fn get_button_style_extension(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl EtomoButtonStyleExtension {
    /// Java private `EtomoButtonStyleExtension(ImageObserver)`.
    fn new(image_observer: Option<&Rc<JComponent>>) -> EtomoButtonStyleExtension {
        EtomoButtonStyleExtension {
            base: ButtonStyleExtension::new(
                false,
                Some(Rc::new(
                CompleteIcon::new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
                    Some(&scaled_image::ETOMO), None, Some(&scaled_image::ETOMO_PRESSED), Some(&scaled_image::ETOMO_ROLLOVER),
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
    pub fn get_instance(image_observer: &Rc<JComponent>) -> Rc<EtomoButtonStyleExtension> {
        INSTANCE.with(|instance| {
            if instance.borrow().is_none() {
                *instance.borrow_mut() = Some(Rc::new(EtomoButtonStyleExtension::new(Some(
                    image_observer,
                ))));
            }
            instance.borrow().clone().unwrap()
        })
    }
}
