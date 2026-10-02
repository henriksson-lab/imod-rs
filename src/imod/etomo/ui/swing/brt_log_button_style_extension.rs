//! `IMOD/Etomo/src/etomo/ui/swing/BrtLogButtonStyleExtension.java`.
//!
//! Button with log icon, with colors that match the mult-tomogram
//! interfaces.
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
    /// Java `private static BrtLogButtonStyleExtension INSTANCE = null`.
    static INSTANCE: RefCell<Option<Rc<BrtLogButtonStyleExtension>>> = const { RefCell::new(None) };
}

/// Java `BrtLogButtonStyleExtension extends ButtonStyleExtension`.
pub struct BrtLogButtonStyleExtension {
    base: ButtonStyleExtension,
}

impl Deref for BrtLogButtonStyleExtension {
    type Target = ButtonStyleExtension;
    fn deref(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ButtonStyleExtensionVirtual for BrtLogButtonStyleExtension {
    fn get_button_style_extension(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl BrtLogButtonStyleExtension {
    /// Java private `BrtLogButtonStyleExtension(ImageObserver)`.
    fn new(image_observer: Option<&Rc<JComponent>>) -> BrtLogButtonStyleExtension {
        BrtLogButtonStyleExtension {
            base: ButtonStyleExtension::new(
                false,
                Some(Rc::new(
                CompleteIcon::new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
                    Some(&scaled_image::BRT_LOG), None, Some(&scaled_image::BRT_LOG_PRESSED), Some(&scaled_image::BRT_LOG_ROLLOVER),
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
    pub fn get_instance(image_observer: &Rc<JComponent>) -> Rc<BrtLogButtonStyleExtension> {
        INSTANCE.with(|instance| {
            if instance.borrow().is_none() {
                *instance.borrow_mut() = Some(Rc::new(BrtLogButtonStyleExtension::new(Some(
                    image_observer,
                ))));
            }
            instance.borrow().clone().unwrap()
        })
    }
}
