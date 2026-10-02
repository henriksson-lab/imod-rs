//! `IMOD/Etomo/src/etomo/ui/swing/ImodButtonStyleExtension.java`.
//!
//! A button with the 3dmod icon.  (The Java has no class comment.)
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
    /// Java `private static ImodButtonStyleExtension INSTANCE = null`.
    static INSTANCE: RefCell<Option<Rc<ImodButtonStyleExtension>>> = const { RefCell::new(None) };
}

/// Java `ImodButtonStyleExtension extends ButtonStyleExtension`.
pub struct ImodButtonStyleExtension {
    base: ButtonStyleExtension,
}

impl Deref for ImodButtonStyleExtension {
    type Target = ButtonStyleExtension;
    fn deref(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ButtonStyleExtensionVirtual for ImodButtonStyleExtension {
    fn get_button_style_extension(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ImodButtonStyleExtension {
    /// Java private `ImodButtonStyleExtension(ImageObserver)`.
    fn new(image_observer: Option<&Rc<JComponent>>) -> ImodButtonStyleExtension {
        ImodButtonStyleExtension {
            base: ButtonStyleExtension::new(
                false,
                Some(Rc::new(
                CompleteIcon::new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
                    Some(&scaled_image::IMOD), None, Some(&scaled_image::IMOD_PRESSED), Some(&scaled_image::IMOD_ROLLOVER),
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
    pub fn get_instance(image_observer: &Rc<JComponent>) -> Rc<ImodButtonStyleExtension> {
        INSTANCE.with(|instance| {
            if instance.borrow().is_none() {
                *instance.borrow_mut() =
                    Some(Rc::new(ImodButtonStyleExtension::new(Some(image_observer))));
            }
            instance.borrow().clone().unwrap()
        })
    }
}
