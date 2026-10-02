//! `IMOD/Etomo/src/etomo/ui/swing/ClearButtonStyleExtension.java`.
//!
//! An extension which gives a button an eraser icon.
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
    /// Java `private static ClearButtonStyleExtension INSTANCE = null`.
    static INSTANCE: RefCell<Option<Rc<ClearButtonStyleExtension>>> = const { RefCell::new(None) };
}

/// Java `ClearButtonStyleExtension extends ButtonStyleExtension`.
pub struct ClearButtonStyleExtension {
    base: ButtonStyleExtension,
}

impl Deref for ClearButtonStyleExtension {
    type Target = ButtonStyleExtension;
    fn deref(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ButtonStyleExtensionVirtual for ClearButtonStyleExtension {
    fn get_button_style_extension(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ClearButtonStyleExtension {
    /// Java private `ClearButtonStyleExtension(ImageObserver)`.
    fn new(image_observer: Option<&Rc<JComponent>>) -> ClearButtonStyleExtension {
        ClearButtonStyleExtension {
            base: ButtonStyleExtension::new(
                false,
                Some(Rc::new(
                CompleteIcon::new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
                    Some(&scaled_image::CLEAR), None, None, None,
                    image_observer,
                    false,
                ),
            )),
                Some(Rc::new(
                CompleteIcon::new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
                    Some(&scaled_image::CLEAR_BLUE), None, None, None,
                    image_observer,
                    false,
                ),
            )),
                Some(Rc::new(
                CompleteIcon::new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
                    Some(&scaled_image::CLEAR_RED), None, None, None,
                    image_observer,
                    false,
                ),
            )),
                None,
                true,
            ),
        }
    }

    /// Java `static getInstance(ImageObserver)`.
    pub fn get_instance(image_observer: &Rc<JComponent>) -> Rc<ClearButtonStyleExtension> {
        INSTANCE.with(|instance| {
            if instance.borrow().is_none() {
                *instance.borrow_mut() = Some(Rc::new(ClearButtonStyleExtension::new(Some(
                    image_observer,
                ))));
            }
            instance.borrow().clone().unwrap()
        })
    }
}
