//! `IMOD/Etomo/src/etomo/ui/swing/FileOpenButtonStyleExtension.java`.
//!
//! A button style that added a folder icon.
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
    /// Java `private static FileOpenButtonStyleExtension INSTANCE = null`.
    static INSTANCE: RefCell<Option<Rc<FileOpenButtonStyleExtension>>> = const { RefCell::new(None) };
}

/// Java `FileOpenButtonStyleExtension extends ButtonStyleExtension`.
pub struct FileOpenButtonStyleExtension {
    base: ButtonStyleExtension,
}

impl Deref for FileOpenButtonStyleExtension {
    type Target = ButtonStyleExtension;
    fn deref(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl ButtonStyleExtensionVirtual for FileOpenButtonStyleExtension {
    fn get_button_style_extension(&self) -> &ButtonStyleExtension {
        &self.base
    }
}

impl FileOpenButtonStyleExtension {
    /// Java private `FileOpenButtonStyleExtension(ImageObserver, boolean)`.
    /// (Java never reads `debug`.)
    fn new(image_observer: Option<&Rc<JComponent>>, _debug: bool) -> FileOpenButtonStyleExtension {
        FileOpenButtonStyleExtension {
            base: ButtonStyleExtension::new(
                false,
                Some(Rc::new(
                CompleteIcon::new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
                    Some(&scaled_image::OPEN_FILE), None, None, Some(&scaled_image::OPEN_FILE_FOOL),
                    image_observer,
                    false,
                ),
            )),
                Some(Rc::new(
                CompleteIcon::new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
                    Some(&scaled_image::OPEN_FILE_PEET), None, None, None,
                    image_observer,
                    false,
                ),
            )),
                Some(Rc::new(
                CompleteIcon::new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
                    Some(&scaled_image::OPEN_FILE_RED), None, None, None,
                    image_observer,
                    false,
                ),
            )),
                None,
                true,
            ),
        }
    }

    /// Java `synchronized static getInstance(ImageObserver)`.  The lock has no Rust counterpart:
    /// styles are only used on the EDT.
    pub fn get_instance(image_observer: &Rc<JComponent>) -> Rc<FileOpenButtonStyleExtension> {
        INSTANCE.with(|instance| {
            if instance.borrow().is_none() {
                *instance.borrow_mut() = Some(Rc::new(FileOpenButtonStyleExtension::new(
                    Some(image_observer),
                    true,
                )));
            }
            instance.borrow().clone().unwrap()
        })
    }
}
