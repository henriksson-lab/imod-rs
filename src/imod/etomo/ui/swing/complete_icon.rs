//! `IMOD/Etomo/src/etomo/ui/swing/CompleteIcon.java`.
//!
//! The four icons of a button (normal, selected, pressed, rollover).  Icons are
//! painting and are not modelled: each `ImageIcon` is held as the name of the image it
//! shows, and setting icons on a button is recorded as comments.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use super::scaled_image::ScaledImage;
use crate::imod::etomo::jdk::{Dimension, JComponent};

/// Java `CompleteIcon`.
pub struct CompleteIcon {
    /// Java `icon` (the image it shows).
    icon: Option<String>,
    /// Java `selectedIcon`.
    selected_icon: Option<String>,
    /// Java `pressedIcon`.
    pressed_icon: Option<String>,
    /// Java `rolloverIcon`.
    rollover_icon: Option<String>,
    /// Java `width` (unused in the Java too).
    #[allow(dead_code)]
    width: Cell<Option<i32>>,
    /// Java `height` (unused in the Java too).
    #[allow(dead_code)]
    height: Cell<Option<i32>>,
    /// Java `iconSize`.
    icon_size: RefCell<Option<Dimension>>,
    /// Java `iconSizeSet`.
    icon_size_set: Cell<bool>,
}

impl CompleteIcon {
    /// Java `CompleteIcon(String, String, String, String)`.
    pub fn new_string_string_string_string(
        image_file: Option<&str>,
        selected_image_file: Option<&str>,
        pressed_image_file: Option<&str>,
        rollover_image_file: Option<&str>,
    ) -> CompleteIcon {
        CompleteIcon {
            icon: CompleteIcon::create_icon_string(image_file),
            selected_icon: CompleteIcon::create_icon_string(selected_image_file),
            pressed_icon: CompleteIcon::create_icon_string(pressed_image_file),
            rollover_icon: CompleteIcon::create_icon_string(rollover_image_file),
            width: Cell::new(None),
            height: Cell::new(None),
            icon_size: RefCell::new(None),
            icon_size_set: Cell::new(false),
        }
    }

    /// Java `CompleteIcon(ScaledImage, ScaledImage, ScaledImage, ScaledImage,
    /// ImageObserver, boolean)`.
    pub fn new_scaled_image_scaled_image_scaled_image_scaled_image_image_observer_boolean(
        image: Option<&ScaledImage>,
        selected_image: Option<&ScaledImage>,
        pressed_image: Option<&ScaledImage>,
        rollover_image: Option<&ScaledImage>,
        image_observer: Option<&Rc<JComponent>>,
        debug: bool,
    ) -> CompleteIcon {
        CompleteIcon {
            icon: CompleteIcon::create_icon_scaled_image_image_observer_boolean(
                image,
                image_observer,
                debug,
            ),
            selected_icon: CompleteIcon::create_icon_scaled_image_image_observer_boolean(
                selected_image,
                image_observer,
                debug,
            ),
            pressed_icon: CompleteIcon::create_icon_scaled_image_image_observer_boolean(
                pressed_image,
                image_observer,
                debug,
            ),
            rollover_icon: CompleteIcon::create_icon_scaled_image_image_observer_boolean(
                rollover_image,
                image_observer,
                debug,
            ),
            width: Cell::new(None),
            height: Cell::new(None),
            icon_size: RefCell::new(None),
            icon_size_set: Cell::new(false),
        }
    }

    /// Java static `createIcon(String)`.  Java returns null when the
    /// `images/<imageFile>` resource is not found; resources are not modelled, so every
    /// named image is taken to exist.
    pub fn create_icon_string(image_file: Option<&str>) -> Option<String> {
        let image_file = image_file?;
        // Swing painting: new ImageIcon(ClassLoader.getSystemResource("images/" +
        // imageFile)).
        Some(image_file.to_string())
    }

    /// Java static `createIcon(ScaledImage, ImageObserver, boolean)`.
    pub fn create_icon_scaled_image_image_observer_boolean(
        scaled_image: Option<&ScaledImage>,
        _image_observer: Option<&Rc<JComponent>>,
        _debug: bool,
    ) -> Option<String> {
        let scaled_image = scaled_image?;
        // Swing painting: imageIcon = new ImageIcon(scaledImage.getImage(imageObserver));
        // imageIcon.setImageObserver(imageObserver).
        Some(scaled_image.get_image().to_string())
    }

    /// Java `toString()`.
    ///
    /// Fixed in translation (`CompleteIcon.java:83`): Java throws a
    /// NullPointerException when there is no normal icon; this returns "null".
    pub fn to_string(&self) -> String {
        self.icon.clone().unwrap_or_else(|| "null".to_string())
    }

    /// Java final `setup(AbstractButton)`.
    pub fn setup(&self, button: Option<&Rc<JComponent>>) {
        if button.is_none() {
            return;
        }
        if self.icon.is_some() {
            // Swing painting: button.setIcon(icon).
        }
        if self.selected_icon.is_some() {
            // Swing painting: button.setSelectedIcon(selectedIcon).
        }
        if self.pressed_icon.is_some() {
            // Swing painting: button.setPressedIcon(pressedIcon).
        }
        if self.rollover_icon.is_some() {
            // Swing painting: button.setRolloverIcon(rolloverIcon).
        }
    }

    /// Java synchronized `getIconSize()`.  Returns the maximum icon size, or null.
    ///
    /// Fixed in translation (`CompleteIcon.java:120-121`): Java measures `pressedIcon`
    /// twice and never `rolloverIcon`; this measures the rollover icon in the second
    /// call.
    pub fn get_icon_size(&self) -> Option<Dimension> {
        if self.icon_size.borrow().is_some() || self.icon_size_set.get() {
            return *self.icon_size.borrow();
        }
        self.icon_size_set.set(true);
        let mut icon_size = Dimension {
            width: 0,
            height: 0,
        };
        self.max_icon_size(self.icon.as_deref(), &mut icon_size);
        self.max_icon_size(self.selected_icon.as_deref(), &mut icon_size);
        self.max_icon_size(self.pressed_icon.as_deref(), &mut icon_size);
        self.max_icon_size(self.rollover_icon.as_deref(), &mut icon_size);
        // Avoid setting or returning an invalid iconSize.
        if icon_size.width <= 0 && icon_size.height <= 0 {
            *self.icon_size.borrow_mut() = None;
        } else {
            *self.icon_size.borrow_mut() = Some(icon_size);
        }
        *self.icon_size.borrow()
    }

    /// Java private `maxIconSize(ImageIcon, Dimension)`.  Use the maximum width and
    /// height, comparing imageIcon with iconSize.
    fn max_icon_size(&self, image_icon: Option<&str>, _icon_size: &mut Dimension) {
        if image_icon.is_none() {
            return;
        }
        // Swing painting: width = imageIcon.getIconWidth(); height =
        // imageIcon.getIconHeight(); widen iconSize to them.  Image sizes are not
        // modelled, so no icon contributes a size.
    }

    /// Java final `setSelectedAppearance(AbstractButton, boolean)`.  Use to cause a
    /// button with no toggle functionality to appear to have toggle functionality.
    pub fn set_selected_appearance(&self, button: Option<&Rc<JComponent>>, _selected: bool) {
        if button.is_none() {
            return;
        }
        if self.icon.is_some() && self.selected_icon.is_some() {
            // Swing painting: button.setIcon(selected ? selectedIcon : icon).
        }
    }
}
