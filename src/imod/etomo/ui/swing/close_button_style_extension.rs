//! `IMOD/Etomo/src/etomo/ui/swing/CloseButtonStyleExtension.java`.
#![allow(dead_code)]
use super::button_style_extension::{
    ButtonStyleExtension, CompleteIconBoundary, ImageObserverBoundary,
};
use std::sync::{Arc, LazyLock, Mutex};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CloseScaledImage {
    SmallX,
    SmallXPressed,
    SmallXRollover,
}
pub type CloseCompleteIconBoundary = CompleteIconBoundary<CloseScaledImage>;
pub struct CloseButtonStyleExtension {
    pub button_style_extension: ButtonStyleExtension<CloseScaledImage>,
}
static INSTANCE: LazyLock<Mutex<Option<Arc<CloseButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));
impl CloseButtonStyleExtension {
    fn new(image_observer: Option<Arc<dyn ImageObserverBoundary>>) -> Self {
        Self {
            button_style_extension: ButtonStyleExtension::new(
                false,
                Some(CloseCompleteIconBoundary {
                    image: CloseScaledImage::SmallX,
                    selected_image: None,
                    pressed_image: Some(CloseScaledImage::SmallXPressed),
                    rollover_image: Some(CloseScaledImage::SmallXRollover),
                    image_observer,
                    debug: false,
                    icon_size: None,
                }),
                None,
                None,
                None,
                true,
            ),
        }
    }
    pub fn get_instance(image_observer: Option<Arc<dyn ImageObserverBoundary>>) -> Arc<Self> {
        let mut instance = INSTANCE
            .lock()
            .expect("CloseButtonStyleExtension mutex poisoned");
        if instance.is_none() {
            *instance = Some(Arc::new(Self::new(image_observer)));
        }
        Arc::clone(instance.as_ref().expect("Java INSTANCE assigned above"))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_has_pressed_and_rollover_icon() {
        let value = CloseButtonStyleExtension::new(None);
        let icon = value.button_style_extension.icon.unwrap();
        assert_eq!(icon.pressed_image, Some(CloseScaledImage::SmallXPressed));
        assert_eq!(icon.rollover_image, Some(CloseScaledImage::SmallXRollover));
    }
}
