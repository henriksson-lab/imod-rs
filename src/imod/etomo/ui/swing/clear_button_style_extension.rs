//! `IMOD/Etomo/src/etomo/ui/swing/ClearButtonStyleExtension.java`.
#![allow(dead_code)]
use super::button_style_extension::{
    ButtonStyleExtension, CompleteIconBoundary, ImageObserverBoundary,
};
use std::sync::{Arc, LazyLock, Mutex};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClearScaledImage {
    Clear,
    ClearBlue,
    ClearRed,
}
pub type ClearCompleteIconBoundary = CompleteIconBoundary<ClearScaledImage>;
pub struct ClearButtonStyleExtension {
    pub button_style_extension: ButtonStyleExtension<ClearScaledImage>,
}
static INSTANCE: LazyLock<Mutex<Option<Arc<ClearButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));
impl ClearButtonStyleExtension {
    /// Java private `ClearButtonStyleExtension(ImageObserver)`.
    fn new(image_observer: Option<Arc<dyn ImageObserverBoundary>>) -> Self {
        Self {
            button_style_extension: ButtonStyleExtension::new(
                false,
                Some(ClearCompleteIconBoundary {
                    image: ClearScaledImage::Clear,
                    selected_image: None,
                    pressed_image: None,
                    rollover_image: None,
                    image_observer: image_observer.clone(),
                    debug: false,
                    icon_size: None,
                }),
                Some(ClearCompleteIconBoundary {
                    image: ClearScaledImage::ClearBlue,
                    selected_image: None,
                    pressed_image: None,
                    rollover_image: None,
                    image_observer: image_observer.clone(),
                    debug: false,
                    icon_size: None,
                }),
                Some(ClearCompleteIconBoundary {
                    image: ClearScaledImage::ClearRed,
                    selected_image: None,
                    pressed_image: None,
                    rollover_image: None,
                    image_observer,
                    debug: false,
                    icon_size: None,
                }),
                None,
                true,
            ),
        }
    }
    /// Java static `getInstance(ImageObserver)`.
    pub fn get_instance(image_observer: Option<Arc<dyn ImageObserverBoundary>>) -> Arc<Self> {
        let mut instance = INSTANCE
            .lock()
            .expect("ClearButtonStyleExtension mutex poisoned");
        if instance.is_none() {
            *instance = Some(Arc::new(Self::new(image_observer)));
        }
        Arc::clone(instance.as_ref().expect("Java INSTANCE assigned above"))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Observer;
    impl ImageObserverBoundary for Observer {}
    #[test]
    fn constructor_has_clear_states() {
        let value = ClearButtonStyleExtension::new(Some(Arc::new(Observer)));
        assert_eq!(
            value.button_style_extension.icon.as_ref().unwrap().image,
            ClearScaledImage::Clear
        );
        assert_eq!(
            value
                .button_style_extension
                .template_icon
                .as_ref()
                .unwrap()
                .image,
            ClearScaledImage::ClearBlue
        );
        assert_eq!(
            value
                .button_style_extension
                .error_icon
                .as_ref()
                .unwrap()
                .image,
            ClearScaledImage::ClearRed
        );
    }
}
