//! `IMOD/Etomo/src/etomo/ui/swing/EtomoLogButtonStyleExtension.java`.
//!
//! Image observation and decoding belong to the native GUI frontend.  This
//! source unit retains the exact `ButtonStyleExtension` constructor arguments
//! and Java singleton ownership; its observer is captured only while the
//! singleton is first constructed.
#![allow(dead_code)]

pub use super::button_style_extension::ImageObserverBoundary;
use super::button_style_extension::{ButtonStyleExtension, CompleteIconBoundary};
use std::sync::{Arc, LazyLock, Mutex};

/// Java `ScaledImage` constants passed to `CompleteIcon` by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EtomoLogScaledImage {
    /// Java `ScaledImage.ETOMO_LOG` (`projlogicon.png`).
    EtomoLog,
    /// Java `ScaledImage.ETOMO_LOG_PRESSED` (`projlogicon-pressed.png`).
    EtomoLogPressed,
    /// Java `ScaledImage.ETOMO_LOG_ROLLOVER` (`projlogicon-rollover.png`).
    EtomoLogRollover,
}

pub type EtomoLogCompleteIconBoundary = CompleteIconBoundary<EtomoLogScaledImage>;

/// Java package-private final `EtomoLogButtonStyleExtension`.
pub struct EtomoLogButtonStyleExtension {
    /// Java superclass `ButtonStyleExtension` state.
    pub button_style_extension: ButtonStyleExtension<EtomoLogScaledImage>,
}

/// Java private static `EtomoLogButtonStyleExtension.INSTANCE`.
static INSTANCE: LazyLock<Mutex<Option<Arc<EtomoLogButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));

impl EtomoLogButtonStyleExtension {
    /// Java private `EtomoLogButtonStyleExtension(ImageObserver)`.
    fn new(image_observer: Option<Arc<dyn ImageObserverBoundary>>) -> Self {
        Self {
            button_style_extension: ButtonStyleExtension::new(
                false,
                Some(EtomoLogCompleteIconBoundary {
                    image: EtomoLogScaledImage::EtomoLog,
                    selected_image: None,
                    pressed_image: Some(EtomoLogScaledImage::EtomoLogPressed),
                    rollover_image: Some(EtomoLogScaledImage::EtomoLogRollover),
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

    /// Java static `getInstance(ImageObserver)`.
    pub fn get_instance(
        image_observer: Option<Arc<dyn ImageObserverBoundary>>,
    ) -> Arc<EtomoLogButtonStyleExtension> {
        let mut instance = INSTANCE
            .lock()
            .expect("EtomoLogButtonStyleExtension mutex poisoned");
        if instance.is_none() {
            *instance = Some(Arc::new(Self::new(image_observer)));
        }
        Arc::clone(instance.as_ref().expect("Java INSTANCE assigned above"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    struct ImageObserver;
    impl ImageObserverBoundary for ImageObserver {}

    #[test]
    fn constructor_preserves_the_complete_icon_argument_order() {
        let instance = EtomoLogButtonStyleExtension::new(Some(Arc::new(ImageObserver)));
        let style = &instance.button_style_extension;

        assert!(!style.text_gap);
        assert_eq!(
            style.icon.as_ref().unwrap().image,
            EtomoLogScaledImage::EtomoLog
        );
        assert_eq!(style.icon.as_ref().unwrap().selected_image, None);
        assert_eq!(
            style.icon.as_ref().unwrap().pressed_image,
            Some(EtomoLogScaledImage::EtomoLogPressed)
        );
        assert_eq!(
            style.icon.as_ref().unwrap().rollover_image,
            Some(EtomoLogScaledImage::EtomoLogRollover)
        );
        assert!(style.icon.as_ref().unwrap().image_observer.is_some());
        assert!(!style.icon.as_ref().unwrap().debug);
        assert!(style.template_icon.is_none());
        assert!(style.error_icon.is_none());
        assert!(style.preferred_size.is_none());
        assert!(style.size_from_image);
    }

    #[test]
    fn get_instance_returns_the_java_singleton() {
        let first = EtomoLogButtonStyleExtension::get_instance(Some(Arc::new(ImageObserver)));
        let second = EtomoLogButtonStyleExtension::get_instance(None);

        assert!(Arc::ptr_eq(&first, &second));
    }
}
