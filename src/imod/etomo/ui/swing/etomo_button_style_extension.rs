//! `IMOD/Etomo/src/etomo/ui/swing/EtomoButtonStyleExtension.java`.
//!
//! Image observation and decoding belong to the native GUI frontend.  This
//! source unit retains the exact `ButtonStyleExtension` constructor arguments
//! and Java singleton ownership; its observer is captured only while the
//! singleton is first constructed.
#![allow(dead_code)]

use std::sync::{Arc, LazyLock, Mutex};

/// Boundary for Java `java.awt.image.ImageObserver`.
pub trait ImageObserverBoundary: Send + Sync {}

/// Java `ScaledImage` constants passed to `CompleteIcon` by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EtomoScaledImage {
    /// Java `ScaledImage.ETOMO` (`etomoicon.png`).
    Etomo,
    /// Java `ScaledImage.ETOMO_PRESSED` (`etomoicon-pressed.png`).
    EtomoPressed,
    /// Java `ScaledImage.ETOMO_ROLLOVER` (`etomoicon-rollover.png`).
    EtomoRollover,
}

/// GUI-bound `CompleteIcon` construction made by this source unit.
pub struct EtomoCompleteIconBoundary {
    pub image: EtomoScaledImage,
    pub selected_image: Option<EtomoScaledImage>,
    pub pressed_image: Option<EtomoScaledImage>,
    pub rollover_image: Option<EtomoScaledImage>,
    pub image_observer: Option<Arc<dyn ImageObserverBoundary>>,
    pub debug: bool,
}

/// GUI-bound `ButtonStyleExtension` superclass constructor state.
pub struct ButtonStyleExtensionBoundary {
    pub text_gap: bool,
    pub icon: EtomoCompleteIconBoundary,
    pub template_icon: Option<()>,
    pub error_icon: Option<()>,
    pub preferred_size: Option<()>,
    pub size_from_image: bool,
}

/// Java package-private final `EtomoButtonStyleExtension`.
pub struct EtomoButtonStyleExtension {
    /// Java superclass `ButtonStyleExtension` state.
    pub button_style_extension: ButtonStyleExtensionBoundary,
}

/// Java private static `EtomoButtonStyleExtension.INSTANCE`.
static INSTANCE: LazyLock<Mutex<Option<Arc<EtomoButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));

impl EtomoButtonStyleExtension {
    /// Java private `EtomoButtonStyleExtension(ImageObserver)`.
    fn new(image_observer: Option<Arc<dyn ImageObserverBoundary>>) -> Self {
        Self {
            button_style_extension: ButtonStyleExtensionBoundary {
                text_gap: false,
                icon: EtomoCompleteIconBoundary {
                    image: EtomoScaledImage::Etomo,
                    selected_image: None,
                    pressed_image: Some(EtomoScaledImage::EtomoPressed),
                    rollover_image: Some(EtomoScaledImage::EtomoRollover),
                    image_observer,
                    debug: false,
                },
                template_icon: None,
                error_icon: None,
                preferred_size: None,
                size_from_image: true,
            },
        }
    }

    /// Java static `getInstance(ImageObserver)`.
    pub fn get_instance(
        image_observer: Option<Arc<dyn ImageObserverBoundary>>,
    ) -> Arc<EtomoButtonStyleExtension> {
        let mut instance = INSTANCE
            .lock()
            .expect("EtomoButtonStyleExtension mutex poisoned");
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
        let instance = EtomoButtonStyleExtension::new(Some(Arc::new(ImageObserver)));
        let style = &instance.button_style_extension;

        assert!(!style.text_gap);
        assert_eq!(style.icon.image, EtomoScaledImage::Etomo);
        assert_eq!(style.icon.selected_image, None);
        assert_eq!(
            style.icon.pressed_image,
            Some(EtomoScaledImage::EtomoPressed)
        );
        assert_eq!(
            style.icon.rollover_image,
            Some(EtomoScaledImage::EtomoRollover)
        );
        assert!(style.icon.image_observer.is_some());
        assert!(!style.icon.debug);
        assert!(style.template_icon.is_none());
        assert!(style.error_icon.is_none());
        assert!(style.preferred_size.is_none());
        assert!(style.size_from_image);
    }

    #[test]
    fn get_instance_returns_the_java_singleton() {
        let first = EtomoButtonStyleExtension::get_instance(Some(Arc::new(ImageObserver)));
        let second = EtomoButtonStyleExtension::get_instance(None);

        assert!(Arc::ptr_eq(&first, &second));
    }
}
