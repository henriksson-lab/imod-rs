//! `IMOD/Etomo/src/etomo/ui/swing/BrtLogButtonStyleExtension.java`.
//!
//! `ImageObserver` and image decoding are owned by the native GUI frontend.
//! This source unit retains the exact `ButtonStyleExtension` constructor
//! configuration and its process-wide Java singleton.  The observer is an
//! explicit frontend boundary and is retained by the first singleton
//! construction, matching Java's `CompleteIcon` ownership.
#![allow(dead_code)]

pub use super::button_style_extension::ImageObserverBoundary;
use super::button_style_extension::{ButtonStyleExtension, CompleteIconBoundary};
use std::sync::{Arc, LazyLock, Mutex};

/// Boundary for Java `java.awt.image.ImageObserver`.
///
/// A native GUI adapter implements this marker while it owns image loading and
/// image-update notifications.  `BrtLogButtonStyleExtension.java` only passes
/// the observer to `CompleteIcon`; it does not invoke it itself.

/// Java `ScaledImage` constants passed to `CompleteIcon` by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BrtLogScaledImage {
    /// Java `ScaledImage.BRT_LOG` (`logicon.png`).
    BrtLog,
    /// Java `ScaledImage.BRT_LOG_PRESSED` (`logicon-pressed.png`).
    BrtLogPressed,
    /// Java `ScaledImage.BRT_LOG_ROLLOVER` (`logicon-rollover.png`).
    BrtLogRollover,
}

/// GUI-bound `CompleteIcon` construction made by this source unit.
///
/// `CompleteIcon.java` is a separate source unit.  This descriptor is the
/// exact argument bundle that `BrtLogButtonStyleExtension` sends to it, rather
/// than a replacement image implementation.
pub type BrtLogCompleteIconBoundary = CompleteIconBoundary<BrtLogScaledImage>;

/// GUI-bound `ButtonStyleExtension` superclass constructor state.
///
/// `ButtonStyleExtension.java` is a separate source unit.  Keeping this
/// source's six constructor arguments as state makes the inherited styling
/// observable without fabricating a second button toolkit.

/// Java package-private final `BrtLogButtonStyleExtension`.
pub struct BrtLogButtonStyleExtension {
    /// Java superclass `ButtonStyleExtension` state.
    pub button_style_extension: ButtonStyleExtension<BrtLogScaledImage>,
}

/// Java private static `BrtLogButtonStyleExtension.INSTANCE`.
static INSTANCE: LazyLock<Mutex<Option<Arc<BrtLogButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));

impl BrtLogButtonStyleExtension {
    /// Java private `BrtLogButtonStyleExtension(ImageObserver)`.
    fn new(image_observer: Option<Arc<dyn ImageObserverBoundary>>) -> Self {
        Self {
            button_style_extension: ButtonStyleExtension::new(
                false,
                Some(BrtLogCompleteIconBoundary {
                    image: BrtLogScaledImage::BrtLog,
                    selected_image: None,
                    pressed_image: Some(BrtLogScaledImage::BrtLogPressed),
                    rollover_image: Some(BrtLogScaledImage::BrtLogRollover),
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
    ///
    /// Java only uses the supplied observer when `INSTANCE` is null; later
    /// callers receive that same singleton and cannot replace its observer.
    pub fn get_instance(
        image_observer: Option<Arc<dyn ImageObserverBoundary>>,
    ) -> Arc<BrtLogButtonStyleExtension> {
        let mut instance = INSTANCE
            .lock()
            .expect("BrtLogButtonStyleExtension mutex poisoned");
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
        let instance = BrtLogButtonStyleExtension::new(Some(Arc::new(ImageObserver)));
        let style = &instance.button_style_extension;

        assert!(!style.text_gap);
        assert_eq!(
            style.icon.as_ref().unwrap().image,
            BrtLogScaledImage::BrtLog
        );
        assert_eq!(style.icon.as_ref().unwrap().selected_image, None);
        assert_eq!(
            style.icon.as_ref().unwrap().pressed_image,
            Some(BrtLogScaledImage::BrtLogPressed)
        );
        assert_eq!(
            style.icon.as_ref().unwrap().rollover_image,
            Some(BrtLogScaledImage::BrtLogRollover)
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
        let first = BrtLogButtonStyleExtension::get_instance(Some(Arc::new(ImageObserver)));
        let second = BrtLogButtonStyleExtension::get_instance(None);

        assert!(Arc::ptr_eq(&first, &second));
    }
}
