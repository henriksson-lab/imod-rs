//! `IMOD/Etomo/src/etomo/ui/swing/FileOpenButtonStyleExtension.java`.
#![allow(dead_code)]

pub use super::button_style_extension::ImageObserverBoundary;
use super::button_style_extension::{ButtonStyleExtension, CompleteIconBoundary};
use std::sync::{Arc, LazyLock, Mutex};

/// Java `ScaledImage` constants passed to `CompleteIcon` by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileOpenScaledImage {
    OpenFile,
    OpenFileFool,
    OpenFilePeet,
    OpenFileRed,
}

/// This source's distinct `CompleteIcon` image type using the canonical slot record.
pub type FileOpenCompleteIconBoundary = CompleteIconBoundary<FileOpenScaledImage>;

/// Java final `FileOpenButtonStyleExtension`.
pub struct FileOpenButtonStyleExtension {
    pub button_style_extension: ButtonStyleExtension<FileOpenScaledImage>,
}

static INSTANCE: LazyLock<Mutex<Option<Arc<FileOpenButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));

impl FileOpenButtonStyleExtension {
    /// Java private `FileOpenButtonStyleExtension(ImageObserver, boolean)`.
    fn new(image_observer: Option<Arc<dyn ImageObserverBoundary>>, _debug: bool) -> Self {
        Self {
            button_style_extension: ButtonStyleExtension::new(
                false,
                Some(FileOpenCompleteIconBoundary {
                    image: FileOpenScaledImage::OpenFile,
                    selected_image: None,
                    pressed_image: None,
                    rollover_image: Some(FileOpenScaledImage::OpenFileFool),
                    image_observer: image_observer.clone(),
                    debug: false,
                    icon_size: None,
                }),
                Some(FileOpenCompleteIconBoundary {
                    image: FileOpenScaledImage::OpenFilePeet,
                    selected_image: None,
                    pressed_image: None,
                    rollover_image: None,
                    image_observer: image_observer.clone(),
                    debug: false,
                    icon_size: None,
                }),
                Some(FileOpenCompleteIconBoundary {
                    image: FileOpenScaledImage::OpenFileRed,
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

    /// Java synchronized static `getInstance(ImageObserver)`.
    pub fn get_instance(
        image_observer: Option<Arc<dyn ImageObserverBoundary>>,
    ) -> Arc<FileOpenButtonStyleExtension> {
        let mut instance = INSTANCE
            .lock()
            .expect("FileOpenButtonStyleExtension mutex poisoned");
        if instance.is_none() {
            *instance = Some(Arc::new(Self::new(image_observer, true)));
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
    fn constructor_preserves_the_three_complete_icon_argument_sets() {
        let instance = FileOpenButtonStyleExtension::new(Some(Arc::new(Observer)), true);
        let style = &instance.button_style_extension;
        assert!(!style.text_gap);
        assert_eq!(
            style.icon.as_ref().unwrap().image,
            FileOpenScaledImage::OpenFile
        );
        assert_eq!(
            style.icon.as_ref().unwrap().rollover_image,
            Some(FileOpenScaledImage::OpenFileFool)
        );
        assert_eq!(
            style.template_icon.as_ref().unwrap().image,
            FileOpenScaledImage::OpenFilePeet
        );
        assert_eq!(
            style.error_icon.as_ref().unwrap().image,
            FileOpenScaledImage::OpenFileRed
        );
        assert!(style.size_from_image);
    }

    #[test]
    fn singleton_keeps_the_first_observer() {
        let first = FileOpenButtonStyleExtension::get_instance(Some(Arc::new(Observer)));
        let second = FileOpenButtonStyleExtension::get_instance(None);
        assert!(Arc::ptr_eq(&first, &second));
    }
}
