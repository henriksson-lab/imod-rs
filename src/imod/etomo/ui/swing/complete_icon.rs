//! `IMOD/Etomo/src/etomo/ui/swing/CompleteIcon.java`.
//!
//! Image decoding, resource lookup, image observers, and Swing button painting
//! belong to the native GUI adapter.  This unit keeps the four `ImageIcon`
//! slots and the source's sizing/assignment rules without introducing a
//! second widget toolkit.
#![allow(dead_code)]

use std::path::PathBuf;

use super::panel::Dimension;

/// Native-GUI representation of Java `ImageIcon` as used by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageIcon {
    /// The source resource URL, represented by its local source-tree path.
    pub resource: Option<PathBuf>,
    /// Java `ImageIcon.getIconWidth()` result supplied by the frontend.
    pub width: i32,
    /// Java `ImageIcon.getIconHeight()` result supplied by the frontend.
    pub height: i32,
    /// Java `ImageIcon.setImageObserver(ImageObserver)` state.
    pub image_observer_set: bool,
}

/// Boundary for Java `java.awt.image.ImageObserver`.
pub trait ImageObserver {}

/// Boundary for the separately translated Java `ScaledImage` source unit.
///
/// `CompleteIcon.java` only calls `ScaledImage.getImage(ImageObserver)`; image
/// scaling and loading stay owned by that source unit/native frontend.
pub trait ScaledImage {
    fn get_image(&self, image_observer: Option<&dyn ImageObserver>) -> Option<ImageIcon>;
}

/// Boundary for the four Java `AbstractButton.set*Icon` calls in this unit.
pub trait CompleteIconButton {
    fn set_icon(&mut self, icon: ImageIcon);
    fn set_selected_icon(&mut self, icon: ImageIcon);
    fn set_pressed_icon(&mut self, icon: ImageIcon);
    fn set_rollover_icon(&mut self, icon: ImageIcon);
}

/// Source-visible state for a native `AbstractButton` adapter.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompleteIconButtonState {
    pub icon: Option<ImageIcon>,
    pub selected_icon: Option<ImageIcon>,
    pub pressed_icon: Option<ImageIcon>,
    pub rollover_icon: Option<ImageIcon>,
}

impl CompleteIconButton for CompleteIconButtonState {
    fn set_icon(&mut self, icon: ImageIcon) {
        self.icon = Some(icon);
    }

    fn set_selected_icon(&mut self, icon: ImageIcon) {
        self.selected_icon = Some(icon);
    }

    fn set_pressed_icon(&mut self, icon: ImageIcon) {
        self.pressed_icon = Some(icon);
    }

    fn set_rollover_icon(&mut self, icon: ImageIcon) {
        self.rollover_icon = Some(icon);
    }
}

/// Java package-private final `CompleteIcon`.
#[derive(Clone, Debug)]
pub struct CompleteIcon {
    pub icon: Option<ImageIcon>,
    pub selected_icon: Option<ImageIcon>,
    pub pressed_icon: Option<ImageIcon>,
    pub rollover_icon: Option<ImageIcon>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub icon_size: Option<Dimension>,
    pub icon_size_set: bool,
}

impl CompleteIcon {
    /// Java `CompleteIcon(String, String, String, String)`.
    pub fn new_from_files(
        image_file: Option<&str>,
        selected_image_file: Option<&str>,
        pressed_image_file: Option<&str>,
        rollover_image_file: Option<&str>,
    ) -> Self {
        Self {
            icon: Self::create_icon(image_file),
            selected_icon: Self::create_icon(selected_image_file),
            pressed_icon: Self::create_icon(pressed_image_file),
            rollover_icon: Self::create_icon(rollover_image_file),
            width: None,
            height: None,
            icon_size: None,
            icon_size_set: false,
        }
    }

    /// Java `CompleteIcon(ScaledImage, ScaledImage, ScaledImage, ScaledImage,
    /// ImageObserver, boolean)`.  `debug` is deliberately retained although
    /// the Java method does not use it.
    pub fn new_from_scaled_images(
        image: Option<&dyn ScaledImage>,
        selected_image: Option<&dyn ScaledImage>,
        pressed_image: Option<&dyn ScaledImage>,
        rollover_image: Option<&dyn ScaledImage>,
        image_observer: Option<&dyn ImageObserver>,
        debug: bool,
    ) -> Self {
        Self {
            icon: Self::create_icon_from_scaled_image(image, image_observer, debug),
            selected_icon: Self::create_icon_from_scaled_image(
                selected_image,
                image_observer,
                debug,
            ),
            pressed_icon: Self::create_icon_from_scaled_image(pressed_image, image_observer, debug),
            rollover_icon: Self::create_icon_from_scaled_image(
                rollover_image,
                image_observer,
                debug,
            ),
            width: None,
            height: None,
            icon_size: None,
            icon_size_set: false,
        }
    }

    /// Java static `createIcon(String)`.
    pub fn create_icon(image_file: Option<&str>) -> Option<ImageIcon> {
        let image_file = image_file?;
        let resource = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("IMOD/Etomo/src/images")
            .join(image_file);
        if !resource.is_file() {
            return None;
        }
        Some(ImageIcon {
            resource: Some(resource),
            width: -1,
            height: -1,
            image_observer_set: false,
        })
    }

    /// Java static `createIcon(ScaledImage, ImageObserver, boolean)`.
    pub fn create_icon_from_scaled_image(
        scaled_image: Option<&dyn ScaledImage>,
        image_observer: Option<&dyn ImageObserver>,
        _debug: bool,
    ) -> Option<ImageIcon> {
        let mut image_icon = scaled_image?.get_image(image_observer)?;
        image_icon.image_observer_set = true;
        Some(image_icon)
    }

    /// Java `toString`.  As in Java, calling this with a null primary icon is
    /// invalid; Rust reports that condition rather than silently inventing one.
    pub fn to_string(&self) -> String {
        self.icon
            .as_ref()
            .expect("Java CompleteIcon.toString dereferences icon")
            .resource
            .as_ref()
            .map_or_else(String::new, |resource| resource.display().to_string())
    }

    /// Java `setup(AbstractButton)`.
    pub fn setup(&self, button: Option<&mut dyn CompleteIconButton>) {
        let Some(button) = button else {
            return;
        };
        if let Some(icon) = &self.icon {
            button.set_icon(icon.clone());
        }
        if let Some(selected_icon) = &self.selected_icon {
            button.set_selected_icon(selected_icon.clone());
        }
        if let Some(pressed_icon) = &self.pressed_icon {
            button.set_pressed_icon(pressed_icon.clone());
        }
        if let Some(rollover_icon) = &self.rollover_icon {
            button.set_rollover_icon(rollover_icon.clone());
        }
    }

    /// Java synchronized `getIconSize`.
    pub fn get_icon_size(&mut self) -> Option<Dimension> {
        if self.icon_size.is_some() || self.icon_size_set {
            return self.icon_size;
        }
        self.icon_size_set = true;
        let mut icon_size = Dimension {
            width: 0,
            height: 0,
        };
        Self::max_icon_size(self.icon.as_ref(), &mut icon_size);
        Self::max_icon_size(self.selected_icon.as_ref(), &mut icon_size);
        Self::max_icon_size(self.pressed_icon.as_ref(), &mut icon_size);
        // This duplicate pressed-icon comparison is present in the Java source.
        Self::max_icon_size(self.pressed_icon.as_ref(), &mut icon_size);
        if icon_size.width <= 0 && icon_size.height <= 0 {
            self.icon_size = None;
        } else {
            self.icon_size = Some(icon_size);
        }
        self.icon_size
    }

    /// Java private `maxIconSize(ImageIcon, Dimension)`.
    fn max_icon_size(image_icon: Option<&ImageIcon>, icon_size: &mut Dimension) {
        let Some(image_icon) = image_icon else {
            return;
        };
        if image_icon.width > icon_size.width {
            icon_size.width = image_icon.width;
        }
        if image_icon.height > icon_size.height {
            icon_size.height = image_icon.height;
        }
    }

    /// Java `setSelectedAppearance(AbstractButton, boolean)`.
    pub fn set_selected_appearance(
        &self,
        button: Option<&mut dyn CompleteIconButton>,
        selected: bool,
    ) {
        let Some(button) = button else {
            return;
        };
        if self.icon.is_some() && self.selected_icon.is_some() {
            button.set_icon(
                if selected {
                    self.selected_icon.as_ref()
                } else {
                    self.icon.as_ref()
                }
                .expect("Java null checks above guarantee an icon")
                .clone(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Observer;
    impl ImageObserver for Observer {}

    struct TestScaledImage(Option<ImageIcon>);
    impl ScaledImage for TestScaledImage {
        fn get_image(&self, _image_observer: Option<&dyn ImageObserver>) -> Option<ImageIcon> {
            self.0.clone()
        }
    }

    #[test]
    fn file_constructor_uses_system_images_resource_and_preserves_missing_icons() {
        let icon = CompleteIcon::new_from_files(Some("x.png"), None, Some("x-pressed.png"), None);
        assert!(
            icon.icon
                .as_ref()
                .is_some_and(|icon| icon.resource.is_some())
        );
        assert!(icon.selected_icon.is_none());
        assert!(icon.pressed_icon.is_some());
        assert!(icon.rollover_icon.is_none());
        assert!(icon.to_string().ends_with("images/x.png"));
    }

    #[test]
    fn scaled_constructor_sets_observer_and_setup_assigns_each_non_null_slot() {
        let image = TestScaledImage(Some(ImageIcon {
            resource: None,
            width: 12,
            height: 8,
            image_observer_set: false,
        }));
        let selected = TestScaledImage(Some(ImageIcon {
            resource: None,
            width: 10,
            height: 20,
            image_observer_set: false,
        }));
        let mut icon = CompleteIcon::new_from_scaled_images(
            Some(&image),
            Some(&selected),
            None,
            None,
            Some(&Observer),
            false,
        );
        assert!(
            icon.icon
                .as_ref()
                .is_some_and(|image| image.image_observer_set)
        );
        let mut button = CompleteIconButtonState::default();
        icon.setup(Some(&mut button));
        assert_eq!(button.icon.as_ref().map(|icon| icon.width), Some(12));
        assert_eq!(
            button.selected_icon.as_ref().map(|icon| icon.height),
            Some(20)
        );
        assert!(button.pressed_icon.is_none());
        assert_eq!(
            icon.get_icon_size(),
            Some(Dimension {
                width: 12,
                height: 20
            })
        );
    }

    #[test]
    fn selected_appearance_only_changes_primary_icon_when_both_icons_exist() {
        let icon = CompleteIcon {
            icon: Some(ImageIcon {
                resource: None,
                width: 1,
                height: 1,
                image_observer_set: false,
            }),
            selected_icon: Some(ImageIcon {
                resource: None,
                width: 2,
                height: 2,
                image_observer_set: false,
            }),
            pressed_icon: None,
            rollover_icon: None,
            width: None,
            height: None,
            icon_size: None,
            icon_size_set: false,
        };
        let mut button = CompleteIconButtonState::default();
        icon.set_selected_appearance(Some(&mut button), true);
        assert_eq!(button.icon.as_ref().map(|icon| icon.width), Some(2));
        icon.set_selected_appearance(Some(&mut button), false);
        assert_eq!(button.icon.as_ref().map(|icon| icon.width), Some(1));
    }

    #[test]
    fn sizing_memoizes_null_result_after_all_nonpositive_icons() {
        let mut icon = CompleteIcon {
            icon: Some(ImageIcon {
                resource: None,
                width: -1,
                height: -1,
                image_observer_set: false,
            }),
            selected_icon: None,
            pressed_icon: None,
            rollover_icon: Some(ImageIcon {
                resource: None,
                width: 50,
                height: 60,
                image_observer_set: false,
            }),
            width: None,
            height: None,
            icon_size: None,
            icon_size_set: false,
        };
        // Java compares pressed twice and never compares rollover.
        assert_eq!(icon.get_icon_size(), None);
        assert!(icon.icon_size_set);
        assert_eq!(icon.get_icon_size(), None);
    }
}
