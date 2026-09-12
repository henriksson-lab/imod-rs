//! `IMOD/Etomo/src/etomo/ui/swing/ButtonStyleExtension.java`.
#![allow(dead_code)]

use std::sync::Arc;

use super::appearance_extension::FlagType;

/// Boundary for Java `java.awt.image.ImageObserver`.
pub trait ImageObserverBoundary: Send + Sync {}

/// The four `CompleteIcon` constructor slots owned by one button style.
#[derive(Clone)]
pub struct CompleteIconBoundary<I> {
    pub image: I,
    pub selected_image: Option<I>,
    pub pressed_image: Option<I>,
    pub rollover_image: Option<I>,
    pub image_observer: Option<Arc<dyn ImageObserverBoundary>>,
    pub debug: bool,
    pub icon_size: Option<(i32, i32)>,
}

/// Native GUI calls on Java `AbstractButton` reached by this source unit.
pub trait ButtonStyleButton<I> {
    fn set_icon(&mut self, icon: &CompleteIconBoundary<I>);
    fn set_icon_text_gap(&mut self, gap: i32);
    fn set_horizontal_alignment_left(&mut self);
    fn set_preferred_size(&mut self, size: (i32, i32));
    fn set_maximum_size(&mut self, size: (i32, i32));
    fn set_selected_appearance(&mut self, selected: bool);
}

/// Java package-private abstract `ButtonStyleExtension`.
#[derive(Clone)]
pub struct ButtonStyleExtension<I> {
    pub text_gap: bool,
    pub icon: Option<CompleteIconBoundary<I>>,
    pub template_icon: Option<CompleteIconBoundary<I>>,
    pub error_icon: Option<CompleteIconBoundary<I>>,
    pub preferred_size: Option<(i32, i32)>,
    pub size_from_image: bool,
}

impl<I> ButtonStyleExtension<I> {
    pub const PADDING_FOR_ICON_BUTTON: i32 = 5;

    /// Java `ButtonStyleExtension(boolean, CompleteIcon, CompleteIcon,
    /// CompleteIcon, Dimension, boolean)`.
    pub fn new(
        text_gap: bool,
        icon: Option<CompleteIconBoundary<I>>,
        template_icon: Option<CompleteIconBoundary<I>>,
        error_icon: Option<CompleteIconBoundary<I>>,
        preferred_size: Option<(i32, i32)>,
        size_from_image: bool,
    ) -> Self {
        Self {
            text_gap,
            icon,
            template_icon,
            error_icon,
            preferred_size,
            size_from_image,
        }
    }

    /// Java synchronized `setup(AbstractButton, String, boolean)`.
    pub fn setup<B: ButtonStyleButton<I>>(
        &mut self,
        button: Option<&mut B>,
        label: Option<&str>,
        _debug: bool,
    ) {
        let Some(button) = button else { return };
        if let Some(icon) = &self.icon {
            button.set_icon(icon);
        }
        if self.text_gap && label.is_some() && (self.icon.is_some() || self.template_icon.is_some())
        {
            button.set_icon_text_gap(10);
            button.set_horizontal_alignment_left();
        }
        if !self.text_gap
            && label.is_none()
            && self.preferred_size.is_none()
            && self.size_from_image
        {
            let mut icon_size = None;
            if let Some(icon) = &self.icon {
                icon_size = self.max_icon_size(icon, icon_size);
            }
            if let Some(icon) = &self.template_icon {
                icon_size = self.max_icon_size(icon, icon_size);
            }
            if let Some(icon) = &self.template_icon {
                icon_size = self.max_icon_size(icon, icon_size);
            }
            if let Some(icon) = &self.error_icon {
                icon_size = self.max_icon_size(icon, icon_size);
            }
            if let Some((width, height)) =
                icon_size.filter(|(width, height)| *width > 0 && *height > 0)
            {
                self.preferred_size = Some((
                    width + Self::PADDING_FOR_ICON_BUTTON,
                    height + Self::PADDING_FOR_ICON_BUTTON,
                ));
            }
        }
        if let Some(size) = self.preferred_size {
            button.set_preferred_size(size);
            button.set_maximum_size(size);
        }
    }

    /// Java private `maxIconSize(CompleteIcon, Dimension)`.
    fn max_icon_size(
        &self,
        complete_icon: &CompleteIconBoundary<I>,
        icon_size: Option<(i32, i32)>,
    ) -> Option<(i32, i32)> {
        let new_icon_size = complete_icon.icon_size?;
        Some(match icon_size {
            None => new_icon_size,
            Some((width, height)) => (width.max(new_icon_size.0), height.max(new_icon_size.1)),
        })
    }

    /// Java final `updateAppearance(AbstractButton, FlagType, boolean, boolean)`.
    pub fn update_appearance<B: ButtonStyleButton<I>>(
        &self,
        button: Option<&mut B>,
        flag_type: Option<FlagType>,
        implement_toggle: bool,
        selected: bool,
    ) {
        let Some(button) = button else { return };
        let mut current_icon = None;
        if let Some(flag_type) = flag_type {
            if flag_type.is_template() {
                current_icon = self.template_icon.as_ref();
            } else if flag_type.is_error() {
                current_icon = self.error_icon.as_ref();
            }
        }
        let current_icon = current_icon.or(self.icon.as_ref());
        if let Some(icon) = current_icon {
            button.set_icon(icon);
            if implement_toggle {
                button.set_selected_appearance(selected);
            }
        }
    }
}

/// Shared native-boundary record for Java `ButtonStyleExtension` calls from
/// widgets whose concrete Swing buttons remain outside this crate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ButtonStyleExtensionBoundary {
    pub style_name: &'static str,
    pub label: Option<String>,
    pub debug: bool,
    pub update_count: usize,
    pub setup_count: usize,
    pub last_flag_type: Option<FlagType>,
    pub last_implement_toggle: bool,
    pub last_selected: bool,
    pub last_update: Option<(Option<FlagType>, bool, bool)>,
}

impl ButtonStyleExtensionBoundary {
    /// Java widget construction selecting a concrete `ButtonStyleExtension`.
    pub fn new(style_name: &'static str) -> Self {
        Self {
            style_name,
            label: None,
            debug: false,
            update_count: 0,
            setup_count: 0,
            last_flag_type: None,
            last_implement_toggle: false,
            last_selected: false,
            last_update: None,
        }
    }

    /// Java `ButtonStyleExtension.setup(AbstractButton, String, boolean)`.
    pub fn setup(&mut self, label: Option<&str>, debug: bool) {
        self.label = label.map(str::to_owned);
        self.debug = debug;
        self.setup_count += 1;
    }

    /// Java `ButtonStyleExtension.updateAppearance(AbstractButton, FlagType, boolean, boolean)`.
    pub fn update_appearance(
        &mut self,
        flag_type: Option<FlagType>,
        implement_toggle: bool,
        selected: bool,
    ) {
        self.update_count += 1;
        self.last_flag_type = flag_type;
        self.last_implement_toggle = implement_toggle;
        self.last_selected = selected;
        self.last_update = Some((flag_type, implement_toggle, selected));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Button {
        icon_set: bool,
        gap: Option<i32>,
        left: bool,
        preferred: Option<(i32, i32)>,
        maximum: Option<(i32, i32)>,
        selected: Option<bool>,
    }
    impl ButtonStyleButton<&'static str> for Button {
        fn set_icon(&mut self, _: &CompleteIconBoundary<&'static str>) {
            self.icon_set = true;
        }
        fn set_icon_text_gap(&mut self, gap: i32) {
            self.gap = Some(gap);
        }
        fn set_horizontal_alignment_left(&mut self) {
            self.left = true;
        }
        fn set_preferred_size(&mut self, size: (i32, i32)) {
            self.preferred = Some(size);
        }
        fn set_maximum_size(&mut self, size: (i32, i32)) {
            self.maximum = Some(size);
        }
        fn set_selected_appearance(&mut self, selected: bool) {
            self.selected = Some(selected);
        }
    }
    fn icon(image: &'static str, size: Option<(i32, i32)>) -> CompleteIconBoundary<&'static str> {
        CompleteIconBoundary {
            image,
            selected_image: None,
            pressed_image: None,
            rollover_image: None,
            image_observer: None,
            debug: false,
            icon_size: size,
        }
    }
    #[test]
    fn setup_uses_max_icon_size_and_padding() {
        let mut style = ButtonStyleExtension::new(
            false,
            Some(icon("a", Some((4, 9)))),
            Some(icon("b", Some((10, 2)))),
            None,
            None,
            true,
        );
        let mut button = Button::default();
        style.setup(Some(&mut button), None, false);
        assert_eq!(button.preferred, Some((15, 14)));
        assert_eq!(button.maximum, Some((15, 14)));
    }
    #[test]
    fn update_uses_error_icon_and_toggle() {
        let style = ButtonStyleExtension::new(
            false,
            Some(icon("normal", None)),
            None,
            Some(icon("error", None)),
            None,
            false,
        );
        let mut button = Button::default();
        style.update_appearance(Some(&mut button), Some(FlagType::ERROR), true, true);
        assert!(button.icon_set);
        assert_eq!(button.selected, Some(true));
    }
}
