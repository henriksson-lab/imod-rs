//! `IMOD/Etomo/src/etomo/ui/swing/ComponentStyleExtension.java`.
//!
//! Swing owns concrete `java.awt.Component` instances.  The
//! `ComponentStyleComponent` trait is the exact GUI boundary needed here:
//! this source unit reads enabled state and assigns foreground/background
//! colors, without reimplementing a widget toolkit.
#![allow(dead_code)]

use super::appearance_extension::{Color, ComponentBoundary, FlagType};

/// GUI boundary for the `java.awt.Component` methods used by this source unit.
pub trait ComponentStyleComponent {
    /// Java `Component.isEnabled()`.
    fn is_enabled(&self) -> bool;

    /// Java `Component.setForeground(Color)`.
    fn set_foreground(&mut self, color: Color);

    /// Java `Component.setBackground(Color)`.
    fn set_background(&mut self, color: Color);
}

impl ComponentStyleComponent for ComponentBoundary {
    fn is_enabled(&self) -> bool {
        ComponentBoundary::is_enabled(self)
    }

    fn set_foreground(&mut self, color: Color) {
        ComponentBoundary::set_foreground(self, color);
    }

    fn set_background(&mut self, color: Color) {
        ComponentBoundary::set_background(self, color);
    }
}

/// Java package-private final `ComponentStyleExtension`.
pub struct ComponentStyleExtension;

/// Java static `ComponentStyleExtension.INSTANCE`.
pub static INSTANCE: ComponentStyleExtension = ComponentStyleExtension;

impl ComponentStyleExtension {
    /// Java private `ComponentStyleExtension()`.
    const fn new() -> Self {
        Self
    }

    /// Java `updateAppearance(Component, FlagType, Color, Color)`.
    pub fn update_appearance(
        &self,
        component: Option<&mut dyn ComponentStyleComponent>,
        flag_type: Option<FlagType>,
        default_foreground: Option<Color>,
        default_background: Option<Color>,
    ) {
        let Some(component) = component else {
            return;
        };
        if flag_type.is_none() || !component.is_enabled() {
            if let Some(default_foreground) = default_foreground {
                component.set_foreground(default_foreground);
            }
            if let Some(default_background) = default_background {
                component.set_background(default_background);
            }
        } else if let Some(flag_type) = flag_type {
            if !flag_type.background {
                component.set_foreground(flag_type.color);
            } else {
                component.set_background(flag_type.color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::ui::swing::appearance_extension::{BLACK, WARNING_BACKGROUND};

    #[test]
    fn null_component_returns_without_changes() {
        INSTANCE.update_appearance(None, Some(FlagType::ERROR), Some(BLACK), None);
    }

    #[test]
    fn absent_flag_restores_each_non_null_default() {
        let mut component = ComponentBoundary {
            foreground: Some(FlagType::ERROR.color),
            background: Some(FlagType::WARNING.color),
            enabled: true,
        };

        INSTANCE.update_appearance(
            Some(&mut component),
            None,
            Some(BLACK),
            Some(WARNING_BACKGROUND),
        );

        assert_eq!(component.foreground, Some(BLACK));
        assert_eq!(component.background, Some(WARNING_BACKGROUND));
    }

    #[test]
    fn disabled_component_restores_defaults_even_when_flagged() {
        let mut component = ComponentBoundary {
            foreground: Some(BLACK),
            background: None,
            enabled: false,
        };

        INSTANCE.update_appearance(
            Some(&mut component),
            Some(FlagType::ERROR),
            Some(FlagType::TEMPLATE.color),
            Some(WARNING_BACKGROUND),
        );

        assert_eq!(component.foreground, Some(FlagType::TEMPLATE.color));
        assert_eq!(component.background, Some(WARNING_BACKGROUND));
    }

    #[test]
    fn foreground_and_background_flags_change_only_their_source_property() {
        let mut component = ComponentBoundary::default();
        component.background = Some(BLACK);

        INSTANCE.update_appearance(Some(&mut component), Some(FlagType::ERROR), None, None);
        assert_eq!(component.foreground, Some(FlagType::ERROR.color));
        assert_eq!(component.background, Some(BLACK));

        INSTANCE.update_appearance(Some(&mut component), Some(FlagType::WARNING), None, None);
        assert_eq!(component.foreground, Some(FlagType::ERROR.color));
        assert_eq!(component.background, Some(FlagType::WARNING.color));
    }

    #[test]
    fn private_constructor_has_the_same_stateless_extension() {
        let extension = ComponentStyleExtension::new();
        let mut component = ComponentBoundary::default();
        extension.update_appearance(Some(&mut component), Some(FlagType::ERROR), None, None);
        assert_eq!(component.foreground, Some(FlagType::ERROR.color));
    }
}
