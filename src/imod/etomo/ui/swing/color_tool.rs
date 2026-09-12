//! `IMOD/Etomo/src/etomo/ui/swing/ColorTool.java`.
//!
//! `UIManager.getColor` is the native look-and-feel boundary.  This module
//! retains the two source singleton instances and their cached foreground
//! colors; a frontend supplies UI-manager values before either singleton is
//! first requested.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use super::ui_utilities::Color;

/// Native-GUI representation of the color entries queried through Java
/// `UIManager.getColor(String)` by this source unit.
static UI_MANAGER_COLORS: LazyLock<Mutex<HashMap<String, Color>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Native GUI boundary for `UIManager.put(property, Color)` / look-and-feel
/// installation.  As in Java, values affect a `ColorTool` only when its
/// singleton is constructed and caches them.
pub fn set_ui_manager_color(property: &str, color: Color) {
    UI_MANAGER_COLORS
        .lock()
        .expect("ColorTool UIManager color map lock")
        .insert(property.into(), color);
}

/// Java static final `ColorTool.TOGGLE_BUTTON`.
pub static TOGGLE_BUTTON: LazyLock<ColorTool> = LazyLock::new(|| {
    ColorTool::new(
        Some("ToggleButton.foreground"),
        Some("ToggleButton.disabledText"),
    )
});

/// Java static final `ColorTool.BUTTON`.
pub static BUTTON: LazyLock<ColorTool> =
    LazyLock::new(|| ColorTool::new(Some("Button.foreground"), Some("Button.disabledText")));

/// Java package-private final `ColorTool`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColorTool {
    pub enabled_foreground_property: Option<String>,
    pub disabled_foreground_property: Option<String>,
    pub enabled_foreground: Color,
    pub disabled_foreground: Color,
}

impl ColorTool {
    /// Java private `ColorTool(String, String)`.
    fn new(
        enabled_foreground_property: Option<&str>,
        disabled_foreground_property: Option<&str>,
    ) -> Self {
        let mut color_tool = Self {
            enabled_foreground_property: enabled_foreground_property.map(str::to_owned),
            disabled_foreground_property: disabled_foreground_property.map(str::to_owned),
            enabled_foreground: Color::default(),
            disabled_foreground: Color::default(),
        };
        color_tool.enabled_foreground = color_tool.get_color(enabled_foreground_property);
        color_tool.disabled_foreground = color_tool.get_color(disabled_foreground_property);
        color_tool
    }

    /// Java static `getButtonInstance(boolean)`.
    pub fn get_button_instance(toggle_button: bool) -> &'static Self {
        if toggle_button {
            &TOGGLE_BUTTON
        } else {
            &BUTTON
        }
    }

    /// Java `toString()`.
    pub fn to_string(&self) -> String {
        if std::ptr::eq(self, &*TOGGLE_BUTTON) {
            return "TOGGLE_BUTTON".into();
        }
        if std::ptr::eq(self, &*BUTTON) {
            return "BUTTON".into();
        }
        // Java falls through to Object.toString(), whose identity hash is not
        // stable or meaningful outside the JVM.  Preserve that identity-only
        // distinction without claiming a Java hash value.
        format!("{}@{:p}", std::any::type_name::<Self>(), self)
    }

    /// Java private `getColor(String)`.
    fn get_color(&self, property: Option<&str>) -> Color {
        if let Some(property) = property
            && let Some(color) = UI_MANAGER_COLORS
                .lock()
                .expect("ColorTool UIManager color map lock")
                .get(property)
                .copied()
        {
            return color;
        }
        if property.is_none() || property == self.enabled_foreground_property.as_deref() {
            return Color {
                red: 51,
                green: 51,
                blue: 51,
            };
        }
        Color {
            red: 153,
            green: 153,
            blue: 153,
        }
    }

    /// Java `getForeground(boolean)`.
    pub fn get_foreground(&self, enabled: bool) -> Color {
        if enabled {
            self.enabled_foreground
        } else {
            self.disabled_foreground
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_fallbacks_distinguish_enabled_and_disabled_properties() {
        let tool = ColorTool::new(
            Some("ColorTool.test.enabled"),
            Some("ColorTool.test.disabled"),
        );
        assert_eq!(
            tool.get_foreground(true),
            Color {
                red: 51,
                green: 51,
                blue: 51
            }
        );
        assert_eq!(
            tool.get_foreground(false),
            Color {
                red: 153,
                green: 153,
                blue: 153
            }
        );
    }

    #[test]
    fn null_property_uses_the_enabled_fallback_as_in_java() {
        let tool = ColorTool::new(
            Some("ColorTool.test.enabled"),
            Some("ColorTool.test.disabled"),
        );
        assert_eq!(
            tool.get_color(None),
            Color {
                red: 51,
                green: 51,
                blue: 51
            }
        );
    }

    #[test]
    fn ui_manager_value_is_cached_by_the_constructor() {
        set_ui_manager_color(
            "ColorTool.test.cached",
            Color {
                red: 7,
                green: 8,
                blue: 9,
            },
        );
        let tool = ColorTool::new(
            Some("ColorTool.test.cached"),
            Some("ColorTool.test.disabled"),
        );
        set_ui_manager_color(
            "ColorTool.test.cached",
            Color {
                red: 1,
                green: 2,
                blue: 3,
            },
        );
        assert_eq!(
            tool.get_foreground(true),
            Color {
                red: 7,
                green: 8,
                blue: 9
            }
        );
    }

    #[test]
    fn button_selection_returns_the_source_singletons_and_names() {
        let toggle = ColorTool::get_button_instance(true);
        let button = ColorTool::get_button_instance(false);
        assert!(std::ptr::eq(toggle, &*TOGGLE_BUTTON));
        assert!(std::ptr::eq(button, &*BUTTON));
        assert_eq!(toggle.to_string(), "TOGGLE_BUTTON");
        assert_eq!(button.to_string(), "BUTTON");
    }
}
