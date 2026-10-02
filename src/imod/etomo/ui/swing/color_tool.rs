//! `IMOD/Etomo/src/etomo/ui/swing/ColorTool.java`.
//!
//! Gets colors for various types of widgets.  Can be used to customize colors.
//!
//! `UIManager.getColor` reads the look and feel's defaults, which `jdk.rs` does not
//! model: the lookup finds nothing, so [`ColorTool::get_color`] takes its fallback
//! colours, exactly as the Java does under a look and feel without those keys.

use std::sync::LazyLock;

use crate::imod::etomo::jdk::Color;

/// Java `static final ColorTool TOGGLE_BUTTON =
/// new ColorTool("ToggleButton.foreground", "ToggleButton.disabledText")`.
pub static TOGGLE_BUTTON: LazyLock<ColorTool> =
    LazyLock::new(|| ColorTool::new("ToggleButton.foreground", "ToggleButton.disabledText"));

/// Java `static final ColorTool BUTTON = new ColorTool("Button.foreground",
/// "Button.disabledText")`.
pub static BUTTON: LazyLock<ColorTool> =
    LazyLock::new(|| ColorTool::new("Button.foreground", "Button.disabledText"));

/// Java `final class ColorTool`.
pub struct ColorTool {
    /// Java `enabledForegroundProperty`.
    enabled_foreground_property: String,
    /// Java `disabledForegroundProperty` (only read by the constructor, as in Java).
    #[allow(dead_code)]
    disabled_foreground_property: String,
    /// Java `enabledForeground`.
    enabled_foreground: Color,
    /// Java `disabledForeground`.
    disabled_foreground: Color,
}

impl ColorTool {
    /// Java private `ColorTool(String, String)`.
    fn new(enabled_foreground_property: &str, disabled_foreground_property: &str) -> ColorTool {
        let mut color_tool = ColorTool {
            enabled_foreground_property: enabled_foreground_property.to_owned(),
            disabled_foreground_property: disabled_foreground_property.to_owned(),
            // Assigned below, as the Java does after the two properties.
            enabled_foreground: (0, 0, 0),
            disabled_foreground: (0, 0, 0),
        };
        color_tool.enabled_foreground = color_tool.get_color(Some(enabled_foreground_property));
        color_tool.disabled_foreground = color_tool.get_color(Some(disabled_foreground_property));
        color_tool
    }

    /// Java static `getButtonInstance(boolean)`.
    pub fn get_button_instance(toggle_button: bool) -> &'static ColorTool {
        if toggle_button { &TOGGLE_BUTTON } else { &BUTTON }
    }

    /// Java private `getColor(String)`.
    fn get_color(&self, property: Option<&str>) -> Color {
        // Color color = UIManager.getColor(property): look-and-feel defaults are not
        // modelled, so the lookup returns null.
        let color: Option<Color> = None;
        if let Some(color) = color {
            return color;
        }
        if property.is_none_or(|property| property == self.enabled_foreground_property) {
            return (51, 51, 51);
        }
        (153, 153, 153)
    }

    /// Java `getForeground(boolean)`.
    pub fn get_foreground(&self, enabled: bool) -> Color {
        if enabled { self.enabled_foreground } else { self.disabled_foreground }
    }
}

/// Java `@Override toString()`.
impl std::fmt::Display for ColorTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if std::ptr::eq(self, &*TOGGLE_BUTTON) {
            return f.write_str("TOGGLE_BUTTON");
        }
        if std::ptr::eq(self, &*BUTTON) {
            return f.write_str("BUTTON");
        }
        // super.toString(): Object's class name and identity hash.
        write!(f, "etomo.ui.swing.ColorTool@{:x}", self as *const ColorTool as usize)
    }
}
