//! `IMOD/Etomo/src/etomo/ui/swing/Minibutton.java`.
//!
//! Java's `JButton` painting and hit testing are a native Swing boundary.  The
//! source-owned round/square state, colour selection, dimensions, and last
//! paint requests are retained here for a frontend to execute faithfully.
#![allow(dead_code)]

use super::input_cell::InputCellComponent;
use super::panel::Dimension;
use super::ui_utilities::{Color, Icon, UiUtilities};

/// Java package-private final `Minibutton`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Minibutton {
    pub text: Option<String>,
    pub icon: Option<Icon>,
    pub disabled_icon: Option<Icon>,
    pub pressed_icon: Option<Icon>,
    pub round: bool,
    pub rollover_color: Option<Color>,
    pub pressed_color: Option<Color>,
    pub outline_color: Option<Color>,
    pub foreground: Color,
    pub background: Color,
    pub border: Option<&'static str>,
    pub content_area_filled: bool,
    pub focusable: bool,
    pub preferred_size: Dimension,
    pub maximum_size: Dimension,
    pub width: i32,
    pub height: i32,
    pub enabled: bool,
    pub name: Option<String>,
    pub action_command: Option<String>,
    pub tooltip: Option<String>,
    pub action_listener_count: usize,
    pub mouse_listener_count: usize,
    pub armed: bool,
    pub rollover: bool,
    pub last_component_paint_color: Option<Color>,
    pub last_border_paint_color: Option<Color>,
    pub border_antialias: bool,
    pub border_render_quality: bool,
}

impl InputCellComponent for Minibutton {
    fn set_background(&mut self, color: Color) {
        self.background = color;
    }

    fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Minibutton {
    /// Java private `Minibutton(String, Icon)`.
    pub fn new(text: Option<&str>, icon: Option<Icon>) -> Self {
        Self {
            text: text.map(str::to_owned),
            icon,
            disabled_icon: None,
            pressed_icon: None,
            round: false,
            rollover_color: None,
            pressed_color: None,
            outline_color: None,
            foreground: Color {
                red: 0,
                green: 0,
                blue: 0,
            },
            background: Color {
                red: 238,
                green: 238,
                blue: 238,
            },
            border: None,
            content_area_filled: true,
            focusable: false,
            preferred_size: Dimension::default(),
            maximum_size: Dimension::default(),
            width: 0,
            height: 0,
            enabled: true,
            name: None,
            action_command: text.map(str::to_owned),
            tooltip: None,
            action_listener_count: 0,
            mouse_listener_count: 0,
            armed: false,
            rollover: false,
            last_component_paint_color: None,
            last_border_paint_color: None,
            border_antialias: false,
            border_render_quality: false,
        }
    }

    /// Java private `Minibutton(String, boolean, Color, Color, Color, Color)`.
    pub fn new_round(
        text: Option<&str>,
        round: bool,
        color: Option<Color>,
        rollover_color: Option<Color>,
        pressed_color: Option<Color>,
        outline_color: Option<Color>,
    ) -> Self {
        let mut value = Self::new(text, None);
        value.round = round;
        if round {
            value.outline_color = Some(outline_color.unwrap_or(value.foreground));
            if let Some(color) = color {
                value.background = color;
            }
            let color = value.background;
            value.rollover_color = Some(rollover_color.unwrap_or(Color {
                red: (color.red * 10 / 7).min(255),
                green: (color.green * 10 / 7).min(255),
                blue: (color.blue * 10 / 7).min(255),
            }));
            value.pressed_color = Some(pressed_color.unwrap_or(Color {
                red: color.red * 7 / 10,
                green: color.green * 7 / 10,
                blue: color.blue * 7 / 10,
            }));
            value.content_area_filled = false;
        }
        value
    }

    /// Java static `getSquareInstance(String, Border)`.
    pub fn get_square_instance(label: Option<&str>, border: Option<&'static str>) -> Self {
        let mut value = Self::new(label, None);
        value.border = border;
        value.set_size();
        value
    }

    /// Java static `getSquareInstance(Icon, Border)`.
    pub fn get_square_icon_instance(icon: Option<Icon>, border: Option<&'static str>) -> Self {
        let mut value = Self::new(None, icon);
        value.border = border;
        value.set_size();
        value
    }

    /// Java static `getSquareInstance(Border)`.
    pub fn get_square_empty_instance(border: Option<&'static str>) -> Self {
        let mut value = Self::new(None, None);
        value.border = border;
        value.set_size();
        value
    }

    /// Java static `getRoundInstance(String, boolean, boolean, Color, Color, Color, Color)`.
    pub fn get_round_instance(
        label: Option<&str>,
        italics: bool,
        small: bool,
        color: Option<Color>,
        rollover_color: Option<Color>,
        pressed_color: Option<Color>,
        outline_color: Option<Color>,
    ) -> Self {
        let label_size = label.map_or(1, str::len);
        let text = format!(
            "{}{}{}{}",
            if italics { "<html><i>" } else { "" },
            if label_size > 1 { " " } else { "" },
            label.unwrap_or(""),
            if label_size > 1 { " " } else { "" }
        );
        let mut value = Self::new_round(
            Some(&text),
            true,
            color,
            rollover_color,
            pressed_color,
            outline_color,
        );
        value.border = Some(if small { "EmptyBorder" } else { "EtchedBorder" });
        value.set_size();
        value
    }

    /// Java static `getBlueInstance(String, boolean, boolean)`.
    pub fn get_blue_instance(label: Option<&str>, italics: bool, small: bool) -> Self {
        Self::get_round_instance(
            label,
            italics,
            small,
            Some(Color {
                red: 176,
                green: 248,
                blue: 255,
            }),
            Some(Color {
                red: 203,
                green: 232,
                blue: 255,
            }),
            Some(Color {
                red: 134,
                green: 189,
                blue: 255,
            }),
            Some(Color {
                red: 15,
                green: 4,
                blue: 75,
            }),
        )
    }

    /// Java static `getGreenInstance(String, boolean, boolean)`.
    pub fn get_green_instance(label: Option<&str>, italics: bool, small: bool) -> Self {
        Self::get_round_instance(
            label,
            italics,
            small,
            Some(Color {
                red: 188,
                green: 254,
                blue: 186,
            }),
            Some(Color {
                red: 231,
                green: 254,
                blue: 245,
            }),
            Some(Color {
                red: 86,
                green: 226,
                blue: 138,
            }),
            Some(Color {
                red: 0,
                green: 40,
                blue: 2,
            }),
        )
    }

    /// Java `setIcon(Image)`.
    pub fn set_image_icon(&mut self, image: Option<Icon>) {
        if let Some(image) = image {
            self.icon = Some(image);
        }
    }

    /// Java `setSize()`.
    pub fn set_size(&mut self) {
        let mut size = UiUtilities::get_preferred_size(
            &super::ui_utilities::AbstractButton {
                icon: self.icon,
                ..Default::default()
            },
            self.text.as_deref(),
        );
        if size.width < size.height {
            size.width = size.height;
        }
        self.set_size_dimension(size);
    }

    /// Java overridden `setSize(Dimension)`.
    pub fn set_size_dimension(&mut self, size: Dimension) {
        self.preferred_size = size;
        self.maximum_size = size;
        self.width = size.width;
        self.height = size.height;
    }

    /// Java overridden `paintComponent(Graphics)`.
    pub fn paint_component(&mut self) {
        if self.round {
            self.last_component_paint_color = Some(if self.armed {
                self.pressed_color.unwrap_or(Color {
                    red: self.background.red * 7 / 10,
                    green: self.background.green * 7 / 10,
                    blue: self.background.blue * 7 / 10,
                })
            } else if self.rollover {
                self.rollover_color.unwrap_or(Color {
                    red: (self.background.red * 10 / 7).min(255),
                    green: (self.background.green * 10 / 7).min(255),
                    blue: (self.background.blue * 10 / 7).min(255),
                })
            } else {
                self.background
            });
        }
    }

    /// Java overridden `paintBorder(Graphics)`.
    pub fn paint_border(&mut self) {
        if self.round {
            self.border_antialias = true;
            self.border_render_quality = true;
            self.last_border_paint_color = Some(self.outline_color.unwrap_or(self.foreground));
        }
    }

    /// Java overridden `contains(int, int)`.
    pub fn contains(&self, x: i32, y: i32) -> bool {
        if !self.round {
            return x >= 0 && y >= 0 && x < self.width && y < self.height;
        }
        if self.width <= 0 || self.height <= 0 {
            return false;
        }
        let horizontal = (2 * x - self.width) as i64;
        let vertical = (2 * y - self.height) as i64;
        horizontal * horizontal * (self.height as i64).pow(2)
            + vertical * vertical * (self.width as i64).pow(2)
            <= (self.width as i64).pow(2) * (self.height as i64).pow(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_factories_make_the_preferred_dimension_square() {
        let button = Minibutton::get_square_icon_instance(
            Some(Icon {
                width: 7,
                height: 19,
            }),
            Some("BevelBorder.RAISED"),
        );
        assert!(!button.round);
        assert_eq!(button.preferred_size.width, button.preferred_size.height);
        assert!(!button.focusable);
    }

    #[test]
    fn round_buttons_preserve_color_model_painting_and_ellipse_hit_testing() {
        let mut button = Minibutton::get_blue_instance(Some("A"), false, true);
        button.set_size_dimension(Dimension {
            width: 20,
            height: 20,
        });
        button.rollover = true;
        button.paint_component();
        button.paint_border();
        assert_eq!(
            button.last_component_paint_color,
            Some(Color {
                red: 203,
                green: 232,
                blue: 255
            })
        );
        assert_eq!(
            button.last_border_paint_color,
            Some(Color {
                red: 15,
                green: 4,
                blue: 75
            })
        );
        assert!(button.contains(10, 10));
        assert!(!button.contains(0, 0));
    }
}
