//! `IMOD/Etomo/src/etomo/ui/swing/EtchedBorder.java`.
#![allow(dead_code)]

use super::beveled_border::{Border, EtchedBorderState, TitledBorder};
use super::ui_utilities::Color;
use crate::imod::etomo::util::utilities::APRIL_FOOLS;
use std::sync::LazyLock;

/// Java `EtchedBorder.rcsid`.
pub const RCSID: &str = "$Id$";
/// Java static `highlight`.
pub static HIGHLIGHT: LazyLock<Color> = LazyLock::new(|| {
    if !*APRIL_FOOLS {
        Color {
            red: 248,
            green: 254,
            blue: 255,
        }
    } else {
        Color {
            red: 255,
            green: 231,
            blue: 205,
        }
    }
});
/// Java static `shadow`.
pub static SHADOW: LazyLock<Color> = LazyLock::new(|| {
    if !*APRIL_FOOLS {
        Color {
            red: 121,
            green: 124,
            blue: 136,
        }
    } else {
        Color {
            red: 138,
            green: 152,
            blue: 219,
        }
    }
});

/// Java public `EtchedBorder`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EtchedBorder {
    pub titled_border: TitledBorder,
}
impl EtchedBorder {
    /// Java package-private `EtchedBorder(String)`.
    pub fn new(title: &str) -> Self {
        Self {
            titled_border: TitledBorder {
                border: Border::Etched(EtchedBorderState {
                    highlight: *HIGHLIGHT,
                    shadow: *SHADOW,
                }),
                title: title.to_owned(),
            },
        }
    }
    /// Java `setTitle(String)`.
    pub fn set_title(&mut self, title: &str) {
        self.titled_border.title = title.to_owned();
    }
    /// Java `getTitle()`.
    pub fn get_title(&self) -> &str {
        &self.titled_border.title
    }
    /// Java `getBorder()`.
    pub fn get_border(&self) -> &TitledBorder {
        &self.titled_border
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_keeps_source_colors_and_title() {
        let border = EtchedBorder::new("Patch tracking");
        assert_eq!(border.get_title(), "Patch tracking");
        assert_eq!(
            border.get_border().border,
            Border::Etched(EtchedBorderState {
                highlight: *HIGHLIGHT,
                shadow: *SHADOW
            })
        );
    }
    #[test]
    fn set_title_updates_titled_border() {
        let mut border = EtchedBorder::new("Old");
        border.set_title("New");
        assert_eq!(border.get_border().title, "New");
    }
}
