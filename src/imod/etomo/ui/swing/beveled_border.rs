//! `IMOD/Etomo/src/etomo/ui/swing/BeveledBorder.java`.
//!
//! `BorderFactory` and border painting are Swing operations.  The state passed
//! to that boundary is retained here, including the source's initial etched
//! border before `TitledBorder.setBorder` replaces it with the lowered bevel.
#![allow(dead_code)]

use std::sync::LazyLock;

use crate::imod::etomo::ui::swing::ui_utilities::Color;
use crate::imod::etomo::util::utilities::APRIL_FOOLS;

/// Java `BevelBorder.LOWERED`.
pub const LOWERED: i32 = 1;

/// Java static `highlightOuter`.
pub static HIGHLIGHT_OUTER: LazyLock<Color> = LazyLock::new(|| {
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

/// Java static `shadowOuter`.
pub static SHADOW_OUTER: LazyLock<Color> = LazyLock::new(|| {
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

/// Java static `shadowInner`.
pub static SHADOW_INNER: LazyLock<Color> = LazyLock::new(|| {
    if !*APRIL_FOOLS {
        Color {
            red: 84,
            green: 86,
            blue: 95,
        }
    } else {
        Color {
            red: 103,
            green: 101,
            blue: 204,
        }
    }
});

/// Java `BorderFactory.createEtchedBorder(Color, Color)` result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EtchedBorderState {
    pub highlight: Color,
    pub shadow: Color,
}

/// Java `BorderFactory.createBevelBorder(int, Color, Color, Color, Color)` result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BevelBorderState {
    pub bevel_type: i32,
    pub highlight_outer: Color,
    pub highlight_inner: Color,
    pub shadow_outer: Color,
    pub shadow_inner: Color,
}

/// Java `javax.swing.border.Border`, restricted to the two factory results
/// constructed by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Border {
    Etched(EtchedBorderState),
    Bevel(BevelBorderState),
}

/// Java `TitledBorder` state delivered to the Swing border-painting boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TitledBorder {
    pub border: Border,
    pub title: String,
}

/// Java package-private `BeveledBorder` fields and methods.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BeveledBorder {
    pub border: Border,
    pub titled_border: TitledBorder,
}

impl BeveledBorder {
    /// Java `BeveledBorder(String)`.
    pub fn new(title: &str) -> Self {
        let mut titled_border = TitledBorder {
            border: Border::Etched(EtchedBorderState {
                highlight: *HIGHLIGHT_OUTER,
                shadow: *SHADOW_OUTER,
            }),
            title: title.into(),
        };

        let border = Border::Bevel(BevelBorderState {
            bevel_type: LOWERED,
            highlight_outer: *HIGHLIGHT_OUTER,
            highlight_inner: Color {
                red: 255,
                green: 255,
                blue: 255,
            },
            shadow_outer: *SHADOW_OUTER,
            shadow_inner: *SHADOW_OUTER,
        });
        titled_border.border = border;

        Self {
            border,
            titled_border,
        }
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
    fn constructor_replaces_the_etched_border_with_the_source_lowered_bevel() {
        let border = BeveledBorder::new("Fine alignment");
        let expected = Border::Bevel(BevelBorderState {
            bevel_type: LOWERED,
            highlight_outer: *HIGHLIGHT_OUTER,
            highlight_inner: Color {
                red: 255,
                green: 255,
                blue: 255,
            },
            shadow_outer: *SHADOW_OUTER,
            shadow_inner: *SHADOW_OUTER,
        });

        assert_eq!(border.border, expected);
        assert_eq!(border.titled_border.border, expected);
    }

    #[test]
    fn get_border_returns_the_title_and_replaced_border() {
        let border = BeveledBorder::new("Patch tracking");

        assert_eq!(border.get_border().title, "Patch tracking");
        assert_eq!(border.get_border().border, border.border);
    }

    #[test]
    fn static_colors_follow_the_source_april_fools_selection() {
        let expected = if *APRIL_FOOLS {
            (
                Color {
                    red: 255,
                    green: 231,
                    blue: 205,
                },
                Color {
                    red: 138,
                    green: 152,
                    blue: 219,
                },
                Color {
                    red: 103,
                    green: 101,
                    blue: 204,
                },
            )
        } else {
            (
                Color {
                    red: 248,
                    green: 254,
                    blue: 255,
                },
                Color {
                    red: 121,
                    green: 124,
                    blue: 136,
                },
                Color {
                    red: 84,
                    green: 86,
                    blue: 95,
                },
            )
        };

        assert_eq!((*HIGHLIGHT_OUTER, *SHADOW_OUTER, *SHADOW_INNER), expected);
    }
}
