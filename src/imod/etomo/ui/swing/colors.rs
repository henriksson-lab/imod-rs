//! `IMOD/Etomo/src/etomo/ui/swing/Colors.java`.
//!
//! `java.awt.Color` and `javax.swing.plaf.ColorUIResource` are both the RGB triple
//! [`Color`] of the Swing stand-in (`ColorUIResource` only marks a colour as belonging
//! to the look and feel, which is painting).  The lazily built colours keep Java's
//! build-once caches.

use std::sync::OnceLock;

use crate::imod::etomo::jdk::Color;
use crate::imod::etomo::util::utilities;

/// Java `ColorUIResource`: the same RGB triple as `Color`.
pub type ColorUIResource = Color;

/// Java `CELL_FOREGROUND`.
pub const CELL_FOREGROUND: ColorUIResource = (0, 0, 0);
/// Java `CELL_NOT_IN_USE_FOREGROUND`.
pub const CELL_NOT_IN_USE_FOREGROUND: ColorUIResource = (102, 102, 102);
/// Java `CELL_ERROR_BACKGROUND`.
pub const CELL_ERROR_BACKGROUND: ColorUIResource = (255, 204, 204);
/// Java `CELL_ERROR_BACKGROUND_NOT_EDITABLE`.
pub const CELL_ERROR_BACKGROUND_NOT_EDITABLE: ColorUIResource = (230, 184, 184); // 223,179,179?
/// Java `BACKGROUND`.
pub const BACKGROUND: ColorUIResource = (255, 255, 255);
/// Java `WARNING_BACKGROUND`.
pub const WARNING_BACKGROUND: ColorUIResource = (255, 255, 204);
/// Java `WARNING_BACKGROUND_NOT_EDITABLE`.
pub const WARNING_BACKGROUND_NOT_EDITABLE: ColorUIResource = (230, 230, 184);
/// Java `HIGHLIGHT_BACKGROUND`.
pub const HIGHLIGHT_BACKGROUND: ColorUIResource = (204, 255, 255);
/// Java `HIGHLIGHT_BACKGROUND_NOT_EDITABLE`.
pub const HIGHLIGHT_BACKGROUND_NOT_EDITABLE: ColorUIResource = (184, 230, 230);
/// Java `RUN_HIGHLIGHT_BACKGROUND`.
pub const RUN_HIGHLIGHT_BACKGROUND: ColorUIResource = (204, 255, 204);
/// Java `RUN_HIGHLIGHT_BACKGROUND_NOT_EDITABLE`.
pub const RUN_HIGHLIGHT_BACKGROUND_NOT_EDITABLE: ColorUIResource = (184, 230, 184);
/// Java `VIOLET`.
pub const VIOLET: ColorUIResource = (199, 173, 224);

/// Java `FOREGROUND`.
pub const FOREGROUND: ColorUIResource = (0, 0, 0);

/// Java `BACKGROUND_GREYOUT`.
pub const BACKGROUND_GREYOUT: ColorUIResource = (25, 25, 25);
/// Java `CELL_DISABLED_FOREGROUND`.
pub const CELL_DISABLED_FOREGROUND: ColorUIResource = (120, 120, 120);
/// Java `AVAILABLE_BACKGROUND`.
pub const AVAILABLE_BACKGROUND: Color = (224, 240, 255);
/// Java `AVAILABLE_BORDER`.
pub const AVAILABLE_BORDER: Color = (153, 204, 255);
/// Java `FIELD_HIGHLIGHT`.
pub const FIELD_HIGHLIGHT: Color = (0, 0, 185);
/// Java private `BACKGROUND_ADJUSTMENT` (unused in the Java too).
#[allow(dead_code)]
const BACKGROUND_ADJUSTMENT: i32 = 20;
/// Java `HEADER_BACKGROUND`.
pub const HEADER_BACKGROUND: ColorUIResource = (239, 239, 239);

/// Java private static `backgroundA` (built on first use, then kept).
static BACKGROUND_A: OnceLock<Color> = OnceLock::new();
/// Java private static `backgroundB`.
static BACKGROUND_B: OnceLock<Color> = OnceLock::new();
/// Java private static `backgroundJoin`.
static BACKGROUND_JOIN: OnceLock<Color> = OnceLock::new();
/// Java private static `backgroundParallel`.
static BACKGROUND_PARALLEL: OnceLock<Color> = OnceLock::new();
/// Java private static `backgroundBatchruntomo`.
static BACKGROUND_BATCHRUNTOMO: OnceLock<Color> = OnceLock::new();
/// Java private static `backgroundSerialSections`.
static BACKGROUND_SERIAL_SECTIONS: OnceLock<Color> = OnceLock::new();
/// Java private static `backgroundTools`.
static BACKGROUND_TOOLS: OnceLock<Color> = OnceLock::new();
/// Java private static `cellNotEditableBackground`.
static CELL_NOT_EDITABLE_BACKGROUND: OnceLock<ColorUIResource> = OnceLock::new();

/// Java `getBackgroundA()`.
pub fn get_background_a() -> Color {
    *BACKGROUND_A.get_or_init(|| {
        if !*utilities::APRIL_FOOLS {
            (173, 199, 224) // saphire
        } else {
            (163, 214, 247)
        }
    })
}

/// Java `getBackgroundB()`.
pub fn get_background_b() -> Color {
    *BACKGROUND_B.get_or_init(|| {
        if !*utilities::APRIL_FOOLS {
            (173, 224, 199) // jade
        } else {
            (255, 216, 141)
        }
    })
}

/// Java `getBackgroundJoin()`.
pub fn get_background_join() -> Color {
    *BACKGROUND_JOIN.get_or_init(|| {
        if !*utilities::APRIL_FOOLS {
            VIOLET
        } else {
            (162, 167, 255)
        }
    })
}

/// Java `getBackgroundParallel()`.
pub fn get_background_parallel() -> Color {
    *BACKGROUND_PARALLEL.get_or_init(|| {
        if !*utilities::APRIL_FOOLS {
            (186, 224, 173) // lime
        } else {
            (255, 253, 216)
        }
    })
}

/// Java `getBackgroundBatchruntomo()`.
pub fn get_background_batchruntomo() -> Color {
    *BACKGROUND_BATCHRUNTOMO.get_or_init(|| {
        if !*utilities::APRIL_FOOLS {
            VIOLET
        } else {
            (255, 239, 192)
        }
    })
}

/// Java `getBackgroundSerialSections()`.
pub fn get_background_serial_sections() -> Color {
    *BACKGROUND_SERIAL_SECTIONS.get_or_init(|| {
        if !*utilities::APRIL_FOOLS {
            (218, 232, 250)
        } else {
            (194, 247, 159)
        }
    })
}

/// Java `getBackgroundTools()`.
pub fn get_background_tools() -> Color {
    *BACKGROUND_TOOLS.get_or_init(|| {
        if !*utilities::APRIL_FOOLS {
            (173, 212, 224) // azure
        } else {
            (52, 130, 218)
        }
    })
}

/// Java `getCellNotEditableBackground()`.
pub fn get_cell_not_editable_background() -> ColorUIResource {
    *CELL_NOT_EDITABLE_BACKGROUND.get_or_init(|| subtract_color(BACKGROUND, BACKGROUND_GREYOUT))
}

/// Java `subtractColor(Color, Color)`.  Java's `ColorUIResource(int, int, int)` throws
/// `IllegalArgumentException` for a component outside 0..=255; the checked arithmetic
/// raises the same failure (no caller reaches it: every argument pair is a constant).
pub fn subtract_color(color: Color, subtract_color: Color) -> ColorUIResource {
    (
        color
            .0
            .checked_sub(subtract_color.0)
            .expect("Color parameter outside of expected range: Red"),
        color
            .1
            .checked_sub(subtract_color.1)
            .expect("Color parameter outside of expected range: Green"),
        color
            .2
            .checked_sub(subtract_color.2)
            .expect("Color parameter outside of expected range: Blue"),
    )
}

/// Java private `addColor(Color, Color)` (no caller in the Java either).
#[allow(dead_code)]
fn add_color(color: Color, subtract_color: Color) -> ColorUIResource {
    (
        color
            .0
            .checked_add(subtract_color.0)
            .expect("Color parameter outside of expected range: Red"),
        color
            .1
            .checked_add(subtract_color.1)
            .expect("Color parameter outside of expected range: Green"),
        color
            .2
            .checked_add(subtract_color.2)
            .expect("Color parameter outside of expected range: Blue"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_not_editable_background_is_background_less_greyout() {
        assert_eq!(get_cell_not_editable_background(), (230, 230, 230));
    }
}
