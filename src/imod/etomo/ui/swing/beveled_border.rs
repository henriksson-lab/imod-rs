//! `IMOD/Etomo/src/etomo/ui/swing/BeveledBorder.java`.
//!
//! A titled border drawn as a lowered bevel.  Only the title is modelled (the
//! stand-in's `TitledBorder`); the etched and bevel borders and their colours are
//! painting, kept as comments and colour constants.

use std::rc::Rc;
use std::sync::LazyLock;

use crate::imod::etomo::jdk::{Color, TitledBorder};
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java static `highlightOuter`.
pub static HIGHLIGHT_OUTER: LazyLock<Color> = LazyLock::new(|| {
    if !*utilities::APRIL_FOOLS {
        (248, 254, 255)
    } else {
        (255, 231, 205)
    }
});
/// Java static `shadowOuter`.
pub static SHADOW_OUTER: LazyLock<Color> = LazyLock::new(|| {
    if !*utilities::APRIL_FOOLS {
        (121, 124, 136)
    } else {
        (138, 152, 219)
    }
});
/// Java static `shadowInner`.
pub static SHADOW_INNER: LazyLock<Color> = LazyLock::new(|| {
    if !*utilities::APRIL_FOOLS {
        (84, 86, 95)
    } else {
        (103, 101, 204)
    }
});

/// Java `BeveledBorder`.
pub struct BeveledBorder {
    // Java field `border`: BorderFactory.createBevelBorder(BevelBorder.LOWERED,
    // highlightOuter, Color.white, shadowOuter, shadowOuter) - painting, not modelled.
    /// Java field `titledBorder`.
    titled_border: Rc<TitledBorder>,
}

impl BeveledBorder {
    /// Java `BeveledBorder(String)`.
    pub fn new(title: Option<&str>) -> BeveledBorder {
        // Swing painting: the TitledBorder is drawn with
        // BorderFactory.createEtchedBorder(highlightOuter, shadowOuter).
        let titled_border = Rc::new(TitledBorder::new(title));
        // Swing painting: border = BorderFactory.createBevelBorder(BevelBorder.LOWERED,
        // highlightOuter, Color.white, shadowOuter, shadowOuter);
        // titledBorder.setBorder(border).
        BeveledBorder { titled_border }
    }

    /// Java `getBorder()`.
    pub fn get_border(&self) -> Rc<TitledBorder> {
        self.titled_border.clone()
    }
}
