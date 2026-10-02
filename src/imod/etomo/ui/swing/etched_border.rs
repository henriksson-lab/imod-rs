//! `IMOD/Etomo/src/etomo/ui/swing/EtchedBorder.java`.
//!
//! A titled etched border.  Only the title is modelled (the stand-in's
//! `TitledBorder`); the etched border and its colours are painting.

use std::rc::Rc;
use std::sync::LazyLock;

use crate::imod::etomo::jdk::{Color, TitledBorder};
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

// TODO these should be gotten from the app some how
/// Java private static `highlight`.
static HIGHLIGHT: LazyLock<Color> = LazyLock::new(|| {
    if !*utilities::APRIL_FOOLS {
        (248, 254, 255)
    } else {
        (255, 231, 205)
    }
});
/// Java private static `shadow`.
static SHADOW: LazyLock<Color> = LazyLock::new(|| {
    if !*utilities::APRIL_FOOLS {
        (121, 124, 136)
    } else {
        (138, 152, 219)
    }
});

/// Java `EtchedBorder`.
pub struct EtchedBorder {
    /// Java `titledBorder`.
    titled_border: Rc<TitledBorder>,
}

impl EtchedBorder {
    /// Java `EtchedBorder(String)`.
    pub fn new(title: Option<&str>) -> EtchedBorder {
        // Swing painting: drawn with BorderFactory.createEtchedBorder(highlight, shadow).
        let _ = (*HIGHLIGHT, *SHADOW);
        EtchedBorder {
            titled_border: Rc::new(TitledBorder::new(title)),
        }
    }

    /// Java `setTitle(String)`.
    pub fn set_title(&self, title: Option<&str>) {
        self.titled_border.set_title(title);
    }

    /// Java `getTitle()`.
    pub fn get_title(&self) -> Option<String> {
        self.titled_border.get_title()
    }

    /// Java `getBorder()`.
    pub fn get_border(&self) -> Rc<TitledBorder> {
        self.titled_border.clone()
    }
}
