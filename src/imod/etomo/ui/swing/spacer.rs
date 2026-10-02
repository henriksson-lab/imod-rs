//! `IMOD/Etomo/src/etomo/ui/swing/Spacer.java`.
//!
//! A rigid area of a fixed size that remembers its preferred width.

use std::rc::Rc;

use crate::imod::etomo::jdk::{Dimension, JComponent};

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java package-private `final class Spacer`.
pub struct Spacer {
    /// Java final `rigidArea` (`Box.createRigidArea(dimension)`; its size is layout).
    rigid_area: Rc<JComponent>,
    /// Java final `preferredWidth`.
    preferred_width: i32,
}

impl Spacer {
    /// Java `Spacer(Dimension)`.
    pub fn new(dimension: Dimension) -> Spacer {
        Spacer {
            // Swing layout: Box.createRigidArea(dimension).
            rigid_area: JComponent::new_other(),
            preferred_width: dimension.width,
        }
    }

    /// Java `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        self.preferred_width
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.rigid_area.clone()
    }
}
