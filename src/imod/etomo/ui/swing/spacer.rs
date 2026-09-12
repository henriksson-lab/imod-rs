//! `IMOD/Etomo/src/etomo/ui/swing/Spacer.java`.
#![allow(dead_code)]

use super::panel::Dimension;

/// Java package-private final `Spacer` and its native `Box` rigid area.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Spacer {
    pub rigid_area: Dimension,
    pub preferred_width: i32,
}

impl Spacer {
    /// Java `Spacer(Dimension)`.
    pub fn new(dimension: Dimension) -> Self {
        Self {
            rigid_area: dimension,
            preferred_width: dimension.width,
        }
    }
    /// Java `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        self.preferred_width
    }
    /// Java `getComponent()` at the native `Component` boundary.
    pub fn get_component(&self) -> Dimension {
        self.rigid_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_width_even_for_zero_height_rigid_areas() {
        let spacer = Spacer::new(Dimension {
            width: 5,
            height: 0,
        });
        assert_eq!(spacer.get_preferred_width(), 5);
        assert_eq!(
            spacer.get_component(),
            Dimension {
                width: 5,
                height: 0
            }
        );
    }
}
