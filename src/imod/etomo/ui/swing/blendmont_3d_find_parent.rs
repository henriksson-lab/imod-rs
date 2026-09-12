//! `IMOD/Etomo/src/etomo/ui/swing/Blendmont3dFindParent.java`.
//!
//! This package-private Java interface is a parent boundary.  It owns no Swing
//! state: its sole method hands the caller the source-visible
//! `ConstEtomoNumber` bead-pixel value.  Rust returns an owned clone because
//! Java object-reference values can be retained independently by the caller;
//! `ConstEtomoNumber` is the translated value representation of that object.
#![allow(dead_code)]

use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;

/// Java `Blendmont3dFindParent.rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `Blendmont3dFindParent`.
pub trait Blendmont3dFindParent {
    /// Java `getUnbinnedBeadPixels()`.
    fn get_unbinned_bead_pixels(&self) -> ConstEtomoNumber;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::r#type::const_etomo_number::Type;
    use crate::imod::etomo::r#type::etomo_number::EtomoNumber;

    struct Parent {
        unbinned_bead_pixels: ConstEtomoNumber,
    }

    impl Blendmont3dFindParent for Parent {
        fn get_unbinned_bead_pixels(&self) -> ConstEtomoNumber {
            self.unbinned_bead_pixels.clone()
        }
    }

    #[test]
    fn get_unbinned_bead_pixels_returns_the_parent_value() {
        let mut number = EtomoNumber::new_with_type(Some(Type::Double));
        number.set_string(Some("12.5"));
        let parent = Parent {
            unbinned_bead_pixels: number.base,
        };

        assert_eq!(parent.get_unbinned_bead_pixels().get_double(), 12.5);
    }

    #[test]
    fn rcsid_matches_the_java_source() {
        assert_eq!(RCSID, "$Id$");
    }
}
