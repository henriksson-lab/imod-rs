//! `IMOD/Etomo/src/etomo/ui/swing/Newstack3dFindParent.java`.
#![allow(dead_code)]

use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;

/// Java `Newstack3dFindParent.rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `Newstack3dFindParent`.
pub trait Newstack3dFindParent {
    /// Java `getUnbinnedBeadPixels()`.
    fn get_unbinned_bead_pixels(&self) -> ConstEtomoNumber;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
    struct Parent;
    impl Newstack3dFindParent for Parent {
        fn get_unbinned_bead_pixels(&self) -> ConstEtomoNumber {
            EtomoNumber::new().base
        }
    }
    #[test]
    fn source_contract_returns_const_etomo_number() {
        assert!(Parent.get_unbinned_bead_pixels().is_null());
    }
}
