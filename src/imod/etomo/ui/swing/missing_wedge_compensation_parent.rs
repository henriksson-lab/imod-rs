//! `IMOD/Etomo/src/etomo/ui/swing/MissingWedgeCompensationParent.java`.
#![allow(dead_code)]

/// Java package-private `MissingWedgeCompensationParent`.
pub trait MissingWedgeCompensationParent {
    /// Java `isVolumeTableEmpty()`.
    fn is_volume_table_empty(&self) -> bool;
    /// Java `isReferenceParticleSelected()`.
    fn is_reference_particle_selected(&self) -> bool;
    /// Java `updateDisplay(boolean)`.
    fn update_display(&mut self, init: bool);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Parent;
    impl MissingWedgeCompensationParent for Parent {
        fn is_volume_table_empty(&self) -> bool {
            true
        }
        fn is_reference_particle_selected(&self) -> bool {
            false
        }
        fn update_display(&mut self, _: bool) {}
    }

    #[test]
    fn source_contract_is_implementable() {
        let mut parent = Parent;
        assert!(parent.is_volume_table_empty());
        assert!(!parent.is_reference_particle_selected());
        parent.update_display(false);
    }
}
