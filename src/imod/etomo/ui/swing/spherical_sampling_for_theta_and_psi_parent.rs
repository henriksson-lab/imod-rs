//! `IMOD/Etomo/src/etomo/ui/swing/SphericalSamplingForThetaAndPsiParent.java`.
#![allow(dead_code)]

/// Java package-private `SphericalSamplingForThetaAndPsiParent`.
pub trait SphericalSamplingForThetaAndPsiParent {
    /// Java `updateDisplay(boolean)`.
    fn update_display(&mut self, init: bool);
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Parent(bool);
    impl SphericalSamplingForThetaAndPsiParent for Parent {
        fn update_display(&mut self, init: bool) {
            self.0 = init;
        }
    }
    #[test]
    fn update_display_preserves_init_argument() {
        let mut parent = Parent(false);
        parent.update_display(true);
        assert!(parent.0);
    }
}
