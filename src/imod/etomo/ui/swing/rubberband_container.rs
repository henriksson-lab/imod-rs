//! `IMOD/Etomo/src/etomo/ui/swing/RubberbandContainer.java`.

/// Java package-private `RubberbandContainer`.
pub trait RubberbandContainer {
    /// Java `setRubberbandContainerZMin(String)`.
    fn set_rubberband_container_z_min(&mut self, z_min: &str);
    /// Java `setRubberbandContainerZMax(String)`.
    fn set_rubberband_container_z_max(&mut self, z_max: &str);
}

#[cfg(test)]
mod tests {
    use super::RubberbandContainer;
    #[derive(Default)]
    struct Container {
        min: String,
        max: String,
    }
    impl RubberbandContainer for Container {
        fn set_rubberband_container_z_min(&mut self, v: &str) {
            self.min = v.into()
        }
        fn set_rubberband_container_z_max(&mut self, v: &str) {
            self.max = v.into()
        }
    }
    #[test]
    fn z_coordinate_calls_keep_both_strings() {
        let mut c = Container::default();
        c.set_rubberband_container_z_min("2");
        c.set_rubberband_container_z_max("9");
        assert_eq!((c.min, c.max), ("2".into(), "9".into()));
    }
}
