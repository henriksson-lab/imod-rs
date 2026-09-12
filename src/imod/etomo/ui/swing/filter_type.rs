//! `IMOD/Etomo/src/etomo/ui/swing/FilterType.java`.
#![allow(dead_code)]

/// Java package-private `FilterType`.
pub trait FilterType {
    /// Java `isRadialFilter()`.
    fn is_radial_filter(&self) -> bool;

    /// Java `isHighFrequencyFilter()`.
    fn is_high_frequency_filter(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Filter;

    impl FilterType for Filter {
        fn is_radial_filter(&self) -> bool {
            true
        }

        fn is_high_frequency_filter(&self) -> bool {
            false
        }
    }

    #[test]
    fn interface_retains_independent_filter_queries() {
        let filter = Filter;
        assert!(filter.is_radial_filter());
        assert!(!filter.is_high_frequency_filter());
    }
}
