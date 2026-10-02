//! `IMOD/Etomo/src/etomo/ui/swing/FilterType.java`.

/// Java package-private `interface FilterType`.
pub trait FilterType {
    /// Java `isRadialFilter()`.
    fn is_radial_filter(&self) -> bool;

    /// Java `isHighFrequencyFilter()`.
    fn is_high_frequency_filter(&self) -> bool;
}
