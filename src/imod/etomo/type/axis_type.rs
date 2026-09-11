//! `IMOD/Etomo/src/etomo/type/AxisType.java`.
//!
//! Java's typesafe-enum pattern (a private constructor plus `public static final`
//! singletons) is mirrored as a Rust enum with one variant per singleton; Java's
//! identity comparisons (`this == SINGLE_AXIS`) become variant matches.
#![allow(dead_code)]

/// Java `AxisType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisType {
    /// Java `SINGLE_AXIS`.
    SingleAxis,
    /// Java `DUAL_AXIS`.
    DualAxis,
    /// Java `NOT_SET`.
    NotSet,
}

impl AxisType {
    /// Java `rcsid`.
    pub const RCSID: &'static str = "$Id$";

    /// Java field `name`, set by the private `AxisType(String)` constructor.
    pub fn name(self) -> &'static str {
        match self {
            Self::SingleAxis => "Single Axis",
            Self::DualAxis => "Dual Axis",
            Self::NotSet => "Not Set",
        }
    }

    /// Java `getInstance`.  Searches line for the name member variable.  Uses
    /// `indexOf` - not `equals`.
    pub fn get_instance(line: Option<&str>) -> Option<AxisType> {
        let line = match line {
            None => return None,
            Some(line) => line,
        };
        if line.find(Self::SingleAxis.name()).is_some() {
            return Some(Self::SingleAxis);
        }
        if line.find(Self::DualAxis.name()).is_some() {
            return Some(Self::DualAxis);
        }
        if line.find(Self::NotSet.name()).is_some() {
            return Some(Self::NotSet);
        }
        None
    }

    /// Java `fromString`.  Takes a string representation of an AxisType type and
    /// returns the correct static object.  The string is case insensitive.  Null is
    /// returned if the string is not one of the possibilities from `toString()`.
    pub fn from_string(name: &str) -> Option<AxisType> {
        if name.eq_ignore_ascii_case(&Self::SingleAxis.to_string()) {
            return Some(Self::SingleAxis);
        }
        if name.eq_ignore_ascii_case(&Self::DualAxis.to_string()) {
            return Some(Self::DualAxis);
        }
        if name.eq_ignore_ascii_case(&Self::NotSet.to_string()) {
            return Some(Self::NotSet);
        }
        if name.eq_ignore_ascii_case(Self::SingleAxis.get_value()) {
            return Some(Self::SingleAxis);
        }
        if name.eq_ignore_ascii_case(Self::DualAxis.get_value()) {
            return Some(Self::DualAxis);
        }
        if name.eq_ignore_ascii_case(Self::NotSet.get_value()) {
            return Some(Self::NotSet);
        }
        None
    }

    /// Java `getValue`.
    pub fn get_value(self) -> &'static str {
        if self == Self::SingleAxis {
            return "single";
        }
        if self == Self::DualAxis {
            return "dual";
        }
        ""
    }
}

/// Java `toString`.  Returns a string representation of the object.
impl std::fmt::Display for AxisType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
