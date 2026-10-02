//! `IMOD/Etomo/src/etomo/logic/ClusteredPointsAllowed.java`.
//!
//! For backwards compatibility with previous version which encorporated elongated.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `static final` singleton; `CLUSTERED` is also an associated constant.
#![allow(dead_code)]

use crate::imod::etomo::r#type::const_etomo_number::Number;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `ClusteredPointsAllowed`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClusteredPointsAllowed {
    /// Java `CLUSTERED = new ClusteredPointsAllowed(1)`.
    Clustered,
    /// Java private `ELONGATED_ONE_THIRD = new ClusteredPointsAllowed(2)` (deprecated -
    /// no longer valid values).
    ElongatedOneThird,
    /// Java private `ELONGATED_TWO_THIRDS = new ClusteredPointsAllowed(3)` (deprecated -
    /// no longer valid values).
    ElongatedTwoThirds,
    /// Java private `ELONGATED_ALL = new ClusteredPointsAllowed(4)` (deprecated - no
    /// longer valid values).
    ElongatedAll,
}

impl ClusteredPointsAllowed {
    /// Java `CLUSTERED`.
    pub const CLUSTERED: ClusteredPointsAllowed = ClusteredPointsAllowed::Clustered;
    /// Java private `ELONGATED_ONE_THIRD`.
    const ELONGATED_ONE_THIRD: ClusteredPointsAllowed = ClusteredPointsAllowed::ElongatedOneThird;
    /// Java private `ELONGATED_TWO_THIRDS`.
    const ELONGATED_TWO_THIRDS: ClusteredPointsAllowed = ClusteredPointsAllowed::ElongatedTwoThirds;
    /// Java private `ELONGATED_ALL`.
    const ELONGATED_ALL: ClusteredPointsAllowed = ClusteredPointsAllowed::ElongatedAll;

    /// Java field `value`.
    fn value(self) -> i32 {
        match self {
            Self::Clustered => 1,
            Self::ElongatedOneThird => 2,
            Self::ElongatedTwoThirds => 3,
            Self::ElongatedAll => 4,
        }
    }

    /// Java `getInstance(int)`.
    pub fn get_instance(value: i32) -> Option<ClusteredPointsAllowed> {
        if value == Self::CLUSTERED.value() {
            return Some(Self::CLUSTERED);
        }
        if value == Self::ELONGATED_ONE_THIRD.value() {
            return Some(Self::ELONGATED_ONE_THIRD);
        }
        if value == Self::ELONGATED_TWO_THIRDS.value() {
            return Some(Self::ELONGATED_TWO_THIRDS);
        }
        if value == Self::ELONGATED_ALL.value() {
            return Some(Self::ELONGATED_ALL);
        }
        None
    }

    /// Java `getInstanceFromDisplayValue(Number)` (deprecated).
    pub fn get_instance_from_display_value(
        display_value: Option<Number>,
    ) -> Option<ClusteredPointsAllowed> {
        let display_value = display_value?;
        Self::get_instance(display_value.int_value().wrapping_add(1))
    }

    /// Java `isElongated`.
    pub fn is_elongated(self) -> bool {
        self != Self::CLUSTERED
    }

    /// Java `getValue`.
    pub fn get_value(self) -> i32 {
        self.value()
    }

    /// Java `convertToDisplayValue`: for backwards compatibility.
    pub fn convert_to_display_value(self) -> i32 {
        self.value() - 1
    }
}

/// Java `toString`: `Integer.toString(value)`.
impl std::fmt::Display for ClusteredPointsAllowed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}
