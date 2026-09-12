//! `IMOD/Etomo/src/etomo/type/MatchMode.java`.

#![allow(dead_code)]

/// Java's two identity singleton instances, represented as Rust enum values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchMode {
    BToA,
    AToB,
}

impl MatchMode {
    /// Java `toString()`.
    pub fn to_string(self) -> &'static str {
        match self {
            Self::BToA => "B_TO_A",
            Self::AToB => "A_TO_B",
        }
    }

    /// Java `getInstance(String)`.
    pub fn get_instance_string(string: Option<&str>) -> Option<Self> {
        let string = string?;
        if string.eq_ignore_ascii_case("B_TO_A") {
            Some(Self::BToA)
        } else if string.eq_ignore_ascii_case("A_TO_B") {
            Some(Self::AToB)
        } else {
            None
        }
    }

    /// Java `getInstance(boolean)`.
    pub fn get_instance_match_b_to_a(match_b_to_a: bool) -> Self {
        if match_b_to_a { Self::BToA } else { Self::AToB }
    }
}

#[cfg(test)]
mod tests {
    use super::MatchMode;

    #[test]
    fn source_identity_values_round_trip_through_both_factories() {
        assert_eq!(MatchMode::BToA.to_string(), "B_TO_A");
        assert_eq!(MatchMode::AToB.to_string(), "A_TO_B");
        assert_eq!(
            MatchMode::get_instance_string(Some("b_to_a")),
            Some(MatchMode::BToA)
        );
        assert_eq!(MatchMode::get_instance_string(Some("other")), None);
        assert_eq!(MatchMode::get_instance_string(None), None);
        assert_eq!(MatchMode::get_instance_match_b_to_a(false), MatchMode::AToB);
    }
}
