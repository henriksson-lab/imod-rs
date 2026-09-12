//! `IMOD/Etomo/src/etomo/type/FiducialMatch.java`.

#![allow(dead_code)]

/// Java's identity singleton fiducial matching choices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FiducialMatch {
    NotSet,
    BothSides,
    OneSide,
    OneSideInverted,
    UseModel,
    UseModelOnly,
}

impl FiducialMatch {
    /// Java `getInstance(String)`.
    pub fn get_instance(input: &str) -> Option<Self> {
        match input {
            "Not Set" => Some(Self::NotSet),
            "BothSides" => Some(Self::BothSides),
            "OneSide" => Some(Self::OneSide),
            "OneSideInverted" => Some(Self::OneSideInverted),
            "UseModel" => Some(Self::UseModel),
            "UseModelOnly" => Some(Self::UseModelOnly),
            _ => None,
        }
    }

    /// Java `toString()`.
    pub fn to_string(self) -> &'static str {
        match self {
            Self::NotSet => "Not Set",
            Self::BothSides => "BothSides",
            Self::OneSide => "OneSide",
            Self::OneSideInverted => "OneSideInverted",
            Self::UseModel => "UseModel",
            Self::UseModelOnly => "UseModelOnly",
        }
    }

    /// Java `fromString(String)`.
    pub fn from_string(name: &str) -> Option<Self> {
        [
            Self::BothSides,
            Self::OneSide,
            Self::OneSideInverted,
            Self::UseModel,
            Self::UseModelOnly,
        ]
        .into_iter()
        .find(|value| value.to_string().eq_ignore_ascii_case(name))
    }

    /// Java `getOption()`.
    pub fn get_option(self) -> Option<&'static str> {
        match self {
            Self::NotSet => None,
            Self::BothSides => Some("2"),
            Self::OneSide => Some("1"),
            Self::OneSideInverted => Some("-1"),
            Self::UseModel => Some("0"),
            Self::UseModelOnly => Some("-2"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FiducialMatch;

    #[test]
    fn source_factories_keep_their_different_not_set_handling() {
        assert_eq!(
            FiducialMatch::get_instance("Not Set"),
            Some(FiducialMatch::NotSet)
        );
        assert_eq!(FiducialMatch::from_string("not set"), None);
        assert_eq!(
            FiducialMatch::from_string("usemodelonly"),
            Some(FiducialMatch::UseModelOnly)
        );
        assert_eq!(FiducialMatch::UseModel.get_option(), Some("0"));
    }
}
