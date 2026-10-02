//! `IMOD/Etomo/src/etomo/type/DataSource.java`.
//!
//! "This is the equivalent of an enum from C/C++."  The Java typesafe-enum pattern (a
//! private constructor plus two `public static final` singletons) is a Rust enum with
//! one variant per singleton; Java's identity comparisons become variant equality.

/// Java `DataSource`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataSource {
    /// Java `CCD`, constructed with name "CCD".
    Ccd,
    /// Java `FILM`, constructed with name "Film".
    Film,
}

impl DataSource {
    /// Java `rcsid`.
    pub const RCSID: &'static str = "$Id$";

    /// Java field `name`, set by the private `DataSource(String)` constructor.
    fn name(self) -> &'static str {
        match self {
            Self::Ccd => "CCD",
            Self::Film => "Film",
        }
    }

    /// Java `fromString`.  `String.compareToIgnoreCase(...) == 0` is a
    /// case-insensitive equality test.  Returns null (`None`) for anything else.
    pub fn from_string(name: &str) -> Option<DataSource> {
        if name.eq_ignore_ascii_case(&Self::Ccd.to_string()) {
            return Some(Self::Ccd);
        }
        if name.eq_ignore_ascii_case(&Self::Film.to_string()) {
            return Some(Self::Film);
        }
        None
    }
}

/// Java `toString`.
impl std::fmt::Display for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_case_insensitively() {
        assert_eq!(DataSource::Ccd.to_string(), "CCD");
        assert_eq!(DataSource::Film.to_string(), "Film");
        assert_eq!(DataSource::from_string("ccd"), Some(DataSource::Ccd));
        assert_eq!(DataSource::from_string("FILM"), Some(DataSource::Film));
        assert_eq!(DataSource::from_string("tape"), None);
    }
}
