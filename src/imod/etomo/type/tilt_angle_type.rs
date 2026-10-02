//! `IMOD/Etomo/src/etomo/type/TiltAngleType.java`.
//!
//! A typesafe enum (private constructors, four `public static final` singletons compared
//! by identity), mirrored as a Rust enum.

/// Java `TiltAngleType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TiltAngleType {
    /// Java `EXTRACT = new TiltAngleType("Extract", "Extract tilt angles from raw stack
    /// header or mdoc file")`.
    Extract,
    /// Java `RANGE = new TiltAngleType("Range", "Specify the starting angle and step
    /// (degrees)")`.
    Range,
    /// Java `FILE = new TiltAngleType("File", "Tilt angles in existing rawtlt file")`.
    File,
    /// Java `LIST = new TiltAngleType("List")`.
    List,
}

impl TiltAngleType {
    /// Java field `name`.
    fn name(self) -> &'static str {
        match self {
            Self::Extract => "Extract",
            Self::Range => "Range",
            Self::File => "File",
            Self::List => "List",
        }
    }

    /// Java field `descr`; null for `LIST`, which uses the one-argument constructor.
    fn descr(self) -> Option<&'static str> {
        match self {
            Self::Extract => Some("Extract tilt angles from raw stack header or mdoc file"),
            Self::Range => Some("Specify the starting angle and step (degrees)"),
            Self::File => Some("Tilt angles in existing rawtlt file"),
            Self::List => None,
        }
    }

    /// Java `getDescr`.
    pub fn get_descr(self) -> &'static str {
        match self.descr() {
            None => self.name(),
            Some(descr) => descr,
        }
    }

    /// Java `fromString`.  `compareToIgnoreCase(...) == 0` is case-insensitive
    /// equality; anything else returns null (`None`), as the source's TODO notes.
    pub fn from_string(name: &str) -> Option<TiltAngleType> {
        if name.eq_ignore_ascii_case(&Self::Extract.to_string()) {
            return Some(Self::Extract);
        }
        if name.eq_ignore_ascii_case(&Self::Range.to_string()) {
            return Some(Self::Range);
        }
        if name.eq_ignore_ascii_case(&Self::File.to_string()) {
            return Some(Self::File);
        }
        if name.eq_ignore_ascii_case(&Self::List.to_string()) {
            return Some(Self::List);
        }
        // TODO Don't return null throw an exception for bad arguments
        None
    }

    /// Java `parseInt`.
    pub fn parse_int(type_spec: i32) -> Option<TiltAngleType> {
        if type_spec == -1 {
            return Some(Self::List);
        }
        if type_spec == 0 {
            return Some(Self::File);
        }
        if type_spec == 1 {
            return Some(Self::Range);
        }
        None
    }

    /// Java `toInt`.
    pub fn to_int(self) -> i32 {
        if self.name() == "List" {
            return -1;
        }
        if self.name() == "File" {
            return 0;
        }
        if self.name() == "Range" {
            return 1;
        }
        -1000
    }
}

/// Java `toString`.
impl std::fmt::Display for TiltAngleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_mappings() {
        assert_eq!(
            TiltAngleType::from_string("range"),
            Some(TiltAngleType::Range)
        );
        assert_eq!(TiltAngleType::from_string("bogus"), None);
        assert_eq!(TiltAngleType::parse_int(-1), Some(TiltAngleType::List));
        assert_eq!(TiltAngleType::parse_int(2), None);
        assert_eq!(TiltAngleType::Extract.to_int(), -1000);
        assert_eq!(TiltAngleType::File.to_int(), 0);
        assert_eq!(TiltAngleType::List.get_descr(), "List");
        assert_eq!(
            TiltAngleType::Range.get_descr(),
            "Specify the starting angle and step (degrees)"
        );
    }
}
