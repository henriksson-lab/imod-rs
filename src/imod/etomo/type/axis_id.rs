//! `IMOD/Etomo/src/etomo/type/AxisID.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton; Java's identity comparisons (`this == ONLY`)
//! become variant matches.
#![allow(dead_code)]

/// Java `ONLY_EXT_STRING`.
const ONLY_EXT_STRING: &str = "";
/// Java `FIRST_EXT_STRING`.
const FIRST_EXT_STRING: &str = "a";
/// Java `SECOND_EXT_STRING`.
const SECOND_EXT_STRING: &str = "b";

/// Java `ONLY_KEY`.
const ONLY_KEY: &str = "Only";
/// Java `FIRST_KEY`.
const FIRST_KEY: &str = "First";
/// Java `SECOND_KEY`.
const SECOND_KEY: &str = "Second";

/// Java `AxisID`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisID {
    /// Java `ONLY`, constructed with `ONLY_KEY` and axisOfExtension 0.
    Only,
    /// Java `FIRST`, constructed with `FIRST_KEY` and axisOfExtension 1.
    First,
    /// Java `SECOND`, constructed with `SECOND_KEY` and axisOfExtension 2.
    Second,
}

impl AxisID {
    /// Java field `key`, set by the private `AxisID(String, int)` constructor.
    pub fn key(self) -> &'static str {
        match self {
            Self::Only => ONLY_KEY,
            Self::First => FIRST_KEY,
            Self::Second => SECOND_KEY,
        }
    }

    /// Java field `axisOfExtension`: batchruntomo parameter based on the axis ID.
    fn axis_of_extension_field(self) -> i32 {
        match self {
            Self::Only => 0,
            Self::First => 1,
            Self::Second => 2,
        }
    }

    /// Java `dumpState`.
    pub fn dump_state(self) {
        eprintln!("[key:{}]", self.key());
    }

    /// Java `getAxisOfExtension`.
    pub fn get_axis_of_extension(self) -> i32 {
        self.axis_of_extension_field()
    }

    /// Java `getKey`.
    pub fn get_key(self) -> &'static str {
        self.key()
    }

    /// Java `getInstanceFromFileName`.
    pub fn get_instance_from_file_name(
        axis_type: Option<super::axis_type::AxisType>,
        file_name: Option<&str>,
    ) -> AxisID {
        if axis_type == Some(super::axis_type::AxisType::SingleAxis) {
            return AxisID::Only;
        }
        let file_name = match file_name {
            // Assume A axis if not information
            None => return AxisID::First,
            Some(file_name) => file_name,
        };
        // Strip off the extension.
        let left_side = crate::imod::etomo::util::utilities::remove_extension(Some(file_name));
        if let Some(left_side) = left_side {
            if left_side.ends_with(SECOND_EXT_STRING) {
                return AxisID::Second;
            }
        }
        AxisID::First
    }

    /// Java `getInstance` (String overload).
    pub fn get_instance(extension: Option<&str>) -> Option<AxisID> {
        let extension = match extension {
            None => return None,
            Some(extension) => extension,
        };
        if extension == ONLY_EXT_STRING {
            return Some(Self::Only);
        }
        if extension == FIRST_EXT_STRING {
            return Some(Self::First);
        }
        if extension == SECOND_EXT_STRING {
            return Some(Self::Second);
        }
        None
    }

    /// Java `getInstanceIgnoreCase`.
    pub fn get_instance_ignore_case(extension: Option<&str>) -> Option<AxisID> {
        let extension = match extension {
            None => return None,
            Some(extension) => extension,
        };
        if extension.eq_ignore_ascii_case(ONLY_EXT_STRING) {
            return Some(Self::Only);
        }
        if extension.eq_ignore_ascii_case(FIRST_EXT_STRING) {
            return Some(Self::First);
        }
        if extension.eq_ignore_ascii_case(SECOND_EXT_STRING) {
            return Some(Self::Second);
        }
        None
    }

    /// Java `getOtherAxisID`.
    pub fn get_other_axis_id(self) -> Option<AxisID> {
        if self == Self::Only {
            return None;
        }
        if self == Self::First {
            return Some(Self::Second);
        }
        if self == Self::Second {
            return Some(Self::First);
        }
        None
    }

    /// Java `isSameAxis`.  Returns true if this instance is the same axis.
    /// `AxisID.SECOND` is the B axis.  Everything else is the A axis (including
    /// null).
    pub fn is_same_axis(self, axis_id: Option<AxisID>) -> bool {
        if self == Self::Second {
            return axis_id == Some(Self::Second);
        }
        axis_id != Some(Self::Second)
    }

    /// Java `getInstance` (char overload).
    pub fn get_instance_from_char(extension: char) -> Option<AxisID> {
        if FIRST_EXT_STRING.chars().next().unwrap() == extension {
            return Some(Self::First);
        }
        if SECOND_EXT_STRING.chars().next().unwrap() == extension {
            return Some(Self::Second);
        }
        None
    }

    /// Java `getExtension`.  Returns the extension associated with the specific
    /// AxisID.  Used for creating file names.
    pub fn get_extension(self) -> String {
        if self == Self::Only {
            return ONLY_EXT_STRING.to_string();
        }
        if self == Self::First {
            return FIRST_EXT_STRING.to_string();
        }
        if self == Self::Second {
            return SECOND_EXT_STRING.to_string();
        }
        // Unreachable: the Java class has exactly the three singletons above.
        "ERROR".to_string()
    }

    /// Java `getUpperCaseExtension`.
    pub fn get_upper_case_extension(self) -> String {
        if self == Self::Only {
            return ONLY_EXT_STRING.to_string();
        }
        if self == Self::First {
            return FIRST_EXT_STRING.to_uppercase();
        }
        if self == Self::Second {
            return SECOND_EXT_STRING.to_uppercase();
        }
        // Unreachable: the Java class has exactly the three singletons above.
        "ERROR".to_string()
    }

    /// Java `getExtensionLength`.
    pub fn get_extension_length() -> i32 {
        1
    }

    /// Java `getCapitalizedKeyString`.  Returns A for only and first, otherwise B.
    pub fn get_capitalized_key_string(self) -> &'static str {
        if self == Self::Only {
            // Generallly only is retrieved the same way as first.
            return "A";
        }
        if self == Self::First {
            return "A";
        }
        if self == Self::Second {
            return "B";
        }
        // Unreachable: the Java class has exactly the three singletons above.
        "ERROR"
    }

    /// Java `getInstanceFromKey`.  Takes a string representation of an AxisID type
    /// and returns the correct static object.  The string is case insensitive.
    /// Null is returned if the string is not one of the possibilities from
    /// `toString()`.
    pub fn get_instance_from_key(key: &str) -> Option<AxisID> {
        if key.eq_ignore_ascii_case(Self::Only.key()) {
            return Some(Self::Only);
        }
        if key.eq_ignore_ascii_case(Self::First.key()) {
            return Some(Self::First);
        }
        if key.eq_ignore_ascii_case(Self::Second.key()) {
            return Some(Self::Second);
        }
        None
    }
}

/// Java `toString`.  Returns a string representation of the object.
impl std::fmt::Display for AxisID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.key())
    }
}
