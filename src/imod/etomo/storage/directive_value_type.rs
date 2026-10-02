//! `IMOD/Etomo/src/etomo/storage/DirectiveValueType.java`.
//!
//! Java's typesafe-enum pattern (a private constructor plus `public static final`
//! singletons) is mirrored as a Rust enum with one variant per singleton; Java's
//! identity comparisons become variant matches.
#![allow(dead_code)]

/// Java `DirectiveValueType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DirectiveValueType {
    /// Java `BOOLEAN`, constructed with `("Bool", 3)`.
    Boolean,
    /// Java `FLOATING_POINT`, constructed with `("Float", 3)`.
    FloatingPoint,
    /// Java `FLOATING_POINT_PAIR`, constructed with `("2 Float", 6)`.
    FloatingPointPair,
    /// Java `INTEGER`, constructed with `("Int", 3)`.
    Integer,
    /// Java `INTEGER_PAIR`, constructed with `("2 Int", 6)`.
    IntegerPair,
    /// Java `LIST`, constructed with `("List", 9)`.
    List,
    /// Java `STRING`, constructed with `("String", 9)`.
    String,
    /// Java `UNKNOWN`, constructed with `("Unknown", 15)`.
    Unknown,
    /// Java `FILE`, constructed with `("File", 15)`.
    File,
}

impl DirectiveValueType {
    /// Java `rcsid`.
    pub const RCSID: &'static str = "$Id:$";

    /// Java private final field `tag`.
    fn tag(self) -> &'static str {
        match self {
            Self::Boolean => "Bool",
            Self::FloatingPoint => "Float",
            Self::FloatingPointPair => "2 Float",
            Self::Integer => "Int",
            Self::IntegerPair => "2 Int",
            Self::List => "List",
            Self::String => "String",
            Self::Unknown => "Unknown",
            Self::File => "File",
        }
    }

    /// Java private final field `columns`.
    fn columns(self) -> i32 {
        match self {
            Self::Boolean => 3,
            Self::FloatingPoint => 3,
            Self::FloatingPointPair => 6,
            Self::Integer => 3,
            Self::IntegerPair => 6,
            Self::List => 9,
            Self::String => 9,
            Self::Unknown => 15,
            Self::File => 15,
        }
    }

    /// Java package-private static `getInstance(String)`.
    pub(crate) fn get_instance(input: Option<&str>) -> DirectiveValueType {
        let input = match input {
            None => return Self::Unknown,
            Some(input) => input,
        };
        if input == Self::Boolean.tag() {
            return Self::Boolean;
        }
        if input == Self::FloatingPoint.tag() {
            return Self::FloatingPoint;
        }
        if input == Self::FloatingPointPair.tag() {
            return Self::FloatingPointPair;
        }
        if input == Self::Integer.tag() {
            return Self::Integer;
        }
        if input == Self::IntegerPair.tag() {
            return Self::IntegerPair;
        }
        if input == Self::List.tag() {
            return Self::List;
        }
        if input == Self::String.tag() {
            return Self::String;
        }
        if input == Self::File.tag() {
            return Self::File;
        }
        Self::Unknown
    }

    /// Java `getColumns`.
    pub fn get_columns(self) -> i32 {
        self.columns()
    }
}

/// Java `toString`.
impl std::fmt::Display for DirectiveValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.tag())
    }
}
