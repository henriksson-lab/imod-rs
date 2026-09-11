//! `IMOD/Etomo/src/etomo/type/ViewType.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton.
//!
//! Deviation: the Java class implements `EnumeratedType`, whose `getValue()` returns a
//! `ConstEtomoNumber`.  `EnumeratedType` itself is not translated, so the methods it
//! declares are inherent methods here.  The `index` field is the source's `EtomoNumber`;
//! because a Rust enum variant carries no per-instance storage, `index()` builds the
//! field's value on demand and `get_value` returns it by value rather than by reference.
#![allow(dead_code)]

use super::const_etomo_number::ConstEtomoNumber;
use super::etomo_number::EtomoNumber;

/// Java `ViewType`.  View type definitions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewType {
    /// Java `SINGLE_VIEW`, constructed with title "Single View", paramValue "single",
    /// index 0.
    SingleView,
    /// Java `MONTAGE`, constructed with title "Montage", paramValue "montage",
    /// index 1.
    Montage,
}

impl ViewType {
    /// Java `DEFAULT`.
    pub const DEFAULT: ViewType = ViewType::SingleView;

    /// Java field `title`.
    fn title(self) -> &'static str {
        match self {
            Self::SingleView => "Single View",
            Self::Montage => "Montage",
        }
    }

    /// Java field `paramValue`.
    fn param_value(self) -> &'static str {
        match self {
            Self::SingleView => "single",
            Self::Montage => "montage",
        }
    }

    /// Java field `index`, a `private final EtomoNumber` the constructor fills in with
    /// `this.index.set(index)`.
    fn index(self) -> EtomoNumber {
        let mut index = EtomoNumber::new();
        index.set_int(match self {
            Self::SingleView => 0,
            Self::Montage => 1,
        });
        index
    }

    /// Java `getParamValue`.
    pub fn get_param_value(self) -> &'static str {
        self.param_value()
    }

    /// Java `getValue`.
    pub fn get_value(self) -> ConstEtomoNumber {
        self.index().base
    }

    /// Java `isDefault`.
    pub fn is_default(self) -> bool {
        self == Self::DEFAULT
    }

    /// Java `getLabel`.
    pub fn get_label(self) -> Option<&'static str> {
        None
    }

    /// Java `fromString`.  Takes a string representation of an ViewType type and
    /// returns the correct static object.  The string is case insensitive.  Null is
    /// returned if the string is not one of the possibilities from `toString()` or
    /// `getParamValue()`.
    pub fn from_string(name: &str) -> Option<ViewType> {
        if name.eq_ignore_ascii_case(&Self::SingleView.to_string()) {
            return Some(Self::SingleView);
        }
        if name.eq_ignore_ascii_case(&Self::Montage.to_string()) {
            return Some(Self::Montage);
        }
        if name.eq_ignore_ascii_case(Self::SingleView.get_param_value()) {
            return Some(Self::SingleView);
        }
        if name.eq_ignore_ascii_case(Self::Montage.get_param_value()) {
            return Some(Self::Montage);
        }
        None
    }

    /// Java `getInstance`.  The parameter is declared `EnumeratedType` in the source
    /// and compared by identity against the two singletons; with `EnumeratedType`
    /// untranslated the parameter is narrowed to `ViewType` here.
    pub fn get_instance(enumerated_type: ViewType) -> ViewType {
        if enumerated_type == Self::SingleView {
            return Self::SingleView;
        }
        if enumerated_type == Self::Montage {
            return Self::Montage;
        }
        Self::DEFAULT
    }
}

/// Java `toString`.  Returns a string representation of the object.
impl std::fmt::Display for ViewType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.title())
    }
}
