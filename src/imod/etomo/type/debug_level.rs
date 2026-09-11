//! `IMOD/Etomo/src/etomo/type/DebugLevel.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! distinct `public static final` singleton; `LOW`, `HIGH` and `DEFAULT_DEBUG` are
//! aliases of existing singletons and are translated as associated constants.
#![allow(dead_code)]

/// Java `DebugLevel`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DebugLevel {
    /// Java `OFF` (value 0).  No debug.
    Off,
    /// Java `LIMITED` (value -1).  No debug except for limited level debugging.
    Limited,
    /// Java `STANDARD` (value 1).  Same as debug without an argument.
    Standard,
    /// Java `EXTRA` (value 2).
    Extra,
    /// Java `VERBOSE` (value 3).
    Verbose,
    /// Java `EXTRA_VERBOSE` (value 4).
    ExtraVerbose,
}

impl DebugLevel {
    /// Java `LOW`.
    pub const LOW: DebugLevel = DebugLevel::Standard;
    /// Java `HIGH`.
    pub const HIGH: DebugLevel = DebugLevel::Extra;
    /// Java `DEFAULT_DEBUG`.
    pub const DEFAULT_DEBUG: DebugLevel = DebugLevel::Standard;

    /// Java field `value`, set by the private `DebugLevel(int)` constructor.
    fn value(self) -> i32 {
        match self {
            Self::Off => 0,
            Self::Limited => -1,
            Self::Standard => 1,
            Self::Extra => 2,
            Self::Verbose => 3,
            Self::ExtraVerbose => 4,
        }
    }

    /// Java `getInstance`.
    pub fn get_instance(s_value: &str) -> DebugLevel {
        // Java: Integer.parseInt(sValue), catching NumberFormatException.
        if let Ok(value) = s_value.parse::<i32>() {
            if value == Self::Off.value() {
                return Self::Off;
            }
            if value == Self::Limited.value() {
                return Self::Limited;
            }
            if value == Self::Standard.value() {
                return Self::Standard;
            }
            if value == Self::Extra.value() {
                return Self::Extra;
            }
            if value == Self::Verbose.value() {
                return Self::Verbose;
            }
            if value == Self::ExtraVerbose.value() {
                return Self::ExtraVerbose;
            }
        }
        Self::DEFAULT_DEBUG
    }

    /// Java `getOffInstance`.
    pub fn get_off_instance() -> DebugLevel {
        Self::Off
    }

    /// Java `getValue`.
    pub fn get_value(self) -> i32 {
        self.value()
    }

    /// Java `isLimited`.
    pub fn is_limited(self) -> bool {
        self == Self::Limited
    }

    /// Java `isOn`.
    pub fn is_on(self) -> bool {
        self != Self::Off && self != Self::Limited
    }

    /// Java `isExtra`.
    pub fn is_extra(self) -> bool {
        self == Self::Extra || self == Self::Verbose || self == Self::ExtraVerbose
    }

    /// Java `isVerbose`.
    pub fn is_verbose(self) -> bool {
        self == Self::Verbose || self == Self::ExtraVerbose
    }

    /// Java `isExtraVerbose`.
    pub fn is_extra_verbose(self) -> bool {
        self == Self::ExtraVerbose
    }

    /// Java `ge`.
    pub fn ge(self, debug_level: DebugLevel) -> bool {
        self.value() >= debug_level.value()
    }
}

/// Java `toString`.
impl std::fmt::Display for DebugLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == Self::Off {
            return f.write_str("off");
        }
        if *self == Self::Limited {
            return f.write_str("limited");
        }
        if *self == Self::Standard {
            return f.write_str("standard");
        }
        if *self == Self::Extra {
            return f.write_str("extra");
        }
        if *self == Self::Verbose {
            return f.write_str("verbose");
        }
        if *self == Self::ExtraVerbose {
            return f.write_str("extraVerbose");
        }
        f.write_str("unknown")
    }
}
