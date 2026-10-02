//! `IMOD/Etomo/src/etomo/type/XTiltOption.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton; the singletons are also associated constants
//! under their Java names.
#![allow(dead_code)]

/// Java `XTiltOption`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XTiltOption {
    /// Java `FIX = new XTiltOption(0)`.
    Fix,
    /// Java `AUTOMAP_SAME = new XTiltOption(4)`.
    AutomapSame,
}

impl XTiltOption {
    /// Java `FIX`.
    pub const FIX: XTiltOption = XTiltOption::Fix;
    /// Java `AUTOMAP_SAME`.
    pub const AUTOMAP_SAME: XTiltOption = XTiltOption::AutomapSame;

    /// Java field `option`.
    fn option(self) -> i32 {
        match self {
            Self::Fix => 0,
            Self::AutomapSame => 4,
        }
    }

    /// Java `getInstance(int)`.
    pub fn get_instance(option: i32) -> Option<XTiltOption> {
        if Self::FIX.option() == option {
            return Some(Self::FIX);
        }
        if Self::AUTOMAP_SAME.option() == option {
            return Some(Self::AUTOMAP_SAME);
        }
        None
    }

    /// Java `getOption`.
    pub fn get_option(self) -> i32 {
        self.option()
    }
}
