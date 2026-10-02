//! `IMOD/Etomo/src/etomo/type/PosSampleType.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton; the singletons are also associated constants
//! under their Java names.
#![allow(dead_code)]

/// Java `PosSampleType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PosSampleType {
    /// Java `SAMPLES = new PosSampleType(0)`.
    Samples,
    /// Java `WHOLE = new PosSampleType(1)`.
    Whole,
    /// Java `WHOLE_CRYO = new PosSampleType(2)`.
    WholeCryo,
}

impl PosSampleType {
    /// Java `SAMPLES`.
    pub const SAMPLES: PosSampleType = PosSampleType::Samples;
    /// Java `WHOLE`.
    pub const WHOLE: PosSampleType = PosSampleType::Whole;
    /// Java `WHOLE_CRYO`.
    pub const WHOLE_CRYO: PosSampleType = PosSampleType::WholeCryo;

    /// Java field `sharedValue`.
    fn shared_value(self) -> i32 {
        match self {
            Self::Samples => 0,
            Self::Whole => 1,
            Self::WholeCryo => 2,
        }
    }

    /// Java `getInstance(int)`.
    pub fn get_instance(value: i32) -> Option<PosSampleType> {
        if value == Self::SAMPLES.shared_value() {
            return Some(Self::SAMPLES);
        }
        if value == Self::WHOLE.shared_value() {
            return Some(Self::WHOLE);
        }
        if value == Self::WHOLE_CRYO.shared_value() {
            return Some(Self::WHOLE_CRYO);
        }
        None
    }

    /// Java `getValue`.
    pub fn get_value(self) -> i32 {
        self.shared_value()
    }
}
