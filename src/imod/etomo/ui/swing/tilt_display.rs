//! `IMOD/Etomo/src/etomo/ui/swing/TiltDisplay.java`.
#![allow(dead_code)]

use std::io;

use super::process_display::ProcessDisplay;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;

/// Checked failures declared by Java `TiltDisplay.getParameters(TiltParam,boolean)`.
#[derive(Debug)]
pub enum TiltDisplayException {
    NumberFormat(String),
    InvalidParameter(InvalidParameterException),
    Io(io::Error),
}
impl std::fmt::Display for TiltDisplayException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NumberFormat(value) => f.write_str(value),
            Self::InvalidParameter(value) => value.fmt(f),
            Self::Io(value) => value.fmt(f),
        }
    }
}
impl std::error::Error for TiltDisplayException {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidParameter(value) => Some(value),
            Self::Io(value) => Some(value),
            Self::NumberFormat(_) => None,
        }
    }
}

/// Java `TiltDisplay`.
pub trait TiltDisplay: ProcessDisplay {
    type TiltParam;
    type SplittiltParam;
    fn get_parameters(
        &self,
        param: &mut Self::TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException>;
    fn get_splittilt_parameters(
        &self,
        param: &mut Self::SplittiltParam,
        do_validation: bool,
    ) -> bool;
    fn allow_tilt_com_save(&self) -> bool;
    fn set_debug(&mut self, debug: bool);
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Display;
    impl ProcessDisplay for Display {}
    impl TiltDisplay for Display {
        type TiltParam = String;
        type SplittiltParam = String;
        fn get_parameters(
            &self,
            p: &mut String,
            validation: bool,
        ) -> Result<bool, TiltDisplayException> {
            *p = "tilt".into();
            Ok(validation)
        }
        fn get_splittilt_parameters(&self, p: &mut String, validation: bool) -> bool {
            *p = "split".into();
            validation
        }
        fn allow_tilt_com_save(&self) -> bool {
            true
        }
        fn set_debug(&mut self, _: bool) {}
    }
    #[test]
    fn both_source_overloads_are_distinct() {
        let mut tilt = String::new();
        let mut split = String::new();
        assert!(Display.get_parameters(&mut tilt, true).unwrap());
        assert!(Display.get_splittilt_parameters(&mut split, true));
        assert_eq!((tilt, split), ("tilt".into(), "split".into()));
    }
}
