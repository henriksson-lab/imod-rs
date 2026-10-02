//! `IMOD/Etomo/src/etomo/comscript/ConstXfproductParam.java`.

use super::fortran_input_string::FortranInputString;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java class `ConstXfproductParam`.  `XfproductParam` extends it and holds it as its
/// `base`, reaching the package-private fields directly as the Java subclass does.
#[derive(Clone, Debug)]
pub struct ConstXfproductParam {
    /// Java package-private `inputFile1`.  Null when a script gives the keyword with no
    /// value (`ComScriptCommand.getValue`).
    pub(crate) input_file1: Option<String>,
    /// Java package-private `inputFile2`.
    pub(crate) input_file2: Option<String>,
    /// Java package-private `outputFile`.
    pub(crate) output_file: Option<String>,
    /// Java package-private `scaleShifts`.
    pub(crate) scale_shifts: FortranInputString,
}

impl ConstXfproductParam {
    /// Java's implicit `ConstXfproductParam()` with the field initialisers.
    pub fn new() -> ConstXfproductParam {
        ConstXfproductParam {
            input_file1: Some(String::new()),
            input_file2: Some(String::new()),
            output_file: Some(String::new()),
            scale_shifts: FortranInputString::new(2),
        }
    }

    /// Java `getInputFile1`.
    pub fn get_input_file1(&self) -> Option<&str> {
        self.input_file1.as_deref()
    }

    /// Java `getInputFile2`.
    pub fn get_input_file2(&self) -> Option<&str> {
        self.input_file2.as_deref()
    }

    /// Java `getOutputFile`.
    pub fn get_output_file(&self) -> Option<&str> {
        self.output_file.as_deref()
    }

    /// Java `getScaleShifts`.
    pub fn get_scale_shifts(&self) -> String {
        self.scale_shifts.to_string()
    }
}

impl Default for ConstXfproductParam {
    fn default() -> ConstXfproductParam {
        ConstXfproductParam::new()
    }
}
