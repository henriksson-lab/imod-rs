//! `IMOD/Etomo/src/etomo/comscript/FortranInputStringList.java`.
//!
//! A collection of FortranInputStrings.  Useful when success com script entries
//! accumulate.  Should handle FortranInputString of different lengths and types.
//!
//! Copyright: Copyright (c) 2005
//!
//! Organization: Boulder Laboratory for 3-Dimensional Electron Microscopy of Cells
//! (BL3DEM), University of Colorado

use super::com_script_command::ComScriptCommand;
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `FortranInputStringList`.
pub struct FortranInputStringList {
    /// Java package-private field `key`, initialised to null.
    pub key: Option<String>,
    /// Java package-private field `array`, initialised to null.
    pub array: Option<Vec<FortranInputString>>,
}

impl FortranInputStringList {
    /// Java `FortranInputStringList(String)`.
    pub fn new(key: Option<&str>) -> FortranInputStringList {
        FortranInputStringList {
            key: key.map(|key| key.to_string()),
            array: None,
        }
    }

    /// Java `parse(ComScriptCommand)`.
    ///
    /// Fixed in translation: the source sets `array = null` and then assigns
    /// `array[i]` without ever allocating it, so any keyword with a value throws
    /// `NullPointerException`.  The evident intent is an array of `values.length`
    /// elements, which is what this allocates (`BUGS.md`).
    pub fn parse(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), FortranInputSyntaxException> {
        self.array = None;
        let values = script_command.get_values(self.key.as_deref());
        if values.is_empty() {
            return Ok(());
        }
        let mut array: Vec<FortranInputString> = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            array.push(FortranInputString::get_instance_from_list(
                values[i].as_deref(),
            )?);
        }
        self.array = Some(array);
        Ok(())
    }

    /// Java `getDouble`.
    pub fn get_double(&self) -> Option<Vec<f64>> {
        let array = match &self.array {
            None => return None,
            Some(array) => array,
        };
        let mut total_elements: i32 = 0;
        for i in 0..array.len() {
            total_elements += array[i].size();
        }
        let mut double_list = vec![0.0f64; total_elements as usize];
        let mut current_index = 0usize;
        for i in 0..array.len() {
            for j in 0..array[i].size() {
                double_list[current_index] = array[i].get_double_index(j);
                current_index += 1;
            }
        }
        Some(double_list)
    }
}
