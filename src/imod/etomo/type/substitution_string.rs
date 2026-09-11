//! `IMOD/Etomo/src/etomo/type/SubstitutionString.java`.
//!
//! Description: String containing variables.
//!
//! Copyright: Copyright 2016 - 2023 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! `java.lang.String.indexOf`, `lastIndexOf` and `substring` are UTF-16 code-unit
//! operations, so the private `substitute` works over a `Vec<u16>` here.
#![allow(dead_code)]

/// Java private `START`.
const START: &str = "%{";
/// Java private `END`.
const END: &str = "}";

/// Java private `START_LEN`, `START.length()`.
const START_LEN: i32 = 2;
/// Java private `END_LEN`, `END.length()`.
const END_LEN: i32 = 1;

/// Java public final `SubstitutionString`.
pub struct SubstitutionString {
    /// Java private final field `string`.
    string: Option<String>,
    /// Java private field `externalVariableName`, initialised to null.
    external_variable_name: Option<String>,
    /// Java private field `variableName`, initialised to null.
    variable_name: Option<String>,
    /// Java private field `variableValue`, initialised to null.
    variable_value: Option<String>,
}

impl SubstitutionString {
    /// Java `SubstitutionString(String, String)`.
    ///
    /// `externalVariableName` - variable name without `%{}` - may not contain "%{" or
    /// "}".
    pub fn new(string: Option<&str>, external_variable_name: Option<&str>) -> SubstitutionString {
        SubstitutionString {
            string: string.map(|string| string.to_string()),
            external_variable_name: external_variable_name
                .map(|external_variable_name| external_variable_name.to_string()),
            variable_name: None,
            variable_value: None,
        }
    }

    /// Java `setVariable(String, String)`.
    ///
    /// `variableName` - variable name without `%{}` - may not contain "%{" or "}".
    /// `value` - no effect if null.
    pub fn set_variable(&mut self, variable_name: Option<&str>, value: Option<&str>) {
        self.variable_name = variable_name.map(|variable_name| variable_name.to_string());
        self.variable_value = value.map(|value| value.to_string());
    }

    /// Java package-private static `substitute(String, String, String)`.
    ///
    /// Substitutes the matching variableValue for every instance of `%{variableName}` in
    /// string.
    pub fn substitute_variable(
        string: Option<&str>,
        variable_name: Option<&str>,
        variable_value: Option<&str>,
    ) -> Option<String> {
        SubstitutionString::substitute_external(string, variable_name, variable_value, None, -1)
    }

    /// Java `substitute(int)`.
    ///
    /// Substitutes the matching value for every instance of `%{name}` in string.
    pub fn substitute(&self, external_variable_value: i32) -> Option<String> {
        SubstitutionString::substitute_external(
            self.string.as_deref(),
            self.variable_name.as_deref(),
            self.variable_value.as_deref(),
            self.external_variable_name.as_deref(),
            external_variable_value,
        )
    }

    /// Java private static `substitute(String, String, String, String, int)`.
    ///
    /// Substitutes the matching value for every instance of `%{name}` in string.
    fn substitute_external(
        string: Option<&str>,
        variable_name: Option<&str>,
        variable_value: Option<&str>,
        external_variable_name: Option<&str>,
        external_variable_value: i32,
    ) -> Option<String> {
        let units: Vec<u16> = match string {
            None => return None,
            Some(string) => string.encode_utf16().collect(),
        };
        let start_units: Vec<u16> = START.encode_utf16().collect();
        let end_units: Vec<u16> = END.encode_utf16().collect();
        // `string.indexOf(START)`; `-1` when absent.
        let index_of_start = units
            .windows(start_units.len())
            .position(|window| window == start_units.as_slice())
            .map_or(-1_i32, |index| index as i32);
        let index_of_end = units
            .windows(end_units.len())
            .position(|window| window == end_units.as_slice())
            .map_or(-1_i32, |index| index as i32);
        if string == Some("") || index_of_start == -1 || index_of_end == -1 {
            return string.map(|string| string.to_string());
        }
        let mut builder: Vec<u16> = Vec::new();
        let mut index: i32 = 0;
        while index < units.len() as i32 {
            // Find the end of the variable
            let end = units[index as usize..]
                .windows(end_units.len())
                .position(|window| window == end_units.as_slice())
                .map_or(-1_i32, |position| index + position as i32);
            if end != -1 {
                // Find the start of the variable, working back from end
                let start = units[..(end as usize + start_units.len()).min(units.len())]
                    .windows(start_units.len())
                    .rposition(|window| window == start_units.as_slice())
                    .map_or(-1_i32, |position| position as i32);
                if start != -1 && start >= index {
                    builder.extend_from_slice(&units[index as usize..start as usize]);
                    let var = String::from_utf16_lossy(
                        &units[(start + START_LEN) as usize..end as usize],
                    );
                    let value = SubstitutionString::get_variable_value(
                        Some(&var),
                        variable_name,
                        variable_value,
                        external_variable_name,
                        external_variable_value,
                    );
                    match value {
                        Some(value) => {
                            // Substitute value of variable
                            builder.extend(value.encode_utf16());
                        }
                        None => {
                            // No matching variable - leave variable in place
                            builder.extend_from_slice(
                                &units[start as usize..(end + END_LEN) as usize],
                            );
                        }
                    }
                    index = end + END_LEN;
                } else {
                    // No %{'s left in string
                    builder.extend_from_slice(&units[index as usize..]);
                    break;
                }
            } else {
                // No }'s left in string
                builder.extend_from_slice(&units[index as usize..]);
                break;
            }
        }
        Some(String::from_utf16_lossy(&builder))
    }

    /// Java private static `getVariableValue(String, String, String, String, int)`.
    ///
    /// Returns the value matching the name corresponding to `testVariableName`.
    fn get_variable_value(
        test_variable_name: Option<&str>,
        variable_name: Option<&str>,
        variable_value: Option<&str>,
        external_variable_name: Option<&str>,
        external_variable_value: i32,
    ) -> Option<String> {
        let test_variable_name = test_variable_name?;
        if Some(test_variable_name) == variable_name {
            return variable_value.map(|variable_value| variable_value.to_string());
        }
        if Some(test_variable_name) == external_variable_name {
            return Some(external_variable_value.to_string());
        }
        None
    }
}

/// Java `toString()`.
impl std::fmt::Display for SubstitutionString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Java returns the field itself, so a null field prints as the null reference.
        f.write_str(self.string.as_deref().unwrap_or("null"))
    }
}
