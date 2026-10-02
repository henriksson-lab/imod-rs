//! `IMOD/Etomo/src/etomo/comscript/PythonInfoParam.java`.

/// Java `PythonInfoParam`.
#[derive(Clone, Debug, Default)]
pub struct PythonInfoParam {
    greater_then_version: Option<i32>,
    command_array: Option<Vec<String>>,
}

impl PythonInfoParam {
    /// Java `PythonInfoParam()`.
    pub fn new() -> PythonInfoParam {
        PythonInfoParam::default()
    }

    /// Java `setGreaterThenVersion`.
    pub fn set_greater_then_version(&mut self, greater_then_version: i32) {
        self.greater_then_version = Some(greater_then_version);
    }

    /// Java `getCommandArray`.  The script is wrapped in double quotes, as
    /// the Java writes it, so Python evaluates a string literal and prints
    /// nothing; `isPython3` then reads no "True" and answers false, which is
    /// also the answer off Cygwin, the only platform the test distinguishes.
    pub fn get_command_array(&mut self) -> Option<Vec<String>> {
        if self.command_array.is_none()
            && let Some(greater_then_version) = self.greater_then_version
        {
            self.command_array = Some(vec![
                "python".to_owned(),
                "-c".to_owned(),
                format!(
                    "\"import sys ;result = sys.platform == 'cygwin' and sys.version_info[0] > {greater_then_version};sys.stdout.write(str(result) + '\\n')\""
                ),
            ]);
        }
        self.command_array.clone()
    }
}
