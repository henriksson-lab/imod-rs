//! `IMOD/Etomo/src/etomo/comscript/SplitcombineParam.java`.

use crate::imod::etomo::etomo_director;

/// Java `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "splitcombine";

/// Java `SplitcombineParam`.
#[derive(Clone, Debug, Default)]
pub struct SplitcombineParam {
    /// Java `commandArray`, null until built.
    command_array: Option<Vec<String>>,
}

impl SplitcombineParam {
    /// Java's implicit `SplitcombineParam()`.
    pub fn new() -> SplitcombineParam {
        SplitcombineParam {
            command_array: None,
        }
    }

    /// Java final `getCommand`.
    pub fn get_command(&mut self) -> Vec<String> {
        if self.command_array.is_none() {
            self.build_command();
        }
        self.command_array.clone().unwrap_or_default()
    }

    /// Java private final `buildCommand`.
    fn build_command(&mut self) {
        let mut command: Vec<String> = Vec::new();
        command.push("python".to_owned());
        command.push("-u".to_owned());
        // Java string concatenation writes a null path as "null".
        let python_script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command.push(format!("{python_script_path}{COMMAND_NAME}"));
        command.push("volcombine".to_owned());
        let command_size = command.len();
        let mut command_array = Vec::with_capacity(command_size);
        for i in 0..command_size {
            command_array.push(command[i].clone());
        }
        self.command_array = Some(command_array);
    }
}
