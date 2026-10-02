//! `IMOD/Etomo/src/etomo/comscript/SplittiltParam.java`.

use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;

/// Java `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "splittilt";

/// Java `SplittiltParam`.
pub struct SplittiltParam {
    /// Built on first use by `getCommand`, which the process manager calls
    /// through a shared reference.
    command_array: std::sync::Mutex<Option<Vec<String>>>,
    num_machines: EtomoNumber,
    axis_id: AxisID,
    /// Prevents direct parallel writing.
    separate_chunks: bool,
    name: Option<String>,
}

impl SplittiltParam {
    /// Java `SplittiltParam(AxisID)`.
    pub fn new(axis_id: AxisID) -> SplittiltParam {
        let mut num_machines = EtomoNumber::new();
        num_machines.set_null_is_valid(false);
        num_machines.set_valid_floor(1);
        SplittiltParam {
            command_array: std::sync::Mutex::new(None),
            num_machines,
            axis_id,
            separate_chunks: false,
            name: Some("tilt".to_owned()),
        }
    }

    /// Java final `getCommand`.
    pub fn get_command(&self) -> Vec<String> {
        if self.command_array.lock().unwrap().is_none() {
            self.build_command();
        }
        self.command_array
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_default()
    }

    /// Java private final `buildCommand`.
    fn build_command(&self) {
        let mut command: Vec<String> = Vec::new();
        command.push("python".to_owned());
        command.push("-u".to_owned());
        // Java string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command.push(format!("{script_path}{COMMAND_NAME}"));
        command.push("-n".to_owned());
        command.push(self.num_machines.to_string());
        if self.separate_chunks {
            command.push("-c".to_owned());
        }
        // Java string concatenation writes a null name as "null"
        command.push(format!(
            "{}{}",
            self.name.as_deref().unwrap_or("null"),
            self.axis_id.get_extension()
        ));
        let command_size = command.len();
        let mut command_array: Vec<String> = Vec::with_capacity(command_size);
        for element in command.into_iter().take(command_size) {
            command_array.push(element);
        }
        *self.command_array.lock().unwrap() = Some(command_array);
    }

    /// Java `setNumMachines(String)`.
    pub fn set_num_machines(&mut self, num_machines: Option<&str>) -> &ConstEtomoNumber {
        self.num_machines.set_string(num_machines);
        &self.num_machines
    }

    /// Java `setSeparateChunks(boolean)`.
    pub fn set_separate_chunks(&mut self, separate_chunks: bool) {
        self.separate_chunks = separate_chunks;
    }

    /// Java `setName(String)`.
    pub fn set_name(&mut self, input: Option<&str>) {
        self.name = input.map(str::to_owned);
    }
}
