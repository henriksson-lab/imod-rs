//! `IMOD/Etomo/src/etomo/comscript/SplitCorrectionParam.java`.

use super::const_split_correction_param::ConstSplitCorrectionParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java final `SplitCorrectionParam`.
pub struct SplitCorrectionParam {
    axis_id: AxisID,
    cpus: EtomoNumber,
    max_z: i32,
    /// Built on first use by `getCommand`, which the process manager calls
    /// through a shared reference.
    command_array: std::sync::Mutex<Option<Vec<String>>>,
}

impl SplitCorrectionParam {
    /// Java `SplitCorrectionParam(AxisID)`.
    pub fn new(axis_id: AxisID) -> SplitCorrectionParam {
        SplitCorrectionParam {
            axis_id,
            cpus: EtomoNumber::new(),
            max_z: 0,
            command_array: std::sync::Mutex::new(None),
        }
    }

    /// Java private `buildCommand`.
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
        command.push(format!("{script_path}{}", ProcessName::SPLIT_CORRECTION));
        if !self.cpus.is_null() && !self.cpus.equals_int(0) {
            command.push("-m".to_owned());
            // target # of chunks
            let target: i32 = 2i32.wrapping_mul(self.cpus.get_int());
            // max slices
            let mut max_slices = EtomoNumber::new();
            max_slices.set_int(
                self.max_z
                    .wrapping_add(target)
                    .wrapping_sub(1)
                    .wrapping_div(target),
            );
            command.push(max_slices.to_string());
        }
        command.push(ProcessName::CTF_CORRECTION.get_comscript(self.axis_id));
        let command_size = command.len();
        let mut command_array: Vec<String> = Vec::with_capacity(command_size);
        for element in command.into_iter().take(command_size) {
            command_array.push(element);
        }
        *self.command_array.lock().unwrap() = Some(command_array);
    }

    /// Java `setCpus(String)`.
    pub fn set_cpus(&mut self, input: Option<&str>) {
        self.cpus.set_string(input);
    }

    /// Java `setMaxZ(int)`.
    pub fn set_max_z(&mut self, input: i32) {
        self.max_z = input;
    }
}

impl ConstSplitCorrectionParam for SplitCorrectionParam {
    /// Java `getCommand`.
    fn get_command(&self) -> Vec<String> {
        if self.command_array.lock().unwrap().is_none() {
            self.build_command();
        }
        self.command_array
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_default()
    }
}
