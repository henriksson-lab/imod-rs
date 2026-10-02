//! `IMOD/Etomo/src/etomo/comscript/SplitBatchParam.java`.

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java `COMMAND_NAME = ProcessName.SPLIT_BATCH.toString()`.
pub fn command_name() -> String {
    ProcessName::SPLIT_BATCH.to_string()
}

/// Java private static `AXIS_ID`.
const AXIS_ID: AxisID = AxisID::Only;

/// Java final `SplitBatchParam`.
pub struct SplitBatchParam {
    command_file: StringParameter,
    max_gpus_for_one_job: ScriptParameter,
    command_array: Option<Vec<String>>,
}

impl SplitBatchParam {
    /// Java `SplitBatchParam(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> SplitBatchParam {
        let mut command_file = StringParameter::new("CommandFile");
        command_file.set(
            file_type::CLASS
                .batch_run_tomo_comscript
                .get_file_name(Some(manager), Some(AXIS_ID))
                .as_deref(),
        );
        SplitBatchParam {
            command_file,
            max_gpus_for_one_job: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                "MaxGPUsForOneJob",
            ),
            command_array: None,
        }
    }

    /// Java `getCommand`.
    pub fn get_command(&mut self) -> Vec<String> {
        if self.command_array.is_none() {
            self.build_command();
        }
        self.command_array.clone().unwrap()
    }

    /// Java private `buildCommand`.
    fn build_command(&mut self) {
        let mut command: Vec<String> = Vec::new();
        command.push("python".to_owned());
        command.push("-u".to_owned());
        // Java string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command.push(format!("{script_path}{}", command_name()));
        command.push(format!("-{}", self.command_file.get_name()));
        command.push(self.command_file.to_string());
        if !self.max_gpus_for_one_job.is_null() {
            command.push(format!("-{}", self.max_gpus_for_one_job.get_name()));
            command.push(self.max_gpus_for_one_job.to_string());
        }
        let command_size = command.len();
        let mut command_array = Vec::with_capacity(command_size);
        for i in 0..command_size {
            command_array.push(command[i].clone());
        }
        self.command_array = Some(command_array);
    }

    /// Java `setMaxGPUsForOneJob(String)`.
    pub fn set_max_gpus_for_one_job_string(&mut self, input: Option<&str>) {
        self.max_gpus_for_one_job.set_string(input);
    }

    /// Java `setMaxGPUsForOneJob(int)`.
    pub fn set_max_gpus_for_one_job_int(&mut self, input: i32) {
        self.max_gpus_for_one_job.set_int(input);
    }

    /// Java `resetMaxGPUsForOneJob`.
    pub fn reset_max_gpus_for_one_job(&mut self) {
        self.max_gpus_for_one_job.reset();
    }
}
