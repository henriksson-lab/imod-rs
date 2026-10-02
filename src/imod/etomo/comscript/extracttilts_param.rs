//! `IMOD/Etomo/src/etomo/comscript/ExtracttiltsParam.java`.
//!
//! Builds the `extracttilts <stack> <rawtilt>` command line.

use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::dataset_files;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "extracttilts";

/// Java `ExtracttiltsParam`.
pub struct ExtracttiltsParam {
    axis_id: AxisID,
    manager: &'static dyn BaseManager,
    command_array: Option<Vec<String>>,
}

impl ExtracttiltsParam {
    /// Java `ExtracttiltsParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> ExtracttiltsParam {
        ExtracttiltsParam {
            axis_id,
            manager,
            command_array: None,
        }
    }

    /// Java `getCommand()`.
    pub fn get_command(&mut self) -> Vec<String> {
        if self.command_array.is_none() {
            self.build_command();
        }
        self.command_array.clone().unwrap_or_default()
    }

    /// Java private `buildCommand()`.
    fn build_command(&mut self) {
        let mut command: Vec<String> = Vec::new();
        // `BaseManager.getIMODBinPath() + COMMAND_NAME`: a null path concatenates as "null"
        command.push(format!(
            "{}{}",
            base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_string()),
            COMMAND_NAME
        ));
        // Java assigns `dataset = manager.getName()` and never reads it.
        let _dataset = self.manager.get_name();
        // A null stack name is a null element in Java's list; "null" here.
        command.push(
            dataset_files::get_stack_name(self.manager, Some(self.axis_id))
                .unwrap_or_else(|| "null".to_string()),
        );
        command.push(dataset_files::get_raw_tilt_name(
            self.manager,
            Some(self.axis_id),
        ));
        let command_size = command.len();
        let mut command_array = vec![String::new(); command_size];
        for i in 0..command_size {
            command_array[i] = command[i].clone();
        }
        self.command_array = Some(command_array);
    }
}
