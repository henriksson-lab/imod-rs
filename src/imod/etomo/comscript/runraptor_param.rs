//! `IMOD/Etomo/src/etomo/comscript/RunraptorParam.java`.
//!
//! Parameters for runraptor: the command line eTomo's RAPTOR panel runs
//! (`python -u <IMOD bin>/runraptor -PID -diam D -mark M <stack>`).  Where
//! that line runs is the process layer's business: `SystemProgram` starts our
//! own `runraptor` for it (`system_program::resolve_command_array`).

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::ui_expert_utilities::UIExpertUtilities;
use crate::imod::etomo::util::dataset_files;

/// Java private static `DIAM_OPTION`.
const DIAM_OPTION: &str = "diam";
/// Java private static `MARK_OPTION`.
const MARK_OPTION: &str = "mark";

/// Java final `RunraptorParam`.
pub struct RunraptorParam {
    command: Vec<String>,
    diam: EtomoNumber,
    mark: EtomoNumber,
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    use_raw_stack: bool,
}

impl RunraptorParam {
    /// Java `RunraptorParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> RunraptorParam {
        let mut diam = EtomoNumber::new();
        let mut mark = EtomoNumber::new();
        diam.set_valid_floor(1);
        mark.set_valid_floor(10);
        RunraptorParam {
            command: Vec::new(),
            diam,
            mark,
            manager,
            axis_id,
            use_raw_stack: false,
        }
    }

    /// Java private `buildCommand`.
    fn build_command(&mut self) {
        self.command.push("python".to_owned());
        self.command.push("-u".to_owned());
        // `EtomoDirector.INSTANCE.getPythonScriptPath() + ProcessName.RUNRAPTOR`: Java
        // string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        self.command
            .push(format!("{script_path}{}", ProcessName::RUNRAPTOR));
        self.command.push("-PID".to_owned());
        self.command.push(format!("-{DIAM_OPTION}"));
        self.command.push(self.diam.to_string());
        self.command.push(format!("-{MARK_OPTION}"));
        self.command.push(self.mark.to_string());
        // The source adds whatever the name is; a null name is a null element that
        // `ProcessBuilder` rejects with a NullPointerException.  Here nothing is
        // added, and runraptor then refuses the short command with its usage.
        let stack = if self.use_raw_stack {
            dataset_files::get_stack_name(self.manager, Some(self.axis_id))
        } else {
            dataset_files::get_prealigned_stack_name(self.manager, Some(self.axis_id))
        };
        if let Some(stack) = stack {
            self.command.push(stack);
        }
    }

    /// Java `setDiam(String, boolean)`.  Returns an error message, or `None`.
    pub fn set_diam(&mut self, input: &str, may_be_binned: bool) -> Option<String> {
        self.diam.set_string(Some(input));
        if self.diam.is_null() {
            return Some(format!("Empty {DIAM_OPTION} parameter."));
        }
        if !self.diam.is_valid() {
            return Some(self.diam.get_invalid_reason());
        }
        if may_be_binned {
            // `Math.round(diam.getInt() / binning)`: an int quotient (the binning is
            // an Integer), so `Math.round(float)` of a whole number
            let binning = UIExpertUtilities::INSTANCE
                .get_stack_binning_base_manager_axis_id_file_type(
                    self.manager,
                    self.axis_id,
                    &file_type::CLASS.prealigned_stack,
                );
            let quotient = self.diam.get_int().wrapping_div(binning);
            self.diam.set_int(quotient);
            if !self.diam.is_valid() {
                return Some(format!(
                    "The binned diameter is {}.  {}",
                    *self.diam,
                    self.diam.get_invalid_reason()
                ));
            }
        }
        None
    }

    /// Java `setMark(String)`.  Returns an error message, or `None`.
    pub fn set_mark(&mut self, input: &str) -> Option<String> {
        self.mark.set_string(Some(input));
        if self.mark.is_null() {
            return Some(format!("Empty {MARK_OPTION} parameter."));
        }
        if !self.mark.is_valid() {
            return Some(self.mark.get_invalid_reason());
        }
        None
    }

    /// Java `setUseRawStack(boolean)`.
    pub fn set_use_raw_stack(&mut self, input: bool) {
        self.use_raw_stack = input;
    }

    /// Java `getProcessName()`.
    pub fn get_process_name(&self) -> ProcessName {
        ProcessName::RUNRAPTOR
    }

    /// Java `getCommandArray()`.  Echoes the command on standard error, as the
    /// source does.
    pub fn get_command_array(&mut self) -> Vec<String> {
        if self.command.is_empty() {
            self.build_command();
        }
        let command_array = self.command.clone();
        for element in &command_array {
            eprint!("{element} ");
        }
        if !command_array.is_empty() {
            eprintln!();
        }
        command_array
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::application_manager::ApplicationManager;

    #[test]
    fn mark_and_diameter_are_validated_as_the_source_does() {
        // Managers live on the event dispatch thread, as in Java.
        crate::imod::etomo::util::event_queue::invoke_and_wait(|| {
            let manager = ApplicationManager::new(None, Some(AxisID::Only));
            let mut param = RunraptorParam::new(manager, AxisID::Only);
            assert_eq!(param.set_mark(""), Some("Empty mark parameter.".to_owned()));
            // floor 10
            assert!(param.set_mark("9").is_some());
            assert_eq!(param.set_mark("20"), None);
            assert_eq!(
                param.set_diam(" ", false),
                Some("Empty diam parameter.".to_owned())
            );
            // floor 1
            assert!(param.set_diam("0", false).is_some());
            assert_eq!(param.set_diam("11", false), None);
            assert_eq!(param.get_process_name(), ProcessName::RUNRAPTOR);
        });
    }

    #[test]
    fn command_is_python_runraptor_with_pid_diam_and_mark() {
        // Managers live on the event dispatch thread, as in Java.
        crate::imod::etomo::util::event_queue::invoke_and_wait(|| {
            let manager = ApplicationManager::new(None, Some(AxisID::Only));
            let mut param = RunraptorParam::new(manager, AxisID::Only);
            param.set_mark("20");
            param.set_diam("11", false);
            let command = param.get_command_array();
            assert_eq!(&command[..2], ["python", "-u"]);
            assert!(command[2].ends_with("runraptor"), "{command:?}");
            assert_eq!(&command[3..8], ["-PID", "-diam", "11", "-mark", "20"]);
            // Built once, as the source's `command.isEmpty()` test does
            param.set_mark("30");
            assert_eq!(param.get_command_array(), command);
        });
    }
}
