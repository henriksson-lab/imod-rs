//! `IMOD/Etomo/src/etomo/storage/ProcessSignal.java`.
//!
//! One recognised line of a batchruntomo dataset log.  The four overloads of the
//! static `getInstance` carry the suffix of their parameter types.  String indexes are
//! byte offsets (as `java_lang_string_*` in `util::utilities` use them); the tags
//! searched for are ASCII.

use regex::Regex;

use crate::imod::etomo::logic::com_file_extension_tool::ComFileExtensionTool;
use crate::imod::etomo::process::process_output_strings::{
    BRT_DATASET_LOG_CLOSED_TAG, BRT_REACHED_STEP_TAG, BRT_START_AXIS, BRT_STARTED_DATASET_TAG,
    BRT_STEP_MSG_ID, BRT_STEP_SUCCESS_MSG_ID,
};
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::step::Step;
use crate::imod::etomo::util::utilities::{java_lang_string_index_of_from, java_lang_string_split};

/// Java final `ProcessSignal`.
#[derive(Clone, Debug)]
pub struct ProcessSignal {
    /// Java private final field `timestamp`.
    timestamp: Option<String>,
    /// Java private final field `finished`.
    finished: bool,
    /// Java private final field `processName`.
    process_name: Option<ProcessName>,
    /// Java private final field `step`.
    step: Option<Step>,
    /// Java private final field `line`.
    line: String,
    /// Java private field `axisID`, initialised to null.
    axis_id: Option<AxisID>,
    /// Java private field `processState`, initialised to null.
    process_state: Option<ProcessState>,
    /// Java private field `dialogType`, initialised to null.
    dialog_type: Option<DialogType>,
}

impl ProcessSignal {
    /// Java private `ProcessSignal(String, boolean, String)`.
    fn new_string_boolean_string(timestamp: &str, finished: bool, line: &str) -> ProcessSignal {
        ProcessSignal {
            timestamp: Some(timestamp.to_string()),
            finished,
            line: line.to_string(),
            step: None,
            process_name: None,
            axis_id: None,
            process_state: None,
            dialog_type: None,
        }
    }

    /// Java private `ProcessSignal(AxisID, String)`.
    fn new_axis_id_string(axis_id: Option<AxisID>, line: &str) -> ProcessSignal {
        ProcessSignal {
            axis_id,
            line: line.to_string(),
            timestamp: None,
            finished: false,
            step: None,
            process_name: None,
            process_state: None,
            dialog_type: None,
        }
    }

    /// Java private `ProcessSignal(ProcessState, ProcessName, AxisID, String)`.
    fn new_process_state_process_name_axis_id_string(
        process_state: ProcessState,
        process_name: ProcessName,
        axis_id: Option<AxisID>,
        line: &str,
    ) -> ProcessSignal {
        ProcessSignal {
            process_state: Some(process_state),
            process_name: Some(process_name),
            axis_id,
            line: line.to_string(),
            step: None,
            timestamp: None,
            finished: false,
            dialog_type: None,
        }
    }

    /// Java private `ProcessSignal(Step, String)`.  The step signal.  It has no axis
    /// information.
    fn new_step_string(step: Step, line: &str) -> ProcessSignal {
        ProcessSignal {
            step: Some(step),
            process_name: None,
            axis_id: None,
            line: line.to_string(),
            timestamp: None,
            finished: false,
            process_state: None,
            dialog_type: None,
        }
    }

    /// Java package-private static `getInstance(AxisType, String, boolean)`.  Parses
    /// line and returns a ProcessSignal if the line contains information.  Otherwise
    /// returns null.  `timestamp`: when true only gets timestamps.
    pub(crate) fn get_instance_axis_type_string_boolean(
        axis_type: Option<AxisType>,
        line: Option<&str>,
        timestamp: bool,
    ) -> Option<ProcessSignal> {
        let line = line?;
        let mut tag_index;
        // Look for timestamp.
        tag_index = (java_lang_string_index_of_from(line, BRT_STARTED_DATASET_TAG, 0) as i32);
        if tag_index != -1 {
            return Self::get_instance_string_int_boolean(line, tag_index, false);
        }
        tag_index = (java_lang_string_index_of_from(line, BRT_DATASET_LOG_CLOSED_TAG, 0) as i32);
        if tag_index != -1 {
            return Self::get_instance_string_int_boolean(line, tag_index, true);
        }
        if timestamp {
            return None;
        }
        // Look for information about the current axis
        tag_index = (java_lang_string_index_of_from(line, BRT_START_AXIS, 0) as i32);
        if tag_index != -1 {
            return Self::get_axis_instance(line, tag_index);
        }
        // Look for a process lines or a step line.
        tag_index = (java_lang_string_index_of_from(line, BRT_STEP_MSG_ID, 0) as i32);
        if tag_index != -1 {
            return Self::get_instance_axis_type_string_int_process_state(
                axis_type,
                line,
                tag_index,
                ProcessState::InProgress,
            );
        } else {
            tag_index = (java_lang_string_index_of_from(line, BRT_STEP_SUCCESS_MSG_ID, 0) as i32);
            if tag_index != -1 {
                return Self::get_instance_axis_type_string_int_process_state(
                    axis_type,
                    line,
                    tag_index,
                    ProcessState::Complete,
                );
            } else if (java_lang_string_index_of_from(line, BRT_REACHED_STEP_TAG, 0) as i32) != -1 {
                return Self::get_instance_string_int(line, tag_index);
            }
        }
        None
    }

    /// Java private static `getInstance(String, int)`.  Find a process signal from a
    /// Step string.
    fn get_instance_string_int(line: &str, tag_index: i32) -> Option<ProcessSignal> {
        let _ = tag_index;
        // Assumes:
        // step is at the end of the line
        // Step is preceeded by white space
        let line_array = java_lang_string_split(line, &Regex::new(r"\s+").unwrap());
        if line_array.is_empty() {
            return None;
        }
        let step = Step::get_instance(Some(&line_array[line_array.len() - 1]))?;
        Some(ProcessSignal::new_step_string(step, line))
    }

    /// Java private static `getInstance(String, int, boolean)`.  Find a process signal
    /// with a dataset started or finished tag.  This is to get the timestamp.
    fn get_instance_string_int_boolean(
        line: &str,
        tag_index: i32,
        finished: bool,
    ) -> Option<ProcessSignal> {
        // Batchruntomo started on data set BB, Wed Apr 19 17:37:39 2017
        // Batchruntomo finished with data set, Mon Apr 24 13:29:46 2017
        // Assumes:
        // The timestamp is between the last comma and the tag.
        let line_array = java_lang_string_split(
            &line[..tag_index as usize],
            &Regex::new(r"\s*\,\s*").unwrap(),
        );
        if line_array.is_empty() {
            return None;
        }
        Some(ProcessSignal::new_string_boolean_string(
            &line_array[line_array.len() - 1],
            finished,
            line,
        ))
    }

    /// Java private static `getAxisInstance(String, int)`.  Find a process signal with
    /// an axis started tag.  This is to get the current axis being worked on.
    fn get_axis_instance(line: &str, tag_index: i32) -> Option<ProcessSignal> {
        // Completed axis A of dataset BB through step 7
        // Starting axis B [:LOG]
        // Assumes:
        // The axis letter, preceded by whitespace, follows the tag.
        let rest = java_lang_string_trim(&line[tag_index as usize + BRT_START_AXIS.len()..]);
        // ProcessSignal.java:181-183 takes `.substring(0, 1)` of the trimmed remainder,
        // which throws StringIndexOutOfBoundsException when nothing follows the tag.
        // Fixed in translation: an empty remainder gives a signal with a null axisID
        // (what `AxisID.getInstanceIgnoreCase` returns for an unrecognised letter).
        let axis_id = match rest.chars().next() {
            None => None,
            Some(first) => AxisID::get_instance_ignore_case(Some(&first.to_string())),
        };
        Some(ProcessSignal::new_axis_id_string(axis_id, line))
    }

    /// Java private static `getInstance(AxisType, String, int, ProcessState)`.  Find a
    /// process signal with a process name.
    fn get_instance_axis_type_string_int_process_state(
        axis_type: Option<AxisType>,
        line: &str,
        tag_index: i32,
        process_state: ProcessState,
    ) -> Option<ProcessSignal> {
        // Assumes:
        // The tag comes after the process name
        // the process name includes the extension
        // The process name's extension comes after anything that looks like the
        // extension.
        // the process name contains no spaces
        // the process name is preceeded by white space
        let mut process_end_index = ComFileExtensionTool::last_index_of(Some(line), tag_index)
            - (if axis_type == Some(AxisType::DualAxis) {
                1
            } else {
                0
            });
        if process_end_index < 0 {
            return None;
        }
        let mut axis_id = Some(AxisID::First);
        if axis_type == Some(AxisType::DualAxis) {
            // `line.charAt(processEndIndex)`; a byte inside a multi-byte character is
            // not an axis letter.
            let index = process_end_index as usize;
            axis_id = if line.is_char_boundary(index) {
                match line[index..].chars().next() {
                    None => None,
                    Some(c) => AxisID::get_instance_from_char(c),
                }
            } else {
                None
            };
            if axis_id.is_none() {
                // a process without an axisID such as volcombine.
                process_end_index += 1;
            }
        }
        let line_array = java_lang_string_split(
            &line[..process_end_index as usize],
            &Regex::new(r"\s+").unwrap(),
        );
        if line_array.is_empty() {
            return None;
        }
        let process_name = ProcessName::get_instance(Some(&line_array[line_array.len() - 1]))?;
        Some(
            ProcessSignal::new_process_state_process_name_axis_id_string(
                process_state,
                process_name,
                axis_id,
                line,
            ),
        )
    }

    /// Java `getAxisID()`.
    pub fn get_axis_id(&self) -> Option<AxisID> {
        self.axis_id
    }

    /// Java `getStep()`.
    pub fn get_step(&self) -> Option<Step> {
        self.step
    }

    /// Java `setAxisID(AxisID)`.
    pub fn set_axis_id(&mut self, input: Option<AxisID>) {
        self.axis_id = input;
    }

    /// Java `setProcessState(ProcessState)`.
    pub fn set_process_state(&mut self, input: Option<ProcessState>) {
        self.process_state = input;
    }

    /// Java `setDialogType(DialogType)`.
    pub fn set_dialog_type(&mut self, input: Option<DialogType>) {
        self.dialog_type = input;
    }

    /// Java `getProcessState()`.
    pub fn get_process_state(&self) -> Option<ProcessState> {
        self.process_state
    }

    /// Java `getProcessName()`.
    pub fn get_process_name(&self) -> Option<ProcessName> {
        self.process_name
    }

    /// Java `isTimestamp()`.
    pub fn is_timestamp(&self) -> bool {
        self.timestamp.is_some()
    }

    /// Java `isAxisID()`.
    pub fn is_axis_id(&self) -> bool {
        self.axis_id.is_some()
            && self.timestamp.is_none()
            && self.process_name.is_none()
            && self.step.is_none()
    }

    /// Java `isFinished()`.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Java `equalsTimestamp(String)`.
    pub fn equals_timestamp(&self, input: Option<&str>) -> bool {
        match (&self.timestamp, input) {
            (Some(timestamp), Some(input)) => timestamp == input,
            _ => false,
        }
    }

    /// Java `getTimestamp()`.
    pub fn get_timestamp(&self) -> Option<&str> {
        self.timestamp.as_deref()
    }

    /// Java `getDialogType()`.
    pub fn get_dialog_type(&self) -> Option<DialogType> {
        self.dialog_type
    }
}

/// Java `toString()`.  A null member prints as "null".
impl std::fmt::Display for ProcessSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn or_null<T: std::fmt::Display>(value: Option<T>) -> String {
            match value {
                None => "null".to_string(),
                Some(value) => value.to_string(),
            }
        }
        write!(
            f,
            "[timestamp:{},finished:{},processName:{},axisID:{},step:{},processState:{},dialogType:{},line:\n{}]",
            or_null(self.timestamp.as_deref()),
            self.finished,
            or_null(self.process_name),
            or_null(self.axis_id),
            or_null(self.step),
            or_null(self.process_state),
            or_null(self.dialog_type),
            self.line
        )
    }
}
