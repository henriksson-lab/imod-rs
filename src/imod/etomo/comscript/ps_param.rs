//! `IMOD/Etomo/src/etomo/comscript/PsParam.java`.
//!
//! Builds a `ps` command (over ssh for another host) and parses its output.

use super::ssh_param;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::os_type::OSType;
use crate::imod::etomo::r#type::time::Time;

/// Java private static `PID_HEADER`.
const PID_HEADER: &str = "PID";
/// Java private static `PARENT_PID_HEADER`.
const PARENT_PID_HEADER: &str = "PPID";
/// Java private static `GROUP_PID_HEADER`.
const GROUP_PID_HEADER: &str = "PGID";
/// Java private static `WINDOW_PID_HEADER`.
const WINDOW_PID_HEADER: &str = "WINPID";
/// Java private static `TERMINAL_HEADER`.
const TERMINAL_HEADER: &str = "TTY";
/// Java private static `USER_ID_HEADER`.
const USER_ID_HEADER: &str = "UID";

/// Java `String.indexOf(String)`: -1 when not found.  `ps` output is ASCII, so
/// the byte offset is the UTF-16 index.
fn index_of(string: &str, target: &str) -> i32 {
    match string.find(target) {
        None => -1,
        Some(index) => index as i32,
    }
}

/// Java `row.substring(start, end).trim()`.  Java throws
/// `StringIndexOutOfBoundsException` when a header was missing (its index is -1)
/// or the row is shorter than the column; see `Values::get_pid`.
fn substring_trim(row: &str, start: i32, end: i32) -> String {
    if start < 0 || end < start || end as usize > row.len() {
        return String::new();
    }
    match row.get(start as usize..end as usize) {
        None => String::new(),
        Some(substring) => substring.trim_matches(|c: char| c <= ' ').to_string(),
    }
}

/// Java public final `PsParam`.
pub struct PsParam {
    /// Java private final field `startTimeHeader`.
    start_time_header: String,
    /// Java private final field `command`.
    command: Vec<String>,
    /// Java private final field `valuesArray`.
    values_array: Vec<Values>,
    /// Java private field `output`.
    output: Option<Vec<Option<String>>>,
    pid_start_index: i32,
    pid_end_index: i32,
    parent_pid_start_index: i32,
    parent_pid_end_index: i32,
    group_pid_start_index: i32,
    group_pid_end_index: i32,
    window_pid_start_index: i32,
    window_pid_end_index: i32,
    terminal_start_index: i32,
    terminal_end_index: i32,
    user_id_start_index: i32,
    user_id_end_index: i32,
    start_time_start_index: i32,
    start_time_end_index: i32,
    pid_column: bool,
    parent_pid_column: bool,
    group_pid_column: bool,
    window_pid_column: bool,
    terminal_column: bool,
    user_id_column: bool,
    start_time_column: bool,
    /// Java private field `debug`.
    debug: DebugLevel,
}

impl PsParam {
    /// Java `PsParam(BaseManager, AxisID, String, OSType, String, boolean)`.
    /// Builds the ps commmand.  Uses the pid to limit the output to one process.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        pid: Option<&str>,
        os_type: OSType,
        host_name: Option<&str>,
        will_run_on_worker_thread: bool,
    ) -> PsParam {
        let mut instance = PsParam {
            start_time_header: if os_type == OSType::Windows {
                "STIME".to_string()
            } else {
                "STARTED".to_string()
            },
            command: Vec::new(),
            values_array: Vec::new(),
            output: None,
            pid_start_index: -1,
            pid_end_index: -1,
            parent_pid_start_index: -1,
            parent_pid_end_index: -1,
            group_pid_start_index: -1,
            group_pid_end_index: -1,
            window_pid_start_index: -1,
            window_pid_end_index: -1,
            terminal_start_index: -1,
            terminal_end_index: -1,
            user_id_start_index: -1,
            user_id_end_index: -1,
            start_time_start_index: -1,
            start_time_end_index: -1,
            pid_column: true,
            parent_pid_column: false,
            group_pid_column: true,
            window_pid_column: false,
            terminal_column: false,
            user_id_column: false,
            start_time_column: true,
            debug: etomo_director::ARGUMENTS.lock().unwrap().get_debug_level(),
        };
        if let Some(host_name) = host_name
            && host_name != "*"
            && Some(host_name.to_string())
                != Network::get_local_host_name(
                    manager,
                    axis_id,
                    manager.get_property_user_dir().as_deref(),
                )
        {
            // If the timeout option cannot be added to ssh then only ssh if the caller
            // of this constructor promises to run the command in a worker thread. An
            // ssh on the main thread can lock up the user interface.
            if will_run_on_worker_thread || ssh_param::INSTANCE.is_timeout_available(manager) {
                let ssh_command = ssh_param::INSTANCE.get_command(manager, true, Some(host_name));
                for element in ssh_command {
                    if !element.is_empty() {
                        instance.command.push(element);
                    }
                }
            }
        }
        // The calling function expects at least one line be returned even when the
        // pid is not found.
        if os_type == OSType::Windows {
            instance.command.push("python".to_string());
            let python_script_path = etomo_director::INSTANCE
                .get_python_script_path()
                .as_deref()
                .unwrap_or("null")
                .to_string();
            instance
                .command
                .push(format!("{python_script_path}b3dwinps"));
        } else {
            instance.command.push("ps".to_string());
        }
        if os_type == OSType::Windows {
            instance.parent_pid_column = true;
            instance.window_pid_column = true;
            instance.terminal_column = true;
            instance.user_id_column = true;
        } else if os_type == OSType::Mac {
            instance.command.push("-A".to_string());
        } else if let Some(pid) = pid {
            instance.command.push("-p".to_string());
            instance.command.push(pid.to_string());
        }
        if os_type != OSType::Windows {
            instance.command.push("-o".to_string());
            instance.command.push("pid,pgid,lstart".to_string());
        }
        instance
    }

    /// Java `getRow`.
    pub fn get_row(&mut self) -> Row<'_> {
        Row::new(self)
    }

    /// Java `getCommandArray`.
    pub fn get_command_array(&self) -> &Vec<String> {
        &self.command
    }

    /// Java `setOutput(String[])`.  Sets the output of the ps command.  Sets the
    /// indices of fields that can be parsed.
    pub fn set_output(&mut self, output: Option<Vec<Option<String>>>) {
        self.output = output;
        self.values_array.clear();
        let header = match &self.output {
            Some(output) if output.len() >= 2 => {
                // Java `output[0].indexOf(...)` on a null header line throws
                // NullPointerException; an absent header is searched as empty here.
                output[0].clone().unwrap_or_default()
            }
            _ => return,
        };
        // set the indexes based on the header
        let mut end_index = -1;
        if self.pid_column {
            self.pid_start_index = end_index + 1;
            self.pid_end_index = index_of(&header, PID_HEADER) + PID_HEADER.len() as i32;
            end_index = self.pid_end_index;
        }
        if self.parent_pid_column {
            self.parent_pid_start_index = end_index + 1;
            self.parent_pid_end_index =
                index_of(&header, PARENT_PID_HEADER) + PARENT_PID_HEADER.len() as i32;
            end_index = self.parent_pid_end_index;
        }
        if self.group_pid_column {
            self.group_pid_start_index = end_index + 1;
            self.group_pid_end_index =
                index_of(&header, GROUP_PID_HEADER) + GROUP_PID_HEADER.len() as i32;
            end_index = self.group_pid_end_index;
        }
        if self.window_pid_column {
            self.window_pid_start_index = end_index + 1;
            self.window_pid_end_index =
                index_of(&header, WINDOW_PID_HEADER) + WINDOW_PID_HEADER.len() as i32;
            end_index = self.window_pid_end_index;
        }
        if self.terminal_column {
            self.terminal_start_index = end_index + 1;
            self.terminal_end_index =
                index_of(&header, TERMINAL_HEADER) + TERMINAL_HEADER.len() as i32;
            end_index = self.terminal_end_index;
        }
        if self.user_id_column {
            self.user_id_start_index = end_index + 1;
            self.user_id_end_index =
                index_of(&header, USER_ID_HEADER) + USER_ID_HEADER.len() as i32;
            end_index = self.user_id_end_index;
        }
        if self.start_time_column {
            self.start_time_start_index = end_index + 1;
            self.start_time_end_index =
                index_of(&header, &self.start_time_header) + self.start_time_header.len() as i32;
        }
        self.load_values();
    }

    /// Java private `loadValues`.  Loads the output of the ps command into
    /// valuesArray.
    fn load_values(&mut self) {
        if !self.values_array.is_empty() {
            return;
        }
        let output = match &self.output {
            None => return,
            Some(output) => output,
        };
        for i in 1..output.len() {
            if self.debug.is_verbose() {
                eprintln!("{}", output[i].as_deref().unwrap_or("null"));
            }
            if let Some(line) = &output[i]
                && line.len() as i32 >= self.start_time_end_index
            {
                self.values_array.push(Values::new(line.clone()));
            }
        }
    }

    /// Java package-private `findRow(String)`.  Returns the index of the row in the
    /// ps output where the pid is found.
    pub fn find_row(&mut self, pid: Option<&str>) -> i32 {
        let (start, end) = (self.pid_start_index, self.pid_end_index);
        for i in 0..self.values_array.len() {
            let values = &mut self.values_array[i];
            if Some(values.get_pid(start, end).as_str()) == pid {
                return i as i32;
            }
        }
        -1
    }

    /// Java `findRow(String, String, Time)`.  Returns true if the pid, groupPid,
    /// and startTime are found on a row.
    pub fn find_row_with_start_time(
        &mut self,
        pid: Option<&str>,
        group_pid: Option<&str>,
        start_time: Option<&Time>,
    ) -> bool {
        let start_time_string = match start_time {
            None => "null".to_string(),
            Some(start_time) => start_time.to_string(),
        };
        if self.debug.is_verbose() {
            eprintln!(
                "Looking for a ps row with pid={},groupPid={},startTime={}",
                pid.unwrap_or("null"),
                group_pid.unwrap_or("null"),
                start_time_string
            );
        }
        let indices = self.indices();
        for i in 0..self.values_array.len() {
            let values = &mut self.values_array[i];
            if Some(values.get_pid(indices.0, indices.1).as_str()) == pid
                && Some(values.get_group_pid(indices.2, indices.3).as_str()) == group_pid
            {
                if self.debug.is_extra_verbose() {
                    eprintln!(
                        "Checking the startTime of a ps row with pid={},groupPid={},startTime={}",
                        pid.unwrap_or("null"),
                        group_pid.unwrap_or("null"),
                        start_time_string
                    );
                }
                // Java passes a null startTime on to `almostEquals`; a null time
                // matches nothing here.
                if let Some(start_time) = start_time
                    && values
                        .get_start_time(indices.4, indices.5)
                        .almost_equals(start_time)
                {
                    return true;
                }
            }
        }
        false
    }

    /// The column indices `Values` reads through its outer instance.
    fn indices(&self) -> (i32, i32, i32, i32, i32, i32) {
        (
            self.get_pid_start_index(),
            self.get_pid_end_index(),
            self.get_group_pid_start_index(),
            self.get_group_pid_end_index(),
            self.get_start_time_start_index(),
            self.get_start_time_end_index(),
        )
    }

    /// Java package-private `getGroupPid(int)`.
    pub fn get_group_pid(&mut self, row_index: i32) -> Option<String> {
        if row_index == -1 || self.values_array.len() as i32 <= row_index {
            return None;
        }
        let indices = self.indices();
        Some(self.values_array[row_index as usize].get_group_pid(indices.2, indices.3))
    }

    /// Java package-private `getStartTime(int)`.
    pub fn get_start_time(&mut self, row_index: i32) -> Option<Time> {
        if row_index == -1 || self.values_array.len() as i32 <= row_index {
            return None;
        }
        let indices = self.indices();
        Some(
            self.values_array[row_index as usize]
                .get_start_time(indices.4, indices.5)
                .clone(),
        )
    }

    /// Java package-private `getPidStartIndex`.
    pub fn get_pid_start_index(&self) -> i32 {
        self.pid_start_index
    }

    /// Java package-private `getPidEndIndex`.
    pub fn get_pid_end_index(&self) -> i32 {
        self.pid_end_index
    }

    /// Java package-private `getGroupPidStartIndex`.
    pub fn get_group_pid_start_index(&self) -> i32 {
        self.group_pid_start_index
    }

    /// Java package-private `getGroupPidEndIndex`.
    pub fn get_group_pid_end_index(&self) -> i32 {
        self.group_pid_end_index
    }

    /// Java package-private `getStartTimeStartIndex`.
    pub fn get_start_time_start_index(&self) -> i32 {
        self.start_time_start_index
    }

    /// Java package-private `getStartTimeEndIndex`.
    pub fn get_start_time_end_index(&self) -> i32 {
        self.start_time_end_index
    }
}

/// Java private final inner class `Values`.  The Java inner class reads the
/// column indices through its outer instance; they are passed in here.
struct Values {
    /// Java private final field `row`.
    row: String,
    /// Java private field `pid`.
    pid: Option<String>,
    /// Java private field `groupPid`.
    group_pid: Option<String>,
    /// Java private field `startTime`.
    start_time: Option<Time>,
}

impl Values {
    /// Java private `Values(String)`.
    fn new(row: String) -> Values {
        Values {
            row,
            pid: None,
            group_pid: None,
            start_time: None,
        }
    }

    /// Java `toString`.
    fn to_string(&mut self, indices: (i32, i32, i32, i32, i32, i32)) -> String {
        let pid = self.get_pid(indices.0, indices.1);
        let group_pid = self.get_group_pid(indices.2, indices.3);
        let start_time = self.get_start_time(indices.4, indices.5).to_string();
        format!(
            "[pid={pid},groupPid={group_pid},startTime={start_time},row={}]",
            self.row
        )
    }

    /// Java package-private `getPid`.  PsParam.java:327 (and the two getters
    /// below) throw `StringIndexOutOfBoundsException` when the header lacked the
    /// column (`indexOf` gave -1) or the columns overlap; an empty value is
    /// returned instead (fixed in translation).
    fn get_pid(&mut self, pid_start_index: i32, pid_end_index: i32) -> String {
        if let Some(pid) = &self.pid {
            return pid.clone();
        }
        let pid = substring_trim(&self.row, pid_start_index, pid_end_index);
        self.pid = Some(pid.clone());
        pid
    }

    /// Java package-private `getGroupPid`.
    fn get_group_pid(&mut self, group_pid_start_index: i32, group_pid_end_index: i32) -> String {
        if let Some(group_pid) = &self.group_pid {
            return group_pid.clone();
        }
        let group_pid = substring_trim(&self.row, group_pid_start_index, group_pid_end_index);
        self.group_pid = Some(group_pid.clone());
        group_pid
    }

    /// Java package-private `getStartTime`.
    fn get_start_time(&mut self, start_time_start_index: i32, start_time_end_index: i32) -> &Time {
        if self.start_time.is_none() {
            self.start_time = Some(Time::new(&substring_trim(
                &self.row,
                start_time_start_index,
                start_time_end_index,
            )));
        }
        self.start_time.as_ref().unwrap()
    }
}

/// Java public inner class `Row`.
pub struct Row<'a> {
    /// Java private final field `psParam`.
    ps_param: &'a mut PsParam,
    /// Java private field `index`.
    index: i32,
}

impl<'a> Row<'a> {
    /// Java package-private `Row(PsParam)`.
    fn new(ps_param: &'a mut PsParam) -> Row<'a> {
        Row {
            ps_param,
            index: -1,
        }
    }

    /// Java `find(String)`.  Find the process which matches the pid.
    pub fn find(&mut self, pid: Option<&str>) -> bool {
        self.index = self.ps_param.find_row(pid);
        if self.index != -1 {
            return true;
        }
        false
    }

    /// Java `getGroupPid`.
    pub fn get_group_pid(&mut self) -> Option<String> {
        if self.ps_param.debug.is_extra() {
            eprintln!(
                "psParam.getGroupPid({}):{}",
                self.index,
                self.ps_param
                    .get_group_pid(self.index)
                    .unwrap_or_else(|| "null".to_string())
            );
        }
        self.ps_param.get_group_pid(self.index)
    }

    /// Java `getStartTime`.
    pub fn get_start_time(&mut self) -> Option<Time> {
        if self.ps_param.debug.is_extra() {
            eprintln!(
                "psParam.getStartTime({}):{}",
                self.index,
                match self.ps_param.get_start_time(self.index) {
                    None => "null".to_string(),
                    Some(start_time) => start_time.to_string(),
                }
            );
        }
        self.ps_param.get_start_time(self.index)
    }
}
