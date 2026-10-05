//! `IMOD/Etomo/src/etomo/comscript/ProcesschunksParam.java`.
//!
//! Parameters for processchunks, which runs a set of chunk command files on
//! the selected computers, GPUs or cluster queue.  The command is built once
//! per instance (there is no reset function); the values that can change for a
//! resume (resume, nice, the machine list) clear it so it is rebuilt.
//!
//! **Shape.**  The param is handed to `BaseProcessManager::start_detached_process`
//! as an `Arc<dyn DetachedCommandDetails + Send + Sync>` and is kept by the
//! manager for a resume, so every member the Java changes after the command is
//! submitted - and the lazily built `commandArray` itself - lives in the
//! `Mutex<State>`, and those setters take `&self`.  `subcommandDetails` and
//! `subcommandMode` are returned by reference through the `Command` and
//! `ParallelParam` traits, so they are ordinary fields set through `&mut self`
//! before the param is shared, which is when the Java sets them.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::detached_command_details::DetachedCommandDetails;
use super::field_interface::FieldInterface;
use super::parallel_param::ParallelParam;
use super::process_details::ProcessDetails;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::storage::pc_option_type::PcOptionType;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number};
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::queue_mode::QueueMode;
use crate::imod::etomo::r#type::queue_type::QueueType;
use crate::imod::etomo::r#type::substitution_string::SubstitutionString;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::java_hash_map::JavaHashMap;
use crate::imod::etomo::util::remote_path;
use crate::imod::etomo::util::utilities;

/// Java `NICE_CEILING`.
pub const NICE_CEILING: i32 = 19;
/// Java `DROP_VALUE`.
pub const DROP_VALUE: i32 = 5;
/// Java `WORKING_DIR_OPTION`.
pub const WORKING_DIR_OPTION: &str = "-w";
/// Java `BRT_CHECK_NAME_SUFFIX`.
pub const BRT_CHECK_NAME_SUFFIX: &str = "-pb";
/// Java private static `LIST_DIVIDER`.
const LIST_DIVIDER: &str = ":";

/// Java `s.matches("\\s*")`.
fn matches_whitespace(s: &str) -> bool {
    s.chars()
        .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
}

/// Java's `FileKey outputImageFileKey`, which may be a `FileType` (the
/// subclass); `getOutputImageFileType` tests `instanceof FileType`.
#[derive(Clone, Debug)]
pub enum OutputImageFileKey {
    /// A plain `FileKey`.
    FileKey(FileKey),
    /// A `FileType` instance.
    FileType(Arc<FileType>),
}

/// Java public static final nested class `Element`.
#[derive(Clone, Debug)]
pub struct Element {
    /// Java private final field `number`.
    number: i32,
    /// Java private final field `deviceArray`.
    device_array: Option<Vec<String>>,
}

impl Element {
    /// Java private `Element(int, String[])`.
    fn new(number: i32, device_array: Option<Vec<String>>) -> Element {
        Element {
            number,
            device_array,
        }
    }

    /// Java `getNumber`.
    pub fn get_number(&self) -> i32 {
        self.number
    }

    /// Java `getDeviceArray`.
    pub fn get_device_array(&self) -> Option<&Vec<String>> {
        self.device_array.as_ref()
    }
}

/// Java private static final nested class `Mode` (two identity-compared
/// instances).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Java `Mode.STANDARD`.
    Standard,
    /// Java `Mode.BATCH_RUN_TOMO`.
    BatchRunTomo,
}

impl Mode {
    /// Java private static `getInstance(InterfaceType)`.
    fn get_instance_interface_type(interface_type: Option<InterfaceType>) -> Mode {
        if interface_type == Some(InterfaceType::BatchRunTomo) {
            return Mode::BatchRunTomo;
        }
        Mode::get_default_instance()
    }

    /// Java private static `getInstance(DialogType)`.
    fn get_instance_dialog_type(dialog_type: Option<DialogType>) -> Mode {
        match dialog_type {
            None => Mode::get_default_instance(),
            Some(dialog_type) => {
                Mode::get_instance_interface_type(dialog_type.get_interface_type())
            }
        }
    }

    /// Java private static `getDefaultInstance`.
    fn get_default_instance() -> Mode {
        Mode::Standard
    }
}

/// The Java instance fields that change after the param is shared.
struct State {
    /// Java private final field `resume`.
    resume: EtomoBoolean2,
    /// Java private final field `nice`.
    nice: EtomoNumber,
    /// Java private final field `machineMap`, a `LinkedHashMap`: insertion order.
    machine_map: Vec<(String, Element)>,
    /// Java private final field `cpuNumber`.
    cpu_number: EtomoNumber,
    /// Java private final field `gpusPerClusterJob`.
    gpus_per_cluster_job: EtomoNumber,
    /// Java private final field `secondaryNumber`.
    secondary_number: EtomoNumber,
    /// Java private final field `coresPerNode`.
    cores_per_node: EtomoNumber,
    /// Java private final field `coresPerClusterJob`.
    cores_per_cluster_job: EtomoNumber,
    /// Java private field `commandArray`.
    command_array: Option<Vec<String>>,
    /// Java private field `valid`.
    valid: bool,
    /// Java private field `debug`.
    debug: bool,
    /// Java private field `queueCommand`.
    queue_command: Option<String>,
    /// Java private field `queue`.
    queue: Option<String>,
    /// Java private field `subdirName`.
    subdir_name: Option<String>,
    /// Java private field `test`; never read in the source.
    #[allow(dead_code)]
    test: bool,
    /// Java private field `gpuProcessing`.
    gpu_processing: bool,
    /// Java private field `initialize`.
    initialize: Option<SubstitutionString>,
    /// Java private field `deinitialize`.
    deinitialize: Option<String>,
    /// Java private field `premadeMachineList`.
    premade_machine_list: Option<String>,
    /// Java private final field `multiProc`.
    multi_proc: EtomoNumber,
    /// Java private field `gpuMachineList` (a `StringBuilder`).
    gpu_machine_list: Option<String>,
    /// Java private field `queueMode`.
    queue_mode: Option<QueueMode>,
    /// Java private field `secondaryQueueCommand`.
    secondary_queue_command: Option<String>,
    /// Java private field `secondaryQueue`.
    secondary_queue: Option<String>,
}

/// Java public final `ProcesschunksParam implements DetachedCommandDetails,
/// ParallelParam`.
pub struct ProcesschunksParam {
    /// Java private final field `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final field `axisID`.
    axis_id: AxisID,
    /// Java private final field `rootName`.
    root_name: Option<String>,
    /// Java private final field `outputImageFileKey`.
    output_image_file_key: Option<OutputImageFileKey>,
    /// Java private field `subcommandDetails`.
    subcommand_details: Option<Arc<dyn CommandDetails + Send + Sync>>,
    /// Java private field `subcommandMode`.
    subcommand_mode: Option<Box<dyn CommandMode + Send + Sync>>,
    /// Java private final field `subcommandProcessName`.
    subcommand_process_name: Option<String>,
    /// Java private final field `mode`.
    mode: Mode,
    /// Every other Java field; see the module comment.
    state: Mutex<State>,
}

impl ProcesschunksParam {
    /// Java private `ProcesschunksParam(BaseManager, AxisID, String, String,
    /// FileKey, Mode)`.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        root_name: Option<String>,
        subcommand_process_name: Option<String>,
        output_image_file_key: Option<OutputImageFileKey>,
        mode: Option<Mode>,
    ) -> ProcesschunksParam {
        let mut cores_per_node = EtomoNumber::new();
        let mut cores_per_cluster_job = EtomoNumber::new_with_name("-JC");
        cores_per_node.set_int(1);
        cores_per_cluster_job.set_int(1);
        let mode = match mode {
            Some(mode) => mode,
            None => Mode::get_default_instance(),
        };
        let instance = ProcesschunksParam {
            manager,
            axis_id,
            root_name,
            output_image_file_key,
            subcommand_details: None,
            subcommand_mode: None,
            subcommand_process_name,
            mode,
            state: Mutex::new(State {
                resume: EtomoBoolean2::new(),
                nice: EtomoNumber::new(),
                machine_map: Vec::new(),
                cpu_number: EtomoNumber::new_with_name("-q"),
                gpus_per_cluster_job: EtomoNumber::new_with_name("-JG"),
                secondary_number: EtomoNumber::new_with_name("-SN"),
                cores_per_node,
                cores_per_cluster_job,
                command_array: None,
                valid: true,
                debug: false,
                queue_command: None,
                queue: None,
                subdir_name: Some(String::new()),
                test: false,
                gpu_processing: false,
                initialize: None,
                deinitialize: None,
                premade_machine_list: None,
                multi_proc: EtomoNumber::new_with_name("-M"),
                gpu_machine_list: None,
                queue_mode: None,
                secondary_queue_command: None,
                secondary_queue: None,
            }),
        };
        instance.init();
        instance
    }

    /// Java static `getInstance(BaseManager, AxisID, String, FileKey)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        root_name: Option<&str>,
        output_image_file_key: Option<OutputImageFileKey>,
    ) -> ProcesschunksParam {
        ProcesschunksParam::new(
            manager,
            axis_id,
            root_name.map(str::to_string),
            root_name.map(str::to_string),
            output_image_file_key,
            None,
        )
    }

    /// Java static `getInstance(BaseManager, AxisID, String, FileKey,
    /// InterfaceType)`.
    pub fn get_instance_interface_type(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        root_name: Option<&str>,
        output_image_file_key: Option<OutputImageFileKey>,
        interface_type: Option<InterfaceType>,
    ) -> ProcesschunksParam {
        ProcesschunksParam::new(
            manager,
            axis_id,
            root_name.map(str::to_string),
            root_name.map(str::to_string),
            output_image_file_key,
            Some(Mode::get_instance_interface_type(interface_type)),
        )
    }

    /// Java static `getInstance(BaseManager, AxisID, String, FileKey,
    /// DialogType)`.
    pub fn get_instance_dialog_type(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        root_name: Option<&str>,
        output_image_file_key: Option<OutputImageFileKey>,
        dialog_type: Option<DialogType>,
    ) -> ProcesschunksParam {
        ProcesschunksParam::new(
            manager,
            axis_id,
            root_name.map(str::to_string),
            root_name.map(str::to_string),
            output_image_file_key,
            Some(Mode::get_instance_dialog_type(dialog_type)),
        )
    }

    /// Java static `getInstance(BaseManager, AxisID, ProcessName, FileKey)`.
    /// Sets rootName to ProcessName + AxisID.
    pub fn get_instance_process_name(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        process_name: ProcessName,
        output_image_file_key: Option<OutputImageFileKey>,
    ) -> ProcesschunksParam {
        ProcesschunksParam::new(
            manager,
            axis_id,
            Some(format!("{}{}", process_name, axis_id.get_extension())),
            Some(process_name.to_string()),
            output_image_file_key,
            None,
        )
    }

    /// Java private `init`.
    fn init(&self) {
        let mut state = self.state.lock().unwrap();
        state
            .nice
            .set_int(self.manager.get_parallel_processing_default_nice());
        state.nice.set_floor(cpu_adoc::INSTANCE.get_min_nice());
        state.nice.set_ceiling(NICE_CEILING);
    }

    /// Java `setSubcommandMode(CommandMode)`.
    pub fn set_subcommand_mode(&mut self, input: Option<Box<dyn CommandMode + Send + Sync>>) {
        self.subcommand_mode = input;
    }

    // Updates done

    /// Java `setPremadeMachineList(String)`.
    pub fn set_premade_machine_list(&self, input: Option<&str>) {
        self.state.lock().unwrap().premade_machine_list = input.map(str::to_string);
    }

    /// Java `setResume(boolean)`.  Set resume.  This value can be set after the
    /// command is built because it comes from the parallel panel and can be
    /// changed for a resume.  Causes commandArray to be set to null.
    pub fn set_resume(&self, resume: bool) {
        let mut state = self.state.lock().unwrap();
        if state.resume.equals_boolean(resume) {
            return;
        }
        state.command_array = None;
        state.resume.set_boolean(resume);
    }

    /// Java `setNice(Number)`.  Set nice.  This value can be set after the
    /// command is built because it comes from the parallel panel and can be
    /// changed for a resume.  Causes commandArray to be set to null.
    pub fn set_nice(&self, nice: Option<Number>) {
        let mut state = self.state.lock().unwrap();
        if state.nice.equals_number(nice) {
            return;
        }
        state.command_array = None;
        state.nice.set_number(nice);
    }

    /// Java `setSubdirName(String)`.
    pub fn set_subdir_name(&self, input: Option<&str>) {
        self.state.lock().unwrap().subdir_name = input.map(str::to_string);
    }

    /// Java `setMultiProc(String)`.
    pub fn set_multi_proc_string(&self, input: Option<&str>) {
        self.state.lock().unwrap().multi_proc.set_string(input);
    }

    /// Java `setMultiProc(int)`.
    pub fn set_multi_proc_int(&self, input: i32) {
        self.state.lock().unwrap().multi_proc.set_int(input);
    }

    /// Java `correctMultiProc(int)`.  The number of jobs cannot exceed the
    /// number of rows that are first run.
    pub fn correct_multi_proc(&self, run_list_size: i32) {
        let mut state = self.state.lock().unwrap();
        if !state.multi_proc.is_null() && state.multi_proc.gt_int(run_list_size) {
            state.multi_proc.set_int(run_list_size.max(2));
        }
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, input: bool) {
        self.state.lock().unwrap().debug = input;
    }

    /// Java `setSubcommandDetails(CommandDetails)`.
    pub fn set_subcommand_details(&mut self, input: Option<Arc<dyn CommandDetails + Send + Sync>>) {
        self.subcommand_details = input;
    }

    // <p>Update done</p>

    /// Java `setQueue(String)`.
    pub fn set_queue(&self, queue: Option<&str>) {
        self.state.lock().unwrap().queue = queue.map(str::to_string);
    }

    /// Java `setQueueCommand(String)`.
    pub fn set_queue_command(&self, command: Option<&str>) {
        self.state.lock().unwrap().queue_command = command.map(str::to_string);
    }

    /// Java `setQueueMode(QueueMode)`.
    pub fn set_queue_mode(&self, queue_mode: Option<QueueMode>) {
        self.state.lock().unwrap().queue_mode = queue_mode;
    }

    /// Java `setInitialize(SubstitutionString)`.
    pub fn set_initialize(&self, input: Option<SubstitutionString>) {
        self.state.lock().unwrap().initialize = input;
    }

    /// Java `setDeinitialize(String)`.
    pub fn set_deinitialize(&self, input: Option<&str>) {
        self.state.lock().unwrap().deinitialize = input.map(str::to_string);
    }

    /// Java `setCoresPerNode(ConstEtomoNumber)`.
    pub fn set_cores_per_node(&self, input: Option<&ConstEtomoNumber>) {
        self.state
            .lock()
            .unwrap()
            .cores_per_node
            .set_const_etomo_number(input);
    }

    /// Java `setCoresPerClusterJob(ConstEtomoNumber)`.
    pub fn set_cores_per_cluster_job(&self, input: Option<&ConstEtomoNumber>) {
        self.state
            .lock()
            .unwrap()
            .cores_per_cluster_job
            .set_const_etomo_number(input);
    }

    /// Java `setGpusPerClusterJob(String)`.
    pub fn set_gpus_per_cluster_job(&self, input: Option<&str>) {
        self.state
            .lock()
            .unwrap()
            .gpus_per_cluster_job
            .set_string(input);
    }

    /// Java `setSecondaryQueueCommand(String)`.
    pub fn set_secondary_queue_command(&self, secondary_queue_command: Option<&str>) {
        self.state.lock().unwrap().secondary_queue_command =
            secondary_queue_command.map(str::to_string);
    }

    /// Java `setSecondaryQueue(String)`.
    pub fn set_secondary_queue(&self, secondary_queue: Option<&str>) {
        self.state.lock().unwrap().secondary_queue = secondary_queue.map(str::to_string);
    }

    /// Java `setSecondaryNumber(String)`.
    pub fn set_secondary_number_string(&self, secondary_number: Option<&str>) {
        self.state
            .lock()
            .unwrap()
            .secondary_number
            .set_string(secondary_number);
    }

    /// Java `setSecondaryNumber(int)`.
    pub fn set_secondary_number_int(&self, secondary_number: i32) {
        self.state
            .lock()
            .unwrap()
            .secondary_number
            .set_int(secondary_number);
    }

    /// Java `resetSecondaryQueue`.
    pub fn reset_secondary_queue(&self) {
        let mut state = self.state.lock().unwrap();
        state.secondary_queue = None;
        state.secondary_queue_command = None;
        state.secondary_number.reset();
    }

    /// Java `setCPUNumber(String)`.
    pub fn set_cpu_number_string(&self, input: Option<&str>) {
        self.state.lock().unwrap().cpu_number.set_string(input);
    }

    /// Java `setGpuProcessing(boolean)`.
    pub fn set_gpu_processing(&self, input: bool) {
        self.state.lock().unwrap().gpu_processing = input;
    }

    /// Java `setCPUNumber(ConstEtomoNumber)`.
    pub fn set_cpu_number_const_etomo_number(&self, input: Option<&ConstEtomoNumber>) {
        self.state
            .lock()
            .unwrap()
            .cpu_number
            .set_const_etomo_number(input);
    }

    /// Java `equalsRootName(ProcessName, AxisID)`.
    ///
    /// ProcesschunksParam.java:363-366 dereference `rootName` and `processName`
    /// without a null test once the first test fails, throwing
    /// NullPointerException for a null root name with a process name, or a
    /// non-blank root name with a null process name.  Fixed in translation: both
    /// are "not equal".
    pub fn equals_root_name(
        &self,
        process_name: Option<ProcessName>,
        axis_id: Option<AxisID>,
    ) -> bool {
        if self.root_name.as_deref().is_none_or(matches_whitespace) && process_name.is_none() {
            return true;
        }
        let (Some(root_name), Some(process_name)) = (&self.root_name, process_name) else {
            return false;
        };
        match axis_id {
            None => *root_name == process_name.to_string(),
            Some(axis_id) => *root_name == format!("{}{}", process_name, axis_id.get_extension()),
        }
    }

    /// Java `getRootName`.
    pub fn get_root_name(&self) -> Option<String> {
        self.root_name.clone()
    }

    /// Java `getResume`, returned as a `ConstEtomoNumber` in the source; the
    /// `EtomoBoolean2` itself keeps its boolean `toString`.
    pub fn get_resume(&self) -> EtomoBoolean2 {
        self.state.lock().unwrap().resume.clone()
    }

    /// Java `isSubdirNameEmpty`.
    pub fn is_subdir_name_empty(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.subdir_name.as_deref().is_none_or(matches_whitespace)
    }

    /// Java `getSubdirName`.
    pub fn get_subdir_name(&self) -> Option<String> {
        self.state.lock().unwrap().subdir_name.clone()
    }

    /// Java `getComputerMap`.
    pub fn get_computer_map(&self) -> HashMap<String, String> {
        let mut computer_monitor_map = HashMap::new();
        let state = self.state.lock().unwrap();
        for (key, element) in state.machine_map.iter() {
            computer_monitor_map.insert(key.clone(), element.get_number().to_string());
        }
        computer_monitor_map
    }

    /// Java `getSecondaryQueue`.
    pub fn get_secondary_queue(&self) -> Option<String> {
        self.state.lock().unwrap().secondary_queue.clone()
    }

    /// Java `resetMachineName`.  Clears machinesNames.  This value can be set
    /// after the command is built because it comes from the parallel panel and
    /// can be changed for a resume.  Causes commandArray to be set to null.
    pub fn reset_machine_name(&self) {
        let mut state = self.state.lock().unwrap();
        if state.machine_map.is_empty() {
            return;
        }
        state.command_array = None;
        state.machine_map.clear();
    }

    /// Java `addMachineName(String, int, String[])`.  Build machineNames and
    /// computerMap.
    ///
    /// The source throws `IllegalStateException("can't change parameter values
    /// after command is built")` once the command exists; that message is the
    /// `Err` here.
    pub fn add_machine_name(
        &self,
        machine_name: &str,
        number: i32,
        device_array: Option<Vec<String>>,
    ) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        if state.command_array.is_some() {
            return Err("can't change parameter values after command is built".to_string());
        }
        if number > 0 {
            // `LinkedHashMap.put` replaces the value of an existing key in place.
            let element = Element::new(number, device_array);
            match state
                .machine_map
                .iter_mut()
                .find(|(key, _)| key == machine_name)
            {
                Some(entry) => entry.1 = element,
                None => state.machine_map.push((machine_name.to_string(), element)),
            }
            // computerMonitorMap.put(machineName, String.valueOf(number));
        }
        Ok(())
    }

    /// Java `setGPUMachineList(String)`.
    pub fn set_gpu_machine_list(&self, input: Option<&str>) {
        let mut gpu_machine_list = String::new();
        // `StringBuilder.append(null)` appends "null".
        gpu_machine_list.push_str(input.unwrap_or("null"));
        self.state.lock().unwrap().gpu_machine_list = Some(gpu_machine_list);
    }

    /// Java `resetGPUMachineList`.
    pub fn reset_gpu_machine_list(&self) {
        self.state.lock().unwrap().gpu_machine_list = None;
    }

    /// Java `addGPUMachine(String, int, String[])`.
    ///
    /// ProcesschunksParam.java:476-478 index `deviceArray` for every `i <
    /// number`, throwing ArrayIndexOutOfBoundsException when it has fewer
    /// devices.  Fixed in translation: the list stops at the last device, as
    /// `buildMachineList` does.
    pub fn add_gpu_machine(
        &self,
        machine: Option<&str>,
        number: i32,
        device_array: Option<&[String]>,
    ) {
        let Some(machine) = machine else {
            return;
        };
        if !matches_whitespace(machine) && number > 0 {
            let mut state = self.state.lock().unwrap();
            let mut first = false;
            if state.gpu_machine_list.is_none() {
                state.gpu_machine_list = Some(String::new());
                first = true;
            }
            let gpu_machine_list = state.gpu_machine_list.as_mut().unwrap();
            if !first {
                gpu_machine_list.push(',');
            }
            gpu_machine_list.push_str(machine);
            if let Some(device_array) = device_array {
                for i in 0..number as usize {
                    if device_array.len() <= i {
                        break;
                    }
                    gpu_machine_list.push_str(&format!("{LIST_DIVIDER}{}", device_array[i]));
                }
            }
        }
    }

    /// Java `getShortCommandName`.
    pub fn get_short_command_name(&self) -> String {
        "pc".to_string()
    }

    /// Java `validate`.
    pub fn validate(&self) -> Option<String> {
        let state = self.state.lock().unwrap();
        if (state.queue_command.is_none() && state.machine_map.is_empty())
            || (state.queue_command.is_some() && state.cpu_number.lt_int(0))
        {
            return Some("No cores where selected.".to_string());
        }
        None
    }

    /// Java `getProcessName` - its body, reading the locked fields, so that
    /// `buildCommand` (which runs under the lock) can call it.
    fn get_process_name_from(state: &State) -> ProcessName {
        if state.cpu_number.is_null() || state.queue_command.is_some() {
            return ProcessName::PROCESSCHUNKS;
        }
        if utilities::is_windows_os() {
            if state.cpu_number.gt_int(56) {
                return ProcessName::PROCHUNKS_CSH;
            }
            return ProcessName::PROCESSCHUNKS;
        }
        if state.cpu_number.gt_int(240) {
            return ProcessName::PROCHUNKS_CSH;
        }
        ProcessName::PROCESSCHUNKS
    }

    /// Java `getProcessName`, non-optional.
    pub fn get_process_name_value(&self) -> ProcessName {
        ProcesschunksParam::get_process_name_from(&self.state.lock().unwrap())
    }

    /// Java private `buildCommand`.
    fn build_command(&self, state: &mut State) {
        state.valid = true;
        let mut command: Vec<String> = Vec::new();
        command.push(ProcesschunksParam::get_process_name_from(state).to_string());
        if state.gpu_processing && (self.mode != Mode::BatchRunTomo) && state.queue.is_none() {
            command.push("-G".to_string());
        }
        if state.resume.is() {
            command.push("-r".to_string());
        }
        if state.queue_command.is_none() {
            command.push("-g".to_string());
            command.push("-n".to_string());
            command.push(state.nice.to_string());
        }
        if let Some(gpu_machine_list) = &state.gpu_machine_list {
            command.push("-p".to_string());
            command.push(gpu_machine_list.clone());
        }
        // Get Node->pcOptionsMap
        let mut node_map = None;
        let pc_option_type;
        if let Some(queue) = &state.queue {
            pc_option_type = PcOptionType::PcOptionTypeQueue;
            let node = Network::get_queue(Some(queue));
            if let Some(node) = node {
                node_map = node.get_pc_options_map();
                if let Some(node_map) = &node_map {
                    for (node_key, node_element) in node_map.iter() {
                        // Java adds a null value as a null element, which
                        // `ProcessBuilder` then rejects with a NullPointerException.
                        // Fixed in translation: an option with no queue value is left
                        // out, as the computer options below are.
                        let Some(node_value) = node_element.get_queue() else {
                            continue;
                        };
                        command.push(format!("-{node_key}"));
                        command.push(node_value);
                    }
                }
            }
        } else {
            pc_option_type = PcOptionType::PcOptionTypeComputer;
        }
        // Get CpuAdoc->pcOptionsMap
        let computer_map = cpu_adoc::INSTANCE.get_pc_options_map();
        if let Some(computer_map) = computer_map {
            for (comp_key, comp_element) in computer_map.iter() {
                let comp_value = comp_element.get_value(pc_option_type);
                if let Some(comp_value) = comp_value {
                    if let Some(node_map) = &node_map {
                        if node_map.iter().any(|(key, _)| key == comp_key) {
                            continue;
                        }
                    }
                    command.push(format!("-{comp_key}"));
                    command.push(comp_value);
                }
            }
        }
        if !state.multi_proc.is_null() {
            command.push(state.multi_proc.get_name().to_string());
            command.push(state.multi_proc.to_string());
        }
        let mut remote_user_dir = None;
        // Batchruntomo with processchunks should only use global remote paths.
        match remote_path::INSTANCE.get_remote_path(
            self.manager,
            self.manager.get_property_user_dir().as_deref(),
            self.axis_id,
            Some(self.mode == Mode::BatchRunTomo),
        ) {
            Ok(path) => remote_user_dir = path,
            Err(e) => {
                // `UIHarness.INSTANCE.openMessageDialog`: the command may be built off
                // the event dispatch thread, so the dialog is posted to it.
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!(
                        "ERROR:  Remote path error.  Unabled to run {}.\n\n{}",
                        ProcesschunksParam::get_process_name_from(state),
                        e.get_message()
                    ),
                    "Processchunks Error".to_string(),
                    Some(self.axis_id),
                );
                state.valid = false;
            }
        }
        let subdir_name_empty = state.subdir_name.as_deref().is_none_or(matches_whitespace);
        if let Some(mut remote_user_dir) = remote_user_dir {
            command.push(WORKING_DIR_OPTION.to_string());
            if !subdir_name_empty {
                remote_user_dir.push(std::path::MAIN_SEPARATOR);
                remote_user_dir.push_str(state.subdir_name.as_deref().unwrap());
            }
            command.push(remote_user_dir);
        }
        // DNM 12/19/21: processchunks now has a different default for queues, so it is
        // inappropriate to send a fixed value here, and there is no need to send a
        // default value anyway
        // command.add("-d");
        // command.add(String.valueOf(DROP_VALUE));
        command.push("-c".to_string());
        let mut commands_file_name = String::new();
        if !subdir_name_empty {
            // commandsFileName.append("\"../");
            commands_file_name.push_str("../");
        }
        // When batchruntomo is run via processchunks, the two processes need to be
        // controlled with separate check files.
        commands_file_name.push_str(&dataset_files::get_commands_file_name(
            state.subdir_name.as_deref(),
            self.root_name.as_deref().unwrap_or("null"),
            Some(if self.mode == Mode::BatchRunTomo {
                BRT_CHECK_NAME_SUFFIX
            } else {
                ""
            }),
        ));
        /* if (!isSubdirNameEmpty()) { commandsFileName.append("\""); } */
        command.push(commands_file_name);
        command.push("-P".to_string());
        if state.queue_command.is_none() {
            // add machine names
            let machine_list = if state
                .premade_machine_list
                .as_deref()
                .is_none_or(matches_whitespace)
            {
                self.build_machine_list(state)
            } else {
                Some(state.premade_machine_list.clone().unwrap())
            };
            if let Some(machine_list) = machine_list {
                command.push(machine_list);
            }
        } else {
            command.push("-Q".to_string());
            // Java adds a null queue as a null element; "null" is how the rest of
            // the program prints it.
            command.push(state.queue.clone().unwrap_or_else(|| "null".to_string()));
            if let Some(initialize) = &state.initialize {
                command.push("-I".to_string());
                let mut cpus = 1;
                if !state.cpu_number.is_null() {
                    cpus = state.cpu_number.get_int();
                }
                let i_cores_per_node = state.cores_per_node.get_int();
                // ProcesschunksParam.java:787 divides by `coresPerNode`, throwing
                // ArithmeticException when it was set to 0.  Fixed in translation:
                // zero cores per node is treated as one.
                let nodes = cpus
                    .wrapping_add(i_cores_per_node)
                    .wrapping_sub(1)
                    .checked_div(i_cores_per_node)
                    .unwrap_or(cpus);
                command.push(
                    initialize
                        .substitute(nodes)
                        .unwrap_or_else(|| "null".to_string()),
                );
            }
            if let Some(deinitialize) = &state.deinitialize {
                command.push("-D".to_string());
                command.push(deinitialize.clone());
            }
            command.push(state.cpu_number.get_name().to_string());
            command.push(state.cpu_number.to_string());
            // Mode 2 and 2c
            if let Some(queue_mode) = state.queue_mode
                && queue_mode.is_type(QueueType::Node)
            {
                command.push(state.cores_per_cluster_job.get_name().to_string());
                command.push(state.cores_per_cluster_job.to_string());
                // Mode 2
                if queue_mode == QueueMode::NodeWithGpu {
                    if !state.gpus_per_cluster_job.is_null() {
                        command.push(state.gpus_per_cluster_job.get_name().to_string());
                        command.push(state.gpus_per_cluster_job.to_string());
                    }
                }
            }
            if let Some(secondary_queue_command) = &state.secondary_queue_command {
                command.push("-SQ".to_string());
                command.push(secondary_queue_command.clone());
                if !state.secondary_number.is_null() {
                    command.push(state.secondary_number.get_name().to_string());
                    command.push(state.secondary_number.to_string());
                }
            }
            // command.add("\"" + queueCommand + "\"");
            command.push(state.queue_command.clone().unwrap());
        }
        command.push(self.root_name.clone().unwrap_or_else(|| "null".to_string()));
        let command_size = command.len();
        let mut command_array = Vec::with_capacity(command_size);
        for i in 0..command_size {
            command_array.push(command[i].clone());
        }
        if state.debug {
            for i in 0..command_array.len() {
                if i > 0 {
                    eprint!(" ");
                }
                eprint!("{}", command_array[i]);
            }
            if !command_array.is_empty() {
                eprintln!();
            }
        }
        state.command_array = Some(command_array);
    }

    /// Java private `buildMachineList`.  Builds and returns a string version of
    /// machineMap where each entry key (the machine name) is repeated a number of
    /// times equals to the entry value (# CPUs).
    /// CPU: comp,comp,comp
    /// GPU: comp or comp:4:1:2
    fn build_machine_list(&self, state: &State) -> Option<String> {
        if state.machine_map.is_empty() {
            return None;
        }
        let mut machine_list = String::new();
        let len = state.machine_map.len();
        for (index, (machine, element)) in state.machine_map.iter().enumerate() {
            let has_next = index + 1 < len;
            let number = element.get_number();
            if number <= 0 {
                continue;
            }
            machine_list.push_str(machine);
            let device_array = element.get_device_array();
            match device_array {
                Some(device_array) if state.gpu_processing => {
                    for i in 0..number as usize {
                        if device_array.len() <= i {
                            break;
                        }
                        machine_list.push(':');
                        machine_list.push_str(&device_array[i]);
                    }
                }
                _ => {
                    for _ in 1..number {
                        machine_list.push(',');
                        machine_list.push_str(machine);
                    }
                }
            }
            if has_next {
                machine_list.push(',');
            }
        }
        Some(machine_list)
    }

    /// Java private `backSlashSpaces(String)`.  Put a back slash in front of each
    /// space in directoryPath.
    ///
    /// ProcesschunksParam.java:889 returns from inside the loop after the first
    /// space, so only the first space of a path was escaped and a path with two
    /// spaces broke the run file.  Fixed in translation: the loop escapes every
    /// space, as the method's comment describes.
    fn back_slash_spaces(directory_path: Option<&str>) -> Option<String> {
        let mut directory_path = directory_path?.to_string();
        // see if directory path has any spaces
        let mut space_index = directory_path.find(' ');
        while let Some(index) = space_index {
            // find each space and add a backslash to it
            directory_path = format!(
                "{}\\ {}",
                &directory_path[..index],
                &directory_path[index + 1..]
            );
            let starting_index = index + 2;
            if starting_index >= directory_path.len() {
                break;
            }
            space_index = directory_path[starting_index..]
                .find(' ')
                .map(|found| found + starting_index);
        }
        Some(directory_path)
    }

    /// Java `reorderComputerMapGpuFirst`.
    ///
    /// `tempMachineMap` is `new HashMap<String, Element>(machineMap)`, so the
    /// computers are tested, and the rest re-added after the first GPU computer, in
    /// Java's `HashMap` order (`JavaHashMap`), not in the machine map's own order.
    pub fn reorder_computer_map_gpu_first(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        let mut temp_machine_map: JavaHashMap<String, Element> =
            JavaHashMap::from_map(state.machine_map.iter().cloned());
        let machine_set: Vec<(String, Element)> = temp_machine_map
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let mut gpu_found = false;
        for machine in machine_set {
            if Network::is_gpu_available(
                &machine.0,
                self.manager,
                self.axis_id,
                self.manager.get_property_user_dir().as_deref(),
            ) {
                state.machine_map.clear();
                temp_machine_map.remove(&machine.0);

                state.machine_map.push(machine);
                // `machineMap.putAll(tempMachineMap)`: a `LinkedHashMap` appends
                // the new keys in `tempMachineMap`'s iteration order.
                state.machine_map.extend(
                    temp_machine_map
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone())),
                );
                gpu_found = true;
                break;
            }
        }
        gpu_found
    }
}

impl Command for ProcesschunksParam {
    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        Some(self.get_process_name_value().to_string())
    }

    /// Java `getCommandMode`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    /// Java `isMessageReporter`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getCommandOutputFile`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandInputFile`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getSubcommandDetails`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        match &self.subcommand_details {
            None => None,
            Some(details) => Some(details.as_ref() as &dyn CommandDetails),
        }
    }

    /// Java `getSubcommandProcessName`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        self.subcommand_process_name.clone()
    }

    /// Java `getAxisID`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(self.get_process_name_value().to_string())
    }

    /// Java deprecated `getOutputImageFileType`.
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        if self.output_image_file_key.is_none() {
            if let Some(subcommand_details) = &self.subcommand_details {
                return subcommand_details.get_output_image_file_type();
            }
        }
        if let Some(OutputImageFileKey::FileType(file_type)) = &self.output_image_file_key {
            return Some(Arc::clone(file_type));
        }
        None
    }

    /// Java `getOutputImageFileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        if self.output_image_file_key.is_none() {
            if let Some(subcommand_details) = &self.subcommand_details {
                return subcommand_details.get_output_image_file_key();
            }
        }
        match &self.output_image_file_key {
            None => None,
            Some(OutputImageFileKey::FileKey(file_key)) => Some(file_key.clone()),
            // `FileType` is a `FileKey` subclass: its `FileKey` part.
            Some(OutputImageFileKey::FileType(file_type)) => Some((***file_type).clone()),
        }
    }

    /// Java deprecated `getOutputImageFileType2`.
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        if let Some(subcommand_details) = &self.subcommand_details {
            return subcommand_details.get_output_image_file_type2();
        }
        if let Some(OutputImageFileKey::FileType(file_type)) = &self.output_image_file_key {
            return Some(Arc::clone(file_type));
        }
        None
    }

    /// Java `getOutputImageFileKey2`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        if let Some(subcommand_details) = &self.subcommand_details {
            return subcommand_details.get_output_image_file_key2();
        }
        None
    }

    /// Java `getCommandLine`.
    fn get_command_line(&self) -> Option<String> {
        let command_array = self.get_command_array();
        let command_array = match command_array {
            Some(command_array) if !command_array.is_empty() => command_array,
            _ => return None,
        };
        let mut buffer = command_array[0].clone();
        for i in 1..command_array.len() {
            buffer.push(' ');
            buffer.push_str(&command_array[i]);
        }
        Some(buffer)
    }

    /// Java `getCommandArray`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        let mut state = self.state.lock().unwrap();
        if state.command_array.is_none() {
            self.build_command(&mut state);
        }
        state.command_array.clone()
    }

    /// Java `getProcessName`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(self.get_process_name_value())
    }

    /// The `CommandDetails` view: this class implements `ProcessDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl DetachedCommandDetails for ProcesschunksParam {
    /// Java `getCommandString`.  Gets a string version of the command array
    /// which can be used safely, even if there are embedded spaces in the
    /// directory paths, because the directory path spaces have been
    /// back-slashed.  The command file is already quoted and doesn't need a
    /// backslash this is because in Windows the command contains backslashes for
    /// the path and has to be quoted.
    fn get_command_string(&self) -> Option<String> {
        let command_array = self.get_command_array()?;
        let mut buffer = String::new();
        let mut found_dir = false;
        for i in 0..command_array.len() {
            let mut command = command_array[i].clone();
            // add back slashes to the spaces in any directory path
            let mut found_dir_option = false;
            if command == WORKING_DIR_OPTION {
                // found an option which takes a directory path
                found_dir_option = true;
                found_dir = true;
            }
            if !found_dir_option && found_dir {
                // add back slashes to the spaces in this directory path
                found_dir = false;
                command = ProcesschunksParam::back_slash_spaces(Some(&command)).unwrap();
            }
            // add each option to the buffer
            if i == 0 {
                buffer.push_str(&command);
            } else {
                buffer.push(' ');
                buffer.push_str(&command);
            }
        }
        Some(buffer)
    }

    /// Java `isValid`.
    fn is_valid(&self) -> bool {
        self.state.lock().unwrap().valid
    }

    /// Java `isCommandNiced`.  Niced when running on a queue.  CommandNice is not
    /// the same as the nice parameter.
    fn is_command_niced(&self) -> bool {
        self.state.lock().unwrap().queue_command.is_some()
    }

    /// Java `getNiceCommand`.  Returns nice command for a queue.
    fn get_nice_command(&self) -> Option<String> {
        if self.state.lock().unwrap().queue_command.is_none() {
            return Some(String::new());
        }
        Some("nice +18".to_string())
    }
}

impl ParallelParam for ProcesschunksParam {
    /// Java `getSubcommandMode`.
    fn get_subcommand_mode(&self) -> Option<&dyn CommandMode> {
        match &self.subcommand_mode {
            None => None,
            Some(mode) => Some(mode.as_ref() as &dyn CommandMode),
        }
    }
}

impl Loggable for ProcesschunksParam {
    /// Java `getLogMessage`, which returns null: nothing to log.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }

    /// Java `getName`.
    fn get_name(&self) -> String {
        self.get_process_name_value().to_string()
    }
}

/// Every Java `ProcessDetails` getter throws
/// `IllegalArgumentException("field=" + field)`; that is `None` here.
impl ProcessDetails for ProcesschunksParam {
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        None
    }

    fn get_iterator_element_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList> {
        None
    }

    fn get_boolean_value(&self, _field: &dyn FieldInterface) -> Option<bool> {
        None
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    fn get_hashtable(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<super::process_details::Hashtable> {
        None
    }

    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }

    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    fn get_int_key_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::int_key_list::IntKeyList> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::ProcesschunksParam;

    #[test]
    fn back_slash_spaces_escapes_every_space() {
        assert_eq!(
            ProcesschunksParam::back_slash_spaces(Some("/a b/c d")).as_deref(),
            Some("/a\\ b/c\\ d")
        );
        assert_eq!(
            ProcesschunksParam::back_slash_spaces(Some("/ab")).as_deref(),
            Some("/ab")
        );
        assert_eq!(ProcesschunksParam::back_slash_spaces(None), None);
    }
}
