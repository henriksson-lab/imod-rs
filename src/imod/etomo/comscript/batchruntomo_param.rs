//! `IMOD/Etomo/src/etomo/comscript/BatchruntomoParam.java`.
//!
//! Parameters for batchruntomo.  In batch mode the parameters go into the
//! batchruntomo command file; in validation and rename mode the param runs
//! `python -u <IMOD bin>/batchruntomo ...` itself through `SystemProgram`
//! (`run`).
//!
//! **Shape.**  `updateComScriptCommand` changes the param in the source (it
//! resets the translate-path lists, sets `namingStyle` and `remoteDirectory`,
//! and can clear `valid`), while the Rust `CommandParam` contract takes
//! `&self`; those four fields are therefore behind a `Mutex`/atomic.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use regex::Regex;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::{self, CommandMode};
use super::command_param::{CommandParam, ParseComScriptError};
use super::string_list::StringList;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::process_messages::ProcessMessages;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::storage::directive_file::DirectiveFile;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Number, Type, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::extension::Extension;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::queue_mode::QueueMode;
use crate::imod::etomo::r#type::queue_type::QueueType;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::remote_path;
use crate::imod::etomo::util::utilities::{java_io_file_get_absolute_path, java_lang_string_split};

/// Java private static `VALIDATION_TYPE_BATCH_DIRECTIVE`.
const VALIDATION_TYPE_BATCH_DIRECTIVE: i32 = 1;
/// Java private static `VALIDATION_TYPE_TEMPLATE`.
const VALIDATION_TYPE_TEMPLATE: i32 = 2;
/// Java `CPU_MACHINE_LIST_TAG`.
pub const CPU_MACHINE_LIST_TAG: &str = "CPUMachineList";
/// Java `GPU_MACHINE_LIST_TAG`.
pub const GPU_MACHINE_LIST_TAG: &str = "GPUMachineList";
/// Java `MACHINE_LIST_LOCAL_VALUE`.
pub const MACHINE_LIST_LOCAL_VALUE: &str = "1";
/// Java `DELIVER_TO_DIRECTORY_TAG`.
pub const DELIVER_TO_DIRECTORY_TAG: &str = "DeliverToDirectory";
/// Java `EMAIL_ADDRESS_TAG`.
pub const EMAIL_ADDRESS_TAG: &str = "EmailAddress";
/// Java `ROOT_NAME_TAG`.
pub const ROOT_NAME_TAG: &str = "RootName";
/// Java `ENDING_STEP_TAG`.
pub const ENDING_STEP_TAG: &str = "EndingStep";
/// Java `STARTING_STEP_TAG`.
pub const STARTING_STEP_TAG: &str = "StartingStep";
/// Java `MAKE_SUB_DIRECTORY_TAG`.
pub const MAKE_SUB_DIRECTORY_TAG: &str = "MakeSubDirectory";
/// Java private static `CPU_MACHINE_LIST_DIVIDER` (deprecated).
const CPU_MACHINE_LIST_DIVIDER: &str = "#";
/// Java private static `LIST_DIVIDER`.
const LIST_DIVIDER: &str = ":";
/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::BATCHRUNTOMO;
/// Java `USE_EXISTING_ALIGNMENT_TAG`.
pub const USE_EXISTING_ALIGNMENT_TAG: &str = "UseExistingAlignment";
// values
/// Java `ROOT_NAME_PREFIX`.
pub const ROOT_NAME_PREFIX: &str = "batch";
/// Java `SharedConstants.PARAMETER_PREFIX`.
const PARAMETER_PREFIX: char = '-';

/// Java `CHECK_FILE_VALUE = FileType.CHECK_FILE`.
pub fn check_file_value() -> Arc<FileType> {
    file_type::CLASS.check_file.clone()
}

/// Java public static final nested class `BatchruntomoParam.Mode`.  The
/// source's `BATCH` is private; `getCommandMode` still hands it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `VALIDATION`.
    Validation,
    /// Java private `BATCH`.
    Batch,
    /// Java `RENAME`.
    Rename,
}

/// The source has no `toString`; the variant name stands in for
/// `Object.toString()`'s identity string.
impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::Validation => "Validation",
            Mode::Batch => "Batch",
            Mode::Rename => "Rename",
        })
    }
}

impl CommandMode for Mode {}

/// Java private static final nested class `InterleavedIndex`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InterleavedIndex {
    index: usize,
    tag: &'static str,
}

impl InterleavedIndex {
    // validation and batch
    /// Java `DIRECTIVE_FILE`.
    const DIRECTIVE_FILE: InterleavedIndex = InterleavedIndex {
        index: 0,
        tag: "DirectiveFile",
    };
    // batch only
    /// Java `ROOT_NAME`.
    const ROOT_NAME: InterleavedIndex = InterleavedIndex {
        index: 1,
        tag: ROOT_NAME_TAG,
    };
    /// Java `CURRENT_LOCATION`.
    const CURRENT_LOCATION: InterleavedIndex = InterleavedIndex {
        index: 2,
        tag: "CurrentLocation",
    };

    /// Java `VALIDATION_LENGTH`.
    const VALIDATION_LENGTH: usize = 1;
    /// Java `BATCH_LENGTH`.
    const BATCH_LENGTH: usize = 3;

    /// Java private static `getTags(CommandMode)`.
    fn get_tags(mode: Option<&dyn CommandMode>) -> Option<Vec<Option<String>>> {
        if command_mode::equals_mode(mode, &Mode::Validation) {
            return Some(vec![Some(InterleavedIndex::DIRECTIVE_FILE.tag.to_owned())]);
        }
        if command_mode::equals_mode(mode, &Mode::Batch) {
            return Some(vec![
                Some(InterleavedIndex::DIRECTIVE_FILE.tag.to_owned()),
                Some(InterleavedIndex::ROOT_NAME.tag.to_owned()),
                Some(InterleavedIndex::CURRENT_LOCATION.tag.to_owned()),
            ]);
        }
        None
    }
}

/// Java `BatchruntomoParam`.
pub struct BatchruntomoParam {
    validation_type: EtomoNumber,

    deliver_to_directory: StringParameter,
    nice_value: StringParameter,
    email_address: StringParameter,
    ending_step: ScriptParameter,
    starting_step: ScriptParameter,
    smtp_server: StringParameter,
    use_existing_alignment: EtomoBoolean2,
    make_sub_directory: EtomoBoolean2,
    etomo_debug: ScriptParameter,
    /// Set by `updateComScriptCommand`.
    naming_style: Mutex<ScriptParameter>,
    // Rename parameters
    root_name: StringParameter,
    axis_of_extension: ScriptParameter,
    stack_extension: StringParameter,
    // for mode 1
    queue_command: StringParameter,
    // for mode 2 and 2c
    max_jobs_on_queue: ScriptParameter,
    cores_per_cluster_job: ScriptParameter,
    gpus_per_cluster_job: ScriptParameter,
    // for mode 3
    gpu_queue_command: StringParameter,
    max_gpu_jobs_on_queue: ScriptParameter,
    /// Reset and set by `updateComScriptCommand`.
    translate_paths_from_array: Mutex<StringList>,
    /// Reset and set by `updateComScriptCommand`.
    translate_paths_to_array: Mutex<StringList>,
    /// Set by `updateComScriptCommand`.
    remote_directory: Mutex<StringParameter>,

    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    mode: Mode,
    /// Holds the parameters that should be clustered together for each dataset.
    /// See InterleavedIndex.
    interleaved_parameters: Vec<Vec<String>>,
    for_update: bool,
    parallel_processing: bool,

    batchruntomo: Option<Arc<SystemProgram>>,
    /// Java field `exitValue`; `run` declares a local of the same name and never
    /// assigns the field.
    #[allow(dead_code)]
    exit_value: i32,
    /// Cleared by `updateComScriptCommand` on a mount-rule error.
    valid: AtomicBool,
    cpu_machine_list: Option<String>,
    gpu_machine_list: Option<String>,
    delivered_location_validation_set: Option<HashSet<String>>,
    root_name_validation_set: Option<HashSet<String>>,
    queue_mode: Option<QueueMode>,
}

impl BatchruntomoParam {
    /// Java private `BatchruntomoParam(BaseManager, AxisID, CommandMode, boolean,
    /// boolean)`.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        mode: Mode,
        for_update: bool,
        parallel_processing: bool,
    ) -> BatchruntomoParam {
        let mut translate_paths_from_array = StringList::new_with_key(Some("TranslatePathsFrom"));
        let mut translate_paths_to_array = StringList::new_with_key(Some("TranslatePathsTo"));
        translate_paths_from_array.set_successive_entries_accumulate();
        translate_paths_to_array.set_successive_entries_accumulate();
        let mut interleaved_parameters: Vec<Vec<String>>;
        if mode == Mode::Batch {
            interleaved_parameters = vec![Vec::new(); InterleavedIndex::BATCH_LENGTH];
            interleaved_parameters[InterleavedIndex::ROOT_NAME.index] = Vec::new();
            interleaved_parameters[InterleavedIndex::CURRENT_LOCATION.index] = Vec::new();
        } else {
            interleaved_parameters = vec![Vec::new(); InterleavedIndex::VALIDATION_LENGTH];
        }
        interleaved_parameters[InterleavedIndex::DIRECTIVE_FILE.index] = Vec::new();
        let mut etomo_debug = ScriptParameter::new_with_name("EtomoDebug");
        // `EtomoDirector.INSTANCE.getArguments().getDebugLevel()` is never null here.
        let debug_level = etomo_director::ARGUMENTS.lock().unwrap().get_debug_level();
        etomo_debug.set_display_value_int(debug_level.get_value());
        BatchruntomoParam {
            validation_type: EtomoNumber::new(),
            deliver_to_directory: StringParameter::new(DELIVER_TO_DIRECTORY_TAG),
            nice_value: StringParameter::new("NiceValue"),
            email_address: StringParameter::new(EMAIL_ADDRESS_TAG),
            ending_step: ScriptParameter::new_with_type_and_name(Type::Double, ENDING_STEP_TAG),
            starting_step: ScriptParameter::new_with_type_and_name(Type::Double, STARTING_STEP_TAG),
            smtp_server: StringParameter::new("SMTPserver"),
            use_existing_alignment: EtomoBoolean2::new_with_name(USE_EXISTING_ALIGNMENT_TAG),
            make_sub_directory: EtomoBoolean2::new_with_name(MAKE_SUB_DIRECTORY_TAG),
            etomo_debug,
            naming_style: Mutex::new(ScriptParameter::new_with_name("NamingStyle")),
            root_name: StringParameter::new("RootName"),
            axis_of_extension: ScriptParameter::new_with_name("AxisOfExtension"),
            stack_extension: StringParameter::new("StackExtension"),
            queue_command: StringParameter::new("QueueCommand"),
            max_jobs_on_queue: ScriptParameter::new_with_name("MaxJobsOnQueue"),
            cores_per_cluster_job: ScriptParameter::new_with_name("CoresPerClusterJob"),
            gpus_per_cluster_job: ScriptParameter::new_with_name("GPUsPerClusterJob"),
            gpu_queue_command: StringParameter::new("GPUQueueCommand"),
            max_gpu_jobs_on_queue: ScriptParameter::new_with_name("MaxGPUJobsOnQueue"),
            translate_paths_from_array: Mutex::new(translate_paths_from_array),
            translate_paths_to_array: Mutex::new(translate_paths_to_array),
            remote_directory: Mutex::new(StringParameter::new("RemoteDirectory")),
            manager,
            axis_id,
            mode,
            interleaved_parameters,
            for_update,
            parallel_processing,
            batchruntomo: None,
            exit_value: -1,
            valid: AtomicBool::new(true),
            cpu_machine_list: None,
            gpu_machine_list: None,
            delivered_location_validation_set: None,
            root_name_validation_set: None,
            queue_mode: None,
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        parallel_processing: bool,
    ) -> BatchruntomoParam {
        BatchruntomoParam::new(manager, axis_id, Mode::Batch, false, parallel_processing)
    }

    /// Java static `getInstanceForUpdate`.
    pub fn get_instance_for_update(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        parallel_processing: bool,
    ) -> BatchruntomoParam {
        BatchruntomoParam::new(manager, axis_id, Mode::Batch, true, parallel_processing)
    }

    /// Java static `getValidationInstance`.
    pub fn get_validation_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> BatchruntomoParam {
        BatchruntomoParam::new(manager, axis_id, Mode::Validation, false, false)
    }

    /// Java static `getRenameInputImageFilesInstance`.
    pub fn get_rename_input_image_files_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> BatchruntomoParam {
        BatchruntomoParam::new(manager, axis_id, Mode::Rename, false, false)
    }

    /// Java `addDirectiveFile(File)`.
    pub fn add_directive_file(&mut self, directive_file: Option<&Path>) {
        if let Some(directive_file) = directive_file {
            self.interleaved_parameters[InterleavedIndex::DIRECTIVE_FILE.index].push(
                java_io_file_get_absolute_path(&directive_file.to_string_lossy()),
            );
        }
    }

    /// Java `resetCPUMachineList`.
    pub fn reset_cpu_machine_list(&mut self) {
        self.cpu_machine_list = None;
    }

    /// Java `addCPUMachine`.
    pub fn add_cpu_machine(&mut self, machine: Option<&str>, number: i32) {
        if let Some(machine) = machine
            && !java_lang_string_matches_whitespace(machine)
            && number > 0
        {
            let mut first = false;
            if self.cpu_machine_list.is_none() {
                self.cpu_machine_list = Some(String::new());
                first = true;
            }
            let cpu_machine_list = self.cpu_machine_list.as_mut().unwrap();
            if !first {
                cpu_machine_list.push(',');
            }
            cpu_machine_list.push_str(&format!("{machine}{LIST_DIVIDER}{number}"));
        }
    }

    /// Java `getCPUMachineMap`.
    pub fn get_cpu_machine_map(&self) -> Option<HashMap<String, String>> {
        if let Some(cpu_machine_list) = &self.cpu_machine_list {
            return BatchruntomoParam::convert_to_machine_map(Some(cpu_machine_list), false);
        }
        None
    }

    /// Java `getGPUMachineMap`.
    pub fn get_gpu_machine_map(&self) -> Option<HashMap<String, String>> {
        if let Some(gpu_machine_list) = &self.gpu_machine_list {
            return BatchruntomoParam::convert_to_machine_map(Some(gpu_machine_list), true);
        }
        None
    }

    /// Java private static `convertToMachineMap`.  Convert a CPU or GPU machine
    /// list string to a map containing computer and # of PUs.  Returns `None` if
    /// this is not a parallel processing list.
    fn convert_to_machine_map(
        machine_list: Option<&str>,
        gpu: bool,
    ) -> Option<HashMap<String, String>> {
        if let Some(machine_list) = machine_list
            && !machine_list.is_empty()
        {
            let list = machine_list.to_owned();
            if list != MACHINE_LIST_LOCAL_VALUE {
                let machine_array = java_lang_string_split(machine_list, &Regex::new(",").unwrap());
                if !machine_array.is_empty() {
                    let mut machine_map: HashMap<String, String> = HashMap::new();
                    let mut divider: Option<&str> = None;
                    for i in 0..machine_array.len() {
                        let mut machine: Option<Vec<String>> = None;
                        // Try to set the divider. Each element will use the same, or no
                        // divider.
                        if let Some(divider) = divider {
                            machine = Some(java_lang_string_split(
                                &machine_array[i],
                                &Regex::new(&regex::escape(divider)).unwrap(),
                            ));
                        } else if machine_array[i].contains(CPU_MACHINE_LIST_DIVIDER) {
                            divider = Some(CPU_MACHINE_LIST_DIVIDER);
                            machine = Some(java_lang_string_split(
                                &machine_array[i],
                                &Regex::new(&regex::escape(CPU_MACHINE_LIST_DIVIDER)).unwrap(),
                            ));
                        } else if machine_array[i].contains(LIST_DIVIDER) {
                            divider = Some(LIST_DIVIDER);
                            machine = Some(java_lang_string_split(
                                &machine_array[i],
                                &Regex::new(&regex::escape(LIST_DIVIDER)).unwrap(),
                            ));
                        }
                        match machine {
                            None => {
                                machine_map.insert(machine_array[i].clone(), "1".to_owned());
                            }
                            Some(machine) => {
                                if machine.len() > 1 {
                                    if gpu {
                                        // for gpu put in the number of gpu ids.
                                        // lupin:2,akira:1:2
                                        machine_map.insert(
                                            machine[0].clone(),
                                            (machine.len() as i32 - 1).to_string(),
                                        );
                                    } else {
                                        // for cpu, there should be only one number - the
                                        // number of CPUs.
                                        // ranma:4,kiki:4
                                        machine_map.insert(machine[0].clone(), machine[1].clone());
                                    }
                                } else if let Some(name) = machine.first() {
                                    machine_map.insert(name.clone(), "1".to_owned());
                                }
                                // BatchruntomoParam.java:415 reads `machine[0]` even when
                                // the split left nothing (an element that is only
                                // dividers, e.g. ":"), which throws
                                // ArrayIndexOutOfBoundsException.  Fixed in translation:
                                // such an element names no machine and is skipped.
                            }
                        }
                    }
                    return Some(machine_map);
                }
            }
        }
        None
    }

    /// Java `resetGPUMachineList`.
    pub fn reset_gpu_machine_list(&mut self) {
        self.gpu_machine_list = None;
    }

    /// Java `addGPUMachine`.
    pub fn add_gpu_machine(
        &mut self,
        machine: Option<&str>,
        number: i32,
        device_array: Option<&[String]>,
    ) {
        if let Some(machine) = machine
            && !java_lang_string_matches_whitespace(machine)
            && number > 0
        {
            let mut first = false;
            if self.gpu_machine_list.is_none() {
                self.gpu_machine_list = Some(String::new());
                first = true;
            }
            let gpu_machine_list = self.gpu_machine_list.as_mut().unwrap();
            if !first {
                gpu_machine_list.push(',');
            }
            gpu_machine_list.push_str(machine);
            if let Some(device_array) = device_array {
                for i in 0..number as usize {
                    // BatchruntomoParam.java:442 indexes `deviceArray[i]` for every
                    // `i < number`, which throws ArrayIndexOutOfBoundsException when
                    // the array is shorter.  Fixed in translation: only the devices
                    // present are listed.
                    let Some(device) = device_array.get(i) else {
                        break;
                    };
                    gpu_machine_list.push_str(&format!("{LIST_DIVIDER}{device}"));
                }
            }
        }
    }

    /// Java `setAxisOfExtension`.
    pub fn set_axis_of_extension(&mut self, axis_id: AxisID) {
        self.axis_of_extension
            .set_int(axis_id.get_axis_of_extension());
    }

    /// Java `setQueueCommand`.
    pub fn set_queue_command(&mut self, queue_command: Option<&str>) {
        self.queue_command.set(queue_command);
    }

    /// Java `resetQueueCommand`.
    pub fn reset_queue_command(&mut self) {
        self.queue_command.reset();
    }

    /// Java `setMaxJobsOnQueue(String)`.
    pub fn set_max_jobs_on_queue_string(&mut self, max_jobs_on_queue: Option<&str>) {
        self.max_jobs_on_queue.set_string(max_jobs_on_queue);
    }

    /// Java `setMaxJobsOnQueue(int)`.
    pub fn set_max_jobs_on_queue_int(&mut self, max_jobs_on_queue: i32) {
        self.max_jobs_on_queue.set_int(max_jobs_on_queue);
    }

    /// Java `setMaxJobsOnQueue(ConstEtomoNumber)`.
    pub fn set_max_jobs_on_queue_const_etomo_number(
        &mut self,
        max_jobs_on_queue: Option<&ConstEtomoNumber>,
    ) {
        self.max_jobs_on_queue
            .set_const_etomo_number(max_jobs_on_queue);
    }

    /// Java `setCoresPerClusterJob`.
    pub fn set_cores_per_cluster_job(&mut self, cores_per_cluster_job: Option<&ConstEtomoNumber>) {
        self.cores_per_cluster_job
            .set_const_etomo_number(cores_per_cluster_job);
    }

    /// Java `resetCoresPerClusterJob`.
    pub fn reset_cores_per_cluster_job(&mut self) {
        self.cores_per_cluster_job.reset();
    }

    /// Java `setGPUsPerClusterJob`.
    pub fn set_gpus_per_cluster_job(&mut self, gpus_per_cluster_job: Option<&str>) {
        self.gpus_per_cluster_job.set_string(gpus_per_cluster_job);
    }

    /// Java `resetGPUsPerClusterJob`.
    pub fn reset_gpus_per_cluster_job(&mut self) {
        self.gpus_per_cluster_job.reset();
    }

    /// Java `setGPUQueueCommand`.
    pub fn set_gpu_queue_command(&mut self, gpu_queue_command: Option<&str>) {
        self.gpu_queue_command.set(gpu_queue_command);
    }

    /// Java `setMaxGPUJobsOnQueue(String)`.
    pub fn set_max_gpu_jobs_on_queue_string(&mut self, max_gpu_jobs_on_queue: Option<&str>) {
        self.max_gpu_jobs_on_queue.set_string(max_gpu_jobs_on_queue);
    }

    /// Java `setMaxGPUJobsOnQueue(int)`.
    pub fn set_max_gpu_jobs_on_queue_int(&mut self, max_gpu_jobs_on_queue: i32) {
        self.max_gpu_jobs_on_queue.set_int(max_gpu_jobs_on_queue);
    }

    /// Java `resetSecondaryQueue`.
    pub fn reset_secondary_queue(&mut self) {
        self.gpu_queue_command.reset();
        self.max_gpu_jobs_on_queue.reset();
    }

    /// Java `setQueueMode`.
    pub fn set_queue_mode(&mut self, queue_mode: Option<QueueMode>) {
        self.queue_mode = queue_mode;
    }

    /// Java `setRootName`.
    pub fn set_root_name(&mut self, root_name: Option<&str>) {
        self.root_name.set(root_name);
    }

    /// Java `setStackExtension`.
    pub fn set_stack_extension(&mut self, extension: Option<&Extension>) {
        if let Some(extension) = extension {
            self.stack_extension.set(Some(&extension.to_string()));
        } else {
            self.stack_extension.reset();
        }
    }

    /// Java `setNamingStyle`.
    pub fn set_naming_style(&mut self, image_filename_style: ImageFilenameStyle) {
        self.naming_style
            .get_mut()
            .unwrap()
            .set_const_etomo_number(Some(&image_filename_style.get_value()));
    }

    /// Java `setNiceValue(Number)`.
    pub fn set_nice_value(&mut self, input: Option<Number>) {
        match input {
            None => self.nice_value.reset(),
            Some(input) => self.nice_value.set(Some(&input.to_string())),
        }
    }

    /// Java `getNiceValue`.
    pub fn get_nice_value(&self) -> String {
        self.nice_value.to_string()
    }

    /// Java `isDeliverToDirectorySet`.
    pub fn is_deliver_to_directory_set(&self) -> bool {
        !self.deliver_to_directory.is_empty()
    }

    /// Java `getDeliverToDirectory`.
    pub fn get_deliver_to_directory(&self) -> String {
        self.deliver_to_directory.to_string()
    }

    /// Java `isMakeSubDirectory`.
    pub fn is_make_sub_directory(&self) -> bool {
        self.make_sub_directory.is()
    }

    /// Java `setMakeSubDirectory`.
    pub fn set_make_sub_directory(&mut self, input: bool) {
        self.make_sub_directory.set_boolean(input);
        self.deliver_to_directory.reset();
    }

    /// Java `getEndingStep`.
    pub fn get_ending_step(&self) -> String {
        self.ending_step.to_string()
    }

    /// Java `isEndingStepSet`.
    pub fn is_ending_step_set(&self) -> bool {
        !self.ending_step.is_null()
    }

    /// Java `getStartingStep`.
    pub fn get_starting_step(&self) -> String {
        self.starting_step.to_string()
    }

    /// Java `setDeliverToDirectory(File)`.
    pub fn set_deliver_to_directory(&mut self, input: Option<&Path>) {
        if let Some(input) = input {
            self.deliver_to_directory
                .set(Some(&java_io_file_get_absolute_path(
                    &input.to_string_lossy(),
                )));
        }
        self.make_sub_directory.reset();
    }

    /// Java `setEndingStep`.
    pub fn set_ending_step(&mut self, input: Option<&ConstEtomoNumber>) {
        self.ending_step.set_const_etomo_number(input);
    }

    /// Java `resetEndingStep`.
    pub fn reset_ending_step(&mut self) {
        self.ending_step.reset();
    }

    /// Java `setStartingStep`.
    pub fn set_starting_step(&mut self, input: Option<&ConstEtomoNumber>) {
        self.starting_step.set_const_etomo_number(input);
    }

    /// Java `setUseExistingAlignment`.
    pub fn set_use_existing_alignment(&mut self, input: bool) {
        self.use_existing_alignment.set_boolean(input);
    }

    /// Java `resetStartingStep`.
    pub fn reset_starting_step(&mut self) {
        self.starting_step.reset();
    }

    /// Java `resetDeliver`.
    pub fn reset_deliver(&mut self) {
        self.deliver_to_directory.reset();
        self.make_sub_directory.reset();
    }

    /// Java `addRootName`.  `deliver_to_directory`: stacks will be in
    /// subdirectories under a single root directory, so they must have unique
    /// names.
    pub fn add_root_name(
        &mut self,
        input: Option<&str>,
        deliver_to_directory: bool,
        _dual: bool,
        do_validation: bool,
        err_msg: Option<&mut String>,
    ) -> bool {
        if self.mode != Mode::Batch {
            eprintln!(
                "Warning: attempting to set a root name in a non-batch mode batchruntomo command"
            );
            return false;
        }
        let mut retval = true;
        // Java `String` elements may be null; `HashSet` and `ArrayList` accept a
        // null, which later prints as "null".
        let input = input.unwrap_or("null").to_owned();
        if do_validation && deliver_to_directory {
            if self.root_name_validation_set.is_none() {
                self.root_name_validation_set = Some(HashSet::new());
            }
            let root_name_validation_set = self.root_name_validation_set.as_mut().unwrap();
            if root_name_validation_set.contains(&input) {
                self.valid.store(false, Ordering::SeqCst);
                retval = false;
                if let Some(err_msg) = err_msg {
                    err_msg.push_str(&format!("Dataset root name must be unique: {input}.  "));
                }
            } else {
                root_name_validation_set.insert(input.clone());
            }
        }
        self.interleaved_parameters[InterleavedIndex::ROOT_NAME.index].push(input);
        retval
    }

    /// Java `getNumRootNames`.
    pub fn get_num_root_names(&self) -> i32 {
        if self.mode == Mode::Batch {
            return self.interleaved_parameters[InterleavedIndex::ROOT_NAME.index].len() as i32;
        }
        0
    }

    /// Java `addCurrentLocation`.  Sets the currentLocation to the original
    /// stack location preferentially.  Using the original location allows the
    /// use of MakeSubDirectory after delivery.  `deliver_off`: no delivery, so
    /// location must be unique; `err_msg` will be used if it is not null.
    pub fn add_current_location(
        &mut self,
        original_stack_location: Option<&str>,
        current_location: Option<&str>,
        deliver_off: bool,
        do_validation: bool,
        err_msg: Option<&mut String>,
    ) -> bool {
        if self.mode != Mode::Batch {
            eprintln!(
                "Warning: attempting to set a current location in a non-batch mode batchruntomo command"
            );
            return false;
        }

        let mut retval = true;
        let current_location = current_location.unwrap_or("null").to_owned();
        if do_validation && deliver_off {
            // Validate with the current location
            if self.delivered_location_validation_set.is_none() {
                self.delivered_location_validation_set = Some(HashSet::new());
            }
            let delivered_location_validation_set =
                self.delivered_location_validation_set.as_mut().unwrap();
            if delivered_location_validation_set.contains(&current_location) {
                self.valid.store(false, Ordering::SeqCst);
                retval = false;
                if let Some(err_msg) = err_msg {
                    err_msg.push_str("Dataset location must be unique");
                }
            } else {
                delivered_location_validation_set.insert(current_location.clone());
            }
        }
        // Add the original location to command if possible.
        if let Some(original_stack_location) = original_stack_location
            && !java_lang_string_matches_whitespace(original_stack_location)
        {
            self.interleaved_parameters[InterleavedIndex::CURRENT_LOCATION.index]
                .push(original_stack_location.to_owned());
        } else {
            self.interleaved_parameters[InterleavedIndex::CURRENT_LOCATION.index]
                .push(current_location);
        }
        retval
    }

    /// Java `setCPUMachineList`.
    pub fn set_cpu_machine_list(&mut self, input: Option<&str>) {
        // `StringBuilder.append(null)` appends "null".
        self.cpu_machine_list = Some(input.unwrap_or("null").to_owned());
    }

    /// Java `setGPUMachineList`.
    pub fn set_gpu_machine_list(&mut self, input: Option<&str>) {
        // `StringBuilder.append(null)` appends "null".
        self.gpu_machine_list = Some(input.unwrap_or("null").to_owned());
    }

    /// Java `isEmailAddressNull`.
    pub fn is_email_address_null(&self) -> bool {
        self.email_address.is_empty()
    }

    /// Java `setEmailAddress`.
    pub fn set_email_address(&mut self, input: Option<&str>) {
        self.email_address.set(input);
    }

    /// Java `setSmtpServer`.
    pub fn set_smtp_server(&mut self, input: Option<&str>) {
        self.smtp_server.set(input);
    }

    /// Java `resetEmailAddress`.
    pub fn reset_email_address(&mut self) {
        self.email_address.reset();
    }

    /// Java `getEmailAddress`.
    pub fn get_email_address(&self) -> String {
        self.email_address.to_string()
    }

    /// Java `isGpuMachineListNull`.
    pub fn is_gpu_machine_list_null(&self) -> bool {
        match &self.gpu_machine_list {
            None => true,
            Some(gpu_machine_list) => gpu_machine_list.is_empty(),
        }
    }

    /// Java `isUseExistingAlignment`.
    pub fn is_use_existing_alignment(&self) -> bool {
        self.use_existing_alignment.is()
    }

    /// Java `gpuMachineListEquals`.
    pub fn gpu_machine_list_equals(&self, input: Option<&str>) -> bool {
        match &self.gpu_machine_list {
            None => false,
            Some(gpu_machine_list) => Some(gpu_machine_list.as_str()) == input,
        }
    }

    /// Java `isCpuMachineListNull`.
    pub fn is_cpu_machine_list_null(&self) -> bool {
        match &self.cpu_machine_list {
            None => true,
            Some(cpu_machine_list) => cpu_machine_list.is_empty(),
        }
    }

    /// Java `addDirectiveFile(DirectiveFile)`.
    pub fn add_directive_file_directive_file(&mut self, directive_file: Option<&DirectiveFile>) {
        if let Some(directive_file) = directive_file {
            let file = directive_file.get_file();
            if let Some(file) = file {
                self.interleaved_parameters[InterleavedIndex::DIRECTIVE_FILE.index]
                    .push(java_io_file_get_absolute_path(&file.to_string_lossy()));
            }
        }
    }

    /// Java `isValid`.
    pub fn is_valid(&self) -> bool {
        if self.mode == Mode::Validation {
            return (self
                .validation_type
                .equals_int(VALIDATION_TYPE_BATCH_DIRECTIVE)
                || self.validation_type.equals_int(VALIDATION_TYPE_TEMPLATE))
                && !self.interleaved_parameters[InterleavedIndex::DIRECTIVE_FILE.index].is_empty();
        }
        if self.mode == Mode::Rename {
            return !self.root_name.is_empty() && !self.naming_style.lock().unwrap().is_null();
        }
        self.valid.load(Ordering::SeqCst)
    }

    /// Java `setValidationType`.
    pub fn set_validation_type(&mut self, directive_driven_automation: bool) {
        if directive_driven_automation {
            self.validation_type
                .set_int(VALIDATION_TYPE_BATCH_DIRECTIVE);
        } else {
            self.validation_type.set_int(VALIDATION_TYPE_TEMPLATE);
        }
    }

    /// Java `run`.  Create the system program and execute the batchruntomo
    /// command.
    pub fn run(&mut self) -> i32 {
        if self.batchruntomo.is_none() {
            // Create a new SystemProgram object for copytomocom, set the
            // working directory and stdin array.
            // Do not use the -e flag for tcsh since David's scripts handle the failure
            // of commands and then report appropriately. The exception to this is the
            // com scripts which require the -e flag. RJG: 2003-11-06
            let program = if self.mode != Mode::Validation {
                SystemProgram::new_array(
                    Some(self.manager),
                    self.manager.get_property_user_dir(),
                    self.get_command_array(),
                    AxisID::Only,
                )
            } else {
                SystemProgram::get_multi_line_instance(
                    Some(self.manager),
                    self.manager.get_property_user_dir(),
                    self.get_command_array(),
                    AxisID::Only,
                )
            };
            program.set_message_prepend_tag(Some("Beginning to process template file"));
            self.batchruntomo = Some(Arc::new(program));
        }
        let exit_value;
        // Execute the script
        let batchruntomo = self.batchruntomo.as_ref().unwrap();
        batchruntomo.run();
        exit_value = batchruntomo.get_exit_value();
        exit_value
    }

    /// Java `getStdErrorString`.
    pub fn get_std_error_string(&self) -> Option<String> {
        match &self.batchruntomo {
            None => Some("ERROR: Batchruntomo is null.".to_owned()),
            Some(batchruntomo) => batchruntomo.get_std_error_string(),
        }
    }

    /// Java `getStdOutputString`.
    pub fn get_std_output_string(&self) -> Option<String> {
        match &self.batchruntomo {
            None => Some("ERROR: Batchruntomo is null.".to_owned()),
            Some(batchruntomo) => batchruntomo.get_std_output_string(),
        }
    }

    /// Java `getStdError`.
    pub fn get_std_error(&self) -> Option<Vec<String>> {
        match &self.batchruntomo {
            None => Some(vec!["ERROR: Batchruntomo is null.".to_owned()]),
            Some(batchruntomo) => batchruntomo.get_std_error(),
        }
    }

    /// Java `getStdOutput`.
    pub fn get_std_output(&self) -> Option<Vec<String>> {
        if let Some(batchruntomo) = &self.batchruntomo {
            return batchruntomo.get_std_output();
        }
        None
    }

    /// Java `getProcessMessages`.  Returns the warnings, one warning per element;
    /// make sure that warnings get into the error log.
    pub fn get_process_messages(&self) -> Option<MutexGuard<'_, ProcessMessages>> {
        match &self.batchruntomo {
            None => None,
            Some(batchruntomo) => Some(batchruntomo.get_process_messages()),
        }
    }

    /// Java `getNamingStyle`.  The source returns the live parameter; this is a
    /// copy of its value.
    pub fn get_naming_style(&self) -> ConstEtomoNumber {
        let naming_style = self.naming_style.lock().unwrap();
        ConstEtomoNumber::new_from_instance(Some(&naming_style.base.base))
    }
}

impl CommandParam for BatchruntomoParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // reset
        for i in 0..self.interleaved_parameters.len() {
            self.interleaved_parameters[i].clear();
        }
        self.deliver_to_directory.reset();
        self.make_sub_directory.reset();
        self.cpu_machine_list = None;
        self.gpu_machine_list = None;
        self.nice_value.reset();
        self.email_address.reset();
        if let Some(root_name_validation_set) = &mut self.root_name_validation_set {
            root_name_validation_set.clear();
        }
        self.smtp_server.reset();
        self.use_existing_alignment.reset();
        self.queue_command.reset();
        self.max_jobs_on_queue.reset();
        self.cores_per_cluster_job.reset();
        self.gpus_per_cluster_job.reset();
        self.gpu_queue_command.reset();
        self.max_gpu_jobs_on_queue.reset();
        self.etomo_debug.reset();
        // Rename parameters aren't used in the comfile.
        // parse
        // The interleaved parameters are all based on the .ebt file:
        // rootName: based on .ebt file properties
        // currentLocation: based on .ebt file properties
        // directiveFile: based on .ebt file properties
        self.deliver_to_directory.parse(script_command)?;
        self.make_sub_directory.parse(script_command)?;
        // BatchruntomoParam.java:201-204 `append`s `getValue`, which is null for a
        // keyword present without a value, and `StringBuilder.append(null)` adds
        // the text "null" - a machine named "null" that the com file never
        // contained.  Fixed in translation: a missing value appends nothing.
        let mut cpu_machine_list = String::new();
        if let Some(value) = script_command.get_value(Some(CPU_MACHINE_LIST_TAG))? {
            cpu_machine_list.push_str(&value);
        }
        self.cpu_machine_list = Some(cpu_machine_list);
        let mut gpu_machine_list = String::new();
        if let Some(value) = script_command.get_value(Some(GPU_MACHINE_LIST_TAG))? {
            gpu_machine_list.push_str(&value);
        }
        self.gpu_machine_list = Some(gpu_machine_list);
        self.nice_value.parse(script_command)?;
        self.email_address.parse(script_command)?;
        self.ending_step.parse(script_command)?;
        self.starting_step.parse(script_command)?;
        if self.for_update {
            self.smtp_server.parse(script_command)?;
        }
        self.use_existing_alignment.parse(script_command)?;
        self.queue_command.parse(script_command)?;
        self.max_jobs_on_queue.parse(script_command)?;
        self.cores_per_cluster_job.parse(script_command)?;
        self.gpus_per_cluster_job.parse(script_command)?;
        self.gpu_queue_command.parse(script_command)?;
        self.max_gpu_jobs_on_queue.parse(script_command)?;
        self.etomo_debug.parse(script_command)?;
        // NamingStyle parse is only used for repairing missing imageFilenameStyle
        // (Bug# 2386). Value is overridden from meta data before saving.
        self.naming_style.get_mut().unwrap().parse(script_command)?;
        // Rename parameters aren't used in the comfile.
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        let tags = InterleavedIndex::get_tags(Some(&self.mode));
        let values: Vec<Option<Vec<Option<String>>>> = self
            .interleaved_parameters
            .iter()
            .map(|list| Some(list.iter().cloned().map(Some).collect()))
            .collect();
        script_command.set_values_interleaved(tags.as_deref(), Some(&values));
        self.queue_command.delete_from_com_script(script_command);
        self.gpu_queue_command
            .delete_from_com_script(script_command);
        self.max_jobs_on_queue
            .delete_from_com_script(script_command);
        self.gpus_per_cluster_job
            .delete_from_com_script(script_command);
        self.max_gpu_jobs_on_queue
            .delete_from_com_script(script_command);
        self.cores_per_cluster_job
            .delete_from_com_script(script_command);
        let mut translate_paths_from_array = self.translate_paths_from_array.lock().unwrap();
        let mut translate_paths_to_array = self.translate_paths_to_array.lock().unwrap();
        translate_paths_from_array.delete_all_from_com_script(script_command);
        translate_paths_to_array.delete_all_from_com_script(script_command);
        self.remote_directory
            .lock()
            .unwrap()
            .delete_from_com_script(script_command);
        // BatchruntomoParam.java:245 assigns `numGlobalEntries` and never reads it;
        // the call still loads the mount rules.
        let _num_global_entries =
            remote_path::INSTANCE.num_global_entries(self.manager, self.axis_id);
        translate_paths_from_array.reset();
        translate_paths_to_array.reset();
        if self.mode == Mode::Batch {
            let mut naming_style = self.naming_style.lock().unwrap();
            // `manager.getBaseMetaData().getImageFilenameStyle().toString()`.
            if let Some(meta_data) = self.manager.get_base_meta_data() {
                naming_style.set_string(Some(
                    &meta_data.base().get_image_filename_style().to_string(),
                ));
            }
            naming_style.update_com_script(script_command);
            drop(naming_style);
            if let Some(queue_mode) = self.queue_mode {
                // queue mode 1
                if queue_mode == QueueMode::QueueWithSingleCpu {
                    self.queue_command.update_com_script(script_command);
                    self.max_jobs_on_queue.update_com_script(script_command);
                }
                // queue mode 2 & 2c
                else if queue_mode.is_type(QueueType::Node) {
                    self.cores_per_cluster_job.update_com_script(script_command);
                    // queue Mode 2
                    if queue_mode == QueueMode::NodeWithGpu {
                        self.gpus_per_cluster_job.update_com_script(script_command);
                    }
                }
                // queue Mode 3
                // This is in addition to queue mode 1 or 2c.
                self.gpu_queue_command.update_com_script(script_command);
                self.max_gpu_jobs_on_queue.update_com_script(script_command);
            }
            translate_paths_from_array.reset();
            translate_paths_to_array.reset();
            if self.parallel_processing {
                // When using batchruntomo with processchunks, only use global mount
                // rules.
                let local_rules =
                    remote_path::INSTANCE.local_global_rule_iterator(self.manager, self.axis_id);
                let mut local_rules = local_rules.into_iter().map(Some);
                translate_paths_from_array.set_all(Some(&mut local_rules));
                translate_paths_from_array.update_com_script(script_command)?;
                let remote_rules =
                    remote_path::INSTANCE.remote_global_rule_iterator(self.manager, self.axis_id);
                let mut remote_rules = remote_rules.into_iter().map(Some);
                translate_paths_to_array.set_all(Some(&mut remote_rules));
                translate_paths_to_array.update_com_script(script_command)?;
            }
        }
        drop(translate_paths_from_array);
        drop(translate_paths_to_array);
        script_command.set_value(
            Some("CheckFile"),
            check_file_value()
                .get_file_name(Some(self.manager), None)
                .as_deref(),
        );
        self.deliver_to_directory.update_com_script(script_command);
        self.make_sub_directory.update_com_script(script_command);
        if self.queue_mode.is_none() {
            match &self.cpu_machine_list {
                Some(cpu_machine_list) if !cpu_machine_list.is_empty() => {
                    script_command
                        .set_value(Some(CPU_MACHINE_LIST_TAG), Some(cpu_machine_list.as_str()));
                }
                _ => script_command.delete_key_all(Some(CPU_MACHINE_LIST_TAG)),
            }
            match &self.gpu_machine_list {
                Some(gpu_machine_list) if !gpu_machine_list.is_empty() => {
                    script_command
                        .set_value(Some(GPU_MACHINE_LIST_TAG), Some(gpu_machine_list.as_str()));
                }
                _ => script_command.delete_key_all(Some(GPU_MACHINE_LIST_TAG)),
            }
        } else {
            script_command.delete_key_all(Some(CPU_MACHINE_LIST_TAG));
            script_command.delete_key_all(Some(GPU_MACHINE_LIST_TAG));
        }
        self.nice_value.update_com_script(script_command);
        self.email_address.update_com_script(script_command);
        self.ending_step.update_com_script(script_command);
        self.starting_step.update_com_script(script_command);
        self.smtp_server.update_com_script(script_command);
        self.use_existing_alignment
            .update_com_script(script_command);
        self.etomo_debug.update_com_script(script_command);
        // When using batchruntomo with processchunks, only use global mount rules.
        let remote = remote_path::INSTANCE.get_remote_path(
            self.manager,
            self.manager.get_property_user_dir().as_deref(),
            self.axis_id,
            if self.parallel_processing {
                Some(true)
            } else {
                None
            },
        );
        match remote {
            Ok(remote) => {
                let mut remote_directory = self.remote_directory.lock().unwrap();
                remote_directory.set(remote.as_deref());
                remote_directory.update_com_script(script_command);
            }
            Err(e) => {
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!(
                        "ERROR:  Remote path error.  Unabled to run batchruntomo.\n\n{}",
                        e.get_message()
                    ),
                    "Batchruntomo Error".to_owned(),
                    Some(self.axis_id),
                );
                self.valid.store(false, Ordering::SeqCst);
            }
        }
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}

impl Command for BatchruntomoParam {
    /// Java `getCommandArray`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        if self.mode == Mode::Batch {
            // A null file name is a one-element array holding null in the source; it
            // is an empty array here.
            return Some(
                file_type::CLASS
                    .batch_run_tomo_comscript
                    .get_file_name(Some(self.manager), Some(self.axis_id))
                    .into_iter()
                    .collect(),
            );
        }
        let mut command: Vec<String> = Vec::new();
        command.push("python".to_owned());
        command.push("-u".to_owned());
        // Java string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command.push(format!("{script_path}{PROCESS_NAME}"));
        if self.mode == Mode::Validation {
            command.push("-validation".to_owned());
            command.push(self.validation_type.to_string());
            let len = self.interleaved_parameters[InterleavedIndex::DIRECTIVE_FILE.index].len();
            for i in 0..len {
                command.push("-directive".to_owned());
                let value = &self.interleaved_parameters[InterleavedIndex::DIRECTIVE_FILE.index][i];
                command.push(value.clone());
            }
        } else if self.mode == Mode::Rename {
            let naming_style = self.naming_style.lock().unwrap();
            command.push(format!("{PARAMETER_PREFIX}{}", self.root_name.get_name()));
            command.push(self.root_name.to_string());
            command.push(format!("{PARAMETER_PREFIX}{}", naming_style.get_name()));
            command.push(naming_style.to_string());
            command.push(format!(
                "{PARAMETER_PREFIX}{}",
                self.axis_of_extension.get_name()
            ));
            command.push(self.axis_of_extension.to_string());
            command.push(format!(
                "{PARAMETER_PREFIX}{}",
                self.stack_extension.get_name()
            ));
            command.push(self.stack_extension.to_string());
        }
        let mut command_array = Vec::with_capacity(command.len());
        for i in 0..command.len() {
            command_array.push(command[i].clone());
        }
        Some(command_array)
    }

    /// Java `getCommandLine`.  Return the current command line string.  Command
    /// will change if more parameters are added.
    fn get_command_line(&self) -> Option<String> {
        if self.mode == Mode::Batch {
            return self.get_command();
        }
        if let Some(batchruntomo) = &self.batchruntomo {
            return Some(batchruntomo.get_command_line());
        }
        let command = self.get_command_array().unwrap_or_default();
        let mut command_line = String::new();
        for i in 0..command.len() {
            command_line.push_str(&format!("{} ", command[i]));
        }
        Some(command_line)
    }

    /// Java `getCommandMode`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        Some(&self.mode)
    }

    /// Java `getCommandOutputFile`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getAxisID`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        if self.mode == Mode::Batch {
            return file_type::CLASS
                .batch_run_tomo_comscript
                .get_file_name(Some(self.manager), Some(self.axis_id));
        }
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getSubcommandDetails`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getProcessName`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    /// Java `getCommandInputFile`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `isMessageReporter`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }
}
