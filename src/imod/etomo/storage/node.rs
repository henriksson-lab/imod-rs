//! `IMOD/Etomo/src/etomo/storage/Node.java`.
//!
//! Copyright: Copyright 2010 - 2023 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Shape.**  A `Node` is built and loaded by `CpuAdoc` (or `createLocalInstance`)
//! and then shared, read-only, with every caller: the instances are `Arc<Node>`.  The
//! one field the source changes after loading, `coresPerClusterJob` (in
//! `workaround`), is behind a `Mutex`.  Java's identity comparisons against
//! `LOCAL_HOST_INSTANCE` and `ignoredNode` are pointer comparisons.
//!
//! `ReadOnlySection`/`ReadOnlyAttribute` reach the autodoc through the raw pointers of
//! the autodoc translation; `load` copies every value it needs out of the section, so no
//! pointer outlives the call.

use super::autodoc::attribute::Attribute;
use super::autodoc::read_only_attribute::ReadOnlyAttribute;
use super::autodoc::read_only_attribute_list::ReadOnlyAttributeList;
use super::autodoc::read_only_section::ReadOnlySection;
use super::autodoc::read_only_statement_list::ReadOnlyStatementList;
use super::autodoc::section::Section;
use super::cpu_adoc;
use super::network::Network;
use super::pc_option_type::PcOptionType;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::pc_option_element::{PcOptionElement, PcOptionsMap};
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::queue_mode::QueueMode;
use crate::imod::etomo::r#type::substitution_string::SubstitutionString;
use crate::imod::etomo::util::environment_variable;
use crate::imod::etomo::util::utilities::java_lang_string_split;
use regex::Regex;
use std::sync::{Arc, LazyLock, Mutex};

/// Java package-private `NUMBER_KEY`.
pub(crate) const NUMBER_KEY: &str = "number";
/// Java package-private `TYPE_KEY`.
pub(crate) const TYPE_KEY: &str = "type";
/// Java package-private `SPEED_KEY`.
pub(crate) const SPEED_KEY: &str = "speed";
/// Java package-private `MEMORY_KEY`.
pub(crate) const MEMORY_KEY: &str = "memory";
/// Java package-private `OS_KEY`.
pub(crate) const OS_KEY: &str = "os";
/// Java package-private `NCORES_KEY`.
pub(crate) const NCORES_KEY: &str = "ncores";
/// Java package-private `GPUS_PER_CLUSTER_JOB_BACKWARD_COMPATIBILITY_KEY`.  The
/// gpusPerNode key is kept for backward compatibility.  The correction name
/// gpusPerClusterJob.
pub(crate) const GPUS_PER_CLUSTER_JOB_BACKWARD_COMPATIBILITY_KEY: &str = "gpusPerNode";
/// Java package-private `GPUS_PER_CLUSTER_JOB_KEY`.
pub(crate) const GPUS_PER_CLUSTER_JOB_KEY: &str = "gpusPerClusterJob";
/// Java private `GPUS_PER_CLUSTER_JOB_DEFAULT`, an `Integer`.
const GPUS_PER_CLUSTER_JOB_DEFAULT: i32 = 1;

/// Java `LOCAL_HOST_NAME`.
pub const LOCAL_HOST_NAME: &str = "localhost";

/// Java package-private static `LOCAL_HOST_INSTANCE`.  Created by Network when
/// cpu.adoc is missing.  The `Mutex` is also the class lock of the `synchronized`
/// `createLocalInstance`.
pub(crate) static LOCAL_HOST_INSTANCE: Mutex<Option<Arc<Node>>> = Mutex::new(None);

/// Java `"\\s*,\\s*"`, with Java's `\s` class.
static COMMA_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[ \t\n\u{0B}\u{0C}\r]*,[ \t\n\u{0B}\u{0C}\r]*").unwrap());

/// Java `Node`.
pub struct Node {
    /// Java private field `pcOptionsMap`, initialised to null.
    pc_options_map: Option<PcOptionsMap>,
    /// Java private field `queue`.  Type of Node.  Default is computer.
    queue: bool,
    /// Java private field `name`, initialised to "".
    name: String,
    /// Java private field `number`.
    number: EtomoNumber,
    /// Java private field `gpu`.
    gpu: bool,
    /// Java private field `gpuLocal`.
    gpu_local: bool,
    /// Java private field `excludeInterface`.
    exclude_interface: Option<InterfaceType>,
    /// Java private field `userArray`.
    user_array: Option<Vec<String>>,
    /// Java private field `memory`, initialised to "".
    memory: Option<String>,
    /// Java private field `os`, initialised to "".
    os: Option<String>,
    /// Java private field `speed`, initialised to "".
    speed: Option<String>,
    /// Java private field `type`, initialised to "".
    r#type: Option<String>,
    /// Java private field `gpumemory`, initialised to "".
    gpumemory: Option<String>,
    /// Java private field `gpuncores`, initialised to "".
    gpuncores: Option<String>,
    /// Java private field `gpuspeed`, initialised to "".
    gpuspeed: Option<String>,
    /// Java private field `gputype`, initialised to "".
    gputype: Option<String>,
    /// Java private field `gpuDeviceArray`.
    gpu_device_array: Option<Vec<String>>,
    // Queue attribute
    /// Java private field `command`, initialised to "".
    command: Option<String>,
    /// Java private field `initialize`, a `SubstitutionString`, initialised to null.
    initialize: Option<SubstitutionString>,
    /// The attribute value `initialize` was built from.  `ProcesschunksParam` takes the
    /// Java reference; `SubstitutionString` is not `Clone`, so `getParameters` hands it
    /// a new instance built the same way (`new SubstitutionString(value, "nodes")`).
    initialize_value: Option<String>,
    /// Java private field `deinitialize`, initialised to "".
    deinitialize: Option<String>,
    /// Java private field `coresPerNode`.
    cores_per_node: Option<EtomoNumber>,
    /// Java private field `coresPerClusterJob`; `workaround` may set it after loading.
    cores_per_cluster_job: Mutex<Option<EtomoNumber>>,
    /// Java private field `gpusPerClusterJob`.
    gpus_per_cluster_job: Option<EtomoNumber>,
}

impl Node {
    /// Java private constructor `Node()`.
    fn new() -> Node {
        let mut number = EtomoNumber::new();
        number.set_display_value_int(1);
        number.set_default_int(1);
        Node {
            pc_options_map: None,
            queue: false,
            name: String::new(),
            number,
            gpu: false,
            gpu_local: false,
            exclude_interface: None,
            user_array: None,
            memory: Some(String::new()),
            os: Some(String::new()),
            speed: Some(String::new()),
            r#type: Some(String::new()),
            gpumemory: Some(String::new()),
            gpuncores: Some(String::new()),
            gpuspeed: Some(String::new()),
            gputype: Some(String::new()),
            gpu_device_array: None,
            command: Some(String::new()),
            initialize: None,
            initialize_value: None,
            deinitialize: Some(String::new()),
            cores_per_node: None,
            cores_per_cluster_job: Mutex::new(None),
            gpus_per_cluster_job: None,
        }
    }

    /// Java `getQueueMode`.  For batchruntomo.  A queue is mode 1 if no
    /// coresPerClusterJob, coresPerClusterJob=1, and no gpu.  A queue is mode 2c if
    /// coresPerClusterJobs>1, and no gpu.  Queue is mode 2 if has gpu(s).  A queue is
    /// invalid if it's mode 1 and it's number is 1 or 2 as a queue this small would
    /// cause deadlock when used with the batchruntomo interface.
    pub fn get_queue_mode(&self) -> QueueMode {
        if self.initialize.is_some() {
            // Exclusive allocation mode queues are treated as mode 1.
            return QueueMode::QueueWithSingleCpu;
        }
        if !self.gpu {
            let cores_per_cluster_job = self.cores_per_cluster_job.lock().unwrap();
            if cores_per_cluster_job
                .as_ref()
                .is_none_or(|cores_per_cluster_job| cores_per_cluster_job.le_int(1))
            {
                if self.number.gt_int(2) {
                    return QueueMode::QueueWithSingleCpu;
                }
                return QueueMode::Invalid;
            }
            return QueueMode::NodeWithoutGpu;
        }
        QueueMode::NodeWithGpu
    }

    /// Java `isSecondaryQueue`.  Secondary queues have a single GPU.
    pub fn is_secondary_queue(&self) -> bool {
        if self.queue
            && self.is_gpus_per_cluster_job()
            && self.gpus_per_cluster_job_equals(GPUS_PER_CLUSTER_JOB_DEFAULT)
        {
            return true;
        }
        false
    }

    /// Java `getTotalGPUs`.  Return the number of GPUs available on this node.
    pub fn get_total_gpus(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> i32 {
        // If gpuLocal is set and this is not the local host, then none of the GPUs are
        // available.
        let is_local_host_instance = LOCAL_HOST_INSTANCE
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|instance| std::ptr::eq(&**instance, self));
        let localhost =
            is_local_host_instance || self.is_local_host(manager, axis_id, property_user_dir);
        if !self.gpu || (self.gpu_local && !localhost) {
            return 0;
        }
        match &self.gpu_device_array {
            None => 1,
            Some(gpu_device_array) => gpu_device_array.len() as i32,
        }
    }

    /// Java package-private static `getComputerInstance`.
    pub(crate) fn get_computer_instance() -> Node {
        Node::new()
    }

    /// Java package-private static `getQueueInstance`.
    pub(crate) fn get_queue_instance() -> Node {
        let mut instance = Node::new();
        instance.queue = true;
        instance
    }

    /// Java package-private `validate`.  Return true if node is valid.
    pub(crate) fn validate(&self) -> bool {
        // A warning needs to be popped up for a queue having a coresPerNode entry but no
        // initialize command (Bug# 2439). CoresPerClusterJob is now used for things that
        // used to be covered by coresPerNode.
        if self.queue
            && self.cores_per_cluster_job.lock().unwrap().is_none()
            && self.cores_per_node.is_some()
            && self.initialize.is_none()
        {
            return false;
        }
        true
    }

    /// Java package-private `workaround`.  Returns an explanatory message if a
    /// workaround was applied.
    pub(crate) fn workaround(&self) -> Option<String> {
        // When the queue has a coresPerNode entry but no initialize command (Bug# 2439),
        // the workaround is using the value of coresPerNode for CoresPerClusterJob.
        let mut cores_per_cluster_job = self.cores_per_cluster_job.lock().unwrap();
        if self.queue
            && cores_per_cluster_job.is_none()
            && self.cores_per_node.is_some()
            && self.initialize.is_none()
        {
            // Workaround
            let mut number = EtomoNumber::new();
            number.set_const_etomo_number(self.cores_per_node.as_ref().map(|n| &n.base));
            *cores_per_cluster_job = Some(number);
            return Some(
                "In this case, the attribute coresPerClusterJob will be used instead.".to_string(),
            );
        }
        None
    }

    /// Java `getParameters(ProcesschunksParam)`.
    pub fn get_parameters_processchunks(&self, param: &ProcesschunksParam) {
        if self.queue {
            param.set_queue_command(self.command.as_deref());
            param.set_initialize(
                self.initialize.as_ref().map(|_| {
                    SubstitutionString::new(self.initialize_value.as_deref(), Some("nodes"))
                }),
            );
            param.set_deinitialize(self.deinitialize.as_deref());
            if let Some(cores_per_node) = &self.cores_per_node {
                param.set_cores_per_node(Some(&cores_per_node.base));
            }
            if let Some(cores_per_cluster_job) = &*self.cores_per_cluster_job.lock().unwrap() {
                param.set_cores_per_cluster_job(Some(&cores_per_cluster_job.base));
            }
            if self.is_gpus_per_cluster_job() {
                param.set_gpus_per_cluster_job(self.get_gpus_per_cluster_job().as_deref());
            }
        }
    }

    /// Java `getSecondaryParameters(ProcesschunksParam)`.
    pub fn get_secondary_parameters_processchunks(&self, param: &ProcesschunksParam) {
        if self.queue {
            param.set_secondary_queue(Some(&self.name));
            param.set_secondary_queue_command(self.command.as_deref());
        }
    }

    /// Java `getSecondaryParameters(BatchruntomoParam)`.
    pub fn get_secondary_parameters_batchruntomo(&self, param: &mut BatchruntomoParam) {
        if self.queue {
            param.set_gpu_queue_command(self.command.as_deref());
        }
    }

    /// Java `getParameters(BatchruntomoParam)`.
    pub fn get_parameters_batchruntomo(&self, param: &mut BatchruntomoParam) {
        if self.queue {
            param.set_queue_command(self.command.as_deref());
        } else {
            param.reset_queue_command();
        }
        match &*self.cores_per_cluster_job.lock().unwrap() {
            Some(cores_per_cluster_job) => {
                param.set_cores_per_cluster_job(Some(&cores_per_cluster_job.base));
            }
            None => param.reset_cores_per_cluster_job(),
        }
        if self.is_gpus_per_cluster_job() {
            param.set_gpus_per_cluster_job(self.get_gpus_per_cluster_job().as_deref());
        } else {
            param.reset_gpus_per_cluster_job();
        }
    }

    /// Java package-private static synchronized `createLocalInstance`.  Create a
    /// virtual cpu.adoc consisting of one entry.  Called by Network when cpu.adoc is
    /// missing.
    ///
    pub(crate) fn create_local_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) {
        let mut local_host_instance = LOCAL_HOST_INSTANCE.lock().unwrap();
        if local_host_instance.is_some() {
            return;
        }
        let mut instance = Node::new();
        instance.name = LOCAL_HOST_NAME.to_string();
        // See if LOCAL_INSTANCE.number should be greater then 1 and set it if
        // necessary.
        let mut imod_processors = EtomoNumber::new();
        imod_processors.set_string(Some(&environment_variable::INSTANCE.get_value(
            Some(manager),
            property_user_dir,
            "IMOD_PROCESSORS",
            Some(axis_id),
        )));
        // `EtomoDirector.getUserConfiguration()`: the values this method reads.
        let (parallel_processing, cpus, gpu_processing, no_of_local_gpus) =
            etomo_director::INSTANCE.with_user_configuration(|user_configuration| {
                (
                    user_configuration.is_parallel_processing(),
                    user_configuration.get_cpus().clone(),
                    user_configuration.is_gpu_processing(),
                    user_configuration.get_local_gpus_int(),
                )
            });
        if !imod_processors.is_null() && imod_processors.is_valid() {
            instance
                .number
                .set_const_etomo_number(Some(&imod_processors.base));
        } else if parallel_processing {
            instance.number.set_const_etomo_number(Some(&cpus));
        }
        // Set GPU processing.
        if gpu_processing {
            // `userConfiguration.getLocalGPUsInt()`.
            if no_of_local_gpus <= 0 {
                instance.gpu = false;
            } else {
                instance.gpu = true;
                if no_of_local_gpus > 1 {
                    let mut gpu_device_array = Vec::with_capacity(no_of_local_gpus as usize);
                    for i in 1..=no_of_local_gpus {
                        gpu_device_array.push(i.to_string());
                    }
                    instance.gpu_device_array = Some(gpu_device_array);
                }
            }
        }
        *local_host_instance = Some(Arc::new(instance));
    }

    /// Java package-private `load(ReadOnlySection)`.
    ///
    /// # Safety
    /// `section` must belong to a live autodoc.
    pub(crate) unsafe fn load(&mut self, section: &Section) {
        self.name = ReadOnlyStatementList::get_name(section).unwrap_or_default();
        let mut attribute: Option<&Attribute> =
            unsafe { ReadOnlySection::get_attribute(section, Some("exclude-interface")).as_ref() };
        if let Some(attribute) = attribute {
            self.exclude_interface = InterfaceType::get_instance(attribute.get_value().as_deref());
        }
        attribute = unsafe { ReadOnlySection::get_attribute(section, Some("users")).as_ref() };
        if let Some(attribute) = attribute {
            let list = attribute.get_value();
            if let Some(list) = list {
                self.user_array = Some(java_lang_string_split(&list, &COMMA_PATTERN));
            }
        }
        attribute = unsafe { ReadOnlySection::get_attribute(section, Some(NUMBER_KEY)).as_ref() };
        if let Some(attribute) = attribute {
            self.number.set_string(attribute.get_value().as_deref());
        }
        attribute = unsafe { ReadOnlySection::get_attribute(section, Some("memory")).as_ref() };
        if let Some(attribute) = attribute {
            self.memory = attribute.get_value();
        }
        attribute = unsafe { ReadOnlySection::get_attribute(section, Some("os")).as_ref() };
        if let Some(attribute) = attribute {
            self.os = attribute.get_value();
        }
        attribute = unsafe { ReadOnlySection::get_attribute(section, Some("speed")).as_ref() };
        if let Some(attribute) = attribute {
            self.speed = attribute.get_value();
        }
        attribute = unsafe { ReadOnlySection::get_attribute(section, Some(TYPE_KEY)).as_ref() };
        if let Some(attribute) = attribute {
            self.r#type = attribute.get_value();
        }
        let gpu_attribute: Option<&Attribute> =
            unsafe { ReadOnlySection::get_attribute(section, Some(cpu_adoc::GPU_KEY)).as_ref() };
        if let Some(gpu_attribute) = gpu_attribute {
            attribute = unsafe {
                gpu_attribute
                    .get_attribute_by_name(Some(MEMORY_KEY))
                    .as_ref()
            };
            if let Some(attribute) = attribute {
                self.gpumemory = attribute.get_value();
            }
            attribute = unsafe {
                gpu_attribute
                    .get_attribute_by_name(Some(NCORES_KEY))
                    .as_ref()
            };
            if let Some(attribute) = attribute {
                self.gpuncores = attribute.get_value();
            }
            attribute = unsafe {
                gpu_attribute
                    .get_attribute_by_name(Some(SPEED_KEY))
                    .as_ref()
            };
            if let Some(attribute) = attribute {
                self.gpuspeed = attribute.get_value();
            }
            attribute = unsafe { gpu_attribute.get_attribute_by_name(Some(TYPE_KEY)).as_ref() };
            if let Some(attribute) = attribute {
                self.gputype = attribute.get_value();
            }
        }
        unsafe {
            self.load_gpu(ReadOnlySection::get_attribute(section, Some(cpu_adoc::GPU_KEY)).as_ref())
        };
        if self.queue {
            attribute =
                unsafe { ReadOnlySection::get_attribute(section, Some("command")).as_ref() };
            if let Some(attribute) = attribute {
                self.command = attribute.get_value();
            }
            // coresPerNode
            attribute =
                unsafe { ReadOnlySection::get_attribute(section, Some("coresPerNode")).as_ref() };
            if let Some(attribute) = attribute {
                let mut cores_per_node = EtomoNumber::new();
                cores_per_node.set_display_value_int(1);
                cores_per_node.set_string(attribute.get_value().as_deref());
                self.cores_per_node = Some(cores_per_node);
            }
            // coresPerClusterJob
            attribute = unsafe {
                ReadOnlySection::get_attribute(section, Some("coresPerClusterJob")).as_ref()
            };
            if let Some(attribute) = attribute {
                let mut cores_per_cluster_job = EtomoNumber::new();
                cores_per_cluster_job.set_display_value_int(1);
                cores_per_cluster_job.set_string(attribute.get_value().as_deref());
                *self.cores_per_cluster_job.lock().unwrap() = Some(cores_per_cluster_job);
            }
            // gpusPerClusterJob
            attribute = unsafe {
                ReadOnlySection::get_attribute(section, Some(GPUS_PER_CLUSTER_JOB_KEY)).as_ref()
            };
            // Backward compatibility: gpusPerNode is kept only for backward compatibility
            // with gpusPerClusterJob.
            if attribute.is_none() {
                attribute = unsafe {
                    ReadOnlySection::get_attribute(
                        section,
                        Some(GPUS_PER_CLUSTER_JOB_BACKWARD_COMPATIBILITY_KEY),
                    )
                    .as_ref()
                };
            }
            if let Some(attribute) = attribute {
                let mut gpus_per_cluster_job = EtomoNumber::new();
                gpus_per_cluster_job.set_display_value_int(GPUS_PER_CLUSTER_JOB_DEFAULT);
                gpus_per_cluster_job.set_string(attribute.get_value().as_deref());
                self.gpus_per_cluster_job = Some(gpus_per_cluster_job);
            }
            attribute =
                unsafe { ReadOnlySection::get_attribute(section, Some("initialize")).as_ref() };
            if let Some(attribute) = attribute {
                let value = attribute.get_value();
                self.initialize = Some(SubstitutionString::new(value.as_deref(), Some("nodes")));
                self.initialize_value = value;
            }
            attribute =
                unsafe { ReadOnlySection::get_attribute(section, Some("deinitialize")).as_ref() };
            if let Some(attribute) = attribute {
                self.deinitialize = attribute.get_value();
            }
            attribute =
                unsafe { ReadOnlySection::get_attribute(section, Some("pc-option")).as_ref() };
            if let Some(attribute) = attribute {
                // Construct a pcOptionsMap here.
                let mut pc_options_map: PcOptionsMap = Vec::new();
                // Get option name and value
                let pc_option_option_list = attribute.get_children();
                // Node.java:383-384 calls `iterator()` on `getChildren()`, which is null
                // for a `pc-option` attribute with no children, and throws a
                // NullPointerException.  Fixed in translation: no children, no options.
                if let Some(pc_option_option_list) = unsafe { pc_option_option_list.as_ref() } {
                    let mut pc_option_option_iter = pc_option_option_list.iterator();
                    while pc_option_option_iter.has_next() {
                        let option_attrib: &Attribute =
                            unsafe { &**pc_option_option_iter.next().unwrap() };
                        let option = option_attrib.get_name();
                        let value = option_attrib.get_value();
                        match pc_options_map.iter_mut().find(|(key, _)| *key == option) {
                            Some((_, element)) => {
                                element.set_pc_option_element(
                                    Some(PcOptionType::PcOptionTypeQueue),
                                    value.as_deref(),
                                );
                            }
                            None => {
                                let element = PcOptionElement::new(
                                    Some(&option),
                                    Some(PcOptionType::PcOptionTypeQueue),
                                    value.as_deref(),
                                );
                                pc_options_map.push((option, element));
                            }
                        }
                    }
                }
                self.pc_options_map = Some(pc_options_map);
            }
        }
    }

    /// Java private `loadGpu(ReadOnlyAttribute)`.  Handle:
    /// gpu=
    /// gpu.load=
    /// gpu.device=
    ///
    /// # Safety
    /// `gpu_attribute` must belong to a live autodoc.
    unsafe fn load_gpu(&mut self, gpu_attribute: Option<&Attribute>) {
        self.gpu = false;
        self.gpu_local = false;
        self.gpu_device_array = None;
        let mut error = false;
        if let Some(gpu_attribute) = gpu_attribute {
            // gpu.device
            let mut attribute: Option<&Attribute> =
                unsafe { gpu_attribute.get_attribute_by_name(Some("device")).as_ref() };
            if let Some(attribute) = attribute {
                let value = attribute.get_value();
                match value {
                    Some(value) => {
                        let array = java_lang_string_split(&value, &COMMA_PATTERN);
                        if !array.is_empty() {
                            self.gpu_device_array = Some(array);
                        } else {
                            error = true;
                        }
                    }
                    None => error = true,
                }
            }
            // gpu.local
            attribute = unsafe { gpu_attribute.get_attribute_by_name(Some("local")).as_ref() };
            let mut value = EtomoNumber::new();
            if let Some(attribute) = attribute {
                value.set_string(attribute.get_value().as_deref());
                if !value.is_null() && value.gt_int(0) {
                    self.gpu_local = true;
                } else {
                    error = true;
                }
            }
            if !error {
                self.gpu = true;
            }
        }
    }

    /// Java `isLocalHost`.  Returns true if the name member variable equals the output
    /// of the "hostname" command, or matches it (equal to one or more sections of the
    /// host name).
    pub fn is_local_host(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> bool {
        if self.name == LOCAL_HOST_NAME {
            return true;
        }
        let local_host_name = Network::get_local_host_name(manager, axis_id, property_user_dir);
        let local_host_name = match local_host_name {
            None => return false,
            Some(local_host_name) => local_host_name,
        };
        if Node::match_host_computer(Some(&self.name), Some(&local_host_name)) {
            return true;
        }
        false
    }

    /// Java package-private static `matchHostComputer(String, String)`.  Return true if
    /// nodeName matches a hostName of a computer.  The name of a node may be a shortened
    /// version of the host name.  An example using the host name calvin.int.hobbes.edu:
    /// The node name must be either the entire host name, or one or more sections of
    /// the host name (calvin, calvin.int, or calvin.int.hobbes).
    pub(crate) fn match_host_computer(node_name: Option<&str>, host_name: Option<&str>) -> bool {
        let (node_name, host_name) = match (node_name, host_name) {
            (Some(node_name), Some(host_name)) => (node_name, host_name),
            _ => return false,
        };
        if node_name == host_name {
            return true;
        }
        let dot = b'.';
        if !host_name.contains('.') || !host_name.starts_with(node_name) {
            return false;
        }
        // nodeName must consist only of complete sections.
        if host_name.as_bytes()[node_name.len()] == dot {
            return true;
        }
        false
    }

    /// Java `isGpu`.
    pub fn is_gpu(&self) -> bool {
        self.gpu
    }

    /// Java `isGpuLocal`.
    pub fn is_gpu_local(&self) -> bool {
        self.gpu_local
    }

    /// Java `getName`.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Java `isExcludedInterface(InterfaceType)`.  Use the excludeInterface member
    /// variable to decide whether an interface should be excluded.
    pub fn is_excluded_interface(&self, input: Option<InterfaceType>) -> bool {
        if self.exclude_interface.is_none() || self.exclude_interface != input {
            return false;
        }
        true
    }

    /// Java package-private `getExcludeInterface`.
    pub(crate) fn get_exclude_interface(&self) -> Option<InterfaceType> {
        self.exclude_interface
    }

    /// Java `isMemoryEmpty`.
    pub fn is_memory_empty(&self) -> bool {
        self.memory
            .as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
    }

    /// Java `getMemory`.
    pub fn get_memory(&self) -> Option<String> {
        self.memory.clone()
    }

    /// Java private `setNumber(int)`.
    fn set_number(&mut self, input: i32) {
        self.number.set_int(input);
    }

    /// Java `getCpus`.
    pub fn get_cpus(&self) -> &ConstEtomoNumber {
        &self.number.base
    }

    /// Java package-private `isNumberGt1`.
    pub(crate) fn is_number_gt1(&self) -> bool {
        let number = self.get_cpus();
        !number.is_null() && number.gt_int(1)
    }

    /// Java package-private `isGpuGt1`.
    pub(crate) fn is_gpu_gt1(&self) -> bool {
        let gpu_array = self.get_gpu_device_array();
        gpu_array.is_some_and(|gpu_array| gpu_array.len() > 1)
    }

    /// Java `getGpuNumber`.
    pub fn get_gpu_number(&self) -> i32 {
        match &self.gpu_device_array {
            Some(gpu_device_array) if self.gpu => gpu_device_array.len() as i32,
            _ => 1,
        }
    }

    /// Java `getGpuDeviceArray`.
    pub fn get_gpu_device_array(&self) -> Option<&[String]> {
        self.gpu_device_array.as_deref()
    }

    /// Java `isOsEmpty`.
    pub fn is_os_empty(&self) -> bool {
        self.os
            .as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
    }

    /// Java `getOs`.
    pub fn get_os(&self) -> Option<String> {
        self.os.clone()
    }

    /// Java `isSpeedEmpty`.
    pub fn is_speed_empty(&self) -> bool {
        self.speed
            .as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
    }

    /// Java `getSpeed`.
    pub fn get_speed(&self) -> Option<String> {
        self.speed.clone()
    }

    /// Java `isTypeEmpty`.
    pub fn is_type_empty(&self) -> bool {
        self.r#type
            .as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
    }

    /// Java `getType`.
    pub fn get_type(&self) -> Option<String> {
        self.r#type.clone()
    }

    /// Java `isType`.
    pub fn is_type(&self) -> bool {
        self.r#type
            .as_deref()
            .is_some_and(|value| !java_lang_string_matches_whitespace(value))
    }

    /// Java `isSpeed`.
    pub fn is_speed(&self) -> bool {
        self.speed
            .as_deref()
            .is_some_and(|value| !java_lang_string_matches_whitespace(value))
    }

    /// Java `isMemory`.
    pub fn is_memory(&self) -> bool {
        self.memory
            .as_deref()
            .is_some_and(|value| !java_lang_string_matches_whitespace(value))
    }

    /// Java `isOs`.
    pub fn is_os(&self) -> bool {
        self.os
            .as_deref()
            .is_some_and(|value| !java_lang_string_matches_whitespace(value))
    }

    /// Java `isGpuType`.
    pub fn is_gpu_type(&self) -> bool {
        self.gputype
            .as_deref()
            .is_some_and(|value| !java_lang_string_matches_whitespace(value))
    }

    /// Java `isGpuSpeed`.
    pub fn is_gpu_speed(&self) -> bool {
        self.gpuspeed
            .as_deref()
            .is_some_and(|value| !java_lang_string_matches_whitespace(value))
    }

    /// Java `isGpusPerClusterJob`.
    pub fn is_gpus_per_cluster_job(&self) -> bool {
        // If this is a queue and there is a gpu, then gpusPerClusterJob exists or is
        // implied to be 1.
        self.queue && self.gpu
    }

    /// Java `getGpusPerClusterJob`.
    pub fn get_gpus_per_cluster_job(&self) -> Option<String> {
        if let Some(gpus_per_cluster_job) = &self.gpus_per_cluster_job {
            return Some(gpus_per_cluster_job.to_defaulted_string());
        }
        // If this is a queue and there is a gpu, then gpusPerClusterJob is implied to
        // be 1.
        if self.is_gpus_per_cluster_job() {
            return Some(GPUS_PER_CLUSTER_JOB_DEFAULT.to_string());
        }
        None
    }

    /// Java private `gpusPerClusterJobEquals(int)`.
    fn gpus_per_cluster_job_equals(&self, input: i32) -> bool {
        if let Some(gpus_per_cluster_job) = &self.gpus_per_cluster_job {
            return gpus_per_cluster_job.equals_int(input);
        }
        self.is_gpus_per_cluster_job() && input == GPUS_PER_CLUSTER_JOB_DEFAULT
    }

    /// Java `isGpuMemory`.
    pub fn is_gpu_memory(&self) -> bool {
        self.gpumemory
            .as_deref()
            .is_some_and(|value| !java_lang_string_matches_whitespace(value))
    }

    /// Java `isGpuNcores`.
    pub fn is_gpu_ncores(&self) -> bool {
        self.gpuncores
            .as_deref()
            .is_some_and(|value| !java_lang_string_matches_whitespace(value))
    }

    /// Java `getGpuType`.
    pub fn get_gpu_type(&self) -> Option<String> {
        self.gputype.clone()
    }

    /// Java `getGpuMemory`.
    pub fn get_gpu_memory(&self) -> Option<String> {
        self.gpumemory.clone()
    }

    /// Java `getGpuNcores`.
    pub fn get_gpu_ncores(&self) -> Option<String> {
        self.gpuncores.clone()
    }

    /// Java `getGpuSpeed`.
    pub fn get_gpu_speed(&self) -> Option<String> {
        self.gpuspeed.clone()
    }

    /// Java `getCommand`.
    pub fn get_command(&self) -> Option<String> {
        self.command.clone()
    }

    /// Java `isExcludedUser(String)`.  Use the users array to decide whether a user
    /// should be excluded.  If the userArray exists, users not on the list are
    /// excluded.  If the userArray does not exist, no users are excluded.
    pub fn is_excluded_user(&self, user: Option<&str>) -> bool {
        let user_array = match &self.user_array {
            None => return false,
            Some(user_array) if user_array.is_empty() => return false,
            Some(user_array) => user_array,
        };
        for i in 0..user_array.len() {
            if Some(user_array[i].as_str()) == user {
                return false;
            }
        }
        true
    }

    /// Java `getPcOptionsMap`.  The source returns its own map; the entries are
    /// copied here because the node is shared.
    pub fn get_pc_options_map(&self) -> Option<PcOptionsMap> {
        self.pc_options_map.clone()
    }
}

/// Java `toString`.
impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut user_array_string = String::new();
        if let Some(user_array) = &self.user_array {
            let mut string_builder = String::new();
            for i in 0..user_array.len() {
                if !string_builder.is_empty() {
                    string_builder.push(',');
                }
                string_builder.push_str(&user_array[i]);
            }
            if !string_builder.is_empty() {
                user_array_string = string_builder;
            }
        }
        write!(
            f,
            "name:{},number:{},queue:{},excludeInterface:{}\nuserArray:{}",
            self.name,
            self.number,
            self.queue,
            match self.exclude_interface {
                None => "null".to_string(),
                Some(exclude_interface) => exclude_interface.to_string(),
            },
            user_array_string
        )
    }
}
