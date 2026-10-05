//! `IMOD/Etomo/src/etomo/storage/CpuAdoc.java`.
//!
//! Description: Represents the cpu.adoc file except the mount rules, which are handled
//! by RemotePath.
//!
//! Assumptions: The Computer section names in cpu.adoc must be unique.
//!
//! Copyright: Copyright 2006 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! @threadsafe @immutable @singleton
//!
//! **Shape.**  `INSTANCE` is shared by every thread, and `load` is `synchronized`, so
//! the fields `load` fills sit behind one `Mutex`; every public method takes `&self`.
//! The autodoc translation keeps its registry per thread and hands out raw pointers, so
//! `load` copies everything it needs (the `Node`s, the attributes) out of the autodoc
//! and keeps no pointer.  The `Hashtable`s of nodes are `HashMap`s of shared
//! `Arc<Node>`s; the `LinkedHashMap` of pc-options is a [`PcOptionsMap`].

use super::autodoc::attribute::Attribute;
use super::autodoc::autodoc::Autodoc;
use super::autodoc::autodoc_factory;
use super::autodoc::read_only_attribute::ReadOnlyAttribute;
use super::autodoc::read_only_attribute_list::ReadOnlyAttributeList;
use super::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use super::autodoc::read_only_section_list::ReadOnlySectionList;
use super::autodoc::read_only_statement_list::ReadOnlyStatementList;
use super::log_file::LogFileError;
use super::network::Network;
use super::node::{self, Node};
use super::pc_option_type::PcOptionType;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::pc_option_element::{PcOptionElement, PcOptionsMap};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::processor_type::ProcessorType;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::queue_mode::QueueMode;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::environment_variable;
use crate::imod::etomo::util::event_queue;
use crate::imod::etomo::util::utilities::java_lang_string_split;
use regex::Regex;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

/// Java `FILE_NAME`: `AutodocFactory.CPU + AutodocFactory.Extension.DEFAULT.toString()`.
pub static FILE_NAME: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}",
        autodoc_factory::CPU,
        autodoc_factory::extension::DEFAULT
    )
});
/// Java `MAN_PAGE`:
/// `AutodocFactory.CPU + AutodocFactory.Extension.DEFAULT.getExtensionString()`.
pub static MAN_PAGE: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}",
        autodoc_factory::CPU,
        autodoc_factory::extension::DEFAULT.get_extension_string()
    )
});
/// Java `DEFAULT_COMPUTER_SECTION_TYPE`.
pub const DEFAULT_COMPUTER_SECTION_TYPE: &str = "Computer";
/// Java private `DEFAULT_QUEUE_SECTION_TYPE`.
const DEFAULT_QUEUE_SECTION_TYPE: &str = "Queue";
/// Java package-private `GPU_KEY`.
pub(crate) const GPU_KEY: &str = "gpu";
/// Java private `SPEED_KEY`.
const SPEED_KEY: &str = "speed";
/// Java private `MEMORY_KEY`.
const MEMORY_KEY: &str = "memory";

/// Java private `MIN_NICE_DEFAULT`.
const MIN_NICE_DEFAULT: i32 = 0;

/// Java `"\\s*,\\s*"`, with Java's `\s` class.
static COMMA_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[ \t\n\u{0B}\u{0C}\r]*,[ \t\n\u{0B}\u{0C}\r]*").unwrap());
/// Java `"\\."`.
static DOT_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new("\\.").unwrap());

/// Java `INSTANCE`.
pub static INSTANCE: LazyLock<CpuAdoc> = LazyLock::new(CpuAdoc::new);

/// The fields `load` fills.
struct State {
    /// Java private field `pcOptionsMap`, initialised to null.
    pc_options_map: Option<PcOptionsMap>,
    /// Java private final field `computerList`: list of computer names.
    computer_list: Vec<String>,
    /// Java private final field `computerMap`, a `Hashtable`.  Its walks (the
    /// `Network` totals and maximum) do not depend on the order, so a Rust `HashMap`.
    computer_map: HashMap<String, Arc<Node>>,
    /// Java private final field `queueList`.
    queue_list: Vec<String>,
    /// Java private final field `queueMap`, a `Hashtable`.  Its walks (`hasQueues`,
    /// `hasSecondaryQueues`, `validate`, whose message names no queue, `workaround`,
    /// whose one message is the same for every queue, and the `Network` totals) do
    /// not depend on the order, so a Rust `HashMap`.
    queue_map: HashMap<String, Arc<Node>>,
    /// Java private final field `minNice`.
    min_nice: EtomoNumber,
    /// Java private final field `maxTilt`.
    max_tilt: EtomoNumber,
    /// Java private final field `maxVolcombine`.
    max_volcombine: EtomoNumber,
    /// Java private final field `orListMap`.
    or_list_map: OrListMap,
    /// Java private field `separateChunks`, initialised to false.
    separate_chunks: bool,
    /// Java private field `usersColumn`, initialised to false.
    users_column: bool,
    /// Java private field `speedUnits`, initialised to "".
    speed_units: Option<String>,
    /// Java private field `memoryUnits`, initialised to "".
    memory_units: Option<String>,
    /// Java private field `gpuSpeedUnits`, initialised to "".  Not in use - has not been
    /// added to cpuadoc man page.
    gpu_speed_units: Option<String>,
    /// Java private field `gpuMemoryUnits`, initialised to "".  Not in use - has not
    /// been added to cpuadoc man page.
    gpu_memory_units: Option<String>,
    /// Java private field `pcOptionType`, initialised to "" and never read.
    pc_option_type: String,
    /// Java private field `loadUnits`, initialised to `new String[0]`.
    load_units: Vec<String>,
    /// Java private field `envVar`, initialised to false and never read.
    env_var: bool,
    /// Java private field `userConfig`, initialised to false and never read.
    user_config: bool,
    /// Java private field `loaded`, initialised to false.
    loaded: bool,
    /// Java private field `exists`, initialised to false.
    exists: bool,
}

/// Java `CpuAdoc`.
pub struct CpuAdoc {
    /// Java private final field `user`: `System.getProperty("user.name")`.  The JVM
    /// property is the login name; `$USER` stands in for it.
    user: Option<String>,
    /// Java private final field `computerSectionType`.
    computer_section_type: Option<String>,
    /// Java private final field `queueSectionType`.
    queue_section_type: Option<String>,
    /// The mutable fields.
    state: Mutex<State>,
}

impl CpuAdoc {
    /// Java private constructor `CpuAdoc()`.
    fn new() -> CpuAdoc {
        let mut min_nice = EtomoNumber::new();
        min_nice.set_display_value_int(MIN_NICE_DEFAULT);
        min_nice.set_default_int(MIN_NICE_DEFAULT);
        let arguments = etomo_director::ARGUMENTS.lock().unwrap();
        let computer_section_type = if arguments.is_computer_section() {
            // Can be set to null to create a table without computers.
            arguments.get_computer_section().map(str::to_string)
        } else {
            Some(DEFAULT_COMPUTER_SECTION_TYPE.to_string())
        };
        let queue_section_type = if arguments.is_queue_section() {
            // Can be set to null to create a table without queues.
            arguments.get_queue_section().map(str::to_string)
        } else {
            Some(DEFAULT_QUEUE_SECTION_TYPE.to_string())
        };
        drop(arguments);
        CpuAdoc {
            user: std::env::var("USER").ok(),
            computer_section_type,
            queue_section_type,
            state: Mutex::new(State {
                pc_options_map: None,
                computer_list: Vec::new(),
                computer_map: HashMap::new(),
                queue_list: Vec::new(),
                queue_map: HashMap::new(),
                min_nice,
                max_tilt: EtomoNumber::new(),
                max_volcombine: EtomoNumber::new(),
                or_list_map: OrListMap::new(),
                separate_chunks: false,
                users_column: false,
                speed_units: Some(String::new()),
                memory_units: Some(String::new()),
                gpu_speed_units: Some(String::new()),
                gpu_memory_units: Some(String::new()),
                pc_option_type: String::new(),
                load_units: Vec::new(),
                env_var: false,
                user_config: false,
                loaded: false,
                exists: false,
            }),
        }
    }

    /// Java `getComputerSectionType`.
    pub fn get_computer_section_type(&self) -> Option<String> {
        self.computer_section_type.clone()
    }

    /// Java package-private `getComputers`: `computerMap.values()` (no `load`).
    pub(crate) fn get_computers(&self) -> Vec<Arc<Node>> {
        self.state
            .lock()
            .unwrap()
            .computer_map
            .values()
            .cloned()
            .collect()
    }

    /// Java package-private `getQueues`: `queueMap.values()` (no `load`).
    pub(crate) fn get_queues(&self) -> Vec<Arc<Node>> {
        self.state
            .lock()
            .unwrap()
            .queue_map
            .values()
            .cloned()
            .collect()
    }

    /// Java `isSeparateChunks`.
    pub fn is_separate_chunks(&self) -> bool {
        self.load().separate_chunks
    }

    /// Java `isUsersColumn`.
    pub fn is_users_column(&self) -> bool {
        self.load().users_column
    }

    /// Java `getMinNice`.
    pub fn get_min_nice(&self) -> i32 {
        self.load().min_nice.get_int()
    }

    /// Java `getSpeedUnits`.
    pub fn get_speed_units(&self) -> Option<String> {
        self.load().speed_units.clone()
    }

    /// Java `getGpuSpeedUnits` (no `load`).
    pub fn get_gpu_speed_units(&self) -> Option<String> {
        self.state.lock().unwrap().gpu_speed_units.clone()
    }

    /// Java `getMemoryUnits`.
    pub fn get_memory_units(&self) -> Option<String> {
        self.load().memory_units.clone()
    }

    /// Java `getGpuMemoryUnits`.
    pub fn get_gpu_memory_units(&self) -> Option<String> {
        self.load().gpu_memory_units.clone()
    }

    /// Java `getLoadUnitsArray`.
    pub fn get_load_units_array(&self) -> Vec<String> {
        self.load().load_units.clone()
    }

    /// Java `getLoadUnits`.
    pub fn get_load_units(&self) -> i32 {
        self.load().load_units.len() as i32
    }

    /// Java `getMaxTilt`.
    pub fn get_max_tilt(&self) -> ConstEtomoNumber {
        self.load().max_tilt.base.clone()
    }

    /// Java `getMaxVolcombine`.
    pub fn get_max_volcombine(&self) -> ConstEtomoNumber {
        self.load().max_volcombine.base.clone()
    }

    /// Java package-private `isGpuComputerListEmpty(Node)`.  Returns true if there is
    /// no computerList entries, or false if any of the computerList entries have a
    /// non-local-only GPU.  `ignored_node`: if this node is non-null and present in the
    /// list and has a single GPU, the list is still considered empty.
    pub(crate) fn is_gpu_computer_list_empty(&self, ignored_node: Option<&Arc<Node>>) -> bool {
        let state = self.load();
        if state.computer_list.is_empty() {
            return true;
        }
        for i in 0..state.computer_list.len() {
            let node = &state.computer_map[&state.computer_list[i]];
            let is_ignored = ignored_node.is_some_and(|ignored| Arc::ptr_eq(node, ignored));
            if (ignored_node.is_none() || !is_ignored || (is_ignored && node.get_gpu_number() > 1))
                && node.is_gpu()
            {
                return false;
            }
        }
        true
    }

    /// Java package-private `isGpuQueueListEmpty`.  Returns true if no queue list
    /// entries with a GPU are found.
    pub(crate) fn is_gpu_queue_list_empty(&self) -> bool {
        let state = self.load();
        if state.queue_list.is_empty() {
            return true;
        }
        for i in 0..state.queue_list.len() {
            let node = state.queue_map.get(&state.queue_list[i]);
            if node.is_some_and(|node| node.is_gpu()) {
                return false;
            }
        }
        true
    }

    /// Java package-private `isComputerListEmpty`.
    pub(crate) fn is_computer_list_empty(&self) -> bool {
        self.load().computer_list.is_empty()
    }

    /// Java package-private `isQueueListEmpty`.
    pub(crate) fn is_queue_list_empty(&self) -> bool {
        self.load().queue_list.is_empty()
    }

    /// Java `hasQueues(QueueMode)`.
    pub fn has_queues(&self, queue_mode: Option<QueueMode>) -> bool {
        let state = self.load();
        if state.queue_list.is_empty() {
            return false;
        }
        let queue_mode = match queue_mode {
            None => return true,
            Some(queue_mode) => queue_mode,
        };
        for node in state.queue_map.values() {
            if node.get_queue_mode() == queue_mode {
                return true;
            }
        }
        false
    }

    /// Java `hasSecondaryQueues`.
    pub fn has_secondary_queues(&self) -> bool {
        let state = self.load();
        if state.queue_list.is_empty() {
            return false;
        }
        for node in state.queue_map.values() {
            if node.is_secondary_queue() {
                return true;
            }
        }
        false
    }

    /// Java package-private `getComputer(int)`.  Return
    /// `computerMap.get(computerList[index])` or null.
    pub(crate) fn get_computer_by_index(&self, index: i32) -> Option<Arc<Node>> {
        let state = self.load();
        // `catch (IndexOutOfBoundsException e) { return null; }`.
        if index < 0 || index as usize >= state.computer_list.len() {
            return None;
        }
        state
            .computer_map
            .get(&state.computer_list[index as usize])
            .cloned()
    }

    /// Java `getLocalHostComputer`.  Get the Computer section for the current computer.
    pub fn get_local_host_computer(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> Option<Arc<Node>> {
        // Search for "localhost":
        let local_host = self.get_computer(Some(node::LOCAL_HOST_NAME));
        if local_host.is_some() {
            return local_host;
        }
        // Search for the computer name:
        let local_host_name = Network::get_local_host_name(manager, axis_id, property_user_dir)?;
        self.get_computer(Some(&local_host_name))
    }

    /// Java package-private `getLocalHostCpus`.  Get the number of CPUs for the local
    /// host, or null if there is no local host entry.
    pub(crate) fn get_local_host_cpus(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> Option<i32> {
        let local_host = self.get_local_host_computer(manager, axis_id, property_user_dir)?;
        let cpus = local_host.get_cpus();
        if !cpus.is_null() {
            return Some(cpus.get_int());
        }
        None
    }

    /// Java package-private `getLocalHostGpus`.  Returns the number of GPUs from the
    /// local host section.  Returns null if there is no local host section.
    pub(crate) fn get_local_host_gpus(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> Option<i32> {
        let local_host = self.get_local_host_computer(manager, axis_id, property_user_dir)?;
        Some(local_host.get_total_gpus(manager, axis_id, property_user_dir))
    }

    /// Java package-private `getQueue(int)`.
    ///
    /// CpuAdoc.java:358 indexes `queueList` without a bounds check, so an index out of
    /// range throws `IndexOutOfBoundsException` (and the public `Network.getQueue(int)`
    /// passes any index through).  Fixed in translation: out of range answers null, as
    /// `getComputer(int)` does.
    pub(crate) fn get_queue_by_index(&self, index: i32) -> Option<Arc<Node>> {
        let state = self.load();
        if index < 0 || index as usize >= state.queue_list.len() {
            return None;
        }
        state
            .queue_map
            .get(&state.queue_list[index as usize])
            .cloned()
    }

    /// Java package-private `getQueue(String)`.  Get queue from queueMap by name.
    /// Return null if not found.
    pub(crate) fn get_queue(&self, name: &str) -> Option<Arc<Node>> {
        self.load().queue_map.get(name).cloned()
    }

    /// Java package-private `getComputer(String)`.  Get computer from computerMap by
    /// name.  Return null if not found.
    pub(crate) fn get_computer(&self, host_name: Option<&str>) -> Option<Arc<Node>> {
        let state = self.load();
        let host_name = host_name?;
        if let Some(node) = state.computer_map.get(host_name) {
            return Some(Arc::clone(node));
        }
        // The section name used in cpu.adoc may not be a complete host name.
        for section_name in state.computer_list.iter() {
            if Node::match_host_computer(Some(section_name), Some(host_name)) {
                return state.computer_map.get(section_name).cloned();
            }
        }
        None
    }

    /// Java package-private `getComputerListSize`.
    pub(crate) fn get_computer_list_size(&self) -> i32 {
        self.load().computer_list.len() as i32
    }

    /// Java package-private `getQueueListSize`.
    pub(crate) fn get_queue_list_size(&self) -> i32 {
        self.load().queue_list.len() as i32
    }

    /// Java `isViable`.  Returns true if cpu.adoc exists and contains at least one
    /// section that can be loaded.  A cpu.adoc with no sections that can be loaded is
    /// treated as if it does not exist.
    pub fn is_viable(&self) -> bool {
        let state = self.load();
        state.exists && (!state.computer_list.is_empty() || !state.queue_list.is_empty())
    }

    /// Java private `fileExists`.
    fn file_exists(&self) -> bool {
        self.load().exists
    }

    /// Java private synchronized `load`.  Returns the loaded state, still locked.
    fn load(&self) -> MutexGuard<'_, State> {
        let mut state = self.state.lock().unwrap();
        if state.loaded {
            return state;
        }
        state.loaded = true;
        let autodoc = self.get_autodoc();
        if let Some(autodoc) = autodoc.and_then(|autodoc| unsafe { autodoc.as_ref() })
            && ReadOnlyAutodoc::exists(autodoc)
        {
            state.exists = true;
            state.separate_chunks = self.load_boolean_attribute(autodoc, "separate-chunks");
            self.load_attribute_autodoc(&mut state.min_nice, autodoc, "min", "nice");
            state.users_column = self.load_boolean_attribute(autodoc, "users-column");
            let mut attrib: Option<&Attribute> =
                unsafe { ReadOnlyAutodoc::get_attribute(autodoc, Some("max")).as_ref() };
            if let Some(attrib) = attrib {
                self.load_attribute(&mut state.max_tilt, Some(attrib), "tilt");
                self.load_attribute(&mut state.max_volcombine, Some(attrib), "volcombine");
            }
            attrib = unsafe { ReadOnlyAutodoc::get_attribute(autodoc, Some("units")).as_ref() };
            if let Some(units) = attrib {
                state.speed_units = self.load_string_attribute(Some(units), SPEED_KEY);
                state.memory_units = self.load_string_attribute(Some(units), MEMORY_KEY);
                state.load_units = self.load_string_list_attribute(Some(units), "load");
                attrib = unsafe { units.get_attribute_by_name(Some(GPU_KEY)).as_ref() };
                if let Some(gpu) = attrib {
                    state.gpu_speed_units = self.load_string_attribute(Some(gpu), SPEED_KEY);
                    state.gpu_memory_units = self.load_string_attribute(Some(gpu), MEMORY_KEY);
                }
            }
            attrib = unsafe { ReadOnlyAutodoc::get_attribute(autodoc, Some("pc-option")).as_ref() };
            if let Some(attrib) = attrib {
                // Construct a pcOptionsMap here.
                let mut pc_options_map: PcOptionsMap = Vec::new();
                // CpuAdoc.java:459-460 calls `iterator()` on `getChildren()`, which is
                // null for a `pc-option` attribute with no children, and throws a
                // NullPointerException (likewise for a computer/queue type below).  Fixed
                // in translation: no children, no options.
                if let Some(pc_option_type_list) = unsafe { attrib.get_children().as_ref() } {
                    let mut pc_option_type_iter = pc_option_type_list.iterator();
                    while pc_option_type_iter.has_next() {
                        // Get Computer/Queue or option name
                        let type_attrib: &Attribute =
                            unsafe { &**pc_option_type_iter.next().unwrap() };
                        let key = type_attrib.get_name();
                        let r#type = PcOptionType::get_instance(Some(&key));
                        if r#type.is_some() {
                            // Get option and value for computer/queue type
                            if let Some(pc_option_option_list) =
                                unsafe { type_attrib.get_children().as_ref() }
                            {
                                let mut pc_option_option_iter = pc_option_option_list.iterator();
                                while pc_option_option_iter.has_next() {
                                    let option_attrib: &Attribute =
                                        unsafe { &**pc_option_option_iter.next().unwrap() };
                                    let option = option_attrib.get_name();
                                    let value = option_attrib.get_value();
                                    match pc_options_map.iter_mut().find(|(k, _)| *k == option) {
                                        Some((_, element)) => {
                                            element.set_pc_option_element(r#type, value.as_deref());
                                        }
                                        None => {
                                            let element = PcOptionElement::new(
                                                Some(&option),
                                                r#type,
                                                value.as_deref(),
                                            );
                                            pc_options_map.push((option, element));
                                        }
                                    }
                                }
                            }
                        } else {
                            // For "pc-option.A=xyz", where types like computer/queue are
                            // not present.
                            let value = type_attrib.get_value();
                            match pc_options_map.iter_mut().find(|(k, _)| *k == key) {
                                Some((_, element)) => {
                                    element.set_pc_option_element(r#type, value.as_deref());
                                }
                                None => {
                                    let element =
                                        PcOptionElement::new(Some(&key), r#type, value.as_deref());
                                    pc_options_map.push((key, element));
                                }
                            }
                        }
                    }
                }
                state.pc_options_map = Some(pc_options_map);
            }
            unsafe { self.load_computers(&mut state, autodoc) };
            unsafe { self.load_queues(&mut state, autodoc) };
        }
        let err_msg = self.validate(&state);
        if let Some(err_msg) = err_msg {
            let workaround = self.workaround(&state);
            // `UIHarness.INSTANCE.openMessageDialog(errMsg, workaround())`; `load` runs
            // on whichever thread first asks, so the dialog is posted to the event
            // dispatch thread.  The harness takes the additional messages as a sorted
            // set where the source keeps a `LinkedHashSet`.
            let additional_messages: Option<Vec<String>> = workaround;
            event_queue::invoke_later(move || {
                ui_harness::INSTANCE.with(|harness| {
                    harness.open_message_dialog_string_array_linked_hash_set(
                        Some(err_msg.as_slice()),
                        additional_messages.as_deref(),
                    );
                });
            });
        }
        state
    }

    /// Java private `validate`.  Performs validations meant to run after loading.
    /// Returns null if valid, otherwise returns an error title and message.
    fn validate(&self, state: &State) -> Option<Vec<String>> {
        // A warning needs to be popped up for a queue having a coresPerNode entry but no
        // initialize command (Bug# 2439). CoresPerClusterJob is now used for things that
        // used to be covered by coresPerNode.
        if !state.queue_list.is_empty() {
            let mut invalid_name_list: Option<Vec<String>> = None;
            for node in state.queue_map.values() {
                if !node.validate() {
                    invalid_name_list
                        .get_or_insert_with(Vec::new)
                        .push(node.get_name().to_string());
                }
            }
            if invalid_name_list.is_some() {
                return Some(vec![
                    format!("Error in {}", *FILE_NAME),
                    format!(
                        "Invalid queue(s) found in {} sections in the {} file.  Invalid queue(s) have the coresPerNode attribute but no initialize attribute.",
                        self.queue_section_type.as_deref().unwrap_or("null"),
                        *FILE_NAME
                    ),
                ]);
            }
        }
        None
    }

    /// Java private `workaround`.  Let's each queue node perform a workaround.  The
    /// Java `LinkedHashSet` is kept as an ordered list of unique messages.
    fn workaround(&self, state: &State) -> Option<Vec<String>> {
        // When a queue has a CoresPerNode entry but no initialize command (Bug# 2439),
        // use CoresPerNode to fill in CoresPerClusterJob if CoresPerClusterJob is
        // missing.
        let mut workaround_messages: Option<Vec<String>> = None;
        if !state.queue_list.is_empty() {
            for node in state.queue_map.values() {
                // Saved unique workaround messages in order.
                let temp = node.workaround();
                if let Some(temp) = temp {
                    let messages = workaround_messages.get_or_insert_with(Vec::new);
                    if !messages.contains(&temp) {
                        messages.push(temp);
                    }
                }
            }
        }
        workaround_messages
    }

    /// Java private `loadStringAttribute(ReadOnlyAttribute, String)`.
    fn load_string_attribute(&self, attrib: Option<&Attribute>, key: &str) -> Option<String> {
        let attrib = match attrib {
            None => return Some(String::new()),
            Some(attrib) => attrib,
        };
        match unsafe { attrib.get_attribute_by_name(Some(key)).as_ref() } {
            None => Some(String::new()),
            Some(attrib) => attrib.get_value(),
        }
    }

    /// Java private `loadAttribute(EtomoNumber, ReadOnlyAttribute, String)`.
    fn load_attribute(&self, number: &mut EtomoNumber, attrib: Option<&Attribute>, key: &str) {
        number.reset();
        let attrib = match attrib {
            None => return,
            Some(attrib) => attrib,
        };
        let attrib = match unsafe { attrib.get_attribute_by_name(Some(key)).as_ref() } {
            None => return,
            Some(attrib) => attrib,
        };
        number.set_string(attrib.get_value().as_deref());
    }

    /// Java private `loadStringListAttribute(ReadOnlyAttribute, String)`.
    fn load_string_list_attribute(&self, attrib: Option<&Attribute>, key: &str) -> Vec<String> {
        let attrib = match attrib {
            None => return Vec::new(),
            Some(attrib) => attrib,
        };
        let attrib = match unsafe { attrib.get_attribute_by_name(Some(key)).as_ref() } {
            None => return Vec::new(),
            Some(attrib) => attrib,
        };
        match attrib.get_value() {
            None => Vec::new(),
            Some(list) => java_lang_string_split(&list, &COMMA_PATTERN),
        }
    }

    /// Java private `getAutodoc`.
    ///
    /// The source passes a null `AxisID`; the Rust factory takes a non-null one, and
    /// `AxisID.ONLY` stands in (the cpu autodoc is not per-axis).
    fn get_autodoc(&self) -> Option<*mut Autodoc> {
        let mut autodoc: Option<*mut Autodoc> = None;
        match unsafe {
            autodoc_factory::get_instance(None, Some(autodoc_factory::CPU), AxisID::Only, false)
        } {
            Ok(instance) => {
                if !instance.is_null() {
                    autodoc = Some(instance);
                }
            }
            // `catch (final LockException e) {}`.
            Err(LogFileError::Lock(_)) => {}
            // `e.printStackTrace()`.
            Err(e) => eprintln!("{}", e),
        }
        if autodoc.is_none() {
            eprintln!(
                "Missing ${}/cpu.adoc file.\nParallel processing cannot be used.\nSee $IMOD_DIR/autodoc/cpu.adoc.",
                environment_variable::CALIB_DIR
            );
        }
        autodoc
    }

    /// Java private `loadComputers(ReadOnlyAutodoc)`.
    ///
    /// # Safety
    /// `autodoc` must be live.
    unsafe fn load_computers(&self, state: &mut State, autodoc: &Autodoc) {
        let computer_section_type = match &self.computer_section_type {
            None => return,
            Some(computer_section_type) => computer_section_type,
        };
        let location =
            ReadOnlySectionList::get_section_location_by_type(autodoc, Some(computer_section_type));
        let mut location = match location {
            None => return,
            Some(location) => location,
        };
        loop {
            let section =
                unsafe { ReadOnlySectionList::next_section(autodoc, Some(&mut location)) };
            let section = match unsafe { section.as_ref() } {
                None => break,
                Some(section) => section,
            };
            let mut computer = Node::get_computer_instance();
            unsafe { computer.load(section) };
            if !computer.is_excluded_user(self.user.as_deref()) {
                let name = ReadOnlyStatementList::get_name(section).unwrap_or_default();
                let computer = Arc::new(computer);
                state.computer_list.push(name.clone());
                state.computer_map.insert(name, Arc::clone(&computer));
                state
                    .or_list_map
                    .add_node(ProcessorType::Cpu, Some(&computer));
            }
        }
    }

    /// Java `isNumberGt1(InterfaceType, ProcessorType)`.
    pub fn is_number_gt1(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        state
            .or_list_map
            .is_value_true(interface_type, processor_type, Some(OrValue::NumberGt1))
    }

    /// Java `isGpusPerClusterJob(InterfaceType, ProcessorType)`.
    pub fn is_gpus_per_cluster_job(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        state.or_list_map.is_value_true(
            interface_type,
            processor_type,
            Some(OrValue::GpusPerClusterJob),
        )
    }

    /// Java `isGpuGt1(InterfaceType, ProcessorType)`.
    pub fn is_gpu_gt1(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        state
            .or_list_map
            .is_value_true(interface_type, processor_type, Some(OrValue::GpuGt1))
    }

    /// Java `isType(InterfaceType, ProcessorType)`.
    pub fn is_type(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        if processor_type == ProcessorType::Gpu {
            return false;
        }
        state
            .or_list_map
            .is_value_true(interface_type, processor_type, Some(OrValue::CpuType))
    }

    /// Java `isSpeed(InterfaceType, ProcessorType)`.
    pub fn is_speed(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        if processor_type == ProcessorType::Gpu {
            return false;
        }
        state
            .or_list_map
            .is_value_true(interface_type, processor_type, Some(OrValue::Speed))
    }

    /// Java `isMemory(InterfaceType, ProcessorType)`.
    pub fn is_memory(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        if processor_type == ProcessorType::Gpu {
            return false;
        }
        state
            .or_list_map
            .is_value_true(interface_type, processor_type, Some(OrValue::Memory))
    }

    /// Java `isOs(InterfaceType, ProcessorType)`.
    pub fn is_os(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        state
            .or_list_map
            .is_value_true(interface_type, processor_type, Some(OrValue::Os))
    }

    /// Java `isGpuType(InterfaceType, ProcessorType)`.
    pub fn is_gpu_type(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        if processor_type == ProcessorType::Cpu {
            return false;
        }
        state
            .or_list_map
            .is_value_true(interface_type, processor_type, Some(OrValue::GpuType))
    }

    /// Java `isGpuSpeed(InterfaceType, ProcessorType)`.
    pub fn is_gpu_speed(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        if processor_type == ProcessorType::Cpu {
            return false;
        }
        state
            .or_list_map
            .is_value_true(interface_type, processor_type, Some(OrValue::GpuSpeed))
    }

    /// Java `isGpuMemory(InterfaceType, ProcessorType)`.
    pub fn is_gpu_memory(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        if processor_type == ProcessorType::Cpu {
            return false;
        }
        state
            .or_list_map
            .is_value_true(interface_type, processor_type, Some(OrValue::GpuMemory))
    }

    /// Java `isGpuNcores(InterfaceType, ProcessorType)`.
    pub fn is_gpu_ncores(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> bool {
        let state = self.load();
        if processor_type == ProcessorType::Cpu {
            return false;
        }
        state
            .or_list_map
            .is_value_true(interface_type, processor_type, Some(OrValue::GpuNcores))
    }

    /// Java private `loadQueues(ReadOnlyAutodoc)`.
    ///
    /// # Safety
    /// `autodoc` must be live.
    unsafe fn load_queues(&self, state: &mut State, autodoc: &Autodoc) {
        let queue_section_type = match &self.queue_section_type {
            None => return,
            Some(queue_section_type) => queue_section_type,
        };
        let location =
            ReadOnlySectionList::get_section_location_by_type(autodoc, Some(queue_section_type));
        let mut location = match location {
            None => return,
            Some(location) => location,
        };
        loop {
            let section =
                unsafe { ReadOnlySectionList::next_section(autodoc, Some(&mut location)) };
            let section = match unsafe { section.as_ref() } {
                None => break,
                Some(section) => section,
            };
            let mut queue = Node::get_queue_instance();
            unsafe { queue.load(section) };
            if !queue.is_excluded_user(self.user.as_deref()) {
                let name = ReadOnlyStatementList::get_name(section).unwrap_or_default();
                let queue = Arc::new(queue);
                state.queue_list.push(name.clone());
                state.queue_map.insert(name, Arc::clone(&queue));
                state
                    .or_list_map
                    .add_node(ProcessorType::Queue, Some(&queue));
            }
        }
    }

    /// Java private `loadBooleanAttribute(ReadOnlyAutodoc, String)`.
    fn load_boolean_attribute(&self, autodoc: &Autodoc, key: &str) -> bool {
        let attrib: Option<&Attribute> =
            unsafe { ReadOnlyAutodoc::get_attribute(autodoc, Some(key)).as_ref() };
        if let Some(attrib) = attrib
            && attrib
                .get_value()
                .as_deref()
                .is_none_or(|value| value != "0")
        {
            return true;
        }
        false
    }

    /// Java private `loadAttribute(EtomoNumber, ReadOnlyAutodoc, String, String)`.
    fn load_attribute_autodoc(
        &self,
        number: &mut EtomoNumber,
        autodoc: &Autodoc,
        key1: &str,
        key2: &str,
    ) {
        number.reset();
        let attrib = match unsafe { ReadOnlyAutodoc::get_attribute(autodoc, Some(key1)).as_ref() } {
            None => return,
            Some(attrib) => attrib,
        };
        let attrib = match unsafe { attrib.get_attribute_by_name(Some(key2)).as_ref() } {
            None => return,
            Some(attrib) => attrib,
        };
        number.set_string(attrib.get_value().as_deref());
    }

    /// Java private `loadStringAttribute(ReadOnlyAutodoc, String, String)`.
    fn load_string_attribute_autodoc(
        &self,
        autodoc: &Autodoc,
        key1: &str,
        key2: &str,
    ) -> Option<String> {
        let attrib = match unsafe { ReadOnlyAutodoc::get_attribute(autodoc, Some(key1)).as_ref() } {
            None => return Some(String::new()),
            Some(attrib) => attrib,
        };
        match unsafe { attrib.get_attribute_by_name(Some(key2)).as_ref() } {
            None => Some(String::new()),
            Some(attrib) => attrib.get_value(),
        }
    }

    /// Java private `loadStringListAttribute(ReadOnlyAutodoc, String, String)`.
    fn load_string_list_attribute_autodoc(
        &self,
        autodoc: &Autodoc,
        key1: &str,
        key2: &str,
    ) -> Vec<String> {
        let attrib = match unsafe { ReadOnlyAutodoc::get_attribute(autodoc, Some(key1)).as_ref() } {
            None => return Vec::new(),
            Some(attrib) => attrib,
        };
        let attrib = match unsafe { attrib.get_attribute_by_name(Some(key2)).as_ref() } {
            None => return Vec::new(),
            Some(attrib) => attrib,
        };
        match attrib.get_value() {
            None => Vec::new(),
            Some(list) => java_lang_string_split(&list, &COMMA_PATTERN),
        }
    }

    /// Java `getPcOptionsMap` (no `load`).  The source returns its own map; the entries
    /// are copied here because the instance is shared.
    pub fn get_pc_options_map(&self) -> Option<PcOptionsMap> {
        self.state.lock().unwrap().pc_options_map.clone()
    }
}

/// Java private static final class `OrListMap`.  Keeps track of boolean values derived
/// from the cpu.adoc nodes.  Each value is true if it is true for any node that meets
/// the criteria.  The orMap member variable will contain a set of these values for each
/// sectionType (computer and queue), and each excluded interface mentioned in the
/// cpu.adoc.  This class does not take the user list into account.
struct OrListMap {
    /// Java private field `interfaceTypeSet`: section type + interface.  Its one walk
    /// ORs values into each member's list, which does not depend on the order.
    interface_type_set: Option<HashSet<String>>,
    /// Java private field `orMap`: key is section type + interface, value is an orList.
    or_map: Option<HashMap<String, [bool; OR_VALUE_TOTAL]>>,
}

impl OrListMap {
    /// Java private constructor `OrListMap()`.
    fn new() -> OrListMap {
        OrListMap {
            interface_type_set: None,
            or_map: None,
        }
    }

    /// Java private `isValueTrue`.  Returns true of the orValue for this processor and
    /// interface type is true.
    fn is_value_true(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
        or_value: Option<OrValue>,
    ) -> bool {
        let (or_value, or_map) = match (or_value, &self.or_map) {
            (Some(or_value), Some(or_map)) => (or_value, or_map),
            _ => return false,
        };
        // First look for the interfaceType-level orList
        if let Some(or_list) = or_map.get(&self.get_key_interface(interface_type, processor_type)) {
            return or_list[or_value.index()];
        }
        // If the interfaceType-level orList doesn't exist, use the processor-level
        // orList.
        if let Some(or_list) = or_map.get(self.get_key(processor_type)) {
            return or_list[or_value.index()];
        }
        false
    }

    /// Java private `addNode`.  Add the boolean values contained in the node parameter.
    ///
    /// CpuAdoc.java:880-881 tests `interfaceTypeSet.contains(excludeInterfaceType
    /// .toString())` but adds `getKey(excludeInterfaceType, processorType)` (the section
    /// type plus the interface) to the set, so the test never matches and every node
    /// with an excluded interface re-copies the processor-level orList over the
    /// interface-level one, wiping out the exclusion the earlier nodes established.
    /// Fixed in translation: the test uses the key the set holds.
    fn add_node(&mut self, processor_type: ProcessorType, node: Option<&Arc<Node>>) {
        let node = match node {
            None => return,
            Some(node) => node,
        };
        let section_key = self.get_key(processor_type).to_string();
        // If there is an excluded interface, and hasn't been seen before, add it to the
        // interface name list.
        let exclude_interface_type = node.get_exclude_interface();
        if let Some(exclude_interface_type) = exclude_interface_type {
            let interface_key =
                self.get_key_interface(Some(exclude_interface_type), processor_type);
            // The excluded interface name is new - add it to the interface name list.
            if self
                .interface_type_set
                .as_ref()
                .is_none_or(|set| !set.contains(&interface_key))
            {
                self.interface_type_set
                    .get_or_insert_with(HashSet::new)
                    .insert(interface_key.clone());
                // The new interface name gets a copy of the orList for this
                // processorType.
                if let Some(or_map) = &mut self.or_map
                    && let Some(or_list) = or_map.get(&section_key).copied()
                {
                    let mut new_or_list = OrListMap::create_or_list();
                    for i in 0..OR_VALUE_TOTAL {
                        new_or_list[i] = or_list[i];
                    }
                    or_map.insert(interface_key, new_or_list);
                }
            }
        }
        // Set any true boolean values in the processor-level orList
        let mut value_list = OrListMap::create_or_list();
        let mut true_value_found = false;
        for i in 0..OR_VALUE_TOTAL {
            // Hold on to the values from the node.
            value_list[i] = OrValue::ARRAY[i].is_true(node);
            if value_list[i] {
                true_value_found = true;
                let or_map = self.or_map.get_or_insert_with(HashMap::new);
                let or_list = or_map
                    .entry(section_key.clone())
                    .or_insert_with(OrListMap::create_or_list);
                or_list[i] = true;
            }
        }
        // Set any true boolean values in any interface-level orLists
        if true_value_found
            && let Some(interface_type_set) = &self.interface_type_set
            && self.or_map.is_some()
        {
            let keys: Vec<String> = interface_type_set.iter().cloned().collect();
            for key in keys.iter() {
                let interface_type = self.decode_key(processor_type, Some(key));
                if let Some(interface_type) = interface_type
                    && Some(interface_type) != exclude_interface_type
                {
                    for i in 0..value_list.len() {
                        if value_list[i] {
                            let key = self.get_key_interface(Some(interface_type), processor_type);
                            let or_map = self.or_map.as_mut().unwrap();
                            let or_list =
                                or_map.entry(key).or_insert_with(OrListMap::create_or_list);
                            or_list[i] = true;
                        }
                    }
                }
            }
        }
    }

    /// Java private `createOrList`.
    fn create_or_list() -> [bool; OR_VALUE_TOTAL] {
        [false; OR_VALUE_TOTAL]
    }

    /// Java private `getKey(ProcessorType)`.  Returns the default queue section type
    /// when the queue processor type is passed in.  Otherwise returns the default
    /// computer section type.
    fn get_key(&self, processor_type: ProcessorType) -> &'static str {
        if processor_type == ProcessorType::Queue {
            return DEFAULT_QUEUE_SECTION_TYPE;
        }
        DEFAULT_COMPUTER_SECTION_TYPE
    }

    /// Java private `getKey(InterfaceType, ProcessorType)`.
    fn get_key_interface(
        &self,
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
    ) -> String {
        format!(
            "{}{}",
            self.get_key(processor_type),
            match interface_type {
                Some(interface_type) => format!(".{}", interface_type),
                None => String::new(),
            }
        )
    }

    /// Java private `decodeKey`.  Returns the interface type in the key, if the key
    /// contains a matching processor type.
    fn decode_key(
        &self,
        processor_type: ProcessorType,
        key: Option<&str>,
    ) -> Option<InterfaceType> {
        let first_part = self.get_key(processor_type);
        let key = key?;
        if !key.contains('.') || !key.starts_with(first_part) {
            return None;
        }
        let array = java_lang_string_split(key, &DOT_PATTERN);
        if array.len() < 2 {
            return None;
        }
        InterfaceType::get_instance(Some(&array[1]))
    }
}

/// Java private static final `OrValue.TOTAL`: total number of class instances.
const OR_VALUE_TOTAL: usize = 11;

/// Java private static final class `OrValue`.  An enum of boolean values derived from
/// the cpu.adoc nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrValue {
    /// Java `CPU_TYPE` (`Node.TYPE_KEY`).
    CpuType,
    /// Java `NUMBER_GT_1` (`Node.NUMBER_KEY`).
    NumberGt1,
    /// Java `GPU_GT_1` (`Node.NUMBER_KEY`).
    GpuGt1,
    /// Java `MEMORY` (`Node.MEMORY_KEY`).
    Memory,
    /// Java `OS` (`Node.OS_KEY`).
    Os,
    /// Java `SPEED` (`Node.SPEED_KEY`).
    Speed,
    /// Java `GPU_TYPE` (`Node.TYPE_KEY`).
    GpuType,
    /// Java `GPU_MEMORY` (`Node.MEMORY_KEY`).
    GpuMemory,
    /// Java `GPU_NCORES` (`Node.NCORES_KEY`).
    GpuNcores,
    /// Java `GPU_SPEED` (`Node.SPEED_KEY`).
    GpuSpeed,
    /// Java `GPUS_PER_CLUSTER_JOB` (`Node.GPUS_PER_CLUSTER_JOB_KEY`).
    GpusPerClusterJob,
}

impl OrValue {
    /// Java private static `ARRAY`, in construction (index) order.
    const ARRAY: [OrValue; OR_VALUE_TOTAL] = [
        OrValue::CpuType,
        OrValue::NumberGt1,
        OrValue::GpuGt1,
        OrValue::Memory,
        OrValue::Os,
        OrValue::Speed,
        OrValue::GpuType,
        OrValue::GpuMemory,
        OrValue::GpuNcores,
        OrValue::GpuSpeed,
        OrValue::GpusPerClusterJob,
    ];

    /// Java private field `key`.
    fn key(self) -> &'static str {
        match self {
            OrValue::CpuType | OrValue::GpuType => node::TYPE_KEY,
            OrValue::NumberGt1 | OrValue::GpuGt1 => node::NUMBER_KEY,
            OrValue::Memory | OrValue::GpuMemory => node::MEMORY_KEY,
            OrValue::Os => node::OS_KEY,
            OrValue::Speed | OrValue::GpuSpeed => node::SPEED_KEY,
            OrValue::GpuNcores => node::NCORES_KEY,
            OrValue::GpusPerClusterJob => node::GPUS_PER_CLUSTER_JOB_KEY,
        }
    }

    /// Java private field `index`.
    fn index(self) -> usize {
        self as usize
    }

    /// Java private `isTrue(Node)`.
    fn is_true(self, node: &Node) -> bool {
        match self {
            OrValue::CpuType => node.is_type(),
            OrValue::NumberGt1 => node.is_number_gt1(),
            OrValue::GpuGt1 => node.is_gpu_gt1(),
            OrValue::Memory => node.is_memory(),
            OrValue::Os => node.is_os(),
            OrValue::Speed => node.is_speed(),
            OrValue::GpuType => node.is_gpu_type(),
            OrValue::GpuMemory => node.is_gpu_memory(),
            OrValue::GpuNcores => node.is_gpu_ncores(),
            OrValue::GpuSpeed => node.is_gpu_speed(),
            OrValue::GpusPerClusterJob => node.is_gpus_per_cluster_job(),
        }
    }
}
