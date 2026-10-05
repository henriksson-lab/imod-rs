//! `IMOD/Etomo/src/etomo/process/ProcessData.java`.
//!
//! Process data to allow identification of processes that Etomo is no longer
//! managing because it exited after they started.  Allows Etomo to prevent two
//! processes from running on an axis, even when the running process is unmanaged.
//! Saved in the data file.  The process is found again with `ps` (`PsParam`, run
//! through `SystemProgram`, over ssh for another host); the host is the one
//! `Network.getLocalHostName` (`b3dhostname`) reports.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::ps_param::PsParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::storage::storable::StorableValue;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_string_property::ConstStringProperty;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::os_type::{self, OSType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::string_property::StringProperty;
use crate::imod::etomo::r#type::time::Time;
use crate::imod::etomo::util::java_hash_map::JavaHashMap;
use std::collections::BTreeMap;
use std::convert::Infallible;

const PID_KEY: &str = "PID";
const GROUP_PID_KEY: &str = "GroupPID";
const START_TIME_KEY: &str = "StartTime";
const PROCESS_NAME_KEY: &str = "ProcessName";
/// Java declares `OS_TYPE_KEY = "OS"` and never reads it; the OS is stored under
/// `OSType.KEY`.
const OS_TYPE_KEY: &str = "OS";
const COMPUTER_KEY: &str = "Computer";
const LINE_NUMBER_KEY: &str = "LineNumber";
const LINE_NUMBER_DEFAULT: i32 = 0;

/// Java final `ProcessData implements Storable`.
pub struct ProcessData {
    /// Java `EtomoNumber displayID = new EtomoNumber("DisplayID")`.
    display_id: EtomoNumber,
    /// Java `StringProperty factoryID = new StringProperty("FactoryID")`.
    factory_id: StringProperty,
    /// Java `StringProperty subProcessName`.
    sub_process_name: StringProperty,
    /// Java `StringProperty subDirName`.
    sub_dir_name: StringProperty,
    /// Java `StringProperty hostName`.
    host_name: StringProperty,
    /// Java `StringProperty lastProcess`.
    last_process: StringProperty,
    /// Java `StringProperty secondaryQueue`.
    secondary_queue: StringProperty,
    axis_id: AxisID,
    process_data_prepend: String,
    /// Java final `manager`.  Managers are process-lifetime objects in both
    /// implementations, hence the shared static reference convention.
    manager: Option<&'static dyn BaseManager>,
    pid: Option<String>,
    group_pid: Option<String>,
    /// Java `Time startTime`.
    start_time: Option<Time>,
    process_name: Option<ProcessName>,
    do_not_load: bool,
    /// Java `OSType osType`.
    os_type: Option<OSType>,
    ssh_failed: bool,
    computer_map: Option<BTreeMap<String, String>>,
    /// Java `ProcessingMethod`.
    processing_method: Option<ProcessingMethod>,
    dialog_type: Option<DialogType>,
    /// Java `debug = EtomoDirector.INSTANCE.getArguments().getDebugLevel()`.
    debug: DebugLevel,
    line_number: i32,
    /// Java `EtomoNumber numDone = new EtomoNumber("NumDone")`.
    num_done: EtomoNumber,
    chunk_map: Option<BTreeMap<i32, Chunk>>,
}

impl ProcessData {
    /// Java package-private `ProcessData(AxisID, BaseManager)`.
    pub fn new(axis_id: Option<AxisID>, manager: Option<&'static dyn BaseManager>) -> Self {
        let axis_id = match axis_id {
            Some(AxisID::Only) | None => AxisID::First,
            Some(axis) => axis,
        };
        // `displayID.setDisplayValue(-1)`, `numDone.setDisplayValue(0)`.
        let mut display_id = EtomoNumber::new_with_name("DisplayID");
        display_id.set_display_value_int(-1);
        let mut num_done = EtomoNumber::new_with_name("NumDone");
        num_done.set_display_value_int(0);
        Self {
            display_id,
            factory_id: StringProperty::new_with_key(Some("FactoryID")),
            sub_process_name: StringProperty::new_with_key(Some("SubProcessName")),
            sub_dir_name: StringProperty::new_with_key(Some("SubDirName")),
            host_name: StringProperty::new_with_key(Some("HostName")),
            last_process: StringProperty::new_with_key(Some("LastProcess")),
            secondary_queue: StringProperty::new_with_key(Some("SecondaryQueue")),
            axis_id,
            process_data_prepend: format!("ProcessData.{}", axis_id.get_extension()),
            manager,
            pid: None,
            group_pid: None,
            start_time: None,
            process_name: None,
            do_not_load: false,
            os_type: None,
            ssh_failed: false,
            computer_map: None,
            processing_method: None,
            dialog_type: None,
            debug: etomo_director::ARGUMENTS.lock().unwrap().get_debug_level(),
            line_number: LINE_NUMBER_DEFAULT,
            num_done,
            chunk_map: None,
        }
    }

    /// Java static `getManagedInstance`.  Get an instance of ProcessData which is
    /// associated with a process managed by Etomo.  It cannot be loaded from the param
    /// file.  This instance will have a fixed process name, and a host name and OS taken
    /// from the current computer.
    pub fn get_managed_instance(
        axis_id: Option<AxisID>,
        manager: Option<&'static dyn BaseManager>,
        process_name: Option<ProcessName>,
    ) -> Self {
        let mut process_data = Self::new(axis_id, manager);
        process_data.process_name = process_name;
        // `processData.hostName.set(Network.getLocalHostName(manager, axisID,
        // manager.getPropertyUserDir()))`; every caller passes its manager.
        if let Some(manager) = manager {
            let host_name = Network::get_local_host_name(
                manager,
                axis_id.unwrap_or(AxisID::Only),
                manager.get_property_user_dir().as_deref(),
            );
            process_data.host_name.set(host_name.as_deref());
        }
        process_data.os_type = Some(OSType::get_instance());
        process_data.do_not_load = true;
        process_data
    }
    /// Java package-private `dumpState`.
    ///
    /// `computerMap.toString()` lists a Java `HashMap`, in Java's order
    /// (`JavaHashMap`).  The map is kept as a `BTreeMap` here, so two computers that
    /// share a `HashMap` bucket are listed in key order, where Java lists them in the
    /// order they were put (the processchunks machine list, or the `Properties`
    /// order of the data file's keys on a reconnect).
    pub fn dump_state(&self) {
        eprintln!(
            "[processDataPrepend:{},pid:{},\ngroupPid:{},doNotLoad:{},sshFailed:{},computerMap:",
            self.process_data_prepend,
            self.pid.as_deref().unwrap_or("null"),
            self.group_pid.as_deref().unwrap_or("null"),
            self.do_not_load,
            self.ssh_failed
        );
        if let Some(computer_map) = &self.computer_map {
            // Every Java computerMap is a `new HashMap<String, String>()` filled
            // by `put`.
            let mut java_map: JavaHashMap<String, String> = JavaHashMap::new();
            for (key, value) in computer_map {
                java_map.insert(key.clone(), value.clone());
            }
            eprintln!(
                "{}",
                java_map.to_java_string(|key| key.clone(), |value| value.clone())
            );
        }
        self.print_chunk_map();
    }
    /// Java `toString`.
    pub fn to_source_string(&self) -> String {
        format!(
            "ProcessData values:\nprocessName={:?}\npid={:?}\ngroupPid={:?}\nstartTime={:?}\nsubProcessName={:?}\nsubDirName={:?}\nhostName={:?}\nosType={:?}\ndisplayID={}\nfactoryID={:?}\ndialogType={:?}\nlastProcess={:?}\nprocessingMethod:{:?}",
            self.process_name,
            self.pid,
            self.group_pid,
            self.start_time,
            self.sub_process_name.to_string_option(),
            self.sub_dir_name.to_string_option(),
            self.host_name.to_string_option(),
            self.os_type,
            self.display_id.to_string(),
            self.factory_id.to_string_option(),
            self.dialog_type,
            self.last_process.to_string_option(),
            self.processing_method
        )
    }
    /// Java package-private `setDisplayKey`.
    pub fn set_display_key(
        &mut self,
        process_result_display: Option<
            &dyn crate::imod::etomo::r#type::process_result_display::ProcessResultDisplay,
        >,
    ) {
        if let Some(process_result_display) = process_result_display {
            self.display_id
                .set_int(process_result_display.get_display_id());
            self.factory_id
                .set(process_result_display.get_factory_id().as_deref());
        }
    }
    /// Java package-private `setDialogType`.
    pub fn set_dialog_type(&mut self, input: Option<DialogType>) {
        self.dialog_type = input;
    }
    /// Java `getDialogType`.
    pub fn get_dialog_type(&self) -> Option<DialogType> {
        self.dialog_type
    }
    /// Java package-private `setLastProcess`.
    pub fn set_last_process(
        &mut self,
        process_series: &crate::imod::etomo::process_series::ProcessSeries,
        resumable: bool,
    ) {
        if process_series.will_process_list_be_dropped() && resumable {
            eprintln!("WARNING:  Not compatible with ProcessSeries.processList.");
        }
        self.last_process
            .set(process_series.get_last_process().as_deref());
    }
    /// Java `getLastProcess`.
    pub fn get_last_process(&self) -> Option<String> {
        if self.last_process.is_empty() {
            return None;
        }
        self.last_process.to_string_option()
    }
    /// Java `getSecondaryQueue`.
    pub fn get_secondary_queue(&self) -> Option<String> {
        if self.secondary_queue.is_empty() {
            return None;
        }
        self.secondary_queue.to_string_option()
    }
    /// Java package-private `setSubProcessName`.
    pub fn set_sub_process_name(&mut self, input: Option<&str>) {
        self.sub_process_name.set(input);
    }
    /// Java package-private `setSubDirName`.
    pub fn set_sub_dir_name(&mut self, input: Option<&str>) {
        self.sub_dir_name.set(input);
    }
    /// Java `isEmpty`.
    pub fn is_empty(&self) -> bool {
        self.pid.is_none() || self.group_pid.is_none() || self.start_time.is_none()
    }
    /// Java `isRunning`.  Look for the process data in the ps output.  Returns true if
    /// the process data is found in the ps output, false if the process data is not
    /// found or this instance is empty.
    pub fn is_running(&mut self) -> bool {
        if self.is_empty() {
            return false;
        }
        let pid = self.pid.clone();
        let Some(mut param) = self.run_ps(pid.as_deref()) else {
            return false;
        };
        param.find_row_with_start_time(
            self.pid.as_deref(),
            self.group_pid.as_deref(),
            self.start_time.as_ref(),
        )
    }
    /// Java `isOnDifferentHost`.
    pub fn is_on_different_host(&self) -> bool {
        if !self.host_name.is_empty() {
            let Some(manager) = self.manager else {
                return false;
            };
            return !self.host_name.equals(
                Network::get_local_host_name(
                    manager,
                    self.axis_id,
                    manager.get_property_user_dir().as_deref(),
                )
                .as_deref(),
            );
        }
        false
    }
    /// Java `isSshFailed`.
    pub fn is_ssh_failed(&self) -> bool {
        self.ssh_failed
    }
    /// Java package-private `setComputerMap`.
    pub fn set_computer_map(&mut self, computer_map: Option<BTreeMap<String, String>>) {
        self.computer_map = computer_map;
    }
    /// Java package-private `setSecondaryQueue`.
    pub fn set_secondary_queue(&mut self, secondary_queue: Option<&str>) {
        self.secondary_queue.set(secondary_queue);
    }
    /// Java `setProcessingMethod`.
    pub fn set_processing_method(&mut self, processing_method: Option<ProcessingMethod>) {
        self.processing_method = processing_method;
    }
    /// Java package-private `setPid`.  Use the pid to get the process data from the ps
    /// output.
    pub fn set_pid(&mut self, pid: Option<&str>) {
        self.pid = None;
        self.group_pid = None;
        self.start_time = None;
        let Some(pid) = pid else {
            return;
        };
        // `pid.matches("\\s*+")`
        if pid.chars().all(char::is_whitespace) {
            return;
        }
        let Some(mut param) = self.run_ps(Some(pid)) else {
            return;
        };
        let mut row = param.get_row();
        if row.find(Some(pid)) {
            self.pid = Some(pid.to_owned());
            self.group_pid = row.get_group_pid();
            self.start_time = row.get_start_time();
        }
    }
    /// Java private `runPs`.  Run ps.  Use the -p pid option.
    ///
    /// Java dereferences `manager` for the property user directory; every
    /// instance that reaches here was built with one.  Without one there is no
    /// `ps` to run and `None` stands for the missing row.
    pub fn run_ps(&mut self, pid: Option<&str>) -> Option<PsParam> {
        if self.debug.is_verbose() {
            eprintln!("ProcessData.runPs");
        }
        let manager = self.manager?;
        // `osType` is null only for an instance loaded without an OS entry, which
        // `OSType.getInstance(props, prepend)` never leaves; Linux is its default.
        let mut param = PsParam::new(
            manager,
            self.axis_id,
            pid,
            self.os_type.unwrap_or(os_type::DEFAULT),
            // `hostName.toString()`: "" when empty.
            self.host_name.to_string_option().as_deref(),
            false,
        );
        let ps = SystemProgram::new_array(
            Some(manager),
            manager.get_property_user_dir(),
            Some(param.get_command_array().clone()),
            self.axis_id,
        );
        ps.run();
        let stdout = ps.get_std_output();
        // Ps should always return something - usually as header, but on Mac it will
        // return a bunch of result lines because we are not using the -p pid option.
        self.ssh_failed = match &stdout {
            None => true,
            Some(stdout) => stdout.is_empty(),
        };
        param.set_output(stdout.map(|stdout| stdout.into_iter().map(Some).collect()));
        Some(param)
    }
    /// Java `store(Properties)`.
    pub fn store_properties(&self, properties: &mut BTreeMap<String, String>) {
        self.store_with_prepend(properties, "");
    }
    /// Java package-private `getPid`.
    pub fn get_pid(&self) -> Option<String> {
        self.pid.clone()
    }
    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> Option<ProcessName> {
        self.process_name
    }
    /// Java package-private `getSubProcessName`.
    /// `subProcessName.toString()`: "" when empty.
    pub fn get_sub_process_name(&self) -> Option<String> {
        self.sub_process_name.to_string_option()
    }
    /// Java package-private `getSubDirName`: the `ConstStringProperty`, here its
    /// `toString()` ("" when empty).
    pub fn get_sub_dir_name(&self) -> Option<String> {
        self.sub_dir_name.to_string_option()
    }
    /// Java `getDisplayID`: `displayID.getInt()` (-1, the display value, when
    /// not set).
    pub fn get_display_id(&self) -> i32 {
        self.display_id.get_int()
    }
    /// Java `getFactoryID`: `factoryID.toString()`, "" when empty.
    pub fn get_factory_id(&self) -> Option<String> {
        self.factory_id.to_string_option()
    }
    /// Java `getHostName`: `hostName.toString()`.
    pub fn get_host_name(&self) -> String {
        self.host_name.to_string_option().unwrap_or_default()
    }
    /// Java package-private `getComputerMap`.
    pub fn get_computer_map(&self) -> Option<&BTreeMap<String, String>> {
        self.computer_map.as_ref()
    }
    /// Java package-private `getProcessingMethod`.
    pub fn get_processing_method(&self) -> Option<ProcessingMethod> {
        self.processing_method
    }
    /// Java private `createPrepend`.
    pub fn create_prepend(&self, prepend: &str) -> String {
        if prepend.is_empty() {
            self.process_data_prepend.clone()
        } else {
            format!("{prepend}.{prepend}")
        }
    }
    /// Java private `removeMap`.
    pub fn remove_map(
        &self,
        properties: &mut BTreeMap<String, String>,
        group: &str,
        element_key: &str,
    ) {
        let prefix = format!("{group}{element_key}.");
        properties.retain(|key, _| !key.trim().starts_with(&prefix));
    }
    /// Java package-private `resetLineNumber()`.
    pub fn reset_line_number(&mut self) {
        self.line_number = LINE_NUMBER_DEFAULT;
    }
    /// Java synchronized `resetLineNumber(int)`.
    pub fn reset_chunk_line_number(&mut self, chunk_index: i32) {
        self.build_chunk_map(chunk_index);
        self.chunk_map
            .as_mut()
            .unwrap()
            .get_mut(&chunk_index)
            .unwrap()
            .reset_line_number();
    }
    /// Java package-private `resetNumDone`.
    pub fn reset_num_done(&mut self) {
        self.num_done.reset();
    }
    /// Java package-private `gtLineNumber(int)`.
    pub fn gt_line_number(&self, input: i32) -> bool {
        self.line_number > input
    }
    /// Java synchronized `gtLineNumber(int,int)`.
    pub fn gt_chunk_line_number(&mut self, chunk_index: i32, input: i32) -> bool {
        self.build_chunk_map(chunk_index);
        self.chunk_map.as_ref().unwrap()[&chunk_index].gt_line_number(input)
    }
    /// Java package-private `getLineNumber()`.
    pub fn get_line_number(&self) -> i32 {
        self.line_number
    }
    /// Java synchronized `getLineNumber(int)`.
    pub fn get_chunk_line_number(&mut self, chunk_index: i32) -> i32 {
        self.build_chunk_map(chunk_index);
        self.chunk_map.as_ref().unwrap()[&chunk_index].get_line_number()
    }
    /// Java synchronized `buildChunkMap(int)`.
    pub fn build_chunk_map(&mut self, chunk_index: i32) {
        self.chunk_map
            .get_or_insert_with(BTreeMap::new)
            .entry(chunk_index)
            .or_insert_with(Chunk::new);
    }
    /// Java synchronized `buildChunkMap(int,Chunk)`.
    fn build_chunk_map_with_chunk(&mut self, chunk_index: i32, chunk: Chunk) {
        self.chunk_map
            .get_or_insert_with(BTreeMap::new)
            .insert(chunk_index, chunk);
    }
    /// Java package-private `getNumDone`.
    pub fn get_num_done(&self) -> i32 {
        self.num_done.get_int()
    }
    /// Java package-private `incrementLineNumber()`.
    pub fn increment_line_number(&mut self) {
        self.line_number += 1;
    }
    /// Java synchronized `incrementLineNumber(int)`.
    pub fn increment_chunk_line_number(&mut self, chunk_index: i32) {
        self.build_chunk_map(chunk_index);
        self.chunk_map
            .as_mut()
            .unwrap()
            .get_mut(&chunk_index)
            .unwrap()
            .increment_line_number();
    }
    /// Java package-private `printChunkMap`.
    ///
    /// Upstream bug fixed in translation (ProcessData.java:552-565): `entryFound` is
    /// never set and `entryFound ? "," : "" + entry.getValue()` binds the `+` inside
    /// the conditional, so Java prints the line numbers run together and then
    /// "null" as if the map were empty.  Here the values are separated by "," and
    /// "null" is printed only for an empty map, as the code evidently intends.
    pub fn print_chunk_map(&self) {
        let Some(chunk_map) = &self.chunk_map else {
            eprintln!("chunkMap:null");
            return;
        };
        eprint!("chunkMap:");
        let mut entry_found = false;
        // A `TreeMap<Integer, Chunk>`: ascending keys.
        for chunk in chunk_map.values() {
            eprint!(
                "{}{}",
                if entry_found { "," } else { "" },
                chunk.line_number
            );
            entry_found = true;
        }
        if !entry_found {
            eprintln!("null");
        } else {
            eprintln!();
        }
    }
    /// Java package-private `incrementNumDone`.
    pub fn increment_num_done(&mut self) {
        self.num_done.increment();
    }
    /// Java private `storeMap`.
    pub fn store_map(
        &self,
        properties: &mut BTreeMap<String, String>,
        group: &str,
        map: Option<&BTreeMap<String, String>>,
        key: &str,
    ) {
        if let Some(map) = map {
            for (map_key, value) in map {
                properties.insert(format!("{group}{key}.{map_key}"), value.clone());
            }
        }
    }
    /// Java private `storeChunkMap`.
    pub fn store_chunk_map(&self, properties: &mut BTreeMap<String, String>, group: &str) {
        if let Some(map) = &self.chunk_map {
            for (index, chunk) in map {
                chunk.store(properties, group, *index);
            }
        }
    }
    /// Java `load(Properties)`.
    pub fn load_properties(&mut self, properties: &mut BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }
    /// Java `reset`.
    pub fn reset(&mut self) {
        self.pid = None;
        self.group_pid = None;
        self.start_time = None;
        self.process_name = None;
        self.sub_process_name.reset();
        self.sub_dir_name.reset();
        self.display_id.reset();
        self.factory_id.reset();
        self.os_type = None;
        self.host_name.reset();
        if let Some(map) = &mut self.computer_map {
            map.clear();
        }
        self.processing_method = None;
        self.dialog_type = None;
        self.last_process.reset();
        self.secondary_queue.reset();
        self.line_number = LINE_NUMBER_DEFAULT;
        self.num_done.reset();
        if let Some(map) = &mut self.chunk_map {
            map.clear();
        }
    }
    /// Java private `loadMap`.
    pub fn load_map(
        &self,
        properties: &BTreeMap<String, String>,
        group: &str,
        mut map: Option<BTreeMap<String, String>>,
        key: &str,
        default_string: &str,
    ) -> Option<BTreeMap<String, String>> {
        let prefix = format!("{group}{key}.");
        for (property, value) in properties {
            if property.trim().starts_with(&prefix) {
                // `props.getProperty(enumKey, defaultString)`: the key is present,
                // so its value (the default is never used).
                let _ = default_string;
                map.get_or_insert_with(BTreeMap::new).insert(
                    property[property.find(&prefix).unwrap() + prefix.len()..].to_owned(),
                    value.clone(),
                );
            }
        }
        map
    }
    /// Java private `loadChunkMap`.
    pub fn load_chunk_map(&mut self, properties: &BTreeMap<String, String>, prepend: &str) {
        for key in properties.keys() {
            if let Some(chunk) = Chunk::get_instance(properties, prepend, key.trim()) {
                if let Some(index) = Chunk::get_index(key) {
                    self.build_chunk_map_with_chunk(index, chunk);
                }
            }
        }
    }
}

impl StorableValue for ProcessData {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_properties(properties);
    }
    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        let group = format!("{prepend}.");
        // Remove everything containing COMPUTER_KEY to prevent entries that
        // where removed from computerMap from remaining in props.
        self.remove_map(properties, &group, COMPUTER_KEY);
        Chunk::remove_all(properties, &prepend);
        if self.is_empty() {
            // Remove everything if no process was added to processData.
            properties.remove(&format!("{group}{PID_KEY}"));
            properties.remove(&format!("{group}{GROUP_PID_KEY}"));
            properties.remove(&format!("{group}{START_TIME_KEY}"));
            properties.remove(&format!("{group}{PROCESS_NAME_KEY}"));
            // `ProcessingMethod.remove(props, prepend)`.
            properties.remove(&ProcessingMethod::create_key(Some(&prepend)));
            self.display_id
                .remove_with_prepend(properties, Some(&prepend));
            self.factory_id.remove(Some(properties), Some(&prepend));
            self.sub_process_name
                .remove(Some(properties), Some(&prepend));
            self.sub_dir_name.remove(Some(properties), Some(&prepend));
            self.host_name.remove(Some(properties), Some(&prepend));
            properties.remove(&format!("{group}{}", os_type::KEY));
            DialogType::remove(properties, &prepend);
            self.last_process.remove(Some(properties), Some(&prepend));
            self.secondary_queue
                .remove(Some(properties), Some(&prepend));
            properties.remove(&format!("{group}{LINE_NUMBER_KEY}"));
        } else {
            properties.insert(format!("{group}{PID_KEY}"), self.pid.clone().unwrap());
            properties.insert(
                format!("{group}{GROUP_PID_KEY}"),
                self.group_pid.clone().unwrap(),
            );
            properties.insert(
                format!("{group}{START_TIME_KEY}"),
                self.start_time.as_ref().unwrap().to_string(),
            );
            match self.process_name {
                None => {
                    properties.remove(&format!("{group}{PROCESS_NAME_KEY}"));
                }
                Some(name) => {
                    properties.insert(format!("{group}{PROCESS_NAME_KEY}"), name.to_string());
                }
            }
            ConstEtomoNumber::store_with_prepend(&self.display_id, properties, Some(&prepend));
            self.factory_id
                .store_with_prepend(Some(properties), Some(&prepend));
            self.sub_process_name
                .store_with_prepend(Some(properties), Some(&prepend));
            self.sub_dir_name
                .store_with_prepend(Some(properties), Some(&prepend));
            self.host_name
                .store_with_prepend(Some(properties), Some(&prepend));
            // `processingMethod.store(props, prepend)` / `ProcessingMethod.remove`.
            let processing_key = ProcessingMethod::create_key(Some(&prepend));
            match self.processing_method {
                None => {
                    properties.remove(&processing_key);
                }
                Some(method) => {
                    properties.insert(processing_key, method.to_string());
                }
            }
            match self.os_type {
                None => {
                    properties.remove(&format!("{group}{}", os_type::KEY));
                }
                Some(os_type) => os_type.store(properties, &prepend),
            }
            match self.dialog_type {
                None => DialogType::remove(properties, &prepend),
                Some(dialog) => dialog.store_with_prepend(properties, &prepend),
            }
            self.last_process
                .store_with_prepend(Some(properties), Some(&prepend));
            self.secondary_queue
                .store_with_prepend(Some(properties), Some(&prepend));
            properties.insert(
                format!("{group}{LINE_NUMBER_KEY}"),
                self.line_number.to_string(),
            );
            ConstEtomoNumber::store_with_prepend(&self.num_done, properties, Some(&prepend));
            // Store everything in computerMap in props.
            self.store_map(properties, &group, self.computer_map.as_ref(), COMPUTER_KEY);
            self.store_chunk_map(properties, &prepend);
        }
    }
    fn load(&mut self, properties: &mut BTreeMap<String, String>) {
        self.load_properties(properties);
    }
    fn load_with_prepend(&mut self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        assert!(
            !self.do_not_load,
            "Trying to load into into thread data that belongs to a managed process."
        );
        self.reset();
        let prepend = self.create_prepend(prepend);
        let group = format!("{prepend}.");
        self.pid = properties.get(&format!("{group}{PID_KEY}")).cloned();
        self.group_pid = properties.get(&format!("{group}{GROUP_PID_KEY}")).cloned();
        self.start_time = properties
            .get(&format!("{group}{START_TIME_KEY}"))
            .map(|start_time| Time::new(start_time));
        self.process_name = properties
            .get(&format!("{group}{PROCESS_NAME_KEY}"))
            .and_then(|s| ProcessName::get_instance_with_axis(s, self.axis_id));
        // The `StringProperty` loads take a mutable `Properties` (they drop a
        // backward-compatible key, which none of these has).
        let mut props = properties.clone();
        self.sub_process_name
            .load_with_prepend(Some(&mut props), Some(&prepend));
        self.sub_dir_name
            .load_with_prepend(Some(&mut props), Some(&prepend));
        self.display_id
            .load_with_prepend(properties, Some(&prepend));
        self.factory_id
            .load_with_prepend(Some(&mut props), Some(&prepend));
        self.host_name
            .load_with_prepend(Some(&mut props), Some(&prepend));
        self.processing_method = ProcessingMethod::get_instance(
            properties
                .get(&ProcessingMethod::create_key(Some(&prepend)))
                .map(String::as_str),
        );
        self.os_type = Some(OSType::get_instance_from_props(properties, &prepend));
        self.dialog_type = DialogType::load(properties, &prepend);
        self.last_process
            .load_with_prepend(Some(&mut props), Some(&prepend));
        self.secondary_queue
            .load_with_prepend(Some(&mut props), Some(&prepend));
        let mut et_line_number = EtomoNumber::new_with_name(LINE_NUMBER_KEY);
        et_line_number.load_with_prepend(properties, Some(&prepend));
        self.num_done.load_with_prepend(properties, Some(&prepend));
        if et_line_number.is_null() {
            self.line_number = LINE_NUMBER_DEFAULT;
        } else {
            self.line_number = et_line_number.get_int();
        }
        let computer_map = self.computer_map.take();
        self.computer_map = self.load_map(properties, &group, computer_map, COMPUTER_KEY, "0");
        self.load_chunk_map(properties, &prepend);
    }
}

/// Java private static final nested `Chunk`.
#[derive(Debug)]
struct Chunk {
    line_number: i32,
}
impl Chunk {
    const CHUNK_KEY: &'static str = "Chunk";
    fn new() -> Self {
        Self {
            line_number: LINE_NUMBER_DEFAULT,
        }
    }
    fn remove_all(properties: &mut BTreeMap<String, String>, prepend: &str) {
        let group = Self::build_group(prepend);
        properties.retain(|key, _| !key.trim().starts_with(&group));
    }
    fn get_instance(
        properties: &BTreeMap<String, String>,
        prepend: &str,
        key: &str,
    ) -> Option<Self> {
        if key.trim().is_empty()
            || !key.starts_with(&Self::build_group(prepend))
            || !key.ends_with(LINE_NUMBER_KEY)
        {
            return None;
        }
        properties
            .get(key)
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|v| *v >= 0)
            .map(|line_number| Self { line_number })
    }
    fn get_index(key: &str) -> Option<i32> {
        let mut parts = key.split('.').rev();
        parts.next()?;
        parts.next()?.parse().ok()
    }
    fn build_group(prepend: &str) -> String {
        format!(
            "{}{}{}.",
            prepend,
            if prepend.ends_with('.') { "" } else { "." },
            Self::CHUNK_KEY
        )
    }
    fn build_key(prepend: &str, index: i32) -> String {
        format!("{}{}.", Self::build_group(prepend), index)
    }
    fn store(&self, properties: &mut BTreeMap<String, String>, prepend: &str, index: i32) {
        properties.insert(
            format!("{}{}", Self::build_key(prepend, index), LINE_NUMBER_KEY),
            self.line_number.to_string(),
        );
    }
    fn get_line_number(&self) -> i32 {
        self.line_number
    }
    fn reset_line_number(&mut self) {
        self.line_number = LINE_NUMBER_DEFAULT
    }
    fn increment_line_number(&mut self) {
        self.line_number += 1
    }
    fn gt_line_number(&self, input: i32) -> bool {
        self.line_number > input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn storable_round_trip_preserves_chunk_and_computer_data() {
        let mut data = ProcessData::new(Some(AxisID::First), None);
        data.pid = Some("10".into());
        data.group_pid = Some("10".into());
        data.start_time = Some(Time::new("10:11:12"));
        data.set_sub_dir_name(Some("chunks"));
        data.set_processing_method(Some(ProcessingMethod::PpGpu));
        data.set_computer_map(Some(BTreeMap::from([("host".into(), "4".into())])));
        data.increment_chunk_line_number(3);
        data.increment_chunk_line_number(3);
        let mut props = BTreeMap::new();
        data.store(&mut props);
        let mut loaded = ProcessData::new(Some(AxisID::First), None);
        loaded.load(&mut props);
        assert_eq!(loaded.get_sub_dir_name().as_deref(), Some("chunks"));
        assert_eq!(loaded.get_chunk_line_number(3), 2);
        assert_eq!(loaded.get_computer_map().unwrap()["host"], "4");
        assert_eq!(
            loaded.get_processing_method(),
            Some(ProcessingMethod::PpGpu)
        );
    }
}
