//! `IMOD/Etomo/src/etomo/process/ProcessData.java`.
//!
//! Persisted process identity for a process which survived an eTomo restart.
//! The property vocabulary and chunk bookkeeping are represented directly.  The Java
//! `PsParam`, `Network`, `OSType`, `Time`, `ProcessingMethod`, and `BaseManager`
//! units have not all crossed this dependency frontier; their operations remain explicit
//! null-equivalent boundaries rather than simulated process-state answers.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use std::collections::BTreeMap;
use std::convert::Infallible;

const PID_KEY: &str = "PID";
const GROUP_PID_KEY: &str = "GroupPID";
const START_TIME_KEY: &str = "StartTime";
const PROCESS_NAME_KEY: &str = "ProcessName";
const OS_TYPE_KEY: &str = "OS";
const COMPUTER_KEY: &str = "Computer";
const LINE_NUMBER_KEY: &str = "LineNumber";
const LINE_NUMBER_DEFAULT: i32 = 0;

/// One local `ps` row consumed by Java `PsParam` in `runPs`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PsRecord {
    pub pid: String,
    pub group_pid: String,
    pub start_time: String,
}

/// Java final `ProcessData implements Storable`.
pub struct ProcessData {
    display_id: i32,
    factory_id: Option<String>,
    sub_process_name: Option<String>,
    sub_dir_name: Option<String>,
    host_name: Option<String>,
    last_process: Option<String>,
    secondary_queue: Option<String>,
    axis_id: AxisID,
    process_data_prepend: String,
    /// Java final `manager`.  Managers are process-lifetime objects in both
    /// implementations, hence the shared static reference convention.
    manager: Option<&'static dyn BaseManager>,
    pid: Option<String>,
    group_pid: Option<String>,
    /// Java `Time`; its serialized source representation is retained verbatim.
    start_time: Option<String>,
    process_name: Option<ProcessName>,
    do_not_load: bool,
    /// Java `OSType`; its serialized value is retained verbatim.
    os_type: Option<String>,
    ssh_failed: bool,
    computer_map: Option<BTreeMap<String, String>>,
    /// Java `ProcessingMethod`.
    processing_method: Option<ProcessingMethod>,
    dialog_type: Option<DialogType>,
    /// Java `DebugLevel`.
    debug: Option<DebugLevel>,
    line_number: i32,
    num_done: i32,
    chunk_map: Option<BTreeMap<i32, Chunk>>,
}

impl ProcessData {
    /// Java package-private `ProcessData(AxisID, BaseManager)`.
    pub fn new(axis_id: Option<AxisID>, manager: Option<&'static dyn BaseManager>) -> Self {
        let axis_id = match axis_id {
            Some(AxisID::Only) | None => AxisID::First,
            Some(axis) => axis,
        };
        Self {
            display_id: -1,
            factory_id: None,
            sub_process_name: None,
            sub_dir_name: None,
            host_name: None,
            last_process: None,
            secondary_queue: None,
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
            debug: None,
            line_number: LINE_NUMBER_DEFAULT,
            num_done: 0,
            chunk_map: None,
        }
    }

    /// Java static `getManagedInstance`.  Host/OS discovery has unavailable Network and
    /// OSType dependencies, but its managed/load invariant and fixed process name hold.
    pub fn get_managed_instance(
        axis_id: Option<AxisID>,
        manager: Option<&'static dyn BaseManager>,
        process_name: ProcessName,
    ) -> Self {
        let mut data = Self::new(axis_id, manager);
        data.process_name = Some(process_name);
        data.do_not_load = true;
        data
    }
    /// Java package-private `dumpState`.
    pub fn dump_state(&self) {
        eprintln!(
            "[processDataPrepend:{},pid:{:?},\ngroupPid:{:?},doNotLoad:{},sshFailed:{},computerMap:{:?}]",
            self.process_data_prepend,
            self.pid,
            self.group_pid,
            self.do_not_load,
            self.ssh_failed,
            self.computer_map
        );
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
            self.sub_process_name,
            self.sub_dir_name,
            self.host_name,
            self.os_type,
            self.display_id,
            self.factory_id,
            self.dialog_type,
            self.last_process,
            self.processing_method
        )
    }
    /// Java package-private `setDisplayKey`.
    pub fn set_display_key(&mut self, process_result_display: Option<Infallible>) {
        let _ = process_result_display;
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
    pub fn set_last_process(&mut self, process_series: Option<Infallible>, resumable: bool) {
        let _ = (process_series, resumable);
    }
    /// Java `getLastProcess`.
    pub fn get_last_process(&self) -> Option<String> {
        self.last_process.clone().filter(|s| !s.is_empty())
    }
    /// Java `getSecondaryQueue`.
    pub fn get_secondary_queue(&self) -> Option<String> {
        self.secondary_queue.clone().filter(|s| !s.is_empty())
    }
    /// Java package-private `setSubProcessName`.
    pub fn set_sub_process_name(&mut self, input: Option<&str>) {
        self.sub_process_name = input.filter(|s| !s.trim().is_empty()).map(str::to_owned);
    }
    /// Java package-private `setSubDirName`.
    pub fn set_sub_dir_name(&mut self, input: Option<&str>) {
        self.sub_dir_name = input.filter(|s| !s.trim().is_empty()).map(str::to_owned);
    }
    /// Java `isEmpty`.
    pub fn is_empty(&self) -> bool {
        self.pid.is_none() || self.group_pid.is_none() || self.start_time.is_none()
    }
    /// Java `isRunning`; local process records use the OS PID probe while
    /// remote-host records remain unavailable until the SSH process runner lands.
    pub fn is_running(&self) -> bool {
        if self.is_empty() || self.is_on_different_host() {
            return false;
        }
        let Some(pid) = self.pid.as_deref().and_then(|pid| pid.parse::<i32>().ok()) else {
            return false;
        };
        #[cfg(unix)]
        unsafe {
            // `kill(pid, 0)` is the POSIX query used by the Java PsParam path:
            // EPERM still means the process exists.
            libc::kill(pid, 0) == 0
                || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
        }
        #[cfg(not(unix))]
        {
            false
        }
    }
    /// Java `isOnDifferentHost`.  `Network` compares the persisted host with
    /// the local machine before attempting a PID query.  Keep `localhost`
    /// local for records created by the Rust runner, then use gethostname on
    /// POSIX; a loaded nonempty name that cannot match is necessarily remote.
    pub fn is_on_different_host(&self) -> bool {
        let Some(host_name) = self
            .host_name
            .as_deref()
            .filter(|name| !name.trim().is_empty())
        else {
            return false;
        };
        if host_name.eq_ignore_ascii_case("localhost")
            || host_name == "127.0.0.1"
            || host_name == "::1"
        {
            return false;
        }
        #[cfg(unix)]
        {
            let mut buffer = [0i8; 256];
            let local = unsafe {
                if libc::gethostname(buffer.as_mut_ptr(), buffer.len()) != 0 {
                    return true;
                }
                std::ffi::CStr::from_ptr(buffer.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            // Hostname services commonly return the short host name while a
            // persisted record may contain its FQDN; Java Network treats these
            // as the same local computer.
            let local_short = local.split('.').next().unwrap_or(&local);
            let record_short = host_name.split('.').next().unwrap_or(host_name);
            !host_name.eq_ignore_ascii_case(&local)
                && !record_short.eq_ignore_ascii_case(local_short)
        }
        #[cfg(not(unix))]
        {
            true
        }
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
        self.secondary_queue = secondary_queue
            .filter(|s| !s.trim().is_empty())
            .map(str::to_owned);
    }
    /// Java `setProcessingMethod`.
    pub fn set_processing_method(&mut self, processing_method: Option<ProcessingMethod>) {
        self.processing_method = processing_method;
    }
    /// Java package-private `setPid`, retaining a local PID/provenance record.
    pub fn set_pid(&mut self, pid: Option<&str>) {
        self.pid = None;
        self.group_pid = None;
        self.start_time = None;
        if let Some(pid) = pid.filter(|pid| !pid.trim().is_empty()) {
            if let Some(record) = self.run_ps(Some(pid)) {
                self.pid = Some(record.pid);
                self.group_pid = Some(record.group_pid);
                self.start_time = Some(record.start_time);
            }
        }
    }
    /// Register a child launched by the typed Rust `SystemProgram` path.
    pub fn set_local_process(&mut self, pid: u32, process_name: Option<ProcessName>) {
        self.set_pid(Some(&pid.to_string()));
        self.process_name = process_name;
        self.host_name = Some("localhost".to_owned());
        self.os_type = Some(std::env::consts::OS.to_owned());
    }
    /// Java private `runPs`.  The translated local process runner uses the
    /// same PID query instead of manufacturing process-group/start metadata.
    /// Remote records remain a deliberate SSH-runner frontier.
    pub fn run_ps(&mut self, pid: Option<&str>) -> Option<PsRecord> {
        let pid = pid?.trim();
        if pid.is_empty() || self.is_on_different_host() {
            self.ssh_failed = self.is_on_different_host();
            return None;
        }
        #[cfg(unix)]
        {
            if self.debug.is_some_and(DebugLevel::is_verbose) {
                eprintln!("ProcessData.runPs");
            }
            let output = std::process::Command::new("ps")
                .args(["-o", "pid=", "-o", "pgid=", "-o", "lstart=", "-p", pid])
                .output();
            let Ok(output) = output else {
                self.ssh_failed = true;
                return None;
            };
            let stdout = String::from_utf8_lossy(&output.stdout);
            self.ssh_failed = stdout.trim().is_empty();
            let mut fields = stdout.split_whitespace();
            let actual_pid = fields.next()?;
            let group_pid = fields.next()?;
            let start_time = fields.collect::<Vec<_>>().join(" ");
            if actual_pid != pid || start_time.is_empty() {
                return None;
            }
            Some(PsRecord {
                pid: actual_pid.to_owned(),
                group_pid: group_pid.to_owned(),
                start_time,
            })
        }
        #[cfg(not(unix))]
        {
            let _ = pid;
            self.ssh_failed = true;
            None
        }
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
    pub fn get_sub_process_name(&self) -> Option<String> {
        self.sub_process_name.clone()
    }
    /// Java package-private `getSubDirName` (ConstStringProperty represented by value).
    pub fn get_sub_dir_name(&self) -> Option<String> {
        self.sub_dir_name.clone()
    }
    /// Java `getDisplayID`.
    pub fn get_display_id(&self) -> i32 {
        self.display_id
    }
    /// Java `getFactoryID`.
    pub fn get_factory_id(&self) -> Option<String> {
        self.factory_id.clone()
    }
    /// Java `getHostName`.
    pub fn get_host_name(&self) -> String {
        self.host_name.clone().unwrap_or_default()
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
        self.num_done = 0;
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
        self.num_done
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
    pub fn print_chunk_map(&self) {
        eprintln!("chunkMap:{:?}", self.chunk_map);
    }
    /// Java package-private `incrementNumDone`.
    pub fn increment_num_done(&mut self) {
        self.num_done += 1;
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
    pub fn load_properties(&mut self, properties: &BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }
    /// Java `reset`.
    pub fn reset(&mut self) {
        self.pid = None;
        self.group_pid = None;
        self.start_time = None;
        self.process_name = None;
        self.sub_process_name = None;
        self.sub_dir_name = None;
        self.display_id = -1;
        self.factory_id = None;
        self.os_type = None;
        self.host_name = None;
        if let Some(map) = &mut self.computer_map {
            map.clear();
        }
        self.processing_method = None;
        self.dialog_type = None;
        self.last_process = None;
        self.secondary_queue = None;
        self.line_number = LINE_NUMBER_DEFAULT;
        self.num_done = 0;
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
                map.get_or_insert_with(BTreeMap::new).insert(
                    property[property.find(&prefix).unwrap() + prefix.len()..].to_owned(),
                    if value.is_empty() {
                        default_string.to_owned()
                    } else {
                        value.clone()
                    },
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

impl Storable for ProcessData {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_properties(properties);
    }
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        let group = format!("{prepend}.");
        self.remove_map(properties, &group, COMPUTER_KEY);
        Chunk::remove_all(properties, &prepend);
        if self.is_empty() {
            for key in [
                PID_KEY,
                GROUP_PID_KEY,
                START_TIME_KEY,
                PROCESS_NAME_KEY,
                OS_TYPE_KEY,
                LINE_NUMBER_KEY,
            ] {
                properties.remove(&format!("{group}{key}"));
            }
            for key in [
                "DisplayID",
                "FactoryID",
                "SubProcessName",
                "SubDirName",
                "HostName",
                "LastProcess",
                "SecondaryQueue",
                "NumDone",
                "DialogType",
                "ProcessingMethod",
            ] {
                properties.remove(&format!("{prepend}.{key}"));
            }
            return;
        }
        properties.insert(format!("{group}{PID_KEY}"), self.pid.clone().unwrap());
        properties.insert(
            format!("{group}{GROUP_PID_KEY}"),
            self.group_pid.clone().unwrap(),
        );
        properties.insert(
            format!("{group}{START_TIME_KEY}"),
            self.start_time.clone().unwrap(),
        );
        if let Some(name) = self.process_name {
            properties.insert(format!("{group}{PROCESS_NAME_KEY}"), name.to_string());
        } else {
            properties.remove(&format!("{group}{PROCESS_NAME_KEY}"));
        }
        properties.insert(format!("{prepend}.DisplayID"), self.display_id.to_string());
        for (key, value) in [
            ("FactoryID", &self.factory_id),
            ("SubProcessName", &self.sub_process_name),
            ("SubDirName", &self.sub_dir_name),
            ("HostName", &self.host_name),
            ("LastProcess", &self.last_process),
            ("SecondaryQueue", &self.secondary_queue),
            ("OS", &self.os_type),
        ] {
            if let Some(value) = value {
                properties.insert(format!("{prepend}.{key}"), value.clone());
            } else {
                properties.remove(&format!("{prepend}.{key}"));
            }
        }
        if let Some(dialog) = self.dialog_type {
            dialog.store_with_prepend(properties, &prepend);
        } else {
            properties.remove(&format!("{prepend}.DialogType"));
        }
        let processing_key = ProcessingMethod::create_key(Some(&prepend));
        if let Some(method) = self.processing_method {
            properties.insert(processing_key, method.to_string());
        } else {
            properties.remove(&processing_key);
        }
        properties.insert(
            format!("{group}{LINE_NUMBER_KEY}"),
            self.line_number.to_string(),
        );
        properties.insert(format!("{prepend}.NumDone"), self.num_done.to_string());
        self.store_map(properties, &group, self.computer_map.as_ref(), COMPUTER_KEY);
        self.store_chunk_map(properties, &prepend);
    }
    fn load(&mut self, properties: &BTreeMap<String, String>) {
        self.load_properties(properties);
    }
    fn load_with_prepend(&mut self, properties: &BTreeMap<String, String>, prepend: &str) {
        assert!(
            !self.do_not_load,
            "Trying to load into into thread data that belongs to a managed process."
        );
        self.reset();
        let prepend = self.create_prepend(prepend);
        let group = format!("{prepend}.");
        self.pid = properties.get(&format!("{group}{PID_KEY}")).cloned();
        self.group_pid = properties.get(&format!("{group}{GROUP_PID_KEY}")).cloned();
        self.start_time = properties.get(&format!("{group}{START_TIME_KEY}")).cloned();
        self.process_name = properties
            .get(&format!("{group}{PROCESS_NAME_KEY}"))
            .and_then(|s| ProcessName::get_instance_with_axis(s, self.axis_id));
        self.display_id = properties
            .get(&format!("{prepend}.DisplayID"))
            .and_then(|value| value.parse().ok())
            .unwrap_or(-1);
        self.factory_id = properties.get(&format!("{prepend}.FactoryID")).cloned();
        self.sub_process_name = properties
            .get(&format!("{prepend}.SubProcessName"))
            .cloned();
        self.sub_dir_name = properties.get(&format!("{prepend}.SubDirName")).cloned();
        self.host_name = properties.get(&format!("{prepend}.HostName")).cloned();
        self.last_process = properties.get(&format!("{prepend}.LastProcess")).cloned();
        self.secondary_queue = properties
            .get(&format!("{prepend}.SecondaryQueue"))
            .cloned();
        self.os_type = properties.get(&format!("{prepend}.OS")).cloned();
        self.dialog_type = DialogType::load(properties, &prepend);
        self.processing_method = ProcessingMethod::get_instance(
            properties
                .get(&ProcessingMethod::create_key(Some(&prepend)))
                .map(String::as_str),
        );
        self.line_number = properties
            .get(&format!("{prepend}.{LINE_NUMBER_KEY}"))
            .and_then(|value| value.parse().ok())
            .unwrap_or(LINE_NUMBER_DEFAULT);
        self.num_done = properties
            .get(&format!("{prepend}.NumDone"))
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
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
        data.start_time = Some("now".into());
        data.set_sub_dir_name(Some("chunks"));
        data.set_processing_method(Some(ProcessingMethod::PpGpu));
        data.set_computer_map(Some(BTreeMap::from([("host".into(), "4".into())])));
        data.increment_chunk_line_number(3);
        data.increment_chunk_line_number(3);
        let mut props = BTreeMap::new();
        data.store(&mut props);
        let mut loaded = ProcessData::new(Some(AxisID::First), None);
        loaded.load(&props);
        assert_eq!(loaded.get_sub_dir_name().as_deref(), Some("chunks"));
        assert_eq!(loaded.get_chunk_line_number(3), 2);
        assert_eq!(loaded.get_computer_map().unwrap()["host"], "4");
        assert_eq!(
            loaded.get_processing_method(),
            Some(ProcessingMethod::PpGpu)
        );
    }

    #[test]
    fn persisted_host_distinguishes_local_and_remote_process_records() {
        let mut data = ProcessData::new(Some(AxisID::First), None);
        data.host_name = Some("localhost".into());
        assert!(!data.is_on_different_host());
        data.host_name = Some("remote.example.invalid".into());
        assert!(data.is_on_different_host());
    }

    #[cfg(unix)]
    #[test]
    fn set_pid_uses_real_process_group_and_start_metadata() {
        let mut data = ProcessData::new(Some(AxisID::First), None);
        let pid = std::process::id().to_string();
        data.set_pid(Some(&pid));
        assert_eq!(data.get_pid().as_deref(), Some(pid.as_str()));
        let mut properties = BTreeMap::new();
        data.store_properties(&mut properties);
        let group_pid = properties
            .iter()
            .find(|(key, _)| key.ends_with(".GroupPID"))
            .map(|(_, value)| value.as_str());
        let start_time = properties
            .iter()
            .find(|(key, _)| key.ends_with(".StartTime"))
            .map(|(_, value)| value.as_str());
        assert!(group_pid.is_some_and(|value| !value.is_empty()));
        assert!(start_time.is_some_and(|value| value.split_whitespace().count() >= 5));
        assert!(!data.is_ssh_failed());
    }
}
