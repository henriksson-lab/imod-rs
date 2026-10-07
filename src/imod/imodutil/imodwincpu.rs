//! Translation of `IMOD/imodutil/imodwincpu.cpp`, with the two headers it
//! includes, `IMOD/imodutil/CpuUsage.h` (`CCpuUsage`) and
//! `IMOD/imodutil/PerfCounters.h` (`CPerfCounters<LONGLONG>`).
//!
//! The program prints the system-wide CPU usage over an interval, read from
//! the performance counters behind `HKEY_PERFORMANCE_DATA`.  It is
//! Windows-only in the source (it includes `<windows.h>` and ATL), and is what
//! processchunks' machine probe and eTomo's load monitor run there in place of
//! `w`.  The module is compiled on Windows only (`imodutil/mod.rs`).
//!
//! The performance data block is a byte buffer of variable-length records;
//! the source walks it with pointer arithmetic over `PERF_*` structures.  Here
//! a pointer into the block is a byte offset into the buffer, and each
//! structure's fields are read at their `winperf.h` offsets (64-bit layout:
//! `CounterOffset` at 36 of `PERF_COUNTER_DEFINITION`, `NumCounters` and
//! `NumInstances` at 32 and 40 of `PERF_OBJECT_TYPE`).
//! A read past the end of the data -- which the source would make through a
//! stray pointer when a counter is missing -- yields the "not found" value
//! instead (`BUGS.md`, fixed in translation).

use core::ffi::{c_char, c_void};

const SYSTEM_OBJECT_INDEX: u32 = 2; // 'System' object
const PROCESS_OBJECT_INDEX: u32 = 230; // 'Process' object
const PROCESSOR_OBJECT_INDEX: u32 = 238; // 'Processor' object
const TOTAL_PROCESSOR_TIME_COUNTER_INDEX: u32 = 240; // '% Total processor time' counter (valid in WinNT under 'System' object)
const PROCESSOR_TIME_COUNTER_INDEX: u32 = 6; // '% processor time' counter (for Win2K/XP)

/// `PerfCounters.h:6-7`.
const TOTALBYTES: u32 = 100 * 1024;
const BYTEINCREMENT: u32 = 10 * 1024;

/// `PERF_NO_INSTANCES` (`winperf.h`).
const PERF_NO_INSTANCES: i32 = -1;

type Hkey = *mut c_void;
const HKEY_LOCAL_MACHINE: Hkey = 0x8000_0002_usize as Hkey;
const HKEY_PERFORMANCE_DATA: Hkey = 0x8000_0004_usize as Hkey;
const ERROR_SUCCESS: i32 = 0;
const ERROR_MORE_DATA: i32 = 234;
const KEY_READ: u32 = 0x0002_0019;
const KEY_WRITE: u32 = 0x0002_0006;
const REG_DWORD: u32 = 4;
const VER_PLATFORM_WIN32_WINDOWS: u32 = 1;
const VER_PLATFORM_WIN32_NT: u32 = 2;

#[repr(C)]
struct OsVersionInfoA {
    dw_os_version_info_size: u32,
    dw_major_version: u32,
    dw_minor_version: u32,
    dw_build_number: u32,
    dw_platform_id: u32,
    sz_csd_version: [c_char; 128],
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetVersionExA(info: *mut OsVersionInfoA) -> i32;
    fn Sleep(milliseconds: u32);
}

#[link(name = "advapi32")]
unsafe extern "system" {
    fn RegOpenKeyExA(
        key: Hkey,
        sub_key: *const c_char,
        options: u32,
        sam: u32,
        result: *mut Hkey,
    ) -> i32;
    fn RegSetValueExA(
        key: Hkey,
        value_name: *const c_char,
        reserved: u32,
        value_type: u32,
        data: *const u8,
        size: u32,
    ) -> i32;
    fn RegQueryValueExA(
        key: Hkey,
        value_name: *const c_char,
        reserved: *mut u32,
        value_type: *mut u32,
        data: *mut u8,
        size: *mut u32,
    ) -> i32;
    fn RegCloseKey(key: Hkey) -> i32;
}

/// `main` (`imodwincpu.cpp:22`).
pub fn imodwincpu(argv: &[String]) -> i32 {
    let mut usage_a = CCpuUsage::new();
    let mut interval: i32 = 1000;
    if argv.len() > 1 {
        // `atof`: the longest leading decimal number, 0 when there is none
        let text = argv[1].trim_start();
        let mut end = 0;
        for index in (0..=text.len()).rev() {
            if text.is_char_boundary(index) && text[..index].parse::<f64>().is_ok() {
                end = index;
                break;
            }
        }
        let value = text[..end].parse::<f64>().unwrap_or(0.);
        interval = (1000. * value) as i32;
    }
    usage_a.get_cpu_usage();
    // SAFETY: `Sleep` takes a plain count.
    unsafe { Sleep(interval as u32) };
    let system_wide_cpu_usage = usage_a.get_cpu_usage();
    let text = format!("Percent CPU usage = {system_wide_cpu_usage}\n");
    let _ = std::io::Write::write_all(
        &mut crate::imod::libcfshr::b3dutil::ImodFile::Stdout,
        text.as_bytes(),
    );
    0
}

/// `PLATFORM` (`imodwincpu.cpp:65`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Platform {
    Winnt,
    Win2kXp,
    Win9x,
    Unknown,
}

/// `GetPlatform` (`imodwincpu.cpp:70`).
fn get_platform() -> Platform {
    let mut osvi = OsVersionInfoA {
        dw_os_version_info_size: core::mem::size_of::<OsVersionInfoA>() as u32,
        dw_major_version: 0,
        dw_minor_version: 0,
        dw_build_number: 0,
        dw_platform_id: 0,
        sz_csd_version: [0; 128],
    };
    // SAFETY: a correctly sized `OSVERSIONINFOA` with its size set.
    if unsafe { GetVersionExA(&mut osvi) } == 0 {
        return Platform::Unknown;
    }
    match osvi.dw_platform_id {
        VER_PLATFORM_WIN32_WINDOWS => Platform::Win9x,
        VER_PLATFORM_WIN32_NT => {
            if osvi.dw_major_version == 4 {
                Platform::Winnt
            } else {
                Platform::Win2kXp
            }
        }
        _ => Platform::Unknown,
    }
}

/// `static PLATFORM Platform = GetPlatform();`, shared by the three
/// `GetCpuUsage` overloads (each has its own static with the same value).
fn platform() -> Platform {
    static PLATFORM: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *PLATFORM.get_or_init(|| get_platform() as u8) {
        0 => Platform::Winnt,
        1 => Platform::Win2kXp,
        2 => Platform::Win9x,
        _ => Platform::Unknown,
    }
}

/// `class CCpuUsage` (`CpuUsage.h:6`).
pub struct CCpuUsage {
    m_b_first_time: bool,
    m_ln_old_value: i64,
    m_old_perf_time_100n_sec: i64,
}

impl CCpuUsage {
    /// The `CCpuUsage` constructor, `CCpuUsage::CCpuUsage`
    /// (`imodwincpu.cpp:90`).
    pub fn new() -> Self {
        CCpuUsage {
            m_b_first_time: true,
            m_ln_old_value: 0,
            m_old_perf_time_100n_sec: 0,
        }
    }

    /// `CCpuUsage::EnablePerformaceCounters` (`imodwincpu.cpp:101`), with the
    /// header's default `bEnable = TRUE`.  `CRegKey::Open` asks for read and
    /// write access, and `SetValue(DWORD, name)` writes a `REG_DWORD`; its
    /// result is not checked, as in the source.
    pub fn enable_performace_counters(&mut self, b_enable: bool) -> bool {
        if get_platform() != Platform::Win2kXp {
            return true;
        }
        for path in [
            c"SYSTEM\\CurrentControlSet\\Services\\PerfOS\\Performance",
            c"SYSTEM\\CurrentControlSet\\Services\\PerfProc\\Performance",
        ] {
            let mut reg_key: Hkey = core::ptr::null_mut();
            // SAFETY: the key is opened, written and closed here.
            unsafe {
                if RegOpenKeyExA(
                    HKEY_LOCAL_MACHINE,
                    path.as_ptr(),
                    0,
                    KEY_READ | KEY_WRITE,
                    &mut reg_key,
                ) != ERROR_SUCCESS
                {
                    return false;
                }
                let value: u32 = if b_enable { 0 } else { 1 };
                RegSetValueExA(
                    reg_key,
                    c"Disable Performance Counters".as_ptr(),
                    0,
                    REG_DWORD,
                    (&value as *const u32).cast(),
                    4,
                );
                RegCloseKey(reg_key);
            }
        }
        true
    }

    /// `CCpuUsage::GetCpuUsage()` (`imodwincpu.cpp:128`): the system-wide CPU
    /// usage; the first call returns 0 and keeps the sample.
    pub fn get_cpu_usage(&mut self) -> i32 {
        let platform = platform();

        if self.m_b_first_time {
            self.enable_performace_counters(true);
        }

        // Cpu usage counter is 8 byte length.
        let mut perf_counters = CPerfCounters;
        let mut sz_instance = String::new();

        let dw_object_index;
        let dw_cpu_usage_index;
        match platform {
            Platform::Winnt => {
                dw_object_index = SYSTEM_OBJECT_INDEX;
                dw_cpu_usage_index = TOTAL_PROCESSOR_TIME_COUNTER_INDEX;
            }
            Platform::Win2kXp => {
                dw_object_index = PROCESSOR_OBJECT_INDEX;
                dw_cpu_usage_index = PROCESSOR_TIME_COUNTER_INDEX;
                sz_instance = "_Total".to_owned();
            }
            _ => return -1,
        }

        let cpu_usage: i32;
        let ln_new_value = perf_counters.get_counter_value(
            dw_object_index,
            dw_cpu_usage_index,
            Some(&sz_instance),
        );
        let new_perf_time_100n_sec = perf_time_100n_sec();

        if self.m_b_first_time {
            self.m_b_first_time = false;
            self.m_ln_old_value = ln_new_value;
            self.m_old_perf_time_100n_sec = new_perf_time_100n_sec;
            return 0;
        }

        let ln_value_delta = ln_new_value.wrapping_sub(self.m_ln_old_value);
        let delta_perf_time_100n_sec =
            new_perf_time_100n_sec as f64 - self.m_old_perf_time_100n_sec as f64;

        self.m_ln_old_value = ln_new_value;
        self.m_old_perf_time_100n_sec = new_perf_time_100n_sec;

        let a = ln_value_delta as f64 / delta_perf_time_100n_sec;

        let f = (1.0 - a) * 100.0;
        cpu_usage = (f + 0.5) as i32; // rounding the result
        if cpu_usage < 0 {
            return 0;
        }
        cpu_usage
    }

    /// `CCpuUsage::GetCpuUsage(LPCTSTR pProcessName)` (`imodwincpu.cpp:194`):
    /// the CPU usage of the named process instance.
    pub fn get_cpu_usage_process_name(&mut self, p_process_name: &str) -> i32 {
        let _platform = platform();

        if self.m_b_first_time {
            self.enable_performace_counters(true);
        }

        // Cpu usage counter is 8 byte length.
        let mut perf_counters = CPerfCounters;

        let dw_object_index = PROCESS_OBJECT_INDEX;
        let dw_cpu_usage_index = PROCESSOR_TIME_COUNTER_INDEX;
        let sz_instance = p_process_name.to_owned();

        let cpu_usage: i32;
        let ln_new_value = perf_counters.get_counter_value(
            dw_object_index,
            dw_cpu_usage_index,
            Some(&sz_instance),
        );
        let new_perf_time_100n_sec = perf_time_100n_sec();

        if self.m_b_first_time {
            self.m_b_first_time = false;
            self.m_ln_old_value = ln_new_value;
            self.m_old_perf_time_100n_sec = new_perf_time_100n_sec;
            return 0;
        }

        let ln_value_delta = ln_new_value.wrapping_sub(self.m_ln_old_value);
        let delta_perf_time_100n_sec =
            new_perf_time_100n_sec as f64 - self.m_old_perf_time_100n_sec as f64;

        self.m_ln_old_value = ln_new_value;
        self.m_old_perf_time_100n_sec = new_perf_time_100n_sec;

        let a = ln_value_delta as f64 / delta_perf_time_100n_sec;

        cpu_usage = (a * 100.) as i32;
        if cpu_usage < 0 {
            return 0;
        }
        cpu_usage
    }

    /// `CCpuUsage::GetCpuUsage(DWORD dwProcessID)` (`imodwincpu.cpp:244`):
    /// the CPU usage of the process with that ID.
    pub fn get_cpu_usage_process_id(&mut self, dw_process_id: u32) -> i32 {
        let _platform = platform();

        if self.m_b_first_time {
            self.enable_performace_counters(true);
        }

        // Cpu usage counter is 8 byte length.
        let mut perf_counters = CPerfCounters;

        let dw_object_index = PROCESS_OBJECT_INDEX;
        let dw_cpu_usage_index = PROCESSOR_TIME_COUNTER_INDEX;

        let cpu_usage: i32;
        let ln_new_value = perf_counters.get_counter_value_for_process_id(
            dw_object_index,
            dw_cpu_usage_index,
            dw_process_id,
        );
        let new_perf_time_100n_sec = perf_time_100n_sec();

        if self.m_b_first_time {
            self.m_b_first_time = false;
            self.m_ln_old_value = ln_new_value;
            self.m_old_perf_time_100n_sec = new_perf_time_100n_sec;
            return 0;
        }

        let ln_value_delta = ln_new_value.wrapping_sub(self.m_ln_old_value);
        let delta_perf_time_100n_sec =
            new_perf_time_100n_sec as f64 - self.m_old_perf_time_100n_sec as f64;

        self.m_ln_old_value = ln_new_value;
        self.m_old_perf_time_100n_sec = new_perf_time_100n_sec;

        let a = ln_value_delta as f64 / delta_perf_time_100n_sec;

        cpu_usage = (a * 100.) as i32;
        if cpu_usage < 0 {
            return 0;
        }
        cpu_usage
    }
}

impl Drop for CCpuUsage {
    /// The `CCpuUsage` destructor, `CCpuUsage::~CCpuUsage`
    /// (`imodwincpu.cpp:97`): empty.
    fn drop(&mut self) {}
}

impl Default for CCpuUsage {
    fn default() -> Self {
        Self::new()
    }
}

/// `pPerfData->PerfTime100nSec` of the block the last query returned
/// (`PERF_DATA_BLOCK` offset 72: four `WCHAR`s, seven `DWORD`/`LONG`s, a
/// 16-byte `SYSTEMTIME`, padding to 8, then `PerfTime` and `PerfFreq`).
fn perf_time_100n_sec() -> i64 {
    BUFFER.with_borrow(|buffer| read_i64(buffer, 72))
}

thread_local! {
    /// `static CBuffer Buffer(TOTALBYTES)` in `QueryPerformanceData`
    /// (`PerfCounters.h:126`): one block, grown when the query needs more.
    static BUFFER: std::cell::RefCell<Vec<u8>> =
        std::cell::RefCell::new(vec![0u8; TOTALBYTES as usize]);
}

fn read_u32(buffer: &[u8], offset: usize) -> Option<u32> {
    buffer
        .get(offset..offset + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_i64(buffer: &[u8], offset: usize) -> i64 {
    buffer
        .get(offset..offset + 8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .unwrap_or(0)
}

/// `template <class T> class CPerfCounters` (`PerfCounters.h:9`), for
/// `T = LONGLONG`.
struct CPerfCounters;

impl CPerfCounters {
    /// `GetCounterValue(PERF_DATA_BLOCK **, DWORD, DWORD, LPCTSTR)`
    /// (`PerfCounters.h:20`).
    fn get_counter_value(
        &mut self,
        dw_object_index: u32,
        dw_counter_index: u32,
        p_instance_name: Option<&str>,
    ) -> i64 {
        self.query_performance_data(dw_object_index, dw_counter_index);
        BUFFER.with_borrow(|buffer| {
            let mut ln_value: i64 = 0;

            // Get the first object type.
            let mut p_perf_obj = Self::first_object(buffer);

            // Look for the given object index
            let num_object_types = read_u32(buffer, 28).unwrap_or(0);
            for _ in 0..num_object_types {
                if read_u32(buffer, p_perf_obj + 12) == Some(dw_object_index) {
                    ln_value = Self::get_counter_value_object(
                        buffer,
                        p_perf_obj,
                        dw_counter_index,
                        p_instance_name,
                    );
                    break;
                }

                p_perf_obj = Self::next_object(buffer, p_perf_obj);
            }
            ln_value
        })
    }

    /// `GetCounterValueForProcessID(PERF_DATA_BLOCK **, DWORD, DWORD, DWORD)`
    /// (`PerfCounters.h:46`).
    fn get_counter_value_for_process_id(
        &mut self,
        dw_object_index: u32,
        dw_counter_index: u32,
        dw_process_id: u32,
    ) -> i64 {
        self.query_performance_data(dw_object_index, dw_counter_index);
        BUFFER.with_borrow(|buffer| {
            let mut ln_value: i64 = 0;

            // Get the first object type.
            let mut p_perf_obj = Self::first_object(buffer);

            // Look for the given object index
            let num_object_types = read_u32(buffer, 28).unwrap_or(0);
            for _ in 0..num_object_types {
                if read_u32(buffer, p_perf_obj + 12) == Some(dw_object_index) {
                    ln_value = Self::get_counter_value_for_process_id_object(
                        buffer,
                        p_perf_obj,
                        dw_counter_index,
                        dw_process_id,
                    );
                    break;
                }

                p_perf_obj = Self::next_object(buffer, p_perf_obj);
            }
            ln_value
        })
    }

    /// `QueryPerformanceData` (`PerfCounters.h:120`).  The performance data
    /// is read through `HKEY_PERFORMANCE_DATA`, which makes the system collect
    /// it from the object managers; the buffer grows by `BYTEINCREMENT` until
    /// the data fit.
    fn query_performance_data(&mut self, dw_object_index: u32, _dw_counter_index: u32) {
        BUFFER.with_borrow_mut(|buffer| {
            let mut buffer_size = buffer.len() as u32;
            let key_name = std::ffi::CString::new(format!("{dw_object_index}")).unwrap();

            buffer.fill(0);
            loop {
                // SAFETY: `buffer` holds `buffer_size` writable bytes.
                let l_res = unsafe {
                    RegQueryValueExA(
                        HKEY_PERFORMANCE_DATA,
                        key_name.as_ptr(),
                        core::ptr::null_mut(),
                        core::ptr::null_mut(),
                        buffer.as_mut_ptr(),
                        &mut buffer_size,
                    )
                };
                if l_res != ERROR_MORE_DATA {
                    break;
                }
                // Get a buffer that is big enough.
                buffer_size += BYTEINCREMENT;
                buffer.resize(buffer_size as usize, 0);
            }
        });
    }

    /// `GetCounterValue(PPERF_OBJECT_TYPE, DWORD, LPCTSTR)`
    /// (`PerfCounters.h:166`): the counter's value in the object, or for an
    /// object with instances, in the instance named `p_instance_name`
    /// (compared case-insensitively, as `stricmp`).
    fn get_counter_value_object(
        buffer: &[u8],
        p_perf_obj: usize,
        dw_counter_index: u32,
        p_instance_name: Option<&str>,
    ) -> i64 {
        let mut p_counter_block: Option<usize> = None;

        // Get the first counter.
        let mut p_perf_cntr = Self::first_counter(buffer, p_perf_obj);

        let num_counters = read_u32(buffer, p_perf_obj + 32).unwrap_or(0);
        for _ in 0..num_counters {
            if read_u32(buffer, p_perf_cntr + 4) == Some(dw_counter_index) {
                break;
            }

            // Get the next counter.
            p_perf_cntr = Self::next_counter(buffer, p_perf_cntr);
        }

        let num_instances = read_u32(buffer, p_perf_obj + 40).unwrap_or(0) as i32;
        if num_instances == PERF_NO_INSTANCES {
            p_counter_block =
                Some(p_perf_obj + read_u32(buffer, p_perf_obj + 4).unwrap_or(0) as usize);
        } else {
            let mut p_perf_inst = Self::first_instance(buffer, p_perf_obj);

            // Look for instance pInstanceName
            let input = p_instance_name.unwrap_or("");
            for _ in 0..num_instances {
                let name_offset = read_u32(buffer, p_perf_inst + 16).unwrap_or(0) as usize;
                let mut name: Vec<u16> = Vec::new();
                let mut at = p_perf_inst + name_offset;
                while let Some(unit) = buffer.get(at..at + 2) {
                    let unit = u16::from_le_bytes([unit[0], unit[1]]);
                    if unit == 0 {
                        break;
                    }
                    name.push(unit);
                    at += 2;
                }
                if String::from_utf16_lossy(&name).eq_ignore_ascii_case(input) {
                    p_counter_block =
                        Some(p_perf_inst + read_u32(buffer, p_perf_inst).unwrap_or(0) as usize);
                    break;
                }

                // Get the next instance.
                p_perf_inst = Self::next_instance(buffer, p_perf_inst);
            }
        }

        if let Some(block) = p_counter_block {
            let Some(counter_offset) = read_u32(buffer, p_perf_cntr + 36) else {
                return -1;
            };
            let at = block + counter_offset as usize;
            if at + 8 > buffer.len() {
                return -1;
            }
            return read_i64(buffer, at);
        }
        -1
    }

    /// `GetCounterValueForProcessID(PPERF_OBJECT_TYPE, DWORD, DWORD)`
    /// (`PerfCounters.h:232`): the counter's value for the instance whose
    /// "ID Process" counter (784) equals `dw_process_id`.
    fn get_counter_value_for_process_id_object(
        buffer: &[u8],
        p_perf_obj: usize,
        dw_counter_index: u32,
        dw_process_id: u32,
    ) -> i64 {
        let proc_id_counter: u32 = 784;

        let mut b_process_id_exist = false;
        let mut p_the_requested_perf_cntr: Option<usize> = None;
        let mut p_proc_id_perf_cntr: Option<usize> = None;
        let mut p_counter_block: Option<usize> = None;

        // Get the first counter.
        let mut p_perf_cntr = Self::first_counter(buffer, p_perf_obj);

        let num_counters = read_u32(buffer, p_perf_obj + 32).unwrap_or(0);
        for _ in 0..num_counters {
            let index = read_u32(buffer, p_perf_cntr + 4);
            if index == Some(proc_id_counter) {
                p_proc_id_perf_cntr = Some(p_perf_cntr);
                if p_the_requested_perf_cntr.is_some() {
                    break;
                }
            }

            if index == Some(dw_counter_index) {
                p_the_requested_perf_cntr = Some(p_perf_cntr);
                if p_proc_id_perf_cntr.is_some() {
                    break;
                }
            }

            // Get the next counter.
            p_perf_cntr = Self::next_counter(buffer, p_perf_cntr);
        }

        let num_instances = read_u32(buffer, p_perf_obj + 40).unwrap_or(0) as i32;
        if num_instances == PERF_NO_INSTANCES {
            p_counter_block =
                Some(p_perf_obj + read_u32(buffer, p_perf_obj + 4).unwrap_or(0) as usize);
        } else {
            let mut p_perf_inst = Self::first_instance(buffer, p_perf_obj);

            // Without an "ID Process" counter the source dereferences a null
            // counter pointer; no instance matches here instead.
            if let Some(proc_id_cntr) = p_proc_id_perf_cntr {
                for _ in 0..num_instances {
                    let block = p_perf_inst + read_u32(buffer, p_perf_inst).unwrap_or(0) as usize;
                    p_counter_block = Some(block);
                    let offset = read_u32(buffer, proc_id_cntr + 36).unwrap_or(0) as usize;
                    // `int processID = *(T*)...`: the low 32 bits
                    let process_id = read_i64(buffer, block + offset) as i32;
                    if process_id as u32 == dw_process_id {
                        b_process_id_exist = true;
                        break;
                    }

                    // Get the next instance.
                    p_perf_inst = Self::next_instance(buffer, p_perf_inst);
                }
            }
        }

        if b_process_id_exist
            && let (Some(block), Some(requested)) = (p_counter_block, p_the_requested_perf_cntr)
        {
            let offset = read_u32(buffer, requested + 36).unwrap_or(0) as usize;
            if block + offset + 8 > buffer.len() {
                return -1;
            }
            return read_i64(buffer, block + offset);
        }
        -1
    }

    // Functions used to navigate through the performance data.

    /// `FirstObject` (`PerfCounters.h:301`): past `PERF_DATA_BLOCK.HeaderLength`.
    fn first_object(buffer: &[u8]) -> usize {
        read_u32(buffer, 24).unwrap_or(0) as usize
    }

    /// `NextObject` (`PerfCounters.h:306`): past `TotalByteLength`.
    fn next_object(buffer: &[u8], perf_obj: usize) -> usize {
        perf_obj + read_u32(buffer, perf_obj).unwrap_or(0) as usize
    }

    /// `FirstCounter` (`PerfCounters.h:311`): past the object's `HeaderLength`.
    fn first_counter(buffer: &[u8], perf_obj: usize) -> usize {
        perf_obj + read_u32(buffer, perf_obj + 8).unwrap_or(0) as usize
    }

    /// `NextCounter` (`PerfCounters.h:316`): past the counter's `ByteLength`.
    fn next_counter(buffer: &[u8], perf_cntr: usize) -> usize {
        perf_cntr + read_u32(buffer, perf_cntr).unwrap_or(0) as usize
    }

    /// `FirstInstance` (`PerfCounters.h:321`): past the object's
    /// `DefinitionLength`.
    fn first_instance(buffer: &[u8], perf_obj: usize) -> usize {
        perf_obj + read_u32(buffer, perf_obj + 4).unwrap_or(0) as usize
    }

    /// `NextInstance` (`PerfCounters.h:326`): past the instance and its
    /// counter block.
    fn next_instance(buffer: &[u8], perf_inst: usize) -> usize {
        let perf_cntr_blk = perf_inst + read_u32(buffer, perf_inst).unwrap_or(0) as usize;
        perf_cntr_blk + read_u32(buffer, perf_cntr_blk).unwrap_or(0) as usize
    }
}
