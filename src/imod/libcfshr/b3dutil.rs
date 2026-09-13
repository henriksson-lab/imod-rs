//! Selected bottom-up functions from `IMOD/libcfshr/b3dutil.c`.
#![allow(dead_code)]

use core::sync::atomic::{AtomicI32, Ordering};
use core::{
    ffi::{c_char, c_void},
    ptr,
};

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
    static mut stdout: *mut libc::FILE;
    static mut stdin: *mut libc::FILE;
}

const WRITE_SBYTES_DEFAULT: i32 = 1;
const WRITE_SBYTES_ENV_VAR: &str = "WRITE_MODE0_SIGNED";
const WRITE_FLOATS_16BIT: &str = "IMOD_WRITE_FLOATS_16BIT";
const READ_SBYTES_ENV_VAR: &str = "READ_MODE0_SIGNED";
const IMOD_MRC_STAMP: i32 = 1_146_047_817;
const MRC_FLAGS_SBYTES: i32 = 1;
const OUTPUT_TYPE_TIFF: i32 = 1;
const OUTPUT_TYPE_MRC: i32 = 2;
const OUTPUT_TYPE_HDF: i32 = 5;
const OUTPUT_TYPE_JPEG: i32 = 6;
const OUTPUT_TYPE_DEFAULT: i32 = OUTPUT_TYPE_MRC;
const OUTPUT_TYPE_ENV_VAR: &str = "IMOD_OUTPUT_FORMAT";
const INVERT_MRC_ORIGIN_DEFAULT: i32 = 1;
const INVERT_MRC_ORIGIN_ENV_VAR: &str = "INVERT_MRC_ORIGIN";
static S_WRITE_BYTES_OVERRIDE: AtomicI32 = AtomicI32::new(-1);
static S_OUTPUT_TYPE_OVERRIDE: AtomicI32 = AtomicI32::new(-1);
static S_WRITE_4_BIT_MODE: AtomicI32 = AtomicI32::new(0);
static S_WRITE_16_BIT_FLOATS: AtomicI32 = AtomicI32::new(-1);
static S_INVERT_MRC_ORIGIN_OVERRIDE: AtomicI32 = AtomicI32::new(-1);
static S_ALL_BIG_TIFF_OVERRIDE: AtomicI32 = AtomicI32::new(-1);
static S_B3DRAN_FIRST_TIME: AtomicI32 = AtomicI32::new(1);
static S_B3DRAN_LAST_SEED: AtomicI32 = AtomicI32::new(0);
static mut S_LOCK_FILES: [i32; 8] = [0; 8];
static mut S_LOCKS_USED: [i32; 8] = [0; 8];
static mut S_LOCK_TIMEOUTS: [f32; 8] = [0.; 8];
static mut S_INITED_LOCKS: i32 = 0;
static mut S_DFLT_LOCK_TIMEOUT: f32 = 30.;
static mut STORE_ERROR: i32 = 0;
static mut ERROR_MESS: [u8; 512] = [0; 512];

/// Matches C `b3dError(FILE *, const char *, ...)` (`b3dutil.c:862`).
/// `std::fmt::Arguments` is the Rust counterpart of the C format/varargs pair.
pub fn b3d_error(fout: *mut libc::FILE, arguments: core::fmt::Arguments<'_>) {
    let message = arguments.to_string();
    let message_length = message
        .bytes()
        .position(|byte| byte == 0)
        .unwrap_or(message.len())
        .min(511);
    unsafe {
        core::ptr::write_bytes(core::ptr::addr_of_mut!(ERROR_MESS).cast::<u8>(), 0, 512);
        core::ptr::copy_nonoverlapping(
            message.as_ptr(),
            core::ptr::addr_of_mut!(ERROR_MESS).cast::<u8>(),
            message_length,
        );
        if fout == stderr && STORE_ERROR < 0 {
            libc::fputs(
                core::ptr::addr_of!(ERROR_MESS).cast::<libc::c_char>(),
                stdout,
            );
        } else if !fout.is_null() && STORE_ERROR <= 0 {
            libc::fputs(core::ptr::addr_of!(ERROR_MESS).cast::<libc::c_char>(), fout);
        }
    }
}

/// Matches C `b3dSetStoreError(int)` (`b3dutil.c:881`).
pub fn b3d_set_store_error(value: i32) {
    unsafe { STORE_ERROR = value }
}

/// Matches C `b3dGetStoreError(void)` (`b3dutil.c:887`).
pub fn b3d_get_store_error() -> i32 {
    unsafe { STORE_ERROR }
}

/// Matches C `b3dGetError(void)` (`b3dutil.c:892`).
pub fn b3d_get_error() -> *mut libc::c_char {
    core::ptr::addr_of_mut!(ERROR_MESS).cast::<libc::c_char>()
}

/// Matches C `overrideWriteBytes(int)` (`b3dutil.c:383`).
pub fn override_write_bytes(value: i32) {
    S_WRITE_BYTES_OVERRIDE.store(value, Ordering::SeqCst);
}

/// Matches C `writeBytesSigned(void)` (`b3dutil.c:399`).
pub fn write_bytes_signed() -> i32 {
    let mut value = WRITE_SBYTES_DEFAULT;
    let override_value = S_WRITE_BYTES_OVERRIDE.load(Ordering::SeqCst);
    if override_value >= 0 {
        return override_value;
    }
    if let Ok(env_ptr) = std::env::var(WRITE_SBYTES_ENV_VAR) {
        value = env_ptr.parse::<i32>().unwrap_or(0);
    }
    value
}

/// Matches C `overrideInvertMrcOrigin(int)` (`b3dutil.c:502`).
pub fn override_invert_mrc_origin(value: i32) {
    S_INVERT_MRC_ORIGIN_OVERRIDE.store(value, Ordering::SeqCst);
}

/// Matches C `invertMrcOriginOnOutput(void)` (`b3dutil.c:519`).
pub fn invert_mrc_origin_on_output() -> i32 {
    let mut value = INVERT_MRC_ORIGIN_DEFAULT;
    let override_value = S_INVERT_MRC_ORIGIN_OVERRIDE.load(Ordering::SeqCst);
    if override_value >= 0 {
        return override_value;
    }
    if let Ok(environment_value) = std::env::var(INVERT_MRC_ORIGIN_ENV_VAR) {
        value = environment_value.parse::<i32>().unwrap_or(0);
    }
    value
}

/// Matches C `setOrClearFlags(b3dUInt32 *, b3dUInt32, int)` (`b3dutil.c:1205`).
pub fn set_or_clear_flags(flags: &mut u32, mask: u32, state: i32) {
    if state != 0 {
        *flags |= mask;
    } else {
        *flags &= !mask;
    }
}

/// Matches C `overrideOutputType(int)` (`b3dutil.c:538`).
pub fn override_output_type(output_type: i32) {
    S_OUTPUT_TYPE_OVERRIDE.store(output_type, Ordering::SeqCst);
}

/// Matches C `b3dOutputFileType(void)` (`b3dutil.c:560`).
pub fn b3d_output_file_type() -> i32 {
    let mut output_type = OUTPUT_TYPE_DEFAULT;
    if let Ok(environment_type) = std::env::var(OUTPUT_TYPE_ENV_VAR) {
        if environment_type == "MRC" {
            output_type = OUTPUT_TYPE_MRC;
        } else if environment_type == "TIFF" || environment_type == "TIF" {
            output_type = OUTPUT_TYPE_TIFF;
        } else if environment_type == "JPEG" || environment_type == "JPG" {
            output_type = OUTPUT_TYPE_JPEG;
        } else if environment_type == "HDF" {
            output_type = OUTPUT_TYPE_HDF;
        }
    }
    let override_type = S_OUTPUT_TYPE_OVERRIDE.load(Ordering::SeqCst);
    if override_type == OUTPUT_TYPE_MRC
        || override_type == OUTPUT_TYPE_TIFF
        || override_type == OUTPUT_TYPE_HDF
        || override_type == OUTPUT_TYPE_JPEG
    {
        output_type = override_type;
    }
    output_type
}

/// Matches C `setOutputTypeFromString(const char *)` (`b3dutil.c:597`).
pub fn set_output_type_from_string(type_string: &str) -> i32 {
    let mut return_value = -1;
    if type_string == "MRC" || type_string == "mrc" {
        return_value = OUTPUT_TYPE_MRC;
    } else if type_string == "TIFF"
        || type_string == "TIF"
        || type_string == "tiff"
        || type_string == "tif"
    {
        return_value = OUTPUT_TYPE_TIFF;
    } else if type_string == "JPEG"
        || type_string == "JPG"
        || type_string == "jpeg"
        || type_string == "jpg"
    {
        return_value = OUTPUT_TYPE_JPEG;
    } else if type_string == "HDF" || type_string == "hdf" {
        return_value = OUTPUT_TYPE_HDF;
    }
    if return_value > 0 {
        override_output_type(return_value);
    }
    return_value
}

/// Matches C `set4BitOutputMode(int)` (`b3dutil.c:724`).
pub fn set_4_bit_output_mode(in_val: i32) {
    S_WRITE_4_BIT_MODE.store(in_val, Ordering::SeqCst);
}

/// Matches C `write4BitModeForBytes(void)` (`b3dutil.c:739`).
pub fn write_4_bit_mode_for_bytes() -> i32 {
    S_WRITE_4_BIT_MODE.load(Ordering::SeqCst)
}

/// Matches C `write16BitModeForFloats(void)` (`b3dutil.c:792`).
pub fn write_16_bit_mode_for_floats() -> i32 {
    let mut value = 0;
    let saved_value = S_WRITE_16_BIT_FLOATS.load(Ordering::SeqCst);
    if saved_value >= 0 {
        return saved_value;
    }
    if let Ok(env_ptr) = std::env::var(WRITE_FLOATS_16BIT) {
        value = env_ptr.parse::<i32>().unwrap_or(0);
    }
    value
}

/// Matches C `b3dHeaderItemBytes(int *, int *)` (`b3dutil.c:1154`).
pub fn b3d_header_item_bytes() -> (i32, [i32; 32]) {
    let extra_bytes: [i16; 11] = [2, 6, 4, 2, 2, 4, 2, 4, 2, 4, 2];
    let mut nbytes = [0; 32];
    for i in 0..extra_bytes.len() {
        nbytes[i] = extra_bytes[i] as i32;
    }
    (extra_bytes.len() as i32, nbytes)
}

/// Matches C `extraIsNbytesAndFlags(int, int)` (`b3dutil.c:1177`).
pub fn extra_is_nbytes_and_flags(nint: i32, nreal: i32) -> i32 {
    let (flag_count, extra_bytes) = b3d_header_item_bytes();
    let mut extra_tot = 0;
    for i in 0..flag_count as usize {
        if nreal & (1 << i) != 0 {
            extra_tot += extra_bytes[i];
        }
    }
    if nint < 0 || nreal < 0 || nint + nreal == 0 || extra_tot != nint || nreal >= (1 << flag_count)
    {
        return 0;
    }
    1
}

/// Matches C `dataSizeForMode(int, int *, int *)` (`b3dutil.c:1113`).
pub fn data_size_for_mode(mode: i32, data_size: &mut i32, channels: &mut i32) -> i32 {
    match mode {
        0 => {
            *data_size = 1;
            *channels = 1;
        }
        1 | 6 => {
            *data_size = 2;
            *channels = 1;
        }
        2 => {
            *data_size = 4;
            *channels = 1;
        }
        3 => {
            *data_size = 2;
            *channels = 2;
        }
        4 => {
            *data_size = 4;
            *channels = 2;
        }
        16 => {
            *data_size = 1;
            *channels = 3;
        }
        99 => {
            *data_size = 4;
            *channels = 3;
        }
        _ => return -1,
    }
    0
}

/// Matches C `readBytesSigned(int, int, int, float, float)` (`b3dutil.c:425`).
pub fn read_bytes_signed(stamp: i32, flags: i32, mode: i32, dmin: f32, dmax: f32) -> i32 {
    let mut env_val = 0;
    if mode != 0 {
        return 0;
    }
    if let Ok(env_ptr) = std::env::var(READ_SBYTES_ENV_VAR) {
        env_val = env_ptr.parse::<i32>().unwrap_or(0);
    }
    if stamp == IMOD_MRC_STAMP {
        let mut value = flags & MRC_FLAGS_SBYTES;
        if env_val < -1 {
            value = 0;
        }
        if env_val > 1 {
            value = 1;
        }
        return value;
    }
    let mut value;
    if dmin < 0.0 && dmax < 128.0 {
        value = 1;
    } else if dmin >= 0.0 && dmax >= 128.0 {
        value = 0;
    } else if dmin < 0.0 && dmax >= 128.0 {
        value = if -dmin > dmax - 128.0 { 1 } else { 0 };
    } else {
        value = 0;
    }
    if env_val < 0 {
        value = 0;
    }
    if env_val > 0 {
        value = 1;
    }
    value
}

/// Matches C `imodGetpid(void)` (`b3dutil.c:347`).
pub fn imod_getpid() -> i32 {
    unsafe { libc::getpid() }
}
/// The floating-point argument conversion for `imodconfig.h:13`'s
/// `#define SPRINTF(ast) ast = QString::asprintf`, generated by
/// `IMOD/setup2:810`.
///
/// Every `SPRINTF` site in `IMOD/3dmod` formats through `QString::asprintf`,
/// not through libc, and Qt does not use the C library's formatter.  Measured
/// against Qt5Core over `%g`, `%f`, `%.3f`, `%.6g`, `%7.3f`, `%10.6f` and
/// `%13.6f` with negative and positive zero, both infinities, NaN, 1e20,
/// 1e-300, 1e-5 and ordinary values, the two agree everywhere except one
/// case: for **negative zero** Qt prints `0` / `0.000000` where libc prints
/// `-0` / `-0.000000`.  Pass every floating-point argument of a `SPRINTF`
/// site through this, so a `-0.` that the source's own arithmetic produces
/// prints the way the source's own toolkit prints it.
///
/// This is not a tidying helper: it is the translation of a build-generated
/// macro at a toolkit boundary the crate does not have.  `-0. == 0.` is true
/// in IEEE, so the comparison catches both zeros and leaves NaN and the
/// infinities alone.
pub fn sprintf_arg(value: f64) -> f64 {
    if value == 0. { 0. } else { value }
}
/// Matches C `imodVersion` (`b3dutil.c:148`).
///
/// `VERSION`, `VERSION_NAME`, and `COPYRIGHT_YEARS` are generated into
/// `imodconfig.h` by `IMOD/setup2:411-419` from `IMOD/.version` ("5.2.17") and
/// `IMOD/setup2:10` ("1994-2025") for the pinned revision.  The stale 4.8.16 /
/// 1994-2014 pair in `IMOD/sysdep/win/VC-imodconfig.h` is a checked-in Visual
/// Studio config, not the configuration this revision builds with.
pub unsafe fn imod_version(program_name: *const c_char) -> i32 {
    if !program_name.is_null() {
        libc::printf(
            c"%s Version %s %s %s\n".as_ptr(),
            program_name,
            c"5.2.17".as_ptr(),
            IMOD_BUILD_DATE.as_ptr(),
            IMOD_BUILD_TIME.as_ptr(),
        );
    }
    5217
}
/// C `__DATE__` and `__TIME__` for this build, in the identical C field
/// formats.  They are compilation metadata rather than input-dependent output,
/// so the deterministic part compared against the reference is the field
/// layout, not the timestamp value.
pub const IMOD_BUILD_DATE: &core::ffi::CStr =
    match core::ffi::CStr::from_bytes_with_nul(concat!(env!("IMOD_BUILD_DATE"), "\0").as_bytes()) {
        Ok(value) => value,
        Err(_) => panic!("build date"),
    };
pub const IMOD_BUILD_TIME: &core::ffi::CStr =
    match core::ffi::CStr::from_bytes_with_nul(concat!(env!("IMOD_BUILD_TIME"), "\0").as_bytes()) {
        Ok(value) => value,
        Err(_) => panic!("build time"),
    };
/// Matches C `imodCopyright` (`b3dutil.c:157`).
pub fn imod_copyright() {
    unsafe {
        libc::printf(
            c"Copyright (C) %s by the %s\n".as_ptr(),
            c"1994-2025".as_ptr(),
            c"Regents of the University of Colorado".as_ptr(),
        );
    }
}
/// Matches C `imodUsageHeader` (`b3dutil.c:165`).
pub unsafe fn imod_usage_header(program_name: *const c_char) {
    imod_version(program_name);
    imod_copyright();
}
/// Matches C `IMOD_DIR_or_default` (`b3dutil.c:178`).
pub unsafe fn imod_dir_or_default(assumed: *mut i32) -> *mut c_char {
    let environment = libc::getenv(c"IMOD_DIR".as_ptr());
    if !assumed.is_null() {
        *assumed = if environment.is_null() {
            if libc::access(c"/usr/local/IMOD".as_ptr(), libc::F_OK) == 0 {
                1
            } else {
                2
            }
        } else {
            0
        };
    }
    if environment.is_null() {
        c"/usr/local/IMOD".as_ptr().cast_mut()
    } else {
        environment
    }
}
/// Matches C `imodProgName` (`b3dutil.c:215`). Allocation behavior is retained for `.exe` paths.
pub unsafe fn imod_prog_name(full_name: *const c_char) -> *mut c_char {
    let forward = libc::strrchr(full_name, b'/' as i32);
    let backward = libc::strrchr(full_name, b'\\' as i32);
    let tail = if backward > forward {
        backward
    } else {
        forward
    };
    if tail.is_null() {
        return full_name.cast_mut();
    }
    let tail = tail.add(1);
    let length = libc::strlen(tail);
    let extension = libc::strstr(tail, c".exe".as_ptr());
    if extension.is_null() || extension != tail.add(length - 4) {
        return tail;
    }
    let output = libc::strdup(tail);
    if !output.is_null() {
        *output.add(length - 4) = 0;
    }
    output
}
/// Matches C `imodBackupFile` (`b3dutil.c:241`).
pub unsafe fn imod_backup_file(filename: *const c_char) -> i32 {
    let mut stat = core::mem::zeroed::<libc::stat>();
    if libc::stat(filename, &mut stat) != 0 {
        return 0;
    }
    let backup = libc::malloc(libc::strlen(filename) + 3).cast::<c_char>();
    if backup.is_null() {
        return -2;
    }
    libc::sprintf(backup, c"%s~".as_ptr(), filename);
    if libc::stat(backup, &mut stat) == 0 && libc::remove(backup) != 0 {
        libc::free(backup.cast());
        return -1;
    }
    let result = libc::rename(filename, backup);
    libc::free(backup.cast());
    result
}
/// Matches C `b3dOpenFile` (`b3dutil.c:302`).
pub unsafe fn b3d_open_file(name: *const c_char, mut mode: *const c_char) -> *mut libc::FILE {
    if libc::strcmp(mode, c"ro".as_ptr()) == 0 || libc::strcmp(mode, c"RO".as_ptr()) == 0 {
        mode = c"r".as_ptr();
    } else if libc::strcmp(mode, c"old".as_ptr()) == 0 || libc::strcmp(mode, c"OLD".as_ptr()) == 0 {
        mode = c"r+".as_ptr();
    } else if libc::strcmp(mode, c"new".as_ptr()) == 0 || libc::strcmp(mode, c"NEW".as_ptr()) == 0 {
        mode = c"w+".as_ptr();
    }
    if *mode == b'w' as c_char {
        imod_backup_file(name);
    }
    libc::fopen(name, mode)
}
/// Matches C `pidToStderr` (`b3dutil.c:359`).
pub fn pid_to_stderr() {
    unsafe {
        libc::fprintf(stderr, c"Shell PID: %d\n".as_ptr(), libc::getpid());
        libc::fflush(stderr);
    }
}
/// Matches C `imodgetpid(void)` (`b3dutil.c:353`).
pub fn imodgetpid() -> i32 {
    imod_getpid()
}
/// Matches C `imodgetstamp(void)` (`b3dutil.c:372`).
pub fn imodgetstamp() -> i32 {
    IMOD_MRC_STAMP
}

/// Matches C `b3dShiftBytes` (`b3dutil.c:474`).
pub unsafe fn b3d_shift_bytes(
    usbuf: *mut u8,
    sbuf: *mut i8,
    nx: i32,
    ny: i32,
    direction: i32,
    bytes_signed: i32,
) {
    if bytes_signed == 0 {
        return;
    }
    let nxy = nx as usize * ny as usize;
    for i in 0..nxy {
        if direction >= 0 {
            *sbuf.add(i) = (*usbuf.add(i) as i32 - 128) as i8;
        } else {
            *usbuf.add(i) = (*sbuf.add(i) as i32 + 128) as u8;
        }
    }
}
/// Matches C `overrideAllBigTiff` (`b3dutil.c:646`).
pub fn override_all_big_tiff(value: i32) {
    S_ALL_BIG_TIFF_OVERRIDE.store(value, Ordering::SeqCst);
}
/// Matches C `makeAllBigTiff` (`b3dutil.c:662`).
pub fn make_all_big_tiff() -> i32 {
    let mut value = std::env::var("IMOD_ALL_BIG_TIFF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let override_value = S_ALL_BIG_TIFF_OVERRIDE.load(Ordering::SeqCst);
    if override_value >= 0 {
        value = override_value;
    }
    value
}
/// Matches C `setNextOutputSize` (`b3dutil.c:678`).
pub fn set_next_output_size(nx: i32, ny: i32, nz: i32, mode: i32) {
    let mut bytes = 0;
    let mut channels = 0;
    if data_size_for_mode(mode, &mut bytes, &mut channels) != 0 {
        return;
    }
    override_all_big_tiff(
        if (nx as f64 * ny as f64) * nz as f64 * channels as f64 * bytes as f64 > 4.0e9 {
            1
        } else {
            0
        },
    );
}
/// Matches C `setTiffCompressionType` (`b3dutil.c:699`).
pub fn set_tiff_compression_type(type_index: i32, override_env: i32) -> i32 {
    let type_map = ['1', '5', '7', '8'];
    if !(0..4).contains(&type_index) {
        return 1;
    }
    if override_env == 0 && std::env::var_os("IMOD_TIFF_COMPRESSION").is_some() {
        return 0;
    }
    unsafe {
        libc::putenv(
            format!("IMOD_TIFF_COMPRESSION={}", type_map[type_index as usize])
                .leak()
                .as_mut_ptr()
                .cast(),
        );
    }
    0
}
/// Matches C `setFloat16outputMode` (`b3dutil.c:751`).
pub fn set_float_16_output_mode(in_val: i32, test_for_mrc: i32) {
    if in_val != 0 && test_for_mrc != 0 && b3d_output_file_type() != OUTPUT_TYPE_MRC {
        b3d_error(
            unsafe { stderr },
            format_args!("Mode 12 output is available only when the output file type is MRC"),
        );
    }
    S_WRITE_16_BIT_FLOATS.store(in_val, Ordering::SeqCst);
}
/// Matches C `setFloatOutputForEnteredMode` (`b3dutil.c:770`).
pub fn set_float_output_for_entered_mode(mode: i32) -> i32 {
    if mode != 2 && mode != 12 {
        return mode;
    }
    let val = if mode == 12 { 1 } else { 0 };
    set_float_16_output_mode(val, val);
    2
}

/// Matches C `f2cString` (`b3dutil.c:811`). Caller owns the returned allocation.
pub unsafe fn f2c_string(string: *const c_char, string_size: i32) -> *mut c_char {
    let mut index = string_size - 1;
    while index >= 0 && *string.add(index as usize) == b' ' as c_char {
        index -= 1;
    }
    let output = libc::malloc((index + 2) as usize).cast::<c_char>();
    if output.is_null() {
        return ptr::null_mut();
    }
    if index >= 0 {
        ptr::copy_nonoverlapping(string, output, index as usize + 1);
    }
    *output.add((index + 1) as usize) = 0;
    output
}
/// Matches C `c2fString` (`b3dutil.c:835`).
pub unsafe fn c2f_string(
    mut c_string: *const c_char,
    mut fortran_string: *mut c_char,
    mut size: i32,
) -> i32 {
    while *c_string != 0 && size > 0 {
        *fortran_string = *c_string;
        fortran_string = fortran_string.add(1);
        c_string = c_string.add(1);
        size -= 1;
    }
    if *c_string != 0 {
        return -1;
    }
    while size > 0 {
        *fortran_string = b' ' as c_char;
        fortran_string = fortran_string.add(1);
        size -= 1;
    }
    0
}
/// Matches C `b3dFseek` (`b3dutil.c:899`).
pub unsafe fn b3d_fseek(file: *mut libc::FILE, offset: i32, flag: i32) -> i32 {
    if file == stdin {
        0
    } else {
        libc::fseek(file, offset as libc::c_long, flag)
    }
}
/// Matches C `b3dFread` (`b3dutil.c:919`).
pub unsafe fn b3d_fread(
    buffer: *mut c_void,
    size: usize,
    count: usize,
    file: *mut libc::FILE,
) -> usize {
    libc::fread(buffer, size, count, file)
}
/// Matches C `b3dFwrite` (`b3dutil.c:953`).
pub unsafe fn b3d_fwrite(
    buffer: *const c_void,
    size: usize,
    count: usize,
    file: *mut libc::FILE,
) -> usize {
    libc::fwrite(buffer, size, count, file)
}
/// Matches C `b3dRewind` (`b3dutil.c:964`).
pub unsafe fn b3d_rewind(file: *mut libc::FILE) {
    b3d_fseek(file, 0, libc::SEEK_SET);
}
/// Matches C `mrc_big_seek` (`b3dutil.c:976`).
pub unsafe fn mrc_big_seek(
    file: *mut libc::FILE,
    base: i32,
    size1: i32,
    size2: i32,
    mut flag: i32,
) -> i32 {
    if base != 0 || ((size1 == 0 || size2 == 0) && flag == libc::SEEK_SET) {
        let err = b3d_fseek(file, base, flag);
        if err != 0 {
            return err;
        }
        flag = libc::SEEK_CUR;
    }
    if size1 == 0 || size2 == 0 {
        return 0;
    }
    let abs1 = size1.abs();
    let abs2 = size2.abs();
    let smaller = abs1.min(abs2);
    let mut bigger = abs1.max(abs2);
    let step_limit = 2_000_000_000 / bigger;
    let mut todo = smaller;
    if (size1 < 0) != (size2 < 0) {
        bigger = -bigger;
    }
    while todo > 0 {
        let doing = todo.min(step_limit);
        let err = b3d_fseek(file, doing * bigger, flag);
        if err != 0 {
            return err;
        }
        todo -= doing;
        flag = libc::SEEK_CUR;
    }
    0
}
/// Matches C `mrcHugeSeek` (`b3dutil.c:1046`).
pub unsafe fn mrc_huge_seek(
    file: *mut libc::FILE,
    mut base: i32,
    x: i32,
    y: i32,
    z: i32,
    nx: i32,
    ny: i32,
    dsize: i32,
    flag: i32,
) -> i32 {
    let test_size = nx as f64 * ny as f64 * dsize as f64;
    base += x * dsize;
    if test_size.abs() < 2.0e9 {
        base += nx * y * dsize;
        mrc_big_seek(file, base, nx * ny * dsize, z, flag)
    } else {
        mrc_big_seek(file, base, nx * dsize, y + ny * z, flag)
    }
}
/// Matches C `fgetline` (`b3dutil.c:1075`).
pub unsafe fn fgetline(file: *mut libc::FILE, string: *mut c_char, limit: i32) -> i32 {
    if file.is_null() || limit < 3 {
        return -1;
    }
    let mut index = 0;
    let mut character = libc::EOF;
    while index < limit - 1 {
        character = libc::fgetc(file);
        if character == libc::EOF || character == b'\n' as i32 {
            break;
        }
        *string.add(index as usize) = character as c_char;
        index += 1;
    }
    if index > 0 && *string.add(index as usize - 1) == b'\r' as c_char {
        index -= 1;
    }
    *string.add(index as usize) = 0;
    if character == libc::EOF {
        -(index + 2)
    } else {
        index
    }
}

/// Matches C `numberInList` (`b3dutil.c:1215`).
pub unsafe fn number_in_list(number: i32, list: *const i32, count: i32, no_list_value: i32) -> i32 {
    if list.is_null() || count == 0 {
        return no_list_value;
    }
    for index in 0..count as usize {
        if number == *list.add(index) {
            return 1;
        }
    }
    0
}
/// Matches C `balancedGroupLimits` (`b3dutil.c:1237`).
pub unsafe fn balanced_group_limits(
    total: i32,
    groups: i32,
    group: i32,
    start: *mut i32,
    end: *mut i32,
) {
    let base = total / groups;
    let remainder = total % groups;
    *start = group * base + group.min(remainder);
    *end = (group + 1) * base + (group + 1).min(remainder) - 1;
}
/// Matches C `groupLimitsRemainderAtEnd` (`b3dutil.c:1256`).
pub unsafe fn group_limits_remainder_at_end(
    total: i32,
    groups: i32,
    group: i32,
    start: *mut i32,
    end: *mut i32,
) {
    let mut inverse_start = 0;
    let mut inverse_end = 0;
    balanced_group_limits(
        total,
        groups,
        groups - 1 - group,
        &mut inverse_start,
        &mut inverse_end,
    );
    *start = total - 1 - inverse_end;
    *end = total - 1 - inverse_start;
}
/// Matches C `b3dIMin` (`b3dutil.c:1285`). Rust’s slice is the stable equivalent of C varargs.
pub fn b3d_i_min(values: &[i32]) -> i32 {
    let mut extreme = 0;
    for (index, value) in values.iter().enumerate() {
        if index == 0 || *value < extreme {
            extreme = *value;
        }
    }
    extreme
}
/// Matches C `b3dIMax` (`b3dutil.c:1307`). Rust’s slice is the stable equivalent of C varargs.
pub fn b3d_i_max(values: &[i32]) -> i32 {
    let mut extreme = 0;
    for (index, value) in values.iter().enumerate() {
        if index == 0 || *value > extreme {
            extreme = *value;
        }
    }
    extreme
}
/// Matches C `makeLinePointers` (`b3dutil.c:1269`). Caller owns returned allocation.
pub unsafe fn make_line_pointers(
    array: *mut c_void,
    xsize: i32,
    ysize: i32,
    dsize: i32,
) -> *mut *mut u8 {
    let lines = libc::malloc(ysize as usize * core::mem::size_of::<*mut u8>()).cast::<*mut u8>();
    if lines.is_null() {
        return ptr::null_mut();
    }
    for index in 0..ysize as usize {
        *lines.add(index) = array
            .cast::<u8>()
            .add(xsize as usize * index * dsize as usize);
    }
    lines
}
/// Matches C `cputime` (`b3dutil.c:1327`).
pub fn cputime() -> f64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut time);
    }
    time.tv_sec as f64 + time.tv_nsec as f64 / 1.0e9
}
/// Matches C `b3dMilliSleep` (`b3dutil.c:1354`).
pub fn b3d_milli_sleep(milliseconds: i32) -> i32 {
    let mut request = libc::timespec {
        tv_sec: (milliseconds / 1000) as _,
        tv_nsec: (1_000_000 * (milliseconds % 1000)) as _,
    };
    let mut remain = request;
    let mut result = 0;
    unsafe {
        while libc::nanosleep(&request, &mut remain) != 0 {
            if *libc::__errno_location() != libc::EINTR {
                return -1;
            }
            request = remain;
            result += 1;
        }
    }
    result
}
/// Matches C `totalCudaCores` (`b3dutil.c:1435`).
pub fn total_cuda_cores(major: i32, minor: i32, multiprocessors: i32) -> i32 {
    let capability = (major << 4) + minor;
    let limits = [0x10, 0x20, 0x21, 0x30, 0x50, -1];
    let cores = [8, 32, 48, 192, 128, -1];
    let mut index = 0;
    while limits[index] > 0 {
        if capability < limits[index + 1] || limits[index + 1] < 0 {
            break;
        }
        index += 1;
    }
    cores[index] * multiprocessors
}
/// Matches C `b3dPhysicalMemory` (`b3dutil.c:1454`).
pub fn b3d_physical_memory() -> f64 {
    if let Ok(value) = std::env::var("IMOD_PHYSICAL_MEMORY") {
        let limit = value.parse::<i32>().unwrap_or(0);
        return if limit >= 10 {
            limit as f64 * 1024. * 1024.
        } else {
            0.
        };
    }
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let size = libc::sysconf(libc::_SC_PAGE_SIZE);
        if pages >= 0 && size >= 0 {
            (pages * size) as f64
        } else {
            0.
        }
    }
}
/// Matches C `b3dAddressableMemory` (`b3dutil.c:1493`).
pub fn b3d_addressable_memory() -> f64 {
    let mut memory = b3d_physical_memory();
    if core::mem::size_of::<*const c_void>() == 4 {
        memory = memory.min(4.0e9);
    }
    memory
}
/// Matches C `standardMemoryLimitMB` (`b3dutil.c:1513`).
pub fn standard_memory_limit_mb(half_point: i32) -> f64 {
    let physical = b3d_addressable_memory() / (1024. * 1024.);
    if physical == 0. {
        return 0.;
    }
    if physical < half_point as f64 {
        (0.75 * physical)
            .min(physical - 1000.)
            .clamp(400., half_point as f64 / 2.)
    } else {
        physical / 2.
    }
}
/// Matches C `b3drand` (`b3dutil.c:1754`).
pub fn b3drand() -> f32 {
    unsafe { libc::rand() as f32 / libc::RAND_MAX as f32 }
}
/// Matches C `b3dsrand` (`b3dutil.c:1763`).
pub unsafe fn b3dsrand(seed: *const i32) {
    libc::srand(*seed as u32);
}
/// Matches C `b3dran` (`b3dutil.c:1776`).
pub unsafe fn b3dran(seed: *const i32) -> f32 {
    if S_B3DRAN_FIRST_TIME.load(Ordering::SeqCst) != 0
        || *seed != S_B3DRAN_LAST_SEED.load(Ordering::SeqCst)
    {
        libc::srand(*seed as u32);
        S_B3DRAN_LAST_SEED.store(*seed, Ordering::SeqCst);
        S_B3DRAN_FIRST_TIME.store(0, Ordering::SeqCst);
    }
    b3drand()
}
/// Matches C `angleWithinLimits` (`b3dutil.c:1791`).
pub fn angle_within_limits(mut angle: f32, lower_limit: f32, upper_limit: f32) -> f64 {
    let lower = lower_limit.min(upper_limit);
    let upper = lower_limit.max(upper_limit);
    let range = upper - lower;
    while angle <= lower {
        angle += range;
    }
    while angle > upper {
        angle -= range;
    }
    angle as f64
}
/// Matches C `b3dSetLockTimeout` (`b3dutil.c:1859`).
pub unsafe fn b3d_set_lock_timeout(timeout: f32) {
    S_DFLT_LOCK_TIMEOUT = timeout;
}
/// Matches C `b3dOpenLockFile` (`b3dutil.c:1876`).
pub unsafe fn b3d_open_lock_file(filename: *const c_char) -> i32 {
    if S_INITED_LOCKS == 0 {
        for index in 0..8 {
            S_LOCKS_USED[index] = -1;
        }
    }
    S_INITED_LOCKS = 1;
    let mut index = 0;
    while index < 8 && S_LOCKS_USED[index] >= 0 {
        index += 1;
    }
    if index >= 8 {
        return -1;
    }
    let descriptor = libc::open(filename, libc::O_RDWR);
    if descriptor < 0 {
        return -2;
    }
    S_LOCK_FILES[index] = descriptor;
    S_LOCKS_USED[index] = 0;
    S_LOCK_TIMEOUTS[index] = S_DFLT_LOCK_TIMEOUT;
    index as i32
}
/// Matches C `b3dLockFile` (`b3dutil.c:1922`).
pub unsafe fn b3d_lock_file(index: i32) -> i32 {
    if !(0..8).contains(&index) {
        return -1;
    }
    let index = index as usize;
    if S_LOCKS_USED[index] < 0 {
        return -2;
    }
    if S_LOCKS_USED[index] > 0 {
        S_LOCKS_USED[index] += 1;
        return 0;
    }
    let mut lock = libc::flock {
        l_type: libc::F_WRLCK as _,
        l_whence: libc::SEEK_SET as _,
        l_start: 0,
        l_len: 1024,
        l_pid: 0,
    };
    let mut started = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut started);
    loop {
        if libc::fcntl(S_LOCK_FILES[index], libc::F_SETLK, &lock) >= 0 {
            S_LOCKS_USED[index] += 1;
            return 0;
        }
        let mut now = started;
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now);
        if now.tv_sec as f64 + now.tv_nsec as f64 / 1.0e9
            - (started.tv_sec as f64 + started.tv_nsec as f64 / 1.0e9)
            >= S_LOCK_TIMEOUTS[index] as f64
        {
            return 1;
        }
        b3d_milli_sleep(50);
    }
}
/// Matches C `b3dUnlockFile` (`b3dutil.c:1970`).
pub unsafe fn b3d_unlock_file(index: i32) -> i32 {
    if !(0..8).contains(&index) {
        return -1;
    }
    let index = index as usize;
    if S_LOCKS_USED[index] < 0 {
        return -2;
    }
    if S_LOCKS_USED[index] == 0 {
        return -3;
    }
    if S_LOCKS_USED[index] == 1 {
        let lock = libc::flock {
            l_type: libc::F_UNLCK as _,
            l_whence: libc::SEEK_SET as _,
            l_start: 0,
            l_len: 1024,
            l_pid: 0,
        };
        if libc::fcntl(S_LOCK_FILES[index], libc::F_SETLK, &lock) < 0 {
            return 1;
        }
    }
    S_LOCKS_USED[index] -= 1;
    0
}
/// Matches C `b3dCloseLockFile` (`b3dutil.c:2009`).
pub unsafe fn b3d_close_lock_file(index: i32) -> i32 {
    if !(0..8).contains(&index) {
        return -1;
    }
    let index = index as usize;
    if S_LOCKS_USED[index] < 0 {
        return -2;
    }
    if libc::close(S_LOCK_FILES[index]) < 0 {
        return 1;
    }
    S_LOCKS_USED[index] = -1;
    0
}

/// Matches C `imodbackupfile` (`b3dutil.c:282`).
pub unsafe fn imodbackupfile(filename: *const c_char, length: i32) -> i32 {
    let string = f2c_string(filename, length);
    if string.is_null() {
        return -2;
    }
    let result = imod_backup_file(string);
    libc::free(string.cast());
    result
}
/// Matches C `imodgetenv` (`b3dutil.c:333`).
pub unsafe fn imodgetenv(
    variable: *const c_char,
    value: *mut c_char,
    variable_size: i32,
    value_size: i32,
) -> i32 {
    let string = f2c_string(variable, variable_size);
    if string.is_null() {
        return -1;
    }
    let environment = libc::getenv(string);
    libc::free(string.cast());
    if environment.is_null() {
        1
    } else {
        c2f_string(environment, value, value_size)
    }
}
/// Matches C `pidtostderr` (`b3dutil.c:366`).
pub fn pidtostderr() {
    pid_to_stderr();
}
/// Matches C `overridewritebytes` (`b3dutil.c:389`).
pub unsafe fn overridewritebytes(value: *const i32) {
    override_write_bytes(*value);
}
/// Matches C `writebytessigned` (`b3dutil.c:411`).
pub fn writebytessigned() -> i32 {
    write_bytes_signed()
}
/// Matches C `readbytessigned` (`b3dutil.c:462`).
pub unsafe fn readbytessigned(
    stamp: *const i32,
    flags: *const i32,
    mode: *const i32,
    minimum: *const f32,
    maximum: *const f32,
) -> i32 {
    read_bytes_signed(*stamp, *flags, *mode, *minimum, *maximum)
}
/// Matches C `b3dshiftbytes` (`b3dutil.c:490`).
pub unsafe fn b3dshiftbytes(
    unsigned: *mut u8,
    signed: *mut i8,
    nx: *const i32,
    ny: *const i32,
    direction: *const i32,
    bytes_signed: *const i32,
) {
    b3d_shift_bytes(unsigned, signed, *nx, *ny, *direction, *bytes_signed);
}
/// Matches C `overrideinvertmrcorigin` (`b3dutil.c:508`).
pub unsafe fn overrideinvertmrcorigin(value: *const i32) {
    override_invert_mrc_origin(*value);
}
/// Matches C `overrideoutputtype` (`b3dutil.c:552`).
pub unsafe fn overrideoutputtype(value: *const i32) {
    override_output_type(*value);
}
/// Matches C `b3doutputfiletype` (`b3dutil.c:590`).
pub fn b3doutputfiletype() -> i32 {
    b3d_output_file_type()
}
/// Matches C `setoutputtypefromstring` (`b3dutil.c:628`).
pub unsafe fn setoutputtypefromstring(string: *const c_char, length: i32) -> i32 {
    let converted = f2c_string(string, length);
    if converted.is_null() {
        return -2;
    }
    let result =
        set_output_type_from_string(core::ffi::CStr::from_ptr(converted).to_str().unwrap_or(""));
    libc::free(converted.cast());
    result
}
/// Matches C `overrideallbigtiff` (`b3dutil.c:652`).
pub unsafe fn overrideallbigtiff(value: *const i32) {
    override_all_big_tiff(*value);
}
/// Matches C `setnextoutputsize` (`b3dutil.c:687`).
pub unsafe fn setnextoutputsize(nx: *const i32, ny: *const i32, nz: *const i32, mode: *const i32) {
    set_next_output_size(*nx, *ny, *nz, *mode);
}
/// Matches C `settiffcompressiontype` (`b3dutil.c:713`).
pub unsafe fn settiffcompressiontype(index: *const i32, override_environment: *const i32) -> i32 {
    set_tiff_compression_type(*index, *override_environment)
}
/// Matches C `set4bitoutputmode` (`b3dutil.c:730`).
pub unsafe fn set4bitoutputmode(value: *const i32) {
    set_4_bit_output_mode(*value);
}
/// Matches C `setfloat16outputmode` (`b3dutil.c:759`).
pub unsafe fn setfloat16outputmode(value: *const i32, test_for_mrc: *const i32) {
    set_float_16_output_mode(*value, *test_for_mrc);
}
/// Matches C `setfloatoutputforenteredmode` (`b3dutil.c:780`).
pub unsafe fn setfloatoutputforenteredmode(mode: *const i32) -> i32 {
    set_float_output_for_entered_mode(*mode)
}
/// Matches C `write16bitmodeforfloats` (`b3dutil.c:804`).
pub fn write16bitmodeforfloats() -> i32 {
    write_16_bit_mode_for_floats()
}
/// Matches C `b3dheaderitembytes` (`b3dutil.c:1168`).
pub unsafe fn b3dheaderitembytes(flags: *mut i32, bytes: *mut i32) {
    let (count, items) = b3d_header_item_bytes();
    *flags = count;
    for index in 0..count as usize {
        *bytes.add(index) = items[index];
    }
}
/// Matches C `extraisnbytesandflags` (`b3dutil.c:1198`).
pub unsafe fn extraisnbytesandflags(nint: *const i32, nreal: *const i32) -> i32 {
    extra_is_nbytes_and_flags(*nint, *nreal)
}
/// Matches C `numberinlist` (`b3dutil.c:1227`).
pub unsafe fn numberinlist(
    number: *const i32,
    list: *const i32,
    count: *const i32,
    no_list_value: *const i32,
) -> i32 {
    number_in_list(*number, list, *count, *no_list_value)
}
/// Matches C `balancedgrouplimits` (`b3dutil.c:1246`).
pub unsafe fn balancedgrouplimits(
    total: *const i32,
    groups: *const i32,
    group: *const i32,
    start: *mut i32,
    end: *mut i32,
) {
    balanced_group_limits(*total, *groups, *group, start, end);
}
/// Matches C `wallTime` from included `coresprocsthreads.c`.
pub fn wall_time() -> f64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time);
    }
    time.tv_sec as f64 + time.tv_nsec as f64 / 1.0e9
}
/// Matches C `walltime` (`b3dutil.c:1345`).
pub fn walltime() -> f64 {
    wall_time()
}
/// Matches C `b3dmillisleep` (`b3dutil.c:1375`).
pub unsafe fn b3dmillisleep(milliseconds: *const i32) -> i32 {
    b3d_milli_sleep(*milliseconds)
}
/// Matches C `numCoresAndLogicalProcs` from included `coresprocsthreads.c` on Linux.
pub unsafe fn num_cores_and_logical_procs(physical: *mut i32, logical: *mut i32) -> i32 {
    *physical = 0;
    *logical = 0;
    let text = match std::fs::read_to_string("/proc/cpuinfo") {
        Ok(value) => value,
        Err(_) => return 1,
    };
    let mut identifiers = std::collections::BTreeSet::new();
    let mut physical_id = String::new();
    let mut core_id = String::new();
    for paragraph in text.split("\n\n") {
        for line in paragraph.lines() {
            if let Some(value) = line.strip_prefix("physical id\t: ") {
                physical_id = value.to_string();
            }
            if let Some(value) = line.strip_prefix("core id\t\t: ") {
                core_id = value.to_string();
            }
        }
        if !physical_id.is_empty() || !core_id.is_empty() {
            identifiers.insert((physical_id.clone(), core_id.clone()));
        }
    }
    *logical = text.matches("processor\t:").count() as i32;
    *physical = identifiers.len() as i32;
    if *physical == 0 {
        *physical = *logical;
    }
    if *logical == 0 { 1 } else { 0 }
}
/// Matches C `b3dCpuIsAMD` from included `coresprocsthreads.c` on Linux.
pub fn b3d_cpu_is_amd() -> i32 {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|text| if text.contains("AuthenticAMD") { 1 } else { 0 })
        .unwrap_or(0)
}
/// Matches C `numOMPthreads` (`coresprocsthreads.c:193`), the `_OPENMP` branch.
///
/// The reference build links OpenMP (`libcfshr.so` imports `GOMP_parallel`), so
/// this is the branch it compiles.  Returning 1 unconditionally — the `#else`
/// at `coresprocsthreads.c:281` — is not merely a performance difference: the
/// count selects the `balancedGroupLimits` partition and, in
/// `sliceNoiseTaperPad`, which of the 16 static `pseudoVals` seeds are used, so
/// a different count yields a different noise field.
///
/// `omp_get_num_procs()` is the number of processors available to the process;
/// `std::thread::available_parallelism` is its closest counterpart.  The
/// Apple/M1 arm is not translated: this is a Linux target.
#[unsafe(no_mangle)]
pub extern "C" fn num_omp_threads(optimal_threads: i32) -> i32 {
    static NUM_PROCS: AtomicI32 = AtomicI32::new(-1);
    static OMP_NUM_PROCS: AtomicI32 = AtomicI32::new(-1);
    let mut num_threads = optimal_threads;

    // One-time determination of the number of physical and logical cores.
    if NUM_PROCS.load(Ordering::SeqCst) < 0 {
        let omp_num_procs = std::thread::available_parallelism()
            .map(|value| value.get() as i32)
            .unwrap_or(1);
        OMP_NUM_PROCS.store(omp_num_procs, Ordering::SeqCst);
        let mut num_procs = omp_num_procs;
        let mut processor_core_count = 0_i32;
        let mut logical_processor_count = 0_i32;
        let mut physical_procs = 0_i32;
        if unsafe {
            num_cores_and_logical_procs(&mut processor_core_count, &mut logical_processor_count)
        } == 0
            && processor_core_count > 0
            && logical_processor_count == num_procs
        {
            physical_procs = processor_core_count;
        }
        if std::env::var_os("IMOD_REPORT_CORES").is_some() {
            unsafe {
                libc::printf(
                    c"core count = %d  logical processors = %d  OMP num = %d => physical processors = %d\n"
                        .as_ptr(),
                    processor_core_count,
                    logical_processor_count,
                    num_procs,
                    physical_procs,
                );
                libc::fflush(stdout);
            }
        }
        if physical_procs > 0 {
            num_procs = num_procs.min(physical_procs);
        }
        NUM_PROCS.store(num_procs, Ordering::SeqCst);
    }
    let num_procs = NUM_PROCS.load(Ordering::SeqCst);

    // Limit by number of real cores.
    num_threads = 1.max(num_procs.min(num_threads));

    // `coresprocsthreads.c:236-241` caches this in a `static`.  Here it is
    // re-read each call, a deliberate and documented deviation: the only
    // observable difference is that a change to the environment mid-process is
    // honoured rather than ignored, which no IMOD command does, and it lets a
    // test pin the count deterministically instead of inheriting the machine's
    // core count.
    let lim_threads = 0.max(
        std::env::var("OMP_NUM_THREADS")
            .ok()
            .map(|value| value.trim().parse::<i32>().unwrap_or(0))
            .unwrap_or(0),
    );
    if lim_threads > 0 {
        num_threads = lim_threads.min(num_threads);
    }

    // Same deviation as above (`coresprocsthreads.c:254-270` caches this).
    let force_threads = {
        let mut force = 0_i32;
        if let Ok(value) = std::env::var("IMOD_FORCE_OMP_THREADS") {
            if value == "ALL_CORES" {
                if num_procs > 0 {
                    force = num_procs;
                }
            } else if value == "ALL_HYPER" {
                let omp_num_procs = OMP_NUM_PROCS.load(Ordering::SeqCst);
                if omp_num_procs > 0 {
                    force = omp_num_procs;
                }
            } else {
                force = 0.max(value.trim().parse::<i32>().unwrap_or(0));
            }
        }
        force
    };
    if force_threads > 0 {
        num_threads = force_threads;
    }

    if std::env::var_os("IMOD_REPORT_CORES").is_some() {
        unsafe {
            libc::printf(
                c"numProcs %d  limThreads %d  numThreads %d\n".as_ptr(),
                num_procs,
                lim_threads,
                num_threads,
            );
            libc::fflush(stdout);
        }
    }
    num_threads
}
/// Matches C `numompthreads` (`b3dutil.c:1407`).
pub unsafe fn numompthreads(optimal_threads: *const i32) -> i32 {
    num_omp_threads(*optimal_threads)
}
/// Matches C `b3dOMPthreadNum` (`b3dutil.c:1416`) for the no-OpenMP build.
#[unsafe(no_mangle)]
pub extern "C" fn b3d_omp_thread_num() -> i32 {
    0
}
/// Matches C `b3dompthreadnum` (`b3dutil.c:1425`).
pub fn b3dompthreadnum() -> i32 {
    b3d_omp_thread_num() + 1
}
/// Matches C `b3dphysicalmemory` (`b3dutil.c:1484`).
pub fn b3dphysicalmemory() -> f64 {
    b3d_physical_memory()
}
/// Matches C `b3daddressablememory` (`b3dutil.c:1502`).
pub fn b3daddressablememory() -> f64 {
    b3d_addressable_memory()
}
/// Matches C `standardmemorylimitmb` (`b3dutil.c:1535`).
pub unsafe fn standardmemorylimitmb(half_point: *const i32) -> f64 {
    standard_memory_limit_mb(*half_point)
}
/// Matches C `addToArgVector` (`b3dutil.c:1542`), a file-static helper.
///
/// Its only call sites are inside `expandArgList`'s `#ifdef _WIN32` branch, so
/// nothing reaches it on this platform; it is translated because the
/// definition itself is not conditionally compiled.
unsafe fn add_to_arg_vector(
    arg: *const c_char,
    arg_vec: *mut *mut *mut c_char,
    vec_size: *mut i32,
    num_in_vec: *mut i32,
    pattern: *const c_char,
    num_prefix: i32,
) -> i32 {
    unsafe {
        let quantum = 8;
        if *num_in_vec >= *vec_size {
            if *vec_size != 0 {
                let grown = libc::realloc(
                    (*arg_vec).cast(),
                    (*vec_size + quantum) as usize * core::mem::size_of::<*mut c_char>(),
                );
                *arg_vec = grown.cast();
            } else {
                *arg_vec =
                    libc::malloc(quantum as usize * core::mem::size_of::<*mut c_char>()).cast();
            }
            if (*arg_vec).is_null() {
                return 1;
            }
            *vec_size += quantum;
        }
        let slot = (*arg_vec).offset(*num_in_vec as isize);
        if !pattern.is_null() && num_prefix != 0 {
            *slot = libc::malloc(num_prefix as usize + libc::strlen(arg) + 1).cast();
        } else {
            *slot = libc::strdup(arg);
        }
        if (*slot).is_null() {
            return 1;
        }
        if !pattern.is_null() && num_prefix != 0 {
            libc::strncpy(*slot, pattern, num_prefix as usize);
            libc::strcpy((*slot).offset(num_prefix as isize), arg);
        }
        *num_in_vec += 1;
        0
    }
}
/// Matches C `expandArgList` (`b3dutil.c:1579`).
///
/// The whole body is inside `#ifdef _WIN32` / `#else`.  This is the `#else`
/// branch selected on this platform (`b3dutil.c:1712-1716`), which performs no
/// expansion at all.  The Windows branch walks the argument vector with
/// `FindFirstFile`/`FindNextFile` to expand `*` and `?` wildcards, and is not
/// translated: it is unselected here and unreachable on a Unix target.
pub unsafe fn expand_arg_list(
    arguments: *const *const c_char,
    count: i32,
    new_count: *mut i32,
    allocated: *mut i32,
    no_match: *mut i32,
) -> *mut *mut c_char {
    *allocated = 0;
    *no_match = -1;
    *new_count = count;
    arguments.cast_mut().cast()
}
/// Matches C `replaceFileArgVec` (`b3dutil.c:1722`).
///
/// Unlike `expandArgList` this body is not conditionally compiled, so it is
/// translated in full.  On this platform `expandArgList` returns the original
/// vector with `ifAlloc == 0` and `noMatchInd == -1`, which makes the two error
/// paths and the replacement path unreachable; they are kept because the source
/// keeps them.
pub unsafe fn replace_file_arg_vec(
    arguments: *mut *const *const c_char,
    count: *mut i32,
    first: *mut i32,
    allocated: *mut i32,
) -> i32 {
    unsafe {
        let mut new_num = 0_i32;
        let mut no_match_ind = 0_i32;
        *allocated = 0;
        if *first >= *count {
            return 0;
        }
        let new_vec = expand_arg_list(
            (*arguments).offset(*first as isize),
            *count - *first,
            &mut new_num,
            allocated,
            &mut no_match_ind,
        );
        if new_vec.is_null() {
            b3d_error(
                stdout,
                format_args!(
                    "ERROR: {} - Allocating memory for expanded argument list\n",
                    core::ffi::CStr::from_ptr(imod_prog_name(*(*arguments))).to_string_lossy()
                ),
            );
            return -1;
        }
        if no_match_ind >= 0 {
            libc::free(new_vec.cast());
            b3d_error(
                stdout,
                format_args!(
                    "ERROR: {} - No files match entry {}\n",
                    core::ffi::CStr::from_ptr(imod_prog_name(*(*arguments))).to_string_lossy(),
                    core::ffi::CStr::from_ptr(
                        *(*arguments).offset((no_match_ind + *first) as isize)
                    )
                    .to_string_lossy()
                ),
            );
            return 1;
        }
        if *allocated != 0 {
            *arguments = new_vec.cast_const().cast();
            *count = new_num;
            *first = 0;
        }
        0
    }
}
/// Matches C `anglewithinlimits` (`b3dutil.c:1805`).
pub unsafe fn anglewithinlimits(angle: *const f32, lower: *const f32, upper: *const f32) -> f64 {
    angle_within_limits(*angle, *lower, *upper)
}
/// Matches C `getStandardGpuOptions` (`b3dutil.c:1818`) for the environment half of the source contract.
pub unsafe fn get_standard_gpu_options(
    if_gpu_by_environment: *mut i32,
    action_fail_option: *mut i32,
    action_fail_environment: *mut i32,
) -> i32 {
    let mut use_gpu = -1;
    *if_gpu_by_environment = 0;
    if let Ok(value) = std::env::var("IMOD_USE_GPU") {
        *if_gpu_by_environment = 1;
        use_gpu = value.parse().unwrap_or(0);
    }
    if let Ok(value) = std::env::var("IMOD_USE_GPU2") {
        *if_gpu_by_environment = 1;
        use_gpu = value.parse().unwrap_or(0);
    }
    if !action_fail_option.is_null() && !action_fail_environment.is_null() {
        *action_fail_option = 0;
        *action_fail_environment = 0;
    }
    use_gpu
}
/// Matches C `b3dsetlocktimeout` (`b3dutil.c:1865`).
pub unsafe fn b3dsetlocktimeout(timeout: *const f32) {
    b3d_set_lock_timeout(*timeout);
}
/// Matches C `b3dopenlockfile` (`b3dutil.c:1905`).
pub unsafe fn b3dopenlockfile(filename: *const c_char, length: i32) -> i32 {
    let string = f2c_string(filename, length);
    if string.is_null() {
        return -3;
    }
    let result = b3d_open_lock_file(string);
    libc::free(string.cast());
    result
}
/// Matches C `b3dlockfile` (`b3dutil.c:1960`).
pub unsafe fn b3dlockfile(index: *const i32) -> i32 {
    b3d_lock_file(*index)
}
/// Matches C `b3dunlockfile` (`b3dutil.c:2000`).
pub unsafe fn b3dunlockfile(index: *const i32) -> i32 {
    b3d_unlock_file(*index)
}
/// Matches C `b3dcloselockfile` (`b3dutil.c:2027`).
pub unsafe fn b3dcloselockfile(index: *const i32) -> i32 {
    b3d_close_lock_file(*index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_overrides_have_the_source_precedence() {
        override_write_bytes(0);
        assert_eq!(write_bytes_signed(), 0);
        override_write_bytes(1);
        assert_eq!(write_bytes_signed(), 1);
        set_4_bit_output_mode(4);
        assert_eq!(write_4_bit_mode_for_bytes(), 4);
        set_4_bit_output_mode(0);
    }

    #[test]
    fn origin_override_and_flag_mutation_follow_source_rules() {
        override_invert_mrc_origin(0);
        assert_eq!(invert_mrc_origin_on_output(), 0);
        override_invert_mrc_origin(4);
        assert_eq!(invert_mrc_origin_on_output(), 4);
        override_invert_mrc_origin(-1);
        let mut flags = 0b1010_u32;
        set_or_clear_flags(&mut flags, 0b0101, 1);
        assert_eq!(flags, 0b1111);
        set_or_clear_flags(&mut flags, 0b0110, 0);
        assert_eq!(flags, 0b1001);
    }

    #[test]
    fn output_type_strings_retain_source_case_and_override_rules() {
        assert_eq!(set_output_type_from_string("TIFF"), OUTPUT_TYPE_TIFF);
        assert_eq!(b3d_output_file_type(), OUTPUT_TYPE_TIFF);
        assert_eq!(set_output_type_from_string("JpEg"), -1);
        assert_eq!(b3d_output_file_type(), OUTPUT_TYPE_TIFF);
        assert_eq!(set_output_type_from_string("hdf"), OUTPUT_TYPE_HDF);
        assert_eq!(b3d_output_file_type(), OUTPUT_TYPE_HDF);
        override_output_type(-1);
    }

    #[test]
    fn error_storage_uses_the_source_buffer_and_store_flag() {
        b3d_set_store_error(1);
        b3d_error(core::ptr::null_mut(), format_args!("header {}", 42));
        assert_eq!(b3d_get_store_error(), 1);
        let error = unsafe { core::ffi::CStr::from_ptr(b3d_get_error()) };
        assert_eq!(error.to_bytes(), b"header 42");
        b3d_set_store_error(0);
    }

    #[test]
    fn extra_is_nbytes_and_flags_uses_the_serialem_table() {
        assert_eq!(extra_is_nbytes_and_flags(8, 3), 1);
        assert_eq!(extra_is_nbytes_and_flags(7, 3), 0);
        assert_eq!(extra_is_nbytes_and_flags(8, 1 << 11), 0);
    }

    #[test]
    fn data_size_for_mode_retains_source_mode_table() {
        let mut data_size = 0;
        let mut channels = 0;
        assert_eq!(data_size_for_mode(4, &mut data_size, &mut channels), 0);
        assert_eq!((data_size, channels), (4, 2));
        assert_eq!(data_size_for_mode(99, &mut data_size, &mut channels), 0);
        assert_eq!((data_size, channels), (4, 3));
        assert_eq!(data_size_for_mode(12, &mut data_size, &mut channels), -1);
    }

    #[test]
    fn read_bytes_signed_uses_source_stamp_and_minmax_rules() {
        assert_eq!(
            read_bytes_signed(IMOD_MRC_STAMP, MRC_FLAGS_SBYTES, 0, 0.0, 255.0),
            1
        );
        assert_eq!(read_bytes_signed(IMOD_MRC_STAMP, 0, 2, -5.0, 5.0), 0);
        assert_eq!(read_bytes_signed(0, 0, 0, -20.0, 100.0), 1);
        assert_eq!(read_bytes_signed(0, 0, 0, -20.0, 200.0), 0);
    }

    #[test]
    fn core_b3dutil_value_operations_follow_source() {
        let mut signed = [0_i8; 3];
        let mut bytes = [0_u8, 128, 255];
        unsafe { b3d_shift_bytes(bytes.as_mut_ptr(), signed.as_mut_ptr(), 3, 1, 1, 1) };
        assert_eq!(signed, [-128, 0, 127]);
        unsafe { b3d_shift_bytes(bytes.as_mut_ptr(), signed.as_mut_ptr(), 3, 1, -1, 1) };
        assert_eq!(bytes, [0, 128, 255]);
        let values = [4, 7, 9];
        assert_eq!(unsafe { number_in_list(7, values.as_ptr(), 3, -1) }, 1);
        assert_eq!(unsafe { number_in_list(2, values.as_ptr(), 3, -1) }, 0);
        let mut start = 0;
        let mut end = 0;
        unsafe { balanced_group_limits(10, 3, 1, &mut start, &mut end) };
        assert_eq!((start, end), (4, 6));
        assert_eq!(total_cuda_cores(3, 5, 2), 384);
        assert_eq!(angle_within_limits(-10., 0., 360.), 350.);
    }

    #[test]
    fn fortran_string_conversion_matches_blank_handling() {
        let input = b"abc   ";
        let converted = unsafe { f2c_string(input.as_ptr().cast(), input.len() as i32) };
        assert_eq!(
            unsafe { core::ffi::CStr::from_ptr(converted) }.to_bytes(),
            b"abc"
        );
        unsafe { libc::free(converted.cast()) };
        let mut output = [0_i8; 5];
        assert_eq!(
            unsafe { c2f_string(c"abc".as_ptr(), output.as_mut_ptr(), 5) },
            0
        );
        assert_eq!(
            &output,
            &[b'a' as i8, b'b' as i8, b'c' as i8, b' ' as i8, b' ' as i8]
        );
    }
}
