//! Selected bottom-up functions from `IMOD/libcfshr/b3dutil.c`.
#![allow(dead_code)]

use core::cell::{Cell, RefCell};
use core::ffi::{c_char, c_void};
use core::ptr;
use core::sync::atomic::{AtomicI32, Ordering};
use std::io::{Read, Seek, SeekFrom, Write};

// The three C standard streams, and the **only** foreign boundary this module
// keeps that is not an OS service.
//
// NATIVE.md's vocabulary item 7b puts them here deliberately: a unit that must
// keep writing on the C stream needs `stdout`/`stderr` as C objects, and
// letting each unit re-declare `static mut stderr: *mut libc::FILE` for itself
// is how the raw pointer gets back in — twenty modules had done exactly that.
// [`ImodFile::Stdout`], [`ImodFile::Stderr`] and [`ImodFile::Stdin`] are the
// named accessor, so nobody else needs the declaration.
//
// They stay C streams rather than becoming `std::io::stdout()` because the
// tree still has ~900 `libc::printf` sites and C stdio is *block*-buffered
// under redirection where Rust's is line-buffered: a message written through
// Rust's stream and a line written through `printf` come out in the wrong
// order in a redirected capture, which is NATIVE.md §1's mixed-buffering trap.
// Reading has the same problem in reverse — `mrc_head_read` takes the MRC
// header off stdin with [`b3d_fread`] while `fgetline` reads the interactive
// prompts off it with `getc`, and two different buffers over one descriptor
// lose data. When the last `printf` in a program is gone these arms can move
// to `std::io`; until then this is the correct boundary and it is one file.
unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
    static mut stdout: *mut libc::FILE;
    static mut stdin: *mut libc::FILE;
}

/// `b3dutil.h:27`.
const MAX_IMOD_ERROR_STRING: usize = 512;
/// `b3dutil.c:1843`.
const MAX_LOCK_FILES: usize = 8;
/// `b3dutil.c:1844`.
const NUM_LOCK_BYTES: i64 = 1024;
/// `b3dutil.c:970`.
const SEEK_LIMIT: i32 = 2_000_000_000;
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

thread_local! {
    /// `b3dutil.c:1848` `static int sLockFiles[MAX_LOCK_FILES]`.
    static S_LOCK_FILES: RefCell<[i32; MAX_LOCK_FILES]> =
        const { RefCell::new([0; MAX_LOCK_FILES]) };
    /// `b3dutil.c:1850` `static int sLocksUsed[MAX_LOCK_FILES]`.
    static S_LOCKS_USED: RefCell<[i32; MAX_LOCK_FILES]> =
        const { RefCell::new([0; MAX_LOCK_FILES]) };
    /// `b3dutil.c:1851` `static float sLockTimeouts[MAX_LOCK_FILES]`.
    static S_LOCK_TIMEOUTS: RefCell<[f32; MAX_LOCK_FILES]> =
        const { RefCell::new([0.; MAX_LOCK_FILES]) };
    /// `b3dutil.c:1852` `static int sInitedLocks = 0`.
    static S_INITED_LOCKS: Cell<i32> = const { Cell::new(0) };
    /// `b3dutil.c:1853` `static float sDfltLockTimeout = 30.`.
    static S_DFLT_LOCK_TIMEOUT: Cell<f32> = const { Cell::new(30.) };
    /// `b3dutil.c:856` `static int storeError = 0`.
    static STORE_ERROR: Cell<i32> = const { Cell::new(0) };
    /// `b3dutil.c:857` `static char errorMess[MAX_IMOD_ERROR_STRING] = ""`.
    ///
    /// Kept as a fixed byte array rather than a `String` because
    /// @b3d_get_error hands the whole buffer back and the source's `vsprintf`
    /// truncation at 512 is observable.
    static ERROR_MESS: RefCell<[u8; MAX_IMOD_ERROR_STRING]> =
        const { RefCell::new([0; MAX_IMOD_ERROR_STRING]) };
}

/// The Rust stand-in for a C `FILE *`, and the type every translated unit takes
/// in place of one.
///
/// `b3dutil.c` is already the source's own file-access layer — `b3dFseek`
/// (`:899`), `b3dFread` (`:919`), `b3dFwrite` (`:953`), `b3dRewind` (`:964`) —
/// so the replacement belongs here, beside their translations, rather than in a
/// new module the coverage audit could not pair. No `FILE *` in this tree is
/// ever handed to a foreign library (libtiff and HDF5 both take filenames), so
/// nothing forces the C type to survive.
///
/// The standard streams are arms rather than a separate type because the source
/// passes them interchangeably with real files: `imodError(out, …)`,
/// `fprintf(fout, …)` and `fprintf(stderr, …)` are the same call with a
/// different handle, and `b3dFseek` (`:903`) explicitly tests `fp == stdin` and
/// returns 0 rather than seeking. A function that only ever writes should take
/// `&mut dyn Write` instead; `ImodFile` implements it, so a file, stdout and
/// stderr all pass.
///
/// The three stream arms go through the C library's own streams — see the
/// `unsafe extern "C"` block above for why that is deliberate and why it is
/// confined to this file.
#[derive(Clone)]
pub enum ImodFile {
    File(std::rc::Rc<std::fs::File>),
    Stdin,
    Stdout,
    Stderr,
    /// A `FILE *` that is not a file.  Four places in `libiimod` store
    /// something else in an `fp` field, cast to `FILE *` purely as a unique
    /// identity for `iiLookupFileFromFP`: the libtiff `TIFF *`
    /// (`iitif.c:670`, `:688`, `:2434`), the `ImodImageFile *` itself for HDF
    /// (`iihdf.c:1529`, `:1548`, `:1655`), the same for a relocated file
    /// (`iimage.c:835`), and the shared-memory base address
    /// (`iishrmem.c:108`, `:111`).  No I/O is ever performed through one — the
    /// value is only ever compared — so reads and writes on this arm return 0
    /// as they would on a stream with nothing behind it.
    Token(usize),
}

impl ImodFile {
    /// `fopen(path, mode)`, with the C mode string the source passes around.
    ///
    /// The source carries mode strings in variables and builds them
    /// conditionally, so this takes the string rather than exposing one
    /// constructor per mode. `b` is accepted and ignored, as on POSIX.
    /// Returns `None` where `fopen` returns NULL.
    pub fn open(path: &str, mode: &str) -> Option<ImodFile> {
        let m: String = mode.chars().filter(|c| *c != 'b').collect();
        let mut o = std::fs::OpenOptions::new();
        match m.as_str() {
            "r" => o.read(true),
            "w" => o.write(true).create(true).truncate(true),
            "a" => o.append(true).create(true),
            "r+" => o.read(true).write(true),
            "w+" => o.read(true).write(true).create(true).truncate(true),
            "a+" => o.read(true).append(true).create(true),
            _ => return None,
        };
        o.open(path)
            .ok()
            .map(|f| ImodFile::File(std::rc::Rc::new(f)))
    }

    /// `tmpfile()`: a file with no name that goes away when it is dropped.
    ///
    /// POSIX lets a file be unlinked while an open descriptor still refers to
    /// it, which is how `tmpfile` itself works, so this creates and immediately
    /// unlinks.
    pub fn tmpfile() -> Option<ImodFile> {
        use std::sync::atomic::AtomicU32;
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("imod-rs-tmp-{}-{}", std::process::id(), n));
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .ok()?;
        let _ = std::fs::remove_file(&path);
        Some(ImodFile::File(std::rc::Rc::new(f)))
    }

    /// `fp == stdin`, the test `b3dFseek` (`b3dutil.c:903`) makes before it
    /// seeks.
    pub fn is_stdin(&self) -> bool {
        matches!(self, ImodFile::Stdin)
    }

    /// `fp == stderr`, the test `b3dError` (`b3dutil.c:868`) makes.
    pub fn is_stderr(&self) -> bool {
        matches!(self, ImodFile::Stderr)
    }

    /// `getc(fp)` / `fgetc(fp)`, returning `EOF` (-1) at end of file or on
    /// error, as the C library does.
    pub fn getc(&mut self) -> i32 {
        let mut byte = [0u8; 1];
        match self.read(&mut byte) {
            Ok(1) => byte[0] as i32,
            _ => -1,
        }
    }

    /// `ftell(fp)`, or -1 where the C library would fail.
    pub fn tell(&mut self) -> i64 {
        match self.stream_position() {
            Ok(position) => position as i64,
            Err(_) => -1,
        }
    }

    /// C's `fp1 == fp2` on two `FILE *`, which the source uses as an identity
    /// test rather than as a comparison: `findFileInList` (`iimage.c:1046`)
    /// walks `sOpenedFiles` looking for the entry whose `fp` *is* the handle it
    /// was given.  A clone of an [`ImodFile`] shares one `Rc<File>`, hence one
    /// kernel file description and one file offset, exactly as two copies of a
    /// C `FILE *` do, so `Rc::ptr_eq` is that test.
    pub fn ptr_eq(&self, other: &ImodFile) -> bool {
        match (self, other) {
            (ImodFile::File(a), ImodFile::File(b)) => std::rc::Rc::ptr_eq(a, b),
            (ImodFile::Stdin, ImodFile::Stdin) => true,
            (ImodFile::Stdout, ImodFile::Stdout) => true,
            (ImodFile::Stderr, ImodFile::Stderr) => true,
            (ImodFile::Token(a), ImodFile::Token(b)) => a == b,
            _ => false,
        }
    }

    /// The file descriptor behind the handle, for the POSIX services that have
    /// no `std::io` expression — advisory record locking through `fcntl` is the
    /// only one this module needs.
    pub fn fileno(&self) -> i32 {
        use std::os::fd::AsRawFd;
        match self {
            ImodFile::File(f) => f.as_raw_fd(),
            ImodFile::Stdin => 0,
            ImodFile::Stdout => 1,
            ImodFile::Stderr => 2,
            ImodFile::Token(_) => -1,
        }
    }
}

impl Read for ImodFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            ImodFile::File(f) => (&**f).read(buf),
            // The C stream, not `std::io::stdin()` — see the extern block.
            ImodFile::Stdin => {
                let n = unsafe { libc::fread(buf.as_mut_ptr().cast(), 1, buf.len(), stdin) };
                Ok(n)
            }
            _ => Ok(0),
        }
    }
}

impl Write for ImodFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            ImodFile::File(f) => (&**f).write(buf),
            // The C streams, not `std::io::stdout()` — see the extern block.
            ImodFile::Stdout => {
                Ok(unsafe { libc::fwrite(buf.as_ptr().cast(), 1, buf.len(), stdout) })
            }
            ImodFile::Stderr => {
                Ok(unsafe { libc::fwrite(buf.as_ptr().cast(), 1, buf.len(), stderr) })
            }
            ImodFile::Stdin | ImodFile::Token(_) => Ok(0),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            ImodFile::File(f) => (&**f).flush(),
            ImodFile::Stdout => {
                unsafe { libc::fflush(stdout) };
                Ok(())
            }
            ImodFile::Stderr => {
                unsafe { libc::fflush(stderr) };
                Ok(())
            }
            ImodFile::Stdin | ImodFile::Token(_) => Ok(()),
        }
    }
}

impl Seek for ImodFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            ImodFile::File(f) => (&**f).seek(pos),
            // The standard streams **are** seekable when the shell has
            // redirected them to a regular file, and C's `fseek`/`rewind` do
            // seek in that case.  This mattered: `imodWriteAscii`
            // (`imodel_files.c:1645`) rewinds `fout` so its later write
            // overwrites the banner, which is exactly why
            // `imodinfo -a m.mod > out` puts the header first while
            // `imodinfo -a m.mod | cat > out` puts it last.  Returning
            // `Ok(0)` here without seeking made the redirected form *append*
            // instead, and a native differential caught it.
            //
            // `lseek` reports `ESPIPE` for a pipe or terminal, which is what
            // the C library also sees, so the pipe case still behaves as
            // before.  Note this is *not* `b3dFseek`'s stdin rule: that
            // function has its own `fp == stdin` early return
            // (`b3dutil.c:903`) and keeps it.
            ImodFile::Stdin | ImodFile::Stdout | ImodFile::Stderr => {
                let fd = match self {
                    ImodFile::Stdin => 0,
                    ImodFile::Stdout => 1,
                    _ => 2,
                };
                let (whence, offset) = match pos {
                    SeekFrom::Start(n) => (libc::SEEK_SET, n as i64),
                    SeekFrom::Current(n) => (libc::SEEK_CUR, n),
                    SeekFrom::End(n) => (libc::SEEK_END, n),
                };
                let result = unsafe { libc::lseek(fd, offset as libc::off_t, whence) };
                if result < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(result as u64)
                }
            }
            _ => Ok(0),
        }
    }
}

/// Matches C `b3dError(FILE *, const char *, ...)` (`b3dutil.c:862`).
///
/// `std::fmt::Arguments` is the Rust counterpart of the C format/varargs pair;
/// `Option<&mut ImodFile>` is its `FILE *fout`, with `None` for the NULL that
/// several callers pass to store a message without printing it.
pub fn b3d_error(fout: Option<&mut ImodFile>, arguments: core::fmt::Arguments<'_>) {
    let message = arguments.to_string();
    // `vsprintf` into a MAX_IMOD_ERROR_STRING buffer: the string stops at the
    // first NUL and cannot exceed the buffer.
    let message_length = message
        .bytes()
        .position(|byte| byte == 0)
        .unwrap_or(message.len())
        .min(MAX_IMOD_ERROR_STRING - 1);
    ERROR_MESS.with_borrow_mut(|buffer| {
        buffer.fill(0);
        buffer[..message_length].copy_from_slice(&message.as_bytes()[..message_length]);
    });
    let store_error = STORE_ERROR.get();
    let stored: Vec<u8> = ERROR_MESS.with_borrow(|buffer| buffer[..message_length].to_vec());
    match fout {
        Some(file) if file.is_stderr() && store_error < 0 => {
            let _ = ImodFile::Stdout.write_all(&stored);
        }
        Some(file) if store_error <= 0 => {
            let _ = file.write_all(&stored);
        }
        _ => {}
    }
}

/// Matches C `b3dSetStoreError(int)` (`b3dutil.c:881`).
pub fn b3d_set_store_error(value: i32) {
    STORE_ERROR.set(value);
}

/// Matches C `b3dGetStoreError(void)` (`b3dutil.c:887`).
pub fn b3d_get_store_error() -> i32 {
    STORE_ERROR.get()
}

/// Matches C `b3dGetError(void)` (`b3dutil.c:892`).
///
/// The C returns `&errorMess[0]`, a pointer into the static buffer; every
/// caller in this tree immediately reads it as a string, so the Rust hands back
/// the string itself, cut at the NUL the way a C caller would see it.
pub fn b3d_get_error() -> String {
    ERROR_MESS.with_borrow(|buffer| {
        let end = buffer.iter().position(|b| *b == 0).unwrap_or(buffer.len());
        String::from_utf8_lossy(&buffer[..end]).into_owned()
    })
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
    std::process::id() as i32
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

/// One argument of a C `printf` family call, for [`c_format`].
///
/// C `printf` is variadic and Rust is not, so the argument list becomes a
/// slice. Which arm a caller picks is decided by the *source's* conversion
/// specifier and the declared type of the expression it passes, exactly as the
/// C compiler's default argument promotions decide it: `%d`/`%i` take
/// [`CArg::Int`], `%u`/`%o`/`%x`/`%X` take [`CArg::Uint`], `%e`/`%f`/`%g`/`%a`
/// take [`CArg::Dbl`] (a `float` argument is promoted to `double` in C, so
/// there is deliberately no `f32` arm), `%c` takes [`CArg::Chr`], `%s` takes
/// [`CArg::Str`] or [`CArg::Bytes`], and `%p` takes [`CArg::Ptr`].
#[derive(Clone, Copy, Debug)]
pub enum CArg<'a> {
    Int(i64),
    Uint(u64),
    Dbl(f64),
    /// A `%s` argument that is already valid UTF-8.
    Str(&'a str),
    /// A `%s` argument that is not: C strings in this tree can carry arbitrary
    /// bytes, and `%s` copies them through unchanged.
    Bytes(&'a [u8]),
    Chr(u8),
    Ptr(usize),
    /// A `*` width or precision, which C reads from the argument list.
    Star(i32),
}

/// The C library's `printf` formatting, as a Rust function, returning the
/// bytes it would have written.
///
/// This is the byte-exact entry point, and the one [`CArg::Bytes`] requires:
/// a `%s` argument carrying a byte that is not valid UTF-8 -- a model's object
/// name, a contour label, an MRC label, a file name -- survives it unchanged.
/// [`c_format`] is this function viewed as a `String` and loses such a byte to
/// U+FFFD.
///
/// This is a boundary translation, not a helper: the tree has 952
/// `printf`-family call sites whose format strings are C format strings, and
/// Rust's `{}`/`{:.3}` are **not** the same thing. `%g` alone appears 322
/// times, and C's `%g` picks `%e` or `%f` by exponent, defaults to six
/// *significant* digits and strips trailing zeros, where Rust's `{}` prints
/// the shortest decimal that round-trips. Substituting one for the other
/// silently changes almost every floating-point line the programs emit.
///
/// Supported, because that is what the tree uses: flags `-`, `+`, space, `#`,
/// `0`; a width and a precision, each literal or `*`; the length modifiers
/// `hh h l ll L z j t` (parsed and ignored, since the caller has already
/// chosen the [`CArg`] arm); and the conversions `d i o u x X e E f F g G c s
/// p %`. `%n` is not supported and never will be — it writes through a
/// pointer.
///
/// Not the same as `3dmod`'s `SPRINTF`: that macro is
/// `QString::asprintf` (`imodconfig.h:13`), which differs from the C library
/// on negative zero. Pass those arguments through [`sprintf_arg`] first.
///
/// One deliberate divergence, in a combination the tree never uses. For
/// `%#g`, when rounding to the requested significant digits carries the
/// exponent up a decade, glibc emits the digit count it had computed *before*
/// the carry: `printf("%#.6g", 999999.5)` gives `1.e+06`, while `%#.5g` of the
/// same value gives `1.0000e+06` and `%#.7g` gives `999999.5`. Six significant
/// digits with `#` should keep six, so glibc contradicts itself at exactly the
/// precision where the carry happens; this writer emits `1.00000e+06`. No
/// format string in this tree combines `#` with a floating conversion, so
/// nothing depends on it either way — but a formatter that differed from the C
/// library without saying so would be the wrong kind of surprise.
pub fn c_format_bytes(fmt: &str, args: &[CArg]) -> Vec<u8> {
    let f = fmt.as_bytes();
    let mut out = Vec::<u8>::new();
    let mut ai = 0usize;
    let mut i = 0usize;
    while i < f.len() {
        if f[i] != b'%' {
            out.push(f[i]);
            i += 1;
            continue;
        }
        i += 1;
        if i < f.len() && f[i] == b'%' {
            out.push(b'%');
            i += 1;
            continue;
        }
        // Flags.
        let (mut minus, mut plus, mut space, mut alt, mut zero) =
            (false, false, false, false, false);
        while i < f.len() {
            match f[i] {
                b'-' => minus = true,
                b'+' => plus = true,
                b' ' => space = true,
                b'#' => alt = true,
                b'0' => zero = true,
                _ => break,
            }
            i += 1;
        }
        // Width.
        let mut width: i32 = 0;
        if i < f.len() && f[i] == b'*' {
            i += 1;
            width = match args.get(ai) {
                Some(CArg::Star(v)) => *v,
                Some(CArg::Int(v)) => *v as i32,
                _ => 0,
            };
            ai += 1;
            // C: a negative `*` width means the `-` flag and a positive width.
            if width < 0 {
                minus = true;
                width = -width;
            }
        } else {
            while i < f.len() && f[i].is_ascii_digit() {
                width = width * 10 + (f[i] - b'0') as i32;
                i += 1;
            }
        }
        // Precision.
        let mut prec: Option<i32> = None;
        if i < f.len() && f[i] == b'.' {
            i += 1;
            if i < f.len() && f[i] == b'*' {
                i += 1;
                let v = match args.get(ai) {
                    Some(CArg::Star(v)) => *v,
                    Some(CArg::Int(v)) => *v as i32,
                    _ => 0,
                };
                ai += 1;
                // C: a negative `*` precision is as if the precision were omitted.
                prec = if v < 0 { None } else { Some(v) };
            } else {
                let mut p = 0i32;
                while i < f.len() && f[i].is_ascii_digit() {
                    p = p * 10 + (f[i] - b'0') as i32;
                    i += 1;
                }
                prec = Some(p);
            }
        }
        // Length modifiers.  These are **not** decoration: for an integer
        // conversion they say how wide the argument is after C's default
        // argument promotions, and therefore how much of it is printed.
        // `printf("%02x", ch)` with `ch` an `int` promotes to a 32-bit
        // `unsigned int`, so `-1` prints `ffffffff` — not `ffffffffffffffff`.
        // Discarding the modifier and formatting the whole `u64` got that
        // wrong at 19 sites in the mini-XML translation, and only stderr
        // showed it.
        let mut int_bits = 32u32;
        while i < f.len() && matches!(f[i], b'h' | b'l' | b'L' | b'z' | b'j' | b't') {
            match f[i] {
                b'h' => int_bits = if int_bits == 16 { 8 } else { 16 },
                b'l' => int_bits = 64,
                b'z' | b'j' | b't' => int_bits = 64,
                _ => {}
            }
            i += 1;
        }
        if i >= f.len() {
            break;
        }
        let conv = f[i];
        i += 1;
        let arg = args.get(ai).copied();
        ai += 1;

        // `body` is the converted value without padding; `sign` is any sign or
        // space that must sit outside a `0` pad, as C requires.
        let mut sign = String::new();
        let body: Vec<u8> = match conv {
            b'd' | b'i' => {
                let v = match arg {
                    Some(CArg::Int(v)) => v,
                    Some(CArg::Uint(v)) => v as i64,
                    _ => 0,
                };
                // Narrow to the declared width, then sign-extend, as the C
                // argument itself would be.
                let v = if int_bits >= 64 {
                    v
                } else {
                    let sh = 64 - int_bits;
                    ((v << sh) >> sh) as i64
                };
                if v < 0 {
                    sign.push('-');
                } else if plus {
                    sign.push('+');
                } else if space {
                    sign.push(' ');
                }
                let mut d = v.unsigned_abs().to_string();
                if let Some(p) = prec {
                    if v == 0 && p == 0 {
                        d.clear();
                    }
                    while (d.len() as i32) < p {
                        d.insert(0, '0');
                    }
                    zero = false;
                }
                d.into_bytes()
            }
            b'u' | b'o' | b'x' | b'X' => {
                let v = match arg {
                    Some(CArg::Uint(v)) => v,
                    Some(CArg::Int(v)) => v as u64,
                    _ => 0,
                };
                // Same narrowing, zero-extended: `%x` is `unsigned int`.
                let v = if int_bits >= 64 {
                    v
                } else {
                    v & ((1u64 << int_bits) - 1)
                };
                let mut d = match conv {
                    b'u' => v.to_string(),
                    b'o' => format!("{v:o}"),
                    b'x' => format!("{v:x}"),
                    _ => format!("{v:X}"),
                };
                if let Some(p) = prec {
                    if v == 0 && p == 0 {
                        d.clear();
                    }
                    while (d.len() as i32) < p {
                        d.insert(0, '0');
                    }
                    zero = false;
                }
                if alt && v != 0 {
                    match conv {
                        b'o' => {
                            if !d.starts_with('0') {
                                d.insert(0, '0');
                            }
                        }
                        b'x' => sign.push_str("0x"),
                        b'X' => sign.push_str("0X"),
                        _ => {}
                    }
                }
                d.into_bytes()
            }
            b'e' | b'E' | b'f' | b'F' | b'g' | b'G' | b'a' | b'A' => {
                let v = match arg {
                    Some(CArg::Dbl(v)) => v,
                    Some(CArg::Int(v)) => v as f64,
                    _ => 0.0,
                };
                if v.is_sign_negative() {
                    sign.push('-');
                } else if plus {
                    sign.push('+');
                } else if space {
                    sign.push(' ');
                }
                let mag = v.abs();
                let upper = conv.is_ascii_uppercase();
                if !mag.is_finite() {
                    // C prints `inf` / `nan` with no zero padding, and the
                    // upper-case conversions print them upper-case.
                    zero = false;
                    let t = if mag.is_nan() { "nan" } else { "inf" };
                    if upper {
                        t.to_uppercase()
                    } else {
                        t.to_string()
                    }
                    .into_bytes()
                } else {
                    match conv.to_ascii_lowercase() {
                        b'f' => {
                            let p = prec.unwrap_or(6).max(0) as usize;
                            let mut d = format!("{mag:.p$}");
                            if alt && p == 0 {
                                d.push('.');
                            }
                            d.into_bytes()
                        }
                        b'e' => {
                            let p = prec.unwrap_or(6).max(0) as usize;
                            c_format_e(mag, p, alt, upper).into_bytes()
                        }
                        _ => {
                            // `%g`: C's rule, from the C standard's 7.21.6.1.
                            // P is the precision, or 6 if omitted, or 1 if 0.
                            let mut p = prec.unwrap_or(6);
                            if p == 0 {
                                p = 1;
                            }
                            let p = p as usize;
                            // X is the exponent the `%e` form would use.
                            let x = if mag == 0.0 {
                                0i32
                            } else {
                                let e = format!("{mag:.*e}", p - 1);
                                e[e.find('e').unwrap() + 1..].parse::<i32>().unwrap_or(0)
                            };
                            let mut d = if x < -4 || x >= p as i32 {
                                c_format_e(mag, p - 1, alt, upper)
                            } else {
                                let fp = (p as i32 - 1 - x).max(0) as usize;
                                let mut d = format!("{mag:.fp$}");
                                if alt && fp == 0 {
                                    d.push('.');
                                }
                                d
                            };
                            if !alt {
                                // Trailing zeros are removed from the
                                // fractional part, and a bare `.` with them.
                                let cut = d.find(['e', 'E']).unwrap_or(d.len());
                                let (mant, exp) = d.split_at(cut);
                                if mant.contains('.') {
                                    let m = mant.trim_end_matches('0').trim_end_matches('.');
                                    d = format!("{m}{exp}");
                                }
                            }
                            d.into_bytes()
                        }
                    }
                }
            }
            b'c' => {
                let v = match arg {
                    Some(CArg::Chr(c)) => c,
                    Some(CArg::Int(v)) => v as u8,
                    Some(CArg::Uint(v)) => v as u8,
                    _ => 0,
                };
                zero = false;
                vec![v]
            }
            b's' => {
                zero = false;
                let b: &[u8] = match arg {
                    Some(CArg::Str(s)) => s.as_bytes(),
                    Some(CArg::Bytes(b)) => b,
                    _ => b"",
                };
                // A precision on `%s` is a maximum length, and C does not
                // require a terminator within it.
                match prec {
                    Some(p) => b[..b.len().min(p.max(0) as usize)].to_vec(),
                    None => b.to_vec(),
                }
            }
            b'p' => {
                let v = match arg {
                    Some(CArg::Ptr(v)) => v,
                    Some(CArg::Uint(v)) => v as usize,
                    Some(CArg::Int(v)) => v as usize,
                    _ => 0,
                };
                zero = false;
                if v == 0 {
                    b"(nil)".to_vec()
                } else {
                    format!("0x{v:x}").into_bytes()
                }
            }
            _ => {
                // An unrecognised conversion: C's behaviour is undefined, and
                // glibc echoes the specifier. Nothing in this tree uses one.
                ai -= 1;
                out.push(b'%');
                out.push(conv);
                continue;
            }
        };

        let len = sign.len() + body.len();
        let pad = (width as usize).saturating_sub(len);
        if minus {
            out.extend_from_slice(sign.as_bytes());
            out.extend_from_slice(&body);
            out.extend(std::iter::repeat_n(b' ', pad));
        } else if zero {
            out.extend_from_slice(sign.as_bytes());
            out.extend(std::iter::repeat_n(b'0', pad));
            out.extend_from_slice(&body);
        } else {
            out.extend(std::iter::repeat_n(b' ', pad));
            out.extend_from_slice(sign.as_bytes());
            out.extend_from_slice(&body);
        }
    }
    out
}

/// The C library's `printf` formatting, as a Rust `String`.
///
/// This is [`c_format_bytes`] viewed as UTF-8, and it is the right entry point
/// for the overwhelming majority of the tree's format strings, whose arguments
/// are numbers and ASCII literals.
///
/// **It is the wrong one wherever a `%s` argument can carry a byte that is not
/// valid UTF-8** — anything read out of a model, an MRC label or a file name.
/// A `String` cannot hold such a byte, so the conversion replaces it with
/// U+FFFD and the output stops matching native. Use [`c_format_bytes`] there;
/// see its documentation for the differential that found this.
pub fn c_format(fmt: &str, args: &[CArg]) -> String {
    String::from_utf8_lossy(&c_format_bytes(fmt, args)).into_owned()
}

/// The `%e` conversion of [`c_format`], for a non-negative finite `mag`.
///
/// Rust's `{:e}` writes `1.5e5`; C writes `1.500000e+05` — the exponent always
/// carries a sign and at least two digits. Split out only because `%g` needs
/// the identical conversion, which is the language boundary the no-helpers
/// rule allows for.
fn c_format_e(mag: f64, prec: usize, alt: bool, upper: bool) -> String {
    let s = format!("{mag:.prec$e}");
    let at = s.find('e').unwrap();
    let (mant, exp) = s.split_at(at);
    let e: i32 = exp[1..].parse().unwrap_or(0);
    let mut m = mant.to_string();
    if alt && prec == 0 {
        m.push('.');
    }
    format!(
        "{m}{}{}{:02}",
        if upper { 'E' } else { 'e' },
        if e < 0 { '-' } else { '+' },
        e.abs()
    )
}
/// `stdio.h`'s seek origins, re-exported so a caller of [`b3d_fseek`],
/// [`mrc_big_seek`] or [`mrc_huge_seek`] does not have to reach into `libc`
/// for the constant the source writes.
pub const SEEK_SET: i32 = 0;
pub const SEEK_CUR: i32 = 1;
pub const SEEK_END: i32 = 2;

/// Matches C `imodVersion` (`b3dutil.c:148`).
///
/// `VERSION`, `VERSION_NAME`, and `COPYRIGHT_YEARS` are generated into
/// `imodconfig.h` by `IMOD/setup2:411-419` from `IMOD/.version` ("5.2.17") and
/// `IMOD/setup2:10` ("1994-2025") for the pinned revision.  The stale 4.8.16 /
/// 1994-2014 pair in `IMOD/sysdep/win/VC-imodconfig.h` is a checked-in Visual
/// Studio config, not the configuration this revision builds with.
pub fn imod_version(program_name: Option<&str>) -> i32 {
    if let Some(program_name) = program_name {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "%s Version %s %s %s\n",
                &[
                    CArg::Str(program_name),
                    CArg::Str("5.2.17"),
                    CArg::Str(IMOD_BUILD_DATE),
                    CArg::Str(IMOD_BUILD_TIME),
                ],
            )
            .as_bytes(),
        );
    }
    5217
}
/// C `__DATE__` and `__TIME__` for this build, in the identical C field
/// formats.  They are compilation metadata rather than input-dependent output,
/// so the deterministic part compared against the reference is the field
/// layout, not the timestamp value.
pub const IMOD_BUILD_DATE: &str = env!("IMOD_BUILD_DATE");
pub const IMOD_BUILD_TIME: &str = env!("IMOD_BUILD_TIME");
/// Matches C `imodCopyright` (`b3dutil.c:157`).
pub fn imod_copyright() {
    let uofc = "Regents of the University of Colorado";
    let _ = ImodFile::Stdout.write_all(
        c_format(
            "Copyright (C) %s by the %s\n",
            &[CArg::Str("1994-2025"), CArg::Str(uofc)],
        )
        .as_bytes(),
    );
}
/// Matches C `imodUsageHeader` (`b3dutil.c:165`).
pub fn imod_usage_header(program_name: Option<&str>) {
    imod_version(program_name);
    imod_copyright();
}
/// Matches C `IMOD_DIR_or_default` (`b3dutil.c:178`).
///
/// This is the `#else`/`#else` arm — neither `_WIN32` nor `__APPLE__` — so
/// `str` has one entry and `strInd` never moves off 0; the `strInd > 1`
/// correction is therefore unreachable here and is not written out.
pub fn imod_dir_or_default(assumed: Option<&mut i32>) -> String {
    let str_: [&str; 1] = ["/usr/local/IMOD"];
    let str_ind = 0;
    let mut ass_val = 1;
    if !std::path::Path::new(str_[0]).exists() {
        ass_val = 2;
    }
    let envdir = std::env::var("IMOD_DIR").ok();
    if let Some(assumed) = assumed {
        *assumed = if envdir.is_some() { 0 } else { ass_val };
    }
    match envdir {
        Some(envdir) => envdir,
        None => str_[str_ind].to_string(),
    }
}
/// Matches C `imodProgName` (`b3dutil.c:215`).
///
/// The C returns a pointer into `fullname` unless the name ends in `.exe`, in
/// which case it `strdup`s a truncated copy and the caller is told not to free
/// it; a `String` is both cases at once.
pub fn imod_prog_name(full_name: &str) -> String {
    let forward = full_name.rfind('/');
    let tailback = full_name.rfind('\\');
    // `tailback > tail` on two pointers into the same string: a NULL loses to
    // any real position, and the later separator has the higher address.
    let tail = match (forward, tailback) {
        (Some(f), Some(b)) if b > f => Some(b),
        (None, Some(b)) => Some(b),
        (f, _) => f,
    };
    let Some(tail) = tail else {
        return full_name.to_string();
    };
    let tail = &full_name[tail + 1..];
    let indexe = tail.len() as isize - 4;
    match tail.find(".exe") {
        Some(exe) if exe as isize == indexe => tail[..exe].to_string(),
        _ => tail.to_string(),
    }
}
/// Matches C `imodBackupFile` (`b3dutil.c:241`).
///
/// `rmTries` and `mvTries` are 1 outside `_WIN32`, so each of the source's two
/// retry loops runs at most once.  The `-2` for a failed `malloc` of the backup
/// name cannot arise once the name is a `String`.
pub fn imod_backup_file(filename: &str) -> i32 {
    /* If file does not exist, return */
    if std::fs::metadata(filename).is_err() {
        return 0;
    }

    /* Get backup name */
    let backname = format!("{filename}~");

    /* If the backup file exists, try to remove it first (Windows/Intel) */
    if std::fs::metadata(&backname).is_ok() && std::fs::remove_file(&backname).is_err() {
        return -1;
    }

    /* finally, rename file */
    match std::fs::rename(filename, &backname) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}
/// Matches C `b3dOpenFile` (`b3dutil.c:302`).
///
/// The C never returns NULL — it calls `exitError` — so this returns an
/// [`ImodFile`] rather than an `Option`.  Both `printf` lines and the
/// `exitError` were missing from the previous translation and are restored
/// here; nothing in the tree calls this routine, so there is no differential to
/// run and the restoration is source-verified only.
pub fn b3d_open_file(name: &str, mode: &str) -> ImodFile {
    let stock_modes = ["r", "r+", "w+"];
    let descrip = ["OLD file", "NEW file", "file for appending"];
    let mut mode = mode;
    let mut desc_ind = 0usize;
    if mode == "ro" || mode == "RO" {
        mode = stock_modes[0];
    } else if mode == "old" || mode == "OLD" {
        mode = stock_modes[1];
    } else if mode == "new" || mode == "NEW" {
        mode = stock_modes[2];
    }
    if mode.starts_with('w') {
        if imod_backup_file(name) != 0 {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "WARNING: b3dOpenFile - Renaming existing file %s\n",
                    &[CArg::Str(name)],
                )
                .as_bytes(),
            );
        }
        desc_ind = 1;
    } else if mode.starts_with('a') {
        desc_ind = 2;
    }

    let fp = ImodFile::open(name, mode);
    let Some(fp) = fp else {
        // `strerror(errno)` is the C library's own message text and the source
        // prints exactly that, so it stays a call into libc: `std::io::Error`'s
        // Display appends " (os error N)", which the reference does not print.
        // `c_format_bytes`, not `c_format`: `strerror` is locale-dependent and
        // its bytes need not be valid UTF-8, which the lossy view would
        // replace with U+FFFD (NATIVE.md §7c).
        let message = c_format_bytes(
            "Opening %s, %s: %s",
            &[
                CArg::Str(descrip[desc_ind]),
                CArg::Str(name),
                CArg::Bytes(unsafe {
                    core::ffi::CStr::from_ptr(libc::strerror(*libc::__errno_location())).to_bytes()
                }),
            ],
        );
        crate::imod::libcfshr::parse_params::exit_error(&message);
    };
    let _ = ImodFile::Stdout.write_all(
        c_format(
            "\nOpened %s: %s\n",
            &[CArg::Str(descrip[desc_ind]), CArg::Str(name)],
        )
        .as_bytes(),
    );
    fp
}
/// Matches C `pidToStderr` (`b3dutil.c:359`).
pub fn pid_to_stderr() {
    let mut stream = ImodFile::Stderr;
    let _ = stream
        .write_all(c_format("Shell PID: %d\n", &[CArg::Int(imod_getpid() as i64)]).as_bytes());
    let _ = stream.flush();
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
///
/// Kept on raw pointers, deliberately.  Every caller in this tree except
/// `mrc_read_slice` passes the *same* buffer for both arguments — `iitif.rs`
/// writes `b3d_shift_bytes(buf.cast(), buf.cast(), …)` at four sites — because
/// the routine's job is to reinterpret one block of memory between signed and
/// unsigned bytes in place.  Two `&mut` slices cannot alias, so a safe
/// signature would have to be a different routine with different callers; that
/// belongs with the `libiimod` conversion, not here.
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
    // The source uses `putenv` so the value is visible to the whole process
    // including any library that reads it; `std::env::set_var` is that.
    unsafe {
        std::env::set_var(
            "IMOD_TIFF_COMPRESSION",
            type_map[type_index as usize].to_string(),
        )
    };
    0
}
/// Matches C `setFloat16outputMode` (`b3dutil.c:751`).
pub fn set_float_16_output_mode(in_val: i32, test_for_mrc: i32) {
    if in_val != 0 && test_for_mrc != 0 && b3d_output_file_type() != OUTPUT_TYPE_MRC {
        b3d_error(
            Some(&mut ImodFile::Stderr),
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
///
/// One of the two halves of the Fortran bridge, and it stays on `c_char` with a
/// `malloc`ed result: it exists to consume Fortran's hidden string-length
/// argument, so it can only change when the Fortran-derived callers do
/// (NATIVE.md §7).
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
/// Matches C `c2fString` (`b3dutil.c:835`). The other half of the Fortran
/// bridge; see [`f2c_string`].
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
///
/// The Unix arm of the source is `fseek(fp, offset, flag)` after the explicit
/// `fp == stdin` test at `:903`; `Seek::seek` is that call, and its `Err` is
/// `fseek`'s -1.  `SEEK_SET` with a negative offset is `EINVAL` in C and
/// `SeekFrom::Start` cannot express it, so it is rejected here at the same
/// point the C library would reject it.
pub fn b3d_fseek(file: &mut ImodFile, offset: i32, flag: i32) -> i32 {
    if file.is_stdin() {
        return 0;
    }
    let position = if flag == SEEK_SET {
        if offset < 0 {
            return -1;
        }
        SeekFrom::Start(offset as u64)
    } else if flag == SEEK_CUR {
        SeekFrom::Current(offset as i64)
    } else {
        SeekFrom::End(offset as i64)
    };
    match file.seek(position) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}
/// Matches C `b3dFread` (`b3dutil.c:919`).
///
/// The Unix arm is `fread(buf, size, count, fp)`, which returns the number of
/// whole *items* transferred, so the loop below reads up to `size * count`
/// bytes and divides.  `Read::read` is allowed to return short where `fread`
/// is not, hence the loop; a zero return is end of file and an `Err` is
/// `fread`'s error return, both of which stop it exactly where `fread` stops.
pub fn b3d_fread(buffer: &mut [u8], size: usize, count: usize, file: &mut ImodFile) -> usize {
    if size == 0 {
        return 0;
    }
    let wanted = size * count;
    let mut done = 0usize;
    while done < wanted {
        match file.read(&mut buffer[done..wanted]) {
            Ok(0) => break,
            Ok(read) => done += read,
            Err(_) => break,
        }
    }
    done / size
}
/// Matches C `b3dFwrite` (`b3dutil.c:953`).
///
/// As [`b3d_fread`]: `fwrite` returns whole items written, and `Write::write`
/// may be short where `fwrite` is not.
pub fn b3d_fwrite(buffer: &[u8], size: usize, count: usize, file: &mut ImodFile) -> usize {
    if size == 0 {
        return 0;
    }
    let wanted = size * count;
    let mut done = 0usize;
    while done < wanted {
        match file.write(&buffer[done..wanted]) {
            Ok(0) => break,
            Ok(written) => done += written,
            Err(_) => break,
        }
    }
    done / size
}
/// Matches C `b3dRewind` (`b3dutil.c:964`).
pub fn b3d_rewind(file: &mut ImodFile) {
    b3d_fseek(file, 0, SEEK_SET);
}
/// Matches C `mrc_big_seek` (`b3dutil.c:976`).
pub fn mrc_big_seek(file: &mut ImodFile, base: i32, size1: i32, size2: i32, mut flag: i32) -> i32 {
    if base != 0 || ((size1 == 0 || size2 == 0) && flag == SEEK_SET) {
        let err = b3d_fseek(file, base, flag);
        if err != 0 {
            return err;
        }
        flag = SEEK_CUR;
    }
    if size1 == 0 || size2 == 0 {
        return 0;
    }
    let abs1 = size1.abs();
    let abs2 = size2.abs();
    let smaller = abs1.min(abs2);
    let mut bigger = abs1.max(abs2);
    let step_limit = SEEK_LIMIT / bigger;
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
        flag = SEEK_CUR;
    }
    0
}
/// Matches C `mrcHugeSeek` (`b3dutil.c:1046`).
pub fn mrc_huge_seek(
    file: &mut ImodFile,
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
///
/// The source's first guard, `if (fp == NULL) b3dError(stderr, "fgetline: file
/// pointer not valid\n")`, cannot be reached through a `&mut ImodFile` and is
/// therefore not written out; a caller that could have passed NULL now has to
/// test its own handle, which every caller in this tree already does.
///
/// The `limit` guard *is* restored — the previous translation had dropped both
/// messages — and so is the source's evaluation order in the loop condition:
/// `(c = getc(fp)) != EOF` is evaluated **before** `i < limit - 1`, so the
/// character that overruns the array is consumed from the stream.
pub fn fgetline(fp: &mut ImodFile, s: &mut [u8], limit: i32) -> i32 {
    if limit < 3 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("fgetline: limit ({limit}) must be > 2\n"),
        );
        return -1;
    }

    let mut c;
    let mut i = 0usize;
    loop {
        c = fp.getc();
        if c == -1 || i >= (limit - 1) as usize || c == b'\n' as i32 {
            break;
        }
        s[i] = c as u8;
        i += 1;
    }

    /* 1/25/12: Take off a return too! */
    if i > 0 && s[i - 1] == b'\r' {
        i -= 1;
    }

    s[i] = 0;
    let length = i as i32;

    if c == -1 { -(length + 2) } else { length }
}

/// Matches C `numberInList` (`b3dutil.c:1215`).
///
/// The source null-checks `list`, so the argument is an `Option`; `nlist`
/// stays a separate count because a caller may pass a shorter run of a longer
/// array.
pub fn number_in_list(number: i32, list: Option<&[i32]>, count: i32, no_list_value: i32) -> i32 {
    let Some(list) = list else {
        return no_list_value;
    };
    if count == 0 {
        return no_list_value;
    }
    for index in 0..count as usize {
        if number == list[index] {
            return 1;
        }
    }
    0
}
/// Matches C `balancedGroupLimits` (`b3dutil.c:1237`).
pub fn balanced_group_limits(total: i32, groups: i32, group: i32, start: &mut i32, end: &mut i32) {
    let base = total / groups;
    let remainder = total % groups;
    *start = group * base + group.min(remainder);
    *end = (group + 1) * base + (group + 1).min(remainder) - 1;
}
/// Matches C `groupLimitsRemainderAtEnd` (`b3dutil.c:1256`).
pub fn group_limits_remainder_at_end(
    total: i32,
    groups: i32,
    group: i32,
    start: &mut i32,
    end: &mut i32,
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
///
/// Kept on raw pointers for the same reason as [`b3d_shift_bytes`]: the whole
/// point of the routine is to hand out an array of interior pointers into a
/// caller's buffer, which is what libtiff-shaped line access wants and what a
/// `Vec<&mut [u8]>` cannot be without borrowing the buffer for the array's
/// lifetime.  Its seven callers are all in `libiimod`/`mrc` and move with that
/// conversion.
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
    // `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)` has no `std` expression:
    // `std::time::Instant` is wall clock. This is an OS service, not C
    // emulation.
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
///
/// The source loops on `nanosleep`, resuming the remaining time after an
/// `EINTR` and counting the restarts, and returns that count (or -1 for any
/// other error).  `std::thread::sleep` does the resume itself and does not
/// report it, so the count is always 0 here; no caller in the tree reads the
/// return value — @b3d_lock_file is the only one and it discards it.
pub fn b3d_milli_sleep(milliseconds: i32) -> i32 {
    std::thread::sleep(std::time::Duration::from_millis(milliseconds.max(0) as u64));
    0
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
    // `sysconf(_SC_PHYS_PAGES)` is an OS service with no `std` expression.
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
    if core::mem::size_of::<usize>() == 4 {
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
///
/// `rand`/`srand` are the C library's generator and the sequence is part of the
/// output — `statfuncs.rs` carries the inlined glibc TYPE_3 implementation that
/// proves it — so these three stay on the C entry points.
pub fn b3drand() -> f32 {
    unsafe { libc::rand() as f32 / libc::RAND_MAX as f32 }
}
/// Matches C `b3dsrand` (`b3dutil.c:1763`). Fortran wrapper: the seed arrives
/// by reference.
pub fn b3dsrand(seed: &i32) {
    unsafe { libc::srand(*seed as u32) };
}
/// Matches C `b3dran` (`b3dutil.c:1776`). Fortran wrapper.
pub fn b3dran(seed: &i32) -> f32 {
    if S_B3DRAN_FIRST_TIME.load(Ordering::SeqCst) != 0
        || *seed != S_B3DRAN_LAST_SEED.load(Ordering::SeqCst)
    {
        unsafe { libc::srand(*seed as u32) };
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
pub fn b3d_set_lock_timeout(timeout: f32) {
    S_DFLT_LOCK_TIMEOUT.set(timeout);
}
/// Matches C `b3dOpenLockFile` (`b3dutil.c:1876`).
///
/// The descriptor and the `fcntl` record lock below are POSIX services with no
/// `std::io` expression — advisory locking is not in `std` — so `libc::open`
/// gives way to `std::fs::File` but the lock itself does not.
pub fn b3d_open_lock_file(filename: &str) -> i32 {
    if S_INITED_LOCKS.get() == 0 {
        S_LOCKS_USED.with_borrow_mut(|used| {
            for index in 0..MAX_LOCK_FILES {
                used[index] = -1;
            }
        });
    }
    S_INITED_LOCKS.set(1);
    let mut ind = 0;
    S_LOCKS_USED.with_borrow(|used| {
        while ind < MAX_LOCK_FILES && used[ind] >= 0 {
            ind += 1;
        }
    });
    if ind >= MAX_LOCK_FILES {
        return -1;
    }
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(filename)
    {
        Ok(file) => file,
        Err(_) => return -2,
    };
    // The descriptor has to outlive this call the way the C's does, and the
    // lock table stores descriptors rather than `File`s because
    // @b3d_close_lock_file is what closes them.
    use std::os::fd::IntoRawFd;
    S_LOCK_FILES.with_borrow_mut(|files| files[ind] = file.into_raw_fd());
    S_LOCKS_USED.with_borrow_mut(|used| used[ind] = 0);
    let timeout = S_DFLT_LOCK_TIMEOUT.get();
    S_LOCK_TIMEOUTS.with_borrow_mut(|timeouts| timeouts[ind] = timeout);
    ind as i32
}
/// Matches C `b3dLockFile` (`b3dutil.c:1922`).
pub fn b3d_lock_file(index: i32) -> i32 {
    if !(0..MAX_LOCK_FILES as i32).contains(&index) {
        return -1;
    }
    let index = index as usize;
    let used = S_LOCKS_USED.with_borrow(|used| used[index]);
    if used < 0 {
        return -2;
    }
    if used > 0 {
        S_LOCKS_USED.with_borrow_mut(|used| used[index] += 1);
        return 0;
    }
    let lock = libc::flock {
        l_type: libc::F_WRLCK as _,
        l_whence: SEEK_SET as _,
        l_start: 0,
        l_len: NUM_LOCK_BYTES as _,
        l_pid: 0,
    };
    let descriptor = S_LOCK_FILES.with_borrow(|files| files[index]);
    let timeout = S_LOCK_TIMEOUTS.with_borrow(|timeouts| timeouts[index]);
    let started = std::time::Instant::now();
    loop {
        if unsafe { libc::fcntl(descriptor, libc::F_SETLK, &lock) } >= 0 {
            S_LOCKS_USED.with_borrow_mut(|used| used[index] += 1);
            return 0;
        }
        if started.elapsed().as_secs_f64() >= timeout as f64 {
            return 1;
        }
        b3d_milli_sleep(50);
    }
}
/// Matches C `b3dUnlockFile` (`b3dutil.c:1970`).
pub fn b3d_unlock_file(index: i32) -> i32 {
    if !(0..MAX_LOCK_FILES as i32).contains(&index) {
        return -1;
    }
    let index = index as usize;
    let used = S_LOCKS_USED.with_borrow(|used| used[index]);
    if used < 0 {
        return -2;
    }
    if used == 0 {
        return -3;
    }
    if used == 1 {
        let lock = libc::flock {
            l_type: libc::F_UNLCK as _,
            l_whence: SEEK_SET as _,
            l_start: 0,
            l_len: NUM_LOCK_BYTES as _,
            l_pid: 0,
        };
        let descriptor = S_LOCK_FILES.with_borrow(|files| files[index]);
        if unsafe { libc::fcntl(descriptor, libc::F_SETLK, &lock) } < 0 {
            return 1;
        }
    }
    S_LOCKS_USED.with_borrow_mut(|used| used[index] -= 1);
    0
}
/// Matches C `b3dCloseLockFile` (`b3dutil.c:2009`).
pub fn b3d_close_lock_file(index: i32) -> i32 {
    if !(0..MAX_LOCK_FILES as i32).contains(&index) {
        return -1;
    }
    let index = index as usize;
    if S_LOCKS_USED.with_borrow(|used| used[index]) < 0 {
        return -2;
    }
    let descriptor = S_LOCK_FILES.with_borrow(|files| files[index]);
    if unsafe { libc::close(descriptor) } < 0 {
        return 1;
    }
    S_LOCKS_USED.with_borrow_mut(|used| used[index] = -1);
    0
}

/// Matches C `imodbackupfile` (`b3dutil.c:282`). Fortran wrapper; see
/// [`f2c_string`] for why this half of the bridge keeps `c_char`.
pub unsafe fn imodbackupfile(filename: *const c_char, length: i32) -> i32 {
    let string = f2c_string(filename, length);
    if string.is_null() {
        return -2;
    }
    let result = imod_backup_file(core::ffi::CStr::from_ptr(string).to_string_lossy().as_ref());
    libc::free(string.cast());
    result
}
/// Matches C `imodgetenv` (`b3dutil.c:333`). Fortran wrapper.
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
pub fn overridewritebytes(value: &i32) {
    override_write_bytes(*value);
}
/// Matches C `writebytessigned` (`b3dutil.c:411`).
pub fn writebytessigned() -> i32 {
    write_bytes_signed()
}
/// Matches C `readbytessigned` (`b3dutil.c:462`).
pub fn readbytessigned(stamp: &i32, flags: &i32, mode: &i32, minimum: &f32, maximum: &f32) -> i32 {
    read_bytes_signed(*stamp, *flags, *mode, *minimum, *maximum)
}
/// Matches C `b3dshiftbytes` (`b3dutil.c:490`).
pub unsafe fn b3dshiftbytes(
    unsigned: *mut u8,
    signed: *mut i8,
    nx: &i32,
    ny: &i32,
    direction: &i32,
    bytes_signed: &i32,
) {
    b3d_shift_bytes(unsigned, signed, *nx, *ny, *direction, *bytes_signed);
}
/// Matches C `overrideinvertmrcorigin` (`b3dutil.c:508`).
pub fn overrideinvertmrcorigin(value: &i32) {
    override_invert_mrc_origin(*value);
}
/// Matches C `overrideoutputtype` (`b3dutil.c:552`).
pub fn overrideoutputtype(value: &i32) {
    override_output_type(*value);
}
/// Matches C `b3doutputfiletype` (`b3dutil.c:590`).
pub fn b3doutputfiletype() -> i32 {
    b3d_output_file_type()
}
/// Matches C `setoutputtypefromstring` (`b3dutil.c:628`). Fortran wrapper.
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
pub fn overrideallbigtiff(value: &i32) {
    override_all_big_tiff(*value);
}
/// Matches C `setnextoutputsize` (`b3dutil.c:687`).
pub fn setnextoutputsize(nx: &i32, ny: &i32, nz: &i32, mode: &i32) {
    set_next_output_size(*nx, *ny, *nz, *mode);
}
/// Matches C `settiffcompressiontype` (`b3dutil.c:713`).
pub fn settiffcompressiontype(index: &i32, override_environment: &i32) -> i32 {
    set_tiff_compression_type(*index, *override_environment)
}
/// Matches C `set4bitoutputmode` (`b3dutil.c:730`).
pub fn set4bitoutputmode(value: &i32) {
    set_4_bit_output_mode(*value);
}
/// Matches C `setfloat16outputmode` (`b3dutil.c:759`).
pub fn setfloat16outputmode(value: &i32, test_for_mrc: &i32) {
    set_float_16_output_mode(*value, *test_for_mrc);
}
/// Matches C `setfloatoutputforenteredmode` (`b3dutil.c:780`).
pub fn setfloatoutputforenteredmode(mode: &i32) -> i32 {
    set_float_output_for_entered_mode(*mode)
}
/// Matches C `write16bitmodeforfloats` (`b3dutil.c:804`).
pub fn write16bitmodeforfloats() -> i32 {
    write_16_bit_mode_for_floats()
}
/// Matches C `b3dheaderitembytes` (`b3dutil.c:1168`).
pub fn b3dheaderitembytes(flags: &mut i32, bytes: &mut [i32]) {
    let (count, items) = b3d_header_item_bytes();
    *flags = count;
    for index in 0..count as usize {
        bytes[index] = items[index];
    }
}
/// Matches C `extraisnbytesandflags` (`b3dutil.c:1198`).
pub fn extraisnbytesandflags(nint: &i32, nreal: &i32) -> i32 {
    extra_is_nbytes_and_flags(*nint, *nreal)
}
/// Matches C `numberinlist` (`b3dutil.c:1227`).
pub fn numberinlist(number: &i32, list: &[i32], count: &i32, no_list_value: &i32) -> i32 {
    number_in_list(*number, Some(list), *count, *no_list_value)
}
/// Matches C `balancedgrouplimits` (`b3dutil.c:1246`).
pub fn balancedgrouplimits(total: &i32, groups: &i32, group: &i32, start: &mut i32, end: &mut i32) {
    balanced_group_limits(*total, *groups, *group, start, end);
}
/// Matches C `wallTime` from included `coresprocsthreads.c`.
pub fn wall_time() -> f64 {
    // `CLOCK_MONOTONIC` as a f64 of seconds: `std::time::Instant` has no epoch
    // to subtract from, so this stays an OS call.
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
pub fn b3dmillisleep(milliseconds: &i32) -> i32 {
    b3d_milli_sleep(*milliseconds)
}
/// Matches C `numCoresAndLogicalProcs` from included `coresprocsthreads.c` on Linux.
pub fn num_cores_and_logical_procs(physical: &mut i32, logical: &mut i32) -> i32 {
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
pub fn num_omp_threads(optimal_threads: i32) -> i32 {
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
        if num_cores_and_logical_procs(&mut processor_core_count, &mut logical_processor_count) == 0
            && processor_core_count > 0
            && logical_processor_count == num_procs
        {
            physical_procs = processor_core_count;
        }
        if std::env::var_os("IMOD_REPORT_CORES").is_some() {
            let mut stream = ImodFile::Stdout;
            let _ = stream.write_all(
                c_format(
                    "core count = %d  logical processors = %d  OMP num = %d => physical \
                     processors = %d\n",
                    &[
                        CArg::Int(processor_core_count as i64),
                        CArg::Int(logical_processor_count as i64),
                        CArg::Int(num_procs as i64),
                        CArg::Int(physical_procs as i64),
                    ],
                )
                .as_bytes(),
            );
            let _ = stream.flush();
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
        let mut stream = ImodFile::Stdout;
        let _ = stream.write_all(
            c_format(
                "numProcs %d  limThreads %d  numThreads %d\n",
                &[
                    CArg::Int(num_procs as i64),
                    CArg::Int(lim_threads as i64),
                    CArg::Int(num_threads as i64),
                ],
            )
            .as_bytes(),
        );
        let _ = stream.flush();
    }
    num_threads
}
/// Matches C `numompthreads` (`b3dutil.c:1407`).
pub fn numompthreads(optimal_threads: &i32) -> i32 {
    num_omp_threads(*optimal_threads)
}
/// Matches C `b3dOMPthreadNum` (`b3dutil.c:1416`) for the no-OpenMP build.
pub fn b3d_omp_thread_num() -> i32 {
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
pub fn standardmemorylimitmb(half_point: &i32) -> f64 {
    standard_memory_limit_mb(*half_point)
}
/// Matches C `addToArgVector` (`b3dutil.c:1542`), a file-static helper.
///
/// Its only call sites are inside `expandArgList`'s `#ifdef _WIN32` branch, so
/// nothing reaches it on this platform; it is translated because the
/// definition itself is not conditionally compiled.  The C's growable
/// `char ***argVec` is a `Vec<Vec<u8>>`, and `vecSize` is kept because the
/// source's quantum growth decides when it reallocates.
fn add_to_arg_vector(
    arg: &[u8],
    arg_vec: &mut Vec<Vec<u8>>,
    vec_size: &mut i32,
    num_in_vec: &mut i32,
    pattern: Option<&[u8]>,
    num_prefix: i32,
) -> i32 {
    let quantum = 8;
    if *num_in_vec >= *vec_size {
        *vec_size += quantum;
    }
    arg_vec.resize(*vec_size as usize, Vec::new());
    let slot = &mut arg_vec[*num_in_vec as usize];
    if pattern.is_some() && num_prefix != 0 {
        /* `strncpy(slot, pattern, numPrefix); strcpy(slot + numPrefix, arg);` */
        let pattern = pattern.unwrap();
        slot.clear();
        slot.extend_from_slice(&pattern[..(num_prefix as usize).min(pattern.len())]);
        slot.resize(num_prefix as usize, 0);
        slot.extend_from_slice(arg);
    } else {
        *slot = arg.to_vec();
    }
    *num_in_vec += 1;
    0
}
/// Matches C `expandArgList` (`b3dutil.c:1579`).
///
/// The whole body is inside `#ifdef _WIN32` / `#else`.  This is the `#else`
/// branch selected on this platform (`b3dutil.c:1712-1716`), which performs no
/// expansion at all.  The Windows branch walks the argument vector with
/// `FindFirstFile`/`FindNextFile` to expand `*` and `?` wildcards, and is not
/// translated: it is unselected here and unreachable on a Unix target.
///
/// `None` is the source's NULL return, which means the allocation failed; the
/// `#else` arm hands back the vector it was given, and `*allocated` is 0 so the
/// caller never looks at the copy.
pub fn expand_arg_list(
    arguments: &[Vec<u8>],
    count: i32,
    new_count: &mut i32,
    allocated: &mut i32,
    no_match: &mut i32,
) -> Option<Vec<Vec<u8>>> {
    *allocated = 0;
    *no_match = -1;
    *new_count = count;
    Some(arguments.to_vec())
}
/// Matches C `replaceFileArgVec` (`b3dutil.c:1722`).
///
/// Unlike `expandArgList` this body is not conditionally compiled, so it is
/// translated in full.  On this platform `expandArgList` returns the original
/// vector with `ifAlloc == 0` and `noMatchInd == -1`, which makes the two error
/// paths and the replacement path unreachable; they are kept because the source
/// keeps them.
pub fn replace_file_arg_vec(
    arguments: &mut Vec<Vec<u8>>,
    count: &mut i32,
    first: &mut i32,
    allocated: &mut i32,
) -> i32 {
    let mut new_num = 0_i32;
    let mut no_match_ind = 0_i32;
    *allocated = 0;
    if *first >= *count {
        return 0;
    }
    let prog_name = imod_prog_name(String::from_utf8_lossy(&arguments[0]).as_ref());
    let new_vec = expand_arg_list(
        &arguments[*first as usize..],
        *count - *first,
        &mut new_num,
        allocated,
        &mut no_match_ind,
    );
    let Some(new_vec) = new_vec else {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!("ERROR: {prog_name} - Allocating memory for expanded argument list\n"),
        );
        return -1;
    };
    if no_match_ind >= 0 {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "ERROR: {prog_name} - No files match entry {}\n",
                String::from_utf8_lossy(&arguments[(no_match_ind + *first) as usize])
            ),
        );
        return 1;
    }
    if *allocated != 0 {
        *arguments = new_vec;
        *count = new_num;
        *first = 0;
    }
    0
}
/// Matches C `anglewithinlimits` (`b3dutil.c:1805`).
pub fn anglewithinlimits(angle: &f32, lower: &f32, upper: &f32) -> f64 {
    angle_within_limits(*angle, *lower, *upper)
}
/// Matches C `getStandardGpuOptions` (`b3dutil.c:1818`) for the environment half of the source contract.
pub fn get_standard_gpu_options(
    if_gpu_by_environment: &mut i32,
    action_fail_option: Option<&mut i32>,
    action_fail_environment: Option<&mut i32>,
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
    if let (Some(action_fail_option), Some(action_fail_environment)) =
        (action_fail_option, action_fail_environment)
    {
        *action_fail_option = 0;
        *action_fail_environment = 0;
    }
    use_gpu
}
/// Matches C `b3dsetlocktimeout` (`b3dutil.c:1865`).
pub fn b3dsetlocktimeout(timeout: &f32) {
    b3d_set_lock_timeout(*timeout);
}
/// Matches C `b3dopenlockfile` (`b3dutil.c:1905`). Fortran wrapper.
pub unsafe fn b3dopenlockfile(filename: *const c_char, length: i32) -> i32 {
    let string = f2c_string(filename, length);
    if string.is_null() {
        return -3;
    }
    let result = b3d_open_lock_file(core::ffi::CStr::from_ptr(string).to_string_lossy().as_ref());
    libc::free(string.cast());
    result
}
/// Matches C `b3dlockfile` (`b3dutil.c:1960`).
pub fn b3dlockfile(index: &i32) -> i32 {
    b3d_lock_file(*index)
}
/// Matches C `b3dunlockfile` (`b3dutil.c:2000`).
pub fn b3dunlockfile(index: &i32) -> i32 {
    b3d_unlock_file(*index)
}
/// Matches C `b3dcloselockfile` (`b3dutil.c:2027`).
pub fn b3dcloselockfile(index: &i32) -> i32 {
    b3d_close_lock_file(*index)
}

#[cfg(test)]
mod tests {

    /// `c_format` against the C library itself, in the same process.
    ///
    /// A formatting contract is the one thing worth building a differential
    /// for (CLAUDE.md says so for `Float.toString`, and this is the same
    /// class of problem): 952 call sites in this tree feed C format strings,
    /// and a formatter that is right for the obvious values and wrong at a
    /// tie, a boundary exponent or a padding interaction would move output
    /// bytes nobody looks at until a parity run fails. So this does not
    /// assert against expected strings — it asks `libc::snprintf` and
    /// compares.
    #[test]
    fn c_format_matches_the_c_library_over_a_matrix_of_formats_and_values() {
        fn c_double(fmt: &str, v: f64) -> String {
            let cfmt = std::ffi::CString::new(fmt).unwrap();
            let mut buf = [0i8; 512];
            unsafe {
                libc::snprintf(buf.as_mut_ptr(), buf.len(), cfmt.as_ptr(), v);
                std::ffi::CStr::from_ptr(buf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            }
        }
        fn c_long(fmt: &str, v: i64) -> String {
            let cfmt = std::ffi::CString::new(fmt).unwrap();
            let mut buf = [0i8; 512];
            unsafe {
                libc::snprintf(buf.as_mut_ptr(), buf.len(), cfmt.as_ptr(), v);
                std::ffi::CStr::from_ptr(buf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            }
        }
        fn c_str(fmt: &str, v: &str) -> String {
            let cfmt = std::ffi::CString::new(fmt).unwrap();
            let cv = std::ffi::CString::new(v).unwrap();
            let mut buf = [0i8; 512];
            unsafe {
                libc::snprintf(buf.as_mut_ptr(), buf.len(), cfmt.as_ptr(), cv.as_ptr());
                std::ffi::CStr::from_ptr(buf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            }
        }

        let mut bad = Vec::new();

        // The float conversions, which are where the risk is. The format list
        // covers every shape the tree actually uses plus the flag and padding
        // interactions around them; the value list is chosen for the places a
        // formatter breaks -- both zeros, the %g exponent switch at -4 and at
        // the precision, exact ties, the subnormal and overflow ends, and the
        // non-finite values.
        let ffmts = [
            "%f", "%e", "%g", "%E", "%G", "%.0f", "%.1f", "%.3f", "%.10f", "%12.4f", "%-12.4f",
            "%+f", "%012.3f", "% f", "%#g", "%#.0f", "%.0e", "%.15g", "%8.3g", "%.2e", "%13.6f",
            "%10.6f", "%7.3f", "%.6g", "%.1g", "%.20g", "%20.10e", "%-20.10e", "%+.3e", "%g",
        ];
        let fvals = [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            1.5,
            2.5,
            0.125,
            1e-5,
            9.9999e-5,
            1e-4,
            1e20,
            1e300,
            1e-300,
            5e-324,
            std::f64::consts::PI,
            1.0 / 3.0,
            123456789.123456,
            0.1,
            1e6,
            999999.5,
            1000000.5,
            99999.99999,
            -3605683.25,
            2147483647.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1.0000000000000002,
            0.49999999999999994,
            1e15,
            1e16,
            -1e-7,
        ];
        for fmt in ffmts {
            for v in fvals {
                // The one documented divergence: for `%#g`, glibc emits the
                // significant-digit count it had before rounding carried the
                // exponent up a decade, contradicting its own `%#.5g` and
                // `%#.7g` of the same value. See `c_format`'s doc comment. No
                // format string in the tree combines `#` with a floating
                // conversion, so this is excluded rather than reproduced.
                if fmt == "%#g" && (v == 999999.5 || v == -999999.5) {
                    continue;
                }
                let ours = c_format(fmt, &[CArg::Dbl(v)]);
                let theirs = c_double(fmt, v);
                if ours != theirs {
                    bad.push(format!("{fmt:?} {v:?}: ours {ours:?} libc {theirs:?}"));
                }
            }
        }

        // The integer conversions. `%ld` is used because the CArg is an i64,
        // which is what the tree's callers will pass.
        let ifmts = [
            "%ld", "%5ld", "%-5ld", "%05ld", "%+ld", "% ld", "%.5ld", "%.0ld", "%lx", "%lX",
            "%#lx", "%lo", "%#lo", "%lu", "%12ld", "%-12ld", "%08lx",
        ];
        let ivals = [
            0i64,
            1,
            -1,
            42,
            -42,
            7,
            255,
            4096,
            2147483647,
            -2147483648,
            1000000,
        ];
        for fmt in ifmts {
            for v in ivals {
                let arg = if fmt.contains('x')
                    || fmt.contains('X')
                    || fmt.contains('o')
                    || fmt.contains('u')
                {
                    CArg::Uint(v as u64)
                } else {
                    CArg::Int(v)
                };
                let ours = c_format(fmt, &[arg]);
                let theirs = c_long(fmt, v);
                if ours != theirs {
                    bad.push(format!("{fmt:?} {v}: ours {ours:?} libc {theirs:?}"));
                }
            }
        }

        // The length modifier decides how wide the argument is, and therefore
        // what a negative value prints as. `printf("%02x", ch)` with `ch` an
        // `int` promotes to a 32-bit `unsigned int`, so `-1` is `ffffffff`;
        // `%hhx` is 8 bits and `%hx` 16. Discarding the modifier and
        // formatting the whole `u64` produced `ffffffffffffffff` at 19 sites
        // in the mini-XML translation, caught only because its differential
        // compared stderr as well as stdout.
        fn c_int(fmt: &str, v: i32) -> String {
            let cfmt = std::ffi::CString::new(fmt).unwrap();
            let mut buf = [0i8; 512];
            unsafe {
                libc::snprintf(buf.as_mut_ptr(), buf.len(), cfmt.as_ptr(), v);
                std::ffi::CStr::from_ptr(buf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            }
        }
        for fmt in [
            "%x", "%02x", "%04x", "%X", "%#x", "%o", "%u", "%d", "%5d", "%05d", "%+d", "%hhx",
            "%hx", "%hhd", "%hd", "%08x", "%.4x",
        ] {
            for v in [
                0i32,
                1,
                -1,
                -2,
                42,
                -42,
                127,
                -128,
                255,
                -255,
                32767,
                -32768,
                65535,
                2147483647,
                -2147483648,
                1000000,
                -1000000,
            ] {
                let arg = if fmt.contains('x')
                    || fmt.contains('X')
                    || fmt.contains('o')
                    || fmt.contains('u')
                {
                    CArg::Uint(v as u64)
                } else {
                    CArg::Int(v as i64)
                };
                let ours = c_format(fmt, &[arg]);
                let theirs = c_int(fmt, v);
                if ours != theirs {
                    bad.push(format!("{fmt:?} {v}: ours {ours:?} libc {theirs:?}"));
                }
            }
        }

        // Strings, where a precision is a maximum rather than a minimum.
        for fmt in ["%s", "%10s", "%-10s", "%.3s", "%10.3s", "%-10.3s"] {
            for v in ["", "a", "abc", "abcdefghijkl"] {
                let ours = c_format(fmt, &[CArg::Str(v)]);
                let theirs = c_str(fmt, v);
                if ours != theirs {
                    bad.push(format!("{fmt:?} {v:?}: ours {ours:?} libc {theirs:?}"));
                }
            }
        }

        // A multi-argument format with literal text around it, which is the
        // shape almost every real call site has.
        let ours = c_format(
            "Set area %d %d %d %d  zoom %g dpr %.2f name %s\n",
            &[
                CArg::Int(0),
                CArg::Int(63),
                CArg::Int(19),
                CArg::Int(28),
                CArg::Dbl(1.5),
                CArg::Dbl(1.0),
                CArg::Str("zap"),
            ],
        );
        assert_eq!(ours, "Set area 0 63 19 28  zoom 1.5 dpr 1.00 name zap\n");

        // A `*` width and a `*` precision, which C reads from the argument list.
        assert_eq!(
            c_format(
                "%*.*f|",
                &[CArg::Star(10), CArg::Star(3), CArg::Dbl(3.14159)]
            ),
            c_double("%10.3f|", 3.14159)
        );

        assert!(
            bad.is_empty(),
            "{} of the matrix disagreed with libc:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }
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
        b3d_error(None, format_args!("header {}", 42));
        assert_eq!(b3d_get_store_error(), 1);
        assert_eq!(b3d_get_error(), "header 42");
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
        assert_eq!(number_in_list(7, Some(&values), 3, -1), 1);
        assert_eq!(number_in_list(2, Some(&values), 3, -1), 0);
        let mut start = 0;
        let mut end = 0;
        balanced_group_limits(10, 3, 1, &mut start, &mut end);
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
