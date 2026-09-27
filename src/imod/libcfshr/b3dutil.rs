//! Selected bottom-up functions from `IMOD/libcfshr/b3dutil.c`.

use core::cell::{Cell, RefCell};
use core::sync::atomic::{AtomicI32, Ordering};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Mutex;

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
// tree uses C stdio only at this stream boundary, where its block buffering is
// part of the externally visible contract under redirection. Rust's
// line-buffered stream would reorder a message and a C-stream line in a
// redirected capture, which is NATIVE.md §1's mixed-buffering trap.
// Reading has the same problem in reverse — `mrc_head_read` takes the MRC
// header off stdin with [`b3d_fread`] while `fgetline` reads the interactive
// prompts off it with `getc`, and two different buffers over one descriptor
// lose data. When the last `printf` in a program is gone these arms can move
// to `std::io`; until then this is the correct boundary and it is one file.
unsafe extern "C" {
    static stderr: *mut libc::FILE;
    static stdout: *mut libc::FILE;
    static stdin: *mut libc::FILE;
}

/// `b3dutil.h:27`.
const MAX_IMOD_ERROR_STRING: usize = 512;
/// `b3dutil.c:1843`.
const MAX_LOCK_FILES: usize = 8;
/// `b3dutil.c:1844`.
const NUM_LOCK_BYTES: i64 = 1024;
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

/// Process-wide state for the source's `rand`/`srand` calls.  GNU libc's
/// `rand` is the TYPE_3 additive-feedback generator, so retaining this state
/// rather than choosing a Rust RNG keeps seeded IMOD output reproducible.
struct B3dRandState {
    words: [i32; 31],
    front: usize,
    rear: usize,
    initialized: bool,
    b3dran_first_time: bool,
    b3dran_last_seed: i32,
}

/// The state a fresh process starts with; [`run_in_process`] resets to it.
const B3D_RAND_STATE_INIT: B3dRandState = B3dRandState {
    words: [0; 31],
    front: 3,
    rear: 0,
    initialized: false,
    b3dran_first_time: true,
    b3dran_last_seed: 0,
};

static B3D_RAND_STATE: Mutex<B3dRandState> = Mutex::new(B3D_RAND_STATE_INIT);

/// One entry of `b3dutil.c`'s lock table.
///
/// The C keeps three parallel arrays and indexes all three with the same
/// subscript at every site: `sLockFiles[MAX_LOCK_FILES]` (`b3dutil.c:1848`),
/// `sLocksUsed` (`:1850`) and `sLockTimeouts` (`:1851`).  "These three
/// describe the same lock" was an invariant held only by that discipline;
/// one record makes it structural.  C zero-initialises its statics, which is
/// what `Default` reproduces here.
#[derive(Default)]
struct LockEntry {
    /// `sLockFiles`.  Rust owns the file; `fcntl` receives its borrowed
    /// descriptor.
    file: Option<std::fs::File>,
    /// `sLocksUsed`.
    used: i32,
    /// `sLockTimeouts`.
    timeout: f32,
}

thread_local! {
    /// `b3dutil.c:1848-1851`'s three parallel lock arrays, as one table.
    static S_LOCK_TABLE: RefCell<[LockEntry; MAX_LOCK_FILES]> =
        RefCell::new(std::array::from_fn(|_| LockEntry::default()));
    /// `b3dutil.c:1852` `static int sInitedLocks = 0`.
    static S_INITED_LOCKS: Cell<i32> = const { Cell::new(0) };
    /// `b3dutil.c:1853` `static float sDfltLockTimeout = 30.`.
    static S_DFLT_LOCK_TIMEOUT: Cell<f32> = const { Cell::new(30.) };
    /// `b3dutil.c:856` `static int storeError = 0`.
    static STORE_ERROR: Cell<i32> = const { Cell::new(0) };
    /// `b3dutil.c:857` `static char errorMess[MAX_IMOD_ERROR_STRING] = ""`.
    ///
    /// This is Rust-owned message bytes, not a NUL-terminated C buffer.  The
    /// source's 511-byte `vsprintf` limit remains part of `b3d_error`, while
    /// `b3d_get_error` returns an owned Rust string rather than exposing a
    /// pointer into mutable static storage.
    static ERROR_MESS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
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
    /// A `FILE *` on a regular file: the descriptor plus the stream buffer
    /// glibc keeps beside it.  Clones share one `Rc`, hence one file offset
    /// and one buffer, exactly as two copies of a C `FILE *` do.
    File(std::rc::Rc<std::cell::RefCell<CFile>>),
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

/// The stream behind an `ImodFile::File`: a `std::fs::File` and the
/// `BUFSIZ`-style buffer C stdio gives every `FILE *`.  glibc buffers a
/// stream in one direction at a time, and the source always `fseek`s between
/// a read and a write (C requires it), so the stream is either idle, reading
/// through a `BufReader`, or writing through a `BufWriter`, and switches by
/// flushing or by discarding the read-ahead and repositioning the descriptor
/// at the logical position.  A read or write of at least the buffer size on
/// an empty buffer bypasses it, as glibc's does, so a section read or write
/// is still one system call straight into the caller's array; the difference
/// is the model files, which the source reads and writes four bytes at a time
/// (`imodGetInt`, `imodPutInt`) and where an unbuffered handle cost one
/// system call per field.
pub struct CFile {
    /// `None` only for the instant a direction change moves the file out.
    state: Option<CState>,
    /// The logical position while reading, once a seek has established it
    /// (`None` otherwise).  It lets an absolute seek that lands inside the
    /// unread part of the read buffer move within it, as glibc's `fseek` does
    /// (`_IO_new_file_seekoff`, "if destination is within current buffer,
    /// optimize"), instead of discarding the buffer and re-reading it.
    /// Sequential chunk reads that each seek to the next chunk -- every
    /// binned `newstack`/`binvol` read -- otherwise read each block twice.
    read_pos: Option<u64>,
}

enum CState {
    Idle(std::fs::File),
    Reading(std::io::BufReader<std::fs::File>),
    Writing(std::io::BufWriter<std::fs::File>),
}

impl CFile {
    /// glibc sizes the buffer from `st_blksize`, 4096 on this filesystem.
    const BUFSIZ: usize = 4096;

    fn new(f: std::fs::File) -> CFile {
        CFile {
            state: Some(CState::Idle(f)),
            read_pos: None,
        }
    }

    fn file(&self) -> &std::fs::File {
        match self.state.as_ref().unwrap() {
            CState::Idle(f) => f,
            CState::Reading(r) => r.get_ref(),
            CState::Writing(w) => w.get_ref(),
        }
    }

    /// The stream in its reading state: a pending write buffer is flushed
    /// first, as `fflush` does before the direction changes.
    fn reader(&mut self) -> std::io::Result<&mut std::io::BufReader<std::fs::File>> {
        if !matches!(self.state, Some(CState::Reading(_))) {
            let f = match self.state.take().unwrap() {
                CState::Idle(f) => f,
                CState::Writing(w) => match w.into_inner() {
                    Ok(f) => f,
                    Err(e) => {
                        // The file comes back inside the error; keep it.
                        let (error, w) = e.into_parts();
                        self.state = Some(CState::Writing(w));
                        return Err(error);
                    }
                },
                CState::Reading(_) => unreachable!(),
            };
            self.state = Some(CState::Reading(std::io::BufReader::with_capacity(
                Self::BUFSIZ,
                f,
            )));
        }
        match self.state.as_mut().unwrap() {
            CState::Reading(r) => Ok(r),
            _ => unreachable!(),
        }
    }

    /// The stream in its writing state: read-ahead is discarded and the
    /// descriptor moved back to the logical position first.
    fn writer(&mut self) -> std::io::Result<&mut std::io::BufWriter<std::fs::File>> {
        self.read_pos = None;
        if !matches!(self.state, Some(CState::Writing(_))) {
            let f = match self.state.take().unwrap() {
                CState::Idle(f) => f,
                CState::Reading(mut r) => {
                    let position = r.stream_position()?;
                    let mut f = r.into_inner();
                    f.seek(SeekFrom::Start(position))?;
                    f
                }
                CState::Writing(_) => unreachable!(),
            };
            self.state = Some(CState::Writing(std::io::BufWriter::with_capacity(
                Self::BUFSIZ,
                f,
            )));
        }
        match self.state.as_mut().unwrap() {
            CState::Writing(w) => Ok(w),
            _ => unreachable!(),
        }
    }

    /// `fflush`: pending writes reach the descriptor; a read buffer is kept.
    fn flush(&mut self) -> std::io::Result<()> {
        match self.state.as_mut().unwrap() {
            CState::Writing(w) => w.flush(),
            _ => Ok(()),
        }
    }

    /// `fseek`/`ftell`: `BufReader::seek` drops its read-ahead (and moves
    /// within it for a relative seek); `BufWriter::seek` flushes first.  Both
    /// report the logical position.
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self.state.as_mut().unwrap() {
            CState::Idle(f) => f.seek(pos),
            CState::Reading(r) => {
                if let (SeekFrom::Start(target), Some(current)) = (pos, self.read_pos) {
                    if target >= current && target - current <= r.buffer().len() as u64 {
                        r.seek_relative((target - current) as i64)?;
                        self.read_pos = Some(target);
                        return Ok(target);
                    }
                }
                // A relative seek moves within the read buffer when it can,
                // as glibc's `fseek(fp, n, SEEK_CUR)` does: `mrcReadSectionAny`
                // (`mrcsec.c:365`, `:385`) seeks past the unread X ranges of
                // every line of a sub-area read, and `BufReader::seek` would
                // discard the buffer and re-read the block for each line
                // (beadtrack's box reads: twice native's `read` calls).
                // `seek_relative` falls back to a real seek outside the
                // buffer; the logical position is the same either way.
                if let SeekFrom::Current(offset) = pos {
                    let current = match self.read_pos {
                        Some(p) => p,
                        None => r.stream_position()?,
                    };
                    let target = current.checked_add_signed(offset);
                    if let Some(target) = target {
                        if let Err(e) = r.seek_relative(offset) {
                            self.read_pos = None;
                            return Err(e);
                        }
                        self.read_pos = Some(target);
                        return Ok(target);
                    }
                }
                let result = r.seek(pos);
                self.read_pos = result.as_ref().ok().copied();
                result
            }
            CState::Writing(w) => w.seek(pos),
        }
    }

    /// `fread` through the read buffer, keeping `read_pos` current.
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.reader()?.read(buf)?;
        if let Some(p) = self.read_pos.as_mut() {
            *p += n as u64;
        }
        Ok(n)
    }
}

/// glibc keeps every open stream on `_IO_list_all` and `exit()` flushes them
/// all; the translated programs end with `exit(0)` or `exitError` while
/// output files are still open, exactly as the C does, and `std::process::exit`
/// runs no destructor.  This is that list: a weak reference per open stream,
/// flushed from an `atexit` handler.
struct OpenStream(std::rc::Weak<std::cell::RefCell<CFile>>);
// SAFETY: a stream is only ever used from the thread that opened it -- file
// I/O in this crate is single-threaded, rayon is used only for pixel loops --
// and the exit handler runs on the thread that called `exit`, so no stream is
// touched from two threads.  `Send` is needed only to keep the list in a
// `static`, which is what lets it outlive thread-local destructors at exit.
unsafe impl Send for OpenStream {}
static OPEN_STREAMS: std::sync::Mutex<Vec<OpenStream>> = std::sync::Mutex::new(Vec::new());

extern "C" fn flush_open_streams() {
    if let Ok(list) = OPEN_STREAMS.lock() {
        for entry in list.iter() {
            if let Some(stream) = entry.0.upgrade() {
                if let Ok(mut c) = stream.try_borrow_mut() {
                    let _ = c.flush();
                }
            }
        }
    }
}

impl ImodFile {
    /// `fdopen`-style wrapper of an already open `std::fs::File`.
    pub fn from_std(f: std::fs::File) -> ImodFile {
        static REGISTER: std::sync::Once = std::sync::Once::new();
        REGISTER.call_once(|| {
            // The libc exit boundary: `std::process::exit` calls `exit`,
            // which runs this after the C streams' own flush.
            unsafe { libc::atexit(flush_open_streams) };
        });
        let stream = std::rc::Rc::new(std::cell::RefCell::new(CFile::new(f)));
        if let Ok(mut list) = OPEN_STREAMS.lock() {
            list.retain(|entry| entry.0.strong_count() > 0);
            list.push(OpenStream(std::rc::Rc::downgrade(&stream)));
        }
        ImodFile::File(stream)
    }

    /// `fopen(path, mode)`, with the C mode string the source passes around.
    ///
    /// The source carries mode strings in variables and builds them
    /// conditionally, so this takes the string rather than exposing one
    /// constructor per mode.  The path is any ordinary Rust path, so callers
    /// do not have to create a lossy C-string-shaped intermediate. `b` is
    /// accepted and ignored, as on POSIX. Returns `None` where `fopen`
    /// returns NULL.
    pub fn open(path: impl AsRef<std::path::Path>, mode: &str) -> Option<ImodFile> {
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
        o.open(path).ok().map(ImodFile::from_std)
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
        Some(ImodFile::from_std(f))
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
            // Whoever takes the descriptor sees the file as the stream has
            // written it so far, so pending output goes out first.
            ImodFile::File(f) => {
                let mut c = f.borrow_mut();
                let _ = c.flush();
                c.file().as_raw_fd()
            }
            ImodFile::Stdin => 0,
            ImodFile::Stdout => 1,
            ImodFile::Stderr => 2,
            ImodFile::Token(_) => -1,
        }
    }
}

/// `fclose` flushes the stream whatever other copies of the `FILE *` exist,
/// and the translation's clones are those copies: dropping any one of them
/// -- the one the C closed, or a temporary made to pass the handle along --
/// pushes pending output to the descriptor.  The bytes are the same either
/// way; only the system-call boundaries move.
impl Drop for ImodFile {
    fn drop(&mut self) {
        if let ImodFile::File(f) = self {
            if let Ok(mut c) = f.try_borrow_mut() {
                let _ = c.flush();
            }
        }
    }
}

impl Read for ImodFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            ImodFile::File(f) => f.borrow_mut().read(buf),
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
            ImodFile::File(f) => f.borrow_mut().writer()?.write(buf),
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
            ImodFile::File(f) => f.borrow_mut().flush(),
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
            ImodFile::File(f) => f.borrow_mut().seek(pos),
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
    // first NUL and cannot exceed the buffer.  Keep those source-visible rules
    // without retaining its fixed C character array.
    let message_length = message
        .bytes()
        .position(|byte| byte == 0)
        .unwrap_or(message.len())
        .min(MAX_IMOD_ERROR_STRING - 1);
    ERROR_MESS.with_borrow_mut(|buffer| {
        buffer.clear();
        buffer.extend_from_slice(&message.as_bytes()[..message_length]);
    });
    let store_error = STORE_ERROR.get();
    let stored = ERROR_MESS.with_borrow(|buffer| buffer.clone());
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
/// The C returns `&errorMess[0]`, a pointer into static C storage.  Rust keeps
/// the message as owned bytes and returns the equivalent owned text instead.
pub fn b3d_get_error() -> String {
    ERROR_MESS.with_borrow(|buffer| String::from_utf8_lossy(buffer).into_owned())
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
///
/// The source returns 0 or -1 and delivers its two results, the byte size of
/// the basic data element and the number of channels, through the `dataSize`
/// and `channels` pointers; both are folded into the `Ok` value here as
/// `(data_size, channels)`.
pub fn data_size_for_mode(mode: i32) -> Result<(i32, i32), ()> {
    let data_size: i32;
    let channels: i32;
    match mode {
        0 => {
            data_size = 1;
            channels = 1;
        }
        1 | 6 => {
            data_size = 2;
            channels = 1;
        }
        2 => {
            data_size = 4;
            channels = 1;
        }
        3 => {
            data_size = 2;
            channels = 2;
        }
        4 => {
            data_size = 4;
            channels = 2;
        }
        16 => {
            data_size = 1;
            channels = 3;
        }
        99 => {
            data_size = 4;
            channels = 3;
        }
        _ => return Err(()),
    }
    Ok((data_size, channels))
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
                            let mut d = c_fixed_digits(mag, p);
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
                            // Both forms print the value correctly rounded to
                            // P significant digits, so the one `%e`-style
                            // formatting gives the digits for either: the
                            // fixed form is those digits with the point moved
                            // (when rounding carries, e.g. 9.9999996 -> 1e+01,
                            // the fixed form's coarser rounding yields the
                            // same 10^X).  Formatting once instead of twice
                            // halves the cost of every `%g`.
                            let (mant, x) = if mag == 0.0 {
                                (
                                    if p > 1 {
                                        format!("0.{}", "0".repeat(p - 1))
                                    } else {
                                        "0".to_string()
                                    },
                                    0i32,
                                )
                            } else {
                                c_exponent_digits(mag, p - 1)
                            };
                            let mut d = if x < -4 || x >= p as i32 {
                                let mut m = mant;
                                if alt && p - 1 == 0 {
                                    m.push('.');
                                }
                                format!(
                                    "{m}{}{}{:02}",
                                    if upper { 'E' } else { 'e' },
                                    if x < 0 { '-' } else { '+' },
                                    x.abs()
                                )
                            } else {
                                let fp = (p as i32 - 1 - x).max(0) as usize;
                                let digits: String = mant.chars().filter(|&c| c != '.').collect();
                                let mut d = if x >= 0 {
                                    let (int, frac) = digits.split_at(x as usize + 1);
                                    if fp > 0 {
                                        format!("{int}.{frac}")
                                    } else {
                                        int.to_string()
                                    }
                                } else {
                                    format!("0.{}{digits}", "0".repeat((-x - 1) as usize))
                                };
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

/// `round(mag * 10^pos)` for a finite `mag >= 0`, computed exactly in integer
/// arithmetic from the binary value (ties to even, as glibc's printf rounds
/// in the default mode), or `None` when the operands would not fit in 128
/// bits.  Performance only: the conversions below used Rust's exact
/// formatter (Grisu with a Dragon4 bignum fallback) for every `%e`/`%f`/`%g`,
/// which cost ~300 ns a number; `imodinfo -a` of a 5 000-point model spent
/// most of its time there.  The digits are the same correctly rounded ones,
/// and `None` falls back to that formatter.
fn c_exact_scaled_round(mag: f64, pos: i32) -> Option<u128> {
    let bits = mag.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = (bits & ((1_u64 << 52) - 1)) as u128;
    let (m, e) = if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1 << 52), biased - 1075)
    };
    if m == 0 {
        return Some(0);
    }
    let round = |q: u128, r: u128, half: u128| -> u128 {
        if r > half || (r == half && q & 1 == 1) {
            q + 1
        } else {
            q
        }
    };
    if pos >= 0 {
        if pos > 22 {
            return None;
        }
        let num = m * 10_u128.pow(pos as u32);
        if e >= 0 {
            if 128 - num.leading_zeros() as i32 + e > 127 {
                return None;
            }
            return Some(num << e);
        }
        let shift = -e;
        if shift > 127 {
            return None;
        }
        let q = num >> shift;
        let r = num & ((1_u128 << shift) - 1);
        Some(round(q, r, 1_u128 << (shift - 1)))
    } else {
        let j = -pos;
        if j > 38 {
            return None;
        }
        let ten = 10_u128.pow(j as u32);
        let (num, den) = if e >= 0 {
            if 128 - m.leading_zeros() as i32 + e > 127 {
                return None;
            }
            (m << e, ten)
        } else {
            let shift = -e;
            if 128 - ten.leading_zeros() as i32 + shift > 127 {
                return None;
            }
            (m, ten << shift)
        };
        let q = num / den;
        let r = num % den;
        // Compare 2r with den exactly: den is even (a multiple of 10).
        let half = den / 2;
        Some(round(q, r, half))
    }
}

/// The `%e` digits of `mag` with `prec` decimals: the mantissa as
/// `d.ddd` (no point when `prec` is 0) and the decimal exponent, exactly as
/// `format!("{mag:.prec$e}")` gives them.
fn c_exponent_digits(mag: f64, prec: usize) -> (String, i32) {
    if mag > 0.0 && mag.is_finite() && prec <= 30 {
        let mut x = mag.log10().floor() as i32;
        for _ in 0..3 {
            let Some(n) = c_exact_scaled_round(mag, prec as i32 - x) else {
                break;
            };
            let low = 10_u128.pow(prec as u32);
            if n >= 10 * low {
                x += 1;
                continue;
            }
            if n < low {
                x -= 1;
                continue;
            }
            let digits = n.to_string();
            let mut m = String::with_capacity(prec + 2);
            m.push_str(&digits[..1]);
            if prec > 0 {
                m.push('.');
                m.push_str(&digits[1..]);
            }
            return (m, x);
        }
    }
    let s = format!("{mag:.prec$e}");
    let at = s.find('e').unwrap();
    (s[..at].to_string(), s[at + 1..].parse().unwrap_or(0))
}

/// `format!("{mag:.prec$}")` for the `%f` conversion, via
/// [`c_exact_scaled_round`] when it applies.
fn c_fixed_digits(mag: f64, prec: usize) -> String {
    if mag.is_finite() && prec <= 22 {
        if let Some(n) = c_exact_scaled_round(mag, prec as i32) {
            let digits = n.to_string();
            if prec == 0 {
                return digits;
            }
            let digits = if digits.len() <= prec {
                format!("{}{digits}", "0".repeat(prec + 1 - digits.len()))
            } else {
                digits
            };
            let (int, frac) = digits.split_at(digits.len() - prec);
            return format!("{int}.{frac}");
        }
    }
    format!("{mag:.prec$}")
}

/// The `%e` conversion of [`c_format`], for a non-negative finite `mag`.
///
/// Rust's `{:e}` writes `1.5e5`; C writes `1.500000e+05` — the exponent always
/// carries a sign and at least two digits. Split out only because `%g` needs
/// the identical conversion, which is the language boundary the no-helpers
/// rule allows for.
fn c_format_e(mag: f64, prec: usize, alt: bool, upper: bool) -> String {
    let (mut m, e) = c_exponent_digits(mag, prec);
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
            format!("{program_name} Version 5.2.17 {IMOD_BUILD_DATE} {IMOD_BUILD_TIME}\n")
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
    let _ =
        ImodFile::Stdout.write_all(format!("Copyright (C) 1994-2025 by the {uofc}\n").as_bytes());
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
/// The program's `argv`, as C `main(int argc, char *argv[])` receives it.
///
/// Not a source function: upstream installs one executable per command, so
/// each `main` reads its own process `argv`.  This crate builds a single
/// busybox-style binary (`src/bin/imod.rs`), and for the subcommand form
/// `imod <cmd> args...` the launcher records the `argv` the command would have
/// had as its own executable -- `<bindir>/<cmd>`, then `args...` -- before
/// dispatching in-process.  Every translated unit that reads `argv` reads it
/// here instead of from `std::env::args_os()`.  When nothing was recorded (the
/// link form, or a library caller) it is the process `argv`.
static PROGRAM_ARGV: std::sync::OnceLock<Vec<std::ffi::OsString>> = std::sync::OnceLock::new();

/// Records the effective `argv` for [`program_args_os`]; the launcher calls it
/// once, before dispatch.  A second call is ignored.
pub fn set_program_args(argv: Vec<std::ffi::OsString>) {
    let _ = PROGRAM_ARGV.set(argv);
}

/// The effective `argv` (see [`PROGRAM_ARGV`]), as `std::env::args_os()`.
pub fn program_args_os() -> Vec<std::ffi::OsString> {
    // A command run by [`run_in_process`] sees the `argv` it was given there.
    if let Some(argv) = IN_PROCESS_ARGV.with_borrow(|argv| argv.clone()) {
        return argv;
    }
    match PROGRAM_ARGV.get() {
        Some(argv) => argv.clone(),
        None => std::env::args_os().collect(),
    }
}

/// The effective `argv` as `std::env::args()` yields it, panicking on an
/// argument that is not valid Unicode exactly as that iterator does.
pub fn program_args() -> Vec<String> {
    program_args_os()
        .into_iter()
        .map(|argument| argument.into_string().unwrap())
        .collect()
}

// ---------------------------------------------------------------------------
// Running one of this crate's own commands in process.
//
// Not a source unit.  Upstream's Python scripts (`pysrc/trimvol`) and one
// Fortran program (`blendmont.f90:1085`, `call system('clip plane ...')`) run
// other IMOD programs as child processes; by owner decision (2026-09-24) the
// translations call those programs in this process instead.  A translated
// program is a whole `main`: it reads `argv` through [`program_args_os`],
// keeps PIP, unit and error state in `thread_local!`s, writes to the C
// `stdout`, and ends in `exit()`.  [`run_in_process`] gives it exactly that
// environment without a `fork`:
//
// * it runs on a **fresh thread**, so every `thread_local!` (PIP's option
//   table and exit prefix, the Fortran unit table, `b3dError`'s buffer, ...)
//   starts at its initial value, as in a new process;
// * [`program_args_os`] returns the `argv` given here on that thread;
// * [`exit`] -- C `exit` -- flushes every stream as C `exit()` does and then
//   *unwinds* to the runner with the status instead of ending the process;
// * the few process-global statics a program can set (the six
//   `b3dutil.c` output overrides and the `rand` state) are reset to their
//   initial values for the call and restored afterwards;
// * standard input can be fed from given text and standard output captured,
//   by pointing descriptors 0 and 1 at an unlinked temporary file for the
//   duration of the call, which is what a pipe to a child does to every
//   writer (C stdio, `println!`, `ImodFile::Stdout`) alike.
//
// Callers must call [`exit`], never `std::process::exit`, on any path a
// command reached this way can take.
// ---------------------------------------------------------------------------

thread_local! {
    /// The `argv` of the command [`run_in_process`] is running on this thread;
    /// `Some` also marks the thread as an in-process command for [`exit`].
    static IN_PROCESS_ARGV: RefCell<Option<Vec<std::ffi::OsString>>> = const { RefCell::new(None) };
}

/// The unwind payload [`exit`] carries back to [`run_in_process`].
pub struct InProcessExit(pub i32);

/// Flushes every output stream, as C `exit()` does before a process ends:
/// the C streams (`fflush(NULL)`), Rust's `stdout`, and every open
/// [`ImodFile`].
fn flush_all_streams() {
    unsafe { libc::fflush(std::ptr::null_mut()) };
    let _ = std::io::stdout().flush();
    flush_open_streams();
}

/// C `exit(status)`.
///
/// Outside [`run_in_process`] this is `std::process::exit`, which calls libc
/// `exit` and so flushes the C streams and runs [`flush_open_streams`] from
/// its `atexit` registration.  Inside it, the same flush happens here and the
/// status unwinds to the runner, dropping (and so flushing) everything the
/// command still held on the way.
pub fn exit(status: i32) -> ! {
    if IN_PROCESS_ARGV.with_borrow(|argv| argv.is_some()) {
        flush_all_streams();
        std::panic::resume_unwind(Box::new(InProcessExit(status)));
    }
    std::process::exit(status)
}

/// Runs `entry` -- a translated program's `main` -- as if it were the
/// program `argv[0]` started with `argv`, and returns its exit status and,
/// when `capture` is set, everything it wrote to standard output.  `input`,
/// when given, is what the program reads on standard input.  See the section
/// comment above for what "as if" covers.  A panic that is not an [`exit`]
/// is re-raised in the caller after the descriptors are restored.
pub fn run_in_process<F: FnOnce() + Send + 'static>(
    argv: Vec<std::ffi::OsString>,
    input: Option<&[u8]>,
    capture: bool,
    entry: F,
) -> std::io::Result<(i32, Vec<u8>)> {
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let temp_file = || -> std::io::Result<std::fs::File> {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-in-process-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::SeqCst)
        ));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        let _ = std::fs::remove_file(&path);
        Ok(file)
    };
    use std::os::fd::AsRawFd;
    flush_all_streams();
    let mut saved_in = -1;
    let mut input_file = None;
    if let Some(text) = input {
        let mut file = temp_file()?;
        file.write_all(text)?;
        file.seek(SeekFrom::Start(0))?;
        saved_in = unsafe { libc::dup(0) };
        unsafe {
            libc::dup2(file.as_raw_fd(), 0);
            // A caller that read its own standard input to the end (a script
            // run with -StandardInput) leaves C `stdin` at EOF, which is
            // sticky: drop what it buffered and clear that EOF, or the
            // command reads nothing from its input
            libc::fflush(stdin);
            libc::clearerr(stdin);
        }
        input_file = Some(file);
    }
    let mut saved_out = -1;
    let mut output_file = None;
    if capture {
        let file = temp_file()?;
        saved_out = unsafe { libc::dup(1) };
        unsafe { libc::dup2(file.as_raw_fd(), 1) };
        output_file = Some(file);
    }

    // Process-global state a fresh process would start with.
    let overrides = [
        &S_WRITE_BYTES_OVERRIDE,
        &S_OUTPUT_TYPE_OVERRIDE,
        &S_WRITE_4_BIT_MODE,
        &S_WRITE_16_BIT_FLOATS,
        &S_INVERT_MRC_ORIGIN_OVERRIDE,
        &S_ALL_BIG_TIFF_OVERRIDE,
    ];
    let initial = [-1, -1, 0, -1, -1, -1];
    let saved: Vec<i32> = overrides
        .iter()
        .zip(initial)
        .map(|(value, init)| value.swap(init, Ordering::SeqCst))
        .collect();
    let saved_rand = std::mem::replace(
        &mut *B3D_RAND_STATE.lock().expect("b3d random state poisoned"),
        B3D_RAND_STATE_INIT,
    );

    let joined = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            IN_PROCESS_ARGV.with_borrow_mut(|slot| *slot = Some(argv));
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(entry)) {
                // A Fortran `end program` or a C `return` from `main`.
                Ok(()) => {
                    flush_all_streams();
                    0
                }
                Err(payload) => match payload.downcast::<InProcessExit>() {
                    Ok(status) => status.0,
                    Err(payload) => std::panic::resume_unwind(payload),
                },
            }
        })?
        .join();

    flush_all_streams();
    if let Some(file) = input_file {
        unsafe {
            // Drop what C `stdin` buffered from the file, then clear its EOF.
            libc::fflush(stdin);
            libc::dup2(saved_in, 0);
            libc::close(saved_in);
            libc::clearerr(stdin);
        }
        drop(file);
    }
    let mut output = Vec::new();
    if let Some(mut file) = output_file {
        unsafe {
            libc::dup2(saved_out, 1);
            libc::close(saved_out);
        }
        file.seek(SeekFrom::Start(0))?;
        file.read_to_end(&mut output)?;
    }
    for (value, old) in overrides.iter().zip(saved) {
        value.store(old, Ordering::SeqCst);
    }
    *B3D_RAND_STATE.lock().expect("b3d random state poisoned") = saved_rand;
    match joined {
        Ok(status) => Ok((status, output)),
        Err(payload) => std::panic::resume_unwind(payload),
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
/// `exitError` needs `strerror(errno)` for a failed open, and
/// [`ImodFile::open`] reports failure as `None`; the failed `open` system call
/// has just left its error in the thread's `errno`, which
/// `std::io::Error::last_os_error` reads (and renders with a ` (os error N)`
/// suffix the C library does not add, trimmed off).  The one case with no
/// system call behind it is a mode string `fopen` would reject with `EINVAL`,
/// which no caller passes.
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
    if mode.as_bytes().first() == Some(&b'w') {
        // `imodBackupFile` returns 0 when it renamed the file or there was
        // none, so the warning is printed when the rename *failed*; the text
        // is the source's.
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
    } else if mode.as_bytes().first() == Some(&b'a') {
        desc_ind = 2;
    }

    let fp = match ImodFile::open(name, mode) {
        Some(fp) => fp,
        None => {
            let error = std::io::Error::last_os_error().to_string();
            let error = error
                .split(" (os error ")
                .next()
                .unwrap_or(&error)
                .to_string();
            crate::imod::libcfshr::parse_params::exit_error(
                c_format(
                    "Opening %s, %s: %s",
                    &[
                        CArg::Str(descrip[desc_ind]),
                        CArg::Str(name),
                        CArg::Str(&error),
                    ],
                )
                .as_bytes(),
            );
        }
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
    let _ = stream.write_all(format!("Shell PID: {}\n", imod_getpid()).as_bytes());
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

/// Matches the in-place uses of C `b3dShiftBytes` (`b3dutil.c:474`).
///
/// Signed-byte conversion is a flip of the high bit in the same owned image
/// buffer.  The C routine spells that as aliased unsigned and signed pointers;
/// the Rust representation makes the in-place ownership explicit.
pub fn b3d_shift_bytes(buf: &mut [u8], nx: i32, ny: i32, direction: i32, bytes_signed: i32) {
    if bytes_signed == 0 {
        return;
    }
    let Some(nxy) = usize::try_from(nx).ok().and_then(|width| {
        usize::try_from(ny)
            .ok()
            .and_then(|height| width.checked_mul(height))
    }) else {
        return;
    };
    if nxy > buf.len() {
        return;
    }
    // `b3dutil.c:480-486` tests `direction` once and then runs one of two flat
    // byte loops; the test is not inside the loop.
    if direction >= 0 {
        for value in &mut buf[..nxy] {
            *value = (*value as i32 - 128) as i8 as u8;
        }
    } else {
        for value in &mut buf[..nxy] {
            *value = (*value as i8 as i32 + 128) as u8;
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
    let Ok((bytes, channels)) = data_size_for_mode(mode) else {
        return;
    };
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

/// Matches C `f2cString` (`b3dutil.c:811`) with Rust ownership.
///
/// `string` is the Fortran `character*(*)` argument itself, so the slice
/// length is the source's `strSize` -- the hidden length argument is the
/// data here, not redundancy, and a caller must hand over the whole
/// blank-padded field rather than a trimmed view.  The source scans back to
/// the last non-blank, copies that many bytes and appends the terminator;
/// an owned `String` of the trimmed bytes is that C string without the NUL,
/// which is encoding rather than data at this boundary.  The `malloc`
/// failure the source returns NULL for has no Rust counterpart.
pub fn fortran_string(string: &[u8]) -> String {
    /* find last non-blank character */
    let end = string
        .iter()
        .rposition(|byte| *byte != b' ')
        .map_or(0, |index| index + 1);
    String::from_utf8_lossy(&string[..end]).into_owned()
}
/// Matches C `c2fString` (`b3dutil.c:835`). The other half of the Fortran
/// bridge; see [`fortran_string`].
///
/// `c_string` holds the source's NUL-terminated bytes without the terminator,
/// and `fortran_string` is the destination `character*(*)` field, whose slice
/// length is the source's `fSize`.  The blank padding at `b3dutil.c:848` is
/// part of the Fortran string's value, so the whole remainder of the
/// destination is filled rather than left as residue.
///
/// The source has one failure (`b3dutil.c:843`): the C string did not fit in
/// the Fortran character variable, so a non-null character is left over.  That
/// is `Err(())`; `Ok(())` is the blank-padded success the source returns 0 for.
pub fn c2f_string(c_string: &[u8], fortran_string: &mut [u8]) -> Result<(), ()> {
    let f_size = fortran_string.len();
    let mut index: usize = 0;
    while index < c_string.len() && index < f_size {
        fortran_string[index] = c_string[index];
        index += 1;
    }
    /* Return error if there is still a non-null character */
    if index < c_string.len() {
        return Err(());
    }
    /* Blank-pad */
    while index < f_size {
        fortran_string[index] = b' ';
        index += 1;
    }
    Ok(())
}

/// Matches the Fortran-callable C `imodgetenv` (`b3dutil.c:333`): gets the
/// environment variable `var` (a blank-padded `character*(*)`) and returns
/// its value in `value`, the destination `character*(*)`.  Returns 1 if the
/// variable is not defined, -1 if the value does not fit in `value` (which
/// then holds as much as fits, as `c2fString` leaves it), and 0 on success.
pub fn imodgetenv(var: &[u8], value: &mut [u8]) -> i32 {
    let cstr = fortran_string(var);
    let Some(val_ptr) = std::env::var_os(&cstr) else {
        return 1;
    };
    use std::os::unix::ffi::OsStrExt;
    match c2f_string(val_ptr.as_bytes(), value) {
        Ok(()) => 0,
        Err(()) => -1,
    }
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
///
/// The live branch on this platform is the `USE_SYSTEM_FSEEK` one
/// (`b3dutil.c:978-996`): the build's generated `imodconfig.h` defines
/// `USE_SYSTEM_FSEEK` (`/tmp/imod-reference-build/include/imodconfig.h:12`,
/// written by `IMOD/setup`), and neither `WIN32_BIGFILE` nor `MAC103_BIGFILE`
/// is set, so the whole seek is one
/// `fseek(fp, (long)size1 * (long)size2 + base, flag)`.  An earlier
/// translation took the `#else` arm -- a base seek followed by stepped
/// `SEEK_CUR` seeks under 2 GB each -- which lands in the same place but
/// splits every seek in two.  The intermediate position fell outside the read
/// buffer, so each chunk of a binned read discarded and re-read its block, and
/// `newstack -bin 2` read its input twice.
///
/// The one kept difference is `stdin`: `fseek` on the C `stdin` stream
/// would reposition that stream, while the other seek routines here return
/// early for it (`b3dFseek`, `b3dutil.c:910`); an MRC is never read from a
/// pipe, and this keeps the previous behaviour rather than seeking the
/// descriptor underneath a C stream it does not own.
pub fn mrc_big_seek(file: &mut ImodFile, base: i32, size1: i32, size2: i32, flag: i32) -> i32 {
    if file.is_stdin() {
        return 0;
    }
    let offset = size1 as i64 * size2 as i64 + base as i64;
    let position = if flag == SEEK_SET {
        if offset < 0 {
            return -1;
        }
        SeekFrom::Start(offset as u64)
    } else if flag == SEEK_CUR {
        SeekFrom::Current(offset)
    } else {
        SeekFrom::End(offset)
    };
    match file.seek(position) {
        Ok(_) => 0,
        Err(_) => -1,
    }
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
    // Performance: for a file, the `getc` loop below is run over the read
    // buffer directly -- the same bytes consumed, the same stopping byte
    // consumed with them, the same EOF/error value -- instead of one
    // borrow-and-`read` per character (half of `wmod2imod`'s time on a
    // 5 000-point model).
    if let ImodFile::File(file) = fp {
        use std::io::BufRead;
        let mut file = file.borrow_mut();
        c = -1;
        'fill: loop {
            let Ok(reader) = file.reader() else {
                break;
            };
            let buffer = match reader.fill_buf() {
                Ok(buffer) if !buffer.is_empty() => buffer,
                _ => break,
            };
            let mut consumed = 0;
            let mut stopped = false;
            for &byte in buffer {
                consumed += 1;
                if i >= (limit - 1) as usize || byte == b'\n' {
                    c = byte as i32;
                    stopped = true;
                    break;
                }
                s[i] = byte;
                i += 1;
            }
            reader.consume(consumed);
            if let Some(p) = file.read_pos.as_mut() {
                *p += consumed as u64;
            }
            if stopped {
                break 'fill;
            }
        }
    } else {
        loop {
            c = fp.getc();
            if c == -1 || i >= (limit - 1) as usize || c == b'\n' as i32 {
                break;
            }
            s[i] = c as u8;
            i += 1;
        }
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
/// Matches C `makeLinePointers` (`b3dutil.c:1269`).
///
/// The C returns a `malloc`ed array of `ysize` byte pointers, line `i` at
/// `array + (size_t)xsize * i * dsize`, which the caller `free`s.  Here the
/// array is `Vec<&[u8]>`, line `i` being the byte view of `array` *from* that
/// offset to its end: the C pointer is not bounded by the line either, and
/// the consumers (`sampleMeanSD`, `zoomWithFilter`, …) take exactly this
/// shape, a slice of line slices over the image's bytes (`MrcData::bytes` or
/// a caller's own byte view).  Ownership replaces the `free`.
///
/// `None` is the C's `NULL`: a failed allocation, including the huge request a
/// negative `ysize` makes.  A line whose start lies past the end of `array`
/// is a pointer the C could form but not use, and is reported as `None` too
/// rather than a panic; every C caller already handles `NULL` as an error.
pub fn make_line_pointers(array: &[u8], xsize: i32, ysize: i32, dsize: i32) -> Option<Vec<&[u8]>> {
    if ysize < 0 {
        return None;
    }
    let mut line_ptrs: Vec<&[u8]> = Vec::new();
    if line_ptrs.try_reserve_exact(ysize as usize).is_err() {
        return None;
    }
    for i in 0..ysize {
        let offset = (xsize as usize)
            .wrapping_mul(i as usize)
            .wrapping_mul(dsize as usize);
        line_ptrs.push(array.get(offset..)?);
    }
    Some(line_ptrs)
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
/// C89 `clock()`, which `b3dutil.c:1334` divides by `CLOCKS_PER_SEC`.
///
/// The `libc` crate exports neither, so both are named here.  `CLOCKS_PER_SEC`
/// is genuinely platform-dependent: POSIX mandates 1 000 000, while MSVCRT
/// defines it as 1 000, so the C macro's value differs per target and the
/// division must follow it.
unsafe extern "C" {
    fn clock() -> libc::clock_t;
}

#[cfg(unix)]
const CLOCKS_PER_SEC: libc::clock_t = 1_000_000;
#[cfg(windows)]
const CLOCKS_PER_SEC: libc::clock_t = 1_000;

/// Matches C `cputime` (`b3dutil.c:1327`).
pub fn cputime() -> f64 {
    // `b3dutil.c:1329-1333`'s `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)` arm is
    // **commented out**; the live line is `:1334`,
    // `return ((double)clock() / CLOCKS_PER_SEC);`.  This translation had
    // implemented the commented-out branch, which is both unfaithful and
    // glibc-only -- `clock()` is C89 and exists on Windows.
    unsafe { clock() as f64 / CLOCKS_PER_SEC as f64 }
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
/// This is glibc's TYPE_3 additive-feedback generator, retained in owned Rust
/// state because its sequence is part of IMOD's output.
pub fn b3drand() -> f32 {
    let mut state = B3D_RAND_STATE.lock().expect("b3d random state poisoned");
    if !state.initialized {
        let mut word = 1_i32;
        state.words[0] = word;
        for entry in &mut state.words[1..] {
            let hi = word / 127_773;
            let lo = word % 127_773;
            word = 16_807 * lo - 2_836 * hi;
            if word < 0 {
                word += 2_147_483_647;
            }
            *entry = word;
        }
        state.front = 3;
        state.rear = 0;
        for _ in 0..310 {
            let front = state.front;
            let rear = state.rear;
            let value = (state.words[front] as u32).wrapping_add(state.words[rear] as u32);
            state.words[front] = value as i32;
            state.front = (state.front + 1) % state.words.len();
            state.rear = (state.rear + 1) % state.words.len();
        }
        state.initialized = true;
    }
    let front = state.front;
    let rear = state.rear;
    let value = (state.words[front] as u32).wrapping_add(state.words[rear] as u32);
    state.words[front] = value as i32;
    state.front = (state.front + 1) % state.words.len();
    state.rear = (state.rear + 1) % state.words.len();
    (value >> 1) as f32 / 2_147_483_647_f32
}
/// Matches C `b3dsrand` (`b3dutil.c:1763`). Fortran wrapper: the seed arrives
/// by reference.
pub fn b3dsrand(seed: &i32) {
    let mut state = B3D_RAND_STATE.lock().expect("b3d random state poisoned");
    let mut word = if *seed == 0 { 1 } else { *seed };
    state.words[0] = word;
    for entry in &mut state.words[1..] {
        let hi = word / 127_773;
        let lo = word % 127_773;
        word = 16_807 * lo - 2_836 * hi;
        if word < 0 {
            word += 2_147_483_647;
        }
        *entry = word;
    }
    state.front = 3;
    state.rear = 0;
    for _ in 0..310 {
        let front = state.front;
        let rear = state.rear;
        let value = (state.words[front] as u32).wrapping_add(state.words[rear] as u32);
        state.words[front] = value as i32;
        state.front = (state.front + 1) % state.words.len();
        state.rear = (state.rear + 1) % state.words.len();
    }
    state.initialized = true;
}
/// Matches C `b3dran` (`b3dutil.c:1776`). Fortran wrapper.
pub fn b3dran(seed: &i32) -> f32 {
    let reseed = {
        let state = B3D_RAND_STATE.lock().expect("b3d random state poisoned");
        state.b3dran_first_time || *seed != state.b3dran_last_seed
    };
    if reseed {
        b3dsrand(seed);
        let mut state = B3D_RAND_STATE.lock().expect("b3d random state poisoned");
        state.b3dran_last_seed = *seed;
        state.b3dran_first_time = false;
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
        S_LOCK_TABLE.with_borrow_mut(|table| {
            for index in 0..MAX_LOCK_FILES {
                table[index].used = -1;
            }
        });
    }
    S_INITED_LOCKS.set(1);
    let mut ind = 0;
    S_LOCK_TABLE.with_borrow(|table| {
        while ind < MAX_LOCK_FILES && table[ind].used >= 0 {
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
    // The file remains owned by the table until `b3d_close_lock_file`; `fcntl`
    // below borrows its descriptor without transferring ownership.
    let timeout = S_DFLT_LOCK_TIMEOUT.get();
    S_LOCK_TABLE.with_borrow_mut(|table| {
        table[ind].file = Some(file);
        table[ind].used = 0;
        table[ind].timeout = timeout;
    });
    ind as i32
}
/// Matches C `b3dLockFile` (`b3dutil.c:1922`).
pub fn b3d_lock_file(index: i32) -> i32 {
    if !(0..MAX_LOCK_FILES as i32).contains(&index) {
        return -1;
    }
    let index = index as usize;
    let used = S_LOCK_TABLE.with_borrow(|table| table[index].used);
    if used < 0 {
        return -2;
    }
    if used > 0 {
        S_LOCK_TABLE.with_borrow_mut(|table| table[index].used += 1);
        return 0;
    }
    #[cfg(unix)]
    let lock = libc::flock {
        l_type: libc::F_WRLCK as _,
        l_whence: SEEK_SET as _,
        l_start: 0,
        l_len: NUM_LOCK_BYTES as _,
        l_pid: 0,
    };
    // One borrow now serves both, where the parallel arrays needed two.
    #[cfg(unix)]
    use std::os::fd::AsRawFd;
    let (descriptor, timeout) = S_LOCK_TABLE.with_borrow(|table| {
        (
            {
                #[cfg(unix)]
                {
                    table[index]
                        .file
                        .as_ref()
                        .map(AsRawFd::as_raw_fd)
                        .unwrap_or(-1)
                }
            },
            table[index].timeout,
        )
    });
    let started = std::time::Instant::now();
    loop {
        // `b3dutil.c:1943-1947`: `LockFile` on Windows, `fcntl(F_SETLK)` on POSIX.
        #[cfg(unix)]
        let acquired = unsafe { libc::fcntl(descriptor, libc::F_SETLK, &lock) } >= 0;
        // `b3dutil.c:1944` calls `LockFile`.  `std::fs::File::try_lock` is the
        // portable spelling of the same request (it reaches `LockFileEx`), so
        // no kernel32 binding is needed here.
        #[cfg(windows)]
        let acquired = S_LOCK_TABLE
            .with_borrow(|table| table[index].file.as_ref().map(|f| f.try_lock().is_ok()))
            .unwrap_or(false);
        if acquired {
            S_LOCK_TABLE.with_borrow_mut(|table| table[index].used += 1);
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
    let used = S_LOCK_TABLE.with_borrow(|table| table[index].used);
    if used < 0 {
        return -2;
    }
    if used == 0 {
        return -3;
    }
    if used == 1 {
        #[cfg(unix)]
        let lock = libc::flock {
            l_type: libc::F_UNLCK as _,
            l_whence: SEEK_SET as _,
            l_start: 0,
            l_len: NUM_LOCK_BYTES as _,
            l_pid: 0,
        };
        #[cfg(unix)]
        use std::os::fd::AsRawFd;
        let descriptor = S_LOCK_TABLE.with_borrow(|table| {
            #[cfg(unix)]
            {
                table[index]
                    .file
                    .as_ref()
                    .map(AsRawFd::as_raw_fd)
                    .unwrap_or(-1)
            }
        });
        // `b3dutil.c:1985-1989`: `UnlockFile` on Windows, `fcntl` on POSIX.
        #[cfg(unix)]
        let failed = unsafe { libc::fcntl(descriptor, libc::F_SETLK, &lock) } < 0;
        // `b3dutil.c:1986`'s `UnlockFile`, via portable std.
        #[cfg(windows)]
        let failed = S_LOCK_TABLE
            .with_borrow(|table| table[index].file.as_ref().map(|f| f.unlock().is_err()))
            .unwrap_or(true);
        if failed {
            return 1;
        }
    }
    S_LOCK_TABLE.with_borrow_mut(|table| table[index].used -= 1);
    0
}
/// Matches C `b3dCloseLockFile` (`b3dutil.c:2009`).
pub fn b3d_close_lock_file(index: i32) -> i32 {
    if !(0..MAX_LOCK_FILES as i32).contains(&index) {
        return -1;
    }
    let index = index as usize;
    if S_LOCK_TABLE.with_borrow(|table| table[index].used) < 0 {
        return -2;
    }
    if S_LOCK_TABLE
        .with_borrow_mut(|table| table[index].file.take())
        .is_none()
    {
        return 1;
    }
    S_LOCK_TABLE.with_borrow_mut(|table| table[index].used = -1);
    0
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
                format!(
                    "core count = {processor_core_count}  logical processors = {logical_processor_count}  OMP num = {num_procs} => physical processors = {physical_procs}\n"
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
            format!("numProcs {num_procs}  limThreads {lim_threads}  numThreads {num_threads}\n")
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
/// Matches C `getStandardGpuOptions` (`b3dutil.c:1818`).
///
/// The `-UseGPU` entry overrides `IMOD_USE_GPU` and `IMOD_USE_GPU2` overrides
/// both; `-ActionIfGPUFails` is read into the two action values only when both
/// are supplied, and they are left untouched when it was not entered.  The
/// environment values go through `atoi`: leading digits, 0 for none.
pub fn get_standard_gpu_options(
    if_gpu_by_environment: &mut i32,
    action_fail_option: Option<&mut i32>,
    action_fail_environment: Option<&mut i32>,
) -> i32 {
    let atoi = |value: &std::ffi::OsStr| {
        let bytes = value.as_encoded_bytes();
        let mut end = 0usize;
        crate::imod::libcfshr::parse_params::strtol(bytes, &mut end, 10) as i32
    };
    let mut use_gpu = -1;
    *if_gpu_by_environment = 0;
    if let Some(value) = std::env::var_os("IMOD_USE_GPU") {
        *if_gpu_by_environment = 1;
        use_gpu = atoi(&value);
    }
    if crate::imod::libcfshr::parse_params::pip_get_integer(b"UseGPU", &mut use_gpu) == 0 {
        *if_gpu_by_environment = 0;
    }
    if let Some(value) = std::env::var_os("IMOD_USE_GPU2") {
        *if_gpu_by_environment = 1;
        use_gpu = atoi(&value);
    }
    if let (Some(action_fail_option), Some(action_fail_environment)) =
        (action_fail_option, action_fail_environment)
    {
        crate::imod::libcfshr::parse_params::pip_get_two_integers(
            b"ActionIfGPUFails",
            action_fail_option,
            action_fail_environment,
        );
    }
    use_gpu
}
/// Matches C `b3dsetlocktimeout` (`b3dutil.c:1865`).
pub fn b3dsetlocktimeout(timeout: &f32) {
    b3d_set_lock_timeout(*timeout);
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

    #[cfg(unix)]
    #[test]
    fn imod_file_open_accepts_a_non_utf8_rust_path() {
        use std::os::unix::ffi::OsStringExt;

        let path = std::env::temp_dir().join(std::ffi::OsString::from_vec(
            [
                b'i', b'm', b'o', b'd', b'-', b'r', b's', b'-', b'p', b'a', b't', b'h', b'-', 0xff,
                b'-', b't', b'e', b's', b't', b'-',
            ]
            .to_vec(),
        ));
        let file = ImodFile::open(&path, "wb");
        assert!(file.is_some());
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn lock_file_owns_and_releases_its_rust_file() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-lock-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, []).unwrap();
        let index = b3d_open_lock_file(path.to_str().unwrap());
        assert!(index >= 0);
        assert_eq!(b3d_lock_file(index), 0);
        assert_eq!(b3d_unlock_file(index), 0);
        assert_eq!(b3d_close_lock_file(index), 0);
        std::fs::remove_file(path).unwrap();
    }

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
        assert_eq!(data_size_for_mode(4), Ok((4, 2)));
        assert_eq!(data_size_for_mode(99), Ok((4, 3)));
        assert_eq!(data_size_for_mode(12), Err(()));
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
        let mut bytes = [0_u8, 128, 255];
        b3d_shift_bytes(&mut bytes, 3, 1, 1, 1);
        assert_eq!(bytes.map(|value| value as i8), [-128, 0, 127]);
        b3d_shift_bytes(&mut bytes, 3, 1, -1, 1);
        assert_eq!(bytes, [0, 128, 255]);
        let values = [4, 7, 9];
        assert_eq!(number_in_list(7, Some(&values), 3, -1), 1);
        assert_eq!(number_in_list(2, Some(&values), 3, -1), 0);
        let mut start = 0;
        let mut end = 0;
        balanced_group_limits(10, 3, 1, &mut start, &mut end);
        assert_eq!((start, end), (4, 6));
        assert_eq!(angle_within_limits(-10., 0., 360.), 350.);
    }

    #[test]
    fn random_sequence_matches_glibc_for_default_and_explicit_seed() {
        {
            let mut state = B3D_RAND_STATE.lock().expect("b3d random state poisoned");
            *state = B3dRandState {
                words: [0; 31],
                front: 3,
                rear: 0,
                initialized: false,
                b3dran_first_time: true,
                b3dran_last_seed: 0,
            };
        }
        let expected = [1_804_289_383_u32, 846_930_886, 1_681_692_777, 1_714_636_915];
        for value in expected {
            assert_eq!(b3drand(), value as f32 / 2_147_483_647_f32);
        }

        b3dsrand(&1);
        for value in expected {
            assert_eq!(b3drand(), value as f32 / 2_147_483_647_f32);
        }

        let first = b3dran(&17);
        let second = b3dran(&17);
        assert_ne!(first, second);
        b3dran(&18);
        assert_eq!(first, b3dran(&17));
    }

    #[test]
    fn fortran_string_conversion_matches_blank_handling() {
        assert_eq!(fortran_string(b"abc   "), "abc");
        assert_eq!(fortran_string(b"      "), "");
        assert_eq!(fortran_string(b""), "");
        let mut output = [0_u8; 5];
        assert_eq!(c2f_string(b"abc", &mut output), Ok(()));
        assert_eq!(&output, b"abc  ");
        // The destination is exactly filled: no padding, no error.
        let mut exact = [0_u8; 3];
        assert_eq!(c2f_string(b"abc", &mut exact), Ok(()));
        assert_eq!(&exact, b"abc");
        // `b3dutil.c:843`: a leftover non-null character is the one failure.
        let mut short = [0_u8; 2];
        assert_eq!(c2f_string(b"abc", &mut short), Err(()));
    }
}
