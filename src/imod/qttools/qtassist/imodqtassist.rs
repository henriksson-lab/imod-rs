//! Translation of `IMOD/qttools/qtassist/imodqtassist.{h,cpp}`.
//!
//! The qmake project also copies `IMOD/3dmod/imod_assistant.{h,cpp}` into this
//! directory before compilation.  `ImodAssistant` below is consequently part
//! of this executable's source closure.  The Qt Assistant process itself is an
//! external boundary: `QProcess` becomes `std::process::Command`, and the Qt
//! event loop that delivers `QProcess::finished` and the 50 ms timer becomes
//! the polling loop in `imodqtassist`.
//!
//! The reference is built with `-DQT_THREAD_SUPPORT` (qmake `DEFINES`) and
//! against Qt 5.15, so the thread arm of `timerEvent` is the live one, and the
//! `QT_VERSION >= 0x050f00` arm of `main` connects a signal named
//! `errorOccurred` that `ImodAssistant` does not have (`imod_assistant.h:26`
//! declares `error`).  The connection fails at run time (Qt prints
//! `QObject::connect: No such signal ImodAssistant::errorOccurred(...)`), so
//! `AssistantListener::assistantError` is never reached and every
//! `emit error(...)` in `ImodAssistant` goes nowhere.  That is what the
//! reference binary does, measured.  Fixed in translation (BUGS.md): the
//! connection is evidently meant to reach the listener, so `imodqtassist`
//! sets `ImodAssistant::error_connected` and the errors are printed as
//! `WARNING: Qt Assistant generated error: ...`.
//!
//! Also fixed (BUGS.md): when the assistant exits during the page sends,
//! `assistantExited` deletes the `QProcess` and native then writes through
//! the deleted object (`imod_assistant.cpp:199-222`, segfault, rc 139); here
//! the exit is taken and nothing more is written (see `show_page`).

use std::io::{self, BufRead, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use crate::imod::libcfshr::b3dutil::{b3d_physical_memory, imod_dir_or_default};
use crate::imod::libcfshr::coresprocsthreads::num_cores_and_logical_procs;

/// `TIMER_INTERVAL` (`imodqtassist.cpp:37`), milliseconds.
const TIMER_INTERVAL: u64 = 50;
/// `MAX_LINE` (`imodqtassist.cpp:38`).
const MAX_LINE: usize = 1024;

/// The file statics `threadLine`, `gotLine` and `lineLen`
/// (`imodqtassist.cpp:39-41`), behind the static `QMutex mutex` (`:43`).
struct ThreadShared {
    thread_line: [u8; MAX_LINE],
    got_line: bool,
    line_len: i32,
}

static MUTEX: Mutex<ThreadShared> = Mutex::new(ThreadShared {
    thread_line: [0; MAX_LINE],
    got_line: false,
    line_len: 0,
});

/// C++ `AssistantListener` (`imodqtassist.h`).
pub struct AssistantListener {
    m_warned: bool,
}

impl AssistantListener {
    /// C++ `AssistantListener::AssistantListener` (`imodqtassist.h`).
    pub fn new() -> Self {
        Self { m_warned: false }
    }

    /// C++ `AssistantListener::assistantError` (`imodqtassist.cpp:180`).
    ///
    /// Unreachable in the reference build (see the module comment); reached
    /// here since the connection is fixed.
    pub fn assistant_error(&self, msg: &str) {
        let mut err = io::stderr();
        let _ = write!(err, "WARNING: Qt Assistant generated error: {msg}\n");
        let _ = err.flush();
    }

    /// C++ `AssistantListener::timerEvent` (`imodqtassist.cpp:189`),
    /// `QT_THREAD_SUPPORT` arm.  Returns `false` where the source calls
    /// `QApplication::exit(0)`.
    pub fn timer_event(&mut self, imod_help: &mut ImodAssistant) -> bool {
        let mut line = [0_u8; MAX_LINE];
        let err: i32;

        // If there is thread support, see if the thread got a line
        // Copy the line while we have the mutex
        let got_one;
        let line_len;
        let quit;
        {
            let mut shared = MUTEX.lock().unwrap_or_else(|e| e.into_inner());
            got_one = shared.got_line;
            shared.got_line = false;
            if got_one {
                let len = shared
                    .thread_line
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(MAX_LINE);
                line[..len].copy_from_slice(&shared.thread_line[..len]);
            }
            // `lineLen` and `threadLine[0]` are read below without the mutex in
            // the source; the thread does not write them again until
            // `gotLine` has been cleared, so reading them here is the same.
            line_len = shared.line_len;
            quit = shared.line_len == 1 && shared.thread_line[0] == b'q';
        }
        if !got_one {
            return true;
        }

        // Exit if blank line
        if quit {
            return false;
        }
        if line_len == 0 {
            return true;
        }

        // Otherwise show page and report results ( err > 0 not possible any more)
        let len = line.iter().position(|&c| c == 0).unwrap_or(MAX_LINE);
        let page = &line[..len];
        err = imod_help.show_page(page);
        let mut stderr = io::stderr();
        if err > 0 {
            let _ = stderr.write_all(b"WARNING: Page ");
            let _ = stderr.write_all(page);
            let _ = stderr.write_all(b" not found\n");
        } else {
            let _ = stderr.write_all(b"Page ");
            let _ = stderr.write_all(page);
            let _ = stderr.write_all(b" displayed\n");
        }
        if err < 0 && !self.m_warned {
            let _ = stderr.write_all(b"WARNING: qhc file not found\n");
            self.m_warned = true;
        }

        let _ = stderr.flush();
        true
    }
}

impl Default for AssistantListener {
    fn default() -> Self {
        Self::new()
    }
}

/// C++ `AssistantThread` (`imodqtassist.h`).
pub struct AssistantThread;

impl AssistantThread {
    /// C++ `AssistantThread::run` (`imodqtassist.cpp:265`).
    ///
    /// The source's `readLine(threadLine)` writes the static buffer outside
    /// the mutex; the main thread only copies it while `gotLine` is set, which
    /// is when this thread is not reading, so reading into a local and copying
    /// it in under the lock is the same exchange.
    pub fn run(&self) {
        let mut got_one;
        loop {
            got_one = MUTEX.lock().unwrap_or_else(|e| e.into_inner()).got_line;

            // Do not go for another line until the main thread has cleared the flag
            if got_one {
                thread::sleep(Duration::from_micros(20000));
                continue;
            }

            let mut thread_line = [0_u8; MAX_LINE];
            let line_len = read_line(&mut thread_line);
            let mut shared = MUTEX.lock().unwrap_or_else(|e| e.into_inner());
            shared.thread_line = thread_line;
            shared.line_len = line_len;
            shared.got_line = true;
            drop(shared);
            if line_len == 1 && thread_line[0] == b'q' {
                return;
            }
        }
    }
}

/// The copied C++ `ImodAssistant` class (`IMOD/3dmod/imod_assistant.cpp`).
pub struct ImodAssistant {
    /// `mAssistant != NULL`.
    m_assistant: bool,
    /// The running `QProcess`, when it started.
    child: Option<Child>,
    child_input: Option<ChildStdin>,
    m_path: String,
    m_qhc: String,
    m_imod_dir: String,
    m_prefix: String,
    m_title: String,
    m_assumed_imod: i32,
    m_keep_side_bar: bool,
    m_exiting: bool,
    /// Whether anything is connected to the `error` signal; see the module
    /// comment for why this is `false` in `imodqtassist`.
    error_connected: bool,
}

impl ImodAssistant {
    /// C++ `ImodAssistant::ImodAssistant` (`imod_assistant.cpp:47`).
    pub fn new(
        path: &str,
        qhc_file: Option<&str>,
        message_title: Option<&str>,
        absolute: bool,
        keep_side_bar: bool,
        prefix: Option<&str>,
        pref_absolute: bool,
    ) -> Self {
        let mut m_assumed_imod = 0;
        let mut m_title = String::new();
        if let Some(message_title) = message_title {
            m_title = format!("{message_title} Error");
        }

        // Get IMOD_DIR or fallback if necessary
        let m_imod_dir = imod_dir_or_default(Some(&mut m_assumed_imod));

        // For Qt 5 on Mac (as far as we know) assistant needs the platform plugin, and
        // this is also the case for accessing image plugins in windows and Mac, and setting
        // the imodplug directory on the QT_PLUGIN_PATH will provide this
        // `QDir::separator()` is '/' here: the source appends the old value
        // after a slash, not a path-list colon.
        let qtplug = std::env::var_os("QT_PLUGIN_PATH");
        let mut plug_str = format!("{m_imod_dir}/lib/imodplug/");
        if let Some(qtplug) = qtplug {
            plug_str.push('/');
            plug_str.push_str(&qtplug.to_string_lossy());
        }
        // SAFETY: `setenv` in the source; `main` constructs this object
        // before it starts the reader thread (`imodqtassist.cpp:151,164`), so
        // no other thread is reading the environment.
        unsafe { std::env::set_var("QT_PLUGIN_PATH", &plug_str) };

        // Set up path to help files; either absolute or under IMOD_DIR
        let m_path = if absolute {
            path.to_owned()
        } else {
            format!("{m_imod_dir}/{path}")
        };
        let mut m_qhc = qhc_file.unwrap_or("").to_owned();
        if m_qhc == "IMOD.adp" {
            m_qhc = "IMOD.qhc".to_owned();
        }
        m_qhc = format!("{m_path}/{m_qhc}");
        let mut m_prefix = "qthelp://bl3demc/IMOD/".to_owned();
        if let Some(prefix) = prefix {
            if pref_absolute {
                m_prefix = prefix.to_owned();
            } else {
                m_prefix = format!("{m_prefix}{prefix}");
            }
        }
        Self {
            m_assistant: false,
            child: None,
            child_input: None,
            m_path,
            m_qhc,
            m_imod_dir,
            m_prefix,
            m_title,
            m_assumed_imod,
            m_keep_side_bar: keep_side_bar,
            m_exiting: false,
            error_connected: false,
        }
    }

    /// C++ `ImodAssistant::~ImodAssistant` (`imod_assistant.cpp:109`).
    ///
    /// `QProcess::terminate` sends SIGTERM and does not wait.
    pub fn close(&mut self) {
        self.m_exiting = true;
        if self.m_assistant {
            if let Some(child) = self.child.as_ref() {
                // SAFETY: plain kill(2) on the pid of a child this process
                // spawned and has not reaped.
                unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
            }
        }
    }

    /// C++ `ImodAssistant::showPage` (`imod_assistant.cpp:123`).
    pub fn show_page(&mut self, page: &[u8]) -> i32 {
        let mut full_path = self.m_prefix.clone();
        let mut send_twice = false;
        if !self.m_prefix.ends_with('/') {
            full_path.push('/');
        }
        full_path.push_str(&String::from_utf8_lossy(page));

        if !std::path::Path::new(&self.m_qhc).exists() {
            let mut file_only = format!("Cannot find help collection file: {}", self.m_qhc);
            if self.m_assumed_imod != 0 {
                file_only.push_str(&format!(
                    "\nThis is probably because IMOD_DIR is not defined\nand was assumed to be {}",
                    self.m_imod_dir
                ));
            }
            // `QMessageBox::warning(0, mTitle, fileOnly, ...)` when a title was
            // given; `imodqtassist` passes none.
            if !self.m_title.is_empty() {
                eprintln!("{}: {file_only}", self.m_title);
            }
            self.error(&file_only);
            return -1;
        }

        // Get the assistant object the first time
        if !self.m_assistant {
            // Open the assistant in qtlib if one exists, otherwise take the one on
            // path.
            let mut ass_path = format!("{}/qtlib", self.m_imod_dir);
            if !std::path::Path::new(&format!("{ass_path}/assistant")).exists() {
                ass_path = String::new();
            }
            if !ass_path.is_empty() {
                ass_path.push('/');
            }
            ass_path.push_str("assistant");
            self.m_assistant = true;
            let mut command = Command::new(&ass_path);

            // Undo the high-DPI scaling set up on program start on Linux
            if std::env::var_os("QT_SCALE_FACTOR").is_some() {
                command.env("QT_SCALE_FACTOR", "1.");
            }
            let showhide = if self.m_keep_side_bar {
                "-show"
            } else {
                "-hide"
            };
            command.args([
                "-collectionFile",
                self.m_qhc.as_str(),
                "-enableRemoteControl",
                "-showUrl",
                full_path.as_str(),
            ]);
            command.args([showhide, "contents", showhide, "index", showhide, "search"]);
            if self.m_keep_side_bar {
                command.args(["-activate", "contents"]);
            } else {
                command.args(["-hide", "bookmarks"]);
            }
            command.stdin(Stdio::piped());
            match command.spawn() {
                Ok(mut child) => {
                    self.child_input = child.stdin.take();
                    self.child = Some(child);
                }
                Err(_) => {
                    // A `QProcess` that fails to start delivers `finished`
                    // during the `processEvents` below, `assistantExited`
                    // deletes it, and the `QTextStream` then writes through
                    // the deleted object: the reference binary segfaults
                    // (rc 139) on the first page when no `assistant` can be
                    // run.  Fixed in translation (BUGS.md): the
                    // `assistantExited` step is taken and nothing is written
                    // (the error it emits is reported), returning 0.
                    self.assistant_exited(255, false);
                    return 0;
                }
            }
            // `mAssistant->waitForStarted(3000); QApplication::processEvents();`
            if self.poll_exited() {
                return 0;
            }
            send_twice = true;
        }

        // Want one level of the Table of contents; that entry was wrong for a long time
        if self.m_keep_side_bar {
            full_path.push_str("; expandToc 1;");
        }
        // `QTextStream str(mAssistant); str << ... << '\0' << endl;` -- a
        // write the stream cannot deliver is dropped silently.
        if let Some(input) = self.child_input.as_mut() {
            let _ = write!(input, "setSource {full_path}\0\n");
            let _ = input.flush();
        }

        // On Mac, multiple sends were needed to keep from getting multiple tabs,
        // or about:blank or Qt Assistant help page.  With sending the page on
        // startup, it was better but still needed another send to get the Toc right
        // The time delay was needed on Win laptop
        if send_twice {
            for _len in 0..2 {
                // `QApplication::processEvents()`; an exit delivered here
                // leaves the source writing through a deleted `QProcess`
                // (segfault in the reference), so the translation stops.
                if self.poll_exited() {
                    return 0;
                }
                thread::sleep(Duration::from_millis(400));
                if let Some(input) = self.child_input.as_mut() {
                    let _ = write!(input, "setSource {full_path}\0\n");
                    let _ = input.flush();
                }
            }
        }
        0
    }

    /// The Qt event loop's delivery of `QProcess::finished` to
    /// `assistantExited`: returns whether it was delivered.
    pub fn poll_exited(&mut self) -> bool {
        let status = match self.child.as_mut().map(|child| child.try_wait()) {
            Some(Ok(Some(status))) => status,
            _ => return false,
        };
        match status.code() {
            Some(code) => self.assistant_exited(code, true),
            None => self.assistant_exited(0, false),
        }
        true
    }

    /// C++ `ImodAssistant::assistantExited` (`imod_assistant.cpp:229`).
    pub fn assistant_exited(&mut self, exit_code: i32, normal_exit: bool) {
        let full_msg;
        self.child_input = None;
        self.child = None;
        self.m_assistant = false;
        if self.m_exiting {
            return;
        }
        if !normal_exit {
            full_msg = "Abnormal exit trying to run Qt Assistant".to_owned();
        } else if exit_code != 0 {
            full_msg = format!("Qt Assistant exited with an error (return code {exit_code})");
        } else {
            return;
        }

        self.error(&full_msg);
        if !self.m_title.is_empty() {
            eprintln!("{}: {full_msg}", self.m_title);
        }
    }

    /// C++ signal `ImodAssistant::error` (`imod_assistant.h:26`); delivered to
    /// `AssistantListener::assistantError` only when connected.
    pub fn error(&self, msg: &str) {
        if self.error_connected {
            AssistantListener::new().assistant_error(msg);
        }
    }
}

/// C++ static `usage` (`imodqtassist.cpp:48`).
pub fn usage() {
    print!("Usage: imodqtassist [options] path_to_help_collection_file\n");
    print!("   Options:\n");
    print!("\t-a\tpath_to_help_collection is absolute, not relative to $IMOD_DIR\n");
    print!(
        "\t-p file\tName of qhc (help collection) file (IMOD.adp \n \t\t   can be entered instead of IMOD.qhc)\n"
    );
    print!("\t-q pref\tPrefix to use in front of help page name\n");
    print!("\t-b\tPrefix is absolute, not appended to qthelp://bl3demc/IMOD/\n");
    print!("\t-k\tKeep the sidebar (do not hide it)\n");
    print!("\t-t\tReport ideal thread count (number of processors) and exit\n");
    print!("\t-d\tReport resolution of desktop in dots/inch (DPI) and exit\n");
    print!("\t-db\tJust output X resolution of desktop in dots/inch (DPI) and exit\n");
    let _ = io::stdout().flush();
}

/// C++ static `readLine` (`imodqtassist.cpp:289`): `fgets` of at most
/// `MAX_LINE - 1` bytes, trailing CR/LF stripped, `strlen` returned.
pub fn read_line(line: &mut [u8; MAX_LINE]) -> i32 {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut count = 0;
    while count < MAX_LINE - 1 {
        let byte = match input.fill_buf() {
            Ok(buf) if !buf.is_empty() => buf[0],
            _ => break,
        };
        input.consume(1);
        line[count] = byte;
        count += 1;
        if byte == b'\n' {
            break;
        }
    }
    if count == 0 {
        line[0] = 0x00;
        return 0;
    }
    line[count] = 0x00;
    let mut len = line.iter().position(|&c| c == 0).unwrap_or(MAX_LINE);
    while len > 0 && (line[len - 1] == b'\n' || line[len - 1] == b'\r') {
        line[len - 1] = 0x00;
        len -= 1;
    }
    len as i32
}

/// C++ `main` from `imodqtassist.cpp:65`.
pub fn imodqtassist(argv: &[String]) -> i32 {
    let argc = argv.len();
    let mut ind = 1;
    let mut qhc = None;
    let mut abs_path = false;
    let mut keep_bar = false;
    let mut pref_abs = false;
    let mut prefix = None;

    if argc == 2 && argv[ind] == "-t" {
        let counts = num_cores_and_logical_procs();
        // `QThread::idealThreadCount()`: Qt 5 on Linux counts the CPUs in
        // the process's affinity mask.
        let ideal = thread::available_parallelism().map_or(1, usize::from);
        print!(
            "Qt ideal thread count = {ideal}   physical cores = {}   logical processors = {}   system memory {:.0} MB\n",
            counts.physical,
            counts.logical,
            b3d_physical_memory() / (1024. * 1024.)
        );
        let _ = io::stdout().flush();
        return 0;
    }

    // `diaSetQtLibraryPath(); QApplication qapp(argc, argv);
    // setlocale(LC_NUMERIC, "C");` -- no Qt application object here.

    if argc == 2 && argv[ind].starts_with("-d") {
        // `QApplication::primaryScreen()->physicalDotsPerInch*()`: there is
        // no Qt screen in this build, so the desktop resolution is not
        // available.  The reference prints it and exits 0.
        eprintln!("ERROR: Desktop DPI requires the Qt QScreen backend, which is not translated");
        return 1;
    }

    if argc < 2 {
        usage();
        return 1;
    }

    // Parse arguments
    while argc - ind > 1 {
        if argv[ind].as_bytes().first() == Some(&b'-') {
            match argv[ind].as_bytes().get(1).copied() {
                Some(b'a') => abs_path = true,
                Some(b'k') => keep_bar = true,
                Some(b'p') => {
                    ind += 1;
                    qhc = Some(argv[ind].as_str());
                }
                Some(b'q') => {
                    ind += 1;
                    prefix = Some(argv[ind].as_str());
                }
                Some(b'b') => pref_abs = true,
                Some(b'h') => {
                    usage();
                    return 0;
                }
                _ => {
                    eprintln!("ERROR: unknown argument {}", argv[ind]);
                    return 1;
                }
            }
        } else {
            eprintln!("ERROR: too many arguments");
            return 1;
        }
        ind += 1;
    }

    // start the help object
    let mut imod_help =
        ImodAssistant::new(&argv[ind], qhc, None, abs_path, keep_bar, prefix, pref_abs);

    // Get a listener for the errors and timer.  The `errorOccurred`
    // connection fails in the reference (module comment); fixed in
    // translation (BUGS.md): the listener is connected.
    let mut listener = AssistantListener::new();
    imod_help.error_connected = true;

    // If using threads, start a thread to read stdin
    let ass_thread = AssistantThread;
    thread::spawn(move || ass_thread.run());

    // Start timer to watch for input
    let retval = loop {
        thread::sleep(Duration::from_millis(TIMER_INTERVAL));
        imod_help.poll_exited();
        if !listener.timer_event(&mut imod_help) {
            break 0;
        }
    };

    // Upon exit, delete the help object to close help, and kill thread if
    // needed
    imod_help.close();
    retval
}

#[cfg(test)]
mod tests {
    use super::imodqtassist;

    #[test]
    fn help_option_returns_success() {
        assert_eq!(
            imodqtassist(&[
                "imodqtassist".to_owned(),
                "-h".to_owned(),
                "help-directory".to_owned(),
            ]),
            0
        );
    }
}
