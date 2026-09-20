//! Translation of `IMOD/qttools/qtassist/imodqtassist.{h,cpp}`.
//!
//! The qmake project also copies `IMOD/3dmod/imod_assistant.{h,cpp}` into this
//! directory before compilation.  `ImodAssistant` below is consequently part
//! of this executable's source closure.  It retains the actual Qt Assistant
//! remote-control protocol instead of attempting to reproduce its GUI.

use std::fs;
use std::io::{self, BufRead, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use crate::imod::libcfshr::b3dutil::{b3d_physical_memory, imod_dir_or_default};
use crate::imod::libcfshr::coresprocsthreads::num_cores_and_logical_procs;

const TIMER_INTERVAL: Duration = Duration::from_millis(50);
const MAX_LINE: usize = 1024;

/// C++ `AssistantListener` (`imodqtassist.h`).
pub struct AssistantListener {
    warned: bool,
}

impl AssistantListener {
    /// C++ `AssistantListener::AssistantListener`.
    pub fn new() -> Self {
        Self { warned: false }
    }

    /// C++ `AssistantListener::assistantError`.
    pub fn assistant_error(&self, message: &str) {
        eprintln!("WARNING: Qt Assistant generated error: {message}");
    }

    /// C++ `AssistantListener::timerEvent`.
    pub fn timer_event(
        &mut self,
        receiver: &Receiver<(usize, String)>,
        imod_help: &mut ImodAssistant,
    ) -> bool {
        let (line_len, line) = match receiver.try_recv() {
            Ok(line) => line,
            Err(TryRecvError::Empty) => return true,
            Err(TryRecvError::Disconnected) => return false,
        };

        if line_len == 1 && line == "q" {
            return false;
        }
        if line_len == 0 {
            return true;
        }

        let error = imod_help.show_page(&line);
        if error > 0 {
            eprintln!("WARNING: Page {line} not found");
        } else {
            eprintln!("Page {line} displayed");
        }
        if error < 0 && !self.warned {
            eprintln!("WARNING: qhc file not found");
            self.warned = true;
        }
        true
    }
}

impl Default for AssistantListener {
    fn default() -> Self {
        Self::new()
    }
}

/// C++ `AssistantThread` (`imodqtassist.h`).
pub struct AssistantThread {
    sender: Sender<(usize, String)>,
}

impl AssistantThread {
    /// C++ `AssistantThread::AssistantThread`.
    pub fn new(sender: Sender<(usize, String)>) -> Self {
        Self { sender }
    }

    /// C++ `AssistantThread::run`.
    pub fn run(self) {
        let stdin = io::stdin();
        let mut input = stdin.lock();
        loop {
            let mut line = String::new();
            let line_len = read_line(&mut input, &mut line);
            let is_quit = line_len == 1 && line == "q";
            if self.sender.send((line_len, line)).is_err() || is_quit {
                return;
            }
        }
    }
}

/// The copied C++ `ImodAssistant` class (`IMOD/3dmod/imod_assistant.cpp`).
pub struct ImodAssistant {
    qhc: String,
    imod_dir: String,
    prefix: String,
    assumed_imod: i32,
    keep_side_bar: bool,
    assistant: Option<Child>,
    assistant_input: Option<ChildStdin>,
    title: String,
    exiting: bool,
}

impl ImodAssistant {
    /// C++ `ImodAssistant::ImodAssistant`.
    pub fn new(
        path: &str,
        qhc_file: Option<&str>,
        message_title: Option<&str>,
        absolute: bool,
        keep_side_bar: bool,
        prefix: Option<&str>,
        pref_absolute: bool,
    ) -> Self {
        let mut assumed_imod = 0;
        let imod_dir = unsafe { imod_dir_or_default(Some(&mut assumed_imod)) };
        let path = if absolute {
            path.to_owned()
        } else {
            format!("{imod_dir}/{path}")
        };
        let mut qhc = qhc_file.unwrap_or("").to_owned();
        if qhc == "IMOD.adp" {
            qhc = "IMOD.qhc".to_owned();
        }
        qhc = format!("{path}/{qhc}");
        let mut page_prefix = "qthelp://bl3demc/IMOD/".to_owned();
        if let Some(prefix) = prefix {
            page_prefix = if pref_absolute {
                prefix.to_owned()
            } else {
                format!("{page_prefix}{prefix}")
            };
        }
        Self {
            qhc,
            imod_dir,
            prefix: page_prefix,
            assumed_imod,
            keep_side_bar,
            assistant: None,
            assistant_input: None,
            title: message_title
                .map(|title| format!("{title} Error"))
                .unwrap_or_default(),
            exiting: false,
        }
    }

    /// C++ `ImodAssistant::~ImodAssistant`.
    pub fn close(&mut self) {
        self.exiting = true;
        self.assistant_input.take();
        if let Some(child) = self.assistant.as_mut() {
            let _ = child.kill();
        }
        self.assistant.take();
    }

    /// C++ `ImodAssistant::showPage`.
    pub fn show_page(&mut self, page: &str) -> i32 {
        let mut full_path = self.prefix.clone();
        if !full_path.ends_with('/') {
            full_path.push('/');
        }
        full_path.push_str(page);
        if fs::metadata(&self.qhc).is_err() {
            let mut message = format!("Cannot find help collection file: {}", self.qhc);
            if self.assumed_imod != 0 {
                message.push_str(&format!(
                    "\nThis is probably because IMOD_DIR is not defined\nand was assumed to be {}",
                    self.imod_dir
                ));
            }
            if !self.title.is_empty() {
                eprintln!("{}: {message}", self.title);
            }
            self.assistant_error(&message);
            return -1;
        }
        let mut send_twice = false;
        if self.assistant.is_none() {
            let bundled = format!("{}/qtlib/assistant", self.imod_dir);
            let executable = if fs::metadata(&bundled).is_ok() {
                bundled
            } else {
                "assistant".to_owned()
            };
            let show_hide = if self.keep_side_bar { "-show" } else { "-hide" };
            let mut command = Command::new(executable);
            command
                .arg("-collectionFile")
                .arg(&self.qhc)
                .arg("-enableRemoteControl")
                .arg("-showUrl")
                .arg(&full_path)
                .arg(show_hide)
                .arg("contents")
                .arg(show_hide)
                .arg("index")
                .arg(show_hide)
                .arg("search")
                .stdin(Stdio::piped());
            if self.keep_side_bar {
                command.arg("-activate").arg("contents");
            } else {
                command.arg("-hide").arg("bookmarks");
            }
            match command.spawn() {
                Ok(mut child) => {
                    self.assistant_input = child.stdin.take();
                    self.assistant = Some(child);
                    send_twice = true;
                }
                Err(error) => {
                    self.assistant_error(&format!("Unable to run Qt Assistant: {error}"));
                    return 1;
                }
            }
        }
        if self.keep_side_bar {
            full_path.push_str("; expandToc 1;");
        }
        if let Some(input) = self.assistant_input.as_mut() {
            if write!(input, "setSource {full_path}\0\n").is_err() || input.flush().is_err() {
                self.assistant_error("Unable to send remote-control command to Qt Assistant");
                return 1;
            }
            if send_twice {
                for _ in 0..2 {
                    thread::sleep(Duration::from_millis(400));
                    if write!(input, "setSource {full_path}\0\n").is_err() || input.flush().is_err()
                    {
                        self.assistant_error(
                            "Unable to send remote-control command to Qt Assistant",
                        );
                        return 1;
                    }
                }
            }
        }
        0
    }

    /// C++ `ImodAssistant::assistantExited`.
    pub fn assistant_exited(&mut self, exit_code: i32, normal_exit: bool) {
        self.assistant_input.take();
        self.assistant.take();
        if self.exiting {
            return;
        }
        let message = if !normal_exit {
            "Abnormal exit trying to run Qt Assistant".to_owned()
        } else if exit_code != 0 {
            format!("Qt Assistant exited with an error (return code {exit_code})")
        } else {
            return;
        };
        self.assistant_error(&message);
        if !self.title.is_empty() {
            eprintln!("{}: {message}", self.title);
        }
    }

    /// C++ signal `ImodAssistant::error`.
    pub fn assistant_error(&self, message: &str) {
        eprintln!("WARNING: Qt Assistant generated error: {message}");
    }
}

impl Drop for ImodAssistant {
    fn drop(&mut self) {
        self.close();
    }
}

/// C++ static `usage`.
pub fn usage() {
    println!("Usage: imodqtassist [options] path_to_help_collection_file");
    println!("   Options:");
    println!("\t-a\tpath_to_help_collection is absolute, not relative to $IMOD_DIR");
    println!("\t-p file\tName of qhc (help collection) file (IMOD.adp");
    println!(" \t\t   can be entered instead of IMOD.qhc)");
    println!("\t-q pref\tPrefix to use in front of help page name");
    println!("\t-b\tPrefix is absolute, not appended to qthelp://bl3demc/IMOD/");
    println!("\t-k\tKeep the sidebar (do not hide it)");
    println!("\t-t\tReport ideal thread count (number of processors) and exit");
    println!("\t-d\tReport resolution of desktop in dots/inch (DPI) and exit");
    println!("\t-db\tJust output X resolution of desktop in dots/inch (DPI) and exit");
}

/// C++ static `readLine`.
pub fn read_line<R: BufRead>(input: &mut R, line: &mut String) -> usize {
    line.clear();
    let mut bytes = Vec::new();
    match input.read_until(b'\n', &mut bytes) {
        Ok(0) | Err(_) => return 0,
        Ok(_) => {}
    }
    bytes.truncate(MAX_LINE - 1);
    while matches!(bytes.last(), Some(b'\n' | b'\r')) {
        bytes.pop();
    }
    *line = String::from_utf8_lossy(&bytes).into_owned();
    line.len()
}

/// C++ `main` from `imodqtassist.cpp`.
pub fn imodqtassist(arguments: &[String]) -> i32 {
    let mut index = 1;
    if arguments.len() == 2 && arguments[index] == "-t" {
        let counts = num_cores_and_logical_procs();
        let ideal = thread::available_parallelism().map_or(1, usize::from);
        println!(
            "Qt ideal thread count = {ideal}   physical cores = {}   logical processors = {}   system memory {:.0} MB",
            counts.physical,
            counts.logical,
            b3d_physical_memory() / (1024. * 1024.)
        );
        return 0;
    }
    // `-d` and `-db` require Qt's active QScreen.  Unlike the old Qt shim,
    // this executable deliberately does not manufacture a display value.
    if arguments.len() == 2 && arguments[index].starts_with("-d") {
        eprintln!("ERROR: Desktop DPI requires the Qt QScreen backend, which is not translated");
        return 1;
    }
    if arguments.len() < 2 {
        usage();
        return 1;
    }
    let mut qhc = None;
    let mut absolute_path = false;
    let mut keep_bar = false;
    let mut prefix_absolute = false;
    let mut prefix = None;
    while arguments.len() - index > 1 {
        let argument = &arguments[index];
        if !argument.starts_with('-') {
            eprintln!("ERROR: too many arguments");
            return 1;
        }
        match argument.as_bytes().get(1).copied() {
            Some(b'a') => absolute_path = true,
            Some(b'k') => keep_bar = true,
            Some(b'p') => {
                index += 1;
                if index >= arguments.len() {
                    eprintln!("ERROR: option -p requires an argument");
                    return 1;
                }
                qhc = Some(arguments[index].as_str());
            }
            Some(b'q') => {
                index += 1;
                if index >= arguments.len() {
                    eprintln!("ERROR: option -q requires an argument");
                    return 1;
                }
                prefix = Some(arguments[index].as_str());
            }
            Some(b'b') => prefix_absolute = true,
            Some(b'h') => {
                usage();
                return 0;
            }
            _ => {
                eprintln!("ERROR: unknown argument {argument}");
                return 1;
            }
        }
        index += 1;
    }
    let mut imod_help = ImodAssistant::new(
        &arguments[index],
        qhc,
        None,
        absolute_path,
        keep_bar,
        prefix,
        prefix_absolute,
    );
    let mut listener = AssistantListener::new();
    let (sender, receiver) = mpsc::channel();
    let assistant_thread = AssistantThread::new(sender);
    let reader = thread::spawn(move || assistant_thread.run());
    loop {
        if !listener.timer_event(&receiver, &mut imod_help) {
            break;
        }
        thread::sleep(TIMER_INTERVAL);
    }
    imod_help.close();
    let _ = reader.join();
    0
}

#[cfg(test)]
mod tests {
    use super::{imodqtassist, read_line};
    use std::io::Cursor;

    #[test]
    fn read_line_strips_source_line_endings() {
        let mut input = Cursor::new(b"page.html\r\n".to_vec());
        let mut line = String::new();
        assert_eq!(read_line(&mut input, &mut line), 9);
        assert_eq!(line, "page.html");
    }

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
