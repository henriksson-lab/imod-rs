//! Rust mapping of `IMOD/qttools/sendevent/imodsendevent.h` and
//! `IMOD/qttools/sendevent/imodsendevent.cpp`.
//!
//! Qt's `QClipboard::dataChanged` signal is represented by polling the actual
//! X clipboard.  This preserves the wire protocol (clipboard text followed by
//! `Window_ID OK` or `Window_ID ERROR`) without pretending that an unavailable
//! Qt event loop is present.  The X11 boundary deliberately reports an error
//! when `xclip` or an X display is unavailable.

use std::ffi::CString;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct ImodSendEvent {
    pub win_id: i32,
    pub debug_out: bool,
    pub retry_limit: i32,
    pub retry_count: i32,
    pub time_str: String,
    pub cmd_str: String,
    pub last_cb_text: String,
    pub exit_code: Option<i32>,
    pub clipboard_owner: Option<Child>,
}

impl ImodSendEvent {
    /// Maps `ImodSendEvent::ImodSendEvent(int &, char **)`.
    pub fn new() -> Self {
        Self {
            win_id: 0,
            debug_out: false,
            retry_limit: 0,
            retry_count: 0,
            time_str: String::new(),
            cmd_str: String::new(),
            last_cb_text: String::new(),
            exit_code: None,
            clipboard_owner: None,
        }
    }

    /// Maps `ImodSendEvent::timerEvent(QTimerEvent *)`.
    pub fn timer_event(&mut self) {
        if self.retry_count + 1 < self.retry_limit {
            self.retry_count += 1;
            self.time_str.push(' ');
            let qstr = format!("{}{}", self.time_str, self.cmd_str);
            let text = Command::new("xclip")
                .args(["-selection", "clipboard", "-o"])
                .output()
                .ok()
                .and_then(|output| {
                    output
                        .status
                        .success()
                        .then(|| String::from_utf8(output.stdout).ok())
                        .flatten()
                })
                .unwrap_or_default();
            if text != self.last_cb_text {
                self.clipboard_changed(&text);
            }
            if self.debug_out {
                eprintln!("Imodsendevent - resending {qstr} ");
            }
            if let Some(mut owner) = self.clipboard_owner.take() {
                let _ = owner.kill();
                let _ = owner.wait();
            }
            let mut child = match Command::new("xclip")
                .args(["-selection", "clipboard", "-i", "-quiet"])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
            {
                Ok(child) => child,
                Err(error) => {
                    eprintln!("ERROR: imodsendevent - cannot access X clipboard: {error}");
                    self.exit_code = Some(2);
                    return;
                }
            };
            if let Some(mut stdin) = child.stdin.take() {
                if stdin.write_all(qstr.as_bytes()).is_err() {
                    eprintln!("ERROR: imodsendevent - cannot write X clipboard");
                    self.exit_code = Some(2);
                    let _ = child.kill();
                    let _ = child.wait();
                    return;
                }
            }
            self.last_cb_text = qstr;
            self.clipboard_owner = Some(child);
            return;
        }
        eprintln!("ERROR: imodsendevent - timeout before response received from target 3dmod");
        self.exit_code = Some(2);
    }

    /// Maps `ImodSendEvent::clipboardChanged()`.
    pub fn clipboard_changed(&mut self, text: &str) {
        if self.debug_out {
            eprintln!("Imodsendevent - clipboard = {text}");
        }
        if text.is_empty() {
            return;
        }
        let Some(index) = text.find(' ') else {
            return;
        };
        if index == 0 {
            return;
        }
        if text[..index].parse::<u32>().ok() != Some(self.win_id as u32) {
            return;
        }
        if text[index + 1..] == *"OK" {
            self.exit_code = Some(0);
            return;
        }
        if text[index + 1..] != *"ERROR" {
            return;
        }
        eprintln!("ERROR: imodsendevent - message received but error occurred executing it");
        self.exit_code = Some(3);
    }
}

impl Drop for ImodSendEvent {
    fn drop(&mut self) {
        if let Some(mut owner) = self.clipboard_owner.take() {
            let _ = owner.kill();
            let _ = owner.wait();
        }
    }
}

/// Maps `imodsendevent.cpp:main`.
pub fn imodsendevent(arguments: &[String]) -> i32 {
    let mut event = ImodSendEvent::new();
    let mut timeout = 5.0_f64;
    let mut arg_index = 1;
    while arg_index < arguments.len().saturating_sub(1) {
        let argument = &arguments[arg_index];
        if !argument.starts_with('-') {
            break;
        }
        match argument.as_bytes().get(1) {
            Some(b't') => {
                arg_index += 1;
                let Some(value) = arguments.get(arg_index) else {
                    eprintln!("ERROR: imodsendevent - invalid timeout entry ");
                    return 3;
                };
                let Ok(c_value) = CString::new(value.as_str()) else {
                    eprintln!("ERROR: imodsendevent - invalid timeout entry {value}");
                    return 3;
                };
                let mut end_pointer = std::ptr::null_mut();
                let parsed = unsafe { libc::strtod(c_value.as_ptr(), &mut end_pointer) };
                let consumed = unsafe { end_pointer.offset_from(c_value.as_ptr()) } as usize;
                if consumed < value.len() {
                    eprintln!("ERROR: imodsendevent - invalid timeout entry {value}");
                    return 3;
                }
                timeout = parsed;
            }
            Some(b'D') => event.debug_out = true,
            Some(b'h') => {
                println!("   Usage: imodsendevent [-t timeout] [-D] Window_ID action [arguments]");
                return 0;
            }
            _ => {
                eprintln!("ERROR: imodsendevent - invalid argument {argument}");
                return 3;
            }
        }
        arg_index += 1;
    }
    let num_args = arguments.len().saturating_sub(arg_index);
    if num_args < 2 {
        eprintln!(
            "ERROR: imodsendevent - Wrong number of arguments\n   Usage: imodsendevent [-t timeout] [-D] Window_ID action [arguments]"
        );
        return 3;
    }
    let window_argument = &arguments[arg_index];
    let Ok(c_window_argument) = CString::new(window_argument.as_str()) else {
        eprintln!("ERROR: imodsendevent - invalid characters in window ID entry {window_argument}");
        return 3;
    };
    let mut end_pointer = std::ptr::null_mut();
    event.win_id = unsafe { libc::strtol(c_window_argument.as_ptr(), &mut end_pointer, 10) as i32 };
    let consumed = unsafe { end_pointer.offset_from(c_window_argument.as_ptr()) } as usize;
    if consumed < window_argument.len() {
        eprintln!("ERROR: imodsendevent - invalid characters in window ID entry {window_argument}");
        return 3;
    }
    let action_argument = &arguments[arg_index + 1];
    let Ok(c_action_argument) = CString::new(action_argument.as_str()) else {
        eprintln!("ERROR: imodsendevent - invalid characters in action entry {action_argument}");
        return 3;
    };
    let mut end_pointer = std::ptr::null_mut();
    unsafe {
        libc::strtol(c_action_argument.as_ptr(), &mut end_pointer, 10);
    }
    let consumed = unsafe { end_pointer.offset_from(c_action_argument.as_ptr()) } as usize;
    if consumed < action_argument.len() {
        eprintln!("ERROR: imodsendevent - invalid characters in action entry {action_argument}");
        return 3;
    }

    // QTime::currentTime() only uses local minute, second, and millisecond in
    // the C++ source.  `localtime_r` is the direct C-library equivalent.
    let milliseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_millis() as i32;
    let mut local_time = unsafe { std::mem::zeroed::<libc::tm>() };
    let mut now: libc::time_t = 0;
    unsafe {
        libc::time(&mut now);
        libc::localtime_r(&now, &mut local_time);
    }
    let time_stamp = 60_000 * local_time.tm_min + 1_000 * local_time.tm_sec + milliseconds;
    event.time_str = format!("{window_argument} {time_stamp} ");
    event.cmd_str = arguments[arg_index + 1..].join(" ");
    let qstr = format!("{}{}", event.time_str, event.cmd_str);

    let interval = (1000.0 * timeout + 0.5) as i64;
    if event.debug_out {
        eprintln!("Imodsendevent sending: {qstr}");
    }
    let mut child = match Command::new("xclip")
        .args(["-selection", "clipboard", "-i", "-quiet"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            eprintln!("ERROR: imodsendevent - cannot access X clipboard: {error}");
            return 2;
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if stdin.write_all(qstr.as_bytes()).is_err() {
            eprintln!("ERROR: imodsendevent - cannot write X clipboard");
            let _ = child.kill();
            let _ = child.wait();
            return 2;
        }
    }
    event.last_cb_text = qstr;
    event.clipboard_owner = Some(child);
    std::thread::sleep(Duration::from_millis(10));
    if event
        .clipboard_owner
        .as_mut()
        .and_then(|owner| owner.try_wait().ok())
        .flatten()
        .is_some()
    {
        eprintln!("ERROR: imodsendevent - cannot access X clipboard (Qt/X clipboard unavailable)");
        return 2;
    }

    let started = Instant::now();
    loop {
        let text = Command::new("xclip")
            .args(["-selection", "clipboard", "-o"])
            .output()
            .ok()
            .and_then(|output| {
                output
                    .status
                    .success()
                    .then(|| String::from_utf8(output.stdout).ok())
                    .flatten()
            });
        let Some(text) = text else {
            eprintln!(
                "ERROR: imodsendevent - cannot read X clipboard (Qt/X clipboard unavailable)"
            );
            return 2;
        };
        if text != event.last_cb_text {
            event.clipboard_changed(&text);
        }
        if let Some(exit_code) = event.exit_code {
            return exit_code;
        }
        if interval > 0 && started.elapsed() >= Duration::from_millis(interval as u64) {
            event.timer_event();
            if let Some(exit_code) = event.exit_code {
                return exit_code;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use super::ImodSendEvent;

    #[test]
    fn clipboard_changed_accepts_only_matching_ok_and_error() {
        let mut event = ImodSendEvent::new();
        event.win_id = 17;
        event.clipboard_changed("17 OK");
        assert_eq!(event.exit_code, Some(0));
        event.exit_code = None;
        event.clipboard_changed("18 OK");
        assert_eq!(event.exit_code, None);
        event.clipboard_changed("17 ERROR");
        assert_eq!(event.exit_code, Some(3));
    }
}
