//! Translation of `IMOD/3dmod/imod_assistant.cpp` and `imod_assistant.h`.
//!
//! `QProcess`, `QApplication::processEvents`, and `QMessageBox` are Qt
//! boundaries.  The process launch and Assistant remote-control protocol are
//! retained with `std::process`; callers can install [`ImodAssistant::error`]
//! to receive the source `error(const QString &)` signal.
#![allow(dead_code)]

use std::fs;
use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};

use crate::imod::libcfshr::b3dutil::{b3d_milli_sleep, imod_dir_or_default};

/// `QProcess::ExitStatus` used by `ImodAssistant::assistantExited`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AssistantExitStatus {
    #[default]
    NormalExit,
    CrashExit,
}

/// `ImodAssistant` (`imod_assistant.h`).
///
/// The `m_*` fields retain the original object's state.  `assistant` and
/// `assistant_input` are the native-process counterpart of `QProcess`; Qt
/// message boxes and the event loop remain an explicit frontend boundary.
pub struct ImodAssistant {
    pub m_path: String,
    pub m_qhc: String,
    pub m_imod_dir: String,
    pub m_prefix: String,
    pub m_assumed_imod: i32,
    pub m_keep_side_bar: bool,
    pub m_assistant: Option<Child>,
    pub m_assistant_input: Option<ChildStdin>,
    pub m_title: String,
    pub m_exiting: bool,
    /// Source signal `error(const QString &)`.  A Qt signal connection maps to
    /// this callback; absent one, the Qt/message-box boundary reports stderr.
    pub error: Option<fn(&str)>,
}

/// `ImodAssistant()`: construct an owned help-process controller.
#[allow(clippy::too_many_arguments)]
pub fn imod_assistant(
    path: &str,
    qhc_file: Option<&str>,
    message_title: Option<&str>,
    absolute: bool,
    keep_side_bar: bool,
    prefix: Option<&str>,
    pref_absolute: bool,
) -> ImodAssistant {
    ImodAssistant::new(
        path,
        qhc_file,
        message_title,
        absolute,
        keep_side_bar,
        prefix,
        pref_absolute,
    )
}

/// `~ImodAssistant()`: terminate an active helper and release its pipes.
pub fn free_imod_assistant(mut assistant: ImodAssistant) {
    assistant.close();
}

impl ImodAssistant {
    /// `ImodAssistant::ImodAssistant`.
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
        let m_imod_dir = unsafe { imod_dir_or_default(Some(&mut m_assumed_imod)) };

        // The Windows standalone-directory fallback in the source is a
        // platform-specific Qt file check.
        #[cfg(windows)]
        let m_imod_dir = if m_assumed_imod != 0 && !std::path::Path::new(&m_imod_dir).exists() {
            if std::path::Path::new(r"C:\\Program Files\\IMOD").exists() {
                r"C:\\Program Files\\IMOD".to_owned()
            } else if std::path::Path::new(r"C:\\Program Files\\3dmod").exists() {
                r"C:\\Program Files\\3dmod".to_owned()
            } else {
                m_imod_dir
            }
        } else {
            m_imod_dir
        };

        // `setenv("QT_PLUGIN_PATH", ...)` in the source is retained.  This
        // affects Qt Assistant and is intentionally process-global.
        let mut plug_str = format!("{m_imod_dir}/lib/imodplug/");
        if let Some(qt_plug) = std::env::var_os("QT_PLUGIN_PATH") {
            plug_str.push(std::path::MAIN_SEPARATOR);
            plug_str.push_str(&qt_plug.to_string_lossy());
        }
        unsafe { std::env::set_var("QT_PLUGIN_PATH", plug_str) };

        let m_path = if absolute {
            path.to_owned()
        } else {
            format!("{m_imod_dir}/{path}")
        };
        let mut qhc = qhc_file.unwrap_or_default().to_owned();
        if qhc == "IMOD.adp" {
            qhc = "IMOD.qhc".to_owned();
        }
        let m_qhc = format!("{m_path}/{qhc}");
        let mut m_prefix = "qthelp://bl3demc/IMOD/".to_owned();
        if let Some(prefix) = prefix {
            m_prefix = if pref_absolute {
                prefix.to_owned()
            } else {
                format!("{m_prefix}{prefix}")
            };
        }
        Self {
            m_path,
            m_qhc,
            m_imod_dir,
            m_prefix,
            m_assumed_imod,
            m_keep_side_bar: keep_side_bar,
            m_assistant: None,
            m_assistant_input: None,
            m_title: message_title
                .map(|title| format!("{title} Error"))
                .unwrap_or_default(),
            m_exiting: false,
            error: None,
        }
    }

    /// Destructor body from `ImodAssistant::~ImodAssistant`.
    pub fn close(&mut self) {
        self.m_exiting = true;
        self.m_assistant_input.take();
        if let Some(assistant) = self.m_assistant.as_mut() {
            // `terminate()` maps to terminating the native child process.
            let _ = assistant.kill();
        }
        self.m_assistant.take();
    }

    /// `ImodAssistant::showPage`.
    ///
    /// Returns zero on a successful Assistant start/remote command, `-1` for
    /// a missing collection file, and one when a native process boundary fails.
    pub fn show_page(&mut self, page: &str) -> i32 {
        let mut full_path = self.m_prefix.clone();
        if !full_path.ends_with('/') {
            full_path.push('/');
        }
        full_path.push_str(page);

        if !std::path::Path::new(&self.m_qhc).exists() {
            let mut file_only = format!("Cannot find help collection file: {}", self.m_qhc);
            if self.m_assumed_imod != 0 {
                file_only.push_str(&format!(
                    "\nThis is probably because IMOD_DIR is not defined\nand was assumed to be {}",
                    self.m_imod_dir
                ));
            }
            if !self.m_title.is_empty() {
                // `QMessageBox::warning` is a frontend/Qt boundary.
                eprintln!("{}: {file_only}", self.m_title);
            }
            self.assistant_error(&file_only);
            return -1;
        }

        let mut send_twice = false;
        if self.m_assistant.is_none() {
            #[cfg(windows)]
            let ass_path = {
                let bin = format!("{}/bin", self.m_imod_dir);
                if std::path::Path::new(&format!("{bin}/assistant.exe")).exists() {
                    format!("{bin}/assistant")
                } else if std::path::Path::new(&format!("{}/assistant.exe", self.m_imod_dir))
                    .exists()
                {
                    format!("{}/assistant", self.m_imod_dir)
                } else {
                    "assistant".to_owned()
                }
            };
            #[cfg(target_os = "macos")]
            let ass_path = format!(
                "{}/qtlib/Assistant.app/Contents/MacOS/Assistant",
                self.m_imod_dir
            );
            #[cfg(all(not(windows), not(target_os = "macos")))]
            let ass_path = {
                let bundled = format!("{}/qtlib/assistant", self.m_imod_dir);
                if fs::metadata(&bundled).is_ok() {
                    bundled
                } else {
                    "assistant".to_owned()
                }
            };
            let show_hide = if self.m_keep_side_bar {
                "-show"
            } else {
                "-hide"
            };
            let mut command = Command::new(ass_path);
            command
                .arg("-collectionFile")
                .arg(&self.m_qhc)
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
            if self.m_keep_side_bar {
                command.arg("-activate").arg("contents");
            } else {
                command.arg("-hide").arg("bookmarks");
            }
            match command.spawn() {
                Ok(mut assistant) => {
                    self.m_assistant_input = assistant.stdin.take();
                    self.m_assistant = Some(assistant);
                    send_twice = true;
                }
                Err(error) => {
                    self.assistant_error(&format!("Unable to run Qt Assistant: {error}"));
                    return 1;
                }
            }
        }

        if self.m_keep_side_bar {
            // The translated crate targets current Qt behavior (`QT_VERSION >=
            // 0x040700`); the old `expandToc 0` branch is obsolete.
            full_path.push_str("; expandToc 1;");
        }
        if let Some(input) = self.m_assistant_input.as_mut() {
            if write!(input, "setSource {full_path}\0\n").is_err() || input.flush().is_err() {
                self.assistant_error("Unable to send remote-control command to Qt Assistant");
                return 1;
            }
            if send_twice {
                for _ in 0..2 {
                    // Source calls QApplication::processEvents then b3dMilliSleep(400).
                    // Event processing is a Qt boundary; delay/protocol are retained.
                    b3d_milli_sleep(400);
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

    /// `ImodAssistant::assistantExited`.
    pub fn assistant_exited(&mut self, exit_code: i32, exit_status: AssistantExitStatus) {
        self.m_assistant_input.take();
        self.m_assistant.take();
        if self.m_exiting {
            return;
        }
        let full_msg = if exit_status != AssistantExitStatus::NormalExit {
            "Abnormal exit trying to run Qt Assistant".to_owned()
        } else if exit_code != 0 {
            format!("Qt Assistant exited with an error (return code {exit_code})")
        } else {
            return;
        };
        // The source deliberately suppresses this signal on Mac.
        #[cfg(not(target_os = "macos"))]
        self.assistant_error(&full_msg);
        if !self.m_title.is_empty() {
            // `QMessageBox::warning` is a Qt boundary.
            eprintln!("{}: {full_msg}", self.m_title);
        }
    }

    /// Source signal `ImodAssistant::error`.
    pub fn assistant_error(&self, message: &str) {
        if let Some(error) = self.error {
            error(message);
        } else {
            eprintln!("WARNING: Qt Assistant generated error: {message}");
        }
    }
}

impl Drop for ImodAssistant {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_collection_returns_source_error_code() {
        let mut assistant = ImodAssistant {
            m_path: "/does-not-exist".into(),
            m_qhc: "/does-not-exist/IMOD.qhc".into(),
            m_imod_dir: "/usr/local/IMOD".into(),
            m_prefix: "qthelp://bl3demc/IMOD/3dmodHelp".into(),
            m_assumed_imod: 0,
            m_keep_side_bar: false,
            m_assistant: None,
            m_assistant_input: None,
            m_title: String::new(),
            m_exiting: false,
            error: None,
        };
        assert_eq!(assistant.show_page("help.html"), -1);
    }

    #[test]
    fn normal_assistant_exit_has_no_error() {
        let mut assistant = ImodAssistant {
            m_path: String::new(),
            m_qhc: String::new(),
            m_imod_dir: String::new(),
            m_prefix: String::new(),
            m_assumed_imod: 0,
            m_keep_side_bar: false,
            m_assistant: None,
            m_assistant_input: None,
            m_title: String::new(),
            m_exiting: false,
            error: None,
        };
        assistant.assistant_exited(0, AssistantExitStatus::NormalExit);
        assert!(assistant.m_assistant.is_none());
    }

    #[test]
    fn lifecycle_facades_own_the_unstarted_helper() {
        let assistant = imod_assistant("/does-not-exist", None, None, true, false, None, false);
        assert!(assistant.m_assistant.is_none());
        free_imod_assistant(assistant);
    }
}
