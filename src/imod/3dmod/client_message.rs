//! Translation of `IMOD/3dmod/client_message.cpp` and `client_message.h`.
//!
//! The message grammar, duplicate-stamp response, initial-load deferral, and
//! action/argument protocol are owned here.  Clipboard ownership, Qt timers,
//! stdin readiness, and calls into the live 3dmod/3dmodv viewer remain direct
//! boundaries: [`ClientMessageBoundary::call`] receives the original callee
//! name and its source arguments.
#![allow(dead_code, unused_variables)]

pub const MESSAGE_NO_ACTION: i32 = 0;
pub const MESSAGE_OPEN_MODEL: i32 = 1;
pub const MESSAGE_SAVE_MODEL: i32 = 2;
pub const MESSAGE_VIEW_MODEL: i32 = 3;
pub const MESSAGE_QUIT: i32 = 4;
pub const MESSAGE_RAISE_WINDOWS: i32 = 5;
pub const MESSAGE_MODEL_MODE: i32 = 6;
pub const MESSAGE_OPEN_KEEP_BW: i32 = 7;
pub const MESSAGE_OPEN_BEADFIXER: i32 = 8;
pub const MESSAGE_ONE_ZAP_OPEN: i32 = 9;
pub const MESSAGE_RUBBERBAND: i32 = 10;
pub const MESSAGE_OBJ_PROPERTIES: i32 = 11;
pub const MESSAGE_NEWOBJ_PROPERTIES: i32 = 12;
pub const MESSAGE_SLICER_ANGLES: i32 = 13;
pub const MESSAGE_PLUGIN_EXECUTE: i32 = 14;
pub const MESSAGE_OBJ_PROPS_2: i32 = 15;
pub const MESSAGE_NEWOBJ_PROPS_2: i32 = 16;
pub const MESSAGE_GHOST_MODE: i32 = 17;
pub const MESSAGE_ZAP_HQ_MODE: i32 = 18;
pub const MESSAGE_OPEN_DIALOGS: i32 = 19;
pub const MESSAGE_MODEL_CHANGED: i32 = 20;
pub const MESSAGE_OBJ_PROPS_3: i32 = 21;
pub const MESSAGE_NEWOBJ_PROPS_3: i32 = 22;
pub const MESSAGE_EDGE_FOR_MIDAS: i32 = 23;
pub const MESSAGE_MIDAS_SET_EDGE: i32 = 24;
pub const MESSAGE_MULTIZ_PANELS: i32 = 25;

pub const MESSAGE_BEADFIX_OPENFILE: i32 = 1;
pub const MESSAGE_BEADFIX_REREAD: i32 = 2;
pub const MESSAGE_BEADFIX_SEEDMODE: i32 = 3;
pub const MESSAGE_BEADFIX_AUTOCENTER: i32 = 4;
pub const MESSAGE_BEADFIX_DIAMETER: i32 = 5;
pub const MESSAGE_BEADFIX_OPERATION: i32 = 6;
pub const MESSAGE_BEADFIX_SKIPLIST: i32 = 7;
pub const MESSAGE_BEADFIX_DELALLSEC: i32 = 8;
pub const MESSAGE_BEADFIX_CLEARSKIP: i32 = 9;

pub const SPACE_KEEPER: &str = "%?@%*#!";
pub const STDIN_INTERVAL: i32 = 50;
pub const MAX_LINE: usize = 256;

/// Source state used by `ImodClipboard` guards and `ourWindowID`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClientMessageViewState {
    pub info_window_present: bool,
    pub imodv_closed: bool,
    pub imodv_standalone: bool,
    pub info_window_id: u32,
    pub imodv_window_id: u32,
}

/// Direct Qt/viewer/system boundary for this source unit.  `call` preserves
/// each source callee rather than introducing a parallel command vocabulary.
pub trait ClientMessageBoundary {
    fn state(&self) -> ClientMessageViewState;
    fn clipboard_text(&mut self) -> String;
    fn set_clipboard_text(&mut self, text: String);
    fn stderr(&mut self, text: &str);
    fn call(&mut self, callee: &'static str, args: &[String]) -> i32;
    /// `imodPlugMessage(App->cvi, &sMessageStrings, &arg)`.  The mutable
    /// index is intentional: plugins consume their variable-length payload
    /// through the same argument pointer used by the C++ call.
    fn imod_plug_message(&mut self, strings: &[String], arg: &mut usize) -> i32 {
        let _ = self.call("imodPlugMessage", &strings[*arg..]);
        0
    }
    fn schedule_timeout(&mut self, _milliseconds: i32) {}
    /// Returns a ready stdin line.  `None` is the source `select`/thread
    /// no-data case; `Some("")` is stdin disconnection.
    fn stdin_line(&mut self) -> Option<String> {
        None
    }
}

/// `QTimer` members of `ImodClipboard`.  Actual Qt timer delivery is the
/// boundary above; this records source start/stop state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClientTimer {
    pub interval: i32,
    pub active: bool,
}

/// Windows-only `StdinThread` (`client_message.h`).  It is retained on every
/// target so the paired source class has an auditable Rust counterpart.  The
/// native thread creation and mutex handoff are system-boundary operations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StdinThread;

impl StdinThread {
    /// `StdinThread::run`.
    pub fn run(&mut self, b: &mut dyn ClientMessageBoundary) {
        loop {
            let Some(line) = b.stdin_line() else { return };
            let line = read_line(&line);
            b.call("StdinThread.publishLine", std::slice::from_ref(&line));
            if line == "4" || line.is_empty() {
                return;
            }
        }
    }
}

/// `ImodClipboard` (`client_message.h`).
#[derive(Clone, Debug)]
pub struct ImodClipboard {
    pub m_clip_hack_timer: Option<ClientTimer>,
    pub m_stdin_timer: Option<ClientTimer>,
    pub m_handling: bool,
    pub m_exiting: i32,
    pub m_use_stdin: bool,
    pub m_disconnected: bool,
    pub m_deferred_handling: i32,
    pub m_saved_clipboard: String,
    message_action: i32,
    message_strings: Vec<String>,
    message_stamp: i32,
    initial_load: bool,
    last_response: i32,
}

/// `ImodClipboard()`: initialize clipboard or stdin protocol state through
/// the caller-provided Qt/viewer boundary.
pub fn imod_clipboard(
    use_stdin: bool,
    will_load_images: bool,
    boundary: &mut dyn ClientMessageBoundary,
) -> ImodClipboard {
    ImodClipboard::new(use_stdin, will_load_images, boundary)
}

impl ImodClipboard {
    /// `ImodClipboard::ImodClipboard`.
    pub fn new(use_stdin: bool, will_load_images: bool, b: &mut dyn ClientMessageBoundary) -> Self {
        let saved = if use_stdin {
            String::new()
        } else {
            b.clipboard_text()
        };
        Self {
            m_clip_hack_timer: None,
            m_stdin_timer: use_stdin.then_some(ClientTimer {
                interval: STDIN_INTERVAL,
                active: true,
            }),
            m_handling: false,
            m_exiting: 0,
            m_use_stdin: use_stdin,
            m_disconnected: false,
            m_deferred_handling: 0,
            m_saved_clipboard: saved,
            message_action: MESSAGE_NO_ACTION,
            message_strings: Vec::new(),
            message_stamp: -1,
            initial_load: will_load_images,
            last_response: 0,
        }
    }

    /// `ImodClipboard::startDisconnect` (the Windows thread request is a
    /// system boundary; other source platforms are no-ops).
    pub fn start_disconnect(&mut self, b: &mut dyn ClientMessageBoundary) {
        if cfg!(windows) && self.m_use_stdin {
            b.stderr("REQUEST STOP LISTENING\n");
        }
    }

    /// `ImodClipboard::waitForDisconnect`.  Thread waiting/termination is a
    /// Windows system boundary and succeeds on the non-Windows source path.
    pub fn wait_for_disconnect(&mut self, _b: &mut dyn ClientMessageBoundary) -> i32 {
        0
    }

    /// `ImodClipboard::disconnectedFromStderr`.
    pub fn disconnected_from_stderr(&self) -> bool {
        self.m_disconnected
    }

    /// `ImodClipboard::doneWithLoad`.
    pub fn done_with_load(&mut self, b: &mut dyn ClientMessageBoundary) {
        self.initial_load = false;
        if !self.message_strings.is_empty() {
            self.m_deferred_handling = -1;
            if self.m_handling {
                b.schedule_timeout(100);
            } else {
                self.clip_timeout(b);
            }
        }
    }

    /// `ImodClipboard::clipboardChanged`.
    pub fn clipboard_changed(&mut self, b: &mut dyn ClientMessageBoundary) {
        if self.m_handling || !self.handle_message(b) {
            return;
        }
        self.m_handling = true;
        b.schedule_timeout(10);
    }

    /// `ImodClipboard::clipTimeout`.
    pub fn clip_timeout(&mut self, b: &mut dyn ClientMessageBoundary) {
        if self.m_exiting != 0 {
            if self.initial_load && self.m_exiting < 2 {
                self.m_exiting += 1;
                b.schedule_timeout(50);
                return;
            }
            let s = b.state();
            b.call(
                if s.imodv_closed || !s.imodv_standalone {
                    "imod_quit"
                } else {
                    "imodvQuit"
                },
                &[],
            );
        } else {
            if self.m_deferred_handling < 0 {
                self.m_deferred_handling = 1;
                self.m_handling = true;
            }
            if self.execute_message(b) {
                self.m_exiting = 1;
                b.schedule_timeout(200);
            }
            self.m_handling = false;
        }
    }

    /// `ImodClipboard::clipHackTimeout`.
    pub fn clip_hack_timeout(&mut self, b: &mut dyn ClientMessageBoundary) {
        let text = b.clipboard_text();
        if text == self.m_saved_clipboard {
            return;
        }
        self.m_saved_clipboard = text;
        self.clipboard_changed(b);
    }

    /// `ImodClipboard::stdinTimeout`.
    pub fn stdin_timeout(&mut self, b: &mut dyn ClientMessageBoundary) {
        if self.m_handling {
            return;
        }
        let Some(mut text) = b.stdin_line() else {
            return;
        };
        if text.len() >= MAX_LINE {
            text.truncate(MAX_LINE - 1);
        }
        while text.ends_with(['\n', '\r']) {
            text.pop();
        }
        if text.is_empty() {
            if let Some(timer) = &mut self.m_stdin_timer {
                timer.active = false;
            }
            self.send_response(1, b);
            self.m_disconnected = true;
            return;
        }
        let tmp_strings = self.split_with_escaped_spaces(&text);
        if self.initial_load {
            self.message_strings.extend(tmp_strings);
        } else {
            self.message_strings = tmp_strings;
        }
        self.m_handling = true;
        b.schedule_timeout(10);
    }

    /// `ImodClipboard::handleMessage`.
    pub fn handle_message(&mut self, b: &mut dyn ClientMessageBoundary) -> bool {
        let state = b.state();
        if !state.info_window_present && state.imodv_closed {
            return false;
        }
        let text = b.clipboard_text();
        if text.is_empty() {
            return false;
        }
        let tmp_strings = self.split_with_escaped_spaces(&text);
        if tmp_strings.len() < 3 {
            return false;
        }
        if tmp_strings[0].parse::<u32>().unwrap_or(0) != self.our_window_id(b) {
            return false;
        }
        let new_stamp = tmp_strings[1].parse::<i32>().unwrap_or(0);
        if new_stamp == self.message_stamp {
            self.send_response(-1, b);
            return false;
        }
        self.message_stamp = new_stamp;
        if !self.initial_load {
            self.message_strings.clear();
        }
        self.message_strings.extend_from_slice(&tmp_strings[2..]);
        true
    }

    /// `ImodClipboard::splitWithEscapedSpaces`.
    pub fn split_with_escaped_spaces(&self, text: &str) -> Vec<String> {
        text.replace("\\\\ ", SPACE_KEEPER)
            .split(' ')
            .filter(|s| !s.is_empty())
            .map(|s| s.replace(SPACE_KEEPER, " "))
            .collect()
    }

    /// `ImodClipboard::executeMessage`.
    pub fn execute_message(&mut self, b: &mut dyn ClientMessageBoundary) -> bool {
        const REQUIRED_ARGS: [usize; 26] = [
            0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 5, 0, 2, 4, 4, 3, 1, 1, 0, 6, 6, 0, 4, 4,
        ];
        let mut succeeded = 1;
        let mut arg = 0;
        let num_args = self.message_strings.len();
        while arg < num_args {
            self.message_action = self.message_strings[arg].parse().unwrap_or(0);
            let action = self.message_action;
            if action >= 0
                && (action as usize) < REQUIRED_ARGS.len()
                && arg + REQUIRED_ARGS[action as usize] >= num_args
            {
                b.stderr(&format!(
                    "imodExecuteMessage: not enough values sent with action command {action}\n"
                ));
                succeeded = 0;
                break;
            }
            if self.initial_load {
                if action >= 0 && (action as usize) < REQUIRED_ARGS.len() {
                    arg += REQUIRED_ARGS[action as usize];
                }
                if arg < num_args - 1 && action == MESSAGE_MODEL_MODE {
                    arg += 1;
                }
                if action == MESSAGE_PLUGIN_EXECUTE || action == MESSAGE_QUIT {
                    arg = num_args;
                }
                arg += 1;
                continue;
            }
            let s = b.state();
            if s.imodv_closed || !s.imodv_standalone {
                match action {
                    MESSAGE_OPEN_MODEL | MESSAGE_OPEN_KEEP_BW => {
                        let name = self.message_strings[arg + 1].clone();
                        succeeded = -1;
                        self.send_response(1, b);
                        b.call("inputRaiseWindows", &[]);
                        b.call(
                            if action == MESSAGE_OPEN_KEEP_BW {
                                "openModel.keepBW"
                            } else {
                                "openModel"
                            },
                            &[name],
                        );
                        arg += 1;
                    }
                    MESSAGE_SAVE_MODEL => {
                        succeeded = -1;
                        self.send_response(1, b);
                        b.call("SaveModel", &[]);
                    }
                    MESSAGE_VIEW_MODEL => {
                        b.call("imod_autosave", &[]);
                        b.call("inputRaiseWindows", &[]);
                        b.call("imodv_open", &[]);
                    }
                    MESSAGE_QUIT => {
                        arg = num_args;
                    }
                    MESSAGE_RAISE_WINDOWS => {
                        b.call("inputRaiseWindows", &[]);
                    }
                    MESSAGE_MODEL_MODE => {
                        let value = if arg < num_args - 1 {
                            arg += 1;
                            self.message_strings[arg].clone()
                        } else {
                            "1".into()
                        };
                        let movie_val = value.parse::<i32>().unwrap_or(0);
                        b.call(
                            "imod_set_mmode",
                            &[if movie_val > 0 { "1" } else { "0" }.into()],
                        );
                        if movie_val < 0 {
                            b.call(
                                "imodMovieXYZT",
                                &[
                                    ((-movie_val & 1) != 0) as i32,
                                    ((-movie_val & 2) != 0) as i32,
                                    ((-movie_val & 4) != 0) as i32,
                                    ((-movie_val & 8) != 0) as i32,
                                ]
                                .map(|value| value.to_string()),
                            );
                        }
                    }
                    MESSAGE_OPEN_BEADFIXER => {
                        b.call("imodPlugOpenByName", &["Bead Fixer".into()]);
                        b.call("clientMessage.findClosestBeadfixSection", &[]);
                    }
                    MESSAGE_ONE_ZAP_OPEN => {
                        b.call("inputRaiseWindows", &[]);
                        b.call("imod_zap_open_if_none", &[]);
                    }
                    MESSAGE_RUBBERBAND => {
                        b.call("zapReportRubberband", &[]);
                    }
                    MESSAGE_SLICER_ANGLES => {
                        b.call("slicerReportAngles", &[]);
                    }
                    MESSAGE_OBJ_PROPERTIES
                    | MESSAGE_NEWOBJ_PROPERTIES
                    | MESSAGE_OBJ_PROPS_2
                    | MESSAGE_NEWOBJ_PROPS_2
                    | MESSAGE_OBJ_PROPS_3
                    | MESSAGE_NEWOBJ_PROPS_3 => {
                        let n = REQUIRED_ARGS[action as usize];
                        b.call(
                            "clientMessage.objectProperties",
                            &self.message_strings[arg + 1..=arg + n],
                        );
                        arg += n;
                    }
                    MESSAGE_GHOST_MODE => {
                        b.call(
                            "clientMessage.ghostMode",
                            &self.message_strings[arg + 1..=arg + 3],
                        );
                        arg += 3;
                    }
                    MESSAGE_ZAP_HQ_MODE => {
                        b.call(
                            "clientMessage.zapHighQuality",
                            &self.message_strings[arg + 1..=arg + 1],
                        );
                        arg += 1;
                    }
                    MESSAGE_OPEN_DIALOGS => {
                        b.call(
                            "imodvOpenSelectedWindows",
                            &self.message_strings[arg + 1..=arg + 1],
                        );
                        b.call(
                            "InfoWindow.openSelectedWindows",
                            &self.message_strings[arg + 1..=arg + 1],
                        );
                        arg += 1;
                    }
                    MESSAGE_MODEL_CHANGED => {
                        b.call("imod_model_changed.print", &[]);
                    }
                    MESSAGE_PLUGIN_EXECUTE => {
                        arg += 1;
                        if b.imod_plug_message(&self.message_strings, &mut arg) != 0 {
                            succeeded = 0;
                            arg = num_args;
                        }
                    }
                    MESSAGE_EDGE_FOR_MIDAS => {
                        b.call("inputFindEdgeForMidas", &[]);
                    }
                    MESSAGE_MIDAS_SET_EDGE => {
                        b.call(
                            "clientMessage.midasSetEdge",
                            &self.message_strings[arg + 1..=arg + 4],
                        );
                        arg += 4;
                    }
                    MESSAGE_MULTIZ_PANELS => {
                        b.call(
                            "clientMessage.multiZPanels",
                            &self.message_strings[arg + 1..=arg + 4],
                        );
                        arg += 4;
                    }
                    _ => {
                        b.stderr(&format!(
                            "imodExecuteMessage: action {action} not recognized\n"
                        ));
                        succeeded = 0;
                        arg = num_args;
                    }
                }
            } else {
                match action {
                    MESSAGE_QUIT => arg = num_args,
                    MESSAGE_RAISE_WINDOWS => {
                        b.call("imodvInputRaise", &[]);
                    }
                    MESSAGE_OPEN_DIALOGS => {
                        arg += 1;
                        b.call("imodvOpenSelectedWindows", &self.message_strings[arg..=arg]);
                    }
                    _ => {
                        b.stderr(&format!(
                            "imodExecuteMessage: action {action} not recognized by 3dmodv\n"
                        ));
                        succeeded = 0;
                        arg = num_args;
                    }
                }
            }
            arg += 1;
        }
        if succeeded >= 0 && self.m_deferred_handling <= 0 {
            self.send_response(succeeded, b);
        }
        if self.m_deferred_handling > 0 {
            self.m_deferred_handling = 0;
        }
        self.message_action == MESSAGE_QUIT
    }

    /// `ImodClipboard::sendResponse`.
    pub fn send_response(&mut self, mut succeeded: i32, b: &mut dyn ClientMessageBoundary) {
        if succeeded < 0 {
            succeeded = self.last_response;
        }
        self.last_response = succeeded;
        let response = if succeeded != 0 { "OK" } else { "ERROR" };
        if self.m_use_stdin {
            b.stderr(&format!("{response}\n"));
        } else {
            let window_id = self.our_window_id(b);
            b.set_clipboard_text(format!("{window_id} {response}"));
        }
    }

    /// `ImodClipboard::ourWindowID`.
    pub fn our_window_id(&self, b: &mut dyn ClientMessageBoundary) -> u32 {
        let s = b.state();
        if s.imodv_closed || !s.imodv_standalone {
            s.info_window_id
        } else {
            s.imodv_window_id
        }
    }
}

/// `readLine`, factored out in the C file for the Unix and Windows reader.
pub fn read_line(line: &str) -> String {
    let mut text = line.chars().take(MAX_LINE - 1).collect::<String>();
    while text.ends_with(['\n', '\r']) {
        text.pop();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Boundary {
        clipboard: String,
        stderr: Vec<String>,
        calls: Vec<(String, Vec<String>)>,
        scheduled: Vec<i32>,
        stdin: Vec<Option<String>>,
        state: ClientMessageViewState,
    }
    impl ClientMessageBoundary for Boundary {
        fn state(&self) -> ClientMessageViewState {
            self.state
        }
        fn clipboard_text(&mut self) -> String {
            self.clipboard.clone()
        }
        fn set_clipboard_text(&mut self, text: String) {
            self.clipboard = text;
        }
        fn stderr(&mut self, text: &str) {
            self.stderr.push(text.into());
        }
        fn call(&mut self, callee: &'static str, args: &[String]) -> i32 {
            self.calls.push((callee.into(), args.into()));
            0
        }
        fn schedule_timeout(&mut self, ms: i32) {
            self.scheduled.push(ms);
        }
        fn stdin_line(&mut self) -> Option<String> {
            self.stdin.remove(0)
        }
    }
    fn normal() -> Boundary {
        Boundary {
            state: ClientMessageViewState {
                info_window_present: true,
                imodv_closed: true,
                imodv_standalone: false,
                info_window_id: 42,
                imodv_window_id: 8,
            },
            ..Default::default()
        }
    }

    #[test]
    fn clipboard_protocol_deduplicates_stamp_and_preserves_escaped_space() {
        let mut b = normal();
        b.clipboard = "42 7 1 /tmp/a\\\\ b.mod".into();
        let mut c = ImodClipboard::new(false, false, &mut b);
        assert!(c.handle_message(&mut b));
        assert_eq!(c.message_strings, ["1", "/tmp/a b.mod"]);
        c.execute_message(&mut b);
        assert_eq!(
            b.calls[1],
            ("openModel".into(), vec!["/tmp/a b.mod".into()])
        );
        b.clipboard = "42 7 5".into();
        assert!(!c.handle_message(&mut b));
        assert_eq!(b.clipboard, "42 OK");
    }

    #[test]
    fn initial_load_defers_actions_then_processes_them() {
        let mut b = normal();
        let mut c = ImodClipboard::new(true, true, &mut b);
        b.stdin.push(Some("5\n".into()));
        c.stdin_timeout(&mut b);
        c.clip_timeout(&mut b);
        assert!(b.calls.is_empty());
        c.done_with_load(&mut b);
        assert_eq!(b.calls[0].0, "inputRaiseWindows");
    }

    #[test]
    fn source_constructor_facade_reads_the_owned_clipboard_boundary() {
        let mut b = normal();
        b.clipboard = "saved clipboard".into();
        let clipboard = imod_clipboard(false, false, &mut b);
        assert_eq!(clipboard.m_saved_clipboard, "saved clipboard");
    }

    #[test]
    fn standalone_accepts_only_source_message_subset() {
        let mut b = normal();
        b.state.imodv_closed = false;
        b.state.imodv_standalone = true;
        let mut c = ImodClipboard::new(false, false, &mut b);
        c.message_strings = vec![MESSAGE_RAISE_WINDOWS.to_string()];
        c.execute_message(&mut b);
        assert_eq!(b.calls[0].0, "imodvInputRaise");
        assert_eq!(b.clipboard, "8 OK");
    }

    #[test]
    fn read_line_strips_both_native_endings() {
        assert_eq!(read_line("four\r\n"), "four");
    }
}
