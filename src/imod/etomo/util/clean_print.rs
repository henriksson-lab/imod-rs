//! `IMOD/Etomo/src/etomo/util/CleanPrint.java`.
//!
//! Selective printing to a log file.  Can prevent duplicate messages.  Messages can be
//! blocked when not needed.
//!
//! The `stream` field is a `java.io.PrintStream`; only `System.err` (the default) and
//! `System.out` are reachable from the callers in this crate, so the field is carried as
//! the `Stream` enum below rather than as an open stream object.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::sync::{LazyLock, Mutex};

use crate::imod::etomo::etomo_director;
use crate::imod::etomo::util::utilities;

/// The `java.io.PrintStream` a `CleanPrint` writes to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stream {
    /// `System.err`, the field's default.
    Err,
    /// `System.out`.
    Out,
}

/// Java `TEST_FLAG`.
pub const TEST_FLAG: i32 = 1 << 0; // 1
/// Java `DEBUG_FLAG`.
pub const DEBUG_FLAG: i32 = 1 << 1; // 10

/// Java `staticAllowedLabels`.  Only messages with a label in `staticAllowedLabels` will
/// be printed.  When `onlyWhenAllowed` is set, this is always true.  When
/// `onlyWhenAllowed` is off this is only true when `staticAllowedLabels` is not empty or
/// null.
static STATIC_ALLOWED_LABELS: Mutex<Option<BTreeSet<String>>> = Mutex::new(None);
/// Java `staticBlocked`.  Prevent instances that have `blockable` set from printing.
static STATIC_BLOCKED: Mutex<bool> = Mutex::new(true);

/// Java `CleanPrint`.
pub struct CleanPrint {
    /// Java field `onlyWhenAllowed`.  Label must be in `staticAllowedLabels`, or
    /// `messageLabel` must be in `allowedMessageLabels`, for the message to be printed.
    only_when_allowed: bool,
    /// Java field `blockable`.  Gives printing control to `staticBlocked`.
    blockable: bool,
    /// Java field `ignoreRepeatMessages`.  Prevents printing a message that is the same
    /// as the previous message.
    ignore_repeat_messages: bool,
    /// Java field `allowedMessageLabels`.
    allowed_message_labels: Option<BTreeSet<String>>,
    /// Java field `emphasize`.  Add ">>>" before each message.
    emphasize: bool,
    /// Java field `printTimestamp`.  Print a timestamp before each message.
    pub print_timestamp: bool,
    /// Java field `label`.  Optional label printed at the beginning of each message.
    pub label: Option<String>,
    /// Java field `stream`.  Where to send the message.
    stream: Stream,
    /// Java field `prevMessage`.  Used with `ignoreRepeatMessages`.
    prev_message: Mutex<Option<String>>,
}

/// Java `CLEAN_PRINT`: `public static final CleanPrint CLEAN_PRINT = getInstance();`.
pub static CLEAN_PRINT: LazyLock<CleanPrint> = LazyLock::new(CleanPrint::get_instance);

impl CleanPrint {
    /// Java `CleanPrint(Boolean, Boolean, Boolean, String[], Boolean, Boolean, String,
    /// PrintStream)`.  Use null to set any final member variable to its default.
    pub fn new(
        only_when_allowed: Option<bool>,
        blockable: Option<bool>,
        ignore_repeat_messages: Option<bool>,
        allowed_message_labels: Option<&[String]>,
        emphasize: Option<bool>,
        print_timestamp: Option<bool>,
        label: Option<&str>,
        stream: Option<Stream>,
    ) -> CleanPrint {
        CleanPrint {
            // OnlyWhenAllowed default:
            only_when_allowed: only_when_allowed.unwrap_or(false),
            // Blockable default:
            blockable: blockable.unwrap_or(false),
            // IgnoreRepeatMessages default:
            ignore_repeat_messages: ignore_repeat_messages.unwrap_or(false),
            // AllowedMessageLabels default:
            allowed_message_labels: match allowed_message_labels {
                None => None,
                Some(allowed_message_labels) if allowed_message_labels.is_empty() => None,
                Some(allowed_message_labels) => {
                    Some(allowed_message_labels.iter().cloned().collect())
                }
            },
            // Emphasize default:
            emphasize: emphasize.unwrap_or(false),
            // PrintTimestamp default:
            print_timestamp: print_timestamp.unwrap_or(false),
            // Label default is null.
            label: label.map(|label| label.to_string()),
            // Stream default:
            stream: stream.unwrap_or(Stream::Err),
            prev_message: Mutex::new(None),
        }
    }

    /// Java `getInstance()`.
    pub fn get_instance() -> CleanPrint {
        CleanPrint::new(None, None, None, None, None, None, None, None)
    }

    /// Java `getInstance(String[], String)`.
    pub fn get_instance_with_message_labels(
        allowed_message_labels: Option<&[String]>,
        label: Option<&str>,
    ) -> CleanPrint {
        CleanPrint::new(
            None,
            None,
            None,
            allowed_message_labels,
            None,
            None,
            label,
            None,
        )
    }

    /// Java `getInstance(String)`.
    pub fn get_instance_with_label(label: Option<&str>) -> CleanPrint {
        CleanPrint::new(None, None, None, None, None, None, label, None)
    }

    /// Java `getInstance(boolean, String)`.
    pub fn get_instance_blockable(blockable: bool, label: Option<&str>) -> CleanPrint {
        CleanPrint::new(None, Some(blockable), None, None, None, None, label, None)
    }

    /// Java `getInstance(Boolean, String)`.
    pub fn get_instance_only_when_allowed(
        only_when_allowed: Option<bool>,
        label: Option<&str>,
    ) -> CleanPrint {
        CleanPrint::new(only_when_allowed, None, None, None, None, None, label, None)
    }

    /// Java `block`.
    pub fn block(block: bool) {
        *STATIC_BLOCKED.lock().unwrap() = block;
    }

    /// Java `addAllowedLabel`.
    pub fn add_allowed_label(allowed_label: Option<&str>) {
        if allowed_label.is_none() || utilities::is_empty(allowed_label) {
            return;
        }
        let allowed_label = allowed_label.unwrap();
        let mut static_allowed_labels = STATIC_ALLOWED_LABELS.lock().unwrap();
        if static_allowed_labels.is_none() {
            *static_allowed_labels = Some(BTreeSet::new());
        }
        let set = static_allowed_labels.as_mut().unwrap();
        if !set.contains(allowed_label) {
            set.insert(allowed_label.to_string());
        }
    }

    /// Java `print(String, String, boolean)`.
    pub fn print_message_label(
        &self,
        message_label: Option<&str>,
        message: Option<&str>,
        dump_stack: bool,
    ) {
        self.print_flags(None, message_label, message, dump_stack);
    }

    /// Java `print(Integer, String, String, boolean)`.  Print a message if allowed.
    ///
    /// * `message_label` - optional label that follows the label member variable.
    /// * `message` - optional main part of the message.
    /// * `dump_stack` - optional stack dump after the message.
    pub fn print_flags(
        &self,
        flags: Option<i32>,
        message_label: Option<&str>,
        message: Option<&str>,
        dump_stack: bool,
    ) {
        if let Some(flags) = flags {
            if (flags & TEST_FLAG) != 0 && !etomo_director::ARGUMENTS.lock().unwrap().is_test() {
                return;
            }
            if (flags & DEBUG_FLAG) != 0 && !etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                return;
            }
        }
        if !self.print_timestamp
            && self.label.is_none()
            && message_label.is_none()
            && message.is_none()
            && !dump_stack
        {
            return;
        }

        let static_allowed_labels = STATIC_ALLOWED_LABELS.lock().unwrap();
        let allowed_labels_set = match static_allowed_labels.as_ref() {
            None => false,
            Some(static_allowed_labels) => !static_allowed_labels.is_empty(),
        };
        let allowed_message_label_set = match self.allowed_message_labels.as_ref() {
            None => false,
            Some(allowed_message_labels) => !allowed_message_labels.is_empty(),
        };

        if self.only_when_allowed && !allowed_labels_set && !allowed_message_label_set {
            return;
        }
        if allowed_labels_set
            && match self.label.as_deref() {
                None => true,
                Some(label) => !static_allowed_labels.as_ref().unwrap().contains(label),
            }
        {
            return;
        }
        drop(static_allowed_labels);
        if self.blockable && *STATIC_BLOCKED.lock().unwrap() {
            return;
        }
        {
            let prev_message = self.prev_message.lock().unwrap();
            if self.ignore_repeat_messages
                && prev_message.is_some()
                && prev_message.as_deref() == message
            {
                return;
            }
        }
        if allowed_message_label_set
            && match message_label {
                None => true,
                Some(message_label) => !self
                    .allowed_message_labels
                    .as_ref()
                    .unwrap()
                    .contains(message_label),
            }
        {
            return;
        }

        let mut builder = String::new();

        if self.print_timestamp {
            if self.emphasize {
                builder.push_str(">>>");
            }
            builder.push('[');
            builder.push_str(&utilities::java_lang_system_current_time_millis().to_string());
            builder.push_str("]\n");
        }

        if let Some(label) = self.label.as_deref() {
            if self.emphasize {
                builder.push_str(">>>");
            }
            builder.push_str(label);
            builder.push(':');
            if message_label.is_none() && message.is_none() {
                builder.push('\n');
            }
        }
        if let Some(message_label) = message_label {
            if self.emphasize && self.label.is_none() {
                builder.push_str(">>>");
            }
            builder.push_str(message_label);
            builder.push(':');
            if message.is_none() {
                builder.push('\n');
            }
        }
        if let Some(message) = message {
            if self.emphasize && self.label.is_none() && message_label.is_none() {
                builder.push_str(">>>");
            }
            builder.push_str(message);
            builder.push('\n');
        }

        match self.stream {
            Stream::Err => eprintln!("{}", builder),
            Stream::Out => println!("{}", builder),
        }

        if let Some(message) = message {
            *self.prev_message.lock().unwrap() = Some(message.to_string());
        }

        if dump_stack {
            // Deviation: `Thread.dumpStack()` prints the calling Java thread's frames.
            // There are no Java frames in this process; see
            // `crate::imod::etomo::util::stack_trace`.
            match self.stream {
                Stream::Err => eprintln!(),
                Stream::Out => println!(),
            }
        }
    }

    /// Java `printLabels`.
    pub fn print_labels(&self, message_label: Option<&str>) {
        self.print_flags(None, message_label, None, false);
    }

    /// Java `print(String)`.
    pub fn print(&self, message: Option<&str>) {
        self.print_flags(None, None, message, false);
    }

    /// Java `print(Integer, String)`.
    pub fn print_with_flags(&self, flags: Option<i32>, message: Option<&str>) {
        self.print_flags(flags, None, message, false);
    }

    /// Java `printAndBlock(String, boolean)`.
    pub fn print_and_block(&self, message_label: Option<&str>, and_block: bool) {
        self.print_flags(None, message_label, None, false);
        CleanPrint::block(and_block);
    }

    /// Java `printAndBlock(String, String, boolean)`.
    pub fn print_and_block_message(
        &self,
        message_label: Option<&str>,
        message: Option<&str>,
        and_block: bool,
    ) {
        self.print_flags(None, message_label, message, false);
        CleanPrint::block(and_block);
    }

    /// Java `print(String, String)`.
    pub fn print_labelled(&self, message_label: Option<&str>, message: Option<&str>) {
        self.print_flags(None, message_label, message, false);
    }

    /// Java `print(String, boolean)`.
    pub fn print_dump_stack(&self, message: Option<&str>, dump_stack: bool) {
        self.print_flags(None, None, message, dump_stack);
    }
}
