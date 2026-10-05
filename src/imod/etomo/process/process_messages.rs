//! `IMOD/Etomo/src/etomo/process/ProcessMessages.java`.
//!
//! Parses the output of a process (or a log file) for messages - errors, warnings,
//! information, log lines, log files and success - and keeps them in rated lists.  LOG
//! messages and LOGFILE files go to the manager's project log (and to the secondary log
//! when one is set).
//!
//! **Parser.**  `EtomoDirector.NEW_MESSAGE_PARSER` is `true`, so every
//! `if (!EtomoDirector.NEW_MESSAGE_PARSER)` arm is dead, and so are the methods only it
//! calls: `parse()`, `parseMessagePrepend`, `parseSuccessLine`, `parsePipWarning`,
//! `parseSingleLineMessage`, `parseMultiLineMessage`, the three `storeMessageOLD`
//! overloads, the `messagePrepend` field and `MAX_MESSAGE_SIZE` (DEAD_CODE.md).  The
//! `MessageParser` is a field here; it is taken out for the length of each parse and
//! given this object, as the Java parser holds it.
//!
//! **Threads.**  Java's `synchronized` methods are the `Mutex` every holder keeps the
//! object in.  The string feed is a channel read by a parse thread that holds a
//! reference to that `Mutex` ([`MessagesRef`]); the thread takes each string without
//! the lock (Java's blocking `ArrayBlockingQueue.take`) and parses it with the lock
//! held.  The Java queue's capacity of 1000 is not reproduced: a feeder holds the lock
//! while it puts, so a full queue would block the parse thread forever.
//!
//! **Project log.**  `BaseManager.logSimpleMessage` is called from whichever thread
//! parses; the manager posts the call to the event dispatch thread, as Java's
//! `EtomoLogger` posts each append.
#![allow(dead_code)]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

use super::message::Message;
use super::message_parser::MessageParser;
use super::output_buffer_manager::OutputBufferManager;
use super::process_output_strings;
use super::tag_interface::TagList;
use crate::imod::etomo::arguments::GrabItParameter;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::file_writer::{FileWriter, FileWriterRef};
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, ReaderId};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::queue::Queue;
use crate::imod::etomo::util::utilities;

/// Java private static final `ALT_ERROR_TAG1`.
const ALT_ERROR_TAG1: &str = "Errno";
/// Java private static final `ALT_ERROR_TAG2`.
const ALT_ERROR_TAG2: &str = "Traceback";
/// Java private static final `ERROR_TAGS = { MessageType.ERROR.tag, ALT_ERROR_TAG1,
/// ALT_ERROR_TAG2 }`.
const ERROR_TAGS: [&str; 3] = ["ERROR:", ALT_ERROR_TAG1, ALT_ERROR_TAG2];
/// Java private static final `IGNORE_TAG`.
const IGNORE_TAG: [&str; 2] = ["prnstr('ERROR:", "log.write('ERROR:"];
/// Java private static final `STRING_FEED_QUEUE_CAPACITY` (see the module comment).
const STRING_FEED_QUEUE_CAPACITY: i32 = 1000;
/// Java private static final `DEBUG = EtomoDirector.INSTANCE.getArguments().isDebug()`.
static DEBUG: LazyLock<bool> =
    LazyLock::new(|| etomo_director::ARGUMENTS.lock().unwrap().is_debug());

/// Java `Line.END_FEED_TOKEN`.
pub const END_FEED_TOKEN: &str =
    "This the END of the String Feed!!!  239asdlkjgsafT$LSFJsGHW($(gjhaehgpasjdhf0w235";

/// A Java reference to a `ProcessMessages`, shared between threads.  Its `Mutex` is
/// Java's `synchronized`.
pub type MessagesRef = Arc<Mutex<ProcessMessages>>;

/// A manager's `List<ProcessMessages> messagesArray`: one list shared with the process
/// layer, whose elements are object references (shared with the monitors that feed
/// them) and may be null (`ProcesschunksBatchRunTomoMonitor` fills gaps with nulls).
pub type MessagesArray = Arc<Mutex<Vec<Option<MessagesRef>>>>;

/// Java `String.matches("\\s*")`.
fn java_lang_string_matches_whitespace(string: &str) -> bool {
    string
        .chars()
        .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
}

/// Java `String.trim()`.
fn java_lang_string_trim(string: &str) -> String {
    string.trim_matches(|c: char| c <= ' ').to_owned()
}

/// Java package-private static final nested `ListType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ListType {
    /// Java `CHUNK_ERROR = new ListType("CHUNK_ERROR", true)`.
    ChunkError,
    /// Java `ERROR`.
    Error,
    /// Java `INFO`.
    Info,
    /// Java `WARNING`.
    Warning,
    /// Java `LOGGED`.
    Logged,
    /// Java `FLAG`.
    Flag,
    /// Java `CHUNK_WARNING = new ListType("CHUNK_WARNING", true)`.
    ChunkWarning,
}

impl ListType {
    /// Java private final `name`.
    fn name(self) -> &'static str {
        match self {
            ListType::ChunkError => "CHUNK_ERROR",
            ListType::Error => "ERROR",
            ListType::Info => "INFO",
            ListType::Warning => "WARNING",
            ListType::Logged => "LOGGED",
            ListType::Flag => "FLAG",
            ListType::ChunkWarning => "CHUNK_WARNING",
        }
    }

    /// Java package-private `isChunk()`.
    pub fn is_chunk(self) -> bool {
        matches!(self, ListType::ChunkError | ListType::ChunkWarning)
    }
}

/// Java `toString()`: the name.
impl std::fmt::Display for ListType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Java public static final nested `MessageType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MessageType {
    /// Java `CHUNK_ERROR = new MessageType("CHUNK ERROR:", ListType.CHUNK_ERROR, false)`.
    ChunkError,
    /// Java `ERROR = new MessageType("ERROR:", ListType.ERROR, false)`.
    Error,
    /// Java `INFO = new MessageType("INFO:", ListType.INFO, false)`.
    Info,
    /// Java `WARNING = new MessageType("WARNING:", ListType.WARNING, false)`.
    Warning,
    /// Java `LOG = new MessageType(UIHarness.LOG_TAG + ":", "[:" + UIHarness.LOG_TAG +
    /// "]", null, false)`.  Always goes to the project log.
    Log,
    /// Java `LOG_FILE = new MessageType("LOGFILE:", null, false)`.  Followed by the
    /// name of a file.
    LogFile,
    /// Java `PIP_WARNING_START = new MessageType("PIP WARNING:", ListType.INFO, true)`.
    PipWarningStart,
    /// Java `PIP_WARNING_END = new MessageType("Using fallback options in main
    /// program", ListType.INFO, true)`.
    PipWarningEnd,
    /// Java `SUCCESS = new MessageType(null, null, true)`.
    Success,
    /// Java `PREPEND = new MessageType(null, null, true)`.
    Prepend,
    /// Java `CHUNK_WARNING = new MessageType(null, ListType.CHUNK_WARNING, false)`.
    ChunkWarning,
}

impl MessageType {
    /// Java private final field `tag`.
    pub fn tag(self) -> Option<&'static str> {
        match self {
            MessageType::ChunkError => Some("CHUNK ERROR:"),
            MessageType::Error => Some("ERROR:"),
            MessageType::Info => Some("INFO:"),
            MessageType::Warning => Some("WARNING:"),
            MessageType::Log => Some("LOG:"),
            MessageType::LogFile => Some("LOGFILE:"),
            MessageType::PipWarningStart => Some("PIP WARNING:"),
            MessageType::PipWarningEnd => Some("Using fallback options in main program"),
            MessageType::Success | MessageType::Prepend | MessageType::ChunkWarning => None,
        }
    }

    /// Java private final field `secondaryTag`.
    pub fn secondary_tag(self) -> Option<&'static str> {
        match self {
            MessageType::Log => Some("[:LOG]"),
            _ => None,
        }
    }

    /// Java private final field `listType`.
    pub fn list_type(self) -> Option<ListType> {
        match self {
            MessageType::ChunkError => Some(ListType::ChunkError),
            MessageType::Error => Some(ListType::Error),
            MessageType::Info => Some(ListType::Info),
            MessageType::Warning => Some(ListType::Warning),
            MessageType::PipWarningStart | MessageType::PipWarningEnd => Some(ListType::Info),
            MessageType::ChunkWarning => Some(ListType::ChunkWarning),
            MessageType::Log
            | MessageType::LogFile
            | MessageType::Success
            | MessageType::Prepend => None,
        }
    }

    /// Java private final field `exclusive` (parsed separately).
    pub fn exclusive(self) -> bool {
        matches!(
            self,
            MessageType::PipWarningStart
                | MessageType::PipWarningEnd
                | MessageType::Success
                | MessageType::Prepend
        )
    }

    /// Java `getTag()`.
    pub fn get_tag(self) -> Option<&'static str> {
        self.tag()
    }

    /// Java package-private `getSecondaryTag()`.
    pub fn get_secondary_tag(self) -> Option<&'static str> {
        self.secondary_tag()
    }

    /// Java package-private `getListType()`.
    pub fn get_list_type(self) -> Option<ListType> {
        self.list_type()
    }

    /// Java `isType(String)`.  Returns true if tag or secondary tag is in string.
    pub fn is_type(self, string: Option<&str>) -> bool {
        let Some(string) = string else {
            return false;
        };
        self.secondary_tag()
            .is_some_and(|secondary_tag| string.contains(secondary_tag))
            || self.tag().is_some_and(|tag| string.contains(tag))
    }
}

/// Java `toString()`: the tag (`"null"` for the untagged types).
impl std::fmt::Display for MessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.tag().unwrap_or("null"))
    }
}

/// Java `public final class ProcessMessages`.
pub struct ProcessMessages {
    /// Java private final `chunks`.
    chunks: bool,
    /// Java private final `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `logAllMessages`.  Send messages to project log instead of
    /// storing them.
    log_all_messages: bool,
    /// Java private final `errorOverrideLogTag`.
    error_override_log_tag: Option<String>,
    /// Java private final `errorTags`.
    error_tags: Vec<String>,
    /// Java private final `parser` (taken out while it parses).
    parser: Option<MessageParser>,
    /// Java private final `errorTagsAlwaysMultiline`.
    error_tags_always_multiline: Vec<bool>,
    /// Java private final `currentLine`.
    current_line: Line,
    /// Java private final `logInfoMessages`.
    log_info_messages: bool,
    /// Java private final `messageBuilder`.
    message_builder: MessageBuilder,

    /// Java private `outputBufferManager`: the lines of the `OutputBufferManager`
    /// being parsed (it is complete when it is handed over).
    output_buffer_manager: Option<Vec<String>>,
    /// Java private `bufferedReader`.
    buffered_reader: Option<std::io::Lines<BufReader<File>>>,
    /// Java private `processOutputString`.
    process_output_string: Option<String>,
    /// Java private `processOutputStringArray`.
    process_output_string_array: Option<Vec<String>>,
    /// Java private `logFile`.
    log_file: Option<Arc<Handle>>,
    /// Java private `logFileReaderId`.
    log_file_reader_id: Option<ReaderId>,
    /// Java private `index`.
    index: i32,
    /// Java private `infoList`.
    info_list: Option<RatedList>,
    /// Java private `warningList`.
    warning_list: Option<RatedList>,
    /// Java private `errorList`.
    error_list: Option<RatedList>,
    /// Java private `chunkErrorList`.
    chunk_error_list: Option<RatedList>,
    /// Java private `chunkWarningList`.
    chunk_warning_list: Option<RatedList>,
    /// Java private `successTag1`.
    success_tag1: Option<String>,
    /// Java private `successTag2`.
    success_tag2: Option<String>,
    /// Java private `success`.
    success: bool,
    /// Java private `multiLineAllMessages`.  Multi line error, warning, and info
    /// strings; terminated by an empty line.  Does not affect LOG strings.
    multi_line_all_messages: bool,
    /// Java private `multiLineWarning` (overridden by multiLineAllMessages).
    multi_line_warning: bool,
    /// Java private `multiLineInfo` (overridden by multiLineAllMessages).
    multi_line_info: bool,
    /// Java private `messagePrependTag`.
    message_prepend_tag: Option<String>,
    /// Java private `stringFeed`: the sending end of the string feed.
    string_feed: Option<Sender<String>>,
    /// Java private `stringFeedThread`.
    string_feed_thread: Option<ThreadId>,
    /// Signalled when the string feed thread ends (Java `Thread.join`).
    string_feed_done: Option<Arc<(Mutex<bool>, Condvar)>>,
    /// The string the parse thread took from the feed and has not loaded yet.
    string_feed_pending: Option<String>,
    /// Java private `hibernate`.
    hibernate: bool,
    /// Java private `allowMultiLineLog` (not overridden by multiLineAllMessages).
    allow_multi_line_log: bool,
    /// Java private `debug`.
    debug: bool,
    /// Java private `secondaryLog`.
    secondary_log: Option<FileWriterRef>,
}

impl ProcessMessages {
    /// Java static `getInstance(BaseManager, AxisID)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
    ) -> ProcessMessages {
        ProcessMessages::new(
            manager, axis_id, false, false, None, None, false, false, false, None, None, false,
            false, true, false,
        )
    }

    /// Java static `getInstance(BaseManager, AxisID, boolean allowMultiLineLog)`.
    pub fn get_instance_allow_multi_line_log(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        allow_multi_line_log: bool,
    ) -> ProcessMessages {
        ProcessMessages::new(
            manager,
            axis_id,
            false,
            false,
            None,
            None,
            false,
            false,
            false,
            None,
            None,
            false,
            allow_multi_line_log,
            true,
            false,
        )
    }

    /// Java static `getInstance(BaseManager, AxisID, String successTag)`.
    pub fn get_instance_success_tag(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        success_tag: Option<&str>,
    ) -> ProcessMessages {
        ProcessMessages::new(
            manager,
            axis_id,
            false,
            true,
            success_tag,
            None,
            false,
            false,
            false,
            None,
            None,
            false,
            false,
            true,
            true,
        )
    }

    /// Java static `getInstance(BaseManager, AxisID, String successTag1, String
    /// successTag2)`.
    pub fn get_instance_success_tags(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        success_tag1: Option<&str>,
        success_tag2: Option<&str>,
    ) -> ProcessMessages {
        ProcessMessages::new(
            manager,
            axis_id,
            false,
            true,
            success_tag1,
            success_tag2,
            false,
            false,
            false,
            None,
            None,
            false,
            false,
            true,
            true,
        )
    }

    /// Java static `getMultiLineInstance(BaseManager, AxisID)`.
    pub fn get_multi_line_instance(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
    ) -> ProcessMessages {
        ProcessMessages::new(
            manager, axis_id, true, false, None, None, false, false, false, None, None, false,
            false, true, false,
        )
    }

    /// Java static `getMultiLineInstance(BaseManager, AxisID, boolean
    /// allowMultiLineLog)`.
    pub fn get_multi_line_instance_allow_multi_line_log(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        allow_multi_line_log: bool,
    ) -> ProcessMessages {
        ProcessMessages::new(
            manager,
            axis_id,
            true,
            false,
            None,
            None,
            false,
            false,
            false,
            None,
            None,
            false,
            allow_multi_line_log,
            true,
            false,
        )
    }

    /// Java static `getInstanceForParallelProcessing(BaseManager, AxisID, boolean)`.
    pub fn get_instance_for_parallel_processing(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        multi_line_messages: bool,
    ) -> ProcessMessages {
        ProcessMessages::new(
            manager,
            axis_id,
            multi_line_messages,
            true,
            None,
            None,
            false,
            false,
            false,
            None,
            None,
            false,
            false,
            true,
            true,
        )
    }

    /// Java static `getLoggedInstance(BaseManager, AxisID, boolean multiLineMessages,
    /// boolean logAllMessages, String errorOverrideLogTag, String errorTag, boolean
    /// alwaysMultiline, boolean allowMultiLineLog)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_logged_instance(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        multi_line_messages: bool,
        log_all_messages: bool,
        error_override_log_tag: Option<&str>,
        error_tag: Option<&str>,
        always_multiline: bool,
        allow_multi_line_log: bool,
    ) -> ProcessMessages {
        ProcessMessages::new(
            manager,
            axis_id,
            multi_line_messages,
            false,
            None,
            None,
            false,
            false,
            log_all_messages,
            error_override_log_tag,
            error_tag,
            always_multiline,
            allow_multi_line_log,
            true,
            false,
        )
    }

    /// Java static `getBatchruntomoTestInstance(...)`: as `getLoggedInstance`, with
    /// information messages not written to the project log.
    #[allow(clippy::too_many_arguments)]
    pub fn get_batchruntomo_test_instance(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        multi_line_messages: bool,
        log_all_messages: bool,
        error_override_log_tag: Option<&str>,
        error_tag: Option<&str>,
        always_multiline: bool,
        allow_multi_line_log: bool,
    ) -> ProcessMessages {
        ProcessMessages::new(
            manager,
            axis_id,
            multi_line_messages,
            false,
            None,
            None,
            false,
            false,
            log_all_messages,
            error_override_log_tag,
            error_tag,
            always_multiline,
            allow_multi_line_log,
            false,
            false,
        )
    }

    /// Java static `getMultiLineInstance(BaseManager, AxisID, boolean
    /// multiLineWarning, boolean multiLineInfo, boolean logInfoMessages)`.
    pub fn get_multi_line_instance_with_options(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        multi_line_warning: bool,
        multi_line_info: bool,
        log_info_messages: bool,
    ) -> ProcessMessages {
        ProcessMessages::new(
            manager,
            axis_id,
            false,
            false,
            None,
            None,
            multi_line_warning,
            multi_line_info,
            false,
            None,
            None,
            false,
            false,
            log_info_messages,
            false,
        )
    }

    /// Java private `ProcessMessages(BaseManager, AxisID, boolean multiLineMessages,
    /// boolean chunks, String successTag1, String successTag2, boolean
    /// multiLineWarning, boolean multiLineInfo, boolean logAllMessages, String
    /// errorOverrideLogTag, String errorTag, boolean errorTagAlwaysMultiline, boolean
    /// allowMultiLineLog, boolean logInfoMessages, boolean debug)`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        multi_line_messages: bool,
        chunks: bool,
        success_tag1: Option<&str>,
        success_tag2: Option<&str>,
        multi_line_warning: bool,
        multi_line_info: bool,
        log_all_messages: bool,
        error_override_log_tag: Option<&str>,
        error_tag: Option<&str>,
        error_tag_always_multiline: bool,
        allow_multi_line_log: bool,
        log_info_messages: bool,
        debug: bool,
    ) -> ProcessMessages {
        let (error_tags, error_tags_always_multiline) = match error_tag {
            None => (
                ERROR_TAGS.iter().map(|tag| (*tag).to_owned()).collect(),
                vec![false, false, true],
            ),
            Some(error_tag) => (
                vec![
                    MessageType::Error.tag().unwrap().to_owned(),
                    error_tag.to_owned(),
                    ALT_ERROR_TAG1.to_owned(),
                    ALT_ERROR_TAG2.to_owned(),
                ],
                vec![false, error_tag_always_multiline, false, true],
            ),
        };
        let mut instance = ProcessMessages {
            chunks,
            manager,
            axis_id,
            log_all_messages,
            error_override_log_tag: error_override_log_tag.map(str::to_owned),
            error_tags,
            parser: None,
            error_tags_always_multiline,
            current_line: Line::new(),
            log_info_messages,
            message_builder: MessageBuilder::new(),
            output_buffer_manager: None,
            buffered_reader: None,
            process_output_string: None,
            process_output_string_array: None,
            log_file: None,
            log_file_reader_id: None,
            index: -1,
            info_list: None,
            warning_list: None,
            error_list: None,
            chunk_error_list: None,
            chunk_warning_list: None,
            success_tag1: success_tag1.map(str::to_owned),
            success_tag2: success_tag2.map(str::to_owned),
            success: false,
            multi_line_all_messages: multi_line_messages,
            multi_line_warning,
            multi_line_info,
            message_prepend_tag: None,
            string_feed: None,
            string_feed_thread: None,
            string_feed_done: None,
            string_feed_pending: None,
            hibernate: false,
            allow_multi_line_log,
            debug,
            secondary_log: None,
        };
        // EtomoDirector.NEW_MESSAGE_PARSER is true.
        instance.parser = Some(MessageParser::get_instance(
            &instance,
            error_tag,
            error_tag_always_multiline,
            debug,
        ));
        instance
    }

    /// Java static `grabIt()`: parses each file named on the command line (called by
    /// `EtomoDirector.main`; no manager or main panel is available).
    pub fn grab_it() {
        let (file_name_list, grab_it_parameter) = {
            let arguments = etomo_director::ARGUMENTS.lock().unwrap();
            (
                arguments.get_param_file_name_list().to_vec(),
                arguments.get_grab_it_parameter(),
            )
        };
        // Construct a ProcessMessage instance.
        let mut process_messages = if grab_it_parameter == Some(GrabItParameter::CopyTomoComs) {
            // multi-line warning and info
            ProcessMessages::get_multi_line_instance_with_options(
                None,
                AxisID::Only,
                true,
                true,
                false,
            )
        } else if grab_it_parameter == Some(GrabItParameter::ParallelProcessing) {
            // chunks, multi-line messages off
            ProcessMessages::get_instance_for_parallel_processing(None, AxisID::Only, false)
        } else if grab_it_parameter == Some(GrabItParameter::Batchruntomo) {
            ProcessMessages::get_batchruntomo_test_instance(
                None,
                AxisID::Only,
                true,
                false,
                Some(process_output_strings::BRT_BATCH_RUN_TOMO_ERROR_TAG),
                Some(process_output_strings::BRT_ABORT_TAG),
                false,
                false,
            )
        } else {
            ProcessMessages::new(
                None,
                AxisID::Only,
                false,
                false,
                None,
                None,
                false,
                false,
                false,
                None,
                None,
                false,
                false,
                false,
                false,
            )
        };
        // Process each file.
        for file_name in &file_name_list {
            let result = (|| -> Result<(), LogFileError> {
                let log_file = LogFile::get_instance_file(Some(Path::new(file_name)), None)?;
                process_messages.clear();
                process_messages.add_process_output_log_file(log_file)?;
                let _list_type_iterator = process_messages.list_type_iterator();
                Ok(())
            })();
            match result {
                Ok(()) => {}
                // `catch (final LockException e) {}`
                Err(LogFileError::Lock(_)) => {}
                // `catch (final LogFileException | IOException e)`
                Err(e) => eprintln!("{e}"),
            }
        }
    }

    /// Java package-private `setDebug(boolean)`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `isLogAllMessages()`.
    pub fn is_log_all_messages(&self) -> bool {
        self.log_all_messages
    }

    /// Java `isLogInfoMessages()`.
    pub fn is_log_info_messages(&self) -> bool {
        self.log_info_messages
    }

    /// Java `isMultiLineAllMessages()`.
    pub fn is_multi_line_all_messages(&self) -> bool {
        self.multi_line_all_messages
    }

    /// Java `isMultiLineWarning()`.
    pub fn is_multi_line_warning(&self) -> bool {
        self.multi_line_warning
    }

    /// Java `isMultiLineInfo()`.
    pub fn is_multi_line_info(&self) -> bool {
        self.multi_line_info
    }

    /// Java `isAllowMultiLineLog()`.
    pub fn is_allow_multi_line_log(&self) -> bool {
        self.allow_multi_line_log
    }

    /// Java `getErrorOverrideLogTag()`.
    pub fn get_error_override_log_tag(&self) -> Option<&str> {
        self.error_override_log_tag.as_deref()
    }

    /// Java `getSuccessTag1()`.
    pub fn get_success_tag1(&self) -> Option<&str> {
        self.success_tag1.as_deref()
    }

    /// Java `getSuccessTag2()`.
    pub fn get_success_tag2(&self) -> Option<&str> {
        self.success_tag2.as_deref()
    }

    /// Java `isChunks()`.
    pub fn is_chunks(&self) -> bool {
        self.chunks
    }

    /// Java package-private `hibernate()`.  Instance should ignore all input.  Lines
    /// sent to instance will be lost.
    pub fn hibernate(&mut self) {
        self.hibernate = true;
    }

    /// Java package-private `wake()`.  Instance should stop hibernating and behave
    /// normally.
    pub fn wake(&mut self) {
        self.hibernate = false;
    }

    /// Java `listTypeIterator()`.
    pub fn list_type_iterator(&self) -> ListTypeIterator<'_> {
        ListTypeIterator {
            process_messages: self,
            cur_list_type: None,
        }
    }

    /// Java `startStringFeed()`.  Starts the thread that parses the strings sent to the
    /// string feed (Java `ParseThread`).  `this` is the reference the thread holds.
    pub fn start_string_feed(this: &MessagesRef) {
        let mut process_messages = this.lock().unwrap();
        // synchronized (stringFeedLock)
        if process_messages.string_feed.is_some() {
            return;
        }
        let (sender, receiver) = std::sync::mpsc::channel::<String>();
        process_messages.string_feed = Some(sender);
        // Start string feed thread
        // Run parse() on a separate thread.  NextLine will wait for stringFeed when no
        // other input is available.
        let done = Arc::new((Mutex::new(false), Condvar::new()));
        process_messages.string_feed_done = Some(done.clone());
        let shared = this.clone();
        let handle = std::thread::spawn(move || ParseThread::run(&shared, receiver, &done));
        process_messages.string_feed_thread = Some(handle.thread().id());
    }

    /// Java `stopStringFeed()`.  Sends a stop token to the string feed and then waits
    /// (up to a second) for the string feed thread to complete.  All previously added
    /// strings in the string feed will be processed before the string feed thread exits.
    pub fn stop_string_feed(this: &MessagesRef) {
        this.lock().unwrap().feed_string(END_FEED_TOKEN);
        let done = {
            let process_messages = this.lock().unwrap();
            // synchronized (stringFeedLock)
            if process_messages.string_feed_thread.is_none() {
                return;
            }
            process_messages.string_feed_done.clone()
        };
        // Calling thread waits until run thread stops
        if *DEBUG {
            eprintln!("Waiting for stringFeed");
        }
        if let Some(done) = done {
            let (lock, condvar) = &*done;
            let finished = lock.lock().unwrap();
            let _ = condvar
                .wait_timeout_while(finished, Duration::from_millis(1000), |finished| !*finished)
                .unwrap();
        }
        if *DEBUG {
            eprintln!("StringFeed stopped");
        }
    }

    /// Java package-private `feedEndMessage()`.  When multilinemessage is set, must
    /// send an extra message to get all the messages to be processed, because the
    /// parse may be waiting to see if there is more of the last message.
    pub fn feed_end_message(&mut self) {
        if self.hibernate {
            return;
        }
        self.feed_string("");
    }

    /// Java `clear()`.  Clear all lists.
    pub fn clear(&mut self) {
        if self.hibernate {
            return;
        }
        for list in [
            &mut self.info_list,
            &mut self.warning_list,
            &mut self.error_list,
            &mut self.chunk_error_list,
            &mut self.chunk_warning_list,
        ] {
            if let Some(list) = list {
                list.clear();
            }
        }
    }

    /// Java `dumpState()`.
    pub fn dump_state(&self) {
        if !*DEBUG {
            return;
        }
        eprint!(
            "[chunks:{},processOutputString:{},processOutputStringArray:",
            self.chunks,
            self.process_output_string.as_deref().unwrap_or("null")
        );
        if let Some(process_output_string_array) = &self.process_output_string_array {
            eprint!("{{");
            for (i, string) in process_output_string_array.iter().enumerate() {
                eprint!("{string}");
                if i < process_output_string_array.len() - 1 {
                    eprint!(",");
                }
            }
            eprint!("}}");
        }
        eprint!(
            ",index:{},line:{},infoList:",
            self.index,
            self.current_line.line.as_deref().unwrap_or("null")
        );
        if let Some(info_list) = &self.info_list {
            eprintln!("{info_list}");
        }
        eprint!(",warningList:");
        if let Some(warning_list) = &self.warning_list {
            eprintln!("{warning_list}");
        }
        eprint!(",errorList:");
        if let Some(error_list) = &self.error_list {
            eprintln!("{error_list}");
        }
        eprint!(",chunkErrorList:");
        if let Some(chunk_error_list) = &self.chunk_error_list {
            eprintln!("{chunk_error_list}");
        }
        eprint!(",chunkWarningList:");
        if let Some(chunk_warning_list) = &self.chunk_warning_list {
            eprintln!("{chunk_warning_list}");
        }
        eprint!(
            ",successTag1:{},successTag2:{},\nsuccess:{},multiLineMessages:{}]",
            self.success_tag1.as_deref().unwrap_or("null"),
            self.success_tag2.as_deref().unwrap_or("null"),
            self.success,
            self.multi_line_all_messages
        );
    }

    /// Java package-private `feedString(String)`.  Sends the string to the string
    /// feed, so it can be processed by the worker thread running `ParseThread.run`.
    pub fn feed_string(&mut self, string: &str) {
        if self.hibernate {
            return;
        }
        // synchronized (stringFeedLock)
        if let Some(string_feed) = &self.string_feed {
            // The receiver is gone only when the parse thread has ended, which also
            // clears `stringFeed`.
            let _ = string_feed.send(string.to_owned());
            return;
        }
        // If the string feed isn't operating, treat as a normal process output.
        self.add_process_output(Some(string));
    }

    /// Java `isStringFeed()`.
    pub fn is_string_feed(&self) -> bool {
        self.string_feed.is_some()
    }

    /// Java package-private `feedNewline(MessageType)`.  Sends an empty message of the
    /// type listed in parameter to the stringFeed.  For a LOG message, acts as a
    /// newline because the LOG tags is removed.
    pub fn feed_newline(&mut self, r#type: Option<MessageType>) {
        if self.hibernate {
            return;
        }
        match r#type {
            None => self.feed_string(""),
            Some(r#type) => self.feed_string(r#type.tag().unwrap_or("null")),
        }
    }

    /// Java package-private `feedMessage(MessageType, String)`.  Sends a single-line
    /// message of the type listed in parameter to the stringFeed, prepending the type
    /// tag when the message does not start with it, and a blank line after it so that
    /// it will be treated as single line message.
    pub fn feed_message(&mut self, r#type: Option<MessageType>, message: Option<&str>) {
        if self.hibernate {
            return;
        }
        match (r#type, message) {
            (None, message) => self.feed_string(message.unwrap_or("null")),
            (Some(r#type), None) => self.feed_string(r#type.tag().unwrap_or("null")),
            (Some(r#type), Some(message))
                if message.starts_with(r#type.tag().unwrap_or("null")) =>
            {
                self.feed_string(message)
            }
            (Some(r#type), Some(message)) => {
                self.feed_string(&format!("{} {}", r#type.tag().unwrap_or("null"), message))
            }
        }
        self.feed_string("");
    }

    /// Java package-private `setMessagePrependTag(String)`.  While the tag is set, the
    /// most recent line containing it is added to the beginning of the next
    /// error/warning message.
    ///
    /// Upstream bug fixed in translation (ProcessMessages.java:532): the new-parser arm
    /// passes the field `messagePrependTag` - which only the old parser's arm sets, so
    /// it is always null - to `parser.setPrepend`, so the prepend never works; the
    /// argument is passed here (BUGS.md).
    pub fn set_message_prepend_tag(&mut self, tag: Option<&str>) {
        if let Some(parser) = &mut self.parser {
            parser.set_prepend(tag);
        }
    }

    /// Runs the parser on this object (Java `parser.parse(header)`).
    fn parse_with_parser(&mut self, header: Option<&str>) {
        let Some(mut parser) = self.parser.take() else {
            return;
        };
        parser.parse(self, header);
        self.parser = Some(parser);
    }

    /// Java synchronized `addProcessOutput(OutputBufferManager)`.
    pub fn add_process_output_output_buffer_manager(
        &mut self,
        process_output: &OutputBufferManager,
    ) {
        if self.hibernate {
            return;
        }
        self.output_buffer_manager = Some(process_output.get_lines());
        self.parse_with_parser(None);
    }

    /// Java package-private `setMultiParse(boolean)`.  Set this when sending single
    /// strings or parsing during a run.  When multi-parse is in use, you must call
    /// endParse when all the output has been parsed.
    pub fn set_multi_parse(&mut self, multi_parse: bool) {
        if let Some(parser) = &mut self.parser {
            parser.set_multi_parse(multi_parse);
        }
    }

    /// Java package-private `endParse()`.  Must be called after strings are parsed
    /// when multi-parse is on.
    pub fn end_parse(&mut self) {
        let Some(mut parser) = self.parser.take() else {
            return;
        };
        parser.end_parse(self);
        self.parser = Some(parser);
    }

    /// Java synchronized `addProcessOutput(String header, String[] processOutput)`.
    pub fn add_process_output_lines(&mut self, header: Option<&str>, process_output: &[String]) {
        if self.hibernate {
            return;
        }
        self.process_output_string_array = Some(process_output.to_vec());
        self.parse_with_parser(header);
    }

    /// Java synchronized `addProcessOutput(File) throws FileNotFoundException`.
    pub fn add_process_output_file(&mut self, process_output: &Path) -> std::io::Result<()> {
        if self.hibernate {
            return Ok(());
        }
        // Open the file as a stream
        let file_stream = File::open(process_output)?;
        self.buffered_reader = Some(BufReader::new(file_stream).lines());
        self.parse_with_parser(None);
        Ok(())
    }

    /// Java synchronized `addProcessOutput(LogFile.Handle) throws LogFileException,
    /// IOException, LockException`.
    pub fn add_process_output_log_file(
        &mut self,
        process_output: Arc<Handle>,
    ) -> Result<(), LogFileError> {
        if self.hibernate {
            return Ok(());
        }
        // Open the log file
        self.log_file_reader_id = process_output.open_reader()?;
        self.log_file = Some(process_output);
        self.parse_with_parser(None);
        Ok(())
    }

    /// Java synchronized `addProcessOutput(String)`.  When using without string feed,
    /// call setMultiParse(true) before any calls to this function, and endParse after
    /// all parsing is done.
    pub fn add_process_output(&mut self, process_output: Option<&str>) {
        if self.hibernate {
            return;
        }
        // Open the file as a stream
        self.process_output_string = process_output.map(str::to_owned);
        self.parse_with_parser(None);
    }

    /// Java synchronized `add(ProcessMessages)`.
    pub fn add_process_messages(&mut self, process_messages: Option<&ProcessMessages>) {
        if self.hibernate {
            return;
        }
        let Some(process_messages) = process_messages else {
            return;
        };
        self.add_rated_list(MessageType::Error, process_messages.error_list.as_ref());
        self.add_rated_list(MessageType::Warning, process_messages.warning_list.as_ref());
        self.add_rated_list(MessageType::Info, process_messages.info_list.as_ref());
        self.add_rated_list(
            MessageType::ChunkError,
            process_messages.chunk_error_list.as_ref(),
        );
        self.add_rated_list(
            MessageType::ChunkWarning,
            process_messages.chunk_warning_list.as_ref(),
        );
    }

    /// Java synchronized `add(MessageType, String header, ProcessMessages)`.
    pub fn add_from(
        &mut self,
        r#type: MessageType,
        header: Option<&str>,
        process_messages: Option<&ProcessMessages>,
    ) {
        MessageBuilder::add_from_process_messages(self, Some(r#type), header, process_messages);
    }

    /// Java synchronized `add(MessageType, MessageType fromType, String header,
    /// ProcessMessages)`.
    pub fn add_from_type(
        &mut self,
        r#type: MessageType,
        from_type: MessageType,
        header: Option<&str>,
        process_messages: Option<&ProcessMessages>,
    ) {
        MessageBuilder::add_from_type_process_messages(
            self,
            Some(r#type),
            Some(from_type),
            header,
            process_messages,
        );
    }

    /// Java synchronized `add(MessageType, RatedList)`.
    fn add_rated_list(&mut self, r#type: MessageType, input: Option<&RatedList>) {
        let list = input.map(|input| input.get_list().to_vec());
        MessageBuilder::add_list(self, Some(r#type), None, list);
    }

    /// Java synchronized `storeMessage(String, Queue<Message>, boolean)`.
    pub(crate) fn store_message_queue(
        &mut self,
        header: Option<&str>,
        message_queue: Option<&mut Queue<Message>>,
        chunk_message: bool,
    ) {
        MessageBuilder::add_queue(self, header, message_queue, chunk_message, false);
    }

    /// Java synchronized `storeMessage(String, TagInterface, boolean)`.
    pub(crate) fn store_message_tag(
        &mut self,
        header: Option<&str>,
        tags: &mut TagList,
        tag: usize,
        chunk_message: bool,
    ) {
        MessageBuilder::add_tag(self, header, Some((tags, tag)), chunk_message, false);
    }

    /// Java synchronized `add(MessageType, String)`.
    pub fn add_message(&mut self, r#type: MessageType, input: &str) {
        MessageBuilder::add_message(self, Some(r#type), Some(input));
    }

    /// Java synchronized `add(MessageType)`.  Add empty message.
    pub fn add_empty(&mut self, r#type: MessageType) {
        MessageBuilder::add_empty(self, Some(r#type));
    }

    /// Java synchronized `add(MessageType, String header, String[] input)`.
    pub fn add_array(
        &mut self,
        r#type: MessageType,
        header: Option<&str>,
        input: Option<&[String]>,
    ) {
        MessageBuilder::add_array(self, Some(r#type), header, input);
    }

    /// Java `size(MessageType)`.
    pub fn size(&self, r#type: MessageType) -> usize {
        match self.get_list_message_type(Some(r#type)) {
            None => 0,
            Some(list) => list.size(),
        }
    }

    /// Java `get(MessageType, int)`.
    pub fn get(&self, r#type: MessageType, index: usize) -> Option<&str> {
        if let Some(list) = self.get_list_message_type(Some(r#type))
            && index < list.size()
        {
            return Some(list.get(index));
        }
        None
    }

    /// Java `match(MessageType, String[])`.  Returns messages which contain one of the
    /// match strings.
    pub fn match_messages(
        &self,
        r#type: MessageType,
        match_string_array: &[&str],
    ) -> Option<Vec<String>> {
        if self.is_empty(Some(r#type)) {
            return None;
        }
        let mut matches = Vec::new();
        if let Some(list) = self.get_list_message_type(Some(r#type)) {
            for message in list.iterator() {
                for match_string in match_string_array {
                    if message.contains(match_string) {
                        matches.push(message.clone());
                        break;
                    }
                }
            }
            if !matches.is_empty() {
                return Some(matches);
            }
        }
        None
    }

    /// Java `getLast(MessageType)` (deprecated 11/28/16).
    pub fn get_last(&self, r#type: Option<MessageType>) -> Option<&str> {
        let r#type = r#type?;
        let list = self.get_list_message_type(Some(r#type))?;
        if list.size() == 0 {
            return None;
        }
        Some(list.get(list.size() - 1))
    }

    /// Java final package-private `print()`.
    pub fn print_all(&self) {
        self.print(Some(MessageType::Error));
        self.print(Some(MessageType::Warning));
        self.print(Some(MessageType::Info));
    }

    /// Java `isEmpty(MessageType)`.
    pub fn is_empty(&self, r#type: Option<MessageType>) -> bool {
        let Some(r#type) = r#type else {
            return false;
        };
        match self.get_list_message_type(Some(r#type)) {
            None => true,
            Some(list) => list.is_empty(),
        }
    }

    /// Java package-private `isSuccess()`.
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Java `print(MessageType)`.  Returns true if something was printed (only with
    /// `--debug`).
    pub fn print(&self, r#type: Option<MessageType>) -> bool {
        let Some(r#type) = r#type else {
            return false;
        };
        let Some(list) = self.get_list_message_type(Some(r#type)) else {
            return false;
        };
        if *DEBUG {
            for i in 0..list.size() {
                eprintln!("{}", list.get(i));
            }
            return list.size() > 0;
        }
        false
    }

    /// Java `print(ListType)`.  Returns true if something was printed (only with
    /// `--debug`).
    pub fn print_list_type(&self, r#type: Option<ListType>) -> bool {
        let Some(r#type) = r#type else {
            return false;
        };
        let Some(list) = self.get_list(r#type) else {
            return false;
        };
        if *DEBUG {
            for i in 0..list.size() {
                eprintln!("{}", list.get(i));
            }
            return list.size() > 0;
        }
        false
    }

    /// Java static `getErrorIndex(String)`: the index of the first error tag (in tag
    /// order) found in the line; `None` is the source's -1.
    pub fn get_error_index(line: &str) -> Option<usize> {
        for tag in ERROR_TAGS {
            if let Some(index) = line.find(tag) {
                return Some(index);
            }
        }
        None
    }

    /// Java `getNextLine()`.  Uses nextLine to return the next output line.
    pub fn get_next_line(&mut self) -> Option<String> {
        if self.next_line() {
            return self.current_line.line.clone();
        }
        None
    }

    /// Java private `nextLine()`.  Figure out which type of process output is being
    /// read and call the corresponding nextLine function.  On the string feed thread
    /// the string taken from the feed is loaded (Java waits for one; here the parse
    /// thread takes it before it parses, see the module comment).
    fn next_line(&mut self) -> bool {
        // A message with an end tag ends before the end of the line.  In this case
        // process the rest of the line.
        if !self.current_line.is_done() && Line::next_message(self) {
            return true;
        }
        // String feed thread - do NOT wait for a string, except on the string feed
        // thread.
        if self.string_feed.is_some()
            && self.string_feed_thread == Some(std::thread::current().id())
        {
            // This is the string feed thread - take the string
            let Some(string) = self.string_feed_pending.take() else {
                return false;
            };
            Line::load(self, Some(string));
            return !self.current_line.is_end_feed();
        }
        let mut retval = false;
        if self.output_buffer_manager.is_some() {
            retval = self.next_output_buffer_manager_line();
        } else if self.buffered_reader.is_some() {
            retval = self.next_buffered_reader_line();
        } else if self.log_file.is_some() {
            retval = self.next_log_file_line();
        } else if self.process_output_string.is_some() {
            let process_output_string = self.process_output_string.take();
            Line::load(self, process_output_string);
            retval = true;
        } else if self.process_output_string_array.is_some() {
            retval = self.next_string_array_line();
        }
        retval
    }

    /// Java private `nextOutputBufferManagerLine()`.  Increment index and place the
    /// trimmed entry at index into the current line; at the end set the source to null
    /// and the index to -1.
    fn next_output_buffer_manager_line(&mut self) -> bool {
        self.index += 1;
        let size = self.output_buffer_manager.as_ref().unwrap().len() as i32;
        if self.index >= size {
            self.index = -1;
            self.output_buffer_manager = None;
            self.current_line.reset();
            return false;
        }
        let line = java_lang_string_trim(
            &self.output_buffer_manager.as_ref().unwrap()[self.index as usize],
        );
        Line::load(self, Some(line));
        true
    }

    /// Java private `nextStringArrayLine()`.
    fn next_string_array_line(&mut self) -> bool {
        self.index += 1;
        let length = self.process_output_string_array.as_ref().unwrap().len() as i32;
        if self.index >= length {
            self.index = -1;
            self.process_output_string_array = None;
            self.current_line.reset();
            return false;
        }
        let line = java_lang_string_trim(
            &self.process_output_string_array.as_ref().unwrap()[self.index as usize],
        );
        Line::load(self, Some(line));
        true
    }

    /// Java private `nextBufferedReaderLine()`.  Does not change index.
    fn next_buffered_reader_line(&mut self) -> bool {
        match self.buffered_reader.as_mut().unwrap().next() {
            None => {
                self.current_line.reset();
                self.buffered_reader = None;
                false
            }
            Some(Ok(line)) => {
                Line::load(self, Some(line));
                true
            }
            Some(Err(e)) => {
                // `e.printStackTrace()`
                eprintln!("{e}");
                self.buffered_reader = None;
                self.current_line.reset();
                false
            }
        }
    }

    /// Java private `nextLogFileLine()`.
    fn next_log_file_line(&mut self) -> bool {
        let log_file = self.log_file.clone().unwrap();
        let result = match &self.log_file_reader_id {
            Some(reader_id) => log_file.read_line(reader_id),
            // `LogFile.readLine` with a null reader id fails its lock test.
            None => Err(LogFileError::Unlocked(
                crate::imod::etomo::storage::log_file::UnlockedException::new_id(None, None),
            )),
        };
        match result {
            Ok(None) => {
                self.current_line.reset();
                log_file.close_id(self.log_file_reader_id.as_deref());
                self.log_file = None;
                self.log_file_reader_id = None;
                false
            }
            Ok(Some(line)) => {
                Line::load(self, Some(line));
                true
            }
            // `catch (final LogFileException e)` / `catch (IOException e)`
            Err(e) => {
                eprintln!("{e}");
                log_file.close_id(self.log_file_reader_id.as_deref());
                self.log_file = None;
                self.log_file_reader_id = None;
                self.current_line.reset();
                false
            }
        }
    }

    /// Java private `getList(MessageType, boolean)` (without creating).
    fn get_list_message_type(&self, r#type: Option<MessageType>) -> Option<&RatedList> {
        self.get_list(r#type?.list_type()?)
    }

    /// Java `iterator(ListType)`.
    pub fn iterator(&self, list_type: ListType) -> Option<impl Iterator<Item = &String>> {
        self.get_list(list_type).map(RatedList::iterator)
    }

    /// Java private `getList(ListType, false)`.
    fn get_list(&self, r#type: ListType) -> Option<&RatedList> {
        match r#type {
            ListType::ChunkError => self.chunk_error_list.as_ref(),
            ListType::ChunkWarning => self.chunk_warning_list.as_ref(),
            ListType::Error => self.error_list.as_ref(),
            ListType::Info => self.info_list.as_ref(),
            ListType::Warning => self.warning_list.as_ref(),
            ListType::Logged | ListType::Flag => None,
        }
    }

    /// Java private `getList(ListType, true)`.
    fn get_list_create(&mut self, r#type: Option<ListType>) -> Option<&mut RatedList> {
        match r#type? {
            ListType::ChunkError => Some(self.chunk_error_list.get_or_insert_with(RatedList::new)),
            ListType::ChunkWarning => {
                Some(self.chunk_warning_list.get_or_insert_with(RatedList::new))
            }
            ListType::Error => Some(self.error_list.get_or_insert_with(RatedList::new)),
            ListType::Info => Some(self.info_list.get_or_insert_with(RatedList::new)),
            ListType::Warning => Some(self.warning_list.get_or_insert_with(RatedList::new)),
            ListType::Logged | ListType::Flag => None,
        }
    }

    /// Java synchronized `setSecondaryLog(AxisID, File)`.
    pub fn set_secondary_log(
        &mut self,
        axis_id: Option<AxisID>,
        secondary_log_file: Option<&Path>,
    ) {
        self.reset_secondary_log();
        if let Some(secondary_log_file) = secondary_log_file {
            let secondary_log = self
                .secondary_log
                .get_or_insert_with(|| Arc::new(Mutex::new(FileWriter::new())));
            secondary_log
                .lock()
                .unwrap()
                .set_file(self.manager, axis_id, Some(secondary_log_file));
        }
    }

    /// Java synchronized `getSecondaryLog()`.
    pub fn get_secondary_log(&self) -> Option<FileWriterRef> {
        self.secondary_log.clone()
    }

    /// Java synchronized `resetSecondaryLog()`.
    pub fn reset_secondary_log(&mut self) {
        if let Some(secondary_log) = &self.secondary_log {
            secondary_log.lock().unwrap().reset();
        }
    }

    /// Java synchronized `closeSecondaryLog()`.
    pub fn close_secondary_log(&mut self) {
        if let Some(secondary_log) = &self.secondary_log {
            secondary_log.lock().unwrap().close();
        }
    }

    /// Java synchronized `isSecondaryLogOpen()`.
    pub fn is_secondary_log_open(&self) -> bool {
        if let Some(secondary_log) = &self.secondary_log {
            return secondary_log.lock().unwrap().is_open();
        }
        false
    }
}

/// A copy of the object's state.  Java hands the same `ProcessMessages` reference to
/// the event dispatch thread (message popups, the log); a reader on the other thread
/// gets this copy instead.  An open file or log-file reader stays with the original.
impl Clone for ProcessMessages {
    fn clone(&self) -> ProcessMessages {
        ProcessMessages {
            chunks: self.chunks,
            manager: self.manager,
            axis_id: self.axis_id,
            log_all_messages: self.log_all_messages,
            error_override_log_tag: self.error_override_log_tag.clone(),
            error_tags: self.error_tags.clone(),
            parser: self.parser.clone(),
            error_tags_always_multiline: self.error_tags_always_multiline.clone(),
            current_line: self.current_line.clone(),
            log_info_messages: self.log_info_messages,
            message_builder: self.message_builder.clone(),
            output_buffer_manager: self.output_buffer_manager.clone(),
            buffered_reader: None,
            process_output_string: self.process_output_string.clone(),
            process_output_string_array: self.process_output_string_array.clone(),
            log_file: None,
            log_file_reader_id: None,
            index: self.index,
            info_list: self.info_list.clone(),
            warning_list: self.warning_list.clone(),
            error_list: self.error_list.clone(),
            chunk_error_list: self.chunk_error_list.clone(),
            chunk_warning_list: self.chunk_warning_list.clone(),
            success_tag1: self.success_tag1.clone(),
            success_tag2: self.success_tag2.clone(),
            success: self.success,
            multi_line_all_messages: self.multi_line_all_messages,
            multi_line_warning: self.multi_line_warning,
            multi_line_info: self.multi_line_info,
            message_prepend_tag: self.message_prepend_tag.clone(),
            string_feed: None,
            string_feed_thread: None,
            string_feed_done: None,
            string_feed_pending: None,
            hibernate: self.hibernate,
            allow_multi_line_log: self.allow_multi_line_log,
            debug: self.debug,
            secondary_log: self.secondary_log.clone(),
        }
    }
}

/// Java `toString()`.
impl std::fmt::Display for ProcessMessages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for list in [
            &self.info_list,
            &self.warning_list,
            &self.error_list,
            &self.chunk_error_list,
            &self.chunk_warning_list,
        ] {
            if let Some(list) = list {
                for i in 0..list.size() {
                    writeln!(f, "{}", list.get(i))?;
                }
            }
        }
        Ok(())
    }
}

/// Java private final inner `ParseThread implements Runnable`.
struct ParseThread;

impl ParseThread {
    /// Java `run()`.  Gets lines from stringFeed (must be set up with
    /// startStringFeed).  Stops when the END_STRING_FEED_TOKEN is received.  Clears the
    /// string feed queue when it ends.  Each string is taken without the lock (Java's
    /// blocking `take`) and parsed with it.
    fn run(this: &MessagesRef, receiver: Receiver<String>, done: &Arc<(Mutex<bool>, Condvar)>) {
        loop {
            if this.lock().unwrap().current_line.is_end_feed() {
                break;
            }
            // `stringFeed.take()`
            let string = receiver
                .recv()
                .unwrap_or_else(|_| END_FEED_TOKEN.to_owned());
            let mut process_messages = this.lock().unwrap();
            process_messages.string_feed_pending = Some(string);
            process_messages.parse_with_parser(None);
        }
        {
            // synchronized (stringFeedLock)
            let mut process_messages = this.lock().unwrap();
            process_messages.string_feed = None;
            process_messages.string_feed_thread = None;
        }
        let (lock, condvar) = &**done;
        *lock.lock().unwrap() = true;
        condvar.notify_all();
    }
}

/// Java package-private static final nested `ProcessMessagesTestHarness`.
pub struct ProcessMessagesTestHarness;

impl ProcessMessagesTestHarness {
    /// Java `ProcessMessagesTestHarness()`.
    pub fn new() -> ProcessMessagesTestHarness {
        ProcessMessagesTestHarness
    }

    /// Java `testSubLine()`.  Returns null if succeeded, or error message.
    pub fn test_sub_line(&self) -> Option<String> {
        let tag = "Tag:";
        let orig_tag = format!("First {tag}");
        let not_the_tag = "hello:";
        let sub_line = SubLine::get_instance(
            Some(&format!(
                "{orig_tag} blah1 {tag}{not_the_tag} {tag} blah2{tag}blah3 "
            )),
            0,
            Some(&orig_tag),
            Some(tag),
        );
        let expected = format!("{tag}{not_the_tag}\n{tag} blah2\n{tag}blah3");
        let actual = sub_line.map(|sub_line| sub_line.get_message());
        if actual.as_deref() != Some(expected.as_str()) {
            return Some(
                "ProcessMessagesTestHarness.assertSubLine failed:each subLine returns its \
                 portion of the line (trimmed),and inserts a newline before the next portion."
                    .to_owned(),
            );
        }
        None
    }

    /// Java `testSecondaryLogTag()`.  Returns null if succeeded, or error message.
    pub fn test_secondary_log_tag(&self) -> Option<String> {
        let mut process_messages = ProcessMessages::get_instance(None, AxisID::Only);
        let msg = "This message should be logged.";
        Line::load(&mut process_messages, Some(format!("{msg}[:LOG]")));
        if process_messages.current_line.message_type != Some(MessageType::Log) {
            return Some(format!(
                "ending LOG tag was not recognized, processMessages.currentLine.messageType:{}",
                process_messages
                    .current_line
                    .message_type
                    .map_or("null".to_owned(), |message_type| message_type.to_string())
            ));
        }
        // `currentLine.equals(msg)` is `Object.equals` between a Line and a String,
        // which is false.
        None
    }
}

impl Default for ProcessMessagesTestHarness {
    fn default() -> ProcessMessagesTestHarness {
        ProcessMessagesTestHarness::new()
    }
}

/// Java private static final nested `SubLine`.  Class to handle a line with further
/// tags.  Only one tag string can be checked.  Does not strip tags.  Doesn't handle
/// start and end tags.
#[derive(Clone, Debug)]
struct SubLine {
    /// Java private final `line`.
    line: String,
    /// Java private final `startIndex`.
    start_index: i32,
    /// Java private final `endIndex`.
    end_index: i32,
    /// Java private final `next`.
    next: Option<Box<SubLine>>,
}

impl SubLine {
    /// Java private `SubLine(String, int, String)`.
    fn new(line: &str, start_index: i32, tag: &str) -> SubLine {
        let next = SubLine::get_instance(Some(line), start_index, Some(tag), Some(tag));
        let end_index = match &next {
            Some(next) => next.start_index,
            None => -1,
        };
        SubLine {
            line: line.to_owned(),
            start_index,
            end_index,
            next: next.map(Box::new),
        }
    }

    /// Java private static `getInstance(String, int, String, String)`.  Returns an
    /// instance of SubLine if tag is found after lastIndex + lastTag.length.
    fn get_instance(
        line: Option<&str>,
        last_index: i32,
        last_tag: Option<&str>,
        tag: Option<&str>,
    ) -> Option<SubLine> {
        let (Some(line), Some(last_tag), Some(tag)) = (line, last_tag, tag) else {
            return None;
        };
        let from = (last_index + last_tag.len() as i32) as usize;
        if from > line.len() {
            return None;
        }
        let index = line[from..].find(tag)? + from;
        Some(SubLine::new(line, index as i32, tag))
    }

    /// Java private `getMessage()`.
    fn get_message(&self) -> String {
        self.substring()
            + &match &self.next {
                None => String::new(),
                Some(next) => "\n".to_owned() + &next.get_message(),
            }
    }

    /// Java private `substring()`.
    fn substring(&self) -> String {
        line_substring(&self.line, self.start_index, self.end_index)
    }
}

/// The body shared by Java's `SubLine.substring()` and `Line.substring()` (two copies of
/// the same code in the source).
fn line_substring(line: &str, start_index: i32, end_index: i32) -> String {
    let length = line.len() as i32;
    if start_index >= length || (start_index >= 0 && end_index >= 0 && start_index >= end_index) {
        // Empty string
        return String::new();
    }
    let use_start_index = start_index > 0;
    let use_end_index = end_index >= 0 && end_index < length;
    if line.is_empty() {
        return line.to_owned();
    }
    let substring = if use_start_index || use_end_index {
        if use_start_index && use_end_index {
            &line[start_index as usize..end_index as usize]
        } else if use_start_index {
            &line[start_index as usize..]
        } else {
            &line[..end_index as usize]
        }
    } else {
        line
    };
    java_lang_string_trim(substring)
}

/// Java package-private final inner `Line`.
#[derive(Clone, Debug)]
struct Line {
    /// Java private `line`.
    line: Option<String>,
    /// Java private `messageType`.
    message_type: Option<MessageType>,
    /// Java private `endMessageType`.
    end_message_type: Option<MessageType>,
    /// Java private `startIndex`.
    start_index: i32,
    /// Java private `endIndex`.
    end_index: i32,
    /// Java private `tagIndex`: for list types with arrays of tags.  Refers to the tag
    /// array that existed when load() was run.
    tag_index: i32,
    /// Java private `alwaysMultiLine`.
    always_multi_line: bool,
    /// Java private `subLine`.
    sub_line: Option<SubLine>,
    /// Java private `truncateLine`.
    truncate_line: bool,
}

impl Line {
    /// Java private `Line()`.
    fn new() -> Line {
        Line {
            line: None,
            message_type: None,
            end_message_type: None,
            start_index: -1,
            end_index: -1,
            tag_index: -1,
            always_multi_line: false,
            sub_line: None,
            truncate_line: false,
        }
    }

    /// Java private `reset()`.  `alwaysMultiLine` is not reset in the source.
    fn reset(&mut self) {
        self.line = None;
        self.message_type = None;
        self.end_message_type = None;
        self.start_index = -1;
        self.end_index = -1;
        self.tag_index = -1;
        self.sub_line = None;
        self.truncate_line = false;
    }

    /// Java private `getMessage()`.  Preferred way to retrieve the line.  Line is
    /// modified based on the message type.
    fn get_message(&self) -> String {
        self.substring()
            + &match &self.sub_line {
                None => String::new(),
                Some(sub_line) => "\n".to_owned() + &sub_line.get_message(),
            }
    }

    /// Java private `substring()`.  `line.length()` on a null line throws in the
    /// source; it is only called on a loaded line.
    fn substring(&self) -> String {
        line_substring(
            self.line
                .as_deref()
                .expect("java.lang.NullPointerException"),
            self.start_index,
            self.end_index,
        )
    }

    /// Java private `isDone()`.  Done if there is no end index, or it goes to the end
    /// of the line.  Assumes that no end index tag gets stripped.
    fn is_done(&self) -> bool {
        match &self.line {
            None => true,
            Some(line) => {
                self.truncate_line
                    || self.end_index < 0
                    || self.end_index >= line.len() as i32
                    || self.sub_line.is_some()
            }
        }
    }

    /// Java private `isEndFeed()`.
    fn is_end_feed(&self) -> bool {
        self.line.as_deref() == Some(END_FEED_TOKEN)
    }

    /// Java private `nextMessage()`.  Returns true if the line was not done, and
    /// load() was called on the part of the line after the end tag.
    fn next_message(process_messages: &mut ProcessMessages) -> bool {
        if process_messages.current_line.is_done() {
            return false;
        }
        let line = &process_messages.current_line;
        let rest = line.line.as_deref().unwrap()[line.end_index as usize..].to_owned();
        Line::load(process_messages, Some(rest));
        true
    }

    /// Java private `isEmpty()`.
    fn is_empty(&self) -> bool {
        self.line.as_deref().is_none_or(str::is_empty)
    }

    /// Java `equals(MessageType, int)`.  Returns true if the line contains the same
    /// tag described by the parameters.  `inputTagIndex` is only used for ERROR list
    /// types.
    fn equals(&self, input_message_type: Option<MessageType>, input_tag_index: i32) -> bool {
        if input_message_type.is_none() && self.message_type.is_none() {
            return true;
        }
        if input_message_type == self.message_type {
            if self.message_type != Some(MessageType::Error) {
                return true;
            }
            return self.tag_index == input_tag_index;
        }
        false
    }

    /// Java private `load(String)`.  Find the message type of a line.  Save
    /// information about the tag which matched: where is was in the line, and which tag
    /// it was.  The inner class reads its `ProcessMessages`' settings.
    fn load(process_messages: &mut ProcessMessages, line: Option<String>) {
        let chunks = process_messages.chunks;
        let multi_line_all_messages = process_messages.multi_line_all_messages;
        let success_tag1 = process_messages.success_tag1.clone();
        let success_tag2 = process_messages.success_tag2.clone();
        let message_prepend_tag = process_messages.message_prepend_tag.clone();
        let error_tags = process_messages.error_tags.clone();
        let error_tags_always_multiline = process_messages.error_tags_always_multiline.clone();
        let this = &mut process_messages.current_line;
        this.reset();
        this.line = line;
        let Some(line) = this.line.clone() else {
            return;
        };
        let find = |tag: &str| line.find(tag).map_or(-1, |index| index as i32);
        this.start_index = find(MessageType::PipWarningStart.tag().unwrap());
        if this.start_index != -1 {
            this.message_type = Some(MessageType::PipWarningStart);
        } else {
            this.end_index = find(MessageType::PipWarningEnd.tag().unwrap());
            if this.end_index != -1 {
                // end tag - do not strip tag.
                this.end_index += MessageType::PipWarningEnd.tag().unwrap().len() as i32;
                if this.message_type.is_none() {
                    this.start_index = 0;
                    this.message_type = Some(MessageType::PipWarningEnd);
                } else {
                    this.end_message_type = Some(MessageType::PipWarningEnd);
                }
            } else {
                this.start_index = find(MessageType::LogFile.tag().unwrap());
                if this.start_index != -1 {
                    this.message_type = Some(MessageType::LogFile);
                    // The tag should be striped from log file messages.
                    this.start_index += MessageType::LogFile.tag().unwrap().len() as i32;
                } else {
                    this.start_index = find(MessageType::Warning.tag().unwrap());
                    if this.start_index != -1 {
                        this.message_type = Some(MessageType::Warning);
                    } else if chunks && {
                        this.start_index = find(MessageType::ChunkError.tag().unwrap());
                        this.start_index != -1
                    } {
                        // CHUNK_ERROR takes precedence over ERROR
                        this.message_type = Some(MessageType::ChunkError);
                        // Errors may be added to the chunk error line.  They are errors
                        // associated with the chunk and should not be treated as process
                        // errors, so put them on separate lines but keep them with the
                        // chunk error.  This is only needed for single-line chunk errors
                        if !multi_line_all_messages {
                            this.sub_line = SubLine::get_instance(
                                Some(&line),
                                this.start_index,
                                MessageType::ChunkError.tag(),
                                MessageType::Error.tag(),
                            );
                            if let Some(sub_line) = &this.sub_line {
                                this.end_index = sub_line.start_index;
                            }
                        }
                    } else {
                        this.start_index = find(MessageType::Info.tag().unwrap());
                        if this.start_index != -1 {
                            this.message_type = Some(MessageType::Info);
                        } else {
                            this.start_index = find(MessageType::Log.tag().unwrap());
                            if this.start_index != -1 {
                                this.message_type = Some(MessageType::Log);
                                // The tag should be striped from log messages.
                                this.start_index += MessageType::Log.tag().unwrap().len() as i32;
                            } else {
                                this.end_index = find(MessageType::Log.secondary_tag().unwrap());
                                if this.end_index != -1 {
                                    this.message_type = Some(MessageType::Log);
                                    // The tag, and everything after it, should be
                                    // stripped from the log message.
                                    this.truncate_line = true;
                                } else {
                                    // look for an error message
                                    for (i, error_tag) in error_tags.iter().enumerate() {
                                        this.start_index = find(error_tag);
                                        if this.start_index != -1 {
                                            // Check list of text that looks like an error
                                            // message but isn't.
                                            for ignore_tag in IGNORE_TAG {
                                                if line.contains(ignore_tag) {
                                                    this.start_index = -1;
                                                    break;
                                                }
                                            }
                                            if this.start_index != -1 {
                                                this.message_type = Some(MessageType::Error);
                                                this.tag_index = i as i32;
                                                this.always_multi_line =
                                                    error_tags_always_multiline[i];
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if this.message_type.is_none() && (success_tag1.is_some() || success_tag2.is_some()) {
            // Look for success tags
            // Looks for success tags in the line.  Sets success = true if all set tags
            // are found.  Tags are checked in order.  Tag 1 must come first and the tags
            // must not overlap.
            if let (Some(success_tag1), Some(success_tag2)) = (&success_tag1, &success_tag2) {
                this.start_index = find(success_tag1);
                if this.start_index != -1 {
                    let substring = &line[this.start_index as usize + success_tag1.len()..];
                    if substring.contains(success_tag2.as_str()) {
                        this.message_type = Some(MessageType::Success);
                    }
                }
            } else if let Some(success_tag1) = &success_tag1 {
                this.start_index = find(success_tag1);
                if this.start_index != -1 {
                    this.message_type = Some(MessageType::Success);
                }
            } else {
                this.start_index = find(success_tag2.as_deref().unwrap());
                if this.start_index != -1 {
                    this.message_type = Some(MessageType::Success);
                }
            }
        }
        if this.message_type.is_none()
            && let Some(message_prepend_tag) = &message_prepend_tag
        {
            this.start_index = find(message_prepend_tag);
            if this.start_index != -1 {
                this.message_type = Some(MessageType::Prepend);
            }
        }
    }
}

/// Java `toString()`.
impl std::fmt::Display for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[messageType:{},startIndex:{},tagIndex:{}\n{}",
            self.message_type
                .map_or("null".to_owned(), |message_type| message_type.to_string()),
            self.start_index,
            self.tag_index,
            self.line.as_deref().unwrap_or("null")
        )
    }
}

/// Java private final inner `MessageBuilder`.  Adds messages to a list specified by
/// MessageType.  Filters out uninformative messages.  Handles messages from any
/// source.  Handles redirecting messages to the log - and the override which
/// redirects messages back the a list.  Handles duplicate chunk messages.  The inner
/// class's methods take the `ProcessMessages` they belong to.
#[derive(Clone, Debug)]
struct MessageBuilder {
    /// Java private final `fromIterator`.
    from_iterator: MultiSourceStringIterator,
    /// Java private `emptyToList`.
    empty_to_list: bool,
}

impl MessageBuilder {
    /// Java private `MessageBuilder()`.
    fn new() -> MessageBuilder {
        MessageBuilder {
            from_iterator: MultiSourceStringIterator::new(),
            empty_to_list: false,
        }
    }

    /// Java private `add(MessageType)`.  Adds an empty message to a list denoted by
    /// messageType.
    fn add_empty(process_messages: &mut ProcessMessages, message_type: Option<MessageType>) {
        MessageBuilder::add_all(
            process_messages,
            message_type,
            None,
            None,
            None,
            Some(""),
            None,
            None,
            false,
            true,
        );
    }

    /// Java private `add(MessageType, String)`.  Adds message to a list denoted by
    /// messageType.
    fn add_message(
        process_messages: &mut ProcessMessages,
        message_type: Option<MessageType>,
        from_message: Option<&str>,
    ) {
        MessageBuilder::add_all(
            process_messages,
            message_type,
            None,
            None,
            None,
            from_message,
            None,
            None,
            false,
            true,
        );
    }

    /// Java private `add(MessageType, String, ProcessMessages)`.  Adds messages to a
    /// list denoted by messageType; header is added to toList right before the first
    /// message is added.
    fn add_from_process_messages(
        process_messages: &mut ProcessMessages,
        message_type: Option<MessageType>,
        header: Option<&str>,
        from_process_messages: Option<&ProcessMessages>,
    ) {
        if let (Some(message_type), Some(from_process_messages)) =
            (message_type, from_process_messages)
        {
            let list = from_process_messages
                .get_list_message_type(Some(message_type))
                .map(|list| list.get_list().to_vec());
            MessageBuilder::add_all(
                process_messages,
                Some(message_type),
                None,
                None,
                header,
                None,
                list,
                None,
                false,
                true,
            );
        }
    }

    /// Java private `add(MessageType, MessageType, String, ProcessMessages)`.
    fn add_from_type_process_messages(
        process_messages: &mut ProcessMessages,
        message_type: Option<MessageType>,
        from_message_type: Option<MessageType>,
        header: Option<&str>,
        from_process_messages: Option<&ProcessMessages>,
    ) {
        if let (Some(message_type), Some(from_process_messages)) =
            (message_type, from_process_messages)
        {
            let list = from_process_messages
                .get_list_message_type(from_message_type)
                .map(|list| list.get_list().to_vec());
            MessageBuilder::add_all(
                process_messages,
                Some(message_type),
                from_message_type,
                None,
                header,
                None,
                list,
                None,
                false,
                true,
            );
        }
    }

    /// Java private `add(MessageType, String, List<String>)`.
    fn add_list(
        process_messages: &mut ProcessMessages,
        message_type: Option<MessageType>,
        header: Option<&str>,
        from_list: Option<Vec<String>>,
    ) {
        MessageBuilder::add_all(
            process_messages,
            message_type,
            None,
            None,
            header,
            None,
            from_list,
            None,
            false,
            true,
        );
    }

    /// Java private `add(MessageType, String, String[])`.
    fn add_array(
        process_messages: &mut ProcessMessages,
        message_type: Option<MessageType>,
        header: Option<&str>,
        from_array: Option<&[String]>,
    ) {
        MessageBuilder::add_all(
            process_messages,
            message_type,
            None,
            None,
            header,
            None,
            None,
            from_array.map(<[String]>::to_vec),
            false,
            true,
        );
    }

    /// Java private `add(String, Queue<Message>, boolean, boolean)`.  Add messages
    /// from messageQueue - destructively - using remove.
    fn add_queue(
        process_messages: &mut ProcessMessages,
        header: Option<&str>,
        message_queue: Option<&mut Queue<Message>>,
        chunk_message: bool,
        allow_log_override: bool,
    ) {
        let Some(message_queue) = message_queue else {
            return;
        };
        while !message_queue.is_empty() {
            if let Some(mut message) = message_queue.pop() {
                let message_string = message.get_message_string();
                MessageBuilder::add_all(
                    process_messages,
                    Some(message.get_message_type()),
                    None,
                    message.get_list_type(),
                    header,
                    Some(&message_string),
                    None,
                    None,
                    chunk_message,
                    allow_log_override,
                );
            }
        }
    }

    /// Java private `add(String, TagInterface, boolean, boolean)`.
    fn add_tag(
        process_messages: &mut ProcessMessages,
        header: Option<&str>,
        tag: Option<(&mut TagList, usize)>,
        chunk_message: bool,
        allow_log_override: bool,
    ) {
        let Some((tags, tag)) = tag else {
            return;
        };
        let message_type = tags.get_message_type(tag);
        let list_type = tags.get_list_type(tag);
        let message_string = tags.get_message_string(tag);
        MessageBuilder::add_all(
            process_messages,
            Some(message_type),
            None,
            list_type,
            header,
            message_string.as_deref(),
            None,
            None,
            chunk_message,
            allow_log_override,
        );
    }

    /// Java private synchronized `add(MessageType, MessageType, ListType, String,
    /// String, List<String>, String[], boolean, boolean)`.  Adds messages to a list
    /// denoted by messageType.  This function takes all sources of messages.
    #[allow(clippy::too_many_arguments)]
    fn add_all(
        process_messages: &mut ProcessMessages,
        message_type: Option<MessageType>,
        mut from_message_type: Option<MessageType>,
        mut list_type: Option<ListType>,
        mut header: Option<&str>,
        from_message: Option<&str>,
        from_list: Option<Vec<String>>,
        from_array: Option<Vec<String>>,
        chunk_message: bool,
        allow_log_override: bool,
    ) {
        if process_messages.hibernate {
            return;
        }
        if message_type == Some(MessageType::Success) {
            process_messages.success = true;
            return;
        }
        if from_message_type.is_none() {
            from_message_type = message_type;
        }
        // Set list type.
        if list_type.is_none()
            && let Some(message_type) = message_type
        {
            list_type = message_type.get_list_type();
        }
        if chunk_message {
            if list_type == Some(ListType::Error) {
                list_type = Some(ListType::ChunkError);
            }
            if list_type == Some(ListType::Warning) {
                list_type = Some(ListType::ChunkWarning);
            }
        }
        // Non-logged messages will be stored in toList.
        let to_list_exists = process_messages.get_list_create(list_type).is_some();
        if !to_list_exists {
            // May be putting homeless messages into the etomo error log.
            // Prevent large stretches of empty error log.
            if !process_messages.message_builder.empty_to_list {
                process_messages.message_builder.empty_to_list = true;
                eprintln!();
            }
        } else {
            process_messages.message_builder.empty_to_list = false;
        }
        // settings
        let mut prevent_duplicates = false;
        let mut log_override_tag: Option<String> = None;
        if !process_messages.log_all_messages {
            if from_message_type == Some(MessageType::ChunkError)
                || from_message_type == Some(MessageType::ChunkWarning)
            {
                prevent_duplicates = true;
            }
        } else if allow_log_override && message_type == Some(MessageType::Error) {
            if let Some(error_override_log_tag) = &process_messages.error_override_log_tag {
                log_override_tag = Some(error_override_log_tag.clone());
            }
        }
        let log_messages = process_messages.log_all_messages || list_type == Some(ListType::Logged);
        let mut log_header = header;
        // Reuse the iterator.
        process_messages
            .message_builder
            .from_iterator
            .reset(from_message, from_list, from_array);
        // go through from messages and put the keepers into toList
        while process_messages.message_builder.from_iterator.has_next() {
            let message = process_messages.message_builder.from_iterator.next();
            // Rate the message and store it or drop it according to the rating.
            let rating = MessageRating::rate(message.as_deref());
            if rating.is_drop() {
                continue;
            }
            let message = message.unwrap();
            // ToList is a RatedList.  It will only save the highest rated messages it
            // receives so the alternative messages will only be used if there aren't
            // any keepers.
            if !log_messages
                || log_override_tag
                    .as_deref()
                    .is_some_and(|log_override_tag| message.contains(log_override_tag))
            {
                if MessageBuilder::add_message_to_list(
                    process_messages,
                    header,
                    &message,
                    prevent_duplicates,
                    list_type,
                    rating,
                ) {
                    header = None;
                }
            } else if MessageBuilder::log_message(
                process_messages,
                message_type,
                log_header,
                &message,
            ) {
                log_header = None;
            }
        }
    }

    /// Java private `addMessage(String, String, boolean, RatedList, MessageRating)`.
    /// Add an unused header and a message to toList.  Returns true if message added.
    fn add_message_to_list(
        process_messages: &mut ProcessMessages,
        header: Option<&str>,
        message: &str,
        prevent_duplicates: bool,
        list_type: Option<ListType>,
        rating: MessageRating,
    ) -> bool {
        let Some(to_list) = process_messages.get_list_create(list_type) else {
            if let Some(header) = header {
                eprintln!("\n{header}");
            }
            eprintln!("{message}");
            return true;
        };
        if prevent_duplicates && to_list.contains(message) {
            return false;
        }
        to_list.add_rated(header, message, Some(rating))
    }

    /// Java private `logMessage(MessageType, String, String)`.  Log a message.
    /// Returns true if message logged or placed in the error log because there is no
    /// manager.
    ///
    /// Upstream bugs fixed in translation (ProcessMessages.java:2187, 2223): a
    /// LOGFILE message with a null manager reads `manager.getPropertyUserDir()`, and
    /// the debug branch with a null manager logs the header through the manager; both
    /// throw `NullPointerException`.  Here the file is taken relative to the current
    /// directory and the header is printed to standard error (BUGS.md).
    fn log_message(
        process_messages: &mut ProcessMessages,
        message_type: Option<MessageType>,
        header: Option<&str>,
        message: &str,
    ) -> bool {
        let manager = process_messages.manager;
        let secondary_log = process_messages.secondary_log.clone();
        if message_type == Some(MessageType::LogFile) {
            // Log from a file
            let file = match manager.and_then(|manager| manager.get_property_user_dir()) {
                Some(property_user_dir) => {
                    PathBuf::from(utilities::java_io_file_new(&property_user_dir, message))
                }
                None => PathBuf::from(message),
            };
            if file.exists() && file.is_file() && File::open(&file).is_ok() {
                if let Some(manager) = manager {
                    if let Some(header) = header {
                        manager.log_simple_message(Some(header), secondary_log.clone());
                    }
                    manager.log_simple_message_file(Some(&file), secondary_log);
                    return true;
                }
                if *DEBUG {
                    if let Some(header) = header {
                        eprintln!("{header}");
                    }
                    eprintln!(
                        "Messages logged in {}",
                        utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                    );
                    return true;
                }
            } else if *DEBUG {
                if let Some(header) = header {
                    eprintln!("{header}");
                }
                eprintln!(
                    "Warning: unable to log from file:{}",
                    utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                );
                return false;
            }
        }
        // Log message
        else if let Some(manager) = manager {
            if let Some(header) = header {
                manager.log_simple_message(Some(header), secondary_log.clone());
            }
            manager.log_simple_message(Some(message), secondary_log);
            return true;
        }
        if *DEBUG {
            if let Some(header) = header {
                eprintln!("{header}");
            }
            eprintln!("{message}");
            return true;
        }
        false
    }
}

/// Java private static final nested `MessageRating`: four identity-compared instances.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MessageRating {
    /// Java private final `rating`.
    rating: i32,
    /// Java private final `maxToStore`.
    max_to_store: i32,
}

impl MessageRating {
    /// Java `DROP`.
    const DROP: MessageRating = MessageRating {
        rating: 0,
        max_to_store: 0,
    };
    /// Java `ALTERNATIVE_B`.
    const ALTERNATIVE_B: MessageRating = MessageRating {
        rating: 1,
        max_to_store: 1,
    };
    /// Java `ALTERNATIVE_A`.
    const ALTERNATIVE_A: MessageRating = MessageRating {
        rating: 2,
        max_to_store: 1,
    };
    /// Java `KEEPER`.
    const KEEPER: MessageRating = MessageRating {
        rating: 3,
        max_to_store: -1,
    };
    /// Java `MAX = KEEPER`.
    const MAX: MessageRating = MessageRating::KEEPER;

    /// Java private static `rate(String)`.  Does not return null.
    fn rate(message: Option<&str>) -> MessageRating {
        // return null for all uninformative messages
        let Some(message) = message else {
            return MessageRating::DROP;
        };
        if java_lang_string_matches_whitespace(message)
            || (message.contains("PID:") && message.encode_utf16().count() <= 20)
        {
            return MessageRating::DROP;
        }
        if message.contains("exited with status 1") {
            if message.contains("python -u") {
                return MessageRating::ALTERNATIVE_B;
            }
            return MessageRating::ALTERNATIVE_A;
        }
        MessageRating::KEEPER
    }

    /// Java private `isDrop()`.
    fn is_drop(self) -> bool {
        self.max_to_store == 0
    }

    /// Java private `isStore(int)`.
    fn is_store(self, already_stored: i32) -> bool {
        if self.max_to_store == 0 {
            return false;
        }
        if self.max_to_store < 0 {
            return true;
        }
        already_stored < self.max_to_store
    }

    /// Java private `gt(MessageRating)`.  A null message is treated as uninitialized
    /// and having the lowest rating.
    fn gt(self, message_rating: Option<MessageRating>) -> bool {
        match message_rating {
            None => true,
            Some(message_rating) => self.rating > message_rating.rating,
        }
    }

    /// Java private `lt(MessageRating)`.
    fn lt(self, message_rating: Option<MessageRating>) -> bool {
        match message_rating {
            None => false,
            Some(message_rating) => self.rating < message_rating.rating,
        }
    }
}

/// Java `toString()`.
impl std::fmt::Display for MessageRating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if *self == MessageRating::DROP {
            "DROP"
        } else if *self == MessageRating::ALTERNATIVE_B {
            "ALTERNATIVE_B"
        } else if *self == MessageRating::ALTERNATIVE_A {
            "ALTERNATIVE_A"
        } else if *self == MessageRating::KEEPER {
            "KEEPER"
        } else {
            "unknown"
        })
    }
}

/// Java private static final nested `RatedList`.  Saves the most highly rated messages
/// it gets.  Messages with a lower rating are not kept.  Unrated messages are treated
/// as the most highly rated.
#[derive(Clone, Debug)]
struct RatedList {
    /// Java private final `list`.
    list: Vec<String>,
    /// Java private `messageRating`.
    message_rating: Option<MessageRating>,
    /// Java private `mostRecentHeader`.  Hold onto the recent header in case messages
    /// need to be stored under it.
    most_recent_header: Option<String>,
    /// Java private `headerLines`.
    header_lines: i32,
}

impl RatedList {
    /// Java private `RatedList()`.
    fn new() -> RatedList {
        RatedList {
            list: Vec::new(),
            message_rating: None,
            most_recent_header: None,
            header_lines: 0,
        }
    }

    /// Java private `add(String, String, MessageRating)`.
    fn add_rated(
        &mut self,
        header: Option<&str>,
        message: &str,
        new_message_rating: Option<MessageRating>,
    ) -> bool {
        let mut add_header = false;
        if let Some(header) = header
            && self.most_recent_header.as_deref() != Some(header)
        {
            self.most_recent_header = Some(header.to_owned());
            // This causes the header to be added to the list if the message is added.
            add_header = true;
        }
        // Assume an unrated message is a keeper.
        let new_message_rating = new_message_rating.unwrap_or(MessageRating::MAX);
        if new_message_rating.is_drop() {
            return false;
        }
        // Don't save anything with a rating lower then the current rating.
        if new_message_rating.lt(self.message_rating) {
            return false;
        }
        if new_message_rating.gt(self.message_rating) {
            // If the new message rating is higher then the current rating, then dump
            // all the preceding (lower rated) messages.
            self.message_rating = Some(new_message_rating);
            self.list.clear();
            // Make sure the most recent header is add if a message is saved
            add_header = true;
            self.header_lines = 0;
        }
        // Add the header and message.
        if new_message_rating.is_store(self.list.len() as i32 - self.header_lines) {
            // When there is a new header or the list was just cleared add the header
            // before the message.
            if add_header && let Some(most_recent_header) = &self.most_recent_header {
                self.list.push(most_recent_header.clone());
                self.list.push(String::new());
                self.header_lines += 2;
            }
            self.list.push(message.to_owned());
            return true;
        }
        false
    }

    /// Java private `add(String)`.  Add a message without a rating.  This bumps the
    /// rating up to the highest level.
    fn add(&mut self, message: &str) {
        if self.message_rating != Some(MessageRating::MAX) {
            self.message_rating = Some(MessageRating::MAX);
            self.list.clear();
        }
        self.list.push(message.to_owned());
    }

    /// Java private `contains(Object)`.
    fn contains(&self, object: &str) -> bool {
        self.list.iter().any(|element| element == object)
    }

    /// Java private `clear()`.
    fn clear(&mut self) {
        self.list.clear();
    }

    /// Java private `isEmpty()`.
    fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Java private `size()`.
    fn size(&self) -> usize {
        self.list.len()
    }

    /// Java private `get(int)`.
    fn get(&self, index: usize) -> &str {
        &self.list[index]
    }

    /// Java private `getList()`.
    fn get_list(&self) -> &[String] {
        &self.list
    }

    /// Java private `iterator()`.
    fn iterator(&self) -> std::slice::Iter<'_, String> {
        self.list.iter()
    }
}

/// Java `toString()`: `"[messageRating=" + messageRating + ",list=" + list + "]"`.
impl std::fmt::Display for RatedList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[messageRating={},list=[{}]]",
            self.message_rating
                .map_or("null".to_owned(), |message_rating| message_rating
                    .to_string()),
            self.list.join(", ")
        )
    }
}

/// Java private static final nested `MultiSourceStringIterator`.  Iterator for one of
/// the three sources it can store: String, List<String>, or String[].  Chooses one
/// source to iterate.  Checks in the order: String, Iterator<String>, String[].
#[derive(Clone, Debug)]
struct MultiSourceStringIterator {
    /// Java private `fromMessage`.
    from_message: Option<String>,
    /// Java private `singleMessage`.
    single_message: bool,
    /// Java private `fromIterator`: the list and the position in it.
    from_iterator: Option<(Vec<String>, usize)>,
    /// Java private `fromArray`.
    from_array: Option<Vec<String>>,
    /// Java private `fromArrayIndex`.
    from_array_index: i32,
}

impl MultiSourceStringIterator {
    /// Java private `MultiSourceStringIterator()`.
    fn new() -> MultiSourceStringIterator {
        MultiSourceStringIterator {
            from_message: None,
            single_message: false,
            from_iterator: None,
            from_array: None,
            from_array_index: -1,
        }
    }

    /// Java private `reset(String, List<String>, String[])`.
    ///
    /// Upstream bug fixed in translation (ProcessMessages.java:2483): the source sets
    /// `fromArrayIndex = -1`, and `hasNext` requires `fromArrayIndex >= 0`, so an array
    /// source never yields anything - every `add(MessageType, String, String[])` (the
    /// "Standard error output:" of a failed process) is silently dropped.  The index
    /// starts at 0 here (BUGS.md).
    fn reset(
        &mut self,
        from_message: Option<&str>,
        from_list: Option<Vec<String>>,
        from_array: Option<Vec<String>>,
    ) {
        self.from_message = from_message.map(str::to_owned);
        self.single_message = from_message.is_some();
        self.from_iterator = from_list.map(|from_list| (from_list, 0));
        self.from_array = from_array;
        self.from_array_index = 0;
    }

    /// Java private `hasNext()`.  True if selected source has any thing left to
    /// return.
    fn has_next(&self) -> bool {
        if self.single_message {
            return self.from_message.is_some();
        }
        if let Some((list, position)) = &self.from_iterator {
            return *position < list.len();
        }
        if let Some(from_array) = &self.from_array {
            return self.from_array_index >= 0
                && (self.from_array_index as usize) < from_array.len();
        }
        false
    }

    /// Java private `next()`.  Returns selected source's next string.  For a single
    /// message, this deletes the message.
    fn next(&mut self) -> Option<String> {
        if self.single_message {
            return self.from_message.take();
        }
        if let Some((list, position)) = &mut self.from_iterator {
            let next = list.get(*position).cloned();
            *position += 1;
            return next;
        }
        if self.from_array.is_some() && self.has_next() {
            let next = self.from_array.as_ref().unwrap()[self.from_array_index as usize].clone();
            self.from_array_index += 1;
            return Some(next);
        }
        None
    }
}

/// Java public final inner `ListTypeIterator`.  Iterates over the list types whose
/// lists are not empty: errors, chunk errors, warnings, chunk warnings, info.  `FLAG`
/// marks the end.
pub struct ListTypeIterator<'a> {
    process_messages: &'a ProcessMessages,
    /// Java package-private `curListType`.
    cur_list_type: Option<ListType>,
}

impl ListTypeIterator<'_> {
    /// Java `hasNext()`.
    pub fn has_next(&self) -> bool {
        self.get_next(self.cur_list_type) != ListType::Flag
    }

    /// Java `next()`.
    pub fn next(&mut self) -> ListType {
        let next = self.get_next(self.cur_list_type);
        self.cur_list_type = Some(next);
        next
    }

    /// Java private `getNext(ListType)`.
    fn get_next(&self, mut list_type: Option<ListType>) -> ListType {
        loop {
            let next = ListTypeIterator::increment(list_type);
            if next == ListType::Flag {
                return ListType::Flag;
            }
            list_type = Some(next);
            if let Some(list) = self.process_messages.get_list(next)
                && !list.is_empty()
            {
                return next;
            }
        }
    }

    /// Java private `increment(ListType)`.
    fn increment(list_type: Option<ListType>) -> ListType {
        match list_type {
            None => ListType::Error,
            Some(ListType::Error) => ListType::ChunkError,
            Some(ListType::ChunkError) => ListType::Warning,
            Some(ListType::Warning) => ListType::ChunkWarning,
            Some(ListType::ChunkWarning) => ListType::Info,
            Some(_) => ListType::Flag,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_sub_line_and_secondary_log_tag() {
        let harness = ProcessMessagesTestHarness::new();
        assert_eq!(harness.test_sub_line(), None);
        assert_eq!(harness.test_secondary_log_tag(), None);
    }

    #[test]
    fn multi_line_error_ends_on_blank_line() {
        let mut messages = ProcessMessages::get_multi_line_instance(None, AxisID::Only);
        messages.add_process_output_lines(
            None,
            &[
                "ERROR: first".to_owned(),
                "second".to_owned(),
                String::new(),
                "WARNING: careful".to_owned(),
            ],
        );
        messages.end_parse();
        assert_eq!(messages.size(MessageType::Error), 2);
        assert_eq!(messages.get(MessageType::Error, 0), Some("ERROR: first\n"));
        assert_eq!(messages.get(MessageType::Error, 1), Some("second\n"));
        assert_eq!(
            messages.get(MessageType::Warning, 0),
            Some("WARNING: careful\n")
        );
    }

    #[test]
    fn string_feed_parses_on_its_thread() {
        let messages: MessagesRef = Arc::new(Mutex::new(ProcessMessages::get_instance(
            None,
            AxisID::Only,
        )));
        ProcessMessages::start_string_feed(&messages);
        messages
            .lock()
            .unwrap()
            .feed_string("WARNING: queued warning");
        ProcessMessages::stop_string_feed(&messages);
        let messages = messages.lock().unwrap();
        assert!(!messages.is_string_feed());
        assert_eq!(
            messages.get(MessageType::Warning, 0),
            Some("WARNING: queued warning")
        );
    }

    #[test]
    fn parses_upstream_test_errors_log_with_traceback_multiline() {
        let mut messages = ProcessMessages::get_instance(None, AxisID::Only);
        messages
            .add_process_output_file(Path::new("IMOD/Etomo/unitTestData/testErrors.log"))
            .unwrap();
        messages.end_parse();
        assert_eq!(messages.size(MessageType::Error), 4);
        assert_eq!(
            messages.get(MessageType::Error, 0),
            Some("ERROR: A. error line")
        );
        assert_eq!(
            messages.get(MessageType::Error, 1),
            Some("Errno C. error line")
        );
        assert_eq!(
            messages.get(MessageType::Error, 2),
            Some("Traceback E. error line\n")
        );
        assert_eq!(
            messages.get(MessageType::Error, 3),
            Some("F. second error line\n")
        );
    }

    #[test]
    fn enclosed_chunk_errors_keep_contents() {
        let mut messages =
            ProcessMessages::get_instance_for_parallel_processing(None, AxisID::Only, false);
        messages.add_process_output_lines(
            None,
            &[
                "CHUNK ERROR: one".to_owned(),
                "ERROR: associated".to_owned(),
                "END CHUNK ERROR trailing".to_owned(),
            ],
        );
        assert_eq!(messages.size(MessageType::ChunkError), 2);
        assert_eq!(
            messages.get(MessageType::ChunkError, 0),
            Some("CHUNK ERROR: one\n")
        );
        assert_eq!(
            messages.get(MessageType::ChunkError, 1),
            Some("ERROR: associated\n")
        );
    }

    #[test]
    fn pip_warning_has_precedence_and_flag_requires_order() {
        let mut messages = ProcessMessages::get_instance_success_tags(
            None,
            AxisID::Only,
            Some("first"),
            Some("second"),
        );
        messages.add_process_output_lines(
            None,
            &[
                "PIP WARNING: bad ERROR: hidden".to_owned(),
                "continuation".to_owned(),
                "Using fallback options in main program".to_owned(),
                "second then first".to_owned(),
                "first then second".to_owned(),
            ],
        );
        assert_eq!(messages.size(MessageType::Info), 3);
        assert!(
            messages
                .get(MessageType::Info, 0)
                .unwrap()
                .contains("ERROR: hidden")
        );
        assert_eq!(messages.size(MessageType::Error), 0);
        assert!(messages.is_success());
    }

    #[test]
    fn prepend_line_is_consumed_by_next_warning() {
        let mut messages = ProcessMessages::get_instance(None, AxisID::Only);
        messages.set_message_prepend_tag(Some("context:"));
        messages.add_process_output_lines(
            None,
            &[
                "context: section 4".to_owned(),
                "WARNING: missing input".to_owned(),
            ],
        );
        assert_eq!(
            messages.get(MessageType::Warning, 0),
            Some("context: section 4\nWARNING: missing input")
        );
    }

    #[test]
    fn standard_error_array_is_added_under_its_header() {
        let mut messages = ProcessMessages::get_instance(None, AxisID::Only);
        messages.add_array(
            MessageType::Error,
            Some("Standard error output:"),
            Some(&["bad thing".to_owned()]),
        );
        assert_eq!(messages.size(MessageType::Error), 3);
        assert_eq!(
            messages.get(MessageType::Error, 0),
            Some("Standard error output:")
        );
        assert_eq!(messages.get(MessageType::Error, 2), Some("bad thing"));
    }
}
