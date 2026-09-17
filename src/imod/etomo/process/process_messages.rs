//! `IMOD/Etomo/src/etomo/process/ProcessMessages.java`.
//!
//! This keeps the new Java parser's precedence, multiline lifetime, message ratings,
//! chunk duplicate suppression, and string-feed semantics.  UI display is deliberately
//! not performed here: `AbstractFrame` consumes these typed lists at its direct boundary.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use super::message::Message;
use super::message_parser::MessageParser;

pub const END_FEED_TOKEN: &str =
    "This the END of the String Feed!!!  239asdlkjgsafT$LSFJsGHW($(gjhaehgpasjdhf0w235";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ListType {
    ChunkError,
    Error,
    Info,
    Warning,
    Logged,
    Flag,
    ChunkWarning,
}
impl ListType {
    pub fn is_chunk(self) -> bool {
        matches!(self, Self::ChunkError | Self::ChunkWarning)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MessageType {
    ChunkError,
    Error,
    Info,
    Warning,
    Log,
    LogFile,
    PipWarningStart,
    PipWarningEnd,
    Success,
    Prepend,
    ChunkWarning,
}
impl MessageType {
    pub fn tag(self) -> Option<&'static str> {
        match self {
            Self::ChunkError => Some("CHUNK ERROR:"),
            Self::Error => Some("ERROR:"),
            Self::Info => Some("INFO:"),
            Self::Warning => Some("WARNING:"),
            Self::Log => Some("LOG:"),
            Self::LogFile => Some("LOGFILE:"),
            Self::PipWarningStart => Some("PIP WARNING:"),
            Self::PipWarningEnd => Some("Using fallback options in main program"),
            _ => None,
        }
    }
    pub fn secondary_tag(self) -> Option<&'static str> {
        if self == Self::Log {
            Some("[:LOG]")
        } else {
            None
        }
    }
    pub fn list_type(self) -> Option<ListType> {
        match self {
            Self::ChunkError => Some(ListType::ChunkError),
            Self::Error => Some(ListType::Error),
            Self::Info | Self::PipWarningStart | Self::PipWarningEnd => Some(ListType::Info),
            Self::Warning => Some(ListType::Warning),
            Self::Log | Self::LogFile => Some(ListType::Logged),
            Self::Success => Some(ListType::Flag),
            Self::ChunkWarning => Some(ListType::ChunkWarning),
            Self::Prepend => None,
        }
    }
    pub fn exclusive(self) -> bool {
        matches!(
            self,
            Self::PipWarningStart | Self::PipWarningEnd | Self::Success | Self::Prepend
        )
    }
    pub fn is_type(self, string: Option<&str>) -> bool {
        string.is_some_and(|s| {
            self.tag().is_some_and(|t| s.contains(t))
                || self.secondary_tag().is_some_and(|t| s.contains(t))
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct RatedList {
    list: Vec<String>,
    rating: Option<MessageRating>,
    most_recent_header: Option<String>,
    header_lines: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum MessageRating {
    Drop,
    AlternativeB,
    AlternativeA,
    Keeper,
}
impl MessageRating {
    fn rate(message: &str) -> Self {
        if message.trim().is_empty() || (message.contains("PID:") && message.len() <= 20) {
            Self::Drop
        } else if message.contains("exited with status 1") {
            if message.contains("python -u") {
                Self::AlternativeB
            } else {
                Self::AlternativeA
            }
        } else {
            Self::Keeper
        }
    }
    fn max_to_store(self) -> isize {
        match self {
            Self::Drop => 0,
            Self::AlternativeB | Self::AlternativeA => 1,
            Self::Keeper => -1,
        }
    }
}
impl RatedList {
    fn add_rated(&mut self, header: Option<&str>, message: String, rating: MessageRating) -> bool {
        let mut add_header = false;
        if header.map(str::to_owned) != self.most_recent_header {
            self.most_recent_header = header.map(str::to_owned);
            add_header = header.is_some();
        }
        if rating == MessageRating::Drop || self.rating.is_some_and(|old| rating < old) {
            return false;
        }
        if self.rating.is_none_or(|old| rating > old) {
            self.rating = Some(rating);
            self.list.clear();
            self.header_lines = 0;
            add_header = true;
        }
        let stored = self.list.len().saturating_sub(self.header_lines) as isize;
        if rating.max_to_store() >= 0 && stored >= rating.max_to_store() {
            return false;
        }
        if add_header {
            if let Some(header) = &self.most_recent_header {
                self.list.push(header.clone());
                self.list.push(String::new());
                self.header_lines += 2;
            }
        }
        self.list.push(message);
        true
    }
    fn add(&mut self, message: String) {
        if self.rating != Some(MessageRating::Keeper) {
            self.rating = Some(MessageRating::Keeper);
            self.list.clear();
        }
        self.list.push(message);
    }
    pub fn get(&self, index: usize) -> Option<&str> {
        self.list.get(index).map(String::as_str)
    }
    pub fn size(&self) -> usize {
        self.list.len()
    }
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
    pub fn clear(&mut self) {
        self.list.clear();
    }
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.list.iter()
    }
}

/// Java `ProcessMessages`.  Manager logging is represented as `logged_messages` until
/// `BaseManager.logSimpleMessage` is directly wired; no output is discarded.
pub struct ProcessMessages {
    chunks: bool,
    log_all_messages: bool,
    error_override_log_tag: Option<String>,
    log_info_messages: bool,
    multi_line_all_messages: bool,
    multi_line_warning: bool,
    multi_line_info: bool,
    allow_multi_line_log: bool,
    success_tag1: Option<String>,
    success_tag2: Option<String>,
    success: bool,
    hibernate: bool,
    debug: bool,
    info_list: Option<RatedList>,
    warning_list: Option<RatedList>,
    error_list: Option<RatedList>,
    chunk_error_list: Option<RatedList>,
    chunk_warning_list: Option<RatedList>,
    logged_messages: Vec<String>,
    /// Optional secondary project-log stream.  The GUI/application layer owns
    /// persistence; keeping this separate mirrors Java's `FileWriter` routing.
    secondary_log: Option<Vec<String>>,
    input: VecDeque<String>,
    /// Java used a worker thread and blocking queue here.  The Rust port keeps the
    /// queue explicitly and drains it at `stop_string_feed`; this preserves ordered
    /// delivery without making this model object depend on a UI/runtime executor.
    string_feed: bool,
    parser: Option<MessageParser>,
}

/// Java inner `ListTypeIterator`.  Its order is deliberately the source order rather
/// than enum declaration order: errors, chunk errors, warnings, chunk warnings, info.
pub struct ListTypeIterator<'a> {
    process_messages: &'a ProcessMessages,
    current: usize,
}
impl<'a> ListTypeIterator<'a> {
    pub fn has_next(&self) -> bool {
        [
            ListType::Error,
            ListType::ChunkError,
            ListType::Warning,
            ListType::ChunkWarning,
            ListType::Info,
        ][self.current..]
            .iter()
            .any(|list_type| {
                !self
                    .process_messages
                    .list(Some(*list_type))
                    .is_none_or(RatedList::is_empty)
            })
    }
    pub fn next(&mut self) -> Option<ListType> {
        let values = [
            ListType::Error,
            ListType::ChunkError,
            ListType::Warning,
            ListType::ChunkWarning,
            ListType::Info,
        ];
        while self.current < values.len() {
            let value = values[self.current];
            self.current += 1;
            if !self
                .process_messages
                .list(Some(value))
                .is_none_or(RatedList::is_empty)
            {
                return Some(value);
            }
        }
        None
    }
}
impl ProcessMessages {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        multi_line_messages: bool,
        chunks: bool,
        success_tag1: Option<String>,
        success_tag2: Option<String>,
        multi_line_warning: bool,
        multi_line_info: bool,
        log_all_messages: bool,
        error_override_log_tag: Option<String>,
        error_tag: Option<String>,
        error_tag_always_multiline: bool,
        allow_multi_line_log: bool,
        log_info_messages: bool,
        debug: bool,
    ) -> Self {
        let mut result = Self {
            chunks,
            log_all_messages,
            error_override_log_tag,
            log_info_messages,
            multi_line_all_messages: multi_line_messages,
            multi_line_warning,
            multi_line_info,
            allow_multi_line_log,
            success_tag1,
            success_tag2,
            success: false,
            hibernate: false,
            debug,
            info_list: None,
            warning_list: None,
            error_list: None,
            chunk_error_list: None,
            chunk_warning_list: None,
            logged_messages: vec![],
            secondary_log: None,
            input: VecDeque::new(),
            string_feed: false,
            parser: None,
        };
        result.parser = Some(MessageParser::get_instance(
            &result,
            error_tag.as_deref(),
            error_tag_always_multiline,
            debug,
        ));
        result
    }
    pub fn get_instance() -> Self {
        Self::new(
            false, false, None, None, false, false, false, None, None, false, false, true, false,
        )
    }
    pub fn get_multi_line_instance() -> Self {
        Self::new(
            true, false, None, None, false, false, false, None, None, false, false, true, false,
        )
    }
    pub fn get_instance_for_parallel_processing(multi_line_messages: bool) -> Self {
        Self::new(
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
    #[allow(clippy::too_many_arguments)]
    pub fn get_logged_instance(
        multi_line_messages: bool,
        log_all_messages: bool,
        error_override_log_tag: Option<String>,
        error_tag: Option<String>,
        always_multiline: bool,
        allow_multi_line_log: bool,
    ) -> Self {
        Self::new(
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
    /// Java `getBatchruntomoTestInstance`: identical setup except information is
    /// deliberately not written to the project log.
    #[allow(clippy::too_many_arguments)]
    pub fn get_batchruntomo_test_instance(
        multi_line_messages: bool,
        log_all_messages: bool,
        error_override_log_tag: Option<String>,
        error_tag: Option<String>,
        always_multiline: bool,
        allow_multi_line_log: bool,
    ) -> Self {
        Self::new(
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
    pub fn is_log_all_messages(&self) -> bool {
        self.log_all_messages
    }
    pub fn is_log_info_messages(&self) -> bool {
        self.log_info_messages
    }
    pub fn is_multi_line_all_messages(&self) -> bool {
        self.multi_line_all_messages
    }
    pub fn is_multi_line_warning(&self) -> bool {
        self.multi_line_warning
    }
    pub fn is_multi_line_info(&self) -> bool {
        self.multi_line_info
    }
    pub fn is_allow_multi_line_log(&self) -> bool {
        self.allow_multi_line_log
    }
    pub fn get_error_override_log_tag(&self) -> Option<&str> {
        self.error_override_log_tag.as_deref()
    }
    pub fn get_success_tag1(&self) -> Option<&str> {
        self.success_tag1.as_deref()
    }
    pub fn get_success_tag2(&self) -> Option<&str> {
        self.success_tag2.as_deref()
    }
    pub fn is_chunks(&self) -> bool {
        self.chunks
    }
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    pub fn hibernate(&mut self) {
        self.hibernate = true;
    }
    /// Java `listTypeIterator`.
    pub fn list_type_iterator(&self) -> ListTypeIterator<'_> {
        ListTypeIterator {
            process_messages: self,
            current: 0,
        }
    }
    pub fn wake(&mut self) {
        self.hibernate = false;
    }
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
    pub fn set_multi_parse(&mut self, multi_parse: bool) {
        if let Some(parser) = &mut self.parser {
            parser.set_multi_parse(multi_parse);
        }
    }
    /// Java `setMessagePrependTag`.  The tag's whole line is attached to the next
    /// error/warning tag that accepts a prepend, exactly at the parser boundary.
    pub fn set_message_prepend_tag(&mut self, tag: Option<&str>) {
        if let Some(parser) = &mut self.parser {
            parser.set_prepend(tag);
        }
    }
    pub fn end_parse(&mut self) {
        self.with_parser(|parser, this| parser.end_parse(this));
    }
    /// Begins a buffered equivalent of Java's asynchronous string feed.
    pub fn start_string_feed(&mut self) {
        if !self.hibernate {
            self.string_feed = true;
        }
    }
    /// Drains all queued feed strings before returning, matching Java's `join`.
    pub fn stop_string_feed(&mut self) {
        if !self.string_feed {
            return;
        }
        self.string_feed = false;
        self.with_parser(|parser, this| parser.parse(this, None));
        self.with_parser(|parser, this| parser.end_parse(this));
    }
    pub fn add_process_output(&mut self, process_output: impl AsRef<str>) {
        if self.hibernate {
            return;
        }
        self.input.push_back(process_output.as_ref().to_owned());
        if self.string_feed {
            return;
        }
        self.with_parser(|parser, this| parser.parse(this, None));
    }
    pub fn add_process_output_lines(
        &mut self,
        header: Option<&str>,
        lines: impl IntoIterator<Item = String>,
    ) {
        if self.hibernate {
            return;
        }
        self.input.extend(lines);
        if self.string_feed {
            return;
        }
        self.with_parser(|parser, this| parser.parse(this, header));
    }
    pub fn add_process_output_file(&mut self, file: &Path) -> std::io::Result<()> {
        let text = fs::read_to_string(file)?;
        self.add_process_output_lines(None, text.lines().map(str::to_owned));
        Ok(())
    }
    fn with_parser(&mut self, function: impl FnOnce(&mut MessageParser, &mut Self)) {
        let mut parser = self
            .parser
            .take()
            .expect("ProcessMessages parser initialized");
        function(&mut parser, self);
        self.parser = Some(parser);
    }
    pub fn get_next_line(&mut self) -> Option<String> {
        self.input.pop_front()
    }
    pub fn feed_string(&mut self, string: impl AsRef<str>) {
        self.add_process_output(string);
    }
    pub fn feed_end_message(&mut self) {
        self.feed_string("");
    }
    pub fn feed_newline(&mut self, ty: Option<MessageType>) {
        self.feed_string(ty.and_then(MessageType::tag).unwrap_or(""));
    }
    pub fn feed_message(&mut self, ty: Option<MessageType>, message: Option<&str>) {
        let line = match (ty, message) {
            (Some(ty), Some(message)) if !message.starts_with(ty.tag().unwrap_or("")) => {
                format!("{} {}", ty.tag().unwrap_or(""), message)
            }
            (Some(ty), None) => ty.tag().unwrap_or("").to_owned(),
            (_, Some(message)) => message.to_owned(),
            _ => String::new(),
        };
        self.feed_string(line);
        self.feed_string("");
    }
    pub fn is_success(&self) -> bool {
        self.success
    }
    pub fn is_string_feed(&self) -> bool {
        self.string_feed
    }
    pub fn size(&self, ty: MessageType) -> usize {
        self.list(ty.list_type()).map_or(0, RatedList::size)
    }
    pub fn get(&self, ty: MessageType, index: usize) -> Option<&str> {
        self.list(ty.list_type()).and_then(|l| l.get(index))
    }
    pub fn get_last(&self, ty: MessageType) -> Option<&str> {
        self.list(ty.list_type())
            .and_then(|l| l.list.last().map(String::as_str))
    }
    pub fn is_empty(&self, ty: MessageType) -> bool {
        self.list(ty.list_type()).is_none_or(RatedList::is_empty)
    }
    pub fn match_messages(&self, ty: MessageType, matches: &[&str]) -> Option<Vec<String>> {
        let values: Vec<_> = self
            .list(ty.list_type())?
            .iter()
            .filter(|m| matches.iter().any(|s| m.contains(s)))
            .cloned()
            .collect();
        (!values.is_empty()).then_some(values)
    }
    pub fn iterator(&self, ty: ListType) -> Option<impl Iterator<Item = &String>> {
        self.list(Some(ty)).map(RatedList::iter)
    }
    pub fn logged_messages(&self) -> &[String] {
        &self.logged_messages
    }
    /// Java `setSecondaryLog`; the app can take these records and persist them.
    pub fn set_secondary_log(&mut self) {
        self.secondary_log = Some(vec![]);
    }
    pub fn get_secondary_log(&self) -> Option<&[String]> {
        self.secondary_log.as_deref()
    }
    pub fn reset_secondary_log(&mut self) {
        if let Some(log) = &mut self.secondary_log {
            log.clear();
        }
    }
    pub fn close_secondary_log(&mut self) {
        self.secondary_log = None;
    }
    pub fn is_secondary_log_open(&self) -> bool {
        self.secondary_log.is_some()
    }
    /// Merges typed lists from another process output, equivalent to Java `add`.
    pub fn add_process_messages(&mut self, source: &ProcessMessages) {
        for ty in [
            ListType::Error,
            ListType::Warning,
            ListType::Info,
            ListType::ChunkError,
            ListType::ChunkWarning,
        ] {
            if let Some(messages) = source.iterator(ty) {
                for message in messages {
                    let ty = match ty {
                        ListType::Error => MessageType::Error,
                        ListType::Warning => MessageType::Warning,
                        ListType::Info => MessageType::Info,
                        ListType::ChunkError => MessageType::ChunkError,
                        ListType::ChunkWarning => MessageType::ChunkWarning,
                        ListType::Logged | ListType::Flag => continue,
                    };
                    self.store_typed_message(
                        ty,
                        None,
                        message.clone(),
                        ty.list_type().is_some_and(ListType::is_chunk),
                    );
                }
            }
        }
    }
    /// Destructively stores queued parser messages, matching Java's queue overload.
    pub(crate) fn store_messages(
        &mut self,
        header: Option<&str>,
        messages: &mut VecDeque<Message>,
        chunk_message: bool,
    ) {
        while let Some(mut message) = messages.pop_front() {
            let text = message.get_message_string();
            self.store(
                message.get_message_type(),
                message.get_list_type(),
                header,
                text,
                chunk_message,
                false,
            );
        }
    }
    pub(crate) fn store_message(
        &mut self,
        header: Option<&str>,
        message: &mut Message,
        chunk_message: bool,
    ) {
        let text = message.get_message_string();
        self.store(
            message.get_message_type(),
            message.get_list_type(),
            header,
            text,
            chunk_message,
            false,
        );
    }
    pub(crate) fn store_tag_message(
        &mut self,
        ty: MessageType,
        list_type: Option<ListType>,
        header: Option<&str>,
        text: Option<String>,
        chunk_message: bool,
    ) {
        if let Some(text) = text {
            self.store(ty, list_type, header, text, chunk_message, false);
        }
    }
    /// Rust replacement for Java's `storeMessageOLD(Message, boolean)` overload.
    /// Parser-internal messages retain their captured tag metadata, so this goes
    /// through exactly the same rating, chunk and logging policy as parsed output.
    pub(crate) fn store_message_old(&mut self, message: &mut Message, chunk_message: bool) {
        self.store_message(None, message, chunk_message);
    }
    /// Stores a typed one-line message at the public parser boundary.
    pub fn store_typed_message(
        &mut self,
        ty: MessageType,
        header: Option<&str>,
        text: impl Into<String>,
        chunk_message: bool,
    ) {
        self.store(
            ty,
            ty.list_type(),
            header,
            text.into(),
            chunk_message,
            false,
        );
    }
    /// Debug representation of the parser model, replacing Java's stderr-only
    /// `dumpState` so Rust callers can route it through their own diagnostics.
    pub fn dump_state(&self) -> String {
        format!(
            "[chunks:{},queued:{},stringFeed:{},info:{},warning:{},error:{},chunkError:{},chunkWarning:{},success:{}]",
            self.chunks,
            self.input.len(),
            self.string_feed,
            self.info_list.as_ref().map_or(0, RatedList::size),
            self.warning_list.as_ref().map_or(0, RatedList::size),
            self.error_list.as_ref().map_or(0, RatedList::size),
            self.chunk_error_list.as_ref().map_or(0, RatedList::size),
            self.chunk_warning_list.as_ref().map_or(0, RatedList::size),
            self.success,
        )
    }
    /// Java `print(MessageType)`, expressed as a diagnostic string rather than
    /// writing to stderr from this state object.  It reports whether anything exists.
    pub fn print(&self, ty: MessageType) -> Option<String> {
        self.list(ty.list_type())
            .filter(|list| !list.is_empty())
            .map(|list| list.iter().cloned().collect::<Vec<_>>().join("\n"))
    }
    pub fn print_list(&self, ty: ListType) -> Option<String> {
        self.list(Some(ty))
            .filter(|list| !list.is_empty())
            .map(|list| list.iter().cloned().collect::<Vec<_>>().join("\n"))
    }
    fn store(
        &mut self,
        ty: MessageType,
        mut list_type: Option<ListType>,
        header: Option<&str>,
        text: String,
        chunk_message: bool,
        allow_log_override: bool,
    ) {
        if self.hibernate {
            return;
        }
        if ty == MessageType::Success {
            self.success = true;
            return;
        }
        if chunk_message {
            list_type = match list_type {
                Some(ListType::Error) => Some(ListType::ChunkError),
                Some(ListType::Warning) => Some(ListType::ChunkWarning),
                x => x,
            };
        }
        let rating = MessageRating::rate(&text);
        if rating == MessageRating::Drop {
            return;
        }
        let should_log = self.log_all_messages || list_type == Some(ListType::Logged);
        let overridden = allow_log_override
            && ty == MessageType::Error
            && self
                .error_override_log_tag
                .as_ref()
                .is_some_and(|tag| text.contains(tag));
        if should_log && !overridden {
            let target = self
                .secondary_log
                .as_mut()
                .unwrap_or(&mut self.logged_messages);
            if let Some(header) = header {
                target.push(header.to_owned());
            }
            if ty == MessageType::LogFile {
                self.log_file_into(&text);
            } else {
                target.push(text);
            }
            return;
        }
        let Some(list_type) = list_type else { return };
        let prevent_duplicates = !self.log_all_messages
            && matches!(ty, MessageType::ChunkError | MessageType::ChunkWarning);
        let list = self.list_mut(list_type);
        if prevent_duplicates && list.list.contains(&text) {
            return;
        }
        let _ = list.add_rated(header, text, rating);
    }
    fn log_file_into(&mut self, name: &str) {
        match fs::read_to_string(name) {
            Ok(text) => self
                .secondary_log
                .as_mut()
                .unwrap_or(&mut self.logged_messages)
                .extend(text.lines().map(str::to_owned)),
            Err(_) if self.debug => self
                .secondary_log
                .as_mut()
                .unwrap_or(&mut self.logged_messages)
                .push(format!("Warning: unable to log from file:{name}")),
            Err(_) => {}
        }
    }
    fn list(&self, ty: Option<ListType>) -> Option<&RatedList> {
        match ty? {
            ListType::Error => self.error_list.as_ref(),
            ListType::Info => self.info_list.as_ref(),
            ListType::Warning => self.warning_list.as_ref(),
            ListType::ChunkError => self.chunk_error_list.as_ref(),
            ListType::ChunkWarning => self.chunk_warning_list.as_ref(),
            _ => None,
        }
    }
    fn list_mut(&mut self, ty: ListType) -> &mut RatedList {
        match ty {
            ListType::Error => self.error_list.get_or_insert_default(),
            ListType::Info => self.info_list.get_or_insert_default(),
            ListType::Warning => self.warning_list.get_or_insert_default(),
            ListType::ChunkError => self.chunk_error_list.get_or_insert_default(),
            ListType::ChunkWarning => self.chunk_warning_list.get_or_insert_default(),
            _ => unreachable!("logged/flag list has no RatedList"),
        }
    }
    pub fn get_error_index(line: &str) -> Option<usize> {
        ["ERROR:", "Errno", "Traceback"]
            .iter()
            .filter_map(|tag| line.find(tag))
            .next()
    }
}
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
                for value in list.iter() {
                    writeln!(f, "{value}")?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_upstream_test_errors_log_with_traceback_multiline() {
        let mut messages = ProcessMessages::get_instance();
        messages
            .add_process_output_file(Path::new("IMOD/Etomo/unitTestData/testErrors.log"))
            .unwrap();
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
        let mut messages = ProcessMessages::get_instance_for_parallel_processing(false);
        messages.add_process_output_lines(
            None,
            [
                "CHUNK ERROR: one".into(),
                "ERROR: associated".into(),
                "END CHUNK ERROR trailing".into(),
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
        let mut messages = ProcessMessages::new(
            false,
            false,
            Some("first".into()),
            Some("second".into()),
            false,
            false,
            false,
            None,
            None,
            false,
            false,
            true,
            false,
        );
        messages.add_process_output_lines(
            None,
            [
                "PIP WARNING: bad ERROR: hidden".into(),
                "continuation".into(),
                "Using fallback options in main program".into(),
                "second then first".into(),
                "first then second".into(),
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
        let mut messages = ProcessMessages::get_instance();
        messages.set_message_prepend_tag(Some("context:"));
        messages.add_process_output_lines(
            None,
            ["context: section 4".into(), "WARNING: missing input".into()],
        );
        assert_eq!(
            messages.get(MessageType::Warning, 0),
            Some("context: section 4\nWARNING: missing input")
        );
    }

    #[test]
    fn string_feed_drains_in_order_when_stopped() {
        let mut messages = ProcessMessages::get_instance();
        messages.start_string_feed();
        messages.feed_string("WARNING: queued warning");
        assert!(messages.is_string_feed());
        assert!(messages.is_empty(MessageType::Warning));
        messages.stop_string_feed();
        assert!(!messages.is_string_feed());
        assert_eq!(
            messages.get(MessageType::Warning, 0),
            Some("WARNING: queued warning")
        );
    }

    #[test]
    fn secondary_log_isolated_from_default_log() {
        let mut messages = ProcessMessages::get_instance();
        messages.set_secondary_log();
        messages.store_typed_message(MessageType::Log, Some("header"), "record", false);
        assert!(messages.logged_messages().is_empty());
        assert_eq!(
            messages.get_secondary_log(),
            Some(["header".to_owned(), "record".to_owned()].as_slice())
        );
        messages.close_secondary_log();
        assert!(!messages.is_secondary_log_open());
    }
}
