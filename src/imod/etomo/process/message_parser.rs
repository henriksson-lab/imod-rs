//! `IMOD/Etomo/src/etomo/process/MessageParser.java` and its direct tag-parser closure
//! (`Tag`, `EolTag`, `FlagTag`, `EnclosedTag`, `PrependTag`, `TagInterface`).
#![allow(dead_code)]

use std::collections::VecDeque;

use super::message::Message;
use super::process_messages::{END_FEED_TOKEN, ListType, MessageType, ProcessMessages};

#[derive(Clone, Debug)]
enum TagKind {
    Normal,
    Enclosed { end: String, strip_end: bool },
    Eol,
    Flag { second: Option<String> },
    Prepend,
}
#[derive(Clone, Debug)]
struct Tag {
    ty: MessageType,
    text: String,
    multi_line: bool,
    strip: bool,
    list: Option<ListType>,
    takes_prepend: bool,
    antitags: Vec<String>,
    kind: TagKind,
    line: Option<String>,
    start: Option<usize>,
    end: Option<usize>,
    open: bool,
    closed: bool,
    prepend: Option<String>,
}
impl Tag {
    fn normal(
        ty: MessageType,
        text: &str,
        multi_line: bool,
        strip: bool,
        list: Option<ListType>,
        takes_prepend: bool,
    ) -> Self {
        Self {
            ty,
            text: text.into(),
            multi_line,
            strip,
            list,
            takes_prepend,
            antitags: vec![],
            kind: TagKind::Normal,
            line: None,
            start: None,
            end: None,
            open: false,
            closed: false,
            prepend: None,
        }
    }
    fn parse(&mut self, line: &str) -> bool {
        self.line = Some(line.to_owned());
        self.start = None;
        self.end = None;
        self.open = false;
        self.closed = false;
        if line.is_empty()
            || self.text.is_empty()
            || self.antitags.iter().any(|tag| line.contains(tag))
        {
            return false;
        }
        match &self.kind {
            TagKind::Eol => {
                if let Some(end) = line.find(&self.text) {
                    self.end = Some(end);
                    self.open = true;
                    self.closed = !self.multi_line;
                    return true;
                }
            }
            TagKind::Flag { second } => {
                if let Some(pos) = line.find(&self.text) {
                    if second
                        .as_ref()
                        .is_none_or(|tag| line[pos + self.text.len()..].contains(tag))
                    {
                        self.open = true;
                        self.closed = true;
                        return true;
                    }
                }
            }
            TagKind::Enclosed { end, strip_end } => {
                let start = line.find(&self.text);
                let from = start.map_or(0, |n| n + self.text.len());
                let close = line[from..].find(end).map(|n| n + from);
                if let Some(start) = start {
                    self.start = Some(if self.strip {
                        start + self.text.len()
                    } else {
                        start
                    });
                    self.open = true;
                }
                if let Some(close) = close {
                    self.end = Some(if *strip_end { close } else { close + end.len() });
                    self.closed = true;
                }
                return self.open || self.closed;
            }
            TagKind::Prepend | TagKind::Normal => {
                if let Some(start) = line.find(&self.text) {
                    self.start = Some(if self.strip {
                        start + self.text.len()
                    } else {
                        start
                    });
                    self.open = true;
                    self.closed = !self.multi_line;
                    return true;
                }
            }
        }
        false
    }
    fn message(&mut self) -> Option<String> {
        let line = self.line.as_deref()?;
        let value = match (self.start, self.end) {
            (Some(start), Some(end)) => &line[start..end],
            (Some(start), None) => &line[start..],
            (None, Some(end)) => &line[..end],
            _ => return None,
        }
        .trim();
        let mut value = value.to_owned();
        if self.open && self.takes_prepend {
            if let Some(prepend) = self.prepend.take() {
                if !prepend.is_empty() {
                    value = format!("{prepend}\n{value}");
                }
            }
        }
        Some(value)
    }
    fn is_enclosed(&self) -> bool {
        matches!(self.kind, TagKind::Enclosed { .. })
    }
    fn chunk(&self) -> bool {
        self.list.is_some_and(ListType::is_chunk)
    }
    fn delete_message_string(&mut self) {
        self.line = None;
        self.start = None;
        self.end = None;
        self.open = false;
        self.closed = false;
    }
}

/// Java package-private `MessageParser`.
pub struct MessageParser {
    tags: Vec<Tag>,
    multiline: VecDeque<Message>,
    multiline_tag: Option<usize>,
    chunk_message: bool,
    finished: bool,
    debug: bool,
}
impl MessageParser {
    pub fn get_instance(
        process_messages: &ProcessMessages,
        error_tag: Option<&str>,
        error_tag_always_multiline: bool,
        debug: bool,
    ) -> Self {
        let mut parser = Self {
            tags: vec![],
            multiline: VecDeque::new(),
            multiline_tag: None,
            chunk_message: false,
            finished: true,
            debug,
        };
        parser.init(process_messages, error_tag, error_tag_always_multiline);
        parser
    }
    fn init(
        &mut self,
        pm: &ProcessMessages,
        error_tag: Option<&str>,
        error_tag_always_multiline: bool,
    ) {
        let logged = pm.is_log_all_messages();
        let multiline_all = pm.is_multi_line_all_messages();
        let mut pip = Tag::normal(
            MessageType::PipWarningStart,
            "PIP WARNING:",
            true,
            false,
            Some(ListType::Info),
            true,
        );
        pip.kind = TagKind::Enclosed {
            end: "Using fallback options in main program".into(),
            strip_end: false,
        };
        self.tags.push(pip);
        self.tags.push(Tag::normal(
            MessageType::LogFile,
            "LOGFILE:",
            false,
            true,
            Some(ListType::Logged),
            false,
        ));
        self.tags.push(Tag::normal(
            MessageType::Warning,
            "WARNING:",
            multiline_all || pm.is_multi_line_warning(),
            false,
            if logged {
                Some(ListType::Logged)
            } else {
                Some(ListType::Warning)
            },
            true,
        ));
        if pm.is_chunks() {
            let mut tag = Tag::normal(
                MessageType::ChunkError,
                "CHUNK ERROR:",
                true,
                false,
                Some(ListType::ChunkError),
                true,
            );
            tag.kind = TagKind::Enclosed {
                end: "END CHUNK ERROR".into(),
                strip_end: true,
            };
            self.tags.push(tag);
        }
        self.tags.push(Tag::normal(
            MessageType::Info,
            "INFO:",
            multiline_all || pm.is_multi_line_info(),
            false,
            if pm.is_log_info_messages() || logged {
                Some(ListType::Logged)
            } else {
                Some(ListType::Info)
            },
            false,
        ));
        self.tags.push(Tag::normal(
            MessageType::Log,
            "LOG:",
            pm.is_allow_multi_line_log(),
            true,
            Some(ListType::Logged),
            false,
        ));
        let mut eol = Tag::normal(
            MessageType::Log,
            "[:LOG]",
            pm.is_allow_multi_line_log(),
            true,
            Some(ListType::Logged),
            false,
        );
        eol.kind = TagKind::Eol;
        self.tags.push(eol);
        let antitags = vec!["prnstr('ERROR:".into(), "log.write('ERROR:".into()];
        if let Some(override_tag) = pm.get_error_override_log_tag() {
            let mut tag = Tag::normal(
                MessageType::Error,
                override_tag,
                multiline_all && error_tag_always_multiline,
                false,
                Some(ListType::Error),
                true,
            );
            tag.antitags = antitags.clone();
            self.tags.push(tag);
        }
        let mut basic = Tag::normal(
            MessageType::Error,
            "ERROR:",
            multiline_all,
            false,
            if logged {
                Some(ListType::Logged)
            } else {
                Some(ListType::Error)
            },
            true,
        );
        basic.antitags = antitags.clone();
        self.tags.push(basic);
        if let Some(tag) = error_tag {
            self.tags.push(Tag::normal(
                MessageType::Error,
                tag,
                multiline_all || error_tag_always_multiline,
                false,
                if logged {
                    Some(ListType::Logged)
                } else {
                    Some(ListType::Error)
                },
                true,
            ));
        }
        let mut errno = Tag::normal(
            MessageType::Error,
            "Errno",
            multiline_all,
            false,
            if logged {
                Some(ListType::Logged)
            } else {
                Some(ListType::Error)
            },
            true,
        );
        errno.antitags = antitags.clone();
        self.tags.push(errno);
        let mut traceback = Tag::normal(
            MessageType::Error,
            "Traceback",
            true,
            false,
            if logged {
                Some(ListType::Logged)
            } else {
                Some(ListType::Error)
            },
            true,
        );
        traceback.antitags = antitags;
        self.tags.push(traceback);
        if let Some(tag1) = pm.get_success_tag1() {
            let mut flag = Tag::normal(
                MessageType::Success,
                tag1,
                false,
                false,
                Some(ListType::Flag),
                false,
            );
            flag.kind = TagKind::Flag {
                second: pm.get_success_tag2().map(str::to_owned),
            };
            self.tags.push(flag);
        }
    }
    pub(crate) fn set_multi_parse(&mut self, multi_parse: bool) {
        self.finished = !multi_parse;
    }
    pub(crate) fn end_parse(&mut self, pm: &mut ProcessMessages) {
        self.finished = true;
        self.parse(pm, None);
    }
    pub fn set_prepend(&mut self, prepend: Option<&str>) {
        self.tags
            .retain(|tag| !matches!(tag.kind, TagKind::Prepend));
        if let Some(text) = prepend {
            let mut tag = Tag::normal(MessageType::Prepend, text, false, false, None, false);
            tag.kind = TagKind::Prepend;
            self.tags.push(tag);
        }
    }
    fn find_tag(&mut self, line: &str) -> Option<usize> {
        for (index, tag) in self.tags.iter_mut().enumerate() {
            if tag.parse(line) {
                if matches!(tag.kind, TagKind::Prepend) {
                    let value = line.to_owned();
                    for receiver in &mut self.tags {
                        if receiver.takes_prepend {
                            receiver.prepend = Some(value.clone());
                        }
                    }
                    return None;
                }
                return Some(index);
            }
        }
        None
    }
    pub(crate) fn parse(&mut self, pm: &mut ProcessMessages, mut header: Option<&str>) {
        while let Some(line) = pm.get_next_line() {
            if line == END_FEED_TOKEN {
                continue;
            }
            let mut possible = self.find_tag(&line);
            let mut tag = None;
            if let Some(open_index) = self.multiline_tag {
                let open = &self.tags[open_index];
                if !open.is_enclosed() {
                    if possible.is_some_and(|index| {
                        open.ty == MessageType::Error && self.tags[index].ty == MessageType::Error
                    }) {
                        tag = possible.take();
                    }
                    if line.is_empty() || tag.is_some() {
                        self.store_queue(pm, header);
                        header = None;
                        if self.tags[open_index].chunk() {
                            self.chunk_message = false;
                        }
                        self.multiline_tag = None;
                        if tag.is_none() {
                            continue;
                        }
                    }
                } else if self.tags[open_index].parse(&line) && self.tags[open_index].closed {
                    let source = &mut self.tags[open_index];
                    let mut message =
                        Message::new(source.ty, source.list, true, source.chunk(), true);
                    message.append(source.message().as_deref());
                    source.delete_message_string();
                    self.multiline.push_back(message);
                    self.store_queue(pm, header);
                    header = None;
                    self.chunk_message = false;
                    self.multiline_tag = None;
                    continue;
                }
                if tag.is_none() {
                    let source = &self.tags[open_index];
                    let mut message = Message::new(
                        source.ty,
                        source.list,
                        source.is_enclosed(),
                        source.chunk(),
                        source.multi_line,
                    );
                    message.append(Some(&line));
                    self.multiline.push_back(message);
                    continue;
                }
            }
            let tag = tag.or(possible);
            if let Some(index) = tag {
                let source = &mut self.tags[index];
                if source.multi_line && source.open && !source.closed {
                    let mut message = Message::new(
                        source.ty,
                        source.list,
                        source.is_enclosed(),
                        source.chunk(),
                        source.multi_line,
                    );
                    message.append(source.message().as_deref());
                    source.delete_message_string();
                    if message.is_chunk() {
                        self.chunk_message = true;
                    }
                    self.multiline.push_back(message);
                    self.multiline_tag = Some(index);
                } else if !source.multi_line || (source.open && source.closed) {
                    let ty = source.ty;
                    let list = source.list;
                    let text = if source.ty == MessageType::Success {
                        Some(String::new())
                    } else {
                        source.message()
                    };
                    source.delete_message_string();
                    pm.store_tag_message(ty, list, header, text, self.chunk_message);
                    header = None;
                } else {
                    source.delete_message_string();
                }
            }
        }
        if !pm.is_string_feed() && self.finished && !self.multiline.is_empty() {
            self.store_queue(pm, header);
            self.multiline_tag = None;
            self.chunk_message = false;
        }
    }
    fn store_queue(&mut self, pm: &mut ProcessMessages, header: Option<&str>) {
        while let Some(mut message) = self.multiline.pop_front() {
            pm.store_message(header, &mut message, self.chunk_message);
        }
    }
}
