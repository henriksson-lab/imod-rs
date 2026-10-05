//! `IMOD/Etomo/src/etomo/process/MessageParser.java`.
//!
//! Parses process output, line by line, for messages and hands the messages found to
//! `ProcessMessages`.  Java's parser holds its `ProcessMessages`; here the parser is a
//! field of the `ProcessMessages`, which lends itself to each call that reads it.

use super::enclosed_tag::EnclosedTag;
use super::eol_tag::EolTag;
use super::flag_tag::FlagTag;
use super::message::Message;
use super::prepend_tag::PrependTag;
use super::process_messages::{END_FEED_TOKEN, ListType, MessageType, ProcessMessages};
use super::tag::Tag;
use super::tag_interface::{TagKind, TagList};
use crate::imod::etomo::util::queue::Queue;

/// Java private static final `STANDARD_ERROR_TAGS`.
const STANDARD_ERROR_TAGS: [&str; 3] = ["ERROR:", "Errno", "Traceback"];

/// Java package-private `final class MessageParser`.
#[derive(Clone, Debug)]
pub(crate) struct MessageParser {
    /// Java private final `tags`.
    tags: TagList,
    /// Java private final `debug`.
    debug: bool,
    /// Java private `multilineMessageQueue`.
    multiline_message_queue: Option<Queue<Message>>,
    /// Java private `multilineTag`.
    multiline_tag: Option<usize>,
    /// Java private `prependTag`.
    prepend_tag: Option<usize>,
    /// Java private `chunkMessage`.  Number of chunk messages on the multi-line
    /// message stack.
    chunk_message: bool,
    /// Java private `finished`.
    finished: bool,
}

impl MessageParser {
    /// Java private `MessageParser(ProcessMessages, boolean)`.
    fn new(debug: bool) -> MessageParser {
        MessageParser {
            tags: TagList::new(),
            debug,
            multiline_message_queue: None,
            multiline_tag: None,
            prepend_tag: None,
            chunk_message: false,
            finished: false,
        }
    }

    /// Java static `getInstance(ProcessMessages, String, boolean, boolean)`.
    pub(crate) fn get_instance(
        process_messages: &ProcessMessages,
        error_tag: Option<&str>,
        error_tag_always_multiline: bool,
        debug: bool,
    ) -> MessageParser {
        let mut instance = MessageParser::new(debug);
        instance.init(process_messages, error_tag, error_tag_always_multiline);
        instance
    }

    /// Java `setMultiParse(boolean)`.  When this is on, the parser will leave
    /// multi-line messages open after it runs out of strings to parse.  This is
    /// essential when parsing string by string, or parsing during a run.  You must call
    /// endParse when the parse is done.
    pub(crate) fn set_multi_parse(&mut self, multi_parse: bool) {
        self.finished = !multi_parse;
    }

    /// Java `endParse()`.  Makes the parser clean up - ending and saving any potential
    /// multi-line messages.
    pub(crate) fn end_parse(&mut self, process_messages: &mut ProcessMessages) {
        self.finished = true;
        self.parse(process_messages, None);
    }

    /// Java private `init(String, boolean)`.  The order in which tags are stored
    /// becomes their precedence.
    fn init(
        &mut self,
        process_messages: &ProcessMessages,
        error_tag: Option<&str>,
        error_tag_always_multiline: bool,
    ) {
        let log_all_messages = process_messages.is_log_all_messages();
        let multi_line_all_messages = process_messages.is_multi_line_all_messages();
        // pip warning
        // Pip warnings block other message; other messages become part of the pip
        // warning.
        let message_type = MessageType::PipWarningStart;
        self.tags.add(TagKind::Enclosed(EnclosedTag::new(
            message_type,
            message_type.get_tag(),
            MessageType::PipWarningEnd.get_tag().unwrap(),
            false,
            message_type.get_list_type(),
            true,
        )));
        // logfile
        // Logfile messages cause the contents of a file to be log (placed in the
        // project log if available).
        let message_type = MessageType::LogFile;
        self.tags.add(TagKind::Tag(Tag::new(
            message_type,
            message_type.get_tag(),
            false,
            true,
            Some(ListType::Logged),
            false,
        )));
        // warning
        let message_type = MessageType::Warning;
        self.tags.add(TagKind::Tag(Tag::new(
            message_type,
            message_type.get_tag(),
            multi_line_all_messages || process_messages.is_multi_line_warning(),
            false,
            if log_all_messages {
                Some(ListType::Logged)
            } else {
                message_type.get_list_type()
            },
            true,
        )));
        // chunk error
        if process_messages.is_chunks() {
            let message_type = MessageType::ChunkError;
            self.tags.add(TagKind::Enclosed(EnclosedTag::new(
                message_type,
                message_type.get_tag(),
                "END CHUNK ERROR",
                true,
                message_type.get_list_type(),
                true,
            )));
        }
        // info
        let message_type = MessageType::Info;
        self.tags.add(TagKind::Tag(Tag::new(
            message_type,
            message_type.get_tag(),
            multi_line_all_messages || process_messages.is_multi_line_info(),
            false,
            if process_messages.is_log_info_messages() || log_all_messages {
                Some(ListType::Logged)
            } else {
                message_type.get_list_type()
            },
            false,
        )));
        // log
        // Log tags are always stripped.
        let multi_line_log = process_messages.is_allow_multi_line_log();
        let message_type = MessageType::Log;
        self.tags.add(TagKind::Tag(Tag::new(
            message_type,
            message_type.get_tag(),
            multi_line_log,
            true,
            Some(ListType::Logged),
            false,
        )));
        // End of line tags cause a truncation of everything after the tag.
        self.tags.add(TagKind::Eol(EolTag::new(
            message_type,
            message_type.get_secondary_tag(),
            multi_line_log,
            Some(ListType::Logged),
        )));
        // error
        // Use antitags to avoid messages that are not actually error messages.
        let antitags = ["prnstr('ERROR:", "log.write('ERROR:"];
        // Override log tag errors cannot be logged.  Treated the same as other errors.
        // Takes presedence because it most likely contains an error tag - so it must be
        // found first.
        let message_type = MessageType::Error;
        if let Some(error_override_log_tag) = process_messages.get_error_override_log_tag() {
            self.tags.add(TagKind::Tag(Tag::new_antitags(
                message_type,
                Some(error_override_log_tag),
                multi_line_all_messages && error_tag_always_multiline,
                false,
                message_type.get_list_type(),
                true,
                &antitags,
            )));
        }
        // Basic error message.
        self.tags.add(TagKind::Tag(Tag::new_antitags(
            message_type,
            message_type.get_tag(),
            multi_line_all_messages,
            false,
            if log_all_messages {
                Some(ListType::Logged)
            } else {
                message_type.get_list_type()
            },
            true,
            &antitags,
        )));
        // Optional error message.
        if let Some(error_tag) = error_tag {
            self.tags.add(TagKind::Tag(Tag::new(
                message_type,
                Some(error_tag),
                multi_line_all_messages || error_tag_always_multiline,
                false,
                if log_all_messages {
                    Some(ListType::Logged)
                } else {
                    message_type.get_list_type()
                },
                true,
            )));
        }
        // Alternative error messages.
        self.tags.add(TagKind::Tag(Tag::new(
            message_type,
            Some(STANDARD_ERROR_TAGS[1]),
            multi_line_all_messages,
            false,
            if log_all_messages {
                Some(ListType::Logged)
            } else {
                message_type.get_list_type()
            },
            true,
        )));
        self.tags.add(TagKind::Tag(Tag::new(
            message_type,
            Some(STANDARD_ERROR_TAGS[2]),
            true,
            false,
            if log_all_messages {
                Some(ListType::Logged)
            } else {
                message_type.get_list_type()
            },
            true,
        )));
        // success - flags success - uses one or two tags (in order and not overlapping)
        if let Some(success_tag1) = process_messages.get_success_tag1() {
            self.tags.add(TagKind::Flag(FlagTag::new(
                MessageType::Success,
                success_tag1,
                process_messages.get_success_tag2(),
                Some(ListType::Flag),
            )));
        }
    }

    /// Java synchronized `setPrepend(String)`.  Creates a prepend tag and adds it to
    /// tags that require it.  Adds prepend tag to tags list.  Succeeding calls just
    /// modify the prepend tag's tag string and remove the current message.
    pub(crate) fn set_prepend(&mut self, prepend_tag_string: Option<&str>) {
        if self.prepend_tag.is_none() {
            let Some(prepend_tag_string) = prepend_tag_string else {
                return;
            };
            let mut prepend_tag = PrependTag::new(prepend_tag_string);
            let size = self.tags.size();
            for i in 0..size {
                if self.tags.takes_prepend(i) {
                    prepend_tag.add_prepend_taker(i);
                }
            }
            // Add prepend with lowest precedence.
            self.prepend_tag = Some(self.tags.add(TagKind::Prepend(prepend_tag)));
        }
        let prepend_tag = self.prepend_tag.unwrap();
        match prepend_tag_string {
            // The prepend tag has been removed.
            None => PrependTag::delete_tag(&mut self.tags, prepend_tag),
            Some(prepend_tag_string) => {
                PrependTag::set_tag(&mut self.tags, prepend_tag, Some(prepend_tag_string))
            }
        }
    }

    /// Java synchronized `parse(String)` (and `parse()`, with a null header).  Store
    /// all messages in all available lines.
    pub(crate) fn parse(&mut self, process_messages: &mut ProcessMessages, header: Option<&str>) {
        let mut header = header.map(str::to_owned);
        // Process each line.
        loop {
            let line = process_messages.get_next_line();
            // Stop processing and clean up if necessary on a null line.
            let Some(line) = line else {
                break;
            };
            // Ignore the end stringfeed token
            if line == END_FEED_TOKEN {
                continue;
            }
            // Save the interior of a multiline message without parsing, or close a
            // multiline message.
            // See if the current line matches a tag.
            let mut possible_tag = self.find_tag(Some(&line));
            let mut tag: Option<usize> = None;

            // Handle existing multi-line message.
            if self
                .multiline_message_queue
                .as_ref()
                .is_some_and(|queue| !queue.is_empty())
                && let Some(multiline_tag) = self.multiline_tag
            {
                // Ignore the interior of a multiline message that is closed with an
                // empty line.
                if !self.tags.is_enclosed(multiline_tag) {
                    // The possible tag is part of this message, unless this message is
                    // an error and so is the possible tag.
                    if let Some(possible) = possible_tag
                        && self.tags.get_message_type(multiline_tag) == MessageType::Error
                        && self.tags.get_message_type(possible) == MessageType::Error
                    {
                        tag = Some(possible);
                        possible_tag = None;
                    }
                    // Close message.  It can close on an empty line, or if the
                    // multi-line message is an error, it can close on another error.
                    if line.is_empty() || tag.is_some() {
                        // Store the multiline message
                        process_messages.store_message_queue(
                            header.as_deref(),
                            self.multiline_message_queue.as_mut(),
                            self.chunk_message,
                        );
                        header = None;
                        self.multiline_message_queue.as_mut().unwrap().clear();
                        if self.tags.is_chunk(multiline_tag) {
                            self.chunk_message = false;
                        }
                        self.multiline_tag = None;
                        // Can only leave this interation if there isn't a new tag.
                        if tag.is_none() {
                            continue;
                        }
                    }
                }
                // Search only for the close tag of an enclosed multiline message
                else if self.tags.parse(multiline_tag, Some(&line))
                    && self.tags.is_closed(multiline_tag)
                {
                    let mut message = Message::new(&self.tags, multiline_tag);
                    message.append(self.tags.get_message_string(multiline_tag).as_deref());
                    self.tags.delete_message_string(multiline_tag);
                    self.multiline_tag = None;
                    let chunk = message.is_chunk();
                    let queue = self.multiline_message_queue.as_mut().unwrap();
                    queue.push(Some(message));
                    process_messages.store_message_queue(
                        header.as_deref(),
                        Some(queue),
                        self.chunk_message,
                    );
                    header = None;
                    self.multiline_message_queue.as_mut().unwrap().clear();
                    if chunk {
                        self.chunk_message = false;
                    }
                    continue;
                }
                // Save the interior line.  Even if it's been identified as a tag, if it
                // did not close the multi-line tag, it's part of this tag.
                if tag.is_none() {
                    let multiline_tag = self.multiline_tag.unwrap();
                    let mut message = Message::new(&self.tags, multiline_tag);
                    message.append(Some(&line));
                    self.multiline_message_queue
                        .as_mut()
                        .unwrap()
                        .push(Some(message));
                    continue;
                }
            }
            // Not currently working on a multi-line message, or completed one by
            // finding another message.
            if tag.is_none() {
                tag = possible_tag;
            }
            // Handle tag.
            if let Some(tag) = tag {
                // Start a multiline message
                if self.tags.is_multi_line(tag)
                    && self.tags.is_open(tag)
                    && !self.tags.is_closed(tag)
                {
                    if self.multiline_message_queue.is_none() {
                        self.multiline_message_queue = Some(Queue::new());
                    }
                    self.multiline_tag = Some(tag);
                    let mut message = Message::new(&self.tags, tag);
                    message.append(self.tags.get_message_string(tag).as_deref());
                    self.tags.delete_message_string(tag);
                    let chunk = message.is_chunk();
                    self.multiline_message_queue
                        .as_mut()
                        .unwrap()
                        .push(Some(message));
                    if chunk {
                        self.chunk_message = true;
                    }
                }
                // Handle single line message
                else if !self.tags.is_multi_line(tag)
                    || (self.tags.is_open(tag) && self.tags.is_closed(tag))
                {
                    process_messages.store_message_tag(
                        header.as_deref(),
                        &mut self.tags,
                        tag,
                        self.chunk_message,
                    );
                    header = None;
                    self.tags.delete_message_string(tag);
                } else {
                    // No real message was found
                    self.tags.delete_message_string(tag);
                }
            }
        }
        // Handle clean up if necessary
        // Assume process output is complete when output ends - except when string feed
        // is in use or the parse is unfinished.
        if !process_messages.is_string_feed()
            && self.finished
            && self
                .multiline_message_queue
                .as_ref()
                .is_some_and(|queue| !queue.is_empty())
        {
            let chunk = self
                .multiline_message_queue
                .as_ref()
                .unwrap()
                .peek()
                .unwrap()
                .is_chunk();
            process_messages.store_message_queue(
                header.as_deref(),
                self.multiline_message_queue.as_mut(),
                self.chunk_message,
            );
            self.multiline_message_queue.as_mut().unwrap().clear();
            if chunk {
                self.chunk_message = false;
            }
        }
    }

    /// Java private `findTag(String)`.
    fn find_tag(&mut self, line: Option<&str>) -> Option<usize> {
        let line = line?;
        if line.is_empty() {
            return None;
        }
        let size = self.tags.size();
        for i in 0..size {
            if self.tags.parse(i, Some(line)) {
                if self.tags.is_prepend(i) {
                    return None;
                }
                return Some(i);
            }
        }
        None
    }
}
