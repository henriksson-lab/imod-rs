//! `IMOD/Etomo/src/etomo/process/Message.java`.
//!
//! One message found by `MessageParser`.  Java holds the `TagInterface` that found it;
//! every tag property a message reads (type, list type, chunk, enclosed, multi-line) is
//! final in the tag classes, so the message records them when it is made, together with
//! the tag's reference for `equalsEndTag`.

use super::process_messages::{ListType, MessageType};
use super::tag_interface::TagList;

/// Java package-private `final class Message`.
#[derive(Clone, Debug)]
pub(crate) struct Message {
    /// Java private final `builder`.
    builder: String,
    /// Java private final `tag` (its index in the parser's tag list).
    tag: usize,
    /// `tag.getMessageType()`.
    message_type: MessageType,
    /// `tag.getListType()`.
    list_type: Option<ListType>,
    /// `tag.isEnclosed()`.
    enclosed: bool,
    /// `tag.isChunk()`.
    chunk: bool,
    /// Java private `endWithNewLine`.
    end_with_new_line: bool,
}

impl Message {
    /// Java `Message(TagInterface)`.
    pub(crate) fn new(tags: &TagList, tag: usize) -> Message {
        Message {
            builder: String::new(),
            tag,
            message_type: tags.get_message_type(tag),
            list_type: tags.get_list_type(tag),
            enclosed: tags.is_enclosed(tag),
            chunk: tags.is_chunk(tag),
            end_with_new_line: tags.is_multi_line(tag),
        }
    }

    /// Java package-private `getMessageString()`.
    pub(crate) fn get_message_string(&mut self) -> String {
        if self.end_with_new_line {
            self.builder.push('\n');
            self.end_with_new_line = false;
        }
        self.builder.clone()
    }

    /// Java package-private `append(String)`.
    pub(crate) fn append(&mut self, string: Option<&str>) {
        let Some(string) = string else {
            return;
        };
        if !self.builder.is_empty() {
            self.builder.push('\n');
        }
        self.builder.push_str(string);
    }

    /// Java package-private `equalsEndTag(TagInterface)`.
    pub(crate) fn equals_end_tag(&self, tags: &TagList, tag: Option<usize>) -> bool {
        tags.equals_end_tag(self.tag, tag)
    }

    /// Java package-private `isEnclosed()`.
    pub(crate) fn is_enclosed(&self) -> bool {
        self.enclosed
    }

    /// Java package-private `getMessageType()`.
    pub(crate) fn get_message_type(&self) -> MessageType {
        self.message_type
    }

    /// Java package-private `getListType()`.
    pub(crate) fn get_list_type(&self) -> Option<ListType> {
        self.list_type
    }

    /// Java package-private `isChunk()`.
    pub(crate) fn is_chunk(&self) -> bool {
        self.chunk
    }
}

/// Java `toString()`.
impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.builder)
    }
}
