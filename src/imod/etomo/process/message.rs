//! `IMOD/Etomo/src/etomo/process/Message.java`.
#![allow(dead_code)]

use super::process_messages::{ListType, MessageType};

/// Java package-private `Message`.  The tag properties are copied when a message opens;
/// this is the Rust ownership equivalent of Java's retained `TagInterface` reference.
#[derive(Clone, Debug)]
pub(crate) struct Message {
    builder: String,
    message_type: MessageType,
    list_type: Option<ListType>,
    enclosed: bool,
    chunk: bool,
    end_with_new_line: bool,
}

impl Message {
    pub(crate) fn new(
        message_type: MessageType,
        list_type: Option<ListType>,
        enclosed: bool,
        chunk: bool,
        multi_line: bool,
    ) -> Self {
        Self {
            builder: String::new(),
            message_type,
            list_type,
            enclosed,
            chunk,
            end_with_new_line: multi_line,
        }
    }
    pub(crate) fn get_message_string(&mut self) -> String {
        if self.end_with_new_line {
            self.builder.push('\n');
            self.end_with_new_line = false;
        }
        self.builder.clone()
    }
    pub(crate) fn append(&mut self, string: Option<&str>) {
        let Some(string) = string else { return };
        if !self.builder.is_empty() {
            self.builder.push('\n');
        }
        self.builder.push_str(string);
    }
    pub(crate) fn is_enclosed(&self) -> bool {
        self.enclosed
    }
    pub(crate) fn get_message_type(&self) -> MessageType {
        self.message_type
    }
    pub(crate) fn get_list_type(&self) -> Option<ListType> {
        self.list_type
    }
    pub(crate) fn is_chunk(&self) -> bool {
        self.chunk
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.builder)
    }
}
