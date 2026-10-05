//! `IMOD/Etomo/src/etomo/process/EolTag.java`.
//!
//! A tag that ends the message: everything from the tag to the end of the line is
//! dropped.

use super::process_messages::{ListType, MessageType};
use super::tag::java_lang_string_trim;

/// Java `final class EolTag implements TagInterface`.
#[derive(Clone, Debug)]
pub(crate) struct EolTag {
    /// Java private final `type`.
    r#type: MessageType,
    /// Java private final `listType`.
    list_type: Option<ListType>,
    /// Java private final `tag`.
    tag: Option<String>,
    /// Java private final `multiLine`.
    multi_line: bool,
    /// Java private `line`.
    line: Option<String>,
    /// Java private `endIndex`.
    end_index: i32,
    /// Java private `open`.
    open: bool,
    /// Java private `closed`.
    closed: bool,
}

impl EolTag {
    /// Java `EolTag(MessageType, String, boolean, ListType)`.
    pub(crate) fn new(
        r#type: MessageType,
        tag: Option<&str>,
        multi_line: bool,
        list_type: Option<ListType>,
    ) -> EolTag {
        EolTag {
            r#type,
            list_type,
            tag: tag.map(str::to_owned),
            multi_line,
            line: None,
            end_index: -1,
            open: false,
            closed: false,
        }
    }

    /// Java `parse(String)`.  `line.indexOf(null)` throws in the source; every
    /// instance is built with a tag.
    pub(crate) fn parse(&mut self, line: Option<&str>) -> bool {
        self.reset(line);
        let Some(line) = line else {
            return false;
        };
        let tag = self.tag.as_deref().expect("java.lang.NullPointerException");
        if let Some(end_index) = line.find(tag) {
            self.end_index = end_index as i32;
            // Strip end tag
            self.open = true;
            if !self.multi_line {
                self.closed = true;
            }
            return true;
        }
        self.end_index = -1;
        false
    }

    /// Java final `getMessageString()`.
    pub(crate) fn get_message_string(&self) -> Option<String> {
        let line = self.line.as_deref()?;
        if self.end_index == -1 {
            return None;
        }
        Some(java_lang_string_trim(&line[..self.end_index as usize]))
    }

    /// Java `deleteMessageString()`.
    pub(crate) fn delete_message_string(&mut self) {
        self.reset(None);
    }

    /// Java private `reset(String)`.
    fn reset(&mut self, line: Option<&str>) {
        self.line = line.map(str::to_owned);
        self.end_index = -1;
        self.open = false;
        self.closed = false;
    }

    /// Java `isOpen()`.
    pub(crate) fn is_open(&self) -> bool {
        self.open
    }

    /// Java `isClosed()`.
    pub(crate) fn is_closed(&self) -> bool {
        self.closed
    }

    /// Java `isChunk()`.
    pub(crate) fn is_chunk(&self) -> bool {
        self.list_type.is_some_and(ListType::is_chunk)
    }

    /// Java `isMultiLine()`.
    pub(crate) fn is_multi_line(&self) -> bool {
        self.multi_line
    }

    /// Java `takesPrepend()`.
    pub(crate) fn takes_prepend(&self) -> bool {
        false
    }

    /// Java `getMessageType()`.
    pub(crate) fn get_message_type(&self) -> MessageType {
        self.r#type
    }

    /// Java `getListType()`.
    pub(crate) fn get_list_type(&self) -> Option<ListType> {
        self.list_type
    }
}

/// Java `toString()`: `getClass() + "," + type + "," + tag + "," + line`.
impl std::fmt::Display for EolTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "class etomo.process.EolTag,{},{},{}",
            self.r#type,
            self.tag.as_deref().unwrap_or("null"),
            self.line.as_deref().unwrap_or("null")
        )
    }
}
