//! `IMOD/Etomo/src/etomo/process/FlagTag.java`.
//!
//! A tag that only flags that it was found (success), with an optional second tag that
//! must follow the first.

use super::process_messages::{ListType, MessageType};

/// Java `final class FlagTag implements TagInterface`.
#[derive(Clone, Debug)]
pub(crate) struct FlagTag {
    /// Java private final `type`.
    r#type: MessageType,
    /// Java private final `tag1`.
    tag1: String,
    /// Java private final `tag2`.
    tag2: Option<String>,
    /// Java private final `listType`.
    list_type: Option<ListType>,
    /// Java private `found`.
    found: bool,
}

impl FlagTag {
    /// Java `FlagTag(MessageType, String, String, ListType)`.
    pub(crate) fn new(
        r#type: MessageType,
        tag1: &str,
        tag2: Option<&str>,
        list_type: Option<ListType>,
    ) -> FlagTag {
        FlagTag {
            r#type,
            tag1: tag1.to_owned(),
            tag2: tag2.map(str::to_owned),
            list_type,
            found: false,
        }
    }

    /// Java `parse(String)`.
    pub(crate) fn parse(&mut self, line: Option<&str>) -> bool {
        self.reset();
        let line = match line {
            Some(line) if !line.is_empty() => line,
            _ => return false,
        };
        if let Some(index) = line.find(self.tag1.as_str()) {
            self.found = true;
            if let Some(tag2) = &self.tag2 {
                let index = index + self.tag1.len();
                if !line[index..].contains(tag2.as_str()) {
                    self.found = false;
                }
            }
        }
        self.found
    }

    /// Java `deleteMessageString()`.
    pub(crate) fn delete_message_string(&mut self) {
        self.reset();
    }

    /// Java private `reset()`.
    fn reset(&mut self) {
        self.found = false;
    }

    /// Java `getMessageType()`.
    pub(crate) fn get_message_type(&self) -> MessageType {
        self.r#type
    }

    /// Java `getListType()`.
    pub(crate) fn get_list_type(&self) -> Option<ListType> {
        self.list_type
    }

    /// Java `getPrepend()`.
    pub(crate) fn get_prepend(&self) -> Option<usize> {
        None
    }

    /// Java `getMessageString()`.
    pub(crate) fn get_message_string(&self) -> Option<String> {
        None
    }

    /// Java `isOpen()`.
    pub(crate) fn is_open(&self) -> bool {
        self.found
    }

    /// Java `isClosed()`.
    pub(crate) fn is_closed(&self) -> bool {
        self.found
    }

    /// Java `isMultiLine()`.
    pub(crate) fn is_multi_line(&self) -> bool {
        false
    }

    /// Java `takesPrepend()`.
    pub(crate) fn takes_prepend(&self) -> bool {
        false
    }

    /// Java `isChunk()`.
    pub(crate) fn is_chunk(&self) -> bool {
        false
    }
}

/// Java `toString()`: `getClass() + "," + type + "," + tag1 + "," + tag2`.
impl std::fmt::Display for FlagTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "class etomo.process.FlagTag,{},{},{}",
            self.r#type,
            self.tag1,
            self.tag2.as_deref().unwrap_or("null")
        )
    }
}
