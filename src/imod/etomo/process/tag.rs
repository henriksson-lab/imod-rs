//! `IMOD/Etomo/src/etomo/process/Tag.java`.
//!
//! A message tag: finds a tag string in a line and returns the message that starts
//! there.  `EnclosedTag` and `PrependTag` extend it (`base`).  A tag's prepend is the
//! index of a `PrependTag` in the parser's [`TagList`].

use super::process_messages::{ListType, MessageType};
use super::tag_interface::{TagKind, TagList};

/// Java `String.trim()`: strips the characters at or below U+0020.
pub(crate) fn java_lang_string_trim(string: &str) -> String {
    string.trim_matches(|c: char| c <= ' ').to_owned()
}

/// Java `class Tag implements TagInterface`.
#[derive(Clone, Debug)]
pub(crate) struct Tag {
    /// Java private final `type`.
    r#type: MessageType,
    /// Java private final `listType`.
    list_type: Option<ListType>,
    /// Java private final `multiLine`.
    multi_line: bool,
    /// Java private final `stripTag`.
    strip_tag: bool,
    /// Java private final `takesPrepend`.
    takes_prepend: bool,
    /// Java private final `antitags`.
    antitags: Option<Vec<String>>,
    /// Java private `tag`.
    tag: Option<String>,
    /// Java private `prependLine`.
    prepend_line: Option<String>,
    /// Java private `prepend` (a `PrependTag`).
    prepend: Option<usize>,
    /// Java private `line`.
    line: Option<String>,
    /// Java private `startIndex`.
    start_index: i32,
    /// Java private `open`.
    open: bool,
    /// Java private `closed`.
    closed: bool,
}

impl Tag {
    /// Java `Tag(MessageType, String, boolean, boolean, ListType, boolean)`.
    pub(crate) fn new(
        r#type: MessageType,
        tag: Option<&str>,
        multi_line: bool,
        strip_tag: bool,
        list_type: Option<ListType>,
        takes_prepend: bool,
    ) -> Tag {
        Tag {
            r#type,
            list_type,
            multi_line,
            strip_tag,
            takes_prepend,
            antitags: None,
            tag: tag.map(str::to_owned),
            prepend_line: None,
            prepend: None,
            line: None,
            start_index: -1,
            open: false,
            closed: false,
        }
    }

    /// Java `Tag(MessageType, String, boolean, boolean, ListType, boolean, String[])`.
    pub(crate) fn new_antitags(
        r#type: MessageType,
        tag: Option<&str>,
        multi_line: bool,
        strip_tag: bool,
        list_type: Option<ListType>,
        takes_prepend: bool,
        antitags: &[&str],
    ) -> Tag {
        let mut instance = Tag::new(r#type, tag, multi_line, strip_tag, list_type, takes_prepend);
        instance.antitags = Some(
            antitags
                .iter()
                .map(|antitag| (*antitag).to_owned())
                .collect(),
        );
        instance
    }

    /// Java `toString()`; `class_info` is the subclass's `"class:" + getClass()` (empty
    /// for a plain `Tag`).
    pub(crate) fn to_string_class_info(&self, class_info: &str) -> String {
        format!(
            "line:{},\n{},type:{},tag:{},multiLine:{},open:{},closed:{}",
            self.line.as_deref().unwrap_or("null"),
            class_info,
            self.r#type,
            self.tag.as_deref().unwrap_or("null"),
            self.multi_line,
            self.open,
            self.closed
        )
    }

    /// Java `parse(String)`.  `reset` is the plain `Tag`'s (an `EnclosedTag` resets its
    /// own state before calling this).
    pub(crate) fn parse(&mut self, line: Option<&str>) -> bool {
        self.reset(line);
        let Some(line) = line else {
            return false;
        };
        let Some(tag) = self.tag.clone() else {
            return false;
        };
        if line.is_empty() {
            return false;
        }
        // Ignore lines containing antitags.
        if let Some(antitags) = &self.antitags {
            for antitag in antitags {
                if line.contains(antitag.as_str()) {
                    return false;
                }
            }
        }
        if let Some(index) = line.find(tag.as_str()) {
            self.start_index = index as i32;
            // Found the start tag of a message
            if self.strip_tag {
                self.start_index += tag.len() as i32;
            }
            self.open = true;
            if !self.multi_line {
                self.closed = true;
            }
            return true;
        }
        self.start_index = -1;
        false
    }

    /// Java package-private `reset(String)`.
    pub(crate) fn reset(&mut self, line: Option<&str>) {
        self.line = line.map(str::to_owned);
        self.start_index = -1;
        self.open = false;
        self.closed = false;
    }

    /// Java package-private `setClosed()`.
    pub(crate) fn set_closed(&mut self) {
        self.closed = true;
    }

    /// Java package-private `setStartIndex(int)`.
    pub(crate) fn set_start_index(&mut self, start_index: i32) {
        self.start_index = start_index;
    }

    /// Java package-private `setTag(String)`.  `deleteMessageString` is virtual.
    pub(crate) fn set_tag(tags: &mut TagList, index: usize, new_tag: Option<&str>) {
        let Some(new_tag) = new_tag else {
            return;
        };
        tags.tag_mut(index).unwrap().tag = Some(new_tag.to_owned());
        tags.delete_message_string(index);
    }

    /// Java package-private `deleteTag()`.  `deleteMessageString` is virtual.
    pub(crate) fn delete_tag(tags: &mut TagList, index: usize) {
        tags.tag_mut(index).unwrap().tag = None;
        tags.delete_message_string(index);
    }

    /// Java `deleteMessageString()`.
    pub(crate) fn delete_message_string(&mut self) {
        self.reset(None);
    }

    /// Java package-private `addPrepend(String)`.
    pub(crate) fn add_prepend(
        tags: &mut TagList,
        index: usize,
        message_string: Option<String>,
    ) -> Option<String> {
        let this = tags.tag(index).unwrap();
        if !this.open || !this.takes_prepend || this.prepend_line.is_none() {
            return message_string;
        }
        let prepend_string = this.prepend_line.clone();
        let prepend = this.prepend;
        if let Some(prepend) = prepend {
            tags.delete_message_string(prepend);
        }
        let prepend_string = match prepend_string {
            Some(prepend_string) if !prepend_string.is_empty() => prepend_string,
            _ => return message_string,
        };
        if let Some(message_string) = message_string {
            return Some(prepend_string + "\n" + &message_string);
        }
        Some(prepend_string)
    }

    /// Java `getMessageString()`.
    pub(crate) fn get_message_string(tags: &mut TagList, index: usize) -> Option<String> {
        let this = tags.tag(index).unwrap();
        let mut message_string = None;
        if let Some(line) = &this.line
            && this.start_index != -1
        {
            if this.start_index == 0 {
                message_string = Some(line.clone());
            } else {
                message_string = Some(line[this.start_index as usize..].to_owned());
            }
        }
        let message_string = message_string?;
        Tag::add_prepend(tags, index, Some(java_lang_string_trim(&message_string)))
    }

    /// Java package-private `getTagLength()`.
    pub(crate) fn get_tag_length(&self) -> i32 {
        match &self.tag {
            None => 0,
            Some(tag) => tag.len() as i32,
        }
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

    /// Java `getListType()`.
    pub(crate) fn get_list_type(&self) -> Option<ListType> {
        self.list_type
    }

    /// Java final `isMultiLine()`.
    pub(crate) fn is_multi_line(&self) -> bool {
        self.multi_line
    }

    /// Java `getMessageType()`.
    pub(crate) fn get_message_type(&self) -> MessageType {
        self.r#type
    }

    /// Java `setPrepend(PrependTag)`.
    pub(crate) fn set_prepend(tags: &mut TagList, index: usize, prepend: Option<usize>) {
        let Some(prepend) = prepend else {
            return;
        };
        if !tags.tag(index).unwrap().takes_prepend {
            return;
        }
        let prepend_line = match tags.get(prepend) {
            TagKind::Prepend(prepend_tag) => prepend_tag.get_message_string(),
            _ => None,
        };
        let this = tags.tag_mut(index).unwrap();
        this.prepend = Some(prepend);
        this.prepend_line = prepend_line;
    }

    /// Java `deletePrepend()`.
    pub(crate) fn delete_prepend(&mut self) {
        self.prepend = None;
        self.prepend_line = None;
    }

    /// Java `takesPrepend()`.
    pub(crate) fn takes_prepend(&self) -> bool {
        self.takes_prepend
    }

    /// Java final package-private `getLine()`.
    pub(crate) fn get_line(&self) -> Option<&str> {
        self.line.as_deref()
    }

    /// Java final package-private `getStartIndex()`.
    pub(crate) fn get_start_index(&self) -> i32 {
        self.start_index
    }
}

/// Java `toString()` of a plain `Tag`.
impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string_class_info(""))
    }
}
