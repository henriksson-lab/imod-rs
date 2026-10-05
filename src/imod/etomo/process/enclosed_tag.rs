//! `IMOD/Etomo/src/etomo/process/EnclosedTag.java`.
//!
//! A multi-line message with a start tag and an end tag.  Java
//! `final class EnclosedTag extends Tag`: the superclass state is `base`.

use super::process_messages::{ListType, MessageType};
use super::tag::{Tag, java_lang_string_trim};
use super::tag_interface::{TagKind, TagList};
use crate::imod::etomo::util::stack_trace::StackTrace;

/// Java `final class EnclosedTag extends Tag`.
#[derive(Clone, Debug)]
pub(crate) struct EnclosedTag {
    /// Java superclass `Tag`.
    pub(crate) base: Tag,
    /// Java private final `endTag`.
    end_tag: String,
    /// Java private final `stripEndTag`.
    strip_end_tag: bool,
    /// Java private `endIndex`.
    end_index: i32,
}

impl EnclosedTag {
    /// Java `EnclosedTag(MessageType, String, String, boolean, ListType, boolean)`.
    pub(crate) fn new(
        r#type: MessageType,
        tag: Option<&str>,
        end_tag: &str,
        strip_end_tag: bool,
        list_type: Option<ListType>,
        takes_prepend: bool,
    ) -> EnclosedTag {
        EnclosedTag {
            base: Tag::new(r#type, tag, true, false, list_type, takes_prepend),
            end_tag: end_tag.to_owned(),
            strip_end_tag,
            end_index: -1,
        }
    }

    /// Java final `parse(String)`.  `super.parse` calls the virtual `reset`, which is
    /// this class's; its own state was reset just before.
    pub(crate) fn parse(&mut self, line: Option<&str>) -> bool {
        self.reset(line);
        let open_found = self.base.parse(line);
        self.end_index = -1;
        let line = match line {
            Some(line) if !line.is_empty() => line,
            _ => return false,
        };
        // Parse for the end tag
        let mut index = 0;
        if open_found {
            index = self.base.get_start_index() + self.base.get_tag_length();
        }
        if let Some(found) = line[index as usize..].find(self.end_tag.as_str()) {
            self.end_index = index + found as i32;
            if !self.strip_end_tag {
                self.end_index += self.end_tag.len() as i32;
            }
            self.base.set_closed();
            return true;
        }
        self.end_index = -1;
        open_found
    }

    /// Java package-private override `reset(String)`.
    pub(crate) fn reset(&mut self, line: Option<&str>) {
        self.base.reset(line);
        self.end_index = -1;
    }

    /// Java `deleteMessageString()`: `reset(null)`, which is this class's.
    pub(crate) fn delete_message_string(&mut self) {
        self.reset(None);
    }

    /// Java final `getMessageString()`.
    pub(crate) fn get_message_string(tags: &mut TagList, index: usize) -> Option<String> {
        let TagKind::Enclosed(this) = tags.get(index) else {
            return None;
        };
        let line = this.base.get_line();
        let start_index = this.base.get_start_index();
        let end_index = this.end_index;
        let mut message_string = None;
        if let Some(line) = line
            && (start_index != -1 || end_index != -1)
        {
            if start_index != -1 && end_index != -1 {
                message_string = Some(line[start_index as usize..end_index as usize].to_owned());
            } else if end_index != -1 {
                message_string = Some(line[..end_index as usize].to_owned());
            } else {
                // already contains prepend;
                return Tag::get_message_string(tags, index);
            }
        }
        let message_string = message_string?;
        Tag::add_prepend(tags, index, Some(java_lang_string_trim(&message_string)))
    }

    /// Java `equalsEndTag(TagInterface)`.
    pub(crate) fn equals_end_tag(tags: &TagList, index: usize, tag: Option<usize>) -> bool {
        let Some(tag) = tag else {
            return false;
        };
        if tag == index {
            return true;
        }
        let TagKind::Enclosed(this) = tags.get(index) else {
            return false;
        };
        if let TagKind::Enclosed(enclosed_tag) = tags.get(tag) {
            return this.end_tag == enclosed_tag.end_tag;
        } else if tags.is_enclosed(tag) {
            eprintln!("ERROR: Unknown enclosed tag.");
            // `Thread.dumpStack()`.
            StackTrace::new_with_thread(None, None).print(Some("Stack trace"), false);
        }
        false
    }
}

/// Java `toString()` (inherited from `Tag`, with the class name).
impl std::fmt::Display for EnclosedTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            &self
                .base
                .to_string_class_info("class:class etomo.process.EnclosedTag"),
        )
    }
}
