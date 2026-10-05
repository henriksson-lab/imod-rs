//! `IMOD/Etomo/src/etomo/process/PrependTag.java`.
//!
//! A line saved to be put in front of the next message of a tag that takes a prepend.
//! Java `final class PrependTag extends Tag`: the superclass state is `base`; the
//! prepend takers are tags of the same [`TagList`].

use super::process_messages::MessageType;
use super::tag::Tag;
use super::tag_interface::{TagKind, TagList};

/// Java `final class PrependTag extends Tag`.
#[derive(Clone, Debug)]
pub(crate) struct PrependTag {
    /// Java superclass `Tag`.
    pub(crate) base: Tag,
    /// Java private final `prependTakers`.
    prepend_takers: Vec<usize>,
}

impl PrependTag {
    /// Java `PrependTag(String)`.
    pub(crate) fn new(tag: &str) -> PrependTag {
        PrependTag {
            base: Tag::new(MessageType::Prepend, Some(tag), false, false, None, false),
            prepend_takers: Vec::new(),
        }
    }

    /// Java package-private `addPrependTaker(TagInterface)`.
    pub(crate) fn add_prepend_taker(&mut self, tag: usize) {
        self.prepend_takers.push(tag);
    }

    /// Java `parse(String)`.
    pub(crate) fn parse(tags: &mut TagList, index: usize, line: Option<&str>) -> bool {
        let TagKind::Prepend(this) = tags.get_mut(index) else {
            return false;
        };
        if this.base.parse(line) {
            this.base.set_start_index(0);
            let prepend_takers = this.prepend_takers.clone();
            for taker in prepend_takers {
                tags.set_prepend(taker, Some(index));
            }
            return true;
        }
        false
    }

    /// Java `getMessageString()`.  The whole line is the prepend; it is not trimmed.
    pub(crate) fn get_message_string(&self) -> Option<String> {
        let mut message_string = None;
        let line = self.base.get_line();
        if let Some(line) = line
            && self.base.get_start_index() != -1
        {
            // Whole line is part of prepend
            message_string = Some(line.to_owned());
        }
        // Prepend is not trimmed
        message_string
    }

    /// Java `deleteMessageString()`.
    pub(crate) fn delete_message_string(tags: &mut TagList, index: usize) {
        let TagKind::Prepend(this) = tags.get_mut(index) else {
            return;
        };
        this.base.reset(None);
        let prepend_takers = this.prepend_takers.clone();
        for taker in prepend_takers {
            tags.delete_prepend(taker);
        }
    }

    /// Java package-private `setTag(String)` (inherited; `deleteMessageString` is this
    /// class's).
    pub(crate) fn set_tag(tags: &mut TagList, index: usize, new_tag: Option<&str>) {
        Tag::set_tag(tags, index, new_tag);
    }

    /// Java package-private `deleteTag()` (inherited).
    pub(crate) fn delete_tag(tags: &mut TagList, index: usize) {
        Tag::delete_tag(tags, index);
    }
}

/// Java `toString()` (inherited from `Tag`, with the class name).
impl std::fmt::Display for PrependTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            &self
                .base
                .to_string_class_info("class:class etomo.process.PrependTag"),
        )
    }
}
