//! `IMOD/Etomo/src/etomo/process/TagInterface.java`.
//!
//! The interface of the message parser's tags.  The tags of one `MessageParser` refer
//! to each other (a `PrependTag` holds its prepend takers, and each taker holds the
//! `PrependTag`), so they live in one [`TagList`] owned by the parser and a Java
//! reference to a tag is its index there.  Each interface method is dispatched on the
//! concrete class by `TagList`; the methods that follow another tag's reference take
//! the list.

use super::enclosed_tag::EnclosedTag;
use super::eol_tag::EolTag;
use super::flag_tag::FlagTag;
use super::prepend_tag::PrependTag;
use super::process_messages::{ListType, MessageType};
use super::tag::Tag;

/// The concrete classes implementing Java `interface TagInterface`.
#[derive(Clone, Debug)]
pub(crate) enum TagKind {
    Tag(Tag),
    Enclosed(EnclosedTag),
    Eol(EolTag),
    Flag(FlagTag),
    Prepend(PrependTag),
}

/// The parser's tags.  A Java `TagInterface` reference is an index into this list.
#[derive(Clone, Debug, Default)]
pub(crate) struct TagList {
    tags: Vec<TagKind>,
}

impl TagList {
    /// An empty list.
    pub(crate) fn new() -> TagList {
        TagList { tags: Vec::new() }
    }

    /// `List.add(TagInterface)`: adds the tag and returns its reference.
    pub(crate) fn add(&mut self, tag: TagKind) -> usize {
        self.tags.push(tag);
        self.tags.len() - 1
    }

    /// `List.size()`.
    pub(crate) fn size(&self) -> usize {
        self.tags.len()
    }

    /// The `Tag` part of a `Tag`, `EnclosedTag` or `PrependTag` (Java `Tag` and its two
    /// subclasses).
    pub(crate) fn tag(&self, index: usize) -> Option<&Tag> {
        match &self.tags[index] {
            TagKind::Tag(tag) => Some(tag),
            TagKind::Enclosed(tag) => Some(&tag.base),
            TagKind::Prepend(tag) => Some(&tag.base),
            TagKind::Eol(_) | TagKind::Flag(_) => None,
        }
    }

    /// Mutable form of [`TagList::tag`].
    pub(crate) fn tag_mut(&mut self, index: usize) -> Option<&mut Tag> {
        match &mut self.tags[index] {
            TagKind::Tag(tag) => Some(tag),
            TagKind::Enclosed(tag) => Some(&mut tag.base),
            TagKind::Prepend(tag) => Some(&mut tag.base),
            TagKind::Eol(_) | TagKind::Flag(_) => None,
        }
    }

    /// The tag at `index`.
    pub(crate) fn get(&self, index: usize) -> &TagKind {
        &self.tags[index]
    }

    /// The tag at `index`, mutably.
    pub(crate) fn get_mut(&mut self, index: usize) -> &mut TagKind {
        &mut self.tags[index]
    }

    /// Java `deleteMessageString()`.
    pub(crate) fn delete_message_string(&mut self, index: usize) {
        match &mut self.tags[index] {
            TagKind::Tag(tag) => tag.delete_message_string(),
            TagKind::Enclosed(tag) => tag.delete_message_string(),
            TagKind::Eol(tag) => tag.delete_message_string(),
            TagKind::Flag(tag) => tag.delete_message_string(),
            TagKind::Prepend(_) => PrependTag::delete_message_string(self, index),
        }
    }

    /// Java `getMessageString()`.  A taker of a prepend adds the prepend line, which
    /// deletes the prepend tag's message (and so every taker's prepend).
    pub(crate) fn get_message_string(&mut self, index: usize) -> Option<String> {
        match &self.tags[index] {
            TagKind::Tag(_) => Tag::get_message_string(self, index),
            TagKind::Enclosed(_) => EnclosedTag::get_message_string(self, index),
            TagKind::Eol(tag) => tag.get_message_string(),
            TagKind::Flag(tag) => tag.get_message_string(),
            TagKind::Prepend(tag) => tag.get_message_string(),
        }
    }

    /// Java `getMessageType()`.
    pub(crate) fn get_message_type(&self, index: usize) -> MessageType {
        match &self.tags[index] {
            TagKind::Tag(tag) => tag.get_message_type(),
            TagKind::Enclosed(tag) => tag.base.get_message_type(),
            TagKind::Eol(tag) => tag.get_message_type(),
            TagKind::Flag(tag) => tag.get_message_type(),
            TagKind::Prepend(tag) => tag.base.get_message_type(),
        }
    }

    /// Java `equalsEndTag(TagInterface)`.
    pub(crate) fn equals_end_tag(&self, index: usize, tag: Option<usize>) -> bool {
        match &self.tags[index] {
            TagKind::Enclosed(_) => EnclosedTag::equals_end_tag(self, index, tag),
            TagKind::Tag(_) | TagKind::Eol(_) | TagKind::Flag(_) | TagKind::Prepend(_) => false,
        }
    }

    /// Java `isChunk()`.
    pub(crate) fn is_chunk(&self, index: usize) -> bool {
        match &self.tags[index] {
            TagKind::Tag(tag) => tag.is_chunk(),
            TagKind::Enclosed(tag) => tag.base.is_chunk(),
            TagKind::Eol(tag) => tag.is_chunk(),
            TagKind::Flag(tag) => tag.is_chunk(),
            TagKind::Prepend(tag) => tag.base.is_chunk(),
        }
    }

    /// Java `getListType()`.
    pub(crate) fn get_list_type(&self, index: usize) -> Option<ListType> {
        match &self.tags[index] {
            TagKind::Tag(tag) => tag.get_list_type(),
            TagKind::Enclosed(tag) => tag.base.get_list_type(),
            TagKind::Eol(tag) => tag.get_list_type(),
            TagKind::Flag(tag) => tag.get_list_type(),
            TagKind::Prepend(tag) => tag.base.get_list_type(),
        }
    }

    /// Java `takesPrepend()`.
    pub(crate) fn takes_prepend(&self, index: usize) -> bool {
        match &self.tags[index] {
            TagKind::Tag(tag) => tag.takes_prepend(),
            TagKind::Enclosed(tag) => tag.base.takes_prepend(),
            TagKind::Eol(tag) => tag.takes_prepend(),
            TagKind::Flag(tag) => tag.takes_prepend(),
            TagKind::Prepend(tag) => tag.base.takes_prepend(),
        }
    }

    /// Java `setPrepend(PrependTag)`.
    pub(crate) fn set_prepend(&mut self, index: usize, prepend: Option<usize>) {
        match &self.tags[index] {
            TagKind::Tag(_) | TagKind::Enclosed(_) => Tag::set_prepend(self, index, prepend),
            TagKind::Eol(_) | TagKind::Flag(_) | TagKind::Prepend(_) => {}
        }
    }

    /// Java `deletePrepend()`.
    pub(crate) fn delete_prepend(&mut self, index: usize) {
        match &mut self.tags[index] {
            TagKind::Tag(tag) => tag.delete_prepend(),
            TagKind::Enclosed(tag) => tag.base.delete_prepend(),
            TagKind::Eol(_) | TagKind::Flag(_) | TagKind::Prepend(_) => {}
        }
    }

    /// Java `isPrepend()`.
    pub(crate) fn is_prepend(&self, index: usize) -> bool {
        matches!(self.tags[index], TagKind::Prepend(_))
    }

    /// Java `isEnclosed()`.
    pub(crate) fn is_enclosed(&self, index: usize) -> bool {
        matches!(self.tags[index], TagKind::Enclosed(_))
    }

    /// Java `parse(String)`.
    pub(crate) fn parse(&mut self, index: usize, line: Option<&str>) -> bool {
        match &mut self.tags[index] {
            TagKind::Tag(tag) => tag.parse(line),
            TagKind::Enclosed(tag) => tag.parse(line),
            TagKind::Eol(tag) => tag.parse(line),
            TagKind::Flag(tag) => tag.parse(line),
            TagKind::Prepend(_) => PrependTag::parse(self, index, line),
        }
    }

    /// Java `isMultiLine()`.
    pub(crate) fn is_multi_line(&self, index: usize) -> bool {
        match &self.tags[index] {
            TagKind::Tag(tag) => tag.is_multi_line(),
            TagKind::Enclosed(tag) => tag.base.is_multi_line(),
            TagKind::Eol(tag) => tag.is_multi_line(),
            TagKind::Flag(tag) => tag.is_multi_line(),
            TagKind::Prepend(tag) => tag.base.is_multi_line(),
        }
    }

    /// Java `isOpen()`.
    pub(crate) fn is_open(&self, index: usize) -> bool {
        match &self.tags[index] {
            TagKind::Tag(tag) => tag.is_open(),
            TagKind::Enclosed(tag) => tag.base.is_open(),
            TagKind::Eol(tag) => tag.is_open(),
            TagKind::Flag(tag) => tag.is_open(),
            TagKind::Prepend(tag) => tag.base.is_open(),
        }
    }

    /// Java `isClosed()`.
    pub(crate) fn is_closed(&self, index: usize) -> bool {
        match &self.tags[index] {
            TagKind::Tag(tag) => tag.is_closed(),
            TagKind::Enclosed(tag) => tag.base.is_closed(),
            TagKind::Eol(tag) => tag.is_closed(),
            TagKind::Flag(tag) => tag.is_closed(),
            TagKind::Prepend(tag) => tag.base.is_closed(),
        }
    }

    /// Java `isFlag()`.
    pub(crate) fn is_flag(&self, index: usize) -> bool {
        matches!(self.tags[index], TagKind::Flag(_))
    }
}
