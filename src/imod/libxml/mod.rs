//! Translation of the vendored Mini-XML 2.10 library (`IMOD/libxml/`) and its
//! public header `IMOD/include/mxml.h`.
//!
//! The header's constants, typedefs and structures live here; each `.c` file is
//! mirrored by the module of the same name.
//!
//! # The node arena
//!
//! `struct mxml_node_s` is an intrusive doubly linked tree: every node carries
//! `parent`, `child`, `last_child`, `prev` and `next` pointers into the same
//! malloc heap, and `mxml_new`/`mxml_free` (`mxml-node.c:747`, `:690`) are the
//! only allocator calls.  Rust cannot express that graph with `&` references,
//! so the heap those two functions allocate from is modelled explicitly as
//! [`MxmlArena`]: a `Vec` of node slots with a free list, and `Option<usize>`
//! slot indices in every link field.  A node pointer in the C sources is an
//! `Option<usize>` here, `NULL` is `None`, and every function that the C gives
//! a node pointer takes the arena as an extra first argument.  That extra
//! argument and the `MxmlArena::node`/`node_mut` accessors are the only
//! additions to the source's own function set.
pub mod mxml_attr;
pub mod mxml_entity;
pub mod mxml_file;
pub mod mxml_get;
pub mod mxml_index;
pub mod mxml_node;
pub mod mxml_private;
pub mod mxml_search;
pub mod mxml_set;
pub mod mxml_string;

use core::any::Any;
use core::ffi::c_int;

/* --- IMOD/include/mxml.h: constants ----------------------------------- */

/// Matches C `MXML_MAJOR_VERSION` (`mxml.h:38`).
pub const MXML_MAJOR_VERSION: c_int = 2;
/// Matches C `MXML_MINOR_VERSION` (`mxml.h:39`).
pub const MXML_MINOR_VERSION: c_int = 10;

/// Matches C `MXML_TAB` (`mxml.h:41`).
pub const MXML_TAB: c_int = 8;

/// Matches C `MXML_NO_CALLBACK` (`mxml.h:43`).
pub const MXML_NO_CALLBACK: MxmlLoadCb = None;
/// Matches C `MXML_TEXT_CALLBACK` (`mxml.h:50`).
pub const MXML_TEXT_CALLBACK: MxmlLoadCb = None;

/// Matches C `MXML_NO_PARENT` (`mxml.h:54`), the C `NULL` parent pointer.
pub const MXML_NO_PARENT: Option<usize> = None;

pub const MXML_DESCEND: c_int = 1;
pub const MXML_NO_DESCEND: c_int = 0;
pub const MXML_DESCEND_FIRST: c_int = -1;

pub const MXML_WS_BEFORE_OPEN: c_int = 0;
pub const MXML_WS_AFTER_OPEN: c_int = 1;
pub const MXML_WS_BEFORE_CLOSE: c_int = 2;
pub const MXML_WS_AFTER_CLOSE: c_int = 3;

pub const MXML_ADD_BEFORE: c_int = 0;
pub const MXML_ADD_AFTER: c_int = 1;
/// Matches C `MXML_ADD_TO_PARENT` (`mxml.h:67`), the C `NULL` child pointer.
pub const MXML_ADD_TO_PARENT: Option<usize> = None;

/* --- IMOD/include/mxml.h: data types ---------------------------------- */

/// Matches C `mxml_sax_event_t` (`mxml.h:74`).
pub type MxmlSaxEvent = c_int;
pub const MXML_SAX_CDATA: MxmlSaxEvent = 0;
pub const MXML_SAX_COMMENT: MxmlSaxEvent = 1;
pub const MXML_SAX_DATA: MxmlSaxEvent = 2;
pub const MXML_SAX_DIRECTIVE: MxmlSaxEvent = 3;
pub const MXML_SAX_ELEMENT_CLOSE: MxmlSaxEvent = 4;
pub const MXML_SAX_ELEMENT_OPEN: MxmlSaxEvent = 5;

/// Matches C `mxml_type_t` (`mxml.h:83`).
pub type MxmlType = c_int;
pub const MXML_IGNORE: MxmlType = -1;
pub const MXML_ELEMENT: MxmlType = 0;
pub const MXML_INTEGER: MxmlType = 1;
pub const MXML_OPAQUE: MxmlType = 2;
pub const MXML_REAL: MxmlType = 3;
pub const MXML_TEXT: MxmlType = 4;
pub const MXML_CUSTOM: MxmlType = 5;

/// Matches C `mxml_custom_destroy_cb_t` (`mxml.h:93`).
///
/// The C callback is handed the `void *` it must `free`; here the custom data
/// is an owned `Box`, so the callback is given a borrow of it and the box is
/// dropped afterwards by `mxml_free` (`mxml-node.c:735`).
pub type MxmlCustomDestroyCb = Option<fn(&mut dyn Any)>;
/// Matches C `mxml_error_cb_t` (`mxml.h:96`).
pub type MxmlErrorCb = Option<fn(&[u8])>;
/// Matches C `mxml_custom_load_cb_t` (`mxml.h:157`).
pub type MxmlCustomLoadCb = Option<fn(&mut MxmlArena, usize, &[u8]) -> c_int>;
/// Matches C `mxml_custom_save_cb_t` (`mxml.h:160`).
///
/// The C callback returns a `malloc`ed string that the caller frees; the owned
/// `Vec` carries the same ownership transfer.
pub type MxmlCustomSaveCb = Option<fn(&MxmlArena, usize) -> Option<Vec<u8>>>;
/// Matches C `mxml_entity_cb_t` (`mxml.h:163`).
pub type MxmlEntityCb = Option<fn(&[u8]) -> c_int>;
/// Matches C `mxml_load_cb_t` (`mxml.h:166`).
pub type MxmlLoadCb = Option<fn(&MxmlArena, Option<usize>) -> MxmlType>;
/// Matches C `mxml_save_cb_t` (`mxml.h:169`).
///
/// The C callback returns a `const char *` into storage it owns and the writer
/// only reads it; returning owned bytes is the same contract without the
/// static buffer every implementation would otherwise need.
pub type MxmlSaveCb = Option<fn(&MxmlArena, usize, c_int) -> Option<Vec<u8>>>;
/// Matches C `mxml_sax_cb_t` (`mxml.h:172`).
pub type MxmlSaxCb = Option<fn(&mut MxmlArena, Option<usize>, MxmlSaxEvent, &mut dyn Any)>;

/// Matches C `mxml_attr_t` (`mxml.h:99`); C field order `name`, `value`.
///
/// `name` is `strdup`ed at every assignment and is never `NULL` once the
/// attribute exists; `value` is `NULL` for a valueless attribute.
pub struct MxmlAttr {
    pub name: Vec<u8>,
    pub value: Option<Vec<u8>>,
}

/// Matches C `mxml_element_t` (`mxml.h:105`); C field order `name`,
/// `num_attrs`, `attrs`.
///
/// `num_attrs` is kept beside the `Vec` because it is the count the C grows,
/// tests and writes back; `attrs` replaces the `realloc`ed array.
pub struct MxmlElement {
    pub name: Option<Vec<u8>>,
    pub num_attrs: c_int,
    pub attrs: Vec<MxmlAttr>,
}

/// Matches C `mxml_text_t` (`mxml.h:112`).
pub struct MxmlText {
    pub whitespace: c_int,
    pub string: Option<Vec<u8>>,
}

/// Matches C `mxml_custom_t` (`mxml.h:118`).
pub struct MxmlCustom {
    pub data: Option<Box<dyn Any>>,
    pub destroy: MxmlCustomDestroyCb,
}

/// Matches C `mxml_value_t` (`mxml.h:124`).
///
/// The C union is discriminated by the node's `type` field, which
/// `mxml_new` (`mxml-node.c:814`) is the only writer of in the whole library,
/// so the union is expressed as the tagged enum it already is.  `Ignore` is
/// the all-zero union of a node created with a type outside the six.
pub enum MxmlValue {
    Element(MxmlElement),
    Integer(c_int),
    Opaque(Option<Vec<u8>>),
    Real(f64),
    Text(MxmlText),
    Custom(MxmlCustom),
    Ignore,
}

/// Matches C `struct mxml_node_s` (`mxml.h:134`); C field order `type`,
/// `next`, `prev`, `parent`, `child`, `last_child`, `value`, `ref_count`,
/// `user_data`.
///
/// Every link is a slot index into the [`MxmlArena`] that allocated the node.
pub struct MxmlNode {
    pub type_: MxmlType,
    pub next: Option<usize>,
    pub prev: Option<usize>,
    pub parent: Option<usize>,
    pub child: Option<usize>,
    pub last_child: Option<usize>,
    pub value: MxmlValue,
    pub ref_count: c_int,
    pub user_data: Option<Box<dyn Any>>,
}

/// The heap that C `mxml_new` and `mxml_free` (`mxml-node.c:747`, `:690`)
/// allocate nodes from.
///
/// This type has no counterpart in `mxml.h`: it is the owner the C code gets
/// from `malloc`.  `nodes` is the slot table — a `None` slot is freed memory —
/// and `free` is the list of slots `mxml_free` released, which `mxml_new`
/// reuses the way `malloc` reuses a freed block.
pub struct MxmlArena {
    pub nodes: Vec<Option<MxmlNode>>,
    pub free: Vec<usize>,
}

impl MxmlArena {
    /// A heap with no nodes in it.
    pub fn new() -> MxmlArena {
        MxmlArena {
            nodes: Vec::new(),
            free: Vec::new(),
        }
    }

    /// The node in slot `index`, the arena's stand-in for `*node`.
    pub fn node(&self, index: usize) -> &MxmlNode {
        self.nodes[index]
            .as_ref()
            .expect("mxml: node index refers to a freed slot")
    }

    /// The node in slot `index`, the arena's stand-in for `*node`.
    pub fn node_mut(&mut self, index: usize) -> &mut MxmlNode {
        self.nodes[index]
            .as_mut()
            .expect("mxml: node index refers to a freed slot")
    }
}

/// Matches C `struct mxml_index_s` (`mxml.h:148`); C field order `attr`,
/// `num_nodes`, `alloc_nodes`, `cur_node`, `nodes`.
///
/// `alloc_nodes` stays because the C grows `nodes` in blocks of 64
/// (`mxml-index.c:340`) and that block size is observable through the field.
pub struct MxmlIndex {
    pub attr: Option<Vec<u8>>,
    pub num_nodes: c_int,
    pub alloc_nodes: c_int,
    pub cur_node: c_int,
    pub nodes: Vec<usize>,
}

pub type mxml_node_t = MxmlNode;
pub type mxml_index_t = MxmlIndex;
pub type mxml_type_t = MxmlType;
pub type mxml_save_cb_t = MxmlSaveCb;

pub use mxml_attr::*;
pub use mxml_entity::*;
pub use mxml_file::*;
pub use mxml_get::*;
pub use mxml_index::*;
pub use mxml_node::*;
pub use mxml_private::*;
pub use mxml_search::*;
pub use mxml_set::*;
pub use mxml_string::*;
