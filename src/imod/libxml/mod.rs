//! Translation of the vendored Mini-XML 2.10 library (`IMOD/libxml/`) and its
//! public header `IMOD/include/mxml.h`.
//!
//! The header's constants, typedefs and structures live here; each `.c` file is
//! mirrored by the module of the same name.  Every structure keeps the C field
//! order and layout so the modules can be as C-shaped as the original.
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

use core::ffi::{c_char, c_int, c_void};

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

/// Matches C `MXML_NO_PARENT` (`mxml.h:54`).
pub const MXML_NO_PARENT: *mut MxmlNode = core::ptr::null_mut();

pub const MXML_DESCEND: c_int = 1;
pub const MXML_NO_DESCEND: c_int = 0;
pub const MXML_DESCEND_FIRST: c_int = -1;

pub const MXML_WS_BEFORE_OPEN: c_int = 0;
pub const MXML_WS_AFTER_OPEN: c_int = 1;
pub const MXML_WS_BEFORE_CLOSE: c_int = 2;
pub const MXML_WS_AFTER_CLOSE: c_int = 3;

pub const MXML_ADD_BEFORE: c_int = 0;
pub const MXML_ADD_AFTER: c_int = 1;
/// Matches C `MXML_ADD_TO_PARENT` (`mxml.h:67`).
pub const MXML_ADD_TO_PARENT: *mut MxmlNode = core::ptr::null_mut();

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
pub type MxmlCustomDestroyCb = Option<unsafe extern "C" fn(*mut c_void)>;
/// Matches C `mxml_error_cb_t` (`mxml.h:96`).
pub type MxmlErrorCb = Option<unsafe extern "C" fn(*const c_char)>;
/// Matches C `mxml_custom_load_cb_t` (`mxml.h:157`).
pub type MxmlCustomLoadCb = Option<unsafe extern "C" fn(*mut MxmlNode, *const c_char) -> c_int>;
/// Matches C `mxml_custom_save_cb_t` (`mxml.h:160`).
pub type MxmlCustomSaveCb = Option<unsafe extern "C" fn(*mut MxmlNode) -> *mut c_char>;
/// Matches C `mxml_entity_cb_t` (`mxml.h:163`).
pub type MxmlEntityCb = Option<unsafe extern "C" fn(*const c_char) -> c_int>;
/// Matches C `mxml_load_cb_t` (`mxml.h:166`).
pub type MxmlLoadCb = Option<unsafe extern "C" fn(*mut MxmlNode) -> MxmlType>;
/// Matches C `mxml_save_cb_t` (`mxml.h:169`).
pub type MxmlSaveCb = Option<unsafe extern "C" fn(*mut MxmlNode, c_int) -> *const c_char>;
/// Matches C `mxml_sax_cb_t` (`mxml.h:172`).
pub type MxmlSaxCb = Option<unsafe extern "C" fn(*mut MxmlNode, MxmlSaxEvent, *mut c_void)>;

/// Matches C `mxml_attr_t` (`mxml.h:99`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MxmlAttr {
    pub name: *mut c_char,
    pub value: *mut c_char,
}

/// Matches C `mxml_element_t` (`mxml.h:105`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MxmlElement {
    pub name: *mut c_char,
    pub num_attrs: c_int,
    pub attrs: *mut MxmlAttr,
}

/// Matches C `mxml_text_t` (`mxml.h:112`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MxmlText {
    pub whitespace: c_int,
    pub string: *mut c_char,
}

/// Matches C `mxml_custom_t` (`mxml.h:118`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MxmlCustom {
    pub data: *mut c_void,
    pub destroy: MxmlCustomDestroyCb,
}

/// Matches C `mxml_value_t` (`mxml.h:124`).
#[repr(C)]
pub union MxmlValue {
    pub element: MxmlElement,
    pub integer: c_int,
    pub opaque: *mut c_char,
    pub real: f64,
    pub text: MxmlText,
    pub custom: MxmlCustom,
}

/// Matches C `struct mxml_node_s` (`mxml.h:134`).
#[repr(C)]
pub struct MxmlNode {
    pub type_: MxmlType,
    pub next: *mut MxmlNode,
    pub prev: *mut MxmlNode,
    pub parent: *mut MxmlNode,
    pub child: *mut MxmlNode,
    pub last_child: *mut MxmlNode,
    pub value: MxmlValue,
    pub ref_count: c_int,
    pub user_data: *mut c_void,
}

/// Matches C `struct mxml_index_s` (`mxml.h:148`).
#[repr(C)]
pub struct MxmlIndex {
    pub attr: *mut c_char,
    pub num_nodes: c_int,
    pub alloc_nodes: c_int,
    pub cur_node: c_int,
    pub nodes: *mut *mut MxmlNode,
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
