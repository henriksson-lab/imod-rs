//! Direct translation surface for vendored Mini-XML 2.10.
pub mod mxml_attr;
pub mod mxml_entity;
pub mod mxml_file;
pub mod mxml_get;
pub mod mxml_index;
pub mod mxml_node;
pub mod mxml_private;
pub mod mxml_search;
pub mod mxml_set;

use core::ffi::{c_char, c_int, c_void};

pub const MXML_IGNORE: c_int = -1;
pub const MXML_ELEMENT: c_int = 0;
pub const MXML_INTEGER: c_int = 1;
pub const MXML_OPAQUE: c_int = 2;
pub const MXML_REAL: c_int = 3;
pub const MXML_TEXT: c_int = 4;
pub const MXML_CUSTOM: c_int = 5;
pub const MXML_DESCEND: c_int = 1;
pub const MXML_NO_DESCEND: c_int = 0;
pub const MXML_DESCEND_FIRST: c_int = -1;
pub const MXML_WS_BEFORE_OPEN: c_int = 0;
pub const MXML_WS_AFTER_OPEN: c_int = 1;
pub const MXML_WS_BEFORE_CLOSE: c_int = 2;
pub const MXML_WS_AFTER_CLOSE: c_int = 3;
pub const MXML_ADD_BEFORE: c_int = 0;
pub const MXML_ADD_AFTER: c_int = 1;
pub type MxmlType = c_int;
pub type MxmlLoadCb = Option<unsafe extern "C" fn(*mut MxmlNode) -> MxmlType>;
pub type MxmlSaveCb = Option<unsafe extern "C" fn(*mut MxmlNode, c_int) -> *const c_char>;
pub type MxmlCustomDestroyCb = Option<unsafe extern "C" fn(*mut c_void)>;
pub type MxmlEntityCb = Option<unsafe extern "C" fn(*const c_char) -> c_int>;
pub type MxmlErrorCb = Option<unsafe extern "C" fn(*const c_char)>;
pub type MxmlCustomLoadCb = Option<unsafe extern "C" fn(*mut MxmlNode, *const c_char) -> c_int>;
pub type MxmlCustomSaveCb = Option<unsafe extern "C" fn(*mut MxmlNode) -> *mut c_char>;
pub type MxmlSaxCb = Option<unsafe extern "C" fn(*mut MxmlNode, c_int, *mut c_void)>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MxmlAttr {
    pub name: *mut c_char,
    pub value: *mut c_char,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MxmlElement {
    pub name: *mut c_char,
    pub num_attrs: c_int,
    pub attrs: *mut MxmlAttr,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MxmlText {
    pub whitespace: c_int,
    pub string: *mut c_char,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MxmlCustom {
    pub data: *mut c_void,
    pub destroy: MxmlCustomDestroyCb,
}
#[repr(C)]
pub union MxmlValue {
    pub element: MxmlElement,
    pub integer: c_int,
    pub opaque: *mut c_char,
    pub real: f64,
    pub text: MxmlText,
    pub custom: MxmlCustom,
}
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
