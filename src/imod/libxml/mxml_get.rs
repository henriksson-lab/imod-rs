//! Translation of `IMOD/libxml/mxml-get.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::*;
use core::ffi::{c_char, c_int, c_void};

/// Matches C `mxmlGetCDATA` (`mxml-get.c:33`).
pub unsafe fn mxml_get_cdata(node: *mut MxmlNode) -> *const c_char {
    /*
     * Range check input...
     */

    if node.is_null()
        || (*node).type_ != MXML_ELEMENT
        || libc::strncmp((*node).value.element.name, c"![CDATA[".as_ptr(), 8) != 0
    {
        return core::ptr::null();
    }

    /*
     * Return the text following the CDATA declaration...
     */

    (*node).value.element.name.add(8)
}

/// Matches C `mxmlGetCustom` (`mxml-get.c:59`).
pub unsafe fn mxml_get_custom(node: *mut MxmlNode) -> *const c_void {
    /*
     * Range check input...
     */

    if node.is_null() {
        return core::ptr::null();
    }

    /*
     * Return the custom value...
     */

    if (*node).type_ == MXML_CUSTOM {
        (*node).value.custom.data
    } else if (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_CUSTOM
    {
        (*(*node).child).value.custom.data
    } else {
        core::ptr::null()
    }
}

/// Matches C `mxmlGetElement` (`mxml-get.c:88`).
pub unsafe fn mxml_get_element(node: *mut MxmlNode) -> *const c_char {
    /*
     * Range check input...
     */

    if node.is_null() || (*node).type_ != MXML_ELEMENT {
        return core::ptr::null();
    }

    /*
     * Return the element name...
     */

    (*node).value.element.name
}

/// Matches C `mxmlGetFirstChild` (`mxml-get.c:111`).
pub unsafe fn mxml_get_first_child(node: *mut MxmlNode) -> *mut MxmlNode {
    /*
     * Range check input...
     */

    if node.is_null() || (*node).type_ != MXML_ELEMENT {
        return core::ptr::null_mut();
    }

    /*
     * Return the first child node...
     */

    (*node).child
}

/// Matches C `mxmlGetInteger` (`mxml-get.c:136`).
pub unsafe fn mxml_get_integer(node: *mut MxmlNode) -> c_int {
    /*
     * Range check input...
     */

    if node.is_null() {
        return 0;
    }

    /*
     * Return the integer value...
     */

    if (*node).type_ == MXML_INTEGER {
        (*node).value.integer
    } else if (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_INTEGER
    {
        (*(*node).child).value.integer
    } else {
        0
    }
}

/// Matches C `mxmlGetLastChild` (`mxml-get.c:165`).
pub unsafe fn mxml_get_last_child(node: *mut MxmlNode) -> *mut MxmlNode {
    /*
     * Range check input...
     */

    if node.is_null() || (*node).type_ != MXML_ELEMENT {
        return core::ptr::null_mut();
    }

    /*
     * Return the node type...
     */

    (*node).last_child
}

/// Matches C `mxmlGetNextSibling` (`mxml-get.c:186`).
pub unsafe fn mxml_get_next_sibling(node: *mut MxmlNode) -> *mut MxmlNode {
    /*
     * Range check input...
     */

    if node.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Return the node type...
     */

    (*node).next
}

/// Matches C `mxmlGetOpaque` (`mxml-get.c:210`).
pub unsafe fn mxml_get_opaque(node: *mut MxmlNode) -> *const c_char {
    /*
     * Range check input...
     */

    if node.is_null() {
        return core::ptr::null();
    }

    /*
     * Return the opaque value...
     */

    if (*node).type_ == MXML_OPAQUE {
        (*node).value.opaque
    } else if (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_OPAQUE
    {
        (*(*node).child).value.opaque
    } else {
        core::ptr::null()
    }
}

/// Matches C `mxmlGetParent` (`mxml-get.c:239`).
pub unsafe fn mxml_get_parent(node: *mut MxmlNode) -> *mut MxmlNode {
    /*
     * Range check input...
     */

    if node.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Return the parent node...
     */

    (*node).parent
}

/// Matches C `mxmlGetPrevSibling` (`mxml-get.c:262`).
pub unsafe fn mxml_get_prev_sibling(node: *mut MxmlNode) -> *mut MxmlNode {
    /*
     * Range check input...
     */

    if node.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Return the previous sibling node...
     */

    (*node).prev
}

/// Matches C `mxmlGetReal` (`mxml-get.c:287`).
pub unsafe fn mxml_get_real(node: *mut MxmlNode) -> f64 {
    /*
     * Range check input...
     */

    if node.is_null() {
        return 0.0;
    }

    /*
     * Return the real value...
     */

    if (*node).type_ == MXML_REAL {
        (*node).value.real
    } else if (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_REAL
    {
        (*(*node).child).value.real
    } else {
        0.0
    }
}

/// Matches C `mxmlGetText` (`mxml-get.c:320`).
pub unsafe fn mxml_get_text(node: *mut MxmlNode, whitespace: *mut c_int) -> *const c_char {
    /*
     * Range check input...
     */

    if node.is_null() {
        if !whitespace.is_null() {
            *whitespace = 0;
        }

        return core::ptr::null();
    }

    /*
     * Return the text value...
     */

    if (*node).type_ == MXML_TEXT {
        if !whitespace.is_null() {
            *whitespace = (*node).value.text.whitespace;
        }

        (*node).value.text.string
    } else if (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_TEXT
    {
        if !whitespace.is_null() {
            *whitespace = (*(*node).child).value.text.whitespace;
        }

        (*(*node).child).value.text.string
    } else {
        if !whitespace.is_null() {
            *whitespace = 0;
        }

        core::ptr::null()
    }
}

/// Matches C `mxmlGetType` (`mxml-get.c:369`).
pub unsafe fn mxml_get_type(node: *mut MxmlNode) -> MxmlType {
    /*
     * Range check input...
     */

    if node.is_null() {
        return MXML_IGNORE;
    }

    /*
     * Return the node type...
     */

    (*node).type_
}

/// Matches C `mxmlGetUserData` (`mxml-get.c:390`).
pub unsafe fn mxml_get_user_data(node: *mut MxmlNode) -> *mut c_void {
    /*
     * Range check input...
     */

    if node.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Return the user data pointer...
     */

    (*node).user_data
}
