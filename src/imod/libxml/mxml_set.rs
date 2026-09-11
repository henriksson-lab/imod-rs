//! Translation of `IMOD/libxml/mxml-set.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::*;
use core::ffi::{c_char, c_int, c_void};

/// Matches C `mxmlSetCDATA` (`mxml-set.c:32`).
pub unsafe fn mxml_set_cdata(mut node: *mut MxmlNode, data: *const c_char) -> c_int {
    /*
     * Range check input...
     */

    if !node.is_null()
        && (*node).type_ == MXML_ELEMENT
        && libc::strncmp((*node).value.element.name, c"![CDATA[".as_ptr(), 8) != 0
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_ELEMENT
        && libc::strncmp((*(*node).child).value.element.name, c"![CDATA[".as_ptr(), 8) == 0
    {
        node = (*node).child;
    }

    if node.is_null()
        || (*node).type_ != MXML_ELEMENT
        || data.is_null()
        || libc::strncmp((*node).value.element.name, c"![CDATA[".as_ptr(), 8) != 0
    {
        return -1;
    }

    /*
     * Free any old element value and set the new value...
     */

    if !(*node).value.element.name.is_null() {
        libc::free((*node).value.element.name as *mut c_void);
    }

    (*node).value.element.name = _mxml_strdupf(c"![CDATA[%s]]".as_ptr(), data as *mut c_void);

    0
}

/// Matches C `mxmlSetCustom` (`mxml-set.c:70`).
pub unsafe fn mxml_set_custom(
    mut node: *mut MxmlNode,
    data: *mut c_void,
    destroy: MxmlCustomDestroyCb,
) -> c_int {
    /*
     * Range check input...
     */

    if !node.is_null()
        && (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_CUSTOM
    {
        node = (*node).child;
    }

    if node.is_null() || (*node).type_ != MXML_CUSTOM {
        return -1;
    }

    /*
     * Free any old element value and set the new value...
     */

    if !(*node).value.custom.data.is_null() {
        if let Some(f) = (*node).value.custom.destroy {
            f((*node).value.custom.data);
        }
    }

    (*node).value.custom.data = data;
    (*node).value.custom.destroy = destroy;

    0
}

/// Matches C `mxmlSetElement` (`mxml-set.c:104`).
pub unsafe fn mxml_set_element(node: *mut MxmlNode, name: *const c_char) -> c_int {
    /*
     * Range check input...
     */

    if node.is_null() || (*node).type_ != MXML_ELEMENT || name.is_null() {
        return -1;
    }

    /*
     * Free any old element value and set the new value...
     */

    if !(*node).value.element.name.is_null() {
        libc::free((*node).value.element.name as *mut c_void);
    }

    (*node).value.element.name = libc::strdup(name);

    0
}

/// Matches C `mxmlSetInteger` (`mxml-set.c:130`).
pub unsafe fn mxml_set_integer(mut node: *mut MxmlNode, integer: c_int) -> c_int {
    /*
     * Range check input...
     */

    if !node.is_null()
        && (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_INTEGER
    {
        node = (*node).child;
    }

    if node.is_null() || (*node).type_ != MXML_INTEGER {
        return -1;
    }

    /*
     * Set the new value and return...
     */

    (*node).value.integer = integer;

    0
}

/// Matches C `mxmlSetOpaque` (`mxml-set.c:158`).
pub unsafe fn mxml_set_opaque(mut node: *mut MxmlNode, opaque: *const c_char) -> c_int {
    /*
     * Range check input...
     */

    if !node.is_null()
        && (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_OPAQUE
    {
        node = (*node).child;
    }

    if node.is_null() || (*node).type_ != MXML_OPAQUE || opaque.is_null() {
        return -1;
    }

    /*
     * Free any old opaque value and set the new value...
     */

    if !(*node).value.opaque.is_null() {
        libc::free((*node).value.opaque as *mut c_void);
    }

    (*node).value.opaque = libc::strdup(opaque);

    0
}

/// Matches C `mxmlSetReal` (`mxml-set.c:190`).
pub unsafe fn mxml_set_real(mut node: *mut MxmlNode, real: f64) -> c_int {
    /*
     * Range check input...
     */

    if !node.is_null()
        && (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_REAL
    {
        node = (*node).child;
    }

    if node.is_null() || (*node).type_ != MXML_REAL {
        return -1;
    }

    /*
     * Set the new value and return...
     */

    (*node).value.real = real;

    0
}

/// Matches C `mxmlSetText` (`mxml-set.c:218`).
pub unsafe fn mxml_set_text(
    mut node: *mut MxmlNode,
    whitespace: c_int,
    string: *const c_char,
) -> c_int {
    /*
     * Range check input...
     */

    if !node.is_null()
        && (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_TEXT
    {
        node = (*node).child;
    }

    if node.is_null() || (*node).type_ != MXML_TEXT || string.is_null() {
        return -1;
    }

    /*
     * Free any old string value and set the new value...
     */

    if !(*node).value.text.string.is_null() {
        libc::free((*node).value.text.string as *mut c_void);
    }

    (*node).value.text.whitespace = whitespace;
    (*node).value.text.string = libc::strdup(string);

    0
}

/// Matches C `mxmlSetTextf` (`mxml-set.c:255`).
///
/// The C function is `mxmlSetTextf(node, whitespace, format, ...)` and hands
/// the started `va_list` to `_mxml_strdupf`, which expects a variable argument
/// list rather than a `va_list` — the mismatch is in the vendored source.  The
/// shape is preserved here with the single explicit argument that stable Rust
/// allows; see `_mxml_strdupf`.
pub unsafe fn mxml_set_textf(
    mut node: *mut MxmlNode,
    whitespace: c_int,
    format: *const c_char,
    arg: *mut c_void,
) -> c_int {
    /*
     * Range check input...
     */

    if !node.is_null()
        && (*node).type_ == MXML_ELEMENT
        && !(*node).child.is_null()
        && (*(*node).child).type_ == MXML_TEXT
    {
        node = (*node).child;
    }

    if node.is_null() || (*node).type_ != MXML_TEXT || format.is_null() {
        return -1;
    }

    /*
     * Free any old string value and set the new value...
     */

    if !(*node).value.text.string.is_null() {
        libc::free((*node).value.text.string as *mut c_void);
    }

    (*node).value.text.whitespace = whitespace;
    (*node).value.text.string = _mxml_strdupf(format, arg);

    0
}

/// Matches C `mxmlSetUserData` (`mxml-set.c:299`).
pub unsafe fn mxml_set_user_data(node: *mut MxmlNode, data: *mut c_void) -> c_int {
    /*
     * Range check input...
     */

    if node.is_null() {
        return -1;
    }

    /*
     * Set the user data pointer and return...
     */

    (*node).user_data = data;

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `mxmlSetXxx` first descends from an element into a matching child
    /// (`mxml-set.c:141` and friends), so setting text on the element that owns
    /// a text node succeeds and rewrites the child.  The native
    /// `libimxml.so` returns 0 and reads back "bye".
    #[test]
    fn set_text_descends_from_the_element_into_its_child() {
        unsafe {
            let tree =
                mxml_load_string(core::ptr::null_mut(), c"<root>hello</root>".as_ptr(), None);
            assert!(!tree.is_null());
            assert_eq!((*tree).type_, MXML_ELEMENT);
            assert_eq!(mxml_set_text(tree, 0, c"bye".as_ptr()), 0);
            let mut ws: c_int = 0;
            assert_eq!(
                std::ffi::CStr::from_ptr(mxml_get_text(tree, &raw mut ws)).to_bytes(),
                b"bye"
            );
            mxml_delete(tree);
        }
    }
}
