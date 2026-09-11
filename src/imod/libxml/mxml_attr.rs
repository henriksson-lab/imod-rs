//! Translation of `IMOD/libxml/mxml-attr.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::*;
use core::ffi::{c_char, c_int, c_void};

/// Matches C `mxmlElementDeleteAttr` (`mxml-attr.c:38`).
pub unsafe fn mxml_element_delete_attr(node: *mut MxmlNode, name: *const c_char) {
    let mut i: c_int;
    let mut attr: *mut MxmlAttr;

    /*
     * Range check input...
     */

    if node.is_null() || (*node).type_ != MXML_ELEMENT || name.is_null() {
        return;
    }

    /*
     * Look for the attribute...
     */

    i = (*node).value.element.num_attrs;
    attr = (*node).value.element.attrs;
    while i > 0 {
        if libc::strcmp((*attr).name, name) == 0 {
            /*
             * Delete this attribute...
             */

            libc::free((*attr).name as *mut c_void);
            libc::free((*attr).value as *mut c_void);

            i -= 1;
            if i > 0 {
                libc::memmove(
                    attr as *mut c_void,
                    attr.add(1) as *const c_void,
                    i as usize * core::mem::size_of::<MxmlAttr>(),
                );
            }

            (*node).value.element.num_attrs -= 1;

            if (*node).value.element.num_attrs == 0 {
                libc::free((*node).value.element.attrs as *mut c_void);
            }
            return;
        }

        i -= 1;
        attr = attr.add(1);
    }
}

/// Matches C `mxmlElementGetAttr` (`mxml-attr.c:96`).
pub unsafe fn mxml_element_get_attr(node: *mut MxmlNode, name: *const c_char) -> *const c_char {
    let mut i: c_int;
    let mut attr: *mut MxmlAttr;

    /*
     * Range check input...
     */

    if node.is_null() || (*node).type_ != MXML_ELEMENT || name.is_null() {
        return core::ptr::null();
    }

    /*
     * Look for the attribute...
     */

    i = (*node).value.element.num_attrs;
    attr = (*node).value.element.attrs;
    while i > 0 {
        if libc::strcmp((*attr).name, name) == 0 {
            return (*attr).value;
        }

        i -= 1;
        attr = attr.add(1);
    }

    /*
     * Didn't find attribute, so return NULL...
     */

    core::ptr::null()
}

/// Matches C `mxmlElementSetAttr` (`mxml-attr.c:154`).
pub unsafe fn mxml_element_set_attr(
    node: *mut MxmlNode,
    name: *const c_char,
    value: *const c_char,
) {
    let valuec: *mut c_char;

    /*
     * Range check input...
     */

    if node.is_null() || (*node).type_ != MXML_ELEMENT || name.is_null() {
        return;
    }

    if !value.is_null() {
        valuec = libc::strdup(value);
    } else {
        valuec = core::ptr::null_mut();
    }

    if mxml_set_attr(node, name, valuec) != 0 {
        libc::free(valuec as *mut c_void);
    }
}

/// Matches C `mxmlElementSetAttrf` (`mxml-attr.c:194`).
///
/// The C function is `mxmlElementSetAttrf(node, name, format, ...)`.  Stable
/// Rust cannot define a C-variadic function, so the single variable argument is
/// taken explicitly; `_mxml_strdupf` carries the same restriction.  The only
/// call in IMOD is `mxmlElementSetAttrf(node, name, "%d", value)` in
/// `libcfshr/mxmlwrap.c:375`.
pub unsafe fn mxml_element_set_attrf(
    node: *mut MxmlNode,
    name: *const c_char,
    format: *const c_char,
    arg: *mut c_void,
) {
    let value: *mut c_char;

    /*
     * Range check input...
     */

    if node.is_null() || (*node).type_ != MXML_ELEMENT || name.is_null() || format.is_null() {
        return;
    }

    /*
     * Format the value...
     */

    value = _mxml_strdupf(format, arg);

    if value.is_null() {
        let mut msg: [c_char; 1024] = [0; 1024];
        libc::snprintf(
            msg.as_mut_ptr(),
            core::mem::size_of::<[c_char; 1024]>(),
            c"Unable to allocate memory for attribute '%s' in element %s!".as_ptr(),
            name,
            (*node).value.element.name,
        );
        mxml_error(msg.as_ptr());
    } else if mxml_set_attr(node, name, value) != 0 {
        libc::free(value as *mut c_void);
    }
}

/// Matches C static `mxml_set_attr` (`mxml-attr.c:234`).
pub unsafe fn mxml_set_attr(node: *mut MxmlNode, name: *const c_char, value: *mut c_char) -> c_int {
    let mut i: c_int;
    let mut attr: *mut MxmlAttr;

    /*
     * Look for the attribute...
     */

    i = (*node).value.element.num_attrs;
    attr = (*node).value.element.attrs;
    while i > 0 {
        if libc::strcmp((*attr).name, name) == 0 {
            /*
             * Free the old value as needed...
             */

            if !(*attr).value.is_null() {
                libc::free((*attr).value as *mut c_void);
            }

            (*attr).value = value;

            return 0;
        }

        i -= 1;
        attr = attr.add(1);
    }

    /*
     * Add a new attribute...
     */

    if (*node).value.element.num_attrs == 0 {
        attr = libc::malloc(core::mem::size_of::<MxmlAttr>()) as *mut MxmlAttr;
    } else {
        attr = libc::realloc(
            (*node).value.element.attrs as *mut c_void,
            ((*node).value.element.num_attrs + 1) as usize * core::mem::size_of::<MxmlAttr>(),
        ) as *mut MxmlAttr;
    }

    if attr.is_null() {
        let mut msg: [c_char; 1024] = [0; 1024];
        libc::snprintf(
            msg.as_mut_ptr(),
            core::mem::size_of::<[c_char; 1024]>(),
            c"Unable to allocate memory for attribute '%s' in element %s!".as_ptr(),
            name,
            (*node).value.element.name,
        );
        mxml_error(msg.as_ptr());
        return -1;
    }

    (*node).value.element.attrs = attr;
    attr = attr.add((*node).value.element.num_attrs as usize);

    (*attr).name = libc::strdup(name);
    if (*attr).name.is_null() {
        let mut msg: [c_char; 1024] = [0; 1024];
        libc::snprintf(
            msg.as_mut_ptr(),
            core::mem::size_of::<[c_char; 1024]>(),
            c"Unable to allocate memory for attribute '%s' in element %s!".as_ptr(),
            name,
            (*node).value.element.name,
        );
        mxml_error(msg.as_ptr());
        return -1;
    }

    (*attr).value = value;

    (*node).value.element.num_attrs += 1;

    0
}
