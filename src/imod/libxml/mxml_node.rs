//! Translation of `IMOD/libxml/mxml-node.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::*;
use core::ffi::{c_char, c_int, c_void};

/// Matches C `mxmlAdd` (`mxml-node.c:41`).
pub unsafe fn mxml_add(
    parent: *mut MxmlNode,
    where_: c_int,
    child: *mut MxmlNode,
    node: *mut MxmlNode,
) {
    /*
     * Range check input...
     */

    if parent.is_null() || node.is_null() {
        return;
    }

    /*
     * Remove the node from any existing parent...
     */

    if !(*node).parent.is_null() {
        mxml_remove(node);
    }

    /*
     * Reset pointers...
     */

    (*node).parent = parent;

    match where_ {
        MXML_ADD_BEFORE => {
            if child.is_null() || child == (*parent).child || (*child).parent != parent {
                /*
                 * Insert as first node under parent...
                 */

                (*node).next = (*parent).child;

                if !(*parent).child.is_null() {
                    (*(*parent).child).prev = node;
                } else {
                    (*parent).last_child = node;
                }

                (*parent).child = node;
            } else {
                /*
                 * Insert node before this child...
                 */

                (*node).next = child;
                (*node).prev = (*child).prev;

                if !(*child).prev.is_null() {
                    (*(*child).prev).next = node;
                } else {
                    (*parent).child = node;
                }

                (*child).prev = node;
            }
        }

        MXML_ADD_AFTER => {
            if child.is_null() || child == (*parent).last_child || (*child).parent != parent {
                /*
                 * Insert as last node under parent...
                 */

                (*node).parent = parent;
                (*node).prev = (*parent).last_child;

                if !(*parent).last_child.is_null() {
                    (*(*parent).last_child).next = node;
                } else {
                    (*parent).child = node;
                }

                (*parent).last_child = node;
            } else {
                /*
                 * Insert node after this child...
                 */

                (*node).prev = child;
                (*node).next = (*child).next;

                if !(*child).next.is_null() {
                    (*(*child).next).prev = node;
                } else {
                    (*parent).last_child = node;
                }

                (*child).next = node;
            }
        }

        _ => {}
    }
}

/// Matches C `mxmlDelete` (`mxml-node.c:165`).
pub unsafe fn mxml_delete(node: *mut MxmlNode) {
    let mut current: *mut MxmlNode;
    let mut next: *mut MxmlNode;

    /*
     * Range check input...
     */

    if node.is_null() {
        return;
    }

    /*
     * Remove the node from its parent, if any...
     */

    mxml_remove(node);

    /*
     * Delete children...
     */

    current = (*node).child;
    while !current.is_null() {
        /*
         * Get the next node...
         */

        next = (*current).child;
        if !next.is_null() {
            /*
             * Free parent nodes after child nodes have been freed...
             */

            (*current).child = core::ptr::null_mut();
            current = next;
            continue;
        }

        next = (*current).next;
        if next.is_null() {
            /*
             * Next node is the parent, which we'll free as needed...
             */

            next = (*current).parent;
            if next == node {
                next = core::ptr::null_mut();
            }
        }

        /*
         * Free child...
         */

        mxml_free(current);

        current = next;
    }

    /*
     * Then free the memory used by the parent node...
     */

    mxml_free(node);
}

/// Matches C `mxmlGetRefCount` (`mxml-node.c:230`).
pub unsafe fn mxml_get_ref_count(node: *mut MxmlNode) -> c_int {
    /*
     * Range check input...
     */

    if node.is_null() {
        return 0;
    }

    /*
     * Return the reference count...
     */

    (*node).ref_count
}

/// Matches C `mxmlNewCDATA` (`mxml-node.c:259`).
pub unsafe fn mxml_new_cdata(parent: *mut MxmlNode, data: *const c_char) -> *mut MxmlNode {
    let node: *mut MxmlNode;

    /*
     * Range check input...
     */

    if data.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Create the node and set the name value...
     */

    node = mxml_new(parent, MXML_ELEMENT);
    if !node.is_null() {
        (*node).value.element.name = _mxml_strdupf(c"![CDATA[%s]]".as_ptr(), data as *mut c_void);
    }

    node
}

/// Matches C `mxmlNewCustom` (`mxml-node.c:299`).
pub unsafe fn mxml_new_custom(
    parent: *mut MxmlNode,
    data: *mut c_void,
    destroy: MxmlCustomDestroyCb,
) -> *mut MxmlNode {
    let node: *mut MxmlNode;

    /*
     * Create the node and set the value...
     */

    node = mxml_new(parent, MXML_CUSTOM);
    if !node.is_null() {
        (*node).value.custom.data = data;
        (*node).value.custom.destroy = destroy;
    }

    node
}

/// Matches C `mxmlNewElement` (`mxml-node.c:335`).
pub unsafe fn mxml_new_element(parent: *mut MxmlNode, name: *const c_char) -> *mut MxmlNode {
    let node: *mut MxmlNode;

    /*
     * Range check input...
     */

    if name.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Create the node and set the element name...
     */

    node = mxml_new(parent, MXML_ELEMENT);
    if !node.is_null() {
        (*node).value.element.name = libc::strdup(name);
    }

    node
}

/// Matches C `mxmlNewInteger` (`mxml-node.c:373`).
pub unsafe fn mxml_new_integer(parent: *mut MxmlNode, integer: c_int) -> *mut MxmlNode {
    let node: *mut MxmlNode;

    /*
     * Create the node and set the element name...
     */

    node = mxml_new(parent, MXML_INTEGER);
    if !node.is_null() {
        (*node).value.integer = integer;
    }

    node
}

/// Matches C `mxmlNewOpaque` (`mxml-node.c:404`).
pub unsafe fn mxml_new_opaque(parent: *mut MxmlNode, opaque: *const c_char) -> *mut MxmlNode {
    let node: *mut MxmlNode;

    /*
     * Range check input...
     */

    if opaque.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Create the node and set the element name...
     */

    node = mxml_new(parent, MXML_OPAQUE);
    if !node.is_null() {
        (*node).value.opaque = libc::strdup(opaque);
    }

    node
}

/// Matches C `mxmlNewReal` (`mxml-node.c:442`).
pub unsafe fn mxml_new_real(parent: *mut MxmlNode, real: f64) -> *mut MxmlNode {
    let node: *mut MxmlNode;

    /*
     * Create the node and set the element name...
     */

    node = mxml_new(parent, MXML_REAL);
    if !node.is_null() {
        (*node).value.real = real;
    }

    node
}

/// Matches C `mxmlNewText` (`mxml-node.c:477`).
pub unsafe fn mxml_new_text(
    parent: *mut MxmlNode,
    whitespace: c_int,
    string: *const c_char,
) -> *mut MxmlNode {
    let node: *mut MxmlNode;

    /*
     * Range check input...
     */

    if string.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Create the node and set the text value...
     */

    node = mxml_new(parent, MXML_TEXT);
    if !node.is_null() {
        (*node).value.text.whitespace = whitespace;
        (*node).value.text.string = libc::strdup(string);
    }

    node
}

/// Matches C `mxmlNewTextf` (`mxml-node.c:521`).
///
/// The C function is `mxmlNewTextf(parent, whitespace, format, ...)`.  Stable
/// Rust cannot define a C-variadic function, so the `va_list` that the C body
/// starts is taken as a parameter and forwarded to `_mxml_vstrdupf` unchanged.
pub unsafe fn mxml_new_textf(
    parent: *mut MxmlNode,
    whitespace: c_int,
    format: *const c_char,
    ap: *mut c_void,
) -> *mut MxmlNode {
    let node: *mut MxmlNode;

    /*
     * Range check input...
     */

    if format.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Create the node and set the text value...
     */

    node = mxml_new(parent, MXML_TEXT);
    if !node.is_null() {
        (*node).value.text.whitespace = whitespace;
        (*node).value.text.string = _mxml_vstrdupf(format, ap);
    }

    node
}

/// Matches C `mxmlRemove` (`mxml-node.c:565`).
pub unsafe fn mxml_remove(node: *mut MxmlNode) {
    /*
     * Range check input...
     */

    if node.is_null() || (*node).parent.is_null() {
        return;
    }

    /*
     * Remove from parent...
     */

    if !(*node).prev.is_null() {
        (*(*node).prev).next = (*node).next;
    } else {
        (*(*node).parent).child = (*node).next;
    }

    if !(*node).next.is_null() {
        (*(*node).next).prev = (*node).prev;
    } else {
        (*(*node).parent).last_child = (*node).prev;
    }

    (*node).parent = core::ptr::null_mut();
    (*node).prev = core::ptr::null_mut();
    (*node).next = core::ptr::null_mut();
}

/// Matches C `mxmlNewXML` (`mxml-node.c:629`).
pub unsafe fn mxml_new_xml(version: *const c_char) -> *mut MxmlNode {
    let mut element: [c_char; 1024] = [0; 1024];

    libc::snprintf(
        element.as_mut_ptr(),
        core::mem::size_of::<[c_char; 1024]>(),
        c"?xml version=\"%s\" encoding=\"utf-8\"?".as_ptr(),
        if !version.is_null() {
            version
        } else {
            c"1.0".as_ptr()
        },
    );

    mxml_new_element(core::ptr::null_mut(), element.as_ptr())
}

/// Matches C `mxmlRelease` (`mxml-node.c:651`).
pub unsafe fn mxml_release(node: *mut MxmlNode) -> c_int {
    if !node.is_null() {
        (*node).ref_count -= 1;
        if (*node).ref_count <= 0 {
            mxml_delete(node);
            0
        } else {
            (*node).ref_count
        }
    } else {
        -1
    }
}

/// Matches C `mxmlRetain` (`mxml-node.c:675`).
pub unsafe fn mxml_retain(node: *mut MxmlNode) -> c_int {
    if !node.is_null() {
        (*node).ref_count += 1;
        (*node).ref_count
    } else {
        -1
    }
}

/// Matches C static `mxml_free` (`mxml-node.c:690`).
pub unsafe fn mxml_free(node: *mut MxmlNode) {
    let mut i: c_int;

    match (*node).type_ {
        MXML_ELEMENT => {
            if !(*node).value.element.name.is_null() {
                libc::free((*node).value.element.name as *mut c_void);
            }

            if (*node).value.element.num_attrs != 0 {
                i = 0;
                while i < (*node).value.element.num_attrs {
                    if !(*(*node).value.element.attrs.add(i as usize))
                        .name
                        .is_null()
                    {
                        libc::free(
                            (*(*node).value.element.attrs.add(i as usize)).name as *mut c_void,
                        );
                    }
                    if !(*(*node).value.element.attrs.add(i as usize))
                        .value
                        .is_null()
                    {
                        libc::free(
                            (*(*node).value.element.attrs.add(i as usize)).value as *mut c_void,
                        );
                    }
                    i += 1;
                }

                libc::free((*node).value.element.attrs as *mut c_void);
            }
        }

        MXML_INTEGER => { /* Nothing to do */ }

        MXML_OPAQUE => {
            if !(*node).value.opaque.is_null() {
                libc::free((*node).value.opaque as *mut c_void);
            }
        }

        MXML_REAL => { /* Nothing to do */ }

        MXML_TEXT => {
            if !(*node).value.text.string.is_null() {
                libc::free((*node).value.text.string as *mut c_void);
            }
        }

        MXML_CUSTOM => {
            if !(*node).value.custom.data.is_null() {
                if let Some(f) = (*node).value.custom.destroy {
                    f((*node).value.custom.data);
                }
            }
        }

        _ => {}
    }

    /*
     * Free this node...
     */

    libc::free(node as *mut c_void);
}

/// Matches C static `mxml_new` (`mxml-node.c:747`).
pub unsafe fn mxml_new(parent: *mut MxmlNode, type_: MxmlType) -> *mut MxmlNode {
    let node: *mut MxmlNode;

    /*
     * Allocate memory for the node...
     */

    node = libc::calloc(1, core::mem::size_of::<MxmlNode>()) as *mut MxmlNode;
    if node.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Set the node type...
     */

    (*node).type_ = type_;
    (*node).ref_count = 1;

    /*
     * Add to the parent if present...
     */

    if !parent.is_null() {
        mxml_add(parent, MXML_ADD_AFTER, MXML_ADD_TO_PARENT, node);
    }

    /*
     * Return the new node...
     */

    node
}
