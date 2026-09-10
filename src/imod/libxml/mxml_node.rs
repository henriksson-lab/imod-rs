use super::*;
use core::ffi::{c_char, c_int, c_void};
use std::ffi::{CStr, CString};

unsafe fn dup(s: *const c_char) -> *mut c_char {
    if s.is_null() {
        core::ptr::null_mut()
    } else {
        unsafe { libc::strdup(s) }
    }
}
unsafe fn mxml_free(node: *mut MxmlNode) {
    unsafe {
        match (*node).type_ {
            MXML_ELEMENT => {
                let e = (*node).value.element;
                libc::free(e.name.cast());
                for i in 0..e.num_attrs {
                    let a = e.attrs.add(i as usize);
                    libc::free((*a).name.cast());
                    libc::free((*a).value.cast());
                }
                libc::free(e.attrs.cast());
            }
            MXML_OPAQUE => libc::free((*node).value.opaque.cast()),
            MXML_TEXT => libc::free((*node).value.text.string.cast()),
            MXML_CUSTOM => {
                let c = (*node).value.custom;
                if let Some(d) = c.destroy {
                    if !c.data.is_null() {
                        d(c.data)
                    }
                }
            }
            _ => {}
        };
        libc::free(node.cast());
    }
}
unsafe fn mxml_new(parent: *mut MxmlNode, type_: MxmlType) -> *mut MxmlNode {
    unsafe {
        let node = libc::calloc(1, core::mem::size_of::<MxmlNode>()).cast::<MxmlNode>();
        if node.is_null() {
            return node;
        }
        (*node).type_ = type_;
        (*node).ref_count = 1;
        if !parent.is_null() {
            mxml_add(parent, MXML_ADD_AFTER, core::ptr::null_mut(), node);
        }
        node
    }
}
pub unsafe fn mxml_add(
    parent: *mut MxmlNode,
    where_: c_int,
    child: *mut MxmlNode,
    node: *mut MxmlNode,
) {
    unsafe {
        if parent.is_null() || node.is_null() {
            return;
        }
        if !(*node).parent.is_null() {
            mxml_remove(node);
        }
        (*node).parent = parent;
        if where_ == MXML_ADD_BEFORE {
            if child.is_null() || child == (*parent).child || (*child).parent != parent {
                (*node).next = (*parent).child;
                if !(*parent).child.is_null() {
                    (*(*parent).child).prev = node;
                } else {
                    (*parent).last_child = node;
                }
                (*parent).child = node;
            } else {
                (*node).next = child;
                (*node).prev = (*child).prev;
                if !(*child).prev.is_null() {
                    (*(*child).prev).next = node;
                } else {
                    (*parent).child = node;
                }
                (*child).prev = node;
            }
        } else if where_ == MXML_ADD_AFTER {
            if child.is_null() || child == (*parent).last_child || (*child).parent != parent {
                (*node).prev = (*parent).last_child;
                if !(*parent).last_child.is_null() {
                    (*(*parent).last_child).next = node;
                } else {
                    (*parent).child = node;
                }
                (*parent).last_child = node;
            } else {
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
    }
}
pub unsafe fn mxml_delete(node: *mut MxmlNode) {
    unsafe {
        if node.is_null() {
            return;
        }
        mxml_remove(node);
        let mut current = (*node).child;
        while !current.is_null() {
            let mut next = (*current).child;
            if !next.is_null() {
                (*current).child = core::ptr::null_mut();
                current = next;
                continue;
            }
            next = (*current).next;
            if next.is_null() {
                next = (*current).parent;
                if next == node {
                    next = core::ptr::null_mut();
                }
            }
            mxml_free(current);
            current = next;
        }
        mxml_free(node);
    }
}
pub unsafe fn mxml_get_ref_count(node: *mut MxmlNode) -> c_int {
    unsafe { if node.is_null() { 0 } else { (*node).ref_count } }
}
pub unsafe fn mxml_new_cdata(parent: *mut MxmlNode, data: *const c_char) -> *mut MxmlNode {
    unsafe {
        if data.is_null() {
            return core::ptr::null_mut();
        }
        let node = mxml_new(parent, MXML_ELEMENT);
        if !node.is_null() {
            let x = CStr::from_ptr(data).to_bytes();
            let mut v = b"![CDATA[".to_vec();
            v.extend_from_slice(x);
            v.extend_from_slice(b"]]");
            (*node).value.element.name = CString::new(v).unwrap().into_raw();
        }
        node
    }
}
pub unsafe fn mxml_new_custom(
    parent: *mut MxmlNode,
    data: *mut c_void,
    destroy: MxmlCustomDestroyCb,
) -> *mut MxmlNode {
    unsafe {
        let n = mxml_new(parent, MXML_CUSTOM);
        if !n.is_null() {
            (*n).value.custom = MxmlCustom { data, destroy };
        }
        n
    }
}
pub unsafe fn mxml_new_element(parent: *mut MxmlNode, name: *const c_char) -> *mut MxmlNode {
    unsafe {
        if name.is_null() {
            return core::ptr::null_mut();
        }
        let n = mxml_new(parent, MXML_ELEMENT);
        if !n.is_null() {
            (*n).value.element.name = dup(name)
        }
        n
    }
}
pub unsafe fn mxml_new_integer(parent: *mut MxmlNode, integer: c_int) -> *mut MxmlNode {
    unsafe {
        let n = mxml_new(parent, MXML_INTEGER);
        if !n.is_null() {
            (*n).value.integer = integer;
        }
        n
    }
}
pub unsafe fn mxml_new_opaque(parent: *mut MxmlNode, opaque: *const c_char) -> *mut MxmlNode {
    unsafe {
        if opaque.is_null() {
            return core::ptr::null_mut();
        }
        let n = mxml_new(parent, MXML_OPAQUE);
        if !n.is_null() {
            (*n).value.opaque = dup(opaque)
        }
        n
    }
}
pub unsafe fn mxml_new_real(parent: *mut MxmlNode, real: f64) -> *mut MxmlNode {
    unsafe {
        let n = mxml_new(parent, MXML_REAL);
        if !n.is_null() {
            (*n).value.real = real;
        }
        n
    }
}
pub unsafe fn mxml_new_text(
    parent: *mut MxmlNode,
    whitespace: c_int,
    string: *const c_char,
) -> *mut MxmlNode {
    unsafe {
        if string.is_null() {
            return core::ptr::null_mut();
        }
        let n = mxml_new(parent, MXML_TEXT);
        if !n.is_null() {
            (*n).value.text = MxmlText {
                whitespace,
                string: dup(string),
            }
        }
        n
    }
}
pub unsafe fn mxml_new_textf(
    parent: *mut MxmlNode,
    whitespace: c_int,
    format: *const c_char,
) -> *mut MxmlNode {
    unsafe { mxml_new_text(parent, whitespace, format) }
}
pub unsafe fn mxml_remove(node: *mut MxmlNode) {
    unsafe {
        if node.is_null() || (*node).parent.is_null() {
            return;
        }
        if !(*node).prev.is_null() {
            (*(*node).prev).next = (*node).next
        } else {
            (*(*node).parent).child = (*node).next
        }
        if !(*node).next.is_null() {
            (*(*node).next).prev = (*node).prev
        } else {
            (*(*node).parent).last_child = (*node).prev
        }
        (*node).parent = core::ptr::null_mut();
        (*node).prev = core::ptr::null_mut();
        (*node).next = core::ptr::null_mut();
    }
}
pub unsafe fn mxml_new_xml(version: *const c_char) -> *mut MxmlNode {
    unsafe {
        let s = if version.is_null() {
            "1.0".to_string()
        } else {
            CStr::from_ptr(version).to_string_lossy().into_owned()
        };
        let x = CString::new(format!("?xml version=\"{}\" encoding=\"utf-8\"?", s)).unwrap();
        mxml_new_element(core::ptr::null_mut(), x.as_ptr())
    }
}
pub unsafe fn mxml_release(node: *mut MxmlNode) -> c_int {
    unsafe {
        if node.is_null() {
            return -1;
        }
        (*node).ref_count -= 1;
        if (*node).ref_count <= 0 {
            mxml_delete(node);
            0
        } else {
            (*node).ref_count
        }
    }
}
pub unsafe fn mxml_retain(node: *mut MxmlNode) -> c_int {
    unsafe {
        if node.is_null() {
            -1
        } else {
            (*node).ref_count += 1;
            (*node).ref_count
        }
    }
}
