use super::*;
use core::ffi::c_int;
pub unsafe fn mxml_walk_next(
    node: *mut MxmlNode,
    top: *mut MxmlNode,
    descend: c_int,
) -> *mut MxmlNode {
    unsafe {
        if node.is_null() {
            return core::ptr::null_mut();
        }
        if !(*node).child.is_null() && descend != 0 {
            return (*node).child;
        }
        if node == top {
            return core::ptr::null_mut();
        }
        if !(*node).next.is_null() {
            return (*node).next;
        }
        if !(*node).parent.is_null() && (*node).parent != top {
            let mut n = (*node).parent;
            while (*n).next.is_null() {
                if (*n).parent == top || (*n).parent.is_null() {
                    return core::ptr::null_mut();
                }
                n = (*n).parent;
            }
            return (*n).next;
        }
        core::ptr::null_mut()
    }
}
pub unsafe fn mxml_walk_prev(
    node: *mut MxmlNode,
    top: *mut MxmlNode,
    descend: c_int,
) -> *mut MxmlNode {
    unsafe {
        if node.is_null() || node == top {
            return core::ptr::null_mut();
        }
        if !(*node).prev.is_null() {
            if !(*(*node).prev).last_child.is_null() && descend != 0 {
                let mut n = (*(*node).prev).last_child;
                while !(*n).last_child.is_null() {
                    n = (*n).last_child
                }
                n
            } else {
                (*node).prev
            }
        } else if (*node).parent != top {
            (*node).parent
        } else {
            core::ptr::null_mut()
        }
    }
}
pub unsafe fn mxml_find_element(
    mut node: *mut MxmlNode,
    top: *mut MxmlNode,
    name: *const i8,
    attr: *const i8,
    value: *const i8,
    descend: c_int,
) -> *mut MxmlNode {
    unsafe {
        if node.is_null() || top.is_null() || attr.is_null() && !value.is_null() {
            return core::ptr::null_mut();
        }
        node = mxml_walk_next(node, top, descend);
        while !node.is_null() {
            if (*node).type_ == MXML_ELEMENT
                && !(*node).value.element.name.is_null()
                && (name.is_null() || libc::strcmp((*node).value.element.name, name) == 0)
            {
                if attr.is_null() {
                    return node;
                }
                let x = super::mxml_element_get_attr(node, attr);
                if !x.is_null() && (value.is_null() || libc::strcmp(x, value) == 0) {
                    return node;
                }
            }
            node = if descend == MXML_DESCEND {
                mxml_walk_next(node, top, MXML_DESCEND)
            } else {
                (*node).next
            }
        }
        core::ptr::null_mut()
    }
}
pub unsafe fn mxml_find_path(top: *mut MxmlNode, path: *const i8) -> *mut MxmlNode {
    unsafe {
        if top.is_null() || path.is_null() {
            return core::ptr::null_mut();
        }
        let bytes = std::ffi::CStr::from_ptr(path).to_bytes();
        if bytes.is_empty() {
            return core::ptr::null_mut();
        }
        let mut n = top;
        let mut deep = MXML_DESCEND_FIRST;
        for part in bytes.split(|x| *x == b'/') {
            if part == b"*" {
                deep = MXML_DESCEND;
                continue;
            }
            let p = std::ffi::CString::new(part).unwrap();
            n = mxml_find_element(n, n, p.as_ptr(), core::ptr::null(), core::ptr::null(), deep);
            if n.is_null() {
                return n;
            }
            deep = MXML_DESCEND_FIRST;
        }
        if !(*n).child.is_null() && (*(*n).child).type_ != MXML_ELEMENT {
            (*n).child
        } else {
            n
        }
    }
}
