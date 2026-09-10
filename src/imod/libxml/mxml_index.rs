use super::*;
use core::ffi::{c_char, c_int};
pub unsafe fn mxml_index_delete(ind: *mut MxmlIndex) {
    unsafe {
        if ind.is_null() {
            return;
        }
        libc::free((*ind).attr.cast());
        libc::free((*ind).nodes.cast());
        libc::free(ind.cast())
    }
}
pub unsafe fn mxml_index_enum(ind: *mut MxmlIndex) -> *mut MxmlNode {
    unsafe {
        if ind.is_null() || (*ind).cur_node >= (*ind).num_nodes {
            core::ptr::null_mut()
        } else {
            let n = *(*ind).nodes.add((*ind).cur_node as usize);
            (*ind).cur_node += 1;
            n
        }
    }
}
pub unsafe fn mxml_index_find(
    ind: *mut MxmlIndex,
    element: *const c_char,
    value: *const c_char,
) -> *mut MxmlNode {
    unsafe {
        if ind.is_null() {
            return core::ptr::null_mut();
        }
        for i in 0..(*ind).num_nodes {
            let n = *(*ind).nodes.add(i as usize);
            if (element.is_null() || libc::strcmp((*n).value.element.name, element) == 0)
                && (value.is_null()
                    || (*ind).attr.is_null()
                    || libc::strcmp(super::mxml_element_get_attr(n, (*ind).attr), value) == 0)
            {
                (*ind).cur_node = i + 1;
                return n;
            }
        }
        core::ptr::null_mut()
    }
}
pub unsafe fn mxml_index_get_count(ind: *mut MxmlIndex) -> c_int {
    unsafe { if ind.is_null() { 0 } else { (*ind).num_nodes } }
}
pub unsafe fn mxml_index_new(
    node: *mut MxmlNode,
    element: *const c_char,
    attr: *const c_char,
) -> *mut MxmlIndex {
    unsafe {
        if node.is_null() {
            return core::ptr::null_mut();
        }
        let ind = libc::calloc(1, core::mem::size_of::<MxmlIndex>()).cast::<MxmlIndex>();
        if ind.is_null() {
            return ind;
        }
        (*ind).attr = if attr.is_null() {
            core::ptr::null_mut()
        } else {
            libc::strdup(attr)
        };
        let mut n = node;
        loop {
            if (*n).type_ == MXML_ELEMENT
                && (element.is_null() || libc::strcmp((*n).value.element.name, element) == 0)
            {
                let p: *mut *mut MxmlNode = libc::realloc(
                    (*ind).nodes.cast(),
                    ((*ind).num_nodes + 1) as usize * core::mem::size_of::<*mut MxmlNode>(),
                )
                .cast();
                if p.is_null() {
                    mxml_index_delete(ind);
                    return core::ptr::null_mut();
                }
                (*ind).nodes = p;
                *(*ind).nodes.add((*ind).num_nodes as usize) = n;
                (*ind).num_nodes += 1;
            }
            n = super::mxml_walk_next(n, node, MXML_DESCEND);
            if n.is_null() {
                break;
            }
        }
        (*ind).alloc_nodes = (*ind).num_nodes;
        ind
    }
}
pub unsafe fn mxml_index_reset(ind: *mut MxmlIndex) -> *mut MxmlNode {
    unsafe {
        if ind.is_null() {
            core::ptr::null_mut()
        } else {
            (*ind).cur_node = 0;
            mxml_index_enum(ind)
        }
    }
}
