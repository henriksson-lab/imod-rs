use super::*;
use core::ffi::{c_char, c_int};
use std::ffi::CStr;
unsafe fn mxml_set_attr(node: *mut MxmlNode, name: *const c_char, value: *mut c_char) -> c_int {
    unsafe {
        let e = &mut (*node).value.element;
        for i in 0..e.num_attrs {
            let a = e.attrs.add(i as usize);
            if libc::strcmp((*a).name, name) == 0 {
                libc::free((*a).value.cast());
                (*a).value = value;
                return 0;
            }
        }
        let p = libc::realloc(
            e.attrs.cast(),
            ((e.num_attrs + 1) as usize) * core::mem::size_of::<MxmlAttr>(),
        )
        .cast::<MxmlAttr>();
        if p.is_null() {
            return -1;
        }
        e.attrs = p;
        let a = p.add(e.num_attrs as usize);
        (*a).name = libc::strdup(name);
        if (*a).name.is_null() {
            return -1;
        }
        (*a).value = value;
        e.num_attrs += 1;
        0
    }
}
pub unsafe fn mxml_element_delete_attr(node: *mut MxmlNode, name: *const c_char) {
    unsafe {
        if node.is_null() || (*node).type_ != MXML_ELEMENT || name.is_null() {
            return;
        }
        let e = &mut (*node).value.element;
        for i in 0..e.num_attrs {
            let a = e.attrs.add(i as usize);
            if libc::strcmp((*a).name, name) == 0 {
                libc::free((*a).name.cast());
                libc::free((*a).value.cast());
                let remain = e.num_attrs - i - 1;
                if remain > 0 {
                    core::ptr::copy(a.add(1), a, remain as usize)
                }
                e.num_attrs -= 1;
                if e.num_attrs == 0 {
                    libc::free(e.attrs.cast());
                    e.attrs = core::ptr::null_mut()
                }
                return;
            }
        }
    }
}
pub unsafe fn mxml_element_get_attr(node: *mut MxmlNode, name: *const c_char) -> *const c_char {
    unsafe {
        if node.is_null() || (*node).type_ != MXML_ELEMENT || name.is_null() {
            return core::ptr::null();
        }
        let e = (*node).value.element;
        for i in 0..e.num_attrs {
            let a = e.attrs.add(i as usize);
            if libc::strcmp((*a).name, name) == 0 {
                return (*a).value;
            }
        }
        core::ptr::null()
    }
}
pub unsafe fn mxml_element_set_attr(
    node: *mut MxmlNode,
    name: *const c_char,
    value: *const c_char,
) {
    unsafe {
        if node.is_null() || (*node).type_ != MXML_ELEMENT || name.is_null() {
            return;
        }
        let v = if value.is_null() {
            core::ptr::null_mut()
        } else {
            libc::strdup(value)
        };
        if mxml_set_attr(node, name, v) != 0 {
            libc::free(v.cast())
        }
    }
}
pub unsafe fn mxml_element_set_attrf(
    node: *mut MxmlNode,
    name: *const c_char,
    format: *const c_char,
) {
    unsafe { mxml_element_set_attr(node, name, format) }
}
