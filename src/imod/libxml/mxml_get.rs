use super::*;
use core::ffi::{c_int, c_void};
pub unsafe fn mxml_get_cdata(n: *mut MxmlNode) -> *const i8 {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_ELEMENT {
            return core::ptr::null();
        }
        let p = (*n).value.element.name;
        if p.is_null() || libc::strncmp(p, c"![CDATA[".as_ptr(), 8) != 0 {
            core::ptr::null()
        } else {
            p.add(8)
        }
    }
}
pub unsafe fn mxml_get_custom(n: *mut MxmlNode) -> *const c_void {
    unsafe {
        if n.is_null() {
            return core::ptr::null();
        }
        if (*n).type_ == MXML_CUSTOM {
            (*n).value.custom.data
        } else if (*n).type_ == MXML_ELEMENT
            && !(*n).child.is_null()
            && (*(*n).child).type_ == MXML_CUSTOM
        {
            (*(*n).child).value.custom.data
        } else {
            core::ptr::null()
        }
    }
}
pub unsafe fn mxml_get_element(n: *mut MxmlNode) -> *const i8 {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_ELEMENT {
            core::ptr::null()
        } else {
            (*n).value.element.name
        }
    }
}
pub unsafe fn mxml_get_first_child(n: *mut MxmlNode) -> *mut MxmlNode {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_ELEMENT {
            core::ptr::null_mut()
        } else {
            (*n).child
        }
    }
}
pub unsafe fn mxml_get_integer(n: *mut MxmlNode) -> c_int {
    unsafe {
        if n.is_null() {
            0
        } else if (*n).type_ == MXML_INTEGER {
            (*n).value.integer
        } else if (*n).type_ == MXML_ELEMENT
            && !(*n).child.is_null()
            && (*(*n).child).type_ == MXML_INTEGER
        {
            (*(*n).child).value.integer
        } else {
            0
        }
    }
}
pub unsafe fn mxml_get_last_child(n: *mut MxmlNode) -> *mut MxmlNode {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_ELEMENT {
            core::ptr::null_mut()
        } else {
            (*n).last_child
        }
    }
}
pub unsafe fn mxml_get_next_sibling(n: *mut MxmlNode) -> *mut MxmlNode {
    unsafe {
        if n.is_null() {
            core::ptr::null_mut()
        } else {
            (*n).next
        }
    }
}
pub unsafe fn mxml_get_opaque(n: *mut MxmlNode) -> *const i8 {
    unsafe {
        if n.is_null() {
            core::ptr::null()
        } else if (*n).type_ == MXML_OPAQUE {
            (*n).value.opaque
        } else if (*n).type_ == MXML_ELEMENT
            && !(*n).child.is_null()
            && (*(*n).child).type_ == MXML_OPAQUE
        {
            (*(*n).child).value.opaque
        } else {
            core::ptr::null()
        }
    }
}
pub unsafe fn mxml_get_parent(n: *mut MxmlNode) -> *mut MxmlNode {
    unsafe {
        if n.is_null() {
            core::ptr::null_mut()
        } else {
            (*n).parent
        }
    }
}
pub unsafe fn mxml_get_prev_sibling(n: *mut MxmlNode) -> *mut MxmlNode {
    unsafe {
        if n.is_null() {
            core::ptr::null_mut()
        } else {
            (*n).prev
        }
    }
}
pub unsafe fn mxml_get_real(n: *mut MxmlNode) -> f64 {
    unsafe {
        if n.is_null() {
            0.0
        } else if (*n).type_ == MXML_REAL {
            (*n).value.real
        } else if (*n).type_ == MXML_ELEMENT
            && !(*n).child.is_null()
            && (*(*n).child).type_ == MXML_REAL
        {
            (*(*n).child).value.real
        } else {
            0.0
        }
    }
}
pub unsafe fn mxml_get_text(n: *mut MxmlNode, w: *mut c_int) -> *const i8 {
    unsafe {
        let v = if !n.is_null() && (*n).type_ == MXML_TEXT {
            Some((*n).value.text)
        } else if !n.is_null()
            && (*n).type_ == MXML_ELEMENT
            && !(*n).child.is_null()
            && (*(*n).child).type_ == MXML_TEXT
        {
            Some((*(*n).child).value.text)
        } else {
            None
        };
        match v {
            Some(t) => {
                if !w.is_null() {
                    *w = t.whitespace
                }
                t.string
            }
            None => {
                if !w.is_null() {
                    *w = 0
                }
                core::ptr::null()
            }
        }
    }
}
pub unsafe fn mxml_get_type(n: *mut MxmlNode) -> MxmlType {
    unsafe { if n.is_null() { MXML_IGNORE } else { (*n).type_ } }
}
pub unsafe fn mxml_get_user_data(n: *mut MxmlNode) -> *mut c_void {
    unsafe {
        if n.is_null() {
            core::ptr::null_mut()
        } else {
            (*n).user_data
        }
    }
}
