use super::*;
use core::ffi::{c_char, c_int, c_void};
pub unsafe fn mxml_set_cdata(n: *mut MxmlNode, data: *const c_char) -> c_int {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_ELEMENT || data.is_null() {
            return -1;
        }
        libc::free((*n).value.element.name.cast());
        let mut v = b"![CDATA[".to_vec();
        v.extend_from_slice(std::ffi::CStr::from_ptr(data).to_bytes());
        v.extend_from_slice(b"]]");
        (*n).value.element.name = std::ffi::CString::new(v).unwrap().into_raw();
        0
    }
}
pub unsafe fn mxml_set_custom(
    n: *mut MxmlNode,
    data: *mut c_void,
    destroy: MxmlCustomDestroyCb,
) -> c_int {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_CUSTOM {
            return -1;
        }
        let old = (*n).value.custom;
        if let Some(f) = old.destroy {
            if !old.data.is_null() {
                f(old.data)
            }
        }
        (*n).value.custom = MxmlCustom { data, destroy };
        0
    }
}
pub unsafe fn mxml_set_element(n: *mut MxmlNode, name: *const c_char) -> c_int {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_ELEMENT || name.is_null() {
            -1
        } else {
            libc::free((*n).value.element.name.cast());
            (*n).value.element.name = libc::strdup(name);
            if (*n).value.element.name.is_null() {
                -1
            } else {
                0
            }
        }
    }
}
pub unsafe fn mxml_set_integer(n: *mut MxmlNode, x: c_int) -> c_int {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_INTEGER {
            -1
        } else {
            (*n).value.integer = x;
            0
        }
    }
}
pub unsafe fn mxml_set_opaque(n: *mut MxmlNode, x: *const c_char) -> c_int {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_OPAQUE || x.is_null() {
            -1
        } else {
            let p = libc::strdup(x);
            if p.is_null() {
                -1
            } else {
                libc::free((*n).value.opaque.cast());
                (*n).value.opaque = p;
                0
            }
        }
    }
}
pub unsafe fn mxml_set_real(n: *mut MxmlNode, x: f64) -> c_int {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_REAL {
            -1
        } else {
            (*n).value.real = x;
            0
        }
    }
}
pub unsafe fn mxml_set_text(n: *mut MxmlNode, w: c_int, x: *const c_char) -> c_int {
    unsafe {
        if n.is_null() || (*n).type_ != MXML_TEXT || x.is_null() {
            -1
        } else {
            let p = libc::strdup(x);
            if p.is_null() {
                -1
            } else {
                libc::free((*n).value.text.string.cast());
                (*n).value.text = MxmlText {
                    whitespace: w,
                    string: p,
                };
                0
            }
        }
    }
}
pub unsafe fn mxml_set_textf(n: *mut MxmlNode, w: c_int, x: *const c_char) -> c_int {
    unsafe { mxml_set_text(n, w, x) }
}
pub unsafe fn mxml_set_user_data(n: *mut MxmlNode, x: *mut c_void) -> c_int {
    unsafe {
        if n.is_null() {
            -1
        } else {
            (*n).user_data = x;
            0
        }
    }
}
