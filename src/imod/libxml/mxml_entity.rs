use super::*;
use core::ffi::{c_char, c_int};
const ENTITIES: &[(i32, &[u8])] = &[
    (34, b"quot\0"),
    (38, b"amp\0"),
    (39, b"apos\0"),
    (60, b"lt\0"),
    (62, b"gt\0"),
];
pub unsafe fn mxml_entity_add_callback(cb: MxmlEntityCb) -> c_int {
    let mut g = mxml_global().lock().unwrap();
    if g.num_entity_cbs >= 100 {
        return -1;
    }
    let i = g.num_entity_cbs as usize;
    g.entity_cbs[i] = cb;
    g.num_entity_cbs += 1;
    0
}
pub unsafe fn mxml_entity_get_name(val: c_int) -> *const c_char {
    for (v, n) in ENTITIES {
        if *v == val {
            return n.as_ptr().cast();
        }
    }
    core::ptr::null()
}
pub unsafe fn mxml_entity_get_value(name: *const c_char) -> c_int {
    unsafe {
        if name.is_null() {
            return -1;
        }
        for (v, n) in ENTITIES {
            if libc::strcmp(name, n.as_ptr().cast()) == 0 {
                return *v;
            }
        }
        mxml_entity_cb(name)
    }
}
pub unsafe fn mxml_entity_remove_callback(cb: MxmlEntityCb) {
    let mut g = mxml_global().lock().unwrap();
    for i in 0..g.num_entity_cbs as usize {
        if g.entity_cbs[i].map(|x| x as usize) == cb.map(|x| x as usize) {
            g.num_entity_cbs -= 1;
            g.entity_cbs.copy_within(i + 1.., i);
            let last = g.num_entity_cbs as usize;
            g.entity_cbs[last] = None;
            return;
        }
    }
}
pub unsafe fn mxml_entity_cb(name: *const c_char) -> c_int {
    let g = mxml_global().lock().unwrap();
    for i in 0..g.num_entity_cbs as usize {
        if let Some(cb) = g.entity_cbs[i] {
            let x = unsafe { cb(name) };
            if x >= 0 {
                return x;
            }
        }
    }
    -1
}
