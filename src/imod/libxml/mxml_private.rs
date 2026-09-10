use super::*;
use core::ffi::{c_char, c_int};
use std::sync::{Mutex, OnceLock};
pub struct MxmlGlobal {
    pub error_cb: MxmlErrorCb,
    pub entity_cbs: [MxmlEntityCb; 100],
    pub num_entity_cbs: c_int,
    pub wrap: c_int,
    pub custom_load_cb: MxmlCustomLoadCb,
    pub custom_save_cb: MxmlCustomSaveCb,
}
unsafe impl Send for MxmlGlobal {}
static GLOBAL: OnceLock<Mutex<MxmlGlobal>> = OnceLock::new();
pub fn mxml_global() -> &'static Mutex<MxmlGlobal> {
    GLOBAL.get_or_init(|| {
        Mutex::new(MxmlGlobal {
            error_cb: None,
            entity_cbs: [None; 100],
            num_entity_cbs: 0,
            wrap: 0,
            custom_load_cb: None,
            custom_save_cb: None,
        })
    })
}
pub unsafe fn mxml_error(format: *const c_char) {
    unsafe {
        if let Some(cb) = mxml_global().lock().unwrap().error_cb {
            cb(format)
        }
    }
}
pub unsafe extern "C" fn mxml_ignore_cb(_: *mut MxmlNode) -> MxmlType {
    MXML_IGNORE
}
pub unsafe extern "C" fn mxml_integer_cb(_: *mut MxmlNode) -> MxmlType {
    MXML_INTEGER
}
pub unsafe extern "C" fn mxml_opaque_cb(_: *mut MxmlNode) -> MxmlType {
    MXML_OPAQUE
}
pub unsafe extern "C" fn mxml_real_cb(_: *mut MxmlNode) -> MxmlType {
    MXML_REAL
}
