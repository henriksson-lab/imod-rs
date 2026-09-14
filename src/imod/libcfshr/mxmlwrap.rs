//! Translation of `IMOD/libcfshr/mxmlwrap.c` and `IMOD/include/mxmlwrap.h`.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::ilist::*;
use crate::imod::libcfshr::parse_params::{
    PIP_DOUBLE, PIP_FLOAT, PIP_INTEGER, pip_get_line_of_values,
};
use crate::imod::libxml::*;
use core::cell::{Cell, RefCell};
use core::ffi::{c_char, c_void};

const NODE_LIST_QUANTUM: i32 = 32;
static mut S_NODE_LISTS: *mut *mut Ilist = core::ptr::null_mut();
static mut S_NUM_LISTS: i32 = 0;

thread_local! {
    /// The `mxml_node_t *` values the C keeps in `sNodeLists` are arena slot
    /// indices now, so the arena each document was parsed into has to live
    /// beside its node list; this is that storage, indexed by the same
    /// `xmlInd`.
    static S_ARENAS: RefCell<Vec<Option<MxmlArena>>> = const { RefCell::new(Vec::new()) };
    /// Matches C static `sLastLevel`.
    static S_LAST_LEVEL: Cell<i32> = const { Cell::new(-1) };
}

/// Matches C static `getOrAddFreeList`.
pub unsafe fn get_or_add_free_list() -> i32 {
    let mut xml_ind = 0;
    while xml_ind < S_NUM_LISTS {
        if (*S_NODE_LISTS.add(xml_ind as usize)).is_null() {
            break;
        }
        xml_ind += 1;
    }
    if xml_ind >= S_NUM_LISTS {
        let new_lists =
            libc::malloc(((S_NUM_LISTS + 1) as usize) * core::mem::size_of::<*mut Ilist>())
                .cast::<*mut Ilist>();
        if new_lists.is_null() {
            return -1;
        }
        if S_NUM_LISTS != 0 {
            core::ptr::copy_nonoverlapping(S_NODE_LISTS, new_lists, S_NUM_LISTS as usize);
        }
        libc::free(S_NODE_LISTS.cast::<c_void>());
        S_NODE_LISTS = new_lists;
        S_NUM_LISTS += 1;
    }
    *S_NODE_LISTS.add(xml_ind as usize) =
        ilist_new(core::mem::size_of::<usize>() as i32, NODE_LIST_QUANTUM)
            .map_or(core::ptr::null_mut(), Box::into_raw);
    if (*S_NODE_LISTS.add(xml_ind as usize)).is_null() {
        xml_ind = -1;
    } else {
        ilist_quantum(&mut **S_NODE_LISTS.add(xml_ind as usize), NODE_LIST_QUANTUM);
        S_ARENAS.with_borrow_mut(|arenas| {
            while arenas.len() <= xml_ind as usize {
                arenas.push(None);
            }
        });
    }
    xml_ind
}

/// Matches C static `getNodeAtIndex`.
///
/// The C returns the `mxml_node_t *` stored in the list; the list now holds
/// arena slot indices, so this returns one of those.
pub unsafe fn get_node_at_index(xml_ind: i32, node_ind: i32, error: *mut i32) -> Option<usize> {
    if xml_ind < 0 || xml_ind >= S_NUM_LISTS {
        *error = -2;
        return None;
    }
    let node_list = *S_NODE_LISTS.add(xml_ind as usize);
    if node_list.is_null() {
        *error = -2;
        return None;
    }
    if node_ind < 0 || node_ind >= ilist_size(node_list.as_ref()) {
        *error = -3;
        return None;
    }
    *error = 0;
    let item = ilist_item(node_list.as_mut(), node_ind)?;
    Some(usize::from_ne_bytes(
        item[..core::mem::size_of::<usize>()].try_into().unwrap(),
    ))
}

/// Matches C static `getElementString`.
///
/// The C hands back a pointer into the node's opaque value; the bytes are
/// copied into `string` with a terminating NUL instead, because the callers
/// pass them on to `pipGetLineOfValues`, which is still C-shaped.
pub unsafe fn get_element_string(xml_ind: i32, node_ind: i32, string: &mut Vec<u8>) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    let mut result = -4;
    S_ARENAS.with_borrow(|arenas| {
        let Some(arena) = arenas.get(xml_ind as usize).and_then(|a| a.as_ref()) else {
            return;
        };
        let child = mxml_get_first_child(arena, node);
        if child.is_none()
            || mxml_get_type(arena, child) != MXML_OPAQUE
            || mxml_get_last_child(arena, node) != child
        {
            return;
        }
        if let MxmlValue::Opaque(opaque) = &arena.node(child.unwrap()).value {
            // `mxmlwrap.c:511` hands back a borrowed `const char *` into
            // mini-XML's own NUL-terminated buffer.  Here the bytes are copied
            // and stay **length-delimited**: the three `PipGetLineOfValues`
            // callers below take a `&[u8]` and scan the whole slice, so a
            // trailing NUL would be an unparseable character rather than a
            // terminator.  `ixmlGetStringValue` adds one back for its `strdup`.
            string.clear();
            string.extend_from_slice(opaque.as_deref().unwrap_or(b""));
            result = 0;
        }
    });
    result
}

/// Matches C static `processLoadedNodes`.
pub unsafe fn process_loaded_nodes(
    mut arena: MxmlArena,
    xml: Option<usize>,
    with_xml_decl: i32,
    root_element: *mut *mut c_char,
) -> i32 {
    let xml_ind = get_or_add_free_list();
    if xml_ind < 0 {
        mxml_delete(&mut arena, xml);
        return xml_ind;
    }
    ilist_append(
        &mut **S_NODE_LISTS.add(xml_ind as usize),
        &xml.unwrap_or(usize::MAX).to_ne_bytes(),
    );
    let mut top = xml;
    if with_xml_decl != 0 {
        top = mxml_walk_next(&arena, xml, xml, MXML_DESCEND);
        if mxml_get_type(&arena, top) == MXML_OPAQUE {
            top = mxml_walk_next(&arena, top, xml, MXML_DESCEND);
        }
        ilist_append(
            &mut **S_NODE_LISTS.add(xml_ind as usize),
            &top.unwrap_or(usize::MAX).to_ne_bytes(),
        );
    }
    *root_element = core::ptr::null_mut();
    if let Some(root) = mxml_get_element(&arena, top) {
        let mut owned: Vec<u8> = root.to_vec();
        owned.push(0);
        *root_element = libc::strdup(owned.as_ptr().cast::<c_char>());
    }
    S_ARENAS.with_borrow_mut(|arenas| arenas[xml_ind as usize] = Some(arena));
    xml_ind
}

/// Matches C `ixmlReadFile`.
pub unsafe extern "C" fn ixml_read_file(
    filename: *const c_char,
    with_xml_decl: i32,
    root_element: *mut *mut c_char,
) -> i32 {
    let path = std::ffi::CStr::from_ptr(filename)
        .to_string_lossy()
        .into_owned();
    let Some(mut fp) = ImodFile::open(&path, "r") else {
        return -2;
    };
    let mut arena = MxmlArena::new();
    let xml = mxml_load_file(&mut arena, MXML_NO_PARENT, &mut fp, Some(mxml_opaque_cb));
    if xml.is_none() {
        return -3;
    }
    drop(fp);
    process_loaded_nodes(arena, xml, with_xml_decl, root_element)
}

/// Matches C `ixmlLoadString`.
pub unsafe extern "C" fn ixml_load_string(
    string: *const c_char,
    with_xml_decl: i32,
    root_element: *mut *mut c_char,
) -> i32 {
    let mut arena = MxmlArena::new();
    let xml = mxml_load_string(
        &mut arena,
        MXML_NO_PARENT,
        std::ffi::CStr::from_ptr(string).to_bytes(),
        Some(mxml_opaque_cb),
    );
    if xml.is_none() {
        return -3;
    }
    process_loaded_nodes(arena, xml, with_xml_decl, root_element)
}

/// Matches C `ixmlNewNodeList`.
pub unsafe extern "C" fn ixml_new_node_list(root_element: *const c_char) -> i32 {
    let mut xml_ind = get_or_add_free_list();
    if xml_ind >= 0 {
        let mut arena = MxmlArena::new();
        let xml = mxml_new_xml(&mut arena, Some(b"1.0"));
        if xml.is_none() {
            xml_ind = -2;
        } else {
            ilist_append(
                &mut **S_NODE_LISTS.add(xml_ind as usize),
                &xml.unwrap().to_ne_bytes(),
            );
            let top = mxml_new_element(
                &mut arena,
                xml,
                Some(std::ffi::CStr::from_ptr(root_element).to_bytes()),
            );
            if top.is_none() {
                xml_ind = -2;
            } else {
                ilist_append(
                    &mut **S_NODE_LISTS.add(xml_ind as usize),
                    &top.unwrap().to_ne_bytes(),
                );
            }
        }
        if xml_ind >= 0 {
            S_ARENAS.with_borrow_mut(|arenas| arenas[xml_ind as usize] = Some(arena));
        }
    }
    xml_ind
}

/// Matches C `ixmlWriteFile`.
pub unsafe extern "C" fn ixml_write_file(xml_ind: i32, filename: *const c_char) -> i32 {
    let mut err = 0;
    let xml = get_node_at_index(xml_ind, 0, &mut err);
    if xml.is_none() {
        return err;
    }
    let path = std::ffi::CStr::from_ptr(filename)
        .to_string_lossy()
        .into_owned();
    let Some(mut file) = ImodFile::open(&path, "w") else {
        return -1;
    };
    mxml_set_wrap_margin(0);
    S_LAST_LEVEL.set(-1);
    let callback: MxmlSaveCb = Some(ixml_whitespace_cb);
    S_ARENAS.with_borrow(|arenas| {
        let Some(arena) = arenas.get(xml_ind as usize).and_then(|a| a.as_ref()) else {
            return -1;
        };
        mxml_save_file(arena, xml, &mut file, callback)
    })
}

/// Matches C `ixmlFindElements`.
pub unsafe extern "C" fn ixml_find_elements(
    xml_ind: i32,
    node_ind: i32,
    tag: *const c_char,
    found_ind: *mut i32,
    num_found: *mut i32,
) -> i32 {
    let mut err = 0;
    *num_found = 0;
    let parent = get_node_at_index(xml_ind, node_ind, &mut err);
    if parent.is_none() {
        return err;
    }
    let tag = std::ffi::CStr::from_ptr(tag).to_bytes();
    let mut nodes: Vec<usize> = Vec::new();
    let found = S_ARENAS.with_borrow(|arenas| {
        let Some(arena) = arenas.get(xml_ind as usize).and_then(|a| a.as_ref()) else {
            return false;
        };
        let mut node = mxml_get_first_child(arena, parent);
        while let Some(cur) = node {
            if mxml_get_element(arena, node) == Some(tag) {
                nodes.push(cur);
            }
            node = mxml_get_next_sibling(arena, node);
        }
        true
    });
    if !found {
        return -2;
    }
    *found_ind = ilist_size((*S_NODE_LISTS.add(xml_ind as usize)).as_ref());
    for node in nodes {
        if ilist_append(
            &mut **S_NODE_LISTS.add(xml_ind as usize),
            &node.to_ne_bytes(),
        ) != 0
        {
            return -1;
        }
        *num_found += 1;
    }
    0
}

/// Matches C `ixmlGetStringValue`.
pub unsafe extern "C" fn ixml_get_string_value(
    xml_ind: i32,
    node_ind: i32,
    string: *mut *mut c_char,
) -> i32 {
    let mut value: Vec<u8> = Vec::new();
    let err = get_element_string(xml_ind, node_ind, &mut value);
    if err != 0 {
        return err;
    }
    // `get_element_string` is length-delimited; `strdup` needs the NUL.
    value.push(0);
    *string = libc::strdup(value.as_ptr().cast::<c_char>());
    if (*string).is_null() { -1 } else { 0 }
}

/// Matches C `ixmlGetIntegerValue`.
pub unsafe extern "C" fn ixml_get_integer_value(xml_ind: i32, node_ind: i32, val: *mut i32) -> i32 {
    let mut string: Vec<u8> = Vec::new();
    let mut num = 1;
    let err = get_element_string(xml_ind, node_ind, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(
        &string,
        &string,
        crate::imod::libcfshr::parse_params::PipValueArray::Int(core::slice::from_raw_parts_mut(
            val, 1,
        )),
        PIP_INTEGER,
        &mut num,
        1,
    )
}

/// Matches C `ixmlGetFloatValue`.
pub unsafe extern "C" fn ixml_get_float_value(xml_ind: i32, node_ind: i32, val: *mut f32) -> i32 {
    let mut string: Vec<u8> = Vec::new();
    let mut num = 1;
    let err = get_element_string(xml_ind, node_ind, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(
        &string,
        &string,
        crate::imod::libcfshr::parse_params::PipValueArray::Float(core::slice::from_raw_parts_mut(
            val, 1,
        )),
        PIP_FLOAT,
        &mut num,
        1,
    )
}

/// Matches C `ixmlGetDoubleValue`.
pub unsafe extern "C" fn ixml_get_double_value(xml_ind: i32, node_ind: i32, val: *mut f64) -> i32 {
    let mut string: Vec<u8> = Vec::new();
    let mut num = 1;
    let err = get_element_string(xml_ind, node_ind, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(
        &string,
        &string,
        crate::imod::libcfshr::parse_params::PipValueArray::Double(
            core::slice::from_raw_parts_mut(val, 1),
        ),
        PIP_DOUBLE,
        &mut num,
        1,
    )
}

/// Matches C `ixmlGetStringAttribute`.
pub unsafe extern "C" fn ixml_get_string_attribute(
    xml_ind: i32,
    node_ind: i32,
    name: *const c_char,
    string: *mut *mut c_char,
) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    let name = std::ffi::CStr::from_ptr(name).to_bytes();
    let mut value: Option<Vec<u8>> = None;
    S_ARENAS.with_borrow(|arenas| {
        let Some(arena) = arenas.get(xml_ind as usize).and_then(|a| a.as_ref()) else {
            return;
        };
        let MxmlValue::Element(element) = &arena.node(node.unwrap()).value else {
            return;
        };
        for ind in 0..element.num_attrs {
            let attr = &element.attrs[ind as usize];
            if attr.name == name {
                value = Some(attr.value.clone().unwrap_or_default());
                return;
            }
        }
    });
    let Some(value) = value else {
        return 1;
    };
    *string = libc::strdup(value.as_ptr().cast::<c_char>());
    if (*string).is_null() {
        return -1;
    }
    0
}

/// Matches C `ixmlGetIntegerAttribute`.
pub unsafe extern "C" fn ixml_get_integer_attribute(
    xml_ind: i32,
    node_ind: i32,
    name: *const c_char,
    val: *mut i32,
) -> i32 {
    let mut err = 0;
    let mut num = 1;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    let name = std::ffi::CStr::from_ptr(name).to_bytes();
    let mut value: Option<Vec<u8>> = None;
    S_ARENAS.with_borrow(|arenas| {
        let Some(arena) = arenas.get(xml_ind as usize).and_then(|a| a.as_ref()) else {
            return;
        };
        let MxmlValue::Element(element) = &arena.node(node.unwrap()).value else {
            return;
        };
        for ind in 0..element.num_attrs {
            let attr = &element.attrs[ind as usize];
            if attr.name == name {
                value = Some(attr.value.clone().unwrap_or_default());
                return;
            }
        }
    });
    let Some(value) = value else {
        return 1;
    };
    pip_get_line_of_values(
        &value,
        &value,
        crate::imod::libcfshr::parse_params::PipValueArray::Int(core::slice::from_raw_parts_mut(
            val, 1,
        )),
        PIP_INTEGER,
        &mut num,
        1,
    )
}

/// Matches C `ixmlAddElement`.
pub unsafe extern "C" fn ixml_add_element(xml_ind: i32, node_ind: i32, tag: *const c_char) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    let tag = std::ffi::CStr::from_ptr(tag).to_bytes();
    let elem = S_ARENAS.with_borrow_mut(|arenas| {
        let Some(arena) = arenas.get_mut(xml_ind as usize).and_then(|a| a.as_mut()) else {
            return None;
        };
        mxml_new_element(arena, node, Some(tag))
    });
    let Some(elem) = elem else {
        return -1;
    };
    if ilist_append(
        &mut **S_NODE_LISTS.add(xml_ind as usize),
        &elem.to_ne_bytes(),
    ) != 0
    {
        return -1;
    }
    ilist_size((*S_NODE_LISTS.add(xml_ind as usize)).as_ref()) - 1
}

/// Matches C `ixmlSetStringValue`.
pub unsafe extern "C" fn ixml_set_string_value(
    xml_ind: i32,
    node_ind: i32,
    string: *const c_char,
) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    let string = std::ffi::CStr::from_ptr(string).to_bytes();
    S_ARENAS.with_borrow_mut(|arenas| {
        let Some(arena) = arenas.get_mut(xml_ind as usize).and_then(|a| a.as_mut()) else {
            return -1;
        };
        if mxml_new_text(arena, node, 0, Some(string)).is_none() {
            return -1;
        }
        err
    })
}

/// Matches C `ixmlSetIntegerValue`.
pub unsafe extern "C" fn ixml_set_integer_value(xml_ind: i32, node_ind: i32, val: i32) -> i32 {
    let mut buffer = [0 as c_char; 64];
    libc::snprintf(buffer.as_mut_ptr(), buffer.len(), c"%d".as_ptr(), val);
    ixml_set_string_value(xml_ind, node_ind, buffer.as_ptr())
}

/// Matches C `ixmlSetFloatValue`.
pub unsafe extern "C" fn ixml_set_float_value(xml_ind: i32, node_ind: i32, val: f32) -> i32 {
    let mut buffer = [0 as c_char; 64];
    libc::snprintf(
        buffer.as_mut_ptr(),
        buffer.len(),
        c"%g".as_ptr(),
        val as f64,
    );
    ixml_set_string_value(xml_ind, node_ind, buffer.as_ptr())
}

/// Matches C `ixmlSetDoubleValue`.
pub unsafe extern "C" fn ixml_set_double_value(xml_ind: i32, node_ind: i32, val: f64) -> i32 {
    let mut buffer = [0 as c_char; 64];
    libc::snprintf(buffer.as_mut_ptr(), buffer.len(), c"%g".as_ptr(), val);
    ixml_set_string_value(xml_ind, node_ind, buffer.as_ptr())
}

/// Matches C `ixmlAddStringAttribute`.
pub unsafe extern "C" fn ixml_add_string_attribute(
    xml_ind: i32,
    node_ind: i32,
    name: *const c_char,
    value: *const c_char,
) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    let name = std::ffi::CStr::from_ptr(name).to_bytes();
    let value = if value.is_null() {
        None
    } else {
        Some(std::ffi::CStr::from_ptr(value).to_bytes())
    };
    S_ARENAS.with_borrow_mut(|arenas| {
        if let Some(arena) = arenas.get_mut(xml_ind as usize).and_then(|a| a.as_mut()) {
            mxml_element_set_attr(arena, node, Some(name), value);
        }
    });
    0
}

/// Matches C `ixmlAddIntegerAttribute`.
pub unsafe extern "C" fn ixml_add_integer_attribute(
    xml_ind: i32,
    node_ind: i32,
    name: *const c_char,
    value: i32,
) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    let name = std::ffi::CStr::from_ptr(name).to_bytes();
    let mut buffer = [0 as c_char; 64];
    libc::snprintf(buffer.as_mut_ptr(), buffer.len(), c"%d".as_ptr(), value);
    let text = std::ffi::CStr::from_ptr(buffer.as_ptr())
        .to_bytes()
        .to_vec();
    S_ARENAS.with_borrow_mut(|arenas| {
        if let Some(arena) = arenas.get_mut(xml_ind as usize).and_then(|a| a.as_mut()) {
            mxml_element_set_attr(arena, node, Some(name), Some(&text));
        }
    });
    0
}

/// Matches C `ixmlPopListIndexes`.
pub unsafe extern "C" fn ixml_pop_list_indexes(xml_ind: i32, first_ind: i32) -> i32 {
    if xml_ind < 0 || xml_ind >= S_NUM_LISTS || (*S_NODE_LISTS.add(xml_ind as usize)).is_null() {
        return -2;
    }
    if first_ind < 2 || first_ind >= ilist_size((*S_NODE_LISTS.add(xml_ind as usize)).as_ref()) {
        return -3;
    }
    ilist_truncate(&mut **S_NODE_LISTS.add(xml_ind as usize), first_ind);
    0
}

/// Matches C `ixmlClear`.
pub unsafe extern "C" fn ixml_clear(index: i32) {
    if index < 0 && index >= S_NUM_LISTS {
        return;
    }
    let mut err = 0;
    let xml = get_node_at_index(index, 0, &mut err);
    S_ARENAS.with_borrow_mut(|arenas| {
        if let Some(slot) = arenas.get_mut(index as usize) {
            if let Some(arena) = slot.as_mut() {
                mxml_delete(arena, xml);
            }
            *slot = None;
        }
    });
    ilist_delete(Some(Box::from_raw(*S_NODE_LISTS.add(index as usize))));
    *S_NODE_LISTS.add(index as usize) = core::ptr::null_mut();
}

/// Matches C `ixmlResetLastLevel`.
pub unsafe extern "C" fn ixml_reset_last_level() {
    S_LAST_LEVEL.set(-1);
}

/// Matches C `ixmlWhitespace_cb`.
///
/// The C returns a pointer into the static `sWhitespaceBuffer`; the callback
/// returns owned bytes now, so the buffer is gone.
pub fn ixml_whitespace_cb(arena: &MxmlArena, node: usize, where_: i32) -> Option<Vec<u8>> {
    if where_ != MXML_WS_BEFORE_OPEN && where_ != MXML_WS_BEFORE_CLOSE {
        return None;
    }
    let mut level = -1;
    let mut parent = arena.node(node).parent;
    while let Some(pp) = parent {
        level += 1;
        parent = arena.node(pp).parent;
    }
    if level > 16 {
        level = 16;
    } else if level < 0 {
        level = 0;
    }
    if S_LAST_LEVEL.get() < 0 {
        S_LAST_LEVEL.set(level);
        return None;
    }
    if level == S_LAST_LEVEL.get() && where_ == MXML_WS_BEFORE_CLOSE {
        return None;
    }
    S_LAST_LEVEL.set(level);
    let spaces = [b' '; 32];
    let mut out: Vec<u8> = vec![b'\n'];
    out.extend_from_slice(&spaces[..(2 * level) as usize]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{LazyLock, Mutex};

    static TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[test]
    fn loaded_xml_indexes_elements_values_and_attributes() {
        let _lock = TEST_LOCK.lock().unwrap();
        unsafe {
            let mut root = core::ptr::null_mut();
            let xml = c"<Root><Item number=\"12\">3.5</Item><Item number=\"13\">7</Item></Root>";
            let index = ixml_load_string(xml.as_ptr(), 0, &mut root);
            assert!(index >= 0);
            assert_eq!(std::ffi::CStr::from_ptr(root).to_bytes(), b"Root");
            libc::free(root.cast());
            let mut found = -1;
            let mut number = -1;
            assert_eq!(
                ixml_find_elements(index, 0, c"Item".as_ptr(), &mut found, &mut number),
                0
            );
            assert_eq!(number, 2);
            let mut value = 0.0f32;
            assert_eq!(ixml_get_float_value(index, found, &mut value), 0);
            assert_eq!(value, 3.5);
            let mut attribute = 0;
            assert_eq!(
                ixml_get_integer_attribute(index, found, c"number".as_ptr(), &mut attribute),
                0
            );
            assert_eq!(attribute, 12);
            ixml_clear(index);
        }
    }

    #[test]
    fn new_list_adds_nodes_and_source_error_indexes() {
        let _lock = TEST_LOCK.lock().unwrap();
        unsafe {
            let index = ixml_new_node_list(c"Root".as_ptr());
            assert!(index >= 0);
            let element = ixml_add_element(index, 1, c"Child".as_ptr());
            assert_eq!(element, 2);
            assert_eq!(ixml_set_integer_value(index, element, 41), 0);
            assert_eq!(
                ixml_add_integer_attribute(index, element, c"count".as_ptr(), 4),
                0
            );
            let mut count = 0;
            assert_eq!(
                ixml_get_integer_attribute(index, element, c"count".as_ptr(), &mut count),
                0
            );
            assert_eq!(count, 4);
            assert_eq!(ixml_pop_list_indexes(index, 1), -3);
            assert_eq!(ixml_pop_list_indexes(index, 2), 0);
            assert_eq!(ixml_add_element(-1, 0, c"x".as_ptr()), -2);
            ixml_clear(index);
        }
    }
}
