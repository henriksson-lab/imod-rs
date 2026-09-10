//! Translation of `IMOD/libcfshr/mxmlwrap.c` and `IMOD/include/mxmlwrap.h`.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::imod::libcfshr::ilist::*;
use crate::imod::libcfshr::parse_params::{
    PIP_DOUBLE, PIP_FLOAT, PIP_INTEGER, pip_get_line_of_values,
};
use crate::imod::libxml::*;
use core::ffi::{c_char, c_void};

const NODE_LIST_QUANTUM: i32 = 32;
static mut S_NODE_LISTS: *mut *mut Ilist = core::ptr::null_mut();
static mut S_NUM_LISTS: i32 = 0;
static mut S_LAST_LEVEL: i32 = -1;
static mut S_WHITESPACE_BUFFER: [c_char; 36] = [0; 36];

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
    *S_NODE_LISTS.add(xml_ind as usize) = ilist_new(
        core::mem::size_of::<*mut MxmlNode>() as i32,
        NODE_LIST_QUANTUM,
    );
    if (*S_NODE_LISTS.add(xml_ind as usize)).is_null() {
        xml_ind = -1;
    } else {
        ilist_quantum(*S_NODE_LISTS.add(xml_ind as usize), NODE_LIST_QUANTUM);
    }
    xml_ind
}

/// Matches C static `getNodeAtIndex`.
pub unsafe fn get_node_at_index(xml_ind: i32, node_ind: i32, error: *mut i32) -> *mut MxmlNode {
    if xml_ind < 0 || xml_ind >= S_NUM_LISTS {
        *error = -2;
        return core::ptr::null_mut();
    }
    let node_list = *S_NODE_LISTS.add(xml_ind as usize);
    if node_list.is_null() {
        *error = -2;
        return core::ptr::null_mut();
    }
    if node_ind < 0 || node_ind >= ilist_size(node_list) {
        *error = -3;
        return core::ptr::null_mut();
    }
    *error = 0;
    let list_ptr = ilist_item(node_list, node_ind).cast::<*mut MxmlNode>();
    *list_ptr
}

/// Matches C static `getElementString`.
pub unsafe fn get_element_string(xml_ind: i32, node_ind: i32, string: *mut *const c_char) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_null() {
        return err;
    }
    let child = mxml_get_first_child(node);
    if child.is_null() || mxml_get_type(child) != MXML_OPAQUE || mxml_get_last_child(node) != child
    {
        return -4;
    }
    *string = (*child).value.opaque;
    0
}

/// Matches C static `processLoadedNodes`.
pub unsafe fn process_loaded_nodes(
    xml: *mut MxmlNode,
    with_xml_decl: i32,
    root_element: *mut *mut c_char,
) -> i32 {
    let xml_ind = get_or_add_free_list();
    if xml_ind < 0 {
        mxml_delete(xml);
        return xml_ind;
    }
    ilist_append(
        *S_NODE_LISTS.add(xml_ind as usize),
        (&raw const xml).cast_mut().cast(),
    );
    let mut top = xml;
    if with_xml_decl != 0 {
        top = mxml_walk_next(xml, xml, MXML_DESCEND);
        if (*top).type_ == MXML_OPAQUE {
            top = mxml_walk_next(top, xml, MXML_DESCEND);
        }
        ilist_append(
            *S_NODE_LISTS.add(xml_ind as usize),
            (&raw const top).cast_mut().cast(),
        );
    }
    let root = mxml_get_element(top);
    *root_element = core::ptr::null_mut();
    if !root.is_null() {
        *root_element = libc::strdup(root);
    }
    xml_ind
}

/// Matches C `ixmlReadFile`.
pub unsafe extern "C" fn ixml_read_file(
    filename: *const c_char,
    with_xml_decl: i32,
    root_element: *mut *mut c_char,
) -> i32 {
    let fp = libc::fopen(filename, c"r".as_ptr());
    if fp.is_null() {
        return -2;
    }
    let xml = mxml_load_file(core::ptr::null_mut(), fp, Some(mxml_opaque_cb));
    if xml.is_null() {
        return -3;
    }
    libc::fclose(fp);
    process_loaded_nodes(xml, with_xml_decl, root_element)
}
/// Matches C `ixmlLoadString`.
pub unsafe extern "C" fn ixml_load_string(
    string: *const c_char,
    with_xml_decl: i32,
    root_element: *mut *mut c_char,
) -> i32 {
    let xml = mxml_load_string(core::ptr::null_mut(), string, Some(mxml_opaque_cb));
    if xml.is_null() {
        return -3;
    }
    process_loaded_nodes(xml, with_xml_decl, root_element)
}
/// Matches C `ixmlNewNodeList`.
pub unsafe extern "C" fn ixml_new_node_list(root_element: *const c_char) -> i32 {
    let mut xml_ind = get_or_add_free_list();
    if xml_ind >= 0 {
        let xml = mxml_new_xml(c"1.0".as_ptr());
        if xml.is_null() {
            xml_ind = -2;
        } else {
            ilist_append(
                *S_NODE_LISTS.add(xml_ind as usize),
                (&raw const xml).cast_mut().cast(),
            );
            let top = mxml_new_element(xml, root_element);
            if top.is_null() {
                xml_ind = -2;
            } else {
                ilist_append(
                    *S_NODE_LISTS.add(xml_ind as usize),
                    (&raw const top).cast_mut().cast(),
                );
            }
        }
    }
    xml_ind
}
/// Matches C `ixmlWriteFile`.
pub unsafe extern "C" fn ixml_write_file(xml_ind: i32, filename: *const c_char) -> i32 {
    let mut err = 0;
    let xml = get_node_at_index(xml_ind, 0, &mut err);
    if xml.is_null() {
        return err;
    }
    let file = libc::fopen(filename, c"w".as_ptr());
    if file.is_null() {
        return -1;
    }
    mxml_set_wrap_margin(0);
    S_LAST_LEVEL = -1;
    let callback: MxmlSaveCb = Some(core::mem::transmute(
        ixml_whitespace_cb as unsafe extern "C" fn(*mut c_void, i32) -> *const c_char,
    ));
    err = mxml_save_file(xml, file, callback);
    libc::fclose(file);
    err
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
    if parent.is_null() {
        return err;
    }
    let mut node = mxml_get_first_child(parent);
    *found_ind = ilist_size(*S_NODE_LISTS.add(xml_ind as usize));
    while !node.is_null() {
        let key = mxml_get_element(node);
        if !key.is_null() && libc::strcmp(key, tag) == 0 {
            if ilist_append(
                *S_NODE_LISTS.add(xml_ind as usize),
                (&raw const node).cast_mut().cast(),
            ) != 0
            {
                return -1;
            }
            *num_found += 1;
        }
        node = mxml_get_next_sibling(node);
    }
    0
}
/// Matches C `ixmlGetStringValue`.
pub unsafe extern "C" fn ixml_get_string_value(
    xml_ind: i32,
    node_ind: i32,
    string: *mut *mut c_char,
) -> i32 {
    let mut value = core::ptr::null();
    let err = get_element_string(xml_ind, node_ind, &mut value);
    if err != 0 {
        return err;
    }
    *string = libc::strdup(value);
    if (*string).is_null() { -1 } else { 0 }
}
/// Matches C `ixmlGetIntegerValue`.
pub unsafe extern "C" fn ixml_get_integer_value(xml_ind: i32, node_ind: i32, val: *mut i32) -> i32 {
    let mut string = core::ptr::null();
    let mut num = 1;
    let err = get_element_string(xml_ind, node_ind, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(string, string, val.cast(), PIP_INTEGER, &mut num, 1)
}
/// Matches C `ixmlGetFloatValue`.
pub unsafe extern "C" fn ixml_get_float_value(xml_ind: i32, node_ind: i32, val: *mut f32) -> i32 {
    let mut string = core::ptr::null();
    let mut num = 1;
    let err = get_element_string(xml_ind, node_ind, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(string, string, val.cast(), PIP_FLOAT, &mut num, 1)
}
/// Matches C `ixmlGetDoubleValue`.
pub unsafe extern "C" fn ixml_get_double_value(xml_ind: i32, node_ind: i32, val: *mut f64) -> i32 {
    let mut string = core::ptr::null();
    let mut num = 1;
    let err = get_element_string(xml_ind, node_ind, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(string, string, val.cast(), PIP_DOUBLE, &mut num, 1)
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
    if node.is_null() {
        return err;
    }
    let element = (*node).value.element;
    for ind in 0..element.num_attrs {
        let attr = element.attrs.add(ind as usize);
        if libc::strcmp((*attr).name, name) == 0 {
            *string = libc::strdup((*attr).value);
            if (*string).is_null() {
                return -1;
            }
            return 0;
        }
    }
    1
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
    if node.is_null() {
        return err;
    }
    let element = (*node).value.element;
    for ind in 0..element.num_attrs {
        let attr = element.attrs.add(ind as usize);
        if libc::strcmp((*attr).name, name) == 0 {
            return pip_get_line_of_values(
                (*attr).value,
                (*attr).value,
                val.cast(),
                PIP_INTEGER,
                &mut num,
                1,
            );
        }
    }
    1
}
/// Matches C `ixmlAddElement`.
pub unsafe extern "C" fn ixml_add_element(xml_ind: i32, node_ind: i32, tag: *const c_char) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_null() {
        return err;
    }
    let elem = mxml_new_element(node, tag);
    if elem.is_null() {
        return -1;
    }
    if ilist_append(
        *S_NODE_LISTS.add(xml_ind as usize),
        (&raw const elem).cast_mut().cast(),
    ) != 0
    {
        return -1;
    }
    ilist_size(*S_NODE_LISTS.add(xml_ind as usize)) - 1
}
/// Matches C `ixmlSetStringValue`.
pub unsafe extern "C" fn ixml_set_string_value(
    xml_ind: i32,
    node_ind: i32,
    string: *const c_char,
) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_null() {
        return err;
    }
    if mxml_new_text(node, 0, string).is_null() {
        err = -1;
    }
    err
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
    if node.is_null() {
        return err;
    }
    mxml_element_set_attr(node, name, value);
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
    if node.is_null() {
        return err;
    }
    let mut buffer = [0 as c_char; 64];
    libc::snprintf(buffer.as_mut_ptr(), buffer.len(), c"%d".as_ptr(), value);
    mxml_element_set_attr(node, name, buffer.as_ptr());
    0
}
/// Matches C `ixmlPopListIndexes`.
pub unsafe extern "C" fn ixml_pop_list_indexes(xml_ind: i32, first_ind: i32) -> i32 {
    if xml_ind < 0 || xml_ind >= S_NUM_LISTS || (*S_NODE_LISTS.add(xml_ind as usize)).is_null() {
        return -2;
    }
    if first_ind < 2 || first_ind >= ilist_size(*S_NODE_LISTS.add(xml_ind as usize)) {
        return -3;
    }
    ilist_truncate(*S_NODE_LISTS.add(xml_ind as usize), first_ind);
    0
}
/// Matches C `ixmlClear`.
pub unsafe extern "C" fn ixml_clear(index: i32) {
    if index < 0 && index >= S_NUM_LISTS {
        return;
    }
    let xml = ilist_item(*S_NODE_LISTS.add(index as usize), 0).cast::<*mut MxmlNode>();
    mxml_delete(*xml);
    ilist_delete(*S_NODE_LISTS.add(index as usize));
    *S_NODE_LISTS.add(index as usize) = core::ptr::null_mut();
}
/// Matches C `ixmlResetLastLevel`.
pub unsafe extern "C" fn ixml_reset_last_level() {
    S_LAST_LEVEL = -1;
}
/// Matches C `ixmlWhitespace_cb`.
pub unsafe extern "C" fn ixml_whitespace_cb(node_void: *mut c_void, where_: i32) -> *const c_char {
    let node = node_void.cast::<MxmlNode>();
    if where_ != MXML_WS_BEFORE_OPEN && where_ != MXML_WS_BEFORE_CLOSE {
        return core::ptr::null();
    }
    let mut level = -1;
    let mut parent = (*node).parent;
    while !parent.is_null() {
        level += 1;
        parent = (*parent).parent;
    }
    if level > 16 {
        level = 16;
    } else if level < 0 {
        level = 0;
    }
    if S_LAST_LEVEL < 0 {
        S_LAST_LEVEL = level;
        return core::ptr::null();
    }
    if level == S_LAST_LEVEL && where_ == MXML_WS_BEFORE_CLOSE {
        return core::ptr::null();
    }
    S_LAST_LEVEL = level;
    let mut spaces = [b' ' as c_char; 33];
    spaces[32] = 0;
    libc::snprintf(
        (&raw mut S_WHITESPACE_BUFFER).cast::<c_char>(),
        36,
        c"\n%s".as_ptr(),
        spaces.as_ptr().add((32 - 2 * level) as usize),
    );
    (&raw const S_WHITESPACE_BUFFER).cast::<c_char>()
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
