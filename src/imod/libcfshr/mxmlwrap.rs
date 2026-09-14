//! Translation of `IMOD/libcfshr/mxmlwrap.c` and `IMOD/include/mxmlwrap.h`.

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use crate::imod::libcfshr::parse_params::{
    PIP_DOUBLE, PIP_FLOAT, PIP_INTEGER, pip_get_line_of_values,
};
use crate::imod::libxml::*;
use core::cell::{Cell, RefCell};
thread_local! {
    /// The C stores node pointers in `sNodeLists`; the Rust XML arena uses
    /// stable slot indices, so each document keeps its indexed node list here.
    static S_NODE_LISTS: RefCell<Vec<Option<Vec<usize>>>> = const { RefCell::new(Vec::new()) };
    /// Each indexed node list owns its matching XML arena at the same `xmlInd`.
    static S_ARENAS: RefCell<Vec<Option<MxmlArena>>> = const { RefCell::new(Vec::new()) };
    /// Matches C static `sLastLevel`.
    static S_LAST_LEVEL: Cell<i32> = const { Cell::new(-1) };
}

/// Matches C static `getOrAddFreeList`.
pub fn get_or_add_free_list() -> i32 {
    let xml_ind = S_NODE_LISTS.with_borrow_mut(|lists| {
        if let Some(index) = lists.iter().position(Option::is_none) {
            lists[index] = Some(Vec::new());
            return Some(index);
        }
        if lists.try_reserve(1).is_err() {
            return None;
        }
        lists.push(Some(Vec::new()));
        Some(lists.len() - 1)
    });
    let Some(xml_ind) = xml_ind else {
        return -1;
    };
    let arenas_ready = S_ARENAS.with_borrow_mut(|arenas| {
        if arenas.len() <= xml_ind && arenas.try_reserve(xml_ind + 1 - arenas.len()).is_err() {
            return false;
        }
        while arenas.len() <= xml_ind {
            arenas.push(None);
        }
        true
    });
    if !arenas_ready {
        S_NODE_LISTS.with_borrow_mut(|lists| lists[xml_ind] = None);
        return -1;
    }
    xml_ind as i32
}

/// Matches C static `getNodeAtIndex`.
///
/// The C returns the `mxml_node_t *` stored in the list; the list now holds
/// arena slot indices, so this returns one of those.
pub fn get_node_at_index(xml_ind: i32, node_ind: i32, error: &mut i32) -> Option<usize> {
    if xml_ind < 0 {
        *error = -2;
        return None;
    }
    let node = S_NODE_LISTS.with_borrow(|lists| {
        lists
            .get(xml_ind as usize)
            .and_then(|list| list.as_ref())
            .and_then(|list| list.get(node_ind.max(0) as usize))
            .copied()
    });
    if node.is_none()
        && S_NODE_LISTS.with_borrow(|lists| {
            lists
                .get(xml_ind as usize)
                .is_none_or(|list| list.is_none())
        })
    {
        *error = -2;
        return None;
    }
    if node_ind < 0 || node.is_none() {
        *error = -3;
        return None;
    }
    *error = 0;
    node
}

/// Matches C static `getElementString`.
///
/// The C hands back a pointer into the node's opaque value; the bytes are
/// copied into `string` with a terminating NUL instead, because the callers
/// pass them on to `pipGetLineOfValues`, which is still C-shaped.
pub fn get_element_string(xml_ind: i32, node_ind: i32, string: &mut Vec<u8>) -> i32 {
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
pub fn process_loaded_nodes(
    mut arena: MxmlArena,
    xml: Option<usize>,
    with_xml_decl: i32,
    root_element: &mut Option<Vec<u8>>,
) -> i32 {
    let xml_ind = get_or_add_free_list();
    if xml_ind < 0 {
        mxml_delete(&mut arena, xml);
        return xml_ind;
    }
    S_NODE_LISTS.with_borrow_mut(|lists| {
        lists[xml_ind as usize]
            .as_mut()
            .unwrap()
            .push(xml.unwrap_or(usize::MAX));
    });
    let mut top = xml;
    if with_xml_decl != 0 {
        top = mxml_walk_next(&arena, xml, xml, MXML_DESCEND);
        if mxml_get_type(&arena, top) == MXML_OPAQUE {
            top = mxml_walk_next(&arena, top, xml, MXML_DESCEND);
        }
        S_NODE_LISTS.with_borrow_mut(|lists| {
            lists[xml_ind as usize]
                .as_mut()
                .unwrap()
                .push(top.unwrap_or(usize::MAX));
        });
    }
    *root_element = None;
    if let Some(root) = mxml_get_element(&arena, top) {
        // `strdup(root)`: the caller owns the copy, which is the `Vec` now.
        *root_element = Some(root.to_vec());
    }
    S_ARENAS.with_borrow_mut(|arenas| arenas[xml_ind as usize] = Some(arena));
    xml_ind
}

/// Matches C `ixmlReadFile`.
pub fn ixml_read_file(
    filename: &[u8],
    with_xml_decl: i32,
    root_element: &mut Option<Vec<u8>>,
) -> i32 {
    let path = String::from_utf8_lossy(filename).into_owned();
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
pub fn ixml_load_string(
    string: &[u8],
    with_xml_decl: i32,
    root_element: &mut Option<Vec<u8>>,
) -> i32 {
    let mut arena = MxmlArena::new();
    let xml = mxml_load_string(&mut arena, MXML_NO_PARENT, string, Some(mxml_opaque_cb));
    if xml.is_none() {
        return -3;
    }
    process_loaded_nodes(arena, xml, with_xml_decl, root_element)
}

/// Matches C `ixmlNewNodeList`.
pub fn ixml_new_node_list(root_element: &[u8]) -> i32 {
    let mut xml_ind = get_or_add_free_list();
    if xml_ind >= 0 {
        let mut arena = MxmlArena::new();
        let xml = mxml_new_xml(&mut arena, Some(b"1.0"));
        if xml.is_none() {
            xml_ind = -2;
        } else {
            S_NODE_LISTS.with_borrow_mut(|lists| {
                lists[xml_ind as usize].as_mut().unwrap().push(xml.unwrap());
            });
            let top = mxml_new_element(&mut arena, xml, Some(root_element));
            if top.is_none() {
                xml_ind = -2;
            } else {
                S_NODE_LISTS.with_borrow_mut(|lists| {
                    lists[xml_ind as usize].as_mut().unwrap().push(top.unwrap());
                });
            }
        }
        if xml_ind >= 0 {
            S_ARENAS.with_borrow_mut(|arenas| arenas[xml_ind as usize] = Some(arena));
        }
    }
    xml_ind
}

/// Matches C `ixmlWriteFile`.
pub fn ixml_write_file(xml_ind: i32, filename: &[u8]) -> i32 {
    let mut err = 0;
    let xml = get_node_at_index(xml_ind, 0, &mut err);
    if xml.is_none() {
        return err;
    }
    let path = String::from_utf8_lossy(filename).into_owned();
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
pub fn ixml_find_elements(
    xml_ind: i32,
    node_ind: i32,
    tag: &[u8],
    found_ind: &mut i32,
    num_found: &mut i32,
) -> i32 {
    let mut err = 0;
    *num_found = 0;
    let parent = get_node_at_index(xml_ind, node_ind, &mut err);
    if parent.is_none() {
        return err;
    }
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
    *found_ind =
        S_NODE_LISTS.with_borrow(|lists| lists[xml_ind as usize].as_ref().unwrap().len() as i32);
    for node in nodes {
        if S_NODE_LISTS.with_borrow_mut(|lists| {
            let list = lists[xml_ind as usize].as_mut().unwrap();
            if list.try_reserve(1).is_err() {
                true
            } else {
                list.push(node);
                false
            }
        }) {
            return -1;
        }
        *num_found += 1;
    }
    0
}

/// Matches C `ixmlGetStringValue`.
pub fn ixml_get_string_value(xml_ind: i32, node_ind: i32, string: &mut Vec<u8>) -> i32 {
    let mut value: Vec<u8> = Vec::new();
    let err = get_element_string(xml_ind, node_ind, &mut value);
    if err != 0 {
        return err;
    }
    // `*string = strdup(value)`: the caller owns the copy, which is its `Vec`.
    *string = value;
    0
}

/// Matches C `ixmlGetIntegerValue`.
pub fn ixml_get_integer_value(xml_ind: i32, node_ind: i32, val: &mut i32) -> i32 {
    let mut string: Vec<u8> = Vec::new();
    let mut num = 1;
    let err = get_element_string(xml_ind, node_ind, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(
        &string,
        &string,
        crate::imod::libcfshr::parse_params::PipValueArray::Int(core::slice::from_mut(val)),
        PIP_INTEGER,
        &mut num,
        1,
    )
}

/// Matches C `ixmlGetFloatValue`.
pub fn ixml_get_float_value(xml_ind: i32, node_ind: i32, val: &mut f32) -> i32 {
    let mut string: Vec<u8> = Vec::new();
    let mut num = 1;
    let err = get_element_string(xml_ind, node_ind, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(
        &string,
        &string,
        crate::imod::libcfshr::parse_params::PipValueArray::Float(core::slice::from_mut(val)),
        PIP_FLOAT,
        &mut num,
        1,
    )
}

/// Matches C `ixmlGetDoubleValue`.
pub fn ixml_get_double_value(xml_ind: i32, node_ind: i32, val: &mut f64) -> i32 {
    let mut string: Vec<u8> = Vec::new();
    let mut num = 1;
    let err = get_element_string(xml_ind, node_ind, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(
        &string,
        &string,
        crate::imod::libcfshr::parse_params::PipValueArray::Double(core::slice::from_mut(val)),
        PIP_DOUBLE,
        &mut num,
        1,
    )
}

/// Matches C `ixmlGetStringAttribute`.
pub fn ixml_get_string_attribute(
    xml_ind: i32,
    node_ind: i32,
    name: &[u8],
    string: &mut Vec<u8>,
) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    let mut value: Option<Vec<u8>> = None;
    S_ARENAS.with_borrow(|arenas| {
        let Some(arena) = arenas.get(xml_ind as usize).and_then(|a| a.as_ref()) else {
            return;
        };
        let MxmlValue::Element(element) = &arena.node(node.unwrap()).value else {
            return;
        };
        for attr in &element.attrs {
            if attr.name == name {
                value = Some(attr.value.clone().unwrap_or_default());
                return;
            }
        }
    });
    let Some(value) = value else {
        return 1;
    };
    // `*string = strdup(value)`: the caller owns the copy.
    *string = value;
    0
}

/// Matches C `ixmlGetIntegerAttribute`.
pub fn ixml_get_integer_attribute(xml_ind: i32, node_ind: i32, name: &[u8], val: &mut i32) -> i32 {
    let mut err = 0;
    let mut num = 1;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    let mut value: Option<Vec<u8>> = None;
    S_ARENAS.with_borrow(|arenas| {
        let Some(arena) = arenas.get(xml_ind as usize).and_then(|a| a.as_ref()) else {
            return;
        };
        let MxmlValue::Element(element) = &arena.node(node.unwrap()).value else {
            return;
        };
        for attr in &element.attrs {
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
        crate::imod::libcfshr::parse_params::PipValueArray::Int(core::slice::from_mut(val)),
        PIP_INTEGER,
        &mut num,
        1,
    )
}

/// Matches C `ixmlAddElement`.
pub fn ixml_add_element(xml_ind: i32, node_ind: i32, tag: &[u8]) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    let elem = S_ARENAS.with_borrow_mut(|arenas| {
        let Some(arena) = arenas.get_mut(xml_ind as usize).and_then(|a| a.as_mut()) else {
            return None;
        };
        mxml_new_element(arena, node, Some(tag))
    });
    let Some(elem) = elem else {
        return -1;
    };
    if S_NODE_LISTS.with_borrow_mut(|lists| {
        let list = lists[xml_ind as usize].as_mut().unwrap();
        if list.try_reserve(1).is_err() {
            true
        } else {
            list.push(elem);
            false
        }
    }) {
        return -1;
    }
    S_NODE_LISTS.with_borrow(|lists| lists[xml_ind as usize].as_ref().unwrap().len() as i32 - 1)
}

/// Matches C `ixmlSetStringValue`.
pub fn ixml_set_string_value(xml_ind: i32, node_ind: i32, string: &[u8]) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
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
pub fn ixml_set_integer_value(xml_ind: i32, node_ind: i32, val: i32) -> i32 {
    // `char buf[64]; snprintf(buf, 64, "%d", val);`
    let buffer = c_format_bytes("%d", &[CArg::Int(val as i64)]);
    ixml_set_string_value(xml_ind, node_ind, &buffer)
}

/// Matches C `ixmlSetFloatValue`.
pub fn ixml_set_float_value(xml_ind: i32, node_ind: i32, val: f32) -> i32 {
    let buffer = c_format_bytes("%g", &[CArg::Dbl(val as f64)]);
    ixml_set_string_value(xml_ind, node_ind, &buffer)
}

/// Matches C `ixmlSetDoubleValue`.
pub fn ixml_set_double_value(xml_ind: i32, node_ind: i32, val: f64) -> i32 {
    let buffer = c_format_bytes("%g", &[CArg::Dbl(val)]);
    ixml_set_string_value(xml_ind, node_ind, &buffer)
}

/// Matches C `ixmlAddStringAttribute`.
pub fn ixml_add_string_attribute(
    xml_ind: i32,
    node_ind: i32,
    name: &[u8],
    value: Option<&[u8]>,
) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    S_ARENAS.with_borrow_mut(|arenas| {
        if let Some(arena) = arenas.get_mut(xml_ind as usize).and_then(|a| a.as_mut()) {
            mxml_element_set_attr(arena, node, Some(name), value);
        }
    });
    0
}

/// Matches C `ixmlAddIntegerAttribute`.
pub fn ixml_add_integer_attribute(xml_ind: i32, node_ind: i32, name: &[u8], value: i32) -> i32 {
    let mut err = 0;
    let node = get_node_at_index(xml_ind, node_ind, &mut err);
    if node.is_none() {
        return err;
    }
    // `char buf[64]; snprintf(buf, 64, "%d", value);`
    let text = c_format_bytes("%d", &[CArg::Int(value as i64)]);
    S_ARENAS.with_borrow_mut(|arenas| {
        if let Some(arena) = arenas.get_mut(xml_ind as usize).and_then(|a| a.as_mut()) {
            mxml_element_set_attr(arena, node, Some(name), Some(&text));
        }
    });
    0
}

/// Matches C `ixmlPopListIndexes`.
pub fn ixml_pop_list_indexes(xml_ind: i32, first_ind: i32) -> i32 {
    if xml_ind < 0
        || S_NODE_LISTS.with_borrow(|lists| {
            lists
                .get(xml_ind as usize)
                .is_none_or(|list| list.is_none())
        })
    {
        return -2;
    }
    if first_ind < 2
        || S_NODE_LISTS.with_borrow(|lists| {
            first_ind as usize >= lists[xml_ind as usize].as_ref().unwrap().len()
        })
    {
        return -3;
    }
    S_NODE_LISTS.with_borrow_mut(|lists| {
        lists[xml_ind as usize]
            .as_mut()
            .unwrap()
            .truncate(first_ind as usize);
    });
    0
}

/// Matches C `ixmlClear`.
pub fn ixml_clear(index: i32) {
    if index < 0
        || S_NODE_LISTS
            .with_borrow(|lists| lists.get(index as usize).is_none_or(|list| list.is_none()))
    {
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
    S_NODE_LISTS.with_borrow_mut(|lists| lists[index as usize] = None);
}

/// Matches C `ixmlResetLastLevel`.
pub fn ixml_reset_last_level() {
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
        let mut root = None;
        let xml = b"<Root><Item number=\"12\">3.5</Item><Item number=\"13\">7</Item></Root>";
        let index = ixml_load_string(xml, 0, &mut root);
        assert!(index >= 0);
        assert_eq!(root.as_deref(), Some(&b"Root"[..]));
        let mut found = -1;
        let mut number = -1;
        assert_eq!(
            ixml_find_elements(index, 0, b"Item", &mut found, &mut number),
            0
        );
        assert_eq!(number, 2);
        let mut value = 0.0f32;
        assert_eq!(ixml_get_float_value(index, found, &mut value), 0);
        assert_eq!(value, 3.5);
        let mut double_value = 0.0f64;
        assert_eq!(ixml_get_double_value(index, found, &mut double_value), 0);
        assert_eq!(double_value, 3.5);
        let mut integer_value = 0;
        assert_eq!(
            ixml_get_integer_value(index, found + 1, &mut integer_value),
            0
        );
        assert_eq!(integer_value, 7);
        let mut attribute = 0;
        assert_eq!(
            ixml_get_integer_attribute(index, found, b"number", &mut attribute),
            0
        );
        assert_eq!(attribute, 12);
        ixml_clear(index);
    }

    #[test]
    fn new_list_adds_nodes_and_source_error_indexes() {
        let _lock = TEST_LOCK.lock().unwrap();
        let index = ixml_new_node_list(b"Root");
        assert!(index >= 0);
        let element = ixml_add_element(index, 1, b"Child");
        assert_eq!(element, 2);
        assert_eq!(ixml_set_integer_value(index, element, 41), 0);
        assert_eq!(ixml_add_integer_attribute(index, element, b"count", 4), 0);
        let mut count = 0;
        assert_eq!(
            ixml_get_integer_attribute(index, element, b"count", &mut count),
            0
        );
        assert_eq!(count, 4);
        assert_eq!(ixml_pop_list_indexes(index, 1), -3);
        assert_eq!(ixml_pop_list_indexes(index, 2), 0);
        assert_eq!(ixml_add_element(-1, 0, b"x"), -2);
        ixml_clear(index);
    }
}
