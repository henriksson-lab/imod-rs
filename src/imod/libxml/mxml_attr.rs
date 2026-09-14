//! Translation of `IMOD/libxml/mxml-attr.c`.
#![allow(dead_code)]

use super::*;
use core::ffi::c_int;

/// Matches C `mxmlElementDeleteAttr` (`mxml-attr.c:38`).
pub fn mxml_element_delete_attr(arena: &mut MxmlArena, node: Option<usize>, name: Option<&[u8]>) {
    let mut i: c_int;
    let mut attr: usize;

    /*
     * Range check input...
     */

    let Some(node) = node else {
        return;
    };
    if arena.node(node).type_ != MXML_ELEMENT {
        return;
    }
    let Some(name) = name else {
        return;
    };
    let MxmlValue::Element(element) = &mut arena.node_mut(node).value else {
        return;
    };

    /*
     * Look for the attribute...
     */

    i = element.num_attrs;
    attr = 0;
    while i > 0 {
        if element.attrs[attr].name == name {
            /*
             * Delete this attribute...  (the C frees the name and value and
             * then memmoves the tail of the array down over the slot.)
             */

            element.attrs.remove(attr);

            element.num_attrs -= 1;

            /* The C frees the whole array when the last attribute goes. */
            return;
        }

        i -= 1;
        attr += 1;
    }
}

/// Matches C `mxmlElementGetAttr` (`mxml-attr.c:96`).
pub fn mxml_element_get_attr<'a>(
    arena: &'a MxmlArena,
    node: Option<usize>,
    name: Option<&[u8]>,
) -> Option<&'a [u8]> {
    let mut i: c_int;
    let mut attr: usize;

    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };
    if arena.node(node).type_ != MXML_ELEMENT {
        return None;
    }
    let Some(name) = name else {
        return None;
    };
    let MxmlValue::Element(element) = &arena.node(node).value else {
        return None;
    };

    /*
     * Look for the attribute...
     */

    i = element.num_attrs;
    attr = 0;
    while i > 0 {
        if element.attrs[attr].name == name {
            return element.attrs[attr].value.as_deref();
        }

        i -= 1;
        attr += 1;
    }

    /*
     * Didn't find attribute, so return NULL...
     */

    None
}

/// Matches C `mxmlElementSetAttr` (`mxml-attr.c:154`).
pub fn mxml_element_set_attr(
    arena: &mut MxmlArena,
    node: Option<usize>,
    name: Option<&[u8]>,
    value: Option<&[u8]>,
) {
    let valuec: Option<Vec<u8>>;

    /*
     * Range check input...
     */

    let Some(node) = node else {
        return;
    };
    if arena.node(node).type_ != MXML_ELEMENT {
        return;
    }
    let Some(name) = name else {
        return;
    };

    if let Some(value) = value {
        valuec = Some(value.to_vec());
    } else {
        valuec = None;
    }

    if mxml_set_attr(arena, node, name, valuec) != 0 {
        /* The C frees `valuec` here; the owned Vec was moved in and dropped. */
    }
}

/// Matches C `mxmlElementSetAttrf` (`mxml-attr.c:194`).
///
/// The C function is `mxmlElementSetAttrf(node, name, format, ...)`.  Stable
/// Rust cannot define a C-variadic function, so the single variable argument is
/// taken explicitly; `_mxml_strdupf` carries the same restriction.  The only
/// call in IMOD is `mxmlElementSetAttrf(node, name, "%d", value)` in
/// `libcfshr/mxmlwrap.c:375`.
pub fn mxml_element_set_attrf(
    arena: &mut MxmlArena,
    node: Option<usize>,
    name: Option<&[u8]>,
    format: Option<&[u8]>,
    arg: &[u8],
) {
    let value: Vec<u8>;

    /*
     * Range check input...
     */

    let Some(node) = node else {
        return;
    };
    if arena.node(node).type_ != MXML_ELEMENT {
        return;
    }
    let Some(name) = name else {
        return;
    };
    let Some(format) = format else {
        return;
    };

    /*
     * Format the value...
     */

    value = _mxml_strdupf(format, arg);

    /*
     * The C reports "Unable to allocate memory for attribute '%s' in element
     * %s!" when the format allocation returns NULL; a Vec aborts instead.
     */
    if mxml_set_attr(arena, node, name, Some(value)) != 0 {
        /* The C frees `value` here; the owned Vec was moved in and dropped. */
    }
}

/// Matches C static `mxml_set_attr` (`mxml-attr.c:234`).
///
/// The C takes ownership of `value` on success and leaves it to the caller to
/// free on failure; the owned `Option<Vec<u8>>` carries that transfer, and the
/// two failure arms — `realloc` and `strdup` returning NULL — cannot occur.
pub fn mxml_set_attr(
    arena: &mut MxmlArena,
    node: usize,
    name: &[u8],
    value: Option<Vec<u8>>,
) -> c_int {
    let mut i: c_int;
    let mut attr: usize;

    let MxmlValue::Element(element) = &mut arena.node_mut(node).value else {
        return -1;
    };

    /*
     * Look for the attribute...
     */

    i = element.num_attrs;
    attr = 0;
    while i > 0 {
        if element.attrs[attr].name == name {
            /*
             * Free the old value as needed...
             */

            element.attrs[attr].value = value;

            return 0;
        }

        i -= 1;
        attr += 1;
    }

    /*
     * Add a new attribute...
     */

    element.attrs.push(MxmlAttr {
        name: name.to_vec(),
        value,
    });

    element.num_attrs += 1;

    0
}
