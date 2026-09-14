//! Translation of `IMOD/libxml/mxml-set.c`.
#![allow(dead_code)]

use super::*;
use core::any::Any;
use core::ffi::c_int;

/// Matches C `mxmlSetCDATA` (`mxml-set.c:32`).
pub fn mxml_set_cdata(arena: &mut MxmlArena, node: Option<usize>, data: Option<&[u8]>) -> c_int {
    /*
     * Range check input...
     */

    let mut node = node;

    if let Some(n) = node
        && arena.node(n).type_ == MXML_ELEMENT
        && match &arena.node(n).value {
            MxmlValue::Element(element) => !element
                .name
                .as_deref()
                .is_some_and(|name| name.starts_with(b"![CDATA[")),
            _ => false,
        }
        && let Some(child) = arena.node(n).child
        && arena.node(child).type_ == MXML_ELEMENT
        && match &arena.node(child).value {
            MxmlValue::Element(element) => element
                .name
                .as_deref()
                .is_some_and(|name| name.starts_with(b"![CDATA[")),
            _ => false,
        }
    {
        node = Some(child);
    }

    let Some(node) = node else {
        return -1;
    };
    if arena.node(node).type_ != MXML_ELEMENT {
        return -1;
    }
    let Some(data) = data else {
        return -1;
    };
    let MxmlValue::Element(element) = &mut arena.node_mut(node).value else {
        return -1;
    };
    if !element
        .name
        .as_deref()
        .is_some_and(|name| name.starts_with(b"![CDATA["))
    {
        return -1;
    }

    /*
     * Free any old element value and set the new value...
     */

    element.name = Some(_mxml_strdupf(b"![CDATA[%s]]", data));

    0
}

/// Matches C `mxmlSetCustom` (`mxml-set.c:70`).
pub fn mxml_set_custom(
    arena: &mut MxmlArena,
    node: Option<usize>,
    data: Option<Box<dyn Any>>,
    destroy: MxmlCustomDestroyCb,
) -> c_int {
    /*
     * Range check input...
     */

    let mut node = node;

    if let Some(n) = node
        && arena.node(n).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(n).child
        && arena.node(child).type_ == MXML_CUSTOM
    {
        node = Some(child);
    }

    let Some(node) = node else {
        return -1;
    };
    if arena.node(node).type_ != MXML_CUSTOM {
        return -1;
    }

    /*
     * Free any old element value and set the new value...
     */

    let MxmlValue::Custom(custom) = &mut arena.node_mut(node).value else {
        return -1;
    };
    if let Some(old) = custom.data.as_deref_mut()
        && let Some(f) = custom.destroy
    {
        f(old);
    }

    custom.data = data;
    custom.destroy = destroy;

    0
}

/// Matches C `mxmlSetElement` (`mxml-set.c:104`).
pub fn mxml_set_element(arena: &mut MxmlArena, node: Option<usize>, name: Option<&[u8]>) -> c_int {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return -1;
    };
    if arena.node(node).type_ != MXML_ELEMENT {
        return -1;
    }
    let Some(name) = name else {
        return -1;
    };

    /*
     * Free any old element value and set the new value...
     */

    let MxmlValue::Element(element) = &mut arena.node_mut(node).value else {
        return -1;
    };
    element.name = Some(name.to_vec());

    0
}

/// Matches C `mxmlSetInteger` (`mxml-set.c:130`).
pub fn mxml_set_integer(arena: &mut MxmlArena, node: Option<usize>, integer: c_int) -> c_int {
    /*
     * Range check input...
     */

    let mut node = node;

    if let Some(n) = node
        && arena.node(n).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(n).child
        && arena.node(child).type_ == MXML_INTEGER
    {
        node = Some(child);
    }

    let Some(node) = node else {
        return -1;
    };
    if arena.node(node).type_ != MXML_INTEGER {
        return -1;
    }

    /*
     * Set the new value and return...
     */

    arena.node_mut(node).value = MxmlValue::Integer(integer);

    0
}

/// Matches C `mxmlSetOpaque` (`mxml-set.c:158`).
pub fn mxml_set_opaque(arena: &mut MxmlArena, node: Option<usize>, opaque: Option<&[u8]>) -> c_int {
    /*
     * Range check input...
     */

    let mut node = node;

    if let Some(n) = node
        && arena.node(n).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(n).child
        && arena.node(child).type_ == MXML_OPAQUE
    {
        node = Some(child);
    }

    let Some(node) = node else {
        return -1;
    };
    if arena.node(node).type_ != MXML_OPAQUE {
        return -1;
    }
    let Some(opaque) = opaque else {
        return -1;
    };

    /*
     * Free any old opaque value and set the new value...
     */

    arena.node_mut(node).value = MxmlValue::Opaque(Some(opaque.to_vec()));

    0
}

/// Matches C `mxmlSetReal` (`mxml-set.c:190`).
pub fn mxml_set_real(arena: &mut MxmlArena, node: Option<usize>, real: f64) -> c_int {
    /*
     * Range check input...
     */

    let mut node = node;

    if let Some(n) = node
        && arena.node(n).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(n).child
        && arena.node(child).type_ == MXML_REAL
    {
        node = Some(child);
    }

    let Some(node) = node else {
        return -1;
    };
    if arena.node(node).type_ != MXML_REAL {
        return -1;
    }

    /*
     * Set the new value and return...
     */

    arena.node_mut(node).value = MxmlValue::Real(real);

    0
}

/// Matches C `mxmlSetText` (`mxml-set.c:218`).
pub fn mxml_set_text(
    arena: &mut MxmlArena,
    node: Option<usize>,
    whitespace: c_int,
    string: Option<&[u8]>,
) -> c_int {
    /*
     * Range check input...
     */

    let mut node = node;

    if let Some(n) = node
        && arena.node(n).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(n).child
        && arena.node(child).type_ == MXML_TEXT
    {
        node = Some(child);
    }

    let Some(node) = node else {
        return -1;
    };
    if arena.node(node).type_ != MXML_TEXT {
        return -1;
    }
    let Some(string) = string else {
        return -1;
    };

    /*
     * Free any old string value and set the new value...
     */

    let MxmlValue::Text(text) = &mut arena.node_mut(node).value else {
        return -1;
    };
    text.whitespace = whitespace;
    text.string = Some(string.to_vec());

    0
}

/// Matches C `mxmlSetTextf` (`mxml-set.c:255`).
///
/// The C function is `mxmlSetTextf(node, whitespace, format, ...)` and hands
/// the started `va_list` to `_mxml_strdupf`, which expects a variable argument
/// list rather than a `va_list` — the mismatch is in the vendored source.  The
/// shape is preserved here with the single explicit argument that stable Rust
/// allows; see `_mxml_strdupf`.
pub fn mxml_set_textf(
    arena: &mut MxmlArena,
    node: Option<usize>,
    whitespace: c_int,
    format: Option<&[u8]>,
    arg: &[u8],
) -> c_int {
    /*
     * Range check input...
     */

    let mut node = node;

    if let Some(n) = node
        && arena.node(n).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(n).child
        && arena.node(child).type_ == MXML_TEXT
    {
        node = Some(child);
    }

    let Some(node) = node else {
        return -1;
    };
    if arena.node(node).type_ != MXML_TEXT {
        return -1;
    }
    let Some(format) = format else {
        return -1;
    };

    /*
     * Free any old string value and set the new value...
     */

    let string = _mxml_strdupf(format, arg);
    let MxmlValue::Text(text) = &mut arena.node_mut(node).value else {
        return -1;
    };
    text.whitespace = whitespace;
    text.string = Some(string);

    0
}

/// Matches C `mxmlSetUserData` (`mxml-set.c:299`).
pub fn mxml_set_user_data(
    arena: &mut MxmlArena,
    node: Option<usize>,
    data: Option<Box<dyn Any>>,
) -> c_int {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return -1;
    };

    /*
     * Set the user data pointer and return...
     */

    arena.node_mut(node).user_data = data;

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `mxmlSetXxx` first descends from an element into a matching child
    /// (`mxml-set.c:141` and friends), so setting text on the element that owns
    /// a text node succeeds and rewrites the child.  The native
    /// `libimxml.so` returns 0 and reads back "bye".
    ///
    /// The tree the reference parsed from `<root>hello</root>` is built here
    /// with the node API instead: `mxml_file`'s parser is a later conversion
    /// wave, and the shape it produces for that document with no load callback
    /// is exactly an element with one whitespace-free text child.
    #[test]
    fn set_text_descends_from_the_element_into_its_child() {
        let arena = &mut MxmlArena::new();
        let tree = mxml_new_element(arena, MXML_NO_PARENT, Some(b"root"));
        assert!(tree.is_some());
        mxml_new_text(arena, tree, 0, Some(b"hello"));
        assert_eq!(mxml_get_type(arena, tree), MXML_ELEMENT);
        assert_eq!(mxml_set_text(arena, tree, 0, Some(b"bye")), 0);
        let mut ws: c_int = 0;
        assert_eq!(
            mxml_get_text(arena, tree, Some(&mut ws)),
            Some(b"bye".as_slice())
        );
        mxml_delete(arena, tree);
    }
}
