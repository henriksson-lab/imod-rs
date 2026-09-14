//! Translation of `IMOD/libxml/mxml-get.c`.
#![allow(dead_code)]

use super::*;
use core::any::Any;
use core::ffi::c_int;

/// Matches C `mxmlGetCDATA` (`mxml-get.c:33`).
pub fn mxml_get_cdata(arena: &MxmlArena, node: Option<usize>) -> Option<&[u8]> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };
    if arena.node(node).type_ != MXML_ELEMENT {
        return None;
    }
    let MxmlValue::Element(element) = &arena.node(node).value else {
        return None;
    };
    let name = element.name.as_deref()?;
    if !name.starts_with(b"![CDATA[") {
        return None;
    }

    /*
     * Return the text following the CDATA declaration...
     */

    Some(&name[8..])
}

/// Matches C `mxmlGetCustom` (`mxml-get.c:59`).
pub fn mxml_get_custom(arena: &MxmlArena, node: Option<usize>) -> Option<&dyn Any> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };

    /*
     * Return the custom value...
     */

    if arena.node(node).type_ == MXML_CUSTOM {
        match &arena.node(node).value {
            MxmlValue::Custom(custom) => custom.data.as_deref(),
            _ => None,
        }
    } else if arena.node(node).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(node).child
        && arena.node(child).type_ == MXML_CUSTOM
    {
        match &arena.node(child).value {
            MxmlValue::Custom(custom) => custom.data.as_deref(),
            _ => None,
        }
    } else {
        None
    }
}

/// Matches C `mxmlGetElement` (`mxml-get.c:88`).
pub fn mxml_get_element(arena: &MxmlArena, node: Option<usize>) -> Option<&[u8]> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };
    if arena.node(node).type_ != MXML_ELEMENT {
        return None;
    }

    /*
     * Return the element name...
     */

    match &arena.node(node).value {
        MxmlValue::Element(element) => element.name.as_deref(),
        _ => None,
    }
}

/// Matches C `mxmlGetFirstChild` (`mxml-get.c:111`).
pub fn mxml_get_first_child(arena: &MxmlArena, node: Option<usize>) -> Option<usize> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };
    if arena.node(node).type_ != MXML_ELEMENT {
        return None;
    }

    /*
     * Return the first child node...
     */

    arena.node(node).child
}

/// Matches C `mxmlGetInteger` (`mxml-get.c:136`).
pub fn mxml_get_integer(arena: &MxmlArena, node: Option<usize>) -> c_int {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return 0;
    };

    /*
     * Return the integer value...
     */

    if arena.node(node).type_ == MXML_INTEGER {
        match &arena.node(node).value {
            MxmlValue::Integer(integer) => *integer,
            _ => 0,
        }
    } else if arena.node(node).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(node).child
        && arena.node(child).type_ == MXML_INTEGER
    {
        match &arena.node(child).value {
            MxmlValue::Integer(integer) => *integer,
            _ => 0,
        }
    } else {
        0
    }
}

/// Matches C `mxmlGetLastChild` (`mxml-get.c:165`).
pub fn mxml_get_last_child(arena: &MxmlArena, node: Option<usize>) -> Option<usize> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };
    if arena.node(node).type_ != MXML_ELEMENT {
        return None;
    }

    /*
     * Return the node type...
     */

    arena.node(node).last_child
}

/// Matches C `mxmlGetNextSibling` (`mxml-get.c:186`).
pub fn mxml_get_next_sibling(arena: &MxmlArena, node: Option<usize>) -> Option<usize> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };

    /*
     * Return the node type...
     */

    arena.node(node).next
}

/// Matches C `mxmlGetOpaque` (`mxml-get.c:210`).
pub fn mxml_get_opaque(arena: &MxmlArena, node: Option<usize>) -> Option<&[u8]> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };

    /*
     * Return the opaque value...
     */

    if arena.node(node).type_ == MXML_OPAQUE {
        match &arena.node(node).value {
            MxmlValue::Opaque(opaque) => opaque.as_deref(),
            _ => None,
        }
    } else if arena.node(node).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(node).child
        && arena.node(child).type_ == MXML_OPAQUE
    {
        match &arena.node(child).value {
            MxmlValue::Opaque(opaque) => opaque.as_deref(),
            _ => None,
        }
    } else {
        None
    }
}

/// Matches C `mxmlGetParent` (`mxml-get.c:239`).
pub fn mxml_get_parent(arena: &MxmlArena, node: Option<usize>) -> Option<usize> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };

    /*
     * Return the parent node...
     */

    arena.node(node).parent
}

/// Matches C `mxmlGetPrevSibling` (`mxml-get.c:262`).
pub fn mxml_get_prev_sibling(arena: &MxmlArena, node: Option<usize>) -> Option<usize> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };

    /*
     * Return the previous sibling node...
     */

    arena.node(node).prev
}

/// Matches C `mxmlGetReal` (`mxml-get.c:287`).
pub fn mxml_get_real(arena: &MxmlArena, node: Option<usize>) -> f64 {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return 0.0;
    };

    /*
     * Return the real value...
     */

    if arena.node(node).type_ == MXML_REAL {
        match &arena.node(node).value {
            MxmlValue::Real(real) => *real,
            _ => 0.0,
        }
    } else if arena.node(node).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(node).child
        && arena.node(child).type_ == MXML_REAL
    {
        match &arena.node(child).value {
            MxmlValue::Real(real) => *real,
            _ => 0.0,
        }
    } else {
        0.0
    }
}

/// Matches C `mxmlGetText` (`mxml-get.c:320`).
pub fn mxml_get_text<'a>(
    arena: &'a MxmlArena,
    node: Option<usize>,
    whitespace: Option<&mut c_int>,
) -> Option<&'a [u8]> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        if let Some(whitespace) = whitespace {
            *whitespace = 0;
        }

        return None;
    };

    /*
     * Return the text value...
     */

    if arena.node(node).type_ == MXML_TEXT {
        let MxmlValue::Text(text) = &arena.node(node).value else {
            return None;
        };
        if let Some(whitespace) = whitespace {
            *whitespace = text.whitespace;
        }

        text.string.as_deref()
    } else if arena.node(node).type_ == MXML_ELEMENT
        && let Some(child) = arena.node(node).child
        && arena.node(child).type_ == MXML_TEXT
    {
        let MxmlValue::Text(text) = &arena.node(child).value else {
            return None;
        };
        if let Some(whitespace) = whitespace {
            *whitespace = text.whitespace;
        }

        text.string.as_deref()
    } else {
        if let Some(whitespace) = whitespace {
            *whitespace = 0;
        }

        None
    }
}

/// Matches C `mxmlGetType` (`mxml-get.c:369`).
pub fn mxml_get_type(arena: &MxmlArena, node: Option<usize>) -> MxmlType {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return MXML_IGNORE;
    };

    /*
     * Return the node type...
     */

    arena.node(node).type_
}

/// Matches C `mxmlGetUserData` (`mxml-get.c:390`).
pub fn mxml_get_user_data(arena: &MxmlArena, node: Option<usize>) -> Option<&dyn Any> {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return None;
    };

    /*
     * Return the user data pointer...
     */

    arena.node(node).user_data.as_deref()
}
