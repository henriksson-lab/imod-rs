//! Translation of `IMOD/libxml/mxml-get.c`.

use super::*;
use core::any::Any;

/// Matches C `mxmlGetElement` (`mxml-get.c:88`).
pub fn mxml_get_element(arena: &MxmlArena, node: Option<usize>) -> Option<&[u8]> {
    /*
     * Range check input...
     */

    let node = node?;
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

    let node = node?;
    if arena.node(node).type_ != MXML_ELEMENT {
        return None;
    }

    /*
     * Return the first child node...
     */

    arena.node(node).child
}

/// Matches C `mxmlGetLastChild` (`mxml-get.c:165`).
pub fn mxml_get_last_child(arena: &MxmlArena, node: Option<usize>) -> Option<usize> {
    /*
     * Range check input...
     */

    let node = node?;
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

    let node = node?;

    /*
     * Return the node type...
     */

    arena.node(node).next
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
