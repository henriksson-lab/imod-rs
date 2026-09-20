//! Translation of `IMOD/libxml/mxml-search.c`.

use super::*;

/// Matches C `mxmlWalkNext` (`mxml-search.c:185`).
pub fn mxml_walk_next(
    arena: &MxmlArena,
    node: Option<usize>,
    top: Option<usize>,
    descend: i32,
) -> Option<usize> {
    let mut node = node;

    if node.is_none() {
        None
    } else if arena.node(node.unwrap()).child.is_some() && descend != 0 {
        arena.node(node.unwrap()).child
    } else if node == top {
        None
    } else if arena.node(node.unwrap()).next.is_some() {
        arena.node(node.unwrap()).next
    } else if arena.node(node.unwrap()).parent.is_some() && arena.node(node.unwrap()).parent != top
    {
        node = arena.node(node.unwrap()).parent;

        while arena.node(node.unwrap()).next.is_none() {
            if arena.node(node.unwrap()).parent == top || arena.node(node.unwrap()).parent.is_none()
            {
                return None;
            } else {
                node = arena.node(node.unwrap()).parent;
            }
        }

        arena.node(node.unwrap()).next
    } else {
        None
    }
}
