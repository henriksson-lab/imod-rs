//! Translation of `IMOD/libxml/mxml-search.c`.
#![allow(dead_code)]

use super::*;
use core::ffi::c_int;

/// Matches C `mxmlFindElement` (`mxml-search.c:32`).
pub fn mxml_find_element(
    arena: &MxmlArena,
    node: Option<usize>,
    top: Option<usize>,
    name: Option<&[u8]>,
    attr: Option<&[u8]>,
    value: Option<&[u8]>,
    descend: c_int,
) -> Option<usize> {
    let mut node = node;

    /*
     * Range check input...
     */

    if node.is_none() || top.is_none() || (attr.is_none() && value.is_some()) {
        return None;
    }

    /*
     * Start with the next node...
     */

    node = mxml_walk_next(arena, node, top, descend);

    /*
     * Loop until we find a matching element...
     */

    while let Some(current) = node {
        /*
         * See if this node matches...
         */

        let element_name = match &arena.node(current).value {
            MxmlValue::Element(element) => element.name.as_deref(),
            _ => None,
        };

        if arena.node(current).type_ == MXML_ELEMENT
            && element_name.is_some()
            && (name.is_none() || element_name == name)
        {
            /*
             * See if we need to check for an attribute...
             */

            if attr.is_none() {
                return node; /* No attribute search, return it... */
            }

            /*
             * Check that the attribute exists and has the expected value...
             */

            let temp: Option<&[u8]> = mxml_element_get_attr(arena, node, attr);
            if temp.is_some() && (value.is_none() || value == temp) {
                return node; /* Found it! */
            }
        }

        /*
         * No match, move on to the next node...
         */

        if descend == MXML_DESCEND {
            node = mxml_walk_next(arena, node, top, MXML_DESCEND);
        } else {
            node = arena.node(current).next;
        }
    }

    None
}

/// Matches C `mxmlFindPath` (`mxml-search.c:106`).
pub fn mxml_find_path(arena: &MxmlArena, top: Option<usize>, path: &[u8]) -> Option<usize> {
    let mut node: Option<usize>;

    /*
     * Range check input...
     */

    if top.is_none() || path.is_empty() {
        return None;
    }

    /*
     * Search each element in the path...
     */

    let mut path = path;
    node = top;
    while !path.is_empty() {
        /*
         * Handle wildcards...
         */

        let descend: c_int;
        if path.starts_with(b"*/") {
            path = &path[2..];
            descend = MXML_DESCEND;
        } else {
            descend = MXML_DESCEND_FIRST;
        }

        /*
         * Get the next element in the path...
         */

        let pathsep: usize = match path.iter().position(|&c| c == b'/') {
            Some(index) => index,
            None => path.len(),
        };

        /* The C copies into a 256-byte buffer and rejects anything longer. */
        if pathsep == 0 || pathsep >= 256 {
            return None;
        }

        let element: &[u8] = &path[..pathsep];

        if pathsep < path.len() {
            path = &path[pathsep + 1..];
        } else {
            path = &path[pathsep..];
        }

        /*
         * Search for the element...
         */

        node = mxml_find_element(arena, node, node, Some(element), None, None, descend);
        if node.is_none() {
            return None;
        }
    }

    /*
     * If we get this far, return the node or its first child...
     */

    let node = node?;
    if let Some(child) = arena.node(node).child
        && arena.node(child).type_ != MXML_ELEMENT
    {
        Some(child)
    } else {
        Some(node)
    }
}

/// Matches C `mxmlWalkNext` (`mxml-search.c:185`).
pub fn mxml_walk_next(
    arena: &MxmlArena,
    node: Option<usize>,
    top: Option<usize>,
    descend: c_int,
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

/// Matches C `mxmlWalkPrev` (`mxml-search.c:229`).
pub fn mxml_walk_prev(
    arena: &MxmlArena,
    node: Option<usize>,
    top: Option<usize>,
    descend: c_int,
) -> Option<usize> {
    let mut node = node;

    if node.is_none() || node == top {
        None
    } else if let Some(prev) = arena.node(node.unwrap()).prev {
        if arena.node(prev).last_child.is_some() && descend != 0 {
            node = arena.node(prev).last_child;

            while arena.node(node.unwrap()).last_child.is_some() {
                node = arena.node(node.unwrap()).last_child;
            }

            node
        } else {
            Some(prev)
        }
    } else if arena.node(node.unwrap()).parent != top {
        arena.node(node.unwrap()).parent
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mxmlFindPath` starts each step from the node it already has and uses
    /// `MXML_DESCEND_FIRST` (`mxml-search.c:150`), so a path that names the top
    /// element itself finds nothing while a leading `*/` does.  Both results
    /// below match the reference `libimxml.so`.
    ///
    /// The trees the reference parsed from `<a><b><c>v</c></b></a>` and from
    /// `<a><b><b><c>deep</c></b></b><d>top</d></a>` with `mxml_opaque_cb` are
    /// built here with the node API instead: `mxml_file`'s parser is a later
    /// conversion wave, and an opaque load callback gives each element's text
    /// exactly one MXML_OPAQUE child.
    #[test]
    fn find_path_needs_a_wildcard_rather_than_the_top_element_name() {
        let arena = &mut MxmlArena::new();

        let tree = mxml_new_element(arena, MXML_NO_PARENT, Some(b"a"));
        let b = mxml_new_element(arena, tree, Some(b"b"));
        let c = mxml_new_element(arena, b, Some(b"c"));
        mxml_new_opaque(arena, c, Some(b"v"));
        assert!(tree.is_some());

        let n = mxml_find_path(arena, tree, b"*/c");
        assert!(n.is_some());
        assert_eq!(mxml_get_type(arena, n), MXML_OPAQUE);
        assert_eq!(mxml_get_opaque(arena, n), Some(b"v".as_slice()));

        assert!(mxml_find_path(arena, tree, b"a/b/c").is_none());
        assert!(mxml_find_path(arena, tree, b"nosuch").is_none());
        assert!(mxml_find_path(arena, tree, b"*/nosuch").is_none());
        assert!(mxml_find_path(arena, tree, b"/c").is_none());

        let n = mxml_find_path(arena, tree, b"b/c");
        assert!(n.is_some());
        assert_eq!(mxml_get_opaque(arena, n), Some(b"v".as_slice()));
        let n = mxml_find_path(arena, tree, b"b");
        assert!(n.is_some());
        assert_eq!(mxml_get_type(arena, n), MXML_ELEMENT);
        assert_eq!(mxml_get_element(arena, n), Some(b"b".as_slice()));
        mxml_delete(arena, tree);

        /*
         * A nested repeat of the same tag separates MXML_DESCEND_FIRST
         * from MXML_DESCEND: "b/c" finds nothing, the wildcard path and
         * "b/b/c" do.
         */
        let tree = mxml_new_element(arena, MXML_NO_PARENT, Some(b"a"));
        let b1 = mxml_new_element(arena, tree, Some(b"b"));
        let b2 = mxml_new_element(arena, b1, Some(b"b"));
        let c = mxml_new_element(arena, b2, Some(b"c"));
        mxml_new_opaque(arena, c, Some(b"deep"));
        let d = mxml_new_element(arena, tree, Some(b"d"));
        mxml_new_opaque(arena, d, Some(b"top"));
        assert!(tree.is_some());

        assert!(mxml_find_path(arena, tree, b"b/c").is_none());
        for path in [b"b/b/c".as_slice(), b"*/c".as_slice()] {
            let n = mxml_find_path(arena, tree, path);
            assert!(n.is_some(), "{path:?}");
            assert_eq!(mxml_get_opaque(arena, n), Some(b"deep".as_slice()));
        }
        for path in [b"d".as_slice(), b"*/d".as_slice()] {
            let n = mxml_find_path(arena, tree, path);
            assert!(n.is_some(), "{path:?}");
            assert_eq!(mxml_get_opaque(arena, n), Some(b"top".as_slice()));
        }
        mxml_delete(arena, tree);
    }
}
