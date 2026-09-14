//! Translation of `IMOD/libxml/mxml-index.c`.
#![allow(dead_code)]

use super::*;

/// Matches C `mxmlIndexDelete` (`mxml-index.c:41`).
///
/// The C frees `ind->attr`, `ind->nodes` and the index itself; taking the
/// index by value is that transfer of ownership, and the `None` arm is the
/// C's `if (!ind) return`.
pub fn mxml_index_delete(ind: Option<MxmlIndex>) {
    /*
     * Range check input...
     */

    let Some(ind) = ind else {
        return;
    };

    /*
     * Free memory...
     */

    drop(ind);
}

/// Matches C `mxmlIndexEnum` (`mxml-index.c:73`).
pub fn mxml_index_enum(ind: Option<&mut MxmlIndex>) -> Option<usize> {
    /*
     * Range check input...
     */

    let Some(ind) = ind else {
        return None;
    };

    /*
     * Return the next node...
     */

    if ind.cur_node < ind.nodes.len() {
        let node = ind.nodes[ind.cur_node];
        ind.cur_node += 1;
        Some(node)
    } else {
        None
    }
}

/// Matches C `mxmlIndexFind` (`mxml-index.c:104`).
pub fn mxml_index_find(
    arena: &MxmlArena,
    ind: Option<&mut MxmlIndex>,
    element: Option<&[u8]>,
    value: Option<&[u8]>,
) -> Option<usize> {
    let mut current: usize;
    let mut first: usize;
    let mut last: usize;

    /*
     * Range check input...
     */

    let Some(ind) = ind else {
        return None;
    };
    if ind.attr.is_none() && value.is_some() {
        return None;
    }

    /*
     * If both element and value are NULL, just enumerate the nodes in the
     * index...
     */

    if element.is_none() && value.is_none() {
        return mxml_index_enum(Some(ind));
    }

    /*
     * If there are no nodes in the index, return NULL...
     */

    if ind.nodes.is_empty() {
        return None;
    }

    /*
     * If cur_node == 0, then find the first matching node...
     */

    if ind.cur_node == 0 {
        /*
         * Find the first node using a modified binary search algorithm...
         */

        first = 0;
        last = ind.nodes.len() - 1;

        while (last - first) > 1 {
            current = (first + last) / 2;

            if index_find(arena, ind, element, value, ind.nodes[current]).is_eq() {
                /*
                 * Found a match, move back to find the first...
                 */

                while current > 0
                    && index_find(arena, ind, element, value, ind.nodes[current - 1]).is_eq()
                {
                    current -= 1;
                }

                /*
                 * Return the first match and save the index to the next...
                 */

                ind.cur_node = current + 1;

                return Some(ind.nodes[current]);
            } else if index_find(arena, ind, element, value, ind.nodes[current]).is_lt() {
                last = current;
            } else {
                first = current;
            }
        }

        /*
         * If we get this far, then we found exactly 0 or 1 matches...
         */

        current = first;
        while current <= last {
            if index_find(arena, ind, element, value, ind.nodes[current]).is_eq() {
                /*
                 * Found exactly one (or possibly two) match...
                 */

                ind.cur_node = current + 1;
                return Some(ind.nodes[current]);
            }
            current += 1;
        }

        /*
         * No matches...
         */

        ind.cur_node = ind.nodes.len();

        None
    } else if ind.cur_node < ind.nodes.len()
        && index_find(arena, ind, element, value, ind.nodes[ind.cur_node]).is_eq()
    {
        /*
         * Return the next matching node...
         */

        let node = ind.nodes[ind.cur_node];
        ind.cur_node += 1;
        Some(node)
    } else {
        /*
         * No more matches...
         */

        ind.cur_node = ind.nodes.len();

        None
    }
}

/// Matches C `mxmlIndexGetCount` (`mxml-index.c:229`).
pub fn mxml_index_get_count(ind: Option<&MxmlIndex>) -> usize {
    /*
     * Range check input...
     */

    let Some(ind) = ind else {
        return 0;
    };

    /*
     * Return the number of nodes in the index...
     */

    ind.nodes.len()
}

/// Matches C `mxmlIndexNew` (`mxml-index.c:257`).
///
/// The C `calloc`s the index and returns a pointer; the owned value carries
/// the same ownership, and the two allocation-failure arms — which report
/// "Unable to allocate %d bytes for index" and abort the build — cannot occur
/// with a `Vec`.
pub fn mxml_index_new(
    arena: &MxmlArena,
    node: Option<usize>,
    element: Option<&[u8]>,
    attr: Option<&[u8]>,
) -> Option<MxmlIndex> {
    let mut ind: MxmlIndex;
    let mut current: Option<usize>;

    /*
     * Range check input...
     */

    node?;

    /*
     * Create a new index...
     */

    ind = MxmlIndex {
        attr: None,
        cur_node: 0,
        nodes: Vec::with_capacity(64),
    };

    if let Some(attr) = attr {
        ind.attr = Some(attr.to_vec());
    }

    if element.is_none() && attr.is_none() {
        current = node;
    } else {
        current = mxml_find_element(arena, node, node, element, attr, None, MXML_DESCEND);
    }

    while let Some(cur) = current {
        ind.nodes.push(cur);

        current = mxml_find_element(arena, current, node, element, attr, None, MXML_DESCEND);
    }

    /*
     * Sort nodes based upon the search criteria...
     */

    if ind.nodes.len() > 1 {
        let attr = ind.attr.clone();
        ind.nodes
            .sort_by(|first, second| index_compare(arena, attr.as_deref(), *first, *second));
    }

    /*
     * Return the new index...
     */

    Some(ind)
}

/// Matches C `mxmlIndexReset` (`mxml-index.c:404`).
pub fn mxml_index_reset(ind: Option<&mut MxmlIndex>) -> Option<usize> {
    /*
     * Range check input...
     */

    let Some(ind) = ind else {
        return None;
    };

    /*
     * Set the index to the first element...
     */

    ind.cur_node = 0;

    /*
     * Return the first node...
     */

    if !ind.nodes.is_empty() {
        Some(ind.nodes[0])
    } else {
        None
    }
}

/// Matches C static `index_compare` (`mxml-index.c:434`).
pub fn index_compare(
    arena: &MxmlArena,
    attr_name: Option<&[u8]>,
    first: usize,
    second: usize,
) -> core::cmp::Ordering {
    /*
     * Check the element name...  The C `strcmp`s two element names that hold
     * no NUL, so the byte-slice ordering carries the sign strcmp returns and
     * the sign is all the caller uses.
     */

    let first_name = match &arena.node(first).value {
        MxmlValue::Element(element) => element.name.as_deref(),
        _ => None,
    }
    .expect("mxml: index_compare dereferences the element name the C strcmps");
    let second_name = match &arena.node(second).value {
        MxmlValue::Element(element) => element.name.as_deref(),
        _ => None,
    }
    .expect("mxml: index_compare dereferences the element name the C strcmps");

    let diff = first_name.cmp(second_name);
    if !diff.is_eq() {
        return diff;
    }

    /*
     * Check the attribute value...
     */

    if attr_name.is_some() {
        let first_attr = mxml_element_get_attr(arena, Some(first), attr_name)
            .expect("mxml: index_compare dereferences the attribute the C strcmps");
        let second_attr = mxml_element_get_attr(arena, Some(second), attr_name)
            .expect("mxml: index_compare dereferences the attribute the C strcmps");

        let diff = first_attr.cmp(second_attr);
        if !diff.is_eq() {
            return diff;
        }
    }

    /*
     * No difference, return 0...
     */

    core::cmp::Ordering::Equal
}

/// Matches C static `index_find` (`mxml-index.c:471`).
pub fn index_find(
    arena: &MxmlArena,
    ind: &MxmlIndex,
    element: Option<&[u8]>,
    value: Option<&[u8]>,
    node: usize,
) -> core::cmp::Ordering {
    /*
     * Check the element name...
     */

    if let Some(element) = element {
        let name = match &arena.node(node).value {
            MxmlValue::Element(e) => e.name.as_deref(),
            _ => None,
        }
        .expect("mxml: index_find dereferences the element name the C strcmps");

        let diff = element.cmp(&name);
        if !diff.is_eq() {
            return diff;
        }
    }

    /*
     * Check the attribute value...
     */

    if let Some(value) = value {
        let attr = mxml_element_get_attr(arena, Some(node), ind.attr.as_deref())
            .expect("mxml: index_find dereferences the attribute the C strcmps");

        let diff = value.cmp(&attr);
        if !diff.is_eq() {
            return diff;
        }
    }

    /*
     * No difference, return 0...
     */

    core::cmp::Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mxmlIndexNew` sorts the nodes, and `mxmlIndexReset` returns `nodes[0]` *without*
    /// advancing `cur_node`, so an enumeration loop that starts with a reset
    /// visits the first node twice.  Every value below is the reference
    /// `libimxml.so` result for the same input.
    ///
    /// The tree the reference parsed from
    /// `<root><child x="3"/><child x="1"/><child x="2"/></root>` is built here
    /// with the node API: `mxml_file`'s parser is a later conversion wave.
    #[test]
    fn index_is_sorted_and_reset_does_not_consume() {
        let arena = &mut MxmlArena::new();
        let tree = mxml_new_element(arena, MXML_NO_PARENT, Some(b"root"));
        for x in [b"3".as_slice(), b"1".as_slice(), b"2".as_slice()] {
            let child = mxml_new_element(arena, tree, Some(b"child"));
            mxml_element_set_attr(arena, child, Some(b"x"), Some(x));
        }
        assert!(tree.is_some());

        let mut ind = mxml_index_new(arena, tree, Some(b"child"), Some(b"x"))
            .expect("mxmlIndexNew returned an index");
        assert_eq!(mxml_index_get_count(Some(&ind)), 3);

        let mut seen: Vec<Vec<u8>> = Vec::new();
        let mut n = mxml_index_reset(Some(&mut ind));
        while n.is_some() {
            let a = mxml_element_get_attr(arena, n, Some(b"x")).unwrap();
            seen.push(a.to_vec());
            n = mxml_index_enum(Some(&mut ind));
        }
        assert_eq!(
            seen,
            vec![b"1".to_vec(), b"1".to_vec(), b"2".to_vec(), b"3".to_vec()]
        );

        mxml_index_reset(Some(&mut ind));
        let n = mxml_index_find(arena, Some(&mut ind), Some(b"child"), Some(b"2"));
        assert!(n.is_some());
        assert_eq!(
            mxml_element_get_attr(arena, n, Some(b"x")),
            Some(b"2".as_slice())
        );
        mxml_index_reset(Some(&mut ind));
        assert!(mxml_index_find(arena, Some(&mut ind), Some(b"child"), Some(b"zzz")).is_none());
        mxml_index_delete(Some(ind));
        mxml_delete(arena, tree);
    }
}
