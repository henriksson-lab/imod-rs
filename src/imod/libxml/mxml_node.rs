//! Translation of `IMOD/libxml/mxml-node.c`.
//!
//! `mxml_new` and `mxml_free` are the library's only allocator calls, so this
//! module is where the [`MxmlArena`] node slots are taken and released.  Every
//! function that the C hands a `mxml_node_t *` takes the arena plus the slot
//! index of that node.

use super::*;
use core::any::Any;

/// Matches C `mxmlAdd` (`mxml-node.c:41`).
pub fn mxml_add(
    arena: &mut MxmlArena,
    parent: Option<usize>,
    where_: i32,
    child: Option<usize>,
    node: Option<usize>,
) {
    /*
     * Range check input...
     */

    let (Some(parent), Some(node)) = (parent, node) else {
        return;
    };

    /*
     * Remove the node from any existing parent...
     */

    if arena.node(node).parent.is_some() {
        mxml_remove(arena, Some(node));
    }

    /*
     * Reset pointers...
     */

    arena.node_mut(node).parent = Some(parent);

    match where_ {
        MXML_ADD_BEFORE => {
            let child = child.filter(|child| {
                Some(*child) != arena.node(parent).child
                    && arena.node(*child).parent == Some(parent)
            });
            let Some(child) = child else {
                /*
                 * Insert as first node under parent...
                 */

                arena.node_mut(node).next = arena.node(parent).child;

                if let Some(first) = arena.node(parent).child {
                    arena.node_mut(first).prev = Some(node);
                } else {
                    arena.node_mut(parent).last_child = Some(node);
                }

                arena.node_mut(parent).child = Some(node);
                return;
            };
            {
                /*
                 * Insert node before this child...
                 */

                arena.node_mut(node).next = Some(child);
                arena.node_mut(node).prev = arena.node(child).prev;

                if let Some(prev) = arena.node(child).prev {
                    arena.node_mut(prev).next = Some(node);
                } else {
                    arena.node_mut(parent).child = Some(node);
                }

                arena.node_mut(child).prev = Some(node);
            }
        }

        MXML_ADD_AFTER => {
            let child = child.filter(|child| {
                Some(*child) != arena.node(parent).last_child
                    && arena.node(*child).parent == Some(parent)
            });
            let Some(child) = child else {
                /*
                 * Insert as last node under parent...
                 */

                arena.node_mut(node).parent = Some(parent);
                arena.node_mut(node).prev = arena.node(parent).last_child;

                if let Some(last) = arena.node(parent).last_child {
                    arena.node_mut(last).next = Some(node);
                } else {
                    arena.node_mut(parent).child = Some(node);
                }

                arena.node_mut(parent).last_child = Some(node);
                return;
            };
            {
                /*
                 * Insert node after this child...
                 */

                arena.node_mut(node).prev = Some(child);
                arena.node_mut(node).next = arena.node(child).next;

                if let Some(next) = arena.node(child).next {
                    arena.node_mut(next).prev = Some(node);
                } else {
                    arena.node_mut(parent).last_child = Some(node);
                }

                arena.node_mut(child).next = Some(node);
            }
        }

        _ => {}
    }
}

/// Matches C `mxmlDelete` (`mxml-node.c:165`).
pub fn mxml_delete(arena: &mut MxmlArena, node: Option<usize>) {
    let mut current: Option<usize>;
    let mut next: Option<usize>;

    /*
     * Range check input...
     */

    let Some(node) = node else {
        return;
    };

    /*
     * Remove the node from its parent, if any...
     */

    mxml_remove(arena, Some(node));

    /*
     * Delete children...
     */

    current = arena.node(node).child;
    while let Some(cur) = current {
        /*
         * Get the next node...
         */

        next = arena.node(cur).child;
        if next.is_some() {
            /*
             * Free parent nodes after child nodes have been freed...
             */

            arena.node_mut(cur).child = None;
            current = next;
            continue;
        }

        next = arena.node(cur).next;
        if next.is_none() {
            /*
             * Next node is the parent, which we'll free as needed...
             */

            next = arena.node(cur).parent;
            if next == Some(node) {
                next = None;
            }
        }

        /*
         * Free child...
         */

        mxml_free(arena, cur);

        current = next;
    }

    /*
     * Then free the memory used by the parent node...
     */

    mxml_free(arena, node);
}

/// Matches C `mxmlNewCustom` (`mxml-node.c:299`).
pub fn mxml_new_custom(
    arena: &mut MxmlArena,
    parent: Option<usize>,
    data: Option<Box<dyn Any>>,
    destroy: MxmlCustomDestroyCb,
) -> Option<usize> {
    /*
     * Create the node and set the value...
     */

    let node = mxml_new(arena, parent, MXML_CUSTOM);
    if let Some(node) = node
        && let MxmlValue::Custom(custom) = &mut arena.node_mut(node).value
    {
        custom.data = data;
        custom.destroy = destroy;
    }

    node
}

/// Matches C `mxmlNewElement` (`mxml-node.c:335`).
pub fn mxml_new_element(
    arena: &mut MxmlArena,
    parent: Option<usize>,
    name: Option<&[u8]>,
) -> Option<usize> {
    /*
     * Range check input...
     */

    let name = name?;

    /*
     * Create the node and set the element name...
     */

    let node = mxml_new(arena, parent, MXML_ELEMENT);
    if let Some(node) = node
        && let MxmlValue::Element(element) = &mut arena.node_mut(node).value
    {
        element.name = Some(name.to_vec());
    }

    node
}

/// Matches C `mxmlNewInteger` (`mxml-node.c:373`).
pub fn mxml_new_integer(
    arena: &mut MxmlArena,
    parent: Option<usize>,
    integer: i32,
) -> Option<usize> {
    /*
     * Create the node and set the element name...
     */

    let node = mxml_new(arena, parent, MXML_INTEGER);
    if let Some(node) = node {
        arena.node_mut(node).value = MxmlValue::Integer(integer);
    }

    node
}

/// Matches C `mxmlNewOpaque` (`mxml-node.c:404`).
pub fn mxml_new_opaque(
    arena: &mut MxmlArena,
    parent: Option<usize>,
    opaque: Option<&[u8]>,
) -> Option<usize> {
    /*
     * Range check input...
     */

    let opaque = opaque?;

    /*
     * Create the node and set the element name...
     */

    let node = mxml_new(arena, parent, MXML_OPAQUE);
    if let Some(node) = node {
        arena.node_mut(node).value = MxmlValue::Opaque(Some(opaque.to_vec()));
    }

    node
}

/// Matches C `mxmlNewReal` (`mxml-node.c:442`).
pub fn mxml_new_real(arena: &mut MxmlArena, parent: Option<usize>, real: f64) -> Option<usize> {
    /*
     * Create the node and set the element name...
     */

    let node = mxml_new(arena, parent, MXML_REAL);
    if let Some(node) = node {
        arena.node_mut(node).value = MxmlValue::Real(real);
    }

    node
}

/// Matches C `mxmlNewText` (`mxml-node.c:477`).
pub fn mxml_new_text(
    arena: &mut MxmlArena,
    parent: Option<usize>,
    whitespace: i32,
    string: Option<&[u8]>,
) -> Option<usize> {
    /*
     * Range check input...
     */

    let string = string?;

    /*
     * Create the node and set the text value...
     */

    let node = mxml_new(arena, parent, MXML_TEXT);
    if let Some(node) = node
        && let MxmlValue::Text(text) = &mut arena.node_mut(node).value
    {
        text.whitespace = whitespace;
        text.string = Some(string.to_vec());
    }

    node
}

/// Matches C `mxmlRemove` (`mxml-node.c:565`).
pub fn mxml_remove(arena: &mut MxmlArena, node: Option<usize>) {
    /*
     * Range check input...
     */

    let Some(node) = node else {
        return;
    };
    let Some(parent) = arena.node(node).parent else {
        return;
    };

    /*
     * Remove from parent...
     */

    if let Some(prev) = arena.node(node).prev {
        arena.node_mut(prev).next = arena.node(node).next;
    } else {
        arena.node_mut(parent).child = arena.node(node).next;
    }

    if let Some(next) = arena.node(node).next {
        arena.node_mut(next).prev = arena.node(node).prev;
    } else {
        arena.node_mut(parent).last_child = arena.node(node).prev;
    }

    arena.node_mut(node).parent = None;
    arena.node_mut(node).prev = None;
    arena.node_mut(node).next = None;
}

/// Matches C `mxmlNewXML` (`mxml-node.c:629`).
pub fn mxml_new_xml(arena: &mut MxmlArena, version: Option<&[u8]>) -> Option<usize> {
    let mut element: Vec<u8> = Vec::new();

    /*
     * snprintf(element, sizeof(element), "?xml version=\"%s\" encoding=\"utf-8\"?",
     *          version ? version : "1.0") -- into a 1024-byte buffer.
     */
    element.extend_from_slice(b"?xml version=\"");
    element.extend_from_slice(version.unwrap_or(b"1.0"));
    element.extend_from_slice(b"\" encoding=\"utf-8\"?");
    element.truncate(1023);

    mxml_new_element(arena, MXML_NO_PARENT, Some(&element))
}

/// Matches C `mxmlRelease` (`mxml-node.c:651`).
pub fn mxml_release(arena: &mut MxmlArena, node: Option<usize>) -> i32 {
    if let Some(node) = node {
        arena.node_mut(node).ref_count -= 1;
        if arena.node(node).ref_count <= 0 {
            mxml_delete(arena, Some(node));
            0
        } else {
            arena.node(node).ref_count
        }
    } else {
        -1
    }
}

/// Matches C static `mxml_free` (`mxml-node.c:690`).
///
/// Taking the node out of its arena slot is the `free(node)` at the end; every
/// `free` the C does on the value is that value's own drop, so the arms below
/// keep only what the C does beyond freeing — calling the custom destructor.
pub fn mxml_free(arena: &mut MxmlArena, node: usize) {
    let slot = node;
    let node = arena.nodes[slot]
        .take()
        .expect("mxml: mxml_free on a freed slot");

    match node.type_ {
        MXML_ELEMENT => { /* free(name); free each attr's name and value; free(attrs) */ }

        MXML_INTEGER => { /* Nothing to do */ }

        MXML_OPAQUE => { /* free(opaque) */ }

        MXML_REAL => { /* Nothing to do */ }

        MXML_TEXT => { /* free(text.string) */ }

        MXML_CUSTOM => {
            let MxmlValue::Custom(mut custom) = node.value else {
                unreachable!("mxml: MXML_CUSTOM node without a custom value")
            };
            if let Some(data) = custom.data.as_deref_mut()
                && let Some(destroy) = custom.destroy
            {
                destroy(data);
            }
        }

        _ => {}
    }

    /*
     * Free this node...
     */

    arena.free.push(slot);
}

/// Matches C static `mxml_new` (`mxml-node.c:747`).
pub fn mxml_new(arena: &mut MxmlArena, parent: Option<usize>, type_: MxmlType) -> Option<usize> {
    let node: usize;

    /*
     * Allocate memory for the node...  A freed slot is reused the way malloc
     * reuses a freed block; the C cannot fail to allocate here in practice and
     * a Rust allocation failure aborts rather than returning NULL.
     */

    let fresh = MxmlNode {
        type_: 0,
        next: None,
        prev: None,
        parent: None,
        child: None,
        last_child: None,
        value: match type_ {
            MXML_ELEMENT => MxmlValue::Element(MxmlElement {
                name: None,
                attrs: Vec::new(),
            }),
            MXML_INTEGER => MxmlValue::Integer(0),
            MXML_OPAQUE => MxmlValue::Opaque(None),
            MXML_REAL => MxmlValue::Real(0.0),
            MXML_TEXT => MxmlValue::Text(MxmlText {
                whitespace: 0,
                string: None,
            }),
            MXML_CUSTOM => MxmlValue::Custom(MxmlCustom {
                data: None,
                destroy: None,
            }),
            _ => MxmlValue::Ignore,
        },
        ref_count: 0,
        user_data: None,
    };

    if let Some(slot) = arena.free.pop() {
        arena.nodes[slot] = Some(fresh);
        node = slot;
    } else {
        arena.nodes.push(Some(fresh));
        node = arena.nodes.len() - 1;
    }

    /*
     * Set the node type...
     */

    arena.node_mut(node).type_ = type_;
    arena.node_mut(node).ref_count = 1;

    /*
     * Add to the parent if present...
     */

    if parent.is_some() {
        mxml_add(
            arena,
            parent,
            MXML_ADD_AFTER,
            MXML_ADD_TO_PARENT,
            Some(node),
        );
    }

    /*
     * Return the new node...
     */

    Some(node)
}
