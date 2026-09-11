//! Translation of `IMOD/libxml/mxml-index.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::*;
use core::ffi::{c_char, c_int, c_void};

/// Matches C `mxmlIndexDelete` (`mxml-index.c:41`).
pub unsafe fn mxml_index_delete(ind: *mut MxmlIndex) {
    /*
     * Range check input...
     */

    if ind.is_null() {
        return;
    }

    /*
     * Free memory...
     */

    if !(*ind).attr.is_null() {
        libc::free((*ind).attr as *mut c_void);
    }

    if (*ind).alloc_nodes != 0 {
        libc::free((*ind).nodes as *mut c_void);
    }

    libc::free(ind as *mut c_void);
}

/// Matches C `mxmlIndexEnum` (`mxml-index.c:73`).
pub unsafe fn mxml_index_enum(ind: *mut MxmlIndex) -> *mut MxmlNode {
    /*
     * Range check input...
     */

    if ind.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Return the next node...
     */

    if (*ind).cur_node < (*ind).num_nodes {
        let node = *(*ind).nodes.add((*ind).cur_node as usize);
        (*ind).cur_node += 1;
        node
    } else {
        core::ptr::null_mut()
    }
}

/// Matches C `mxmlIndexFind` (`mxml-index.c:104`).
pub unsafe fn mxml_index_find(
    ind: *mut MxmlIndex,
    element: *const c_char,
    value: *const c_char,
) -> *mut MxmlNode {
    let mut diff: c_int;
    let mut current: c_int;
    let mut first: c_int;
    let mut last: c_int;

    /*
     * Range check input...
     */

    if ind.is_null() || ((*ind).attr.is_null() && !value.is_null()) {
        return core::ptr::null_mut();
    }

    /*
     * If both element and value are NULL, just enumerate the nodes in the
     * index...
     */

    if element.is_null() && value.is_null() {
        return mxml_index_enum(ind);
    }

    /*
     * If there are no nodes in the index, return NULL...
     */

    if (*ind).num_nodes == 0 {
        return core::ptr::null_mut();
    }

    /*
     * If cur_node == 0, then find the first matching node...
     */

    if (*ind).cur_node == 0 {
        /*
         * Find the first node using a modified binary search algorithm...
         */

        first = 0;
        last = (*ind).num_nodes - 1;

        while (last - first) > 1 {
            current = (first + last) / 2;

            diff = index_find(ind, element, value, *(*ind).nodes.add(current as usize));
            if diff == 0 {
                /*
                 * Found a match, move back to find the first...
                 */

                while current > 0
                    && index_find(
                        ind,
                        element,
                        value,
                        *(*ind).nodes.add((current - 1) as usize),
                    ) == 0
                {
                    current -= 1;
                }

                /*
                 * Return the first match and save the index to the next...
                 */

                (*ind).cur_node = current + 1;

                return *(*ind).nodes.add(current as usize);
            } else if diff < 0 {
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
            if index_find(ind, element, value, *(*ind).nodes.add(current as usize)) == 0 {
                /*
                 * Found exactly one (or possibly two) match...
                 */

                (*ind).cur_node = current + 1;
                return *(*ind).nodes.add(current as usize);
            }
            current += 1;
        }

        /*
         * No matches...
         */

        (*ind).cur_node = (*ind).num_nodes;

        core::ptr::null_mut()
    } else if (*ind).cur_node < (*ind).num_nodes
        && index_find(
            ind,
            element,
            value,
            *(*ind).nodes.add((*ind).cur_node as usize),
        ) == 0
    {
        /*
         * Return the next matching node...
         */

        let node = *(*ind).nodes.add((*ind).cur_node as usize);
        (*ind).cur_node += 1;
        node
    } else {
        /*
         * No more matches...
         */

        (*ind).cur_node = (*ind).num_nodes;

        core::ptr::null_mut()
    }
}

/// Matches C `mxmlIndexGetCount` (`mxml-index.c:229`).
pub unsafe fn mxml_index_get_count(ind: *mut MxmlIndex) -> c_int {
    /*
     * Range check input...
     */

    if ind.is_null() {
        return 0;
    }

    /*
     * Return the number of nodes in the index...
     */

    (*ind).num_nodes
}

/// Matches C `mxmlIndexNew` (`mxml-index.c:257`).
pub unsafe fn mxml_index_new(
    node: *mut MxmlNode,
    element: *const c_char,
    attr: *const c_char,
) -> *mut MxmlIndex {
    let ind: *mut MxmlIndex;
    let mut current: *mut MxmlNode;
    let mut temp: *mut *mut MxmlNode;

    /*
     * Range check input...
     */

    if node.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Create a new index...
     */

    ind = libc::calloc(1, core::mem::size_of::<MxmlIndex>()) as *mut MxmlIndex;
    if ind.is_null() {
        let mut msg: [c_char; 1024] = [0; 1024];
        libc::snprintf(
            msg.as_mut_ptr(),
            core::mem::size_of::<[c_char; 1024]>(),
            c"Unable to allocate %d bytes for index - %s".as_ptr(),
            core::mem::size_of::<MxmlIndex>() as c_int,
            libc::strerror(*libc::__errno_location()),
        );
        mxml_error(msg.as_ptr());
        return core::ptr::null_mut();
    }

    if !attr.is_null() {
        (*ind).attr = libc::strdup(attr);
    }

    if element.is_null() && attr.is_null() {
        current = node;
    } else {
        current = mxml_find_element(node, node, element, attr, core::ptr::null(), MXML_DESCEND);
    }

    while !current.is_null() {
        if (*ind).num_nodes >= (*ind).alloc_nodes {
            if (*ind).alloc_nodes == 0 {
                temp =
                    libc::malloc(64 * core::mem::size_of::<*mut MxmlNode>()) as *mut *mut MxmlNode;
            } else {
                temp = libc::realloc(
                    (*ind).nodes as *mut c_void,
                    ((*ind).alloc_nodes + 64) as usize * core::mem::size_of::<*mut MxmlNode>(),
                ) as *mut *mut MxmlNode;
            }

            if temp.is_null() {
                /*
                 * Unable to allocate memory for the index, so abort...
                 */

                let mut msg: [c_char; 1024] = [0; 1024];
                libc::snprintf(
                    msg.as_mut_ptr(),
                    core::mem::size_of::<[c_char; 1024]>(),
                    c"Unable to allocate %d bytes for index: %s".as_ptr(),
                    (((*ind).alloc_nodes + 64) as usize * core::mem::size_of::<*mut MxmlNode>())
                        as c_int,
                    libc::strerror(*libc::__errno_location()),
                );
                mxml_error(msg.as_ptr());

                mxml_index_delete(ind);
                return core::ptr::null_mut();
            }

            (*ind).nodes = temp;
            (*ind).alloc_nodes += 64;
        }

        *(*ind).nodes.add((*ind).num_nodes as usize) = current;
        (*ind).num_nodes += 1;

        current = mxml_find_element(
            current,
            node,
            element,
            attr,
            core::ptr::null(),
            MXML_DESCEND,
        );
    }

    /*
     * Sort nodes based upon the search criteria...
     */

    if (*ind).num_nodes > 1 {
        index_sort(ind, 0, (*ind).num_nodes - 1);
    }

    /*
     * Return the new index...
     */

    ind
}

/// Matches C `mxmlIndexReset` (`mxml-index.c:404`).
pub unsafe fn mxml_index_reset(ind: *mut MxmlIndex) -> *mut MxmlNode {
    /*
     * Range check input...
     */

    if ind.is_null() {
        return core::ptr::null_mut();
    }

    /*
     * Set the index to the first element...
     */

    (*ind).cur_node = 0;

    /*
     * Return the first node...
     */

    if (*ind).num_nodes != 0 {
        *(*ind).nodes.add(0)
    } else {
        core::ptr::null_mut()
    }
}

/// Matches C static `index_compare` (`mxml-index.c:434`).
pub unsafe fn index_compare(
    ind: *mut MxmlIndex,
    first: *mut MxmlNode,
    second: *mut MxmlNode,
) -> c_int {
    let mut diff: c_int;

    /*
     * Check the element name...
     */

    diff = libc::strcmp((*first).value.element.name, (*second).value.element.name);
    if diff != 0 {
        return diff;
    }

    /*
     * Check the attribute value...
     */

    if !(*ind).attr.is_null() {
        diff = libc::strcmp(
            mxml_element_get_attr(first, (*ind).attr),
            mxml_element_get_attr(second, (*ind).attr),
        );
        if diff != 0 {
            return diff;
        }
    }

    /*
     * No difference, return 0...
     */

    0
}

/// Matches C static `index_find` (`mxml-index.c:471`).
pub unsafe fn index_find(
    ind: *mut MxmlIndex,
    element: *const c_char,
    value: *const c_char,
    node: *mut MxmlNode,
) -> c_int {
    let mut diff: c_int;

    /*
     * Check the element name...
     */

    if !element.is_null() {
        diff = libc::strcmp(element, (*node).value.element.name);
        if diff != 0 {
            return diff;
        }
    }

    /*
     * Check the attribute value...
     */

    if !value.is_null() {
        diff = libc::strcmp(value, mxml_element_get_attr(node, (*ind).attr));
        if diff != 0 {
            return diff;
        }
    }

    /*
     * No difference, return 0...
     */

    0
}

/// Matches C static `index_sort` (`mxml-index.c:518`).
///
/// Sort the nodes in an index...
///
/// This function implements the classic quicksort algorithm...
pub unsafe fn index_sort(ind: *mut MxmlIndex, mut left: c_int, right: c_int) {
    let mut pivot: *mut MxmlNode;
    let mut temp: *mut MxmlNode;
    let mut templ: c_int;
    let mut tempr: c_int;

    /*
     * Loop until we have sorted all the way to the right...
     */

    loop {
        /*
         * Sort the pivot in the current partition...
         */

        pivot = *(*ind).nodes.add(left as usize);

        templ = left;
        tempr = right;
        while templ < tempr {
            /*
             * Move left while left node <= pivot node...
             */

            while (templ < right)
                && index_compare(ind, *(*ind).nodes.add(templ as usize), pivot) <= 0
            {
                templ += 1;
            }

            /*
             * Move right while right node > pivot node...
             */

            while (tempr > left) && index_compare(ind, *(*ind).nodes.add(tempr as usize), pivot) > 0
            {
                tempr -= 1;
            }

            /*
             * Swap nodes if needed...
             */

            if templ < tempr {
                temp = *(*ind).nodes.add(templ as usize);
                *(*ind).nodes.add(templ as usize) = *(*ind).nodes.add(tempr as usize);
                *(*ind).nodes.add(tempr as usize) = temp;
            }
        }

        /*
         * When we get here, tempr <= templ...
         */

        if index_compare(ind, pivot, *(*ind).nodes.add(tempr as usize)) > 0 {
            *(*ind).nodes.add(left as usize) = *(*ind).nodes.add(tempr as usize);
            *(*ind).nodes.add(tempr as usize) = pivot;
        }

        /*
         * Recursively sort the left partition as needed...
         */

        if left < (tempr - 1) {
            index_sort(ind, left, tempr - 1);
        }

        left = tempr + 1;
        if right <= left {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mxmlIndexNew` allocates in blocks of 64 and quicksorts the nodes
    /// (`mxml-index.c:340`), and `mxmlIndexReset` returns `nodes[0]` *without*
    /// advancing `cur_node`, so an enumeration loop that starts with a reset
    /// visits the first node twice.  Every value below is the reference
    /// `libimxml.so` result for the same input.
    #[test]
    fn index_is_sorted_allocated_in_64s_and_reset_does_not_consume() {
        unsafe {
            let tree = mxml_load_string(
                core::ptr::null_mut(),
                c"<root><child x=\"3\"/><child x=\"1\"/><child x=\"2\"/></root>".as_ptr(),
                Some(mxml_opaque_cb),
            );
            assert!(!tree.is_null());
            let ind = mxml_index_new(tree, c"child".as_ptr(), c"x".as_ptr());
            assert!(!ind.is_null());
            assert_eq!(mxml_index_get_count(ind), 3);
            assert_eq!((*ind).alloc_nodes, 64);

            let mut seen: Vec<Vec<u8>> = Vec::new();
            let mut n = mxml_index_reset(ind);
            while !n.is_null() {
                let a = mxml_element_get_attr(n, c"x".as_ptr());
                seen.push(std::ffi::CStr::from_ptr(a).to_bytes().to_vec());
                n = mxml_index_enum(ind);
            }
            assert_eq!(
                seen,
                vec![b"1".to_vec(), b"1".to_vec(), b"2".to_vec(), b"3".to_vec()]
            );

            mxml_index_reset(ind);
            let n = mxml_index_find(ind, c"child".as_ptr(), c"2".as_ptr());
            assert!(!n.is_null());
            assert_eq!(
                std::ffi::CStr::from_ptr(mxml_element_get_attr(n, c"x".as_ptr())).to_bytes(),
                b"2"
            );
            mxml_index_reset(ind);
            assert!(mxml_index_find(ind, c"child".as_ptr(), c"zzz".as_ptr()).is_null());
            mxml_index_delete(ind);
            mxml_delete(tree);
        }
    }
}
