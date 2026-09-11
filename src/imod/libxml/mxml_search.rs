//! Translation of `IMOD/libxml/mxml-search.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::*;
use core::ffi::{c_char, c_int, c_void};

/// Matches C `mxmlFindElement` (`mxml-search.c:32`).
pub unsafe fn mxml_find_element(
    mut node: *mut MxmlNode,
    top: *mut MxmlNode,
    name: *const c_char,
    attr: *const c_char,
    value: *const c_char,
    descend: c_int,
) -> *mut MxmlNode {
    let mut temp: *const c_char;

    /*
     * Range check input...
     */

    if node.is_null() || top.is_null() || (attr.is_null() && !value.is_null()) {
        return core::ptr::null_mut();
    }

    /*
     * Start with the next node...
     */

    node = mxml_walk_next(node, top, descend);

    /*
     * Loop until we find a matching element...
     */

    while !node.is_null() {
        /*
         * See if this node matches...
         */

        if (*node).type_ == MXML_ELEMENT
            && !(*node).value.element.name.is_null()
            && (name.is_null() || libc::strcmp((*node).value.element.name, name) == 0)
        {
            /*
             * See if we need to check for an attribute...
             */

            if attr.is_null() {
                return node; /* No attribute search, return it... */
            }

            /*
             * Check that the attribute exists and has the expected value...
             */

            temp = mxml_element_get_attr(node, attr);
            if !temp.is_null() && (value.is_null() || libc::strcmp(value, temp) == 0) {
                return node; /* Found it! */
            }
        }

        /*
         * No match, move on to the next node...
         */

        if descend == MXML_DESCEND {
            node = mxml_walk_next(node, top, MXML_DESCEND);
        } else {
            node = (*node).next;
        }
    }

    core::ptr::null_mut()
}

/// Matches C `mxmlFindPath` (`mxml-search.c:106`).
pub unsafe fn mxml_find_path(top: *mut MxmlNode, mut path: *const c_char) -> *mut MxmlNode {
    let mut node: *mut MxmlNode;
    let mut element: [c_char; 256] = [0; 256];
    let mut pathsep: *const c_char;
    let mut descend: c_int;

    /*
     * Range check input...
     */

    if top.is_null() || path.is_null() || *path == 0 {
        return core::ptr::null_mut();
    }

    /*
     * Search each element in the path...
     */

    node = top;
    while *path != 0 {
        /*
         * Handle wildcards...
         */

        if libc::strncmp(path, c"*/".as_ptr(), 2) == 0 {
            path = path.add(2);
            descend = MXML_DESCEND;
        } else {
            descend = MXML_DESCEND_FIRST;
        }

        /*
         * Get the next element in the path...
         */

        pathsep = libc::strchr(path, '/' as c_int);
        if pathsep.is_null() {
            pathsep = path.add(libc::strlen(path));
        }

        if pathsep == path
            || (pathsep as usize - path as usize) >= core::mem::size_of::<[c_char; 256]>()
        {
            return core::ptr::null_mut();
        }

        libc::memcpy(
            element.as_mut_ptr() as *mut c_void,
            path as *const c_void,
            pathsep as usize - path as usize,
        );
        element[pathsep as usize - path as usize] = 0;

        if *pathsep != 0 {
            path = pathsep.add(1);
        } else {
            path = pathsep;
        }

        /*
         * Search for the element...
         */

        node = mxml_find_element(
            node,
            node,
            element.as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
            descend,
        );
        if node.is_null() {
            return core::ptr::null_mut();
        }
    }

    /*
     * If we get this far, return the node or its first child...
     */

    if !(*node).child.is_null() && (*(*node).child).type_ != MXML_ELEMENT {
        (*node).child
    } else {
        node
    }
}

/// Matches C `mxmlWalkNext` (`mxml-search.c:185`).
pub unsafe fn mxml_walk_next(
    mut node: *mut MxmlNode,
    top: *mut MxmlNode,
    descend: c_int,
) -> *mut MxmlNode {
    if node.is_null() {
        core::ptr::null_mut()
    } else if !(*node).child.is_null() && descend != 0 {
        (*node).child
    } else if node == top {
        core::ptr::null_mut()
    } else if !(*node).next.is_null() {
        (*node).next
    } else if !(*node).parent.is_null() && (*node).parent != top {
        node = (*node).parent;

        while (*node).next.is_null() {
            if (*node).parent == top || (*node).parent.is_null() {
                return core::ptr::null_mut();
            } else {
                node = (*node).parent;
            }
        }

        (*node).next
    } else {
        core::ptr::null_mut()
    }
}

/// Matches C `mxmlWalkPrev` (`mxml-search.c:229`).
pub unsafe fn mxml_walk_prev(
    mut node: *mut MxmlNode,
    top: *mut MxmlNode,
    descend: c_int,
) -> *mut MxmlNode {
    if node.is_null() || node == top {
        core::ptr::null_mut()
    } else if !(*node).prev.is_null() {
        if !(*(*node).prev).last_child.is_null() && descend != 0 {
            node = (*(*node).prev).last_child;

            while !(*node).last_child.is_null() {
                node = (*node).last_child;
            }

            node
        } else {
            (*node).prev
        }
    } else if (*node).parent != top {
        (*node).parent
    } else {
        core::ptr::null_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mxmlFindPath` starts each step from the node it already has and uses
    /// `MXML_DESCEND_FIRST` (`mxml-search.c:150`), so a path that names the top
    /// element itself finds nothing while a leading `*/` does.  Both results
    /// below match the reference `libimxml.so`.
    #[test]
    fn find_path_needs_a_wildcard_rather_than_the_top_element_name() {
        unsafe {
            let tree = mxml_load_string(
                core::ptr::null_mut(),
                c"<a><b><c>v</c></b></a>".as_ptr(),
                Some(mxml_opaque_cb),
            );
            assert!(!tree.is_null());

            let n = mxml_find_path(tree, c"*/c".as_ptr());
            assert!(!n.is_null());
            assert_eq!((*n).type_, MXML_OPAQUE);
            assert_eq!(
                std::ffi::CStr::from_ptr(mxml_get_opaque(n)).to_bytes(),
                b"v"
            );

            assert!(mxml_find_path(tree, c"a/b/c".as_ptr()).is_null());
            assert!(mxml_find_path(tree, c"nosuch".as_ptr()).is_null());
            assert!(mxml_find_path(tree, c"*/nosuch".as_ptr()).is_null());
            assert!(mxml_find_path(tree, c"/c".as_ptr()).is_null());

            let n = mxml_find_path(tree, c"b/c".as_ptr());
            assert!(!n.is_null());
            assert_eq!(
                std::ffi::CStr::from_ptr(mxml_get_opaque(n)).to_bytes(),
                b"v"
            );
            let n = mxml_find_path(tree, c"b".as_ptr());
            assert!(!n.is_null());
            assert_eq!((*n).type_, MXML_ELEMENT);
            assert_eq!(
                std::ffi::CStr::from_ptr(mxml_get_element(n)).to_bytes(),
                b"b"
            );
            mxml_delete(tree);

            /*
             * A nested repeat of the same tag separates MXML_DESCEND_FIRST
             * from MXML_DESCEND: "b/c" finds nothing, the wildcard path and
             * "b/b/c" do.
             */
            let tree = mxml_load_string(
                core::ptr::null_mut(),
                c"<a><b><b><c>deep</c></b></b><d>top</d></a>".as_ptr(),
                Some(mxml_opaque_cb),
            );
            assert!(!tree.is_null());
            assert!(mxml_find_path(tree, c"b/c".as_ptr()).is_null());
            for path in [c"b/b/c", c"*/c"] {
                let n = mxml_find_path(tree, path.as_ptr());
                assert!(!n.is_null(), "{path:?}");
                assert_eq!(
                    std::ffi::CStr::from_ptr(mxml_get_opaque(n)).to_bytes(),
                    b"deep"
                );
            }
            for path in [c"d", c"*/d"] {
                let n = mxml_find_path(tree, path.as_ptr());
                assert!(!n.is_null(), "{path:?}");
                assert_eq!(
                    std::ffi::CStr::from_ptr(mxml_get_opaque(n)).to_bytes(),
                    b"top"
                );
            }
            mxml_delete(tree);
        }
    }
}
