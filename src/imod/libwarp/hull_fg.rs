//! Translation of `IMOD/libwarp/hull-fg.c`.
#![allow(dead_code)]
use crate::imod::libwarp::hull::{
    CDIM, FG_LIST, Fg, SITE_NUM, Simplex, Site, TREE_LIST, Tree, new_block_fg, new_block_tree,
    visit_hull,
};
use crate::imod::libwarp::hull_ch::{HULL_INFINITY, MO, find_alpha};
static mut FACES_GR_T: *mut Fg = core::ptr::null_mut();
static mut FG_OUT: *mut libc::FILE = core::ptr::null_mut();
static mut P_FG_X_DEPTH: i32 = 0;
unsafe extern "C" {
    static mut stdout: *mut libc::FILE;
}

/// Original `splay` (`hull-fg.c:112`).
pub unsafe fn splay(site: Site, mut tree: *mut Tree) -> *mut Tree {
    unsafe {
        if tree.is_null() {
            return tree;
        }
        let mut node: Tree = core::mem::zeroed();
        let mut left = &mut node as *mut Tree;
        let mut right = left;
        let mut left_size = 0;
        let mut right_size = 0;
        loop {
            let compare = SITE_NUM.unwrap()(site) - SITE_NUM.unwrap()((*tree).key);
            if compare < 0 {
                if (*tree).left.is_null() {
                    break;
                }
                if SITE_NUM.unwrap()(site) < SITE_NUM.unwrap()((*(*tree).left).key) {
                    let y = (*tree).left;
                    (*tree).left = (*y).right;
                    (*y).right = tree;
                    (*tree).size = if (*tree).left.is_null() {
                        0
                    } else {
                        (*(*tree).left).size
                    } + if (*tree).right.is_null() {
                        0
                    } else {
                        (*(*tree).right).size
                    } + 1;
                    tree = y;
                    if (*tree).left.is_null() {
                        break;
                    }
                }
                (*right).left = tree;
                right = tree;
                tree = (*tree).left;
                right_size += 1 + if (*right).right.is_null() {
                    0
                } else {
                    (*(*right).right).size
                };
            } else if compare > 0 {
                if (*tree).right.is_null() {
                    break;
                }
                if SITE_NUM.unwrap()(site) > SITE_NUM.unwrap()((*(*tree).right).key) {
                    let y = (*tree).right;
                    (*tree).right = (*y).left;
                    (*y).left = tree;
                    tree = y;
                    if (*tree).right.is_null() {
                        break;
                    }
                }
                (*left).right = tree;
                left = tree;
                tree = (*tree).right;
                left_size += 1 + if (*left).left.is_null() {
                    0
                } else {
                    (*(*left).left).size
                };
            } else {
                break;
            }
        }
        left_size += if (*tree).left.is_null() {
            0
        } else {
            (*(*tree).left).size
        };
        right_size += if (*tree).right.is_null() {
            0
        } else {
            (*(*tree).right).size
        };
        (*tree).size = left_size + right_size + 1;
        (*left).right = (*tree).left;
        (*right).left = (*tree).right;
        (*tree).left = node.right;
        (*tree).right = node.left;
        tree
    }
}

/// Original `insert` (`hull-fg.c:181`).
pub unsafe fn insert(site: Site, mut tree: *mut Tree) -> *mut Tree {
    unsafe {
        if !tree.is_null() {
            tree = splay(site, tree);
            if SITE_NUM.unwrap()(site) == SITE_NUM.unwrap()((*tree).key) {
                return tree;
            }
        }
        if TREE_LIST.is_null() {
            new_block_tree(1);
        }
        let node = TREE_LIST;
        TREE_LIST = (*node).next;
        if tree.is_null() {
            (*node).left = core::ptr::null_mut();
            (*node).right = core::ptr::null_mut();
        } else if SITE_NUM.unwrap()(site) < SITE_NUM.unwrap()((*tree).key) {
            (*node).left = (*tree).left;
            (*node).right = tree;
            (*tree).left = core::ptr::null_mut();
            (*tree).size = 1 + if (*tree).right.is_null() {
                0
            } else {
                (*(*tree).right).size
            };
        } else {
            (*node).right = (*tree).right;
            (*node).left = tree;
            (*tree).right = core::ptr::null_mut();
            (*tree).size = 1 + if (*tree).left.is_null() {
                0
            } else {
                (*(*tree).left).size
            };
        }
        (*node).key = site;
        (*node).size =
            1 + if (*node).left.is_null() {
                0
            } else {
                (*(*node).left).size
            } + if (*node).right.is_null() {
                0
            } else {
                (*(*node).right).size
            };
        node
    }
}

/// Original `find_rank` (`hull-fg.c:235`).
pub unsafe fn find_rank(mut rank: i32, mut tree: *mut Tree) -> *mut Tree {
    unsafe {
        if tree.is_null() || rank < 0 || rank >= (*tree).size {
            return core::ptr::null_mut();
        }
        loop {
            let left = if (*tree).left.is_null() {
                0
            } else {
                (*(*tree).left).size
            };
            if rank < left {
                tree = (*tree).left
            } else if rank > left {
                rank -= left + 1;
                tree = (*tree).right
            } else {
                return tree;
            }
        }
    }
}

/// Original `delete` (`hull-fg.c:211`).
pub unsafe fn delete(site: Site, mut tree: *mut Tree) -> *mut Tree {
    unsafe {
        if tree.is_null() {
            return core::ptr::null_mut();
        }
        let size = (*tree).size;
        tree = splay(site, tree);
        if SITE_NUM.unwrap()(site) != SITE_NUM.unwrap()((*tree).key) {
            return tree;
        }
        let replacement = if (*tree).left.is_null() {
            (*tree).right
        } else {
            let left = splay(site, (*tree).left);
            (*left).right = (*tree).right;
            left
        };
        core::ptr::write_bytes(tree.cast::<u8>(), 0, core::mem::size_of::<Tree>());
        (*tree).next = TREE_LIST;
        TREE_LIST = tree;
        if !replacement.is_null() {
            (*replacement).size = size - 1;
        }
        replacement
    }
}

/// Original `printtree_flat_inner` (`hull-fg.c:255`).
pub unsafe fn printtree_flat_inner(tree: *mut Tree) {
    unsafe {
        if tree.is_null() {
            return;
        }
        printtree_flat_inner((*tree).right);
        libc::printf(c"%p ".as_ptr(), (*tree).key);
        libc::fflush(stdout);
        printtree_flat_inner((*tree).left);
    }
}
/// Original `printtree_flat` (`hull-fg.c:264`).
pub unsafe fn printtree_flat(tree: *mut Tree) {
    unsafe {
        if tree.is_null() {
            libc::printf(c"<empty tree>".as_ptr());
            return;
        }
        printtree_flat_inner(tree);
    }
}
/// Original `printtree` (`hull-fg.c:273`).
pub unsafe fn printtree(tree: *mut Tree, depth: i32) {
    unsafe {
        if tree.is_null() {
            return;
        }
        printtree((*tree).right, depth + 1);
        for _ in 0..depth {
            libc::printf(c"  ".as_ptr());
        }
        libc::printf(c"%p(%d)\n".as_ptr(), (*tree).key, (*tree).size);
        libc::fflush(stdout);
        printtree((*tree).left, depth + 1);
    }
}

/// Original `find_fg` (`hull-fg.c:295`).
pub unsafe fn find_fg(simplex: *mut Simplex, query: i32) -> *mut Fg {
    unsafe {
        if query == 0 {
            return FACES_GR_T;
        }
        if FACES_GR_T.is_null() {
            if FG_LIST.is_null() {
                new_block_fg(1);
            }
            FACES_GR_T = FG_LIST;
            FG_LIST = (*FG_LIST).next;
            (*FACES_GR_T).ref_count = 1;
        }
        let mut face = FACES_GR_T;
        for index in 0..CDIM {
            if query & (1 << index) != 0 {
                let vertex = (*simplex)
                    .neigh
                    .as_mut_ptr()
                    .add(index as usize)
                    .read()
                    .vert;
                let tree = insert(vertex, (*face).facets);
                (*face).facets = tree;
                if (*tree).fgs.is_null() {
                    if FG_LIST.is_null() {
                        new_block_fg(1);
                    }
                    (*tree).fgs = FG_LIST;
                    FG_LIST = (*FG_LIST).next;
                    (*(*tree).fgs).ref_count = 1;
                }
                face = (*tree).fgs;
            }
        }
        face
    }
}

/// Original `add_to_fg` (`hull-fg.c:312`).
pub unsafe fn add_to_fg(
    simplex: *mut Simplex,
    _: *mut core::ffi::c_void,
) -> *mut core::ffi::c_void {
    unsafe {
        let count = 1 << CDIM;
        for query in 1..count {
            find_fg(simplex, query);
        }
        for query in 1..count {
            let face = find_fg(simplex, query);
            for index in 0..CDIM {
                let bit = 1 << index;
                if query & bit == 0 {
                    let vertex = (*simplex)
                        .neigh
                        .as_mut_ptr()
                        .add(index as usize)
                        .read()
                        .vert;
                    let tree = insert(vertex, (*face).facets);
                    (*face).facets = tree;
                    (*tree).fgs = find_fg(simplex, query | bit);
                }
            }
        }
        core::ptr::null_mut()
    }
}

/// Original `build_fg` (`hull-fg.c:340`).
pub unsafe fn build_fg(root: *mut Simplex) -> *mut Fg {
    unsafe {
        FACES_GR_T = core::ptr::null_mut();
        visit_hull(root, add_to_fg);
        FACES_GR_T
    }
}

/// Original `visit_fg_i` (`hull-fg.c:363`).
pub unsafe fn visit_fg_i(
    visitor: unsafe fn(*mut Tree, i32, i32),
    tree: *mut Tree,
    depth: i32,
    visit_number: i32,
    boundary: i32,
) {
    unsafe {
        if tree.is_null() {
            return;
        }
        let mut propagated_boundary = boundary;
        assert!(!(*tree).fgs.is_null());
        if (*(*tree).fgs).mark != visit_number as i16 {
            (*(*tree).fgs).mark = visit_number as i16;
            if (*tree).key != core::ptr::addr_of_mut!(HULL_INFINITY).cast()
                && MO[SITE_NUM.unwrap()((*tree).key) as usize] == 0
            {
                propagated_boundary = 0;
            }
            visitor(tree, depth, propagated_boundary);
            visit_fg_i(
                visitor,
                (*(*tree).fgs).facets,
                depth + 1,
                visit_number,
                propagated_boundary,
            );
        }
        visit_fg_i(visitor, (*tree).left, depth, visit_number, boundary);
        visit_fg_i(visitor, (*tree).right, depth, visit_number, boundary);
    }
}

/// Original `visit_fg` (`hull-fg.c:386`).
pub unsafe fn visit_fg(face_graph: *mut Fg, visitor: unsafe fn(*mut Tree, i32, i32)) {
    unsafe {
        static mut VISIT_NUMBER: i32 = 0;
        VISIT_NUMBER += 1;
        visit_fg_i(visitor, (*face_graph).facets, 0, VISIT_NUMBER, 1);
    }
}

/// Original `visit_fg_i_far` (`hull-fg.c:395`).
pub unsafe fn visit_fg_i_far(
    visitor: unsafe fn(*mut Tree, i32),
    tree: *mut Tree,
    depth: i32,
    visit_number: i32,
) -> i32 {
    unsafe {
        if tree.is_null() {
            return 0;
        }
        let mut boundary = 0;
        assert!(!(*tree).fgs.is_null());
        if (*(*tree).fgs).mark != visit_number as i16 {
            (*(*tree).fgs).mark = visit_number as i16;
            boundary = ((*tree).key == core::ptr::addr_of_mut!(HULL_INFINITY).cast()
                || MO[SITE_NUM.unwrap()((*tree).key) as usize] != 0) as i32;
            if boundary == 0
                && visit_fg_i_far(visitor, (*(*tree).fgs).facets, depth + 1, visit_number) == 0
            {
                visitor(tree, depth);
            }
        }
        (visit_fg_i_far(visitor, (*tree).left, depth, visit_number) != 0 || boundary != 0) as i32
            | (visit_fg_i_far(visitor, (*tree).right, depth, visit_number) != 0) as i32
    }
}

/// Original `visit_fg_far` (`hull-fg.c:413`).
pub unsafe fn visit_fg_far(face_graph: *mut Fg, visitor: unsafe fn(*mut Tree, i32)) {
    unsafe {
        static mut VISIT_NUMBER: i32 = 0;
        VISIT_NUMBER -= 1;
        visit_fg_i_far(visitor, (*face_graph).facets, 0, VISIT_NUMBER);
    }
}

/// Original `p_fg_x` (`hull-fg.c:413`).
pub unsafe fn p_fg_x(tree: *mut Tree, depth: i32, _: i32) {
    unsafe {
        static mut INDICES: [i32; 8] = [0; 8];
        INDICES[depth as usize] = SITE_NUM.unwrap()((*tree).key);
        if depth == P_FG_X_DEPTH {
            for index in 0..=depth {
                libc::fprintf(
                    FG_OUT,
                    c"%d%s".as_ptr(),
                    INDICES[index as usize],
                    if index == depth {
                        c"\n".as_ptr()
                    } else {
                        c" ".as_ptr()
                    },
                );
            }
        }
    }
}

/// Original `print_fg_alt` (`hull-fg.c:426`).
pub unsafe fn print_fg_alt(face_graph: *mut Fg, file: *mut libc::FILE, depth: i32) {
    unsafe {
        FG_OUT = file;
        if face_graph.is_null() {
            return;
        }
        P_FG_X_DEPTH = depth;
        visit_fg(face_graph, p_fg_x);
        libc::fclose(FG_OUT);
    }
}

/// Original `p_fg` (`hull-fg.c:413`).
pub unsafe fn p_fg(tree: *mut Tree, depth: i32, _: i32) {
    unsafe {
        static mut INDICES: [i32; 8] = [0; 8];
        static mut MULTIPLIERS: [f64; 8] = [0.; 8];
        /* `hull-fg.c:400` indexes the multiplier table by `pdim`, not by the
        traversal depth, and scales each step by `mult_up` (`hull-io.c:30`,
        always 1.0). */
        if MULTIPLIERS[0] == 0. {
            let pdim = *(&raw const crate::imod::libwarp::pointops::PDIM);
            MULTIPLIERS[pdim as usize] = 1.;
            for index in (0..pdim).rev() {
                MULTIPLIERS[index as usize] =
                    crate::imod::libwarp::hull_io::MULT_UP * MULTIPLIERS[(index + 1) as usize];
            }
        }
        INDICES[depth as usize] = SITE_NUM.unwrap()((*tree).key);
        for index in 0..=depth {
            libc::fprintf(FG_OUT, c"%d ".as_ptr(), INDICES[index as usize]);
        }
        libc::fprintf(
            FG_OUT,
            c"\t%G\n".as_ptr(),
            (*(*tree).fgs).vol / MULTIPLIERS[depth as usize],
        );
    }
}

/// Original `print_fg` (`hull-fg.c:436`).
pub unsafe fn print_fg(face_graph: *mut Fg, file: *mut libc::FILE) {
    unsafe {
        FG_OUT = file;
        visit_fg(face_graph, p_fg);
    }
}

static mut FG_HIST: [[f64; 100]; 100] = [[0.; 100]; 100];
static mut FG_HIST_BAD: [[f64; 100]; 100] = [[0.; 100]; 100];
static mut FG_HIST_FAR: [[f64; 100]; 100] = [[0.; 100]; 100];

/// Original `h_fg` (`hull-fg.c:440`).
pub unsafe fn h_fg(tree: *mut Tree, depth: i32, bad: i32) {
    unsafe {
        let facets = (*(*tree).fgs).facets;
        if facets.is_null() {
            return;
        }
        if bad != 0 {
            FG_HIST_BAD[depth as usize][(*facets).size as usize] += 1.;
        } else {
            FG_HIST[depth as usize][(*facets).size as usize] += 1.;
        }
    }
}

/// Original `h_fg_far` (`hull-fg.c:449`).
pub unsafe fn h_fg_far(tree: *mut Tree, depth: i32) {
    unsafe {
        let facets = (*(*tree).fgs).facets;
        if !facets.is_null() {
            FG_HIST_FAR[depth as usize][(*facets).size as usize] += 1.;
        }
    }
}

/// Original `print_hist_fg` (`hull-fg.c:454`).
pub unsafe fn print_hist_fg(root: *mut Simplex, mut graph: *mut Fg, file: *mut libc::FILE) {
    unsafe {
        let mut good = [0.; 100];
        let mut bad = [0.; 100];
        let mut far = [0.; 100];
        for i in 0..20 {
            for j in 0..100 {
                FG_HIST[i][j] = 0.;
                FG_HIST_BAD[i][j] = 0.;
                FG_HIST_FAR[i][j] = 0.;
            }
        }
        if root.is_null() {
            return;
        }
        find_alpha(root);
        if graph.is_null() {
            graph = build_fg(root);
        }
        visit_fg(graph, h_fg);
        visit_fg_far(graph, h_fg_far);
        for j in 0..100 {
            for i in 0..20 {
                good[i] += FG_HIST[i][j];
                bad[i] += FG_HIST_BAD[i][j];
                far[i] += FG_HIST_FAR[i][j];
            }
        }
        let mut last = 19;
        while last >= 0 && good[last] == 0. && bad[last] == 0. {
            last -= 1;
        }
        libc::fprintf(file, c"totals\t".as_ptr());
        for k in 0..=last {
            libc::fprintf(
                file,
                c"%s%d/%d/%d".as_ptr(),
                if k == 0 {
                    c"\t".as_ptr()
                } else {
                    c"\t\t\t".as_ptr()
                },
                far[k] as i32,
                good[k] as i32,
                (good[k] + bad[k]) as i32,
            );
        }
        for j in 0..100 {
            let mut depth = 19;
            while depth >= 0 && FG_HIST[depth][j] == 0. && FG_HIST_BAD[depth][j] == 0. {
                depth -= 1;
            }
            if depth < 0 {
                continue;
            }
            libc::fprintf(file, c"\n%d\t".as_ptr(), j as i32);
            for k in 0..=depth {
                if FG_HIST[k][j] != 0. || FG_HIST_BAD[k][j] != 0. {
                    libc::fprintf(
                        file,
                        c"%2.1f/%2.1f/%2.1f".as_ptr(),
                        if far[k] != 0. {
                            100. * FG_HIST_FAR[k][j] / far[k] + 0.05
                        } else {
                            0.
                        },
                        if good[k] != 0. {
                            100. * FG_HIST[k][j] / good[k] + 0.05
                        } else {
                            0.
                        },
                        100. * (FG_HIST[k][j] + FG_HIST_BAD[k][j]) / (good[k] + bad[k]) + 0.05,
                    );
                }
            }
        }
        libc::fprintf(file, c"\n".as_ptr());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static mut CALLS: i32 = 0;
    unsafe fn site_number(_: Site) -> i32 {
        0
    }
    unsafe fn count(_: *mut Tree, _: i32, _: i32) {
        unsafe {
            CALLS += 1;
        }
    }

    #[test]
    fn visit_fg_marks_a_graph_node_once_per_source_visit_number() {
        unsafe {
            SITE_NUM = Some(site_number);
            let tree = libc::calloc(1, core::mem::size_of::<Tree>()).cast::<Tree>();
            let face = libc::calloc(1, core::mem::size_of::<Fg>()).cast::<Fg>();
            (*tree).fgs = face;
            (*face).facets = tree;
            (*face).mark = -1;
            CALLS = 0;
            visit_fg(face, count);
            assert_eq!(core::ptr::read_volatile(core::ptr::addr_of!(CALLS)), 1);
            libc::free(tree.cast());
            libc::free(face.cast());
            SITE_NUM = None;
        }
    }
}
