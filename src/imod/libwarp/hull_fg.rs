//! Translation of `IMOD/libwarp/hull-fg.c` ("fg.c : face graph of hull, and
//! splay trees").
//!
//! Nothing in IMOD reaches this file: `build_fg`, `print_fg` and
//! `print_hist_fg` are only called from `hull-ch.c`'s `find_volumes` and from
//! `hullmain.c`, which is not vendored.  It is translated whole all the same.
//!
//! The source's `FILE *FG_OUT` global is replaced by a writer captured in the
//! visitor closure, since `visit_fg` takes a function pointer that has no
//! other way to reach it.
#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libwarp::hull::{
    CDIM, HULL_INFINITY, HullStorage, MAXDIM, SITE_NUM, SPLAY_N, Site, Tree, new_block_fg,
    new_block_tree, visit_hull,
};
use crate::imod::libwarp::hull_ch::{MO, find_alpha, nullify_basis};
use crate::imod::libwarp::hull_io::MULT_UP;
use crate::imod::libwarp::pointops::PDIM;

/// The signature of `visit_fg`'s `void (*v_fg)(Tree *, int, int)`
/// (`hull-fg.c:363`), as a closure so it can carry the output writer.
pub type VisitFgFunc<'a> = &'a mut dyn FnMut(&mut HullStorage, usize, i32, i32);
/// The signature of `visit_fg_far`'s `void (*v_fg)(Tree *, int)`
/// (`hull-fg.c:395`).
pub type VisitFgFarFunc<'a> = &'a mut dyn FnMut(&mut HullStorage, usize, i32);

thread_local! {
    /// C `fg *faces_gr_t` (`hull-fg.c:289`).
    static FACES_GR_T: Cell<usize> = const { Cell::new(0) };
    /// C `int p_fg_x_depth` (`hull-fg.c:418`).
    static P_FG_X_DEPTH: Cell<i32> = const { Cell::new(0) };
    /// `visit_fg`'s `static int fg_vn` (`hull-fg.c:387`).
    static VISIT_FG_VN: Cell<i32> = const { Cell::new(0) };
    /// `visit_fg_far`'s `static int fg_vn` (`hull-fg.c:414`).
    static VISIT_FG_FAR_VN: Cell<i32> = const { Cell::new(0) };
    /// `p_fg`'s `static int fa[MAXDIM]` (`hull-fg.c:399`).
    static P_FG_FA: Cell<[i32; MAXDIM]> = const { Cell::new([0; MAXDIM]) };
    /// `p_fg`'s `static double mults[MAXDIM]` (`hull-fg.c:401`).
    static P_FG_MULTS: Cell<[f64; MAXDIM]> = const { Cell::new([0.; MAXDIM]) };
    /// `p_fg_x`'s `static int fa[MAXDIM]` (`hull-fg.c:422`).
    static P_FG_X_FA: Cell<[i32; MAXDIM]> = const { Cell::new([0; MAXDIM]) };
    /// `p_fg_x`'s `static point fp[MAXDIM]` (`hull-fg.c:423`).
    static P_FG_X_FP: Cell<[Site; MAXDIM]> = const { Cell::new([0; MAXDIM]) };
    /// C `double fg_hist[100][100]` (`hull-fg.c:446`).
    static FG_HIST: RefCell<Vec<[f64; 100]>> = RefCell::new(vec![[0.; 100]; 100]);
    /// C `double fg_hist_bad[100][100]` (`hull-fg.c:446`).
    static FG_HIST_BAD: RefCell<Vec<[f64; 100]>> = RefCell::new(vec![[0.; 100]; 100]);
    /// C `double fg_hist_far[100][100]` (`hull-fg.c:446`).
    static FG_HIST_FAR: RefCell<Vec<[f64; 100]>> = RefCell::new(vec![[0.; 100]; 100]);
}

/// C `node_size(x)` (`hull-fg.c:107`).  Index 0 is the source's `NULL`, whose
/// slot carries size 0, so the macro's own NULL test is the arena's.
#[inline]
fn node_size(st: &HullStorage, x: usize) -> i32 {
    if x != 0 { st.tree[x].size } else { 0 }
}

/// C `compare(i,j)` (`hull-fg.c:102`).
#[inline]
fn compare(i: Site, j: Site) -> i32 {
    SITE_NUM.get().unwrap()(i) - SITE_NUM.get().unwrap()(j)
}

/// Original `splay` (`hull-fg.c:112`).
///
/// Splay using the key `i` (which may or may not be in the tree).  The
/// starting root is `t`; size fields are maintained.
pub fn splay(st: &mut HullStorage, i: Site, mut t: usize) -> usize {
    if t == 0 {
        return t;
    }
    /* Tree N; N.left = N.right = NULL; l = r = &N; */
    st.tree[SPLAY_N].left = 0;
    st.tree[SPLAY_N].right = 0;
    let mut l = SPLAY_N;
    let mut r = SPLAY_N;
    let _root_size = node_size(st, t);
    let mut l_size = 0;
    let mut r_size = 0;

    loop {
        let comp = compare(i, st.tree[t].key);
        if comp < 0 {
            if st.tree[t].left == 0 {
                break;
            }
            if compare(i, st.tree[st.tree[t].left].key) < 0 {
                /* rotate right */
                let y = st.tree[t].left;
                st.tree[t].left = st.tree[y].right;
                st.tree[y].right = t;
                st.tree[t].size =
                    node_size(st, st.tree[t].left) + node_size(st, st.tree[t].right) + 1;
                t = y;
                if st.tree[t].left == 0 {
                    break;
                }
            }
            /* link right */
            st.tree[r].left = t;
            r = t;
            t = st.tree[t].left;
            r_size += 1 + node_size(st, st.tree[r].right);
        } else if comp > 0 {
            if st.tree[t].right == 0 {
                break;
            }
            if compare(i, st.tree[st.tree[t].right].key) > 0 {
                /* rotate left */
                let y = st.tree[t].right;
                st.tree[t].right = st.tree[y].left;
                st.tree[y].left = t;
                st.tree[t].size =
                    node_size(st, st.tree[t].left) + node_size(st, st.tree[t].right) + 1;
                t = y;
                if st.tree[t].right == 0 {
                    break;
                }
            }
            /* link left */
            st.tree[l].right = t;
            l = t;
            t = st.tree[t].right;
            l_size += 1 + node_size(st, st.tree[l].left);
        } else {
            break;
        }
    }
    /* Now l_size and r_size are the sizes of the left and right trees we just
    built. */
    l_size += node_size(st, st.tree[t].left);
    r_size += node_size(st, st.tree[t].right);
    st.tree[t].size = l_size + r_size + 1;

    st.tree[l].right = 0;
    st.tree[r].left = 0;

    /* The following two loops correct the size fields of the right path from
    the left child of the root and the right path from the left child of the
    root. */
    let mut y = st.tree[SPLAY_N].right;
    while y != 0 {
        st.tree[y].size = l_size;
        l_size -= 1 + node_size(st, st.tree[y].left);
        y = st.tree[y].right;
    }
    let mut y = st.tree[SPLAY_N].left;
    while y != 0 {
        st.tree[y].size = r_size;
        r_size -= 1 + node_size(st, st.tree[y].right);
        y = st.tree[y].left;
    }

    /* assemble */
    st.tree[l].right = st.tree[t].left;
    st.tree[r].left = st.tree[t].right;
    st.tree[t].left = st.tree[SPLAY_N].right;
    st.tree[t].right = st.tree[SPLAY_N].left;

    t
}

/// Original static `insert` (`hull-fg.c:181`) — insert key `i` into the tree
/// `t`, if it is not already there.
pub fn insert(st: &mut HullStorage, i: Site, mut t: usize) -> usize {
    if t != 0 {
        t = splay(st, i, t);
        if compare(i, st.tree[t].key) == 0 {
            /* it's already there */
            return t;
        }
    }
    /* NEWL(Tree, new) */
    if st.tree_list == 0 {
        new_block_tree(st, 1);
    }
    let new = st.tree_list;
    assert!(new != 0);
    st.tree_list = st.tree[new].next;

    if t == 0 {
        st.tree[new].left = 0;
        st.tree[new].right = 0;
    } else if compare(i, st.tree[t].key) < 0 {
        st.tree[new].left = st.tree[t].left;
        st.tree[new].right = t;
        st.tree[t].left = 0;
        st.tree[t].size = 1 + node_size(st, st.tree[t].right);
    } else {
        st.tree[new].right = st.tree[t].right;
        st.tree[new].left = t;
        st.tree[t].right = 0;
        st.tree[t].size = 1 + node_size(st, st.tree[t].left);
    }
    st.tree[new].key = i;
    st.tree[new].size = 1 + node_size(st, st.tree[new].left) + node_size(st, st.tree[new].right);
    new
}

/// Original static `delete` (`hull-fg.c:211`) — deletes `i` from the tree if
/// it's there.
pub fn delete(st: &mut HullStorage, i: Site, mut t: usize) -> usize {
    if t == 0 {
        return 0;
    }
    let tsize = st.tree[t].size;
    t = splay(st, i, t);
    if compare(i, st.tree[t].key) == 0 {
        /* found it */
        let x = if st.tree[t].left == 0 {
            st.tree[t].right
        } else {
            let left = st.tree[t].left;
            let x = splay(st, i, left);
            st.tree[x].right = st.tree[t].right;
            x
        };
        /* FREEL(Tree, t) */
        let next = st.tree_list;
        st.tree[t] = Tree::default();
        st.tree[t].next = next;
        st.tree_list = t;
        if x != 0 {
            st.tree[x].size = tsize - 1;
        }
        x
    } else {
        /* It wasn't there */
        t
    }
}

/// Original `find_rank` (`hull-fg.c:235`) — returns the node in the tree with
/// the given rank, `0` if there is no such node.
pub fn find_rank(st: &mut HullStorage, mut r: i32, mut t: usize) -> usize {
    if r < 0 || r >= node_size(st, t) {
        return 0;
    }
    loop {
        let lsize = node_size(st, st.tree[t].left);
        if r < lsize {
            t = st.tree[t].left;
        } else if r > lsize {
            r = r - lsize - 1;
            t = st.tree[t].right;
        } else {
            return t;
        }
    }
}

/// Original `printtree_flat_inner` (`hull-fg.c:255`).
///
/// **Deviation.** The source prints the key's *address* with `%p`; a site is
/// an arena index here, so that is what `%p` receives.
pub fn printtree_flat_inner(st: &mut HullStorage, t: usize) {
    if t == 0 {
        return;
    }
    let right = st.tree[t].right;
    printtree_flat_inner(st, right);
    /* DNM: cast to void * to prevent warning */
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(c_format("%p ", &[CArg::Ptr(st.tree[t].key)]).as_bytes());
    let _ = out.flush();
    let left = st.tree[t].left;
    printtree_flat_inner(st, left);
}

/// Original `printtree_flat` (`hull-fg.c:264`).
pub fn printtree_flat(st: &mut HullStorage, t: usize) {
    if t == 0 {
        let mut out = ImodFile::Stdout;
        let _ = out.write_all(b"<empty tree>");
        return;
    }
    printtree_flat_inner(st, t);
}

/// Original `printtree` (`hull-fg.c:273`).  Same `%p` deviation as
/// [`printtree_flat_inner`].
pub fn printtree(st: &mut HullStorage, t: usize, d: i32) {
    if t == 0 {
        return;
    }
    let right = st.tree[t].right;
    printtree(st, right, d + 1);
    let mut out = ImodFile::Stdout;
    for _ in 0..d {
        let _ = out.write_all(b"  ");
    }
    /* DNM: cast to void * to prevent warning */
    let _ = out.write_all(
        c_format(
            "%p(%d)\n",
            &[CArg::Ptr(st.tree[t].key), CArg::Int(st.tree[t].size as i64)],
        )
        .as_bytes(),
    );
    let _ = out.flush();
    let left = st.tree[t].left;
    printtree(st, left, d + 1);
}

/// Original `find_fg` (`hull-fg.c:295`).
pub fn find_fg(st: &mut HullStorage, s: usize, q: i32) -> usize {
    if q == 0 {
        return FACES_GR_T.get();
    }
    if FACES_GR_T.get() == 0 {
        /* NEWLRC(fg, faces_gr_t) */
        if st.fg_list == 0 {
            new_block_fg(st, 1);
        }
        let f = st.fg_list;
        assert!(f != 0);
        st.fg_list = st.fg[f].next;
        st.fg[f].ref_count = 1;
        FACES_GR_T.set(f);
    }
    let mut f = FACES_GR_T.get();
    let cdim = CDIM.get();
    for si in 0..cdim {
        if q & (1 << si) != 0 {
            let vert = st.simplex[s].neigh[si as usize].vert;
            let facets = st.fg[f].facets;
            let t = insert(st, vert, facets);
            st.fg[f].facets = t;
            if st.tree[t].fgs == 0 {
                /* NEWLRC(fg, (t->fgs)) */
                if st.fg_list == 0 {
                    new_block_fg(st, 1);
                }
                let g = st.fg_list;
                assert!(g != 0);
                st.fg_list = st.fg[g].next;
                st.fg[g].ref_count = 1;
                st.tree[t].fgs = g;
            }
            f = st.tree[t].fgs;
        }
    }
    f
}

/// Original `add_to_fg` (`hull-fg.c:312`).
pub fn add_to_fg(st: &mut HullStorage, s: usize) -> usize {
    let cdim = CDIM.get();
    let qmax = 1 << cdim;

    /* sort neigh by site number */
    let mut si = 2;
    while si < cdim {
        let mut sj = si;
        while sj > 1
            && SITE_NUM.get().unwrap()(st.simplex[s].neigh[(sj - 1) as usize].vert)
                > SITE_NUM.get().unwrap()(st.simplex[s].neigh[sj as usize].vert)
        {
            let t = st.simplex[s].neigh[(sj - 1) as usize];
            st.simplex[s].neigh[(sj - 1) as usize] = st.simplex[s].neigh[sj as usize];
            st.simplex[s].neigh[sj as usize] = t;
            sj -= 1;
        }
        si += 1;
    }

    let b = st.simplex[s].normal;
    nullify_basis(st, b);
    st.simplex[s].normal = 0;
    let b = st.simplex[s].neigh[0].basis;
    nullify_basis(st, b);
    st.simplex[s].neigh[0].basis = 0;

    /* insert subsets */
    for q in 1..qmax {
        find_fg(st, s, q);
    }

    /* include all superset relations */
    for q in 1..qmax {
        let fq = find_fg(st, s, q);
        assert!(fq != 0);
        let mut m = 1;
        for si in 0..cdim {
            if q & m == 0 {
                let vert = st.simplex[s].neigh[si as usize].vert;
                let facets = st.fg[fq].facets;
                let t = insert(st, vert, facets);
                st.fg[fq].facets = t;
                let g = find_fg(st, s, q | m);
                let facets = st.fg[fq].facets;
                st.tree[facets].fgs = g;
            }
            m <<= 1;
        }
    }
    0
}

/// Original `build_fg` (`hull-fg.c:340`).
pub fn build_fg(st: &mut HullStorage, root: usize) -> usize {
    FACES_GR_T.set(0);
    visit_hull(st, root, &mut |st, s| add_to_fg(st, s));
    FACES_GR_T.get()
}

/// Original `visit_fg_i` (`hull-fg.c:346`).
pub fn visit_fg_i(
    st: &mut HullStorage,
    v_fg: VisitFgFunc,
    t: usize,
    depth: i32,
    vn: i32,
    boundary: i32,
) {
    let mut boundaryc = boundary;

    if t == 0 {
        return;
    }

    assert!(st.tree[t].fgs != 0);
    let fgs = st.tree[t].fgs;
    if st.fg[fgs].mark != vn as i16 {
        st.fg[fgs].mark = vn as i16;
        let key = st.tree[t].key;
        if key != HULL_INFINITY && MO.with_borrow(|m| m[SITE_NUM.get().unwrap()(key) as usize]) == 0
        {
            boundaryc = 0;
        }
        v_fg(st, t, depth, boundaryc);
        let facets = st.fg[fgs].facets;
        visit_fg_i(st, v_fg, facets, depth + 1, vn, boundaryc);
    }
    let left = st.tree[t].left;
    visit_fg_i(st, v_fg, left, depth, vn, boundary);
    let right = st.tree[t].right;
    visit_fg_i(st, v_fg, right, depth, vn, boundary);
}

/// Original `visit_fg` (`hull-fg.c:363`).
pub fn visit_fg(st: &mut HullStorage, faces_gr: usize, v_fg: VisitFgFunc) {
    VISIT_FG_VN.set(VISIT_FG_VN.get() + 1);
    let vn = VISIT_FG_VN.get();
    let facets = st.fg[faces_gr].facets;
    visit_fg_i(st, v_fg, facets, 0, vn, 1);
}

/// Original `visit_fg_i_far` (`hull-fg.c:373`).
pub fn visit_fg_i_far(
    st: &mut HullStorage,
    v_fg: VisitFgFarFunc,
    t: usize,
    depth: i32,
    vn: i32,
) -> i32 {
    let mut nb = 0;

    if t == 0 {
        return 0;
    }

    assert!(st.tree[t].fgs != 0);
    let fgs = st.tree[t].fgs;
    if st.fg[fgs].mark != vn as i16 {
        st.fg[fgs].mark = vn as i16;
        let key = st.tree[t].key;
        nb = (key == HULL_INFINITY
            || MO.with_borrow(|m| m[SITE_NUM.get().unwrap()(key) as usize]) != 0)
            as i32;
        let facets = st.fg[fgs].facets;
        if nb == 0 && visit_fg_i_far(st, v_fg, facets, depth + 1, vn) == 0 {
            v_fg(st, t, depth);
        }
    }
    let left = st.tree[t].left;
    nb = (visit_fg_i_far(st, v_fg, left, depth, vn) != 0 || nb != 0) as i32;
    let right = st.tree[t].right;
    nb = (visit_fg_i_far(st, v_fg, right, depth, vn) != 0 || nb != 0) as i32;
    nb
}

/// Original `visit_fg_far` (`hull-fg.c:390`).
pub fn visit_fg_far(st: &mut HullStorage, faces_gr: usize, v_fg: VisitFgFarFunc) {
    VISIT_FG_FAR_VN.set(VISIT_FG_FAR_VN.get() - 1);
    let vn = VISIT_FG_FAR_VN.get();
    let facets = st.fg[faces_gr].facets;
    visit_fg_i_far(st, v_fg, facets, 0, vn);
}

/// Original `p_fg` (`hull-fg.c:398`).
pub fn p_fg(st: &mut HullStorage, t: usize, depth: i32, _bad: i32, f: &mut dyn Write) {
    let mut mults = P_FG_MULTS.get();
    if mults[0] == 0. {
        let pdim = PDIM.get();
        mults[pdim as usize] = 1.;
        for i in (0..pdim).rev() {
            mults[i as usize] = MULT_UP.get() * mults[(i + 1) as usize];
        }
        P_FG_MULTS.set(mults);
    }

    let mut fa = P_FG_FA.get();
    fa[depth as usize] = SITE_NUM.get().unwrap()(st.tree[t].key);
    P_FG_FA.set(fa);
    for i in 0..=depth {
        let _ = f.write_all(c_format("%d ", &[CArg::Int(fa[i as usize] as i64)]).as_bytes());
    }
    let fgs = st.tree[t].fgs;
    let _ = f.write_all(
        c_format(
            "\t%G\n",
            &[CArg::Dbl(st.fg[fgs].vol / mults[depth as usize])],
        )
        .as_bytes(),
    );
}

/// Original `p_fg_x` (`hull-fg.c:420`).
pub fn p_fg_x(st: &mut HullStorage, t: usize, depth: i32, _bad: i32, f: &mut dyn Write) {
    let mut fa = P_FG_X_FA.get();
    let mut fp = P_FG_X_FP.get();

    fa[depth as usize] = SITE_NUM.get().unwrap()(st.tree[t].key);
    fp[depth as usize] = st.tree[t].key;
    P_FG_X_FA.set(fa);
    P_FG_X_FP.set(fp);

    if depth == P_FG_X_DEPTH.get() {
        for i in 0..=depth {
            let _ = f.write_all(
                c_format(
                    "%d%s",
                    &[
                        CArg::Int(fa[i as usize] as i64),
                        CArg::Str(if i == depth { "\n" } else { " " }),
                    ],
                )
                .as_bytes(),
            );
        }
    }
}

/// Original `print_fg_alt` (`hull-fg.c:432`).
///
/// The source `fclose`s `FG_OUT` on the way out; the file is taken by value
/// here so that dropping it does the same.
pub fn print_fg_alt(st: &mut HullStorage, faces_gr: usize, mut f: ImodFile, fd: i32) {
    if faces_gr == 0 {
        return;
    }
    P_FG_X_DEPTH.set(fd);
    visit_fg(st, faces_gr, &mut |st, t, depth, bad| {
        p_fg_x(st, t, depth, bad, &mut f)
    });
    /* fclose(FG_OUT) */
}

/// Original `print_fg` (`hull-fg.c:441`).
pub fn print_fg(st: &mut HullStorage, faces_gr: usize, f: &mut dyn Write) {
    visit_fg(st, faces_gr, &mut |st, t, depth, bad| {
        p_fg(st, t, depth, bad, f)
    });
}

/// Original `h_fg` (`hull-fg.c:446`).
pub fn h_fg(st: &mut HullStorage, t: usize, depth: i32, bad: i32) {
    let fgs = st.tree[t].fgs;
    let facets = st.fg[fgs].facets;
    if facets == 0 {
        return;
    }
    let size = st.tree[facets].size as usize;
    if bad != 0 {
        FG_HIST_BAD.with_borrow_mut(|h| h[depth as usize][size] += 1.);
        return;
    }
    FG_HIST.with_borrow_mut(|h| h[depth as usize][size] += 1.);
}

/// Original `h_fg_far` (`hull-fg.c:455`).
pub fn h_fg_far(st: &mut HullStorage, t: usize, depth: i32) {
    let fgs = st.tree[t].fgs;
    let facets = st.fg[fgs].facets;
    if facets != 0 {
        let size = st.tree[facets].size as usize;
        FG_HIST_FAR.with_borrow_mut(|h| h[depth as usize][size] += 1.);
    }
}

/// Original `print_hist_fg` (`hull-fg.c:460`).
pub fn print_hist_fg(st: &mut HullStorage, root: usize, faces_gr: usize, f: &mut dyn Write) {
    let mut tot_good = [0f64; 100];
    let mut tot_bad = [0f64; 100];
    let mut tot_far = [0f64; 100];
    for i in 0..20 {
        tot_good[i] = 0.;
        tot_bad[i] = 0.;
        tot_far[i] = 0.;
        for j in 0..100 {
            FG_HIST.with_borrow_mut(|h| h[i][j] = 0.);
            FG_HIST_BAD.with_borrow_mut(|h| h[i][j] = 0.);
            FG_HIST_FAR.with_borrow_mut(|h| h[i][j] = 0.);
        }
    }
    if root == 0 {
        return;
    }

    find_alpha(st, root);

    let faces_gr = if faces_gr == 0 {
        build_fg(st, root)
    } else {
        faces_gr
    };

    visit_fg(st, faces_gr, &mut |st, t, depth, bad| {
        h_fg(st, t, depth, bad)
    });
    visit_fg_far(st, faces_gr, &mut |st, t, depth| h_fg_far(st, t, depth));

    for j in 0..100 {
        for i in 0..20 {
            tot_good[i] += FG_HIST.with_borrow(|h| h[i][j]);
            tot_bad[i] += FG_HIST_BAD.with_borrow(|h| h[i][j]);
            tot_far[i] += FG_HIST_FAR.with_borrow(|h| h[i][j]);
        }
    }

    let mut i = 19i32;
    while i >= 0 && tot_good[i as usize] == 0. && tot_bad[i as usize] == 0. {
        i -= 1;
    }
    let _ = f.write_all(b"totals\t");
    for k in 0..=i {
        if k == 0 {
            let _ = f.write_all(b"\t");
        } else {
            let _ = f.write_all(b"\t\t\t");
        }
        let _ = f.write_all(
            c_format(
                "%d/%d/%d",
                &[
                    CArg::Int(tot_far[k as usize] as i32 as i64),
                    CArg::Int(tot_good[k as usize] as i32 as i64),
                    CArg::Int((tot_good[k as usize] as i32 + tot_bad[k as usize] as i32) as i64),
                ],
            )
            .as_bytes(),
        );
    }

    for j in 0..100 {
        let mut i = 19i32;
        while i >= 0
            && FG_HIST.with_borrow(|h| h[i as usize][j]) == 0.
            && FG_HIST_BAD.with_borrow(|h| h[i as usize][j]) == 0.
        {
            i -= 1;
        }
        if i == -1 {
            continue;
        }
        let _ = f.write_all(c_format("\n%d\t", &[CArg::Int(j as i64)]).as_bytes());
        let _ = f.flush();

        for k in 0..=i {
            if k == 0 {
                let _ = f.write_all(b"\t");
            } else {
                let _ = f.write_all(b"\t\t\t");
            }
            let hk = FG_HIST.with_borrow(|h| h[k as usize][j]);
            let hkb = FG_HIST_BAD.with_borrow(|h| h[k as usize][j]);
            let hkf = FG_HIST_FAR.with_borrow(|h| h[k as usize][j]);
            if hk != 0. || hkb != 0. {
                let _ = f.write_all(
                    c_format(
                        "%2.1f/%2.1f/%2.1f",
                        &[
                            CArg::Dbl(if tot_far[k as usize] != 0. {
                                100. * hkf / tot_far[k as usize] + 0.05
                            } else {
                                0.
                            }),
                            CArg::Dbl(if tot_good[k as usize] != 0. {
                                100. * hk / tot_good[k as usize] + 0.05
                            } else {
                                0.
                            }),
                            CArg::Dbl(
                                100. * (hk + hkb) / (tot_good[k as usize] + tot_bad[k as usize])
                                    + 0.05,
                            ),
                        ],
                    )
                    .as_bytes(),
                );
            }
        }
    }
    let _ = f.write_all(b"\n");
}
