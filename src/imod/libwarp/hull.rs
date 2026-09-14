//! Translation of `IMOD/libwarp/hull.c` and `hull.h`.
//!
//! # Modelling Clarkson's storage
//!
//! The source is the 1995 Clarkson hull code, and three of its habits have no
//! direct Rust spelling:
//!
//! * **Flexible array members.** `basis_s` ends in `Coord vecs[1]` and
//!   `simplex` in `neighbor neigh[1]`, and both are `malloc`'d bigger —
//!   `basis_s_size = sizeof(basis_s) + (2*rdim-1)*sizeof(Coord)`,
//!   `simplex_size = sizeof(simplex) + (rdim-1)*sizeof(neighbor)`
//!   (`hull-ch.c:800-801`).  `rdim` is bounded by `MAXDIM` (8), so the tails
//!   become fixed `[Coord; 2 * MAXDIM]` and `[Neighbor; MAXDIM]` arrays.  The
//!   whole struct is then `Copy`, which is what `copy_simp`'s
//!   `memcpy(new, s, simplex_size)` and `FREEL`'s `memset(p, 0, X_size)` need.
//!   Copying the slots past `rdim` is not a behaviour change: nothing ever
//!   writes past `cdim - 1 <= rdim - 1`, so those slots hold zero on both
//!   sides.
//!
//! * **`STORAGE(X)` block allocators** (`stormacs.h:37`).  Each type has a
//!   `malloc`'d array of 10 000 objects chained through an `X->next` free
//!   list, and `free_X_storage` frees every block.  Here each is a `Vec<X>`
//!   inside [`HullStorage`] with the same free-list head as a `usize` index,
//!   and "free every block" is a `truncate` back to the reserved head.
//!   **Index 0 is the source's `NULL`** in every arena; nothing writes it.
//!
//! * **`s->neigh - 1` is `&s->peak`.**  `neigh` follows `peak` in the struct,
//!   so `for (i = -1, sn = s->neigh - 1; i < cdim; i++, sn++)` walks the peak
//!   and then the neighbours (`hull.c:83`, `stormacs.h:130`, `hull-io.c:169`).
//!   Every such loop here runs `i` from `-1` and selects `peak` at `-1`.
//!
//! A `site` is a `Coord*` in the source — either `NULL`, `hull_infinity`, or
//! an address inside the driver's point array.  Here it is a `usize` index
//! into [`HullStorage::sites`], so the pointer arithmetic
//! `(psite - sPointArray) / 2` in `hullwrap.c:103` stays index arithmetic.
#![allow(dead_code)]

use std::cell::{Cell, RefCell};

use crate::imod::libwarp::hull_ch::{out_of_flat, sees};
use crate::imod::libwarp::hull_io::{
    print_neighbor_full, print_simplex, print_simplex_f, print_triang,
};

/// C `MAXDIM` (`hull.h:29`).
pub const MAXDIM: usize = 8;
/// C `BLOCKSIZE` (`hull.h:30`); declared by the source and unused by it.
pub const BLOCKSIZE: i32 = 100_000;
/// C `MAXBLOCKS` (`hull.h:31`); declared by the source and unused by it.
pub const MAXBLOCKS: i32 = 1000;
/// C `DEBUG` (`hull.h:32`).  Every `DEBS(qq)` block is `if (DEBUG > qq)`, so
/// only the `-8`, `-10` and `-20` levels are compiled live.
pub const DEBUG: i32 = -7;
/// C `CHECK_OVERSHOOT` (`hull.h:33`).
pub const CHECK_OVERSHOOT: i32 = 1;
/// C `MAXPOINTS` (`hull.h:172`).
pub const MAXPOINTS: usize = 10_000;
/// C `max_blocks` (`stormacs.h:21`).
pub const MAX_BLOCKS: i32 = 10_000;
/// C `Nobj` (`stormacs.h:22`).
pub const NOBJ: usize = 10_000;

/// C `Coord` (`points.h:10`).
pub type Coord = f64;

/// C `site` / `point` (`hull.h:81`, `points.h:11`).
///
/// An index into [`HullStorage::sites`]; `0` is the source's `NULL`.
pub type Site = usize;

/// The site index of `hull_infinity` (`hull-ch.c:180`), which
/// [`HullStorage::new`] lays down at the head of the site arena.
pub const HULL_INFINITY: Site = 1;

/// C `basis_s` (`hull.h:88`).  `vecs[1]` is the flexible tail; see the module
/// comment.
#[derive(Clone, Copy, Default)]
pub struct Basis {
    /// free list
    pub next: usize,
    /// storage management
    pub ref_count: i32,
    /// the log base 2 of total scaling of vector
    pub lscale: i32,
    /// sums of squared norms of a part and b part
    pub sqa: Coord,
    pub sqb: Coord,
    /// the actual vectors, extended by malloc'ing bigger
    pub vecs: [Coord; 2 * MAXDIM],
}

/// C `neighbor` (`hull.h:98`).
#[derive(Clone, Copy, Default)]
pub struct Neighbor {
    /// vertex of simplex
    pub vert: Site,
    /// neighbor sharing all vertices but vert
    pub simp: usize,
    /// derived vectors
    pub basis: usize,
}

/// C `simplex` (`hull.h:104`).  `neigh[1]` is the flexible tail.
#[derive(Clone, Copy, Default)]
pub struct Simplex {
    /// free list
    pub next: usize,
    /// number of last site visiting this simplex
    pub visit: i32,
    /// DNM: change from short to int so it can store an index
    pub mark: i32,
    /// normal vector pointing inward
    pub normal: usize,
    /// if null, remaining vertices give facet
    pub peak: Neighbor,
    /// neighbors of simplex
    pub neigh: [Neighbor; MAXDIM],
}

/// C `tree_node` (`hull.h:118`).
#[derive(Clone, Copy, Default)]
pub struct Tree {
    pub left: usize,
    pub right: usize,
    pub key: Site,
    /// maintained to be the number of nodes rooted here
    pub size: i32,
    pub fgs: usize,
    /// freelist
    pub next: usize,
}

/// C `fg_node` (`hull.h:128`).
#[derive(Clone, Copy, Default)]
pub struct Fg {
    pub facets: usize,
    /// of Voronoi face dual to this
    pub dist: f64,
    pub vol: f64,
    /// freelist
    pub next: usize,
    pub mark: i16,
    pub ref_count: i32,
}

/// C `visit_func` (`hull.h:134`).
///
/// The source's second `void *` argument is only ever used by
/// `print_simplex`'s "latch my static `FILE *`" convention, which a captured
/// writer replaces, so it is gone; the `void *` return is a simplex index,
/// `0` for `NULL`.
pub type VisitFunc<'a> = &'a mut dyn FnMut(&mut HullStorage, usize) -> usize;

/// C `test_func` (`hull.h:135`).
pub type TestFunc<'a> = &'a mut dyn FnMut(&mut HullStorage, usize, i32) -> i32;

/// C `gsitef` (`hull.h:151`).
pub type GetSite = fn() -> Site;
/// C `site_n` (`hull.h:155`).
pub type SiteNum = fn(Site) -> i32;

/// Number of reserved slots at the head of `HullStorage::simplex`: index 0,
/// the source's `NULL`.
pub const SIMPLEX_RESERVED: usize = 1;
/// Reserved slots at the head of `HullStorage::basis`: index 0 is `NULL`, and
/// 1..=4 are `hull-ch.c`'s four bases that live outside the block pool — see
/// [`HullStorage::new`].
pub const BASIS_RESERVED: usize = 5;
/// Reserved slots at the head of `HullStorage::tree`: index 0 is `NULL` and
/// index 1 is [`SPLAY_N`].
pub const TREE_RESERVED: usize = 2;

/// The `Tree` slot standing in for `splay`'s stack-local `Tree N`
/// (`hull-fg.c:114`), whose address the source hands to `l` and `r`.  `splay`
/// is not recursive and not re-entered, so one slot is exactly one `N`.
pub const SPLAY_N: usize = 1;
/// Reserved slots at the head of `HullStorage::fg`: index 0, `NULL`.
pub const FG_RESERVED: usize = 1;
/// Reserved coordinates at the head of `HullStorage::sites`: index 0 is the
/// `NULL` site and 1..=10 are `hull_infinity`.
pub const SITES_RESERVED: usize = 11;

/// The `basis_s` index of `hull-ch.c:172`'s file-static `tt_basis`, whose
/// address the source hands out as `tt_basisp`.
pub const TT_BASIS: usize = 1;
/// The `basis_s` index behind `hull-ch.c:65`'s `sP_neigh.basis`.
pub const S_P_NEIGH_BASIS: usize = 2;
/// The `basis_s` index behind `hull-ch.c:66`'s `sB`.
pub const S_B_BASIS: usize = 3;
/// The `basis_s` index behind `check_perps`'s function-static `b`
/// (`hull-ch.c:406`).
pub const CHECK_PERPS_BASIS: usize = 4;

/// The four `STORAGE(X)` arenas (`stormacs.h:37`) plus the memory the sites
/// live in.
///
/// The source's block tables are `malloc`'d globals; this is threaded through
/// the cluster as an explicit `&mut` because the alternative — a `RefCell`
/// borrowed at every access — would panic the moment `connect`, `make_facets`,
/// `extend_simplices`, `vols` or the splay tree recursed.
pub struct HullStorage {
    pub simplex: Vec<Simplex>,
    /// C `simplex_list` (`hull.c:37`, `STORAGE(simplex)`).
    pub simplex_list: usize,
    pub basis: Vec<Basis>,
    /// C `basis_s_list` (`hull-ch.c:182`, `STORAGE(basis_s)`).
    pub basis_list: usize,
    pub tree: Vec<Tree>,
    /// C `Tree_list` (`hull-fg.c:97`, `STORAGE(Tree)`).
    pub tree_list: usize,
    pub fg: Vec<Fg>,
    /// C `fg_list` (`hull-fg.c:291`, `STORAGE(fg)`).
    pub fg_list: usize,
    /// The bytes the source's `Coord*` sites point into.
    pub sites: Vec<Coord>,
}

impl HullStorage {
    /// Lays out the reserved head of each arena.
    ///
    /// `basis[1]` is `hull-ch.c:172`'s `tt_basis = {0,1,-1,0,0,{0}}`, whose
    /// address the source stores in neighbours and reference-counts;
    /// `basis[2]`, `basis[3]` and `basis[4]` are the three bases `out_of_flat`,
    /// `sees` and `check_perps` `malloc` once and reuse for the life of the
    /// process.  None of the four is ever on the block free list, so a
    /// reserved slot is exactly what the source has.
    pub fn new() -> HullStorage {
        let mut basis = vec![Basis::default(); BASIS_RESERVED];
        basis[TT_BASIS] = Basis {
            next: 0,
            ref_count: 1,
            lscale: -1,
            sqa: 0.,
            sqb: 0.,
            vecs: [0.; 2 * MAXDIM],
        };
        let mut sites = vec![0.; SITES_RESERVED];
        /* Coord hull_infinity[10]={57.2,0,0,0,0}; point at infinity for vd;
        value not used (`hull-ch.c:180`). */
        sites[HULL_INFINITY] = 57.2;
        HullStorage {
            simplex: vec![Simplex::default(); SIMPLEX_RESERVED],
            simplex_list: 0,
            basis,
            basis_list: 0,
            tree: vec![Tree::default(); TREE_RESERVED],
            tree_list: 0,
            fg: vec![Fg::default(); FG_RESERVED],
            fg_list: 0,
            sites,
        }
    }
}

impl Default for HullStorage {
    fn default() -> HullStorage {
        HullStorage::new()
    }
}

thread_local! {
    /// The process-wide hull heap.  `hull_triangulate` is the only borrower;
    /// see [`HullStorage`].
    pub static STORAGE: RefCell<HullStorage> = RefCell::new(HullStorage::new());

    /// C `site p` (`hull.c:28`), the current site.
    pub static P: Cell<Site> = const { Cell::new(0) };
    /// C `int pnum` (`hull.c:29`).
    pub static PNUM: Cell<i32> = const { Cell::new(0) };
    /// C `int rdim` (`hull.c:31`), region dimension.
    pub static RDIM: Cell<i32> = const { Cell::new(0) };
    /// C `int cdim` (`hull.c:32`), number of sites currently specifying region.
    pub static CDIM: Cell<i32> = const { Cell::new(0) };
    /// C `int site_size` (`hull.c:33`).
    pub static SITE_SIZE: Cell<i32> = const { Cell::new(0) };
    /// C `int point_size` (`hull.c:34`).
    pub static POINT_SIZE: Cell<i32> = const { Cell::new(0) };
    /// C `gsitef *get_site` (`hull-ch.c:735`).
    pub static GET_SITE: Cell<Option<GetSite>> = const { Cell::new(None) };
    /// C `site_n *site_num` (`hull-ch.c:736`).
    pub static SITE_NUM: Cell<Option<SiteNum>> = const { Cell::new(None) };

    /// C `size_t simplex_size` (`hull.c:36`, `STORAGE(simplex)`).
    pub static SIMPLEX_SIZE: Cell<usize> = const { Cell::new(0) };
    /// C `size_t basis_s_size` (`hull-ch.c:182`).
    pub static BASIS_SIZE: Cell<usize> = const { Cell::new(0) };
    /// C `size_t Tree_size` (`hull-fg.c:97`).
    pub static TREE_SIZE: Cell<usize> = const { Cell::new(0) };
    /// C `size_t fg_size` (`hull-fg.c:291`).
    pub static FG_SIZE: Cell<usize> = const { Cell::new(0) };

    /// `static int num_simplex_blocks` inside `STORAGE(simplex)`.
    static SIMPLEX_BLOCKS: Cell<i32> = const { Cell::new(0) };
    /// `static int num_basis_s_blocks` inside `STORAGE(basis_s)`.
    static BASIS_BLOCKS: Cell<i32> = const { Cell::new(0) };
    /// `static int num_Tree_blocks` inside `STORAGE(Tree)`.
    static TREE_BLOCKS: Cell<i32> = const { Cell::new(0) };
    /// `static int num_fg_blocks` inside `STORAGE(fg)`.
    static FG_BLOCKS: Cell<i32> = const { Cell::new(0) };

    /// C `static simplex **sVTGst` (`hull.c:43`) — moved out of
    /// `visit_triang_gen` by DNM so `hullCleanup` can free it.
    static S_VTGST: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// C `static simplex **sSst` (`hull.c:44`).
    static S_SST: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// `visit_triang_gen`'s `static int vnum = -1` (`hull.c:68`).
    static VTG_VNUM: Cell<i32> = const { Cell::new(-1) };
    /// `visit_triang_gen`'s `static int ss = 2000` (`hull.c:69`).
    static VTG_SS: Cell<i32> = const { Cell::new(2000) };
    /// `search`'s `static int ss = MAXDIM` (`hull.c:255`).
    static SEARCH_SS: Cell<i32> = const { Cell::new(MAXDIM as i32) };
    /// `make_facets`'s `static simplex *ns` (`hull.c:186`).
    static MAKE_FACETS_NS: Cell<usize> = const { Cell::new(0) };
}

/// Source `STORAGE(simplex)` expansion (`hull.c:36`, `stormacs.h:37`).
pub fn new_block_simplex(st: &mut HullStorage, make_blocks: i32) -> usize {
    if make_blocks != 0 {
        assert!(SIMPLEX_BLOCKS.get() < MAX_BLOCKS);
        let block = st.simplex.len();
        SIMPLEX_BLOCKS.set(SIMPLEX_BLOCKS.get() + 1);
        /* malloc(Nobj * X_size) then memset(xbt, 0, Nobj * X_size). */
        st.simplex.resize(block + NOBJ, Simplex::default());
        /* xlm = INCP(X, xbt, Nobj); for (i=0;i<Nobj;i++) {xlm = INCP(X,xlm,-1);
        xlm->next = X_list; X_list = xlm;} -- the walk is downwards, so the
        head of the list ends up being the first object of the block. */
        let mut item = block + NOBJ;
        for _ in 0..NOBJ {
            item -= 1;
            st.simplex[item].next = st.simplex_list;
            st.simplex_list = item;
        }
        return st.simplex_list;
    }
    /* for (i=0;i<num_X_blocks;i++) free(X_block_table[i]); */
    st.simplex.truncate(SIMPLEX_RESERVED);
    SIMPLEX_BLOCKS.set(0);
    st.simplex_list = 0;
    /* `make_facets`'s static would otherwise name a truncated slot; the
    source's pointer dangles here too and is never read before it is
    reassigned.  See `hull_ch::free_hull_storage`. */
    MAKE_FACETS_NS.set(0);
    0
}

/// Source `free_simplex_storage` (`stormacs.h:82`).
pub fn free_simplex_storage(st: &mut HullStorage) {
    new_block_simplex(st, 0);
}

/// Source `STORAGE(basis_s)` expansion (`hull-ch.c:182`).
pub fn new_block_basis(st: &mut HullStorage, make_blocks: i32) -> usize {
    if make_blocks != 0 {
        assert!(BASIS_BLOCKS.get() < MAX_BLOCKS);
        let block = st.basis.len();
        BASIS_BLOCKS.set(BASIS_BLOCKS.get() + 1);
        st.basis.resize(block + NOBJ, Basis::default());
        let mut item = block + NOBJ;
        for _ in 0..NOBJ {
            item -= 1;
            st.basis[item].next = st.basis_list;
            st.basis_list = item;
        }
        return st.basis_list;
    }
    st.basis.truncate(BASIS_RESERVED);
    BASIS_BLOCKS.set(0);
    st.basis_list = 0;
    0
}

/// Source `free_basis_s_storage` (`stormacs.h:82`).
pub fn free_basis_storage(st: &mut HullStorage) {
    new_block_basis(st, 0);
}

/// Source `STORAGE(Tree)` expansion (`hull-fg.c:97`).
pub fn new_block_tree(st: &mut HullStorage, make_blocks: i32) -> usize {
    if make_blocks != 0 {
        assert!(TREE_BLOCKS.get() < MAX_BLOCKS);
        let block = st.tree.len();
        TREE_BLOCKS.set(TREE_BLOCKS.get() + 1);
        st.tree.resize(block + NOBJ, Tree::default());
        let mut item = block + NOBJ;
        for _ in 0..NOBJ {
            item -= 1;
            st.tree[item].next = st.tree_list;
            st.tree_list = item;
        }
        return st.tree_list;
    }
    st.tree.truncate(TREE_RESERVED);
    TREE_BLOCKS.set(0);
    st.tree_list = 0;
    0
}

/// Source `free_Tree_storage` (`stormacs.h:82`).
pub fn free_tree_storage(st: &mut HullStorage) {
    new_block_tree(st, 0);
}

/// Source `STORAGE(fg)` expansion (`hull-fg.c:291`).
pub fn new_block_fg(st: &mut HullStorage, make_blocks: i32) -> usize {
    if make_blocks != 0 {
        assert!(FG_BLOCKS.get() < MAX_BLOCKS);
        let block = st.fg.len();
        FG_BLOCKS.set(FG_BLOCKS.get() + 1);
        st.fg.resize(block + NOBJ, Fg::default());
        let mut item = block + NOBJ;
        for _ in 0..NOBJ {
            item -= 1;
            st.fg[item].next = st.fg_list;
            st.fg_list = item;
        }
        return st.fg_list;
    }
    st.fg.truncate(FG_RESERVED);
    FG_BLOCKS.set(0);
    st.fg_list = 0;
    0
}

/// Source `free_fg_storage` (`stormacs.h:82`).
pub fn free_fg_storage(st: &mut HullStorage) {
    new_block_fg(st, 0);
}

/// Original `hullCleanup` (`hull.c:51`).  Called by `warpfiles.c:975`.
pub fn hull_cleanup() {
    S_VTGST.with_borrow_mut(|v| {
        v.clear();
        v.shrink_to_fit();
    });
    S_SST.with_borrow_mut(|v| {
        v.clear();
        v.shrink_to_fit();
    });
}

/// Original `visit_triang_gen` (`hull.c:58`).
///
/// Starting at `s`, visit simplices `t` such that `test(s,i,0)` is true, and
/// `t` is the `i`'th neighbor of `s`; apply visit function to all visited
/// simplices; when visit returns nonNULL, exit and return its value.
pub fn visit_triang_gen(st: &mut HullStorage, s: usize, visit: VisitFunc, test: TestFunc) -> usize {
    let mut tms = 0usize;

    VTG_VNUM.set(VTG_VNUM.get() - 1);
    let vnum = VTG_VNUM.get();
    if S_VTGST.with_borrow(|v| v.is_empty()) {
        let ss = VTG_SS.get();
        S_VTGST.with_borrow_mut(|v| v.resize(ss as usize + MAXDIM + 1, 0));
    }
    if s != 0 {
        S_VTGST.with_borrow_mut(|v| v[tms] = s);
        tms += 1;
    }
    while tms != 0 {
        if tms > VTG_SS.get() as usize {
            /* DEBEXP(-1, tms) -- DEBUG (-7) > -1 is false. */
            VTG_SS.set(VTG_SS.get() + VTG_SS.get());
            let ss = VTG_SS.get();
            S_VTGST.with_borrow_mut(|v| v.resize(ss as usize + MAXDIM + 1, 0));
        }
        tms -= 1;
        let t = S_VTGST.with_borrow(|v| v[tms]);
        if t == 0 || st.simplex[t].visit == vnum {
            continue;
        }
        st.simplex[t].visit = vnum;
        /* DNM: add suggested parens */
        let v = visit(st, t);
        if v != 0 {
            return v;
        }
        let cdim = CDIM.get();
        for i in -1..cdim {
            let sn = if i < 0 {
                st.simplex[t].peak
            } else {
                st.simplex[t].neigh[i as usize]
            };
            /* The source reads `sn->simp->visit` before testing `sn->simp`;
            index 0 is a real slot here, and its `visit` is never written, so
            the read is defined and the second test still rejects it. */
            if st.simplex[sn.simp].visit != vnum && sn.simp != 0 && test(st, t, i) != 0 {
                S_VTGST.with_borrow_mut(|v| v[tms] = sn.simp);
                tms += 1;
            }
        }
    }
    0
}

/// Original static `truet` (`hull.c:94`).
fn truet(_st: &mut HullStorage, _s: usize, _i: i32) -> i32 {
    1
}

/// Original `visit_triang` (`hull.c:96`) — visit the whole triangulation.
pub fn visit_triang(st: &mut HullStorage, root: usize, visit: VisitFunc) -> usize {
    visit_triang_gen(st, root, visit, &mut |st, s, i| truet(st, s, i))
}

/// Original static `hullt` (`hull.c:101`).
fn hullt(_st: &mut HullStorage, _s: usize, i: i32) -> i32 {
    (i > -1) as i32
}

/// Original static `facet_test` (`hull.c:103`).
fn facet_test(st: &mut HullStorage, s: usize) -> usize {
    if st.simplex[s].peak.vert == 0 { s } else { 0 }
}

/// Original `visit_hull` (`hull.c:105`) — visit all simplices with facets of
/// the current hull.
pub fn visit_hull(st: &mut HullStorage, root: usize, visit: VisitFunc) -> usize {
    let start = visit_triang(st, root, &mut |st, s| facet_test(st, s));
    visit_triang_gen(st, start, visit, &mut |st, s, i| hullt(st, s, i))
}

/// Original `op_simp` (`hull.c:129`) — the neighbor entry of `a` containing
/// `b`.  Returns the neighbour's index within `a`, `-1` standing for `peak`;
/// the source returns a `neighbor *` into the simplex, which an arena cannot.
pub fn op_simp(st: &mut HullStorage, a: usize, b: usize) -> i32 {
    /* lookup(a,b,simp,simplex) (`hull.c:112`).  Note the macro starts at
    `a->neigh`, not `a->neigh - 1`. */
    let cdim = CDIM.get();
    let mut i = 0;
    while st.simplex[a].neigh[i as usize].simp != b && i < cdim {
        i += 1;
    }
    if i < cdim {
        return i;
    }
    let mut dfile = crate::imod::libcfshr::b3dutil::ImodFile::Stderr;
    let _ = std::io::Write::write_all(&mut dfile, b"adjacency failure,op_simp:\n");
    /* DEBTR(-10) */
    let _ = std::io::Write::write_all(
        &mut dfile,
        crate::imod::libcfshr::b3dutil::c_format(
            "hull.c line %d \n",
            &[crate::imod::libcfshr::b3dutil::CArg::Int(132)],
        )
        .as_bytes(),
    );
    let _ = std::io::Write::flush(&mut dfile);
    print_simplex_f(st, a, &mut dfile, Some(print_neighbor_full));
    print_simplex(st, b, &mut dfile);
    let _ = std::io::Write::write_all(&mut dfile, b"---------------------\n");
    print_triang(st, a, &mut dfile, print_neighbor_full);
    std::process::exit(1)
}

/// Original `op_vert` (`hull.c:132`) — the neighbor entry of `a` containing
/// `b`.  Returns the neighbour's index within `a`.
pub fn op_vert(st: &mut HullStorage, a: usize, b: Site) -> i32 {
    /* lookup(a,b,vert,site) (`hull.c:112`). */
    let cdim = CDIM.get();
    let mut i = 0;
    while st.simplex[a].neigh[i as usize].vert != b && i < cdim {
        i += 1;
    }
    if i < cdim {
        return i;
    }
    let mut dfile = crate::imod::libcfshr::b3dutil::ImodFile::Stderr;
    let _ = std::io::Write::write_all(&mut dfile, b"adjacency failure,op_vert:\n");
    let _ = std::io::Write::write_all(
        &mut dfile,
        crate::imod::libcfshr::b3dutil::c_format(
            "hull.c line %d \n",
            &[crate::imod::libcfshr::b3dutil::CArg::Int(135)],
        )
        .as_bytes(),
    );
    let _ = std::io::Write::flush(&mut dfile);
    print_simplex_f(st, a, &mut dfile, Some(print_neighbor_full));
    crate::imod::libwarp::hull_ch::print_site(st, b, &mut dfile);
    let _ = std::io::Write::write_all(&mut dfile, b"---------------------\n");
    print_triang(st, a, &mut dfile, print_neighbor_full);
    std::process::exit(1)
}

/// Original static `connect` (`hull.c:139`) — make neighbor connections
/// between newly created simplices incident to `p`.
fn connect(st: &mut HullStorage, s: usize) {
    if s == 0 {
        return;
    }
    /* assert(!s->peak.vert && s->peak.simp->peak.vert==p
       && !op_vert(s,p)->simp->peak.vert); -- NDEBUG is not set in the vendored
    build, but the assertion has no side effect beyond `op_vert`, whose only
    effect is the diagnostic exit it cannot reach here. */
    let pnum = PNUM.get();
    if st.simplex[s].visit == pnum {
        return;
    }
    st.simplex[s].visit = pnum;
    let seen = st.simplex[s].peak.simp;
    let op = op_simp(st, seen, s);
    let xfi = neighbor_of(st, seen, op).vert;
    let p = P.get();
    let cdim = CDIM.get();
    for i in 0..cdim {
        let mut xb = st.simplex[s].neigh[i as usize].vert;
        if p == xb {
            continue;
        }
        let mut sb = seen;
        let mut sf = st.simplex[s].neigh[i as usize].simp;
        let mut xf = xfi;
        if st.simplex[sf].peak.vert == 0 {
            /* are we done already? */
            let k = op_vert(st, seen, xb);
            sf = neighbor_of(st, seen, k).simp;
            if st.simplex[sf].peak.vert != 0 {
                continue;
            }
        } else {
            loop {
                xb = xf;
                let k = op_simp(st, sf, sb);
                xf = neighbor_of(st, sf, k).vert;
                sb = sf;
                let k = op_vert(st, sb, xb);
                sf = neighbor_of(st, sb, k).simp;
                if st.simplex[sf].peak.vert == 0 {
                    break;
                }
            }
        }

        st.simplex[s].neigh[i as usize].simp = sf;
        let k = op_vert(st, sf, xf);
        set_neighbor_simp(st, sf, k, s);

        connect(st, sf);
    }
}

/// The source's `sn = s->neigh + i` read, with `i == -1` naming `s->peak`; see
/// the module comment.
#[inline]
fn neighbor_of(st: &HullStorage, s: usize, i: i32) -> Neighbor {
    if i < 0 {
        st.simplex[s].peak
    } else {
        st.simplex[s].neigh[i as usize]
    }
}

/// The write half of [`neighbor_of`] for the `->simp` field.
#[inline]
fn set_neighbor_simp(st: &mut HullStorage, s: usize, i: i32, value: usize) {
    if i < 0 {
        st.simplex[s].peak.simp = value;
    } else {
        st.simplex[s].neigh[i as usize].simp = value;
    }
}

/// Original static `make_facets` (`hull.c:181`) — visit simplices `s` with
/// `sees(p,s)`, and make a facet for every neighbor of `s` not seen by `p`.
fn make_facets(st: &mut HullStorage, seen: usize) -> usize {
    if seen == 0 {
        return 0;
    }
    /* DEBS(-1) assert(sees(p,seen) && !seen->peak.vert); EDEBS */
    let p = P.get();
    st.simplex[seen].peak.vert = p;

    let cdim = CDIM.get();
    let pnum = PNUM.get();
    for i in 0..cdim {
        let n = st.simplex[seen].neigh[i as usize].simp;
        if pnum != st.simplex[n].visit {
            st.simplex[n].visit = pnum;
            if sees(st, p, n) != 0 {
                make_facets(st, n);
            }
        }
        if st.simplex[n].peak.vert != 0 {
            continue;
        }
        /* copy_simp(ns, seen) (`stormacs.h:139`): NEWL then
        memcpy(new, s, simplex_size) then mod_refs(inc, s). */
        if st.simplex_list == 0 {
            new_block_simplex(st, 1);
        }
        let ns = st.simplex_list;
        assert!(ns != 0);
        st.simplex_list = st.simplex[ns].next;
        MAKE_FACETS_NS.set(ns);
        /* The whole struct is copied, including the neighbour slots past
        `rdim`; nothing ever writes them, so they are zero on both sides. */
        st.simplex[ns] = st.simplex[seen];
        for imr in -1..cdim {
            let b = neighbor_of(st, seen, imr).basis;
            if b != 0 {
                st.basis[b].ref_count += 1;
            }
        }
        st.simplex[ns].visit = 0;
        st.simplex[ns].peak.vert = 0;
        st.simplex[ns].normal = 0;
        st.simplex[ns].peak.simp = seen;
        /* ns->Sb -= ns->neigh[i].basis->sqb; */
        /* NULLIFY(basis_s, ns->neigh[i].basis) */
        let b = st.simplex[ns].neigh[i as usize].basis;
        if b != 0 {
            st.basis[b].ref_count -= 1;
            if st.basis[b].ref_count == 0 {
                let next = st.basis_list;
                st.basis[b] = Basis::default();
                st.basis[b].next = next;
                st.basis_list = b;
            }
        }
        st.simplex[ns].neigh[i as usize].basis = 0;
        st.simplex[ns].neigh[i as usize].vert = p;
        st.simplex[seen].neigh[i as usize].simp = ns;
        let k = op_simp(st, n, seen);
        set_neighbor_simp(st, n, k, ns);
    }
    MAKE_FACETS_NS.get()
}

/// Original static `extend_simplices` (`hull.c:219`).
///
/// `p` lies outside flat containing previous sites; make `p` a vertex of every
/// current simplex, and create some new simplices.
fn extend_simplices(st: &mut HullStorage, s: usize) -> usize {
    let cdim = CDIM.get();
    let ocdim = cdim - 1;
    let pnum = PNUM.get();
    let p = P.get();

    if st.simplex[s].visit == pnum {
        return if st.simplex[s].peak.vert != 0 {
            st.simplex[s].neigh[ocdim as usize].simp
        } else {
            s
        };
    }
    st.simplex[s].visit = pnum;
    st.simplex[s].neigh[ocdim as usize].vert = p;
    /* NULLIFY(basis_s, s->normal) */
    let b = st.simplex[s].normal;
    if b != 0 {
        st.basis[b].ref_count -= 1;
        if st.basis[b].ref_count == 0 {
            let next = st.basis_list;
            st.basis[b] = Basis::default();
            st.basis[b].next = next;
            st.basis_list = b;
        }
    }
    st.simplex[s].normal = 0;
    /* NULLIFY(basis_s, s->neigh[0].basis) */
    let b = st.simplex[s].neigh[0].basis;
    if b != 0 {
        st.basis[b].ref_count -= 1;
        if st.basis[b].ref_count == 0 {
            let next = st.basis_list;
            st.basis[b] = Basis::default();
            st.basis[b].next = next;
            st.basis_list = b;
        }
    }
    st.simplex[s].neigh[0].basis = 0;

    if st.simplex[s].peak.vert == 0 {
        let peak_simp = st.simplex[s].peak.simp;
        let ext = extend_simplices(st, peak_simp);
        st.simplex[s].neigh[ocdim as usize].simp = ext;
        return s;
    }

    /* copy_simp(ns, s) */
    if st.simplex_list == 0 {
        new_block_simplex(st, 1);
    }
    let ns = st.simplex_list;
    assert!(ns != 0);
    st.simplex_list = st.simplex[ns].next;
    st.simplex[ns] = st.simplex[s];
    for imr in -1..cdim {
        let b = neighbor_of(st, s, imr).basis;
        if b != 0 {
            st.basis[b].ref_count += 1;
        }
    }
    st.simplex[s].neigh[ocdim as usize].simp = ns;
    st.simplex[ns].peak.vert = 0;
    st.simplex[ns].peak.simp = s;
    st.simplex[ns].neigh[ocdim as usize] = st.simplex[s].peak;
    /* inc_ref(basis_s, s->peak.basis) */
    let b = st.simplex[s].peak.basis;
    if b != 0 {
        st.basis[b].ref_count += 1;
    }
    for i in 0..cdim {
        let nsn = st.simplex[ns].neigh[i as usize].simp;
        let ext = extend_simplices(st, nsn);
        st.simplex[ns].neigh[i as usize].simp = ext;
    }
    ns
}

/// Original static `search` (`hull.c:252`) — return a simplex `s` that
/// corresponds to a facet of the current hull, and `sees(p, s)`.
fn search(st: &mut HullStorage, root: usize) -> usize {
    let mut tms = 0usize;

    if S_SST.with_borrow(|v| v.is_empty()) {
        let ss = SEARCH_SS.get();
        S_SST.with_borrow_mut(|v| v.resize(ss as usize + MAXDIM + 1, 0));
    }
    let peak = st.simplex[root].peak.simp;
    S_SST.with_borrow_mut(|v| v[tms] = peak);
    tms += 1;
    let pnum = PNUM.get();
    let p = P.get();
    st.simplex[root].visit = pnum;
    let cdim = CDIM.get();
    if sees(st, p, root) == 0 {
        for i in 0..cdim {
            let sn = st.simplex[root].neigh[i as usize].simp;
            S_SST.with_borrow_mut(|v| v[tms] = sn);
            tms += 1;
        }
    }
    while tms != 0 {
        if tms > SEARCH_SS.get() as usize {
            SEARCH_SS.set(SEARCH_SS.get() + SEARCH_SS.get());
            let ss = SEARCH_SS.get();
            S_SST.with_borrow_mut(|v| v.resize(ss as usize + MAXDIM + 1, 0));
        }
        tms -= 1;
        let s = S_SST.with_borrow(|v| v[tms]);
        if st.simplex[s].visit == pnum {
            continue;
        }
        st.simplex[s].visit = pnum;
        if sees(st, p, s) == 0 {
            continue;
        }
        if st.simplex[s].peak.vert == 0 {
            return s;
        }
        for i in 0..cdim {
            let sn = st.simplex[s].neigh[i as usize].simp;
            S_SST.with_borrow_mut(|v| v[tms] = sn);
            tms += 1;
        }
    }
    0
}

/// Original static `get_another_site` (`hull.c:282`).
fn get_another_site() -> Site {
    /* static int scount = 0;
    if (!(++scount%1)) {fprintf(DFILE,"site %d...", scount);}
    check_triang(); */
    let pnext = GET_SITE.get().unwrap()();
    if pnext == 0 {
        return 0;
    }
    PNUM.set(SITE_NUM.get().unwrap()(pnext) + 2);
    pnext
}

/// Original `buildhull` (`hull.c:297`).
pub fn buildhull(st: &mut HullStorage, root: usize) {
    while CDIM.get() < RDIM.get() {
        P.set(get_another_site());
        if P.get() == 0 {
            return;
        }
        let p = P.get();
        if out_of_flat(st, root, p) != 0 {
            extend_simplices(st, root);
        } else {
            let s = search(st, root);
            let f = make_facets(st, s);
            connect(st, f);
        }
    }
    /* DNM: add suggested parens */
    loop {
        P.set(get_another_site());
        if P.get() == 0 {
            break;
        }
        let s = search(st, root);
        let f = make_facets(st, s);
        connect(st, f);
    }
}
