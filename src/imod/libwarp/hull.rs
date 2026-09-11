//! Translation of `IMOD/libwarp/hull.c` and `hull.h`.
#![allow(dead_code)]

use crate::imod::libwarp::hull_ch::sees;
use core::ffi::{c_int, c_void};

pub const MAXDIM: usize = 8;
pub const NOBJ: usize = 10_000;

/// C `Coord` and `point` (`points.h`).
pub type Coord = f64;
pub type Site = *mut Coord;

/// C `basis_s` (`hull.h`), with the source flexible `vecs[1]` tail.
#[repr(C)]
pub struct Basis {
    pub next: *mut Basis,
    pub ref_count: c_int,
    pub lscale: c_int,
    pub sqa: Coord,
    pub sqb: Coord,
    pub vecs: [Coord; 1],
}

/// C `neighbor` (`hull.h`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Neighbor {
    pub vert: Site,
    pub simp: *mut Simplex,
    pub basis: *mut Basis,
}

/// C `simplex` (`hull.h`), with the source flexible `neigh[1]` tail.
#[repr(C)]
pub struct Simplex {
    pub next: *mut Simplex,
    pub visit: c_int,
    pub mark: c_int,
    pub normal: *mut Basis,
    pub peak: Neighbor,
    pub neigh: [Neighbor; 1],
}

/// C `tree_node` (`hull.h`).
#[repr(C)]
pub struct Tree {
    pub left: *mut Tree,
    pub right: *mut Tree,
    pub key: Site,
    pub size: c_int,
    pub fgs: *mut Fg,
    pub next: *mut Tree,
}

/// C `fg_node` (`hull.h`).
#[repr(C)]
pub struct Fg {
    pub facets: *mut Tree,
    pub dist: f64,
    pub vol: f64,
    pub next: *mut Fg,
    pub mark: i16,
    pub ref_count: c_int,
}

pub type VisitFunc = unsafe fn(*mut Simplex, *mut c_void) -> *mut c_void;
pub type TestFunc = unsafe fn(*mut Simplex, c_int, *mut c_void) -> c_int;
pub type GetSite = unsafe fn() -> Site;
pub type SiteNum = unsafe fn(Site) -> c_int;

/// C global state from `hull.c`, `hull-ch.c`, and `hullwrap.c`.
pub static mut P: Site = core::ptr::null_mut();
pub static mut PNUM: c_int = 0;
pub static mut RDIM: c_int = 0;
pub static mut CDIM: c_int = 0;
pub static mut SITE_SIZE: c_int = 0;
pub static mut POINT_SIZE: c_int = 0;
pub static mut GET_SITE: Option<GetSite> = None;
pub static mut SITE_NUM: Option<SiteNum> = None;
pub static mut SIMPLEX_SIZE: usize = 0;
pub static mut BASIS_SIZE: usize = 0;
pub static mut SIMPLEX_LIST: *mut Simplex = core::ptr::null_mut();
pub static mut BASIS_LIST: *mut Basis = core::ptr::null_mut();
pub static mut TREE_LIST: *mut Tree = core::ptr::null_mut();
pub static mut FG_LIST: *mut Fg = core::ptr::null_mut();

/// Source `STORAGE(simplex)` expansion (`hull.c:38`).
pub unsafe fn new_block_simplex(make_blocks: c_int) -> *mut Simplex {
    unsafe {
        static mut BLOCKS: [*mut Simplex; 10_000] = [core::ptr::null_mut(); 10_000];
        static mut NUM_BLOCKS: c_int = 0;
        if make_blocks != 0 {
            assert!(NUM_BLOCKS < 10_000);
            let block = libc::malloc(NOBJ * SIMPLEX_SIZE).cast::<Simplex>();
            BLOCKS[NUM_BLOCKS as usize] = block;
            NUM_BLOCKS += 1;
            libc::memset(block.cast(), 0, NOBJ * SIMPLEX_SIZE);
            assert!(!block.is_null());
            let mut item = block
                .cast::<u8>()
                .add(NOBJ * SIMPLEX_SIZE)
                .cast::<Simplex>();
            for _ in 0..NOBJ {
                item = item.cast::<u8>().sub(SIMPLEX_SIZE).cast::<Simplex>();
                (*item).next = SIMPLEX_LIST;
                SIMPLEX_LIST = item;
            }
            return SIMPLEX_LIST;
        }
        for index in 0..NUM_BLOCKS {
            libc::free(BLOCKS[index as usize].cast());
        }
        NUM_BLOCKS = 0;
        SIMPLEX_LIST = core::ptr::null_mut();
        core::ptr::null_mut()
    }
}

/// Source `STORAGE(simplex)` expansion (`hull.c:38`).
pub unsafe fn free_simplex_storage() {
    unsafe {
        new_block_simplex(0);
    }
}

/// Source `STORAGE(basis_s)` expansion (`hull-ch.c:152`).
pub unsafe fn new_block_basis(make_blocks: c_int) -> *mut Basis {
    unsafe {
        static mut BLOCKS: [*mut Basis; 10_000] = [core::ptr::null_mut(); 10_000];
        static mut NUM_BLOCKS: c_int = 0;
        if make_blocks != 0 {
            assert!(NUM_BLOCKS < 10_000);
            let block = libc::malloc(NOBJ * BASIS_SIZE).cast::<Basis>();
            BLOCKS[NUM_BLOCKS as usize] = block;
            NUM_BLOCKS += 1;
            libc::memset(block.cast(), 0, NOBJ * BASIS_SIZE);
            assert!(!block.is_null());
            let mut item = block.cast::<u8>().add(NOBJ * BASIS_SIZE).cast::<Basis>();
            for _ in 0..NOBJ {
                item = item.cast::<u8>().sub(BASIS_SIZE).cast::<Basis>();
                (*item).next = BASIS_LIST;
                BASIS_LIST = item;
            }
            return BASIS_LIST;
        }
        for index in 0..NUM_BLOCKS {
            libc::free(BLOCKS[index as usize].cast());
        }
        NUM_BLOCKS = 0;
        BASIS_LIST = core::ptr::null_mut();
        core::ptr::null_mut()
    }
}

/// Source `STORAGE(basis_s)` expansion (`hull-ch.c:152`).
pub unsafe fn free_basis_storage() {
    unsafe {
        new_block_basis(0);
    }
}

/// Source `STORAGE(Tree)` expansion (`hull-fg.c:97`).
pub unsafe fn new_block_tree(make_blocks: c_int) -> *mut Tree {
    unsafe {
        static mut BLOCKS: [*mut Tree; 10_000] = [core::ptr::null_mut(); 10_000];
        static mut COUNT: i32 = 0;
        static mut SIZE: usize = 0;
        if make_blocks != 0 {
            if SIZE == 0 {
                SIZE = core::mem::size_of::<Tree>();
            }
            assert!(COUNT < 10_000);
            let block = libc::malloc(NOBJ * SIZE).cast::<Tree>();
            BLOCKS[COUNT as usize] = block;
            COUNT += 1;
            libc::memset(block.cast(), 0, NOBJ * SIZE);
            assert!(!block.is_null());
            let mut item = block.cast::<u8>().add(NOBJ * SIZE).cast::<Tree>();
            for _ in 0..NOBJ {
                item = item.cast::<u8>().sub(SIZE).cast();
                (*item).next = TREE_LIST;
                TREE_LIST = item;
            }
            return TREE_LIST;
        }
        for index in 0..COUNT {
            libc::free(BLOCKS[index as usize].cast());
        }
        COUNT = 0;
        TREE_LIST = core::ptr::null_mut();
        core::ptr::null_mut()
    }
}

/// Source `free_Tree_storage` macro expansion (`hull-fg.c:97`).
pub unsafe fn free_tree_storage() {
    unsafe {
        new_block_tree(0);
    }
}

/// Source `STORAGE(fg)` expansion (`hull-fg.c:291`).
pub unsafe fn new_block_fg(make_blocks: c_int) -> *mut Fg {
    unsafe {
        static mut BLOCKS: [*mut Fg; 10_000] = [core::ptr::null_mut(); 10_000];
        static mut COUNT: i32 = 0;
        if make_blocks != 0 {
            assert!(COUNT < 10_000);
            let block = libc::malloc(NOBJ * core::mem::size_of::<Fg>()).cast::<Fg>();
            BLOCKS[COUNT as usize] = block;
            COUNT += 1;
            libc::memset(block.cast(), 0, NOBJ * core::mem::size_of::<Fg>());
            assert!(!block.is_null());
            let mut item = block.add(NOBJ);
            for _ in 0..NOBJ {
                item = item.sub(1);
                (*item).next = FG_LIST;
                FG_LIST = item;
            }
            return FG_LIST;
        }
        for index in 0..COUNT {
            libc::free(BLOCKS[index as usize].cast());
        }
        COUNT = 0;
        FG_LIST = core::ptr::null_mut();
        core::ptr::null_mut()
    }
}

/// Source `free_fg_storage` macro expansion (`hull-fg.c:291`).
pub unsafe fn free_fg_storage() {
    unsafe {
        new_block_fg(0);
    }
}

/// Original `visit_triang_gen` (`hull.c:58`).
/// `hull.c:43-44`.  The comment there records that these were deliberately
/// moved out of `visit_triang_gen` and `search` to file scope, and individual
/// push/pop macros provided, specifically so `hullCleanup` can free them
/// before exit.  Keeping them as function-local statics makes that impossible.
static mut S_VTGST: *mut *mut Simplex = core::ptr::null_mut();
static mut S_SST: *mut *mut Simplex = core::ptr::null_mut();

/// Original `hullCleanup` (`hull.c:51`).  Called by `warpfiles.c:975`.
pub unsafe fn hull_cleanup() {
    unsafe {
        libc::free(S_VTGST.cast());
        S_VTGST = core::ptr::null_mut();
        libc::free(S_SST.cast());
        S_SST = core::ptr::null_mut();
    }
}

pub unsafe fn visit_triang_gen(
    simplex: *mut Simplex,
    visit: VisitFunc,
    test: TestFunc,
) -> *mut c_void {
    unsafe {
        static mut VNUM: i32 = -1;
        static mut SIZE: i32 = 2000;
        let mut top = 0;
        VNUM -= 1;
        if S_VTGST.is_null() {
            S_VTGST =
                libc::malloc((SIZE as usize + MAXDIM + 1) * core::mem::size_of::<*mut Simplex>())
                    .cast();
        }
        if !simplex.is_null() {
            *S_VTGST.add(top) = simplex;
            top += 1;
        }
        while top != 0 {
            if top > SIZE as usize {
                SIZE += SIZE;
                S_VTGST = libc::realloc(
                    S_VTGST.cast(),
                    (SIZE as usize + MAXDIM + 1) * core::mem::size_of::<*mut Simplex>(),
                )
                .cast();
            }
            top -= 1;
            let current = *S_VTGST.add(top);
            if current.is_null() || (*current).visit == VNUM {
                continue;
            }
            (*current).visit = VNUM;
            let result = visit(current, core::ptr::null_mut());
            if !result.is_null() {
                return result;
            }
            for index in -1..CDIM {
                let neighbour = (*current).neigh.as_mut_ptr().offset(index as isize);
                if !(*neighbour).simp.is_null()
                    && (*(*neighbour).simp).visit != VNUM
                    && test(current, index, core::ptr::null_mut()) != 0
                {
                    *S_VTGST.add(top) = (*neighbour).simp;
                    top += 1;
                }
            }
        }
        core::ptr::null_mut()
    }
}

/// Original static `truet` (`hull.c:94`).
unsafe fn truet(_: *mut Simplex, _: c_int, _: *mut c_void) -> c_int {
    1
}
/// Original `visit_triang` (`hull.c:96`).
pub unsafe fn visit_triang(root: *mut Simplex, visit: VisitFunc) -> *mut c_void {
    unsafe { visit_triang_gen(root, visit, truet) }
}
/// Original static `hullt` (`hull.c:101`).
unsafe fn hullt(_: *mut Simplex, index: c_int, _: *mut c_void) -> c_int {
    (index > -1) as i32
}
/// Original static `facet_test` (`hull.c:103`).
unsafe fn facet_test(simplex: *mut Simplex, _: *mut c_void) -> *mut c_void {
    unsafe {
        if (*simplex).peak.vert.is_null() {
            simplex.cast()
        } else {
            core::ptr::null_mut()
        }
    }
}
/// Original `visit_hull` (`hull.c:105`).
pub unsafe fn visit_hull(root: *mut Simplex, visit: VisitFunc) -> *mut c_void {
    unsafe { visit_triang_gen(visit_triang(root, facet_test).cast(), visit, hullt) }
}

/// Original `op_simp` (`hull.c:129`).
pub unsafe fn op_simp(simplex: *mut Simplex, other: *mut Simplex) -> *mut Neighbor {
    unsafe {
        for index in 0..CDIM {
            let neighbour = (*simplex).neigh.as_mut_ptr().add(index as usize);
            if (*neighbour).simp == other {
                return neighbour;
            }
        }
        /* The `lookup` macro's failure arm (`hull.c:119-127`).  `DEBTR(-10)`
        is taken because `DEBUG` is -7 (`hull.h:30`). */
        let dfile = crate::imod::libwarp::hull_io::DFILE;
        libc::fprintf(dfile, c"adjacency failure,op_simp:\n".as_ptr());
        libc::fprintf(dfile, c"hull.c line %d \n".as_ptr(), 121);
        libc::fflush(dfile);
        crate::imod::libwarp::hull_io::print_simplex_f(
            simplex,
            dfile,
            Some(crate::imod::libwarp::hull_io::print_neighbor_full),
        );
        crate::imod::libwarp::hull_io::print_simplex(other, dfile.cast());
        libc::fprintf(dfile, c"---------------------\n".as_ptr());
        crate::imod::libwarp::hull_io::print_triang(
            simplex,
            dfile,
            Some(crate::imod::libwarp::hull_io::print_neighbor_full),
        );
        libc::exit(1)
    }
}

/// Original `op_vert` (`hull.c:132`).
pub unsafe fn op_vert(simplex: *mut Simplex, vertex: Site) -> *mut Neighbor {
    unsafe {
        for index in 0..CDIM {
            let neighbour = (*simplex).neigh.as_mut_ptr().add(index as usize);
            if (*neighbour).vert == vertex {
                return neighbour;
            }
        }
        /* The `lookup` macro's failure arm (`hull.c:119-127`), with
        `print_site` for the `whatt` = `site` expansion. */
        let dfile = crate::imod::libwarp::hull_io::DFILE;
        libc::fprintf(dfile, c"adjacency failure,op_vert:\n".as_ptr());
        libc::fprintf(dfile, c"hull.c line %d \n".as_ptr(), 121);
        libc::fflush(dfile);
        crate::imod::libwarp::hull_io::print_simplex_f(
            simplex,
            dfile,
            Some(crate::imod::libwarp::hull_io::print_neighbor_full),
        );
        crate::imod::libwarp::hull_ch::print_site(vertex, dfile);
        libc::fprintf(dfile, c"---------------------\n".as_ptr());
        crate::imod::libwarp::hull_io::print_triang(
            simplex,
            dfile,
            Some(crate::imod::libwarp::hull_io::print_neighbor_full),
        );
        libc::exit(1)
    }
}

/// Original static `get_another_site` (`hull.c:282`).
unsafe fn get_another_site() -> Site {
    unsafe {
        let point = GET_SITE.unwrap()();
        if point.is_null() {
            return core::ptr::null_mut();
        }
        PNUM = SITE_NUM.unwrap()(point) + 2;
        point
    }
}

/// Original static `connect` (`hull.c:139`).
unsafe fn connect(simplex: *mut Simplex) {
    unsafe {
        if simplex.is_null() {
            return;
        }
        if (*simplex).visit == PNUM {
            return;
        }
        (*simplex).visit = PNUM;
        let seen = (*simplex).peak.simp;
        let xfi = (*op_simp(seen, simplex)).vert;
        for index in 0..CDIM {
            let neighbour = (*simplex).neigh.as_mut_ptr().add(index as usize);
            let mut xb = (*neighbour).vert;
            if P == xb {
                continue;
            }
            let mut previous = seen;
            let mut next = (*neighbour).simp;
            let mut xf = xfi;
            if (*next).peak.vert.is_null() {
                next = (*op_vert(seen, xb)).simp;
                if !(*next).peak.vert.is_null() {
                    continue;
                }
            } else {
                while !(*next).peak.vert.is_null() {
                    xb = xf;
                    xf = (*op_simp(next, previous)).vert;
                    previous = next;
                    next = (*op_vert(previous, xb)).simp;
                }
            }
            (*neighbour).simp = next;
            (*op_vert(next, xf)).simp = simplex;
            connect(next);
        }
    }
}

/// Original static `make_facets` (`hull.c:181`).
unsafe fn make_facets(seen: *mut Simplex) -> *mut Simplex {
    unsafe {
        static mut NEW_SIMPLEX: *mut Simplex = core::ptr::null_mut();
        if seen.is_null() {
            return core::ptr::null_mut();
        }
        (*seen).peak.vert = P;
        for index in 0..CDIM {
            let boundary = (*seen).neigh.as_mut_ptr().add(index as usize);
            let neighbour = (*boundary).simp;
            if PNUM != (*neighbour).visit {
                (*neighbour).visit = PNUM;
                if sees(P, neighbour) != 0 {
                    make_facets(neighbour);
                }
            }
            if !(*neighbour).peak.vert.is_null() {
                continue;
            }
            if SIMPLEX_LIST.is_null() {
                new_block_simplex(1);
            }
            NEW_SIMPLEX = SIMPLEX_LIST;
            SIMPLEX_LIST = (*SIMPLEX_LIST).next;
            libc::memcpy(NEW_SIMPLEX.cast(), seen.cast(), SIMPLEX_SIZE);
            for offset in -1..CDIM {
                let copied = (*NEW_SIMPLEX).neigh.as_mut_ptr().offset(offset as isize);
                if !(*copied).basis.is_null() {
                    (*(*copied).basis).ref_count += 1;
                }
            }
            (*NEW_SIMPLEX).visit = 0;
            (*NEW_SIMPLEX).peak.vert = core::ptr::null_mut();
            (*NEW_SIMPLEX).normal = core::ptr::null_mut();
            (*NEW_SIMPLEX).peak.simp = seen;
            let new_neighbour = (*NEW_SIMPLEX).neigh.as_mut_ptr().add(index as usize);
            if !(*new_neighbour).basis.is_null() {
                (*(*new_neighbour).basis).ref_count -= 1;
            }
            (*new_neighbour).basis = core::ptr::null_mut();
            (*new_neighbour).vert = P;
            (*boundary).simp = NEW_SIMPLEX;
            (*op_simp(neighbour, seen)).simp = NEW_SIMPLEX;
        }
        NEW_SIMPLEX
    }
}

/// Original static `extend_simplices` (`hull.c:219`).
unsafe fn extend_simplices(simplex: *mut Simplex) -> *mut Simplex {
    unsafe {
        let old_dimension = CDIM - 1;
        if (*simplex).visit == PNUM {
            return if !(*simplex).peak.vert.is_null() {
                (*simplex)
                    .neigh
                    .as_mut_ptr()
                    .add(old_dimension as usize)
                    .read()
                    .simp
            } else {
                simplex
            };
        }
        (*simplex).visit = PNUM;
        let last = (*simplex).neigh.as_mut_ptr().add(old_dimension as usize);
        (*last).vert = P;
        if !(*simplex).normal.is_null() {
            (*(*simplex).normal).ref_count -= 1;
            (*simplex).normal = core::ptr::null_mut();
        }
        let first = (*simplex).neigh.as_mut_ptr();
        if !(*first).basis.is_null() {
            (*(*first).basis).ref_count -= 1;
            (*first).basis = core::ptr::null_mut();
        }
        if (*simplex).peak.vert.is_null() {
            (*last).simp = extend_simplices((*simplex).peak.simp);
            return simplex;
        }
        if SIMPLEX_LIST.is_null() {
            new_block_simplex(1);
        }
        let new_simplex = SIMPLEX_LIST;
        SIMPLEX_LIST = (*SIMPLEX_LIST).next;
        libc::memcpy(new_simplex.cast(), simplex.cast(), SIMPLEX_SIZE);
        for index in -1..CDIM {
            let copied = (*new_simplex).neigh.as_mut_ptr().offset(index as isize);
            if !(*copied).basis.is_null() {
                (*(*copied).basis).ref_count += 1;
            }
        }
        (*last).simp = new_simplex;
        (*new_simplex).peak.vert = core::ptr::null_mut();
        (*new_simplex).peak.simp = simplex;
        *(*new_simplex)
            .neigh
            .as_mut_ptr()
            .add(old_dimension as usize) = (*simplex).peak;
        if !(*simplex).peak.basis.is_null() {
            (*(*simplex).peak.basis).ref_count += 1;
        }
        for index in 0..CDIM {
            let neighbour = (*new_simplex).neigh.as_mut_ptr().add(index as usize);
            (*neighbour).simp = extend_simplices((*neighbour).simp);
        }
        new_simplex
    }
}

/// Original static `search` (`hull.c:252`).
unsafe fn search(root: *mut Simplex) -> *mut Simplex {
    unsafe {
        static mut SIZE: i32 = MAXDIM as i32;
        let mut top = 0_usize;
        if S_SST.is_null() {
            S_SST =
                libc::malloc((SIZE as usize + MAXDIM + 1) * core::mem::size_of::<*mut Simplex>())
                    .cast();
        }
        *S_SST.add(top) = (*root).peak.simp;
        top += 1;
        (*root).visit = PNUM;
        if sees(P, root) == 0 {
            for index in 0..CDIM {
                *S_SST.add(top) = (*root).neigh.as_mut_ptr().add(index as usize).read().simp;
                top += 1;
            }
        }
        while top != 0 {
            if top > SIZE as usize {
                SIZE += SIZE;
                S_SST = libc::realloc(
                    S_SST.cast(),
                    (SIZE as usize + MAXDIM + 1) * core::mem::size_of::<*mut Simplex>(),
                )
                .cast();
            }
            top -= 1;
            let simplex = *S_SST.add(top);
            if (*simplex).visit == PNUM {
                continue;
            }
            (*simplex).visit = PNUM;
            if sees(P, simplex) == 0 {
                continue;
            }
            if (*simplex).peak.vert.is_null() {
                return simplex;
            }
            for index in 0..CDIM {
                *S_SST.add(top) = (*simplex)
                    .neigh
                    .as_mut_ptr()
                    .add(index as usize)
                    .read()
                    .simp;
                top += 1;
            }
        }
        core::ptr::null_mut()
    }
}

/// Original `buildhull` (`hull.c:297`).
pub unsafe fn buildhull(root: *mut Simplex) {
    unsafe {
        while CDIM < RDIM {
            P = get_another_site();
            if P.is_null() {
                return;
            }
            if crate::imod::libwarp::hull_ch::out_of_flat(root, P) != 0 {
                extend_simplices(root);
            } else {
                connect(make_facets(search(root)));
            }
        }
        loop {
            P = get_another_site();
            if P.is_null() {
                break;
            }
            connect(make_facets(search(root)));
        }
    }
}
