//! Translation of `IMOD/libwarp/hullwrap.c` and `hullwrap.h` — David
//! Mastronarde's wrapper around the Clarkson hull Delaunay triangulation.
//!
//! Two deviations, both forced by the arena model in [`HullStorage`]:
//!
//! * The source keeps `sPointArray = hio->pointlist`, i.e. the sites point
//!   straight into the caller's array.  A site is an index into the hull's own
//!   coordinate arena here, so `hull_triangulate` copies the point list in and
//!   records its base; `site_numm`'s `(psite - sPointArray) / 2` stays the
//!   same subtraction.  Nothing reads the caller's array back afterwards.
//!
//! * The source keeps `sHio`, a file-static alias to the caller's `struct
//!   hullio`, and reads `verbose`, `heightBaseCrit` and `areaFractionCrit`
//!   from inside the visit callbacks while `hull_triangulate` still holds the
//!   struct.  A `&mut HullIo` cannot be aliased, so those three are hoisted
//!   into thread-locals at the top of `hull_triangulate` and the two output
//!   lists are built in thread-locals and moved into `hio` at the end.
#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libwarp::hull::{
    HULL_INFINITY, HullStorage, SITES_RESERVED, STORAGE, Site, visit_hull,
};
use crate::imod::libwarp::hull_ch::{build_convex_hull, free_hull_storage};

/// C `struct hullio` (`hullwrap.h:12`).
pub struct HullIo {
    pub pointlist: Vec<f64>,
    pub numberofpoints: i32,
    /// The source's `int *trianglelist`; `NULL` is an empty `Vec` here.
    pub trianglelist: Vec<i32>,
    /// The source's `int *neighborlist`.
    pub neighborlist: Vec<i32>,
    pub numberoftriangles: i32,
    pub height_base_crit: f64,
    pub area_fraction_crit: f64,
    pub min_num_for_pruning: i32,
    pub verbose: i32,
}

thread_local! {
    /// C `static double *sPointArray` (`hullwrap.c:17`), as the site index of
    /// the first coordinate.
    static S_POINT_ARRAY: Cell<usize> = const { Cell::new(SITES_RESERVED) };
    /// The coordinates `sPointArray` points at.
    ///
    /// The hull arena holds the same values and is what every site read goes
    /// through; this second copy exists only because `get_next_site`'s
    /// `verbose > 1` report reads two coordinates from *inside*
    /// `hull_triangulate`'s borrow of the arena, which a `RefCell` cannot
    /// re-enter.
    static S_POINT_LIST: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// C `static int sNumPoints` (`hullwrap.c:18`).
    static S_NUM_POINTS: Cell<i32> = const { Cell::new(0) };
    /// C `static int sNextPoint` (`hullwrap.c:19`).
    static S_NEXT_POINT: Cell<i32> = const { Cell::new(0) };
    /// C `static int sNumTri` (`hullwrap.c:20`).
    static S_NUM_TRI: Cell<i32> = const { Cell::new(0) };
    /// C `static int sListInd` (`hullwrap.c:21`).
    static S_LIST_IND: Cell<i32> = const { Cell::new(0) };
    /// C `static int sNumPrune` (`hullwrap.c:22`).
    static S_NUM_PRUNE: Cell<i32> = const { Cell::new(0) };
    /// C `static int sPruneCrit` (`hullwrap.c:23`).
    static S_PRUNE_CRIT: Cell<i32> = const { Cell::new(0) };
    /// C `static double sTotalArea` (`hullwrap.c:24`).
    static S_TOTAL_AREA: Cell<f64> = const { Cell::new(0.) };
    /// `sHio->verbose` (`hullwrap.c:25`); see the module comment.
    static S_VERBOSE: Cell<i32> = const { Cell::new(0) };
    /// `sHio->heightBaseCrit`.
    static S_HEIGHT_BASE_CRIT: Cell<f64> = const { Cell::new(0.) };
    /// `sHio->areaFractionCrit`.
    static S_AREA_FRACTION_CRIT: Cell<f64> = const { Cell::new(0.) };
    /// `sHio->trianglelist`, filled by `saveTriangle`.
    static S_TRIANGLELIST: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    /// `sHio->neighborlist`.
    static S_NEIGHBORLIST: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
}

/// Original `hull_triangulate` (`hullwrap.c:38`) — the external call to do the
/// triangulation.
pub fn hull_triangulate(hio: &mut HullIo) -> i32 {
    STORAGE.with_borrow_mut(|st| {
        /* DFILE = stderr: every write in this cluster names ImodFile::Stderr
        directly, so there is nothing to assign. */
        S_NUM_POINTS.set(hio.numberofpoints);
        S_NEXT_POINT.set(0);
        S_VERBOSE.set(hio.verbose);
        S_HEIGHT_BASE_CRIT.set(hio.height_base_crit);
        S_AREA_FRACTION_CRIT.set(hio.area_fraction_crit);

        /* sPointArray = hio->pointlist: the sites live in the hull arena. */
        st.sites.truncate(SITES_RESERVED);
        S_POINT_ARRAY.set(SITES_RESERVED);
        st.sites.extend_from_slice(&hio.pointlist);
        S_POINT_LIST.with_borrow_mut(|v| {
            v.clear();
            v.extend_from_slice(&hio.pointlist);
        });

        let mut err = ImodFile::Stderr;
        if hio.verbose != 0 {
            let _ = err.write_all(
                c_format(
                    "Number of points %d\n",
                    &[CArg::Int(S_NUM_POINTS.get() as i64)],
                )
                .as_bytes(),
            );
        }

        let root = build_convex_hull(st, get_next_site, site_numm, 2, 1);

        S_NUM_TRI.set(0);
        S_NUM_PRUNE.set(0);
        S_TOTAL_AREA.set(0.);
        S_PRUNE_CRIT.set(-1);
        visit_hull(st, root, &mut |st, s| mark_outside_get_area(st, s));

        /* Prune triangles at edge if there are enough and criteria are set */
        if S_NUM_TRI.get() >= hio.min_num_for_pruning
            && hio.area_fraction_crit > 0.
            && hio.height_base_crit > 0.
        {
            visit_hull(st, root, &mut |st, s| mark_for_pruning(st, s));
            S_PRUNE_CRIT.set(0);
        }

        S_NUM_TRI.set(0);
        visit_hull(st, root, &mut |st, s| set_triangle_number(st, s));
        if hio.verbose != 0 {
            let _ = err.write_all(
                c_format(
                    "%d triangles,  %d pruned at edge, total areax2 %.0f\n",
                    &[
                        CArg::Int(S_NUM_TRI.get() as i64),
                        CArg::Int(S_NUM_PRUNE.get() as i64),
                        CArg::Dbl(S_TOTAL_AREA.get()),
                    ],
                )
                .as_bytes(),
            );
        }

        /* Allocate output arrays */
        hio.numberoftriangles = S_NUM_TRI.get();
        hio.trianglelist = Vec::new();
        hio.neighborlist = Vec::new();
        if S_NUM_TRI.get() == 0 {
            free_hull_storage(st);
            return 0;
        }

        /* The source's two `malloc`s and their shared NULL check, which a
        `Vec` cannot fail. */
        S_TRIANGLELIST.with_borrow_mut(|v| {
            v.clear();
            v.resize(3 * S_NUM_TRI.get() as usize, 0);
        });
        S_NEIGHBORLIST.with_borrow_mut(|v| {
            v.clear();
            v.resize(3 * S_NUM_TRI.get() as usize, 0);
        });

        /* Build the lists */
        S_LIST_IND.set(0);
        visit_hull(st, root, &mut |st, s| save_triangle(st, s));

        hio.trianglelist = S_TRIANGLELIST.with_borrow(|v| v.clone());
        hio.neighborlist = S_NEIGHBORLIST.with_borrow(|v| v.clone());

        free_hull_storage(st);
        0
    })
}

/// Original static `site_numm` (`hullwrap.c:96`) — return the point number
/// given its address.
fn site_numm(psite: Site) -> i32 {
    if psite == HULL_INFINITY {
        return -1;
    }
    if psite == 0 {
        return -2;
    }
    let j = (psite as isize - S_POINT_ARRAY.get() as isize) / 2;
    if j < 0 || j >= S_NUM_POINTS.get() as isize {
        return -3;
    }
    if S_VERBOSE.get() > 1 {
        let mut err = ImodFile::Stderr;
        let _ = err.write_all(
            c_format(
                "site_numm returning %d for %p\n",
                &[CArg::Int(j as i64), CArg::Ptr(psite)],
            )
            .as_bytes(),
        );
    }
    j as i32
}

/// Original static `get_next_site` (`hullwrap.c:112`) — return the next point
/// in the array.
fn get_next_site() -> Site {
    if S_NEXT_POINT.get() >= S_NUM_POINTS.get() {
        return 0;
    }
    let site = S_POINT_ARRAY.get() + 2 * S_NEXT_POINT.get() as usize;
    if S_VERBOSE.get() > 1 {
        let n = S_NEXT_POINT.get() as usize;
        let (x, y) = S_POINT_LIST.with_borrow(|v| (v[2 * n], v[2 * n + 1]));
        let mut err = ImodFile::Stderr;
        let _ = err.write_all(
            c_format(
                "get_next_site returning %d  %f %f\n",
                &[
                    CArg::Int(S_NEXT_POINT.get() as i64),
                    CArg::Dbl(x),
                    CArg::Dbl(y),
                ],
            )
            .as_bytes(),
        );
    }
    S_NEXT_POINT.set(S_NEXT_POINT.get() + 1);
    site
}

/// Original static `markOutsideGetArea` (`hullwrap.c:127`).
///
/// Mark triangles that connect hull to infinity, and compute a total area.
/// Sadly, it did not work to save an area as a structure member of simplex;
/// after storing it here, it was zero on next visit.
fn mark_outside_get_area(st: &mut HullStorage, s: usize) -> usize {
    if s == 0 {
        return 0;
    }
    st.simplex[s].mark = -2;
    for i in 0..3 {
        if site_numm(st.simplex[s].neigh[i].vert) < 0 {
            return 0;
        }
    }

    st.simplex[s].mark = 0;
    let v0 = st.simplex[s].neigh[0].vert;
    let v1 = st.simplex[s].neigh[1].vert;
    let v2 = st.simplex[s].neigh[2].vert;
    let dx10 = st.sites[v1] - st.sites[v0];
    let dx20 = st.sites[v2] - st.sites[v0];
    let dy10 = st.sites[v1 + 1] - st.sites[v0 + 1];
    let dy20 = st.sites[v2 + 1] - st.sites[v0 + 1];
    S_TOTAL_AREA.set(S_TOTAL_AREA.get() + (dx10 * dy20 - dx20 * dy10).abs());
    S_NUM_TRI.set(S_NUM_TRI.get() + 1);
    0
}

/// Original static `markForPruning` (`hullwrap.c:153`) — check triangles at
/// the hull edge for adequate height-base ratio or fraction of total area.
fn mark_for_pruning(st: &mut HullStorage, s: usize) -> usize {
    if s == 0 || st.simplex[s].mark == 0 {
        return 0;
    }
    for i in 0..3usize {
        if site_numm(st.simplex[s].neigh[i].vert) < 0 {
            let sn = st.simplex[s].neigh[i].simp;
            if sn != 0 {
                let ip1 = (i + 1) % 3;
                let ip2 = (i + 2) % 3;
                let p1 = st.simplex[s].neigh[ip1].vert;
                let p2 = st.simplex[s].neigh[ip2].vert;
                let dxb = st.sites[p1] - st.sites[p2];
                let dyb = st.sites[p1 + 1] - st.sites[p2 + 1];
                let basesq = dxb * dxb + dyb * dyb;
                let n0 = st.simplex[sn].neigh[0].vert;
                let n1 = st.simplex[sn].neigh[1].vert;
                let n2 = st.simplex[sn].neigh[2].vert;
                let dx10 = st.sites[n1] - st.sites[n0];
                let dx20 = st.sites[n2] - st.sites[n0];
                let dy10 = st.sites[n1 + 1] - st.sites[n0 + 1];
                let dy20 = st.sites[n2 + 1] - st.sites[n0 + 1];
                let areax2 = (dx10 * dy20 - dx20 * dy10).abs();
                let mut err = ImodFile::Stderr;
                if S_VERBOSE.get() != 0 {
                    let base = basesq.sqrt();
                    let _ = err.write_all(
                        c_format(
                            "edge %.0f,%.0f to %.0f,%.0f, base %.1f, area %.0f, hgt %.1f  h/b %.3f",
                            &[
                                CArg::Dbl(st.sites[p1]),
                                CArg::Dbl(st.sites[p1 + 1]),
                                CArg::Dbl(st.sites[p2]),
                                CArg::Dbl(st.sites[p2 + 1]),
                                CArg::Dbl(base),
                                CArg::Dbl(areax2),
                                CArg::Dbl(areax2 / base),
                                CArg::Dbl(areax2 / basesq),
                            ],
                        )
                        .as_bytes(),
                    );
                }
                if areax2 / basesq < S_HEIGHT_BASE_CRIT.get()
                    && areax2 / S_TOTAL_AREA.get() < S_AREA_FRACTION_CRIT.get()
                {
                    st.simplex[sn].mark = -1;
                    S_NUM_PRUNE.set(S_NUM_PRUNE.get() + 1);
                    if S_VERBOSE.get() != 0 {
                        let _ = err.write_all(b" pruned");
                    }
                }
                if S_VERBOSE.get() != 0 {
                    let mut out = ImodFile::Stdout;
                    let _ = out.write_all(b" \n");
                }
            }
            return 0;
        }
    }
    0
}

/// Original static `setTriangleNumber` (`hullwrap.c:194`) — count the retained
/// triangles, and set mark value to the triangle number.
fn set_triangle_number(st: &mut HullStorage, s: usize) -> usize {
    if s == 0 || st.simplex[s].mark < S_PRUNE_CRIT.get() {
        return 0;
    }
    st.simplex[s].mark = S_NUM_TRI.get();
    S_NUM_TRI.set(S_NUM_TRI.get() + 1);
    0
}

/// Original static `saveTriangle` (`hullwrap.c:203`) — save each triangle's
/// vertices and neighbors to arrays in counterclockwise order.
fn save_triangle(st: &mut HullStorage, s: usize) -> usize {
    let mut ist = 0i32;
    let mut iend = 2i32;
    let mut idir = 1i32;
    if s == 0 || st.simplex[s].mark < S_PRUNE_CRIT.get() {
        return 0;
    }

    /* Test for counterclockwise triangle */
    let v0 = st.simplex[s].neigh[0].vert;
    let v1 = st.simplex[s].neigh[1].vert;
    let v2 = st.simplex[s].neigh[2].vert;
    let dx10 = st.sites[v1] - st.sites[v0];
    let dx20 = st.sites[v2] - st.sites[v0];
    let dy10 = st.sites[v1 + 1] - st.sites[v0 + 1];
    let dy20 = st.sites[v2 + 1] - st.sites[v0 + 1];
    if dx10 * dy20 - dx20 * dy10 < 0. {
        idir = -1;
        ist = 2;
        iend = 0;
    }

    /* Put the vertices and neighbors into arrays in the right order */
    let mut i = ist;
    while idir * (iend - i) >= 0 {
        let ind = S_LIST_IND.get() as usize;
        let vert = st.simplex[s].neigh[i as usize].vert;
        S_TRIANGLELIST.with_borrow_mut(|v| v[ind] = site_numm(vert));
        let nsimp = st.simplex[s].neigh[i as usize].simp;
        S_NEIGHBORLIST.with_borrow_mut(|v| v[ind] = st.simplex[nsimp].mark);
        S_LIST_IND.set(S_LIST_IND.get() + 1);
        i += idir;
    }
    0
}
