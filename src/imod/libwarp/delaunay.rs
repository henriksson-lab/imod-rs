//! Translation of `IMOD/libwarp/delaunay.c` and `delaunay.h`.
//!
//! `IMOD/libwarp/Makefile:34` compiles `delaunay.c` with `-DCLARKSON_HULL`, so
//! the `#ifdef CLARKSON_HULL` arm is the one translated here: the backend is
//! `hullwrap`'s Clarkson hull, not Shewchuk's `triangle`. The `#else` arm's
//! `tio_init`, `triangulate()` call, edge list and the eleven
//! `triangulateio`-only fields it frees have no counterpart in the built
//! library and are not translated.
#![allow(dead_code)]

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libwarp::hullwrap::{HullIo, hull_triangulate};
use crate::imod::libwarp::istack::Istack;
use crate::imod::libwarp::istack::{istack_create, istack_pop, istack_push, istack_reset};
use crate::imod::libwarp::nn::{NN_VERBOSE, Point};
use crate::imod::libwarp::nncommon::circle_contains;
use crate::imod::libwarp::nncommon_vulnerable::circle_build1;

/// C `N_SEARCH_TURNON` (`delaunay.c:53`).
const N_SEARCH_TURNON: i32 = 20;
/// C `N_FLAGS_TURNON` (`delaunay.c:54`); declared by the source and unused.
const N_FLAGS_TURNON: i32 = 1000;
/// C `N_FLAGS_INC` (`delaunay.c:55`); the increment `delaunay_addflag` grew
/// `flagids` by, which is the `Vec`'s own growth now and not observable.
const N_FLAGS_INC: i32 = 100;

/// C `triangle` (`delaunay.h:29`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Triangle {
    pub vids: [i32; 3],
}

/// C `triangle_neighbours` (`delaunay.h:33`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TriangleNeighbours {
    pub tids: [i32; 3],
}

/// C `circle` (`delaunay.h:37`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Circle {
    pub x: f64,
    pub y: f64,
    pub r: f64,
}

/// C `struct delaunay` (`delaunay.h:60`).
///
/// Every `malloc`'d array becomes a `Vec`, and `int** point_triangles` — an
/// array of per-point arrays — becomes `Vec<Vec<i32>>`, which is the same
/// shape without the two-level free. The one real change of ownership is
/// `points`: the C keeps a *shallow* alias to the caller's array
/// (`delaunay.c:355`, "only shallow copy of the input data is contained in
/// struct delaunay"), and `delaunay_destroy` deliberately does not free it.
/// Here the triangulation owns its copy. Only `nnai_interpolate` and
/// `nnhpi_modify_data` write through that alias, and nothing in this tree
/// reads the caller's array back afterwards.
///
/// The source's counts are the lengths of the arrays they counted, so they
/// are not fields: `npoints` is `points.len()`, `ntriangles` is
/// `triangles.len()` (`circles`, `neighbours` and `flags` are the same
/// length), `n_point_triangles[i]` is `point_triangles[i].len()`, `nedges` is
/// `edges.len() / 2` — always 0 for the Clarkson hull — and `nflags` is
/// `flagids.len()`, with `nflagsallocated` being the `Vec`'s own capacity.
pub struct Delaunay {
    /// `[npoints]`
    pub points: Vec<Point>,
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,

    /// `[ntriangles]`
    pub triangles: Vec<Triangle>,
    /// `[ntriangles]`
    pub circles: Vec<Circle>,
    /// for `delaunay_xytoi()`; `[ntriangles]`
    pub neighbours: Vec<TriangleNeighbours>,

    /// `point_triangles[i][j]` is index of j-th triangle i-th point belongs to
    pub point_triangles: Vec<Vec<i32>>,

    /// n-th edge is formed by `points[edges[n*2]]` and `points[edges[n*2+1]]`
    pub edges: Vec<i32>,

    /*
     * Work data for delaunay_circles_find(). Placed here for efficiency
     * reasons. Should be moved to the procedure if parallelizable code
     * needed.
     */
    /// `[ntriangles]`
    pub flags: Vec<i32>,
    /// last search result, used in start up of a new search
    pub first_id: i32,
    pub t_in: Option<Istack>,
    pub t_out: Option<Istack>,

    /*
     * to keep track of flags set to 1 in the case of very large data sets
     */
    pub flagids: Vec<i32>,
}

/// Original static `delaunay_create` (`delaunay.c:127`).
fn delaunay_create() -> Delaunay {
    let d = Delaunay {
        points: Vec::new(),
        xmin: f64::MAX,
        xmax: -f64::MAX,
        ymin: f64::MAX,
        ymax: -f64::MAX,
        triangles: Vec::new(),
        circles: Vec::new(),
        neighbours: Vec::new(),
        point_triangles: Vec::new(),
        edges: Vec::new(),
        flags: Vec::new(),
        first_id: -1,
        t_in: None,
        t_out: None,
        flagids: Vec::new(),
    };

    d
}

/// Original static `tio_destroy`, `CLARKSON_HULL` arm (`delaunay.c:104`).
fn tio_destroy(tio: &mut HullIo) {
    // `free(tio->pointlist)`, `free(tio->trianglelist)`,
    // `free(tio->neighborlist)`, each guarded by a NULL test that an empty
    // `Vec` makes unnecessary.
    tio.pointlist = Vec::new();
    tio.trianglelist = Vec::new();
    tio.neighborlist = Vec::new();
}

/// Original static `tio2delaunay`, `CLARKSON_HULL` arm (`delaunay.c:157`).
///
/// The `#ifndef CLARKSON_HULL` assertions on `tio_out->numberofpoints` and the
/// edge-list copy at the end belong to the `triangle` backend and are not in
/// the built library.
fn tio2delaunay(tio_out: &mut HullIo, d: &mut Delaunay) {
    for i in 0..d.points.len() {
        let p = &d.points[i];

        if p.x < d.xmin {
            d.xmin = p.x;
        }
        if p.x > d.xmax {
            d.xmax = p.x;
        }
        if p.y < d.ymin {
            d.ymin = p.y;
        }
        if p.y > d.ymax {
            d.ymax = p.y;
        }
    }
    if NN_VERBOSE.get() != 0 {
        let mut err = ImodFile::Stderr;
        let _ = err.write_all(b"input:\n");
        for i in 0..d.points.len() {
            let p = &d.points[i];

            let _ = err.write_all(
                c_format(
                    "  %d: %15.7g %15.7g %15.7g\n",
                    &[
                        CArg::Int(i as i64),
                        CArg::Dbl(p.x),
                        CArg::Dbl(p.y),
                        CArg::Dbl(p.z),
                    ],
                )
                .as_bytes(),
            );
        }
    }

    /* d->ntriangles = tio_out->numberoftriangles: three vertex indices per
    triangle in the hull's list. */
    let ntriangles = (tio_out.trianglelist.len() / 3) as i32;
    if ntriangles > 0 {
        d.triangles = vec![Triangle::default(); ntriangles as usize];
        d.neighbours = vec![TriangleNeighbours::default(); ntriangles as usize];
        d.circles = vec![Circle::default(); ntriangles as usize];
        d.point_triangles = vec![Vec::new(); d.points.len()];
        d.flags = vec![0i32; ntriangles as usize];
    }

    if NN_VERBOSE.get() != 0 {
        let _ = ImodFile::Stderr.write_all(b"triangles:\n");
    }
    for i in 0..ntriangles {
        let offset = i * 3;
        let mut t = Triangle::default();
        let mut nb = TriangleNeighbours::default();
        let mut c = Circle::default();
        let _status: i32;

        t.vids[0] = tio_out.trianglelist[offset as usize];
        t.vids[1] = tio_out.trianglelist[(offset + 1) as usize];
        t.vids[2] = tio_out.trianglelist[(offset + 2) as usize];

        nb.tids[0] = tio_out.neighborlist[offset as usize];
        nb.tids[1] = tio_out.neighborlist[(offset + 1) as usize];
        nb.tids[2] = tio_out.neighborlist[(offset + 2) as usize];

        _status = circle_build1(
            &mut c,
            &d.points[t.vids[0] as usize],
            &d.points[t.vids[1] as usize],
            &d.points[t.vids[2] as usize],
        );
        // `assert(status)`; the source builds with assertions on.
        assert!(_status != 0);

        d.triangles[i as usize] = t;
        d.neighbours[i as usize] = nb;
        d.circles[i as usize] = c;

        if NN_VERBOSE.get() != 0 {
            let _ = ImodFile::Stderr.write_all(
                c_format(
                    "  %d: (%d,%d,%d)\n",
                    &[
                        CArg::Int(i as i64),
                        CArg::Int(t.vids[0] as i64),
                        CArg::Int(t.vids[1] as i64),
                        CArg::Int(t.vids[2] as i64),
                    ],
                )
                .as_bytes(),
            );
        }
    }

    /* The source counts each point's triangles into n_point_triangles,
    allocates point_triangles[i] to that count, zeroes the counts and fills
    the arrays in a third pass over the triangles.  The fill order is the
    same here, and the count is the array's length. */
    for i in 0..ntriangles {
        let t = d.triangles[i as usize];

        for j in 0..3 {
            let vid = t.vids[j];

            d.point_triangles[vid as usize].push(i);
        }
    }
}

/// Original `delaunay_build`, `CLARKSON_HULL` arm (`delaunay.c:293`).
///
/// For Clarkson Hull there are no holes or segments; to prune triangles at the
/// edge of the hull, `ns` carries the minimum number of triangles to prune
/// with and `holes` carries the two criteria, a height/base ratio and a
/// fraction of total area.
pub fn delaunay_build(
    np: i32,
    points: &[Point],
    ns: i32,
    _segments: Option<&[i32]>,
    nh: i32,
    holes: Option<&[f64]>,
) -> Option<Delaunay> {
    let mut d = delaunay_create();
    let mut tio_in = HullIo {
        pointlist: Vec::new(),
        trianglelist: Vec::new(),
        neighborlist: Vec::new(),
        height_base_crit: 0.,
        area_fraction_crit: 0.,
        min_num_for_pruning: 0,
        verbose: 0,
    };

    if np == 0 {
        // `free(d); return NULL;`
        drop(d);
        return None;
    }

    tio_in.pointlist = vec![0.0f64; (np * 2) as usize];
    /* tio_in.numberofpoints = np: the pair count of `pointlist`. */
    let mut j = 0usize;
    for i in 0..np {
        tio_in.pointlist[j] = points[i as usize].x;
        j += 1;
        tio_in.pointlist[j] = points[i as usize].y;
        j += 1;
    }

    if NN_VERBOSE.get() != 0 {
        let _ = ImodFile::Stderr.flush();
    }

    /*
     * climax
     */
    tio_in.verbose = NN_VERBOSE.get();
    tio_in.height_base_crit = 0.;
    tio_in.min_num_for_pruning = 2;
    tio_in.area_fraction_crit = 0.;
    if nh > 0 && holes.is_some() {
        let holes = holes.unwrap();
        tio_in.min_num_for_pruning = ns;
        tio_in.height_base_crit = holes[0];
        tio_in.area_fraction_crit = holes[1];
    }
    hull_triangulate(&mut tio_in);

    if NN_VERBOSE.get() != 0 {
        let _ = ImodFile::Stderr.flush();
    }

    /* d->npoints = np; d->points = points: the source aliases the caller's
    array and bounds every read by `np`, so the copy is exactly the first `np`
    points and `d.points.len()` is `npoints`. */
    d.points = points[..np as usize].to_vec();

    tio2delaunay(&mut tio_in, &mut d);
    tio_destroy(&mut tio_in);

    Some(d)
}

/// Original `delaunay_destroy` (`delaunay.c:372`).
pub fn delaunay_destroy(d: Option<Delaunay>) {
    if d.is_none() {
        return;
    }

    // The source's nine guarded `free`s, the two `istack_destroy`s and the
    // final `free(d)` are all the one drop here.
    drop(d);
}

/// Original static `onrightside` (`delaunay.c:405`).
///
/// Returns whether the point p is on the right side of the vector (p0, p1).
fn onrightside(p: &Point, p0: &Point, p1: &Point) -> i32 {
    ((p1.x - p.x) * (p0.y - p.y) > (p0.x - p.x) * (p1.y - p.y)) as i32
}

/// Original `delaunay_xytoi` (`delaunay.c:417`).
///
/// Two source-level degeneracies live here and neither is fixed, because
/// fixing either would change what the program does:
///
/// * With a **collinear** input the hull returns no triangles, `d->triangles`
///   stays NULL, and `t = &d->triangles[id]` (`delaunay.c:433`) dereferences
///   it for any query inside the — degenerate — bounding box. Verified against
///   the vendored C: four collinear points and a query on the line SIGSEGVs.
///   Safe Rust cannot reproduce a NULL dereference; this panics on the index
///   instead. That is the one behaviour difference, and it is UB on the C side.
/// * The `do { for (i…) } while (i < 3)` walk can cycle between two degenerate
///   triangles and never terminate. Reproduced as written.
pub fn delaunay_xytoi(d: &Delaunay, p: &Point, id: i32) -> i32 {
    let mut t: Triangle;
    let mut i: i32;
    let mut id = id;

    if p.x < d.xmin || p.x > d.xmax || p.y < d.ymin || p.y > d.ymax {
        return -1;
    }

    if id < 0 || id > d.triangles.len() as i32 {
        id = 0;
    }
    t = d.triangles[id as usize];
    loop {
        i = 0;
        while i < 3 {
            let i1 = (i + 1) % 3;

            if onrightside(
                p,
                &d.points[t.vids[i as usize] as usize],
                &d.points[t.vids[i1 as usize] as usize],
            ) != 0
            {
                id = d.neighbours[id as usize].tids[((i + 2) % 3) as usize];
                if id < 0 {
                    return id;
                }
                t = d.triangles[id as usize];
                break;
            }
            i += 1;
        }
        if i >= 3 {
            break;
        }
    }

    id
}

/// Original static `delaunay_addflag` (`delaunay.c:445`).
fn delaunay_addflag(d: &mut Delaunay, i: i32) {
    /* The source grows `flagids` by N_FLAGS_INC when nflags reaches
    nflagsallocated, then stores at [nflags++]. */
    d.flagids.push(i);
}

/// Original static `delaunay_resetflags` (`delaunay.c:455`).
fn delaunay_resetflags(d: &mut Delaunay) {
    for i in 0..d.flagids.len() {
        let id = d.flagids[i];
        d.flags[id as usize] = 0;
    }
    /* d->nflags = 0 */
    d.flagids.clear();
}

/// Original `delaunay_circles_find` (`delaunay.c:483`).
///
/// Finds all tricircles the specified point belongs to.
///
/// The C hands the caller `*out = d->t_out->v`, a live borrow of the
/// triangulation's own stack, which cannot coexist with the `&mut Delaunay`
/// every caller also holds. `out` is filled with the same `*n` values instead.
/// The two failure paths that set `*out = NULL` leave it empty with `*n == 0`.
pub fn delaunay_circles_find(d: &mut Delaunay, p: &Point, n: &mut i32, out: &mut Vec<i32>) {
    /*
     * This flag was introduced as a hack to handle some degenerate cases. It
     * is set to 1 only if the triangle associated with the first circle is
     * already known to contain the point. In this case the circle is assumed
     * to contain the point without a check. In my practice this turned
     * useful in some cases when point p coincided with one of the vertices
     * of a thin triangle.
     */
    let mut contains = 0;
    let mut i: i32;

    if d.t_in.is_none() {
        d.t_in = Some(istack_create());
        d.t_out = Some(istack_create());
    }

    /*
     * if there are only a few data points, do linear search
     */
    if d.triangles.len() as i32 <= N_SEARCH_TURNON {
        istack_reset(d.t_out.as_mut().unwrap());

        for i in 0..d.triangles.len() as i32 {
            let c = d.circles[i as usize];
            if circle_contains(&c, p) != 0 {
                istack_push(d.t_out.as_mut().unwrap(), i);
            }
        }

        let t_out = d.t_out.as_ref().unwrap();
        *n = t_out.v.len() as i32;
        out.clear();
        out.extend_from_slice(&t_out.v);

        return;
    }
    /*
     * otherwise, do a more complicated stuff
     */

    /*
     * It is important to have a reasonable seed here. If the last search
     * was successful -- start with the last found tricircle, otherwhile (i)
     * try to find a triangle containing p; if fails then (ii) check
     * tricircles from the last search; if fails then (iii) make linear
     * search through all tricircles
     */
    if d.first_id < 0 || circle_contains(&d.circles[d.first_id as usize], p) == 0 {
        /*
         * if any triangle contains p -- start with this triangle
         */
        d.first_id = delaunay_xytoi(d, p, d.first_id);
        contains = (d.first_id >= 0) as i32;

        /*
         * if no triangle contains p, there still is a chance that it is
         * inside some of circumcircles
         */
        if d.first_id < 0 {
            let nn = d.t_out.as_ref().unwrap().v.len() as i32;
            let mut tid = -1;

            /*
             * first check results of the last search
             */
            i = 0;
            while i < nn {
                tid = d.t_out.as_ref().unwrap().v[i as usize];
                if circle_contains(&d.circles[tid as usize], p) != 0 {
                    break;
                }
                i += 1;
            }
            /*
             * if unsuccessful, search through all circles
             */
            if tid < 0 || i == nn {
                let nt = d.triangles.len() as f64;

                tid = 0;
                while (tid as f64) < nt {
                    if circle_contains(&d.circles[tid as usize], p) != 0 {
                        break;
                    }
                    tid += 1;
                }
                if (tid as f64) == nt {
                    istack_reset(d.t_out.as_mut().unwrap());
                    *n = 0;
                    out.clear();
                    return; /* failed */
                }
            }
            d.first_id = tid;
        }
    }

    istack_reset(d.t_in.as_mut().unwrap());
    istack_reset(d.t_out.as_mut().unwrap());

    let first_id = d.first_id;
    istack_push(d.t_in.as_mut().unwrap(), first_id);
    d.flags[first_id as usize] = 1;
    delaunay_addflag(d, first_id);

    /*
     * main cycle
     */
    while d.t_in.as_ref().unwrap().v.len() > 0 {
        let tid = istack_pop(d.t_in.as_mut().unwrap());
        let t = d.triangles[tid as usize];

        if contains != 0 || circle_contains(&d.circles[tid as usize], p) != 0 {
            istack_push(d.t_out.as_mut().unwrap(), tid);
            for i in 0..3 {
                let vid = t.vids[i as usize];
                let nt = d.point_triangles[vid as usize].len();

                for j in 0..nt {
                    let ntid = d.point_triangles[vid as usize][j];

                    if d.flags[ntid as usize] == 0 {
                        istack_push(d.t_in.as_mut().unwrap(), ntid);
                        d.flags[ntid as usize] = 1;
                        delaunay_addflag(d, ntid);
                    }
                }
            }
        }
        contains = 0;
    }

    {
        let t_out = d.t_out.as_ref().unwrap();
        *n = t_out.v.len() as i32;
        out.clear();
        out.extend_from_slice(&t_out.v);
    }
    delaunay_resetflags(d);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_backend_preserves_source_bounds_when_no_triangle_is_possible() {
        let points = [
            Point {
                x: -2.,
                y: 4.,
                z: 0.,
            },
            Point {
                x: 3.,
                y: -1.,
                z: 0.,
            },
        ];
        let d = delaunay_build(2, &points, 0, None, 0, None).unwrap();
        assert_eq!(d.triangles.len(), 0);
        assert_eq!(d.xmin, -2.);
        assert_eq!(d.xmax, 3.);
        assert_eq!(d.ymin, -1.);
        assert_eq!(d.ymax, 4.);
        delaunay_destroy(Some(d));
    }

    #[test]
    fn build_of_a_single_triangle_maps_points_and_finds_its_circle() {
        let points = [
            Point {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Point {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Point {
                x: 0.,
                y: 1.,
                z: 0.,
            },
        ];
        let mut d = delaunay_build(3, &points, 0, None, 0, None).unwrap();
        assert_eq!(d.triangles.len(), 1);
        let inside = Point {
            x: 0.25,
            y: 0.25,
            z: 0.,
        };
        assert_eq!(delaunay_xytoi(&d, &inside, -1), 0);
        let outside = Point {
            x: 5.,
            y: 5.,
            z: 0.,
        };
        assert_eq!(delaunay_xytoi(&d, &outside, -1), -1);

        let mut n = -1;
        let mut out: Vec<i32> = Vec::new();
        delaunay_circles_find(&mut d, &inside, &mut n, &mut out);
        assert_eq!(n, 1);
        assert_eq!(out, vec![0]);
        assert_eq!(N_FLAGS_TURNON, 1000);
        delaunay_destroy(Some(d));
    }
}
