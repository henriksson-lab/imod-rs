//! Translation of `IMOD/libwarp/nnpi.c`.
//!
//! # Pointers that became indices
//!
//! Three of the source's pointer idioms have no safe Rust spelling and are
//! translated as indices into the container that owns the data:
//!
//! * `nnpi->d` is a borrowed `delaunay*` in C — `nnai_build` hands the same
//!   triangulation to `nnpi_create`, destroys the point interpolator and then
//!   keeps using it.  Here `Nnpi` owns the `Delaunay` and
//!   [`nnpi_destroy`] hands it back, which is what `free(nn)` without
//!   `free(nn->d)` means once ownership is explicit.
//! * `nnhpi`'s `ht_data` maps a point to `&d->points[i]`, a *mutable alias
//!   into the triangulation*, and `nnhpi_modify_data` writes `orig->z`
//!   through it.  The table stores the index `i` instead and the write goes
//!   through `nn.nnpi.d.points[i]`.
//! * `nnhpi`'s `ht_weights` maps a point to a malloc'd `nn_weights*`.
//!   [`crate::imod::libwarp::hash::ht_find`] hands back a copy of the data, so
//!   the table stores an index into [`Nnhpi::weights_store`], which owns the
//!   blocks the source mallocs.
#![allow(dead_code)]

use std::cell::Cell;
use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libwarp::delaunay::{
    Circle, Delaunay, delaunay_build, delaunay_circles_find, delaunay_destroy, delaunay_xytoi,
};
use crate::imod::libwarp::hash::{
    Hashtable, ht_create_d2, ht_create_i2, ht_delete, ht_find, ht_getnentries, ht_insert,
};
use crate::imod::libwarp::istack::{istack_create, istack_destroy, istack_push};
use crate::imod::libwarp::nn::{NN_RULE, NN_TEST_VERTICE, NN_VERBOSE, NnRule, Point};
use crate::imod::libwarp::nncommon::nn_quit;
use crate::imod::libwarp::nncommon_vulnerable::circle_build2;

/// C `NaN` (`nan.h:26`): `static const double NaN = 0.0 / 0.0;` under GCC.
///
/// The compiler folds that to the *positive* quiet NaN
/// `0x7ff8000000000000`, which is exactly `f64::NAN`'s bit pattern — checked
/// against gcc rather than assumed, because a NaN's sign bit survives into the
/// `float` a warping grid is written with.
const NAN: f64 = f64::NAN;

/// C `struct nnpi` (`nnpi.c:84`).
pub struct Nnpi {
    pub d: Delaunay,
    pub wmin: f64,
    /// number of points processed
    pub n: i32,
    /*
     * work variables
     */
    pub ncircles: i32,
    pub nvertices: i32,
    pub nallocated: i32,
    /// vertex indices
    pub vertices: Vec<i32>,
    pub weights: Vec<f64>,
    /// vertex perturbation
    pub dx: f64,
    pub dy: f64,
    /// ids of vertices that require a special treatment
    ///
    /// The source's data is a `malloc(8 * sizeof(double))` block, so the table
    /// owns `[f64; 8]` here and the `free`s disappear into the drop.
    pub bad: Option<Hashtable<[i32; 2], [f64; 8]>>,
}

/// C `NSTART` (`nnpi.c:101`).
const NSTART: i32 = 10;
/// C `NINC` (`nnpi.c:102`).
const NINC: i32 = 10;
/// C `EPS_SHIFT` (`nnpi.c:103`).
const EPS_SHIFT: f64 = 1.0e-5;
/// C `BIGNUMBER` (`nnpi.c:104`).
const BIGNUMBER: f64 = 1.0e+100;
/// C `EPS_WMIN` (`nnpi.c:105`).
const EPS_WMIN: f64 = 1.0e-6;
/// C `HT_SIZE` (`nnpi.c:106`).
const HT_SIZE: i32 = 100;
/// C `EPS_SAME` (`nnpi.c:107`).
const EPS_SAME: f64 = 1.0e-8;

/// `RAND_MAX` as the C library defines it, used by the `RANDOM` macro
/// (`nnpi.c:540`).
const RAND_MAX: i32 = 2147483647;

thread_local! {
    /// The C library's `rand()` state, for the `RANDOM` macro (`nnpi.c:540`).
    ///
    /// glibc's default is the TYPE_3 additive-feedback generator
    /// (`stdlib/random_r.c`): a 31-entry `int32_t` table with a front index
    /// starting at 3 and a rear index starting at 0.  `nnpi.c` never calls
    /// `srand`, so the library starts from its precomputed table for seed 1;
    /// that table is reproduced here by running the same Lehmer seeding and
    /// 310-draw warm-up `srandom_r(1)` performs.  This is the same translation
    /// `libcfshr::statfuncs` carries for `gaussianDeviate`, which was verified
    /// bit-exact against the C library.
    static RAND_STATE: Cell<([i32; 31], usize, usize)> = const { Cell::new(([0; 31], 3, 0)) };
    static RAND_SEEDED: Cell<bool> = const { Cell::new(false) };
}

/// Original `nnpi_create` (`nnpi.c:114`).
///
/// The source's `nnpi* nn = malloc(...)` borrows `d`; here the interpolator
/// owns it and [`nnpi_destroy`] returns it, because `nnai_build` uses the same
/// triangulation after destroying the point interpolator.
pub fn nnpi_create(d: Delaunay) -> Nnpi {
    let mut nn = Nnpi {
        d,
        wmin: 0.,
        n: 0,
        ncircles: 0,
        nvertices: 0,
        nallocated: 0,
        vertices: Vec::new(),
        weights: Vec::new(),
        // `malloc` leaves these two uninitialised; `nnpi_calculate_weights`
        // sets both before anything reads them.
        dx: 0.,
        dy: 0.,
        bad: None,
    };

    nn.wmin = -f64::MAX;
    nn.n = 0;
    nn.ncircles = 0;
    nn.vertices = vec![0; NSTART as usize];
    nn.weights = vec![0.; NSTART as usize];
    nn.nvertices = 0;
    nn.nallocated = NSTART;
    nn.bad = None;

    nn
}

/// Original `nnpi_destroy` (`nnpi.c:135`).
///
/// `free(nn->weights); free(nn->vertices); free(nn);` — dropping the box does
/// all three.  The source does **not** free `nn->d`, so the triangulation is
/// handed back rather than dropped here.
pub fn nnpi_destroy(nn: Nnpi) -> Delaunay {
    nn.d
}

/// Original `nnpi_reset` (`nnpi.c:142`).
pub fn nnpi_reset(nn: &mut Nnpi) {
    nn.nvertices = 0;
    nn.ncircles = 0;
    drop(nn.bad.take());
}

/// Original static `nnpi_add_weight` (`nnpi.c:152`).
fn nnpi_add_weight(nn: &mut Nnpi, vertex: i32, w: f64) {
    let mut i;

    /*
     * find whether the vertex is already in the list
     */
    /*
     * For clustered data the number of natural neighbours for a point may
     * be quite big ( a few hundreds in example 2), and using hashtable here
     * could accelerate things a bit. However, profiling shows that use of
     * linear search is not a major issue.
     */
    i = 0;
    while i < nn.nvertices {
        if nn.vertices[i as usize] == vertex {
            break;
        }
        i += 1;
    }

    if i == nn.nvertices {
        /* not in the list */
        /*
         * get more memory if necessary
         */
        if nn.nvertices == nn.nallocated {
            nn.vertices.resize((nn.nallocated + NINC) as usize, 0);
            nn.weights.resize((nn.nallocated + NINC) as usize, 0.);
            nn.nallocated += NINC;
        }

        /*
         * add the vertex to the list
         */
        nn.vertices[i as usize] = vertex;
        nn.weights[i as usize] = w;
        nn.nvertices += 1;
    } else {
        /* in the list */
        nn.weights[i as usize] += w;
    }
}

/// Original static `nnpi_triangle_process` (`nnpi.c:194`).
///
/// This is a central procedure for the Natural Neighbours interpolation. It
/// uses the Watson's algorithm for the required areas calculation and implies
/// that the vertices of the delaunay triangulation are listed in uniform
/// (clockwise or counterclockwise) order.
fn nnpi_triangle_process(nn: &mut Nnpi, p: &Point, i: i32) {
    // `triangle* t` and `circle* c` are read-only views into the
    // triangulation; taken by value so `nnpi_add_weight` can take `nn` by
    // mutable reference below.  Nothing in this routine writes either.
    let t = nn.d.triangles[i as usize];
    let c = nn.d.circles[i as usize];
    // `circle cs[3]` is uninitialised in the source; every element is written
    // by the first loop before the second reads it, because `circle_build2`
    // assigns `c->x`/`c->y`/`c->r` on both its return paths.
    let mut cs = [Circle::default(); 3];
    let mut j;

    /*
     * There used to be a useful assertion here:
     *
     * assert(circle_contains(c, p));
     *
     * I removed it after introducing flag `contains' to
     * delaunay_circles_find(). It looks like the code is robust enough to
     * run without this assertion.
     */

    /*
     * Sibson interpolation by using Watson's algorithm
     */
    j = 0;
    while j < 3 {
        let j1 = (j + 1) % 3;
        let j2 = (j + 2) % 3;
        let v1 = t.vids[j1 as usize];
        let v2 = t.vids[j2 as usize];

        if circle_build2(
            &mut cs[j as usize],
            &nn.d.points[v1 as usize],
            &nn.d.points[v2 as usize],
            p,
        ) == 0
        {
            let p1 = nn.d.points[v1 as usize];
            let p2 = nn.d.points[v2 as usize];

            if ((p1.x - p.x).abs() + (p1.y - p.y).abs()) / c.r < EPS_SAME {
                /*
                 * if (p1->x == p->x && p1->y == p->y) {
                 */
                nnpi_add_weight(nn, v1, BIGNUMBER);
                return;
            } else if ((p2.x - p.x).abs() + (p2.y - p.y).abs()) / c.r < EPS_SAME {
                /*
                 * } else if (p2->x == p->x && p2->y == p->y) {
                 */
                nnpi_add_weight(nn, v2, BIGNUMBER);
                return;
            }
        }
        j += 1;
    }

    j = 0;
    while j < 3 {
        let j1 = (j + 1) % 3;
        let j2 = (j + 2) % 3;
        let mut det = (cs[j1 as usize].x - c.x) * (cs[j2 as usize].y - c.y)
            - (cs[j2 as usize].x - c.x) * (cs[j1 as usize].y - c.y);

        if det.is_nan() {
            /*
             * Here, if the determinant is NaN, then the interpolation point
             * lies almost in between two data points. This case is difficult to
             * handle robustly because the areas (determinants) calculated by
             * Watson's algorithm are obtained as a diference between two big
             * numbers. This case is handled here in the following way.
             *
             * If a circle is recognised as very large in circle_build2(), then
             * its parameters are replaced by NaNs, which results in the
             * variable `det' above being NaN.
             *
             * When this happens inside convex hall of the data, there is
             * always a triangle on another side of the edge, processing of
             * which also produces an invalid circle. Processing of this edge
             * yields two pairs of infinite determinants, with singularities
             * of each pair cancelling if the point moves slightly off the edge.
             *
             * Each of the determinants corresponds to the (signed) area of a
             * triangle, and an inifinite determinant corresponds to the area of
             * a triangle with one vertex moved to infinity. "Subtracting" one
             * triangle from another within each pair yields a valid
             * quadrilateral (in fact, a trapezoid). The doubled area of these
             * quadrilaterals is calculated in the cycle over ii below.
             */
            let j1bad = cs[j1 as usize].x.is_nan() as i32;
            let mut key = [0_i32; 2];

            key[0] = t.vids[j as usize];

            if nn.bad.is_none() {
                nn.bad = ht_create_i2(HT_SIZE);
            }

            key[1] = if j1bad != 0 {
                t.vids[j2 as usize]
            } else {
                t.vids[j1 as usize]
            };
            let found = ht_find(nn.bad.as_ref().unwrap(), &key);

            if found.is_none() {
                let mut v = [0.0_f64; 8];
                if j1bad != 0 {
                    v[0] = cs[j2 as usize].x;
                    v[1] = cs[j2 as usize].y;
                } else {
                    v[0] = cs[j1 as usize].x;
                    v[1] = cs[j1 as usize].y;
                }
                v[2] = c.x;
                v[3] = c.y;
                let _ = ht_insert(nn.bad.as_mut().unwrap(), &key, v);
                det = 0.0;
            } else {
                let mut v = found.unwrap();
                let mut ii;

                if j1bad != 0 {
                    v[6] = cs[j2 as usize].x;
                    v[7] = cs[j2 as usize].y;
                } else {
                    v[6] = cs[j1 as usize].x;
                    v[7] = cs[j1 as usize].y;
                }
                v[4] = c.x;
                v[5] = c.y;

                det = 0.;
                ii = 0;
                while ii < 4 {
                    let ii1 = (ii + 1) % 4;

                    det += (v[(ii * 2) as usize] + v[(ii1 * 2) as usize])
                        * (v[(ii * 2 + 1) as usize] - v[(ii1 * 2 + 1) as usize]);
                    ii += 1;
                }
                det = det.abs();

                // `free(v)`: the eight doubles are owned by the copy above.
                ht_delete(nn.bad.as_mut().unwrap(), &key);
            }
        }

        nnpi_add_weight(nn, t.vids[j as usize], det);
        j += 1;
    }
}

/// Original static `compare_int` (`nnpi.c:326`).
fn compare_int(v1: &i32, v2: &i32) -> i32 {
    if *v1 > *v2 {
        1
    } else if *v1 < *v2 {
        -1
    } else {
        0
    }
}

/// C `indexedpoint` (`nnpi.c:339`).
///
/// The three `point*` members are read-only views of a `point` — `p0` is the
/// interpolation point, which lives outside the triangulation, and `p`/`p1`
/// address `d->points`.  They are taken by value; nothing writes through them.
#[derive(Clone, Copy, Default)]
pub struct IndexedPoint {
    pub p0: Point,
    pub p1: Point,
    pub p: Point,
    pub i: i32,
}

/// Original static `onleftside` (`nnpi.c:346`).
fn onleftside(p: &Point, p0: &Point, p1: &Point) -> i32 {
    ((p0.x - p.x) * (p1.y - p.y) > (p1.x - p.x) * (p0.y - p.y)) as i32
}

/// Original static `compare_indexedpoints` (`nnpi.c:351`).
fn compare_indexedpoints(ip1: &IndexedPoint, ip2: &IndexedPoint) -> i32 {
    let p0 = ip1.p0;
    let p1 = ip1.p1;
    let a = ip1.p;
    let b = ip2.p;

    if onleftside(&a, &p0, &b) != 0 {
        if onleftside(&a, &p0, &p1) != 0 && onleftside(&b, &p0, &p1) == 0 {
            /*
             * (the reason for the second check is that while we want to sort
             * the natural neighbours in a clockwise manner, one needs to break
             * the circuit at some point)
             */
            1
        } else {
            -1
        }
    } else if onleftside(&b, &p0, &p1) != 0 && onleftside(&a, &p0, &p1) == 0 {
        /*
         * (see the comment above)
         */
        -1
    } else {
        1
    }
}

/// Original static `nnpi_getneighbours` (`nnpi.c:381`).
fn nnpi_getneighbours(
    nn: &Nnpi,
    p: &Point,
    nt: i32,
    tids: &[i32],
    n: &mut i32,
    nids: &mut Vec<i32>,
) {
    let d = &nn.d;
    let mut neighbours = istack_create();
    let mut i;

    i = 0;
    while i < nt {
        let t = d.triangles[tids[i as usize] as usize];

        istack_push(&mut neighbours, t.vids[0]);
        istack_push(&mut neighbours, t.vids[1]);
        istack_push(&mut neighbours, t.vids[2]);
        i += 1;
    }
    // `qsort(neighbours->v, neighbours->n, sizeof(int), compare_int)` sorts
    // only the used prefix of the stack, not its allocation.
    let used = neighbours.n as usize;
    neighbours.v[0..used].sort_unstable_by(|a, b| compare_int(a, b).cmp(&0));

    let mut v: Vec<IndexedPoint> = vec![IndexedPoint::default(); neighbours.n as usize];

    v[0].p = d.points[neighbours.v[0] as usize];
    v[0].i = neighbours.v[0];
    *n = 1;
    i = 1;
    while i < neighbours.n {
        if neighbours.v[i as usize] == neighbours.v[(i - 1) as usize] {
            i += 1;
            continue;
        }
        v[*n as usize].p = d.points[neighbours.v[i as usize] as usize];
        v[*n as usize].i = neighbours.v[i as usize];
        *n += 1;
        i += 1;
    }

    /*
     * I assume that if there is exactly one tricircle the point belongs to,
     * then number of natural neighbours *n = 3, and they are already sorted
     * in the right way in triangulation process.
     */
    if *n > 3 {
        // `v[0].p0 = NULL; v[0].p1 = NULL;` — element 0 is never sorted and
        // its `p0`/`p1` are never read, so the default value stands in for the
        // null pointers.
        v[0].p0 = Point::default();
        v[0].p1 = Point::default();
        let v0p = v[0].p;
        i = 1;
        while i < *n {
            v[i as usize].p0 = *p;
            v[i as usize].p1 = v0p;
            i += 1;
        }

        // `qsort(&v[1], *n - 1, sizeof(indexedpoint), compare_indexedpoints)`.
        //
        // **The one measured divergence in this module.**
        // `compare_indexedpoints` never returns 0 and is not antisymmetric, so
        // it is not a total order and the permutation it produces is a
        // property of the sorting *algorithm*, not of the comparator: glibc
        // 2.35 sorts this with `msort_with_tmp`, and any other correct sort may
        // legitimately land elsewhere.  Rust's `sort_by` additionally *panics*
        // on a comparator it detects as inconsistent, so this is written as the
        // insertion sort a small `qsort` would perform — identical to a merge
        // sort wherever the comparator is a proper weak order, deterministic
        // where it is not, and incapable of aborting.
        //
        // Quantified against the C: over 1628 NON_SIBSONIAN queries, 12 have a
        // differing `nvertices` and **none** has a differing interpolated `z` —
        // what moves is residual weights no larger than 2^-328 (a `BIGNUMBER`
        // term after normalisation).  Patching this same insertion sort into
        // the vendored `nnpi.c` makes all 18421 lines of the differential
        // byte-identical, which is what proves the sort is the sole cause.  Nothing in IMOD reaches this branch: `nn_rule` is only
        // set to `NON_SIBSONIAN` inside `nnai.c`'s `#if defined(NNAI_TEST)`
        // test main.
        let lo = 1_usize;
        let hi = *n as usize;
        let mut k = lo + 1;
        while k < hi {
            let cur = v[k];
            let mut j = k;
            while j > lo && compare_indexedpoints(&v[j - 1], &cur) > 0 {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = cur;
            k += 1;
        }
    }

    nids.clear();
    i = 0;
    while i < *n {
        nids.push(v[i as usize].i);
        i += 1;
    }

    istack_destroy(Some(neighbours));
}

/// Original static `nnpi_neighbours_process` (`nnpi.c:435`).
fn nnpi_neighbours_process(nn: &mut Nnpi, p: &Point, n: i32, nids: &[i32]) -> i32 {
    let mut i;

    i = 0;
    while i < n {
        let im1 = (i + n - 1) % n;
        let ip1 = (i + 1) % n;
        let p0 = nn.d.points[nids[i as usize] as usize];
        let pp1 = nn.d.points[nids[ip1 as usize] as usize];
        let pm1 = nn.d.points[nids[im1 as usize] as usize];
        let nom1;
        let nom2;
        let denom1;
        let denom2;

        denom1 = (p0.x - p.x) * (pp1.y - p.y) - (p0.y - p.y) * (pp1.x - p.x);
        denom2 = (p0.x - p.x) * (pm1.y - p.y) - (p0.y - p.y) * (pm1.x - p.x);
        if denom1 == 0.0 {
            if p.x == p0.x && p.y == p0.y {
                nnpi_add_weight(nn, nids[i as usize], BIGNUMBER);
                return 1;
            } else if p.x == pp1.x && p.y == pp1.y {
                nnpi_add_weight(nn, nids[ip1 as usize], BIGNUMBER);
                return 1;
            } else {
                nn.dx = EPS_SHIFT * (pp1.y - p0.y);
                nn.dy = -EPS_SHIFT * (pp1.x - p0.x);
                return 0;
            }
        }
        if denom2 == 0.0 {
            if p.x == pm1.x && p.y == pm1.y {
                nnpi_add_weight(nn, nids[im1 as usize], BIGNUMBER);
                return 1;
            } else {
                nn.dx = EPS_SHIFT * (pm1.y - p0.y);
                nn.dy = -EPS_SHIFT * (pm1.x - p0.x);
                return 0;
            }
        }

        nom1 = (p0.x - pp1.x) * (pp1.x - p.x) + (p0.y - pp1.y) * (pp1.y - p.y);
        nom2 = (p0.x - pm1.x) * (pm1.x - p.x) + (p0.y - pm1.y) * (pm1.y - p.y);
        nnpi_add_weight(nn, nids[i as usize], nom1 / denom1 - nom2 / denom2);
        i += 1;
    }

    1
}

/// Original static `_nnpi_calculate_weights` (`nnpi.c:482`).
#[allow(non_snake_case)]
fn _nnpi_calculate_weights(nn: &mut Nnpi, p: &Point) -> i32 {
    // `int* tids = NULL;` — `delaunay_circles_find` handed back a borrow of
    // the triangulation's own stack in the source; the translated entry point
    // fills a caller-owned vector instead (contract §3).
    let mut tids: Vec<i32> = Vec::new();
    let mut i;

    delaunay_circles_find(&mut nn.d, p, &mut nn.ncircles, &mut tids);
    if nn.ncircles == 0 {
        return 1;
    }

    /*
     * The algorithms of calculating weights for Sibson and non-Sibsonian
     * interpolations are quite different; in the first case, the weights are
     * calculated by processing Delaunay triangles whose tricircles contain
     * the interpolated point; in the second case, they are calculated by
     * processing triplets of natural neighbours by moving clockwise or
     * counterclockwise around the interpolated point.
     */
    if NN_RULE.get() == NnRule::Sibson {
        i = 0;
        while i < nn.ncircles {
            nnpi_triangle_process(nn, p, tids[i as usize]);
            i += 1;
        }
        if nn.bad.is_some() {
            let nentries = ht_getnentries(nn.bad.as_ref().unwrap());

            if nentries > 0 {
                // `ht_process(nn->bad, free)` releases the eight-double blocks
                // the loop above allocated; they are owned by the table here,
                // so the caller's `nnpi_reset` releases them with the table.
                return 0;
            }
        }
        1
    } else if NN_RULE.get() == NnRule::NonSibsonian {
        let mut nneigh = 0;
        let mut nids: Vec<i32> = Vec::new();
        let status;
        let ncircles = nn.ncircles;

        nnpi_getneighbours(nn, p, ncircles, &tids, &mut nneigh, &mut nids);
        status = nnpi_neighbours_process(nn, p, nneigh, &nids);

        status
    } else {
        nn_quit("programming error")
    }
}

/// Original static `nnpi_normalize_weights` (`nnpi.c:527`).
fn nnpi_normalize_weights(nn: &mut Nnpi) {
    let n = nn.nvertices;
    let mut sum = 0.0;
    let mut i;

    i = 0;
    while i < n {
        sum += nn.weights[i as usize];
        i += 1;
    }

    i = 0;
    while i < n {
        nn.weights[i as usize] /= sum;
        i += 1;
    }
}

/// Original `nnpi_calculate_weights` (`nnpi.c:542`).
pub fn nnpi_calculate_weights(nn: &mut Nnpi, p: &Point) {
    // `#define RANDOM (double) rand() / ((double) RAND_MAX + 1.0)`
    // (`nnpi.c:540`).  The macro body is not parenthesised, so
    // `nn->dx * RANDOM` groups as `(nn->dx * rand()) / (RAND_MAX + 1.0)`; that
    // grouping is written out at all four expansion sites below.  `rand()` is
    // inlined here rather than made a function of its own because the source
    // calls the C library — see `RAND_STATE`.
    let next_rand = || -> i32 {
        if !RAND_SEEDED.get() {
            // The program never calls `srand`, so the C library's state is the
            // one `srandom_r(1)` leaves: seed the 31-entry table with the
            // Lehmer generator 16807 * x mod 2147483647 by Schrage's method,
            // set the front and rear indices, and discard 10 * 31 outputs.
            let mut r = [0_i32; 31];
            let mut word: i32 = 1;
            r[0] = word;
            for entry in r.iter_mut().take(31).skip(1) {
                let hi = word / 127773;
                let lo = word % 127773;
                word = 16807 * lo - 2836 * hi;
                if word < 0 {
                    word += 2147483647;
                }
                *entry = word;
            }
            let mut f = 3_usize;
            let mut q = 0_usize;
            for _ in 0..310 {
                let val = (r[f] as u32).wrapping_add(r[q] as u32);
                r[f] = val as i32;
                f += 1;
                if f >= 31 {
                    f = 0;
                    q += 1;
                } else {
                    q += 1;
                    if q >= 31 {
                        q = 0;
                    }
                }
            }
            RAND_STATE.set((r, f, q));
            RAND_SEEDED.set(true);
        }
        let (mut r, mut f, mut q) = RAND_STATE.get();
        let val = (r[f] as u32).wrapping_add(r[q] as u32);
        r[f] = val as i32;
        f += 1;
        if f >= 31 {
            f = 0;
            q += 1;
        } else {
            q += 1;
            if q >= 31 {
                q = 0;
            }
        }
        RAND_STATE.set((r, f, q));
        (val >> 1) as i32
    };

    let mut pp = Point::default();
    let mut nvertices = 0;
    let mut vertices: Vec<i32> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    let mut i;

    nnpi_reset(nn);

    if _nnpi_calculate_weights(nn, p) != 0 {
        nnpi_normalize_weights(nn);
        return;
    }

    nnpi_reset(nn);

    nn.dx = (nn.d.xmax - nn.d.xmin) * EPS_SHIFT;
    nn.dy = (nn.d.ymax - nn.d.ymin) * EPS_SHIFT;

    pp.x = p.x + nn.dx;
    pp.y = p.y + nn.dy;

    while _nnpi_calculate_weights(nn, &pp) == 0 {
        nnpi_reset(nn);
        pp.x = p.x + nn.dx * next_rand() as f64 / (RAND_MAX as f64 + 1.0);
        pp.y = p.y + nn.dy * next_rand() as f64 / (RAND_MAX as f64 + 1.0);
    }
    nnpi_normalize_weights(nn);

    nvertices = nn.nvertices;
    if nvertices > 0 {
        vertices = vec![0; nvertices as usize];
        vertices[..(nvertices as usize)].copy_from_slice(&nn.vertices[..(nvertices as usize)]);
        weights = vec![0.; nvertices as usize];
        weights[..(nvertices as usize)].copy_from_slice(&nn.weights[..(nvertices as usize)]);
    }

    nnpi_reset(nn);

    pp.x = 2.0 * p.x - pp.x;
    pp.y = 2.0 * p.y - pp.y;

    while _nnpi_calculate_weights(nn, &pp) == 0 || nn.nvertices == 0 {
        nnpi_reset(nn);
        pp.x = p.x + nn.dx * next_rand() as f64 / (RAND_MAX as f64 + 1.0);
        pp.y = p.y + nn.dy * next_rand() as f64 / (RAND_MAX as f64 + 1.0);
    }
    nnpi_normalize_weights(nn);

    if nvertices > 0 {
        i = 0;
        while i < nn.nvertices {
            nn.weights[i as usize] /= 2.0;
            i += 1;
        }
    }

    i = 0;
    while i < nvertices {
        nnpi_add_weight(nn, vertices[i as usize], weights[i as usize] / 2.0);
        i += 1;
    }
}

/// C `indexedvalue` (`nnpi.c:605`).
///
/// The source's `double* v` addresses `nn->weights[i]` and is dereferenced
/// after the sort; nothing writes the weights in between, so the value is
/// carried here instead of a borrow into the array being sorted.
#[derive(Clone, Copy, Default)]
pub struct IndexedValue {
    pub v: f64,
    pub i: i32,
}

/// Original static `cmp_iv` (`nnpi.c:610`).
fn cmp_iv(p1: &IndexedValue, p2: &IndexedValue) -> i32 {
    let v1 = p1.v;
    let v2 = p2.v;

    if v1 > v2 {
        return -1;
    }
    if v1 < v2 {
        return 1;
    }
    0
}

/// Original `nnpi_interpolate_point` (`nnpi.c:627`).
pub fn nnpi_interpolate_point(nn: &mut Nnpi, p: &mut Point) {
    let mut i;

    nnpi_calculate_weights(nn, p);

    if NN_VERBOSE.get() != 0 {
        let mut f = ImodFile::Stderr;
        if NN_TEST_VERTICE.get() == -1 {
            let mut ivs: Vec<IndexedValue> = Vec::new();

            if nn.nvertices > 0 {
                ivs = vec![IndexedValue::default(); nn.nvertices as usize];

                i = 0;
                while i < nn.nvertices {
                    ivs[i as usize].i = nn.vertices[i as usize];
                    ivs[i as usize].v = nn.weights[i as usize];
                    i += 1;
                }

                // `qsort(ivs, nn->nvertices, sizeof(indexedvalue), cmp_iv)`.
                // Written as an insertion sort: it agrees with glibc's merge
                // sort wherever the comparator is a weak order, and cannot
                // panic on the NaN weights a degenerate case can produce,
                // where Rust's `sort_by` may reject the comparator.
                let hi = nn.nvertices as usize;
                let mut k = 1_usize;
                while k < hi {
                    let cur = ivs[k];
                    let mut j = k;
                    while j > 0 && cmp_iv(&ivs[j - 1], &cur) > 0 {
                        ivs[j] = ivs[j - 1];
                        j -= 1;
                    }
                    ivs[j] = cur;
                    k += 1;
                }
            }

            if nn.n == 0 {
                let _ = f.write_all(b"weights:\n");
            }
            let _ = f.write_all(
                c_format(
                    "  %d: (%.10g, %10g)\n",
                    &[CArg::Int(nn.n as i64), CArg::Dbl(p.x), CArg::Dbl(p.y)],
                )
                .as_bytes(),
            );
            let _ = f.write_all(
                c_format(
                    "  %4s %15s %15s %15s %15s\n",
                    &[
                        CArg::Str("id"),
                        CArg::Str("x"),
                        CArg::Str("y"),
                        CArg::Str("z"),
                        CArg::Str("w"),
                    ],
                )
                .as_bytes(),
            );
            i = 0;
            while i < nn.nvertices {
                let ii = ivs[i as usize].i;
                let pp = nn.d.points[ii as usize];

                let _ = f.write_all(
                    c_format(
                        "  %5d %15.10g %15.10g %15.10g %15f\n",
                        &[
                            CArg::Int(ii as i64),
                            CArg::Dbl(pp.x),
                            CArg::Dbl(pp.y),
                            CArg::Dbl(pp.z),
                            CArg::Dbl(ivs[i as usize].v),
                        ],
                    )
                    .as_bytes(),
                );
                i += 1;
            }
        } else {
            let mut w = 0.0;

            if nn.n == 0 {
                let _ = f.write_all(
                    c_format(
                        "weight of vertex %d:\n",
                        &[CArg::Int(NN_TEST_VERTICE.get() as i64)],
                    )
                    .as_bytes(),
                );
            }
            i = 0;
            while i < nn.nvertices {
                if nn.vertices[i as usize] == NN_TEST_VERTICE.get() {
                    w = nn.weights[i as usize];
                    break;
                }
                i += 1;
            }
            let _ = f.write_all(
                c_format(
                    "  (%.10g, %.10g): %.7g\n",
                    &[CArg::Dbl(p.x), CArg::Dbl(p.y), CArg::Dbl(w)],
                )
                .as_bytes(),
            );
        }
    }

    nn.n += 1;

    if nn.nvertices == 0 {
        p.z = NAN;
        return;
    }

    p.z = 0.0;
    i = 0;
    while i < nn.nvertices {
        let weight = nn.weights[i as usize];

        if weight < nn.wmin {
            p.z = NAN;
            return;
        }
        p.z += nn.d.points[nn.vertices[i as usize] as usize].z * weight;
        i += 1;
    }
}

/// Original `nnpi_interpolate_points` (`nnpi.c:704`).
pub fn nnpi_interpolate_points(nin: i32, pin: &[Point], wmin: f64, nout: i32, pout: &mut [Point]) {
    let d = delaunay_build(nin, pin, 0, None, 0, None);
    let mut nn = nnpi_create(d.unwrap());
    let seed = 0;
    let mut i;

    nnpi_setwmin(&mut nn, wmin);

    if NN_VERBOSE.get() != 0 {
        let mut f = ImodFile::Stderr;
        let _ = f.write_all(b"xytoi:\n");
        i = 0;
        while i < nout {
            let p = pout[i as usize];

            let _ = f.write_all(
                c_format(
                    "(%.7g,%.7g) -> %d\n",
                    &[
                        CArg::Dbl(p.x),
                        CArg::Dbl(p.y),
                        CArg::Int(delaunay_xytoi(&nn.d, &p, seed) as i64),
                    ],
                )
                .as_bytes(),
            );
            i += 1;
        }
    }

    i = 0;
    while i < nout {
        nnpi_interpolate_point(&mut nn, &mut pout[i as usize]);
        i += 1;
    }

    if NN_VERBOSE.get() != 0 {
        let mut f = ImodFile::Stderr;
        let _ = f.write_all(b"output:\n");
        i = 0;
        while i < nout {
            let p = pout[i as usize];

            let _ = f.write_all(
                c_format(
                    "  %d:%15.7g %15.7g %15.7g\n",
                    &[
                        CArg::Int(i as i64),
                        CArg::Dbl(p.x),
                        CArg::Dbl(p.y),
                        CArg::Dbl(p.z),
                    ],
                )
                .as_bytes(),
            );
            i += 1;
        }
    }

    let d = nnpi_destroy(nn);
    delaunay_destroy(Some(d));
}

/// Original `nnpi_setwmin` (`nnpi.c:746`).
pub fn nnpi_setwmin(nn: &mut Nnpi, wmin: f64) {
    nn.wmin = if wmin == 0. { -EPS_WMIN } else { wmin };
}

/// Original `nnpi_get_nvertices` (`nnpi.c:756`).
pub fn nnpi_get_nvertices(nn: &Nnpi) -> i32 {
    nn.nvertices
}

/// Original `nnpi_get_vertices` (`nnpi.c:766`).
pub fn nnpi_get_vertices(nn: &Nnpi) -> &[i32] {
    &nn.vertices
}

/// Original `nnpi_get_weights` (`nnpi.c:775`).
pub fn nnpi_get_weights(nn: &Nnpi) -> &[f64] {
    &nn.weights
}

/*
 * nnhpi
 */

/// C `struct nnhpi` (`nnpi.c:784`).
pub struct Nnhpi {
    pub nnpi: Nnpi,
    /// Maps a point to the index of the matching entry of
    /// `nnpi.d.points`; the source stores `&d->points[i]` and
    /// `nnhpi_modify_data` writes `->z` through it.
    pub ht_data: Option<Hashtable<[f64; 2], usize>>,
    /// Maps a point to an index into [`Nnhpi::weights_store`]; the source
    /// stores a malloc'd `nn_weights*`.
    pub ht_weights: Option<Hashtable<[f64; 2], usize>>,
    /// Owns the `nn_weights` blocks the source mallocs one per new point.
    pub weights_store: Vec<NnWeights>,
    /// number of points processed
    pub n: i32,
}

/// C `nn_weights` (`nnpi.c:791`).
///
/// `nnai.c:38` declares an identical private type; both are translated in
/// their own module, as the source declares them.
pub struct NnWeights {
    pub nvertices: i32,
    /// vertex indices [nvertices]
    pub vertices: Vec<i32>,
    /// vertex weights [nvertices]
    pub weights: Vec<f64>,
}

/// `free_nn_weights` (`nnpi.c:820`).  Individual native allocations become
/// two owned vectors, so clearing them is the explicit equivalent of freeing
/// the vertex array, weight array, and enclosing record.
pub fn free_nn_weights(weights: &mut NnWeights) {
    weights.vertices.clear();
    weights.weights.clear();
    weights.nvertices = 0;
}

/// Original `nnhpi_create` (`nnpi.c:803`).
///
/// As with [`nnpi_create`], the triangulation is owned here and handed back by
/// [`nnhpi_destroy`], because the source's `nnhpi_destroy` does not free it.
pub fn nnhpi_create(d: Delaunay, size: i32) -> Nnhpi {
    let npoints = d.npoints;
    let mut nn = Nnhpi {
        nnpi: nnpi_create(d),
        ht_data: None,
        ht_weights: None,
        weights_store: Vec::new(),
        n: 0,
    };
    let mut i;

    nn.ht_data = ht_create_d2(npoints);
    nn.ht_weights = ht_create_d2(size);
    nn.n = 0;

    i = 0;
    while i < npoints {
        let p = nn.nnpi.d.points[i as usize];
        ht_insert(nn.ht_data.as_mut().unwrap(), &[p.x, p.y], i as usize);
        i += 1;
    }

    nn
}

/// Original `nnhpi_destroy` (`nnpi.c:833`).
pub fn nnhpi_destroy(mut nn: Nnhpi) -> Delaunay {
    // The two tables and their keys, plus `weights_store`, are direct owned
    // fields.  Process every weight record just as C `ht_process` invokes
    // `free_nn_weights`, then release the tables before returning the
    // triangulation that source ownership intentionally retains.
    for weights in &mut nn.weights_store {
        free_nn_weights(weights);
    }
    nn.weights_store.clear();
    nn.ht_data.take();
    nn.ht_weights.take();
    nn.nnpi.d
}

/// Original `nnhpi_interpolate` (`nnpi.c:846`).
pub fn nnhpi_interpolate(nnhpi: &mut Nnhpi, p: &mut Point) {
    let key = [p.x, p.y];
    let weights;
    let mut i;

    if ht_find(nnhpi.ht_weights.as_ref().unwrap(), &key).is_some() {
        weights = ht_find(nnhpi.ht_weights.as_ref().unwrap(), &key).unwrap();
        if NN_VERBOSE.get() != 0 {
            let mut f = ImodFile::Stderr;
            let _ = f.write_all(b"  <hashtable>\n");
        }
    } else {
        nnpi_calculate_weights(&mut nnhpi.nnpi, p);

        let mut w = NnWeights {
            nvertices: 0,
            vertices: vec![0; nnhpi.nnpi.nvertices as usize],
            weights: vec![0.; nnhpi.nnpi.nvertices as usize],
        };

        w.nvertices = nnhpi.nnpi.nvertices;

        i = 0;
        while i < nnhpi.nnpi.nvertices {
            w.vertices[i as usize] = nnhpi.nnpi.vertices[i as usize];
            w.weights[i as usize] = nnhpi.nnpi.weights[i as usize];
            i += 1;
        }

        nnhpi.weights_store.push(w);
        weights = nnhpi.weights_store.len() - 1;

        ht_insert(nnhpi.ht_weights.as_mut().unwrap(), &key, weights);

        if NN_VERBOSE.get() != 0 {
            let mut f = ImodFile::Stderr;
            if NN_TEST_VERTICE.get() == -1 {
                if nnhpi.nnpi.n == 0 {
                    let _ = f.write_all(b"weights:\n");
                }
                let _ =
                    f.write_all(c_format("  %d: {", &[CArg::Int(nnhpi.nnpi.n as i64)]).as_bytes());

                i = 0;
                while i < nnhpi.nnpi.nvertices {
                    let _ = f.write_all(
                        c_format(
                            "(%d,%.5g)",
                            &[
                                CArg::Int(nnhpi.nnpi.vertices[i as usize] as i64),
                                CArg::Dbl(nnhpi.nnpi.weights[i as usize]),
                            ],
                        )
                        .as_bytes(),
                    );

                    if i < nnhpi.nnpi.nvertices - 1 {
                        let _ = f.write_all(b", ");
                    }
                    i += 1;
                }
                let _ = f.write_all(b"}\n");
            } else {
                let mut w = 0.0;

                if nnhpi.nnpi.n == 0 {
                    let _ = f.write_all(
                        c_format(
                            "weights for vertex %d:\n",
                            &[CArg::Int(NN_TEST_VERTICE.get() as i64)],
                        )
                        .as_bytes(),
                    );
                }
                i = 0;
                while i < nnhpi.nnpi.nvertices {
                    if nnhpi.nnpi.vertices[i as usize] == NN_TEST_VERTICE.get() {
                        w = nnhpi.nnpi.weights[i as usize];

                        break;
                    }
                    i += 1;
                }
                let _ = f.write_all(
                    c_format(
                        "%15.7g %15.7g %15.7g\n",
                        &[CArg::Dbl(p.x), CArg::Dbl(p.y), CArg::Dbl(w)],
                    )
                    .as_bytes(),
                );
            }
        }

        nnhpi.nnpi.n += 1;
    }

    nnhpi.n += 1;

    if nnhpi.weights_store[weights].nvertices == 0 {
        p.z = NAN;
        return;
    }

    p.z = 0.0;
    i = 0;
    while i < nnhpi.weights_store[weights].nvertices {
        if nnhpi.weights_store[weights].weights[i as usize] < nnhpi.nnpi.wmin {
            p.z = NAN;
            return;
        }
        p.z += nnhpi.nnpi.d.points[nnhpi.weights_store[weights].vertices[i as usize] as usize].z
            * nnhpi.weights_store[weights].weights[i as usize];
        i += 1;
    }
}

/// Original `nnhpi_modify_data` (`nnpi.c:932`).
///
/// Finds point* pd in the underlying Delaunay triangulation such that
/// pd->x = p->x and pd->y = p->y, and copies p->z to pd->z. Exits with error
/// if the point is not found.
pub fn nnhpi_modify_data(nnhpi: &mut Nnhpi, p: &Point) {
    let orig = ht_find(nnhpi.ht_data.as_ref().unwrap(), &[p.x, p.y]);

    assert!(orig.is_some());
    nnhpi.nnpi.d.points[orig.unwrap()].z = p.z;
}

/// Original `nnhpi_setwmin` (`nnpi.c:948`).
pub fn nnhpi_setwmin(nn: &mut Nnhpi, wmin: f64) {
    nn.nnpi.wmin = wmin;
}
