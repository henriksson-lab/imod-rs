//! Translation of `IMOD/libwarp/hull-ch.c` ("ch.c : numerical functions for
//! hull computation").
//!
//! `DEBUG` is `-7` (`hull.h:32`) and every `DEBS(qq)` block is
//! `if (DEBUG > qq)`, so only the `-8`, `-10` and `-20` levels are live here:
//! `sc`'s overshoot warning (`:188`) and `reduce_inner`'s failure report
//! (`:321`).  The `DEBS(-1)`, `DEBS(-2)`, `DEBS(0)` and `DEBS(-7)` blocks —
//! including both `check_perps` calls inside `get_normal_sede` — are compiled
//! out, and are noted where they sit rather than translated into dead code.

use std::cell::Cell;
use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libwarp::hull::{
    BASIS_SIZE, Basis, CDIM, CHECK_PERPS_BASIS, Coord, FG_SIZE, Fg, GET_SITE, HULL_INFINITY,
    HullStorage, MAXDIM, MAXPOINTS, Neighbor, P, POINT_SIZE, RDIM, S_B_BASIS, S_P_NEIGH_BASIS,
    SIMPLEX_SIZE, SITE_NUM, SITE_SIZE, Simplex, Site, TREE_SIZE, TT_BASIS, Tree, VisitFunc,
    buildhull, free_basis_storage, free_fg_storage, free_simplex_storage, free_tree_storage,
    new_block_basis, new_block_simplex, visit_hull, visit_triang_gen,
};
use crate::imod::libwarp::hull_fg::print_fg;
use crate::imod::libwarp::hull_io::{
    MAXS, MINS, panic, print_basis, print_neighbor_full, print_simplex_f,
};
use crate::imod::libwarp::pointops::{PDIM, print_point};

/// C `NEARZERO(d)` (`hull-ch.c:34`).
#[inline]
fn nearzero(d: f64) -> bool {
    d < f32::EPSILON as f64 && d > -(f32::EPSILON as f64)
}

thread_local! {
    /// C `short check_overshoot_f` (`hull-ch.c:31`).
    pub static CHECK_OVERSHOOT_F: Cell<i16> = const { Cell::new(0) };
    /// C `simplex *ch_root` (`hull-ch.c:33`).
    pub static CH_ROOT: Cell<usize> = const { Cell::new(0) };
    /// C `double Huge` (`hull-ch.c:44`).
    pub static HUGE: Cell<f64> = const { Cell::new(0.) };
    /// C `double HugeCrit` (`hull-ch.c:44`).
    pub static HUGE_CRIT: Cell<f64> = const { Cell::new(0.) };
    /// C `int basis_vec_size` (`hull-ch.c:52`).
    pub static BASIS_VEC_SIZE: Cell<usize> = const { Cell::new(0) };
    /// C `int exact_bits` (`hull-ch.c:139`).
    pub static EXACT_BITS: Cell<i32> = const { Cell::new(0) };
    /// C `float b_err_min` (`hull-ch.c:140`).
    pub static B_ERR_MIN: Cell<f32> = const { Cell::new(0.) };
    /// C `float b_err_min_sq` (`hull-ch.c:140`).
    pub static B_ERR_MIN_SQ: Cell<f32> = const { Cell::new(0.) };
    /// C `static short vd` (`hull-ch.c:145`).
    static VD: Cell<i16> = const { Cell::new(0) };
    /// C `static basis_s *infinity_basis` (`hull-ch.c:149`).
    static INFINITY_BASIS: Cell<usize> = const { Cell::new(0) };

    /// C `static neighbor sP_neigh = {0,0,0}` (`hull-ch.c:67`), vertex half.
    static S_P_NEIGH_VERT: Cell<Site> = const { Cell::new(0) };
    /// `sP_neigh.basis`: `0` until `out_of_flat` first "mallocs" it, after
    /// which it names the reserved arena slot.
    static S_P_NEIGH_B: Cell<usize> = const { Cell::new(0) };
    /// C `static basis_s *sB = NULL` (`hull-ch.c:68`).
    static S_B: Cell<usize> = const { Cell::new(0) };
    /// `check_perps`'s `static basis_s *b = NULL` (`hull-ch.c:404`).
    static CHECK_PERPS_B: Cell<usize> = const { Cell::new(0) };

    /// `sc`'s `static int lscale` (`hull-ch.c:178`).  A function static, so it
    /// is carried across calls; a Rust `let` would silently reset it.
    static SC_LSCALE: Cell<i32> = const { Cell::new(0) };
    /// `sc`'s `static double max_scale` (`hull-ch.c:179`).
    static SC_MAX_SCALE: Cell<f64> = const { Cell::new(0.) };
    /// `sc`'s `static double ldetbound` (`hull-ch.c:180`).
    static SC_LDETBOUND: Cell<f64> = const { Cell::new(0.) };
    /// `sc`'s `static double Sb` (`hull-ch.c:181`).
    static SC_SB: Cell<f64> = const { Cell::new(0.) };
    /// The `static int messcount` of `warning(-10, ...)` at `hull-ch.c:188`.
    static SC_MESSCOUNT: Cell<i32> = const { Cell::new(0) };
    /// The `static int messcount` of `check_overshoot` at `hull-ch.c:111`.
    static AXPY_MESSCOUNT: Cell<i32> = const { Cell::new(0) };
    /// The `static int messcount` of `check_overshoot` at `hull-ch.c:128`.
    static VSCALE_A_MESSCOUNT: Cell<i32> = const { Cell::new(0) };
    /// The `static int messcount` of `check_overshoot` at `hull-ch.c:132`.
    static VSCALE_X_MESSCOUNT: Cell<i32> = const { Cell::new(0) };
    /// `reduce_inner`'s `static int failcount` (`hull-ch.c:285`).
    static REDUCE_FAILCOUNT: Cell<i32> = const { Cell::new(0) };
    /// `alph_test`'s `static double alpha` (`hull-ch.c:567`).
    static ALPHA: Cell<f64> = const { Cell::new(0.) };
    /// `vols`'s `static simplex *s` (`hull-ch.c:679`).
    static VOLS_S: Cell<usize> = const { Cell::new(0) };

    /// C `int A[100], B[100], C[100], D[100]` runtime stat tables
    /// (`hull-ch.c:168`); only `B` is ever written and nothing reads any of
    /// them.
    ///
    /// **Deviation.** `reduce_inner` writes `B[j]` for `j` up to 249
    /// (`hull-ch.c:310`), so the source overruns `B` into `C` whenever the
    /// reduction takes more than a hundred passes.  `B` is sized to the loop
    /// bound here instead of reproducing an out-of-bounds write; nothing reads
    /// the tables, so the only difference is that `C` is no longer clobbered.
    pub static STAT_A: Cell<[i32; 100]> = const { Cell::new([0; 100]) };
    pub static STAT_B: Cell<[i32; 250]> = const { Cell::new([0; 250]) };
    pub static STAT_C: Cell<[i32; 100]> = const { Cell::new([0; 100]) };
    pub static STAT_D: Cell<[i32; 100]> = const { Cell::new([0; 100]) };
    /// C `int tot, totinf, bigt` (`hull-ch.c:169`).
    pub static TOT: Cell<i32> = const { Cell::new(0) };
    pub static TOTINF: Cell<i32> = const { Cell::new(0) };
    pub static BIGT: Cell<i32> = const { Cell::new(0) };

    /// C `short mi[MAXPOINTS]` (`hull-ch.c:614`).
    pub static MI: std::cell::RefCell<Vec<i16>> = std::cell::RefCell::new(vec![0; MAXPOINTS]);
    /// C `short mo[MAXPOINTS]` (`hull-ch.c:614`).
    pub static MO: std::cell::RefCell<Vec<i16>> = std::cell::RefCell::new(vec![0; MAXPOINTS]);
}

/// The C library's `logb` (ISO C99 7.12.6.11), which `hull-ch.c` calls at four
/// sites and Rust's `f64` has no equivalent for.
///
/// This is a libm boundary translation, not a helper: `logb(x)` extracts the
/// binary **exponent**, `floor(log2(|x|))` computed exactly, and is *not*
/// `log2`.  `logb(3.0)` is `1.0` where `3.0f64.log2()` is `1.585`, and `sc`
/// divides the result by two and compares it against an exact-arithmetic bit
/// budget, so substituting `log2` changes every scaling decision the reduction
/// makes.  `hull.h:143` even carries the source's own note, "on SGI machines:
/// returns floor of log base 2".
fn logb(x: f64) -> f64 {
    let a = x.abs();
    if a == 0. {
        return f64::NEG_INFINITY;
    }
    if !a.is_finite() {
        /* logb(+-inf) is +inf; logb(NaN) is NaN. */
        return a;
    }
    let biased = ((a.to_bits() >> 52) & 0x7ff) as i32;
    if biased == 0 {
        /* Subnormal: scale by 2^54 and take the true unbiased exponent. */
        let scaled = a * 18_014_398_509_481_984.0;
        (((scaled.to_bits() >> 52) & 0x7ff) as i32 - 1023 - 54) as f64
    } else {
        (biased - 1023) as f64
    }
}

/// Original `hullchCleanup` (`hull-ch.c:70`).
///
/// The source `free`s the two bases `out_of_flat` and `sees` allocated and
/// re-`NULL`s the pointers; the arena slots are reserved rather than
/// allocated, so this resets their contents and the two indices instead.
pub fn hull_ch_cleanup(st: &mut HullStorage) {
    st.basis[S_P_NEIGH_BASIS] = Basis::default();
    S_P_NEIGH_B.set(0);
    st.basis[S_B_BASIS] = Basis::default();
    S_B.set(0);
}

/// Original static `Vec_dot` (`hull-ch.c:79`).
fn vec_dot(x: &[Coord], y: &[Coord]) -> Coord {
    let mut sum: Coord = 0.;
    for i in 0..RDIM.get() as usize {
        sum += x[i] * y[i];
    }
    sum
}

/// Original static `Vec_dot_pdim` (`hull-ch.c:86`).
fn vec_dot_pdim(x: &[Coord], y: &[Coord]) -> Coord {
    let mut sum: Coord = 0.;
    for i in 0..PDIM.get() as usize {
        sum += x[i] * y[i];
    }
    /* check_overshoot(sum); */
    sum
}

/// Original static `Norm2` (`hull-ch.c:94`).
fn norm2(x: &[Coord]) -> Coord {
    let mut sum: Coord = 0.;
    for i in 0..RDIM.get() as usize {
        sum += x[i] * x[i];
    }
    sum
}

/// Original static `Ax_plus_y` (`hull-ch.c:101`).
fn ax_plus_y(a: Coord, x: &[Coord], y: &mut [Coord]) {
    for i in 0..RDIM.get() as usize {
        y[i] += a * x[i];
    }
}

/// Original static `Ax_plus_y_test` (`hull-ch.c:108`).
fn ax_plus_y_test(a: Coord, x: &[Coord], y: &mut [Coord]) {
    for i in 0..RDIM.get() as usize {
        /* check_overshoot(*y + a * *x) (`hull-ch.c:111`) */
        if CHECK_OVERSHOOT_F.get() != 0 && y[i] + a * x[i] > 9e15 {
            AXPY_MESSCOUNT.set(AXPY_MESSCOUNT.get() + 1);
            let count = AXPY_MESSCOUNT.get();
            let mut d = ImodFile::Stderr;
            if count <= 10 {
                let _ = d.write_all(b"overshot exact arithmetic\n");
                let _ = d.flush();
                let _ = d.write_all(c_format("hull-ch.c line %d \n", &[CArg::Int(111)]).as_bytes());
                let _ = d.flush();
            }
            if count == 10 {
                let _ = d.write_all(b"consider yourself warned\n");
                let _ = d.flush();
            }
        }
        y[i] += a * x[i];
    }
}

/// Original static `Vec_scale_test` (`hull-ch.c:123`).
fn vec_scale_test(n: i32, a: Coord, x: &mut [Coord]) {
    /* check_overshoot(a) (`hull-ch.c:128`) */
    if CHECK_OVERSHOOT_F.get() != 0 && a > 9e15 {
        VSCALE_A_MESSCOUNT.set(VSCALE_A_MESSCOUNT.get() + 1);
        let count = VSCALE_A_MESSCOUNT.get();
        let mut d = ImodFile::Stderr;
        if count <= 10 {
            let _ = d.write_all(b"overshot exact arithmetic\n");
            let _ = d.flush();
            let _ = d.write_all(c_format("hull-ch.c line %d \n", &[CArg::Int(128)]).as_bytes());
            let _ = d.flush();
        }
        if count == 10 {
            let _ = d.write_all(b"consider yourself warned\n");
            let _ = d.flush();
        }
    }

    for i in 0..n as usize {
        x[i] *= a;
        /* check_overshoot(*xx) (`hull-ch.c:132`) */
        if CHECK_OVERSHOOT_F.get() != 0 && x[i] > 9e15 {
            VSCALE_X_MESSCOUNT.set(VSCALE_X_MESSCOUNT.get() + 1);
            let count = VSCALE_X_MESSCOUNT.get();
            let mut d = ImodFile::Stderr;
            if count <= 10 {
                let _ = d.write_all(b"overshot exact arithmetic\n");
                let _ = d.flush();
                let _ = d.write_all(c_format("hull-ch.c line %d \n", &[CArg::Int(132)]).as_bytes());
                let _ = d.flush();
            }
            if count == 10 {
                let _ = d.write_all(b"consider yourself warned\n");
                let _ = d.flush();
            }
        }
    }
}

/// Original `print_site` (`hull-ch.c:158`).
pub fn print_site(st: &HullStorage, p: Site, f: &mut dyn Write) {
    let pdim = PDIM.get();
    print_point(
        f,
        pdim,
        if p == 0 {
            None
        } else {
            Some(&st.sites[p..p + pdim as usize])
        },
    );
    let _ = f.write_all(b"\n");
}

/// Original static `sc` (`hull-ch.c:175`) — amount by which to scale up
/// vector, for `reduce_inner`.
fn sc(st: &mut HullStorage, v: usize, s: usize, k: i32, j: i32) -> f64 {
    if j < 10 {
        let labound = logb(st.basis[v].sqa) / 2.;
        /* max_scale = exact_bits - labound - 0.66*(k-2) - 1 - DELIFT, DELIFT 0 */
        SC_MAX_SCALE.set(EXACT_BITS.get() as f64 - labound - 0.66 * (k - 2) as f64 - 1.);
        if SC_MAX_SCALE.get() < 1. {
            /* warning(-10, overshot exact arithmetic) (`hull-ch.c:188`) */
            SC_MESSCOUNT.set(SC_MESSCOUNT.get() + 1);
            let count = SC_MESSCOUNT.get();
            let mut d = ImodFile::Stderr;
            if count <= 10 {
                let _ = d.write_all(b"overshot exact arithmetic\n");
                let _ = d.flush();
                let _ = d.write_all(c_format("hull-ch.c line %d \n", &[CArg::Int(188)]).as_bytes());
                let _ = d.flush();
            }
            if count == 10 {
                let _ = d.write_all(b"consider yourself warned\n");
                let _ = d.flush();
            }
            SC_MAX_SCALE.set(1.);
        }

        if j == 0 {
            /* ldetbound = DELIFT */
            SC_LDETBOUND.set(0.);
            SC_SB.set(0.);
            for i in (1..k).rev() {
                let snib = st.simplex[s].neigh[i as usize].basis;
                SC_SB.set(SC_SB.get() + st.basis[snib].sqb);
                SC_LDETBOUND.set(SC_LDETBOUND.get() + logb(st.basis[snib].sqb) / 2. + 1.);
                SC_LDETBOUND.set(SC_LDETBOUND.get() - st.basis[snib].lscale as f64);
            }
        }
    }
    if st.basis[v].sqb <= 0.
        || SC_LDETBOUND.get() - st.basis[v].lscale as f64 + logb(st.basis[v].sqb) / 2. + 1. < 0.
    {
        /* DEBS(-2) ... EDEBS -- DEBUG (-7) > -2 is false. */
        0.
    } else {
        /* lscale = (int)floor(logb(2*Sb/(v->sqb + v->sqa*b_err_min)))/2 --
        the `/2` is C integer division on the already-truncated `int`. */
        let lscale =
            logb(2. * SC_SB.get() / (st.basis[v].sqb + st.basis[v].sqa * B_ERR_MIN.get() as f64))
                .floor() as i32
                / 2;
        SC_LSCALE.set(lscale);
        if SC_LSCALE.get() as f64 > SC_MAX_SCALE.get() {
            SC_LSCALE.set(SC_MAX_SCALE.get().floor() as i32);
        } else if SC_LSCALE.get() < 0 {
            SC_LSCALE.set(0);
        }
        st.basis[v].lscale += SC_LSCALE.get();
        /* two_to(lscale) (`hull-ch.c:173`) */
        let x = SC_LSCALE.get();
        if x < 20 {
            (1i32 << x) as f64
        } else {
            2f64.powi(x)
        }
    }
}

/// Original static `reduce_inner` (`hull-ch.c:269`).
fn reduce_inner(st: &mut HullStorage, v: usize, s: usize, k: i32) -> i32 {
    let rdim = RDIM.get() as usize;

    /* lower_terms(v); */
    let sq = norm2(&st.basis[v].vecs[..]);
    st.basis[v].sqa = sq;
    st.basis[v].sqb = sq;
    if k <= 1 {
        /* memcpy(vb, va, basis_vec_size): VA is vecs+rdim, VB is vecs, and the
        two halves are disjoint. */
        st.basis[v].vecs.copy_within(rdim..2 * rdim, 0);
        return 1;
    }
    /* The `if (vd) {...}` block at `hull-ch.c:281` is commented out. */

    for j in 0..250 {
        st.basis[v].vecs.copy_within(rdim..2 * rdim, 0);
        for i in (1..k).rev() {
            let snibv = st.simplex[s].neigh[i as usize].basis;
            /* `snibv` and `v` are distinct: `reduce`'s `v` is `neigh[k].basis`
            or a simplex normal, and this loop runs 1..k-1.  A whole-`Basis`
            copy is how the two disjoint borrows are expressed. */
            let snib = st.basis[snibv];
            let dd = -vec_dot(&snib.vecs[..], &st.basis[v].vecs[..]) / snib.sqb;
            ax_plus_y(dd, &snib.vecs[rdim..], &mut st.basis[v].vecs[..]);
        }
        let sqb = norm2(&st.basis[v].vecs[..]);
        st.basis[v].sqb = sqb;
        let sqa = norm2(&st.basis[v].vecs[rdim..]);
        st.basis[v].sqa = sqa;

        if 2. * st.basis[v].sqb >= st.basis[v].sqa {
            STAT_B.with(|b| {
                let mut t = b.get();
                t[j as usize] += 1;
                b.set(t);
            });
            return 1;
        }

        let scale = sc(st, v, s, k, j);
        vec_scale_test(RDIM.get(), scale, &mut st.basis[v].vecs[rdim..]);

        for i in (1..k).rev() {
            let snibv = st.simplex[s].neigh[i as usize].basis;
            let snib = st.basis[snibv];
            let mut dd = -vec_dot(&snib.vecs[..], &st.basis[v].vecs[rdim..]) / snib.sqb;
            dd = (dd + 0.5).floor();
            ax_plus_y_test(dd, &snib.vecs[rdim..], &mut st.basis[v].vecs[rdim..]);
        }
    }
    let failcount = REDUCE_FAILCOUNT.get();
    REDUCE_FAILCOUNT.set(failcount + 1);
    if failcount < 10 {
        /* DEB(-8, reduce_inner failed on:) DEBTR(-8) -- both live. */
        let mut d = ImodFile::Stderr;
        let _ = d.write_all(b"reduce_inner failed on:\n");
        let _ = d.flush();
        let _ = d.write_all(c_format("hull-ch.c line %d \n", &[CArg::Int(322)]).as_bytes());
        let _ = d.flush();
        print_basis(st, &mut d, v);
        print_simplex_f(st, s, &mut d, Some(print_neighbor_full));
    }
    0
}

/// Original static `reduce` (`hull-ch.c:335`).
///
/// The source takes `basis_s **v` so it can install a freshly allocated basis
/// in the caller's field; an arena index cannot be borrowed out of the storage
/// that is also passed in, so the (possibly new) index comes back with the
/// result and the caller stores it where the source's pointer pointed.
fn reduce(st: &mut HullStorage, mut v: usize, p: Site, s: usize, k: i32) -> (usize, i32) {
    let rdim = RDIM.get() as usize;
    let pdim = PDIM.get() as usize;
    let tt = st.simplex[s].neigh[0].vert;

    if v == 0 {
        /* NEWLRC(basis_s, (*v)) */
        if st.basis_list == 0 {
            new_block_basis(st, 1);
        }
        v = st.basis_list;
        assert!(v != 0);
        st.basis_list = st.basis[v].next;
        st.basis[v].ref_count = 1;
    } else {
        st.basis[v].lscale = 0;
    }

    if VD.get() != 0 {
        if p == HULL_INFINITY {
            /* memcpy(*v, infinity_basis, basis_s_size) */
            st.basis[v] = st.basis[INFINITY_BASIS.get()];
        } else {
            /* trans(z, p, tt) then lift(z, s) -- z is VB(*v). */
            for itrn in 0..pdim {
                let d = st.sites[p + itrn] - st.sites[tt + itrn];
                st.basis[v].vecs[itrn + rdim] = d;
                st.basis[v].vecs[itrn] = d;
            }
            /* lift: ldexp(x, -DELIFT) with DELIFT 0 is x. */
            let lifted = vec_dot_pdim(&st.basis[v].vecs[..], &st.basis[v].vecs[..]);
            st.basis[v].vecs[2 * rdim - 1] = lifted;
            st.basis[v].vecs[rdim - 1] = lifted;
        }
    } else {
        for itrn in 0..pdim {
            let d = st.sites[p + itrn] - st.sites[tt + itrn];
            st.basis[v].vecs[itrn + rdim] = d;
            st.basis[v].vecs[itrn] = d;
        }
    }
    let r = reduce_inner(st, v, s, k);
    (v, r)
}

/// `NULLIFY(basis_s, v)` (`stormacs.h:104`) applied to an arena slot: the
/// `dec_ref` half, returning the index the caller then clears.
pub(crate) fn nullify_basis(st: &mut HullStorage, b: usize) {
    if b != 0 {
        st.basis[b].ref_count -= 1;
        if st.basis[b].ref_count == 0 {
            /* FREEL(basis_s, v): memset then push onto the free list. */
            let next = st.basis_list;
            st.basis[b] = Basis::default();
            st.basis[b].next = next;
            st.basis_list = b;
        }
    }
}

/// Original `get_basis_sede` (`hull-ch.c:350`).
pub fn get_basis_sede(st: &mut HullStorage, s: usize) {
    let mut k = 1;
    let cdim = CDIM.get();

    if VD.get() != 0 && st.simplex[s].neigh[0].vert == HULL_INFINITY && cdim > 1 {
        /* SWAP(neighbor, *sn0, *sn) */
        let t = st.simplex[s].neigh[0];
        st.simplex[s].neigh[0] = st.simplex[s].neigh[1];
        st.simplex[s].neigh[1] = t;
        let b = st.simplex[s].neigh[0].basis;
        nullify_basis(st, b);
        st.simplex[s].neigh[0].basis = 0;
        st.simplex[s].neigh[0].basis = TT_BASIS;
        st.basis[TT_BASIS].ref_count += 1;
    } else if st.simplex[s].neigh[0].basis == 0 {
        st.simplex[s].neigh[0].basis = TT_BASIS;
        st.basis[TT_BASIS].ref_count += 1;
    } else {
        while k < cdim && st.simplex[s].neigh[k as usize].basis != 0 {
            k += 1;
        }
    }
    while k < cdim {
        let b = st.simplex[s].neigh[k as usize].basis;
        nullify_basis(st, b);
        st.simplex[s].neigh[k as usize].basis = 0;
        let vert = st.simplex[s].neigh[k as usize].vert;
        let (nb, _) = reduce(st, 0, vert, s, k);
        st.simplex[s].neigh[k as usize].basis = nb;
        k += 1;
    }
}

/// Original `out_of_flat` (`hull-ch.c:375`).
pub fn out_of_flat(st: &mut HullStorage, root: usize, p: Site) -> i32 {
    if S_P_NEIGH_B.get() == 0 {
        /* sP_neigh.basis = malloc(basis_s_size): a reserved slot, not a
        pool object, exactly as in the source. */
        S_P_NEIGH_B.set(S_P_NEIGH_BASIS);
    }

    S_P_NEIGH_VERT.set(p);
    CDIM.set(CDIM.get() + 1);
    let cdim = CDIM.get();
    st.simplex[root].neigh[(cdim - 1) as usize].vert = st.simplex[root].peak.vert;
    let b = st.simplex[root].neigh[(cdim - 1) as usize].basis;
    nullify_basis(st, b);
    st.simplex[root].neigh[(cdim - 1) as usize].basis = 0;
    get_basis_sede(st, root);
    if VD.get() != 0 && st.simplex[root].neigh[0].vert == HULL_INFINITY {
        return 1;
    }
    let (nb, _) = reduce(st, S_P_NEIGH_B.get(), p, root, CDIM.get());
    S_P_NEIGH_B.set(nb);
    if st.basis[S_P_NEIGH_B.get()].sqa != 0. {
        return 1;
    }
    CDIM.set(CDIM.get() - 1);
    0
}

/// Original static `cosangle_sq` (`hull-ch.c:393`).
fn cosangle_sq(vv: &[Coord], wv: &[Coord]) -> f64 {
    let dd = vec_dot(vv, wv);
    dd * dd / norm2(vv) / norm2(wv)
}

/// Original `check_perps` (`hull-ch.c:402`).
pub fn check_perps(st: &mut HullStorage, s: usize) -> i32 {
    let cdim = CDIM.get();
    let rdim = RDIM.get() as usize;
    let pdim = PDIM.get() as usize;

    for i in 1..cdim {
        let b = st.simplex[s].neigh[i as usize].basis;
        if nearzero(st.basis[b].sqb) {
            return 0;
        }
    }
    let bb = if CHECK_PERPS_B.get() == 0 {
        CHECK_PERPS_B.set(CHECK_PERPS_BASIS);
        CHECK_PERPS_BASIS
    } else {
        st.basis[CHECK_PERPS_B.get()].lscale = 0;
        CHECK_PERPS_B.get()
    };
    let tt = st.simplex[s].neigh[0].vert;
    for i in 1..cdim {
        let y = st.simplex[s].neigh[i as usize].vert;
        if VD.get() != 0 && y == HULL_INFINITY {
            st.basis[bb] = st.basis[INFINITY_BASIS.get()];
        } else {
            for itrn in 0..pdim {
                let d = st.sites[y + itrn] - st.sites[tt + itrn];
                st.basis[bb].vecs[itrn + rdim] = d;
                st.basis[bb].vecs[itrn] = d;
            }
            if VD.get() != 0 {
                let lifted = vec_dot_pdim(&st.basis[bb].vecs[..], &st.basis[bb].vecs[..]);
                st.basis[bb].vecs[2 * rdim - 1] = lifted;
                st.basis[bb].vecs[rdim - 1] = lifted;
            }
        }
        if st.simplex[s].normal != 0 {
            let normal = st.simplex[s].normal;
            let nb = st.basis[normal];
            if cosangle_sq(&st.basis[bb].vecs[..], &nb.vecs[..]) > B_ERR_MIN_SQ.get() as f64 {
                /* DEBS(0) ... EDEBS -- dead; only the return survives. */
                return 0;
            }
        }
        for j in i + 1..cdim {
            let ob = st.simplex[s].neigh[j as usize].basis;
            let obb = st.basis[ob];
            if cosangle_sq(&st.basis[bb].vecs[..], &obb.vecs[..]) > B_ERR_MIN_SQ.get() as f64 {
                /* DEBS(0) ... EDEBS -- dead. */
                return 0;
            }
        }
    }
    1
}

/// Original `get_normal_sede` (`hull-ch.c:440`).
pub fn get_normal_sede(st: &mut HullStorage, s: usize) {
    get_basis_sede(st, s);
    let cdim = CDIM.get();
    if RDIM.get() == 3 && cdim == 3 {
        let ab = st.simplex[s].neigh[1].basis;
        let bb = st.simplex[s].neigh[2].basis;
        let a = st.basis[ab].vecs;
        let b = st.basis[bb].vecs;
        /* NEWLRC(basis_s, s->normal) */
        if st.basis_list == 0 {
            new_block_basis(st, 1);
        }
        let normal = st.basis_list;
        assert!(normal != 0);
        st.basis_list = st.basis[normal].next;
        st.basis[normal].ref_count = 1;
        st.simplex[s].normal = normal;
        st.basis[normal].vecs[0] = a[1] * b[2] - a[2] * b[1];
        st.basis[normal].vecs[1] = a[2] * b[0] - a[0] * b[2];
        st.basis[normal].vecs[2] = a[0] * b[1] - a[1] * b[0];
        let sqb = norm2(&st.basis[normal].vecs[..]);
        st.basis[normal].sqb = sqb;
        let ch_root = CH_ROOT.get();
        /* for (i=cdim+1, rn = ch_root->neigh+cdim-1; i; i--, rn--) walks down
        to `neigh[-1]`, which is `peak`. */
        for ri in (-1..cdim).rev() {
            let rn_vert = if ri < 0 {
                st.simplex[ch_root].peak.vert
            } else {
                st.simplex[ch_root].neigh[ri as usize].vert
            };
            let mut j = 0;
            while j < cdim && rn_vert != st.simplex[s].neigh[j as usize].vert {
                j += 1;
            }
            if j < cdim {
                continue;
            }
            if rn_vert == HULL_INFINITY {
                if st.basis[normal].vecs[2] > -(B_ERR_MIN.get() as f64) {
                    continue;
                }
            } else if sees(st, rn_vert, s) == 0 {
                continue;
            }
            st.basis[normal].vecs[0] = -st.basis[normal].vecs[0];
            st.basis[normal].vecs[1] = -st.basis[normal].vecs[1];
            st.basis[normal].vecs[2] = -st.basis[normal].vecs[2];
            break;
        }
        /* DEBS(-1) if (!check_perps(s)) exit(1); EDEBS -- dead. */
        return;
    }

    let ch_root = CH_ROOT.get();
    for ri in (-1..cdim).rev() {
        let rn_vert = if ri < 0 {
            st.simplex[ch_root].peak.vert
        } else {
            st.simplex[ch_root].neigh[ri as usize].vert
        };
        let mut j = 0;
        while j < cdim && rn_vert != st.simplex[s].neigh[j as usize].vert {
            j += 1;
        }
        if j < cdim {
            continue;
        }
        let normal = st.simplex[s].normal;
        let (nb, _) = reduce(st, normal, rn_vert, s, cdim);
        st.simplex[s].normal = nb;
        if st.basis[nb].sqb != 0. {
            break;
        }
    }
    /* DEBS(-1) if (!check_perps(s)) {DEBTR(-1) exit(1);} EDEBS -- dead. */
}

/// Original `get_normal` (`hull-ch.c:481`).
pub fn get_normal(st: &mut HullStorage, s: usize) {
    get_normal_sede(st, s);
}

/// Original `sees` (`hull-ch.c:483`).
pub fn sees(st: &mut HullStorage, p: Site, s: usize) -> i32 {
    let rdim = RDIM.get() as usize;
    let pdim = PDIM.get() as usize;

    let sb = if S_B.get() == 0 {
        /* sB = malloc(basis_s_size): the source leaves it uninitialised and
        does not set `lscale` on this first call.  The reserved slot starts
        zeroed, which is the value a fresh `malloc` gives in practice. */
        S_B.set(S_B_BASIS);
        S_B_BASIS
    } else {
        st.basis[S_B.get()].lscale = 0;
        S_B.get()
    };
    if CDIM.get() == 0 {
        return 0;
    }
    if st.simplex[s].normal == 0 {
        get_normal_sede(st, s);
        for i in 0..CDIM.get() {
            let b = st.simplex[s].neigh[i as usize].basis;
            nullify_basis(st, b);
            st.simplex[s].neigh[i as usize].basis = 0;
        }
    }
    let tt = st.simplex[s].neigh[0].vert;
    if VD.get() != 0 {
        if p == HULL_INFINITY {
            st.basis[sb] = st.basis[INFINITY_BASIS.get()];
        } else {
            for itrn in 0..pdim {
                let d = st.sites[p + itrn] - st.sites[tt + itrn];
                st.basis[sb].vecs[itrn + rdim] = d;
                st.basis[sb].vecs[itrn] = d;
            }
            let lifted = vec_dot_pdim(&st.basis[sb].vecs[..], &st.basis[sb].vecs[..]);
            st.basis[sb].vecs[2 * rdim - 1] = lifted;
            st.basis[sb].vecs[rdim - 1] = lifted;
        }
    } else {
        for itrn in 0..pdim {
            let d = st.sites[p + itrn] - st.sites[tt + itrn];
            st.basis[sb].vecs[itrn + rdim] = d;
            st.basis[sb].vecs[itrn] = d;
        }
    }
    for _i in 0..3 {
        let normal = st.simplex[s].normal;
        let nb = st.basis[normal];
        let dd = vec_dot(&st.basis[sb].vecs[..], &nb.vecs[..]);
        if dd == 0.0 {
            /* DEBS(-7) ... EDEBS -- DEBUG (-7) > -7 is false. */
            return 0;
        }
        let dds = dd * dd / nb.sqb / norm2(&st.basis[sb].vecs[..]);
        if dds > B_ERR_MIN_SQ.get() as f64 {
            return (dd < 0.) as i32;
        }
        get_basis_sede(st, s);
        let cdim = CDIM.get();
        reduce_inner(st, sb, s, cdim);
    }
    /* DEBS(-7) if (i==3) {...} EDEBS -- dead. */
    0
}

/// Original static `radsq` (`hull-ch.c:527`) — square of ratio of
/// circumcircle radius to max edge length for Delaunay tetrahedra.
fn radsq(st: &mut HullStorage, s: usize) -> f64 {
    let cdim = CDIM.get();
    for i in 0..cdim {
        if st.simplex[s].neigh[i as usize].vert == HULL_INFINITY {
            return HUGE.get();
        }
    }

    if st.simplex[s].normal == 0 {
        get_normal_sede(st, s);
    }

    /* compute circumradius */
    let rdim = RDIM.get() as usize;
    let n = st.basis[st.simplex[s].normal].vecs;

    if nearzero(n[rdim - 1]) {
        return HUGE.get();
    }

    vec_dot_pdim(&n[..], &n[..]) / 4. / n[rdim - 1] / n[rdim - 1]
}

/// Original static `zero_marks` (`hull-ch.c:548`).
fn zero_marks(st: &mut HullStorage, s: usize) -> usize {
    st.simplex[s].mark = 0;
    0
}

/// Original static `one_marks` (`hull-ch.c:550`).
fn one_marks(st: &mut HullStorage, s: usize) -> usize {
    st.simplex[s].mark = 1;
    0
}

/// Original `alph_test` (`hull-ch.c:560`) — returns 1 if not an alpha-facet.
pub fn alph_test(st: &mut HullStorage, s: usize, i: i32, alphap: Option<f64>) -> i32 {
    if let Some(a) = alphap {
        ALPHA.set(a);
        if s == 0 {
            return 1;
        }
    }
    if i == -1 {
        return 0;
    }

    let cdim = CDIM.get();
    let si = st.simplex[s].neigh[i as usize].simp;
    let scn = (cdim - 1) as usize;
    let sin = i as usize;

    for k in 0..cdim {
        if st.simplex[s].neigh[k as usize].vert == HULL_INFINITY && k != i {
            return 1;
        }
    }
    let rs = radsq(st, s);
    let rsi = radsq(st, si);

    if rs < ALPHA.get() && rsi < ALPHA.get() {
        return 1;
    }

    /* swap_points(scn->vert, sin->vert) (`hull-ch.c:554`) */
    let t = st.simplex[s].neigh[scn].vert;
    st.simplex[s].neigh[scn].vert = st.simplex[s].neigh[sin].vert;
    st.simplex[s].neigh[sin].vert = t;
    let b = st.simplex[s].neigh[sin].basis;
    nullify_basis(st, b);
    st.simplex[s].neigh[sin].basis = 0;
    CDIM.set(CDIM.get() - 1);
    get_basis_sede(st, s);
    let normal = st.simplex[s].normal;
    let cdim1 = CDIM.get();
    let (nb, _) = reduce(st, normal, HULL_INFINITY, s, cdim1);
    st.simplex[s].normal = nb;
    let rsfi = radsq(st, s);

    let mut k = 0;
    while k < CDIM.get() && st.simplex[si].neigh[k as usize].simp != s {
        k += 1;
    }

    let scn_vert = st.simplex[s].neigh[scn].vert;
    let ssees = sees(st, scn_vert, s);
    let mut nsees = 0;
    if ssees == 0 {
        let v = st.simplex[si].neigh[k as usize].vert;
        nsees = sees(st, v, s);
    }
    let t = st.simplex[s].neigh[scn].vert;
    st.simplex[s].neigh[scn].vert = st.simplex[s].neigh[sin].vert;
    st.simplex[s].neigh[sin].vert = t;
    CDIM.set(CDIM.get() + 1);
    let b = st.simplex[s].normal;
    nullify_basis(st, b);
    st.simplex[s].normal = 0;
    let b = st.simplex[s].neigh[sin].basis;
    nullify_basis(st, b);
    st.simplex[s].neigh[sin].basis = 0;

    if ssees != 0 {
        return (ALPHA.get() < rs) as i32;
    }
    if nsees != 0 {
        return (ALPHA.get() < rsi) as i32;
    }

    assert!(rsfi <= rs + f32::EPSILON as f64 && rsfi <= rsi + f32::EPSILON as f64);

    (ALPHA.get() <= rsfi) as i32
}

/// Original static `conv_facetv` (`hull-ch.c:608`).
fn conv_facetv(st: &mut HullStorage, s: usize) -> usize {
    for i in 0..CDIM.get() {
        if st.simplex[s].neigh[i as usize].vert == HULL_INFINITY {
            return s;
        }
    }
    0
}

/// Original static `mark_points` (`hull-ch.c:616`).
fn mark_points(st: &mut HullStorage, s: usize) -> usize {
    for i in 0..CDIM.get() {
        let vert = st.simplex[s].neigh[i as usize].vert;
        if vert == HULL_INFINITY {
            continue;
        }
        let snum = SITE_NUM.get().unwrap()(vert) as usize;
        if st.simplex[s].mark != 0 {
            MO.with_borrow_mut(|m| m[snum] = 1);
        } else {
            MI.with_borrow_mut(|m| m[snum] = 1);
        }
    }
    0
}

/// Original `visit_outside_ashape` (`hull-ch.c:629`).
pub fn visit_outside_ashape(st: &mut HullStorage, root: usize, visit: VisitFunc) -> usize {
    let start = visit_hull(st, root, &mut |st, s| conv_facetv(st, s));
    visit_triang_gen(st, start, visit, &mut |st, s, i| alph_test(st, s, i, None))
}

/// Original static `check_ashape` (`hull-ch.c:633`).
fn check_ashape(st: &mut HullStorage, root: usize, alpha: f64) -> i32 {
    MI.with_borrow_mut(|m| m.iter_mut().for_each(|v| *v = 0));
    MO.with_borrow_mut(|m| m.iter_mut().for_each(|v| *v = 0));

    visit_hull(st, root, &mut |st, s| zero_marks(st, s));

    alph_test(st, 0, 0, Some(alpha));
    visit_outside_ashape(st, root, &mut |st, s| one_marks(st, s));

    visit_hull(st, root, &mut |st, s| mark_points(st, s));

    for i in 0..MAXPOINTS {
        if MO.with_borrow(|m| m[i]) != 0 && MI.with_borrow(|m| m[i]) == 0 {
            return 0;
        }
    }

    1
}

/// Original `find_alpha` (`hull-ch.c:651`).
pub fn find_alpha(st: &mut HullStorage, root: usize) -> f64 {
    let mut al = 0f32;
    let mut ah = 0f32;

    for i in 0..PDIM.get() as usize {
        let mx = MAXS.with(|m| m.get()[i]);
        let mn = MINS.with(|m| m.get()[i]);
        ah += ((mx - mn) * (mx - mn)) as f32;
    }
    check_ashape(st, root, ah as f64);
    for _ in 0..17 {
        /* DNM: take assignment out of function call, break lines */
        let am = (al + ah) / 2.;
        if check_ashape(st, root, am as f64) != 0 {
            ah = am;
        } else {
            al = am;
        }
        if (ah - al) / ah < 0.5 {
            break;
        }
    }
    1.1 * ah as f64
}

/// Original static `vols` (`hull-ch.c:677`).
///
/// `s` and `sn` are function statics in the source, taken from the simplex
/// free list on the first call and reused by every later one; `sn` is
/// `s->neigh`, so it is the same simplex here.
fn vols(st: &mut HullStorage, f: usize, t: usize, n: usize, depth: i32) {
    let tdim = CDIM.get();
    let mut nn = 0usize;

    if t == 0 {
        return;
    }

    if VOLS_S.get() == 0 {
        /* NEWL(simplex, s) */
        if st.simplex_list == 0 {
            new_block_simplex(st, 1);
        }
        let s = st.simplex_list;
        assert!(s != 0);
        st.simplex_list = st.simplex[s].next;
        VOLS_S.set(s);
    }
    let s = VOLS_S.get();
    CDIM.set(depth);
    st.simplex[s].normal = n;
    let key = st.tree[t].key;
    let signum = if depth > 1 && sees(st, key, s) != 0 {
        -1
    } else {
        1
    };
    CDIM.set(tdim);

    let tfgs = st.tree[t].fgs;
    if st.fg[tfgs].dist == 0. {
        st.simplex[s].neigh[(depth - 1) as usize].vert = key;
        let b = st.simplex[s].neigh[(depth - 1) as usize].basis;
        nullify_basis(st, b);
        st.simplex[s].neigh[(depth - 1) as usize].basis = 0;
        CDIM.set(depth);
        get_basis_sede(st, s);
        CDIM.set(tdim);
        let (newnn, _) = reduce(st, nn, HULL_INFINITY, s, depth);
        nn = newnn;
        let nnv = st.basis[nn].vecs;
        let rdim = RDIM.get() as usize;

        /* DNM: change tests ==Huge to > HugeCrit and != Huge to < HugeCrit */
        if key == HULL_INFINITY || st.fg[f].dist > HUGE_CRIT.get() || nearzero(nnv[rdim - 1]) {
            st.fg[tfgs].dist = HUGE.get();
        } else {
            st.fg[tfgs].dist =
                vec_dot_pdim(&nnv[..], &nnv[..]) / 4. / nnv[rdim - 1] / nnv[rdim - 1];
        }
        if st.fg[tfgs].facets == 0 {
            st.fg[tfgs].vol = 1.;
        } else {
            let facets = st.fg[tfgs].facets;
            vols(st, tfgs, facets, nn, depth + 1);
        }
    }

    assert!(st.fg[f].dist < HUGE_CRIT.get() || st.fg[tfgs].dist > HUGE_CRIT.get());
    if st.fg[tfgs].dist > HUGE_CRIT.get() || st.fg[tfgs].vol > HUGE_CRIT.get() {
        st.fg[f].vol = HUGE.get();
    } else {
        let sqq = st.fg[tfgs].dist - st.fg[f].dist;
        if nearzero(sqq) {
            st.fg[f].vol = 0.;
        } else {
            st.fg[f].vol +=
                signum as f64 * sqq.sqrt() * st.fg[tfgs].vol / (CDIM.get() - depth + 1) as f64;
        }
    }
    let left = st.tree[t].left;
    vols(st, f, left, n, depth);
    let right = st.tree[t].right;
    vols(st, f, right, n, depth);
}

/// Original `find_volumes` (`hull-ch.c:730`).
pub fn find_volumes(st: &mut HullStorage, faces_gr: usize, f: &mut dyn Write) {
    if faces_gr == 0 {
        return;
    }
    let facets = st.fg[faces_gr].facets;
    vols(st, faces_gr, facets, 0, 1);
    print_fg(st, faces_gr, f);
}

/// Original `set_ch_root` (`hull-ch.c:742`) — set root to `s`, for purposes of
/// getting normals etc.
pub fn set_ch_root(s: usize) {
    CH_ROOT.set(s);
}

/// Original `build_convex_hull` (`hull-ch.c:745`).
///
/// `get_s` returns next site each call; hull construction stops when NULL
/// returned.  `site_numm` returns number of site when given site.  `dim` is
/// the dimension of the point set.  If `vdd` then return Delaunay
/// triangulation.
pub fn build_convex_hull(
    st: &mut HullStorage,
    get_s: crate::imod::libwarp::hull::GetSite,
    site_numm: crate::imod::libwarp::hull::SiteNum,
    dim: i16,
    vdd: i16,
) -> usize {
    /* DNM: This made it crash with Intel compiler debug compile
    if (!Huge) Huge = DBL_MAX*DBL_MAX; */
    if HUGE.get() == 0. {
        HUGE.set(0.999 * f64::MAX);
        HUGE_CRIT.set(0.99 * HUGE.get());
    }

    CDIM.set(0);
    GET_SITE.set(Some(get_s));
    SITE_NUM.set(Some(site_numm));
    PDIM.set(dim as i32);
    VD.set(vdd);

    EXACT_BITS
        .set((f64::MANTISSA_DIGITS as f64 * (f64::RADIX as f64).ln() / 2.0f64.ln()).floor() as i32);
    B_ERR_MIN.set(
        (f64::EPSILON * MAXDIM as f64 * (1i32 << MAXDIM) as f64 * MAXDIM as f64 * 3.01) as f32,
    );
    B_ERR_MIN_SQ.set(B_ERR_MIN.get() * B_ERR_MIN.get());

    assert!(GET_SITE.get().is_some());
    assert!(SITE_NUM.get().is_some());

    let pdim = PDIM.get();
    RDIM.set(if VD.get() != 0 { pdim + 1 } else { pdim });
    let rdim = RDIM.get();
    if rdim > MAXDIM as i32 {
        panic(&c_format(
            "dimension bound MAXDIM exceeded; rdim=%d; pdim=%d\n",
            &[CArg::Int(rdim as i64), CArg::Int(pdim as i64)],
        ));
    }
    /* fprintf(DFILE, "rdim=%d; pdim=%d\n", rdim, pdim); fflush(DFILE); */

    SITE_SIZE.set(size_of::<Coord>() as i32 * pdim);
    POINT_SIZE.set(SITE_SIZE.get());
    BASIS_VEC_SIZE.set(size_of::<Coord>() * rdim as usize);
    BASIS_SIZE.set(size_of::<Basis>() + (2 * rdim as usize - 1) * size_of::<Coord>());
    SIMPLEX_SIZE.set(size_of::<Simplex>() + (rdim as usize - 1) * size_of::<Neighbor>());
    TREE_SIZE.set(size_of::<Tree>());
    FG_SIZE.set(size_of::<Fg>());

    let p;
    if VD.get() != 0 {
        p = HULL_INFINITY;
        /* NEWLRC(basis_s, infinity_basis) */
        if st.basis_list == 0 {
            new_block_basis(st, 1);
        }
        let ib = st.basis_list;
        assert!(ib != 0);
        st.basis_list = st.basis[ib].next;
        st.basis[ib].ref_count = 1;
        INFINITY_BASIS.set(ib);
        st.basis[ib].vecs[(2 * rdim - 1) as usize] = 1.;
        st.basis[ib].vecs[(rdim - 1) as usize] = 1.;
        st.basis[ib].sqa = 1.;
        st.basis[ib].sqb = 1.;
    } else {
        p = get_s();
        if p == 0 {
            P.set(0);
            return 0;
        }
    }
    P.set(p);

    /* NEWL(simplex, root) */
    if st.simplex_list == 0 {
        new_block_simplex(st, 1);
    }
    let root = st.simplex_list;
    assert!(root != 0);
    st.simplex_list = st.simplex[root].next;

    CH_ROOT.set(root);

    /* copy_simp(s, root): NEWL then memcpy then mod_refs(inc, root).  `cdim`
    is 0 here, so the reference loop runs once, over `peak`. */
    if st.simplex_list == 0 {
        new_block_simplex(st, 1);
    }
    let s = st.simplex_list;
    assert!(s != 0);
    st.simplex_list = st.simplex[s].next;
    st.simplex[s] = st.simplex[root];
    let cdim = CDIM.get();
    for imr in -1..cdim {
        let b = if imr < 0 {
            st.simplex[root].peak.basis
        } else {
            st.simplex[root].neigh[imr as usize].basis
        };
        if b != 0 {
            st.basis[b].ref_count += 1;
        }
    }

    st.simplex[root].peak.vert = p;
    st.simplex[root].peak.simp = s;
    st.simplex[s].peak.simp = root;

    buildhull(st, root);
    root
}

/// Original `free_hull_storage` (`hull-ch.c:817`).
///
/// Freeing every block leaves the source's surviving statics — `vols`'s `s`,
/// `make_facets`'s `ns`, `infinity_basis` — pointing into freed memory.  An
/// arena index would instead be out of range and panic, so the three are reset
/// here and in [`free_simplex_storage`]; the source reads none of them before
/// the next `build_convex_hull` reassigns them.
pub fn free_hull_storage(st: &mut HullStorage) {
    free_basis_storage(st);
    INFINITY_BASIS.set(0);
    free_simplex_storage(st);
    VOLS_S.set(0);
    free_tree_storage(st);
    free_fg_storage(st);
}
