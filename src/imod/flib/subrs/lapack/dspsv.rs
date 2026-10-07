//! LAPACK `DSPSV` boundary (`IMOD/flib/subrs/lapack/dspsv.f`, with
//! `dsptrf.f`/`dsptrs.f` and the BLAS routines under them).
//!
//! This is **not a translation**.  LAPACK/BLAS are out of scope to translate
//! (CLAUDE.md, the LAPACK/BLAS paragraph; `ORDER.md` §9 "LAPACK/BLAS"), and by
//! user decision they are provided through the pure-Rust crate `faer` (0.24,
//! `default-features = false`, so sequential like the reference).  The
//! factorisation is `faer`'s `LBLᵀ` with `PivotingStrategy::Partial` and its
//! unblocked kernel (`block_size = 0`), the closest `faer` offers to `DSPTRF`.
//!
//! `DSPTRF` with `UPLO = 'U'` factors `A = U D Uᵀ` from the **last** column
//! backwards; `faer` factors `P A Pᵀ = L B Lᵀ` from the first column forwards.
//! Reversing the index order (`J A J`, `J` the exchange matrix) turns one into
//! the other: `J U J` is unit lower triangular.  So the `'U'` case feeds `faer`
//! the reversed matrix and maps positions back with `K = N - k`; `'L'` needs no
//! reversal.
//!
//! **The pivot rule is not `DSPTRF`'s**, and no `faer` strategy is.  Both use
//! `alpha = (1 + sqrt(17)) / 8`, the column maximum `colmax` and the maximum
//! `rowmax` in the row/column of that element, but the test for keeping a 1x1
//! pivot without an interchange differs:
//!
//! ```text
//! DSPTRF (dsptrf.f):          |a_kk| >= alpha * colmax * (colmax / rowmax)   (Bunch–Kaufman)
//! faer 0.24 (factor.rs:840-841):|a_kk| >= alpha * rowmax * (rowmax / colmax)
//! ```
//!
//! Since `rowmax >= colmax` the `faer` test is stricter, so on a general
//! indefinite matrix `faer` interchanges or takes a 2x2 block where `DSPTRF`
//! does not.  Measured: `IPIV` differs in 265 of the ~400 random indefinite
//! test systems, while every diagonally dominant system — including every
//! tiltalign-shaped one — takes the same (null) pivot sequence and identical
//! `IPIV`.  The solution is inside the CLAUDE.md exemption either way (see the
//! tolerance on [`dspsv`]); `IPIV`, and in principle the position reported in
//! `INFO`, are only guaranteed identical when the two rules agree.  A different
//! `IDAMAX` tie-break (first maximum in LAPACK order vs. first in `faer`'s) is a
//! second, smaller source of the same thing.

use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::lblt::factor::{
    LbltParams, PivotingStrategy, cholesky_in_place, cholesky_in_place_scratch,
};
use faer::linalg::cholesky::lblt::solve::{solve_in_place, solve_in_place_scratch};
use faer::{Auto, Col, Mat};

/// Fortran `DSPSV( UPLO, N, NRHS, AP, IPIV, B, LDB, INFO )`, reached from C as
/// `dspsv_` (`IMOD/flib/tiltalign/solve_xyzd.cpp:17-24`).
///
/// Solves `A X = B` for symmetric `A` (order `n`) held in packed storage `ap`,
/// with `nrhs` right-hand sides in the column-major `b` (leading dimension
/// `ldb`).  The signature is LAPACK's, with the scalars by value and `info` as
/// the out-parameter the source reads.
///
/// **The only caller.** `solvePackedSums` (`solve_xyzd.cpp:455`) calls
/// `dspsv("U", &m, &one, ss, b, &ss[ind - 1], &m, &ierr, 1)` with
/// `m = mp - 1` and `ind = packedIndex(1, mp)`, i.e. the RHS is column `mp` of
/// the same packed array, starting right after the `m(m+1)/2` elements of `AP`.
/// A translation passes the two disjoint halves of one `split_at_mut` of `ss`
/// at `m * (m + 1) / 2`.  Note also that the source passes its `double *b`
/// (`bl`) as **`IPIV`**: the integer pivots are written over the start of
/// `bl`, which is harmless on success (`solvePackedSums` then overwrites all
/// `m` entries of `bl`) but is what `solveXyzd` reads back as coordinates when
/// `ierr != 0`, since it never tests `ierr` (`solve_xyzd.cpp:225-240`).  So
/// `ipiv` is filled in `DSPTRF`'s encoding (positive `KP` for a 1x1 block,
/// `-KP` in both entries of a 2x2 block), including when `info > 0`, from
/// `faer`'s own pivot sequence — which is `DSPTRF`'s only where the two pivot
/// rules agree (module comment).  The translator of `solveXyzd` has to
/// reproduce the aliasing itself (the integer pairs land in `bl` as the bits
/// of doubles).
///
/// **INFO.** `info < 0`: argument error, handled as LAPACK does — `XERBLA`
/// prints ` ** On entry to DSPSV  parameter number  N had an illegal value`
/// on stdout and `STOP`s (exit status 0).  `info = K > 0`: `DSPTRF` met a
/// column whose diagonal and off-diagonal parts were all exactly zero at step
/// `K` (`dsptrf.f`, `MAX( ABSAKK, COLMAX ).EQ.ZERO`), i.e. `D(K,K)` is exactly
/// zero; the first such `K` in factorisation order is reported, the
/// factorisation continues (so `ipiv` is complete) and `b` is left untouched,
/// since `DSPSV` calls `DSPTRS` only when `INFO = 0`.  `faer` reports none of
/// this; it is detected from the 1x1 blocks of `D` (in `faer` a 1x1 block has a
/// zero pivot exactly when its whole column was zero, the same condition).  A
/// 2x2 block is never tested for singularity, because `DSPTRF` does not test
/// it either (the pivot criterion keeps `|d11 d22| < d21^2`).  `INFO` matched
/// the reference in every singular test case (zero row/column at the first,
/// middle and last position, the zero matrix, an exactly eliminating rank-one
/// matrix, n = 1..30, both `UPLO`), but see the pivot-rule caveat in the
/// module comment: with several zero columns, or a 2x2 interchange that moves
/// the zero column, the reported `K` could differ.
///
/// **NaN input is outside what can be matched.**  A NaN column is what
/// `solvePackedSums` produces for an all-zero column of sums (its SD is 0, and
/// it divides by it).  In `DSPTRF` every comparison with the NaN is false, so it
/// falls through to a 2x2 pivot with `KP = IMAX` — and at `K = 1` `IMAX` was
/// never assigned and `IPIV(K-1)` is `IPIV(0)`, out of bounds.  Measured: the
/// reference `dspsv_` with the NaN column last (`K = N`, `N = 1, 4, 20`) ends in
/// `DGER`'s `XERBLA`, printing ` ** On entry to DGER   parameter number  1 had
/// an illegal value` and `STOP`ping the whole process with status 0; with the
/// NaN column first it returns `INFO = 0` and an all-NaN `b`.  Fixed in
/// translation (2026-09-26): a NaN or infinity anywhere in `AP` is reported as
/// `INFO = K > 0` for the first packed column holding one, in factorisation
/// order, with `IPIV` the identity and `b` untouched -- the same contract as
/// an exactly singular column, so callers take their existing failure path.
///
/// **Deviation.** On return `AP` holds `DSPTRF`'s packed factor in LAPACK; here
/// it is left unchanged.  No caller in the closure reads `AP` after the call
/// (`solveXyzd` re-zeroes `sprod` on entry, `solve_xyzd.cpp:120`).
///
/// **Tolerance** (CLAUDE.md LAPACK/BLAS exemption), measured against the
/// reference `libb3dlapk.so` `dspsv_` by `tests::reference_differential`, with
/// `rel = max_i |x_i - x_ref_i| / max_i |x_ref_i|` and `cond1` the 1-norm
/// condition number of `A`:
///
/// | class (cases) | max `rel` | max `rel / (eps cond1)` |
/// |---|---:|---:|
/// | random indefinite, n = 1..199 (134) | 7.6e-12 | 0.13 |
/// | diagonally dominant SPD (134) | 6.6e-16 | 1.24 |
/// | zero diagonal, forced 2x2 (132) | 2.1e-12 | 0.28 |
/// | Hilbert n = 2..12 (22) | 7.3e-2 | 0.016 |
/// | badly scaled, 1e-8 range (4) | 4.4e-13 | 3e-12 |
/// | tiltalign-shaped, m = 3..207 (6) | 1.3e-15 | 0.30 |
///
/// **Committed bound: `rel <= 16 * eps * max(cond1, 1)` for every system, and
/// additionally `rel <= 1e-10` for the tiltalign-shaped ones; `INFO`
/// identical always; `IPIV` identical for the tiltalign-shaped ones.**  The
/// bound is relative to conditioning because an absolute one is either
/// meaningless for an ill-conditioned matrix (Hilbert n = 12 legitimately
/// differs at 7e-2) or far looser than needed for a well-conditioned one.  This
/// is a single direct solve, so there is no iteration for a difference to grow
/// over here; the growth check belongs to the end-to-end `tiltalign`
/// differential, where this solve seeds `metro`.
pub fn dspsv(
    uplo: &str,
    n: i32,
    nrhs: i32,
    ap: &mut [f64],
    ipiv: &mut [i32],
    b: &mut [f64],
    ldb: i32,
    info: &mut i32,
) {
    let first = uplo.as_bytes().first().copied().unwrap_or(b' ');
    let upper = first.eq_ignore_ascii_case(&b'U');

    *info = 0;
    if !upper && !first.eq_ignore_ascii_case(&b'L') {
        *info = -1;
    } else if n < 0 {
        *info = -2;
    } else if nrhs < 0 {
        *info = -3;
    } else if ldb < 1.max(n) {
        *info = -7;
    }
    if *info != 0 {
        // XERBLA( 'DSPSV ', -INFO ): WRITE( *, FMT = 9999 ) SRNAME, INFO; STOP
        println!(
            " ** On entry to DSPSV  parameter number {:2} had an illegal value",
            -*info
        );
        crate::imod::libcfshr::b3dutil::exit(0);
    }

    let nn = n as usize;
    if nn == 0 {
        return;
    }

    // Non-finite matrix: fixed in translation (2026-09-26, `BUGS.md` "Reference
    // `dsptrf` on a NaN column").  The reference `DSPTRF` reads an unset `IMAX`
    // and writes `IPIV(0)` on a NaN column (see the NaN paragraph above).
    // Defined behaviour: the matrix is reported singular at the first packed
    // column, in `DSPTRF`'s factorisation order (`K = N..1` for 'U', `1..N`
    // for 'L'), that holds a NaN or infinity; `IPIV` is the identity and `B`
    // is left untouched, exactly as for an exactly singular column.
    let ncols = nn;
    let column_start = |k: usize| -> (usize, usize) {
        // Packed range of LAPACK column k (1-based) of the stored triangle.
        if upper {
            let start = (k - 1) * k / 2;
            (start, start + k)
        } else {
            let start = (k - 1) * (2 * ncols - k + 2) / 2;
            (start, start + ncols - k + 1)
        }
    };
    let order: Vec<usize> = if upper {
        (1..=nn).rev().collect()
    } else {
        (1..=nn).collect()
    };
    if let Some(&k) = order.iter().find(|&&k| {
        let (lo, hi) = column_start(k);
        ap[lo..hi].iter().any(|v| !v.is_finite())
    }) {
        for (i, p) in ipiv[..nn].iter_mut().enumerate() {
            *p = (i + 1) as i32;
        }
        *info = k as i32;
        return;
    }

    // Dense column-major copy of the symmetric matrix, in faer's forward order.
    // 'U': element (r, c) of the reversed matrix is A(n-1-r, n-1-c), an upper
    // element of A at packed position i + j(j+1)/2 (0-based, i <= j).
    // 'L': A(i, j), i >= j, at packed position i + j(2n-j-1)/2.
    let mut a = Mat::<f64>::zeros(nn, nn);
    for c in 0..nn {
        for r in c..nn {
            let v = if upper {
                let (i, j) = (nn - 1 - r, nn - 1 - c);
                ap[i + j * (j + 1) / 2]
            } else {
                ap[r + c * (2 * nn - c - 1) / 2]
            };
            a[(r, c)] = v;
            a[(c, r)] = v;
        }
    }

    let mut params = <LbltParams as Auto<f64>>::auto();
    params.pivoting = PivotingStrategy::Partial;
    // DSPTRF is unblocked; so is this (block_size < 2 selects faer's
    // unblocked kernel), which keeps the update order column by column.
    params.block_size = 0;

    let nrhs_u = nrhs as usize;
    let mut mem = MemBuffer::new(StackReq::any_of(&[
        cholesky_in_place_scratch::<usize, f64>(nn, Par::Seq, params.into()),
        solve_in_place_scratch::<usize, f64>(nn, nrhs_u.max(1), Par::Seq),
    ]));
    let stack = MemStack::new(&mut mem);

    let mut subdiag = Col::<f64>::zeros(nn);
    let mut perm_fwd = vec![0usize; nn];
    let mut perm_inv = vec![0usize; nn];
    let (_, perm) = cholesky_in_place(
        a.as_mut(),
        subdiag.as_diagonal_mut(),
        &mut perm_fwd,
        &mut perm_inv,
        Par::Seq,
        stack,
        params.into(),
    );

    // Recover faer's per-step interchanges p_k (p_k >= k) from the composed
    // permutation: faer builds it as `perm.swap(k, p_k)` for k = 0..n, and a
    // position k is never touched after step k.
    let fwd = perm.arrays().0;
    let mut cur: Vec<usize> = (0..nn).collect();
    let mut at: Vec<usize> = (0..nn).collect();
    let mut piv = vec![0usize; nn];
    for k in 0..nn {
        let p = at[fwd[k]];
        piv[k] = p;
        let (vk, vp) = (cur[k], cur[p]);
        cur.swap(k, p);
        at[vk] = p;
        at[vp] = k;
    }

    // INFO and IPIV in DSPTRF's encoding.  Position k of faer's order is
    // LAPACK column K = n - k for 'U' and K = k + 1 for 'L'.
    let lapack_index = |k: usize| -> i32 {
        if upper {
            (nn - k) as i32
        } else {
            (k + 1) as i32
        }
    };
    let mut k = 0usize;
    while k < nn {
        if subdiag[k] != 0.0 {
            let kp = -lapack_index(piv[k + 1]);
            ipiv[(lapack_index(k) - 1) as usize] = kp;
            ipiv[(lapack_index(k + 1) - 1) as usize] = kp;
            k += 2;
        } else {
            if a[(k, k)] == 0.0 && *info == 0 {
                *info = lapack_index(k);
            }
            ipiv[(lapack_index(k) - 1) as usize] = lapack_index(piv[k]);
            k += 1;
        }
    }
    if *info != 0 || nrhs_u == 0 {
        return;
    }

    let ldb_u = ldb as usize;
    let mut x = Mat::<f64>::zeros(nn, nrhs_u);
    for j in 0..nrhs_u {
        for i in 0..nn {
            x[(i, j)] = if upper {
                b[(nn - 1 - i) + j * ldb_u]
            } else {
                b[i + j * ldb_u]
            };
        }
    }
    solve_in_place(
        a.as_ref(),
        a.as_ref().diagonal(),
        subdiag.as_diagonal(),
        perm,
        x.as_mut(),
        Par::Seq,
        stack,
    );
    for j in 0..nrhs_u {
        for i in 0..nn {
            let v = x[(i, j)];
            if upper {
                b[(nn - 1 - i) + j * ldb_u] = v;
            } else {
                b[i + j * ldb_u] = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Isolation differential against the reference build's `dspsv_`
    //! (`/tmp/imod-reference-build/buildlib/libb3dlapk.so`, BLAS from
    //! `libb3dblas.so`).  Skipped, with a message, when the reference tree is
    //! absent.  Run with `--nocapture` to see the error distribution and the
    //! timing table.
    use super::dspsv;
    use std::ffi::c_char;
    use std::time::Instant;

    type DspsvFn = unsafe extern "C" fn(
        *const c_char,
        *const i32,
        *const i32,
        *mut f64,
        *mut i32,
        *mut f64,
        *const i32,
        *mut i32,
        usize,
    );

    fn load_reference() -> Option<DspsvFn> {
        // The reference build is a Linux tree of `.so` files.
        #[cfg(not(target_os = "linux"))]
        return None;
        #[cfg(target_os = "linux")]
        let dir = "/tmp/imod-reference-build/buildlib";
        // SAFETY: test-only foreign boundary; the libraries are the reference
        // build's own gfortran LAPACK/BLAS and stay loaded for the process.
        #[cfg(target_os = "linux")]
        unsafe {
            // The reference libraries leave the gfortran runtime unresolved
            // (XERBLA's WRITE/STOP); the reference executables link it.
            let gfortran = libc::dlopen(
                c"libgfortran.so.5".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_GLOBAL,
            );
            let blas = libc::dlopen(
                c"/tmp/imod-reference-build/buildlib/libb3dblas.so".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_GLOBAL,
            );
            let lapk = libc::dlopen(
                c"/tmp/imod-reference-build/buildlib/libb3dlapk.so".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_GLOBAL,
            );
            if gfortran.is_null() || blas.is_null() || lapk.is_null() {
                println!("dspsv differential skipped: cannot load {dir}/libb3d{{blas,lapk}}.so");
                return None;
            }
            let sym = libc::dlsym(lapk, c"dspsv_".as_ptr());
            if sym.is_null() {
                println!("dspsv differential skipped: no dspsv_ in libb3dlapk.so");
                return None;
            }
            Some(std::mem::transmute::<*mut libc::c_void, DspsvFn>(sym))
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    /// Packs a dense symmetric `n x n` (row-major `d[i*n+j]`) into `uplo`
    /// packed storage.
    fn pack(d: &[f64], n: usize, upper: bool) -> Vec<f64> {
        let mut ap = Vec::with_capacity(n * (n + 1) / 2);
        for j in 0..n {
            if upper {
                for i in 0..=j {
                    ap.push(d[i * n + j]);
                }
            } else {
                for i in j..n {
                    ap.push(d[i * n + j]);
                }
            }
        }
        ap
    }

    struct Outcome {
        info: i32,
        ipiv: Vec<i32>,
        b: Vec<f64>,
    }

    fn run_rust(uplo: &str, n: usize, ap: &[f64], b: &[f64], nrhs: usize) -> Outcome {
        let mut ap = ap.to_vec();
        let mut b = b.to_vec();
        let mut ipiv = vec![0i32; n.max(1)];
        let mut info = 0;
        dspsv(
            uplo,
            n as i32,
            nrhs as i32,
            &mut ap,
            &mut ipiv,
            &mut b,
            n.max(1) as i32,
            &mut info,
        );
        Outcome { info, ipiv, b }
    }

    fn run_ref(f: DspsvFn, uplo: &str, n: usize, ap: &[f64], b: &[f64], nrhs: usize) -> Outcome {
        let mut ap = ap.to_vec();
        let mut b = b.to_vec();
        let mut ipiv = vec![0i32; n.max(1)];
        let mut info = 0;
        let (ni, nr, ld) = (n as i32, nrhs as i32, n.max(1) as i32);
        let u = if uplo == "U" { c"U" } else { c"L" };
        // SAFETY: all buffers are sized as LAPACK requires.
        unsafe {
            f(
                u.as_ptr(),
                &ni,
                &nr,
                ap.as_mut_ptr(),
                ipiv.as_mut_ptr(),
                b.as_mut_ptr(),
                &ld,
                &mut info,
                1,
            );
        }
        Outcome { info, ipiv, b }
    }

    /// 1-norm condition number, with the inverse taken from the reference.
    fn cond1(f: DspsvFn, d: &[f64], n: usize) -> f64 {
        let ap = pack(d, n, true);
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let inv = run_ref(f, "U", n, &ap, &eye, n);
        if inv.info != 0 {
            return f64::INFINITY;
        }
        let norm1 = |m: &[f64]| {
            (0..n)
                .map(|j| (0..n).map(|i| m[i * n + j].abs()).sum::<f64>())
                .fold(0.0, f64::max)
        };
        norm1(d) * norm1(&inv.b)
    }

    /// Relative max-norm difference of the solutions.
    fn rel_err(x: &[f64], r: &[f64]) -> f64 {
        let num = x
            .iter()
            .zip(r)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let den = r.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if den == 0.0 { num } else { num / den }
    }

    /// tiltalign's `solveXyzd` normal equations for `npt` real points over
    /// `nview` views, built the way `solve_xyzd.cpp:140-225` builds them
    /// (last point's coordinates eliminated as minus the sum of the others),
    /// then scaled by `solvePackedSums` (`:425-452`).  Returns the packed
    /// `ss` (AP followed by the RHS column) and `m`.
    fn tiltalign_system(rng: &mut Rng, npt: usize, nview: usize) -> (Vec<f64>, usize) {
        let m = 3 * (npt - 1);
        let mp = m + 1;
        let mut sx = vec![0.0f64; mp];
        let mut ss = vec![0.0f64; mp * (mp + 1) / 2];
        let pidx = |i: usize, j: usize| i + (j - 1) * j / 2; // 1-based
        let pts: Vec<[f64; 3]> = (0..npt)
            .map(|_| [rng.next() * 500.0, rng.next() * 500.0, rng.next() * 100.0])
            .collect();
        let mut nrows = 0;
        for iv in 0..nview {
            let tilt = (-60.0 + 120.0 * iv as f64 / (nview - 1) as f64).to_radians();
            let rot = 0.1 * rng.next();
            let (ct, st, cr, sr) = (tilt.cos(), tilt.sin(), rot.cos(), rot.sin());
            // x' = cr*(ct*x + st*z) - sr*y ; y' = sr*(ct*x + st*z) + cr*y
            let coefs = [[cr * ct, -sr, cr * st], [sr * ct, cr, sr * st]];
            for ip in 0..npt {
                for c in coefs.iter() {
                    let mut row = vec![0.0f32; mp];
                    let proj: f64 =
                        (0..3).map(|k| c[k] * pts[ip][k]).sum::<f64>() + rng.next() * 0.5;
                    if ip < npt - 1 {
                        for k in 0..3 {
                            row[3 * ip + k] = c[k] as f32;
                        }
                    } else {
                        for q in 0..npt - 1 {
                            for k in 0..3 {
                                row[3 * q + k] = -c[k] as f32;
                            }
                        }
                    }
                    row[m] = proj as f32;
                    for j in 1..=mp {
                        sx[j - 1] += row[j - 1] as f64;
                        for i in 1..=j {
                            ss[pidx(i, j) - 1] += row[i - 1] as f64 * row[j - 1] as f64;
                        }
                    }
                    nrows += 1;
                }
            }
        }
        let mut sd = vec![0.0f64; mp];
        for i in 1..=mp {
            sd[i - 1] = ((ss[pidx(i, i) - 1] - sx[i - 1] * sx[i - 1] / nrows as f64)
                / (nrows as f64 - 1.0))
                .sqrt();
        }
        for j in 1..=mp {
            let ind = pidx(1, j);
            for i in 1..=j {
                ss[ind + i - 2] /= sd[i - 1] * sd[j - 1];
            }
        }
        (ss, m)
    }

    #[test]
    fn reference_differential() {
        let Some(f) = load_reference() else { return };
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let eps = f64::EPSILON;
        // (class, max rel err, max rel err / (eps * cond))
        let mut stats: Vec<(&str, f64, f64, usize)> = Vec::new();
        let mut record = |class: &'static str, e: f64, c: f64| {
            let r = e / (eps * c);
            if let Some(s) = stats.iter_mut().find(|s| s.0 == class) {
                s.1 = s.1.max(e);
                s.2 = s.2.max(r);
                s.3 += 1;
            } else {
                stats.push((class, e, r, 1));
            }
        };
        let mut ipiv_mismatch = 0;

        let mut check =
            |class: &'static str, d: &[f64], n: usize, upper: bool, rng: &mut Rng, cond: f64| {
                let uplo = if upper { "U" } else { "L" };
                let ap = pack(d, n, upper);
                let b: Vec<f64> = (0..n).map(|_| rng.next()).collect();
                let x = run_rust(uplo, n, &ap, &b, 1);
                let r = run_ref(f, uplo, n, &ap, &b, 1);
                assert_eq!(x.info, r.info, "{class} n={n} uplo={uplo}: INFO");
                if x.ipiv[..n] != r.ipiv[..n] {
                    ipiv_mismatch += 1;
                }
                if r.info == 0 {
                    let e = rel_err(&x.b, &r.b);
                    record(class, e, cond);
                    assert!(
                        e <= 16.0 * eps * cond.max(1.0),
                        "{class} n={n} uplo={uplo}: rel {e:e} cond {cond:e}"
                    );
                } else {
                    assert_eq!(x.b, b, "{class}: B must be untouched when INFO > 0");
                }
            };

        for n in (1..=200).step_by(3) {
            for upper in [true, false] {
                // Symmetric indefinite, entries uniform in [-1, 1].
                let mut d = vec![0.0; n * n];
                for j in 0..n {
                    for i in 0..=j {
                        let v = rng.next();
                        d[i * n + j] = v;
                        d[j * n + i] = v;
                    }
                }
                let c = cond1(f, &d, n);
                check("indefinite", &d, n, upper, &mut rng, c);
                // SPD: diagonally dominant.
                for i in 0..n {
                    d[i * n + i] = n as f64 + rng.next().abs();
                }
                let c = cond1(f, &d, n);
                check("spd", &d, n, upper, &mut rng, c);
                // Zero diagonal (forces 2x2 pivots).
                for i in 0..n {
                    d[i * n + i] = 0.0;
                }
                let c = cond1(f, &d, n);
                check("zero-diag", &d, n, upper, &mut rng, c);
            }
        }
        // Ill-conditioned: Hilbert matrices, and badly scaled indefinite ones.
        for n in 2..=12 {
            for upper in [true, false] {
                let d: Vec<f64> = (0..n * n)
                    .map(|k| 1.0 / ((k / n + k % n) as f64 + 1.0))
                    .collect();
                let c = cond1(f, &d, n);
                check("hilbert", &d, n, upper, &mut rng, c);
            }
        }
        for n in [5, 20, 60, 120] {
            let s: Vec<f64> = (0..n)
                .map(|i| 10f64.powf(-8.0 * i as f64 / n as f64))
                .collect();
            let mut d = vec![0.0; n * n];
            for j in 0..n {
                for i in 0..=j {
                    let v = rng.next() * s[i] * s[j];
                    d[i * n + j] = v;
                    d[j * n + i] = v;
                }
            }
            let c = cond1(f, &d, n);
            check("scaled", &d, n, true, &mut rng, c);
        }
        // Exactly singular: a zero row/column at several positions, the zero
        // matrix, and a rank-one matrix of powers of two (exact elimination).
        for n in [1, 2, 3, 7, 30] {
            for zero_at in [0, n / 2, n - 1] {
                for upper in [true, false] {
                    let mut d = vec![0.0; n * n];
                    for j in 0..n {
                        for i in 0..=j {
                            let v = rng.next();
                            d[i * n + j] = v;
                            d[j * n + i] = v;
                        }
                    }
                    for k in 0..n {
                        d[zero_at * n + k] = 0.0;
                        d[k * n + zero_at] = 0.0;
                    }
                    check("singular", &d, n, upper, &mut rng, 1.0);
                    let z = vec![0.0; n * n];
                    check("singular", &z, n, upper, &mut rng, 1.0);
                    let v: Vec<f64> = (0..n).map(|i| (1u64 << (i % 5)) as f64).collect();
                    let r1: Vec<f64> = (0..n * n).map(|k| v[k / n] * v[k % n]).collect();
                    check("singular", &r1, n, upper, &mut rng, 1.0);
                }
            }
        }

        // tiltalign-shaped: the caller's own packing, RHS inside `ss`.
        let mut tilt_max = 0.0f64;
        let mut tilt_ratio = 0.0f64;
        for (npt, nview) in [(2, 5), (4, 41), (10, 61), (25, 61), (50, 81), (70, 121)] {
            let (ss, m) = tiltalign_system(&mut rng, npt, nview);
            let na = m * (m + 1) / 2;
            let x = run_rust("U", m, &ss[..na], &ss[na..na + m], 1);
            let r = run_ref(f, "U", m, &ss[..na], &ss[na..na + m], 1);
            assert_eq!(x.info, r.info, "tiltalign m={m}: INFO");
            assert_eq!(x.ipiv, r.ipiv, "tiltalign m={m}: IPIV");
            let e = rel_err(&x.b, &r.b);
            let mut d = vec![0.0; m * m];
            for j in 0..m {
                for i in 0..=j {
                    d[i * m + j] = ss[i + j * (j + 1) / 2];
                    d[j * m + i] = ss[i + j * (j + 1) / 2];
                }
            }
            let c = cond1(f, &d, m);
            tilt_max = tilt_max.max(e);
            tilt_ratio = tilt_ratio.max(e / (eps * c));
            assert!(
                e <= 1e-10 && e <= 16.0 * eps * c,
                "tiltalign m={m}: rel {e:e} cond {c:e}"
            );
        }

        for (class, e, r, count) in &stats {
            println!(
                "dspsv vs reference: {class:<11} {count:4} cases  max rel {e:9.2e}  max rel/(eps*cond1) {r:9.2e}"
            );
        }
        println!(
            "dspsv vs reference: tiltalign   max rel {tilt_max:9.2e}  max rel/(eps*cond1) {tilt_ratio:9.2e}"
        );
        // Not asserted: faer's 1x1-without-interchange test differs from
        // DSPTRF's (see the module comment), so general indefinite matrices
        // take different pivots.
        println!("dspsv vs reference: IPIV differs in {ipiv_mismatch} random cases (pivot rule)");

        // NaN input is deliberately not compared in-process: see the NaN
        // paragraph on `dspsv` — the reference can reach XERBLA's STOP, which
        // would end this test binary with status 0.

        // Timing, tiltalign-shaped systems.
        for npt in [18, 35, 68, 101, 135, 168] {
            let (ss, m) = tiltalign_system(&mut rng, npt, 61);
            let na = m * (m + 1) / 2;
            let reps = (2_000_000 / (m * m * m / 3 + 1)).clamp(3, 2000);
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(run_rust("U", m, &ss[..na], &ss[na..na + m], 1));
            }
            let tr = t.elapsed().as_secs_f64() / reps as f64;
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(run_ref(f, "U", m, &ss[..na], &ss[na..na + m], 1));
            }
            let tn = t.elapsed().as_secs_f64() / reps as f64;
            println!(
                "dspsv timing m={m:4}: faer {:9.3} ms  reference {:9.3} ms  ratio {:5.2}",
                tr * 1e3,
                tn * 1e3,
                tr / tn
            );
        }
    }

    /// `BUGS.md` "Reference `dsptrf` on a NaN column": defined as singular.
    #[test]
    fn non_finite_column_is_reported_singular() {
        // 3x3 'U' packed: A11 A12 A22 A13 A23 A33; NaN column 1 (first packed).
        let mut ap = [f64::NAN, f64::NAN, 2.0, f64::NAN, 1.0, 3.0];
        let mut b = [1.0, 2.0, 3.0];
        let mut ipiv = [0; 3];
        let mut info = 0;
        dspsv("U", 3, 1, &mut ap, &mut ipiv, &mut b, 3, &mut info);
        // Column 3 holds A13 = NaN and is the first in 'U' order (K = N..1).
        assert_eq!(info, 3);
        assert_eq!(ipiv, [1, 2, 3]);
        assert_eq!(b, [1.0, 2.0, 3.0]);
        // NaN only in the last column's diagonal.
        let mut ap = [4.0, 1.0, 5.0, 0.0, 1.0, f64::NAN];
        let mut info = 0;
        dspsv(
            "U",
            3,
            1,
            &mut ap,
            &mut ipiv,
            &mut [1.0, 2.0, 3.0],
            3,
            &mut info,
        );
        assert_eq!(info, 3);
        // 'L', NaN in column 2 only (A22, A32).
        let mut ap = [4.0, 0.0, 1.0, f64::NAN, f64::NAN, 6.0];
        let mut info = 0;
        dspsv(
            "L",
            3,
            1,
            &mut ap,
            &mut ipiv,
            &mut [1.0, 2.0, 3.0],
            3,
            &mut info,
        );
        assert_eq!(info, 2);
        let mut ap = [f64::NAN];
        let mut info = 0;
        dspsv("U", 1, 1, &mut ap, &mut ipiv, &mut [1.0], 1, &mut info);
        assert_eq!(info, 1);
    }
}
