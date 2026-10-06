//! LAPACK `DSYSV` boundary (`IMOD/flib/subrs/lapack/dsysv.f`, with
//! `dsytrf.f`/`dsytrs.f`, `ilaenv.f` and the BLAS routines under them).
//!
//! **Not a translation**: LAPACK/BLAS are provided through `faer` (CLAUDE.md,
//! the LAPACK/BLAS paragraph).  `DSYSV` factors the full-storage symmetric
//! matrix with the same Bunch–Kaufman `U D Uᵀ` (`DSYTRF`, blocked) that
//! `DSPTRF` applies to packed storage, so the solve is done by packing the
//! referenced triangle and calling [`dspsv`], whose `faer` mapping, pivot-rule
//! caveat and `INFO` contract are documented there.  Only the `UPLO`
//! triangle of `A` is read, as in LAPACK.
//!
//! The one caller is `flattenwarp.c` (`prepareTPS`/`fitTPS`, thin plate
//! spline smoothing), which first queries the workspace with `LWORK = -1`:
//! `DSYSV` answers `N * NB` with `NB = ILAENV(1, 'DSYTRF', ...)`, 64 in the
//! reference `ilaenv.f` (`:205`), and the same value is returned here.
//!
//! **Tolerance** (CLAUDE.md exemption; checked against the reference
//! `libb3dlapk.so` `dsysv_` by `tests::reference_differential` on thin plate
//! spline systems of the shape `fitTPS` builds, n = 4..124): `max|x - x_ref| /
//! max|x_ref| <= 1e-9`.  `INFO` identical.
//!
//! **Deviation.** On return LAPACK leaves the factor in `A` and the pivots in
//! `IPIV`; here `A` is unchanged and `IPIV` holds `DSPTRF`'s pivots (the two
//! encodings agree).  `flattenwarp` reloads `A` before every solve and never
//! reads `IPIV`.

use super::dspsv::dspsv;

/// Fortran `DSYSV( UPLO, N, NRHS, A, LDA, IPIV, B, LDB, WORK, LWORK, INFO )`,
/// called from C as `dsysv_` (`IMOD/include/lapackc.h`).  `work[0]` receives
/// the optimal `LWORK`.
#[allow(clippy::too_many_arguments)]
pub fn dsysv(
    uplo: &str,
    n: i32,
    nrhs: i32,
    a: &mut [f64],
    lda: i32,
    ipiv: &mut [i32],
    b: &mut [f64],
    ldb: i32,
    work: &mut [f64],
    lwork: i32,
    info: &mut i32,
) {
    let first = uplo.as_bytes().first().copied().unwrap_or(b' ');
    let upper = first.eq_ignore_ascii_case(&b'U');
    let lquery = lwork == -1;
    *info = 0;
    if !upper && !first.eq_ignore_ascii_case(&b'L') {
        *info = -1;
    } else if n < 0 {
        *info = -2;
    } else if nrhs < 0 {
        *info = -3;
    } else if lda < 1.max(n) {
        *info = -5;
    } else if ldb < 1.max(n) {
        *info = -8;
    } else if lwork < 1 && !lquery {
        *info = -10;
    }
    let mut lwkopt = 0;
    if *info == 0 {
        // NB = ILAENV( 1, 'DSYTRF', UPLO, N, -1, -1, -1 ) = 64; LWKOPT = N*NB
        lwkopt = n * 64;
        work[0] = lwkopt as f64;
    }
    if *info != 0 {
        // XERBLA( 'DSYSV ', -INFO ): WRITE( *, FMT = 9999 ) SRNAME, INFO; STOP
        println!(
            " ** On entry to DSYSV  parameter number {:2} had an illegal value",
            -*info
        );
        crate::imod::libcfshr::b3dutil::exit(0);
    }
    if lquery {
        return;
    }
    let nn = n as usize;
    let ld = lda as usize;
    // Pack the referenced triangle, column by column (LAPACK packed order).
    let mut ap = Vec::with_capacity(nn * (nn + 1) / 2);
    for j in 0..nn {
        if upper {
            for i in 0..=j {
                ap.push(a[i + j * ld]);
            }
        } else {
            for i in j..nn {
                ap.push(a[i + j * ld]);
            }
        }
    }
    dspsv(uplo, n, nrhs, &mut ap, ipiv, b, ldb, info);
    work[0] = lwkopt as f64;
}

#[cfg(test)]
mod tests {
    //! Isolation differential against the reference build's `dsysv_`.
    //! Skipped, with a message, when the reference tree is absent.
    use super::dsysv;
    use std::ffi::c_char;

    type DsysvFn = unsafe extern "C" fn(
        *const c_char,
        *const i32,
        *const i32,
        *mut f64,
        *const i32,
        *mut i32,
        *mut f64,
        *const i32,
        *mut f64,
        *const i32,
        *mut i32,
        usize,
    );

    fn load_reference() -> Option<DsysvFn> {
        // SAFETY: test-only foreign boundary; the reference build's own
        // gfortran LAPACK/BLAS, loaded for the rest of the process.
        unsafe {
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
                println!("dsysv differential skipped: reference LAPACK not loadable");
                return None;
            }
            let sym = libc::dlsym(lapk, c"dsysv_".as_ptr());
            if sym.is_null() {
                println!("dsysv differential skipped: no dsysv_");
                return None;
            }
            Some(std::mem::transmute::<*mut libc::c_void, DsysvFn>(sym))
        }
    }

    /// A thin plate spline system as `fitTPS` builds it: `U(r)` kernel on
    /// `np` scattered points, the affine border and zero corner, `lambda` on
    /// the diagonal.
    fn tps_system(np: usize, seed: u64, lambda: f64) -> (Vec<f64>, Vec<f64>) {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let pts: Vec<(f64, f64, f64)> = (0..np).map(|_| (next(), next(), next())).collect();
        let n = np + 3;
        let mut a = vec![0.0; n * n];
        let mut y = vec![0.0; n];
        let mut alpha = 0.0;
        for r in 0..np {
            a[r * n + n - 3] = 1.;
            a[r * n + n - 2] = pts[r].0;
            a[r * n + n - 1] = pts[r].1;
            a[n * (n - 3) + r] = 1.;
            a[n * (n - 2) + r] = pts[r].0;
            a[n * (n - 1) + r] = pts[r].1;
            y[r] = pts[r].2;
            for c in r + 1..np {
                let (dx, dy) = (pts[r].0 - pts[c].0, pts[r].1 - pts[c].1);
                let rr = (dx * dx + dy * dy).sqrt();
                alpha += rr;
                let u = if rr > 0. { rr * rr * rr.ln() } else { 0. };
                a[r * n + c] = u;
                a[c * n + r] = u;
            }
        }
        alpha *= 2. / (np * np) as f64;
        for i in 0..np {
            a[i * n + i] = lambda * alpha * alpha / np as f64;
        }
        (a, y)
    }

    #[test]
    fn reference_differential() {
        let Some(reference) = load_reference() else {
            return;
        };
        let mut worst: f64 = 0.;
        for (k, np) in [1usize, 3, 7, 12, 30, 60, 121].iter().enumerate() {
            for lambda in [1e-3, 1.0, 1e3] {
                let (a, y) = tps_system(*np, 0x9e37_79b9 + k as u64, lambda);
                let n = (np + 3) as i32;
                let one = 1;
                // Workspace query.
                let (mut ours_q, mut info) = ([0.0; 1], 0);
                let mut ipiv = vec![0; n as usize];
                let (mut a1, mut b1) = (a.clone(), y.clone());
                dsysv("U", n, 1, &mut a1, n, &mut ipiv, &mut b1, n, &mut ours_q, -1, &mut info);
                let mut ref_q = [0.0f64; 1];
                let mut rinfo = 0;
                let mut ripiv = vec![0; n as usize];
                let mut a2 = a.clone();
                let mut b2 = y.clone();
                unsafe {
                    reference(
                        c"U".as_ptr(),
                        &n,
                        &one,
                        a2.as_mut_ptr(),
                        &n,
                        ripiv.as_mut_ptr(),
                        b2.as_mut_ptr(),
                        &n,
                        ref_q.as_mut_ptr(),
                        &-1,
                        &mut rinfo,
                        1,
                    );
                }
                assert_eq!(ours_q[0], ref_q[0], "workspace query n={n}");
                let lwork = ref_q[0] as i32;
                let mut work = vec![0.0; lwork as usize];
                dsysv("U", n, 1, &mut a1, n, &mut ipiv, &mut b1, n, &mut work, lwork, &mut info);
                let mut rwork = vec![0.0; lwork as usize];
                unsafe {
                    reference(
                        c"U".as_ptr(),
                        &n,
                        &one,
                        a2.as_mut_ptr(),
                        &n,
                        ripiv.as_mut_ptr(),
                        b2.as_mut_ptr(),
                        &n,
                        rwork.as_mut_ptr(),
                        &lwork,
                        &mut rinfo,
                        1,
                    );
                }
                assert_eq!(info, rinfo, "INFO n={n} lambda={lambda}");
                let scale = b2.iter().fold(0f64, |m, v| m.max(v.abs()));
                let diff = b1.iter().zip(&b2).fold(0f64, |m, (x, r)| m.max((x - r).abs()));
                let rel = diff / scale.max(f64::MIN_POSITIVE);
                worst = worst.max(rel);
                assert!(rel <= 1e-9, "n={n} lambda={lambda}: rel {rel:e}");
            }
        }
        println!("dsysv vs reference: max rel {worst:e}");
    }
}
