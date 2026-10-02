//! Translation of `IMOD/raptor/opencv/cxsvd.cpp` (the parts RAPTOR
//! reaches): the double-precision singular value decomposition
//! `icvSVD_64f` and back substitution `icvSVBkSb_64f`, with the `cvSVD` and
//! `cvSVBkSb` entry points as `cvSolve( A, b, x, CV_SVD )` calls them
//! (`A` with at least as many rows as columns, `CV_SVD_U_T + CV_SVD_V_T`,
//! a one-column `w`).  The single-precision routines are not reached.
//!
//! The C walks its matrices with pointers, including one element before
//! the current Householder vector (`hv[-1]`) and one column before the
//! current sub-matrix (`y[-1]`).  Here each such pointer is a buffer and an
//! offset.  The Householder vector `hv` lives in the `U` output, the `V`
//! output or the scratch vector `hv0`, depending on the step.

use super::cxerror::{CV_STS_BAD_SIZE, CV_STS_UNMATCHED_SIZES, cv_error};
use super::cxtypes::*;

/// `CV_SVD_MODIFY_A`.
pub const CV_SVD_MODIFY_A: i32 = 1;
/// `CV_SVD_U_T`.
pub const CV_SVD_U_T: i32 = 2;
/// `CV_SVD_V_T`.
pub const CV_SVD_V_T: i32 = 4;

/// `icvMatrAXPY_64f( int m, int n, const double* x, int dx, const double* a,
/// double* y, int dy )` (`cxsvd.cpp:65`): `y[0:m,0:n] += diag(a[0:1,0:m]) *
/// x[0:m,0:n]`.  Offsets and strides are in elements.
#[allow(clippy::too_many_arguments)]
fn icv_matr_axpy_64f(
    m: i32,
    n: i32,
    x: &[f64],
    mut xo: usize,
    dx: usize,
    a: &[f64],
    ao: usize,
    y: &mut [f64],
    mut yo: usize,
    dy: usize,
) {
    let n = n.max(0) as usize;
    for i in 0..m as usize {
        let s = a[ao + i];

        let mut j = 0usize;
        while j + 4 <= n {
            let mut t0 = y[yo + j] + s * x[xo + j];
            let mut t1 = y[yo + j + 1] + s * x[xo + j + 1];
            y[yo + j] = t0;
            y[yo + j + 1] = t1;
            t0 = y[yo + j + 2] + s * x[xo + j + 2];
            t1 = y[yo + j + 3] + s * x[xo + j + 3];
            y[yo + j + 2] = t0;
            y[yo + j + 3] = t1;
            j += 4;
        }

        while j < n {
            y[yo + j] += s * x[xo + j];
            j += 1;
        }
        xo += dx;
        yo += dy;
    }
}

/// `icvMatrAXPY3_64f( int m, int n, const double* x, int l, double* y,
/// double h )` (`cxsvd.cpp:94`).  `x` is `xv[1..]` with `x[-1]` at `xv[0]`
/// (a copy of the vector, which the C reads while it writes rows below it
/// in the same array); `y` starts at `y[yo]`, and `y[-1]` is `y[yo - 1]`.
fn icv_matr_axpy3_64f(m: i32, n: i32, xv: &[f64], l: usize, y: &mut [f64], mut yo: usize, h: f64) {
    let n = n.max(0) as usize;
    let x = |j: usize| xv[j + 1];
    for _ in 1..m {
        let mut s = 0f64;

        yo += l;

        let mut j = 0usize;
        while j + 4 <= n {
            s += x(j) * y[yo + j]
                + x(j + 1) * y[yo + j + 1]
                + x(j + 2) * y[yo + j + 2]
                + x(j + 3) * y[yo + j + 3];
            j += 4;
        }

        while j < n {
            s += x(j) * y[yo + j];
            j += 1;
        }

        s *= h;
        y[yo - 1] = s * xv[0];

        let mut j = 0usize;
        while j + 4 <= n {
            let mut t0 = y[yo + j] + s * x(j);
            let mut t1 = y[yo + j + 1] + s * x(j + 1);
            y[yo + j] = t0;
            y[yo + j + 1] = t1;
            t0 = y[yo + j + 2] + s * x(j + 2);
            t1 = y[yo + j + 3] + s * x(j + 3);
            y[yo + j + 2] = t0;
            y[yo + j + 3] = t1;
            j += 4;
        }

        while j < n {
            y[yo + j] += s * x(j);
            j += 1;
        }
    }
}

/// The `icvGivens_64f( n, x, y, c, s )` macro (`cxsvd.cpp:47`) on two rows
/// `xo`, `yo` of the same array.
fn icv_givens_64f(n: i32, d: &mut [f64], xo: usize, yo: usize, c: f64, s: f64) {
    for i in 0..n.max(0) as usize {
        let t0 = d[xo + i];
        let t1 = d[yo + i];
        d[xo + i] = t0 * c + t1 * s;
        d[yo + i] = -t0 * s + t1 * c;
    }
}

/// `pythag( double a, double b )` (`cxsvd.cpp:208`): accurate hypotenuse
/// calculation.
fn pythag(mut a: f64, mut b: f64) -> f64 {
    a = a.abs();
    b = b.abs();
    if a > b {
        b /= a;
        a *= (1. + b * b).sqrt();
    } else if b != 0. {
        a /= b;
        a = b * (1. + a * a).sqrt();
    }

    a
}

/// `MAX_ITERS`.
const MAX_ITERS: i32 = 30;

/// Where the C's `hv` points: into `U`, into `V`, or into `hv0` (offset 1).
#[derive(Clone, Copy)]
enum Hv {
    U(usize),
    V(usize),
    H0,
}

/// `icvSVD_64f( double* a, int lda, int m, int n, double* w, double* uT,
/// int lduT, int nu, double* vT, int ldvT, double* buffer )`
/// (`cxsvd.cpp:232`).  `u`/`v` are `None` for the C's null `uT`/`vT`.
#[allow(clippy::too_many_arguments)]
fn icv_svd_64f(
    a: &mut [f64],
    lda: usize,
    m: i32,
    n: i32,
    w: &mut [f64],
    mut u: Option<&mut [f64]>,
    ldu_t: usize,
    nu: i32,
    mut v: Option<&mut [f64]>,
    ldv_t: usize,
) {
    let nm = n as usize;
    let nv = n;
    let mut e = vec![0f64; nm + 1];
    let mut temp = vec![0f64; nm];
    let mut hv0 = vec![0f64; (m + 2) as usize];
    let mut ku0 = 0f64;
    let mut kv0 = 0f64;
    let mut anorm = 0f64;
    let mut scale;
    let mut h;
    let mut ao = 0usize; // `a`
    let mut ut = 0usize; // `uT - u0`
    let mut vt = 0usize; // `vT - v0`
    let mut w1 = 0usize;
    let mut e1 = 1usize;

    // `memset( w, 0, nm * sizeof( w[0] ))`; `e` is zero-filled.
    w[..nm].fill(0.);

    let mut m1 = m;
    let mut n1 = n;

    macro_rules! hvbuf {
        ($hv:expr) => {
            match $hv {
                Hv::U(o) => (&mut **u.as_mut().unwrap(), o),
                Hv::V(o) => (&mut **v.as_mut().unwrap(), o),
                Hv::H0 => (&mut hv0[..], 1usize),
            }
        };
    }

    // transform a to bi-diagonal form
    loop {
        if m1 == 0 {
            break;
        }

        scale = 0.;
        h = 0.;
        let update_u = u.is_some() && m1 > m - nu;
        let hv = if update_u { Hv::U(ut) } else { Hv::H0 };

        {
            let (hb, ho) = hvbuf!(hv);
            let mut a1 = ao;
            for j in 0..m1 as usize {
                let t = a[a1];
                hb[ho + j] = t;
                scale += t.abs();
                a1 += lda;
            }
        }

        if scale != 0. {
            let mut f = 1. / scale;
            let g;
            let mut s = 0f64;
            let (hb, ho) = hvbuf!(hv);

            for j in 0..m1 as usize {
                hb[ho + j] *= f;
                let t = hb[ho + j];
                s += t * t;
            }

            let mut gg = s.sqrt();
            f = hb[ho];
            if f >= 0. {
                gg = -gg;
            }
            g = gg;
            hb[ho] = f - g;
            h = 1. / (f * g - s);

            for t in temp[..n1 as usize].iter_mut() {
                *t = 0.;
            }

            // calc temp[0:n-i] = a[i:m,i:n]'*hv[0:m-i]
            icv_matr_axpy_64f(m1, n1 - 1, a, ao + 1, lda, hb, ho, &mut temp, 1, 0);
            for k in 1..n1 as usize {
                temp[k] *= h;
            }

            // modify a: a[i:m,i:n] = a[i:m,i:n] + hv[0:m-i]*temp[0:n-i]'
            icv_matr_axpy_64f(m1, n1 - 1, &temp, 1, 0, hb, ho, a, ao + 1, lda);
            w[w1] = g * scale;
        }
        w1 += 1;

        // store -2/(hv'*hv)
        if update_u {
            if m1 == m {
                ku0 = h;
            } else {
                let (hb, ho) = hvbuf!(hv);
                hb[ho - 1] = h;
            }
        }

        ao += 1;
        n1 -= 1;
        if v.is_some() {
            vt += ldv_t + 1;
        }

        if n1 == 0 {
            break;
        }

        scale = 0.;
        h = 0.;
        let update_v = v.is_some() && n1 > n - nv;

        let hv = if update_v { Hv::V(vt) } else { Hv::H0 };

        {
            let (hb, ho) = hvbuf!(hv);
            for j in 0..n1 as usize {
                let t = a[ao + j];
                hb[ho + j] = t;
                scale += t.abs();
            }
        }

        if scale != 0. {
            let mut f = 1. / scale;
            let mut s = 0f64;
            let (hb, ho) = hvbuf!(hv);

            for j in 0..n1 as usize {
                hb[ho + j] *= f;
                let t = hb[ho + j];
                s += t * t;
            }

            let mut g = s.sqrt();
            f = hb[ho];
            if f >= 0. {
                g = -g;
            }
            hb[ho] = f - g;
            h = 1. / (f * g - s);
            hb[ho - 1] = 0.;

            // update a[i:m:i+1:n] = a[i:m,i+1:n] + (a[i:m,i+1:n]*hv[0:m-i])*...
            let xv: Vec<f64> = hb[ho - 1..ho + n1 as usize].to_vec();
            icv_matr_axpy3_64f(m1, n1, &xv, lda, a, ao, h);

            e[e1] = g * scale;
        }
        e1 += 1;

        // store -2/(hv'*hv)
        if update_v {
            if n1 == n {
                kv0 = h;
            } else {
                let (hb, ho) = hvbuf!(hv);
                hb[ho - 1] = h;
            }
        }

        ao += lda;
        m1 -= 1;
        if u.is_some() {
            ut += ldu_t + 1;
        }
    }

    m1 -= (m1 != 0) as i32;
    n1 -= (n1 != 0) as i32;

    // accumulate left transformations
    if let Some(ud) = u.as_deref_mut() {
        m1 = m - m1;
        let mut uo = m1 as usize * ldu_t;
        for i in m1..nu {
            for k in m1..m {
                ud[uo + k as usize] = 0.;
            }
            ud[uo + i as usize] = 1.;
            uo += ldu_t;
        }

        let mut i = m1 - 1;
        while i >= 0 {
            let lh = nu - i;
            let l = m - i;

            let hvo = (ldu_t + 1) * i as usize;
            h = if i == 0 { ku0 } else { ud[hvo - 1] };

            assert!(h <= 0.);

            if h != 0. {
                let xv: Vec<f64> = ud[hvo..hvo + l as usize].to_vec();
                icv_matr_axpy3_64f(lh, l - 1, &xv, ldu_t, ud, hvo + 1, h);

                let s = ud[hvo] * h;
                for k in 0..l as usize {
                    ud[hvo + k] *= s;
                }
                ud[hvo] += 1.;
            } else {
                for j in 1..l as usize {
                    ud[hvo + j] = 0.;
                }
                for j in 1..lh as usize {
                    ud[hvo + j * ldu_t] = 0.;
                }
                ud[hvo] = 1.;
            }
            i -= 1;
        }
    }

    // accumulate right transformations
    if let Some(vd) = v.as_deref_mut() {
        n1 = n - n1;
        let mut vo = n1 as usize * ldv_t;
        for i in n1..nv {
            for k in n1..n {
                vd[vo + k as usize] = 0.;
            }
            vd[vo + i as usize] = 1.;
            vo += ldv_t;
        }

        let mut i = n1 - 1;
        while i >= 0 {
            let lh = nv - i;
            let l = n - i;
            let hvo = (ldv_t + 1) * i as usize;
            h = if i == 0 { kv0 } else { vd[hvo - 1] };

            assert!(h <= 0.);

            if h != 0. {
                let xv: Vec<f64> = vd[hvo..hvo + l as usize].to_vec();
                icv_matr_axpy3_64f(lh, l - 1, &xv, ldv_t, vd, hvo + 1, h);

                let s = vd[hvo] * h;
                for k in 0..l as usize {
                    vd[hvo + k] *= s;
                }
                vd[hvo] += 1.;
            } else {
                for j in 1..l as usize {
                    vd[hvo + j] = 0.;
                }
                for j in 1..lh as usize {
                    vd[hvo + j * ldv_t] = 0.;
                }
                vd[hvo] = 1.;
            }
            i -= 1;
        }
    }

    for i in 0..nm {
        let mut tnorm = w[i].abs();
        tnorm += e[i].abs();

        if anorm < tnorm {
            anorm = tnorm;
        }
    }

    anorm *= f64::EPSILON;

    // diagonalization of the bidiagonal form
    let mut k = nm as i32 - 1;
    while k >= 0 {
        let ku = k as usize;
        let mut z = 0f64;
        let mut iters = 0;

        loop {
            // do iterations
            let mut c;
            let mut s;
            let mut f;
            let mut g;
            let mut x;
            let mut y;
            let mut flag = 0;

            // test for splitting
            let mut l = k;
            while l >= 0 {
                if e[l as usize].abs() <= anorm {
                    flag = 1;
                    break;
                }
                assert!(l > 0);
                if w[(l - 1) as usize].abs() <= anorm {
                    break;
                }
                l -= 1;
            }

            if flag == 0 {
                c = 0.;
                s = 1.;

                let mut i = l;
                while i <= k {
                    let iu = i as usize;
                    f = s * e[iu];

                    e[iu] *= c;

                    if anorm + f.abs() == anorm {
                        break;
                    }

                    g = w[iu];
                    h = pythag(f, g);
                    w[iu] = h;
                    c = g / h;
                    s = -f / h;

                    if let Some(ud) = u.as_deref_mut() {
                        icv_givens_64f(m, ud, ldu_t * (l - 1) as usize, ldu_t * iu, c, s);
                    }
                    i += 1;
                }
            }

            z = w[ku];
            if l == k || {
                let hit = iters == MAX_ITERS;
                iters += 1;
                hit
            } {
                break;
            }

            // shift from bottom 2x2 minor
            let lu = l as usize;
            x = w[lu];
            y = w[ku - 1];
            g = e[ku - 1];
            h = e[ku];
            f = 0.5 * (((g + z) / h) * ((g - z) / y) + y / h - h / y);
            g = pythag(f, 1.);
            if f < 0. {
                g = -g;
            }
            f = x - (z / x) * z + (h / x) * (y / (f + g) - h);
            // next QR transformation
            c = 1.;
            s = 1.;

            for i in (lu + 1)..=ku {
                g = e[i];
                y = w[i];
                h = s * g;
                g *= c;
                z = pythag(f, h);
                e[i - 1] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = -x * s + g * c;
                h = y * s;
                y *= c;

                if let Some(vd) = v.as_deref_mut() {
                    icv_givens_64f(n, vd, ldv_t * (i - 1), ldv_t * i, c, s);
                }

                z = pythag(f, h);
                w[i - 1] = z;

                // rotation can be arbitrary if z == 0
                if z != 0. {
                    c = f / z;
                    s = h / z;
                }
                f = c * g + s * y;
                x = -s * g + c * y;

                if let Some(ud) = u.as_deref_mut() {
                    icv_givens_64f(m, ud, ldu_t * (i - 1), ldu_t * i, c, s);
                }
            }

            e[lu] = 0.;
            e[ku] = f;
            w[ku] = x;
        } // end of iteration loop

        if iters > MAX_ITERS {
            break;
        }

        if z < 0. {
            w[ku] = -z;
            if let Some(vd) = v.as_deref_mut() {
                for j in 0..n as usize {
                    vd[j + ku * ldv_t] = -vd[j + ku * ldv_t];
                }
            }
        }
        k -= 1;
    } // end of diagonalization loop

    // sort singular values and corresponding values
    for i in 0..nm {
        let mut k = i;
        for j in i + 1..nm {
            if w[k] < w[j] {
                k = j;
            }
        }

        if k != i {
            w.swap(i, k);

            if let Some(vd) = v.as_deref_mut() {
                for j in 0..n as usize {
                    vd.swap(j + ldv_t * k, j + ldv_t * i);
                }
            }

            if let Some(ud) = u.as_deref_mut() {
                for j in 0..m as usize {
                    ud.swap(j + ldu_t * k, j + ldu_t * i);
                }
            }
        }
    }
}

/// `icvSVBkSb_64f( int m, int n, const double* w, const double* uT, int
/// lduT, const double* vT, int ldvT, const double* b, int ldb, int nb,
/// double* x, int ldx, double* buffer )` (`cxsvd.cpp:1023`).
#[allow(clippy::too_many_arguments)]
fn icv_svbksb_64f(
    m: i32,
    n: i32,
    w: &[f64],
    u_t: &[f64],
    ldu_t: usize,
    v_t: &[f64],
    ldv_t: usize,
    b: Option<&[f64]>,
    ldb: usize,
    mut nb: i32,
    x: &mut [f64],
    ldx: usize,
) {
    let mut threshold = 0f64;
    let nm = m.min(n) as usize;
    let (mu, nu) = (m as usize, n as usize);

    if b.is_none() {
        nb = m;
    }
    let nbu = nb as usize;
    let mut buffer = vec![0f64; nbu.max(mu.max(nu))];

    for i in 0..nu {
        for k in 0..nbu {
            x[i * ldx + k] = 0.;
        }
    }

    for i in 0..nm {
        threshold += w[i];
    }
    threshold *= 2. * f64::EPSILON;

    // vT * inv(w) * uT * b
    let (mut uo, mut vo) = (0usize, 0usize);
    for i in 0..nm {
        let mut wi = w[i];

        if wi > threshold {
            wi = 1. / wi;

            if nb == 1 {
                let mut s = 0f64;
                if let Some(b) = b {
                    if ldb == 1 {
                        let mut j = 0usize;
                        while j + 4 <= mu {
                            s += u_t[uo + j] * b[j]
                                + u_t[uo + j + 1] * b[j + 1]
                                + u_t[uo + j + 2] * b[j + 2]
                                + u_t[uo + j + 3] * b[j + 3];
                            j += 4;
                        }
                        while j < mu {
                            s += u_t[uo + j] * b[j];
                            j += 1;
                        }
                    } else {
                        for j in 0..mu {
                            s += u_t[uo + j] * b[j * ldb];
                        }
                    }
                } else {
                    s = u_t[uo];
                }
                s *= wi;
                if ldx == 1 {
                    let mut j = 0usize;
                    while j + 4 <= nu {
                        let mut t0 = x[j] + s * v_t[vo + j];
                        let mut t1 = x[j + 1] + s * v_t[vo + j + 1];
                        x[j] = t0;
                        x[j + 1] = t1;
                        t0 = x[j + 2] + s * v_t[vo + j + 2];
                        t1 = x[j + 3] + s * v_t[vo + j + 3];
                        x[j + 2] = t0;
                        x[j + 3] = t1;
                        j += 4;
                    }

                    while j < nu {
                        x[j] += s * v_t[vo + j];
                        j += 1;
                    }
                } else {
                    for j in 0..nu {
                        x[j * ldx] += s * v_t[vo + j];
                    }
                }
            } else {
                if let Some(b) = b {
                    for t in buffer[..nbu].iter_mut() {
                        *t = 0.;
                    }
                    icv_matr_axpy_64f(m, nb, b, 0, ldb, u_t, uo, &mut buffer, 0, 0);
                    for j in 0..nbu {
                        buffer[j] *= wi;
                    }
                } else {
                    for j in 0..nbu {
                        buffer[j] = u_t[uo + j] * wi;
                    }
                }
                let bufc = buffer.clone();
                icv_matr_axpy_64f(n, nb, &bufc, 0, 0, v_t, vo, x, 0, ldx);
            }
        }
        uo += ldu_t;
        vo += ldv_t;
    }
}

/// `cvSVD( CvArr* aarr, CvArr* warr, CvArr* uarr, CvArr* varr, int flags )`
/// (`cxsvd.cpp:1212`) for `CV_64FC1`, `a` with at least as many rows as
/// columns, `w` a continuous vector of `n` elements (so `tw` is `w`'s own
/// data), and `flags == CV_SVD_U_T + CV_SVD_V_T` (no temporary `U`, no
/// transposition afterwards): the form `cvSolve` uses.
pub fn cv_svd(a: CvMatRef<'_>, w: &mut CvMat, u: &mut CvMat, v: &mut CvMat, flags: i32) {
    assert!(
        a.rows >= a.cols && flags == CV_SVD_U_T + CV_SVD_V_T,
        "cvSVD: RAPTOR reaches only the cvSolve form"
    );
    let m = a.rows;
    let n = a.cols;
    let w_rows = w.rows;
    let w_cols = w.cols;
    let w_is_mat = w_cols > 1 && w_rows > 1;
    assert!(!w_is_mat && cv_is_mat_cont(w.type_) && w_cols + w_rows - 1 == n);

    // U: `u_rows = u->cols; u_cols = u->rows` (CV_SVD_U_T)
    let u_rows = u.cols;
    let u_cols = u.rows;
    if u_rows != m || (u_cols != m && u_cols != n) {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "cvSVD",
            "U matrix has unappropriate size",
            "cxsvd.cpp",
            1295,
        );
    }
    // V: `v_rows = v->cols; v_cols = v->rows` (CV_SVD_V_T)
    if v.cols != n || v.rows != n {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "cvSVD",
            "V matrix has unappropriate size",
            "cxsvd.cpp",
            1333,
        );
    }

    // `cvInitMatHeader( &tmat, m, n, type, buffer + a_buf_offset*pix_size )`
    // and `cvCopy( a, &tmat )`: a continuous copy of `a`.
    let mut tmat = vec![0f64; (m * n) as usize];
    for r in 0..m {
        for c in 0..n {
            tmat[(r * n + c) as usize] = a.elem(r, c);
        }
    }
    let lda = if m == 1 { 0 } else { n as usize };

    let ldu_t = (u.step / 8) as usize;
    let ldv_t = (v.step / 8) as usize;
    icv_svd_64f(
        &mut tmat,
        lda,
        m,
        n,
        &mut w.data,
        Some(&mut u.data),
        ldu_t,
        u_cols,
        Some(&mut v.data),
        ldv_t,
    );
}

/// `cvSVBkSb( const CvArr* warr, const CvArr* uarr, const CvArr* varr,
/// const CvArr* barr, CvArr* xarr, int flags )` (`cxsvd.cpp:1450`) for
/// `CV_64FC1` with `flags == CV_SVD_U_T + CV_SVD_V_T` and a continuous
/// vector `w`, as `cvSolve` calls it.
pub fn cv_svbksb(
    w: &CvMat,
    u: &CvMat,
    v: &CvMat,
    b: Option<CvMatRef<'_>>,
    x: &mut CvMat,
    flags: i32,
) {
    assert!(flags == CV_SVD_U_T + CV_SVD_V_T);
    let u_rows = u.cols;
    let u_cols = u.rows;
    let v_rows = v.cols;
    let v_cols = v.rows;

    let m = u_rows;
    let n = v_rows;
    let nm = n.min(m);

    if (u_rows != u_cols && v_rows != v_cols) || x.rows != v_rows {
        cv_error(
            CV_STS_BAD_SIZE,
            "cvSVBkSb",
            "V or U matrix must be square",
            "cxsvd.cpp",
            1526,
        );
    }

    assert!((w.rows == 1 || w.cols == 1) && w.rows + w.cols - 1 == nm && cv_is_mat_cont(w.type_));

    if let Some(b) = b {
        if b.cols != x.cols || b.rows != m {
            cv_error(
                CV_STS_UNMATCHED_SIZES,
                "cvSVBkSb",
                "b matrix must have (m x x->cols) size",
                "cxsvd.cpp",
                1554,
            );
        }
    }

    let ldx = (x.step / 8) as usize;
    let (bd, ldb, nb) = match b {
        Some(b) => (Some(b.data), (b.step / 8) as usize, b.cols),
        None => (None, 0, 0),
    };
    icv_svbksb_64f(
        m,
        n,
        &w.data,
        &u.data,
        (u.step / 8) as usize,
        &v.data,
        (v.step / 8) as usize,
        bd,
        ldb,
        nb,
        &mut x.data,
        ldx,
    );
}
