//! Translation of `IMOD/libcfshr/lsqrblas.c` and its local `cblas.h` offsets.
//!
//! `cblas.h:38` defines `OFFSET(N, incX)` as
//! `((incX) > 0 ? 0 : ((N) - 1) * (-(incX)))`; it is a macro, so it is written
//! out at each of the five places the source expands it rather than becoming a
//! function this unit does not have.

/// Original `cblas_daxpy` (`lsqrblas.c:48`).
pub fn cblas_daxpy(n: i32, alpha: f64, x: &[f64], inc_x: i32, y: &mut [f64], inc_y: i32) {
    if n <= 0 {
        return;
    }
    if alpha == 0.0 {
        return;
    }

    if inc_x == 1 && inc_y == 1 {
        let m = n % 4;

        let mut i = 0;
        while i < m {
            y[i as usize] += alpha * x[i as usize];
            i += 1;
        }

        i = m;
        while i + 3 < n {
            y[i as usize] += alpha * x[i as usize];
            y[(i + 1) as usize] += alpha * x[(i + 1) as usize];
            y[(i + 2) as usize] += alpha * x[(i + 2) as usize];
            y[(i + 3) as usize] += alpha * x[(i + 3) as usize];
            i += 4;
        }
    } else {
        let mut ix = if inc_x > 0 { 0 } else { (n - 1) * -inc_x };
        let mut iy = if inc_y > 0 { 0 } else { (n - 1) * -inc_y };

        for _ in 0..n {
            y[iy as usize] += alpha * x[ix as usize];
            ix += inc_x;
            iy += inc_y;
        }
    }
}

/// Original `cblas_dcopy` (`lsqrblas.c:91`).
pub fn cblas_dcopy(n: i32, x: &[f64], inc_x: i32, y: &mut [f64], inc_y: i32) {
    let mut ix = if inc_x > 0 { 0 } else { (n - 1) * -inc_x };
    let mut iy = if inc_y > 0 { 0 } else { (n - 1) * -inc_y };

    for _ in 0..n {
        y[iy as usize] = x[ix as usize];
        ix += inc_x;
        iy += inc_y;
    }
}

/// Original `cblas_ddot` (`lsqrblas.c:115`).
pub fn cblas_ddot(n: i32, x: &[f64], inc_x: i32, y: &[f64], inc_y: i32) -> f64 {
    let mut r = 0.0;
    let mut ix = if inc_x > 0 { 0 } else { (n - 1) * -inc_x };
    let mut iy = if inc_y > 0 { 0 } else { (n - 1) * -inc_y };

    for _ in 0..n {
        r += x[ix as usize] * y[iy as usize];
        ix += inc_x;
        iy += inc_y;
    }

    r
}

/// Original `cblas_dnrm2` (`lsqrblas.c:139`).
pub fn cblas_dnrm2(n: i32, x: &[f64], inc_x: i32) -> f64 {
    let mut scale = 0.0;
    let mut ssq = 1.0;
    let mut ix = 0;

    if n <= 0 || inc_x <= 0 {
        return 0.;
    } else if n == 1 {
        return x[0].abs();
    }

    for _ in 0..n {
        let value = x[ix as usize];

        if value != 0.0 {
            let ax = value.abs();

            if scale < ax {
                ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
                scale = ax;
            } else {
                ssq += (ax / scale) * (ax / scale);
            }
        }

        ix += inc_x;
    }

    scale * ssq.sqrt()
}

/// Original `cblas_dscal` (`lsqrblas.c:178`).
pub fn cblas_dscal(n: i32, alpha: f64, x: &mut [f64], inc_x: i32) {
    if inc_x <= 0 {
        return;
    }

    let mut ix = if inc_x > 0 { 0 } else { (n - 1) * -inc_x };

    for _ in 0..n {
        x[ix as usize] *= alpha;
        ix += inc_x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_blas_operations_and_strides() {
        let x = [1., 2., 3., 4.];
        let mut y = [10., 20., 30., 40.];
        cblas_daxpy(4, 2., &x, 1, &mut y, 1);
        assert_eq!(y, [12., 24., 36., 48.]);
        cblas_dscal(2, 0.5, &mut y, 2);
        assert_eq!(y, [6., 24., 18., 48.]);
        let mut copied = [0.; 2];
        cblas_dcopy(2, &x, -2, &mut copied, 1);
        assert_eq!(copied, [3., 1.]);
        assert_eq!(cblas_ddot(2, &x, -2, &copied, 1), 10.);
        assert_eq!(cblas_dnrm2(2, &x, 2), 10_f64.sqrt());
        assert_eq!(cblas_dnrm2(2, &x, -1), 0.);
    }
}
