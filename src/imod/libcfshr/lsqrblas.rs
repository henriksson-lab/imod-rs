//! Translation of `IMOD/libcfshr/lsqrblas.c` and its local `cblas.h` offsets.
#![allow(dead_code)]

/// Original `cblas_daxpy` (`lsqrblas.c:48`).
pub unsafe fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, inc_x: i32, y: *mut f64, inc_y: i32) {
    if n <= 0 || alpha == 0. {
        return;
    }
    if inc_x == 1 && inc_y == 1 {
        let m = n % 4;
        let mut i = 0;
        while i < m {
            unsafe { *y.add(i as usize) += alpha * *x.add(i as usize) };
            i += 1;
        }
        while i + 3 < n {
            unsafe {
                *y.add(i as usize) += alpha * *x.add(i as usize);
                *y.add((i + 1) as usize) += alpha * *x.add((i + 1) as usize);
                *y.add((i + 2) as usize) += alpha * *x.add((i + 2) as usize);
                *y.add((i + 3) as usize) += alpha * *x.add((i + 3) as usize);
            }
            i += 4;
        }
    } else {
        let mut ix = if inc_x > 0 { 0 } else { (n - 1) * -inc_x };
        let mut iy = if inc_y > 0 { 0 } else { (n - 1) * -inc_y };
        for _ in 0..n {
            unsafe { *y.add(iy as usize) += alpha * *x.add(ix as usize) };
            ix += inc_x;
            iy += inc_y;
        }
    }
}

/// Original `cblas_dcopy` (`lsqrblas.c:91`).
pub unsafe fn cblas_dcopy(n: i32, x: *const f64, inc_x: i32, y: *mut f64, inc_y: i32) {
    let mut ix = if inc_x > 0 { 0 } else { (n - 1) * -inc_x };
    let mut iy = if inc_y > 0 { 0 } else { (n - 1) * -inc_y };
    for _ in 0..n {
        unsafe { *y.add(iy as usize) = *x.add(ix as usize) };
        ix += inc_x;
        iy += inc_y;
    }
}

/// Original `cblas_ddot` (`lsqrblas.c:115`).
pub unsafe fn cblas_ddot(n: i32, x: *const f64, inc_x: i32, y: *const f64, inc_y: i32) -> f64 {
    let mut result = 0.;
    let mut ix = if inc_x > 0 { 0 } else { (n - 1) * -inc_x };
    let mut iy = if inc_y > 0 { 0 } else { (n - 1) * -inc_y };
    for _ in 0..n {
        unsafe { result += *x.add(ix as usize) * *y.add(iy as usize) };
        ix += inc_x;
        iy += inc_y;
    }
    result
}

/// Original `cblas_dnrm2` (`lsqrblas.c:139`).
pub unsafe fn cblas_dnrm2(n: i32, x: *const f64, inc_x: i32) -> f64 {
    let mut scale = 0.;
    let mut ssq = 1.;
    let mut ix = 0;
    if n <= 0 || inc_x <= 0 {
        return 0.;
    } else if n == 1 {
        return unsafe { (*x).abs() };
    }
    for _ in 0..n {
        let value = unsafe { *x.add(ix as usize) };
        if value != 0. {
            let absolute = value.abs();
            if scale < absolute {
                ssq = 1. + ssq * (scale / absolute) * (scale / absolute);
                scale = absolute;
            } else {
                ssq += (absolute / scale) * (absolute / scale);
            }
        }
        ix += inc_x;
    }
    scale * ssq.sqrt()
}

/// Original `cblas_dscal` (`lsqrblas.c:178`).
pub unsafe fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, inc_x: i32) {
    if inc_x <= 0 {
        return;
    }
    let mut ix = if inc_x > 0 { 0 } else { (n - 1) * -inc_x };
    for _ in 0..n {
        unsafe { *x.add(ix as usize) *= alpha };
        ix += inc_x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_blas_operations_and_strides() {
        unsafe {
            let x = [1., 2., 3., 4.];
            let mut y = [10., 20., 30., 40.];
            cblas_daxpy(4, 2., x.as_ptr(), 1, y.as_mut_ptr(), 1);
            assert_eq!(y, [12., 24., 36., 48.]);
            cblas_dscal(2, 0.5, y.as_mut_ptr(), 2);
            assert_eq!(y, [6., 24., 18., 48.]);
            let mut copied = [0.; 2];
            cblas_dcopy(2, x.as_ptr(), -2, copied.as_mut_ptr(), 1);
            assert_eq!(copied, [3., 1.]);
            assert_eq!(cblas_ddot(2, x.as_ptr(), -2, copied.as_ptr(), 1), 10.);
            assert_eq!(cblas_dnrm2(2, x.as_ptr(), 2), 10_f64.sqrt());
            assert_eq!(cblas_dnrm2(2, x.as_ptr(), -1), 0.);
        }
    }
}
