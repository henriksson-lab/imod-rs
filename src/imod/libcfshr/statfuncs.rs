//! Translation of `IMOD/libcfshr/statfuncs.c` and its `cfsemshare.h` APIs.

static mut VAL_SET: i32 = -1;
static mut LAST_SEED: i32 = 0;
static mut SAVED_VAL: f32 = 0.;

/// Original `tValue` (`statfuncs.c:42`).
pub fn t_value(signif: f64, ndf: i32) -> f64 {
    if signif >= 1. {
        return 1.0e10;
    }
    if signif <= 0.5 {
        return 0.;
    }
    let mut x1 = 0.;
    let mut y1 = 1. - incomp_beta(0.5 * ndf as f64, 0.5, ndf as f64 / ndf as f64) / 2.;
    let mut x2 = 0.;
    let mut y2 = y1;
    while x1 < 1.0e4 {
        x2 = if x1 > 1. { x1 + 0.5 } else { x1 + 0.1 };
        y2 = 1. - incomp_beta(0.5 * ndf as f64, 0.5, ndf as f64 / (ndf as f64 + x2 * x2)) / 2.;
        if y2 == signif {
            return x2;
        }
        if y2 > signif {
            break;
        }
        x1 = x2;
        y1 = y2;
    }
    if y2 < signif {
        return 1.0e10;
    }
    let mut xnew = x2;
    for _ in 0..200 {
        let slope = (y2 - y1) / (x2 - x1);
        xnew = x2 - (y2 - signif) / slope;
        let ynew = 1.
            - incomp_beta(
                0.5 * ndf as f64,
                0.5,
                ndf as f64 / (ndf as f64 + xnew * xnew),
            ) / 2.;
        if (ynew - signif).abs() < 1.0e-7 {
            return xnew;
        }
        if ynew < signif {
            x1 = xnew;
            y1 = ynew;
        } else {
            x2 = xnew;
            y2 = ynew;
        }
        if x2 - x1 < 5.0e-7 {
            break;
        }
    }
    xnew
}

/// Original `dtvalue` (`statfuncs.c:94`).
pub unsafe fn dtvalue(signif: *mut f64, ndf: *mut i32) -> f64 {
    unsafe { t_value(*signif, *ndf) }
}

/// Original `fValue` (`statfuncs.c:108`).
pub fn f_value(signif: f64, ndf1: i32, ndf2: i32) -> f64 {
    if signif >= 1. {
        return 1.0e10;
    }
    if signif <= 0. {
        return 0.;
    }
    let mut x1 = 0.;
    let mut y1 = 1. - incomp_beta(0.5 * ndf2 as f64, 0.5 * ndf1 as f64, 1.);
    let mut x2 = 0.;
    let mut y2 = y1;
    while x1 < 1.0e4 {
        x2 = if x1 > 1. { x1 + 0.5 } else { x1 + 0.1 };
        y2 = 1.
            - incomp_beta(
                0.5 * ndf2 as f64,
                0.5 * ndf1 as f64,
                ndf2 as f64 / (ndf2 as f64 + ndf1 as f64 * x2),
            );
        if y2 == signif {
            return x2;
        }
        if y2 > signif {
            break;
        }
        x1 = x2;
        y1 = y2;
    }
    if y2 < signif {
        return 1.0e10;
    }
    let mut xnew = x2;
    for _ in 0..200 {
        let slope = (y2 - y1) / (x2 - x1);
        xnew = x2 - (y2 - signif) / slope;
        let ynew = 1.
            - incomp_beta(
                0.5 * ndf2 as f64,
                0.5 * ndf1 as f64,
                ndf2 as f64 / (ndf2 as f64 + ndf1 as f64 * xnew),
            );
        if (ynew - signif).abs() < 1.0e-7 {
            return xnew;
        }
        if ynew < signif {
            x1 = xnew;
            y1 = ynew;
        } else {
            x2 = xnew;
            y2 = ynew;
        }
        if x2 - x1 < 5.0e-7 {
            break;
        }
    }
    xnew
}

/// Original `dfvalue` (`statfuncs.c:157`).
pub unsafe fn dfvalue(signif: *mut f64, ndf1: *mut i32, ndf2: *mut i32) -> f64 {
    unsafe { f_value(*signif, *ndf1, *ndf2) }
}

/// Original `errFunc` (`statfuncs.c:185`).
pub fn err_func(x: f64) -> f64 {
    let x2 = x * x;
    if x.abs() < 3.5 {
        let mut er = 1.;
        let mut r = 1.;
        for k in 1..=50 {
            r = r * x2 / (k as f64 + 0.5);
            er += r;
            if r.abs() <= er.abs() * 1.0e-15 {
                break;
            }
        }
        2. / std::f64::consts::PI.sqrt() * x * (-x2).exp() * er
    } else {
        let mut er = 1.;
        let mut r = 1.;
        for k in 1..=12 {
            r = -r * (k as f64 - 0.5) / x2;
            er += r;
        }
        let mut result = 1. - (-x2).exp() / (x.abs() * std::f64::consts::PI.sqrt()) * er;
        if x < 0. {
            result = -result;
        }
        result
    }
}

/// Original `errfunc` (`statfuncs.c:220`).
pub unsafe fn errfunc(x: *mut f64) -> f64 {
    unsafe { err_func(*x) }
}

/// Original `incompBeta` (`statfuncs.c:237`).
pub fn incomp_beta(a: f64, b: f64, x: f64) -> f64 {
    let s0 = (a + 1.) / (a + b + 2.);
    let bt = beta_func(a, b);
    if x <= s0 {
        let mut dk = [0.; 51];
        for k in 1..=20 {
            dk[2 * k - 1] =
                k as f64 * (b - k as f64) * x / (a + 2. * k as f64 - 1.) / (a + 2. * k as f64);
        }
        for k in 0..=20 {
            dk[2 * k] = -(a + k as f64) * (a + b + k as f64) * x
                / (a + 2. * k as f64)
                / (a + 2. * k as f64 + 1.);
        }
        let mut t1 = 0.;
        for k in (1..=20).rev() {
            t1 = dk[k - 1] / (1. + t1);
        }
        x.powf(a) * (1. - x).powf(b) / (a * bt) / (1. + t1)
    } else {
        let mut fk = [0.; 51];
        for k in 1..=20 {
            fk[2 * k - 1] = k as f64 * (a - k as f64) * (1. - x)
                / (b + 2. * k as f64 - 1.)
                / (b + 2. * k as f64);
        }
        for k in 0..=20 {
            fk[2 * k] = -(b + k as f64) * (a + b + k as f64) * (1. - x)
                / (b + 2. * k as f64)
                / (b + 2. * k as f64 + 1.);
        }
        let mut t2 = 0.;
        for k in (1..=20).rev() {
            t2 = fk[k - 1] / (1. + t2);
        }
        1. - x.powf(a) * (1. - x).powf(b) / (b * bt) / (1. + t2)
    }
}

/// Original `incompbeta` (`statfuncs.c:270`).
pub unsafe fn incompbeta(a: *mut f64, b: *mut f64, x: *mut f64) -> f64 {
    unsafe { incomp_beta(*a, *b, *x) }
}

/// Original `betaFunc` (`statfuncs.c:277`).
pub fn beta_func(p: f64, q: f64) -> f64 {
    (ln_gamma(p) + ln_gamma(q) - ln_gamma(p + q)).exp()
}

/// Original `lnGamma` (`statfuncs.c:293`).
pub fn ln_gamma(x: f64) -> f64 {
    const A: [f64; 10] = [
        8.333333333333333e-2,
        -2.777777777777778e-3,
        7.936507936507937e-4,
        -5.952380952380952e-4,
        8.417508417508418e-4,
        -1.917526917526918e-3,
        6.410256410256410e-3,
        -2.955065359477124e-2,
        1.796443723688307e-1,
        -1.39243221690590,
    ];
    if x == 1. || x == 2. {
        return 0.;
    }
    let mut x0 = x;
    let mut n = 0;
    if x <= 7. {
        n = (7. - x) as i32;
        x0 = x + n as f64;
    }
    let x2 = 1. / (x0 * x0);
    let mut gl0 = A[9];
    for k in (1..=9).rev() {
        gl0 = gl0 * x2 + A[k - 1];
    }
    let mut gl = gl0 / x0 + 0.5 * (2. * std::f64::consts::PI).ln() + (x0 - 0.5) * x0.ln() - x0;
    if x <= 7. {
        for _ in 1..=n {
            gl -= (x0 - 1.).ln();
            x0 -= 1.;
        }
    }
    gl
}

/// Original `gaussianDeviate` (`statfuncs.c:348`).
pub unsafe fn gaussian_deviate(seed: i32) -> f32 {
    if unsafe { VAL_SET < 0 || seed != LAST_SEED } {
        let use_seed = if seed > 0 {
            seed as u32
        } else {
            unsafe { libc::time(core::ptr::null_mut()) as u32 }
        };
        unsafe {
            libc::srand(use_seed);
            VAL_SET = 0;
            LAST_SEED = seed;
        }
    }
    if unsafe { VAL_SET > 0 } {
        unsafe {
            VAL_SET = 0;
            return SAVED_VAL;
        }
    }
    let mut rad_sq = 2.;
    let (mut val1, mut val2);
    loop {
        val1 = 2. * unsafe { libc::rand() } as f32 / 2_147_483_647. - 1.;
        val2 = 2. * unsafe { libc::rand() } as f32 / 2_147_483_647. - 1.;
        rad_sq = val1 * val1 + val2 * val2;
        if rad_sq < 1. && rad_sq > 0. {
            break;
        }
    }
    let fac = (-2. * rad_sq.ln() / rad_sq).sqrt();
    unsafe {
        SAVED_VAL = val2 * fac;
        VAL_SET = 1;
    }
    val1 * fac
}

/// Original `gaussiandeviate` (`statfuncs.c:379`).
pub unsafe fn gaussiandeviate(value: *mut f32, seed: *mut i32) {
    unsafe {
        *value = gaussian_deviate(*seed);
    }
}

/// Original `twoGaussianDeviates` (`statfuncs.c:391`).
pub unsafe fn two_gaussian_deviates(value1: *mut f32, value2: *mut f32, pseudo: *mut i32) {
    if unsafe { *pseudo == 0 } {
        unsafe {
            *pseudo = libc::time(core::ptr::null_mut()) as i32 & 0xFFFFF;
        }
    }
    let mut rad_sq = 2.;
    let (mut val1, mut val2);
    loop {
        unsafe {
            *pseudo = 197 * (*pseudo + 1) & 0xFFFFF;
            val1 = 2. * *pseudo as f32 / 0xFFFFF as f32 - 1.;
            *pseudo = 197 * (*pseudo + 1) & 0xFFFFF;
            val2 = 2. * *pseudo as f32 / 0xFFFFF as f32 - 1.;
        }
        rad_sq = val1 * val1 + val2 * val2;
        if rad_sq < 1. && rad_sq > 0. {
            break;
        }
    }
    let fac = (-2. * rad_sq.ln() / rad_sq).sqrt();
    unsafe {
        *value1 = val1 * fac;
        *value2 = val2 * fac;
    }
}

/// Original `twogaussiandeviates` (`statfuncs.c:416`).
pub unsafe fn twogaussiandeviates(value1: *mut f32, value2: *mut f32, pseudo: *mut i32) {
    unsafe {
        two_gaussian_deviates(value1, value2, pseudo);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn special_functions_and_distribution_limits_match_known_values() {
        assert!((err_func(1.) - 0.84270079).abs() < 1.0e-7);
        assert!((incomp_beta(2., 2., 0.5) - 0.5).abs() < 1.0e-12);
        assert!((beta_func(2., 3.) - 1. / 12.).abs() < 1.0e-12);
        assert!((ln_gamma(5.) - 24_f64.ln()).abs() < 1.0e-11);
        assert!((t_value(0.975, 10) - 2.22814).abs() < 2.0e-4);
        assert!((f_value(0.95, 5, 10) - 3.32583).abs() < 3.0e-4);
        assert_eq!(t_value(0.5, 10), 0.);
        assert_eq!(f_value(0., 5, 10), 0.);
    }
    #[test]
    fn gaussian_generators_are_seeded_and_finite() {
        unsafe {
            let first = gaussian_deviate(12345);
            let second = gaussian_deviate(12345);
            gaussian_deviate(7);
            let repeated = gaussian_deviate(12345);
            assert!(first.is_finite() && second.is_finite());
            assert_eq!(first, repeated);
            let mut one = 0.;
            let mut two = 0.;
            let mut pseudo = 123;
            two_gaussian_deviates(&mut one, &mut two, &mut pseudo);
            assert!(one.is_finite() && two.is_finite() && pseudo != 123);
        }
    }
}
