//! Translation of `IMOD/libcfshr/filtxcorr.c`.
//!
//! The public routines retain the C source's raw-pointer contracts.  This is
//! intentional: callers pass padded FFT lines and aliased work buffers.
#![allow(dead_code, unused_variables)]

use core::ffi::c_char;
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use super::robuststat::{rs_set_sort_index_offset, rs_sort_indexed_floats};

// C file-scope state for the peak-finder API (`filtxcorr.c:443-450`).
static S_APPLY_LIMITS: AtomicI32 = AtomicI32::new(0);
static S_LIMIT_XLO: AtomicI32 = AtomicI32::new(0);
static S_LIMIT_XHI: AtomicI32 = AtomicI32::new(0);
static S_LIMIT_YLO: AtomicI32 = AtomicI32::new(0);
static S_LIMIT_YHI: AtomicI32 = AtomicI32::new(0);
static S_LIMIT_ANGLE: AtomicU32 = AtomicU32::new(0);
static S_PEAK_FIND_ERROR: AtomicI32 = AtomicI32::new(0);
static S_WILL_CHECK_ERROR: AtomicI32 = AtomicI32::new(0);

/// `niceFrame` (`filtxcorr.c:94`).
pub fn nice_frame(num: i32, idnum: i32, limit: i32) -> i32 {
    let mut numin = 2 * ((num + 1) / 2);
    loop {
        let mut numtmp = numin;
        for ifac in 2..=limit {
            while numtmp % ifac == 0 {
                numtmp /= ifac;
            }
        }
        if numtmp <= 1 {
            return numin;
        }
        numin += idnum;
    }
}
pub unsafe fn niceframe(num: *const i32, idnum: *const i32, limit: *const i32) -> i32 {
    unsafe { nice_frame(*num, *idnum, *limit) }
}

/// `XCorrSetCTFnoScl` (`filtxcorr.c:159`).
pub unsafe fn xcorr_set_ctf_no_scl(
    sigma1: f32,
    sigma2: f32,
    radius1: f32,
    radius2: f32,
    ctf: *mut f32,
    nx: i32,
    ny: i32,
    delta: *mut f32,
    nsize_out: *mut i32,
) {
    unsafe {
        *delta = 0.;
        if sigma1 == 0. && sigma2 == 0. {
            return;
        }
        let mut nsize = (2 * nx).max(2 * ny).max(1024).min(8192);
        let asize = nsize as f32;
        nsize += 1;
        let alpha = if sigma1.abs() > 1.0e-6 {
            -0.5 / (sigma1 * sigma1)
        } else {
            0.
        };
        let beta1 = if sigma2.abs() > 1.0e-6 {
            -0.5 / (sigma2 * sigma2)
        } else {
            0.
        };
        *delta = 1. / (0.71 * asize);
        let (radius1p, radius1n) = if radius1 >= 0. {
            (radius1, 0.)
        } else {
            (0., -radius1)
        };
        let mut delmax: f32 = 0.;
        if sigma1 < -1.0e-6 {
            for j in 0..nsize {
                let s = j as f32 * *delta;
                delmax = delmax.max(s * s * (alpha * s * s).exp());
            }
        }
        for j in 0..nsize {
            let s = j as f32 * *delta;
            let mut value = if s < radius1p {
                (beta1 * (s - radius1p) * (s - radius1p)).exp()
            } else if s > radius2 {
                (beta1 * (s - radius2) * (s - radius2)).exp()
            } else {
                1.
            };
            if sigma2 < -1.0e-6 {
                value = 1. - value;
            }
            if sigma1 > 1.0e-6 {
                value = if s < radius1n {
                    0.
                } else {
                    value * (1. - (alpha * (s - radius1n) * (s - radius1n)).exp())
                };
            } else if sigma1 < -1.0e-6 {
                value *= s * s * (alpha * s * s).exp() / delmax;
            }
            *ctf.add(j as usize) = if value < 1.0e-6 { 0. } else { value };
        }
        *nsize_out = nsize;
    }
}
pub unsafe fn xcorr_set_ctf(
    sigma1: f32,
    sigma2: f32,
    radius1: f32,
    radius2: f32,
    ctf: *mut f32,
    nx: i32,
    ny: i32,
    delta: *mut f32,
) {
    unsafe {
        let mut nsize = 0;
        xcorr_set_ctf_no_scl(
            sigma1, sigma2, radius1, radius2, ctf, nx, ny, delta, &mut nsize,
        );
        if *delta == 0. {
            return;
        }
        let mut sum = 0.;
        for j in 1..nsize {
            sum += *ctf.add(j as usize);
        }
        for j in 1..nsize {
            *ctf.add(j as usize) *= (nsize - 1) as f32 / sum;
        }
    }
}
pub unsafe fn setctfwsr(
    s1: *const f32,
    s2: *const f32,
    r1: *const f32,
    r2: *const f32,
    ctf: *mut f32,
    nx: *const i32,
    ny: *const i32,
    delta: *mut f32,
) {
    unsafe { xcorr_set_ctf(*s1, *s2, *r1, *r2, ctf, *nx, *ny, delta) }
}
pub unsafe fn setctfnoscl(
    s1: *const f32,
    s2: *const f32,
    r1: *const f32,
    r2: *const f32,
    ctf: *mut f32,
    nx: *const i32,
    ny: *const i32,
    delta: *mut f32,
    nsize: *mut i32,
) {
    unsafe { xcorr_set_ctf_no_scl(*s1, *s2, *r1, *r2, ctf, *nx, *ny, delta, nsize) }
}

pub unsafe fn dose_filter_value(
    _start: f32,
    end: f32,
    frequency: f32,
    mut afac: f32,
    mut bfac: f32,
    mut cfac: f32,
    scale: f32,
    atten: *mut f32,
) {
    unsafe {
        if afac == 0. {
            afac = 0.24499
        };
        if bfac == 0. {
            bfac = -1.6649
        };
        if cfac == 0. {
            cfac = 2.8141
        };
        *atten = 1.;
        if frequency != 0. {
            let critical = scale * (afac * frequency.powf(bfac) + cfac);
            *atten = (-0.5 * end / critical).exp();
        }
    }
}
pub unsafe fn dose_weight_filter(
    start: f32,
    end: f32,
    pixel: f32,
    a: f32,
    b: f32,
    c: f32,
    scale: f32,
    ctf: *mut f32,
    n: i32,
    max: f32,
    delta: *mut f32,
) {
    unsafe {
        *delta = max / (n - 1) as f32;
        for i in 0..n {
            dose_filter_value(
                start,
                end,
                *delta * i as f32 / pixel,
                a,
                b,
                c,
                scale,
                ctf.add(i as usize),
            );
        }
    }
}
pub unsafe fn doseweightfilter(
    start: *const f32,
    end: *const f32,
    pixel: *const f32,
    a: *const f32,
    b: *const f32,
    c: *const f32,
    scale: *const f32,
    ctf: *mut f32,
    n: *const i32,
    max: *const f32,
    delta: *mut f32,
) {
    unsafe {
        dose_weight_filter(
            *start, *end, *pixel, *a, *b, *c, *scale, ctf, *n, *max, delta,
        )
    }
}

/// `XCorrMeanZero` (`filtxcorr.c:415`).
pub unsafe fn xcorr_mean_zero(array: *mut f32, nxdim: i32, nx: i32, ny: i32) {
    unsafe {
        let mut sum = 0.;
        for y in 0..ny {
            for x in 0..nx {
                sum += *array.add((x + y * nxdim) as usize);
            }
        }
        let mean = sum / (nx * ny) as f32;
        for y in 0..ny {
            for x in 0..nx {
                let p = array.add((x + y * nxdim) as usize);
                *p -= mean;
            }
        }
    }
}
pub unsafe fn meanzero(a: *mut f32, d: *const i32, x: *const i32, y: *const i32) {
    unsafe { xcorr_mean_zero(a, *d, *x, *y) }
}

pub fn parabolic_fit_position(y1: f32, y2: f32, y3: f32) -> f64 {
    let denom = (2.0 * (y1 + y3 - 2. * y2)) as f64;
    let mut cx = if denom.abs() > (1.0e-2 * (y1 - y3)).abs() as f64 {
        (y1 - y3) as f64 / denom
    } else {
        0.
    };
    cx = cx.clamp(-0.5, 0.5);
    cx
}
pub unsafe fn parabolicfitposition(a: *const f32, b: *const f32, c: *const f32) -> f64 {
    unsafe { parabolic_fit_position(*a, *b, *c) }
}
pub unsafe fn conjugate_product(a: *mut f32, b: *const f32, nx: i32, ny: i32) {
    unsafe {
        for j in (0..ny * (nx + 2)).step_by(2) {
            let ar = *a.add(j as usize);
            let ai = *a.add((j + 1) as usize);
            let br = *b.add(j as usize);
            let bi = *b.add((j + 1) as usize);
            *a.add(j as usize) = ar * br + ai * bi;
            *a.add((j + 1) as usize) = ai * br - ar * bi;
        }
    }
}
pub unsafe fn conjugateproduct(a: *mut f32, b: *const f32, nx: *const i32, ny: *const i32) {
    unsafe { conjugate_product(a, b, *nx, *ny) }
}

pub unsafe fn subarea_cc_coefficient(
    a: *const f32,
    b: *const f32,
    d: i32,
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    dx: i32,
    dy: i32,
) -> f64 {
    unsafe {
        let n = ((x1 + 1 - x0) * (y1 + 1 - y0)) as f64;
        let (mut asum, mut bsum, mut csum, mut asq, mut bsq) = (0., 0., 0., 0., 0.);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let av = *a.add((x + y * d) as usize) as f64;
                let bv = *b.add((x - dx + (y - dy) * d) as usize) as f64;
                asum += av;
                bsum += bv;
                csum += av * bv;
                asq += av * av;
                bsq += bv * bv;
            }
        }
        let den = (n * asq - asum * asum) * (n * bsq - bsum * bsum);
        if den <= 0. {
            0.
        } else {
            (n * csum - asum * bsum) / den.sqrt()
        }
    }
}
pub unsafe fn cc_coefficient_two_pads(
    a: *const f32,
    b: *const f32,
    d: i32,
    nx: i32,
    ny: i32,
    xpeak: f32,
    ypeak: f32,
    nxpa: i32,
    nypa: i32,
    nxpb: i32,
    nypb: i32,
    min: i32,
    nsum: *mut i32,
) -> f64 {
    unsafe {
        let dx = (xpeak + 0.5).floor() as i32;
        let dy = (ypeak + 0.5).floor() as i32;
        let xs = nxpa.max(nxpb + dx);
        let xe = (nx - nxpa).min(nx - nxpb + dx);
        let ys = nypa.max(nypb + dy);
        let ye = (ny - nypa).min(ny - nypb + dy);
        *nsum = (xe - xs).max(0) * (ye - ys).max(0);
        if xe < xs || ye < ys || *nsum < min {
            0.
        } else {
            subarea_cc_coefficient(a, b, d, xs, xe - 1, ys, ye - 1, dx, dy)
        }
    }
}
pub unsafe fn xcorr_cc_coefficient(
    a: *const f32,
    b: *const f32,
    d: i32,
    nx: i32,
    ny: i32,
    x: f32,
    y: f32,
    px: i32,
    py: i32,
    n: *mut i32,
) -> f64 {
    unsafe { cc_coefficient_two_pads(a, b, d, nx, ny, x, y, px, py, px, py, 25, n) }
}
pub unsafe fn weighted_corr_from_sums(
    a: f64,
    asq: f64,
    b: f64,
    bsq: f64,
    ab: f64,
    w: f64,
    sums: *mut f64,
    _desc: *const c_char,
) -> f64 {
    unsafe {
        if !sums.is_null() {
            for (i, v) in [a, asq, b, bsq, ab, w].into_iter().enumerate() {
                *sums.add(i) = v;
            }
        }
        let am = a / w;
        let bm = b / w;
        let den = (asq / w - am * am) * (bsq / w - bm * bm);
        if den <= 0. {
            0.
        } else {
            (ab / w - am * bm) / den.sqrt()
        }
    }
}

pub unsafe fn slice_gaussian_kernel(mat: *mut f32, dim: i32, sigma: f32) {
    unsafe {
        // `filtxcorr.c:376-385`: `mid` is a double, so `i - mid` and `j - mid`
        // are double, `sigma * sigma` is computed in float and then widened by
        // the division, and `exp` is the double routine whose result is cast to
        // float once at the end.  Evaluating the whole expression in f32
        // instead shifts the kernel weights by an ulp or two, which is visible
        // in every pixel of a Gaussian-smoothed image.
        let mid = (dim - 1) as f64 / 2.;
        let mut sum = 0_f32;
        for y in 0..dim {
            for x in 0..dim {
                let sigma_squared = sigma * sigma;
                let v = (-((x as f64 - mid) * (x as f64 - mid)
                    + (y as f64 - mid) * (y as f64 - mid))
                    / sigma_squared as f64)
                    .exp() as f32;
                *mat.add((x + y * dim) as usize) = v;
                sum += v;
            }
        }
        for i in 0..dim * dim {
            *mat.add(i as usize) /= sum;
        }
    }
}
pub unsafe fn scaled_gaussian_kernel(mat: *mut f32, dim: *mut i32, limit: i32, sigma: f32) {
    unsafe {
        // `filtxcorr.c:392` casts ceil() to int first, then does the doubling
        // and the +1 in integer arithmetic.
        *dim = 2 * (sigma as f64).ceil() as i32 + 1;
        *dim = (*dim).min(limit);
        slice_gaussian_kernel(mat, *dim, sigma)
    }
}
pub unsafe fn apply_kernel_filter(
    a: *const f32,
    b: *mut f32,
    d: i32,
    nx: i32,
    ny: i32,
    mat: *const f32,
    k: i32,
) {
    unsafe {
        let below = k / 2;
        for oy in 0..ny {
            for ox in 0..nx {
                let mut sum = 0.;
                for iy in 0..k {
                    for ix in 0..k {
                        let x = (ox + ix - below).clamp(0, nx - 1);
                        let y = (oy + iy - below).clamp(0, ny - 1);
                        sum += *mat.add((ix + iy * k) as usize) * *a.add((x + y * d) as usize);
                    }
                }
                *b.add((ox + oy * d) as usize) = sum;
            }
        }
    }
}

/// `indicesForFFTwrap` (`filtxcorr.c:1881`).
pub unsafe fn indices_for_fft_wrap(
    ny: i32,
    direction: i32,
    iy_out: *mut i32,
    iy_low: *mut i32,
    iy_high: *mut i32,
) -> i32 {
    unsafe {
        *iy_out = 0;
        *iy_low = 0;
        *iy_high = ny / 2;
        if ny % 2 == 0 {
            return 1;
        }
        if direction != 0 {
            *iy_out = ny - 1;
            *iy_low = ny - 2;
            *iy_high = *iy_low - ny / 2;
            -1
        } else {
            *iy_low = 1;
            *iy_high = *iy_low + ny / 2;
            1
        }
    }
}
/// `wrapFFTslice` (`filtxcorr.c:1842`).
pub unsafe fn wrap_fft_slice(array: *mut f32, tmp: *mut f32, nx: i32, ny: i32, direction: i32) {
    unsafe {
        let mut out = 0;
        let mut low = 0;
        let mut high = 0;
        let inc = indices_for_fft_wrap(ny, direction, &mut out, &mut low, &mut high);
        let width = (2 * nx) as usize;
        if ny % 2 != 0 {
            core::ptr::copy_nonoverlapping(array.add((2 * nx * out) as usize), tmp, width);
            for _ in 0..ny / 2 {
                core::ptr::copy(
                    array.add((2 * nx * high) as usize),
                    array.add((2 * nx * out) as usize),
                    width,
                );
                core::ptr::copy(
                    array.add((2 * nx * low) as usize),
                    array.add((2 * nx * high) as usize),
                    width,
                );
                out += inc;
                low += inc;
                high += inc;
            }
            core::ptr::copy_nonoverlapping(tmp, array.add((2 * nx * out) as usize), width);
        } else {
            for _ in 0..ny / 2 {
                core::ptr::copy_nonoverlapping(array.add((2 * nx * low) as usize), tmp, width);
                core::ptr::copy(
                    array.add((2 * nx * high) as usize),
                    array.add((2 * nx * low) as usize),
                    width,
                );
                core::ptr::copy_nonoverlapping(tmp, array.add((2 * nx * high) as usize), width);
                low += 1;
                high += 1;
            }
        }
    }
}
pub unsafe fn wrapfftslice(
    a: *mut f32,
    t: *mut f32,
    nx: *const i32,
    ny: *const i32,
    dir: *const i32,
) {
    unsafe { wrap_fft_slice(a, t, *nx, *ny, *dir) }
}

/// `fourierShiftImage` (`filtxcorr.c:1906`).
pub unsafe fn fourier_shift_image(
    fft: *mut f32,
    nx: i32,
    ny: i32,
    dx: f32,
    dy: f32,
    temp: *mut f32,
) {
    unsafe {
        // `float pi = 3.141593;` -- not the full-precision constant.
        let pi: f32 = 3.141593;
        if dx == 0. && dy == 0. {
            return;
        }
        let nxfft = nx / 2 + 1;
        let ndim = nx + 2;
        for x in 0..nxfft {
            // `freq = 0.5 * ix / (nxFFT - 1.);` is a double expression stored into a float.
            let freq = (0.5 * x as f64 / (nxfft as f64 - 1.)) as f32;
            // `arg` is a double and cos/sin are the double routines, cast to float once.
            let arg = -2. * pi as f64 * freq as f64 * dx as f64;
            *temp.add((2 * x) as usize) = arg.cos() as f32;
            *temp.add((2 * x + 1) as usize) = arg.sin() as f32;
        }
        for y in 0..ny {
            let mut fy = y as f32 / ny as f32;
            if fy > 0.5 {
                fy = (fy as f64 - 1.) as f32;
            }
            let arg = -2. * pi as f64 * fy as f64 * dy as f64;
            let (c, s) = (arg.cos() as f32, arg.sin() as f32);
            for x in 0..nxfft {
                let q = 2 * x;
                let pr = *temp.add(q as usize) * c - *temp.add((q + 1) as usize) * s;
                let pi = *temp.add((q + 1) as usize) * c + *temp.add(q as usize) * s;
                let p = (q + y * ndim) as usize;
                let re = *fft.add(p);
                let im = *fft.add(p + 1);
                *fft.add(p) = pr * re - pi * im;
                *fft.add(p + 1) = pi * re + pr * im;
            }
        }
    }
}
pub unsafe fn fourier_reduce_image(
    input: *mut f32,
    nxi: i32,
    nyi: i32,
    out: *mut f32,
    nxo: i32,
    nyo: i32,
    dx: f32,
    dy: f32,
    temp: *mut f32,
) {
    unsafe {
        let fac = nxi as f32 / nxo as f32;
        // `float dxy = -(redFac - 1) / (2. * redFac);` -- the `2. *` makes the divide double.
        let d = ((-(fac - 1.)) as f64 / (2. * fac as f64)) as f32;
        let mut dst = out;
        for loopi in 0..2 {
            let (start, end) = if loopi == 0 {
                (0, nyo - nyo / 2)
            } else {
                (nyi - nyo / 2, nyi)
            };
            for y in start..end {
                for x in 0..nxo + 2 {
                    *dst = *input.add((y * (nxi + 2) + x) as usize) / fac;
                    dst = dst.add(1);
                }
            }
        }
        if !temp.is_null() {
            fourier_shift_image(out, nxo, nyo, dx / fac + d, dy / fac + d, temp)
        }
    }
}
pub unsafe fn fourier_expand_image(
    input: *mut f32,
    nxi: i32,
    nyi: i32,
    out: *mut f32,
    nxo: i32,
    nyo: i32,
    dx: f32,
    dy: f32,
    temp: *mut f32,
) {
    unsafe {
        let fac = nxo as f32 / nxi as f32;
        // `float dxy = (expFac - 1) / (2. * expFac);` -- integer 1 here, but a double divide.
        let d = ((fac - 1.) as f64 / (2. * fac as f64)) as f32;
        core::ptr::write_bytes(out, 0, ((nxo + 2) * nyo) as usize);
        if !temp.is_null() {
            fourier_shift_image(input, nxi, nyi, dx + d, dy + d, temp)
        }
        let mut src = input;
        for loopi in 0..2 {
            let (start, end) = if loopi == 0 {
                (0, nyi - nyi / 2)
            } else {
                (nyo - nyi / 2, nyo)
            };
            for y in start..end {
                for x in 0..nxi + 2 {
                    *out.add((y * (nxo + 2) + x) as usize) = *src * fac;
                    src = src.add(1);
                }
            }
        }
    }
}
pub unsafe fn fourier_crop_sizes(
    size: i32,
    factor: f32,
    pad: f32,
    min_pad: i32,
    limit: i32,
    full: *mut i32,
    crop: *mut i32,
    actual: *mut f32,
) -> i32 {
    unsafe {
        if factor == 1. {
            return 1;
        }
        // `useFac = 1. / factor;` is a double divide stored into a float.
        let usefac = if factor < 1. {
            (1. / factor as f64) as f32
        } else {
            factor
        };
        let mut denom = 0;
        let mut numer = 0;
        for d in [1, 2, 3, 4, 5, 6, 8, 10] {
            // `B3DNINT(a)` is `(int)floor((a) + 0.5)`, evaluated in double.
            let n = ((usefac * d as f32) as f64 + 0.5).floor() as i32;
            *actual = n as f32 / d as f32;
            if (usefac - *actual).abs() < 0.001 {
                denom = d;
                numer = n;
                break;
            }
        }
        if denom == 0 {
            return 2;
        }
        if factor < 1. {
            *actual = (1. / *actual as f64) as f32;
            core::mem::swap(&mut denom, &mut numer);
        }
        let p = (min_pad as f32).max(pad * size as f32) as i32;
        let divisor = 2 * numer;
        let base = divisor * ((size + 2 * p + divisor - 1) / divisor);
        let f = nice_frame(base, divisor, limit);
        *crop = f * denom / numer;
        *full = f;
        0
    }
}

/// `XCorrFilterPart` (`filtxcorr.c:271`).
pub unsafe fn xcorr_filter_part(
    fft: *const f32,
    array: *mut f32,
    nx: i32,
    ny: i32,
    ctf: *const f32,
    delta: f32,
) {
    unsafe {
        let nx2 = nx / 2;
        let nx2p1 = nx2 + 1;
        let dx = 1.0 / nx as f32;
        let dy = 1.0 / ny as f32;
        let mut last = (0.707 / delta) as i32;
        while last > 1 && *ctf.add(last as usize) == 0. {
            last -= 1;
        }
        let maxf = (last + 1) as f32 * delta;
        let nxmax = ((maxf / dx + 1.0) as i32).clamp(1, nx2);
        for iy in 0..ny {
            let mut fy = iy as f32 * dy;
            if fy > 0.5 {
                fy = 1. - fy;
            }
            let base = iy * nx2p1;
            if fy > maxf {
                for ix in 0..=nx2 {
                    let ind = 2 * (base + ix);
                    *array.add(ind as usize) = 0.;
                    *array.add((ind + 1) as usize) = 0.;
                }
            } else {
                for ix in 0..=nxmax {
                    let ind = 2 * (base + ix);
                    let freq = ((ix as f32 * dx).powi(2) + fy * fy).sqrt();
                    let f = *ctf.add((freq / delta + 0.5) as usize);
                    *array.add(ind as usize) = *fft.add(ind as usize) * f;
                    *array.add((ind + 1) as usize) = *fft.add((ind + 1) as usize) * f;
                }
                // C deliberately stops at nxDiv2 (exclusive), retaining the
                // Nyquist pair when it is not visited by either loop.
                for ix in nxmax + 1..nx2 {
                    let ind = 2 * (base + ix);
                    *array.add(ind as usize) = 0.;
                    *array.add((ind + 1) as usize) = 0.;
                }
            }
        }
    }
}

/// Original static `peakHalfWidth` (`filtxcorr.c:841`).
unsafe fn peak_half_width(
    array: *mut f32,
    ix_peak: i32,
    iy_peak: i32,
    nx: i32,
    ny: i32,
    delx: i32,
    dely: i32,
) -> f32 {
    unsafe {
        let nxdim = nx + 2;
        let peak = *array.add((ix_peak + iy_peak * nxdim) as usize);
        let scale = ((delx * delx + dely * dely) as f32).sqrt();
        let mut last_val = peak;
        let mut dist = 1;
        while dist < nx.min(ny) / 4 {
            let ix = (ix_peak + dist * delx + nx) % nx;
            let iy = (iy_peak + dist * dely + ny) % ny;
            let val = *array.add((ix + iy * nxdim) as usize);
            if val < peak / 2. {
                return scale * (dist as f32 + (last_val - peak / 2.) / (last_val - val) - 1.);
            }
            last_val = val;
            dist += 1;
        }
        scale * dist as f32
    }
}

/// Original static `accumRotatedCorner` (`filtxcorr.c:861`).
unsafe fn accum_rotated_corner(
    limit_x: i32,
    limit_y: i32,
    cosine: f32,
    sine: f32,
    lim_xlo: *mut i32,
    lim_xhi: *mut i32,
    lim_ylo: *mut i32,
    lim_yhi: *mut i32,
) {
    unsafe {
        let idx = (cosine * limit_x as f32 - sine * limit_y as f32) as i32;
        let idy = (sine * limit_x as f32 + cosine * limit_y as f32) as i32;
        *lim_xlo = (*lim_xlo).min(idx);
        *lim_xhi = (*lim_xhi).max(idx);
        *lim_ylo = (*lim_ylo).min(idy);
        *lim_yhi = (*lim_yhi).max(idy);
    }
}

/// Original static `computeTestLimits` (`filtxcorr.c:873`).
unsafe fn compute_test_limits(
    nx: i32,
    ny: i32,
    test_xlo: *mut i32,
    test_xhi: *mut i32,
    test_ylo: *mut i32,
    test_yhi: *mut i32,
    cos_lim: *mut f32,
    sin_lim: *mut f32,
) {
    unsafe {
        *test_xlo = (-nx / 2).max(S_LIMIT_XLO.load(Ordering::SeqCst));
        *test_xhi = (nx / 2 - 1).min(S_LIMIT_XHI.load(Ordering::SeqCst));
        *test_ylo = (-ny / 2).max(S_LIMIT_YLO.load(Ordering::SeqCst));
        *test_yhi = (ny / 2 - 1).min(S_LIMIT_YHI.load(Ordering::SeqCst));
        let angle = f32::from_bits(S_LIMIT_ANGLE.load(Ordering::SeqCst));
        if angle != 0. {
            *cos_lim = (angle * 0.017453292519943295).cos();
            *sin_lim = (angle * 0.017453292519943295).sin();
            *test_xlo = 2_000_000_000;
            *test_xhi = -2_000_000_000;
            *test_ylo = 2_000_000_000;
            *test_yhi = -2_000_000_000;
            let xlo = S_LIMIT_XLO.load(Ordering::SeqCst);
            let xhi = S_LIMIT_XHI.load(Ordering::SeqCst);
            let ylo = S_LIMIT_YLO.load(Ordering::SeqCst);
            let yhi = S_LIMIT_YHI.load(Ordering::SeqCst);
            accum_rotated_corner(
                xhi + 1,
                yhi + 1,
                *cos_lim,
                *sin_lim,
                test_xlo,
                test_xhi,
                test_ylo,
                test_yhi,
            );
            accum_rotated_corner(
                xhi + 1,
                ylo - 1,
                *cos_lim,
                *sin_lim,
                test_xlo,
                test_xhi,
                test_ylo,
                test_yhi,
            );
            accum_rotated_corner(
                xlo - 1,
                yhi + 1,
                *cos_lim,
                *sin_lim,
                test_xlo,
                test_xhi,
                test_ylo,
                test_yhi,
            );
            accum_rotated_corner(
                xlo - 1,
                ylo - 1,
                *cos_lim,
                *sin_lim,
                test_xlo,
                test_xhi,
                test_ylo,
                test_yhi,
            );
            *test_xlo = (-nx / 2).max(*test_xlo - 1);
            *test_xhi = (nx / 2 - 1).min(*test_xhi + 1);
            *test_ylo = (-ny / 2).max(*test_ylo - 1);
            *test_yhi = (ny / 2 - 1).min(*test_yhi + 1);
        }
    }
}
pub unsafe fn xcorr_peak_find_width(
    array: *mut f32,
    nxdim: i32,
    ny: i32,
    xpeak: *mut f32,
    ypeak: *mut f32,
    peak: *mut f32,
    width: *mut f32,
    width_min: *mut f32,
    max_peaks: i32,
    min_strength: f32,
) {
    unsafe {
        let nx = nxdim - 2;
        S_PEAK_FIND_ERROR.store(0, Ordering::SeqCst);
        for i in 0..max_peaks {
            *peak.add(i as usize) = -1.0e30;
            *xpeak.add(i as usize) = 0.;
            *ypeak.add(i as usize) = 0.;
        }
        // This is deliberately the same two-pass selection as XCorrPeakFindWidth:
        // find the global value first when it sets a strength threshold, then retain
        // only eight-neighbour local maxima.  The C implementation parallelizes the
        // scans; the ordering here is serial but the comparisons and ties are identical.
        let apply = S_APPLY_LIMITS.load(Ordering::SeqCst);
        let xlo = S_LIMIT_XLO.load(Ordering::SeqCst);
        let xhi = S_LIMIT_XHI.load(Ordering::SeqCst);
        let ylo = S_LIMIT_YLO.load(Ordering::SeqCst);
        let yhi = S_LIMIT_YHI.load(Ordering::SeqCst);
        let angle = f32::from_bits(S_LIMIT_ANGLE.load(Ordering::SeqCst));
        let test_angle = apply != 0 && angle != 0.;
        let mut cosine = 1.;
        let mut sine = 0.;
        let (mut txlo, mut txhi, mut tylo, mut tyhi) = (xlo, xhi, ylo, yhi);
        if apply != 0 {
            compute_test_limits(
                nx,
                ny,
                &mut txlo,
                &mut txhi,
                &mut tylo,
                &mut tyhi,
                &mut cosine,
                &mut sine,
            );
        }
        let xcen = 0.5 * (xlo + xhi) as f32;
        let ycen = 0.5 * (ylo + yhi) as f32;
        let xrad_sq = ((xhi - xlo) as f32 / 2.).max(1.).powi(2);
        let yrad_sq = ((yhi - ylo) as f32 / 2.).max(1.).powi(2);
        let mut threshold = 0.;
        if max_peaks < 2 || min_strength > 0. {
            let mut best = -1e30_f32;
            let mut bix = 0;
            let mut biy = 0;
            let ystart = if apply != 0 { tylo } else { 0 };
            let yend = if apply != 0 { tyhi } else { ny - 1 };
            for idy_line in ystart..=yend {
                let iy = if idy_line < 0 {
                    idy_line + ny
                } else {
                    idy_line
                };
                for ix in 0..nx {
                    let mut idx = if ix > nx / 2 { ix - nx } else { ix };
                    let mut idy = idy_line;
                    if apply != 0 {
                        if idx < txlo || idx > txhi {
                            continue;
                        }
                        if test_angle {
                            let ixrot = (idx as f32 * cosine + idy as f32 * sine).round() as i32;
                            idy = (-idx as f32 * sine + idy as f32 * cosine).round() as i32;
                            idx = ixrot;
                            if idy < ylo || idy > yhi {
                                continue;
                            }
                        }
                        if idx < xlo || idx > xhi {
                            continue;
                        }
                        if apply < 0 {
                            let cx = idx as f32 - xcen;
                            let cy = idy as f32 - ycen;
                            if cx * cx / xrad_sq + cy * cy / yrad_sq > 1. {
                                continue;
                            }
                        }
                    }
                    let val = *array.add((ix + iy * nxdim) as usize);
                    if val > best {
                        best = val;
                        bix = ix;
                        biy = iy;
                    }
                }
            }
            if best > -0.9e30 {
                *peak = best;
                *xpeak = bix as f32;
                *ypeak = biy as f32;
            }
            threshold = min_strength * *peak;
        }
        if max_peaks > 1 {
            // C stores at most maxPeaks maxima per line then obtains the same top set.
            // Keeping the line cap here preserves its bounded candidate semantics.
            let mut candidates: Vec<(f32, i32, i32)> = Vec::new();
            let ystart = if apply != 0 { tylo } else { -ny / 2 };
            let yend = if apply != 0 { tyhi } else { ny / 2 - 1 };
            for idy_line in ystart..=yend {
                let iy = if idy_line < 0 {
                    idy_line + ny
                } else {
                    idy_line
                };
                let mut line: Vec<(f32, i32)> = Vec::new();
                for ix in 0..nx {
                    let mut idx = if ix > nx / 2 { ix - nx } else { ix };
                    let mut idy = idy_line;
                    if apply != 0 {
                        if idx < txlo || idx > txhi {
                            continue;
                        }
                        if test_angle {
                            let ixrot = (idx as f32 * cosine + idy as f32 * sine).round() as i32;
                            idy = (-idx as f32 * sine + idy as f32 * cosine).round() as i32;
                            idx = ixrot;
                            if idy < ylo || idy > yhi {
                                continue;
                            }
                        }
                        if idx < xlo || idx > xhi {
                            continue;
                        }
                        if apply < 0 {
                            let cx = idx as f32 - xcen;
                            let cy = idy as f32 - ycen;
                            if cx * cx / xrad_sq + cy * cy / yrad_sq > 1. {
                                continue;
                            }
                        }
                    }
                    let val = *array.add((ix + iy * nxdim) as usize);
                    if val <= threshold {
                        continue;
                    }
                    let xm = (ix + nx - 1) % nx;
                    let xp = (ix + 1) % nx;
                    let ym = (iy + ny - 1) % ny;
                    let yp = (iy + 1) % ny;
                    if val > *array.add((xm + ym * nxdim) as usize)
                        && val >= *array.add((xp + ym * nxdim) as usize)
                        && val > *array.add((xm + iy * nxdim) as usize)
                        && val >= *array.add((xp + iy * nxdim) as usize)
                        && val > *array.add((xm + yp * nxdim) as usize)
                        && val >= *array.add((xp + yp * nxdim) as usize)
                        && val > *array.add((ix + ym * nxdim) as usize)
                        && val >= *array.add((ix + yp * nxdim) as usize)
                    {
                        line.push((val, ix));
                        if line.len() > max_peaks as usize {
                            let low = line
                                .iter()
                                .enumerate()
                                .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
                                .unwrap()
                                .0;
                            line.remove(low);
                        }
                    }
                }
                candidates.extend(line.into_iter().map(|(v, x)| (v, x, iy)));
            }
            candidates.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            for (i, (v, x, y)) in candidates.into_iter().take(max_peaks as usize).enumerate() {
                *peak.add(i) = v;
                *xpeak.add(i) = x as f32;
                *ypeak.add(i) = y as f32;
            }
        }
        for i in 0..max_peaks {
            if *peak.add(i as usize) < -0.9e30 {
                continue;
            }
            let ix = *xpeak.add(i as usize) as i32;
            let iy = *ypeak.add(i as usize) as i32;
            let cx = parabolic_fit_position(
                *array.add(((ix + nx - 1) % nx + iy * nxdim) as usize),
                *peak.add(i as usize),
                *array.add(((ix + 1) % nx + iy * nxdim) as usize),
            ) as f32;
            let cy = parabolic_fit_position(
                *array.add((ix + ((iy + ny - 1) % ny) * nxdim) as usize),
                *peak.add(i as usize),
                *array.add((ix + ((iy + 1) % ny) * nxdim) as usize),
            ) as f32;
            let mut px = ix as f32 + cx;
            let mut py = iy as f32 + cy;
            if px > nx as f32 / 2. {
                px -= nx as f32;
            }
            if py > ny as f32 / 2. {
                py -= ny as f32;
            }
            *xpeak.add(i as usize) = px;
            *ypeak.add(i as usize) = py;
            if !width.is_null() && !width_min.is_null() {
                let values = [
                    peak_half_width(array, ix, iy, nx, ny, 1, 0)
                        + peak_half_width(array, ix, iy, nx, ny, -1, 0),
                    peak_half_width(array, ix, iy, nx, ny, 0, 1)
                        + peak_half_width(array, ix, iy, nx, ny, 0, -1),
                    peak_half_width(array, ix, iy, nx, ny, 1, 1)
                        + peak_half_width(array, ix, iy, nx, ny, -1, -1),
                    peak_half_width(array, ix, iy, nx, ny, 1, -1)
                        + peak_half_width(array, ix, iy, nx, ny, -1, 1),
                ];
                *width.add(i as usize) = values.iter().sum::<f32>() / 4.;
                *width_min.add(i as usize) = values.into_iter().fold(f32::INFINITY, f32::min);
            }
        }
        S_APPLY_LIMITS.store(0, Ordering::SeqCst);
        S_LIMIT_ANGLE.store(0, Ordering::SeqCst);
        S_WILL_CHECK_ERROR.store(0, Ordering::SeqCst);
    }
}
pub unsafe fn get_peak_find_test_limits(
    nx: i32,
    ny: i32,
    lim_xlo: *mut i32,
    lim_xhi: *mut i32,
    lim_ylo: *mut i32,
    lim_yhi: *mut i32,
) {
    unsafe {
        let mut cosine = 0.;
        let mut sine = 0.;
        compute_test_limits(
            nx,
            ny,
            lim_xlo,
            lim_xhi,
            lim_ylo,
            lim_yhi,
            &mut cosine,
            &mut sine,
        );
    }
}
pub unsafe fn xcorr_peak_find(
    array: *mut f32,
    nxdim: i32,
    ny: i32,
    xpeak: *mut f32,
    ypeak: *mut f32,
    peak: *mut f32,
    max_peaks: i32,
) {
    unsafe {
        xcorr_peak_find_width(
            array,
            nxdim,
            ny,
            xpeak,
            ypeak,
            peak,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            max_peaks,
            0.,
        )
    }
}
pub fn store_peak_find_error(value: i32) {
    S_WILL_CHECK_ERROR.store(value, Ordering::SeqCst);
}
pub fn get_peak_find_error() -> i32 {
    S_PEAK_FIND_ERROR.load(Ordering::SeqCst)
}
pub fn set_peak_find_limits(xlo: i32, xhi: i32, ylo: i32, yhi: i32, ellipse: i32) {
    S_APPLY_LIMITS.store(if ellipse != 0 { -1 } else { 1 }, Ordering::SeqCst);
    S_LIMIT_XLO.store(xlo, Ordering::SeqCst);
    S_LIMIT_XHI.store(xhi, Ordering::SeqCst);
    S_LIMIT_YLO.store(ylo, Ordering::SeqCst);
    S_LIMIT_YHI.store(yhi, Ordering::SeqCst);
}
pub fn set_peak_find_angle(angle: f32) {
    S_LIMIT_ANGLE.store(angle.to_bits(), Ordering::SeqCst);
}
/// `findManyXCorrPeaks` (`filtxcorr.c:1029`).
///
/// Translated statement by statement.  The previous version collected every
/// local maximum into a `Vec` and sorted once at the end; the source instead
/// keeps a running `threshold` (the lowest peak currently retained), skips
/// anything below it, and sorts-and-repacks into an alternating pair of
/// buffers whenever `maxGrow` is reached.  That is not just an optimisation:
/// the threshold changes which peaks are examined at all, and the peak test
/// itself is deliberately asymmetric — `>` against the lower/left neighbours
/// and `>=` against the upper/right ones — so ties resolve to one particular
/// pixel.
pub unsafe fn find_many_xcorr_peaks(
    array: *mut f32,
    nxdim: i32,
    ny: i32,
    ix_offset: i32,
    iy_offset: i32,
    xpeak: *mut f32,
    ypeak: *mut f32,
    peak: *mut f32,
    max_peaks: i32,
    max_grow: i32,
    num_found: *mut i32,
) -> i32 {
    unsafe {
        let mut threshold = -1.0e30_f32;
        let mut min_found = 1.0e30_f32;
        let mut num_peaks = 0_i32;
        let mut use_temp2 = 1_i32;
        let mut if_first = 1_i32;

        if max_grow as f32 <= 1.05 * max_peaks as f32 {
            return 1;
        }

        let indexes = libc::malloc(max_grow as usize * core::mem::size_of::<i32>()).cast::<i32>();
        let ix_temp1 = libc::malloc(max_grow as usize * core::mem::size_of::<i16>()).cast::<i16>();
        let ix_temp2 = libc::malloc(max_grow as usize * core::mem::size_of::<i16>()).cast::<i16>();
        let iy_temp1 = libc::malloc(max_grow as usize * core::mem::size_of::<i16>()).cast::<i16>();
        let iy_temp2 = libc::malloc(max_grow as usize * core::mem::size_of::<i16>()).cast::<i16>();
        let peak_temp1 =
            libc::malloc(max_grow as usize * core::mem::size_of::<f32>()).cast::<f32>();
        let peak_temp2 =
            libc::malloc(max_grow as usize * core::mem::size_of::<f32>()).cast::<f32>();
        if ix_temp1.is_null()
            || ix_temp2.is_null()
            || iy_temp1.is_null()
            || iy_temp2.is_null()
            || peak_temp1.is_null()
            || peak_temp2.is_null()
            || indexes.is_null()
        {
            libc::free(indexes.cast());
            libc::free(ix_temp1.cast());
            libc::free(ix_temp2.cast());
            libc::free(iy_temp1.cast());
            libc::free(iy_temp2.cast());
            libc::free(peak_temp1.cast());
            libc::free(peak_temp2.cast());
            return -1;
        }

        let mut ix_temp = ix_temp1;
        let mut iy_temp = iy_temp1;
        let mut peak_temp = peak_temp1;
        let ix_start = 0.max(ix_offset + 1);
        let mut ix_end = nxdim.min(nxdim - ix_offset - 1);
        if ix_offset < -1 {
            ix_end = nxdim - 2;
        }
        let iy_start = 0.max(iy_offset + 1);
        let iy_end = ny.min(ny - iy_offset - 1);

        for iy in iy_start..iy_end {
            let ybase = nxdim * iy;
            let yprev = (iy + ny - 1) % ny;
            let ynext = (iy + 1) % ny;
            for ix in ix_start..ix_end {
                let xbase = ybase + ix;
                let val = *array.add(xbase as usize);

                // Ignore anything below the lowest peak currently retained.
                if val < threshold {
                    continue;
                }

                let mut is_peak = 0;
                if iy_offset < 0 && (iy == 0 || iy == ny - 1 || ix == 0 || ix == ix_end - 1) {
                    let xprev = (ix + ix_end - 1) % ix_end;
                    let xnext = (ix + 1) % ix_end;
                    if val < *array.add((xprev + iy * nxdim) as usize)
                        || val < *array.add((xnext + iy * nxdim) as usize)
                        || val < *array.add((xprev + yprev * nxdim) as usize)
                        || val < *array.add((ix + yprev * nxdim) as usize)
                        || val < *array.add((xnext + yprev * nxdim) as usize)
                        || val < *array.add((xprev + ynext * nxdim) as usize)
                        || val < *array.add((ix + ynext * nxdim) as usize)
                        || val < *array.add((xnext + ynext * nxdim) as usize)
                    {
                        continue;
                    }
                    is_peak = 1;
                }

                if is_peak != 0
                    || (val > *array.add((xbase - nxdim) as usize)
                        && val >= *array.add((xbase + nxdim) as usize)
                        && val > *array.add((xbase - 1) as usize)
                        && val >= *array.add((xbase + 1) as usize)
                        && val > *array.add((xbase + nxdim - 1) as usize)
                        && val >= *array.add((xbase + 1 - nxdim) as usize)
                        && val > *array.add((xbase + nxdim + 1) as usize)
                        && val >= *array.add((xbase - 1 - nxdim) as usize))
                {
                    *ix_temp.add(num_peaks as usize) = ix as i16;
                    *iy_temp.add(num_peaks as usize) = iy as i16;
                    *peak_temp.add(num_peaks as usize) = -val;
                    *indexes.add(num_peaks as usize) = num_peaks;
                    num_peaks += 1;

                    if num_peaks <= max_peaks && val < min_found {
                        min_found = val;
                    }
                    if num_peaks == max_peaks {
                        threshold = min_found;
                    }

                    if num_peaks == max_grow {
                        sort_and_repack(
                            indexes,
                            ix_temp,
                            iy_temp,
                            peak_temp,
                            max_peaks,
                            num_peaks,
                            if use_temp2 != 0 { ix_temp2 } else { ix_temp1 },
                            if use_temp2 != 0 { iy_temp2 } else { iy_temp1 },
                            if use_temp2 != 0 {
                                peak_temp2
                            } else {
                                peak_temp1
                            },
                            if_first,
                        );
                        if_first = 0;
                        num_peaks = max_peaks;
                        ix_temp = if use_temp2 != 0 { ix_temp2 } else { ix_temp1 };
                        iy_temp = if use_temp2 != 0 { iy_temp2 } else { iy_temp1 };
                        peak_temp = if use_temp2 != 0 {
                            peak_temp2
                        } else {
                            peak_temp1
                        };
                        use_temp2 = 1 - use_temp2;
                        threshold = -*peak_temp.add((max_peaks - 1) as usize);
                    }
                }
            }
        }

        sort_and_repack(
            indexes,
            ix_temp,
            iy_temp,
            peak_temp,
            max_peaks,
            num_peaks,
            if use_temp2 != 0 { ix_temp2 } else { ix_temp1 },
            if use_temp2 != 0 { iy_temp2 } else { iy_temp1 },
            if use_temp2 != 0 {
                peak_temp2
            } else {
                peak_temp1
            },
            if_first,
        );
        ix_temp = if use_temp2 != 0 { ix_temp2 } else { ix_temp1 };
        iy_temp = if use_temp2 != 0 { iy_temp2 } else { iy_temp1 };
        peak_temp = if use_temp2 != 0 {
            peak_temp2
        } else {
            peak_temp1
        };
        num_peaks = num_peaks.min(max_peaks);

        for ind in 0..num_peaks {
            *peak.add(ind as usize) = -*peak_temp.add(ind as usize);
            let ix = *ix_temp.add(ind as usize) as i32;
            let iy = *iy_temp.add(ind as usize) as i32;
            let xbase = ix + iy * nxdim;
            let (cx, cy);
            if iy_offset < 0 && (ix == 0 || iy == 0 || ix == ix_end - 1 || iy == ny - 1) {
                let xprev = (ix + ix_end - 1) % ix_end;
                let xnext = (ix + 1) % ix_end;
                let yprev = (iy + ny - 1) % ny;
                let ynext = (iy + 1) % ny;
                cx = parabolic_fit_position(
                    *array.add((xprev + iy * nxdim) as usize),
                    *array.add(xbase as usize),
                    *array.add((xnext + iy * nxdim) as usize),
                ) as f32;
                cy = parabolic_fit_position(
                    *array.add((ix + yprev * nxdim) as usize),
                    *array.add(xbase as usize),
                    *array.add((ix + ynext * nxdim) as usize),
                ) as f32;
            } else {
                cx = parabolic_fit_position(
                    *array.add((xbase - 1) as usize),
                    *array.add(xbase as usize),
                    *array.add((xbase + 1) as usize),
                ) as f32;
                cy = parabolic_fit_position(
                    *array.add((xbase - nxdim) as usize),
                    *array.add(xbase as usize),
                    *array.add((xbase + nxdim) as usize),
                ) as f32;
            }
            *xpeak.add(ind as usize) = ix as f32 + cx;
            *ypeak.add(ind as usize) = iy as f32 + cy;
        }
        *num_found = num_peaks;

        if iy_offset < 0 {
            for ind in 0..num_peaks {
                if *xpeak.add(ind as usize) > (ix_end / 2) as f32 {
                    *xpeak.add(ind as usize) -= ix_end as f32;
                }
                if *ypeak.add(ind as usize) > (ny / 2) as f32 {
                    *ypeak.add(ind as usize) -= ny as f32;
                }
            }
        }

        libc::free(indexes.cast());
        libc::free(ix_temp1.cast());
        libc::free(ix_temp2.cast());
        libc::free(iy_temp1.cast());
        libc::free(iy_temp2.cast());
        libc::free(peak_temp1.cast());
        libc::free(peak_temp2.cast());
        0
    }
}

/// `sortAndRepack` (`filtxcorr.c:1204`), the file-static helper that sorts the
/// new part of the peak arrays and repacks the ones being kept into a different
/// set of arrays.  `COPY_FROM_TO` (`filtxcorr.c:1198`) is expanded at each of
/// its three use sites.
unsafe fn sort_and_repack(
    indexes: *mut i32,
    ix_from: *const i16,
    iy_from: *const i16,
    peak_from: *mut f32,
    keep_peaks: i32,
    num_peaks: i32,
    ix_to: *mut i16,
    iy_to: *mut i16,
    peak_to: *mut f32,
    if_first: i32,
) {
    unsafe {
        let mut ind: i32;
        let mut from: i32;
        let mut low: i32;
        let mut high: i32;

        /* First time or not enough peaks, just sort and repack */
        if if_first > 0 || num_peaks <= keep_peaks {
            rs_sort_indexed_floats(peak_from, indexes, num_peaks);
            ind = 0;
            while ind
                < if keep_peaks < num_peaks {
                    keep_peaks
                } else {
                    num_peaks
                }
            {
                from = *indexes.add(ind as usize);
                *ix_to.add(ind as usize) = *ix_from.add(from as usize);
                *iy_to.add(ind as usize) = *iy_from.add(from as usize);
                *peak_to.add(ind as usize) = *peak_from.add(from as usize);
                *indexes.add(ind as usize) = ind;
                ind += 1;
            }
        } else {
            /* Otherwise sort the upper part of the array, setting index offset
            appropriately */
            rs_set_sort_index_offset(keep_peaks);
            rs_sort_indexed_floats(
                peak_from.add(keep_peaks as usize),
                indexes.add(keep_peaks as usize),
                num_peaks - keep_peaks,
            );

            /* Merge the two sections by taking the lowest value from each eat each step */
            low = 0;
            high = keep_peaks;
            ind = 0;
            while ind < keep_peaks && low < keep_peaks && high < num_peaks {
                if *peak_from.add(*indexes.add(low as usize) as usize)
                    < *peak_from.add(*indexes.add(high as usize) as usize)
                {
                    from = *indexes.add(low as usize);
                    low += 1;
                } else {
                    from = *indexes.add(high as usize);
                    high += 1;
                }
                *ix_to.add(ind as usize) = *ix_from.add(from as usize);
                *iy_to.add(ind as usize) = *iy_from.add(from as usize);
                *peak_to.add(ind as usize) = *peak_from.add(from as usize);
                *indexes.add(ind as usize) = ind;
                ind += 1;
            }

            /* Finish up with one or the other if deficient */
            low = if low < keep_peaks { low } else { high };
            while ind < keep_peaks {
                from = *indexes.add(low as usize);
                low += 1;
                *ix_to.add(ind as usize) = *ix_from.add(from as usize);
                *iy_to.add(ind as usize) = *iy_from.add(from as usize);
                *peak_to.add(ind as usize) = *peak_from.add(from as usize);
                *indexes.add(ind as usize) = ind;
                ind += 1;
            }
        }
    }
}

pub unsafe fn find_spaced_xcorr_peaks(
    array: *mut f32,
    nxdim: i32,
    ix_min: i32,
    ix_max: i32,
    iy_min: i32,
    iy_max: i32,
    xpeak: *mut f32,
    ypeak: *mut f32,
    peak: *mut f32,
    max_peaks: i32,
    min_spacing: f32,
    num_peaks: *mut i32,
    min_strength: f32,
) -> i32 {
    unsafe {
        // Match the source's block reduction: one locally maximal candidate is
        // retained from each minSpacing/sqrt(2) square before the spacing pass.
        let block_size = (min_spacing / 2_f32.sqrt()) as i32;
        if block_size <= 0 {
            *num_peaks = 0;
            return 0;
        }
        let nblocks_x = ((ix_max - (ix_min + 1)) + block_size - 1) / block_size;
        let nblocks_y = ((iy_max - (iy_min + 1)) + block_size - 1) / block_size;
        let mut candidates: Vec<(f32, i32, i32, i32, i32)> = Vec::new();
        for iby in 0..nblocks_y {
            let ys = iy_min + 1 + iby * block_size;
            let ye = (ys + block_size).min(iy_max);
            for ibx in 0..nblocks_x {
                let xs = ix_min + 1 + ibx * block_size;
                let xe = (xs + block_size).min(ix_max);
                let mut value = -1e30_f32;
                let mut px = -1;
                let mut py = 0;
                for iy in ys..ye {
                    for ix in xs..xe {
                        let val = *array.add((ix + iy * nxdim) as usize);
                        if val < value {
                            continue;
                        }
                        if val > *array.add((ix + (iy - 1) * nxdim) as usize)
                            && val >= *array.add((ix + (iy + 1) * nxdim) as usize)
                            && val > *array.add((ix - 1 + iy * nxdim) as usize)
                            && val >= *array.add((ix + 1 + iy * nxdim) as usize)
                            && val > *array.add((ix - 1 + (iy - 1) * nxdim) as usize)
                            && val >= *array.add((ix + 1 + (iy - 1) * nxdim) as usize)
                            && val > *array.add((ix + 1 + (iy + 1) * nxdim) as usize)
                            && val >= *array.add((ix - 1 + (iy + 1) * nxdim) as usize)
                        {
                            px = ix;
                            py = iy;
                            value = val;
                        }
                    }
                }
                if px > 0 {
                    candidates.push((-value, px, py, ibx, iby));
                }
            }
        }
        candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        // peak values are inverted like C's peakAll.  Mark lower neighbouring
        // block representatives dead before emitting the sorted survivors.
        let criterion = min_spacing * min_spacing;
        for index in 0..candidates.len().saturating_sub(1) {
            if candidates[index].0 > 1e29 {
                continue;
            }
            let (_, x, y, bx, by) = candidates[index];
            for other in index + 1..candidates.len() {
                if candidates[other].0 > 1e29 {
                    continue;
                }
                if (candidates[other].3 - bx).abs() > 2 || (candidates[other].4 - by).abs() > 2 {
                    continue;
                }
                let dx = (x - candidates[other].1) as f32;
                let dy = (y - candidates[other].2) as f32;
                if dx * dx + dy * dy < criterion {
                    candidates[other].0 = 1e30;
                }
            }
        }
        let mut used = 0;
        for (negative, ix, iy, _, _) in candidates {
            if used >= max_peaks || negative > 1e29 {
                continue;
            }
            let value = -negative;
            if used != 0 && min_strength > 0. && value < min_strength * *peak {
                break;
            }
            let base = ix + iy * nxdim;
            *peak.add(used as usize) = value;
            *xpeak.add(used as usize) = ix as f32
                + parabolic_fit_position(
                    *array.add((base - 1) as usize),
                    value,
                    *array.add((base + 1) as usize),
                ) as f32;
            *ypeak.add(used as usize) = iy as f32
                + parabolic_fit_position(
                    *array.add((base - nxdim) as usize),
                    value,
                    *array.add((base + nxdim) as usize),
                ) as f32;
            used += 1;
        }
        *num_peaks = used as i32;
        0
    }
}
pub unsafe fn weighted_cc_coefficient(
    a: *const f32,
    b: *const f32,
    nx_dim: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    dx: i32,
    dy: i32,
    aw: *const f32,
    bw: *const f32,
    nx_weight: i32,
    bin: i32,
    xoffset: i32,
    yoffset: i32,
) -> f64 {
    unsafe {
        let (mut wsum, mut asum, mut bsum, mut asq, mut bsq, mut ab) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for iy in iy0..=iy1 {
            let abase = iy * nx_dim;
            let bbase = (iy - dy) * nx_dim - dx;
            let awbase = (iy / bin + yoffset) * nx_weight + xoffset;
            let bwbase = ((iy - dy) / bin + yoffset) * nx_weight + xoffset;
            for ix in ix0..=ix1 {
                let av = *a.add((ix + abase) as usize) as f64;
                let bv = *b.add((ix + bbase) as usize) as f64;
                let weight = (*aw.add((ix / bin + awbase) as usize)
                    * *bw.add(((ix - dx) / bin + bwbase) as usize))
                    as f64;
                wsum += weight;
                asum += av * weight;
                bsum += bv * weight;
                asq += av * av * weight;
                bsq += bv * bv * weight;
                ab += av * bv * weight;
            }
        }
        weighted_corr_from_sums(
            asum,
            asq,
            bsum,
            bsq,
            ab,
            wsum,
            core::ptr::null_mut(),
            core::ptr::null(),
        )
    }
}
/// `fourierShiftVolume` (`filtxcorr.c:2070`).
pub unsafe fn fourier_shift_volume(
    fft: *mut f32,
    nx_pad: i32,
    ny_pad: i32,
    nz_pad: i32,
    dx: f32,
    dy: f32,
    dz: f32,
    temp: *mut f32,
) {
    unsafe {
        // `float pi = 3.141593;` -- not the full-precision constant.
        let pi: f32 = 3.141593;
        if dx == 0. && dy == 0. && dz == 0. {
            return;
        }
        let nx_fft = nx_pad / 2 + 1;
        let nx_dim = nx_pad + 2;
        for ix in 0..nx_fft {
            // `freq = 0.5 * ix / (nxFFT - 1.);` is a double expression stored into a float.
            let freq = (0.5 * ix as f64 / (nx_fft as f64 - 1.)) as f32;
            // `arg` is a double and cos/sin are the double routines, cast to float once.
            let arg = -2.0 * pi as f64 * freq as f64 * dx as f64;
            *temp.add((2 * ix) as usize) = arg.cos() as f32;
            *temp.add((2 * ix + 1) as usize) = arg.sin() as f32;
        }
        for iz in 0..nz_pad {
            let mut zfreq = iz as f32 / nz_pad as f32;
            if zfreq > 0.5 {
                zfreq = (zfreq as f64 - 1.0) as f32;
            }
            let zarg = -2.0 * pi as f64 * zfreq as f64 * dz as f64;
            let (zcos, zsin) = (zarg.cos() as f32, zarg.sin() as f32);
            for iy in 0..ny_pad {
                let mut yfreq = iy as f32 / ny_pad as f32;
                if yfreq > 0.5 {
                    yfreq = (yfreq as f64 - 1.0) as f32;
                }
                let yarg = -2.0 * pi as f64 * yfreq as f64 * dy as f64;
                let (ycos, ysin) = (yarg.cos() as f32, yarg.sin() as f32);
                let yzre = ycos * zcos - ysin * zsin;
                let yzim = ycos * zsin + ysin * zcos;
                let base = (iy * nx_dim + iz * nx_dim * ny_pad) as usize;
                for ix in 0..nx_fft {
                    let xind = 2 * ix;
                    let pre =
                        *temp.add(xind as usize) * yzre - *temp.add((xind + 1) as usize) * yzim;
                    let pim =
                        *temp.add((xind + 1) as usize) * yzre + *temp.add(xind as usize) * yzim;
                    let ind = base + xind as usize;
                    let real = *fft.add(ind);
                    let imag = *fft.add(ind + 1);
                    *fft.add(ind) = pre * real - pim * imag;
                    *fft.add(ind + 1) = pim * real + pre * imag;
                }
            }
        }
    }
}
/// `fourierReduceVolume` (`filtxcorr.c:2158`).
pub unsafe fn fourier_reduce_volume(
    input: *mut f32,
    nxi: i32,
    nyi: i32,
    nzi: i32,
    out: *mut f32,
    nxo: i32,
    nyo: i32,
    nzo: i32,
    dx: f32,
    dy: f32,
    dz: f32,
    temp: *mut f32,
) {
    unsafe {
        let xfac = nxi as f32 / nxo as f32;
        // `float dxy = -(redFac - 1) / (2. * redFac);` -- the `2. *` makes the divide double.
        let dxy = ((-(xfac - 1.0)) as f64 / (2.0 * xfac as f64)) as f32;
        let zfac = nzi as f32 / nzo as f32;
        let zd = ((-(zfac - 1.0)) as f64 / (2.0 * zfac as f64)) as f32;
        // `1. / pow(redFac * redFac * zRedFac, 1./3.)` -- the double `pow`, rounded once.
        let scale = (1.0 / ((xfac * xfac * zfac) as f64).powf(1.0 / 3.0)) as f32;
        let mut dst = out;
        for zloop in 0..2 {
            let (zs, ze) = if zloop == 0 {
                (0, nzo - nzo / 2)
            } else {
                (nzi - nzo / 2, nzi)
            };
            for iz in zs..ze {
                for yloop in 0..2 {
                    let (ys, ye) = if yloop == 0 {
                        (0, nyo - nyo / 2)
                    } else {
                        (nyi - nyo / 2, nyi)
                    };
                    for iy in ys..ye {
                        let base = iy * (nxi + 2) + iz * nyi * (nxi + 2);
                        for ix in 0..nxo + 2 {
                            *dst = *input.add((base + ix) as usize) * scale;
                            dst = dst.add(1);
                        }
                    }
                }
            }
        }
        if !temp.is_null() {
            fourier_shift_volume(
                out,
                nxo,
                nyo,
                nzo,
                dx / xfac + dxy,
                dy / xfac + dxy,
                dz / zfac + zd,
                temp,
            );
        }
    }
}
/// `fourierExpandVolume` (`filtxcorr.c:2226`).
pub unsafe fn fourier_expand_volume(
    input: *mut f32,
    nxi: i32,
    nyi: i32,
    nzi: i32,
    out: *mut f32,
    nxo: i32,
    nyo: i32,
    nzo: i32,
    dx: f32,
    dy: f32,
    dz: f32,
    temp: *mut f32,
) {
    unsafe {
        let xfac = nxo as f32 / nxi as f32;
        let zfac = nzo as f32 / nzi as f32;
        // `float dxy = (expFac - 1.) / (2. * expFac);` -- a wholly double expression.
        let dxy = ((xfac as f64 - 1.0) / (2.0 * xfac as f64)) as f32;
        let zd = ((zfac as f64 - 1.0) / (2.0 * zfac as f64)) as f32;
        core::ptr::write_bytes(out, 0, ((nxo + 2) * nyo * nzo) as usize);
        if !temp.is_null() {
            fourier_shift_volume(input, nxi, nyi, nzi, dx + dxy, dy + dxy, dz + zd, temp);
        }
        let mut src = input;
        for zloop in 0..2 {
            let (zs, ze) = if zloop == 0 {
                (0, nzi - nzi / 2)
            } else {
                (nzo - nzi / 2, nzo)
            };
            for iz in zs..ze {
                for yloop in 0..2 {
                    let (ys, ye) = if yloop == 0 {
                        (0, nyi - nyi / 2)
                    } else {
                        (nyo - nyi / 2, nyo)
                    };
                    for iy in ys..ye {
                        let base = iy * (nxo + 2) + iz * nyo * (nxo + 2);
                        for ix in 0..nxi + 2 {
                            *out.add((base + ix) as usize) = *src * xfac;
                            src = src.add(1);
                        }
                    }
                }
            }
        }
    }
}
pub unsafe fn fourier_ring_corr(
    a: *const f32,
    b: *const f32,
    nx: i32,
    ny: i32,
    corr: *mut f32,
    max: i32,
    delta: f32,
    temp: *mut f32,
) {
    unsafe {
        let prod = temp;
        let asum = prod.add(max as usize);
        let bsum = asum.add(max as usize);
        let counts = bsum.add(max as usize).cast::<i32>();
        for r in 0..max {
            *prod.add(r as usize) = 0.;
            *asum.add(r as usize) = 0.;
            *bsum.add(r as usize) = 0.;
            *counts.add(r as usize) = 0;
        }
        for iy in 0..ny {
            let mut yy = iy as f32 / ny as f32;
            if yy > 0.5 {
                yy = 1. - yy;
            }
            let base = iy * (nx + 2);
            for ix in (0..nx + 2).step_by(2) {
                let xx = ix as f32 / nx as f32 / 2.;
                let ring = ((xx * xx + yy * yy).sqrt() / delta) as i32;
                if ring < max && (ix > 0 || iy >= ny / 2) {
                    let ind = (base + ix) as usize;
                    let ar = *a.add(ind);
                    let ai = *a.add(ind + 1);
                    let br = *b.add(ind);
                    let bi = *b.add(ind + 1);
                    *counts.add(ring as usize) += 1;
                    *asum.add(ring as usize) += ar * ar + ai * ai;
                    *bsum.add(ring as usize) += br * br + bi * bi;
                    *prod.add(ring as usize) += ar * br + ai * bi;
                }
            }
        }
        for r in 0..max {
            let av = *asum.add(r as usize);
            let bv = *bsum.add(r as usize);
            *corr.add(r as usize) = if *counts.add(r as usize) != 0 && av * bv > 0. {
                *prod.add(r as usize) / (av * bv).sqrt()
            } else {
                0.
            };
        }
    }
}

// Fortran entry points in the same source unit.  The build system historically
// selects their trailing underscore spelling; Rust keeps the source function names
// snake-cased while preserving pointer/value calling conventions.
pub unsafe fn filterpart(
    a: *const f32,
    b: *mut f32,
    nx: *const i32,
    ny: *const i32,
    ctf: *const f32,
    delta: *const f32,
) {
    unsafe { xcorr_filter_part(a, b, *nx, *ny, ctf, *delta) }
}
pub unsafe fn xcorrpeakfindwidth(
    array: *mut f32,
    nxdim: *const i32,
    ny: *const i32,
    xpeak: *mut f32,
    ypeak: *mut f32,
    peak: *mut f32,
    width: *mut f32,
    width_min: *mut f32,
    max_peaks: *const i32,
    min_strength: *const f32,
) {
    unsafe {
        xcorr_peak_find_width(
            array,
            *nxdim,
            *ny,
            xpeak,
            ypeak,
            peak,
            width,
            width_min,
            *max_peaks,
            *min_strength,
        )
    }
}
pub unsafe fn getpeakfindtestlimits(
    nx: *const i32,
    ny: *const i32,
    xlo: *mut i32,
    xhi: *mut i32,
    ylo: *mut i32,
    yhi: *mut i32,
) {
    unsafe { get_peak_find_test_limits(*nx, *ny, xlo, xhi, ylo, yhi) }
}
pub unsafe fn xcorrpeakfind(
    array: *mut f32,
    nxdim: *const i32,
    ny: *const i32,
    xpeak: *mut f32,
    ypeak: *mut f32,
    peak: *mut f32,
    max_peaks: *const i32,
) {
    unsafe { xcorr_peak_find(array, *nxdim, *ny, xpeak, ypeak, peak, *max_peaks) }
}
pub unsafe fn storepeakfinderror(value: *const i32) {
    unsafe { store_peak_find_error(*value) }
}
pub fn getpeakfinderror() -> i32 {
    get_peak_find_error()
}
pub unsafe fn setpeakfindlimits(
    xlo: *const i32,
    xhi: *const i32,
    ylo: *const i32,
    yhi: *const i32,
    ellipse: *const i32,
) {
    unsafe { set_peak_find_limits(*xlo, *xhi, *ylo, *yhi, *ellipse) }
}
pub unsafe fn setpeakfindangle(angle: *const f32) {
    unsafe { set_peak_find_angle(*angle) }
}
pub unsafe fn cccoefficienttwopads(
    a: *const f32,
    b: *const f32,
    nxdim: *const i32,
    nx: *const i32,
    ny: *const i32,
    xpeak: *const f32,
    ypeak: *const f32,
    nx_pad_a: *const i32,
    ny_pad_a: *const i32,
    nx_pad_b: *const i32,
    ny_pad_b: *const i32,
    min_pixels: *const i32,
    nsum: *mut i32,
) -> f64 {
    unsafe {
        cc_coefficient_two_pads(
            a,
            b,
            *nxdim,
            *nx,
            *ny,
            *xpeak,
            *ypeak,
            *nx_pad_a,
            *ny_pad_a,
            *nx_pad_b,
            *ny_pad_b,
            *min_pixels,
            nsum,
        )
    }
}
pub unsafe fn cccoefficient(
    a: *const f32,
    b: *const f32,
    nxdim: *const i32,
    nx: *const i32,
    ny: *const i32,
    xpeak: *const f32,
    ypeak: *const f32,
    nx_pad: *const i32,
    ny_pad: *const i32,
    nsum: *mut i32,
) -> f64 {
    unsafe {
        xcorr_cc_coefficient(
            a, b, *nxdim, *nx, *ny, *xpeak, *ypeak, *nx_pad, *ny_pad, nsum,
        )
    }
}
pub unsafe fn slicegaussiankernel(mat: *mut f32, dim: *const i32, sigma: *const f32) {
    unsafe { slice_gaussian_kernel(mat, *dim, *sigma) }
}
pub unsafe fn scaledgaussiankernel(
    mat: *mut f32,
    dim: *mut i32,
    limit: *const i32,
    sigma: *const f32,
) {
    unsafe { scaled_gaussian_kernel(mat, dim, *limit, *sigma) }
}
pub unsafe fn applykernelfilter(
    a: *const f32,
    b: *mut f32,
    d: *const i32,
    x: *const i32,
    y: *const i32,
    mat: *const f32,
    k: *const i32,
) {
    unsafe { apply_kernel_filter(a, b, *d, *x, *y, mat, *k) }
}
pub unsafe fn fouriershiftimage(
    a: *mut f32,
    x: *const i32,
    y: *const i32,
    dx: *const f32,
    dy: *const f32,
    t: *mut f32,
) {
    unsafe { fourier_shift_image(a, *x, *y, *dx, *dy, t) }
}
pub unsafe fn fourierreduceimage(
    a: *mut f32,
    xi: *const i32,
    yi: *const i32,
    b: *mut f32,
    xo: *const i32,
    yo: *const i32,
    dx: *const f32,
    dy: *const f32,
    t: *mut f32,
) {
    unsafe { fourier_reduce_image(a, *xi, *yi, b, *xo, *yo, *dx, *dy, t) }
}
pub unsafe fn fourierexpandimage(
    a: *mut f32,
    xi: *const i32,
    yi: *const i32,
    b: *mut f32,
    xo: *const i32,
    yo: *const i32,
    dx: *const f32,
    dy: *const f32,
    t: *mut f32,
) {
    unsafe { fourier_expand_image(a, *xi, *yi, b, *xo, *yo, *dx, *dy, t) }
}
pub unsafe fn fouriershiftvolume(
    a: *mut f32,
    x: *const i32,
    y: *const i32,
    z: *const i32,
    dx: *const f32,
    dy: *const f32,
    dz: *const f32,
    t: *mut f32,
) {
    unsafe { fourier_shift_volume(a, *x, *y, *z, *dx, *dy, *dz, t) }
}
pub unsafe fn fourierreducevolume(
    a: *mut f32,
    xi: *const i32,
    yi: *const i32,
    zi: *const i32,
    b: *mut f32,
    xo: *const i32,
    yo: *const i32,
    zo: *const i32,
    dx: *const f32,
    dy: *const f32,
    dz: *const f32,
    t: *mut f32,
) {
    unsafe { fourier_reduce_volume(a, *xi, *yi, *zi, b, *xo, *yo, *zo, *dx, *dy, *dz, t) }
}
pub unsafe fn fourierexpandvolume(
    a: *mut f32,
    xi: *const i32,
    yi: *const i32,
    zi: *const i32,
    b: *mut f32,
    xo: *const i32,
    yo: *const i32,
    zo: *const i32,
    dx: *const f32,
    dy: *const f32,
    dz: *const f32,
    t: *mut f32,
) {
    unsafe { fourier_expand_volume(a, *xi, *yi, *zi, b, *xo, *yo, *zo, *dx, *dy, *dz, t) }
}
pub unsafe fn fourierringcorr(
    a: *const f32,
    b: *const f32,
    x: *const i32,
    y: *const i32,
    c: *mut f32,
    max: *const i32,
    d: *const f32,
    t: *mut f32,
) {
    unsafe { fourier_ring_corr(a, b, *x, *y, c, *max, *d, t) }
}
pub unsafe fn fouriercropsizes(
    s: *const i32,
    f: *const f32,
    p: *const f32,
    min: *const i32,
    l: *const i32,
    full: *mut i32,
    crop: *mut i32,
    actual: *mut f32,
) -> i32 {
    unsafe { fourier_crop_sizes(*s, *f, *p, *min, *l, full, crop, actual) }
}

#[cfg(test)]
mod tests {
    use super::{
        find_many_xcorr_peaks, find_spaced_xcorr_peaks, fourier_crop_sizes, fourier_expand_volume,
        fourier_reduce_volume, fourier_shift_volume, wrap_fft_slice, xcorr_peak_find_width,
    };

    #[test]
    fn fourier_shift_volume_arg_is_evaluated_in_double() {
        // `arg` is a double in the C and cos/sin are the double routines; the
        // shift amounts here are chosen so that an all-f32 evaluation differs.
        const SHIFT: [u32; 160] = [
            0xc0400000, 0x40124924, 0x40ed414d, 0x3fea7ba6, 0xc0570245, 0x411ef380, 0xc0a72e20,
            0x4068aa38, 0xc04aa27a, 0xc00328b7, 0x40c8bd90, 0x40ad752d, 0xc0e2bbb6, 0x410fa33b,
            0xc0db24bb, 0x400fafb6, 0xc045e345, 0xc025b854, 0xc0bdb1a0, 0x40d922fa, 0x405250d2,
            0x40af5263, 0xc1016573, 0x3d85b820, 0xc03bfda0, 0xc05b171a, 0xc10f8cb7, 0x40795910,
            0xc0c0d606, 0x404656fa, 0x41008777, 0x40818b0c, 0x3fdf31d0, 0x409ae40d, 0x4126db6d,
            0x3fedb6db, 0x40e57dba, 0x3fa363fa, 0xc0587ea2, 0x411542c5, 0xc0583b74, 0x409a30ca,
            0xc062d87e, 0xbf9855a9, 0x3fce29fc, 0x40f725f8, 0xc12b2cca, 0x3ff305a2, 0xc0c1f5dd,
            0xc0361f6c, 0xc05f93e6, 0xbfce1411, 0xc00c49cf, 0x4104951c, 0xc13a6ddb, 0xbffdb734,
            0xc0b2e61c, 0xc0a29feb, 0x3f002976, 0xc0857dec, 0xc0c20422, 0x40e23782, 0xc06c2c1a,
            0x40ac4f44, 0xc086022a, 0xc0eac7d0, 0x3fb524be, 0xc0910e82, 0xc0f7c52a, 0xc0cf9ad0,
            0x40d46c57, 0xc0073aa9, 0x3ebddd38, 0x4115b6e9, 0xc05b339c, 0x4086da42, 0x4048b0d6,
            0x41277825, 0x3ffe5e32, 0x40e7f3a5, 0xc10af880, 0x40b112fe, 0xc0c5d085, 0xbf018222,
            0xbf812a39, 0xc06722d4, 0xc0cec5d9, 0x409f27de, 0xc0e14b76, 0xc10c4f36, 0xc0d1ffd9,
            0xc0223508, 0xbf1d767c, 0xc07a76e9, 0xc10b0735, 0x3fdeb606, 0xc0be319f, 0x400af814,
            0x3fe0e643, 0xc0f6c2d1, 0xbb38b700, 0xc08cd3a1, 0xc1154632, 0xc0170bf4, 0xc0d2ffe9,
            0xbf8f8680, 0x4091cb4f, 0xc0f15f6a, 0x409b4d51, 0xbf9e3bdf, 0x40dfcf3b, 0x40f77ea6,
            0x409c1644, 0x40a85970, 0xc1053d9c, 0x40a16d5a, 0xc0b56f54, 0xbf5498c4, 0xbed40454,
            0xc06e8573, 0x40a98d04, 0xc0b580bd, 0x40ec17fe, 0x40f6ee50, 0x3f655246, 0x40cef6d0,
            0xc04e31c8, 0x40016c6f, 0x40c1f302, 0x40bb7bfd, 0x4097ab23, 0x4129e2ef, 0xbf942943,
            0x40e93f83, 0xc06f3442, 0x3fdbcf26, 0x4048b9a5, 0x4109b698, 0x403bf1de, 0x40b7f6d2,
            0xc06f3442, 0x40ebf8bb, 0xc08f3406, 0x3f967488, 0xbf683f5b, 0x411e62f0, 0xbe8be748,
            0x40dba29c, 0xc10a5514, 0xc0456d35, 0x40a806e0, 0x3f1b2ed7, 0x40d74458, 0xc1065816,
            0x4096367a, 0xc0b6fcce, 0x40daa189, 0x40ee7de9, 0x3ef7c798, 0x40c0bc0e,
        ];
        let dx: f32 = -0.5 + 1.0 / 41.0;
        let dz: f32 = -0.5 + 2.0 / 41.0;
        let mut fft = (0..160)
            .map(|i| ((i * 37) % 97) as f32 / 7.0 - 3.0)
            .collect::<Vec<f32>>();
        let mut temp = vec![0_f32; 16];
        unsafe { fourier_shift_volume(fft.as_mut_ptr(), 8, 4, 4, dx, dx, dz, temp.as_mut_ptr()) };
        assert_eq!(
            fft.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(),
            SHIFT.to_vec()
        );
    }

    #[test]
    fn fourier_expand_volume_matches_native_bit_patterns() {
        // Captured from the native libcfshr `fourierExpandVolume`.
        const EXPAND: [u32; 160] = [
            0xc0800000, 0x40b45d18, 0x401ffe3d, 0x410692cf, 0x411e6455, 0x40da0bc7, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x403504f1, 0xc0941b53, 0x409c2c48, 0xc0e0ffa1, 0x409d1741,
            0xc13745d1, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x408ffe20, 0x40da0bc7, 0x412cda4b, 0x4039a17a, 0xc0000002, 0xc091745c,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x3f8ba2dd, 0xc0f45d17, 0xbe7774a0, 0xc1382d23,
            0x40da0bc5, 0xc05e2900, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ];
        let mut input = (0..24)
            .map(|i| ((i * 53) % 89) as f32 / 11.0 - 2.0)
            .collect::<Vec<f32>>();
        let mut out = vec![0_f32; 10 * 4 * 4];
        let mut temp = vec![0_f32; 16];
        unsafe {
            fourier_expand_volume(
                input.as_mut_ptr(),
                4,
                2,
                2,
                out.as_mut_ptr(),
                8,
                4,
                4,
                0.,
                0.,
                0.,
                temp.as_mut_ptr(),
            )
        };
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(),
            EXPAND.to_vec()
        );
    }

    #[test]
    fn fourier_crop_sizes_matches_native_sizes_and_factors() {
        // (size, factor bits, return code, fullPadSize, cropPadSize, actualFac bits)
        // captured from the native libcfshr `fourierCropSizes`.
        const CASES: [(i32, u32, i32, i32, i32, u32); 50] = [
            (64, 0x40000000, 0, 96, 48, 0x40000000),
            (64, 0x3fc00000, 0, 96, 64, 0x3fc00000),
            (64, 0x3f000000, 0, 96, 192, 0x3f000000),
            (64, 0x3f2aaaab, 0, 96, 144, 0x3f2aaaab),
            (64, 0x40400000, 0, 96, 32, 0x40400000),
            (64, 0x40200000, 0, 100, 40, 0x40200000),
            (64, 0x3eaaaaab, 0, 96, 288, 0x3eaaaaab),
            (64, 0x3f99999a, 0, 96, 80, 0x3f99999a),
            (64, 0x3f4ccccd, 0, 96, 120, 0x3f4ccccd),
            (64, 0x40e00000, 0, 98, 14, 0x40e00000),
            (48, 0x40000000, 0, 80, 40, 0x40000000),
            (48, 0x3fc00000, 0, 84, 56, 0x3fc00000),
            (48, 0x3f000000, 0, 80, 160, 0x3f000000),
            (48, 0x3f2aaaab, 0, 80, 120, 0x3f2aaaab),
            (48, 0x40400000, 0, 84, 28, 0x40400000),
            (48, 0x40200000, 0, 80, 32, 0x40200000),
            (48, 0x3eaaaaab, 0, 80, 240, 0x3eaaaaab),
            (48, 0x3f99999a, 0, 84, 70, 0x3f99999a),
            (48, 0x3f4ccccd, 0, 80, 100, 0x3f4ccccd),
            (48, 0x40e00000, 0, 84, 12, 0x40e00000),
            (5, 0x40000000, 0, 40, 20, 0x40000000),
            (5, 0x3fc00000, 0, 42, 28, 0x3fc00000),
            (5, 0x3f000000, 0, 40, 80, 0x3f000000),
            (5, 0x3f2aaaab, 0, 40, 60, 0x3f2aaaab),
            (5, 0x40400000, 0, 42, 14, 0x40400000),
            (5, 0x40200000, 0, 40, 16, 0x40200000),
            (5, 0x3eaaaaab, 0, 40, 120, 0x3eaaaaab),
            (5, 0x3f99999a, 0, 48, 40, 0x3f99999a),
            (5, 0x3f4ccccd, 0, 40, 50, 0x3f4ccccd),
            (5, 0x40e00000, 0, 42, 6, 0x40e00000),
            (100, 0x40000000, 0, 132, 66, 0x40000000),
            (100, 0x3fc00000, 0, 132, 88, 0x3fc00000),
            (100, 0x3f000000, 0, 132, 264, 0x3f000000),
            (100, 0x3f2aaaab, 0, 132, 198, 0x3f2aaaab),
            (100, 0x40400000, 0, 132, 44, 0x40400000),
            (100, 0x40200000, 0, 140, 56, 0x40200000),
            (100, 0x3eaaaaab, 0, 132, 396, 0x3eaaaaab),
            (100, 0x3f99999a, 0, 132, 110, 0x3f99999a),
            (100, 0x3f4ccccd, 0, 144, 180, 0x3f4ccccd),
            (100, 0x40e00000, 0, 140, 20, 0x40e00000),
            (257, 0x40000000, 0, 300, 150, 0x40000000),
            (257, 0x3fc00000, 0, 294, 196, 0x3fc00000),
            (257, 0x3f000000, 0, 294, 588, 0x3f000000),
            (257, 0x3f2aaaab, 0, 300, 450, 0x3f2aaaab),
            (257, 0x40400000, 0, 294, 98, 0x40400000),
            (257, 0x40200000, 0, 300, 120, 0x40200000),
            (257, 0x3eaaaaab, 0, 294, 882, 0x3eaaaaab),
            (257, 0x3f99999a, 0, 300, 250, 0x3f99999a),
            (257, 0x3f4ccccd, 0, 312, 390, 0x3f4ccccd),
            (257, 0x40e00000, 0, 294, 42, 0x40e00000),
        ];
        for (size, factor, code, full_expect, crop_expect, actual_expect) in CASES {
            let (mut full, mut crop, mut actual) = (-1, -1, -1_f32);
            let got = unsafe {
                fourier_crop_sizes(
                    size,
                    f32::from_bits(factor),
                    0.01,
                    16,
                    15,
                    &mut full,
                    &mut crop,
                    &mut actual,
                )
            };
            assert_eq!(
                (got, full, crop, actual.to_bits()),
                (code, full_expect, crop_expect, actual_expect),
                "size {} factor {:08x}",
                size,
                factor
            );
        }
    }

    #[test]
    fn fourier_shift_and_reduce_volume_match_native_bit_patterns() {
        // Captured from the native libcfshr `fourierShiftVolume` and
        // `fourierReduceVolume` (reference build) for these exact inputs.  The
        // C uses `float pi = 3.141593;` and a *double* `arg`, so the phase
        // factors are one ulp away from an all-f32 evaluation.
        const SHIFT: [u32; 160] = [
            0xc0400000, 0x40124924, 0x40f1245f, 0x3f9ed6c6, 0xbfe08852, 0x41256e0c, 0xc0872f59,
            0x4098570a, 0xc0695db0, 0xbf76b362, 0x40e58673, 0xc0850cff, 0x40dd2364, 0x4111cd51,
            0x3f828c72, 0x40e44c5e, 0xc037e29d, 0x403521e2, 0x40d5ed3e, 0x40c14e7b, 0x3f4c8f1c,
            0xc0cad5cd, 0x40d8adb0, 0x408d858f, 0x3f6720cf, 0x408d6c62, 0x411c78f2, 0x3e3da1b0,
            0x40d723a1, 0xbf58f7c0, 0xbf99c916, 0x410ea34e, 0xc06f3443, 0x40624632, 0x4051cd0a,
            0x4121297f, 0x4032db1b, 0x40d742d8, 0xc117a160, 0x403c5076, 0xbf93a321, 0x40b8aa80,
            0xc06bde81, 0x3f214d80, 0x40b12c64, 0x40b3db44, 0xc0eb6fd8, 0x40ffdc12, 0xc0cd3d8c,
            0x3ff639ab, 0xbfa90818, 0x4067377a, 0x41087b84, 0x3f5713da, 0x3f567d80, 0x413ca1b8,
            0xc044dcfa, 0x40dcd2b2, 0xc081492f, 0x3f939da4, 0x40f431e6, 0xc0aad62e, 0x4091e53e,
            0xc0957f50, 0x405f1007, 0x40f640fa, 0xbfbd75e4, 0x4090641f, 0x41025526, 0x40bf29d1,
            0x40c1f304, 0x405bcf2a, 0xc0c2360b, 0x40e43632, 0xc0a97c0e, 0x3f997e83, 0xc04bb1be,
            0x41273e15, 0xbfe4ba21, 0x40e99d5a, 0xbfe593dc, 0x4122426f, 0xc0868d21, 0x4091e7fb,
            0xc0646afc, 0xbf933de7, 0x3f9cc136, 0x4100fc7b, 0xc12e5429, 0x40323e14, 0x3f64495a,
            0x40df4d02, 0xc03cd553, 0x40292fc8, 0x40d54b06, 0x40badf6a, 0x40b2df0e, 0x403dcf10,
            0xc0d29ee9, 0x408c533c, 0x3f375864, 0x408af308, 0x4119f944, 0x3d698e10, 0x40d38c23,
            0xbf81fc51, 0xbfa14aca, 0x410b8b6c, 0xc06f3443, 0x4055581c, 0x40558de6, 0x411e119c,
            0x403a0a14, 0x40d1e2bb, 0xc11521b3, 0x40341c94, 0xc0ae2faa, 0xbfe53970, 0xbd849fa0,
            0xc06ff40c, 0x40e443f6, 0xc043d00e, 0x408db6db, 0x411b6db7, 0xbf45c09a, 0x40cf78f1,
            0xc061c774, 0x3fb61a75, 0x40aea381, 0x40cd9787, 0x41302bae, 0x406f343f, 0x40a028bf,
            0x40ad914d, 0xbf260caa, 0x4081f9b1, 0x4110af66, 0x3fbb7fad, 0x40ce6674, 0xbe8850b8,
            0x41043e84, 0xbe383700, 0x4059dde7, 0x40488dd2, 0x4106f715, 0xc0a84d4e, 0x40a9c973,
            0xc08b99e6, 0x40838572, 0x4103545e, 0xbf83fe18, 0x40a5e490, 0x41061603, 0x40d7e8e6,
            0x40c1f304, 0x4087c3b8, 0xc0c9b7c4, 0x40fcf544, 0xc0b7da01, 0x3fef8045,
        ];
        const REDUCE: [u32; 24] = [
            0xbf800000, 0x3fb45d18, 0xbf85bfe3, 0x3ff6e1b9, 0xbfda0bca, 0x401e6454, 0x3f941b50,
            0x40183887, 0xbfe75c95, 0xbdb39af0, 0xc01a2e8b, 0xb4c0ba2f, 0x3f941b52, 0x3d83a698,
            0x3f204f3d, 0x3fc182c3, 0xbf45d179, 0x401a2e8b, 0xc02e8ba2, 0x3ea2e8ad, 0x3db11290,
            0x3f9bb9b4, 0xbfbd3f5c, 0x3f941b4d,
        ];

        let mut fft = (0..160)
            .map(|i| ((i * 37) % 97) as f32 / 7.0 - 3.0)
            .collect::<Vec<f32>>();
        let mut temp = vec![0_f32; 16];
        unsafe {
            fourier_shift_volume(
                fft.as_mut_ptr(),
                8,
                4,
                4,
                -0.375,
                0.3125,
                -0.1875,
                temp.as_mut_ptr(),
            )
        };
        let got = fft.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        assert_eq!(got, SHIFT.to_vec());

        let mut input = (0..160)
            .map(|i| ((i * 53) % 89) as f32 / 11.0 - 2.0)
            .collect::<Vec<f32>>();
        let mut out = vec![0_f32; 6 * 2 * 2];
        unsafe {
            fourier_reduce_volume(
                input.as_mut_ptr(),
                8,
                4,
                4,
                out.as_mut_ptr(),
                4,
                2,
                2,
                0.,
                0.,
                0.,
                temp.as_mut_ptr(),
            )
        };
        let got = out.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        assert_eq!(got, REDUCE.to_vec());
    }

    #[test]
    fn wrap_fft_slice_matches_even_and_odd_source_permutations() {
        let mut even = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let mut temporary = vec![0.; 4];
        unsafe { wrap_fft_slice(even.as_mut_ptr(), temporary.as_mut_ptr(), 2, 4, 0) };
        assert_eq!(
            even,
            vec![
                8., 9., 10., 11., 12., 13., 14., 15., 0., 1., 2., 3., 4., 5., 6., 7.
            ]
        );

        let mut odd = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        unsafe { wrap_fft_slice(odd.as_mut_ptr(), temporary.as_mut_ptr(), 2, 3, 0) };
        assert_eq!(odd, vec![8., 9., 10., 11., 0., 1., 2., 3., 4., 5., 6., 7.]);
        unsafe { wrap_fft_slice(odd.as_mut_ptr(), temporary.as_mut_ptr(), 2, 3, 1) };
        assert_eq!(odd, (0..12).map(|value| value as f32).collect::<Vec<_>>());
    }

    #[test]
    fn peak_find_uses_eight_neighbours_subpixel_fit_and_widths() {
        // A 6 x 6 real image has padded C stride 8.  The central peak's X
        // shoulders make the source parabola move it +0.1 pixel.
        let mut array = vec![0_f32; 8 * 6];
        for y in 0..6 {
            for x in 0..6 {
                array[x + y * 8] = 1.;
            }
        }
        array[2 + 3 * 8] = 10.;
        array[1 + 3 * 8] = 4.;
        array[3 + 3 * 8] = 6.;
        array[2 + 2 * 8] = 5.;
        array[2 + 4 * 8] = 5.;
        let (mut xp, mut yp, mut value, mut width, mut width_min) = (0., 0., 0., 0., 0.);
        unsafe {
            xcorr_peak_find_width(
                array.as_mut_ptr(),
                8,
                6,
                &mut xp,
                &mut yp,
                &mut value,
                &mut width,
                &mut width_min,
                1,
                0.,
            )
        };
        assert_eq!(value, 10.);
        assert!((xp - 2.1).abs() < 1.0e-5);
        assert_eq!(yp, 3.);
        assert!(width > 0. && width_min > 0.);
    }

    #[test]
    fn many_and_spaced_peak_selection_returns_interpolated_local_maxima() {
        let mut array = vec![0_f32; 10 * 8];
        array[3 + 3 * 10] = 9.;
        array[6 + 5 * 10] = 7.;
        let (mut xp, mut yp, mut peaks, mut count) = ([0.; 2], [0.; 2], [0.; 2], 0);
        unsafe {
            assert_eq!(
                find_many_xcorr_peaks(
                    array.as_mut_ptr(),
                    10,
                    8,
                    0,
                    0,
                    xp.as_mut_ptr(),
                    yp.as_mut_ptr(),
                    peaks.as_mut_ptr(),
                    2,
                    3,
                    &mut count
                ),
                0
            )
        };
        assert_eq!(count, 2);
        assert_eq!(peaks, [9., 7.]);
        unsafe {
            assert_eq!(
                find_spaced_xcorr_peaks(
                    array.as_mut_ptr(),
                    10,
                    0,
                    8,
                    0,
                    7,
                    xp.as_mut_ptr(),
                    yp.as_mut_ptr(),
                    peaks.as_mut_ptr(),
                    2,
                    2.,
                    &mut count,
                    0.
                ),
                0
            )
        };
        assert_eq!(count, 2);
        assert_eq!(peaks, [9., 7.]);
    }
}

#[cfg(test)]
mod find_many_peaks_reference {
    use super::*;

    /// Values captured from a C driver linked directly against the reference
    /// `libcfshr.so` at /tmp/imod-reference-build/buildlib, calling
    /// `findManyXCorrPeaks` on a deterministic array full of ties (values 0..3
    /// from a fixed LCG), which is where the source's asymmetric `>` / `>=`
    /// peak test and its threshold/repack ordering actually show.
    #[test]
    fn find_many_xcorr_peaks_matches_the_reference_driver() {
        let (nxdim, ny) = (34_i32, 32_i32);
        let mut a = vec![0.0_f32; (nxdim * ny) as usize];
        let mut s: u32 = 12345;
        for j in 0..ny {
            for i in 0..nxdim {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                // 0..3 gives adjacent plateaus, so the source's asymmetric
                // `>` / `>=` peak test actually decides the outcome.
                a[(i + j * nxdim) as usize] = ((s >> 18) & 0x3) as f32;
            }
        }
        let cases = [(-1_i32, -1_i32), (0, 0), (2, 2), (-3, -1)];
        let expected: [&[(f32, f32, f32)]; 4] = [
            &[
                (-6.0, -2.75, 3.0),
                (-0.25, -2.0, 3.0),
                (7.5, -1.83333397, 3.0),
                (11.0, -2.16666603, 3.0),
                (-12.25, -2.0, 3.0),
                (-3.0, -1.5, 3.0),
                (2.0, -1.0, 3.0),
                (14.5, -1.10000038, 3.0),
            ],
            &[
                (21.75, 30.0, 3.0),
                (31.0, 30.5, 3.0),
                (24.8999996, 20.5, 3.0),
                (9.5, 21.5, 3.0),
                (12.75, 22.5, 3.0),
                (6.75, 23.166666, 3.0),
                (10.833333, 23.166666, 3.0),
                (20.0, 23.0, 3.0),
            ],
            &[
                (19.8999996, 25.833334, 3.0),
                (23.0, 26.5, 3.0),
                (27.0, 26.166666, 3.0),
                (6.5, 27.833334, 3.0),
                (12.1000004, 28.1000004, 3.0),
                (19.25, 28.5, 3.0),
                (6.0, 16.75, 3.0),
                (11.0, 17.25, 3.0),
            ],
            &[
                (7.5, -1.83333397, 3.0),
                (11.0, -2.16666603, 3.0),
                (-10.25, -2.0, 3.0),
                (-0.5, -1.5, 3.0),
                (2.0, -1.0, 3.0),
                (14.5, -1.10000038, 3.0),
                (14.5, -1.10000038, 3.0),
                (-13.5, -0.5, 3.0),
            ],
        ];
        for (index, &(ix_off, iy_off)) in cases.iter().enumerate() {
            let mut xp = [0.0_f32; 64];
            let mut yp = [0.0_f32; 64];
            let mut pk = [0.0_f32; 64];
            let mut n = 0_i32;
            let rc = unsafe {
                find_many_xcorr_peaks(
                    a.as_mut_ptr(),
                    nxdim,
                    ny,
                    ix_off,
                    iy_off,
                    xp.as_mut_ptr(),
                    yp.as_mut_ptr(),
                    pk.as_mut_ptr(),
                    8,
                    32,
                    &mut n,
                )
            };
            assert_eq!(rc, 0, "case {index}");
            if expected[index].is_empty() {
                continue;
            }
            assert_eq!(n as usize, expected[index].len(), "case {index} count");
            for (k, &(ex, ey, ep)) in expected[index].iter().enumerate() {
                assert_eq!(xp[k].to_bits(), ex.to_bits(), "case {index} x[{k}]");
                assert_eq!(yp[k].to_bits(), ey.to_bits(), "case {index} y[{k}]");
                assert_eq!(pk[k].to_bits(), ep.to_bits(), "case {index} peak[{k}]");
            }
        }
    }
}
