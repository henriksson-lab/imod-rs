//! Translation of `IMOD/libcfshr/filtxcorr.c`.
//!
//! The public routines retain the C source's raw-pointer contracts.  This is
//! intentional: callers pass padded FFT lines and aliased work buffers.
#![allow(dead_code, unused_variables)]

use core::ffi::c_char;
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};

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
        let mid = (dim - 1) as f32 / 2.;
        let mut sum = 0.;
        for y in 0..dim {
            for x in 0..dim {
                let v = (-((x as f32 - mid).powi(2) + (y as f32 - mid).powi(2)) / (sigma * sigma))
                    .exp();
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
        *dim = (2. * sigma.ceil() + 1.) as i32;
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

pub unsafe fn fourier_shift_image(
    fft: *mut f32,
    nx: i32,
    ny: i32,
    dx: f32,
    dy: f32,
    temp: *mut f32,
) {
    unsafe {
        if dx == 0. && dy == 0. {
            return;
        }
        let nxfft = nx / 2 + 1;
        let ndim = nx + 2;
        for x in 0..nxfft {
            let arg = -2. * core::f32::consts::PI * 0.5 * x as f32 / (nxfft - 1) as f32 * dx;
            *temp.add((2 * x) as usize) = arg.cos();
            *temp.add((2 * x + 1) as usize) = arg.sin();
        }
        for y in 0..ny {
            let mut fy = y as f32 / ny as f32;
            if fy > 0.5 {
                fy -= 1.;
            }
            let arg = -2. * core::f32::consts::PI * fy * dy;
            let (c, s) = (arg.cos(), arg.sin());
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
            let d = -(fac - 1.) / (2. * fac);
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
        core::ptr::write_bytes(out, 0, ((nxo + 2) * nyo) as usize);
        let fac = nxo as f32 / nxi as f32;
        if !temp.is_null() {
            let d = (fac - 1.) / (2. * fac);
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
        let usefac = if factor < 1. { 1. / factor } else { factor };
        let mut denom = 0;
        let mut numer = 0;
        for d in [1, 2, 3, 4, 5, 6, 8, 10] {
            let n = (usefac * d as f32).round() as i32;
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
            *actual = 1. / *actual;
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
        if (max_grow as f32) <= 1.05 * max_peaks as f32 {
            return 1;
        }
        let nx = nxdim - 2;
        let xs = (ix_offset + 1).max(0);
        let mut xe = (nxdim - ix_offset - 1).min(nxdim);
        if ix_offset < -1 {
            xe = nxdim - 2;
        }
        let ys = (iy_offset + 1).max(0);
        let ye = (ny - iy_offset - 1).min(ny);
        let mut found: Vec<(f32, i32, i32)> = Vec::new();
        for iy in ys..ye {
            for ix in xs..xe {
                let v = *array.add((ix + iy * nxdim) as usize);
                let edge = iy_offset < 0 && (iy == 0 || iy == ny - 1 || ix == 0 || ix == xe - 1);
                let xl = (ix + xe - 1) % xe;
                let xr = (ix + 1) % xe;
                let yu = (iy + ny - 1) % ny;
                let yd = (iy + 1) % ny;
                let local = if edge {
                    v >= *array.add((xl + iy * nxdim) as usize)
                        && v >= *array.add((xr + iy * nxdim) as usize)
                        && v >= *array.add((xl + yu * nxdim) as usize)
                        && v >= *array.add((ix + yu * nxdim) as usize)
                        && v >= *array.add((xr + yu * nxdim) as usize)
                        && v >= *array.add((xl + yd * nxdim) as usize)
                        && v >= *array.add((ix + yd * nxdim) as usize)
                        && v >= *array.add((xr + yd * nxdim) as usize)
                } else {
                    v > *array.add((ix - 1 + (iy - 1) * nxdim) as usize)
                        && v >= *array.add((ix + 1 + (iy - 1) * nxdim) as usize)
                        && v > *array.add((ix - 1 + iy * nxdim) as usize)
                        && v >= *array.add((ix + 1 + iy * nxdim) as usize)
                        && v > *array.add((ix - 1 + (iy + 1) * nxdim) as usize)
                        && v >= *array.add((ix + 1 + (iy + 1) * nxdim) as usize)
                        && v > *array.add((ix + (iy - 1) * nxdim) as usize)
                        && v >= *array.add((ix + (iy + 1) * nxdim) as usize)
                };
                if local {
                    found.push((v, ix, iy));
                }
            }
        }
        found.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
        *num_found = found.len().min(max_peaks as usize) as i32;
        for (i, (v, ix, iy)) in found.into_iter().take(max_peaks as usize).enumerate() {
            *peak.add(i) = v;
            let xl = (ix + xe - 1) % xe;
            let xr = (ix + 1) % xe;
            let yu = (iy + ny - 1) % ny;
            let yd = (iy + 1) % ny;
            let edge = iy_offset < 0 && (ix == 0 || iy == 0 || ix == xe - 1 || iy == ny - 1);
            let mut px = ix as f32
                + if edge {
                    parabolic_fit_position(
                        *array.add((xl + iy * nxdim) as usize),
                        v,
                        *array.add((xr + iy * nxdim) as usize),
                    ) as f32
                } else {
                    parabolic_fit_position(
                        *array.add((ix - 1 + iy * nxdim) as usize),
                        v,
                        *array.add((ix + 1 + iy * nxdim) as usize),
                    ) as f32
                };
            let mut py = iy as f32
                + if edge {
                    parabolic_fit_position(
                        *array.add((ix + yu * nxdim) as usize),
                        v,
                        *array.add((ix + yd * nxdim) as usize),
                    ) as f32
                } else {
                    parabolic_fit_position(
                        *array.add((ix + (iy - 1) * nxdim) as usize),
                        v,
                        *array.add((ix + (iy + 1) * nxdim) as usize),
                    ) as f32
                };
            if iy_offset < 0 {
                if px > nx as f32 / 2. {
                    px -= nx as f32;
                }
                if py > ny as f32 / 2. {
                    py -= ny as f32;
                }
            }
            *xpeak.add(i) = px;
            *ypeak.add(i) = py;
        }
        0
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
        if dx == 0. && dy == 0. && dz == 0. {
            return;
        }
        let nx_fft = nx_pad / 2 + 1;
        let nx_dim = nx_pad + 2;
        for ix in 0..nx_fft {
            let arg = -2.0 * core::f32::consts::PI * 0.5 * ix as f32 / (nx_fft - 1) as f32 * dx;
            *temp.add((2 * ix) as usize) = arg.cos();
            *temp.add((2 * ix + 1) as usize) = arg.sin();
        }
        for iz in 0..nz_pad {
            let mut zfreq = iz as f32 / nz_pad as f32;
            if zfreq > 0.5 {
                zfreq -= 1.0;
            }
            let zarg = -2.0 * core::f32::consts::PI * zfreq * dz;
            let (zcos, zsin) = (zarg.cos(), zarg.sin());
            for iy in 0..ny_pad {
                let mut yfreq = iy as f32 / ny_pad as f32;
                if yfreq > 0.5 {
                    yfreq -= 1.0;
                }
                let yarg = -2.0 * core::f32::consts::PI * yfreq * dy;
                let (ycos, ysin) = (yarg.cos(), yarg.sin());
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
        let zfac = nzi as f32 / nzo as f32;
        let scale = 1.0 / (xfac * xfac * zfac).powf(1.0 / 3.0);
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
            let dxy = -(xfac - 1.0) / (2.0 * xfac);
            let zd = -(zfac - 1.0) / (2.0 * zfac);
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
        core::ptr::write_bytes(out, 0, ((nxo + 2) * nyo * nzo) as usize);
        let xfac = nxo as f32 / nxi as f32;
        let zfac = nzo as f32 / nzi as f32;
        if !temp.is_null() {
            let dxy = (xfac - 1.0) / (2.0 * xfac);
            let zd = (zfac - 1.0) / (2.0 * zfac);
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
        find_many_xcorr_peaks, find_spaced_xcorr_peaks, wrap_fft_slice, xcorr_peak_find_width,
    };

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
