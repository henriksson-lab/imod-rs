//! Translation of `IMOD/libcfshr/filtxcorr.c`.
//!
//! Every buffer is a slice here.  Two of the source's contracts do not survive
//! that on their own and are named instead of being lost: `XCorrFilterPart`'s
//! `array` "can be the same as [fft]" (`filtxcorr.c:265`), which [`FilterIn`]
//! spells; and `fourierRingCorr`'s `temp`, which the source both retypes as
//! `int *` partway along and allows to be the same array as `ringCorrs`
//! (`filtxcorr.c:2287`, `:2296-2299`) -- that one is a local allocation here,
//! documented at the point of deviation.
#![allow(dead_code, unused_variables)]

use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, num_omp_threads};
use rayon::ThreadPoolBuilder;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

use super::robuststat::{rs_set_sort_index_offset, rs_sort_indexed_floats};

// C file-scope state for the peak-finder API (`filtxcorr.c:443-450`).
/// `MAX_SPCP_THREADS` (`filtxcorr.c:1254`).
pub const MAX_SPCP_THREADS: i32 = 8;

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
pub fn niceframe(num: &i32, idnum: &i32, limit: &i32) -> i32 {
    nice_frame(*num, *idnum, *limit)
}

/// `XCorrSetCTFnoScl` (`filtxcorr.c:159`).
pub fn xcorr_set_ctf_no_scl(
    sigma1: f32,
    sigma2: f32,
    radius1: f32,
    radius2: f32,
    ctf: &mut [f32],
    nx: i32,
    ny: i32,
    delta: &mut f32,
    nsize_out: &mut i32,
) {
    // `double beta1, beta2, alpha;` -- these are doubles in the source, and
    // every `exp` argument they appear in is therefore a double expression.
    let mut alpha = 0.0f64;
    let mut beta1 = 0.0f64;
    let mut delmax = 0.0f32;

    *delta = 0.;
    if sigma1 == 0. && sigma2 == 0. {
        return;
    }

    let mut nsize = 1024i32;
    if 2 * nx > nsize {
        nsize = 2 * nx;
    }
    if 2 * ny > nsize {
        nsize = 2 * ny;
    }
    if nsize > 8192 {
        nsize = 8192;
    }

    let asize = nsize as f32;
    nsize += 1;
    if (if sigma1 >= 0. { sigma1 } else { -sigma1 }) > 1.0e-6 {
        alpha = -0.5 / (sigma1 * sigma1) as f64;
    }
    if (if sigma2 >= 0. { sigma2 } else { -sigma2 }) > 1.0e-6 {
        beta1 = -0.5 / (sigma2 * sigma2) as f64;
    }
    let beta2 = beta1;

    /* Yes, delta is twice as big as it should be and values are generated
    out to 1.41.  Maybe it should have been 0.71/asize, but since it goes
    past 0.866 it is good for 3D filtering too. */
    *delta = 1.0f32 / (0.71f32 * asize);
    let (radius1p, radius1n) = if radius1 >= 0. {
        (radius1, 0.0f32)
    } else {
        (0.0f32, -radius1)
    };

    /* For negative sigma1, find the maximum to allow scaling to maximum of 1 */
    // `s` is accumulated by `delta` in each of the source's four loops, never
    // recomputed as `j * delta`; the two differ by an ulp that grows with j.
    if sigma1 < -1.0e-6 {
        let mut s = 0.0f32;
        for _j in 0..nsize {
            let ssqrd = s * s;
            let deltmp = (ssqrd as f64 * (alpha * ssqrd as f64).exp()) as f32;
            if delmax < deltmp {
                delmax = deltmp;
            }
            s += *delta;
        }
    }

    let mut s = 0.0f32;
    for j in 0..nsize as usize {
        if s < radius1p {
            ctf[j] = (beta1 * (s - radius1p) as f64 * (s - radius1p) as f64).exp() as f32;
        } else if s > radius2 {
            ctf[j] = (beta2 * (s - radius2) as f64 * (s - radius2) as f64).exp() as f32;
        } else {
            ctf[j] = 1.0;
        }

        s += *delta;
        if sigma2 < -1.0e-6 {
            ctf[j] = 1.0f32 - ctf[j];
        }
    }
    if sigma1 > 1.0e-6 {
        let mut s = 0.0f32;
        for j in 0..nsize as usize {
            if s < radius1n {
                ctf[j] = 0.;
            } else {
                // `(float)exp(...)` is rounded to float before the subtraction.
                ctf[j] = ctf[j]
                    * (1.0f32
                        - (alpha * (s - radius1n) as f64 * (s - radius1n) as f64).exp() as f32);
            }
            s += *delta;
        }
    } else if sigma1 < -1.0e-6 {
        let mut s = 0.0f32;
        for j in 0..nsize as usize {
            let ssqrd = s * s;
            ctf[j] =
                ((ctf[j] * ssqrd) as f64 * (alpha * ssqrd as f64).exp() / delmax as f64) as f32;
            s += *delta;
        }
    }

    // Set small numbers to zero to avoid numerical problems and slow inverse
    // FFT's in 3D or 2D
    for j in 0..nsize as usize {
        if ctf[j] < 1.0e-6 {
            ctf[j] = 0.;
        }
    }
    *nsize_out = nsize;
}
pub fn xcorr_set_ctf(
    sigma1: f32,
    sigma2: f32,
    radius1: f32,
    radius2: f32,
    ctf: &mut [f32],
    nx: i32,
    ny: i32,
    delta: &mut f32,
) {
    let mut nsize = 0;
    xcorr_set_ctf_no_scl(
        sigma1, sigma2, radius1, radius2, ctf, nx, ny, delta, &mut nsize,
    );
    if *delta == 0. {
        return;
    }
    let mut sum = 0.;
    for j in 1..nsize {
        sum += ctf[j as usize];
    }
    for j in 1..nsize {
        ctf[j as usize] *= (nsize - 1) as f32 / sum;
    }
}
pub fn setctfwsr(
    s1: &f32,
    s2: &f32,
    r1: &f32,
    r2: &f32,
    ctf: &mut [f32],
    nx: &i32,
    ny: &i32,
    delta: &mut f32,
) {
    xcorr_set_ctf(*s1, *s2, *r1, *r2, ctf, *nx, *ny, delta)
}
pub fn setctfnoscl(
    s1: &f32,
    s2: &f32,
    r1: &f32,
    r2: &f32,
    ctf: &mut [f32],
    nx: &i32,
    ny: &i32,
    delta: &mut f32,
    nsize: &mut i32,
) {
    xcorr_set_ctf_no_scl(*s1, *s2, *r1, *r2, ctf, *nx, *ny, delta, nsize)
}

pub fn dose_filter_value(
    _start: f32,
    end: f32,
    frequency: f32,
    mut afac: f32,
    mut bfac: f32,
    mut cfac: f32,
    scale: f32,
    atten: &mut f32,
) {
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
        /* 4/18/18: Removed complete attenuation above "optimal" dose per Ben Himes 2/24/18
        advice that it seems to hurt, and not doing it is current Grigorieff lab thinking */
        // `pow` is the double routine and `-0.5` is a double literal, so both
        // expressions are evaluated in double and rounded once on store.
        let critical = (scale as f64
            * (afac as f64 * (frequency as f64).powf(bfac as f64) + cfac as f64))
            as f32;
        *atten = (-0.5 * end as f64 / critical as f64).exp() as f32;
    }
}
pub fn dose_weight_filter(
    start: f32,
    end: f32,
    pixel: f32,
    a: f32,
    b: f32,
    c: f32,
    scale: f32,
    ctf: &mut [f32],
    n: i32,
    max: f32,
    delta: &mut f32,
) {
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
            &mut ctf[i as usize],
        );
    }
}
pub fn doseweightfilter(
    start: &f32,
    end: &f32,
    pixel: &f32,
    a: &f32,
    b: &f32,
    c: &f32,
    scale: &f32,
    ctf: &mut [f32],
    n: &i32,
    max: &f32,
    delta: &mut f32,
) {
    dose_weight_filter(
        *start, *end, *pixel, *a, *b, *c, *scale, ctf, *n, *max, delta,
    )
}

/// `XCorrMeanZero` (`filtxcorr.c:415`).
pub fn xcorr_mean_zero(array: &mut [f32], nxdim: i32, nx: i32, ny: i32) {
    // The source sums each row into `tsum` and adds the row totals into `sum`;
    // a single flat accumulation is a different order and a different mean.
    let mut sum = 0.0f32;
    for iy in 0..ny {
        let mut tsum = 0.0f32;
        let ixbase = iy * nxdim;
        for ix in 0..nx {
            tsum += array[(ix + ixbase) as usize];
        }
        sum += tsum;
    }
    let dmean = sum / (nx * ny) as f32;

    for iy in 0..ny {
        let ixbase = iy * nxdim;
        for ix in 0..nx {
            array[(ix + ixbase) as usize] -= dmean;
        }
    }
}
pub fn meanzero(a: &mut [f32], d: &i32, x: &i32, y: &i32) {
    xcorr_mean_zero(a, *d, *x, *y)
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
pub fn parabolicfitposition(a: &f32, b: &f32, c: &f32) -> f64 {
    parabolic_fit_position(*a, *b, *c)
}
pub fn conjugate_product(a: &mut [f32], b: &[f32], nx: i32, ny: i32) {
    for j in (0..ny * (nx + 2)).step_by(2) {
        let ar = a[j as usize];
        let ai = a[(j + 1) as usize];
        let br = b[j as usize];
        let bi = b[(j + 1) as usize];
        a[j as usize] = ar * br + ai * bi;
        a[(j + 1) as usize] = ai * br - ar * bi;
    }
}
pub fn conjugateproduct(a: &mut [f32], b: &[f32], nx: &i32, ny: &i32) {
    conjugate_product(a, b, *nx, *ny)
}

pub fn subarea_cc_coefficient(
    a: &[f32],
    b: &[f32],
    d: i32,
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    dx: i32,
    dy: i32,
) -> f64 {
    let n = ((x1 + 1 - x0) * (y1 + 1 - y0)) as f64;
    let (mut asum, mut bsum, mut csum, mut asq, mut bsq) = (0., 0., 0., 0., 0.);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let av = a[(x + y * d) as usize] as f64;
            let bv = b[(x - dx + (y - dy) * d) as usize] as f64;
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
pub fn cc_coefficient_two_pads(
    a: &[f32],
    b: &[f32],
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
    nsum: &mut i32,
) -> f64 {
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
pub fn xcorr_cc_coefficient(
    a: &[f32],
    b: &[f32],
    d: i32,
    nx: i32,
    ny: i32,
    x: f32,
    y: f32,
    px: i32,
    py: i32,
    n: &mut i32,
) -> f64 {
    cc_coefficient_two_pads(a, b, d, nx, ny, x, y, px, py, px, py, 25, n)
}
pub fn weighted_corr_from_sums(
    a: f64,
    asq: f64,
    b: f64,
    bsq: f64,
    ab: f64,
    w: f64,
    sums: Option<&mut [f64]>,
    desc: &str,
) -> f64 {
    // `if (descrip && strlen(descrip)) printf(...)` -- a NULL `descrip` and
    // an empty one behave the same, so `&str` covers both.
    if !desc.is_empty() {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "%s %g %g %g %g %g %g\n",
                &[
                    CArg::Str(desc),
                    CArg::Dbl(a),
                    CArg::Dbl(asq),
                    CArg::Dbl(bsq),
                    CArg::Dbl(b),
                    CArg::Dbl(ab),
                    CArg::Dbl(w),
                ],
            )
            .as_bytes(),
        );
    }
    if let Some(sums) = sums {
        for (i, v) in [a, asq, b, bsq, ab, w].into_iter().enumerate() {
            sums[i] = v;
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

pub fn slice_gaussian_kernel(mat: &mut [f32], dim: i32, sigma: f32) {
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
            let v = (-((x as f64 - mid) * (x as f64 - mid) + (y as f64 - mid) * (y as f64 - mid))
                / sigma_squared as f64)
                .exp() as f32;
            mat[(x + y * dim) as usize] = v;
            sum += v;
        }
    }
    for i in 0..dim * dim {
        mat[i as usize] /= sum;
    }
}
pub fn scaled_gaussian_kernel(mat: &mut [f32], dim: &mut i32, limit: i32, sigma: f32) {
    // `filtxcorr.c:392` casts ceil() to int first, then does the doubling
    // and the +1 in integer arithmetic.
    *dim = 2 * (sigma as f64).ceil() as i32 + 1;
    *dim = (*dim).min(limit);
    slice_gaussian_kernel(mat, *dim, sigma)
}
pub fn apply_kernel_filter(
    a: &[f32],
    b: &mut [f32],
    d: i32,
    nx: i32,
    ny: i32,
    mat: &[f32],
    k: i32,
) {
    let below = k / 2;

    // `filtxcorr.c:1782-1784`: the same size-dependent thread count as
    // `cubinterp`, then `numOMPthreads`, then `#pragma omp parallel for` over
    // `iyo`.  Iteration `iyo` reads `array` and `mat` and writes only
    // `brray[ixo + nxdim * iyo]`, so the output rows are disjoint, there is no
    // reduction, and each output pixel's sum is accumulated by the same
    // sequential `iy`/`ix` loops as before: the result does not depend on the
    // schedule or on the thread count.
    let mut num_threads = (0.04 * (nx as f64 * ny as f64).sqrt() + 0.5).floor() as i32;
    num_threads = num_omp_threads(num_threads);

    // One task per group of rows, so at most `numThreads` run at once and the
    // crate honours `OMP_NUM_THREADS` / `IMOD_FORCE_OMP_THREADS` the way
    // `numOMPthreads` does.
    let rows_per_group = if num_threads > 1 {
        (ny as usize).div_ceil(num_threads as usize).max(1)
    } else {
        (ny as usize).max(1)
    };
    let run_group = |(g, brows): (usize, &mut [f32])| {
        let oy0 = (g * rows_per_group) as i32;
        let oy1 = ((g + 1) * rows_per_group).min(ny as usize) as i32;
        for oy in oy0..oy1 {
            for ox in 0..nx {
                let mut sum = 0.;
                for iy in 0..k {
                    for ix in 0..k {
                        let x = (ox + ix - below).clamp(0, nx - 1);
                        let y = (oy + iy - below).clamp(0, ny - 1);
                        sum += mat[(ix + iy * k) as usize] * a[(x + y * d) as usize];
                    }
                }
                brows[(ox + (oy - oy0) * d) as usize] = sum;
            }
        }
    };
    if num_threads > 1 {
        // Native's OpenMP runtime creates at most `omp_get_num_procs()` workers
        // and `numOMPthreads` never asks for more than the physical-core count
        // (or whatever `OMP_NUM_THREADS` / `IMOD_FORCE_OMP_THREADS` allow).
        // rayon's default global pool is sized from `available_parallelism()`,
        // which counts *logical* processors, so it is bounded to the same count
        // here.  `build_global` succeeds for whichever translated unit reaches
        // it first and returns an error afterwards, which is the intended
        // no-op: every unit asks for the same size.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_omp_threads(i32::MAX) as usize)
            .build_global();
        b.par_chunks_mut(rows_per_group * d as usize)
            .enumerate()
            .for_each(run_group);
    } else {
        b.chunks_mut(rows_per_group * d as usize)
            .enumerate()
            .for_each(run_group);
    }
}

/// `indicesForFFTwrap` (`filtxcorr.c:1881`).
pub fn indices_for_fft_wrap(
    ny: i32,
    direction: i32,
    iy_out: &mut i32,
    iy_low: &mut i32,
    iy_high: &mut i32,
) -> i32 {
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
/// `wrapFFTslice` (`filtxcorr.c:1842`).
pub fn wrap_fft_slice(array: &mut [f32], tmp: &mut [f32], nx: i32, ny: i32, direction: i32) {
    let mut out = 0;
    let mut low = 0;
    let mut high = 0;
    let inc = indices_for_fft_wrap(ny, direction, &mut out, &mut low, &mut high);
    let width = (2 * nx) as usize;
    if ny % 2 != 0 {
        let o = (2 * nx * out) as usize;
        tmp[..width].copy_from_slice(&array[o..o + width]);
        for _ in 0..ny / 2 {
            array.copy_within(
                (2 * nx * high) as usize..(2 * nx * high) as usize + width,
                (2 * nx * out) as usize,
            );
            array.copy_within(
                (2 * nx * low) as usize..(2 * nx * low) as usize + width,
                (2 * nx * high) as usize,
            );
            out += inc;
            low += inc;
            high += inc;
        }
        let o = (2 * nx * out) as usize;
        array[o..o + width].copy_from_slice(&tmp[..width]);
    } else {
        for _ in 0..ny / 2 {
            let l = (2 * nx * low) as usize;
            tmp[..width].copy_from_slice(&array[l..l + width]);
            array.copy_within(
                (2 * nx * high) as usize..(2 * nx * high) as usize + width,
                l,
            );
            let h = (2 * nx * high) as usize;
            array[h..h + width].copy_from_slice(&tmp[..width]);
            low += 1;
            high += 1;
        }
    }
}
pub fn wrapfftslice(a: &mut [f32], t: &mut [f32], nx: &i32, ny: &i32, dir: &i32) {
    wrap_fft_slice(a, t, *nx, *ny, *dir)
}

/// `fourierShiftImage` (`filtxcorr.c:1906`).
pub fn fourier_shift_image(fft: &mut [f32], nx: i32, ny: i32, dx: f32, dy: f32, temp: &mut [f32]) {
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
        temp[(2 * x) as usize] = arg.cos() as f32;
        temp[(2 * x + 1) as usize] = arg.sin() as f32;
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
            let pr = temp[q as usize] * c - temp[(q + 1) as usize] * s;
            let pi = temp[(q + 1) as usize] * c + temp[q as usize] * s;
            let p = (q + y * ndim) as usize;
            let re = fft[p];
            let im = fft[p + 1];
            fft[p] = pr * re - pi * im;
            fft[p + 1] = pi * re + pr * im;
        }
    }
}
pub fn fourier_reduce_image(
    input: &[f32],
    nxi: i32,
    nyi: i32,
    out: &mut [f32],
    nxo: i32,
    nyo: i32,
    dx: f32,
    dy: f32,
    temp: Option<&mut [f32]>,
) {
    let fac = nxi as f32 / nxo as f32;
    // `float dxy = -(redFac - 1) / (2. * redFac);` -- the `2. *` makes the divide double.
    let d = ((-(fac - 1.)) as f64 / (2. * fac as f64)) as f32;
    let mut dst = 0usize;
    for loopi in 0..2 {
        let (start, end) = if loopi == 0 {
            (0, nyo - nyo / 2)
        } else {
            (nyi - nyo / 2, nyi)
        };
        for y in start..end {
            for x in 0..nxo + 2 {
                out[dst] = input[(y * (nxi + 2) + x) as usize] / fac;
                dst += 1;
            }
        }
    }
    if let Some(temp) = temp {
        fourier_shift_image(out, nxo, nyo, dx / fac + d, dy / fac + d, temp)
    }
}
pub fn fourier_expand_image(
    input: &mut [f32],
    nxi: i32,
    nyi: i32,
    out: &mut [f32],
    nxo: i32,
    nyo: i32,
    dx: f32,
    dy: f32,
    temp: Option<&mut [f32]>,
) {
    let fac = nxo as f32 / nxi as f32;
    // `float dxy = (expFac - 1) / (2. * expFac);` -- integer 1 here, but a double divide.
    let d = ((fac - 1.) as f64 / (2. * fac as f64)) as f32;
    out[..((nxo + 2) * nyo) as usize].fill(0.);
    if let Some(temp) = temp {
        fourier_shift_image(input, nxi, nyi, dx + d, dy + d, temp)
    }
    let mut src = 0usize;
    for loopi in 0..2 {
        let (start, end) = if loopi == 0 {
            (0, nyi - nyi / 2)
        } else {
            (nyo - nyi / 2, nyo)
        };
        for y in start..end {
            for x in 0..nxi + 2 {
                out[(y * (nxo + 2) + x) as usize] = input[src] * fac;
                src += 1;
            }
        }
    }
}
pub fn fourier_crop_sizes(
    size: i32,
    factor: f32,
    pad: f32,
    min_pad: i32,
    limit: i32,
    full: &mut i32,
    crop: &mut i32,
    actual: &mut f32,
) -> i32 {
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

/// The source's `XCorrFilterPart` input transform, whose documentation says
/// `array` "can be the same as [fft]" (`filtxcorr.c:265-267`).  Both of this
/// tree's callers do exactly that, and Rust cannot hold a `&` and a `&mut` to
/// one buffer, so `InPlace` names the aliased case; the filter is applied
/// element by element at matching indexes, which is why it works in place.
#[derive(Copy, Clone)]
pub enum FilterIn<'a> {
    Fft(&'a [f32]),
    InPlace,
}

/// `XCorrFilterPart` (`filtxcorr.c:271`).
pub fn xcorr_filter_part(
    fft: FilterIn,
    array: &mut [f32],
    nx: i32,
    ny: i32,
    ctf: &[f32],
    delta: f32,
) {
    let nx2 = nx / 2;
    let nx2p1 = nx2 + 1;
    let dx = (1.0 / nx as f64) as f32;
    let dy = (1.0 / ny as f64) as f32;

    /* Find last non-zero filter value in range that matters */
    // `for (ix = 0.707 / delta; ...)` -- `0.707` is a double literal, so
    // the quotient is a double truncated to `int`.
    let mut last = (0.707 / delta as f64) as i32;
    while last > 1 && ctf[last as usize] == 0. {
        last -= 1;
    }

    /* Get a frequency limit to apply in Y and a limit to X indexes */
    let maxf = (last + 1) as f32 * delta;
    // `B3DCLAMP(nxMax, 1, nxDiv2)` is `MAX(1, MIN(nxDiv2, nxMax))`, in
    // that nesting; `i32::clamp` panics when `nxDiv2 < 1`.
    let mut nxmax = (maxf / dx + 1.0) as i32;
    nxmax = if nx2 < nxmax { nx2 } else { nxmax };
    nxmax = if 1 > nxmax { 1 } else { nxmax };

    /*   apply filter function on fft, put result in array */
    for iy in 0..=ny - 1 {
        let mut fy = iy as f32 * dy;
        let index = iy * nx2p1;
        if fy > 0.5 {
            fy = 1.0f32 - fy;
        }
        if fy > maxf {
            for ix in 0..=nx2 {
                let ind = 2 * (index + ix);
                array[ind as usize] = 0.;
                array[(ind + 1) as usize] = 0.;
            }
        } else {
            // `double ysq = y * y;` -- the product is single precision and
            // then widens, and `x` is *accumulated* by `delx` rather than
            // recomputed as `ix * delx`.
            let ysq = (fy * fy) as f64;
            let mut x = 0.0f32;
            for ix in 0..=nxmax {
                let ind = 2 * (index + ix);
                let indp1 = ind + 1;
                let s = ((x * x) as f64 + ysq).sqrt() as f32;
                let indf = (s / delta + 0.5f32) as i32;
                let f = ctf[indf as usize];
                let (a, b) = match fft {
                    FilterIn::Fft(fft) => (fft[ind as usize], fft[indp1 as usize]),
                    FilterIn::InPlace => (array[ind as usize], array[indp1 as usize]),
                };
                array[ind as usize] = a * f;
                array[indp1 as usize] = b * f;
                x += dx;
            }
            // C deliberately stops at nxDiv2 (exclusive), retaining the
            // Nyquist pair when it is not visited by either loop.
            for ix in nxmax + 1..nx2 {
                let ind = 2 * (index + ix);
                array[ind as usize] = 0.;
                array[(ind + 1) as usize] = 0.;
            }
        }
    }
}

/// Original static `peakHalfWidth` (`filtxcorr.c:841`).
fn peak_half_width(
    array: &[f32],
    ix_peak: i32,
    iy_peak: i32,
    nx: i32,
    ny: i32,
    delx: i32,
    dely: i32,
) -> f32 {
    let nxdim = nx + 2;
    let peak = array[(ix_peak + iy_peak * nxdim) as usize];
    // `(float)sqrt((double)delx * delx + dely * dely)`.
    let scale = ((delx as f64 * delx as f64 + (dely * dely) as f64).sqrt()) as f32;
    let mut last_val = peak;
    let mut dist = 1;
    while dist < (if nx < ny { nx } else { ny }) / 4 {
        let ix = (ix_peak + dist * delx + nx) % nx;
        let iy = (iy_peak + dist * dely + ny) % ny;
        let val = array[(ix + iy * nxdim) as usize];
        // `peak / 2.` is a double, so the comparison and the interpolation
        // below are both done in double and rounded once by the `(float)`.
        if (val as f64) < peak as f64 / 2. {
            return scale
                * ((dist as f64 + (last_val as f64 - peak as f64 / 2.) / (last_val - val) as f64
                    - 1.) as f32);
        }
        last_val = val;
        dist += 1;
    }
    scale * dist as f32
}

/// Original static `accumRotatedCorner` (`filtxcorr.c:861`).
fn accum_rotated_corner(
    limit_x: i32,
    limit_y: i32,
    cosine: f32,
    sine: f32,
    lim_xlo: &mut i32,
    lim_xhi: &mut i32,
    lim_ylo: &mut i32,
    lim_yhi: &mut i32,
) {
    let idx = (cosine * limit_x as f32 - sine * limit_y as f32) as i32;
    let idy = (sine * limit_x as f32 + cosine * limit_y as f32) as i32;
    // `ACCUM_MIN(a, b)` is `a = a < b ? a : b`, which keeps the second operand
    // when the comparison is false.
    *lim_xlo = if *lim_xlo < idx { *lim_xlo } else { idx };
    *lim_xhi = if *lim_xhi > idx { *lim_xhi } else { idx };
    *lim_ylo = if *lim_ylo < idy { *lim_ylo } else { idy };
    *lim_yhi = if *lim_yhi > idy { *lim_yhi } else { idy };
}

/// Original static `computeTestLimits` (`filtxcorr.c:873`).
fn compute_test_limits(
    nx: i32,
    ny: i32,
    test_xlo: &mut i32,
    test_xhi: &mut i32,
    test_ylo: &mut i32,
    test_yhi: &mut i32,
    cos_lim: &mut f32,
    sin_lim: &mut f32,
) {
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
pub fn xcorr_peak_find_width(
    array: &[f32],
    nxdim: i32,
    ny: i32,
    xpeak: &mut [f32],
    ypeak: &mut [f32],
    peak: &mut [f32],
    mut width: Option<&mut [f32]>,
    mut width_min: Option<&mut [f32]>,
    max_peaks: i32,
    min_strength: f32,
) {
    let nx = nxdim - 2;
    let apply = S_APPLY_LIMITS.load(Ordering::SeqCst);
    let s_limit_xlo = S_LIMIT_XLO.load(Ordering::SeqCst);
    let s_limit_xhi = S_LIMIT_XHI.load(Ordering::SeqCst);
    let s_limit_ylo = S_LIMIT_YLO.load(Ordering::SeqCst);
    let s_limit_yhi = S_LIMIT_YHI.load(Ordering::SeqCst);
    let s_limit_angle = f32::from_bits(S_LIMIT_ANGLE.load(Ordering::SeqCst));
    let test_angle = apply != 0 && s_limit_angle != 0.;
    let mut cos_lim_ang = 1.0f32;
    let mut sin_lim_ang = 0.0f32;
    let mut threshold = 0.0f32;
    let mut test_lim_xlo = s_limit_xlo;
    let mut test_lim_xhi = s_limit_xhi;
    let mut test_lim_ylo = s_limit_ylo;
    let mut test_lim_yhi = s_limit_yhi;
    // `B3DNINT(a)` is `(int)floor(a + 0.5)`, not `round()`.
    let nint = |v: f32| (v as f64 + 0.5).floor() as i32;

    S_PEAK_FIND_ERROR.store(0, Ordering::SeqCst);
    let ixm_alloc = (ny * max_peaks) as usize;
    // The source carves `peakTemp`, `tempCopy`, `ixTemp`, `iyTemp` and
    // `numInLine` out of one `B3DMALLOC(float, 4 * ny * maxPeaks + ny)`, with
    // the last three reinterpreted as ints; the allocation cannot fail here,
    // so the `sWillCheckError` / exit path is unreachable.
    let mut peak_temp = vec![0.0f32; ixm_alloc.max(ny as usize)];
    let mut ix_temp = vec![0i32; ixm_alloc.max(ny as usize)];
    let mut iy_temp = vec![0i32; ixm_alloc.max(ny as usize)];
    let mut num_in_line = vec![0i32; ny as usize];

    /* If using elliptical limits, compute center and squares of radii */
    let mut x_lim_cen = 0.0f32;
    let mut y_lim_cen = 0.0f32;
    let mut x_lim_rad_sq = 0.0f32;
    let mut y_lim_rad_sq = 0.0f32;
    if apply < 0 {
        x_lim_cen = (0.5 * (s_limit_xlo + s_limit_xhi) as f64) as f32;
        let cx = {
            let v = (s_limit_xhi - s_limit_xlo) as f64 / 2.;
            (if 1.0f64 > v { 1.0f64 } else { v }) as f32
        };
        x_lim_rad_sq = cx * cx;
        y_lim_cen = (0.5 * (s_limit_ylo + s_limit_yhi) as f64) as f32;
        let cy = {
            let v = (s_limit_yhi - s_limit_ylo) as f64 / 2.;
            (if 1.0f64 > v { 1.0f64 } else { v }) as f32
        };
        y_lim_rad_sq = cy * cy;
    }
    if apply != 0 {
        compute_test_limits(
            nx,
            ny,
            &mut test_lim_xlo,
            &mut test_lim_xhi,
            &mut test_lim_ylo,
            &mut test_lim_yhi,
            &mut cos_lim_ang,
            &mut sin_lim_ang,
        );
    }
    // The `numOMPthreads` count only sizes the OpenMP team; the scans here are
    // serial and their comparison order is the same.

    /* find peaks */
    for i in 0..max_peaks as usize {
        peak[i] = -1.0e30;
        xpeak[i] = 0.;
        ypeak[i] = 0.;
    }

    /* Look for highest peak if looking for one peak or if there is a minimum strength */
    if max_peaks < 2 || min_strength > 0. {
        for i in 0..ny as usize {
            peak_temp[i] = -1.0e30;
            ix_temp[i] = 0;
            iy_temp[i] = 0;
            num_in_line[i] = 0;
        }

        /* Find one peak within the limits */
        if apply != 0 {
            for idy_line in test_lim_ylo..=test_lim_yhi {
                let iy = if idy_line < 0 {
                    idy_line + ny
                } else {
                    idy_line
                };
                // The source sets `idy` once per line and the angle branch below
                // *overwrites* it, so later columns on the line see the rotated
                // value.  That is reproduced here.
                let mut idy = idy_line;
                for ix in 0..nx {
                    let mut idx = if ix > nx / 2 { ix - nx } else { ix };
                    if idx < test_lim_xlo || idx > test_lim_xhi {
                        continue;
                    }
                    if test_angle {
                        if array[(ix + iy * nxdim) as usize] < peak_temp[iy as usize] {
                            continue;
                        }
                        let ixrot = nint(idx as f32 * cos_lim_ang + idy as f32 * sin_lim_ang);
                        idy = nint(-idx as f32 * sin_lim_ang + idy as f32 * cos_lim_ang);
                        idx = ixrot;
                        if idy < s_limit_ylo || idy > s_limit_yhi {
                            continue;
                        }
                    }
                    if idx >= s_limit_xlo
                        && idx <= s_limit_xhi
                        && array[(ix + iy * nxdim) as usize] > peak_temp[iy as usize]
                    {
                        if apply < 0 {
                            let cx = idx as f32 - x_lim_cen;
                            let cy = idy as f32 - y_lim_cen;
                            if cx * cx / x_lim_rad_sq + cy * cy / y_lim_rad_sq > 1. {
                                continue;
                            }
                        }
                        peak_temp[iy as usize] = array[(ix + iy * nxdim) as usize];
                        ix_temp[iy as usize] = ix;
                        iy_temp[iy as usize] = iy;
                    }
                }
            }
        } else {
            /* Or just find the one peak in the whole area */
            for iy in 0..ny {
                for ix in iy * nxdim..nx + iy * nxdim {
                    if array[ix as usize] > peak_temp[iy as usize] {
                        peak_temp[iy as usize] = array[ix as usize];
                        ix_temp[iy as usize] = ix - iy * nxdim;
                        iy_temp[iy as usize] = iy;
                    }
                }
            }
        }

        let mut ixpeak = 0;
        let mut iypeak = 0;
        for iy in 0..ny as usize {
            if peak_temp[iy] > peak[0] {
                peak[0] = peak_temp[iy];
                ixpeak = ix_temp[iy];
                iypeak = iy_temp[iy];
            }
        }

        if peak[0] > -0.9e30 {
            xpeak[0] = ixpeak as f32;
            ypeak[0] = iypeak as f32;
        }

        threshold = min_strength * peak[0];
    }

    /* Now find all requested peaks */
    if max_peaks > 1 {
        for i in 0..ny as usize {
            num_in_line[i] = 0;
        }

        // Look for local peaks and keep track of all on each line
        let ystart = if apply != 0 { test_lim_ylo } else { -ny / 2 };
        let yend = if apply != 0 { test_lim_yhi } else { ny / 2 - 1 };
        for idy_line in ystart..=yend {
            let iy = if idy_line < 0 {
                idy_line + ny
            } else {
                idy_line
            };
            for ix in 0..nx {
                if apply != 0 {
                    let mut idx = if ix > nx / 2 { ix - nx } else { ix };
                    let mut idy = idy_line;
                    if test_angle {
                        let ixrot = nint(idx as f32 * cos_lim_ang + idy as f32 * sin_lim_ang);
                        let iyrot = nint(-idx as f32 * sin_lim_ang + idy as f32 * cos_lim_ang);
                        idx = ixrot;
                        idy = iyrot;
                        if idy < s_limit_ylo || idy > s_limit_yhi {
                            continue;
                        }
                    }
                    if idx < s_limit_xlo || idx > s_limit_xhi {
                        continue;
                    }

                    // Apply elliptical test
                    if apply < 0 {
                        let cx = idx as f32 - x_lim_cen;
                        let cy = idy as f32 - y_lim_cen;
                        if cx * cx / x_lim_rad_sq + cy * cy / y_lim_rad_sq > 1. {
                            continue;
                        }
                    }
                }
                let local = array[(ix + iy * nxdim) as usize];
                if local > threshold {
                    // evaluate point for truly being local peak
                    // Allow equality on one side, otherwise identical adjacent values are lost
                    let ixm = (ix + nx - 1) % nx;
                    let ixp = (ix + 1) % nx;
                    let iyb = iy * nxdim;
                    let iybp = ((iy + 1) % ny) * nxdim;
                    let iybm = ((iy + ny - 1) % ny) * nxdim;

                    if local > array[(ix + iybm) as usize]
                        && local >= array[(ix + iybp) as usize]
                        && local > array[(ixm + iyb) as usize]
                        && local >= array[(ixp + iyb) as usize]
                        && local > array[(ixm + iybp) as usize]
                        && local >= array[(ixp + iybm) as usize]
                        && local > array[(ixp + iybp) as usize]
                        && local >= array[(ixm + iybm) as usize]
                    {
                        // Add peak to list for this line
                        if num_in_line[iy as usize] < max_peaks {
                            let idx2 = (iy * max_peaks + num_in_line[iy as usize]) as usize;
                            num_in_line[iy as usize] += 1;
                            peak_temp[idx2] = local;
                            ix_temp[idx2] = ix;
                        } else {
                            // Or find the lowest peak and replace - should be very rare
                            let mut lowest = local;
                            let mut idx2: i32 = -1;
                            for i in 0..num_in_line[iy as usize] {
                                if peak_temp[(iy * max_peaks + i) as usize] < lowest {
                                    idx2 = iy * max_peaks + i;
                                    lowest = peak_temp[idx2 as usize];
                                }
                            }
                            if idx2 >= 0 {
                                peak_temp[idx2 as usize] = local;
                                ix_temp[idx2 as usize] = ix;
                            }
                        }
                    }
                }
            }
        }

        // Consolidate the lists
        let mut num_peaks = 0usize;
        for iy in 0..ny {
            for i in 0..num_in_line[iy as usize] {
                let idx2 = (iy * max_peaks + i) as usize;
                peak_temp[num_peaks] = peak_temp[idx2];
                ix_temp[num_peaks] = ix_temp[idx2];
                iy_temp[num_peaks] = iy;
                num_peaks += 1;
            }
        }

        // Get the maxPeaks + 1 highest value fast
        let mut temp_copy = peak_temp[..num_peaks].to_vec();
        let sel = {
            let v = num_peaks as i32 - max_peaks;
            if 1 > v { 1 } else { v }
        };
        let val = crate::imod::libcfshr::percentile::percentile_float(
            sel,
            &mut temp_copy,
            num_peaks as i32,
        );

        // Loop on peaks and test against that after removing top peak from list
        peak[0] = -1.0e30;
        for ix in 0..num_peaks {
            if peak_temp[ix] >= val && peak_temp[ix] > peak[(max_peaks - 1) as usize] {
                // Insert peak into the list
                for i in 0..max_peaks as usize {
                    if peak[i] < peak_temp[ix] {
                        let mut j = max_peaks as usize - 1;
                        while j > i {
                            peak[j] = peak[j - 1];
                            xpeak[j] = xpeak[j - 1];
                            ypeak[j] = ypeak[j - 1];
                            j -= 1;
                        }
                        peak[i] = peak_temp[ix];
                        xpeak[i] = ix_temp[ix] as f32;
                        ypeak[i] = iy_temp[ix] as f32;
                        break;
                    }
                }
            }
        }
    }

    for i in 0..max_peaks {
        if peak[i as usize] < -0.9e30 {
            continue;
        }
        let ix = xpeak[i as usize] as i32;
        let iy = ypeak[i as usize] as i32;
        let cx = parabolic_fit_position(
            array[((ix + nx - 1) % nx + iy * nxdim) as usize],
            peak[i as usize],
            array[((ix + 1) % nx + iy * nxdim) as usize],
        ) as f32;
        let cy = parabolic_fit_position(
            array[(ix + ((iy + ny - 1) % ny) * nxdim) as usize],
            peak[i as usize],
            array[(ix + ((iy + 1) % ny) * nxdim) as usize],
        ) as f32;
        let mut px = ix as f32 + cx;
        let mut py = iy as f32 + cy;
        if px > nx as f32 / 2. {
            px -= nx as f32;
        }
        if py > ny as f32 / 2. {
            py -= ny as f32;
        }
        xpeak[i as usize] = px;
        ypeak[i as usize] = py;
        if let (Some(width), Some(width_min)) = (width.as_deref_mut(), width_min.as_deref_mut()) {
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
            // `avgSD(widthTemp, 4, &width[i], &cx, &cy)` refines the mean
            // with a second pass; a plain `sum / 4` is not the same value.
            let (mut cx2, mut cy2) = (0.0f32, 0.0f32);
            crate::imod::libcfshr::simplestat::avg_sd(
                &values,
                4,
                &mut width[i as usize],
                &mut cx2,
                &mut cy2,
            );
            // `B3DMIN` chained from the first pair, not from an infinity
            // seed: it keeps the second operand when the test is false.
            let mut wmin = if values[0] < values[1] {
                values[0]
            } else {
                values[1]
            };
            wmin = if wmin < values[2] { wmin } else { values[2] };
            wmin = if wmin < values[3] { wmin } else { values[3] };
            width_min[i as usize] = wmin;
        }
    }
    S_APPLY_LIMITS.store(0, Ordering::SeqCst);
    S_LIMIT_ANGLE.store(0, Ordering::SeqCst);
    S_WILL_CHECK_ERROR.store(0, Ordering::SeqCst);
}
pub fn get_peak_find_test_limits(
    nx: i32,
    ny: i32,
    lim_xlo: &mut i32,
    lim_xhi: &mut i32,
    lim_ylo: &mut i32,
    lim_yhi: &mut i32,
) {
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
pub fn xcorr_peak_find(
    array: &[f32],
    nxdim: i32,
    ny: i32,
    xpeak: &mut [f32],
    ypeak: &mut [f32],
    peak: &mut [f32],
    max_peaks: i32,
) {
    xcorr_peak_find_width(
        array, nxdim, ny, xpeak, ypeak, peak, None, None, max_peaks, 0.,
    )
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
pub fn find_many_xcorr_peaks(
    array: &[f32],
    nxdim: i32,
    ny: i32,
    ix_offset: i32,
    iy_offset: i32,
    xpeak: &mut [f32],
    ypeak: &mut [f32],
    peak: &mut [f32],
    max_peaks: i32,
    max_grow: i32,
    num_found: &mut i32,
) -> i32 {
    let mut threshold = -1.0e30_f32;
    let mut min_found = 1.0e30_f32;
    let mut num_peaks = 0_i32;
    let mut use_temp2 = 1_i32;
    let mut if_first = 1_i32;

    if max_grow as f32 <= 1.05 * max_peaks as f32 {
        return 1;
    }

    /* Allocate arrays and test them */
    // The source's seven `B3DMALLOC` failure returns (-1) are unreachable
    // here.  `ixTemp1`/`ixTemp2` and their partners are the two halves of
    // a ping-pong pair, indexed by `cur` instead of by a swapped pointer.
    let mut indexes = vec![0i32; max_grow as usize];
    let mut ix_temp: [Vec<i16>; 2] = [vec![0i16; max_grow as usize], vec![0i16; max_grow as usize]];
    let mut iy_temp: [Vec<i16>; 2] = [vec![0i16; max_grow as usize], vec![0i16; max_grow as usize]];
    let mut peak_temp: [Vec<f32>; 2] = [
        vec![0.0f32; max_grow as usize],
        vec![0.0f32; max_grow as usize],
    ];

    /* Set initial pointers */
    let mut cur = 0usize;
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
            let val = array[xbase as usize];

            // Ignore anything below the lowest peak currently retained.
            if val < threshold {
                continue;
            }

            let mut is_peak = 0;
            if iy_offset < 0 && (iy == 0 || iy == ny - 1 || ix == 0 || ix == ix_end - 1) {
                let xprev = (ix + ix_end - 1) % ix_end;
                let xnext = (ix + 1) % ix_end;
                if val < array[(xprev + iy * nxdim) as usize]
                    || val < array[(xnext + iy * nxdim) as usize]
                    || val < array[(xprev + yprev * nxdim) as usize]
                    || val < array[(ix + yprev * nxdim) as usize]
                    || val < array[(xnext + yprev * nxdim) as usize]
                    || val < array[(xprev + ynext * nxdim) as usize]
                    || val < array[(ix + ynext * nxdim) as usize]
                    || val < array[(xnext + ynext * nxdim) as usize]
                {
                    continue;
                }
                is_peak = 1;
            }

            if is_peak != 0
                || (val > array[(xbase - nxdim) as usize]
                    && val >= array[(xbase + nxdim) as usize]
                    && val > array[(xbase - 1) as usize]
                    && val >= array[(xbase + 1) as usize]
                    && val > array[(xbase + nxdim - 1) as usize]
                    && val >= array[(xbase + 1 - nxdim) as usize]
                    && val > array[(xbase + nxdim + 1) as usize]
                    && val >= array[(xbase - 1 - nxdim) as usize])
            {
                ix_temp[cur][num_peaks as usize] = ix as i16;
                iy_temp[cur][num_peaks as usize] = iy as i16;
                peak_temp[cur][num_peaks as usize] = -val;
                indexes[num_peaks as usize] = num_peaks;
                num_peaks += 1;

                if num_peaks <= max_peaks && val < min_found {
                    min_found = val;
                }
                if num_peaks == max_peaks {
                    threshold = min_found;
                }

                if num_peaks == max_grow {
                    let dst = if use_temp2 != 0 { 1usize } else { 0usize };
                    {
                        let (ixa, ixb) = ix_temp.split_at_mut(1);
                        let (iya, iyb) = iy_temp.split_at_mut(1);
                        let (pka, pkb) = peak_temp.split_at_mut(1);
                        let (ix_f, ix_t) = if cur == 0 {
                            (&ixa[0][..], &mut ixb[0])
                        } else {
                            (&ixb[0][..], &mut ixa[0])
                        };
                        let (iy_f, iy_t) = if cur == 0 {
                            (&iya[0][..], &mut iyb[0])
                        } else {
                            (&iyb[0][..], &mut iya[0])
                        };
                        let (pk_f, pk_t) = if cur == 0 {
                            (&pka[0][..], &mut pkb[0])
                        } else {
                            (&pkb[0][..], &mut pka[0])
                        };
                        debug_assert_ne!(cur, dst);
                        sort_and_repack(
                            &mut indexes,
                            ix_f,
                            iy_f,
                            pk_f,
                            max_peaks,
                            num_peaks,
                            ix_t,
                            iy_t,
                            pk_t,
                            if_first,
                        );
                    }
                    if_first = 0;
                    num_peaks = max_peaks;
                    cur = dst;
                    use_temp2 = 1 - use_temp2;
                    threshold = -peak_temp[cur][(max_peaks - 1) as usize];
                }
            }
        }
    }

    let dst = if use_temp2 != 0 { 1usize } else { 0usize };
    {
        let (ixa, ixb) = ix_temp.split_at_mut(1);
        let (iya, iyb) = iy_temp.split_at_mut(1);
        let (pka, pkb) = peak_temp.split_at_mut(1);
        let (ix_f, ix_t) = if cur == 0 {
            (&ixa[0][..], &mut ixb[0])
        } else {
            (&ixb[0][..], &mut ixa[0])
        };
        let (iy_f, iy_t) = if cur == 0 {
            (&iya[0][..], &mut iyb[0])
        } else {
            (&iyb[0][..], &mut iya[0])
        };
        let (pk_f, pk_t) = if cur == 0 {
            (&pka[0][..], &mut pkb[0])
        } else {
            (&pkb[0][..], &mut pka[0])
        };
        sort_and_repack(
            &mut indexes,
            ix_f,
            iy_f,
            pk_f,
            max_peaks,
            num_peaks,
            ix_t,
            iy_t,
            pk_t,
            if_first,
        );
    }
    cur = dst;
    num_peaks = num_peaks.min(max_peaks);

    for ind in 0..num_peaks {
        peak[ind as usize] = -peak_temp[cur][ind as usize];
        let ix = ix_temp[cur][ind as usize] as i32;
        let iy = iy_temp[cur][ind as usize] as i32;
        let xbase = ix + iy * nxdim;
        let (cx, cy);
        if iy_offset < 0 && (ix == 0 || iy == 0 || ix == ix_end - 1 || iy == ny - 1) {
            let xprev = (ix + ix_end - 1) % ix_end;
            let xnext = (ix + 1) % ix_end;
            let yprev = (iy + ny - 1) % ny;
            let ynext = (iy + 1) % ny;
            cx = parabolic_fit_position(
                array[(xprev + iy * nxdim) as usize],
                array[xbase as usize],
                array[(xnext + iy * nxdim) as usize],
            ) as f32;
            cy = parabolic_fit_position(
                array[(ix + yprev * nxdim) as usize],
                array[xbase as usize],
                array[(ix + ynext * nxdim) as usize],
            ) as f32;
        } else {
            cx = parabolic_fit_position(
                array[(xbase - 1) as usize],
                array[xbase as usize],
                array[(xbase + 1) as usize],
            ) as f32;
            cy = parabolic_fit_position(
                array[(xbase - nxdim) as usize],
                array[xbase as usize],
                array[(xbase + nxdim) as usize],
            ) as f32;
        }
        xpeak[ind as usize] = ix as f32 + cx;
        ypeak[ind as usize] = iy as f32 + cy;
    }
    *num_found = num_peaks;

    if iy_offset < 0 {
        for ind in 0..num_peaks {
            if xpeak[ind as usize] > (ix_end / 2) as f32 {
                xpeak[ind as usize] -= ix_end as f32;
            }
            if ypeak[ind as usize] > (ny / 2) as f32 {
                ypeak[ind as usize] -= ny as f32;
            }
        }
    }

    0
}

/// `sortAndRepack` (`filtxcorr.c:1204`), the file-static helper that sorts the
/// new part of the peak arrays and repacks the ones being kept into a different
/// set of arrays.  `COPY_FROM_TO` (`filtxcorr.c:1198`) is expanded at each of
/// its three use sites.
fn sort_and_repack(
    indexes: &mut [i32],
    ix_from: &[i16],
    iy_from: &[i16],
    peak_from: &[f32],
    keep_peaks: i32,
    num_peaks: i32,
    ix_to: &mut [i16],
    iy_to: &mut [i16],
    peak_to: &mut [f32],
    if_first: i32,
) {
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
            from = indexes[ind as usize];
            ix_to[ind as usize] = ix_from[from as usize];
            iy_to[ind as usize] = iy_from[from as usize];
            peak_to[ind as usize] = peak_from[from as usize];
            indexes[ind as usize] = ind;
            ind += 1;
        }
    } else {
        /* Otherwise sort the upper part of the array, setting index offset
        appropriately */
        rs_set_sort_index_offset(keep_peaks);
        rs_sort_indexed_floats(
            &peak_from[keep_peaks as usize..],
            &mut indexes[keep_peaks as usize..],
            num_peaks - keep_peaks,
        );

        /* Merge the two sections by taking the lowest value from each eat each step */
        low = 0;
        high = keep_peaks;
        ind = 0;
        while ind < keep_peaks && low < keep_peaks && high < num_peaks {
            if peak_from[indexes[low as usize] as usize]
                < peak_from[indexes[high as usize] as usize]
            {
                from = indexes[low as usize];
                low += 1;
            } else {
                from = indexes[high as usize];
                high += 1;
            }
            ix_to[ind as usize] = ix_from[from as usize];
            iy_to[ind as usize] = iy_from[from as usize];
            peak_to[ind as usize] = peak_from[from as usize];
            indexes[ind as usize] = ind;
            ind += 1;
        }

        /* Finish up with one or the other if deficient */
        low = if low < keep_peaks { low } else { high };
        while ind < keep_peaks {
            from = indexes[low as usize];
            low += 1;
            ix_to[ind as usize] = ix_from[from as usize];
            iy_to[ind as usize] = iy_from[from as usize];
            peak_to[ind as usize] = peak_from[from as usize];
            indexes[ind as usize] = ind;
            ind += 1;
        }
    }
}

pub fn find_spaced_xcorr_peaks(
    array: &[f32],
    nxdim: i32,
    ix_min: i32,
    ix_max: i32,
    iy_min: i32,
    iy_max: i32,
    xpeak: &mut [f32],
    ypeak: &mut [f32],
    peak: &mut [f32],
    max_peaks: i32,
    min_spacing: f32,
    num_peaks: &mut i32,
    min_strength: f32,
) -> i32 {
    let block_size = (min_spacing as f64 / 2.0f64.sqrt()) as i32;
    let nblocks_x = ((ix_max - (ix_min + 1)) + block_size - 1) / block_size;
    let nblocks_y = ((iy_max - (iy_min + 1)) + block_size - 1) / block_size;
    let tot_blocks = nblocks_x * nblocks_y;

    // Tested up to 12 threads, ~80% efficient at 8 on somewhat less than 2K images
    let mut num_threads = {
        let v = (0.005 * ((ix_max - ix_min) as f64 * (iy_max - iy_min) as f64).sqrt()) as i32;
        if MAX_SPCP_THREADS < v {
            MAX_SPCP_THREADS
        } else {
            v
        }
    };
    num_threads = crate::imod::libcfshr::b3dutil::num_omp_threads(num_threads);
    // The `#pragma omp parallel for` over `iyBlock` is run serially here, so
    // every block lands in thread 0's slice; the repack below then walks the
    // blocks in `iyBlock`, `ixBlock` order exactly as one thread would.
    let lanes = if num_threads > 1 { num_threads } else { 1 };
    let ind_alloc = (tot_blocks * lanes) as usize;
    // The five `B3DMALLOC`s cannot fail here, so the source's -1 return is
    // unreachable.
    let mut ix_all = vec![0i16; ind_alloc];
    let mut iy_all = vec![0i16; ind_alloc];
    let mut peak_all = vec![0.0f32; ind_alloc];
    let mut indexes = vec![0i32; ind_alloc];
    let mut box_ind_all = vec![-1i32; ind_alloc];

    let mut num_pk_thread = vec![0i32; lanes as usize];
    let mut ix_thread = vec![0i16; ind_alloc];
    let mut iy_thread = vec![0i16; ind_alloc];
    let mut peak_thread = vec![0.0f32; ind_alloc];
    let mut box_ind_thread = vec![-1i32; ind_alloc];

    for iy_block in 0..nblocks_y {
        let thrd = crate::imod::libcfshr::b3dutil::b3d_omp_thread_num();

        // Block start and end in Y
        let iy_start = iy_min + 1 + iy_block * block_size;
        let iy_end = if iy_start + block_size < iy_max {
            iy_start + block_size
        } else {
            iy_max
        };

        for ix_block in 0..nblocks_x {
            // Block start and end in X
            let ix_start = ix_min + 1 + ix_block * block_size;
            let ix_end = if ix_start + block_size < ix_max {
                ix_start + block_size
            } else {
                ix_max
            };

            let mut max_val = -1.0e30f32;
            let mut ix_peak = -1;
            let mut iy_peak = 0;
            for iy in iy_start..iy_end {
                let ybase = nxdim * iy;
                for ix in ix_start..ix_end {
                    let xbase = ybase + ix;
                    let val = array[xbase as usize];
                    if val < max_val {
                        continue;
                    }

                    /* Test for actual peak value */
                    if val > array[(xbase - nxdim) as usize]
                        && val >= array[(xbase + nxdim) as usize]
                        && val > array[(xbase - 1) as usize]
                        && val >= array[(xbase + 1) as usize]
                        && val > array[(xbase + nxdim - 1) as usize]
                        && val >= array[(xbase + 1 - nxdim) as usize]
                        && val > array[(xbase + nxdim + 1) as usize]
                        && val >= array[(xbase - 1 - nxdim) as usize]
                    {
                        /* Found a peak above the current max: save it */
                        ix_peak = ix;
                        iy_peak = iy;
                        max_val = val;
                    }
                }
            }

            // Found a peak in the block, add it to arrays
            if ix_peak > 0 {
                let base = (thrd * tot_blocks) as usize;
                let ind = num_pk_thread[thrd as usize];
                num_pk_thread[thrd as usize] += 1;
                ix_thread[base + ind as usize] = ix_peak as i16;
                iy_thread[base + ind as usize] = iy_peak as i16;
                peak_thread[base + ind as usize] = max_val;
                box_ind_thread[base + (ix_block + iy_block * nblocks_x) as usize] = ind;
            }
        }
    }

    // Repack to be contiguous for sorting and invert peak for sorting too
    let mut num_found = 0usize;
    for thrd in 0..lanes {
        let base = (thrd * tot_blocks) as usize;
        let xbase = num_found as i32;
        for ind in 0..num_pk_thread[thrd as usize] as usize {
            indexes[num_found] = num_found as i32;
            ix_all[num_found] = ix_thread[base + ind];
            iy_all[num_found] = iy_thread[base + ind];
            peak_all[num_found] = -peak_thread[base + ind];
            num_found += 1;
        }
        for ind in 0..tot_blocks as usize {
            if box_ind_thread[base + ind] >= 0 {
                box_ind_all[ind] = box_ind_thread[base + ind] + xbase;
            }
        }
    }
    rs_sort_indexed_floats(&peak_all, &mut indexes, num_found as i32);

    /* Eliminate any lesser peaks too close to stronger one */
    let space_crit = min_spacing * min_spacing;
    for sind in 0..num_found.saturating_sub(1) {
        let ind = indexes[sind] as usize;
        if peak_all[ind] > 1.0e29 {
            continue;
        }

        /* Need to loop up to 2 blocks away except for corner blocks but it costs more to
        test if in corner */
        let ix_block = (ix_all[ind] as i32 - (ix_min + 1)) / block_size;
        let iy_block = (iy_all[ind] as i32 - (iy_min + 1)) / block_size;
        let ix_start = if 0 > ix_block - 2 { 0 } else { ix_block - 2 };
        let ix_end = if nblocks_x < ix_block + 3 {
            nblocks_x
        } else {
            ix_block + 3
        };
        let iy_start = if 0 > iy_block - 2 { 0 } else { iy_block - 2 };
        let iy_end = if nblocks_y < iy_block + 3 {
            nblocks_y
        } else {
            iy_block + 3
        };
        for iy in iy_start..iy_end {
            for ix in ix_start..ix_end {
                if ix == ix_block && iy == iy_block {
                    continue;
                }
                let jnd = box_ind_all[(ix + iy * nblocks_x) as usize];
                if jnd >= 0
                    && peak_all[jnd as usize] < 1.0e29
                    && peak_all[jnd as usize] >= peak_all[ind]
                {
                    let dx = (ix_all[ind] as i32 - ix_all[jnd as usize] as i32) as f32;
                    let dy = (iy_all[ind] as i32 - iy_all[jnd as usize] as i32) as f32;
                    if dx * dx + dy * dy < space_crit {
                        peak_all[jnd as usize] = 1.0e30;
                    }
                }
            }
        }
    }

    /* Get interpolated position for each and return into arrays */
    let mut ind_out = 0usize;
    for sind in 0..num_found {
        if ind_out >= max_peaks as usize {
            break;
        }
        let ind = indexes[sind] as usize;
        if peak_all[ind] > 1.0e29 {
            continue;
        }
        if ind_out != 0 && min_strength > 0. && -peak_all[ind] < min_strength * peak[0] {
            break;
        }
        peak[ind_out] = -peak_all[ind];
        let ix = ix_all[ind] as i32;
        let iy = iy_all[ind] as i32;
        let xbase = ix + iy * nxdim;
        let cx = parabolic_fit_position(
            array[(xbase - 1) as usize],
            array[xbase as usize],
            array[(xbase + 1) as usize],
        ) as f32;
        let cy = parabolic_fit_position(
            array[(xbase - nxdim) as usize],
            array[xbase as usize],
            array[(xbase + nxdim) as usize],
        ) as f32;
        xpeak[ind_out] = ix as f32 + cx;
        ypeak[ind_out] = iy as f32 + cy;
        ind_out += 1;
    }
    *num_peaks = ind_out as i32;
    0
}
pub fn weighted_cc_coefficient(
    a: &[f32],
    b: &[f32],
    nx_dim: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    dx: i32,
    dy: i32,
    aw: &[f32],
    bw: &[f32],
    nx_weight: i32,
    bin: i32,
    xoffset: i32,
    yoffset: i32,
) -> f64 {
    let (mut wsum, mut asum, mut bsum, mut asq, mut bsq, mut ab) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for iy in iy0..=iy1 {
        let abase = iy * nx_dim;
        let bbase = (iy - dy) * nx_dim - dx;
        let awbase = (iy / bin + yoffset) * nx_weight + xoffset;
        let bwbase = ((iy - dy) / bin + yoffset) * nx_weight + xoffset;
        // The source accumulates each row into its own double and adds the row
        // totals into the grand sums; one flat accumulation is a different
        // order.  Every product inside the row is single precision -- `aval`,
        // `bval` and `wgt` are all `float` -- and widens only for the `+=`.
        let (mut wgt_tmp, mut aw_tmp, mut bw_tmp) = (0f64, 0f64, 0f64);
        let (mut aw_tmp_sq, mut bw_tmp_sq, mut abw_tmp) = (0f64, 0f64, 0f64);
        for ix in ix0..=ix1 {
            let aval = a[(ix + abase) as usize];
            let bval = b[(ix + bbase) as usize];
            let awgt = aw[(ix / bin + awbase) as usize];
            let bwgt = bw[((ix - dx) / bin + bwbase) as usize];
            let wgt = awgt * bwgt;
            wgt_tmp += wgt as f64;
            aw_tmp += (aval * wgt) as f64;
            bw_tmp += (bval * wgt) as f64;
            aw_tmp_sq += (aval * aval * wgt) as f64;
            bw_tmp_sq += (bval * bval * wgt) as f64;
            abw_tmp += (aval * bval * wgt) as f64;
        }
        wsum += wgt_tmp;
        asum += aw_tmp;
        bsum += bw_tmp;
        asq += aw_tmp_sq;
        bsq += bw_tmp_sq;
        ab += abw_tmp;
    }
    weighted_corr_from_sums(asum, asq, bsum, bsq, ab, wsum, None, "")
}
/// `fourierShiftVolume` (`filtxcorr.c:2070`).
pub fn fourier_shift_volume(
    fft: &mut [f32],
    nx_pad: i32,
    ny_pad: i32,
    nz_pad: i32,
    dx: f32,
    dy: f32,
    dz: f32,
    temp: &mut [f32],
) {
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
        temp[(2 * ix) as usize] = arg.cos() as f32;
        temp[(2 * ix + 1) as usize] = arg.sin() as f32;
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
                let pre = temp[xind as usize] * yzre - temp[(xind + 1) as usize] * yzim;
                let pim = temp[(xind + 1) as usize] * yzre + temp[xind as usize] * yzim;
                let ind = base + xind as usize;
                let real = fft[ind];
                let imag = fft[ind + 1];
                fft[ind] = pre * real - pim * imag;
                fft[ind + 1] = pim * real + pre * imag;
            }
        }
    }
}
/// `fourierReduceVolume` (`filtxcorr.c:2158`).
pub fn fourier_reduce_volume(
    input: &[f32],
    nxi: i32,
    nyi: i32,
    nzi: i32,
    out: &mut [f32],
    nxo: i32,
    nyo: i32,
    nzo: i32,
    dx: f32,
    dy: f32,
    dz: f32,
    temp: Option<&mut [f32]>,
) {
    let xfac = nxi as f32 / nxo as f32;
    // `float dxy = -(redFac - 1) / (2. * redFac);` -- the `2. *` makes the divide double.
    let dxy = ((-(xfac - 1.0)) as f64 / (2.0 * xfac as f64)) as f32;
    let zfac = nzi as f32 / nzo as f32;
    let zd = ((-(zfac - 1.0)) as f64 / (2.0 * zfac as f64)) as f32;
    // `1. / pow(redFac * redFac * zRedFac, 1./3.)` -- the double `pow`, rounded once.
    let scale = (1.0 / ((xfac * xfac * zfac) as f64).powf(1.0 / 3.0)) as f32;
    let mut dst = 0usize;
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
                        out[dst] = input[(base + ix) as usize] * scale;
                        dst += 1;
                    }
                }
            }
        }
    }
    if let Some(temp) = temp {
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
/// `fourierExpandVolume` (`filtxcorr.c:2226`).
pub fn fourier_expand_volume(
    input: &mut [f32],
    nxi: i32,
    nyi: i32,
    nzi: i32,
    out: &mut [f32],
    nxo: i32,
    nyo: i32,
    nzo: i32,
    dx: f32,
    dy: f32,
    dz: f32,
    temp: Option<&mut [f32]>,
) {
    let xfac = nxo as f32 / nxi as f32;
    let zfac = nzo as f32 / nzi as f32;
    // `float dxy = (expFac - 1.) / (2. * expFac);` -- a wholly double expression.
    let dxy = ((xfac as f64 - 1.0) / (2.0 * xfac as f64)) as f32;
    let zd = ((zfac as f64 - 1.0) / (2.0 * zfac as f64)) as f32;
    out[..((nxo + 2) * nyo * nzo) as usize].fill(0.);
    if let Some(temp) = temp {
        fourier_shift_volume(input, nxi, nyi, nzi, dx + dxy, dy + dxy, dz + zd, temp);
    }
    let mut src = 0usize;
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
                        out[(base + ix) as usize] = input[src] * xfac;
                        src += 1;
                    }
                }
            }
        }
    }
}
pub fn fourier_ring_corr(
    a: &[f32],
    b: &[f32],
    nx: i32,
    ny: i32,
    corr: &mut [f32],
    max: i32,
    delta: f32,
    temp: &mut [f32],
) {
    // `float *prodReal = temp; asum = prodReal + maxRings;
    //  bsum = asum + maxRings; int *nsum = (int *)bsum + maxRings`
    // (`filtxcorr.c:2296-2299`).  The last window is the same `temp`
    // storage reinterpreted as ints, which Rust cannot do without
    // `unsafe`, and the documentation allows `temp` to be the same array
    // as `ringCorrs`, which Rust cannot express as two `&mut` either --
    // so the four windows are local here and `temp` is left alone.
    let _ = temp;
    // `float deltaX = 1. / nxReal;` -- a double quotient stored into a float,
    // and `xx = ix * deltaX / 2.` multiplies by it rather than dividing again.
    let delta_x = (1.0 / nx as f64) as f32;
    let delta_y = (1.0 / ny as f64) as f32;
    let mut prod = vec![0.0f32; max as usize];
    let mut asum = vec![0.0f32; max as usize];
    let mut bsum = vec![0.0f32; max as usize];
    let mut nsum = vec![0i32; max as usize];
    for ring in 0..max as usize {
        asum[ring] = 0.;
        bsum[ring] = 0.;
        prod[ring] = 0.;
        nsum[ring] = 0;
    }

    /* We are summing only over a half-plane of the full FFT, ignoring the symmetric part
    Summing over the whole plane would double the real component and the magnitude sums
    and cancel out the sum of imaginary components, so we don't bother with the latter */
    for iy in 0..ny {
        let mut yy = iy as f32 * delta_y;
        if yy > 0.5 {
            yy = (1. - yy as f64) as f32;
        }
        let yysqr = yy * yy;
        let base = iy * (nx + 2);
        let mut ix = 0;
        while ix < nx + 2 {
            // `float xx = ix * deltaX / 2.;` -- the `/ 2.` is done in double
            // but the result is stored back into a float before it is squared.
            let xx = ((ix as f32 * delta_x) as f64 / 2.) as f32;
            let freq = ((xx * xx + yysqr) as f64).sqrt() as f32;
            let ring = (freq / delta) as i32;
            if ring < max && (ix > 0 || iy >= ny / 2) {
                let ind = (base + ix) as usize;
                let a_real = a[ind];
                let a_imag = a[ind + 1];
                let b_real = b[ind];
                let b_imag = b[ind + 1];
                nsum[ring as usize] += 1;
                asum[ring as usize] += a_real * a_real + a_imag * a_imag;
                bsum[ring as usize] += b_real * b_real + b_imag * b_imag;
                prod[ring as usize] += a_real * b_real + a_imag * b_imag;
            }
            ix += 2;
        }
    }

    for ring in 0..max as usize {
        corr[ring] = if nsum[ring] != 0 && (asum[ring] * bsum[ring]) as f64 > 0. {
            // `prodReal[ring] / sqrt(asum * bsum)` -- `sqrt` is the double
            // routine, so the division is done in double.
            (prod[ring] as f64 / ((asum[ring] * bsum[ring]) as f64).sqrt()) as f32
        } else {
            0.
        };
    }
}

// Fortran entry points in the same source unit.  The build system historically
// selects their trailing underscore spelling; Rust keeps the source function names
// snake-cased while preserving pointer/value calling conventions.
pub fn filterpart(a: FilterIn, b: &mut [f32], nx: &i32, ny: &i32, ctf: &[f32], delta: &f32) {
    xcorr_filter_part(a, b, *nx, *ny, ctf, *delta)
}
pub fn xcorrpeakfindwidth(
    array: &[f32],
    nxdim: &i32,
    ny: &i32,
    xpeak: &mut [f32],
    ypeak: &mut [f32],
    peak: &mut [f32],
    width: Option<&mut [f32]>,
    width_min: Option<&mut [f32]>,
    max_peaks: &i32,
    min_strength: &f32,
) {
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
pub fn getpeakfindtestlimits(
    nx: &i32,
    ny: &i32,
    xlo: &mut i32,
    xhi: &mut i32,
    ylo: &mut i32,
    yhi: &mut i32,
) {
    get_peak_find_test_limits(*nx, *ny, xlo, xhi, ylo, yhi)
}
pub fn xcorrpeakfind(
    array: &[f32],
    nxdim: &i32,
    ny: &i32,
    xpeak: &mut [f32],
    ypeak: &mut [f32],
    peak: &mut [f32],
    max_peaks: &i32,
) {
    xcorr_peak_find(array, *nxdim, *ny, xpeak, ypeak, peak, *max_peaks)
}
pub fn storepeakfinderror(value: &i32) {
    store_peak_find_error(*value)
}
pub fn getpeakfinderror() -> i32 {
    get_peak_find_error()
}
pub fn setpeakfindlimits(xlo: &i32, xhi: &i32, ylo: &i32, yhi: &i32, ellipse: &i32) {
    set_peak_find_limits(*xlo, *xhi, *ylo, *yhi, *ellipse)
}
pub fn setpeakfindangle(angle: &f32) {
    set_peak_find_angle(*angle)
}
pub fn cccoefficienttwopads(
    a: &[f32],
    b: &[f32],
    nxdim: &i32,
    nx: &i32,
    ny: &i32,
    xpeak: &f32,
    ypeak: &f32,
    nx_pad_a: &i32,
    ny_pad_a: &i32,
    nx_pad_b: &i32,
    ny_pad_b: &i32,
    min_pixels: &i32,
    nsum: &mut i32,
) -> f64 {
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
pub fn cccoefficient(
    a: &[f32],
    b: &[f32],
    nxdim: &i32,
    nx: &i32,
    ny: &i32,
    xpeak: &f32,
    ypeak: &f32,
    nx_pad: &i32,
    ny_pad: &i32,
    nsum: &mut i32,
) -> f64 {
    xcorr_cc_coefficient(
        a, b, *nxdim, *nx, *ny, *xpeak, *ypeak, *nx_pad, *ny_pad, nsum,
    )
}
pub fn slicegaussiankernel(mat: &mut [f32], dim: &i32, sigma: &f32) {
    slice_gaussian_kernel(mat, *dim, *sigma)
}
pub fn scaledgaussiankernel(mat: &mut [f32], dim: &mut i32, limit: &i32, sigma: &f32) {
    scaled_gaussian_kernel(mat, dim, *limit, *sigma)
}
pub fn applykernelfilter(
    a: &[f32],
    b: &mut [f32],
    d: &i32,
    x: &i32,
    y: &i32,
    mat: &[f32],
    k: &i32,
) {
    apply_kernel_filter(a, b, *d, *x, *y, mat, *k)
}
pub fn fouriershiftimage(a: &mut [f32], x: &i32, y: &i32, dx: &f32, dy: &f32, t: &mut [f32]) {
    fourier_shift_image(a, *x, *y, *dx, *dy, t)
}
pub fn fourierreduceimage(
    a: &[f32],
    xi: &i32,
    yi: &i32,
    b: &mut [f32],
    xo: &i32,
    yo: &i32,
    dx: &f32,
    dy: &f32,
    t: Option<&mut [f32]>,
) {
    fourier_reduce_image(a, *xi, *yi, b, *xo, *yo, *dx, *dy, t)
}
pub fn fourierexpandimage(
    a: &mut [f32],
    xi: &i32,
    yi: &i32,
    b: &mut [f32],
    xo: &i32,
    yo: &i32,
    dx: &f32,
    dy: &f32,
    t: Option<&mut [f32]>,
) {
    fourier_expand_image(a, *xi, *yi, b, *xo, *yo, *dx, *dy, t)
}
pub fn fouriershiftvolume(
    a: &mut [f32],
    x: &i32,
    y: &i32,
    z: &i32,
    dx: &f32,
    dy: &f32,
    dz: &f32,
    t: &mut [f32],
) {
    fourier_shift_volume(a, *x, *y, *z, *dx, *dy, *dz, t)
}
pub fn fourierreducevolume(
    a: &[f32],
    xi: &i32,
    yi: &i32,
    zi: &i32,
    b: &mut [f32],
    xo: &i32,
    yo: &i32,
    zo: &i32,
    dx: &f32,
    dy: &f32,
    dz: &f32,
    t: Option<&mut [f32]>,
) {
    fourier_reduce_volume(a, *xi, *yi, *zi, b, *xo, *yo, *zo, *dx, *dy, *dz, t)
}
pub fn fourierexpandvolume(
    a: &mut [f32],
    xi: &i32,
    yi: &i32,
    zi: &i32,
    b: &mut [f32],
    xo: &i32,
    yo: &i32,
    zo: &i32,
    dx: &f32,
    dy: &f32,
    dz: &f32,
    t: Option<&mut [f32]>,
) {
    fourier_expand_volume(a, *xi, *yi, *zi, b, *xo, *yo, *zo, *dx, *dy, *dz, t)
}
pub fn fourierringcorr(
    a: &[f32],
    b: &[f32],
    x: &i32,
    y: &i32,
    c: &mut [f32],
    max: &i32,
    d: &f32,
    t: &mut [f32],
) {
    fourier_ring_corr(a, b, *x, *y, c, *max, *d, t)
}
pub fn fouriercropsizes(
    s: &i32,
    f: &f32,
    p: &f32,
    min: &i32,
    l: &i32,
    full: &mut i32,
    crop: &mut i32,
    actual: &mut f32,
) -> i32 {
    fourier_crop_sizes(*s, *f, *p, *min, *l, full, crop, actual)
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
        fourier_shift_volume(&mut fft, 8, 4, 4, dx, dx, dz, &mut temp);
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
        fourier_expand_volume(
            &mut input,
            4,
            2,
            2,
            &mut out,
            8,
            4,
            4,
            0.,
            0.,
            0.,
            Some(&mut temp),
        );
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
            let got = fourier_crop_sizes(
                size,
                f32::from_bits(factor),
                0.01,
                16,
                15,
                &mut full,
                &mut crop,
                &mut actual,
            );
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
        fourier_shift_volume(&mut fft, 8, 4, 4, -0.375, 0.3125, -0.1875, &mut temp);
        let got = fft.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        assert_eq!(got, SHIFT.to_vec());

        let mut input = (0..160)
            .map(|i| ((i * 53) % 89) as f32 / 11.0 - 2.0)
            .collect::<Vec<f32>>();
        let mut out = vec![0_f32; 6 * 2 * 2];
        fourier_reduce_volume(
            &input,
            8,
            4,
            4,
            &mut out,
            4,
            2,
            2,
            0.,
            0.,
            0.,
            Some(&mut temp),
        );
        let got = out.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        assert_eq!(got, REDUCE.to_vec());
    }

    #[test]
    fn wrap_fft_slice_matches_even_and_odd_source_permutations() {
        let mut even = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let mut temporary = vec![0.; 4];
        wrap_fft_slice(&mut even, &mut temporary, 2, 4, 0);
        assert_eq!(
            even,
            vec![
                8., 9., 10., 11., 12., 13., 14., 15., 0., 1., 2., 3., 4., 5., 6., 7.
            ]
        );

        let mut odd = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        wrap_fft_slice(&mut odd, &mut temporary, 2, 3, 0);
        assert_eq!(odd, vec![8., 9., 10., 11., 0., 1., 2., 3., 4., 5., 6., 7.]);
        wrap_fft_slice(&mut odd, &mut temporary, 2, 3, 1);
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
        let (mut xp, mut yp, mut value) = ([0.0f32], [0.0f32], [0.0f32]);
        let (mut width, mut width_min) = ([0.0f32], [0.0f32]);
        xcorr_peak_find_width(
            &array,
            8,
            6,
            &mut xp,
            &mut yp,
            &mut value,
            Some(&mut width),
            Some(&mut width_min),
            1,
            0.,
        );
        assert_eq!(value[0], 10.);
        assert!((xp[0] - 2.1).abs() < 1.0e-5);
        assert_eq!(yp[0], 3.);
        assert!(width[0] > 0. && width_min[0] > 0.);
    }

    #[test]
    fn many_and_spaced_peak_selection_returns_interpolated_local_maxima() {
        let mut array = vec![0_f32; 10 * 8];
        array[3 + 3 * 10] = 9.;
        array[6 + 5 * 10] = 7.;
        let (mut xp, mut yp, mut peaks, mut count) = ([0.; 2], [0.; 2], [0.; 2], 0);
        assert_eq!(
            find_many_xcorr_peaks(
                &array, 10, 8, 0, 0, &mut xp, &mut yp, &mut peaks, 2, 3, &mut count
            ),
            0
        );
        assert_eq!(count, 2);
        assert_eq!(peaks, [9., 7.]);
        assert_eq!(
            find_spaced_xcorr_peaks(
                &array, 10, 0, 8, 0, 7, &mut xp, &mut yp, &mut peaks, 2, 2., &mut count, 0.
            ),
            0
        );
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
            let rc = find_many_xcorr_peaks(
                &a, nxdim, ny, ix_off, iy_off, &mut xp, &mut yp, &mut pk, 8, 32, &mut n,
            );
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
