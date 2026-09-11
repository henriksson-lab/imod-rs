//! Translation of `IMOD/libcfshr/taperpad.c`.
#![allow(dead_code)]
use crate::imod::libcfshr::b3dutil::{b3d_i_min, balanced_group_limits, num_omp_threads};
use crate::imod::libcfshr::robuststat::rs_fast_median_in_place;
use crate::imod::libcfshr::simplestat::sums_to_avg_sd_dbl;

use core::ffi::c_void;

const BYTE: i32 = 0;
const SHORT: i32 = 1;
const FLOAT: i32 = 2;
const USHORT: i32 = 6;
const RGB: i32 = 16;

/// C static `copyToCenter`.
unsafe fn copy_to_center(
    array: *mut c_void,
    typ: i32,
    nxbox: i32,
    nybox: i32,
    out: *mut f32,
    nxdim: i32,
    nx: i32,
    ny: i32,
    ixlo: *mut i32,
    ixhi: *mut i32,
    iylo: *mut i32,
    iyhi: *mut i32,
) {
    unsafe {
        *ixlo = (nx - nxbox) / 2;
        *ixhi = *ixlo + nxbox;
        *iylo = (ny - nybox) / 2;
        *iyhi = *iylo + nybox;
        for iy in (0..nybox).rev() {
            for ix in (0..nxbox).rev() {
                let ind = (ix + iy * nxbox) as usize;
                let value = match typ {
                    BYTE => *array.cast::<u8>().add(ind) as f32,
                    SHORT => *array.cast::<i16>().add(ind) as f32,
                    USHORT => *array.cast::<u16>().add(ind) as f32,
                    FLOAT => *array.cast::<f32>().add(ind),
                    _ => 0.,
                };
                *out.add((*ixlo + ix + (*iylo + iy) * nxdim) as usize) = value;
            }
        }
    }
}

/// C `sliceTaperOutPad`.
pub unsafe fn slice_taper_out_pad(
    array: *mut c_void,
    typ: i32,
    nxbox: i32,
    nybox: i32,
    out: *mut f32,
    nxdim: i32,
    nx: i32,
    ny: i32,
    ifmean: i32,
    dmeanin: f32,
) {
    unsafe {
        let (mut xl, mut xh, mut yl, mut yh) = (0, 0, 0, 0);
        copy_to_center(
            array, typ, nxbox, nybox, out, nxdim, nx, ny, &mut xl, &mut xh, &mut yl, &mut yh,
        );
        if nxbox == nx && nybox == ny {
            return;
        }
        let mean = if ifmean != 0 {
            dmeanin
        } else {
            slice_edge_mean(out, nxdim, xl, xh - 1, yl, yh - 1) as f32
        };
        let (mut xtop, mut ytop) = (nx - 1, ny - 1);
        if nx - xh > xl {
            xtop -= 1;
            for y in 0..ny {
                *out.add((nx - 1 + y * nxdim) as usize) = mean;
            }
        }
        if ny - yh > yl {
            ytop -= 1;
            for x in 0..nx {
                *out.add((x + (ny - 1) * nxdim) as usize) = mean;
            }
        }
        for y in yl..yh {
            let left = *out.add((xl + y * nxdim) as usize);
            let right = *out.add((xh - 1 + y * nxdim) as usize);
            for x in 0..xl {
                let w = x as f32 / xl as f32;
                *out.add((x + y * nxdim) as usize) = (1. - w) * mean + w * left;
                *out.add((xtop - x + y * nxdim) as usize) = (1. - w) * mean + w * right;
            }
        }
        for y in 0..yl {
            let w = y as f32 / yl as f32;
            let p = (1. - w) * mean;
            for x in 0..nx {
                *out.add((x + y * nxdim) as usize) = p + w * *out.add((x + yl * nxdim) as usize);
                *out.add((x + (ytop - y) * nxdim) as usize) =
                    p + w * *out.add((x + (yh - 1) * nxdim) as usize);
            }
        }
    }
}
pub unsafe fn taperoutpad(
    array: *mut c_void,
    nxbox: *mut i32,
    nybox: *mut i32,
    out: *mut f32,
    nxdim: *mut i32,
    nx: *mut i32,
    ny: *mut i32,
    ifmean: *mut i32,
    dmean: *mut f32,
) {
    unsafe {
        slice_taper_out_pad(
            array, FLOAT, *nxbox, *nybox, out, *nxdim, *nx, *ny, *ifmean, *dmean,
        )
    }
}

/// C `sliceTaperInPad`.
pub unsafe fn slice_taper_in_pad(
    input: *mut c_void,
    typ: i32,
    nxdimin: i32,
    ixstart: i32,
    ixend: i32,
    iystart: i32,
    iyend: i32,
    out: *mut f32,
    nxdimout: i32,
    nx: i32,
    ny: i32,
    nxtaper: i32,
    nytaper: i32,
) {
    unsafe {
        let nxbox = ixend + 1 - ixstart;
        let nybox = iyend + 1 - iystart;
        let xlow = nx / 2 - nxbox / 2 - 1;
        let xhigh = xlow + nxbox;
        let ylow = ny / 2 - nybox / 2 - 1;
        let yhigh = ylow + nybox;
        for y in (iystart..=iyend).rev() {
            for x in (ixstart..=ixend).rev() {
                let ind = (x + y * nxdimin) as usize;
                let v = match typ {
                    BYTE => *input.cast::<u8>().add(ind) as f32,
                    SHORT => *input.cast::<i16>().add(ind) as f32,
                    USHORT => *input.cast::<u16>().add(ind) as f32,
                    FLOAT => *input.cast::<f32>().add(ind),
                    RGB => {
                        let p = input.cast::<u8>().add(ind * 3);
                        *p as f32 + *p.add(1) as f32 + *p.add(2) as f32
                    }
                    _ => 0.,
                };
                *out.add((xhigh - (ixend - x) + (ylow + 1 + y - iystart) * nxdimout) as usize) = v;
            }
        }
        let mean = slice_edge_mean(out, nxdimout, xlow + 1, xhigh, ylow + 1, yhigh) as f32;
        if nxbox != nx || nybox != ny {
            for y in ylow + 1..=yhigh {
                for x in 0..=xlow {
                    *out.add((x + y * nxdimout) as usize) = mean;
                }
                for x in xhigh + 1..nx {
                    *out.add((x + y * nxdimout) as usize) = mean;
                }
            }
            for y in 0..=ylow {
                for x in 0..nx {
                    *out.add((x + y * nxdimout) as usize) = mean;
                }
            }
            for y in yhigh + 1..ny {
                for x in 0..nx {
                    *out.add((x + y * nxdimout) as usize) = mean;
                }
            }
        }
        for y in 0..(nybox + 1) / 2 {
            let mut fy = 1.;
            let mut xlim = nxtaper;
            if y < nytaper {
                fy = (y as f32 + 1.) / (nytaper as f32 + 1.);
                xlim = (nxbox + 1) / 2;
            }
            for x in 0..xlim {
                let mut fx = 1.;
                if x < nxtaper {
                    fx = (x as f32 + 1.) / (nxtaper as f32 + 1.);
                }
                let f = fx.min(fy);
                if f < 1. {
                    let x1 = x + 1 + xlow;
                    let mut x2 = xhigh - x;
                    let y1 = y + 1 + ylow;
                    let mut y2 = yhigh - y;
                    if x1 == x2 {
                        x2 = 0
                    }
                    if y1 == y2 {
                        y2 = 0
                    }
                    for (xx, yy) in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)] {
                        let q = out.add((xx + yy * nxdimout) as usize);
                        *q = f * (*q - mean) + mean;
                    }
                }
            }
        }
    }
}
pub unsafe fn taperinpad(
    array: *mut c_void,
    nxbox: *mut i32,
    nybox: *mut i32,
    out: *mut f32,
    nxdim: *mut i32,
    nx: *mut i32,
    ny: *mut i32,
    nxt: *mut i32,
    nyt: *mut i32,
) {
    unsafe {
        slice_taper_in_pad(
            array,
            FLOAT,
            *nxbox,
            0,
            *nxbox - 1,
            0,
            *nybox - 1,
            out,
            *nxdim,
            *nx,
            *ny,
            *nxt,
            *nyt,
        )
    }
}
pub unsafe fn taperinpadex(
    array: *mut c_void,
    nxdimin: *mut i32,
    x0: *mut i32,
    x1: *mut i32,
    y0: *mut i32,
    y1: *mut i32,
    out: *mut f32,
    nxdim: *mut i32,
    nx: *mut i32,
    ny: *mut i32,
    nxt: *mut i32,
    nyt: *mut i32,
) {
    unsafe {
        slice_taper_in_pad(
            array, FLOAT, *nxdimin, *x0, *x1, *y0, *y1, out, *nxdim, *nx, *ny, *nxt, *nyt,
        )
    }
}

/// C `sliceSmoothOutPad`.
pub unsafe fn slice_smooth_out_pad(
    array: *mut c_void,
    typ: i32,
    nxbox: i32,
    nybox: i32,
    out: *mut f32,
    nxdim: i32,
    nx: i32,
    ny: i32,
) {
    unsafe {
        let (mut xl, mut xh, mut yl, mut yh) = (0, 0, 0, 0);
        copy_to_center(
            array, typ, nxbox, nybox, out, nxdim, nx, ny, &mut xl, &mut xh, &mut yl, &mut yh,
        );
        let num_pad = [4, xl, nx - xh, yl, ny - yh].into_iter().max().unwrap();
        for ipad in 1..=num_pad {
            let x_line_lo = xl - ipad;
            let x_line_hi = xh + ipad - 1;
            let xmin = x_line_lo.max(0);
            let x_lim_lo = (x_line_lo + 1).max(0);
            let xmax = x_line_hi.min(nx - 1);
            let x_lim_hi = (x_line_hi - 1).min(nx - 1);
            let y_line_lo = yl - ipad;
            let y_line_hi = yh + ipad - 1;
            let ymin = y_line_lo.max(0);
            let y_lim_lo = (y_line_lo + 1).max(0);
            let ymax = y_line_hi.min(ny - 1);
            let y_lim_hi = (y_line_hi - 1).min(ny - 1);

            let mut diry = 1;
            for iy in [y_line_lo, y_line_hi] {
                if iy < 0 || iy >= ny {
                    diry = -1;
                    continue;
                }
                for ix in xmin..=xmax {
                    let mut xside = -1;
                    let mut xstart = ix - ipad;
                    let mut ystart = 0;
                    let mut yend = 0;
                    if xstart < x_lim_lo {
                        xstart = x_lim_lo;
                        if xstart != 0 {
                            ystart = iy + 2 * diry;
                            yend = ystart + diry * (xstart - (ix - ipad) - 1);
                            yend = (y_lim_hi - 1).min((y_lim_lo + 1).max(yend));
                            if diry * ystart <= diry * yend {
                                xside = xstart;
                            }
                        }
                    }
                    let mut xend = ix + ipad;
                    if xend > x_lim_hi {
                        xend = x_lim_hi;
                        if xend < nx && xside < 0 {
                            ystart = iy + 2 * diry;
                            yend = ystart + diry * (ix + ipad - xend - 1);
                            yend = (y_lim_hi - 1).min((y_lim_lo + 1).max(yend));
                            if diry * ystart <= diry * yend {
                                xside = xend;
                            }
                        }
                    }
                    let mut sum = 0.;
                    let mut nsum = xend + 1 - xstart;
                    for x in xstart..=xend {
                        sum += *out.add((x + nxdim * (iy + diry)) as usize);
                    }
                    if xside >= 0 {
                        let mut y = ystart;
                        while diry * y <= diry * yend {
                            sum += *out.add((xside + nxdim * y) as usize);
                            nsum += 1;
                            y += diry;
                        }
                    }
                    *out.add((ix + nxdim * iy) as usize) = sum / nsum as f32;
                }
                diry = -1;
            }

            let mut dirx = 1;
            for ix in [x_line_lo, x_line_hi] {
                if ix < 0 || ix >= nx {
                    dirx = -1;
                    continue;
                }
                for iy in ymin..=ymax {
                    let mut yside = -1;
                    let mut ystart = iy - ipad;
                    let mut xstart = 0;
                    let mut xend = 0;
                    if ystart < y_lim_lo {
                        ystart = y_lim_lo;
                        if ystart != 0 {
                            xstart = ix + 2 * dirx;
                            xend = xstart + dirx * (ystart - (iy - ipad) - 1);
                            xend = (x_lim_hi - 1).min((x_lim_lo + 1).max(xend));
                            if dirx * xstart <= dirx * xend {
                                yside = ystart;
                            }
                        }
                    }
                    let mut yend = iy + ipad;
                    if yend > y_lim_hi {
                        yend = y_lim_hi;
                        if yend < ny && yside < 0 {
                            xstart = ix + 2 * dirx;
                            xend = xstart + dirx * (iy + ipad - yend - 1);
                            xend = (x_lim_hi - 1).min((x_lim_lo + 1).max(xend));
                            if dirx * xstart <= dirx * xend {
                                yside = yend;
                            }
                        }
                    }
                    let mut sum = 0.;
                    let mut nsum = yend + 1 - ystart;
                    for y in ystart..=yend {
                        sum += *out.add((ix + dirx + nxdim * y) as usize);
                    }
                    if yside >= 0 {
                        let mut x = xstart;
                        while dirx * x <= dirx * xend {
                            sum += *out.add((x + nxdim * yside) as usize);
                            nsum += 1;
                            x += dirx;
                        }
                    }
                    *out.add((ix + nxdim * iy) as usize) = sum / nsum as f32;
                }
                dirx = -1;
            }
        }
    }
}
pub unsafe fn smoothoutpad(
    array: *mut c_void,
    nxbox: *mut i32,
    nybox: *mut i32,
    out: *mut f32,
    nxdim: *mut i32,
    nx: *mut i32,
    ny: *mut i32,
) {
    unsafe { slice_smooth_out_pad(array, FLOAT, *nxbox, *nybox, out, *nxdim, *nx, *ny) }
}

/// C static `getRunningMeanSD` (`taperpad.c:838`).
///
/// Function to get mean and SD in a smaple box and compute it all along an edge
unsafe fn get_running_mean_sd(
    box_start: *mut f32,
    noise_rows: i32,
    noise_length: i32,
    row_stride: i32,
    len_stride: i32,
    nbox: i32,
    dmean: f32,
    samples: *mut f32,
    sample_sq: *mut f32,
    num_sample: i32,
    means: *mut f32,
    sds: *mut f32,
) {
    unsafe {
        let mut samp_sum = 0.0_f64;
        let mut samp_sq_sum = 0.0_f64;
        let mut val: f32;
        let mut val_sq: f32;
        let mut tmp_mean: f32 = 0.;
        let mut tmp_sd: f32 = 0.;
        let mut sample_ind: i32;
        let mut len: i32;
        let mut mean_ind: i32;
        let mut mean_base = noise_length / 2;

        // Load the box and get first mean/SD
        sample_ind = 0;
        len = 0;
        while len < noise_length {
            for row in 0..noise_rows {
                val = *box_start.offset((len * len_stride + row * row_stride) as isize) - dmean;
                val_sq = val * val;
                *samples.offset(sample_ind as isize) = val;
                *sample_sq.offset(sample_ind as isize) = val_sq;
                sample_ind += 1;
                samp_sum += val as f64;
                samp_sq_sum += val_sq as f64;
            }
            len += 1;
        }
        sums_to_avg_sd_dbl(
            samp_sum,
            samp_sq_sum,
            num_sample,
            1,
            &mut tmp_mean,
            &mut tmp_sd,
        );
        *means.offset(mean_base as isize) = dmean + tmp_mean;
        *sds.offset(mean_base as isize) = tmp_sd;

        // Loop along the length, pulling samples out of sum then adding new ones in
        sample_ind = 0;
        mean_ind = mean_base + 1;
        while len < nbox - noise_length {
            for row in 0..noise_rows {
                samp_sum -= *samples.offset(sample_ind as isize) as f64;
                samp_sq_sum -= *sample_sq.offset(sample_ind as isize) as f64;
                val = *box_start.offset((len * len_stride + row * row_stride) as isize) - dmean;
                val_sq = val * val;
                *samples.offset(sample_ind as isize) = val;
                *sample_sq.offset(sample_ind as isize) = val_sq;
                sample_ind += 1;
                samp_sum += val as f64;
                samp_sq_sum += val_sq as f64;
            }
            sample_ind %= num_sample;
            sums_to_avg_sd_dbl(
                samp_sum,
                samp_sq_sum,
                num_sample,
                1,
                &mut tmp_mean,
                &mut tmp_sd,
            );
            *means.offset(mean_ind as isize) = dmean + tmp_mean;
            *sds.offset(mean_ind as isize) = tmp_sd;
            mean_ind += 1;
            len += 1;
        }

        // Copy the endpoints to complete the arrays
        for l in 0..mean_base {
            *means.offset(l as isize) = *means.offset(mean_base as isize);
            *sds.offset(l as isize) = *sds.offset(mean_base as isize);
        }
        mean_base = mean_ind - 1;
        for l in mean_ind..nbox {
            *means.offset(l as isize) = *means.offset(mean_base as isize);
            *sds.offset(l as isize) = *sds.offset(mean_base as isize);
        }
    }
}

const MAX_NOISE_ROWS: i32 = 5;
const MAX_NOISE_LENGTH: i32 = 120;
const MAX_SAMPLES: i32 = MAX_NOISE_ROWS * MAX_NOISE_LENGTH;
const SNTP_MAX_THREADS: i32 = 16;

/// C file-scope `static int pseudoVals[SNTP_MAX_THREADS]` (`taperpad.c:580`).
///
/// It is `static`, so the generator state carries over from one
/// `sliceNoiseTaperPad` call to the next within a process.
static mut PSEUDO_VALS: [i32; SNTP_MAX_THREADS as usize] = [
    123456, 654321, 368341, 789234, 234561, 543216, 683413, 892347, 345612, 432165, 834136, 923478,
    456123, 321654, 341368, 234789,
];

/// C `sliceNoiseTaperPad` (`taperpad.c:569`).
///
/// The `#pragma omp parallel for` loops are run sequentially here: every thread
/// owns a disjoint Y range (`balancedGroupLimits`) and its own generator seed
/// `pseudoVals[thread]`, so running the thread loop in order reproduces the
/// parallel result exactly.  `numThreads` still comes from `numOMPthreads`,
/// which is what selects both the Y partition and the set of seeds used.
///
/// Deviation, deliberate and the only one: the six scratch arrays are carved
/// out of a locally allocated buffer laid out exactly as the source lays out
/// `temp` (`fracx`, `meanXpart`, `fracy`, `meanYpart`, `means`, `SDs`) instead
/// of out of the caller's `temp`.  The source needs
/// `2 * ixlo + 2 * iylo + 2 * max(nxbox, nybox)` floats there; the Rust
/// `newstack` caller (`newstack.rs:2600`) currently passes a buffer sized only
/// `2 * (nxFSpad / 2 + 2)`, so writing through `temp` would run off the end of
/// its `Vec`.  Nothing the caller does reads `temp` back, so the values are
/// unaffected.
pub unsafe fn slice_noise_taper_pad(
    array: *mut c_void,
    typ: i32,
    nxbox: i32,
    nybox: i32,
    brray: *mut f32,
    nxdim: i32,
    nx: i32,
    ny: i32,
    noise_length: i32,
    noise_rows: i32,
    temp: *mut f32,
) {
    unsafe {
        let (mut ixlo, mut ixhi, mut iylo, mut iyhi) = (0, 0, 0, 0);
        let mut nxtop: i32;
        let mut nytop: i32;
        let num_samples: i32;
        let mut ix_base: i32;
        let mut pseudo: i32;
        let mut edge_xadd: i32 = 0;
        let mut edge_yadd: i32 = 0;
        let dmean: f32;
        let ran_fac: f32;
        let mut frac_min: f32;
        let mut corn_mean: f32;
        let mut corn_sd: f32;
        let max_sds: f32 = 1.73f32;
        let mut corn_xmean: f32;
        let mut corn_ymean: f32;
        let mut corn_xsd: f32;
        let mut corn_ysd: f32;
        let save_corn_mean: f32;
        let save_corn_sd: f32;
        let mut wgt_sum: f32;
        let mut samples = [0.0f32; MAX_SAMPLES as usize];
        let mut sample_sq = [0.0f32; MAX_SAMPLES as usize];
        let fracx: *mut f32;
        let fracy: *mut f32;
        let mean_xpart: *mut f32;
        let mean_ypart: *mut f32;
        let means: *mut f32;
        let sds: *mut f32;
        let mut max_threads: i32 = 4;
        let num_threads: i32;
        let mut iy_start = [0i32; SNTP_MAX_THREADS as usize];
        let mut iy_end = [0i32; SNTP_MAX_THREADS as usize];

        // Do not enforce those defined limits, take the given number of rows
        // and limit the length of the sample by the product
        let noise_rows = b3d_i_min(&[noise_rows, nxbox / 2, nybox / 2]);
        let noise_length =
            2 * (b3d_i_min(&[noise_length, MAX_SAMPLES / noise_rows, nxbox, nybox]) / 2);
        num_samples = noise_rows * noise_length;

        copy_to_center(
            array, typ, nxbox, nybox, brray, nxdim, nx, ny, &mut ixlo, &mut ixhi, &mut iylo,
            &mut iyhi,
        );

        // Do the taper if there is any padding
        if nxbox == nx && nybox == ny {
            return;
        }

        // Need a mean from which to compute deviations
        if nxbox > 400 {
            edge_xadd = 1 + nxbox / 2048;
        }
        if nybox > 400 {
            edge_yadd = 1 + nybox / 2048;
        }
        dmean = slice_edge_mean(
            brray,
            nxdim,
            ixlo + edge_xadd,
            (ixhi - 1) - edge_xadd,
            iylo + edge_yadd,
            (iyhi - 1) - edge_yadd,
        ) as f32;
        nxtop = nx - 1;
        nytop = ny - 1;

        // Set up fractions and mean components.  See the note above: this is
        // the source's `temp` layout, backed by a local buffer.
        let _ = temp;
        let mut scratch = vec![
            0.0f32;
            (2 * ixlo + 2 * iylo + 2 * if nxbox > nybox { nxbox } else { nybox })
                as usize
        ];
        fracx = scratch.as_mut_ptr();
        mean_xpart = fracx.offset(ixlo as isize);
        fracy = mean_xpart.offset(ixlo as isize);
        mean_ypart = fracy.offset(iylo as isize);
        means = mean_ypart.offset(iylo as isize);
        sds = means.offset(if nxbox > nybox { nxbox } else { nybox } as isize);
        for iy in 0..iylo {
            *fracy.offset(iy as isize) = iy as f32 / iylo as f32;
            *mean_ypart.offset(iy as isize) =
                ((1. - *fracy.offset(iy as isize) as f64) * dmean as f64) as f32;
        }
        for ix in 0..ixlo {
            *fracx.offset(ix as isize) = ix as f32 / ixlo as f32;
            *mean_xpart.offset(ix as isize) =
                ((1. - *fracx.offset(ix as isize) as f64) * dmean as f64) as f32;
        }

        //  if there is a mismatch between left and right, add a column on
        //  right; similarly for bottom versus top, add a row on top
        if nx - ixhi > ixlo {
            nxtop -= 1;
            for iy in 0..ny {
                *brray.offset((nx - 1 + iy * nxdim) as isize) = dmean;
            }
        }
        if ny - iyhi > iylo {
            nytop -= 1;
            for ix in 0..nx {
                *brray.offset((ix + (ny - 1) * nxdim) as isize) = dmean;
            }
        }

        // Multiply 20-bit random numbers by this factor to get a range of 2 * maxSDs
        ran_fac = max_sds / 0x7FFFF as f32;

        // Set up the number of threads and divide the first Y range (iylo) into groups
        max_threads = if iylo < max_threads {
            iylo
        } else {
            max_threads
        };
        // The source computes a thread count from the image area and clamps
        // it, then immediately overwrites it with `numOMPthreads(maxThreads)`.
        // Both statements are kept so the arithmetic stays visible, but only
        // the second one reaches `numThreads`.
        let mut nthr = (((((ixlo + iylo) * (nx + ny)) as f64).sqrt() / 170.) + 0.5).floor() as i32;
        nthr = {
            let clamped = if max_threads < nthr {
                max_threads
            } else {
                nthr
            };
            if 1 > clamped { 1 } else { clamped }
        };
        let _ = nthr;
        nthr = num_omp_threads(max_threads);
        nthr = if nthr < SNTP_MAX_THREADS {
            nthr
        } else {
            SNTP_MAX_THREADS
        };
        num_threads = if iylo < nthr { iylo } else { nthr };
        for thread in 0..num_threads {
            balanced_group_limits(
                iylo,
                num_threads,
                thread,
                &mut iy_start[thread as usize],
                &mut iy_end[thread as usize],
            );
        }

        // Get mean/SD and fill bottom
        get_running_mean_sd(
            brray.offset((ixlo + iylo * nxdim) as isize),
            noise_rows,
            noise_length,
            nxdim,
            1,
            nxbox,
            dmean,
            samples.as_mut_ptr(),
            sample_sq.as_mut_ptr(),
            num_samples,
            means,
            sds,
        );

        for thread in 0..num_threads {
            pseudo = PSEUDO_VALS[thread as usize];
            for iy in iy_start[thread as usize]..=iy_end[thread as usize] {
                ix_base = ixlo + iy * nxdim;
                for ix in 0..nxbox {
                    // This is a linear (mixed?) congruential generator with a
                    // period of 2^20 its deficiencies (small period of low
                    // order bits) are of no concern here
                    pseudo = (197i32.wrapping_mul(pseudo.wrapping_add(1))) & 0xFFFFF;
                    *brray.offset((ix + ix_base) as isize) = (*means.offset(ix as isize)
                        + (pseudo as f32 * ran_fac - max_sds) * *sds.offset(ix as isize))
                        * *fracy.offset(iy as isize)
                        + *mean_ypart.offset(iy as isize);
                }
            }
            PSEUDO_VALS[thread as usize] = pseudo;
        }

        // Save left-hand corner mean for the end, and right mean before getting new means
        save_corn_mean = *means;
        save_corn_sd = *sds;
        corn_xmean = *means.offset((nxbox - 1) as isize);
        corn_xsd = *sds.offset((nxbox - 1) as isize);

        // Right side mean/SD, bottom right corner then right side
        get_running_mean_sd(
            brray.offset((ixlo + nxbox - noise_rows + iylo * nxdim) as isize),
            noise_rows,
            noise_length,
            1,
            nxdim,
            nybox,
            dmean,
            samples.as_mut_ptr(),
            sample_sq.as_mut_ptr(),
            num_samples,
            means,
            sds,
        );

        corn_ymean = *means;
        corn_ysd = *sds;
        // FILL_CORNER(nxtop - ix, iy)
        pseudo = PSEUDO_VALS[0];
        for iy in 0..iylo {
            for ix in 0..ixlo {
                frac_min = if *fracx.offset(ix as isize) < *fracy.offset(iy as isize) {
                    *fracx.offset(ix as isize)
                } else {
                    *fracy.offset(iy as isize)
                };
                wgt_sum = (if 0.01f64
                    > (*fracx.offset(ix as isize) + *fracy.offset(iy as isize)) as f64
                {
                    0.01f64
                } else {
                    (*fracx.offset(ix as isize) + *fracy.offset(iy as isize)) as f64
                }) as f32;
                corn_mean = (corn_xmean * *fracx.offset(ix as isize)
                    + corn_ymean * *fracy.offset(iy as isize))
                    / wgt_sum;
                corn_sd = (corn_xsd * *fracx.offset(ix as isize)
                    + corn_ysd * *fracy.offset(iy as isize))
                    / wgt_sum;
                pseudo = (197i32.wrapping_mul(pseudo.wrapping_add(1))) & 0xFFFFF;
                *brray.offset(((nxtop - ix) + (iy) * nxdim) as isize) =
                    (((corn_mean + (pseudo as f32 * ran_fac - max_sds) * corn_sd) * frac_min)
                        as f64
                        + dmean as f64 * (1. - frac_min as f64)) as f32;
            }
        }
        PSEUDO_VALS[0] = pseudo;

        for thread in 0..num_threads {
            balanced_group_limits(
                nybox,
                num_threads,
                thread,
                &mut iy_start[thread as usize],
                &mut iy_end[thread as usize],
            );
        }

        for thread in 0..num_threads {
            pseudo = PSEUDO_VALS[thread as usize];
            for iy in iy_start[thread as usize]..=iy_end[thread as usize] {
                ix_base = nxtop + (iy + iylo) * nxdim;
                for ix in 0..ixlo {
                    pseudo = (197i32.wrapping_mul(pseudo.wrapping_add(1))) & 0xFFFFF;
                    *brray.offset((ix_base - ix) as isize) = (*means.offset(iy as isize)
                        + (pseudo as f32 * ran_fac - max_sds) * *sds.offset(iy as isize))
                        * *fracx.offset(ix as isize)
                        + *mean_xpart.offset(ix as isize);
                }
            }
            PSEUDO_VALS[thread as usize] = pseudo;
        }

        corn_ymean = *means.offset((nybox - 1) as isize);
        corn_ysd = *sds.offset((nybox - 1) as isize);

        // Top mean/SD, top right corner and top side
        get_running_mean_sd(
            brray.offset((ixlo + (iylo + nybox - noise_rows) * nxdim) as isize),
            noise_rows,
            noise_length,
            nxdim,
            1,
            nxbox,
            dmean,
            samples.as_mut_ptr(),
            sample_sq.as_mut_ptr(),
            num_samples,
            means,
            sds,
        );
        corn_xmean = *means.offset((nxbox - 1) as isize);
        corn_xsd = *sds.offset((nxbox - 1) as isize);
        // FILL_CORNER(nxtop - ix, nytop - iy)
        pseudo = PSEUDO_VALS[0];
        for iy in 0..iylo {
            for ix in 0..ixlo {
                frac_min = if *fracx.offset(ix as isize) < *fracy.offset(iy as isize) {
                    *fracx.offset(ix as isize)
                } else {
                    *fracy.offset(iy as isize)
                };
                wgt_sum = (if 0.01f64
                    > (*fracx.offset(ix as isize) + *fracy.offset(iy as isize)) as f64
                {
                    0.01f64
                } else {
                    (*fracx.offset(ix as isize) + *fracy.offset(iy as isize)) as f64
                }) as f32;
                corn_mean = (corn_xmean * *fracx.offset(ix as isize)
                    + corn_ymean * *fracy.offset(iy as isize))
                    / wgt_sum;
                corn_sd = (corn_xsd * *fracx.offset(ix as isize)
                    + corn_ysd * *fracy.offset(iy as isize))
                    / wgt_sum;
                pseudo = (197i32.wrapping_mul(pseudo.wrapping_add(1))) & 0xFFFFF;
                *brray.offset(((nxtop - ix) + (nytop - iy) * nxdim) as isize) =
                    (((corn_mean + (pseudo as f32 * ran_fac - max_sds) * corn_sd) * frac_min)
                        as f64
                        + dmean as f64 * (1. - frac_min as f64)) as f32;
            }
        }
        PSEUDO_VALS[0] = pseudo;

        for thread in 0..num_threads {
            balanced_group_limits(
                iylo,
                num_threads,
                thread,
                &mut iy_start[thread as usize],
                &mut iy_end[thread as usize],
            );
        }

        for thread in 0..num_threads {
            pseudo = PSEUDO_VALS[thread as usize];
            for iy in iy_start[thread as usize]..=iy_end[thread as usize] {
                ix_base = ixlo + (nytop - iy) * nxdim;
                for ix in 0..nxbox {
                    pseudo = (197i32.wrapping_mul(pseudo.wrapping_add(1))) & 0xFFFFF;
                    *brray.offset((ix + ix_base) as isize) = (*means.offset(ix as isize)
                        + (pseudo as f32 * ran_fac - max_sds) * *sds.offset(ix as isize))
                        * *fracy.offset(iy as isize)
                        + *mean_ypart.offset(iy as isize);
                }
            }
            PSEUDO_VALS[thread as usize] = pseudo;
        }

        corn_xmean = *means;
        corn_xsd = *sds;

        // Left side mean, then top left corner, then left side
        get_running_mean_sd(
            brray.offset((ixlo + iylo * nxdim) as isize),
            noise_rows,
            noise_length,
            1,
            nxdim,
            nybox,
            dmean,
            samples.as_mut_ptr(),
            sample_sq.as_mut_ptr(),
            num_samples,
            means,
            sds,
        );
        corn_ymean = *means.offset((nybox - 1) as isize);
        corn_ysd = *sds.offset((nybox - 1) as isize);
        // FILL_CORNER(ix, nytop - iy)
        pseudo = PSEUDO_VALS[0];
        for iy in 0..iylo {
            for ix in 0..ixlo {
                frac_min = if *fracx.offset(ix as isize) < *fracy.offset(iy as isize) {
                    *fracx.offset(ix as isize)
                } else {
                    *fracy.offset(iy as isize)
                };
                wgt_sum = (if 0.01f64
                    > (*fracx.offset(ix as isize) + *fracy.offset(iy as isize)) as f64
                {
                    0.01f64
                } else {
                    (*fracx.offset(ix as isize) + *fracy.offset(iy as isize)) as f64
                }) as f32;
                corn_mean = (corn_xmean * *fracx.offset(ix as isize)
                    + corn_ymean * *fracy.offset(iy as isize))
                    / wgt_sum;
                corn_sd = (corn_xsd * *fracx.offset(ix as isize)
                    + corn_ysd * *fracy.offset(iy as isize))
                    / wgt_sum;
                pseudo = (197i32.wrapping_mul(pseudo.wrapping_add(1))) & 0xFFFFF;
                *brray.offset(((ix) + (nytop - iy) * nxdim) as isize) =
                    (((corn_mean + (pseudo as f32 * ran_fac - max_sds) * corn_sd) * frac_min)
                        as f64
                        + dmean as f64 * (1. - frac_min as f64)) as f32;
            }
        }
        PSEUDO_VALS[0] = pseudo;

        for thread in 0..num_threads {
            balanced_group_limits(
                nybox,
                num_threads,
                thread,
                &mut iy_start[thread as usize],
                &mut iy_end[thread as usize],
            );
        }

        for thread in 0..num_threads {
            pseudo = PSEUDO_VALS[thread as usize];
            for iy in iy_start[thread as usize]..=iy_end[thread as usize] {
                ix_base = (iy + iylo) * nxdim;
                for ix in 0..ixlo {
                    pseudo = (197i32.wrapping_mul(pseudo.wrapping_add(1))) & 0xFFFFF;
                    *brray.offset((ix_base + ix) as isize) = (*means.offset(iy as isize)
                        + (pseudo as f32 * ran_fac - max_sds) * *sds.offset(iy as isize))
                        * *fracx.offset(ix as isize)
                        + *mean_xpart.offset(ix as isize);
                }
            }
            PSEUDO_VALS[thread as usize] = pseudo;
        }

        // Finish up with bottom left corner
        corn_ymean = *means;
        corn_ysd = *sds;
        corn_xmean = save_corn_mean;
        corn_xsd = save_corn_sd;

        // FILL_CORNER(ix, iy)
        pseudo = PSEUDO_VALS[0];
        for iy in 0..iylo {
            for ix in 0..ixlo {
                frac_min = if *fracx.offset(ix as isize) < *fracy.offset(iy as isize) {
                    *fracx.offset(ix as isize)
                } else {
                    *fracy.offset(iy as isize)
                };
                wgt_sum = (if 0.01f64
                    > (*fracx.offset(ix as isize) + *fracy.offset(iy as isize)) as f64
                {
                    0.01f64
                } else {
                    (*fracx.offset(ix as isize) + *fracy.offset(iy as isize)) as f64
                }) as f32;
                corn_mean = (corn_xmean * *fracx.offset(ix as isize)
                    + corn_ymean * *fracy.offset(iy as isize))
                    / wgt_sum;
                corn_sd = (corn_xsd * *fracx.offset(ix as isize)
                    + corn_ysd * *fracy.offset(iy as isize))
                    / wgt_sum;
                pseudo = (197i32.wrapping_mul(pseudo.wrapping_add(1))) & 0xFFFFF;
                *brray.offset(((ix) + (iy) * nxdim) as isize) =
                    (((corn_mean + (pseudo as f32 * ran_fac - max_sds) * corn_sd) * frac_min)
                        as f64
                        + dmean as f64 * (1. - frac_min as f64)) as f32;
            }
        }
        PSEUDO_VALS[0] = pseudo;
    }
}

pub unsafe fn slicenoisetaperpad(
    array: *mut f32,
    nxbox: *mut i32,
    nybox: *mut i32,
    out: *mut f32,
    nxdim: *mut i32,
    nx: *mut i32,
    ny: *mut i32,
    len: *mut i32,
    rows: *mut i32,
    temp: *mut f32,
) {
    unsafe {
        slice_noise_taper_pad(
            array.cast(),
            FLOAT,
            *nxbox,
            *nybox,
            out,
            *nxdim,
            *nx,
            *ny,
            *len,
            *rows,
            temp,
        )
    }
}

/// C `sliceEdgeMean`.
pub unsafe fn slice_edge_mean(a: *mut f32, nxdim: i32, xl: i32, xh: i32, yl: i32, yh: i32) -> f64 {
    unsafe {
        let mut sum = 0.;
        for x in xl..=xh {
            sum +=
                *a.add((x + yl * nxdim) as usize) as f64 + *a.add((x + yh * nxdim) as usize) as f64
        }
        for y in yl + 1..yh {
            sum +=
                *a.add((xl + y * nxdim) as usize) as f64 + *a.add((xh + y * nxdim) as usize) as f64
        }
        sum / (2 * (xh - xl + yh - yl)) as f64
    }
}
pub unsafe fn sliceedgemean(
    a: *mut f32,
    n: *mut i32,
    xl: *mut i32,
    xh: *mut i32,
    yl: *mut i32,
    yh: *mut i32,
) -> f64 {
    unsafe { slice_edge_mean(a, *n, *xl - 1, *xh - 1, *yl - 1, *yh - 1) }
}
/// C `imageEdgeMean`.
pub unsafe fn image_edge_mean(
    a: *mut c_void,
    typ: i32,
    n: i32,
    xl: i32,
    xh: i32,
    yl: i32,
    yh: i32,
) -> f32 {
    unsafe {
        let mut v = Vec::with_capacity((2 * (xh - xl + yh - yl)) as usize);
        for x in xl..=xh {
            for y in [yl, yh] {
                let i = (x + y * n) as usize;
                v.push(match typ {
                    BYTE => *a.cast::<u8>().add(i) as f64,
                    SHORT => *a.cast::<i16>().add(i) as f64,
                    USHORT => *a.cast::<u16>().add(i) as f64,
                    FLOAT => *a.cast::<f32>().add(i) as f64,
                    _ => 0.,
                })
            }
        }
        for y in yl + 1..yh {
            for x in [xl, xh] {
                let i = (x + y * n) as usize;
                v.push(match typ {
                    BYTE => *a.cast::<u8>().add(i) as f64,
                    SHORT => *a.cast::<i16>().add(i) as f64,
                    USHORT => *a.cast::<u16>().add(i) as f64,
                    FLOAT => *a.cast::<f32>().add(i) as f64,
                    _ => 0.,
                })
            }
        }
        (v.iter().sum::<f64>() / v.len() as f64) as f32
    }
}
/// C `sliceEdgeMedian`.
/// `sliceEdgeMedian` (`taperpad.c:1002`).
///
/// Translated statement by statement.  The previous version took
/// `q[q.len() / 2]` as each side's median, which differs from the source two
/// ways: `rsFastMedianInPlace` (`robuststat.c:187-192`) uses the 1-based
/// `percentileFloat((n+1)/2)` and, for an **even** count, averages it with
/// `percentileFloat(n/2+1)` and rounds back to float; and `percentileFloat`
/// returns 0 for a non-positive count, so a side with no samples contributes
/// zero rather than indexing an empty slice.  On a 64x48 image every edge has
/// an even sample count, so the old form returned 129.78 where the reference
/// gives 129.38, and a 2-line region aborted the process.
///
/// Note the source reuses one `samples` buffer: with `meanOfSides` clear, the
/// four loops append into it and only `median4` — the median of everything
/// collected — is returned.
pub unsafe fn slice_edge_median(
    array: *mut f32,
    nxdim: i32,
    ixlo: i32,
    ixhi: i32,
    iylo: i32,
    iyhi: i32,
    mean_of_sides: i32,
) -> f64 {
    unsafe {
        const MAX_MED_SAMPLE: i32 = 10000;
        let mut samples = vec![0.0_f32; MAX_MED_SAMPLE as usize];
        let mut num_sample = 0_i32;
        let (mut median1, mut median2, mut median3, mut median4) = (0.0_f32, 0.0, 0.0, 0.0);
        let num_on_edge = 2 * ((ixhi - ixlo) + (iyhi + 1 - iylo));
        let samp_interval = (num_on_edge + MAX_MED_SAMPLE - 1) / MAX_MED_SAMPLE;

        let mut ix = ixlo;
        while ix <= ixhi {
            samples[num_sample as usize] = *array.add((ix + iylo * nxdim) as usize);
            num_sample += 1;
            ix += samp_interval;
        }
        if mean_of_sides != 0 {
            rs_fast_median_in_place(samples.as_mut_ptr(), num_sample, &mut median1);
            num_sample = 0;
        }
        let mut ix = ixlo;
        while ix <= ixhi {
            samples[num_sample as usize] = *array.add((ix + iyhi * nxdim) as usize);
            num_sample += 1;
            ix += samp_interval;
        }
        if mean_of_sides != 0 {
            rs_fast_median_in_place(samples.as_mut_ptr(), num_sample, &mut median2);
            num_sample = 0;
        }
        let mut iy = iylo + 1;
        while iy < iyhi {
            samples[num_sample as usize] = *array.add((ixlo + iy * nxdim) as usize);
            num_sample += 1;
            iy += samp_interval;
        }
        if mean_of_sides != 0 {
            rs_fast_median_in_place(samples.as_mut_ptr(), num_sample, &mut median3);
            num_sample = 0;
        }
        let mut iy = iylo + 1;
        while iy < iyhi {
            samples[num_sample as usize] = *array.add((ixhi + iy * nxdim) as usize);
            num_sample += 1;
            iy += samp_interval;
        }
        rs_fast_median_in_place(samples.as_mut_ptr(), num_sample, &mut median4);
        if mean_of_sides != 0 {
            return ((median1 + median2 + median3 + median4) / 4.) as f64;
        }
        median4 as f64
    }
}
pub unsafe fn sliceedgemedian(
    a: *mut f32,
    n: *mut i32,
    xl: *mut i32,
    xh: *mut i32,
    yl: *mut i32,
    yh: *mut i32,
    mean: *mut i32,
) -> f64 {
    unsafe { slice_edge_median(a, *n, *xl - 1, *xh - 1, *yl - 1, *yh - 1, *mean) }
}
/// C `sliceSplitFill`.
pub unsafe fn slice_split_fill(
    a: *mut f32,
    nxbox: i32,
    nybox: i32,
    out: *mut f32,
    n: i32,
    nx: i32,
    ny: i32,
    iffill: i32,
    fillin: f32,
) {
    unsafe {
        let d =
            (0..nxbox * nybox).map(|i| *a.add(i as usize)).sum::<f32>() / (nxbox * nybox) as f32;
        let mut fill = d;
        let mut bias = d;
        if nxbox != nx || nybox != ny {
            if iffill != 0 {
                fill = fillin
            } else {
                let mut s = 0.;
                for x in 0..nxbox {
                    s += *a.add(x as usize) + *a.add((x + (nybox - 1) * nxbox) as usize)
                }
                for y in 1..nybox - 1 {
                    s += *a.add((y * nxbox) as usize) + *a.add((nxbox - 1 + y * nxbox) as usize)
                }
                fill = s / (2 * nxbox + 2 * (nybox - 2)) as f32;
            }
            bias = fill + (d - fill) * (nxbox * nybox) as f32 / (nx * ny) as f32;
            for y in 0..ny {
                for x in 0..nx {
                    *out.add((x + y * n) as usize) = fill - bias;
                }
            }
        }
        for y in 0..nybox {
            for x in 0..nxbox {
                let xx = (x - nxbox / 2 + nx) % nx;
                let yy = (y - nybox / 2 + ny) % ny;
                *out.add((xx + yy * n) as usize) = *a.add((x + y * nxbox) as usize) - bias;
            }
        }
    }
}
pub unsafe fn splitfill(
    a: *mut f32,
    nxbox: *mut i32,
    nybox: *mut i32,
    out: *mut f32,
    n: *mut i32,
    nx: *mut i32,
    ny: *mut i32,
    iffill: *mut i32,
    fill: *mut f32,
) {
    unsafe { slice_split_fill(a, *nxbox, *nybox, out, *n, *nx, *ny, *iffill, *fill) }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GOLD_PAD_FIRST: [u32; 165] = [
        0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000,
        0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0xc640e400,
        0xc640e400, 0x419d0000, 0x41526c9c, 0xc105b166, 0x40e7c7bc, 0xc0d1e86c, 0xc1152c90,
        0x427e4dc9, 0xc1ab9804, 0x41a5f9eb, 0x427334da, 0xc1b67e44, 0x419d0000, 0x419d0000,
        0xc640e400, 0xc640e400, 0x419d0000, 0x4286db64, 0xc2c80000, 0xc2740000, 0xc1b00000,
        0x41880000, 0x42460000, 0x42b10000, 0x42ff0000, 0xc2b60000, 0xc0bba6c0, 0x419d0000,
        0x419d0000, 0xc640e400, 0xc640e400, 0x419d0000, 0x42715836, 0xc2500000, 0xc1500000,
        0x419c0000, 0x426a0000, 0x42c30000, 0x43020000, 0xc2a40000, 0xc22c0000, 0xc19ce810,
        0x419d0000, 0x419d0000, 0xc640e400, 0xc640e400, 0x419d0000, 0x4260079e, 0xc0800000,
        0x41e40000, 0x42870000, 0x42d50000, 0x430b0000, 0xc2920000, 0xc2080000, 0xbfc00000,
        0x42493f03, 0x419d0000, 0x419d0000, 0xc640e400, 0xc640e400, 0x419d0000, 0xc22a39aa,
        0x42160000, 0x42990000, 0x42da0000, 0x43140000, 0xc2800000, 0xc1c80000, 0x40f00000,
        0x423a0000, 0x429be874, 0x419d0000, 0x419d0000, 0xc640e400, 0xc640e400, 0x419d0000,
        0x4280a09f, 0x42ab0000, 0x42ec0000, 0xc2bc0000, 0xc25c0000, 0xc1b40000, 0x41840000,
        0x425e0000, 0x42b00000, 0x428a05c7, 0x419d0000, 0x419d0000, 0xc640e400, 0xc640e400,
        0x419d0000, 0x4127272d, 0x42fe0000, 0xc2aa0000, 0xc2380000, 0xc1580000, 0x41cc0000,
        0x42810000, 0x42c20000, 0x43080000, 0xc12cf890, 0x419d0000, 0x419d0000, 0xc640e400,
        0xc640e400, 0x419d0000, 0x42636b1d, 0xc2323ab4, 0x42320da0, 0xc1898be9, 0xc2087b50,
        0xc2072d2a, 0x41f3ebfa, 0x41ad0cfa, 0x4129d00d, 0x41df7f0b, 0x419d0000, 0x419d0000,
        0xc640e400, 0xc640e400, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000,
        0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000,
        0x419d0000, 0xc640e400, 0xc640e400, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000,
        0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000,
        0x419d0000, 0x419d0000, 0xc640e400, 0xc640e400,
    ];
    const GOLD_PAD_SECOND: [u32; 165] = [
        0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000,
        0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0xc640e400,
        0xc640e400, 0x419d0000, 0x42239ad2, 0x41b9087e, 0x4210c4cd, 0x4185a0a1, 0x40a162a4,
        0x4257d5f2, 0x426e2d51, 0x42426daf, 0xc0dfb1fc, 0x42769650, 0x419d0000, 0x419d0000,
        0xc640e400, 0xc640e400, 0x419d0000, 0xc1d6922e, 0xc2c80000, 0xc2740000, 0xc1b00000,
        0x41880000, 0x42460000, 0x42b10000, 0x42ff0000, 0xc2b60000, 0x426d09b1, 0x419d0000,
        0x419d0000, 0xc640e400, 0xc640e400, 0x419d0000, 0x429dce74, 0xc2500000, 0xc1500000,
        0x419c0000, 0x426a0000, 0x42c30000, 0x43020000, 0xc2a40000, 0xc22c0000, 0xc1f12e7c,
        0x419d0000, 0x419d0000, 0xc640e400, 0xc640e400, 0x419d0000, 0x4226e8ba, 0xc0800000,
        0x41e40000, 0x42870000, 0x42d50000, 0x430b0000, 0xc2920000, 0xc2080000, 0xbfc00000,
        0x42a237f8, 0x419d0000, 0x419d0000, 0xc640e400, 0xc640e400, 0x419d0000, 0x40864f85,
        0x42160000, 0x42990000, 0x42da0000, 0x43140000, 0xc2800000, 0xc1c80000, 0x40f00000,
        0x423a0000, 0x42718ec5, 0x419d0000, 0x419d0000, 0xc640e400, 0xc640e400, 0x419d0000,
        0xc2054ab9, 0x42ab0000, 0x42ec0000, 0xc2bc0000, 0xc25c0000, 0xc1b40000, 0x41840000,
        0x425e0000, 0x42b00000, 0x41366636, 0x419d0000, 0x419d0000, 0xc640e400, 0xc640e400,
        0x419d0000, 0x4256f046, 0x42fe0000, 0xc2aa0000, 0xc2380000, 0xc1580000, 0x41cc0000,
        0x42810000, 0x42c20000, 0x43080000, 0x429617c4, 0x419d0000, 0x419d0000, 0xc640e400,
        0xc640e400, 0x419d0000, 0x4270c9cd, 0x427ff6ac, 0x42a9759f, 0x420bb0cd, 0xc21be14a,
        0x429c0ded, 0x42420bd0, 0xc1ef76d0, 0xc1144354, 0x41b743a0, 0x419d0000, 0x419d0000,
        0xc640e400, 0xc640e400, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000,
        0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000,
        0x419d0000, 0xc640e400, 0xc640e400, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000,
        0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000, 0x419d0000,
        0x419d0000, 0x419d0000, 0xc640e400, 0xc640e400,
    ];
    const GOLD_BIG_PAD_FNV: u64 = 0x02d9ca3252a752a7;
    const GOLD_BIG_PAD_SPOT: [u32; 8] = [
        0x41e47dc1, 0xc2b80000, 0xc0200000, 0x42d70000, 0xc2580000, 0xc1500000, 0x42990000,
        0xc640e400,
    ];

    /// Deterministic input shared with the C golden generator.
    fn pad_src(i: i32) -> f32 {
        ((i * 37) % 251) as f32 - 100.0f32 + 0.5f32 * ((i * 17) % 13) as f32
    }

    /// `pseudoVals` is `static` in the source, so a test that pins exact
    /// values has to start from the declared state.
    fn reset_pseudo_vals() {
        unsafe {
            PSEUDO_VALS = [
                123456, 654321, 368341, 789234, 234561, 543216, 683413, 892347, 345612, 432165,
                834136, 923478, 456123, 321654, 341368, 234789,
            ];
        }
    }

    fn assert_pad_bits(got: &[f32], want: &[u32], what: &str) {
        let bad = (0..want.len())
            .filter(|i| got[*i].to_bits() != want[*i])
            .collect::<Vec<_>>();
        assert!(
            bad.is_empty(),
            "{what}: {} of {} values differ from the native sliceNoiseTaperPad, first at {:?} \
             (got {:#010x} = {}, want {:#010x} = {})",
            bad.len(),
            want.len(),
            bad.first(),
            got[bad[0]].to_bits(),
            got[bad[0]],
            want[bad[0]],
            f32::from_bits(want[bad[0]])
        );
    }

    /// Every value written by `sliceNoiseTaperPad` must match the reference
    /// `libcfshr` bit for bit.  The 8x6 -> 13x11 geometry is chosen so that
    /// `nx - ixhi > ixlo` and `ny - iyhi > iylo` both hold, which is what
    /// makes the source drop `nxtop`/`nytop` by one and fill the extra column
    /// and row with `dmean` (`taperpad.c:634-644`); indexing the sides off
    /// `nx - 1` / `ny - 1` instead misplaces every side and corner pixel.
    /// The same data pins `b3dIMin`'s leading argument being a **count**
    /// (`taperpad.c:590-592` asks for the min of 3 and of 4 values, not of 4
    /// and 5), `getRunningMeanSD` going through `sumsToAvgSDdbl` with its
    /// `n - 1` denominator, and the exact congruential sequence.
    ///
    /// The second call re-runs the identical arguments: it must produce
    /// *different* values, because `pseudoVals` is static and carries the
    /// generator state over between calls.
    #[test]
    fn noise_taper_pad_matches_the_native_routine_bit_for_bit() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        reset_pseudo_vals();
        let mut input = (0..8 * 6).map(pad_src).collect::<Vec<f32>>();
        let mut temp = vec![0.0f32; 4096];
        let mut first = vec![-12345.0f32; 15 * 11];
        let mut second = vec![-12345.0f32; 15 * 11];
        unsafe {
            slice_noise_taper_pad(
                input.as_mut_ptr().cast(),
                FLOAT,
                8,
                6,
                first.as_mut_ptr(),
                15,
                13,
                11,
                20,
                4,
                temp.as_mut_ptr(),
            );
            slice_noise_taper_pad(
                input.as_mut_ptr().cast(),
                FLOAT,
                8,
                6,
                second.as_mut_ptr(),
                15,
                13,
                11,
                20,
                4,
                temp.as_mut_ptr(),
            );
        }
        assert_pad_bits(&first, &GOLD_PAD_FIRST, "first call");
        assert_pad_bits(
            &second,
            &GOLD_PAD_SECOND,
            "second call (static pseudoVals carried over)",
        );
        assert_ne!(
            first, second,
            "static pseudoVals must carry over between calls"
        );
        // The extra column and row the source adds when the padding is
        // lopsided are plain `dmean`, and columns past `nx` are never written.
        for iy in 0..11 {
            assert_eq!(first[14 + iy * 15].to_bits(), (-12345.0f32).to_bits());
            assert_eq!(first[13 + iy * 15].to_bits(), (-12345.0f32).to_bits());
        }
    }

    /// `taperpad.c:611-616` insets the edge-mean rectangle by
    /// `edgeXadd`/`edgeYadd` once the box passes 400 pixels, which changes
    /// `dmean` and therefore every padded value.  401 x 403 is the smallest
    /// size that turns both insets on.
    #[test]
    fn noise_taper_pad_insets_the_edge_mean_for_boxes_over_400() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        reset_pseudo_vals();
        let (nxbox, nybox, nx, ny, nxdim) = (401, 403, 420, 424, 422);
        let mut input = (0..nxbox * nybox).map(pad_src).collect::<Vec<f32>>();
        let mut temp = vec![0.0f32; 1 << 20];
        let mut out = vec![-12345.0f32; (nxdim * ny) as usize];
        unsafe {
            slice_noise_taper_pad(
                input.as_mut_ptr().cast(),
                FLOAT,
                nxbox,
                nybox,
                out.as_mut_ptr(),
                nxdim,
                nx,
                ny,
                20,
                4,
                temp.as_mut_ptr(),
            );
        }
        let mut hash: u64 = 1469598103934665603;
        for value in &out {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(1099511628211);
            }
        }
        let spot = (0..8)
            .map(|i| out[(i * 37799) % (nxdim * ny) as usize].to_bits())
            .collect::<Vec<u32>>();
        assert_eq!(
            spot.as_slice(),
            GOLD_BIG_PAD_SPOT.as_slice(),
            "sampled values differ from the native sliceNoiseTaperPad"
        );
        assert_eq!(
            hash, GOLD_BIG_PAD_FNV,
            "FNV-1a over the whole padded array differs from the native sliceNoiseTaperPad"
        );
    }

    #[test]
    fn taper_and_split() {
        unsafe {
            let mut a = [1., 2., 3., 4.];
            let mut out = [0.; 36];
            slice_taper_out_pad(
                a.as_mut_ptr().cast(),
                FLOAT,
                2,
                2,
                out.as_mut_ptr(),
                6,
                6,
                6,
                0,
                0.,
            );
            assert_eq!(out[2 + 2 * 6], 1.);
            assert!((slice_edge_mean(out.as_mut_ptr(), 6, 2, 3, 2, 3) - 2.5).abs() < 1e-6);
            let mut s = [0.; 16];
            slice_split_fill(a.as_mut_ptr(), 2, 2, s.as_mut_ptr(), 4, 4, 4, 0, 0.);
            assert!((s.iter().sum::<f32>()).abs() < 1e-6);
        }
    }

    #[test]
    fn taper_in_preserves_the_extracted_pixels_and_noise_fills_all_corners() {
        unsafe {
            let mut input = [1., 2., 3., 4.];
            let mut tapered = [0.; 36];
            slice_taper_in_pad(
                input.as_mut_ptr().cast(),
                FLOAT,
                2,
                0,
                1,
                0,
                1,
                tapered.as_mut_ptr(),
                6,
                6,
                6,
                0,
                0,
            );
            assert!(tapered.iter().all(|value| value.is_finite()));
            assert!(tapered.iter().any(|value| *value != 0.));

            let mut image = [
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ];
            let mut noise_padded = [0.; 64];
            let mut temp = [0.; 64];
            slice_noise_taper_pad(
                image.as_mut_ptr().cast(),
                FLOAT,
                4,
                4,
                noise_padded.as_mut_ptr(),
                8,
                8,
                8,
                4,
                2,
                temp.as_mut_ptr(),
            );
            for value in noise_padded {
                assert!(value.is_finite());
            }
        }
    }
}
