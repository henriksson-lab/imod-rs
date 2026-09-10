//! Translation of `IMOD/libcfshr/taperpad.c`.
#![allow(dead_code)]

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

/// C static `getRunningMeanSD`.
unsafe fn get_running_mean_sd(
    box_start: *mut f32,
    rows: i32,
    len: i32,
    rowstride: i32,
    lenstride: i32,
    nbox: i32,
    mean: f32,
    samples: *mut f32,
    squares: *mut f32,
    num: i32,
    means: *mut f32,
    sds: *mut f32,
) {
    unsafe {
        let mut sum = 0_f64;
        let mut sq = 0_f64;
        let mut si = 0;
        for l in 0..len {
            for r in 0..rows {
                let v = *box_start.add((l * lenstride + r * rowstride) as usize) - mean;
                *samples.add(si as usize) = v;
                *squares.add(si as usize) = v * v;
                sum += v as f64;
                sq += (v * v) as f64;
                si += 1;
            }
        }
        let base = len / 2;
        let mut av = sum / num as f64;
        *means.add(base as usize) = mean + av as f32;
        *sds.add(base as usize) = ((sq / num as f64 - av * av).max(0.)).sqrt() as f32;
        let mut at = base + 1;
        si = 0;
        for l in len..nbox - len {
            for r in 0..rows {
                let v = *samples.add(si as usize);
                sum -= v as f64;
                sq -= *squares.add(si as usize) as f64;
                let nv = *box_start.add((l * lenstride + r * rowstride) as usize) - mean;
                *samples.add(si as usize) = nv;
                *squares.add(si as usize) = nv * nv;
                sum += nv as f64;
                sq += (nv * nv) as f64;
                si = (si + 1) % num;
            }
            av = sum / num as f64;
            *means.add(at as usize) = mean + av as f32;
            *sds.add(at as usize) = ((sq / num as f64 - av * av).max(0.)).sqrt() as f32;
            at += 1;
        }
        for i in 0..base {
            *means.add(i as usize) = *means.add(base as usize);
            *sds.add(i as usize) = *sds.add(base as usize)
        }
        for i in at..nbox {
            *means.add(i as usize) = *means.add((at - 1) as usize);
            *sds.add(i as usize) = *sds.add((at - 1) as usize)
        }
    }
}

/// C `sliceNoiseTaperPad` (the deterministic sequential equivalent of the C OpenMP loops).
pub unsafe fn slice_noise_taper_pad(
    array: *mut c_void,
    typ: i32,
    nxbox: i32,
    nybox: i32,
    out: *mut f32,
    nxdim: i32,
    nx: i32,
    ny: i32,
    noise_length: i32,
    noise_rows: i32,
    temp: *mut f32,
) {
    unsafe {
        let mut rows = noise_rows.min(3).min(nxbox / 2).min(nybox / 2);
        if rows < 1 {
            rows = 1
        };
        let len = 2 * (noise_length.min(600 / rows).min(nxbox).min(nybox) / 2);
        if len < 2 {
            slice_taper_out_pad(array, typ, nxbox, nybox, out, nxdim, nx, ny, 0, 0.);
            return;
        }
        let (mut xl, mut xh, mut yl, mut yh) = (0, 0, 0, 0);
        copy_to_center(
            array, typ, nxbox, nybox, out, nxdim, nx, ny, &mut xl, &mut xh, &mut yl, &mut yh,
        );
        if nxbox == nx && nybox == ny {
            return;
        }
        let mean = slice_edge_mean(out, nxdim, xl, xh - 1, yl, yh - 1) as f32;
        let mut fracx = vec![0.; xl as usize];
        let mut fracy = vec![0.; yl as usize];
        let mut xpart = vec![0.; xl as usize];
        let mut ypart = vec![0.; yl as usize];
        for i in 0..xl {
            fracx[i as usize] = i as f32 / xl as f32;
            xpart[i as usize] = (1. - fracx[i as usize]) * mean;
        }
        for i in 0..yl {
            fracy[i as usize] = i as f32 / yl as f32;
            ypart[i as usize] = (1. - fracy[i as usize]) * mean;
        }
        let mut samples = vec![0.; (rows * len) as usize];
        let mut squares = vec![0.; (rows * len) as usize];
        let mut means = vec![0.; nxbox.max(nybox) as usize];
        let mut sds = means.clone();
        let mut pseudo = 123456_i32;
        get_running_mean_sd(
            out.add((xl + yl * nxdim) as usize),
            rows,
            len,
            nxdim,
            1,
            nxbox,
            mean,
            samples.as_mut_ptr(),
            squares.as_mut_ptr(),
            rows * len,
            means.as_mut_ptr(),
            sds.as_mut_ptr(),
        );
        for iy in 0..yl {
            for ix in 0..nxbox {
                pseudo = (197 * (pseudo + 1)) & 0xfffff;
                *out.add((xl + ix + iy * nxdim) as usize) = (means[ix as usize]
                    + (pseudo as f32 * 1.73 / 0x7ffff as f32 - 1.73) * sds[ix as usize])
                    * fracy[iy as usize]
                    + ypart[iy as usize];
            }
        }
        let save_corn_mean = means[0];
        let save_corn_sd = sds[0];
        let mut corn_x_mean = means[(nxbox - 1) as usize];
        let mut corn_x_sd = sds[(nxbox - 1) as usize];
        get_running_mean_sd(
            out.add((xl + nxbox - rows + yl * nxdim) as usize),
            rows,
            len,
            1,
            nxdim,
            nybox,
            mean,
            samples.as_mut_ptr(),
            squares.as_mut_ptr(),
            rows * len,
            means.as_mut_ptr(),
            sds.as_mut_ptr(),
        );
        let mut corn_y_mean = means[0];
        let mut corn_y_sd = sds[0];
        for iy in 0..yl {
            for ix in 0..xl {
                let frac_min = fracx[ix as usize].min(fracy[iy as usize]);
                let weight_sum = 0.01_f32.max(fracx[ix as usize] + fracy[iy as usize]);
                let corn_mean = (corn_x_mean * fracx[ix as usize]
                    + corn_y_mean * fracy[iy as usize])
                    / weight_sum;
                let corn_sd =
                    (corn_x_sd * fracx[ix as usize] + corn_y_sd * fracy[iy as usize]) / weight_sum;
                pseudo = (197 * (pseudo + 1)) & 0xfffff;
                *out.add((nx - 1 - ix + iy * nxdim) as usize) = (corn_mean
                    + (pseudo as f32 * 1.73 / 0x7ffff as f32 - 1.73) * corn_sd)
                    * frac_min
                    + mean * (1. - frac_min);
            }
        }
        for iy in 0..nybox {
            for ix in 0..xl {
                pseudo = (197 * (pseudo + 1)) & 0xfffff;
                *out.add((nx - 1 - ix + (iy + yl) * nxdim) as usize) = (means[iy as usize]
                    + (pseudo as f32 * 1.73 / 0x7ffff as f32 - 1.73) * sds[iy as usize])
                    * fracx[ix as usize]
                    + xpart[ix as usize];
            }
        }
        corn_y_mean = means[(nybox - 1) as usize];
        corn_y_sd = sds[(nybox - 1) as usize];
        get_running_mean_sd(
            out.add((xl + (yl + nybox - rows) * nxdim) as usize),
            rows,
            len,
            nxdim,
            1,
            nxbox,
            mean,
            samples.as_mut_ptr(),
            squares.as_mut_ptr(),
            rows * len,
            means.as_mut_ptr(),
            sds.as_mut_ptr(),
        );
        corn_x_mean = means[(nxbox - 1) as usize];
        corn_x_sd = sds[(nxbox - 1) as usize];
        for iy in 0..yl {
            for ix in 0..xl {
                let frac_min = fracx[ix as usize].min(fracy[iy as usize]);
                let weight_sum = 0.01_f32.max(fracx[ix as usize] + fracy[iy as usize]);
                let corn_mean = (corn_x_mean * fracx[ix as usize]
                    + corn_y_mean * fracy[iy as usize])
                    / weight_sum;
                let corn_sd =
                    (corn_x_sd * fracx[ix as usize] + corn_y_sd * fracy[iy as usize]) / weight_sum;
                pseudo = (197 * (pseudo + 1)) & 0xfffff;
                *out.add((nx - 1 - ix + (ny - 1 - iy) * nxdim) as usize) = (corn_mean
                    + (pseudo as f32 * 1.73 / 0x7ffff as f32 - 1.73) * corn_sd)
                    * frac_min
                    + mean * (1. - frac_min);
            }
        }
        for iy in 0..yl {
            for ix in 0..nxbox {
                pseudo = (197 * (pseudo + 1)) & 0xfffff;
                *out.add((xl + ix + (ny - 1 - iy) * nxdim) as usize) = (means[ix as usize]
                    + (pseudo as f32 * 1.73 / 0x7ffff as f32 - 1.73) * sds[ix as usize])
                    * fracy[iy as usize]
                    + ypart[iy as usize];
            }
        }
        corn_x_mean = means[0];
        corn_x_sd = sds[0];
        get_running_mean_sd(
            out.add((xl + yl * nxdim) as usize),
            rows,
            len,
            1,
            nxdim,
            nybox,
            mean,
            samples.as_mut_ptr(),
            squares.as_mut_ptr(),
            rows * len,
            means.as_mut_ptr(),
            sds.as_mut_ptr(),
        );
        corn_y_mean = means[(nybox - 1) as usize];
        corn_y_sd = sds[(nybox - 1) as usize];
        for iy in 0..yl {
            for ix in 0..xl {
                let frac_min = fracx[ix as usize].min(fracy[iy as usize]);
                let weight_sum = 0.01_f32.max(fracx[ix as usize] + fracy[iy as usize]);
                let corn_mean = (corn_x_mean * fracx[ix as usize]
                    + corn_y_mean * fracy[iy as usize])
                    / weight_sum;
                let corn_sd =
                    (corn_x_sd * fracx[ix as usize] + corn_y_sd * fracy[iy as usize]) / weight_sum;
                pseudo = (197 * (pseudo + 1)) & 0xfffff;
                *out.add((ix + (ny - 1 - iy) * nxdim) as usize) = (corn_mean
                    + (pseudo as f32 * 1.73 / 0x7ffff as f32 - 1.73) * corn_sd)
                    * frac_min
                    + mean * (1. - frac_min);
            }
        }
        for iy in 0..nybox {
            for ix in 0..xl {
                pseudo = (197 * (pseudo + 1)) & 0xfffff;
                *out.add((ix + (iy + yl) * nxdim) as usize) = (means[iy as usize]
                    + (pseudo as f32 * 1.73 / 0x7ffff as f32 - 1.73) * sds[iy as usize])
                    * fracx[ix as usize]
                    + xpart[ix as usize];
            }
        }
        corn_y_mean = means[0];
        corn_y_sd = sds[0];
        corn_x_mean = save_corn_mean;
        corn_x_sd = save_corn_sd;
        for iy in 0..yl {
            for ix in 0..xl {
                let frac_min = fracx[ix as usize].min(fracy[iy as usize]);
                let weight_sum = 0.01_f32.max(fracx[ix as usize] + fracy[iy as usize]);
                let corn_mean = (corn_x_mean * fracx[ix as usize]
                    + corn_y_mean * fracy[iy as usize])
                    / weight_sum;
                let corn_sd =
                    (corn_x_sd * fracx[ix as usize] + corn_y_sd * fracy[iy as usize]) / weight_sum;
                pseudo = (197 * (pseudo + 1)) & 0xfffff;
                *out.add((ix + iy * nxdim) as usize) = (corn_mean
                    + (pseudo as f32 * 1.73 / 0x7ffff as f32 - 1.73) * corn_sd)
                    * frac_min
                    + mean * (1. - frac_min);
            }
        }
        let _ = temp;
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
pub unsafe fn slice_edge_median(
    a: *mut f32,
    n: i32,
    xl: i32,
    xh: i32,
    yl: i32,
    yh: i32,
    mean_sides: i32,
) -> f64 {
    unsafe {
        let interval = ((2 * ((xh - xl) + (yh + 1 - yl)) + 9999) / 10000).max(1);
        let mut all: Vec<f32> = Vec::new();
        let mut med = Vec::new();
        for side in 0..4 {
            let mut q = Vec::new();
            if side < 2 {
                let y = if side == 0 { yl } else { yh };
                for x in (xl..=xh).step_by(interval as usize) {
                    q.push(*a.add((x + y * n) as usize));
                }
            } else {
                let x = if side == 2 { xl } else { xh };
                for y in (yl + 1..yh).step_by(interval as usize) {
                    q.push(*a.add((x + y * n) as usize));
                }
            }
            all.extend(q.iter());
            q.sort_by(|x, y| x.total_cmp(y));
            med.push(q[q.len() / 2] as f64);
        }
        if mean_sides != 0 {
            med.iter().sum::<f64>() / 4.
        } else {
            all.sort_by(|x, y| x.total_cmp(y));
            all[all.len() / 2] as f64
        }
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
