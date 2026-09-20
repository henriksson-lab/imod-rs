//! Translation of `IMOD/libiimod/mrcslice.c` and `include/mrcslice.h`.
#![allow(unused_variables)]

use crate::imod::libcfshr::islice::{
    Islice, Istack, MrcData, slice_create, slice_get_val, slice_init, slice_put_val,
};
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT,
    MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader, mrc_head_new, mrc_head_write,
    mrc_mread_slice, mrc_write_slice,
};

/// `sliceReadMRC` from `mrcslice.c:945`.
///
/// Returns a slice holding one plane of `hin` at coordinate `sno` along
/// `axis`, using the file pointer in `hin`.  Calls `mrc_mread_slice`, which
/// swaps bytes if needed.
pub fn slice_read_mrc(hin: &mut MrcHeader, sno: i32, axis: u8) -> Option<Islice> {
    // A *clone* of the handle, not a borrow of `hin.fp`: `mrc_read_slice`
    // (`mrcfiles.c:1049`) saves and restores `hdata->fp` around its call,
    // so a `&mut` into that very field would be left aliasing a `None`
    // the moment the callee takes it.  The C copies the `FILE *` by value
    // and a clone shares the same open file, which is the same thing.
    let mut fin = hin.fp.clone()?;
    let Some(buf) = mrc_mread_slice(&mut fin, hin, sno, axis) else {
        return None;
    };
    // `mrcslice.c:960-978`: the switch has no assignment in its `default`
    // arm, so an unrecognised axis leaves nx and ny indeterminate in C.
    // Rust cannot reproduce an indeterminate local; zero is used and the
    // deviation is confined to an axis the callers never pass.
    let (nx, ny) = match axis as u8 {
        b'x' | b'X' => (hin.nx, hin.nz),
        b'y' | b'Y' => (hin.ny, hin.nz),
        b'z' | b'Z' => (hin.nx, hin.ny),
        _ => (0, 0),
    };
    // `mrcslice.c:951-988`: the Islice comes from `malloc` and only `mean`
    // is set before `sliceInit`; the read buffer becomes the slice's data.
    let mut slice = Islice {
        data: MrcData::default(),
        xsize: 0,
        ysize: 0,
        mode: 0,
        csize: 0,
        dsize: 0,
        min: 0.,
        max: 0.,
        mean: hin.amean,
        index: 0,
        cval: [0.; 4],
    };
    if slice_init(&mut slice, nx, ny, hin.mode, buf) != 0 {
        return None;
    }
    Some(slice)
}

/// `sliceReadSubm` from mrcslice.c:996.
///
/// This preserves the source's direct-section path for contained Y/Z areas and
/// its full-plane/read-then-box path for X sections and out-of-bounds areas.
pub fn slice_read_subm(
    hin: &mut MrcHeader,
    sec_num: i32,
    axis: u8,
    xsize: i32,
    ysize: i32,
    xcen: i32,
    ycen: i32,
) -> Option<Islice> {
    let (nx, ny) = match axis as u8 {
        b'x' | b'X' => (hin.ny, hin.nz),
        b'y' | b'Y' => (hin.nx, hin.nz),
        b'z' | b'Z' => (hin.nx, hin.ny),
        _ => return None,
    };
    let llx = xcen - xsize / 2;
    let lly = ycen - ysize / 2;
    let urx = llx + xsize;
    let ury = lly + ysize;
    if matches!(axis as u8, b'z' | b'Z' | b'y' | b'Y')
        && llx >= 0
        && lly >= 0
        && urx <= nx
        && ury <= ny
    {
        let mut li = LoadInfo::default();
        crate::imod::libiimod::mrcfiles::mrc_init_li(Some(&mut li), None);
        crate::imod::libiimod::mrcfiles::mrc_init_li(Some(&mut li), Some(hin));
        li.xmin = llx;
        li.xmax = urx - 1;
        if matches!(axis as u8, b'y' | b'Y') {
            li.axis = 2;
            li.zmin = lly;
            li.zmax = ury - 1;
        } else {
            li.ymin = lly;
            li.ymax = ury - 1;
        }
        let mut slice = slice_create(urx - llx, ury - lly, hin.mode)?;
        slice.mean = hin.amean;
        if crate::imod::libiimod::mrcsec::mrc_read_section(
            hin,
            &mut li,
            slice.data.bytes_mut(),
            sec_num,
        ) == 0
        {
            return Some(slice);
        }
        return None;
    }
    let Some(buffer) =
        crate::imod::libiimod::mrcfiles::mrc_mread_slice(&mut hin.fp.clone()?, hin, sec_num, axis)
    else {
        return None;
    };
    // `mrcslice.c:1046-1076`: as in `sliceReadMRC`, the read buffer becomes
    // the slice's data through `sliceInit`.
    let mut slice = Islice {
        data: MrcData::default(),
        xsize: 0,
        ysize: 0,
        mode: 0,
        csize: 0,
        dsize: 0,
        min: 0.,
        max: 0.,
        mean: hin.amean,
        index: 0,
        cval: [0.; 4],
    };
    if slice_init(&mut slice, nx, ny, hin.mode, buffer) != 0 {
        return None;
    }
    if slice_box_in(&mut slice, llx, lly, urx, ury) != 0 {
        return None;
    }
    Some(slice)
}

/// `sliceNewMode` from mrcslice.c:31.
pub fn slice_new_mode(s: &mut Islice, mode: i32) -> i32 {
    slice_new_mode_ex(s, mode, 1)
}

/// `sliceNewModeEx` from mrcslice.c:40.
pub fn slice_new_mode_ex(s: &mut Islice, mode: i32, free_data: i32) -> i32 {
    if s.mode == mode {
        return mode;
    }
    if mode == MRC_MODE_FLOAT {
        return if slice_float_ex(s, free_data) != 0 {
            -1
        } else {
            mode
        };
    }
    let Some(mut ns) = slice_create(s.xsize, s.ysize, mode) else {
        return -1;
    };
    let (limit, lo, hi) = match mode {
        MRC_MODE_BYTE | MRC_MODE_RGB => (true, 0., 255.),
        MRC_MODE_SHORT => (true, -32768., 32767.),
        MRC_MODE_USHORT => (true, 0., 65535.),
        _ => (false, 0., 0.),
    };
    for j in 0..s.ysize {
        for i in 0..s.xsize {
            let mut val = [0.; 4];
            slice_get_val(s, i, j, &mut val);
            if s.mode == MRC_MODE_COMPLEX_FLOAT || s.mode == MRC_MODE_COMPLEX_SHORT {
                if mode != MRC_MODE_COMPLEX_FLOAT && mode != MRC_MODE_COMPLEX_SHORT {
                    val[0] = (val[0] * val[0] + val[1] * val[1]).sqrt();
                }
            } else if s.mode == MRC_MODE_RGB {
                val[0] = val[0] * 0.3 + val[1] * 0.59 + val[2] * 0.11;
                if mode == MRC_MODE_COMPLEX_FLOAT || mode == MRC_MODE_COMPLEX_SHORT {
                    val[1] = 0.;
                }
            } else if mode == MRC_MODE_COMPLEX_FLOAT || mode == MRC_MODE_COMPLEX_SHORT {
                val[1] = 0.;
            }
            if mode == MRC_MODE_RGB {
                if limit {
                    val[0] = val[0].clamp(lo, hi);
                }
                val[1] = val[0];
                val[2] = val[0];
            }
            if limit {
                for q in 0..s.csize.min(3) as usize {
                    val[q] = val[q].clamp(lo, hi);
                }
            }
            slice_put_val(ns.as_mut(), i, j, val);
        }
    }
    // Storage is always owned by Islice now; replacing it drops the old
    // Vec regardless of the historical C freeData flag.
    s.data = ns.data;
    s.mode = ns.mode;
    s.csize = ns.csize;
    s.dsize = ns.dsize;
    mode
}

/// `sliceFloat` from mrcslice.c:226.
pub fn slice_float(slice: &mut Islice) -> i32 {
    slice_float_ex(slice, 1)
}

/// `sliceFloatEx` from mrcslice.c:235.
pub fn slice_float_ex(slice: &mut Islice, free_data: i32) -> i32 {
    if slice.mode == MRC_MODE_FLOAT {
        return 0;
    }
    let Some(mut tsl) = slice_create(slice.xsize, slice.ysize, MRC_MODE_FLOAT) else {
        return -1;
    };
    for j in 0..slice.ysize {
        for i in 0..slice.xsize {
            let mut val = [0.; 4];
            slice_get_val(slice, i, j, &mut val);
            if slice.mode == MRC_MODE_COMPLEX_SHORT || slice.mode == MRC_MODE_COMPLEX_FLOAT {
                val[0] = (val[0] * val[0] + val[1] * val[1]).sqrt();
            } else if slice.mode == MRC_MODE_RGB {
                val[0] = val[0] * 0.3 + val[1] * 0.59 + val[2] * 0.11;
            }
            slice_put_val(tsl.as_mut(), i, j, val);
        }
    }
    slice.data = tsl.data;
    slice.mode = tsl.mode;
    slice.csize = tsl.csize;
    slice.dsize = tsl.dsize;
    0
}

/// `sliceComplexFloat` from mrcslice.c:307.
pub fn slice_complex_float(slice: &mut Islice) -> i32 {
    if slice.mode > MRC_MODE_COMPLEX_SHORT && slice.mode != MRC_MODE_USHORT {
        return -1;
    }
    let Some(mut tsl) = slice_create(slice.xsize, slice.ysize, MRC_MODE_COMPLEX_FLOAT) else {
        return -1;
    };
    for j in 0..slice.ysize {
        for i in 0..slice.xsize {
            let mut v = [0.; 4];
            slice_get_val(slice, i, j, &mut v);
            v[1] = 0.;
            slice_put_val(tsl.as_mut(), i, j, v);
        }
    }
    slice.data = tsl.data;
    slice.mode = tsl.mode;
    slice.csize = tsl.csize;
    slice.dsize = tsl.dsize;
    0
}

/// `sliceMMM` from mrcslice.c:336.
pub fn slice_mmm(s: &mut Islice) -> i32 {
    let mut val = [0.; 4];
    slice_get_val(s, 0, 0, &mut val);
    if s.mode == MRC_MODE_COMPLEX_FLOAT {
        val[0] = (val[0] * val[0] + val[1] * val[1]).sqrt();
    }
    s.min = val[0];
    s.max = val[0];
    if s.xsize == 0 || s.ysize == 0 {
        crate::imod::libcfshr::b3dutil::b3d_error(
            Some(&mut crate::imod::libcfshr::b3dutil::ImodFile::Stderr),
            format_args!("sliceMMM: Warning, empty slice.\n"),
        );
        return -1;
    }
    let mut sum = 0_f64;
    for j in 0..s.ysize {
        let mut tsum = 0_f64;
        for i in 0..s.xsize {
            slice_get_val(s, i, j, &mut val);
            if s.mode == MRC_MODE_COMPLEX_FLOAT {
                val[0] = (val[0] * val[0] + val[1] * val[1]).sqrt();
            }
            // `mrcslice.c:369-372` compares with `>` / `<`, which are false
            // for a NaN operand: once `s.min` is NaN (pixel 0,0) it stays NaN.
            // `f32::min`/`max` would silently recover the non-NaN operand.
            if s.min > val[0] {
                s.min = val[0];
            }
            if s.max < val[0] {
                s.max = val[0];
            }
            tsum += val[0] as f64;
        }
        sum += tsum;
    }
    // `mrcslice.c:381`: `sum / (float)(s->xsize * s->ysize)` -- the divisor
    // is rounded through float before the double division.
    s.mean = (sum / f64::from((s.xsize * s.ysize) as f32)) as f32;
    0
}

/// `fullArrayMinMaxMean` from mrcslice.c:385.  The C wraps the caller's
/// array in a stack `Islice` through `sliceInit`; here the storage is lent
/// to a temporary slice and handed back, so nothing is copied.
pub fn full_array_min_max_mean(
    array: &mut MrcData,
    typ: i32,
    nx: i32,
    ny: i32,
) -> Option<(f32, f32, f32)> {
    let mut sl = Islice {
        data: MrcData::default(),
        xsize: 0,
        ysize: 0,
        mode: 0,
        csize: 0,
        dsize: 0,
        min: 0.,
        max: 0.,
        mean: 0.,
        index: 0,
        cval: [0.; 4],
    };
    let init = slice_init(&mut sl, nx, ny, typ, std::mem::take(array));
    let result = if init == 0 {
        slice_mmm(&mut sl);
        Some((sl.min, sl.max, sl.mean))
    } else {
        None
    };
    *array = std::mem::take(&mut sl.data);
    result
}

/// `corr_conj` from mrcslice.c:482.  The C takes two pointers, and every
/// autocorrelation caller passes the same array twice; `None` for `h` is
/// that aliased call, reading `g`'s own pair before the pair is written,
/// exactly as the C's load order does.
pub fn corr_conj(g: &mut [f32], h: Option<&[f32]>, size: i32) -> i32 {
    for i in 0..size as usize {
        let real = i * 2;
        let imag = real + 1;
        let temp = g[real];
        let (rtmp, itmp) = match h {
            Some(h) => (h[real], h[imag]),
            None => (g[real], g[imag]),
        };
        g[real] = rtmp * temp + itmp * g[imag];
        g[imag] = itmp * temp - rtmp * g[imag];
    }
    0
}
pub fn slice_add_const(slice: &mut Islice, c: [f32; 4]) -> i32 {
    for j in 0..slice.ysize {
        for i in 0..slice.xsize {
            let mut v = [0.; 4];
            slice_get_val(slice, i, j, &mut v);
            for q in 0..slice.csize.min(3) as usize {
                v[q] += c[q];
            }
            slice_put_val(slice, i, j, v);
        }
    }
    0
}
/// `sliceMultConst` (`mrcslice.c:553`).
///
/// The source is a `switch` on `csize` in which **`case 2` has no `break`**
/// (`mrcslice.c:575`), so a two-channel slice runs the two-channel loop and
/// then falls through and runs the three-channel loop as well — its first two
/// channels are scaled twice, by `c[i]` squared.  A generic loop over `csize`
/// scales them once, which diverges for complex data.
///
/// Structurally faithful but **execution-unverified**: no command-line input
/// has been found that reaches this function with `csize == 2`.  A complex
/// float MRC authored by the reference `clip fft`, run through
/// `clip average -l 2`, produces identical output whether or not the
/// fall-through is present, so that case does not exercise it.
pub fn slice_mult_const(slice: &mut Islice, c: [f32; 4]) -> i32 {
    match slice.csize {
        1 => {
            for j in 0..slice.ysize {
                for i in 0..slice.xsize {
                    let mut v = [0.; 4];
                    slice_get_val(slice, i, j, &mut v);
                    v[0] *= c[0];
                    slice_put_val(slice, i, j, v);
                }
            }
        }
        2 => {
            for j in 0..slice.ysize {
                for i in 0..slice.xsize {
                    let mut v = [0.; 4];
                    slice_get_val(slice, i, j, &mut v);
                    v[0] *= c[0];
                    v[1] *= c[1];
                    slice_put_val(slice, i, j, v);
                }
            }
            // No `break` at `mrcslice.c:575`: control falls into case 3.
            for j in 0..slice.ysize {
                for i in 0..slice.xsize {
                    let mut v = [0.; 4];
                    slice_get_val(slice, i, j, &mut v);
                    v[0] *= c[0];
                    v[1] *= c[1];
                    v[2] *= c[2];
                    slice_put_val(slice, i, j, v);
                }
            }
        }
        3 => {
            for j in 0..slice.ysize {
                for i in 0..slice.xsize {
                    let mut v = [0.; 4];
                    slice_get_val(slice, i, j, &mut v);
                    v[0] *= c[0];
                    v[1] *= c[1];
                    v[2] *= c[2];
                    slice_put_val(slice, i, j, v);
                }
            }
        }
        _ => {}
    }
    0
}
pub fn mrc_slice_valscale(s: &mut Islice, in_scale: f64) -> i32 {
    slice_mult_const(s, [in_scale as f32; 4])
}

pub fn mrc_slice_lie(sin: &mut Islice, fixed: f64, alpha: f64) -> i32 {
    let scale = alpha as f32;
    let offset = (1. - scale) * fixed as f32;
    let (min, max) = match sin.mode {
        MRC_MODE_BYTE | MRC_MODE_RGB => (0., 255.),
        MRC_MODE_SHORT | MRC_MODE_COMPLEX_SHORT => (-32768., 32767.),
        MRC_MODE_USHORT => (0., 65535.),
        _ => (0., 0.),
    };
    // `mrcslice.c:631-658` declares `Ival val` once (`:609`) and splits the
    // pixel loop on `csize == 1`, hoisting the channel test out of the body.
    let mut v = [0.; 4];
    if sin.csize == 1 {
        for j in 0..sin.ysize {
            for i in 0..sin.xsize {
                slice_get_val(sin, i, j, &mut v);
                v[0] = offset + scale * v[0];
                if max != 0. {
                    if v[0] > max {
                        v[0] = max;
                    }
                    if v[0] < min {
                        v[0] = min;
                    }
                }
                slice_put_val(sin, i, j, v);
            }
        }
    } else {
        for j in 0..sin.ysize {
            for i in 0..sin.xsize {
                slice_get_val(sin, i, j, &mut v);
                for q in 0..sin.csize.min(3) as usize {
                    v[q] = offset + scale * v[q];
                    if max != 0. {
                        if v[q] > max {
                            v[q] = max;
                        }
                        if v[q] < min {
                            v[q] = min;
                        }
                    }
                }
                slice_put_val(sin, i, j, v);
            }
        }
    }
    0
}

pub fn slice_box(sl: &mut Islice, llx: i32, lly: i32, urx: i32, ury: i32) -> Option<Islice> {
    let nx = urx - llx;
    let ny = ury - lly;
    let mut sout = slice_create(nx, ny, sl.mode)?;
    /* Do simple line copies if there is no filling of edges */
    let (mut csize, mut dsize) = (0, 0);
    if llx >= 0
        && lly >= 0
        && urx <= sl.xsize
        && ury <= sl.ysize
        && crate::imod::libiimod::mrcfiles::mrc_getdcsize(sl.mode, &mut csize, &mut dsize) == 0
    {
        // `mrcslice.c:688-694`: `memcpy` of `pixSize * nx` bytes per line.
        let pix_size = (csize * dsize) as usize;
        let (input, output) = (sl.data.bytes(), sout.data.bytes_mut());
        for y in 0..ny as usize {
            let j = lly as usize + y;
            let in_ptr = (j * sl.xsize as usize + llx as usize) * pix_size;
            let out_ptr = y * nx as usize * pix_size;
            output[out_ptr..out_ptr + pix_size * nx as usize]
                .copy_from_slice(&input[in_ptr..in_ptr + pix_size * nx as usize]);
        }
        return Some(sout);
    }
    /* Or use gets and puts which is a lot slower */
    for (y, j) in (lly..ury).enumerate() {
        for (x, i) in (llx..urx).enumerate() {
            let mut v = [0.; 4];
            slice_get_val(sl, i, j, &mut v);
            slice_put_val(sout.as_mut(), x as i32, y as i32, v);
        }
    }
    Some(sout)
}
pub fn slice_box_in(sl: &mut Islice, llx: i32, lly: i32, urx: i32, ury: i32) -> i32 {
    if llx == 0 && lly == 0 && urx == sl.xsize && ury == sl.ysize {
        return 0;
    }
    let Some(sout) = slice_box(sl, llx, lly, urx, ury) else {
        return -1;
    };
    let replacement = sout;
    sl.data = replacement.data;
    sl.xsize = replacement.xsize;
    sl.ysize = replacement.ysize;
    sl.mode = replacement.mode;
    sl.csize = replacement.csize;
    sl.dsize = replacement.dsize;
    sl.min = replacement.min;
    sl.max = replacement.max;
    sl.mean = replacement.mean;
    sl.index = replacement.index;
    sl.cval = replacement.cval;
    0
}
pub fn slice_resize_in(sl: &mut Islice, x: i32, y: i32) -> i32 {
    if sl.xsize == x && sl.ysize == y {
        return 0;
    }
    let llx = sl.xsize / 2 - x / 2;
    let lly = sl.ysize / 2 - y / 2;
    slice_box_in(sl, llx, lly, llx + x, lly + y)
}
pub fn mrc_slice_resize(slin: &mut Islice, nx: i32, ny: i32) -> Option<Islice> {
    let mut sout = slice_create(nx, ny, slin.mode)?;
    let sx = (slin.xsize - nx) / 2;
    let sy = (slin.ysize - ny) / 2;
    let p = [slin.mean; 4];
    for j in 0..ny {
        for i in 0..nx {
            let x = i + sx;
            let y = j + sy;
            let mut v = p;
            if x >= 0 && y >= 0 && x < slin.xsize && y < slin.ysize {
                slice_get_val(slin, x, y, &mut v);
            }
            slice_put_val(sout.as_mut(), i, j, v);
        }
    }
    Some(sout)
}
pub fn slice_mirror(s: &mut Islice, axis: u8) -> i32 {
    match axis as char {
        'x' | 'X' => {
            for j in 0..s.ysize / 2 {
                for i in 0..s.xsize {
                    let mut a = [0.; 4];
                    let mut b = [0.; 4];
                    slice_get_val(s, i, j, &mut a);
                    slice_get_val(s, i, s.ysize - j - 1, &mut b);
                    slice_put_val(s, i, j, b);
                    slice_put_val(s, i, s.ysize - j - 1, a);
                }
            }
            0
        }
        'y' | 'Y' => {
            for i in 0..s.xsize / 2 {
                for j in 0..s.ysize {
                    let mut a = [0.; 4];
                    let mut b = [0.; 4];
                    slice_get_val(s, i, j, &mut a);
                    slice_get_val(s, s.xsize - i - 1, j, &mut b);
                    slice_put_val(s, i, j, b);
                    slice_put_val(s, s.xsize - i - 1, j, a);
                }
            }
            0
        }
        _ => -1,
    }
}

/// `sliceWrapFFTLines` from mrcslice.c:833.
pub fn slice_wrap_fft_lines(s: &mut Islice, direction: i32) -> i32 {
    let nx = s.xsize;
    let ny = s.ysize;
    if nx < 0 || ny < 0 {
        return 1;
    }
    let pixsize = if s.mode == MRC_MODE_COMPLEX_FLOAT {
        8
    } else if s.mode == MRC_MODE_COMPLEX_SHORT {
        if ny & 1 != 0 {
            return 3;
        }
        4
    } else {
        return 1;
    };
    if pixsize == 8 {
        let mut buffer = Vec::<f32>::new();
        if buffer.try_reserve_exact((2 * nx) as usize).is_err() {
            return 2;
        }
        buffer.resize((2 * nx) as usize, 0.);
        crate::imod::libcfshr::filtxcorr::wrap_fft_slice(
            s.data.f_mut(),
            &mut buffer,
            nx,
            ny,
            direction,
        );
        return 0;
    }
    /* Swap lines in complex short slice with old code for even ny */
    let line = (2 * nx) as usize;
    let mut buffer = Vec::<i16>::new();
    if buffer.try_reserve_exact(line).is_err() {
        return 2;
    }
    buffer.resize(line, 0);
    let d = s.data.s_mut();
    for i in 0..ny / 2 {
        let ind1 = i as usize * line;
        let ind2 = (i + ny / 2) as usize * line;
        buffer.copy_from_slice(&d[ind1..ind1 + line]);
        d.copy_within(ind2..ind2 + line, ind1);
        d[ind2..ind2 + line].copy_from_slice(&buffer);
    }
    0
}

/// `sliceReduceMirroredFFT` from mrcslice.c:873.
pub fn slice_reduce_mirrored_fft(s: &mut Islice) -> i32 {
    if s.mode != MRC_MODE_COMPLEX_FLOAT {
        return 1;
    }
    let nx = s.xsize;
    let nfloats = nx + 2;
    let values = s.data.f_mut();
    for j in 0..s.ysize {
        let tmp1 = values[(j * 2 * nx) as usize];
        let tmp2 = values[(j * 2 * nx + 1) as usize];
        for i in 0..nx {
            values[(i + j * nfloats) as usize] = values[(i + nx + j * 2 * nx) as usize];
        }
        values[(nx + j * nfloats) as usize] = tmp1;
        values[(nx + 1 + j * nfloats) as usize] = tmp2;
    }
    s.xsize = nx / 2 + 1;
    0
}

/// `sliceWriteMRCfile` from mrcslice.c:903.
pub fn slice_write_mrcfile(filename: &str, slice: &mut Islice) -> i32 {
    let Some(mut file) = crate::imod::libcfshr::b3dutil::ImodFile::open(filename, "wb") else {
        return -1;
    };
    let mut hout = MrcHeader::default();
    mrc_head_new(&mut hout, slice.xsize, slice.ysize, 1, slice.mode);
    slice_mmm(slice);
    hout.amin = slice.min;
    hout.amax = slice.max;
    hout.amean = slice.mean;
    if mrc_head_write(&mut file, &mut hout) != 0 {
        drop(file);
        return -2;
    }
    let error = mrc_write_slice(slice.data.bytes(), &mut file, &mut hout, 0, b'z');
    drop(file);
    error
}

pub fn slice_gradient(sin: &mut Islice) -> Option<Islice> {
    let mut s = slice_create(sin.xsize, sin.ysize, sin.mode)?;
    for j in 0..sin.ysize {
        for i in 0..sin.xsize - 1 {
            let (mut v, mut n) = ([0.; 4], [0.; 4]);
            slice_get_val(sin, i, j, &mut v);
            slice_get_val(sin, i + 1, j, &mut n);
            v[0] = (n[0] - v[0]).abs();
            slice_put_val(s.as_mut(), i, j, v);
        }
    }
    for i in 0..sin.xsize {
        for j in 0..sin.ysize - 1 {
            let (mut v, mut n, mut g) = ([0.; 4], [0.; 4], [0.; 4]);
            slice_get_val(sin, i, j, &mut v);
            slice_get_val(sin, i, j + 1, &mut n);
            slice_get_val(s.as_mut(), i, j, &mut g);
            g[0] = ((n[0] - v[0]).abs() + g[0]) / 2.;
            slice_put_val(s.as_mut(), i, j, g);
        }
        let mut v = [0.; 4];
        slice_get_val(s.as_mut(), i, sin.ysize - 2, &mut v);
        slice_put_val(s.as_mut(), i, sin.ysize - 1, v);
    }
    for j in 0..sin.ysize {
        let mut v = [0.; 4];
        slice_get_val(s.as_mut(), sin.xsize - 2, j, &mut v);
        slice_put_val(s.as_mut(), sin.xsize - 1, j, v);
    }
    slice_mmm(s.as_mut());
    Some(s)
}
pub fn mrc_bandpass_filter(sin: &mut Islice, low: f64, high: f64) -> i32 {
    if sin.mode != MRC_MODE_COMPLEX_FLOAT {
        return -1;
    }
    let xscale = 0.5 / (sin.xsize as f64 - 1.);
    for j in 0..sin.ysize {
        for i in 0..sin.xsize {
            let dx = xscale * i as f64;
            // C `mrcslice.c:1183`: `dy = (float)j / sin->ysize - 0.5;` divides
            // in float (both operands convert to float) and only then widens
            // for the `- 0.5`, so the quotient is rounded to float first.
            let dy = (j as f32 / sin.ysize as f32) as f64 - 0.5;
            let dist = (dx * dx + dy * dy).sqrt();
            let mut m = if low > 0. {
                if dist < 0.00001 {
                    0.
                } else {
                    1. / (1. + (low / dist).powf(3.))
                }
            } else {
                1.
            };
            if high > 0. {
                m *= 1. / (1. + (dist / high).powf(3.));
            }
            let mut v = [0.; 4];
            slice_get_val(sin, i, j, &mut v);
            // C `mrcslice.c:1197-1198`: `val[0] *= mval;` with `mval` double
            // multiplies in double and rounds to float once on the store.
            v[0] = (v[0] as f64 * m) as f32;
            v[1] = (v[1] as f64 * m) as f32;
            slice_put_val(sin, i, j, v);
        }
    }
    0
}
pub fn slice_quad_interpolate(sl: &mut Islice, x: f64, y: f64, val: &mut [f32; 4]) {
    let xi = (x + 0.5).floor() as i32;
    let yi = (y + 0.5).floor() as i32;
    let dx = (x - xi as f64) as f32;
    let dy = (y - yi as f64) as f32;
    let (mut x1, mut x2, mut y1, mut y2) = ([0.; 4], [0.; 4], [0.; 4], [0.; 4]);
    slice_get_val(sl, xi, yi, val);
    slice_get_val(sl, xi - 1, yi, &mut x1);
    slice_get_val(sl, xi + 1, yi, &mut x2);
    slice_get_val(sl, xi, yi - 1, &mut y1);
    slice_get_val(sl, xi, yi + 1, &mut y2);
    for q in 0..sl.csize.min(3) as usize {
        let a = (x1[q] + x2[q]) * 0.5 - (*val)[q];
        let b = (y1[q] + y2[q]) * 0.5 - (*val)[q];
        let c = (x2[q] - x1[q]) * 0.5;
        let d = (y2[q] - y1[q]) * 0.5;
        val[q] = a * dx * dx + b * dy * dy + c * dx + d * dy + val[q];
    }
}
pub fn mrc_slice_rotates(
    slin: &mut Islice,
    sout: &mut Islice,
    angle: f64,
    cx: f64,
    cy: f64,
) -> i32 {
    let a = angle * 0.017453293;
    let co = (-a).cos();
    let si = (-a).sin();
    let x2 = sout.xsize as f64 * 0.5;
    let y2 = sout.ysize as f64 * 0.5;
    for j in 0..sout.ysize {
        for i in 0..sout.xsize {
            let x = (i as f64 - x2) * co - (j as f64 - y2) * si + cx;
            let y = (i as f64 - x2) * si + (j as f64 - y2) * co + cy;
            let mut v = [0.; 4];
            slice_quad_interpolate(slin, x, y, &mut v);
            slice_put_val(sout, i, j, v);
        }
    }
    0
}
pub fn mrc_slice_zooms(
    sin: &mut Islice,
    sout: &mut Islice,
    xz: f64,
    yz: f64,
    cx: f64,
    cy: f64,
) -> i32 {
    if xz == 0. || yz == 0. {
        return 1;
    }
    let sbx = xz * cx - sout.xsize as f64 * 0.5;
    let sby = yz * cy - sout.ysize as f64 * 0.5;
    for j in 0..sout.ysize {
        for i in 0..sout.xsize {
            let mut v = [0.; 4];
            slice_quad_interpolate(sin, i as f64 / xz + sbx, j as f64 / yz + sby, &mut v);
            slice_put_val(sout, i, j, v);
        }
    }
    0
}
pub fn mrc_slice_wrap(s: &mut Islice) -> i32 {
    let mx = s.xsize / 2;
    let my = s.ysize / 2;
    for j in 0..my {
        for i in 0..mx {
            let (mut a, mut b) = ([0.; 4], [0.; 4]);
            slice_get_val(s, i, j, &mut a);
            slice_get_val(s, i + mx, j + my, &mut b);
            slice_put_val(s, i, j, b);
            slice_put_val(s, i + mx, j + my, a);
        }
        for i in mx..s.xsize {
            let (mut a, mut b) = ([0.; 4], [0.; 4]);
            slice_get_val(s, i, j, &mut a);
            slice_get_val(s, i - mx, j + my, &mut b);
            slice_put_val(s, i, j, b);
            slice_put_val(s, i - mx, j + my, a);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corr_conj_uses_the_fft_arrays_as_bounded_pairs() {
        let mut first = [2.0_f32, 3.0, -1.0, 4.0];
        let second = [5.0_f32, 7.0, 2.0, -6.0];
        assert_eq!(corr_conj(&mut first, Some(&second), 2), 0);
        assert_eq!(first, [31.0, -1.0, -26.0, -2.0]);
    }

    #[test]
    fn reducing_mirrored_fft_uses_owned_byte_storage() {
        let mut slice = slice_create(2, 1, MRC_MODE_COMPLEX_FLOAT).unwrap();
        slice
            .data
            .f_mut()
            .copy_from_slice(&[10.0_f32, 20.0, 30.0, 40.0]);
        assert_eq!(slice_reduce_mirrored_fft(&mut slice), 0);
        assert_eq!(slice.xsize, 2);
        assert_eq!(slice.data.f(), [30.0_f32, 40.0, 10.0, 20.0]);
    }

    #[test]
    fn wrapping_fft_lines_uses_a_borrowed_slice() {
        let mut slice = slice_create(2, 4, MRC_MODE_COMPLEX_FLOAT).unwrap();
        for (i, v) in slice.data.f_mut().iter_mut().enumerate() {
            *v = i as f32;
        }
        assert_eq!(slice_wrap_fft_lines(&mut slice, 0), 0);
        assert_eq!(
            slice.data.f(),
            vec![
                8., 9., 10., 11., 12., 13., 14., 15., 0., 1., 2., 3., 4., 5., 6., 7.
            ]
        );
    }

    #[test]
    fn slice_write_mrcfile_round_trips_a_real_mrc_stack() {
        let path = format!(
            "/tmp/imod-rs-mrcslice-{}-{}.mrc",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        );
        let mut slice = slice_create(3, 2, MRC_MODE_BYTE).unwrap();
        for (index, value) in [2.0_f32, 7.0, 1.0, 8.0, 2.0, 8.0].into_iter().enumerate() {
            slice_put_val(
                &mut slice,
                (index % 3) as i32,
                (index / 3) as i32,
                [value; 4],
            );
        }
        assert_eq!(slice_write_mrcfile(&path, &mut slice), 0);
        let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "rb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(
            crate::imod::libiimod::mrcfiles::mrc_head_read(&mut file, &mut header),
            0
        );
        let mut values = [0_u8; 6];
        assert_eq!(
            crate::imod::libiimod::mrcfiles::mrc_read_slice(
                &mut values,
                &mut file,
                &mut header,
                0,
                b'Z',
            ),
            0
        );
        assert_eq!(values, [2, 7, 1, 8, 2, 8]);
        drop(file);
        std::fs::remove_file(&path).unwrap();
    }
}
