//! Translation of `IMOD/libiimod/mrcslice.c` and `include/mrcslice.h`.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::islice::{Islice, Istack, slice_create, slice_get_val, slice_put_val};
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
pub fn slice_read_mrc(hin: &mut MrcHeader, sno: i32, axis: u8) -> Option<Box<Islice>> {
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
    let mut slice = slice_create(nx, ny, hin.mode)?;
    slice.data.copy_from_slice(&buf);
    slice.mean = hin.amean;
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
) -> Option<Box<Islice>> {
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
        if crate::imod::libiimod::mrcsec::mrc_read_section(hin, &mut li, &mut slice.data, sec_num)
            == 0
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
    let mut slice = slice_create(nx, ny, hin.mode)?;
    slice.data.copy_from_slice(&buffer);
    slice.mean = hin.amean;
    if slice_box_in(slice.as_mut(), llx, lly, urx, ury) != 0 {
        return None;
    }
    Some(slice)
}

/// `sliceReadFloat` from mrcslice.c:1089.
pub fn slice_read_float(hin: &mut MrcHeader, secno: i32) -> Option<Box<Islice>> {
    if crate::imod::libcfshr::islice::slice_mode_if_real(hin.mode) < 0 {
        return None;
    }
    let Some(mut slice) = slice_create(hin.nx, hin.ny, MRC_MODE_FLOAT) else {
        return None;
    };
    let mut values = vec![0_f32; (hin.nx * hin.ny) as usize];
    if crate::imod::libiimod::mrcfiles::mrc_read_float_slice(&mut values, hin, secno) != 0 {
        return None;
    }
    for (bytes, value) in slice.data.chunks_exact_mut(size_of::<f32>()).zip(values) {
        bytes.copy_from_slice(&value.to_ne_bytes());
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
            s.min = s.min.min(val[0]);
            s.max = s.max.max(val[0]);
            tsum += val[0] as f64;
        }
        sum += tsum;
    }
    s.mean = (sum / (s.xsize * s.ysize) as f64) as f32;
    0
}

pub fn full_array_min_max_mean(
    array: &[u8],
    typ: i32,
    nx: i32,
    ny: i32,
) -> Option<(f32, f32, f32)> {
    let mut slice = slice_create(nx, ny, typ)?;
    if slice.data.len() != array.len() {
        return None;
    }
    slice.data.copy_from_slice(array);
    slice_mmm(slice.as_mut());
    Some((slice.min, slice.max, slice.mean))
}

pub fn mrc_slice_getvol(v: &Istack, sno: i32, axis: u8) -> Option<Box<Islice>> {
    let first = v.slices.first()?;
    match axis as char {
        'y' | 'Y' => {
            let mut out = slice_create(first.xsize, v.slices.len() as i32, first.mode)?;
            for (k, source) in v.slices.iter().enumerate() {
                for i in 0..out.xsize {
                    let mut value = [0.; 4];
                    slice_get_val(source.as_ref(), i, sno, &mut value);
                    slice_put_val(out.as_mut(), i, k as i32, value);
                }
            }
            Some(out)
        }
        'x' | 'X' => {
            let mut out = slice_create(first.ysize, v.slices.len() as i32, first.mode)?;
            for (k, source) in v.slices.iter().enumerate() {
                for j in 0..out.xsize {
                    let mut value = [0.; 4];
                    slice_get_val(source.as_ref(), sno, j, &mut value);
                    slice_put_val(out.as_mut(), j, k as i32, value);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

pub fn mrc_slice_putvol(v: &mut Istack, s: Box<Islice>, sno: i32, axis: u8) -> i32 {
    match axis as char {
        'z' | 'Z' => {
            let Some(slot) = v.slices.get_mut(sno as usize) else {
                return -1;
            };
            *slot = s;
        }
        'y' | 'Y' => {
            for (k, target) in v.slices.iter_mut().enumerate() {
                for i in 0..s.xsize {
                    let mut value = [0.; 4];
                    slice_get_val(s.as_ref(), i, k as i32, &mut value);
                    slice_put_val(target.as_mut(), i, sno, value);
                }
            }
        }
        'x' | 'X' => {
            for (k, target) in v.slices.iter_mut().enumerate() {
                for j in 0..s.ysize {
                    let mut value = [0.; 4];
                    slice_get_val(s.as_ref(), j, k as i32, &mut value);
                    slice_put_val(target.as_mut(), sno, j, value);
                }
            }
        }
        _ => return -1,
    }
    0
}

pub fn corr_conj(g: &mut [f32], h: &[f32]) -> i32 {
    if g.len() != h.len() || !g.len().is_multiple_of(2) {
        return -1;
    }
    for (g_pair, h_pair) in g.chunks_exact_mut(2).zip(h.chunks_exact(2)) {
        let (temp, imaginary) = (g_pair[0], g_pair[1]);
        let (rtmp, itmp) = (h_pair[0], h_pair[1]);
        g_pair[0] = rtmp * temp + itmp * imaginary;
        g_pair[1] = itmp * temp - rtmp * imaginary;
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
    for j in 0..sin.ysize {
        for i in 0..sin.xsize {
            let mut v = [0.; 4];
            slice_get_val(sin, i, j, &mut v);
            for q in 0..sin.csize.min(3) as usize {
                v[q] = offset + scale * v[q];
                if max != 0. {
                    v[q] = v[q].clamp(min, max);
                }
            }
            slice_put_val(sin, i, j, v);
        }
    }
    0
}

pub fn slice_box(sl: &mut Islice, llx: i32, lly: i32, urx: i32, ury: i32) -> Option<Box<Islice>> {
    let mut sout = slice_create(urx - llx, ury - lly, sl.mode)?;
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
    let replacement = *sout;
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
pub fn mrc_slice_resize(slin: &mut Islice, nx: i32, ny: i32) -> Option<Box<Islice>> {
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
        if s.data.len() % size_of::<f32>() != 0 {
            return 1;
        }
        let mut values = s
            .data
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        if values.len() != (2 * nx * ny) as usize {
            return 1;
        }
        let mut buffer = Vec::<f32>::new();
        if buffer.try_reserve_exact((2 * nx) as usize).is_err() {
            return 2;
        }
        buffer.resize((2 * nx) as usize, 0.);
        crate::imod::libcfshr::filtxcorr::wrap_fft_slice(
            &mut values,
            &mut buffer,
            nx,
            ny,
            direction,
        );
        for (bytes, value) in s.data.chunks_exact_mut(size_of::<f32>()).zip(values) {
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        return 0;
    }
    let mut buffer = Vec::<u8>::new();
    if buffer.try_reserve_exact((pixsize * nx) as usize).is_err() {
        return 2;
    }
    buffer.resize((pixsize * nx) as usize, 0);
    for i in 0..ny / 2 {
        let line_bytes = (pixsize * nx) as usize;
        let ind1 = i as usize * line_bytes;
        let ind2 = (i + ny / 2) as usize * line_bytes;
        if ind2 + line_bytes > s.data.len() {
            return 1;
        }
        buffer.copy_from_slice(&s.data[ind1..ind1 + line_bytes]);
        s.data.copy_within(ind2..ind2 + line_bytes, ind1);
        s.data[ind2..ind2 + line_bytes].copy_from_slice(&buffer);
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
    if nx < 0 || s.ysize < 0 || s.data.len() % size_of::<f32>() != 0 {
        return 1;
    }
    let mut values = s
        .data
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let input_values = (2 * nx * s.ysize) as usize;
    let reduced_values = (nfloats * s.ysize) as usize;
    if values.len() < input_values.max(reduced_values) {
        return 1;
    }
    for j in 0..s.ysize {
        let tmp1 = values[(j * 2 * nx) as usize];
        let tmp2 = values[(j * 2 * nx + 1) as usize];
        for i in 0..nx {
            values[(i + j * nfloats) as usize] = values[(i + nx + j * 2 * nx) as usize];
        }
        values[(nx + j * nfloats) as usize] = tmp1;
        values[(nx + 1 + j * nfloats) as usize] = tmp2;
    }
    for (bytes, value) in s.data.chunks_exact_mut(size_of::<f32>()).zip(values) {
        bytes.copy_from_slice(&value.to_ne_bytes());
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
    let error = mrc_write_slice(&slice.data, &mut file, &mut hout, 0, b'z');
    drop(file);
    error
}

/// `mrcWriteImageToFile` from mrcslice.c:931.
pub fn mrc_write_image_to_file(filename: &str, array: &[u8], mode: i32, nx: i32, ny: i32) -> i32 {
    let Some(mut slice) = slice_create(nx, ny, mode) else {
        return -1;
    };
    if slice.data.len() != array.len() {
        return -1;
    }
    slice.data.copy_from_slice(array);
    slice_write_mrcfile(filename, slice.as_mut())
}

pub fn slice_gradient(sin: &mut Islice) -> Option<Box<Islice>> {
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
pub fn mrc_slice_rotate(
    slin: &mut Islice,
    angle: f64,
    xsize: i32,
    ysize: i32,
    cx: f64,
    cy: f64,
) -> Option<Box<Islice>> {
    let mut sout = slice_create(xsize, ysize, slin.mode)?;
    mrc_slice_rotates(slin, sout.as_mut(), angle, cx, cy);
    Some(sout)
}
pub fn mrc_slice_translate(
    sin: &mut Islice,
    dx: f64,
    dy: f64,
    xsize: i32,
    ysize: i32,
) -> Option<Box<Islice>> {
    let mut sout = slice_create(xsize, ysize, sin.mode)?;
    for j in 0..ysize {
        for i in 0..xsize {
            let mut v = [0.; 4];
            slice_quad_interpolate(sin, i as f64 + dx, j as f64 + dy, &mut v);
            slice_put_val(sout.as_mut(), i, j, v);
        }
    }
    Some(sout)
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
pub fn mrc_slice_zoom(
    sin: &mut Islice,
    xz: f64,
    yz: f64,
    xsize: i32,
    ysize: i32,
    cx: f64,
    cy: f64,
) -> Option<Box<Islice>> {
    if xz == 0. || yz == 0. {
        return None;
    }
    let mut sout = slice_create(xsize, ysize, sin.mode)?;
    mrc_slice_zooms(sin, sout.as_mut(), xz, yz, cx, cy);
    Some(sout)
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
pub fn mrc_slice_real(sin: &mut Islice) -> Option<Box<Islice>> {
    if sin.mode != MRC_MODE_COMPLEX_FLOAT {
        return None;
    }
    let mut sout = slice_create(sin.xsize, sin.ysize, MRC_MODE_FLOAT)?;
    if sin.data.len() % size_of::<f32>() != 0 || sout.data.len() % size_of::<f32>() != 0 {
        return None;
    }
    for (out, input) in sout
        .data
        .chunks_exact_mut(size_of::<f32>())
        .zip(sin.data.chunks_exact(2 * size_of::<f32>()))
    {
        out.copy_from_slice(&input[..size_of::<f32>()]);
    }
    Some(sout)
}
pub fn mrc_slice_lie_img(sin: &mut Islice, mask: &mut Islice, alpha: f64) -> i32 {
    let a = alpha as f32;
    let ma = 1. - a;
    for j in 0..sin.ysize {
        for i in 0..sin.xsize {
            let (mut v, mut w) = ([0.; 4], [0.; 4]);
            slice_get_val(sin, i, j, &mut v);
            slice_get_val(mask, i, j, &mut w);
            v[0] = ma * v[0] + a * w[0];
            let (lo, hi) = match sin.mode {
                MRC_MODE_BYTE => (0., 255.),
                MRC_MODE_SHORT => (-32768., 32767.),
                MRC_MODE_USHORT => (0., 65535.),
                _ => (f32::NEG_INFINITY, f32::INFINITY),
            };
            v[0] = v[0].clamp(lo, hi);
            if sin.csize == 3 {
                v[1] = ma * v[1] + a * w[1];
                v[2] = ma * v[2] + a * w[2];
            }
            slice_put_val(sin, i, j, v);
        }
    }
    0
}
pub fn mrc_vol_wrap(v: &mut Istack) -> i32 {
    for slice in &mut v.slices {
        mrc_slice_wrap(slice.as_mut());
    }
    let midpoint = v.slices.len() / 2;
    for k in 0..midpoint {
        v.slices.swap(k, k + midpoint);
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
        assert_eq!(corr_conj(&mut first, &second), 0);
        assert_eq!(first, [31.0, -1.0, -26.0, -2.0]);
    }

    #[test]
    fn corr_conj_rejects_non_complex_or_mismatched_slices() {
        let mut first = [2.0_f32, 3.0, -1.0];
        assert_eq!(corr_conj(&mut first, &[5.0, 7.0, 2.0]), -1);
        assert_eq!(corr_conj(&mut first, &[5.0, 7.0]), -1);
    }

    #[test]
    fn complex_real_copy_preserves_native_float_bytes_without_casting_storage() {
        let mut complex = slice_create(2, 1, MRC_MODE_COMPLEX_FLOAT).unwrap();
        complex.data = [1.5_f32, -9.0, -2.25, 7.0]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect();
        let real = mrc_slice_real(complex.as_mut()).unwrap();
        assert_eq!(
            real.data,
            [1.5_f32, -2.25]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn reducing_mirrored_fft_uses_owned_byte_storage() {
        let mut slice = slice_create(2, 1, MRC_MODE_COMPLEX_FLOAT).unwrap();
        slice.data = [10.0_f32, 20.0, 30.0, 40.0]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect();
        assert_eq!(slice_reduce_mirrored_fft(slice.as_mut()), 0);
        assert_eq!(slice.xsize, 2);
        assert_eq!(
            slice.data,
            [30.0_f32, 40.0, 10.0, 20.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn wrapping_fft_lines_uses_a_borrowed_slice() {
        let mut slice = slice_create(2, 4, MRC_MODE_COMPLEX_FLOAT).unwrap();
        slice.data = (0..16)
            .map(|value| value as f32)
            .flat_map(f32::to_ne_bytes)
            .collect();
        assert_eq!(slice_wrap_fft_lines(slice.as_mut(), 0), 0);
        let values = slice
            .data
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            values,
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
                slice.as_mut(),
                (index % 3) as i32,
                (index / 3) as i32,
                [value; 4],
            );
        }
        assert_eq!(slice_write_mrcfile(&path, slice.as_mut()), 0);
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
