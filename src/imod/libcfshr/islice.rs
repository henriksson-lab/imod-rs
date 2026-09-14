//! Translation of `IMOD/libcfshr/islice.c` and its direct `mrcslice.h` layouts.
#![allow(dead_code)]
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_4BIT, MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, MRC_MODE_HALF_FLOAT,
    MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT,
};
const SMOOTH_KERNEL: [[i32; 3]; 3] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
const SHARPEN_KERNEL: [[i32; 3]; 3] = [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]];
const LAPLACIAN_KERNEL: [[i32; 3]; 3] = [[1, 1, 1], [1, -4, 1], [1, 1, 1]];
pub struct Islice {
    pub data: Vec<u8>,
    pub xsize: i32,
    pub ysize: i32,
    pub mode: i32,
    pub csize: i32,
    pub dsize: i32,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub index: i32,
    pub cval: [f32; 4],
}
pub struct Istack {
    pub slices: Vec<Box<Islice>>,
}
pub fn slice_create(xsize: i32, ysize: i32, mode: i32) -> Option<Box<Islice>> {
    let xysize = (xsize as usize).wrapping_mul(ysize as usize);
    if xysize / xsize as usize != ysize as usize {
        return None;
    }
    let mut dsize = 0;
    let mut csize = 0;
    if crate::imod::libcfshr::b3dutil::data_size_for_mode(mode, &mut dsize, &mut csize) != 0 {
        return None;
    }
    Some(Box::new(Islice {
        data: vec![0; xysize * dsize as usize * csize as usize],
        xsize,
        ysize,
        mode,
        csize,
        dsize,
        min: 0.,
        max: 0.,
        mean: 0.,
        index: -1,
        cval: [0.; 4],
    }))
}
pub unsafe fn slice_init(s: *mut Islice, xsize: i32, ysize: i32, mode: i32, data: Vec<u8>) -> i32 {
    let mut dsize = 0;
    let mut csize = 0;
    if crate::imod::libcfshr::b3dutil::data_size_for_mode(mode, &mut dsize, &mut csize) != 0 {
        return -1;
    }
    let Some(bytes) = usize::try_from(xsize)
        .ok()
        .and_then(|x| usize::try_from(ysize).ok().and_then(|y| x.checked_mul(y)))
        .and_then(|pixels| pixels.checked_mul(dsize as usize))
        .and_then(|size| size.checked_mul(csize as usize))
    else {
        return -1;
    };
    if data.len() != bytes {
        return -1;
    }
    unsafe {
        (*s).xsize = xsize;
        (*s).ysize = ysize;
        (*s).mode = mode;
        (*s).dsize = dsize;
        (*s).csize = csize;
        (*s).data = data;
    }
    0
}
pub fn slice_free(_s: Box<Islice>) {}
pub unsafe fn slice_clear(s: *mut Islice, val: [f32; 4]) {
    unsafe {
        (*s).min = slice_get_val_magnitude(val, (*s).mode);
        (*s).max = (*s).min;
        (*s).mean = (*s).min;
        for j in 0..(*s).ysize as u32 {
            for i in 0..(*s).xsize as u32 {
                slice_put_val(s, i as i32, j as i32, val);
            }
        }
    }
}
pub fn slice_mode(mst: &[u8]) -> i32 {
    {
        let value = mst;
        if value == b"byte" || value == b"0" {
            0
        } else if value == b"sbyte" {
            -2
        } else if value == b"ubyte" {
            -3
        } else if value == b"short" || value == b"1" {
            1
        } else if value == b"float" || value == b"2" {
            2
        } else if value == b"3" {
            3
        } else if value == b"complex" || value == b"4" {
            4
        } else if value == b"ushort" || value == b"6" {
            6
        } else if value == b"16" || value == b"rgb" {
            16
        } else {
            -1
        }
    }
}
pub fn slice_mode_if_real(mrc_mode: i32) -> i32 {
    if mrc_mode == MRC_MODE_BYTE || mrc_mode == MRC_MODE_4BIT {
        0
    } else if mrc_mode == MRC_MODE_SHORT {
        1
    } else if mrc_mode == MRC_MODE_USHORT {
        6
    } else if mrc_mode == MRC_MODE_FLOAT || mrc_mode == MRC_MODE_HALF_FLOAT {
        2
    } else {
        -1
    }
}
pub unsafe fn slice_get_x_size(slice: *mut Islice) -> i32 {
    unsafe { (*slice).xsize }
}
pub unsafe fn slice_get_y_size(slice: *mut Islice) -> i32 {
    unsafe { (*slice).ysize }
}
pub unsafe fn slice_get_val(s: *mut Islice, x: i32, y: i32, val: *mut [f32; 4]) -> i32 {
    unsafe {
        if x < 0 || y < 0 || x >= (*s).xsize || y >= (*s).ysize {
            (*val)[0] = (*s).mean;
            return -1;
        }
        let mut index = x as usize + y as usize * (*s).xsize as usize;
        (*val)[1] = 0.;
        match (*s).mode {
            0 => (*val)[0] = *(*s).data.as_mut_ptr().add(index) as f32,
            1 => (*val)[0] = *(*s).data.as_mut_ptr().cast::<i16>().add(index) as f32,
            6 => (*val)[0] = *(*s).data.as_mut_ptr().cast::<u16>().add(index) as f32,
            2 => (*val)[0] = *(*s).data.as_mut_ptr().cast::<f32>().add(index),
            3 => {
                index *= 2;
                (*val)[0] = *(*s).data.as_mut_ptr().cast::<i16>().add(index) as f32;
                (*val)[1] = *(*s).data.as_mut_ptr().cast::<i16>().add(index + 1) as f32;
            }
            4 => {
                index *= 2;
                (*val)[0] = *(*s).data.as_mut_ptr().cast::<f32>().add(index);
                (*val)[1] = *(*s).data.as_mut_ptr().cast::<f32>().add(index + 1);
            }
            16 => {
                index *= 3;
                (*val)[0] = *(*s).data.as_mut_ptr().add(index) as f32;
                (*val)[1] = *(*s).data.as_mut_ptr().add(index + 1) as f32;
                (*val)[2] = *(*s).data.as_mut_ptr().add(index + 2) as f32;
            }
            99 => {
                index *= 3;
                (*val)[0] = *(*s).data.as_mut_ptr().cast::<f32>().add(index);
                (*val)[1] = *(*s).data.as_mut_ptr().cast::<f32>().add(index + 1);
                (*val)[2] = *(*s).data.as_mut_ptr().cast::<f32>().add(index + 2);
            }
            _ => return -1,
        };
        0
    }
}
pub unsafe fn slice_put_val(s: *mut Islice, x: i32, y: i32, val: [f32; 4]) -> i32 {
    unsafe {
        if x < 0 || y < 0 || x >= (*s).xsize || y >= (*s).ysize {
            return -1;
        }
        let mut i = x as usize + y as usize * (*s).xsize as usize;
        // `islice.c:270-285` casts the float straight to the narrow integer
        // type.  In C that truncates toward zero into an int and then keeps the
        // low bits, so -1.0f stores 255 and 300.0f stores 44.  Rust's float
        // `as u8` SATURATES instead (0 and 255), which silently rewrites pixel
        // data: `clip multiply` of a byte file by a float file wrote 15300 of
        // 15360 pixels as 127.  Going through `as i32` first reproduces the C
        // for every value in i32 range; outside it the C is undefined anyway.
        match (*s).mode {
            0 => *(*s).data.as_mut_ptr().add(i) = val[0] as i32 as u8,
            1 => *(*s).data.as_mut_ptr().cast::<i16>().add(i) = val[0] as i32 as i16,
            6 => *(*s).data.as_mut_ptr().cast::<u16>().add(i) = val[0] as i32 as u16,
            2 => *(*s).data.as_mut_ptr().cast::<f32>().add(i) = val[0],
            3 => {
                i *= 2;
                *(*s).data.as_mut_ptr().cast::<i16>().add(i) = val[0] as i32 as i16;
                *(*s).data.as_mut_ptr().cast::<i16>().add(i + 1) = val[1] as i32 as i16
            }
            4 => {
                i *= 2;
                *(*s).data.as_mut_ptr().cast::<f32>().add(i) = val[0];
                *(*s).data.as_mut_ptr().cast::<f32>().add(i + 1) = val[1]
            }
            16 => {
                i *= 3;
                *(*s).data.as_mut_ptr().add(i) = val[0] as i32 as u8;
                *(*s).data.as_mut_ptr().add(i + 1) = val[1] as i32 as u8;
                *(*s).data.as_mut_ptr().add(i + 2) = val[2] as i32 as u8
            }
            99 => {
                i *= 3;
                *(*s).data.as_mut_ptr().cast::<f32>().add(i) = val[0];
                *(*s).data.as_mut_ptr().cast::<f32>().add(i + 1) = val[1];
                *(*s).data.as_mut_ptr().cast::<f32>().add(i + 2) = val[2]
            }
            _ => return -1,
        };
        0
    }
}
pub unsafe fn slice_get_pixel_magnitude(s: *mut Islice, x: i32, y: i32) -> f32 {
    let mut val = [0.; 4];
    let mut m: f32;
    unsafe {
        slice_get_val(s, x, y, &mut val);
        if (*s).csize == 1 {
            return val[0];
        }
        if (*s).csize == 2 {
            m = val[0] * val[0] + val[1] * val[1];
            return m.sqrt();
        }
        m = val[0] * 0.3;
        m += val[1] * 0.59;
        m += val[2] * 0.11;
        m
    }
}
pub fn slice_get_val_magnitude(val: [f32; 4], mode: i32) -> f32 {
    if mode == 3 || mode == 4 {
        return (val[0] * val[0] + val[1] * val[1]).sqrt();
    }
    if mode == 16 {
        return val[0] * 0.3 + val[1] * 0.59 + val[2] * 0.11;
    }
    val[0]
}
pub unsafe fn slice_min_max(s: *mut Islice) -> i32 {
    unsafe {
        let mut imin: i32;
        let mut imax: i32;
        let mut ival: i32;
        let mut fmin: f32;
        let mut fmax: f32;
        let mut fval: f32;
        match (*s).mode {
            0 => {
                imin = *(*s).data.as_mut_ptr() as i32;
                imax = *(*s).data.as_mut_ptr() as i32;
                for i in 1..((*s).xsize * (*s).ysize) {
                    ival = *(*s).data.as_mut_ptr().add(i as usize) as i32;
                    if imin > ival {
                        imin = ival;
                    }
                    if imax < ival {
                        imax = ival;
                    }
                }
                (*s).min = imin as f32;
                (*s).max = imax as f32;
            }
            1 => {
                imin = *(*s).data.as_mut_ptr().cast::<i16>() as i32;
                imax = *(*s).data.as_mut_ptr().cast::<i16>() as i32;
                for i in 1..((*s).xsize * (*s).ysize) {
                    ival = *(*s).data.as_mut_ptr().cast::<i16>().add(i as usize) as i32;
                    if imin > ival {
                        imin = ival;
                    }
                    if imax < ival {
                        imax = ival;
                    }
                }
                (*s).min = imin as f32;
                (*s).max = imax as f32;
            }
            6 => {
                imin = *(*s).data.as_mut_ptr().cast::<u16>() as i32;
                imax = *(*s).data.as_mut_ptr().cast::<u16>() as i32;
                for i in 1..((*s).xsize * (*s).ysize) {
                    ival = *(*s).data.as_mut_ptr().cast::<u16>().add(i as usize) as i32;
                    if imin > ival {
                        imin = ival;
                    }
                    if imax < ival {
                        imax = ival;
                    }
                }
                (*s).min = imin as f32;
                (*s).max = imax as f32;
            }
            2 => {
                fmin = *(*s).data.as_mut_ptr().cast::<f32>();
                fmax = *(*s).data.as_mut_ptr().cast::<f32>();
                for i in 1..((*s).xsize * (*s).ysize) {
                    fval = *(*s).data.as_mut_ptr().cast::<f32>().add(i as usize);
                    if fmin > fval {
                        fmin = fval;
                    }
                    if fmax < fval {
                        fmax = fval;
                    }
                }
                (*s).min = fmin;
                (*s).max = fmax;
            }
            _ => return 1,
        }
        0
    }
}
pub unsafe fn slice_scale_and_free(sout: *mut Islice, sin: *mut Islice) {
    unsafe {
        let mut aval = 0.;
        let mut mval = 1.;
        if (*sin).min != 0. || (*sin).max != 0. {
            slice_min_max(sout);
            mval = ((*sin).max - (*sin).min) / ((*sout).max - (*sout).min);
            aval = (*sin).min - mval * (*sout).min;
        }
        let imax = (*sin).xsize * (*sin).ysize;
        // `islice.c:448-460` narrows with `(unsigned char)`, which truncates
        // toward zero and keeps the low bits rather than saturating.
        match (*sout).mode {
            2 => {
                for i in 0..imax {
                    *(*sin).data.as_mut_ptr().add(i as usize) =
                        (*(*sout).data.as_mut_ptr().cast::<f32>().add(i as usize) * mval + aval)
                            as i32 as u8;
                }
            }
            1 => {
                for i in 0..imax {
                    *(*sin).data.as_mut_ptr().add(i as usize) =
                        (*(*sout).data.as_mut_ptr().cast::<i16>().add(i as usize) as f32 * mval
                            + aval) as i32 as u8;
                }
            }
            6 => {
                for i in 0..imax {
                    *(*sin).data.as_mut_ptr().add(i as usize) =
                        (*(*sout).data.as_mut_ptr().cast::<u16>().add(i as usize) as f32 * mval
                            + aval) as i32 as u8;
                }
            }
            0 => {
                for i in 0..imax {
                    *(*sin).data.as_mut_ptr().add(i as usize) =
                        (*(*sout).data.as_mut_ptr().add(i as usize) as f32 * mval + aval) as i32
                            as u8;
                }
            }
            _ => {}
        }
    }
}
pub unsafe fn slice_byte_edge_laplacian(sin: *mut Islice) -> i32 {
    unsafe { slice_byte_convolve(sin, &LAPLACIAN_KERNEL) }
}
pub unsafe fn slice_byte_sharpen(sin: *mut Islice) -> i32 {
    unsafe { slice_byte_convolve(sin, &SHARPEN_KERNEL) }
}
pub unsafe fn slice_byte_smooth(sin: *mut Islice) -> i32 {
    unsafe { slice_byte_convolve(sin, &SMOOTH_KERNEL) }
}
pub unsafe fn slice_byte_convolve(sin: *mut Islice, mask: *const [[i32; 3]; 3]) -> i32 {
    unsafe {
        let Some(mut sout) = slice_create((*sin).xsize, (*sin).ysize, 1) else {
            return -1;
        };
        let imax = (*sin).xsize - 1;
        let jmax = (*sin).ysize - 1;
        for i in 1..imax {
            for j in 1..jmax {
                let val = *(*sin)
                    .data
                    .as_mut_ptr()
                    .add((i + 1 + (j + 1) * (*sin).xsize) as usize)
                    as i32
                    * (*mask)[0][0]
                    + *(*sin)
                        .data
                        .as_mut_ptr()
                        .add((i + (j + 1) * (*sin).xsize) as usize) as i32
                        * (*mask)[0][1]
                    + *(*sin)
                        .data
                        .as_mut_ptr()
                        .add((i - 1 + (j + 1) * (*sin).xsize) as usize)
                        as i32
                        * (*mask)[0][2]
                    + *(*sin)
                        .data
                        .as_mut_ptr()
                        .add((i + 1 + j * (*sin).xsize) as usize) as i32
                        * (*mask)[1][0]
                    + *(*sin)
                        .data
                        .as_mut_ptr()
                        .add((i + j * (*sin).xsize) as usize) as i32
                        * (*mask)[1][1]
                    + *(*sin)
                        .data
                        .as_mut_ptr()
                        .add((i - 1 + j * (*sin).xsize) as usize) as i32
                        * (*mask)[1][2]
                    + *(*sin)
                        .data
                        .as_mut_ptr()
                        .add((i + 1 + (j - 1) * (*sin).xsize) as usize)
                        as i32
                        * (*mask)[2][0]
                    + *(*sin)
                        .data
                        .as_mut_ptr()
                        .add((i + (j - 1) * (*sin).xsize) as usize) as i32
                        * (*mask)[2][1]
                    + *(*sin)
                        .data
                        .as_mut_ptr()
                        .add((i - 1 + (j - 1) * (*sin).xsize) as usize)
                        as i32
                        * (*mask)[2][2];
                *(*sout)
                    .data
                    .as_mut_ptr()
                    .cast::<i16>()
                    .add((i + j * (*sout).xsize) as usize) = val as i16;
            }
        }
        for j in 1..jmax {
            *(*sout)
                .data
                .as_mut_ptr()
                .cast::<i16>()
                .add((j * (*sout).xsize) as usize) = *(*sout)
                .data
                .as_mut_ptr()
                .cast::<i16>()
                .add((1 + j * (*sout).xsize) as usize);
            *(*sout)
                .data
                .as_mut_ptr()
                .cast::<i16>()
                .add((imax + j * (*sout).xsize) as usize) = *(*sout)
                .data
                .as_mut_ptr()
                .cast::<i16>()
                .add((imax - 1 + j * (*sout).xsize) as usize);
        }
        for i in 0..=imax {
            *(*sout).data.as_mut_ptr().cast::<i16>().add(i as usize) = *(*sout)
                .data
                .as_mut_ptr()
                .cast::<i16>()
                .add((i + (*sout).xsize) as usize);
            *(*sout)
                .data
                .as_mut_ptr()
                .cast::<i16>()
                .add((i + jmax * (*sout).xsize) as usize) = *(*sout)
                .data
                .as_mut_ptr()
                .cast::<i16>()
                .add((i + (jmax - 1) * (*sout).xsize) as usize);
        }
        slice_scale_and_free(sout.as_mut(), sin);
        0
    }
}
/// Matches C `slice_mat_filter(Islice *, float *, int)` (`islice.c:547`).
///
/// The float path deliberately delegates to the corresponding complete C-unit
/// translation; non-float input retains the original get/multiply/put path.
pub unsafe fn slice_mat_filter(sin: *mut Islice, mat: *mut f32, dim: i32) -> Option<Box<Islice>> {
    const MAX_STATIC_KERNEL: i32 = 9;
    let mut sout = unsafe { slice_create((*sin).xsize, (*sin).ysize, MRC_MODE_FLOAT) }?;
    unsafe {
        if (*sin).mode == MRC_MODE_FLOAT {
            crate::imod::libcfshr::filtxcorr::apply_kernel_filter(
                core::slice::from_raw_parts(
                    (*sin).data.as_mut_ptr().cast::<f32>(),
                    ((*sin).xsize * (*sin).ysize) as usize,
                ),
                core::slice::from_raw_parts_mut(
                    sout.data.as_mut_ptr().cast::<f32>(),
                    (sout.xsize * sout.ysize) as usize,
                ),
                (*sin).xsize,
                (*sin).xsize,
                (*sin).ysize,
                core::slice::from_raw_parts(mat, (dim * dim) as usize),
                dim,
            );
        } else {
            let mut num_threads = 1;
            if dim <= MAX_STATIC_KERNEL {
                num_threads = crate::imod::libcfshr::b3dutil::num_omp_threads(
                    (0.04 * (((*sin).xsize * (*sin).ysize) as f64).sqrt()).round() as i32,
                );
            }
            if num_threads > 1 {
                for j in 0..(*sin).ysize {
                    for i in 0..(*sin).xsize {
                        let mut smat = [0.0f32; (MAX_STATIC_KERNEL * MAX_STATIC_KERNEL) as usize];
                        mrc_slice_mat_getimat(sin, i, j, dim, smat.as_mut_ptr());
                        let mut val = [0.0f32; 4];
                        val[0] = mrc_slice_mat_mult(mat, smat.as_ptr(), dim);
                        slice_put_val(sout.as_mut(), i, j, val);
                    }
                }
            } else {
                let mut imat = vec![0.0f32; (dim * dim) as usize];
                for j in 0..(*sin).ysize {
                    for i in 0..(*sin).xsize {
                        mrc_slice_mat_getimat(sin, i, j, dim, imat.as_mut_ptr());
                        let mut val = [0.0f32; 4];
                        val[0] = mrc_slice_mat_mult(mat, imat.as_ptr(), dim);
                        slice_put_val(sout.as_mut(), i, j, val);
                    }
                }
            }
        }
    }
    Some(sout)
}
pub unsafe fn mrc_slice_mat_getimat(sin: *mut Islice, x: i32, y: i32, dim: i32, mat: *mut f32) {
    unsafe {
        let xs = x - dim / 2;
        let xe = xs + dim;
        let ys = y - dim / 2;
        let ye = ys + dim;
        if xs >= 0 && xe < (*sin).xsize && ys >= 0 && ye < (*sin).ysize {
            for j in ys..ye {
                for i in xs..xe {
                    *mat.add((i - xs + dim * (j - ys)) as usize) =
                        slice_get_pixel_magnitude(sin, i, j);
                }
            }
        } else {
            for j in ys..ye {
                for i in xs..xe {
                    let ic = if i > (*sin).xsize - 1 {
                        (*sin).xsize - 1
                    } else if i < 0 {
                        0
                    } else {
                        i
                    };
                    let jc = if j > (*sin).ysize - 1 {
                        (*sin).ysize - 1
                    } else if j < 0 {
                        0
                    } else {
                        j
                    };
                    *mat.add((i - xs + dim * (j - ys)) as usize) =
                        slice_get_pixel_magnitude(sin, ic, jc);
                }
            }
        }
    }
}
pub unsafe fn mrc_slice_mat_mult(m1: *const f32, m2: *const f32, dim: i32) -> f32 {
    unsafe {
        let mut rval = 0.;
        let elements = dim * dim;
        for i in 0..elements {
            rval += *m1.add(i as usize) * *m2.add(i as usize);
        }
        rval
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_mode_table_matches_source() {
        assert_eq!(slice_mode_if_real(MRC_MODE_4BIT), 0);
        assert_eq!(slice_mode_if_real(MRC_MODE_HALF_FLOAT), 2);
        assert_eq!(slice_mode_if_real(16), -1);
    }

    #[test]
    fn string_mode_table_matches_source_case_and_aliases() {
        assert_eq!(slice_mode(b"byte"), 0);
        assert_eq!(slice_mode(b"sbyte"), -2);
        assert_eq!(slice_mode(b"complex"), 4);
        assert_eq!(slice_mode(b"16"), 16);
        assert_eq!(slice_mode(b"BYTE"), -1);
    }
    #[test]
    fn size_accessors_read_owned_slice_dimensions() {
        let mut slice = Islice {
            data: Vec::new(),
            xsize: 7,
            ysize: 9,
            mode: 0,
            csize: 0,
            dsize: 0,
            min: 0.,
            max: 0.,
            mean: 0.,
            index: 0,
            cval: [0.; 4],
        };
        unsafe {
            assert_eq!(slice_get_x_size(&mut slice), 7);
            assert_eq!(slice_get_y_size(&mut slice), 9);
        }
    }
    #[test]
    fn init_retains_source_fields_and_invalid_mode_result() {
        let mut s = slice_create(1, 1, 0).unwrap();
        unsafe {
            assert_eq!(slice_init(s.as_mut(), 4, 5, 1, vec![0; 40]), 0);
            assert_eq!(
                (s.xsize, s.ysize, s.mode, s.dsize, s.csize),
                (4, 5, 1, 2, 1)
            );
            assert_eq!(slice_init(s.as_mut(), 1, 1, 12, Vec::new()), -1);
        }
    }
    #[test]
    fn free_consumes_owned_slice() {
        slice_free(slice_create(2, 2, MRC_MODE_BYTE).unwrap());
    }
    #[test]
    fn create_uses_owned_allocation_and_mode_checks() {
        let s = slice_create(3, 2, MRC_MODE_SHORT).unwrap();
        assert_eq!(
            (s.xsize, s.ysize, s.dsize, s.csize, s.index),
            (3, 2, 2, 1, -1)
        );
        assert_eq!(s.data.len(), 12);
        assert!(slice_create(1, 1, 12).is_none());
    }
    #[test]
    fn value_magnitude_preserves_scalar_complex_and_rgb_rules() {
        assert_eq!(slice_get_val_magnitude([3., 4., 0., 0.], 4), 5.);
        assert!((slice_get_val_magnitude([100., 100., 100., 0.], 16) - 100.).abs() < 1e-5);
        assert_eq!(slice_get_val_magnitude([7., 0., 0., 0.], 1), 7.);
    }
    #[test]
    fn get_value_preserves_mode_layouts_and_bounds() {
        unsafe {
            let mut s = slice_create(3, 1, MRC_MODE_BYTE).unwrap();
            s.data.copy_from_slice(&[4, 5, 6]);
            s.mean = 9.;
            let mut v = [0.; 4];
            assert_eq!(slice_get_val(s.as_mut(), 1, 0, &mut v), 0);
            assert_eq!(v[0], 5.);
            assert_eq!(slice_get_val(s.as_mut(), 3, 0, &mut v), -1);
            assert_eq!(v[0], 9.);
            let mut s = slice_create(1, 1, MRC_MODE_COMPLEX_FLOAT).unwrap();
            *s.data.as_mut_ptr().cast::<f32>() = 3.;
            *s.data.as_mut_ptr().cast::<f32>().add(1) = 4.;
            s.xsize = 1;
            assert_eq!(slice_get_val(s.as_mut(), 0, 0, &mut v), 0);
            assert_eq!((v[0], v[1]), (3., 4.));
        }
    }
    #[test]
    fn put_clear_and_pixel_magnitude_preserve_source_data_and_channel_rules() {
        unsafe {
            let mut s = slice_create(2, 2, MRC_MODE_BYTE).unwrap();
            slice_clear(s.as_mut(), [7., 0., 0., 0.]);
            assert_eq!(s.min, 7.);
            assert_eq!(s.max, 7.);
            assert_eq!(s.mean, 7.);
            assert_eq!(slice_put_val(s.as_mut(), 1, 1, [9., 0., 0., 0.]), 0);
            assert_eq!(slice_get_pixel_magnitude(s.as_mut(), 1, 1), 9.);
            assert_eq!(slice_put_val(s.as_mut(), 2, 1, [1., 0., 0., 0.]), -1);

            let mut complex = slice_create(1, 1, MRC_MODE_COMPLEX_FLOAT).unwrap();
            assert_eq!(slice_put_val(complex.as_mut(), 0, 0, [3., 4., 0., 0.]), 0);
            assert_eq!(slice_get_pixel_magnitude(complex.as_mut(), 0, 0), 5.);

            let mut rgb = slice_create(1, 1, MRC_MODE_RGB).unwrap();
            assert_eq!(slice_put_val(rgb.as_mut(), 0, 0, [100., 100., 100., 0.]), 0);
            assert!((slice_get_pixel_magnitude(rgb.as_mut(), 0, 0) - 100.).abs() < 1e-5);
        }
    }
    #[test]
    fn min_max_preserves_each_supported_source_mode_and_rejects_other_modes() {
        unsafe {
            let mut slice = slice_create(3, 1, MRC_MODE_BYTE).unwrap();
            slice.data.copy_from_slice(&[8, 2, 7]);
            assert_eq!(slice_min_max(slice.as_mut()), 0);
            assert_eq!((slice.min, slice.max), (2., 8.));

            let mut slice = slice_create(3, 1, MRC_MODE_SHORT).unwrap();
            let data = slice.data.as_mut_ptr().cast::<i16>();
            *data = -7;
            *data.add(1) = 12;
            *data.add(2) = 3;
            assert_eq!(slice_min_max(slice.as_mut()), 0);
            assert_eq!((slice.min, slice.max), (-7., 12.));

            let mut slice = slice_create(3, 1, MRC_MODE_USHORT).unwrap();
            let data = slice.data.as_mut_ptr().cast::<u16>();
            *data = 9;
            *data.add(1) = 14;
            *data.add(2) = 4;
            assert_eq!(slice_min_max(slice.as_mut()), 0);
            assert_eq!((slice.min, slice.max), (4., 14.));

            let mut slice = slice_create(3, 1, MRC_MODE_FLOAT).unwrap();
            let data = slice.data.as_mut_ptr().cast::<f32>();
            *data = -1.5;
            *data.add(1) = 4.25;
            *data.add(2) = 0.;
            assert_eq!(slice_min_max(slice.as_mut()), 0);
            assert_eq!((slice.min, slice.max), (-1.5, 4.25));

            slice.mode = 4;
            assert_eq!(slice_min_max(slice.as_mut()), 1);
        }
    }
    #[test]
    fn scale_and_free_preserves_source_scaling_gate_and_data_modes() {
        unsafe {
            let mut sin = slice_create(2, 1, MRC_MODE_BYTE).unwrap();
            sin.min = 10.;
            sin.max = 110.;
            let mut sout = slice_create(2, 1, MRC_MODE_FLOAT).unwrap();
            *sout.data.as_mut_ptr().cast::<f32>() = 0.;
            *sout.data.as_mut_ptr().cast::<f32>().add(1) = 10.;
            slice_scale_and_free(sout.as_mut(), sin.as_mut());
            assert_eq!(sin.data, [10, 110]);

            let mut sout = slice_create(2, 1, MRC_MODE_SHORT).unwrap();
            *sout.data.as_mut_ptr().cast::<i16>() = 2;
            *sout.data.as_mut_ptr().cast::<i16>().add(1) = 3;
            sin.min = 0.;
            sin.max = 0.;
            slice_scale_and_free(sout.as_mut(), sin.as_mut());
            assert_eq!(sin.data, [2, 3]);
        }
    }
    #[test]
    fn byte_convolution_and_fixed_kernel_wrappers_preserve_source_handoff() {
        unsafe {
            let mut sin = slice_create(3, 3, MRC_MODE_BYTE).unwrap();
            sin.data.fill(10);
            assert_eq!(
                slice_byte_convolve(sin.as_mut(), &[[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                0
            );
            assert_eq!(sin.data, [10; 9]);

            sin.data.fill(10);
            assert_eq!(slice_byte_edge_laplacian(sin.as_mut()), 0);
            assert_eq!(sin.data, [40; 9]);
            sin.data.fill(10);
            assert_eq!(slice_byte_sharpen(sin.as_mut()), 0);
            assert_eq!(sin.data, [10; 9]);
            sin.data.fill(10);
            assert_eq!(slice_byte_smooth(sin.as_mut()), 0);
            assert_eq!(sin.data, [160; 9]);
        }
    }
    #[test]
    fn matrix_primitives_preserve_dot_product_and_edge_replication() {
        unsafe {
            let m1 = [1., 2., 3., 4.];
            let m2 = [4., 3., 2., 1.];
            assert_eq!(mrc_slice_mat_mult(m1.as_ptr(), m2.as_ptr(), 2), 20.);

            let mut sin = slice_create(3, 3, MRC_MODE_BYTE).unwrap();
            sin.data.copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
            let mut mat = [0.; 9];
            mrc_slice_mat_getimat(sin.as_mut(), 0, 0, 3, mat.as_mut_ptr());
            assert_eq!(mat, [1., 1., 2., 1., 1., 2., 4., 4., 5.]);
        }
    }
}
