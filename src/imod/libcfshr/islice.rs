//! Translation of `IMOD/libcfshr/islice.c` and its direct `mrcslice.h` layouts.
#![allow(dead_code)]
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_4BIT, MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_HALF_FLOAT, MRC_MODE_SHORT,
    MRC_MODE_USHORT,
};
const SMOOTH_KERNEL: [[i32; 3]; 3] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
const SHARPEN_KERNEL: [[i32; 3]; 3] = [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]];
const LAPLACIAN_KERNEL: [[i32; 3]; 3] = [[1, 1, 1], [1, -4, 1], [1, 1, 1]];
#[repr(C)]
pub union MrcData {
    pub b: *mut u8,
    pub s: *mut i16,
    pub us: *mut u16,
    pub f: *mut f32,
}
#[repr(C)]
pub struct Islice {
    pub data: MrcData,
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
/// C layout of `Istack` / `MRCvolume` from `mrcslice.h`.
#[repr(C)]
pub struct Istack {
    pub vol: *mut *mut Islice,
    pub zsize: i32,
}
pub unsafe fn slice_create(xsize: i32, ysize: i32, mode: i32) -> *mut Islice {
    let xysize = (xsize as usize).wrapping_mul(ysize as usize);
    if xysize / xsize as usize != ysize as usize {
        return core::ptr::null_mut();
    }
    let s = unsafe { libc::malloc(core::mem::size_of::<Islice>()).cast::<Islice>() };
    if s.is_null() {
        return s;
    }
    unsafe {
        if crate::imod::libcfshr::b3dutil::data_size_for_mode(
            mode,
            &mut (*s).dsize,
            &mut (*s).csize,
        ) != 0
        {
            libc::free(s.cast());
            return core::ptr::null_mut();
        }
        (*s).xsize = xsize;
        (*s).ysize = ysize;
        (*s).mode = mode;
        (*s).index = -1;
        (*s).data.b = libc::malloc(xysize * (*s).dsize as usize * (*s).csize as usize).cast();
        if (*s).data.b.is_null() {
            libc::free(s.cast());
            return core::ptr::null_mut();
        }
    }
    s
}
pub unsafe fn slice_init(
    s: *mut Islice,
    xsize: i32,
    ysize: i32,
    mode: i32,
    data: *mut core::ffi::c_void,
) -> i32 {
    unsafe {
        (*s).xsize = xsize;
        (*s).ysize = ysize;
        (*s).mode = mode;
        (*s).data.b = data.cast();
    }
    if crate::imod::libcfshr::b3dutil::data_size_for_mode(
        mode,
        unsafe { &mut (*s).dsize },
        unsafe { &mut (*s).csize },
    ) != 0
    {
        return -1;
    }
    0
}
pub unsafe fn slice_free(s: *mut Islice) {
    unsafe {
        if s.is_null() {
            return;
        }
        if !(*s).data.b.is_null() {
            libc::free((*s).data.b.cast());
        }
        libc::free(s.cast());
    }
}
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
pub unsafe fn slice_mode(mst: *const core::ffi::c_char) -> i32 {
    unsafe {
        let value = core::ffi::CStr::from_ptr(mst).to_bytes();
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
            0 => (*val)[0] = *(*s).data.b.add(index) as f32,
            1 => (*val)[0] = *(*s).data.s.add(index) as f32,
            6 => (*val)[0] = *(*s).data.us.add(index) as f32,
            2 => (*val)[0] = *(*s).data.f.add(index),
            3 => {
                index *= 2;
                (*val)[0] = *(*s).data.s.add(index) as f32;
                (*val)[1] = *(*s).data.s.add(index + 1) as f32;
            }
            4 => {
                index *= 2;
                (*val)[0] = *(*s).data.f.add(index);
                (*val)[1] = *(*s).data.f.add(index + 1);
            }
            16 => {
                index *= 3;
                (*val)[0] = *(*s).data.b.add(index) as f32;
                (*val)[1] = *(*s).data.b.add(index + 1) as f32;
                (*val)[2] = *(*s).data.b.add(index + 2) as f32;
            }
            99 => {
                index *= 3;
                (*val)[0] = *(*s).data.f.add(index);
                (*val)[1] = *(*s).data.f.add(index + 1);
                (*val)[2] = *(*s).data.f.add(index + 2);
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
            0 => *(*s).data.b.add(i) = val[0] as i32 as u8,
            1 => *(*s).data.s.add(i) = val[0] as i32 as i16,
            6 => *(*s).data.us.add(i) = val[0] as i32 as u16,
            2 => *(*s).data.f.add(i) = val[0],
            3 => {
                i *= 2;
                *(*s).data.s.add(i) = val[0] as i32 as i16;
                *(*s).data.s.add(i + 1) = val[1] as i32 as i16
            }
            4 => {
                i *= 2;
                *(*s).data.f.add(i) = val[0];
                *(*s).data.f.add(i + 1) = val[1]
            }
            16 => {
                i *= 3;
                *(*s).data.b.add(i) = val[0] as i32 as u8;
                *(*s).data.b.add(i + 1) = val[1] as i32 as u8;
                *(*s).data.b.add(i + 2) = val[2] as i32 as u8
            }
            99 => {
                i *= 3;
                *(*s).data.f.add(i) = val[0];
                *(*s).data.f.add(i + 1) = val[1];
                *(*s).data.f.add(i + 2) = val[2]
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
                imin = *(*s).data.b as i32;
                imax = *(*s).data.b as i32;
                for i in 1..((*s).xsize * (*s).ysize) {
                    ival = *(*s).data.b.add(i as usize) as i32;
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
                imin = *(*s).data.s as i32;
                imax = *(*s).data.s as i32;
                for i in 1..((*s).xsize * (*s).ysize) {
                    ival = *(*s).data.s.add(i as usize) as i32;
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
                imin = *(*s).data.us as i32;
                imax = *(*s).data.us as i32;
                for i in 1..((*s).xsize * (*s).ysize) {
                    ival = *(*s).data.us.add(i as usize) as i32;
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
                fmin = *(*s).data.f;
                fmax = *(*s).data.f;
                for i in 1..((*s).xsize * (*s).ysize) {
                    fval = *(*s).data.f.add(i as usize);
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
                    *(*sin).data.b.add(i as usize) =
                        (*(*sout).data.f.add(i as usize) * mval + aval) as i32 as u8;
                }
            }
            1 => {
                for i in 0..imax {
                    *(*sin).data.b.add(i as usize) =
                        (*(*sout).data.s.add(i as usize) as f32 * mval + aval) as i32 as u8;
                }
            }
            6 => {
                for i in 0..imax {
                    *(*sin).data.b.add(i as usize) =
                        (*(*sout).data.us.add(i as usize) as f32 * mval + aval) as i32 as u8;
                }
            }
            0 => {
                for i in 0..imax {
                    *(*sin).data.b.add(i as usize) =
                        (*(*sout).data.b.add(i as usize) as f32 * mval + aval) as i32 as u8;
                }
            }
            _ => {}
        }
        slice_free(sout);
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
        let sout = slice_create((*sin).xsize, (*sin).ysize, 1);
        let imax = (*sin).xsize - 1;
        let jmax = (*sin).ysize - 1;
        for i in 1..imax {
            for j in 1..jmax {
                let val = *(*sin).data.b.add((i + 1 + (j + 1) * (*sin).xsize) as usize) as i32
                    * (*mask)[0][0]
                    + *(*sin).data.b.add((i + (j + 1) * (*sin).xsize) as usize) as i32
                        * (*mask)[0][1]
                    + *(*sin).data.b.add((i - 1 + (j + 1) * (*sin).xsize) as usize) as i32
                        * (*mask)[0][2]
                    + *(*sin).data.b.add((i + 1 + j * (*sin).xsize) as usize) as i32
                        * (*mask)[1][0]
                    + *(*sin).data.b.add((i + j * (*sin).xsize) as usize) as i32 * (*mask)[1][1]
                    + *(*sin).data.b.add((i - 1 + j * (*sin).xsize) as usize) as i32
                        * (*mask)[1][2]
                    + *(*sin).data.b.add((i + 1 + (j - 1) * (*sin).xsize) as usize) as i32
                        * (*mask)[2][0]
                    + *(*sin).data.b.add((i + (j - 1) * (*sin).xsize) as usize) as i32
                        * (*mask)[2][1]
                    + *(*sin).data.b.add((i - 1 + (j - 1) * (*sin).xsize) as usize) as i32
                        * (*mask)[2][2];
                *(*sout).data.s.add((i + j * (*sout).xsize) as usize) = val as i16;
            }
        }
        for j in 1..jmax {
            *(*sout).data.s.add((j * (*sout).xsize) as usize) =
                *(*sout).data.s.add((1 + j * (*sout).xsize) as usize);
            *(*sout).data.s.add((imax + j * (*sout).xsize) as usize) =
                *(*sout).data.s.add((imax - 1 + j * (*sout).xsize) as usize);
        }
        for i in 0..=imax {
            *(*sout).data.s.add(i as usize) = *(*sout).data.s.add((i + (*sout).xsize) as usize);
            *(*sout).data.s.add((i + jmax * (*sout).xsize) as usize) = *(*sout)
                .data
                .s
                .add((i + (jmax - 1) * (*sout).xsize) as usize);
        }
        slice_scale_and_free(sout, sin);
        0
    }
}
/// Matches C `slice_mat_filter(Islice *, float *, int)` (`islice.c:547`).
///
/// The float path deliberately delegates to the corresponding complete C-unit
/// translation; non-float input retains the original get/multiply/put path.
pub unsafe fn slice_mat_filter(sin: *mut Islice, mat: *mut f32, dim: i32) -> *mut Islice {
    const MAX_STATIC_KERNEL: i32 = 9;
    let sout = unsafe { slice_create((*sin).xsize, (*sin).ysize, MRC_MODE_FLOAT) };
    if sout.is_null() {
        return core::ptr::null_mut();
    }
    unsafe {
        if (*sin).mode == MRC_MODE_FLOAT {
            crate::imod::libcfshr::filtxcorr::apply_kernel_filter(
                core::slice::from_raw_parts((*sin).data.f, ((*sin).xsize * (*sin).ysize) as usize),
                core::slice::from_raw_parts_mut(
                    (*sout).data.f,
                    ((*sout).xsize * (*sout).ysize) as usize,
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
                        slice_put_val(sout, i, j, val);
                    }
                }
            } else {
                let imat =
                    libc::malloc((dim * dim) as usize * core::mem::size_of::<f32>()).cast::<f32>();
                if imat.is_null() {
                    return core::ptr::null_mut();
                }
                for j in 0..(*sin).ysize {
                    for i in 0..(*sin).xsize {
                        mrc_slice_mat_getimat(sin, i, j, dim, imat);
                        let mut val = [0.0f32; 4];
                        val[0] = mrc_slice_mat_mult(mat, imat, dim);
                        slice_put_val(sout, i, j, val);
                    }
                }
                libc::free(imat.cast());
            }
        }
    }
    sout
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
        unsafe {
            assert_eq!(slice_mode(c"byte".as_ptr()), 0);
            assert_eq!(slice_mode(c"sbyte".as_ptr()), -2);
            assert_eq!(slice_mode(c"complex".as_ptr()), 4);
            assert_eq!(slice_mode(c"16".as_ptr()), 16);
            assert_eq!(slice_mode(c"BYTE".as_ptr()), -1);
        }
    }
    #[test]
    fn size_accessors_retain_source_raw_layout() {
        let mut slice = Islice {
            data: MrcData {
                b: core::ptr::null_mut(),
            },
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
        let mut s: Islice = unsafe { core::mem::zeroed() };
        let mut bytes = [0u8; 2];
        unsafe {
            assert_eq!(slice_init(&mut s, 4, 5, 1, bytes.as_mut_ptr().cast()), 0);
            assert_eq!(
                (s.xsize, s.ysize, s.mode, s.dsize, s.csize),
                (4, 5, 1, 2, 1)
            );
            assert_eq!(slice_init(&mut s, 1, 1, 12, core::ptr::null_mut()), -1);
        }
    }
    #[test]
    fn free_handles_null_and_source_owned_allocations() {
        unsafe {
            slice_free(core::ptr::null_mut());
            let s = libc::calloc(1, core::mem::size_of::<Islice>()).cast::<Islice>();
            assert!(!s.is_null());
            (*s).data.b = libc::malloc(8).cast();
            slice_free(s);
        }
    }
    #[test]
    fn create_uses_source_allocation_and_mode_checks() {
        unsafe {
            let s = slice_create(3, 2, 1);
            assert!(!s.is_null());
            assert_eq!(
                ((*s).xsize, (*s).ysize, (*s).dsize, (*s).csize, (*s).index),
                (3, 2, 2, 1, -1)
            );
            slice_free(s);
            assert!(slice_create(1, 1, 12).is_null());
        }
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
            let mut b = [4u8, 5, 6];
            let mut s: Islice = core::mem::zeroed();
            s.data.b = b.as_mut_ptr();
            s.xsize = 3;
            s.ysize = 1;
            s.mode = 0;
            s.mean = 9.;
            let mut v = [0.; 4];
            assert_eq!(slice_get_val(&mut s, 1, 0, &mut v), 0);
            assert_eq!(v[0], 5.);
            assert_eq!(slice_get_val(&mut s, 3, 0, &mut v), -1);
            assert_eq!(v[0], 9.);
            let mut c = [3f32, 4.];
            s.data.f = c.as_mut_ptr();
            s.xsize = 1;
            s.mode = 4;
            assert_eq!(slice_get_val(&mut s, 0, 0, &mut v), 0);
            assert_eq!((v[0], v[1]), (3., 4.));
        }
    }
    #[test]
    fn put_clear_and_pixel_magnitude_preserve_source_data_and_channel_rules() {
        unsafe {
            let s = slice_create(2, 2, 0);
            assert!(!s.is_null());
            slice_clear(s, [7., 0., 0., 0.]);
            assert_eq!((*s).min, 7.);
            assert_eq!((*s).max, 7.);
            assert_eq!((*s).mean, 7.);
            assert_eq!(slice_put_val(s, 1, 1, [9., 0., 0., 0.]), 0);
            assert_eq!(slice_get_pixel_magnitude(s, 1, 1), 9.);
            assert_eq!(slice_put_val(s, 2, 1, [1., 0., 0., 0.]), -1);
            slice_free(s);

            let complex = slice_create(1, 1, 4);
            assert!(!complex.is_null());
            assert_eq!(slice_put_val(complex, 0, 0, [3., 4., 0., 0.]), 0);
            assert_eq!(slice_get_pixel_magnitude(complex, 0, 0), 5.);
            slice_free(complex);

            let rgb = slice_create(1, 1, 16);
            assert!(!rgb.is_null());
            assert_eq!(slice_put_val(rgb, 0, 0, [100., 100., 100., 0.]), 0);
            assert!((slice_get_pixel_magnitude(rgb, 0, 0) - 100.).abs() < 1e-5);
            slice_free(rgb);
        }
    }
    #[test]
    fn min_max_preserves_each_supported_source_mode_and_rejects_other_modes() {
        unsafe {
            let mut bytes = [8u8, 2, 7];
            let mut slice: Islice = core::mem::zeroed();
            slice.data.b = bytes.as_mut_ptr();
            slice.xsize = 3;
            slice.ysize = 1;
            slice.mode = 0;
            assert_eq!(slice_min_max(&mut slice), 0);
            assert_eq!((slice.min, slice.max), (2., 8.));

            let mut shorts = [-7i16, 12, 3];
            slice.data.s = shorts.as_mut_ptr();
            slice.mode = 1;
            assert_eq!(slice_min_max(&mut slice), 0);
            assert_eq!((slice.min, slice.max), (-7., 12.));

            let mut ushorts = [9u16, 14, 4];
            slice.data.us = ushorts.as_mut_ptr();
            slice.mode = 6;
            assert_eq!(slice_min_max(&mut slice), 0);
            assert_eq!((slice.min, slice.max), (4., 14.));

            let mut floats = [-1.5f32, 4.25, 0.];
            slice.data.f = floats.as_mut_ptr();
            slice.mode = 2;
            assert_eq!(slice_min_max(&mut slice), 0);
            assert_eq!((slice.min, slice.max), (-1.5, 4.25));

            slice.mode = 4;
            assert_eq!(slice_min_max(&mut slice), 1);
        }
    }
    #[test]
    fn scale_and_free_preserves_source_scaling_gate_and_data_modes() {
        unsafe {
            let mut bytes = [0u8; 2];
            let mut sin: Islice = core::mem::zeroed();
            sin.data.b = bytes.as_mut_ptr();
            sin.xsize = 2;
            sin.ysize = 1;
            sin.min = 10.;
            sin.max = 110.;
            let sout = slice_create(2, 1, 2);
            assert!(!sout.is_null());
            *(*sout).data.f = 0.;
            *(*sout).data.f.add(1) = 10.;
            slice_scale_and_free(sout, &mut sin);
            assert_eq!(bytes, [10, 110]);

            let sout = slice_create(2, 1, 1);
            assert!(!sout.is_null());
            *(*sout).data.s = 2;
            *(*sout).data.s.add(1) = 3;
            sin.min = 0.;
            sin.max = 0.;
            slice_scale_and_free(sout, &mut sin);
            assert_eq!(bytes, [2, 3]);
        }
    }
    #[test]
    fn byte_convolution_and_fixed_kernel_wrappers_preserve_source_handoff() {
        unsafe {
            let mut bytes = [10u8; 9];
            let mut sin: Islice = core::mem::zeroed();
            sin.data.b = bytes.as_mut_ptr();
            sin.xsize = 3;
            sin.ysize = 3;
            assert_eq!(
                slice_byte_convolve(&mut sin, &[[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                0
            );
            assert_eq!(bytes, [10; 9]);

            bytes = [10; 9];
            assert_eq!(slice_byte_edge_laplacian(&mut sin), 0);
            assert_eq!(bytes, [40; 9]);
            bytes = [10; 9];
            assert_eq!(slice_byte_sharpen(&mut sin), 0);
            assert_eq!(bytes, [10; 9]);
            bytes = [10; 9];
            assert_eq!(slice_byte_smooth(&mut sin), 0);
            assert_eq!(bytes, [160; 9]);
        }
    }
    #[test]
    fn matrix_primitives_preserve_dot_product_and_edge_replication() {
        unsafe {
            let m1 = [1., 2., 3., 4.];
            let m2 = [4., 3., 2., 1.];
            assert_eq!(mrc_slice_mat_mult(m1.as_ptr(), m2.as_ptr(), 2), 20.);

            let mut bytes = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
            let mut sin: Islice = core::mem::zeroed();
            sin.data.b = bytes.as_mut_ptr();
            sin.xsize = 3;
            sin.ysize = 3;
            sin.mode = 0;
            sin.csize = 1;
            let mut mat = [0.; 9];
            mrc_slice_mat_getimat(&mut sin, 0, 0, 3, mat.as_mut_ptr());
            assert_eq!(mat, [1., 1., 2., 1., 1., 2., 4., 4., 5.]);
        }
    }
}
