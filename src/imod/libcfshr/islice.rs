//! Translation of `IMOD/libcfshr/islice.c` and its direct `mrcslice.h` layouts.
#![allow(dead_code)]
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_4BIT, MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, MRC_MODE_HALF_FLOAT,
    MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT,
};
const SMOOTH_KERNEL: [[i32; 3]; 3] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
const SHARPEN_KERNEL: [[i32; 3]; 3] = [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]];
const LAPLACIAN_KERNEL: [[i32; 3]; 3] = [[1, 1, 1], [1, -4, 1], [1, 1, 1]];
/// Complete function inventory for `islice.c`.
pub const ISLICE_SOURCE_FUNCTIONS: &[&str] = &[
    "sliceCreate",
    "sliceInit",
    "sliceFree",
    "sliceClear",
    "sliceMode",
    "sliceModeIfReal",
    "sliceGetXSize",
    "sliceGetYSize",
    "sliceGetVal",
    "slicePutVal",
    "sliceGetPixelMagnitude",
    "sliceGetValMagnitude",
    "sliceMinMax",
    "sliceScaleAndFree",
    "sliceByteEdgeLaplacian",
    "sliceByteSharpen",
    "sliceByteSmooth",
    "sliceByteConvolve",
    "slice_mat_filter",
    "mrc_slice_mat_getimat",
    "mrc_slice_mat_mult",
];
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

impl AsRef<Islice> for Islice {
    fn as_ref(&self) -> &Islice {
        self
    }
}

impl AsMut<Islice> for Islice {
    fn as_mut(&mut self) -> &mut Islice {
        self
    }
}
pub struct Istack {
    /// Consecutive owned slices; C used an array of independently allocated
    /// pointers, but no translated caller needs their addresses to be stable.
    pub slices: Vec<Islice>,
}
pub fn slice_create(xsize: i32, ysize: i32, mode: i32) -> Option<Islice> {
    let xysize = (xsize as usize).wrapping_mul(ysize as usize);
    if xysize / xsize as usize != ysize as usize {
        return None;
    }
    let mut dsize = 0;
    let mut csize = 0;
    if crate::imod::libcfshr::b3dutil::data_size_for_mode(mode, &mut dsize, &mut csize) != 0 {
        return None;
    }
    Some(Islice {
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
    })
}
pub fn slice_init(s: &mut Islice, xsize: i32, ysize: i32, mode: i32, data: Vec<u8>) -> i32 {
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
    s.xsize = xsize;
    s.ysize = ysize;
    s.mode = mode;
    s.dsize = dsize;
    s.csize = csize;
    s.data = data;
    0
}
pub fn slice_free(_s: Islice) {}
pub fn slice_clear(s: &mut Islice, val: [f32; 4]) {
    s.min = slice_get_val_magnitude(val, s.mode);
    s.max = s.min;
    s.mean = s.min;
    for j in 0..s.ysize {
        for i in 0..s.xsize {
            slice_put_val(s, i, j, val);
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
pub fn slice_get_x_size(slice: &Islice) -> i32 {
    slice.xsize
}
pub fn slice_get_y_size(slice: &Islice) -> i32 {
    slice.ysize
}
pub fn slice_get_val(s: &Islice, x: i32, y: i32, val: &mut [f32; 4]) -> i32 {
    if x < 0 || y < 0 || x >= s.xsize || y >= s.ysize {
        val[0] = s.mean;
        return -1;
    }
    let index = x as usize + y as usize * s.xsize as usize;
    val[1] = 0.;
    match s.mode {
        0 => val[0] = s.data[index] as f32,
        1 => {
            val[0] = i16::from_ne_bytes(s.data[index * 2..index * 2 + 2].try_into().unwrap()) as f32
        }
        6 => {
            val[0] = u16::from_ne_bytes(s.data[index * 2..index * 2 + 2].try_into().unwrap()) as f32
        }
        2 => val[0] = f32::from_ne_bytes(s.data[index * 4..index * 4 + 4].try_into().unwrap()),
        3 => {
            let index = index * 4;
            val[0] = i16::from_ne_bytes(s.data[index..index + 2].try_into().unwrap()) as f32;
            val[1] = i16::from_ne_bytes(s.data[index + 2..index + 4].try_into().unwrap()) as f32;
        }
        4 => {
            let index = index * 8;
            val[0] = f32::from_ne_bytes(s.data[index..index + 4].try_into().unwrap());
            val[1] = f32::from_ne_bytes(s.data[index + 4..index + 8].try_into().unwrap());
        }
        16 => {
            let index = index * 3;
            val[0] = s.data[index] as f32;
            val[1] = s.data[index + 1] as f32;
            val[2] = s.data[index + 2] as f32;
        }
        99 => {
            let index = index * 12;
            val[0] = f32::from_ne_bytes(s.data[index..index + 4].try_into().unwrap());
            val[1] = f32::from_ne_bytes(s.data[index + 4..index + 8].try_into().unwrap());
            val[2] = f32::from_ne_bytes(s.data[index + 8..index + 12].try_into().unwrap());
        }
        _ => return -1,
    }
    0
}
pub fn slice_put_val(s: &mut Islice, x: i32, y: i32, val: [f32; 4]) -> i32 {
    if x < 0 || y < 0 || x >= s.xsize || y >= s.ysize {
        return -1;
    }
    let i = x as usize + y as usize * s.xsize as usize;
    // `islice.c:270-285` casts the float straight to the narrow integer
    // type.  In C that truncates toward zero into an int and then keeps the
    // low bits, so -1.0f stores 255 and 300.0f stores 44.  Rust's float
    // `as u8` SATURATES instead (0 and 255), which silently rewrites pixel
    // data: `clip multiply` of a byte file by a float file wrote 15300 of
    // 15360 pixels as 127.  Going through `as i32` first reproduces the C
    // for every value in i32 range; outside it the C is undefined anyway.
    match s.mode {
        0 => s.data[i] = val[0] as i32 as u8,
        1 => s.data[i * 2..i * 2 + 2].copy_from_slice(&(val[0] as i32 as i16).to_ne_bytes()),
        6 => s.data[i * 2..i * 2 + 2].copy_from_slice(&(val[0] as i32 as u16).to_ne_bytes()),
        2 => s.data[i * 4..i * 4 + 4].copy_from_slice(&val[0].to_ne_bytes()),
        3 => {
            let i = i * 4;
            s.data[i..i + 2].copy_from_slice(&(val[0] as i32 as i16).to_ne_bytes());
            s.data[i + 2..i + 4].copy_from_slice(&(val[1] as i32 as i16).to_ne_bytes());
        }
        4 => {
            let i = i * 8;
            s.data[i..i + 4].copy_from_slice(&val[0].to_ne_bytes());
            s.data[i + 4..i + 8].copy_from_slice(&val[1].to_ne_bytes());
        }
        16 => {
            let i = i * 3;
            s.data[i] = val[0] as i32 as u8;
            s.data[i + 1] = val[1] as i32 as u8;
            s.data[i + 2] = val[2] as i32 as u8;
        }
        99 => {
            let i = i * 12;
            s.data[i..i + 4].copy_from_slice(&val[0].to_ne_bytes());
            s.data[i + 4..i + 8].copy_from_slice(&val[1].to_ne_bytes());
            s.data[i + 8..i + 12].copy_from_slice(&val[2].to_ne_bytes());
        }
        _ => return -1,
    }
    0
}
pub fn slice_get_pixel_magnitude(s: &Islice, x: i32, y: i32) -> f32 {
    let mut val = [0.; 4];
    slice_get_val(s, x, y, &mut val);
    let csize = s.csize;
    if csize == 1 {
        return val[0];
    }
    if csize == 2 {
        return (val[0] * val[0] + val[1] * val[1]).sqrt();
    }
    val[0] * 0.3 + val[1] * 0.59 + val[2] * 0.11
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
pub fn slice_min_max(s: &mut Islice) -> i32 {
    match s.mode {
        0 => {
            let (min, max) = s
                .data
                .iter()
                .map(|&value| value as f32)
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
                    (min.min(value), max.max(value))
                });
            s.min = min;
            s.max = max;
        }
        1 => {
            let (min, max) = s
                .data
                .chunks_exact(2)
                .map(|bytes| i16::from_ne_bytes(bytes.try_into().unwrap()) as f32)
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
                    (min.min(value), max.max(value))
                });
            s.min = min;
            s.max = max;
        }
        6 => {
            let (min, max) = s
                .data
                .chunks_exact(2)
                .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()) as f32)
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
                    (min.min(value), max.max(value))
                });
            s.min = min;
            s.max = max;
        }
        2 => {
            let (min, max) = s
                .data
                .chunks_exact(4)
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
                    (min.min(value), max.max(value))
                });
            s.min = min;
            s.max = max;
        }
        _ => return 1,
    }
    0
}
pub fn slice_scale_and_free(sout: &mut Islice, sin: &mut Islice) {
    let mut aval = 0.;
    let mut mval = 1.;
    if sin.min != 0. || sin.max != 0. {
        slice_min_max(sout);
        mval = (sin.max - sin.min) / (sout.max - sout.min);
        aval = sin.min - mval * sout.min;
    }
    let imax = (sin.xsize * sin.ysize) as usize;
    // `islice.c:448-460` narrows with `(unsigned char)`, which truncates
    // toward zero and keeps the low bits rather than saturating.
    match sout.mode {
        2 => {
            for i in 0..imax {
                sin.data[i] = (f32::from_ne_bytes(sout.data[i * 4..i * 4 + 4].try_into().unwrap())
                    * mval
                    + aval) as i32 as u8;
            }
        }
        1 => {
            for i in 0..imax {
                sin.data[i] = (i16::from_ne_bytes(sout.data[i * 2..i * 2 + 2].try_into().unwrap())
                    as f32
                    * mval
                    + aval) as i32 as u8;
            }
        }
        6 => {
            for i in 0..imax {
                sin.data[i] = (u16::from_ne_bytes(sout.data[i * 2..i * 2 + 2].try_into().unwrap())
                    as f32
                    * mval
                    + aval) as i32 as u8;
            }
        }
        0 => {
            for i in 0..imax {
                sin.data[i] = (sout.data[i] as f32 * mval + aval) as i32 as u8;
            }
        }
        _ => {}
    }
}
pub fn slice_byte_edge_laplacian(sin: &mut Islice) -> i32 {
    slice_byte_convolve(sin, &LAPLACIAN_KERNEL)
}
pub fn slice_byte_sharpen(sin: &mut Islice) -> i32 {
    slice_byte_convolve(sin, &SHARPEN_KERNEL)
}
pub fn slice_byte_smooth(sin: &mut Islice) -> i32 {
    slice_byte_convolve(sin, &SMOOTH_KERNEL)
}
pub fn slice_byte_convolve(sin: &mut Islice, mask: &[[i32; 3]; 3]) -> i32 {
    let Some(mut sout) = slice_create(sin.xsize, sin.ysize, 1) else {
        return -1;
    };
    let imax = sin.xsize - 1;
    let jmax = sin.ysize - 1;
    if sin.xsize < 2 || sin.ysize < 2 {
        return -1;
    }
    let index = |x: i32, y: i32| (x + y * sin.xsize) as usize;
    for i in 1..imax {
        for j in 1..jmax {
            let val = sin.data[index(i + 1, j + 1)] as i32 * mask[0][0]
                + sin.data[index(i, j + 1)] as i32 * mask[0][1]
                + sin.data[index(i - 1, j + 1)] as i32 * mask[0][2]
                + sin.data[index(i + 1, j)] as i32 * mask[1][0]
                + sin.data[index(i, j)] as i32 * mask[1][1]
                + sin.data[index(i - 1, j)] as i32 * mask[1][2]
                + sin.data[index(i + 1, j - 1)] as i32 * mask[2][0]
                + sin.data[index(i, j - 1)] as i32 * mask[2][1]
                + sin.data[index(i - 1, j - 1)] as i32 * mask[2][2];
            let output = index(i, j) * 2;
            sout.data[output..output + 2].copy_from_slice(&(val as i16).to_ne_bytes());
        }
    }
    for j in 1..jmax {
        let left = index(0, j) * 2;
        let left_source = index(1, j) * 2;
        sout.data.copy_within(left_source..left_source + 2, left);
        let right = index(imax, j) * 2;
        let right_source = index(imax - 1, j) * 2;
        sout.data.copy_within(right_source..right_source + 2, right);
    }
    for i in 0..=imax {
        let top = index(i, 0) * 2;
        let top_source = index(i, 1) * 2;
        sout.data.copy_within(top_source..top_source + 2, top);
        let bottom = index(i, jmax) * 2;
        let bottom_source = index(i, jmax - 1) * 2;
        sout.data
            .copy_within(bottom_source..bottom_source + 2, bottom);
    }
    slice_scale_and_free(sout.as_mut(), sin);
    0
}
/// Matches C `slice_mat_filter(Islice *, float *, int)` (`islice.c:547`).
///
/// The float path deliberately delegates to the corresponding complete C-unit
/// translation; non-float input retains the original get/multiply/put path.
pub fn slice_mat_filter(sin: &Islice, mat: &[f32], dim: i32) -> Option<Islice> {
    const MAX_STATIC_KERNEL: i32 = 9;
    let dim = usize::try_from(dim).ok()?;
    let matrix_elements = dim.checked_mul(dim)?;
    if mat.len() < matrix_elements {
        return None;
    }
    let mut sout = slice_create(sin.xsize, sin.ysize, MRC_MODE_FLOAT)?;
    if sin.mode == MRC_MODE_FLOAT {
        let input = sin
            .data
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut output = vec![0.; input.len()];
        crate::imod::libcfshr::filtxcorr::apply_kernel_filter(
            &input,
            &mut output,
            sin.xsize,
            sin.xsize,
            sin.ysize,
            mat,
            dim as i32,
        );
        for (bytes, value) in sout.data.chunks_exact_mut(4).zip(output) {
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
    } else {
        let mut num_threads = 1;
        if dim <= MAX_STATIC_KERNEL as usize {
            num_threads = crate::imod::libcfshr::b3dutil::num_omp_threads(
                (0.04 * ((sin.xsize * sin.ysize) as f64).sqrt()).round() as i32,
            );
        }
        if num_threads > 1 {
            for j in 0..sin.ysize {
                for i in 0..sin.xsize {
                    let mut smat = [0.0f32; (MAX_STATIC_KERNEL * MAX_STATIC_KERNEL) as usize];
                    mrc_slice_mat_getimat(sin, i, j, dim as i32, &mut smat);
                    let output = (i + j * sin.xsize) as usize * 4;
                    sout.data[output..output + 4]
                        .copy_from_slice(&mrc_slice_mat_mult(mat, &smat, dim as i32).to_ne_bytes());
                }
            }
        } else {
            let mut imat = vec![0.0f32; matrix_elements];
            for j in 0..sin.ysize {
                for i in 0..sin.xsize {
                    mrc_slice_mat_getimat(sin, i, j, dim as i32, &mut imat);
                    let output = (i + j * sin.xsize) as usize * 4;
                    sout.data[output..output + 4]
                        .copy_from_slice(&mrc_slice_mat_mult(mat, &imat, dim as i32).to_ne_bytes());
                }
            }
        }
    }
    Some(sout)
}
pub fn mrc_slice_mat_getimat(sin: &Islice, x: i32, y: i32, dim: i32, mat: &mut [f32]) {
    let xs = x - dim / 2;
    let xe = xs + dim;
    let ys = y - dim / 2;
    let ye = ys + dim;
    if xs >= 0 && xe < sin.xsize && ys >= 0 && ye < sin.ysize {
        for j in ys..ye {
            for i in xs..xe {
                mat[(i - xs + dim * (j - ys)) as usize] = slice_get_pixel_magnitude(sin, i, j);
            }
        }
    } else {
        for j in ys..ye {
            for i in xs..xe {
                let ic = i.clamp(0, sin.xsize - 1);
                let jc = j.clamp(0, sin.ysize - 1);
                mat[(i - xs + dim * (j - ys)) as usize] = slice_get_pixel_magnitude(sin, ic, jc);
            }
        }
    }
}
pub fn mrc_slice_mat_mult(m1: &[f32], m2: &[f32], dim: i32) -> f32 {
    m1.iter()
        .zip(m2)
        .take((dim * dim) as usize)
        .map(|(first, second)| first * second)
        .sum()
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
        assert_eq!(slice_get_x_size(&slice), 7);
        assert_eq!(slice_get_y_size(&slice), 9);
    }
    #[test]
    fn init_retains_source_fields_and_invalid_mode_result() {
        let mut s = slice_create(1, 1, 0).unwrap();
        assert_eq!(slice_init(s.as_mut(), 4, 5, 1, vec![0; 40]), 0);
        assert_eq!(
            (s.xsize, s.ysize, s.mode, s.dsize, s.csize),
            (4, 5, 1, 2, 1)
        );
        assert_eq!(slice_init(s.as_mut(), 1, 1, 12, Vec::new()), -1);
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
        let mut s = slice_create(3, 1, MRC_MODE_BYTE).unwrap();
        s.data.copy_from_slice(&[4, 5, 6]);
        s.mean = 9.;
        let mut v = [0.; 4];
        assert_eq!(slice_get_val(s.as_mut(), 1, 0, &mut v), 0);
        assert_eq!(v[0], 5.);
        assert_eq!(slice_get_val(s.as_mut(), 3, 0, &mut v), -1);
        assert_eq!(v[0], 9.);
        let mut s = slice_create(1, 1, MRC_MODE_COMPLEX_FLOAT).unwrap();
        s.data[..4].copy_from_slice(&3f32.to_ne_bytes());
        s.data[4..].copy_from_slice(&4f32.to_ne_bytes());
        s.xsize = 1;
        assert_eq!(slice_get_val(s.as_mut(), 0, 0, &mut v), 0);
        assert_eq!((v[0], v[1]), (3., 4.));
    }
    #[test]
    fn put_clear_and_pixel_magnitude_preserve_source_data_and_channel_rules() {
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
    #[test]
    fn min_max_preserves_each_supported_source_mode_and_rejects_other_modes() {
        let mut slice = slice_create(3, 1, MRC_MODE_BYTE).unwrap();
        slice.data.copy_from_slice(&[8, 2, 7]);
        assert_eq!(slice_min_max(slice.as_mut()), 0);
        assert_eq!((slice.min, slice.max), (2., 8.));

        let mut slice = slice_create(3, 1, MRC_MODE_SHORT).unwrap();
        slice.data.copy_from_slice(
            &[
                (-7i16).to_ne_bytes(),
                12i16.to_ne_bytes(),
                3i16.to_ne_bytes(),
            ]
            .concat(),
        );
        assert_eq!(slice_min_max(slice.as_mut()), 0);
        assert_eq!((slice.min, slice.max), (-7., 12.));

        let mut slice = slice_create(3, 1, MRC_MODE_USHORT).unwrap();
        slice.data.copy_from_slice(
            &[9u16.to_ne_bytes(), 14u16.to_ne_bytes(), 4u16.to_ne_bytes()].concat(),
        );
        assert_eq!(slice_min_max(slice.as_mut()), 0);
        assert_eq!((slice.min, slice.max), (4., 14.));

        let mut slice = slice_create(3, 1, MRC_MODE_FLOAT).unwrap();
        slice.data.copy_from_slice(
            &[
                (-1.5f32).to_ne_bytes(),
                4.25f32.to_ne_bytes(),
                0f32.to_ne_bytes(),
            ]
            .concat(),
        );
        assert_eq!(slice_min_max(slice.as_mut()), 0);
        assert_eq!((slice.min, slice.max), (-1.5, 4.25));

        slice.mode = 4;
        assert_eq!(slice_min_max(slice.as_mut()), 1);
    }
    #[test]
    fn scale_and_free_preserves_source_scaling_gate_and_data_modes() {
        let mut sin = slice_create(2, 1, MRC_MODE_BYTE).unwrap();
        sin.min = 10.;
        sin.max = 110.;
        let mut sout = slice_create(2, 1, MRC_MODE_FLOAT).unwrap();
        sout.data
            .copy_from_slice(&[0f32.to_ne_bytes(), 10f32.to_ne_bytes()].concat());
        slice_scale_and_free(sout.as_mut(), sin.as_mut());
        assert_eq!(sin.data, [10, 110]);

        let mut sout = slice_create(2, 1, MRC_MODE_SHORT).unwrap();
        sout.data
            .copy_from_slice(&[2i16.to_ne_bytes(), 3i16.to_ne_bytes()].concat());
        sin.min = 0.;
        sin.max = 0.;
        slice_scale_and_free(sout.as_mut(), sin.as_mut());
        assert_eq!(sin.data, [2, 3]);
    }
    #[test]
    fn byte_convolution_and_fixed_kernel_wrappers_preserve_source_handoff() {
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
    #[test]
    fn matrix_primitives_preserve_dot_product_and_edge_replication() {
        let m1 = [1., 2., 3., 4.];
        let m2 = [4., 3., 2., 1.];
        assert_eq!(mrc_slice_mat_mult(&m1, &m2, 2), 20.);

        let mut sin = slice_create(3, 3, MRC_MODE_BYTE).unwrap();
        sin.data.copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut mat = [0.; 9];
        mrc_slice_mat_getimat(sin.as_mut(), 0, 0, 3, &mut mat);
        assert_eq!(mat, [1., 1., 2., 1., 1., 2., 4., 4., 5.]);
    }

    #[test]
    fn matrix_filter_uses_native_slice_and_kernel_references() {
        let mut byte_slice = slice_create(2, 2, MRC_MODE_BYTE).unwrap();
        byte_slice.data.copy_from_slice(&[1, 2, 3, 4]);
        let byte_output = slice_mat_filter(byte_slice.as_mut(), &[1.], 1).unwrap();
        assert_eq!(
            byte_output
                .data
                .chunks_exact(4)
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>(),
            vec![1., 2., 3., 4.]
        );

        let mut float_slice = slice_create(2, 2, MRC_MODE_FLOAT).unwrap();
        float_slice.data.copy_from_slice(
            &[
                1f32.to_ne_bytes(),
                2f32.to_ne_bytes(),
                3f32.to_ne_bytes(),
                4f32.to_ne_bytes(),
            ]
            .concat(),
        );
        let float_output = slice_mat_filter(float_slice.as_mut(), &[1.], 1).unwrap();
        assert_eq!(float_output.data, float_slice.data);
        assert!(slice_mat_filter(float_slice.as_mut(), &[], 1).is_none());
    }

    #[test]
    fn stack_has_contiguous_owned_slices() {
        let slice = slice_create(2, 2, MRC_MODE_BYTE).unwrap();
        let stack = Istack {
            slices: vec![slice],
        };
        assert_eq!(stack.slices[0].data.len(), 4);
        assert_eq!(ISLICE_SOURCE_FUNCTIONS.len(), 21);
    }
}
