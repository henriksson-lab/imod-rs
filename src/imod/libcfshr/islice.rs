//! Translation of `IMOD/libcfshr/islice.c` and its direct `mrcslice.h` layouts.
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_4BIT, MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_HALF_FLOAT, MRC_MODE_SHORT,
    MRC_MODE_USHORT,
};
#[cfg(test)]
use crate::imod::libiimod::mrcfiles::{MRC_MODE_COMPLEX_FLOAT, MRC_MODE_RGB};
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
/// C `union MRCdata` (`mrcslice.h:45-51`): the pixel storage of an
/// `Islice`, typed by the union member the C reads it through.  `b` also
/// carries RGB (three bytes per pixel), `s` complex short (two per pixel) and
/// `f` complex float and `SLICE_MODE_MAX` (two and three floats per pixel).
///
/// The C punned freely between members; here a read through the wrong member
/// is a translation defect and panics by name.  `bytes`/`bytes_mut` are the
/// `void *`/`data.b`-as-raw-memory view the I/O layer takes.
#[derive(Clone, Debug, PartialEq)]
pub enum MrcData {
    /// `unsigned char *b`
    B(Vec<u8>),
    /// `b3dInt16 *s`
    S(Vec<i16>),
    /// `b3dUInt16 *us`
    Us(Vec<u16>),
    /// `b3dFloat *f`
    F(Vec<f32>),
}

impl Default for MrcData {
    fn default() -> Self {
        MrcData::B(Vec::new())
    }
}

impl MrcData {
    /// The union member the C reads a slice of MRC `mode` through.  Modes
    /// `sliceCreate` rejects (4-bit, half float) stay raw bytes.
    fn member_name(&self) -> &'static str {
        match self {
            MrcData::B(_) => "b",
            MrcData::S(_) => "s",
            MrcData::Us(_) => "us",
            MrcData::F(_) => "f",
        }
    }
    /// Zero-filled storage for `bytes` bytes of pixel data in MRC `mode`,
    /// typed by the member the C reads that mode through.  `vec![0; n]`
    /// goes through `calloc`, so like the C's `malloc` the pages are not
    /// touched until they are written; a `resize` after `try_reserve` would
    /// memset every page up front and double the page faults of a slice
    /// that is about to be filled anyway.  `None` is for a size that cannot
    /// exist; allocation failure itself aborts, as `vec!` does.
    pub fn try_zeroed(mode: i32, bytes: usize) -> Option<MrcData> {
        Some(match mode {
            1 | 3 => MrcData::S(vec![0; bytes / 2]),
            6 => MrcData::Us(vec![0; bytes / 2]),
            2 | 4 | 99 => MrcData::F(vec![0.; bytes / 4]),
            _ => MrcData::B(vec![0; bytes]),
        })
    }
    pub fn b(&self) -> &[u8] {
        match self {
            MrcData::B(v) => v,
            other => panic!("Islice data read as .b but holds .{}", other.member_name()),
        }
    }
    pub fn b_mut(&mut self) -> &mut [u8] {
        match self {
            MrcData::B(v) => v,
            other => panic!(
                "Islice data written as .b but holds .{}",
                other.member_name()
            ),
        }
    }
    pub fn s(&self) -> &[i16] {
        match self {
            MrcData::S(v) => v,
            other => panic!("Islice data read as .s but holds .{}", other.member_name()),
        }
    }
    pub fn s_mut(&mut self) -> &mut [i16] {
        match self {
            MrcData::S(v) => v,
            other => panic!(
                "Islice data written as .s but holds .{}",
                other.member_name()
            ),
        }
    }
    pub fn us(&self) -> &[u16] {
        match self {
            MrcData::Us(v) => v,
            other => panic!("Islice data read as .us but holds .{}", other.member_name()),
        }
    }
    pub fn us_mut(&mut self) -> &mut [u16] {
        match self {
            MrcData::Us(v) => v,
            other => panic!(
                "Islice data written as .us but holds .{}",
                other.member_name()
            ),
        }
    }
    pub fn f(&self) -> &[f32] {
        match self {
            MrcData::F(v) => v,
            other => panic!("Islice data read as .f but holds .{}", other.member_name()),
        }
    }
    pub fn f_mut(&mut self) -> &mut [f32] {
        match self {
            MrcData::F(v) => v,
            other => panic!(
                "Islice data written as .f but holds .{}",
                other.member_name()
            ),
        }
    }
    /// The storage as raw bytes, whatever member holds it — the C's
    /// `data.b` used as `void *` for `memcpy`, `fread` and `fwrite`.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: every member is a plain-old-data type with no padding, so
        // its `len * size_of` bytes are all initialised and readable.
        unsafe {
            match self {
                MrcData::B(v) => v,
                MrcData::S(v) => core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 2),
                MrcData::Us(v) => core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 2),
                MrcData::F(v) => core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4),
            }
        }
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as for `bytes`; additionally every bit pattern is a valid
        // `i16`/`u16`/`f32`, so writing arbitrary bytes cannot break the member.
        unsafe {
            match self {
                MrcData::B(v) => v,
                MrcData::S(v) => {
                    core::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 2)
                }
                MrcData::Us(v) => {
                    core::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 2)
                }
                MrcData::F(v) => {
                    core::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 4)
                }
            }
        }
    }
    /// Size of the storage in bytes — `xsize * ysize * dsize * csize` for a
    /// slice built by `sliceCreate`.
    pub fn byte_len(&self) -> usize {
        match self {
            MrcData::B(v) => v.len(),
            MrcData::S(v) => v.len() * 2,
            MrcData::Us(v) => v.len() * 2,
            MrcData::F(v) => v.len() * 4,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.byte_len() == 0
    }
}

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
        crate::imod::libcfshr::b3dutil::b3d_error(
            Some(&mut crate::imod::libcfshr::b3dutil::ImodFile::Stderr),
            format_args!("ERROR: sliceCreate - slice is too large for a 32-bit computer.\n"),
        );
        return None;
    }
    let mut dsize = 0;
    let mut csize = 0;
    if crate::imod::libcfshr::b3dutil::data_size_for_mode(mode, &mut dsize, &mut csize) != 0 {
        crate::imod::libcfshr::b3dutil::b3d_error(
            Some(&mut crate::imod::libcfshr::b3dutil::ImodFile::Stderr),
            format_args!("ERROR: sliceCreate - Unsupported data mode {}.\n", mode),
        );
        return None;
    }
    let Some(data) = MrcData::try_zeroed(mode, xysize * dsize as usize * csize as usize) else {
        crate::imod::libcfshr::b3dutil::b3d_error(
            Some(&mut crate::imod::libcfshr::b3dutil::ImodFile::Stderr),
            format_args!("ERROR: sliceCreate - failed to allocate slice memory.\n"),
        );
        return None;
    };
    Some(Islice {
        data,
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
/// C `sliceInit` (`islice.c:86`): the slice takes ownership of `data`, which
/// the caller has already sized and typed for `mode`.  As in the C, the size
/// fields and the data are stored before the mode is checked.
pub fn slice_init(s: &mut Islice, xsize: i32, ysize: i32, mode: i32, data: MrcData) -> i32 {
    s.xsize = xsize;
    s.ysize = ysize;
    s.mode = mode;
    s.data = data;
    if crate::imod::libcfshr::b3dutil::data_size_for_mode(mode, &mut s.dsize, &mut s.csize) != 0 {
        crate::imod::libcfshr::b3dutil::b3d_error(
            Some(&mut crate::imod::libcfshr::b3dutil::ImodFile::Stderr),
            format_args!("ERROR: sliceInit - Unsupported data mode {}.\n", mode),
        );
        return -1;
    }
    0
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
#[inline(always)]
pub fn slice_get_val(s: &Islice, x: i32, y: i32, val: &mut [f32; 4]) -> i32 {
    if x < 0 || y < 0 || x >= s.xsize || y >= s.ysize {
        val[0] = s.mean;
        return -1;
    }
    let index = x as usize + y as usize * s.xsize as usize;
    val[1] = 0.;
    match s.mode {
        0 => val[0] = s.data.b()[index] as f32,
        1 => val[0] = s.data.s()[index] as f32,
        6 => val[0] = s.data.us()[index] as f32,
        2 => val[0] = s.data.f()[index],
        3 => {
            let index = index * 2;
            let d = s.data.s();
            val[0] = d[index] as f32;
            val[1] = d[index + 1] as f32;
        }
        4 => {
            let index = index * 2;
            let d = s.data.f();
            val[0] = d[index];
            val[1] = d[index + 1];
        }
        16 => {
            let index = index * 3;
            let d = s.data.b();
            val[0] = d[index] as f32;
            val[1] = d[index + 1] as f32;
            val[2] = d[index + 2] as f32;
        }
        99 => {
            let index = index * 3;
            let d = s.data.f();
            val[0] = d[index];
            val[1] = d[index + 1];
            val[2] = d[index + 2];
        }
        _ => return -1,
    }
    0
}
#[inline(always)]
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
        0 => s.data.b_mut()[i] = val[0] as i32 as u8,
        1 => s.data.s_mut()[i] = val[0] as i32 as i16,
        6 => s.data.us_mut()[i] = val[0] as i32 as u16,
        2 => s.data.f_mut()[i] = val[0],
        3 => {
            let i = i * 2;
            let d = s.data.s_mut();
            d[i] = val[0] as i32 as i16;
            d[i + 1] = val[1] as i32 as i16;
        }
        4 => {
            let i = i * 2;
            let d = s.data.f_mut();
            d[i] = val[0];
            d[i + 1] = val[1];
        }
        16 => {
            let i = i * 3;
            let d = s.data.b_mut();
            d[i] = val[0] as i32 as u8;
            d[i + 1] = val[1] as i32 as u8;
            d[i + 2] = val[2] as i32 as u8;
        }
        99 => {
            let i = i * 3;
            let d = s.data.f_mut();
            d[i] = val[0];
            d[i + 1] = val[1];
            d[i + 2] = val[2];
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
    // `islice.c:365-431`: integer modes compare as `int`, the float mode
    // with `>`/`<` — which are false for NaN, so a NaN first pixel sticks.
    let n = (s.xsize * s.ysize) as usize;
    match s.mode {
        0 => {
            let d = s.data.b();
            let mut imin = d[0] as i32;
            let mut imax = imin;
            for &v in &d[1..n] {
                let ival = v as i32;
                if imin > ival {
                    imin = ival;
                }
                if imax < ival {
                    imax = ival;
                }
            }
            s.min = imin as f32;
            s.max = imax as f32;
        }
        1 => {
            let d = s.data.s();
            let mut imin = d[0] as i32;
            let mut imax = imin;
            for &v in &d[1..n] {
                let ival = v as i32;
                if imin > ival {
                    imin = ival;
                }
                if imax < ival {
                    imax = ival;
                }
            }
            s.min = imin as f32;
            s.max = imax as f32;
        }
        6 => {
            let d = s.data.us();
            let mut imin = d[0] as i32;
            let mut imax = imin;
            for &v in &d[1..n] {
                let ival = v as i32;
                if imin > ival {
                    imin = ival;
                }
                if imax < ival {
                    imax = ival;
                }
            }
            s.min = imin as f32;
            s.max = imax as f32;
        }
        2 => {
            let d = s.data.f();
            let mut fmin = d[0];
            let mut fmax = fmin;
            for &fval in &d[1..n] {
                if fmin > fval {
                    fmin = fval;
                }
                if fmax < fval {
                    fmax = fval;
                }
            }
            s.min = fmin;
            s.max = fmax;
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
    let out = sin.data.b_mut();
    match sout.mode {
        2 => {
            let src = sout.data.f();
            for i in 0..imax {
                out[i] = (src[i] * mval + aval) as i32 as u8;
            }
        }
        1 => {
            let src = sout.data.s();
            for i in 0..imax {
                out[i] = (src[i] as f32 * mval + aval) as i32 as u8;
            }
        }
        6 => {
            let src = sout.data.us();
            for i in 0..imax {
                out[i] = (src[i] as f32 * mval + aval) as i32 as u8;
            }
        }
        0 => {
            let src = sout.data.b();
            for i in 0..imax {
                out[i] = (src[i] as f32 * mval + aval) as i32 as u8;
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
    let xsize = sin.xsize;
    let index = |x: i32, y: i32| (x + y * xsize) as usize;
    {
        let src = sin.data.b();
        let out = sout.data.s_mut();
        for i in 1..imax {
            for j in 1..jmax {
                let val = src[index(i + 1, j + 1)] as i32 * mask[0][0]
                    + src[index(i, j + 1)] as i32 * mask[0][1]
                    + src[index(i - 1, j + 1)] as i32 * mask[0][2]
                    + src[index(i + 1, j)] as i32 * mask[1][0]
                    + src[index(i, j)] as i32 * mask[1][1]
                    + src[index(i - 1, j)] as i32 * mask[1][2]
                    + src[index(i + 1, j - 1)] as i32 * mask[2][0]
                    + src[index(i, j - 1)] as i32 * mask[2][1]
                    + src[index(i - 1, j - 1)] as i32 * mask[2][2];
                out[index(i, j)] = val as i16;
            }
        }
        for j in 1..jmax {
            out[index(0, j)] = out[index(1, j)];
            out[index(imax, j)] = out[index(imax - 1, j)];
        }
        for i in 0..=imax {
            out[index(i, 0)] = out[index(i, 1)];
            out[index(i, jmax)] = out[index(i, jmax - 1)];
        }
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
        crate::imod::libcfshr::filtxcorr::apply_kernel_filter(
            sin.data.f(),
            sout.data.f_mut(),
            sin.xsize,
            sin.xsize,
            sin.ysize,
            mat,
            dim as i32,
        );
    } else {
        let mut num_threads = 1;
        if dim <= MAX_STATIC_KERNEL as usize {
            num_threads = crate::imod::libcfshr::b3dutil::num_omp_threads(
                (0.04 * ((sin.xsize * sin.ysize) as f64).sqrt()).round() as i32,
            );
        }
        let out = sout.data.f_mut();
        if num_threads > 1 {
            for j in 0..sin.ysize {
                for i in 0..sin.xsize {
                    let mut smat = [0.0f32; (MAX_STATIC_KERNEL * MAX_STATIC_KERNEL) as usize];
                    mrc_slice_mat_getimat(sin, i, j, dim as i32, &mut smat);
                    out[(i + j * sin.xsize) as usize] = mrc_slice_mat_mult(mat, &smat, dim as i32);
                }
            }
        } else {
            let mut imat = vec![0.0f32; matrix_elements];
            for j in 0..sin.ysize {
                for i in 0..sin.xsize {
                    mrc_slice_mat_getimat(sin, i, j, dim as i32, &mut imat);
                    out[(i + j * sin.xsize) as usize] = mrc_slice_mat_mult(mat, &imat, dim as i32);
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
    fn init_retains_source_fields_and_invalid_mode_result() {
        let mut s = slice_create(1, 1, 0).unwrap();
        assert_eq!(slice_init(s.as_mut(), 4, 5, 1, MrcData::S(vec![0; 20])), 0);
        assert_eq!(
            (s.xsize, s.ysize, s.mode, s.dsize, s.csize),
            (4, 5, 1, 2, 1)
        );
        assert_eq!(slice_init(s.as_mut(), 1, 1, 12, MrcData::default()), -1);
    }
    #[test]
    fn create_uses_owned_allocation_and_mode_checks() {
        let s = slice_create(3, 2, MRC_MODE_SHORT).unwrap();
        assert_eq!(
            (s.xsize, s.ysize, s.dsize, s.csize, s.index),
            (3, 2, 2, 1, -1)
        );
        assert_eq!(s.data.byte_len(), 12);
        assert_eq!(s.data.s().len(), 6);
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
        s.data.b_mut().copy_from_slice(&[4, 5, 6]);
        s.mean = 9.;
        let mut v = [0.; 4];
        assert_eq!(slice_get_val(s.as_mut(), 1, 0, &mut v), 0);
        assert_eq!(v[0], 5.);
        assert_eq!(slice_get_val(s.as_mut(), 3, 0, &mut v), -1);
        assert_eq!(v[0], 9.);
        let mut s = slice_create(1, 1, MRC_MODE_COMPLEX_FLOAT).unwrap();
        s.data.f_mut().copy_from_slice(&[3., 4.]);
        s.xsize = 1;
        assert_eq!(slice_get_val(s.as_mut(), 0, 0, &mut v), 0);
        assert_eq!((v[0], v[1]), (3., 4.));
    }
    #[test]
    fn put_and_pixel_magnitude_preserve_source_data_and_channel_rules() {
        let mut s = slice_create(2, 2, MRC_MODE_BYTE).unwrap();
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
        slice.data.b_mut().copy_from_slice(&[8, 2, 7]);
        assert_eq!(slice_min_max(slice.as_mut()), 0);
        assert_eq!((slice.min, slice.max), (2., 8.));

        let mut slice = slice_create(3, 1, MRC_MODE_SHORT).unwrap();
        slice.data.s_mut().copy_from_slice(&[-7, 12, 3]);
        assert_eq!(slice_min_max(slice.as_mut()), 0);
        assert_eq!((slice.min, slice.max), (-7., 12.));

        let mut slice = slice_create(3, 1, MRC_MODE_USHORT).unwrap();
        slice.data.us_mut().copy_from_slice(&[9, 14, 4]);
        assert_eq!(slice_min_max(slice.as_mut()), 0);
        assert_eq!((slice.min, slice.max), (4., 14.));

        let mut slice = slice_create(3, 1, MRC_MODE_FLOAT).unwrap();
        slice.data.f_mut().copy_from_slice(&[-1.5, 4.25, 0.]);
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
        sout.data.f_mut().copy_from_slice(&[0., 10.]);
        slice_scale_and_free(sout.as_mut(), sin.as_mut());
        assert_eq!(sin.data.b(), [10, 110]);

        let mut sout = slice_create(2, 1, MRC_MODE_SHORT).unwrap();
        sout.data.s_mut().copy_from_slice(&[2, 3]);
        sin.min = 0.;
        sin.max = 0.;
        slice_scale_and_free(sout.as_mut(), sin.as_mut());
        assert_eq!(sin.data.b(), [2, 3]);
    }
    #[test]
    fn byte_convolution_and_fixed_kernel_wrappers_preserve_source_handoff() {
        let mut sin = slice_create(3, 3, MRC_MODE_BYTE).unwrap();
        sin.data.b_mut().fill(10);
        assert_eq!(
            slice_byte_convolve(sin.as_mut(), &[[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            0
        );
        assert_eq!(sin.data.b(), [10; 9]);

        sin.data.b_mut().fill(10);
        assert_eq!(slice_byte_edge_laplacian(sin.as_mut()), 0);
        assert_eq!(sin.data.b(), [40; 9]);
        sin.data.b_mut().fill(10);
        assert_eq!(slice_byte_sharpen(sin.as_mut()), 0);
        assert_eq!(sin.data.b(), [10; 9]);
        sin.data.b_mut().fill(10);
        assert_eq!(slice_byte_smooth(sin.as_mut()), 0);
        assert_eq!(sin.data.b(), [160; 9]);
    }
    #[test]
    fn matrix_primitives_preserve_dot_product_and_edge_replication() {
        let m1 = [1., 2., 3., 4.];
        let m2 = [4., 3., 2., 1.];
        assert_eq!(mrc_slice_mat_mult(&m1, &m2, 2), 20.);

        let mut sin = slice_create(3, 3, MRC_MODE_BYTE).unwrap();
        sin.data
            .b_mut()
            .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut mat = [0.; 9];
        mrc_slice_mat_getimat(sin.as_mut(), 0, 0, 3, &mut mat);
        assert_eq!(mat, [1., 1., 2., 1., 1., 2., 4., 4., 5.]);
    }

    #[test]
    fn matrix_filter_uses_native_slice_and_kernel_references() {
        let mut byte_slice = slice_create(2, 2, MRC_MODE_BYTE).unwrap();
        byte_slice.data.b_mut().copy_from_slice(&[1, 2, 3, 4]);
        let byte_output = slice_mat_filter(byte_slice.as_mut(), &[1.], 1).unwrap();
        assert_eq!(byte_output.data.f(), [1., 2., 3., 4.]);

        let mut float_slice = slice_create(2, 2, MRC_MODE_FLOAT).unwrap();
        float_slice.data.f_mut().copy_from_slice(&[1., 2., 3., 4.]);
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
        assert_eq!(stack.slices[0].data.byte_len(), 4);
        assert_eq!(ISLICE_SOURCE_FUNCTIONS.len(), 21);
    }
}
