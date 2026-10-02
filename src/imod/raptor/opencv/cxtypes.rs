//! Translation of `IMOD/raptor/opencv/cxtypes.h`: the OpenCV 1.x data types
//! and constants the RAPTOR program reaches.
//!
//! OpenCV's `CvArr*` is a run-time polymorphic header (`CvMat`, `IplImage`,
//! `CvMatND`, `CvSparseMat`) and every entry point dispatches on its element
//! type.  RAPTOR reaches exactly two element types: every matrix it creates
//! is `CV_64FC1` (`cvCreateMat(.., CV_64FC1)` at every call site) and every
//! image `IPL_DEPTH_32F` with one channel (`cvCreateImage(.., IPL_DEPTH_32F,
//! 1)` at every call site); inside `cvMatchTemplate` OpenCV itself adds
//! `CV_64FC1` work matrices.  So the Rust types are that closed set:
//! [`CvMat`] owns `f64` data, [`IplImage`] owns `f32` data, and
//! [`CvMatRef`] is the header `cvGetCol`/`cvGetRow`/`cvGetRows`/
//! `cvGetSubRect` fill in over someone else's `f64` data.  The dispatch
//! arms for the other element types are unreachable from RAPTOR and are
//! recorded in `DEAD_CODE.md`.
//!
//! `cvReleaseMat`/`cvReleaseImage` are `Drop`; the `refcount` and
//! `hdr_refcount` fields that support them in C are not needed.

/// `CV_CN_SHIFT`.
pub const CV_CN_SHIFT: i32 = 3;
/// `CV_DEPTH_MAX`.
pub const CV_DEPTH_MAX: i32 = 1 << CV_CN_SHIFT;
/// `CV_8U`.
pub const CV_8U: i32 = 0;
/// `CV_8S`.
pub const CV_8S: i32 = 1;
/// `CV_16U`.
pub const CV_16U: i32 = 2;
/// `CV_16S`.
pub const CV_16S: i32 = 3;
/// `CV_32S`.
pub const CV_32S: i32 = 4;
/// `CV_32F`.
pub const CV_32F: i32 = 5;
/// `CV_64F`.
pub const CV_64F: i32 = 6;
/// `CV_CN_MAX`.
pub const CV_CN_MAX: i32 = 64;
/// `CV_MAKETYPE(depth, cn)`.
pub const fn cv_maketype(depth: i32, cn: i32) -> i32 {
    depth + ((cn - 1) << CV_CN_SHIFT)
}
/// `CV_32FC1`.
pub const CV_32FC1: i32 = cv_maketype(CV_32F, 1);
/// `CV_64FC1`.
pub const CV_64FC1: i32 = cv_maketype(CV_64F, 1);
/// `CV_MAT_CN_MASK`.
pub const CV_MAT_CN_MASK: i32 = (CV_CN_MAX - 1) << CV_CN_SHIFT;
/// `CV_MAT_DEPTH_MASK`.
pub const CV_MAT_DEPTH_MASK: i32 = CV_DEPTH_MAX - 1;
/// `CV_MAT_TYPE_MASK`.
pub const CV_MAT_TYPE_MASK: i32 = CV_DEPTH_MAX * CV_CN_MAX - 1;
/// `CV_MAT_CONT_FLAG_SHIFT`.
pub const CV_MAT_CONT_FLAG_SHIFT: i32 = 14;
/// `CV_MAT_CONT_FLAG`.
pub const CV_MAT_CONT_FLAG: i32 = 1 << CV_MAT_CONT_FLAG_SHIFT;
/// `CV_MAGIC_MASK`.
pub const CV_MAGIC_MASK: i32 = 0xFFFF0000_u32 as i32;
/// `CV_MAT_MAGIC_VAL`.
pub const CV_MAT_MAGIC_VAL: i32 = 0x42420000;
/// `IPL_DEPTH_32F`.
pub const IPL_DEPTH_32F: i32 = 32;

/// `CV_MAT_DEPTH(flags)`.
pub const fn cv_mat_depth(flags: i32) -> i32 {
    flags & CV_MAT_DEPTH_MASK
}
/// `CV_MAT_CN(flags)`.
pub const fn cv_mat_cn(flags: i32) -> i32 {
    ((flags & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1
}
/// `CV_MAT_TYPE(flags)`.
pub const fn cv_mat_type(flags: i32) -> i32 {
    flags & CV_MAT_TYPE_MASK
}
/// `CV_IS_MAT_CONT(flags)`.
pub const fn cv_is_mat_cont(flags: i32) -> bool {
    (flags & CV_MAT_CONT_FLAG) != 0
}

/// `CvSize` (and `cvSize(width, height)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CvSize {
    pub width: i32,
    pub height: i32,
}

impl CvSize {
    /// `cvSize(width, height)`.
    pub const fn new(width: i32, height: i32) -> CvSize {
        CvSize { width, height }
    }
}

/// `CvPoint` (and `cvPoint(x, y)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CvPoint {
    pub x: i32,
    pub y: i32,
}

/// `CvRect` (and `cvRect(x, y, width, height)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CvRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

/// `CvScalar`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CvScalar {
    pub val: [f64; 4],
}

impl CvScalar {
    /// `cvScalar(val0, val1, val2, val3)` (the C defaults are 0).
    pub const fn new(val0: f64, val1: f64, val2: f64, val3: f64) -> CvScalar {
        CvScalar {
            val: [val0, val1, val2, val3],
        }
    }

    /// `cvScalarAll(val0123)`.
    pub const fn all(val0123: f64) -> CvScalar {
        CvScalar { val: [val0123; 4] }
    }
}

/// `CvMat` of type `CV_64FC1`, owning its data (`data.db`).  `type_` holds
/// the full header flags as `cvCreateMat` sets them (`CV_MAT_MAGIC_VAL |
/// CV_MAT_CONT_FLAG | CV_64FC1`) and `step` is in bytes, as in C.
#[derive(Clone, Debug)]
pub struct CvMat {
    pub type_: i32,
    pub step: i32,
    pub data: Vec<f64>,
    pub rows: i32,
    pub cols: i32,
}

/// A `CvMat` header over another matrix's `f64` data: what `cvGetCol`,
/// `cvGetRow`, `cvGetRows` and `cvGetSubRect` fill in.  `data` starts at the
/// header's first element (`data.db`), `step` is in bytes.  Every read-only
/// matrix argument of the translated entry points is one of these;
/// [`CvMat::view`] makes one over a whole matrix.
#[derive(Clone, Copy, Debug)]
pub struct CvMatRef<'a> {
    pub type_: i32,
    pub step: i32,
    pub data: &'a [f64],
    pub rows: i32,
    pub cols: i32,
}

/// `IplImage` of depth `IPL_DEPTH_32F` with one channel, no ROI and no COI
/// (the only kind RAPTOR creates), owning its data (`imageData` as
/// `float*`).  `width_step` is in bytes.
#[derive(Clone, Debug)]
pub struct IplImage {
    pub n_channels: i32,
    pub depth: i32,
    pub width: i32,
    pub height: i32,
    pub image_size: i32,
    pub width_step: i32,
    pub image_data: Vec<f32>,
}

/// `cvRound( double value )` (`cxtypes.h:205`), the `CV_SSE2` arm this
/// x86-64 build compiles: `_mm_cvtsd_si32`, which rounds half to even under
/// the default rounding mode and gives `INT_MIN` for NaN or an out-of-range
/// value.
pub fn cv_round(value: f64) -> i32 {
    let r = value.round_ties_even();
    if r.is_nan() || r < i32::MIN as f64 || r > i32::MAX as f64 {
        i32::MIN
    } else {
        r as i32
    }
}

/// `CV_ELEM_SIZE(type)` for the two element types reached: 4 bytes for
/// `CV_32FC1`, 8 for `CV_64FC1`.
pub const fn cv_elem_size(type_: i32) -> i32 {
    match cv_mat_type(type_) {
        CV_32FC1 => 4,
        _ => 8,
    }
}

impl<'a> CvMatRef<'a> {
    /// `((double*)(mat->data.ptr + (size_t)row*mat->step))[col]`: the
    /// element addressing every `CV_64FC1` kernel uses.
    #[inline]
    pub fn elem(&self, row: i32, col: i32) -> f64 {
        self.data[row as usize * (self.step / 8) as usize + col as usize]
    }
}

impl CvMat {
    /// The destination form of [`CvMatRef::elem`]: the index of element
    /// (`row`, `col`) in `data`.
    #[inline]
    pub fn index(&self, row: i32, col: i32) -> usize {
        row as usize * (self.step / 8) as usize + col as usize
    }
}
