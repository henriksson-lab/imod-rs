//! Translation of `IMOD/raptor/opencv/cxarray.cpp` (the parts RAPTOR
//! reaches): matrix and image creation, element access and the sub-array
//! headers `cvGetRows`/`cvGetCols`/`cvGetSubRect`.
//!
//! `cvCreateMat` is `cvCreateMatHeader` followed by `cvCreateData`, and
//! `cvCreateImage` is `cvCreateImageHeader` (`cvInitImageHeader`) followed
//! by `cvCreateData`; each pair is one constructor here.  The C leaves the
//! allocated data uninitialised; a `Vec` is zero-filled.  No reached path
//! reads an element before writing it.
//!
//! Folded into the Rust types: `cvCreateMatHeader`, `cvCreateData`,
//! `cvCreateImageHeader` and `cvInitImageHeader` are the constructors;
//! `cvReleaseMat`, `cvReleaseImage`, `cvReleaseImageHeader`, `cvReleaseData`
//! and `cvDecRefData` are `Drop`; `cvGetMat` of a matrix is the matrix itself,
//! and of an image it is `cvInitMatHeader` over the image data, which
//! `cvtemplmatch.rs` and `cxmeansdv.rs` spell out where they need it
//! (continuous when `widthStep` is `width*4`).  `cvGetSubRect` is used only
//! inside `icvCrossCorr` and is written out there as offsets.

use super::cxerror::{CV_STS_BAD_SIZE, CV_STS_OUT_OF_RANGE, cv_error};
use super::cxtypes::*;

impl CvMat {
    /// `cvCreateMat(rows, cols, type)` (`cxarray.cpp:96`): `cvCreateMatHeader`
    /// (`cxarray.cpp:124`), `icvCheckHuge` and `cvCreateData`.  Only
    /// `CV_64FC1` is reached.
    pub fn create(rows: i32, cols: i32, type_: i32) -> CvMat {
        let type_ = cv_mat_type(type_);
        assert!(
            type_ == CV_64FC1,
            "cvCreateMat: RAPTOR creates only CV_64FC1 matrices"
        );

        if rows <= 0 || cols <= 0 {
            cv_error(
                CV_STS_BAD_SIZE,
                "cvCreateMatHeader",
                "Non-positive width or height",
                "cxarray.cpp",
                135,
            );
        }

        let min_step = cv_elem_size(type_) * cols;
        let step = if rows == 1 { 0 } else { min_step };
        let mut flags = CV_MAT_MAGIC_VAL
            | type_
            | if step == 0 || step == min_step {
                CV_MAT_CONT_FLAG
            } else {
                0
            };
        // icvCheckHuge
        if step as i64 * rows as i64 > i32::MAX as i64 {
            flags &= !CV_MAT_CONT_FLAG;
        }

        CvMat {
            type_: flags,
            step,
            data: vec![0.; rows as usize * cols as usize],
            rows,
            cols,
        }
    }

    /// `cvGetSize(arr)` (`cxarray.cpp:1421`).
    pub fn get_size(&self) -> CvSize {
        CvSize {
            width: self.cols,
            height: self.rows,
        }
    }

    /// `cvGetReal2D(arr, y, x)` (`cxarray.cpp:2378`).
    pub fn get_real_2d(&self, y: i32, x: i32) -> f64 {
        self.view().get_real_2d(y, x)
    }

    /// `cvSetReal2D(arr, y, x, value)` (`cxarray.cpp:2647`).
    pub fn set_real_2d(&mut self, y: i32, x: i32, value: f64) {
        if (y as u32) >= (self.rows as u32) || (x as u32) >= (self.cols as u32) {
            cv_error(
                CV_STS_OUT_OF_RANGE,
                "cvSetReal2D",
                "index is out of range",
                "cxarray.cpp",
                2659,
            );
        }
        let k = y as usize * (self.step / 8) as usize + x as usize;
        self.data[k] = value;
    }

    /// `cvmSet(mat, row, col, value)` (`cxcore.h`, inline): a direct store
    /// through `mat->data.ptr + (size_t)mat->step*row`.
    pub fn cvm_set(&mut self, row: i32, col: i32, value: f64) {
        assert!((row as u32) < (self.rows as u32) && (col as u32) < (self.cols as u32));
        let k = row as usize * (self.step / 8) as usize + col as usize;
        self.data[k] = value;
    }

    /// `cvCloneMat(src)` (`cxarray.cpp:248`): a new header of the same size
    /// and type (`cvCreateMatHeader`) and a copy of the data (`cvCopy`).
    pub fn clone_mat(&self) -> CvMat {
        let mut dst = CvMat::create(self.rows, self.cols, self.type_);
        let n = self.rows as usize * self.cols as usize;
        dst.data[..n].copy_from_slice(&self.data[..n]);
        dst
    }

    /// The matrix itself as a read-only header argument (`(CvArr*)mat`).
    pub fn view(&self) -> CvMatRef<'_> {
        CvMatRef {
            type_: self.type_,
            step: self.step,
            data: &self.data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// `cvGetCol(arr, submat, col)` (`cxcore.h`: `cvGetCols(arr, submat, col,
    /// col + 1)`, `cxarray.cpp:1574`).
    pub fn get_col(&self, col: i32) -> CvMatRef<'_> {
        cv_get_cols(self.view(), col, col + 1)
    }

    /// `cvGetRow(arr, submat, row)` (`cxcore.h`: `cvGetRows(arr, submat,
    /// row, row + 1, 1)`, `cxarray.cpp:1515`).
    pub fn get_row(&self, row: i32) -> CvMatRef<'_> {
        cv_get_rows(self.view(), row, row + 1, 1)
    }

    /// `cvGetRows(arr, submat, start_row, end_row, delta_row)`
    /// (`cxarray.cpp:1515`).
    pub fn get_rows(&self, start_row: i32, end_row: i32, delta_row: i32) -> CvMatRef<'_> {
        cv_get_rows(self.view(), start_row, end_row, delta_row)
    }
}

impl<'a> CvMatRef<'a> {
    /// `cvGetSize(arr)` (`cxarray.cpp:1421`).
    pub fn get_size(&self) -> CvSize {
        CvSize {
            width: self.cols,
            height: self.rows,
        }
    }

    /// `cvGetReal2D(arr, y, x)` (`cxarray.cpp:2378`): the `CvMat` arm and
    /// `icvGetReal` for `CV_64F`.
    pub fn get_real_2d(&self, y: i32, x: i32) -> f64 {
        if (y as u32) >= (self.rows as u32) || (x as u32) >= (self.cols as u32) {
            cv_error(
                CV_STS_OUT_OF_RANGE,
                "cvGetReal2D",
                "index is out of range",
                "cxarray.cpp",
                2391,
            );
        }
        self.data[y as usize * (self.step / 8) as usize + x as usize]
    }
}

/// `cvGetRows( const CvArr* arr, CvMat* submat, int start_row, int
/// end_row, int delta_row )` (`cxarray.cpp:1515`).
pub fn cv_get_rows(
    mat: CvMatRef<'_>,
    start_row: i32,
    end_row: i32,
    delta_row: i32,
) -> CvMatRef<'_> {
    if (start_row as u32) >= (mat.rows as u32)
        || (end_row as u32) > (mat.rows as u32)
        || delta_row <= 0
    {
        cv_error(CV_STS_OUT_OF_RANGE, "cvGetRows", "", "cxarray.cpp", 1536);
    }

    let rows;
    let mut step;
    if delta_row == 1 {
        rows = end_row - start_row;
        step = mat.step & if rows > 1 { -1 } else { 0 };
    } else {
        rows = (end_row - start_row + delta_row - 1) / delta_row;
        step = mat.step * delta_row;
    }

    step &= if rows > 1 { -1 } else { 0 };
    let offset = start_row as usize * (mat.step / 8) as usize;
    let type_ = (mat.type_ | if step == 0 { CV_MAT_CONT_FLAG } else { 0 })
        & if delta_row != 1 {
            !CV_MAT_CONT_FLAG
        } else {
            -1
        };
    CvMatRef {
        type_,
        step,
        data: &mat.data[offset..],
        rows,
        cols: mat.cols,
    }
}

/// `cvGetCols( const CvArr* arr, CvMat* submat, int start_col, int
/// end_col )` (`cxarray.cpp:1574`).
pub fn cv_get_cols(mat: CvMatRef<'_>, start_col: i32, end_col: i32) -> CvMatRef<'_> {
    if (start_col as u32) >= (mat.cols as u32) || (end_col as u32) > (mat.cols as u32) {
        cv_error(CV_STS_OUT_OF_RANGE, "cvGetCols", "", "cxarray.cpp", 1593);
    }

    let rows = mat.rows;
    let cols = end_col - start_col;
    let step = mat.step & if rows > 1 { -1 } else { 0 };
    let type_ = mat.type_
        & if step != 0 && cols < mat.cols {
            !CV_MAT_CONT_FLAG
        } else {
            -1
        };
    CvMatRef {
        type_,
        step,
        data: &mat.data[start_col as usize..],
        rows,
        cols,
    }
}

impl IplImage {
    /// `cvCreateImage(size, depth, channels)` (`cxarray.cpp:3318`):
    /// `cvCreateImageHeader` / `cvInitImageHeader` (`cxarray.cpp:3345`) with
    /// `IPL_ORIGIN_TL` and `CV_DEFAULT_IMAGE_ROW_ALIGN` (4), then
    /// `cvCreateData`.  Only `IPL_DEPTH_32F` with one channel is reached.
    pub fn create(size: CvSize, depth: i32, channels: i32) -> IplImage {
        assert!(
            depth == IPL_DEPTH_32F && channels == 1,
            "cvCreateImage: RAPTOR creates only IPL_DEPTH_32F one-channel images"
        );
        let align = 4;

        if size.width < 0 || size.height < 0 {
            cv_error(
                -25,
                "cvInitImageHeader",
                "Bad input roi",
                "cxarray.cpp",
                3366,
            );
        }

        let n_channels = channels.max(1);
        let width_step = (((size.width * n_channels * (depth & !IPL_DEPTH_SIGN) + 7) / 8) + align
            - 1)
            & !(align - 1);
        let image_size = width_step * size.height;

        IplImage {
            n_channels,
            depth,
            width: size.width,
            height: size.height,
            image_size,
            width_step,
            image_data: vec![0f32; (image_size / 4) as usize],
        }
    }

    /// `cvGetSize(arr)` (`cxarray.cpp:1421`) of an image with no ROI.
    pub fn get_size(&self) -> CvSize {
        CvSize {
            width: self.width,
            height: self.height,
        }
    }
}

/// `IPL_DEPTH_SIGN`.
const IPL_DEPTH_SIGN: i32 = 0x80000000_u32 as i32;
