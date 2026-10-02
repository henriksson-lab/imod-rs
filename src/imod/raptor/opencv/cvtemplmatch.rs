//! Translation of `IMOD/raptor/opencv/cvtemplmatch.cpp`: `icvCrossCorr` and
//! `cvMatchTemplate`, for the one-channel `IPL_DEPTH_32F` images RAPTOR
//! passes.
//!
//! In `icvCrossCorr` the working depth is `CV_64F` (the image depth is above
//! `CV_8U`), the template, image and result have one channel, so the
//! multi-channel `cvSplit`/`cvMerge`/`cvAcc` steps and the scratch buffer are
//! not reached; with `anchor` (0,0) and a result of `(W-w+1)x(H-h+1)` every
//! image block lies inside the image, so the `cvCopyMakeBorder` padding is
//! not reached either.  The IPP matching hooks are null in this build.

use super::cvsumpixels::cv_integral;
use super::cxdxt::{
    CV_DXT_FORWARD, CV_DXT_INVERSE, CV_DXT_MUL_CONJ, CV_DXT_SCALE, cv_dft, cv_get_optimal_dft_size,
    cv_mul_spectrums,
};
use super::cxerror::{CV_STS_BAD_ARG, CV_STS_OUT_OF_RANGE, CV_STS_UNMATCHED_SIZES, cv_error};
use super::cxmeansdv::cv_avg_sdv;
use super::cxtypes::*;

/// `CV_TM_SQDIFF`.
pub const CV_TM_SQDIFF: i32 = 0;
/// `CV_TM_SQDIFF_NORMED`.
pub const CV_TM_SQDIFF_NORMED: i32 = 1;
/// `CV_TM_CCORR`.
pub const CV_TM_CCORR: i32 = 2;
/// `CV_TM_CCORR_NORMED`.
pub const CV_TM_CCORR_NORMED: i32 = 3;
/// `CV_TM_CCOEFF`.
pub const CV_TM_CCOEFF: i32 = 4;
/// `CV_TM_CCOEFF_NORMED`.
pub const CV_TM_CCOEFF_NORMED: i32 = 5;

/// `icvCrossCorr( const CvArr* _img, const CvArr* _templ, CvArr* _corr,
/// CvPoint anchor )` (`cvtemplmatch.cpp:44`) with `anchor` (0,0), as
/// `cvMatchTemplate` calls it.
fn icv_cross_corr(img: &IplImage, templ: &IplImage, corr: &mut IplImage) {
    const BLOCK_SCALE: f64 = 4.5;
    const MIN_BLOCK_SIZE: i32 = 256;

    let (img_rows, img_cols) = (img.height, img.width);
    let (templ_rows, templ_cols) = (templ.height, templ.width);
    let (corr_rows, corr_cols) = (corr.height, corr.width);
    let img_step = (img.width_step / 4) as usize;
    let templ_step = (templ.width_step / 4) as usize;
    let corr_step = (corr.width_step / 4) as usize;

    if img_cols < templ_cols || img_rows < templ_rows {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "icvCrossCorr",
            "Such a combination of image and template/filter size is not supported",
            "cvtemplmatch.cpp",
            110,
        );
    }

    if corr_rows > img_rows + templ_rows - 1 || corr_cols > img_cols + templ_cols - 1 {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "icvCrossCorr",
            "output image should not be greater than (W + w - 1)x(H + h - 1)",
            "cvtemplmatch.cpp",
            115,
        );
    }

    let mut blocksize = CvSize {
        width: cv_round(templ_cols as f64 * BLOCK_SCALE),
        height: 0,
    };
    blocksize.width = blocksize.width.max(MIN_BLOCK_SIZE - templ_cols + 1);
    blocksize.width = blocksize.width.min(corr_cols);
    blocksize.height = cv_round(templ_rows as f64 * BLOCK_SCALE);
    blocksize.height = blocksize.height.max(MIN_BLOCK_SIZE - templ_rows + 1);
    blocksize.height = blocksize.height.min(corr_rows);

    let mut dftsize = CvSize {
        width: cv_get_optimal_dft_size(blocksize.width + templ_cols - 1),
        height: 0,
    };
    if dftsize.width == 1 {
        dftsize.width = 2;
    }
    dftsize.height = cv_get_optimal_dft_size(blocksize.height + templ_rows - 1);
    if dftsize.width <= 0 || dftsize.height <= 0 {
        cv_error(
            CV_STS_OUT_OF_RANGE,
            "icvCrossCorr",
            "the input arrays are too big",
            "cvtemplmatch.cpp",
            134,
        );
    }

    // recompute block size
    blocksize.width = dftsize.width - templ_cols + 1;
    blocksize.width = blocksize.width.min(corr_cols);
    blocksize.height = dftsize.height - templ_rows + 1;
    blocksize.height = blocksize.height.min(corr_rows);

    let mut dft_img = CvMat::create(dftsize.height, dftsize.width, CV_64FC1);
    let mut dft_templ = CvMat::create(dftsize.height, dftsize.width, CV_64FC1);
    let dstep = (dftsize.width) as usize; // both are continuous, rows > 1 or not
    let dft_img_step = dft_img.step;
    let dft_img_cont = cv_is_mat_cont(dft_img.type_);
    let dft_templ_step = dft_templ.step;
    let dft_templ_cont = cv_is_mat_cont(dft_templ.type_);

    // compute DFT of each template plane (one plane)
    {
        // `cvConvert( src, dst )`: icvCvtTo_64f_C1R from CV_32F
        for r in 0..templ_rows as usize {
            for c in 0..templ_cols as usize {
                let t0 = templ.image_data[r * templ_step + c] as f64;
                dft_templ.data[r * dstep + c] = t0;
            }
        }

        if dft_templ.cols > templ_cols {
            // `cvZero` of the sub-rectangle right of the template
            for r in 0..templ_rows as usize {
                for c in templ_cols as usize..dft_templ.cols as usize {
                    dft_templ.data[r * dstep + c] = 0.;
                }
            }
        }

        cv_dft(
            &mut dft_templ.data,
            dftsize.height,
            dftsize.width,
            dft_templ_step,
            dft_templ_cont,
            CV_DXT_FORWARD + CV_DXT_SCALE,
            templ_rows,
        );
    }

    // calculate correlation by blocks
    let mut y = 0;
    while y < corr_rows {
        let mut x = 0;
        while x < corr_cols {
            let mut csz = CvSize {
                width: blocksize.width,
                height: blocksize.height,
            };
            let x0 = x;
            let y0 = y;
            let x1 = 0.max(x0);
            let y1 = 0.max(y0);
            csz.width = csz.width.min(corr_cols - x);
            csz.height = csz.height.min(corr_rows - y);
            let isz = CvSize {
                width: csz.width + templ_cols - 1,
                height: csz.height + templ_rows - 1,
            };
            let x2 = img_cols.min(x0 + isz.width);
            let y2 = img_rows.min(y0 + isz.height);

            assert!(
                x2 - x1 >= isz.width && y2 - y1 >= isz.height,
                "icvCrossCorr: cvCopyMakeBorder is not reached from cvMatchTemplate"
            );

            // `cvConvert( src, dst1 )`: the image block into dft_img
            for r in 0..(y2 - y1) as usize {
                for c in 0..(x2 - x1) as usize {
                    let t0 = img.image_data[(y1 as usize + r) * img_step + x1 as usize + c] as f64;
                    dft_img.data[r * dstep + c] = t0;
                }
            }

            if dftsize.width > isz.width {
                for r in 0..dftsize.height as usize {
                    for c in isz.width as usize..dftsize.width as usize {
                        dft_img.data[r * dstep + c] = 0.;
                    }
                }
            }

            cv_dft(
                &mut dft_img.data,
                dftsize.height,
                dftsize.width,
                dft_img_step,
                dft_img_cont,
                CV_DXT_FORWARD,
                isz.height,
            );

            cv_mul_spectrums(
                &mut dft_img.data,
                dft_img_step,
                &dft_templ.data,
                dft_templ_step,
                dftsize.height,
                dftsize.width,
                dft_img_cont && dft_templ_cont,
                CV_DXT_MUL_CONJ,
            );
            cv_dft(
                &mut dft_img.data,
                dftsize.height,
                dftsize.width,
                dft_img_step,
                dft_img_cont,
                CV_DXT_INVERSE,
                csz.height,
            );

            // `cvConvert( src, dst )`: icvCvtTo_32f_C1R from CV_64F
            for r in 0..csz.height as usize {
                for c in 0..csz.width as usize {
                    let t0 = dft_img.data[r * dstep + c] as f32;
                    corr.image_data[(y as usize + r) * corr_step + x as usize + c] = t0;
                }
            }
            x += blocksize.width;
        }
        y += blocksize.height;
    }
}

/// `cvMatchTemplate( const CvArr* _img, const CvArr* _templ, CvArr*
/// _result, int method )` (`cvtemplmatch.cpp:297`) for one-channel
/// `IPL_DEPTH_32F` images.
pub fn cv_match_template(img: &IplImage, templ: &IplImage, result: &mut IplImage, method: i32) {
    let num_type = if method == CV_TM_CCORR || method == CV_TM_CCORR_NORMED {
        0
    } else if method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED {
        1
    } else {
        2
    };
    let is_normed = method == CV_TM_CCORR_NORMED
        || method == CV_TM_SQDIFF_NORMED
        || method == CV_TM_CCOEFF_NORMED;

    let (img, templ) = if img.height < templ.height || img.width < templ.width {
        (templ, img)
    } else {
        (img, templ)
    };

    if result.height != img.height - templ.height + 1 || result.width != img.width - templ.width + 1
    {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "cvMatchTemplate",
            "output image should be (W - w + 1)x(H - h + 1)",
            "cvtemplmatch.cpp",
            348,
        );
    }

    if method < CV_TM_SQDIFF || method > CV_TM_CCOEFF_NORMED {
        cv_error(
            CV_STS_BAD_ARG,
            "cvMatchTemplate",
            "unknown comparison method",
            "cvtemplmatch.cpp",
            351,
        );
    }

    // `is_normed && cn == 1 && templ->rows > 8 ...`: the IPP function
    // pointers are null, so the generic code below always runs.

    icv_cross_corr(img, templ, result);

    if method == CV_TM_CCORR {
        return;
    }

    let inv_area = 1. / (templ.height as f64 * templ.width as f64);

    let mut sum = CvMat::create(img.height + 1, img.width + 1, CV_64FC1);
    let mut templ_mean;
    let mut templ_norm = 0f64;
    let mut templ_sum2 = 0f64;
    let sqsum;
    if method == CV_TM_CCOEFF {
        panic!("cvMatchTemplate: CV_TM_CCOEFF (cvIntegral without a squared sum) is not reached");
    } else {
        let templ_sdv;
        let mut sq = CvMat::create(img.height + 1, img.width + 1, CV_64FC1);
        cv_integral(img, &mut sum, &mut sq);
        (templ_mean, templ_sdv) = cv_avg_sdv(templ);

        templ_norm = templ_sdv.val[0] * templ_sdv.val[0]
            + templ_sdv.val[1] * templ_sdv.val[1]
            + templ_sdv.val[2] * templ_sdv.val[2]
            + templ_sdv.val[3] * templ_sdv.val[3];

        if templ_norm < f64::EPSILON && method == CV_TM_CCOEFF_NORMED {
            // `cvSet( result, cvScalarAll(1.) )`
            result.image_data.fill(1.);
            return;
        }

        templ_sum2 = templ_norm
            + templ_mean.val[0] * templ_mean.val[0]
            + templ_mean.val[1] * templ_mean.val[1]
            + templ_mean.val[2] * templ_mean.val[2]
            + templ_mean.val[3] * templ_mean.val[3];

        if num_type != 1 {
            templ_mean = CvScalar::all(0.);
            templ_norm = templ_sum2;
        }

        templ_sum2 /= inv_area;
        templ_norm = templ_norm.sqrt();
        templ_norm /= inv_area.sqrt(); // care of accuracy here
        sqsum = sq;
    }

    let sum_step = (sum.step / 8) as usize;
    let sqsum_step = (sqsum.step / 8) as usize;
    // p0..p3 and q0..q3 as offsets into `sum.data` / `sqsum.data`
    let p0 = 0usize;
    let p1 = p0 + templ.width as usize;
    let p2 = templ.height as usize * sum_step;
    let p3 = p2 + templ.width as usize;
    let q0 = 0usize;
    let q1 = q0 + templ.width as usize;
    let q2 = templ.height as usize * sqsum_step;
    let q3 = q2 + templ.width as usize;
    let sd = &sum.data;
    let qd = &sqsum.data;
    let rstep = (result.width_step / 4) as usize;

    for i in 0..result.height as usize {
        let mut idx = i * sum_step;
        let mut idx2 = i * sqsum_step;

        for j in 0..result.width as usize {
            let rr = i * rstep + j;
            let mut num = result.image_data[rr] as f64;
            let mut t;
            let mut wnd_mean2 = 0f64;
            let mut wnd_sum2 = 0f64;

            if num_type == 1 {
                t = sd[p0 + idx] - sd[p1 + idx] - sd[p2 + idx] + sd[p3 + idx];
                wnd_mean2 += t * t;
                num -= t * templ_mean.val[0];

                wnd_mean2 *= inv_area;
            }

            if is_normed || num_type == 2 {
                t = qd[q0 + idx2] - qd[q1 + idx2] - qd[q2 + idx2] + qd[q3 + idx2];
                wnd_sum2 += t;

                if num_type == 2 {
                    num = wnd_sum2 - 2. * num + templ_sum2;
                }
            }

            if is_normed {
                let d = wnd_sum2 - wnd_mean2;
                t = (if d < 0. { 0. } else { d }).sqrt() * templ_norm;
                if t > f64::EPSILON {
                    num /= t;
                    if num.abs() > 1. {
                        num = if num > 0. { 1. } else { -1. };
                    }
                } else {
                    num = if method != CV_TM_SQDIFF_NORMED || num < f64::EPSILON {
                        0.
                    } else {
                        1.
                    };
                }
            }

            result.image_data[rr] = num as f32;
            idx += 1;
            idx2 += 1;
        }
    }
}
