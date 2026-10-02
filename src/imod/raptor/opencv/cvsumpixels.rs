//! Translation of `IMOD/raptor/opencv/cvsumpixels.cpp` (the parts RAPTOR
//! reaches): `cvIntegral` of a one-channel `IPL_DEPTH_32F` image into
//! `CV_64FC1` sum and squared-sum images, without the tilted sum
//! (`icvIntegralImage_32f64f_C1R`).
//!
//! `icvInitIntegralImageTable` (the depth dispatch table) collapses to the one
//! kernel.

use super::cxerror::{CV_STS_UNMATCHED_SIZES, cv_error};
use super::cxtypes::*;

/// `icvIntegralImage_32f64f_C1R` (`ICV_DEF_INTEGRAL_OP_C1( 32f64f, float,
/// double, double, double, CV_NOP, CV_SQR )`, `cvsumpixels.cpp:44`), the
/// `sqsum != 0 && tilted == 0` branch.  Steps in elements.
fn icv_integral_image_32f64f_c1r(
    src: &[f32],
    srcstep: usize,
    sum: &mut [f64],
    sumstep: usize,
    sqsum: &mut [f64],
    sqsumstep: usize,
    size: CvSize,
) {
    let width = size.width as usize;

    for v in sum[..width + 1].iter_mut() {
        *v = 0.;
    }
    let mut so = sumstep + 1;

    for v in sqsum[..width + 1].iter_mut() {
        *v = 0.;
    }
    let mut qo = sqsumstep + 1;
    let mut o = 0usize;

    for _ in 0..size.height {
        sum[so - 1] = 0.;
        sqsum[qo - 1] = 0.;

        let mut s = 0f64;
        let mut sq = 0f64;
        for x in 0..width {
            let it = src[o + x] as f64;
            let mut t = it;
            let mut tq = it * it;
            s += t;
            sq += tq;
            t = sum[so + x - sumstep] + s;
            tq = sqsum[qo + x - sqsumstep] + sq;
            sum[so + x] = t;
            sqsum[qo + x] = tq;
        }
        o += srcstep;
        so += sumstep;
        qo += sqsumstep;
    }
}

/// `cvIntegral( const CvArr* image, CvArr* sumImage, CvArr* sumSqImage,
/// CvArr* tiltedSumImage )` (`cvsumpixels.cpp:311`) with a squared sum and
/// no tilted sum, as `cvMatchTemplate` calls it.
pub fn cv_integral(image: &IplImage, sum: &mut CvMat, sqsum: &mut CvMat) {
    if sum.cols != image.width + 1 || sum.rows != image.height + 1 {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "cvIntegralImage",
            "",
            "cvsumpixels.cpp",
            344,
        );
    }
    if sum.rows != sqsum.rows || sum.cols != sqsum.cols {
        cv_error(
            CV_STS_UNMATCHED_SIZES,
            "cvIntegralImage",
            "",
            "cvsumpixels.cpp",
            357,
        );
    }

    let size = image.get_size();
    let src_step = (image.width_step / 4) as usize;
    let sum_step = (sum.step / 8) as usize;
    let sqsum_step = (sqsum.step / 8) as usize;

    icv_integral_image_32f64f_c1r(
        &image.image_data,
        src_step,
        &mut sum.data,
        sum_step,
        &mut sqsum.data,
        sqsum_step,
        size,
    );
}
