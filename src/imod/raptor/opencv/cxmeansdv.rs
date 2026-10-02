//! Translation of `IMOD/raptor/opencv/cxmeansdv.cpp` (the parts RAPTOR
//! reaches): `cvAvgSdv` of a one-channel `IPL_DEPTH_32F` image without a
//! mask or COI (the template in `cvMatchTemplate`), through
//! `icvMean_StdDev_32f_C1R`.
//!
//! `icvInitMean_StdDevRTable` and its sibling tables collapse to the one
//! kernel.

use super::cxtypes::*;

/// `icvMean_StdDev_32f_C1R` (`ICV_DEF_MEAN_SDV_FUNC_2D( 32f, 1, float,
/// double, double, double )`, `cxmeansdv.cpp:416`).  `step` in elements.
fn icv_mean_std_dev_32f_c1r(src: &[f32], step: usize, size: CvSize, mean: &mut f64, sdv: &mut f64) {
    let mut s0 = 0f64;
    let mut sq0 = 0f64;
    let pix = size.width * size.height;
    let len = size.width as usize;
    let mut o = 0usize;

    for _ in 0..size.height {
        let mut x = 0usize;
        // ICV_MEAN_SDV_COI_CASE( double, double, CV_SQR, len, 1 )
        while x + 4 <= len {
            let mut t0 = src[o + x] as f64;
            let mut t1 = src[o + x + 1] as f64;

            s0 += t0 + t1;
            sq0 += t0 * t0 + t1 * t1;

            t0 = src[o + x + 2] as f64;
            t1 = src[o + x + 3] as f64;

            s0 += t0 + t1;
            sq0 += t0 * t0 + t1 * t1;
            x += 4;
        }

        while x < len {
            let t0 = src[o + x] as f64;

            s0 += t0;
            sq0 += t0 * t0;
            x += 1;
        }
        o += step;
    }

    // ICV_MEAN_SDV_EXIT_C1( s, sq )
    let scale = if pix != 0 { 1. / pix as f64 } else { 0. };
    let mut tmp = scale * s0;
    *mean = tmp;
    tmp = scale * sq0 - tmp * tmp;
    *sdv = (if tmp < 0. { 0. } else { tmp }).sqrt();
}

/// `cvAvgSdv( const CvArr* img, CvScalar* _mean, CvScalar* _sdv, const
/// void* mask )` (`cxmeansdv.cpp:676`) for a one-channel `IPL_DEPTH_32F`
/// image and no mask: returns (`mean`, `sdv`).
pub fn cv_avg_sdv(img: &IplImage) -> (CvScalar, CvScalar) {
    let mut mean = CvScalar::default();
    let mut sdv = CvScalar::default();

    // `cvGetMat` of the image: a one-row image, or one whose `widthStep` is
    // `width*4`, is continuous.
    let mut size = img.get_size();
    let min_step = if img.height <= 1 { 0 } else { img.width * 4 };
    let step = if img.height <= 1 { 0 } else { img.width_step };
    let mut mat_step = (img.width_step / 4) as usize;
    if step == min_step {
        size.width *= size.height;
        size.height = 1;
        mat_step = 0;
    }

    icv_mean_std_dev_32f_c1r(
        &img.image_data,
        mat_step,
        size,
        &mut mean.val[0],
        &mut sdv.val[0],
    );

    (mean, sdv)
}
