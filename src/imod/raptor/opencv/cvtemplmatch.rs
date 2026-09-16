//! Owned translation of `IMOD/raptor/opencv/cvtemplmatch.cpp`.

use super::cvutils::CvPoint;
use super::cxconvert::CvChannelMatrix;
use super::cxutils::CvUtilsError;

pub const CV_TM_SQDIFF: i32 = 0;
pub const CV_TM_SQDIFF_NORMED: i32 = 1;
pub const CV_TM_CCORR: i32 = 2;
pub const CV_TM_CCORR_NORMED: i32 = 3;
pub const CV_TM_CCOEFF: i32 = 4;
pub const CV_TM_CCOEFF_NORMED: i32 = 5;

/// Scalar depths admitted by `cvMatchTemplate` (source: 8u and 32f).
pub trait CvTemplateValue: Copy {
    fn value(self) -> f64;
}
impl CvTemplateValue for u8 {
    fn value(self) -> f64 {
        self as f64
    }
}
impl CvTemplateValue for f32 {
    fn value(self) -> f64 {
        self as f64
    }
}

/// C `icvCrossCorr`.  `anchor` locates template element (0,0) in the output;
/// the usual valid correlation is [`cv_match_template`] with anchor (0,0).
pub fn icv_cross_corr<T: CvTemplateValue>(
    image: &CvChannelMatrix<T>,
    template: &CvChannelMatrix<T>,
    output: &mut CvChannelMatrix<f32>,
    anchor: CvPoint,
) -> Result<(), CvUtilsError> {
    if image.channels != template.channels
        || output.channels != 1
        || image.channels == 0
        || template.rows > image.rows
        || template.cols > image.cols
    {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    for y in 0..output.rows {
        for x in 0..output.cols {
            let mut sum = 0.0;
            for ty in 0..template.rows {
                for tx in 0..template.cols {
                    let iy = y as i32 + ty as i32 - anchor.y;
                    let ix = x as i32 + tx as i32 - anchor.x;
                    if iy >= 0 && ix >= 0 && iy < image.rows as i32 && ix < image.cols as i32 {
                        for channel in 0..image.channels {
                            sum += image.data[(iy as usize * image.cols + ix as usize)
                                * image.channels
                                + channel]
                                .value()
                                * template.data
                                    [(ty * template.cols + tx) * template.channels + channel]
                                    .value();
                        }
                    }
                }
            }
            output.data[y * output.cols + x] = sum as f32;
        }
    }
    Ok(())
}

/// C `cvMatchTemplate`, including all six source comparison methods. Output
/// is a one-channel f32 valid-correlation matrix.
pub fn cv_match_template<T: CvTemplateValue>(
    image: &CvChannelMatrix<T>,
    template: &CvChannelMatrix<T>,
    result: &mut CvChannelMatrix<f32>,
    method: i32,
) -> Result<(), CvUtilsError> {
    if !(CV_TM_SQDIFF..=CV_TM_CCOEFF_NORMED).contains(&method) {
        return Err(CvUtilsError::BadArgument);
    }
    if image.channels != template.channels
        || image.rows < template.rows
        || image.cols < template.cols
        || result.channels != 1
        || result.rows != image.rows - template.rows + 1
        || result.cols != image.cols - template.cols + 1
    {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    let count = (template.rows * template.cols) as f64;
    let mut means = vec![0.0; image.channels];
    let mut template_energy = 0.0;
    for channel in 0..image.channels {
        for y in 0..template.rows {
            for x in 0..template.cols {
                means[channel] +=
                    template.data[(y * template.cols + x) * image.channels + channel].value();
            }
        }
        means[channel] /= count;
        for y in 0..template.rows {
            for x in 0..template.cols {
                let value =
                    template.data[(y * template.cols + x) * image.channels + channel].value();
                let adjusted = if method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED {
                    value - means[channel]
                } else {
                    value
                };
                template_energy += adjusted * adjusted;
            }
        }
    }
    if method == CV_TM_CCOEFF_NORMED && template_energy < f64::EPSILON {
        result.data.fill(1.0);
        return Ok(());
    }
    for oy in 0..result.rows {
        for ox in 0..result.cols {
            let mut cross = 0.0;
            let mut image_energy = 0.0;
            let mut window_means = vec![0.0; image.channels];
            if method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED {
                for channel in 0..image.channels {
                    for y in 0..template.rows {
                        for x in 0..template.cols {
                            window_means[channel] += image.data
                                [((oy + y) * image.cols + ox + x) * image.channels + channel]
                                .value();
                        }
                    }
                    window_means[channel] /= count;
                }
            }
            for channel in 0..image.channels {
                for y in 0..template.rows {
                    for x in 0..template.cols {
                        let image_value = image.data
                            [((oy + y) * image.cols + ox + x) * image.channels + channel]
                            .value();
                        let template_value = template.data
                            [(y * template.cols + x) * image.channels + channel]
                            .value();
                        if method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED {
                            let adjusted_image = image_value - window_means[channel];
                            cross += adjusted_image * (template_value - means[channel]);
                            image_energy += adjusted_image * adjusted_image;
                        } else {
                            cross += image_value * template_value;
                            image_energy += image_value * image_value;
                        }
                    }
                }
            }
            let mut value = match method {
                CV_TM_SQDIFF | CV_TM_SQDIFF_NORMED => image_energy - 2.0 * cross + template_energy,
                _ => cross,
            };
            if method == CV_TM_SQDIFF_NORMED
                || method == CV_TM_CCORR_NORMED
                || method == CV_TM_CCOEFF_NORMED
            {
                let denominator = (image_energy.max(0.0)).sqrt() * template_energy.sqrt();
                value = if denominator > f64::EPSILON {
                    (value / denominator).clamp(-1.0, 1.0)
                } else if method == CV_TM_SQDIFF_NORMED && value >= f64::EPSILON {
                    1.0
                } else {
                    0.0
                };
            }
            result.data[oy * result.cols + ox] = value as f32;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_template_methods_find_exact_patch() {
        let image = CvChannelMatrix::new(3, 3, 1, vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let template = CvChannelMatrix::new(2, 2, 1, vec![4u8, 5, 7, 8]).unwrap();
        let mut result = CvChannelMatrix::new(2, 2, 1, vec![0.; 4]).unwrap();
        cv_match_template(&image, &template, &mut result, CV_TM_SQDIFF).unwrap();
        assert_eq!(result.data[3], 0.);
        cv_match_template(&image, &template, &mut result, CV_TM_CCORR_NORMED).unwrap();
        assert_eq!(result.data[3], 1.);
        cv_match_template(&image, &template, &mut result, CV_TM_CCOEFF_NORMED).unwrap();
        assert_eq!(result.data[3], 1.);
    }
}
