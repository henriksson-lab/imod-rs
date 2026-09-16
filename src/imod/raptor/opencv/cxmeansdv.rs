//! Owned generic translation of `IMOD/raptor/opencv/cxmeansdv.cpp`.
//!
//! Its C macro/table families differ only in scalar depth, channel count,
//! masking, and COI.  `cv_avg_sdv` expresses every generated kernel with
//! checked owned matrices while retaining the source's population deviation
//! calculation and zero result for an empty mask.

use super::cxerror::CvStatus;
use super::cxmean::CvScalar;
use super::cxminmaxloc::{CvMask, CvMatrix};

/// Scalar depths supported by the source's `Mean_StdDev` dispatch tables.
pub trait CvMeanSdvValue: Copy {
    fn to_mean_sdv_f64(self) -> f64;
}

macro_rules! impl_mean_sdv_value {
    ($($type:ty),+ $(,)?) => {
        $(
            impl CvMeanSdvValue for $type {
                fn to_mean_sdv_f64(self) -> f64 { self as f64 }
            }
        )+
    };
}

impl_mean_sdv_value!(u8, u16, i16, i32, f32, f64);

/// C `cvAvgSdv` plus every macro-generated `icvMean_StdDev_*` kernel.
///
/// `coi` selects C's one-based channel of interest. Without it, the original
/// supports at most four channels because `CvScalar` has four lanes.
pub fn cv_avg_sdv<T: CvMeanSdvValue>(
    matrix: &CvMatrix<T>,
    mask: Option<&CvMask>,
    coi: Option<usize>,
) -> Result<(CvScalar, CvScalar), CvStatus> {
    if coi.is_none() && matrix.channels > 4 {
        return Err(CvStatus::sts_out_of_range);
    }
    let selected_channel = match coi {
        Some(channel) if channel == 0 || channel > matrix.channels => {
            return Err(CvStatus::sts_bad_arg);
        }
        Some(channel) => Some(channel - 1),
        None => None,
    };
    if let Some(mask) = mask {
        if mask.row_stride < matrix.columns
            || mask.values.len() < (matrix.rows - 1) * mask.row_stride + matrix.columns
        {
            return Err(CvStatus::sts_bad_mask);
        }
    }

    let channels = if selected_channel.is_some() {
        1
    } else {
        matrix.channels
    };
    let mut sum = [0.0; 4];
    let mut square_sum = [0.0; 4];
    let mut pixels = 0usize;
    for row in 0..matrix.rows {
        for column in 0..matrix.columns {
            if mask.is_some_and(|mask| mask.values[row * mask.row_stride + column] == 0) {
                continue;
            }
            pixels += 1;
            let start = row * matrix.row_stride + column * matrix.channels;
            for output_channel in 0..channels {
                let input_channel = selected_channel.unwrap_or(output_channel);
                let value = matrix.values[start + input_channel].to_mean_sdv_f64();
                sum[output_channel] += value;
                square_sum[output_channel] += value * value;
            }
        }
    }

    let scale = if pixels == 0 {
        0.0
    } else {
        1.0 / pixels as f64
    };
    let mut mean = CvScalar { values: [0.0; 4] };
    let mut standard_deviation = CvScalar { values: [0.0; 4] };
    for channel in 0..channels {
        mean.values[channel] = sum[channel] * scale;
        standard_deviation.values[channel] = (square_sum[channel] * scale
            - mean.values[channel] * mean.values[channel])
            .max(0.0)
            .sqrt();
    }
    Ok((mean, standard_deviation))
}

#[cfg(test)]
mod tests {
    use super::cv_avg_sdv;
    use crate::imod::raptor::opencv::cxerror::CvStatus;
    use crate::imod::raptor::opencv::cxminmaxloc::{CvMask, CvMatrix};

    #[test]
    fn computes_population_mean_and_deviation_per_channel_with_stride() {
        let matrix = CvMatrix::new(2, 2, 2, 5, vec![1_i16, 2, 3, 4, 99, 5, 6, 7, 8, 99]).unwrap();
        let (mean, sdv) = cv_avg_sdv(&matrix, None, None).unwrap();
        assert_eq!(mean.values, [4.0, 5.0, 0.0, 0.0]);
        assert!((sdv.values[0] - (5.0_f64).sqrt()).abs() < 1e-12);
        assert!((sdv.values[1] - (5.0_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn mask_and_coi_follow_source_selection_rules() {
        let matrix =
            CvMatrix::new(2, 2, 3, 6, vec![1_u8, 10, 3, 2, 20, 4, 5, 30, 7, 6, 40, 8]).unwrap();
        let mask = CvMask::new(2, 2, 2, vec![1, 0, 1, 0]).unwrap();
        let (mean, sdv) = cv_avg_sdv(&matrix, Some(&mask), Some(2)).unwrap();
        assert_eq!(mean.values, [20.0, 0.0, 0.0, 0.0]);
        assert_eq!(sdv.values, [10.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn empty_mask_and_invalid_channel_paths_match_c_results() {
        let matrix = CvMatrix::new(1, 2, 1, 2, vec![3_f32, 7.0]).unwrap();
        let mask = CvMask::new(1, 2, 2, vec![0, 0]).unwrap();
        let (mean, sdv) = cv_avg_sdv(&matrix, Some(&mask), None).unwrap();
        assert_eq!(mean.values, [0.0; 4]);
        assert_eq!(sdv.values, [0.0; 4]);
        assert_eq!(
            cv_avg_sdv(&matrix, None, Some(2)),
            Err(CvStatus::sts_bad_arg)
        );
    }

    #[test]
    fn rejects_more_than_four_channels_without_coi() {
        let matrix = CvMatrix::new(1, 1, 5, 5, vec![1_i32; 5]).unwrap();
        assert_eq!(
            cv_avg_sdv(&matrix, None, None),
            Err(CvStatus::sts_out_of_range)
        );
        assert_eq!(cv_avg_sdv(&matrix, None, Some(5)).unwrap().0.values[0], 1.0);
    }
}
