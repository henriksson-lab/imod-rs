//! Mean-value functions from `IMOD/raptor/opencv/cxmean.cpp`.

use super::cvutils::CvSize;

/// C `CvScalar`, represented by its four double-precision lanes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CvScalar {
    pub values: [f64; 4],
}

/// Owned contiguous matrix accepted by [`cv_avg`].
#[derive(Clone, Debug, PartialEq)]
pub struct CvMeanMatrix<T> {
    pub size: CvSize,
    pub channels: usize,
    pub data: Vec<T>,
}

/// Numeric source depths supported by the original mean dispatch tables.
pub trait CvMeanValue: Copy {
    const MASK_SUPPORTED: bool;
    fn to_mean_f64(self) -> f64;
}

macro_rules! impl_mean_value {
    ($type:ty, $masked:expr) => {
        impl CvMeanValue for $type {
            const MASK_SUPPORTED: bool = $masked;
            fn to_mean_f64(self) -> f64 {
                self as f64
            }
        }
    };
}

impl_mean_value!(u8, true);
impl_mean_value!(i8, false);
impl_mean_value!(u16, true);
impl_mean_value!(i16, true);
impl_mean_value!(i32, true);
impl_mean_value!(f32, true);
impl_mean_value!(f64, true);

/// C `CvStatus` failures reachable from the mean functions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvMeanError {
    BadArgument,
    BadMask,
    UnmatchedSizes,
    OutOfRange,
    UnsupportedFormat,
}

macro_rules! define_mean_channels {
    ($name:ident, $type:ty, $channels:expr) => {
        pub fn $name(
            src: &[$type],
            step: usize,
            mask: &[u8],
            mask_step: usize,
            size: CvSize,
            mean: &mut [f64; 4],
        ) -> Result<(), CvMeanError> {
            let row_width = size
                .width
                .checked_mul($channels)
                .ok_or(CvMeanError::BadArgument)?;
            if step < row_width
                || mask_step < size.width
                || src.len() < size.height.saturating_sub(1).saturating_mul(step) + row_width
                || mask.len() < size.height.saturating_sub(1).saturating_mul(mask_step) + size.width
            {
                return Err(CvMeanError::BadArgument);
            }
            let mut sums = [0.0; 4];
            let mut pixels = 0usize;
            for y in 0..size.height {
                for x in 0..size.width {
                    if mask[y * mask_step + x] != 0 {
                        for channel in 0..$channels {
                            sums[channel] += src[y * step + x * $channels + channel] as f64;
                        }
                        pixels += 1;
                    }
                }
            }
            let scale = if pixels == 0 {
                0.0
            } else {
                1.0 / pixels as f64
            };
            for channel in 0..$channels {
                mean[channel] = sums[channel] * scale;
            }
            Ok(())
        }
    };
}

macro_rules! define_mean_coi {
    ($name:ident, $type:ty) => {
        pub fn $name(
            src: &[$type],
            step: usize,
            mask: &[u8],
            mask_step: usize,
            size: CvSize,
            channels: usize,
            coi: usize,
            mean: &mut [f64; 4],
        ) -> Result<(), CvMeanError> {
            let row_width = size
                .width
                .checked_mul(channels)
                .ok_or(CvMeanError::BadArgument)?;
            if channels == 0
                || coi == 0
                || coi > channels
                || step < row_width
                || mask_step < size.width
                || src.len() < size.height.saturating_sub(1).saturating_mul(step) + row_width
                || mask.len() < size.height.saturating_sub(1).saturating_mul(mask_step) + size.width
            {
                return Err(CvMeanError::BadArgument);
            }
            let mut sum = 0.0;
            let mut pixels = 0usize;
            for y in 0..size.height {
                for x in 0..size.width {
                    if mask[y * mask_step + x] != 0 {
                        sum += src[y * step + x * channels + coi - 1] as f64;
                        pixels += 1;
                    }
                }
            }
            mean[0] = if pixels == 0 {
                0.0
            } else {
                sum / pixels as f64
            };
            Ok(())
        }
    };
}

// Rust macro identifiers cannot be concatenated on stable Rust, so the source
// macro expansion is listed by depth below.
define_mean_channels!(icv_mean_8u_c1mr, u8, 1);
define_mean_channels!(icv_mean_8u_c2mr, u8, 2);
define_mean_channels!(icv_mean_8u_c3mr, u8, 3);
define_mean_channels!(icv_mean_8u_c4mr, u8, 4);
define_mean_coi!(icv_mean_8u_cncmr, u8);
define_mean_channels!(icv_mean_16u_c1mr, u16, 1);
define_mean_channels!(icv_mean_16u_c2mr, u16, 2);
define_mean_channels!(icv_mean_16u_c3mr, u16, 3);
define_mean_channels!(icv_mean_16u_c4mr, u16, 4);
define_mean_coi!(icv_mean_16u_cncmr, u16);
define_mean_channels!(icv_mean_16s_c1mr, i16, 1);
define_mean_channels!(icv_mean_16s_c2mr, i16, 2);
define_mean_channels!(icv_mean_16s_c3mr, i16, 3);
define_mean_channels!(icv_mean_16s_c4mr, i16, 4);
define_mean_coi!(icv_mean_16s_cncmr, i16);
define_mean_channels!(icv_mean_32s_c1mr, i32, 1);
define_mean_channels!(icv_mean_32s_c2mr, i32, 2);
define_mean_channels!(icv_mean_32s_c3mr, i32, 3);
define_mean_channels!(icv_mean_32s_c4mr, i32, 4);
define_mean_coi!(icv_mean_32s_cncmr, i32);
define_mean_channels!(icv_mean_32f_c1mr, f32, 1);
define_mean_channels!(icv_mean_32f_c2mr, f32, 2);
define_mean_channels!(icv_mean_32f_c3mr, f32, 3);
define_mean_channels!(icv_mean_32f_c4mr, f32, 4);
define_mean_coi!(icv_mean_32f_cncmr, f32);
define_mean_channels!(icv_mean_64f_c1mr, f64, 1);
define_mean_channels!(icv_mean_64f_c2mr, f64, 2);
define_mean_channels!(icv_mean_64f_c3mr, f64, 3);
define_mean_channels!(icv_mean_64f_c4mr, f64, 4);
define_mean_coi!(icv_mean_64f_cncmr, f64);

/// Owned generic translation of `cvAvg`.
pub fn cv_avg<T: CvMeanValue>(
    matrix: &CvMeanMatrix<T>,
    mask: Option<&[u8]>,
    coi: Option<usize>,
) -> Result<CvScalar, CvMeanError> {
    let pixels = matrix
        .size
        .width
        .checked_mul(matrix.size.height)
        .ok_or(CvMeanError::BadArgument)?;
    let elements = pixels
        .checked_mul(matrix.channels)
        .ok_or(CvMeanError::BadArgument)?;
    if matrix.channels == 0 || matrix.data.len() != elements {
        return Err(CvMeanError::BadArgument);
    }
    if let Some(mask) = mask {
        if !T::MASK_SUPPORTED {
            return Err(CvMeanError::UnsupportedFormat);
        }
        if mask.len() != pixels {
            return Err(CvMeanError::UnmatchedSizes);
        }
        let selected_channel = coi.unwrap_or(0);
        if selected_channel != 0 && (selected_channel > matrix.channels) {
            return Err(CvMeanError::OutOfRange);
        }
        if selected_channel == 0 && matrix.channels > 4 {
            return Err(CvMeanError::OutOfRange);
        }
        let mut result = CvScalar { values: [0.0; 4] };
        let mut count = 0usize;
        for pixel in 0..pixels {
            if mask[pixel] != 0 {
                if selected_channel == 0 {
                    for channel in 0..matrix.channels {
                        result.values[channel] +=
                            matrix.data[pixel * matrix.channels + channel].to_mean_f64();
                    }
                } else {
                    result.values[0] +=
                        matrix.data[pixel * matrix.channels + selected_channel - 1].to_mean_f64();
                }
                count += 1;
            }
        }
        if count != 0 {
            let scale = 1.0 / count as f64;
            if selected_channel == 0 {
                for channel in 0..matrix.channels {
                    result.values[channel] *= scale;
                }
            } else {
                result.values[0] *= scale;
            }
        }
        Ok(result)
    } else {
        if matrix.channels > 4 {
            return Err(CvMeanError::OutOfRange);
        }
        let mut result = CvScalar { values: [0.0; 4] };
        for pixel in 0..pixels {
            for channel in 0..matrix.channels.min(4) {
                result.values[channel] +=
                    matrix.data[pixel * matrix.channels + channel].to_mean_f64();
            }
        }
        let scale = if pixels == 0 {
            0.0
        } else {
            1.0 / pixels as f64
        };
        for value in &mut result.values {
            *value *= scale;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_channel_kernels_match_the_c_mean_contract() {
        let source = [1_u8, 10, 2, 20, 3, 30];
        let mask = [1_u8, 0, 1];
        let mut mean = [0.0; 4];
        icv_mean_8u_c2mr(
            &source,
            6,
            &mask,
            3,
            CvSize {
                width: 3,
                height: 1,
            },
            &mut mean,
        )
        .unwrap();
        assert_eq!(mean, [2.0, 20.0, 0.0, 0.0]);
        icv_mean_8u_cncmr(
            &source,
            6,
            &mask,
            3,
            CvSize {
                width: 3,
                height: 1,
            },
            2,
            2,
            &mut mean,
        )
        .unwrap();
        assert_eq!(mean[0], 20.0);
    }

    #[test]
    fn cv_avg_handles_mask_and_unmasked_multichannel_data() {
        let matrix = CvMeanMatrix {
            size: CvSize {
                width: 2,
                height: 2,
            },
            channels: 3,
            data: vec![1_i16, 10, 100, 2, 20, 200, 3, 30, 300, 4, 40, 400],
        };
        assert_eq!(
            cv_avg(&matrix, None, None).unwrap().values,
            [2.5, 25.0, 250.0, 0.0]
        );
        assert_eq!(
            cv_avg(&matrix, Some(&[1, 0, 1, 0]), None).unwrap().values,
            [2.0, 20.0, 200.0, 0.0]
        );
        assert_eq!(
            cv_avg(&matrix, Some(&[1, 0, 1, 0]), Some(2))
                .unwrap()
                .values,
            [20.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn masked_i8_is_the_explicitly_unsupported_c_dispatch_case() {
        let matrix = CvMeanMatrix {
            size: CvSize {
                width: 1,
                height: 1,
            },
            channels: 1,
            data: vec![3_i8],
        };
        assert_eq!(
            cv_avg(&matrix, Some(&[1]), None),
            Err(CvMeanError::UnsupportedFormat)
        );
    }

    #[test]
    fn unmasked_more_than_four_channels_is_not_silently_truncated() {
        let matrix = CvMeanMatrix {
            size: CvSize {
                width: 1,
                height: 1,
            },
            channels: 5,
            data: vec![1_u8; 5],
        };
        assert_eq!(cv_avg(&matrix, None, None), Err(CvMeanError::OutOfRange));
    }
}
