//! Pixel summation, non-zero counting, and reduction from
//! `IMOD/raptor/opencv/cxsumpixels.cpp`.

use super::cvutils::CvSize;
use super::cxmean::{CvMeanMatrix, CvMeanValue, CvScalar};

/// Errors represented by the public C entry points in this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvSumError {
    BadArgument,
    BadNumChannels,
    OutOfRange,
    UnsupportedFormat,
}

/// C `CV_REDUCE_*` operation selected by [`cv_reduce`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvReduceOp {
    Sum,
    Average,
    Maximum,
    Minimum,
}

/// Source depths installed in `cvSum`'s dispatch tables.  In particular,
/// OpenCV 1.x leaves the `CV_8S` table entries null.
pub trait CvSumValue: CvMeanValue {}
impl CvSumValue for u8 {}
impl CvSumValue for u16 {}
impl CvSumValue for i16 {}
impl CvSumValue for i32 {}
impl CvSumValue for f32 {}
impl CvSumValue for f64 {}

/// Values with the same non-zero test used by `cvCountNonZero`.
pub trait CvNonZero: Copy {
    fn is_cv_non_zero(self) -> bool;
}

macro_rules! impl_non_zero {
    ($type:ty) => {
        impl CvNonZero for $type {
            fn is_cv_non_zero(self) -> bool {
                self != 0 as $type
            }
        }
    };
}

impl_non_zero!(u8);
impl_non_zero!(i8);
impl_non_zero!(u16);
impl_non_zero!(i16);
impl_non_zero!(i32);
impl_non_zero!(f32);
impl_non_zero!(f64);

/// Owned generic translation of `cvSum`.
pub fn cv_sum<T: CvSumValue>(
    matrix: &CvMeanMatrix<T>,
    coi: Option<usize>,
) -> Result<CvScalar, CvSumError> {
    let pixels = matrix
        .size
        .width
        .checked_mul(matrix.size.height)
        .ok_or(CvSumError::BadArgument)?;
    if matrix.channels == 0
        || matrix.data.len()
            != pixels
                .checked_mul(matrix.channels)
                .ok_or(CvSumError::BadArgument)?
    {
        return Err(CvSumError::BadArgument);
    }
    let selected_channel = coi.unwrap_or(0);
    if selected_channel > matrix.channels {
        return Err(CvSumError::OutOfRange);
    }
    if selected_channel == 0 && matrix.channels > 4 {
        return Err(CvSumError::OutOfRange);
    }
    let mut sum = CvScalar { values: [0.0; 4] };
    for pixel in 0..pixels {
        if selected_channel == 0 {
            for channel in 0..matrix.channels {
                sum.values[channel] += matrix.data[pixel * matrix.channels + channel].to_mean_f64();
            }
        } else {
            sum.values[0] +=
                matrix.data[pixel * matrix.channels + selected_channel - 1].to_mean_f64();
        }
    }
    Ok(sum)
}

/// Owned generic translation of `cvCountNonZero`.
pub fn cv_count_non_zero<T: CvNonZero>(
    matrix: &CvMeanMatrix<T>,
    coi: Option<usize>,
) -> Result<usize, CvSumError> {
    let pixels = matrix
        .size
        .width
        .checked_mul(matrix.size.height)
        .ok_or(CvSumError::BadArgument)?;
    if matrix.channels == 0
        || matrix.data.len()
            != pixels
                .checked_mul(matrix.channels)
                .ok_or(CvSumError::BadArgument)?
    {
        return Err(CvSumError::BadArgument);
    }
    let selected_channel = coi.unwrap_or(0);
    if selected_channel == 0 && matrix.channels != 1 {
        return Err(CvSumError::BadNumChannels);
    }
    if selected_channel > matrix.channels {
        return Err(CvSumError::OutOfRange);
    }
    Ok((0..pixels)
        .filter(|&pixel| {
            matrix.data[pixel * matrix.channels + selected_channel.saturating_sub(1)]
                .is_cv_non_zero()
        })
        .count())
}

/// Owned generic translation of `cvReduce`.
///
/// `dimension == 0` reduces all rows into one row; `dimension == 1` reduces
/// each row into one column.  As in the C implementation, channels are kept
/// separate.  The output is `f64`, covering every C source/destination depth
/// combination without pointer casts or temporary `CvMat` allocations.
pub fn cv_reduce<T: CvSumValue>(
    source: &CvMeanMatrix<T>,
    dimension: usize,
    operation: CvReduceOp,
) -> Result<CvMeanMatrix<f64>, CvSumError> {
    let pixels = source
        .size
        .width
        .checked_mul(source.size.height)
        .ok_or(CvSumError::BadArgument)?;
    if dimension > 1
        || source.size.width == 0
        || source.size.height == 0
        || source.channels == 0
        || source.data.len()
            != pixels
                .checked_mul(source.channels)
                .ok_or(CvSumError::BadArgument)?
    {
        return Err(CvSumError::BadArgument);
    }
    let (output_size, reductions) = if dimension == 0 {
        (
            CvSize {
                width: source.size.width,
                height: 1,
            },
            source.size.height,
        )
    } else {
        (
            CvSize {
                width: 1,
                height: source.size.height,
            },
            source.size.width,
        )
    };
    let mut output = CvMeanMatrix {
        size: output_size,
        channels: source.channels,
        data: vec![0.0; output_size.width * output_size.height * source.channels],
    };
    for output_y in 0..output_size.height {
        for output_x in 0..output_size.width {
            for channel in 0..source.channels {
                let first = if dimension == 0 {
                    source.data[output_x * source.channels + channel].to_mean_f64()
                } else {
                    source.data[output_y * source.size.width * source.channels + channel]
                        .to_mean_f64()
                };
                let mut value = first;
                for reduced in 1..reductions {
                    let source_index = if dimension == 0 {
                        (reduced * source.size.width + output_x) * source.channels + channel
                    } else {
                        (output_y * source.size.width + reduced) * source.channels + channel
                    };
                    let candidate = source.data[source_index].to_mean_f64();
                    value = match operation {
                        CvReduceOp::Sum | CvReduceOp::Average => value + candidate,
                        CvReduceOp::Maximum => value.max(candidate),
                        CvReduceOp::Minimum => value.min(candidate),
                    };
                }
                if operation == CvReduceOp::Average && reductions != 0 {
                    value /= reductions as f64;
                }
                output.data
                    [(output_y * output_size.width + output_x) * source.channels + channel] = value;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cv_sum_separates_channels_and_supports_coi() {
        let matrix = CvMeanMatrix {
            size: CvSize {
                width: 2,
                height: 2,
            },
            channels: 3,
            data: vec![1_i32, 10, 100, 2, 20, 200, 3, 30, 300, 4, 40, 400],
        };
        assert_eq!(
            cv_sum(&matrix, None).unwrap().values,
            [10.0, 100.0, 1000.0, 0.0]
        );
        assert_eq!(
            cv_sum(&matrix, Some(2)).unwrap().values,
            [100.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn count_nonzero_enforces_the_c_single_channel_contract() {
        let matrix = CvMeanMatrix {
            size: CvSize {
                width: 3,
                height: 1,
            },
            channels: 1,
            data: vec![0_i16, -1, 2],
        };
        assert_eq!(cv_count_non_zero(&matrix, None), Ok(2));
        let multi = CvMeanMatrix {
            size: CvSize {
                width: 1,
                height: 1,
            },
            channels: 2,
            data: vec![0_u8, 3],
        };
        assert_eq!(
            cv_count_non_zero(&multi, None),
            Err(CvSumError::BadNumChannels)
        );
        assert_eq!(cv_count_non_zero(&multi, Some(2)), Ok(1));
    }

    #[test]
    fn reduce_matches_sum_average_minimum_and_maximum_dimensions() {
        let matrix = CvMeanMatrix {
            size: CvSize {
                width: 3,
                height: 2,
            },
            channels: 1,
            data: vec![1_f32, 5., 3., 2., 4., 6.],
        };
        assert_eq!(
            cv_reduce(&matrix, 0, CvReduceOp::Sum).unwrap().data,
            [3., 9., 9.]
        );
        assert_eq!(
            cv_reduce(&matrix, 1, CvReduceOp::Average).unwrap().data,
            [3., 4.]
        );
        assert_eq!(
            cv_reduce(&matrix, 0, CvReduceOp::Minimum).unwrap().data,
            [1., 4., 3.]
        );
        assert_eq!(
            cv_reduce(&matrix, 1, CvReduceOp::Maximum).unwrap().data,
            [5., 6.]
        );
    }

    #[test]
    fn reduce_rejects_empty_input_instead_of_indexing_a_seed_value() {
        let empty: CvMeanMatrix<f64> = CvMeanMatrix {
            size: CvSize {
                width: 0,
                height: 1,
            },
            channels: 1,
            data: vec![],
        };
        assert_eq!(
            cv_reduce(&empty, 0, CvReduceOp::Sum),
            Err(CvSumError::BadArgument)
        );
    }
}
