//! Owned generic translation of `IMOD/raptor/opencv/cvsumpixels.cpp`.
//!
//! Despite its filename this source implements `cvIntegral`. Its C depth and
//! channel dispatch tables become one generic integral-image calculation over
//! the shared strided matrix representation.

use super::cxerror::CvStatus;
use super::cxminmaxloc::CvMatrix;

/// Output images returned by C `cvIntegral`.
#[derive(Clone, Debug, PartialEq)]
pub struct CvIntegralImages {
    /// Always present `(rows + 1) × (columns + 1)` summed image.
    pub sum: CvMatrix<f64>,
    /// Present when C `sumSqImage` was supplied.
    pub squared_sum: Option<CvMatrix<f64>>,
    /// Present when C `tiltedSumImage` was supplied; it is single-channel.
    pub tilted_sum: Option<CvMatrix<f64>>,
}

/// Source scalar depths supported by `icvIntegralImage_*` tables.
pub trait CvIntegralValue: Copy {
    fn to_integral_f64(self) -> f64;
}

macro_rules! impl_integral_value {
    ($($type:ty),+ $(,)?) => {
        $(
            impl CvIntegralValue for $type {
                fn to_integral_f64(self) -> f64 { self as f64 }
            }
        )+
    };
}

impl_integral_value!(u8, f32, f64);

/// C `cvIntegral` and all generated `icvIntegralImage_*` kernels.
///
/// The output depth is consistently `f64`: C allows a special `u8 → i32`
/// sum output, but its values are exactly represented here without retaining
/// a runtime C depth tag. `tilted_sum` requires `squared_sum`, as in C.
pub fn cv_integral<T: CvIntegralValue>(
    source: &CvMatrix<T>,
    squared_sum: bool,
    tilted_sum: bool,
) -> Result<CvIntegralImages, CvStatus> {
    if tilted_sum && !squared_sum {
        return Err(CvStatus::sts_null_ptr);
    }
    if tilted_sum && source.channels != 1 {
        return Err(CvStatus::sts_not_implemented);
    }
    let output_columns = source
        .columns
        .checked_add(1)
        .ok_or(CvStatus::sts_unmatched_sizes)?;
    let output_rows = source
        .rows
        .checked_add(1)
        .ok_or(CvStatus::sts_unmatched_sizes)?;
    let output_stride = output_columns
        .checked_mul(source.channels)
        .ok_or(CvStatus::sts_unmatched_sizes)?;
    let output_len = output_rows
        .checked_mul(output_stride)
        .ok_or(CvStatus::sts_unmatched_sizes)?;
    let mut sum = CvMatrix::new(
        output_rows,
        output_columns,
        source.channels,
        output_stride,
        vec![0.0; output_len],
    )?;
    let mut square = squared_sum.then(|| {
        CvMatrix::new(
            output_rows,
            output_columns,
            source.channels,
            output_stride,
            vec![0.0; output_len],
        )
        .expect("validated integral output dimensions")
    });

    for row in 0..source.rows {
        for column in 0..source.columns {
            for channel in 0..source.channels {
                let source_value = source.values
                    [row * source.row_stride + column * source.channels + channel]
                    .to_integral_f64();
                let index = (row + 1) * output_stride + (column + 1) * source.channels + channel;
                let above = row * output_stride + (column + 1) * source.channels + channel;
                let left = (row + 1) * output_stride + column * source.channels + channel;
                let diagonal = row * output_stride + column * source.channels + channel;
                sum.values[index] =
                    source_value + sum.values[above] + sum.values[left] - sum.values[diagonal];
                if let Some(square) = &mut square {
                    square.values[index] =
                        source_value * source_value + square.values[above] + square.values[left]
                            - square.values[diagonal];
                }
            }
        }
    }

    let tilted_sum = if tilted_sum {
        let mut tilted = CvMatrix::new(
            output_rows,
            output_columns,
            1,
            output_columns,
            vec![0.0; output_rows * output_columns],
        )?;
        let mut buffer = vec![0.0; source.columns + 1];
        for column in 0..source.columns {
            let value = source.values[column].to_integral_f64();
            buffer[column] = value;
            tilted.values[output_columns + column + 1] = value;
        }
        if source.columns == 1 {
            buffer[1] = 0.0;
        }
        for row in 1..source.rows {
            let row_start = row * source.row_stride;
            let output_start = (row + 1) * output_columns;
            let previous_output_start = row * output_columns;
            let mut current = source.values[row_start].to_integral_f64();
            tilted.values[output_start + 1] =
                tilted.values[previous_output_start + 1] + current + buffer[1];
            for column in 1..source.columns.saturating_sub(1) {
                let prior = buffer[column];
                buffer[column - 1] = prior + current;
                current = source.values[row_start + column].to_integral_f64();
                tilted.values[output_start + column + 1] = prior
                    + buffer[column + 1]
                    + current
                    + tilted.values[previous_output_start + column];
            }
            if source.columns > 1 {
                let column = source.columns - 1;
                let prior = buffer[column];
                buffer[column - 1] = prior + current;
                current = source.values[row_start + column].to_integral_f64();
                tilted.values[output_start + column + 1] =
                    current + prior + tilted.values[previous_output_start + column];
                buffer[column] = current;
            }
        }
        Some(tilted)
    } else {
        None
    };
    Ok(CvIntegralImages {
        sum,
        squared_sum: square,
        tilted_sum,
    })
}

#[cfg(test)]
mod tests {
    use super::cv_integral;
    use crate::imod::raptor::opencv::cxerror::CvStatus;
    use crate::imod::raptor::opencv::cxminmaxloc::CvMatrix;

    #[test]
    fn computes_padded_sum_and_square_integral_images() {
        let source = CvMatrix::new(2, 2, 1, 2, vec![1_u8, 2, 3, 4]).unwrap();
        let images = cv_integral(&source, true, false).unwrap();
        assert_eq!(images.sum.values, [0., 0., 0., 0., 1., 3., 0., 4., 10.]);
        assert_eq!(
            images.squared_sum.unwrap().values,
            [0., 0., 0., 0., 1., 5., 0., 10., 30.]
        );
    }

    #[test]
    fn computes_each_channel_independently_with_source_stride() {
        let source = CvMatrix::new(1, 2, 2, 5, vec![1_f32, 10., 2., 20., 99.]).unwrap();
        let images = cv_integral(&source, false, false).unwrap();
        assert_eq!(
            images.sum.values,
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 10., 3., 30.]
        );
    }

    #[test]
    fn tilted_requires_squared_sum_and_single_channel() {
        let single = CvMatrix::new(1, 2, 1, 2, vec![1_f64, 2.]).unwrap();
        let tilted = cv_integral(&single, true, true)
            .unwrap()
            .tilted_sum
            .unwrap();
        assert_eq!(tilted.values, [0., 0., 0., 0., 1., 2.]);
        assert_eq!(
            cv_integral(&single, false, true),
            Err(CvStatus::sts_null_ptr)
        );
        let multichannel = CvMatrix::new(1, 1, 2, 2, vec![1_u8, 2]).unwrap();
        assert_eq!(
            cv_integral(&multichannel, true, true),
            Err(CvStatus::sts_not_implemented)
        );
    }
}
