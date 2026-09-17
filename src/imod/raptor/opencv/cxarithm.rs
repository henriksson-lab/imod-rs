//! Arithmetic entry points from `IMOD/raptor/opencv/cxarithm.cpp`.

use super::cxmean::{CvMeanMatrix, CvMeanValue, CvScalar};

/// C arithmetic failure paths represented without pointer error state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvArithmError {
    BadArgument,
    BadMask,
    UnmatchedFormats,
    UnmatchedSizes,
}

/// Saturating destination conversions from the source `CV_CAST_*` macros.
pub trait CvArithmValue: CvMeanValue {
    fn from_arithm(value: f64) -> Self;
}

macro_rules! impl_arithm_int {
    ($type:ty) => {
        impl CvArithmValue for $type {
            fn from_arithm(value: f64) -> Self {
                value
                    .round()
                    .clamp(<$type>::MIN as f64, <$type>::MAX as f64) as $type
            }
        }
    };
}
impl_arithm_int!(u8);
impl_arithm_int!(u16);
impl_arithm_int!(i16);
impl_arithm_int!(i32);
impl CvArithmValue for f32 {
    fn from_arithm(value: f64) -> Self {
        value as f32
    }
}
impl CvArithmValue for f64 {
    fn from_arithm(value: f64) -> Self {
        value
    }
}

fn matrix_and_mask<T>(
    source: &CvMeanMatrix<T>,
    destination: &CvMeanMatrix<T>,
    mask: Option<&[u8]>,
) -> Result<usize, CvArithmError> {
    let pixels = source
        .size
        .width
        .checked_mul(source.size.height)
        .ok_or(CvArithmError::BadArgument)?;
    if source.channels == 0
        || source.size != destination.size
        || source.channels != destination.channels
        || source.data.len()
            != pixels
                .checked_mul(source.channels)
                .ok_or(CvArithmError::BadArgument)?
        || destination.data.len() != source.data.len()
    {
        return Err(CvArithmError::UnmatchedFormats);
    }
    if let Some(mask) = mask {
        if mask.len() != pixels {
            return Err(CvArithmError::BadMask);
        }
    }
    Ok(pixels)
}

/// `cvSub`, with source masking semantics (unmasked destination elements stay unchanged).
pub fn cv_sub<T: CvArithmValue>(
    first: &CvMeanMatrix<T>,
    second: &CvMeanMatrix<T>,
    destination: &mut CvMeanMatrix<T>,
    mask: Option<&[u8]>,
) -> Result<(), CvArithmError> {
    let pixels = matrix_and_mask(first, destination, mask)?;
    if second.size != first.size
        || second.channels != first.channels
        || second.data.len() != first.data.len()
    {
        return Err(CvArithmError::UnmatchedSizes);
    }
    for pixel in 0..pixels {
        if mask.map_or(true, |m| m[pixel] != 0) {
            for channel in 0..first.channels {
                let index = pixel * first.channels + channel;
                destination.data[index] = T::from_arithm(
                    first.data[index].to_mean_f64() - second.data[index].to_mean_f64(),
                );
            }
        }
    }
    Ok(())
}

/// `cvAdd`, with the same masked-destination preservation as `cvSub`.
pub fn cv_add<T: CvArithmValue>(
    first: &CvMeanMatrix<T>,
    second: &CvMeanMatrix<T>,
    destination: &mut CvMeanMatrix<T>,
    mask: Option<&[u8]>,
) -> Result<(), CvArithmError> {
    let pixels = matrix_and_mask(first, destination, mask)?;
    if second.size != first.size
        || second.channels != first.channels
        || second.data.len() != first.data.len()
    {
        return Err(CvArithmError::UnmatchedSizes);
    }
    for pixel in 0..pixels {
        if mask.map_or(true, |m| m[pixel] != 0) {
            for channel in 0..first.channels {
                let index = pixel * first.channels + channel;
                destination.data[index] = T::from_arithm(
                    first.data[index].to_mean_f64() + second.data[index].to_mean_f64(),
                );
            }
        }
    }
    Ok(())
}

/// `cvSubRS`.
pub fn cv_sub_rs<T: CvArithmValue>(
    source: &CvMeanMatrix<T>,
    scalar: CvScalar,
    destination: &mut CvMeanMatrix<T>,
    mask: Option<&[u8]>,
) -> Result<(), CvArithmError> {
    let pixels = matrix_and_mask(source, destination, mask)?;
    for pixel in 0..pixels {
        if mask.map_or(true, |m| m[pixel] != 0) {
            for channel in 0..source.channels {
                let index = pixel * source.channels + channel;
                destination.data[index] =
                    T::from_arithm(scalar.values[channel] - source.data[index].to_mean_f64());
            }
        }
    }
    Ok(())
}

/// `cvAddS`.
pub fn cv_add_s<T: CvArithmValue>(
    source: &CvMeanMatrix<T>,
    scalar: CvScalar,
    destination: &mut CvMeanMatrix<T>,
    mask: Option<&[u8]>,
) -> Result<(), CvArithmError> {
    let pixels = matrix_and_mask(source, destination, mask)?;
    for pixel in 0..pixels {
        if mask.map_or(true, |m| m[pixel] != 0) {
            for channel in 0..source.channels {
                let index = pixel * source.channels + channel;
                destination.data[index] =
                    T::from_arithm(source.data[index].to_mean_f64() + scalar.values[channel]);
            }
        }
    }
    Ok(())
}

/// `cvMul`.
pub fn cv_mul<T: CvArithmValue>(
    first: &CvMeanMatrix<T>,
    second: &CvMeanMatrix<T>,
    destination: &mut CvMeanMatrix<T>,
    scale: f64,
) -> Result<(), CvArithmError> {
    let pixels = matrix_and_mask(first, destination, None)?;
    if second.size != first.size
        || second.channels != first.channels
        || second.data.len() != first.data.len()
    {
        return Err(CvArithmError::UnmatchedSizes);
    }
    for pixel in 0..pixels {
        for channel in 0..first.channels {
            let index = pixel * first.channels + channel;
            destination.data[index] = T::from_arithm(
                first.data[index].to_mean_f64() * second.data[index].to_mean_f64() * scale,
            );
        }
    }
    Ok(())
}

/// `cvDiv`; a zero denominator produces the source unit's zero result.
pub fn cv_div<T: CvArithmValue>(
    first: Option<&CvMeanMatrix<T>>,
    second: &CvMeanMatrix<T>,
    destination: &mut CvMeanMatrix<T>,
    scale: f64,
) -> Result<(), CvArithmError> {
    let source = first.unwrap_or(second);
    let pixels = matrix_and_mask(source, destination, None)?;
    if second.size != source.size
        || second.channels != source.channels
        || second.data.len() != source.data.len()
    {
        return Err(CvArithmError::UnmatchedSizes);
    }
    if let Some(first) = first {
        if first.size != second.size
            || first.channels != second.channels
            || first.data.len() != second.data.len()
        {
            return Err(CvArithmError::UnmatchedSizes);
        }
    }
    for pixel in 0..pixels {
        for channel in 0..second.channels {
            let index = pixel * second.channels + channel;
            let denominator = second.data[index].to_mean_f64();
            let numerator = first.map_or(1.0, |matrix| matrix.data[index].to_mean_f64());
            destination.data[index] = T::from_arithm(if denominator == 0.0 {
                0.0
            } else {
                numerator * scale / denominator
            });
        }
    }
    Ok(())
}

/// `cvAddWeighted`.
pub fn cv_add_weighted<T: CvArithmValue>(
    first: &CvMeanMatrix<T>,
    alpha: f64,
    second: &CvMeanMatrix<T>,
    beta: f64,
    gamma: f64,
    destination: &mut CvMeanMatrix<T>,
) -> Result<(), CvArithmError> {
    let pixels = matrix_and_mask(first, destination, None)?;
    if second.size != first.size
        || second.channels != first.channels
        || second.data.len() != first.data.len()
    {
        return Err(CvArithmError::UnmatchedSizes);
    }
    for pixel in 0..pixels {
        for channel in 0..first.channels {
            let index = pixel * first.channels + channel;
            destination.data[index] = T::from_arithm(
                first.data[index].to_mean_f64() * alpha
                    + second.data[index].to_mean_f64() * beta
                    + gamma,
            );
        }
    }
    Ok(())
}

/// `icvAddWeighted_8u_fast_C1R` (`cxarithm.cpp:1869`).  This preserves the
/// source's 14-bit lookup-table rounding used for large single-channel u8
/// images, including independent source/destination row strides.
pub fn icv_add_weighted_8u_fast_c1r(
    first: &[u8],
    first_stride: usize,
    alpha: f64,
    second: &[u8],
    second_stride: usize,
    beta: f64,
    gamma: f64,
    destination: &mut [u8],
    destination_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), CvArithmError> {
    let valid = |data_len: usize, stride: usize| {
        stride >= width && (height == 0 || data_len >= (height - 1) * stride + width)
    };
    if !valid(first.len(), first_stride)
        || !valid(second.len(), second_stride)
        || !valid(destination.len(), destination_stride)
    {
        return Err(CvArithmError::BadArgument);
    }
    const SHIFT: i32 = 14;
    let scale = (1_i32 << SHIFT) as f64;
    // The C kernel increments its double temporaries while filling the
    // tables.  Preserve that evaluation order: `j * alpha` can land on a
    // different side of `cvRound`'s boundary for non-binary coefficients.
    let mut first_table = [0_i64; 256];
    let mut second_table = [0_i64; 256];
    let mut first_value: f64 = 0.0;
    let mut second_value = gamma * scale + (1_i32 << (SHIFT - 1)) as f64;
    let first_step = alpha * scale;
    let second_step = beta * scale;
    for value in 0..256 {
        first_table[value] = first_value.round() as i64;
        second_table[value] = second_value.round() as i64;
        first_value += first_step;
        second_value += second_step;
    }
    for row in 0..height {
        for column in 0..width {
            let total = first_table[first[row * first_stride + column] as usize]
                + second_table[second[row * second_stride + column] as usize];
            destination[row * destination_stride + column] = (total >> SHIFT).clamp(0, 255) as u8;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::cvutils::CvSize;
    use super::*;
    #[test]
    fn arithmetic_entry_points_preserve_masks_saturation_and_zero_division() {
        let first = CvMeanMatrix {
            size: CvSize {
                width: 2,
                height: 1,
            },
            channels: 2,
            data: vec![10_u8, 30, 250, 4],
        };
        let second = CvMeanMatrix {
            size: first.size,
            channels: 2,
            data: vec![2_u8, 20, 0, 2],
        };
        let mut out = CvMeanMatrix {
            size: first.size,
            channels: 2,
            data: vec![9_u8; 4],
        };
        cv_sub(&first, &second, &mut out, Some(&[1, 0])).unwrap();
        assert_eq!(out.data, [8, 10, 9, 9]);
        cv_add(&first, &second, &mut out, Some(&[0, 1])).unwrap();
        assert_eq!(out.data, [8, 10, 250, 6]);
        cv_add_s(&first, CvScalar { values: [10.; 4] }, &mut out, None).unwrap();
        assert_eq!(out.data, [20, 40, 255, 14]);
        cv_div(Some(&first), &second, &mut out, 1.).unwrap();
        assert_eq!(out.data, [5, 2, 0, 2]);
        cv_add_weighted(&first, 0.5, &second, 0.5, 1., &mut out).unwrap();
        assert_eq!(out.data, [7, 26, 126, 4]);
    }
    #[test]
    fn native_u8_weighted_path_uses_fixed_point_and_strides() {
        let first = [10_u8, 20, 99, 40, 50, 99];
        let second = [20_u8, 10, 99, 60, 70, 99];
        let mut output = [77_u8; 6];
        icv_add_weighted_8u_fast_c1r(&first, 3, 0.5, &second, 3, 0.5, 1., &mut output, 3, 2, 2)
            .unwrap();
        assert_eq!(output, [16, 16, 77, 51, 61, 77]);
    }
}
