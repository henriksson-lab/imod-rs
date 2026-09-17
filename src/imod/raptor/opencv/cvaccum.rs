//! Accumulation operations from `IMOD/raptor/opencv/cvaccum.cpp`.

use super::cxmean::CvMeanMatrix;

/// Failures represented by the `cvAcc` family C error paths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvAccumError {
    BadArgument,
    BadMask,
    UnmatchedFormats,
    UnmatchedSizes,
}

/// Source depths accepted by every native `cvAcc` dispatch table.
pub trait CvAccumSource: Copy {
    fn to_accum_f64(self) -> f64;
}
impl CvAccumSource for u8 {
    fn to_accum_f64(self) -> f64 {
        self as f64
    }
}
impl CvAccumSource for f32 {
    fn to_accum_f64(self) -> f64 {
        self as f64
    }
}

/// `cvAcc`: add source samples into an `f32` accumulator.
pub fn cv_acc<T: CvAccumSource>(
    source: &CvMeanMatrix<T>,
    accumulator: &mut CvMeanMatrix<f32>,
    mask: Option<&[u8]>,
) -> Result<(), CvAccumError> {
    let pixels = source
        .size
        .width
        .checked_mul(source.size.height)
        .ok_or(CvAccumError::BadArgument)?;
    if source.channels == 0
        || source.channels != accumulator.channels
        || source.size != accumulator.size
        || source.data.len()
            != pixels
                .checked_mul(source.channels)
                .ok_or(CvAccumError::BadArgument)?
        || accumulator.data.len()
            != pixels
                .checked_mul(accumulator.channels)
                .ok_or(CvAccumError::BadArgument)?
    {
        return Err(CvAccumError::UnmatchedFormats);
    }
    if let Some(mask) = mask {
        if mask.len() != pixels {
            return Err(CvAccumError::BadMask);
        }
    }
    for pixel in 0..pixels {
        if mask.map_or(true, |selected| selected[pixel] != 0) {
            for channel in 0..source.channels {
                let index = pixel * source.channels + channel;
                accumulator.data[index] += source.data[index].to_accum_f64() as f32;
            }
        }
    }
    Ok(())
}

/// `cvSquareAcc`: add squared source samples into an `f32` accumulator.
pub fn cv_square_acc<T: CvAccumSource>(
    source: &CvMeanMatrix<T>,
    accumulator: &mut CvMeanMatrix<f32>,
    mask: Option<&[u8]>,
) -> Result<(), CvAccumError> {
    let pixels = source
        .size
        .width
        .checked_mul(source.size.height)
        .ok_or(CvAccumError::BadArgument)?;
    if source.channels == 0
        || source.channels != accumulator.channels
        || source.size != accumulator.size
        || source.data.len()
            != pixels
                .checked_mul(source.channels)
                .ok_or(CvAccumError::BadArgument)?
        || accumulator.data.len()
            != pixels
                .checked_mul(accumulator.channels)
                .ok_or(CvAccumError::BadArgument)?
    {
        return Err(CvAccumError::UnmatchedFormats);
    }
    if let Some(mask) = mask {
        if mask.len() != pixels {
            return Err(CvAccumError::BadMask);
        }
    }
    for pixel in 0..pixels {
        if mask.map_or(true, |selected| selected[pixel] != 0) {
            for channel in 0..source.channels {
                let index = pixel * source.channels + channel;
                let value = source.data[index].to_accum_f64() as f32;
                accumulator.data[index] += value * value;
            }
        }
    }
    Ok(())
}

/// `cvMultiplyAcc`: add products of two source matrices into an `f32` accumulator.
pub fn cv_multiply_acc<T: CvAccumSource>(
    first: &CvMeanMatrix<T>,
    second: &CvMeanMatrix<T>,
    accumulator: &mut CvMeanMatrix<f32>,
    mask: Option<&[u8]>,
) -> Result<(), CvAccumError> {
    let pixels = first
        .size
        .width
        .checked_mul(first.size.height)
        .ok_or(CvAccumError::BadArgument)?;
    if first.channels == 0
        || first.size != second.size
        || first.size != accumulator.size
        || first.channels != second.channels
        || first.channels != accumulator.channels
        || first.data.len()
            != pixels
                .checked_mul(first.channels)
                .ok_or(CvAccumError::BadArgument)?
        || second.data.len() != first.data.len()
        || accumulator.data.len() != first.data.len()
    {
        return Err(CvAccumError::UnmatchedSizes);
    }
    if let Some(mask) = mask {
        if mask.len() != pixels {
            return Err(CvAccumError::BadMask);
        }
    }
    for pixel in 0..pixels {
        if mask.map_or(true, |selected| selected[pixel] != 0) {
            for channel in 0..first.channels {
                let index = pixel * first.channels + channel;
                accumulator.data[index] += first.data[index].to_accum_f64() as f32
                    * second.data[index].to_accum_f64() as f32;
            }
        }
    }
    Ok(())
}

/// `cvRunningAvg`: update an `f32` accumulator with `alpha * source + (1-alpha) * accumulator`.
pub fn cv_running_avg<T: CvAccumSource>(
    source: &CvMeanMatrix<T>,
    accumulator: &mut CvMeanMatrix<f32>,
    alpha: f64,
    mask: Option<&[u8]>,
) -> Result<(), CvAccumError> {
    let pixels = source
        .size
        .width
        .checked_mul(source.size.height)
        .ok_or(CvAccumError::BadArgument)?;
    if source.channels == 0
        || source.channels != accumulator.channels
        || source.size != accumulator.size
        || source.data.len()
            != pixels
                .checked_mul(source.channels)
                .ok_or(CvAccumError::BadArgument)?
        || accumulator.data.len()
            != pixels
                .checked_mul(accumulator.channels)
                .ok_or(CvAccumError::BadArgument)?
    {
        return Err(CvAccumError::UnmatchedFormats);
    }
    if let Some(mask) = mask {
        if mask.len() != pixels {
            return Err(CvAccumError::BadMask);
        }
    }
    for pixel in 0..pixels {
        if mask.map_or(true, |selected| selected[pixel] != 0) {
            for channel in 0..source.channels {
                let index = pixel * source.channels + channel;
                let alpha = alpha as f32;
                accumulator.data[index] = accumulator.data[index] * (1.0 - alpha)
                    + source.data[index].to_accum_f64() as f32 * alpha;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::cvutils::CvSize;
    use super::*;

    #[test]
    fn accumulation_operations_preserve_masks_and_channels() {
        let source = CvMeanMatrix {
            size: CvSize {
                width: 2,
                height: 1,
            },
            channels: 2,
            data: vec![2_u8, 3, 4, 5],
        };
        let second = CvMeanMatrix {
            size: source.size,
            channels: 2,
            data: vec![10_u8, 20, 30, 40],
        };
        let mut accumulator = CvMeanMatrix {
            size: source.size,
            channels: 2,
            data: vec![1_f32; 4],
        };
        cv_acc(&source, &mut accumulator, Some(&[1, 0])).unwrap();
        assert_eq!(accumulator.data, [3., 4., 1., 1.]);
        cv_square_acc(&source, &mut accumulator, Some(&[0, 1])).unwrap();
        assert_eq!(accumulator.data, [3., 4., 17., 26.]);
        cv_multiply_acc(&source, &second, &mut accumulator, None).unwrap();
        assert_eq!(accumulator.data, [23., 64., 137., 226.]);
    }

    #[test]
    fn running_average_matches_the_source_formula() {
        let source = CvMeanMatrix {
            size: CvSize {
                width: 2,
                height: 1,
            },
            channels: 1,
            data: vec![10_f32, 20.],
        };
        let mut accumulator = CvMeanMatrix {
            size: source.size,
            channels: 1,
            data: vec![2_f32, 4.],
        };
        cv_running_avg(&source, &mut accumulator, 0.25, Some(&[1, 0])).unwrap();
        assert_eq!(accumulator.data, [4., 4.]);
    }
}
