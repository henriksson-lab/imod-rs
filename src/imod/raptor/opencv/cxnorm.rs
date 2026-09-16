//! Owned generic translation of `IMOD/raptor/opencv/cxnorm.cpp`.
//!
//! The C source expands raw-pointer kernels for every supported scalar depth,
//! norm kind, mask state, difference state, and COI.  The generic operation
//! below is the equivalent dispatch without C function tables or byte casts.

use super::cxerror::CvStatus;
use super::cxminmaxloc::{CvMask, CvMatrix};

/// C `CV_C`, `CV_L1`, and `CV_L2` norm kinds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvNormKind {
    C,
    L1,
    L2,
}

impl CvNormKind {
    /// Decodes C `CV_*` flags, including the ignored `CV_DIFF` bit.
    pub fn from_c_flags(flags: i32) -> Result<(Self, bool), CvStatus> {
        let relative = flags & 8 != 0;
        match flags & 7 {
            1 => Ok((Self::C, relative)),
            2 => Ok((Self::L1, relative)),
            4 => Ok((Self::L2, relative)),
            _ => Err(CvStatus::sts_bad_flag),
        }
    }
}

/// Scalar depths accepted by the source's norm dispatch tables.
pub trait CvNormValue: Copy {
    fn to_norm_f64(self) -> f64;
}

macro_rules! impl_norm_value {
    ($($type:ty),+ $(,)?) => {
        $(
            impl CvNormValue for $type {
                fn to_norm_f64(self) -> f64 { self as f64 }
            }
        )+
    };
}

impl_norm_value!(u8, u16, i16, i32, f32, f64);

/// C `cvNorm` plus the source's generated `icvNorm*` and `icvNormDiff*`
/// kernel families.
///
/// `second` activates difference norms. For relative difference norms, as in
/// C, the denominator is the norm of `second` (the public `imgB` argument).
/// `coi` is a one-based selected channel; a mask on a multi-channel matrix
/// requires it, matching `cvNorm`'s C validation.
pub fn cv_norm<T: CvNormValue>(
    first: &CvMatrix<T>,
    second: Option<&CvMatrix<T>>,
    kind: CvNormKind,
    relative: bool,
    mask: Option<&CvMask>,
    coi: Option<usize>,
) -> Result<f64, CvStatus> {
    if let Some(second) = second {
        if first.rows != second.rows || first.columns != second.columns {
            return Err(CvStatus::sts_unmatched_sizes);
        }
        if first.channels != second.channels {
            return Err(CvStatus::sts_unmatched_formats);
        }
    }
    let selected_channel = match coi {
        Some(channel) if channel == 0 || channel > first.channels => {
            return Err(CvStatus::bad_coi);
        }
        Some(channel) => Some(channel - 1),
        None => None,
    };
    if mask.is_some() && first.channels > 1 && selected_channel.is_none() {
        return Err(CvStatus::sts_bad_arg);
    }
    if let Some(mask) = mask {
        if mask.row_stride < first.columns
            || mask.values.len() < (first.rows - 1) * mask.row_stride + first.columns
        {
            return Err(CvStatus::sts_bad_mask);
        }
    }

    let mut base: f64 = 0.0;
    let mut difference: f64 = 0.0;
    let channels = if selected_channel.is_some() {
        1
    } else {
        first.channels
    };
    for row in 0..first.rows {
        for column in 0..first.columns {
            if mask.is_some_and(|mask| mask.values[row * mask.row_stride + column] == 0) {
                continue;
            }
            let first_start = row * first.row_stride + column * first.channels;
            let second_start =
                second.map(|matrix| row * matrix.row_stride + column * matrix.channels);
            for output_channel in 0..channels {
                let channel = selected_channel.unwrap_or(output_channel);
                let first_value = first.values[first_start + channel].to_norm_f64();
                let second_value = second
                    .map(|matrix| matrix.values[second_start.unwrap() + channel].to_norm_f64())
                    .unwrap_or(0.0);
                let base_value = if second.is_some() {
                    second_value
                } else {
                    first_value
                };
                let difference_value = first_value - second_value;
                match kind {
                    CvNormKind::C => {
                        base = base.max(base_value.abs());
                        if second.is_some() {
                            difference = difference.max(difference_value.abs());
                        }
                    }
                    CvNormKind::L1 => {
                        base += base_value.abs();
                        if second.is_some() {
                            difference += difference_value.abs();
                        }
                    }
                    CvNormKind::L2 => {
                        base += base_value * base_value;
                        if second.is_some() {
                            difference += difference_value * difference_value;
                        }
                    }
                }
            }
        }
    }
    if kind == CvNormKind::L2 {
        base = base.sqrt();
        difference = difference.sqrt();
    }
    if second.is_none() {
        return Ok(base);
    }
    if relative {
        Ok(difference / (base + f64::EPSILON))
    } else {
        Ok(difference)
    }
}

#[cfg(test)]
mod tests {
    use super::{CvNormKind, cv_norm};
    use crate::imod::raptor::opencv::cxerror::CvStatus;
    use crate::imod::raptor::opencv::cxminmaxloc::{CvMask, CvMatrix};

    #[test]
    fn norm_kinds_match_source_single_array_kernels() {
        let matrix = CvMatrix::new(1, 3, 1, 3, vec![-3_i16, 4, -12]).unwrap();
        assert_eq!(
            cv_norm(&matrix, None, CvNormKind::C, false, None, None).unwrap(),
            12.0
        );
        assert_eq!(
            cv_norm(&matrix, None, CvNormKind::L1, false, None, None).unwrap(),
            19.0
        );
        assert_eq!(
            cv_norm(&matrix, None, CvNormKind::L2, false, None, None).unwrap(),
            13.0
        );
    }

    #[test]
    fn relative_difference_uses_public_second_array_as_denominator() {
        let first = CvMatrix::new(1, 2, 1, 2, vec![4_f64, 8.0]).unwrap();
        let second = CvMatrix::new(1, 2, 1, 2, vec![2_f64, 4.0]).unwrap();
        let result = cv_norm(&first, Some(&second), CvNormKind::L2, true, None, None).unwrap();
        assert!((result - 1.0).abs() < 1e-14);
    }

    #[test]
    fn mask_requires_coi_for_multichannel_and_uses_selected_strided_channel() {
        let matrix =
            CvMatrix::new(2, 2, 2, 5, vec![1_u8, 10, 2, 20, 99, 3, 30, 4, 40, 99]).unwrap();
        let mask = CvMask::new(2, 2, 2, vec![1, 0, 0, 1]).unwrap();
        assert_eq!(
            cv_norm(&matrix, None, CvNormKind::L1, false, Some(&mask), None),
            Err(CvStatus::sts_bad_arg)
        );
        assert_eq!(
            cv_norm(&matrix, None, CvNormKind::L1, false, Some(&mask), Some(2)).unwrap(),
            50.0
        );
    }

    #[test]
    fn c_flags_and_shape_errors_follow_source_validation() {
        assert_eq!(CvNormKind::from_c_flags(4 | 8), Ok((CvNormKind::L2, true)));
        assert_eq!(CvNormKind::from_c_flags(0), Err(CvStatus::sts_bad_flag));
        let first = CvMatrix::new(1, 1, 1, 1, vec![1_i32]).unwrap();
        let second = CvMatrix::new(1, 2, 1, 2, vec![1_i32, 2]).unwrap();
        assert_eq!(
            cv_norm(&first, Some(&second), CvNormKind::C, false, None, None),
            Err(CvStatus::sts_unmatched_sizes)
        );
    }
}
