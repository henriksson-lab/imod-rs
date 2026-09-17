//! Owned translation of `IMOD/raptor/opencv/cxconvert.cpp`.

use super::cxutils::CvUtilsError;

/// Scalar conversion table generated from the source's depth macros.
pub trait CvConvertValue: Copy + Default {
    fn as_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}
macro_rules! value {
    ($t:ty, $f:expr) => {
        impl CvConvertValue for $t {
            fn as_f64(self) -> f64 {
                self as f64
            }
            fn from_f64(v: f64) -> Self {
                $f(v)
            }
        }
    };
}
value!(u8, |v: f64| v.round().clamp(0., 255.) as u8);
value!(i8, |v: f64| v.round().clamp(-128., 127.) as i8);
value!(u16, |v: f64| v.round().clamp(0., 65535.) as u16);
value!(i16, |v: f64| v.round().clamp(-32768., 32767.) as i16);
value!(
    i32,
    |v: f64| v.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
);
value!(f32, |v: f64| v as f32);
value!(f64, |v: f64| v);

/// Owned interleaved array for source channel operations.
#[derive(Clone, Debug, PartialEq)]
pub struct CvChannelMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub channels: usize,
    pub data: Vec<T>,
}
impl<T> CvChannelMatrix<T> {
    pub fn new(
        rows: usize,
        cols: usize,
        channels: usize,
        data: Vec<T>,
    ) -> Result<Self, CvUtilsError> {
        if channels == 0
            || rows.checked_mul(cols).and_then(|n| n.checked_mul(channels)) != Some(data.len())
        {
            return Err(CvUtilsError::UnmatchedSizes);
        }
        Ok(Self {
            rows,
            cols,
            channels,
            data,
        })
    }
}

fn is_valid_channel_matrix<T>(matrix: &CvChannelMatrix<T>) -> bool {
    matrix.channels != 0
        && matrix
            .rows
            .checked_mul(matrix.cols)
            .and_then(|pixels| pixels.checked_mul(matrix.channels))
            == Some(matrix.data.len())
}

/// C `cvSplit`; destinations are source null pointers represented as `None`.
pub fn cv_split<T: Copy>(
    source: &CvChannelMatrix<T>,
    destinations: &mut [Option<CvChannelMatrix<T>>],
) -> Result<(), CvUtilsError> {
    if !is_valid_channel_matrix(source)
        || source.channels < 2
        || source.channels > 4
        || destinations.len() != 4
    {
        return Err(CvUtilsError::BadArgument);
    }
    let indices: Vec<_> = destinations
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|_| i))
        .collect();
    if indices.len() != 1 && indices.len() != source.channels {
        return Err(CvUtilsError::BadArgument);
    }
    for &c in &indices {
        let d = destinations[c].as_ref().unwrap();
        if !is_valid_channel_matrix(d) {
            return Err(CvUtilsError::BadArgument);
        }
        if d.channels != 1 {
            return Err(CvUtilsError::UnsupportedFormat);
        }
        if d.rows != source.rows || d.cols != source.cols {
            return Err(CvUtilsError::UnmatchedSizes);
        }
    }
    for &c in &indices {
        let d = destinations[c].as_mut().unwrap();
        for pixel in 0..source.rows * source.cols {
            d.data[pixel] = source.data[pixel * source.channels + c];
        }
    }
    Ok(())
}
/// C `cvMerge`; sources are source null pointers represented as `None`.
pub fn cv_merge<T: Copy>(
    sources: &[Option<CvChannelMatrix<T>>],
    destination: &mut CvChannelMatrix<T>,
) -> Result<(), CvUtilsError> {
    if !is_valid_channel_matrix(destination)
        || destination.channels < 2
        || destination.channels > 4
        || sources.len() != 4
    {
        return Err(CvUtilsError::BadArgument);
    }
    let indices: Vec<_> = sources
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|_| i))
        .collect();
    if indices.len() != 1 && indices.len() != destination.channels {
        return Err(CvUtilsError::BadArgument);
    }
    for &c in &indices {
        let s = sources[c].as_ref().unwrap();
        if !is_valid_channel_matrix(s) {
            return Err(CvUtilsError::BadArgument);
        }
        if s.channels != 1 {
            return Err(CvUtilsError::UnsupportedFormat);
        }
        if s.rows != destination.rows || s.cols != destination.cols {
            return Err(CvUtilsError::UnmatchedSizes);
        }
    }
    for &c in &indices {
        let s = sources[c].as_ref().unwrap();
        for pixel in 0..destination.rows * destination.cols {
            destination.data[pixel * destination.channels + c] = s.data[pixel];
        }
    }
    Ok(())
}
/// C `cvMixChannels`; a negative source index produces zeroes.
pub fn cv_mix_channels<T: Copy + Default>(
    sources: &[CvChannelMatrix<T>],
    destinations: &mut [CvChannelMatrix<T>],
    from_to: &[(i32, usize)],
) -> Result<(), CvUtilsError> {
    if destinations.is_empty() || from_to.is_empty() {
        return Err(CvUtilsError::OutOfRange);
    }
    if sources
        .iter()
        .chain(destinations.iter())
        .any(|matrix| !is_valid_channel_matrix(matrix))
    {
        return Err(CvUtilsError::BadArgument);
    }
    let reference = sources
        .first()
        .or_else(|| destinations.first())
        .ok_or(CvUtilsError::BadArgument)?;
    if sources
        .iter()
        .chain(destinations.iter())
        .any(|m| m.rows != reference.rows || m.cols != reference.cols)
    {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    let src_total: usize = sources.iter().map(|m| m.channels).sum();
    let dst_total: usize = destinations.iter().map(|m| m.channels).sum();
    if from_to
        .iter()
        .any(|&(s, d)| s >= src_total as i32 || d >= dst_total)
    {
        return Err(CvUtilsError::OutOfRange);
    }
    for pixel in 0..reference.rows * reference.cols {
        for &(from, to) in from_to {
            let mut remaining = to;
            let mut di = 0;
            while remaining >= destinations[di].channels {
                remaining -= destinations[di].channels;
                di += 1;
            }
            let value = if from < 0 {
                T::default()
            } else {
                let mut remaining = from as usize;
                let mut si = 0;
                while remaining >= sources[si].channels {
                    remaining -= sources[si].channels;
                    si += 1;
                }
                sources[si].data[pixel * sources[si].channels + remaining]
            };
            let channels = destinations[di].channels;
            destinations[di].data[pixel * channels + remaining] = value;
        }
    }
    Ok(())
}
/// C `cvConvertScaleAbs`.
pub fn cv_convert_scale_abs<T: CvConvertValue>(
    source: &[T],
    destination: &mut [u8],
    scale: f64,
    shift: f64,
) -> Result<(), CvUtilsError> {
    if source.len() != destination.len() {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    for (s, d) in source.iter().zip(destination) {
        *d = (s.as_f64() * scale + shift).round().abs().clamp(0., 255.) as u8;
    }
    Ok(())
}
/// C `cvConvertScale`, with the source generated destination-depth dispatch represented by `CvConvertValue`.
pub fn cv_convert_scale<S: CvConvertValue, D: CvConvertValue>(
    source: &[S],
    destination: &mut [D],
    scale: f64,
    shift: f64,
) -> Result<(), CvUtilsError> {
    if source.len() != destination.len() {
        return Err(CvUtilsError::UnmatchedSizes);
    }
    for (s, d) in source.iter().zip(destination) {
        *d = D::from_f64(s.as_f64() * scale + shift);
    }
    Ok(())
}
/// Source helper `icvCvt_32f64f`.
pub fn icv_cvt_32f64f(source: &[f32], destination: &mut [f64]) -> Result<(), CvUtilsError> {
    cv_convert_scale(source, destination, 1., 0.)
}
/// Source helper `icvCvt_64f32f`.
pub fn icv_cvt_64f32f(source: &[f64], destination: &mut [f32]) -> Result<(), CvUtilsError> {
    cv_convert_scale(source, destination, 1., 0.)
}
/// Source helper `icvScale_32f`.
pub fn icv_scale_32f(
    source: &[f32],
    destination: &mut [f32],
    scale: f32,
    shift: f32,
) -> Result<(), CvUtilsError> {
    cv_convert_scale(source, destination, scale as f64, shift as f64)
}
/// Source helper `icvScale_64f`.
pub fn icv_scale_64f(
    source: &[f64],
    destination: &mut [f64],
    scale: f64,
    shift: f64,
) -> Result<(), CvUtilsError> {
    cv_convert_scale(source, destination, scale, shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn channel_and_conversion_paths() {
        let s = CvChannelMatrix::new(1, 2, 3, vec![1u8, 2, 3, 4, 5, 6]).unwrap();
        let mut p = [
            Some(CvChannelMatrix::new(1, 2, 1, vec![0; 2]).unwrap()),
            Some(CvChannelMatrix::new(1, 2, 1, vec![0; 2]).unwrap()),
            Some(CvChannelMatrix::new(1, 2, 1, vec![0; 2]).unwrap()),
            None,
        ];
        cv_split(&s, &mut p).unwrap();
        assert_eq!(p[1].as_ref().unwrap().data, vec![2, 5]);
        let mut d = CvChannelMatrix::new(1, 2, 3, vec![0; 6]).unwrap();
        cv_merge(&p, &mut d).unwrap();
        assert_eq!(d.data, s.data);
        let mut a = [0; 3];
        cv_convert_scale_abs(&[-2i16, 3, 400], &mut a, 1., 0.).unwrap();
        assert_eq!(a, [2, 3, 255]);
        let mut b = [0u8; 3];
        cv_convert_scale(&[-1i16, 100, 1000], &mut b, 0.5, 1.).unwrap();
        assert_eq!(b, [1, 51, 255]);
    }

    #[test]
    fn malformed_public_channel_matrix_returns_an_error_before_indexing() {
        let source = CvChannelMatrix {
            rows: 1,
            cols: 2,
            channels: 2,
            data: vec![1_u8; 3],
        };
        let mut destinations = [None, None, None, None];
        assert_eq!(
            cv_split(&source, &mut destinations),
            Err(CvUtilsError::BadArgument)
        );
    }
}
