//! Matrix/point utilities from `IMOD/raptor/opencv/cvutils.cpp`.
//!
//! The original unit is predominantly byte-pointer border copying.  These
//! entry points retain its byte-stride and interleaved-channel semantics while
//! using checked slices and owned point/matrix data.

/// C `CvSize` used by the byte-copy kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CvSize {
    pub width: usize,
    pub height: usize,
}

/// C `CvPoint` used as the top/left border offset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CvPoint {
    pub x: i32,
    pub y: i32,
}

/// C point formats accepted by `cvPointSeqFromMat`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CvPoint2 {
    I32 { x: i32, y: i32 },
    F32 { x: f32, y: f32 },
}

/// Safe owned replacement for the source point matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CvPointMatrix {
    pub width: usize,
    pub height: usize,
    pub continuous: bool,
    pub points: Vec<CvPoint2>,
}

/// Safe owned replacement for the contour header/block pair initialized by
/// `cvPointSeqFromMat`.
#[derive(Clone, Debug, PartialEq)]
pub struct CvPointSequence {
    pub kind: i32,
    pub points: Vec<CvPoint2>,
}

/// An owned interleaved byte matrix for `cvCopyMakeBorder`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CvByteMatrix {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub row_stride: usize,
    pub data: Vec<u8>,
}

/// C's supported `IPL_BORDER_*` values in this translation unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvBorderType {
    Constant,
    Replicate,
    Reflect101,
}

/// Errors represented by C `CvStatus`/`CV_ERROR` paths here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvUtilsError {
    BadArgument,
    BadSize,
    UnmatchedFormats,
    UnsupportedFormat,
}

/// Owned `cvPointSeqFromMat`.
pub fn cv_point_seq_from_mat(
    sequence_kind: i32,
    matrix: CvPointMatrix,
) -> Result<CvPointSequence, CvUtilsError> {
    if !matrix.continuous || (matrix.width != 1 && matrix.height != 1) {
        return Err(CvUtilsError::BadArgument);
    }
    if matrix.width.checked_mul(matrix.height) != Some(matrix.points.len()) {
        return Err(CvUtilsError::BadArgument);
    }
    Ok(CvPointSequence {
        kind: sequence_kind,
        points: matrix.points,
    })
}

/// `icvCopyReplicateBorder_8u`.
pub fn icv_copy_replicate_border_8u(
    src: &[u8],
    src_step: usize,
    src_roi: CvSize,
    dst: &mut [u8],
    dst_step: usize,
    dst_roi: CvSize,
    top: usize,
    left: usize,
    channels: usize,
) -> Result<(), CvUtilsError> {
    if channels == 0
        || src_step
            < src_roi
                .width
                .checked_mul(channels)
                .ok_or(CvUtilsError::BadArgument)?
        || dst_step
            < dst_roi
                .width
                .checked_mul(channels)
                .ok_or(CvUtilsError::BadArgument)?
        || src.len()
            < src_roi.height.saturating_sub(1).saturating_mul(src_step) + src_roi.width * channels
        || dst.len()
            < dst_roi.height.saturating_sub(1).saturating_mul(dst_step) + dst_roi.width * channels
        || top + src_roi.height > dst_roi.height
        || left + src_roi.width > dst_roi.width
    {
        return Err(CvUtilsError::BadArgument);
    }
    for y in 0..dst_roi.height {
        let source_y = y.saturating_sub(top).min(src_roi.height - 1);
        for x in 0..dst_roi.width {
            let source_x = x.saturating_sub(left).min(src_roi.width - 1);
            for channel in 0..channels {
                dst[y * dst_step + x * channels + channel] =
                    src[source_y * src_step + source_x * channels + channel];
            }
        }
    }
    Ok(())
}

/// `icvCopyReflect101Border_8u`.
pub fn icv_copy_reflect101_border_8u(
    src: &[u8],
    src_step: usize,
    src_roi: CvSize,
    dst: &mut [u8],
    dst_step: usize,
    dst_roi: CvSize,
    top: usize,
    left: usize,
    channels: usize,
) -> Result<(), CvUtilsError> {
    if channels == 0
        || src_roi.width == 0
        || src_roi.height == 0
        || src_step
            < src_roi
                .width
                .checked_mul(channels)
                .ok_or(CvUtilsError::BadArgument)?
        || dst_step
            < dst_roi
                .width
                .checked_mul(channels)
                .ok_or(CvUtilsError::BadArgument)?
        || src.len() < (src_roi.height - 1).saturating_mul(src_step) + src_roi.width * channels
        || dst.len() < (dst_roi.height - 1).saturating_mul(dst_step) + dst_roi.width * channels
        || top + src_roi.height > dst_roi.height
        || left + src_roi.width > dst_roi.width
    {
        return Err(CvUtilsError::BadArgument);
    }
    for y in 0..dst_roi.height {
        let mut source_y = y as isize - top as isize;
        if src_roi.height == 1 {
            source_y = 0;
        } else {
            while source_y < 0 || source_y >= src_roi.height as isize {
                source_y = if source_y < 0 {
                    -source_y
                } else {
                    2 * src_roi.height as isize - source_y - 2
                };
            }
        }
        for x in 0..dst_roi.width {
            let mut source_x = x as isize - left as isize;
            if src_roi.width == 1 {
                source_x = 0;
            } else {
                while source_x < 0 || source_x >= src_roi.width as isize {
                    source_x = if source_x < 0 {
                        -source_x
                    } else {
                        2 * src_roi.width as isize - source_x - 2
                    };
                }
            }
            for channel in 0..channels {
                dst[y * dst_step + x * channels + channel] =
                    src[source_y as usize * src_step + source_x as usize * channels + channel];
            }
        }
    }
    Ok(())
}

/// `icvCopyConstBorder_8u`.
pub fn icv_copy_const_border_8u(
    src: &[u8],
    src_step: usize,
    src_roi: CvSize,
    dst: &mut [u8],
    dst_step: usize,
    dst_roi: CvSize,
    top: usize,
    left: usize,
    channels: usize,
    value: &[u8],
) -> Result<(), CvUtilsError> {
    if channels == 0
        || value.len() < channels
        || src_step
            < src_roi
                .width
                .checked_mul(channels)
                .ok_or(CvUtilsError::BadArgument)?
        || dst_step
            < dst_roi
                .width
                .checked_mul(channels)
                .ok_or(CvUtilsError::BadArgument)?
        || src.len()
            < src_roi.height.saturating_sub(1).saturating_mul(src_step) + src_roi.width * channels
        || dst.len()
            < dst_roi.height.saturating_sub(1).saturating_mul(dst_step) + dst_roi.width * channels
        || top + src_roi.height > dst_roi.height
        || left + src_roi.width > dst_roi.width
    {
        return Err(CvUtilsError::BadArgument);
    }
    for y in 0..dst_roi.height {
        for x in 0..dst_roi.width {
            for channel in 0..channels {
                dst[y * dst_step + x * channels + channel] = value[channel];
            }
        }
    }
    for y in 0..src_roi.height {
        let source_row = y * src_step;
        let destination_row = (top + y) * dst_step + left * channels;
        dst[destination_row..destination_row + src_roi.width * channels]
            .copy_from_slice(&src[source_row..source_row + src_roi.width * channels]);
    }
    Ok(())
}

/// Owned `cvCopyMakeBorder`.
pub fn cv_copy_make_border(
    source: &CvByteMatrix,
    destination: &mut CvByteMatrix,
    offset: CvPoint,
    border_type: CvBorderType,
    value: &[u8],
) -> Result<(), CvUtilsError> {
    let offset_x = usize::try_from(offset.x).map_err(|_| CvUtilsError::BadArgument)?;
    let offset_y = usize::try_from(offset.y).map_err(|_| CvUtilsError::BadArgument)?;
    if source.channels == 0
        || source.channels != destination.channels
        || source.row_stride
            < source
                .width
                .checked_mul(source.channels)
                .ok_or(CvUtilsError::BadArgument)?
        || destination.row_stride
            < destination
                .width
                .checked_mul(destination.channels)
                .ok_or(CvUtilsError::BadArgument)?
        || source.data.len()
            < source
                .height
                .saturating_sub(1)
                .saturating_mul(source.row_stride)
                + source.width * source.channels
        || destination.data.len()
            < destination
                .height
                .saturating_sub(1)
                .saturating_mul(destination.row_stride)
                + destination.width * destination.channels
    {
        return Err(CvUtilsError::UnmatchedFormats);
    }
    if source.width + offset_x > destination.width || source.height + offset_y > destination.height
    {
        return Err(CvUtilsError::BadSize);
    }
    match border_type {
        CvBorderType::Replicate => icv_copy_replicate_border_8u(
            &source.data,
            source.row_stride,
            CvSize {
                width: source.width,
                height: source.height,
            },
            &mut destination.data,
            destination.row_stride,
            CvSize {
                width: destination.width,
                height: destination.height,
            },
            offset_y,
            offset_x,
            source.channels,
        ),
        CvBorderType::Reflect101 => icv_copy_reflect101_border_8u(
            &source.data,
            source.row_stride,
            CvSize {
                width: source.width,
                height: source.height,
            },
            &mut destination.data,
            destination.row_stride,
            CvSize {
                width: destination.width,
                height: destination.height,
            },
            offset_y,
            offset_x,
            source.channels,
        ),
        CvBorderType::Constant => icv_copy_const_border_8u(
            &source.data,
            source.row_stride,
            CvSize {
                width: source.width,
                height: source.height,
            },
            &mut destination.data,
            destination.row_stride,
            CvSize {
                width: destination.width,
                height: destination.height,
            },
            offset_y,
            offset_x,
            source.channels,
            value,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicate_border_matches_the_c_kernel_for_interleaved_pixels() {
        let source = [1, 10, 2, 20, 3, 30, 4, 40];
        let mut destination = [0; 24];
        icv_copy_replicate_border_8u(
            &source,
            4,
            CvSize {
                width: 2,
                height: 2,
            },
            &mut destination,
            6,
            CvSize {
                width: 3,
                height: 4,
            },
            1,
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            destination,
            [
                1, 10, 1, 10, 2, 20, 1, 10, 1, 10, 2, 20, 3, 30, 3, 30, 4, 40, 3, 30, 3, 30, 4, 40
            ]
        );
    }

    #[test]
    fn reflect101_uses_the_neighbor_not_the_edge_pixel() {
        let source = [10, 20, 30];
        let mut destination = [0; 7];
        icv_copy_reflect101_border_8u(
            &source,
            3,
            CvSize {
                width: 3,
                height: 1,
            },
            &mut destination,
            7,
            CvSize {
                width: 7,
                height: 1,
            },
            0,
            2,
            1,
        )
        .unwrap();
        assert_eq!(destination, [30, 20, 10, 20, 30, 20, 10]);
    }

    #[test]
    fn constant_border_and_owned_point_sequence_preserve_data() {
        let source = CvByteMatrix {
            width: 2,
            height: 1,
            channels: 1,
            row_stride: 2,
            data: vec![4, 5],
        };
        let mut destination = CvByteMatrix {
            width: 4,
            height: 3,
            channels: 1,
            row_stride: 4,
            data: vec![0; 12],
        };
        cv_copy_make_border(
            &source,
            &mut destination,
            CvPoint { x: 1, y: 1 },
            CvBorderType::Constant,
            &[9],
        )
        .unwrap();
        assert_eq!(destination.data, [9, 9, 9, 9, 9, 4, 5, 9, 9, 9, 9, 9]);
        let sequence = cv_point_seq_from_mat(
            7,
            CvPointMatrix {
                width: 1,
                height: 2,
                continuous: true,
                points: vec![
                    CvPoint2::I32 { x: 1, y: 2 },
                    CvPoint2::F32 { x: 3.0, y: 4.0 },
                ],
            },
        )
        .unwrap();
        assert_eq!(sequence.kind, 7);
        assert_eq!(sequence.points.len(), 2);
    }
}
