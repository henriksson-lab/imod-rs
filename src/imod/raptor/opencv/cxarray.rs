//! Owned Rust translation in progress for `IMOD/raptor/opencv/cxarray.cpp`.
//!
//! This unit is the legacy OpenCV array foundation.  Its C `CvMat`,
//! `CvMatND`, `CvSparseMat`, and `IplImage` headers separate allocation from
//! raw data pointers; the Rust forms keep shape, stride, channels, and data
//! together so all later translated kernels have checked owned storage.

use std::collections::BTreeMap;

use super::cxerror::CvStatus;
use super::cxmean::CvScalar;

/// C `CvRect`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CvRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

/// C `CvTermCriteria`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CvTermCriteria {
    pub use_iterations: bool,
    pub use_epsilon: bool,
    pub max_iterations: usize,
    pub epsilon: f64,
}

/// C `cvCheckTermCriteria`.
pub fn cv_check_term_criteria(
    criteria: CvTermCriteria,
    default_epsilon: f64,
    default_max_iterations: usize,
) -> Result<CvTermCriteria, CvStatus> {
    if !criteria.use_iterations && !criteria.use_epsilon {
        return Err(CvStatus::sts_bad_arg);
    }
    if criteria.use_iterations && criteria.max_iterations == 0 {
        return Err(CvStatus::sts_bad_arg);
    }
    if criteria.use_epsilon && criteria.epsilon < 0.0 {
        return Err(CvStatus::sts_bad_arg);
    }
    Ok(CvTermCriteria {
        use_iterations: true,
        use_epsilon: true,
        max_iterations: if criteria.use_iterations {
            criteria.max_iterations
        } else {
            default_max_iterations.max(1)
        },
        epsilon: if criteria.use_epsilon {
            criteria.epsilon.max(0.0)
        } else {
            default_epsilon.max(0.0)
        },
    })
}

/// Owned C `CvMat` with element, rather than byte, stride.
#[derive(Clone, Debug, PartialEq)]
pub struct CvArray<T> {
    pub rows: usize,
    pub columns: usize,
    pub channels: usize,
    pub row_stride: usize,
    pub data: Vec<T>,
}

impl<T: Default + Clone> CvArray<T> {
    /// C `cvCreateMat`/`cvCreateMatHeader`/`cvCreateData`.
    pub fn new(rows: usize, columns: usize, channels: usize) -> Result<Self, CvStatus> {
        Self::with_stride(
            rows,
            columns,
            channels,
            columns
                .checked_mul(channels)
                .ok_or(CvStatus::sts_bad_size)?,
        )
    }
    pub fn with_stride(
        rows: usize,
        columns: usize,
        channels: usize,
        row_stride: usize,
    ) -> Result<Self, CvStatus> {
        if rows == 0
            || columns == 0
            || channels == 0
            || row_stride
                < columns
                    .checked_mul(channels)
                    .ok_or(CvStatus::sts_bad_size)?
        {
            return Err(CvStatus::sts_bad_size);
        }
        Ok(Self {
            rows,
            columns,
            channels,
            row_stride,
            data: vec![T::default(); rows.checked_mul(row_stride).ok_or(CvStatus::sts_bad_size)?],
        })
    }
}
impl<T> CvArray<T> {
    /// C `cvReleaseMat`/`cvReleaseData`: consuming owned storage is release.
    pub fn release(self) {}
    /// C `cvCloneMat`.
    pub fn clone_mat(&self) -> Self
    where
        T: Clone,
    {
        self.clone()
    }
    /// C `cvGetRawData`/`cvGetSize`/`cvGetDims`.
    pub fn raw_data(&self) -> (&[T], usize) {
        (&self.data, self.row_stride)
    }
    pub fn size(&self) -> (usize, usize) {
        (self.columns, self.rows)
    }
    pub fn dimensions(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }
    /// C `cvGetRows`, `cvGetCols`, and `cvGetDiag`.
    pub fn rows(&self, start: usize, end: usize) -> Result<Self, CvStatus>
    where
        T: Clone,
    {
        self.sub_rect(CvRect {
            x: 0,
            y: start as i32,
            width: self.columns as i32,
            height: end.saturating_sub(start) as i32,
        })
    }
    pub fn columns(&self, start: usize, end: usize) -> Result<Self, CvStatus>
    where
        T: Clone,
    {
        self.sub_rect(CvRect {
            x: start as i32,
            y: 0,
            width: end.saturating_sub(start) as i32,
            height: self.rows as i32,
        })
    }
    pub fn diagonal(&self, diagonal: isize) -> Result<Vec<&[T]>, CvStatus> {
        let start_row = if diagonal < 0 {
            diagonal.unsigned_abs()
        } else {
            0
        };
        let start_column = if diagonal > 0 { diagonal as usize } else { 0 };
        if start_row >= self.rows || start_column >= self.columns {
            return Err(CvStatus::sts_bad_size);
        };
        Ok(
            (0..(self.rows - start_row).min(self.columns - start_column))
                .map(|offset| self.get(start_row + offset, start_column + offset).unwrap())
                .collect(),
        )
    }
    /// C `cvInitMatHeader`/`cvSetData`.
    pub fn from_data(
        rows: usize,
        columns: usize,
        channels: usize,
        row_stride: usize,
        data: Vec<T>,
    ) -> Result<Self, CvStatus> {
        if rows == 0
            || columns == 0
            || channels == 0
            || row_stride
                < columns
                    .checked_mul(channels)
                    .ok_or(CvStatus::sts_bad_size)?
            || data.len()
                < (rows - 1)
                    .checked_mul(row_stride)
                    .and_then(|v| v.checked_add(columns * channels))
                    .ok_or(CvStatus::sts_bad_size)?
        {
            return Err(CvStatus::sts_bad_size);
        }
        Ok(Self {
            rows,
            columns,
            channels,
            row_stride,
            data,
        })
    }
    pub fn get(&self, row: usize, column: usize) -> Option<&[T]> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        let start = row
            .checked_mul(self.row_stride)?
            .checked_add(column.checked_mul(self.channels)?)?;
        self.data.get(start..start.checked_add(self.channels)?)
    }
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut [T]> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        let start = row
            .checked_mul(self.row_stride)?
            .checked_add(column.checked_mul(self.channels)?)?;
        self.data.get_mut(start..start.checked_add(self.channels)?)
    }
    /// C `cvGetSubRect`, `cvGetRows`, `cvGetCols`: safe owned materialized view.
    pub fn sub_rect(&self, rect: CvRect) -> Result<Self, CvStatus>
    where
        T: Clone,
    {
        if rect.x < 0
            || rect.y < 0
            || rect.width <= 0
            || rect.height <= 0
            || rect.x as usize + rect.width as usize > self.columns
            || rect.y as usize + rect.height as usize > self.rows
        {
            return Err(CvStatus::sts_bad_size);
        }
        let mut data =
            Vec::with_capacity(rect.width as usize * rect.height as usize * self.channels);
        for row in rect.y as usize..(rect.y + rect.height) as usize {
            data.extend_from_slice(
                self.get(row, rect.x as usize)
                    .unwrap()
                    .iter()
                    .chain(
                        (rect.x as usize + 1..(rect.x + rect.width) as usize)
                            .flat_map(|column| self.get(row, column).unwrap()),
                    )
                    .cloned()
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
        }
        Self::from_data(
            rect.height as usize,
            rect.width as usize,
            self.channels,
            rect.width as usize * self.channels,
            data,
        )
    }
    /// C `cvReshape` and `cvReshapeMatND` for contiguous logical values.
    pub fn reshape(&self, rows: usize, channels: usize) -> Result<Self, CvStatus>
    where
        T: Clone,
    {
        if rows == 0 || channels == 0 {
            return Err(CvStatus::sts_bad_size);
        }
        let values: Vec<T> = (0..self.rows)
            .flat_map(|row| {
                (0..self.columns)
                    .flat_map(move |column| self.get(row, column).unwrap().iter().cloned())
            })
            .collect();
        if values.len() % (rows * channels) != 0 {
            return Err(CvStatus::sts_bad_size);
        }
        Self::from_data(
            rows,
            values.len() / (rows * channels),
            channels,
            values.len() / rows,
            values,
        )
    }
    /// C `cvReshapeMatND`, materialized into owned N-D storage.
    pub fn reshape_nd(
        &self,
        dimensions: Vec<usize>,
        channels: usize,
    ) -> Result<CvArrayNd<T>, CvStatus>
    where
        T: Clone,
    {
        let values: Vec<T> = (0..self.rows)
            .flat_map(|row| {
                (0..self.columns)
                    .flat_map(move |column| self.get(row, column).unwrap().iter().cloned())
            })
            .collect();
        CvArrayNd::from_data(dimensions, channels, values)
    }
}

/// Owned C `CvMatND`.
#[derive(Clone, Debug, PartialEq)]
pub struct CvArrayNd<T> {
    pub dimensions: Vec<usize>,
    pub channels: usize,
    pub data: Vec<T>,
}
impl<T: Default + Clone> CvArrayNd<T> {
    pub fn new(dimensions: Vec<usize>, channels: usize) -> Result<Self, CvStatus> {
        let count = dimensions
            .iter()
            .try_fold(channels, |n, &d| n.checked_mul(d))
            .ok_or(CvStatus::sts_bad_size)?;
        if dimensions.is_empty() || dimensions.iter().any(|&d| d == 0) || channels == 0 {
            return Err(CvStatus::sts_bad_size);
        }
        Ok(Self {
            dimensions,
            channels,
            data: vec![T::default(); count],
        })
    }
}
impl<T> CvArrayNd<T> {
    pub fn release(self) {}
    pub fn clone_mat_nd(&self) -> Self
    where
        T: Clone,
    {
        self.clone()
    }
    /// C `cvInitMatNDHeader`/`cvSetData` with owned data replacing C's pointer.
    pub fn from_data(
        dimensions: Vec<usize>,
        channels: usize,
        data: Vec<T>,
    ) -> Result<Self, CvStatus> {
        if dimensions.is_empty()
            || dimensions.iter().any(|&dimension| dimension == 0)
            || channels == 0
        {
            return Err(CvStatus::sts_bad_size);
        }
        let count = dimensions
            .iter()
            .try_fold(channels, |total, &dimension| total.checked_mul(dimension))
            .ok_or(CvStatus::sts_bad_size)?;
        if data.len() != count {
            return Err(CvStatus::sts_bad_size);
        }
        Ok(Self {
            dimensions,
            channels,
            data,
        })
    }
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }
    pub fn offset(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.dimensions.len() {
            return None;
        }
        let mut offset = 0;
        for (&value, &size) in index.iter().zip(&self.dimensions) {
            if value >= size {
                return None;
            }
            offset = offset * size + value;
        }
        Some(offset * self.channels)
    }
    pub fn get(&self, index: &[usize]) -> Option<&[T]> {
        let offset = self.offset(index)?;
        self.data.get(offset..offset.checked_add(self.channels)?)
    }
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut [T]> {
        let offset = self.offset(index)?;
        self.data
            .get_mut(offset..offset.checked_add(self.channels)?)
    }
}

/// C `cvInitNArrayIterator`/`cvNextNArraySlice`, represented by contiguous
/// first-dimension slices of an owned N-D array.
pub fn cv_n_array_slices<T>(array: &CvArrayNd<T>) -> Vec<&[T]> {
    let width = array.dimensions.last().copied().unwrap_or(0) * array.channels;
    if width == 0 {
        return Vec::new();
    }
    array.data.chunks(width).collect()
}

/// Owned state for C `CvNArrayIterator`.  Each yielded item contains the
/// corresponding contiguous slice from every input array.
pub struct CvNArrayIterator<'a, T> {
    arrays: Vec<&'a CvArrayNd<T>>,
    next_slice: usize,
    slice_len: usize,
    slice_count: usize,
}

/// C `cvInitNArrayIterator`.  Rust borrows make the temporary stub headers
/// and pointer arrays unnecessary.
pub fn cv_init_n_array_iterator<'a, T>(
    arrays: Vec<&'a CvArrayNd<T>>,
) -> Result<CvNArrayIterator<'a, T>, CvStatus> {
    let first = arrays.first().ok_or(CvStatus::sts_out_of_range)?;
    if arrays
        .iter()
        .any(|array| array.dimensions != first.dimensions || array.channels != first.channels)
    {
        return Err(CvStatus::sts_unmatched_sizes);
    }
    let slice_len = first
        .dimensions
        .last()
        .copied()
        .ok_or(CvStatus::sts_bad_size)?
        .checked_mul(first.channels)
        .ok_or(CvStatus::sts_bad_size)?;
    Ok(CvNArrayIterator {
        slice_count: first.data.len() / slice_len,
        arrays,
        next_slice: 0,
        slice_len,
    })
}

/// C `cvNextNArraySlice`.
pub fn cv_next_n_array_slice<'a, T>(
    iterator: &mut CvNArrayIterator<'a, T>,
) -> Option<Vec<&'a [T]>> {
    let start = iterator.next_slice.checked_mul(iterator.slice_len)?;
    if iterator.next_slice >= iterator.slice_count {
        return None;
    }
    iterator.next_slice += 1;
    Some(
        iterator
            .arrays
            .iter()
            .map(|array| &array.data[start..start + iterator.slice_len])
            .collect(),
    )
}

/// Logical scalar depth and channel count returned by C `cvGetElemType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvElementDepth {
    U8,
    I8,
    U16,
    I16,
    I32,
    F32,
    F64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CvElementType {
    pub depth: CvElementDepth,
    pub channels: usize,
}

/// Scalar access trait for C `cvGet*`, `cvSet*`, and real-value conversion.
pub trait CvArrayValue: Copy + Default {
    const DEPTH: CvElementDepth;
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}
macro_rules! impl_array_value { ($($type:ty => $depth:ident),+ $(,)?) => { $(impl CvArrayValue for $type { const DEPTH: CvElementDepth = CvElementDepth::$depth; fn to_f64(self)->f64{self as f64} fn from_f64(value:f64)->Self{value as $type} })+ }; }
impl_array_value!(u8 => U8, i8 => I8, u16 => U16, i16 => I16, i32 => I32, f32 => F32, f64 => F64);

impl<T: CvArrayValue> CvArray<T> {
    /// C `cvGet1D`/`cvGet2D` and `cvGetReal*`.
    pub fn get_scalar(&self, row: usize, column: usize) -> Option<CvScalar> {
        Some(cv_raw_to_scalar(self.get(row, column)?, T::to_f64))
    }
    pub fn get_real(&self, row: usize, column: usize) -> Option<f64> {
        self.get(row, column)?.first().copied().map(T::to_f64)
    }
    /// C `cvSet1D`/`cvSet2D` and `cvSetReal*`.
    pub fn set_scalar(
        &mut self,
        row: usize,
        column: usize,
        scalar: CvScalar,
    ) -> Result<(), CvStatus> {
        let values = cv_scalar_to_raw(scalar, T::from_f64, self.channels);
        self.get_mut(row, column)
            .ok_or(CvStatus::sts_bad_size)?
            .copy_from_slice(&values);
        Ok(())
    }
    pub fn set_real(&mut self, row: usize, column: usize, value: f64) -> Result<(), CvStatus> {
        self.get_mut(row, column).ok_or(CvStatus::sts_bad_size)?[0] = T::from_f64(value);
        Ok(())
    }
    /// C `cvClearND` for one 2-D element.
    pub fn clear_element(&mut self, row: usize, column: usize) -> Result<(), CvStatus> {
        for value in self.get_mut(row, column).ok_or(CvStatus::sts_bad_size)? {
            *value = T::default();
        }
        Ok(())
    }
}
impl<T: CvArrayValue> CvArrayNd<T> {
    pub fn get_scalar(&self, index: &[usize]) -> Option<CvScalar> {
        Some(cv_raw_to_scalar(self.get(index)?, T::to_f64))
    }
    pub fn get_real(&self, index: &[usize]) -> Option<f64> {
        self.get(index)?.first().copied().map(T::to_f64)
    }
    pub fn set_scalar(&mut self, index: &[usize], scalar: CvScalar) -> Result<(), CvStatus> {
        let channels = self.channels;
        let values = cv_scalar_to_raw(scalar, T::from_f64, channels);
        self.get_mut(index)
            .ok_or(CvStatus::sts_bad_size)?
            .copy_from_slice(&values);
        Ok(())
    }
    pub fn set_real(&mut self, index: &[usize], value: f64) -> Result<(), CvStatus> {
        self.get_mut(index).ok_or(CvStatus::sts_bad_size)?[0] = T::from_f64(value);
        Ok(())
    }
    pub fn clear(&mut self, index: &[usize]) -> Result<(), CvStatus> {
        for value in self.get_mut(index).ok_or(CvStatus::sts_bad_size)? {
            *value = T::default();
        }
        Ok(())
    }
}

/// Owned C `CvSparseMat`.
#[derive(Clone, Debug, PartialEq)]
pub struct CvSparseArray<T> {
    pub dimensions: Vec<usize>,
    pub channels: usize,
    pub values: BTreeMap<Vec<usize>, Vec<T>>,
}
impl<T> CvSparseArray<T> {
    pub fn new(dimensions: Vec<usize>, channels: usize) -> Result<Self, CvStatus> {
        if dimensions.is_empty() || dimensions.iter().any(|&d| d == 0) || channels == 0 {
            Err(CvStatus::sts_bad_size)
        } else {
            Ok(Self {
                dimensions,
                channels,
                values: BTreeMap::new(),
            })
        }
    }
    pub fn get(&self, index: &[usize]) -> Option<&[T]> {
        self.values.get(index).map(Vec::as_slice)
    }
    pub fn insert(&mut self, index: Vec<usize>, value: Vec<T>) -> Result<(), CvStatus> {
        if index.len() != self.dimensions.len()
            || index.iter().zip(&self.dimensions).any(|(&i, &d)| i >= d)
            || value.len() != self.channels
        {
            Err(CvStatus::sts_bad_arg)
        } else {
            self.values.insert(index, value);
            Ok(())
        }
    }
    pub fn clear(&mut self) {
        self.values.clear()
    }
}

/// Owned C `IplImage` with ROI and channel-of-interest metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct CvImage<T> {
    pub array: CvArray<T>,
    pub roi: Option<CvRect>,
    pub coi: Option<usize>,
}
/// C IPL allocator callback configuration, retained as safe construction hooks.
pub struct CvIplAllocators<T> {
    pub create_image:
        Option<Box<dyn Fn(usize, usize, usize) -> Result<CvImage<T>, CvStatus> + Send + Sync>>,
}
impl<T> Default for CvIplAllocators<T> {
    fn default() -> Self {
        Self { create_image: None }
    }
}
/// C `cvSetIPLAllocators`; callers retain the hook rather than installing a
/// process-global raw callback.
pub fn cv_set_ipl_allocators<T>(
    create_image: Option<
        Box<dyn Fn(usize, usize, usize) -> Result<CvImage<T>, CvStatus> + Send + Sync>,
    >,
) -> CvIplAllocators<T> {
    CvIplAllocators { create_image }
}
impl<T: Default + Clone> CvImage<T> {
    pub fn new(width: usize, height: usize, channels: usize) -> Result<Self, CvStatus> {
        Ok(Self {
            array: CvArray::new(height, width, channels)?,
            roi: None,
            coi: None,
        })
    }
}
impl<T: Clone> CvImage<T> {
    pub fn set_roi(&mut self, roi: CvRect) -> Result<(), CvStatus> {
        self.array.sub_rect(roi)?;
        self.roi = Some(roi);
        Ok(())
    }
    pub fn reset_roi(&mut self) {
        self.roi = None
    }
    pub fn set_coi(&mut self, coi: Option<usize>) -> Result<(), CvStatus> {
        if coi.is_some_and(|value| value == 0 || value > self.array.channels) {
            Err(CvStatus::bad_coi)
        } else {
            self.coi = coi;
            Ok(())
        }
    }
    /// C `cvGetImageROI`.
    pub fn roi(&self) -> CvRect {
        self.roi.unwrap_or(CvRect {
            x: 0,
            y: 0,
            width: self.array.columns as i32,
            height: self.array.rows as i32,
        })
    }
    /// C `cvCloneImage`.
    pub fn clone_image(&self) -> Self {
        self.clone()
    }
    /// C `cvGetImage` materializes image metadata from a matrix.
    pub fn from_array(array: CvArray<T>) -> Self {
        Self {
            array,
            roi: None,
            coi: None,
        }
    }
    /// C `cvGetMat`: image ROI materialized as an owned matrix.
    pub fn get_mat(&self) -> Result<CvArray<T>, CvStatus> {
        self.array.sub_rect(self.roi())
    }
}

/// C `cvGetMat` for an owned dense matrix is identity cloning; C's temporary
/// header is unnecessary because the returned value owns its storage.
pub fn cv_get_mat<T: Clone>(array: &CvArray<T>) -> CvArray<T> {
    array.clone_mat()
}
pub fn cv_create_mat<T: Default + Clone>(
    rows: usize,
    columns: usize,
    channels: usize,
) -> Result<CvArray<T>, CvStatus> {
    CvArray::new(rows, columns, channels)
}
pub fn cv_create_mat_header<T>(
    rows: usize,
    columns: usize,
    channels: usize,
    row_stride: usize,
    data: Vec<T>,
) -> Result<CvArray<T>, CvStatus> {
    CvArray::from_data(rows, columns, channels, row_stride, data)
}
pub fn cv_init_mat_header<T>(
    rows: usize,
    columns: usize,
    channels: usize,
    row_stride: usize,
    data: Vec<T>,
) -> Result<CvArray<T>, CvStatus> {
    cv_create_mat_header(rows, columns, channels, row_stride, data)
}
pub fn cv_release_mat<T>(array: CvArray<T>) {
    array.release()
}
pub fn cv_clone_mat<T: Clone>(array: &CvArray<T>) -> CvArray<T> {
    array.clone_mat()
}
pub fn cv_create_mat_nd<T: Default + Clone>(
    dimensions: Vec<usize>,
    channels: usize,
) -> Result<CvArrayNd<T>, CvStatus> {
    CvArrayNd::new(dimensions, channels)
}
pub fn cv_create_mat_nd_header<T>(
    dimensions: Vec<usize>,
    channels: usize,
    data: Vec<T>,
) -> Result<CvArrayNd<T>, CvStatus> {
    CvArrayNd::from_data(dimensions, channels, data)
}
pub fn cv_init_mat_nd_header<T>(
    dimensions: Vec<usize>,
    channels: usize,
    data: Vec<T>,
) -> Result<CvArrayNd<T>, CvStatus> {
    cv_create_mat_nd_header(dimensions, channels, data)
}
pub fn cv_clone_mat_nd<T: Clone>(array: &CvArrayNd<T>) -> CvArrayNd<T> {
    array.clone_mat_nd()
}
pub fn cv_release_mat_nd<T>(array: CvArrayNd<T>) {
    array.release()
}
pub fn cv_create_sparse_mat<T>(
    dimensions: Vec<usize>,
    channels: usize,
) -> Result<CvSparseArray<T>, CvStatus> {
    CvSparseArray::new(dimensions, channels)
}
pub fn cv_clone_sparse_mat<T: Clone>(array: &CvSparseArray<T>) -> CvSparseArray<T> {
    array.clone()
}
pub fn cv_release_sparse_mat<T>(_array: CvSparseArray<T>) {}

/// C `cvCreateData` in the owned model.  Header and data are created together.
pub fn cv_create_data<T: Default + Clone>(
    rows: usize,
    columns: usize,
    channels: usize,
) -> Result<CvArray<T>, CvStatus> {
    cv_create_mat(rows, columns, channels)
}
/// C `cvSetData`; ownership replaces a raw external pointer and its lifetime.
pub fn cv_set_data<T>(
    array: &mut CvArray<T>,
    row_stride: usize,
    data: Vec<T>,
) -> Result<(), CvStatus> {
    *array = CvArray::from_data(array.rows, array.columns, array.channels, row_stride, data)?;
    Ok(())
}
/// C `cvReleaseData`; clearing is the non-destructive equivalent for a retained Rust header.
pub fn cv_release_data<T>(array: &mut CvArray<T>) {
    array.data.clear();
}
pub fn cv_get_raw_data<T>(array: &CvArray<T>) -> (&[T], usize) {
    array.raw_data()
}
pub trait CvElementTyped {
    type Element: CvArrayValue;
    fn cv_channels(&self) -> usize;
}
impl<T: CvArrayValue> CvElementTyped for CvArray<T> {
    type Element = T;
    fn cv_channels(&self) -> usize {
        self.channels
    }
}
impl<T: CvArrayValue> CvElementTyped for CvArrayNd<T> {
    type Element = T;
    fn cv_channels(&self) -> usize {
        self.channels
    }
}
impl<T: CvArrayValue> CvElementTyped for CvSparseArray<T> {
    type Element = T;
    fn cv_channels(&self) -> usize {
        self.channels
    }
}
impl<T: CvArrayValue> CvElementTyped for CvImage<T> {
    type Element = T;
    fn cv_channels(&self) -> usize {
        self.array.channels
    }
}
/// C `cvGetElemType`, generalized over all owned array headers.
pub fn cv_get_elem_type<A: CvElementTyped>(array: &A) -> CvElementType {
    CvElementType {
        depth: A::Element::DEPTH,
        channels: array.cv_channels(),
    }
}
pub trait CvDimensions {
    fn cv_dimensions(&self) -> Vec<usize>;
}
impl<T> CvDimensions for CvArray<T> {
    fn cv_dimensions(&self) -> Vec<usize> {
        self.dimensions().to_vec()
    }
}
impl<T> CvDimensions for CvArrayNd<T> {
    fn cv_dimensions(&self) -> Vec<usize> {
        self.dimensions.clone()
    }
}
impl<T> CvDimensions for CvSparseArray<T> {
    fn cv_dimensions(&self) -> Vec<usize> {
        self.dimensions.clone()
    }
}
impl<T> CvDimensions for CvImage<T> {
    fn cv_dimensions(&self) -> Vec<usize> {
        self.array.dimensions().to_vec()
    }
}
/// C `cvGetDims`, with a Rust-owned dimension list instead of a caller buffer.
pub fn cv_get_dims<A: CvDimensions>(array: &A) -> Vec<usize> {
    array.cv_dimensions()
}
pub fn cv_get_dim_size<A: CvDimensions>(array: &A, dimension: usize) -> Option<usize> {
    array.cv_dimensions().get(dimension).copied()
}
pub trait CvSizeLike {
    fn cv_size(&self) -> (usize, usize);
}
impl<T> CvSizeLike for CvArray<T> {
    fn cv_size(&self) -> (usize, usize) {
        self.size()
    }
}
impl<T> CvSizeLike for CvArrayNd<T> {
    fn cv_size(&self) -> (usize, usize) {
        let width = self.dimensions.last().copied().unwrap_or(0);
        let height = self.dimensions[..self.dimensions.len().saturating_sub(1)]
            .iter()
            .product();
        (width, height)
    }
}
impl<T> CvSizeLike for CvImage<T> {
    fn cv_size(&self) -> (usize, usize) {
        self.array.size()
    }
}
/// C `cvGetSize` projected as OpenCV's width/height pair for all dense forms.
pub fn cv_get_size<A: CvSizeLike>(array: &A) -> (usize, usize) {
    array.cv_size()
}
pub fn cv_get_sub_rect<T: Clone>(array: &CvArray<T>, rect: CvRect) -> Result<CvArray<T>, CvStatus> {
    array.sub_rect(rect)
}
pub fn cv_get_rows<T: Clone>(
    array: &CvArray<T>,
    start: usize,
    end: usize,
) -> Result<CvArray<T>, CvStatus> {
    array.rows(start, end)
}
pub fn cv_get_cols<T: Clone>(
    array: &CvArray<T>,
    start: usize,
    end: usize,
) -> Result<CvArray<T>, CvStatus> {
    array.columns(start, end)
}
pub fn cv_get_diag<T>(array: &CvArray<T>, diagonal: isize) -> Result<Vec<&[T]>, CvStatus> {
    array.diagonal(diagonal)
}

/// Safe borrowed equivalents of C's `cvPtr*` raw-pointer APIs.
pub fn cv_ptr_1d<T>(array: &CvArray<T>, index: usize) -> Option<&[T]> {
    array.get(0, index)
}
pub fn cv_ptr_2d<T>(array: &CvArray<T>, row: usize, column: usize) -> Option<&[T]> {
    array.get(row, column)
}
pub fn cv_ptr_3d<T>(
    array: &CvArrayNd<T>,
    first: usize,
    second: usize,
    third: usize,
) -> Option<&[T]> {
    array.get(&[first, second, third])
}
pub fn cv_ptr_nd<'a, T>(array: &'a CvArrayNd<T>, index: &[usize]) -> Option<&'a [T]> {
    array.get(index)
}

pub fn cv_get_1d<T: CvArrayValue>(array: &CvArray<T>, index: usize) -> Option<CvScalar> {
    array.get_scalar(0, index)
}
pub fn cv_get_2d<T: CvArrayValue>(
    array: &CvArray<T>,
    row: usize,
    column: usize,
) -> Option<CvScalar> {
    array.get_scalar(row, column)
}
pub fn cv_get_3d<T: CvArrayValue>(
    array: &CvArrayNd<T>,
    first: usize,
    second: usize,
    third: usize,
) -> Option<CvScalar> {
    array.get_scalar(&[first, second, third])
}
pub fn cv_get_nd<T: CvArrayValue>(array: &CvArrayNd<T>, index: &[usize]) -> Option<CvScalar> {
    array.get_scalar(index)
}
pub fn cv_get_real_1d<T: CvArrayValue>(array: &CvArray<T>, index: usize) -> Option<f64> {
    array.get_real(0, index)
}
pub fn cv_get_real_2d<T: CvArrayValue>(
    array: &CvArray<T>,
    row: usize,
    column: usize,
) -> Option<f64> {
    array.get_real(row, column)
}
pub fn cv_get_real_3d<T: CvArrayValue>(
    array: &CvArrayNd<T>,
    first: usize,
    second: usize,
    third: usize,
) -> Option<f64> {
    array.get_real(&[first, second, third])
}
pub fn cv_get_real_nd<T: CvArrayValue>(array: &CvArrayNd<T>, index: &[usize]) -> Option<f64> {
    array.get_real(index)
}
pub fn cv_set_1d<T: CvArrayValue>(
    array: &mut CvArray<T>,
    index: usize,
    value: CvScalar,
) -> Result<(), CvStatus> {
    array.set_scalar(0, index, value)
}
pub fn cv_set_2d<T: CvArrayValue>(
    array: &mut CvArray<T>,
    row: usize,
    column: usize,
    value: CvScalar,
) -> Result<(), CvStatus> {
    array.set_scalar(row, column, value)
}
pub fn cv_set_3d<T: CvArrayValue>(
    array: &mut CvArrayNd<T>,
    first: usize,
    second: usize,
    third: usize,
    value: CvScalar,
) -> Result<(), CvStatus> {
    array.set_scalar(&[first, second, third], value)
}
pub fn cv_set_nd<T: CvArrayValue>(
    array: &mut CvArrayNd<T>,
    index: &[usize],
    value: CvScalar,
) -> Result<(), CvStatus> {
    array.set_scalar(index, value)
}
pub fn cv_set_real_1d<T: CvArrayValue>(
    array: &mut CvArray<T>,
    index: usize,
    value: f64,
) -> Result<(), CvStatus> {
    array.set_real(0, index, value)
}
pub fn cv_set_real_2d<T: CvArrayValue>(
    array: &mut CvArray<T>,
    row: usize,
    column: usize,
    value: f64,
) -> Result<(), CvStatus> {
    array.set_real(row, column, value)
}
pub fn cv_set_real_3d<T: CvArrayValue>(
    array: &mut CvArrayNd<T>,
    first: usize,
    second: usize,
    third: usize,
    value: f64,
) -> Result<(), CvStatus> {
    array.set_real(&[first, second, third], value)
}
pub fn cv_set_real_nd<T: CvArrayValue>(
    array: &mut CvArrayNd<T>,
    index: &[usize],
    value: f64,
) -> Result<(), CvStatus> {
    array.set_real(index, value)
}
pub fn cv_clear_nd<T: CvArrayValue>(
    array: &mut CvArrayNd<T>,
    index: &[usize],
) -> Result<(), CvStatus> {
    array.clear(index)
}
pub fn cv_reshape<T: Clone>(
    array: &CvArray<T>,
    rows: usize,
    channels: usize,
) -> Result<CvArray<T>, CvStatus> {
    array.reshape(rows, channels)
}
pub fn cv_reshape_mat_nd<T: Clone>(
    array: &CvArray<T>,
    dimensions: Vec<usize>,
    channels: usize,
) -> Result<CvArrayNd<T>, CvStatus> {
    array.reshape_nd(dimensions, channels)
}

/// C `cvGetMat` when its `CvArr` is an image.
pub fn cv_get_image_mat<T: Clone>(image: &CvImage<T>) -> Result<CvArray<T>, CvStatus> {
    image.get_mat()
}

/// C `cvGetImage`: materializes a safe owned image header/data pair.
pub fn cv_get_image<T: Clone>(array: &CvArray<T>) -> CvImage<T> {
    CvImage::from_array(array.clone_mat())
}

/// C `cvCreateImageHeader`/`cvInitImageHeader`/`cvCreateImage` are represented
/// by this checked constructor; `T::default` initializes C's allocated bytes.
pub fn cv_create_image<T: Default + Clone>(
    width: usize,
    height: usize,
    channels: usize,
) -> Result<CvImage<T>, CvStatus> {
    CvImage::new(width, height, channels)
}
pub fn cv_create_image_header<T: Default + Clone>(
    width: usize,
    height: usize,
    channels: usize,
) -> Result<CvImage<T>, CvStatus> {
    cv_create_image(width, height, channels)
}
pub fn cv_init_image_header<T: Default + Clone>(
    image: &mut CvImage<T>,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<(), CvStatus> {
    *image = cv_create_image(width, height, channels)?;
    Ok(())
}
pub fn cv_release_image_header<T>(_image: CvImage<T>) {}
pub fn cv_release_image<T>(image: CvImage<T>) {
    cv_release_image_header(image)
}
pub fn cv_clone_image<T: Clone>(image: &CvImage<T>) -> CvImage<T> {
    image.clone_image()
}
pub fn cv_set_image_roi<T: Clone>(image: &mut CvImage<T>, roi: CvRect) -> Result<(), CvStatus> {
    image.set_roi(roi)
}
pub fn cv_reset_image_roi<T: Clone>(image: &mut CvImage<T>) {
    image.reset_roi()
}
pub fn cv_get_image_roi<T: Clone>(image: &CvImage<T>) -> CvRect {
    image.roi()
}
pub fn cv_set_image_coi<T: Clone>(
    image: &mut CvImage<T>,
    coi: Option<usize>,
) -> Result<(), CvStatus> {
    image.set_coi(coi)
}
pub fn cv_get_image_coi<T>(image: &CvImage<T>) -> Option<usize> {
    image.coi
}

/// C scalar conversion/value helpers. The generic safe caller supplies a lane conversion.
pub fn cv_scalar_to_raw<T: Copy>(
    scalar: CvScalar,
    convert: impl Fn(f64) -> T,
    channels: usize,
) -> Vec<T> {
    (0..channels)
        .map(|channel| convert(scalar.values[channel.min(3)]))
        .collect()
}
pub fn cv_raw_to_scalar<T: Copy>(values: &[T], convert: impl Fn(T) -> f64) -> CvScalar {
    let mut scalar = CvScalar { values: [0.0; 4] };
    for (index, &value) in values.iter().take(4).enumerate() {
        scalar.values[index] = convert(value)
    }
    scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_headers_preserve_stride_and_scalar_lanes() {
        let mut array = cv_create_mat::<u8>(2, 3, 2).unwrap();
        cv_set_2d(
            &mut array,
            1,
            2,
            CvScalar {
                values: [7.0, 9.0, 0.0, 0.0],
            },
        )
        .unwrap();
        assert_eq!(
            cv_get_2d(&array, 1, 2).unwrap().values,
            [7.0, 9.0, 0.0, 0.0]
        );
        assert_eq!(cv_get_raw_data(&array).1, 6);
        assert_eq!(cv_get_dims(&array), vec![2, 3]);
        assert_eq!(
            cv_get_elem_type(&array),
            CvElementType {
                depth: CvElementDepth::U8,
                channels: 2
            }
        );
    }

    #[test]
    fn nd_access_reshape_and_iterator_are_checked() {
        let mut array = cv_create_mat_nd::<i32>(vec![2, 2, 3], 1).unwrap();
        cv_set_real_3d(&mut array, 1, 0, 2, 42.0).unwrap();
        assert_eq!(cv_get_real_nd(&array, &[1, 0, 2]), Some(42.0));
        let copy = cv_clone_mat_nd(&array);
        let mut iterator = cv_init_n_array_iterator(vec![&array, &copy]).unwrap();
        assert_eq!(cv_next_n_array_slice(&mut iterator).unwrap()[0].len(), 3);
        assert_eq!(cv_get_dim_size(&array, 2), Some(3));
    }

    #[test]
    fn image_roi_is_owned_and_coi_checked() {
        let mut image = cv_create_image::<f32>(4, 3, 2).unwrap();
        cv_set_image_roi(
            &mut image,
            CvRect {
                x: 1,
                y: 1,
                width: 2,
                height: 2,
            },
        )
        .unwrap();
        cv_set_image_coi(&mut image, Some(2)).unwrap();
        assert_eq!(cv_get_image_mat(&image).unwrap().size(), (2, 2));
        assert!(cv_set_image_coi(&mut image, Some(3)).is_err());
    }
}
