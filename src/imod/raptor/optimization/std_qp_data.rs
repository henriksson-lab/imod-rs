//! Translation of `IMOD/raptor/optimization/STDQPdata.{h,cpp}`.

/// Owned replacement for OpenCV's heap-allocated `CvMat`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<T> {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn new(rows: usize, columns: usize, values: Vec<T>) -> Self {
        assert_eq!(values.len(), rows * columns);
        Self {
            rows,
            columns,
            values,
        }
    }
}

/// C++ `STDQPdata`, with native ownership replacing `CvMat *`.
#[derive(Clone, Debug, PartialEq)]
pub struct StdQpData<T> {
    pub answer_mat: Option<Matrix<T>>,
    pub num_itrs: i32,
    pub exit_flag: i32,
    pub gap: f64,
}

impl<T> Default for StdQpData<T> {
    fn default() -> Self {
        Self {
            answer_mat: None,
            num_itrs: 0,
            exit_flag: 0,
            gap: 0.0,
        }
    }
}

impl<T> StdQpData<T> {
    /// C++ `STDQPdata::STDQPdata`.
    pub fn new() -> Self {
        Self::default()
    }

    /// C++ `STDQPdata::clear`; dropping the owned matrix is the former
    /// `cvReleaseMat` operation. The QP result metadata is intentionally kept.
    pub fn clear(&mut self) {
        self.answer_mat = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{Matrix, StdQpData};

    #[test]
    fn default_has_no_opencv_matrix() {
        let data = StdQpData::<f32>::new();
        assert!(data.answer_mat.is_none());
        assert_eq!((data.num_itrs, data.exit_flag, data.gap), (0, 0, 0.0));
    }

    #[test]
    fn clear_drops_matrix_and_preserves_result_metadata() {
        let mut data = StdQpData {
            answer_mat: Some(Matrix::new(1, 2, vec![2.0, 3.0])),
            num_itrs: 4,
            exit_flag: -1,
            gap: 0.25,
        };
        data.clear();
        assert!(data.answer_mat.is_none());
        assert_eq!((data.num_itrs, data.exit_flag, data.gap), (4, -1, 0.25));
    }
}
