//! Translation of `IMOD/raptor/optimization/estimation3ddata.{h,cpp}`.

use super::std_qp_data::Matrix;

/// Projection-model state formerly held in three OpenCV `CvMat` allocations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Estimation3dData {
    pub resid_mean_perc: f64,
    pub resid_mean: f64,
    pub g: Option<Matrix<f64>>,
    pub p: Option<Matrix<f64>>,
    pub t: Option<Matrix<f64>>,
}

impl Estimation3dData {
    /// C++ `estimation3ddata::estimation3ddata()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// C++ `estimation3ddata::estimation3ddata(int T, int M)`.
    pub fn with_dimensions(views: usize, markers: usize) -> Self {
        Self {
            g: Some(Matrix::new(2 * views, 3, vec![0.0; 6 * views])),
            t: Some(Matrix::new(2 * views, 1, vec![0.0; 2 * views])),
            p: Some(Matrix::new(3, markers, vec![0.0; 3 * markers])),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Estimation3dData;

    #[test]
    fn default_constructor_has_no_matrix_allocations() {
        let data = Estimation3dData::new();
        assert!(data.g.is_none() && data.p.is_none() && data.t.is_none());
    }

    #[test]
    fn dimensional_constructor_uses_source_matrix_shapes_and_zeroes() {
        let data = Estimation3dData::with_dimensions(4, 7);
        assert_eq!(
            (
                data.g.as_ref().unwrap().rows,
                data.g.as_ref().unwrap().columns
            ),
            (8, 3)
        );
        assert_eq!(
            (
                data.t.as_ref().unwrap().rows,
                data.t.as_ref().unwrap().columns
            ),
            (8, 1)
        );
        assert_eq!(
            (
                data.p.as_ref().unwrap().rows,
                data.p.as_ref().unwrap().columns
            ),
            (3, 7)
        );
        assert!(data.g.unwrap().values.iter().all(|value| *value == 0.0));
    }
}
