//! Translation of `IMOD/raptor/optimization/SFMdata.{h,cpp}`.

use super::contour::Contour;

/// C++ `SFMdata`, with `Option` ownership replacing nullable contour pointers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SfmData {
    pub reproj_x: Option<Contour>,
    pub reproj_y: Option<Contour>,
    pub contour_x: Option<Contour>,
    pub contour_y: Option<Contour>,
    pub marker_type: Option<Vec<i32>>,
}

impl SfmData {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn from_contours(contour_x: Contour, contour_y: Contour, marker_type: Vec<i32>) -> Self {
        assert_eq!(contour_x.num_trajectories, contour_y.num_trajectories);
        assert_eq!(marker_type.len(), contour_x.num_trajectories);
        let reproj_x = Contour {
            x_or_y: 1,
            num_trajectories: contour_x.num_trajectories,
            num_frames: contour_x.num_frames,
            scores: vec![0.0; contour_x.num_trajectories * contour_x.num_frames],
        };
        let reproj_y = Contour {
            x_or_y: 2,
            num_trajectories: contour_y.num_trajectories,
            num_frames: contour_y.num_frames,
            scores: vec![0.0; contour_y.num_trajectories * contour_y.num_frames],
        };
        Self {
            reproj_x: Some(reproj_x),
            reproj_y: Some(reproj_y),
            contour_x: Some(contour_x),
            contour_y: Some(contour_y),
            marker_type: Some(marker_type),
        }
    }
    pub fn with_reprojections(
        contour_x: Contour,
        contour_y: Contour,
        reproj_x: Contour,
        reproj_y: Contour,
        marker_type: Vec<i32>,
    ) -> Self {
        assert_eq!(marker_type.len(), contour_x.num_trajectories);
        Self {
            reproj_x: Some(reproj_x),
            reproj_y: Some(reproj_y),
            contour_x: Some(contour_x),
            contour_y: Some(contour_y),
            marker_type: Some(marker_type),
        }
    }
    /// C++ `SFMdata::clearPointers`.
    pub fn clear_pointers(&mut self) {
        self.reproj_x = None;
        self.reproj_y = None;
        self.contour_x = None;
        self.contour_y = None;
        self.marker_type = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_clones_inputs_and_zeroes_reprojections() {
        let x = Contour {
            x_or_y: 1,
            num_trajectories: 2,
            num_frames: 3,
            scores: vec![0.; 6],
        };
        let y = Contour {
            x_or_y: 2,
            num_trajectories: 2,
            num_frames: 3,
            scores: vec![0.; 6],
        };
        let data = SfmData::from_contours(x, y, vec![4, 5]);
        assert_eq!(data.reproj_x.unwrap().scores, vec![0.; 6]);
        assert_eq!(data.marker_type.unwrap(), vec![4, 5]);
    }
    #[test]
    fn clear_releases_all_owned_state() {
        let mut data = SfmData::from_contours(
            Contour {
                x_or_y: 1,
                num_trajectories: 1,
                num_frames: 1,
                scores: vec![0.],
            },
            Contour {
                x_or_y: 2,
                num_trajectories: 1,
                num_frames: 1,
                scores: vec![0.],
            },
            vec![1],
        );
        data.clear_pointers();
        assert_eq!(data, SfmData::new());
    }
}
