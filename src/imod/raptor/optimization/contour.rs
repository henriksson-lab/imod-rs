//! Safe translation of `IMOD/raptor/optimization/contour.{h,cpp}`.

use crate::imod::raptor::main_classes::frame::Frame;
use crate::imod::raptor::main_classes::point2d::Point2d;

/// C++ `contour::minContourValue`.
pub const MIN_CONTOUR_VALUE: f64 = 1.0e-6;

/// A row-major coordinate matrix for a set of trajectories.
///
/// `x_or_y` is 1 for X coordinates and 2 for Y coordinates, exactly as in
/// RAPTOR's source.  `scores` replaces the owned OpenCV `CvMat`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Contour {
    pub x_or_y: i32,
    pub num_trajectories: usize,
    pub num_frames: usize,
    pub scores: Vec<f64>,
}

impl Contour {
    /// C++ `contour::minContourValue`.
    pub const MIN_CONTOUR_VALUE: f64 = MIN_CONTOUR_VALUE;

    /// C++ `contour::contour()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// C++ `contour::contour(int, int, int)`.
    pub fn with_dimensions(rows: usize, columns: usize, x_or_y: i32) -> Self {
        Self {
            x_or_y,
            num_trajectories: rows,
            num_frames: columns,
            scores: vec![0.0; rows * columns],
        }
    }

    /// C++ `contour::contour(CvMat *, int)`, with a checked owned matrix.
    pub fn from_scores(rows: usize, columns: usize, scores: Vec<f64>, x_or_y: i32) -> Option<Self> {
        (scores.len() == rows.checked_mul(columns)?).then_some(Self {
            x_or_y,
            num_trajectories: rows,
            num_frames: columns,
            scores,
        })
    }

    /// C++ `contour(vector<trajectory>, int, int)`.
    ///
    /// A trajectory is represented by its owned points.  The source indexes
    /// matrix columns by `Point2D::frameID`; invalid frame IDs are ignored
    /// instead of causing an out-of-bounds write.
    pub fn from_trajectories<P>(
        trajectories: &[Vec<Point2d<P>>],
        coordinate: i32,
        num_frames: usize,
    ) -> Self {
        let mut contour = Self::with_dimensions(trajectories.len(), num_frames, coordinate);
        for (trajectory_index, trajectory) in trajectories.iter().enumerate() {
            for point in trajectory {
                if let Ok(frame_index) = usize::try_from(point.frame_id)
                    && frame_index < num_frames
                {
                    contour.scores[trajectory_index * num_frames + frame_index] = if coordinate == 1
                    {
                        point.x as f64
                    } else if coordinate == 2 {
                        point.y as f64
                    } else {
                        0.0
                    };
                }
            }
        }
        contour
    }

    /// C++ `contour(vector<trajectory>, int, int, vector<frame> *)`.
    ///
    /// Discarded frames are omitted from the matrix and retained frame IDs are
    /// compacted to consecutive columns.
    pub fn from_trajectories_with_frames<T, P>(
        trajectories: &[Vec<Point2d<P>>],
        coordinate: i32,
        num_frames: usize,
        frames: &[Frame<T, P>],
    ) -> Self {
        let mut positions = vec![None; num_frames];
        let mut columns = 0;
        for (frame_index, frame) in frames.iter().take(num_frames).enumerate() {
            if !frame.discard {
                positions[frame_index] = Some(columns);
                columns += 1;
            }
        }

        let mut contour = Self::with_dimensions(trajectories.len(), columns, coordinate);
        for (trajectory_index, trajectory) in trajectories.iter().enumerate() {
            for point in trajectory {
                if let Ok(frame_index) = usize::try_from(point.frame_id)
                    && let Some(Some(column)) = positions.get(frame_index)
                {
                    contour.scores[trajectory_index * columns + column] = if coordinate == 1 {
                        point.x as f64
                    } else if coordinate == 2 {
                        point.y as f64
                    } else {
                        0.0
                    };
                }
            }
        }
        contour
    }

    /// C++ `contour::getNumTraj`.
    pub fn get_num_traj(&self) -> usize {
        self.num_trajectories
    }

    /// C++ `contour::getNumFrame`.
    pub fn get_num_frame(&self) -> usize {
        self.num_frames
    }

    /// C++ `contour::calculateNNZ`.
    pub fn calculate_nnz(&self) -> usize {
        self.scores
            .iter()
            .filter(|&&value| value > MIN_CONTOUR_VALUE)
            .count()
    }

    /// C++ `contour::print`, returned as text rather than writing an ostream.
    pub fn print(&self) -> String {
        let mut output = String::new();
        for row_index in 0..self.num_trajectories {
            let start = row_index * self.num_frames;
            let row = &self.scores[start..start + self.num_frames];
            for value in row {
                output.push_str(&format!("{value}  "));
            }
            output.push('\n');
        }
        output.push('\n');
        output
    }

    /// C++ `contour::contour2trajectory`.
    ///
    /// Returns `None` for source conditions that terminated the process:
    /// inconsistent matrix shapes or a different number of retained frames
    /// than contour columns.
    pub fn contour_to_trajectories<T, P>(
        contour_x: &Self,
        contour_y: &Self,
        frames: &[Frame<T, P>],
    ) -> Option<Vec<Vec<Point2d>>> {
        if contour_x.num_trajectories != contour_y.num_trajectories
            || contour_x.num_frames != contour_y.num_frames
            || contour_x.scores.len()
                != contour_x
                    .num_trajectories
                    .checked_mul(contour_x.num_frames)?
            || contour_y.scores.len()
                != contour_y
                    .num_trajectories
                    .checked_mul(contour_y.num_frames)?
        {
            return None;
        }

        let frame_ids: Vec<i32> = frames
            .iter()
            .filter(|frame| !frame.discard)
            .map(|frame| frame.frame_id)
            .collect();
        if frame_ids.len() != contour_x.num_frames {
            return None;
        }

        let mut trajectories = Vec::with_capacity(contour_x.num_trajectories);
        for row in 0..contour_x.num_trajectories {
            let mut trajectory = Vec::new();
            for column in 0..contour_x.num_frames {
                let index = row * contour_x.num_frames + column;
                if contour_x.scores[index] > MIN_CONTOUR_VALUE {
                    let mut point = Point2d::default();
                    point.x = contour_x.scores[index] as f32;
                    point.y = contour_y.scores[index] as f32;
                    point.frame_id = frame_ids[column];
                    trajectory.push(point);
                }
            }
            trajectories.push(trajectory);
        }
        Some(trajectories)
    }
}

/// Source-named `contour2trajectory` (`contour.cpp:177`).
pub fn contour2trajectory<T, P>(
    contour_x: &Contour,
    contour_y: &Contour,
    frames: &[Frame<T, P>],
) -> Option<Vec<Vec<Point2d>>> {
    Contour::contour_to_trajectories(contour_x, contour_y, frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_constructor_clones_scores_and_counts_nonzero_values() {
        let contour = Contour::from_scores(2, 2, vec![0.0, 1.0e-6, 1.1e-6, -3.0], 1).unwrap();
        assert_eq!((contour.get_num_traj(), contour.get_num_frame()), (2, 2));
        assert_eq!(contour.calculate_nnz(), 1);
        assert_eq!(contour.print(), "0  0.000001  \n0.0000011  -3  \n\n");
        assert!(Contour::from_scores(2, 2, vec![0.0; 3], 1).is_none());
    }

    #[test]
    fn trajectory_constructor_uses_frame_ids_and_selected_coordinate() {
        let trajectories = vec![vec![Point2d::<u32>::new(3, 7, 2), Point2d::new(8, 4, 0)]];
        let contour = Contour::from_trajectories(&trajectories, 2, 3);
        assert_eq!(contour.scores, vec![4.0, 0.0, 7.0]);
    }

    #[test]
    fn discarded_frames_are_compacted_and_round_trip_to_trajectories() {
        let frames = vec![
            Frame::<(), u32>::new((), 0, 1, 1),
            Frame {
                discard: true,
                frame_id: 1,
                ..Frame::default()
            },
            Frame::<(), u32>::new((), 2, 1, 1),
        ];
        let points = vec![vec![Point2d::<u32>::new(2, 3, 0), Point2d::new(5, 7, 2)]];
        let x = Contour::from_trajectories_with_frames(&points, 1, 3, &frames);
        let y = Contour::from_trajectories_with_frames(&points, 2, 3, &frames);
        assert_eq!(x.scores, vec![2.0, 5.0]);
        let output = Contour::contour_to_trajectories(&x, &y, &frames).unwrap();
        assert_eq!(
            output[0]
                .iter()
                .map(|point| (point.x, point.y, point.frame_id))
                .collect::<Vec<_>>(),
            vec![(2.0, 3.0, 0), (5.0, 7.0, 2)]
        );
    }
}
