//! Translation of `IMOD/raptor/optimization/contour.h` and `contour.cpp`.

use crate::imod::cxx_stream::cout;
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::raptor::main_classes::frame::Frame;
use crate::imod::raptor::main_classes::point2d::Point2D;
use crate::imod::raptor::opencv::cxtypes::{CV_64FC1, CvMat};
use crate::imod::raptor::trajectory::trajectory::Trajectory;

/// `class contour`: one coordinate (x or y) of every trajectory in every
/// frame, trajectories in rows.  `scores` is never NULL on a reached path
/// (the default constructor that leaves it NULL is not reached).
#[derive(Clone, Debug)]
pub struct Contour {
    /// Contains the values: for example contour_x contains x coordinates.
    pub scores: CvMat,
    pub xory: i32,
    pub num_traj: i32,
    pub num_frame: i32,
}

/// `contour::minContourValue` (`contour.cpp:12`).
pub const MIN_CONTOUR_VALUE: f64 = 1e-6;

impl Contour {
    /// `contour(int rows, int cols, int _xory)` (`contour.cpp:18`):
    /// initialize contour to zero.
    pub fn zeros(rows: i32, cols: i32, xory: i32) -> Contour {
        let mut scores = CvMat::create(rows, cols, CV_64FC1);
        scores.set_zero();
        Contour {
            scores,
            xory,
            num_traj: rows,
            num_frame: cols,
        }
    }

    /// `contour(CvMat* cc, int _xory)` (`contour.cpp:27`).
    pub fn from_mat(cc: &CvMat, xory: i32) -> Contour {
        Contour {
            xory,
            num_traj: cc.rows,
            num_frame: cc.cols,
            scores: cc.clone_mat(),
        }
    }

    /// `contour(vector<trajectory> traj, int coordinate, int numFrames,
    /// vector<frame>* frames)` (`contour.cpp:57`).  `points` is the arena the
    /// trajectories index.
    pub fn from_trajectories(
        traj: &[Trajectory],
        coordinate: i32,
        num_frames: i32,
        frames: &[Frame],
        points: &[Point2D],
    ) -> Contour {
        // create a map between frameId and real position in the contour
        let mut pos: Vec<i32> = vec![0; num_frames as usize];
        let mut count = 0;
        for ii in 0..num_frames as usize {
            if frames[ii].discard {
                pos[ii] = -1;
            } else {
                pos[ii] = count;
                count += 1;
            }
        }
        let num_traj = traj.len() as i32;
        let num_frame = count;
        let mut scores = CvMat::create(num_traj, num_frame, CV_64FC1);
        scores.set_zero();
        for i in 0..traj.len() {
            for j in 0..traj[i].p.len() {
                let point = &points[traj[i].p[j]];
                let pos_aux = point.frame_id;
                if pos[pos_aux as usize] >= 0 {
                    if coordinate == 1 {
                        scores.set_real_2d(i as i32, pos[pos_aux as usize], point.x as f64);
                    } else if coordinate == 2 {
                        scores.set_real_2d(i as i32, pos[pos_aux as usize], point.y as f64);
                    }
                }
            }
        }
        Contour {
            scores,
            xory: coordinate,
            num_traj,
            num_frame,
        }
    }

    /// `contour::calculateNNZ()` (`contour.h:41`).
    pub fn calculate_nnz(&self) -> i32 {
        let mut nnz = 0;
        let data = &self.scores.data;
        for kk in 0..(self.num_traj * self.num_frame) as usize {
            if data[kk] > MIN_CONTOUR_VALUE {
                nnz += 1;
            }
        }
        nnz
    }

    /// `contour::getNumTraj()`.
    pub fn get_num_traj(&self) -> i32 {
        self.num_traj
    }

    /// `contour::getNumFrame()`.
    pub fn get_num_frame(&self) -> i32 {
        self.num_frame
    }

    /// `contour::contour2trajectory(contour c_x, contour c_y, vector<frame>*
    /// frames)` (`contour.cpp:162`).  The new points are appended to
    /// `points`.  `this` is not read; the C++ passes the contours by value,
    /// which only copies them.
    pub fn contour2trajectory(
        &self,
        c_x: &Contour,
        c_y: &Contour,
        frames: &[Frame],
        points: &mut Vec<Point2D>,
    ) -> Vec<Trajectory> {
        // create a map from contours to frameID
        let mut map: Vec<i32> = vec![0; c_x.scores.cols as usize];
        let mut count = 0i32;
        for kk in 0..frames.len() {
            if !frames[kk].discard {
                map[count as usize] = kk as i32;
                count += 1;
            }
        }
        if count != c_x.scores.cols {
            cout(&format!(
                "ERROR: at contour2trajectory;Number of cols={};Number of valid frames={}.They should be the same\n",
                c_x.scores.cols, count
            ));
            exit(-1);
        }
        let mut t: Vec<Trajectory> = Vec::new();
        for i in 0..c_x.scores.rows {
            let mut new_traj = Trajectory::default();
            let mut pos = (i * c_x.scores.cols) as usize;
            for j in 0..c_x.scores.cols as usize {
                if c_x.scores.data[pos] > 1e-6 {
                    let mut new_point = Point2D::new();
                    new_point.x = c_x.scores.data[pos] as f32;
                    new_point.y = c_y.scores.data[pos] as f32;
                    new_point.frame_id = map[j];
                    points.push(new_point);
                    new_traj.p.push(points.len() - 1);
                }
                pos += 1;
            }
            t.push(new_traj);
        }
        t
    }
}
