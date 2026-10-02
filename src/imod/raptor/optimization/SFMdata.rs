//! Translation of `IMOD/raptor/optimization/SFMdata.h` and `SFMdata.cpp`.

use super::contour::Contour;

/// `class SFMdata`.  The four contours are never NULL on a reached path (the
/// default constructor that leaves them NULL is not reached).  The copy
/// constructor is `Clone`.
#[derive(Clone, Debug)]
pub struct SfmData {
    pub reproj_x: Contour,
    pub reproj_y: Contour,
    pub contour_x: Contour,
    pub contour_y: Contour,
    /// When we have different markers with different sizes we need to know
    /// each trajectory.
    pub marker_type: Vec<i32>,
}

impl SfmData {
    /// `SFMdata(contour c_x, contour c_y, int* mType)` (`SFMdata.cpp:22`).
    pub fn new(c_x: &Contour, c_y: &Contour, m_type: &[i32]) -> SfmData {
        SfmData {
            contour_x: c_x.clone(),
            contour_y: c_y.clone(),
            reproj_x: Contour::zeros(c_x.num_traj, c_x.num_frame, 1), // initialize to zero
            reproj_y: Contour::zeros(c_y.num_traj, c_y.num_frame, 2), // initialize to zero
            marker_type: m_type[..c_x.num_traj as usize].to_vec(),
        }
    }

    /// `SFMdata(contour c_x, contour c_y, contour r_x, contour r_y, int*
    /// mType)` (`SFMdata.cpp:32`).
    pub fn with_reprojection(
        c_x: &Contour,
        c_y: &Contour,
        r_x: &Contour,
        r_y: &Contour,
        m_type: &[i32],
    ) -> SfmData {
        SfmData {
            contour_x: c_x.clone(),
            contour_y: c_y.clone(),
            reproj_x: r_x.clone(),
            reproj_y: r_y.clone(),
            marker_type: m_type[..c_x.num_traj as usize].to_vec(),
        }
    }
}
