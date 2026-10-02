//! Translation of `IMOD/raptor/mainClasses/paircorrespondence.h`.

/// `class pairCorrespondence`: a correspondence between a point of one frame
/// and a point of another.  The C++ `frame*` members are indices into the
/// program's `vector<frame>` (a frame's index is its `frameID`).
#[derive(Clone, Copy, Debug)]
pub struct PairCorrespondence {
    pub point1_index: i32,
    pub point2_index: i32,
    pub frame1: usize,
    pub frame2: usize,
    /// The score of correspondence between a pair of points.
    pub score: f64,
}

impl PairCorrespondence {
    /// `pairCorrespondence(point1_index, point2_index, frame1, frame2,
    /// score)` (`paircorrespondence.h:12`).
    pub fn new(
        point1_index: i32,
        point2_index: i32,
        frame1: usize,
        frame2: usize,
        score: f64,
    ) -> PairCorrespondence {
        PairCorrespondence {
            point1_index,
            point2_index,
            frame1,
            frame2,
            score,
        }
    }
}
