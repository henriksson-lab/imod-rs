//! Translation of `IMOD/raptor/mainClasses/point2d.h` and `point2d.cpp`.
//!
//! The C++ program shares `Point2D*` between the frames that detect the
//! points, the correspondences and the trajectories built from them, and
//! mutates them through every alias (`used`, `pairwise_score`, the `index0`
//! and `index1` lists).  Here the points live in one arena, a
//! `Vec<Point2D>` owned by `main`, and every C++ `Point2D*` is a
//! [`PointId`] into it.  `delete` of a point is therefore a no-op; the arena
//! is freed when `main` ends.  The copy constructor and `operator=` are
//! `Clone`, the destructor (which only clears the two lists) is `Drop`.
//!
//! The members the C++ constructors leave uninitialised (`x`, `y`,
//! `markerType`, `frameID` and `score` in the default constructor;
//! `markerType` and `score` in the three-argument one) start at zero here.
//! No reached path reads one of them before writing it.

use super::constants::INITIAL_PAIRWISE_SCORE;
use super::paircorrespondence::PairCorrespondence;

/// A C++ `Point2D*`: an index into the program's point arena.
pub type PointId = usize;

/// `class Point2D`.
#[derive(Clone, Debug)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
    /// Needed to identify to which marker size belongs to when there are
    /// different markers sizes in the same sample.
    pub marker_type: i32,
    pub frame_id: i32,
    pub score: f64,
    pub pairwise_score: f64,
    pub used: bool,
    pub used_pair_correspondence: bool,
    pub index0: Vec<PairCorrespondence>,
    pub index1: Vec<PairCorrespondence>,
}

impl Point2D {
    /// `Point2D()` (`point2d.h:24`).
    pub fn new() -> Point2D {
        Point2D {
            x: 0.0,
            y: 0.0,
            marker_type: 0,
            frame_id: 0,
            score: 0.0,
            pairwise_score: INITIAL_PAIRWISE_SCORE,
            used: false,
            used_pair_correspondence: false,
            index0: Vec::new(),
            index1: Vec::new(),
        }
    }

    /// `Point2D(int a, int b, int ID)` (`point2d.h:33`).
    pub fn with_position(a: i32, b: i32, id: i32) -> Point2D {
        Point2D {
            x: a as f32,
            y: b as f32,
            marker_type: 0,
            frame_id: id,
            score: 0.0,
            pairwise_score: -1E+12,
            used: false,
            used_pair_correspondence: false,
            index0: Vec::new(),
            index1: Vec::new(),
        }
    }
}
