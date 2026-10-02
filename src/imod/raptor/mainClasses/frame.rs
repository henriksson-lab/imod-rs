//! Translation of `IMOD/raptor/mainClasses/frame.h` and `frame.cpp`.
//!
//! The `ioMRC* vol` member is set by the constructor and read nowhere, so it
//! is not kept.

use super::point2d::PointId;

/// `class frame`: one projection of the tilt series.
#[derive(Clone, Debug)]
pub struct Frame {
    /// The markers found in this frame.
    pub p: Vec<PointId>,
    pub frame_id: i32,
    pub width: i32,
    pub height: i32,
    /// If true indicates we should not use this frame.
    pub discard: bool,
}

impl Frame {
    /// `frame(ioMRC* vol, int frameID, int width, int height)` (`frame.cpp:11`).
    pub fn new(frame_id: i32, width: i32, height: i32) -> Frame {
        Frame {
            p: Vec::new(),
            frame_id,
            width,
            height,
            discard: false,
        }
    }
}
