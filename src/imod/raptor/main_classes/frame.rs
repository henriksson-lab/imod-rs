//! Safe translation of `IMOD/raptor/mainClasses/frame.{h,cpp}`.

use super::point2d::Point2d;

/// C++ `frame`, owning an optional image payload and its detected markers.
///
/// `T` replaces the source's non-owning `ioMRC *`.  Callers choose their
/// image representation and move it into the frame, so no raw lifetime or
/// deallocation protocol is needed.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame<T = (), P = ()> {
    pub volume: Option<T>,
    pub points: Vec<Point2d<P>>,
    pub frame_id: i32,
    pub width: i32,
    pub height: i32,
    pub discard: bool,
}

impl<T, P> Default for Frame<T, P> {
    /// C++ default constructor. Rust initializes the C++ source's otherwise
    /// indeterminate scalar fields to neutral values.
    fn default() -> Self {
        Self {
            volume: None,
            points: Vec::new(),
            frame_id: 0,
            width: 0,
            height: 0,
            discard: false,
        }
    }
}

impl<T, P> Frame<T, P> {
    /// C++ `frame(ioMRC *, int, int, int)`.
    pub fn new(volume: T, frame_id: i32, width: i32, height: i32) -> Self {
        Self {
            volume: Some(volume),
            points: Vec::new(),
            frame_id,
            width,
            height,
            discard: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_constructor_has_no_volume_or_points() {
        let frame = Frame::<Vec<u8>>::default();
        assert!(frame.volume.is_none() && frame.points.is_empty());
        assert!(!frame.discard);
    }
    #[test]
    fn full_constructor_owns_volume_and_geometry() {
        let frame = Frame::<Vec<u8>>::new(vec![1, 2], 7, 40, 30);
        assert_eq!(
            (
                frame.volume.unwrap(),
                frame.frame_id,
                frame.width,
                frame.height
            ),
            (vec![1, 2], 7, 40, 30)
        );
        assert!(!frame.discard);
    }
}
