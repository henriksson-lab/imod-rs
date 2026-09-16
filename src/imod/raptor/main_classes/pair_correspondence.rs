//! Translation of `IMOD/raptor/mainClasses/paircorrespondence.h`.

/// C++ `pairCorrespondence`: correspondence between two point indexes.
///
/// The C++ `frame *` members had no ownership/lifetime contract.  The Rust
/// form owns optional caller-selected frame references (`F`), avoiding raw
/// cursors while retaining both source fields.
#[derive(Clone, Debug, PartialEq)]
pub struct PairCorrespondence<F = ()> {
    pub point1_index: i32,
    pub point2_index: i32,
    pub frame1: Option<F>,
    pub frame2: Option<F>,
    pub score: f64,
}

impl<F> PairCorrespondence<F> {
    /// C++ `pairCorrespondence(int, int, frame *, frame *, double)`.
    pub fn new(point1_index: i32, point2_index: i32, frame1: F, frame2: F, score: f64) -> Self {
        Self {
            point1_index,
            point2_index,
            frame1: Some(frame1),
            frame2: Some(frame2),
            score,
        }
    }
}

impl<F> Default for PairCorrespondence<F> {
    fn default() -> Self {
        Self {
            point1_index: 0,
            point2_index: 0,
            frame1: None,
            frame2: None,
            score: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_owns_both_frame_references_and_score() {
        let pair = PairCorrespondence::new(2, 5, 7_i32, 8_i32, 1.25);
        assert_eq!(
            (
                pair.point1_index,
                pair.point2_index,
                pair.frame1,
                pair.frame2,
                pair.score
            ),
            (2, 5, Some(7), Some(8), 1.25)
        );
    }
    #[test]
    fn default_has_no_frame_links() {
        assert_eq!(PairCorrespondence::<i32>::default().frame1, None);
    }
}
