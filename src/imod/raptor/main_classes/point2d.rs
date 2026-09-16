//! Safe translation of `IMOD/raptor/mainClasses/point2d.{h,cpp}`.

use super::constants::INITIAL_PAIRWISE_SCORE;
use super::pair_correspondence::PairCorrespondence;
use std::fmt;
use std::str::FromStr;

/// The coordinate-taking C++ constructors explicitly select this score rather
/// than `initialPairwiseScore`, which only the default constructor uses.
const CONSTRUCTED_PAIRWISE_SCORE: f64 = -1.0e12;

/// C++ `Point2D`; the correspondence payload stays generic because its owning
/// `pairCorrespondence`/`frame` units are separate source translations.
#[derive(Clone, Debug, PartialEq)]
pub struct Point2d<T = PairCorrespondence> {
    pub x: f32,
    pub y: f32,
    pub marker_type: i32,
    pub frame_id: i32,
    pub score: f64,
    pub pairwise_score: f64,
    pub used: bool,
    pub used_pair_correspondence: bool,
    pub index0: Vec<T>,
    pub index1: Vec<T>,
}

impl<T> Default for Point2d<T> {
    fn default() -> Self {
        Self {
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
}

impl<T> Point2d<T> {
    /// C++ `Point2D(int, int, int)`.
    pub fn new(x: i32, y: i32, frame_id: i32) -> Self {
        Self {
            x: x as f32,
            y: y as f32,
            frame_id,
            pairwise_score: CONSTRUCTED_PAIRWISE_SCORE,
            ..Self::default()
        }
    }
    /// C++ `Point2D(int, int, int, int)`.
    pub fn with_marker_type(x: i32, y: i32, frame_id: i32, marker_type: i32) -> Self {
        Self {
            marker_type,
            ..Self::new(x, y, frame_id)
        }
    }
}

impl<T> fmt::Display for Point2d<T> {
    /// C++ `operator<<`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "({}, {})", self.x, self.y)
    }
}

impl<T> FromStr for Point2d<T> {
    type Err = std::num::ParseFloatError;
    /// C++ `operator>>`: reads its two whitespace-delimited coordinates.
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let mut fields = input.split_whitespace();
        let x = fields.next().unwrap_or("0").parse()?;
        let y = fields.next().unwrap_or("0").parse()?;
        Ok(Self {
            x,
            y,
            ..Self::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructors_preserve_source_coordinate_and_marker_values() {
        let point = Point2d::<u32>::with_marker_type(4, -3, 9, 2);
        assert_eq!(
            (point.x, point.y, point.frame_id, point.marker_type),
            (4., -3., 9, 2)
        );
        assert!(!point.used && !point.used_pair_correspondence);
        assert_eq!(point.pairwise_score, CONSTRUCTED_PAIRWISE_SCORE);
    }
    #[test]
    fn stream_equivalents_parse_and_format_coordinates() {
        let point: Point2d = "1.5 -2".parse().unwrap();
        assert_eq!(point.to_string(), "(1.5, -2)");
    }
}
