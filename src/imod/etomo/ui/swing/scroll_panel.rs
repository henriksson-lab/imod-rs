//! `IMOD/Etomo/src/etomo/ui/swing/ScrollPanel.java`.
//!
//! Swing's JPanel/Scrollable native layout is the presentation boundary.  The
//! source implementation itself has only the five Scrollable responses below.
#![allow(dead_code)]

/// Java `ScrollPanel`, package-private and extending `JPanel`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScrollPanel {
    /// `JPanel.getPreferredSize()` supplied by the native layout backend.
    pub preferred_size: (i32, i32),
}
impl ScrollPanel {
    /// Java implicit default constructor `ScrollPanel()`.
    pub fn new() -> Self {
        Self::default()
    }
    /// `getPreferredScrollableViewportSize()`.
    pub fn get_preferred_scrollable_viewport_size(&self) -> (i32, i32) {
        self.preferred_size
    }
    /// `getScrollableUnitIncrement(Rectangle, int, int)`.
    pub fn get_scrollable_unit_increment(
        &self,
        _visible_rect: (i32, i32, i32, i32),
        _orientation: i32,
        _direction: i32,
    ) -> i32 {
        0
    }
    /// `getScrollableBlockIncrement(Rectangle, int, int)`.
    pub fn get_scrollable_block_increment(
        &self,
        _visible_rect: (i32, i32, i32, i32),
        _orientation: i32,
        _direction: i32,
    ) -> i32 {
        0
    }
    /// `getScrollableTracksViewportWidth()`.
    pub fn get_scrollable_tracks_viewport_width(&self) -> bool {
        false
    }
    /// `getScrollableTracksViewportHeight()`.
    pub fn get_scrollable_tracks_viewport_height(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_scrollable_responses_match_java() {
        let p = ScrollPanel {
            preferred_size: (100, 50),
        };
        assert_eq!(p.get_preferred_scrollable_viewport_size(), (100, 50));
        assert_eq!(p.get_scrollable_unit_increment((0, 0, 1, 1), 0, 1), 0);
        assert_eq!(p.get_scrollable_block_increment((0, 0, 1, 1), 0, 1), 0);
        assert!(!p.get_scrollable_tracks_viewport_width());
        assert!(!p.get_scrollable_tracks_viewport_height());
    }
}
