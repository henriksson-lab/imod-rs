//! Translation of `IMOD/3dmod/formv_sequence.cpp` and `formv_sequence.h`.
//! Autodoc and Qt table/file-dialog calls remain explicit native boundaries.
#![allow(dead_code)]
pub const MAX_OBJ_ONOFF: usize = 5000;
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MovieSegment {
    pub num_frames: i32,
    pub view_num: i32,
    pub label: String,
}
pub trait MovieSequenceNativeBoundary {
    fn set_delete_on_close(&mut self);
    fn set_always_show_tool_tips(&mut self);
    fn connect_sequence_signals(&mut self);
    fn current_row(&self) -> i32;
    fn select_row(&mut self, row: i32);
    fn set_rows(&mut self, segments: &[MovieSegment]);
    fn set_enabled(&mut self, which: i32, value: bool);
    fn movie_get_segment(&self) -> MovieSegment;
    fn movie_set_segment(&mut self, segment: &MovieSegment);
    fn movie_set_terminus(&mut self, which: i32, segment: &MovieSegment);
    fn movie_make(&mut self, all: bool) -> bool;
    fn save_sequence(&mut self, segments: &[MovieSegment]) -> i32;
    fn load_sequence(&mut self) -> Option<Vec<MovieSegment>>;
    fn choice_save_before_load(&mut self) -> i32;
    fn choice_save_before_close(&mut self) -> i32;
    fn accept_close(&mut self);
    fn ignore_close(&mut self);
    fn movie_sequence_closing(&mut self);
    fn close(&mut self);
    fn close_key(&self) -> bool;
    fn imodv_key_press(&mut self);
    fn imodv_key_release(&mut self);
    fn check_and_set_mac_menu(&mut self);
}
pub const ADD_AFTER: i32 = 0;
pub const ADD_BEFORE: i32 = 1;
pub const REPLACE: i32 = 2;
pub const DELETE: i32 = 3;
pub const SET_MOVIE: i32 = 4;
pub const SET_START: i32 = 5;
pub const SET_END: i32 = 6;
pub const RUN_ALL: i32 = 7;
pub const SAVE: i32 = 8;
pub const IMODV_MOVIE_START_STATE: i32 = 0;
pub const IMODV_MOVIE_END_STATE: i32 = 1;
#[derive(Debug, Default)]
pub struct MovieSequenceForm {
    pub m_segments: Vec<MovieSegment>,
    pub m_making_movie: bool,
    pub m_movie_enabled: bool,
    pub m_modified: bool,
    pub max_view: i32,
}
impl MovieSequenceForm {
    pub fn new(
        segments: Vec<MovieSegment>,
        max_view: i32,
        n: &mut dyn MovieSequenceNativeBoundary,
    ) -> Self {
        let mut f = Self {
            m_segments: segments,
            max_view,
            ..Default::default()
        };
        f.init(n);
        f
    }
    pub fn destroy(&mut self) {}
    pub fn language_change(&mut self) {}
    pub fn top_change_event(&mut self, font: bool, n: &mut dyn MovieSequenceNativeBoundary) {
        n.check_and_set_mac_menu();
        if font {
            self.set_font_dependent_widths(n)
        }
    }
    pub fn init(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        n.set_delete_on_close();
        n.set_always_show_tool_tips();
        n.connect_sequence_signals();
        self.set_font_dependent_widths(n);
        self.m_modified = false
    }
    pub fn set_for_four_rows(&mut self) {}
    pub fn update_enables(
        &mut self,
        movie_enabled: bool,
        making_movie: bool,
        n: &mut dyn MovieSequenceNativeBoundary,
    ) {
        let size = self.m_segments.len();
        let selected = size > 0 && n.current_row() >= 0;
        n.set_enabled(
            ADD_AFTER,
            movie_enabled && (!size.ne(&0) || n.current_row() >= 0),
        );
        n.set_enabled(ADD_BEFORE, movie_enabled && selected);
        n.set_enabled(REPLACE, movie_enabled && selected);
        n.set_enabled(DELETE, selected);
        n.set_enabled(SET_MOVIE, movie_enabled && !making_movie && selected);
        n.set_enabled(SET_START, !making_movie && selected);
        n.set_enabled(SET_END, !making_movie && selected);
        n.set_enabled(RUN_ALL, movie_enabled && !making_movie);
        n.set_enabled(SAVE, size > 0);
        self.m_making_movie = making_movie;
        self.m_movie_enabled = movie_enabled
    }
    pub fn add_at_index(&mut self, index: i32, n: &mut dyn MovieSequenceNativeBoundary) {
        let segment = n.movie_get_segment();
        let index = index.max(0) as usize;
        if index >= self.m_segments.len() {
            self.m_segments.push(segment)
        } else {
            self.m_segments.insert(index, segment)
        }
        self.load_table(n);
        n.select_row(index as i32);
        self.update_enables(self.m_movie_enabled, self.m_making_movie, n);
        self.m_modified = true
    }
    pub fn add_after_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        let index = n.current_row() + 1;
        if index == 0 && !self.m_segments.is_empty() {
            return;
        }
        self.add_at_index(index, n)
    }
    pub fn add_before_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        let index = n.current_row();
        if index < 0 || self.m_segments.is_empty() {
            return;
        }
        self.add_at_index(index, n)
    }
    pub fn replace_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        let index = n.current_row();
        if index < 0 || index as usize >= self.m_segments.len() {
            return;
        }
        let mut segment = n.movie_get_segment();
        if !self.m_segments[index as usize].label.is_empty() {
            segment.label = self.m_segments[index as usize].label.clone()
        }
        self.m_segments[index as usize] = segment;
        self.load_row(index, n);
        self.m_modified = true
    }
    pub fn delete_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        let mut index = n.current_row();
        if index < 0 || index as usize >= self.m_segments.len() {
            return;
        }
        self.m_segments.remove(index as usize);
        self.load_table(n);
        if index as usize >= self.m_segments.len() {
            index = self.m_segments.len() as i32 - 1
        }
        if index >= 0 {
            n.select_row(index)
        }
        self.update_enables(self.m_movie_enabled, self.m_making_movie, n);
        self.m_modified = true
    }
    pub fn set_movie_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        if let Some(s) = self.m_segments.get(n.current_row().max(0) as usize) {
            n.movie_set_segment(s)
        }
    }
    pub fn set_start_or_end(&mut self, which: i32, n: &mut dyn MovieSequenceNativeBoundary) {
        if let Some(s) = self.m_segments.get(n.current_row().max(0) as usize) {
            n.movie_set_terminus(which, s)
        }
    }
    pub fn set_start_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        self.set_start_or_end(IMODV_MOVIE_START_STATE, n)
    }
    pub fn set_end_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        self.set_start_or_end(IMODV_MOVIE_END_STATE, n)
    }
    pub fn run_all_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        for s in &self.m_segments {
            n.movie_set_segment(s);
            if n.movie_make(true) {
                break;
            }
        }
    }
    pub fn save_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        self.save_sequence(n);
    }
    pub fn save_sequence(&mut self, n: &mut dyn MovieSequenceNativeBoundary) -> i32 {
        let err = n.save_sequence(&self.m_segments);
        self.m_modified = err < 0;
        err
    }
    pub fn load_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        if self.m_modified {
            let c = n.choice_save_before_load();
            if c == 3 || (c == 1 && self.save_sequence(n) < 0) {
                return;
            }
        }
        let Some(s) = n.load_sequence() else { return };
        self.m_segments = s;
        self.load_table(n);
        n.select_row(0);
        self.update_enables(self.m_movie_enabled, self.m_making_movie, n);
        self.m_modified = false
    }
    pub fn load_table(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        n.set_rows(&self.m_segments)
    }
    pub fn load_row(&mut self, _: i32, n: &mut dyn MovieSequenceNativeBoundary) {
        self.load_table(n)
    }
    pub fn entry_changed(
        &mut self,
        row: i32,
        column: i32,
        text: &str,
        n: &mut dyn MovieSequenceNativeBoundary,
    ) {
        let Some(s) = self.m_segments.get_mut(row.max(0) as usize) else {
            return;
        };
        if column > 1 {
            s.label = text.trim().into()
        } else if column == 0 {
            s.num_frames = text.trim().parse().unwrap_or(0).max(2)
        } else {
            s.view_num = text.trim().parse().unwrap_or(0).clamp(1, self.max_view - 1)
        }
        self.load_row(row, n);
        self.m_modified = true
    }
    pub fn set_font_dependent_widths(&mut self, _: &mut dyn MovieSequenceNativeBoundary) {}
    pub fn top_close_event(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        if self.m_modified {
            let c = n.choice_save_before_close();
            if c == 3 || (c == 1 && self.save_sequence(n) < 0) {
                n.ignore_close();
                return;
            }
        }
        n.movie_sequence_closing();
        self.m_segments.clear();
        n.accept_close()
    }
    pub fn key_press_event(&mut self, navigation: bool, n: &mut dyn MovieSequenceNativeBoundary) {
        if navigation {
            return;
        }
        if n.close_key() {
            n.close()
        } else {
            n.imodv_key_press()
        }
    }
    pub fn key_release_event(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        n.imodv_key_release()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        row: i32,
        rows: Vec<MovieSegment>,
        current: MovieSegment,
    }
    impl MovieSequenceNativeBoundary for N {
        fn set_delete_on_close(&mut self) {}
        fn set_always_show_tool_tips(&mut self) {}
        fn connect_sequence_signals(&mut self) {}
        fn current_row(&self) -> i32 {
            self.row
        }
        fn select_row(&mut self, x: i32) {
            self.row = x
        }
        fn set_rows(&mut self, x: &[MovieSegment]) {
            self.rows = x.into()
        }
        fn set_enabled(&mut self, _: i32, _: bool) {}
        fn movie_get_segment(&self) -> MovieSegment {
            self.current.clone()
        }
        fn movie_set_segment(&mut self, _: &MovieSegment) {}
        fn movie_set_terminus(&mut self, _: i32, _: &MovieSegment) {}
        fn movie_make(&mut self, _: bool) -> bool {
            false
        }
        fn save_sequence(&mut self, _: &[MovieSegment]) -> i32 {
            0
        }
        fn load_sequence(&mut self) -> Option<Vec<MovieSegment>> {
            None
        }
        fn choice_save_before_load(&mut self) -> i32 {
            2
        }
        fn choice_save_before_close(&mut self) -> i32 {
            2
        }
        fn accept_close(&mut self) {}
        fn ignore_close(&mut self) {}
        fn movie_sequence_closing(&mut self) {}
        fn close(&mut self) {}
        fn close_key(&self) -> bool {
            false
        }
        fn imodv_key_press(&mut self) {}
        fn imodv_key_release(&mut self) {}
        fn check_and_set_mac_menu(&mut self) {}
    }
    #[test]
    fn add_replace_delete_and_entry_limits_match_source() {
        let mut n = N {
            row: -1,
            current: MovieSegment {
                num_frames: 9,
                view_num: 2,
                label: "x".into(),
            },
            ..Default::default()
        };
        let mut f = MovieSequenceForm::new(vec![], 4, &mut n);
        f.add_after_clicked(&mut n);
        f.entry_changed(0, 0, "1", &mut n);
        f.entry_changed(0, 1, "9", &mut n);
        assert_eq!(
            (f.m_segments[0].num_frames, f.m_segments[0].view_num),
            (2, 3)
        );
        f.delete_clicked(&mut n);
        assert!(f.m_segments.is_empty());
    }
}
