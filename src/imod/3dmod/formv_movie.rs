//! Translation of `IMOD/3dmod/formv_movie.cpp` and `formv_movie.h`.
//! Native form and `mv_movie.cpp` operations are explicit boundaries.
#![allow(dead_code)]
pub const MOVIE_FIELDS: usize = 12;
pub trait ImodvMovieNativeBoundary {
    fn set_delete_on_close(&mut self);
    fn set_always_show_tool_tips(&mut self);
    fn connect_movie_signals(&mut self);
    fn snap_format(&self) -> String;
    fn snap_format2(&self) -> String;
    fn set_non_tif_text(&mut self, which: i32, text: &str);
    fn set_enabled(&mut self, which: i32, value: bool);
    fn set_checked(&mut self, which: i32, value: bool);
    fn set_group(&mut self, which: i32, value: i32);
    fn set_spin(&mut self, which: i32, value: i32);
    fn edit_text(&self, item: usize, start: bool) -> String;
    fn set_edit_text(&mut self, item: usize, start: bool, text: &str);
    fn set_frames(&mut self, value: i32);
    fn frames(&self) -> i32;
    fn montage_frames(&self) -> i32;
    fn movie_full_axis(&mut self, axis: i32);
    fn movie_set_start(&mut self);
    fn movie_set_end(&mut self);
    fn movie_sequence_dialog(&mut self);
    fn movie_make(&mut self, all: bool);
    fn movie_stop(&mut self);
    fn movie_closing(&mut self);
    fn movie_quit(&mut self);
    fn accept_close(&mut self);
    fn close_key(&self) -> bool;
    fn imodv_key_press(&mut self);
    fn imodv_key_release(&mut self);
    fn check_and_set_mac_menu(&mut self);
}
#[derive(Debug)]
pub struct ImodvMovieForm {
    pub m_reverse: bool,
    pub m_str: String,
    pub m_long_way: bool,
    pub m_rgb_tiff: i32,
    pub m_movie_mont: i32,
    pub m_write_files: bool,
    pub m_fps: i32,
}
impl Default for ImodvMovieForm {
    fn default() -> Self {
        Self {
            m_reverse: false,
            m_str: String::new(),
            m_long_way: false,
            m_rgb_tiff: 0,
            m_movie_mont: 0,
            m_write_files: false,
            m_fps: 0,
        }
    }
}
impl ImodvMovieForm {
    pub fn new(n: &mut dyn ImodvMovieNativeBoundary) -> Self {
        let mut f = Self::default();
        f.init(n);
        f
    }
    pub fn destroy(&mut self) {}
    pub fn language_change(&mut self) {}
    pub fn init(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.set_delete_on_close();
        n.set_always_show_tool_tips();
        n.connect_movie_signals();
        self.set_non_tif_label(n)
    }
    pub fn set_non_tif_label(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.set_non_tif_text(0, &format!("{}s", n.snap_format()));
        let second = n.snap_format2();
        n.set_enabled(0, !second.is_empty());
        if second.is_empty() && self.m_rgb_tiff > 1 {
            n.set_group(1, 1);
            self.m_rgb_tiff = 1
        } else if !second.is_empty() {
            n.set_non_tif_text(1, &format!("{second}s"))
        }
    }
    pub fn full_x_pressed(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.movie_full_axis(1)
    }
    pub fn full_y_pressed(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.movie_full_axis(2)
    }
    pub fn set_start_pressed(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.movie_set_start()
    }
    pub fn set_end_pressed(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.movie_set_end()
    }
    pub fn reverse_toggled(&mut self, x: bool) {
        self.m_reverse = x
    }
    pub fn long_way_toggled(&mut self, x: bool) {
        self.m_long_way = x
    }
    pub fn movie_mont_selected(&mut self, x: i32) {
        self.m_movie_mont = x
    }
    pub fn rgb_tiff_selected(&mut self, x: i32) {
        self.m_rgb_tiff = x
    }
    pub fn write_toggled(&mut self, x: bool, n: &mut dyn ImodvMovieNativeBoundary) {
        self.m_write_files = x;
        n.set_enabled(1, !x);
        n.set_enabled(2, !x)
    }
    pub fn fps_changed(&mut self, x: i32) {
        self.m_fps = x
    }
    pub fn sequence_clicked(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.movie_sequence_dialog()
    }
    pub fn make_pressed(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.movie_make(false)
    }
    pub fn stop_pressed(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.movie_stop()
    }
    pub fn read_start_end(
        &mut self,
        item: usize,
        n: &mut dyn ImodvMovieNativeBoundary,
    ) -> (f32, f32) {
        (
            n.edit_text(item, true).parse().unwrap_or(0.),
            n.edit_text(item, false).parse().unwrap_or(0.),
        )
    }
    pub fn set_start(&mut self, item: usize, mut v: f32, n: &mut dyn ImodvMovieNativeBoundary) {
        if item < 3 && v.abs() < 0.01 {
            v = 0.
        }
        n.set_edit_text(item, true, &format!("{v:.4}"))
    }
    pub fn set_end(&mut self, item: usize, mut v: f32, n: &mut dyn ImodvMovieNativeBoundary) {
        if item < 3 && v.abs() < 0.01 {
            v = 0.
        }
        n.set_edit_text(item, false, &format!("{v:.4}"))
    }
    pub fn set_button_states(
        &mut self,
        long: bool,
        reverse: bool,
        mont: i32,
        rgb: i32,
        write: bool,
        fps: i32,
        n: &mut dyn ImodvMovieNativeBoundary,
    ) {
        n.set_checked(0, long);
        n.set_checked(1, reverse);
        n.set_checked(2, write);
        n.set_group(0, mont);
        n.set_spin(0, fps);
        n.set_enabled(1, !write);
        n.set_enabled(2, !write);
        self.m_rgb_tiff = if n.snap_format2().is_empty() && rgb > 1 {
            1
        } else {
            rgb
        };
        n.set_group(1, self.m_rgb_tiff);
        self.manage_sensitivities(mont, n);
        self.m_long_way = long;
        self.m_reverse = reverse;
        self.m_movie_mont = mont;
        self.m_write_files = write;
        self.m_fps = fps
    }
    pub fn get_button_states(&self) -> (i32, i32, i32, i32, i32, i32) {
        (
            self.m_long_way as i32,
            self.m_reverse as i32,
            self.m_movie_mont,
            self.m_rgb_tiff,
            self.m_write_files as i32,
            self.m_fps,
        )
    }
    pub fn get_frame_boxes(&mut self, n: &mut dyn ImodvMovieNativeBoundary) -> (i32, i32) {
        (n.frames(), n.montage_frames())
    }
    pub fn set_frame_boxes(&mut self, a: i32, b: i32, n: &mut dyn ImodvMovieNativeBoundary) {
        n.set_frames(a);
        n.set_spin(3, b)
    }
    pub fn sequence_open(&mut self, state: bool, n: &mut dyn ImodvMovieNativeBoundary) {
        n.set_enabled(20, !state)
    }
    pub fn manage_sensitivities(&mut self, mont: i32, n: &mut dyn ImodvMovieNativeBoundary) {
        let e = mont == 0;
        for i in 0..MOVIE_FIELDS {
            n.set_enabled(100 + i as i32, e)
        }
        n.set_enabled(3, e);
        n.set_enabled(4, e);
        n.set_enabled(5, e);
        n.set_enabled(6, e);
        n.set_enabled(7, e);
        n.set_enabled(8, e);
        n.set_enabled(9, mont != 0);
        n.set_enabled(10, e)
    }
    pub fn top_close_event(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.movie_closing();
        n.accept_close()
    }
    pub fn key_press_event(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        if n.close_key() {
            n.movie_quit()
        } else {
            n.imodv_key_press()
        }
    }
    pub fn key_release_event(&mut self, n: &mut dyn ImodvMovieNativeBoundary) {
        n.imodv_key_release()
    }
    pub fn top_change_event(&mut self, _font: bool, n: &mut dyn ImodvMovieNativeBoundary) {
        n.check_and_set_mac_menu()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        en: Vec<(i32, bool)>,
        start: [String; 12],
        end: [String; 12],
    }
    impl ImodvMovieNativeBoundary for N {
        fn set_delete_on_close(&mut self) {}
        fn set_always_show_tool_tips(&mut self) {}
        fn connect_movie_signals(&mut self) {}
        fn snap_format(&self) -> String {
            "JPEG".into()
        }
        fn snap_format2(&self) -> String {
            String::new()
        }
        fn set_non_tif_text(&mut self, _: i32, _: &str) {}
        fn set_enabled(&mut self, a: i32, b: bool) {
            self.en.push((a, b))
        }
        fn set_checked(&mut self, _: i32, _: bool) {}
        fn set_group(&mut self, _: i32, _: i32) {}
        fn set_spin(&mut self, _: i32, _: i32) {}
        fn edit_text(&self, a: usize, b: bool) -> String {
            if b {
                self.start[a].clone()
            } else {
                self.end[a].clone()
            }
        }
        fn set_edit_text(&mut self, a: usize, b: bool, x: &str) {
            if b {
                self.start[a] = x.into()
            } else {
                self.end[a] = x.into()
            }
        }
        fn set_frames(&mut self, _: i32) {}
        fn frames(&self) -> i32 {
            0
        }
        fn montage_frames(&self) -> i32 {
            0
        }
        fn movie_full_axis(&mut self, _: i32) {}
        fn movie_set_start(&mut self) {}
        fn movie_set_end(&mut self) {}
        fn movie_sequence_dialog(&mut self) {}
        fn movie_make(&mut self, _: bool) {}
        fn movie_stop(&mut self) {}
        fn movie_closing(&mut self) {}
        fn movie_quit(&mut self) {}
        fn accept_close(&mut self) {}
        fn close_key(&self) -> bool {
            false
        }
        fn imodv_key_press(&mut self) {}
        fn imodv_key_release(&mut self) {}
        fn check_and_set_mac_menu(&mut self) {}
    }
    #[test]
    fn tiny_rotation_is_zeroed_and_montage_disables_edits() {
        let mut n = N::default();
        let mut f = ImodvMovieForm::new(&mut n);
        f.set_start(0, 0.001, &mut n);
        assert_eq!(n.start[0], "0.0000");
        f.manage_sensitivities(1, &mut n);
        assert!(n.en.contains(&(100, false)));
    }
}
