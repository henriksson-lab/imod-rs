//! Translation of `IMOD/3dmod/form_moviecon.cpp` and `form_moviecon.h`.
//! `moviecon.cpp` is an explicit direct native-operation boundary; form state
//! mirrors Qt widgets and never substitutes movie rendering.
#![allow(dead_code)]
/// Every `imc*` call made by the source form.
pub trait MovieConOperations {
    fn axis_selected(&mut self, v: i32);
    fn slider_changed(&mut self, w: i32, v: i32);
    fn increment_rate(&mut self, v: i32);
    fn rate_entered(&mut self, v: f32);
    fn snap_selected(&mut self, v: i32);
    fn extent_selected(&mut self, v: i32);
    fn start_here_selected(&mut self, v: i32);
    fn set_snap_montage(&mut self, v: bool);
    fn set_montage_factor(&mut self, v: i32);
    fn set_snap_whole_mont(&mut self, v: i32);
    fn set_scale_sizes(&mut self, v: bool);
    fn set_size_scaling(&mut self, v: i32);
    fn set_slicer_montage(&mut self, v: bool);
    fn set_slicer_mont_factor(&mut self, v: i32);
    fn set_scale_thicks(&mut self, v: bool);
    fn set_thick_scaling(&mut self, v: i32);
    fn reset_pressed(&mut self);
    fn closing(&mut self);
    fn control_key(&mut self, release: bool, key: i32);
}
/// Original `MovieController` form.
#[derive(Clone, Debug, Default)]
pub struct MovieController {
    pub top_window_open: bool,
    pub delete_on_close: bool,
    pub always_show_tooltips: bool,
    pub axis: i32,
    pub snapshot: i32,
    pub extent: i32,
    pub start_here: i32,
    pub sliders: [i32; 3],
    pub slider_ranges: [(i32, i32); 2],
    pub sliders_enabled: bool,
    pub rate_box: String,
    pub actual_rate: String,
    pub time_enabled: bool,
    pub montage: bool,
    pub montage_factor: i32,
    pub montage_spin_enabled: bool,
    pub whole_image: i32,
    pub scale_sizes: bool,
    pub size_scaling: i32,
    pub slicer_montage: bool,
    pub slicer_montage_factor: i32,
    pub scale_thicks: bool,
    pub thick_scaling: i32,
    pub rgb_label: String,
    pub png_enabled: bool,
}
pub fn movie_controller_new() -> MovieController {
    let mut c = MovieController {
        top_window_open: true,
        ..Default::default()
    };
    c.init();
    c
}
impl MovieController {
    pub fn destroy(&mut self) {
        self.top_window_open = false
    }
    pub fn language_change(&mut self) {}
    pub fn init(&mut self) {
        self.delete_on_close = true;
        self.always_show_tooltips = true;
        self.snapshot = 0;
        self.extent = 0;
        self.start_here = 0;
        self.axis = 2;
        self.set_font_dependent_widths();
    }
    pub fn set_font_dependent_widths(&mut self) {}
    pub fn set_non_tif_label(
        &mut self,
        first: &str,
        second: Option<&str>,
        ops: &mut dyn MovieConOperations,
    ) {
        self.rgb_label = first.into();
        self.png_enabled = second.is_some_and(|x| !x.is_empty());
        if !self.png_enabled && self.snapshot > 2 {
            self.snapshot = 2;
            ops.snap_selected(2);
        }
    }
    pub fn axis_selected(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.axis = v;
        ops.axis_selected(v)
    }
    pub fn slider_changed(&mut self, ops: &mut dyn MovieConOperations, w: i32, v: i32, drag: bool) {
        if let Some(x) = self.sliders.get_mut(w.max(0) as usize) {
            *x = v
        }
        ops.slider_changed(w, v)
    }
    pub fn up_pressed(&mut self, ops: &mut dyn MovieConOperations) {
        ops.increment_rate(-1)
    }
    pub fn down_pressed(&mut self, ops: &mut dyn MovieConOperations) {
        ops.increment_rate(1)
    }
    pub fn rate_entered(&mut self, ops: &mut dyn MovieConOperations) {
        if let Ok(v) = self.rate_box.trim().parse() {
            ops.rate_entered(v)
        }
    }
    pub fn snapshot_selected(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.snapshot = v;
        ops.snap_selected(v)
    }
    pub fn extent_selected(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.extent = v;
        ops.extent_selected(v)
    }
    pub fn start_here_selected(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.start_here = v;
        ops.start_here_selected(v)
    }
    pub fn montage_toggled(&mut self, ops: &mut dyn MovieConOperations, state: bool) {
        self.montage = state;
        self.montage_spin_enabled = state && self.whole_image == 0;
        ops.set_snap_montage(state)
    }
    pub fn new_montage_value(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.montage_factor = v;
        ops.set_montage_factor(v)
    }
    pub fn whole_image_selected(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.whole_image = v;
        self.montage_spin_enabled = v == 0;
        ops.set_snap_whole_mont(v)
    }
    pub fn scale_thick_toggled(&mut self, ops: &mut dyn MovieConOperations, state: bool) {
        self.scale_sizes = state;
        ops.set_scale_sizes(state)
    }
    pub fn scaling_changed(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.size_scaling = v;
        ops.set_size_scaling(v)
    }
    pub fn slicer_mont_toggled(&mut self, ops: &mut dyn MovieConOperations, state: bool) {
        self.slicer_montage = state;
        ops.set_slicer_montage(state)
    }
    pub fn new_slicer_mont_value(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.slicer_montage_factor = v;
        ops.set_slicer_mont_factor(v)
    }
    pub fn scale_slicer_thick_toggled(&mut self, ops: &mut dyn MovieConOperations, state: bool) {
        self.scale_thicks = state;
        ops.set_scale_thicks(state)
    }
    pub fn scale_thick_changed(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.thick_scaling = v;
        ops.set_thick_scaling(v)
    }
    pub fn reset_pressed(&mut self, ops: &mut dyn MovieConOperations) {
        ops.reset_pressed()
    }
    pub fn enable_time(&mut self, v: i32) {
        self.time_enabled = v != 0
    }
    pub fn set_actual_rate(&mut self, s: &str) {
        self.actual_rate = s.into()
    }
    pub fn set_rate_box(&mut self, v: f32) {
        self.rate_box = if v < 10. {
            format!("{v:5.2}")
        } else {
            format!("{v:5.1}")
        }
    }
    pub fn set_sliders(
        &mut self,
        start: i32,
        max_start: i32,
        end: i32,
        min_end: i32,
        max_end: i32,
        increment: i32,
        enable: i32,
    ) {
        self.slider_ranges = [(1, max_start), (min_end, max_end)];
        self.sliders = [start, end, increment];
        self.sliders_enabled = enable != 0
    }
    pub fn top_close_event(&mut self, ops: &mut dyn MovieConOperations) {
        ops.closing();
        self.top_window_open = false
    }
    pub fn key_press_event(&mut self, ops: &mut dyn MovieConOperations, key: i32, close: bool) {
        if close {
            self.top_window_open = false
        } else {
            ops.control_key(false, key)
        }
    }
    pub fn key_release_event(&mut self, ops: &mut dyn MovieConOperations, key: i32) {
        ops.control_key(true, key)
    }
    pub fn top_change_event(&mut self, font: bool) {
        if font {
            self.set_font_dependent_widths()
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Ops {
        v: Vec<String>,
    }
    impl MovieConOperations for Ops {
        fn axis_selected(&mut self, x: i32) {
            self.v.push(format!("axis{x}"))
        }
        fn slider_changed(&mut self, _: i32, _: i32) {}
        fn increment_rate(&mut self, _: i32) {}
        fn rate_entered(&mut self, _: f32) {}
        fn snap_selected(&mut self, _: i32) {}
        fn extent_selected(&mut self, _: i32) {}
        fn start_here_selected(&mut self, _: i32) {}
        fn set_snap_montage(&mut self, _: bool) {}
        fn set_montage_factor(&mut self, _: i32) {}
        fn set_snap_whole_mont(&mut self, _: i32) {}
        fn set_scale_sizes(&mut self, _: bool) {}
        fn set_size_scaling(&mut self, _: i32) {}
        fn set_slicer_montage(&mut self, _: bool) {}
        fn set_slicer_mont_factor(&mut self, _: i32) {}
        fn set_scale_thicks(&mut self, _: bool) {}
        fn set_thick_scaling(&mut self, _: i32) {}
        fn reset_pressed(&mut self) {}
        fn closing(&mut self) {}
        fn control_key(&mut self, _: bool, _: i32) {}
    }
    #[test]
    fn source_control_slots_forward() {
        let mut c = movie_controller_new();
        let mut o = Ops::default();
        c.axis_selected(&mut o, 1);
        assert_eq!(o.v, ["axis1"]);
        c.set_rate_box(3.5);
        assert_eq!(c.rate_box, " 3.50");
    }
}
