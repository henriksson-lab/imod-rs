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
    fn close_top_window(&mut self) {}
    fn accept_close_event(&mut self) {}
    fn check_and_set_mac_menu(&mut self) {}
    fn widget_change_event(&mut self) {}
    fn font_width(&self, _: &str) -> i32 {
        0
    }
    fn set_rate_box_maximum_width(&mut self, _: i32) {}
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
    pub scale_spin_enabled: bool,
    pub scale_check_enabled: bool,
    pub montage_radio_enabled: bool,
    pub whole_radio_enabled: bool,
    pub subarea_radio_enabled: bool,
    pub whole_image: i32,
    pub scale_sizes: bool,
    pub size_scaling: i32,
    pub slicer_montage: bool,
    pub slicer_montage_factor: i32,
    pub scale_thicks: bool,
    pub thick_scaling: i32,
    pub slicer_montage_spin_enabled: bool,
    pub slicer_scale_spin_enabled: bool,
    pub slicer_scale_check_enabled: bool,
    pub rgb_label: String,
    pub png_label: String,
    pub png_enabled: bool,
    pub close_event_accepted: bool,
    pub mac_menu_checked: bool,
}
pub fn movie_controller_new(ops: &mut dyn MovieConOperations) -> MovieController {
    let mut c = MovieController {
        top_window_open: true,
        ..Default::default()
    };
    c.init(ops);
    c
}
impl MovieController {
    pub fn destroy(&mut self) {
        self.top_window_open = false
    }
    pub fn language_change(&mut self) {}
    pub fn init(&mut self, ops: &mut dyn MovieConOperations) {
        self.delete_on_close = true;
        self.always_show_tooltips = true;
        self.snapshot = 0;
        self.extent = 0;
        self.start_here = 0;
        self.axis = 2;
        self.set_font_dependent_widths(ops);
    }
    pub fn set_font_dependent_widths(&mut self, ops: &mut dyn MovieConOperations) {
        let width = ops.font_width("8888.888");
        ops.set_rate_box_maximum_width(width);
    }
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
        } else if self.png_enabled {
            self.png_label = second.unwrap().into();
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
        ops.rate_entered(self.rate_box.trim().parse().unwrap_or(0.0))
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
        self.scale_spin_enabled = state && self.scale_sizes;
        self.scale_check_enabled = state;
        self.montage_radio_enabled = state;
        self.whole_radio_enabled = state;
        self.subarea_radio_enabled = state;
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
        self.scale_spin_enabled = state;
        ops.set_scale_sizes(state)
    }
    pub fn scaling_changed(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.size_scaling = v;
        ops.set_size_scaling(v)
    }
    pub fn slicer_mont_toggled(&mut self, ops: &mut dyn MovieConOperations, state: bool) {
        self.slicer_montage = state;
        self.slicer_montage_spin_enabled = state;
        self.slicer_scale_spin_enabled = state && self.scale_thicks;
        self.slicer_scale_check_enabled = state;
        ops.set_slicer_montage(state)
    }
    pub fn new_slicer_mont_value(&mut self, ops: &mut dyn MovieConOperations, v: i32) {
        self.slicer_montage_factor = v;
        ops.set_slicer_mont_factor(v)
    }
    pub fn scale_slicer_thick_toggled(&mut self, ops: &mut dyn MovieConOperations, state: bool) {
        self.scale_thicks = state;
        self.slicer_scale_spin_enabled = state;
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
        ops.accept_close_event();
        self.close_event_accepted = true;
        self.top_window_open = false
    }
    pub fn key_press_event(&mut self, ops: &mut dyn MovieConOperations, key: i32, close: bool) {
        if close {
            ops.close_top_window();
            self.top_window_open = false
        } else {
            ops.control_key(false, key)
        }
    }
    pub fn key_release_event(&mut self, ops: &mut dyn MovieConOperations, key: i32) {
        ops.control_key(true, key)
    }
    pub fn top_change_event(&mut self, ops: &mut dyn MovieConOperations, font: bool) {
        ops.widget_change_event();
        ops.check_and_set_mac_menu();
        self.mac_menu_checked = true;
        if font {
            self.set_font_dependent_widths(ops)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Ops {
        v: Vec<String>,
        rate_widths: Vec<i32>,
    }
    impl MovieConOperations for Ops {
        fn axis_selected(&mut self, x: i32) {
            self.v.push(format!("axis{x}"))
        }
        fn slider_changed(&mut self, _: i32, _: i32) {}
        fn increment_rate(&mut self, _: i32) {}
        fn rate_entered(&mut self, x: f32) {
            self.v.push(format!("rate{x}"))
        }
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
        fn closing(&mut self) {
            self.v.push("closing".into())
        }
        fn control_key(&mut self, _: bool, _: i32) {}
        fn accept_close_event(&mut self) {
            self.v.push("accept".into())
        }
        fn check_and_set_mac_menu(&mut self) {
            self.v.push("mac-menu".into())
        }
        fn widget_change_event(&mut self) {
            self.v.push("change-event".into())
        }
        fn font_width(&self, text: &str) -> i32 {
            text.len() as i32
        }
        fn set_rate_box_maximum_width(&mut self, width: i32) {
            self.rate_widths.push(width)
        }
    }
    #[test]
    fn source_control_slots_forward() {
        let mut o = Ops::default();
        let mut c = movie_controller_new(&mut o);
        c.axis_selected(&mut o, 1);
        assert_eq!(o.v, ["axis1"]);
        c.set_rate_box(3.5);
        assert_eq!(c.rate_box, " 3.50");
    }

    #[test]
    fn source_numeric_conversion_enablement_and_close_event_branches() {
        let mut o = Ops::default();
        let mut c = movie_controller_new(&mut o);
        c.rate_box = "invalid".into();
        c.rate_entered(&mut o);
        assert!(o.v.iter().any(|value| value == "rate0"));
        c.whole_image = 1;
        c.scale_sizes = true;
        c.montage_toggled(&mut o, true);
        assert!(!c.montage_spin_enabled);
        assert!(c.scale_spin_enabled && c.montage_radio_enabled && c.subarea_radio_enabled);
        c.top_close_event(&mut o);
        assert!(c.close_event_accepted);
        c.top_change_event(&mut o, true);
        assert!(c.mac_menu_checked);
        assert_eq!(o.rate_widths, [8, 8]);
        assert!(o.v.ends_with(&["change-event".into(), "mac-menu".into()]));
    }
}
