//! Translation of `IMOD/3dmod/form_cont_edit.cpp` and `form_cont_edit.h`.
//!
//! The generated Qt form and the `ice*`/`ivw*` application calls remain an
//! explicit boundary.  The state changes and call ordering are the source
//! implementation, rather than a replacement contour-editing UI.
#![allow(dead_code)]

use crate::imod::three_dmod::mv_window::{Key, KeyEvent};

pub const MAX_POINT_SIZE: f32 = 999.0;
pub const IMOD_GHOST_NEXTSEC: i32 = 1;
pub const IMOD_GHOST_PREVSEC: i32 = 2;
pub const IMOD_GHOST_SURFACE: i32 = 1 << 2;
pub const IMOD_GHOST_ALLOBJ: i32 = 1 << 3;
pub const IMOD_GHOST_LIGHTER: i32 = 1 << 4;
pub const IMOD_GHOST_2SHADES: i32 = 1 << 5;
pub const IMOD_GHOST_ALLSCAT: i32 = 1 << 6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContSurfPointChangeEvent {
    FontChange,
    Other,
}

/// Qt widgets, the `cont_edit.cpp` callbacks, preferences, and DockingDialog
/// operations called by `ContSurfPoint`.
pub trait ContSurfPointNativeBoundary {
    fn set_delete_on_close(&mut self);
    fn set_always_show_tool_tips(&mut self);
    fn connect_cont_surf_point_signals(&mut self);
    fn set_point_size_minimum_width(&mut self, width: i32);
    fn font_width(&self, text: &str) -> i32;
    fn set_button_width(&mut self, button: i32, rounded: bool, factor: f32, text: &str);
    fn rounded_style(&self) -> bool;
    fn set_mouse_size_checked(&mut self, state: bool);
    fn wheel_for_size(&self) -> i32;
    fn adjust_size(&mut self);
    fn ice_surf_goto(&mut self, value: i32);
    fn ice_cont_in_surf(&mut self, direction: i32);
    fn ice_surf_new(&mut self);
    fn ice_ghost_toggled(&mut self, state: i32, flag: i32);
    fn ice_label_changed(&mut self, text: &str, which: i32);
    fn ice_closed_open(&mut self, open: i32);
    fn ice_time_changed(&mut self, value: i32);
    fn ice_label_with_measure(&mut self, area: bool);
    fn hot_slider_active(&self, ctrl_pressed: bool) -> bool;
    fn ice_point_size(&mut self, size: f32);
    fn point_size_edit_text(&self) -> String;
    fn ice_set_wheel_for_size(&mut self, state: i32);
    fn ice_label_finished(&mut self, which: i32);
    fn ice_ghost_interval(&mut self, value: i32);
    fn set_enabled(&mut self, control: i32, state: bool);
    fn set_checked(&mut self, control: i32, state: bool);
    fn set_point_size_text(&mut self, text: &str);
    fn set_slider_value(&mut self, value: i32);
    fn slider_maximum(&self) -> i32;
    fn set_slider_maximum(&mut self, value: i32);
    fn set_spin_value(&mut self, control: i32, value: i32);
    fn set_spin_minimum_maximum_value(
        &mut self,
        control: i32,
        minimum: i32,
        maximum: i32,
        value: i32,
    );
    fn set_spin_special_value_text(&mut self, control: i32, text: &str);
    fn set_surface_maximum_text(&mut self, text: &str);
    fn set_edit_text(&mut self, control: i32, text: &str);
    fn ice_closing(&mut self);
    fn accept_close_event(&mut self);
    fn close_key(&self, event: KeyEvent) -> bool;
    fn close_top_window(&mut self);
    fn hot_slider_enabled(&self) -> bool;
    fn hot_slider_key(&self, key: Key) -> bool;
    fn grab_keyboard(&mut self);
    fn release_keyboard(&mut self);
    fn ivw_control_key(&mut self, release: i32, event: KeyEvent);
    fn check_and_set_mac_menu(&mut self, event: ContSurfPointChangeEvent);
}

pub const USE_LENGTH_BUTTON: i32 = 0;
pub const USE_AREA_BUTTON: i32 = 1;
pub const CLOSED_RADIO: i32 = 2;
pub const OPEN_RADIO: i32 = 3;
pub const DOWN_GHOST: i32 = 4;
pub const UP_GHOST: i32 = 5;
pub const SURFACE_GHOST: i32 = 6;
pub const LIGHTER_GHOST: i32 = 7;
pub const ALL_OBJECT_GHOST: i32 = 8;
pub const TWO_SHADES: i32 = 9;
pub const ALL_SCATTERED: i32 = 10;
pub const POINT_SIZE_EDIT: i32 = 11;
pub const POINT_SIZE_SLIDER: i32 = 12;
pub const SURFACE_SPIN: i32 = 13;
pub const TIME_SPIN: i32 = 14;
pub const SURFACE_LABEL_EDIT: i32 = 15;
pub const CONTOUR_EDIT: i32 = 16;
pub const POINT_LABEL_EDIT: i32 = 17;

/// `ContSurfPoint` (`form_cont_edit.h`).
#[derive(Clone, Debug, Default)]
pub struct ContSurfPoint {
    pub m_size_displayed: f32,
    pub m_slider_pressed: bool,
    pub m_ctrl_pressed: bool,
}

impl ContSurfPoint {
    /// `ContSurfPoint::ContSurfPoint`.
    pub fn new(native: &mut dyn ContSurfPointNativeBoundary) -> Self {
        let mut form = Self::default();
        form.init(native);
        form
    }
    /// `ContSurfPoint::~ContSurfPoint`.
    pub fn destroy(&mut self) {}
    /// `ContSurfPoint::languageChange`.
    pub fn language_change(&mut self) {}
    /// `ContSurfPoint::init`.
    pub fn init(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        n.set_delete_on_close();
        n.set_always_show_tool_tips();
        n.connect_cont_surf_point_signals();
        self.m_ctrl_pressed = false;
        self.m_slider_pressed = false;
        self.set_font_dependent_widths(n);
        n.set_mouse_size_checked(n.wheel_for_size() != 0);
        n.adjust_size();
    }
    /// `ContSurfPoint::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        n.set_point_size_minimum_width(n.font_width("  888.8-Default "));
        n.set_button_width(USE_LENGTH_BUTTON, n.rounded_style(), 1.2, "Length");
        n.set_button_width(USE_AREA_BUTTON, n.rounded_style(), 1.2, "Area");
    }
    /// `ContSurfPoint::surfaceChanged`.
    pub fn surface_changed(&mut self, value: i32, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_surf_goto(value)
    }
    /// `ContSurfPoint::upContPressed`.
    pub fn up_cont_pressed(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_cont_in_surf(1)
    }
    /// `ContSurfPoint::downContPressed`.
    pub fn down_cont_pressed(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_cont_in_surf(-1)
    }
    /// `ContSurfPoint::newSurfPressed`.
    pub fn new_surf_pressed(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_surf_new()
    }
    /// `ContSurfPoint::surfGhostToggled`.
    pub fn surf_ghost_toggled(&mut self, state: bool, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_ghost_toggled(state as i32, IMOD_GHOST_SURFACE)
    }
    /// `ContSurfPoint::surfLabelChanged`.
    pub fn surf_label_changed(&mut self, text: &str, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_label_changed(text, 2)
    }
    /// `ContSurfPoint::closedClicked`.
    pub fn closed_clicked(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        self.set_closed_open(0, 1, n);
        n.ice_closed_open(0)
    }
    /// `ContSurfPoint::openClicked`.
    pub fn open_clicked(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        self.set_closed_open(1, 1, n);
        n.ice_closed_open(1)
    }
    /// `ContSurfPoint::timeChanged`.
    pub fn time_changed(&mut self, value: i32, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_time_changed(value)
    }
    /// `ContSurfPoint::contLabelChanged`.
    pub fn cont_label_changed(&mut self, text: &str, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_label_changed(text, 0)
    }
    /// `ContSurfPoint::useLengthPressed`.
    pub fn use_length_pressed(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_label_with_measure(false)
    }
    /// `ContSurfPoint::useAreaPressed`.
    pub fn use_area_pressed(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_label_with_measure(true)
    }
    /// `ContSurfPoint::pointSliderChanged`.
    pub fn point_slider_changed(&mut self, value: i32, n: &mut dyn ContSurfPointNativeBoundary) {
        self.display_point_size(value as f32 / 10.0, 0, n);
        if !self.m_slider_pressed || n.hot_slider_active(self.m_ctrl_pressed) {
            n.ice_point_size(self.m_size_displayed)
        }
    }
    /// `ContSurfPoint::displayPointSize`.
    pub fn display_point_size(
        &mut self,
        value: f32,
        defval: i32,
        n: &mut dyn ContSurfPointNativeBoundary,
    ) {
        let mut text = format!("{value:.1}");
        if defval != 0 {
            text.push_str("-Default")
        }
        n.set_point_size_text(&text);
        self.m_size_displayed = value;
    }
    /// `ContSurfPoint::pointSliderPressed`.
    pub fn point_slider_pressed(&mut self) {
        self.m_slider_pressed = true
    }
    /// `ContSurfPoint::pointSliderReleased`.
    pub fn point_slider_released(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        self.m_slider_pressed = false;
        n.ice_point_size(self.m_size_displayed)
    }
    /// `ContSurfPoint::pointSizeEntered`.
    pub fn point_size_entered(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        let value = n
            .point_size_edit_text()
            .parse::<f32>()
            .unwrap_or(0.0)
            .clamp(0.0, MAX_POINT_SIZE);
        self.set_point_size(value, 0, n);
        n.ice_point_size(value);
    }
    /// `ContSurfPoint::mouseSizeToggled`.
    pub fn mouse_size_toggled(&mut self, state: bool, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_set_wheel_for_size(state as i32)
    }
    /// `ContSurfPoint::pointLabelChanged`.
    pub fn point_label_changed(&mut self, text: &str, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_label_changed(text, 1)
    }
    /// `ContSurfPoint::pointLabelFinished`.
    pub fn point_label_finished(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_label_finished(1)
    }
    /// `ContSurfPoint::ghostChanged`.
    pub fn ghost_changed(&mut self, value: i32, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_ghost_interval(value)
    }
    /// `ContSurfPoint::upGhostToggled`.
    pub fn up_ghost_toggled(&mut self, state: bool, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_ghost_toggled(state as i32, IMOD_GHOST_NEXTSEC)
    }
    /// `ContSurfPoint::downGhostToggled`.
    pub fn down_ghost_toggled(
        &mut self,
        state: bool,
        up_checked: bool,
        n: &mut dyn ContSurfPointNativeBoundary,
    ) {
        n.ice_ghost_toggled(state as i32, IMOD_GHOST_PREVSEC);
        n.set_enabled(TWO_SHADES, up_checked && state)
    }
    /// `ContSurfPoint::lighterGhostToggled`.
    pub fn lighter_ghost_toggled(&mut self, state: bool, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_ghost_toggled(state as i32, IMOD_GHOST_LIGHTER)
    }
    /// `ContSurfPoint::twoShadesToggled`.
    pub fn two_shades_toggled(&mut self, state: bool, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_ghost_toggled(state as i32, IMOD_GHOST_2SHADES)
    }
    /// `ContSurfPoint::allObjGhostToggled`.
    pub fn all_obj_ghost_toggled(&mut self, state: bool, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_ghost_toggled(state as i32, IMOD_GHOST_ALLOBJ)
    }
    /// `ContSurfPoint::allScatteredToggled`.
    pub fn all_scattered_toggled(&mut self, state: bool, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_ghost_toggled(state as i32, IMOD_GHOST_ALLSCAT)
    }
    /// `ContSurfPoint::setClosedOpen`.
    pub fn set_closed_open(
        &mut self,
        open: i32,
        enabled: i32,
        n: &mut dyn ContSurfPointNativeBoundary,
    ) {
        n.set_enabled(CLOSED_RADIO, enabled != 0);
        n.set_enabled(OPEN_RADIO, enabled != 0);
        n.set_checked(CLOSED_RADIO, open <= 0);
        n.set_checked(OPEN_RADIO, open > 0);
    }
    /// `ContSurfPoint::setGhostState`.
    pub fn set_ghost_state(
        &mut self,
        interval: i32,
        ghostmode: i32,
        n: &mut dyn ContSurfPointNativeBoundary,
    ) {
        n.set_spin_value(0, interval);
        n.set_checked(DOWN_GHOST, ghostmode & IMOD_GHOST_PREVSEC != 0);
        n.set_checked(UP_GHOST, ghostmode & IMOD_GHOST_NEXTSEC != 0);
        n.set_checked(SURFACE_GHOST, ghostmode & IMOD_GHOST_SURFACE != 0);
        n.set_checked(LIGHTER_GHOST, ghostmode & IMOD_GHOST_LIGHTER != 0);
        n.set_checked(ALL_OBJECT_GHOST, ghostmode & IMOD_GHOST_ALLOBJ != 0);
        n.set_checked(TWO_SHADES, ghostmode & IMOD_GHOST_2SHADES != 0);
        n.set_checked(ALL_SCATTERED, ghostmode & IMOD_GHOST_ALLSCAT != 0);
        n.set_enabled(
            TWO_SHADES,
            ghostmode & IMOD_GHOST_PREVSEC != 0 && ghostmode & IMOD_GHOST_NEXTSEC != 0,
        );
    }
    /// `ContSurfPoint::setPointSize`.
    pub fn set_point_size(
        &mut self,
        size: f32,
        defval: i32,
        n: &mut dyn ContSurfPointNativeBoundary,
    ) {
        n.set_enabled(POINT_SIZE_EDIT, defval >= 0);
        n.set_enabled(POINT_SIZE_SLIDER, defval >= 0);
        if defval < 0 {
            n.set_slider_value(0);
            n.set_point_size_text("No Point")
        } else {
            let value = (10.0 * size + 0.5).floor() as i32;
            if value > n.slider_maximum() {
                n.set_slider_maximum(value)
            }
            n.set_slider_value(value);
            self.display_point_size(size, defval, n);
        }
    }
    /// `ContSurfPoint::setSurface`.
    pub fn set_surface(
        &mut self,
        value: i32,
        max_val: i32,
        n: &mut dyn ContSurfPointNativeBoundary,
    ) {
        n.set_enabled(SURFACE_SPIN, max_val > 0);
        if value < 0 {
            n.set_spin_special_value_text(SURFACE_SPIN, "--x--");
            n.set_spin_value(SURFACE_SPIN, 0)
        } else {
            n.set_spin_special_value_text(SURFACE_SPIN, "");
            n.set_spin_minimum_maximum_value(SURFACE_SPIN, 0, max_val, value)
        }
        n.set_surface_maximum_text(&format!("/ {max_val}"));
    }
    /// `ContSurfPoint::setTimeIndex`.
    pub fn set_time_index(
        &mut self,
        value: i32,
        mut max_val: i32,
        n: &mut dyn ContSurfPointNativeBoundary,
    ) {
        n.set_enabled(TIME_SPIN, value >= 0 && max_val > 0);
        if max_val == 0 && value > 0 {
            max_val = value
        }
        if value < 0 {
            n.set_spin_special_value_text(
                TIME_SPIN,
                if value < -1 { "No Time" } else { "No Cont" },
            );
            n.set_spin_value(TIME_SPIN, 0)
        } else {
            n.set_spin_special_value_text(TIME_SPIN, "");
            n.set_spin_minimum_maximum_value(TIME_SPIN, 0, max_val, value)
        }
    }
    /// `ContSurfPoint::setLabels`.
    pub fn set_labels(
        &mut self,
        surf: &str,
        no_surf: i32,
        cont: &str,
        no_cont: i32,
        point: &str,
        no_point: i32,
        n: &mut dyn ContSurfPointNativeBoundary,
    ) {
        n.set_enabled(SURFACE_LABEL_EDIT, no_surf == 0);
        n.set_edit_text(
            SURFACE_LABEL_EDIT,
            if no_surf != 0 { "No Surface" } else { surf },
        );
        n.set_enabled(CONTOUR_EDIT, no_cont == 0);
        n.set_edit_text(CONTOUR_EDIT, if no_cont != 0 { "No Contour" } else { cont });
        n.set_enabled(POINT_LABEL_EDIT, no_point == 0);
        n.set_edit_text(
            POINT_LABEL_EDIT,
            if no_point != 0 { "No Point" } else { point },
        );
    }
    /// `ContSurfPoint::topCloseEvent`.
    pub fn top_close_event(&mut self, n: &mut dyn ContSurfPointNativeBoundary) {
        n.ice_closing();
        n.accept_close_event()
    }
    /// `ContSurfPoint::keyPressEvent`.
    pub fn key_press_event(&mut self, event: KeyEvent, n: &mut dyn ContSurfPointNativeBoundary) {
        if n.close_key(event) {
            n.close_top_window()
        } else {
            if n.hot_slider_enabled() && n.hot_slider_key(event.key) {
                self.m_ctrl_pressed = true;
                n.grab_keyboard()
            };
            n.ivw_control_key(0, event)
        }
    }
    /// `ContSurfPoint::keyReleaseEvent`.
    pub fn key_release_event(&mut self, event: KeyEvent, n: &mut dyn ContSurfPointNativeBoundary) {
        if n.hot_slider_key(event.key) {
            self.m_ctrl_pressed = false;
            n.release_keyboard()
        };
        n.ivw_control_key(1, event)
    }
    /// `ContSurfPoint::topChangeEvent`.
    pub fn top_change_event(
        &mut self,
        event: ContSurfPointChangeEvent,
        n: &mut dyn ContSurfPointNativeBoundary,
    ) {
        n.check_and_set_mac_menu(event);
        if event == ContSurfPointChangeEvent::FontChange {
            self.set_font_dependent_widths(n)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native {
        calls: Vec<String>,
        text: String,
        slider_max: i32,
    }
    impl ContSurfPointNativeBoundary for Native {
        fn set_delete_on_close(&mut self) {}
        fn set_always_show_tool_tips(&mut self) {}
        fn connect_cont_surf_point_signals(&mut self) {}
        fn set_point_size_minimum_width(&mut self, _: i32) {}
        fn font_width(&self, _: &str) -> i32 {
            1
        }
        fn set_button_width(&mut self, _: i32, _: bool, _: f32, _: &str) {}
        fn rounded_style(&self) -> bool {
            false
        }
        fn set_mouse_size_checked(&mut self, _: bool) {}
        fn wheel_for_size(&self) -> i32 {
            0
        }
        fn adjust_size(&mut self) {}
        fn ice_surf_goto(&mut self, _: i32) {}
        fn ice_cont_in_surf(&mut self, _: i32) {}
        fn ice_surf_new(&mut self) {}
        fn ice_ghost_toggled(&mut self, s: i32, f: i32) {
            self.calls.push(format!("ghost:{s}:{f}"))
        }
        fn ice_label_changed(&mut self, _: &str, _: i32) {}
        fn ice_closed_open(&mut self, _: i32) {}
        fn ice_time_changed(&mut self, _: i32) {}
        fn ice_label_with_measure(&mut self, _: bool) {}
        fn hot_slider_active(&self, _: bool) -> bool {
            false
        }
        fn ice_point_size(&mut self, x: f32) {
            self.calls.push(format!("size:{x:.1}"))
        }
        fn point_size_edit_text(&self) -> String {
            self.text.clone()
        }
        fn ice_set_wheel_for_size(&mut self, _: i32) {}
        fn ice_label_finished(&mut self, _: i32) {}
        fn ice_ghost_interval(&mut self, _: i32) {}
        fn set_enabled(&mut self, c: i32, s: bool) {
            self.calls.push(format!("enabled:{c}:{s}"))
        }
        fn set_checked(&mut self, c: i32, s: bool) {
            self.calls.push(format!("checked:{c}:{s}"))
        }
        fn set_point_size_text(&mut self, x: &str) {
            self.text = x.into()
        }
        fn set_slider_value(&mut self, _: i32) {}
        fn slider_maximum(&self) -> i32 {
            self.slider_max
        }
        fn set_slider_maximum(&mut self, x: i32) {
            self.slider_max = x
        }
        fn set_spin_value(&mut self, _: i32, _: i32) {}
        fn set_spin_minimum_maximum_value(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn set_spin_special_value_text(&mut self, _: i32, _: &str) {}
        fn set_surface_maximum_text(&mut self, _: &str) {}
        fn set_edit_text(&mut self, _: i32, _: &str) {}
        fn ice_closing(&mut self) {}
        fn accept_close_event(&mut self) {}
        fn close_key(&self, _: KeyEvent) -> bool {
            false
        }
        fn close_top_window(&mut self) {}
        fn hot_slider_enabled(&self) -> bool {
            false
        }
        fn hot_slider_key(&self, _: Key) -> bool {
            false
        }
        fn grab_keyboard(&mut self) {}
        fn release_keyboard(&mut self) {}
        fn ivw_control_key(&mut self, _: i32, _: KeyEvent) {}
        fn check_and_set_mac_menu(&mut self, _: ContSurfPointChangeEvent) {}
    }
    #[test]
    fn ghost_mode_maps_each_upstream_flag() {
        let mut n = Native::default();
        let mut f = ContSurfPoint::new(&mut n);
        f.set_ghost_state(
            3,
            IMOD_GHOST_PREVSEC | IMOD_GHOST_NEXTSEC | IMOD_GHOST_ALLOBJ,
            &mut n,
        );
        assert!(n.calls.contains(&format!("enabled:{TWO_SHADES}:true")));
    }
    #[test]
    fn point_size_rounds_limits_and_releases() {
        let mut n = Native {
            text: "1000".into(),
            ..Default::default()
        };
        let mut f = ContSurfPoint::new(&mut n);
        f.point_size_entered(&mut n);
        assert_eq!(f.m_size_displayed, MAX_POINT_SIZE);
        assert_eq!(n.slider_max, 9990);
        assert!(n.calls.contains(&"size:999.0".into()));
    }
    #[test]
    fn no_time_preserves_upstream_special_case() {
        let mut n = Native::default();
        let mut f = ContSurfPoint::new(&mut n);
        f.set_time_index(-2, 0, &mut n);
        assert!(n.calls.contains(&format!("enabled:{TIME_SPIN}:false")));
    }
}
