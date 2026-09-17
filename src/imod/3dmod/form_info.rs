//! Translation of `IMOD/3dmod/form_info.cpp` and `form_info.h`.
//!
//! Qt widget manipulation and `info_cb.cpp` calls remain explicit direct
//! boundaries; the control state and source callbacks are kept here.
#![allow(dead_code)]

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InfoImageState {
    pub num_times: i32,
    pub multi_file_z: i32,
    pub image_pyramid: bool,
}

/// Native Qt/info callback boundary used directly by this source unit.
pub trait InfoNativeBoundary {
    fn setup_ui(&mut self) {}
    fn set_delete_on_close(&mut self) {}
    fn set_always_show_tool_tips(&mut self) {}
    fn connect_info_signals(&mut self) {}
    fn set_mode_group(&mut self, _: i32) {}
    fn set_info_tool_tips(&mut self) {}
    fn set_keep_on_top_icons(&mut self) {}
    fn set_info_button_fixed_widths(&mut self) {}
    fn hide_big_model_label(&mut self) {}
    fn rounded_style(&self) -> bool {
        false
    }
    fn set_auto_button_width(&mut self, _: bool, _: f32, _: &str) {}
    fn font_width(&self, _: &str) -> i32 {
        0
    }
    fn font_height(&self) -> i32 {
        0
    }
    fn set_ocp_xyz_minimum_sizes(&mut self, _: i32, _: i32, _: i32, _: i32) {}
    fn image_state(&self) -> InfoImageState;
    fn get_float_flags(&self) -> (i32, i32, i32);
    fn info_new_xyz(&mut self, value: [i32; 3]);
    fn info_new_ocp(&mut self, item: i32, value: i32, edited: i32);
    fn info_new_bw(&mut self, which: i32, value: i32, pressed: i32);
    fn info_new_lh(&mut self, which: i32, value: i32, pressed: i32);
    fn info_mm_selected(&mut self, item: i32);
    fn info_float(&mut self, state: i32);
    fn input_raise_windows(&mut self);
    fn keep_on_top(&mut self, state: bool);
    fn info_subset(&mut self, state: i32);
    fn info_t_ramps(&mut self, state: i32);
    fn auto_contrast_targets(&self) -> (i32, i32);
    fn info_auto_contrast(&mut self, mean: i32, sd: i32);
    fn input_undo_redo(&mut self, redo: bool);
    fn retranslate_ui(&mut self);
}

/// `InfoControls` (`form_info.h`), including the form's widget values.
#[derive(Clone, Debug)]
pub struct InfoControls {
    pub m_show_point: bool,
    pub m_displayed_white: i32,
    pub m_displayed_black: i32,
    pub m_displayed_low: i32,
    pub m_displayed_high: i32,
    pub m_ctrl_pressed: bool,
    pub m_black_pressed: bool,
    pub m_white_pressed: bool,
    pub m_low_pressed: bool,
    pub m_high_pressed: bool,
    pub m_scale_min: f32,
    pub m_scale_max: f32,
    pub m_show_lh_real: bool,
    pub m_last_xyzval: [i32; 3],
    pub m_last_xyzmax: [i32; 3],
    pub m_last_ocpval: [i32; 3],
    pub m_last_ocpmax: [i32; 3],
    pub m_str: String,
    pub m_ocp_value: [i32; 3],
    pub m_ocp_max: [i32; 3],
    pub m_xyz_value: [i32; 3],
    pub m_xyz_max: [i32; 3],
    pub float_checked: bool,
    pub subarea_checked: bool,
    pub t_ramps_checked: bool,
    pub show_checked: bool,
    pub float_enabled: bool,
    pub subarea_enabled: bool,
    pub auto_enabled: bool,
    pub undo_enabled: bool,
    pub redo_enabled: bool,
    pub t_ramps_visible: bool,
    pub low_high_visible: bool,
    pub movie_model: i32,
    pub object_fore_color: u32,
    pub object_back_color: u32,
    pub model_name: String,
    pub image_name: String,
    pub big_model_visible: bool,
    pub model_label_width: i32,
    pub width: i32,
    pub height_hint: i32,
}
impl Default for InfoControls {
    fn default() -> Self {
        Self {
            m_show_point: false,
            m_displayed_white: 0,
            m_displayed_black: 0,
            m_displayed_low: 0,
            m_displayed_high: 0,
            m_ctrl_pressed: false,
            m_black_pressed: false,
            m_white_pressed: false,
            m_low_pressed: false,
            m_high_pressed: false,
            m_scale_min: 0.,
            m_scale_max: 0.,
            m_show_lh_real: false,
            m_last_xyzval: [0; 3],
            m_last_xyzmax: [0; 3],
            m_last_ocpval: [0; 3],
            m_last_ocpmax: [0; 3],
            m_str: String::new(),
            m_ocp_value: [0; 3],
            m_ocp_max: [0; 3],
            m_xyz_value: [0; 3],
            m_xyz_max: [0; 3],
            float_checked: false,
            subarea_checked: false,
            t_ramps_checked: false,
            show_checked: false,
            float_enabled: true,
            subarea_enabled: true,
            auto_enabled: true,
            undo_enabled: false,
            redo_enabled: false,
            t_ramps_visible: true,
            low_high_visible: true,
            movie_model: 0,
            object_fore_color: 0,
            object_back_color: 0,
            model_name: String::new(),
            image_name: String::new(),
            big_model_visible: false,
            model_label_width: 0,
            width: 454,
            height_hint: 376,
        }
    }
}
impl InfoControls {
    /// `InfoControls()` source constructor.
    pub fn new(native: &mut dyn InfoNativeBoundary) -> Self {
        native.setup_ui();
        let mut out = Self::default();
        out.init(native);
        out
    }
    /// `InfoControls::~InfoControls`.
    pub fn destroy(&mut self) {}
    /// `InfoControls::languageChange`.
    pub fn language_change(&mut self, native: &mut dyn InfoNativeBoundary) {
        #[cfg(not(target_os = "linux"))]
        native.retranslate_ui();
        #[cfg(target_os = "linux")]
        let _ = native;
    }
    /// `InfoControls::init`.
    pub fn init(&mut self, native: &mut dyn InfoNativeBoundary) {
        native.set_delete_on_close();
        native.set_always_show_tool_tips();
        native.connect_info_signals();
        self.m_ctrl_pressed = false;
        self.m_black_pressed = false;
        self.m_white_pressed = false;
        self.m_low_pressed = false;
        self.m_high_pressed = false;
        self.set_show_point(1);
        let (float_on, subarea, t_ramps) = native.get_float_flags();
        self.float_checked = float_on != 0;
        self.subarea_checked = subarea != 0;
        self.t_ramps_checked = t_ramps != 0;
        self.show_or_hide_ramps(native);
        native.set_info_tool_tips();
        native.set_mode_group(0);
        self.m_last_ocpval = [-2; 3];
        self.m_last_ocpmax = [-2; 3];
        self.m_last_xyzval = [-2; 3];
        self.m_last_xyzmax = [-2; 3];
        self.set_font_dependent_widths(native);
        native.set_keep_on_top_icons();
        native.set_info_button_fixed_widths();
        self.set_undo_redo(false, false);
        self.big_model_visible = false;
        native.hide_big_model_label();
    }
    /// `InfoControls::showOrHideRamps`.
    pub fn show_or_hide_ramps(&mut self, native: &mut dyn InfoNativeBoundary) {
        let cvi = native.image_state();
        self.t_ramps_visible = !(cvi.num_times < 2 || cvi.multi_file_z > 0 || cvi.image_pyramid);
    }
    /// `InfoControls::hideLowHighGrid`.
    pub fn hide_low_high_grid(&mut self) {
        self.low_high_visible = false;
        self.height_hint -= 20;
    }
    /// `InfoControls::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self, native: &mut dyn InfoNativeBoundary) {
        native.set_auto_button_width(native.rounded_style(), 1.35, "Auto");
        let width = native.font_width("88888888");
        let label_width = native.font_width("/ 8888");
        let height = (1.45 * native.font_height() as f32).round() as i32;
        for index in 0..3 {
            native.set_ocp_xyz_minimum_sizes(index, width, height, label_width);
        }
    }
    /// `InfoControls::adjustedHeightHint`.
    pub fn adjusted_height_hint(&mut self) -> i32 {
        if self.low_high_visible {
            self.height_hint
        } else {
            self.height_hint - 24
        }
    }
    /// `InfoControls::xyzChanged`.
    pub fn xyz_changed(&mut self, item: i32, value: i32, n: &mut dyn InfoNativeBoundary) {
        self.m_last_xyzval[item as usize] = value;
        self.m_xyz_value[item as usize] = value;
        n.info_new_xyz(self.m_last_xyzval);
    }
    /// `InfoControls::ocpChanged`.
    pub fn ocp_changed(&mut self, item: i32, mut value: i32, n: &mut dyn InfoNativeBoundary) {
        let i = item as usize;
        let mut diff = self.m_last_ocpval[i] - value;
        if diff < 0 {
            diff = -diff
        };
        let _ = diff;
        let edited = if self.m_show_point { 0 } else { 1 };
        if value == 0 {
            value = if self.m_last_ocpval[i] == 1 {
                self.m_last_ocpmax[i]
            } else {
                1
            };
            self.m_ocp_value[i] = value;
        }
        if self.m_last_ocpval[i] != value {
            n.info_new_ocp(item, value, edited)
        }
        self.m_last_ocpval[i] = value;
    }
    /// `InfoControls::blackChanged`.
    pub fn black_changed(&mut self, value: i32, n: &mut dyn InfoNativeBoundary) {
        n.info_new_bw(0, value, self.m_black_pressed as i32)
    }
    /// `InfoControls::blackPressed`.
    pub fn black_pressed(&mut self) {
        self.m_black_pressed = true
    }
    /// `InfoControls::blackReleased`.
    pub fn black_released(&mut self, n: &mut dyn InfoNativeBoundary) {
        self.m_black_pressed = false;
        self.black_changed(self.m_displayed_black, n)
    }
    /// `InfoControls::displayBlack`.
    pub fn display_black(&mut self, value: i32) {
        self.m_str = value.to_string();
        self.m_displayed_black = value
    }
    /// `InfoControls::whiteChanged`.
    pub fn white_changed(&mut self, value: i32, n: &mut dyn InfoNativeBoundary) {
        n.info_new_bw(1, value, self.m_white_pressed as i32)
    }
    /// `InfoControls::whitePressed`.
    pub fn white_pressed(&mut self) {
        self.m_white_pressed = true
    }
    /// `InfoControls::whiteReleased`.
    pub fn white_released(&mut self, n: &mut dyn InfoNativeBoundary) {
        self.m_white_pressed = false;
        self.white_changed(self.m_displayed_white, n)
    }
    /// `InfoControls::displayWhite`.
    pub fn display_white(&mut self, value: i32) {
        self.m_str = value.to_string();
        self.m_displayed_white = value
    }
    /// `InfoControls::lowChanged`.
    pub fn low_changed(&mut self, value: i32, n: &mut dyn InfoNativeBoundary) {
        n.info_new_lh(0, value, self.m_low_pressed as i32)
    }
    /// `InfoControls::lowPressed`.
    pub fn low_pressed(&mut self) {
        self.m_low_pressed = true
    }
    /// `InfoControls::lowReleased`.
    pub fn low_released(&mut self, n: &mut dyn InfoNativeBoundary) {
        self.m_low_pressed = false;
        self.low_changed(self.m_displayed_low, n)
    }
    /// `InfoControls::displayLow`.
    pub fn display_low(&mut self, value: i32) {
        self.format_lhvalue(value);
        self.m_displayed_low = value
    }
    /// `InfoControls::highChanged`.
    pub fn high_changed(&mut self, value: i32, n: &mut dyn InfoNativeBoundary) {
        n.info_new_lh(1, value, self.m_high_pressed as i32)
    }
    /// `InfoControls::highPressed`.
    pub fn high_pressed(&mut self) {
        self.m_high_pressed = true
    }
    /// `InfoControls::highReleased`.
    pub fn high_released(&mut self, n: &mut dyn InfoNativeBoundary) {
        self.m_high_pressed = false;
        self.high_changed(self.m_displayed_high, n)
    }
    /// `InfoControls::displayHigh`.
    pub fn display_high(&mut self, value: i32) {
        self.format_lhvalue(value);
        self.m_displayed_high = value
    }
    /// `InfoControls::formatLHvalue`.
    pub fn format_lhvalue(&mut self, value: i32) {
        let scaled =
            value as f32 * (self.m_scale_max - self.m_scale_min) / 65535. + self.m_scale_min;
        self.m_str = if self.m_show_lh_real {
            format!("{scaled:.4}")
        } else {
            (scaled.round() as i32).to_string()
        }
    }
    /// `InfoControls::movieModelSelected`.
    pub fn movie_model_selected(&mut self, item: i32, n: &mut dyn InfoNativeBoundary) {
        n.info_mm_selected(item)
    }
    /// `InfoControls::floatToggled`.
    pub fn float_toggled(&mut self, state: bool, n: &mut dyn InfoNativeBoundary) {
        n.info_float(state as i32)
    }
    /// `InfoControls::raisePressed`.
    pub fn raise_pressed(&mut self, n: &mut dyn InfoNativeBoundary) {
        n.input_raise_windows()
    }
    /// `InfoControls::showPointToggled`.
    pub fn show_point_toggled(&mut self, state: bool) {
        self.m_show_point = state
    }
    /// `InfoControls::keepOnTopToggled`.
    pub fn keep_on_top_toggled(&mut self, state: bool, n: &mut dyn InfoNativeBoundary) {
        n.keep_on_top(state)
    }
    /// `InfoControls::subareaToggled`.
    pub fn subarea_toggled(&mut self, state: bool, n: &mut dyn InfoNativeBoundary) {
        n.info_subset(state as i32)
    }
    /// `InfoControls::tRampsToggled`.
    pub fn t_ramps_toggled(&mut self, state: bool, n: &mut dyn InfoNativeBoundary) {
        n.info_t_ramps(state as i32)
    }
    /// `InfoControls::autoClicked`.
    pub fn auto_clicked(&mut self, n: &mut dyn InfoNativeBoundary) {
        let (mean, sd) = n.auto_contrast_targets();
        n.info_auto_contrast(mean, sd)
    }
    /// `InfoControls::undoClicked`.
    pub fn undo_clicked(&mut self, n: &mut dyn InfoNativeBoundary) {
        n.input_undo_redo(false)
    }
    /// `InfoControls::redoClicked`.
    pub fn redo_clicked(&mut self, n: &mut dyn InfoNativeBoundary) {
        n.input_undo_redo(true)
    }
    /// `InfoControls::setFloat`.
    pub fn set_float(&mut self, state: i32) {
        self.float_enabled = state >= 0;
        self.subarea_enabled = state >= 0;
        self.auto_enabled = state >= 0;
        self.float_checked = state > 0
    }
    /// `InfoControls::setSubarea`.
    pub fn set_subarea(&mut self, state: i32) {
        self.subarea_checked = state > 0
    }
    /// `InfoControls::setUndoRedo`.
    pub fn set_undo_redo(&mut self, undo_on: bool, redo_on: bool) {
        self.undo_enabled = undo_on;
        self.redo_enabled = redo_on
    }
    /// `InfoControls::setBWSliders`.
    pub fn set_bw_sliders(&mut self, black: i32, white: i32) {
        self.display_black(black);
        self.display_white(white)
    }
    /// `InfoControls::setLHSliders`.
    pub fn set_lh_sliders(&mut self, low: i32, high: i32, smin: f32, smax: f32, show_real: bool) {
        self.m_scale_min = smin;
        self.m_scale_max = smax;
        self.m_show_lh_real = show_real;
        self.display_low(low);
        self.display_high(high)
    }
    /// C++ `setLHSliders(low, high)` overload.
    pub fn set_lh_sliders_default_scale(&mut self, low: i32, high: i32) {
        self.set_lh_sliders(
            low,
            high,
            self.m_scale_min,
            self.m_scale_max,
            self.m_show_lh_real,
        )
    }
    /// `InfoControls::setMovieModel`.
    pub fn set_movie_model(&mut self, which: i32) {
        self.movie_model = which
    }
    /// `InfoControls::updateOCP`.
    pub fn update_ocp(&mut self, new_val: [i32; 3], max_val: [i32; 3]) {
        for i in 0..3 {
            if self.m_last_ocpmax[i] != max_val[i] {
                self.m_ocp_max[i] = max_val[i].max(0)
            }
            if self.m_last_ocpval[i] != new_val[i] {
                self.m_ocp_value[i] = new_val[i].max(0)
            }
            self.m_last_ocpmax[i] = max_val[i];
            self.m_last_ocpval[i] = new_val[i];
        }
    }
    /// `InfoControls::updateXYZ`, retaining the source comparison against `mLastXYZval`.
    pub fn update_xyz(&mut self, new_val: [i32; 3], max_val: [i32; 3]) {
        for i in 0..3 {
            if max_val[i] != self.m_last_xyzval[i] {
                self.m_xyz_max[i] = max_val[i]
            }
            if new_val[i] != self.m_last_xyzval[i] {
                self.m_xyz_value[i] = new_val[i]
            }
            self.m_last_xyzval[i] = new_val[i];
            self.m_last_xyzmax[i] = max_val[i];
        }
    }
    /// `InfoControls::setObjectColor`.
    pub fn set_object_color(&mut self, fore_color: u32, back_color: u32) {
        self.object_fore_color = fore_color;
        self.object_back_color = back_color
    }
    /// `InfoControls::setModelName`.
    pub fn set_model_name(&mut self, name: &str) {
        self.m_str = name.into();
        let max_width = self.width - 10;
        if self.m_str.len() as i32 > max_width {
            let len = self.m_str.len() / 2;
            self.m_str = format!(
                "{}...{}",
                &self.m_str[..len.saturating_sub(1)],
                &self.m_str[len.saturating_add(1)..]
            )
        }
        self.big_model_visible = self.m_str.len() as i32 + 1 > self.model_label_width;
        self.model_name = self.m_str.clone()
    }
    /// `InfoControls::setImageName`.
    pub fn set_image_name(&mut self, name: &str) {
        self.m_str = name.into();
        self.image_name = self.m_str.clone()
    }
    /// `InfoControls::setShowPoint`.
    pub fn set_show_point(&mut self, state: i32) {
        self.m_show_point = state != 0;
        self.show_checked = self.m_show_point
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N(Vec<String>);
    impl InfoNativeBoundary for N {
        fn setup_ui(&mut self) {}
        fn set_delete_on_close(&mut self) {}
        fn set_always_show_tool_tips(&mut self) {}
        fn connect_info_signals(&mut self) {}
        fn set_mode_group(&mut self, _: i32) {}
        fn set_info_tool_tips(&mut self) {}
        fn set_keep_on_top_icons(&mut self) {}
        fn set_info_button_fixed_widths(&mut self) {}
        fn hide_big_model_label(&mut self) {}
        fn rounded_style(&self) -> bool {
            false
        }
        fn set_auto_button_width(&mut self, _: bool, _: f32, _: &str) {}
        fn font_width(&self, _: &str) -> i32 {
            0
        }
        fn font_height(&self) -> i32 {
            0
        }
        fn set_ocp_xyz_minimum_sizes(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn image_state(&self) -> InfoImageState {
            InfoImageState::default()
        }
        fn get_float_flags(&self) -> (i32, i32, i32) {
            (0, 0, 0)
        }
        fn info_new_xyz(&mut self, v: [i32; 3]) {
            self.0.push(format!("xyz:{v:?}"))
        }
        fn info_new_ocp(&mut self, i: i32, v: i32, e: i32) {
            self.0.push(format!("ocp:{i}:{v}:{e}"))
        }
        fn info_new_bw(&mut self, i: i32, v: i32, p: i32) {
            self.0.push(format!("bw:{i}:{v}:{p}"))
        }
        fn info_new_lh(&mut self, _: i32, _: i32, _: i32) {}
        fn info_mm_selected(&mut self, _: i32) {}
        fn info_float(&mut self, _: i32) {}
        fn input_raise_windows(&mut self) {}
        fn keep_on_top(&mut self, _: bool) {}
        fn info_subset(&mut self, _: i32) {}
        fn info_t_ramps(&mut self, _: i32) {}
        fn auto_contrast_targets(&self) -> (i32, i32) {
            (0, 0)
        }
        fn info_auto_contrast(&mut self, _: i32, _: i32) {}
        fn input_undo_redo(&mut self, _: bool) {}
        fn retranslate_ui(&mut self) {}
    }
    #[test]
    fn black_release_sends_final_unpressed() {
        let mut n = N::default();
        let mut f = InfoControls::new(&mut n);
        f.display_black(9);
        f.black_pressed();
        f.black_changed(7, &mut n);
        f.black_released(&mut n);
        assert_eq!(n.0, ["bw:0:7:1", "bw:0:9:0"])
    }
    #[test]
    fn ocp_zero_wraps() {
        let mut n = N::default();
        let mut f = InfoControls::new(&mut n);
        f.m_last_ocpval[0] = 1;
        f.m_last_ocpmax[0] = 5;
        f.ocp_changed(0, 0, &mut n);
        assert_eq!(n.0, ["ocp:0:5:0"])
    }
}
