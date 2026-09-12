//! Translation of `IMOD/3dmod/zap_classes.cpp` and `zap_classes.h`.
//!
//! Qt widgets, the live OpenGL widget, and `ZapFuncs` are deliberately kept as
//! explicit boundaries.  This unit owns the source window/widget state and
//! forwards every source slot/event to that boundary.
#![allow(dead_code)]

use crate::imod::three_dmod::imod_input::{INPUT_CTRL, INPUT_SHIFT};
use crate::imod::three_dmod::utilities::{PopupEntry, util_lookup_popup_hit};

pub const NUM_TOOLBUTTONS: usize = 8;
pub const NUM_TIMEBUTTONS: usize = 1;
pub const ZAP_TOGGLE_RESOL: usize = 0;
pub const ZAP_TOGGLE_ZLOCK: usize = 1;
pub const ZAP_TOGGLE_CENTER: usize = 2;
pub const ZAP_TOGGLE_INSERT: usize = 3;
pub const ZAP_TOGGLE_RUBBER: usize = 4;
pub const ZAP_TOGGLE_LASSO: usize = 5;
pub const ZAP_TOGGLE_ARROW: usize = 6;
pub const ZAP_TOGGLE_TIMELOCK: usize = 7;
pub const MULTIZ_MAX_PANELS: i32 = 20;
pub const MIN_SLIDER_WIDTH: i32 = 20;
pub const MAX_SLIDER_WIDTH: i32 = 100;

/// `sPopupTable` in `zap_classes.cpp`, excluding its C sentinel row.
const ZAP_POPUP_TABLE: &[PopupEntry] = &[
    PopupEntry {
        text: "Toggle automatic section advance",
        key: 'Z' as i32,
        ctrl: false,
        shift: true,
        main_index: 0,
    },
    PopupEntry {
        text: "Toggle modeling direction",
        key: 'I' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
    PopupEntry {
        text: "Toggle centering mode",
        key: 'K' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
    PopupEntry {
        text: "Print area information, raise Info window",
        key: 'I' as i32,
        ctrl: false,
        shift: true,
        main_index: 0,
    },
    PopupEntry {
        text: "Toggle rubber band",
        key: 'B' as i32,
        ctrl: false,
        shift: true,
        main_index: 0,
    },
    PopupEntry {
        text: "Report distance from current point to cursor",
        key: 'Q' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
    PopupEntry {
        text: "Toggle adjusting contour with mouse",
        key: 'P' as i32,
        ctrl: false,
        shift: true,
        main_index: 0,
    },
    PopupEntry {
        text: "Add contours on section to selection list",
        key: 'A' as i32,
        ctrl: true,
        shift: false,
        main_index: 0,
    },
    PopupEntry {
        text: "Add contours from all objects to selection",
        key: 'A' as i32,
        ctrl: true,
        shift: true,
        main_index: 0,
    },
    PopupEntry {
        text: "Resize window to image or rubber band",
        key: 'R' as i32,
        ctrl: false,
        shift: true,
        main_index: 0,
    },
    PopupEntry {
        text: "Resize area within rubber band to fit window",
        key: 'R' as i32,
        ctrl: true,
        shift: true,
        main_index: 0,
    },
];

/// Native Qt, OpenGL, image-view, and `ZapFuncs` boundary.
pub trait ZapNativeBoundary {
    fn step_zoom(&mut self, step: i32);
    fn entered_zoom(&mut self, zoom: f32);
    fn entered_section(&mut self, section: i32);
    fn help(&mut self);
    fn print_info(&mut self);
    fn step_time(&mut self, step: i32);
    fn state_toggled(&mut self, index: usize, state: i32);
    fn setup_panels(&mut self) -> i32;
    fn update_gl(&mut self);
    fn key_input(&mut self, key: i32, modifiers: i32);
    fn key_release(&mut self, key: i32, modifiers: i32);
    fn general_event(&mut self, event: i32);
    fn mouse_press(&mut self, event: i32);
    fn mouse_release(&mut self, event: i32);
    fn mouse_move(&mut self, event: i32);
    fn paint(&mut self);
    fn resize(&mut self, width: i32, height: i32);
    fn closing(&mut self);
    fn imod_loop_started(&self) -> bool;
    fn hot_slider_active(&self, ctrl_pressed: bool) -> bool;
    fn hot_slider_key(&self, key: i32) -> bool;
    fn tilt_angles(&self) -> Option<Vec<f32>>;
    fn zsize(&self) -> i32;
    fn current_section(&self) -> i32;
    fn time(&self) -> i32;
    fn num_xpanels(&self) -> i32;
    fn num_ypanels(&self) -> i32;
    fn panel_zstep(&self) -> i32;
    fn draw_in_center(&self) -> i32;
    fn draw_in_others(&self) -> i32;
    fn set_num_xpanels(&mut self, value: i32);
    fn set_num_ypanels(&mut self, value: i32);
    fn set_panel_zstep(&mut self, value: i32);
    fn set_draw_in_center(&mut self, value: i32);
    fn set_draw_in_others(&mut self, value: i32);
    fn wall_time_msec(&self) -> i32;
    fn process_events(&mut self);
    fn rounded_style(&self) -> bool {
        false
    }
    fn info_button_exists(&self) -> bool {
        false
    }
    fn low_section_button_exists(&self) -> bool {
        false
    }
    fn high_section_button_exists(&self) -> bool {
        false
    }
    fn button_width(&self, _: bool, _: f32, _: &str) -> i32 {
        0
    }
    fn set_button_fixed_width(&mut self, _: &str, _: i32) {}
    fn widget_change_event(&mut self) {}
    fn check_and_set_mac_menu(&mut self) {}
    /// `ZapFuncs::screenChanged`, called with the new device-pixel ratio.
    fn screen_changed(&mut self, _: f32) {}
}

/// `ZapWindow` state formerly held in the paired Qt header.
#[derive(Clone, Debug)]
pub struct ZapWindow {
    pub m_toggle_states: [i32; NUM_TOOLBUTTONS],
    pub m_zoom_text: String,
    pub m_section_text: String,
    pub m_low_section_text: String,
    pub m_high_section_text: String,
    pub m_time_label: String,
    pub m_time_num_label: String,
    pub m_size_text: String,
    pub m_angle_text: String,
    pub m_sec_slider: i32,
    pub m_sec_slider_max: i32,
    pub m_size_angle_state: i32,
    pub m_sec_pressed: bool,
    pub m_displayed_section: i32,
    pub m_ctrl_pressed: bool,
    pub m_low_high_visible: bool,
    pub m_panels: bool,
    pub m_rows: i32,
    pub m_columns: i32,
    pub m_zstep: i32,
    pub m_closed: bool,
}
impl Default for ZapWindow {
    fn default() -> Self {
        Self {
            m_sec_slider_max: 1,
            m_size_angle_state: -1,
            ..Self {
                m_toggle_states: [0; NUM_TOOLBUTTONS],
                m_zoom_text: String::new(),
                m_section_text: String::new(),
                m_low_section_text: String::new(),
                m_high_section_text: String::new(),
                m_time_label: String::new(),
                m_time_num_label: String::new(),
                m_size_text: String::new(),
                m_angle_text: String::new(),
                m_sec_slider: 1,
                m_sec_slider_max: 1,
                m_size_angle_state: -1,
                m_sec_pressed: false,
                m_displayed_section: 0,
                m_ctrl_pressed: false,
                m_low_high_visible: false,
                m_panels: false,
                m_rows: 1,
                m_columns: 1,
                m_zstep: 1,
                m_closed: false,
            }
        }
    }
}
impl ZapWindow {
    /// `ZapWindow::ZapWindow` (Qt construction itself is the boundary).
    pub fn new(time_label: &str, panels: bool, n: &mut dyn ZapNativeBoundary) -> Self {
        let mut s = Self {
            m_panels: panels,
            m_sec_slider_max: n.zsize(),
            m_rows: n.num_ypanels(),
            m_columns: n.num_xpanels(),
            m_zstep: n.panel_zstep(),
            ..Default::default()
        };
        s.set_low_high_section_state(0);
        if !time_label.is_empty() {
            s.set_time_label(n.time(), time_label);
        }
        s.set_font_dependent_widths(n);
        s
    }
    pub fn destroy(&mut self) {}
    pub fn set_font_dependent_widths(&mut self, n: &mut dyn ZapNativeBoundary) {
        let rounded = n.rounded_style();
        if n.info_button_exists() {
            let width = n.button_width(rounded, 1., "I");
            n.set_button_fixed_width("mInfoButton", 10 + width);
        }
        let width = n.button_width(rounded, 1.2, "Help");
        n.set_button_fixed_width("mHelpButton", width);
        if n.low_section_button_exists() {
            let width = n.button_width(rounded, 1., "Lo");
            n.set_button_fixed_width("mLowSectionButton", 8 + width);
        }
        if n.high_section_button_exists() {
            let width = n.button_width(rounded, 1., "Hi");
            n.set_button_fixed_width("mHighSectionButton", 8 + width);
        }
    }
    pub fn change_event(&mut self, font_change: bool, n: &mut dyn ZapNativeBoundary) {
        n.widget_change_event();
        n.check_and_set_mac_menu();
        if font_change {
            self.set_font_dependent_widths(n)
        }
    }
    pub fn zoom_up(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.step_zoom(1)
    }
    pub fn zoom_down(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.step_zoom(-1)
    }
    pub fn new_zoom(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.entered_zoom(self.m_zoom_text.parse().unwrap_or(0.0))
    }
    pub fn new_section(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.entered_section(self.m_section_text.parse().unwrap_or(0))
    }
    pub fn slider_changed(&mut self, value: i32, n: &mut dyn ZapNativeBoundary) {
        if !self.m_sec_pressed || n.hot_slider_active(self.m_ctrl_pressed) {
            n.entered_section(value)
        } else {
            self.set_section_text(value, n)
        }
    }
    pub fn sec_pressed(&mut self) {
        self.m_sec_pressed = true
    }
    pub fn sec_released(&mut self, n: &mut dyn ZapNativeBoundary) {
        self.m_sec_pressed = false;
        n.entered_section(self.m_displayed_section)
    }
    pub fn help(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.help()
    }
    pub fn info(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.print_info()
    }
    pub fn time_back(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.step_time(-1)
    }
    pub fn time_forward(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.step_time(1)
    }
    pub fn toggle_clicked(&mut self, index: usize, checked: bool, n: &mut dyn ZapNativeBoundary) {
        let state = i32::from(checked);
        self.m_toggle_states[index] = state;
        n.state_toggled(index, state)
    }
    pub fn set_low_high_section_state(&mut self, rubberband_selected: i32) {
        self.m_low_high_visible = rubberband_selected != 0 && !self.m_panels;
        if !self.m_low_high_visible {
            self.m_low_section_text.clear();
            self.m_high_section_text.clear()
        }
    }
    pub fn set_toggle_state(&mut self, index: usize, state: i32) {
        self.m_toggle_states[index] = i32::from(state != 0)
    }
    pub fn set_zoom_text(&mut self, zoom: f32) {
        let mut text = format!("{zoom:.4}");
        if text.ends_with("00") {
            text.truncate(text.len() - 2)
        };
        self.m_zoom_text = text
    }
    pub fn set_size_text(&mut self, winx: i32, winy: i32, n: &dyn ZapNativeBoundary) {
        if !self.m_panels && n.tilt_angles().is_none() {
            self.set_size_angle_state(n);
            self.m_size_text = format!(" {winx}x{winy}")
        }
    }
    pub fn set_section_text(&mut self, section: i32, n: &dyn ZapNativeBoundary) {
        self.m_section_text = section.to_string();
        self.m_sec_slider = section;
        self.m_displayed_section = section;
        if let Some(angles) = n.tilt_angles() {
            self.set_size_angle_state(n);
            let a = angles
                .get(section.saturating_sub(1) as usize)
                .copied()
                .unwrap_or(0.);
            self.m_angle_text = format!("&nbsp;{a:.1}&deg;")
        }
    }
    pub fn set_size_angle_state(&mut self, n: &dyn ZapNativeBoundary) {
        let state = i32::from(n.tilt_angles().is_some());
        if state != self.m_size_angle_state {
            self.m_size_angle_state = state
        }
    }
    pub fn set_max_z(&mut self, max_z: i32) {
        self.m_sec_slider_max = max_z
    }
    pub fn set_time_label(&mut self, time: i32, label: &str) {
        self.m_time_num_label = format!(" ({time:3})");
        self.m_time_label = label.into()
    }
    pub fn rows_changed(&mut self, value: i32, n: &mut dyn ZapNativeBoundary) {
        self.m_rows = value;
        n.set_num_ypanels(value);
        if n.setup_panels() == 0 {
            n.update_gl()
        }
    }
    pub fn columns_changed(&mut self, value: i32, n: &mut dyn ZapNativeBoundary) {
        self.m_columns = value;
        n.set_num_xpanels(value);
        if n.setup_panels() == 0 {
            n.update_gl()
        }
    }
    pub fn z_step_changed(&mut self, value: i32, n: &mut dyn ZapNativeBoundary) {
        self.m_zstep = value;
        n.set_panel_zstep(value);
        n.update_gl()
    }
    pub fn low_section(&self) -> String {
        self.m_low_section_text.clone()
    }
    pub fn high_section(&self) -> String {
        self.m_high_section_text.clone()
    }
    pub fn set_low_section(&mut self, n: &dyn ZapNativeBoundary) {
        self.m_low_section_text = (n.current_section() + 1).to_string()
    }
    pub fn set_high_section(&mut self, n: &dyn ZapNativeBoundary) {
        self.m_high_section_text = (n.current_section() + 1).to_string()
    }
    pub fn draw_center_toggled(&mut self, state: bool, n: &mut dyn ZapNativeBoundary) {
        n.set_draw_in_center(i32::from(state));
        n.update_gl()
    }
    pub fn draw_others_toggled(&mut self, state: bool, n: &mut dyn ZapNativeBoundary) {
        n.set_draw_in_others(i32::from(state));
        n.update_gl()
    }
    /// `ZapWindow::screenChanged`.
    pub fn screen_changed(&mut self, device_pixel_ratio: f32, n: &mut dyn ZapNativeBoundary) {
        n.screen_changed(device_pixel_ratio)
    }
    pub fn key_press_event(&mut self, key: i32, modifiers: i32, n: &mut dyn ZapNativeBoundary) {
        if n.hot_slider_key(key) {
            self.m_ctrl_pressed = true
        };
        n.key_input(key, modifiers)
    }
    pub fn key_release_event(&mut self, key: i32, modifiers: i32, n: &mut dyn ZapNativeBoundary) {
        if n.hot_slider_key(key) {
            self.m_ctrl_pressed = false
        };
        n.key_release(key, modifiers)
    }
    pub fn wheel_event(&mut self, event: i32, n: &mut dyn ZapNativeBoundary) {
        n.general_event(event)
    }
    pub fn close_event(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.closing();
        self.m_closed = true
    }
    /// `ZapWindow::toolbarMenuEvent` followed by its mapped `contextMenuHit` slot.
    ///
    /// Native Qt popup construction is the boundary; `index` is the selected mapped
    /// action.  `utilLookupPopupHit` appends the shared default actions after this
    /// source-specific table, exactly as the C++ call with `-1` does.
    pub fn toolbar_menu_event(&mut self, index: i32, n: &mut dyn ZapNativeBoundary) {
        let Some((key, ctrl, shift)) =
            util_lookup_popup_hit(index.max(0) as usize, ZAP_POPUP_TABLE, -1)
        else {
            return;
        };
        let modifiers = (if ctrl { INPUT_CTRL } else { 0 }) | (if shift { INPUT_SHIFT } else { 0 });
        self.context_menu_hit(key, modifiers as i32, n)
    }
    pub fn context_menu_hit(&mut self, key: i32, modifiers: i32, n: &mut dyn ZapNativeBoundary) {
        n.key_input(key, modifiers)
    }
}

/// `ZapGL` state and timer/event forwarding from `zap_classes.h`.
#[derive(Clone, Debug)]
pub struct ZapGl {
    pub m_init_width: i32,
    pub m_init_height: i32,
    pub m_init_left: i32,
    pub m_init_top: i32,
    pub m_first_draw: i32,
    pub m_mouse_pressed: bool,
    pub m_mouse_in_window: i32,
    pub m_num_first_draws: i32,
    pub m_timer_id: i32,
    pub m_last_draw_msec: i32,
    pub m_scheduled_draw: bool,
    pub m_scheduled_resize: bool,
    pub m_scheduled_bump: i32,
}
impl ZapGl {
    pub fn new() -> Self {
        Self {
            m_init_width: 0,
            m_init_height: 0,
            m_init_left: 0,
            m_init_top: 0,
            m_first_draw: 3,
            m_mouse_pressed: false,
            m_mouse_in_window: -1,
            m_num_first_draws: 3,
            m_timer_id: 0,
            m_last_draw_msec: 0,
            m_scheduled_draw: false,
            m_scheduled_resize: false,
            m_scheduled_bump: 2,
        }
    }
    pub fn destroy(&mut self) {}
    pub fn set_buffer_swap_auto(&mut self, _: bool) {}
    pub fn update_gl(&mut self, n: &mut dyn ZapNativeBoundary) {
        self.paint_gl(n)
    }
    pub fn swap_buffers(&mut self) {}
    pub fn extra_cursor_in_window(&self) -> bool {
        self.m_mouse_pressed || self.m_mouse_in_window != 0
    }
    pub fn paint_gl(&mut self, n: &mut dyn ZapNativeBoundary) {
        if self.m_first_draw >= self.m_num_first_draws {
            self.m_timer_id = 10;
            self.m_first_draw -= 1
        };
        if n.imod_loop_started() && (self.m_init_width != 0 || self.m_init_height != 0) {
            n.paint()
        }
    }
    pub fn timer_event(&mut self, n: &mut dyn ZapNativeBoundary) {
        if !n.imod_loop_started() || (self.m_init_width == 0 && self.m_init_height == 0) {
            return;
        };
        if self.m_scheduled_resize {
            self.cancel_resize();
            return;
        };
        let start = n.wall_time_msec();
        if self.m_first_draw < 2 && !self.m_scheduled_draw {
            self.cancel_redraw()
        };
        n.update_gl();
        self.m_last_draw_msec = n.wall_time_msec() - start;
        self.m_first_draw = (self.m_first_draw - 1).max(0);
        n.process_events()
    }
    pub fn schedule_redraw(&mut self, interval: i32) {
        self.m_timer_id = interval;
        self.m_scheduled_draw = true
    }
    pub fn cancel_redraw(&mut self) {
        self.m_timer_id = 0;
        self.m_scheduled_draw = false
    }
    pub fn schedule_resize(&mut self, interval: i32) {
        if self.m_first_draw <= 0 && self.m_timer_id == 0 {
            self.m_timer_id = interval
        };
        self.m_scheduled_resize = true
    }
    pub fn cancel_resize(&mut self) {
        self.m_timer_id = 0;
        self.m_scheduled_resize = false
    }
    pub fn get_last_draw_msec(&self) -> i32 {
        self.m_last_draw_msec
    }
    pub fn resize_gl(&mut self, width: i32, height: i32, n: &mut dyn ZapNativeBoundary) {
        n.resize(width, height)
    }
    pub fn mouse_press_event(&mut self, event: i32, n: &mut dyn ZapNativeBoundary) {
        self.m_mouse_pressed = true;
        n.mouse_press(event)
    }
    pub fn mouse_release_event(&mut self, event: i32, n: &mut dyn ZapNativeBoundary) {
        self.m_mouse_pressed = false;
        n.mouse_release(event)
    }
    pub fn mouse_move_event(&mut self, event: i32, n: &mut dyn ZapNativeBoundary) {
        self.m_mouse_in_window = 1;
        n.mouse_move(event)
    }
    pub fn enter_event(&mut self, event: i32, n: &mut dyn ZapNativeBoundary) {
        self.m_mouse_in_window = 1;
        n.general_event(event)
    }
    pub fn leave_event(&mut self, event: i32, n: &mut dyn ZapNativeBoundary) {
        self.m_mouse_in_window = 0;
        n.general_event(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        calls: Vec<String>,
        x: i32,
        y: i32,
        z: i32,
        widths: Vec<(String, i32)>,
    }
    impl ZapNativeBoundary for N {
        fn step_zoom(&mut self, s: i32) {
            self.calls.push(format!("zoom{s}"))
        }
        fn entered_zoom(&mut self, z: f32) {
            self.calls.push(format!("entered{z}"))
        }
        fn entered_section(&mut self, s: i32) {
            self.calls.push(format!("sec{s}"))
        }
        fn help(&mut self) {}
        fn print_info(&mut self) {}
        fn step_time(&mut self, _: i32) {}
        fn state_toggled(&mut self, i: usize, s: i32) {
            self.calls.push(format!("toggle{i}:{s}"))
        }
        fn setup_panels(&mut self) -> i32 {
            0
        }
        fn update_gl(&mut self) {
            self.calls.push("gl".into())
        }
        fn key_input(&mut self, key: i32, modifiers: i32) {
            self.calls.push(format!("key{key}:{modifiers}"))
        }
        fn key_release(&mut self, _: i32, _: i32) {}
        fn general_event(&mut self, _: i32) {}
        fn mouse_press(&mut self, _: i32) {}
        fn mouse_release(&mut self, _: i32) {}
        fn mouse_move(&mut self, _: i32) {}
        fn paint(&mut self) {}
        fn resize(&mut self, _: i32, _: i32) {}
        fn closing(&mut self) {}
        fn imod_loop_started(&self) -> bool {
            true
        }
        fn hot_slider_active(&self, _: bool) -> bool {
            false
        }
        fn hot_slider_key(&self, _: i32) -> bool {
            false
        }
        fn tilt_angles(&self) -> Option<Vec<f32>> {
            None
        }
        fn zsize(&self) -> i32 {
            20
        }
        fn current_section(&self) -> i32 {
            4
        }
        fn time(&self) -> i32 {
            1
        }
        fn num_xpanels(&self) -> i32 {
            self.x
        }
        fn num_ypanels(&self) -> i32 {
            self.y
        }
        fn panel_zstep(&self) -> i32 {
            self.z
        }
        fn draw_in_center(&self) -> i32 {
            0
        }
        fn draw_in_others(&self) -> i32 {
            0
        }
        fn set_num_xpanels(&mut self, v: i32) {
            self.x = v
        }
        fn set_num_ypanels(&mut self, v: i32) {
            self.y = v
        }
        fn set_panel_zstep(&mut self, v: i32) {
            self.z = v
        }
        fn set_draw_in_center(&mut self, _: i32) {}
        fn set_draw_in_others(&mut self, _: i32) {}
        fn wall_time_msec(&self) -> i32 {
            7
        }
        fn process_events(&mut self) {}
        fn rounded_style(&self) -> bool {
            true
        }
        fn info_button_exists(&self) -> bool {
            true
        }
        fn low_section_button_exists(&self) -> bool {
            true
        }
        fn high_section_button_exists(&self) -> bool {
            true
        }
        fn button_width(&self, _: bool, _: f32, text: &str) -> i32 {
            text.len() as i32
        }
        fn set_button_fixed_width(&mut self, button: &str, width: i32) {
            self.widths.push((button.into(), width));
        }
        fn widget_change_event(&mut self) {
            self.calls.push("change".into());
        }
        fn check_and_set_mac_menu(&mut self) {
            self.calls.push("menu".into());
        }
        fn screen_changed(&mut self, dpr: f32) {
            self.calls.push(format!("screen{dpr}"));
        }
    }
    #[test]
    fn window_slots_forward_and_preserve_fields() {
        let mut n = N::default();
        let mut w = ZapWindow::new("T", true, &mut n);
        w.set_zoom_text(2.5);
        w.new_zoom(&mut n);
        w.columns_changed(3, &mut n);
        w.toggle_clicked(1, true, &mut n);
        assert_eq!(w.m_sec_slider_max, 20);
        assert_eq!(n.x, 3);
        assert!(n.calls.contains(&"toggle1:1".into()));
    }
    #[test]
    fn font_change_uses_source_optional_button_widths() {
        let mut native = N::default();
        let mut window = ZapWindow::new("", false, &mut native);
        assert_eq!(
            native.widths,
            [
                ("mInfoButton".into(), 11),
                ("mHelpButton".into(), 4),
                ("mLowSectionButton".into(), 10),
                ("mHighSectionButton".into(), 10),
            ]
        );
        window.change_event(true, &mut native);
        assert_eq!(native.widths.len(), 8);
        assert!(native.calls.ends_with(&["change".into(), "menu".into()]));
    }
    #[test]
    fn screen_and_toolbar_events_follow_source_routes() {
        let mut native = N::default();
        let mut window = ZapWindow::new("", false, &mut native);
        window.screen_changed(1.5, &mut native);
        // Row 8 is Ctrl+A in `zap_classes.cpp`'s `sPopupTable`.
        window.toolbar_menu_event(7, &mut native);
        // The first shared default row follows the eleven source-specific rows.
        window.toolbar_menu_event(11, &mut native);
        assert!(native.calls.contains(&"screen1.5".into()));
        assert!(
            native
                .calls
                .contains(&format!("key{}:{}", 'A' as i32, INPUT_CTRL))
        );
        assert!(native.calls.contains(&format!("key{}:0", 'O' as i32)));
    }
    #[test]
    fn gl_mouse_and_timers_follow_source_state() {
        let mut n = N::default();
        let mut g = ZapGl::new();
        g.m_init_width = 1;
        g.mouse_press_event(1, &mut n);
        g.mouse_release_event(2, &mut n);
        g.schedule_redraw(12);
        g.timer_event(&mut n);
        assert!(!g.m_mouse_pressed);
        assert_eq!(g.m_last_draw_msec, 0);
    }
}
