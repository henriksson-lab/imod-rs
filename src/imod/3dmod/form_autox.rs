//! Translation of `IMOD/3dmod/form_autox.cpp` together with `form_autox.h`.
//!
//! `AutoxWindow` is a Qt Designer child of a `DockingDialog`. Widget,
//! `MultiSlider`, and docking event wiring are explicit native GUI operations;
//! no in-memory dialog is substituted for the original form.
#![allow(dead_code)]

use crate::imod::three_dmod::mv_window::{Key, KeyEvent};
use core::ffi::c_void;

pub const AUTOX_MAX_RESOLUTION: i32 = 200;

/// `QEvent` types inspected by `AutoxWindow::topChangeEvent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoxChangeEvent {
    FontChange,
    Other,
}

/// Direct Qt and `autox.cpp` calls performed by this source unit. Implementors
/// own real widgets/signals and paired image-view state.
pub trait AutoxNativeBoundary {
    fn setup_ui(&mut self);
    fn set_delete_on_close(&mut self);
    fn set_always_show_tool_tips(&mut self);
    fn connect_autox_signals(&mut self);
    fn create_multi_slider(
        &mut self,
        number: i32,
        labels: [&str; 2],
        minimum: i32,
        maximum: i32,
    ) -> *mut c_void;
    fn multi_slider_set_range(&mut self, slider: i32, minimum: i32, maximum: i32);
    fn multi_slider_set_decimals(&mut self, slider: i32, decimals: i32);
    fn multi_slider_set_tool_tip(&mut self, slider: i32, text: &str);
    fn create_contrast_group(&mut self) -> *mut c_void;
    fn contrast_group_add_button(&mut self, group: *mut c_void, which: i32);
    fn set_button_width(&mut self, label: &str, rounded_style: bool, factor: f64) -> i32;
    fn set_fixed_width(&mut self, which: i32, width: i32);
    fn rounded_style(&self) -> bool;
    fn set_contrast_group(&mut self, group: *mut c_void, contrast: i32);
    fn multi_slider_set_value(&mut self, slider: i32, value: i32);
    fn set_alt_mouse_checked(&mut self, checked: bool);
    fn set_diagonals_checked(&mut self, checked: bool);
    fn close_top_window(&mut self);
    fn accept_close_event(&mut self);
    fn grab_keyboard(&mut self);
    fn release_keyboard(&mut self);
    fn retranslate_ui(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn widget_change_event(&mut self);
    fn hot_slider_active(&self, control_pressed: bool) -> bool;
    fn hot_slider_enabled(&self) -> bool;
    fn hot_slider_key(&self, key: Key) -> bool;
    fn close_key(&self, event: KeyEvent) -> bool;
    fn autox_slider(&mut self, which: i32, value: i32);
    fn autox_contrast_selected(&mut self, which: i32);
    fn autox_altmouse(&mut self, state: i32);
    fn autox_follow_diagonals(&mut self, state: i32);
    fn autox_build(&mut self);
    fn autox_clear(&mut self);
    fn autox_fill(&mut self);
    fn autox_next(&mut self);
    fn autox_expand(&mut self);
    fn autox_shrink(&mut self);
    fn autox_smooth(&mut self);
    fn autox_closing(&mut self);
    fn ivw_control_key(&mut self, release: i32, event: KeyEvent);
}

/// `AutoxWindow` (`form_autox.h`), preserving the C++ field ownership.
#[repr(C)]
#[derive(Debug)]
pub struct AutoxWindow {
    pub m_top_win: *mut c_void,
    pub m_ctrl_pressed: bool,
    pub m_sliders: *mut c_void,
    pub contrast_group: *mut c_void,
}

impl AutoxWindow {
    /// `AutoxWindow::AutoxWindow`.
    pub fn new(m_top_win: *mut c_void, native: &mut dyn AutoxNativeBoundary) -> Self {
        native.setup_ui();
        let mut window = Self {
            m_top_win,
            m_ctrl_pressed: false,
            m_sliders: core::ptr::null_mut(),
            contrast_group: core::ptr::null_mut(),
        };
        window.init(native);
        window
    }
    /// `AutoxWindow::~AutoxWindow`.
    pub fn destroy(&mut self) {}
    /// `AutoxWindow::languageChange`.
    pub fn language_change(&mut self, native: &mut dyn AutoxNativeBoundary) {
        native.retranslate_ui();
    }
    /// `AutoxWindow::init`.
    pub fn init(&mut self, native: &mut dyn AutoxNativeBoundary) {
        self.m_ctrl_pressed = false;
        native.set_delete_on_close();
        native.set_always_show_tool_tips();
        native.connect_autox_signals();
        self.m_sliders = native.create_multi_slider(2, ["Threshold", "Resolution"], 0, 254);
        native.multi_slider_set_range(1, 0, AUTOX_MAX_RESOLUTION);
        native.multi_slider_set_decimals(1, 2);
        native.multi_slider_set_tool_tip(0, "Set threshold for high-contrast viewing");
        native.multi_slider_set_tool_tip(1, "Set resolution in pixels for making contours");
        self.contrast_group = native.create_contrast_group();
        native.contrast_group_add_button(self.contrast_group, 0);
        native.contrast_group_add_button(self.contrast_group, 1);
        self.set_font_dependent_widths(native);
    }
    /// `AutoxWindow::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self, native: &mut dyn AutoxNativeBoundary) {
        let mut width = native.set_button_width("Clear", native.rounded_style(), 1.2);
        native.set_fixed_width(0, width);
        native.set_fixed_width(1, width);
        native.set_fixed_width(2, width);
        width = native.set_button_width("Expand", native.rounded_style(), 1.2);
        native.set_fixed_width(3, width);
        native.set_fixed_width(4, width);
        native.set_fixed_width(5, width);
    }
    /// `AutoxWindow::contrastSelected`.
    pub fn contrast_selected(&mut self, which: i32, native: &mut dyn AutoxNativeBoundary) {
        native.autox_contrast_selected(which);
    }
    /// `AutoxWindow::sliderChanged`.
    pub fn slider_changed(
        &mut self,
        which: i32,
        value: i32,
        dragging: bool,
        native: &mut dyn AutoxNativeBoundary,
    ) {
        if which != 0 || !dragging || native.hot_slider_active(self.m_ctrl_pressed) {
            native.autox_slider(which, value);
        }
    }
    /// `AutoxWindow::altMouse`.
    pub fn alt_mouse(&mut self, state: bool, native: &mut dyn AutoxNativeBoundary) {
        native.autox_altmouse(state as i32);
    }
    /// `AutoxWindow::followDiagonals`.
    pub fn follow_diagonals(&mut self, state: bool, native: &mut dyn AutoxNativeBoundary) {
        native.autox_follow_diagonals(state as i32);
    }
    /// `AutoxWindow::buildPressed`.
    pub fn build_pressed(&mut self, native: &mut dyn AutoxNativeBoundary) {
        native.autox_build();
    }
    /// `AutoxWindow::clearPressed`.
    pub fn clear_pressed(&mut self, native: &mut dyn AutoxNativeBoundary) {
        native.autox_clear();
    }
    /// `AutoxWindow::fillPressed`.
    pub fn fill_pressed(&mut self, native: &mut dyn AutoxNativeBoundary) {
        native.autox_fill();
    }
    /// `AutoxWindow::nextPressed`.
    pub fn next_pressed(&mut self, native: &mut dyn AutoxNativeBoundary) {
        native.autox_next();
    }
    /// `AutoxWindow::expandPressed`.
    pub fn expand_pressed(&mut self, native: &mut dyn AutoxNativeBoundary) {
        native.autox_expand();
    }
    /// `AutoxWindow::shrinkPressed`.
    pub fn shrink_pressed(&mut self, native: &mut dyn AutoxNativeBoundary) {
        native.autox_shrink();
    }
    /// `AutoxWindow::smoothPressed`.
    pub fn smooth_pressed(&mut self, native: &mut dyn AutoxNativeBoundary) {
        native.autox_smooth();
    }
    /// `AutoxWindow::setStates`.
    pub fn set_states(
        &mut self,
        contrast: i32,
        threshold: i32,
        resolution: i32,
        alt_mouse: i32,
        follow: i32,
        native: &mut dyn AutoxNativeBoundary,
    ) {
        native.set_contrast_group(self.contrast_group, contrast);
        native.multi_slider_set_value(0, threshold);
        native.multi_slider_set_value(1, resolution);
        native.set_alt_mouse_checked(alt_mouse != 0);
        native.set_diagonals_checked(follow != 0);
    }
    /// `AutoxWindow::topCloseEvent`.
    pub fn top_close_event(&mut self, native: &mut dyn AutoxNativeBoundary) {
        native.autox_closing();
        native.accept_close_event();
    }
    /// `AutoxWindow::keyPressEvent`.
    pub fn key_press_event(&mut self, event: KeyEvent, native: &mut dyn AutoxNativeBoundary) {
        if native.close_key(event) {
            native.close_top_window();
        } else {
            if native.hot_slider_enabled() && native.hot_slider_key(event.key) {
                self.m_ctrl_pressed = true;
                native.grab_keyboard();
            }
            native.ivw_control_key(0, event);
        }
    }
    /// `AutoxWindow::keyReleaseEvent`.
    pub fn key_release_event(&mut self, event: KeyEvent, native: &mut dyn AutoxNativeBoundary) {
        if native.hot_slider_key(event.key) {
            self.m_ctrl_pressed = false;
            native.release_keyboard();
        }
        native.ivw_control_key(1, event);
    }
    /// `AutoxWindow::topChangeEvent`.
    pub fn top_change_event(
        &mut self,
        event: AutoxChangeEvent,
        native: &mut dyn AutoxNativeBoundary,
    ) {
        native.widget_change_event();
        native.check_and_set_mac_menu();
        if event == AutoxChangeEvent::FontChange {
            self.set_font_dependent_widths(native);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native {
        calls: Vec<String>,
        hot: bool,
        marker: u8,
    }
    impl AutoxNativeBoundary for Native {
        fn setup_ui(&mut self) {
            self.calls.push("setup".into())
        }
        fn set_delete_on_close(&mut self) {
            self.calls.push("delete".into())
        }
        fn set_always_show_tool_tips(&mut self) {
            self.calls.push("tips".into())
        }
        fn connect_autox_signals(&mut self) {
            self.calls.push("connect".into())
        }
        fn create_multi_slider(&mut self, _: i32, _: [&str; 2], _: i32, _: i32) -> *mut c_void {
            &mut self.marker as *mut u8 as *mut c_void
        }
        fn multi_slider_set_range(&mut self, a: i32, b: i32, c: i32) {
            self.calls.push(format!("range:{a}:{b}:{c}"))
        }
        fn multi_slider_set_decimals(&mut self, a: i32, b: i32) {
            self.calls.push(format!("decimal:{a}:{b}"))
        }
        fn multi_slider_set_tool_tip(&mut self, a: i32, _: &str) {
            self.calls.push(format!("tip:{a}"))
        }
        fn create_contrast_group(&mut self) -> *mut c_void {
            &mut self.marker as *mut u8 as *mut c_void
        }
        fn contrast_group_add_button(&mut self, _: *mut c_void, a: i32) {
            self.calls.push(format!("button:{a}"))
        }
        fn set_button_width(&mut self, a: &str, _: bool, _: f64) -> i32 {
            if a == "Clear" { 40 } else { 55 }
        }
        fn set_fixed_width(&mut self, a: i32, b: i32) {
            self.calls.push(format!("width:{a}:{b}"))
        }
        fn rounded_style(&self) -> bool {
            false
        }
        fn set_contrast_group(&mut self, _: *mut c_void, a: i32) {
            self.calls.push(format!("contrast:{a}"))
        }
        fn multi_slider_set_value(&mut self, a: i32, b: i32) {
            self.calls.push(format!("value:{a}:{b}"))
        }
        fn set_alt_mouse_checked(&mut self, a: bool) {
            self.calls.push(format!("alt:{a}"))
        }
        fn set_diagonals_checked(&mut self, a: bool) {
            self.calls.push(format!("diag:{a}"))
        }
        fn close_top_window(&mut self) {
            self.calls.push("close".into())
        }
        fn accept_close_event(&mut self) {
            self.calls.push("accept".into())
        }
        fn grab_keyboard(&mut self) {
            self.calls.push("grab".into())
        }
        fn release_keyboard(&mut self) {
            self.calls.push("release".into())
        }
        fn retranslate_ui(&mut self) {
            self.calls.push("translate".into())
        }
        fn check_and_set_mac_menu(&mut self) {
            self.calls.push("menu".into())
        }
        fn widget_change_event(&mut self) {
            self.calls.push("change".into())
        }
        fn hot_slider_active(&self, _: bool) -> bool {
            self.hot
        }
        fn hot_slider_enabled(&self) -> bool {
            true
        }
        fn hot_slider_key(&self, key: Key) -> bool {
            key == Key::Character('C')
        }
        fn close_key(&self, event: KeyEvent) -> bool {
            event.key == Key::Character('Q')
        }
        fn autox_slider(&mut self, a: i32, b: i32) {
            self.calls.push(format!("slider:{a}:{b}"))
        }
        fn autox_contrast_selected(&mut self, a: i32) {
            self.calls.push(format!("select:{a}"))
        }
        fn autox_altmouse(&mut self, a: i32) {
            self.calls.push(format!("altmouse:{a}"))
        }
        fn autox_follow_diagonals(&mut self, a: i32) {
            self.calls.push(format!("follow:{a}"))
        }
        fn autox_build(&mut self) {
            self.calls.push("build".into())
        }
        fn autox_clear(&mut self) {
            self.calls.push("clear".into())
        }
        fn autox_fill(&mut self) {
            self.calls.push("fill".into())
        }
        fn autox_next(&mut self) {
            self.calls.push("next".into())
        }
        fn autox_expand(&mut self) {
            self.calls.push("expand".into())
        }
        fn autox_shrink(&mut self) {
            self.calls.push("shrink".into())
        }
        fn autox_smooth(&mut self) {
            self.calls.push("smooth".into())
        }
        fn autox_closing(&mut self) {
            self.calls.push("closing".into())
        }
        fn ivw_control_key(&mut self, a: i32, _: KeyEvent) {
            self.calls.push(format!("key:{a}"))
        }
    }
    #[test]
    fn threshold_drag_is_deferred_but_resolution_is_not() {
        let mut n = Native::default();
        let mut f = AutoxWindow::new(core::ptr::null_mut(), &mut n);
        f.slider_changed(0, 12, true, &mut n);
        f.slider_changed(1, 34, true, &mut n);
        assert!(!n.calls.iter().any(|x| x == "slider:0:12"));
        assert!(n.calls.iter().any(|x| x == "slider:1:34"));
    }
    #[test]
    fn hot_slider_forwards_threshold_drag() {
        let mut n = Native {
            hot: true,
            ..Default::default()
        };
        let mut f = AutoxWindow::new(core::ptr::null_mut(), &mut n);
        f.key_press_event(
            KeyEvent {
                key: Key::Character('C'),
                ..KeyEvent::default()
            },
            &mut n,
        );
        f.slider_changed(0, 12, true, &mut n);
        assert!(n.calls.iter().any(|x| x == "grab"));
        assert!(n.calls.iter().any(|x| x == "slider:0:12"));
    }
    #[test]
    fn set_states_keeps_source_order() {
        let mut n = Native::default();
        let mut f = AutoxWindow::new(core::ptr::null_mut(), &mut n);
        f.set_states(1, 100, 50, 1, 0, &mut n);
        assert_eq!(
            &n.calls[n.calls.len() - 5..],
            [
                "contrast:1",
                "value:0:100",
                "value:1:50",
                "alt:true",
                "diag:false"
            ]
        );
    }

    #[test]
    fn change_event_calls_base_then_mac_menu_and_font_width_path() {
        let mut n = Native::default();
        let mut f = AutoxWindow::new(core::ptr::null_mut(), &mut n);
        f.top_change_event(AutoxChangeEvent::FontChange, &mut n);
        assert!(n.calls.windows(2).any(|calls| calls == ["change", "menu"]));
        assert!(n.calls.iter().any(|call| call == "width:0:40"));
    }
}
