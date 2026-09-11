//! Translation of `IMOD/3dmod/slicer_classes.cpp` and `slicer_classes.h`.
//!
//! Qt widget construction, timer ownership, and GL context dispatch are represented by
//! `SlicerNativeBoundary`.  Calls into `sslice.cpp` remain the `SlicerCore` boundary;
//! this module retains the source window/widget state and every source event method.
#![allow(dead_code)]

pub const MAX_SLICER_TOGGLES: usize = 9;
pub const SLICER_TOGGLE_HIGHRES: usize = 0;
pub const SLICER_TOGGLE_LOCK: usize = 1;
pub const SLICER_TOGGLE_CENTER: usize = 2;
pub const SLICER_TOGGLE_BAND: usize = 3;
pub const SLICER_TOGGLE_ARROW: usize = 4;
pub const SLICER_TOGGLE_FFT: usize = 5;
pub const SLICER_TOGGLE_ZSCALE: usize = 6;
pub const SLICER_TOGGLE_TIMELOCK: usize = 7;
pub const SLICER_TOGGLE_SHIFTLOCK: usize = 8;
pub const SLICER_LIMIT_INVALID: i32 = 0;
pub const SLICER_LIMIT_TRUNCATE: i32 = 1;
pub const SLICER_LIMIT_VALID: i32 = 2;

pub const FILE_LIST: [[&str; 2]; MAX_SLICER_TOGGLES] = [
    [":/images/lowres.png", ":/images/highres.png"],
    [":/images/unlock.png", ":/images/lock.png"],
    [":/images/smartCenter.png", ":/images/keepCenter.png"],
    [":/images/rubberband.png", ":/images/rubberband2.png"],
    [":/images/arrowBlack.png", ":/images/arrowRed.png"],
    [":/images/fft.png", ":/images/fftRed.png"],
    [":/images/zscale.png", ":/images/zscaleOn.png"],
    [":/images/timeUnlock.png", ":/images/timeLock.png"],
    [":/images/shiftlockoff.png", ":/images/shiftlockon.png"],
];
pub const S_TOGGLE_TIPS: [&str; MAX_SLICER_TOGGLES] = [
    "Toggle between regular and high-resolution (interpolated) image",
    "Lock window at current position",
    "Keep current image or model point centered (classic mode, hot key K)",
    "Toggle rubberband on or off (resize with first mouse, move with second, hot key Shift+B)",
    "Toggle arrow on or off (draw with first mouse)",
    "Toggle between showing image and FFT, hot key Shift+S",
    "Toggle applying model Z-scale to image",
    "Lock window at current time",
    "Use keypad and mouse as if Shift key were down to rotate slice",
];
pub const S_SLIDER_LABELS: [&str; 4] = [
    "X rotation",
    "Y rotation",
    "Z rotation",
    "View axis position",
];
/// `PopupEntry` values in `sPopupTable`; the source `mainIndex` is zero for
/// every one of these entries and is therefore intentionally not stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlicerPopupEntry {
    pub text: &'static str,
    pub key: i32,
    pub ctrl: i32,
    pub shift: i32,
}
pub const S_POPUP_TABLE: [SlicerPopupEntry; 21] = [
    SlicerPopupEntry {
        text: "Set Angles Based on Model Points",
        key: -1,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Align current and previous points along X axis",
        key: 88,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Align current and previous points along Y axis",
        key: 89,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Align current and previous points along Z axis",
        key: 90,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Align first and last points along X axis",
        key: 88,
        ctrl: 0,
        shift: 1,
    },
    SlicerPopupEntry {
        text: "Align first and last points along Y axis",
        key: 89,
        ctrl: 0,
        shift: 1,
    },
    SlicerPopupEntry {
        text: "Align first and last points along Z axis",
        key: 90,
        ctrl: 0,
        shift: 1,
    },
    SlicerPopupEntry {
        text: "Make current contour flat in slice",
        key: 87,
        ctrl: 0,
        shift: 1,
    },
    SlicerPopupEntry {
        text: "",
        key: -2,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Toggle keeping slice centered on current point",
        key: 75,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Toggle the rubber band on or off",
        key: 66,
        ctrl: 0,
        shift: 1,
    },
    SlicerPopupEntry {
        text: "Toggle FFT mode on or off",
        key: 70,
        ctrl: 0,
        shift: 1,
    },
    SlicerPopupEntry {
        text: "Show slice lines in ZaP and XYZ windows",
        key: 76,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Decrease displayed image thickness",
        key: 95,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Increase displayed image thickness",
        key: 43,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Decrease displayed model thickness",
        key: 57,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Increase displayed model thickness",
        key: 48,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Report distance from current point to cursor",
        key: 81,
        ctrl: 0,
        shift: 0,
    },
    SlicerPopupEntry {
        text: "Resize window to rubber band area",
        key: 82,
        ctrl: 0,
        shift: 1,
    },
    SlicerPopupEntry {
        text: "Resize area within rubber band to fit window",
        key: 82,
        ctrl: 1,
        shift: 1,
    },
    SlicerPopupEntry {
        text: "",
        key: 0,
        ctrl: 0,
        shift: 0,
    },
];

/// Platform-neutral equivalent of a Qt event passed to slicer source code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlicerEvent {
    pub key: i32,
    pub modifiers: i32,
    pub x: i32,
    pub y: i32,
}

/// Direct paired `SlicerFuncs`/`sslice.cpp` calls.
pub trait SlicerCore {
    fn help(&mut self);
    fn step_zoom(&mut self, dir: i32);
    fn step_time(&mut self, dir: i32);
    fn entered_zoom(&mut self, zoom: f32);
    fn show_slice(&mut self);
    fn fill_cache(&mut self);
    fn state_toggled(&mut self, index: usize, state: i32);
    fn angle_changed(&mut self, axis: i32, value: i32, dragging: bool);
    fn image_thickness(&mut self, depth: i32);
    fn model_thickness(&mut self, depth: f32);
    fn resize(&mut self, width: i32, height: i32);
    fn cube_resize(&mut self, width: i32, height: i32);
    fn closing(&mut self);
    fn key_input(&mut self, event: SlicerEvent);
    fn key_release(&mut self, event: SlicerEvent);
    fn mouse_press(&mut self, event: SlicerEvent);
    fn mouse_release(&mut self, event: SlicerEvent);
    fn mouse_move(&mut self, event: SlicerEvent);
    fn general_event(&mut self, event: SlicerEvent);
    fn paint(&mut self);
    fn cube_paint(&mut self);
    fn angles_from_contour(&mut self) -> bool;
    fn check_movie_limits(&mut self);
    fn set_current_or_new_row(&mut self, new_row: bool);
    fn set_angles_from_row(&mut self);
    fn rotate_on_view_axis(&mut self, dx: i32, dy: i32, dz: i32);
    fn set_band_low_high_limit(&mut self, which: i32);
    fn set_linked_state(&mut self, state: bool);
    fn view_axis_step_size(&self) -> f32;
    fn add_cube_to_frame(&mut self);
    fn is_closing(&self) -> bool;
    fn auto_link(&self) -> i32;
    fn set_continuous(&mut self, state: bool);
}

/// Native Qt/OpenGL operations owned by this paired source unit.
pub trait SlicerNativeBoundary {
    fn construct_slicer_window(
        &mut self,
        max_angles: [f32; 3],
        time_label: &str,
        rgba: bool,
        double_buffer: bool,
        enable_depth: bool,
    );
    fn build_toolbar2(&mut self, free_bar: bool, max_angles: [f32; 3], step_size: f32);
    fn set_font_dependent_widths(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn toolbar_size_hints(&self) -> (i32, i32, i32);
    fn set_save_angle_toolbar_visible(&mut self, visible: bool, insert_break: Option<bool>);
    fn destroy_embedded_toolbar2(&mut self);
    fn destroy_free_toolbar2(&mut self);
    fn show_toolbar2(&mut self);
    fn defer_free_toolbar2_show(&mut self);
    fn close_linked_slicers(&mut self);
    fn slicer_view_axis_step_change(&mut self, delta: i32);
    fn set_slider_value(&mut self, which: i32, value: i32);
    fn set_slider_min_max_value(&mut self, which: i32, min: i32, max: i32, value: i32);
    fn set_model_thickness(&mut self, value: f32);
    fn set_image_thickness(&mut self, value: i32);
    fn set_toggle_checked(&mut self, index: usize, checked: bool);
    fn set_rotation_center_state(&mut self, checked: bool);
    fn set_zoom_text(&mut self, text: &str);
    fn zoom_text(&self) -> String;
    fn focus_window(&mut self);
    fn set_time_label(&mut self, number: &str, label: &str);
    fn set_low_high_color(&mut self, which: i32, state: i32);
    fn set_low_high_visible(&mut self, which: i32, visible: bool);
    fn set_band_size(&mut self, text: &str, visible: bool);
    fn ignore_closing(&self, already_closing: bool) -> bool;
    fn accept_close(&mut self);
    fn delete_slicer_funcs(&mut self);
    fn popup_key(&mut self, index: i32) -> Option<SlicerEvent>;
    fn imod_loop_started(&self) -> bool;
    fn start_timer(&mut self, interval: i32) -> i32;
    fn kill_timer(&mut self, id: i32);
    fn update_gl(&mut self);
    fn window_size(&self) -> (i32, i32);
    fn resize_window(&mut self, width: i32, height: i32);
    fn cube_exists(&self) -> bool;
    fn mouse_left_button_only(&self) -> bool;
    fn is_windows(&self) -> bool;
}

/// `SlicerWindow` (`slicer_classes.h`).
#[derive(Clone, Debug)]
pub struct SlicerWindow {
    pub m_toggle_states: [i32; MAX_SLICER_TOGGLES],
    pub m_low_high_states: [i32; 2],
    pub m_break_before_ang_bar: i32,
    pub m_time_bar: bool,
    pub m_tool_bar2: bool,
    pub m_free_bar2: bool,
    pub m_cube: bool,
    pub s_max_angles: [f32; 3],
}
impl SlicerWindow {
    /// `SlicerWindow::SlicerWindow`.
    pub fn new(
        core: &dyn SlicerCore,
        max_angles: [f32; 3],
        time_label: &str,
        rgba: bool,
        double_buffer: bool,
        enable_depth: bool,
        n: &mut dyn SlicerNativeBoundary,
    ) -> Self {
        n.construct_slicer_window(max_angles, time_label, rgba, double_buffer, enable_depth);
        let mut window = Self {
            m_toggle_states: [0; MAX_SLICER_TOGGLES],
            m_low_high_states: [SLICER_LIMIT_INVALID - 1; 2],
            m_break_before_ang_bar: 0,
            m_time_bar: !time_label.is_empty(),
            m_tool_bar2: true,
            m_free_bar2: false,
            m_cube: false,
            s_max_angles: max_angles,
        };
        window.build_tool_bar2(false, core, n);
        window.set_font_dependent_widths(n);
        window
    }
    /// `SlicerWindow::~SlicerWindow`.
    pub fn destroy(&mut self) {}
    /// `SlicerWindow::buildToolBar2`.
    pub fn build_tool_bar2(
        &mut self,
        free_bar: bool,
        core: &dyn SlicerCore,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        n.build_toolbar2(free_bar, self.s_max_angles, core.view_axis_step_size());
        self.m_tool_bar2 = !free_bar;
        self.m_free_bar2 = free_bar;
    }
    /// `SlicerWindow::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self, n: &mut dyn SlicerNativeBoundary) {
        n.set_font_dependent_widths()
    }
    /// `SlicerWindow::changeEvent`.
    pub fn change_event(&mut self, font_change: bool, n: &mut dyn SlicerNativeBoundary) {
        n.check_and_set_mac_menu();
        if font_change {
            self.set_font_dependent_widths(n)
        }
    }
    /// `SlicerWindow::showSaveAngleToolbar`.
    pub fn show_save_angle_toolbar(&mut self, n: &mut dyn SlicerNativeBoundary) {
        let (angle, before, width) = n.toolbar_size_hints();
        let need_break = i32::from(angle + before >= width);
        n.set_save_angle_toolbar_visible(
            true,
            (need_break != self.m_break_before_ang_bar).then_some(need_break != 0),
        );
        self.m_break_before_ang_bar = need_break;
    }
    /// `SlicerWindow::manageAutoLink`.
    pub fn manage_auto_link(
        &mut self,
        new_state: i32,
        core: &mut dyn SlicerCore,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        if new_state == 1 {
            if self.m_tool_bar2 {
                n.destroy_embedded_toolbar2()
            }
            self.m_tool_bar2 = false;
            self.build_tool_bar2(true, core, n);
            self.set_font_dependent_widths(n);
            if self.m_cube {
                n.start_timer(200);
            }
            self.m_cube = false;
        } else if core.auto_link() == 1 {
            n.destroy_free_toolbar2();
            self.m_free_bar2 = false;
            if !core.is_closing() {
                self.build_tool_bar2(false, core, n);
                self.set_font_dependent_widths(n);
                core.add_cube_to_frame();
                self.m_cube = true
            }
        } else if self.m_tool_bar2 {
            n.show_toolbar2()
        }
    }
    /// `SlicerWindow::freeBarClose`.
    pub fn free_bar_close(&mut self, n: &mut dyn SlicerNativeBoundary) {
        // `imodDialogManager.windowList` and each Qt `close()` belong to Qt's
        // window manager; the boundary performs the exact controller/slave walk.
        // Kept as the paired source operation rather than synthesizing a window list.
        n.close_linked_slicers();
    }
    /// `SlicerWindow::zoomUp`.
    pub fn zoom_up(&mut self, core: &mut dyn SlicerCore) {
        core.step_zoom(1)
    }
    /// `SlicerWindow::zoomDown`.
    pub fn zoom_down(&mut self, core: &mut dyn SlicerCore) {
        core.step_zoom(-1)
    }
    /// `SlicerWindow::timeBack`.
    pub fn time_back(&mut self, core: &mut dyn SlicerCore) {
        core.step_time(-1)
    }
    /// `SlicerWindow::timeForward`.
    pub fn time_forward(&mut self, core: &mut dyn SlicerCore) {
        core.step_time(1)
    }
    /// `SlicerWindow::newZoom`.
    pub fn new_zoom(&mut self, core: &mut dyn SlicerCore, n: &mut dyn SlicerNativeBoundary) {
        core.entered_zoom(n.zoom_text().parse().unwrap_or(0.));
        n.focus_window()
    }
    /// `SlicerWindow::rotationClicked`.
    pub fn rotation_clicked(&mut self, dx: i32, dy: i32, dz: i32, core: &mut dyn SlicerCore) {
        core.rotate_on_view_axis(dx, dy, dz)
    }
    /// `SlicerWindow::stepSizeChanged`.
    pub fn step_size_changed(&mut self, delta: i32, n: &mut dyn SlicerNativeBoundary) {
        n.slicer_view_axis_step_change(delta);
    }
    /// `SlicerWindow::shiftToggled`.
    pub fn shift_toggled(&mut self, state: bool, core: &mut dyn SlicerCore) {
        core.state_toggled(SLICER_TOGGLE_SHIFTLOCK, i32::from(state))
    }
    /// `SlicerWindow::imageThicknessChanged`.
    pub fn image_thickness_changed(
        &mut self,
        depth: i32,
        core: &mut dyn SlicerCore,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        core.image_thickness(depth);
        n.focus_window()
    }
    /// `SlicerWindow::modelThicknessChanged`.
    pub fn model_thickness_changed(
        &mut self,
        depth: f64,
        core: &mut dyn SlicerCore,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        core.model_thickness(depth as f32);
        n.focus_window()
    }
    /// `SlicerWindow::help`.
    pub fn help(&mut self, core: &mut dyn SlicerCore) {
        core.help()
    }
    /// `SlicerWindow::angleChanged`.
    pub fn angle_changed(
        &mut self,
        which: i32,
        value: i32,
        dragging: bool,
        core: &mut dyn SlicerCore,
    ) {
        core.angle_changed(which, value, dragging)
    }
    /// `SlicerWindow::toggleClicked`.
    pub fn toggle_clicked(&mut self, index: usize, state: bool, core: &mut dyn SlicerCore) {
        self.m_toggle_states[index] = i32::from(state);
        core.state_toggled(index, self.m_toggle_states[index])
    }
    /// `SlicerWindow::showslicePressed`.
    pub fn showslice_pressed(&mut self, core: &mut dyn SlicerCore) {
        core.show_slice()
    }
    /// `SlicerWindow::contourPressed`.
    pub fn contour_pressed(&mut self, core: &mut dyn SlicerCore) {
        if !core.angles_from_contour() {
            core.check_movie_limits()
        }
    }
    /// `SlicerWindow::fillCachePressed`.
    pub fn fill_cache_pressed(&mut self, core: &mut dyn SlicerCore) {
        core.fill_cache()
    }
    /// `SlicerWindow::lowHighClicked`.
    pub fn low_high_clicked(&mut self, which: i32, core: &mut dyn SlicerCore) {
        core.set_band_low_high_limit(which)
    }
    /// `SlicerWindow::saveAngClicked`.
    pub fn save_ang_clicked(&mut self, core: &mut dyn SlicerCore) {
        core.set_current_or_new_row(false)
    }
    /// `SlicerWindow::setAngClicked`.
    pub fn set_ang_clicked(&mut self, core: &mut dyn SlicerCore) {
        core.set_angles_from_row()
    }
    /// `SlicerWindow::newRowClicked`.
    pub fn new_row_clicked(&mut self, core: &mut dyn SlicerCore) {
        core.set_current_or_new_row(true)
    }
    /// `SlicerWindow::continuousToggled`.
    pub fn continuous_toggled(&mut self, state: bool, core: &mut dyn SlicerCore) {
        core.set_continuous(state)
    }
    /// `SlicerWindow::linkToggled`.
    pub fn link_toggled(&mut self, state: bool, core: &mut dyn SlicerCore) {
        core.set_linked_state(state)
    }
    /// `SlicerWindow::setAngles`.
    pub fn set_angles(&mut self, angles: [f32; 3], n: &mut dyn SlicerNativeBoundary) {
        for (axis, angle) in angles.into_iter().enumerate() {
            n.set_slider_value(axis as i32, (angle * 10. + 0.5).floor() as i32)
        }
    }
    /// `SlicerWindow::setViewAxisPosition`.
    pub fn set_view_axis_position(
        &mut self,
        amin: i32,
        amax: i32,
        current: i32,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        n.set_slider_min_max_value(3, amin, amax, current)
    }
    /// `SlicerWindow::setModelThickness`.
    pub fn set_model_thickness(&mut self, depth: f32, n: &mut dyn SlicerNativeBoundary) {
        n.set_model_thickness(depth)
    }
    /// `SlicerWindow::setImageThickness`.
    pub fn set_image_thickness(&mut self, depth: i32, n: &mut dyn SlicerNativeBoundary) {
        n.set_image_thickness(depth)
    }
    /// `SlicerWindow::setToggleState`.
    pub fn set_toggle_state(&mut self, index: usize, state: i32, n: &mut dyn SlicerNativeBoundary) {
        self.m_toggle_states[index] = i32::from(state != 0);
        if index == SLICER_TOGGLE_SHIFTLOCK {
            n.set_rotation_center_state(state != 0)
        } else {
            n.set_toggle_checked(index, state != 0)
        }
    }
    /// `SlicerWindow::setZoomText`.
    pub fn set_zoom_text(&mut self, zoom: f32, n: &mut dyn SlicerNativeBoundary) {
        let mut s = format!("{zoom:.4}");
        if s.ends_with("00") {
            s.truncate(s.len() - 2)
        };
        n.set_zoom_text(&s)
    }
    /// `SlicerWindow::setTimeLabel`.
    pub fn set_time_label(&mut self, time: i32, label: &str, n: &mut dyn SlicerNativeBoundary) {
        n.set_time_label(&format!(" ({time:3})"), label)
    }
    /// `SlicerWindow::setLowHighValidity`.
    pub fn set_low_high_validity(
        &mut self,
        which: i32,
        state: i32,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        let state = state.clamp(0, 2);
        if state != self.m_low_high_states[which as usize] {
            self.m_low_high_states[which as usize] = state;
            n.set_low_high_color(which, state)
        }
    }
    /// `SlicerWindow::enableLowHighButtons`.
    pub fn enable_low_high_buttons(&mut self, enable: i32, n: &mut dyn SlicerNativeBoundary) {
        for i in 0..2 {
            self.set_low_high_validity(i, SLICER_LIMIT_INVALID, n);
            n.set_low_high_visible(i, enable != 0)
        }
    }
    /// `SlicerWindow::manageBandSize`.
    pub fn manage_band_size(
        &mut self,
        xsize: i32,
        ysize: i32,
        action: i32,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        n.set_band_size(&format!("{xsize}x{ysize}"), action > 0);
        if action >= 0 {
            n.set_low_high_visible(0, action == 0);
            n.set_low_high_visible(1, action == 0)
        }
    }
    /// `SlicerWindow::keyPressEvent`.
    pub fn key_press_event(&mut self, event: SlicerEvent, core: &mut dyn SlicerCore) {
        core.key_input(event)
    }
    /// `SlicerWindow::keyReleaseEvent`.
    pub fn key_release_event(&mut self, event: SlicerEvent, core: &mut dyn SlicerCore) {
        core.key_release(event)
    }
    /// `SlicerWindow::closeEvent`.
    pub fn close_event(&mut self, core: &mut dyn SlicerCore, n: &mut dyn SlicerNativeBoundary) {
        if n.ignore_closing(core.is_closing()) {
            return;
        };
        core.closing();
        n.accept_close();
        n.delete_slicer_funcs()
    }
    /// `SlicerWindow::toolbarMenuEvent`.
    pub fn toolbar_menu_event(
        &mut self,
        index: i32,
        core: &mut dyn SlicerCore,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        self.context_menu_hit(index, core, n)
    }
    /// `SlicerWindow::contextMenuHit`.
    pub fn context_menu_hit(
        &mut self,
        index: i32,
        core: &mut dyn SlicerCore,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        if let Some(event) = n.popup_key(index) {
            core.key_input(event)
        }
    }
}

/// `SlicerGL` (`slicer_classes.h`).
#[derive(Clone, Debug)]
pub struct SlicerGl {
    pub m_init_width: i32,
    pub m_init_height: i32,
    pub m_first_draw: i32,
    pub m_add_cube_on_timer: bool,
    pub m_mouse_pressed: bool,
    pub m_mouse_in_window: i32,
    pub m_timer_id: i32,
    pub m_scheduled_resize: bool,
    pub m_scheduled_bump: i32,
}
impl Default for SlicerGl {
    fn default() -> Self {
        Self {
            m_init_width: 0,
            m_init_height: 0,
            m_first_draw: 3,
            m_add_cube_on_timer: false,
            m_mouse_pressed: false,
            m_mouse_in_window: -1,
            m_timer_id: 0,
            m_scheduled_resize: false,
            m_scheduled_bump: 2,
        }
    }
}
impl SlicerGl {
    /// `SlicerGL::SlicerGL`.
    pub fn new() -> Self {
        Self::default()
    }
    /// `SlicerGL::~SlicerGL`.
    pub fn destroy(&mut self) {}
    /// `SlicerGL::setBufferSwapAuto`.
    pub fn set_buffer_swap_auto(&mut self, _: bool) {}
    /// `SlicerGL::initializeGL`.
    pub fn initialize_gl(&mut self) {}
    /// `SlicerGL::updateGL` (Qt 6 compatibility inline method).
    pub fn update_gl(&mut self, n: &mut dyn SlicerNativeBoundary) {
        n.update_gl()
    }
    /// `SlicerGL::swapBuffers` (Qt 6 compatibility inline method).
    pub fn swap_buffers(&mut self) {}
    /// `SlicerGL::extraCursorInWindow`.
    pub fn extra_cursor_in_window(&self) -> bool {
        self.m_mouse_pressed || self.m_mouse_in_window != 0
    }
    /// `SlicerGL::paintGL`.
    pub fn paint_gl(&mut self, core: &mut dyn SlicerCore, n: &mut dyn SlicerNativeBoundary) {
        if self.m_first_draw > 2 {
            self.m_timer_id = n.start_timer(10);
            self.m_first_draw -= 1
        }
        if !n.imod_loop_started() || (self.m_init_width == 0 && self.m_init_height == 0) {
            return;
        };
        core.paint()
    }
    /// `SlicerGL::timerEvent`.
    pub fn timer_event(&mut self, core: &mut dyn SlicerCore, n: &mut dyn SlicerNativeBoundary) {
        if !n.imod_loop_started()
            || (self.m_init_width == 0 && self.m_init_height == 0)
            || core.is_closing()
        {
            return;
        }
        if self.m_scheduled_resize {
            self.cancel_resize(n);
            let (w, h) = n.window_size();
            n.resize_window(w + self.m_scheduled_bump, h);
            self.m_scheduled_bump = -self.m_scheduled_bump;
            return;
        }
        if self.m_add_cube_on_timer {
            if self.m_timer_id != 0 {
                n.kill_timer(self.m_timer_id)
            };
            self.m_timer_id = 0;
            if !n.cube_exists() {
                core.add_cube_to_frame()
            };
            self.m_add_cube_on_timer = false;
            return;
        }
        if self.m_first_draw < 2 {
            if self.m_timer_id != 0 {
                n.kill_timer(self.m_timer_id)
            };
            self.m_timer_id = 0
        }
        if self.m_first_draw > 0 {
            n.resize_window(
                self.m_init_width + self.m_first_draw - 1,
                self.m_init_height,
            )
        } else {
            n.update_gl()
        }
        if self.m_first_draw < 2 && !n.cube_exists() {
            core.add_cube_to_frame()
        }
        self.m_first_draw -= 1;
        if n.is_windows() {
            self.m_first_draw = self.m_first_draw.max(0)
        }
    }
    /// `SlicerGL::scheduleResize`.
    pub fn schedule_resize(&mut self, interval: i32, n: &mut dyn SlicerNativeBoundary) {
        if n.is_windows() {
            if self.m_timer_id == 0 {
                self.m_timer_id = n.start_timer(interval)
            };
            return;
        }
        if self.m_first_draw > 0 {
            return;
        };
        if self.m_timer_id == 0 {
            self.m_timer_id = n.start_timer(interval)
        };
        self.m_scheduled_resize = true
    }
    /// `SlicerGL::cancelResize`.
    pub fn cancel_resize(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if self.m_timer_id != 0 {
            n.kill_timer(self.m_timer_id)
        };
        self.m_timer_id = 0;
        self.m_scheduled_resize = false
    }
    /// `SlicerGL::scheduleCubeAdd`.
    pub fn schedule_cube_add(&mut self, interval: i32, n: &mut dyn SlicerNativeBoundary) {
        if self.m_timer_id == 0 {
            self.m_timer_id = n.start_timer(interval)
        };
        self.m_add_cube_on_timer = true
    }
    /// `SlicerGL::resizeGL`.
    pub fn resize_gl(&mut self, width: i32, height: i32, core: &mut dyn SlicerCore) {
        if !core.is_closing() {
            core.resize(width, height)
        }
    }
    /// `SlicerGL::mousePressEvent`.
    pub fn mouse_press_event(&mut self, event: SlicerEvent, core: &mut dyn SlicerCore) {
        self.m_mouse_pressed = true;
        core.mouse_press(event)
    }
    /// `SlicerGL::mouseMoveEvent`.
    pub fn mouse_move_event(&mut self, event: SlicerEvent, core: &mut dyn SlicerCore) {
        self.m_mouse_in_window = 1;
        core.mouse_move(event)
    }
    /// `SlicerGL::mouseReleaseEvent`.
    pub fn mouse_release_event(&mut self, event: SlicerEvent, core: &mut dyn SlicerCore) {
        self.m_mouse_pressed = false;
        core.mouse_release(event)
    }
    /// `SlicerGL::wheelEvent`.
    pub fn wheel_event(&mut self, event: SlicerEvent, core: &mut dyn SlicerCore) {
        core.general_event(event)
    }
    /// `SlicerGL::enterEvent`.
    pub fn enter_event(&mut self, event: SlicerEvent, core: &mut dyn SlicerCore) {
        self.m_mouse_in_window = 1;
        core.general_event(event)
    }
    /// `SlicerGL::leaveEvent`.
    pub fn leave_event(&mut self, event: SlicerEvent, core: &mut dyn SlicerCore) {
        self.m_mouse_in_window = 0;
        core.general_event(event)
    }
}

/// `SlicerCube` (`slicer_classes.h`).
#[derive(Clone, Debug, Default)]
pub struct SlicerCube;
impl SlicerCube {
    pub fn new() -> Self {
        Self
    }
    pub fn destroy(&mut self) {}
    /// `SlicerCube::initializeGL`.
    pub fn initialize_gl(&mut self) {}
    /// `SlicerCube::updateGL` (Qt 6 compatibility inline method).
    pub fn update_gl(&mut self, n: &mut dyn SlicerNativeBoundary) {
        n.update_gl()
    }
    pub fn paint_gl(&mut self, core: &mut dyn SlicerCore) {
        core.cube_paint()
    }
    pub fn resize_gl(&mut self, width: i32, height: i32, core: &mut dyn SlicerCore) {
        core.cube_resize(width, height)
    }
}

/// `HotWidget` (`slicer_classes.h`): source signal forwarding is represented by returned events.
#[derive(Clone, Debug, Default)]
pub struct HotWidget;
impl HotWidget {
    pub fn new() -> Self {
        Self
    }
    pub fn destroy(&mut self) {}
    pub fn key_press_event(&mut self, event: SlicerEvent) -> SlicerEvent {
        event
    }
    pub fn key_release_event(&mut self, event: SlicerEvent) -> SlicerEvent {
        event
    }
    pub fn context_menu_event(&mut self, event: SlicerEvent) -> SlicerEvent {
        event
    }
    pub fn close_event(&mut self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct C {
        calls: Vec<String>,
        closing: bool,
        auto: i32,
    }
    impl SlicerCore for C {
        fn help(&mut self) {}
        fn step_zoom(&mut self, d: i32) {
            self.calls.push(format!("z{d}"))
        }
        fn step_time(&mut self, _: i32) {}
        fn entered_zoom(&mut self, _: f32) {}
        fn show_slice(&mut self) {}
        fn fill_cache(&mut self) {}
        fn state_toggled(&mut self, i: usize, s: i32) {
            self.calls.push(format!("t{i}:{s}"))
        }
        fn angle_changed(&mut self, _: i32, _: i32, _: bool) {}
        fn image_thickness(&mut self, _: i32) {}
        fn model_thickness(&mut self, _: f32) {}
        fn resize(&mut self, _: i32, _: i32) {}
        fn cube_resize(&mut self, _: i32, _: i32) {}
        fn closing(&mut self) {
            self.closing = true
        }
        fn key_input(&mut self, _: SlicerEvent) {}
        fn key_release(&mut self, _: SlicerEvent) {}
        fn mouse_press(&mut self, _: SlicerEvent) {}
        fn mouse_release(&mut self, _: SlicerEvent) {}
        fn mouse_move(&mut self, _: SlicerEvent) {}
        fn general_event(&mut self, _: SlicerEvent) {}
        fn paint(&mut self) {
            self.calls.push("paint".into())
        }
        fn cube_paint(&mut self) {}
        fn angles_from_contour(&mut self) -> bool {
            false
        }
        fn check_movie_limits(&mut self) {
            self.calls.push("limits".into())
        }
        fn set_current_or_new_row(&mut self, _: bool) {}
        fn set_angles_from_row(&mut self) {}
        fn rotate_on_view_axis(&mut self, _: i32, _: i32, _: i32) {}
        fn set_band_low_high_limit(&mut self, _: i32) {}
        fn set_linked_state(&mut self, _: bool) {}
        fn view_axis_step_size(&self) -> f32 {
            1.
        }
        fn add_cube_to_frame(&mut self) {
            self.calls.push("cube".into())
        }
        fn is_closing(&self) -> bool {
            self.closing
        }
        fn auto_link(&self) -> i32 {
            self.auto
        }
        fn set_continuous(&mut self, _: bool) {}
    }
    #[derive(Default)]
    struct N {
        slider: Vec<(i32, i32)>,
        timers: i32,
    }
    impl SlicerNativeBoundary for N {
        fn construct_slicer_window(&mut self, _: [f32; 3], _: &str, _: bool, _: bool, _: bool) {}
        fn build_toolbar2(&mut self, _: bool, _: [f32; 3], _: f32) {}
        fn set_font_dependent_widths(&mut self) {}
        fn check_and_set_mac_menu(&mut self) {}
        fn toolbar_size_hints(&self) -> (i32, i32, i32) {
            (10, 10, 15)
        }
        fn set_save_angle_toolbar_visible(&mut self, _: bool, _: Option<bool>) {}
        fn destroy_embedded_toolbar2(&mut self) {}
        fn destroy_free_toolbar2(&mut self) {}
        fn show_toolbar2(&mut self) {}
        fn defer_free_toolbar2_show(&mut self) {}
        fn close_linked_slicers(&mut self) {}
        fn slicer_view_axis_step_change(&mut self, _: i32) {}
        fn set_slider_value(&mut self, w: i32, v: i32) {
            self.slider.push((w, v))
        }
        fn set_slider_min_max_value(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn set_model_thickness(&mut self, _: f32) {}
        fn set_image_thickness(&mut self, _: i32) {}
        fn set_toggle_checked(&mut self, _: usize, _: bool) {}
        fn set_rotation_center_state(&mut self, _: bool) {}
        fn set_zoom_text(&mut self, _: &str) {}
        fn zoom_text(&self) -> String {
            "1".into()
        }
        fn focus_window(&mut self) {}
        fn set_time_label(&mut self, _: &str, _: &str) {}
        fn set_low_high_color(&mut self, _: i32, _: i32) {}
        fn set_low_high_visible(&mut self, _: i32, _: bool) {}
        fn set_band_size(&mut self, _: &str, _: bool) {}
        fn ignore_closing(&self, _: bool) -> bool {
            false
        }
        fn accept_close(&mut self) {}
        fn delete_slicer_funcs(&mut self) {}
        fn popup_key(&mut self, _: i32) -> Option<SlicerEvent> {
            None
        }
        fn imod_loop_started(&self) -> bool {
            true
        }
        fn start_timer(&mut self, _: i32) -> i32 {
            self.timers += 1;
            self.timers
        }
        fn kill_timer(&mut self, _: i32) {}
        fn update_gl(&mut self) {}
        fn window_size(&self) -> (i32, i32) {
            (10, 10)
        }
        fn resize_window(&mut self, _: i32, _: i32) {}
        fn cube_exists(&self) -> bool {
            false
        }
        fn mouse_left_button_only(&self) -> bool {
            false
        }
        fn is_windows(&self) -> bool {
            false
        }
    }
    #[test]
    fn source_toggle_angle_and_contour_paths() {
        let mut c = C::default();
        let mut n = N::default();
        let mut w = SlicerWindow::new(&c, [90.; 3], "", false, false, false, &mut n);
        w.toggle_clicked(SLICER_TOGGLE_FFT, true, &mut c);
        w.set_angles([1.05, -2., 3.], &mut n);
        w.contour_pressed(&mut c);
        assert_eq!(c.calls, ["t5:1", "limits"]);
        assert_eq!(n.slider, vec![(0, 11), (1, -20), (2, 30)]);
    }
    #[test]
    fn delayed_gl_paint_and_cube_follow_source_state() {
        let mut c = C::default();
        let mut n = N::default();
        let mut gl = SlicerGl::new();
        gl.m_init_width = 20;
        gl.m_init_height = 10;
        gl.paint_gl(&mut c, &mut n);
        gl.timer_event(&mut c, &mut n);
        gl.timer_event(&mut c, &mut n);
        assert!(c.calls.contains(&"paint".into()));
        assert!(c.calls.contains(&"cube".into()));
    }
}
