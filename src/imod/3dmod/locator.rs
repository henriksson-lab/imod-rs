//! Translation of `IMOD/3dmod/locator.cpp` and `locator.h`.
//!
//! The two Qt widgets in the source are represented by their complete source
//! state.  Qt allocation, OpenGL drawing, image-cache access, and ZaP calls
//! are deliberately direct [`LocatorNativeBoundary`] operations.
#![allow(dead_code)]

use crate::imod::three_dmod::imodview::{IMOD_DRAW_IMAGE, IMOD_DRAW_XYZ, ImodView};

/// `LocatorWindow`'s portable key input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocatorKey {
    Minus,
    Plus,
    Equal,
    Other(i32),
}

/// State supplied by `zapSubsetLimits`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocatorSubset {
    pub result: i32,
    pub x0: i32,
    pub y0: i32,
    pub nx: i32,
    pub ny: i32,
}

/// Native Qt/OpenGL/image-cache/ZaP boundary used by the paired source unit.
pub trait LocatorNativeBoundary {
    fn raise_window(&mut self);
    fn create_window(&mut self) -> bool;
    fn set_window_title(&mut self, title: &str);
    fn add_dialog(&mut self);
    fn remove_dialog(&mut self);
    fn show_window(&mut self);
    fn adjust_geometry_and_show(&mut self);
    fn resize_window(&mut self, width: i32, height: i32);
    fn window_height(&self) -> i32;
    fn gl_height(&self) -> i32;
    fn set_colormap(&mut self);
    fn initialize_screen_change(&mut self, device_pixel_ratio: f32);
    fn set_current_size(&mut self, width: i32, height: i32);
    fn resize_viewport_xy(&mut self, width: i32, height: i32);
    fn new_ci_image(&mut self, width: i32, height: i32) -> bool;
    fn free_ci_image(&mut self);
    fn flush_image(&mut self);
    fn update_gl(&mut self);
    fn start_timer(&mut self, milliseconds: i32) -> i32;
    fn kill_timer(&mut self, timer_id: i32);
    fn current_time(&self, vi: &ImodView) -> i32;
    fn load_axis(&self) -> i32;
    fn zap_subset_limits(&self, vi: &ImodView) -> LocatorSubset;
    fn has_pyramid_cache(&self) -> bool;
    fn pyramid_section_area(
        &mut self,
        section: i32,
        zoom: f64,
    ) -> Option<(i32, i32, f32, f32, i32, i32)>;
    fn z_section_time(&mut self, section: i32, time: i32) -> Option<(i32, i32)>;
    fn draw_boxout(&mut self, left: i32, bottom: i32, right: i32, top: i32);
    fn draw_greyscale_pixels_hq(
        &mut self,
        input_size: (i32, i32),
        output: (i32, i32),
        zoom: f64,
        section: i32,
        rampbase: i32,
        rgba: bool,
    );
    fn line_width(&mut self, width: i32);
    fn color_index(&mut self, index: i32);
    fn draw_rectangle(&mut self, x: i32, y: i32, width: i32, height: i32);
    fn set_zoom_label(&mut self, text: &str);
    fn set_cursor_size_all(&mut self);
    fn unset_cursor(&mut self);
    fn raise_if_needed(&mut self);
    fn actual_button(&self, button: i32) -> i32;
    fn buttons(&self, event_buttons: i32) -> i32 {
        event_buttons
    }
    fn zap_set_image_or_band_center(&mut self, x: f32, y: f32, incremental: bool);
    fn default_keys(&mut self, key: LocatorKey);
    fn show_help(&mut self, page: &str);
    fn endpoint(&self) -> i32;
    fn rgba(&self) -> bool;
    fn new_qt_opengl(&self) -> bool;
}

/// `LocatorWindow` fields kept in `locator.h`.
#[derive(Clone, Debug, Default)]
pub struct LocatorWindow {
    pub m_ctrl: i32,
    pub m_zoom_label: String,
    pub m_device_pixel_ratio: f32,
    pub m_closed: bool,
}

impl LocatorWindow {
    /// `LocatorWindow::LocatorWindow`; Qt layout/widget allocation is a boundary.
    pub fn new(device_pixel_ratio: f32) -> Self {
        Self {
            m_device_pixel_ratio: device_pixel_ratio,
            ..Self::default()
        }
    }
    /// `LocatorWindow::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self) {}
    /// `LocatorWindow::changeEvent`.
    pub fn change_event(&mut self, font_change: bool) {
        if font_change {
            self.set_font_dependent_widths()
        }
    }
    /// `LocatorWindow::closeEvent`.
    pub fn close_event(&mut self, state: &mut LocatorState, n: &mut dyn LocatorNativeBoundary) {
        let Some(gl) = state.gl.as_mut() else { return };
        // `ivwRemoveControl` is owned by control.cpp; its caller removes the id.
        gl.m_control_removed = self.m_ctrl;
        n.remove_dialog();
        n.free_ci_image();
        state.window = None;
        self.m_closed = true;
    }
    /// `LocatorWindow::help`.
    pub fn help(&mut self, n: &mut dyn LocatorNativeBoundary) {
        n.show_help("locator.html#TOP")
    }
    /// `LocatorWindow::zoomUp`.
    pub fn zoom_up(&mut self, state: &mut LocatorState, n: &mut dyn LocatorNativeBoundary) {
        if let Some(gl) = state.gl.as_mut() {
            gl.change_size(1.5, self, n)
        }
    }
    /// `LocatorWindow::zoomDown`.
    pub fn zoom_down(&mut self, state: &mut LocatorState, n: &mut dyn LocatorNativeBoundary) {
        if let Some(gl) = state.gl.as_mut() {
            gl.change_size(0.6667, self, n)
        }
    }
    /// `LocatorWindow::keyPressEvent`.
    pub fn key_press_event(
        &mut self,
        key: LocatorKey,
        state: &mut LocatorState,
        n: &mut dyn LocatorNativeBoundary,
    ) {
        match key {
            LocatorKey::Minus => self.zoom_down(state, n),
            LocatorKey::Plus | LocatorKey::Equal => self.zoom_up(state, n),
            _ => n.default_keys(key),
        }
    }
}

/// A Qt mouse event's fields examined by `LocatorGL`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocatorMouseEvent {
    pub x: i32,
    pub y: i32,
    pub buttons: i32,
}

/// `LocatorGL` fields declared in `locator.h`.
#[derive(Clone, Debug)]
pub struct LocatorGl {
    pub m_winx: i32,
    pub m_winy: i32,
    pub m_zoom: f64,
    pub m_last_axis: i32,
    pub m_last_sub_ret: i32,
    pub m_section: i32,
    pub m_time: i32,
    pub m_last_x0: i32,
    pub m_last_y0: i32,
    pub m_last_nx: i32,
    pub m_last_ny: i32,
    pub m_black: i32,
    pub m_white: i32,
    pub m_first_draw: bool,
    pub m_xborder: i32,
    pub m_yborder: i32,
    pub m_timer_id: i32,
    pub m_cursor_set: i32,
    pub m_mouse_x: i32,
    pub m_mouse_y: i32,
    pub m_control_removed: i32,
    /// `mTopWin->mDevicePixelRatio`, retained instead of a Qt pointer.
    pub m_device_pixel_ratio: f32,
}
impl Default for LocatorGl {
    fn default() -> Self {
        Self {
            m_winx: 0,
            m_winy: 0,
            m_zoom: 0.,
            m_last_axis: 0,
            m_last_sub_ret: 0,
            m_section: 0,
            m_time: 0,
            m_last_x0: 0,
            m_last_y0: 0,
            m_last_nx: 0,
            m_last_ny: 0,
            m_black: -1,
            m_white: -1,
            m_first_draw: true,
            m_xborder: 0,
            m_yborder: 0,
            m_timer_id: 0,
            m_cursor_set: 0,
            m_mouse_x: 0,
            m_mouse_y: 0,
            m_control_removed: 0,
            m_device_pixel_ratio: 1.,
        }
    }
}
impl LocatorGl {
    /// `LocatorGL::LocatorGL`.
    pub fn new(device_pixel_ratio: f32) -> Self {
        Self {
            m_device_pixel_ratio: device_pixel_ratio,
            ..Self::default()
        }
    }
    /// `LocatorGL::resizeGL`.
    pub fn resize_gl(
        &mut self,
        width: i32,
        height: i32,
        window: &LocatorWindow,
        n: &mut dyn LocatorNativeBoundary,
    ) {
        let (width, height) = if n.new_qt_opengl() && window.m_device_pixel_ratio > 1. {
            (
                (width as f32 * window.m_device_pixel_ratio) as i32,
                (height as f32 * window.m_device_pixel_ratio) as i32,
            )
        } else {
            (width, height)
        };
        self.m_winx = width;
        self.m_winy = height;
        n.set_current_size(width, height);
        n.resize_viewport_xy(width, height);
        let _ = window;
        n.new_ci_image(width, height);
    }
    /// `LocatorGL::fakeResize`.
    pub fn fake_resize(
        &mut self,
        width: i32,
        height: i32,
        window: &LocatorWindow,
        n: &mut dyn LocatorNativeBoundary,
    ) {
        self.resize_gl(width, height, window, n)
    }
    /// `LocatorGL::changeSize`.
    pub fn change_size(
        &mut self,
        factor: f32,
        window: &LocatorWindow,
        n: &mut dyn LocatorNativeBoundary,
    ) {
        let mut newx = (factor * self.m_winx as f32) as i32;
        newx = newx.max(16);
        let mut newy =
            ((factor * self.m_winy as f32) as i32).max(16) + n.window_height() - n.gl_height();
        if window.m_device_pixel_ratio > 1. {
            newx = (newx as f32 / window.m_device_pixel_ratio) as i32;
            newy = (newy as f32 / window.m_device_pixel_ratio) as i32;
        }
        n.resize_window(newx, newy)
    }
    /// `LocatorGL::drawIfNeeded`.
    pub fn draw_if_needed(
        &mut self,
        vi: &ImodView,
        drawflag: i32,
        n: &mut dyn LocatorNativeBoundary,
    ) {
        let time = n.current_time(vi);
        let section = vi.zmouse.round() as i32;
        let sub = n.zap_subset_limits(vi);
        if section != self.m_section
            || time != self.m_time
            || n.load_axis() != self.m_last_axis
            || sub.result != self.m_last_sub_ret
            || drawflag & IMOD_DRAW_IMAGE != 0
            || vi.black != self.m_black
            || vi.white != self.m_white
            || (sub.result == 0
                && (sub.x0 != self.m_last_x0
                    || sub.y0 != self.m_last_y0
                    || sub.nx != self.m_last_nx
                    || sub.ny != self.m_last_ny))
        {
            n.update_gl();
        }
    }
    /// `LocatorGL::scheduleDraw`.
    pub fn schedule_draw(&mut self, n: &mut dyn LocatorNativeBoundary) {
        if self.m_timer_id == 0 {
            self.m_timer_id = n.start_timer(10)
        }
    }
    /// `LocatorGL::timerEvent`.
    pub fn timer_event(&mut self, vi: &ImodView, n: &mut dyn LocatorNativeBoundary) {
        n.kill_timer(self.m_timer_id);
        self.m_timer_id = 0;
        if self.m_first_draw {
            self.m_first_draw = false;
            n.update_gl()
        } else {
            self.draw_if_needed(vi, IMOD_DRAW_XYZ, n)
        }
    }
    /// `LocatorGL::paintGL`.
    pub fn paint_gl(
        &mut self,
        vi: &ImodView,
        window: &mut LocatorWindow,
        n: &mut dyn LocatorNativeBoundary,
    ) {
        if self.m_first_draw && self.m_timer_id == 0 {
            self.m_timer_id = n.start_timer(10)
        }
        n.set_current_size(self.m_winx, self.m_winy);
        self.m_zoom =
            (self.m_winx as f64 / vi.xsize as f64).min(self.m_winy as f64 / vi.ysize as f64);
        let mut zoom = self.m_zoom;
        let mut imx = vi.xsize;
        let mut imy = vi.ysize;
        self.m_section = vi.zmouse.round() as i32;
        let time = n.current_time(vi);
        if time != self.m_time || vi.black != self.m_black || vi.white != self.m_white {
            n.flush_image()
        }
        self.m_time = time;
        self.m_last_axis = n.load_axis();
        self.m_black = vi.black;
        self.m_white = vi.white;
        if n.has_pyramid_cache() {
            if let Some((x, y, _, _, scale, _)) =
                n.pyramid_section_area(self.m_section, self.m_zoom)
            {
                imx = x;
                imy = y;
                zoom = self.m_zoom * scale as f64;
            }
        } else {
            let _ = n.z_section_time(self.m_section, time);
        }
        let xdraw = (zoom * imx as f64) as i32;
        let ydraw = (zoom * imy as f64) as i32;
        self.m_xborder = (self.m_winx - xdraw) / 2;
        self.m_yborder = (self.m_winy - ydraw) / 2;
        n.draw_boxout(
            self.m_xborder,
            self.m_yborder,
            self.m_xborder + xdraw,
            self.m_yborder + ydraw,
        );
        n.draw_greyscale_pixels_hq(
            (imx, imy),
            (self.m_xborder, self.m_yborder),
            zoom,
            self.m_section,
            vi.rampbase,
            n.rgba(),
        );
        let sub = n.zap_subset_limits(vi);
        self.m_last_sub_ret = sub.result;
        self.m_last_x0 = sub.x0;
        self.m_last_y0 = sub.y0;
        self.m_last_nx = sub.nx;
        self.m_last_ny = sub.ny;
        if self.m_last_sub_ret != 0 {
            return;
        }
        n.line_width(1);
        n.color_index(n.endpoint());
        n.draw_rectangle(
            self.m_xborder + (self.m_zoom * self.m_last_x0 as f64) as i32 - 1,
            self.m_yborder + (self.m_zoom * self.m_last_y0 as f64) as i32 - 1,
            (self.m_zoom * self.m_last_nx as f64) as i32 + 1,
            (self.m_zoom * self.m_last_ny as f64) as i32 + 1,
        );
        window.m_zoom_label = format!("Zoom {:.2}", self.m_zoom);
        n.set_zoom_label(&window.m_zoom_label);
    }
    /// `LocatorGL::getImxy`.
    pub fn get_imxy(&self, x: i32, y: i32) -> (f32, f32) {
        (
            ((x - self.m_xborder) as f64 / self.m_zoom) as f32,
            ((self.m_winy - 1 - y - self.m_yborder) as f64 / self.m_zoom) as f32,
        )
    }
    /// `LocatorGL::mousePressEvent`.
    pub fn mouse_press_event(
        &mut self,
        vi: &ImodView,
        e: LocatorMouseEvent,
        n: &mut dyn LocatorNativeBoundary,
    ) {
        if !(self.m_last_nx < vi.xsize - 1 || self.m_last_ny < vi.ysize - 1) {
            return;
        }
        n.raise_if_needed();
        let (ex, ey) = if self.m_device_pixel_ratio > 1. {
            (
                (e.x as f32 * self.m_device_pixel_ratio) as i32,
                (e.y as f32 * self.m_device_pixel_ratio) as i32,
            )
        } else {
            (e.x, e.y)
        };
        if n.buttons(e.buttons) & n.actual_button(1) != 0 {
            let (x, y) = self.get_imxy(ex, ey);
            n.zap_set_image_or_band_center(x, y, false)
        } else if n.buttons(e.buttons) & n.actual_button(2) != 0 {
            n.set_cursor_size_all();
            self.m_cursor_set = 1;
            self.m_mouse_x = ex;
            self.m_mouse_y = ey
        }
    }
    /// `LocatorGL::mouseReleaseEvent`.
    pub fn mouse_release_event(&mut self, _: LocatorMouseEvent, n: &mut dyn LocatorNativeBoundary) {
        if self.m_cursor_set != 0 {
            n.unset_cursor()
        };
        self.m_cursor_set = 0
    }
    /// `LocatorGL::mouseMoveEvent`.
    pub fn mouse_move_event(&mut self, e: LocatorMouseEvent, n: &mut dyn LocatorNativeBoundary) {
        let (ex, ey) = if self.m_device_pixel_ratio > 1. {
            (
                (e.x as f32 * self.m_device_pixel_ratio) as i32,
                (e.y as f32 * self.m_device_pixel_ratio) as i32,
            )
        } else {
            (e.x, e.y)
        };
        let button2 = n.buttons(e.buttons) & n.actual_button(2);
        if !(self.m_cursor_set != 0 && button2 != 0) {
            return;
        }
        let x = ((ex - self.m_mouse_x) as f64 / self.m_zoom) as f32;
        let y = ((self.m_mouse_y - ey) as f64 / self.m_zoom) as f32;
        self.m_mouse_x = ex;
        self.m_mouse_y = ey;
        n.zap_set_image_or_band_center(x, y, true)
    }
}

/// The source file's `LocWin` and `GLw` residents, made explicit instead of mutable globals.
#[derive(Clone, Debug, Default)]
pub struct LocatorState {
    pub window: Option<LocatorWindow>,
    pub gl: Option<LocatorGl>,
    pub next_control: i32,
}

/// `locatorOpen`.
pub fn locator_open(
    state: &mut LocatorState,
    vi: &ImodView,
    device_pixel_ratio: f32,
    n: &mut dyn LocatorNativeBoundary,
) -> i32 {
    if state.window.is_some() {
        n.raise_window();
        return 0;
    }
    if !n.create_window() {
        return -1;
    }
    let mut window = LocatorWindow::new(device_pixel_ratio);
    let gl = LocatorGl::new(device_pixel_ratio);
    if !n.rgba() {
        n.set_colormap()
    };
    n.set_window_title("3dmod Locator");
    state.next_control += 1;
    window.m_ctrl = state.next_control;
    n.add_dialog();
    n.initialize_screen_change(device_pixel_ratio);
    let (mut winx, mut winy) = if vi.xsize > vi.ysize {
        let x = ((vi.xsize as f32 / 6.) as i32)
            .clamp(128, 512)
            .min(vi.xsize);
        (x, (x * vi.ysize + vi.xsize - 1) / vi.xsize)
    } else {
        let y = ((vi.ysize as f32 / 6.) as i32)
            .clamp(128, 512)
            .min(vi.ysize);
        ((y * vi.xsize + vi.ysize - 1) / vi.ysize, y)
    };
    if device_pixel_ratio > 1. {
        winx = (winx as f32 / device_pixel_ratio) as i32;
        winy = (winy as f32 / device_pixel_ratio) as i32;
    }
    winy += n.window_height() - n.gl_height();
    n.resize_window(winx, winy);
    n.adjust_geometry_and_show();
    state.window = Some(window);
    state.gl = Some(gl);
    0
}
/// `locatorClose_cb`.
pub fn locator_close_cb(state: &mut LocatorState, n: &mut dyn LocatorNativeBoundary) {
    if let Some(mut window) = state.window.take() {
        window.close_event(state, n)
    }
}
/// `locatorDraw_cb`.
pub fn locator_draw_cb(
    state: &mut LocatorState,
    vi: &ImodView,
    drawflag: i32,
    n: &mut dyn LocatorNativeBoundary,
) {
    if state.window.is_none() {
        return;
    };
    if drawflag & IMOD_DRAW_IMAGE != 0 {
        n.flush_image()
    }
    if let Some(gl) = state.gl.as_mut() {
        gl.draw_if_needed(vi, drawflag, n)
    }
}
/// `locatorScheduleDraw`.
pub fn locator_schedule_draw(state: &mut LocatorState, n: &mut dyn LocatorNativeBoundary) {
    if let Some(gl) = state.gl.as_mut() {
        gl.schedule_draw(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        calls: Vec<String>,
        timer: i32,
        subset: LocatorSubset,
    }
    impl LocatorNativeBoundary for N {
        fn raise_window(&mut self) {
            self.calls.push("raise".into())
        }
        fn create_window(&mut self) -> bool {
            true
        }
        fn set_window_title(&mut self, _: &str) {}
        fn add_dialog(&mut self) {}
        fn remove_dialog(&mut self) {}
        fn show_window(&mut self) {}
        fn adjust_geometry_and_show(&mut self) {}
        fn resize_window(&mut self, _: i32, _: i32) {}
        fn window_height(&self) -> i32 {
            30
        }
        fn gl_height(&self) -> i32 {
            20
        }
        fn set_colormap(&mut self) {}
        fn initialize_screen_change(&mut self, _: f32) {}
        fn set_current_size(&mut self, _: i32, _: i32) {}
        fn resize_viewport_xy(&mut self, _: i32, _: i32) {}
        fn new_ci_image(&mut self, _: i32, _: i32) -> bool {
            true
        }
        fn free_ci_image(&mut self) {}
        fn flush_image(&mut self) {}
        fn update_gl(&mut self) {
            self.calls.push("update".into())
        }
        fn start_timer(&mut self, _: i32) -> i32 {
            self.timer += 1;
            self.timer
        }
        fn kill_timer(&mut self, _: i32) {}
        fn current_time(&self, vi: &ImodView) -> i32 {
            vi.cur_time
        }
        fn load_axis(&self) -> i32 {
            0
        }
        fn zap_subset_limits(&self, _: &ImodView) -> LocatorSubset {
            self.subset
        }
        fn has_pyramid_cache(&self) -> bool {
            false
        }
        fn pyramid_section_area(
            &mut self,
            _: i32,
            _: f64,
        ) -> Option<(i32, i32, f32, f32, i32, i32)> {
            None
        }
        fn z_section_time(&mut self, _: i32, _: i32) -> Option<(i32, i32)> {
            Some((1, 1))
        }
        fn draw_boxout(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn draw_greyscale_pixels_hq(
            &mut self,
            _: (i32, i32),
            _: (i32, i32),
            _: f64,
            _: i32,
            _: i32,
            _: bool,
        ) {
        }
        fn line_width(&mut self, _: i32) {}
        fn color_index(&mut self, _: i32) {}
        fn draw_rectangle(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn set_zoom_label(&mut self, _: &str) {}
        fn set_cursor_size_all(&mut self) {}
        fn unset_cursor(&mut self) {}
        fn raise_if_needed(&mut self) {}
        fn actual_button(&self, b: i32) -> i32 {
            b
        }
        fn zap_set_image_or_band_center(&mut self, x: f32, y: f32, inc: bool) {
            self.calls.push(format!("{x:.1},{y:.1},{inc}"))
        }
        fn default_keys(&mut self, _: LocatorKey) {}
        fn show_help(&mut self, _: &str) {}
        fn endpoint(&self) -> i32 {
            1
        }
        fn rgba(&self) -> bool {
            true
        }
        fn new_qt_opengl(&self) -> bool {
            true
        }
    }
    #[test]
    fn locator_opens_and_schedules() {
        let mut s = LocatorState::default();
        let mut v = ImodView::default();
        v.xsize = 600;
        v.ysize = 300;
        let mut n = N::default();
        assert_eq!(locator_open(&mut s, &v, 1., &mut n), 0);
        locator_schedule_draw(&mut s, &mut n);
        assert_eq!(s.gl.as_ref().unwrap().m_timer_id, 1)
    }
    #[test]
    fn locator_mouse_translates_coordinates() {
        let mut gl = LocatorGl {
            m_winx: 100,
            m_winy: 100,
            m_zoom: 2.,
            m_last_nx: 20,
            m_last_ny: 20,
            ..Default::default()
        };
        let mut v = ImodView::default();
        v.xsize = 100;
        v.ysize = 100;
        let mut n = N::default();
        gl.mouse_press_event(
            &v,
            LocatorMouseEvent {
                x: 10,
                y: 20,
                buttons: 1,
            },
            &mut n,
        );
        assert_eq!(n.calls.last().unwrap(), "5.0,39.5,false");
        gl.mouse_press_event(
            &v,
            LocatorMouseEvent {
                x: 10,
                y: 20,
                buttons: 2,
            },
            &mut n,
        );
        gl.mouse_move_event(
            LocatorMouseEvent {
                x: 14,
                y: 18,
                buttons: 2,
            },
            &mut n,
        );
        assert_eq!(n.calls.last().unwrap(), "2.0,1.0,true")
    }
}
