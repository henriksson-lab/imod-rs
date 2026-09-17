//! Translation of `IMOD/3dmod/xyz.cpp` and `xxyz.h`.
//!
//! The Qt widgets, OpenGL command stream, controller registration, and image
//! cache remain explicit endpoints in [`XyzNativeBoundary`].  The geometry,
//! coordinate conversion, slice summing, drag state, and XYZ-window state are
//! kept in this unit rather than being reconstructed by a GUI toolkit.
#![allow(dead_code)]

pub const MAX_XYZ_TOGGLES: usize = 4;
pub const NUM_AXIS: usize = 3;
pub const BM_WIDTH: i32 = 16;
pub const BM_HEIGHT: i32 = 16;
pub const XYZ_BSIZE: i32 = 8;
pub const XYZ_GSIZE: i32 = 16;
pub const ALL_BORDER: i32 = 2 * XYZ_BSIZE + XYZ_GSIZE;
pub const GRAB_LENGTH: i32 = 7;
pub const GRAB_WIDTH: i32 = 3;
pub const XYZ_TOGGLE_RESOL: usize = 0;
pub const XYZ_TOGGLE_LOCKED: usize = 1;
pub const XYZ_TOGGLE_ZSCALE: usize = 2;
pub const XYZ_TOGGLE_TIMELOCK: usize = 3;
pub const NUM_MAINBAR_TOGGLES: usize = 3;
pub const MAX_SLIDER_WIDTH: i32 = 100;
pub const MIN_SLIDER_WIDTH: i32 = 20;
pub const NOTNEW: i32 = -999_999_999;
pub const NOT_IN_BOX: i32 = 0;
pub const X_SLICE_BOX: i32 = 1;
pub const Y_SLICE_BOX: i32 = 2;
pub const Z_SLICE_BOX: i32 = 3;
pub const Y_GADGET_BOX: i32 = 4;
pub const X_GADGET_BOX: i32 = 5;
pub const Z_GADGET_BOX: i32 = 6;
pub const FRACTION_BOX: i32 = 7;
pub const IMOD_MMODEL: i32 = 1;
pub const XYZ_KEY_INSERT: i32 = 0x0100_0006;
pub const XYZ_KEY_0: i32 = b'0' as i32;

/// `ImodView::slice` coordinates consumed by `XyzGL::paintGL`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct XyzSliceOverlay {
    pub zx1: f32,
    pub zy1: f32,
    pub zx2: f32,
    pub zy2: f32,
    pub yx1: f32,
    pub yz1: f32,
    pub yx2: f32,
    pub yz2: f32,
    pub xz1: f32,
    pub xy1: f32,
    pub xz2: f32,
    pub xy2: f32,
}

/// Image/model values consumed directly by `xyz.cpp`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct XyzViewState {
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub xmouse: f32,
    pub ymouse: f32,
    pub zmouse: f32,
    pub zscale: f32,
    pub cur_time: i32,
    pub num_times: i32,
    pub ushort_store: bool,
    pub rgb_store: bool,
    pub has_pyramid_cache: bool,
    pub slice_overlay: XyzSliceOverlay,
}
/// Qt event types used by `XyzWindow::changeEvent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XyzChangeEvent {
    FontChange,
    Other,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XyzFitSize {
    pub width: i32,
    pub height: i32,
}

/// One `FastSegment` from the cache-filling path in `xyz.cpp`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FastSegment {
    pub x_or_y: i32,
    pub length: i32,
    pub line: Vec<u8>,
    pub stride: i32,
}

/// Qt, controller, image/cache, model editing, and OpenGL calls from this
/// source unit.  Implementations own those framework resources.
pub trait XyzNativeBoundary {
    fn xyz_b1_model_action(&mut self, _x: i32, _y: i32) {}
    fn xyz_b2_model_action(&mut self, _x: i32, _y: i32) {}
    fn xyz_b3_model_action(&mut self, _x: i32, _y: i32) {}
    fn set_time_label(&mut self, _number: &str, _label: &str) {}
    fn focus_window(&mut self) {}
    fn xyz_b2_drag(&mut self, _x: i32, _y: i32) {}
    fn xyz_b3_drag(&mut self, _x: i32, _y: i32) {}
    fn wheel_change_point_size(&mut self, _zoom: f32) {}
    fn xyz_b1_press(&mut self, _x: i32, _y: i32) {}
    fn key_press_passed_on(&mut self, _key: i32, _modifiers: i32) {}
    fn toolbar_popup_selection(&mut self) -> Option<i32> {
        None
    }
    fn popup_key_event(&mut self, _index: i32) -> Option<(i32, i32)> {
        None
    }
    fn dispatch_key_press(&mut self, _key: i32, _modifiers: i32) {}
    fn set_axis_slider_range(&mut self, _axis: i32, _min: i32, _max: i32) {}
    fn axis_slider_minimum_width(&self, _axis: i32) -> i32 {
        0
    }
    fn set_axis_slider_width(&mut self, _axis: i32, _width: i32) {}
    fn set_axis_slider_value(&mut self, _axis: usize, _value: i32) {}
    fn set_zoom_text(&mut self, _text: &str) {}
    fn time_index_label(&self, _time: i32) -> String {
        String::new()
    }
    fn set_model_cursor(&mut self, _model_cursor: bool) {}
    fn check_and_set_mac_menu(&mut self) {}
    fn set_help_button_width(&mut self) {}
    fn view_state(&self) -> XyzViewState;
    fn set_location(&mut self, x: Option<i32>, y: Option<i32>, z: Option<i32>);
    fn draw(&mut self, flags: i32);
    fn update_gl(&mut self);
    fn control_priority(&mut self, ctrl: i32);
    fn remove_control(&mut self, ctrl: i32);
    fn remove_dialog(&mut self) {}
    fn set_xyz_apply_zscale(&mut self, _apply: i32) {}
    fn movie_xyzt(&mut self, x: i32, y: i32, z: i32, t: i32);
    fn next_time(&mut self, forward: bool);
    fn cache_fill(&mut self);
    fn help(&mut self, page: &str);
    fn draw_image(&mut self);
    fn draw_model(&mut self);
    fn draw_current_lines(&mut self) {}
    fn draw_current_point(&mut self);
    fn draw_auto(&mut self);
    fn draw_tools(&mut self) {}
    fn scale_bar_draw(&mut self, width: i32, height: i32, zoom: f32, dpr: f32) -> f32;
    /// `XyzGL::setMouseTracking`.
    fn set_mouse_tracking(&mut self, _enabled: bool) {}
    fn grab_keyboard(&mut self) {}
    fn release_keyboard(&mut self) {}
    fn grab_mouse(&mut self) {}
    fn release_mouse(&mut self) {}
    /// `b3dSubareaViewport` + `b3dDrawLine` for `xyzShowSlice`.
    fn draw_slice_overlay_line(
        &mut self,
        _viewport: (i32, i32, i32, i32),
        _line: (i32, i32, i32, i32),
    ) {
    }
    fn reset_xyz_viewport(&mut self, _width: i32, _height: i32) {}
    fn pixel_view_open(&self) -> bool {
        false
    }
    fn pixel_view_new_mouse_position(&mut self, _x: f32, _y: f32, _z: i32) {}
    fn foreground_color(&self) -> i32 {
        0
    }
    fn begin_point_color(&self) -> i32 {
        0
    }
    fn end_point_color(&self) -> i32 {
        0
    }
    fn set_line_width(&mut self, _width: i32) {}
    fn set_color_index(&mut self, _color: i32) {}
    fn draw_line(&mut self, _from: (i32, i32), _to: (i32, i32)) {}
    fn draw_rectangle(&mut self, _x: i32, _y: i32, _width: i32, _height: i32) {}
    fn draw_filled_rectangle(&mut self, _x: i32, _y: i32, _width: i32, _height: i32) {}
    fn set_custom_ghost_color(&mut self, _red: i32, _green: i32, _blue: i32) {}
    fn reset_ghost_color(&mut self) {}
    fn input_next_y(&mut self) {}
    fn input_prev_y(&mut self) {}
    fn input_next_x(&mut self) {}
    fn input_prev_x(&mut self) {}
    fn input_page_up_or_down(&mut self, _direction: i32) {}
    fn input_key_point_move(&mut self, _key: i32) {}
    fn redraw_native(&mut self);
    fn close_native(&mut self);
}

/// `XyzWindow` from `xxyz.h`; widget pointers are represented by their source
/// values while the actual Qt ownership is kept behind `XyzNativeBoundary`.
#[derive(Clone, Debug, PartialEq)]
pub struct XyzWindow {
    pub m_fdata_xy: Vec<u8>,
    pub m_fdata_xz: Vec<u8>,
    pub m_fdata_yz: Vec<u8>,
    pub m_sum_temp: Vec<i32>,
    pub m_lx: i32,
    pub m_ly: i32,
    pub m_lz: i32,
    pub m_tool_max_x: i32,
    pub m_tool_max_y: i32,
    pub m_tool_max_z: i32,
    pub m_ctrl: i32,
    pub m_winx: i32,
    pub m_winy: i32,
    pub m_exposed: i32,
    pub m_zoom: f32,
    pub m_new_screen_zoom: f32,
    pub m_xwoffset1: i32,
    pub m_xwoffset2: i32,
    pub m_ywoffset1: i32,
    pub m_ywoffset2: i32,
    pub m_lmx: i32,
    pub m_lmy: i32,
    pub m_first_mx: i32,
    pub m_first_my: i32,
    pub m_win_xdim1: i32,
    pub m_win_xdim2: i32,
    pub m_win_ydim1: i32,
    pub m_win_ydim2: i32,
    pub m_xorigin1: i32,
    pub m_xorigin2: i32,
    pub m_yorigin1: i32,
    pub m_yorigin2: i32,
    pub m_scale_bar_size: f32,
    pub m_whichbox: i32,
    pub m_mousemode: i32,
    pub m_tool_zoom: f32,
    pub m_screen_changed: bool,
    pub m_last_xsize_change: f32,
    pub m_device_pixel_ratio: f32,
    pub m_lock: i32,
    pub m_time_lock: i32,
    pub m_last_cache_sum: i32,
    pub m_xtrans1: i32,
    pub m_ytrans1: i32,
    pub m_xtrans2: i32,
    pub m_ytrans2: i32,
    pub m_hq: i32,
    pub m_project: i32,
    pub m_apply_zscale: i32,
    pub m_thickness: i32,
    pub m_last_thickness: i32,
    pub m_last_xoffset1: i32,
    pub m_last_yoffset1: i32,
    pub m_last_width1: i32,
    pub m_last_height1: i32,
    pub m_last_xoffset2: i32,
    pub m_last_yoffset2: i32,
    pub m_last_width2: i32,
    pub m_last_height2: i32,
    pub m_last_tile_cache_ind: i32,
    pub m_draw_current_only: i32,
    pub m_xz_fraction: f32,
    pub m_yz_fraction: f32,
    pub m_xlock: i32,
    pub m_ylock: i32,
    pub m_zlock: i32,
    pub m_time_drawn: i32,
    pub m_tool_time: i32,
    pub m_doing_draw: bool,
    pub m_toggle_states: [i32; MAX_XYZ_TOGGLES],
    pub m_displayed_axis_location: [i32; NUM_AXIS],
    pub m_ctrl_pressed: bool,
    pub m_closed: bool,
}

impl Default for XyzWindow {
    fn default() -> Self {
        Self::new(XyzViewState::default())
    }
}
impl XyzWindow {
    /// `newPointOutOfPlane()` source method.
    pub fn new_point_out_of_plane(
        points: &[[f32; 3]],
        wild: bool,
        plane: i32,
        mx: i32,
        my: i32,
        mz: i32,
    ) -> bool {
        let Some(first) = points.first() else {
            return false;
        };
        let one = points.len() == 1;
        let round = |v: f32| (v + 0.5) as i32;
        let first_x = round(first[0]);
        let first_y = round(first[1]);
        let planar_x = !one && points.iter().all(|p| round(p[0]) == first_x);
        let planar_y = !one && points.iter().all(|p| round(p[1]) == first_y);
        let planar_z = !wild && !one;
        if !(planar_x || planar_y || planar_z) {
            return false;
        }
        (plane == X_SLICE_BOX
            && (((planar_y || planar_z) && !planar_x) || ((planar_x || one) && first_x != mx)))
            || (plane == Y_SLICE_BOX
                && (((planar_x || planar_z) && !planar_y) || ((planar_y || one) && first_y != my)))
            || (plane == Z_SLICE_BOX
                && (((planar_x || planar_y) && !planar_z)
                    || ((planar_z || one) && round(first[2]) != mz)))
    }
    /// `setControlAndLimits()` source method.
    pub fn set_control_and_limits(&mut self, native: &mut dyn XyzNativeBoundary) {
        native.control_priority(self.m_ctrl);
    }
    /// `B1Press()` source method.
    pub fn b1_press(&mut self, x: i32, y: i32, native: &mut dyn XyzNativeBoundary) {
        native.xyz_b1_model_action(x, y);
    }
    /// `B2Press()` source method.
    pub fn b2_press(&mut self, x: i32, y: i32, native: &mut dyn XyzNativeBoundary) {
        native.xyz_b2_model_action(x, y);
    }
    /// `B3Press()` source method.
    pub fn b3_press(&mut self, x: i32, y: i32, native: &mut dyn XyzNativeBoundary) {
        native.xyz_b3_model_action(x, y);
    }
    /// `B2Drag()` source method.
    pub fn b2_drag(&mut self, x: i32, y: i32, native: &mut dyn XyzNativeBoundary) {
        native.xyz_b2_drag(x, y);
    }
    /// `B3Drag()` source method.
    pub fn b3_drag(&mut self, x: i32, y: i32, native: &mut dyn XyzNativeBoundary) {
        native.xyz_b3_drag(x, y);
    }
    /// `setTimeLabel()` source method.
    pub fn set_time_label(&mut self, time: i32, label: &str, native: &mut dyn XyzNativeBoundary) {
        native.set_time_label(&format!(" ({time:3})"), label);
    }
    /// `newZoom()` source method.
    pub fn new_zoom(&mut self, entered: f32, native: &mut dyn XyzNativeBoundary) {
        self.m_zoom = entered.max(0.01);
        self.draw(native);
        native.focus_window();
    }
    /// `resizeToFit()` source method, excluding host window-position limits.
    pub fn resize_to_fit(
        &mut self,
        view: XyzViewState,
        toolbar_height: i32,
        toolbar_floating: bool,
        zscale: f32,
        border: i32,
    ) -> XyzFitSize {
        let zscaled = view.zsize as f32 * if self.m_apply_zscale != 0 { zscale } else { 1. };
        self.m_xz_fraction = zscaled / (view.xsize as f32 + zscaled).max(1.);
        self.m_yz_fraction = zscaled / (view.ysize as f32 + zscaled).max(1.);
        let width = (self.m_zoom * (view.xsize as f32 + zscaled)) as i32 + border;
        let image_height = (self.m_zoom * (view.ysize as f32 + zscaled)) as i32 + border;
        XyzFitSize {
            width,
            height: image_height + if toolbar_floating { 0 } else { toolbar_height },
        }
    }
    /// `xyzKey_cb()` source callback.
    pub fn xyz_key_cb(
        &mut self,
        released: bool,
        key: i32,
        modifiers: i32,
        native: &mut dyn XyzNativeBoundary,
    ) {
        if !released {
            native.key_press_passed_on(key, modifiers);
        }
    }
    /// The keypad-Insert branch of `keyPressEvent`.  `elapsed_ms` is the
    /// source `insertTime.elapsed()` value supplied by the GUI event host.
    pub fn keypad_insert_pressed(
        &mut self,
        registry: &mut XyzRegistry,
        keypad: bool,
        movie_mode: bool,
        elapsed_ms: i32,
        x: i32,
        y: i32,
        native: &mut dyn XyzNativeBoundary,
    ) -> bool {
        if !keypad || movie_mode || registry.insert_down {
            return false;
        }
        registry.insert_down = true;
        native.set_mouse_tracking(true);
        native.grab_keyboard();
        native.grab_mouse();
        if elapsed_ms > 250 {
            self.b2_press(x, y, native);
        } else {
            self.b2_drag(x, y, native);
        }
        self.m_lmx = x;
        self.m_lmy = y;
        true
    }
    /// `keyRelease`'s keypad-Insert branch.
    pub fn key_release(
        &mut self,
        registry: &mut XyzRegistry,
        key: i32,
        keypad: bool,
        native: &mut dyn XyzNativeBoundary,
    ) {
        if !registry.insert_down || !keypad || (key != XYZ_KEY_INSERT && key != XYZ_KEY_0) {
            return;
        }
        registry.insert_down = false;
        native.set_mouse_tracking(registry.pixel_view_open);
        native.release_keyboard();
        native.release_mouse();
        if self.m_draw_current_only != 0 {
            self.m_draw_current_only = 0;
            self.draw(native);
        }
    }
    /// Arrow/Page-key branch of `keyPressEvent`.  It retains XYZ's
    /// plane-dependent remapping before calling the translated input host.
    pub fn key_navigation(
        &mut self,
        key: i32,
        keypad: bool,
        plane: i32,
        model_mode: bool,
        view: XyzViewState,
        native: &mut dyn XyzNativeBoundary,
    ) -> bool {
        use crate::imod::three_dmod::imod_input::{
            KEY_DOWN, KEY_LEFT, KEY_PAGE_DOWN, KEY_PAGE_UP, KEY_RIGHT, KEY_UP,
        };

        let mut key_use = key;
        if !matches!(
            key,
            KEY_PAGE_UP | KEY_PAGE_DOWN | KEY_LEFT | KEY_RIGHT | KEY_DOWN | KEY_UP
        ) {
            return false;
        }
        native.control_priority(self.m_ctrl);
        let convert_keys = plane == X_SLICE_BOX || plane == Y_SLICE_BOX;
        if convert_keys {
            key_use = match (plane, key) {
                (X_SLICE_BOX, KEY_PAGE_UP) => KEY_RIGHT,
                (Y_SLICE_BOX, KEY_PAGE_UP) => KEY_UP,
                (X_SLICE_BOX, KEY_PAGE_DOWN) => KEY_LEFT,
                (Y_SLICE_BOX, KEY_PAGE_DOWN) => KEY_DOWN,
                (X_SLICE_BOX, KEY_LEFT) => KEY_PAGE_DOWN,
                (X_SLICE_BOX, KEY_RIGHT) => KEY_PAGE_UP,
                (Y_SLICE_BOX, KEY_DOWN) => KEY_PAGE_DOWN,
                (Y_SLICE_BOX, KEY_UP) => KEY_PAGE_UP,
                _ => key,
            };
        }
        if !keypad && (self.m_lock != 0 || convert_keys) {
            if self.m_lock != 0 {
                match key_use {
                    KEY_PAGE_UP | KEY_PAGE_DOWN => {
                        let delta = if key_use == KEY_PAGE_UP { 1 } else { -1 };
                        let _ = self.set_location(view, NOTNEW, NOTNEW, self.m_zlock + delta);
                    }
                    KEY_LEFT | KEY_RIGHT => {
                        let delta = if key_use == KEY_RIGHT { 1 } else { -1 };
                        let _ = self.set_location(view, self.m_xlock + delta, NOTNEW, NOTNEW);
                    }
                    KEY_UP | KEY_DOWN => {
                        let delta = if key_use == KEY_UP { 1 } else { -1 };
                        let _ = self.set_location(view, NOTNEW, self.m_ylock + delta, NOTNEW);
                    }
                    _ => unreachable!(),
                }
                self.draw(native);
            } else {
                match key_use {
                    KEY_UP => native.input_next_y(),
                    KEY_DOWN => native.input_prev_y(),
                    KEY_RIGHT => native.input_next_x(),
                    KEY_LEFT => native.input_prev_x(),
                    KEY_PAGE_UP => native.input_page_up_or_down(1),
                    KEY_PAGE_DOWN => native.input_page_up_or_down(-1),
                    _ => unreachable!(),
                }
            }
            true
        } else if keypad && model_mode {
            native.input_key_point_move(key_use);
            true
        } else {
            false
        }
    }
    /// Keys `1` and `2` from `keyPressEvent`: move only the XYZ time lock.
    pub fn key_time_lock_step(
        &mut self,
        key: i32,
        view: XyzViewState,
        native: &mut dyn XyzNativeBoundary,
    ) -> bool {
        if key != b'1' as i32 && key != b'2' as i32 {
            return false;
        }
        native.control_priority(self.m_ctrl);
        if self.m_time_lock == 0 {
            return false;
        }
        self.m_time_lock =
            (self.m_time_lock + if key == b'1' as i32 { -1 } else { 1 }).clamp(1, view.num_times);
        self.draw(native);
        true
    }
    /// `XyzWindow()` source constructor.
    /// `XyzWindow::XyzWindow`; Qt construction/control registration is native.
    pub fn new(view: XyzViewState) -> Self {
        let zscaled = view.zsize as f32;
        Self {
            m_fdata_xy: Vec::new(),
            m_fdata_xz: if view.has_pyramid_cache {
                vec![]
            } else {
                vec![0; (view.xsize.max(0) * view.zsize.max(0)) as usize]
            },
            m_fdata_yz: if view.has_pyramid_cache {
                vec![]
            } else {
                vec![0; (view.ysize.max(0) * view.zsize.max(0)) as usize]
            },
            m_sum_temp: vec![],
            m_lx: -1,
            m_ly: -1,
            m_lz: -1,
            m_tool_max_x: view.xsize,
            m_tool_max_y: view.ysize,
            m_tool_max_z: view.zsize,
            m_ctrl: 0,
            m_winx: 1,
            m_winy: 0,
            m_exposed: 0,
            m_zoom: 1.,
            m_new_screen_zoom: 0.,
            m_xwoffset1: 0,
            m_xwoffset2: 0,
            m_ywoffset1: 0,
            m_ywoffset2: 0,
            m_lmx: 0,
            m_lmy: 0,
            m_first_mx: 0,
            m_first_my: 0,
            m_win_xdim1: 0,
            m_win_xdim2: 0,
            m_win_ydim1: 0,
            m_win_ydim2: 0,
            m_xorigin1: XYZ_BSIZE,
            m_xorigin2: 0,
            m_yorigin1: XYZ_BSIZE,
            m_yorigin2: 0,
            m_scale_bar_size: 0.,
            m_whichbox: NOT_IN_BOX,
            m_mousemode: 0,
            m_tool_zoom: -1.,
            m_screen_changed: false,
            m_last_xsize_change: 1.,
            m_device_pixel_ratio: 0.,
            m_lock: 0,
            m_time_lock: 0,
            m_last_cache_sum: -1,
            m_xtrans1: 0,
            m_ytrans1: 0,
            m_xtrans2: 0,
            m_ytrans2: 0,
            m_hq: 0,
            m_project: 0,
            m_apply_zscale: 0,
            m_thickness: 1,
            m_last_thickness: -1,
            m_last_xoffset1: 0,
            m_last_yoffset1: 0,
            m_last_width1: 0,
            m_last_height1: 0,
            m_last_xoffset2: 0,
            m_last_yoffset2: 0,
            m_last_width2: 0,
            m_last_height2: 0,
            m_last_tile_cache_ind: -1,
            m_draw_current_only: 0,
            m_xz_fraction: zscaled / (view.xsize as f32 + zscaled).max(1.),
            m_yz_fraction: zscaled / (view.ysize as f32 + zscaled).max(1.),
            m_xlock: 0,
            m_ylock: 0,
            m_zlock: 0,
            m_time_drawn: -1,
            m_tool_time: 0,
            m_doing_draw: false,
            m_toggle_states: [0; MAX_XYZ_TOGGLES],
            m_displayed_axis_location: [-100; NUM_AXIS],
            m_ctrl_pressed: false,
            m_closed: false,
        }
    }
    pub fn xxyz_open(view: XyzViewState) -> Result<Self, i32> {
        if view.xsize <= 0 || view.ysize <= 0 || view.zsize <= 0 {
            Err(-1)
        } else {
            Ok(Self::new(view))
        }
    }
    /// `setFontDependentWidths()` source method.
    pub fn set_font_dependent_widths(&mut self, native: &mut dyn XyzNativeBoundary) {
        native.set_help_button_width()
    }
    /// `changeEvent()` source method.
    pub fn change_event(&mut self, event: XyzChangeEvent, native: &mut dyn XyzNativeBoundary) {
        native.check_and_set_mac_menu();
        if event == XyzChangeEvent::FontChange {
            self.set_font_dependent_widths(native);
        }
    }
    /// `SetCursor()` source method.
    pub fn set_cursor(&mut self, mode: i32, set_anyway: bool, native: &mut dyn XyzNativeBoundary) {
        if self.m_mousemode == mode && !set_anyway {
            return;
        }
        native.set_model_cursor(mode == IMOD_MMODEL);
        self.m_mousemode = mode;
    }
    /// `setMaxAxis()` source method.
    pub fn set_max_axis(&mut self, axis: i32, max: i32, native: &mut dyn XyzNativeBoundary) {
        native.set_axis_slider_range(axis, 0, max - 1);
        let slider_width = max.clamp(MIN_SLIDER_WIDTH, MAX_SLIDER_WIDTH);
        native.set_axis_slider_width(
            axis,
            slider_width + native.axis_slider_minimum_width(axis) + 5,
        );
    }
    /// `toolbarMenuEvent()` source method.
    pub fn toolbar_menu_event(&mut self, native: &mut dyn XyzNativeBoundary) {
        if let Some(index) = native.toolbar_popup_selection() {
            self.context_menu_hit(index, native);
        }
    }
    /// `contextMenuHit()` source method.
    pub fn context_menu_hit(&mut self, index: i32, native: &mut dyn XyzNativeBoundary) {
        if let Some((key, modifiers)) = native.popup_key_event(index) {
            native.dispatch_key_press(key, modifiers);
        }
    }
    pub fn xyz_scale_bar_size(&self) -> f32 {
        self.m_scale_bar_size
    }
    pub fn allocate_dim(&self, winsize: i32, z_fraction: f32) -> (i32, i32) {
        let mut dim2 = (z_fraction * (winsize - ALL_BORDER) as f32).round() as i32;
        let mut dim1 = winsize - ALL_BORDER - dim2;
        if dim1 < 2 {
            dim2 = (dim2 - (2 - dim1)).max(2);
            dim1 = 2;
        } else if dim2 < 2 {
            dim1 = (2 - dim2).max(2);
            dim2 = 2;
        }
        (dim1, dim2)
    }
    pub fn get_ci_images(&mut self) {
        (self.m_win_xdim1, self.m_win_xdim2) = self.allocate_dim(self.m_winx, self.m_xz_fraction);
        (self.m_win_ydim1, self.m_win_ydim2) = self.allocate_dim(self.m_winy, self.m_yz_fraction);
        if self.m_win_xdim2 > self.m_win_ydim2 {
            self.m_win_xdim1 += self.m_win_xdim2 - self.m_win_ydim2;
            self.m_win_xdim2 = self.m_win_ydim2;
        } else if self.m_win_ydim2 > self.m_win_xdim2 {
            self.m_win_ydim1 += self.m_win_ydim2 - self.m_win_xdim2;
            self.m_win_ydim2 = self.m_win_xdim2;
        }
        self.m_xorigin1 = XYZ_BSIZE;
        self.m_xorigin2 = XYZ_BSIZE + XYZ_GSIZE + self.m_win_xdim1;
        self.m_yorigin1 = XYZ_BSIZE;
        self.m_yorigin2 = XYZ_BSIZE + XYZ_GSIZE + self.m_win_ydim1;
    }
    pub fn get_location(&self, view: XyzViewState) -> (i32, i32, i32) {
        if self.m_lock != 0 {
            (self.m_xlock, self.m_ylock, self.m_zlock)
        } else {
            (view.xmouse as i32, view.ymouse as i32, view.zmouse as i32)
        }
    }
    pub fn set_location(
        &mut self,
        view: XyzViewState,
        x: i32,
        y: i32,
        z: i32,
    ) -> (Option<i32>, Option<i32>, Option<i32>) {
        let mut out = (None, None, None);
        if x != NOTNEW {
            self.m_xlock = x.clamp(0, view.xsize - 1);
            if self.m_lock == 0 {
                out.0 = Some(self.m_xlock);
            }
        }
        if y != NOTNEW {
            self.m_ylock = y.clamp(0, view.ysize - 1);
            if self.m_lock == 0 {
                out.1 = Some(self.m_ylock);
            }
        }
        if z != NOTNEW {
            self.m_zlock = z.clamp(0, view.zsize - 1);
            if self.m_lock == 0 {
                out.2 = Some(self.m_zlock);
            }
        }
        out
    }
    /// `XyzWindow::Getxyz`.
    pub fn getxyz(&self, view: XyzViewState, x: i32, y: i32) -> (i32, f32, f32, i32) {
        let y = self.m_winy - y;
        let (cx, cy, cz) = self.get_location(view);
        let (mut mx, mut my, mut mz) = (cx as f32 + 0.5, cy as f32 + 0.5, cz);
        let scale = 1.0 / self.m_zoom;
        let zscale = scale
            / if self.m_apply_zscale != 0 {
                view.zscale
            } else {
                1.0
            };
        if x >= self.m_xorigin1
            && x <= self.m_xorigin1 + self.m_win_xdim1
            && y >= self.m_yorigin1
            && y <= self.m_yorigin1 + self.m_win_ydim1
        {
            mx = (x as f32 + 0.5 - self.m_xwoffset1 as f32) * scale;
            my = (y as f32 + 0.5 - self.m_ywoffset1 as f32) * scale;
            return (Z_SLICE_BOX, mx, my, mz);
        }
        if x >= self.m_xorigin1
            && x <= self.m_xorigin1 + self.m_win_xdim1
            && y >= self.m_yorigin2
            && y <= self.m_yorigin2 + self.m_win_ydim2
        {
            mx = (x as f32 + 0.5 - self.m_xwoffset1 as f32) * scale;
            mz = ((y - self.m_ywoffset2) as f32 * zscale) as i32;
            return (Y_SLICE_BOX, mx, my, mz);
        }
        if x >= self.m_xorigin2
            && x <= self.m_xorigin2 + self.m_win_xdim2
            && y >= self.m_yorigin1
            && y <= self.m_yorigin1 + self.m_win_ydim1
        {
            my = (y as f32 + 0.5 - self.m_ywoffset1 as f32) * scale;
            mz = ((x - self.m_xwoffset2) as f32 * zscale) as i32;
            return (X_SLICE_BOX, mx, my, mz);
        }
        if x >= self.m_xorigin2 - 1
            && x <= self.m_xorigin2 + self.m_win_xdim2 + 1
            && y >= self.m_yorigin2 - 1
            && y <= self.m_yorigin2 + self.m_win_ydim2 + 1
        {
            mz = (0.5 * (y - self.m_ywoffset2 + x - self.m_xwoffset2) as f32 * scale) as i32;
            return (Z_GADGET_BOX, mx, my, mz);
        }
        if x >= self.m_xorigin1 - 1
            && x <= self.m_xorigin1 + self.m_win_xdim1 + 1
            && y >= self.m_yorigin1 + self.m_win_ydim1
            && y <= self.m_yorigin2
        {
            mx = (x - self.m_xwoffset1) as f32 * scale;
            return (X_GADGET_BOX, mx, my, mz);
        }
        if x >= self.m_xorigin1 + self.m_win_xdim1
            && x <= self.m_xorigin2
            && y >= self.m_yorigin1 - 1
            && y <= self.m_yorigin1 + self.m_win_ydim1 + 1
        {
            my = (y - self.m_ywoffset1) as f32 * scale;
            return (Y_GADGET_BOX, mx, my, mz);
        }
        if x >= self.m_xorigin1 + self.m_win_xdim1
            && x <= self.m_xorigin2
            && y >= self.m_yorigin1 + self.m_win_ydim1
            && y <= self.m_yorigin2
        {
            return (FRACTION_BOX, mx, my, mz);
        }
        (NOT_IN_BOX, mx, my, mz)
    }
    pub fn zoom_up(&mut self, zooms: &[f64]) {
        self.step_zoom(zooms, 1);
    }
    pub fn zoom_down(&mut self, zooms: &[f64]) {
        self.step_zoom(zooms, -1);
    }
    /// `XyzWindow::stepZoom`: selects the next configured pixel zoom rather
    /// than applying an invented multiplicative increment.
    pub fn step_zoom(&mut self, zooms: &[f64], step: i32) {
        if zooms.is_empty() || step == 0 {
            return;
        }
        self.m_zoom =
            crate::imod::three_dmod::b3dgfx::b3d_step_pixel_zoom(zooms, self.m_zoom as f64, step)
                as f32;
    }
    /// Host-aware `stepZoom`.
    pub fn step_zoom_with_boundary(
        &mut self,
        zooms: &[f64],
        step: i32,
        native: &mut dyn XyzNativeBoundary,
    ) {
        native.control_priority(self.m_ctrl);
        self.step_zoom(zooms, step);
        self.draw(native);
    }
    pub fn entered_zoom(&mut self, zoom: f32) {
        self.m_zoom = zoom.max(0.01);
    }
    /// Host-aware `enteredZoom`.
    pub fn entered_zoom_with_boundary(&mut self, zoom: f32, native: &mut dyn XyzNativeBoundary) {
        native.control_priority(self.m_ctrl);
        self.entered_zoom(zoom);
        self.draw(native);
        native.focus_window();
    }
    pub fn set_zoom_text(&self) -> String {
        let mut s = format!("{:.4}", self.m_zoom);
        if s.ends_with("00") {
            s.truncate(s.len() - 2);
        }
        s
    }
    pub fn slider_changed(
        &mut self,
        which: usize,
        value: i32,
        dragging: bool,
        hot_slider: bool,
    ) -> bool {
        if !dragging || hot_slider {
            self.entered_axis_location(which, value);
            true
        } else {
            self.m_displayed_axis_location[which] = value;
            false
        }
    }
    /// Host-aware `sliderChanged`.
    pub fn slider_changed_with_boundary(
        &mut self,
        which: usize,
        value: i32,
        dragging: bool,
        hot_slider: bool,
        view: XyzViewState,
        native: &mut dyn XyzNativeBoundary,
    ) -> bool {
        if !dragging || hot_slider {
            self.entered_axis_location_with_boundary(which, value, view, native);
            true
        } else {
            self.m_displayed_axis_location[which] = value;
            false
        }
    }
    pub fn set_slider(&mut self, which: usize, section: i32) {
        if self.m_displayed_axis_location[which] != section {
            self.m_displayed_axis_location[which] = section;
        }
    }
    pub fn entered_axis_location(&mut self, which: usize, value: i32) {
        match which {
            0 => {
                self.m_xlock = value;
            }
            1 => {
                self.m_ylock = value;
            }
            2 => {
                self.m_zlock = value;
            }
            _ => {}
        }
    }
    /// Host-aware `enteredAxisLocation` with source `setLocation` clamping,
    /// redraw selection, and focus restoration.
    pub fn entered_axis_location_with_boundary(
        &mut self,
        which: usize,
        value: i32,
        view: XyzViewState,
        native: &mut dyn XyzNativeBoundary,
    ) {
        native.control_priority(self.m_ctrl);
        let location = match which {
            0 => self.set_location(view, value, NOTNEW, NOTNEW),
            1 => self.set_location(view, NOTNEW, value, NOTNEW),
            2 => self.set_location(view, NOTNEW, NOTNEW, value),
            _ => return,
        };
        if self.m_lock != 0 {
            self.draw(native);
        } else {
            native.set_location(location.0, location.1, location.2);
            native.draw(crate::imod::three_dmod::imod::IMOD_DRAW_XYZ);
        }
        native.focus_window();
    }
    pub fn thickness_changed(&mut self, view: XyzViewState, value: i32) {
        let old = self.m_thickness;
        self.m_thickness = value;
        if !view.has_pyramid_cache && old == 1 && value > 1 {
            self.allocate_sum_temp(
                (view.xsize * view.ysize)
                    .max(view.zsize * view.ysize)
                    .max(view.xsize * view.zsize) as usize,
                (view.xsize * view.ysize) as usize,
                view.ushort_store,
            );
        } else if value == 1 && old > 1 {
            self.m_sum_temp.clear();
            self.m_fdata_xy.clear();
        }
    }
    /// Host-aware `thicknessChanged`.
    pub fn thickness_changed_with_boundary(
        &mut self,
        view: XyzViewState,
        value: i32,
        native: &mut dyn XyzNativeBoundary,
    ) {
        self.thickness_changed(view, value);
        native.focus_window();
        self.draw(native);
    }
    pub fn allocate_sum_temp(&mut self, max_size: usize, xy_size: usize, ushort: bool) {
        self.m_sum_temp = vec![0; max_size];
        self.m_fdata_xy = vec![0; xy_size * if ushort { 2 } else { 1 }];
    }
    pub fn thickness_to_use(&self, iz: i32, zsize: i32, scale: i32) -> (i32, i32, i32) {
        let use_thickness = (self.m_thickness + scale - 1) / scale;
        (
            use_thickness,
            (iz - use_thickness / 2).max(0),
            (iz + (use_thickness - use_thickness / 2 - 1)).min(zsize - 1),
        )
    }
    pub fn add_into_sum_temp(&mut self, data: &[u8], num_pix: usize, thickness: i32, ushort: bool) {
        if thickness <= 1 {
            return;
        }
        for x in 0..num_pix {
            let v = if ushort {
                u16::from_ne_bytes([data[2 * x], data[2 * x + 1]]) as i32
            } else {
                data[x] as i32
            };
            self.m_sum_temp[x] += v;
        }
    }
    pub fn average_sum_temp(&self, data: &mut [u8], num_pix: usize, num_slices: i32, ushort: bool) {
        if num_slices <= 1 {
            return;
        }
        for x in 0..num_pix {
            let v = (self.m_sum_temp[x] as f32 / num_slices as f32 + 0.5) as u16;
            if ushort {
                let b = v.to_ne_bytes();
                data[2 * x] = b[0];
                data[2 * x + 1] = b[1];
            } else {
                data[x] = v as u8;
            }
        }
    }
    pub fn fill_array_from_tiles(
        &self,
        fdata: &mut [u8],
        segments: &[FastSegment],
        start_inds: &[usize],
        ushort: bool,
        doing_yz: bool,
        istart: i32,
        iend: i32,
        jstart: i32,
        jend: i32,
    ) {
        let pix = if ushort { 2 } else { 1 };
        let fstride = if doing_yz {
            (jend - jstart) as usize
        } else {
            1
        };
        fdata.fill(0);
        for j in jstart..jend {
            let mut next = istart;
            for iseg in start_inds[j as usize]..start_inds[j as usize + 1] {
                let seg = &segments[iseg];
                if seg.x_or_y >= iend {
                    break;
                }
                if seg.x_or_y + seg.length <= istart {
                    continue;
                }
                let end = (seg.x_or_y + seg.length).min(iend);
                let start = seg.x_or_y.max(istart);
                for i in start..end {
                    let dst = if doing_yz {
                        ((i - istart) as usize * fstride + (j - jstart) as usize) * pix
                    } else {
                        ((j - jstart) as usize * (iend - istart) as usize + (i - istart) as usize)
                            * pix
                    };
                    let src = ((i - seg.x_or_y) * seg.stride) as usize * pix;
                    if src + pix <= seg.line.len() && dst + pix <= fdata.len() {
                        fdata[dst..dst + pix].copy_from_slice(&seg.line[src..src + pix]);
                    }
                }
                next = end;
            }
            let _ = next;
        }
    }
    pub fn get_subset_limits(&self) -> (i32, i32, i32, i32) {
        (
            self.m_last_xoffset1,
            self.m_last_yoffset1,
            self.m_last_width1,
            self.m_last_height1,
        )
    }
    pub fn state_toggled(&mut self, index: usize, state: i32, view: XyzViewState) {
        self.m_toggle_states[index] = state;
        match index {
            XYZ_TOGGLE_RESOL => self.m_hq = state,
            XYZ_TOGGLE_LOCKED => {
                self.m_lock = state;
                if state != 0 {
                    self.m_xlock = view.xmouse as i32;
                    self.m_ylock = view.ymouse as i32;
                    self.m_zlock = view.zmouse as i32;
                    self.m_time_lock = view.cur_time;
                }
            }
            XYZ_TOGGLE_ZSCALE => self.m_apply_zscale = state,
            XYZ_TOGGLE_TIMELOCK => self.m_time_lock = if state != 0 { view.cur_time } else { 0 },
            _ => {}
        }
    }
    /// Host-aware `XyzWindow::stateToggled`, including the source redraw
    /// points after display-affecting changes.
    pub fn state_toggled_with_boundary(
        &mut self,
        index: usize,
        state: i32,
        view: XyzViewState,
        native: &mut dyn XyzNativeBoundary,
    ) {
        native.control_priority(self.m_ctrl);
        self.state_toggled(index, state, view);
        let redraw = match index {
            XYZ_TOGGLE_RESOL | XYZ_TOGGLE_ZSCALE => true,
            XYZ_TOGGLE_LOCKED | XYZ_TOGGLE_TIMELOCK => state == 0,
            _ => false,
        };
        if redraw {
            self.draw(native);
        }
    }
    pub fn center_clicked(&mut self, view: XyzViewState) {
        let (x, y, z) = self.get_location(view);
        self.m_xtrans1 = view.xsize / 2 - x;
        self.m_ytrans1 = view.ysize / 2 - y;
        self.m_xtrans2 = view.zsize / 2 - z;
        self.m_ytrans2 = self.m_xtrans2;
    }
    /// Host-aware `centerClicked`.
    pub fn center_clicked_with_boundary(
        &mut self,
        view: XyzViewState,
        native: &mut dyn XyzNativeBoundary,
    ) {
        self.center_clicked(view);
        self.draw(native);
    }
    /// `XyzWindow::help`.
    pub fn help(&mut self, native: &mut dyn XyzNativeBoundary) {
        native.help("xyz.html#TOP");
    }
    /// `XyzWindow::fillCachePressed`.
    pub fn fill_cache_pressed(&mut self, native: &mut dyn XyzNativeBoundary) {
        native.control_priority(self.m_ctrl);
        native.cache_fill();
    }
    /// Shared tail of native point insertion and point movement.  The model
    /// edit itself belongs to the model boundary; this method preserves the
    /// source's location binding and combined redraw after that edit.
    pub fn finish_new_model_point(
        &mut self,
        view: XyzViewState,
        x: i32,
        y: i32,
        z: i32,
        native: &mut dyn XyzNativeBoundary,
    ) {
        let location = self.set_location(view, x, y, z);
        native.set_location(location.0, location.1, location.2);
        native.draw(
            crate::imod::three_dmod::imod::IMOD_DRAW_XYZ
                | crate::imod::three_dmod::imod::IMOD_DRAW_MOD,
        );
    }
    /// `B1Drag()` source method.
    pub fn b1_drag(&mut self, view: XyzViewState, x: i32, y: i32) {
        let (mut x, mut y) = (x, y);
        // `xyz.cpp:B1Drag`: keep a coordinate-gadget drag in its gutter.
        match self.m_whichbox {
            Y_GADGET_BOX => {
                let inverted = self.m_winy - y;
                if inverted < self.m_yorigin1 + 1 {
                    y = self.m_winy - (self.m_yorigin1 + 1);
                }
                if inverted > self.m_yorigin1 + self.m_win_ydim1 {
                    y = self.m_winy - (self.m_yorigin1 + self.m_win_ydim1);
                }
            }
            X_GADGET_BOX => x = x.clamp(self.m_xorigin1, self.m_xorigin1 + self.m_win_xdim1 - 1),
            Z_GADGET_BOX => {
                x = x.clamp(self.m_xorigin2, self.m_xorigin2 + self.m_win_xdim2 - 1);
                let inverted = self.m_winy - y;
                if inverted < self.m_yorigin2 + 1 {
                    y = self.m_winy - (self.m_yorigin2 + 1);
                }
                if inverted > self.m_yorigin2 + self.m_win_ydim2 {
                    y = self.m_winy - (self.m_yorigin2 + self.m_win_ydim2);
                }
            }
            _ => {}
        }
        let factor = if self.m_zoom < 1.0 {
            1.0 / self.m_zoom
        } else {
            1.0
        };
        // B3DNINT is `(int)(value + 0.5)`, not Rust's symmetric `round`.
        let delx = ((x - self.m_lmx) as f32 * factor + 0.5) as i32;
        let dely = ((y - self.m_lmy) as f32 * factor + 0.5) as i32;
        let scale = 1.0 / self.m_zoom;
        match self.m_whichbox {
            X_SLICE_BOX => {
                self.m_xtrans2 += delx;
                self.m_ytrans1 -= dely;
                self.m_ytrans2 += delx
            }
            Y_SLICE_BOX => {
                self.m_xtrans1 += delx;
                self.m_ytrans2 -= dely;
                self.m_xtrans2 -= dely
            }
            Z_SLICE_BOX => {
                self.m_xtrans1 += delx;
                self.m_ytrans1 -= dely
            }
            Z_GADGET_BOX => {
                let value = (0.5
                    * ((self.m_winy - y - 1 - self.m_ywoffset2) + (x - self.m_xwoffset2)) as f32
                    * scale) as i32;
                let (_, _, z) = self.get_location(view);
                if z != value {
                    let _ = self.set_location(view, NOTNEW, NOTNEW, value);
                }
            }
            X_GADGET_BOX => {
                let value = ((x - self.m_xwoffset1) as f32 * scale) as i32;
                let (old, _, _) = self.get_location(view);
                if old != value {
                    let _ = self.set_location(view, value, NOTNEW, NOTNEW);
                }
            }
            Y_GADGET_BOX => {
                let value = ((self.m_winy - y - 1 - self.m_ywoffset1) as f32 * scale) as i32;
                let (_, old, _) = self.get_location(view);
                if old != value {
                    let _ = self.set_location(view, NOTNEW, value, NOTNEW);
                }
            }
            FRACTION_BOX => {
                let dz = -0.5 * (x - self.m_lmx + self.m_lmy - y) as f32;
                let xf = self.m_win_xdim2 as f32 / (self.m_win_xdim1 + self.m_win_xdim2) as f32;
                let yf = self.m_win_ydim2 as f32 / (self.m_win_ydim1 + self.m_win_ydim2) as f32;
                let df = if yf / self.m_yz_fraction > xf / self.m_xz_fraction {
                    (self.m_win_ydim2 as f32 + dz)
                        / (self.m_win_ydim1 + self.m_win_ydim2) as f32
                        / yf
                } else {
                    (self.m_win_xdim2 as f32 + dz)
                        / (self.m_win_xdim1 + self.m_win_xdim2) as f32
                        / xf
                };
                let sx = view.xsize + view.zsize;
                let sy = view.ysize + view.zsize;
                self.m_xz_fraction =
                    (self.m_xz_fraction * df).clamp(2.0 / sx as f32, (sx - 2) as f32 / sx as f32);
                self.m_yz_fraction =
                    (self.m_yz_fraction * df).clamp(2.0 / sy as f32, (sy - 2) as f32 / sy as f32);
                self.get_ci_images();
            }
            _ => {}
        }
    }
    /// Host-aware `B1Drag()`.  The geometry-only entry point above remains
    /// useful to callers that own redraw dispatch; this one preserves the
    /// source's `Draw()` versus `imodDraw(IMOD_DRAW_XYZ)` split.
    pub fn b1_drag_with_boundary(
        &mut self,
        view: XyzViewState,
        x: i32,
        y: i32,
        native: &mut dyn XyzNativeBoundary,
    ) {
        let which_box = self.m_whichbox;
        // `setLocation` always updates these cached coordinates, whereas an
        // unlocked `get_location(view)` reads the caller's pre-draw snapshot.
        let before = (self.m_xlock, self.m_ylock, self.m_zlock);
        self.b1_drag(view, x, y);
        let location_changed = (self.m_xlock, self.m_ylock, self.m_zlock) != before;
        // In the unlocked source path `setLocation` writes only the coordinate
        // selected by its gadget into `ImodView` before issuing `imodDraw`.
        if location_changed && self.m_lock == 0 {
            match which_box {
                X_GADGET_BOX => native.set_location(Some(self.m_xlock), None, None),
                Y_GADGET_BOX => native.set_location(None, Some(self.m_ylock), None),
                Z_GADGET_BOX => native.set_location(None, None, Some(self.m_zlock)),
                _ => {}
            }
        }
        match which_box {
            X_SLICE_BOX | Y_SLICE_BOX | Z_SLICE_BOX | FRACTION_BOX => self.draw(native),
            X_GADGET_BOX | Y_GADGET_BOX | Z_GADGET_BOX if location_changed => {
                if self.m_lock != 0 {
                    self.draw(native);
                } else {
                    native.draw(crate::imod::three_dmod::imod::IMOD_DRAW_XYZ);
                }
            }
            _ => {}
        }
    }
    pub fn step_time(&mut self, view: XyzViewState, step: i32) -> Option<i32> {
        if self.m_time_lock != 0 {
            self.m_time_lock = (self.m_time_lock + step).clamp(1, view.num_times);
            Some(self.m_time_lock)
        } else {
            None
        }
    }
    /// Host-aware `XyzWindow::stepTime`.  The unlocked source path stops the
    /// XYZ movie axes before handing time movement to the shared input path.
    pub fn step_time_with_boundary(
        &mut self,
        view: XyzViewState,
        step: i32,
        native: &mut dyn XyzNativeBoundary,
    ) {
        native.control_priority(self.m_ctrl);
        if self.m_time_lock != 0 {
            self.m_time_lock = (self.m_time_lock + step).clamp(1, view.num_times);
            self.draw(native);
        } else {
            native.movie_xyzt(0, 0, 0, 0);
            native.next_time(step > 0);
        }
    }
    pub fn time_back(&mut self, view: XyzViewState) -> Option<i32> {
        self.step_time(view, -1)
    }
    pub fn time_forward(&mut self, view: XyzViewState) -> Option<i32> {
        self.step_time(view, 1)
    }
    /// `xyzDraw_cb()` source callback.
    pub fn draw(&mut self, n: &mut dyn XyzNativeBoundary) {
        if self.m_doing_draw {
            return;
        }
        self.m_doing_draw = true;
        n.update_gl();
        self.m_doing_draw = false;
    }
    pub fn draw_image(&mut self, n: &mut dyn XyzNativeBoundary) {
        n.draw_image();
    }
    pub fn draw_model(&mut self, n: &mut dyn XyzNativeBoundary) {
        n.draw_model();
    }
    pub fn draw_current_lines(&mut self, n: &mut dyn XyzNativeBoundary) {
        let view = n.view_state();
        let (cx, cy, cz) = self.get_location(view);
        let zoom_z = self.m_zoom
            * if self.m_apply_zscale != 0 {
                view.zscale
            } else {
                1.
            };
        let hlen = GRAB_LENGTH / 2;
        let hwidth = GRAB_WIDTH / 2;
        let xline_y = self.m_yorigin2 - XYZ_GSIZE / 2 - 1;
        let yline_x = self.m_xorigin2 - XYZ_GSIZE / 2 - 1;
        n.set_line_width(1);

        n.set_color_index(n.foreground_color());
        let cenx = (self.m_xwoffset2 as f32 + zoom_z * (cz as f32 + 0.5)) as i32;
        let ceny = (self.m_ywoffset2 as f32 + zoom_z * (cz as f32 + 0.5)) as i32;
        let cenxlim = cenx.clamp(self.m_xorigin2 - 1, self.m_xorigin2 + self.m_win_xdim2 + 1);
        let cenylim = ceny.clamp(self.m_yorigin2, self.m_yorigin2 + self.m_win_ydim2);
        if ceny == cenylim {
            n.draw_line(
                (self.m_xorigin2, ceny),
                (self.m_xorigin2 + self.m_win_xdim2, ceny),
            );
        }
        if cenx == cenxlim {
            n.draw_line(
                (cenx, self.m_yorigin2),
                (cenx, self.m_yorigin2 + self.m_win_ydim2),
            );
        }
        if cenx == cenxlim && ceny == cenylim {
            n.draw_filled_rectangle(cenx - hlen, ceny - hlen, GRAB_LENGTH, GRAB_LENGTH);
        }
        n.draw_rectangle(
            self.m_xorigin1 - 1,
            self.m_yorigin1 - 1,
            self.m_win_xdim1 + 1,
            self.m_win_ydim1 + 1,
        );

        n.set_color_index(n.begin_point_color());
        let cenx = (self.m_xwoffset1 as f32 + self.m_zoom * (cx as f32 + 0.5)) as i32;
        let cenxlim = cenx.clamp(self.m_xorigin1 - 1, self.m_xorigin1 + self.m_win_xdim1);
        n.draw_line(
            (cenxlim, xline_y),
            (self.m_xorigin2 + self.m_win_xdim2, xline_y),
        );
        if cenx == cenxlim {
            n.draw_filled_rectangle(cenx - hwidth, xline_y - hlen, GRAB_WIDTH, GRAB_LENGTH);
        }
        n.draw_rectangle(
            self.m_xorigin2 - 1,
            self.m_yorigin1 - 1,
            self.m_win_xdim2 + 1,
            self.m_win_ydim1 + 1,
        );

        n.set_color_index(n.end_point_color());
        let ceny = (self.m_ywoffset1 as f32 + self.m_zoom * (cy as f32 + 0.5)) as i32;
        let cenylim = ceny.clamp(self.m_yorigin1 - 1, self.m_yorigin1 + self.m_win_ydim1);
        n.draw_line(
            (yline_x, self.m_yorigin2 + self.m_win_ydim2),
            (yline_x, cenylim),
        );
        if ceny == cenylim {
            n.draw_filled_rectangle(yline_x - hlen, ceny - hwidth, GRAB_LENGTH, GRAB_WIDTH);
        }
        n.draw_rectangle(
            self.m_xorigin1 - 1,
            self.m_yorigin2 - 1,
            self.m_win_xdim1 + 1,
            self.m_win_ydim2 + 1,
        );
        n.set_custom_ghost_color(64, 192, 255);
        n.draw_filled_rectangle(
            yline_x - hlen - 2,
            xline_y - hlen - 2,
            GRAB_LENGTH + 4,
            GRAB_LENGTH + 4,
        );
        n.reset_ghost_color();
    }
    pub fn draw_current_point(&mut self, n: &mut dyn XyzNativeBoundary) {
        n.draw_current_point();
    }
    /// `XyzWindow::DrawAuto`.  The only native drawing implementation is
    /// behind `FIX_xyzDrawAuto_BUG`, which is not enabled upstream; the
    /// observable source behavior is therefore deliberately no-op.
    pub fn draw_auto(&mut self, _n: &mut dyn XyzNativeBoundary) {}
    /// `XyzWindow::DrawGhost` is explicitly nonfunctional in the native
    /// source as well.
    pub fn draw_ghost(&mut self, _n: &mut dyn XyzNativeBoundary) {}
    pub fn draw_contour(&mut self, _n: &mut dyn XyzNativeBoundary) {}
    pub fn draw_sym_proj(&mut self, _n: &mut dyn XyzNativeBoundary) {}
    pub fn draw_scat_sym_all_spheres(&mut self, _n: &mut dyn XyzNativeBoundary) {}
    pub fn draw_tools(&mut self, n: &mut dyn XyzNativeBoundary) {
        let view = n.view_state();
        let (cx, cy, cz) = self.get_location(view);
        if self.m_tool_max_x != view.xsize {
            self.m_tool_max_x = view.xsize;
            self.set_max_axis(0, self.m_tool_max_x, n);
        }
        if self.m_tool_max_y != view.ysize {
            self.m_tool_max_y = view.ysize;
            self.set_max_axis(1, self.m_tool_max_y, n);
        }
        if self.m_tool_max_z != view.zsize {
            self.m_tool_max_z = view.zsize;
            self.set_max_axis(2, self.m_tool_max_z, n);
        }
        if self.m_tool_zoom != self.m_zoom {
            if self.m_tool_zoom < 0. {
                self.m_tool_zoom -= 1.;
            }
            if self.m_tool_zoom <= -4. || self.m_tool_zoom > -0.9 {
                self.m_tool_zoom = self.m_zoom;
            }
            n.set_zoom_text(&self.set_zoom_text());
        }
        if view.num_times != 0 {
            let time = if self.m_time_lock != 0 {
                self.m_time_lock
            } else {
                view.cur_time
            };
            if self.m_tool_time != time {
                self.m_tool_time = time;
                let label = n.time_index_label(time);
                self.set_time_label(time, &label, n);
            }
        }
        for (axis, location) in [(0, cx), (1, cy), (2, cz)] {
            if self.m_displayed_axis_location[axis] != location {
                n.set_axis_slider_value(axis, location);
                self.m_displayed_axis_location[axis] = location;
            }
        }
    }
    /// `xyzClose_cb()` source callback.
    pub fn close_event(&mut self, n: &mut dyn XyzNativeBoundary) {
        n.remove_control(self.m_ctrl);
        n.remove_dialog();
        n.movie_xyzt(0, 0, 0, 0);
        self.m_fdata_xy.clear();
        self.m_fdata_xz.clear();
        self.m_fdata_yz.clear();
        self.m_sum_temp.clear();
        n.set_xyz_apply_zscale(self.m_apply_zscale);
        self.m_closed = true;
        n.close_native();
    }
}

/// `XyzGL` from `xxyz.h`; live Qt event dispatch remains a native endpoint.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct XyzGl {
    pub m_closing: bool,
    pub m_init_width: i32,
    pub m_init_height: i32,
    pub m_first_draw: i32,
    pub m_timer_id: i32,
    pub m_mouse_pressed: bool,
    pub m_defer_screen_zoom_change: bool,
    pub m_scheduled_resize: bool,
    pub m_scheduled_bump: i32,
    /// Source file-static `xyzShowSlice`.
    pub m_show_slice: bool,
}

/// Host action selected by `XyzGL::timerEvent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XyzTimerAction {
    Resize(i32, i32),
    Redraw,
}
impl XyzGl {
    /// `mouseMoveEvent()` source method.
    pub fn mouse_move_event(
        &mut self,
        win: &mut XyzWindow,
        x: i32,
        y: i32,
        button1: bool,
        button2: bool,
        button3: bool,
        native: &mut dyn XyzNativeBoundary,
    ) {
        if self.m_closing {
            return;
        }
        if native.pixel_view_open() {
            let (which_box, mouse_x, mouse_y, mouse_z) = win.getxyz(native.view_state(), x, y);
            if which_box != NOT_IN_BOX && which_box <= Z_SLICE_BOX {
                native.pixel_view_new_mouse_position(mouse_x, mouse_y, mouse_z);
            }
        }
        if !self.m_mouse_pressed {
            return;
        }
        if button1 && !button2 && !button3 {
            win.b1_drag_with_boundary(native.view_state(), x, y, native);
        }
        if !button1 && button2 && !button3 {
            native.xyz_b2_drag(x, y);
        }
        if !button1 && !button2 && button3 {
            native.xyz_b3_drag(x, y);
        }
        win.m_lmx = x;
        win.m_lmy = y;
    }
    /// `wheelEvent()` source method.
    pub fn wheel_event(
        &mut self,
        wheel_for_size: bool,
        zoom: f32,
        native: &mut dyn XyzNativeBoundary,
    ) {
        if wheel_for_size {
            native.wheel_change_point_size(zoom);
        }
    }
    /// `mousePressEvent()` source method.
    pub fn mouse_press_event(&mut self) {
        self.m_mouse_pressed = true;
    }
    /// Host-aware `mousePressEvent`, after the host resolves preference-based
    /// physical button mappings into source button 1/2/3 booleans.
    pub fn mouse_press_event_with_boundary(
        &mut self,
        win: &mut XyzWindow,
        x: i32,
        y: i32,
        pressed_button: i32,
        button1: bool,
        button2: bool,
        button3: bool,
        native: &mut dyn XyzNativeBoundary,
    ) {
        self.m_mouse_pressed = true;
        native.control_priority(win.m_ctrl);
        if pressed_button == 1 && !button2 && !button3 {
            win.m_whichbox = win.getxyz(native.view_state(), x, y).0;
        } else if pressed_button == 2 && !button1 && !button3 {
            win.b2_press(x, y, native);
        } else if pressed_button == 3 && !button1 && !button2 {
            win.b3_press(x, y, native);
        }
        win.m_lmx = x;
        win.m_lmy = y;
        win.m_first_mx = x;
        win.m_first_my = y;
    }
    /// `mouseReleaseEvent()` source method.
    pub fn mouse_release_event(
        &mut self,
        primary: bool,
        x: i32,
        y: i32,
        native: &mut dyn XyzNativeBoundary,
    ) {
        self.m_mouse_pressed = false;
        if primary {
            native.xyz_b1_press(x, y);
        }
    }
    /// Host-aware `mouseReleaseEvent`; a primary click is dispatched only
    /// when the source `but1downt.elapsed()` is at most 250 milliseconds.
    pub fn mouse_release_event_with_duration(
        &mut self,
        win: &mut XyzWindow,
        primary: bool,
        elapsed_ms: i32,
        x: i32,
        y: i32,
        native: &mut dyn XyzNativeBoundary,
    ) {
        self.m_mouse_pressed = false;
        if primary && elapsed_ms <= 250 {
            win.b1_press(x, y, native);
        }
    }
    /// `XyzGL()` source constructor.
    pub fn new() -> Self {
        Self {
            m_first_draw: 3,
            m_scheduled_bump: 2,
            ..Default::default()
        }
    }
    pub fn schedule_resize(&mut self) {
        self.m_scheduled_resize = true;
    }
    pub fn cancel_resize(&mut self) {
        self.m_timer_id = 0;
        self.m_scheduled_resize = false;
    }
    /// State portion of native `XyzGL::timerEvent`.  The GUI host performs
    /// the returned resize or GL update, retaining timer ownership itself.
    pub fn timer_event(
        &mut self,
        loop_started: bool,
        window_width: i32,
        window_height: i32,
    ) -> Option<XyzTimerAction> {
        if !loop_started || (self.m_init_width == 0 && self.m_init_height == 0) {
            return None;
        }
        if self.m_scheduled_resize {
            self.cancel_resize();
            let action =
                XyzTimerAction::Resize(window_width + self.m_scheduled_bump, window_height);
            self.m_scheduled_bump = -self.m_scheduled_bump;
            return Some(action);
        }
        if self.m_first_draw < 2 {
            self.m_timer_id = 0;
        }
        let action = if self.m_first_draw > 0 {
            XyzTimerAction::Resize(
                self.m_init_width + self.m_first_draw - 1,
                self.m_init_height,
            )
        } else {
            XyzTimerAction::Redraw
        };
        self.m_first_draw = (self.m_first_draw - 1).max(0);
        Some(action)
    }
    pub fn resize_gl(&mut self, win: &mut XyzWindow, width: i32, height: i32) {
        win.m_winx = width;
        win.m_winy = height;
        win.get_ci_images();
        win.m_exposed = 1;
    }
    /// Host-aware `XyzGL::resizeGL` entry point.  DPI/viewport ownership
    /// remains with the host, while the source control-priority ordering is
    /// retained before the translated size/state update.
    pub fn resize_gl_with_boundary(
        &mut self,
        win: &mut XyzWindow,
        width: i32,
        height: i32,
        native: &mut dyn XyzNativeBoundary,
    ) {
        native.control_priority(win.m_ctrl);
        self.resize_gl(win, width, height);
    }
    pub fn paint_gl(&mut self, win: &mut XyzWindow, n: &mut dyn XyzNativeBoundary) {
        if win.m_exposed == 0 || self.m_closing {
            return;
        }
        win.draw_image(n);
        win.draw_model(n);
        win.draw_current_lines(n);
        win.draw_current_point(n);
        win.draw_auto(n);
        if self.m_show_slice {
            let s = n.view_state().slice_overlay;
            let zoom = win.m_zoom;
            let first = (win.m_xwoffset1, win.m_ywoffset1);
            let second = (win.m_xwoffset2, win.m_ywoffset2);
            n.draw_slice_overlay_line(
                (
                    win.m_xorigin1,
                    win.m_yorigin1,
                    win.m_win_xdim1,
                    win.m_win_ydim1,
                ),
                (
                    (first.0 as f32 + s.zx1 * zoom) as i32,
                    (first.1 as f32 + s.zy1 * zoom) as i32,
                    (first.0 as f32 + s.zx2 * zoom) as i32,
                    (first.1 as f32 + s.zy2 * zoom) as i32,
                ),
            );
            n.draw_slice_overlay_line(
                (
                    win.m_xorigin1,
                    win.m_yorigin2,
                    win.m_win_xdim1,
                    win.m_win_ydim2,
                ),
                (
                    (first.0 as f32 + s.yx1 * zoom) as i32,
                    (second.1 as f32 + s.yz1 * zoom) as i32,
                    (first.0 as f32 + s.yx2 * zoom) as i32,
                    (second.1 as f32 + s.yz2 * zoom) as i32,
                ),
            );
            n.draw_slice_overlay_line(
                (
                    win.m_xorigin2,
                    win.m_yorigin1,
                    win.m_win_xdim2,
                    win.m_win_ydim1,
                ),
                (
                    (second.0 as f32 + s.xz1 * zoom) as i32,
                    (first.1 as f32 + s.xy1 * zoom) as i32,
                    (second.0 as f32 + s.xz2 * zoom) as i32,
                    (first.1 as f32 + s.xy2 * zoom) as i32,
                ),
            );
            n.reset_xyz_viewport(win.m_winx, win.m_winy);
            self.m_show_slice = false;
        }
        win.m_scale_bar_size =
            n.scale_bar_draw(win.m_winx, win.m_winy, win.m_zoom, win.m_device_pixel_ratio);
        win.draw_tools(n);
    }
}

/// Rust-owned equivalent of the source dialog manager's XYZ window list and
/// its `pixelViewOpen`/`insertDown` statics.  The GUI host owns the registry;
/// it contains translated window state rather than toolkit window handles.
#[derive(Default)]
pub struct XyzRegistry {
    /// Windows ordered by control priority, with the first one being the
    /// source `getTopXYZ()` result.
    pub windows: Vec<Box<XyzWindow>>,
    pub pixel_view_open: bool,
    pub insert_down: bool,
}

/// `xyzPixelViewState`.
pub fn xyz_pixel_view_state(
    registry: &mut XyzRegistry,
    state: bool,
    native: &mut dyn XyzNativeBoundary,
) {
    registry.pixel_view_open = state;
    let mouse_tracking = state || registry.insert_down;
    for _xyz in &mut registry.windows {
        native.set_mouse_tracking(mouse_tracking);
    }
}

/// `xyzScaleBarSize`.
pub fn xyz_scale_bar_size(registry: &XyzRegistry) -> f32 {
    get_top_xyz(registry).map_or(-1., XyzWindow::xyz_scale_bar_size)
}

/// `getTopXYZ`.
pub fn get_top_xyz(registry: &XyzRegistry) -> Option<&XyzWindow> {
    registry.windows.first().map(Box::as_ref)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct TimeBoundary {
        ops: Vec<&'static str>,
        locations: Vec<(Option<i32>, Option<i32>, Option<i32>)>,
        draws: Vec<i32>,
        mouse_tracking: Vec<bool>,
        key_moves: Vec<i32>,
        slice_overlay: XyzSliceOverlay,
        slice_lines: Vec<((i32, i32, i32, i32), (i32, i32, i32, i32))>,
        viewport_resets: Vec<(i32, i32)>,
        pixel_open: bool,
        pixel_positions: Vec<(f32, f32, i32)>,
        colors: Vec<i32>,
        lines: Vec<((i32, i32), (i32, i32))>,
        rectangles: Vec<(i32, i32, i32, i32)>,
        filled_rectangles: Vec<(i32, i32, i32, i32)>,
        slider_values: Vec<(usize, i32)>,
        zoom_texts: Vec<String>,
        time_labels: Vec<(String, String)>,
        auto_draws: usize,
    }
    impl XyzNativeBoundary for TimeBoundary {
        fn xyz_b1_model_action(&mut self, _: i32, _: i32) {
            self.ops.push("b1-press");
        }
        fn xyz_b2_model_action(&mut self, _: i32, _: i32) {
            self.ops.push("b2-press");
        }
        fn xyz_b2_drag(&mut self, _: i32, _: i32) {
            self.ops.push("b2-drag");
        }
        fn focus_window(&mut self) {
            self.ops.push("focus");
        }
        fn view_state(&self) -> XyzViewState {
            XyzViewState {
                slice_overlay: self.slice_overlay,
                ..view()
            }
        }
        fn set_location(&mut self, x: Option<i32>, y: Option<i32>, z: Option<i32>) {
            self.locations.push((x, y, z));
        }
        fn draw(&mut self, flags: i32) {
            self.draws.push(flags);
        }
        fn update_gl(&mut self) {
            self.ops.push("draw");
        }
        fn control_priority(&mut self, _: i32) {
            self.ops.push("priority");
        }
        fn remove_control(&mut self, _: i32) {
            self.ops.push("remove-control");
        }
        fn remove_dialog(&mut self) {
            self.ops.push("remove-dialog");
        }
        fn set_xyz_apply_zscale(&mut self, _: i32) {
            self.ops.push("save-zscale");
        }
        fn movie_xyzt(&mut self, _: i32, _: i32, _: i32, _: i32) {
            self.ops.push("stop");
        }
        fn next_time(&mut self, forward: bool) {
            self.ops.push(if forward { "next" } else { "previous" });
        }
        fn cache_fill(&mut self) {
            self.ops.push("cache");
        }
        fn help(&mut self, _: &str) {
            self.ops.push("help");
        }
        fn draw_image(&mut self) {}
        fn draw_model(&mut self) {}
        fn draw_current_lines(&mut self) {}
        fn draw_current_point(&mut self) {}
        fn draw_auto(&mut self) {
            self.auto_draws += 1;
        }
        fn draw_tools(&mut self) {}
        fn scale_bar_draw(&mut self, _: i32, _: i32, _: f32, _: f32) -> f32 {
            0.
        }
        fn set_mouse_tracking(&mut self, enabled: bool) {
            self.mouse_tracking.push(enabled);
        }
        fn grab_keyboard(&mut self) {
            self.ops.push("grab-keyboard");
        }
        fn release_keyboard(&mut self) {
            self.ops.push("release-keyboard");
        }
        fn grab_mouse(&mut self) {
            self.ops.push("grab-mouse");
        }
        fn release_mouse(&mut self) {
            self.ops.push("release-mouse");
        }
        fn draw_slice_overlay_line(
            &mut self,
            viewport: (i32, i32, i32, i32),
            line: (i32, i32, i32, i32),
        ) {
            self.slice_lines.push((viewport, line));
        }
        fn reset_xyz_viewport(&mut self, width: i32, height: i32) {
            self.viewport_resets.push((width, height));
        }
        fn pixel_view_open(&self) -> bool {
            self.pixel_open
        }
        fn pixel_view_new_mouse_position(&mut self, x: f32, y: f32, z: i32) {
            self.pixel_positions.push((x, y, z));
        }
        fn foreground_color(&self) -> i32 {
            10
        }
        fn begin_point_color(&self) -> i32 {
            20
        }
        fn end_point_color(&self) -> i32 {
            30
        }
        fn set_color_index(&mut self, color: i32) {
            self.colors.push(color);
        }
        fn set_axis_slider_value(&mut self, axis: usize, value: i32) {
            self.slider_values.push((axis, value));
        }
        fn set_zoom_text(&mut self, text: &str) {
            self.zoom_texts.push(text.into());
        }
        fn time_index_label(&self, time: i32) -> String {
            format!("T{time}")
        }
        fn set_time_label(&mut self, number: &str, label: &str) {
            self.time_labels.push((number.into(), label.into()));
        }
        fn draw_line(&mut self, from: (i32, i32), to: (i32, i32)) {
            self.lines.push((from, to));
        }
        fn draw_rectangle(&mut self, x: i32, y: i32, width: i32, height: i32) {
            self.rectangles.push((x, y, width, height));
        }
        fn draw_filled_rectangle(&mut self, x: i32, y: i32, width: i32, height: i32) {
            self.filled_rectangles.push((x, y, width, height));
        }
        fn input_next_y(&mut self) {
            self.ops.push("next-y");
        }
        fn input_prev_y(&mut self) {
            self.ops.push("prev-y");
        }
        fn input_next_x(&mut self) {
            self.ops.push("next-x");
        }
        fn input_prev_x(&mut self) {
            self.ops.push("prev-x");
        }
        fn input_page_up_or_down(&mut self, direction: i32) {
            self.ops.push(if direction > 0 {
                "page-up"
            } else {
                "page-down"
            });
        }
        fn input_key_point_move(&mut self, key: i32) {
            self.key_moves.push(key);
        }
        fn redraw_native(&mut self) {}
        fn close_native(&mut self) {}
    }
    fn view() -> XyzViewState {
        XyzViewState {
            xsize: 100,
            ysize: 80,
            zsize: 40,
            zscale: 2.,
            num_times: 6,
            cur_time: 1,
            ..Default::default()
        }
    }
    #[test]
    fn allocation_and_coordinates_follow_xyz() {
        let mut w = XyzWindow::new(view());
        w.m_winx = 300;
        w.m_winy = 240;
        w.get_ci_images();
        assert_eq!(w.m_xorigin1, XYZ_BSIZE);
        assert!(w.m_win_xdim1 >= 2 && w.m_win_xdim2 >= 2);
        w.m_zoom = 1.0;
        w.m_xwoffset1 = w.m_xorigin1;
        w.m_ywoffset1 = w.m_yorigin1;
        let (b, x, y, _) = w.getxyz(view(), w.m_xorigin1, w.m_winy - w.m_yorigin1);
        assert_eq!(b, Z_SLICE_BOX);
        assert_eq!((x, y), (0.5, 0.5));
    }
    #[test]
    fn pixel_view_updates_every_xyz_window_and_preserves_insert_tracking() {
        let mut registry = XyzRegistry {
            windows: vec![
                Box::new(XyzWindow::new(view())),
                Box::new(XyzWindow::new(view())),
            ],
            ..Default::default()
        };
        registry.windows[0].m_scale_bar_size = 12.5;
        let mut native = TimeBoundary::default();

        xyz_pixel_view_state(&mut registry, true, &mut native);
        assert!(registry.pixel_view_open);
        assert_eq!(native.mouse_tracking, vec![true, true]);
        assert_eq!(xyz_scale_bar_size(&registry), 12.5);
        assert!(get_top_xyz(&registry).is_some());

        native.mouse_tracking.clear();
        registry.insert_down = true;
        xyz_pixel_view_state(&mut registry, false, &mut native);
        assert_eq!(native.mouse_tracking, vec![true, true]);

        let empty = XyzRegistry::default();
        assert_eq!(xyz_scale_bar_size(&empty), -1.);
        assert!(get_top_xyz(&empty).is_none());
    }
    #[test]
    fn keypad_insert_follows_the_source_press_release_lifetime() {
        let mut win = XyzWindow::new(view());
        let mut registry = XyzRegistry {
            pixel_view_open: true,
            ..Default::default()
        };
        let mut native = TimeBoundary::default();

        assert!(win.keypad_insert_pressed(&mut registry, true, false, 251, 12, 13, &mut native));
        assert!(registry.insert_down);
        assert_eq!(win.m_lmx, 12);
        assert_eq!(win.m_lmy, 13);
        assert_eq!(native.mouse_tracking, vec![true]);
        assert_eq!(native.ops, vec!["grab-keyboard", "grab-mouse", "b2-press"]);

        win.m_draw_current_only = 1;
        win.key_release(&mut registry, XYZ_KEY_INSERT, true, &mut native);
        assert!(!registry.insert_down);
        assert_eq!(win.m_draw_current_only, 0);
        assert_eq!(native.mouse_tracking, vec![true, true]);
        assert_eq!(
            native.ops,
            vec![
                "grab-keyboard",
                "grab-mouse",
                "b2-press",
                "release-keyboard",
                "release-mouse",
                "draw"
            ]
        );

        assert!(!win.keypad_insert_pressed(&mut registry, false, false, 1, 0, 0, &mut native));
    }
    #[test]
    fn toolbar_entries_preserve_xyz_control_redraw_and_location_routes() {
        let mut win = XyzWindow::new(view());
        win.m_ctrl = 17;
        let mut native = TimeBoundary::default();

        win.step_zoom_with_boundary(&[1., 2.], 1, &mut native);
        assert_eq!(win.m_zoom, 2.);
        assert_eq!(native.ops, vec!["priority", "draw"]);

        native.ops.clear();
        win.entered_zoom_with_boundary(0., &mut native);
        assert_eq!(win.m_zoom, 0.01);
        assert_eq!(native.ops, vec!["priority", "draw", "focus"]);

        native.ops.clear();
        win.entered_axis_location_with_boundary(0, 999, view(), &mut native);
        assert_eq!(win.m_xlock, 99);
        assert_eq!(native.locations, vec![(Some(99), None, None)]);
        assert_eq!(
            native.draws,
            vec![crate::imod::three_dmod::imod::IMOD_DRAW_XYZ]
        );
        assert_eq!(native.ops, vec!["priority", "focus"]);

        native.ops.clear();
        native.draws.clear();
        native.locations.clear();
        assert!(!win.slider_changed_with_boundary(1, 7, true, false, view(), &mut native));
        assert_eq!(win.m_displayed_axis_location[1], 7);
        assert!(native.ops.is_empty());
        assert!(win.slider_changed_with_boundary(1, 7, false, false, view(), &mut native));
        assert_eq!(native.locations, vec![(None, Some(7), None)]);

        native.ops.clear();
        win.thickness_changed_with_boundary(view(), 3, &mut native);
        assert_eq!(win.m_thickness, 3);
        assert_eq!(native.ops, vec!["focus", "draw"]);

        native.ops.clear();
        win.m_lock = 1;
        win.m_xlock = 10;
        win.m_ylock = 20;
        win.m_zlock = 30;
        win.center_clicked_with_boundary(view(), &mut native);
        assert_eq!(
            (win.m_xtrans1, win.m_ytrans1, win.m_xtrans2, win.m_ytrans2),
            (40, 20, -10, -10)
        );
        assert_eq!(native.ops, vec!["draw"]);
    }
    #[test]
    fn xyz_navigation_remaps_plane_keys_before_input_or_locked_motion() {
        use crate::imod::three_dmod::imod_input::{KEY_PAGE_UP, KEY_RIGHT};

        let mut win = XyzWindow::new(view());
        let mut native = TimeBoundary::default();
        assert!(win.key_navigation(KEY_PAGE_UP, false, X_SLICE_BOX, false, view(), &mut native));
        assert_eq!(native.ops, vec!["priority", "next-x"]);

        native.ops.clear();
        win.m_lock = 1;
        win.m_xlock = 99;
        assert!(win.key_navigation(KEY_RIGHT, false, Z_SLICE_BOX, false, view(), &mut native));
        assert_eq!(win.m_xlock, 99);
        assert_eq!(native.ops, vec!["priority", "draw"]);

        win.m_lock = 0;
        assert!(win.key_navigation(KEY_PAGE_UP, true, X_SLICE_BOX, true, view(), &mut native));
        assert_eq!(native.key_moves, vec![KEY_RIGHT]);
    }
    #[test]
    fn xyz_time_lock_keys_clamp_and_redraw_only_when_locked() {
        let mut win = XyzWindow::new(view());
        let mut native = TimeBoundary::default();
        assert!(!win.key_time_lock_step(b'1' as i32, view(), &mut native));
        assert_eq!(native.ops, vec!["priority"]);

        win.m_time_lock = 1;
        native.ops.clear();
        assert!(win.key_time_lock_step(b'1' as i32, view(), &mut native));
        assert_eq!(win.m_time_lock, 1);
        assert_eq!(native.ops, vec!["priority", "draw"]);
        win.m_time_lock = view().num_times;
        assert!(win.key_time_lock_step(b'2' as i32, view(), &mut native));
        assert_eq!(win.m_time_lock, view().num_times);
    }
    #[test]
    fn xyz_close_releases_source_state_in_lifecycle_order() {
        let mut win = XyzWindow::new(view());
        win.m_apply_zscale = 1;
        win.m_fdata_xy = vec![1];
        win.m_fdata_xz = vec![2];
        win.m_fdata_yz = vec![3];
        win.m_sum_temp = vec![4];
        let mut native = TimeBoundary::default();

        win.close_event(&mut native);
        assert!(win.m_closed);
        assert!(win.m_fdata_xy.is_empty());
        assert!(win.m_fdata_xz.is_empty());
        assert!(win.m_fdata_yz.is_empty());
        assert!(win.m_sum_temp.is_empty());
        assert_eq!(
            native.ops,
            vec!["remove-control", "remove-dialog", "stop", "save-zscale"]
        );
    }
    #[test]
    fn xyz_paint_draws_and_consumes_the_three_slice_overlay_lines() {
        let mut win = XyzWindow::new(view());
        win.m_exposed = 1;
        win.m_winx = 200;
        win.m_winy = 100;
        win.m_zoom = 2.;
        win.m_xorigin1 = 1;
        win.m_yorigin1 = 2;
        win.m_xorigin2 = 30;
        win.m_yorigin2 = 40;
        win.m_win_xdim1 = 10;
        win.m_win_ydim1 = 11;
        win.m_win_xdim2 = 12;
        win.m_win_ydim2 = 13;
        win.m_xwoffset1 = 5;
        win.m_ywoffset1 = 7;
        win.m_xwoffset2 = 17;
        win.m_ywoffset2 = 19;
        let mut gl = XyzGl::new();
        gl.m_show_slice = true;
        let mut native = TimeBoundary {
            slice_overlay: XyzSliceOverlay {
                zx1: 1.,
                zy1: 2.,
                zx2: 3.,
                zy2: 4.,
                yx1: 5.,
                yz1: 6.,
                yx2: 7.,
                yz2: 8.,
                xz1: 9.,
                xy1: 10.,
                xz2: 11.,
                xy2: 12.,
            },
            ..Default::default()
        };

        gl.paint_gl(&mut win, &mut native);
        assert_eq!(
            native.slice_lines,
            vec![
                ((1, 2, 10, 11), (7, 11, 11, 15)),
                ((1, 40, 10, 13), (15, 31, 19, 35)),
                ((30, 2, 12, 11), (35, 27, 39, 31)),
            ]
        );
        assert_eq!(native.viewport_resets, vec![(200, 100)]);
        assert!(!gl.m_show_slice);
    }
    #[test]
    fn xyz_mouse_move_updates_pixel_view_without_a_button_press() {
        let mut win = XyzWindow::new(view());
        win.m_winy = 100;
        win.m_xorigin1 = 0;
        win.m_yorigin1 = 0;
        win.m_win_xdim1 = 100;
        win.m_win_ydim1 = 100;
        let mut gl = XyzGl::new();
        let mut native = TimeBoundary {
            pixel_open: true,
            ..Default::default()
        };

        gl.mouse_move_event(&mut win, 5, 95, false, false, false, &mut native);
        assert_eq!(native.pixel_positions, vec![(5.5, 5.5, 0)]);
    }
    #[test]
    fn xyz_press_release_delays_primary_click_and_rejects_a_drag() {
        let mut win = XyzWindow::new(view());
        win.m_winy = 100;
        win.m_xorigin1 = 0;
        win.m_yorigin1 = 0;
        win.m_win_xdim1 = 100;
        win.m_win_ydim1 = 100;
        let mut gl = XyzGl::new();
        let mut native = TimeBoundary::default();

        gl.mouse_press_event_with_boundary(&mut win, 5, 95, 1, true, false, false, &mut native);
        assert!(gl.m_mouse_pressed);
        assert_eq!(win.m_whichbox, Z_SLICE_BOX);
        assert_eq!(
            (win.m_lmx, win.m_lmy, win.m_first_mx, win.m_first_my),
            (5, 95, 5, 95)
        );
        assert_eq!(native.ops, vec!["priority"]);

        gl.mouse_release_event_with_duration(&mut win, true, 250, 5, 95, &mut native);
        assert!(!gl.m_mouse_pressed);
        assert_eq!(native.ops, vec!["priority", "b1-press"]);

        gl.mouse_press_event_with_boundary(&mut win, 5, 95, 1, true, false, false, &mut native);
        gl.mouse_release_event_with_duration(&mut win, true, 251, 5, 95, &mut native);
        assert_eq!(native.ops, vec!["priority", "b1-press", "priority"]);
    }
    #[test]
    fn xyz_current_lines_follow_native_panel_and_grab_geometry() {
        let mut win = XyzWindow::new(view());
        win.m_xorigin1 = 10;
        win.m_yorigin1 = 20;
        win.m_xorigin2 = 50;
        win.m_yorigin2 = 60;
        win.m_win_xdim1 = 30;
        win.m_win_ydim1 = 31;
        win.m_win_xdim2 = 32;
        win.m_win_ydim2 = 33;
        win.m_xwoffset1 = 11;
        win.m_ywoffset1 = 12;
        win.m_xwoffset2 = 51;
        win.m_ywoffset2 = 61;
        win.m_zoom = 2.;
        let mut native = TimeBoundary::default();

        win.draw_current_lines(&mut native);
        assert_eq!(native.colors, vec![10, 20, 30]);
        assert_eq!(
            native.lines,
            vec![
                ((50, 62), (82, 62)),
                ((52, 60), (52, 93)),
                ((12, 51), (82, 51)),
                ((41, 93), (41, 19))
            ]
        );
        assert_eq!(
            native.rectangles,
            vec![(9, 19, 31, 32), (49, 19, 33, 32), (9, 59, 31, 34)]
        );
        assert_eq!(
            native.filled_rectangles,
            vec![(49, 59, 7, 7), (11, 48, 3, 7), (36, 46, 11, 11)]
        );
    }
    #[test]
    fn xyz_draw_tools_synchronizes_zoom_time_and_axis_positions() {
        let mut win = XyzWindow::new(view());
        win.m_zoom = 2.5;
        win.m_tool_zoom = -1.;
        win.m_tool_time = -1;
        let mut native = TimeBoundary::default();
        win.draw_tools(&mut native);
        assert_eq!(native.zoom_texts, vec!["2.50"]);
        assert_eq!(native.time_labels, vec![(" (  1)".into(), "T1".into())]);
        assert_eq!(native.slider_values, vec![(0, 0), (1, 0), (2, 0)]);
    }
    #[test]
    fn thickness_and_tile_copy_follow_source() {
        let mut w = XyzWindow::new(view());
        w.allocate_sum_temp(4, 4, false);
        w.add_into_sum_temp(&[2, 4, 6, 8], 4, 2, false);
        let mut o = [0; 4];
        w.average_sum_temp(&mut o, 4, 2, false);
        assert_eq!(o, [1, 2, 3, 4]);
        let s = [FastSegment {
            x_or_y: 1,
            length: 2,
            line: vec![9, 8],
            stride: 1,
        }];
        let mut d = [7; 4];
        w.fill_array_from_tiles(&mut d, &s, &[0, 1], false, false, 0, 4, 0, 1);
        assert_eq!(d, [0, 9, 8, 0]);
    }
    #[test]
    fn thickness_allocation_does_not_treat_b3d_imax_count_as_a_size() {
        let tiny = XyzViewState {
            xsize: 1,
            ysize: 1,
            zsize: 1,
            ..Default::default()
        };
        let mut w = XyzWindow::new(tiny);
        w.thickness_changed(tiny, 2);
        assert_eq!(w.m_sum_temp.len(), 1);
        assert_eq!(w.m_fdata_xy.len(), 1);
    }
    #[test]
    fn b1_drag_updates_the_three_native_coordinate_gutters() {
        let mut w = XyzWindow::new(view());
        w.m_winy = 100;
        w.m_zoom = 1.;
        w.m_xorigin1 = 10;
        w.m_yorigin1 = 10;
        w.m_xorigin2 = 20;
        w.m_yorigin2 = 30;
        w.m_win_xdim1 = 20;
        w.m_win_xdim2 = 20;
        w.m_win_ydim1 = 20;
        w.m_win_ydim2 = 20;
        w.m_xwoffset1 = 10;
        w.m_ywoffset1 = 10;
        w.m_xwoffset2 = 20;
        w.m_ywoffset2 = 30;
        w.m_whichbox = X_GADGET_BOX;
        w.b1_drag(view(), 15, 0);
        assert_eq!(w.m_xlock, 5);
        w.m_whichbox = Y_GADGET_BOX;
        w.b1_drag(view(), 0, 85);
        assert_eq!(w.m_ylock, 4);
        w.m_whichbox = Z_GADGET_BOX;
        w.b1_drag(view(), 24, 65);
        assert_eq!(w.m_zlock, 4);
    }
    #[test]
    fn b1_drag_uses_native_rounding_for_negative_half_step() {
        let mut w = XyzWindow::new(view());
        w.m_zoom = 1.;
        w.m_whichbox = Z_SLICE_BOX;
        w.m_lmx = 5;
        w.m_lmy = 5;
        w.b1_drag(view(), 4, 5);
        // `(int)(-1. + 0.5)` is zero in C; `f32::round` would be -1.
        assert_eq!(w.m_xtrans1, 0);
        assert_eq!(w.m_ytrans1, 0);
    }
    #[test]
    fn b1_drag_uses_the_source_local_and_viewer_wide_redraw_routes() {
        let mut w = XyzWindow::new(view());
        w.m_whichbox = Z_SLICE_BOX;
        w.m_lmx = 4;
        w.m_lmy = 4;
        let mut native = TimeBoundary::default();
        w.b1_drag_with_boundary(view(), 5, 4, &mut native);
        assert_eq!(native.ops, vec!["draw"]);
        assert!(native.draws.is_empty());

        w.m_whichbox = X_GADGET_BOX;
        w.m_xorigin1 = 0;
        w.m_win_xdim1 = 20;
        w.m_xwoffset1 = 0;
        native.ops.clear();
        w.b1_drag_with_boundary(view(), 10, 0, &mut native);
        assert!(native.ops.is_empty());
        assert_eq!(
            native.draws,
            vec![crate::imod::three_dmod::imod::IMOD_DRAW_XYZ]
        );
        assert_eq!(native.locations, vec![(Some(10), None, None)]);
    }
    #[test]
    fn step_zoom_uses_the_configured_native_zoom_list() {
        let mut w = XyzWindow::new(view());
        let zooms = [0.5, 1., 1.5, 2.];
        w.m_zoom = 1.;
        w.step_zoom(&zooms, 1);
        assert_eq!(w.m_zoom, 1.5);
        w.step_zoom(&zooms, -1);
        assert_eq!(w.m_zoom, 1.);
    }
    #[test]
    fn boundary_time_step_follows_locked_and_unlocked_native_routes() {
        let mut w = XyzWindow::new(view());
        let mut native = TimeBoundary::default();
        w.step_time_with_boundary(view(), 1, &mut native);
        assert_eq!(native.ops, ["priority", "stop", "next"]);
        native.ops.clear();
        w.m_time_lock = 2;
        w.step_time_with_boundary(view(), -1, &mut native);
        assert_eq!(w.m_time_lock, 1);
        assert_eq!(native.ops, ["priority", "draw"]);
    }
    #[test]
    fn boundary_toggle_preserves_native_redraw_rules() {
        let mut w = XyzWindow::new(view());
        let mut native = TimeBoundary::default();
        w.state_toggled_with_boundary(XYZ_TOGGLE_RESOL, 1, view(), &mut native);
        assert_eq!(native.ops, ["priority", "draw"]);
        native.ops.clear();
        w.state_toggled_with_boundary(XYZ_TOGGLE_LOCKED, 1, view(), &mut native);
        assert_eq!(native.ops, ["priority"]);
        native.ops.clear();
        w.state_toggled_with_boundary(XYZ_TOGGLE_LOCKED, 0, view(), &mut native);
        assert_eq!(native.ops, ["priority", "draw"]);
    }
    #[test]
    fn help_and_cache_fill_follow_native_dispatch() {
        let mut w = XyzWindow::new(view());
        let mut native = TimeBoundary::default();
        w.help(&mut native);
        w.fill_cache_pressed(&mut native);
        assert_eq!(native.ops, ["help", "priority", "cache"]);
    }
    #[test]
    fn draw_auto_is_the_source_disabled_noop() {
        let mut w = XyzWindow::new(view());
        let mut native = TimeBoundary::default();
        w.draw_auto(&mut native);
        assert_eq!(native.auto_draws, 0);
    }
    #[test]
    fn finish_new_model_point_binds_location_and_redraws_model_and_xyz() {
        let mut w = XyzWindow::new(view());
        let mut native = TimeBoundary::default();
        w.finish_new_model_point(view(), 4, 5, 6, &mut native);
        assert_eq!(native.locations, [(Some(4), Some(5), Some(6))]);
        assert_eq!(
            native.draws,
            [crate::imod::three_dmod::imod::IMOD_DRAW_XYZ
                | crate::imod::three_dmod::imod::IMOD_DRAW_MOD]
        );
    }
    #[test]
    fn xyz_gl_timer_preserves_native_bump_and_first_draw_sequence() {
        let mut gl = XyzGl::new();
        gl.m_init_width = 100;
        gl.m_init_height = 80;
        gl.m_timer_id = 4;
        gl.m_scheduled_resize = true;
        assert_eq!(
            gl.timer_event(true, 120, 90),
            Some(XyzTimerAction::Resize(122, 90))
        );
        assert_eq!((gl.m_timer_id, gl.m_scheduled_bump), (0, -2));
        assert_eq!(
            gl.timer_event(true, 120, 90),
            Some(XyzTimerAction::Resize(102, 80))
        );
        assert_eq!(
            gl.timer_event(true, 120, 90),
            Some(XyzTimerAction::Resize(101, 80))
        );
        assert_eq!(
            gl.timer_event(true, 120, 90),
            Some(XyzTimerAction::Resize(100, 80))
        );
        assert_eq!(gl.timer_event(true, 120, 90), Some(XyzTimerAction::Redraw));
    }
    #[test]
    fn xyz_gl_resize_prioritizes_control_before_updating_image_geometry() {
        let mut win = XyzWindow::new(view());
        let mut gl = XyzGl::new();
        let mut native = TimeBoundary::default();
        gl.resize_gl_with_boundary(&mut win, 320, 240, &mut native);
        assert_eq!(native.ops, ["priority"]);
        assert_eq!((win.m_winx, win.m_winy, win.m_exposed), (320, 240, 1));
    }
    #[test]
    fn lock_and_time_are_clamped() {
        let mut w = XyzWindow::new(view());
        w.state_toggled(XYZ_TOGGLE_LOCKED, 1, view());
        assert_eq!(w.get_location(view()), (0, 0, 0));
        w.state_toggled(XYZ_TOGGLE_TIMELOCK, 1, view());
        assert_eq!(w.time_forward(view()), Some(2));
        w.set_location(view(), 999, NOTNEW, -5);
        assert_eq!((w.m_xlock, w.m_zlock), (99, 0));
    }
}
