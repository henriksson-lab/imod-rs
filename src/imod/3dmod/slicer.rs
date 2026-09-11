//! Translation of `IMOD/3dmod/slicer.cpp` and `sslice.h`.
//!
//! The Qt event loop, controller list, model editing, image cache and OpenGL
//! drawing APIs in the original are represented by [`SlicerNativeBoundary`].
//! Coordinate transforms, slice geometry, rubber-band coordinates, interpolation
//! and state transitions stay here, in the source translation unit.
#![allow(dead_code)]

use super::slicer_classes::{
    SLICER_LIMIT_INVALID, SLICER_LIMIT_TRUNCATE, SLICER_LIMIT_VALID, SLICER_TOGGLE_ARROW,
    SLICER_TOGGLE_BAND, SLICER_TOGGLE_CENTER, SLICER_TOGGLE_FFT, SLICER_TOGGLE_HIGHRES,
    SLICER_TOGGLE_LOCK, SLICER_TOGGLE_SHIFTLOCK, SLICER_TOGGLE_TIMELOCK, SLICER_TOGGLE_ZSCALE,
    SlicerCore, SlicerEvent,
};

pub const SLICE_ZSCALE_OFF: i32 = 0;
pub const SLICE_ZSCALE_BEFORE: i32 = 1;
pub const SLICE_ZSCALE_AFTER: i32 = 2;
pub const S_MAX_ANGLE: [f32; 3] = [90., 180., 180.];
pub const S_VIEW_AXIS_STEPS: [f32; 8] = [0.1, 0.3, 1., 3., 10., 30., 90., 0.];

/// Rust equivalent of IMOD's `Ipoint`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ipoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl Ipoint {
    fn normalize(&mut self) {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len != 0. {
            self.x /= len;
            self.y /= len;
            self.z /= len;
        }
    }
    fn cross(a: Self, b: Self) -> Self {
        Self {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }
}

/// The part of `ImodView` read or written by `slicer.cpp` proper.
#[derive(Clone, Debug)]
pub struct SlicerView {
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub xybin: i32,
    pub zbin: i32,
    pub zscale: f32,
    pub xmouse: f32,
    pub ymouse: f32,
    pub zmouse: f32,
    pub cur_time: i32,
    pub num_times: i32,
}
impl Default for SlicerView {
    fn default() -> Self {
        Self {
            xsize: 1,
            ysize: 1,
            zsize: 1,
            xybin: 1,
            zbin: 1,
            zscale: 1.,
            xmouse: 0.,
            ymouse: 0.,
            zmouse: 0.,
            cur_time: 1,
            num_times: 1,
        }
    }
}

/// Calls which cross from `slicer.cpp` into Qt, OpenGL, controller, cache,
/// model and movie units.  They deliberately remain explicit source boundaries.
pub trait SlicerNativeBoundary {
    fn draw(&mut self, _draw_flag: i32) {}
    fn draw_slicer_plane(&mut self) {}
    fn update_gl(&mut self) {}
    fn cube_draw(&mut self) {}
    fn set_angles(&mut self, _angles: [f32; 3]) {}
    fn set_zoom_text(&mut self, _zoom: f32) {}
    fn set_toggle_state(&mut self, _index: usize, _state: i32) {}
    fn set_thicknesses(&mut self, _image: i32, _model: f32) {}
    fn set_view_axis_position(&mut self, _min: i32, _max: i32, _current: i32) {}
    fn set_low_high_validity(&mut self, _which: usize, _state: i32) {}
    fn enable_low_high_buttons(&mut self, _enabled: bool) {}
    fn manage_band_size(&mut self, _width: i32, _height: i32, _action: i32) {}
    fn set_cursor(&mut self, _mode: i32, _force: bool) {}
    fn fill_cache(&mut self) {}
    fn help(&mut self) {}
    fn close(&mut self) {}
    fn movie(&mut self, _xmovie: i32, _ymovie: i32, _zmovie: i32) {}
    fn set_movie_limits(&mut self, _axis: i32, _start: i32, _end: i32) {}
    fn input_next_time(&mut self) {}
    fn input_prev_time(&mut self) {}
    fn image_draw(&mut self) {}
    fn model_draw(&mut self) {}
    fn cube_paint(&mut self) {}
}

/// `SlicerFuncs` source state.  Field names follow the systematic snake-case
/// mapping from the paired `sslice.h` members.
#[derive(Clone, Debug)]
pub struct SlicerFuncs {
    pub view: SlicerView,
    pub cx: f32,
    pub cy: f32,
    pub cz: f32,
    pub tang: [f32; 3],
    pub lang: [f32; 3],
    pub locked: i32,
    pub draw_mod_view: i32,
    pub already_drew: bool,
    pub time_lock: i32,
    pub continuous: bool,
    pub linked: bool,
    pub auto_link: i32,
    pub classic: i32,
    pub zoom: f32,
    pub winx: i32,
    pub winy: i32,
    pub hq: i32,
    pub scalez: i32,
    pub fft_mode: i32,
    pub nslice: i32,
    pub depth: f32,
    pub rubberband: i32,
    pub starting_band: i32,
    pub closing: i32,
    pub xstep: [f32; 3],
    pub ystep: [f32; 3],
    pub zstep: [f32; 3],
    pub xo: f32,
    pub yo: f32,
    pub zo: f32,
    pub xzoom: f32,
    pub yzoom: f32,
    pub remaining_zoom: f32,
    pub no_pixel_zoom: bool,
    pub pending: i32,
    pub pendx: f32,
    pub pendy: f32,
    pub pendz: f32,
    pub last_axis_pos: i32,
    pub lastangle: usize,
    pub shift_lock: i32,
    pub mousemode: i32,
    pub need_draw: bool,
    pub doing_draw: bool,
    pub arrow_on: bool,
    pub drawing_arrow: bool,
    pub arrow_head: Vec<Ipoint>,
    pub arrow_tail: Vec<Ipoint>,
    pub arrow_angle: [f32; 3],
    pub band_angle: [f32; 3],
    pub rb_image_x0: f32,
    pub rb_image_x1: f32,
    pub rb_image_y0: f32,
    pub rb_image_y1: f32,
    pub rb_image_z0: f32,
    pub rb_image_z1: f32,
    pub rb_mouse_x0: i32,
    pub rb_mouse_x1: i32,
    pub rb_mouse_y0: i32,
    pub rb_mouse_y1: i32,
    pub rb_start_x0: i32,
    pub rb_start_x1: i32,
    pub rb_start_y0: i32,
    pub rb_start_y1: i32,
    pub band_low_high_limits: [f32; 2],
    pub limit_no_value: i32,
    pub view_axis_index: usize,
    pub image_filled: i32,
    pub cur_buf_size: usize,
    pub matrix: [[f32; 3]; 3],
}

impl SlicerFuncs {
    /// `SlicerFuncs::SlicerFuncs` after Qt construction is delegated to its paired unit.
    pub fn new(mut view: SlicerView, auto_link: i32) -> Self {
        if view.xmouse == 0. && view.ymouse == 0. {
            view.xmouse = view.xsize as f32 / 2.;
            view.ymouse = view.ysize as f32 / 2.;
        }
        let mut s = Self {
            cx: view.xmouse,
            cy: view.ymouse,
            cz: view.zmouse,
            view,
            tang: [0.; 3],
            lang: [0.; 3],
            locked: 0,
            draw_mod_view: 0,
            already_drew: false,
            time_lock: auto_link,
            continuous: false,
            linked: auto_link > 0,
            auto_link: auto_link.min(2),
            classic: 0,
            zoom: 1.,
            winx: 1,
            winy: 1,
            hq: 0,
            scalez: 0,
            fft_mode: 0,
            nslice: 1,
            depth: 1.,
            rubberband: 0,
            starting_band: 0,
            closing: 0,
            xstep: [1., 0., 0.],
            ystep: [0., 1., 0.],
            zstep: [0., 0., 1.],
            xo: 0.,
            yo: 0.,
            zo: 0.,
            xzoom: 1.,
            yzoom: 1.,
            remaining_zoom: 1.,
            no_pixel_zoom: false,
            pending: 0,
            pendx: 0.,
            pendy: 0.,
            pendz: 0.,
            last_axis_pos: 1,
            lastangle: 0,
            shift_lock: 0,
            mousemode: 0,
            need_draw: false,
            doing_draw: false,
            arrow_on: false,
            drawing_arrow: false,
            arrow_head: Vec::new(),
            arrow_tail: Vec::new(),
            arrow_angle: [0.; 3],
            band_angle: [0.; 3],
            rb_image_x0: 0.,
            rb_image_x1: 0.,
            rb_image_y0: 0.,
            rb_image_y1: 0.,
            rb_image_z0: 0.,
            rb_image_z1: 0.,
            rb_mouse_x0: 0,
            rb_mouse_x1: 0,
            rb_mouse_y0: 0,
            rb_mouse_y1: 0,
            rb_start_x0: 0,
            rb_start_x1: 0,
            rb_start_y0: 0,
            rb_start_y1: 0,
            band_low_high_limits: [0.; 2],
            limit_no_value: 0,
            view_axis_index: 2,
            image_filled: 0,
            cur_buf_size: 0,
            matrix: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        };
        s.trans_step();
        s
    }

    /// `SlicerFuncs::setInitialZoom`.
    pub fn set_initial_zoom(&mut self, zoom: f32) {
        self.zoom = zoom;
        if zoom > 1.5 {
            self.hq = 1;
        }
    }
    /// `SlicerFuncs::viewAxisStepSize`.
    pub fn view_axis_step_size(&self) -> f32 {
        S_VIEW_AXIS_STEPS[self.view_axis_index]
    }
    /// `SlicerFuncs::help`.
    pub fn help(&mut self, n: &mut dyn SlicerNativeBoundary) {
        n.help()
    }
    /// `SlicerFuncs::stepZoom`.
    pub fn step_zoom(&mut self, dir: i32, n: &mut dyn SlicerNativeBoundary) {
        self.zoom = if dir > 0 {
            self.zoom * 1.25
        } else {
            self.zoom / 1.25
        };
        self.manage_buffers();
        n.set_zoom_text(self.zoom);
        self.draw_self_and_linked(n);
    }
    /// `SlicerFuncs::enteredZoom`.
    pub fn entered_zoom(&mut self, zoom: f32, n: &mut dyn SlicerNativeBoundary) {
        if self.closing != 0 {
            return;
        }
        self.zoom = zoom.max(0.01);
        self.manage_buffers();
        n.set_zoom_text(self.zoom);
        self.draw_self_and_linked(n);
    }
    /// `SlicerFuncs::stepTime`.
    pub fn step_time(&mut self, step: i32, n: &mut dyn SlicerNativeBoundary) {
        if self.time_lock != 0 {
            self.time_lock = (self.time_lock + step).clamp(1, self.view.num_times);
            self.draw(n);
        } else if step > 0 {
            n.input_next_time()
        } else {
            n.input_prev_time()
        }
    }
    /// `SlicerFuncs::showSlice`.
    pub fn show_slice(&mut self, n: &mut dyn SlicerNativeBoundary) {
        self.trans_step();
        n.draw(0x20 | self.draw_mod_view);
        self.draw_mod_view = 0;
    }
    /// `SlicerFuncs::fillCache`.
    pub fn fill_cache(&mut self, n: &mut dyn SlicerNativeBoundary) {
        n.fill_cache()
    }
    /// `SlicerFuncs::stateToggled`.
    pub fn state_toggled(&mut self, index: usize, state: i32, n: &mut dyn SlicerNativeBoundary) {
        match index {
            SLICER_TOGGLE_LOCK => {
                self.locked = state;
                if state == 0 {
                    self.cx = self.view.xmouse;
                    self.cy = self.view.ymouse;
                    self.cz = self.view.zmouse;
                    self.pending = 0;
                    self.draw(n);
                }
            }
            SLICER_TOGGLE_HIGHRES => {
                self.hq = state;
                self.manage_buffers();
                self.draw_self_and_linked(n);
            }
            SLICER_TOGGLE_CENTER => self.set_classic_mode(state, false, n),
            SLICER_TOGGLE_SHIFTLOCK => self.shift_lock = state,
            SLICER_TOGGLE_BAND => self.toggle_rubberband(true, n),
            SLICER_TOGGLE_ARROW => self.toggle_arrow(true, n),
            SLICER_TOGGLE_FFT => {
                self.fft_mode = state;
                self.draw_self_and_linked(n);
            }
            SLICER_TOGGLE_ZSCALE => {
                self.scalez = state;
                self.draw_self_and_linked(n);
                n.draw_slicer_plane();
            }
            SLICER_TOGGLE_TIMELOCK => {
                self.time_lock = if state != 0 { self.view.cur_time } else { 0 };
                if state == 0 {
                    self.draw(n);
                }
            }
            _ => {}
        }
    }
    /// `SlicerFuncs::toggleArrow`.
    pub fn toggle_arrow(&mut self, draw_win: bool, n: &mut dyn SlicerNativeBoundary) {
        self.arrow_on = !self.arrow_on;
        self.drawing_arrow = self.arrow_on;
        n.set_toggle_state(SLICER_TOGGLE_ARROW, self.arrow_on as i32);
        if self.arrow_on {
            self.arrow_head.push(Ipoint::default());
            self.arrow_tail.push(Ipoint::default());
        } else {
            self.arrow_head.pop();
            self.arrow_tail.pop();
        }
        if self.arrow_on && self.starting_band != 0 {
            self.toggle_rubberband(false, n);
        }
        n.set_cursor(self.mousemode, true);
        if draw_win {
            self.draw(n);
        }
        n.set_cursor(self.mousemode, true);
    }
    /// `SlicerFuncs::clearArrows`.
    pub fn clear_arrows(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if self.closing == 0 {
            if self.arrow_on {
                self.toggle_arrow(false, n);
            }
            self.arrow_head.clear();
            self.arrow_tail.clear();
            self.draw(n);
        }
    }
    /// `SlicerFuncs::startAddedArrow`.
    pub fn start_added_arrow(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if let (Some(h), Some(t)) = (self.arrow_head.last(), self.arrow_tail.last()) {
            if self.drawing_arrow && *h == Ipoint::default() && *t == Ipoint::default() {
                return;
            }
        }
        self.arrow_on = false;
        self.toggle_arrow(false, n);
    }
    /// `SlicerFuncs::setClassicMode`.
    pub fn set_classic_mode(
        &mut self,
        state: i32,
        skip_draw: bool,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        self.classic = state;
        if self.locked == 0 && state != 0 {
            self.cx = self.view.xmouse;
            self.cy = self.view.ymouse;
            self.cz = self.view.zmouse;
        }
        self.pending = 0;
        if !skip_draw {
            self.draw(n);
            n.draw(1);
        }
    }
    /// `SlicerFuncs::angleChanged`.
    pub fn angle_changed(
        &mut self,
        axis: i32,
        value: i32,
        dragging: bool,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        if axis < 3 {
            self.set_forward_matrix();
            self.tang[axis as usize] = value as f32 * 0.1;
            self.lastangle = axis as usize;
        } else {
            let v = self.normal_to_plane();
            let d = (value - self.last_axis_pos) as f32;
            self.cx += d * v.x;
            self.cy += d * v.y;
            self.cz += d * v.z;
        }
        if !dragging {
            self.show_slice(n);
        } else {
            self.trans_step();
            n.cube_draw();
        }
    }
    /// `SlicerFuncs::updateViewAxisPos`.
    pub fn update_view_axis_pos(&mut self, n: &mut dyn SlicerNativeBoundary) {
        let v = self.normal_to_plane();
        let mut nums = [0_i32; 2];
        for (ind, direction) in [-1., 1.].into_iter().enumerate() {
            let mut i = 1;
            loop {
                let x = self.cx + direction * i as f32 * v.x;
                let y = self.cy + direction * i as f32 * v.y;
                let z = self.cz + direction * i as f32 * v.z;
                if x < 0.
                    || x >= self.view.xsize as f32
                    || y < 0.
                    || y >= self.view.ysize as f32
                    || z < 0.
                    || z >= self.view.zsize as f32 - 0.5
                {
                    nums[ind] = i - 1;
                    break;
                }
                i += 1;
            }
        }
        self.last_axis_pos = 1 + nums[0];
        n.set_view_axis_position(1, self.last_axis_pos + nums[1], self.last_axis_pos);
    }
    /// `SlicerFuncs::drawThickControls`.
    pub fn draw_thick_controls(&self, n: &mut dyn SlicerNativeBoundary) {
        n.set_thicknesses(self.nslice, self.depth)
    }
    /// `SlicerFuncs::imageThickness`.
    pub fn image_thickness(&mut self, depth: i32, n: &mut dyn SlicerNativeBoundary) {
        if self.closing == 0 {
            self.nslice = depth.max(1);
            self.draw_thick_controls(n);
            self.draw_self_and_linked(n);
        }
    }
    /// `SlicerFuncs::modelThickness`.
    pub fn model_thickness(&mut self, depth: f32, n: &mut dyn SlicerNativeBoundary) {
        if self.closing == 0 {
            self.depth = if (self.depth - 0.1).abs() < 0.01 && (depth - 1.1).abs() < 0.01 {
                1.
            } else {
                depth.max(0.1)
            };
            self.draw_thick_controls(n);
            self.draw_self_and_linked(n);
        }
    }
    /// `SlicerFuncs::setLinkedState`.
    pub fn set_linked_state(&mut self, state: bool) {
        self.linked = state;
        if !state {
            self.auto_link = 0;
        }
    }
    /// `SlicerFuncs::closing`.
    pub fn closing(&mut self, n: &mut dyn SlicerNativeBoundary) {
        self.closing = 1;
        self.linked = false;
        self.arrow_head.clear();
        self.arrow_tail.clear();
        n.close();
    }
    /// `SlicerFuncs::getSubsetLimits`.
    pub fn get_subset_limits(&self) -> (i32, i32, i32, i32) {
        let xs = (self.cx - 0.7 * self.winx as f32 / self.zoom).max(0.) as i32;
        let xe = (self.cx + 0.7 * self.winx as f32 / self.zoom).min(self.view.xsize as f32) as i32;
        let ys = (self.cy - 0.7 * self.winy as f32 / self.zoom).max(0.) as i32;
        let ye = (self.cy + 0.7 * self.winy as f32 / self.zoom).min(self.view.ysize as f32) as i32;
        (xs, ys, xe - xs, ye - ys)
    }
    /// `SlicerFuncs::setViewAxisRotation`.
    pub fn set_view_axis_rotation(&mut self, x: f32, y: f32, z: f32) {
        self.set_forward_matrix();
        let r = rotation_matrix(x, y, z);
        self.matrix = matrix_mul(self.matrix, r);
        self.tang = natural_angles(self.matrix);
    }
    /// `SlicerFuncs::rotateOnViewAxis`.
    pub fn rotate_on_view_axis(
        &mut self,
        dx: i32,
        dy: i32,
        dz: i32,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        let u = self.view_axis_step_size();
        self.set_view_axis_rotation(dx as f32 * u, dy as f32 * u, dz as f32 * u);
        n.set_angles(self.tang);
        self.show_slice(n);
        self.draw_self_and_linked(n);
    }
    /// `SlicerFuncs::findMovieAxis`.
    pub fn find_movie_axis(&mut self, direction: i32) -> (i32, i32, i32, usize) {
        let v = self.normal_to_plane();
        if v.x.abs() >= v.y.abs() && v.x.abs() >= v.z.abs() {
            (direction, 0, 0, 0)
        } else if v.y.abs() >= v.z.abs() {
            (0, direction, 0, 1)
        } else {
            (0, 0, direction, 2)
        }
    }
    /// `SlicerFuncs::findAxisLimits`.
    pub fn find_axis_limits(&mut self, axis: usize) -> (f32, i32, i32) {
        let v = self.normal_to_plane();
        let current = [self.cx, self.cy, self.cz][axis];
        let size = [self.view.xsize, self.view.ysize, self.view.zsize][axis];
        let comp = [v.x, v.y, v.z][axis];
        if comp.abs() < f32::EPSILON {
            return (current, -1, -1);
        }
        let mut start = -1;
        let mut end = -1;
        for i in 0..size {
            let d = (i as f32 - current) / comp;
            let x = self.cx + d * v.x;
            let y = self.cy + d * v.y;
            let z = self.cz + d * v.z;
            if x >= 0.
                && x <= self.view.xsize as f32 - 1.
                && y >= 0.
                && y <= self.view.ysize as f32 - 1.
                && z >= 0.
                && z <= self.view.zsize as f32 - 1.
            {
                if start < 0 {
                    start = i;
                }
                end = i;
            }
        }
        (current, start, end)
    }
    /// `SlicerFuncs::setMovieLimits`.
    pub fn set_movie_limits(&mut self, axis: usize, n: &mut dyn SlicerNativeBoundary) {
        let (_, start, end) = self.find_axis_limits(axis);
        n.set_movie_limits(axis as i32, start, end);
    }
    /// `SlicerFuncs::startMovieCheckSnap`.
    pub fn start_movie_check_snap(&mut self, direction: i32, n: &mut dyn SlicerNativeBoundary) {
        let (x, y, z, axis) = self.find_movie_axis(direction);
        n.movie(x, y, z);
        self.view.xmouse = self.cx;
        self.view.ymouse = self.cy;
        self.view.zmouse = self.cz;
        self.set_movie_limits(axis, n);
    }
    /// `SlicerFuncs::getZScaleBefore`.
    pub fn get_z_scale_before(&self) -> f32 {
        let mut z = self.view.zbin as f32 / self.view.xybin.max(1) as f32;
        if self.scalez == SLICE_ZSCALE_BEFORE && self.view.zscale > 0. {
            z *= self.view.zscale;
        }
        z
    }
    /// `SlicerFuncs::setxyz`.
    pub fn setxyz(&mut self, x: i32, y: i32) -> i32 {
        let (xm, ym, zm, zmouse) = self.getxyz(x as f32, y as f32, true);
        self.pendx = xm;
        self.pendy = ym;
        self.pendz = zm;
        self.pending = 1;
        self.view.xmouse = xm;
        self.view.ymouse = ym;
        self.view.zmouse = zm;
        zmouse
    }
    /// `SlicerFuncs::getxyz` (both C++ overloads are represented by `f32` input).
    pub fn getxyz(&self, x: f32, y: f32, clamp: bool) -> (f32, f32, f32, i32) {
        let zs = 1. / self.get_z_scale_before();
        let xo = (self.winx / 2) as f32 - x;
        let yo = self.winy as f32 / 2. - (self.winy - 1) as f32 + y;
        let xo = xo / self.xzoom;
        let yo = yo / self.yzoom;
        let mut xm = self.cx - (self.xstep[0] * xo + self.ystep[0] * yo);
        let mut ym = self.cy - (self.xstep[1] * xo + self.ystep[1] * yo);
        let mut zm = self.cz - (self.xstep[2] * xo * zs + self.ystep[2] * yo * zs);
        if clamp {
            xm = xm.clamp(0., self.view.xsize as f32 - 1.);
            ym = ym.clamp(0., self.view.ysize as f32 - 1.);
            zm = zm.clamp(0., self.view.zsize as f32 - 1.);
        }
        (xm, ym, zm, zm.round() as i32)
    }
    /// `SlicerFuncs::getWindowCoords`.
    pub fn get_window_coords(&mut self, mut x: f32, mut y: f32, mut z: f32) -> (f32, f32, f32) {
        let zs = 1. / self.get_z_scale_before();
        self.set_forward_matrix();
        x -= self.cx;
        y -= self.cy;
        z = (z - self.cz) / zs;
        let xn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
        );
        let yn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 0.,
                y: 1.,
                z: 0.,
            },
        );
        let zn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 0.,
                y: 0.,
                z: 1.,
            },
        );
        let xo = xn.x * x + yn.x * y + zn.x * z;
        let yo = xn.y * x + yn.y * y + zn.y * z;
        let zo = xn.z * x + yn.z * y + zn.z * z;
        (
            xo * self.xzoom + self.winx as f32 / 2.,
            yo * self.yzoom + self.winy as f32 / 2.,
            zo,
        )
    }
    /// `SlicerFuncs::bandMouseToImage`.
    pub fn band_mouse_to_image(&mut self) {
        let (a, b, c, _) = self.getxyz(self.rb_mouse_x0 as f32, self.rb_mouse_y0 as f32, false);
        let (d, e, f, _) = self.getxyz(self.rb_mouse_x1 as f32, self.rb_mouse_y1 as f32, false);
        let saved = (self.cx, self.cy, self.cz);
        self.cx = self.view.xsize as f32 / 2.;
        self.cy = self.view.ysize as f32 / 2.;
        self.cz = self.view.zsize as f32 / 2.;
        let (x0, y0, _) = self.get_window_coords(a, b, c);
        let (x1, y1, _) = self.get_window_coords(d, e, f);
        let (rx0, ry0, rz0, _) = self.getxyz(x0, self.winy as f32 - 1. - y0, false);
        let (rx1, ry1, rz1, _) = self.getxyz(x1, self.winy as f32 - 1. - y1, false);
        self.rb_image_x0 = rx0;
        self.rb_image_y0 = ry0;
        self.rb_image_z0 = rz0;
        self.rb_image_x1 = rx1;
        self.rb_image_y1 = ry1;
        self.rb_image_z1 = rz1;
        (self.cx, self.cy, self.cz) = saved;
    }
    /// `SlicerFuncs::bandImageToMouse`.
    pub fn band_image_to_mouse(&mut self) {
        let (x0, y0, _) =
            self.get_window_coords(self.rb_image_x0, self.rb_image_y0, self.rb_image_z0);
        let (x1, y1, _) =
            self.get_window_coords(self.rb_image_x1, self.rb_image_y1, self.rb_image_z1);
        self.rb_mouse_x0 = x0.round() as i32;
        self.rb_mouse_y0 = self.winy - 1 - y0.round() as i32;
        self.rb_mouse_x1 = x1.round() as i32;
        self.rb_mouse_y1 = self.winy - 1 - y1.round() as i32;
    }
    /// `SlicerFuncs::rubberBandImageCoords`.
    pub fn rubber_band_image_coords(&self) -> (f32, f32, f32, f32, f32, f32) {
        (
            self.rb_image_x0.min(self.rb_image_x1),
            self.rb_image_x0.max(self.rb_image_x1),
            self.rb_image_y0.min(self.rb_image_y1),
            self.rb_image_y0.max(self.rb_image_y1),
            self.rb_image_z0.min(self.rb_image_z1),
            self.rb_image_z0.max(self.rb_image_z1),
        )
    }
    /// `SlicerFuncs::setSnapshotLimits`.
    pub fn set_snapshot_limits(&mut self, device_pixel_ratio: f32) -> Option<[i32; 4]> {
        if self.rubberband == 0 {
            return None;
        }
        self.band_image_to_mouse();
        let x0 = (self.rb_mouse_x0 + 1).clamp(0, self.winx - 2);
        let y0 = (self.winy - self.rb_mouse_y1).clamp(0, self.winy - 2);
        let x1 = (self.rb_mouse_x1 - if device_pixel_ratio > 1. { 2 } else { 1 })
            .clamp(x0, self.winx - 1);
        let y1 = (self.winy - self.rb_mouse_y0 - if device_pixel_ratio > 1. { 3 } else { 2 })
            .clamp(y0, self.winy - 1);
        Some([x0, y0, x1 + 1 - x0, y1 + 1 - y0])
    }
    /// `SlicerFuncs::toggleRubberband`.
    pub fn toggle_rubberband(&mut self, draw_win: bool, n: &mut dyn SlicerNativeBoundary) {
        if self.rubberband != 0 || self.starting_band != 0 {
            self.rubberband = 0;
            self.starting_band = 0;
        } else {
            if self.drawing_arrow {
                self.toggle_arrow(false, n);
            }
            self.starting_band = 1;
            self.limit_no_value = self.view.xsize.max(self.view.ysize).max(self.view.zsize);
            self.band_low_high_limits = [2. * self.limit_no_value as f32; 2];
        }
        let on = self.rubberband + self.starting_band;
        n.set_toggle_state(SLICER_TOGGLE_BAND, on);
        n.enable_low_high_buttons(on != 0);
        n.set_cursor(self.mousemode, true);
        if draw_win {
            self.draw(n);
        }
    }
    /// `SlicerFuncs::findBandAxisRange`.
    pub fn find_band_axis_range(&mut self) -> Option<(f32, f32, f32)> {
        if self.rubberband == 0 {
            return None;
        }
        let saved = (self.cx, self.cy, self.cz);
        self.cx = (self.rb_image_x0 + self.rb_image_x1) / 2.;
        self.cy = (self.rb_image_y0 + self.rb_image_y1) / 2.;
        self.cz = (self.rb_image_z0 + self.rb_image_z1) / 2.;
        let (_, _, _, axis) = self.find_movie_axis(1);
        let (current, start, end) = self.find_axis_limits(axis);
        let delta = [self.view.xsize, self.view.ysize, self.view.zsize][axis] as f32 / 2.;
        (self.cx, self.cy, self.cz) = saved;
        if start < 0 {
            None
        } else {
            Some((current - delta, start as f32 - delta, end as f32 - delta))
        }
    }
    /// `SlicerFuncs::checkBandLowHighLimits`.
    pub fn check_band_low_high_limits(
        &mut self,
        n: &mut dyn SlicerNativeBoundary,
    ) -> Result<[f32; 2], i32> {
        let Some((current, start, end)) = self.find_band_axis_range() else {
            return Err(2);
        };
        let mut values = [0.; 2];
        let mut invalid = false;
        for i in 0..2 {
            if self.band_low_high_limits[i] > self.limit_no_value as f32 {
                invalid = true;
                n.set_low_high_validity(i, SLICER_LIMIT_INVALID);
            } else {
                values[i] = (current + self.band_low_high_limits[i]).clamp(start, end);
                n.set_low_high_validity(
                    i,
                    if current + self.band_low_high_limits[i] < start
                        || current + self.band_low_high_limits[i] > end
                    {
                        SLICER_LIMIT_TRUNCATE
                    } else {
                        SLICER_LIMIT_VALID
                    },
                );
            }
        }
        if invalid { Err(1) } else { Ok(values) }
    }
    /// `SlicerFuncs::setBandLowHighLimit` (non-Shift half; Shift movement is Qt input state).
    pub fn set_band_low_high_limit(&mut self, which: usize, n: &mut dyn SlicerNativeBoundary) {
        let (_, _, _, axis) = self.find_movie_axis(1);
        self.band_low_high_limits[which] = self.current_main_axis_distance(axis);
        let _ = self.check_band_low_high_limits(n);
    }
    /// `SlicerFuncs::currentMainAxisDistance`.
    pub fn current_main_axis_distance(&mut self, axis: usize) -> f32 {
        let saved = (self.cx, self.cy, self.cz);
        self.cx = self.view.xsize as f32 / 2.;
        self.cy = self.view.ysize as f32 / 2.;
        self.cz = self.view.zsize as f32 / 2.;
        let (wx, wy, _) = self.get_window_coords(saved.0, saved.1, saved.2);
        let (x, y, z, _) = self.getxyz(wx, self.winy as f32 - 1. - wy, false);
        (self.cx, self.cy, self.cz) = saved;
        [saved.0 - x, saved.1 - y, saved.2 - z][axis]
    }
    /// `SlicerFuncs::resizeBandToWindow`.
    pub fn resize_band_to_window(&mut self, n: &mut dyn SlicerNativeBoundary) {
        self.band_image_to_mouse();
        let dx = self.rb_mouse_x1 as f32 - 1. - self.rb_mouse_x0 as f32;
        let dy = self.rb_mouse_y1 as f32 - 1. - self.rb_mouse_y0 as f32;
        if dx != 0. && dy != 0. {
            self.zoom *= (self.winx as f32 / dx).min(self.winy as f32 / dy);
        }
        self.draw_self_and_linked(n);
        self.show_slice(n);
    }
    /// `SlicerFuncs::fixangle`.
    pub fn fixangle(mut angle: f64) -> f64 {
        let r = std::f64::consts::PI / 180.;
        if angle <= -180. * r {
            angle += 360. * r;
        }
        if angle > 180. * r {
            angle -= 360. * r;
        }
        angle
    }
    /// `SlicerFuncs::setForwardMatrix`.
    pub fn set_forward_matrix(&mut self) {
        self.matrix = rotation_matrix(self.tang[0], self.tang[1], self.tang[2]);
    }
    /// `SlicerFuncs::setInverseMatrix`.
    pub fn set_inverse_matrix(&mut self) {
        self.matrix = rotation_matrix(-self.tang[0], -self.tang[1], -self.tang[2]);
    }
    /// `SlicerFuncs::getNormalToPlane` (both upstream overloads use this return form).
    pub fn normal_to_plane(&mut self) -> Ipoint {
        self.set_inverse_matrix();
        let mut n = matrix_vec(
            self.matrix,
            Ipoint {
                x: 0.,
                y: 0.,
                z: 1.,
            },
        );
        n.z /= self.get_z_scale_before();
        n.normalize();
        n
    }
    /// `SlicerFuncs::translateByRotatedVec`.
    pub fn translate_by_rotated_vec(&mut self, vector: Ipoint, outside_ok: bool) -> bool {
        self.set_inverse_matrix();
        let mut rotated = matrix_vec(self.matrix, vector);
        rotated.z /= self.get_z_scale_before();
        if vector.z != 0. {
            rotated.normalize();
        }
        let x = self.cx + rotated.x;
        let y = self.cy + rotated.y;
        let z = self.cz + rotated.z;
        if outside_ok
            || (x >= 0.
                && x < self.view.xsize as f32
                && y >= 0.
                && y < self.view.ysize as f32
                && z >= 0.
                && z < self.view.zsize as f32 - 0.5)
        {
            self.cx = x;
            self.cy = y;
            self.cz = z;
            true
        } else {
            false
        }
    }
    /// `SlicerFuncs::setAnglesFromPoints`.
    pub fn set_angles_from_points(&mut self, p1: Ipoint, p2: Ipoint, axis: usize) {
        let mut n = Ipoint {
            x: p2.x - p1.x,
            y: p2.y - p1.y,
            z: (p2.z - p1.z) * self.get_z_scale_before(),
        };
        if n == Ipoint::default() {
            return;
        }
        n.normalize();
        let eps = 1.0e-4;
        let r = std::f32::consts::PI / 180.;
        let a: Ipoint;
        if axis == 0 {
            let az = if n.x.abs() > eps || n.y.abs() > eps {
                -n.y.atan2(n.x)
            } else {
                0.
            };
            let val = n.x * az.cos() - n.y * az.sin();
            a = Ipoint {
                x: 0.,
                y: Self::fixangle((90. * r - val.atan2(n.z)) as f64) as f32,
                z: az,
            };
        } else if axis == 1 {
            let az = if n.x.abs() > eps || n.y.abs() > eps {
                Self::fixangle((90. * r - n.y.atan2(n.x)) as f64) as f32
            } else {
                0.
            };
            let val = n.x * az.sin() + n.y * az.cos();
            a = Ipoint {
                x: -n.z.atan2(val),
                y: 0.,
                z: az,
            };
        } else {
            let az = if n.x.abs() > eps || n.y.abs() > eps {
                if n.y >= 0. {
                    n.x.atan2(n.y)
                } else {
                    -n.x.atan2(-n.y)
                }
            } else {
                0.
            };
            let val = n.x * az.sin() + n.y * az.cos();
            a = Ipoint {
                x: if n.z >= 0. {
                    val.atan2(n.z)
                } else {
                    -val.atan2(-n.z)
                },
                y: 0.,
                z: az,
            };
        }
        self.tang = [a.x / r, a.y / r, a.z / r];
    }
    /// `SlicerFuncs::transStep`.
    pub fn trans_step(&mut self) {
        self.set_inverse_matrix();
        let xn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
        );
        let yn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 0.,
                y: 1.,
                z: 0.,
            },
        );
        let zn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 0.,
                y: 0.,
                z: 1.,
            },
        );
        self.xstep = [xn.x, xn.y, xn.z];
        self.ystep = [yn.x, yn.y, yn.z];
        self.zstep = [zn.x, zn.y, zn.z];
        let isize = self.winx as f32 / self.zoom;
        let jsize = self.winy as f32 / self.zoom;
        let zs = 1. / self.get_z_scale_before();
        self.xo = self.cx - isize / 2. * xn.x - jsize / 2. * yn.x;
        self.yo = self.cy - isize / 2. * xn.y - jsize / 2. * yn.y;
        self.zo = self.cz - isize / 2. * xn.z * zs - jsize / 2. * yn.z * zs;
    }
    /// `SlicerFuncs::resize`.
    pub fn resize(&mut self, winx: i32, winy: i32) {
        if self.closing == 0 {
            self.winx = winx;
            self.winy = winy;
            self.manage_buffers();
        }
    }
    /// `SlicerFuncs::cubeResize`.
    pub fn cube_resize(&mut self, _winx: i32, _winy: i32) {}
    /// `SlicerFuncs::manageBuffers`; allocation occurs in this source unit's caller.
    pub fn manage_buffers(&mut self) -> i32 {
        let mut size = (self.winx.max(0) as usize).saturating_mul(self.winy.max(0) as usize);
        if self.hq != 0 && self.zoom < 1. {
            size = ((self.winx as f32 / self.zoom).ceil() as usize + 1)
                * ((self.winy as f32 / self.zoom).ceil() as usize + 1);
        }
        if size <= self.cur_buf_size && (size as f32) > 0.8 * self.cur_buf_size as f32 {
            return 0;
        }
        self.cur_buf_size = size;
        0
    }
    /// `SlicerFuncs::draw`.
    pub fn draw(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if self.doing_draw {
            return;
        }
        self.doing_draw = true;
        n.update_gl();
        n.cube_draw();
        self.need_draw = false;
        self.doing_draw = false;
    }
    /// `SlicerFuncs::drawSelfAndLinked`.
    pub fn draw_self_and_linked(&mut self, n: &mut dyn SlicerNativeBoundary) {
        self.draw(n)
    }
    /// `SlicerFuncs::updateImage`.
    pub fn update_image(&mut self, n: &mut dyn SlicerNativeBoundary) {
        self.image_filled += 1;
        n.update_gl();
        self.image_filled -= 1;
        n.cube_draw();
    }
    /// `SlicerFuncs::paint`.
    pub fn paint(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if self.closing == 0 {
            n.image_draw();
            n.model_draw();
        }
    }
    /// `SlicerFuncs::cubePaint`.
    pub fn cube_paint(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if self.closing == 0 {
            n.cube_paint();
        }
    }
    /// `SlicerFuncs::keyInput`: Qt key decoding remains the actual Qt boundary.
    pub fn key_input(&mut self, event: SlicerEvent, n: &mut dyn SlicerNativeBoundary) {
        match event.key {
            43 => self.step_zoom(1, n),
            45 => self.step_zoom(-1, n),
            _ => {}
        }
    }
    /// `SlicerFuncs::keyRelease`.
    pub fn key_release(&mut self, _event: SlicerEvent) {}
    /// `SlicerFuncs::mousePress`.
    pub fn mouse_press(&mut self, _event: SlicerEvent) {}
    /// `SlicerFuncs::mouseRelease`.
    pub fn mouse_release(&mut self, _event: SlicerEvent) {}
    /// `SlicerFuncs::mouseMove`.
    pub fn mouse_move(&mut self, _event: SlicerEvent) {}
    /// `SlicerFuncs::generalEvent`.
    pub fn general_event(&mut self, _event: SlicerEvent) {}
}

/// `slicerCubicFillin`.  `int_data` chooses the source's `int *` path; in
/// Rust it is explicit to prevent aliasing a `u16` buffer as `i32`.
pub fn slicer_cubic_fillin_u16(
    data: &mut [u16],
    winx: usize,
    winy: usize,
    izoom: usize,
    ilim: usize,
    jlim: usize,
    minval: i32,
    maxval: i32,
) {
    if izoom == 0 || winx == 0 || winy == 0 {
        return;
    }
    for jfill in 0..izoom {
        let dy = jfill as f32 / izoom as f32;
        let dysq = dy * dy;
        let dycub = dy * dysq;
        let fyp = 2. * dysq - dycub - dy;
        let fy = 1. + dycub - 2. * dysq;
        let fyn = dy + dysq - dycub;
        let fyn2 = dycub - dysq;
        let first = if jfill == 0 { 1 } else { 0 };
        for ifill in first..izoom {
            let dx = ifill as f32 / izoom as f32;
            let dxsq = dx * dx;
            let dxcub = dx * dxsq;
            let fxp = 2. * dxsq - dxcub - dx;
            let fx = 1. + dxcub - 2. * dxsq;
            let fxn = dx + dxsq - dxcub;
            let fxn2 = dxcub - dxsq;
            for j in (izoom + jfill..jlim).step_by(izoom) {
                if j < izoom || j + 2 * izoom > winy {
                    continue;
                }
                for i in (izoom + ifill..ilim).step_by(izoom) {
                    if i < izoom || i + 2 * izoom > winx {
                        continue;
                    }
                    let sample = |xx: usize, yy: usize| data[xx + yy * winx] as f32;
                    let row = |yy: usize| {
                        fxp * sample(i - izoom, yy)
                            + fx * sample(i - ifill, yy)
                            + fxn * sample(i - ifill + izoom, yy)
                            + fxn2 * sample(i - ifill + 2 * izoom, yy)
                    };
                    let value = (fyp * row(j - jfill - izoom)
                        + fy * row(j - jfill)
                        + fyn * row(j - jfill + izoom)
                        + fyn2 * row(j - jfill + 2 * izoom))
                    .clamp(minval as f32, maxval as f32);
                    data[i + j * winx] = (value + 0.5) as u16;
                }
            }
        }
    }
}

fn rotation_matrix(x: f32, y: f32, z: f32) -> [[f32; 3]; 3] {
    let (x, y, z) = (x.to_radians(), y.to_radians(), z.to_radians());
    let (sx, cx) = x.sin_cos();
    let (sy, cy) = y.sin_cos();
    let (sz, cz) = z.sin_cos();
    matrix_mul(
        matrix_mul(
            [[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]],
            [[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]],
        ),
        [[1., 0., 0.], [0., cx, -sx], [0., sx, cx]],
    )
}
fn matrix_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}
fn matrix_vec(m: [[f32; 3]; 3], p: Ipoint) -> Ipoint {
    Ipoint {
        x: m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z,
        y: m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z,
        z: m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z,
    }
}
fn natural_angles(m: [[f32; 3]; 3]) -> [f32; 3] {
    let y = (-m[2][0]).asin();
    let x = m[2][1].atan2(m[2][2]);
    let z = m[1][0].atan2(m[0][0]);
    [x.to_degrees(), y.to_degrees(), z.to_degrees()]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N;
    impl SlicerNativeBoundary for N {}
    #[test]
    fn source_coordinate_round_trip_at_zero_rotation() {
        let mut s = SlicerFuncs::new(
            SlicerView {
                xsize: 100,
                ysize: 100,
                zsize: 50,
                xmouse: 50.,
                ymouse: 50.,
                zmouse: 25.,
                ..Default::default()
            },
            0,
        );
        s.resize(200, 100);
        let (x, y, z) = s.get_window_coords(50., 50., 25.);
        assert_eq!((x.round(), y.round(), z.round()), (100., 50., 0.));
        let (ix, iy, iz, _) = s.getxyz(x, s.winy as f32 - 1. - y, true);
        assert!((ix - 50.).abs() < 1.);
        assert!((iy - 50.).abs() < 1.);
        assert!((iz - 25.).abs() < 1.);
    }
    #[test]
    fn normal_and_axis_limits_are_source_geometry() {
        let mut s = SlicerFuncs::new(
            SlicerView {
                xsize: 11,
                ysize: 13,
                zsize: 17,
                xmouse: 5.,
                ymouse: 6.,
                zmouse: 8.,
                ..Default::default()
            },
            0,
        );
        assert_eq!(
            s.normal_to_plane(),
            Ipoint {
                x: 0.,
                y: 0.,
                z: 1.
            }
        );
        assert_eq!(s.find_axis_limits(2), (8., 0, 16));
    }
    #[test]
    fn cubic_fills_grid_intermediate_values() {
        let mut d = vec![0_u16; 7 * 7];
        for y in (0..7).step_by(2) {
            for x in (0..7).step_by(2) {
                d[x + y * 7] = (x + y * 10) as u16;
            }
        }
        slicer_cubic_fillin_u16(&mut d, 7, 7, 2, 7, 7, 0, 1000);
        assert_ne!(d[3 + 3 * 7], 0);
    }
    #[test]
    fn toggle_uses_source_exclusive_arrow_band_state() {
        let mut s = SlicerFuncs::new(SlicerView::default(), 0);
        let mut n = N;
        s.toggle_arrow(false, &mut n);
        assert!(s.arrow_on);
        s.toggle_rubberband(false, &mut n);
        assert!(!s.arrow_on);
        assert_eq!(s.starting_band, 1);
    }
}
