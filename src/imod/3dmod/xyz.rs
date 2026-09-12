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
    fn view_state(&self) -> XyzViewState;
    fn set_location(&mut self, x: Option<i32>, y: Option<i32>, z: Option<i32>);
    fn draw(&mut self, flags: i32);
    fn update_gl(&mut self);
    fn control_priority(&mut self, ctrl: i32);
    fn remove_control(&mut self, ctrl: i32);
    fn movie_xyzt(&mut self, x: i32, y: i32, z: i32, t: i32);
    fn next_time(&mut self, forward: bool);
    fn cache_fill(&mut self);
    fn help(&mut self, page: &str);
    fn draw_image(&mut self);
    fn draw_model(&mut self);
    fn draw_current_lines(&mut self);
    fn draw_current_point(&mut self);
    fn draw_auto(&mut self);
    fn draw_tools(&mut self);
    fn scale_bar_draw(&mut self, width: i32, height: i32, zoom: f32, dpr: f32) -> f32;
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
    pub fn zoom_up(&mut self) {
        self.step_zoom(1);
    }
    pub fn zoom_down(&mut self) {
        self.step_zoom(-1);
    }
    pub fn step_zoom(&mut self, step: i32) {
        self.m_zoom *= if step > 0 { 1.2 } else { 1.0 / 1.2 };
        self.m_zoom = self.m_zoom.max(0.01);
    }
    pub fn entered_zoom(&mut self, zoom: f32) {
        self.m_zoom = zoom.max(0.01);
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
    pub fn thickness_changed(&mut self, view: XyzViewState, value: i32) {
        let old = self.m_thickness;
        self.m_thickness = value;
        if !view.has_pyramid_cache && old == 1 && value > 1 {
            self.allocate_sum_temp(
                (view.xsize * view.ysize)
                    .max(view.zsize * view.ysize)
                    .max(view.xsize * view.zsize)
                    .max(3) as usize,
                (view.xsize * view.ysize) as usize,
                view.ushort_store,
            );
        } else if value == 1 && old > 1 {
            self.m_sum_temp.clear();
            self.m_fdata_xy.clear();
        }
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
    pub fn center_clicked(&mut self, view: XyzViewState) {
        let (x, y, z) = self.get_location(view);
        self.m_xtrans1 = view.xsize / 2 - x;
        self.m_ytrans1 = view.ysize / 2 - y;
        self.m_xtrans2 = view.zsize / 2 - z;
        self.m_ytrans2 = self.m_xtrans2;
    }
    pub fn b1_drag(&mut self, view: XyzViewState, x: i32, y: i32) {
        let factor = if self.m_zoom < 1.0 {
            1.0 / self.m_zoom
        } else {
            1.0
        };
        let delx = ((x - self.m_lmx) as f32 * factor).round() as i32;
        let dely = ((y - self.m_lmy) as f32 * factor).round() as i32;
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
    pub fn step_time(&mut self, view: XyzViewState, step: i32) -> Option<i32> {
        if self.m_time_lock != 0 {
            self.m_time_lock = (self.m_time_lock + step).clamp(1, view.num_times);
            Some(self.m_time_lock)
        } else {
            None
        }
    }
    pub fn time_back(&mut self, view: XyzViewState) -> Option<i32> {
        self.step_time(view, -1)
    }
    pub fn time_forward(&mut self, view: XyzViewState) -> Option<i32> {
        self.step_time(view, 1)
    }
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
        n.draw_current_lines();
    }
    pub fn draw_current_point(&mut self, n: &mut dyn XyzNativeBoundary) {
        n.draw_current_point();
    }
    pub fn draw_auto(&mut self, n: &mut dyn XyzNativeBoundary) {
        n.draw_auto();
    }
    pub fn draw_ghost(&mut self, _n: &mut dyn XyzNativeBoundary) {}
    pub fn draw_contour(&mut self, _n: &mut dyn XyzNativeBoundary) {}
    pub fn draw_sym_proj(&mut self, _n: &mut dyn XyzNativeBoundary) {}
    pub fn draw_scat_sym_all_spheres(&mut self, _n: &mut dyn XyzNativeBoundary) {}
    pub fn draw_tools(&mut self, n: &mut dyn XyzNativeBoundary) {
        n.draw_tools();
    }
    pub fn close_event(&mut self, n: &mut dyn XyzNativeBoundary) {
        n.remove_control(self.m_ctrl);
        n.movie_xyzt(0, 0, 0, 0);
        self.m_fdata_xy.clear();
        self.m_fdata_xz.clear();
        self.m_fdata_yz.clear();
        self.m_sum_temp.clear();
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
}
impl XyzGl {
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
    pub fn resize_gl(&mut self, win: &mut XyzWindow, width: i32, height: i32) {
        win.m_winx = width;
        win.m_winy = height;
        win.get_ci_images();
        win.m_exposed = 1;
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
        win.m_scale_bar_size =
            n.scale_bar_draw(win.m_winx, win.m_winy, win.m_zoom, win.m_device_pixel_ratio);
        win.draw_tools(n);
    }
}

pub fn xyz_pixel_view_state(_state: bool) { /* `xyzPixelViewState`: Qt window-list traversal boundary. */
}
pub fn get_top_xyz() -> Option<()> {
    None /* dialog-manager lookup boundary */
}

#[cfg(test)]
mod tests {
    use super::*;
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
