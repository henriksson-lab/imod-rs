//! Translation of `IMOD/3dmod/xgraph.cpp` and `xgraph.h`.
//!
//! Qt widget creation, OpenGL submission, image access, and controller
//! callbacks are explicit boundary calls.  The graph's source-owned sampling,
//! clipping, scaling, labels, and state remain here.
#![allow(dead_code)]

pub const MAX_GRAPH_TOGGLES: usize = 2;
pub const GRAPH_XAXIS: i32 = 0;
pub const GRAPH_YAXIS: i32 = 1;
pub const GRAPH_ZAXIS: i32 = 2;
pub const GRAPH_CONTOUR: i32 = 3;
pub const GRAPH_HISTOGRAM: i32 = 4;
pub const IMOD_DRAW_COLORMAP: i32 = 1;
pub const IMOD_DRAW_XYZ: i32 = 2;
pub const IMOD_DRAW_ACTIVE: i32 = 4;
pub const IMOD_DRAW_IMAGE: i32 = 8;
pub const IMOD_DRAW_MOD: i32 = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ipoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Icont {
    pub pts: Vec<Ipoint>,
    pub current_point: i32,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphImageState {
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub xmouse: f32,
    pub ymouse: f32,
    pub zmouse: f32,
    pub image_mode: i32,
    pub file_is_jpeg: bool,
    pub no_readable_image: bool,
    pub ushort_store: bool,
    pub cache_base_index: i32,
    pub smin: f32,
    pub smax: f32,
}

/// Native viewer, Qt, cache and OpenGL endpoints called by `xgraph.cpp`.
pub trait XGraphNativeBoundary {
    fn image_state(&self) -> GraphImageState;
    fn subset_limits(&self) -> (i32, i32, i32, i32);
    fn contour(&self) -> Option<Icont>;
    fn file_value(&mut self, x: i32, y: i32, z: i32) -> f32;
    fn fast_value(&mut self, x: i32, y: i32, z: i32) -> f32;
    fn byte_section(&mut self) -> Option<Vec<Vec<u8>>>;
    fn ushort_section(&mut self) -> Option<Vec<Vec<u16>>>;
    fn ushort_byte_map(&mut self) -> Option<Vec<u8>>;
    fn setup_fast_access(&mut self) -> bool;
    fn setup_fast_tile_access(&mut self, cache: i32) -> bool;
    fn load_tiles(&mut self, cache: i32, x: i32, y: i32, w: i32, h: i32, z: i32);
    fn location(&self) -> (i32, i32, i32);
    fn set_location(&mut self, x: i32, y: i32, z: i32);
    fn draw(&mut self, flags: i32);
    fn update_gl(&mut self);
    fn control_priority(&mut self, ctrl: i32);
    fn remove_control(&mut self, ctrl: i32);
    fn close_window(&mut self);
    fn help(&mut self, page: &str);
    fn default_key(&mut self, key: i32);
    fn export(&mut self, rows: &[(i32, f32)]);
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphPlot {
    pub points: Vec<(i32, i32)>,
    pub cursor_x: i32,
    pub width: i32,
    pub height: i32,
}

/// `GraphWindow` from `xgraph.h`; former Qt controls are represented by their
/// source values, while native widget ownership stays at `XGraphNativeBoundary`.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphWindow {
    pub m_width: i32,
    pub m_height: i32,
    pub m_data: Vec<f32>,
    pub m_zoom: f32,
    pub m_axis: i32,
    pub m_locked: i32,
    pub m_ctrl: i32,
    pub m_start: i32,
    pub m_xcur: f32,
    pub m_ycur: f32,
    pub m_zcur: f32,
    pub m_device_pixel_ratio: f32,
    pub m_closing: i32,
    pub m_toggle_states: [i32; MAX_GRAPH_TOGGLES],
    pub m_data_size: i32,
    pub m_alloc_size: i32,
    pub m_center_pt: i32,
    pub m_obj_cur: i32,
    pub m_cont_cur: i32,
    pub m_pt_cur: i32,
    pub m_sub_start: i32,
    pub m_high_res: i32,
    pub m_num_lines: i32,
    pub m_offset: f32,
    pub m_scale: f32,
    pub m_min: f32,
    pub m_max: f32,
    pub m_mean: f32,
    pub m_end: i32,
    pub width_box_enabled: bool,
    pub labels: [String; 6],
}
impl Default for GraphWindow {
    fn default() -> Self {
        Self::new(0.)
    }
}
impl GraphWindow {
    /// `GraphWindow::GraphWindow`; layout/icon construction is native Qt boundary.
    pub fn new(dpr: f32) -> Self {
        Self {
            m_width: 320,
            m_height: 160,
            m_data: vec![],
            m_zoom: if dpr > 0. { dpr } else { 1. },
            m_axis: 0,
            m_locked: 0,
            m_ctrl: 0,
            m_start: 0,
            m_xcur: 0.,
            m_ycur: 0.,
            m_zcur: 0.,
            m_device_pixel_ratio: dpr,
            m_closing: 0,
            m_toggle_states: [0; 2],
            m_data_size: 0,
            m_alloc_size: 0,
            m_center_pt: 0,
            m_obj_cur: 0,
            m_cont_cur: 0,
            m_pt_cur: 0,
            m_sub_start: 0,
            m_high_res: 0,
            m_num_lines: 1,
            m_offset: 0.,
            m_scale: 0.,
            m_min: 0.,
            m_max: 0.,
            m_mean: 0.,
            m_end: 0,
            width_box_enabled: true,
            labels: core::array::from_fn(|_| String::new()),
        }
    }
    /// `GraphWindow::zoomUp`.
    pub fn zoom_up(&mut self) {
        self.m_zoom *= 1.25;
    }
    /// `GraphWindow::zoomDown`.
    pub fn zoom_down(&mut self) {
        self.m_zoom /= 1.25;
    }
    /// `GraphWindow::help`.
    pub fn help(&self, n: &mut dyn XGraphNativeBoundary) {
        n.help("graph.html#TOP");
    }
    /// `GraphWindow::toggleClicked`; return says whether source calls `draw`.
    pub fn toggle_clicked(&mut self, index: usize, state: bool) -> bool {
        if index >= 2 {
            return false;
        }
        self.m_toggle_states[index] = state as i32;
        if index == 0 {
            self.m_high_res = state as i32;
            true
        } else {
            self.m_locked = state as i32;
            !state
        }
    }
    /// `GraphWindow::axisSelected`.
    pub fn axis_selected(&mut self, item: i32) {
        self.m_axis = item;
        self.width_box_enabled = item != GRAPH_ZAXIS && item != GRAPH_HISTOGRAM;
    }
    /// `GraphWindow::setToggleState`.
    pub fn set_toggle_state(&mut self, index: usize, state: i32) {
        if index < 2 {
            self.m_toggle_states[index] = (state != 0) as i32;
        }
    }
    /// `GraphWindow::widthChanged`; return says whether source calls `draw`.
    pub fn width_changed(&mut self, value: i32) -> bool {
        if self.m_closing != 0 {
            return false;
        }
        self.m_num_lines = value;
        true
    }
    /// `GraphWindow::exportToFile`; file chooser/write is native boundary.
    pub fn export_to_file(&self, n: &mut dyn XGraphNativeBoundary) {
        n.export(
            &(0..self.m_data_size as usize)
                .map(|i| (i as i32 + self.m_sub_start, self.m_data[i]))
                .collect::<Vec<_>>(),
        );
    }
    /// `GraphWindow::keyPressEvent`.
    pub fn key_press_event(&mut self, key: i32, n: &mut dyn XGraphNativeBoundary) {
        n.control_priority(self.m_ctrl);
        match key {
            27 => self.close_event(n),
            61 | 43 => {
                self.zoom_up();
                self.draw(n)
            }
            45 => {
                self.zoom_down();
                self.draw(n)
            }
            _ => n.default_key(key),
        }
    }
    /// `GraphWindow::externalKeyEvent`.
    pub fn external_key_event(
        &mut self,
        key: i32,
        released: bool,
        n: &mut dyn XGraphNativeBoundary,
    ) {
        if !released {
            self.key_press_event(key, n)
        }
    }
    /// `GraphWindow::closeEvent`.
    pub fn close_event(&mut self, n: &mut dyn XGraphNativeBoundary) {
        self.m_closing = 1;
        n.remove_control(self.m_ctrl);
        self.m_data.clear();
        self.m_data_size = 0;
        n.close_window();
    }
    /// `GraphWindow::changeEvent`; Mac menu management is native boundary.
    pub fn change_event(&mut self, _font_change: bool) {}
    /// `GraphWindow::draw`.
    pub fn draw(&self, n: &mut dyn XGraphNativeBoundary) {
        n.update_gl();
    }
    /// `GraphWindow::allocDataArray`.
    pub fn alloc_data_array(&mut self, size: i32) -> i32 {
        if size < 0 {
            return 1;
        }
        if size > self.m_alloc_size {
            self.m_data = vec![0.; size as usize];
            self.m_alloc_size = size
        } else {
            self.m_data[..size as usize].fill(0.)
        }
        self.m_data_size = size;
        0
    }
    /// `GraphWindow::fillData`.
    pub fn fill_data(&mut self, n: &mut dyn XGraphNativeBoundary) {
        let s = n.image_state();
        let (ix, iy, nx, ny) = n.subset_limits();
        let (mut cx, mut cy, cz) = (s.xmouse as i32, s.ymouse as i32, (s.zmouse + 0.5) as i32);
        self.m_xcur = s.xmouse;
        self.m_ycur = s.ymouse;
        self.m_zcur = s.zmouse;
        let high = self.m_high_res != 0 && !s.file_is_jpeg;
        if self.m_axis != GRAPH_HISTOGRAM
            && !high
            && s.cache_base_index < 0
            && n.setup_fast_access()
        {
            return;
        }
        match self.m_axis {
            GRAPH_XAXIS => {
                self.m_sub_start = ix;
                if self.alloc_data_array(nx) != 0
                    || cz < 0
                    || cz >= s.zsize
                    || cy < 0
                    || cy >= s.ysize
                {
                    return;
                }
                if cx < ix || cx >= ix + nx {
                    cx = cx.clamp(ix, ix + nx - 1);
                    self.m_xcur = cx as f32;
                    n.set_location(cx, cy, cz);
                    n.draw(IMOD_DRAW_XYZ)
                }
                self.m_center_pt = cx;
                if s.cache_base_index >= 0 && !high {
                    n.load_tiles(
                        s.cache_base_index,
                        ix,
                        cy - self.m_num_lines / 2 - 1,
                        nx,
                        self.m_num_lines + 3,
                        cz,
                    );
                    if n.setup_fast_tile_access(s.cache_base_index) {
                        return;
                    }
                }
                let mut nl = 0;
                for j in 0..self.m_num_lines {
                    let y = cy + j - (self.m_num_lines - 1) / 2;
                    if y < 0 || y >= s.ysize {
                        continue;
                    }
                    nl += 1;
                    for i in 0..nx {
                        self.m_data[i as usize] += if high {
                            n.file_value(i + ix, y, cz)
                        } else {
                            n.fast_value(i + ix, y, cz)
                        }
                    }
                }
                if nl > 1 {
                    for d in &mut self.m_data {
                        *d /= nl as f32
                    }
                }
            }
            GRAPH_YAXIS => {
                self.m_sub_start = iy;
                if self.alloc_data_array(ny) != 0
                    || cx < 0
                    || cx >= s.xsize
                    || cz < 0
                    || cz >= s.zsize
                {
                    return;
                }
                if cy < iy || cy >= iy + ny {
                    cy = cy.clamp(iy, iy + ny - 1);
                    self.m_ycur = cy as f32;
                    n.set_location(cx, cy, cz);
                    n.draw(IMOD_DRAW_XYZ)
                }
                self.m_center_pt = cy;
                if s.cache_base_index >= 0 && !high {
                    n.load_tiles(
                        s.cache_base_index,
                        cx - self.m_num_lines / 2 - 1,
                        iy,
                        self.m_num_lines + 3,
                        ny,
                        cz,
                    );
                    if n.setup_fast_tile_access(s.cache_base_index) {
                        return;
                    }
                }
                let mut nl = 0;
                for j in 0..self.m_num_lines {
                    let x = cx + j - (self.m_num_lines - 1) / 2;
                    if x < 0 || x >= s.xsize {
                        continue;
                    }
                    nl += 1;
                    for i in 0..ny {
                        self.m_data[i as usize] += if high {
                            n.file_value(x, i + iy, cz)
                        } else {
                            n.fast_value(x, i + iy, cz)
                        }
                    }
                }
                if nl > 1 {
                    for d in &mut self.m_data {
                        *d /= nl as f32
                    }
                }
            }
            GRAPH_ZAXIS => {
                self.m_sub_start = 0;
                self.m_center_pt = cz;
                if self.alloc_data_array(s.zsize) != 0
                    || cx < 0
                    || cx >= s.xsize
                    || cy < 0
                    || cy >= s.ysize
                {
                    return;
                }
                if s.cache_base_index >= 0 && !high && n.setup_fast_tile_access(s.cache_base_index)
                {
                    return;
                }
                for i in 0..s.zsize {
                    self.m_data[i as usize] = if high {
                        n.file_value(cx, cy, i)
                    } else {
                        n.fast_value(cx, cy, i)
                    }
                }
            }
            GRAPH_CONTOUR => self.fill_contour_data(n, s, ix, iy, nx, ny, high),
            GRAPH_HISTOGRAM => self.fill_histogram_data(n, s, ix, iy, nx, ny, cx, cy, high),
            _ => return,
        }
        if self.m_axis != GRAPH_HISTOGRAM && self.m_data_size > 0 {
            self.m_mean = self.m_data.iter().sum::<f32>() / self.m_data_size as f32
        }
    }
    /// Contour portion of `GraphWindow::fillData`.
    pub fn fill_contour_data(
        &mut self,
        n: &mut dyn XGraphNativeBoundary,
        s: GraphImageState,
        ix: i32,
        iy: i32,
        nx: i32,
        ny: i32,
        high: bool,
    ) {
        let c = match n.contour() {
            Some(c) if c.pts.len() >= 2 => c,
            _ => return,
        };
        let p = &c.pts;
        let mut cp = c.current_point.clamp(0, p.len() as i32 - 1) as usize;
        self.m_pt_cur = cp as i32;
        let (xe, ye) = (ix + nx - 1, iy + ny - 1);
        let inside = |q: Ipoint| {
            q.x >= ix as f32 && q.x <= xe as f32 && q.y >= iy as f32 && q.y <= ye as f32
        };
        let (mut start, mut end, mut prefix, mut first, mut last) =
            (0, p.len(), 0., p[0], *p.last().unwrap());
        if nx < s.xsize || ny < s.ysize {
            let mut found = None;
            for i in 0..p.len() {
                if inside(p[i]) {
                    found = Some(i);
                    break;
                }
                if i > 0 {
                    prefix += self.point_distance(p[i - 1], p[i])
                }
            }
            let Some(i) = found else { return };
            start = i;
            if i > 0 {
                first = self.make_boundary_point(p[i - 1], p[i], ix, xe, iy, ye);
                prefix += self.point_distance(p[i - 1], first)
            }
            for i in i + 1..p.len() {
                if !inside(p[i]) {
                    end = i + 1;
                    last = self.make_boundary_point(p[i], p[i - 1], ix, xe, iy, ye);
                    break;
                }
            }
        }
        if s.cache_base_index >= 0 && !high && n.setup_fast_tile_access(s.cache_base_index) {
            return;
        }
        self.m_sub_start = (prefix + 0.5) as i32;
        let (mut len, mut prev) = (0., first);
        for i in start.max(1)..end {
            let q = if i == end - 1 && end < p.len() {
                last
            } else {
                p[i]
            };
            len += self.point_distance(prev, q);
            if i == cp {
                self.m_center_pt = (len + 0.5) as i32 + self.m_sub_start
            }
            prev = q
        }
        if cp >= end {
            self.m_center_pt = (len + 0.5) as i32 + self.m_sub_start
        }
        if cp < start {
            cp = start
        }
        self.m_xcur = p[cp].x + 0.5;
        self.m_ycur = p[cp].y + 0.5;
        self.m_zcur = p[cp].z + 0.5;
        if self.alloc_data_array((len + 1.) as i32) != 0 {
            return;
        }
        let (mut total, mut interval, mut cur, mut p1, mut p2, mut vecpt, mut dx, mut dy) =
            (0., 0., start.max(1), first, first, usize::MAX, 0., 0.);
        for i in 0..self.m_data_size {
            while i as f32 > total + interval || interval == 0. {
                total += interval;
                p1 = p2;
                if cur >= end {
                    break;
                }
                p2 = if cur == end - 1 && end < p.len() {
                    last
                } else {
                    p[cur]
                };
                cur += 1;
                interval = self.point_distance(p1, p2)
            }
            let frac = if interval != 0. {
                (i as f32 - total) / interval
            } else {
                0.
            };
            if self.m_num_lines > 1 && cur != vecpt {
                dx = p1.y - p2.y;
                dy = p2.x - p1.x;
                let d = (dx * dx + dy * dy).sqrt();
                if d > 1.0e-3 {
                    dx /= d;
                    dy /= d
                } else {
                    dx = 0.;
                    dy = 1.
                }
                vecpt = cur
            }
            let mut nl = 0;
            for cross in
                -((self.m_num_lines - 1) / 2)..self.m_num_lines - (self.m_num_lines - 1) / 2
            {
                let x = (p1.x + frac * (p2.x - p1.x) + cross as f32 * dx + 0.5) as i32;
                let y = (p1.y + frac * (p2.y - p1.y) + cross as f32 * dy + 0.5) as i32;
                let z = (p1.z + frac * (p2.z - p1.z) + 0.5) as i32;
                if x >= 0 && x < s.xsize && y >= 0 && y < s.ysize && z >= 0 && z < s.zsize {
                    nl += 1;
                    self.m_data[i as usize] += if high {
                        n.file_value(x, y, z)
                    } else {
                        n.fast_value(x, y, z)
                    }
                }
            }
            if nl > 0 {
                self.m_data[i as usize] /= nl as f32
            }
        }
    }
    /// Histogram portion of `GraphWindow::fillData`.
    pub fn fill_histogram_data(
        &mut self,
        n: &mut dyn XGraphNativeBoundary,
        s: GraphImageState,
        ix: i32,
        iy: i32,
        nx: i32,
        ny: i32,
        cx: i32,
        cy: i32,
        high: bool,
    ) {
        if self.alloc_data_array(256) != 0 {
            return;
        }
        self.m_sub_start = 0;
        let mut sum = 0.;
        if s.ushort_store {
            let (Some(im), Some(map)) = (n.ushort_section(), n.ushort_byte_map()) else {
                return;
            };
            for y in iy..iy + ny {
                for x in ix..ix + nx {
                    let v = map[im[y as usize][x as usize] as usize] as usize;
                    self.m_data[v] += 1.;
                    sum += v as f32
                }
            }
            self.m_center_pt = map[im[cy as usize][cx as usize] as usize] as i32
        } else {
            let Some(im) = n.byte_section() else { return };
            for y in iy..iy + ny {
                for x in ix..ix + nx {
                    let v = im[y as usize][x as usize] as usize;
                    self.m_data[v] += 1.;
                    sum += v as f32
                }
            }
            self.m_center_pt = im[cy as usize][cx as usize] as i32
        }
        if high && s.smin != s.smax {
            sum = sum * (s.smax - s.smin) / 255. + s.smin
        }
        self.m_mean = sum / (nx * ny) as f32
    }
    /// `GraphWindow::drawAxis`.
    pub fn draw_axis(&mut self, s: GraphImageState) {
        let f = self.m_high_res != 0 && !s.file_is_jpeg && !matches!(s.image_mode, 0 | 1 | 6);
        self.labels[0] = if f {
            format!("{:90.5}", self.m_min)
        } else {
            format!("{:6}", self.m_min as i32)
        };
        self.labels[1] = if f {
            format!("{:90.5}", self.m_max)
        } else {
            format!("{:6}", self.m_max as i32)
        };
        self.labels[2] = self.m_start.to_string();
        self.labels[3] = ((self.m_start + self.m_end) / 2).to_string();
        self.labels[4] = self.m_end.to_string();
        self.labels[5] = format!(" {:0.5}", self.m_mean);
    }
    /// `GraphWindow::drawPlot`.
    pub fn draw_plot(&mut self) -> GraphPlot {
        if self.m_data.is_empty() {
            return GraphPlot::default();
        }
        let mut zoom = self.m_zoom;
        let mut st = self.m_center_pt - (self.m_width as f32 / 2. / zoom) as i32;
        let mut en = self.m_center_pt + (self.m_width as f32 / 2. / zoom) as i32;
        if self.m_axis == GRAPH_HISTOGRAM {
            st = st.max(0);
            en = en.min(256);
            if en > st {
                zoom = self.m_width as f32 / (en - st) as f32
            }
        }
        let (mut min, mut max) = (1.0e37f32, -1.0e37f32);
        for i in st - self.m_sub_start..en - self.m_sub_start {
            if i >= 0 && (i as usize) < self.m_data.len() {
                min = min.min(self.m_data[i as usize]);
                max = max.max(self.m_data[i as usize])
            }
        }
        let mut extra = 0.02 * (max - min);
        if extra != extra as i32 as f32 && (max == max as i32 as f32 || min == min as i32 as f32) {
            extra = if extra > 0.2 {
                (extra + 1.) as i32 as f32
            } else {
                0.
            }
        }
        if min != 0. {
            min -= extra
        }
        max += extra;
        self.m_min = min;
        self.m_max = max;
        let scale = if max - min != 0. {
            self.m_height as f32 / (max - min)
        } else {
            1.
        };
        let mut points = vec![];
        for i in st - self.m_sub_start..=en - self.m_sub_start {
            if i >= 0 && (i as usize) < self.m_data.len() {
                points.push((
                    ((i + self.m_sub_start - st) as f32 * zoom) as i32,
                    ((self.m_data[i as usize] - min) * scale) as i32,
                ))
            }
        }
        self.m_offset = min;
        self.m_scale = scale;
        self.m_start = st;
        self.m_end = en;
        GraphPlot {
            points,
            cursor_x: ((self.m_center_pt - st) as f32 * zoom) as i32,
            width: self.m_width,
            height: self.m_height,
        }
    }
    /// `GraphWindow::makeBoundaryPoint`.
    pub fn make_boundary_point(
        &self,
        a: Ipoint,
        b: Ipoint,
        ix1: i32,
        ix2: i32,
        iy1: i32,
        iy2: i32,
    ) -> Ipoint {
        let mut tm: f32 = 0.0;
        if (a.x - b.x).abs() > 1.0e-4 {
            for q in [ix1 as f32, ix2 as f32] {
                let t = (q - a.x) / (b.x - a.x);
                if (0. ..1.).contains(&t) {
                    tm = tm.max(t)
                }
            }
        }
        if (a.y - b.y).abs() > 1.0e-4 {
            for q in [iy1 as f32, iy2 as f32] {
                let t = (q - a.y) / (b.y - a.y);
                if (0. ..1.).contains(&t) {
                    tm = tm.max(t)
                }
            }
        }
        Ipoint {
            x: tm * (b.x - a.x) + a.x,
            y: tm * (b.y - a.y) + a.y,
            z: tm * (b.z - a.z) + a.z,
        }
    }
    /// `imodPoint3DScaleDistance` call made by this source unit (unit scale).
    pub fn point_distance(&self, a: Ipoint, b: Ipoint) -> f32 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
    }
}
/// `GraphGL` from `xgraph.h`; actual `CurGLWidget` is native boundary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphGl {
    pub m_drawing: bool,
}
impl GraphGl {
    /// `GraphGL::paintGL`.
    pub fn paint_gl(
        &mut self,
        g: &mut GraphWindow,
        n: &mut dyn XGraphNativeBoundary,
    ) -> Option<GraphPlot> {
        if self.m_drawing {
            return None;
        }
        self.m_drawing = true;
        if g.m_locked == 0 {
            g.fill_data(n)
        }
        let ret = if g.m_data.is_empty() {
            None
        } else {
            let p = g.draw_plot();
            g.draw_axis(n.image_state());
            Some(p)
        };
        self.m_drawing = false;
        ret
    }
    /// `GraphGL::resizeGL`.
    pub fn resize_gl(&self, g: &mut GraphWindow, w: i32, h: i32) {
        g.m_width = w;
        g.m_height = h;
    }
    /// `GraphGL::mousePressEvent`.
    pub fn mouse_press_event(
        &self,
        g: &mut GraphWindow,
        x: i32,
        left: bool,
        n: &mut dyn XGraphNativeBoundary,
    ) {
        n.control_priority(g.m_ctrl);
        if left {
            self.setxyz(g, x, n)
        }
    }
    /// `GraphGL::setxyz`.
    pub fn setxyz(&self, g: &mut GraphWindow, mx: i32, n: &mut dyn XGraphNativeBoundary) {
        let ni = (mx as f32 / g.m_zoom) as i32 + g.m_start;
        let (mut x, mut y, mut z) = n.location();
        match g.m_axis {
            GRAPH_XAXIS => x = ni,
            GRAPH_YAXIS => y = ni,
            GRAPH_ZAXIS => z = ni,
            _ => return,
        }
        n.set_location(x, y, z);
        n.control_priority(g.m_ctrl);
        n.draw(IMOD_DRAW_XYZ);
    }
}
/// `xgraphOpen`; Qt/control setup supplies `ctrl` at boundary.
pub fn xgraph_open(dpr: f32, ctrl: i32) -> GraphWindow {
    let mut g = GraphWindow::new(dpr);
    g.m_ctrl = ctrl;
    g
}
/// `graphClose_cb`.
pub fn graph_close_cb(g: &mut GraphWindow, n: &mut dyn XGraphNativeBoundary) {
    g.close_event(n)
}
/// `graphDraw_cb`.
pub fn graph_draw_cb(g: &mut GraphWindow, flags: i32, n: &mut dyn XGraphNativeBoundary) {
    if g.m_closing != 0 {
        return;
    }
    if flags & IMOD_DRAW_COLORMAP != 0 {
        return;
    }
    let s = n.image_state();
    if flags & IMOD_DRAW_XYZ != 0
        && g.m_locked == 0
        && (g.m_xcur != s.xmouse || g.m_ycur != s.ymouse || g.m_zcur != s.zmouse)
    {
        g.draw(n);
        return;
    }
    if flags & (IMOD_DRAW_ACTIVE | IMOD_DRAW_IMAGE) != 0
        || (flags & IMOD_DRAW_MOD != 0 && g.m_axis == GRAPH_CONTOUR)
    {
        g.draw(n)
    }
}
/// `graphKey_cb`.
pub fn graph_key_cb(
    g: &mut GraphWindow,
    released: bool,
    key: i32,
    n: &mut dyn XGraphNativeBoundary,
) {
    g.external_key_event(key, released, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        s: GraphImageState,
        p: Vec<Vec<u8>>,
        loc: (i32, i32, i32),
    }
    impl XGraphNativeBoundary for N {
        fn image_state(&self) -> GraphImageState {
            self.s
        }
        fn subset_limits(&self) -> (i32, i32, i32, i32) {
            (0, 0, self.s.xsize, self.s.ysize)
        }
        fn contour(&self) -> Option<Icont> {
            None
        }
        fn file_value(&mut self, x: i32, y: i32, _: i32) -> f32 {
            self.p[y as usize][x as usize] as f32
        }
        fn fast_value(&mut self, x: i32, y: i32, z: i32) -> f32 {
            self.file_value(x, y, z)
        }
        fn byte_section(&mut self) -> Option<Vec<Vec<u8>>> {
            Some(self.p.clone())
        }
        fn ushort_section(&mut self) -> Option<Vec<Vec<u16>>> {
            None
        }
        fn ushort_byte_map(&mut self) -> Option<Vec<u8>> {
            None
        }
        fn setup_fast_access(&mut self) -> bool {
            false
        }
        fn setup_fast_tile_access(&mut self, _: i32) -> bool {
            false
        }
        fn load_tiles(&mut self, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32) {}
        fn location(&self) -> (i32, i32, i32) {
            self.loc
        }
        fn set_location(&mut self, x: i32, y: i32, z: i32) {
            self.loc = (x, y, z)
        }
        fn draw(&mut self, _: i32) {}
        fn update_gl(&mut self) {}
        fn control_priority(&mut self, _: i32) {}
        fn remove_control(&mut self, _: i32) {}
        fn close_window(&mut self) {}
        fn help(&mut self, _: &str) {}
        fn default_key(&mut self, _: i32) {}
        fn export(&mut self, _: &[(i32, f32)]) {}
    }
    #[test]
    fn x_axis_averages_lines() {
        let mut n = N {
            s: GraphImageState {
                xsize: 3,
                ysize: 2,
                zsize: 1,
                ..Default::default()
            },
            p: vec![vec![1, 2, 3], vec![4, 5, 6]],
            ..Default::default()
        };
        let mut g = GraphWindow::default();
        g.m_num_lines = 2;
        g.fill_data(&mut n);
        assert_eq!(g.m_data, vec![2.5, 3.5, 4.5]);
    }
    #[test]
    fn boundary_point_matches_source_intersection() {
        let g = GraphWindow::default();
        assert_eq!(
            g.make_boundary_point(
                Ipoint {
                    x: -2.,
                    y: 2.,
                    z: 0.
                },
                Ipoint {
                    x: 2.,
                    y: 2.,
                    z: 0.
                },
                0,
                3,
                0,
                3
            ),
            Ipoint {
                x: 0.,
                y: 2.,
                z: 0.
            }
        );
    }
    #[test]
    fn plot_and_click_map_axis() {
        let mut n = N {
            s: GraphImageState {
                xsize: 3,
                ysize: 1,
                zsize: 1,
                ..Default::default()
            },
            p: vec![vec![1, 3, 2]],
            ..Default::default()
        };
        let mut g = GraphWindow::default();
        g.fill_data(&mut n);
        g.m_width = 30;
        assert_eq!(g.draw_plot().points.len(), 3);
        GraphGl::default().setxyz(&mut g, 17, &mut n);
        assert_eq!(n.loc.0, 2);
    }
}
