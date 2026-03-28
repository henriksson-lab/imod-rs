mod render3d;

use imod_core::Point3f;
use imod_math::min_max_mean;
use imod_mesh::{marching_cubes, IsosurfaceMesh};
use imod_model::{read_model, write_model, ImodContour, ImodModel, ImodObject};
use imod_mrc::MrcReader;
use rfd::FileDialog;
use slint::{Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::env;
use std::rc::Rc;

use render3d::Renderer3D;

slint::include_modules!();

/// Maximum number of undo states to keep.
const MAX_UNDO_HISTORY: usize = 50;

struct ViewerState {
    reader: Option<MrcReader>,
    nx: usize,
    ny: usize,
    nz: usize,
    current_z: usize,
    black: f32,
    white: f32,
    current_data: Vec<f32>,
    // Model overlay
    model: Option<ImodModel>,
    model_path: Option<String>,
    show_model: bool,
    current_object: usize,
    // View mode: 0=ZAP, 1=Slicer, 2=XYZ, 3=Model 3D
    view_mode: i32,
    slicer_angle_x: f32,
    slicer_angle_y: f32,
    // Volume cache for slicer/XYZ (loaded on demand)
    volume: Option<Vec<Vec<f32>>>,
    // 3D model renderer
    renderer3d: Renderer3D,
    // Mouse drag tracking for 3D rotation
    drag_last: Option<(f32, f32)>,
    // Isosurface
    iso_threshold: f32,
    iso_mesh: Option<IsosurfaceMesh>,
    // Undo/redo stacks: store cloned model states
    undo_stack: VecDeque<ImodModel>,
    redo_stack: Vec<ImodModel>,
}

impl ViewerState {
    fn new() -> Self {
        Self {
            reader: None,
            nx: 0, ny: 0, nz: 0,
            current_z: 0,
            black: 0.0,
            white: 255.0,
            current_data: Vec::new(),
            model: None,
            model_path: None,
            show_model: true,
            current_object: 0,
            view_mode: 0,
            slicer_angle_x: 0.0,
            slicer_angle_y: 0.0,
            volume: None,
            renderer3d: Renderer3D::new(800, 600),
            drag_last: None,
            iso_threshold: 128.0,
            iso_mesh: None,
            undo_stack: VecDeque::new(),
            redo_stack: Vec::new(),
        }
    }

    /// Push the current model state onto the undo stack before making a change.
    /// Clears the redo stack since we are branching from this point.
    fn push_undo(&mut self) {
        if let Some(ref model) = self.model {
            if self.undo_stack.len() >= MAX_UNDO_HISTORY {
                self.undo_stack.pop_front();
            }
            self.undo_stack.push_back(model.clone());
            self.redo_stack.clear();
        }
    }

    /// Undo: pop the last state from the undo stack, push current to redo, restore.
    fn undo(&mut self) -> bool {
        if let Some(prev) = self.undo_stack.pop_back() {
            if let Some(ref current) = self.model {
                self.redo_stack.push(current.clone());
            }
            self.model = Some(prev);
            true
        } else {
            false
        }
    }

    /// Redo: pop from redo stack, push current to undo, restore.
    fn redo(&mut self) -> bool {
        if let Some(next) = self.redo_stack.pop() {
            if let Some(ref current) = self.model {
                if self.undo_stack.len() >= MAX_UNDO_HISTORY {
                    self.undo_stack.pop_front();
                }
                self.undo_stack.push_back(current.clone());
            }
            self.model = Some(next);
            true
        } else {
            false
        }
    }

    fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    fn open_file(&mut self, path: &str) -> Result<(), String> {
        let reader = MrcReader::open(path).map_err(|e| e.to_string())?;
        let h = reader.header();
        self.nx = h.nx as usize;
        self.ny = h.ny as usize;
        self.nz = h.nz as usize;
        self.black = h.amin;
        self.white = h.amax;
        self.current_z = 0;
        self.volume = None;
        self.reader = Some(reader);
        // Try to load first slice; if the file has no image data (header-only), show blank
        if let Err(_) = self.load_slice(0) {
            self.current_data = vec![0.0; self.nx * self.ny];
            eprintln!("Warning: could not read image data (header-only file?)");
        }
        Ok(())
    }

    fn load_model(&mut self, path: &str) -> Result<(), String> {
        let model = read_model(path).map_err(|e| e.to_string())?;
        self.model = Some(model);
        self.model_path = Some(path.to_string());
        self.current_object = 0;
        // Clear undo/redo when loading a new model
        self.undo_stack.clear();
        self.redo_stack.clear();
        Ok(())
    }

    fn ensure_volume_loaded(&mut self) -> Result<(), String> {
        if self.volume.is_some() { return Ok(()); }
        if let Some(ref mut reader) = self.reader {
            let mut vol = Vec::with_capacity(self.nz);
            for z in 0..self.nz {
                vol.push(reader.read_slice_f32(z).map_err(|e| e.to_string())?);
            }
            self.volume = Some(vol);
        }
        Ok(())
    }

    fn load_slice(&mut self, z: usize) -> Result<(), String> {
        if let Some(ref vol) = self.volume {
            if z < vol.len() {
                self.current_data = vol[z].clone();
                self.current_z = z;
                return Ok(());
            }
        }
        if let Some(ref mut reader) = self.reader {
            self.current_data = reader.read_slice_f32(z).map_err(|e| e.to_string())?;
            self.current_z = z;
        }
        Ok(())
    }

    fn render_image(&mut self) -> Image {
        match self.view_mode {
            1 => self.render_slicer(),
            2 => self.render_xyz(),
            3 => self.render_model3d(),
            4 => self.render_isosurface_view(),
            _ => self.render_zap(),
        }
    }

    /// Extract and render an isosurface from the loaded volume.
    fn extract_isosurface(&mut self) {
        if self.volume.is_none() {
            return;
        }
        let vol = self.volume.as_ref().unwrap();
        // Flatten Vec<Vec<f32>> into a single contiguous Vec<f32>.
        let flat: Vec<f32> = vol.iter().flat_map(|s| s.iter().copied()).collect();
        let mesh = marching_cubes(&flat, self.nx, self.ny, self.nz, self.iso_threshold);
        self.iso_mesh = Some(mesh);
    }

    fn render_isosurface_view(&mut self) -> Image {
        let w = if self.nx > 0 { self.nx } else { 800 };
        let h = if self.ny > 0 { self.ny } else { 600 };
        self.renderer3d.resize(w, h);

        if self.iso_mesh.is_none() {
            self.extract_isosurface();
        }

        match &self.iso_mesh {
            Some(mesh) => self.renderer3d.render_isosurface(mesh),
            None => {
                // No volume / empty mesh -- render dark background.
                let empty = IsosurfaceMesh {
                    vertices: Vec::new(),
                    normals: Vec::new(),
                    indices: Vec::new(),
                };
                self.renderer3d.render_isosurface(&empty)
            }
        }
    }

    fn render_model3d(&mut self) -> Image {
        // Use image dimensions or a default size
        let w = if self.nx > 0 { self.nx } else { 800 };
        let h = if self.ny > 0 { self.ny } else { 600 };
        self.renderer3d.resize(w, h);
        match &self.model {
            Some(m) => {
                let m = m.clone();
                self.renderer3d.render_model(&m)
            }
            None => {
                // Render empty scene
                let empty = ImodModel::default();
                self.renderer3d.render_model(&empty)
            }
        }
    }

    fn render_zap(&self) -> Image {
        if self.current_data.is_empty() || self.nx == 0 || self.ny == 0 {
            return Image::default();
        }

        let mut pixel_buffer = SharedPixelBuffer::<Rgba8Pixel>::new(self.nx as u32, self.ny as u32);
        let pixels = pixel_buffer.make_mut_bytes();
        self.fill_grayscale(pixels, &self.current_data, self.nx, self.ny);

        // Draw model contours on this Z
        if self.show_model {
            if let Some(ref model) = self.model {
                self.draw_model_overlay(pixels, model, self.nx, self.ny, self.current_z as f32);
            }
        }

        Image::from_rgba8(pixel_buffer)
    }

    fn render_slicer(&self) -> Image {
        let vol = match &self.volume {
            Some(v) => v,
            None => return self.render_zap(), // fallback
        };

        let out_nx = self.nx;
        let out_ny = self.ny;
        let cx = self.nx as f32 / 2.0;
        let cy = self.ny as f32 / 2.0;
        let cz = self.current_z as f32;

        let ax = self.slicer_angle_x * std::f32::consts::PI / 180.0;
        let ay = self.slicer_angle_y * std::f32::consts::PI / 180.0;
        let cos_ax = ax.cos();
        let sin_ax = ax.sin();
        let cos_ay = ay.cos();
        let sin_ay = ay.sin();

        let mut data = vec![0.0f32; out_nx * out_ny];
        for oy in 0..out_ny {
            let dy = oy as f32 - cy;
            for ox in 0..out_nx {
                let dx = ox as f32 - cx;
                // Rotate the sampling plane
                let sx = dx * cos_ay + cx;
                let sy = dy * cos_ax + cy;
                let sz = cz + dx * sin_ay + dy * sin_ax;
                data[oy * out_nx + ox] = self.sample_volume(vol, sx, sy, sz);
            }
        }

        let mut pixel_buffer = SharedPixelBuffer::<Rgba8Pixel>::new(out_nx as u32, out_ny as u32);
        let pixels = pixel_buffer.make_mut_bytes();
        self.fill_grayscale(pixels, &data, out_nx, out_ny);
        Image::from_rgba8(pixel_buffer)
    }

    fn render_xyz(&self) -> Image {
        // XYZ view: show XY, XZ, and YZ planes in a single image
        let vol = match &self.volume {
            Some(v) => v,
            None => return self.render_zap(),
        };

        let gap = 2;
        let out_nx = self.nx + gap + self.nz;
        let out_ny = self.ny + gap + self.nz;

        let mut data = vec![self.black; out_nx * out_ny];

        // Top-left: XY at current Z
        if self.current_z < self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    data[y * out_nx + x] = vol[self.current_z][y * self.nx + x];
                }
            }
        }

        // Top-right: XZ at current Y (center)
        let mid_y = self.ny / 2;
        for z in 0..self.nz {
            for x in 0..self.nx {
                let ox = x;
                let oy = self.ny + gap + z;
                if oy < out_ny {
                    data[oy * out_nx + ox] = vol[z][mid_y * self.nx + x];
                }
            }
        }

        // Bottom-left: YZ at current X (center)
        let mid_x = self.nx / 2;
        for z in 0..self.nz {
            for y in 0..self.ny {
                let ox = self.nx + gap + z;
                let oy = y;
                if ox < out_nx {
                    data[oy * out_nx + ox] = vol[z][y * self.nx + mid_x];
                }
            }
        }

        let mut pixel_buffer = SharedPixelBuffer::<Rgba8Pixel>::new(out_nx as u32, out_ny as u32);
        let pixels = pixel_buffer.make_mut_bytes();
        self.fill_grayscale(pixels, &data, out_nx, out_ny);
        Image::from_rgba8(pixel_buffer)
    }

    fn sample_volume(&self, vol: &[Vec<f32>], x: f32, y: f32, z: f32) -> f32 {
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let z0 = z.floor() as isize;
        if x0 < 0 || x0 + 1 >= self.nx as isize
            || y0 < 0 || y0 + 1 >= self.ny as isize
            || z0 < 0 || z0 + 1 >= self.nz as isize
        {
            return self.black;
        }
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let fz = z - z0 as f32;
        let (x0, y0, z0) = (x0 as usize, y0 as usize, z0 as usize);

        let v00 = vol[z0][y0 * self.nx + x0] * (1.0 - fx) + vol[z0][y0 * self.nx + x0 + 1] * fx;
        let v10 = vol[z0][(y0 + 1) * self.nx + x0] * (1.0 - fx) + vol[z0][(y0 + 1) * self.nx + x0 + 1] * fx;
        let v01 = vol[z0 + 1][y0 * self.nx + x0] * (1.0 - fx) + vol[z0 + 1][y0 * self.nx + x0 + 1] * fx;
        let v11 = vol[z0 + 1][(y0 + 1) * self.nx + x0] * (1.0 - fx) + vol[z0 + 1][(y0 + 1) * self.nx + x0 + 1] * fx;

        let c0 = v00 * (1.0 - fy) + v10 * fy;
        let c1 = v01 * (1.0 - fy) + v11 * fy;
        c0 * (1.0 - fz) + c1 * fz
    }

    fn fill_grayscale(&self, pixels: &mut [u8], data: &[f32], _w: usize, _h: usize) {
        let range = self.white - self.black;
        let scale = if range.abs() > 1e-10 { 255.0 / range } else { 1.0 };

        for (i, &val) in data.iter().enumerate() {
            let byte = ((val - self.black) * scale).clamp(0.0, 255.0) as u8;
            let off = i * 4;
            if off + 3 < pixels.len() {
                pixels[off] = byte;
                pixels[off + 1] = byte;
                pixels[off + 2] = byte;
                pixels[off + 3] = 255;
            }
        }
    }

    fn draw_model_overlay(&self, pixels: &mut [u8], model: &ImodModel, w: usize, h: usize, z: f32) {
        for obj in &model.objects {
            let r = (obj.red * 255.0) as u8;
            let g = (obj.green * 255.0) as u8;
            let b = (obj.blue * 255.0) as u8;

            for cont in &obj.contours {
                // Draw points that are on or near this Z
                let mut prev: Option<(usize, usize)> = None;
                for pt in &cont.points {
                    if (pt.z - z).abs() > 0.6 { prev = None; continue; }
                    let px = pt.x.round() as isize;
                    let py = pt.y.round() as isize;
                    if px < 0 || px >= w as isize || py < 0 || py >= h as isize {
                        prev = None;
                        continue;
                    }
                    let (ux, uy) = (px as usize, py as usize);

                    // Draw a small cross at the point
                    for d in -2i32..=2 {
                        self.set_pixel(pixels, w, h, (px + d as isize) as usize, uy, r, g, b);
                        self.set_pixel(pixels, w, h, ux, (py + d as isize) as usize, r, g, b);
                    }

                    // Draw line to previous point
                    if let Some((prev_x, prev_y)) = prev {
                        self.draw_line(pixels, w, h, prev_x, prev_y, ux, uy, r, g, b);
                    }
                    prev = Some((ux, uy));
                }
            }
        }
    }

    fn set_pixel(&self, pixels: &mut [u8], w: usize, h: usize, x: usize, y: usize, r: u8, g: u8, b: u8) {
        if x < w && y < h {
            let off = (y * w + x) * 4;
            if off + 3 < pixels.len() {
                pixels[off] = r;
                pixels[off + 1] = g;
                pixels[off + 2] = b;
                pixels[off + 3] = 255;
            }
        }
    }

    fn draw_line(&self, pixels: &mut [u8], w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize, r: u8, g: u8, b: u8) {
        // Bresenham
        let dx = (x1 as isize - x0 as isize).abs();
        let dy = -(y1 as isize - y0 as isize).abs();
        let sx: isize = if x0 < x1 { 1 } else { -1 };
        let sy: isize = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut cx = x0 as isize;
        let mut cy = y0 as isize;
        loop {
            self.set_pixel(pixels, w, h, cx as usize, cy as usize, r, g, b);
            if cx == x1 as isize && cy == y1 as isize { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; cx += sx; }
            if e2 <= dx { err += dx; cy += sy; }
        }
    }

    fn pixel_value_at(&self, x: f32, y: f32) -> String {
        let ix = x.floor() as usize;
        let iy = y.floor() as usize;
        if ix < self.nx && iy < self.ny && !self.current_data.is_empty() {
            let val = self.current_data[iy * self.nx + ix];
            format!("({}, {}) = {:.2}", ix, iy, val)
        } else {
            String::new()
        }
    }

    fn info_string(&self) -> String {
        if self.current_data.is_empty() { return String::new(); }
        let (min, max, mean) = min_max_mean(&self.current_data);
        format!("{}x{}x{} | min={:.1} max={:.1} mean={:.1}", self.nx, self.ny, self.nz, min, max, mean)
    }

    fn model_info_string(&self) -> String {
        match &self.model {
            None => "No model".into(),
            Some(m) => {
                let total_cont: usize = m.objects.iter().map(|o| o.contours.len()).sum();
                format!("{} objects, {} contours", m.objects.len(), total_cont)
            }
        }
    }

    fn model_objects_list(&self) -> Vec<ModelObjectEntry> {
        match &self.model {
            None => Vec::new(),
            Some(m) => m.objects.iter().map(|obj| {
                ModelObjectEntry {
                    name: obj.name.clone().into(),
                    color: slint::Color::from_rgb_u8(
                        (obj.red * 255.0) as u8,
                        (obj.green * 255.0) as u8,
                        (obj.blue * 255.0) as u8,
                    ),
                    visible: true,
                    num_contours: obj.contours.len() as i32,
                }
            }).collect()
        }
    }

    /// Ensure a model exists for editing; create an empty one if needed.
    fn ensure_model(&mut self) {
        if self.model.is_none() {
            let mut m = ImodModel::default();
            m.xmax = self.nx as i32;
            m.ymax = self.ny as i32;
            m.zmax = self.nz as i32;
            // Create one default object with one empty contour
            let mut obj = ImodObject::default();
            obj.name = "Object 1".into();
            obj.contours.push(ImodContour::default());
            m.objects.push(obj);
            self.model = Some(m);
            self.current_object = 0;
        }
    }

    /// Add a point to the current object's current contour (matching current Z).
    fn add_point(&mut self, x: f32, y: f32) {
        self.push_undo();
        self.ensure_model();
        let z = self.current_z as f32;
        let model = self.model.as_mut().unwrap();
        let obj = &mut model.objects[self.current_object];

        // Find or create a contour for this Z slice
        let cont_idx = obj.contours.iter().position(|c| {
            c.points.first().map_or(false, |p| (p.z - z).abs() < 0.5)
        });
        let cont_idx = match cont_idx {
            Some(i) => i,
            None => {
                // If only contour is empty, use it; otherwise create new
                if obj.contours.len() == 1 && obj.contours[0].points.is_empty() {
                    0
                } else {
                    obj.contours.push(ImodContour::default());
                    obj.contours.len() - 1
                }
            }
        };
        obj.contours[cont_idx].points.push(Point3f { x, y, z });
    }

    /// Delete the nearest point within `radius` pixels of (x, y) on current Z.
    fn delete_nearest_point(&mut self, x: f32, y: f32, radius: f32) -> bool {
        let z = self.current_z as f32;
        let model = match self.model.as_ref() {
            Some(m) => m,
            None => return false,
        };

        let mut best_dist = radius * radius;
        let mut best: Option<(usize, usize, usize)> = None; // (obj, cont, pt)

        for (oi, obj) in model.objects.iter().enumerate() {
            for (ci, cont) in obj.contours.iter().enumerate() {
                for (pi, pt) in cont.points.iter().enumerate() {
                    if (pt.z - z).abs() > 0.6 { continue; }
                    let d = (pt.x - x).powi(2) + (pt.y - y).powi(2);
                    if d < best_dist {
                        best_dist = d;
                        best = Some((oi, ci, pi));
                    }
                }
            }
        }

        if let Some((oi, ci, pi)) = best {
            self.push_undo();
            let model = self.model.as_mut().unwrap();
            model.objects[oi].contours[ci].points.remove(pi);
            // Remove empty contours
            if model.objects[oi].contours[ci].points.is_empty() && model.objects[oi].contours.len() > 1 {
                model.objects[oi].contours.remove(ci);
            }
            true
        } else {
            false
        }
    }

    /// Save the model to file.
    fn save_model(&self) -> Result<(), String> {
        let model = self.model.as_ref().ok_or("No model to save")?;
        let path = self.model_path.as_deref().unwrap_or("output.mod");
        write_model(path, model).map_err(|e| e.to_string())?;
        eprintln!("Model saved to {}", path);
        Ok(())
    }
}

fn update_window(w: &MainWindow, s: &mut ViewerState) {
    w.set_info_text(s.info_string().into());
    w.set_slice_image(s.render_image());
    w.set_model_info(s.model_info_string().into());
    let objs = s.model_objects_list();
    let model: Rc<VecModel<ModelObjectEntry>> = Rc::new(VecModel::from(objs));
    w.set_model_objects(ModelRc::from(model));
    w.set_can_undo(s.can_undo());
    w.set_can_redo(s.can_redo());
}

fn main() {
    let window = MainWindow::new().unwrap();
    let state = Rc::new(RefCell::new(ViewerState::new()));

    // Parse command-line: imod-viewer [image.mrc] [model.mod]
    let args: Vec<String> = env::args().collect();

    {
        let mut s = state.borrow_mut();
        for arg in &args[1..] {
            if arg.ends_with(".mod") || arg.ends_with(".fid") {
                if let Err(e) = s.load_model(arg) {
                    eprintln!("Error loading model {}: {}", arg, e);
                }
            } else {
                if let Err(e) = s.open_file(arg) {
                    eprintln!("Error opening {}: {}", arg, e);
                } else {
                    window.set_filename(arg.into());
                    window.set_max_z((s.nz.saturating_sub(1)) as i32);
                    window.set_nx(s.nx as i32);
                    window.set_ny(s.ny as i32);
                    window.set_black_level(s.black);
                    window.set_white_level(s.white);
                }
            }
        }
        update_window(&window, &mut s);
    }

    // Z changed
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_z_changed(move |z| {
            let mut s = state.borrow_mut();
            if (z as usize) < s.nz {
                let _ = s.load_slice(z as usize);
                if let Some(w) = ww.upgrade() { update_window(&w, &mut s); }
            }
        });
    }

    // Contrast changed
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_contrast_changed(move |black, white| {
            let mut s = state.borrow_mut();
            s.black = black;
            s.white = white;
            if let Some(w) = ww.upgrade() { w.set_slice_image(s.render_image()); }
        });
    }

    // Mouse moved
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_mouse_moved(move |x, y| {
            let mut s = state.borrow_mut();
            if s.view_mode == 3 || s.view_mode == 4 {
                // In 3D / Iso mode, track mouse drag for rotation
                if let Some((lx, ly)) = s.drag_last {
                    let dx = x - lx;
                    let dy = y - ly;
                    if dx.abs() > 0.1 || dy.abs() > 0.1 {
                        s.renderer3d.rotate(dx, dy);
                        if let Some(w) = ww.upgrade() {
                            w.set_slice_image(s.render_image());
                        }
                    }
                }
                s.drag_last = Some((x, y));
            } else {
                if let Some(w) = ww.upgrade() {
                    w.set_pixel_value_text(s.pixel_value_at(x, y).into());
                }
            }
        });
    }

    // Toggle model
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_toggle_model(move |show| {
            let mut s = state.borrow_mut();
            s.show_model = show;
            if let Some(w) = ww.upgrade() { w.set_slice_image(s.render_image()); }
        });
    }

    // View mode changed
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_view_mode_changed(move |mode| {
            let mut s = state.borrow_mut();
            s.view_mode = mode;
            s.drag_last = None;
            if mode == 1 || mode == 2 || mode == 4 {
                let _ = s.ensure_volume_loaded();
            }
            if mode == 4 {
                s.iso_mesh = None; // force re-extraction
                s.extract_isosurface();
            }
            if let Some(w) = ww.upgrade() { w.set_slice_image(s.render_image()); }
        });
    }

    // Mouse released (reset drag state for 3D)
    {
        let state = state.clone();
        window.on_mouse_released(move || {
            let mut s = state.borrow_mut();
            s.drag_last = None;
        });
    }

    // Slicer angle changed
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_slicer_angle_changed(move |ax, ay| {
            let mut s = state.borrow_mut();
            s.slicer_angle_x = ax;
            s.slicer_angle_y = ay;
            if let Some(w) = ww.upgrade() { w.set_slice_image(s.render_image()); }
        });
    }

    // Iso threshold changed
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_iso_threshold_changed(move |thr| {
            let mut s = state.borrow_mut();
            s.iso_threshold = thr;
            s.iso_mesh = None; // invalidate cache
            s.extract_isosurface();
            if let Some(w) = ww.upgrade() { w.set_slice_image(s.render_image()); }
        });
    }

    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_zoom_changed(move |z| {
            let mut s = state.borrow_mut();
            if s.view_mode == 3 || s.view_mode == 4 {
                s.renderer3d.set_zoom(z);
                if let Some(w) = ww.upgrade() {
                    w.set_slice_image(s.render_image());
                }
            }
        });
    }

    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_open_file(move || {
            let file = FileDialog::new()
                .add_filter("MRC files", &["mrc", "st", "ali", "rec", "map"])
                .add_filter("All files", &["*"])
                .pick_file();
            if let Some(path) = file {
                let path_str = path.to_string_lossy().to_string();
                let mut s = state.borrow_mut();
                if let Err(e) = s.open_file(&path_str) {
                    eprintln!("Error opening {}: {}", path_str, e);
                } else if let Some(w) = ww.upgrade() {
                    w.set_filename(path_str.into());
                    w.set_max_z((s.nz.saturating_sub(1)) as i32);
                    w.set_nx(s.nx as i32);
                    w.set_ny(s.ny as i32);
                    w.set_black_level(s.black);
                    w.set_white_level(s.white);
                    update_window(&w, &mut s);
                }
            }
        });
    }

    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_open_model(move || {
            let file = FileDialog::new()
                .add_filter("Model files", &["mod", "fid"])
                .add_filter("All files", &["*"])
                .pick_file();
            if let Some(path) = file {
                let path_str = path.to_string_lossy().to_string();
                let mut s = state.borrow_mut();
                if let Err(e) = s.load_model(&path_str) {
                    eprintln!("Error loading model {}: {}", path_str, e);
                } else if let Some(w) = ww.upgrade() {
                    update_window(&w, &mut s);
                }
            }
        });
    }

    // Mouse left-click: add point in add mode, or delete in delete mode
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_mouse_clicked(move |x, y| {
            if let Some(w) = ww.upgrade() {
                let edit_mode = w.get_edit_mode();
                let mut s = state.borrow_mut();
                match edit_mode {
                    1 => {
                        // Add points mode
                        s.add_point(x, y);
                        w.set_slice_image(s.render_image());
                        update_window(&w, &mut s);
                        w.set_edit_status(format!("Added point at ({:.0}, {:.0})", x, y).into());
                    }
                    3 => {
                        // Delete points mode
                        if s.delete_nearest_point(x, y, 10.0) {
                            w.set_slice_image(s.render_image());
                            update_window(&w, &mut s);
                            w.set_edit_status("Deleted point".into());
                        } else {
                            w.set_edit_status("No point nearby".into());
                        }
                    }
                    _ => {}
                }
            }
        });
    }

    // Mouse right-click: delete nearest point (in any edit mode except view)
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_mouse_right_clicked(move |x, y| {
            if let Some(w) = ww.upgrade() {
                let edit_mode = w.get_edit_mode();
                if edit_mode != 0 {
                    let mut s = state.borrow_mut();
                    if s.delete_nearest_point(x, y, 10.0) {
                        w.set_slice_image(s.render_image());
                        update_window(&w, &mut s);
                        w.set_edit_status("Deleted point".into());
                    } else {
                        w.set_edit_status("No point nearby".into());
                    }
                }
            }
        });
    }

    // Save model
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_save_model(move || {
            let s = state.borrow();
            match s.save_model() {
                Ok(()) => {
                    if let Some(w) = ww.upgrade() {
                        let path = s.model_path.as_deref().unwrap_or("output.mod");
                        w.set_edit_status(format!("Saved to {}", path).into());
                    }
                }
                Err(e) => {
                    eprintln!("Error saving model: {}", e);
                    if let Some(w) = ww.upgrade() {
                        w.set_edit_status(format!("Save error: {}", e).into());
                    }
                }
            }
        });
    }

    // Undo
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_undo(move || {
            let mut s = state.borrow_mut();
            if s.undo() {
                if let Some(w) = ww.upgrade() {
                    update_window(&w, &mut s);
                    w.set_edit_status("Undo".into());
                }
            } else {
                if let Some(w) = ww.upgrade() {
                    w.set_edit_status("Nothing to undo".into());
                }
            }
        });
    }

    // Redo
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_redo(move || {
            let mut s = state.borrow_mut();
            if s.redo() {
                if let Some(w) = ww.upgrade() {
                    update_window(&w, &mut s);
                    w.set_edit_status("Redo".into());
                }
            } else {
                if let Some(w) = ww.upgrade() {
                    w.set_edit_status("Nothing to redo".into());
                }
            }
        });
    }

    // Key pressed handler (for actions that need Rust-side logic)
    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_key_pressed(move |key| {
            let key_str = key.as_str();
            match key_str {
                "PageUp" => {
                    let mut s = state.borrow_mut();
                    if s.current_z > 0 {
                        let new_z = s.current_z - 1;
                        let _ = s.load_slice(new_z);
                        if let Some(w) = ww.upgrade() {
                            w.set_current_z(new_z as i32);
                            update_window(&w, &mut s);
                        }
                    }
                }
                "PageDown" => {
                    let mut s = state.borrow_mut();
                    if s.current_z + 1 < s.nz {
                        let new_z = s.current_z + 1;
                        let _ = s.load_slice(new_z);
                        if let Some(w) = ww.upgrade() {
                            w.set_current_z(new_z as i32);
                            update_window(&w, &mut s);
                        }
                    }
                }
                "ZoomIn" => {
                    if let Some(w) = ww.upgrade() {
                        let z = w.get_zoom();
                        let new_z = (z * 1.2).min(20.0);
                        w.set_zoom(new_z);
                        w.invoke_zoom_changed(new_z);
                    }
                }
                "ZoomOut" => {
                    if let Some(w) = ww.upgrade() {
                        let z = w.get_zoom();
                        let new_z = (z / 1.2).max(0.05);
                        w.set_zoom(new_z);
                        w.invoke_zoom_changed(new_z);
                    }
                }
                "AddMode" => {
                    if let Some(w) = ww.upgrade() {
                        w.set_edit_status("Add points mode".into());
                    }
                }
                "DeleteMode" => {
                    if let Some(w) = ww.upgrade() {
                        w.set_edit_status("Delete points mode".into());
                    }
                }
                "ViewMode" => {
                    if let Some(w) = ww.upgrade() {
                        w.set_edit_status("View mode".into());
                    }
                }
                _ => {}
            }
        });
    }

    window.run().unwrap();
}
