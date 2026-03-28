use imod_fft::cross_correlate_2d;
use imod_math::min_max_mean;
use imod_mrc::MrcReader;
use imod_transforms::{write_xf_file, LinearTransform};
use rfd::FileDialog;
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};
use std::cell::RefCell;
use std::env;
use std::rc::Rc;

slint::include_modules!();

struct AlignerState {
    reader: Option<MrcReader>,
    nx: usize,
    ny: usize,
    nz: usize,
    black: f32,
    white: f32,
    current_z: usize,
    ref_z: usize,
    current_data: Vec<f32>,
    ref_data: Vec<f32>,
    transforms: Vec<LinearTransform>,
    filepath: String,
}

impl AlignerState {
    fn new() -> Self {
        Self {
            reader: None, nx: 0, ny: 0, nz: 0,
            black: 0.0, white: 255.0,
            current_z: 1, ref_z: 0,
            current_data: Vec::new(), ref_data: Vec::new(),
            transforms: Vec::new(), filepath: String::new(),
        }
    }

    fn open_file(&mut self, path: &str) -> Result<(), String> {
        let reader = MrcReader::open(path).map_err(|e| e.to_string())?;
        let h = reader.header();
        self.nx = h.nx as usize;
        self.ny = h.ny as usize;
        self.nz = h.nz as usize;
        self.black = h.amin;
        self.white = h.amax;
        self.transforms = vec![LinearTransform::identity(); self.nz];
        self.filepath = path.to_string();
        self.reader = Some(reader);
        self.load_pair(0, 1)?;
        Ok(())
    }

    fn load_pair(&mut self, ref_z: usize, cur_z: usize) -> Result<(), String> {
        if let Some(ref mut r) = self.reader {
            self.ref_z = ref_z.min(self.nz - 1);
            self.current_z = cur_z.min(self.nz - 1);
            self.ref_data = r.read_slice_f32(self.ref_z).map_err(|e| e.to_string())?;
            self.current_data = r.read_slice_f32(self.current_z).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn render_slice(&self, data: &[f32]) -> Image {
        if data.is_empty() || self.nx == 0 { return Image::default(); }
        let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(self.nx as u32, self.ny as u32);
        let pixels = buf.make_mut_bytes();
        let range = self.white - self.black;
        let scale = if range.abs() > 1e-10 { 255.0 / range } else { 1.0 };
        for (i, &v) in data.iter().enumerate() {
            let b = ((v - self.black) * scale).clamp(0.0, 255.0) as u8;
            let off = i * 4;
            if off + 3 < pixels.len() {
                pixels[off] = b; pixels[off+1] = b; pixels[off+2] = b; pixels[off+3] = 255;
            }
        }
        Image::from_rgba8(buf)
    }

    fn auto_align_current(&mut self) -> (f32, f32) {
        if self.ref_data.is_empty() || self.current_data.is_empty() { return (0.0, 0.0); }
        let fft_nx = next_pow2(self.nx);
        let fft_ny = next_pow2(self.ny);
        let rp = pad(&self.ref_data, self.nx, self.ny, fft_nx, fft_ny);
        let cp = pad(&self.current_data, self.nx, self.ny, fft_nx, fft_ny);
        let cc = cross_correlate_2d(&rp, &cp, fft_nx, fft_ny);
        let (px, py) = peak(&cc, fft_nx, fft_ny);
        let dx = if px > fft_nx / 2 { px as f32 - fft_nx as f32 } else { px as f32 };
        let dy = if py > fft_ny / 2 { py as f32 - fft_ny as f32 } else { py as f32 };
        (dx, dy)
    }

    fn info_string(&self) -> String {
        if self.ref_data.is_empty() { return String::new(); }
        let (rmin, rmax, rmean) = min_max_mean(&self.ref_data);
        let xf = &self.transforms[self.current_z];
        format!(
            "{}x{}x{}\nRef z={} Cur z={}\nRef: {:.0}..{:.0} mean={:.0}\nXf: dx={:.1} dy={:.1} rot={:.2}",
            self.nx, self.ny, self.nz, self.ref_z, self.current_z,
            rmin, rmax, rmean, xf.dx, xf.dy, xf.rotation_angle()
        )
    }
}

fn pad(data: &[f32], nx: usize, ny: usize, fnx: usize, fny: usize) -> Vec<f32> {
    let s: f64 = data.iter().map(|&v| v as f64).sum();
    let m = (s / data.len() as f64) as f32;
    let mut p = vec![m; fnx * fny];
    let (ox, oy) = ((fnx - nx) / 2, (fny - ny) / 2);
    for y in 0..ny { for x in 0..nx { p[(y+oy)*fnx+(x+ox)] = data[y*nx+x]; } }
    p
}

fn peak(cc: &[f32], nx: usize, ny: usize) -> (usize, usize) {
    let mut mv = f32::NEG_INFINITY;
    let (mut mx, mut my) = (0, 0);
    for y in 0..ny { for x in 0..nx { if cc[y*nx+x] > mv { mv = cc[y*nx+x]; mx = x; my = y; } } }
    (mx, my)
}

fn next_pow2(n: usize) -> usize { let mut p = 1; while p < n { p <<= 1; } p }

fn update_window(w: &MainWindow, s: &AlignerState) {
    w.set_ref_image(s.render_slice(&s.ref_data));
    w.set_cur_image(s.render_slice(&s.current_data));
    w.set_info_text(s.info_string().into());
}

fn main() {
    let window = MainWindow::new().unwrap();
    let state = Rc::new(RefCell::new(AlignerState::new()));

    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let mut s = state.borrow_mut();
        if let Err(e) = s.open_file(&args[1]) {
            eprintln!("Error: {}", e);
        } else {
            window.set_filename(args[1].clone().into());
            window.set_max_z((s.nz.saturating_sub(1)) as i32);
            window.set_nx(s.nx as i32);
            window.set_ny(s.ny as i32);
            window.set_black_level(s.black);
            window.set_white_level(s.white);
            window.set_current_z(1);
            window.set_ref_z(0);
            update_window(&window, &s);
        }
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_open_file(move || {
        let file = FileDialog::new().add_filter("MRC", &["mrc","st","ali","rec","map"]).pick_file();
        if let Some(path) = file {
            let ps = path.to_string_lossy().to_string();
            let mut s = st.borrow_mut();
            if let Err(e) = s.open_file(&ps) { eprintln!("Error: {}", e); return; }
            if let Some(w) = ww.upgrade() {
                w.set_filename(ps.into()); w.set_max_z((s.nz-1) as i32);
                w.set_black_level(s.black); w.set_white_level(s.white);
                update_window(&w, &s);
            }
        }
      });
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_z_changed(move |z| {
        let mut s = st.borrow_mut();
        let rz = s.ref_z;
        let _ = s.load_pair(rz, z as usize);
        if let Some(w) = ww.upgrade() { update_window(&w, &s); }
      });
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_ref_z_changed(move |z| {
        let mut s = st.borrow_mut();
        let cz = s.current_z;
        let _ = s.load_pair(z as usize, cz);
        if let Some(w) = ww.upgrade() { update_window(&w, &s); }
      });
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_contrast_changed(move |b, w_val| {
        let mut s = st.borrow_mut();
        s.black = b; s.white = w_val;
        if let Some(w) = ww.upgrade() { update_window(&w, &s); }
      });
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_transform_changed(move |dx, dy, rot, mag| {
        let mut s = st.borrow_mut();
        let z = s.current_z;
        let rad = rot * std::f32::consts::PI / 180.0;
        let c = rad.cos() * mag;
        let sv = rad.sin() * mag;
        s.transforms[z] = LinearTransform { a11: c, a12: -sv, a21: sv, a22: c, dx, dy };
        if let Some(w) = ww.upgrade() { w.set_info_text(s.info_string().into()); }
      });
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_auto_align(move || {
        let mut s = st.borrow_mut();
        let (dx, dy) = s.auto_align_current();
        let z = s.current_z;
        s.transforms[z].dx = dx;
        s.transforms[z].dy = dy;
        if let Some(w) = ww.upgrade() {
            w.set_dx(dx); w.set_dy(dy);
            w.set_info_text(s.info_string().into());
        }
      });
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_reset_transform(move || {
        let mut s = st.borrow_mut();
        let z = s.current_z;
        s.transforms[z] = LinearTransform::identity();
        if let Some(w) = ww.upgrade() {
            w.set_dx(0.0); w.set_dy(0.0); w.set_rotation(0.0); w.set_magnification(1.0);
            w.set_info_text(s.info_string().into());
        }
      });
    }

    { let st = state.clone();
      window.on_save_transforms(move || {
        let s = st.borrow();
        let xf_path = if s.filepath.ends_with(".st") {
            s.filepath.replace(".st", ".xf")
        } else {
            format!("{}.xf", s.filepath)
        };
        if let Err(e) = write_xf_file(&xf_path, &s.transforms) {
            eprintln!("Error saving: {}", e);
        } else {
            eprintln!("Saved {} transforms to {}", s.transforms.len(), xf_path);
        }
      });
    }

    window.on_apply_transform(|| {});

    window.run().unwrap();
}
