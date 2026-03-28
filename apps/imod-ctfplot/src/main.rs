use imod_fft::{fft_r2c_2d, power_spectrum};
use imod_mrc::MrcReader;
use rfd::FileDialog;
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};
use std::cell::RefCell;
use std::env;
use std::f64::consts::PI;
use std::rc::Rc;

slint::include_modules!();

struct CtfState {
    reader: Option<MrcReader>,
    nx: usize,
    ny: usize,
    nz: usize,
    current_z: usize,
    voltage: f64,
    cs: f64,
    defocus: f64,
    pixel_size: f64,
    amp_contrast: f64,
    current_data: Vec<f32>,
    radial_avg: Vec<f32>,
    per_section_defocus: Vec<f64>,
    filepath: String,
}

impl CtfState {
    fn new() -> Self {
        Self {
            reader: None, nx: 0, ny: 0, nz: 0, current_z: 0,
            voltage: 300.0, cs: 2.7, defocus: 3.0, pixel_size: 1.0, amp_contrast: 0.07,
            current_data: Vec::new(), radial_avg: Vec::new(),
            per_section_defocus: Vec::new(), filepath: String::new(),
        }
    }

    fn open_file(&mut self, path: &str) -> Result<(), String> {
        let reader = MrcReader::open(path).map_err(|e| e.to_string())?;
        let h = reader.header();
        self.nx = h.nx as usize;
        self.ny = h.ny as usize;
        self.nz = h.nz as usize;
        self.pixel_size = h.pixel_size_x() as f64;
        self.per_section_defocus = vec![self.defocus; self.nz];
        self.filepath = path.to_string();
        self.reader = Some(reader);
        self.load_section(0)?;
        Ok(())
    }

    fn load_section(&mut self, z: usize) -> Result<(), String> {
        if let Some(ref mut r) = self.reader {
            self.current_data = r.read_slice_f32(z).map_err(|e| e.to_string())?;
            self.current_z = z;
            self.defocus = self.per_section_defocus[z];
            self.compute_radial_avg();
        }
        Ok(())
    }

    fn compute_radial_avg(&mut self) {
        if self.current_data.is_empty() { return; }
        let nxc = self.nx / 2 + 1;
        let freq = fft_r2c_2d(&self.current_data, self.nx, self.ny);
        let ps = power_spectrum(&freq);

        // Radial average
        let max_r = (self.nx.min(self.ny) / 2) as usize;
        let mut sums = vec![0.0f64; max_r];
        let mut counts = vec![0usize; max_r];

        for j in 0..self.ny {
            let fy = if j <= self.ny / 2 { j } else { self.ny - j };
            for i in 0..nxc {
                let r = ((i * i + fy * fy) as f64).sqrt() as usize;
                if r < max_r {
                    sums[r] += ps[j * nxc + i] as f64;
                    counts[r] += 1;
                }
            }
        }

        self.radial_avg = sums.iter().zip(counts.iter())
            .map(|(&s, &c)| if c > 0 { (s / c as f64).log10().max(-10.0) as f32 } else { 0.0 })
            .collect();
    }

    fn render_power_spectrum(&self) -> Image {
        if self.current_data.is_empty() { return Image::default(); }
        let nxc = self.nx / 2 + 1;
        let freq = fft_r2c_2d(&self.current_data, self.nx, self.ny);
        let ps = power_spectrum(&freq);

        // Log scale
        let log_ps: Vec<f32> = ps.iter().map(|&v| (v as f64 + 1.0).log10() as f32).collect();
        let max_val = log_ps.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = log_ps.iter().cloned().fold(f32::INFINITY, f32::min);
        let range = (max_val - min_val).max(1e-6);

        // Render centered power spectrum
        let out_size = self.nx.min(self.ny);
        let half = out_size / 2;
        let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(out_size as u32, out_size as u32);
        let pixels = buf.make_mut_bytes();

        for oy in 0..out_size {
            for ox in 0..out_size {
                let fx = (ox as isize - half as isize).unsigned_abs();
                let fy_signed = oy as isize - half as isize;
                let fy = if fy_signed < 0 { (self.ny as isize + fy_signed) as usize } else { fy_signed as usize };

                let val = if fx < nxc && fy < self.ny {
                    let v = log_ps[fy * nxc + fx];
                    ((v - min_val) / range * 255.0).clamp(0.0, 255.0) as u8
                } else { 0 };

                let off = (oy * out_size + ox) * 4;
                pixels[off] = val; pixels[off+1] = val; pixels[off+2] = val; pixels[off+3] = 255;
            }
        }

        Image::from_rgba8(buf)
    }

    fn render_ctf_curve(&self) -> Image {
        if self.radial_avg.is_empty() { return Image::default(); }

        let w = 512usize;
        let h = 150usize;
        let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w as u32, h as u32);
        let pixels = buf.make_mut_bytes();

        // Black background
        for p in pixels.chunks_mut(4) { p[0] = 20; p[1] = 20; p[2] = 20; p[3] = 255; }

        let n = self.radial_avg.len();
        let min_v = self.radial_avg.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_v = self.radial_avg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_v - min_v).max(1e-6);

        // Draw radial average (green)
        for x in 0..w {
            let ri = x * n / w;
            if ri >= n { break; }
            let y = ((self.radial_avg[ri] - min_v) / range * (h - 4) as f32) as usize;
            let py = h - 2 - y.min(h - 3);
            let off = (py * w + x) * 4;
            pixels[off] = 0; pixels[off+1] = 255; pixels[off+2] = 0; pixels[off+3] = 255;
        }

        // Draw theoretical CTF (red)
        let wavelength = electron_wavelength(self.voltage);
        let cs_a = self.cs * 1e7;
        let def_a = self.defocus * 1e4; // um -> Angstroms
        let px = self.pixel_size;
        let w2 = self.amp_contrast;
        let w1 = (1.0 - w2 * w2).sqrt();

        for x in 0..w {
            let s = (x as f64 + 0.5) / w as f64 / (2.0 * px); // spatial frequency
            let s2 = s * s;
            let chi = PI * wavelength * s2 * (def_a - 0.5 * cs_a * wavelength * wavelength * s2);
            let ctf = -(w1 * chi.sin() - w2 * chi.cos());
            let ctf2 = (ctf * ctf) as f32;
            // Map CTF^2 to same vertical range
            let y = (ctf2 * (h - 4) as f32) as usize;
            let py = h - 2 - y.min(h - 3);
            let off = (py * w + x) * 4;
            pixels[off] = 255; pixels[off+1] = 50; pixels[off+2] = 50; pixels[off+3] = 255;
        }

        Image::from_rgba8(buf)
    }

    fn fit_defocus_auto(&mut self) {
        // Simple 1D search: find defocus that best matches radial average to CTF^2
        if self.radial_avg.is_empty() { return; }

        let wavelength = electron_wavelength(self.voltage);
        let cs_a = self.cs * 1e7;
        let px = self.pixel_size;
        let w2 = self.amp_contrast;
        let w1 = (1.0 - w2 * w2).sqrt();
        let n = self.radial_avg.len();

        // Normalize radial average
        let min_v = self.radial_avg.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_v = self.radial_avg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_v - min_v).max(1e-6);
        let norm: Vec<f32> = self.radial_avg.iter().map(|&v| (v - min_v) / range).collect();

        let mut best_def = self.defocus;
        let mut best_score = f64::MAX;

        // Search from 0.5 to 15 um in 0.01 um steps
        let mut def = 0.5;
        while def <= 15.0 {
            let def_a = def * 1e4;
            let mut score = 0.0;
            for i in 1..n {
                let s = i as f64 / n as f64 / (2.0 * px);
                let s2 = s * s;
                let chi = PI * wavelength * s2 * (def_a - 0.5 * cs_a * wavelength * wavelength * s2);
                let ctf = -(w1 * chi.sin() - w2 * chi.cos());
                let ctf2 = ctf * ctf;
                let diff = norm[i] as f64 - ctf2;
                score += diff * diff;
            }
            if score < best_score { best_score = score; best_def = def; }
            def += 0.01;
        }

        self.defocus = best_def;
        self.per_section_defocus[self.current_z] = best_def;
    }

    fn info_string(&self) -> String {
        format!(
            "Defocus: {:.2} um\nVoltage: {:.0} kV\nCs: {:.1} mm\nPixel: {:.2} A\nAmp.C: {:.2}",
            self.defocus, self.voltage, self.cs, self.pixel_size, self.amp_contrast
        )
    }
}

fn electron_wavelength(voltage_kv: f64) -> f64 {
    let v = voltage_kv * 1000.0;
    let m0 = 9.10938e-31;
    let e = 1.60218e-19;
    let c = 2.99792e8;
    let h = 6.62607e-34;
    let lambda_m = h / (2.0 * m0 * e * v * (1.0 + e * v / (2.0 * m0 * c * c))).sqrt();
    lambda_m * 1e10
}

fn update_window(w: &MainWindow, s: &CtfState) {
    w.set_power_spectrum(s.render_power_spectrum());
    w.set_ctf_curve(s.render_ctf_curve());
    w.set_info_text(s.info_string().into());
}

fn main() {
    let window = MainWindow::new().unwrap();
    let state = Rc::new(RefCell::new(CtfState::new()));

    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let mut s = state.borrow_mut();
        if let Err(e) = s.open_file(&args[1]) { eprintln!("Error: {}", e); }
        else {
            window.set_filename(args[1].clone().into());
            window.set_max_z((s.nz.saturating_sub(1)) as i32);
            window.set_pixel_size(s.pixel_size as f32);
            update_window(&window, &s);
        }
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_open_file(move || {
        let file = FileDialog::new().add_filter("MRC", &["mrc","st","ali"]).pick_file();
        if let Some(p) = file {
            let ps = p.to_string_lossy().to_string();
            let mut s = st.borrow_mut();
            if let Err(e) = s.open_file(&ps) { eprintln!("Error: {}", e); return; }
            if let Some(w) = ww.upgrade() {
                w.set_filename(ps.into()); w.set_max_z((s.nz-1) as i32);
                w.set_pixel_size(s.pixel_size as f32);
                update_window(&w, &s);
            }
        }
      });
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_z_changed(move |z| {
        let mut s = st.borrow_mut();
        let _ = s.load_section(z as usize);
        if let Some(w) = ww.upgrade() { w.set_defocus(s.defocus as f32); update_window(&w, &s); }
      });
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_params_changed(move || {
        let mut s = st.borrow_mut();
        if let Some(w) = ww.upgrade() {
            s.voltage = w.get_voltage() as f64;
            s.cs = w.get_cs() as f64;
            s.defocus = w.get_defocus() as f64;
            s.pixel_size = w.get_pixel_size() as f64;
            s.amp_contrast = w.get_amplitude_contrast() as f64;
            let cz = s.current_z;
            let def = s.defocus;
            s.per_section_defocus[cz] = def;
            update_window(&w, &s);
        }
      });
    }

    { let st = state.clone(); let ww = window.as_weak();
      window.on_fit_defocus(move || {
        let mut s = st.borrow_mut();
        s.fit_defocus_auto();
        if let Some(w) = ww.upgrade() {
            w.set_defocus(s.defocus as f32);
            update_window(&w, &s);
        }
      });
    }

    { let st = state.clone();
      window.on_save_defocus(move || {
        let s = st.borrow();
        let path = if s.filepath.ends_with(".st") {
            s.filepath.replace(".st", ".defocus")
        } else { format!("{}.defocus", s.filepath) };
        let content: String = s.per_section_defocus.iter().enumerate()
            .map(|(i, &d)| format!("{}\t{:.4}", i, d))
            .collect::<Vec<_>>().join("\n");
        if let Err(e) = std::fs::write(&path, content) {
            eprintln!("Error saving: {}", e);
        } else {
            eprintln!("Saved defocus to {}", path);
        }
      });
    }

    window.run().unwrap();
}
