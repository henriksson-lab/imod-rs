use std::cell::RefCell;
use std::fmt;
use std::path::PathBuf;
use std::rc::Rc;

use rfd::FileDialog;
use slint::SharedString;

use imod_core::MrcMode;
use imod_fft::cross_correlate_2d;
use imod_math;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;
use imod_transforms::{read_tilt_file, write_xf_file, LinearTransform};

slint::include_modules!();

// ---------------------------------------------------------------------------
// Workflow steps
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
enum WorkflowStep {
    Setup = 0,
    PreProcessing = 1,
    CoarseAlignment = 2,
    BeadTracking = 3,
    FineAlignment = 4,
    Positioning = 5,
    FinalAlignment = 6,
    Reconstruction = 7,
    PostProcessing = 8,
}

impl WorkflowStep {
    fn from_index(i: i32) -> Option<Self> {
        match i {
            0 => Some(Self::Setup),
            1 => Some(Self::PreProcessing),
            2 => Some(Self::CoarseAlignment),
            3 => Some(Self::BeadTracking),
            4 => Some(Self::FineAlignment),
            5 => Some(Self::Positioning),
            6 => Some(Self::FinalAlignment),
            7 => Some(Self::Reconstruction),
            8 => Some(Self::PostProcessing),
            _ => None,
        }
    }
}

impl fmt::Display for WorkflowStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Setup => "Setup",
            Self::PreProcessing => "Pre-processing",
            Self::CoarseAlignment => "Coarse Alignment",
            Self::BeadTracking => "Bead Tracking",
            Self::FineAlignment => "Fine Alignment",
            Self::Positioning => "Positioning",
            Self::FinalAlignment => "Final Alignment",
            Self::Reconstruction => "Reconstruction",
            Self::PostProcessing => "Post-processing",
        };
        write!(f, "{}", name)
    }
}

// ---------------------------------------------------------------------------
// Dataset state
// ---------------------------------------------------------------------------

struct DatasetState {
    base: String,
    dir: PathBuf,
    dual_axis: bool,
}

impl DatasetState {
    fn new() -> Self {
        Self { base: String::new(), dir: PathBuf::from("."), dual_axis: false }
    }

    fn path(&self, suffix: &str) -> PathBuf {
        self.dir.join(format!("{}{}", self.base, suffix))
    }
}

// ---------------------------------------------------------------------------
// Step implementations — all use library calls, no process spawning
// ---------------------------------------------------------------------------

fn run_step(step: WorkflowStep, ds: &DatasetState) -> Result<String, String> {
    match step {
        WorkflowStep::Setup => step_setup(ds),
        WorkflowStep::PreProcessing => step_preprocessing(ds),
        WorkflowStep::CoarseAlignment => step_coarse_alignment(ds),
        WorkflowStep::BeadTracking => step_bead_tracking(ds),
        WorkflowStep::FineAlignment => step_fine_alignment(ds),
        WorkflowStep::Positioning => step_positioning(ds),
        WorkflowStep::FinalAlignment => step_final_alignment(ds),
        WorkflowStep::Reconstruction => step_reconstruction(ds),
        WorkflowStep::PostProcessing => step_postprocessing(ds),
    }
}

fn step_setup(ds: &DatasetState) -> Result<String, String> {
    let st_path = ds.path(".st");
    if !st_path.exists() {
        return Err(format!("Stack not found: {}", st_path.display()));
    }
    let reader = MrcReader::open(&st_path).map_err(|e| e.to_string())?;
    let h = reader.header();
    Ok(format!(
        "Dataset: {}\nStack: {} x {} x {}\nPixel: {:.4} A\nMode: {:?}",
        ds.base, h.nx, h.ny, h.nz, h.pixel_size_x(),
        h.data_mode().unwrap_or(MrcMode::Float)
    ))
}

fn step_preprocessing(ds: &DatasetState) -> Result<String, String> {
    let input = ds.path(".st");
    let output = ds.path("_fixed.st");
    let mut reader = MrcReader::open(&input).map_err(|e| e.to_string())?;
    let h = reader.header().clone();
    let (nx, ny, nz) = (h.nx as usize, h.ny as usize, h.nz as usize);

    let out_h = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    let mut writer = MrcWriter::create(&output, out_h).map_err(|e| e.to_string())?;
    let mut total = 0usize;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).map_err(|e| e.to_string())?;
        let mut slice = Slice::from_data(nx, ny, data);
        let (_, sd) = imod_math::mean_sd(&slice.data);
        let thresh = 6.0 * sd;

        for y in 2..ny - 2 {
            for x in 2..nx - 2 {
                let val = slice.get(x, y);
                let mut s = 0.0f32;
                let mut c = 0;
                for dy in -1i32..=1 { for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    s += slice.get((x as i32 + dx) as usize, (y as i32 + dy) as usize);
                    c += 1;
                }}
                let lm = s / c as f32;
                if (val - lm).abs() > thresh { slice.set(x, y, lm); total += 1; }
            }
        }
        writer.write_slice_f32(&slice.data).map_err(|e| e.to_string())?;
    }
    writer.finish(0.0, 0.0, 0.0).map_err(|e| e.to_string())?;
    Ok(format!("Replaced {} hot pixels in {} sections", total, nz))
}

fn step_coarse_alignment(ds: &DatasetState) -> Result<String, String> {
    let input = ds.path(".st");
    let xf_path = ds.path(".prexf");
    let preali = ds.path(".preali");

    let mut reader = MrcReader::open(&input).map_err(|e| e.to_string())?;
    let h = reader.header().clone();
    let (nx, ny, nz) = (h.nx as usize, h.ny as usize, h.nz as usize);
    let fft_nx = next_pow2(nx);
    let fft_ny = next_pow2(ny);

    let mut secs: Vec<Vec<f32>> = Vec::with_capacity(nz);
    for z in 0..nz { secs.push(reader.read_slice_f32(z).map_err(|e| e.to_string())?); }

    let ref_z = nz / 2;
    let mut xforms = vec![LinearTransform::identity(); nz];

    for z in (ref_z + 1)..nz {
        let (dx, dy) = cc_shift(&secs[z - 1], &secs[z], nx, ny, fft_nx, fft_ny);
        xforms[z] = LinearTransform::translation(xforms[z - 1].dx + dx, xforms[z - 1].dy + dy);
    }
    for z in (0..ref_z).rev() {
        let (dx, dy) = cc_shift(&secs[z + 1], &secs[z], nx, ny, fft_nx, fft_ny);
        xforms[z] = LinearTransform::translation(xforms[z + 1].dx + dx, xforms[z + 1].dy + dy);
    }

    write_xf_file(&xf_path, &xforms).map_err(|e| e.to_string())?;

    // Apply transforms
    let out_h = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    let mut writer = MrcWriter::create(&preali, out_h).map_err(|e| e.to_string())?;
    let (xcen, ycen) = (nx as f32 / 2.0, ny as f32 / 2.0);

    for z in 0..nz {
        let src = Slice::from_data(nx, ny, secs[z].clone());
        let inv = xforms[z].inverse();
        let mut dst = Slice::new(nx, ny, 0.0);
        for y in 0..ny { for x in 0..nx {
            let (sx, sy) = inv.apply(xcen, ycen, x as f32, y as f32);
            dst.set(x, y, src.interpolate_bilinear(sx, sy, 0.0));
        }}
        writer.write_slice_f32(&dst.data).map_err(|e| e.to_string())?;
    }
    writer.finish(0.0, 0.0, 0.0).map_err(|e| e.to_string())?;
    Ok(format!("Coarse alignment: {} sections aligned", nz))
}

fn step_bead_tracking(ds: &DatasetState) -> Result<String, String> {
    if !ds.path(".seed").exists() {
        return Err("Seed model not found. Create it with imod-viewer.".into());
    }
    Ok("Seed model found. Full tracking requires interactive seed selection.".into())
}

fn step_fine_alignment(ds: &DatasetState) -> Result<String, String> {
    if !ds.path(".fid").exists() {
        return Err("Fiducial model not found. Run bead tracking first.".into());
    }
    Ok("Fine alignment ready. Fiducial model found.".into())
}

fn step_positioning(_ds: &DatasetState) -> Result<String, String> {
    Ok("Positioning step: applies fine alignment transforms.".into())
}

fn step_final_alignment(ds: &DatasetState) -> Result<String, String> {
    if !ds.path(".xf").exists() {
        return Err("Transform file .xf not found.".into());
    }
    Ok("Final alignment: transforms ready.".into())
}

fn step_reconstruction(ds: &DatasetState) -> Result<String, String> {
    let ali = ds.path(".ali");
    let tlt = ds.path(".tlt");
    let rec = ds.path("_full.rec");

    if !ali.exists() { return Err(format!("{} not found", ali.display())); }
    if !tlt.exists() { return Err(format!("{} not found", tlt.display())); }

    let angles = read_tilt_file(&tlt).map_err(|e| e.to_string())?;
    let mut reader = MrcReader::open(&ali).map_err(|e| e.to_string())?;
    let h = reader.header().clone();
    let (in_nx, in_ny, nz) = (h.nx as usize, h.ny as usize, h.nz as usize);
    let out_nz = in_nx;

    let mut projs: Vec<Vec<f32>> = Vec::with_capacity(nz);
    for z in 0..nz { projs.push(reader.read_slice_f32(z).map_err(|e| e.to_string())?); }

    let cx = in_nx as f32 / 2.0;
    let cz = out_nz as f32 / 2.0;

    let mut out_h = MrcHeader::new(in_nx as i32, out_nz as i32, in_ny as i32, MrcMode::Float);
    out_h.add_label("imod-studio: reconstruction");
    let mut writer = MrcWriter::create(&rec, out_h).map_err(|e| e.to_string())?;

    for iy in 0..in_ny {
        let rows: Vec<&[f32]> = projs.iter().map(|p| &p[iy * in_nx..(iy + 1) * in_nx]).collect();
        let mut sl = vec![0.0f32; in_nx * out_nz];

        for (pi, &deg) in angles.iter().enumerate() {
            if pi >= nz { break; }
            let r = deg * std::f32::consts::PI / 180.0;
            let (cos_t, sin_t) = (r.cos(), r.sin());
            for oz in 0..out_nz {
                let bo = (oz as f32 - cz) * sin_t;
                for ox in 0..in_nx {
                    let px = (ox as f32 - cx) * cos_t + bo + cx;
                    let p0 = px.floor() as isize;
                    if p0 >= 0 && p0 + 1 < in_nx as isize {
                        let f = px - p0 as f32;
                        sl[oz * in_nx + ox] += rows[pi][p0 as usize] * (1.0 - f) + rows[pi][p0 as usize + 1] * f;
                    }
                }
            }
        }

        let inv = 1.0 / nz as f32;
        for v in &mut sl { *v *= inv; }
        writer.write_slice_f32(&sl).map_err(|e| e.to_string())?;
    }
    writer.finish(0.0, 0.0, 0.0).map_err(|e| e.to_string())?;
    Ok(format!("Reconstructed {} x {} x {} -> {}", in_nx, in_ny, nz, rec.display()))
}

fn step_postprocessing(ds: &DatasetState) -> Result<String, String> {
    if !ds.path("_full.rec").exists() {
        return Err("Reconstruction not found. Run reconstruction first.".into());
    }
    Ok("Post-processing: trimvol would be applied here.".into())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cc_shift(a: &[f32], b: &[f32], nx: usize, ny: usize, fnx: usize, fny: usize) -> (f32, f32) {
    let pa = pad(a, nx, ny, fnx, fny);
    let pb = pad(b, nx, ny, fnx, fny);
    let cc = cross_correlate_2d(&pa, &pb, fnx, fny);
    let (px, py) = peak(&cc, fnx, fny);
    let dx = if px > fnx / 2 { px as f32 - fnx as f32 } else { px as f32 };
    let dy = if py > fny / 2 { py as f32 - fny as f32 } else { py as f32 };
    (dx, dy)
}

fn pad(data: &[f32], nx: usize, ny: usize, fnx: usize, fny: usize) -> Vec<f32> {
    let s: f64 = data.iter().map(|&v| v as f64).sum();
    let m = (s / data.len() as f64) as f32;
    let mut p = vec![m; fnx * fny];
    let (ox, oy) = ((fnx - nx) / 2, (fny - ny) / 2);
    for y in 0..ny { for x in 0..nx { p[(y + oy) * fnx + (x + ox)] = data[y * nx + x]; } }
    p
}

fn peak(cc: &[f32], nx: usize, ny: usize) -> (usize, usize) {
    let mut mv = f32::NEG_INFINITY;
    let (mut mx, mut my) = (0, 0);
    for y in 0..ny { for x in 0..nx { if cc[y * nx + x] > mv { mv = cc[y * nx + x]; mx = x; my = y; } } }
    (mx, my)
}

fn next_pow2(n: usize) -> usize { let mut p = 1; while p < n { p <<= 1; } p }

fn append_log(w: &MainWindow, msg: &str) {
    let c = w.get_log_text().to_string();
    w.set_log_text(SharedString::from(if c.is_empty() { msg.to_string() } else { format!("{c}\n{msg}") }));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let window = MainWindow::new().unwrap();
    let state = Rc::new(RefCell::new(DatasetState::new()));

    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_open_dataset(move || {
            let file = FileDialog::new()
                .add_filter("Tilt series", &["st"])
                .add_filter("All files", &["*"])
                .pick_file();
            if let Some(path) = file {
                let mut s = state.borrow_mut();
                // Derive directory and base name from the chosen .st file
                if let Some(parent) = path.parent() {
                    s.dir = parent.to_path_buf();
                }
                let stem = path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "dataset".into());
                s.base = stem.clone();
                if let Some(w) = ww.upgrade() {
                    w.set_dataset_name(SharedString::from(&stem));
                    append_log(&w, &format!("Dataset: {} ({})", stem, path.display()));
                }
            }
        });
    }

    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_axis_changed(move |axis| {
            state.borrow_mut().dual_axis = axis == 1;
            if let Some(w) = ww.upgrade() {
                append_log(&w, &format!("Axis: {}", if axis == 1 { "dual" } else { "single" }));
            }
        });
    }

    {
        let state = state.clone();
        let ww = window.as_weak();
        window.on_run_step(move |idx| {
            let Some(step) = WorkflowStep::from_index(idx) else { return };
            let Some(w) = ww.upgrade() else { return };

            w.set_step_running(true);
            w.set_step_status(format!("Running {step}...").into());
            append_log(&w, &format!("--- {step} ---"));

            match run_step(step, &state.borrow()) {
                Ok(msg) => { append_log(&w, &msg); w.set_step_status(format!("{step}: done").into()); }
                Err(e) => { append_log(&w, &format!("[ERROR] {e}")); w.set_step_status(format!("{step}: error").into()); }
            }
            w.set_step_running(false);
        });
    }

    window.run().unwrap();
}
