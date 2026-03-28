use std::cell::RefCell;
use std::fmt;
use std::path::PathBuf;
use std::rc::Rc;

use rfd::FileDialog;
use slint::SharedString;

use imod_core::{MrcMode, Point3f};
use imod_fft::cross_correlate_2d;
use imod_math;
use imod_model::{read_model, write_model, ImodModel, ImodObject, ImodContour};
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;
use imod_transforms::{
    find_transform, read_tilt_file, read_xf_file, write_tilt_file, write_xf_file,
    LinearTransform, TransformMode,
};

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
    let seed_path = ds.path(".seed");
    if !seed_path.exists() {
        return Err("Seed model not found. Create it with imod-viewer.".into());
    }
    let preali_path = ds.path(".preali");
    if !preali_path.exists() {
        return Err("Pre-aligned stack not found. Run coarse alignment first.".into());
    }

    // Read seed model — each contour is one bead, each point marks (x, y, z=view)
    let seed = read_model(&seed_path).map_err(|e| e.to_string())?;
    if seed.objects.is_empty() {
        return Err("Seed model has no objects.".into());
    }

    // Read the pre-aligned stack
    let mut reader = MrcReader::open(&preali_path).map_err(|e| e.to_string())?;
    let h = reader.header().clone();
    let (nx, ny, nz) = (h.nx as usize, h.ny as usize, h.nz as usize);

    let mut slices: Vec<Vec<f32>> = Vec::with_capacity(nz);
    for z in 0..nz {
        slices.push(reader.read_slice_f32(z).map_err(|e| e.to_string())?);
    }

    // Template box half-size (pixels). Full box = 2*tbox+1.
    let tbox: usize = 16;
    let search_box: usize = 32; // search radius around predicted position
    let fft_size = next_pow2(2 * search_box + 2 * tbox + 1);

    // For each seed bead (contour), bidirectionally track across all views
    let mut tracked_obj = ImodObject::default();
    tracked_obj.name = "fiducials".into();
    tracked_obj.pdrawsize = 8;

    let mut total_views_tracked: usize = 0;
    let mut bead_count: usize = 0;

    for cont in &seed.objects[0].contours {
        if cont.points.is_empty() {
            continue;
        }
        // Use first point as the seed — x, y are pixel coords, z is the view index
        let seed_pt = &cont.points[0];
        let seed_view = seed_pt.z.round() as usize;
        if seed_view >= nz {
            continue;
        }

        let mut positions: Vec<Option<(f32, f32)>> = vec![None; nz];
        positions[seed_view] = Some((seed_pt.x, seed_pt.y));

        // Extract template from seed view
        let seed_slice = Slice::from_data(nx, ny, slices[seed_view].clone());
        let tx0 = (seed_pt.x as isize - tbox as isize).max(0) as usize;
        let ty0 = (seed_pt.y as isize - tbox as isize).max(0) as usize;
        let tw = ((2 * tbox + 1).min(nx - tx0)).min(2 * tbox + 1);
        let th = ((2 * tbox + 1).min(ny - ty0)).min(2 * tbox + 1);
        if tw < 4 || th < 4 {
            continue;
        }
        let template = seed_slice.subregion(tx0, ty0, tw, th);

        // Track forward from seed_view+1 to nz-1
        let mut prev_x = seed_pt.x;
        let mut prev_y = seed_pt.y;
        for z in (seed_view + 1)..nz {
            if let Some((nx2, ny2)) = track_bead_in_view(
                &template, &slices[z], nx, ny, prev_x, prev_y,
                search_box, tbox, fft_size,
            ) {
                positions[z] = Some((nx2, ny2));
                prev_x = nx2;
                prev_y = ny2;
            } else {
                break; // lost bead
            }
        }

        // Track backward from seed_view-1 to 0
        prev_x = seed_pt.x;
        prev_y = seed_pt.y;
        for z in (0..seed_view).rev() {
            if let Some((nx2, ny2)) = track_bead_in_view(
                &template, &slices[z], nx, ny, prev_x, prev_y,
                search_box, tbox, fft_size,
            ) {
                positions[z] = Some((nx2, ny2));
                prev_x = nx2;
                prev_y = ny2;
            } else {
                break;
            }
        }

        // Build tracked contour: one point per view where bead was found
        let mut tracked_cont = ImodContour::default();
        let mut views_this_bead = 0usize;
        for (z, pos) in positions.iter().enumerate() {
            if let Some((px, py)) = pos {
                tracked_cont.points.push(Point3f { x: *px, y: *py, z: z as f32 });
                views_this_bead += 1;
            }
        }

        if views_this_bead >= 3 {
            tracked_obj.contours.push(tracked_cont);
            total_views_tracked += views_this_bead;
            bead_count += 1;
        }
    }

    // Build output model
    let mut fid_model = ImodModel::default();
    fid_model.name = format!("{} fiducials", ds.base);
    fid_model.xmax = nx as i32;
    fid_model.ymax = ny as i32;
    fid_model.zmax = nz as i32;
    fid_model.objects.push(tracked_obj);

    let fid_path = ds.path(".fid");
    write_model(&fid_path, &fid_model).map_err(|e| e.to_string())?;

    let avg_views = if bead_count > 0 {
        total_views_tracked as f32 / bead_count as f32
    } else {
        0.0
    };
    Ok(format!(
        "Tracked {} beads, avg {:.1} views/bead -> {}",
        bead_count, avg_views, fid_path.display()
    ))
}

/// Track a single bead in one view by cross-correlating the template against a
/// search region extracted around the predicted position. Returns refined (x,y)
/// or None if the peak correlation is too weak.
fn track_bead_in_view(
    template: &Slice,
    view_data: &[f32],
    nx: usize,
    ny: usize,
    pred_x: f32,
    pred_y: f32,
    search_box: usize,
    tbox: usize,
    fft_size: usize,
) -> Option<(f32, f32)> {
    let half_search = (search_box + tbox) as isize;
    let sx0 = (pred_x as isize - half_search).max(0) as usize;
    let sy0 = (pred_y as isize - half_search).max(0) as usize;
    let sw = ((2 * half_search as usize + 1).min(nx - sx0)).min(fft_size);
    let sh = ((2 * half_search as usize + 1).min(ny - sy0)).min(fft_size);
    if sw < template.nx || sh < template.ny {
        return None;
    }

    // Extract search region
    let view_slice = Slice::from_data(nx, ny, view_data.to_vec());
    let search_region = view_slice.subregion(sx0, sy0, sw, sh);

    // Pad both to fft_size for cross-correlation
    let pa = pad(&search_region.data, sw, sh, fft_size, fft_size);
    let pb = pad(&template.data, template.nx, template.ny, fft_size, fft_size);
    let cc = cross_correlate_2d(&pa, &pb, fft_size, fft_size);

    // Find peak
    let (px, py) = peak(&cc, fft_size, fft_size);

    // Convert peak position back to image coordinates
    let dx = if px > fft_size / 2 { px as f32 - fft_size as f32 } else { px as f32 };
    let dy = if py > fft_size / 2 { py as f32 - fft_size as f32 } else { py as f32 };

    // Peak is the offset from the search region center to the template center
    let new_x = sx0 as f32 + sw as f32 / 2.0 + dx;
    let new_y = sy0 as f32 + sh as f32 / 2.0 + dy;

    // Reject if new position is too far from predicted
    let dist = ((new_x - pred_x).powi(2) + (new_y - pred_y).powi(2)).sqrt();
    if dist > search_box as f32 * 1.5 {
        return None;
    }

    // Check peak is reasonable (compare to CC mean)
    let cc_mean: f32 = cc.iter().sum::<f32>() / cc.len() as f32;
    let cc_peak = cc[py * fft_size + px];
    if cc_peak <= cc_mean {
        return None;
    }

    Some((new_x, new_y))
}

fn step_fine_alignment(ds: &DatasetState) -> Result<String, String> {
    let fid_path = ds.path(".fid");
    let tlt_path = ds.path(".tlt");
    if !fid_path.exists() {
        return Err("Fiducial model not found. Run bead tracking first.".into());
    }
    if !tlt_path.exists() {
        return Err("Tilt angle file not found.".into());
    }

    let fid_model = read_model(&fid_path).map_err(|e| e.to_string())?;
    let angles = read_tilt_file(&tlt_path).map_err(|e| e.to_string())?;
    let nz = angles.len();

    if fid_model.objects.is_empty() {
        return Err("Fiducial model has no objects.".into());
    }
    let obj = &fid_model.objects[0];

    // Extract bead tracks: for each contour, collect (view, x, y)
    let mut tracks: Vec<Vec<(usize, f32, f32)>> = Vec::new();
    for cont in &obj.contours {
        let mut track = Vec::new();
        for pt in &cont.points {
            let view = pt.z.round() as usize;
            if view < nz {
                track.push((view, pt.x, pt.y));
            }
        }
        if track.len() >= 3 {
            tracks.push(track);
        }
    }

    if tracks.is_empty() {
        return Err("No valid fiducial tracks found.".into());
    }

    // Image center (from the fiducial model dimensions)
    let xcen = fid_model.xmax as f32 / 2.0;
    let ycen = fid_model.ymax as f32 / 2.0;

    // Iterative alignment: estimate 3D bead positions, then solve per-view transforms
    let n_beads = tracks.len();
    let max_iters = 5;

    // Initialize per-view transforms to identity
    let mut xforms = vec![LinearTransform::identity(); nz];

    // Initialize 3D positions as mean of observations for each bead
    let mut bead_3d: Vec<(f64, f64, f64)> = Vec::with_capacity(n_beads);
    for track in &tracks {
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for &(_, x, y) in track {
            sx += (x - xcen) as f64;
            sy += (y - ycen) as f64;
        }
        let n = track.len() as f64;
        bead_3d.push((sx / n, sy / n, 0.0)); // initial Z=0
    }

    let mut rms_residual = f64::MAX;

    for _iteration in 0..max_iters {
        // Step 1: Given current 3D positions and tilt angles, project beads
        // and solve per-view transforms from projected->observed correspondences.
        for view in 0..nz {
            let angle_rad = (angles[view] as f64) * std::f64::consts::PI / 180.0;
            let cos_t = angle_rad.cos();
            let sin_t = angle_rad.sin();

            let mut src_pts: Vec<(f64, f64)> = Vec::new();
            let mut tgt_pts: Vec<(f64, f64)> = Vec::new();

            for (bi, track) in tracks.iter().enumerate() {
                // Find this bead's observation in this view
                if let Some(&(_, obs_x, obs_y)) = track.iter().find(|&&(v, _, _, )| v == view) {
                    // Project 3D position to this view (rotation about Y axis for tilt)
                    let (bx, by, bz) = bead_3d[bi];
                    let proj_x = bx * cos_t + bz * sin_t;
                    let proj_y = by;

                    src_pts.push((proj_x, proj_y));
                    tgt_pts.push(((obs_x - xcen) as f64, (obs_y - ycen) as f64));
                }
            }

            if src_pts.len() >= 3 {
                if let Some(result) = find_transform(&src_pts, &tgt_pts, TransformMode::RotationTranslationMag) {
                    xforms[view] = result.xf;
                }
            }
        }

        // Step 2: Given per-view transforms, re-estimate 3D positions.
        // For each bead, solve for (X, Y, Z) minimizing sum of squared residuals
        // across all views. Use a simple linear least-squares approach.
        let mut total_resid_sq = 0.0f64;
        let mut total_obs = 0usize;

        for (bi, track) in tracks.iter().enumerate() {
            // Build a small linear system: for each observation we have
            //   proj_x = X * cos(tilt) + Z * sin(tilt)
            //   proj_y = Y
            // After applying per-view transform to proj, it should match observed.
            // Simplify: use inverse of per-view xf on observed to get "corrected" projection.

            let mut sum_cy = 0.0f64;
            let mut sum_cx_cos = 0.0f64;
            let mut sum_cx_sin = 0.0f64;
            let mut sum_cos2 = 0.0f64;
            let mut sum_sin2 = 0.0f64;
            let mut sum_cos_sin = 0.0f64;
            let mut n_obs = 0usize;

            for &(view, obs_x, obs_y) in track {
                let angle_rad = (angles[view] as f64) * std::f64::consts::PI / 180.0;
                let cos_t = angle_rad.cos();
                let sin_t = angle_rad.sin();

                // Apply inverse of per-view transform to get corrected projection
                let inv = xforms[view].inverse();
                let (cx, cy) = inv.apply_raw((obs_x - xcen) as f32, (obs_y - ycen) as f32);
                let cx = cx as f64;
                let cy = cy as f64;

                sum_cy += cy;
                sum_cx_cos += cx * cos_t;
                sum_cx_sin += cx * sin_t;
                sum_cos2 += cos_t * cos_t;
                sum_sin2 += sin_t * sin_t;
                sum_cos_sin += cos_t * sin_t;
                n_obs += 1;
            }

            if n_obs < 2 {
                continue;
            }
            let nf = n_obs as f64;

            // Y is simply the mean of corrected Y values
            let new_y = sum_cy / nf;

            // Solve 2x2 system for X and Z:
            // [sum_cos2    sum_cos_sin] [X]   [sum_cx_cos]
            // [sum_cos_sin sum_sin2   ] [Z] = [sum_cx_sin]
            let det = sum_cos2 * sum_sin2 - sum_cos_sin * sum_cos_sin;
            if det.abs() > 1e-12 {
                let new_x = (sum_sin2 * sum_cx_cos - sum_cos_sin * sum_cx_sin) / det;
                let new_z = (sum_cos2 * sum_cx_sin - sum_cos_sin * sum_cx_cos) / det;
                bead_3d[bi] = (new_x, new_y, new_z);
            }

            // Compute residuals for this bead
            for &(view, obs_x, obs_y) in track {
                let angle_rad = (angles[view] as f64) * std::f64::consts::PI / 180.0;
                let cos_t = angle_rad.cos();
                let sin_t = angle_rad.sin();
                let (bx, by, bz) = bead_3d[bi];
                let proj_x = (bx * cos_t + bz * sin_t) as f32;
                let proj_y = by as f32;
                let (pred_x, pred_y) = xforms[view].apply_raw(proj_x, proj_y);
                let dx = (obs_x - xcen) - pred_x;
                let dy = (obs_y - ycen) - pred_y;
                total_resid_sq += (dx * dx + dy * dy) as f64;
                total_obs += 1;
            }
        }

        rms_residual = if total_obs > 0 {
            (total_resid_sq / total_obs as f64).sqrt()
        } else {
            0.0
        };
    }

    // Convert per-view transforms back to full-image coordinates (add center offset)
    let mut out_xforms = Vec::with_capacity(nz);
    for view in 0..nz {
        let xf = &xforms[view];
        // The transform was computed in centered coords; convert to IMOD .xf convention
        // where dx,dy represent shifts already relative to image center.
        out_xforms.push(LinearTransform {
            a11: xf.a11,
            a12: xf.a12,
            a21: xf.a21,
            a22: xf.a22,
            dx: xf.dx,
            dy: xf.dy,
        });
    }

    let xf_path = ds.path(".xf");
    write_xf_file(&xf_path, &out_xforms).map_err(|e| e.to_string())?;

    // Optionally write refined tilt angles (unchanged for now; a full solver
    // would refine these, but we keep the originals)
    let refined_tlt_path = ds.path(".tlt");
    write_tilt_file(&refined_tlt_path, &angles).map_err(|e| e.to_string())?;

    Ok(format!(
        "Fine alignment: {} beads, {} views, RMS residual {:.2} px -> {}",
        n_beads, nz, rms_residual, xf_path.display()
    ))
}

fn step_positioning(ds: &DatasetState) -> Result<String, String> {
    let preali_path = ds.path(".preali");
    let xf_path = ds.path(".xf");
    let ali_path = ds.path(".ali");

    if !preali_path.exists() {
        return Err(format!("{} not found. Run coarse alignment first.", preali_path.display()));
    }
    if !xf_path.exists() {
        return Err(format!("{} not found. Run fine alignment first.", xf_path.display()));
    }

    let xforms = read_xf_file(&xf_path).map_err(|e| e.to_string())?;
    let mut reader = MrcReader::open(&preali_path).map_err(|e| e.to_string())?;
    let h = reader.header().clone();
    let (nx, ny, nz) = (h.nx as usize, h.ny as usize, h.nz as usize);

    if xforms.len() != nz {
        return Err(format!(
            "Transform count ({}) does not match stack depth ({})",
            xforms.len(), nz
        ));
    }

    let out_h = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    let mut writer = MrcWriter::create(&ali_path, out_h).map_err(|e| e.to_string())?;
    let (xcen, ycen) = (nx as f32 / 2.0, ny as f32 / 2.0);

    for z in 0..nz {
        let data = reader.read_slice_f32(z).map_err(|e| e.to_string())?;
        let src = Slice::from_data(nx, ny, data);
        let inv = xforms[z].inverse();
        let mut dst = Slice::new(nx, ny, 0.0);
        for y in 0..ny {
            for x in 0..nx {
                let (sx, sy) = inv.apply(xcen, ycen, x as f32, y as f32);
                dst.set(x, y, src.interpolate_bilinear(sx, sy, 0.0));
            }
        }
        writer.write_slice_f32(&dst.data).map_err(|e| e.to_string())?;
    }
    writer.finish(0.0, 0.0, 0.0).map_err(|e| e.to_string())?;

    Ok(format!("Positioning: applied transforms to {} sections -> {}", nz, ali_path.display()))
}

fn step_final_alignment(ds: &DatasetState) -> Result<String, String> {
    let st_path = ds.path(".st");
    let xf_path = ds.path(".xf");
    let ali_path = ds.path(".ali");

    if !st_path.exists() {
        return Err(format!("{} not found.", st_path.display()));
    }
    if !xf_path.exists() {
        return Err("Transform file .xf not found. Run fine alignment first.".into());
    }

    let xforms = read_xf_file(&xf_path).map_err(|e| e.to_string())?;
    let mut reader = MrcReader::open(&st_path).map_err(|e| e.to_string())?;
    let h = reader.header().clone();
    let (nx, ny, nz) = (h.nx as usize, h.ny as usize, h.nz as usize);

    if xforms.len() != nz {
        return Err(format!(
            "Transform count ({}) does not match stack depth ({})",
            xforms.len(), nz
        ));
    }

    // Also compose with any existing .prexf (coarse alignment) if present
    let prexf_path = ds.path(".prexf");
    let prexf: Option<Vec<LinearTransform>> = if prexf_path.exists() {
        Some(read_xf_file(&prexf_path).map_err(|e| e.to_string())?)
    } else {
        None
    };

    let out_h = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    let mut writer = MrcWriter::create(&ali_path, out_h).map_err(|e| e.to_string())?;
    let (xcen, ycen) = (nx as f32 / 2.0, ny as f32 / 2.0);

    for z in 0..nz {
        let data = reader.read_slice_f32(z).map_err(|e| e.to_string())?;
        let src = Slice::from_data(nx, ny, data);

        // Compose coarse + fine transforms if both available
        let combined = if let Some(ref pre) = prexf {
            if z < pre.len() {
                pre[z].then(&xforms[z])
            } else {
                xforms[z]
            }
        } else {
            xforms[z]
        };

        let inv = combined.inverse();
        let mut dst = Slice::new(nx, ny, 0.0);
        for y in 0..ny {
            for x in 0..nx {
                let (sx, sy) = inv.apply(xcen, ycen, x as f32, y as f32);
                dst.set(x, y, src.interpolate_bilinear(sx, sy, 0.0));
            }
        }
        writer.write_slice_f32(&dst.data).map_err(|e| e.to_string())?;
    }
    writer.finish(0.0, 0.0, 0.0).map_err(|e| e.to_string())?;

    Ok(format!(
        "Final alignment: applied combined transforms to {} sections from original stack -> {}",
        nz, ali_path.display()
    ))
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
    let full_rec_path = ds.path("_full.rec");
    let rec_path = ds.path(".rec");

    if !full_rec_path.exists() {
        return Err("Reconstruction not found. Run reconstruction first.".into());
    }

    // Read the full reconstruction
    let mut reader = MrcReader::open(&full_rec_path).map_err(|e| e.to_string())?;
    let h = reader.header().clone();
    let (nx, ny, nz) = (h.nx as usize, h.ny as usize, h.nz as usize);
    let npix_total = nx * ny * nz;

    // First pass: compute global mean and SD across all slices
    let mut global_sum = 0.0f64;
    let mut global_sum_sq = 0.0f64;
    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;

    let mut all_slices: Vec<Vec<f32>> = Vec::with_capacity(nz);
    for z in 0..nz {
        let data = reader.read_slice_f32(z).map_err(|e| e.to_string())?;
        for &v in &data {
            let vd = v as f64;
            global_sum += vd;
            global_sum_sq += vd * vd;
            if v < global_min { global_min = v; }
            if v > global_max { global_max = v; }
        }
        all_slices.push(data);
    }

    let mean = global_sum / npix_total as f64;
    let variance = (global_sum_sq / npix_total as f64) - mean * mean;
    let sd = if variance > 0.0 { variance.sqrt() } else { 1.0 };

    // Scale to byte range [0, 255] using mean +/- 3*SD -> [0, 255]
    let low = mean - 3.0 * sd;
    let high = mean + 3.0 * sd;
    let range = high - low;
    let scale_factor = if range > 1e-10 { 255.0 / range } else { 1.0 };

    // Write output as byte mode
    let out_h = MrcHeader::new(nx as i32, ny as i32, nz as i32, MrcMode::Byte);
    let mut writer = MrcWriter::create(&rec_path, out_h).map_err(|e| e.to_string())?;

    let mut out_min = 255.0f32;
    let mut out_max = 0.0f32;
    let mut out_sum = 0.0f64;

    for z in 0..nz {
        let mut scaled: Vec<f32> = Vec::with_capacity(nx * ny);
        for &v in &all_slices[z] {
            let sv = ((v as f64 - low) * scale_factor).clamp(0.0, 255.0) as f32;
            if sv < out_min { out_min = sv; }
            if sv > out_max { out_max = sv; }
            out_sum += sv as f64;
            scaled.push(sv);
        }
        writer.write_slice_f32(&scaled).map_err(|e| e.to_string())?;
    }

    let out_mean = (out_sum / npix_total as f64) as f32;
    writer.finish(out_min, out_max, out_mean).map_err(|e| e.to_string())?;

    Ok(format!(
        "Post-processing: {} x {} x {} -> {}\n\
         Input: mean={:.1}, SD={:.1}, range=[{:.1}, {:.1}]\n\
         Output: byte mode, range=[{:.0}, {:.0}], mean={:.1}",
        nx, ny, nz, rec_path.display(),
        mean, sd, global_min, global_max,
        out_min, out_max, out_mean
    ))
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
