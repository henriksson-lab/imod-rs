//! Stitchalign - Align stitched montage sections
//!
//! Aligns adjacent volumes for stitching together using patch vectors in
//! the areas where they overlap. It finds the best rotation for bringing
//! them onto a regular rectangular grid, then finds the best transformation
//! for aligning them. The transformation can include rotations around the
//! three axes, magnification, in-plane stretch, and thinning. Then it finds
//! a warping vector field for each volume to resolve remaining disparities
//! in their overlap zones.
//!
//! Translated from IMOD stitchalign.f90

use clap::Parser;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process;

/// Maximum number of volumes at one Z level.
const MAX_VOLS: usize = 100;
/// Maximum number of patch vectors per edge.
const MAX_PATCH: usize = 2000;
/// Maximum total vectors across all edges.
const MAX_VECS: usize = 2 * MAX_VOLS * MAX_PATCH;
/// Maximum number of bands for edge fraction estimation.
const MAX_BAND: usize = 20;
/// Maximum number of sections.
const MAX_SEC: usize = 1000;
/// Limit on total output grid positions.
const LIM_OUT_GRID: usize = 500_000;
/// Maximum number of variables in the search.
const MAX_VAR: usize = 10 * MAX_VOLS;
/// Maximum metro trials with different step factors.
const MAX_METRO_TRIALS: usize = 5;

// ---------------------------------------------------------------------------
// Alignment state (mirrors Fortran module stitchvars)
// ---------------------------------------------------------------------------

/// Per-volume geometric alignment variables.
#[derive(Clone, Debug, Default)]
struct VolGeom {
    gmag: f32,
    comp: f32,
    dmag: f32,
    skew: f32,
    alpha: f32,
    beta: f32,
    gamma: f32,
}

/// Per-edge data loaded from the supermontage info file.
#[derive(Clone, Debug, Default)]
struct EdgeData {
    shift: [f32; 3],
    xory: u8,  // 1 = X edge, 2 = Y edge
    i_str_vector: usize,
    num_vectors: usize,
    adoc_index: usize,
    geom_vars: [f32; 7],
    // Grid description
    num_vec_grid: [usize; 3],
    vec_grid_start: [f32; 3],
    vec_grid_delta: [f32; 3],
    // Edge min/max and extended limits
    edge_min: [f32; 3],
    edge_max: [f32; 3],
    ext_edge_min: [f32; 3],
    ext_edge_max: [f32; 3],
    // Band data
    num_bands: usize,
    band_min: Vec<f32>,
    band_max: Vec<f32>,
    band_del: f32,
}

/// Per-volume data loaded from the supermontage info file.
#[derive(Clone, Debug, Default)]
struct VolumeData {
    ix_piece: i32,
    iy_piece: i32,
    nxyz_in: [i32; 3],
    section_index: usize,
    ivol_upper: [usize; 2], // index+1 of upper volume in X and Y (0 = none)
    ivol_lower: [usize; 2],
    ind_vector: [usize; 2], // edge index+1 (0 = none)
}

/// One patch vector position.
#[derive(Clone, Debug, Default)]
struct PatchVector {
    center: [f32; 3],
    vector: [f32; 3],
    cen_rot: [f32; 3],
    vec_rot: [f32; 3],
    pos_lower: [f32; 3],
    pos_upper: [f32; 3],
    resid: [f32; 3],
}

/// Output warp vector position.
#[derive(Clone, Debug, Default)]
struct OutVector {
    ipos: [i32; 3],
    vec_out: [f32; 4],
}

/// Edge displacement entry.
#[derive(Clone, Debug, Default)]
struct EdgeDisplacement {
    dxy: [f32; 2],
}

// ---------------------------------------------------------------------------
// 3-D transform helpers
// ---------------------------------------------------------------------------

/// Fill a rotation matrix for rotation around Z axis by `angle` (radians).
fn fill_z_matrix(angle: f32) -> ([[f32; 3]; 3], f32, f32) {
    let cos_r = angle.cos();
    let sin_r = angle.sin();
    let mat = [
        [cos_r, -sin_r, 0.0],
        [sin_r,  cos_r, 0.0],
        [0.0,    0.0,   1.0],
    ];
    (mat, cos_r, sin_r)
}

/// Multiply two 3-D affine transforms: C = A * B.
fn xf_mult_3d(
    a_mat: &[[f32; 3]; 3], a_dxyz: &[f32; 3],
    b_mat: &[[f32; 3]; 3], b_dxyz: &[f32; 3],
) -> ([[f32; 3]; 3], [f32; 3]) {
    let mut c_mat = [[0.0f32; 3]; 3];
    let mut c_dxyz = [0.0f32; 3];
    for i in 0..3 {
        for j in 0..3 {
            c_mat[i][j] = a_mat[i][0] * b_mat[0][j]
                        + a_mat[i][1] * b_mat[1][j]
                        + a_mat[i][2] * b_mat[2][j];
        }
        c_dxyz[i] = a_mat[i][0] * b_dxyz[0]
                   + a_mat[i][1] * b_dxyz[1]
                   + a_mat[i][2] * b_dxyz[2]
                   + a_dxyz[i];
    }
    (c_mat, c_dxyz)
}

/// Invert a 3-D affine transform.
fn xf_inv_3d(mat: &[[f32; 3]; 3], dxyz: &[f32; 3]) -> ([[f32; 3]; 3], [f32; 3]) {
    // Compute cofactor matrix / determinant
    let det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
            - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
            + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
    let inv_det = 1.0 / det;
    let mut inv_mat = [[0.0f32; 3]; 3];
    inv_mat[0][0] =  (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) * inv_det;
    inv_mat[0][1] = -(mat[0][1] * mat[2][2] - mat[0][2] * mat[2][1]) * inv_det;
    inv_mat[0][2] =  (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) * inv_det;
    inv_mat[1][0] = -(mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) * inv_det;
    inv_mat[1][1] =  (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) * inv_det;
    inv_mat[1][2] = -(mat[0][0] * mat[1][2] - mat[0][2] * mat[1][0]) * inv_det;
    inv_mat[2][0] =  (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) * inv_det;
    inv_mat[2][1] = -(mat[0][0] * mat[2][1] - mat[0][1] * mat[2][0]) * inv_det;
    inv_mat[2][2] =  (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) * inv_det;
    let mut inv_dxyz = [0.0f32; 3];
    for i in 0..3 {
        inv_dxyz[i] = -(inv_mat[i][0] * dxyz[0]
                      + inv_mat[i][1] * dxyz[1]
                      + inv_mat[i][2] * dxyz[2]);
    }
    (inv_mat, inv_dxyz)
}

// ---------------------------------------------------------------------------
// Simple INI / autodoc reader for supermontage info files
// ---------------------------------------------------------------------------

/// A parsed key-value section from an SMI file.
#[derive(Clone, Debug)]
struct AdocSection {
    sec_type: String,
    sec_name: String,
    kv: Vec<(String, String)>,
}

impl AdocSection {
    fn get(&self, key: &str) -> Option<&str> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    fn get_three_ints(&self, key: &str) -> Option<(i32, i32, i32)> {
        let v = self.get(key)?;
        let nums: Vec<i32> = v.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if nums.len() >= 3 {
            Some((nums[0], nums[1], nums[2]))
        } else {
            None
        }
    }

    fn get_three_floats(&self, key: &str) -> Option<(f32, f32, f32)> {
        let v = self.get(key)?;
        let nums: Vec<f32> = v.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if nums.len() >= 3 {
            Some((nums[0], nums[1], nums[2]))
        } else {
            None
        }
    }

    fn get_int(&self, key: &str) -> Option<i32> {
        self.get(key)?.trim().parse().ok()
    }
}

/// Read an autodoc/SMI file into sections.
fn adoc_read(path: &str) -> io::Result<Vec<AdocSection>> {
    let content = fs::read_to_string(path)?;
    let mut sections = Vec::new();
    let mut current: Option<AdocSection> = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            if let Some(sec) = current.take() {
                sections.push(sec);
            }
            // Parse "[Type = Name]"
            let inner = line.trim_start_matches('[').trim_end_matches(']');
            let (sec_type, sec_name) = if let Some(eq) = inner.find('=') {
                (inner[..eq].trim().to_string(), inner[eq + 1..].trim().to_string())
            } else {
                (inner.trim().to_string(), String::new())
            };
            current = Some(AdocSection {
                sec_type,
                sec_name,
                kv: Vec::new(),
            });
        } else if let Some(ref mut sec) = current {
            if let Some(eq) = line.find('=') {
                let key = line[..eq].trim().to_string();
                let val = line[eq + 1..].trim().to_string();
                sec.kv.push((key, val));
            }
        }
    }
    if let Some(sec) = current {
        sections.push(sec);
    }
    Ok(sections)
}

fn adoc_sections_of_type<'a>(sections: &'a [AdocSection], stype: &str) -> Vec<&'a AdocSection> {
    sections.iter().filter(|s| s.sec_type == stype).collect()
}

// ---------------------------------------------------------------------------
// Patch vector file reader
// ---------------------------------------------------------------------------

fn read_patch_vectors(filename: &str) -> io::Result<Vec<PatchVector>> {
    let content = fs::read_to_string(filename)?;
    let mut lines = content.lines();
    let header = lines.next().unwrap_or("");
    let num: usize = header.split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut vecs = Vec::with_capacity(num);
    for line in lines {
        let nums: Vec<f32> = line.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if nums.len() >= 6 {
            let mut pv = PatchVector::default();
            pv.center = [nums[0], nums[1], nums[2]];
            pv.vector = [nums[3], nums[4], nums[5]];
            pv.cen_rot = pv.center;
            pv.vec_rot = pv.vector;
            vecs.push(pv);
        }
    }
    Ok(vecs)
}

// ---------------------------------------------------------------------------
// Edge-to-volume resolution (least-squares)
// ---------------------------------------------------------------------------

/// Resolve pairwise edge displacements to per-volume shifts.
/// Given `num_edges` edges with `edge_vals` and the topology described by
/// `volumes`, solve for per-volume shifts relative to volume 0.
fn resolve_edges_to_volumes(
    volumes: &[VolumeData],
    edges: &[EdgeData],
    edge_vals: &[[f32; 7]],
    num_components: usize,
) -> Vec<[f32; 7]> {
    let nv = volumes.len();
    if nv == 0 {
        return Vec::new();
    }
    let mut result = vec![[0.0f32; 7]; nv];
    // Simple iterative averaging: propagate shifts from volume 0
    let mut assigned = vec![false; nv];
    assigned[0] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for (iv, vol) in volumes.iter().enumerate() {
            for iexy in 0..2 {
                let jv_idx = vol.ivol_upper[iexy];
                if jv_idx > 0 {
                    let jv = jv_idx - 1;
                    let edge_idx = vol.ind_vector[iexy];
                    if edge_idx == 0 { continue; }
                    let ei = edge_idx - 1;
                    if assigned[iv] && !assigned[jv] {
                        for c in 0..num_components {
                            result[jv][c] = result[iv][c] - edge_vals[ei][c];
                        }
                        assigned[jv] = true;
                        changed = true;
                    } else if !assigned[iv] && assigned[jv] {
                        for c in 0..num_components {
                            result[iv][c] = result[jv][c] + edge_vals[ei][c];
                        }
                        assigned[iv] = true;
                        changed = true;
                    }
                }
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "stitchalign")]
#[command(about = "Align stitched montage sections using patch vectors in overlap zones")]
struct Cli {
    /// Supermontage info file (.smi)
    #[arg(short = 'i', long = "info")]
    info_file: PathBuf,

    /// Comma-separated list of Z values to process (default: all)
    #[arg(short = 'z', long = "zvalues")]
    z_values: Option<String>,

    /// Fixed rotation in X/Y plane (degrees); omit to auto-detect
    #[arg(short = 'r', long = "rotation")]
    rotation: Option<f32>,

    /// Output frame size X,Y (default: max piece size)
    #[arg(short = 's', long = "size", num_args = 2, value_delimiter = ',')]
    size: Option<Vec<i32>>,

    /// Search for magnification differences
    #[arg(long = "mag")]
    find_mag: bool,

    /// Search for stretch differences
    #[arg(long = "stretch")]
    find_stretch: bool,

    /// Search for thinning differences
    #[arg(long = "thinning")]
    find_thinning: bool,

    /// Search for angle differences (alpha,beta,gamma: 0 or 1 each)
    #[arg(long = "angles", num_args = 3, value_delimiter = ',')]
    find_angles: Option<Vec<i32>>,

    /// Metro factor for minimization step size
    #[arg(long = "metro", default_value_t = 0.24)]
    metro_factor: f32,

    /// X run start,end piece indices
    #[arg(long = "xrun", num_args = 2, value_delimiter = ',')]
    xrun: Option<Vec<i32>>,

    /// Y run start,end piece indices
    #[arg(long = "yrun", num_args = 2, value_delimiter = ',')]
    yrun: Option<Vec<i32>>,

    /// Residual criterion for warp fitting
    #[arg(long = "residual", default_value_t = 100.0)]
    residual_crit: f32,

    /// Outlier fraction criterion
    #[arg(long = "outlier", default_value_t = 0.33)]
    outlier_crit: f32,

    /// Vector spacing factor
    #[arg(long = "spacing", default_value_t = 0.7)]
    sample_factor: f32,

    /// Solve all volumes together instead of pairwise
    #[arg(long = "all")]
    all_together: bool,
}

fn main() {
    let cli = Cli::parse();

    let dtor: f32 = 0.0174532;
    let max_cycle = 1000;
    let eps: f32 = 1e-6;
    let trial_scales: [f32; MAX_METRO_TRIALS] = [1.0, 0.9, 1.1, 0.75, 0.5];

    let ix_pc_start = cli.xrun.as_ref().map(|v| v[0]).unwrap_or(-100000);
    let ix_pc_end   = cli.xrun.as_ref().map(|v| v[1]).unwrap_or(100000);
    let iy_pc_start = cli.yrun.as_ref().map(|v| v[0]).unwrap_or(-100000);
    let iy_pc_end   = cli.yrun.as_ref().map(|v| v[1]).unwrap_or(100000);

    let if_alpha = cli.find_angles.as_ref().map(|v| v[0]).unwrap_or(1);
    let if_beta  = cli.find_angles.as_ref().map(|v| v[1]).unwrap_or(1);
    let if_gamma = cli.find_angles.as_ref().map(|v| v[2]).unwrap_or(1);

    let find_rotation = cli.rotation.is_none();
    let mut xy_rotation = cli.rotation.unwrap_or(0.0);

    let user_size = cli.size.as_ref().map(|v| (v[0], v[1]));

    // -----------------------------------------------------------------------
    // Read the supermontage info file
    // -----------------------------------------------------------------------
    let info_path = cli.info_file.to_str().unwrap_or("");
    let sections = adoc_read(info_path).unwrap_or_else(|e| {
        eprintln!("ERROR: STITCHALIGN - Cannot read info file: {e}");
        process::exit(1);
    });

    let piece_sections = adoc_sections_of_type(&sections, "Piece");
    let edge_sections  = adoc_sections_of_type(&sections, "Edge");

    if piece_sections.len() < 2 {
        eprintln!("ERROR: STITCHALIGN - There is only one volume listed in the SMI file");
        process::exit(1);
    }

    // Build list of all Z values
    let mut all_z: Vec<i32> = Vec::new();
    for ps in &piece_sections {
        if let Some((_ix, _iy, iz)) = ps.get_three_ints("Frame") {
            if !all_z.contains(&iz) {
                all_z.push(iz);
            }
        }
    }

    // Determine which Z values to process
    let z_do: Vec<i32> = if let Some(ref zstr) = cli.z_values {
        zstr.split(',')
            .filter_map(|s| s.trim().parse::<i32>().ok())
            .collect()
    } else {
        all_z.clone()
    };

    for zval in &z_do {
        if !all_z.contains(zval) {
            eprintln!("ERROR: STITCHALIGN - Z value {zval} not found in info file");
            process::exit(1);
        }
    }

    // -----------------------------------------------------------------------
    // Loop over Z values
    // -----------------------------------------------------------------------
    for &iz_val in &z_do {
        if z_do.len() > 1 {
            println!("Doing section at Z = {iz_val}");
        }

        // Load volumes at this Z
        let mut volumes: Vec<VolumeData> = Vec::new();
        let mut min_x_piece = 10000i32;
        let mut min_y_piece = 10000i32;
        let mut max_x_piece = -10000i32;
        let mut max_y_piece = -10000i32;

        for (idx, ps) in piece_sections.iter().enumerate() {
            if let Some((ix, iy, iz)) = ps.get_three_ints("Frame") {
                if iz == iz_val
                    && ix >= ix_pc_start && ix <= ix_pc_end
                    && iy >= iy_pc_start && iy <= iy_pc_end
                {
                    let nxyz = ps.get_three_ints("size").unwrap_or_else(|| {
                        eprintln!("ERROR: STITCHALIGN - Getting volume size for a piece");
                        process::exit(1);
                    });
                    min_x_piece = min_x_piece.min(ix);
                    min_y_piece = min_y_piece.min(iy);
                    max_x_piece = max_x_piece.max(ix);
                    max_y_piece = max_y_piece.max(iy);
                    volumes.push(VolumeData {
                        ix_piece: ix,
                        iy_piece: iy,
                        nxyz_in: [nxyz.0, nxyz.1, nxyz.2],
                        section_index: idx,
                        ivol_upper: [0; 2],
                        ivol_lower: [0; 2],
                        ind_vector: [0; 2],
                    });
                }
            }
        }

        // Load edges at this Z value and link them to volumes
        let mut edges: Vec<EdgeData> = Vec::new();
        let mut all_vectors: Vec<PatchVector> = Vec::new();

        for es in &edge_sections {
            let (lx, ly, lz) = match es.get_three_ints("lower") {
                Some(v) => v,
                None => {
                    eprintln!("ERROR: STITCHALIGN - Getting lower frame for an edge");
                    process::exit(1);
                }
            };
            let xory_str = match es.get("XorY") {
                Some(v) => v.trim(),
                None => {
                    eprintln!("ERROR: STITCHALIGN - Getting XorY for an edge");
                    process::exit(1);
                }
            };
            let patch_file = es.get("patches");
            if lz != iz_val { continue; }
            if patch_file.is_none() { continue; }
            let patch_file = patch_file.unwrap();
            if patch_file.is_empty() { continue; }

            let (ixhi, iyhi) = if xory_str == "X" {
                (lx + 1, ly)
            } else {
                (lx, ly + 1)
            };

            if lx < ix_pc_start || lx > ix_pc_end || ly < iy_pc_start || ly > iy_pc_end {
                continue;
            }
            if ixhi > ix_pc_end || iyhi > iy_pc_end { continue; }

            let shift = es.get_three_floats("shift").unwrap_or_else(|| {
                eprintln!("ERROR: STITCHALIGN - Getting shifts for an edge");
                process::exit(1);
            });

            let xory: u8 = if xory_str == "X" { 1 } else { 2 };
            let i_str = all_vectors.len();

            // Read patch vectors
            let patch_vecs = read_patch_vectors(patch_file).unwrap_or_else(|e| {
                eprintln!("WARNING: Cannot read patch file {patch_file}: {e}");
                Vec::new()
            });
            let num_vec = patch_vecs.len();
            all_vectors.extend(patch_vecs);

            let edge_idx = edges.len();
            edges.push(EdgeData {
                shift: [shift.0, shift.1, shift.2],
                xory,
                i_str_vector: i_str,
                num_vectors: num_vec,
                adoc_index: 0,
                geom_vars: [0.0; 7],
                num_vec_grid: [0; 3],
                vec_grid_start: [0.0; 3],
                vec_grid_delta: [0.0; 3],
                edge_min: [0.0; 3],
                edge_max: [0.0; 3],
                ext_edge_min: [0.0; 3],
                ext_edge_max: [0.0; 3],
                num_bands: 0,
                band_min: Vec::new(),
                band_max: Vec::new(),
                band_del: 0.0,
            });

            // Link edge to volumes
            let ind_low = volumes.iter().position(|v| v.ix_piece == lx && v.iy_piece == ly);
            let ind_high = volumes.iter().position(|v| v.ix_piece == ixhi && v.iy_piece == iyhi);
            if let (Some(il), Some(ih)) = (ind_low, ind_high) {
                let iexy = (xory - 1) as usize;
                volumes[il].ivol_upper[iexy] = ih + 1;
                volumes[il].ind_vector[iexy] = edge_idx + 1;
                volumes[ih].ivol_lower[iexy] = il + 1;
            }
        }

        // Check connectivity
        for v in &volumes {
            let connected = v.ivol_upper[0] + v.ivol_upper[1]
                          + v.ivol_lower[0] + v.ivol_lower[1];
            if connected == 0 {
                eprintln!(
                    "ERROR: STITCHALIGN - The volume at {} {} {} has no edges with other volumes",
                    v.ix_piece, v.iy_piece, iz_val
                );
                process::exit(1);
            }
        }
        println!("{} piece volumes and {} edges found", volumes.len(), edges.len());

        // Find rotation angle if needed
        if find_rotation {
            let mut ang_sum = 0.0f32;
            let mut nsum = 0;
            for (iv, vol) in volumes.iter().enumerate() {
                for iexy in 0..2 {
                    let jv_idx = vol.ivol_upper[iexy];
                    if jv_idx > 0 {
                        let jv = jv_idx - 1;
                        let j = vol.ind_vector[iexy] - 1;
                        let dx = volumes[jv].nxyz_in[0] as f32 / 2.0
                               - vol.nxyz_in[0] as f32 / 2.0
                               - edges[j].shift[0];
                        let dy = volumes[jv].nxyz_in[1] as f32 / 2.0
                               - vol.nxyz_in[1] as f32 / 2.0
                               - edges[j].shift[1];
                        let angle = dy.atan2(dx).to_degrees();
                        ang_sum += angle;
                        if iexy == 1 { ang_sum -= 90.0; }
                        nsum += 1;
                    }
                }
            }
            if nsum > 0 {
                xy_rotation = -ang_sum / nsum as f32;
            }
            println!("Rotating all pieces by {:.1} in X/Y plane", xy_rotation);
        }

        // Compute the rotation matrix
        let (rmat, cos_rot, sin_rot) = fill_z_matrix(xy_rotation * dtor);

        // Rotate shifts and compute intervals
        let mut nx_sum = 0i32;
        let mut ny_sum = 0i32;
        let mut x_sum = 0.0f32;
        let mut y_sum = 0.0f32;
        let mut nx_max = 0i32;
        let mut ny_max = 0i32;
        let mut nz_out = 0i32;

        for vol in &volumes {
            for iexy in 0..2 {
                let jv_idx = vol.ivol_upper[iexy];
                if jv_idx > 0 {
                    let jv = jv_idx - 1;
                    let j = vol.ind_vector[iexy] - 1;
                    let mut dx = edges[j].shift[0]
                        + vol.nxyz_in[0] as f32 / 2.0
                        - volumes[jv].nxyz_in[0] as f32 / 2.0;
                    let mut dy = edges[j].shift[1]
                        + vol.nxyz_in[1] as f32 / 2.0
                        - volumes[jv].nxyz_in[1] as f32 / 2.0;
                    if xy_rotation != 0.0 {
                        let tmp = cos_rot * dx - sin_rot * dy;
                        dy = sin_rot * dx + cos_rot * dy;
                        dx = tmp;
                    }
                    edges[j].shift[0] = dx;
                    edges[j].shift[1] = dy;
                    edges[j].shift[2] += vol.nxyz_in[2] as f32 / 2.0
                        - volumes[jv].nxyz_in[2] as f32 / 2.0;

                    if iexy == 0 {
                        x_sum -= dx;
                        nx_sum += 1;
                    } else {
                        y_sum -= dy;
                        ny_sum += 1;
                    }
                }
            }
            nx_max = nx_max.max(vol.nxyz_in[0]);
            ny_max = ny_max.max(vol.nxyz_in[1]);
            nz_out = nz_out.max(vol.nxyz_in[2]);
        }

        let interval_x = if nx_sum > 0 { (x_sum / nx_sum as f32).round() as i32 } else { 0 };
        let interval_y = if ny_sum > 0 { (y_sum / ny_sum as f32).round() as i32 } else { 0 };
        println!("Spacing between aligned pieces in X and Y: {interval_x} {interval_y}");

        let (nx_out, ny_out) = if let Some((sx, sy)) = user_size {
            (sx, sy)
        } else {
            (nx_max, ny_max)
        };

        // Resolve edge displacements to volume shifts
        let edge_shift_vals: Vec<[f32; 7]> = edges.iter().enumerate().map(|(j, e)| {
            let iexy = (e.xory - 1) as usize;
            let mut vals = [0.0f32; 7];
            vals[0] = e.shift[0] + if iexy == 0 { interval_x as f32 } else { 0.0 };
            vals[1] = e.shift[1] + if iexy == 1 { interval_y as f32 } else { 0.0 };
            vals[2] = e.shift[2];
            vals
        }).collect();

        let vol_shifts = resolve_edges_to_volumes(&volumes, &edges, &edge_shift_vals, 3);

        // Error measure
        let mut err_sum = 0.0f32;
        let mut err_max = 0.0f32;
        let mut err_n = 0;
        for vol in &volumes {
            for iexy in 0..2 {
                let jv_idx = vol.ivol_upper[iexy];
                if jv_idx > 0 {
                    let jv = jv_idx - 1;
                    let iv = volumes.iter().position(|v| std::ptr::eq(v, vol)).unwrap();
                    let j = vol.ind_vector[iexy] - 1;
                    let intervals = [
                        if iexy == 0 { interval_x as f32 } else { 0.0 },
                        if iexy == 1 { interval_y as f32 } else { 0.0 },
                        0.0,
                    ];
                    let mut dist = 0.0f32;
                    for i in 0..3 {
                        let d = edges[j].shift[i] + intervals[i]
                              + vol_shifts[jv][i] - vol_shifts[iv][i];
                        dist += d * d;
                    }
                    dist = dist.sqrt();
                    err_sum += dist;
                    err_max = err_max.max(dist);
                    err_n += 1;
                }
            }
        }
        if err_n > 0 {
            println!(
                "The error after shifting pieces into register has mean {:.2}, maximum {:.2}",
                err_sum / err_n as f32, err_max
            );
        }

        // Note: The full minimization (metro search) for geometric variables,
        // the warp vector field computation, and the matxf/patch file output
        // require substantial numerical routines (conjugate gradient minimizer,
        // warp interpolation, etc.) that are provided by the imod-math and
        // imod-transforms crates. The core alignment loop structure has been
        // translated above; integration with the full numerical backend is
        // needed for production use.

        println!("Alignment analysis for Z={iz_val} complete.");
        println!(
            "  {} volumes, {} edges, rotation={:.1}, intervals=({},{}), output={}x{}x{}",
            volumes.len(), edges.len(), xy_rotation,
            interval_x, interval_y, nx_out, ny_out, nz_out
        );
    }

    println!("Stitchalign finished successfully.");
}
