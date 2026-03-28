//! Stitchvars - Stitching variables computation
//!
//! Computes and manages the shared variables for volume stitching alignment.
//! This is the Rust translation of the stitchvars module, exposed as a CLI
//! tool for inspecting and computing stitching parameters from a supermontage
//! info file.
//!
//! Translated from IMOD stitchvars.f90

use clap::Parser;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process;

/// Maximum number of volumes.
const MAX_VOLS: usize = 100;
/// Maximum number of patch vectors per edge.
const MAX_PATCH: usize = 2000;
/// Maximum total vectors.
const MAX_VECS: usize = 2 * MAX_VOLS * MAX_PATCH;
/// Maximum number of bands for edge fraction estimation.
const MAX_BAND: usize = 20;

/// Alignment variables for all volumes.
#[derive(Clone, Debug)]
struct StitchVars {
    /// Translation offsets per volume [vol][xyz].
    dxyz: Vec<[f32; 3]>,
    /// Global magnification per volume.
    gmag: Vec<f32>,
    /// Compression factor per volume.
    comp: Vec<f32>,
    /// Differential magnification per volume.
    dmag: Vec<f32>,
    /// Skew angle per volume.
    skew: Vec<f32>,
    /// Rotation angles per volume.
    alpha: Vec<f32>,
    beta: Vec<f32>,
    gamma: Vec<f32>,

    /// Corresponding positions in lower and upper volumes.
    pos_lower: Vec<[f32; 3]>,
    pos_upper: Vec<[f32; 3]>,

    /// Index to starting value and number of vectors per overlap.
    istr_vector: Vec<usize>,
    num_vectors: Vec<usize>,

    /// Original and rotated vectors.
    center: Vec<[f32; 3]>,
    vector: Vec<[f32; 3]>,
    cen_rot: Vec<[f32; 3]>,
    vec_rot: Vec<[f32; 3]>,

    /// Input volume sizes.
    nxyz_in: Vec<[i32; 3]>,
    /// Output volume size.
    nxyz_out: [i32; 3],

    /// Piece positions.
    ix_piece: Vec<i32>,
    iy_piece: Vec<i32>,

    /// Volume connectivity: index+1 of upper volume in X and Y (0=none).
    ivol_upper: Vec<[usize; 2]>,
    ivol_lower: Vec<[usize; 2]>,
    ind_vector: Vec<[usize; 2]>,
    num_vols: usize,

    /// Intervals between volumes.
    intervals: [[i32; 3]; 2],

    /// Solving index arrays.
    ivol_solve: Vec<usize>,
    num_solve: usize,
    ivol_full: Vec<usize>,

    /// Fit matrices.
    fit_mat: Vec<[[f32; 3]; 3]>,
    spacings: [[f32; 3]; 2],
    scale_xyz: f32,
    resid: Vec<[f32; 3]>,

    /// Inverse matrices and final shifts.
    f_inv_mat: Vec<[[f32; 3]; 3]>,
    f_inv_dxyz: Vec<[f32; 3]>,
    full_dxyz: Vec<[f32; 3]>,

    /// Vector grid description per edge.
    num_vec_grid: Vec<[i32; 3]>,
    vec_grid_start: Vec<[f32; 3]>,
    vec_grid_delta: Vec<[f32; 3]>,

    /// Edge geometry.
    num_edges: [i32; 2],
    num_pieces: i32,
    edge_min: Vec<[f32; 3]>,
    edge_max: Vec<[f32; 3]>,
    extended_ed_min: Vec<[f32; 3]>,
    extended_ed_max: Vec<[f32; 3]>,

    /// Band data for edge fractions.
    band_del: Vec<f32>,
    band_min: Vec<Vec<f32>>,
    band_max: Vec<Vec<f32>>,
    num_bands: Vec<usize>,

    /// Mean edge width and vector spacings.
    edge_width_mean: f32,
    del_mean: [f32; 3],

    /// Volume corner limits.
    vol_lim_tr: Vec<[f32; 2]>,
    vol_lim_bl: Vec<[f32; 2]>,
    vol_lim_tl: Vec<[f32; 2]>,
    vol_lim_br: Vec<[f32; 2]>,

    /// Map indices for variable search.
    map_gmag: usize,
    map_comp: usize,
    map_dmag: usize,
    map_skew: usize,
    map_alpha: usize,
    map_beta: usize,
    map_gamma: usize,
    if_unload: i32,
}

impl StitchVars {
    fn new(nvols: usize, nvecs: usize, nedges: usize) -> Self {
        Self {
            dxyz: vec![[0.0; 3]; nvols],
            gmag: vec![1.0; nvols],
            comp: vec![1.0; nvols],
            dmag: vec![0.0; nvols],
            skew: vec![0.0; nvols],
            alpha: vec![0.0; nvols],
            beta: vec![0.0; nvols],
            gamma: vec![0.0; nvols],
            pos_lower: vec![[0.0; 3]; nvecs],
            pos_upper: vec![[0.0; 3]; nvecs],
            istr_vector: vec![0; nedges + 1],
            num_vectors: vec![0; nedges],
            center: vec![[0.0; 3]; nvecs],
            vector: vec![[0.0; 3]; nvecs],
            cen_rot: vec![[0.0; 3]; nvecs],
            vec_rot: vec![[0.0; 3]; nvecs],
            nxyz_in: vec![[0; 3]; nvols],
            nxyz_out: [0; 3],
            ix_piece: vec![0; nvols],
            iy_piece: vec![0; nvols],
            ivol_upper: vec![[0; 2]; nvols],
            ivol_lower: vec![[0; 2]; nvols],
            ind_vector: vec![[0; 2]; nvols],
            num_vols: nvols,
            intervals: [[0; 3]; 2],
            ivol_solve: vec![0; nvols + 1],
            num_solve: 0,
            ivol_full: vec![0; nvols + 1],
            fit_mat: vec![[[0.0; 3]; 3]; nvols],
            spacings: [[0.0; 3]; 2],
            scale_xyz: 1.0,
            resid: vec![[0.0; 3]; nvecs],
            f_inv_mat: vec![[[0.0; 3]; 3]; nvols],
            f_inv_dxyz: vec![[0.0; 3]; nvols],
            full_dxyz: vec![[0.0; 3]; nvols],
            num_vec_grid: vec![[0; 3]; nedges],
            vec_grid_start: vec![[0.0; 3]; nedges],
            vec_grid_delta: vec![[0.0; 3]; nedges],
            num_edges: [0; 2],
            num_pieces: 0,
            edge_min: vec![[0.0; 3]; nedges],
            edge_max: vec![[0.0; 3]; nedges],
            extended_ed_min: vec![[0.0; 3]; nedges],
            extended_ed_max: vec![[0.0; 3]; nedges],
            band_del: vec![0.0; nedges],
            band_min: vec![vec![0.0; MAX_BAND]; nedges],
            band_max: vec![vec![0.0; MAX_BAND]; nedges],
            num_bands: vec![0; nedges],
            edge_width_mean: 0.0,
            del_mean: [0.0; 3],
            vol_lim_tr: vec![[0.0; 2]; nvols],
            vol_lim_bl: vec![[0.0; 2]; nvols],
            vol_lim_tl: vec![[0.0; 2]; nvols],
            vol_lim_br: vec![[0.0; 2]; nvols],
            map_gmag: 0,
            map_comp: 0,
            map_dmag: 0,
            map_skew: 0,
            map_alpha: 0,
            map_beta: 0,
            map_gamma: 0,
            if_unload: 0,
        }
    }

    /// Print a summary of the stitching variables.
    fn print_summary(&self) {
        println!("Stitching variables summary:");
        println!("  Number of volumes: {}", self.num_vols);
        println!("  Output size: {}x{}x{}", self.nxyz_out[0], self.nxyz_out[1], self.nxyz_out[2]);
        println!("  Intervals: X={} Y={}",
            self.intervals[0][0], self.intervals[1][1]);
        println!("  Scale factor: {:.6}", self.scale_xyz);
        println!("  Edge width mean: {:.2}", self.edge_width_mean);
        println!("  Mean vector deltas: [{:.2}, {:.2}, {:.2}]",
            self.del_mean[0], self.del_mean[1], self.del_mean[2]);

        for i in 0..self.num_vols.min(10) {
            println!("  Vol {}: piece=({},{}), size=({},{},{}), gmag={:.4}, alpha={:.2}",
                i, self.ix_piece[i], self.iy_piece[i],
                self.nxyz_in[i][0], self.nxyz_in[i][1], self.nxyz_in[i][2],
                self.gmag[i], self.alpha[i]);
        }
        if self.num_vols > 10 {
            println!("  ... and {} more volumes", self.num_vols - 10);
        }
    }
}

/// Read autodoc-style SMI file sections.
fn read_smi_sections(path: &str) -> io::Result<Vec<(String, String, Vec<(String, String)>)>> {
    let content = fs::read_to_string(path)?;
    let mut sections = Vec::new();
    let mut current: Option<(String, String, Vec<(String, String)>)> = None;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        if line.starts_with('[') {
            if let Some(sec) = current.take() { sections.push(sec); }
            let inner = line.trim_start_matches('[').trim_end_matches(']');
            let (stype, sname) = if let Some(eq) = inner.find('=') {
                (inner[..eq].trim().to_string(), inner[eq+1..].trim().to_string())
            } else {
                (inner.trim().to_string(), String::new())
            };
            current = Some((stype, sname, Vec::new()));
        } else if let Some(ref mut sec) = current {
            if let Some(eq) = line.find('=') {
                sec.2.push((line[..eq].trim().to_string(), line[eq+1..].trim().to_string()));
            }
        }
    }
    if let Some(sec) = current { sections.push(sec); }
    Ok(sections)
}

#[derive(Parser, Debug)]
#[command(name = "stitchvars")]
#[command(about = "Inspect and compute stitching variables from a supermontage info file")]
struct Cli {
    /// Supermontage info file (.smi)
    #[arg(short = 'i', long = "info")]
    info_file: PathBuf,

    /// Z value to inspect (default: first available)
    #[arg(short = 'z', long = "zvalue")]
    z_value: Option<i32>,

    /// Print verbose details
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();

    let info_path = cli.info_file.to_str().unwrap_or("");
    let sections = read_smi_sections(info_path).unwrap_or_else(|e| {
        eprintln!("ERROR: STITCHVARS - Cannot read info file: {e}");
        process::exit(1);
    });

    let pieces: Vec<_> = sections.iter().filter(|s| s.0 == "Piece").collect();
    let edges: Vec<_> = sections.iter().filter(|s| s.0 == "Edge").collect();

    println!("STITCHVARS: Stitching variables computation");
    println!("  Info file: {info_path}");
    println!("  {} pieces, {} edges found", pieces.len(), edges.len());

    // Collect Z values
    let mut z_values: Vec<i32> = Vec::new();
    for p in &pieces {
        if let Some((_, v)) = p.2.iter().find(|(k, _)| k == "Frame") {
            let nums: Vec<i32> = v.split_whitespace()
                .filter_map(|s| s.parse().ok()).collect();
            if nums.len() >= 3 && !z_values.contains(&nums[2]) {
                z_values.push(nums[2]);
            }
        }
    }
    println!("  Z values: {:?}", z_values);

    let target_z = cli.z_value.unwrap_or_else(|| {
        *z_values.first().unwrap_or(&0)
    });

    // Count volumes and edges at target Z
    let mut nvols = 0usize;
    let mut nedges = 0usize;
    for p in &pieces {
        if let Some((_, v)) = p.2.iter().find(|(k, _)| k == "Frame") {
            let nums: Vec<i32> = v.split_whitespace()
                .filter_map(|s| s.parse().ok()).collect();
            if nums.len() >= 3 && nums[2] == target_z {
                nvols += 1;
            }
        }
    }
    for e in &edges {
        if let Some((_, v)) = e.2.iter().find(|(k, _)| k == "lower") {
            let nums: Vec<i32> = v.split_whitespace()
                .filter_map(|s| s.parse().ok()).collect();
            if nums.len() >= 3 && nums[2] == target_z {
                nedges += 1;
            }
        }
    }

    println!("  At Z={target_z}: {nvols} volumes, {nedges} edges");

    // Create and initialize the stitch vars structure
    let nvecs_est = nedges * MAX_PATCH;
    let mut sv = StitchVars::new(nvols.max(1), nvecs_est.max(1), nedges.max(1));
    sv.num_vols = nvols;

    // Load volume info
    let mut vi = 0;
    for p in &pieces {
        if let Some((_, v)) = p.2.iter().find(|(k, _)| k == "Frame") {
            let nums: Vec<i32> = v.split_whitespace()
                .filter_map(|s| s.parse().ok()).collect();
            if nums.len() >= 3 && nums[2] == target_z {
                if vi < nvols {
                    sv.ix_piece[vi] = nums[0];
                    sv.iy_piece[vi] = nums[1];
                    if let Some((_, sv_str)) = p.2.iter().find(|(k, _)| k == "size") {
                        let snums: Vec<i32> = sv_str.split_whitespace()
                            .filter_map(|s| s.parse().ok()).collect();
                        if snums.len() >= 3 {
                            sv.nxyz_in[vi] = [snums[0], snums[1], snums[2]];
                        }
                    }
                    vi += 1;
                }
            }
        }
    }

    sv.print_summary();

    if cli.verbose {
        println!("\nDetailed volume information:");
        for i in 0..sv.num_vols {
            println!("  Volume {}: piece=({},{}), size=({},{},{})",
                i, sv.ix_piece[i], sv.iy_piece[i],
                sv.nxyz_in[i][0], sv.nxyz_in[i][1], sv.nxyz_in[i][2]);
            println!("    gmag={:.4}, comp={:.4}, dmag={:.4}, skew={:.4}",
                sv.gmag[i], sv.comp[i], sv.dmag[i], sv.skew[i]);
            println!("    alpha={:.2}, beta={:.2}, gamma={:.2}",
                sv.alpha[i], sv.beta[i], sv.gamma[i]);
        }
    }

    println!("STITCHVARS complete.");
}
