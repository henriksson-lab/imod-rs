//! Translation of `IMOD/imodutil/pickbestseed.cpp` and its header
//! `pickbestseed.h`: picks the best-spaced, best-scoring beads from several
//! tracked seed models (run by `autofidseed`).
//!
//! Fixed in translation (`BUGS.md`, pickbestseed): `addPointsInGaps` looked
//! for already-accepted neighbours in domains `0..numNeighbors` instead of the
//! candidate's neighbouring domains, and kept only the last "too close"
//! verdict instead of any; `computeAreaFracs` weighted each ring by
//! `(2 * ring + delr) * delr` where the ring's area is
//! `(2 * ring + 1) * delr * delr`; `analyzeElongation` read the never-filled
//! `wsums` array (only with `-control 17,...`).  See each site.

use std::cell::Cell;
use std::io::Write;
use std::sync::{Arc, Mutex};

use crate::imod::clip::clip::{ScanArg, sscanf};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, exit, fgetline, imod_backup_file, imod_prog_name,
    imod_usage_header, number_in_list, program_args,
};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_float, pip_get_float_array, pip_get_integer,
    pip_get_string, pip_get_two_floats, pip_get_two_integers, pip_number_of_entries,
    pip_read_or_parse_options,
};
use crate::imod::libcfshr::robuststat::{
    rs_fast_median_in_place, rs_mad_median_outliers, rs_madn, rs_sort_indexed_floats,
};
use crate::imod::libcfshr::simplestat::{avg_sd, ls_fit_pred};
use crate::imod::libimod::icont::{imod_contours_new, imodel_contour_scan};
use crate::imod::libimod::ilabel::{imod_label_item_add, imod_label_new};
use crate::imod::libimod::imodel::{Icont, Imod, Iobj, Ipoint};
use crate::imod::libimod::imodel_files::{imod_read, imod_write};
use crate::imod::libimod::iobj::{MATFLAGS2_SKIP_HIGH, MATFLAGS2_SKIP_LOW};
use crate::imod::libimod::ipoint::{
    imod_point_append, imod_point_distance, imod_point_inside_area, make_area_cont_list,
};
use crate::imod::libimod::istore::{
    GEN_STORE_BYTE, GEN_STORE_COLOR, GEN_STORE_FLOAT, GEN_STORE_MINMAX1, GEN_STORE_SURFACE,
    GEN_STORE_VALUE1, Istore, istore_add_min_max, istore_get_min_max, istore_insert,
};

/// `pickbestseed.h:7`.
const MAX_MODELS: usize = 7;
/// `pickbestseed.h:9`.
const MAX_AREAS: usize = 1000;
/// `pickbestseed.h:10-11`.
const OPTION_FIT_TO_WSUM: i32 = 1;
const OPTION_WSUM_GROUPS: i32 = 2;

/// `#define PI 3.141593` (`pickbestseed.cpp:21`) -- a double, and a truncated
/// literal, not `f64::consts::PI`.
const PI: f64 = 3.141593;
/// `#define MAXLINE 1000`.
const MAXLINE: i32 = 1000;
/// `b3dutil.h`: `#define RADIANS_PER_DEGREE 0.01745329252`.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// `B3DNINT(a)`: `(int)floor((a) + 0.5)`, the `0.5` a double.
macro_rules! b3dnint {
    ($a:expr) => {
        (($a) as f64 + 0.5).floor() as i32
    };
}

/// `B3DMAX(a,b)`: `((a) > (b) ? (a) : (b))`.
macro_rules! b3dmax {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a > b { a } else { b }
    }};
}

/// `B3DMIN(a,b)`: `((a) < (b) ? (a) : (b))`.
macro_rules! b3dmin {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a < b { a } else { b }
    }};
}

/// `printf` with the source's format.
macro_rules! printf {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = ImodFile::Stdout.write_all(&c_format_bytes($fmt, &[$($arg),*]));
    }};
}

/// `exitError` with the source's variadic format.
macro_rules! exit_error_fmt {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {
        exit_error(&c_format_bytes($fmt, &[$($arg),*]))
    };
}

/// `cppdefs.h:21-24` `PRINT1`..`PRINT4`: `cout << #a << " = " << a` joined by
/// `",  "`, ending in `endl`.  `cout` prints a float or double with the
/// default precision 6, which is `%g`; an int as `%d`.  Each argument is a
/// `(name, CArg)` pair, the name being the source's stringized expression.
macro_rules! print_vars {
    ($(($name:expr, $arg:expr)),+ $(,)?) => {{
        let mut text: Vec<u8> = Vec::new();
        let mut first = true;
        $(
            if !first {
                text.extend_from_slice(b",  ");
            }
            first = false;
            text.extend_from_slice($name.as_bytes());
            text.extend_from_slice(b" = ");
            let arg = $arg;
            let fmt = match arg {
                CArg::Dbl(_) => "%g",
                _ => "%d",
            };
            text.extend_from_slice(&c_format_bytes(fmt, &[arg]));
        )+
        let _ = first;
        text.push(b'\n');
        let _ = ImodFile::Stdout.write_all(&text);
    }};
}

/// Structure to hold data from one track, indexed by contour number
/// (`pickbestseed.h:14`).
#[derive(Clone, Copy, Debug, Default)]
struct TrackData {
    model: i32,
    mid_pos: Ipoint,
    domain: i32,
    cand_index: i32,
    top_bot: i32,
    residual: f32,
    edge_sd_mean: f32,
    edge_sd_median: f32,
    edge_sd_sd: f32,
    elong_mean: f32,
    elong_median: f32,
    elong_sd: f32,
    // float outerMAD;
    // float outerBkgd;
    wsum_mean: f32,
}

/// Structure to hold data about a candidate point (`pickbestseed.h:33`).
#[derive(Clone, Copy, Debug, Default)]
struct Candidate {
    contours: [i32; MAX_MODELS],
    domain: i32,
    pos: Ipoint,
    top_bot: i32,
    score: f32,
    mean_deviation: f32,
    clustered: u8,
    overlapped: u8,
    accepted: u8,
    overlap_tmp: u8,
}

/// Structure for data about a grid point (`pickbestseed.h:47`).
#[derive(Clone, Copy, Debug, Default)]
struct GridPoint {
    domain: i32,
    x: f32,
    y: f32,
    area_frac: f32,
    density: f32,
    exclude: i32,
}

/// Structure to hold data about a domain (`pickbestseed.h:56`).
#[derive(Clone, Debug, Default)]
struct GridDomain {
    num_candidates: i32,
    num_accepted: i32,
    cand_start_ind: i32,
    num_tracks: i32,
    track_start_ind: i32,
    num_grid_pts: i32,
    grid_pt_ind: Vec<i32>,
    num_neighbors: i32,
    neighbors: [i32; 9],
}

/// The values of pickbestseed's final `Final:` report (`outputNumAccepted`
/// with `final` set), for a direct caller (`autofidseed`, which parsed the
/// integers after each `=` of that line).  `on_bottom_top` is present when
/// the line reports the two surfaces.
#[derive(Clone, Debug, Default)]
pub struct PickbestseedResult {
    pub final_total: Option<i32>,
    pub on_bottom_top: Option<(i32, i32)>,
}

thread_local! {
    /// Where [`pickbestseed`] records its [`PickbestseedResult`] on this
    /// thread, when a direct caller set it through [`pickbestseed_recording`].
    static RESULT_SINK: std::cell::RefCell<Option<Arc<Mutex<PickbestseedResult>>>> =
        const { std::cell::RefCell::new(None) };
    /// `static int lastPhase = -1` in `outputDensities`.
    static LAST_PHASE: Cell<i32> = const { Cell::new(-1) };
    /// `static int sequence` in `outputDensities`.
    static SEQUENCE: Cell<i32> = const { Cell::new(0) };
}

/// Rust-only: runs program [`pickbestseed`] on this thread with its final
/// counts recorded into `sink`.  The program ends through `exit`, so the
/// values reach the caller through `sink`; run it under
/// `commands::call_in_process`.
pub fn pickbestseed_recording(sink: Arc<Mutex<PickbestseedResult>>) {
    RESULT_SINK.with_borrow_mut(|slot| *slot = Some(sink));
    pickbestseed();
}

/// Rust-only: records into the direct caller's [`PickbestseedResult`], if any.
fn record_result(update: impl FnOnce(&mut PickbestseedResult)) {
    RESULT_SINK.with_borrow(|slot| {
        if let Some(sink) = slot {
            update(&mut sink.lock().expect("pickbestseed result sink"));
        }
    });
}

/// Rust-only adapter: `PipReadOrParseOptions` takes the usage-header callback
/// as `void (*)(const char *)`; `imodUsageHeader` is that function.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// Class `PickSeeds` (`pickbestseed.h:67`).
struct PickSeeds {
    m_track_mods: Vec<Imod>,
    m_tracks: Vec<TrackData>,
    m_domains: Vec<GridDomain>,
    m_del_xdomain: f32,
    m_del_ydomain: f32,
    m_num_xdomains: i32,
    m_num_ydomains: i32,
    m_num_xgrid: i32,
    m_num_ygrid: i32,
    m_num_area_cont: i32,
    m_area_conts: [i32; MAX_AREAS],
    m_area_mod: Option<Imod>,
    m_area_xmin: f32,
    m_area_xmax: f32,
    m_area_ymin: f32,
    m_area_ymax: f32,
    m_candidates: Vec<Candidate>,
    m_num_candidates: i32,
    m_num_accepted: [i32; 3],
    m_grid_points: Vec<GridPoint>,
    m_candid_list: Vec<i32>,
    m_accept_list: Vec<i32>,
    m_rank_index: Vec<i32>,
    m_phase: i32,
    m_append_to_seed: i32,
    m_bead_size: f32,
    m_del_xgrid: f32,
    m_del_ygrid: f32,
    m_exclude_areas: i32,
    m_verbose: i32,
    m_overlap_target: f32,
    m_dens_plot_name: Option<String>,
    m_num_wsum: i32,
    m_wsum_mean: f32,
    m_num_outer_mad: i32,
    m_outer_mad_mean: f32,
    m_outlier_score: f32,

    m_close_diam_frac: f32,
    m_exclude_fac: f32,
    m_ring_spacing_fac: f32,
    m_num_rings: i32,
    m_highest_tilt: f32,
    m_cos_rot: f32,
    m_sin_rot: f32,
    m_cos_tilt: f32,
    m_cluster_crit: f32,
    m_edge_outlier_crit: f32,
    m_elong_for_overlap: f32,
    m_elong_crit_scaling: f32,
    m_option_flags: i32,
    m_elong_combo_angle: f32,
    m_edge_combo_angle: f32,
    m_norm_combo_angle: f32,
}

/// Original: `main` (`pickbestseed.cpp:26`) -- "The shortest main in the West".
pub fn pickbestseed() {
    let mut pick = PickSeeds::new();
    pick.main(&program_args());
    exit(0);
}

impl PickSeeds {
    /// Original: `PickSeeds::PickSeeds` (`pickbestseed.cpp:36`).
    fn new() -> PickSeeds {
        PickSeeds {
            m_track_mods: Vec::new(),
            m_tracks: Vec::new(),
            m_domains: Vec::new(),
            m_del_xdomain: 0.,
            m_del_ydomain: 0.,
            m_num_xdomains: 0,
            m_num_ydomains: 0,
            m_num_xgrid: 0,
            m_num_ygrid: 0,
            m_num_area_cont: 0,
            m_area_conts: [0; MAX_AREAS],
            m_area_mod: None,
            m_area_xmin: 0.,
            m_area_xmax: 0.,
            m_area_ymin: 0.,
            m_area_ymax: 0.,
            m_candidates: Vec::new(),
            m_num_candidates: 0,
            m_num_accepted: [0; 3],
            m_grid_points: Vec::new(),
            m_candid_list: Vec::new(),
            m_accept_list: Vec::new(),
            m_rank_index: Vec::new(),
            m_phase: 1,
            m_append_to_seed: 0,
            m_bead_size: 0.,
            m_del_xgrid: 0.,
            m_del_ygrid: 0.,
            m_exclude_areas: 0,
            m_verbose: 0,
            m_overlap_target: -1.,
            m_dens_plot_name: None,
            m_num_wsum: 0,
            m_wsum_mean: 0.,
            m_num_outer_mad: 0,
            m_outer_mad_mean: 0.,
            m_outlier_score: 0.,
            m_highest_tilt: 0.,
            m_cos_rot: 0.,
            m_sin_rot: 0.,
            m_cos_tilt: 0.,

            // Parameters
            // 1: Deviation between points as fraction of bead diameter for tracks to be close
            m_close_diam_frac: 0.5,
            // 2: Multiple of target spacing at which to exclude points from further searches
            m_exclude_fac: 0.75,
            // 3: Width of rings for finding points when filling gaps, as fraction of target spacing
            m_ring_spacing_fac: 0.25,
            // 4: Number of rings to search
            m_num_rings: 4,
            // 5: Maximum # of bead diameters separation for points to be considered clustered
            m_cluster_crit: 1.375,
            // 7: Scaling for both mElongForOverlap and mEdgeOutlierCrit
            m_elong_crit_scaling: 1.,
            // 12: Criterion for edge SD values or elongations to be considered outliers
            m_edge_outlier_crit: 2.24,
            // 16: Absolute threshold for elongation to be considered overlap
            m_elong_for_overlap: 3.2,
            // 17: Option flags, 1 = divide edgeSD by outerMAD, 2 = equalize over wsum
            m_option_flags: 0,
            // 18: Angle for rotating edgeSdmean vs. edgeSDsd
            m_edge_combo_angle: -59.,
            // 19: Angle for rotating elongMean vs. elongSD
            m_elong_combo_angle: -67.,
            // 20: Angle for rotating normalized edgeSD and elongation values for combining
            m_norm_combo_angle: 45.,
        }
    }

    /// Original: `PickSeeds::main` (`pickbestseed.cpp:79`) -- the real main
    /// procedure.
    fn main(&mut self, argv: &[String]) {
        // Parameters
        // 6: Fraction of points that must be close in two tracks for them to be considered same
        let mut crit_frac_close: f32 = 0.6;
        // 8: Maximum fraction of target density at which to add points in initial phase
        let mut min_density_fac1: f32 = 0.9;
        // 9: Higher fraction of target at which to add points in more desperate searches
        let mut min_density_fac2: f32 = 1.1;
        // 10: Scaling from desired spacing to H for kernel density computation
        let mut spacing_to_hfac: f32 = 1.3;
        // 11: Scaling from desired spacing to density grid spacing
        let mut spacing_to_grid_fac: f32 = 0.2;
        // 13: Ratio of minority to majority for using higher density factor
        let mut surf_ratio_use_dens2: f32 = 0.65;
        // 14: Fraction of nominal spacing allowed for initial addition of points
        let mut init_add_spacing_fac: f32 = 0.85;
        // 15: Fraction of spacing for adding best half of points on next phase
        let mut half_add_spacing_fac: f32 = 0.7;
        // 21: Maximum number above MADNs above median score for a candidate to be accepted
        let mut max_madns_for_score: f32 = 8.;
        // 22: Maximum number above MADNs above median score for a candidate to be accepted
        let mut max_clust_elong_score_madns: f32 = 12.;

        // Defaults for options
        let mut use_clusters = 0;
        let mut use_overlaps = 0;
        let mut two_surf = 0;
        let mut phase_as_surf = 0;
        let mut rotation: f32 = 0.;
        let mut no_beef_up = 0;
        let mut x_border = 0;
        let mut y_border = 0;
        let mut bound_for_count = 0;

        // Indices for the weights, and their default values.
        const WGT_COMPLETE: usize = 0;
        const WGT_NUM_MODELS: usize = 1;
        const WGT_RESIDUAL: usize = 2;
        const WGT_DEVIATION: usize = 3;
        let mut weights: [f32; 4] = [1., 1., 1., 1.];

        let mut filename: Vec<u8> = Vec::new();
        let mut out_name: Vec<u8> = Vec::new();
        let mut elong_name: Option<String> = None;
        let progname = imod_prog_name(argv.first().map(String::as_str).unwrap_or(""));
        let mut num_opt_args = 0;
        let mut num_non_opt_args = 0;
        let mut max_conts: i32;
        let mut nx_image = 0;
        let mut ny_image = 0;
        let mut iz_middle = 0;
        let mut target_number = 0;
        let if_density: i32;
        let if_number: i32;
        let mut j: i32;
        let mut len: i32;
        let mut co: i32 = 0;
        let surf_number: i32;
        let num_domains: i32;
        let max_grid_per_dom: i32;
        let mut ix: i32 = 0;
        let mut iy: i32 = 0;
        let mut ind: i32;
        let mut max_track_len: i32;
        let mut cont_for_track_pos: i32 = 0;
        let mut model_seed_zvalues = [0i32; MAX_MODELS];
        let mut num_seed_zvalues = 0;
        let mut target_density: f32 = 0.;
        let mut total_area: f32;
        let target_spacing: f32;
        let surf_density: f32;
        let surf_spacing: f32;
        let kernel_h: f32;
        let grid_spacing: f32;
        let mut xx: f32 = 0.;
        let mut yy: f32 = 0.;
        let mut dzmin: f32;
        let mut dz: f32;
        let mut frac_close: f32 = 0.;
        let mut mean_dev: f32 = 0.;
        let mut comp_sum: f32;
        let mut res_sum: f32;
        let mut imin: i32 = 0;
        let mut ncum: i32 = 0;
        let mut idom: i32;
        let mut neigh_co: i32;
        let mut neigh_mod: i32;
        let mut co_mod: i32;
        let mut ndev: i32;
        let mut bot_sum: i32;
        let mut top_sum: i32;
        let mut ob: i32;
        let mut use_dens2: i32;
        let mut major_number: i32;
        let num_phase: i32;
        let mut overlap_thresh: i32;
        let mut top_bot: i32 = 0;
        let mut resid: f32 = 0.;
        let mut sd_mean: f32 = 0.;
        let mut sd_med: f32 = 0.;
        let mut sd_sd: f32 = 0.;
        let mut term_dens: f32;
        let mut best_res_for_pos: f32;
        let mut cluster_thresh: f32;
        let mut major_density: f32;
        let mut major_spacing: f32;
        let mut major_kernel_h: f32;
        let mut valmin: f32;
        let mut valmax: f32;
        let mut elo_mean: f32 = 0.;
        let mut elo_med: f32 = 0.;
        let mut elo_sd: f32 = 0.;
        let mut outer_mad: f32 = 0.;
        let mut wsum: f32 = 0.;
        let mut num_models = 0;
        let mut num_clust: i32;
        let mut num_over: i32;
        let mut num_ignore: i32;
        let num_base_cand: i32;
        let mut num_cont_for_count = 0;
        let mut num_in_surf = [0i32; 30];
        let mut num_inside = [0i32; 30];
        let mut num_outside = [0i32; 30];
        let mut tot_inside: i32;
        let mut tot_outside: i32;
        let mut edge_tmp: Vec<f32>;
        let mut look_more: bool;
        let mut bad_at_middle: bool;
        let mut track_list: Vec<i32>;
        let mut base_mod: Option<Imod> = None;
        // The source's `candid` is one stack struct reused for every candidate;
        // `contours` past `numModels` are never set, which only the verbose
        // report reads -- they are defined here as -1.
        let mut candid = Candidate {
            contours: [-1; MAX_MODELS],
            ..Candidate::default()
        };
        let mut all_scores: Vec<f32> = Vec::new();
        let mut score_temp: Vec<f32>;
        let mut score_median: f32 = 0.;
        let mut score_madn: f32 = 0.;
        let mut line = [0u8; MAXLINE as usize];
        let mut store = Istore::default();
        const MAX_COLORS: usize = 10;
        let rgba: [[u8; 4]; MAX_COLORS] = [
            [255, 0, 255, 0],
            [255, 255, 0, 0],
            [0, 255, 255, 0],
            [255, 0, 0, 0],
            [0, 0, 255, 0],
            [255, 128, 0, 0],
            [153, 102, 229, 0],
            [51, 51, 204, 0],
            [229, 153, 102, 0],
            [153, 51, 51, 0],
        ];
        const MAX_COLORS2: usize = 7;
        let rgba2: [[u8; 4]; MAX_COLORS2] = [
            [255, 0, 255, 0],
            [0, 255, 0, 0],
            [255, 255, 0, 0],
            [100, 100, 0, 0],
            [255, 0, 0, 0],
            [85, 170, 255, 0],
            [255, 128, 0, 0],
        ];
        let file_opt_name: [&[u8]; 2] = [b"ElongationFile", b"SurfaceFile"];

        // Fallbacks from    ../manpages/autodoc2man 2 1 pickbestseed
        let num_options = 29;
        let options: [&[u8]; 29] = [
            b"tracked:TrackedModel:FNM:",
            b"surface:SurfaceFile:FNM:",
            b"resid:ElongationFile:FNM:",
            b"output:OutputSeedModel:FN:",
            b"append:AppendToSeedModel:B:",
            b"size:BeadSize:F:",
            b"image:ImageSizeXandY:IP:",
            b"border:BordersInXandY:IP:",
            b"middle:MiddleZvalue:I:",
            b"zseed:SeedZvalue:IM:",
            b"two:TwoSurfaces:B:",
            b"boundary:BoundaryModel:FN:",
            b"exclude:ExcludeInsideAreas:B:",
            b"counting:BoundaryForCounting:B:",
            b"number:TargetNumberOfBeads:I:",
            b"density:TargetDensityOfBeads:F:",
            b"nobeef:LimitMajorityToTarget:B:",
            b"elongated:ElongatedPointsAllowed:I:",
            b"cluster:ClusteredPointsAllowed:I:",
            b"lower:LowerTargetForClustered:F:",
            b"rotation:RotationAngle:F:",
            b"highest:HighestTiltAngle:F:",
            b"weights:WeightsForScore:FA:",
            b"control:ControlValue:FPM:",
            b"phase:PhaseOutput:B:",
            b"root:DensityOutputRootname:CH:",
            b"candidate:CandidateModel:FN:",
            b"verbose:VerboseOutput:I:",
            b"help:usage:B:",
        ];

        let argv_bytes: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
        pip_read_or_parse_options(
            argv_bytes.len() as i32,
            &argv_bytes,
            &options,
            num_options,
            progname.as_bytes(),
            9,
            1,
            1,
            &mut num_opt_args,
            &mut num_non_opt_args,
            Some(imod_usage_header_for_pip),
        );

        // Read the tracked models, get maximum number of contours (They should all match...)
        if pip_number_of_entries(b"TrackedModel", &mut num_models) != 0 {
            exit_error(b"At least one tracked model must be entered");
        }
        if num_models > MAX_MODELS as i32 {
            exit_error_fmt!(
                "At most %d tracked models can be entered",
                CArg::Int(MAX_MODELS as i64)
            );
        }
        pip_number_of_entries(b"SeedZvalue", &mut num_seed_zvalues);
        if num_seed_zvalues != 0 && num_seed_zvalues != num_models {
            exit_error(b"If SeedZvalue is entered, there must be an entry for each tracked model");
        }

        max_conts = 0;
        for i in 0..num_models as usize {
            pip_get_string(b"TrackedModel", &mut filename);
            let fname = String::from_utf8_lossy(&filename).into_owned();
            match imod_read(&fname) {
                Ok(model) => self.m_track_mods.push(model),
                Err(_) => exit_error_fmt!("Reading tracked model %s", CArg::Str(&fname)),
            }
            if self.m_track_mods[i].obj.is_empty() || self.m_track_mods[i].obj[0].cont.is_empty() {
                exit_error_fmt!("No contours in tracked model %s", CArg::Str(&fname));
            }
            max_conts = b3dmax!(max_conts, self.m_track_mods[i].obj[0].cont.len() as i32);
            if num_seed_zvalues != 0 {
                pip_get_integer(b"SeedZvalue", &mut model_seed_zvalues[i]);
            }
        }

        // Get other options
        if pip_get_integer(b"MiddleZvalue", &mut iz_middle) != 0
            || pip_get_float(b"BeadSize", &mut self.m_bead_size) != 0
            || pip_get_two_integers(b"ImageSizeXandY", &mut nx_image, &mut ny_image) != 0
            || pip_get_string(b"OutputSeedModel", &mut out_name) != 0
        {
            exit_error(
                b"Middle Z value, bead size, image size, and output filename must be entered",
            );
        }
        let out_name = String::from_utf8_lossy(&out_name).into_owned();

        pip_get_boolean(b"TwoSurfaces", &mut two_surf);
        pip_get_boolean(b"PhaseOutput", &mut phase_as_surf);
        pip_get_boolean(b"AppendToSeedModel", &mut self.m_append_to_seed);
        pip_get_boolean(b"BoundaryForCounting", &mut bound_for_count);
        pip_get_integer(b"VerboseOutput", &mut self.m_verbose);
        let mut name: Vec<u8> = Vec::new();
        if pip_get_string(b"DensityOutputRootname", &mut name) == 0 {
            self.m_dens_plot_name = Some(String::from_utf8_lossy(&name).into_owned());
        }
        if pip_get_string(b"CandidateModel", &mut name) == 0 {
            elong_name = Some(String::from_utf8_lossy(&name).into_owned());
        }
        pip_get_float(b"RotationAngle", &mut rotation);
        pip_get_float(b"HighestTiltAngle", &mut self.m_highest_tilt);
        pip_get_integer(b"ClusteredPointsAllowed", &mut use_clusters);
        use_clusters = b3dmax!(0, b3dmin!(4, use_clusters));
        if pip_get_integer(b"ElongatedPointsAllowed", &mut use_overlaps) == 0 {
            if use_clusters > 1 {
                exit_error(b"You cannot enter both -elongated and -cluster with a value > 1");
            }
            use_overlaps = b3dmax!(0, b3dmin!(3, use_overlaps));
        } else if use_clusters != 0 {
            use_overlaps = use_clusters - 1;
            use_clusters = 1;
        }

        pip_get_float(b"LowerTargetForClustered", &mut self.m_overlap_target);
        pip_get_boolean(b"LimitMajorityToTarget", &mut no_beef_up);
        pip_get_two_integers(b"BordersInXandY", &mut x_border, &mut y_border);
        iy = 4;
        ix = pip_get_float_array(b"WeightsForScore", &mut weights, &mut iy, 4);
        if ix < 0 || iy != 4 {
            exit_error(b"You must enter exactly 4 values for weights");
        }
        weights[WGT_NUM_MODELS] =
            (weights[WGT_NUM_MODELS] as f64 / b3dmax!(num_models as f64 - 1., 1.)) as f32;

        // Allow any parameter to be set
        pip_number_of_entries(b"ControlValue", &mut ncum);
        for _ in 0..ncum {
            pip_get_two_floats(b"ControlValue", &mut xx, &mut yy);
            // cppdefs.h:17-18: SET_CONTROL_FLOAT / SET_CONTROL_INT print
            // `cout << #b << " set to " << value << endl`.
            macro_rules! set_float {
                ($var:expr, $name:expr) => {{
                    $var = yy;
                    printf!("%s set to %g\n", CArg::Str($name), CArg::Dbl(yy as f64));
                }};
            }
            macro_rules! set_int {
                ($var:expr, $name:expr) => {{
                    $var = b3dnint!(yy);
                    printf!(
                        "%s set to %d\n",
                        CArg::Str($name),
                        CArg::Int(b3dnint!(yy) as i64)
                    );
                }};
            }
            match b3dnint!(xx) {
                1 => set_float!(self.m_close_diam_frac, "mCloseDiamFrac"),
                2 => set_float!(self.m_exclude_fac, "mExcludeFac"),
                3 => set_float!(self.m_ring_spacing_fac, "mRingSpacingFac"),
                4 => set_int!(self.m_num_rings, "mNumRings"),
                5 => set_float!(self.m_cluster_crit, "mClusterCrit"),
                6 => set_float!(crit_frac_close, "critFracClose"),
                7 => set_float!(self.m_elong_crit_scaling, "mElongCritScaling"),
                8 => set_float!(min_density_fac1, "minDensityFac1"),
                9 => set_float!(min_density_fac2, "minDensityFac2"),
                10 => set_float!(spacing_to_hfac, "spacingToHfac"),
                11 => set_float!(spacing_to_grid_fac, "spacingToGridFac"),
                12 => set_float!(self.m_edge_outlier_crit, "mEdgeOutlierCrit"),
                13 => set_float!(surf_ratio_use_dens2, "surfRatioUseDens2"),
                14 => set_float!(init_add_spacing_fac, "initAddSpacingFac"),
                15 => set_float!(half_add_spacing_fac, "halfAddSpacingFac"),
                16 => set_float!(self.m_elong_for_overlap, "mElongForOverlap"),
                17 => set_int!(self.m_option_flags, "mOptionFlags"),
                18 => set_float!(self.m_edge_combo_angle, "mEdgeComboAngle"),
                19 => set_float!(self.m_elong_combo_angle, "mElongComboAngle"),
                20 => set_float!(self.m_norm_combo_angle, "mNormComboAngle"),
                21 => set_float!(max_madns_for_score, "maxMADNsForScore"),
                22 => set_float!(max_clust_elong_score_madns, "maxClustElongScoreMADNs"),
                _ => {}
            }
        }

        // Multiply both elongation criteria, whether entered or not, by the criterion scaling
        self.m_elong_for_overlap *= self.m_elong_crit_scaling;
        self.m_edge_outlier_crit *= self.m_elong_crit_scaling;

        if_density = 1 - pip_get_float(b"TargetDensityOfBeads", &mut target_density);
        if_number = 1 - pip_get_integer(b"TargetNumberOfBeads", &mut target_number);
        if if_number + if_density != 1 {
            exit_error(b"Target number or density must be entered, not both");
        }
        self.m_area_xmin = x_border as f32;
        self.m_area_xmax = (nx_image - x_border) as f32;
        self.m_area_ymin = y_border as f32;
        self.m_area_ymax = (ny_image - y_border) as f32;
        total_area = ((nx_image - 2 * x_border) * (ny_image - 2 * y_border)) as f32;

        // Read base model to append to
        if self.m_append_to_seed != 0 {
            match imod_read(&out_name) {
                Ok(model) => base_mod = Some(model),
                Err(_) => {
                    exit_error_fmt!("Reading existing seed model %s", CArg::Str(&out_name))
                }
            }
        }

        // Read area model if any and get the area
        if pip_get_string(b"BoundaryModel", &mut filename) == 0 {
            let fname = String::from_utf8_lossy(&filename).into_owned();
            match imod_read(&fname) {
                Ok(model) => self.m_area_mod = Some(model),
                Err(_) => exit_error_fmt!("Reading boundary model %s", CArg::Str(&fname)),
            }
            pip_get_boolean(b"ExcludeInsideAreas", &mut self.m_exclude_areas);
            let area_mod = self.m_area_mod.as_ref().unwrap();
            if area_mod.obj.is_empty() || area_mod.obj[0].cont.is_empty() {
                exit_error(b"No contours in object 1 of boundary model");
            }
            if make_area_cont_list(
                &area_mod.obj[0],
                iz_middle,
                &mut self.m_area_conts,
                &mut self.m_num_area_cont,
                MAX_AREAS as i32,
            ) != 0
            {
                exit_error_fmt!(
                    "Too many contours on one section in boundary model for array (limit is %d)",
                    CArg::Int(MAX_AREAS as i64)
                );
            }

            if bound_for_count != 0 {
                num_cont_for_count = self.m_num_area_cont;
                self.m_num_area_cont = 0;
            } else {
                // Get the area by converting each contour to scan contour, clipping each
                // segment to be within borders, and adding up segment lengths
                if self.m_exclude_areas == 0 {
                    total_area = 0.;
                }
                for co in 0..self.m_num_area_cont as usize {
                    let Some(cont) = imodel_contour_scan(Some(
                        &area_mod.obj[0].cont[self.m_area_conts[co] as usize],
                    )) else {
                        exit_error(b"Creating scan contour from boundary contour")
                    };
                    let mut i = 0usize;
                    while (i as i32) < cont.pts.len() as i32 {
                        xx = b3dmax!(self.m_area_xmin, cont.pts[i].x);
                        yy = b3dmin!(self.m_area_xmax, cont.pts[i + 1].x);
                        total_area += b3dmax!(0.0f32, yy - xx)
                            * (if self.m_exclude_areas != 0 { -1. } else { 1. });
                        i += 2;
                    }
                }
            }
        }

        printf!(
            "Total area = %.2f megapixels\n",
            CArg::Dbl(total_area as f64 / 1.0e6)
        );

        self.m_tracks = vec![TrackData::default(); max_conts as usize];
        track_list = vec![0; max_conts as usize];
        edge_tmp = vec![0.; max_conts as usize];
        for i in 0..max_conts as usize {
            self.m_tracks[i].model = -1;
            self.m_tracks[i].domain = -1;
            self.m_tracks[i].cand_index = -1;
        }

        // Read in the residual elongation and top/bottom data
        num_ignore = 0;
        for lp in 0..=two_surf as usize {
            for i in 0..num_models {
                if pip_get_string(file_opt_name[lp], &mut filename) != 0 {
                    if lp != 0 {
                        num_ignore += 1;
                        continue;
                    }
                    exit_error_fmt!(
                        "%d files must be entered with the %s option",
                        CArg::Int(num_models as i64),
                        CArg::Bytes(file_opt_name[lp])
                    );
                }
                let fname = String::from_utf8_lossy(&filename).into_owned();
                let Some(mut fp) = ImodFile::open(&fname, "r") else {
                    exit_error_fmt!("Opening file %s", CArg::Str(&fname))
                };

                loop {
                    len = fgetline(&mut fp, &mut line, MAXLINE);
                    if len == 0 {
                        continue;
                    }
                    if len == -1 {
                        exit_error_fmt!("Reading file %s", CArg::Str(&fname));
                    }
                    if len == -2 {
                        break;
                    }
                    let end = line.iter().position(|&b| b == 0).unwrap_or(line.len());
                    let text = String::from_utf8_lossy(&line[..end]).into_owned();
                    if lp != 0 {
                        sscanf(
                            &text,
                            "%d %d %d %d",
                            &mut [
                                ScanArg::Int(&mut ix),
                                ScanArg::Int(&mut co),
                                ScanArg::Int(&mut iy),
                                ScanArg::Int(&mut top_bot),
                            ],
                        );
                    } else {
                        outer_mad = 0.;
                        wsum = 0.;
                        sscanf(
                            &text,
                            "%d %d %f %f %f %f %f %f %f %f",
                            &mut [
                                ScanArg::Int(&mut ix),
                                ScanArg::Int(&mut co),
                                ScanArg::Flt(&mut resid),
                                ScanArg::Flt(&mut sd_mean),
                                ScanArg::Flt(&mut sd_med),
                                ScanArg::Flt(&mut sd_sd),
                                ScanArg::Flt(&mut elo_mean),
                                ScanArg::Flt(&mut elo_med),
                                ScanArg::Flt(&mut elo_sd),
                                /* &outerMAD, &outerBkgd, */
                                ScanArg::Flt(&mut wsum),
                            ],
                        );
                        let _ = outer_mad;
                        if resid < 0. || sd_mean < 0. {
                            continue;
                        }
                    }
                    if co < 1 || co > self.m_track_mods[i as usize].obj[0].cont.len() as i32 {
                        exit_error_fmt!(
                            "Contour number (%d) out of range in %s",
                            CArg::Int(co as i64),
                            CArg::Str(&fname)
                        );
                    }
                    co -= 1;
                    let track = &mut self.m_tracks[co as usize];
                    if lp != 0 {
                        track.top_bot = top_bot;
                    } else {
                        track.model = i;
                        track.top_bot = 0;
                        track.residual = resid;
                        track.edge_sd_mean = sd_mean;
                        track.edge_sd_median = sd_med;
                        track.edge_sd_sd = sd_sd;
                        track.elong_mean = elo_mean;
                        track.elong_median = elo_med;
                        track.elong_sd = elo_sd;
                        /* mTracks[co].outerMAD = outerMAD;
                        mTracks[co].outerBkgd = outerBkgd; */
                        track.wsum_mean = wsum;
                        if wsum > 0. {
                            self.m_num_wsum += 1;
                            self.m_wsum_mean += wsum;
                        }
                        if outer_mad > 0. {
                            self.m_num_outer_mad += 1;
                            self.m_outer_mad_mean += outer_mad;
                        }
                    }
                    if len < 0 {
                        break;
                    }
                }
            }
        }
        if num_ignore == num_models {
            exit_error(
                b"Surface information must be provided for at least one model if -twosurf is given",
            );
        }
        self.m_outer_mad_mean /= b3dmax!(1, self.m_num_outer_mad) as f32;
        self.m_wsum_mean /= b3dmax!(1, self.m_num_wsum) as f32;
        if self.m_verbose != 0 {
            print_vars!(
                ("mNumWsum", CArg::Int(self.m_num_wsum as i64)),
                ("mWsumMean", CArg::Dbl(self.m_wsum_mean as f64)),
                ("mNumOuterMAD", CArg::Int(self.m_num_outer_mad as i64)),
                ("mOuterMADMean", CArg::Dbl(self.m_outer_mad_mean as f64))
            );
        }

        // Set up the domains; first get the density in number per square pixel
        // Also set up the lower target for overlapped beads if it was a density
        if if_number != 0 {
            target_density = target_number as f32 / total_area;
        } else {
            target_density = (target_density as f64 / 1.0e6) as f32;
            target_number = (target_density * total_area) as i32;
            if self.m_overlap_target > 0. {
                self.m_overlap_target =
                    (self.m_overlap_target as f64 * (total_area as f64 / 1.0e6)) as f32;
            }
        }
        target_spacing = (2. / (target_density as f64 * 3.0f64.sqrt())).sqrt() as f32;
        if self.m_overlap_target < 0. {
            self.m_overlap_target = target_number as f32;
        }

        // Base the domain sizes on the lower density if doing two surfaces, and grid
        // spacing on higher density
        surf_density = (target_density as f64 / (1. + two_surf as f64)) as f32;
        surf_number = (target_number + two_surf) / (1 + two_surf);
        surf_spacing = (2. / (surf_density as f64 * 3.0f64.sqrt())).sqrt() as f32;
        kernel_h = spacing_to_hfac * surf_spacing;
        grid_spacing = spacing_to_grid_fac * target_spacing;
        self.m_num_xgrid = b3dmax!(1., ((nx_image as f32 / grid_spacing).ceil()) as f64) as i32;
        self.m_del_xgrid = (nx_image as f64 / (self.m_num_xgrid as f64 + 1.)) as f32;
        self.m_num_ygrid = b3dmax!(1., ((ny_image as f32 / grid_spacing).ceil()) as f64) as i32;
        self.m_del_ygrid = (ny_image as f64 / (self.m_num_ygrid as f64 + 1.)) as f32;
        if self.m_verbose != 0 {
            print_vars!(
                ("kernelH", CArg::Dbl(kernel_h as f64)),
                ("gridSpacing", CArg::Dbl(grid_spacing as f64)),
                ("mNumXgrid", CArg::Int(self.m_num_xgrid as i64)),
                ("mNumYgrid", CArg::Int(self.m_num_ygrid as i64))
            );
        }
        if self.m_verbose != 0 {
            print_vars!(
                ("surfDensity * 1.e6", CArg::Dbl(surf_density as f64 * 1.0e6)),
                ("surfSpacing", CArg::Dbl(surf_spacing as f64)),
                ("surfNumber", CArg::Int(surf_number as i64))
            );
        }

        // Divide image area into domains and set up array of domains
        self.m_num_xdomains = b3dmax!(1., 0.4999 * nx_image as f64 / kernel_h as f64) as i32;
        self.m_num_ydomains = b3dmax!(1., 0.4999 * ny_image as f64 / kernel_h as f64) as i32;
        num_domains = self.m_num_xdomains * self.m_num_ydomains;
        self.m_del_xdomain = (nx_image as f32) / self.m_num_xdomains as f32;
        self.m_del_ydomain = (ny_image as f32) / self.m_num_ydomains as f32;
        if self.m_verbose != 0 {
            print_vars!(
                ("mNumXdomains", CArg::Int(self.m_num_xdomains as i64)),
                ("mNumYdomains", CArg::Int(self.m_num_ydomains as i64)),
                ("mDelXdomain", CArg::Dbl(self.m_del_xdomain as f64)),
                ("mDelXdomain", CArg::Dbl(self.m_del_xdomain as f64))
            );
        }
        self.m_grid_points =
            vec![GridPoint::default(); (self.m_num_xgrid * self.m_num_ygrid) as usize];
        self.m_domains = vec![GridDomain::default(); num_domains as usize];
        max_grid_per_dom = ((1.
            + ((self.m_del_xdomain as f64 + 1.) as i32) as f32 / self.m_del_xgrid)
            * (1. + ((self.m_del_ydomain as f64 + 1.) as i32) as f32 / self.m_del_ygrid))
            as i32;

        // Initialize domains and make list of neighbors to loop over
        for i in 0..num_domains {
            let nxd = self.m_num_xdomains;
            let nyd = self.m_num_ydomains;
            let domp = &mut self.m_domains[i as usize];
            domp.num_candidates = 0;
            domp.num_accepted = 0;
            domp.num_grid_pts = 0;
            domp.num_tracks = 0;
            domp.grid_pt_ind = vec![0; max_grid_per_dom.max(0) as usize];
            domp.num_neighbors = 0;
            ix = i % nxd;
            iy = i / nxd;
            for jx in -1..=1 {
                for jy in -1..=1 {
                    if ix + jx >= 0 && ix + jx < nxd && iy + jy >= 0 && iy + jy < nyd {
                        domp.neighbors[domp.num_neighbors as usize] = ix + jx + nxd * (iy + jy);
                        domp.num_neighbors += 1;
                    }
                }
            }
        }

        // Initialize grid points and make lists in domains
        for ix in 0..self.m_num_xgrid {
            xx = self.m_del_xgrid * (ix + 1) as f32;
            for iy in 0..self.m_num_ygrid {
                yy = self.m_del_ygrid * (iy + 1) as f32;
                let i = ix + iy * self.m_num_xgrid;
                self.m_grid_points[i as usize].x = xx;
                self.m_grid_points[i as usize].y = yy;
                self.m_grid_points[i as usize].domain = -1;
                ind = self.domain_index(xx, yy);
                if ind >= 0
                    && xx >= self.m_area_xmin
                    && xx <= self.m_area_xmax
                    && yy >= self.m_area_ymin
                    && yy <= self.m_area_ymax
                    && (self.m_num_area_cont == 0
                        || (if imod_point_inside_area(
                            &self.m_area_mod.as_ref().unwrap().obj[0],
                            &self.m_area_conts,
                            self.m_num_area_cont,
                            xx,
                            yy,
                        ) >= 0
                        {
                            0
                        } else {
                            1
                        }) == self.m_exclude_areas)
                {
                    let dom = &mut self.m_domains[ind as usize];
                    dom.grid_pt_ind[dom.num_grid_pts as usize] = i;
                    dom.num_grid_pts += 1;
                    self.m_grid_points[i as usize].domain = ind;
                }
            }
        }

        self.compute_area_fracs(kernel_h);

        // Get domains for the tracks and accumulate the number of tracks per domain
        max_track_len = 0;
        for co in 0..max_conts as usize {
            if self.m_tracks[co].model >= 0 {
                let cont = &self.m_track_mods[self.m_tracks[co].model as usize].obj[0].cont[co];
                max_track_len = b3dmax!(max_track_len, cont.pts.len() as i32);

                // Find nearest Z in track to middle Z
                // Deviation: a contour with no points leaves `imin` as the previous
                // track's value (uninitialised for the first) and the source then
                // reads `cont->pts[imin]` past an empty array; such a track is
                // given no domain here.
                if cont.pts.is_empty() {
                    continue;
                }
                dzmin = 10000.;
                for i in 0..cont.pts.len() {
                    dz = ((cont.pts[i].z - iz_middle as f32) as f64).abs() as f32;
                    if dz < dzmin {
                        dzmin = dz;
                        imin = i as i32;
                    }
                }
                let pt = cont.pts[imin as usize];
                ind = self.domain_index(pt.x, pt.y);
                if ind >= 0 {
                    self.m_tracks[co].domain = ind;
                    self.m_domains[ind as usize].num_tracks += 1;
                    self.m_tracks[co].mid_pos = pt;
                }
            }
        }

        // Make a list of the tracks by domain
        ncum = 0;
        for i in 0..num_domains as usize {
            self.m_domains[i].track_start_ind = ncum;
            ncum += self.m_domains[i].num_tracks;
            self.m_domains[i].num_tracks = 0;
        }
        for co in 0..max_conts as usize {
            ind = self.m_tracks[co].domain;
            if ind >= 0 {
                let dom = &mut self.m_domains[ind as usize];
                track_list[(dom.track_start_ind + dom.num_tracks) as usize] = co as i32;
                dom.num_tracks += 1;
            }
        }

        // Identify common tracks and build list of candidates
        for co in 0..max_conts {
            let cou = co as usize;
            ind = self.m_tracks[cou].domain;
            if ind < 0 || self.m_tracks[cou].cand_index >= 0 {
                continue;
            }
            candid.domain = ind;
            co_mod = self.m_tracks[cou].model;
            for i in 0..num_models as usize {
                candid.contours[i] = -1;
            }
            candid.contours[co_mod as usize] = co;
            self.m_tracks[cou].cand_index = self.m_candidates.len() as i32;
            candid.overlapped = 0;
            candid.overlap_tmp = 0;
            candid.clustered = 0;
            candid.accepted = 0;
            candid.top_bot = 0;
            candid.mean_deviation = 0.;
            top_sum = if self.m_tracks[cou].top_bot == 2 {
                1
            } else {
                0
            };
            bot_sum = if self.m_tracks[cou].top_bot == 1 {
                1
            } else {
                0
            };
            ncum = 1;
            best_res_for_pos = 1.0e30;
            if b3dnint!(self.m_tracks[cou].mid_pos.z) == iz_middle {
                best_res_for_pos = self.m_tracks[cou].residual;
                candid.pos = self.m_tracks[cou].mid_pos;
                cont_for_track_pos = co;
            }

            // Loop on all tracks in neighborhood
            let mut neigh = 0;
            while neigh < self.m_domains[ind as usize].num_neighbors && ncum < num_models {
                idom = self.m_domains[ind as usize].neighbors[neigh as usize];
                let mut i = 0;
                while i < self.m_domains[idom as usize].num_tracks && ncum < num_models {
                    neigh_co =
                        track_list[(self.m_domains[idom as usize].track_start_ind + i) as usize];
                    neigh_mod = self.m_tracks[neigh_co as usize].model;
                    if neigh_mod == co_mod
                        || self.m_tracks[neigh_co as usize].cand_index >= 0
                        || candid.contours[neigh_mod as usize] >= 0
                        || self.m_tracks[neigh_co as usize].domain < 0
                    {
                        i += 1;
                        continue;
                    }
                    self.get_track_deviation(
                        co,
                        co_mod,
                        neigh_co,
                        neigh_mod,
                        &mut mean_dev,
                        &mut frac_close,
                        None,
                    );

                    // If it matches, add it to the candidate and mark it with index; save
                    // deviation
                    if frac_close > crit_frac_close {
                        let nco = neigh_co as usize;
                        candid.contours[neigh_mod as usize] = neigh_co;
                        self.m_tracks[nco].cand_index = self.m_candidates.len() as i32;
                        top_sum += if self.m_tracks[nco].top_bot == 2 {
                            1
                        } else {
                            0
                        };
                        bot_sum += if self.m_tracks[nco].top_bot == 1 {
                            1
                        } else {
                            0
                        };
                        ncum += 1;
                        if b3dnint!(self.m_tracks[nco].mid_pos.z) == iz_middle
                            && best_res_for_pos > self.m_tracks[nco].residual
                        {
                            best_res_for_pos = self.m_tracks[nco].residual;
                            candid.pos = self.m_tracks[nco].mid_pos;
                            cont_for_track_pos = neigh_co;
                        }
                    }
                    i += 1;
                }
                neigh += 1;
            }

            // Get mean deviation between all pairs if there is more than one track
            if ncum > 1 {
                for i in 0..num_models - 1 {
                    for j in i + 1..num_models {
                        if candid.contours[i as usize] >= 0 && candid.contours[j as usize] >= 0 {
                            self.get_track_deviation(
                                candid.contours[i as usize],
                                i,
                                candid.contours[j as usize],
                                j,
                                &mut mean_dev,
                                &mut frac_close,
                                None,
                            );
                            candid.mean_deviation += mean_dev;
                        }
                    }
                }
                candid.mean_deviation /= (ncum * (ncum - 1) / 2) as f32;
            }

            if two_surf != 0 && top_sum > bot_sum {
                candid.top_bot = 1;
            }

            // Get the true position on the middle section or nearest if necessary if it
            // wasn't set from the contour with lowest residual
            if best_res_for_pos > 1.0e29 {
                let mut i = 0;
                while i < num_models {
                    j = candid.contours[((co + i) % num_models) as usize];
                    if j >= 0 && b3dnint!(self.m_tracks[j as usize].mid_pos.z) == iz_middle {
                        candid.pos = self.m_tracks[j as usize].mid_pos;
                        cont_for_track_pos = j;
                        break;
                    }
                    i += 1;
                }
                if i >= num_models {
                    candid.pos = self.m_tracks[cou].mid_pos;
                    cont_for_track_pos = co;
                }
            }

            // Check for bad tracking from izMiddle: see if one of the contours was from
            // there
            if number_in_list(iz_middle, Some(&model_seed_zvalues), num_seed_zvalues, 0) != 0 {
                bad_at_middle = true;
                for i in 0..num_models as usize {
                    j = candid.contours[i];
                    if j >= 0
                        && model_seed_zvalues[self.m_tracks[j as usize].model as usize] == iz_middle
                    {
                        bad_at_middle = false;
                    }
                }

                // If not, find and use the model seed point itself from the track used to
                // set the position
                if bad_at_middle {
                    let model = self.m_tracks[cont_for_track_pos as usize].model;
                    let cont =
                        &self.m_track_mods[model as usize].obj[0].cont[cont_for_track_pos as usize];
                    imin = -1;
                    for i in 0..cont.pts.len() {
                        dz = ((cont.pts[i].z - model_seed_zvalues[model as usize] as f32) as f64)
                            .abs() as f32;
                        if dz < 0.5 {
                            imin = i as i32;
                        }
                    }

                    // If that worked, take it; if not, get it to be dropped in next check
                    if imin >= 0 {
                        candid.pos = cont.pts[imin as usize];
                    } else {
                        candid.pos.x = (x_border as f64 - 1000.) as f32;
                    }
                }
            }

            // Now check if it is inside the borders and abort the whole set of tracks if
            // out
            if candid.pos.x < x_border as f32
                || candid.pos.x >= (nx_image - x_border) as f32
                || candid.pos.y < y_border as f32
                || candid.pos.y >= (ny_image - y_border) as f32
            {
                for i in 0..num_models as usize {
                    if candid.contours[i] >= 0 {
                        self.m_tracks[candid.contours[i] as usize].cand_index = -1;
                        self.m_tracks[candid.contours[i] as usize].domain = -1;
                    }
                }
            } else {
                // Or add the candidate for real
                self.m_candidates.push(candid);
                self.m_domains[ind as usize].num_candidates += 1;
            }
        }

        // Make the candidate list organized by domain
        self.m_num_candidates = self.m_candidates.len() as i32;
        self.m_candid_list = vec![0; self.m_num_candidates as usize];
        self.m_accept_list = vec![0; self.m_num_candidates as usize];
        self.m_rank_index = vec![0; self.m_num_candidates as usize];

        ncum = 0;
        for i in 0..num_domains as usize {
            self.m_domains[i].cand_start_ind = ncum;
            ncum += self.m_domains[i].num_candidates;
            self.m_domains[i].num_candidates = 0;
        }

        for i in 0..self.m_num_candidates as usize {
            ind = self.m_candidates[i].domain;
            if ind >= 0 {
                let dom = &mut self.m_domains[ind as usize];
                self.m_candid_list[(dom.cand_start_ind + dom.num_candidates) as usize] = i as i32;
                dom.num_candidates += 1;
            }
        }

        // Identify elongated points first using all points
        self.analyze_elongation(max_conts, 0, &mut edge_tmp);

        // Identify clustered points as ones with near neighbors by looking at all pairs
        self.m_cos_rot = (PI * rotation as f64 / 180.).cos() as f32;
        self.m_sin_rot = (PI * rotation as f64 / 180.).sin() as f32;
        self.m_cos_tilt = (PI * self.m_highest_tilt as f64 / 180.).cos() as f32;

        for co in 0..self.m_num_candidates as usize {
            ind = self.m_candidates[co].domain;
            for neigh in 0..self.m_domains[ind as usize].num_neighbors as usize {
                idom = self.m_domains[ind as usize].neighbors[neigh];
                for i in 0..self.m_domains[idom as usize].num_candidates {
                    neigh_co = self.m_candid_list
                        [(self.m_domains[idom as usize].cand_start_ind + i) as usize];
                    if neigh_co as usize != co
                        && self.m_candidates[co].top_bot
                            == self.m_candidates[neigh_co as usize].top_bot
                    {
                        if self.beads_are_clustered(
                            &self.m_candidates[co],
                            &self.m_candidates[neigh_co as usize],
                        ) {
                            self.m_candidates[co].clustered = 1;
                            self.m_candidates[neigh_co as usize].clustered = 1;
                        }
                    }
                }
            }
        }

        // Analyze for elongated points again after excluding all the clustered ones which
        // could skew the distribution
        self.analyze_elongation(max_conts, 1, &mut edge_tmp);

        // Report the results
        num_clust = 0;
        num_over = 0;
        for co in 0..self.m_num_candidates as usize {
            let candp = &self.m_candidates[co];
            if candp.overlapped != 0 {
                num_over += 1;
            } else if candp.clustered != 0 {
                num_clust += 1;
            }
            if self.m_verbose != 0 {
                printf!(
                    "%4d %4d %4d %4d  at %4.0f %4.0f  overlap %d clustered %d\n",
                    CArg::Int(co as i64 + 1),
                    CArg::Int(candp.contours[0] as i64 + 1),
                    CArg::Int(candp.contours[1] as i64 + 1),
                    CArg::Int(candp.contours[2] as i64 + 1),
                    CArg::Dbl(candp.pos.x as f64 + 1.),
                    CArg::Dbl(candp.pos.y as f64 + 1.),
                    CArg::Int(candp.overlapped as i64),
                    CArg::Int(candp.clustered as i64)
                );
            }
        }
        printf!(
            "%d candidate points, including %d clustered and %d elongated   [PBS1]\n",
            CArg::Int(self.m_num_candidates as i64),
            CArg::Int(num_clust as i64),
            CArg::Int(num_over as i64)
        );
        if self.m_num_candidates == 0 {
            exit_error(b"No candidate points have been identified");
        }

        // Score the candidates, put scores into separate array for ranking
        // A low score is good
        for ind in 0..self.m_num_candidates as usize {
            ncum = 0;
            comp_sum = 0.;
            res_sum = 0.;
            bot_sum = 0;
            top_sum = 0;
            for i in 0..num_models as usize {
                co = self.m_candidates[ind].contours[i];
                if co >= 0 {
                    let track = &self.m_tracks[co as usize];
                    ncum += 1;
                    comp_sum = (comp_sum as f64
                        + (1.
                            - ((self.m_track_mods[track.model as usize].obj[0].cont[co as usize]
                                .pts
                                .len() as f32)
                                / max_track_len as f32) as f64))
                        as f32;
                    res_sum += track.residual;
                    top_sum += if track.top_bot == 2 { 1 } else { 0 };
                    bot_sum += if track.top_bot == 1 { 1 } else { 0 };
                }
            }
            ndev = ncum * (ncum - 1) / 2;
            let candp = &mut self.m_candidates[ind];
            let numerator = weights[WGT_COMPLETE] * comp_sum / ncum as f32
                + weights[WGT_RESIDUAL] * res_sum / ncum as f32
                + weights[WGT_NUM_MODELS] * (num_models - ncum) as f32
                + weights[WGT_DEVIATION] * candp.mean_deviation;
            let denominator =
                (weights[WGT_COMPLETE] + weights[WGT_RESIDUAL] + weights[WGT_NUM_MODELS]) as f64
                    + (if ndev != 0 {
                        weights[WGT_DEVIATION] as f64
                    } else {
                        0.
                    });
            edge_tmp[ind] = (numerator as f64 / denominator) as f32;
            candp.score = edge_tmp[ind];
            self.m_rank_index[ind] = ind as i32;
            candp.overlapped = ((100 * candp.overlapped as i32) / ncum) as u8;
            if self.m_verbose != 0 {
                printf!(
                    "%d  tb %d %d -> %d  inc %.3f  res %.3f  miss %d  dev %.3f  score %.6f\n",
                    CArg::Int(ind as i64),
                    CArg::Int(bot_sum as i64),
                    CArg::Int(top_sum as i64),
                    CArg::Int(candp.top_bot as i64),
                    CArg::Dbl((comp_sum / ncum as f32) as f64),
                    CArg::Dbl((res_sum / ncum as f32) as f64),
                    CArg::Int((num_models - ncum) as i64),
                    CArg::Dbl(candp.mean_deviation as f64),
                    CArg::Dbl(candp.score as f64)
                );
            }
            all_scores.push(candp.score);
        }

        // Get the median and MADN score and the criterion for outliers to be skipped
        score_temp = vec![0.; all_scores.len()];
        let n = all_scores.len() as i32;
        rs_fast_median_in_place(&mut all_scores, n, &mut score_median);
        rs_madn(
            &all_scores,
            n,
            score_median,
            &mut score_temp,
            &mut score_madn,
        );
        self.m_outlier_score = score_median + max_madns_for_score * score_madn;
        ind = 0;
        ob = 0;
        for i in 0..all_scores.len() {
            if self.m_candidates[i].score >= self.m_outlier_score {
                ind += 1;
                if self.m_candidates[i].clustered != 0 || self.m_candidates[i].overlapped != 0 {
                    ob += 1;
                }
            }
        }
        if self.m_verbose != 0 {
            printf!(
                "Score median %.4f  MADN %.4f,  max allowed %.4f\n",
                CArg::Dbl(score_median as f64),
                CArg::Dbl(score_madn as f64),
                CArg::Dbl(self.m_outlier_score as f64)
            );
        }
        if ind != 0 {
            printf!(
                "%d candidates excluded by outlier scores (%d clustered or elongated)\n",
                CArg::Int(ind as i64),
                CArg::Int(ob as i64)
            );
        }

        rs_sort_indexed_floats(&edge_tmp, &mut self.m_rank_index, self.m_num_candidates);

        for ind in 0..3 {
            self.m_num_accepted[ind] = 0;
        }

        // If appending, go through each existing point and find it in candidate list and
        // accept the candidate
        if self.m_append_to_seed != 0 {
            let base = base_mod.as_ref().unwrap();
            for ob in 0..base.obj.len() {
                for co in 0..base.obj[ob].cont.len() {
                    let cont = &base.obj[ob].cont[co];
                    if cont.pts.is_empty() {
                        continue;
                    }
                    let i = cont.pts.len() / 2;
                    ind = self.domain_index(cont.pts[i].x, cont.pts[i].y);
                    if ind < 0 {
                        continue;
                    }
                    look_more = true;
                    let mut neigh = 0;
                    while neigh < self.m_domains[ind as usize].num_neighbors && look_more {
                        idom = self.m_domains[ind as usize].neighbors[neigh as usize];
                        let mut i = 0;
                        while i < self.m_domains[idom as usize].num_candidates && look_more {
                            neigh_co = self.m_candid_list
                                [(self.m_domains[idom as usize].cand_start_ind + i) as usize];
                            if self.m_candidates[neigh_co as usize].accepted != 0 {
                                i += 1;
                                continue;
                            }
                            let mut j = 0;
                            while j < num_models && look_more {
                                let cj = self.m_candidates[neigh_co as usize].contours[j as usize];
                                if cj >= 0 {
                                    self.get_track_deviation(
                                        cj,
                                        j,
                                        0,
                                        0,
                                        &mut mean_dev,
                                        &mut frac_close,
                                        Some(cont),
                                    );
                                    if frac_close > 0. {
                                        look_more = false;
                                        let tb = self.m_candidates[neigh_co as usize].top_bot;
                                        self.accept_candidate(neigh_co, tb);
                                    }
                                }
                                j += 1;
                            }
                            i += 1;
                        }
                        neigh += 1;
                    }
                }
            }
            self.m_phase += 1;
        }

        // Phase 1:
        // For each surface, go through points in order by ranking and and accept them
        // if they are on that surface, not in cluster, and not too close to existing point
        num_base_cand = self.m_num_accepted[2];
        for top_bot in 0..=two_surf {
            self.add_best_spaced_points(
                top_bot,
                surf_number,
                self.m_num_candidates,
                init_add_spacing_fac * surf_spacing,
            );
            self.add_best_spaced_points(
                top_bot,
                surf_number,
                self.m_num_candidates / 2,
                half_add_spacing_fac * surf_spacing,
            );
        }
        self.output_num_accepted(two_surf, 0);
        self.m_phase += 1;

        // Phase 2:
        // Now go through the gap filling routine for each surface separately
        // In this phase, also allow a higher density threshold for termination
        use_dens2 = 0;
        if two_surf != 0 {
            use_dens2 = -1;
            for top_bot in 0..=two_surf {
                if (self.m_num_accepted[top_bot as usize] as f32)
                    < surf_ratio_use_dens2 * self.m_num_accepted[(1 - top_bot) as usize] as f32
                {
                    use_dens2 = top_bot;
                }
            }
        }
        for top_bot in 0..=two_surf {
            term_dens = surf_density * min_density_fac1;
            if top_bot == use_dens2 {
                term_dens = surf_density * min_density_fac2;
            }
            self.compute_densities(top_bot, kernel_h);
            self.add_points_in_gaps(
                top_bot,
                surf_number,
                term_dens,
                surf_spacing,
                0.,
                0,
                kernel_h,
                2,
            );
            let num_rings = self.m_num_rings;
            self.add_points_in_gaps(
                top_bot,
                surf_number,
                term_dens,
                surf_spacing,
                0.,
                0,
                kernel_h,
                num_rings,
            );
        }
        self.output_num_accepted(two_surf, 0);
        self.m_phase += 1;

        // Phase 3:
        // If two surfaces, now try to beef up the majority surface to make up for
        // deficiency in the minority.
        // Stick with the lower density factor for termination
        major_density = surf_density;
        if two_surf != 0 && self.m_num_accepted[2] < target_number {
            top_bot = if self.m_num_accepted[1] > self.m_num_accepted[0] {
                1
            } else {
                0
            };

            // Revise target number to make up the difference, and do a round of initial
            // adding at that number, then do the gap filling but base the density analysis
            // on the full set of points in order to fill in gaps of the other surface
            major_number = if no_beef_up != 0 {
                surf_number
            } else {
                target_number - self.m_num_accepted[(1 - top_bot) as usize]
            };
            major_density = major_number as f32 / total_area;
            major_spacing = (2. / (major_density as f64 * 3.0f64.sqrt())).sqrt() as f32;
            self.add_best_spaced_points(
                top_bot,
                major_number,
                self.m_num_candidates,
                init_add_spacing_fac * major_spacing,
            );
            major_kernel_h = target_spacing;
            self.compute_area_fracs(major_kernel_h);
            self.compute_densities(2, major_kernel_h);
            self.add_points_in_gaps(
                top_bot,
                major_number,
                target_density * min_density_fac1,
                target_spacing,
                0.,
                0,
                major_kernel_h,
                2,
            );
            let num_rings = self.m_num_rings;
            self.add_points_in_gaps(
                top_bot,
                major_number,
                target_density * min_density_fac1,
                target_spacing,
                0.,
                0,
                major_kernel_h,
                num_rings,
            );
            self.output_num_accepted(two_surf, 0);
        }
        self.m_phase += 1;

        // Phase 4 [5, 6, 7, 8]:
        // Now we are going to use a higher density factor, then
        // add in clustered points, and overlapped points if selected
        num_phase = use_clusters + use_overlaps + 1;
        if self.m_verbose != 0 {
            print_vars!(
                ("numPhase", CArg::Int(num_phase as i64)),
                ("useClusters", CArg::Int(use_clusters as i64)),
                ("useOverlaps", CArg::Int(use_overlaps as i64))
            );
        }
        let mut phase = 0;
        while phase < num_phase && self.m_num_accepted[2] < target_number {
            if phase == 1 {
                self.m_outlier_score = score_median + max_clust_elong_score_madns * score_madn;
            }

            // Do the minority surface first
            top_bot = 0;
            if two_surf != 0 {
                top_bot = if self.m_num_accepted[1] < self.m_num_accepted[0] {
                    1
                } else {
                    0
                };
            }
            overlap_thresh = if phase != 0 {
                ((phase - use_clusters) * 100) / 3
            } else {
                0
            };
            cluster_thresh = if phase != 0 && use_clusters != 0 {
                major_density
            } else {
                0.
            };
            if self.m_verbose != 0 {
                print_vars!(
                    ("phase", CArg::Int(phase as i64)),
                    ("clusterThresh", CArg::Dbl(cluster_thresh as f64)),
                    ("overlapThresh", CArg::Int(overlap_thresh as i64))
                );
            }
            let mut surf = 0;
            while surf < two_surf + 1 && self.m_num_accepted[2] < target_number {
                let num_rings = self.m_num_rings;
                if surf != 0 {
                    // Revise targets every time when doing majority surface
                    major_number = if no_beef_up != 0 {
                        surf_number
                    } else {
                        target_number - self.m_num_accepted[(1 - top_bot) as usize]
                    };
                    major_density = major_number as f32 / total_area;
                    major_spacing = (2. / (major_density as f64 * 3.0f64.sqrt())).sqrt() as f32;
                    major_kernel_h = spacing_to_hfac * major_spacing;
                    self.compute_area_fracs(major_kernel_h);
                    self.compute_densities(top_bot, major_kernel_h);
                    self.add_points_in_gaps(
                        top_bot,
                        major_number,
                        major_density * min_density_fac2,
                        major_spacing,
                        cluster_thresh,
                        overlap_thresh,
                        major_kernel_h,
                        num_rings,
                    );
                } else {
                    // Return to original targets for minority surface
                    self.compute_area_fracs(kernel_h);
                    self.compute_densities(top_bot, kernel_h);
                    self.add_points_in_gaps(
                        top_bot,
                        surf_number,
                        surf_density * min_density_fac2,
                        surf_spacing,
                        cluster_thresh,
                        overlap_thresh,
                        kernel_h,
                        num_rings,
                    );
                }
                top_bot = 1 - top_bot;
                surf += 1;
            }
            self.output_num_accepted(two_surf, 0);
            self.m_phase += 1;
            phase += 1;
        }

        self.output_num_accepted(two_surf, 1);
        if self.m_num_accepted[2] == 0 {
            exit_error_fmt!(
                "No candidate points were acceptable.%s%s%s%s%s",
                CArg::Str(if num_over != 0 || num_clust != 0 {
                    "  Consider allowing"
                } else {
                    ""
                }),
                CArg::Str(if num_clust != 0 { " clustered " } else { "" }),
                CArg::Str(if num_clust != 0 && num_over != 0 {
                    "and/or"
                } else {
                    ""
                }),
                CArg::Str(if num_over != 0 { " elongated " } else { "" }),
                CArg::Str(if num_over != 0 || num_clust != 0 {
                    "points."
                } else {
                    ""
                })
            );
        }

        // Compose the model
        valmin = 1.0e20;
        valmax = -valmin;
        let mut imod: Imod;
        let mid_model = (num_models / 2) as usize;
        if self.m_append_to_seed != 0 {
            imod = base_mod.take().unwrap();
            let obj = &mut imod.obj[0];
            co = obj.cont.len() as i32;
            if self.m_num_accepted[2] > num_base_cand {
                ind = co + self.m_num_accepted[2] - num_base_cand;
                let Some(mut cont) = imod_contours_new(ind) else {
                    exit_error(b"Allocating new array of contours")
                };
                for i in 0..co as usize {
                    cont[i] = obj.cont[i].clone();
                }
                obj.cont = cont;
            }
            if istore_get_min_max(
                &obj.store,
                obj.cont.len() as i32,
                GEN_STORE_MINMAX1,
                &mut xx,
                &mut yy,
            ) != 0
            {
                valmin = xx;
                valmax = yy;
            }
        } else {
            // clear out object of one model if not appending
            imod = std::mem::take(&mut self.m_track_mods[mid_model]);
            let num = self.m_num_accepted[2];
            self.clear_model_allocate_conts(&mut imod, num);
            co = 0;
        }

        // Add candidates as contours, with scores
        store.type_ = GEN_STORE_VALUE1;
        store.flags = GEN_STORE_FLOAT << 2;
        {
            let obj = &mut imod.obj[0];
            for idom in 0..num_domains as usize {
                for i in 0..self.m_domains[idom].num_accepted {
                    ind = self.m_accept_list[(self.m_domains[idom].cand_start_ind + i) as usize];
                    let candp = &self.m_candidates[ind as usize];

                    // Skip ones that were accepted because they matched ones in base model
                    if self.m_append_to_seed != 0 && candp.accepted == 1 {
                        continue;
                    }
                    if imod_point_append(&mut obj.cont[co as usize], candp.pos) == 0 {
                        exit_error(b"Adding point to new model");
                    }

                    // Need to take inverse of the score so that worse ones are "below
                    // threshold".  This seems to spread them out as well as taking a log
                    // does, too
                    store.value.set_f(
                        (1. / ((candp.score + candp.clustered as f32) as f64
                            + candp.overlapped as f64 / 100.)) as f32,
                    );
                    store.index.set_i(co);
                    if istore_insert(&mut obj.store, store) != 0 {
                        exit_error(b"Could not add general storage item");
                    }
                    valmin = b3dmin!(valmin, store.value.f());
                    valmax = b3dmax!(valmax, store.value.f());
                    if phase_as_surf != 0 {
                        obj.cont[co as usize].surf = (two_surf + 1)
                            * (candp.accepted as i32 - 1 - self.m_append_to_seed)
                            + two_surf * candp.top_bot;
                    } else if two_surf != 0 {
                        obj.cont[co as usize].surf = candp.top_bot;
                    }
                    co += 1;
                }
            }
            if istore_add_min_max(&mut obj.store, GEN_STORE_MINMAX1, valmin, valmax) != 0 {
                exit_error(b"Could not add general storage item");
            }

            // Add colors for the surfaces
            ncum = two_surf;
            if phase_as_surf != 0 {
                ncum = (two_surf + 1) * self.m_phase + two_surf - 1;
            }
            obj.surfsize = ncum;
            if self.m_append_to_seed == 0 {
                self.add_surface_colors(obj, ncum, &rgba, MAX_COLORS as i32);
            }

            // Turn off low flag set originally by imodfindbeads and high one just in case
            obj.matflags2 &= !((MATFLAGS2_SKIP_LOW | MATFLAGS2_SKIP_HIGH) as u8);
        }

        if imod_backup_file(&out_name) != 0 {
            printf!(
                "WARNING: pickbestseed - Could not rename existing output file to %s~",
                CArg::Str(&out_name)
            );
        }
        let Some(mut fp) = ImodFile::open(&out_name, "wb") else {
            exit_error(b"Opening file for output model")
        };
        let _ = imod_write(&imod, &mut fp);
        drop(fp);
        if self.m_append_to_seed == 0 {
            self.m_track_mods[mid_model] = imod;
        }

        if let Some(elong_name) = elong_name {
            for i in 0..8 {
                num_in_surf[i] = 0;
                num_inside[i] = 0;
                num_outside[i] = 0;
            }
            tot_inside = 0;
            tot_outside = 0;
            imod_backup_file(&elong_name);
            let mut imod = std::mem::take(&mut self.m_track_mods[mid_model]);
            imod.cindex.object = 0;
            imod.cindex.contour = 0;
            imod.cindex.point = 0;
            let num = self.m_num_candidates;
            self.clear_model_allocate_conts(&mut imod, num);
            for co in 0..self.m_num_candidates as usize {
                let candp = &self.m_candidates[co];
                let obj = &mut imod.obj[0];
                if imod_point_append(&mut obj.cont[co], candp.pos) == 0 {
                    exit_error(b"Adding point to new model");
                }
                obj.cont[co].surf =
                    4 * candp.clustered as i32 + (candp.overlapped as f64 / 33.4).ceil() as i32;
                let surf = obj.cont[co].surf as usize;
                num_in_surf[surf] += 1;

                // Make current contour the first elongated non clustered one
                if candp.overlapped != 0 && candp.clustered == 0 && imod.cindex.contour == 0 {
                    imod.cindex.contour = co as i32;
                }
                if num_cont_for_count != 0 {
                    ind = imod_point_inside_area(
                        &self.m_area_mod.as_ref().unwrap().obj[0],
                        &self.m_area_conts,
                        num_cont_for_count,
                        candp.pos.x,
                        candp.pos.y,
                    );
                    if ind >= 0 {
                        num_inside[surf] += 1;
                        if candp.overlapped != 0 {
                            tot_inside += 1;
                        }
                    } else {
                        num_outside[surf] += 1;
                        if candp.overlapped != 0 {
                            tot_outside += 1;
                        }
                    }
                }
            }
            let obj = &mut imod.obj[0];
            obj.surfsize = 7;
            self.add_surface_colors(obj, 7, &rgba2, MAX_COLORS2 as i32);
            obj.matflags2 &= !((MATFLAGS2_SKIP_LOW | MATFLAGS2_SKIP_HIGH) as u8);
            obj.red = 0.1;
            obj.green = 0.5;
            obj.blue = 0.1;
            obj.linewidth2 = 2;

            // Add labels so user can see what is what
            if obj.label.is_none() {
                obj.label = Some(imod_label_new());
            }
            let label = obj.label.as_mut().unwrap();
            imod_label_item_add(label, Some(b"Not clust. or elong."), 0);
            imod_label_item_add(label, Some(b"Elongated 1"), 1);
            imod_label_item_add(label, Some(b"Elongated 2"), 2);
            imod_label_item_add(label, Some(b"Elongated 3"), 3);
            imod_label_item_add(label, Some(b"Clustered"), 4);
            imod_label_item_add(label, Some(b"Clust., Elong. 1"), 5);
            imod_label_item_add(label, Some(b"Clust., Elong. 2"), 6);
            imod_label_item_add(label, Some(b"Clust., Elong. 3"), 7);

            let Some(mut fp) = ImodFile::open(&elong_name, "wb") else {
                exit_error(b"Opening file for output model")
            };
            let _ = imod_write(&imod, &mut fp);
            drop(fp);

            printf!("Number in candidate model surfaces:");
            for i in 0..8 {
                printf!("   %d", CArg::Int(num_in_surf[i] as i64));
            }
            printf!("\n");
            if num_cont_for_count != 0 {
                printf!("    Inside/Outside:");
                for i in 0..8 {
                    printf!(
                        " %d/%d",
                        CArg::Int(num_inside[i] as i64),
                        CArg::Int(num_outside[i] as i64)
                    );
                }
                printf!(
                    "    Total elongated: %d + %d = %d\n",
                    CArg::Int(tot_inside as i64),
                    CArg::Int(tot_outside as i64),
                    CArg::Int((tot_inside + tot_outside) as i64)
                );
            }
        }
        exit(0);
    }

    /// Original: `PickSeeds::clearModelAllocateConts` (`pickbestseed.cpp:1117`).
    fn clear_model_allocate_conts<'a>(&self, imod: &'a mut Imod, num_cont: i32) -> &'a mut Iobj {
        let obj = &mut imod.obj[0];
        obj.cont.clear();
        let Some(cont) = imod_contours_new(num_cont) else {
            exit_error(b"Allocating new array of contours")
        };
        obj.cont = cont;
        obj.store.clear();
        obj
    }

    /// Original: `PickSeeds::addSurfaceColors` (`pickbestseed.cpp:1131`).
    fn add_surface_colors(&self, obj: &mut Iobj, ncum: i32, rgba: &[[u8; 4]], max_colors: i32) {
        let mut store = Istore::default();
        store.type_ = GEN_STORE_COLOR;
        store.flags = (GEN_STORE_BYTE << 2) | GEN_STORE_SURFACE;
        for i in 0..ncum {
            store.index.set_i(i + 1);
            let ind = (i % max_colors) as usize;
            for j in 0..4 {
                store.value.bytes[j] = rgba[ind][j];
            }
            if istore_insert(&mut obj.store, store) != 0 {
                exit_error(b"Could not add general storage item");
            }
        }
    }

    /// Original: `PickSeeds::addBestSpacedPoints` (`pickbestseed.cpp:1152`).
    ///
    /// Go through points in order of increasing score and add them to the given
    /// side as long as their distance from other points on that side is bigger
    /// than minSpacing.
    fn add_best_spaced_points(
        &mut self,
        top_bot: i32,
        targ_num: i32,
        num_candidates: i32,
        min_spacing: f32,
    ) {
        for j in 0..num_candidates as usize {
            let cand_ind = self.m_rank_index[j];
            let candp = self.m_candidates[cand_ind as usize];
            if candp.overlapped != 0
                || candp.clustered != 0
                || candp.top_bot != top_bot
                || candp.score >= self.m_outlier_score
            {
                continue;
            }
            let mut too_close = false;
            let ind = candp.domain as usize;
            let mut neigh = 0;
            while neigh < self.m_domains[ind].num_neighbors && !too_close {
                let idom = self.m_domains[ind].neighbors[neigh as usize] as usize;
                for i in 0..self.m_domains[idom].num_accepted {
                    let co = self.m_accept_list[(self.m_domains[idom].cand_start_ind + i) as usize];
                    if self.m_candidates[co as usize].top_bot == top_bot
                        && imod_point_distance(&self.m_candidates[co as usize].pos, &candp.pos)
                            < min_spacing
                    {
                        too_close = true;
                        break;
                    }
                }
                neigh += 1;
            }
            if !too_close {
                if self.m_verbose > 1 {
                    printf!(
                        "Accepting ranked %d  candidate %d at %.0f %.0f\n",
                        CArg::Int(j as i64),
                        CArg::Int(cand_ind as i64),
                        CArg::Dbl(candp.pos.x as f64),
                        CArg::Dbl(candp.pos.y as f64)
                    );
                }
                self.accept_candidate(cand_ind, top_bot);
                if self.m_num_accepted[top_bot as usize] >= targ_num {
                    break;
                }
            }
        }
    }

    /// Original: `PickSeeds::addPointsInGaps` (`pickbestseed.cpp:1188`).
    ///
    /// Look around all the places where density is lowest for points to add.
    ///
    /// Fixed in translation (`BUGS.md`, pickbestseed): the search for the
    /// nearest accepted point of each ring candidate read
    /// `mDomains[neigh]` -- the loop counter `0..numNeighbors` used as a
    /// domain index -- where the neighbour domain is
    /// `mDomains[idom].neighbors[neigh]`, and assigned `tooClose` from each
    /// comparison, so only the last accepted point decided it.  Here the
    /// candidate's neighbouring domains are searched and any clustered
    /// accepted point makes it too close.
    #[allow(clippy::too_many_arguments)]
    fn add_points_in_gaps(
        &mut self,
        top_bot: i32,
        targ_num: i32,
        term_dens: f32,
        targ_spacing: f32,
        cluster_thresh: f32,
        overlap_thresh: i32,
        h: f32,
        num_rings: i32,
    ) {
        let mut cand_in_ring: Vec<i32> = Vec::new();
        let mut ring_num: Vec<i32> = Vec::new();
        let mut ind_min: i32;
        let mut dens_min: f32;
        let mut exclude_dist: f32;
        let mut exclude_sq: f32;
        let mut dx: f32;
        let mut dy: f32;
        let mut dist: f32;
        let mut min_score: f32;
        let mut score: f32;
        let mut dist_min: f32;
        let mut too_close: bool;

        for i in 0..(self.m_num_xgrid * self.m_num_ygrid) as usize {
            self.m_grid_points[i].exclude = 0;
        }
        self.output_densities(top_bot);

        // Loop until desired number is achieved
        while self.m_num_accepted[top_bot as usize] < targ_num {
            // Find lowest density non-excluded spot
            ind_min = -1;
            dens_min = 1.0e20;
            cand_in_ring.clear();
            ring_num.clear();
            for i in 0..(self.m_num_xgrid * self.m_num_ygrid) as usize {
                let gp = &self.m_grid_points[i];
                if gp.domain >= 0 && gp.exclude == 0 && gp.density < dens_min {
                    dens_min = gp.density;
                    ind_min = i as i32;
                }
            }

            // Break out of loop if none found
            if ind_min < 0 || dens_min > term_dens {
                if self.m_verbose > 1 && ind_min < 0 {
                    printf!(
                        "tb %d - all areas excluded in density search\n",
                        CArg::Int(top_bot as i64)
                    );
                } else if self.m_verbose > 1 {
                    printf!(
                        "tb %d - min density %.2f above limit %.2f\n",
                        CArg::Int(top_bot as i64),
                        CArg::Dbl(1.0e6 * dens_min as f64),
                        CArg::Dbl(1.0e6 * term_dens as f64)
                    );
                }
                break;
            }

            // Exclude all grid points in the neighborhood from future searches
            let grid_xcen = self.m_grid_points[ind_min as usize].x;
            let grid_ycen = self.m_grid_points[ind_min as usize].y;
            exclude_dist = self.m_exclude_fac * targ_spacing;
            exclude_sq = exclude_dist * exclude_dist;
            let ixdel = (exclude_dist / self.m_del_xgrid).ceil() as i32;
            let iydel = (exclude_dist / self.m_del_ygrid).ceil() as i32;
            let ixmid = ind_min % self.m_num_xgrid;
            let iymid = ind_min / self.m_num_xgrid;
            let mut ixlo = b3dmax!(0, ixmid - ixdel);
            let mut ixhi = b3dmin!(self.m_num_xgrid - 1, ixmid + ixdel);
            let mut iylo = b3dmax!(0, iymid - iydel);
            let mut iyhi = b3dmin!(self.m_num_ygrid - 1, iymid + iydel);
            for iy in iylo..=iyhi {
                for ix in ixlo..=ixhi {
                    let gpt = (ix + iy * self.m_num_xgrid) as usize;
                    if self.m_grid_points[gpt].domain >= 0 {
                        dx = grid_xcen - self.m_grid_points[gpt].x;
                        dy = grid_ycen - self.m_grid_points[gpt].y;
                        dist = dx * dx + dy * dy;
                        if dist < exclude_sq {
                            self.m_grid_points[gpt].exclude = 1;
                        }
                    }
                }
            }

            // Look for points in rings: first define a range of domains to loop in
            let delr = targ_spacing * self.m_ring_spacing_fac;
            exclude_dist = num_rings as f32 * delr;
            exclude_sq = exclude_dist * exclude_dist;
            ixlo = ((grid_xcen - exclude_dist) / self.m_del_xdomain) as i32;
            ixhi = ((grid_xcen + exclude_dist) / self.m_del_xdomain) as i32;
            ixlo = b3dmax!(0, ixlo);
            ixhi = b3dmin!(self.m_num_xdomains - 1, ixhi);
            iylo = ((grid_ycen - exclude_dist) / self.m_del_ydomain) as i32;
            iyhi = ((grid_ycen + exclude_dist) / self.m_del_ydomain) as i32;
            iylo = b3dmax!(0, iylo);
            iyhi = b3dmin!(self.m_num_ydomains - 1, iyhi);

            // Then look at all candidates and make lists of ring numbers
            for iy in iylo..=iyhi {
                for ix in ixlo..=ixhi {
                    let idom = (ix + iy * self.m_num_xdomains) as usize;
                    for i in 0..self.m_domains[idom].num_candidates {
                        let ind =
                            self.m_candid_list[(self.m_domains[idom].cand_start_ind + i) as usize];
                        let cand = &self.m_candidates[ind as usize];
                        if cand.accepted == 0
                            && (cand.overlapped as i32 <= overlap_thresh)
                            && (cand.overlapped == 0
                                || (self.m_num_accepted[2] as f32) < self.m_overlap_target)
                            && (top_bot > 1 || cand.top_bot == top_bot)
                            && cand.score < self.m_outlier_score
                            && (cand.clustered == 0
                                || (dens_min < cluster_thresh
                                    && (self.m_num_accepted[2] as f32) < self.m_overlap_target))
                        {
                            dx = grid_xcen - cand.pos.x;
                            dy = grid_ycen - cand.pos.y;
                            dist = dx * dx + dy * dy;
                            if dist < exclude_sq {
                                let iring = (dist.sqrt() / delr) as i32;
                                cand_in_ring.push(ind);
                                ring_num.push(b3dmin!(num_rings - 1, iring));
                            }
                        }
                    }
                }
            }

            // Process the rings from middle outward
            for iring in 0..num_rings {
                ind_min = -1;
                min_score = 1.0e20;
                for ind in 0..cand_in_ring.len() {
                    if iring == ring_num[ind] {
                        // Find distance to nearest accepted point and determine if one is
                        // just too close
                        let cand = self.m_candidates[cand_in_ring[ind] as usize];
                        let idom = cand.domain as usize;
                        dist_min = 1.0e20;
                        too_close = false;
                        for neigh in 0..self.m_domains[idom].num_neighbors as usize {
                            let ndom = self.m_domains[idom].neighbors[neigh] as usize;
                            for i in 0..self.m_domains[ndom].num_accepted {
                                let ix = self.m_accept_list
                                    [(self.m_domains[ndom].cand_start_ind + i) as usize];
                                if cand.top_bot == self.m_candidates[ix as usize].top_bot {
                                    dist = imod_point_distance(
                                        &self.m_candidates[ix as usize].pos,
                                        &cand.pos,
                                    );
                                    dist_min = b3dmin!(dist, dist_min);
                                    if self
                                        .beads_are_clustered(&cand, &self.m_candidates[ix as usize])
                                    {
                                        too_close = true;
                                    }
                                }
                            }
                        }

                        // Penalize clustered and overlapped ones in the ring so others are
                        // given priority if any still exist
                        // Adjust score by dividing by distance: a very close point gets big
                        // score
                        score = ((cand.score + cand.clustered as f32) as f64
                            + cand.overlapped as f64 / 100.) as f32;
                        if dist_min < 1.0e19 {
                            score /= dist_min;
                        }
                        if score < min_score && !too_close {
                            min_score = score;
                            ind_min = cand_in_ring[ind];
                        }
                    }
                }

                // If a point was found in ring, accept it, update densities, break out of
                // ring loop
                if ind_min >= 0 {
                    let cand = self.m_candidates[ind_min as usize];
                    if self.m_verbose != 0 {
                        printf!(
                            "tb %d  density %.2f, add %d at %.0f, %.0f in ring %d, adjusted score %f\n",
                            CArg::Int(top_bot as i64),
                            CArg::Dbl(dens_min as f64 * 1.0e6),
                            CArg::Int(ind_min as i64),
                            CArg::Dbl(cand.pos.x as f64),
                            CArg::Dbl(cand.pos.y as f64),
                            CArg::Int(iring as i64),
                            CArg::Dbl(min_score as f64)
                        );
                    }
                    self.accept_candidate(ind_min, cand.top_bot);
                    self.revise_densities(ind_min, h);
                    break;
                }
            }
            if ind_min < 0 && self.m_verbose > 1 {
                printf!(
                    "tb %d  min density %.2f at %.0f, %.0f, no point found\n",
                    CArg::Int(top_bot as i64),
                    CArg::Dbl(dens_min as f64 * 1.0e6),
                    CArg::Dbl(grid_xcen as f64),
                    CArg::Dbl(grid_ycen as f64)
                );
            }
        }
        self.output_densities(top_bot);

        // This was used to validate that the adjusted densities are the same as the ones
        // computed from scratch
        /* computeDensities(topBot, H);
        outputDensities(topBot); */
    }

    /// Original: `PickSeeds::acceptCandidate` (`pickbestseed.cpp:1343`).
    ///
    /// Mark a candidate as accepted and add it to the list, increment counts.
    fn accept_candidate(&mut self, index: i32, top_bot: i32) {
        let cand = &mut self.m_candidates[index as usize];
        let idom = cand.domain as usize;
        cand.accepted = self.m_phase as u8;
        let dom = &mut self.m_domains[idom];
        self.m_accept_list[(dom.cand_start_ind + dom.num_accepted) as usize] = index;
        dom.num_accepted += 1;
        self.m_num_accepted[top_bot as usize] += 1;
        self.m_num_accepted[2] += 1;
    }

    /// Original: `PickSeeds::domainIndex` (`pickbestseed.cpp:1356`).
    ///
    /// Return domain index for a given x, y.
    fn domain_index(&self, x: f32, y: f32) -> i32 {
        let ix = (x / self.m_del_xdomain) as i32;
        let iy = (y / self.m_del_ydomain) as i32;
        if ix < 0 || ix >= self.m_num_xdomains || iy < 0 || iy >= self.m_num_ydomains {
            return -1;
        }
        ix + iy * self.m_num_xdomains
    }

    /// Original: `PickSeeds::getTrackDeviation` (`pickbestseed.cpp:1369`).
    ///
    /// Compare two tracks to see if they are close to each other, return the
    /// mean deviation between them and the fraction of points that are close.
    #[allow(clippy::too_many_arguments)]
    fn get_track_deviation(
        &self,
        co1: i32,
        mod1: i32,
        co2: i32,
        mod2: i32,
        mean_dev: &mut f32,
        frac_close: &mut f32,
        base_cont: Option<&Icont>,
    ) {
        let cont1 = &self.m_track_mods[mod1 as usize].obj[0].cont[co1 as usize];
        let cont2 = match base_cont {
            Some(cont) => cont,
            None => &self.m_track_mods[mod2 as usize].obj[0].cont[co2 as usize],
        };
        let mut dx: f32;
        let mut dy: f32;
        let mut dev: f32;
        let mut pt1 = 0usize;
        let mut pt2 = 0usize;
        let mut nsum = 0;
        let mut nclose = 0;
        let mut devsum: f32 = 0.;
        while pt1 < cont1.pts.len() && pt2 < cont2.pts.len() {
            if b3dnint!(cont1.pts[pt1].z) < b3dnint!(cont2.pts[pt2].z) {
                pt1 += 1;
            } else if b3dnint!(cont1.pts[pt1].z) > b3dnint!(cont2.pts[pt2].z) {
                pt2 += 1;
            } else {
                dx = cont1.pts[pt1].x - cont2.pts[pt2].x;
                dy = cont1.pts[pt1].y - cont2.pts[pt2].y;
                dev = (dx * dx + dy * dy).sqrt();
                devsum += dev;
                nsum += 1;
                if dev < self.m_close_diam_frac * self.m_bead_size {
                    nclose += 1;
                }
                pt1 += 1;
                pt2 += 1;
            }
        }
        *frac_close = if nsum != 0 {
            nclose as f32 / nsum as f32
        } else {
            0.
        };
        *mean_dev = if nsum != 0 { devsum / nsum as f32 } else { 0. };
    }

    /// Original: `PickSeeds::analyzeElongation` (`pickbestseed.cpp:1403`).
    ///
    /// Analyze the edge SD and elongation values for all the tracks that have
    /// gone into candidates and mark outliers as overlapped.
    ///
    /// The source's `static` work arrays are allocated here per call; every
    /// element read is written first in the same call.  Fixed in translation
    /// (`BUGS.md`, pickbestseed): `wsums` is allocated and never filled, then
    /// read by the `-control 17` options (fit to, or group by, the weighted
    /// sum); it is loaded here with each track's `wsumMean`, the value the
    /// elongation file provides for exactly that purpose.
    fn analyze_elongation(&mut self, max_conts: i32, which: i32, edge_sds: &mut [f32]) {
        let mut elongs = vec![0f32; max_conts as usize];
        let mut elong_tmp = vec![0f32; max_conts as usize];
        let mut wsums = vec![0f32; max_conts as usize];
        let mut elong_outlie = vec![0f32; max_conts as usize];
        let mut sort_ind = vec![0i32; max_conts as usize];
        let mut cont_ind = vec![0i32; max_conts as usize];
        let mut num_set = 0;
        let mut num_edge = 0usize;
        let mut include_start: i32;
        let mut include_end: i32;
        let mut test_start: i32;
        let mut test_end: i32;
        let mut edge_slope: f32 = 0.;
        let mut edge_intcp: f32 = 0.;
        let mut elong_slope: f32 = 0.;
        let mut elong_intcp: f32 = 0.;
        let mut ro: f32 = 0.;
        let mut se_slope: f32 = 0.;
        let mut se_intcp: f32 = 0.;
        let mut se: f32 = 0.;
        let mut tmp: f32 = 0.;
        let mut tmp2: f32 = 0.;
        let mut num_elong_out = 0;
        let mut num_elong_abs = 0;
        let mut num_tot = 0;
        let mut num_in_group = 0;
        let min_for_groups = 50;
        let min_to_test = 10;
        let test_by_group: bool;
        let cos_elong = (RADIANS_PER_DEGREE * self.m_elong_combo_angle as f64).cos();
        let sin_elong = (RADIANS_PER_DEGREE * self.m_elong_combo_angle as f64).sin();
        let cos_edge = (RADIANS_PER_DEGREE * self.m_edge_combo_angle as f64).cos();
        let sin_edge = (RADIANS_PER_DEGREE * self.m_edge_combo_angle as f64).sin();
        let cos_norm = (RADIANS_PER_DEGREE * self.m_norm_combo_angle as f64).cos();
        let sin_norm = (RADIANS_PER_DEGREE * self.m_norm_combo_angle as f64).sin();

        // Load the data arrays for all contours of all candidates
        for co in 0..max_conts as usize {
            let track = &self.m_tracks[co];
            if track.cand_index >= 0 && self.m_candidates[track.cand_index as usize].clustered == 0
            {
                edge_sds[num_edge] = (cos_edge * track.edge_sd_mean as f64
                    + sin_edge * track.edge_sd_sd as f64)
                    as f32;
                elongs[num_edge] = (cos_elong * track.elong_mean as f64
                    + sin_elong * track.elong_sd as f64) as f32;
                wsums[num_edge] = track.wsum_mean;
                sort_ind[num_edge] = num_edge as i32;
                cont_ind[num_edge] = co as i32;
                num_edge += 1;
            }
        }
        if num_edge == 0 {
            return;
        }
        let n = num_edge as i32;

        // Fit each measure versus the wsum and replace with the residual of the fit
        // (doesn't help)
        if self.m_num_wsum != 0 && (self.m_option_flags & OPTION_FIT_TO_WSUM) != 0 {
            ls_fit_pred(
                &wsums,
                edge_sds,
                n,
                &mut edge_slope,
                &mut edge_intcp,
                &mut ro,
                &mut se_intcp,
                &mut se_slope,
                &mut se,
                0.,
                &mut tmp,
                &mut tmp2,
            );
            if self.m_verbose != 0 {
                printf!(
                    "EdgeSD vs wsum: slope %f sb %f  intcp %f sa %f  ro %f\n",
                    CArg::Dbl(edge_slope as f64),
                    CArg::Dbl(se_slope as f64),
                    CArg::Dbl(edge_intcp as f64),
                    CArg::Dbl(se_intcp as f64),
                    CArg::Dbl(ro as f64)
                );
            }
            ls_fit_pred(
                &wsums,
                &elongs,
                n,
                &mut elong_slope,
                &mut elong_intcp,
                &mut ro,
                &mut se_intcp,
                &mut se_slope,
                &mut se,
                0.,
                &mut tmp,
                &mut tmp2,
            );
            if self.m_verbose != 0 {
                printf!(
                    "elong vs wsum: slope %f sb %f  intcp %f sa %f  ro %f\n",
                    CArg::Dbl(elong_slope as f64),
                    CArg::Dbl(se_slope as f64),
                    CArg::Dbl(elong_intcp as f64),
                    CArg::Dbl(se_intcp as f64),
                    CArg::Dbl(ro as f64)
                );
            }
            for co in 0..num_edge {
                edge_sds[co] -= wsums[co] * edge_slope + edge_intcp;
                elongs[co] -= wsums[co] * elong_slope + elong_intcp;
            }
        }

        // Normalize the edgeSD's to the same SD as the elongs and combine
        avg_sd(edge_sds, n, &mut edge_intcp, &mut edge_slope, &mut ro);
        avg_sd(&elongs, n, &mut elong_intcp, &mut elong_slope, &mut ro);
        for co in 0..num_edge {
            elongs[co] = (sin_norm * edge_sds[co] as f64 * elong_slope as f64 / edge_slope as f64
                + cos_norm * elongs[co] as f64) as f32;
        }

        include_start = 0;
        include_end = n;
        test_start = 0;
        test_end = n;
        test_by_group = self.m_num_wsum != 0
            && (self.m_option_flags & OPTION_WSUM_GROUPS) != 0
            && n >= min_for_groups;
        if test_by_group {
            num_in_group = b3dmax!(min_for_groups, (n + 3) / 4);
            include_end = num_in_group - min_to_test;
            rs_sort_indexed_floats(&wsums, &mut sort_ind, n);
        }

        loop {
            if test_by_group {
                // Advance end from last time by minToTest, limit to numEdge, and get start
                // from that, and set test end at half of test group size past middle of
                // group to analyze, except when analysis goes to end
                include_end = b3dmin!(include_end + min_to_test, n);
                include_start = include_end - num_in_group;
                test_end = (include_start + include_end + min_to_test) / 2;
                if include_end == n {
                    test_end = n;
                }
            }

            // Load the data
            for ind in include_start..include_end {
                elong_tmp[(ind - include_start) as usize] = elongs[sort_ind[ind as usize] as usize];
            }

            // Analyze for outliers, mark as overlapped if it is an outlier on either
            // measure or exceeds the absolute elongation
            rs_mad_median_outliers(
                &elong_tmp,
                include_end - include_start,
                self.m_edge_outlier_crit,
                &mut elong_outlie,
            );
            for ind in test_start - include_start..test_end - include_start {
                let co = cont_ind[sort_ind[(ind + include_start) as usize] as usize] as usize;
                let track = self.m_tracks[co];
                if elong_outlie[ind as usize] > 0. || track.elong_median >= self.m_elong_for_overlap
                {
                    let cand = &mut self.m_candidates[track.cand_index as usize];
                    if which != 0 {
                        cand.overlap_tmp = cand.overlap_tmp.wrapping_add(1);
                    } else {
                        cand.overlapped = cand.overlapped.wrapping_add(1);
                    }
                    num_tot += 1;
                    if elong_outlie[ind as usize] > 0. {
                        num_elong_out += 1;
                    }
                    if track.elong_median >= self.m_elong_for_overlap {
                        num_elong_abs += 1;
                    }
                }
            }
            test_start = test_end;
            if test_end >= n {
                break;
            }
        }

        // Count the candidates marked as overlapped and set the overlap to the max on the
        // second round
        for co in 0..self.m_num_candidates as usize {
            let cand = &mut self.m_candidates[co];
            if which != 0 {
                cand.overlapped = b3dmax!(cand.overlapped, cand.overlap_tmp);
            }
            if cand.overlapped != 0 {
                num_set += 1;
            }
        }

        if self.m_verbose != 0 {
            printf!(
                "%d beads identified as elongated after round %d\n",
                CArg::Int(num_set as i64),
                CArg::Int(which as i64 + 1)
            );
            print_vars!(
                ("numTot", CArg::Int(num_tot as i64)),
                ("numElongOut", CArg::Int(num_elong_out as i64)),
                ("numElongAbs", CArg::Int(num_elong_abs as i64))
            );
        }
    }

    /// Original: `PickSeeds::beadsAreClustered` (`pickbestseed.cpp:1515`).
    ///
    /// Test whether two beads are "clustered", i.e., too close to each other
    /// after considering tilt foreshortening at highest tilt.
    fn beads_are_clustered(&self, cand1: &Candidate, cand2: &Candidate) -> bool {
        let mut dx: f32;
        let mut dy: f32;
        let dxtmp: f32;

        // Rotate the vector so the tilt axis is vertical and foreshorten X by highest
        // tilt angle to determine worst-case separation
        dx = cand1.pos.x - cand2.pos.x;
        dy = cand1.pos.y - cand2.pos.y;
        if self.m_highest_tilt != 0. {
            dxtmp = dx * self.m_cos_rot + dy * self.m_sin_rot;
            dy = -dx * self.m_sin_rot + dy * self.m_cos_rot;
            dx = self.m_cos_tilt * dxtmp;
        }
        (dx * dx + dy * dy).sqrt() < self.m_cluster_crit * self.m_bead_size
    }

    /// Original: `PickSeeds::computeAreaFracs` (`pickbestseed.cpp:1539`).
    ///
    /// Compute a weighted sum of area actually present around a grid point
    /// inside boundaries.  This can be used as a denominator to get density
    /// from the kernel sum of beads around a grid point.
    ///
    /// Fixed in translation (`BUGS.md`, pickbestseed): the source weights ring
    /// `ring` by `PI * (2. * ring + delr) * delr`, the ring *index* where the
    /// annulus area `PI * ((ring + 1)^2 - ring^2) * delr^2` needs the radius
    /// `ring * delr`; here it is `PI * (2. * ring + 1.) * delr * delr`, in both
    /// the normalising sum and each grid point's sum.
    fn compute_area_fracs(&mut self, h: f32) {
        let num_rings = 20;
        let delr = h / num_rings as f32;
        let mut max_sum: f32 = 0.;
        let mut rr: f32;
        let mut weight: f32;
        for ring in 0..num_rings {
            rr = ((ring as f64 + 0.5) * delr as f64) as f32;
            weight = (1. - ((rr / h) * (rr / h)) as f64).powf(3.) as f32;
            max_sum = (max_sum as f64
                + PI * (2. * ring as f64 + 1.) * delr as f64 * delr as f64 * weight as f64)
                as f32;
        }

        // Sample small sectors in small rings to see if they are inside area being
        // considered
        for idom in 0..(self.m_num_xdomains * self.m_num_ydomains) as usize {
            for gpt in 0..self.m_domains[idom].num_grid_pts as usize {
                let mut area_sum: f32 = 0.;
                let gpt_ind = self.m_domains[idom].grid_pt_ind[gpt] as usize;
                let xcen = self.m_grid_points[gpt_ind].x;
                let ycen = self.m_grid_points[gpt_ind].y;
                for ring in 0..num_rings {
                    rr = ((ring as f64 + 0.5) * delr as f64) as f32;
                    let num_samp = b3dnint!(2. * PI * rr as f64 / delr as f64);
                    let mut num_inside = 0;
                    for samp in 0..num_samp {
                        let angle = (2. * PI * samp as f64 / num_samp as f64) as f32;
                        let xx = xcen + rr * angle.cos();
                        let yy = ycen + rr * angle.sin();
                        if xx >= self.m_area_xmin
                            && xx <= self.m_area_xmax
                            && yy >= self.m_area_ymin
                            && yy <= self.m_area_ymax
                        {
                            if self.m_num_area_cont != 0 {
                                // This adds 1 if point is inside and exclude = 0, or if the
                                // point is outside and exclude = 1
                                num_inside += if imod_point_inside_area(
                                    &self.m_area_mod.as_ref().unwrap().obj[0],
                                    &self.m_area_conts,
                                    self.m_num_area_cont,
                                    xx,
                                    yy,
                                ) >= 0
                                {
                                    1 - self.m_exclude_areas
                                } else {
                                    self.m_exclude_areas
                                };
                            } else {
                                num_inside += 1;
                            }
                        }
                    }
                    weight = (1. - ((rr / h) * (rr / h)) as f64).powf(3.) as f32;
                    area_sum = (area_sum as f64
                        + PI * (2. * ring as f64 + 1.)
                            * delr as f64
                            * delr as f64
                            * (weight * num_inside as f32) as f64
                            / num_samp as f64) as f32;
                }
                self.m_grid_points[gpt_ind].area_frac = area_sum / max_sum;
            }
        }
    }

    /// Original: `PickSeeds::computeDensities` (`pickbestseed.cpp:1586`).
    ///
    /// Determine the density at all grid points from scratch.
    fn compute_densities(&mut self, top_bot: i32, h: f32) {
        let hsqr = h * h;
        let densfac = (4. / (PI * hsqr as f64)) as f32;
        for gpt in 0..(self.m_num_xgrid * self.m_num_ygrid) as usize {
            let dom = self.m_grid_points[gpt].domain;
            if dom < 0 {
                continue;
            }
            let mut dsum: f32 = 0.;
            for neigh in 0..self.m_domains[dom as usize].num_neighbors as usize {
                let ind = self.m_domains[dom as usize].neighbors[neigh] as usize;
                for cpt in 0..self.m_domains[ind].num_accepted {
                    let cand = &self.m_candidates[self.m_accept_list
                        [(self.m_domains[ind].cand_start_ind + cpt) as usize]
                        as usize];
                    if top_bot > 1 || cand.top_bot == top_bot {
                        let dx = cand.pos.x - self.m_grid_points[gpt].x;
                        let dy = cand.pos.y - self.m_grid_points[gpt].y;
                        let dist = dx * dx + dy * dy;
                        if dist < hsqr {
                            dsum = (dsum as f64 + (1. - (dist / hsqr) as f64).powf(3.)) as f32;
                        }
                    }
                }
            }
            self.m_grid_points[gpt].density = densfac * dsum / self.m_grid_points[gpt].area_frac;
        }
    }

    /// Original: `PickSeeds::reviseDensities` (`pickbestseed.cpp:1617`).
    ///
    /// When a candidate is accepted, add its contribution to the density
    /// around nearby grid points.
    fn revise_densities(&mut self, accept_new: i32, h: f32) {
        let hsqr = h * h;
        let densfac = (4. / (PI * hsqr as f64)) as f32;
        let cand = self.m_candidates[accept_new as usize];
        let ixlo = b3dmax!(0, ((cand.pos.x - h) / self.m_del_xgrid) as i32 - 1);
        let ixhi = b3dmin!(
            self.m_num_xgrid - 1,
            ((cand.pos.x + h) / self.m_del_xgrid).ceil() as i32 - 1
        );
        let iylo = b3dmax!(0, ((cand.pos.y - h) / self.m_del_ygrid) as i32 - 1);
        let iyhi = b3dmin!(
            self.m_num_ygrid - 1,
            ((cand.pos.y + h) / self.m_del_ygrid).ceil() as i32 - 1
        );
        for iy in iylo..=iyhi {
            for ix in ixlo..=ixhi {
                let gpt = (ix + iy * self.m_num_xgrid) as usize;
                let gp = &mut self.m_grid_points[gpt];
                if gp.domain >= 0 {
                    let dx = cand.pos.x - gp.x;
                    let dy = cand.pos.y - gp.y;
                    let dist = dx * dx + dy * dy;
                    if dist < hsqr {
                        gp.density = (gp.density as f64
                            + densfac as f64 * (1. - (dist / hsqr) as f64).powf(3.)
                                / gp.area_frac as f64) as f32;
                    }
                }
            }
        }
    }

    /// Original: `PickSeeds::outputDensities` (`pickbestseed.cpp:1645`).
    ///
    /// Put out densities in a gnuplot format.
    fn output_densities(&self, top_bot: i32) {
        let Some(root) = self.m_dens_plot_name.as_ref() else {
            return;
        };
        if self.m_phase != LAST_PHASE.get() {
            SEQUENCE.set(1);
            LAST_PHASE.set(self.m_phase);
        } else {
            SEQUENCE.set(SEQUENCE.get() + 1);
        }
        let fname = format!(
            "{}-{}-{}-{}.dat",
            root,
            self.m_phase,
            top_bot,
            SEQUENCE.get()
        );
        let Some(mut fp) = ImodFile::open(&fname, "w") else {
            return;
        };

        let mut text: Vec<u8> = Vec::new();
        for iy in 0..self.m_num_ygrid {
            for ix in 0..self.m_num_xgrid {
                let gp = &self.m_grid_points[(ix + iy * self.m_num_xgrid) as usize];
                text.extend_from_slice(&c_format_bytes(
                    "%f %f %f\n",
                    &[
                        CArg::Dbl(gp.x as f64),
                        CArg::Dbl(gp.y as f64),
                        CArg::Dbl(1.0e6 * gp.density as f64),
                    ],
                ));
            }
            text.push(b'\n');
        }
        let _ = fp.write_all(&text);
    }

    /// Original: `PickSeeds::outputNumAccepted` (`pickbestseed.cpp:1680`).
    ///
    /// Output number of points accepted total and on each surface.  The
    /// `Final:` values are also recorded for a direct caller.
    fn output_num_accepted(&self, two_surf: i32, final_: i32) {
        if final_ != 0 {
            printf!("Final:   ");
        } else {
            printf!(
                "Phase %d: ",
                CArg::Int((self.m_phase - self.m_append_to_seed) as i64)
            );
        }
        printf!(
            "total points accepted = %d",
            CArg::Int(self.m_num_accepted[2] as i64)
        );
        if two_surf != 0 {
            printf!(
                "  -  on bottom = %d , on top = %d",
                CArg::Int(self.m_num_accepted[0] as i64),
                CArg::Int(self.m_num_accepted[1] as i64)
            );
        }
        printf!(if final_ != 0 { "   [PBS2]\n" } else { "\n" });
        if final_ != 0 {
            let accepted = self.m_num_accepted;
            record_result(|result| {
                result.final_total = Some(accepted[2]);
                result.on_bottom_top = if two_surf != 0 {
                    Some((accepted[0], accepted[1]))
                } else {
                    None
                };
            });
        }
    }
}
