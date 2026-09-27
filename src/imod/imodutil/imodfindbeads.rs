//! Translation of `IMOD/imodutil/imodfindbeads.cpp` with its class header
//! `IMOD/imodutil/imodfindbeads.h` merged in.
//!
//! The C++ `FindBeads` class becomes the [`FindBeads`] struct, one method per
//! member function, each named after the original in its doc comment.  The
//! source's pointer arithmetic that selects a member of `PeakEntry`
//! (`*(&mPeakList[j].peak + (element - &mPeakList[0].peak))`) is the
//! [`PeakField`] selector; the two histogram arrays a method writes into are
//! selected by [`Hist`].  `mPeakList`'s live length is `mNumPeaks`: the `Vec`
//! is cleared where the source resets `mNumPeaks` to 0 and pushed where it
//! stores at `mPeakList[mNumPeaks++]`.
//!
//! Direct-call API (`CLAUDE.md`, "Direct calls instead of parsed output"):
//! [`imodfindbeads_recording`] runs the program with every report autofidseed
//! parses recorded into a [`FindbeadsResult`], at the point it is printed.
//!
//! Upstream defects fixed in translation (`BUGS.md`, imodfindbeads), each
//! commented at its site:
//! * `extractDiameter` never incremented `nfit`, so the "fit to ones around
//!   maximum gradient" was always a two-point fit (and the loop's `ind > ndat`
//!   bound read one element past the filled part of `avgden`);
//! * `findStorageThreshold` / `main`: `line` and `histDip` were read
//!   uninitialised when `-store` is positive; both now start empty / -1;
//! * `setupSizeDependentVars`: `minsize` uninitialised when no candidate size
//!   is > 3; it keeps `mBoxSize`;
//! * `contourIsBelowThreshold` scanned other contours' stores when the
//!   contour had none (`istoreLookup` returned -1);
//! * `fillInByTiltFits`: the fit window used a stale `indMin` instead of the
//!   nearest point (the unused `dzMin` shows the intent); `jndWinsOnY` and the
//!   pairing comparison tested `jnd` against itself; the track/track branch
//!   tested `indWinsOnY` for `jnd`; and the pairing/track branch credited the
//!   wins to the wrong entries; `VEC_MINIMUM` on an empty list;
//! * the reference comparison skipped object `boundObj` (1-based) instead of
//!   `boundObj - 1`;
//! * error messages printed a freed filename.
//! `printArray` is dead in the source (only commented-out calls) and is not
//! translated.

use std::cell::RefCell;
use std::io::Write as _;
use std::sync::{Arc, Mutex};

use crate::imod::c_sort::qsort;
use crate::imod::clip::clip::{ScanArg, sscanf};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, exit, fgetline, imod_backup_file, imod_usage_header,
    program_args, wall_time,
};
use crate::imod::libcfshr::beadutil::{bead_integral, make_model_bead};
use crate::imod::libcfshr::cubinterp::cubinterp;
use crate::imod::libcfshr::filtxcorr::{
    FilterIn, conjugate_product, nice_frame, parabolic_fit_position, scaled_gaussian_kernel,
    xcorr_filter_part, xcorr_set_ctf,
};
use crate::imod::libcfshr::gettiltangles::read_tilt_file;
use crate::imod::libcfshr::histogram::scan_histogram;
use crate::imod::libcfshr::islice::{Islice, slice_mat_filter, slice_mode_if_real};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_done, pip_get_boolean, pip_get_float, pip_get_in_out_file, pip_get_integer,
    pip_get_string, pip_get_two_integers, pip_print_help, pip_read_or_parse_options,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libcfshr::scaledsobel::scaled_sobel;
use crate::imod::libcfshr::simplestat::{avg_sd, ls_fit, ls_fit2, ls_fit2_pred, sums_to_avg_sd};
use crate::imod::libcfshr::taperatfill::slice_taper_at_fill;
use crate::imod::libcfshr::taperpad::{
    PadIn, slice_edge_mean, slice_split_fill, slice_taper_out_pad,
};
use crate::imod::libfft::todfft::todfft;
use crate::imod::libiimod::iimage::{ii_fclose, ii_fopen};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_FLOAT, MrcHeader, mrc_head_label, mrc_head_label_cp, mrc_head_new, mrc_head_read,
    mrc_head_write, mrc_write_slice,
};
use crate::imod::libiimod::mrcslice::{SLICE_MODE_FLOAT, slice_read_float};
use crate::imod::libimod::icont::{imod_contour_area, imod_contour_new, imod_contours_new};
use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OPEN, IMOD_UNIT_NM, IMODF_NEW_TO_3DMOD, Imod, Ipoint, imod_new, imod_new_object,
    imod_set_ref_image, imod_trans_from_ref_image,
};
use crate::imod::libimod::imodel_files::{imod_read, imod_write_file};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_PNT_ON_SEC, IMOD_OBJFLAG_USE_VALUE, MATFLAGS2_CONSTANT, MATFLAGS2_SKIP_LOW,
    imod_object_add_contour, iobj_close,
};
use crate::imod::libimod::ipoint::{
    imod_point_append, imod_point_append_xyz, imod_point_distance, imod_point_inside_area,
    imod_point_inside_cont, make_area_cont_list,
};
use crate::imod::libimod::istore::{
    GEN_STORE_FLOAT, GEN_STORE_MINMAX1, GEN_STORE_VALUE1, Istore, StoreUnion, istore_add_min_max,
    istore_get_min_max, istore_insert, istore_lookup,
};
use crate::imod::libimod::iview::imod_objview_from_object;

// Some limits for arrays (`imodfindbeads.h:17-21`)
const MAX_BINS: usize = 10000;
const MAX_GROUPS: usize = 10;
const KERNEL_MAXSIZE: i32 = 7;
const MAX_AREAS: usize = 1000;
const MAX_LINE: usize = 160;

/// `iobj.h:76`: `#define IOBJ_SYM_CIRCLE 0`.
const IOBJ_SYM_CIRCLE: u8 = 0;
/// `b3dutil.h:68`: `#define RADIANS_PER_DEGREE 0.01745329252` -- a double.
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

/// The source's `printf`, through the C-format writer on libc-order stdout.
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

/// `cppdefs.h:22-24`: `PRINT2`..`PRINT4` go through `cout`, whose default
/// float output is `%g`.
fn cout_line(parts: &[(&str, CArg)]) {
    let mut line: Vec<u8> = Vec::new();
    for (i, (name, val)) in parts.iter().enumerate() {
        if i > 0 {
            line.extend_from_slice(b",  ");
        }
        line.extend_from_slice(name.as_bytes());
        line.extend_from_slice(b" = ");
        let fmt = match val {
            CArg::Dbl(_) => "%g",
            _ => "%d",
        };
        line.extend_from_slice(&c_format_bytes(fmt, std::slice::from_ref(val)));
    }
    line.push(b'\n');
    let _ = ImodFile::Stdout.write_all(&line);
}

/// `PeakEntry` (`imodfindbeads.h:4-14`).
#[derive(Clone, Copy, Debug, Default)]
struct PeakEntry {
    xcen: f32,
    ycen: f32,
    iz: i32,
    ccc: f32,
    integral: f32,
    peak: f32,
    cenmean: f32,
    annmean: f32,
    median: f32,
}

/// The member of a `PeakEntry` that the source addresses through a pointer
/// into `mPeakList[0]`.
#[derive(Clone, Copy, PartialEq)]
enum PeakField {
    Peak,
    Annmean,
    Median,
}

impl PeakEntry {
    fn get(&self, field: PeakField) -> f32 {
        match field {
            PeakField::Peak => self.peak,
            PeakField::Annmean => self.annmean,
            PeakField::Median => self.median,
        }
    }
}

/// Which of `mRegHist` / `mKernHist` a histogram routine fills.
#[derive(Clone, Copy)]
enum Hist {
    Reg,
    Kern,
}

/// One report autofidseed reads from imodfindbeads output, recorded where it
/// is printed (Rust-only; see [`imodfindbeads_recording`]).  Values are the
/// ones passed to `printf`; the printed rounding is the format named.
#[derive(Clone, Debug, PartialEq)]
pub enum FindbeadsReport {
    /// `Area (megapixels) included in analysis = %.3f` (no newline; the
    /// program exits right after).
    Area(f64),
    /// `Adjusted parameters based on a new bead size of %.2f`.
    AdjustedSize(f32),
    /// `%d peaks are above threshold of %.3f` (automatic storage threshold).
    PeaksAboveThreshold { num: i32, dip: f32 },
    /// `%d more peaks %s stored in model down to value of %.3f`.
    MorePeaksStored {
        num: i32,
        would_be: bool,
        thresh: f32,
    },
    /// `%d peaks are above histogram dip at %.3f`.
    PeaksAboveDip { num: i32, dip: f32 },
    /// `Failed to find dip in histogram, using fallback threshold for storing
    /// points`.
    UsingFallback,
    /// `Failed to find dip in histogram` (storage threshold).
    FailedToFindDip,
    /// `%d %s` with the deferred `total peaks ...` line: `num` is the leading
    /// number and `text` the rest of the line (ending in a newline).
    TotalPeaksStored { num: i32, text: String },
    /// `%d peaks above threshold of %.3f are being stored in model`.
    PeaksAboveStorageThreshold { num: i32, thresh: f32 },
}

/// What [`imodfindbeads_recording`] collects: every [`FindbeadsReport`] in
/// print order.  With the options autofidseed passes (no `-ref`, `-fill` or
/// `-add`) the last two storage reports are the last two lines of output.
#[derive(Clone, Debug, Default)]
pub struct FindbeadsResult {
    pub reports: Vec<FindbeadsReport>,
}

thread_local! {
    /// Where the program records its [`FindbeadsResult`] on this thread, when
    /// a direct caller set it through [`imodfindbeads_recording`].
    static RESULT_SINK: RefCell<Option<Arc<Mutex<FindbeadsResult>>>> =
        const { RefCell::new(None) };
}

/// Rust-only: runs program [`imodfindbeads`] on this thread with its reported
/// values recorded into `sink`.  The program ends through `exit`, so the
/// values reach the caller through `sink`; run it under
/// `commands::call_in_process`.
pub fn imodfindbeads_recording(sink: Arc<Mutex<FindbeadsResult>>) {
    RESULT_SINK.with_borrow_mut(|slot| *slot = Some(sink));
    imodfindbeads();
}

/// Rust-only: records a report for the direct caller, if any.
fn record(report: FindbeadsReport) {
    RESULT_SINK.with_borrow(|slot| {
        if let Some(sink) = slot {
            sink.lock()
                .expect("imodfindbeads result sink")
                .reports
                .push(report);
        }
    });
}

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// `static int comparePeaks(const void *p1, const void *p2)`.
fn compare_peaks(peak1: &PeakEntry, peak2: &PeakEntry) -> i32 {
    if peak1.ccc == 0. && peak2.ccc != 0. {
        -1
    } else if peak1.ccc != 0. && peak2.ccc == 0. {
        1
    } else if peak1.peak < peak2.peak {
        -1
    } else if peak1.peak > peak2.peak {
        1
    } else {
        0
    }
}

/// `main` (`imodfindbeads.cpp:29`): construct the object and call its main.
pub fn imodfindbeads() {
    let mut find = FindBeads::new();
    find.main(&program_args());
    exit(0);
}

/// `class FindBeads` (`imodfindbeads.h:23-120`).
pub struct FindBeads {
    m_reg_hist: Vec<f32>,
    m_kern_hist: Vec<f32>,
    m_in_head: MrcHeader,
    m_average_fallback: i32,
    m_storage_fallback: i32,
    m_cached_slices: Vec<Option<Islice>>,
    m_use_slice_cache: bool,
    m_filt_slice: Vec<f32>,
    m_corr_slice: Vec<f32>,
    m_full_bead: Vec<f32>,
    m_one_bead: Vec<f32>,
    m_split_bead: Vec<f32>,
    m_peak_list: Vec<PeakEntry>,
    m_zlist: Vec<i32>,
    m_dump_type: i32,
    m_threshold: f32,
    m_peak_thresh: f32,
    m_center_weight: f32,
    m_light_beads: i32,
    m_annulus_pctile: f32,
    m_min_relative_peak: f32,
    m_min_spacing: f32,
    m_scaled_size: f32,
    m_bead_size: f32,
    m_scale_factor: f32,
    m_box_size: i32,
    m_num_peaks: i32,
    m_nx_pad: i32,
    m_ny_pad: i32,
    m_nxp_dim: i32,
    m_nx_out: i32,
    m_ny_out: i32,
    m_rad_center: f32,
    m_rad_inner: f32,
    m_rad_outer: f32,
    m_min_dist: f32,
    m_match_crit: f32,
    m_num_groups: i32,
    m_vkeys: Option<String>,
    m_min_interp: f32,
    m_linear_interp: i32,
    m_write_slice: Vec<f32>,
    m_xoffset: f32,
    m_yoffset: f32,
    m_nx_in: i32,
    m_ny_in: i32,
    m_box_scaled: i32,
    m_box_scaled_orig: i32,
    m_peak_max: f32,
    m_dump_fp: Option<ImodFile>,
    m_in_fp: Option<ImodFile>,
    m_min_guess: i32,
    m_num_obj_orig: i32,
    m_align_xshift: Option<Vec<f32>>,
    m_align_yshift: Option<Vec<f32>>,
    m_list_size: i32,
    m_filt_bead: Vec<f32>,
    m_area_mod: Option<Imod>,
    m_area_conts: [i32; MAX_AREAS],
    m_num_area_cont: i32,
    m_measure_to_use: i32,
    m_exclude_areas: i32,
    m_bead_cen_ofs: f32,
    m_adjust_sizes: i32,
    m_wall_start: f64,
    m_wall_last: f64,
    m_profiling: bool,
    m_min_size_for_adjust: f32,
    m_max_adjust_factor: f32,
    m_min_size_change_for_redo: f32,
    m_avg_xoffset: f32,
    m_avg_yoffset: f32,
    m_tilt_angles: Vec<f32>,
    m_obj_thresh: Vec<f32>,
}

impl FindBeads {
    /// Constructor `FindBeads::FindBeads()`: initialize variables with
    /// defaults.  Members the source leaves unset start at zero.
    pub fn new() -> FindBeads {
        let now = wall_time();
        FindBeads {
            m_reg_hist: vec![0.; MAX_BINS],
            m_kern_hist: vec![0.; MAX_BINS],
            m_in_head: MrcHeader::default(),
            m_average_fallback: 0,
            m_storage_fallback: 0,
            m_cached_slices: Vec::new(),
            m_use_slice_cache: false,
            m_filt_slice: Vec::new(),
            m_corr_slice: Vec::new(),
            m_full_bead: Vec::new(),
            m_one_bead: Vec::new(),
            m_split_bead: Vec::new(),
            m_peak_list: Vec::new(),
            m_zlist: Vec::new(),
            m_dump_type: 1,
            m_threshold: -2.,
            m_peak_thresh: 0.,
            m_center_weight: 2.,
            m_light_beads: 0,
            m_annulus_pctile: -1.,
            m_min_relative_peak: 0.1,
            m_min_spacing: 1.,
            m_scaled_size: 8.,
            m_bead_size: 0.,
            m_scale_factor: 0.,
            m_box_size: 0,
            m_num_peaks: 0,
            m_nx_pad: 0,
            m_ny_pad: 0,
            m_nxp_dim: 0,
            m_nx_out: 0,
            m_ny_out: 0,
            m_rad_center: 0.,
            m_rad_inner: 0.,
            m_rad_outer: 0.,
            m_min_dist: 0.,
            m_match_crit: 0.,
            m_num_groups: 4,
            m_vkeys: None,
            m_min_interp: 1.4,
            m_linear_interp: 0,
            m_write_slice: Vec::new(),
            m_xoffset: 0.,
            m_yoffset: 0.,
            m_nx_in: 0,
            m_ny_in: 0,
            m_box_scaled: 0,
            m_box_scaled_orig: 0,
            m_peak_max: 0.,
            m_dump_fp: None,
            m_in_fp: None,
            m_min_guess: 0,
            m_num_obj_orig: 0,
            m_align_xshift: None,
            m_align_yshift: None,
            m_list_size: 0,
            m_filt_bead: Vec::new(),
            m_area_mod: None,
            m_area_conts: [0; MAX_AREAS],
            m_num_area_cont: 0,
            m_measure_to_use: 1,
            m_exclude_areas: 0,
            m_bead_cen_ofs: 0.,
            m_adjust_sizes: 0,
            m_wall_start: now,
            m_wall_last: now,
            m_profiling: false,
            // Minimum bead size to apply adjustment for
            m_min_size_for_adjust: 5.,
            // Maximum factor to change the bead size by
            m_max_adjust_factor: 1.6,
            // Minimum factor of change for adding another pass
            m_min_size_change_for_redo: 1.05,
            m_avg_xoffset: 0.,
            m_avg_yoffset: 0.,
            m_tilt_angles: Vec::new(),
            m_obj_thresh: Vec::new(),
        }
    }

    /// `FindBeads::main` (`imodfindbeads.cpp:84`): option processing, main
    /// loop, and final output.
    pub fn main(&mut self, argv: &[String]) {
        let progname = "imodfindbeads";
        let mut num_opt_args = 0;
        let mut num_non_opt_args = 0;
        let mut refmod: Option<Imod> = None;
        let mut imod: Imod;
        let mut pnt = Ipoint::default();
        let mut outfp: Option<ImodFile> = None;
        let mut outhead = MrcHeader::default();
        let mut bound_obj: i32 = -1;
        let mut num_guess = 0;
        let mut remake_model_bead = 0;
        let forward = 0;
        let inverse = 1;
        let mut has_ref = false;
        let bin_scale = Ipoint {
            x: 1.,
            y: 1.,
            z: 1.,
        };
        let mut ctf = vec![0f32; 8193];
        // Fixed in translation: `line` is read by `main` after every group
        // but set only by `findStorageThreshold` when `-store` is not positive
        // (uninitialised stack in the source).  It starts empty.
        let mut line = String::new();
        let mut num_ref_pts = 0;
        // Fixed in translation: these three are printed after the reference
        // comparison even when no peak set them.
        let mut num_peak_pts = 0;
        let mut num_peak_match = 0;
        let mut peak_above: f32 = 0.;
        let mut peak_below: f32 = 0.;
        let mut num_below_min;
        let mut num_matched = 0;
        let mut num_unmatched = 0;
        let mut kernel_sigma: f32 = 0.85;
        let mut replace_angle: f32 = -1.;
        let replace_dist_frac: f32 = 0.5;
        let mut fill_in_added = -1;
        let mut debug_track = -1;
        let mut debug_z = -1;
        let mut sigma1: f32 = 0.;
        let mut sigma2: f32 = 0.;
        let mut radius1: f32 = 0.;
        let mut radius2: f32 = 0.;
        let mut iz: i32 = 0;
        let mut npass;
        let mut binning: i32;
        let nzout: i32;
        let mut nxtmp = 0;
        let mut nytmp = 0;
        let mut xscale: f32;
        let mut yscale: f32;
        let mut zscale: f32;
        let mut xtmp: f32 = 0.;
        let mut ytmp: f32 = 0.;
        let mut x_bead_ofs: f32 = 0.;
        let mut y_bead_ofs: f32 = 0.;
        let mut xcen: f32 = 0.;
        let mut ycen: f32 = 0.;
        let mut num_eliminated;
        let cache_limit_mb = 768;
        let mut ctf_delta: f32 = 0.;
        // Fixed in translation: read by the replacement test and the black
        // level after every group, set only by `findStorageThreshold`
        // (uninitialised in the source when `-store` is positive).  -1 is
        // the value that routine uses for "no dip".
        let mut hist_dip: f32 = -1.;
        let mut ob_min: f32 = 0.;
        let mut ob_max: f32 = 0.;
        let mut best_ob = 0usize;
        let mut best_co = 0usize;
        let mut best_pt = 0usize;
        let mut num_replaced;
        let mut ndat = 0;
        let mut kernel = [0f32; (KERNEL_MAXSIZE * KERNEL_MAXSIZE) as usize];
        let mut num_peaks_left = 0;
        let mut co_flag = 0;

        // Fallbacks from    ../manpages/autodoc2man 2 1 imodfindbeads
        let num_options = 43;
        let options: [&[u8]; 43] = [
            b"input:InputImageFile:FN:",
            b"output:OutputModelFile:FN:",
            b"filtered:FilteredImageFile:FN:",
            b"area:AreaModel:FN:",
            b"exclude:ExcludeInsideAreas:B:",
            b"query:QueryAreaOnSection:I:",
            b"prexf:PrealignTransformFile:FN:",
            b"imagebinned:ImagesAreBinned:I:",
            b"add:AddToModel:FN:",
            b"replace:ReplaceAboveAngle:F:",
            b"tiltfile:TiltAngleFile:FN:",
            b"fill:FillInMissingPoints:I:",
            b"ref:ReferenceModel:FN:",
            b"boundary:BoundaryObject:I:",
            b"size:BeadSize:F:",
            b"light:LightBeads:B:",
            b"scaled:ScaledSize:F:",
            b"adjust:AdjustSizes:B:",
            b"interpmin:MinInterpolationFactor:F:",
            b"linear:LinearInterpolation:I:",
            b"center:CenterWeight:F:",
            b"box:BoxSizeScaled:I:",
            b"threshold:ThresholdForAveraging:F:",
            b"store:StorageThreshold:F:",
            b"fallback:FallbackThresholds:IP:",
            b"bkgd:BackgroundGroups:F:",
            b"annulus:AnnulusPercentile:F:",
            b"peakmin:MinRelativeStrength:F:",
            b"spacing:MinSpacing:F:",
            b"sections:SectionsToDo:LI:",
            b"maxsec:MaxSectionsPerAnalysis:I:",
            b"remake:RemakeModelBead:B:",
            b"guess:MinGuessNumBeads:I:",
            b"measure:MeasureToUse:I:",
            b"kernel:KernelSigma:F:",
            b"rad1:FilterRadius1:F:",
            b"rad2:FilterRadius2:F:",
            b"sig1:FilterSigma1:F:",
            b"sig2:FilterSigma2:F:",
            b"verbose:VerboseKeys:CH:",
            b"dump:DumpHistogramFile:FN:",
            b"param:ParameterFile:PF:",
            b"help:usage:B:",
        ];

        // Startup with fallback
        let argv_bytes: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
        pip_read_or_parse_options(
            argv_bytes.len() as i32,
            &argv_bytes,
            &options,
            num_options,
            progname.as_bytes(),
            3,
            1,
            1,
            &mut num_opt_args,
            &mut num_non_opt_args,
            Some(imod_usage_header_for_pip),
        );
        if pip_get_boolean(b"usage", &mut co_flag) == 0 {
            pip_print_help(progname.as_bytes(), 0, 1, 1);
            exit(0);
        }

        // Get input file
        let mut name_bytes: Vec<u8> = Vec::new();
        if pip_get_in_out_file(b"InputImageFile", 0, &mut name_bytes) != 0 {
            exit_error(b"No input image file specified");
        }
        let filename = String::from_utf8_lossy(&name_bytes).into_owned();
        self.m_in_fp = ii_fopen(filename.as_bytes(), "rb");
        let Some(in_fp) = self.m_in_fp.as_mut() else {
            exit_error_fmt!("Opening input image file %s", CArg::Str(&filename));
        };
        // Fixed in translation: the source prints the already-freed name.
        if mrc_head_read(in_fp, &mut self.m_in_head) != 0 {
            exit_error_fmt!("Reading header of image file %s", CArg::Str(&filename));
        }
        xscale = 1.;
        yscale = 1.;
        zscale = 1.;
        if self.m_in_head.mx != 0 && self.m_in_head.xlen != 0. {
            xscale = self.m_in_head.xlen / self.m_in_head.mx as f32;
        }
        if self.m_in_head.my != 0 && self.m_in_head.ylen != 0. {
            yscale = self.m_in_head.ylen / self.m_in_head.my as f32;
        }
        if self.m_in_head.mz != 0 && self.m_in_head.zlen != 0. {
            zscale = self.m_in_head.zlen / self.m_in_head.mz as f32;
        }

        // Check if it is the correct data type and set slice type
        let slice_mode = slice_mode_if_real(self.m_in_head.mode);
        if slice_mode < 0 {
            exit_error_fmt!(
                "File mode is %d; only byte, short, float allowed",
                CArg::Int(self.m_in_head.mode as i64)
            );
        }
        self.m_nx_in = self.m_in_head.nx;
        self.m_ny_in = self.m_in_head.ny;

        // Read area model
        let mut str_val: Vec<u8> = Vec::new();
        if pip_get_string(b"AreaModel", &mut str_val) == 0 {
            let name = String::from_utf8_lossy(&str_val).into_owned();
            let Ok(area) = imod_read(&name) else {
                exit_error_fmt!("Reading area model %s", CArg::Str(&name));
            };
            if area.obj.is_empty() || area.obj[0].cont.is_empty() {
                exit_error(b"No contours in object 1 of area model");
            }
            self.m_area_mod = Some(area);
            pip_get_boolean(b"ExcludeInsideAreas", &mut self.m_exclude_areas);
        }

        // Check if this is a run just to find the area
        if pip_get_integer(b"QueryAreaOnSection", &mut iz) == 0 {
            if self.m_area_mod.is_none() {
                exit_error(b"You must enter an area model to use -query");
            }
            self.area_cont_list_check_err(iz);
            self.m_bead_size = 0.;
            let area = self.m_area_mod.as_ref().unwrap();
            for co in 0..self.m_num_area_cont as usize {
                self.m_bead_size +=
                    imod_contour_area(Some(&area.obj[0].cont[self.m_area_conts[co] as usize]));
            }
            if self.m_exclude_areas != 0 {
                self.m_bead_size =
                    (self.m_in_head.nx * self.m_in_head.ny) as f32 - self.m_bead_size;
            }

            // Autofidseed is looking for 'Area (megapixels)' on any line
            let area_mp = self.m_bead_size as f64 * 1.0e-6;
            printf!(
                "Area (megapixels) included in analysis = %.3f",
                CArg::Dbl(area_mp)
            );
            record(FindbeadsReport::Area(area_mp));
            exit(0);
        }

        // Get output file
        let mut out_bytes: Vec<u8> = Vec::new();
        if pip_get_in_out_file(b"OutputModelFile", 1, &mut out_bytes) != 0 {
            exit_error(b"No output model file specified");
        }
        let out_model = String::from_utf8_lossy(&out_bytes).into_owned();

        // Read reference model
        str_val.clear();
        if pip_get_string(b"ReferenceModel", &mut str_val) == 0 {
            let name = String::from_utf8_lossy(&str_val).into_owned();
            let Ok(rmod) = imod_read(&name) else {
                exit_error_fmt!("Reading reference model %s", CArg::Str(&name));
            };
            if pip_get_integer(b"BoundaryObject", &mut bound_obj) == 0 {
                if bound_obj < 1 || bound_obj > rmod.obj.len() as i32 {
                    exit_error_fmt!(
                        "Boundary object number %d is out of bounds (model has %d objects)",
                        CArg::Int(bound_obj as i64),
                        CArg::Int(rmod.obj.len() as i64)
                    );
                }
            }
            refmod = Some(rmod);
        }

        // Read existing model
        str_val.clear();
        if pip_get_string(b"AddToModel", &mut str_val) == 0 {
            let name = String::from_utf8_lossy(&str_val).into_owned();
            let Ok(amod) = imod_read(&name) else {
                exit_error_fmt!("Reading model to append to: %s", CArg::Str(&name));
            };
            imod = amod;
            let saved_ref = imod.ref_image;
            if saved_ref.is_some() {
                has_ref = true;
            }
            if imod_set_ref_image(&mut imod, &self.m_in_head) != 0 {
                exit_error(b"Setting refImage structure in model");
            }
            if has_ref {
                let saved = saved_ref.unwrap();
                let rimg = imod.ref_image.as_mut().unwrap();
                rimg.otrans = saved.ctrans;
                rimg.orot = saved.crot;
                rimg.oscale = saved.cscale;
            }
            let ref_copy = imod.ref_image.unwrap();
            if imod_trans_from_ref_image(&mut imod, &ref_copy, bin_scale) != 0 {
                exit_error(b"Transforming existing model to match image file");
            }

            // Find out if replacing points and get tilt angles if needed
            pip_get_float(b"ReplaceAboveAngle", &mut replace_angle);
            pip_get_integer(b"FillInMissingPoints", &mut fill_in_added);
            if replace_angle > 0. || fill_in_added > 0 {
                str_val.clear();
                if pip_get_string(b"TiltAngleFile", &mut str_val) != 0 {
                    exit_error(
                        b"Tilt angles must be entered when filling in missing points or if the angle above which to replace points is > 0",
                    );
                }
                let tname = String::from_utf8_lossy(&str_val).into_owned();
                self.m_tilt_angles = vec![0.; (self.m_in_head.nz + 10) as usize];
                let mut nz = self.m_in_head.nz;
                read_tilt_file(
                    &mut nz,
                    &tname,
                    &mut self.m_tilt_angles,
                    self.m_in_head.nz + 10,
                );
                self.m_in_head.nz = nz;
            }

            // Get thresholds for objects if any
            self.m_obj_thresh = vec![-1.0e30; imod.obj.len()];
            for ob in 0..imod.obj.len() {
                if istore_get_min_max(
                    &imod.obj[ob].store,
                    imod.obj[ob].cont.len() as i32,
                    GEN_STORE_MINMAX1,
                    &mut ob_min,
                    &mut ob_max,
                ) != 0
                {
                    self.m_obj_thresh[ob] =
                        ((imod.obj[ob].valblack as f32 * (ob_max - ob_min)) as f64 / 255.
                            + ob_min as f64) as f32;
                }
                let cview = imod.cview;
                if cview > 0 && (cview as usize) < imod.view.len() {
                    let obj = imod.obj[ob].clone();
                    if let Some(ov) = imod.view[cview as usize].objview.get_mut(ob) {
                        imod_objview_from_object(&obj, ov);
                    }
                }
            }

            // The model is likely on a different stack from the current one,
            // so say it is new
            imod.flags |= IMODF_NEW_TO_3DMOD;
        } else {
            // Or create a model
            let Some(nmod) = imod_new() else {
                exit_error(b"Creating output model");
            };
            imod = nmod;
            if xscale != 1.0 {
                imod.pixsize = xscale / 10.;
                imod.units = IMOD_UNIT_NM;
            }
        }
        self.m_num_obj_orig = imod.obj.len() as i32;

        // Set up dump file
        str_val.clear();
        if pip_get_string(b"DumpHistogramFile", &mut str_val) == 0 {
            let name = String::from_utf8_lossy(&str_val).into_owned();
            imod_backup_file(&name);
            self.m_dump_fp = ImodFile::open(&name, "w");
            if self.m_dump_fp.is_none() {
                exit_error_fmt!("Failed to open file for histograms, %s", CArg::Str(&name));
            }
        }

        // Get other parameters
        pip_get_float(b"ScaledSize", &mut self.m_scaled_size);
        // exitError("You must enter a scaled size for the filtering");
        if pip_get_float(b"BeadSize", &mut self.m_bead_size) != 0 {
            exit_error(b"You must enter a bead size");
        }
        pip_get_boolean(b"AdjustSizes", &mut self.m_adjust_sizes);
        pip_get_boolean(b"LightBeads", &mut self.m_light_beads);
        pip_get_integer(b"LinearInterpolation", &mut self.m_linear_interp);
        pip_get_boolean(b"RemakeModelBead", &mut remake_model_bead);
        pip_get_float(b"ThresholdForAveraging", &mut self.m_threshold);
        pip_get_float(b"StorageThreshold", &mut self.m_peak_thresh);
        pip_get_two_integers(
            b"FallbackThresholds",
            &mut self.m_average_fallback,
            &mut self.m_storage_fallback,
        );
        pip_get_float(b"CenterWeight", &mut self.m_center_weight);
        pip_get_integer(b"BackgroundGroups", &mut self.m_num_groups);
        pip_get_float(b"MinInterpolationFactor", &mut self.m_min_interp);
        pip_get_float(b"MinSpacing", &mut self.m_min_spacing);
        pip_get_integer(b"MinGuessNumBeads", &mut num_guess);
        pip_get_integer(b"MeasureToUse", &mut self.m_measure_to_use);
        pip_get_float(b"AnnulusPercentile", &mut self.m_annulus_pctile);
        pip_get_float(b"KernelSigma", &mut kernel_sigma);
        pip_get_float(b"FilterRadius1", &mut radius1);
        pip_get_float(b"FilterRadius2", &mut radius2);
        pip_get_float(b"FilterSigma1", &mut sigma1);
        pip_get_float(b"FilterSigma2", &mut sigma2);
        str_val.clear();
        if pip_get_string(b"VerboseKeys", &mut str_val) == 0 {
            self.m_vkeys = Some(String::from_utf8_lossy(&str_val).into_owned());
        }
        pip_get_two_integers(b"TrackAndZDebug", &mut debug_track, &mut debug_z);
        self.m_num_groups = b3dmin!(MAX_GROUPS as i32, self.m_num_groups);
        if self.m_light_beads != 0 && self.m_annulus_pctile >= 0. {
            self.m_annulus_pctile = (1. - self.m_annulus_pctile as f64) as f32;
        }

        if self.m_adjust_sizes != 0 && self.m_bead_size < self.m_min_size_for_adjust {
            self.m_adjust_sizes = 0;
            printf!(
                "WARNING: imodfindbeads - Bead size is below limit for finding diameter; no adjustment will be done\n"
            );
        }
        let size_orig = self.m_bead_size;

        // If doing automatic thresholds and no peak limit entered, drop it to
        // allow more histogram to be built
        if pip_get_float(b"MinRelativeStrength", &mut self.m_min_relative_peak) != 0
            && self.m_threshold < 0.
        {
            self.m_min_relative_peak = (self.m_min_relative_peak as f64 / 2.) as f32;
        }

        // Make default box size and list of sections to do
        self.m_box_scaled_orig = b3dnint!(3. * self.m_scaled_size + 4.);
        pip_get_integer(b"BoxSizeScaled", &mut self.m_box_scaled_orig);
        str_val.clear();
        if pip_get_string(b"SectionsToDo", &mut str_val) == 0 {
            match parselist(&String::from_utf8_lossy(&str_val)) {
                Ok(list) => self.m_zlist = list,
                Err(_) => exit_error(b"Bad entry in list of sections to do"),
            }
            nzout = self.m_zlist.len() as i32;
        } else {
            nzout = self.m_in_head.nz;
            self.m_zlist = (0..nzout.max(0)).collect();
        }
        for i in 0..nzout as usize {
            if self.m_zlist[i] < 0 || self.m_zlist[i] >= self.m_in_head.nz {
                exit_error_fmt!(
                    "Section # %d is out of range",
                    CArg::Int(self.m_zlist[i] as i64)
                );
            }
        }

        // Get shifts if option is entered
        str_val.clear();
        if pip_get_string(b"PrealignTransformFile", &mut str_val) == 0 {
            let name = String::from_utf8_lossy(&str_val).into_owned();
            let Some(mut xffp) = ImodFile::open(&name, "r") else {
                exit_error_fmt!(
                    "Failed to open prealignment transform file %s\n",
                    CArg::Str(&name)
                );
            };
            let nz = self.m_in_head.nz.max(0) as usize;
            let mut xshift = vec![0f32; nz];
            let mut yshift = vec![0f32; nz];
            binning = 1;
            pip_get_integer(b"ImagesAreBinned", &mut binning);
            if binning <= 0 {
                exit_error(b"Binning entry must be positive");
            }
            let mut buf = [0u8; MAX_LINE];
            for i in 0..nz {
                let ix = fgetline(&mut xffp, &mut buf, MAX_LINE as i32);
                if ix == -1 {
                    // Fixed in translation: the source prints a freed name.
                    exit_error_fmt!("Reading prealignment transform file %s\n", CArg::Str(&name));
                }
                if ix == -2 {
                    break;
                }
                let end = buf.iter().position(|&c| c == 0).unwrap_or(MAX_LINE);
                let text = String::from_utf8_lossy(&buf[..end]).into_owned();
                let (mut a, mut b) = (0f32, 0f32);
                sscanf(
                    &text,
                    "%f %f %f %f %f %f",
                    &mut [
                        ScanArg::Flt(&mut xtmp),
                        ScanArg::Flt(&mut ytmp),
                        ScanArg::Flt(&mut xcen),
                        ScanArg::Flt(&mut ycen),
                        ScanArg::Flt(&mut a),
                        ScanArg::Flt(&mut b),
                    ],
                );
                // sscanf leaves an unconverted target unchanged; the targets
                // start at 0 as the source's cleared arrays do.
                xshift[i] = a / binning as f32;
                yshift[i] = b / binning as f32;
                if ix < 0 {
                    break;
                }
            }
            drop(xffp);
            self.m_align_xshift = Some(xshift);
            self.m_align_yshift = Some(yshift);
        }

        // Figure out division into separate runs and set up for caching read
        // slices
        let mut max_sec = nzout;
        pip_get_integer(b"MaxSectionsPerAnalysis", &mut max_sec);
        if max_sec <= 0 {
            exit_error(b"Maximum number of sections per analysis must be positive");
        }
        let num_runs = (nzout + max_sec - 1) / max_sec;
        let num_zper_run = nzout / num_runs;
        let runs_adding_one = nzout % num_runs;
        let max_zper_run = num_zper_run + if runs_adding_one != 0 { 1 } else { 0 };
        self.m_use_slice_cache =
            (max_zper_run * self.m_nx_in * self.m_ny_in * 4) as f64 / 1.0e6 < cache_limit_mb as f64;
        if self.m_use_slice_cache {
            self.m_cached_slices = (0..max_zper_run).map(|_| None).collect();
        }

        if num_runs > 1 && refmod.is_some() {
            exit_error(
                b"All sections must be analyzed together when comparing with reference model",
            );
        }

        // Open file for filtered images
        str_val.clear();
        if pip_get_string(b"FilteredImageFile", &mut str_val) == 0 {
            let name = String::from_utf8_lossy(&str_val).into_owned();
            if imod_backup_file(&name) != 0 {
                printf!(
                    "WARNING: %s - Error renaming existing image file %s\n",
                    CArg::Str(progname),
                    CArg::Str(&name)
                );
            }
            outfp = ii_fopen(name.as_bytes(), "wb");
            if outfp.is_none() {
                exit_error_fmt!("Opening output image file %s", CArg::Str(&name));
            }
        }
        pip_done();

        imod_backup_file(&out_model);
        let Some(mut model_fp) = ImodFile::open(&out_model, "wb") else {
            exit_error_fmt!("Opening output model %s", CArg::Str(&out_model));
        };

        self.setup_size_dependent_vars();

        let mut list_start = vec![0i32; (nzout + 2) as usize];

        self.profile("Finished startup tasks");
        let mut iz_start: i32 = 0;
        for irun in 0..num_runs {
            let iz_end = iz_start + num_zper_run + if irun < runs_adding_one { 1 } else { 0 };
            printf!(
                "\nAnalyzing group of sections starting with %d, ending with %d\n",
                CArg::Int(self.m_zlist[iz_start as usize] as i64),
                CArg::Int(self.m_zlist[(iz_end - 1) as usize] as i64)
            );
            self.m_min_guess = num_guess * (iz_end - iz_start);

            // Construct a bead
            if irun == 0 || remake_model_bead != 0 {
                make_model_bead(self.m_box_size, self.m_bead_size, &mut self.m_full_bead);
            }
            npass = if self.m_threshold != 0. { 2 } else { 1 };
            self.m_list_size = 0;

            // Loop on one or two passes
            let mut ipass = 1;
            while ipass <= npass {
                self.m_num_peaks = 0;
                self.m_peak_list.clear();
                num_eliminated = 0;
                num_below_min = 0;
                self.m_peak_max = -1.0e30;

                // On the last pass, set up the output file
                if outfp.is_some() && ipass == npass {
                    mrc_head_new(
                        &mut outhead,
                        self.m_nx_out,
                        self.m_ny_out,
                        nzout,
                        MRC_MODE_FLOAT,
                    );
                    mrc_head_label_cp(&self.m_in_head, &mut outhead);
                    mrc_head_label(&mut outhead, b"imodfindbeads: Scaled and Sobel filtered");

                    // Set scale and origin in new header to display the match
                    // input data
                    outhead.xlen = self.m_nx_out as f32 * xscale * self.m_scale_factor;
                    outhead.ylen = self.m_ny_out as f32 * yscale * self.m_scale_factor;
                    outhead.zlen = nzout as f32 * zscale;
                    outhead.xorg -= self.m_xoffset * xscale;
                    outhead.yorg -= self.m_yoffset * yscale;
                    outhead.zorg -= self.m_zlist[0] as f32 * zscale;
                    outhead.amin = 1.0e30;
                    outhead.amax = -1.0e30;
                    outhead.amean = 0.;
                }

                // Scale down and filter the bead
                scaled_sobel(
                    Some(&self.m_full_bead),
                    self.m_box_size,
                    self.m_box_size,
                    self.m_scale_factor,
                    self.m_min_interp,
                    self.m_linear_interp,
                    self.m_center_weight,
                    Some(&mut self.m_filt_bead),
                    &mut nxtmp,
                    &mut nytmp,
                    &mut x_bead_ofs,
                    &mut y_bead_ofs,
                );
                self.m_bead_cen_ofs = ((self.m_box_size as f64 / 2. - x_bead_ofs as f64)
                    / self.m_scale_factor as f64
                    - (self.m_box_scaled / 2) as f64) as f32;

                // Split it into 4 corners of the big array and take the FFT
                slice_split_fill(
                    &self.m_filt_bead,
                    self.m_box_scaled,
                    self.m_box_scaled,
                    &mut self.m_split_bead,
                    self.m_nxp_dim,
                    self.m_nx_pad,
                    self.m_ny_pad,
                    0,
                    0.,
                );
                todfft(
                    &mut self.m_split_bead,
                    self.m_nx_pad,
                    self.m_ny_pad,
                    forward,
                );
                xcorr_set_ctf(
                    sigma1,
                    sigma2,
                    radius1,
                    radius2,
                    &mut ctf,
                    self.m_nx_pad,
                    self.m_ny_pad,
                    &mut ctf_delta,
                );
                if ctf_delta != 0. {
                    xcorr_filter_part(
                        FilterIn::InPlace,
                        &mut self.m_split_bead,
                        self.m_nx_pad,
                        self.m_ny_pad,
                        &ctf,
                        ctf_delta,
                    );
                }
                self.profile("Set up the bead");

                // Loop on images
                for indz in iz_start..iz_end {
                    iz = self.m_zlist[indz as usize];
                    list_start[(indz - iz_start) as usize] = self.m_num_peaks;
                    self.m_num_area_cont = 0;
                    if self.m_area_mod.is_some() {
                        self.area_cont_list_check_err(iz);
                    }

                    // Create a slice and read into it as floats.  A cached
                    // slice is taken out of the cache for the duration and
                    // put back below.
                    let cache_ind = (indz - iz_start) as usize;
                    let base: Islice = if ipass == 1 || !self.m_use_slice_cache {
                        let s = self.read_slice_as_float(iz);
                        self.profile("Read slice");
                        s
                    } else {
                        self.m_cached_slices[cache_ind].take().unwrap()
                    };

                    // Do kernel filtering
                    let filtered: Option<Islice> = if kernel_sigma > 0. {
                        scaled_gaussian_kernel(
                            &mut kernel,
                            &mut ndat,
                            KERNEL_MAXSIZE,
                            kernel_sigma,
                        );
                        let Some(sclsl) = slice_mat_filter(&base, &kernel, ndat, None) else {
                            exit_error(b"Failed to get memory for kernel filtered slice");
                        };
                        self.profile("kernel filtered");
                        Some(sclsl)
                    } else {
                        None
                    };
                    let sl: &Islice = filtered.as_ref().unwrap_or(&base);

                    // Filter it, write it on last pass if requested
                    scaled_sobel(
                        Some(sl.data.f()),
                        self.m_nx_in,
                        self.m_ny_in,
                        self.m_scale_factor,
                        self.m_min_interp,
                        self.m_linear_interp,
                        self.m_center_weight,
                        Some(&mut self.m_filt_slice),
                        &mut nxtmp,
                        &mut nytmp,
                        &mut xtmp,
                        &mut ytmp,
                    );
                    self.profile("Sobel filtered");

                    // Pad into array and correlate it
                    slice_taper_out_pad(
                        PadIn::Float(&self.m_filt_slice),
                        SLICE_MODE_FLOAT,
                        self.m_nx_out,
                        self.m_ny_out,
                        &mut self.m_corr_slice,
                        self.m_nxp_dim,
                        self.m_nx_pad,
                        self.m_ny_pad,
                        0,
                        0.,
                    );
                    todfft(
                        &mut self.m_corr_slice,
                        self.m_nx_pad,
                        self.m_ny_pad,
                        forward,
                    );
                    conjugate_product(
                        &mut self.m_corr_slice,
                        &self.m_split_bead,
                        self.m_nx_pad,
                        self.m_ny_pad,
                    );
                    todfft(
                        &mut self.m_corr_slice,
                        self.m_nx_pad,
                        self.m_ny_pad,
                        inverse,
                    );
                    self.profile("Correlated");

                    // Write slice on last pass
                    if let Some(ofp) = outfp.as_mut() {
                        if ipass == npass {
                            if self.m_center_weight == 0. {
                                // Fixed in translation: the source indexes
                                // `mCorrSlice[ix + mNxpDim * iy - (mNxPad -
                                // mNxOut) / 2]`, which reads before the start
                                // of the array on the first row.  Defined as
                                // the correlation pixel that lines up with
                                // filtered pixel (ix, iy), the mapping
                                // `searchCorrelationPeaks` uses (`ixofs = ix -
                                // (mNxPad - mNxOut) / 2`, same in Y).
                                let xofs = (self.m_nx_pad - self.m_nx_out) / 2;
                                let yofs = (self.m_ny_pad - self.m_ny_out) / 2;
                                for iy in 0..self.m_ny_out {
                                    for ix in 0..self.m_nx_out {
                                        self.m_write_slice[(ix + self.m_nx_out * iy) as usize] =
                                            self.m_corr_slice[(ix
                                                + xofs
                                                + self.m_nxp_dim * (iy + yofs))
                                                as usize];
                                    }
                                }
                            }
                            let write_slice: &[f32] = if self.m_center_weight != 0. {
                                &self.m_filt_slice
                            } else {
                                &self.m_write_slice
                            };
                            let nwrite = (self.m_nx_out * self.m_ny_out) as usize;
                            let bytes: Vec<u8> = write_slice[..nwrite]
                                .iter()
                                .flat_map(|v| v.to_ne_bytes())
                                .collect();
                            if mrc_write_slice(&bytes, ofp, &mut outhead, indz - iz_start, b'z')
                                != 0
                            {
                                exit_error_fmt!(
                                    "Writing filtered image for section %d",
                                    CArg::Int(iz as i64)
                                );
                            }
                            for iy in 0..self.m_ny_out {
                                let mut tsum: f32 = 0.;
                                for ix in 0..self.m_nx_out {
                                    let val = write_slice[(ix + iy * self.m_nx_out) as usize];
                                    tsum += val;
                                    outhead.amin = b3dmin!(outhead.amin, val);
                                    outhead.amax = b3dmax!(outhead.amax, val);
                                }
                                outhead.amean +=
                                    tsum / (self.m_nx_out as f32 * self.m_ny_out as f32);
                            }
                        }
                    }

                    // Search for all peaks in the correlation
                    self.search_correlation_peaks(sl, iz);
                    self.profile("found peaks");

                    // The kernel-filtered slice is freed here; the read slice
                    // goes back into the cache if caching (source:
                    // `if (!mUseSliceCache || kernelSigma > 0.) sliceFree(sl)`).
                    drop(filtered);
                    if self.m_use_slice_cache {
                        self.m_cached_slices[cache_ind] = Some(base);
                    }
                }
                list_start[(iz_end - iz_start) as usize] = self.m_num_peaks;

                // Determine a scaling of peaks by background intensities
                self.analyze_background_groups();
                self.profile("Analyzed groups");

                // Normalize peak values and eliminate ones below minimum: zero
                // out ccc
                for j in 0..self.m_num_peaks as usize {
                    self.m_peak_list[j].peak /= self.m_peak_max;
                    if self.m_peak_list[j].ccc != 0.
                        && self.m_peak_list[j].peak < self.m_min_relative_peak
                    {
                        self.m_peak_list[j].ccc = 0.;
                        num_below_min += 1;
                    }
                }

                let critsq = self.m_min_dist * self.m_min_dist;

                // Eliminate peaks that are too close by zeroing out ccc
                let min_dist = self.m_min_dist;
                for indz in 0..(iz_end - iz_start) as usize {
                    let mut i = list_start[indz];
                    while i < list_start[indz + 1] {
                        let iu = i as usize;
                        if self.m_peak_list[iu].ccc == 0. {
                            i += 1;
                            continue;
                        }
                        xcen = self.m_peak_list[iu].xcen;
                        ycen = self.m_peak_list[iu].ycen;
                        let mut jstr = i - 1;
                        let mut jend = list_start[indz];
                        let mut jdir = -1;
                        while jdir <= 1 && self.m_peak_list[iu].ccc != 0. {
                            let mut j = jstr;
                            while j * jdir <= jend * jdir {
                                let ju = j as usize;
                                let dy = self.m_peak_list[ju].ycen - ycen;
                                if self.m_peak_list[ju].ccc != 0. && jdir as f32 * dy > min_dist {
                                    break;
                                }
                                if self.m_peak_list[ju].ccc == 0. {
                                    j += jdir;
                                    continue;
                                }
                                let dx = self.m_peak_list[ju].xcen - xcen;
                                if dx >= -min_dist && dx <= min_dist {
                                    let distsq = dx * dx + dy * dy;
                                    if distsq <= critsq {
                                        num_eliminated += 1;

                                        // Found a peak too close
                                        // Eliminate current peak and break out
                                        // of loop if it is weaker
                                        if self.m_peak_list[ju].ccc > self.m_peak_list[iu].ccc {
                                            self.m_peak_list[iu].ccc = 0.;
                                            break;
                                        } else {
                                            // Otherwise eliminate the other
                                            // peak and continue
                                            self.m_peak_list[ju].ccc = 0.;
                                        }
                                    }
                                }
                                j += jdir;
                            }
                            jstr = i + 1;
                            jend = list_start[indz + 1] - 1;
                            jdir += 2;
                        }
                        i += 1;
                    }
                }
                self.profile("Eliminated close points");

                if npass > 1 {
                    printf!("Pass %d: ", CArg::Int(ipass as i64));
                }
                num_peaks_left = self.m_num_peaks - num_eliminated - num_below_min;
                printf!(
                    "%d peaks found. %d eliminated as too weak, %d too close, %d remaining\n",
                    CArg::Int(self.m_num_peaks as i64),
                    CArg::Int(num_below_min as i64),
                    CArg::Int(num_eliminated as i64),
                    CArg::Int(num_peaks_left as i64)
                );

                // If doing two passes, now average the beads
                if ipass < npass {
                    self.average_beads(iz_start, iz_end);
                    if self.m_adjust_sizes != 0
                        && ipass == 1
                        && b3dmax!(size_orig, self.m_bead_size)
                            / b3dmin!(size_orig, self.m_bead_size)
                            >= self.m_min_size_change_for_redo
                    {
                        printf!(
                            "Adding another pass because size changed more than %.0f%%\n",
                            CArg::Dbl(100. * (self.m_min_size_change_for_redo as f64 - 1.))
                        );
                        npass += 1;
                    }
                    self.profile("Averaged");
                }
                ipass += 1;
            }

            // Done with the cached slices now
            if self.m_use_slice_cache {
                for indz in iz_start..iz_end {
                    self.m_cached_slices[(indz - iz_start) as usize] = None;
                }
            }

            // After first set of sections, stop adjusting sizes
            self.m_adjust_sizes = 0;

            let ixst = iz_start;
            iz_start = iz_end;
            if num_peaks_left == 0 {
                continue;
            }

            // Determine automatic threshold for putting points out
            let mut thresh_use = self.m_peak_thresh;
            if self.m_peak_thresh <= 0. {
                thresh_use = self.find_storage_threshold(&mut hist_dip, &mut line);
            }

            // Eliminate duplicates from original objects by brute force
            num_eliminated = 0;
            num_replaced = 0;
            let max_replace_dist_sq =
                ((replace_dist_frac * self.m_bead_size) as f64).powf(2.) as f32;
            let critsq = self.m_min_dist * self.m_min_dist;
            let min_dist = self.m_min_dist;

            // Loop on Z,
            for indz in ixst..iz_end {
                iz = self.m_zlist[indz as usize];

                // Loop on peaks at this Z, set flag for replacing for each point
                // if angle qualifies and peak is not below histogram dip
                for i in list_start[(indz - ixst) as usize]..list_start[(indz + 1 - ixst) as usize]
                {
                    let iu = i as usize;
                    if self.m_peak_list[iu].ccc == 0. {
                        continue;
                    }
                    let replacing = (replace_angle == 0.
                        || (replace_angle > 0.
                            && self.m_tilt_angles[iz as usize].abs() >= replace_angle))
                        && !(hist_dip > -1. && self.m_peak_list[iu].peak < hist_dip);
                    let x_peak = self.m_peak_list[iu].xcen + self.m_avg_xoffset;
                    let y_peak = self.m_peak_list[iu].ycen + self.m_avg_yoffset;

                    // Loop through model to find closest point(s)
                    let mut dist_min: f32 = 1.0e30;
                    for ob in 0..self.m_num_obj_orig as usize {
                        if iobj_close(imod.obj[ob].flags) != 0 {
                            continue;
                        }
                        for co in 0..imod.obj[ob].cont.len() {
                            // If there is an object threshold, look up the value
                            // for contour and ignore it if it is below threshold
                            if self.contour_is_below_threshold(&imod, ob, co) {
                                continue;
                            }

                            // Loop on points
                            let cont = &imod.obj[ob].cont[co];
                            for pt in 0..cont.pts.len() {
                                if b3dnint!(cont.pts[pt].z) != iz {
                                    continue;
                                }
                                xcen = cont.pts[pt].x;
                                ycen = cont.pts[pt].y;
                                let dy = y_peak - ycen;
                                if dy < -min_dist || dy > min_dist {
                                    continue;
                                }
                                let dx = x_peak - xcen;
                                if dx < -min_dist || dx > min_dist {
                                    continue;
                                }
                                let distsq = dx * dx + dy * dy;
                                if distsq <= critsq && distsq < dist_min {
                                    // If this could be the closest model point so
                                    // far, loop on ths peaks at this Z to make
                                    // sure there is not a peak closer to the
                                    // model point
                                    let mut closest = true;
                                    for j in list_start[(indz - ixst) as usize]
                                        ..list_start[(indz + 1 - ixst) as usize]
                                    {
                                        let ju = j as usize;
                                        if i == j || self.m_peak_list[ju].ccc == 0. {
                                            continue;
                                        }
                                        let dxj =
                                            self.m_peak_list[ju].xcen + self.m_avg_xoffset - xcen;
                                        if dxj < -min_dist || dxj > min_dist {
                                            continue;
                                        }
                                        let dyj =
                                            self.m_peak_list[ju].ycen + self.m_avg_yoffset - ycen;
                                        if dyj < -min_dist || dyj > min_dist {
                                            continue;
                                        }
                                        if dxj * dxj + dyj * dyj < distsq {
                                            closest = false;
                                            break;
                                        }
                                    }

                                    // If closest, set new minimum and record
                                    // position
                                    if closest {
                                        dist_min = distsq;
                                        if replacing && dist_min <= max_replace_dist_sq {
                                            best_ob = ob;
                                            best_co = co;
                                            best_pt = pt;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // If a qualifying point was found, eliminate and replace if
                    // replacing
                    if dist_min <= critsq {
                        num_eliminated += 1;
                        self.m_peak_list[iu].ccc = 0.;
                        if replacing && dist_min <= max_replace_dist_sq {
                            num_replaced += 1;
                            let cont = &mut imod.obj[best_ob].cont[best_co];
                            cont.pts[best_pt].x = x_peak;
                            cont.pts[best_pt].y = y_peak;
                        }
                    }
                }
            }
            if num_eliminated != 0 {
                printf!(
                    "%d peaks were eliminated as too close to existing points in model\n",
                    CArg::Int(num_eliminated as i64)
                );
            }
            if num_replaced != 0 {
                printf!(
                    "%d of these replaced existing point positions\n",
                    CArg::Int(num_replaced as i64)
                );
            }

            // Count points to be stored
            let mut num_to_save = 0;
            for i in 0..self.m_num_peaks as usize {
                if self.m_peak_list[i].ccc != 0. && self.m_peak_list[i].peak >= thresh_use {
                    num_to_save += 1;
                }
            }
            if self.m_peak_thresh > 0. {
                printf!(
                    "%d peaks above threshold of %.3f are being stored in model\n",
                    CArg::Int(num_to_save as i64),
                    CArg::Dbl(self.m_peak_thresh as f64)
                );
                record(FindbeadsReport::PeaksAboveStorageThreshold {
                    num: num_to_save,
                    thresh: self.m_peak_thresh,
                });
            }
            if num_to_save == 0 {
                exit_error(b"There are no peaks available for saving");
            }

            if imod_new_object(&mut imod) != 0 {
                exit_error(b"Creating new object in model");
            }
            let first_obj = if self.m_num_obj_orig != 0 {
                Some((
                    imod.obj[0].symbol,
                    imod.obj[0].symsize,
                    imod.obj[0].pdrawsize,
                    imod.obj[0].flags,
                ))
            } else {
                None
            };
            let obj = imod.obj.last_mut().unwrap();
            obj.flags |= IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_USE_VALUE;
            obj.matflags2 |= (MATFLAGS2_CONSTANT | MATFLAGS2_SKIP_LOW) as u8;
            obj.symbol = IOBJ_SYM_CIRCLE;
            obj.symsize = 7;
            if let Some((symbol, symsize, pdrawsize, flags)) = first_obj {
                obj.symbol = symbol;
                obj.symsize = symsize;
                obj.pdrawsize = pdrawsize;
                if flags & IMOD_OBJFLAG_PNT_ON_SEC != 0 {
                    obj.flags |= IMOD_OBJFLAG_PNT_ON_SEC;
                }
            }
            let Some(conts) = imod_contours_new(num_to_save) else {
                exit_error(b"Creating contours in model");
            };
            obj.cont = conts;
            let mut ix = 0usize;
            let mut ccc_min: f32 = 10000.;
            let mut ccc_max: f32 = -10000.;
            for i in 0..self.m_num_peaks as usize {
                if self.m_peak_list[i].ccc == 0. || self.m_peak_list[i].peak < thresh_use {
                    continue;
                }

                // Empirically, addition is correct, for + offset, bead is to
                // right, extraction center thus position is to the left
                pnt.x = self.m_peak_list[i].xcen + self.m_avg_xoffset;
                pnt.y = self.m_peak_list[i].ycen + self.m_avg_yoffset;
                pnt.z = self.m_peak_list[i].iz as f32;
                if imod_point_append(&mut obj.cont[ix], pnt) == 0 {
                    exit_error(b"Adding point to contour");
                }

                let store = Istore {
                    type_: GEN_STORE_VALUE1,
                    flags: GEN_STORE_FLOAT << 2,
                    index: StoreUnion::from_i(ix as i32),
                    value: StoreUnion::from_f(self.m_peak_list[i].peak),
                };
                ix += 1;
                if istore_insert(&mut obj.store, store) != 0 {
                    exit_error(b"Could not add general storage item");
                }
                ccc_min = b3dmin!(ccc_min, self.m_peak_list[i].peak);
                ccc_max = b3dmax!(ccc_max, self.m_peak_list[i].peak);
            }
            if !line.is_empty() {
                printf!("%d %s", CArg::Int(obj.cont.len() as i64), CArg::Str(&line));
                record(FindbeadsReport::TotalPeaksStored {
                    num: obj.cont.len() as i32,
                    text: line.clone(),
                });
            }
            if istore_add_min_max(&mut obj.store, GEN_STORE_MINMAX1, ccc_min, ccc_max) != 0 {
                exit_error(b"Could not add general storage item");
            }
            obj.valwhite = 255;
            obj.valblack = 0;
            let mut dx: f32 = -1.0e30;
            if thresh_use > -9999. && hist_dip > ccc_min {
                obj.valblack =
                    b3dnint!(255. * (hist_dip - ccc_min) as f64 / (ccc_max - ccc_min) as f64) as u8;
                dx = hist_dip;
            }
            if fill_in_added > 0 {
                self.m_obj_thresh.push(dx);
            }
        } // end of big loop on sets of sections

        // Finish output file
        if let Some(mut ofp) = outfp.take() {
            outhead.amean /= outhead.nz as f32;
            if mrc_head_write(&mut ofp, &mut outhead) != 0 {
                exit_error(b"Writing header to output image file");
            }
            ii_fclose(&mut ofp);
        }
        if let Some(mut in_fp) = self.m_in_fp.take() {
            ii_fclose(&mut in_fp);
        }
        self.m_in_head.fp = None;
        if let Some(mut dump) = self.m_dump_fp.take() {
            let _ = dump.flush();
        }

        // Fill in missing points
        if fill_in_added > 0 {
            self.fill_in_by_tilt_fits(&mut imod, fill_in_added, debug_track, debug_z);
        }

        // Finish model
        if !imod.obj.is_empty() {
            imod.xmax = self.m_nx_in;
            imod.ymax = self.m_ny_in;
            imod.zmax = self.m_in_head.nz;

            imod_set_ref_image(&mut imod, &self.m_in_head);

            let _ = imod_write_file(&imod, &mut model_fp);
        }
        drop(model_fp);

        let Some(refmod) = refmod else {
            exit(0);
        };

        // Compare reference model to the peak model
        // First count # of actual points
        // Fixed in translation: the source skips object `boundObj`, a 1-based
        // number, in these 0-based loops; the boundary object is
        // `boundObj - 1`.
        let bound_ind: i32 = if bound_obj >= 0 { bound_obj - 1 } else { -1 };
        let bobj = if bound_obj >= 0 {
            Some(&refmod.obj[(bound_obj - 1) as usize])
        } else {
            None
        };
        for ob in 0..refmod.obj.len() {
            if ob as i32 == bound_ind {
                continue;
            }
            for cont in &refmod.obj[ob].cont {
                for pt in &cont.pts {
                    if Self::point_inside_boundary(bobj, pt) != 0 {
                        num_ref_pts += 1;
                    }
                }
            }
        }

        // Next look at each peak inside boundary and look for match
        let np = self.m_num_peaks as usize;
        qsort(&mut self.m_peak_list[..np], &mut |a, b| compare_peaks(a, b));
        for i in 0..np {
            self.m_peak_list[i].integral = -1.;
            if self.m_peak_list[i].ccc != 0. {
                pnt.x = self.m_peak_list[i].xcen;
                pnt.y = self.m_peak_list[i].ycen;
                pnt.z = self.m_peak_list[i].iz as f32;
                if Self::point_inside_boundary(bobj, &pnt) != 0 {
                    self.m_peak_list[i].integral = 0.;
                    let mut ob = 0;
                    while ob < refmod.obj.len() && self.m_peak_list[i].integral == 0. {
                        if ob as i32 == bound_ind {
                            ob += 1;
                            continue;
                        }
                        let mut co = 0;
                        while co < refmod.obj[ob].cont.len() && self.m_peak_list[i].integral == 0. {
                            let cont = &refmod.obj[ob].cont[co];
                            for pt in &cont.pts {
                                if b3dnint!(pt.z) == self.m_peak_list[i].iz
                                    && imod_point_distance(pt, &pnt) < self.m_match_crit
                                {
                                    self.m_peak_list[i].integral = 1.;
                                    break;
                                }
                            }
                            co += 1;
                        }
                        ob += 1;
                    }
                    if self.m_vkeys.as_deref().is_some_and(|k| k.contains('g')) {
                        let p = self.m_peak_list[i];
                        printf!(
                            "Peak: %d %.4f %.4f %f %f %f %f\n",
                            CArg::Int(b3dnint!(p.integral) as i64),
                            CArg::Dbl(p.peak as f64),
                            CArg::Dbl((p.peak * p.ccc) as f64),
                            CArg::Dbl((p.cenmean - p.annmean) as f64),
                            CArg::Dbl(p.cenmean as f64),
                            CArg::Dbl(p.annmean as f64),
                            CArg::Dbl(p.median as f64)
                        );
                    }
                }
            }
        }

        // Walk backwards through list and count up not matched and matched and
        // find place with minimum total error
        let mut err_min: f32 = (num_ref_pts + 1) as f32;
        let mut last_peak: f32 = 1.0;
        for i in (0..np).rev() {
            if self.m_peak_list[i].integral < 0. {
                continue;
            }
            let error = (num_ref_pts + num_unmatched - num_matched) as f32;
            if last_peak >= hist_dip && self.m_peak_list[i].peak < hist_dip {
                printf!(
                    "Threshold criterion of %.3f has error %d, %d fn and %d fp\n",
                    CArg::Dbl(hist_dip as f64),
                    CArg::Int(b3dnint!(error) as i64),
                    CArg::Int((num_ref_pts - num_matched) as i64),
                    CArg::Int(num_unmatched as i64)
                );
            }
            if error < err_min {
                err_min = error;
                peak_above = last_peak;
                peak_below = self.m_peak_list[i].peak;
                num_peak_pts = num_matched + num_unmatched;
                num_peak_match = num_matched;
            }
            last_peak = self.m_peak_list[i].peak;
            if self.m_peak_list[i].integral > 0. {
                num_matched += 1;
            } else {
                num_unmatched += 1;
            }
        }

        printf!(
            "Minimum error %d with a criterion of %.4f (%.4f to %.4f):\n%d unmatched of %d actual points, %.1f%% false negative\n%d unmatched peak points, %.1f%% false positive\n",
            CArg::Int((num_ref_pts + num_peak_pts - 2 * num_peak_match) as i64),
            CArg::Dbl(0.5 * (peak_above as f64 + peak_below as f64)),
            CArg::Dbl(peak_below as f64),
            CArg::Dbl(peak_above as f64),
            CArg::Int((num_ref_pts - num_peak_match) as i64),
            CArg::Int(num_ref_pts as i64),
            CArg::Dbl(100. * (num_ref_pts - num_peak_match) as f64 / num_ref_pts as f64),
            CArg::Int((num_peak_pts - num_peak_match) as i64),
            CArg::Dbl(100. * (num_peak_pts - num_peak_match) as f64 / num_ref_pts as f64)
        );

        exit(0);
    }

    /// `FindBeads::setupSizeDependentVars` (`imodfindbeads.cpp:984`): computes
    /// the sizes for integrals and sobel filtering and allocates arrays that
    /// depend on bead size.
    fn setup_size_dependent_vars(&mut self) {
        let mut nxtmp = 0;
        let mut nytmp = 0;
        let mut xtmp: f32 = 0.;
        let mut ytmp: f32 = 0.;
        let bead = self.m_bead_size as f64;

        // Get radii for the integral
        self.m_rad_center = b3dmax!(1., 0.34 * bead) as f32;
        self.m_rad_inner = b3dmax!(
            self.m_rad_center as f64 + 1.,
            0.5 * bead + b3dmax!(1., 0.1 * bead)
        ) as f32;
        self.m_rad_outer = (self.m_rad_inner as f64 + b3dmax!(2., 0.2 * bead)) as f32;
        self.m_min_dist = self.m_min_spacing * self.m_bead_size;
        self.m_match_crit = b3dmax!(0.2 * bead, 2.) as f32;

        // Get size for scaled slices
        self.m_scale_factor = self.m_bead_size / self.m_scaled_size;
        if self.m_linear_interp < 0 && self.m_scale_factor < 1.2 {
            self.m_linear_interp = 1;
        }
        scaled_sobel(
            None,
            self.m_nx_in,
            self.m_ny_in,
            self.m_scale_factor,
            self.m_min_interp,
            self.m_linear_interp,
            -1.,
            None,
            &mut self.m_nx_out,
            &mut self.m_ny_out,
            &mut self.m_xoffset,
            &mut self.m_yoffset,
        );

        if self.m_vkeys.is_some() {
            printf!(
                "mNxOut %d  mNyOut %d mXoffset %f mYoffset %f mScaleFactor %f\n",
                CArg::Int(self.m_nx_out as i64),
                CArg::Int(self.m_ny_out as i64),
                CArg::Dbl(self.m_xoffset as f64),
                CArg::Dbl(self.m_yoffset as f64),
                CArg::Dbl(self.m_scale_factor as f64)
            );
        }

        // Get the full box size, find the size that best matches the specified
        // scaled size and revise it
        self.m_box_scaled = self.m_box_scaled_orig;
        self.m_box_size = 2 * (b3dnint!(self.m_box_scaled as f32 * self.m_scale_factor) / 2);
        let mut mindiff = 100000;
        let range = (2. * self.m_scale_factor as f64) as i32;
        // Fixed in translation: uninitialised in the source when the first
        // candidate size is already <= 3.
        let mut minsize = self.m_box_size;
        let mut size = self.m_box_size + 2 * range;
        while size >= self.m_box_size - 2 * range {
            if size <= 3 {
                break;
            }
            scaled_sobel(
                None,
                size,
                size,
                self.m_scale_factor,
                self.m_min_interp,
                self.m_linear_interp,
                -1.,
                None,
                &mut nxtmp,
                &mut nytmp,
                &mut xtmp,
                &mut ytmp,
            );
            let diff = if nxtmp > self.m_box_scaled {
                nxtmp - self.m_box_scaled
            } else {
                self.m_box_scaled - nxtmp
            };
            if diff < mindiff {
                mindiff = diff;
                minsize = size;
            }
            size -= 2;
        }
        self.m_box_size = minsize;
        scaled_sobel(
            None,
            self.m_box_size,
            self.m_box_size,
            self.m_scale_factor,
            self.m_min_interp,
            self.m_linear_interp,
            -1.,
            None,
            &mut self.m_box_scaled,
            &mut nytmp,
            &mut xtmp,
            &mut ytmp,
        );

        // Need padded size for arrays being transformed
        self.m_nx_pad = nice_frame(self.m_nx_out, 2, 19);
        self.m_ny_pad = nice_frame(self.m_ny_out, 2, 19);
        self.m_nxp_dim = self.m_nx_pad + 2;
        if self.m_vkeys.is_some() {
            printf!(
                "mNxPad %d  mNyPad %d mBoxSize %d boxScaled %d\n",
                CArg::Int(self.m_nx_pad as i64),
                CArg::Int(self.m_ny_pad as i64),
                CArg::Int(self.m_box_size as i64),
                CArg::Int(self.m_box_scaled as i64)
            );
        }
        printf!(
            "Scaling down by %.2f for Sobel filter; box size = %d, scaled size = %d\n",
            CArg::Dbl(self.m_scale_factor as f64),
            CArg::Int(self.m_box_size as i64),
            CArg::Int(self.m_box_scaled as i64)
        );

        // Get memory for filtered slice, synthetic bead and scaled bead
        let bs = self.m_box_size as usize;
        self.m_filt_bead = vec![0.; (self.m_box_scaled * self.m_box_scaled) as usize];
        self.m_filt_slice = vec![0.; (self.m_nx_out * self.m_ny_out) as usize];
        self.m_corr_slice = vec![0.; (self.m_nxp_dim * self.m_ny_pad) as usize];
        self.m_full_bead = vec![0.; bs * bs];
        self.m_one_bead = vec![0.; bs * bs];
        self.m_split_bead = vec![0.; (self.m_nxp_dim * self.m_ny_pad) as usize];
        if self.m_center_weight == 0. {
            self.m_write_slice = vec![0.; (self.m_nx_out * self.m_ny_out) as usize];
        }
    }

    /// `FindBeads::searchCorrelationPeaks` (`imodfindbeads.cpp:1054`):
    /// examines each point to see if it is a correlation peak, gets peak
    /// strength, ccc, integral and annulus properties and stores peaks.
    fn search_correlation_peaks(&mut self, sl: &Islice, iz: i32) {
        let mut cenmean: f32 = 0.;
        let mut annmean: f32 = 0.;
        let mut median: f32 = 0.;
        let nxp = self.m_nxp_dim;

        // Limit range by prealignment shift
        let mut ixst = (self.m_nx_pad - self.m_nx_out) / 2 + 1;
        let mut ixnd = ixst + self.m_nx_out - 2;
        let mut iyst = (self.m_ny_pad - self.m_ny_out) / 2 + 1;
        let mut iynd = iyst + self.m_ny_out - 2;
        if let (Some(xs), Some(ys)) = (&self.m_align_xshift, &self.m_align_yshift) {
            let sf = self.m_scale_factor as f64;
            let xsh = xs[iz as usize] as f64;
            let ysh = ys[iz as usize] as f64;
            ixst += (b3dmax!(0., xsh) / sf).ceil() as i32;
            ixnd -= (b3dmax!(0., -xsh) / sf).ceil() as i32;
            iyst += (b3dmax!(0., ysh) / sf).ceil() as i32;
            iynd -= (b3dmax!(0., -ysh) / sf).ceil() as i32;
        }
        for iy in iyst..iynd {
            for ix in ixst..ixnd {
                let ind = (ix + iy * nxp) as usize;
                let c = &self.m_corr_slice;
                let n = nxp as usize;
                let cval = c[ind];
                if c[ind - 1] < cval
                    && c[ind + 1] <= cval
                    && c[ind - n] < cval
                    && c[ind + n] <= cval
                    && c[ind - 1 - n] < cval
                    && c[ind + 1 + n] < cval
                    && c[ind + 1 - n] < cval
                    && c[ind - 1 + n] < cval
                {
                    let cx = parabolic_fit_position(c[ind - 1], cval, c[ind + 1]) as f32;
                    let cy = parabolic_fit_position(c[ind - n], cval, c[ind + n]) as f32;

                    // integer offset in scaled, filtered image
                    let ixofs = ix - (self.m_nx_pad - self.m_nx_out) / 2;
                    let iyofs = iy - (self.m_ny_pad - self.m_ny_out) / 2;

                    // Center of feature in full original image
                    let xcen = (ixofs as f32 + cx + self.m_bead_cen_ofs) * self.m_scale_factor
                        + self.m_xoffset;
                    let ycen = (iyofs as f32 + cy + self.m_bead_cen_ofs) * self.m_scale_factor
                        + self.m_yoffset;
                    if self.m_num_area_cont != 0 {
                        let area = self.m_area_mod.as_ref().unwrap();
                        let co = imod_point_inside_area(
                            &area.obj[0],
                            &self.m_area_conts,
                            self.m_num_area_cont,
                            xcen,
                            ycen,
                        );
                        if (co < 0 && self.m_exclude_areas == 0)
                            || (co >= 0 && self.m_exclude_areas != 0)
                        {
                            continue;
                        }
                    }

                    // First validate the peak by polarity of density in full
                    // image
                    let mut integral = bead_integral(
                        sl.data.f(),
                        self.m_nx_in,
                        self.m_nx_in,
                        self.m_ny_in,
                        self.m_rad_center,
                        self.m_rad_inner,
                        self.m_rad_outer,
                        xcen,
                        ycen,
                        &mut cenmean,
                        &mut annmean,
                        Some(&mut self.m_kern_hist),
                        self.m_annulus_pctile,
                        Some(&mut median),
                    ) as f32;
                    if self.m_light_beads == 0 {
                        integral = -integral;
                    }
                    if integral > 0. {
                        // Good, then get a CCC and add to list
                        let ccc = self.template_cc_coefficient(
                            &self.m_filt_slice,
                            self.m_nx_out,
                            self.m_nx_out,
                            self.m_ny_out,
                            &self.m_filt_bead,
                            self.m_box_scaled,
                            self.m_box_scaled,
                            self.m_box_scaled,
                            ixofs - self.m_box_scaled / 2,
                            iyofs - self.m_box_scaled / 2,
                        );
                        if ccc > 0. {
                            if self.m_num_peaks >= self.m_list_size {
                                self.m_peak_list.reserve(10000);
                                self.m_list_size += 10000;
                            }
                            let mut entry = PeakEntry {
                                xcen,
                                ycen,
                                iz,
                                ccc,
                                integral,
                                peak: cval,
                                cenmean,
                                annmean,
                                median,
                            };
                            if self.m_measure_to_use == 1 {
                                entry.peak = integral;
                            }
                            if self.m_measure_to_use == 2 {
                                entry.peak = (b3dmax!((cval * integral) as f64, 0.)).sqrt() as f32;
                            }
                            self.m_peak_max = b3dmax!(self.m_peak_max, entry.peak);
                            self.m_peak_list.push(entry);
                            self.m_num_peaks += 1;
                        }
                    }
                }
            }
        }
    }

    /// `FindBeads::analyzeBackgroundGroups` (`imodfindbeads.cpp:1144`):
    /// divides points into multiple groups if possible based on annulus
    /// medians and tries to find a dip in each histogram, then fits a line to
    /// the dips and uses this to adjust all the peak strengths to a common
    /// basis.
    fn analyze_background_groups(&mut self) {
        let mut ann_min: f32 = 0.;
        let mut ann_max: f32 = 0.;
        let mut nin_hist = 0;
        let min_in_group = 100;
        let max_bal_range: f32 = 4.;
        let mut sel_peak_min: f32 = 0.;
        let mut sel_peak_max: f32 = 0.;
        let mut nin_sel = 0;
        let mut sel_slope: f32 = 0.;
        let mut sel_intcp: f32 = 0.;
        let mut mode_slope: f32 = 0.;
        let mut mode_intcp: f32 = 0.;
        let mut xtmp: f32 = 0.;
        let mut ann_midval = [0f32; MAX_GROUPS];
        let mut ann_dip = [0f32; MAX_GROUPS];
        let mut ann_peak_above = [0f32; MAX_GROUPS];
        let mut ann_use_min: f32 = 0.;
        let mut ann_use_max: f32 = 0.;

        // Adjust for a trend in peak strength with background mean
        // Start with histogram of annular means and set up groups based on this
        let mut mean_med = PeakField::Annmean;
        if self.m_annulus_pctile >= 0. {
            mean_med = PeakField::Median;
        }
        self.selected_min_max(
            mean_med,
            None,
            0.,
            0.,
            &mut ann_min,
            &mut ann_max,
            &mut nin_hist,
        );
        self.kernel_histo_pl(
            mean_med,
            1,
            None,
            0.,
            0.,
            Hist::Reg,
            MAX_BINS as i32,
            ann_min,
            ann_max,
            0.,
            0,
            false,
        );
        if self.m_vkeys.is_some() {
            printf!(
                "min %f max %f ninhist %d\n",
                CArg::Dbl(ann_min as f64),
                CArg::Dbl(ann_max as f64),
                CArg::Int(nin_hist as i64)
            );
        }
        self.m_num_groups = b3dmin!(self.m_num_groups, nin_hist / min_in_group);
        let dxbin = (ann_max - ann_min) / MAX_BINS as f32;
        if self.m_num_groups > 1 {
            let mut lower_ind = 0usize;
            let mut cum_start: f32 = 0.;
            let mut ndat = 0usize;

            // For each group, find upper limit as place where hist reaches
            // target
            for igr in 0..self.m_num_groups {
                let target =
                    (((igr as f64 + 1.) * nin_hist as f64) / self.m_num_groups as f64) as f32;
                let mut sum: f64 = 0.;
                let mut ind = lower_ind;
                let mut cumul = cum_start;
                while ind < MAX_BINS
                    && (cumul + self.m_reg_hist[ind] < target || ind == MAX_BINS - 1)
                {
                    sum += ((ind as f64 + 0.5) * dxbin as f64 + ann_min as f64)
                        * self.m_reg_hist[ind] as f64;
                    cumul += self.m_reg_hist[ind];
                    ind += 1;
                }

                // Get a selected histogram of the peaks
                let lower_lim = lower_ind as f32 * dxbin + ann_min;
                let upper_lim = ind as f32 * dxbin + ann_min;
                self.selected_min_max(
                    PeakField::Peak,
                    Some(mean_med),
                    lower_lim,
                    upper_lim,
                    &mut sel_peak_min,
                    &mut sel_peak_max,
                    &mut nin_sel,
                );
                sel_peak_min = self.m_min_relative_peak * sel_peak_max;
                let mut verbose = 0;
                if self.m_vkeys.as_deref().is_some_and(|k| k.contains('P')) {
                    verbose = 1;
                    printf!("Peak:  selected data set %d\n", CArg::Int(igr as i64 + 1));
                } else if self.m_vkeys.as_deref().is_some_and(|k| k.contains('H')) {
                    verbose = 2;
                }
                self.kernel_histo_pl(
                    PeakField::Peak,
                    1,
                    Some(mean_med),
                    lower_lim,
                    upper_lim,
                    Hist::Kern,
                    MAX_BINS as i32,
                    sel_peak_min,
                    sel_peak_max,
                    (0.1 * sel_peak_max as f64) as f32,
                    verbose,
                    true,
                );
                if self.m_dump_fp.is_some() {
                    printf!(
                        "Type %2d:  Selected histogram for group %d\n",
                        CArg::Int(self.m_dump_type as i64),
                        CArg::Int(igr as i64)
                    );
                    self.m_dump_type += 1;
                }

                match scan_histogram(
                    &self.m_kern_hist,
                    sel_peak_min,
                    sel_peak_max,
                    sel_peak_min,
                    sel_peak_max,
                    true,
                ) {
                    None => {
                        if self.m_vkeys.is_some() {
                            printf!(
                                "No histogram dip: annular mean %.2f to %.2f  peaks %.2f to %.2f\n",
                                CArg::Dbl(lower_lim as f64),
                                CArg::Dbl(upper_lim as f64),
                                CArg::Dbl(sel_peak_min as f64),
                                CArg::Dbl(sel_peak_max as f64)
                            );
                        }
                    }
                    Some(found) => {
                        let dip = found.dip;
                        let peak_above = found.peak_above;
                        ann_midval[ndat] = (sum / (cumul - cum_start) as f64) as f32;
                        if ndat == 0 {
                            ann_use_min = (2. * ann_midval[ndat] as f64 - upper_lim as f64) as f32;
                        }
                        ann_use_max = (2. * ann_midval[ndat] as f64 - lower_lim as f64) as f32;
                        ann_dip[ndat] = dip;
                        ann_peak_above[ndat] = peak_above;
                        if self.m_vkeys.is_some() {
                            printf!(
                                "%.2f  %.2f  %.2f %.2f %.2f %.2f %.2f\n",
                                CArg::Dbl(lower_lim as f64),
                                CArg::Dbl(upper_lim as f64),
                                CArg::Dbl(ann_midval[ndat] as f64),
                                CArg::Dbl(sel_peak_min as f64),
                                CArg::Dbl(sel_peak_max as f64),
                                CArg::Dbl(dip as f64),
                                CArg::Dbl(peak_above as f64)
                            );
                        }
                        ndat += 1;
                    }
                }
                lower_ind = ind;
                cum_start = cumul;
            }

            if ndat > 1 {
                // Fit a line to the points and scale the peaks, find new max
                ls_fit(
                    &ann_midval,
                    &ann_peak_above,
                    ndat as i32,
                    &mut mode_slope,
                    &mut mode_intcp,
                    &mut xtmp,
                );
                ls_fit(
                    &ann_midval,
                    &ann_dip,
                    ndat as i32,
                    &mut sel_slope,
                    &mut sel_intcp,
                    &mut xtmp,
                );
                printf!(
                    "Dips found in %d groups based on bkg mean, means %.5g to %.5g\n",
                    CArg::Int(ndat as i64),
                    CArg::Dbl(ann_midval[0] as f64),
                    CArg::Dbl(ann_midval[ndat - 1] as f64)
                );
                if self.m_vkeys.is_some() {
                    printf!(
                        " Fit of mode vs background has slope %f, intcp %f\n Fit of dip vs background has slope %f, intcp %f\n",
                        CArg::Dbl(mode_slope as f64),
                        CArg::Dbl(mode_intcp as f64),
                        CArg::Dbl(sel_slope as f64),
                        CArg::Dbl(sel_intcp as f64)
                    );
                }

                if (self.m_light_beads == 0 && sel_slope > 0.)
                    || (self.m_light_beads != 0 && sel_slope < 0.)
                {
                    if self.m_light_beads != 0 {
                        let val = (((ann_use_min * sel_slope + sel_intcp) as f64
                            / max_bal_range as f64
                            - sel_intcp as f64)
                            / sel_slope as f64) as f32;
                        ann_use_max = b3dmin!(val, ann_use_max);
                    } else {
                        let val = (((ann_use_max * sel_slope + sel_intcp) as f64
                            / max_bal_range as f64
                            - sel_intcp as f64)
                            / sel_slope as f64) as f32;
                        ann_use_min = b3dmax!(val, ann_use_min);
                    }
                    if self.m_vkeys.is_some() {
                        printf!(
                            "Adjusting for background limited to %.2f to %.2f\n",
                            CArg::Dbl(ann_use_min as f64),
                            CArg::Dbl(ann_use_max as f64)
                        );
                    }
                    self.m_peak_max = -1.0e30;
                    for j in 0..self.m_num_peaks as usize {
                        if self.m_peak_list[j].ccc == 0. {
                            continue;
                        }
                        let mut val = self.m_peak_list[j].get(mean_med);
                        val = b3dmax!(ann_use_min, b3dmin!(ann_use_max, val));
                        self.m_peak_list[j].peak /= val * sel_slope + sel_intcp;
                        self.m_peak_max = b3dmax!(self.m_peak_max, self.m_peak_list[j].peak);
                    }
                }
            }
        }
    }

    /// `FindBeads::averageBeads` (`imodfindbeads.cpp:1294`): finds the
    /// threshold for averaging and then average the beads above threshold.
    fn average_beads(&mut self, iz_start: i32, iz_end: i32) {
        let mut hist_dip: f32 = 0.;
        let mut peak_above: f32 = 0.;
        let mut peak_below: f32 = 0.;
        let mut new_diam: f32 = 0.;
        let mut cen_mean: f32 = 0.;
        let mut ann_mean: f32 = 0.;
        let np = self.m_num_peaks as usize;

        let rad_bead = b3dmax!(
            self.m_rad_center as f64 + 1.,
            0.5 * self.m_bead_size as f64 + b3dmax!(1., 0.03 * self.m_bead_size as f64)
        ) as f32;
        let mut num_start: usize = 0;
        let mut thresh_use = self.m_threshold;
        if self.m_threshold > 1. {
            if self.m_num_peaks > 1 {
                qsort(&mut self.m_peak_list[..np], &mut |a, b| compare_peaks(a, b));
            }
            num_start = b3dmax!(0, self.m_num_peaks - self.m_threshold as i32) as usize;
        } else if self.m_threshold < 0. {
            // Negative threshold: find dip
            if self.m_dump_fp.is_some() {
                printf!("Dumping histograms for finding threshold for averaging:\n");
            }
            if self.find_histo_dip_pl(&mut hist_dip, &mut peak_below, &mut peak_above, None) != 0 {
                if self.m_average_fallback <= 0 || self.m_num_peaks < 2 {
                    exit_error(b"Failed to find dip in smoothed histogram of peaks");
                }
                printf!(
                    "Failed to find dip in histogram; using fallback threshold of %d for averaging\n",
                    CArg::Int(self.m_average_fallback as i64)
                );
                qsort(&mut self.m_peak_list[..np], &mut |a, b| compare_peaks(a, b));
                num_start = b3dmax!(0, self.m_num_peaks - self.m_average_fallback) as usize;
                thresh_use = 2.;
            } else {
                // Set original automatic threshold at 1/4 way from dip to
                // peak, or -mThreshold as the fraction above dip to take
                thresh_use = (0.75 * hist_dip as f64 + 0.25 * peak_above as f64) as f32;
                if self.m_threshold >= -1. {
                    if self.m_num_peaks > 1 {
                        qsort(&mut self.m_peak_list[..np], &mut |a, b| compare_peaks(a, b));
                    }
                    let mut j = 0;
                    while j < np {
                        if self.m_peak_list[j].ccc != 0. && self.m_peak_list[j].peak >= hist_dip {
                            break;
                        }
                        j += 1;
                    }
                    let mut ns = self.m_num_peaks
                        + b3dnint!(self.m_threshold * (self.m_num_peaks - j as i32) as f32);
                    ns = b3dmax!(0, b3dmin!(self.m_num_peaks - 1, ns));
                    num_start = ns as usize;
                    thresh_use = self.m_peak_list[num_start].peak;
                }
                printf!(
                    "Threshold for averaging set to %.3f\n",
                    CArg::Dbl(thresh_use as f64)
                );
            }
        }
        self.profile("Found threshold");

        let amat: [[f32; 2]; 2] = [[1., 0.], [0., 1.]];
        let bs = self.m_box_size;
        let bsq = (bs * bs) as usize;

        for i in 0..bsq {
            self.m_full_bead[i] = 0.;
        }
        let mut nsum = 0;

        // Loop through images again
        for indz in iz_start..iz_end {
            let iz = self.m_zlist[indz as usize];
            let mut loaded: Option<Islice> = None;
            let cache_ind = (indz - iz_start) as usize;
            for j in num_start..np {
                let p = self.m_peak_list[j];
                if p.ccc != 0. && p.iz == iz && (thresh_use > 1. || p.peak >= thresh_use) {
                    if loaded.is_none() {
                        loaded = Some(if self.m_use_slice_cache {
                            self.m_cached_slices[cache_ind].take().unwrap()
                        } else {
                            self.read_slice_as_float(iz)
                        });
                    }
                    let sl = loaded.as_ref().unwrap();

                    // Interpolate the bead into center of array, add it to sum
                    cubinterp(
                        sl.data.f(),
                        &mut self.m_one_bead,
                        self.m_nx_in,
                        self.m_ny_in,
                        bs,
                        bs,
                        &amat,
                        p.xcen,
                        p.ycen,
                        -self.m_avg_xoffset,
                        -self.m_avg_yoffset,
                        1.,
                        self.m_in_head.amean,
                        b3dmax!(0, self.m_linear_interp),
                    );
                    nsum += 1;
                    for i in 0..bsq {
                        self.m_full_bead[i] += self.m_one_bead[i];
                    }
                }
            }
            if let Some(sl) = loaded {
                if self.m_use_slice_cache {
                    self.m_cached_slices[cache_ind] = Some(sl);
                }
            }
        }
        if nsum != 0 {
            for i in 0..bsq {
                self.m_full_bead[i] /= nsum as f32;
            }

            // Get the centroid and shift the average
            let integral = bead_integral(
                &self.m_full_bead,
                bs,
                bs,
                bs,
                self.m_rad_center,
                self.m_rad_inner,
                self.m_rad_outer,
                (bs as f64 / 2.) as f32,
                (bs as f64 / 2.) as f32,
                &mut cen_mean,
                &mut ann_mean,
                None,
                0.,
                Some(&mut new_diam),
            );
            let polarity: f32 = if integral > 0. { 1. } else { -1. };
            let (mut xo, mut yo) = (0f32, 0f32);
            Self::bead_centroid(
                &self.m_full_bead,
                bs,
                bs,
                bs,
                rad_bead,
                (bs as f64 / 2.) as f32,
                (bs as f64 / 2.) as f32,
                ann_mean,
                polarity,
                &mut xo,
                &mut yo,
            );
            self.m_avg_xoffset = xo;
            self.m_avg_yoffset = yo;
            if self.m_vkeys.is_some() {
                printf!(
                    "Center offset of average: %.2f %.2f\n",
                    CArg::Dbl(self.m_avg_xoffset as f64),
                    CArg::Dbl(self.m_avg_yoffset as f64)
                );
            }
            self.m_one_bead[..bsq].copy_from_slice(&self.m_full_bead[..bsq]);
            cubinterp(
                &self.m_one_bead,
                &mut self.m_full_bead,
                bs,
                bs,
                bs,
                bs,
                &amat,
                (bs as f64 / 2.) as f32,
                (bs as f64 / 2.) as f32,
                -self.m_avg_xoffset,
                -self.m_avg_yoffset,
                1.,
                ann_mean,
                0,
            );

            // get new centroid for correcting next time and for final output
            Self::bead_centroid(
                &self.m_full_bead,
                bs,
                bs,
                bs,
                rad_bead,
                (bs as f64 / 2.) as f32,
                (bs as f64 / 2.) as f32,
                ann_mean,
                polarity,
                &mut xo,
                &mut yo,
            );
            self.m_avg_xoffset = xo;
            self.m_avg_yoffset = yo;

            new_diam = self.extract_diameter();
            if new_diam > 0. {
                printf!(
                    "Diameter of average bead at zero-crossing = %.2f\n",
                    CArg::Dbl(new_diam as f64)
                );
            }
            if self.m_adjust_sizes != 0 {
                if new_diam <= 0. {
                    printf!(
                        "WARNING: imodfindbeads - Bead size could not be measured and will not be adjusted\n"
                    );
                } else if b3dmax!(new_diam, self.m_bead_size) / b3dmin!(new_diam, self.m_bead_size)
                    > self.m_max_adjust_factor
                {
                    printf!(
                        "WARNING: imodfindbeads - Measured bead diameter differs too much from specified size to be plausible, so bead size will not be adjusted\n"
                    );
                } else {
                    // autofindseed is looking for "Adjusted parameters" at start
                    // of line
                    printf!(
                        "Adjusted parameters based on a new bead size of %.2f\n",
                        CArg::Dbl(new_diam as f64)
                    );
                    record(FindbeadsReport::AdjustedSize(new_diam));

                    // Save the bead, remake all the arrays, and copy average
                    // into new array with trimming or padding
                    let old_box = self.m_box_size;
                    let save_bead: Vec<f32> =
                        self.m_full_bead[..(old_box * old_box) as usize].to_vec();
                    self.m_bead_size = new_diam;
                    self.setup_size_dependent_vars();
                    let nb = self.m_box_size;
                    if old_box >= nb {
                        let offset = (old_box - nb) / 2;
                        for j in 0..nb {
                            for i in 0..nb {
                                self.m_full_bead[(i + j * nb) as usize] =
                                    save_bead[(i + offset + (j + offset) * old_box) as usize];
                            }
                        }
                    } else {
                        slice_taper_out_pad(
                            PadIn::Float(&save_bead),
                            SLICE_MODE_FLOAT,
                            old_box,
                            old_box,
                            &mut self.m_full_bead,
                            nb,
                            nb,
                            nb,
                            0,
                            0.,
                        );
                    }
                }
            }
        }
    }

    /// `FindBeads::extractDiameter` (`imodfindbeads.cpp:1446`): finds the
    /// diameter of an average or single bead.  (The source's `oneBead`
    /// argument is unused; it reads `mFullBead`.)
    fn extract_diameter(&self) -> f32 {
        let bs = self.m_box_size;
        let edge = slice_edge_mean(&self.m_full_bead, bs, 0, bs - 1, 0, bs - 1) as f32;
        let polarity: f32 = if self.m_light_beads != 0 { 1. } else { -1. };
        let mut xx = [0f32; 210];
        let mut yy = [0f32; 210];
        // `grad` is printed on the first ring before it is set (verbose
        // only); it starts at 0.
        let mut grad: f32 = 0.;
        let mut max_grad: f32 = 0.;
        let ifirst = (0.2 * self.m_bead_size as f64) as i32;
        let mut ndat: usize = 0;
        let mut ind_max: i32 = -1;
        let mut slope: f32 = 0.;
        let mut intcp: f32 = 0.;
        let mut ro: f32 = 0.;

        // Find mean in a series of rings; loop on all pixels that could be in
        // a ring and test each one against inner and outer radii
        let iend = b3dnint!(0.9 * self.m_bead_size as f64);
        let alloc = (iend - ifirst + 10).max(0) as usize;
        let mut diams = vec![0f32; alloc];
        let mut avgden = vec![0f32; alloc];
        for i in ifirst..iend {
            let mut nsum = 0;
            let mut bsum: f32 = 0.;
            let rad = i as f32;
            for jy in (bs / 2 - i - 2)..=(bs / 2 + i + 2) {
                for jx in (bs / 2 - i - 2)..=(bs / 2 + i + 2) {
                    let dx = (jx as f64 - (bs - 1) as f64 / 2.) as f32;
                    let dy = (jy as f64 - (bs - 1) as f64 / 2.) as f32;
                    let mut radsq = dx * dx + dy * dy;
                    if radsq >= rad * rad && radsq as f64 <= (rad as f64 + 1.) * (rad as f64 + 1.) {
                        radsq = polarity * (self.m_full_bead[(jx + bs * jy) as usize] - edge);
                        nsum += 1;
                        bsum += radsq;
                    }
                }
            }
            diams[ndat] = (2. * rad as f64 + 1.) as f32;
            avgden[ndat] = bsum / nsum as f32;

            // Keep track of the point with the maximum gradient prior to it
            if ndat != 0 {
                grad = avgden[ndat - 1] - avgden[ndat];
                if grad > max_grad {
                    max_grad = grad;
                    ind_max = ndat as i32;
                }
            }
            if self.m_vkeys.is_some() {
                printf!(
                    "%.2f  %.2f  %.2f  %.2f\n",
                    CArg::Dbl(diams[ndat] as f64),
                    CArg::Dbl(avgden[ndat] as f64),
                    CArg::Dbl(grad as f64),
                    CArg::Dbl(max_grad as f64)
                );
            }
            ndat += 1;
        }

        if ind_max < 0 {
            return 0.;
        }

        // Fit to ones around maximum gradient with a gradient nearly as big
        let mut nfit: usize = 2;
        let im = ind_max as usize;
        xx[0] = diams[im];
        yy[0] = avgden[im];
        xx[1] = diams[im - 1];
        yy[1] = avgden[im - 1];
        let mut idir = -1;
        while idir <= 1 {
            for idiff in 1..100 {
                let mut ind = ind_max + idir * idiff;
                // Fixed in translation: the source's bound is `ind > ndat`,
                // which reads `avgden[ndat]`, past the filled rings.
                if ind < 1 || ind >= ndat as i32 {
                    break;
                }
                let g = avgden[(ind - 1) as usize] - avgden[ind as usize];
                if (g as f64) < 0.75 * max_grad as f64 {
                    break;
                }
                if idir < 0 {
                    ind -= 1;
                }
                xx[nfit] = diams[ind as usize];
                yy[nfit] = avgden[ind as usize];
                // Fixed in translation: the source never advances `nfit`, so
                // every point it collects overwrites the third slot and the
                // fit is always the two points around the maximum gradient.
                nfit += 1;
            }
            idir += 2;
        }
        ls_fit(&xx, &yy, nfit as i32, &mut slope, &mut intcp, &mut ro);
        -intcp / slope
    }

    /// `FindBeads::findStorageThreshold` (`imodfindbeads.cpp:1560`): finds the
    /// threshold for storing peaks in the model.
    fn find_storage_threshold(&mut self, hist_dip: &mut f32, line: &mut String) -> f32 {
        let mut thresh_use: f32 = -10000.;
        let mut peak_below: f32 = 0.;
        let mut peak_above: f32 = 0.;

        *hist_dip = -1.;
        if self.m_dump_fp.is_some() {
            printf!("Dumping histograms for finding threshold for output points:\n");
        }
        let vkeys = self.m_vkeys.clone();
        if self.find_histo_dip_pl(hist_dip, &mut peak_below, &mut peak_above, vkeys.as_deref()) == 0
        {
            let dxbin: f32 = (1. / MAX_BINS as f64) as f32;

            // Count points above threshold
            let jstr = (*hist_dip / dxbin) as i32;
            let mut nsum = 0;
            let mut sum: f64 = 0.;
            let mut sumsq: f64 = 0.;
            for j in jstr.max(0) as usize..MAX_BINS {
                nsum += b3dnint!(self.m_reg_hist[j]);
                let val = ((j as f64 + 0.5) * dxbin as f64) as f32;
                sum += (self.m_reg_hist[j] * val) as f64;
                sumsq += (self.m_reg_hist[j] * val * val) as f64;
            }
            // (`jstr` is never negative: the dip is found in 0..1.)

            // estimate mean and SD for points above threshold
            if self.m_peak_thresh == 0. {
                let mut sd_above: f32 = 0.;
                let mean_above = (sum / b3dmax!(1, nsum) as f64) as f32;
                if nsum > 1 {
                    let val = ((sumsq - (nsum as f32 * mean_above * mean_above) as f64)
                        / (nsum as f64 - 1.)) as f32;
                    sd_above = (b3dmax!(val as f64, 0.)).sqrt() as f32;
                }
                let jend = ((mean_above as f64 - 5. * sd_above as f64) / dxbin as f64) as i32;
                let mut jdir = 0;
                let mut j = jstr - 1;
                while j >= 0 {
                    jdir += b3dnint!(self.m_reg_hist[j as usize]);
                    if jdir > nsum || (j < jend && jdir > nsum / 10) {
                        break;
                    }
                    j -= 1;
                }

                thresh_use = j as f32 * dxbin;

                // autofidseed wants to see " peaks are above" or "using
                // fallback" on the next to last line and 'total peaks being'
                // on the last line.  But we need to defer the 'total peaks
                // being' output until the real # is known
                printf!(
                    "%d peaks are above threshold of %.3f\n",
                    CArg::Int(nsum as i64),
                    CArg::Dbl(*hist_dip as f64)
                );
                record(FindbeadsReport::PeaksAboveThreshold {
                    num: nsum,
                    dip: *hist_dip,
                });
                let would_be = self.m_num_obj_orig != 0;
                printf!(
                    "%d more peaks %s stored in model down to value of %.3f\n",
                    CArg::Int(jdir as i64),
                    CArg::Str(if would_be { "would be" } else { "being" }),
                    CArg::Dbl(thresh_use as f64)
                );
                record(FindbeadsReport::MorePeaksStored {
                    num: jdir,
                    would_be,
                    thresh: thresh_use,
                });
                line.clear();
            } else {
                // Or find threshold given a relative number
                let jend = -b3dnint!(self.m_peak_thresh * nsum as f32);
                let mut jdir = 0;
                let mut j = MAX_BINS as i32 - 1;
                while j >= 0 {
                    jdir += b3dnint!(self.m_reg_hist[j as usize]);
                    if jdir >= jend {
                        break;
                    }
                    j -= 1;
                }
                thresh_use = b3dmax!(0., (j as f32 * dxbin) as f64) as f32;
                printf!(
                    "%d peaks are above histogram dip at %.3f\n",
                    CArg::Int(nsum as i64),
                    CArg::Dbl(*hist_dip as f64)
                );
                record(FindbeadsReport::PeaksAboveDip {
                    num: nsum,
                    dip: *hist_dip,
                });
                let text = c_format_bytes(
                    "total peaks %s stored in model down to value of %.3f\n",
                    &[
                        CArg::Str(if self.m_num_obj_orig != 0 {
                            "would be"
                        } else {
                            "being"
                        }),
                        CArg::Dbl(thresh_use as f64),
                    ],
                );
                *line = String::from_utf8_lossy(&text[..text.len().min(MAX_LINE - 2)]).into_owned();
                if self.m_num_obj_orig != 0 {
                    printf!("%d %s", CArg::Int(jdir as i64), CArg::Str(line));
                    record(FindbeadsReport::TotalPeaksStored {
                        num: jdir,
                        text: line.clone(),
                    });
                    line.clear();
                }
            }
        } else if self.m_storage_fallback > 0 && self.m_num_peaks > 1 {
            let np = self.m_num_peaks as usize;
            qsort(&mut self.m_peak_list[..np], &mut |a, b| compare_peaks(a, b));
            let num_start = b3dmax!(0, self.m_num_peaks - self.m_storage_fallback);
            thresh_use = self.m_peak_list[num_start as usize].peak;
            printf!(
                "Failed to find dip in histogram, using fallback threshold for storing points\n"
            );
            record(FindbeadsReport::UsingFallback);
            let text = c_format_bytes(
                "%d total peaks being stored in model down to value of %.3f\n",
                &[
                    CArg::Int((self.m_num_peaks - num_start) as i64),
                    CArg::Dbl(thresh_use as f64),
                ],
            );
            *line = String::from_utf8_lossy(&text[..text.len().min(MAX_LINE - 2)]).into_owned();
        } else {
            // Autofidseed wants to see 'Failed to find dip' on the last line of
            // output
            printf!("Failed to find dip in histogram\n");
            record(FindbeadsReport::FailedToFindDip);
        }
        thresh_use
    }

    /// `FindBeads::areaContListCheckErr` (`imodfindbeads.cpp:1664`): calls
    /// `makeAreaContList` for a section and checks and responds to errors.
    fn area_cont_list_check_err(&mut self, iz: i32) {
        let area = self.m_area_mod.as_ref().unwrap();
        let ix = make_area_cont_list(
            &area.obj[0],
            iz,
            &mut self.m_area_conts,
            &mut self.m_num_area_cont,
            MAX_AREAS as i32,
        );
        if ix < 0 {
            exit_error_fmt!(
                "Too many contours on one section in area model for array (limit %d)",
                CArg::Int(MAX_AREAS as i64)
            );
        }
        if ix > 0 {
            exit_error(b"No contours in object 1 of area model");
        }
    }

    /// `FindBeads::pointInsideBoundary` (`imodfindbeads.cpp:1680`): tests if
    /// pnt is inside one of the contours of obj; returns 1 if it is inside 1
    /// or if there are no contours on the point's section.
    fn point_inside_boundary(
        obj: Option<&crate::imod::libimod::imodel::Iobj>,
        pnt: &Ipoint,
    ) -> i32 {
        let mut inside = 1;
        let Some(obj) = obj else {
            return 1;
        };
        for cont in &obj.cont {
            if !cont.pts.is_empty() && b3dnint!(cont.pts[0].z) == b3dnint!(pnt.z) {
                inside = imod_point_inside_cont(cont, pnt);
                if inside != 0 {
                    return 1;
                }
            }
        }
        inside
    }

    /// `FindBeads::readSliceAsFloat` (`imodfindbeads.cpp:1721`): read a slice
    /// at iz in file and convert it to a floating slice, then taper it by 64
    /// pixels at fill edges.
    fn read_slice_as_float(&mut self, iz: i32) -> Islice {
        let Some(mut sl) = slice_read_float(&mut self.m_in_head, iz) else {
            exit_error_fmt!(
                "Creating slice for or reading section %d",
                CArg::Int(iz as i64)
            );
        };

        // Taper fill areas very slowly to avoid big gradients there
        if slice_taper_at_fill(&mut sl, 64, false) != 0 {
            exit_error(b"Getting memory for tapering edges");
        }
        sl
    }

    /// `FindBeads::templateCCCoefficient` (`imodfindbeads.cpp:1741`):
    /// computes CCC in real space between the image in template, size nxt by
    /// nyt and X dimension nxtdim, and a position in the image in array, size
    /// nx by ny and X dimension nxdim.  xoffset, yoffset is the offset from
    /// pixels in the template to pixels in the array.
    #[allow(clippy::too_many_arguments)]
    fn template_cc_coefficient(
        &self,
        array: &[f32],
        nxdim: i32,
        nx: i32,
        ny: i32,
        template_im: &[f32],
        nxtdim: i32,
        nxt: i32,
        nyt: i32,
        xoffset: i32,
        yoffset: i32,
    ) -> f32 {
        let ixst = b3dmax!(0, xoffset);
        let ixnd = b3dmin!(nx - 1, nxt + xoffset - 1);
        let iyst = b3dmax!(0, yoffset);
        let iynd = b3dmin!(ny - 1, nyt + yoffset - 1);
        let nsum = (ixnd + 1 - ixst) * (iynd + 1 - iyst);
        let mut asum: f64 = 0.;
        let mut asumsq: f64 = 0.;
        let mut bsum: f64 = 0.;
        let mut bsumsq: f64 = 0.;
        let mut csum: f64 = 0.;
        if nsum < 16 {
            return 0.;
        }
        for iy in iyst..iynd {
            for ix in ixst..ixnd {
                let aval = array[(ix + iy * nxdim) as usize] as f64;
                let bval = template_im[(ix - xoffset + (iy - yoffset) * nxtdim) as usize] as f64;
                asum += aval;
                asumsq += aval * aval;
                bsum += bval;
                bsumsq += bval * bval;
                csum += aval * bval;
            }
        }
        let nsum = nsum as f64;
        let mut ccc = (nsum * asumsq - asum * asum) * (nsum * bsumsq - bsum * bsum);
        if ccc <= 0. {
            return 0.;
        }
        ccc = (nsum * csum - asum * bsum) / ccc.sqrt();
        ccc as f32
    }

    /// `FindBeads::kernelHistoPL` (`imodfindbeads.cpp:1780`): computes a
    /// standard or kernel histogram from the peaks in peaklist; `element` is
    /// the measure to use; `select` if given is used to select on a different
    /// member of the peak entry, in which case selMin and selMax specify min
    /// and max of range to select.  The histogram is placed in `bins`, and
    /// occupies numBins between firstVal and lastVal.  Peaks with zero ccc
    /// are skipped if skipZeroCCC is set.  h is the kernel width, or 0 for a
    /// standard binned histogram.  `dump` stands for the source's `dumpFp`
    /// argument being `mDumpFp` rather than NULL.
    #[allow(clippy::too_many_arguments)]
    fn kernel_histo_pl(
        &mut self,
        element: PeakField,
        skip_zero_ccc: i32,
        select: Option<PeakField>,
        sel_min: f32,
        sel_max: f32,
        which: Hist,
        num_bins: i32,
        first_val: f32,
        last_val: f32,
        h: f32,
        verbose: i32,
        dump: bool,
    ) {
        let mut bins = std::mem::take(match which {
            Hist::Reg => &mut self.m_reg_hist,
            Hist::Kern => &mut self.m_kern_hist,
        });
        let nb = num_bins as usize;
        let dxbin = (last_val - first_val) / num_bins as f32;
        for b in bins.iter_mut().take(nb) {
            *b = 0.;
        }
        for j in 0..self.m_num_peaks as usize {
            let p = &self.m_peak_list[j];
            if skip_zero_ccc != 0 && p.ccc == 0. {
                continue;
            }
            if let Some(sel) = select {
                let val = p.get(sel);
                if val < sel_min || val >= sel_max {
                    continue;
                }
            }
            let val = p.get(element);

            if verbose == 1 {
                printf!("Peak: %.4f\n", CArg::Dbl(val as f64));
            }
            if h != 0. {
                let mut ist = (((val - h - first_val) as f64) / dxbin as f64).ceil() as i32;
                let mut ind = (((val + h - first_val) as f64) / dxbin as f64).floor() as i32;
                ist = b3dmax!(0, ist);
                ind = b3dmin!(num_bins - 1, ind);
                for i in ist..=ind {
                    let delta = (val - first_val - i as f32 * dxbin) / h;
                    bins[i as usize] += (1. - (delta * delta) as f64).powf(3.) as f32;
                }
            } else {
                let ist = (((val - first_val) as f64) / dxbin as f64).floor() as i32;
                if ist >= 0 && ist < num_bins {
                    bins[ist as usize] += 1.;
                } else if ist == num_bins && ((val - last_val) as f64) < 0.001 * dxbin as f64 {
                    bins[nb - 1] += 1.;
                }
            }
        }

        if h != 0. {
            let scale = (1. / (h as f64 * (1.2 - 2. / 7.))) as f32;
            for b in bins.iter_mut().take(nb) {
                *b *= scale;
            }
        }

        if verbose == 2 {
            for i in 0..nb {
                printf!(
                    "bin: %.4f %f\n",
                    CArg::Dbl((first_val + i as f32 * dxbin) as f64),
                    CArg::Dbl(bins[i] as f64)
                );
            }
        }
        if dump {
            if let Some(fp) = self.m_dump_fp.as_mut() {
                for i in 0..nb {
                    let _ = fp.write_all(&c_format_bytes(
                        "%2d %.4f %f\n",
                        &[
                            CArg::Int(self.m_dump_type as i64),
                            CArg::Dbl((first_val + i as f32 * dxbin) as f64),
                            CArg::Dbl(bins[i] as f64),
                        ],
                    ));
                }
            }
        }
        match which {
            Hist::Reg => self.m_reg_hist = bins,
            Hist::Kern => self.m_kern_hist = bins,
        }
    }

    /// `FindBeads::findHistoDipPL` (`imodfindbeads.cpp:1838`): finds a
    /// histogram dip by starting with a high smoothing and dropping to lower
    /// one.
    fn find_histo_dip_pl(
        &mut self,
        hist_dip: &mut f32,
        peak_below: &mut f32,
        peak_above: &mut f32,
        vkeys: Option<&str>,
    ) -> i32 {
        let mut coarse_h: f32 = 0.2;
        let fine_h: f32 = 0.05;
        let num_cut = 4;
        let frac_guess: f32 = 0.5;
        let mut upper_lim: f32 = 1.0;

        // Build a regular histogram first and use minGuess to find safe upper
        // limit
        let mut verbose = if vkeys.is_some_and(|k| k.contains('p')) {
            1
        } else {
            0
        };
        self.kernel_histo_pl(
            PeakField::Peak,
            1,
            None,
            0.,
            0.,
            Hist::Reg,
            MAX_BINS as i32,
            0.,
            1.,
            0.,
            verbose,
            true,
        );
        if self.m_dump_fp.is_some() {
            printf!(
                "Type %2d:  Regular histogram\n",
                CArg::Int(self.m_dump_type as i64)
            );
            self.m_dump_type += 1;
        }

        if self.m_min_guess != 0 {
            let num_crit = b3dmax!(1, b3dnint!(self.m_min_guess as f32 * frac_guess));
            let mut ncum = 0;
            let mut i = MAX_BINS as i32 - 1;
            while i > 10 {
                ncum += b3dnint!(self.m_reg_hist[i as usize]);
                if ncum >= num_crit {
                    break;
                }
                i -= 1;
            }
            upper_lim = (i as f64 / (MAX_BINS as f64 - 1.)) as f32;
        }

        // Seek a kernel width that gives two peaks in histogram
        let mut i = 0;
        while i < num_cut {
            verbose = 0;
            if let Some(k) = vkeys {
                if k.contains('e') || (k.contains('i') && i == 0) {
                    verbose = 2;
                }
            }
            self.kernel_histo_pl(
                PeakField::Peak,
                1,
                None,
                0.,
                0.,
                Hist::Kern,
                MAX_BINS as i32,
                0.,
                1.,
                coarse_h,
                verbose,
                true,
            );
            if self.m_dump_fp.is_some() {
                printf!(
                    "Type %2d:  Kernel histogram with H = %.3f\n",
                    CArg::Int(self.m_dump_type as i64),
                    CArg::Dbl(coarse_h as f64)
                );
                self.m_dump_type += 1;
            }

            // Cut H if it fails or if the top peak is at 1.0
            if let Some(found) = scan_histogram(&self.m_kern_hist, 0., 1., 0., upper_lim, true) {
                *hist_dip = found.dip;
                *peak_below = found.peak_below;
                *peak_above = found.peak_above;
                if *peak_above < 0.999 {
                    break;
                }
            }
            coarse_h = (coarse_h as f64 * 0.707) as f32;
            i += 1;
        }
        if i == num_cut {
            return 1;
        }

        printf!(
            "Histogram smoothed with H = %.3f has dip at %.3f, peaks at %.3f and %.3f\n",
            CArg::Dbl(coarse_h as f64),
            CArg::Dbl(*hist_dip as f64),
            CArg::Dbl(*peak_below as f64),
            CArg::Dbl(*peak_above as f64)
        );

        verbose = if vkeys.is_some_and(|k| k.contains('f')) {
            2
        } else {
            0
        };
        self.kernel_histo_pl(
            PeakField::Peak,
            1,
            None,
            0.,
            0.,
            Hist::Kern,
            MAX_BINS as i32,
            0.,
            1.,
            fine_h,
            verbose,
            true,
        );
        if self.m_dump_fp.is_some() {
            printf!(
                "Type %2d:  Kernel histogram with fine H = %.3f\n",
                CArg::Int(self.m_dump_type as i64),
                CArg::Dbl(fine_h as f64)
            );
            self.m_dump_type += 1;
        }
        // `findPeaks` 0: only the dip is returned; the peaks keep their
        // values.
        if let Some(found) = scan_histogram(
            &self.m_kern_hist,
            0.,
            1.,
            (0.5 * (*hist_dip as f64 + *peak_below as f64)) as f32,
            (0.5 * (*hist_dip as f64 + *peak_above as f64)) as f32,
            false,
        ) {
            *hist_dip = found.dip;
        }
        printf!(
            "Histogram smoothed with H = %.3f has lowest dip at %.3f\n",
            CArg::Dbl(fine_h as f64),
            CArg::Dbl(*hist_dip as f64)
        );
        0
    }

    /// `FindBeads::selectedMinMax` (`imodfindbeads.cpp:1919`): computes the
    /// min and max and number of peaks, possibly selecting on some value
    /// being in a given range.
    #[allow(clippy::too_many_arguments)]
    fn selected_min_max(
        &self,
        element: PeakField,
        select: Option<PeakField>,
        sel_min: f32,
        sel_max: f32,
        min_val: &mut f32,
        max_val: &mut f32,
        nin_range: &mut i32,
    ) {
        *min_val = 1.0e30;
        *max_val = -1.0e30;
        *nin_range = 0;
        for j in 0..self.m_num_peaks as usize {
            let p = &self.m_peak_list[j];
            if p.ccc == 0. {
                continue;
            }
            if let Some(sel) = select {
                let val = p.get(sel);
                if val < sel_min || val >= sel_max {
                    continue;
                }
            }
            let val = p.get(element);
            *min_val = b3dmin!(*min_val, val);
            *max_val = b3dmax!(*max_val, val);
            *nin_range += 1;
        }
    }

    /// `FindBeads::profile` (`imodfindbeads.cpp:1943`): wall-time profiling
    /// output, off unless `mProfiling` is set (it never is).
    fn profile(&mut self, message: &str) {
        if !self.m_profiling {
            return;
        }
        let now = wall_time();
        printf!(
            "%10.3f %5.3f  ",
            CArg::Dbl(now - self.m_wall_start),
            CArg::Dbl(now - self.m_wall_last)
        );
        self.m_wall_last = now;
        printf!("%s", CArg::Str(message));
        printf!("\n");
    }

    /// `FindBeads::beadCentroid` (`imodfindbeads.cpp:1960`): compute the
    /// centroid of a bead given the background and center radius.
    #[allow(clippy::too_many_arguments)]
    fn bead_centroid(
        array: &[f32],
        _nxdim: i32,
        nx: i32,
        ny: i32,
        r_center: f32,
        xcen: f32,
        ycen: f32,
        bkgd: f32,
        polarity: f32,
        x_offset: &mut f32,
        y_offset: &mut f32,
    ) {
        let xpcen = xcen - 0.5;
        let ypcen = ycen - 0.5;
        let rcensq = r_center * r_center;
        let mut ncen = 0;
        let ixcen = b3dnint!(xpcen);
        let iycen = b3dnint!(ypcen);
        let irad_out = (r_center as f64 + 1.5) as i32;
        let mut xsum: f64 = 0.;
        let mut ysum: f64 = 0.;
        let mut wsum: f64 = 0.;

        for iy in (iycen - irad_out)..=(iycen + irad_out) {
            if iy < 0 || iy >= ny {
                continue;
            }
            let dy = iy as f32 - ypcen;
            let dxsq = (rcensq - dy * dy) as f64;
            let idx = (b3dmax!(0., dxsq).sqrt() + 1.5) as i32;
            for ix in (ixcen - idx)..=(ixcen + idx) {
                if ix < 0 || ix >= nx {
                    continue;
                }
                let dx = ix as f32 - xpcen;
                let radsq = dy * dy + dx * dx;
                if radsq <= rcensq {
                    let wgt = (array[(ix + iy * nx) as usize] - bkgd) * polarity;
                    if wgt > 0. {
                        wsum += wgt as f64;
                        xsum += (wgt * dx) as f64;
                        ysum += (wgt * dy) as f64;
                        ncen += 1;
                    }
                }
            }
        }
        *x_offset = 0.;
        *y_offset = 0.;
        if ncen == 0 {
            return;
        }
        *x_offset = (xsum / wsum) as f32;
        *y_offset = (ysum / wsum) as f32;
    }

    /// `FindBeads::contourIsBelowThreshold` (`imodfindbeads.cpp:2003`): test
    /// whether a given contour is below the stored threshold for its object,
    /// if any.
    fn contour_is_below_threshold(&self, imod: &Imod, ob: usize, co: usize) -> bool {
        if self.m_obj_thresh[ob] < -1.0e29 {
            return false;
        }
        let store = &imod.obj[ob].store;
        let (ind_store, after) = istore_lookup(store, co as i32);
        // Fixed in translation: when the contour has no store items the
        // source's lookup returns -1 and the loop scans the items of the
        // contours before it (`istoreItem(-1)` is NULL, then 0, 1, ...).
        let Some(ind_store) = ind_store else {
            return false;
        };
        for j in ind_store..after {
            if let Some(item) = store.get(j) {
                if item.type_ == GEN_STORE_VALUE1 {
                    return item.value.f() < self.m_obj_thresh[ob];
                }
            }
        }
        false
    }

    /// `FindBeads::fillInByTiltFits` (`imodfindbeads.cpp:2019`): construct
    /// tracks from all the points and try to fill them in and extend them.
    fn fill_in_by_tilt_fits(
        &mut self,
        imod: &mut Imod,
        extrap_forward: i32,
        debug_track: i32,
        debug_z: i32,
    ) {
        let nz = self.m_in_head.nz;
        let nzu = nz as usize;
        let mut ind_min_tilt: usize = 0;
        let xcen = (self.m_in_head.nx as f64 / 2.) as f32;
        let mut xsol: f32 = 0.;
        let mut zsol: f32 = 0.;
        let mut r_avg: f32 = 0.;
        let mut r_sd: f32 = 0.;
        let mut r_max: f32 = 0.;
        let mut xpred: f32 = 0.;
        let mut pred_err: f32 = 0.;
        let mut slope: f32 = 0.;
        let mut intcp: f32 = 0.;
        let mut num_fit: i32 = 0;
        let mut ind_min: usize = 0;
        let mut ind_wins: usize = 0;
        let fill_dir = [-1, 1, 1];
        let mut sin_ang: Vec<f32> = Vec::new();
        let mut cos_ang: Vec<f32> = Vec::new();
        let mut cc: Vec<f32> = Vec::new();
        let mut ss: Vec<f32> = Vec::new();
        let mut xx: Vec<f32> = Vec::new();
        let mut all_z: Vec<f32> = Vec::new();
        let mut all_res_avg: Vec<f32> = Vec::new();
        let mut all_res_sd: Vec<f32> = Vec::new();
        let mut all_res_max: Vec<f32> = Vec::new();
        let mut stat_inds: Vec<i32> = Vec::new();
        let mut pair_wins: Vec<i32>;
        let mut sort_inds: Vec<usize> = Vec::new();
        let mut track_zlists: Vec<Vec<i32>> = Vec::new();
        let mut track_ind_in_z: Vec<Vec<usize>> = Vec::new();
        let mut assign_lists: Vec<Vec<i32>>;
        let mut pairs_or_tracks: Vec<Vec<i32>>;
        let mut xx_lists: Vec<Vec<f32>>;
        let mut yy_lists: Vec<Vec<f32>>;
        let mut x_err_lists: Vec<Vec<f32>>;
        let mut y_err_lists: Vec<Vec<f32>>;
        let mut sd_res_errors: Vec<Vec<f32>>;
        let (mut avg_of_z, mut sd_of_z, mut avg_of_avg, mut sd_of_avg) = (0f32, 0f32, 0f32, 0f32);
        let (mut avg_of_sd, mut sd_of_sd, mut avg_of_max, mut sd_of_max) = (0f32, 0f32, 0f32, 0f32);
        let mut sem: f32 = 0.;
        let mut z_min: f32 = 1.0e30;
        let mut z_max: f32 = -1.0e30;
        let mut yyfit = [0f32; 10];
        let mut zzfit = [0f32; 10];
        let mut pt_fp: Option<ImodFile> = None;

        // Set the max gap the same, otherwise it produces duplicate points
        let max_gap = extrap_forward;
        let bead = self.m_bead_size as f64;
        let bead_ycrit = (0.3 * bead) as f32;
        let bead_yclose = (0.07 * bead) as f32;
        let pair_dy_fac_crit: f32 = 1.33;
        let pair_dy_diff_crit = (0.07 * bead) as f32;
        let pair_sd_fac_crit: f32 = 1.5;
        let pair_sd_diff_crit: f32 = 1.;
        let track_dx_diff_crit = (0.5 * bead) as f32;
        let lim_outside = 3;
        let err_crit: f32 = 2.;
        let min_size_for_stats = 4;
        let verbose = self.m_vkeys.as_deref().is_some_and(|k| k.contains('t'));

        if imod_new_object(imod) != 0 {
            exit_error(b"Creating new object in model");
        }
        let new_ob = imod.obj.len() - 1;
        {
            let (sym, symsize, pdraw, flags0) = (
                imod.obj[0].symbol,
                imod.obj[0].symsize,
                imod.obj[0].pdrawsize,
                imod.obj[0].flags,
            );
            let obj = &mut imod.obj[new_ob];
            obj.flags |= IMOD_OBJFLAG_OPEN;
            obj.symbol = sym;
            obj.symsize = symsize;
            obj.pdrawsize = pdraw;
            if flags0 & IMOD_OBJFLAG_PNT_ON_SEC != 0 {
                obj.flags |= IMOD_OBJFLAG_PNT_ON_SEC;
            }
        }

        // Get cosines/sines
        let mut tilt_min: f32 = 1000.;
        for ind in 0..nzu {
            cos_ang.push((self.m_tilt_angles[ind] as f64 * RADIANS_PER_DEGREE).cos() as f32);
            sin_ang.push((self.m_tilt_angles[ind] as f64 * RADIANS_PER_DEGREE).sin() as f32);
            if self.m_tilt_angles[ind].abs() < tilt_min {
                ind_min_tilt = ind;
                tilt_min = self.m_tilt_angles[ind].abs();
            }
        }

        // Fill lists by Z
        xx_lists = vec![Vec::new(); nzu];
        yy_lists = vec![Vec::new(); nzu];
        assign_lists = vec![Vec::new(); nzu];
        for ob in self.m_num_obj_orig as usize..imod.obj.len() {
            for co in 0..imod.obj[ob].cont.len() {
                // Skip ones below threshold
                if self.contour_is_below_threshold(imod, ob, co) {
                    continue;
                }

                let cont = &imod.obj[ob].cont[co];
                let iz = b3dnint!(cont.pts[0].z) as usize;
                xx_lists[iz].push(cont.pts[0].x - xcen);
                yy_lists[iz].push(cont.pts[0].y);
                assign_lists[iz].push(-1);
            }
        }

        // fit to existing contours with 3 or more points
        for ob in 0..self.m_num_obj_orig as usize {
            for co in 0..imod.obj[ob].cont.len() {
                if imod.obj[ob].cont[co].pts.len() < 3 {
                    continue;
                }
                if self.contour_is_below_threshold(imod, ob, co) {
                    continue;
                }
                let cont = &imod.obj[ob].cont[co];
                cc.clear();
                ss.clear();
                xx.clear();
                for pt in &cont.pts {
                    let iz = b3dnint!(pt.z);
                    if iz >= 0 && iz < nz {
                        cc.push(cos_ang[iz as usize]);
                        ss.push(-sin_ang[iz as usize]);
                        xx.push(pt.x - xcen);
                    }
                }
                Self::tilt_fit_and_resids(
                    &cc, &ss, &xx, &mut xsol, &mut zsol, &mut r_avg, &mut r_sd, &mut r_max,
                );

                z_min = if z_min < zsol { z_min } else { zsol };
                z_max = if z_max > zsol { z_max } else { zsol };
                all_z.push(zsol);
                all_res_avg.push(r_avg);
                all_res_sd.push(r_sd);
                all_res_max.push(r_max);
                num_fit += 1;
            }
        }
        avg_sd(&all_z, num_fit, &mut avg_of_z, &mut sd_of_z, &mut sem);
        avg_sd(
            &all_res_avg,
            num_fit,
            &mut avg_of_avg,
            &mut sd_of_avg,
            &mut sem,
        );
        avg_sd(
            &all_res_sd,
            num_fit,
            &mut avg_of_sd,
            &mut sd_of_sd,
            &mut sem,
        );
        avg_sd(
            &all_res_max,
            num_fit,
            &mut avg_of_max,
            &mut sd_of_max,
            &mut sem,
        );
        if verbose {
            cout_line(&[
                ("zMin", CArg::Dbl(z_min as f64)),
                ("zMax", CArg::Dbl(z_max as f64)),
                ("avgOfZ", CArg::Dbl(avg_of_z as f64)),
                ("sdOfZ", CArg::Dbl(sd_of_z as f64)),
            ]);
        }
        if verbose {
            cout_line(&[
                ("avgOfAvg", CArg::Dbl(avg_of_avg as f64)),
                ("sdOfAvg", CArg::Dbl(sd_of_avg as f64)),
                ("avgOfSD", CArg::Dbl(avg_of_sd as f64)),
                ("sdOfSD", CArg::Dbl(sd_of_sd as f64)),
            ]);
        }

        // Loop on directions from minimum tilt
        let mut next_pos = ind_min_tilt as i32;
        let mut next_neg = ind_min_tilt as i32;
        while next_neg > 0 || next_pos < nz - 1 {
            let mut dir = -1;
            while dir <= 1 {
                // Get the next Z for this direction
                let prevz = if dir > 0 { next_pos } else { next_neg };
                let nextz = prevz + dir;
                if nextz < 0 || nextz >= nz {
                    dir += 2;
                    continue;
                }
                let (pu, nu) = (prevz as usize, nextz as usize);
                if verbose {
                    cout_line(&[
                        ("nextNeg", CArg::Int(next_neg as i64)),
                        ("nextPos", CArg::Int(next_pos as i64)),
                        ("prevz", CArg::Int(prevz as i64)),
                        ("nextz", CArg::Int(nextz as i64)),
                    ]);
                }
                let num = xx_lists[nu].len();
                pairs_or_tracks = vec![Vec::new(); num];
                x_err_lists = vec![Vec::new(); num];
                y_err_lists = vec![Vec::new(); num];
                sd_res_errors = vec![Vec::new(); num];

                let z_low = (z_min as f64 - 0.5 * (z_max - z_min) as f64) as f32;
                let z_high = (z_max as f64 + 0.5 * (z_max - z_min) as f64) as f32;

                // Loop on tracks first
                for track in 0..track_zlists.len() {
                    let tz_min = *track_zlists[track].iter().min().unwrap();
                    let tz_max = *track_zlists[track].iter().max().unwrap();
                    let spot_debug = track as i32 == debug_track && nextz == debug_z;

                    // If the track extends close enough to this z, fit to it
                    // and predict X
                    if (if dir > 0 {
                        nextz - tz_max
                    } else {
                        tz_min - nextz
                    }) > max_gap + 1
                    {
                        continue;
                    }
                    cc.clear();
                    ss.clear();
                    xx.clear();
                    for ind in 0..track_zlists[track].len() {
                        let iz = track_zlists[track][ind] as usize;
                        cc.push(cos_ang[iz]);
                        ss.push(-sin_ang[iz]);
                        xx.push(xx_lists[iz][track_ind_in_z[track][ind]]);
                    }
                    ls_fit2_pred(
                        &cc,
                        &ss,
                        &xx,
                        xx.len() as i32,
                        &mut xsol,
                        &mut zsol,
                        None,
                        cos_ang[nu],
                        -sin_ang[nu],
                        &mut xpred,
                        &mut pred_err,
                    );

                    // Get mean Y from closest 5
                    let mut numy = 0usize;
                    let mut ysum: f32 = 0.;
                    let mut iz = prevz;
                    while iz >= 0 && iz < nz {
                        let mut ind = track_zlists[track].len() as i32 - 1;
                        while ind >= 0 {
                            let iu = ind as usize;
                            if track_zlists[track][iu] == iz {
                                zzfit[numy] = iz as f32;
                                let yv = yy_lists[iz as usize][track_ind_in_z[track][iu]];
                                ysum += yv;
                                yyfit[numy] = yv;
                                numy += 1;
                                if numy >= 5 {
                                    break;
                                }
                            }
                            ind -= 1;
                        }
                        if numy >= 5 {
                            break;
                        }
                        iz -= dir;
                    }
                    let ymean: f32;
                    if numy > 3 {
                        ls_fit(
                            &zzfit,
                            &yyfit,
                            numy as i32,
                            &mut slope,
                            &mut intcp,
                            &mut r_max,
                        );
                        ymean = slope * nextz as f32 + intcp;
                    } else {
                        ymean = ysum / numy as f32;
                    }
                    if spot_debug {
                        cout_line(&[
                            ("num", CArg::Int(numy as i64)),
                            ("ymean", CArg::Dbl(ymean as f64)),
                            ("ysum / num", CArg::Dbl((ysum / numy as f32) as f64)),
                        ]);
                    }

                    // Allow a wider search in case the Z value is not so good
                    // AND to make up for the prediction errors being very low
                    // sometimes
                    let mut xp_low = (zsol as f64 - 0.2 * (z_max - z_min) as f64) as f32;
                    let mut xp_high = (zsol as f64 + 0.2 * (z_max - z_min) as f64) as f32;
                    let xp1 = xsol * cos_ang[nu] - xp_low * sin_ang[nu];
                    let xp2 = xsol * cos_ang[nu] - xp_high * sin_ang[nu];
                    xp_low = b3dmin!(xp1, xp2);
                    xp_high = b3dmax!(xp1, xp2);
                    let v = xpred - pred_err * err_crit;
                    xp_low = if xp_low < v { xp_low } else { v };
                    let v = xpred + pred_err * err_crit;
                    xp_high = if xp_high > v { xp_high } else { v };
                    if spot_debug {
                        cout_line(&[
                            ("xpLow + xcen", CArg::Dbl((xp_low + xcen) as f64)),
                            ("xpHigh + xcen", CArg::Dbl((xp_high + xcen) as f64)),
                            ("xpred + xcen", CArg::Dbl((xpred + xcen) as f64)),
                            ("ymean", CArg::Dbl(ymean as f64)),
                        ]);
                    }

                    // Find closest one in Y that is within prediction range, or
                    // when points are close enough in Y, take one closer to
                    // prediction
                    let mut dx_min: f32 = 1.0e10;
                    let mut dy_min: f32 = 1.0e10;
                    for ind in 0..xx_lists[nu].len() {
                        let dy = (yy_lists[nu][ind] - ymean).abs();
                        if dy < bead_ycrit {
                            let xcur = xx_lists[nu][ind];
                            let dx = (xcur - xpred).abs();
                            if spot_debug {
                                cout_line(&[
                                    ("ind", CArg::Int(ind as i64)),
                                    ("dy", CArg::Dbl(dy as f64)),
                                    ("xcur + xcen", CArg::Dbl((xcur + xcen) as f64)),
                                    ("dx", CArg::Dbl(dx as f64)),
                                ]);
                            }
                            if xcur >= xp_low
                                && xcur <= xp_high
                                && ((dy < dy_min
                                    && dy_min > bead_yclose
                                    && dx < dx_min + track_dx_diff_crit)
                                    || (dy <= bead_yclose && dy_min <= bead_yclose && dx < dx_min)
                                    || (dy <= bead_yclose
                                        && dy_min > bead_yclose
                                        && (dx as f64)
                                            < dx_min as f64 + 1.5 * track_dx_diff_crit as f64))
                            {
                                if spot_debug {
                                    cout_line(&[
                                        ("dx", CArg::Dbl(dx as f64)),
                                        ("dxMin", CArg::Dbl(dx_min as f64)),
                                        ("dy", CArg::Dbl(dy as f64)),
                                        ("dyMin", CArg::Dbl(dy_min as f64)),
                                    ]);
                                }
                                dy_min = dy;
                                dx_min = dx;
                                ind_min = ind;
                            }
                        }
                    }

                    // If found a point, add it as a candidate destination for
                    // that point
                    if dy_min < 1.0e9 {
                        pairs_or_tracks[ind_min].push(track as i32);
                        x_err_lists[ind_min].push(dx_min);
                        y_err_lists[ind_min].push(dy_min);

                        // Fit to get residual sd
                        Self::tilt_fit_and_resids(
                            &cc, &ss, &xx, &mut xsol, &mut zsol, &mut r_avg, &mut r_sd, &mut r_max,
                        );
                        sd_res_errors[ind_min].push(
                            ((dx_min - r_avg).abs() as f64 / b3dmax!(1., r_sd as f64)) as f32,
                        );

                        // Update or add to statistics arrays since we have fit
                        // (not using all these nice statistics!)
                        let jnd = stat_inds[track];
                        if jnd >= 0 {
                            let j = jnd as usize;
                            all_z[j] = zsol;
                            all_res_avg[j] = r_avg;
                            all_res_sd[j] = r_sd;
                            all_res_max[j] = r_max;
                        } else if track_zlists[track].len() >= min_size_for_stats {
                            stat_inds[track] = all_z.len() as i32;
                            all_z.push(zsol);
                            all_res_avg.push(r_avg);
                            all_res_sd.push(r_sd);
                            all_res_max.push(r_max);
                        }
                    }
                }

                // Next loop on unassigned points at prevz
                for ind in 0..xx_lists[pu].len() {
                    if assign_lists[pu][ind] >= 0 {
                        continue;
                    }

                    // Get limits in X implied by some range in Z
                    let x3d1 = (xx_lists[pu][ind] + z_low * sin_ang[pu]) / cos_ang[pu];
                    let x3d2 = (xx_lists[pu][ind] + z_high * sin_ang[pu]) / cos_ang[pu];
                    let xp1 = x3d1 * cos_ang[nu] - z_low * sin_ang[nu];
                    let xp2 = x3d2 * cos_ang[nu] - z_high * sin_ang[nu];
                    let xp_low = b3dmin!(xp1, xp2);
                    let xp_high = b3dmax!(xp1, xp2);

                    // Compute more limits based on current min and max, which
                    // will be preferred for points close enough in Y
                    let x3d1 = (xx_lists[pu][ind] + z_min * sin_ang[pu]) / cos_ang[pu];
                    let x3d2 = (xx_lists[pu][ind] + z_max * sin_ang[pu]) / cos_ang[pu];
                    let xp1 = x3d1 * cos_ang[nu] - z_min * sin_ang[nu];
                    let xp2 = x3d2 * cos_ang[nu] - z_max * sin_ang[nu];
                    let xp_min = b3dmin!(xp1, xp2);
                    let xp_max = b3dmax!(xp1, xp2);
                    let ymean = yy_lists[pu][ind];

                    // Loop on points on next view (nothing is assigned yet
                    // there)
                    let mut dy_min: f32 = 1.0e10;
                    let mut xbest: f32 = 1.0e10;
                    for jnd in 0..xx_lists[nu].len() {
                        // Find one within limits in x that is closest in Y, but
                        // when close enough, prefer one within narrower limits
                        // implied by current min/max Z
                        let dy = (yy_lists[nu][jnd] - ymean).abs();
                        if dy < bead_ycrit {
                            let xcur = xx_lists[nu][jnd];
                            if xcur >= xp_low
                                && xcur <= xp_high
                                && ((dy < dy_min && dy_min > bead_yclose)
                                    || (dy <= bead_yclose
                                        && dy_min <= bead_yclose
                                        && (xbest > xp_max || xbest < xp_min)
                                        && xcur >= xp_min
                                        && xcur <= xp_max))
                            {
                                dy_min = dy;
                                ind_min = jnd;
                                xbest = xcur;
                            }
                        }
                    }

                    // If found a point, add this as a possible pairing
                    if dy_min < 1.0e9 {
                        pairs_or_tracks[ind_min].push(-(ind as i32 + 1));
                        x_err_lists[ind_min]
                            .push((xbest as f64 - (xp_low + xp_high) as f64 / 2.) as f32);
                        y_err_lists[ind_min].push(dy_min);
                        sd_res_errors[ind_min].push(0.);
                        continue;
                    }
                }

                // Go through potential pairings/additions and pick best if more
                // than one
                for co in 0..pairs_or_tracks.len() {
                    let num = pairs_or_tracks[co].len();
                    if num == 0 {
                        continue;
                    }
                    let mut imin = 0usize;
                    if num > 1 {
                        let mut dy_min: f32 = 1.0e10;
                        pair_wins = vec![0; num];
                        if verbose {
                            printf!(
                                "For %.1f %.1f %d, %d pairs or tracks\n",
                                CArg::Dbl((xx_lists[nu][co] + xcen) as f64),
                                CArg::Dbl(yy_lists[nu][co] as f64),
                                CArg::Int(nextz as i64),
                                CArg::Int(num as i64)
                            );
                        }

                        // Get the fallback, the one that gives minimum Y error
                        for ind in 0..num {
                            if y_err_lists[co][ind] < dy_min {
                                dy_min = y_err_lists[co][ind];
                                imin = ind;
                            }
                        }

                        // Consider each pair of assignments and add to win
                        // counter when one does win
                        for ind in 0..num - 1 {
                            for jnd in ind + 1..num {
                                let ye = &y_err_lists[co];
                                let xe = &x_err_lists[co];
                                let sd = &sd_res_errors[co];
                                let pt = &pairs_or_tracks[co];

                                // Common tests for whether the difference in Y
                                // errors is big enough for either one to win on
                                // that basis: smaller by a factor or by an
                                // absolute fraction of bead size
                                let ind_wins_on_y = ye[ind] * pair_dy_fac_crit < ye[jnd]
                                    || ye[ind] + pair_dy_diff_crit < ye[jnd];
                                // Fixed in translation: the source's second
                                // test is `yErr[jnd] + crit < yErr[jnd]`.
                                let jnd_wins_on_y = ye[jnd] * pair_dy_fac_crit < ye[ind]
                                    || ye[jnd] + pair_dy_diff_crit < ye[ind];
                                if verbose
                                    || (nextz == debug_z
                                        && (pt[ind] == debug_track || pt[jnd] == debug_track))
                                {
                                    printf!(
                                        "pot %d  y err %.2f sd err %.2f winsy %d  %d y err  %.2f sd err %.2f winsy %d\n",
                                        CArg::Int(pt[ind] as i64),
                                        CArg::Dbl(ye[ind] as f64),
                                        CArg::Dbl(sd[ind] as f64),
                                        CArg::Int(if ind_wins_on_y { 1 } else { 0 }),
                                        CArg::Int(pt[jnd] as i64),
                                        CArg::Dbl(ye[jnd] as f64),
                                        CArg::Dbl(sd[jnd] as f64),
                                        CArg::Int(if jnd_wins_on_y { 1 } else { 0 })
                                    );
                                }

                                if pt[ind] < 0 && pt[jnd] < 0 {
                                    // Both are pairings: one can win based on Y
                                    // or if both errors are smaller
                                    if ind_wins_on_y || (ye[ind] < ye[jnd] && xe[ind] < xe[jnd]) {
                                        pair_wins[ind] += 1;
                                    } else if jnd_wins_on_y
                                        // Fixed in translation: the source
                                        // compares `jnd` with itself here.
                                        || (ye[jnd] < ye[ind] && xe[jnd] < xe[ind])
                                    {
                                        pair_wins[jnd] += 1;
                                    }
                                } else if pt[ind] >= 0 && pt[jnd] >= 0 {
                                    // Both are tracks: a big enough difference in
                                    // (doctored) residual error sd's can win, or
                                    // one can win on Y if the residual error
                                    // difference is consistent
                                    if sd[ind] * pair_sd_fac_crit < sd[jnd]
                                        || sd[ind] + pair_sd_diff_crit < sd[jnd]
                                        || (sd[ind] < sd[jnd] && ind_wins_on_y)
                                    {
                                        pair_wins[ind] += 1;
                                    } else if sd[jnd] * pair_sd_fac_crit < sd[ind]
                                        || sd[jnd] + pair_sd_diff_crit < sd[ind]
                                        // Fixed in translation: the source
                                        // tests `indWinsOnY` for `jnd`.
                                        || (sd[jnd] < sd[ind] && jnd_wins_on_y)
                                    {
                                        pair_wins[jnd] += 1;
                                    }
                                } else if pt[ind] >= 0 {
                                    // One is a track, one is a pairing: the
                                    // track can win on Y if the resid error
                                    // isn't huge, or on a smaller resid error if
                                    // the other one doesn't win on Y
                                    if (ind_wins_on_y && sd[ind] < 3.)
                                        || (!jnd_wins_on_y && sd[ind] < 1.5)
                                    {
                                        pair_wins[ind] += 1;
                                    } else if jnd_wins_on_y {
                                        pair_wins[jnd] += 1;
                                    }
                                } else {
                                    // Fixed in translation: here `jnd` is the
                                    // track and `ind` the pairing, and the
                                    // source credited each outcome to the
                                    // other entry.
                                    if (jnd_wins_on_y && sd[jnd] < 3.)
                                        || (!ind_wins_on_y && sd[jnd] < 1.5)
                                    {
                                        pair_wins[jnd] += 1;
                                    } else if ind_wins_on_y {
                                        pair_wins[ind] += 1;
                                    }
                                }
                            }
                        }

                        // Count up the wins
                        let mut max_wins = 0;
                        let mut num_at_max = 0;
                        for ind in 0..num {
                            let jnd = pairs_or_tracks[co][ind];
                            if verbose || (nextz == debug_z && jnd == debug_track) {
                                printf!(
                                    "%s %d wins %d times\n",
                                    CArg::Str(if jnd < 0 { "point" } else { "track" }),
                                    CArg::Int(if jnd < 0 { -jnd - 1 } else { jnd } as i64),
                                    CArg::Int(pair_wins[ind] as i64)
                                );
                            }
                            if pair_wins[ind] > max_wins {
                                max_wins = pair_wins[ind];
                                ind_wins = ind;
                                num_at_max = 1;
                            } else if pair_wins[ind] == max_wins {
                                num_at_max += 1;
                            }
                        }

                        // Award the choice if one wins, otherwise it falls back
                        // to the one at min Y
                        if num_at_max == 1 {
                            imin = ind_wins;
                        }
                    }

                    // One way or another, an association has been picked so
                    // assign point
                    if pairs_or_tracks[co][imin] < 0 {
                        // Make a new track
                        let ind = (-pairs_or_tracks[co][imin] - 1) as usize;
                        let ntr = track_zlists.len();
                        assign_lists[pu][ind] = ntr as i32;
                        assign_lists[nu][co] = ntr as i32;
                        track_zlists.push(vec![prevz, nextz]);
                        track_ind_in_z.push(vec![ind, co]);
                        stat_inds.push(-1);
                        if verbose {
                            printf!(
                                "create track %d: %.1f %.1f %d  - %.1f %.1f %d\n",
                                CArg::Int(ntr as i64),
                                CArg::Dbl((xx_lists[pu][ind] + xcen) as f64),
                                CArg::Dbl(yy_lists[pu][ind] as f64),
                                CArg::Int(prevz as i64 + 1),
                                CArg::Dbl((xx_lists[nu][co] + xcen) as f64),
                                CArg::Dbl(yy_lists[nu][co] as f64),
                                CArg::Int(nextz as i64 + 1)
                            );
                        }
                    } else {
                        // Add to track
                        let track = pairs_or_tracks[co][imin] as usize;
                        track_zlists[track].push(nextz);
                        track_ind_in_z[track].push(co);
                        assign_lists[nu][co] = track as i32;
                        if verbose {
                            printf!(
                                "Add to track %d: %.1f %.1f %d\n",
                                CArg::Int(track as i64),
                                CArg::Dbl((xx_lists[nu][co] + xcen) as f64),
                                CArg::Dbl(yy_lists[nu][co] as f64),
                                CArg::Int(nextz as i64 + 1)
                            );
                        }
                    }
                }

                // At this point, update zMin/zMax  (it does not include the
                // result of adding to tracks).  Fixed in translation:
                // `VEC_MINIMUM` of an empty list dereferences `end()`; with no
                // statistics yet the limits are left as they are.
                if !all_z.is_empty() {
                    let mut mn = all_z[0];
                    let mut mx = all_z[0];
                    for &z in &all_z[1..] {
                        if z < mn {
                            mn = z;
                        }
                        if mx < z {
                            mx = z;
                        }
                    }
                    z_min = mn;
                    z_max = mx;
                }

                // Advance appropriate Z and quit when done
                if dir > 0 {
                    next_pos += 1;
                } else {
                    next_neg -= 1;
                }
                dir += 2;
            }
        }

        if verbose {
            pt_fp = ImodFile::open("ifb-tracks.pt", "w");
        }

        // Made tracks, it is time to fill them in and extend them
        let mut new_conts: Vec<crate::imod::libimod::imodel::Icont> = Vec::new();
        for track in 0..track_zlists.len() {
            let iz_min = *track_zlists[track].iter().min().unwrap();
            let iz_max = *track_zlists[track].iter().max().unwrap();
            let num = track_zlists[track].len();

            // Need it sorted so we can fit to nearest points easily
            sort_inds.clear();
            sort_inds.extend(0..num);
            for ind in 0..num.saturating_sub(1) {
                for jnd in ind + 1..num {
                    if track_zlists[track][sort_inds[ind]] > track_zlists[track][sort_inds[jnd]] {
                        sort_inds.swap(ind, jnd);
                    }
                }
            }

            // Output sorted point list in debug
            if let Some(fp) = pt_fp.as_mut() {
                for ind in 0..num {
                    let sind = sort_inds[ind];
                    let iz = track_zlists[track][sind] as usize;
                    let _ = fp.write_all(&c_format_bytes(
                        "%d %.1f %.1f %d\n",
                        &[
                            CArg::Int(track as i64 + 1),
                            CArg::Dbl((xx_lists[iz][track_ind_in_z[track][sind]] + xcen) as f64),
                            CArg::Dbl(yy_lists[iz][track_ind_in_z[track][sind]] as f64),
                            CArg::Int(iz as i64),
                        ],
                    ));
                }
            }

            // If there is anything missing (almost certainly) then set up 3
            // loops for fill so we can limit the number outside the field
            if num > 2 && (iz_min > 0 || iz_max < nz - 1 || (iz_max + 1 - iz_min) as usize > num) {
                let mut fill_zstarts = [0i32; 3];
                let mut fill_zends = [0i32; 3];
                fill_zstarts[1] = iz_min;
                fill_zends[1] = iz_max;
                fill_zstarts[0] = iz_min - 1;
                fill_zends[0] = b3dmax!(0, iz_min - extrap_forward);
                fill_zstarts[2] = iz_max + 1;
                fill_zends[2] = b3dmin!(iz_max + extrap_forward, nz - 1);
                if track as i32 == debug_track {
                    cout_line(&[
                        ("izMin", CArg::Int(iz_min as i64)),
                        ("izMax", CArg::Int(iz_max as i64)),
                        ("fillZends[0]", CArg::Int(fill_zends[0] as i64)),
                        ("fillZends[2]", CArg::Int(fill_zends[2] as i64)),
                    ]);
                }

                // Load all the data for fitting
                cc.clear();
                ss.clear();
                xx.clear();
                all_z.clear();
                for ind in 0..num {
                    let sind = sort_inds[ind];
                    let iz = track_zlists[track][sind] as usize;
                    cc.push(cos_ang[iz]);
                    ss.push(-sin_ang[iz]);
                    xx.push(xx_lists[iz][track_ind_in_z[track][sind]]);
                    all_z.push(yy_lists[iz][track_ind_in_z[track][sind]]);
                }

                // Do the fill loops
                for fllp in 0..3 {
                    // Loop on Z range
                    let mut num_outside = 0;
                    let mut iz = fill_zstarts[fllp];
                    while fill_dir[fllp] * (fill_zends[fllp] - iz) >= 0 {
                        let mut jnd = 0;
                        let mut dz_min = 1000;
                        let mut nearest = 0usize;

                        // Look for a point at this Z
                        for ind in 0..num {
                            let sind = sort_inds[ind];
                            let dz = (iz - track_zlists[track][sind]).abs();
                            if dz == 0 {
                                jnd = 1;
                                break;
                            }
                            // Fixed in translation: the source computes `dz`
                            // and `dzMin` but never records the nearest point,
                            // and then centres the fit on a stale `indMin`.
                            if dz < dz_min {
                                dz_min = dz;
                                nearest = ind;
                            }
                        }

                        // If point in missing, fill it in.  Fit to nearest 5
                        // points
                        if jnd == 0 {
                            let jst = b3dmax!(0, nearest as i32 - 2) as usize;
                            let nfit = b3dmin!(xx.len(), jst + 5) - jst;
                            ls_fit2(
                                &cc[jst..],
                                &ss[jst..],
                                &xx[jst..],
                                nfit as i32,
                                &mut xsol,
                                &mut zsol,
                                None,
                            );

                            // Average nearest 3 for Y or fit line to nearest 5
                            // and limit extrapolation
                            let ymean: f32;
                            if nfit < 4 {
                                let mut ym = 0f32;
                                avg_sd(&all_z[jst..], nfit as i32, &mut ym, &mut r_sd, &mut r_avg);
                                ymean = ym;
                            } else {
                                for ind in 0..nfit {
                                    zzfit[ind] = track_zlists[track][sort_inds[jst + ind]] as f32;
                                }
                                ls_fit(
                                    &zzfit,
                                    &all_z[jst..],
                                    nfit as i32,
                                    &mut slope,
                                    &mut intcp,
                                    &mut r_sd,
                                );
                                z_min = b3dmin!(zzfit[0], zzfit[nfit - 1]);
                                z_max = b3dmax!(zzfit[0], zzfit[nfit - 1]);
                                let dz = b3dmin!(z_max + 1., b3dmax!(z_min - 1., iz as f32)) as i32;
                                ymean = dz as f32 * slope + intcp;
                            }

                            let xbest =
                                xsol * cos_ang[iz as usize] - zsol * sin_ang[iz as usize] + xcen;

                            // Add the contour
                            let Some(mut cont) = imod_contour_new() else {
                                exit_error(b"Creating new contour");
                            };
                            if imod_point_append_xyz(&mut cont, xbest, ymean, iz as f32) == 0 {
                                exit_error(b"Appending point to contour");
                            }
                            new_conts.push(cont);
                            if (xbest as f64 + 0.75 * bead) < 0.
                                || xbest as f64 + 0.75 * bead > self.m_in_head.nx as f64
                            {
                                num_outside += 1;
                            }
                            if track as i32 == debug_track {
                                cout_line(&[
                                    ("numOutside", CArg::Int(num_outside as i64)),
                                    ("xbest", CArg::Dbl(xbest as f64)),
                                ]);
                            }
                            if num_outside >= lim_outside && fllp != 1 {
                                break;
                            }
                        }
                        iz += fill_dir[fllp];
                    }
                }
            }
        }
        for cont in new_conts {
            if imod_object_add_contour(&mut imod.obj[new_ob], cont) < 0 {
                exit_error(b"Adding contour to object");
            }
        }
        drop(pt_fp);
        printf!(
            "%d points added overall to fill gaps and extend tracks\n",
            CArg::Int(imod.obj[new_ob].cont.len() as i64)
        );
    }

    /// `FindBeads::tiltFitAndResids` (`imodfindbeads.cpp:2598`): do an lsFit2
    /// fit and compute residual statistics.
    #[allow(clippy::too_many_arguments)]
    fn tilt_fit_and_resids(
        cc: &[f32],
        ss: &[f32],
        xx: &[f32],
        xsol: &mut f32,
        zsol: &mut f32,
        r_avg: &mut f32,
        r_sd: &mut f32,
        r_max: &mut f32,
    ) {
        ls_fit2(cc, ss, xx, xx.len() as i32, xsol, zsol, None);
        let mut rsum: f32 = 0.;
        let mut rsum_sq: f32 = 0.;
        *r_max = 0.;
        for ind in 0..xx.len() {
            let diff = cc[ind] * *xsol + ss[ind] * *zsol - xx[ind];
            rsum += if diff >= 0. { diff } else { -diff };
            rsum_sq += diff * diff;
            *r_max = if *r_max > diff { *r_max } else { diff };
        }
        sums_to_avg_sd(rsum, rsum_sq, xx.len() as i32, r_avg, r_sd);
    }
}

impl Default for FindBeads {
    fn default() -> Self {
        Self::new()
    }
}
