//! Translation of `IMOD/imodutil/findbeads3d.cpp` with its class header
//! `IMOD/imodutil/findbeads3d.h` merged in.
//!
//! The C++ `FB3d` class becomes the [`FB3d`] struct, one method per member
//! function, each named after the original in its doc comment.  The
//! `fortmodel.h` globals (`fmod*`) that `writePeakModel` fills are the owned
//! [`FortModel`] in `m_fm`.
//!
//! `main`'s single `array` allocation holds the loaded sub-volume followed by
//! the three planes of pixel sums at `indPlanes`; it stays one `Vec` so that
//! `addPeakToSum` and `find_best_corr`, which read a little past the end of
//! the sub-volume exactly as the C does (`ixpP1 = B3DMIN(ixp + 1, nxb)` is
//! one past the last column), see the same neighbouring elements.  The C
//! `malloc`s it; for any volume this program can meaningfully process the
//! allocation is above glibc's mmap threshold, so it starts zeroed, which is
//! what the zero-filled `Vec` reproduces.
//!
//! OpenMP: the unit's two regions (`threecorrs`, `oneCorrCoeff`) are
//! `reduction(+:)` sums over Z, whose combination order depends on the thread
//! count; they run serially here, in the one-thread order.
//!
//! Upstream defects fixed in translation (`BUGS.md`, 2026-09-26), each
//! commented at its site: `addToSortedList` returns -1, not 0, for a rejected
//! value; `findValueInList` returns the count; `writePeakModel`'s black level
//! divides by `peakMax - peakMin` and takes `peakMin` as 0 with no peaks;
//! `getAnalysisLimits` treats the first piece of a `-?minmax` range as a
//! boundary and keeps inner pieces' starts clear of row -1; the bounds
//! messages print `ix0`.
//!
//! Deliberate deviations where the source reads outside an array:
//! * `main` reads `peakVal[indPeak[0]]` (and `cleanSortedList` the same) when
//!   no candidate was found at all, from uninitialised `indPeak[0]`; here
//!   `indPeak` starts zeroed so it reads `peakVal[0]`, which is also unused.

use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, imod_backup_file, imod_usage_header, num_omp_threads,
};
use crate::imod::libcfshr::beadutil::bead_integral;
use crate::imod::libcfshr::filtxcorr::{
    apply_kernel_filter, parabolic_fit_position, scaled_gaussian_kernel, xcorr_mean_zero,
};
use crate::imod::libcfshr::histogram::find_histogram_dip;
use crate::imod::libcfshr::islice::MrcData;
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_done, pip_get_boolean, pip_get_float, pip_get_in_out_file, pip_get_integer,
    pip_get_string, pip_get_three_integers, pip_get_two_floats, pip_get_two_integers,
    pip_read_or_parse_options,
};
use crate::imod::libcfshr::readlinevalues::{
    RLFV_SEPARATE_LINES, ReadValueArray, exit_from_value_read_error, read_lines_for_values,
};
use crate::imod::libcfshr::robuststat::rs_fast_median_in_place;
use crate::imod::libcfshr::simplestat::avg_sd;
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_MODE_FLOAT, MRC_NLABELS};
use crate::imod::libiimod::mrcslice::full_array_min_max_mean;
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_open, iiu_read_sec_part, iiu_set_position, iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_create_header, iiu_print_header, iiu_ret_basic_head, iiu_ret_delta,
    iiu_ret_origin, iiu_ret_tilt, iiu_write_header_str,
};
use crate::imod::libimod::fortmodel::{
    allocate_fort_model, scale_fort_mod_to_image, write_fort_model,
};
use crate::imod::libimod::imodel_fwrap::{
    newimod, putcontvalue, putimageref, putimodflag, putimodmaxes, putimodobjname, putscatsize,
    putvalblackwhite,
};
use std::io::Write as _;

/// `b3dutil.h:68`: `#define RADIANS_PER_DEGREE 0.01745329252` -- a *double*.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// `b3dutil.h:33`: `#define B3DNINT(a) (int)floor((a) + 0.5)`; the `0.5` is a
/// double, so a float argument is widened before the add.
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

/// `B3DABS(a)`: `((a) >= 0 ? (a) : -(a))`.
macro_rules! b3dabs {
    ($a:expr) => {{
        let a = $a;
        if a >= 0 as _ { a } else { -a }
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

/// Lends a float `Vec` to a routine that takes the typed `MrcData` union.
macro_rules! with_mrc_data {
    ($v:expr, |$d:ident| $body:expr) => {{
        let mut $d = MrcData::F(std::mem::take(&mut $v));
        let result = $body;
        if let MrcData::F(back) = $d {
            $v = back;
        }
        result
    }};
}

/// `cppdefs.h:21-24`: `PRINT2`/`PRINT4` go through `cout`, whose default
/// float output is `%g`.
fn cout_line(parts: &[(&str, Vec<u8>)]) {
    let mut line: Vec<u8> = Vec::new();
    for (i, (name, val)) in parts.iter().enumerate() {
        if i > 0 {
            line.extend_from_slice(b",  ");
        }
        line.extend_from_slice(name.as_bytes());
        line.extend_from_slice(b" = ");
        line.extend_from_slice(val);
    }
    line.push(b'\n');
    let _ = ImodFile::Stdout.write_all(&line);
}

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// Not a translated function: a stop for a read before the start of the image
/// array (`threecorrs`, `oneCorrCoeff`), where the reference binary segfaults.
/// The fixed `get_analysis_limits` keeps the analysis box clear of it, so
/// reaching this is a translation defect.
fn below_array_start_abort() -> ! {
    let _ = ImodFile::Stdout.flush();
    eprintln!(
        "findbeads3d: correlation box extends before the start of the loaded volume; \
         the IMOD source reads out of bounds here (findbeads3d.cpp:1519)"
    );
    std::process::abort();
}

/// C++ `main` (`findbeads3d.cpp:18`).
pub fn findbeads3d(arguments: &[String]) -> i32 {
    let mut fb3d = FB3d::new();
    let status = fb3d.main(arguments);
    let _ = ImodFile::Stdout.flush();
    status
}

/// `class FB3d` (`findbeads3d.h:7-81`).
pub struct FB3d {
    m_nx_corr: i32,
    m_ny_corr: i32,
    m_nz_corr: i32,
    m_polarity: f32,
    m_idx_edge: Vec<i32>,
    m_idy_edge: Vec<i32>,
    m_num_edge: i32,
    m_elong_box_size: i32,
    m_elong_temp: Vec<f32>,
    m_edge_pixels: Vec<f32>,
    m_elong_smooth: Vec<f32>,
    m_elong_kernel: [f32; 49],
    m_kern_dim_elong: i32,
    m_ix_elong: Vec<i32>,
    m_iy_elong: Vec<i32>,
    m_elong_mask: Vec<i32>,
    /// The `fortmodel.h` globals (`fmod*`).
    m_fm: FortModel,
}

impl Default for FB3d {
    fn default() -> Self {
        Self::new()
    }
}

impl FB3d {
    /// `FB3d::FB3d` (`findbeads3d.cpp:24`): an empty constructor on a stack
    /// instance; every member is assigned before it is read.
    pub fn new() -> Self {
        Self {
            m_nx_corr: 0,
            m_ny_corr: 0,
            m_nz_corr: 0,
            m_polarity: 0.,
            m_idx_edge: Vec::new(),
            m_idy_edge: Vec::new(),
            m_num_edge: 0,
            m_elong_box_size: 0,
            m_elong_temp: Vec::new(),
            m_edge_pixels: Vec::new(),
            m_elong_smooth: Vec::new(),
            m_elong_kernel: [0.; 49],
            m_kern_dim_elong: 0,
            m_ix_elong: Vec::new(),
            m_iy_elong: Vec::new(),
            m_elong_mask: Vec::new(),
            m_fm: FortModel::default(),
        }
    }

    /// `FB3d::main` (`findbeads3d.cpp:31`).
    pub fn main(&mut self, argv: &[String]) -> i32 {
        const MAX_PIECE: usize = 400;
        const LIM_HISTO: usize = 10000;
        const MAX_LINE: usize = 120;
        let mut len_piece = [[0i32; MAX_PIECE]; 3];
        let mut ind0 = [[0i32; MAX_PIECE]; 3];
        let mut ind1 = [[0i32; MAX_PIECE]; 3];
        let (nx, ny, nz): (i32, i32, i32);
        let mut nxyz = [0i32; 3];
        let lim_peak: i32;
        let mut max_array: i32;
        let mut histo = vec![0f32; LIM_HISTO];
        let mut mxyz = [0i32; 3];
        let mut mode = 0i32;
        let max_shift: i32;
        let mut index = 0i32;
        let mut num_look = 0i32;
        let mut ierr = 0i32;
        let mut num_xpieces = 0i32;
        let mut num_ypieces = 0i32;
        let mut num_zpieces = 0i32;
        let mut num_peaks = 0i32;
        let mut num_corrs: i32;
        let (mut ix_start, mut ix_end, mut iy_start, mut iy_end, mut iz_start, mut iz_end) =
            (0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
        let mut elongation: f32;
        let radius: f32;
        let dist_min: f32;
        let (mut xpeak, mut ypeak, mut zpeak): (f32, f32, f32);
        let elong_cap_limit: f32;
        let elong_fail_limit: f32;
        let (mut dmin, mut dmax, mut dmean) = (0f32, 0f32, 0f32);
        let mut peak_corr: f32 = 0.;
        let mut avg_fallback: f32;
        let mut store_fallback: f32;
        let (mut ix_min, mut ix_max, mut iy_min, mut iy_max, mut iz_min, mut iz_max);
        let mut ix: i32;
        let min_inside: i32;
        let mut max_xsize: i32;
        let (mut ix_peak, mut iy_peak, mut iz_peak): (i32, i32, i32);
        let num_save: i32;
        let (mut ix0, mut ix1, mut iy0, mut iy1, mut iz0, mut iz1): (i32, i32, i32, i32, i32, i32);
        let num_pass: i32;
        let mut min_guess: i32;
        let mut i_verbose: i32;
        let mut ibinning: i32;
        let mut num3_corr_threads: i32;
        let mut light_beads: i32;
        let mut y_elongated: i32;
        let (mut sum_xoffset, mut sum_yoffset, mut sum_zoffset) = (0f32, 0f32, 0f32);
        let mut store_thresh: f32;
        let mut bead_size: f32 = 0.;
        let deg_to_rad: f32;
        let mut hist_dip: f32 = 0.;
        let mut peak_below: f32 = 0.;
        let mut peak_above: f32 = 0.;
        let (mut dx_adjusted, mut dy_adjusted, mut dz_adjusted) = (0f32, 0f32, 0f32);
        let mut avg_thresh: f32;
        let mut peak_rel_min: f32;
        let mut sep_min: f32;
        let mut black_thresh: f32;
        let mut above_avg: f32 = 0.;
        let mut above_sd: f32 = 0.;
        let mut expand_factor: f32;
        let mut nxyz_templ = [0i32; 3];
        let (mut nx_templ, mut ny_templ, mut nz_templ) = (0i32, 0i32, 0i32);
        let (mut templ_xcen, mut templ_ycen, mut templ_zcen) = (0i32, 0i32, 0i32);
        let (mut nx_for_ovlap, mut ny_for_ovlap, nz_for_ovlap): (i32, i32, i32);
        let (mut amin, mut amax, mut amean) = (0f32, 0f32, 0f32);
        let (mut tmin, mut tmax, mut tmean) = (0f32, 0f32, 0f32);
        let (mut dscale, mut dadd): (f32, f32);
        let (mut ann_inner, mut ann_outer) = (0f32, 0f32);
        let mut rad_pix: f32;
        let mut diameter: f32;
        let mut cg_radius: f32 = 0.;
        let mut cg_edge_diam_frac: f32;
        let mut cg_gap_diam_frac: f32;
        let mut cg_edge_width: f32 = 0.;
        let mut cg_gap_width: f32 = 0.;
        let elong_sigma: f32;
        let mut lim_edge: i32 = 0;
        let mut limcg: i32 = 0;
        let mut peak_ccc: f64 = 0.;
        let mut line = [0u8; MAX_LINE];
        let mut loaded: bool;
        let mut found = false;
        let mut do_integral: bool;
        let mut do_template: bool;
        let mut clean_both: i32;
        let mut model_file: Option<String> = None;
        let mut first_file: Option<String> = None;
        let mut text_output: Option<String> = None;
        let mut num_opt_arg = 0i32;
        let mut num_non_opt_arg = 0i32;
        //
        // fallbacks ../manpages/autodoc2man 2 1  findbeads3d
        //
        let num_options = 24;
        let options: [&[u8]; 24] = [
            b"input:InputFile:FN:",
            b"output:OutputFile:FN:",
            b"candidate:CandidateModel:FN:",
            b"size:BeadSize:F:",
            b"binning:BinningOfVolume:I:",
            b"expanded:ExpandedByFactor:F:",
            b"xminmax:XMinAndMax:IP:",
            b"yminmax:YMinAndMax:IP:",
            b"zminmax:ZMinAndMax:IP:",
            b"light:LightBeads:B:",
            b"angle:AngleRange:FP:",
            b"tilt:TiltFile:FN:",
            b"ylong:YAxisElongated:B:",
            b"peakmin:MinRelativeStrength:F:",
            b"threshold:ThresholdForAveraging:F:",
            b"store:StorageThreshold:F:",
            b"fallback:FallbackThresholds:FP:",
            b"spacing:MinSpacing:F:",
            b"both:EliminateBoth:B:",
            b"guess:GuessNumBeads:I:",
            b"max:MaxNumBeads:I:",
            b"verbose:VerboseOutput:I:",
            b"param:ParameterFile:PF:",
            b"help:usage:B:",
        ];
        //
        deg_to_rad = RADIANS_PER_DEGREE as f32;
        elongation = 1.;
        y_elongated = 0;
        self.m_polarity = -1.;
        light_beads = 0;
        clean_both = 0;
        min_inside = 64;
        peak_rel_min = 0.05;
        num_pass = 1;
        let mut max_peaks: i32 = 50000;
        sep_min = 0.9;
        avg_thresh = -2.;
        store_thresh = 0.;
        avg_fallback = 0.;
        store_fallback = 0.;
        i_verbose = 0;
        min_guess = 0;
        ibinning = 1;
        expand_factor = 1.;
        max_array = 200000000;
        elong_cap_limit = 2.5;
        elong_fail_limit = 5.;
        do_integral = false;
        do_template = false;
        cg_edge_diam_frac = 0.1;
        cg_gap_diam_frac = 0.;
        elong_sigma = 0.85;
        //
        // Pip startup: set error, parse options, do help output
        //
        let argv_bytes = argv
            .iter()
            .map(|value| value.as_bytes().to_vec())
            .collect::<Vec<_>>();
        pip_read_or_parse_options(
            argv_bytes.len() as i32,
            &argv_bytes,
            &options,
            num_options,
            b"findbeads3d",
            2,
            1,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
            Some(imod_usage_header_for_pip),
        );
        allocate_fort_model(&mut self.m_fm);
        //
        // Open image file
        //
        let mut image_file: Vec<u8> = Vec::new();
        if pip_get_in_out_file(b"InputFile", 0, &mut image_file) != 0 {
            exit_error(b"No input file specified");
        }

        let mut out_name: Vec<u8> = Vec::new();
        ierr = pip_get_in_out_file(b"OutputFile", 1, &mut out_name);
        if ierr == 0 {
            model_file = Some(String::from_utf8_lossy(&out_name).into_owned());
        }
        unsafe { iiu_open(1, &String::from_utf8_lossy(&image_file), "RO") };
        iiu_print_header(1, Some("Input volume"));
        printf!("\n");
        unsafe {
            iiu_ret_basic_head(
                1,
                nxyz.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                &mut mode,
                &mut dmin,
                &mut dmax,
                &mut dmean,
            );
        }
        nx = nxyz[0];
        ny = nxyz[1];
        nz = nxyz[2];

        let delta = iiu_ret_delta(1);
        let origin = iiu_ret_origin(1);
        let cur_tilt = iiu_ret_tilt(1);

        if pip_get_float(b"BeadSize", &mut bead_size) != 0 {
            exit_error(b"Bead diameter must be entered");
        }
        pip_get_integer(b"BinningOfVolume", &mut ibinning);
        if ibinning < 1 {
            exit_error(b"Binning must be positive");
        }
        pip_get_float(b"ExpandedByFactor", &mut expand_factor);
        if (expand_factor as f64) < 0.02 {
            exit_error(b"Entry for -expanded factor must be positive");
        }
        bead_size = expand_factor * bead_size / ibinning as f32;
        if expand_factor != 1. {
            printf!(
                "Adjusted bead size for binning and expansion factor to %.2f\n",
                CArg::Dbl(bead_size as f64)
            );
        }
        radius = (bead_size as f64 / 2.) as f32;

        let mut templ_name: Vec<u8> = Vec::new();
        if pip_get_string(b"TemplateFile", &mut templ_name) == 0 {
            unsafe { iiu_open(3, &String::from_utf8_lossy(&templ_name), "RO") };
            iiu_print_header(3, Some("Template file"));
            printf!("\n");
            unsafe {
                iiu_ret_basic_head(
                    3,
                    nxyz_templ.as_mut_ptr(),
                    mxyz.as_mut_ptr(),
                    &mut mode,
                    &mut dmin,
                    &mut dmax,
                    &mut dmean,
                );
            }
            nx_templ = nxyz_templ[0];
            ny_templ = nxyz_templ[1];
            nz_templ = nxyz_templ[2];
            do_template = true;
            templ_xcen = nx_templ / 2;
            templ_ycen = ny_templ / 2;
            templ_zcen = nz_templ / 2;
            pip_get_three_integers(
                b"CenterOfTemplateInXYZ",
                &mut templ_xcen,
                &mut templ_ycen,
                &mut templ_zcen,
            );
            let mut text_name: Vec<u8> = Vec::new();
            if pip_get_string(b"TextOutputFile", &mut text_name) == 0 {
                text_output = Some(String::from_utf8_lossy(&text_name).into_owned());
            }
            do_template = true;
            if pip_get_two_floats(b"AnnulusRadii", &mut ann_inner, &mut ann_outer) == 0 {
                do_integral = true;
                if ann_inner == 0. {
                    ann_inner = (radius as f64 + 1.5) as f32;
                }
                if ann_outer == 0. {
                    ann_outer = (ann_inner as f64 + b3dmax!(1.5, 0.667 * radius as f64)) as f32;
                }
                if ann_inner <= radius || (ann_outer as f64) < ann_inner as f64 + 0.9 {
                    exit_error(
                        b"Annulus inner radius must be >= bead radius, outer must be >= inner + 0.9",
                    );
                }

                // Allocate and compute factors for elongation computation too
                cg_edge_diam_frac = 0.1;
                cg_gap_diam_frac = 0.;
                diameter = (2. * radius as f64) as f32;
                cg_radius = (0.5 * (diameter as f64 + b3dmax!(3., 0.15 * diameter as f64))) as f32;
                cg_edge_width = b3dmax!(1.5, (cg_edge_diam_frac * diameter) as f64) as f32;
                cg_gap_width = cg_gap_diam_frac * diameter;
                // `pow(float, 2.f)` is the C++ float overload, folded by gcc to
                // a float square.
                let outer = cg_radius + cg_gap_width + cg_edge_width;
                let inner = cg_radius + cg_gap_width;
                lim_edge = (3.5 * (outer * outer - inner * inner) as f64 + 22.) as i32;
                self.m_idx_edge = vec![0; lim_edge.max(0) as usize];
                self.m_idy_edge = vec![0; lim_edge.max(0) as usize];
                self.m_edge_pixels = vec![0.; lim_edge.max(0) as usize];
                limcg = (cg_radius + 4.) as i32;
            }
        }

        pip_get_boolean(b"YAxisElongated", &mut y_elongated);
        //
        pip_get_float(b"MinRelativeStrength", &mut peak_rel_min);
        peak_rel_min = b3dmax!(0., peak_rel_min as f64).sqrt() as f32;
        pip_get_float(b"StorageThreshold", &mut store_thresh);
        pip_get_float(b"ThresholdForAveraging", &mut avg_thresh);
        //
        pip_get_boolean(b"LightBeads", &mut light_beads);
        if light_beads != 0 {
            self.m_polarity = 1.;
        }
        let mut cand_name: Vec<u8> = Vec::new();
        if pip_get_string(b"CandidateModel", &mut cand_name) == 0 {
            first_file = Some(String::from_utf8_lossy(&cand_name).into_owned());
        }
        //
        pip_get_float(b"MinSpacing", &mut sep_min);
        dist_min = sep_min * bead_size;
        pip_get_boolean(b"EliminateBoth", &mut clean_both);
        pip_get_integer(b"GuessNumBeads", &mut min_guess);
        pip_get_integer(b"MaxNumBeads", &mut max_peaks);
        pip_get_integer(b"VerboseOutput", &mut i_verbose);
        pip_get_two_floats(
            b"FallbackThresholds",
            &mut avg_fallback,
            &mut store_fallback,
        );
        ierr = pip_get_two_floats(b"AngleRange", &mut dx_adjusted, &mut dy_adjusted);
        let mut tilt_name: Vec<u8> = Vec::new();
        ix = pip_get_string(b"TiltFile", &mut tilt_name);
        if ierr == 0 || ix == 0 {
            if ierr + ix == 0 {
                exit_error(b"You cannot enter both an angle range and a tilt file");
            }
            if ix == 0 {
                let tilt_file = String::from_utf8_lossy(&tilt_name).into_owned();
                let Some(mut fp) = ImodFile::open(&tilt_file, "r") else {
                    exit_error_fmt!("Opening tilt file %s", CArg::Str(&tilt_file));
                };
                let mut array = vec![0f32; 2000];
                ierr = read_lines_for_values(
                    &mut fp,
                    &mut ix,
                    2000,
                    &mut line,
                    MAX_LINE as i32,
                    RLFV_SEPARATE_LINES,
                    "f",
                    &mut [ReadValueArray::Floats(&mut array)],
                );
                if ierr != 0 {
                    exit_from_value_read_error(ierr, "tilt angles");
                }
                dx_adjusted = 2000.;
                dy_adjusted = -2000.;
                for i in 0..ix as usize {
                    dx_adjusted = b3dmin!(dx_adjusted, array[i]);
                    dy_adjusted = b3dmax!(dy_adjusted, array[i]);
                }
                drop(fp);
            }
            //
            // Elongation factor from Radermacher 1988 paper
            // cryoposition looks for 'Elongation factor is'
            dz_adjusted = (0.5
                * (b3dabs!(dx_adjusted) + b3dabs!(dy_adjusted)) as f64
                * deg_to_rad as f64) as f32;
            elongation = ((dz_adjusted + dz_adjusted.cos() * dz_adjusted.sin())
                / (dz_adjusted - dz_adjusted.cos() * dz_adjusted.sin()))
            .sqrt();
            if elongation < elong_cap_limit {
                printf!("Elongation factor is %.2f\n", CArg::Dbl(elongation as f64));
            } else if elongation < elong_fail_limit {
                printf!(
                    "Elongation factor computed to be %.2f; limiting it to %.2f\n",
                    CArg::Dbl(elongation as f64),
                    CArg::Dbl(elong_cap_limit as f64)
                );
                elongation = elong_cap_limit;
            } else {
                exit_error_fmt!(
                    "An angular range of %.2f degrees is too low for finding gold",
                    CArg::Dbl((dy_adjusted - dx_adjusted) as f64)
                );
            }
        }

        ix_min = 1;
        iy_min = 1;
        iz_min = 1;
        ix_max = nx;
        iy_max = ny;
        iz_max = nz;
        pip_get_two_integers(b"XMinAndMax", &mut ix_min, &mut ix_max);
        pip_get_two_integers(b"YMinAndMax", &mut iy_min, &mut iy_max);
        pip_get_two_integers(b"ZMinAndMax", &mut iz_min, &mut iz_max);
        if ix_min < 1
            || ix_max > nx
            || ((ix_max - ix_min) as f32) < 2. * bead_size
            || iy_min < 1
            || iy_max > ny
            || ((iy_max - iy_min) as f32) < 2. * bead_size
            || iz_min < 1
            || iz_max > nz
            || ((iz_max - iz_min) as f32) < 2. * bead_size
        {
            exit_error(
                b"Coordinate min and max values are out of range or define too small a volume",
            );
        }
        if max_peaks < 3 {
            exit_error(b"The -MaxNumBeads entry is negative or too small");
        }
        lim_peak = max_peaks + 10;
        let lim_peak_u = lim_peak as usize;
        let mut ind_peak = vec![0i32; lim_peak_u];
        let mut ind_corr = vec![0i32; lim_peak_u];
        let mut peak_val = vec![0f32; lim_peak_u];
        let mut peak_pos = vec![0f32; 3 * lim_peak_u];
        let mut corr_val = vec![0f32; lim_peak_u];
        let mut corr_pos = vec![0f32; 3 * lim_peak_u];
        let mut corr_ccc: Vec<f32> = Vec::new();
        let mut integrals: Vec<f32> = Vec::new();
        let mut ccc2ds: Vec<f32> = Vec::new();
        let mut elongations: Vec<f32> = Vec::new();
        if do_template {
            corr_ccc = vec![0.; lim_peak_u];
        }
        if do_integral {
            integrals = vec![0.; lim_peak_u];
            ccc2ds = vec![0.; lim_peak_u];
            elongations = vec![0.; lim_peak_u];
        }
        //
        pip_done();
        //
        // Determine the correlation box size
        //
        self.m_nx_corr = 2 * b3dnint!(1.5 * radius as f64 + 0.5);
        self.m_ny_corr = self.m_nx_corr;
        self.m_nz_corr = 2 * b3dnint!((0.5 + elongation as f64) * radius as f64 + 0.5);
        // `mNxCorr + B3DCHOICE(doIntegral, 2 * B3DMAX(...) - radius, 0)`: the
        // conditional has type float, so the int adds as float and truncates.
        nx_for_ovlap = if do_integral {
            (self.m_nx_corr as f32
                + ((2 * b3dmax!((ann_outer as f64 + 1.).ceil() as i32, limcg + 2)) as f32 - radius))
                as i32
        } else {
            (self.m_nx_corr as f32 + 0.) as i32
        };
        ny_for_ovlap = nx_for_ovlap;
        let mut nz_for_ovlap_v = self.m_nz_corr;
        if y_elongated != 0 {
            self.m_ny_corr = self.m_nz_corr;
            self.m_nz_corr = self.m_nx_corr;
            ny_for_ovlap = self.m_nz_corr;
            nz_for_ovlap_v = nx_for_ovlap;
        }
        nz_for_ovlap = nz_for_ovlap_v;
        if do_template {
            if templ_xcen - self.m_nx_corr / 2 < 0
                || templ_xcen + self.m_nx_corr / 2 > nx_templ
                || templ_ycen - self.m_ny_corr / 2 < 0
                || templ_ycen + self.m_ny_corr / 2 > ny_templ
                || templ_zcen - self.m_nz_corr / 2 < 0
                || templ_zcen + self.m_nz_corr / 2 > nz_templ
            {
                exit_error(
                    b"Template center is too close to edge of volume to extract full correlation box",
                );
            }
            if do_integral {
                // Allocate for elongation
                self.m_elong_box_size = 2 * limcg + 6;
                ix = self.m_elong_box_size * self.m_elong_box_size;
                self.m_elong_temp = vec![0.; ix as usize];
                self.m_elong_smooth = vec![0.; ix as usize];
                self.m_elong_mask = vec![0; ix as usize];
                self.m_ix_elong = vec![0; ix as usize];
                self.m_iy_elong = vec![0; ix as usize];
                scaled_gaussian_kernel(
                    &mut self.m_elong_kernel,
                    &mut self.m_kern_dim_elong,
                    7,
                    elong_sigma,
                );

                // Make list of edge pixels for elongation
                self.m_num_edge = 0;
                for iy in -limcg..=limcg {
                    if b3dabs!(iy) > ny_for_ovlap / 2 - 2 {
                        continue;
                    }
                    ix = -limcg;
                    while ix <= limcg {
                        if b3dabs!(ix) > self.m_nx_corr / 2 - 2 {
                            ix += 1;
                            continue;
                        }
                        rad_pix = ((ix as f64 - 0.5) * (ix as f64 - 0.5)
                            + (iy as f64 - 0.5) * (iy as f64 - 0.5))
                            .sqrt() as f32;
                        if rad_pix > cg_radius + cg_gap_width
                            && rad_pix <= cg_radius + cg_gap_width + cg_edge_width
                        {
                            self.m_idx_edge[self.m_num_edge as usize] = ix;
                            self.m_idy_edge[self.m_num_edge as usize] = iy;
                            self.m_num_edge += 1;
                        }
                        if self.m_num_edge >= lim_edge {
                            exit_error(b"Programmer error computing size of centroid arrays");
                        }
                        ix += 1;
                    }
                }
            }
        }

        num3_corr_threads = b3dmax!(
            1,
            b3dmin!(
                8,
                b3dnint!(
                    2. * ((self.m_nx_corr * self.m_ny_corr * self.m_nz_corr) as f64 / 32000.)
                        .powf(0.45)
                )
            )
        );
        num3_corr_threads = num_omp_threads(num3_corr_threads);

        {
            let a = max_array as f32 as f64;
            let b = (nx as f64 + 4.) * (ny as f64 + 4.) * (nz as f64 + 4.);
            max_array = (if a < b { a } else { b }) as i32;
        }
        let mut array = vec![0f32; max_array.max(0) as usize];
        let ncorr_tot = (self.m_nx_corr * self.m_ny_corr * self.m_nz_corr) as usize;
        let mut average = vec![0f32; ncorr_tot];
        let max_vol = max_array - self.m_nx_corr * self.m_ny_corr * self.m_nz_corr;
        //
        // Given correlation dimensions, set up the overlaps and minimum sizes
        //
        let nx_overlap = 2 * ((nx_for_ovlap + 1) / 2 + 1);
        let ny_overlap = 2 * ((ny_for_ovlap + 1) / 2 + 1);
        let nz_overlap = 2 * ((nz_for_ovlap + 1) / 2 + 1);
        let mut min_ysize = min_inside + ny_overlap;
        let mut min_zsize = min_inside + nz_overlap;
        let max_z = max_vol / (nx * ny) - 3;
        let max_yz = (((9. + 4. * max_vol as f64 / nx as f64).sqrt() - 3.) / 2.) as i32;
        if i_verbose > 0 {
            printf!(
                "%d %d %d %d %d %d\n",
                CArg::Int(max_vol as i64),
                CArg::Int(nx_overlap as i64),
                CArg::Int(min_ysize as i64),
                CArg::Int(min_zsize as i64),
                CArg::Int(max_z as i64),
                CArg::Int(max_yz as i64)
            );
        }
        //
        if max_z >= nz {
            //
            // the entire load will fit at once, set min's to ny and nz
            //
            min_zsize = nz;
            min_ysize = ny;
        } else if max_z / 2 >= min_zsize {
            //
            // X/Y planes will fit; set miny to ny and increase minZsize
            //
            min_ysize = ny;
            min_zsize = max_z / 2;
        } else if max_yz / 2 >= min_zsize && max_yz / 2 >= min_ysize {
            //
            // X rows will fit in their entirety; increase the min sizes
            //
            min_ysize = max_yz / 2;
            min_zsize = max_yz / 2;
        }
        //
        // Get the definition of the pieces
        //
        {
            let [_, lp1, lp2] = &mut len_piece;
            let [_, a1, a2] = &mut ind0;
            let [_, b1, b2] = &mut ind1;
            self.define_pieces(
                iz_min,
                iz_max,
                nz_overlap,
                min_zsize,
                0,
                MAX_PIECE as i32,
                &mut num_zpieces,
                lp2,
                a2,
                b2,
                &mut ierr,
            );
            if ierr != 0 {
                exit_error(b"Too many pieces for array in Z dimension");
            }
            self.define_pieces(
                iy_min,
                iy_max,
                ny_overlap,
                min_ysize,
                0,
                MAX_PIECE as i32,
                &mut num_ypieces,
                lp1,
                a1,
                b1,
                &mut ierr,
            );
            if ierr != 0 {
                exit_error(b"Too many pieces for array in Y dimension");
            }
        }
        max_xsize = max_vol / (len_piece[1][0] * (len_piece[2][0] + 3));
        self.define_pieces(
            ix_min,
            ix_max,
            nx_overlap,
            0,
            max_xsize,
            MAX_PIECE as i32,
            &mut num_xpieces,
            &mut len_piece[0],
            &mut ind0[0],
            &mut ind1[0],
            &mut ierr,
        );
        if ierr != 0 {
            exit_error(b"Too many pieces for array in X dimension");
        }
        //
        // Redefine the longest dimension of Y or Z
        if ny > nz {
            max_xsize = max_vol / (len_piece[0][0] * (len_piece[2][0] + 3));
            if i_verbose > 0 {
                printf!("New max for y %d\n", CArg::Int(max_xsize as i64));
            }
            self.define_pieces(
                iy_min,
                iy_max,
                ny_overlap,
                0,
                max_xsize,
                MAX_PIECE as i32,
                &mut num_ypieces,
                &mut len_piece[1],
                &mut ind0[1],
                &mut ind1[1],
                &mut ierr,
            );
        } else {
            max_xsize = max_vol / (len_piece[0][0] * (len_piece[1][0] + 3));
            if i_verbose > 0 {
                printf!("New max for z %d\n", CArg::Int(max_xsize as i64));
            }
            self.define_pieces(
                iz_min,
                iz_max,
                nz_overlap,
                0,
                max_xsize,
                MAX_PIECE as i32,
                &mut num_zpieces,
                &mut len_piece[2],
                &mut ind0[2],
                &mut ind1[2],
                &mut ierr,
            );
        }

        if ierr != 0 || len_piece[0][0] * len_piece[1][0] * (len_piece[2][0] + 3) > max_vol {
            exit_error(b"Bug in dividing volume into pieces");
        }

        let nsum = b3dmax!(1, b3dnint!(0.75 * radius as f64));
        if i_verbose > 0 {
            printf!(
                "%d %d %d %d %d %d\n",
                CArg::Int(num_xpieces as i64),
                CArg::Int(num_ypieces as i64),
                CArg::Int(num_zpieces as i64),
                CArg::Int(nx_overlap as i64),
                CArg::Int(ny_overlap as i64),
                CArg::Int(nz_overlap as i64)
            );
            for (axis, count) in [num_xpieces, num_ypieces, num_zpieces]
                .into_iter()
                .enumerate()
            {
                for i in 0..count as usize {
                    printf!(
                        "%d %d %d\n",
                        CArg::Int(ind0[axis][i] as i64),
                        CArg::Int(ind1[axis][i] as i64),
                        CArg::Int(len_piece[axis][i] as i64)
                    );
                }
            }
            printf!("nsum = %d\n", CArg::Int(nsum as i64));
        }
        num_peaks = 0;
        let ind_planes = len_piece[0][0] * len_piece[1][0] * len_piece[2][0] + 1;
        //
        // Scan for peaks in simple pixel sums
        //
        for iz_piece in 0..num_zpieces as usize {
            for iy_piece in 0..num_ypieces as usize {
                for ix_piece in 0..num_xpieces as usize {
                    self.loadvol(
                        1,
                        &mut array,
                        len_piece[0][ix_piece],
                        len_piece[1][iy_piece],
                        ind0[0][ix_piece],
                        ind1[0][ix_piece],
                        ind0[1][iy_piece],
                        ind1[1][iy_piece],
                        ind0[2][iz_piece],
                        ind1[2][iz_piece],
                    );
                    self.get_analysis_limits(
                        ind0[0][ix_piece],
                        ind0[0][0],
                        ind1[0][ix_piece],
                        nx,
                        nx_overlap,
                        nsum,
                        &mut ix_start,
                        &mut ix_end,
                    );
                    self.get_analysis_limits(
                        ind0[1][iy_piece],
                        ind0[1][0],
                        ind1[1][iy_piece],
                        ny,
                        ny_overlap,
                        nsum,
                        &mut iy_start,
                        &mut iy_end,
                    );
                    self.get_analysis_limits(
                        ind0[2][iz_piece],
                        ind0[2][0],
                        ind1[2][iz_piece],
                        nz,
                        nz_overlap,
                        nsum,
                        &mut iz_start,
                        &mut iz_end,
                    );
                    let (vol, planes) = array.split_at_mut(ind_planes as usize);
                    self.find_pixel_sum_peaks(
                        vol,
                        planes,
                        len_piece[0][ix_piece],
                        len_piece[1][iy_piece],
                        len_piece[2][iz_piece],
                        ix_start,
                        ix_end,
                        iy_start,
                        iy_end,
                        iz_start,
                        iz_end,
                        ind0[0][ix_piece],
                        ind0[1][iy_piece],
                        ind0[2][iz_piece],
                        nsum,
                        &mut ind_peak,
                        &mut peak_val,
                        &mut peak_pos,
                        max_peaks,
                        &mut num_peaks,
                        peak_rel_min,
                    );
                }
            }
        }
        //
        // remove points that are too close to each other afer normalizing
        //
        printf!("%d candidate peaks found\n", CArg::Int(num_peaks as i64));
        peak_corr = peak_val[ind_peak[0] as usize];
        for i in 0..num_peaks as usize {
            ix = ind_peak[i];
            peak_val[ix as usize] /= peak_corr;
        }

        self.clean_sorted_list(
            &mut ind_peak,
            &mut peak_val,
            &peak_pos,
            &mut num_peaks,
            dist_min,
            clean_both,
            peak_rel_min,
        );
        printf!(
            "%d candidate peaks left after eliminating close points\n",
            CArg::Int(num_peaks as i64)
        );
        let _ = ImodFile::Stdout.flush();

        // Copy correlation values from repacked index and do histogram (bug fix 2/13/25)
        for i in 0..num_peaks as usize {
            corr_val[i] = peak_val[ind_peak[i] as usize];
        }
        let dip = find_histogram_dip(
            &corr_val[..num_peaks.max(0) as usize],
            min_guess,
            &mut histo,
            0.,
            1.,
            b3dmax!(0, i_verbose - 2),
        );
        ierr = match dip {
            Some(d) => {
                hist_dip = d.dip;
                peak_below = d.peak_below;
                peak_above = d.peak_above;
                0
            }
            None => 1,
        };
        black_thresh = hist_dip;
        //
        if ierr != 0 && avg_thresh < 0. && avg_fallback > 0. {
            printf!(
                "No histogram dip found for initial peaks, using fallback averaging threshold\n"
            );
            avg_thresh = avg_fallback;
        }
        if avg_thresh < 0. {
            if ierr != 0 {
                exit_error(
                    b"No histogram dip found for initial peaks; enter positive -thresh to proceed",
                );
            }
            if avg_thresh < -1. {
                self.find_value_in_list(
                    &corr_val,
                    num_peaks,
                    (hist_dip as f64 + (peak_above - hist_dip) as f64 / 4.) as f32,
                    &mut num_look,
                );
            } else {
                self.find_value_in_list(&corr_val, num_peaks, hist_dip, &mut num_look);
                num_look = b3dmax!(1, b3dnint!(-avg_thresh * num_look as f32));
            }
        } else {
            //
            // Find number to average for number between 0 and 1
            if avg_thresh <= 1. {
                self.find_value_in_list(&corr_val, num_peaks, avg_thresh, &mut num_look);
            } else {
                num_look = b3dnint!(avg_thresh);
            }
            if ierr != 0 {
                black_thresh = 0.;
            }
        }
        printf!(
            "%d peaks being averaged to %s\n",
            CArg::Int(num_look as i64),
            CArg::Str(if do_template {
                "determine scaling for template"
            } else {
                "make reference for correlation"
            })
        );
        self.write_peak_model(
            first_file.as_deref(),
            &ind_peak,
            &peak_val,
            &peak_pos,
            num_peaks,
            black_thresh,
            &nxyz,
            &delta,
            &origin,
            &cur_tilt,
            radius,
        );
        //
        // Now loop through the pieces again looking for points and getting
        // correlation positions
        //
        max_shift = b3dmax!(8, b3dnint!(radius));
        num_corrs = 0;
        num_save = num_peaks;
        for loop_corr in 1..=2 * num_pass {
            //
            // Zero the array on odd loops
            if (loop_corr % 2) == 1 {
                for v in average.iter_mut() {
                    *v = 0.;
                }
            }
            //
            // Copy correlation positions to peak positions on loop 3
            if loop_corr == 3 {
                for i in 0..num_corrs as usize {
                    let ixu = ind_corr[i] as usize;
                    ind_peak[i] = ind_corr[i];
                    peak_val[ixu] = corr_val[ixu];
                    peak_pos[ixu * 3] = corr_pos[ixu * 3];
                    peak_pos[ixu * 3 + 1] = corr_pos[ixu * 3 + 1];
                    peak_pos[ixu * 3 + 2] = corr_pos[ixu * 3 + 2];
                }
                num_peaks = num_corrs;
                num_corrs = 0;
            }
            if loop_corr == 2 * num_pass {
                num_look = num_save;
            }
            if i_verbose > 0 {
                cout_line(&[
                    (
                        "loopCorr",
                        c_format_bytes("%d", &[CArg::Int(loop_corr as i64)]),
                    ),
                    (
                        "numLook",
                        c_format_bytes("%d", &[CArg::Int(num_look as i64)]),
                    ),
                ]);
            }
            for iz_piece in 0..num_zpieces as usize {
                for iy_piece in 0..num_ypieces as usize {
                    for ix_piece in 0..num_xpieces as usize {
                        loaded = num_xpieces * num_ypieces * num_zpieces == 1;
                        self.get_analysis_limits(
                            ind0[0][ix_piece],
                            ind0[0][0],
                            ind1[0][ix_piece],
                            nx,
                            nx_overlap,
                            self.m_nx_corr,
                            &mut ix_start,
                            &mut ix_end,
                        );
                        self.get_analysis_limits(
                            ind0[1][iy_piece],
                            ind0[1][0],
                            ind1[1][iy_piece],
                            ny,
                            ny_overlap,
                            self.m_ny_corr,
                            &mut iy_start,
                            &mut iy_end,
                        );
                        self.get_analysis_limits(
                            ind0[2][iz_piece],
                            ind0[2][0],
                            ind1[2][iz_piece],
                            nz,
                            nz_overlap,
                            self.m_nz_corr,
                            &mut iz_start,
                            &mut iz_end,
                        );
                        //
                        // Loop on points, for ones inside the box, get correlation
                        //
                        if i_verbose > 0 {
                            printf!(
                                "%d %d %d %d %d %d\n",
                                CArg::Int(ix_start as i64),
                                CArg::Int(ix_end as i64),
                                CArg::Int(iy_start as i64),
                                CArg::Int(iy_end as i64),
                                CArg::Int(iz_start as i64),
                                CArg::Int(iz_end as i64)
                            );
                        }
                        let (lx, ly, lz) = (
                            len_piece[0][ix_piece],
                            len_piece[1][iy_piece],
                            len_piece[2][iz_piece],
                        );
                        for i in 0..num_look.max(0) {
                            let ixu = ind_peak[i as usize] as usize;
                            xpeak = peak_pos[ixu * 3] - ind0[0][ix_piece] as f32;
                            ypeak = peak_pos[ixu * 3 + 1] - ind0[1][iy_piece] as f32;
                            zpeak = peak_pos[ixu * 3 + 2] - ind0[2][iz_piece] as f32;
                            // Add the -1 to account for 0-based index wanted here
                            ix_peak = b3dnint!(xpeak) - 1;
                            iy_peak = b3dnint!(ypeak) - 1;
                            iz_peak = b3dnint!(zpeak) - 1;
                            if ix_peak >= ix_start
                                && ix_peak <= ix_end
                                && iy_peak >= iy_start
                                && iy_peak <= iy_end
                                && iz_peak >= iz_start
                                && iz_peak <= iz_end
                            {
                                if !loaded {
                                    self.loadvol(
                                        1,
                                        &mut array,
                                        lx,
                                        ly,
                                        ind0[0][ix_piece],
                                        ind1[0][ix_piece],
                                        ind0[1][iy_piece],
                                        ind1[1][iy_piece],
                                        ind0[2][iz_piece],
                                        ind1[2][iz_piece],
                                    );
                                    loaded = true;
                                }

                                if (loop_corr % 2) == 1 {
                                    //
                                    // On the first round from pixel sums, get centroid before
                                    // adding peak in
                                    if loop_corr == 1 {
                                        ix0 = b3dmax!(0, ix_peak - self.m_nx_corr / 2);
                                        ix1 = b3dmin!(lx, ix0 + self.m_nx_corr) - 1;
                                        iy0 = b3dmax!(0, iy_peak - self.m_ny_corr / 2);
                                        iy1 = b3dmin!(ly, iy0 + self.m_ny_corr) - 1;
                                        iz0 = b3dmax!(0, iz_peak - self.m_nz_corr / 2);
                                        iz1 = b3dmin!(lz, iz0 + self.m_nz_corr) - 1;
                                        self.find_peak_center(
                                            &array,
                                            lx,
                                            ly,
                                            lz,
                                            ix0,
                                            ix1,
                                            iy0,
                                            iy1,
                                            iz0,
                                            iz1,
                                            &mut dx_adjusted,
                                            &mut dy_adjusted,
                                            &mut dz_adjusted,
                                        );
                                        // change - 1 to + 1 to get from pixel coord to real
                                        // center coord
                                        xpeak = ((ix1 + ix0 + 1) as f64 / 2. + dx_adjusted as f64)
                                            as f32;
                                        ypeak = ((iy1 + iy0 + 1) as f64 / 2. + dy_adjusted as f64)
                                            as f32;
                                        zpeak = ((iz1 + iz0 + 1) as f64 / 2. + dz_adjusted as f64)
                                            as f32;
                                    }
                                    self.add_peak_to_sum(
                                        &mut average,
                                        self.m_nx_corr,
                                        self.m_ny_corr,
                                        self.m_nz_corr,
                                        &array,
                                        lx,
                                        ly,
                                        lz,
                                        xpeak,
                                        ypeak,
                                        zpeak,
                                    );
                                    if i == num_look - 1 && loop_corr == 1 {
                                        if i_verbose > 0 {
                                            self.vol_write(
                                                17,
                                                "voladded.st",
                                                &average,
                                                self.m_nx_corr,
                                                self.m_ny_corr,
                                                self.m_nz_corr,
                                            );
                                        }
                                        if let Some((a, b, c)) =
                                            with_mrc_data!(average, |d| full_array_min_max_mean(
                                                &mut d,
                                                MRC_MODE_FLOAT,
                                                self.m_nx_corr * self.m_ny_corr,
                                                self.m_nz_corr
                                            ))
                                        {
                                            (amin, amax, amean) = (a, b, c);
                                        }
                                        // Read in the template
                                        if do_template {
                                            let nxy = (self.m_nx_corr * self.m_ny_corr) as usize;
                                            for iz in 0..self.m_nz_corr {
                                                unsafe {
                                                    iiu_set_position(
                                                        3,
                                                        iz + templ_zcen - self.m_nz_corr / 2,
                                                        0,
                                                    );
                                                }
                                                let err = unsafe {
                                                    iiu_read_sec_part(
                                                        3,
                                                        average
                                                            .as_mut_ptr()
                                                            .add(iz as usize * nxy)
                                                            .cast(),
                                                        self.m_nx_corr,
                                                        templ_xcen - self.m_nx_corr / 2,
                                                        templ_xcen + self.m_nx_corr / 2 - 1,
                                                        templ_ycen - self.m_ny_corr / 2,
                                                        templ_ycen + self.m_ny_corr / 2 - 1,
                                                    )
                                                };
                                                if err != 0 {
                                                    exit_error_fmt!(
                                                        "Reading section %d from template file",
                                                        CArg::Int(
                                                            (iz + templ_zcen - self.m_nz_corr / 2)
                                                                as i64
                                                        )
                                                    );
                                                }
                                            }
                                            if let Some((a, b, c)) =
                                                with_mrc_data!(
                                                    average,
                                                    |d| full_array_min_max_mean(
                                                        &mut d,
                                                        MRC_MODE_FLOAT,
                                                        self.m_nx_corr * self.m_ny_corr,
                                                        self.m_nz_corr
                                                    )
                                                )
                                            {
                                                (tmin, tmax, tmean) = (a, b, c);
                                            }
                                            dscale = (amax - amin) / (tmax - tmin);
                                            dadd = amin - dscale * tmin;
                                            for v in average.iter_mut() {
                                                *v = *v * dscale + dadd;
                                            }
                                            if let Some((a, b, c)) =
                                                with_mrc_data!(
                                                    average,
                                                    |d| full_array_min_max_mean(
                                                        &mut d,
                                                        MRC_MODE_FLOAT,
                                                        self.m_nx_corr * self.m_ny_corr,
                                                        self.m_nz_corr
                                                    )
                                                )
                                            {
                                                (tmin, tmax, tmean) = (a, b, c);
                                            }
                                            if i_verbose > 0 {
                                                let g = |v: f32| {
                                                    c_format_bytes("%g", &[CArg::Dbl(v as f64)])
                                                };
                                                cout_line(&[
                                                    ("amin", g(amin)),
                                                    ("amax", g(amax)),
                                                    ("tmin", g(tmin)),
                                                    ("tmax", g(tmax)),
                                                ]);
                                            }
                                        }
                                    }
                                } else {
                                    self.find_best_corr(
                                        &average,
                                        self.m_nx_corr,
                                        self.m_ny_corr,
                                        self.m_nz_corr,
                                        &array,
                                        lx,
                                        ly,
                                        lz,
                                        ix_start,
                                        ix_end,
                                        iy_start,
                                        iy_end,
                                        iz_start,
                                        iz_end,
                                        &mut xpeak,
                                        &mut ypeak,
                                        &mut zpeak,
                                        max_shift,
                                        &mut found,
                                        &mut peak_corr,
                                        num3_corr_threads,
                                        if do_template {
                                            Some(&mut peak_ccc)
                                        } else {
                                            None
                                        },
                                    );
                                    if found && peak_corr >= 0. {
                                        self.add_to_sorted_list(
                                            &mut ind_corr,
                                            &mut corr_val,
                                            &mut num_corrs,
                                            max_peaks,
                                            peak_rel_min,
                                            peak_corr.sqrt(),
                                            &mut index,
                                        );
                                        if index >= 0 {
                                            let iu = index as usize;
                                            corr_pos[iu * 3] =
                                                xpeak + ind0[0][ix_piece] as f32 + sum_xoffset;
                                            corr_pos[iu * 3 + 1] =
                                                ypeak + ind0[1][iy_piece] as f32 + sum_yoffset;
                                            corr_pos[iu * 3 + 2] =
                                                zpeak + ind0[2][iz_piece] as f32 + sum_zoffset;
                                            if do_template {
                                                corr_ccc[iu] = peak_ccc as f32;
                                            }
                                            if do_integral {
                                                let (mut integ, mut elong, mut c2d) =
                                                    (integrals[iu], elongations[iu], ccc2ds[iu]);
                                                self.integral_in_plane(
                                                    &average,
                                                    &array,
                                                    lx,
                                                    ly,
                                                    lz,
                                                    y_elongated,
                                                    xpeak,
                                                    ypeak,
                                                    zpeak,
                                                    radius,
                                                    ann_inner,
                                                    ann_outer,
                                                    &mut integ,
                                                    &mut elong,
                                                    &mut c2d,
                                                );
                                                integrals[iu] = integ;
                                                elongations[iu] = elong;
                                                ccc2ds[iu] = c2d;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (loop_corr % 2) == 1 {
                self.find_peak_center(
                    &average,
                    self.m_nx_corr,
                    self.m_ny_corr,
                    self.m_nz_corr,
                    0,
                    self.m_nx_corr - 1,
                    0,
                    self.m_ny_corr - 1,
                    0,
                    self.m_nz_corr - 1,
                    &mut sum_xoffset,
                    &mut sum_yoffset,
                    &mut sum_zoffset,
                );
                if i_verbose > 0 {
                    printf!(
                        "Offsets: %f %f %f\n",
                        CArg::Dbl(sum_xoffset as f64),
                        CArg::Dbl(sum_yoffset as f64),
                        CArg::Dbl(sum_zoffset as f64)
                    );
                }
                xcorr_mean_zero(
                    &mut average,
                    self.m_nx_corr,
                    self.m_nx_corr,
                    self.m_ny_corr * self.m_nz_corr,
                );
                if i_verbose > 0 {
                    self.vol_write(
                        17,
                        "volsum.st",
                        &average,
                        self.m_nx_corr,
                        self.m_ny_corr,
                        self.m_nz_corr,
                    );
                }
            }
        }
        //
        printf!(
            "%d peaks found by correlation\n",
            CArg::Int(num_corrs as i64)
        );
        self.clean_sorted_list(
            &mut ind_corr,
            &mut corr_val,
            &corr_pos,
            &mut num_corrs,
            dist_min,
            clean_both,
            peak_rel_min,
        );
        printf!(
            "%d peaks left after eliminating close points\n",
            CArg::Int(num_corrs as i64)
        );
        peak_corr = corr_val[ind_corr[0] as usize];
        for i in 0..num_corrs as usize {
            let ixu = ind_corr[i] as usize;
            corr_val[ixu] /= peak_corr;
            peak_val[i] = corr_val[ixu];
            if i_verbose > 1 && i < 50 {
                printf!(
                    "%8.4f %7.1f %7.1f %7.1f\n",
                    CArg::Dbl(corr_val[ixu] as f64),
                    CArg::Dbl(corr_pos[ixu * 3] as f64),
                    CArg::Dbl(corr_pos[ixu * 3 + 1] as f64),
                    CArg::Dbl(corr_pos[ixu * 3 + 2] as f64)
                );
            }
        }
        //
        let _ = ImodFile::Stdout.flush();
        let dip = find_histogram_dip(
            &peak_val[..num_corrs.max(0) as usize],
            min_guess,
            &mut histo,
            0.,
            1.,
            b3dmax!(0, i_verbose - 2),
        );
        ierr = match dip {
            Some(d) => {
                hist_dip = d.dip;
                peak_below = d.peak_below;
                peak_above = d.peak_above;
                0
            }
            None => 1,
        };
        black_thresh = hist_dip;
        if let Some(text_output) = text_output.as_deref() {
            imod_backup_file(text_output);
            let Some(mut fp) = ImodFile::open(text_output, "w") else {
                exit_error_fmt!("Opening file for text output %s", CArg::Str(text_output));
            };
            for i in 0..num_corrs as usize {
                let ixu = ind_corr[i] as usize;
                let _ = fp.write_all(&c_format_bytes(
                    "%8.2f %8.2f %8.2f %.5f %.5f%s",
                    &[
                        CArg::Dbl(corr_pos[ixu * 3] as f64),
                        CArg::Dbl(corr_pos[ixu * 3 + 1] as f64),
                        CArg::Dbl(corr_pos[ixu * 3 + 2] as f64),
                        CArg::Dbl(corr_val[ixu] as f64),
                        CArg::Dbl(corr_ccc[ixu] as f64),
                        CArg::Str(if do_integral { "" } else { "\n" }),
                    ],
                ));
                if do_integral {
                    let _ = fp.write_all(&c_format_bytes(
                        " %11.5f %7.3f %.5f\n",
                        &[
                            CArg::Dbl(integrals[ixu] as f64),
                            CArg::Dbl(elongations[ixu] as f64),
                            CArg::Dbl(ccc2ds[ixu] as f64),
                        ],
                    ));
                }
            }
            let _ = fp.flush();
            drop(fp);
        }
        //
        // cryoposition looks for 'using fallback storage threshold'
        // and for 'Storing' and 'peaks in model' in one line
        if ierr != 0 && store_thresh <= 0. && store_fallback > 0. {
            printf!(
                "No dip found in histogram of correlation peaks, using fallback storage threshold\n"
            );
            store_thresh = store_fallback;
        }
        if store_thresh <= 0. {
            if ierr != 0 {
                exit_error(
                    b"No dip found in histogram of correlation peaks; enter -store with positive value to proceed",
                );
            }
            self.find_value_in_list(&peak_val, num_corrs, hist_dip, &mut num_look);
            printf!(
                "%d peaks are above the histogram dip\n",
                CArg::Int(num_look as i64)
            );
            if store_thresh < 0. {
                num_peaks = b3dmax!(
                    1,
                    b3dmin!(num_corrs, b3dnint!(-store_thresh * num_look as f32))
                );
                printf!("Storing %d peaks in model\n", CArg::Int(num_peaks as i64));
            } else {
                avg_sd(
                    &peak_val,
                    num_look,
                    &mut above_avg,
                    &mut above_sd,
                    &mut peak_below,
                );
                peak_below = (above_avg as f64 - 5. * above_sd as f64) as f32;
                self.find_value_in_list(&peak_val, num_corrs, peak_below, &mut num_peaks);
                num_peaks = b3dmax!(num_look, b3dmin!(2 * num_look, num_peaks));
                printf!(
                    "Storing an additional %d  peaks in model down to a value of %.4f\n",
                    CArg::Int((num_peaks - num_look) as i64),
                    CArg::Dbl(peak_below as f64)
                );
            }
        } else {
            self.find_value_in_list(
                &peak_val,
                num_corrs,
                b3dmin!(1., store_thresh as f64) as f32,
                &mut num_peaks,
            );
            if ierr != 0 {
                black_thresh = 0.;
            }
            printf!(
                "Storing %d peaks in model above threshold of %.4f\n",
                CArg::Int(num_peaks as i64),
                CArg::Dbl(store_thresh as f64)
            );
        }

        self.write_peak_model(
            model_file.as_deref(),
            &ind_corr,
            &corr_val,
            &corr_pos,
            num_peaks,
            black_thresh,
            &nxyz,
            &delta,
            &origin,
            &cur_tilt,
            radius,
        );
        let _ = (
            mxyz, amean, tmean, peak_below, above_avg, index, dmin, dmax, dmean, ind0, lim_peak,
        );
        let _ = ImodFile::Stdout.flush();
        crate::imod::libcfshr::b3dutil::exit(0);
    }

    /// `FB3d::definePieces` (`findbeads3d.cpp:760`): how to divide an extent
    /// into overlapping pieces.  `ind0`/`ind1` are numbered from zero.
    #[allow(clippy::too_many_arguments)]
    fn define_pieces(
        &self,
        ind_min: i32,
        ind_max: i32,
        n_overlap: i32,
        min_size: i32,
        max_size: i32,
        max_pieces: i32,
        num_pieces: &mut i32,
        len_piece: &mut [i32],
        ind0: &mut [i32],
        ind1: &mut [i32],
        ierr: &mut i32,
    ) {
        let n_total: i32;
        let mut isize: i32;
        let irem: i32;
        //
        // If a minimum size is defined, find the biggest number of pieces that
        // divide into sizes bigger than the minimum
        //
        n_total = ind_max + 1 - ind_min;
        *ierr = 1;
        if min_size > 0 {
            isize = min_size + 1;
            *num_pieces = 0;
            while *num_pieces < max_pieces && isize >= min_size {
                *num_pieces += 1;
                isize = (n_total + (*num_pieces - 1) * n_overlap) / *num_pieces;
            }
            if isize >= min_size {
                return;
            }
            if *num_pieces > 1 {
                *num_pieces -= 1;
            }
        } else {
            //
            // Otherwise just compute the number from the maximum size
            //
            *num_pieces = (n_total - max_size) / (max_size - n_overlap) + 1;
            if *num_pieces <= 0 {
                *num_pieces = 1;
            }
            isize = (n_total + (*num_pieces - 1) * n_overlap + *num_pieces - 1) / *num_pieces;
            if isize > max_size {
                *num_pieces += 1;
            }
            if *num_pieces > max_pieces {
                return;
            }
        }
        //
        // get basic size and remainder to distribute, loop on pieces
        //
        isize = (n_total + (*num_pieces - 1) * n_overlap) / *num_pieces;
        irem = (n_total + (*num_pieces - 1) * n_overlap) % *num_pieces;
        ind0[0] = ind_min - 1;
        for i in 0..*num_pieces as usize {
            len_piece[i] = isize;
            if irem > i as i32 {
                len_piece[i] += 1;
            }
            if i > 0 {
                ind0[i] = ind0[i - 1] + len_piece[i - 1] - n_overlap;
            }
            ind1[i] = ind0[i] + len_piece[i] - 1;
        }
        *ierr = 0;
    }

    /// `FB3d::loadvol` (`findbeads3d.cpp:824`): loads a subset of the volume
    /// on `iunit` into `array`, dimensioned `nx_dim` by `ny_dim`.
    #[allow(clippy::too_many_arguments)]
    fn loadvol(
        &self,
        iunit: i32,
        array: &mut [f32],
        nx_dim: i32,
        ny_dim: i32,
        ix0: i32,
        ix1: i32,
        iy0: i32,
        iy1: i32,
        iz0: i32,
        iz1: i32,
    ) {
        let mut indz: i32 = 0;
        for iz in iz0..=iz1 {
            unsafe { iiu_set_position(iunit, iz, 0) };
            let offset = (nx_dim * ny_dim * indz) as usize;
            let len = ((ix1 + 1 - ix0) * (iy1 + 1 - iy0)) as usize;
            assert!(offset + len <= array.len());
            if unsafe {
                iiu_read_sec_part(
                    iunit,
                    array.as_mut_ptr().add(offset).cast(),
                    nx_dim,
                    ix0,
                    ix1,
                    iy0,
                    iy1,
                )
            } != 0
            {
                exit_error(b"Reading image file");
            }
            indz += 1;
        }
    }

    /// `FB3d::addToSortedList` (`findbeads3d.cpp:852`): maintains `values`
    /// with an ordering in `index`.  `new_index` is -1 when the value is not
    /// added (fixed in translation, see below).
    #[allow(clippy::too_many_arguments)]
    fn add_to_sorted_list(
        &self,
        index: &mut [i32],
        values: &mut [f32],
        num_vals: &mut i32,
        max_vals: i32,
        peak_rel_min: f32,
        val_new: f32,
        new_index: &mut i32,
    ) {
        let (mut less, mut more, mut itest): (i32, i32, i32);
        let new_order: i32;
        //
        // quick test for full list and less than the last item on it
        // And for value less than threshold for storing
        //
        // Fixed in translation (BUGS.md): the source sets `newIndex = 0`
        // here (`findbeads3d.cpp:875`, a 1-based "0 = not added" carried
        // over), but 0 is a valid slot and both callers test `index >= 0`
        // before storing the position, so every rejected candidate moved
        // stored peak 0 to its own position.  -1 is returned instead.
        *new_index = -1;
        if *num_vals == max_vals && values[index[(max_vals - 1) as usize] as usize] >= val_new {
            return;
        }
        if *num_vals > 0 && val_new < peak_rel_min * values[index[0] as usize] {
            return;
        }
        //
        // Handle simple cases of inserting at the front or end of the list
        //
        if *num_vals == 0 || val_new >= values[index[0] as usize] {
            new_order = 0;
        } else if values[index[(*num_vals - 1) as usize] as usize] >= val_new {
            new_order = *num_vals;
        } else {
            //
            // Otherwise search for position.  Set up index of one more and one
            // less than the value.  Divide the interval in two and revise less
            // or more until there is no longer an interval between them
            //
            more = 0;
            less = *num_vals - 1;
            while less - more > 1 {
                itest = (less + more) / 2;
                let v = values[index[itest as usize] as usize];
                if v == val_new {
                    more = itest;
                    less = itest;
                } else if v < val_new {
                    less = itest;
                } else {
                    more = itest;
                }
            }
            new_order = less;
        }
        //
        // If space exists, position is at end of list; otherwise it is the
        // position occupied by the one being bumped off
        //
        if *num_vals < max_vals {
            *new_index = *num_vals;
            *num_vals += 1;
        } else {
            *new_index = index[(max_vals - 1) as usize];
        }
        values[*new_index as usize] = val_new;
        //
        // shift index array up if necessary
        //
        let mut i = *num_vals - 1;
        while i > new_order {
            index[i as usize] = index[(i - 1) as usize];
            i -= 1;
        }
        index[new_order as usize] = *new_index;
    }

    /// `FB3d::cleanSortedList` (`findbeads3d.cpp:936`): eliminates peaks that
    /// are too close to a stronger one.
    #[allow(clippy::too_many_arguments)]
    fn clean_sorted_list(
        &self,
        index: &mut [i32],
        peak_val: &mut [f32],
        peak_pos: &[f32],
        num_vals: &mut i32,
        dist_min: f32,
        clean_both: i32,
        peak_rel_min: f32,
    ) {
        let (mut indi, mut indj): (i32, i32);
        let (mut xx, mut yy, mut zz, mut dx, mut dy, mut dz): (f32, f32, f32, f32, f32, f32);
        let mut knockout: bool;
        //
        let sqr_min = dist_min * dist_min;
        let thresh = peak_rel_min * peak_val[index[0] as usize];
        //
        // Scan from strongest down, knocking out peaks if point is too close
        //
        let mut j: i32 = 0;
        while j < *num_vals - 1 {
            let ju = j as usize;
            indj = index[ju];
            knockout = false;
            if indj >= 0 && peak_val[indj as usize] < thresh {
                indj = -1;
                index[ju] = -1;
                knockout = true;
            }
            if indj >= 0 {
                let jb = indj as usize * 3;
                xx = peak_pos[jb];
                yy = peak_pos[jb + 1];
                zz = peak_pos[jb + 2];
                for i in (j + 1) as usize..*num_vals as usize {
                    indi = index[i];
                    if indi >= 0 && peak_val[indi as usize] >= thresh {
                        let ib = indi as usize * 3;
                        dx = xx - peak_pos[ib];
                        if b3dabs!(dx) < dist_min {
                            dy = yy - peak_pos[ib + 1];
                            if b3dabs!(dy) < dist_min {
                                dz = zz - peak_pos[ib + 2];
                                if b3dabs!(dz) < dist_min && dx * dx + dy * dy + dz * dz < sqr_min {
                                    index[i] = -1;
                                    knockout = true;
                                }
                            }
                        }
                    }
                }
            }
            if knockout && clean_both != 0 {
                index[ju] = -1;
            }
            j += 1;
        }
        //
        // repack index
        //
        let mut jj = 0usize;
        for i in 0..(*num_vals).max(0) as usize {
            if index[i] >= 0 {
                index[jj] = index[i];
                jj += 1;
            }
        }
        *num_vals = jj as i32;
    }

    /// `FB3d::getAnalysisLimits` (`findbeads3d.cpp:1000`): the 0-based limits
    /// for analysing one chunk of the tomogram.  `ind_min` (the start of the
    /// axis's first piece) is not in the source; see below.
    #[allow(clippy::too_many_arguments)]
    fn get_analysis_limits(
        &self,
        ind0: i32,
        ind_min: i32,
        ind1: i32,
        n_total: i32,
        n_overlap: i32,
        ncorr: i32,
        i_start: &mut i32,
        i_end: &mut i32,
    ) {
        // Fixed in translation (BUGS.md).  The source tests `ind0 > 0` for
        // "a piece was loaded below this one", which is false for the first
        // piece of a `-zminmax`/`-xminmax`/`-yminmax` range starting above 1,
        // and then starts at `nOverlap / 2 - 1`.  `find_best_corr` reads
        // `threecorrs` rows down to `iStart - 1 - ncorr / 2`, so for an even
        // `ncorr` whose overlap is `ncorr + 2` that start reaches row -1 of
        // the piece: before the allocation in Z (native segfaults), the
        // previous row in X/Y.  The first piece of the range is treated as
        // the boundary it is, and an inner piece starts no lower than the
        // boundary start `1 + ncorr / 2`; the previous piece already analyses
        // up to one row below that, so nothing is skipped.
        if ind0 > ind_min {
            *i_start = (n_overlap / 2 - 1).max(1 + ncorr / 2);
        } else {
            *i_start = 1 + ncorr / 2;
        }
        if ind1 < n_total - 1 {
            *i_end = ind1 - ind0 - n_overlap / 2;
        } else {
            *i_end = ind1 - ind0 - ncorr / 2 - 1;
        }
    }

    /// `FB3d::findPixelSumPeaks` (`findbeads3d.cpp:1034`): peaks in rolling
    /// sums over cubes `nsum` on a side.  `planes` holds three planes of sums.
    #[allow(clippy::too_many_arguments)]
    fn find_pixel_sum_peaks(
        &self,
        array: &[f32],
        planes: &mut [f32],
        nx: i32,
        ny: i32,
        nz: i32,
        ix_start: i32,
        ix_end: i32,
        iy_start: i32,
        iy_end: i32,
        iz_start: i32,
        iz_end: i32,
        ix_offset: i32,
        iy_offset: i32,
        iz_offset: i32,
        nsum: i32,
        ind_peak: &mut [i32],
        peak_val: &mut [f32],
        peak_pos: &mut [f32],
        max_peaks: i32,
        num_peaks: &mut i32,
        peak_rel_min: f32,
    ) {
        let mut cen: f32;
        let mut sum: f32;
        let (mut ip1, mut ip2, mut ip3): (i32, i32, i32);
        let mut index: i32 = 0;
        let (mut ix0, mut ix1, mut iy0, mut iy1, mut iz0, mut iz1);

        //
        let mut next_plane: i32 = 0;
        let num_before = (nsum - 1) / 2;
        let num_after = nsum / 2;

        // Shift from pixel index coordinates to real coordinates: changed - 1 to + 1
        let pos_shift = (0.5 * (num_after - num_before + 1) as f64) as f32;
        let sum_scale = self.m_polarity / (nsum * nsum * nsum) as f32;
        let nxu = nx as usize;
        let nxy = nx as usize * ny as usize;
        let pl = |ix: i32, iy: i32, ip: i32| -> usize { (ix + nx * (iy + ny * ip)) as usize };
        //
        // Loop on the Z planes with extended limits
        //
        for iz in iz_start - 1..=iz_end + 1 {
            //
            // first compute the sums on the next plane
            //
            for iy in iy_start - 1..=iy_end + 1 {
                for ix in ix_start - 1..=ix_end + 1 {
                    sum = 0.;
                    for jz in iz - num_before..=iz + num_after {
                        for jy in iy - num_before..=iy + num_after {
                            let base = jy as usize * nxu + jz as usize * nxy;
                            let row = &array[base + (ix - num_before) as usize
                                ..=base + (ix + num_after) as usize];
                            for &v in row {
                                sum += v;
                            }
                        }
                    }
                    planes[pl(ix, iy, next_plane)] = sum * sum_scale;
                }
            }
            //
            // If at least 3 planes have been computed, now search for peaks
            //
            if iz > iz_start {
                ip1 = (next_plane + 1) % 3;
                ip2 = (next_plane + 2) % 3;
                ip3 = next_plane;
                for iy in iy_start..=iy_end {
                    for ix in ix_start..=ix_end {
                        cen = planes[pl(ix, iy, ip2)];
                        //
                        // Test for whether greater than all diagonal and ones ahead
                        // in X, Y, or Z, and >= ones behind in X, Y, or Z
                        //
                        let p = |dx: i32, dy: i32, ip: i32| planes[pl(ix + dx, iy + dy, ip)];
                        if cen >= p(-1, 0, ip2)
                            && cen > p(1, 0, ip2)
                            && cen >= p(0, -1, ip2)
                            && cen > p(0, 1, ip2)
                            && cen >= p(0, 0, ip1)
                            && cen > p(0, 0, ip3)
                            && cen > p(0, 1, ip1)
                            && cen >= p(0, -1, ip1)
                            && cen >= p(-1, 0, ip1)
                            && cen >= p(1, 0, ip1)
                            && cen >= p(1, 0, ip3)
                            && cen >= p(-1, 0, ip3)
                            && cen > p(0, 1, ip3)
                            && cen >= p(0, -1, ip3)
                            && cen >= p(-1, -1, ip2)
                            && cen >= p(1, -1, ip2)
                            && cen >= p(1, 1, ip2)
                            && cen >= p(-1, 1, ip2)
                            && cen >= p(-1, -1, ip1)
                            && cen >= p(1, -1, ip1)
                            && cen >= p(1, 1, ip1)
                            && cen >= p(-1, 1, ip1)
                            && cen >= p(-1, -1, ip3)
                            && cen >= p(1, -1, ip3)
                            && cen >= p(1, 1, ip3)
                            && cen >= p(-1, 1, ip3)
                        {
                            //
                            // Got a peak; try to add it to the list
                            //
                            ix0 = ix - self.m_nx_corr / 2;
                            ix1 = ix0 + self.m_nx_corr - 1;
                            iy0 = iy - self.m_ny_corr / 2;
                            iy1 = iy0 + self.m_ny_corr - 1;
                            iz0 = iz - self.m_nz_corr / 2 - 1;
                            iz1 = iz0 + self.m_nz_corr - 1;
                            if ix0 >= 0 && ix1 < nx && iy0 >= 0 && iy1 < ny && iz0 >= 0 && iz1 < nz
                            {
                                self.integrate_peak(
                                    array, nx, ny, nz, ix0, ix1, iy0, iy1, iz0, iz1, &mut cen,
                                );
                                if cen > 0. {
                                    self.add_to_sorted_list(
                                        ind_peak,
                                        peak_val,
                                        num_peaks,
                                        max_peaks,
                                        peak_rel_min,
                                        cen.sqrt(),
                                        &mut index,
                                    );
                                    if index >= 0 {
                                        let b = index as usize * 3;
                                        peak_pos[b] = (ix + ix_offset) as f32 + pos_shift;
                                        peak_pos[b + 1] = (iy + iy_offset) as f32 + pos_shift;

                                        // That - 1 is a worry
                                        peak_pos[b + 2] = (iz + iz_offset) as f32 + pos_shift - 1.;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            next_plane = (next_plane + 1) % 3;
        }
    }

    /// `FB3d::addPeakToSum` (`findbeads3d.cpp:1165`): adds the peak at
    /// `dx_adjusted`.. in `brray` into the sum in `array`.
    #[allow(clippy::too_many_arguments)]
    fn add_peak_to_sum(
        &self,
        array: &mut [f32],
        nxa: i32,
        nya: i32,
        nza: i32,
        brray: &[f32],
        nxb: i32,
        nyb: i32,
        nzb: i32,
        dx_adjusted: f32,
        dy_adjusted: f32,
        dz_adjusted: f32,
    ) {
        let idx = dx_adjusted as i32 - nxa / 2;
        let dx = dx_adjusted - dx_adjusted as i32 as f32;
        let idy = dy_adjusted as i32 - nya / 2;
        let dy = dy_adjusted - dy_adjusted as i32 as f32;
        let idz = dz_adjusted as i32 - nza / 2;
        let dz = dz_adjusted - dz_adjusted as i32 as f32;
        let d11 = ((1. - dx as f64) * (1. - dy as f64)) as f32;
        let d12 = ((1. - dx as f64) * dy as f64) as f32;
        let d21 = (dx as f64 * (1. - dy as f64)) as f32;
        let d22 = dx * dy;
        let omdz = 1. - dz as f64;
        let b = |ix: i32, iy: i32, iz: i32| brray[(ix + nxb * (iy + nyb * iz)) as usize];
        for iz in 0..nza {
            let izp = b3dmax!(0, iz + idz);
            let izp_p1 = b3dmin!(izp + 1, nzb);
            for iy in 0..nya {
                let iyp = b3dmax!(0, iy + idy);
                let iyp_p1 = b3dmin!(iyp + 1, nyb);
                for ix in 0..nxa {
                    let ixp = b3dmax!(0, ix + idx);
                    let ixp_p1 = b3dmin!(ixp + 1, nxb);
                    let s1 = d11 * b(ixp, iyp, izp)
                        + d12 * b(ixp, iyp_p1, izp)
                        + d21 * b(ixp_p1, iyp, izp)
                        + d22 * b(ixp_p1, iyp_p1, izp);
                    let s2 = d11 * b(ixp, iyp, izp_p1)
                        + d12 * b(ixp, iyp_p1, izp_p1)
                        + d21 * b(ixp_p1, iyp, izp_p1)
                        + d22 * b(ixp_p1, iyp_p1, izp_p1);
                    let a = &mut array[(ix + nxa * (iy + nya * iz)) as usize];
                    *a = (*a as f64 + (omdz * s1 as f64 + (dz * s2) as f64)) as f32;
                }
            }
        }
    }

    /// `FB3d::findPeakCenter` (`findbeads3d.cpp:1208`): the centroid of the
    /// pixels in the box above the background at its edges.
    #[allow(clippy::too_many_arguments)]
    fn find_peak_center(
        &self,
        array: &[f32],
        nxa: i32,
        nya: i32,
        nza: i32,
        ix0: i32,
        ix1: i32,
        iy0: i32,
        iy1: i32,
        iz0: i32,
        iz1: i32,
        dx_adjusted: &mut f32,
        dy_adjusted: &mut f32,
        dz_adjusted: &mut f32,
    ) {
        let mut edge: f32 = 0.;
        let mut diff: f32;
        //
        if iy0 < 0 || iy1 >= nya || ix0 < 0 || ix1 >= nxa || iz0 < 0 || iz1 >= nza {
            // Fixed in translation (BUGS.md): the source prints `iz0` first,
            // where `ix0` was meant.
            printf!(
                "bad findPeakCenter %d %d %d %d %d %d %d %d %d\n",
                CArg::Int(ix0 as i64),
                CArg::Int(ix1 as i64),
                CArg::Int(nxa as i64),
                CArg::Int(iy0 as i64),
                CArg::Int(iy1 as i64),
                CArg::Int(nya as i64),
                CArg::Int(iz0 as i64),
                CArg::Int(iz1 as i64),
                CArg::Int(nza as i64)
            );
        }
        self.peak_edge_mean(
            array, nxa, nya, nza, ix0, ix1, iy0, iy1, iz0, iz1, &mut edge,
        );
        //
        // Get weighted sum of pixel indexes
        let mut xsum: f32 = 0.;
        let mut ysum: f32 = 0.;
        let mut zsum: f32 = 0.;
        let mut wsum: f32 = 0.;
        for iz in iz0 + 1..=iz1 - 1 {
            for iy in iy0 + 1..=iy1 - 1 {
                for ix in ix0 + 1..=ix1 - 1 {
                    diff = self.m_polarity * (array[(ix + nxa * (iy + nya * iz)) as usize] - edge);
                    if diff > 0. {
                        wsum += diff;
                        xsum += diff * (ix - ix0) as f32;
                        ysum += diff * (iy - iy0) as f32;
                        zsum += diff * (iz - iz0) as f32;
                    }
                }
            }
        }
        //
        // Get offset from center
        *dx_adjusted = ((xsum / wsum) as f64 - (ix1 - ix0) as f64 / 2.) as f32;
        *dy_adjusted = ((ysum / wsum) as f64 - (iy1 - iy0) as f64 / 2.) as f32;
        *dz_adjusted = ((zsum / wsum) as f64 - (iz1 - iz0) as f64 / 2.) as f32;
    }

    /// `FB3d::integratePeak` (`findbeads3d.cpp:1246`): the integral of a peak
    /// relative to the background at the edges.
    #[allow(clippy::too_many_arguments)]
    fn integrate_peak(
        &self,
        array: &[f32],
        nx: i32,
        ny: i32,
        nz: i32,
        ix0: i32,
        ix1: i32,
        iy0: i32,
        iy1: i32,
        iz0: i32,
        iz1: i32,
        peak: &mut f32,
    ) {
        let mut edge: f32 = 0.;
        //
        if iy0 < 0 || iy1 >= ny || ix0 < 0 || ix1 >= nx || iz0 < 0 || iz1 >= nz {
            // Fixed in translation (BUGS.md): the source prints `iz0` first,
            // where `ix0` was meant.
            printf!(
                "integral %d %d %d %d %d %d %d %d %d\n",
                CArg::Int(ix0 as i64),
                CArg::Int(ix1 as i64),
                CArg::Int(nx as i64),
                CArg::Int(iy0 as i64),
                CArg::Int(iy1 as i64),
                CArg::Int(ny as i64),
                CArg::Int(iz0 as i64),
                CArg::Int(iz1 as i64),
                CArg::Int(nz as i64)
            );
        }
        self.peak_edge_mean(array, nx, ny, nz, ix0, ix1, iy0, iy1, iz0, iz1, &mut edge);
        *peak = 0.;
        let nxu = nx as usize;
        let nxy = nx as usize * ny as usize;
        for iz in iz0 + 1..=iz1 - 1 {
            for iy in iy0 + 1..=iy1 - 1 {
                let base = iy as usize * nxu + iz as usize * nxy;
                for &v in &array[base + (ix0 + 1) as usize..base + ix1 as usize] {
                    *peak += v - edge;
                }
            }
        }
        *peak *= self.m_polarity;
    }

    /// `FB3d::peakEdgeMean` (`findbeads3d.cpp:1269`): the mean along the walls
    /// of a box.
    #[allow(clippy::too_many_arguments)]
    fn peak_edge_mean(
        &self,
        array: &[f32],
        nx: i32,
        ny: i32,
        nz: i32,
        ix0: i32,
        ix1: i32,
        iy0: i32,
        iy1: i32,
        iz0: i32,
        iz1: i32,
        edge: &mut f32,
    ) {
        let mut edge_sum: f32 = 0.;
        if iy0 < 0 || iy1 >= ny || ix0 < 0 || ix1 >= nx || iz0 < 0 || iz1 >= nz {
            // Fixed in translation (BUGS.md): the source prints `iz0` first,
            // where `ix0` was meant.
            printf!(
                "bad edge mean %d %d %d %d %d %d %d %d %d\n",
                CArg::Int(ix0 as i64),
                CArg::Int(ix1 as i64),
                CArg::Int(nx as i64),
                CArg::Int(iy0 as i64),
                CArg::Int(iy1 as i64),
                CArg::Int(ny as i64),
                CArg::Int(iz0 as i64),
                CArg::Int(iz1 as i64),
                CArg::Int(nz as i64)
            );
        }
        let a = |ix: i32, iy: i32, iz: i32| array[(ix + nx * (iy + ny * iz)) as usize];
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                edge_sum += a(ix, iy, iz0) + a(ix, iy, iz1);
            }
        }
        for iz in iz0 + 1..=iz1 - 1 {
            for ix in ix0..=ix1 {
                edge_sum += a(ix, iy0, iz) + a(ix, iy1, iz);
            }
        }
        for iz in iz0 + 1..=iz1 - 1 {
            for iy in iy0 + 1..=iy1 - 1 {
                edge_sum += a(ix0, iy, iz) + a(ix1, iy, iz);
            }
        }
        *edge = edge_sum
            / (2 * ((ix1 + 1 - ix0) * (iy1 + 1 - iy0)
                + (ix1 + 1 - ix0) * (iz1 - 1 - iz0)
                + (iy1 - 1 - iy0) * (iz1 - 1 - iz0))) as f32;
    }

    /// `FB3d::find_best_corr` (`findbeads3d.cpp:1311`): searches for the
    /// location in `brray` with the highest correlation to `array`.
    #[allow(clippy::too_many_arguments)]
    fn find_best_corr(
        &self,
        array: &[f32],
        nxa: i32,
        nya: i32,
        nza: i32,
        brray: &[f32],
        nxb: i32,
        nyb: i32,
        _nzb: i32,
        ix_start: i32,
        ix_end: i32,
        iy_start: i32,
        iy_end: i32,
        iz_start: i32,
        iz_end: i32,
        dx_adjusted: &mut f32,
        dy_adjusted: &mut f32,
        dz_adjusted: &mut f32,
        max_shift: i32,
        found: &mut bool,
        peak_corr: &mut f32,
        num_threads: i32,
        ccc: Option<&mut f64>,
    ) {
        // The center index will be 2 for both sets of arrays
        let mut corrs = [[[0f64; 5]; 5]; 5];
        let mut corr_tmp = [[[0f64; 5]; 5]; 5];
        let mut corr_max: f64;
        let mut done = [[[false; 5]; 5]; 5];
        let mut done_tmp = [[[false; 5]; 5]; 5];
        let idy_sequence: [i32; 9] = [2, 1, 3, 2, 2, 1, 3, 1, 3];
        let idz_sequence: [i32; 9] = [2, 2, 2, 3, 1, 1, 1, 3, 3];
        //
        let (mut idx_global, mut idy_global, mut idz_global): (i32, i32, i32);
        let mut ind_sequence: i32;
        let (mut idy, mut idz): (usize, usize);
        let (mut idy_corr, mut idz_corr, mut idx_corr): (i32, i32, i32);
        let mut ind_max: usize;
        // Uninitialised in the source until the first maximum is recorded.
        let (mut idx_max, mut idy_max, mut idz_max) = (0i32, 0i32, 0i32);
        let (mut cx, mut y1, mut y2, mut y3, mut cy, mut cz): (f32, f32, f32, f32, f32, f32);
        //
        // Minimum # of rows to do in sequence before shifting center
        let min_seq = 5;
        //
        // get global displacement of b
        //
        idx_global = b3dnint!(*dx_adjusted);
        idy_global = b3dnint!(*dy_adjusted);
        idz_global = b3dnint!(*dz_adjusted);
        //
        // clear flags for existence of corr (the array initialiser)
        //
        corr_max = -1.0e30;
        ind_sequence = 0;
        while ind_sequence < 9 {
            idy = idy_sequence[ind_sequence as usize] as usize;
            idz = idz_sequence[ind_sequence as usize] as usize;
            if !(done[1][idy][idz] && done[2][idy][idz] && done[3][idy][idz]) {
                //
                // if the whole row does not exist, do the correlations
                // limit the extent if b is displaced and near an edge
                //
                idx_corr = idx_global - nxa / 2;
                idy_corr = idy_global + idy as i32 - 2 - nya / 2;
                idz_corr = idz_global + idz as i32 - 2 - nza / 2;
                let (mut c1, mut c2, mut c3) = (0f64, 0f64, 0f64);
                self.threecorrs(
                    array,
                    nxa,
                    nya,
                    brray,
                    nxb,
                    nyb,
                    0,
                    nxa - 1,
                    0,
                    nya - 1,
                    0,
                    nza - 1,
                    idx_corr,
                    idy_corr,
                    idz_corr,
                    &mut c1,
                    &mut c2,
                    &mut c3,
                    num_threads,
                );
                corrs[1][idy][idz] = c1;
                corrs[2][idy][idz] = c2;
                corrs[3][idy][idz] = c3;
                done[1][idy][idz] = true;
                done[2][idy][idz] = true;
                done[3][idy][idz] = true;
            }

            // Record indMax as the index, offset by 2
            if corrs[2][idy][idz] > corrs[1][idy][idz] && corrs[2][idy][idz] > corrs[3][idy][idz] {
                ind_max = 2;
            } else if corrs[1][idy][idz] > corrs[2][idy][idz]
                && corrs[1][idy][idz] > corrs[3][idy][idz]
            {
                ind_max = 1;
            } else {
                ind_max = 3;
            }
            if corrs[ind_max][idy][idz] > corr_max {
                corr_max = corrs[ind_max][idy][idz];

                // But keep these as actual offsets
                idx_max = ind_max as i32 - 2;
                idy_max = idy as i32 - 2;
                idz_max = idz as i32 - 2;
            }

            if ind_sequence >= min_seq - 1 && (idx_max != 0 || idy_max != 0 || idz_max != 0) {
                //
                // if there is a new maximum, after a minimum number of rows has
                // been done, shift the done flags and the existing
                // correlations, and reset the sequence
                //
                idx_global += idx_max;
                idy_global += idy_max;
                idz_global += idz_max;
                //
                // but if beyond the limit, return failure
                //
                if b3dabs!(idx_global as f32 - *dx_adjusted) > max_shift as f32
                    || b3dmax!(
                        b3dabs!(idy_global as f32 - *dy_adjusted),
                        b3dabs!(idz_global as f32 - *dz_adjusted)
                    ) > max_shift as f32
                    || idx_global < ix_start
                    || idx_global > ix_end
                    || idy_global < iy_start
                    || idy_global > iy_end
                    || idz_global < iz_start
                    || idz_global > iz_end
                {
                    *found = false;
                    return;
                }
                for iz in 1..=3 {
                    for iy in 1..=3 {
                        for ix in 1..=3 {
                            done_tmp[ix][iy][iz] = false;
                        }
                    }
                }
                for iz in 1..=3usize {
                    for iy in 1..=3usize {
                        for ix in 1..=3usize {
                            let (tx, ty, tz) = (ix + 2 - ind_max, iy + 2 - idy, iz + 2 - idz);
                            done_tmp[tx][ty][tz] = done[ix][iy][iz];
                            corr_tmp[tx][ty][tz] = corrs[ix][iy][iz];
                        }
                    }
                }
                for iz in 1..=3 {
                    for iy in 1..=3 {
                        for ix in 1..=3 {
                            done[ix][iy][iz] = done_tmp[ix][iy][iz];
                            corrs[ix][iy][iz] = corr_tmp[ix][iy][iz];
                        }
                    }
                }
                ind_sequence = -1;
                idx_max = 0;
                idy_max = 0;
                idz_max = 0;
            }
            ind_sequence += 1;
        }
        //
        // do independent parabolic fits in 3 dimensions
        //
        y1 = corrs[1][2][2] as f32;
        y2 = corrs[2][2][2] as f32;
        y3 = corrs[3][2][2] as f32;
        cx = parabolic_fit_position(y1, y2, y3) as f32;
        y1 = corrs[2][1][2] as f32;
        y3 = corrs[2][3][2] as f32;
        cy = parabolic_fit_position(y1, y2, y3) as f32;
        y1 = corrs[2][2][1] as f32;
        y3 = corrs[2][2][3] as f32;
        cz = parabolic_fit_position(y1, y2, y3) as f32;
        //
        *dx_adjusted = idx_global as f32 + cx;
        *dy_adjusted = idy_global as f32 + cy;
        *dz_adjusted = idz_global as f32 + cz;
        *peak_corr = y2;
        *found = true;
        let Some(ccc) = ccc else {
            return;
        };
        if cx < 0. {
            idx_global -= 1;
            cx = (cx as f64 + 1.) as f32;
        }
        if cy < 0. {
            idy_global -= 1;
            cy = (cy as f64 + 1.) as f32;
        }
        if cz < 0. {
            idz_global -= 1;
            cz = (cz as f64 + 1.) as f32;
        }
        idx_corr = idx_global - nxa / 2;
        idy_corr = idy_global - nya / 2;
        idz_corr = idz_global - nza / 2;
        self.one_corr_coeff(
            array,
            nxa,
            nya,
            brray,
            nxb,
            nyb,
            0,
            nxa - 1,
            0,
            nya - 1,
            0,
            nza - 1,
            idx_corr,
            idy_corr,
            idz_corr,
            cx,
            cy,
            cz,
            ccc,
            num_threads,
        );
    }

    /// `FB3d::threecorrs` (`findbeads3d.cpp:1497`): three correlations between
    /// `array` and `brray` at X shifts `idx - 1`, `idx`, `idx + 1`.
    ///
    /// The source's `#pragma omp parallel for reduction(+ : sum1, sum2, sum3)`
    /// over Z runs serially here: a reduction's combination order depends on
    /// the thread count.
    #[allow(clippy::too_many_arguments)]
    fn threecorrs(
        &self,
        array: &[f32],
        nxa: i32,
        nya: i32,
        brray: &[f32],
        nxb: i32,
        nyb: i32,
        ix0: i32,
        ix1: i32,
        iy0: i32,
        iy1: i32,
        iz0: i32,
        iz1: i32,
        idx: i32,
        idy: i32,
        idz: i32,
        corr1: &mut f64,
        corr2: &mut f64,
        corr3: &mut f64,
        _num_threads: i32,
    ) {
        let mut sum1: f64 = 0.;
        let mut sum2: f64 = 0.;
        let mut sum3: f64 = 0.;
        let n = (ix1 + 1 - ix0) as usize;
        // The source reads `brray[ixb - 1]` without a bound; with a `-zminmax`
        // start above 1 the analysis limits let the box reach plane -1 of the
        // loaded piece, before the start of the allocation, and the reference
        // binary segfaults there (`BUGS.md`).  That read is not reproducible.
        let first_b = (iz0 + idz) as i64 * nxb as i64 * nyb as i64
            + (iy0 + idy) as i64 * nxb as i64
            + (ix0 + idx - 1) as i64;
        if first_b < 0 {
            below_array_start_abort();
        }
        for iz in iz0..=iz1 {
            let izb = iz + idz;
            for iy in iy0..=iy1 {
                let iyb = iy + idy;
                let ind_base_a = iy * nxa + iz * nxa * nya;
                let ind_del_b = iyb * nxb + izb * nxb * nyb + idx - ind_base_a;

                // 11/28/14: deleted all the stuff for correlation coefficients, which are
                // not appropriate for featureless particles
                let a0 = (ind_base_a + ix0) as usize;
                let b0 = (ind_base_a + ix0 + ind_del_b - 1) as usize;
                let arow = &array[a0..a0 + n];
                let brow = &brray[b0..b0 + n + 2];
                for k in 0..n {
                    let a = arow[k];
                    sum1 += (a * brow[k]) as f64;
                    sum2 += (a * brow[k + 1]) as f64;
                    sum3 += (a * brow[k + 2]) as f64;
                }
            }
        }

        let nsum = (iz1 + 1 - iz0) * (iy1 + 1 - iy0) * (ix1 + 1 - ix0);
        *corr1 = sum1 / nsum as f64;
        *corr2 = sum2 / nsum as f64;
        *corr3 = sum3 / nsum as f64;
    }

    /// `FB3d::findValueInList` (`findbeads3d.cpp:1537`): returns in `index`
    /// the number of leading values `>= find_val`.
    fn find_value_in_list(&self, peak_val: &[f32], num_peaks: i32, find_val: f32, index: &mut i32) {
        let mut i: i32 = 0;
        while i < num_peaks && peak_val[i as usize] >= find_val {
            i += 1;
        }
        // Fixed in translation (BUGS.md): the source returns `i - 1`
        // (`findbeads3d.cpp:1539`), one less than the count -- the 1-based
        // loop's `i - 1` kept after the loop was made 0-based -- while every
        // caller uses the result as a count (the number averaged, "%d peaks
        // are above the histogram dip", the number stored).  The count is
        // returned.
        *index = i;
    }

    /// `FB3d::oneCorrCoeff` (`findbeads3d.cpp:1546`): the correlation
    /// coefficient at a fractional shift, by trilinear interpolation of
    /// `brray`.  The OpenMP reduction runs serially, as in `threecorrs`.
    #[allow(clippy::too_many_arguments)]
    fn one_corr_coeff(
        &self,
        array: &[f32],
        nxa: i32,
        nya: i32,
        brray: &[f32],
        nxb: i32,
        nyb: i32,
        ix0: i32,
        ix1: i32,
        iy0: i32,
        iy1: i32,
        iz0: i32,
        iz1: i32,
        idx: i32,
        idy: i32,
        idz: i32,
        cx: f32,
        cy: f32,
        cz: f32,
        ccc: &mut f64,
        _num_threads: i32,
    ) {
        let (mut ab_sum, mut a_sum, mut asum_sq, mut b_sum, mut bsum_sq) =
            (0f64, 0f64, 0f64, 0f64, 0f64);
        let mut denom: f64;
        let om_cx = (1. - cx as f64) as f32;
        let om_cy = (1. - cy as f64) as f32;
        let om_cz = (1. - cz as f64) as f32;
        let nxny = nxb * nyb;
        let nxbu = nxb as usize;
        let nxnyu = nxny as usize;
        // As in `threecorrs`: the source would read before the allocation.
        if ((iz0 + idz) as i64 * nxny as i64 + (iy0 + idy) as i64 * nxb as i64 + (ix0 + idx) as i64)
            < 0
        {
            below_array_start_abort();
        }
        for iz in iz0..=iz1 {
            let izb = iz + idz;
            for iy in iy0..=iy1 {
                let iyb = iy + idy;
                let ind_base_a = iy * nxa + iz * nxa * nya;
                let ind_del_b = iyb * nxb + izb * nxny + idx - ind_base_a;

                for ix in ind_base_a + ix0..=ind_base_a + ix1 {
                    let ixb = (ix + ind_del_b) as usize;
                    let bval = om_cz
                        * (om_cy * (om_cx * brray[ixb] + cx * brray[ixb + 1])
                            + cy * (om_cx * brray[ixb + nxbu] + cx * brray[ixb + nxbu + 1]))
                        + cz * (om_cy * (om_cx * brray[ixb + nxnyu] + cx * brray[ixb + 1 + nxnyu])
                            + cy * (om_cx * brray[ixb + nxbu + nxnyu]
                                + cx * brray[ixb + nxbu + 1 + nxnyu]));
                    let a = array[ix as usize];
                    a_sum += a as f64;
                    b_sum += bval as f64;
                    ab_sum += (a * bval) as f64;
                    asum_sq += (a * a) as f64;
                    bsum_sq += (bval * bval) as f64;
                }
            }
        }

        let nsum = ((iz1 + 1 - iz0) * (iy1 + 1 - iy0) * (ix1 + 1 - ix0)) as f64;
        denom = (nsum * asum_sq - a_sum * a_sum) * (nsum * bsum_sq - b_sum * b_sum);

        // Set the ccc to 0 if the denominator is illegal, otherwise limit it
        // to +/-1
        if denom <= 0. {
            *ccc = 0.;
        } else {
            denom = denom.sqrt();
            *ccc = nsum * ab_sum - a_sum * b_sum;
            if denom < *ccc {
                *ccc = if *ccc < 0. { -1. } else { 1. };
            } else {
                *ccc /= denom;
            }
        }
    }

    /// `FB3d::integralInPlane` (`findbeads3d.cpp:1596`): the bead integral,
    /// elongation and 2-D template correlation in the best plane.
    #[allow(clippy::too_many_arguments)]
    fn integral_in_plane(
        &mut self,
        average: &[f32],
        brray: &[f32],
        nxb: i32,
        nyb: i32,
        nzb: i32,
        y_elongated: i32,
        dx_adjusted: f32,
        dy_adjusted: f32,
        dz_adjusted: f32,
        radius: f32,
        ann_inner: f32,
        ann_outer: f32,
        integral: &mut f32,
        elongation: &mut f32,
        ccc2d: &mut f32,
    ) {
        let iplane = b3dnint!(if y_elongated != 0 {
            dy_adjusted
        } else {
            dz_adjusted
        });
        let dy_peak = if y_elongated != 0 {
            dz_adjusted
        } else {
            dy_adjusted
        };
        let ix_cen = b3dnint!(dx_adjusted);
        let iy_cen = b3dnint!(dy_peak);
        let (mut ix_use, mut iy_use): (i32, i32);
        let box_size = self.m_elong_box_size;
        let ix_off = ix_cen - box_size / 2;
        let iy_off = iy_cen - box_size / 2;
        // Uninitialised in the source if no plane beats -1.e10.
        let mut iz_best: i32 = 0;
        let xp_cen = dx_adjusted - ix_cen as f32;
        let yp_cen = dy_peak - iy_cen as f32;
        let (mut cen_mean, mut ann_mean, mut num_cen) = (0f32, 0f32, 0f32);
        let mut one_val: f32;
        let mut ccc: f64;
        *integral = -1.0e10;
        for iz in iplane - 1..=iplane + 1 {
            if y_elongated != 0 {
                one_val = bead_integral(
                    &brray[(nxb * iz) as usize..],
                    nxb * nyb,
                    nxb,
                    nzb,
                    radius,
                    ann_inner,
                    ann_outer,
                    dx_adjusted,
                    dz_adjusted,
                    &mut cen_mean,
                    &mut ann_mean,
                    None,
                    -1.,
                    Some(&mut num_cen),
                ) as f32;
                one_val *= num_cen * self.m_polarity;
                for iy in 0..box_size {
                    iy_use = b3dmin!(nzb - 1, b3dmax!(0, iy + iy_off));
                    for ix in 0..box_size {
                        ix_use = b3dmin!(nxb - 1, b3dmax!(0, ix + ix_off));
                        self.m_elong_temp[(ix + iy * box_size) as usize] =
                            brray[(nxb * iz + ix_use + nxb * nyb * iy_use) as usize];
                    }
                }
            } else {
                one_val = bead_integral(
                    &brray[(nxb * nyb * iz) as usize..],
                    nxb,
                    nxb,
                    nyb,
                    radius,
                    ann_inner,
                    ann_outer,
                    dx_adjusted,
                    dy_adjusted,
                    &mut cen_mean,
                    &mut ann_mean,
                    None,
                    -1.,
                    Some(&mut num_cen),
                ) as f32;
                one_val *= num_cen * self.m_polarity;
                for iy in 0..box_size {
                    iy_use = b3dmin!(nyb - 1, b3dmax!(0, iy + iy_off));
                    for ix in 0..box_size {
                        ix_use = b3dmin!(nxb - 1, b3dmax!(0, ix + ix_off));
                        self.m_elong_temp[(ix + iy * box_size) as usize] =
                            brray[(nxb * nyb * iz + ix_use + nxb * iy_use) as usize];
                    }
                }
            }
            if one_val > *integral {
                *integral = one_val;
                iz_best = iz;
                self.calc_elongation(xp_cen, yp_cen, elongation);
            }
        }

        *ccc2d = -1.0e10;
        let (nxc, nyc, nzc) = (self.m_nx_corr, self.m_ny_corr, self.m_nz_corr);
        for iy in iy_cen - 1..=iy_cen + 1 {
            for ix in ix_cen - 1..=ix_cen + 1 {
                if y_elongated != 0 {
                    ccc = self.template_cc_coefficient(
                        &average[(nxc * nyc / 2) as usize..],
                        nxc * nyc,
                        nxc,
                        nzc,
                        &brray[(nxb * iz_best) as usize..],
                        nxb * nyb,
                        nxb,
                        nzb,
                        ix,
                        iy,
                    );
                } else {
                    ccc = self.template_cc_coefficient(
                        &average[(nxc * nyc * nzc / 2) as usize..],
                        nxc,
                        nxc,
                        nyc,
                        &brray[(nxb * nyb * iz_best) as usize..],
                        nxb,
                        nxb,
                        nyb,
                        ix,
                        iy,
                    );
                }
                if (*ccc2d as f64) < ccc {
                    *ccc2d = ccc as f32;
                }
            }
        }
    }

    /// `FB3d::templateCCCoefficient` (`findbeads3d.cpp:1676`).
    #[allow(clippy::too_many_arguments)]
    fn template_cc_coefficient(
        &self,
        templ: &[f32],
        templ_xdim: i32,
        nx_templ: i32,
        ny_templ: i32,
        brray: &[f32],
        nx_dim: i32,
        nxb: i32,
        nyb: i32,
        ixcen: i32,
        iycen: i32,
    ) -> f64 {
        let (mut asum, mut bsum, mut csum, mut asumsq, mut bsumsq) = (0f64, 0f64, 0f64, 0f64, 0f64);
        let mut ccc: f64;
        let (mut aval, mut bval): (f64, f64);
        let mut nsum: i32 = 0;
        let ix_off = ixcen - nx_templ / 2;
        let iy_off = iycen - ny_templ / 2;
        for iy in 0..ny_templ {
            let by = iy + iy_off;
            if by >= 0 && by < nyb {
                for ix in 0..nx_templ {
                    let bx = ix + ix_off;
                    if bx >= 0 && bx < nxb {
                        aval = templ[(ix + iy * templ_xdim) as usize] as f64;
                        bval = brray[(bx + by * nx_dim) as usize] as f64;
                        asum += aval;
                        asumsq += aval * aval;
                        bsum += bval;
                        bsumsq += bval * bval;
                        csum += aval * bval;
                        nsum += 1;
                    }
                }
            }
        }
        let n = nsum as f64;
        ccc = (n * asumsq - asum * asum) * (n * bsumsq - bsum * bsum);
        if ccc <= 0. {
            return 0.;
        }
        ccc = (n * csum - asum * bsum) / ccc.sqrt();
        ccc
    }

    /// `FB3d::edgeForCG` (`findbeads3d.cpp:1712`): the mean, median or SD of
    /// the edge pixels for taking a centroid.
    #[allow(clippy::too_many_arguments)]
    fn edge_for_cg(
        &mut self,
        box_tmp_is_smooth: bool,
        nx_box: i32,
        ny_box: i32,
        ixcen: i32,
        iycen: i32,
        for_med_or_sd: i32,
        edge: &mut f32,
        edge_sd: &mut f32,
        ierr: &mut i32,
    ) {
        let (mut iy, mut ix): (i32, i32);
        let mut nsum: i32 = 0;
        let mut sum: f32 = 0.;
        *edge = 0.;
        *ierr = 1;
        let box_tmp: &[f32] = if box_tmp_is_smooth {
            &self.m_elong_smooth
        } else {
            &self.m_elong_temp
        };
        //
        // find edge mean - require half the points to be present
        //
        if for_med_or_sd != 0 {
            for i in 0..self.m_num_edge as usize {
                ix = ixcen + self.m_idx_edge[i] - 1;
                iy = iycen + self.m_idy_edge[i] - 1;
                if ix >= 0 && ix < nx_box && iy >= 0 && iy < ny_box {
                    self.m_edge_pixels[nsum as usize] = box_tmp[(iy * nx_box + ix) as usize];
                    nsum += 1;
                }
            }
        } else {
            for i in 0..self.m_num_edge as usize {
                ix = ixcen + self.m_idx_edge[i] - 1;
                iy = iycen + self.m_idy_edge[i] - 1;
                if ix >= 0 && ix < nx_box && iy >= 0 && iy < ny_box {
                    sum += box_tmp[(iy * nx_box + ix) as usize];
                    nsum += 1;
                }
            }
        }
        if nsum < self.m_num_edge / 2 {
            return;
        }
        *ierr = 0;
        if for_med_or_sd < 0 {
            avg_sd(&self.m_edge_pixels, nsum, edge, edge_sd, &mut sum);
        } else if for_med_or_sd > 0 {
            rs_fast_median_in_place(&mut self.m_edge_pixels, nsum, edge);
        } else {
            *edge = sum / nsum as f32;
        }
    }

    /// `FB3d::bestCenterForCG` (`findbeads3d.cpp:1759`): the strongest set of
    /// 4 pixels within 1 pixel of the nominal center.
    #[allow(clippy::too_many_arguments)]
    fn best_center_for_cg(
        &self,
        box_tmp: &[f32],
        nx_box: i32,
        ny_box: i32,
        xpeak: f32,
        ypeak: f32,
        ixcen: &mut i32,
        iycen: &mut i32,
        best: &mut f32,
    ) {
        let (mut ix_best, mut iy_best) = (0i32, 0i32);
        let mut sum4: f32;
        //
        *ixcen = nx_box / 2 + b3dnint!(xpeak);
        *iycen = ny_box / 2 + b3dnint!(ypeak);
        //
        // look around, find most extreme 4 points as center
        //
        if *ixcen >= 2 && *ixcen <= nx_box - 2 && *iycen >= 2 && *iycen <= ny_box - 2 {
            *best = 0.;
            for iy in *iycen - 1..=*iycen + 1 {
                for ix in *ixcen - 1..=*ixcen + 1 {
                    sum4 = box_tmp[((iy - 1) * nx_box + ix - 1) as usize]
                        + box_tmp[((iy - 1) * nx_box + ix) as usize]
                        + box_tmp[(iy * nx_box + ix - 1) as usize]
                        + box_tmp[(iy * nx_box + ix) as usize];
                    if self.m_polarity * sum4 > self.m_polarity * *best || *best == 0. {
                        ix_best = ix;
                        iy_best = iy;
                        *best = sum4;
                    }
                }
            }
            *ixcen = ix_best;
            *iycen = iy_best;
        }
    }

    /// `FB3d::calcElongation` (`findbeads3d.cpp:1795`): a measure of the
    /// elongation of the density at least a fraction above the edge
    /// intensity, from `mElongTemp`.
    fn calc_elongation(&mut self, xpeak: f32, ypeak: f32, elongation: &mut f32) {
        let (mut ixcen, mut iycen) = (0i32, 0i32);
        let mut num_pos: i32;
        let (mut ix, mut iy): (i32, i32);
        let mut ind_check: i32;
        let idelx: [i32; 4] = [-1, 1, 0, 0];
        let idely: [i32; 4] = [0, 0, -1, 1];
        let (xmean, ymean): (f32, f32);
        let (mut edge, mut edge_sd) = (0f32, 0f32);
        let thresh: f32;
        let thresh_frac: f32;
        let root: f32;
        // Uninitialised in the source when the center is too near the edge.
        let mut best_sum: f32 = 0.;
        let (mut dxsum, mut dysum, mut dxsqsum, mut dysqsum, mut dxysum): (
            f64,
            f64,
            f64,
            f64,
            f64,
        );
        let mut i: i32 = 0;
        let box_size = self.m_elong_box_size;
        //
        thresh_frac = 0.20;
        //
        // Smooth the data then find an adjusted center from the smoothed data
        apply_kernel_filter(
            &self.m_elong_temp,
            &mut self.m_elong_smooth,
            box_size,
            box_size,
            box_size,
            &self.m_elong_kernel,
            self.m_kern_dim_elong,
        );
        self.best_center_for_cg(
            &self.m_elong_smooth,
            box_size,
            box_size,
            xpeak,
            ypeak,
            &mut ixcen,
            &mut iycen,
            &mut best_sum,
        );
        *elongation = -1.;
        //
        // Get an edge median regardless of normal setting
        self.edge_for_cg(
            true,
            box_size,
            box_size,
            ixcen,
            iycen,
            1,
            &mut edge,
            &mut edge_sd,
            &mut i,
        );
        if i != 0 || ixcen <= 0 || ixcen > box_size || iycen <= 0 || iycen > box_size {
            return;
        }
        //
        // Get threshold value and start a list of points to check with the center point
        thresh = (edge as f64 + thresh_frac as f64 * (best_sum as f64 / 4. - edge as f64)) as f32;
        num_pos = 1;
        self.m_ix_elong[0] = ixcen;
        self.m_iy_elong[0] = iycen;
        ind_check = 0;
        for v in self.m_elong_mask[..(box_size * box_size) as usize].iter_mut() {
            *v = 0;
        }
        self.m_elong_mask[((iycen - 1) * box_size + ixcen - 1) as usize] = 1;
        dxsum = 0.;
        dysum = 0.;
        //
        // Make a list of pixels above the threshold by checking the four neighbors of each
        // point on list, adding to list and setting a mask as each is found
        while ind_check < num_pos {
            for k in 0..4 {
                ix = b3dmax!(
                    1,
                    b3dmin!(box_size, self.m_ix_elong[ind_check as usize] + idelx[k])
                ) - 1;
                iy = b3dmax!(
                    1,
                    b3dmin!(box_size, self.m_iy_elong[ind_check as usize] + idely[k])
                ) - 1;
                let m = (iy * box_size + ix) as usize;
                if self.m_elong_mask[m] == 0
                    && self.m_polarity * (self.m_elong_smooth[m] - thresh) > 0.
                {
                    self.m_elong_mask[m] = 1;
                    self.m_ix_elong[num_pos as usize] = ix + 1;
                    self.m_iy_elong[num_pos as usize] = iy + 1;
                    num_pos += 1;
                    dxsum += (ix + 1) as f64;
                    dysum += (iy + 1) as f64;
                }
            }
            ind_check += 1;
        }
        if num_pos < 4 {
            return;
        }

        // Get the means and moments and apply the equation for elongation
        xmean = (dxsum / num_pos as f64) as f32;
        ymean = (dysum / num_pos as f64) as f32;
        dxsqsum = 0.;
        dysqsum = 0.;
        dxysum = 0.;
        for k in 0..num_pos as usize {
            // `pow(float, 2.f)` is the C++ float overload, folded by gcc to a
            // float square.
            let dxk = self.m_ix_elong[k] as f32 - xmean;
            let dyk = self.m_iy_elong[k] as f32 - ymean;
            dxsqsum += (dxk * dxk) as f64;
            dysqsum += (dyk * dyk) as f64;
            dxysum += (dxk * dyk) as f64;
        }

        let dd = dxsqsum - dysqsum;
        root = (4. * dxysum * dxysum + dd * dd).sqrt() as f32;
        *elongation =
            ((dxsqsum + dysqsum + root as f64) / (dxsqsum + dysqsum - root as f64)) as f32;
    }

    /// `FB3d::writePeakModel` (`findbeads3d.cpp:1873`): writes a model with
    /// the given number of peaks and sets the black threshold, if any.
    #[allow(clippy::too_many_arguments)]
    fn write_peak_model(
        &mut self,
        first_file: Option<&str>,
        ind_peak: &[i32],
        peak_val: &[f32],
        peak_pos: &[f32],
        num_peaks: i32,
        black_thresh: f32,
        nxyz: &[i32; 3],
        delta: &[f32; 3],
        origin: &[f32; 3],
        cur_tilt: &[f32; 3],
        radius: f32,
    ) {
        let mut iobj: i32;
        let mut ierr: i32 = 0;
        let mut peak: f32;

        let Some(first_file) = first_file else {
            return;
        };

        self.m_fm.n_point = num_peaks;
        self.m_fm.max_mod_obj = num_peaks;
        newimod();
        // Fixed in translation (BUGS.md): `peakVal[indPeak[numPeaks - 1]]`
        // reads `indPeak[-1]` when `numPeaks` is 0 (nothing passes the
        // storing threshold); `peakMin` is defined as 0 then.
        let peak_min = if num_peaks > 0 {
            peak_val[ind_peak[(num_peaks - 1) as usize] as usize]
        } else {
            0.
        };
        let peak_max = peak_val[ind_peak[0] as usize];
        putimodflag(1, 2);
        putimodflag(1, 7);
        let j = b3dmax!(1., 1. + (radius).ceil()) as i32;
        putscatsize(1, j);
        // `ierr =+ putImodMaxes(...)` (`findbeads3d.cpp:1903`) assigns the
        // unary plus; the accumulation meant is written (fixed in
        // translation, BUGS.md -- identical here, `ierr` is still 0).
        ierr += putimodmaxes(nxyz[0], nxyz[1], nxyz[2]);
        for i in 0..self.m_fm.n_point.max(0) as usize {
            peak = peak_val[ind_peak[i] as usize];
            //
            // set up object if it is time for next one
            //
            self.m_fm.obj_color[i][1] = 255;
            self.m_fm.npt_in_obj[i] = 1;
            self.m_fm.ibase_obj[i] = i as i32;

            for jj in 0..3 {
                self.m_fm.p_coord[i][jj] = peak_pos[ind_peak[i] as usize * 3 + jj];
            }
            self.m_fm.p_coord[i][2] = (self.m_fm.p_coord[i][2] as f64 - 0.5) as f32;
            ierr += putcontvalue(1, i as i32 + 1, peak);
            self.m_fm.object[i] = i as i32 + 1;
        }
        iobj = 0;
        // Fixed in translation (BUGS.md): the source writes
        // `255. * (blackThresh - peakMin) / peakMax - peakMin`
        // (`findbeads3d.cpp:1911`), subtracting `peakMin` after the division;
        // the black level is meant to scale `blackThresh` into the peak range,
        // `/ (peakMax - peakMin)`.  With an empty range (one peak) that
        // division is undefined and the level is left at 0.
        if black_thresh > 0. && peak_max > peak_min {
            iobj = b3dmax!(
                0,
                b3dmin!(
                    255,
                    b3dnint!(
                        255. * (black_thresh - peak_min) as f64 / (peak_max - peak_min) as f64
                    )
                )
            );
        }
        putvalblackwhite(1, iobj, 255);
        ierr += putimageref(delta, origin, cur_tilt);
        if ierr != 0 {
            printf!("WARNING: Findbeads3d - Errors occurred creating the model file");
        }
        putimodobjname(1, "3D bead positions");
        let _ = scale_fort_mod_to_image(&mut self.m_fm, 1, 1);
        let _ = write_fort_model(first_file, &mut self.m_fm);
    }

    /// `FB3d::volWrite` (`findbeads3d.cpp:1928`): writes a volume held in a
    /// contiguous array.
    fn vol_write(&self, iunit: i32, file_out: &str, array: &[f32], nx: i32, ny: i32, nz: i32) {
        let (mut tmin, mut tmax, mut tmean): (f32, f32, f32);
        let mut dmin: f32 = 1.0e30;
        let mut dmax: f32 = -1.0e30;
        let mut dmean: f32;
        let nxyz = [nx, ny, nz];
        let mut cell: [f32; 6] = [0., 0., 1., 90., 90., 90.];
        //
        unsafe { iiu_open(iunit, file_out, "NEW") };
        cell[0] = nx as f32;
        cell[1] = ny as f32;
        cell[2] = nz as f32;
        iiu_create_header(
            iunit,
            &nxyz,
            &nxyz,
            2,
            &[[0u8; MRC_LABEL_SIZE]; MRC_NLABELS],
            0,
        );
        iiu_alt_cell(iunit, &cell);
        dmean = 0.;
        tmin = 1.0e30;
        tmax = -1.0e30;
        tmean = 0.;
        let nxy = (nx * ny) as usize;
        for iz in 0..nz as usize {
            let mut sec = array[iz * nxy..(iz + 1) * nxy].to_vec();
            if let Some((a, b, c)) = with_mrc_data!(sec, |d| full_array_min_max_mean(
                &mut d,
                MRC_MODE_FLOAT,
                nx,
                ny
            )) {
                (tmin, tmax, tmean) = (a, b, c);
            }
            unsafe { iiu_write_section(iunit, sec.as_mut_ptr().cast()) };
            dmin = b3dmin!(tmin, dmin);
            dmax = b3dmax!(tmax, dmax);
            dmean += tmean;
        }
        iiu_write_header_str(iunit, "DEBUG VOLUME", 1, dmin, dmax, dmean / nz as f32);
        unsafe { iiu_close(iunit) };
    }
}
