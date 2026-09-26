//! Translation of `IMOD/flib/image/xfsimplex.f90`.
//!
//! This program searches for the best general linear transform between a
//! pair of images by varying either the six formal parameters of the
//! transform, the six "semi-natural" parameters underlying such a transform,
//! or restricted subsets of those semi-natural parameters.
//!
//! The module `simplexvars` is [`SimplexVars`], one value per run (so an
//! in-process run starts from the declared initial state), and `diff`'s
//! saved `deltaLast` lives in it.  The main program maps to [`xfsimplex`];
//! the subroutines `diff`, `dist`, `func`, `readFilterSection` and
//! `checkForWarpFile` and the function `outsideMultiplier` map to [`diff`],
//! [`dist`], [`func`], [`read_filter_section`], [`check_for_warp_file`] and
//! [`outside_multiplier`].  The C routine `simplexDiff` it calls is
//! `simplexdiff.rs`.
//!
//! `brray` is equivalenced in the source with `denLow`, `denHigh`, `ixComp`
//! and `iyComp` at disjoint offsets; the difference measure uses only
//! `brray` and the distance measure only the other four, so they are
//! separate arrays here.  Fortran arrays are 1-based; subscripts keep the
//! source's values and are lowered by one at each access.

use crate::imod::flib::image::simplexdiff::simplex_diff;
use crate::imod::flib::subrs::compat::gfortran_rt::{cvttss2si, format_f};
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{ialprt, imopen, irdsec};
use crate::imod::libcfshr::amat_to_rotmagstr::{amat_to_rotmag, rotmag_to_amat};
use crate::imod::libcfshr::amoeba::{amoebafwrap, amoebainitfwrap};
use crate::imod::libcfshr::b3dutil::{exit, fortran_string};
use crate::imod::libcfshr::filtxcorr::{FilterIn, nice_frame, xcorr_filter_part, xcorr_set_ctf};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_boolean, pip_get_float, pip_get_float_array, pip_get_integer,
    pip_get_two_floats, pip_get_two_integers,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libcfshr::reduce_by_binning::{SLICE_MODE_FLOAT, irepak, reduce_by_binning};
use crate::imod::libcfshr::scaledsobel::scaled_sobel;
use crate::imod::libcfshr::simplestat::array_min_max_mean_sd_fortran;
use crate::imod::libcfshr::taperpad::{PadIn, taperoutpad};
use crate::imod::libfft::odfft::nice_fft_limit;
use crate::imod::libfft::todfft::todfft;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position};
use crate::imod::libiimod::unit_reduced::{iiu_read_binned, iiu_read_reduced};
use crate::imod::libwarp::warputils::read_check_warp_file;
use std::io::{BufRead, BufReader, Write};

/// `parameter (IDIMB = 4100 * 4100)` (`xfsimplex.f90:19`).
const IDIMB: i32 = 4100 * 4100;
/// `parameter (ISUB = IDIMB / 3, LIMSPIR = 1000)` (`xfsimplex.f90:20`).
const ISUB: i32 = IDIMB / 3;
const LIMSPIR: usize = 1000;
/// `parameter (numOptions = 34)` (`xfsimplex.f90:102`).
const NUM_OPTIONS: i32 = 34;
/// Fallback PIP table, the `options(1)` string (`xfsimplex.f90:104-117`).
const OPTIONS: &str = "aimage:AImageFile:FN:@bimage:BImageFile:FN:@output:OutputFile:FN:@\
initial:InitialTransformFile:FN:@useline:UseTransformLine:I:@\
sections:SectionsToUse:IP:@variables:VariablesToSearch:I:@\
limits:LimitsOnSearch:FA:@edge:EdgeToIgnore:F:@xminmax:XMinAndMax:IP:@\
yminmax:YMinAndMax:IP:@binning:BinningToApply:I:@antialias:AntialiasFilter:I:@\
sig1:FilterSigma1:F:@rad1:FilterRadius1:F:@rad2:FilterRadius2:F:@\
sig2:FilterSigma2:F:@after:FilterAfterBinning:B:@sobel:SobelFilter:B:@\
float:FloatOption:I:@ccc:CorrelationCoefficient:B:@local:LocalPatchSize:I:@\
linear:LinearInterpolation:B:@distance:DistanceMeasure:B:@\
near:NearestDistance:I:@radius:RadiusToSearch:F:@density:DensityDifference:F:@\
percent:PercentileRanges:FA:@coarse:CoarseTolerances:FP:@\
final:FinalTolerances:FP:@step:StepSizeFactor:F:@trace:TraceOutput:I:@\
param:ParameterFile:PF:@help:usage:B:";

/// Original module `simplexvars` (`xfsimplex.f90:16`).
pub struct SimplexVars {
    /// `nx, ny, nz` (equivalenced with `nxyz`).
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    /// `real*4, allocatable :: array(:)`
    pub array: Vec<f32>,
    /// `real*4 brray(IDIMB)`
    pub brray: Vec<f32>,
    /// `integer*2 ixComp(ISUB), iyComp(ISUB)`
    pub ix_comp: Vec<i16>,
    pub iy_comp: Vec<i16>,
    /// `real*4 denLow(ISUB), denHigh(ISUB)`
    pub den_low: Vec<f32>,
    pub den_high: Vec<f32>,
    pub nx1: i32,
    pub nx2: i32,
    pub ny1: i32,
    pub ny2: i32,
    pub if_interp: i32,
    pub natural: i32,
    pub num_compare: i32,
    pub num_spiral: i32,
    pub if_dist: i32,
    pub if_trace: i32,
    pub reduction: f32,
    pub delta_min: f32,
    pub sd1: f32,
    pub dist_spiral: [f32; LIMSPIR],
    pub acall: [f32; 6],
    /// `aLimits(2,6)`: `aLimits(k, i)` is `a_limits[i - 1][k - 1]`.
    pub a_limits: [[f32; 2]; 6],
    pub idima: i32,
    pub num_trials: i32,
    pub iv_end: i32,
    pub if_ccc: i32,
    pub idx_spiral: [i32; LIMSPIR],
    pub idy_spiral: [i32; LIMSPIR],
    /// `sxa(numXpatch, numYpatch)` and the other patch arrays, column order.
    pub sxa: Vec<f64>,
    pub sya: Vec<f64>,
    pub sx_sqa: Vec<f64>,
    pub sy_sqa: Vec<f64>,
    pub sxya: Vec<f64>,
    pub num_pix_a: Vec<i32>,
    pub num_xpatch: i32,
    pub num_ypatch: i32,
    pub nxy_patch: i32,
    /// `diff`'s `real*4 deltaLast/0./` with `save deltaLast`.
    pub delta_last: f32,
}

/// `read(5, *)` with no `END=`/`ERR=`: end of input is the gfortran runtime
/// error, status 2.
fn read_list(items: &mut [ListItem]) {
    let _ = std::io::stdout().flush();
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    if list_read(&mut lock, items).is_err() {
        eprintln!("Fortran runtime error: End of file");
        exit(2);
    }
}

/// Fortran `Iw` output of an `integer*4`.
fn fortran_i(value: i32, w: usize) -> String {
    let text = value.to_string();
    if text.len() > w {
        "*".repeat(w)
    } else {
        format!("{text:>w$}")
    }
}

/// Fortran `Fw.d` output of a `real*4`.
fn ff(value: f32, w: usize, d: usize) -> String {
    format_f(value as f64, w, d)
}

/// Fortran `NINT` of a `real*4` (`lroundf`, low 32 bits).
fn nint(x: f32) -> i32 {
    x.round() as i64 as i32
}

/// Rust-only: the values of `xfsimplex`'s closing report
/// (`xfsimplex.f90:410-419`): the number of trials, the final difference
/// measure, the four natural or formal parameters and the two shifts of
/// FORMAT 72, and whether natural parameters were searched (the program then
/// also prints the equivalent transform, FORMAT 70).
#[derive(Clone, Copy, Debug, Default)]
pub struct XfsimplexResult {
    pub num_trials: i32,
    pub delta_min: f32,
    pub params: [f32; 4],
    pub shifts: [f32; 2],
    pub natural: bool,
}

thread_local! {
    /// Where [`xfsimplex`] records its [`XfsimplexResult`] on this thread,
    /// when a direct caller set it through [`xfsimplex_recording`].
    static RESULT_SINK: std::cell::RefCell<Option<std::sync::Arc<std::sync::Mutex<Option<XfsimplexResult>>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Rust-only: runs program [`xfsimplex`] (options from the in-process
/// runner's `argv` and standard input) with its closing values recorded into
/// `sink`, for a direct caller that used to parse its report (`tiltmatch`;
/// `CLAUDE.md`, "Wherever we control both sides, use a direct function call
/// now").  The program's output is unchanged.  It ends through `exit`; run
/// it under `commands::call_in_process`.
pub fn xfsimplex_recording(sink: std::sync::Arc<std::sync::Mutex<Option<XfsimplexResult>>>) {
    RESULT_SINK.with_borrow_mut(|slot| *slot = Some(sink));
    xfsimplex();
}

/// Rust-only: FORMAT 72's line of the closing report, `(i4,f14.7,1x,4f10.5,
/// 2f10.3)`, for `result`, without its ending.
pub fn xfsimplex_final_line(result: &XfsimplexResult) -> String {
    format!(
        "{}{} {}{}{}{}{}{}",
        fortran_i(result.num_trials, 4),
        ff(result.delta_min, 14, 7),
        ff(result.params[0], 10, 5),
        ff(result.params[1], 10, 5),
        ff(result.params[2], 10, 5),
        ff(result.params[3], 10, 5),
        ff(result.shifts[0], 10, 3),
        ff(result.shifts[1], 10, 3)
    )
}

/// Original program `xfsimplex` (`xfsimplex.f90:43`).
pub fn xfsimplex() {
    unsafe {
        let mut sv = SimplexVars {
            nx: 0,
            ny: 0,
            nz: 0,
            array: Vec::new(),
            brray: Vec::new(),
            ix_comp: Vec::new(),
            iy_comp: Vec::new(),
            den_low: Vec::new(),
            den_high: Vec::new(),
            nx1: 0,
            nx2: 0,
            ny1: 0,
            ny2: 0,
            if_interp: 0,
            natural: 0,
            num_compare: 0,
            num_spiral: 0,
            if_dist: 0,
            if_trace: 0,
            reduction: 0.,
            delta_min: 0.,
            sd1: 0.,
            dist_spiral: [0.; LIMSPIR],
            acall: [0.; 6],
            a_limits: [[0.; 2]; 6],
            idima: 0,
            num_trials: 0,
            iv_end: 0,
            if_ccc: 0,
            idx_spiral: [0; LIMSPIR],
            idy_spiral: [0; LIMSPIR],
            sxa: Vec::new(),
            sya: Vec::new(),
            sx_sqa: Vec::new(),
            sy_sqa: Vec::new(),
            sxya: Vec::new(),
            num_pix_a: Vec::new(),
            num_xpatch: 0,
            num_ypatch: 0,
            nxy_patch: 0,
            delta_last: 0.,
        };
        let mut mxyz = [0_i32; 3];
        let mut nxyz = [0_i32; 3];
        let mut nxyz2 = [0_i32; 3];
        //
        let mut input_file1 = String::new();
        let mut input_file2 = String::new();
        let mut out_file = String::new();
        let mut xf_in_file: String;
        //
        let mut a: [f32; 6] = [0., 0., 1., 0., 0., 1.];
        let mut da: [f32; 6] = [1., 1., 0.025, 0.025, 0.025, 0.025];
        let mut amat = [0f32; 4];
        let anat: [f32; 6] = [0., 0., 0., 1., 0., 0.];
        let danat: [f32; 6] = [1., 1., 2., 0.02, 0.02, 2.];
        let mut pp = [0f32; 49];
        let mut yy = [0f32; 7];
        let mut ptol = [0f32; 6];
        let mut ctf = [0f32; 8193];
        // if doing formal params, the a(i) are DX, DY, a11, a12, a21, and a22
        // for natural paramas, the a(i) are DX and DY, Global rotation, global
        // stretch, difference between Y&X-axis stretch, and difference
        // between Y and X axis rotation.
        // the da are the step sizes for the respective a's
        let trace: bool;
        let (mut tsum, mut tsum_sq) = (0f64, 0f64);
        //
        let mut ihist = [0_i32; 1002];
        // range(10,2) / percentile(10,2), with ranLow/rangeHigh and
        // pctLow/pctHigh equivalenced to their two columns.
        let mut ran_low = [0f32; 10];
        let mut range_high = [0f32; 10];
        let mut pct_low = [0f32; 10];
        let mut pct_high = [0f32; 10];
        let mut param_lim = [0f32; 6];
        //
        // default values for potentially input parameters
        let (mut ftol1, mut ptol1, mut ftol2, mut ptol2): (f32, f32, f32, f32) =
            (5.0e-4, 0.02, 5.0e-3, 0.2);
        let mut delta_fac: f32 = 2.;
        let mut if_float_mean: i32 = 1;
        let mut ibinning: i32 = 2;
        let mut num_ranges: i32 = 2;
        let mut idist_redund: i32 = 0;
        let mut radius: f32 = 4.;
        let mut difflim: f32 = 0.05;
        //
        let mut mode = 0_i32;
        let mut ierr: i32;
        let mut num_limit: i32;
        let mut len_temp: i32;
        let mut frac_matt: f32;
        let (mut dmin, mut dmax, mut dmean) = (0f32, 0f32, 0f32);
        let (mut dmin2, mut dmax2, mut dmean2, mut sd2) = (0f32, 0f32, 0f32, 0f32);
        let (mut dmin1, mut dmax1, mut dmean1) = (0f32, 0f32, 0f32);
        let mut delmin = 0f32;
        let mut iter = 0_i32;
        let mut jmin = 0usize;
        let mut iz_ref: i32;
        let mut iz_ali: i32;
        let mut npad: i32;
        let mut if_sobel: i32;
        let mut if_filt_after: i32;
        let mut line_use: i32;
        let mut if_xmin_max: i32;
        let mut if_ymin_max: i32;
        let mut i_anti_filt_type: i32;
        let mut deltac = 0f32;
        let (mut sigma1, mut sigma2, mut radius1, mut radius2): (f32, f32, f32, f32);
        //
        // default values for parameters in module and equivalenced arrays
        //
        pct_low[0] = 0.;
        pct_low[1] = 92.;
        pct_high[0] = 8.;
        pct_high[1] = 100.;
        sv.if_trace = 0;
        frac_matt = 0.05;
        sv.if_dist = 0;
        sv.natural = 0;
        sv.if_interp = 0;
        sv.nxy_patch = 0;
        iz_ref = 0;
        iz_ali = 0;
        for i in 1..=6 {
            sv.a_limits[i - 1][0] = 0.;
            sv.a_limits[i - 1][1] = 0.;
        }
        if_filt_after = 0;
        sigma1 = 0.;
        sigma2 = 0.;
        radius1 = 0.;
        radius2 = 0.;
        npad = 16;
        if_sobel = 0;
        sv.if_ccc = 0;
        line_use = 0;
        if_xmin_max = 0;
        if_ymin_max = 0;
        i_anti_filt_type = 0;
        len_temp = 32000 * 64;
        //
        // Pip startup: set error, parse options, check help, set flag if used
        //
        let (mut num_opt_arg, mut num_non_opt_arg) = (0, 0);
        pip_read_or_parse_options(
            &[OPTIONS],
            NUM_OPTIONS,
            "xfsimplex",
            "ERROR: XFSIMPLEX - ",
            true,
            3,
            2,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
        );
        let pip_input = num_opt_arg + num_non_opt_arg > 0;
        //
        if pip_get_in_out_file(
            "AImageFile",
            1,
            "Enter first image file name",
            &mut input_file1,
            320,
        ) != 0
        {
            exit_error("No first image file specified");
        }
        if pip_get_in_out_file(
            "BImageFile",
            2,
            "Enter second image file name",
            &mut input_file2,
            320,
        ) != 0
        {
            exit_error("No second image file specified");
        }
        if pip_get_in_out_file(
            "OutputFile",
            3,
            "Transform output file name",
            &mut out_file,
            320,
        ) != 0
        {
            exit_error("No transform output file specified");
        }
        if pip_input {
            let mut record = [b' '; 320];
            let _ = pipgetstring_(b"InitialTransformFile", &mut record);
            xf_in_file = fortran_string(&record);
        } else {
            print!(" File with starting transform, or Return if none: ");
            let _ = std::io::stdout().flush();
            // read (5, 40) xfInFile   40 format (a)
            let mut line: Vec<u8> = Vec::new();
            match std::io::stdin().lock().read_until(b'\n', &mut line) {
                Ok(0) | Err(_) => {
                    eprintln!("Fortran runtime error: End of file");
                    exit(2);
                }
                Ok(_) => {}
            }
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            line.truncate(320);
            xf_in_file = fortran_string(&line);
        }
        //
        // open file now to get sizes and try to adjust defaults
        //
        ialprt(false);
        imopen(1, &input_file1, "RO");
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
        [sv.nx, sv.ny, sv.nz] = nxyz;
        //
        if sv.nx <= 128 && sv.ny <= 128 {
            ftol1 *= 2.;
            ptol1 *= 2.;
        }
        //
        if pip_input {
            let _ = pip_get_two_floats(b"CoarseTolerances", &mut ftol2, &mut ptol2);
            let _ = pip_get_two_floats(b"FinalTolerances", &mut ftol1, &mut ptol1);
            let _ = pip_get_float(b"StepSizeFactor", &mut delta_fac);
            let _ = pip_get_integer(b"TraceOutput", &mut sv.if_trace);
            let _ = pip_get_integer(b"UseTransformLine", &mut line_use);
            if line_use < 0 {
                exit_error("Initial transform number out of range");
            }
        } else {
            print!(
                " Enter fractional tolerances in difference measure and in parameter values\n   for terminating final search, corresponding tolerances for initial search\n   (or 0,0 for only one search), factor for initial step size, and 1 or 2 for\n    trace output [{}{}{}{}{}{}]: ",
                ff(ftol1, 6, 4),
                ff(ptol1, 5, 2),
                ff(ftol2, 7, 4),
                ff(ptol2, 5, 2),
                ff(delta_fac, 4, 1),
                fortran_i(sv.if_trace, 2)
            );
            read_list(&mut [
                ListItem::Real(&mut ftol1),
                ListItem::Real(&mut ptol1),
                ListItem::Real(&mut ftol2),
                ListItem::Real(&mut ptol2),
                ListItem::Real(&mut delta_fac),
                ListItem::Integer(&mut sv.if_trace),
            ]);
        }
        trace = sv.if_trace > 0;
        //
        if pip_input {
            let _ = pip_get_integer(b"VariablesToSearch", &mut sv.natural);
            if sv.natural < 0 || sv.natural > 6 {
                exit_error("Number of variables to search is out of range");
            }
        } else {
            loop {
                print!(
                    " 0 to search 6 formal variables, or # of natural variables to search [{}]: ",
                    fortran_i(sv.natural, 1)
                );
                read_list(&mut [ListItem::Integer(&mut sv.natural)]);
                if !(sv.natural < 0 || sv.natural > 6) {
                    break;
                }
            }
        }
        if sv.natural == 0 {
            //all six formal params
            sv.iv_end = 6;
        } else {
            sv.iv_end = sv.natural; //or selected # of natural
            for i in 1..=6 {
                a[i - 1] = anat[i - 1];
                da[i - 1] = danat[i - 1];
            }
        }
        //
        // if reading in a transform, get it and convert to natural if necessary
        //
        if !xf_in_file.is_empty() {
            check_for_warp_file(&xf_in_file);
            let unit1 = dopen(1, &xf_in_file, "old", "f");
            let mut reader = BufReader::new(unit1);
            let (mut a11, mut a12, mut a21, mut a22) = (0f32, 0f32, 0f32, 0f32);
            for _i in 0..=line_use {
                let (mut x0, mut x1) = (a[0], a[1]);
                let result = list_read(
                    &mut reader,
                    &mut [
                        ListItem::Real(&mut a11),
                        ListItem::Real(&mut a12),
                        ListItem::Real(&mut a21),
                        ListItem::Real(&mut a22),
                        ListItem::Real(&mut x0),
                        ListItem::Real(&mut x1),
                    ],
                );
                a[0] = x0;
                a[1] = x1;
                match result {
                    Ok(()) => {}
                    Err(ListReadError::End) => {
                        exit_error("Initial transform number out of range");
                    }
                    Err(_) => exit_error("Reading initial transform file"),
                }
            }
            amat = [a11, a21, a12, a22];
            if sv.natural == 0 {
                a[2] = amat[0];
                a[3] = amat[2];
                a[4] = amat[1];
                a[5] = amat[3];
            } else {
                // amat_to_rotmag(amat, a(3), a(6), a(4), a(5))
                let (theta, ydtheta, smag, ydmag) =
                    amat_to_rotmag(amat[0], amat[2], amat[1], amat[3]);
                a[2] = theta;
                a[5] = ydtheta;
                a[3] = smag;
                a[4] = ydmag;
            }
        }
        //
        for i in 1..=6 {
            sv.acall[i - 1] = a[i - 1];
        }
        //
        num_limit = 0;
        if pip_input {
            let _ = pip_get_float(b"EdgeToIgnore", &mut frac_matt);
            let _ = pip_get_integer(b"FloatOption", &mut if_float_mean);
            let _ = pip_get_integer(b"BinningToApply", &mut ibinning);
            let _ = pip_get_boolean(b"DistanceMeasure", &mut sv.if_dist);
            let _ = pip_get_two_integers(b"SectionsToUse", &mut iz_ref, &mut iz_ali);
            let _ = pip_get_float(b"FilterSigma1", &mut sigma1);
            let _ = pip_get_float(b"FilterSigma2", &mut sigma2);
            let _ = pip_get_float(b"FilterRadius1", &mut radius1);
            let _ = pip_get_float(b"FilterRadius2", &mut radius2);
            let _ = pip_get_boolean(b"FilterAfterBinning", &mut if_filt_after);
            let _ = pip_get_boolean(b"SobelFilter", &mut if_sobel);
            let _ = pip_get_boolean(b"CorrelationCoefficient", &mut sv.if_ccc);
            ibinning = 1.max(ibinning);
            if_xmin_max = 1 - pip_get_two_integers(b"XMinAndMax", &mut sv.nx1, &mut sv.nx2);
            if_ymin_max = 1 - pip_get_two_integers(b"YMinAndMax", &mut sv.ny1, &mut sv.ny2);
            let _ = pip_get_integer(b"LocalPatchSize", &mut sv.nxy_patch);
            let _ = pip_get_integer(b"AntialiasFilter", &mut i_anti_filt_type);
            if i_anti_filt_type < 0 {
                i_anti_filt_type = 5;
            }
            if i_anti_filt_type < 2 || ibinning == 1 {
                i_anti_filt_type = 0;
            }
            if i_anti_filt_type > 0 {
                let mut min_chunk_lines = 10;
                if ibinning > 32 {
                    min_chunk_lines = 3;
                }
                len_temp = (sv.nx * ((min_chunk_lines + 6) * ibinning + 20))
                    .max(10000000_i64.min(sv.nx as i64 * sv.ny as i64) as i32);
                if_filt_after = 1;
            }

            //
            // Get search limits
            if pip_get_float_array(b"LimitsOnSearch", &mut param_lim, &mut num_limit, 6) == 0 {
                if sv.natural == 0 && num_limit > 2 {
                    exit_error(
                        "You can limit only X and Y shifts when searching for formal parameters; try -variables 6",
                    );
                }
                if sv.natural > 0 {
                    num_limit = num_limit.min(sv.natural);
                }
                param_lim[0] /= ibinning as f32;
                param_lim[1] /= ibinning as f32;
            }
        } else {
            print!(" edge fraction to ignore [{}]: ", ff(frac_matt, 4, 2));
            read_list(&mut [ListItem::Real(&mut frac_matt)]);
            //
            print!(
                " 0 to float to range, 1 to float to mean&sd, -1 no float [{}]: ",
                fortran_i(if_float_mean, 1)
            );
            read_list(&mut [ListItem::Integer(&mut if_float_mean)]);
            //
            print!(" Binning to apply to image [{}]: ", fortran_i(ibinning, 1));
            read_list(&mut [ListItem::Integer(&mut ibinning)]);
            ibinning = 1.max(ibinning);
            //
            print!(
                " 0 for difference, 1 for distance measure [{}]: ",
                fortran_i(sv.if_dist, 1)
            );
            read_list(&mut [ListItem::Integer(&mut sv.if_dist)]);
        }
        if sv.if_dist == 0 {
            if pip_input {
                let _ = pip_get_boolean(b"LinearInterpolation", &mut sv.if_interp);
            } else {
                print!(" 1 to use interpolation [{}]: ", fortran_i(sv.if_interp, 1));
                read_list(&mut [ListItem::Integer(&mut sv.if_interp)]);
            }
        } else {
            //
            // change defaults based on image size and reduction by 2
            //
            let mut num_pixels = sv.nx * sv.ny;
            if ibinning > 1 {
                num_pixels /= ibinning * ibinning;
                radius = 4.;
            } else {
                radius = 5.;
            }
            //
            if num_pixels > 480 * 360 {
                idist_redund = 2;
            } else if num_pixels > 240 * 180 {
                idist_redund = 1;
            } else {
                idist_redund = 0;
            }
            //
            // run percentile range from 8 down to 5 as go from 320x240 to 640x480
            //
            let pct_range =
                8f32.min(5f32.max(
                    8. - 3. * (num_pixels - 320 * 240) as f32 / (640 * 480 - 320 * 240) as f32,
                ));
            pct_high[0] = pct_range;
            pct_low[1] = 100. - pct_range;
            //
            if pip_input {
                let _ = pip_get_integer(b"NearestDistance", &mut idist_redund);
                let _ = pip_get_float(b"RadiusToSearch", &mut radius);
                let _ = pip_get_float(b"DensityDifference", &mut difflim);
                let mut ibase = 0;
                // PipGetFloatArray('PercentileRanges', brray, ibase, 20):
                // `brray(1..20)` is `denLow`, which is filled later.
                let mut values = [0f32; 20];
                if pip_get_float_array(b"PercentileRanges", &mut values, &mut ibase, 20) == 0 {
                    if ibase % 2 != 0 {
                        exit_error("You must enter an even number of values for PercentileRanges");
                    }
                    num_ranges = ibase / 2;
                    for i in 1..=num_ranges as usize {
                        pct_low[i - 1] = values[2 * i - 2];
                        pct_high[i - 1] = values[2 * i - 1];
                    }
                }
            } else {
                print!(
                    " distance to search for and eliminate redundancy, 0 not to [{}]: ",
                    fortran_i(idist_redund, 1)
                );
                read_list(&mut [ListItem::Integer(&mut idist_redund)]);
                //
                // get density window and search radius
                print!(" radius to search for match [{}]: ", ff(radius, 3, 1));
                read_list(&mut [ListItem::Real(&mut radius)]);
                //
                print!(
                    " max density difference for match as fraction of range [{}]: ",
                    ff(difflim, 4, 2)
                );
                read_list(&mut [ListItem::Real(&mut difflim)]);
                //
                print!(
                    " number of percentile ranges [{}]: ",
                    fortran_i(num_ranges, 1)
                );
                read_list(&mut [ListItem::Integer(&mut num_ranges)]);
                //
                // get percentile ranges
                // write(*,'(1x,a,6f6.1)') ..., (pctLow(i), pctHigh(i), i = 1, numRanges)
                // (six values per record; the format has no room for more)
                let mut text = String::from(" lower and upper percentiles in ranges - defaults:");
                for i in 1..=num_ranges.min(3) as usize {
                    text.push_str(&ff(pct_low[i - 1], 6, 1));
                    text.push_str(&ff(pct_high[i - 1], 6, 1));
                }
                println!("{text}");
                let (lows, highs) = (&mut pct_low, &mut pct_high);
                let mut items: Vec<ListItem> = Vec::new();
                for (low, high) in lows
                    .iter_mut()
                    .zip(highs.iter_mut())
                    .take(num_ranges as usize)
                {
                    items.push(ListItem::Real(low));
                    items.push(ListItem::Real(high));
                }
                read_list(&mut items);
            }
        }
        pip_done();
        //
        ialprt(true);
        iiu_close(1);
        imopen(1, &input_file1, "RO");
        // NOTE: ABSOLUTELY NEED TO READ HEADER AGAIN
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
        [sv.nx, sv.ny, sv.nz] = nxyz;
        //
        imopen(2, &input_file2, "RO");
        irdhdr(
            2,
            nxyz2.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
        // BUGS.md, fixed in translation: the source compares `nxyz(2)` with
        // `ny`, which is itself, so native never checks the Y size of B;
        // `nxyz2(2)` is meant.
        if nxyz2[0] != sv.nx || nxyz2[1] != sv.ny {
            exit_error("The two images must be the same size in X and Y");
        }
        if iz_ref < 0 || iz_ref >= sv.nz || iz_ali < 0 || iz_ali >= nxyz2[2] {
            exit_error("One of the section numbers is out of range");
        }
        //
        // Open data file for transform
        //
        let mut unit4 = dopen(4, &out_file, "NEW", "F");
        //
        // Get possibly filtered sizes, and check dimensions
        let nx_orig = sv.nx;
        let ny_orig = sv.ny;
        sv.nx /= ibinning;
        sv.ny /= ibinning;
        let nx_pad;
        let ny_pad;
        if if_filt_after == 0 {
            nx_pad = nice_frame(nx_orig + npad, 2, nice_fft_limit());
            ny_pad = nice_frame(ny_orig + npad, 2, nice_fft_limit());
        } else {
            nx_pad = nice_frame(sv.nx + npad, 2, nice_fft_limit());
            ny_pad = nice_frame(sv.ny + npad, 2, nice_fft_limit());
        }
        let _ = &mut npad;
        xcorr_set_ctf(
            sigma1,
            sigma2,
            radius1,
            radius2,
            &mut ctf,
            nx_pad,
            ny_pad,
            &mut deltac,
        );
        //
        // Allocate big array and temp array
        len_temp = len_temp.min(nx_orig * ny_orig);
        sv.idima = (nx_pad + 2) * ny_pad + 10;
        let mut temp: Vec<f32> = Vec::new();
        ierr = 0;
        if sv.idima < 0
            || len_temp < 0
            || sv.array.try_reserve_exact(sv.idima as usize).is_err()
            || temp.try_reserve_exact(len_temp as usize).is_err()
        {
            ierr = 1;
        }
        memory_error(ierr, "large image arrays");
        sv.array.resize(sv.idima as usize, 0.);
        temp.resize(len_temp as usize, 0.);

        if sv.nx * sv.ny > IDIMB {
            exit_error("Image too big for arrays - use higher binning");
        }
        //
        // Reduce the shift parameters, and set the limits now that shift is set
        sv.reduction = ibinning as f32;
        da[0] /= sv.reduction;
        da[1] /= sv.reduction;
        a[0] /= sv.reduction;
        a[1] /= sv.reduction;
        for i in 1..=num_limit as usize {
            if param_lim[i - 1] >= 0. {
                sv.a_limits[i - 1][0] = a[i - 1] - param_lim[i - 1];
                sv.a_limits[i - 1][1] = a[i - 1] + param_lim[i - 1];
            }
        }
        //
        // just get second section to work with
        read_filter_section(
            2,
            &mut sv.array,
            iz_ali,
            nx_orig,
            ny_orig,
            sv.nx,
            sv.ny,
            nx_pad,
            ny_pad,
            ibinning,
            &ctf,
            deltac,
            if_filt_after,
            i_anti_filt_type,
            if_sobel,
            &mut temp,
            len_temp,
        );
        //
        // Just deal with points in central portion
        let matt_x;
        let matt_y;
        if frac_matt >= 1. {
            matt_x = nint(frac_matt / sv.reduction);
            matt_y = matt_x;
        } else {
            matt_x = 0.max(nint(sv.nx as f32 * frac_matt));
            matt_y = 0.max(nint(sv.ny as f32 * frac_matt));
        }
        if matt_x as f32 >= 0.49 * sv.nx as f32 || matt_y as f32 >= 0.49 * sv.ny as f32 {
            exit_error("Fraction or number of pixels to ignore is too large");
        }
        if if_xmin_max != 0 {
            sv.nx1 = (sv.nx1 + ibinning - 1) / ibinning;
            sv.nx2 /= ibinning;
            if sv.nx1 <= 0 || sv.nx2 > sv.nx || sv.nx1 >= sv.nx2 {
                exit_error("Starting and ending X values out of range or reversed");
            }
        } else {
            sv.nx1 = 1 + matt_x;
            sv.nx2 = sv.nx - matt_x;
        }
        if if_ymin_max != 0 {
            sv.ny1 = (sv.ny1 + ibinning - 1) / ibinning;
            sv.ny2 /= ibinning;
            if sv.ny1 <= 0 || sv.ny2 > sv.ny || sv.ny1 >= sv.ny2 {
                exit_error("Starting and ending Y values out of range or reversed");
            }
        } else {
            sv.ny1 = 1 + matt_y;
            sv.ny2 = sv.ny - matt_y;
        }
        //
        // Adjust patch size for binning, set to one patch if none, and allocate
        sv.nxy_patch /= ibinning;
        if sv.nxy_patch != 0 && sv.nxy_patch < 10 {
            exit_error("Local patch size must be at least 10 binned pixels");
        }
        if sv.nxy_patch == 0 {
            sv.nxy_patch = (sv.nx2 + 1 - sv.nx1).max(sv.ny2 + 1 - sv.ny1);
        }
        sv.num_xpatch = (sv.nx2 + 1 - sv.nx1 + sv.nxy_patch - 1) / sv.nxy_patch;
        sv.num_ypatch = (sv.ny2 + 1 - sv.ny1 + sv.nxy_patch - 1) / sv.nxy_patch;
        // print *,nxyPatch, numXpatch, numYpatch
        let npatch = (sv.num_xpatch.max(0) as usize) * (sv.num_ypatch.max(0) as usize);
        sv.sxa = vec![0.; npatch];
        sv.sya = vec![0.; npatch];
        sv.sx_sqa = vec![0.; npatch];
        sv.sy_sqa = vec![0.; npatch];
        sv.sxya = vec![0.; npatch];
        sv.num_pix_a = vec![0; npatch];
        //
        array_min_max_mean_sd_fortran(
            &sv.array,
            &sv.nx,
            &sv.ny,
            &sv.nx1,
            &sv.nx2,
            &sv.ny1,
            &sv.ny2,
            &mut dmin2,
            &mut dmax2,
            &mut tsum,
            &mut tsum_sq,
            &mut dmean2,
            &mut sd2,
        );
        //
        let (nx, ny, nx1, nx2, ny1, ny2) = (sv.nx, sv.ny, sv.nx1, sv.nx2, sv.ny1, sv.ny2);
        if sv.if_dist == 0 {
            // if doing simple difference measure, move array into brray
            sv.brray = vec![0f32; IDIMB as usize];
            for ixy in 1..=(ny * nx) as usize {
                sv.brray[ixy - 1] = sv.array[ixy - 1];
            }
        } else {
            sv.den_low = vec![0f32; ISUB as usize];
            sv.den_high = vec![0f32; ISUB as usize];
            sv.ix_comp = vec![0i16; ISUB as usize];
            sv.iy_comp = vec![0i16; ISUB as usize];
            // if doing distance measure, make histogram of intensities
            for i in 0..=1000 {
                ihist[i] = 0;
            }
            let hist_scale = 1000. / (dmax2 - dmin2);
            for j in ny1..=ny2 {
                let ibase = nx * (j - 1);
                for i in nx1..=nx2 {
                    //
                    let ind = cvttss2si((sv.array[(i + ibase - 1) as usize] - dmin2) * hist_scale);
                    ihist[ind as usize] += 1;
                    //
                }
            }
            // convert to cumulative histogram
            for i in 1..=1000 {
                ihist[i] += ihist[i - 1];
            }
            // find density corresponding to each percentile limit
            for iran in 1..=num_ranges as usize {
                for low_high in 1..=2 {
                    let pct = if low_high == 1 {
                        pct_low[iran - 1]
                    } else {
                        pct_high[iran - 1]
                    };
                    let ncrit = cvttss2si(((nx2 + 1 - nx1) * (ny2 + 1 - ny1)) as f32 * pct / 100.);
                    let mut i: i32 = 0;
                    while i <= 1000 && ncrit > ihist[i as usize] {
                        i += 1;
                    }
                    let value = (i as f32 / hist_scale) + dmin2;
                    if low_high == 1 {
                        ran_low[iran - 1] = value;
                    } else {
                        range_high[iran - 1] = value;
                    }
                }
            }
            // find all points in central part of array within those density
            // limits
            sv.num_compare = 0;
            for iy in ny1..=ny2 {
                let ibase = nx * (iy - 1);
                for ix in nx1..=nx2 {
                    let val = sv.array[(ix + ibase - 1) as usize];
                    'ranges: for iran in 1..=num_ranges as usize {
                        if val >= ran_low[iran - 1] && val <= range_high[iran - 1] {
                            // redundancy reduction: look for previous nearby points in
                            // list
                            let mut take = true;
                            if idist_redund > 0 {
                                let ixm = ix - idist_redund;
                                let iym = iy - idist_redund;
                                let ixp = ix + idist_redund;
                                let mut ic = sv.num_compare;
                                while ic >= 1 {
                                    let ixcm = sv.ix_comp[(ic - 1) as usize] as i32;
                                    let iycm = sv.iy_comp[(ic - 1) as usize] as i32;
                                    // if point in list is nearby and within same density
                                    // range, skip this one
                                    if ixcm >= ixm
                                        && ixcm <= ixp
                                        && iycm >= iym
                                        && sv.den_low[(ic - 1) as usize] >= ran_low[iran - 1]
                                        && sv.den_low[(ic - 1) as usize] <= range_high[iran - 1]
                                    {
                                        take = false;
                                        break;
                                    }
                                    // if gotten back far enough on list, take this point
                                    if iycm < iym || (iycm == iym && ixcm < ixm) {
                                        break;
                                    }
                                    ic -= 1;
                                }
                            }
                            if take {
                                sv.num_compare += 1;
                                let n = (sv.num_compare - 1) as usize;
                                sv.ix_comp[n] = ix as i16;
                                sv.iy_comp[n] = iy as i16;
                                // just store density temporarily here
                                sv.den_low[n] = val;
                            }
                            break 'ranges;
                        }
                    }
                }
            }
            println!("{:>12}  points for comparison", sv.num_compare);
        }
        //
        // Now get first section
        read_filter_section(
            1,
            &mut sv.array,
            iz_ref,
            nx_orig,
            ny_orig,
            sv.nx,
            sv.ny,
            nx_pad,
            ny_pad,
            ibinning,
            &ctf,
            deltac,
            if_filt_after,
            i_anti_filt_type,
            if_sobel,
            &mut temp,
            len_temp,
        );
        //
        array_min_max_mean_sd_fortran(
            &sv.array,
            &sv.nx,
            &sv.ny,
            &sv.nx1,
            &sv.nx2,
            &sv.ny1,
            &sv.ny2,
            &mut dmin1,
            &mut dmax1,
            &mut tsum,
            &mut tsum_sq,
            &mut dmean1,
            &mut sv.sd1,
        );
        //
        // get scale factor for floating second array densities to match that
        // of first
        let mut scale: f32 = 1.;
        let mut d_add: f32 = 0.;
        if if_float_mean == 0 {
            scale = (dmax1 - dmin1) / (dmax2 - dmin2);
            d_add = dmin1 - scale * dmin2;
        } else if if_float_mean > 0 {
            scale = sv.sd1 / sd2;
            d_add = dmean1 - scale * dmean2;
        }
        if sv.if_dist == 0 {
            // for simple difference, rescale whole array
            for ixy in 1..=(nx * ny) as usize {
                sv.brray[ixy - 1] = scale * sv.brray[ixy - 1] + d_add;
            }
            diff(&mut sv, &mut delmin, &a);
        } else {
            // otherwise, for distance,
            // rescale list of densities and add lower and upper window
            let window = scale * 0.5 * difflim * (dmax2 - dmin2);
            for i in 1..=sv.num_compare as usize {
                let val_scale = scale * sv.den_low[i - 1] + d_add;
                sv.den_low[i - 1] = val_scale - window;
                sv.den_high[i - 1] = val_scale + window;
            }
            // find points within search radius
            let lim_dxy = cvttss2si(radius + 1.);
            sv.num_spiral = 0;
            for idx in -lim_dxy..=lim_dxy {
                for idy in -lim_dxy..=lim_dxy {
                    let distance = ((idx * idx + idy * idy) as f32).sqrt();
                    if distance <= radius {
                        sv.num_spiral += 1;
                        let n = (sv.num_spiral - 1) as usize;
                        sv.dist_spiral[n] = distance;
                        sv.idx_spiral[n] = idx;
                        sv.idy_spiral[n] = idy;
                    }
                }
            }
            // order them by distance
            for i in 1..=sv.num_spiral as usize {
                for j in i + 1..=sv.num_spiral as usize {
                    if sv.dist_spiral[i - 1] > sv.dist_spiral[j - 1] {
                        sv.dist_spiral.swap(i - 1, j - 1);
                        sv.idx_spiral.swap(i - 1, j - 1);
                        sv.idy_spiral.swap(i - 1, j - 1);
                    }
                }
            }
            //
            dist(&sv, &mut delmin, &a);
            //
        }
        sv.num_trials = 0;
        sv.delta_min = 1.0e30;
        //
        // DNM 4/29/02: search fails if images match perfectly, so skip if so
        //
        if delmin > 0. {
            let mut ptol_fac = ptol1;
            if ftol2 > 0. || ptol2 > 0. {
                ptol_fac = ptol2;
            }
            let iv_end = sv.iv_end as usize;
            amoebainitfwrap(
                &mut pp,
                &mut yy,
                7,
                iv_end,
                delta_fac,
                ptol_fac,
                &a,
                &da,
                &mut |x: &[f32]| func(&mut sv, x),
                &mut ptol,
            );
            if ftol2 > 0. || ptol2 > 0. {
                amoebafwrap(
                    &mut pp,
                    &mut yy,
                    7,
                    iv_end,
                    ftol2,
                    &mut |x: &[f32]| func(&mut sv, x),
                    &mut iter,
                    &ptol,
                    &mut jmin,
                );
                if trace {
                    println!(" restarting");
                }
                sv.delta_min = 1.0e30;
                for i in 1..=iv_end {
                    a[i - 1] = pp[(jmin - 1) + (i - 1) * 7];
                }
                amoebainitfwrap(
                    &mut pp,
                    &mut yy,
                    7,
                    iv_end,
                    delta_fac,
                    ptol1,
                    &a,
                    &da,
                    &mut |x: &[f32]| func(&mut sv, x),
                    &mut ptol,
                );
            }
            amoebafwrap(
                &mut pp,
                &mut yy,
                7,
                iv_end,
                ftol1,
                &mut |x: &[f32]| func(&mut sv, x),
                &mut iter,
                &ptol,
                &mut jmin,
            );
            //
            for i in 1..=iv_end {
                a[i - 1] = pp[(jmin - 1) + (i - 1) * 7];
            }
            sv.delta_min = yy[jmin - 1];
        } else {
            sv.delta_min = 0.;
        }
        //
        // DEPENDENCY: transferfid expects two lines with natural params
        println!("  FINAL VALUES");
        if sv.if_ccc != 0 {
            sv.delta_min = 1. - sv.delta_min;
        }
        // 72 format(i4,f14.7,1x,4f10.5,2f10.3)
        let result = XfsimplexResult {
            num_trials: sv.num_trials,
            delta_min: sv.delta_min,
            params: [a[2], a[3], a[4], a[5]],
            shifts: [sv.reduction * a[0], sv.reduction * a[1]],
            natural: sv.natural != 0,
        };
        println!("{}", xfsimplex_final_line(&result));
        RESULT_SINK.with_borrow(|slot| {
            if let Some(sink) = slot {
                *sink.lock().expect("xfsimplex result sink") = Some(result);
            }
        });
        //
        // 70 format(4f12.7,2f12.3)
        if sv.natural != 0 {
            amat = rotmag_to_amat(a[2], a[5], a[3], a[4]);
            let line = format!(
                "{}{}{}{}{}{}",
                ff(amat[0], 12, 7),
                ff(amat[2], 12, 7),
                ff(amat[1], 12, 7),
                ff(amat[3], 12, 7),
                ff(sv.reduction * a[0], 12, 3),
                ff(sv.reduction * a[1], 12, 3)
            );
            println!("{line}");
            let _ = writeln!(unit4, "{line}");
        } else {
            let _ = writeln!(
                unit4,
                "{}{}{}{}{}{}",
                ff(a[2], 12, 7),
                ff(a[3], 12, 7),
                ff(a[4], 12, 7),
                ff(a[5], 12, 7),
                ff(sv.reduction * a[0], 12, 3),
                ff(sv.reduction * a[1], 12, 3)
            );
        }
        let _ = unit4.flush();
        drop(unit4);
        //
        iiu_close(1);
        iiu_close(2);
        //
        exit(0);
    }
}

/// Original subroutine `diff` (`xfsimplex.f90:681`).
///
/// Taking the difference or correlation between images.  `crray` and
/// `drray` are the module's `array` and `brray` at every call.
pub fn diff(sv: &mut SimplexVars, delta: &mut f32, a: &[f32; 6]) {
    let mut amat = [0f32; 4];
    let mut sx: f64;
    let mut num_pix: i32;
    let mut if_old_diff: i32;
    let (mut ix_min, mut iy_min) = (0_i32, 0_i32);

    *delta = 0.;
    if_old_diff = 0;
    sx = 0.;
    num_pix = 0;
    let nxp = sv.num_xpatch;
    let pidx = |ixp: i32, iyp: i32| ((ixp - 1) + (iyp - 1) * nxp) as usize;
    for iyp in 1..=sv.num_ypatch {
        for ixp in 1..=sv.num_xpatch {
            sv.sxa[pidx(ixp, iyp)] = 0.;
            sv.sya[pidx(ixp, iyp)] = 0.;
            sv.sxya[pidx(ixp, iyp)] = 0.;
            sv.sx_sqa[pidx(ixp, iyp)] = 0.;
            sv.sy_sqa[pidx(ixp, iyp)] = 0.;
            sv.num_pix_a[pidx(ixp, iyp)] = 0;
        }
    }
    let old_diff = sv.num_xpatch * sv.num_ypatch == 1 && sv.if_ccc == 0;
    if old_diff {
        if_old_diff = 1;
    }
    let xcen = sv.nx as f32 * 0.5 + 0.5; //use + 0.5 to be consistent
    let ycen = sv.ny as f32 * 0.5 + 0.5; //with new cubinterp usage
    //
    if sv.natural == 0 {
        amat[0] = a[2];
        amat[2] = a[3];
        amat[1] = a[4];
        amat[3] = a[5];
    } else {
        amat = rotmag_to_amat(a[2], a[5], a[3], a[4]);
    }
    // print *,((amat(i, j), j=1, 2), i=1, 2)
    let mut x_add = xcen + a[0];
    let mut y_add = ycen + a[1];
    if sv.if_interp == 0 {
        x_add = xcen + a[0] + 0.5; //the 0.5 here gets nearest int
        y_add = ycen + a[1] + 0.5;
    }
    simplex_diff(
        &sv.array,
        &sv.brray,
        sv.nx,
        sv.ny,
        sv.nx1,
        sv.nx2,
        sv.ny1,
        sv.ny2,
        &amat,
        xcen,
        ycen,
        x_add,
        y_add,
        sv.if_interp,
        if_old_diff,
        sv.if_ccc,
        &mut sx,
        &mut num_pix,
        sv.num_xpatch,
        sv.nxy_patch,
        &mut sv.sxa,
        &mut sv.sya,
        &mut sv.sxya,
        &mut sv.sx_sqa,
        &mut sv.sy_sqa,
        &mut sv.num_pix_a,
    );

    if old_diff {
        if num_pix > 0 {
            *delta = (sx / (num_pix as f32 * sv.sd1) as f64) as f32;
        }
    } else {
        //
        // Combine patches with less than half the pixels with nearest patch
        // with more than half
        let num_crit = sv.nxy_patch * sv.nxy_patch / 2;
        for iyp in 1..=sv.num_ypatch {
            for ixp in 1..=sv.num_xpatch {
                if sv.num_pix_a[pidx(ixp, iyp)] > 0 && sv.num_pix_a[pidx(ixp, iyp)] < num_crit {
                    let mut min_dist = 1000000000;
                    //
                    // Find nearest qualifying patch
                    for idy in 0..=iyp.max(sv.num_ypatch - iyp) {
                        if idy * idy <= min_dist {
                            for idir_y in [-1, 1] {
                                let iy = iyp + idir_y * idy;
                                if iy >= 1 && iy <= sv.num_ypatch {
                                    for idx in 0..=ixp.max(sv.num_xpatch - ixp) {
                                        if idx * idx <= min_dist {
                                            for idir_x in [-1, 1] {
                                                let ix = ixp + idir_x * idx;
                                                if ix >= 1
                                                    && ix <= sv.num_xpatch
                                                    && sv.num_pix_a[pidx(ix, iy)] >= num_crit
                                                    && idx * idx + idy * idy < min_dist
                                                {
                                                    min_dist = idx * idx + idy * idy;
                                                    ix_min = ix;
                                                    iy_min = iy;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    //
                    // If found a qualifying closest patch, add the data into it
                    if min_dist < 100000000 {
                        // write(*,'(a,6i6)') 'Pooling', ixp, iyp, npixa(ixp, iyp), ixmin, &
                        // iymin, npixa(ixmin, iymin)
                        let (m, p) = (pidx(ix_min, iy_min), pidx(ixp, iyp));
                        sv.num_pix_a[m] += sv.num_pix_a[p];
                        sv.sxa[m] += sv.sxa[p];
                        sv.sx_sqa[m] += sv.sx_sqa[p];
                        sv.sxya[m] += sv.sxya[p];
                        sv.sya[m] += sv.sya[p];
                        sv.sy_sqa[m] += sv.sy_sqa[p];
                        sv.num_pix_a[p] = 0;
                    }
                }
            }
        }
        //
        // Now add up the delta values from the patches, weighted by number of
        // pixels
        let mut delta_sum: f32 = 0.;
        num_pix = 0;
        for iyp in 1..=sv.num_ypatch {
            for ixp in 1..=sv.num_xpatch {
                let p = pidx(ixp, iyp);
                let npa = sv.num_pix_a[p];
                if npa > 1 {
                    if sv.if_ccc == 0 {
                        //
                        // Get the SD of the diff and scale it by the sd of the data
                        let den = ((sv.sx_sqa[p] - sv.sxa[p] * sv.sxa[p] / npa as f64)
                            / (npa as f32 - 1.) as f64) as f32;
                        if den > 0. {
                            delta_sum += npa as f32 * den.sqrt() / sv.sd1;
                            num_pix += npa;
                        }
                    } else {
                        //
                        // Or get the ccc and use 1 - ccc
                        let den = ((npa as f64 * sv.sx_sqa[p] - sv.sxa[p] * sv.sxa[p])
                            * (npa as f64 * sv.sy_sqa[p] - sv.sya[p] * sv.sya[p]))
                            as f32;
                        if den > 0. {
                            let ccc = ((npa as f64 * sv.sxya[p] - sv.sxa[p] * sv.sya[p])
                                / den.sqrt() as f64) as f32;
                            delta_sum += npa as f32 * (1. - ccc);
                            num_pix += npa;
                        }
                    }
                }
            }
        }
        //
        // Take the weighted average
        if num_pix > 0 {
            *delta = delta_sum / num_pix as f32;
        }
    }
    //
    let delta_fac = outside_multiplier(a, &sv.a_limits);
    *delta *= delta_fac;
    //
    // DNM 5/17/02: if the number of pixels falls below a small threshold
    // return a big delta; otherwise adjust for # of pixels and save value
    // 5/13/06: Normalize to # of Sds difference per pixel
    //
    if delta_fac > 1.
        || num_pix as f32 > 0.02 * ((sv.nx2 + 1 - sv.nx1) * (sv.ny2 + 1 - sv.ny1)) as f32
    {
        sv.delta_last = *delta;
    } else {
        *delta = 10. * sv.delta_last;
    }
}

/// Original subroutine `dist` (`xfsimplex.f90:846`).
///
/// Or measure distance between features.  Every argument after `a` is the
/// module variable of the same name at the only call sites.
pub fn dist(sv: &SimplexVars, delta: &mut f32, a: &[f32; 6]) {
    let nx = sv.nx;
    let ny = sv.ny;
    let amat: [f32; 4];
    //
    let xcen = nx as f32 * 0.5 + 0.5;
    let ycen = ny as f32 * 0.5 + 0.5;
    //
    *delta = 0.;
    //
    let x_add = xcen + a[0] + 0.5;
    let y_add = ycen + a[1] + 0.5;
    if sv.natural == 0 {
        amat = [a[2], a[4], a[3], a[5]];
    } else {
        amat = rotmag_to_amat(a[2], a[5], a[3], a[4]);
    }
    // print *,((amat(i, j), j=1, 2), i=1, 2)
    let dist_max = sv.dist_spiral[(sv.num_spiral - 1) as usize] + 1.;
    for icomp in 1..=sv.num_compare as usize {
        let fj = sv.iy_comp[icomp - 1] as f32 - ycen;
        let fi = sv.ix_comp[icomp - 1] as f32 - xcen;
        //
        let ix = cvttss2si(amat[0] * fi + amat[2] * fj + x_add);
        let iy = cvttss2si(amat[1] * fi + amat[3] * fj + y_add);
        //
        let mut distance = dist_max;
        let crit_low = sv.den_low[icomp - 1];
        let crit_high = sv.den_high[icomp - 1];
        for ispir in 1..=sv.num_spiral as usize {
            let ixa = nx.min(1.max(ix + sv.idx_spiral[ispir - 1]));
            let iya = ny.min(1.max(iy + sv.idy_spiral[ispir - 1]));
            let den1 = sv.array[(ixa - 1 + (iya - 1) * nx) as usize];
            if den1 >= crit_low && den1 <= crit_high {
                distance = sv.dist_spiral[ispir - 1];
                break;
            }
        }
        *delta += distance;
        //
    }
    //
    // 5/13/06: Normalize to per comparison point to match normalization
    // of difference measure
    //
    let delta_fac = outside_multiplier(a, &sv.a_limits);
    *delta = delta_fac * *delta / sv.num_compare as f32;
}

/// Original function `outsideMultiplier` (`xfsimplex.f90:912`).
///
/// find a multiplier for the delta factor if outside limits
pub fn outside_multiplier(a: &[f32; 6], a_limits: &[[f32; 2]; 6]) -> f32 {
    let mut delta_fac: f32 = 1.;
    for i in 1..=6 {
        let (lo, hi) = (a_limits[i - 1][0], a_limits[i - 1][1]);
        if lo < hi {
            let (u, v) = (a[i - 1] - hi, lo - a[i - 1]);
            let outside = (if u > v { u } else { v }) / (hi - lo);
            if outside > 0. {
                let p = 100f32.powf(if 5. < outside { 5. } else { outside });
                delta_fac = if delta_fac > p { delta_fac } else { p };
            }
        }
    }
    delta_fac
}

/// Original subroutine `func` (`xfsimplex.f90:931`): the function called by
/// the minimization routine.
pub fn func(sv: &mut SimplexVars, x: &[f32]) -> f32 {
    let mut a = [0f32; 6];
    let mut delta = 0f32;
    //
    for i in 1..=sv.iv_end as usize {
        a[i - 1] = x[i - 1];
    }
    for i in (sv.iv_end + 1) as usize..=6 {
        a[i - 1] = sv.acall[i - 1];
    }
    //
    if sv.if_dist == 0 {
        diff(sv, &mut delta, &a);
    } else {
        dist(sv, &mut delta, &a);
    }
    let error = delta;
    sv.num_trials += 1;
    if sv.if_trace != 0 {
        let mut star_out = b' ';
        if delta < sv.delta_min {
            star_out = b'*';
            sv.delta_min = delta;
        }
        let mut delta_out = delta;
        if sv.if_ccc != 0 {
            delta_out = 1. - delta;
        }
        if sv.if_trace == 1 || star_out == b'*' {
            // 72 format(1x,a1,i3,f14.7,4f10.5,2f10.3)
            println!(
                " {}{}{}{}{}{}{}{}{}",
                star_out as char,
                fortran_i(sv.num_trials, 3),
                ff(delta_out, 14, 7),
                ff(a[2], 10, 5),
                ff(a[3], 10, 5),
                ff(a[4], 10, 5),
                ff(a[5], 10, 5),
                ff(sv.reduction * a[0], 10, 3),
                ff(sv.reduction * a[1], 10, 3)
            );
        }
    }
    error
}

/// Original subroutine `readFilterSection` (`xfsimplex.f90:973`).
///
/// Read a section with optional fourier filtering before or after
/// binning, and with optional sobel filtering
#[allow(clippy::too_many_arguments)]
pub fn read_filter_section(
    iunit: i32,
    array: &mut [f32],
    iz: i32,
    nx_orig: i32,
    ny_orig: i32,
    nx: i32,
    ny: i32,
    nx_pad: i32,
    ny_pad: i32,
    ibinning: i32,
    ctf: &[f32],
    deltac: f32,
    if_filt_after: i32,
    i_anti_filt_type: i32,
    if_sobel: i32,
    temp: &mut [f32],
    len_temp: i32,
) {
    let mut ierr = 0;
    let mut nx_filt: i32;
    let mut ny_filt: i32;
    let (mut x_offset, mut y_offset) = (0f32, 0f32);
    //
    // Read in the section without or with binning, set size being filtered
    if deltac != 0. && if_filt_after == 0 {
        unsafe {
            iiu_set_position(iunit, iz, 0);
            // `call irdsec(iunit, array, 99)` passes 99 as a plain argument,
            // not `*99`: native never takes the alternate return and a failed
            // read goes on.  BUGS.md, fixed in translation: the `*99` exit
            // ("Reading file") the source evidently meant is taken.
            if irdsec(iunit, array).is_err() {
                exit_error("Reading file");
            }
        }
        nx_filt = nx_orig;
        ny_filt = ny_orig;
    } else {
        if i_anti_filt_type > 0 {
            iiu_read_reduced(
                iunit,
                iz,
                array,
                nx,
                0.,
                0.,
                ibinning as f32,
                nx,
                ny,
                i_anti_filt_type - 1,
                temp,
                len_temp,
                &mut ierr,
            );
        } else {
            iiu_read_binned(
                iunit, iz, array, nx, ny, 0, 0, ibinning, nx, ny, temp, len_temp, &mut ierr,
            );
        }
        if ierr != 0 {
            exit_error("Reading file");
        }
        nx_filt = nx;
        ny_filt = ny;
    }
    //
    // Apply fourier filter
    if deltac != 0. {
        // taperOutPad(array, nxFilt, nyFilt, array, nxPad + 2, nxPad, nyPad, 0, 0):
        // the last 0 is an integer handed to a real argument, the bits of 0.
        taperoutpad(
            PadIn::InPlace,
            &nx_filt,
            &ny_filt,
            array,
            &(nx_pad + 2),
            &nx_pad,
            &ny_pad,
            &0,
            &0.,
        );
        todfft(array, nx_pad, ny_pad, 0);
        xcorr_filter_part(FilterIn::InPlace, array, nx_pad, ny_pad, ctf, deltac);
        todfft(array, nx_pad, ny_pad, 1);
        let ix_low = (nx_pad - nx_filt) / 2;
        let iy_low = (ny_pad - ny_filt) / 2;
        {
            let count = ((nx_pad + 2) * ny_pad) as usize;
            let source: Vec<u8> = array[..count]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            // SAFETY: the `f32` buffer viewed as its bytes, as the source's
            // `irepak(array, array, ...)` passes the same storage.
            let destination = unsafe {
                core::slice::from_raw_parts_mut(array.as_mut_ptr().cast::<u8>(), array.len() * 4)
            };
            irepak(
                destination,
                &source,
                &(nx_pad + 2),
                &ny_pad,
                &ix_low,
                &(ix_low + nx_filt - 1),
                &iy_low,
                &(iy_low + ny_filt - 1),
            );
        }
        //
        // Now bin if necessary
        if if_filt_after == 0 && ibinning > 1 {
            let count = (nx_orig * ny_orig) as usize;
            let source: Vec<u8> = array[..count]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            // SAFETY: as above; `reduce_by_binning(array, ..., array, ...)`.
            let destination = unsafe {
                core::slice::from_raw_parts_mut(array.as_mut_ptr().cast::<u8>(), array.len() * 4)
            };
            reduce_by_binning(
                &source,
                SLICE_MODE_FLOAT,
                nx_orig,
                ny_orig,
                ibinning,
                destination,
                0,
                &mut nx_filt,
                &mut ny_filt,
            );
        }
    }
    //
    // Do sobel if requested: NOTE THAT IT CAN BE DONE IN PLACE AS LONG AS
    // SCALING = 1 BUT THIS SHOULD BE DOCUMENTED
    if if_sobel != 0 {
        // scaledSobel(array, nx, ny, 1., 1., 1, 2, array, ...): the `2` is an
        // integer handed to the real argument `center`, so native sees the
        // float with bit pattern 2 (a denormal).  BUGS.md, fixed in
        // translation: the Sobel centre weight 2.0 the call means.
        let input: Vec<f32> = array[..(nx * ny) as usize].to_vec();
        if scaled_sobel(
            Some(&input),
            nx,
            ny,
            1.,
            1.,
            1,
            2.,
            Some(array),
            &mut nx_filt,
            &mut ny_filt,
            &mut x_offset,
            &mut y_offset,
        ) != 0
        {
            exit_error("Getting memory for sobel filtering");
        }
    }
}

/// Original subroutine `checkForWarpFile` (`xfsimplex.f90:1032`).
///
/// Routine to check for a warp file and exit with error if so
pub fn check_for_warp_file(xf_in_file: &str) {
    let mut strn_tmp = String::new();
    let (mut idx, mut idy, mut itmp, mut i, mut jj) = (0, 0, 0, 0, 0);
    let mut deltac = 0f32;
    let ierr = read_check_warp_file(
        xf_in_file,
        0,
        0,
        &mut idx,
        &mut idy,
        &mut itmp,
        &mut i,
        &mut deltac,
        &mut jj,
        &mut strn_tmp,
    );
    if ierr >= 0 {
        exit_error("The initial transform file contains warping transforms");
    }
    if ierr != -1 {
        println!(
            "\nERROR: A problem occurred testing whether the initial transform file had warpings"
        );
        exit_error(&strn_tmp);
    }
}
