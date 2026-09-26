//! Translation of `IMOD/flib/model/corrsearch3d.f90`.
//!
//! CORRSEARCH3D will determine the 3D displacement between two image
//! volumes at a regular array of positions.  At each position, it
//! extracts a patch of a specified size from each volume, then
//! searches for a local maximum in the cross-correlation between the
//! two patches.  The starting point of the search is based upon the
//! displacements of adjacent patches that have already been determined.
//! If there are no such adjacent patches, or if no maximum is found for
//! displacements within a specified range, the program uses FFT-based
//! cross-correlation instead.
//!
//! One Rust function per program unit: the main program [`corrsearch3d`]
//! and its five contained procedures ([`tilt_vertex_setup`],
//! [`load_extract_process`], [`analyze_local_sds`], [`add_boxes_to_sample`],
//! [`find_boxes_inside_patch`]), which reach the host's variables through
//! [`MainVars`]; and the external units [`find_best_corr`], [`three_corrs`],
//! [`one_corr_coeff`], [`kernel_smooth`], [`smooth_one_plane`],
//! [`fourier_corr`], [`find_xcorr_peak`], [`set_bload`], [`manage_load`],
//! [`load_vol`], [`extract_patch`], [`vol_mean_zero`],
//! [`check_and_set_patches`], [`revise_patch_range`],
//! [`xform_bsource_to_a`], [`sequence_patches`] and [`lsd_load_func`].
//! `dumpVolume` is not translated: every call to it in the source is
//! commented out (`DEAD_CODE.md`).
//!
//! `buffer` is the one `real*4` allocation the source carves into patch,
//! scratch and load areas by 1-based starting index; it is a `Vec<f32>` here
//! and every `buffer(ind...)` actual argument is the slice (or, for
//! `taperInVol`, which the source calls with its input and output aliased,
//! the offset) starting at `ind - 1`.
//!
//! OpenMP: the source parallelises `threeCorrs`, `oneCorrCoeff`
//! (`real*8` `REDUCTION(+)`), `kernelSmooth` (planes) and the per-patch loop
//! of `analyzeLocalSDs`.  The reductions combine per-thread partial sums in
//! thread-completion order, so native's result depends on its thread count;
//! the translation runs every loop sequentially, which is what native computes
//! at `OMP_NUM_THREADS=1` (`TO_OPT.md`: no rayon where the output can depend
//! on the thread count).  The other two loops are thread-independent.
//!
//! `kernelSmooth`'s weight loop is vectorised by gfortran through libmvec:
//! for a 5x5x5 kernel `corrsearch3d.o` evaluates `exp` of the first four `k`
//! values with `_ZGVbN4v_expf` and the fifth with scalar `expf`, and for 3x3x3
//! all with `expf` (disassembly of `kernelsmooth_`).  The two do not always
//! agree to the last bit, so the translation calls the same libmvec entry
//! point ([`kernel_smooth`]).

use crate::imod::flib::model::get_region_contours::{
    check_boundary_conts, get_contour_array_sizes, get_region_contours,
};
use crate::imod::flib::subrs::compat::gfortran_rt::{gfortran_cosd_r4, gfortran_sind_r4, maxss};
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::get_nxyz::get_nxyz;
use crate::imod::flib::subrs::hvem::indmap::indmap;
use crate::imod::flib::subrs::hvem::inside::inside;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::taperinvol::taper_in_vol;
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdpas};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libcfshr::b3dutil::{exit, num_omp_threads, wall_time};
use crate::imod::libcfshr::filtxcorr::{nice_frame, parabolic_fit_position, xcorr_set_ctf};
use crate::imod::libcfshr::histogram::findhistogramdip;
use crate::imod::libcfshr::linearxforms::xfapply;
use crate::imod::libcfshr::multibinstat::{multi_bin_setup, multi_bin_stats};
use crate::imod::libcfshr::parse_params::{
    pip_get_boolean, pip_get_float, pip_get_integer, pip_get_three_floats, pip_get_three_integers,
    pip_get_two_floats, pip_get_two_integers,
};
use crate::imod::libcfshr::percentile::percentile_float;
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libcfshr::robuststat::{rs_fast_madn, rs_median_of_sorted, rs_sort_floats};
use crate::imod::libfft::{nice_fft_limit, thrdfft, using_fftw};
use crate::imod::libiimod::unit_fileio::{iiu_read_sec_part, iiu_set_position};
use std::io::{Seek, SeekFrom, Write};

/// `parameter (LIM_LOCAL_BIN = 6)` (`corrsearch3d.f90:22`).
const LIM_LOCAL_BIN: usize = 6;
/// `parameter (numOptions = 34)` (`corrsearch3d.f90:105`).
const NUM_OPTIONS: i32 = 34;
/// Fallback PIP table `options(1)` (`corrsearch3d.f90:110-124`).
const OPTIONS: &str = "ref:ReferenceFile:FN:@align:FileToAlign:FN:@output:OutputFile:FN:@\
region:RegionModel:FN:@size:PatchSizeXYZ:IT:@number:NumberOfPatchesXYZ:IT:@\
xminmax:XMinAndMax:IP:@yminmax:YMinAndMax:IP:@zminmax:ZMinAndMax:IP:@\
tilt:TiltSeriesSizeXY:IP:@btilt:BTiltSeriesSizeXY:IP:@\
axis:AxisRotationAngle:F:@taper:TapersInXYZ:IT:@pad:PadsInXYZ:IT:@\
maxshift:MaximumShift:I:@volume:VolumeShiftXYZ:FT:@\
initial:InitialShiftXYZ:FT:@bsource:BSourceOrSizeXYZ:CH:@\
bxform:BSourceTransform:FN:@bxborder:BSourceBorderXLoHi:IP:@\
byzborder:BSourceBorderYZLoHi:IP:@bregion:BRegionModel:FN:@\
binnings:LocalSDNumBinnings:I:@box:BoxSizeForLocalSD:IT:@\
elim:EliminateByLocalSD:FP:@kernel:KernelSigma:F:@ksize:KernelSize:F:@\
lowpass:LowPassRadiusSigma:FP:@sigma1:HighPassSigma:F:@\
messages:FlipYZMessages:B:@invert:InvertYLimits:B:@debug:DebugMode:I:@\
param:ParameterFile:PF:@help:usage:B:";

#[link(name = "mvec")]
unsafe extern "C" {
    /// glibc libmvec's 4-lane SSE `expf` (`_ZGVbN4v_expf@GLIBC_2.22`), the
    /// vector variant gfortran calls from `kernelSmooth`.  Its vector ABI
    /// (argument and result in `xmm0`) has no stable Rust FFI type, so it is
    /// declared as a bare symbol and reached through `asm!` in
    /// [`kernel_smooth`].
    fn _ZGVbN4v_expf();
}

/// gfortran `NINT` of a `real*4` (`lroundf`): round half away from zero.
fn nint(x: f32) -> i32 {
    x.round() as i32
}

/// libgfortran `Fw.d` output editing: overflow is `w` asterisks, a leading
/// zero is dropped when that is what makes the value fit, `d = 0` keeps the
/// decimal point, a negative value that rounds to zero keeps its sign.  The
/// digits are the exact binary value rounded to nearest, ties to even, as
/// `snprintf` does.
fn fmt_f(value: f64, w: usize, d: usize) -> String {
    if value.is_nan() {
        return format!("{:>w$}", "NaN");
    }
    if value.is_infinite() {
        let mut text = if value < 0. { "-Infinity" } else { "Infinity" };
        if text.len() > w {
            text = if value < 0. { "-Inf" } else { "Inf" };
        }
        if text.len() > w {
            return "*".repeat(w);
        }
        return format!("{text:>w$}");
    }
    let mut text = format!("{value:.d$}");
    if d == 0 {
        text.push('.');
    }
    if text.len() > w {
        if let Some(rest) = text.strip_prefix("0.") {
            text = format!(".{rest}");
        } else if let Some(rest) = text.strip_prefix("-0.") {
            text = format!("-.{rest}");
        }
    }
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// libgfortran `Iw` output editing.
fn fmt_i(value: i32, w: usize) -> String {
    let text = format!("{value}");
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// One item of a gfortran list-directed output list.
enum Ld<'a> {
    I(i32),
    R(f32),
    S(&'a str),
}

/// libgfortran list-directed `WRITE`/`PRINT *` of one record (`write.c`,
/// `list_formatted_write`): the record begins with a blank, and every item
/// after the first is preceded by a blank separator except a character item
/// that follows another; an integer is `I11`; a `real*4` is the 1PG16.9-like
/// list form (nine significant digits, `F` with four trailing blanks for
/// magnitudes in [0.1, 1e9), `E` with a two-digit exponent otherwise).
fn ld_record(items: &[Ld]) -> String {
    let mut out = String::new();
    let mut prev_char = false;
    for (n, item) in items.iter().enumerate() {
        match item {
            Ld::I(value) => {
                out.push(' ');
                out.push_str(&format!("{value:>11}"));
                prev_char = false;
            }
            Ld::R(value) => {
                let value = *value;
                out.push(' ');
                let field = if value.is_nan() {
                    format!("{:>16}", "NaN")
                } else if value.is_infinite() {
                    format!("{:>16}", if value < 0. { "-Infinity" } else { "Infinity" })
                } else if value == 0. {
                    format!("{:>12}    ", format!("{value:.8}"))
                } else {
                    let scientific = format!("{:.8e}", value.abs());
                    let (mantissa, power) = scientific.split_once('e').unwrap();
                    let k = power.parse::<i32>().unwrap() + 1;
                    if (0..=9).contains(&k) {
                        let mut text = format!("{:.*}", (9 - k) as usize, value);
                        if k == 9 {
                            text.push('.');
                        }
                        format!("{text:>12}    ")
                    } else {
                        let e = k - 1;
                        format!(
                            "{:>16}",
                            format!(
                                "{}{}E{}{:02}",
                                if value < 0. { "-" } else { "" },
                                mantissa,
                                if e < 0 { '-' } else { '+' },
                                e.abs()
                            )
                        )
                    }
                };
                out.push_str(&field);
                prev_char = false;
            }
            Ld::S(text) => {
                if n == 0 || !prev_char {
                    out.push(' ');
                }
                out.push_str(text);
                prev_char = true;
            }
        }
    }
    out
}

/// `105 format(3i6,3f9.2,f10.4,3f9.2,f12.4,f8.3)` (`corrsearch3d.f90:976`)
/// with as many real items as are given (6, 7 or 12 values in all).
fn format_105(ix: i32, iy: i32, iz: i32, reals: &[f32]) -> String {
    let widths: [(usize, usize); 9] = [
        (9, 2),
        (9, 2),
        (9, 2),
        (10, 4),
        (9, 2),
        (9, 2),
        (9, 2),
        (12, 4),
        (8, 3),
    ];
    let mut out = format!("{}{}{}", fmt_i(ix, 6), fmt_i(iy, 6), fmt_i(iz, 6));
    for (value, (w, d)) in reals.iter().zip(widths.iter()) {
        out.push_str(&fmt_f(*value as f64, *w, *d));
    }
    out.push('\n');
    out
}

/// The main program's variables that its contained procedures reach by host
/// association (`tiltVertexSetup`, `loadExtractProcess`, `analyzeLocalSDs`,
/// `addBoxesToSample`, `findBoxesInsidePatch`).
struct MainVars {
    nxyz: [i32; 3],
    nx_patch: i32,
    ny_patch: i32,
    nz_patch: i32,
    if_debug: i32,
    axis_rotation: f32,
    // loadExtractProcess
    buffer: Vec<f32>,
    wall_start: f64,
    wall_cum: [f64; 5],
    ix_dir: i32,
    ix_delta: i32,
    iy_delta: i32,
    iz_delta: i32,
    max_xload: i32,
    load_full_width: bool,
    nx_patch_tmp: i32,
    ny_patch_tmp: i32,
    nz_patch_tmp: i32,
    ind_scratch: i32,
    nx_taper: i32,
    ny_taper: i32,
    nz_taper: i32,
    kernel_size: i32,
    num_smooth_threads: i32,
    // analyzeLocalSDs and friends
    ix_start: i32,
    iy_start: i32,
    iz_start: i32,
    num_xpatch: i32,
    num_ypatch: i32,
    num_zpatch: i32,
    if_flip: i32,
    lim_patch: i32,
    num_lsd_binnings: i32,
    lsd_box_size: [[i32; 3]; LIM_LOCAL_BIN],
    lsd_binning: [[i32; 3]; LIM_LOCAL_BIN],
    lsd_start_coord: [i32; 3],
    lsd_end_coord: [i32; 3],
    lsd_box_start: [[i32; 3]; LIM_LOCAL_BIN],
    num_lsd_boxes: [[i32; 3]; LIM_LOCAL_BIN],
    lsd_buffer_starts: [i32; LIM_LOCAL_BIN + 1],
    lsd_stat_starts: [i32; LIM_LOCAL_BIN + 1],
    lsd_spacing: [[i32; 3]; LIM_LOCAL_BIN],
    ib_start: [i32; 3],
    ib_end: [i32; 3],
    num_in_sample: i32,
    stat_means: Vec<f32>,
    stat_sds: Vec<f32>,
    stat_buffer: Vec<f32>,
    patch_frac_high_sd: Vec<f32>,
    patch_mean_struct: Vec<f32>,
    ibin_best: i32,
    num_hist_bins: i32,
    num_elim_by_sd: i32,
    edge_sample_frac: f32,
    fallback_struct_frac: f32,
    better_scale_crit: f32,
    sd_elim_max_frac: f32,
    fallback_pct_frac: f32,
    frac_for_hist: f32,
    elim_by_sd_crit: f32,
    elim_by_sd_type: f32,
    high_sd_level: f32,
    bottom_hist_frac: f32,
    bad_hist_peak_crit: f32,
    if_hist_verbose: i32,
}

/// Original program `corrsearch3d` (`corrsearch3d.f90:19`).
pub fn corrsearch3d() {
    // `integer*4 nx, ny, nz, nx2, ny2, nz2, nxyz(3), nxyz2(3)` with
    // `equivalence`: the scalars are the array elements.
    let mut nxyz2 = [0_i32; 3];
    let mut dxyz_init = [0.0_f32; 3];
    let mut dxyz_vol = [0.0_f32; 3];
    let mut mxyz = [0_i32; 3];
    let mut nxyz_bsource = [0_i32; 3];
    let mut lim_vert = 0_i32;
    let mut lim_cont = 0_i32;
    let mut ctf = vec![0.0_f32; 8193];
    //
    let mut flip_messages: bool;
    let mut x_vertex: Vec<f32> = Vec::new();
    let mut y_vertex: Vec<f32> = Vec::new();
    let mut zcont: Vec<f32> = Vec::new();
    let mut ind_vert: Vec<i32> = Vec::new();
    let mut num_vertex: Vec<i32> = Vec::new();
    let mut ind_vert_b: Vec<i32> = Vec::new();
    let mut num_vert_b: Vec<i32> = Vec::new();
    let mut x_vertex_b: Vec<f32> = Vec::new();
    let mut y_vertex_b: Vec<f32> = Vec::new();
    let mut zcont_b: Vec<f32> = Vec::new();
    // `character*320 fileA, fileB, outputFile, modelFile, tempFile, xfFile, bModel`
    let mut file_a = String::new();
    let mut file_b = String::new();
    let mut output_file = String::new();
    let mut model_file: String;
    let mut xf_file: String;
    let mut b_model: String;
    //
    let mut xvert_bsource = [0.0_f32; 4];
    let mut yvert_bsource = [0.0_f32; 4];
    let mut xvert_sort = [0.0_f32; 4];
    let mut yvert_sort = [0.0_f32; 4];
    // Set by `tiltVertexSetup` only when a tilt series size was entered, and
    // read only then.
    let mut xvert_tilt = [0.0_f32; 4];
    let mut yvert_tilt = [0.0_f32; 4];
    let mut xvert_tilt_b = [0.0_f32; 4];
    let mut yvert_tilt_b = [0.0_f32; 4];
    //
    let mut max_shift: i32;
    let lim_work: i32;
    let mut num_fourier_patch: i32;
    let mut num_corrs: i32;
    let mut mode = 0_i32;
    let mut nbord_xlow = 0_i32;
    let mut nbord_xhigh = 0_i32;
    let mut nbord_ylow = 0_i32;
    let mut nbord_yhigh = 0_i32;
    let mut nbord_zlow = 0_i32;
    let mut nbord_zhigh = 0_i32;
    let mut nbord_source_xlow: i32;
    let mut nbord_source_xhigh: i32;
    let mut nbord_source_zlow: i32;
    let mut nbord_source_zhigh: i32;
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut tmp: f32;
    let mut radius2: f32;
    let mut sigma1: f32;
    let mut sigma2: f32;
    let mut sigma_kernel: f32;
    let mut delta: f32;
    let mut a_scale: f32;
    let mut b_scale: f32;
    let mut i: i32;
    let mut j: i32;
    let mut ind_yb: i32;
    let mut num_sequence = 0_i32;
    let mut nx_ctf: i32;
    // Uninitialised in the source before the first Fourier correlation
    // compares them (`corrsearch3d.f90:913`; only `nxCTF` is set, `:346`).
    // Fixed in translation (BUGS.md): they start at 0, so the filter is
    // always built for the first correlation -- what the reference build's
    // zeroed static storage gives.
    let mut ny_ctf = 0_i32;
    let mut nz_ctf = 0_i32;
    let mut nx_pad: i32;
    let mut ny_pad: i32;
    let mut nz_pad: i32;
    let num_patch_pixels: i32;
    let ind_patch_a: i32;
    let mut idim: i32;
    let ind_patch_b: i32;
    let ind_load_a: i32;
    let ind_load_b: i32;
    let mut num_cont: i32;
    let idim_optimal: i32;
    let idim_higher: i32;
    let mut ierr: i32;
    let mut num_pos_total = 0_i32;
    let mut num_bcont: i32;
    let mut if_flip_b = 0_i32;
    let mut nx_series: i32;
    let mut ny_series = 0_i32;
    let mut nx_series_b: i32;
    let mut ny_series_b = 0_i32;
    let if_axis: i32;
    let mut xcen: f32;
    let mut ycen: f32;
    let mut zcen: f32;
    let mut xpatch_low: f32;
    let mut xpatch_high: f32;
    let mut zpatch_low: f32;
    let mut zpatch_high: f32;
    let mut ind_p: i32;
    let mut if_use = 0_i32;
    // Host `icontMin`, printed by the `-debug 2` elimination message but
    // never assigned in the main program (`checkBoundaryConts` has its own,
    // `corrsearch3d.f90:576`; native prints the zeroed static, 0).  Fixed in
    // translation (BUGS.md): it receives the contour `checkBoundaryConts`
    // found, so the message names the contour that eliminated the patch.
    let mut icont_min = 0_i32;
    let mut iz0: i32;
    let mut iz1: i32;
    let mut iz_cen = 0_i32;
    let mut iy0: i32;
    let mut iy1: i32;
    let mut iy_cen: i32;
    let mut ix0: i32;
    let mut ix1: i32;
    let mut ix_cen: i32;
    let mut num_adjacent: i32;
    let mut ix_adjacent: i32;
    let mut iy_adjacent: i32;
    let mut iz_adjacent: i32;
    let mut ind_a: i32;
    let mut ix_patch: i32;
    let mut iy_patch: i32;
    let mut iz_patch: i32;
    let (mut load_y0, mut load_y1, mut load_z0, mut load_z1, mut load_x0, mut load_x1) =
        (0_i32, 0_i32, 0_i32, 0_i32, 0_i32, 0_i32);
    let ny_load_ex: i32;
    let nz_load_ex: i32;
    let (mut nx_load, mut ny_load, mut nz_load) = (0_i32, 0_i32, 0_i32);
    let mut num_near: i32;
    let load_extra: i32;
    let mut num_skip: i32;
    let (mut load_yb0, mut load_yb1, mut load_zb0, mut load_zb1, mut load_xb0, mut load_xb1) =
        (0_i32, 0_i32, 0_i32, 0_i32, 0_i32, 0_i32);
    let (mut nx_load_b, mut ny_load_b, mut nz_load_b) = (0_i32, 0_i32, 0_i32);
    let (mut ix_b0, mut ix_b1, mut iy_b0, mut iy_b1, mut iz_b0, mut iz_b1) =
        (0_i32, 0_i32, 0_i32, 0_i32, 0_i32, 0_i32);
    let (mut ix_min, mut ix_max, mut iy_min, mut iy_max, mut iz_min, mut iz_max) =
        (0_i32, 0_i32, 0_i32, 0_i32, 0_i32, 0_i32);
    let mut if_shift_in: i32;
    let mut num_adjacent_look: i32;
    let (mut nx_xcpad, mut ny_xcpad, mut nz_xcpad): (i32, i32, i32);
    let (mut nx_xcbord, mut ny_xcbord, mut nz_xcbord): (i32, i32, i32);
    let nice_lim: i32;
    let mut num3_corr_threads: i32;
    let mut num_ccc_threads: i32;
    let mut a_source = [[0.0_f32; 3]; 3];
    let mut dxyz_source = [0.0_f32; 3];
    let (mut dx_new, mut dy_new, mut dz_new) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut peak = 0.0_f32;
    let mut wsum_adjacent: f32;
    let mut wsum_near: f32;
    let (mut dx_sum, mut dy_sum, mut dz_sum): (f32, f32, f32);
    let per_pos: f32;
    let (mut dx_adjacent, mut dy_adjacent, mut dz_adjacent): (f32, f32, f32);
    let mut zmod_cen: f32;
    let (mut dx_sum_near, mut dy_sum_near, mut dz_sum_near): (f32, f32, f32);
    let (mut dx_near, mut dy_near, mut dz_near): (f32, f32, f32);
    let mut dist_sq: f32;
    let dist_near: f32;
    let cc_ratio: f32;
    let mut wcc: f32;
    let size_switch: f32;
    let mut ymod_cen: f32;
    let dist_adjacent: f32;
    let id_ccc_col = 1_i32;
    let id_frac_col = 5_i32;
    let id_struct_col = 6_i32;
    let mut wall_best: f64;
    let mut wall_ccc: f64;
    let mut found = false;
    let mut num_opt_arg = 0_i32;
    let mut num_non_opt_arg = 0_i32;
    let pip_input: bool;
    let mut fm = FortModel::default();

    let mut h = MainVars {
        nxyz: [0; 3],
        nx_patch: 0,
        ny_patch: 0,
        nz_patch: 0,
        if_debug: 0,
        axis_rotation: 0.,
        buffer: Vec::new(),
        wall_start: 0.,
        wall_cum: [0.; 5],
        ix_dir: 0,
        ix_delta: 0,
        iy_delta: 0,
        iz_delta: 0,
        max_xload: 0,
        load_full_width: false,
        nx_patch_tmp: 0,
        ny_patch_tmp: 0,
        nz_patch_tmp: 0,
        ind_scratch: 0,
        nx_taper: 0,
        ny_taper: 0,
        nz_taper: 0,
        kernel_size: 0,
        num_smooth_threads: 0,
        ix_start: 0,
        iy_start: 0,
        iz_start: 0,
        num_xpatch: 0,
        num_ypatch: 0,
        num_zpatch: 0,
        if_flip: 0,
        lim_patch: 0,
        num_lsd_binnings: 0,
        lsd_box_size: [[0; 3]; LIM_LOCAL_BIN],
        // `lsdBinning(3, LIM_LOCAL_BIN) /1,1,1, 2,2,2, 3,3,3, 4,4,4, 6,6,6, 8,8,8/`
        lsd_binning: [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [6, 6, 6],
            [8, 8, 8],
        ],
        lsd_start_coord: [0; 3],
        lsd_end_coord: [0; 3],
        lsd_box_start: [[0; 3]; LIM_LOCAL_BIN],
        num_lsd_boxes: [[0; 3]; LIM_LOCAL_BIN],
        lsd_buffer_starts: [0; LIM_LOCAL_BIN + 1],
        lsd_stat_starts: [0; LIM_LOCAL_BIN + 1],
        lsd_spacing: [[0; 3]; LIM_LOCAL_BIN],
        ib_start: [0; 3],
        ib_end: [0; 3],
        num_in_sample: 0,
        stat_means: Vec::new(),
        stat_sds: Vec::new(),
        stat_buffer: Vec::new(),
        patch_frac_high_sd: Vec::new(),
        patch_mean_struct: Vec::new(),
        ibin_best: 0,
        num_hist_bins: 0,
        num_elim_by_sd: 0,
        edge_sample_frac: 0.,
        fallback_struct_frac: 0.,
        better_scale_crit: 0.,
        sd_elim_max_frac: 0.,
        fallback_pct_frac: 0.,
        frac_for_hist: 0.,
        elim_by_sd_crit: 0.,
        elim_by_sd_type: 0.,
        high_sd_level: 0.,
        bottom_hist_frac: 0.,
        bad_hist_peak_crit: 0.,
        if_hist_verbose: 0,
    };

    // `read(5,*)`/`read(5,50)` with no `END=`/`ERR=`: the gfortran runtime
    // reports a failed read and stops with status 2.
    let read_abort = |err: ListReadError| -> ! {
        let _ = std::io::stdout().flush();
        match err {
            ListReadError::End => eprintln!("Fortran runtime error: End of file"),
            ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
        }
        exit(2);
    };
    // `read(5, 50) string` with `50 format(a)`.
    let read_line = || -> String {
        let _ = std::io::stdout().flush();
        let mut text = String::new();
        match std::io::stdin().read_line(&mut text) {
            Ok(0) | Err(_) => {
                eprintln!("Fortran runtime error: End of file");
                exit(2);
            }
            Ok(_) => {}
        }
        text.trim_end_matches(['\r', '\n'])
            .trim_end_matches(' ')
            .to_string()
    };
    // `PipGetString(option, string)` into a `character*320` variable.
    let pip_string = |option: &str, value: &mut String| -> i32 {
        let mut record = [b' '; 320];
        let bytes = value.as_bytes();
        let count = bytes.len().min(320);
        record[..count].copy_from_slice(&bytes[..count]);
        let ierr = pipgetstring_(option.as_bytes(), &mut record);
        *value = crate::imod::libcfshr::b3dutil::fortran_string(&record);
        ierr
    };
    // `indPatch(ix, iy, iz)` statement function, 1-based.
    let ind_patch = |h: &MainVars, ix: i32, iy: i32, iz: i32| -> i32 {
        ix + (iy - 1) * h.num_xpatch + (iz - 1) * h.num_xpatch * h.num_ypatch
    };

    //
    // IFDEBUG 1 for debugging output, 2 for dummy patch output at all
    // positions, 3 to do both search and fourier cross-correlation everywhere
    //
    h.if_debug = 0;
    num_fourier_patch = 0;
    num_corrs = 0;
    nxyz_bsource[0] = 0;
    nbord_source_xlow = 36;
    nbord_source_xhigh = 36;
    nbord_source_zlow = 36;
    nbord_source_zhigh = 36;
    max_shift = 10;
    for i in 0..3 {
        dxyz_vol[i] = 0.;
        dxyz_init[i] = 0.;
    }
    model_file = String::new();
    xf_file = String::new();
    b_model = String::new();
    radius2 = 0.;
    sigma1 = 0.;
    sigma2 = 0.;
    sigma_kernel = 0.;
    h.kernel_size = 3;
    size_switch = 1.49;
    delta = 0.;
    if_shift_in = 0;
    dist_adjacent = 1.5;
    dist_near = 4.;
    cc_ratio = 0.33;
    flip_messages = false;
    h.num_lsd_binnings = 0;
    wall_best = 0.;
    h.wall_cum = [0.; 5];
    wall_ccc = 0.;
    idim_optimal = 250000000;
    idim_higher = 490000000;
    h.num_hist_bins = 1000;
    h.edge_sample_frac = 0.5; // Fraction of edge pixels to use for median/MADN
    h.frac_for_hist = 0.98; // Top fraction of boxes to use for histogram (trim above)
    h.bottom_hist_frac = 0.005; // Bottom fraction of boxes for start of histogram
    h.fallback_pct_frac = 0.75; // Percentile for fallback "typical high SD value"
    h.fallback_struct_frac = 0.5; // Fraction of "typical high" to use like histo dip
    h.better_scale_crit = 0.33; // Difference of ratios must rise by this much to use
    h.elim_by_sd_type = 0.;
    h.elim_by_sd_crit = 0.5;
    nx_series = 0;
    nx_series_b = 0;
    h.bad_hist_peak_crit = 5.;
    h.sd_elim_max_frac = 0.95;
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "corrsearch3d",
        "ERROR: CORRSEARCH3D - ",
        true,
        3,
        2,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pip_input = num_opt_arg + num_non_opt_arg > 0;
    //
    // Open image files.
    //
    if pip_get_in_out_file(
        "ReferenceFile",
        1,
        "Image file to align to",
        &mut file_a,
        320,
    ) != 0
    {
        exit_error("No reference file specified");
    }
    if pip_get_in_out_file(
        "FileToAlign",
        2,
        "Image file being aligned",
        &mut file_b,
        320,
    ) != 0
    {
        exit_error("No file to align specified");
    }
    if pip_get_in_out_file(
        "OutputFile",
        3,
        "Output file for displacements",
        &mut output_file,
        320,
    ) != 0
    {
        exit_error("No output file for displacements specified");
    }

    if pip_input {
        ierr = pip_string("RegionModel", &mut model_file);
        ierr = pip_string("BRegionModel", &mut b_model);
    } else {
        print!(" Enter blank line : ");
        let _temp_file = read_line();
        println!(" Enter model file with contours enclosing areas to analyze, or Return for none");
        model_file = read_line();
    }
    //
    imopen(1, &file_a, "RO");
    // SAFETY: `irdhdr` writes three integers into each array and one value
    // into each scalar, all live locals.
    unsafe {
        irdhdr(
            1,
            h.nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
    }
    a_scale = 1.;
    if dmax - dmin > 0. {
        a_scale = 1.0e10;
    }
    if dmax - dmin > 1.0e-10 {
        a_scale = 100. / (dmax - dmin);
    }
    //
    imopen(2, &file_b, "RO");
    // SAFETY: as above.
    unsafe {
        irdhdr(
            2,
            nxyz2.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
    }
    b_scale = 1.;
    if dmax - dmin > 0. {
        b_scale = 1.0e10;
    }
    if dmax - dmin > 1.0e-10 {
        b_scale = 100. / (dmax - dmin);
    }
    let (nx, ny, nz) = (h.nxyz[0], h.nxyz[1], h.nxyz[2]);
    let (nx2, ny2, nz2) = (nxyz2[0], nxyz2[1], nxyz2[2]);
    // print *,'scaling factors', ascale, bscale
    //
    if pip_input {
        if pip_get_three_integers(
            b"PatchSizeXYZ",
            &mut h.nx_patch,
            &mut h.ny_patch,
            &mut h.nz_patch,
        ) != 0
        {
            exit_error("No patch size specified");
        }
        if pip_get_three_integers(
            b"NumberOfPatchesXYZ",
            &mut h.num_xpatch,
            &mut h.num_ypatch,
            &mut h.num_zpatch,
        ) != 0
        {
            exit_error("No number of patches specified");
        }
        if pip_get_two_integers(b"XMinAndMax", &mut ix_min, &mut ix_max)
            + pip_get_two_integers(b"YMinAndMax", &mut iy_min, &mut iy_max)
            + pip_get_two_integers(b"ZMinAndMax", &mut iz_min, &mut iz_max)
            != 0
        {
            exit_error("Min and max coordinates in X, Y, and Z must be entered");
        }
        j = 0;
        ierr = pip_get_boolean(b"InvertYLimits", &mut j);
        if j > 0 {
            i = ny + 1 - iy_min;
            iy_min = ny + 1 - iy_max;
            iy_max = i;
        }
        nbord_xlow = ix_min - 1;
        nbord_xhigh = nx - ix_max;
        nbord_ylow = iy_min - 1;
        nbord_yhigh = ny - iy_max;
        nbord_zlow = iz_min - 1;
        nbord_zhigh = nz - iz_max;
        h.nx_taper = (h.nx_patch + 9) / 10;
        h.ny_taper = (h.ny_patch + 9) / 10;
        h.nz_taper = (h.nz_patch + 9) / 10;
        //
        // If maximum shift is not entered, scale it by square root of largest size over
        // 1000. so it gets bigger for bigger sets
        ierr = pip_get_three_integers(
            b"TapersInXYZ",
            &mut h.nx_taper,
            &mut h.ny_taper,
            &mut h.nz_taper,
        );
        if pip_get_integer(b"MaximumShift", &mut max_shift) != 0 {
            let root = (nx2.max(ny2).max(nz2) as f32 / 1000.).sqrt();
            // `max(1., ...)`: `maxss`, finite here.
            max_shift = (max_shift as f32 * if 1. > root { 1. } else { root }) as i32;
        }
        get_nxyz(true, "BSourceOrSizeXYZ", " ", 5, &mut nxyz_bsource);
        ierr = pip_get_two_integers(
            b"BSourceBorderXLoHi",
            &mut nbord_source_xlow,
            &mut nbord_source_xhigh,
        );
        ierr = pip_get_two_integers(
            b"BSourceBorderYZLoHi",
            &mut nbord_source_zlow,
            &mut nbord_source_zhigh,
        );
        ierr = pip_string("BSourceTransform", &mut xf_file);
        {
            let (a, rest) = dxyz_vol.split_at_mut(1);
            let (b, c) = rest.split_at_mut(1);
            if_shift_in =
                1 - pip_get_three_floats(b"VolumeShiftXYZ", &mut a[0], &mut b[0], &mut c[0]);
        }
        {
            let (a, rest) = dxyz_init.split_at_mut(1);
            let (b, c) = rest.split_at_mut(1);
            ierr = pip_get_three_floats(b"InitialShiftXYZ", &mut a[0], &mut b[0], &mut c[0]);
        }
        ierr = pip_get_two_floats(b"LowPassRadiusSigma", &mut radius2, &mut sigma2);
        ierr = pip_get_float(b"HighPassSigma", &mut sigma1);
        ierr = pip_get_float(b"KernelSigma", &mut sigma_kernel);
        if sigma_kernel > size_switch {
            h.kernel_size = 5;
        }
        ierr = pip_get_integer(b"KernelSize", &mut h.kernel_size);
        if h.kernel_size != 3 && h.kernel_size != 5 {
            exit_error("Kernel size must be 3 or 5");
        }
        ierr = pip_get_logical("FlipYZMessages", &mut flip_messages);
        ierr = pip_get_integer(b"LocalSDNumBinnings", &mut h.num_lsd_binnings);
        if h.num_lsd_binnings > LIM_LOCAL_BIN as i32 {
            exit_error("Number of local binnings is too high");
        }
        if h.num_lsd_binnings > 0 {
            let [b0, b1, b2] = &mut h.lsd_box_size[0];
            if pip_get_three_integers(b"BoxSizeForLocalSD", b0, b1, b2) != 0 {
                exit_error("Box size must be entered for local SD analysis");
            }
            ierr = pip_get_two_floats(
                b"EliminateByLocalSD",
                &mut h.elim_by_sd_type,
                &mut h.elim_by_sd_crit,
            );
            if 2 * h.lsd_box_size[0][0] > h.nx_patch
                || 2 * h.lsd_box_size[0][1] > h.ny_patch
                || 2 * h.lsd_box_size[0][2] > h.nz_patch
            {
                let nxyz_patch = [h.nx_patch, h.ny_patch, h.nz_patch];
                for i in 0..3 {
                    h.lsd_box_size[0][i] = h.lsd_box_size[0][i].min(nxyz_patch[i] / 2);
                }
                println!(
                    "WARNING: Box size for SD analysis too large for patches, changed to{}{}{}",
                    fmt_i(h.lsd_box_size[0][0], 4),
                    fmt_i(h.lsd_box_size[0][1], 4),
                    fmt_i(h.lsd_box_size[0][2], 4)
                );
            }
        }

        if_axis = 1 - pip_get_float(b"AxisRotationAngle", &mut h.axis_rotation);
        ierr = 1 - pip_get_two_integers(b"TiltSeriesSizeXY", &mut nx_series, &mut ny_series);
        let ix = 1 - pip_get_two_integers(b"BTiltSeriesSizeXY", &mut nx_series_b, &mut ny_series_b);
        if if_axis == 0 && ix + ierr > 0 {
            exit_error("Axis rotation angle must be entered for tilt series size to be useful");
        }
        if nxyz_bsource[0] == 0 {
            nx_series_b = 0;
        }
        ierr = pip_get_integer(b"DebugMode", &mut h.if_debug);
    } else {
        print!(" X, Y, and Z size of patches: ");
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [
                ListItem::Integer(&mut h.nx_patch),
                ListItem::Integer(&mut h.ny_patch),
                ListItem::Integer(&mut h.nz_patch),
            ],
        ) {
            read_abort(err);
        }
        print!(" Number of patches in X, Y and Z: ");
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [
                ListItem::Integer(&mut h.num_xpatch),
                ListItem::Integer(&mut h.num_ypatch),
                ListItem::Integer(&mut h.num_zpatch),
            ],
        ) {
            read_abort(err);
        }
        print!(" Border sizes on lower and upper sides in X, Y, and Z: ");
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [
                ListItem::Integer(&mut nbord_xlow),
                ListItem::Integer(&mut nbord_xhigh),
                ListItem::Integer(&mut nbord_ylow),
                ListItem::Integer(&mut nbord_yhigh),
                ListItem::Integer(&mut nbord_zlow),
                ListItem::Integer(&mut nbord_zhigh),
            ],
        ) {
            read_abort(err);
        }
        print!(" Number of pixels to taper in X, Y, Z: ");
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [
                ListItem::Integer(&mut h.nx_taper),
                ListItem::Integer(&mut h.ny_taper),
                ListItem::Integer(&mut h.nz_taper),
            ],
        ) {
            read_abort(err);
        }
        print!(" Maximum shift to analyze for by searching: ");
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Integer(&mut max_shift)],
        ) {
            read_abort(err);
        }
        //
        // get inputs for analyzing existence of data in B file
        //
        println!(
            " Enter name or NX, NY, NZ of the untransformed source for the image file being \
             aligned, or Return for none"
        );
        get_nxyz(false, " ", " ", 5, &mut nxyz_bsource);
        println!(
            " Enter name of file used to transform the image file being aligned, or Return \
             for none"
        );
        xf_file = read_line();
        print!(
            " Border sizes on lower and upper sides in X and in Y or Z in the untransformed\n \
             source for the image file being aligned: "
        );
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [
                ListItem::Integer(&mut nbord_source_xlow),
                ListItem::Integer(&mut nbord_source_xhigh),
                ListItem::Integer(&mut nbord_source_zlow),
                ListItem::Integer(&mut nbord_source_zhigh),
            ],
        ) {
            read_abort(err);
        }
    }
    let _ = ierr;
    h.if_hist_verbose = 0;
    while h.if_debug >= 10 {
        h.if_hist_verbose += 1;
        h.if_debug -= 10;
    }
    let if_debug = h.if_debug;
    //
    // If there is no initial shift entered, enforce centered transform
    // by setting initial shift to half the difference in size
    //
    if if_shift_in == 0 {
        dxyz_vol[0] = (nxyz2[0] - h.nxyz[0]) as f32 / 2.;
        dxyz_vol[1] = (nxyz2[1] - h.nxyz[1]) as f32 / 2.;
        dxyz_vol[2] = (nxyz2[2] - h.nxyz[2]) as f32 / 2.;
    }
    let (dx_volume, dy_volume, dz_volume) = (dxyz_vol[0], dxyz_vol[1], dxyz_vol[2]);
    let (dx_initial, dy_initial, dz_initial) = (dxyz_init[0], dxyz_init[1], dxyz_init[2]);
    //
    // get padding for cross correlations
    //
    nx_xcbord = (h.nx_patch + 4) / 5;
    ny_xcbord = (h.ny_patch + 4) / 5;
    nz_xcbord = (h.nz_patch + 4) / 5;
    if pip_input {
        ierr = pip_get_three_integers(b"PadsInXYZ", &mut nx_xcbord, &mut ny_xcbord, &mut nz_xcbord);
    }
    nice_lim = nice_fft_limit();
    nx_xcpad = nice_frame(h.nx_patch + 2 * nx_xcbord, 2, nice_lim);
    ny_xcpad = nice_frame(h.ny_patch + 2 * ny_xcbord, 2, nice_lim);
    nz_xcpad = nice_frame(h.nz_patch + 2 * nz_xcbord, 2, nice_lim);
    lim_work = if using_fftw() == 0 {
        (nx_xcpad + 2) * nz_xcpad + 10
    } else {
        10
    };
    nx_ctf = nx_xcpad;
    //
    // Initialize transform: since this is a center-based transform,
    // neutralize component of volume coordinate shift due to size difference
    for i in 0..3 {
        for j in 0..3 {
            a_source[i][j] = 0.;
        }
        dxyz_source[i] = (nxyz2[i] - h.nxyz[i]) as f32 / 2. - dxyz_vol[i];
        a_source[i][i] = 1.;
    }
    //
    // require b source dimensions if one is entered or
    // fall back to current B volume size; read in transform and add shifts
    if nxyz_bsource[0] == 0 {
        if !xf_file.is_empty() {
            exit_error("B source file dimensions must be entered to use 3D transforms");
        }
        for i in 0..3 {
            nxyz_bsource[i] = nxyz2[i];
        }
    }
    //
    // Read transform, add volume-shift component of center-based transform
    if !xf_file.is_empty() {
        let file = dopen(1, &xf_file, "ro", "f");
        let mut unit = std::io::BufReader::new(file);
        for i in 0..3 {
            tmp = 0.;
            let [a0, a1, a2] = &mut a_source[i];
            if list_read(
                &mut unit,
                &mut [
                    ListItem::Real(a0),
                    ListItem::Real(a1),
                    ListItem::Real(a2),
                    ListItem::Real(&mut tmp),
                ],
            )
            .is_err()
            {
                // 95
                exit_error("Reading transform file");
            }
            dxyz_source[i] += tmp;
        }
    }
    //
    // get model contours as primary source of flip information and fallback
    // to Y/Z dimensions
    //
    num_cont = 0;
    if !model_file.is_empty() {
        get_contour_array_sizes(&mut fm, &model_file, 1, &mut lim_cont, &mut lim_vert);
        x_vertex = vec![0.0; lim_vert as usize];
        y_vertex = vec![0.0; lim_vert as usize];
        zcont = vec![0.0; lim_cont as usize];
        ind_vert = vec![0; lim_cont as usize];
        num_vertex = vec![0; lim_cont as usize];
        memory_error(0, "arrays for contours");
        get_region_contours(
            &mut fm,
            &model_file,
            "CORRSEARCH3D",
            &mut x_vertex,
            &mut y_vertex,
            &mut num_vertex,
            &mut ind_vert,
            &mut zcont,
            &mut num_cont,
            &mut h.if_flip,
            &mut lim_cont,
            &mut lim_vert,
            1,
        );

        if if_debug > 0 {
            let start = ind_vert[0];
            let values: Vec<f32> = (start..start + num_vertex[0])
                .flat_map(|i| [x_vertex[(i - 1) as usize], y_vertex[(i - 1) as usize]])
                .collect();
            print_f7_f8_pairs(&values);
        }
    } else {
        h.if_flip = 0;
        if nz > ny {
            h.if_flip = 1;
        }
    }
    //
    // now that flip is known, get patch starting positions and delta between
    // patches
    //
    ind_yb = 2;
    if h.if_flip != 0 && flip_messages {
        ind_yb = 3;
    }
    check_and_set_patches(
        nx,
        nbord_xlow,
        nbord_xhigh,
        h.nx_patch,
        &mut h.num_xpatch,
        &mut h.ix_start,
        &mut h.ix_delta,
        1,
    );
    check_and_set_patches(
        ny,
        nbord_ylow,
        nbord_yhigh,
        h.ny_patch,
        &mut h.num_ypatch,
        &mut h.iy_start,
        &mut h.iy_delta,
        ind_yb,
    );
    check_and_set_patches(
        nz,
        nbord_zlow,
        nbord_zhigh,
        h.nz_patch,
        &mut h.num_zpatch,
        &mut h.iz_start,
        &mut h.iz_delta,
        5 - ind_yb,
    );

    num_bcont = 0;
    if !b_model.is_empty() {
        //
        // For b model, get its coordinates in the source native section plane
        // then transform to coordinates in A volume native plane
        //
        println!(" Processing model on source for second volume");
        get_contour_array_sizes(&mut fm, &b_model, 2, &mut lim_cont, &mut lim_vert);
        x_vertex_b = vec![0.0; lim_vert as usize];
        y_vertex_b = vec![0.0; lim_vert as usize];
        zcont_b = vec![0.0; lim_cont as usize];
        ind_vert_b = vec![0; lim_cont as usize];
        num_vert_b = vec![0; lim_cont as usize];
        memory_error(0, "arrays for contours");
        get_region_contours(
            &mut fm,
            &b_model,
            "CORRSEARCH3D",
            &mut x_vertex_b,
            &mut y_vertex_b,
            &mut num_vert_b,
            &mut ind_vert_b,
            &mut zcont_b,
            &mut num_bcont,
            &mut if_flip_b,
            &mut lim_cont,
            &mut lim_vert,
            2,
        );
        for j in 1..=num_bcont {
            let ju = (j - 1) as usize;
            let (start, end) = (ind_vert_b[ju], ind_vert_b[ju] + num_vert_b[ju] - 1);
            if if_debug > 0 {
                let values: Vec<f32> = (start..=end)
                    .flat_map(|i| [x_vertex_b[(i - 1) as usize], y_vertex_b[(i - 1) as usize]])
                    .collect();
                print_f7_f8_pairs(&values);
            }
            for i in start..=end {
                let iu = (i - 1) as usize;
                let (xa, ya) = xform_bsource_to_a(
                    x_vertex_b[iu],
                    y_vertex_b[iu],
                    &nxyz_bsource,
                    &h.nxyz,
                    if_flip_b,
                    h.if_flip,
                    &a_source,
                    &dxyz_source,
                );
                x_vertex_b[iu] = xa;
                y_vertex_b[iu] = ya;
            }
            if if_debug > 0 {
                let values: Vec<f32> = (start..=end)
                    .flat_map(|i| [x_vertex_b[(i - 1) as usize], y_vertex_b[(i - 1) as usize]])
                    .collect();
                print_f7_f8_pairs(&values);
            }
        }
    } else {
        if_flip_b = 0;
        if nxyz_bsource[2] > nxyz_bsource[1] {
            if_flip_b = 1;
        }
    }
    ind_yb = 2;
    if if_flip_b != 0 {
        ind_yb = 3;
    }
    if if_debug > 0 {
        println!(
            "{}",
            ld_record(&[Ld::S("flips"), Ld::I(h.if_flip), Ld::I(if_flip_b)])
        );
    }
    //
    // compute transformed locations of corners of b source volume
    // 10/14/09: This used to be done inside the if below, but it is needed
    // for all evaluations of the patches.
    let bsource_corners = [
        (nbord_source_xlow as f32, nbord_source_zlow as f32),
        (
            (nxyz_bsource[0] - nbord_source_xhigh) as f32,
            nbord_source_zlow as f32,
        ),
        (
            (nxyz_bsource[0] - nbord_source_xhigh) as f32,
            (nxyz_bsource[(ind_yb - 1) as usize] - nbord_source_zhigh) as f32,
        ),
        (
            nbord_source_xlow as f32,
            (nxyz_bsource[(ind_yb - 1) as usize] - nbord_source_zhigh) as f32,
        ),
    ];
    for (k, (xb, yb)) in bsource_corners.iter().enumerate() {
        (xvert_bsource[k], yvert_bsource[k]) = xform_bsource_to_a(
            *xb,
            *yb,
            &nxyz_bsource,
            &h.nxyz,
            if_flip_b,
            h.if_flip,
            &a_source,
            &dxyz_source,
        );
    }
    if if_debug > 0 {
        let mut items = vec![Ld::S("bverts")];
        for i in 0..4 {
            items.push(Ld::R(xvert_bsource[i]));
            items.push(Ld::R(yvert_bsource[i]));
        }
        println!("{}", ld_record(&items));
    }
    if if_shift_in != 0 || !xf_file.is_empty() {
        //
        // now order the coordinates to find middle values that can be
        // used to adjust the entered lower and upper limits, so that the
        // basic grid can be set up to span between those limits
        //
        for i in 0..4 {
            xvert_sort[i] = xvert_bsource[i];
            yvert_sort[i] = yvert_bsource[i];
        }
        for i in 0..3 {
            for j in i + 1..4 {
                if xvert_sort[i] > xvert_sort[j] {
                    tmp = xvert_sort[i];
                    xvert_sort[i] = xvert_sort[j];
                    xvert_sort[j] = tmp;
                }
                if yvert_sort[i] > yvert_sort[j] {
                    tmp = yvert_sort[i];
                    yvert_sort[i] = yvert_sort[j];
                    yvert_sort[j] = tmp;
                }
            }
        }
        //
        // compute revised lower and upper limits for X and Y/Z
        //
        revise_patch_range(
            nx,
            nbord_xlow,
            nbord_xhigh,
            xvert_sort[1],
            xvert_sort[2],
            h.nx_patch,
            &mut h.num_xpatch,
            &mut h.ix_start,
            &mut h.ix_delta,
        );
        if h.if_flip != 0 {
            revise_patch_range(
                nz,
                nbord_zlow,
                nbord_zhigh,
                yvert_sort[1],
                yvert_sort[2],
                h.nz_patch,
                &mut h.num_zpatch,
                &mut h.iz_start,
                &mut h.iz_delta,
            );
        } else {
            revise_patch_range(
                ny,
                nbord_ylow,
                nbord_yhigh,
                yvert_sort[1],
                yvert_sort[2],
                h.ny_patch,
                &mut h.num_ypatch,
                &mut h.iy_start,
                &mut h.iy_delta,
            );
        }
    }
    h.lim_patch = h.num_xpatch * h.num_ypatch * h.num_zpatch + 10;
    let lim_patch = h.lim_patch as usize;
    let mut dx_patch = vec![0.0_f32; lim_patch];
    let mut dy_patch = vec![0.0_f32; lim_patch];
    let mut dz_patch = vec![0.0_f32; lim_patch];
    let mut direct_err = vec![0.0_f32; lim_patch];
    let mut if_done = vec![0_i32; lim_patch];
    let mut ix_sequence = vec![0_i32; lim_patch];
    let mut iy_sequence = vec![0_i32; lim_patch];
    let mut work = vec![0.0_f32; lim_work as usize];
    let mut iz_sequence = vec![0_i32; lim_patch];
    let mut idir_sequence = vec![0_i32; lim_patch];
    let mut dx_direct = vec![0.0_f32; lim_patch];
    let mut dy_direct = vec![0.0_f32; lim_patch];
    let mut dz_direct = vec![0.0_f32; lim_patch];
    let mut cc_direct = vec![0.0_f32; lim_patch];
    let mut corr_coef = vec![0.0_f32; lim_patch];
    memory_error(0, "arrays for patch variables");
    //
    if if_debug != 0 {
        println!(
            "{}",
            ld_record(&[
                Ld::S("Scan limits in X:"),
                Ld::I(h.ix_start),
                Ld::I(h.ix_start + (h.num_xpatch - 1) * h.ix_delta + h.nx_patch),
            ])
        );
        println!(
            "{}",
            ld_record(&[
                Ld::S("Scan limits in Y:"),
                Ld::I(h.iy_start),
                Ld::I(h.iy_start + (h.num_ypatch - 1) * h.iy_delta + h.ny_patch),
            ])
        );
        println!(
            "{}",
            ld_record(&[
                Ld::S("Scan limits in Z:"),
                Ld::I(h.iz_start),
                Ld::I(h.iz_start + (h.num_zpatch - 1) * h.iz_delta + h.nz_patch),
            ])
        );
    }
    //
    // Set up another set of boundaries if the tilt series size was entered
    ierr = 2;
    if h.if_flip > 0 {
        ierr = 3;
    }
    tilt_vertex_setup(
        &h,
        nx_series,
        ny_series,
        h.nxyz[0],
        h.nxyz[(ierr - 1) as usize],
        &mut xvert_tilt,
        &mut yvert_tilt,
    );
    tilt_vertex_setup(
        &h,
        nx_series_b,
        ny_series_b,
        nxyz_bsource[0],
        nxyz_bsource[(ind_yb - 1) as usize],
        &mut xvert_tilt_b,
        &mut yvert_tilt_b,
    );
    if nx_series_b > 0 {
        for i in 0..4 {
            (xvert_tilt_b[i], yvert_tilt_b[i]) = xform_bsource_to_a(
                xvert_tilt_b[i],
                yvert_tilt_b[i],
                &nxyz_bsource,
                &h.nxyz,
                if_flip_b,
                h.if_flip,
                &a_source,
                &dxyz_source,
            );
        }
        if if_debug > 1 {
            let mut line = String::from(" B to A");
            for ix in 0..4 {
                line.push_str(&fmt_f(xvert_tilt_b[ix] as f64, 9, 1));
                line.push_str(&fmt_f(yvert_tilt_b[ix] as f64, 9, 1));
            }
            println!("{line}");
        }
    }
    //
    // Do analysis of local SDs
    if h.num_lsd_binnings > 0 {
        analyze_local_sds(&mut h);
    }
    //
    // prescan for patches inside boundaries, to get total count and flags
    // for whether to do. Loop twice if doing SD analysis so that it can redo it without
    // if too many get eliminated
    //
    for _loop_elim in 1..=1.max(2.min(h.num_lsd_binnings + 1)) {
        num_pos_total = 0;
        h.num_elim_by_sd = 0;
        for iz in 1..=h.num_zpatch {
            zcen = (h.iz_start + (iz - 1) * h.iz_delta + h.nz_patch / 2) as f32;
            for iy in 1..=h.num_ypatch {
                ycen = (h.iy_start + (iy - 1) * h.iy_delta + h.ny_patch / 2) as f32;
                ymod_cen = ycen;
                zmod_cen = zcen;
                if h.if_flip != 0 {
                    zmod_cen = ycen;
                    ymod_cen = zcen;
                }
                //
                for ix in 1..=h.num_xpatch {
                    ind_p = ind_patch(&h, ix, iy, iz);
                    let ipu = (ind_p - 1) as usize;
                    xcen = (h.ix_start + (ix - 1) * h.ix_delta + h.nx_patch / 2) as f32;
                    if_use = 1;
                    //
                    // If SD analysis was done, eliminate first based on that
                    if h.num_lsd_binnings > 0 && nint(h.elim_by_sd_type) > 0 {
                        if (nint(h.elim_by_sd_type) > 1
                            && h.patch_mean_struct[ipu] < h.elim_by_sd_crit * h.high_sd_level)
                            || (nint(h.elim_by_sd_type) == 1
                                && h.patch_frac_high_sd[ipu] < h.elim_by_sd_crit)
                        {
                            if_use = 0;
                            h.num_elim_by_sd += 1;
                            if if_debug == 2 {
                                println!(
                                    "{}{}{} eliminated by SD criterion, frac{}",
                                    fmt_f(xcen as f64, 8, 1),
                                    fmt_f(ycen as f64, 8, 1),
                                    fmt_f(zcen as f64, 8, 1),
                                    fmt_f(h.patch_frac_high_sd[ipu] as f64, 7, 4)
                                );
                            }
                        }
                    }
                    //
                    // If A model was entered, find nearest contour in Z and see if
                    // patch is inside it
                    if num_cont > 0 && if_use == 1 {
                        icont_min = check_boundary_conts(
                            xcen,
                            ymod_cen,
                            zmod_cen,
                            &mut if_use,
                            num_cont,
                            &num_vertex,
                            &x_vertex,
                            &y_vertex,
                            &zcont,
                            &ind_vert,
                        );
                        if if_use == 0 && if_debug == 2 {
                            println!(
                                "{}",
                                ld_record(&[
                                    Ld::R(xcen),
                                    Ld::R(ymod_cen),
                                    Ld::S(" eliminated by A model contour"),
                                    Ld::I(icont_min),
                                ])
                            );
                        }
                    }
                    //
                    // Do the same if still ok and B model was entered
                    if num_bcont > 0 && if_use == 1 {
                        check_boundary_conts(
                            xcen,
                            ymod_cen,
                            zmod_cen,
                            &mut if_use,
                            num_bcont,
                            &num_vert_b,
                            &x_vertex_b,
                            &y_vertex_b,
                            &zcont_b,
                            &ind_vert_b,
                        );
                        if if_use == 0 && if_debug == 2 {
                            println!(
                                "{}",
                                ld_record(&[
                                    Ld::R(xcen),
                                    Ld::R(ymod_cen),
                                    Ld::S(" eliminated by B model"),
                                ])
                            );
                        }
                    }
                    //
                    if if_use == 1 {
                        if_use = 0;
                        //
                        // now make sure all corners of the patch are inside transformed
                        // area from B borders
                        //
                        xpatch_low = xcen - ((h.nx_patch - h.nx_taper) / 2) as f32;
                        xpatch_high = xcen + ((h.nx_patch - h.nx_taper) / 2) as f32;
                        if h.if_flip != 0 {
                            zpatch_low = zcen - ((h.nz_patch - h.nz_taper) / 2) as f32;
                            zpatch_high = zcen + ((h.nz_patch - h.nz_taper) / 2) as f32;
                        } else {
                            zpatch_low = ycen - ((h.ny_patch - h.ny_taper) / 2) as f32;
                            zpatch_high = ycen + ((h.ny_patch - h.ny_taper) / 2) as f32;
                        }
                        if inside(&xvert_bsource, &yvert_bsource, 4, xpatch_low, zpatch_low)
                            && inside(&xvert_bsource, &yvert_bsource, 4, xpatch_low, zpatch_high)
                            && inside(&xvert_bsource, &yvert_bsource, 4, xpatch_high, zpatch_low)
                            && inside(&xvert_bsource, &yvert_bsource, 4, xpatch_high, zpatch_high)
                        {
                            if_use = 1;
                        }
                        if if_use == 0 && if_debug == 2 {
                            println!(
                                "{}",
                                ld_record(&[
                                    Ld::R(xcen),
                                    Ld::R(ymod_cen),
                                    Ld::S(" eliminated by B boundaries"),
                                ])
                            );
                        }
                    }
                    if if_use == 1 && nx_series > 0 {
                        if !inside(&xvert_tilt, &yvert_tilt, 4, xcen, ymod_cen) {
                            if_use = 0;
                        }
                        if if_use == 0 && if_debug == 2 {
                            println!(
                                "{}",
                                ld_record(&[
                                    Ld::R(xcen),
                                    Ld::R(ymod_cen),
                                    Ld::S(" eliminated by A tilt series borders"),
                                ])
                            );
                        }
                    }
                    if if_use == 1 && nx_series_b > 0 {
                        if !inside(&xvert_tilt_b, &yvert_tilt_b, 4, xcen, ymod_cen) {
                            if_use = 0;
                        }
                        if if_use == 0 && if_debug == 2 {
                            println!(
                                "{}",
                                ld_record(&[
                                    Ld::R(xcen),
                                    Ld::R(ymod_cen),
                                    Ld::S(" eliminated by B tilt series borders"),
                                ])
                            );
                        }
                    }

                    if if_use == 0 {
                        if_done[ipu] = -1;
                    } else {
                        if_done[ipu] = 0;
                        num_pos_total += 1;
                    }
                    direct_err[ipu] = -1.;
                }
            }
        }
        //
        if h.num_lsd_binnings > 0 {
            i = h.num_xpatch * h.num_ypatch * h.num_zpatch;
            println!(
                "{} of{} total possible patches eliminated by SD criterion",
                fmt_i(h.num_elim_by_sd, 7),
                fmt_i(i, 7)
            );
            if (num_pos_total < 1 && h.num_elim_by_sd > 0)
                || (num_pos_total < 4 && h.num_elim_by_sd > i / 2)
                || h.num_elim_by_sd as f32 > h.sd_elim_max_frac * i as f32
            {
                h.elim_by_sd_type = 0.;
                println!(" That is too much elimination; trying again without SD criterion [CSD1]");
            } else {
                break;
            }
        }
    }
    if num_pos_total < 1 {
        exit_error("No patches fit within all of the constraints");
    }
    //
    // set indexes at which to load data and compose patches
    // Here is the padding for the B patch direct correlation
    //
    nx_pad = h.nx_patch + 2 * (max_shift + 1);
    ny_pad = h.ny_patch + 2 * (max_shift + 1);
    nz_pad = h.nz_patch + 2 * (max_shift + 1);
    num_patch_pixels = h.nx_patch * h.ny_patch * h.nz_patch;
    //
    // Patch A space is set by Fourier correlation padding, B space by
    // max of Fourier and direct padded volumes
    //
    ind_patch_a = 1;
    ind_patch_b = ind_patch_a + (nx_xcpad + 2) * ny_xcpad * nz_xcpad;
    h.ind_scratch = ind_patch_b;
    idim = (nx_pad * ny_pad * nz_pad).max((nx_xcpad + 2) * ny_xcpad * nz_xcpad);

    // Separate scratch for filtering only needed if filtering B also
    // if (sigmaKernel > 0.) indScratch = indPatchB + idim
    ind_load_a = h.ind_scratch + idim;
    //
    // get maximum load size: exload could be a separate parameter
    // First get it for the optimal memory usage; if that is not big enough,
    // then get it for a higher usage; but in any case make it big enough
    load_extra = 2 * (max_shift + 1);
    ny_load_ex = ny2.min(h.ny_patch + load_extra);
    nz_load_ex = nz2.min(h.nz_patch + load_extra);
    h.max_xload = (idim_optimal - ind_load_a - load_extra * ny_load_ex * nz_load_ex)
        / (h.ny_patch * h.nz_patch + ny_load_ex * nz_load_ex);
    if h.max_xload < h.nx_patch {
        h.max_xload = (idim_higher - ind_load_a - load_extra * ny_load_ex * nz_load_ex)
            / (h.ny_patch * h.nz_patch + ny_load_ex * nz_load_ex);
        h.max_xload = (h.nx_patch + 4).max(h.max_xload);
    }
    h.max_xload = (nx + 2).min(h.max_xload);
    idim = ind_load_a
        + h.max_xload * h.ny_patch * h.nz_patch
        + (nx2 + 2).min(h.max_xload + load_extra) * ny_load_ex * nz_load_ex
        + 10;
    ind_load_b = ind_load_a + h.max_xload * h.ny_patch * h.nz_patch;
    if if_debug > 0 {
        println!(
            "{}",
            ld_record(&[
                Ld::I(num_patch_pixels),
                Ld::I(ind_patch_a),
                Ld::I(ind_patch_b),
                Ld::I(ind_load_a),
                Ld::I(ind_load_b),
                Ld::I(h.max_xload),
            ])
        );
    }
    h.buffer = vec![0.0_f32; idim as usize];
    memory_error(0, "image buffer");
    //
    let mut unit1 = dopen(1, &output_file, "new", "f");
    let _ = unit1.write_all(
        format!(
            "{} positions{}\n",
            fmt_i(num_pos_total, 7),
            fmt_i(id_ccc_col, 3)
        )
        .as_bytes(),
    );

    // Set up number of threads based on analysis relative to 40x20x40 patches
    if h.kernel_size == 3 {
        h.num_smooth_threads =
            1.max(6.min(nint(2. * (num_patch_pixels as f32 / 32000.).powf(0.4))));
    } else {
        h.num_smooth_threads =
            1.max(8.min(nint(4. * (num_patch_pixels as f32 / 32000.).powf(0.6))));
    }
    if if_debug > 0 {
        println!(
            "{}",
            ld_record(&[Ld::S("smooth threads"), Ld::I(h.num_smooth_threads)])
        );
    }
    h.num_smooth_threads = num_omp_threads(h.num_smooth_threads);
    num3_corr_threads = 1.max(8.min(nint(2. * (num_patch_pixels as f32 / 32000.).powf(0.45))));
    if if_debug > 0 {
        println!(
            "{}",
            ld_record(&[Ld::S("search threads"), Ld::I(num3_corr_threads)])
        );
    }
    num3_corr_threads = num_omp_threads(num3_corr_threads);
    num_ccc_threads = 1.max(8.min(nint(4. * (num_patch_pixels as f32 / 32000.).powf(0.6))));
    if if_debug > 0 {
        println!(
            "{}",
            ld_record(&[Ld::S("CCC threads"), Ld::I(num_ccc_threads)])
        );
    }
    num_ccc_threads = num_omp_threads(num_ccc_threads);
    if if_debug > 0 {
        println!(
            "{}",
            ld_record(&[
                Ld::S("Actual threads:"),
                Ld::I(h.num_smooth_threads),
                Ld::I(num3_corr_threads),
                Ld::I(num_ccc_threads),
            ])
        );
    }
    //
    // loop from center out in all directions, X inner, then short dimension,
    // then long dimension of Y and Z
    //
    if h.if_flip != 0 {
        sequence_patches(
            h.num_xpatch,
            h.num_ypatch,
            h.num_zpatch,
            &mut ix_sequence,
            &mut iy_sequence,
            &mut iz_sequence,
            &mut idir_sequence,
            &mut num_sequence,
        );
    } else {
        sequence_patches(
            h.num_xpatch,
            h.num_zpatch,
            h.num_ypatch,
            &mut ix_sequence,
            &mut iz_sequence,
            &mut iy_sequence,
            &mut idir_sequence,
            &mut num_sequence,
        );
    }

    load_x1 = -1;
    load_xb1 = -1;
    num_skip = 0;
    for ind_sequence in 1..=num_sequence {
        let isu = (ind_sequence - 1) as usize;
        ix_patch = ix_sequence[isu];
        iy_patch = iy_sequence[isu];
        iz_patch = iz_sequence[isu];
        h.ix_dir = idir_sequence[isu];
        h.load_full_width = ix_patch == ix_sequence[0];
        iz0 = h.iz_start + (iz_patch - 1) * h.iz_delta;
        iz1 = iz0 + h.nz_patch - 1;
        iz_cen = iz0 + h.nz_patch / 2;
        iy0 = h.iy_start + (iy_patch - 1) * h.iy_delta;
        iy1 = iy0 + h.ny_patch - 1;
        iy_cen = iy0 + h.ny_patch / 2;
        ix0 = h.ix_start + (ix_patch - 1) * h.ix_delta;
        ix1 = ix0 + h.nx_patch - 1;
        ix_cen = ix0 + h.nx_patch / 2;
        // print *,'doing', ixcen, iycen, izcen
        ind_p = ind_patch(&h, ix_patch, iy_patch, iz_patch);
        let ipu = (ind_p - 1) as usize;
        if if_debug == 2 && if_done[ipu] == 0 {
            let _ = unit1.write_all(format_105(ix_cen, iy_cen, iz_cen, &[2., 2., 2.]).as_bytes());
            if_done[ipu] = 1;
        }

        if if_done[ipu] == 0 {
            //
            // find and average adjacent and nearby patches, weighting by the
            // correlation coefficient
            //
            num_adjacent = 0;
            num_near = 0;
            num_adjacent_look = (dist_near + 1.) as i32;
            dx_sum = 0.;
            dy_sum = 0.;
            dz_sum = 0.;
            dx_sum_near = 0.;
            dy_sum_near = 0.;
            dz_sum_near = 0.;
            wsum_adjacent = 0.;
            wsum_near = 0.;
            for ix in -num_adjacent_look..=num_adjacent_look {
                for iy in -num_adjacent_look..=num_adjacent_look {
                    for iz in -num_adjacent_look..=num_adjacent_look {
                        dist_sq = (ix * ix + iy * iy + iz * iz) as f32;
                        if dist_sq > 0. && dist_sq <= dist_near * dist_near {
                            ix_adjacent = ix_patch + ix;
                            iy_adjacent = iy_patch + iy;
                            iz_adjacent = iz_patch + iz;
                            if ix_adjacent > 0
                                && ix_adjacent <= h.num_xpatch
                                && iy_adjacent > 0
                                && iy_adjacent <= h.num_ypatch
                                && iz_adjacent > 0
                                && iz_adjacent <= h.num_zpatch
                            {
                                ind_a = ind_patch(&h, ix_adjacent, iy_adjacent, iz_adjacent);
                                let iau = (ind_a - 1) as usize;
                                if if_done[iau] > 0 {
                                    num_near += 1;
                                    // `max(0.01, corrCoef(indA))` (`corrsearch3d.f90:775`):
                                    // `maxss corrCoef, 0.01` in the reference object,
                                    // so a NaN coefficient (NaN patch data) weighs 0.01.
                                    wcc = maxss(corr_coef[iau], 0.01);
                                    dx_sum_near += dx_patch[iau] * wcc;
                                    dy_sum_near += dy_patch[iau] * wcc;
                                    dz_sum_near += dz_patch[iau] * wcc;
                                    wsum_near += wcc;
                                    if dist_sq <= dist_adjacent * dist_adjacent {
                                        num_adjacent += 1;
                                        dx_sum += dx_patch[iau] * wcc;
                                        dy_sum += dy_patch[iau] * wcc;
                                        dz_sum += dz_patch[iau] * wcc;
                                        wsum_adjacent += wcc;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // nadj=max(1, nadj)
            // `max(.01, wsumAdjacent)` / `max(.01, wsumNear)`
            // (`corrsearch3d.f90:794, 797`): the reference object emits
            // `maxss wsumAdjacent, .01` but `maxss .01, wsumNear`.
            let wadj = maxss(wsum_adjacent, 0.01);
            dx_adjacent = dx_sum / wadj;
            dy_adjacent = dy_sum / wadj;
            dz_adjacent = dz_sum / wadj;
            let wnear = maxss(0.01, wsum_near);
            dx_near = dx_sum_near / wnear;
            dy_near = dy_sum_near / wnear;
            dz_near = dz_sum_near / wnear;
            //
            // Do direct search if there is something adjacent and its correlation
            // coefficients are good enough and it is not very deviant
            //
            if num_adjacent > 0
                && wsum_adjacent / num_adjacent as f32 > cc_ratio * wsum_near / num_near as f32
                && (dx_near - dx_adjacent).abs() < max_shift as f32
                && (dy_near - dy_adjacent).abs() < max_shift as f32
                && (dz_near - dz_adjacent).abs() < max_shift as f32
            {
                set_bload(
                    &mut ix0,
                    &mut ix1,
                    nx2,
                    dx_adjacent,
                    dx_volume,
                    &mut ix_b0,
                    &mut ix_b1,
                );
                set_bload(
                    &mut iy0,
                    &mut iy1,
                    ny2,
                    dy_adjacent,
                    dy_volume,
                    &mut iy_b0,
                    &mut iy_b1,
                );
                set_bload(
                    &mut iz0,
                    &mut iz1,
                    nz2,
                    dz_adjacent,
                    dz_volume,
                    &mut iz_b0,
                    &mut iz_b1,
                );
                if ix1 > ix0
                    && iy1 > iy0
                    && iz1 > iz0
                    && (ix1 + 1 - ix0) * (iy1 + 1 - iy0) * (iz1 + 1 - iz0) >= num_patch_pixels / 2
                {
                    //
                    // revise parameters based on load
                    //
                    h.nx_patch_tmp = ix1 + 1 - ix0;
                    h.ny_patch_tmp = iy1 + 1 - iy0;
                    h.nz_patch_tmp = iz1 + 1 - iz0;
                    nx_pad = h.nx_patch_tmp + 2 * (max_shift + 1);
                    ny_pad = h.ny_patch_tmp + 2 * (max_shift + 1);
                    nz_pad = h.nz_patch_tmp + 2 * (max_shift + 1);
                    // print *,'direct', ixcen, iycen, izcen, nxptmp, nyptmp, nzptmp
                    let _ = std::io::stdout().flush();
                    //
                    // get the a patch from the loaded data into an exact fit,
                    // taper, pad, kernel filter optionally and set to zero mean
                    //
                    let (nxpt, nypt, nzpt) = (h.nx_patch_tmp, h.ny_patch_tmp, h.nz_patch_tmp);
                    load_extract_process(
                        &mut h,
                        1,
                        a_scale,
                        ind_load_a,
                        ind_patch_a,
                        ix0,
                        ix1,
                        iy0,
                        iy1,
                        iz0,
                        iz1,
                        0,
                        [nx, ny, nz],
                        &mut load_x0,
                        &mut load_x1,
                        &mut nx_load,
                        &mut load_y0,
                        &mut load_y1,
                        &mut ny_load,
                        &mut load_z0,
                        &mut load_z1,
                        &mut nz_load,
                        nxpt,
                        nxpt,
                        nypt,
                        nzpt,
                        sigma_kernel,
                    );
                    //
                    // get the b patch from the loaded data padded to maximum shift
                    // We only filter one patch because that is all that is needed to
                    // smooth the CCF in which we are extracting a peak; there, this
                    // smoothing is equivalent to smoothing both by half as much.  The
                    // CCC would be better if both were smoothed equivalently, but this
                    // is not worth it because the B patch takes much longer to smooth
                    load_extract_process(
                        &mut h,
                        2,
                        b_scale,
                        ind_load_b,
                        ind_patch_b,
                        ix_b0,
                        ix_b1,
                        iy_b0,
                        iy_b1,
                        iz_b0,
                        iz_b1,
                        load_extra,
                        nxyz2,
                        &mut load_xb0,
                        &mut load_xb1,
                        &mut nx_load_b,
                        &mut load_yb0,
                        &mut load_yb1,
                        &mut ny_load_b,
                        &mut load_zb0,
                        &mut load_zb1,
                        &mut nz_load_b,
                        nx_pad,
                        nx_pad,
                        ny_pad,
                        nz_pad,
                        0.,
                    );
                    //
                    dx_new = 0.;
                    dy_new = 0.;
                    dz_new = 0.;
                    h.wall_start = wall_time();
                    let a0 = (ind_patch_a - 1) as usize;
                    let b0 = (ind_patch_b - 1) as usize;
                    find_best_corr(
                        &h.buffer[a0..],
                        nxpt,
                        nypt,
                        nzpt,
                        ix0,
                        iy0,
                        iz0,
                        &h.buffer[b0..],
                        nx_pad,
                        ny_pad,
                        nz_pad,
                        ix0 - max_shift - 1,
                        iy0 - max_shift - 1,
                        iz0 - max_shift - 1,
                        ix0,
                        ix1,
                        iy0,
                        iy1,
                        iz0,
                        iz1,
                        &mut dx_new,
                        &mut dy_new,
                        &mut dz_new,
                        max_shift,
                        &mut found,
                        &mut num_corrs,
                        num3_corr_threads,
                    );
                    wall_best += wall_time() - h.wall_start;
                    if found {
                        if_done[ipu] = 1;
                        dx_patch[ipu] = dx_new + nint(dx_adjacent) as f32;
                        dy_patch[ipu] = dy_new + nint(dy_adjacent) as f32;
                        dz_patch[ipu] = dz_new + nint(dz_adjacent) as f32;
                        dx_direct[ipu] = dx_patch[ipu];
                        dy_direct[ipu] = dy_patch[ipu];
                        dz_direct[ipu] = dz_patch[ipu];
                        //
                        // compute correlation coefficient
                        //
                        h.wall_start = wall_time();
                        one_corr_coeff(
                            &h.buffer[a0..],
                            nxpt,
                            nypt,
                            nzpt,
                            &h.buffer[b0..],
                            nx_pad,
                            ny_pad,
                            nz_pad,
                            nxpt,
                            nypt,
                            nzpt,
                            dx_new,
                            dy_new,
                            dz_new,
                            &mut corr_coef[ipu],
                            num_ccc_threads,
                        );
                        cc_direct[ipu] = corr_coef[ipu];
                        wall_ccc += wall_time() - h.wall_start;
                    }
                } else {
                    if_done[ipu] = -1;
                    num_skip += 1;
                }
            }
            //
            // If there are no adjacent patches, or something was fishy,
            // do a full cross corr
            //
            if if_done[ipu] == 0 || if_debug == 3 {
                if load_x1 < 0 && load_xb1 < 0 {
                    dx_near = dx_initial;
                    dy_near = dy_initial;
                    dz_near = dz_initial;
                }
                set_bload(
                    &mut ix0, &mut ix1, nx2, dx_near, dx_volume, &mut ix_b0, &mut ix_b1,
                );
                set_bload(
                    &mut iy0, &mut iy1, ny2, dy_near, dy_volume, &mut iy_b0, &mut iy_b1,
                );
                set_bload(
                    &mut iz0, &mut iz1, nz2, dz_near, dz_volume, &mut iz_b0, &mut iz_b1,
                );
                if ix1 > ix0
                    && iy1 > iy0
                    && iz1 > iz0
                    && (ix1 + 1 - ix0) * (iy1 + 1 - iy0) * (iz1 + 1 - iz0) >= num_patch_pixels / 2
                {
                    //
                    // Adjust load sizes and make new ctf if needed
                    //
                    h.nx_patch_tmp = ix1 + 1 - ix0;
                    h.ny_patch_tmp = iy1 + 1 - iy0;
                    h.nz_patch_tmp = iz1 + 1 - iz0;
                    nx_xcpad = nice_frame(h.nx_patch_tmp + 2 * nx_xcbord, 2, nice_lim);
                    ny_xcpad = nice_frame(h.ny_patch_tmp + 2 * ny_xcbord, 2, nice_lim);
                    nz_xcpad = nice_frame(h.nz_patch_tmp + 2 * nz_xcbord, 2, nice_lim);
                    // print *,'XC', nxptmp, nyptmp, nzptmp, nxXCpad, nyXCpad, nzXCpad, &
                    // ix0, ix1, iy0, iy1, iz0, iz1, ixb0, ixb1, iyb0, iyb1, izb0, izb1
                    let _ = std::io::stdout().flush();
                    if (radius2 > 0. || sigma1 != 0. || sigma2 != 0.)
                        && (nx_ctf != nx_xcpad || ny_ctf != ny_xcpad || nz_ctf != nz_xcpad)
                    {
                        xcorr_set_ctf(
                            sigma1,
                            sigma2,
                            0.,
                            radius2,
                            &mut ctf,
                            nx_xcpad,
                            ny_xcpad.max(nz_xcpad),
                            &mut delta,
                        );
                        nx_ctf = nx_xcpad;
                        ny_ctf = ny_xcpad;
                        nz_ctf = nz_xcpad;
                    }

                    //
                    // get the both patches from the loaded data padded for XCorr
                    //
                    load_extract_process(
                        &mut h,
                        1,
                        a_scale,
                        ind_load_a,
                        ind_patch_a,
                        ix0,
                        ix1,
                        iy0,
                        iy1,
                        iz0,
                        iz1,
                        0,
                        [nx, ny, nz],
                        &mut load_x0,
                        &mut load_x1,
                        &mut nx_load,
                        &mut load_y0,
                        &mut load_y1,
                        &mut ny_load,
                        &mut load_z0,
                        &mut load_z1,
                        &mut nz_load,
                        nx_xcpad + 2,
                        nx_xcpad,
                        ny_xcpad,
                        nz_xcpad,
                        sigma_kernel,
                    );
                    // call dumpVolume(buf(indpatcha), nxXCpad + 2, nxXCpad, &
                    // nyXCpad, nzXCpad, 'dumpa.')
                    load_extract_process(
                        &mut h,
                        2,
                        b_scale,
                        ind_load_b,
                        ind_patch_b,
                        ix_b0,
                        ix_b1,
                        iy_b0,
                        iy_b1,
                        iz_b0,
                        iz_b1,
                        load_extra,
                        nxyz2,
                        &mut load_xb0,
                        &mut load_xb1,
                        &mut nx_load_b,
                        &mut load_yb0,
                        &mut load_yb1,
                        &mut ny_load_b,
                        &mut load_zb0,
                        &mut load_zb1,
                        &mut nz_load_b,
                        nx_xcpad + 2,
                        nx_xcpad,
                        ny_xcpad,
                        nz_xcpad,
                        0.,
                    );

                    let a0 = (ind_patch_a - 1) as usize;
                    let b0 = (ind_patch_b - 1) as usize;
                    {
                        let (lo, hi) = h.buffer.split_at_mut(b0);
                        fourier_corr(
                            &mut lo[a0..],
                            hi,
                            (nx_xcpad + 2) / 2,
                            ny_xcpad,
                            nz_xcpad,
                            &mut work,
                            &ctf,
                            delta,
                        );
                    }

                    find_xcorr_peak(
                        &h.buffer[a0..],
                        nx_xcpad + 2,
                        ny_xcpad,
                        nz_xcpad,
                        &mut dx_new,
                        &mut dy_new,
                        &mut dz_new,
                        &mut peak,
                    );
                    dx_patch[ipu] = dx_new + nint(dx_near) as f32;
                    dy_patch[ipu] = dy_new + nint(dy_near) as f32;
                    dz_patch[ipu] = dz_new + nint(dz_near) as f32;
                    //
                    // For coef, reload the patches without the 2-pixel X padding
                    //
                    load_extract_process(
                        &mut h,
                        1,
                        a_scale,
                        ind_load_a,
                        ind_patch_a,
                        ix0,
                        ix1,
                        iy0,
                        iy1,
                        iz0,
                        iz1,
                        0,
                        [nx, ny, nz],
                        &mut load_x0,
                        &mut load_x1,
                        &mut nx_load,
                        &mut load_y0,
                        &mut load_y1,
                        &mut ny_load,
                        &mut load_z0,
                        &mut load_z1,
                        &mut nz_load,
                        nx_xcpad,
                        nx_xcpad,
                        ny_xcpad,
                        nz_xcpad,
                        sigma_kernel,
                    );
                    load_extract_process(
                        &mut h,
                        2,
                        b_scale,
                        ind_load_b,
                        ind_patch_b,
                        ix_b0,
                        ix_b1,
                        iy_b0,
                        iy_b1,
                        iz_b0,
                        iz_b1,
                        load_extra,
                        nxyz2,
                        &mut load_xb0,
                        &mut load_xb1,
                        &mut nx_load_b,
                        &mut load_yb0,
                        &mut load_yb1,
                        &mut ny_load_b,
                        &mut load_zb0,
                        &mut load_zb1,
                        &mut nz_load_b,
                        nx_xcpad,
                        nx_xcpad,
                        ny_xcpad,
                        nz_xcpad,
                        0.,
                    );
                    one_corr_coeff(
                        &h.buffer[a0..],
                        nx_xcpad,
                        ny_xcpad,
                        nz_xcpad,
                        &h.buffer[b0..],
                        nx_xcpad,
                        ny_xcpad,
                        nz_xcpad,
                        h.nx_patch_tmp,
                        h.ny_patch_tmp,
                        h.nz_patch_tmp,
                        dx_new,
                        dy_new,
                        dz_new,
                        &mut corr_coef[ipu],
                        num_ccc_threads,
                    );

                    if if_done[ipu] == 0 {
                        num_fourier_patch += 1;
                    }
                    if if_debug == 3 && if_done[ipu] > 0 {
                        let ex = dx_direct[ipu] - dx_patch[ipu];
                        let ey = dy_direct[ipu] - dy_patch[ipu];
                        let ez = dz_direct[ipu] - dz_patch[ipu];
                        direct_err[ipu] = (ex * ex + ey * ey + ez * ez).sqrt();
                    }
                    if_done[ipu] = 1;
                } else if if_done[ipu] == 0 {
                    if_done[ipu] = -1;
                    num_skip += 1;
                }
            }
            if if_done[ipu] > 0 {
                let _ = unit1.write_all(
                    format_105(
                        ix_cen,
                        iy_cen,
                        iz_cen,
                        &[dx_patch[ipu], dy_patch[ipu], dz_patch[ipu], corr_coef[ipu]],
                    )
                    .as_bytes(),
                );
                let _ = unit1.flush();
            }
        }
    }
    //
    // If any ones were skipped, need to rewrite the file
    //
    if (num_skip > 0 || h.num_lsd_binnings > 0 || if_debug == 3) && if_debug != 2 {
        let _ = unit1.seek(SeekFrom::Start(0));
        let mut text = String::new();
        if h.num_lsd_binnings > 0 {
            text.push_str(&format!(
                "{} positions{}{}{}\n",
                fmt_i(num_pos_total - num_skip, 7),
                fmt_i(id_ccc_col, 3),
                fmt_i(id_frac_col, 3),
                fmt_i(id_struct_col, 3)
            ));
        } else {
            text.push_str(&format!(
                "{} positions{}\n",
                fmt_i(num_pos_total - num_skip, 7),
                fmt_i(id_ccc_col, 3)
            ));
        }
        for ind_sequence in 1..=num_sequence {
            let isu = (ind_sequence - 1) as usize;
            ix_patch = ix_sequence[isu];
            iy_patch = iy_sequence[isu];
            iz_patch = iz_sequence[isu];
            ind_p = ind_patch(&h, ix_patch, iy_patch, iz_patch);
            let ipu = (ind_p - 1) as usize;
            if if_done[ipu] > 0 {
                iz_cen = h.iz_start + (iz_patch - 1) * h.iz_delta + h.nz_patch / 2;
                iy_cen = h.iy_start + (iy_patch - 1) * h.iy_delta + h.ny_patch / 2;
                ix_cen = h.ix_start + (ix_patch - 1) * h.ix_delta + h.nx_patch / 2;
                if if_debug == 3 {
                    if direct_err[ipu] < 0. {
                        text.push_str(&format_105(
                            ix_cen,
                            iy_cen,
                            iz_cen,
                            &[dx_patch[ipu], dy_patch[ipu], dz_patch[ipu], corr_coef[ipu]],
                        ));
                    } else {
                        text.push_str(&format_105(
                            ix_cen,
                            iy_cen,
                            iz_cen,
                            &[
                                dx_patch[ipu],
                                dy_patch[ipu],
                                dz_patch[ipu],
                                corr_coef[ipu],
                                dx_direct[ipu],
                                dy_direct[ipu],
                                dz_direct[ipu],
                                cc_direct[ipu],
                                direct_err[ipu],
                            ],
                        ));
                    }
                } else if h.num_lsd_binnings > 0 {
                    // `115 format(3i6,3f9.2,f10.4,f8.4,f10.5)`
                    text.push_str(&format!(
                        "{}{}{}{}{}{}{}{}{}\n",
                        fmt_i(ix_cen, 6),
                        fmt_i(iy_cen, 6),
                        fmt_i(iz_cen, 6),
                        fmt_f(dx_patch[ipu] as f64, 9, 2),
                        fmt_f(dy_patch[ipu] as f64, 9, 2),
                        fmt_f(dz_patch[ipu] as f64, 9, 2),
                        fmt_f(corr_coef[ipu] as f64, 10, 4),
                        fmt_f(h.patch_frac_high_sd[ipu] as f64, 8, 4),
                        fmt_f((h.patch_mean_struct[ipu] / h.high_sd_level) as f64, 10, 5)
                    ));
                } else {
                    text.push_str(&format_105(
                        ix_cen,
                        iy_cen,
                        iz_cen,
                        &[dx_patch[ipu], dy_patch[ipu], dz_patch[ipu], corr_coef[ipu]],
                    ));
                }
            }
        }
        // A sequential `WRITE` after `REWIND` makes its record the last one of
        // the file, so what the first pass wrote beyond it is gone.
        let _ = unit1.write_all(text.as_bytes());
        let _ = unit1.set_len(text.len() as u64);
    }
    drop(unit1);
    per_pos = (3. * num_corrs as f32) / 1.max(num_pos_total - num_fourier_patch) as f32;
    println!(
        "{} correlations per position, Fourier correlations computed{} times",
        fmt_f(per_pos as f64, 8, 2),
        fmt_i(num_fourier_patch, 5)
    );
    if if_debug > 0 {
        let mut line = String::from("LETKZ, search, oneCCC:");
        for value in h.wall_cum.iter().chain([wall_best, wall_ccc].iter()) {
            line.push_str(&fmt_f(*value, 8, 4));
        }
        println!("{line}");
    }
    let _ = (peak, lim_work, dmean, mode, iz_cen);
    exit(0);
}

/// `write(*,'(5(f7.0,f8.0))')` of an `(x, y)` vertex list: five pairs per
/// record by format reversion (`corrsearch3d.f90:391`, `:425`, `:432`).
fn print_f7_f8_pairs(values: &[f32]) {
    let mut line = String::new();
    for (k, value) in values.iter().enumerate() {
        if k > 0 && k % 10 == 0 {
            println!("{line}");
            line.clear();
        }
        let w = if k % 2 == 0 { 7 } else { 8 };
        line.push_str(&fmt_f(*value as f64, w, 0));
    }
    println!("{line}");
}

/// Original contained subroutine `tiltVertexSetup` (`corrsearch3d.f90:1042`).
///
/// tiltVertexSetup constructs a bounding box from the original data and
/// rotates it by the negative of the tilt axis angle.  In the data set that
/// this was first tried with, the good area appeared to go about 60 pixels
/// BEYOND the borders of this area.  Even if the good data stops nearer to
/// this boundary, it should be OK to let the patch centers go out to these
/// borders.  Thus this just has xbord and ybord zero.
fn tilt_vertex_setup(
    h: &MainVars,
    nx_tilt: i32,
    ny_tilt: i32,
    nx_rec: i32,
    ny_rec: i32,
    xvert: &mut [f32; 4],
    yvert: &mut [f32; 4],
) {
    let mut rot_mat = [0.0_f32; 6];
    if nx_tilt == 0 || nx_rec == 0 {
        return;
    }
    let xbord = 0.0_f32;
    let ybord = 0.0_f32;
    xvert[0] = nx_rec as f32 / 2. - (nx_tilt as f32 - xbord) / 2.;
    yvert[0] = ny_rec as f32 / 2. - (ny_tilt as f32 - ybord) / 2.;
    xvert[1] = nx_rec as f32 / 2. + (nx_tilt as f32 - xbord) / 2.;
    yvert[1] = yvert[0];
    xvert[2] = xvert[1];
    yvert[2] = ny_rec as f32 / 2. + (ny_tilt as f32 - ybord) / 2.;
    xvert[3] = xvert[0];
    yvert[3] = yvert[2];
    // `rotMat(2,3)`, column major.
    rot_mat[0] = gfortran_cosd_r4(h.axis_rotation);
    rot_mat[2] = gfortran_sind_r4(h.axis_rotation);
    rot_mat[1] = -rot_mat[2];
    rot_mat[3] = rot_mat[0];
    rot_mat[4] = 0.;
    rot_mat[5] = 0.;
    if h.if_debug > 0 {
        let mut line = String::from("raw verts");
        for ix in 0..4 {
            line.push_str(&fmt_f(xvert[ix] as f64, 9, 1));
            line.push_str(&fmt_f(yvert[ix] as f64, 9, 1));
        }
        println!("{line}");
    }
    for ix in 0..4 {
        (xvert[ix], yvert[ix]) = xfapply(
            &rot_mat,
            nx_rec as f32 / 2.,
            ny_rec as f32 / 2.,
            xvert[ix],
            yvert[ix],
        );
    }
    if h.if_debug > 0 {
        let mut line = String::from("tilt verts");
        for ix in 0..4 {
            line.push_str(&fmt_f(xvert[ix] as f64, 9, 1));
            line.push_str(&fmt_f(yvert[ix] as f64, 9, 1));
        }
        println!("{line}");
    }
}

/// Original contained subroutine `loadExtractProcess`
/// (`corrsearch3d.f90:1077`).
///
/// loadExtractProcess takes care of loading data as needed from the
/// given unit IUNIT into the right area of BUFFER, extracting the desired
/// patch from there, tapering it, and smoothing via a scratch patch
/// area if specified
#[allow(clippy::too_many_arguments)]
fn load_extract_process(
    h: &mut MainVars,
    iunit: i32,
    scale: f32,
    ind_load_a: i32,
    ind_patch_a: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    iz0: i32,
    iz1: i32,
    load_extra: i32,
    nxyz: [i32; 3],
    load_x0: &mut i32,
    load_x1: &mut i32,
    nx_load: &mut i32,
    load_y0: &mut i32,
    load_y1: &mut i32,
    ny_load: &mut i32,
    load_z0: &mut i32,
    load_z1: &mut i32,
    nz_load: &mut i32,
    nx_pad_dim: i32,
    nx_pad: i32,
    ny_pad: i32,
    nz_pad: i32,
    sigma_kernel: f32,
) {
    let mut wall_now: f64;
    let l0 = (ind_load_a - 1) as usize;
    let a0 = (ind_patch_a - 1) as usize;

    if h.if_debug > 0 {
        h.wall_start = wall_time();
    }
    manage_load(
        iunit,
        &mut h.buffer[l0..],
        ix0,
        ix1,
        iy0,
        iy1,
        iz0,
        iz1,
        load_extra / 2,
        h.ix_dir,
        h.ix_delta,
        h.max_xload,
        h.load_full_width,
        &nxyz,
        load_x0,
        load_x1,
        nx_load,
        load_y0,
        load_y1,
        ny_load,
        load_z0,
        load_z1,
        nz_load,
    );
    if h.if_debug > 0 {
        wall_now = wall_time();
        h.wall_cum[0] += wall_now - h.wall_start;
        h.wall_start = wall_now;
    }
    //
    // get the patch from the loaded data and scale it at the same time; taper inside
    // and shift mean to zero
    //
    {
        let (lo, hi) = h.buffer.split_at_mut(l0);
        extract_patch(
            hi,
            *nx_load,
            *ny_load,
            *nz_load,
            *load_x0,
            *load_y0,
            *load_z0,
            ix0,
            iy0,
            iz0,
            &mut lo[a0..],
            h.nx_patch_tmp,
            h.ny_patch_tmp,
            h.nz_patch_tmp,
            scale,
        );
    }
    if h.if_debug > 0 {
        wall_now = wall_time();
        h.wall_cum[1] += wall_now - h.wall_start;
        h.wall_start = wall_now;
    }
    let mut ind_taper = ind_patch_a;
    if sigma_kernel > 0. {
        ind_taper = h.ind_scratch;
    }
    taper_in_vol(
        &mut h.buffer,
        a0,
        h.nx_patch_tmp,
        h.ny_patch_tmp,
        h.nz_patch_tmp,
        (ind_taper - 1) as usize,
        nx_pad_dim,
        nx_pad,
        ny_pad,
        nz_pad,
        h.nx_taper,
        h.ny_taper,
        h.nz_taper,
    );
    if h.if_debug > 0 {
        wall_now = wall_time();
        h.wall_cum[2] += wall_now - h.wall_start;
        h.wall_start = wall_now;
    }

    if sigma_kernel > 0. {
        //call dumpVolume(buffer(indtaper), nxpadDim, nxpad, nypad, nzpad, 'taper.')
        let t0 = (ind_taper - 1) as usize;
        let (lo, hi) = h.buffer.split_at_mut(t0);
        kernel_smooth(
            hi,
            &mut lo[a0..],
            nx_pad_dim,
            nx_pad,
            ny_pad,
            nz_pad,
            h.kernel_size,
            sigma_kernel,
            h.num_smooth_threads,
        );
        //call dumpVolume(buffer(indpatcha), nxpadDim, nxpad, nypad, nzpad, 'smooth.')
        if h.if_debug > 0 {
            wall_now = wall_time();
            h.wall_cum[3] += wall_now - h.wall_start;
            h.wall_start = wall_now;
        }
    }
    vol_mean_zero(&mut h.buffer[a0..], nx_pad_dim, nx_pad, ny_pad, nz_pad);
    if h.if_debug > 0 {
        wall_now = wall_time();
        h.wall_cum[4] += wall_now - h.wall_start;
    }
}

/// Original contained subroutine `analyzeLocalSDs` (`corrsearch3d.f90:1143`).
///
/// analyzeLocalSDs sets up parameters for the multi-binning local SD
/// analysis, calls the routines to obtain the statistics, and analyzes the
/// edge values and the overall distribution to set threshold values
fn analyze_local_sds(h: &mut MainVars) {
    let mut num_pix: i32;
    let mut num_boxes: i32;
    let mut num_for_med: i32;
    let mut lsd_func_data = [0_i32; 4];
    let mut idel_samp: i32;
    let mut num_sample: i32;
    let mut num_patch_struct_loop: i32;
    let mut edge_median = [0.0_f32; LIM_LOCAL_BIN];
    let mut edge_madn = [0.0_f32; LIM_LOCAL_BIN];
    let mut fallback_sd = [0.0_f32; LIM_LOCAL_BIN];
    let mut hist_dip = [0.0_f32; LIM_LOCAL_BIN];
    let mut peak_below = [0.0_f32; LIM_LOCAL_BIN];
    let mut peak_above = [0.0_f32; LIM_LOCAL_BIN];
    let mut hist_start: f32;
    let mut hist_end: f32;
    let mut ratio: f32;
    let mut ratio_last: f32;
    let mut ratio_diff: f32;
    let mut box_struct_crit: f32;
    // Set on the first binning the loop accepts, before any read.
    let mut diff_last = 0.0_f32;
    let mut use_fallback: bool;
    let mut ierr: i32;
    let mut tmp: f32;
    let mut ix: i32;
    let nxyz_patch = [h.nx_patch, h.ny_patch, h.nz_patch];
    let if_debug = h.if_debug;
    let nlb = h.num_lsd_binnings as usize;
    //
    // Set up sizes and spacings
    for j in 0..nlb {
        num_pix = 1;
        for i in 0..3 {
            h.lsd_box_size[j][i] = 1.max(nint(
                (h.lsd_box_size[0][i] as f32 * h.lsd_binning[0][i] as f32)
                    / h.lsd_binning[j][i] as f32,
            ));
            num_pix *= h.lsd_box_size[j][i];
            let half = h.lsd_box_size[j][i] as f32 / 2.;
            let tenth = nxyz_patch[i] as f32 / 10.;
            // `min(a, b)`: `minss`, finite here.
            h.lsd_spacing[j][i] = 1.max(nint(if half < tenth { half } else { tenth }));
        }
        if num_pix < 15 {
            exit_error("Box size for local SDs is too small");
        }
    }

    h.lsd_start_coord[0] = h.ix_start;
    h.lsd_start_coord[1] = h.iy_start;
    h.lsd_start_coord[2] = h.iz_start;
    h.lsd_end_coord[0] = h.ix_start + (h.num_xpatch - 1) * h.ix_delta + h.nx_patch;
    h.lsd_end_coord[1] = h.iy_start + (h.num_ypatch - 1) * h.iy_delta + h.ny_patch;
    h.lsd_end_coord[2] = h.iz_start + (h.num_zpatch - 1) * h.iz_delta + h.nz_patch;
    let mut lsd_zind = 3_usize;
    if h.if_flip > 0 {
        lsd_zind = 2;
    }
    h.lsd_start_coord[lsd_zind - 1] = 0;
    h.lsd_end_coord[lsd_zind - 1] = h.nxyz[lsd_zind - 1] - 1;
    ierr = multi_bin_setup(
        &h.lsd_binning,
        &h.lsd_box_size,
        &h.lsd_spacing,
        h.num_lsd_binnings,
        &h.lsd_start_coord,
        &h.lsd_end_coord,
        &mut h.lsd_box_start,
        &mut h.num_lsd_boxes,
        &mut h.lsd_buffer_starts,
        &mut h.lsd_stat_starts,
    );
    if ierr != 0 {
        exit_error("Setting up multi-bin analysis");
    }
    num_boxes = 0;
    for ibin in 0..nlb {
        num_boxes = num_boxes.max(h.lsd_stat_starts[ibin + 1] - h.lsd_stat_starts[ibin]);
    }
    num_sample = num_boxes.min(2000 * h.num_hist_bins);
    let ibin_size = h.lsd_buffer_starts[nlb].max(num_sample + num_sample.max(h.num_hist_bins));
    h.stat_buffer = vec![0.0; ibin_size as usize];
    h.stat_means = vec![0.0; h.lsd_stat_starts[nlb] as usize];
    h.stat_sds = vec![0.0; h.lsd_stat_starts[nlb] as usize];
    h.patch_frac_high_sd = vec![0.0; h.lim_patch as usize];
    h.patch_mean_struct = vec![0.0; h.lim_patch as usize];
    memory_error(0, "arrays for multi-bin analysis");
    lsd_func_data[0] = h.lsd_start_coord[0];
    lsd_func_data[1] = h.lsd_end_coord[0];
    lsd_func_data[2] = h.lsd_start_coord[1];
    lsd_func_data[3] = h.lsd_end_coord[1];
    h.wall_start = wall_time();
    ierr = multi_bin_stats(
        &h.lsd_binning,
        &h.lsd_box_size,
        &h.lsd_spacing,
        h.num_lsd_binnings,
        &h.lsd_start_coord,
        &h.lsd_end_coord,
        &h.lsd_box_start,
        &h.num_lsd_boxes,
        &h.lsd_buffer_starts,
        &h.lsd_stat_starts,
        &mut h.stat_buffer,
        &mut h.stat_means,
        &mut h.stat_sds,
        &mut lsd_func_data,
        lsd_load_func,
    );
    if ierr != 0 {
        exit_error("Doing multi-bin analysis of SDs");
    }
    if if_debug > 0 {
        println!(
            "multiBinStats time{}",
            fmt_f(wall_time() - h.wall_start, 9, 3)
        );
    }

    //
    // Collect values on the top and bottom planes
    use_fallback = true;
    for ibin in 1..=h.num_lsd_binnings {
        let bu = (ibin - 1) as usize;
        //
        // Collect values on the top and bottom planes and analyze lowest fraction for
        // median/MADN
        for i in 0..3 {
            h.ib_start[i] = 1;
            h.ib_end[i] = h.num_lsd_boxes[bu][i];
        }
        h.ib_start[lsd_zind - 1] = h.ib_end[lsd_zind - 1];
        h.num_in_sample = 0;
        add_boxes_to_sample(h, ibin);
        h.ib_start[lsd_zind - 1] = 1;
        h.ib_end[lsd_zind - 1] = 1;
        add_boxes_to_sample(h, ibin);
        //
        // Sort the edge values and take median/MADN of a fraction of them
        rs_sort_floats(&mut h.stat_buffer, h.num_in_sample);
        let frac = h.edge_sample_frac * h.num_in_sample as f32;
        // `max(2., ...)`: `maxss`, finite here.
        num_for_med = (if 2. > frac { 2. } else { frac }) as i32;
        rs_median_of_sorted(&h.stat_buffer, num_for_med, &mut edge_median[bu]);
        {
            let (x, rest) = h.stat_buffer.split_at_mut(num_for_med as usize);
            rs_fast_madn(x, num_for_med, edge_median[bu], rest, &mut edge_madn[bu]);
        }
        println!(
            "\nFor scaling{}: edge SD median = {}, MADN = {}\nAnalyzing histogram of square \
             root of SDs:",
            fmt_i(ibin, 2),
            fmt_f(edge_median[bu] as f64, 9, 2),
            fmt_f(edge_madn[bu] as f64, 8, 2)
        );

        // Get a sample of data for large volumes to do percentile/histogram analysis
        num_boxes = h.lsd_stat_starts[bu + 1] - h.lsd_stat_starts[bu];
        num_sample = num_boxes.min(2000 * h.num_hist_bins);
        idel_samp = num_boxes / num_sample;
        ix = h.lsd_stat_starts[bu] + 1;
        for i in 1..=num_sample {
            h.stat_buffer[(i - 1) as usize] = h.stat_sds[(ix - 1) as usize];
            ix += idel_samp;
        }

        // Find a moderate high percentile as a measure of typical strong structure
        fallback_sd[bu] = percentile_float(
            1.max(nint(h.fallback_pct_frac * num_sample as f32)),
            &mut h.stat_buffer,
            num_sample,
        );

        // Take the square root to spread out the lower end of the histogram
        // Find a high-percentile cutoff value for limiting the histogram range
        for value in h.stat_buffer[..num_sample as usize].iter_mut() {
            *value = value.sqrt();
        }
        //histStart = minval(statBuffer(1:numSample))
        hist_start = percentile_float(
            1.max(nint(h.bottom_hist_frac * num_sample as f32)),
            &mut h.stat_buffer,
            num_sample,
        );
        hist_end = percentile_float(
            1.max(nint(h.frac_for_hist * num_sample as f32)),
            &mut h.stat_buffer,
            num_sample,
        );

        h.wall_start = wall_time();
        let failed = {
            let (values, rest) = h.stat_buffer.split_at_mut(num_sample as usize);
            findhistogramdip(
                values,
                0,
                &mut rest[..h.num_hist_bins as usize],
                hist_start,
                hist_end,
                &mut hist_dip[bu],
                &mut peak_below[bu],
                &mut peak_above[bu],
                h.if_hist_verbose,
            ) != 0
        };
        if failed {
            hist_dip[bu] = -1.;
        } else {
            hist_dip[bu] = hist_dip[bu] * hist_dip[bu];
            peak_below[bu] = peak_below[bu] * peak_below[bu];
            peak_above[bu] = peak_above[bu] * peak_above[bu];
            println!(
                "SD value of dip is {}, peaks at {}{}",
                fmt_f(hist_dip[bu] as f64, 9, 2),
                fmt_f(peak_below[bu] as f64, 9, 2),
                fmt_f(peak_above[bu] as f64, 9, 2)
            );
            tmp = (peak_below[bu] - edge_median[bu]) / edge_madn[bu];
            if tmp > h.bad_hist_peak_crit {
                hist_dip[bu] = -1.;
                println!(
                    "Lower peak is{} MADNs above edge median so this dip is assumed to be wrong",
                    fmt_f(tmp as f64, 9, 2)
                );
            } else {
                use_fallback = false;
            }
        }
        if if_debug > 0 {
            println!(
                "{} of{} boxes; histogram time{}",
                fmt_i(num_sample, 8),
                fmt_i(num_boxes, 12),
                fmt_f(wall_time() - h.wall_start, 9, 3)
            );
        }
    }
    println!();

    // If not using fallback and eliminating by SD, loop twice, first time see if too
    // many are elimated and if so use fallbacks
    num_patch_struct_loop = 2;
    if use_fallback || nint(h.elim_by_sd_type) < 1 {
        num_patch_struct_loop = 1;
    }
    for loop_patch_struct in 1..=num_patch_struct_loop {
        // Find most distinguishable binning based on ones with dips, or based on all
        // fallbacks
        ratio_last = 0.;
        ratio_diff = 0.;
        h.ibin_best = -1;
        for ibin in 1..=h.num_lsd_binnings {
            let bu = (ibin - 1) as usize;
            if use_fallback {
                ratio = (fallback_sd[bu] - edge_median[bu]) / edge_madn[bu];
            } else {
                if hist_dip[bu] < 0. {
                    continue;
                }
                ratio = (peak_above[bu] - edge_median[bu]) / edge_madn[bu];
            }
            if h.ibin_best < 0 {
                h.ibin_best = ibin;
            } else {
                ratio_diff = (ratio - ratio_last) / (ibin - h.ibin_best) as f32;
                if ratio_diff < h.better_scale_crit * diff_last {
                    continue;
                }
                h.ibin_best = ibin;
            }
            ratio_last = ratio;
            diff_last = ratio_diff;
        }
        if h.num_lsd_binnings > 1 {
            println!(
                "Selected scaling #{} for further analysis\n",
                fmt_i(h.ibin_best, 2)
            );
        }
        //
        // Set up criteria now
        let best = (h.ibin_best - 1) as usize;
        if use_fallback {
            box_struct_crit = fallback_sd[best] * h.fallback_struct_frac;
            h.high_sd_level = fallback_sd[best];
            println!(
                "No histogram dips found; using fallback SD of {} for strong structure and \
                 criterion of {}",
                fmt_f(fallback_sd[best] as f64, 9, 2),
                fmt_f(box_struct_crit as f64, 9, 2)
            );
        } else {
            box_struct_crit = hist_dip[best];
            h.high_sd_level = peak_above[best];
        }
        //
        // Now evaluate each patch, find mean SD of blocks in patch and fraction of blocks
        // above the structure criterion.  Yes, this was actually good up to 10 threads
        h.wall_start = wall_time();
        let _threads = num_omp_threads(10);

        // `!$OMP PARALLEL DO` over `izPatch`: each patch's sums are its own, so
        // the loop runs sequentially with the same result.
        for iz_patch in 1..=h.num_zpatch {
            let iz0 = h.iz_start + (iz_patch - 1) * h.iz_delta;
            let (s, e) = find_boxes_inside_patch(h, 3, iz0, iz0 + h.nz_patch - 1);
            h.ib_start[2] = s;
            h.ib_end[2] = e;
            for iy_patch in 1..=h.num_ypatch {
                let iy0 = h.iy_start + (iy_patch - 1) * h.iy_delta;
                let (s, e) = find_boxes_inside_patch(h, 2, iy0, iy0 + h.ny_patch - 1);
                h.ib_start[1] = s;
                h.ib_end[1] = e;
                for ix_patch in 1..=h.num_xpatch {
                    let ix0 = h.ix_start + (ix_patch - 1) * h.ix_delta;
                    let (s, e) = find_boxes_inside_patch(h, 1, ix0, ix0 + h.nx_patch - 1);
                    h.ib_start[0] = s;
                    h.ib_end[0] = e;
                    let ind_p = ix_patch
                        + (iy_patch - 1) * h.num_xpatch
                        + (iz_patch - 1) * h.num_xpatch * h.num_ypatch;
                    let ipu = (ind_p - 1) as usize;
                    num_boxes = 0;
                    h.patch_mean_struct[ipu] = 0.;
                    h.patch_frac_high_sd[ipu] = 0.;
                    let best = (h.ibin_best - 1) as usize;
                    for iz in h.ib_start[2]..=h.ib_end[2] {
                        for iy in h.ib_start[1]..=h.ib_end[1] {
                            for ix in h.ib_start[0]..=h.ib_end[0] {
                                let i = h.lsd_stat_starts[best]
                                    + ((iz - 1) * h.num_lsd_boxes[best][1] + (iy - 1))
                                        * h.num_lsd_boxes[best][0]
                                    + ix;
                                let sd = h.stat_sds[(i - 1) as usize];
                                h.patch_mean_struct[ipu] += sd;
                                if sd >= box_struct_crit {
                                    h.patch_frac_high_sd[ipu] += 1.;
                                }
                                num_boxes += 1;
                            }
                        }
                    }
                    if num_boxes > 0 {
                        h.patch_frac_high_sd[ipu] /= num_boxes as f32;
                        h.patch_mean_struct[ipu] /= num_boxes as f32;
                    } else {
                        h.patch_mean_struct[ipu] = -1.;
                        h.patch_frac_high_sd[ipu] = -1.;
                    }
                }
            }
        }
        if if_debug > 0 {
            println!(
                " patch structure time{}",
                fmt_f(wall_time() - h.wall_start, 9, 3)
            );
        }

        // Count up the eliminations by the criterion
        if loop_patch_struct < num_patch_struct_loop {
            h.num_elim_by_sd = 0;
            for iz in 1..=h.num_zpatch {
                for iy in 1..=h.num_ypatch {
                    for ix in 1..=h.num_xpatch {
                        let ind_p =
                            ix + (iy - 1) * h.num_xpatch + (iz - 1) * h.num_xpatch * h.num_ypatch;
                        let ipu = (ind_p - 1) as usize;
                        //
                        if (nint(h.elim_by_sd_type) > 1
                            && h.patch_mean_struct[ipu] < h.elim_by_sd_crit * h.high_sd_level)
                            || (nint(h.elim_by_sd_type) == 1
                                && h.patch_frac_high_sd[ipu] < h.elim_by_sd_crit)
                        {
                            h.num_elim_by_sd += 1;
                        }
                    }
                }
            }

            // Use fallback if too many
            let i = h.num_xpatch * h.num_ypatch * h.num_zpatch;
            if h.num_elim_by_sd as f32 > h.sd_elim_max_frac * i as f32 {
                use_fallback = true;
                println!(
                    "{} of{} total possible patches would be eliminated by these SD \
                     criterion\nThat is too much elimination; recomputing with fallback \
                     criteria",
                    fmt_i(h.num_elim_by_sd, 7),
                    fmt_i(i, 7)
                );
            } else {
                break;
            }
        }
    }
}

/// Original contained subroutine `addBoxesToSample` (`corrsearch3d.f90:1407`).
///
/// addBoxesToSample adds boxes in the range ibStart, ibEnd into a sample
fn add_boxes_to_sample(h: &mut MainVars, iscl: i32) {
    let su = (iscl - 1) as usize;
    for iz_box in h.ib_start[2]..=h.ib_end[2] {
        for iy_box in h.ib_start[1]..=h.ib_end[1] {
            let ibase = h.lsd_stat_starts[su]
                + ((iz_box - 1) * h.num_lsd_boxes[su][1] + (iy_box - 1)) * h.num_lsd_boxes[su][0];
            for ix_box in h.ib_start[0]..=h.ib_end[0] {
                h.num_in_sample += 1;
                h.stat_buffer[(h.num_in_sample - 1) as usize] =
                    h.stat_sds[(ibase + ix_box - 1) as usize];
            }
        }
    }
}

/// Original contained subroutine `findBoxesInsidePatch`
/// (`corrsearch3d.f90:1425`).
///
/// findBoxesInsidePatch finds the box numbers in a patch whose starting and
/// ending coordinates are in ipStart, ipEnd, and returns the range of boxes
/// (the source's `ibStart`, `ibEnd` dummy arguments).
fn find_boxes_inside_patch(h: &MainVars, ixyz: i32, ip_start: i32, ip_end: i32) -> (i32, i32) {
    let iscl = (h.ibin_best - 1) as usize;
    let d = (ixyz - 1) as usize;
    let start = h.lsd_start_coord[d];
    let binning = h.lsd_binning[iscl][d];
    let box_start = h.lsd_box_start[iscl][d];
    let spacing = h.lsd_spacing[iscl][d];
    let box_size = h.lsd_box_size[iscl][d];
    let mut ib_start = (((ip_start - start) as f32 / binning as f32 - box_start as f32)
        / spacing as f32
        + 1.) as i32;
    while start + binning * (box_start + (ib_start - 1) * spacing) < ip_start {
        ib_start += 1;
    }
    let mut ib_end = ((((ip_end - start) as f32 / binning as f32 + 1. - box_start as f32)
        - box_size as f32)
        / spacing as f32)
        .ceil() as i32
        + 1;
    ib_end = h.num_lsd_boxes[iscl][d].min(ib_end);
    while start + binning * (box_start + (ib_end - 1) * spacing + box_size) < ip_end {
        ib_end -= 1;
    }
    (ib_start, ib_end)
}

/// Original subroutine `findBestCorr` (`corrsearch3d.f90:1466`).
///
/// FIND_BEST_CORR will search for the best 3-D displacement between
/// two volumes in ARRAY and BRRAY.  ARRAY is dimensioned to NXA by NYA
/// by NZA, and its starting index coordinates are LOADAX0, LOADAY0,
/// LOADAZ0.  BRRAY is dimensioned to NXB by NYB by NZB, and its
/// starting index coordinates are LOADBX0, LOADBY0, LOADBZ0.  The volume
/// to be correlated is specified by starting and ending index
/// coordinates IX0, IX1, IY0, IY1, IZ0, IZ1.  DXADJ, DYADJ, DZADJ
/// should contain a starting shift upon entry, and will return the
/// shift with the best correlation.  MAXSHIFT specifies the maximum
/// shift that is allowed.  FOUND is returned as TRUE if a correlation
/// peak is found within the maximum shift.  NCORRS is a variable that
/// allows the calling program to maintain a count of the total number
/// of correlations computed.
///
/// This was originally written to work with patches embedded in larger
/// loaded volumes, with the two volumes potentially loaded separately,
/// hence the unnecessary complexity for the program at hand.
#[allow(clippy::too_many_arguments)]
fn find_best_corr(
    array: &[f32],
    nx_a: i32,
    ny_a: i32,
    _nza: i32,
    load_ax0: i32,
    load_ay0: i32,
    load_az0: i32,
    brray: &[f32],
    nx_b: i32,
    ny_b: i32,
    nzb: i32,
    load_bx0: i32,
    load_by0: i32,
    load_bz0: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    iz0: i32,
    iz1: i32,
    dx_adjacent: &mut f32,
    dy_adjacent: &mut f32,
    dz_adjacent: &mut f32,
    max_shift: i32,
    found: &mut bool,
    num_corrs: &mut i32,
    num_threads: i32,
) {
    // `real*8 corrs(-1:1,-1:1,-1:1)`, `logical done(-1:1,-1:1,-1:1)`, indexed
    // `[iz + 1][iy + 1][ix + 1]`; `corrTmp`/`doneTmp` are `(-2:2, ...)`.
    // `corrs` is read only where `done` has been set.
    let mut corrs = [[[0.0_f64; 3]; 3]; 3];
    let mut corr_tmp = [[[0.0_f64; 5]; 5]; 5];
    let mut done = [[[false; 3]; 3]; 3];
    let mut done_tmp = [[[false; 5]; 5]; 5];
    let idy_sequence: [i32; 9] = [0, -1, 1, 0, 0, -1, 1, -1, 1];
    let idz_sequence: [i32; 9] = [0, 0, 0, 1, -1, -1, -1, 1, 1];
    let mut corr_max: f64;
    let mut ind_sequence: i32;
    let mut ind_max: i32;
    let (mut y1, mut y2, mut y3): (f32, f32, f32);
    //
    // get global displacement of b, including the load offset
    //
    let mut idx_global = nint(*dx_adjacent) + load_ax0 - load_bx0;
    let mut idy_global = nint(*dy_adjacent) + load_ay0 - load_by0;
    let mut idz_global = nint(*dz_adjacent) + load_az0 - load_bz0;
    //
    // clear flags for existence of corr
    //
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                done[iz][iy][ix] = false;
            }
        }
    }

    corr_max = -1.0e30;
    ind_sequence = 1;
    while ind_sequence <= 9 {
        let idy = idy_sequence[(ind_sequence - 1) as usize];
        let idz = idz_sequence[(ind_sequence - 1) as usize];
        let (yy, zz) = ((idy + 1) as usize, (idz + 1) as usize);
        if !(done[zz][yy][0] && done[zz][yy][1] && done[zz][yy][2]) {
            //
            // if the whole row does not exist, do the correlations
            // limit the extent if b is displaced and near an edge
            //
            let idy_corr = idy_global + idy;
            let idz_corr = idz_global + idz;
            let ix0corr = (ix0 - load_ax0).max(-(idx_global - 1));
            let ix1corr = (ix1 - load_ax0).min(nx_b - 1 - (idx_global + 1));
            let iy0corr = (iy0 - load_ay0).max(-idy_global);
            let iy1corr = (iy1 - load_ay0).min(ny_b - 1 - idy_global);
            let iz0corr = (iz0 - load_az0).max(-idz_global);
            let iz1cor = (iz1 - load_az0).min(nzb - 1 - idz_global);
            *num_corrs += 1;
            let (mut c1, mut c2, mut c3) = (0.0_f64, 0.0_f64, 0.0_f64);
            three_corrs(
                array,
                nx_a,
                ny_a,
                brray,
                nx_b,
                ny_b,
                ix0corr,
                ix1corr,
                iy0corr,
                iy1corr,
                iz0corr,
                iz1cor,
                idx_global,
                idy_corr,
                idz_corr,
                &mut c1,
                &mut c2,
                &mut c3,
                num_threads,
            );
            corrs[zz][yy] = [c1, c2, c3];
            done[zz][yy] = [true, true, true];
        }
        let row = corrs[zz][yy];
        if row[1] > row[0] && row[1] > row[2] {
            ind_max = 0;
        } else if row[0] > row[1] && row[0] > row[2] {
            ind_max = -1;
        } else {
            ind_max = 1;
        }
        if row[(ind_max + 1) as usize] > corr_max {
            // print *,'moving by', indmax, idy, idz
            corr_max = row[(ind_max + 1) as usize];
            //
            // if there is a new maximum, shift the done flags and the existing
            // correlations, and reset the sequence
            //
            idx_global += ind_max;
            idy_global += idy;
            idz_global += idz;
            //
            // but if beyond the limit, return failure
            //
            if (idx_global + load_bx0 - load_ax0)
                .abs()
                .max((idy_global + load_by0 - load_ay0).abs())
                .max((idz_global + load_bz0 - load_az0).abs())
                > max_shift
            {
                *found = false;
                return;
            }
            for iz in -1..=1_i32 {
                for iy in -1..=1_i32 {
                    for ix in -1..=1_i32 {
                        done_tmp[(iz + 2) as usize][(iy + 2) as usize][(ix + 2) as usize] = false;
                    }
                }
            }
            for iz in -1..=1_i32 {
                for iy in -1..=1_i32 {
                    for ix in -1..=1_i32 {
                        let (tz, ty, tx) = (
                            (iz - idz + 2) as usize,
                            (iy - idy + 2) as usize,
                            (ix - ind_max + 2) as usize,
                        );
                        let (sz, sy, sx) =
                            ((iz + 1) as usize, (iy + 1) as usize, (ix + 1) as usize);
                        done_tmp[tz][ty][tx] = done[sz][sy][sx];
                        corr_tmp[tz][ty][tx] = corrs[sz][sy][sx];
                    }
                }
            }
            for iz in -1..=1_i32 {
                for iy in -1..=1_i32 {
                    for ix in -1..=1_i32 {
                        let (tz, ty, tx) =
                            ((iz + 2) as usize, (iy + 2) as usize, (ix + 2) as usize);
                        let (sz, sy, sx) =
                            ((iz + 1) as usize, (iy + 1) as usize, (ix + 1) as usize);
                        done[sz][sy][sx] = done_tmp[tz][ty][tx];
                        corrs[sz][sy][sx] = corr_tmp[tz][ty][tx];
                    }
                }
            }
            if ind_max != 0 || idy != 0 || idz != 0 {
                ind_sequence = 0;
            }
        }
        ind_sequence += 1;
    }
    //
    // do independent parabolic fits in 3 dimensions
    //
    y1 = corrs[1][1][0] as f32;
    y2 = corrs[1][1][1] as f32;
    y3 = corrs[1][1][2] as f32;
    let cx = parabolic_fit_position(y1, y2, y3) as f32;
    y1 = corrs[1][0][1] as f32;
    y3 = corrs[1][2][1] as f32;
    let cy = parabolic_fit_position(y1, y2, y3) as f32;
    y1 = corrs[0][1][1] as f32;
    y3 = corrs[2][1][1] as f32;
    let cz = parabolic_fit_position(y1, y2, y3) as f32;
    //
    *dx_adjacent = idx_global as f32 + cx + load_bx0 as f32 - load_ax0 as f32;
    *dy_adjacent = idy_global as f32 + cy + load_by0 as f32 - load_ay0 as f32;
    *dz_adjacent = idz_global as f32 + cz + load_bz0 as f32 - load_az0 as f32;
    *found = true;
    // print *,'returning a peak'
}

/// Original subroutine `threeCorrs` (`corrsearch3d.f90:1627`).
///
/// THREECORRS computes three correlations between volumes in ARRAY
/// and BRRAY.  The volume in ARRAY is dimensioned to NXA by NYA, that
/// in BRRAY is dimensioned to NXB by NYB.  The starting and ending
/// index coordinates (numbered from 0) in ARRAY over which the
/// correlations are to be computed are IX0, IX1, IY0, IY1, IZ0, IZ1.
/// The shift between coordinates in ARRAY and coordinates in B is given
/// by IDX, IDY, IDZ.  The three correlations are returned in CORR1 (for
/// IDX-1), CORR2 (for IDX), and CORR3 (for IDX+1).
///
/// The source's `!$OMP PARALLEL DO ... REDUCTION(+)` over `iz` runs
/// sequentially here: what native computes at one thread.
#[allow(clippy::too_many_arguments)]
fn three_corrs(
    array: &[f32],
    nx_a: i32,
    ny_a: i32,
    brray: &[f32],
    nx_b: i32,
    ny_b: i32,
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
    let mut sum1 = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut sum3 = 0.0_f64;

    for iz in iz0..=iz1 {
        let iz_b = iz + idz;
        for iy in iy0..=iy1 {
            let iyb = iy + idy;
            let ind_base_a = 1 + iy * nx_a + iz * nx_a * ny_a;
            let ind_del_b = 1 + iyb * nx_b + iz_b * nx_b * ny_b + idx - ind_base_a;
            if ix1 < ix0 {
                continue;
            }
            // 1-based `array(ix)`, `brray(ixB - 1)`, `brray(ixB)`, `brray(ixB + 1)`
            // are the 0-based `ix - 1`, `ixB - 2`, `ixB - 1`, `ixB`.
            let a_row = &array[(ind_base_a + ix0 - 1) as usize..=(ind_base_a + ix1 - 1) as usize];
            let b_start = (ind_base_a + ix0 + ind_del_b - 2) as usize;
            let b_row = &brray[b_start..b_start + a_row.len() + 2];
            for (&a, b) in a_row.iter().zip(b_row.windows(3)) {
                sum1 += (a * b[0]) as f64;
                sum2 += (a * b[1]) as f64;
                sum3 += (a * b[2]) as f64;
            }
        }
    }
    let nsum = (iz1 + 1 - iz0) * (iy1 + 1 - iy0) * (ix1 + 1 - ix0);
    *corr1 = sum1 / nsum as f64;
    *corr2 = sum2 / nsum as f64;
    *corr3 = sum3 / nsum as f64;
    // print *,idx, idy, idz, corr1, corr2, corr3
}

/// Original subroutine `oneCorrCoeff` (`corrsearch3d.f90:1707`).
///
/// oneCorrCoeff computes one correlation coefficient between volumes in
/// ARRAY and BRRAY.  The volume in ARRAY is dimensioned to NXA by NYA by
/// NZA, that in BRRAY is dimensioned to NXB by NYB by NZB.  The size of
/// volume to be correlated is NXPATCH by NYPATCH by NZPATCH and it is
/// assumed to be centered in the arrays.  The shift between coordinates
/// in ARRAY and coordinates in B is given by DX, DY, DZ.  The correlation
/// coefficient is returned in CORR2.
///
/// The source's OpenMP reduction runs sequentially, as native at one thread.
#[allow(clippy::too_many_arguments)]
fn one_corr_coeff(
    array: &[f32],
    nx_a: i32,
    ny_a: i32,
    nza: i32,
    brray: &[f32],
    nx_b: i32,
    ny_b: i32,
    nzb: i32,
    nx_patch: i32,
    ny_patch: i32,
    nz_patch: i32,
    dx: f32,
    dy: f32,
    dz: f32,
    corr2: &mut f32,
    _num_threads: i32,
) {
    let mut sum2 = 0.0_f64;
    let mut a_sum = 0.0_f64;
    let mut bsum2 = 0.0_f64;
    let mut asum_sq = 0.0_f64;
    let mut bsum_sq2 = 0.0_f64;
    let idx = nint(dx) + (nx_b - nx_a) / 2;
    let mut ix0 = (nx_a - nx_patch) / 2;
    let ix1 = (nx_patch + ix0 - 1).min(nx_b - 1 - idx);
    ix0 = ix0.max(-idx);
    let idy = nint(dy) + (ny_b - ny_a) / 2;
    let mut iy0 = (ny_a - ny_patch) / 2;
    let iy1 = (ny_patch + iy0 - 1).min(ny_b - 1 - idy);
    iy0 = iy0.max(-idy);
    let idz = nint(dz) + (nzb - nza) / 2;
    let mut iz0 = (nza - nz_patch) / 2;
    let iz1 = (nz_patch + iz0 - 1).min(nzb - 1 - idz);
    iz0 = iz0.max(-idz);

    for iz in iz0..=iz1 {
        let iz_b = iz + idz;
        for iy in iy0..=iy1 {
            let iyb = iy + idy;
            let ind_base_a = 1 + iy * nx_a + iz * nx_a * ny_a;
            let ind_del_b = 1 + iyb * nx_b + iz_b * nx_b * ny_b + idx - ind_base_a;

            // `do ix = indBaseA + ix0, indBaseA + ix1` over `array(ix)` and
            // `brray(ix + indDelB)`, as two row slices.
            if ix1 < ix0 {
                continue;
            }
            let a0 = (ind_base_a + ix0 - 1) as usize;
            let n = (ix1 - ix0 + 1) as usize;
            let b0 = (ind_base_a + ix0 + ind_del_b - 1) as usize;
            for (&a, &b) in array[a0..a0 + n].iter().zip(&brray[b0..b0 + n]) {
                sum2 += (a * b) as f64;
                a_sum += a as f64;
                asum_sq += (a * a) as f64;
                bsum2 += b as f64;
                bsum_sq2 += (b * b) as f64;
            }
        }
    }
    let nsum = (iz1 + 1 - iz0) * (iy1 + 1 - iy0) * (ix1 + 1 - ix0);
    let mut denom =
        ((nsum as f64 * asum_sq - a_sum * a_sum) * (nsum as f64 * bsum_sq2 - bsum2 * bsum2)) as f32;
    //
    // Set the ccc to 0 if the denominator is illegal, otherwise limit it
    // to +/-1
    if denom <= 0. {
        *corr2 = 0.;
    } else {
        denom = denom.sqrt();
        *corr2 = (nsum as f64 * sum2 - a_sum * bsum2) as f32;
        if denom < *corr2 {
            *corr2 = 1.0_f32.copysign(*corr2);
        } else {
            *corr2 /= denom;
        }
    }
    // print *,idx, idy, idz, corr2, denom
}

/// Original subroutine `kernelSmooth` (`corrsearch3d.f90:1783`).
///
/// kernelSmooth applies a 3D gaussian kernel to the data in ARRAY and
/// places the result in BRRAY.  The image size is NX x NY x NZ and the
/// dimensions of the arrays are NXDIM x NY x NZ.  ISIZE specifies the
/// kernel size (3 or 5) and sigma is the standard deviation of the
/// Gaussian
///
/// The weight loop's `exp` is gfortran's: for a 5-wide kernel the first four
/// `k` of each `(i, j)` go through libmvec `_ZGVbN4v_expf` and the fifth
/// through `expf`; for 3 all through `expf` (`corrsearch3d.o`, `kernelsmooth_`).
/// The plane loop is `!$OMP PARALLEL DO` over independent planes, run
/// sequentially.
#[allow(clippy::too_many_arguments)]
fn kernel_smooth(
    array: &[f32],
    brray: &mut [f32],
    nx_dim: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    isize: i32,
    sigma: f32,
    _num_threads: i32,
) {
    // `w(5,5,5)`, `w(i, j, k)` at `[k - 1][j - 1][i - 1]`.
    let mut w = [[[0.0_f32; 5]; 5]; 5];
    let mut wsum: f32;
    //
    // Make up the gaussian kernel
    //
    let mid = (isize + 1) / 2;
    let sig2 = sigma * sigma;
    wsum = 0.;
    for i in 1..=isize {
        for j in 1..=isize {
            let arg = |k: i32| -> f32 {
                -(((i - mid) * (i - mid) + (j - mid) * (j - mid) + (k - mid) * (k - mid)) as f32
                    / sig2)
            };
            let mut k = 1;
            if isize - 1 > 2 {
                use core::arch::x86_64::{_mm_setr_ps, _mm_storeu_ps};
                let mut lanes = [0.0_f32; 4];
                // SAFETY: SSE is part of the x86_64 baseline.  The call follows
                // the x86_64 vector-function ABI (`__m128` in and out of
                // `xmm0`, the C ABI's caller-saved registers clobbered), and
                // `asm!` without `nostack` enters with the stack aligned for a
                // call.
                unsafe {
                    let mut v = _mm_setr_ps(arg(1), arg(2), arg(3), arg(4));
                    core::arch::asm!(
                        "call {f}@PLT",
                        f = sym _ZGVbN4v_expf,
                        inout("xmm0") v,
                        clobber_abi("C"),
                    );
                    _mm_storeu_ps(lanes.as_mut_ptr(), v);
                }
                for (n, value) in lanes.iter().enumerate() {
                    w[n][(j - 1) as usize][(i - 1) as usize] = *value;
                    wsum += *value;
                }
                k = 5;
            }
            while k <= isize {
                let value = arg(k).exp();
                w[(k - 1) as usize][(j - 1) as usize][(i - 1) as usize] = value;
                wsum += value;
                k += 1;
            }
        }
    }
    for i in 0..isize as usize {
        for j in 0..isize as usize {
            for k in 0..isize as usize {
                w[k][j][i] /= wsum;
            }
        }
    }
    //
    // Form the weighted sums: moving this to a subroutine was needed to keep the
    // single-thread performance from being 2x slower than non-OMP
    //
    for iz in 0..=nz - isize {
        smooth_one_plane(array, brray, nx_dim, nx, ny, iz, isize, &w);
    }
    //
    // Copy the walls of the volume
    //
    let at = |ix: i32, iy: i32, iz: i32| -> usize {
        ((ix - 1) + nx_dim * ((iy - 1) + ny * (iz - 1))) as usize
    };
    let less = (mid - 1) / 2;
    let mut iz = 1;
    while iz <= nz - less {
        for iy in 1..=ny {
            let s = at(1, iy, iz);
            brray[s..s + nx as usize].copy_from_slice(&array[s..s + nx as usize]);
            if isize > 3 {
                let s = at(1, iy, iz + 1);
                brray[s..s + nx as usize].copy_from_slice(&array[s..s + nx as usize]);
            }
        }
        iz += nz - less - 1;
    }
    let mut iy = 1;
    while iy <= ny - less {
        for iz in 1..=nz {
            let s = at(1, iy, iz);
            brray[s..s + nx as usize].copy_from_slice(&array[s..s + nx as usize]);
            if isize > 3 {
                let s = at(1, iy + 1, iz);
                brray[s..s + nx as usize].copy_from_slice(&array[s..s + nx as usize]);
            }
        }
        iy += ny - less - 1;
    }
    let mut ix = 1;
    while ix <= nx - less {
        for iz in 1..=nz {
            for iy in 1..=ny {
                brray[at(ix, iy, iz)] = array[at(ix, iy, iz)];
            }
            if isize > 3 {
                for iy in 1..=ny {
                    brray[at(ix + 1, iy, iz)] = array[at(ix + 1, iy, iz)];
                }
            }
        }
        ix += nx - less - 1;
    }
}

/// Original subroutine `smoothOnePlane` (`corrsearch3d.f90:1852`).
///
/// Form the weighted sums for one Z plane: this formulation is 3 times faster
/// than adding everything into one output pixel at a time
#[allow(clippy::too_many_arguments)]
fn smooth_one_plane(
    array: &[f32],
    brray: &mut [f32],
    nx_dim: i32,
    nx: i32,
    ny: i32,
    iz: i32,
    isize: i32,
    w: &[[[f32; 5]; 5]; 5],
) {
    let mid = (isize + 1) / 2;
    let at = |ix: i32, iy: i32, iz: i32| -> usize {
        ((ix - 1) + nx_dim * ((iy - 1) + ny * (iz - 1))) as usize
    };
    for iy in 0..=ny - isize {
        let out = at(mid, iy + mid, iz + mid);
        let len = (nx - isize + 1) as usize;
        brray[out..out + len].fill(0.);
        for k in 1..=isize {
            for j in 1..=isize {
                let src = at(1, iy + j, iz + k);
                let wj = &w[(k - 1) as usize][(j - 1) as usize];
                let row = &mut brray[out..out + len];
                if isize == 3 {
                    let (w1, w2, w3) = (wj[0], wj[1], wj[2]);
                    let (a1, a2, a3) = (
                        &array[src..src + len],
                        &array[src + 1..src + 1 + len],
                        &array[src + 2..src + 2 + len],
                    );
                    for n in 0..len {
                        row[n] = row[n] + a1[n] * w1 + a2[n] * w2 + a3[n] * w3;
                    }
                } else {
                    let (w1, w2, w3, w4, w5) = (wj[0], wj[1], wj[2], wj[3], wj[4]);
                    let (a1, a2, a3, a4, a5) = (
                        &array[src..src + len],
                        &array[src + 1..src + 1 + len],
                        &array[src + 2..src + 2 + len],
                        &array[src + 3..src + 3 + len],
                        &array[src + 4..src + 4 + len],
                    );
                    for n in 0..len {
                        row[n] =
                            row[n] + a1[n] * w1 + a2[n] * w2 + a3[n] * w3 + a4[n] * w4 + a5[n] * w5;
                    }
                }
            }
        }
    }
}

/// Original subroutine `fourierCorr` (`corrsearch3d.f90:1887`).
///
/// Computes a cross-correlation between volumes in ARRAY and BRRAY
/// via fourier transforms and filters by the values in CTF if DELTA,
/// the frequency spacing of the points in CTF, is nonzero.  NXDIM is
/// the X dimension of the FFT, NY and NZ are the Y and Z image dimensions.
/// WORK must be dimensioned at least NXDIM x NZ.
///
/// `complex array(nxDim, ny, nz)` is the interleaved `real*4` pairs.
#[allow(clippy::too_many_arguments)]
fn fourier_corr(
    array: &mut [f32],
    brray: &mut [f32],
    nx_dim: i32,
    ny: i32,
    nz: i32,
    work: &mut [f32],
    ctf: &[f32],
    delta: f32,
) {
    //
    // Get nx in real space and take 3D FFT's
    //
    let nx_real = 2 * (nx_dim - 1);
    thrdfft(array, work, nx_real, ny, nz, 0);
    thrdfft(brray, work, nx_real, ny, nz, 0);
    //
    // multiply complex conjugate of array by brray, put back in array
    // This is different from usual so that we will get the amount b is
    // displaced from A, not the amount to shift B to align to A
    //
    let total = (nx_dim * ny * nz) as usize;
    for n in 0..total {
        let (ar, ai) = (array[2 * n], array[2 * n + 1]);
        let (br, bi) = (brray[2 * n], brray[2 * n + 1]);
        // `conjg(a) * b`: gfortran's complex product of `(ar, -ai)` and
        // `(br, bi)` with Fortran rules (no NaN recovery).
        array[2 * n] = ar * br - (-ai) * bi;
        array[2 * n + 1] = ar * bi + (-ai) * br;
    }
    //
    // Filter if delta set
    //
    if delta > 0. {
        let del_x = 0.5 / (nx_dim as f32 - 1.);
        let del_y = 1. / ny as f32;
        let del_z = 1. / nz as f32;
        for jz in 1..=nz {
            let mut za = (jz - 1) as f32 * del_z;
            if za > 0.5 {
                za = 1. - za;
            }
            for jy in 1..=ny {
                let mut ya = (jy - 1) as f32 * del_y;
                if ya > 0.5 {
                    ya = 1. - ya;
                }
                for jx in 1..=nx_dim {
                    let xa = (jx - 1) as f32 * del_x;
                    let s = (xa * xa + ya * ya + za * za).sqrt();
                    let ind_f = (s / delta + 1.5) as i32;
                    let c = ctf[(ind_f - 1) as usize];
                    let n = ((jx - 1) + nx_dim * ((jy - 1) + ny * (jz - 1))) as usize;
                    // `complex * real` is a full COMPLEX product with `(c, 0.)`:
                    // gfortran honours signed zeros, so the promoted real is not
                    // treated as real-only and both cross terms are computed
                    // (reference object, `fouriercorr_` four `mulss`), so a NaN
                    // or Inf in one part reaches both.
                    let (ar, ai) = (array[2 * n], array[2 * n + 1]);
                    array[2 * n] = ar * c - ai * 0.0;
                    array[2 * n + 1] = ar * 0.0 + ai * c;
                }
            }
        }
    }
    thrdfft(array, work, nx_real, ny, nz, -1);
}

/// Original subroutine `findXcorrPeak` (`corrsearch3d.f90:1944`).
///
/// findXcorrPeak finds the peak in a cross-correlation in ARRAY,
/// dimensioned to NXDIM x NY x NZ and image size NXDIM-2, NY, NZ.
/// It fits a parabola in each dimension to get interpolated peak
/// positions in XPEAK, YPEAK, ZPEAK, and returns peak magnitude in PEAK.
#[allow(clippy::too_many_arguments)]
fn find_xcorr_peak(
    array: &[f32],
    nx_dim: i32,
    ny: i32,
    nz: i32,
    xpeak: &mut f32,
    ypeak: &mut f32,
    zpeak: &mut f32,
    peak: &mut f32,
) {
    let at = |ix: i32, iy: i32, iz: i32| -> f32 {
        array[((ix - 1) + nx_dim * ((iy - 1) + ny * (iz - 1))) as usize]
    };
    // Set by the first comparison for any finite correlation.
    let (mut ix_peak, mut iy_peak, mut iz_peak) = (1_i32, 1_i32, 1_i32);
    //
    let nx = nx_dim - 2;
    *peak = -1.0e30;
    for iz in 1..=nz {
        for iy in 1..=ny {
            for ix in 1..=nx {
                if at(ix, iy, iz) > *peak {
                    *peak = at(ix, iy, iz);
                    ix_peak = ix;
                    iy_peak = iy;
                    iz_peak = iz;
                }
            }
        }
    }
    //
    // simply fit a parabola to the two adjacent points in X or Y or Z
    //
    let mut y1 = at(indmap(ix_peak - 1, nx), iy_peak, iz_peak);
    let y2 = *peak;
    let mut y3 = at(indmap(ix_peak + 1, nx), iy_peak, iz_peak);
    let cx = parabolic_fit_position(y1, y2, y3) as f32;

    y1 = at(ix_peak, indmap(iy_peak - 1, ny), iz_peak);
    y3 = at(ix_peak, indmap(iy_peak + 1, ny), iz_peak);
    let cy = parabolic_fit_position(y1, y2, y3) as f32;

    y1 = at(ix_peak, iy_peak, indmap(iz_peak - 1, nz));
    y3 = at(ix_peak, iy_peak, indmap(iz_peak + 1, nz));
    let cz = parabolic_fit_position(y1, y2, y3) as f32;
    //
    // return adjusted pixel coordinate minus 1
    //
    *xpeak = ix_peak as f32 + cx - 1.;
    *ypeak = iy_peak as f32 + cy - 1.;
    *zpeak = iz_peak as f32 + cz - 1.;
    if *xpeak > (nx / 2) as f32 {
        *xpeak -= nx as f32;
    }
    if *ypeak > (ny / 2) as f32 {
        *ypeak -= ny as f32;
    }
    if *zpeak > (nz / 2) as f32 {
        *zpeak -= nz as f32;
    }
}

/// Original subroutine `setBload` (`corrsearch3d.f90:2000`).
///
/// setBload takes the desired coordinates on an axis, IX0 and IX1, the
/// size in that dimension, NX2, the incremental and initial offsets
/// DXADJ and DXINITIAL, and computes the limits for data that need
/// to be loaded from B in IXB0, IXB1.  It adjusts IX0 and IX1 as
/// necessary to keep everything within limits
fn set_bload(
    ix0: &mut i32,
    ix1: &mut i32,
    nx2: i32,
    dx_adjacent: f32,
    dx_volume: f32,
    ix_b0: &mut i32,
    ix_b1: &mut i32,
) {
    let idx_adjacent = nint(dx_adjacent);
    let idx_volume = nint(dx_volume);
    *ix_b0 = 0.max(*ix0 + idx_adjacent + idx_volume);
    *ix0 = *ix_b0 - idx_adjacent - idx_volume;
    *ix_b1 = (nx2 - 1).min(*ix1 + idx_adjacent + idx_volume);
    *ix1 = *ix_b1 - idx_adjacent - idx_volume;
}

/// Original subroutine `manageLoad` (`corrsearch3d.f90:2018`).
///
/// MANAGELOAD tests whether the desired volume specified by IX0, IX1,
/// IY0, IY1, IZ0, IZ1 is already loaded, given the loaded limits in
/// LOADX0, etc.  If not, it loads the data, with extra amounts specified
/// by LOADEXH and a maximum load in X specified by MAXXLOAD
#[allow(clippy::too_many_arguments)]
fn manage_load(
    iunit: i32,
    buffer: &mut [f32],
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    iz0: i32,
    iz1: i32,
    load_extra_half: i32,
    ix_dir: i32,
    ix_delta: i32,
    max_xload: i32,
    load_full_width: bool,
    nxyz: &[i32; 3],
    load_x0: &mut i32,
    load_x1: &mut i32,
    nx_load: &mut i32,
    load_y0: &mut i32,
    load_y1: &mut i32,
    ny_load: &mut i32,
    load_z0: &mut i32,
    load_z1: &mut i32,
    nz_load: &mut i32,
) {
    let num_more: i32;
    //
    if ix0 >= *load_x0
        && ix1 <= *load_x1
        && iy0 >= *load_y0
        && iy1 <= *load_y1
        && iz0 >= *load_z0
        && iz1 <= *load_z1
    {
        return;
    }
    //
    // need to load new data
    //
    *load_y0 = 0.max(iy0 - load_extra_half);
    *load_y1 = (nxyz[1] - 1).min(iy1 + load_extra_half);
    *load_z0 = 0.max(iz0 - load_extra_half);
    *load_z1 = (nxyz[2] - 1).min(iz1 + load_extra_half);
    //
    // compute limits in X, loading as much as possible
    // but limiting to edge of data and then truncating
    // to the end of a patch
    //
    if max_xload >= nxyz[0] && load_full_width {
        *load_x0 = 0;
        *load_x1 = nxyz[0] - 1;
    } else if ix_dir > 0 {
        *load_x0 = 0.max(ix0 - load_extra_half);
        *load_x1 = (nxyz[0] - 1).min(ix0 + max_xload - 1 + load_extra_half);
        num_more = (*load_x1 - load_extra_half - ix1) / ix_delta;
        *load_x1 = (nxyz[0] - 1).min(ix1 + ix_delta * num_more + load_extra_half);
    } else {
        *load_x1 = (nxyz[0] - 1).min(ix1 + load_extra_half);
        *load_x0 = 0.max(ix1 + 1 - max_xload - load_extra_half);
        num_more = (ix0 - *load_x0 + load_extra_half) / ix_delta;
        *load_x0 = 0.max(ix0 - ix_delta * num_more - load_extra_half);
    }
    // write(*,'(a,i2,12i5)')'loading data', iunit, ix0, ix1, iy0, iy1, iz0, iz1, loadx0, &
    //  loadx1,  loady0, loady1, loadz0, loadz1
    *nx_load = *load_x1 + 1 - *load_x0;
    *ny_load = *load_y1 + 1 - *load_y0;
    *nz_load = *load_z1 + 1 - *load_z0;
    load_vol(
        iunit, buffer, *nx_load, *ny_load, *load_x0, *load_x1, *load_y0, *load_y1, *load_z0,
        *load_z1,
    );
}

/// Original subroutine `loadVol` (`corrsearch3d.f90:2072`).
///
/// LOADVOL loads a subset of the volume from unit IUNIT, into ARRAY
/// assuming dimensions of NXDIM by NYDIM, from index coordinates
/// IX0, IX1, IY0, IY1, IZ0, IZ1.
#[allow(clippy::too_many_arguments)]
fn load_vol(
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
    //
    // print *,iunit, nxdim, nydim, ix0, ix1, iy0, iy1, iz0, iz1
    let mut ind_z = 0;
    for iz in iz0..=iz1 {
        ind_z += 1;
        let start = ((ind_z - 1) * nx_dim * ny_dim) as usize;
        // SAFETY: `imposn`/`irdpas` reach the process-global unit table; the
        // section part lands in `nx_dim` x `ny_dim` floats of `array` at `start`.
        let ok = unsafe {
            iiu_set_position(iunit, iz, 0);
            irdpas(
                iunit,
                &mut array[start..],
                nx_dim,
                ny_dim,
                ix0,
                ix1,
                iy0,
                iy1,
            )
        };
        if ok.is_err() {
            // 99
            exit_error("Error reading file");
        }
    }
}

/// Original subroutine `extractPatch` (`corrsearch3d.f90:2095`).
///
/// EXTRACT_PATCH extracts a patch of dimensions NXPATCH by NYPATCH by
/// NZPATCH into ARRAY from the loaded volume in BUF, whose dimensions
/// are NXLOAD by NYLOAD by NZLOAD.  BUF is loaded from starting index
/// coordinates LOADX0, LOADY0, LOADZ0, and the starting index
/// coordinates of the patch are IX0, IY0, IZ0.  Values are multipled by
/// SCALE.
#[allow(clippy::too_many_arguments)]
fn extract_patch(
    buffer: &[f32],
    nx_load: i32,
    ny_load: i32,
    _nz_load: i32,
    load_x0: i32,
    load_y0: i32,
    load_z0: i32,
    ix0: i32,
    iy0: i32,
    iz0: i32,
    array: &mut [f32],
    nx_patch: i32,
    ny_patch: i32,
    nz_patch: i32,
    scale: f32,
) {
    //
    let ix = ix0 - load_x0;
    let iy = iy0 - load_y0;
    let iz = iz0 - load_z0;
    for kz in 0..nz_patch {
        for ky in 0..ny_patch {
            let src = (ix + nx_load * ((ky + iy) + ny_load * (kz + iz))) as usize;
            let dst = (nx_patch * (ky + ny_patch * kz)) as usize;
            for kx in 0..nx_patch as usize {
                array[dst + kx] = buffer[src + kx] * scale;
            }
        }
    }
}

/// Original subroutine `volMeanZero` (`corrsearch3d.f90:2115`).
///
/// VOLMEANZERO shifts the mean to zero of the volume in ARRAY
/// dimensioned NXDIM by NY by NZ, image size NX by NY by NZ
///
/// `sum(array(1:nx, 1:ny, 1:nz))` is a `real*4` intrinsic accumulated in
/// `real*4` in array-element order, and only then stored to `real*8 arsum`.
fn vol_mean_zero(array: &mut [f32], nx_dim: i32, nx: i32, ny: i32, nz: i32) {
    let mut sum = 0.0_f32;
    for iz in 0..nz {
        for iy in 0..ny {
            let s = (nx_dim * (iy + ny * iz)) as usize;
            for value in &array[s..s + nx as usize] {
                sum += *value;
            }
        }
    }
    let arsum = sum as f64;
    let dmean = (arsum / (nx * ny * nz) as f64) as f32;
    for iz in 0..nz {
        for iy in 0..ny {
            let s = (nx_dim * (iy + ny * iz)) as usize;
            for value in &mut array[s..s + nx as usize] {
                *value -= dmean;
            }
        }
    }
}

/// Original subroutine `checkAndSetPatches` (`corrsearch3d.f90:2131`).
///
/// checkAndSetPatches does error checks and sets the basic start
/// and delta for the patches in one dimension.
#[allow(clippy::too_many_arguments)]
fn check_and_set_patches(
    nx: i32,
    nbord_xlow: i32,
    nbord_xhigh: i32,
    nx_patch: i32,
    num_xpatch: &mut i32,
    ix_start: &mut i32,
    ix_delta: &mut i32,
    iaxis: i32,
) {
    //
    // check basic input properties
    //
    let axis = format!("{} axis", (b'W' + iaxis as u8) as char);
    if nbord_xlow < 0 || nbord_xhigh < 0 {
        exit_error(&format!("A negative border was entered for the {axis}"));
    }
    if nx_patch <= 4 {
        exit_error(&format!("Patch size negative or too small for the {axis}"));
    }
    if *num_xpatch <= 0 {
        exit_error(&format!(
            "Number of patches must be positive for the {axis}"
        ));
    }
    if nx_patch > nx - (nbord_xlow + nbord_xhigh) {
        println!(
            "\nERROR: CORRSEARCH3D -  Patch size ({}) is bigger than specified range ({}) for \
             the {}",
            fmt_i(nx_patch, 4),
            fmt_i(nx - (nbord_xlow + nbord_xhigh), 4),
            axis
        );
        exit(1);
    }
    //
    // If multiple patches, compute the delta and then adjust the number
    // of patches down to require a delta of at least 2
    //
    if *num_xpatch > 1 {
        *ix_start = nbord_xlow;
        *ix_delta = (nx - (nbord_xlow + nbord_xhigh + nx_patch)) / (*num_xpatch - 1);
        while *num_xpatch > 1 && *ix_delta < 2 {
            *num_xpatch -= 1;
            if *num_xpatch > 1 {
                *ix_delta = (nx - (nbord_xlow + nbord_xhigh + nx_patch)) / (*num_xpatch - 1);
            }
        }
    }
    //
    // If only one patch originally or now, center it in range
    //
    if *num_xpatch == 1 {
        *ix_start = (nbord_xlow + nx - nbord_xhigh) / 2 - nx_patch / 2;
        *ix_delta = 1;
    }
}

/// Original subroutine `revisePatchRange` (`corrsearch3d.f90:2178`).
///
/// revisePatchRange adjusts the starting position and number of
/// patches based on the vertex constraints in xvert2, xvert3
#[allow(clippy::too_many_arguments)]
fn revise_patch_range(
    nx: i32,
    nbord_xlow: i32,
    nbord_xhigh: i32,
    xvert2: f32,
    xvert3: f32,
    nx_patch: i32,
    num_xpatch: &mut i32,
    ix_start: &mut i32,
    ix_delta: &mut i32,
) {
    let mut ixlo2 = nbord_xlow.max(nint(xvert2));
    let ixhi2 = (nx - nbord_xhigh).min(nint(xvert3));
    if *num_xpatch > 1 {
        //
        // get new number of intervals inside the limits, and a new delta
        // to span the limits
        //
        let ix_span = 0.max(ixhi2 - ixlo2 - nx_patch);
        let mut numx2 = ix_span / *ix_delta + 1;
        let mut new_xdelta = ix_span / numx2;
        if (new_xdelta as f32) < 0.6 * *ix_delta as f32 {
            //
            // but if delta is too small, drop the number of intervals
            //
            numx2 -= 1;
            if numx2 > 0 {
                new_xdelta = ix_span / numx2;
            } else {
                new_xdelta = *ix_delta;
            }
        }
        //
        // now adjust ixlo2 up by half of remainder to center the patches
        // in the span and find true start and end that fits inside the
        // original low-hi limits
        //
        ixlo2 += (ix_span % new_xdelta) / 2;
        *ix_delta = new_xdelta;
        *ix_start = ixlo2 - *ix_delta * ((ixlo2 - nbord_xlow) / *ix_delta);
        *num_xpatch = (nx - nbord_xhigh - nx_patch - *ix_start) / *ix_delta + 1;
    } else {
        //
        // if only one patch, put it in new middle
        //
        *ix_start = (ixlo2 + ixhi2) / 2 - nx_patch / 2;
    }
}

/// Original subroutine `xformBsourceToA` (`corrsearch3d.f90:2228`).
///
/// Transforms a position XB, YB in B source to XA, YA in A.  NXYZBSRC and
/// NXYZ are the dimensions of B and A, IFFLIPB and IFFLIP are 1 if the
/// long dimension is Z in B or A, ASRC(3, 3) and DXYZSRC(3) have the
/// 3D transformation.  Returns `(xa, ya)`.
#[allow(clippy::too_many_arguments)]
fn xform_bsource_to_a(
    xb: f32,
    yb: f32,
    nxyz_bsource: &[i32; 3],
    nxyz: &[i32; 3],
    if_flip_b: i32,
    if_flip: i32,
    a_source: &[[f32; 3]; 3],
    dxyz_source: &[f32; 3],
) -> (f32, f32) {
    let mut tmp_b = [0.0_f32; 3];
    let mut tmp_a = [0.0_f32; 3];

    let mut ind_yb = 2_usize;
    if if_flip_b != 0 {
        ind_yb = 3;
    }
    tmp_b[0] = xb;
    tmp_b[ind_yb - 1] = yb;
    tmp_b[5 - ind_yb - 1] = nxyz_bsource[5 - ind_yb - 1] as f32 / 2.;
    for i in 0..3 {
        tmp_a[i] = dxyz_source[i] + nxyz[i] as f32 / 2.;
        for j in 0..3 {
            tmp_a[i] += a_source[i][j] * (tmp_b[j] - nxyz_bsource[j] as f32 / 2.);
        }
        // print *,(asrc(i, j), j = 1, 3), dxyzsrc(i), tmpb(i), tmpa(i)
    }
    let xa = tmp_a[0];
    let mut ya = tmp_a[1];
    if if_flip != 0 {
        ya = tmp_a[2];
    }
    (xa, ya)
}

/// Original subroutine `sequencePatches` (`corrsearch3d.f90:2258`).
///
/// Fills arrays IXSEQ, IYSEQ, IZSEQ with a sequence of patch numbers
/// starting from the center outward, progressing in X, then Y, then Z
/// NUMXPAT, NUMYPAT, NUMZPAT is number of patches in each direction;
/// NUMSEQ is returned with total number to loop on
#[allow(clippy::too_many_arguments)]
fn sequence_patches(
    num_xpatch: i32,
    num_ypatch: i32,
    num_zpatch: i32,
    ix_sequence: &mut [i32],
    iy_sequence: &mut [i32],
    iz_sequence: &mut [i32],
    idir_sequence: &mut [i32],
    num_sequence: &mut i32,
) {
    // A Fortran `DO` with step `dir`, as a counted iteration.
    let steps = |start: i32, end: i32, dir: i32| -> Vec<i32> {
        let count = ((end - start + dir) / dir).max(0);
        (0..count).map(|n| start + n * dir).collect()
    };
    //
    let mut iz_patch_start = num_zpatch / 2 + 1;
    let mut iz_patch_end = num_zpatch;
    *num_sequence = 0;
    for iz_dir in [1, -1] {
        for iz_patch in steps(iz_patch_start, iz_patch_end, iz_dir) {
            let mut iy_patch_start = num_ypatch / 2 + 1;
            let mut iy_patch_end = num_ypatch;
            for iy_dir in [1, -1] {
                for iy_patch in steps(iy_patch_start, iy_patch_end, iy_dir) {
                    let mut ix_patch_start = num_xpatch / 2 + 1;
                    let mut ix_patch_end = num_xpatch;
                    for ix_dir in [1, -1] {
                        for ix_patch in steps(ix_patch_start, ix_patch_end, ix_dir) {
                            *num_sequence += 1;
                            let n = (*num_sequence - 1) as usize;
                            ix_sequence[n] = ix_patch;
                            iy_sequence[n] = iy_patch;
                            iz_sequence[n] = iz_patch;
                            idir_sequence[n] = ix_dir;
                        }
                        ix_patch_start -= 1;
                        ix_patch_end = 1;
                    }
                }
                iy_patch_start -= 1;
                iy_patch_end = 1;
            }
        }
        iz_patch_start -= 1;
        iz_patch_end = 1;
    }
}

/// Original function `lsdLoadFunc` (`corrsearch3d.f90:2358`).
///
/// The function for loading data for multibinstats
fn lsd_load_func(iz: &mut i32, idata: &mut [i32], buffer: &mut [f32]) -> i32 {
    // SAFETY: unit 1 is the open reference volume; the section part lands in
    // `idata(2) + 1 - idata(1)` floats per line of `buffer`.
    unsafe {
        iiu_set_position(1, *iz, 0);
        iiu_read_sec_part(
            1,
            buffer.as_mut_ptr().cast(),
            idata[1] + 1 - idata[0],
            idata[0],
            idata[1],
            idata[2],
            idata[3],
        )
    }
}
