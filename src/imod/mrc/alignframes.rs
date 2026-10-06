//! Translation of `IMOD/mrc/alignframes.cpp` and `IMOD/mrc/alignframes.h` —
//! align movie frames and stack multiple frame files.
//!
//! One Rust function per source function: the file-level `main` is
//! [`alignframes`], and every `AliFrame` method is a method on [`AliFrame`],
//! whose fields mirror the class members in declaration order.
//!
//! Ownership notes, all forced by the C's pointers:
//!
//! * The file static `sFA` (`alignframes.cpp:39`) is the [`AliFrame::s_fa`]
//!   field: one instance per run, which is what the static is.
//! * `MrcHeader *mOutHeads[4]` points at one of the four member headers
//!   `mMainHead`, `mUnwgtHead`, `mEvenHead`, `mOddHead`.  Those are
//!   [`AliFrame::m_heads`] (in that order) and `mOutHeads` is
//!   [`AliFrame::m_out_heads`], the index of the header each output uses.
//! * `Islice *mGainSlice` is only ever read through `data.f`; it is the float
//!   data itself, shared (`Rc`) with `FrameAlign`, which keeps the pointer
//!   between frames exactly as `mGainRef` does in the C.
//! * The `unsigned char *` frame buffers are `Vec<f32>` storage (for the
//!   alignment a `short`/`float` view needs, as `malloc` guarantees) viewed as
//!   bytes or as the typed array the mode selects.
//! * `mPartialLinePtrs` (`makeLinePointers` over `mPartialScanBufs`) are the
//!   line views built at the one point they are used.
//! * The GPU calls go through `FrameAlign` to `nogpuframe.rs`, the no-CUDA
//!   stub the reference build links, so they fail and the CPU path is taken,
//!   as natively.  `TEST_SHRMEM` is not defined in the reference build, so its
//!   `#ifdef` arms (the `ShrMemClient` calls and `defectFileToString`) are not
//!   compiled there and are not here.
//!
//! Upstream defects fixed in the translation are listed in `BUGS.md`,
//! section "`alignframes` (2026-10-05)", and marked in place.

use std::io::Write as _;
use std::os::unix::ffi::OsStrExt as _;
use std::rc::Rc;

use crate::imod::clip::correct_defects::{
    CameraDefects, cor_def_expand_gain_reference, cor_def_find_touching_pixels,
    cor_def_flip_defects_in_y, cor_def_parse_defects, cor_def_process_fei_defects,
    cor_def_read_super_gain, cor_def_refine_super_res_ref, cor_def_setup_to_correct,
};
use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, ADOC_ZVALUE_NAME, adoc_change_section_name, adoc_delete_key_value,
    adoc_get_float, adoc_get_integer, adoc_get_number_of_sections, adoc_get_section_name,
    adoc_get_string, adoc_get_two_integers, adoc_lookup_by_name_value, adoc_open_image_metadata,
    adoc_order_write_by_value, adoc_set_float, adoc_set_integer, adoc_set_key_value,
    adoc_set_three_floats, adoc_set_two_integers, adoc_write,
};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, SEEK_SET, b3d_fread, b3d_fseek, b3d_fwrite, b3d_get_error,
    b3d_output_file_type, b3d_physical_memory, balanced_group_limits, c_format_bytes,
    data_size_for_mode, exit, extra_is_nbytes_and_flags, fgetline, imod_backup_file,
    imod_prog_name, imod_usage_header, set_float_output_for_entered_mode, wall_time,
};
use crate::imod::libcfshr::extraheader::{
    get_metadata_by_key, get_metadata_weighting_doses, prior_doses_from_image_doses,
    set_zero_dose_thresh_and_accum,
};
use crate::imod::libcfshr::islice::{Islice, MrcData, slice_create, slice_mode_if_real};
use crate::imod::libcfshr::mxmlwrap::{
    ixml_clear, ixml_find_elements, ixml_get_string_attribute, ixml_get_string_value,
    ixml_load_string,
};
use crate::imod::libcfshr::parse_params::{
    PIP_FLOAT, PipValueArray, exit_error, pip_done, pip_get_boolean, pip_get_float,
    pip_get_float_array, pip_get_integer, pip_get_integer_array, pip_get_line_of_values,
    pip_get_non_option_arg, pip_get_string, pip_get_three_floats, pip_get_three_integers,
    pip_get_two_floats, pip_get_two_integers, pip_number_of_entries, pip_read_or_parse_options,
    strtod, strtol,
};
use crate::imod::libcfshr::reduce_by_binning::extract_with_binning;
use crate::imod::libcfshr::robuststat::{rs_sort_floats, rs_sort_indexed_floats};
use crate::imod::libcfshr::rotateflip::{RotateFlipData, rotate_flip_image};
use crate::imod::libcfshr::samplemeansd::{sample_mean_only, type_for_sample_mean};
use crate::imod::libcfshr::simplestat::{array_min_max_mean, array_min_max_mean_sd};
use crate::imod::libiimod::iilikemrc::ii_assume_dmfile_matches;
use crate::imod::libiimod::iimage::{
    IIFILE_MRC, IIFILE_TIFF, ImodImageFile, MRSA_NOPROC, get_dflt_eersumming_from_env, ii_delete,
    ii_fclose, ii_fopen, ii_lookup_file_from_fp, ii_open_copies_for_threads,
};
use crate::imod::libiimod::iitif::{
    IIFLAG_ANTIALIAS_EER, IIFLAG_EER_USE_LANCZOS, MAX_TIFF_THREADS,
    tiff_gain_reference_for_eer_bytes, tiff_get_array, tiff_get_field, tiff_get_max_eer_super_res,
    tiff_num_read_threads, tiff_parallel_read, tiff_set_eer_read_properties,
};
use crate::imod::libiimod::mrcfiles::{
    IMOD_MRC_STAMP, LoadInfo, MRC_HEADER_SIZE, MRC_LABEL_SIZE, MRC_MODE_BYTE, MRC_MODE_FLOAT,
    MRC_MODE_SHORT, MRC_MODE_USHORT, MRC_NLABELS, MrcHeader, fix_title_padding,
    mrc_copy_extra_header, mrc_get_scale, mrc_head_label, mrc_head_read, mrc_head_write,
    mrc_init_li, mrc_init_output_header, mrc_read_slice, mrc_set_scale,
};
use crate::imod::libiimod::mrcsec::mrc_write_z_float;
use crate::imod::libiimod::mrcslice::{
    SLICE_MODE_FLOAT, SLICE_MODE_SHORT, SLICE_MODE_USHORT, slice_read_mrc,
};
use crate::imod::mrc::framealign::{
    FrameAlign, FrameData, GPU_AVG_SUPER_2X, GPU_AVG_SUPER_4X, GPU_CORRECT_DEFECTS, GPU_DO_BIN_PAD,
    GPU_DO_EVEN_ODD, GPU_DO_GAIN_NORM, GPU_DO_NOISE_TAPER, GPU_DO_PREPROCESS, GPU_DO_UNWGT_SUM,
    GPU_FOR_ALIGNING, GPU_FOR_SUMMING, GPU_STACK_LIM_MASK, GPU_STACK_LIM_SHIFT, GPU_STACK_LIMITED,
    MAX_ALL_VS_ALL, MAX_FILTERS, STACK_FULL_ON_GPU,
};

/// `#define MAX_BINNINGS 6` (`alignframes.h:12`).
pub const MAX_BINNINGS: usize = 6;
/// `#define MAX_LINE 600` (`alignframes.h:13`).
pub const MAX_LINE: usize = 600;
/// `#define MAX_READ_THREADS 16` (`alignframes.h:14`).
pub const MAX_READ_THREADS: usize = 16;

/// `#define SIG2_ROUND_FAC 10000.` (`alignframes.cpp:32`).
const SIG2_ROUND_FAC: f64 = 10000.;
/// `#define FRAME_DOSE_KEY "FrameDosesAndNumber"` (`alignframes.cpp:33`).
const FRAME_DOSE_KEY: &str = "FrameDosesAndNumber";
/// `#define PRIOR_DOSE_KEY "PriorRecordDose"` (`alignframes.cpp:34`).
const PRIOR_DOSE_KEY: &[u8] = b"PriorRecordDose";
/// `#define SRF_NO_VAL -10` (`alignframes.cpp:35`).
const SRF_NO_VAL: i32 = -10;

/// `b3dutil.h:59`.
const OUTPUT_TYPE_MRC: i32 = 2;
/// `iimage.h:67`.
const FEI_EER_METADATA_TAG: i32 = 65001;
/// `tiff.h`: `TIFFTAG_ORIENTATION` and its values.
const TIFFTAG_ORIENTATION: i32 = 274;
const ORIENTATION_TOPLEFT: i16 = 1;
const ORIENTATION_TOPRIGHT: i16 = 2;
const ORIENTATION_BOTRIGHT: i16 = 3;
const ORIENTATION_BOTLEFT: i16 = 4;
const ORIENTATION_LEFTTOP: i16 = 5;
const ORIENTATION_RIGHTTOP: i16 = 6;
const ORIENTATION_RIGHTBOT: i16 = 7;
const ORIENTATION_LEFTBOT: i16 = 8;

/// Indexes of the four member headers in [`AliFrame::m_heads`].
const MAIN_HEAD: usize = 0;
const UNWGT_HEAD: usize = 1;
const EVEN_HEAD: usize = 2;
const ODD_HEAD: usize = 3;

/// `exitError` with the source's variadic format.
macro_rules! exit_error_fmt {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {
        exit_error(&c_format_bytes($fmt, &[$($arg),*]))
    };
}

/// `printf` with the source's format, to the C stdout stream.
macro_rules! printf {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = ImodFile::Stdout.write_all(&c_format_bytes($fmt, &[$($arg),*]));
    }};
}

/// `B3DNINT(a)`: `(int)floor((a) + 0.5)`.
macro_rules! b3dnint {
    ($a:expr) => {
        (($a) as f64 + 0.5).floor() as i32
    };
}

/// `B3DMIN(a,b)`: `((a) < (b) ? (a) : (b))`.
macro_rules! b3dmin {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a < b { a } else { b }
    }};
}

/// `B3DMAX(a,b)`: `((a) > (b) ? (a) : (b))`.
macro_rules! b3dmax {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a > b { a } else { b }
    }};
}

/// The bytes of a frame buffer: the C's `unsigned char *` view of the
/// `malloc`ed storage.
fn buf_bytes(storage: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            storage.as_ptr().cast::<u8>(),
            std::mem::size_of_val(storage),
        )
    }
}

/// Mutable form of [`buf_bytes`].
fn buf_bytes_mut(storage: &mut [f32]) -> &mut [u8] {
    let len = std::mem::size_of_val(storage);
    unsafe { std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast::<u8>(), len) }
}

/// The `void *` plus mode a frame buffer is handed to `FrameAlign` as.  The
/// storage is `f32`-aligned, so every typed view is aligned.
fn frame_data(storage: &[f32], mode: i32, nxy: usize) -> FrameData<'_> {
    let ptr = storage.as_ptr();
    unsafe {
        match mode {
            MRC_MODE_BYTE => FrameData::Byte(std::slice::from_raw_parts(ptr.cast::<u8>(), nxy)),
            MRC_MODE_SHORT => FrameData::Short(std::slice::from_raw_parts(ptr.cast::<i16>(), nxy)),
            MRC_MODE_USHORT => {
                FrameData::UShort(std::slice::from_raw_parts(ptr.cast::<u16>(), nxy))
            }
            _ => FrameData::Float(std::slice::from_raw_parts(ptr, nxy)),
        }
    }
}

/// Storage for a `B3DMALLOC(unsigned char, nbytes)` frame buffer.
fn frame_storage(nbytes: usize) -> Vec<f32> {
    vec![0.; nbytes.div_ceil(4)]
}

/// A C string held in a fixed array: the bytes before the first NUL.
fn c_str(bytes: &[u8]) -> &[u8] {
    &bytes[..bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len())]
}

/// `strstr(haystack, needle)` as an index.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > haystack.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// `std::string::find_last_of(chars)`.
fn find_last_of(s: &[u8], chars: &[u8]) -> Option<usize> {
    s.iter().rposition(|c| chars.contains(c))
}

/// A path from the bytes of a C file name.
fn os_path(name: &[u8]) -> &std::path::Path {
    std::path::Path::new(std::ffi::OsStr::from_bytes(name))
}

/// `strncpy(label, src, MRC_LABEL_SIZE)`: copy and zero-pad.
fn strncpy_label(label: &mut [u8; MRC_LABEL_SIZE], src: &[u8]) {
    let src = c_str(src);
    let n = src.len().min(MRC_LABEL_SIZE);
    label[..n].copy_from_slice(&src[..n]);
    label[n..].fill(0);
}

/// `imodUsageHeader` in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// Class `AliFrame` (`alignframes.h:16`).  Members in declaration order.
pub struct AliFrame {
    pub m_file_copies: [*mut ImodImageFile; MAX_READ_THREADS],
    pub m_in_fp: Option<ImodFile>,
    pub m_in_head: MrcHeader,
    /// `mMainHead, mUnwgtHead, mEvenHead, mOddHead`.
    pub m_heads: [MrcHeader; 4],
    /// `MrcHeader *mOutHeads[4]`: index into `m_heads`.
    pub m_out_heads: [usize; 4],
    pub m_out_names: [Option<Vec<u8>>; 4],
    pub m_parallel_read: bool,
    pub m_names_from_mdoc: bool,
    pub m_rel_frame_starts_found: bool,
    pub m_doing_frame_ts: bool,
    pub m_getting_frc: bool,
    pub m_nx: i32,
    pub m_ny: i32,
    pub m_num_read_threads: i32,
    pub m_in_data_size: i32,
    pub m_debug: i32,
    pub m_num_in_files: i32,
    pub m_dose_accumulates: i32,
    pub m_nx_gain: i32,
    pub m_ny_gain: i32,
    pub m_extra_has_gain_ref: i32,
    pub m_rotation_flip: i32,
    pub m_cor_def_binning: i32,
    pub m_ignore_zvalue: i32,
    pub m_sum_rotation_flip: i32,
    pub m_frames_are_eer: bool,
    pub m_antialias_eer: bool,
    pub m_num_out_files: i32,
    pub m_use_gpu: i32,
    pub m_gpu_flags: i32,
    pub m_test_mode: i32,
    pub m_defer_sum: i32,
    pub m_trunc_limit: f32,
    pub m_gpu_mem_limit: f32,
    pub m_memory_limit: f32,
    pub m_max_data_size: i32,
    pub m_refine_at_end: i32,
    pub m_group_size: i32,
    pub m_use_block_group: bool,
    pub m_max_num_z: i32,
    pub m_hybrid_shifts: i32,
    pub m_start_assess: i32,
    pub m_min_binning_to_test: i32,
    pub m_num_filt_tests: [i32; MAX_BINNINGS],
    pub m_do_spline: i32,
    pub m_full_data_size: i32,
    pub m_num_hold_full: i32,
    pub m_num_bin_tests: i32,
    pub m_num_all_vs_all: i32,
    pub m_sum_pad_size: f32,
    pub m_align_pad_size: f32,
    pub m_full_pad_size: f32,
    pub m_sum_in_one_pass: bool,
    pub m_mdoc_xsize: i32,
    pub m_mdoc_ysize: i32,
    pub m_mdoc_pixel: f32,
    pub m_are_feiframes: bool,

    pub m_wall_read: f64,
    pub m_in_files: Vec<Vec<u8>>,
    /// `unsigned char *mPartialScanBufs[3]`; an empty `Vec` is `NULL`.
    pub m_partial_scan_bufs: [Vec<f32>; 3],
    pub m_zin_partial_bufs: [i32; 3],
    pub m_partial_thresh: [f32; 2],
    pub m_total_dose: f32,
    pub m_gain_name: Option<Vec<u8>>,
    pub m_defect_name: Option<Vec<u8>>,
    pub m_default_byte_scale: f32,
    pub m_initial_dose: f32,
    pub m_dose_scaling: f32,
    pub m_num_sect: i32,
    pub m_adoc_ind: i32,
    pub m_num_mdoc_sect: i32,
    pub m_dose_file_type: i32,
    pub m_max_frame_doses: i32,
    pub m_num_bidir: i32,
    pub m_cam_size_x: i32,
    pub m_cam_size_y: i32,
    pub m_dose_from_mdoc: Vec<f32>,
    pub m_prior_from_mdoc: Vec<f32>,
    pub m_total_dose_vec: Vec<f32>,
    pub m_prior_dose_vec: Vec<f32>,
    pub m_frame_doses: Vec<f32>,
    pub m_zero_dose_thresh: f32,
    pub m_zero_dose_accum: f32,
    pub m_temp_val1: Vec<f32>,
    pub m_reweight_ones: Vec<f32>,
    /// `float *mReweightFilt`, which only ever points at `mReweightOnes`.
    pub m_reweight_filt: bool,
    pub m_iz_piece: Vec<i32>,
    pub m_set_starts: Vec<i32>,
    pub m_saved_frames: Vec<i32>,
    pub m_num_in_sets: Vec<i32>,
    pub m_frame_dose_lines: Vec<Vec<u8>>,
    pub m_fixed_frame_doses: Vec<u8>,
    /// `Islice *mGainSlice`: its float data.
    pub m_gain_slice: Option<Rc<Vec<f32>>>,
    pub m_dark_slice: Option<Islice>,
    pub m_defects: Rc<CameraDefects>,
    pub m_defect_string: String,
    pub m_fei_defect_pad: i32,
    pub m_super_fac_for_defects: i32,
    pub m_in_line: [u8; MAX_LINE + 1],
    pub m_tilt_angles: Vec<f32>,
    pub m_tilt_rel_start_frame: Vec<i32>,
    pub m_tilt_rel_end_frame: Vec<i32>,

    /// `static FrameAlign sFA` (`alignframes.cpp:39`).
    pub s_fa: FrameAlign,
}

/// C `main` (`alignframes.cpp:41`).
pub fn alignframes(arguments: &[String]) -> i32 {
    let mut ali = AliFrame::new();
    ali.main(arguments);
    exit(0);
}

/// Rust-only, for in-process runs (`commands::run_in_process`): the EER read
/// properties `main` sets through `tiffSetEERreadProperties` are process-wide
/// statics in `iitif`, and the gain reference registered for antialiased EER
/// reading is this object's memory.  A process run ends with them; an in-process
/// run returns them to their startup values when the object goes, so a later
/// command in the same process does not read EER files with this run's
/// settings or through a dangling gain reference.
impl Drop for AliFrame {
    fn drop(&mut self) {
        tiff_set_eer_read_properties(-1, -999_999, 0);
    }
}

impl Default for AliFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl AliFrame {
    /// `AliFrame::AliFrame()` (`alignframes.cpp:51`).
    ///
    /// The members the constructor does not set are uninitialised in the C;
    /// they are zero here.
    pub fn new() -> AliFrame {
        AliFrame {
            m_file_copies: [std::ptr::null_mut(); MAX_READ_THREADS],
            m_in_fp: None,
            m_in_head: MrcHeader::default(),
            m_heads: [
                MrcHeader::default(),
                MrcHeader::default(),
                MrcHeader::default(),
                MrcHeader::default(),
            ],
            m_out_heads: [MAIN_HEAD, UNWGT_HEAD, MAIN_HEAD, MAIN_HEAD],
            m_out_names: [None, None, None, None],
            m_parallel_read: false,
            m_names_from_mdoc: false,
            m_rel_frame_starts_found: false,
            m_doing_frame_ts: false,
            m_getting_frc: false,
            m_nx: 0,
            m_ny: 0,
            m_num_read_threads: 0,
            m_in_data_size: 0,
            m_debug: 0,
            m_num_in_files: 0,
            m_dose_accumulates: -1,
            m_nx_gain: 0,
            m_ny_gain: 0,
            m_extra_has_gain_ref: 0,
            m_rotation_flip: 0,
            m_cor_def_binning: 1,
            m_ignore_zvalue: 0,
            m_sum_rotation_flip: SRF_NO_VAL,
            m_frames_are_eer: false,
            m_antialias_eer: false,
            m_num_out_files: 1,
            m_use_gpu: -1,
            m_gpu_flags: 0,
            m_test_mode: 0,
            m_defer_sum: 0,
            m_trunc_limit: 0.,
            m_gpu_mem_limit: 0.,
            m_memory_limit: 12.,
            m_max_data_size: 0,
            m_refine_at_end: 0,
            m_group_size: 1,
            m_use_block_group: false,
            m_max_num_z: 0,
            m_hybrid_shifts: 0,
            m_start_assess: -1,
            m_min_binning_to_test: 100,
            m_num_filt_tests: [0; MAX_BINNINGS],
            m_do_spline: 0,
            m_full_data_size: 0,
            m_num_hold_full: 0,
            m_num_bin_tests: 0,
            m_num_all_vs_all: 0,
            m_sum_pad_size: 0.,
            m_align_pad_size: 0.,
            m_full_pad_size: 0.,
            m_sum_in_one_pass: false,
            m_mdoc_xsize: 0,
            m_mdoc_ysize: 0,
            m_mdoc_pixel: 0.,
            m_are_feiframes: false,
            m_wall_read: 0.,
            m_in_files: Vec::new(),
            m_partial_scan_bufs: [Vec::new(), Vec::new(), Vec::new()],
            m_zin_partial_bufs: [-1, -1, -1],
            m_partial_thresh: [0., 0.],
            m_total_dose: 0.,
            m_gain_name: None,
            m_defect_name: None,
            m_default_byte_scale: 30.,
            m_initial_dose: 0.,
            m_dose_scaling: 1.,
            m_num_sect: 0,
            m_adoc_ind: -1,
            m_num_mdoc_sect: 0,
            m_dose_file_type: -1,
            m_max_frame_doses: 0,
            m_num_bidir: 0,
            m_cam_size_x: 0,
            m_cam_size_y: 0,
            m_dose_from_mdoc: Vec::new(),
            m_prior_from_mdoc: Vec::new(),
            m_total_dose_vec: Vec::new(),
            m_prior_dose_vec: Vec::new(),
            m_frame_doses: Vec::new(),
            m_zero_dose_thresh: 0.,
            m_zero_dose_accum: 0.,
            m_temp_val1: Vec::new(),
            m_reweight_ones: Vec::new(),
            m_reweight_filt: false,
            m_iz_piece: Vec::new(),
            m_set_starts: Vec::new(),
            m_saved_frames: Vec::new(),
            m_num_in_sets: Vec::new(),
            m_frame_dose_lines: Vec::new(),
            m_fixed_frame_doses: Vec::new(),
            m_gain_slice: None,
            m_dark_slice: None,
            m_defects: Rc::new(CameraDefects::new()),
            m_defect_string: String::new(),
            m_fei_defect_pad: -1,
            m_super_fac_for_defects: 0,
            m_in_line: [0; MAX_LINE + 1],
            m_tilt_angles: Vec::new(),
            m_tilt_rel_start_frame: Vec::new(),
            m_tilt_rel_end_frame: Vec::new(),
            s_fa: FrameAlign::new(),
        }
    }

    /// `AliFrame::main` (`alignframes.cpp:118`): "A BIG MAIN METHOD".
    #[allow(clippy::cognitive_complexity)]
    pub fn main(&mut self, arguments: &[String]) {
        let progname_owned = imod_prog_name(arguments.first().map_or("", String::as_str));
        let progname = progname_owned.as_bytes();
        let mut filename: Vec<u8>;
        let mut xf_ext: Option<Vec<u8>> = None;
        let mut xf_name: Vec<u8>;
        let mut sstr: Vec<u8> = Vec::new();
        let mut ext_str: Vec<u8>;
        let mut ordered_angles: Vec<f32> = Vec::new();
        let mut bins_to_test = [0i32; MAX_BINNINGS];
        let mut vary_radius2 = [[0f32; MAX_FILTERS]; MAX_BINNINGS];
        let mut vary_sigma2 = [[0f32; MAX_FILTERS]; MAX_BINNINGS];
        let mut num_times_best = [[0i32; MAX_FILTERS]; MAX_BINNINGS];
        let mut title: [u8; MRC_LABEL_SIZE + 1];

        let mut frame_path: Option<Vec<u8>> = None;
        let mut extra_name: Vec<u8> = Vec::new();
        let mut list_name: Option<Vec<u8>> = None;
        let mut frc_name: Option<Vec<u8>> = None;
        let mut stack_name: Option<Vec<u8>> = None;
        let mut mdoc_name: Option<Vec<u8>> = None;
        let mut tilt_name: Option<Vec<u8>> = None;
        let mut open_ts_names: [Option<Vec<u8>>; 4] = [None, None, None, None];
        let mut out_fps: [Option<ImodFile>; 4] = [None, None, None, None];
        let mut title_descs: [Option<Vec<u8>>; 4] = [None, None, None, None];
        let mut extra_fp: Option<ImodFile> = None;
        let mut stack_fp: Option<ImodFile> = None;
        let mut frc_fp: Option<ImodFile> = None;
        let mut plot_fp: Option<ImodFile> = None;
        let mut file_list_fp: Option<ImodFile> = None;
        let mut read_buf: Vec<f32> = Vec::new();
        let mut sum_buf: Vec<f32> = Vec::new();
        let mut need_alloc: i32;
        let mut buf_alloc_size = 0i32;
        let mut stack_head = MrcHeader::default();
        let mut ii_frames: Option<*mut ImodImageFile> = None;
        let mut summed: Vec<f32>;
        let mut rot_sum: Vec<f32> = Vec::new();
        let mut unwgt_sum: Vec<f32> = Vec::new();
        let mut even_sum: Vec<f32> = Vec::new();
        let mut odd_sum: Vec<f32> = Vec::new();
        let default_binnings: [i32; 5] = [2, 3, 4, 6, 8];
        let mut target_ali_size = 1250i32;
        let mut li = LoadInfo::default();
        let mut do_robust: bool;
        let mut copy_shifts: bool;
        let mut was_good_enough = false;
        let scale_to_mean_sd: bool;
        let mut end_reached = false;
        let mut taper_frac = 0.1f32;
        let mut scale = 1.0f32;
        let mut total_scale = 0.0f32;
        let mut use_scale: f32;
        let mut mean_scale = 0.0f32;
        let mut sd_scale = 0.0f32;
        let mut reorder_by_tilt = 1i32;
        let mut num_summary_lines = 2i32;
        let mut k_factor = 4.5f32;
        let mut shift_limit = 20i32;
        let mut anti_filt_type = 4i32;
        let mut sum_bin = 1i32;
        let size_diff_crit = 0.1f32;
        let mut max_max_weight = 0.1f32;
        let mut good_enough = 0.0f32;
        let mut combine_files = 0i32;
        let mut break_set_size = 0i32;
        let mut ref_radius2 = 0.0f32;
        let ref_sigma2: f32;
        let mut skip_checks = 0i32;
        let mut adjust_mdoc = 0i32;
        let mut spline_smooth: i32;
        let mut min_num_for_spline = 20i32;
        let trim_crit = 10i32;
        let mut iter_crit = 0.1f32;
        let mut group_refine = 0i32;
        let mut sum_rfentered = SRF_NO_VAL;
        let mut drop_mean_crit = -1.0e9f32;

        let mut nz = 0i32;
        let mut align_bin = 0i32;
        let mut ind: i32;
        let mut nx_sum: i32;
        let mut ny_sum: i32;
        let mut ix: i32 = 0;
        let mut iy: i32 = 0;
        let mut itest: i32;
        let mut fa_best_filt = 0i32;
        let mut nx_stack = 0i32;
        let mut ny_stack = 0i32;
        let mut stack_mode = 0i32;
        let mut rel_xbin: i32;
        let mut rel_ybin: i32;
        let mut num_avause: i32;
        let mut tiff_orient: i16 = 0;
        let mut num_single_files = 0i32;
        // `startCombine` is read uninitialised natively in the drift report
        // for an ordinary file (BUGS.md); it starts at 0 here.
        let mut start_combine = 0i32;
        let mut end_combine = 0i32;
        let mut adoc_type = 0i32;
        let mut align_bin_in: i32;
        let mut data_size: i32;
        let mut tind: usize;
        let mut x_scale: f32 = 1.;
        let mut y_scale: f32 = 1.;
        let mut z_scale: f32 = 1.;
        let mut rel_binning: f32 = 1.;
        let mut error: f32;
        let mut min_error: f32;
        let mut min_mean = 0f32;
        let mut half_cross = 0f32;
        let mut trunc_use = 0f32;
        let mut quart_cross = 0f32;
        let mut eighth_cross = 0f32;
        let mut half_nyq = 0f32;
        let mut min_pred = 0f32;
        let mut mem_limits = [0f32; 2];
        let mut full_taper_frac = 0.02f32;

        // framealign uses fullTaperFrac as the padding fraction so a default trimming by the
        // same amount will keep the padded align within the original size, good if it is 4K
        let mut trim_frac = full_taper_frac;
        let mut diff: f64;
        let mut min_diff: f64;
        let mut num_avainput = 7i32;
        let min_fractional_ava = 7i32;
        let mut reverse = 0i32;
        let mut start_frame = -1i32;
        let mut end_frame = -1i32;
        let mut warned_two_pass = false;
        let mut frc_delta_r = 0.005f32;
        let mut ring_corrs = [0f32; 510];
        let mut radius1 = 0.0f32;
        let mut radius2 = 0.06f32;
        let mut sigma1 = 0.03f32;
        let mut sigma2 = 0.0086f32;
        let mut x_shifts: Vec<f32>;
        let mut y_shifts: Vec<f32>;
        let mut best_xshifts: Vec<f32>;
        let mut best_yshifts: Vec<f32>;
        let mut raw_xshifts: Vec<f32>;
        let mut raw_yshifts: Vec<f32>;
        let mut best_xraw: Vec<f32>;
        let mut best_yraw: Vec<f32>;
        let als_bin_entered: i32;
        let target_entered: i32;
        let mut ierr: i32;
        let mut iz: i32 = 0;
        let mut z_start = 0i32;
        let mut z_end = 0i32;
        let mut z_dir = 1i32;
        let mut num_opt_args = 0i32;
        let mut num_non_opt_args = 0i32;
        let mut out_mode = 0i32;
        let mut nx_out: i32;
        let mut ny_out: i32;
        let mut num_in_by_opt = 0i32;
        let mut ifile: i32;
        let mut out_sec_num = 0i32;
        let mut num_varies: i32;
        let mut ind_best_bin = 0i32;
        let mut ind_best_filt = 0i32;
        let mut slide_grp_size: i32;
        let mut use_ind: i32;
        let mut summing_mode = 0i32;
        let mut num_test_loops: i32;
        let mut use_start: i32;
        let mut use_end: i32;
        let mut ind_bin_use: i32;
        let mut ind_filt_use: i32;
        let mut num_filt_use: i32;
        let mut num_done = 0i32;
        let mut num_fetch = 0i32;
        let mut filt: i32;
        let mut stack_bin: i32;
        let mut out_num: i32;
        let mut num_vals = 0i32;
        let mut num_found = 0i32;
        let mut pix_temp = 0f32;
        let phys_mem: f32;
        let mut nz_align = 0i32;
        let mut group_end = 0i32;
        let mut group_start = 0i32;
        let mut iz_low = 0i32;
        let mut iz_high = 0i32;
        let mut use_mode: i32;
        let mut group: i32;
        let mut block_grp_size: i32;
        let mut end_assess = -1i32;
        let mut num_frame_use: i32;
        let mut num_sets = 0i32;
        let mut min_set = 0i32;
        let mut max_set = 0i32;
        let mut file_has_tilts: i32;
        let mut max_read_threads = 1i32;
        let mut extra_has_tilts = 0i32;
        let mut extra_has_axis_pix = 0i32;
        let mut num_all_sets: i32;
        let mut min_set_size: i32;
        let mut max_set_size = 0i32;
        let mut super_file: i32;
        let mut set_in_file: i32;
        let mut iz_read: i32;
        let mut num_undropped_sets: i32;
        let mut skipped_frame = [-1i32; 2];
        let mut original_zval: i32;
        let mut kernel_scale = 1i32;
        let mut starting_file: i32;
        let mut ending_file: i32;
        let mut num_files_to_do: i32;
        let mut max_exclude: i32;
        let mut drift_loop: i32;
        let mut num_drift_loop: i32;
        let mut file_axis = 0f32;
        let mut extra_axis = 0f32;
        let mut file_pix = 0f32;
        let mut extra_pix_size = 0f32;
        let mut option_pix_size = 0.0f32;
        let mut axis_angle = -999.0f32;
        let mut has_extra: bool;
        let entered_scale: bool;
        let entered_mode: bool;
        let mut all_pos: bool;
        let mut all_neg: bool;
        let mut get_need_rf: bool;
        let mut do_abbrev: bool;
        let suppress_initial_shifts: bool;
        let mut excluding_initial = false;
        let mut tilts_vary = true;
        let dropping_by_mean: bool;
        let mut changed_set_order = false;
        let mut rot_flip_entered = false;
        let mut non_imod_mrcframes = false;
        let mut ref_names_from_titles = 0i32;
        let mut even_odd_ouput = 0i32;
        let mut do_unweight = 0i32;
        let use_shr_mem = 0i32;
        let mut eer_zbinning = 10i32;
        let mut eer_super_res = 1i32;
        let mut eer_antialias = 1i32;
        let mut eer_flags: i32;
        let mut stack_bin_x: f32;
        let mut stack_bin_y: f32;
        let mut drift_max_frac_num = 0.2f32;
        let mut drift_max_dist = 0.0f32;
        let mut extra_tilts: Vec<f32> = Vec::new();
        let mut mean_from_mdoc: Vec<f32> = Vec::new();
        let mut temp_min: Vec<f32>;
        let mut temp_max: Vec<f32>;
        let mut set_order_index: Vec<i32> = Vec::new();
        let mut extra_buf: Vec<f32> = Vec::new();
        let mut extra_buf_size = 0i32;
        let mut res_mean = [0f32; MAX_FILTERS + 1];
        let mut pred_mean = [0f32; MAX_FILTERS + 1];
        let mut mean_res_max = [0f32; MAX_FILTERS + 1];
        let mut max_res_max = [0f32; MAX_FILTERS + 1];
        let mut mean_raw_max = [0f32; MAX_FILTERS + 1];
        let mut max_raw_max = [0f32; MAX_FILTERS + 1];
        let mut smooth_dist = [0f32; MAX_FILTERS + 1];
        let mut raw_dist = [0f32; MAX_FILTERS + 1];
        let mut tmin = 0f32;
        let mut tmax = 0f32;
        let mut tmean: f32;
        let mut already_scaled_by: f32;
        let mut tsd = 0f32;
        let mut scale_fac: f32;
        let mut add_fac: f32;

        // Dose weighting variables
        let mut dose_afac = 0.0f32;
        let mut dose_bfac = 0.0f32;
        let mut dose_cfac = 0.0f32;
        let mut prior_temp: f32;
        let mut sum_of_doses = 0f32;
        let mut sum_of_frames: i32;

        // The xf file of a frame set run: open across sets (BUGS.md).
        let mut xf_open = false;

        // Fallbacks from    ../manpages/autodoc2man 2 1 alignframes
        let num_options = 88;
        let options: [&[u8]; 88] = [
            b"input:InputFile:FNM:",
            b"output:OutputImageFile:FN:",
            b"list:ListOfInputFiles:FN:",
            b"break:BreakFramesIntoSets:I:",
            b"saved:SavedFrameListFile:FN:",
            b"gap:MaxGapWithinFrameSet:I:",
            b"skip:SkipFileChecks:B:",
            b"stack:CorrespondingStack:FN:",
            b"mdoc:MetadataFile:FN:",
            b"path:PathToFramesInMdoc:CH:",
            b"ignore:IgnoreZvaluesInMdoc:B:",
            b"adjust:AdjustAndWriteMdoc:B:",
            b"reorder:ReorderByTiltAngle:I:",
            b"pixel:PixelSize:F:",
            b"eer:EERSuperResZSumPadding:IT:",
            b"aaeer:ReadEERWithAntialiasing:I:",
            b"super:SuperGainFactorFile:FN:",
            b"binning:AlignAndSumBinning:IP:",
            b"target:TargetAlignSize:I:",
            b"frames:StartingEndingFrames:IP:",
            b"partial:PartialFrameThresholds:FP:",
            b"drift:DriftLimitDistAndNumber:FP:",
            b"sets:RangeOfSetsToDo:IP:",
            b"ddrop:DropAndReplacementDoses:FP:",
            b"mdrop:DropSetIfMeanBelow:F:",
            b"mode:ModeToOutput:I:",
            b"scale:ScalingOfSum:F:",
            b"total:TotalScalingOfData:F:",
            b"meansd:MeanAndSDtoScaleTo:FP:",
            b"rfsum:SumRotationAndFlip:I:",
            b"tilt:TiltAngleFile:FN:",
            b"axis:AxisRotationAngle:F:",
            b"xfext:TransformExtension:CH:",
            b"frc:FRCOutputFile:FN:",
            b"ring:RingSpacingForFRC:F:",
            b"evenodd:EvenAndOddSumOutput:I:",
            b"lines:LinesOfAlignSummary:I:",
            b"plottable:PlottableShiftFile:FN:",
            b"nosum:NoSumsOutput:B:",
            b"titles:RefAndDefectFromTitles:B:",
            b"gain:GainReferenceFile:FN:",
            b"rotation:RotationAndFlip:I:",
            b"dark:DarkReferenceFile:FN:",
            b"defect:CameraDefectFile:FN:",
            b"double:DoubleDefectCoords:B:",
            b"imagebinned:ImagesAreBinned:F:",
            b"truncate:TruncateAbove:F:",
            b"pair:PairwiseFrames:I:",
            b"reverse:ReverseOrder:B:",
            b"shift:ShiftLimit:I:",
            b"group:GroupSize:I:",
            b"radius2:FilterRadius2:F:",
            b"vary:VaryFilter:FAM:",
            b"hybrid:UseHybridShifts:B:",
            b"refine:RefineAlignment:I:",
            b"rgroup:RefineWithGroupSums:B:",
            b"stop:StopIterationsAtShift:F:",
            b"rrad2:RefineRadius2:F:",
            b"smooth:MinForSplineSmoothing:I:",
            b"gpu:UseGPU:I:",
            b"memory:MemoryLimitGB:FA:",
            b"dtype:TypeOfDoseFile:I:",
            b"dfile:DoseWeightingFile:FN:",
            b"dtotal:FixedTotalDose:F:",
            b"dframe:FixedFrameDoses:F:",
            b"dprior:InitialPriorDose:F:",
            b"bidir:BidirectionalNumViews:I:",
            b"accum:DoseAccumulates:I:",
            b"normalize:NormalizeDoseWeighting:B:",
            b"volt:Voltage:I:",
            b"optimal:OptimalDoseScaling:F:",
            b"critical:CriticalDoseFactors:FT:",
            b"unweight:UnweightedOutputFile:FN:",
            b"test:TestBinnings:IA:",
            b"assess:AssessWithFrames:IP:",
            b"good:GoodEnoughError:F:",
            b"weight:MaxResidualWeight:F:",
            b"trim:TrimFraction:F:",
            b"taper:TaperFraction:F:",
            b"antialias:AntialiasFilter:I:",
            b"radius1:FilterRadius1:F:",
            b"sigma1:FilterSigma1:F:",
            b"sigma2:FilterSigma2:F:",
            b"kfactor:KFactorForFits:F:",
            b"debug:DebugOutput:I:",
            b"flags:FlagsForGPU:I:",
            b"shrmem:ShrMemTest:B:",
            b"help:usage:B:",
        ];

        // Startup with fallback
        let argv = arguments
            .iter()
            .map(|value| value.as_bytes().to_vec())
            .collect::<Vec<_>>();
        pip_read_or_parse_options(
            argv.len() as i32,
            &argv,
            &options,
            num_options,
            progname,
            2,
            1,
            1,
            &mut num_opt_args,
            &mut num_non_opt_args,
            Some(imod_usage_header_for_pip),
        );

        // Get output file and number of input files
        pip_get_boolean(b"NoSumsOutput", &mut self.m_test_mode);
        pip_number_of_entries(b"InputFile", &mut num_in_by_opt);
        self.m_num_in_files = num_in_by_opt + num_non_opt_args;
        let mut out_name0 = Vec::new();
        if pip_get_string(b"OutputImageFile", &mut out_name0) != 0 {
            if self.m_test_mode == 0 {
                if num_non_opt_args == 0 {
                    exit_error(b"No output file specified");
                }
                self.m_num_in_files -= 1;
                self.m_out_names[0] = pip_get_non_option_arg(num_non_opt_args - 1).ok();
            }
        } else {
            self.m_out_names[0] = Some(out_name0);
            if self.m_test_mode != 0 {
                exit_error(b"No output file should be specified when not making sums");
            }
        }

        get_dflt_eersumming_from_env(&mut eer_super_res, &mut eer_zbinning);
        pip_get_three_integers(
            b"EERSuperResZSumPadding",
            &mut eer_super_res,
            &mut eer_zbinning,
            &mut self.m_fei_defect_pad,
        );
        ierr = tiff_get_max_eer_super_res();
        if eer_super_res < 0 || eer_super_res > ierr {
            exit_error_fmt!(
                "Super-resolution for EER files must be between 0 and %d",
                CArg::Int(ierr as i64)
            );
        }
        if eer_zbinning == 0 {
            exit_error(b"Summing of frames for EER files must be non-zero");
        }
        pip_get_integer(b"ReadEERWithAntialiasing", &mut eer_antialias);
        eer_flags = 0;
        if eer_antialias > 0 {
            eer_flags |= IIFLAG_ANTIALIAS_EER;
        }
        if eer_antialias == 1 {
            eer_flags |= IIFLAG_EER_USE_LANCZOS;
        }
        if self.m_fei_defect_pad < 0 {
            self.m_fei_defect_pad = 1;
            if eer_antialias > 0 && eer_super_res < 2 {
                self.m_fei_defect_pad = if eer_super_res < -2 { 40 } else { 20 };
            }
        }

        tiff_set_eer_read_properties(eer_super_res, eer_zbinning, eer_flags);

        let mut list_temp = Vec::new();
        if pip_get_string(b"ListOfInputFiles", &mut list_temp) == 0 {
            if self.m_num_in_files != 0 {
                exit_error(b"You cannot enter input files as arguments with the -list option");
            }
            file_list_fp = ImodFile::open(os_path(&list_temp), "r");
            if file_list_fp.is_none() {
                exit_error_fmt!(
                    "Could not open list of input files, %s",
                    CArg::Bytes(&list_temp)
                );
            }
            list_name = Some(list_temp);
            self.m_num_in_files = 2000000000;
        }
        pip_get_integer(b"BreakFramesIntoSets", &mut break_set_size);
        if break_set_size != 0 && break_set_size < 2 {
            exit_error(b"The entry for -break must be at least 2");
        }
        pip_get_boolean(b"SkipFileChecks", &mut skip_checks);
        if skip_checks != 0 {
            ii_assume_dmfile_matches(1);
        }

        self.m_doing_frame_ts = self.read_analyze_saved_frame_list(break_set_size);

        // See if further trimming of frames is desired
        {
            let (t0, t1) = self.m_partial_thresh.split_at_mut(1);
            if pip_get_two_floats(b"PartialFrameThreshold", &mut t0[0], &mut t1[0]) == 0
                && !self.m_doing_frame_ts
            {
                exit_error(b"You can enter -partial only with a frame list file");
            }
        }
        for ind in 0..2 {
            if self.m_partial_thresh[ind] >= 1. {
                exit_error(
                    b"The threshold for dropping partial frames is relative and must be less than 1",
                );
            }
        }

        // Find out what auxiliary files are being used
        let mut temp = Vec::new();
        if pip_get_string(b"MetadataFile", &mut temp) == 0 {
            mdoc_name = Some(temp);
        }
        let mut temp = Vec::new();
        if pip_get_string(b"CorrespondingStack", &mut temp) == 0 {
            stack_name = Some(temp);
        }
        if self.m_num_in_files > 0 && mdoc_name.is_some() && stack_name.is_some() {
            exit_error(b"You cannot enter -mdoc with -stack; the mdoc would not be used");
        }
        if self.m_num_in_files == 0 && mdoc_name.is_none() {
            exit_error(
                b"Input file(s) must be specified with arguments, an mdoc file, or a list file",
            );
        }

        // Open the mdoc now and set flag to get names from it if necessary
        let dropping_ierr;
        if let Some(mdoc) = mdoc_name.as_deref() {
            let mut num_sect = 0;
            self.open_mdoc_file(mdoc, &mut num_sect, &mut adoc_type);
            self.m_num_sect = num_sect;
            if !self.m_doing_frame_ts
                && (adoc_get_integer(ADOC_GLOBAL_NAME, 0, b"DataMode", &mut stack_mode) != 0
                    || adoc_get_two_integers(
                        ADOC_GLOBAL_NAME,
                        0,
                        b"ImageSize",
                        &mut nx_stack,
                        &mut ny_stack,
                    ) != 0)
            {
                exit_error(b"Getting data mode or image size from mdoc file");
            }

            self.m_names_from_mdoc = self.m_num_in_files == 0;
            if self.m_names_from_mdoc {
                self.m_num_in_files = self.m_num_sect;
                let mut temp = Vec::new();
                if pip_get_string(b"PathToFramesInMdoc", &mut temp) == 0 {
                    frame_path = Some(temp);
                }
            }
            pip_get_boolean(b"IgnoreZvaluesInMdoc", &mut self.m_ignore_zvalue);
            dropping_ierr = pip_get_two_floats(
                b"DropAndReplacementDoses",
                &mut self.m_zero_dose_thresh,
                &mut self.m_zero_dose_accum,
            );
            if pip_get_float(b"DropSetIfMeanBelow", &mut drop_mean_crit) == 0 && dropping_ierr == 0
            {
                exit_error(b"You cannot enter both -ddrop and -mdrop");
            }
            dropping_by_mean = drop_mean_crit > -1.0e8;

            if (self.m_zero_dose_thresh > 0. || dropping_by_mean) && !self.m_names_from_mdoc {
                exit_error(
                    b"You cannot use the -ddrop or -mdrop option unless filenames come from the mdoc file",
                );
            }

            if self.m_test_mode == 0 {
                pip_get_boolean(b"AdjustAndWriteMdoc", &mut adjust_mdoc);
            }
            if self.m_ignore_zvalue != 0 {
                reorder_by_tilt = 0;
            }
        } else {
            dropping_by_mean = drop_mean_crit > -1.0e8;
        }

        // Process mdoc now if doing zero dose or mean dropping
        num_undropped_sets = self.m_num_in_files;
        if self.m_zero_dose_thresh > 0. || dropping_by_mean {
            if self.m_adoc_ind < 0 {
                exit_error(b"Program error, mdoc was supposed to be open");
            }
            self.m_num_mdoc_sect = self.m_num_sect;
            num_undropped_sets = 0;
            if self.m_zero_dose_thresh > 0. {
                // Dropping by dose; get the doses the first of possible several times, count sets
                ierr = self.get_doses_from_mdoc(adoc_type);
                if ierr != 0 {
                    exit_error_fmt!(
                        "Problems occurred accessing data in the mdoc file: %s",
                        CArg::Str(&b3d_get_error())
                    );
                }
                for ind in 0..self.m_num_sect as usize {
                    if self.m_dose_from_mdoc[ind] >= self.m_zero_dose_thresh {
                        num_undropped_sets += 1;
                    }
                }
            } else {
                // Dropping by mean: get the means from MinMaxmean entry and count sets
                let n = self.m_num_sect as usize;
                mean_from_mdoc.resize(n, 0.);
                temp_min = vec![0.; n];
                temp_max = vec![0.; n];
                self.m_iz_piece.resize(n, 0);
                for iz in 0..n {
                    self.m_iz_piece[iz] = iz as i32;
                }
                if get_metadata_by_key(
                    self.m_adoc_ind,
                    adoc_type,
                    self.m_num_sect,
                    "MinMaxMean",
                    4,
                    &mut temp_min,
                    &mut temp_max,
                    &mut mean_from_mdoc,
                    None,
                    &mut num_vals,
                    &mut num_found,
                    self.m_num_sect,
                    &self.m_iz_piece,
                ) != 0
                {
                    exit_error_fmt!(
                        "Getting mean values from .mdoc file: %s",
                        CArg::Str(&b3d_get_error())
                    );
                }
                if num_found < self.m_num_sect {
                    exit_error(b"Some sections in the .mdoc file have no MinMaxMean entry");
                }
                for ind in 0..n {
                    if mean_from_mdoc[ind] >= drop_mean_crit {
                        num_undropped_sets += 1;
                    }
                }
            }
            let kind: &str = if self.m_zero_dose_thresh > 0. {
                "dose"
            } else {
                "mean"
            };
            if num_undropped_sets == 0 {
                exit_error_fmt!(
                    "The %s is below the threshold for all sections in the .mdoc file",
                    CArg::Str(kind)
                );
            }
            if num_undropped_sets < self.m_num_in_files {
                printf!(
                    "Dropping %d frame sets with %ss below the threshold\n",
                    CArg::Int((self.m_num_in_files - num_undropped_sets) as i64),
                    CArg::Str(kind)
                );
            }
        }

        // Frame subset entries
        if pip_get_two_integers(b"StartingEndingFrames", &mut start_frame, &mut end_frame) == 0 {
            if start_frame <= 0 || end_frame < start_frame {
                exit_error(b"Values for starting and ending frames are out of range");
            }
            if self.m_doing_frame_ts {
                exit_error(b"You cannot use -frame with a saved frame list file");
            }
        }
        if pip_get_two_integers(
            b"AssessWithFrames",
            &mut self.m_start_assess,
            &mut end_assess,
        ) == 0
            && (self.m_start_assess <= 0 || end_assess < self.m_start_assess)
        {
            exit_error(
                b"Values for starting and ending frames to use for assessment are out of range",
            );
        }
        if (break_set_size > 0 || self.m_doing_frame_ts) && self.m_start_assess >= 0 {
            exit_error(b"You cannot enter -assess when breaking frames into sets to sum");
        }
        if break_set_size > 0 && self.m_names_from_mdoc {
            exit_error(
                b"You cannot break frames into sets when input filenames come from an mdoc file",
            );
        }

        // Scaling and mode entries
        pip_get_float(b"TotalScalingOfData", &mut total_scale);
        entered_scale = pip_get_float(b"ScalingOfSum", &mut scale) == 0;
        if entered_scale && total_scale > 0. {
            exit_error(b"You cannot enter both -scale and -total");
        }
        scale_to_mean_sd =
            pip_get_two_floats(b"MeanAndSDtoScaleTo", &mut mean_scale, &mut sd_scale) == 0;
        if scale_to_mean_sd && (entered_scale || total_scale > 0.) {
            exit_error(b"You cannot enter -meansd with -scale or -total");
        }
        already_scaled_by = 1.;
        entered_mode = pip_get_integer(b"ModeToOutput", &mut out_mode) == 0;
        if entered_mode && slice_mode_if_real(out_mode) < 0 {
            exit_error_fmt!(
                "Output mode of %d is not allowed",
                CArg::Int(out_mode as i64)
            );
        }
        if entered_mode {
            out_mode = set_float_output_for_entered_mode(out_mode);
        }

        // Get flag to get reference and defects from frame file title, but also bring in
        // the names if any since they override
        pip_get_integer(b"RefAndDefectFromTitles", &mut ref_names_from_titles);
        if ref_names_from_titles != 0 {
            self.m_rotation_flip = -1;
        }
        let mut temp = Vec::new();
        if pip_get_string(b"GainReferenceFile", &mut temp) == 0 {
            self.m_gain_name = Some(temp);
        }
        let mut temp = Vec::new();
        if pip_get_string(b"CameraDefectFile", &mut temp) == 0 {
            self.m_defect_name = Some(temp);
        }
        rot_flip_entered = pip_get_integer(b"RotationAndFlip", &mut self.m_rotation_flip) == 0;
        let mut temp = Vec::new();
        if pip_get_string(b"TiltAngleFile", &mut temp) == 0 {
            tilt_name = Some(temp);
        }
        pip_get_float(b"AxisRotationAngle", &mut axis_angle);
        let mut temp = Vec::new();
        if pip_get_string(b"CorrespondingStack", &mut temp) == 0 {
            stack_name = Some(temp);
        }
        if pip_get_integer(b"SumRotationAndFlip", &mut self.m_sum_rotation_flip) == 0 {
            sum_rfentered = self.m_sum_rotation_flip;
        }
        if pip_get_integer(b"ReorderByTiltAngle", &mut reorder_by_tilt) == 0
            && self.m_ignore_zvalue != 0
        {
            exit_error(b"You cannot enter both -reorder and -ignore");
        }
        target_entered = 1 - pip_get_integer(b"TargetAlignSize", &mut target_ali_size);
        if target_ali_size < 64 {
            exit_error(b"Target size for align reduction is too small");
        }
        pip_get_integer(b"LinesOfAlignSummary", &mut num_summary_lines);
        suppress_initial_shifts = num_summary_lines < 0;
        num_summary_lines = num_summary_lines.abs();
        num_summary_lines = b3dmax!(1, b3dmin!(3, num_summary_lines));
        let mut temp = Vec::new();
        if pip_get_string(b"FRCOutputFile", &mut temp) == 0 {
            frc_name = Some(temp);
        }
        if frc_name.is_some() && self.m_test_mode != 0 {
            exit_error(b"There is no FRC output available when not making sums");
        }
        pip_get_integer(b"EvenAndOddSumOutput", &mut even_odd_ouput);
        even_odd_ouput = b3dmax!(0, b3dmin!(2, even_odd_ouput));
        if even_odd_ouput != 0 && self.m_test_mode != 0 {
            exit_error(b"There is no even and odd output available when not making sums");
        }
        self.m_getting_frc = self.m_test_mode == 0
            && (frc_name.is_some() || num_summary_lines > 2 || even_odd_ouput != 0);

        // Have to find out if the mdoc has varying tilt angles now
        if mdoc_name.is_some() && stack_name.is_none() && tilt_name.is_none() {
            self.get_angles_and_titles_from_mdoc(tilt_name.as_deref(), axis_angle, true);

            // `*std::max_element` of an empty vector dereferences its end
            // natively (BUGS.md); no angles do not vary.
            if self.m_tilt_angles.is_empty() {
                tilts_vary = false;
            } else {
                let mut tmax_angle = self.m_tilt_angles[0];
                let mut tmin_angle = self.m_tilt_angles[0];
                for &angle in &self.m_tilt_angles[1..] {
                    if tmax_angle < angle {
                        tmax_angle = angle;
                    }
                    if angle < tmin_angle {
                        tmin_angle = angle;
                    }
                }
                tilts_vary = tmax_angle - tmin_angle > 1.;
            }
        }

        // Get dose-weighting related options
        let mut frame_doses_opt: Option<Vec<u8>> = None;
        {
            let mut temp = Vec::new();
            let e3 = pip_get_string(b"FixedFrameDoses", &mut temp);
            let e1 = pip_get_float(b"FixedTotalDose", &mut self.m_total_dose);
            let e2 = pip_get_integer(b"TypeOfDoseFile", &mut self.m_dose_file_type);
            if e3 == 0 {
                frame_doses_opt = Some(temp);
            }
            if e1 + e2 + e3 < 2 {
                exit_error(
                    b"You can enter only one of the dose weighting options -dtype, -dtotal, or -dframe",
                );
            }
        }

        title_descs[0] = Some(b"summed frames".to_vec());
        if self.m_dose_file_type > 0 || self.m_total_dose > 0. || frame_doses_opt.is_some() {
            if self.m_test_mode != 0 {
                exit_error(b"You cannot enter dose weighting options with no summing");
            }
            pip_get_three_floats(
                b"CriticalDoseFactors",
                &mut dose_afac,
                &mut dose_bfac,
                &mut dose_cfac,
            );
            pip_get_integer(b"DoseAccumulates", &mut self.m_dose_accumulates);
            if self.m_dose_accumulates < 0 {
                self.m_dose_accumulates = if stack_name.is_some()
                    || (mdoc_name.is_some() && tilts_vary)
                    || tilt_name.is_some()
                    || self.m_doing_frame_ts
                {
                    1
                } else {
                    0
                };
                printf!(
                    "Assuming that dose %s accumulate between frame sets\n",
                    CArg::Str(if self.m_dose_accumulates != 0 {
                        "DOES"
                    } else {
                        "does NOT"
                    })
                );
            }
            self.process_dose_weighting_options(frame_doses_opt.take(), &mut adoc_type);

            // Get option for unweighted output also
            let mut temp = Vec::new();
            do_unweight = 1 - pip_get_string(b"UnweightedOutputFile", &mut temp);
            if do_unweight != 0 {
                self.m_out_names[self.m_num_out_files as usize] = Some(temp);
                self.m_num_out_files += 1;
            }
            title_descs[1] = Some(b"non-DW sum".to_vec());
        }

        // Set up even and odd files
        if even_odd_ouput != 0 {
            xf_name = self.m_out_names[0].clone().unwrap_or_default();
            ext_str = Vec::new();
            if let Some(found) = find_last_of(&xf_name, b".") {
                let mut t = found;
                if t + 5 >= xf_name.len() && t > 1 {
                    ext_str = xf_name[t..].to_vec();
                    xf_name.truncate(t);
                    t -= 1;
                    if even_odd_ouput > 1 && t > 0 && (xf_name[t] == b'a' || xf_name[t] == b'b') {
                        ext_str.insert(0, xf_name[t]);
                        xf_name.truncate(t);
                    }
                }
            }
            let mut name = xf_name.clone();
            name.extend_from_slice(b"_even");
            name.extend_from_slice(&ext_str);
            let n = self.m_num_out_files as usize;
            self.m_out_names[n] = Some(name);
            title_descs[n] = Some(b"even sum".to_vec());
            self.m_out_heads[n] = EVEN_HEAD;
            self.m_num_out_files += 1;
            let mut name = xf_name.clone();
            name.extend_from_slice(b"_odd");
            name.extend_from_slice(&ext_str);
            let n = self.m_num_out_files as usize;
            self.m_out_names[n] = Some(name);
            title_descs[n] = Some(b"odd sum".to_vec());
            self.m_out_heads[n] = ODD_HEAD;
            self.m_num_out_files += 1;
        }

        // Open and read header of every input file before starting
        num_all_sets = 0;
        min_set_size = 0;
        ind = 0;
        while ind < self.m_num_in_files {
            let Some(next) = self.get_next_filename(
                ind,
                num_in_by_opt,
                file_list_fp.as_mut(),
                frame_path.as_deref(),
                list_name.as_deref().unwrap_or(b""),
                &mut end_reached,
            ) else {
                break;
            };
            filename = next;

            // Save filename and check file
            self.m_in_files.push(filename.clone());
            if (self.m_zero_dose_thresh > 0.
                && self.m_dose_from_mdoc[ind as usize] < self.m_zero_dose_thresh)
                || (dropping_by_mean && mean_from_mdoc[ind as usize] < drop_mean_crit)
            {
                ind += 1;
                continue;
            }

            if ind == 0 || skip_checks == 0 {
                let mut head = MrcHeader::default();
                self.m_in_fp = Some(self.open_and_read_header(
                    &filename,
                    &mut head,
                    b"input image",
                    self.m_test_mode != 0 && ind == self.m_num_in_files - 1,
                ));
                self.m_in_head = head;
                self.check_input_file(
                    &filename,
                    &self.m_in_head,
                    if ind != 0 { self.m_nx } else { 0 },
                    self.m_ny,
                    combine_files,
                );
                data_size = data_size_for_mode(self.m_in_head.mode).map_or(0, |(d, _)| d);
                self.m_max_data_size = b3dmax!(self.m_max_data_size, data_size);

                // Determine EER status now from the first file (first???)
                ii_frames = ii_lookup_file_from_fp(self.m_in_fp.as_ref().unwrap());
                if ind == 0 {
                    if let Some(frames) = ii_frames {
                        let frames = unsafe { &*frames };
                        if frames.file == IIFILE_TIFF && frames.num_frames_in_eerfile > 0 {
                            self.m_frames_are_eer = true;
                            if !rot_flip_entered || self.m_rotation_flip < 0 {
                                self.m_rotation_flip = 0;
                            }
                            self.m_antialias_eer = frames.antialias_eerfilter != 0;
                            if self.m_antialias_eer {
                                kernel_scale = frames.eerkernel_scale;
                            }
                        }
                    }
                }

                // If all these conditions are satisfied, it is possibly and FEI file and if it
                // turns out to be so, we can apply rfsum = -1 properly
                if ind == 0
                    && ii_frames.is_some()
                    && sum_rfentered == -1
                    && unsafe { (*ii_frames.unwrap()).file } == IIFILE_MRC
                    && self.m_in_head.y_inverted == 0
                    && self.m_in_head.imod_stamp != IMOD_MRC_STAMP
                {
                    non_imod_mrcframes = true;

                    // But if there is no mdoc file, look for further signatures of an FEI file
                    if mdoc_name.is_none()
                        && self.m_in_head.nlabl == 0
                        && self.m_in_head.mx * self.m_in_head.my * self.m_in_head.mz == 1
                        && self.m_in_head.amin == 0.
                        && self.m_in_head.amax == 0.
                    {
                        self.m_are_feiframes = true;
                    }
                }
            }

            if ind == 0 && ref_names_from_titles != 0 {
                self.check_titles_for_ref_names(&filename, ii_frames);
            }

            // Set size and mode from first file.  Default to not do bytes as output
            if ind == 0 {
                if !entered_mode {
                    out_mode = if self.m_in_head.mode == MRC_MODE_BYTE {
                        MRC_MODE_SHORT
                    } else {
                        self.m_in_head.mode
                    };
                }
                self.m_nx = self.m_in_head.nx;
                self.m_ny = self.m_in_head.ny;
                if self.m_doing_frame_ts {
                    if self.m_in_head.nz as usize != self.m_saved_frames.len() {
                        exit_error_fmt!(
                            "The number of frames (%d) does not match the number of entries in the saved frame list (%d)",
                            CArg::Int(self.m_in_head.nz as i64),
                            CArg::Int(self.m_saved_frames.len() as i64)
                        );
                    }
                    nx_stack = self.m_nx;
                    ny_stack = self.m_ny;
                }

                for tind in 0..self.m_num_out_files as usize {
                    self.m_heads[self.m_out_heads[tind]] = self.m_in_head.clone();
                }
                (x_scale, y_scale, z_scale) = mrc_get_scale(&self.m_in_head);

                // Also get the rotation/flip if needed
                get_need_rf = (self.m_rotation_flip < -1 && self.m_sum_rotation_flip == SRF_NO_VAL)
                    || (self.m_sum_rotation_flip < 0 && self.m_sum_rotation_flip != SRF_NO_VAL);
                if self.m_frames_are_eer
                    && self.m_sum_rotation_flip < 0
                    && self.m_sum_rotation_flip != SRF_NO_VAL
                {
                    // Getting orientation from a TIFF file and converting to r/f value
                    if unsafe {
                        tiff_get_field(
                            ii_frames.unwrap_or(std::ptr::null_mut()),
                            TIFFTAG_ORIENTATION,
                            (&mut tiff_orient as *mut i16).cast(),
                        )
                    } <= 0
                    {
                        exit_error(
                            b"Cannot find orientation tag in TIFF header of first input file",
                        );
                    }
                    match tiff_orient {
                        ORIENTATION_TOPLEFT => self.m_sum_rotation_flip = 0,
                        ORIENTATION_TOPRIGHT => self.m_sum_rotation_flip = 4,
                        ORIENTATION_BOTRIGHT => self.m_sum_rotation_flip = 2,
                        ORIENTATION_BOTLEFT => self.m_sum_rotation_flip = 6,
                        ORIENTATION_LEFTTOP => self.m_sum_rotation_flip = 5,
                        ORIENTATION_RIGHTTOP => self.m_sum_rotation_flip = 3,
                        ORIENTATION_RIGHTBOT => self.m_sum_rotation_flip = 7,
                        ORIENTATION_LEFTBOT => self.m_sum_rotation_flip = 1,
                        _ => exit_error_fmt!(
                            "Unknown value %d for orientation tag in TIFF header",
                            CArg::Int(tiff_orient as i64)
                        ),
                    }
                } else if self.m_rotation_flip < 0 || get_need_rf || total_scale > 0. {
                    for ix in 0..self.m_in_head.nlabl.clamp(0, MRC_NLABELS as i32) as usize {
                        let label = c_str(&self.m_in_head.labels[ix]).to_vec();
                        if self.m_rotation_flip < 0 || get_need_rf {
                            if let Some(pos) = find_bytes(&label, b" r/f ") {
                                let mut end = 0;
                                self.m_rotation_flip =
                                    strtol(&label[pos + 4..], &mut end, 10) as i32;
                                if get_need_rf {
                                    if let Some(pos) = find_bytes(&label, b" need ") {
                                        let mut end = 0;
                                        self.m_sum_rotation_flip =
                                            strtol(&label[pos + 5..], &mut end, 10) as i32;
                                    } else {
                                        self.m_sum_rotation_flip = 0;
                                    }
                                }
                            }
                        }
                        if let Some(pos) = find_bytes(&label, b", scaled by") {
                            let mut end = 0;
                            already_scaled_by = strtod(&label[pos + 11..], &mut end) as f32;
                        }
                    }
                    if self.m_rotation_flip < 0 {
                        exit_error(b"Cannot find r/f entry in header of first input file");
                    }
                }

                if self.m_sum_rotation_flip == SRF_NO_VAL {
                    self.m_sum_rotation_flip = 0;
                }

                // And commit to combining files if one frame and breaking into sets, and disallow
                // frame subsets
                if break_set_size > 0 && self.m_in_head.nz == 1 {
                    combine_files = break_set_size;
                }
                if combine_files > 0 && (start_frame >= 0 || self.m_start_assess >= 0) {
                    exit_error(b"You cannot enter -frames when combining single-frame files");
                }
                if combine_files == 0 && skip_checks != 0 {
                    exit_error(b"You cannot skip file checks unless combining single-frame files");
                }

                // Determine if reading TIFF and if so, get number of threads to use
                if combine_files == 0 {
                    self.m_file_copies[0] = ii_lookup_file_from_fp(self.m_in_fp.as_ref().unwrap())
                        .unwrap_or(std::ptr::null_mut());
                    if self.m_file_copies[0].is_null() {
                        printf!(
                            "WARNING: %s - Could not find iiFile from file pointer to assess whether to read a TIFF file in parallel\n",
                            CArg::Bytes(progname)
                        );
                    }
                    if !self.m_file_copies[0].is_null()
                        && unsafe { (*self.m_file_copies[0]).file } == IIFILE_TIFF
                    {
                        max_read_threads = tiff_num_read_threads(
                            self.m_nx,
                            self.m_ny,
                            unsafe { (*self.m_file_copies[0]).tiff_compression },
                            MAX_READ_THREADS as i32,
                        );
                    }
                }
            }

            // Get frames to use from file and make sure it is legal
            num_frame_use = self.m_in_head.nz;
            if start_frame > 0 {
                num_frame_use = b3dmin!(self.m_in_head.nz, end_frame) + 1 - start_frame;
            }
            if num_frame_use < 1 {
                exit_error_fmt!(
                    "No frames would be included for file %s which has only %d frames",
                    CArg::Bytes(&self.m_in_files[ind as usize]),
                    CArg::Int(self.m_in_head.nz as i64)
                );
            }
            if combine_files == 0 && num_frame_use < break_set_size {
                exit_error_fmt!(
                    "The available frames for file %s is %d, less than the set size of %d",
                    CArg::Bytes(&self.m_in_files[ind as usize]),
                    CArg::Int(num_frame_use as i64),
                    CArg::Int(break_set_size as i64)
                );
            }

            // Check for gain reference if one not entered
            has_extra = self.m_in_head.next != 0
                && extra_is_nbytes_and_flags(
                    self.m_in_head.nint as i32,
                    self.m_in_head.nreal as i32,
                ) == 0;
            iz = 0;
            if has_extra
                && self.m_gain_name.is_none()
                && self.m_in_head.next
                    >= self.m_in_head.nz
                        * 4
                        * (self.m_in_head.nint as i32 + self.m_in_head.nreal as i32)
                        + 4 * self.m_nx * self.m_ny
            {
                iz = 1;
            }
            if ind == 0 {
                self.m_extra_has_gain_ref = iz;
            } else if self.m_extra_has_gain_ref != iz {
                exit_error_fmt!(
                    "All files must have gain references in their extended header if any do; it is missing in %s",
                    CArg::Bytes(&self.m_in_files[ind as usize])
                );
            }

            // Get the min and max set sizes if breaking, or if extra header has tilt angles,
            // or for a frame list file
            min_set = 0;
            if break_set_size > 0 && combine_files == 0 {
                self.min_max_set_size(break_set_size, num_frame_use, &mut min_set, &mut max_set);
                num_sets = num_frame_use / break_set_size;
            } else if self.m_doing_frame_ts {
                num_sets = self.m_num_in_sets.len() as i32;
                min_set = self.m_in_head.nz;
                max_set = 0;

                // The source reuses the file loop's `ind` here; with a frame
                // list there is exactly one input file, so the loop still ends.
                ind = 0;
                while ind < num_sets {
                    min_set = b3dmin!(min_set, self.m_num_in_sets[ind as usize]);
                    max_set = b3dmax!(max_set, self.m_num_in_sets[ind as usize]);
                    ind += 1;
                }
            } else if combine_files == 0 && has_extra {
                let mut fp = self.m_in_fp.take().unwrap();
                let mut head = std::mem::take(&mut self.m_in_head);
                ierr = self.analyze_extra_header(
                    &mut fp,
                    &mut head,
                    start_frame,
                    end_frame,
                    break_set_size > 0,
                    &mut extra_buf,
                    &mut extra_buf_size,
                    &mut extra_tilts,
                    &mut min_set,
                    &mut max_set,
                    &mut file_axis,
                    &mut file_pix,
                );
                self.m_in_head = head;
                self.m_in_fp = Some(fp);

                // Save the axis rotation and pixel size if any, and make sure all files are
                // consistent
                file_has_tilts = if self.m_names_from_mdoc || break_set_size > 0 {
                    0
                } else {
                    ierr / 2
                };
                if ind == 0 {
                    extra_has_axis_pix = ierr % 2;
                    if ierr % 2 != 0 {
                        extra_axis = file_axis;
                        extra_pix_size = file_pix;
                    }
                    extra_has_tilts = file_has_tilts;
                }

                if extra_has_axis_pix != ierr % 2
                    || (extra_has_axis_pix != 0
                        && ((extra_axis - file_axis).abs() as f64 > 0.01
                            || (extra_pix_size - file_pix).abs() as f64 > 0.01))
                {
                    exit_error_fmt!(
                        "All files must have the same axis rotation angles and pixel sizes in the extended header if any do; they differ in %s",
                        CArg::Bytes(&self.m_in_files[ind as usize])
                    );
                }
                if extra_has_tilts != file_has_tilts {
                    exit_error_fmt!(
                        "All files must have valid tilt angles in extended header if any do and if -break is not entered; they are invalid in %s",
                        CArg::Bytes(&self.m_in_files[ind as usize])
                    );
                }
                if extra_has_tilts != 0 {
                    num_sets = extra_tilts.len() as i32;
                    if tilt_name.is_none() && stack_name.is_none() {
                        self.m_tilt_angles.extend_from_slice(&extra_tilts);
                    }
                }

                // Assign the axis angle to be output if not set already
                if extra_has_axis_pix != 0 && axis_angle < -990. {
                    axis_angle = file_axis;
                }
            }

            // Keep track of minimum and maximum set size if sets came out either way
            if min_set != 0 {
                if min_set_size == 0 {
                    min_set_size = min_set;
                    max_set_size = max_set;
                } else {
                    min_set_size = b3dmin!(min_set_size, min_set);
                    max_set_size = b3dmax!(max_set_size, max_set);
                }
                num_all_sets += num_sets;
            }
            if ind == 0 || skip_checks == 0 {
                if let Some(mut fp) = self.m_in_fp.take() {
                    ii_fclose(&mut fp);
                }
            }

            self.m_max_num_z = b3dmax!(self.m_max_num_z, num_frame_use);
            self.m_max_frame_doses = b3dmax!(self.m_max_frame_doses, self.m_in_head.nz);
            ind += 1;
        }

        // Finish up with processing a list of input files: just fix the # of files
        if file_list_fp.is_some() {
            file_list_fp = None;
            self.m_num_in_files = self.m_in_files.len() as i32;
            if self.m_num_in_files == 0 {
                exit_error_fmt!(
                    "There were no input files in the list file %s",
                    CArg::Bytes(list_name.as_deref().unwrap_or(b""))
                );
            }
            printf!(
                "%d files in input file list\n",
                CArg::Int(self.m_num_in_files as i64)
            );
        }
        drop(file_list_fp);

        if ref_names_from_titles != 0
            && self.m_gain_name.is_none()
            && self.m_extra_has_gain_ref == 0
        {
            exit_error(b"No gain reference name was found in the frame file titles");
        }

        // Handle combination of single-frame files or breaking frames into sets
        if combine_files != 0 {
            if self.m_num_in_files < combine_files {
                exit_error_fmt!(
                    "The break entry, %d, is bigger than the number of single-frame input files, %d",
                    CArg::Int(combine_files as i64),
                    CArg::Int(self.m_num_in_files as i64)
                );
            }
            num_single_files = self.m_num_in_files;
            let mut max_num_z = 0;
            self.min_max_set_size(
                combine_files,
                self.m_num_in_files,
                &mut ierr,
                &mut max_num_z,
            );
            self.m_max_num_z = max_num_z;
            self.m_max_frame_doses = self.m_max_num_z;
            self.m_num_in_files /= combine_files;
            printf!(
                "%d files will be combined into %d summed images\n",
                CArg::Int(num_single_files as i64),
                CArg::Int(self.m_num_in_files as i64)
            );
        } else if break_set_size > 0 || extra_has_tilts != 0 || self.m_doing_frame_ts {
            num_single_files = self.m_num_in_files;
            self.m_num_in_files = num_all_sets;
            self.m_max_num_z = max_set_size;
            self.m_max_frame_doses = max_set_size;
            if extra_has_tilts != 0 {
                printf!(
                    "Tilt angles from extended header will be used to break frames into sets\n"
                );
            }
            printf!(
                "Frames from %d files will be broken into %d summed images\n",
                CArg::Int(num_single_files as i64),
                CArg::Int(self.m_num_in_files as i64)
            );
        }
        if self.m_extra_has_gain_ref != 0 {
            printf!("Gain reference from extended header will be applied to frames\n");
        }

        starting_file = 1;
        ending_file = self.m_num_in_files;
        if pip_get_two_integers(b"RangeOfSetsToDo", &mut starting_file, &mut ending_file) == 0
            && (self.m_zero_dose_thresh > 0. || dropping_by_mean)
        {
            exit_error(b"You cannot enter the -ddrop or -mdrop option with a range of sets to do");
        }
        if starting_file > ending_file || starting_file < 1 || ending_file > self.m_num_in_files {
            exit_error(b"Starting or ending set number to process is out of range");
        }

        num_files_to_do = ending_file + 1 - starting_file;
        if self.m_zero_dose_thresh > 0. || dropping_by_mean {
            num_files_to_do = num_undropped_sets;
        }

        // Now that number of "files" is known, and maximum number of sections, return to
        // dealing with dose-weighting
        self.unify_dose_information(break_set_size, combine_files, adoc_type);

        // Adjust default memory if physical memory is available, and get
        phys_mem = (b3d_physical_memory() / (1024. * 1024. * 1024.)) as f32;
        if phys_mem > 0. {
            if phys_mem < 16. {
                self.m_memory_limit = (0.75 * phys_mem as f64) as f32;
            }
            if phys_mem > 24. {
                self.m_memory_limit = (0.5 * phys_mem as f64) as f32;
            }
        }
        ind = 0;
        if pip_get_float_array(b"MemoryLimitGB", &mut mem_limits, &mut ind, 2) == 0 {
            self.m_memory_limit = mem_limits[0];
            if (mem_limits[0] as f64) < -0.95
                || (ind > 1 && (mem_limits[1] as f64) < -0.95)
                || (mem_limits[0].abs() as f64) < 0.05
                || (ind > 1 && (mem_limits[1].abs() as f64) < 0.05)
            {
                exit_error(
                    b"You cannot enter a memory limit below -0.95 or between -0.05 and 0.05",
                );
            }
            if mem_limits[0] < 0. {
                if phys_mem == 0. {
                    exit_error(
                        b"You cannot enter a negative CPU memory limit: system memory not available",
                    );
                }
                self.m_memory_limit *= -phys_mem;
            }
            if ind > 1 {
                self.m_gpu_mem_limit = mem_limits[1];
            }
        }

        // Allocate arrays for shifts
        let max_num_z = self.m_max_num_z.max(0) as usize;
        x_shifts = vec![0.; max_num_z];
        y_shifts = vec![0.; max_num_z];
        best_xshifts = vec![0.; max_num_z];
        best_yshifts = vec![0.; max_num_z];
        raw_xshifts = vec![0.; max_num_z];
        raw_yshifts = vec![0.; max_num_z];
        best_xraw = vec![0.; max_num_z];
        best_yraw = vec![0.; max_num_z];

        // Get lots more options
        pip_get_integer(b"PairwiseFrames", &mut num_avainput);
        pip_get_float(b"TaperFraction", &mut taper_frac);
        if taper_frac == 0. {
            full_taper_frac = 0.05;
        }
        pip_get_float(b"TrimFraction", &mut trim_frac);
        pip_get_boolean(b"ReverseOrder", &mut reverse);
        pip_get_integer(b"ShiftLimit", &mut shift_limit);
        pip_get_float(b"TruncateAbove", &mut self.m_trunc_limit);
        if self.m_trunc_limit > 0. {
            self.m_trunc_limit *= kernel_scale as f32;
        }
        let mut temp = Vec::new();
        if pip_get_string(b"TransformExtension", &mut temp) == 0 {
            xf_ext = Some(temp);
        }
        pip_get_integer(b"DebugOutput", &mut self.m_debug);
        pip_get_float(b"KFactorForFits", &mut k_factor);
        pip_get_float(b"MaxResidualWeight", &mut max_max_weight);
        pip_get_float(b"GoodEnoughError", &mut good_enough);
        pip_get_boolean(b"UseHybridShifts", &mut self.m_hybrid_shifts);
        pip_get_float(b"FilterSigma1", &mut sigma1);
        pip_get_float(b"FilterSigma2", &mut sigma2);
        pip_get_float(b"FilterRadius1", &mut radius1);
        pip_get_float(b"FilterRadius2", &mut radius2);
        pip_get_integer(b"RefineAlignment", &mut self.m_refine_at_end);
        pip_get_float(b"RefineRadius2", &mut ref_radius2);
        pip_get_integer(b"AntialiasFilter", &mut anti_filt_type);
        pip_get_float(b"RingSpacingForFRC", &mut frc_delta_r);
        pip_get_integer(b"UseGPU", &mut self.m_use_gpu);
        pip_get_integer(b"GroupSize", &mut self.m_group_size);
        pip_get_boolean(b"RefineWithGroupSums", &mut group_refine);
        pip_get_float(b"StopIterationsAtShift", &mut iter_crit);
        pip_get_float(b"PixelSize", &mut option_pix_size);
        spline_smooth = pip_get_integer(b"MinForSplineSmoothing", &mut min_num_for_spline);
        if min_num_for_spline < 8 {
            spline_smooth = 0;
        }
        anti_filt_type = b3dmax!(1, b3dmin!(6, anti_filt_type));
        if self.m_refine_at_end < 0 {
            exit_error(b"Entry for -refine cannot be negative");
        }
        self.m_num_all_vs_all = num_avainput;
        if self.m_num_all_vs_all < 0 {
            if num_avainput < -4 {
                exit_error(b"The value for the -pair option cannot be more negative than -4");
            }
            if num_avainput == -1 {
                self.m_num_all_vs_all = b3dmin!(MAX_ALL_VS_ALL as i32, self.m_max_num_z + 4);
            } else {
                self.m_num_all_vs_all = b3dmax!(
                    min_fractional_ava,
                    (self.m_max_num_z - 1 - num_avainput) / (-num_avainput)
                );
            }
        }
        self.m_num_all_vs_all = b3dmin!(MAX_ALL_VS_ALL as i32, self.m_num_all_vs_all);
        if self.m_start_assess > 0 && self.m_num_all_vs_all == 0 {
            exit_error(b"You cannot set frames for assessing fits with cumulative correlations");
        }
        if reverse != 0 && (break_set_size > 0 || extra_has_tilts != 0) {
            exit_error(b"You cannot process in reverse when breaking frames into sets");
        }

        // Get default binning for size: set the target size bigger for K3
        min_diff = 1.0e20;
        ierr = ((self.m_nx as f64) * self.m_ny as f64).sqrt() as i32;
        if target_entered == 0 && ((ierr < 6000 && ierr > 4500) || (ierr < 12000 && ierr > 9000)) {
            target_ali_size = (target_ali_size as f64 * 1.25) as i32;
        }
        for ind in 0..default_binnings.len() {
            diff = (target_ali_size as f64
                - ((self.m_nx as f64) * self.m_ny as f64).sqrt() / default_binnings[ind] as f64)
                .abs();
            if diff < min_diff {
                min_diff = diff;
                align_bin = default_binnings[ind];
            }
        }

        // Set output size based on this binning of frames
        align_bin_in = align_bin;
        als_bin_entered =
            1 - pip_get_two_integers(b"AlignAndSumBinning", &mut align_bin_in, &mut sum_bin);
        if align_bin_in == 0 || sum_bin < 1 || align_bin_in > 16 || sum_bin > 16 {
            exit_error(b"Binning value is out of allowed range");
        }
        if align_bin_in > 0 {
            align_bin = align_bin_in;
        }

        // Get tilt angles from file
        self.read_tilt_angle_file(tilt_name.as_deref());

        // Collect information from stack or mdoc file
        if let Some(stack) = stack_name.as_deref() {
            stack_fp = Some(self.open_and_read_header(stack, &mut stack_head, b"stack", false));
            stack_mode = stack_head.mode;
            nx_stack = stack_head.nx;
            ny_stack = stack_head.ny;
            self.m_heads[MAIN_HEAD] = stack_head.clone();
            self.m_heads[UNWGT_HEAD] = stack_head.clone();
            self.m_num_sect = stack_head.nz;
        }

        // Get tilt angles from mdoc if not gotten yet, transfer titles
        if mdoc_name.is_some() {
            if stack_name.is_some() {
                exit_error(b"You cannot enter both a corresponding stack and an mdoc file");
            }
            self.get_angles_and_titles_from_mdoc(tilt_name.as_deref(), axis_angle, false);
        } else if axis_angle > -990. {
            for out_num in 0..self.m_num_out_files {
                self.add_axis_angle_title(out_num, axis_angle);
            }
        }

        // Now we know if it is from FEI, so if it is original MRC, we can set sumRF
        if sum_rfentered == -1
            && self.m_are_feiframes
            && !self.m_frames_are_eer
            && non_imod_mrcframes
        {
            self.m_sum_rotation_flip = 6;
            printf!(
                "Assuming frames are from Thermo/FEI software: setting sum rotation/flip to %d to flip around X\n",
                CArg::Int(self.m_sum_rotation_flip as i64)
            );
        }

        if self.m_sum_rotation_flip < 0 || self.m_sum_rotation_flip > 7 {
            exit_error(b"Inappropriate value of rotation and flip for sum entered");
        }
        nx_sum = (if self.m_sum_rotation_flip % 2 != 0 {
            self.m_ny
        } else {
            self.m_nx
        }) / sum_bin;
        nx_out = nx_sum;
        ny_sum = (if self.m_sum_rotation_flip % 2 != 0 {
            self.m_nx
        } else {
            self.m_ny
        }) / sum_bin;
        ny_out = ny_sum;

        // Now for tilt angles from either a tilt file or an mdoc, deal with a mismatch between
        // angles and frame sets
        if tilt_name.is_some() || mdoc_name.is_some() {
            self.handle_too_many_tilt_angles(progname);
        }

        // Set default index to sets then see if need to reorder: test for monotonic already
        for ind in 0..self.m_num_in_files {
            set_order_index.push(ind);
        }
        if !self.m_tilt_angles.is_empty() && tilts_vary && reorder_by_tilt != 0 {
            all_neg = true;
            all_pos = true;
            for ind in 1..self.m_tilt_angles.len() {
                if self.m_tilt_angles[ind] as f64 > self.m_tilt_angles[ind - 1] as f64 + 0.01 {
                    all_neg = false;
                }
                if (self.m_tilt_angles[ind] as f64) < self.m_tilt_angles[ind - 1] as f64 - 0.01 {
                    all_pos = false;
                }
            }

            // Do not reorder if already all negative and it is not forced by a 2, or already
            // all positive and it is not forced by -2
            if !(all_neg && reorder_by_tilt < 2) && !(all_pos && reorder_by_tilt > -2) {
                rs_sort_indexed_floats(
                    &self.m_tilt_angles,
                    &mut set_order_index,
                    self.m_num_in_files,
                );
                changed_set_order = true;
                if reorder_by_tilt < 0 {
                    let n = self.m_num_in_files as usize;
                    for ind in 0..n / 2 {
                        iz = set_order_index[ind];
                        set_order_index[ind] = set_order_index[n - 1 - ind];
                        set_order_index[n - 1 - ind] = iz;
                    }
                }
            }
        }
        if !self.m_tilt_angles.is_empty() {
            for ind in 0..self.m_num_in_files as usize {
                ordered_angles.push(self.m_tilt_angles[set_order_index[ind] as usize]);
            }
        }

        // Make sure things work out
        if stack_name.is_some() || (mdoc_name.is_some() && !self.m_doing_frame_ts) {
            let what: &str = if stack_name.is_some() {
                "stack"
            } else {
                "mdoc file"
            };
            if stack_mode != MRC_MODE_BYTE && !entered_mode {
                out_mode = stack_mode;
            }
            if self.m_num_sect < self.m_num_in_files && tilt_name.is_none() {
                exit_error_fmt!(
                    "There are fewer sections in the %s (%d) than frame files or sets (%d)",
                    CArg::Str(what),
                    CArg::Int(self.m_num_sect as i64),
                    CArg::Int(self.m_num_in_files as i64)
                );
            }
            if self.m_num_sect > self.m_num_in_files && tilt_name.is_none() {
                printf!(
                    "WARNING: %s - There are fewer frame sets or files (%d) than sections in the %s (%d)\n",
                    CArg::Bytes(progname),
                    CArg::Int(self.m_num_in_files as i64),
                    CArg::Str(what),
                    CArg::Int(self.m_num_sect as i64)
                );
            }

            // Figure out if size works, requires trimming, or implies a binning relative to stack
            if (nx_stack <= nx_out && ny_stack > ny_out)
                || (nx_stack > nx_out && ny_stack <= ny_out)
            {
                exit_error_fmt!(
                    "The image size from the %s is bigger in one dimension than the frame size",
                    CArg::Str(what)
                );
            }
            ierr = 0;

            // Look for integer binning difference that matches in each direction
            // And make sure each size is close enough after scaling
            if nx_stack <= nx_out {
                rel_xbin = b3dnint!(nx_out as f64 / nx_stack as f64);
                rel_ybin = b3dnint!(ny_out as f64 / ny_stack as f64);
                if (rel_xbin as f64 * nx_stack as f64 - nx_out as f64).abs()
                    > size_diff_crit as f64 * nx_out as f64
                    || (rel_ybin as f64 * ny_stack as f64 - ny_out as f64).abs()
                        > size_diff_crit as f64 * ny_out as f64
                    || rel_xbin != rel_ybin
                {
                    ierr = 1;
                }
                rel_binning = (1. / rel_xbin as f64) as f32;
            } else {
                rel_xbin = b3dnint!(nx_stack as f64 / nx_out as f64);
                rel_ybin = b3dnint!(ny_stack as f64 / ny_out as f64);
                if (rel_xbin as f64 * nx_out as f64 - nx_stack as f64).abs()
                    > size_diff_crit as f64 * nx_stack as f64
                    || (rel_ybin as f64 * ny_out as f64 - ny_stack as f64).abs()
                        > size_diff_crit as f64 * ny_stack as f64
                    || rel_xbin != rel_ybin
                {
                    ierr = 1;
                }
                rel_binning = rel_xbin as f32;
            }
            if ierr != 0 {
                exit_error_fmt!(
                    "The image size does not correspond well enough between the %s and the frames to deduce their relationship",
                    CArg::Str(what)
                );
            }

            // For same binning, trim if there is a small difference
            if rel_xbin == 1 {
                if nx_out - nx_stack > trim_crit || ny_out - ny_stack > trim_crit {
                    printf!(
                        "The image size from the %s is significantly smaller and frames will not be trimmed to that size\n",
                        CArg::Str(what)
                    );
                } else if nx_stack < nx_out || ny_stack < ny_out {
                    printf!(
                        "The image size from the %s is slightly smaller and frames will be trimmed to that size\n",
                        CArg::Str(what)
                    );
                    nx_out = nx_stack;
                    ny_out = ny_stack;
                }
            } else {
                // Different binnings: look at labels to try to adjust it there
                printf!(
                    "The %s is at a different binning from the frames; frame sizes will not be adjusted\n",
                    CArg::Str(what)
                );
                for out_num in 0..self.m_num_out_files as usize {
                    let head = self.m_out_heads[out_num];
                    for ind in 0..self.m_heads[head].nlabl.clamp(0, MRC_NLABELS as i32) as usize {
                        title = [0; MRC_LABEL_SIZE + 1];
                        title[..MRC_LABEL_SIZE].copy_from_slice(&self.m_heads[head].labels[ind]);
                        title[MRC_LABEL_SIZE] = 0x00;
                        if self.adjust_title_binning(&title, &mut sstr, rel_binning, out_num == 0)
                            != 0
                        {
                            strncpy_label(&mut self.m_heads[head].labels[ind], &sstr);
                            fix_title_padding(&mut self.m_heads[head].labels[ind]);
                            break;
                        }
                    }
                }

                // Find and fix title in mdoc too
                if adjust_mdoc != 0 {
                    nz = adoc_get_number_of_sections(b"T").unwrap_or(-1);
                    for ind in 0..nz {
                        if let Ok(name) = adoc_get_section_name(b"T", ind) {
                            if self.adjust_title_binning(&name, &mut sstr, rel_binning, false) != 0
                            {
                                if adoc_change_section_name(b"T", ind, &sstr).is_err() {
                                    exit_error(b"Adjusting title with binning in mdoc file");
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        self.m_group_size = b3dmax!(1, self.m_group_size);
        if self.m_group_size > 1 {
            ierr = b3dmin!(
                self.m_max_num_z,
                self.m_num_all_vs_all + self.m_group_size - 1
            ) + 1
                - self.m_group_size;
            self.m_use_block_group =
                ((ierr + 1 - self.m_group_size) * (ierr - self.m_group_size)) / 2 < ierr;
            if self.m_use_block_group {
                printf!("Using block grouping; too few frames being fit for slide grouping\n");
            } else if self.m_num_all_vs_all <= MAX_ALL_VS_ALL as i32 + 1 - self.m_group_size {
                self.m_num_all_vs_all += self.m_group_size - 1;
            }
        } else {
            group_refine = 0;
        }

        // Set up the binnings to test
        self.m_num_bin_tests = 1;
        self.m_num_filt_tests[0] = 1;
        num_varies = 0;
        vary_radius2[0][0] = radius2;
        vary_sigma2[0][0] = sigma2;
        ierr = 0;
        if pip_get_integer_array(
            b"TestBinnings",
            &mut bins_to_test,
            &mut ierr,
            MAX_BINNINGS as i32,
        ) == 0
        {
            self.m_num_bin_tests = ierr;
            for ind in 0..self.m_num_bin_tests as usize {
                if bins_to_test[ind] < 1 || bins_to_test[ind] > 16 {
                    exit_error_fmt!(
                        "Binning value %d not allowed",
                        CArg::Int(bins_to_test[ind] as i64)
                    );
                }
                self.m_min_binning_to_test = b3dmin!(self.m_min_binning_to_test, bins_to_test[ind]);
            }
        } else {
            if als_bin_entered == 0 || align_bin_in < 0 {
                printf!(
                    "Selected the default binning of %d for this image size\n",
                    CArg::Int(align_bin as i64)
                );
            }
            bins_to_test[0] = align_bin;
            self.m_min_binning_to_test = align_bin;
        }

        // Get the filters to test
        pip_number_of_entries(b"VaryFilter", &mut num_varies);

        for ind in 0..self.m_num_bin_tests as usize {
            if (ind as i32) < num_varies {
                self.m_num_filt_tests[ind] = 0;
                pip_get_float_array(
                    b"VaryFilter",
                    &mut vary_radius2[ind],
                    &mut self.m_num_filt_tests[ind],
                    MAX_FILTERS as i32,
                );
                if num_avainput == 0 && self.m_num_filt_tests[ind] > 1 {
                    exit_error(b"You cannot vary filter values when doing cumulative alignment");
                }
                rs_sort_floats(&mut vary_radius2[ind], self.m_num_filt_tests[ind]);
                use_ind = ind as i32;
            } else {
                use_ind = b3dmax!(0, num_varies - 1);
                self.m_num_filt_tests[ind] = self.m_num_filt_tests[use_ind as usize];
            }
            for iz in 0..self.m_num_filt_tests[ind] as usize {
                vary_radius2[ind][iz] = vary_radius2[use_ind as usize][iz];
                vary_sigma2[ind][iz] = (b3dnint!(
                    SIG2_ROUND_FAC * sigma2 as f64 * vary_radius2[ind][iz] as f64 / radius2 as f64
                ) as f64
                    / SIG2_ROUND_FAC) as f32;
                num_times_best[ind][iz] = 0;
            }
        }

        if ref_radius2 == 0. {
            ref_radius2 = vary_radius2[0][0];
        }
        ref_sigma2 = (b3dnint!(SIG2_ROUND_FAC * sigma2 as f64 * ref_radius2 as f64 / radius2 as f64)
            as f64
            / SIG2_ROUND_FAC) as f32;

        //
        num_drift_loop = 1;
        pip_get_two_floats(
            b"DriftLimitDistAndNumber",
            &mut drift_max_dist,
            &mut drift_max_frac_num,
        );
        if drift_max_dist > 0. {
            if self.m_num_bin_tests > 1 {
                exit_error(b"You cannot enter -drift with test binnings");
            }
            num_drift_loop = 2;
            if drift_max_frac_num <= 0. {
                exit_error(
                    b"The maximum fraction or number of initial frames to drop must be positive",
                );
            }
            if drift_max_frac_num < 1. && drift_max_frac_num >= 0.5 {
                exit_error(b"Maximum fraction of initial frames to drop must be less than 0.5");
            }
        }

        // Get the gain reference, dark referemce, and camera defects
        self.get_gain_dark_defects(use_shr_mem);

        // Handle scaling
        if (self.m_gain_name.is_some() || self.m_extra_has_gain_ref != 0)
            && (self.m_in_head.mode == MRC_MODE_BYTE || self.m_frames_are_eer)
            && !entered_scale
            && total_scale == 0.
            && !scale_to_mean_sd
            && out_mode != MRC_MODE_FLOAT
        {
            printf!(
                "Applying default total scaling of %g because %s are being gain-normalized\n",
                CArg::Dbl(self.m_default_byte_scale as f64),
                CArg::Str(if self.m_frames_are_eer {
                    "electron events"
                } else {
                    "byte values"
                })
            );
            total_scale = self.m_default_byte_scale;
        }
        if total_scale > 0. {
            scale = total_scale / already_scaled_by;
        }

        // Get sizes
        FrameAlign::get_pad_sizes_bytes(
            self.m_nx,
            self.m_ny,
            full_taper_frac,
            sum_bin,
            self.m_min_binning_to_test,
            &mut self.m_full_pad_size,
            &mut self.m_sum_pad_size,
            &mut self.m_align_pad_size,
        );

        // See about GPU
        self.m_full_data_size = std::mem::size_of::<f32>() as i32;
        if self.m_use_gpu != 0 && taper_frac <= 0. && trim_frac <= 0. {
            printf!("The GPU cannot be used when the taper fraction is set to 0\n");
            self.m_use_gpu = -1;
        }

        self.m_do_spline = if spline_smooth != 0 && self.m_max_num_z >= min_num_for_spline {
            1
        } else {
            0
        };
        self.assess_gpu_needs(use_shr_mem, frc_name.as_deref());
        if self.m_gpu_flags != 0 && self.m_super_fac_for_defects > 0 {
            self.m_gpu_flags |= if self.m_super_fac_for_defects > 2 {
                GPU_AVG_SUPER_4X
            } else {
                GPU_AVG_SUPER_2X
            };
        }

        // FRC output file
        if let Some(name) = frc_name.as_deref() {
            imod_backup_file(&String::from_utf8_lossy(name));
            frc_fp = ImodFile::open(os_path(name), "w");
            if frc_fp.is_none() {
                exit_error_fmt!("Opening file for FRC curves, %s", CArg::Bytes(name));
            }
        }

        // Shift output file
        if pip_get_string(b"PlottableShiftFile", &mut extra_name) == 0 {
            imod_backup_file(&String::from_utf8_lossy(&extra_name));
            plot_fp = ImodFile::open(os_path(&extra_name), "w");
            if plot_fp.is_none() {
                exit_error_fmt!(
                    "Opening file for shift curves, %s",
                    CArg::Bytes(&extra_name)
                );
            }
        }
        pip_done();

        let mut stack_copied = false;
        if self.m_test_mode == 0 {
            // Set up output header(s)
            for out_num in 0..self.m_num_out_files as usize {
                let hp = self.m_out_heads[out_num];
                {
                    let head_ptr = &mut self.m_heads[hp];
                    head_ptr.nz = num_files_to_do;
                    head_ptr.mz = num_files_to_do;
                    head_ptr.nx = nx_out;
                    head_ptr.ny = ny_out;
                    head_ptr.mx = head_ptr.nx;
                    head_ptr.my = head_ptr.ny;
                    head_ptr.mode = out_mode;
                    head_ptr.amax = -1.0e30;
                    head_ptr.amin = 1.0e30;
                    head_ptr.amean = 0.;
                }
                if x_scale == 1.0 && extra_has_axis_pix != 0 {
                    x_scale = extra_pix_size;
                    y_scale = extra_pix_size;
                    z_scale = extra_pix_size;
                } else if x_scale == 1.0 && stack_name.is_some() {
                    // Get pixel size from stack if needed, scale by binning difference
                    stack_bin_x = (nx_out * sum_bin / stack_head.nx) as f32;
                    stack_bin_y = (ny_out * sum_bin / stack_head.ny) as f32;
                    stack_bin = b3dnint!(stack_bin_x);
                    if stack_bin > 0
                        && ((b3dnint!(stack_bin_x) as f32 - stack_bin_x).abs() as f64) < 0.05
                        && ((b3dnint!(stack_bin_y) as f32 - stack_bin_y).abs() as f64) < 0.05
                        && stack_bin == b3dnint!(stack_bin_y)
                    {
                        (x_scale, y_scale, z_scale) = mrc_get_scale(&stack_head);
                        x_scale /= stack_bin as f32;
                        y_scale /= stack_bin as f32;
                        z_scale /= stack_bin as f32;
                    }
                } else if x_scale == 1.0
                    && mdoc_name.is_some()
                    && self.m_mdoc_xsize != 0
                    && self.m_mdoc_pixel != 0.
                {
                    // Or get pixel size from mdoc and scale it by size change if any
                    stack_bin_x = (nx_out * sum_bin / self.m_mdoc_xsize) as f32;
                    stack_bin_y = (ny_out * sum_bin / self.m_mdoc_ysize) as f32;
                    stack_bin = b3dnint!(stack_bin_x);
                    if ((b3dnint!(stack_bin_x) as f32 - stack_bin_x).abs() as f64) < 0.05
                        && ((b3dnint!(stack_bin_y) as f32 - stack_bin_y).abs() as f64) < 0.05
                        && stack_bin == b3dnint!(stack_bin_y)
                        && stack_bin > 0
                    {
                        x_scale = self.m_mdoc_pixel / stack_bin as f32;
                        y_scale = x_scale;
                        z_scale = x_scale;
                    }
                }
                if option_pix_size > 0. {
                    x_scale = (10. * option_pix_size as f64) as f32;
                    y_scale = x_scale;
                    z_scale = x_scale;
                }
                mrc_set_scale(
                    &mut self.m_heads[hp],
                    (sum_bin as f32 * x_scale) as f64,
                    (sum_bin as f32 * y_scale) as f64,
                    (sum_bin as f32 * z_scale) as f64,
                );

                // 11/4/20: Axis rotation was already output, no need to do here if fileHasAxisPix

                let desc = title_descs[out_num].as_deref().unwrap_or(b"");
                let label = if sum_bin > 1 {
                    c_format_bytes(
                        "alignframes: %s scaled by %g, reduced %d",
                        &[
                            CArg::Bytes(desc),
                            CArg::Dbl(scale as f64),
                            CArg::Int(sum_bin as i64),
                        ],
                    )
                } else {
                    c_format_bytes(
                        "alignframes: %s scaled by %g",
                        &[CArg::Bytes(desc), CArg::Dbl(scale as f64)],
                    )
                };
                mrc_head_label(&mut self.m_heads[hp], &label);
                mrc_init_output_header(&mut self.m_heads[hp]);

                // Set up output file(s) and li for writing
                let out_name = self.m_out_names[out_num].clone().unwrap_or_default();
                imod_backup_file(&String::from_utf8_lossy(&out_name));
                out_fps[out_num] = ii_fopen(&out_name, "wb");
                if out_fps[out_num].is_none() {
                    exit_error_fmt!("Opening output file %s", CArg::Bytes(&out_name));
                }

                // Also create an .openTS file if it seems to be a tilt series
                if !self.m_tilt_angles.is_empty() || stack_name.is_some() || self.m_doing_frame_ts {
                    let mut name = out_name.clone();
                    name.extend_from_slice(b".openTS");
                    let _ = ImodFile::open(os_path(&name), "w");
                    open_ts_names[out_num] = Some(name);
                }

                mrc_init_li(Some(&mut li), None);
                mrc_init_li(Some(&mut li), Some(&self.m_heads[hp]));
                self.m_heads[hp].fp = out_fps[out_num].clone();

                // Need to test for output file type
                if b3d_output_file_type() == OUTPUT_TYPE_MRC {
                    if stack_name.is_some() && stack_head.next != 0 && self.m_tilt_angles.is_empty()
                    {
                        let head_ptr = &mut self.m_heads[hp];
                        head_ptr.next = stack_head.next;
                        head_ptr.nint = stack_head.nint;
                        head_ptr.nreal = stack_head.nreal;
                        head_ptr.header_size = stack_head.header_size;
                        if mrc_copy_extra_header(&mut stack_head, head_ptr) != 0 {
                            exit_error(b"Copying extended header from stack to output file");
                        }

                        // The source closes the stack here, inside the output
                        // loop, and then copies from the closed stream for the
                        // next output (BUGS.md); it is closed after the loop.
                        stack_copied = true;
                    } else if !self.m_tilt_angles.is_empty() {
                        self.m_num_sect = self.m_tilt_angles.len() as i32;
                        let head_ptr = &mut self.m_heads[hp];
                        head_ptr.next = 4 * b3dmin!(self.m_num_sect, num_files_to_do);
                        head_ptr.nint = 0;
                        head_ptr.nreal = 1;
                        head_ptr.header_size += head_ptr.next;
                        let next = head_ptr.next as usize;
                        let start = (starting_file - 1) as usize;
                        let angles = &ordered_angles[start..start + next / 4];
                        let bytes = unsafe {
                            std::slice::from_raw_parts(angles.as_ptr().cast::<u8>(), next)
                        };
                        let fp = out_fps[out_num].as_mut().unwrap();
                        if b3d_fseek(fp, 1024, SEEK_SET) != 0
                            || b3d_fwrite(bytes, 1, next, fp) as i32 != next as i32
                        {
                            exit_error(b"Writing tilt angles to extended header of output file");
                        }
                    }
                }

                // Do modifications to the mdoc
                if out_num == 0 && adjust_mdoc != 0 && !self.m_doing_frame_ts {
                    let name0 = self.m_out_names[0].clone().unwrap_or_default();
                    if adoc_set_key_value(ADOC_GLOBAL_NAME, 0, b"ImageFile", Some(&name0)) != 0
                        || adoc_set_two_integers(ADOC_GLOBAL_NAME, 0, b"ImageSize", nx_out, ny_out)
                            != 0
                        || adoc_set_integer(ADOC_GLOBAL_NAME, 0, b"DataMode", out_mode) != 0
                        || adoc_get_float(ADOC_GLOBAL_NAME, 0, b"PixelSpacing", &mut pix_temp) != 0
                        || adoc_set_float(
                            ADOC_GLOBAL_NAME,
                            0,
                            b"PixelSpacing",
                            pix_temp * rel_binning,
                        ) != 0
                    {
                        exit_error(b"Adjusting global values in mdoc");
                    }
                }
            }
        }
        if stack_copied {
            if let Some(mut fp) = stack_fp.take() {
                ii_fclose(&mut fp);
            }
        }

        // Get data
        let sum_size = ((self.m_nx / sum_bin) * (self.m_ny / sum_bin)) as usize;
        summed = vec![0.; sum_size];
        if do_unweight != 0 {
            unwgt_sum = vec![0.; sum_size];
        }

        if self.m_sum_rotation_flip != 0 {
            rot_sum = vec![0.; sum_size];
        }

        printf!(
            "Number of sets of frames to align = %d                           [ALF1]\n",
            CArg::Int(num_files_to_do as i64)
        );

        // Loop on frame files or frame sets
        super_file = 0;
        set_in_file = 0;
        num_sets = 0;
        ifile = starting_file - 1;
        while ifile < ending_file {
            min_error = 1.0e30;
            original_zval = ifile;

            // `alignframes.cpp:1374` reads `} if (`: the missing `else` sent
            // combined files to the frame-set branch and named every combined
            // image after the first input file (BUGS.md).
            if combine_files != 0 {
                balanced_group_limits(
                    num_single_files,
                    self.m_num_in_files,
                    ifile,
                    &mut start_combine,
                    &mut end_combine,
                );
                filename = self.m_in_files[start_combine as usize].clone();
            } else if break_set_size > 0 || extra_has_tilts != 0 || self.m_doing_frame_ts {
                filename = self.m_in_files[super_file as usize].clone();
            } else {
                filename = self.m_in_files[set_order_index[ifile as usize] as usize].clone();
                original_zval = set_order_index[ifile as usize];
            }
            if (self.m_zero_dose_thresh > 0.
                && self.m_dose_from_mdoc[original_zval as usize] < self.m_zero_dose_thresh)
                || (dropping_by_mean && mean_from_mdoc[original_zval as usize] < drop_mean_crit)
            {
                ifile += 1;
                continue;
            }

            // Open file if it is time to do so
            if set_in_file == 0 {
                let mut head = MrcHeader::default();
                self.m_in_fp =
                    Some(self.open_and_read_header(&filename, &mut head, b"input image", false));
                self.m_in_head = head;
                self.check_input_file(
                    &filename,
                    &self.m_in_head,
                    self.m_nx,
                    self.m_ny,
                    combine_files,
                );
                nz = self.m_in_head.nz;
                self.m_parallel_read = false;

                // If not combining files, see if the file is a TIFF for parallel reading
                if combine_files == 0 && max_read_threads > 1 {
                    self.m_file_copies[0] = ii_lookup_file_from_fp(self.m_in_fp.as_ref().unwrap())
                        .unwrap_or(std::ptr::null_mut());
                    if !self.m_file_copies[0].is_null()
                        && unsafe { (*self.m_file_copies[0]).file } == IIFILE_TIFF
                    {
                        let mut copies = [std::ptr::null_mut(); MAX_TIFF_THREADS];
                        copies.copy_from_slice(&self.m_file_copies[..MAX_TIFF_THREADS]);
                        self.m_num_read_threads =
                            ii_open_copies_for_threads(&mut copies, max_read_threads);
                        self.m_file_copies[..MAX_TIFF_THREADS].copy_from_slice(&copies);
                        self.m_parallel_read = self.m_num_read_threads > 1;
                    }
                }

                // Get gain reference if it is in there
                if self.m_extra_has_gain_ref != 0 {
                    let nxy = (self.m_nx * self.m_ny) as usize;
                    let gain = Rc::make_mut(self.m_gain_slice.as_mut().unwrap());
                    let gain_bytes = buf_bytes_mut(&mut gain[..nxy]);
                    let fp = self.m_in_fp.as_mut().unwrap();
                    if b3d_fseek(
                        fp,
                        MRC_HEADER_SIZE as i32
                            + 4 * (self.m_in_head.nint as i32 + self.m_in_head.nreal as i32) * nz,
                        SEEK_SET,
                    ) != 0
                        || b3d_fread(gain_bytes, 4, nxy, fp) as i32 != self.m_nx * self.m_ny
                    {
                        exit_error(b"Reading gain reference from extended header");
                    }
                    self.m_nx_gain = self.m_nx;
                    self.m_ny_gain = self.m_ny;
                    if self.m_rotation_flip > 0 {
                        let mut gain = self.m_gain_slice.take().unwrap();
                        self.rotate_flip_gain_reference(Rc::make_mut(&mut gain).as_mut_slice());
                        self.m_gain_slice = Some(gain);
                    }
                }

                // Set or get group limits
                if break_set_size > 0 && combine_files == 0 {
                    num_frame_use = nz;
                    if start_frame > 0 {
                        num_frame_use = b3dmin!(nz, end_frame) + 1 - start_frame;
                    }
                    num_sets = num_frame_use / break_set_size;

                    // Get the set limits and then adjust by the start frame
                    self.m_set_starts.clear();
                    self.m_num_in_sets.clear();
                    for ind in 0..num_sets {
                        balanced_group_limits(
                            num_frame_use,
                            num_sets,
                            ind,
                            &mut start_combine,
                            &mut end_combine,
                        );
                        self.m_set_starts.push(start_combine);
                    }
                    self.m_set_starts.push(end_combine + 1);
                    for ind in 0..num_sets as usize {
                        self.m_num_in_sets
                            .push(self.m_set_starts[ind + 1] - self.m_set_starts[ind]);
                    }
                    if start_frame > 1 {
                        for ind in 0..=num_sets as usize {
                            self.m_set_starts[ind] += start_frame - 1;
                        }
                    }
                } else if self.m_doing_frame_ts {
                    num_sets = self.m_set_starts.len() as i32;
                    set_in_file = ifile;
                } else if extra_has_tilts != 0 {
                    let mut fp = self.m_in_fp.take().unwrap();
                    let mut head = std::mem::take(&mut self.m_in_head);
                    self.analyze_extra_header(
                        &mut fp,
                        &mut head,
                        start_frame,
                        end_frame,
                        false,
                        &mut extra_buf,
                        &mut extra_buf_size,
                        &mut extra_tilts,
                        &mut min_set,
                        &mut max_set,
                        &mut file_axis,
                        &mut file_pix,
                    );
                    self.m_in_head = head;
                    self.m_in_fp = Some(fp);
                    num_sets = extra_tilts.len() as i32;
                    set_in_file = ifile;
                }
            }

            // Now proceed with the current file/set of frames
            if combine_files != 0 {
                nz = end_combine + 1 - start_combine;

                // `fclose` in the source leaves the image file's entry behind
                // in the opened-file list (BUGS.md); it is released here.
                if let Some(mut fp) = self.m_in_fp.take() {
                    ii_fclose(&mut fp);
                }
            } else if num_sets != 0 {
                ind = set_in_file;
                if (break_set_size > 0 || extra_has_tilts != 0 || self.m_doing_frame_ts)
                    && num_single_files == 1
                {
                    ind = set_order_index[set_in_file as usize];
                    original_zval = set_order_index[ifile as usize];
                }
                start_combine = self.m_set_starts[ind as usize];
                nz = self.m_num_in_sets[ind as usize];
                end_combine = start_combine + nz - 1;
            }
            self.extract_file_tail(&filename, &mut sstr);

            self.analyze_for_partial_frames(
                &mut nz,
                &mut start_combine,
                &mut end_combine,
                ifile,
                &mut skipped_frame,
            );

            drift_loop = 0;
            while drift_loop < num_drift_loop {
                nz_align = nz;
                if start_frame > 0 && num_sets == 0 {
                    nz_align = b3dmin!(nz, end_frame) + 1 - start_frame;
                }
                num_avause = num_avainput;
                if num_avause == -1 || num_avause > nz_align {
                    num_avause = nz_align;
                } else if num_avause < 0 {
                    num_avause = b3dmax!(
                        min_fractional_ava,
                        (nz_align - 1 - num_avainput) / (-num_avainput)
                    );
                }
                num_avause = b3dmin!(MAX_ALL_VS_ALL as i32, num_avause);

                // Make per-file decision on grouping
                block_grp_size = 1;
                slide_grp_size = 1;
                if self.m_group_size > 1 {
                    ierr = nz_align + 1 - self.m_group_size;
                    if self.m_use_block_group
                        || ((ierr + 1 - self.m_group_size) * (ierr - self.m_group_size)) / 2 < ierr
                    {
                        block_grp_size = self.m_group_size;
                        if !self.m_use_block_group {
                            printf!(
                                "Using block grouping instead of sliding grouping for file # %d\n",
                                CArg::Int((ifile + 1) as i64)
                            );
                        }
                    } else {
                        slide_grp_size = self.m_group_size;
                        if num_avause <= MAX_ALL_VS_ALL as i32 + 1 - self.m_group_size {
                            num_avause += self.m_group_size - 1;
                        }
                    }
                }

                self.m_in_data_size = data_size_for_mode(self.m_in_head.mode).map_or(0, |(d, _)| d);
                need_alloc = self.m_in_data_size * self.m_nx * self.m_ny;
                if need_alloc > buf_alloc_size {
                    read_buf = frame_storage(need_alloc as usize);
                    buf_alloc_size = need_alloc;
                }

                // The group sum of byte frames is accumulated as shorts, twice the
                // `needAlloc` bytes the source allocates, and a later file can need
                // the sum buffer after the read buffer was sized without it
                // (BUGS.md): it is sized for shorts and allocated when first needed.
                if block_grp_size > 1 {
                    let sum_bytes = b3dmax!(need_alloc, 2 * self.m_nx * self.m_ny) as usize;
                    if sum_buf.len() * 4 < sum_bytes {
                        sum_buf = frame_storage(sum_bytes);
                    }
                }

                nz_align = b3dmax!(1, nz_align / block_grp_size);

                // Set up for spline scaling if criteria met
                self.m_do_spline = 0;
                if spline_smooth > 0 && nz_align >= min_num_for_spline {
                    self.m_do_spline = 1;
                }

                // Estimate memory usage
                // Does summing in two passes work with cumulative alignment?
                tmean = FrameAlign::total_memory_needs(
                    self.m_full_pad_size,
                    self.m_full_data_size,
                    self.m_sum_pad_size,
                    self.m_align_pad_size,
                    num_avause,
                    nz_align,
                    self.m_refine_at_end,
                    self.m_num_bin_tests,
                    self.m_num_filt_tests[0],
                    self.m_hybrid_shifts,
                    self.m_group_size,
                    self.m_do_spline,
                    self.m_gpu_flags,
                    self.m_defer_sum,
                    self.m_test_mode,
                    self.m_start_assess,
                    &mut self.m_sum_in_one_pass,
                    &mut self.m_num_hold_full,
                );
                if tmean > self.m_memory_limit {
                    if self.m_start_assess >= 0 {
                        exit_error_fmt!(
                            "The memory limit is too low to allow initial assessment and summing of %d frames in one pass %s",
                            CArg::Int(nz_align as i64),
                            CArg::Str(if self.m_refine_at_end != 0 || self.m_do_spline != 0 {
                                "with refinement or smoothing at the end"
                            } else {
                                "with this many pairwise comparisons"
                            })
                        );
                    }
                    if self.m_sum_in_one_pass {
                        self.m_sum_in_one_pass = false;
                        if !warned_two_pass {
                            printf!(
                                "Using two passes: a single pass with %d frames requires %.1f GB, above the limit of %.1f GB\n",
                                CArg::Int(nz_align as i64),
                                CArg::Dbl(tmean as f64),
                                CArg::Dbl(self.m_memory_limit as f64)
                            );
                        }
                        warned_two_pass = true;
                        let _ = ImodFile::Stdout.flush();
                    }
                }
                num_test_loops = self.m_num_bin_tests
                    + if self.m_sum_in_one_pass || self.m_test_mode != 0 {
                        0
                    } else {
                        1
                    };

                // Loop on conditions
                was_good_enough = false;
                itest = 0;
                while itest < num_test_loops {
                    // Set summing flag for summing with alignment if it can be done, otherwise set
                    // for sum only on final loop unless assessing from subset, otherwise skip the sum
                    if self.m_sum_in_one_pass {
                        summing_mode = 0;
                    } else if itest == self.m_num_bin_tests {
                        summing_mode = if self.m_start_assess >= 0 { 0 } else { -1 };
                    } else {
                        summing_mode = 1;
                    }

                    // Set limits and indices for the extra loop; if it has to compute alignment
                    // because it was assessed on a subset, then it needs either best filter or the
                    // whole set to do hybrid
                    use_start = if num_sets != 0 { 0 } else { start_frame };
                    use_end = end_frame;
                    if itest == self.m_num_bin_tests {
                        ind_bin_use = ind_best_bin;
                        if self.m_hybrid_shifts != 0 {
                            ind_filt_use = 0;
                            num_filt_use = self.m_num_filt_tests[ind_best_bin as usize];
                        } else {
                            ind_filt_use = ind_best_filt;
                            num_filt_use = 1;
                        }
                    } else {
                        // Otherwise set up for this round
                        ind_bin_use = itest;
                        ind_filt_use = 0;
                        num_filt_use = self.m_num_filt_tests[itest as usize];
                        if self.m_start_assess >= 0 {
                            use_start = self.m_start_assess;
                            use_end = end_assess;
                        }
                    }
                    if self.m_debug % 10 != 0 {
                        printf!(
                            "itest = %d,  summingMode = %d,  indBinUse = %d,  indFiltUse = %d\n",
                            CArg::Int(itest as i64),
                            CArg::Int(summing_mode as i64),
                            CArg::Int(ind_bin_use as i64),
                            CArg::Int(ind_filt_use as i64)
                        );
                    }

                    // Set up actual frame limits for this file
                    z_dir = if reverse != 0 { -1 } else { 1 };
                    if use_start > 0 {
                        z_start = if reverse != 0 {
                            b3dmin!(use_end, nz) - 1
                        } else {
                            use_start - 1
                        };
                        z_end = if reverse != 0 {
                            use_start - 1
                        } else {
                            b3dmin!(use_end, nz) - 1
                        };
                    } else {
                        z_start = if reverse != 0 { nz - 1 } else { 0 };
                        z_end = if reverse != 0 { 0 } else { nz - 1 };
                    }
                    num_fetch = z_dir * (z_end - z_start) + 1;
                    nz_align = b3dmax!(1, num_fetch / block_grp_size);
                    if num_avause < 2 + slide_grp_size {
                        num_filt_use = 1;
                    }

                    // Get file, initialize
                    let bin_use = ind_bin_use as usize;
                    let filt_use = ind_filt_use as usize;
                    ierr = self.s_fa.initialize(
                        sum_bin,
                        bins_to_test[bin_use],
                        trim_frac,
                        num_avause,
                        self.m_refine_at_end,
                        self.m_hybrid_shifts,
                        if summing_mode == 0 && (self.m_defer_sum != 0 || self.m_do_spline != 0) {
                            1
                        } else {
                            0
                        },
                        slide_grp_size,
                        self.m_nx,
                        self.m_ny,
                        full_taper_frac,
                        taper_frac,
                        anti_filt_type - 1,
                        radius1,
                        &vary_radius2[bin_use][filt_use..],
                        sigma1,
                        &vary_sigma2[bin_use][filt_use..],
                        num_filt_use,
                        shift_limit,
                        k_factor,
                        max_max_weight,
                        summing_mode,
                        nz_align,
                        (do_unweight > 0) as i32,
                        self.m_gpu_flags,
                        self.m_debug,
                    );
                    if ierr != 0 {
                        exit_error_fmt!(
                            "Error %d initializing frame summing for file # %d",
                            CArg::Int(ierr as i64),
                            CArg::Int((ifile + 1) as i64)
                        );
                    }

                    // First time, get buffers for even and odd sums.  The even buffer might need to
                    // have the FFT copied to, so need to ask framealign what size it needs to be
                    if even_odd_ouput != 0 && even_sum.is_empty() {
                        if use_shr_mem != 0 {
                            exit_error(b"You cannot test shrmemframe with even/odd output");
                        }
                        even_sum = vec![0.; self.s_fa.get_padded_sum_size() as usize];
                        odd_sum = vec![0.; sum_size];
                    }

                    // Initialize dose weighting: start by getting doses for all underlying frames
                    if (self.m_total_dose > 0. || self.m_dose_file_type > 0)
                        && self.m_test_mode == 0
                    {
                        sum_of_frames = 0;
                        if self.m_dose_file_type > 3 && !self.m_doing_frame_ts {
                            let line = self.m_frame_dose_lines[original_zval as usize].clone();
                            let mut temp_arr = std::mem::take(&mut self.m_temp_val1);
                            let mut frame_doses = std::mem::take(&mut self.m_frame_doses);
                            self.expand_frame_doses_numbers(
                                &line,
                                &mut temp_arr,
                                nz,
                                &mut sum_of_frames,
                                &mut sum_of_doses,
                                Some(&mut frame_doses),
                            );
                            self.m_temp_val1 = temp_arr;
                            self.m_frame_doses = frame_doses;
                            if self.m_frames_are_eer && sum_of_frames == 1 {
                                for iz in (0..nz as usize).rev() {
                                    self.m_frame_doses[iz] = self.m_frame_doses[0] / nz as f32;
                                }
                            } else if sum_of_frames > 0 && sum_of_frames < nz {
                                printf!(
                                    "WARNING: %s - The frame doses and numbers for file # %d include too few frames (%d vs %d)\n",
                                    CArg::Bytes(progname),
                                    CArg::Int((ifile + 1) as i64),
                                    CArg::Int(sum_of_frames as i64),
                                    CArg::Int(nz as i64)
                                );
                            }
                        }

                        // Fall back to equal division
                        if sum_of_frames < nz {
                            for iz in 0..nz as usize {
                                self.m_frame_doses[iz] =
                                    self.m_total_dose_vec[original_zval as usize] / nz as f32;
                            }
                        }

                        // Combine doses for grouped frames and frames actually being used
                        self.m_temp_val1.resize(nz_align as usize, 0.);
                        prior_temp =
                            self.m_initial_dose + self.m_prior_dose_vec[original_zval as usize];
                        if use_start > 0 {
                            for iz in 0..use_start as usize {
                                prior_temp += self.m_frame_doses[iz];
                            }
                        }
                        for group in 0..nz_align {
                            self.frame_group_limits(
                                num_fetch,
                                nz_align,
                                group,
                                &mut group_start,
                                &mut group_end,
                                z_start,
                                z_dir,
                                &mut iz_low,
                                &mut iz_high,
                            );
                            self.m_temp_val1[group as usize] = 0.;
                            iz = iz_low;
                            while z_dir * (iz - iz_high) <= 0 {
                                self.m_temp_val1[group as usize] += self.m_frame_doses[iz as usize];
                                iz += z_dir;
                            }
                        }
                        if self.m_debug % 10 != 0 {
                            printf!(
                                "Prior dose %.3f   total dose %.3f  frame doses:\n",
                                CArg::Dbl(prior_temp as f64),
                                CArg::Dbl(self.m_total_dose_vec[original_zval as usize] as f64)
                            );
                            for iz in 0..nz_align {
                                printf!(" %.3f", CArg::Dbl(self.m_temp_val1[iz as usize] as f64));
                                if (iz + 1) % 12 == 0 || iz == nz_align - 1 {
                                    printf!("\n");
                                }
                            }
                        }
                        let mut filt_size = 0;
                        ierr = self.s_fa.setup_dose_weighting(
                            prior_temp,
                            &self.m_temp_val1,
                            x_scale,
                            self.m_dose_scaling,
                            dose_afac,
                            dose_bfac,
                            dose_cfac,
                            if self.m_reweight_filt {
                                Some(&self.m_reweight_ones)
                            } else {
                                None
                            },
                            &mut filt_size,
                        );
                        if ierr != 0 {
                            exit_error_fmt!(
                                "Error %d setting up dose weighting for file # %d",
                                CArg::Int(ierr as i64),
                                CArg::Int((ifile + 1) as i64)
                            );
                        }
                    }

                    // Loop on frames in selected order, but do groups backwards to put largest at end
                    let nxy = (self.m_nx * self.m_ny) as usize;
                    group = 0;
                    while group < nz_align {
                        self.frame_group_limits(
                            num_fetch,
                            nz_align,
                            group,
                            &mut group_start,
                            &mut group_end,
                            z_start,
                            z_dir,
                            &mut iz_low,
                            &mut iz_high,
                        );

                        // Which buffer `useBuf` points at: 0-2 a partial scan
                        // buffer, 3 `readBuf`, 4 `sumBuf`.
                        let mut use_buf: usize = 3;
                        use_mode = self.m_in_head.mode;
                        iz = iz_low;
                        while z_dir * (iz - iz_high) <= 0 {
                            // `useBuf` returns to `readBuf` for every frame: the
                            // source sets it once per group, so from the second
                            // frame of a group on it added the sum buffer to
                            // itself (BUGS.md).
                            use_buf = 3;
                            if combine_files != 0 {
                                let name = self.m_in_files[(start_combine + iz) as usize].clone();
                                let mut head = MrcHeader::default();
                                let mut fp = self.open_and_read_header(
                                    &name,
                                    &mut head,
                                    b"input image",
                                    false,
                                );
                                self.check_input_file(
                                    &name,
                                    &head,
                                    self.m_nx,
                                    self.m_ny,
                                    combine_files,
                                );
                                if mrc_read_slice(
                                    buf_bytes_mut(&mut read_buf),
                                    &mut fp,
                                    &mut head,
                                    0,
                                    b'Z',
                                ) != 0
                                {
                                    exit_error_fmt!(
                                        "Reading from file # %d: %s",
                                        CArg::Int((start_combine + iz) as i64),
                                        CArg::Bytes(&name)
                                    );
                                }
                                head.fp = None;
                                self.m_in_head = head;

                                // `fclose` in the source (BUGS.md).
                                ii_fclose(&mut fp);
                            } else {
                                iz_read = iz;
                                if num_sets != 0 {
                                    iz_read += start_combine;
                                }

                                for ind in 0..3 {
                                    if iz_read == self.m_zin_partial_bufs[ind] {
                                        use_buf = ind;
                                    }
                                }

                                // The source's `if (ind > 2)` after that loop is
                                // always true, so the frame is always read.
                                self.read_one_frame(buf_bytes_mut(&mut read_buf), iz_read, ifile);
                            }
                            if iz_low != iz_high {
                                if iz == iz_low {
                                    sum_buf.fill(0.);
                                }
                                let source: &[f32] = match use_buf {
                                    0..=2 => &self.m_partial_scan_bufs[use_buf],
                                    _ => &read_buf,
                                };
                                self.add_to_sum_buffer(
                                    buf_bytes(source),
                                    self.m_in_head.mode,
                                    buf_bytes_mut(&mut sum_buf),
                                    &mut use_mode,
                                    nxy as i32,
                                );
                                use_buf = 4;
                            }
                            iz += z_dir;
                        }
                        let storage: &[f32] = match use_buf {
                            0..=2 => &self.m_partial_scan_bufs[use_buf],
                            3 => &read_buf,
                            _ => &sum_buf,
                        };

                        // Get the truncation limit set on first, and on second one also if frame sets
                        if group == 0 || (group == 1 && self.m_doing_frame_ts) {
                            ierr = self.s_fa.set_truncation_limit(
                                buf_bytes(storage),
                                self.m_nx,
                                self.m_ny,
                                use_mode,
                                self.m_trunc_limit,
                                &mut trunc_use,
                            );
                            if ierr != 0 {
                                exit_error(if ierr == 1 {
                                    b"Allocating line pointers for analyzing truncation limit"
                                        as &[u8]
                                } else {
                                    b"Error computing mean and SD with sampling for setting truncation limit"
                                });
                            }
                        }

                        // Pass the frame
                        let dark: Option<&[i16]> = match self.m_dark_slice.as_ref() {
                            Some(slice) => match &slice.data {
                                MrcData::S(v) => Some(v.as_slice()),
                                MrcData::Us(v) => Some(unsafe {
                                    std::slice::from_raw_parts(v.as_ptr().cast::<i16>(), v.len())
                                }),
                                _ => None,
                            },
                            None => None,
                        };
                        let gain = if self.m_gain_slice.is_some() && !self.m_antialias_eer {
                            self.m_gain_slice.clone()
                        } else {
                            None
                        };
                        ierr = self.s_fa.next_frame(
                            frame_data(storage, use_mode, nxy),
                            use_mode,
                            gain,
                            if self.m_antialias_eer {
                                0
                            } else {
                                self.m_nx_gain
                            },
                            if self.m_antialias_eer {
                                0
                            } else {
                                self.m_ny_gain
                            },
                            dark,
                            trunc_use,
                            Some(self.m_defects.clone()),
                            self.m_cam_size_x,
                            self.m_cam_size_y,
                            self.m_cor_def_binning,
                            best_xshifts[group as usize],
                            best_yshifts[group as usize],
                        );
                        if ierr != 0 {
                            exit_error_fmt!(
                                "Error %d processing frame/group %d from %s %d",
                                CArg::Int(ierr as i64),
                                CArg::Int(group as i64),
                                CArg::Str(if num_sets != 0 { "set" } else { "file" }),
                                CArg::Int((ifile + 1) as i64)
                            );
                        }
                        group += 1;
                    }
                    num_done = nz_align;
                    ierr = b3dmin!(num_avause, num_done) + 1 - self.m_group_size;
                    do_robust = ((ierr + 1 - self.m_group_size) * (ierr - self.m_group_size)) / 2
                        >= 2 * ierr
                        && k_factor > 0.;

                    // Finish up and get results;
                    ierr = self.s_fa.finish_align_and_sum(
                        ref_radius2,
                        ref_sigma2,
                        iter_crit,
                        group_refine,
                        self.m_do_spline,
                        &mut summed,
                        &mut x_shifts,
                        &mut y_shifts,
                        &mut raw_xshifts,
                        &mut raw_yshifts,
                        Some(&mut ring_corrs),
                        frc_delta_r,
                        &mut fa_best_filt,
                        &mut smooth_dist,
                        &mut raw_dist,
                        &mut res_mean,
                        &mut pred_mean,
                        &mut mean_res_max,
                        &mut max_res_max,
                        &mut mean_raw_max,
                        &mut max_raw_max,
                        if even_sum.is_empty() {
                            None
                        } else {
                            Some(&mut even_sum)
                        },
                        if odd_sum.is_empty() {
                            None
                        } else {
                            Some(&mut odd_sum)
                        },
                    );
                    if ierr == 3 {
                        exit_error_fmt!(
                            "An unrecoverable error in GPU processing occurred for %s # %d",
                            CArg::Str(if num_sets != 0 { "set" } else { "file" }),
                            CArg::Int(ifile as i64)
                        );
                    } else if ierr != 0 {
                        exit_error_fmt!(
                            "No frames were aligned for file # %d",
                            CArg::Int(ifile as i64)
                        );
                    }

                    // Evaluate whether to drop initial frames due to excessive shift
                    excluding_initial = false;
                    if drift_loop == 0 && num_drift_loop == 2 {
                        max_exclude = b3dnint!(if drift_max_frac_num >= 1. {
                            drift_max_frac_num
                        } else {
                            drift_max_frac_num * num_done as f32
                        });
                        max_exclude = b3dmin!(max_exclude, b3dnint!(0.5 * num_done as f64));
                        ix = 1;
                        while ix <= max_exclude {
                            let i = ix as usize;
                            prior_temp = ((raw_xshifts[i] - raw_xshifts[i - 1]).powf(2.)
                                + (raw_yshifts[i] - raw_yshifts[i - 1]).powf(2.))
                            .sqrt();
                            if prior_temp < drift_max_dist {
                                break;
                            }
                            if !excluding_initial {
                                printf!(
                                    "%s %d: drop frame (drift):",
                                    CArg::Str(if num_sets != 0 { "Set" } else { "File" }),
                                    CArg::Int((ifile + 1) as i64)
                                );
                            }
                            printf!(
                                " %d (%.1f) ",
                                CArg::Int((start_combine + 1) as i64),
                                CArg::Dbl(prior_temp as f64)
                            );
                            nz_align -= 1;
                            start_combine += 1;
                            excluding_initial = true;
                            ix += 1;
                        }
                        nz = nz_align;
                        if excluding_initial {
                            printf!("\n");
                            break;
                        }
                    }

                    // Do the basic summary report starting with the header line
                    if itest < self.m_num_bin_tests {
                        num_fetch = 1;
                        if num_avause != 0 && num_filt_use > 1 {
                            num_fetch = num_filt_use + 1;
                        }
                        if itest == 0 {
                            printf!(
                                "%s %d (%s): %d frames",
                                CArg::Str(if num_sets != 0 { "Set" } else { "File" }),
                                CArg::Int((ifile + 1) as i64),
                                CArg::Bytes(&sstr),
                                CArg::Int((z_dir * (z_end - z_start) + 1) as i64)
                            );
                            if num_sets != 0 {
                                printf!(
                                    " from %d to %d",
                                    CArg::Int((start_combine + 1) as i64),
                                    CArg::Int((end_combine + 1) as i64)
                                );
                                if skipped_frame[0] >= 0 {
                                    printf!(" (skip %d", CArg::Int((skipped_frame[0] + 1) as i64));
                                    if skipped_frame[1] >= 0 {
                                        printf!(" %d", CArg::Int((skipped_frame[1] + 1) as i64));
                                    }
                                    printf!(")");
                                }
                            }
                            if !ordered_angles.is_empty() {
                                printf!(
                                    "   (%.1f deg)",
                                    CArg::Dbl(ordered_angles[ifile as usize] as f64)
                                );
                            }
                            printf!("\n");
                        }

                        // Report residuals and total distance
                        for filt in 0..num_fetch as usize {
                            let it = itest as usize;
                            do_abbrev = false;
                            if b3dmin!(num_avause, num_done) >= 3 {
                                if self.m_num_bin_tests * self.m_num_filt_tests[0] > 1 {
                                    if (filt as i32) < self.m_num_filt_tests[it]
                                        || self.m_num_filt_tests[it] == 1
                                    {
                                        printf!(
                                            "Results with bin = %d  rad2 = %.3f  sig2 = %.4f\n",
                                            CArg::Int(bins_to_test[it] as i64),
                                            CArg::Dbl(
                                                vary_radius2[it][filt.min(MAX_FILTERS - 1)] as f64
                                            ),
                                            CArg::Dbl(
                                                vary_sigma2[it][filt.min(MAX_FILTERS - 1)] as f64
                                            )
                                        );
                                    } else {
                                        printf!(
                                            "Hybrid results,  bin = %d\n",
                                            CArg::Int(bins_to_test[it] as i64)
                                        );
                                    }
                                }
                                if self.m_num_bin_tests * self.m_num_filt_tests[0] > 1
                                    || num_summary_lines > 1
                                {
                                    printf!(
                                        "  %sesid mean = %.3f, mean max = %.2f, max max = %.2f  l-o err = %.3f\n",
                                        CArg::Str(if do_robust { "Wgtd r" } else { "R" }),
                                        CArg::Dbl(res_mean[filt] as f64),
                                        CArg::Dbl(mean_res_max[filt] as f64),
                                        CArg::Dbl(max_res_max[filt] as f64),
                                        CArg::Dbl(pred_mean[filt] as f64)
                                    );
                                } else {
                                    printf!(
                                        " %sesid mean = %.3f, max max = %.2f  l-o= %.3f",
                                        CArg::Str(if do_robust { "Wgtd r" } else { "R" }),
                                        CArg::Dbl(res_mean[filt] as f64),
                                        CArg::Dbl(max_res_max[filt] as f64),
                                        CArg::Dbl(pred_mean[filt] as f64)
                                    );
                                    do_abbrev = true;
                                }
                            }
                            if self.m_num_bin_tests * self.m_num_filt_tests[0] > 1
                                || num_summary_lines > 1
                            {
                                if do_robust && num_avause != 0 {
                                    printf!(
                                        "  Max unweighted resid mean = %.2f, max = %.2f ",
                                        CArg::Dbl(mean_raw_max[filt] as f64),
                                        CArg::Dbl(max_raw_max[filt] as f64)
                                    );
                                } else {
                                    printf!("                                               ");
                                }
                                do_abbrev = false;
                            }
                            printf!(
                                "  Dist = %.2f, %s = %.2f\n",
                                CArg::Dbl(raw_dist[filt] as f64),
                                CArg::Str(if do_abbrev { "smth" } else { "smoothed" }),
                                CArg::Dbl(smooth_dist[filt] as f64)
                            );

                            // Report shifts for frame tilt series
                            if filt as i32 == num_fetch - 1
                                && self.m_doing_frame_ts
                                && !suppress_initial_shifts
                            {
                                printf!("  Initial raw inter-frame shifts:");
                                for ix in 1..b3dmin!(num_done, 5) as usize {
                                    printf!(
                                        "   %.1f",
                                        CArg::Dbl(
                                            ((raw_xshifts[ix] - raw_xshifts[ix - 1]).powf(2.)
                                                + (raw_yshifts[ix] - raw_yshifts[ix - 1]).powf(2.))
                                            .sqrt()
                                                as f64
                                        )
                                    );
                                }
                                printf!("\n");
                            }
                            let _ = ImodFile::Stdout.flush();
                        }
                    }

                    // Keep track of best binning, copy shifts from it
                    copy_shifts = itest == self.m_num_bin_tests && summing_mode == 0;
                    let best = fa_best_filt as usize;
                    error = (pred_mean[best] as f64 * (1. - max_max_weight as f64)
                        + max_res_max[best] as f64 * max_max_weight as f64)
                        as f32;
                    if error < min_error && itest < self.m_num_bin_tests {
                        ind_best_bin = itest;
                        ind_best_filt = fa_best_filt;
                        min_error = error;
                        min_mean = res_mean[best];
                        min_pred = pred_mean[best];
                        copy_shifts = true;
                    }
                    if copy_shifts {
                        if self.m_debug % 10 != 0 {
                            printf!("Copy shifts test %d\n", CArg::Int(itest as i64));
                        }
                        for ix in 0..num_done as usize {
                            best_xshifts[ix] = x_shifts[ix];
                            best_yshifts[ix] = y_shifts[ix];
                            best_xraw[ix] = raw_xshifts[ix];
                            best_yraw[ix] = raw_yshifts[ix];
                            if self.m_debug % 10 != 0 {
                                printf!(
                                    "%.2f  %.2f\n",
                                    CArg::Dbl(x_shifts[ix] as f64),
                                    CArg::Dbl(y_shifts[ix] as f64)
                                );
                            }
                        }
                    }

                    // If the error is now good enough, advance to the end of the test runs
                    if min_error <= good_enough && itest < self.m_num_bin_tests - 1 {
                        itest = self.m_num_bin_tests - 1;
                        was_good_enough = true;
                    }
                    itest += 1;
                } // End of test loop
                if num_drift_loop > 1 && !excluding_initial {
                    break;
                }
                drift_loop += 1;
            } // End of drift loop

            // Pick up the unweighted sum
            if do_unweight != 0 && self.s_fa.get_unweighted_sum(&mut unwgt_sum) != 0 {
                exit_error(b"getting non-dose-weighted sum");
            }

            // Advance set number and wrap it back to 0 after last file; close file when needed
            if num_sets != 0 {
                set_in_file += 1;
                if set_in_file == num_sets {
                    super_file += 1;
                    set_in_file = 0;
                }
            }
            if combine_files == 0 && set_in_file == 0 {
                if self.m_parallel_read {
                    for ind in 1..self.m_num_read_threads as usize {
                        unsafe { ii_delete(self.m_file_copies[ind]) };
                    }
                }
                if let Some(mut fp) = self.m_in_fp.take() {
                    ii_fclose(&mut fp);
                }
                if self.m_debug != 0 && self.m_parallel_read {
                    printf!(
                        "mNumReadThreads = %d,  mWallRead = %g\n",
                        CArg::Int(self.m_num_read_threads as i64),
                        CArg::Dbl(self.m_wall_read)
                    );
                }
            }

            // Write out data at end of loop
            if summing_mode <= 0 {
                if (self.m_debug % 10) > 1 && self.m_getting_frc {
                    for ix in 0..(0.5 / frc_delta_r as f64).floor() as i32 {
                        printf!(
                            "%.4f  %.5f\n",
                            CArg::Dbl((ix as f64 + 0.5) * frc_delta_r as f64),
                            CArg::Dbl(ring_corrs[ix as usize] as f64)
                        );
                    }
                }

                for out_num in 0..self.m_num_out_files as usize {
                    let hp = self.m_out_heads[out_num];
                    let use_sum: &mut Vec<f32> = match out_num {
                        0 => &mut summed,
                        _ if do_unweight != 0 && out_num == 1 => &mut unwgt_sum,
                        _ if out_num as i32 == self.m_num_out_files - 2 => &mut even_sum,
                        _ => &mut odd_sum,
                    };
                    let use_rot = self.m_sum_rotation_flip != 0;
                    if use_rot {
                        rotate_flip_image(
                            RotateFlipData::Float {
                                array: &use_sum[..sum_size],
                                brray: &mut rot_sum[..sum_size],
                            },
                            self.m_nx / sum_bin,
                            self.m_ny / sum_bin,
                            self.m_sum_rotation_flip,
                            0,
                            0,
                            0,
                            &mut ix,
                            &mut iy,
                            0,
                        );
                    }
                    let use_sum: &mut [f32] = if use_rot { &mut rot_sum } else { use_sum };

                    // Trim if size is over
                    ix = (nx_sum - nx_out) / 2;
                    iy = (ny_sum - ny_out) / 2;
                    if ix != 0 || iy != 0 {
                        // In place in the source: the extraction reads ahead of
                        // what it writes, so a copy of the input is the same.
                        let input = use_sum[..sum_size].to_vec();
                        extract_with_binning(
                            buf_bytes(&input),
                            SLICE_MODE_FLOAT,
                            nx_sum,
                            ix,
                            ix + nx_out - 1,
                            iy,
                            iy + ny_out - 1,
                            1,
                            buf_bytes_mut(use_sum),
                            0,
                            &mut iz,
                            &mut ierr,
                        );
                    }

                    // Scale the data and/or apply scale for binning
                    let head_nx = self.m_heads[hp].nx;
                    let head_ny = self.m_heads[hp].ny;
                    use_scale = scale / kernel_scale as f32;
                    if use_scale != 1. || sum_bin > 1 {
                        for iz in 0..(head_nx * head_ny) as usize {
                            use_sum[iz] *= use_scale * sum_bin as f32 * sum_bin as f32;
                        }
                    }

                    // Manage header mmm and write the data
                    tmean = 0.;
                    if scale_to_mean_sd && sd_scale > 0. {
                        let mut sum_dbl = 0f64;
                        let mut sum_sq_dbl = 0f64;
                        array_min_max_mean_sd(
                            use_sum,
                            head_nx,
                            head_ny,
                            0,
                            head_nx - 1,
                            0,
                            head_ny - 1,
                            &mut tmin,
                            &mut tmax,
                            &mut sum_dbl,
                            &mut sum_sq_dbl,
                            &mut tmean,
                            &mut tsd,
                        );
                        diff = sum_dbl;
                        min_diff = sum_sq_dbl;
                        let _ = (diff, min_diff);
                    } else {
                        array_min_max_mean(
                            use_sum,
                            head_nx,
                            head_ny,
                            0,
                            head_nx - 1,
                            0,
                            head_ny - 1,
                            &mut tmin,
                            &mut tmax,
                            &mut tmean,
                        );
                    }

                    // If scaling to mean/sd, set scaling by either the mean or the SD and added
                    // factor as appropriate to match means, scale data and adjust min/max/mean
                    if scale_to_mean_sd {
                        if sd_scale > 0. {
                            scale_fac = sd_scale / tsd;
                            add_fac = mean_scale - scale_fac * tmean;
                        } else {
                            scale_fac = mean_scale / tmean;
                            add_fac = 0.;
                        }
                        for iz in 0..(head_nx * head_ny) as usize {
                            use_sum[iz] = scale_fac * use_sum[iz] + add_fac;
                        }
                        tmean = tmean * scale_fac + add_fac;
                        tmin = tmin * scale_fac + add_fac;
                        tmax = tmax * scale_fac + add_fac;
                    }
                    {
                        let head_ptr = &mut self.m_heads[hp];
                        head_ptr.amin = if head_ptr.amin < tmin {
                            head_ptr.amin
                        } else {
                            tmin
                        };
                        head_ptr.amax = if head_ptr.amax > tmax {
                            head_ptr.amax
                        } else {
                            tmax
                        };
                        head_ptr.amean += tmean / num_files_to_do as f32;
                    }
                    ierr = mrc_write_z_float(&mut self.m_heads[hp], &mut li, use_sum, out_sec_num);
                    if ierr != 0 {
                        exit_error_fmt!(
                            "Writing summed data to file for input file # %d (error # %d)",
                            CArg::Int((ifile + 1) as i64),
                            CArg::Int(ierr as i64)
                        );
                    }
                    if adjust_mdoc != 0 && out_num == 0 {
                        if !self.m_doing_frame_ts {
                            if adoc_get_float(
                                ADOC_ZVALUE_NAME,
                                original_zval,
                                b"PixelSpacing",
                                &mut pix_temp,
                            ) != 0
                            {
                                exit_error(b"Getting pixel spacing in mdoc for output image");
                            }
                            if adoc_set_float(
                                ADOC_ZVALUE_NAME,
                                original_zval,
                                b"PixelSpacing",
                                pix_temp * rel_binning,
                            ) != 0
                            {
                                exit_error(b"Adjusting pixel spacing in mdoc for output image");
                            }
                            if adoc_set_three_floats(
                                ADOC_ZVALUE_NAME,
                                original_zval,
                                b"MinMaxMean",
                                tmin,
                                tmax,
                                tmean,
                            ) != 0
                            {
                                exit_error(b"Adjusting pixel spacing in mdoc for output image");
                            }

                            // Binning was a later addition so allow it not to exist
                            ierr = adoc_get_float(
                                ADOC_ZVALUE_NAME,
                                original_zval,
                                b"Binning",
                                &mut pix_temp,
                            );
                            if ierr < 0 {
                                exit_error(b"Getting binning in mdoc for output image");
                            }
                            if ierr == 0 {
                                pix_temp *= rel_binning;
                                if pix_temp as f64 > 0.55 {
                                    pix_temp = b3dnint!(pix_temp) as f32;
                                }
                                if adoc_set_float(
                                    ADOC_ZVALUE_NAME,
                                    original_zval,
                                    b"Binning",
                                    pix_temp,
                                ) != 0
                                {
                                    exit_error(b"Adjusting binning in mdoc for output image");
                                }
                            }
                        }

                        // Change the Z value to be sequential in the mdoc if either ignoring current
                        // Z values or reordering the processing
                        if self.m_ignore_zvalue != 0 || changed_set_order {
                            let text = c_format_bytes("%d", &[CArg::Int(out_sec_num as i64)]);
                            if adoc_change_section_name(ADOC_ZVALUE_NAME, original_zval, &text)
                                .is_err()
                            {
                                exit_error(b"Changing section name to new Z value");
                            }
                        }
                    }
                }
            }

            if self.m_num_bin_tests * self.m_num_filt_tests[0] > 1 {
                let bb = ind_best_bin as usize;
                let bf = ind_best_filt as usize;
                printf!(
                    "%s %d: %s at bin = %d  rad2 = %.3f  sig %.3f  mean res = %.3f  l-o = %.3f\n",
                    CArg::Str(if num_sets != 0 { "Set" } else { "File" }),
                    CArg::Int((ifile + 1) as i64),
                    CArg::Str(if was_good_enough {
                        "Good enough"
                    } else {
                        "Best"
                    }),
                    CArg::Int(bins_to_test[bb] as i64),
                    CArg::Dbl(vary_radius2[bb][bf] as f64),
                    CArg::Dbl(vary_sigma2[bb][bf] as f64),
                    CArg::Dbl(min_mean as f64),
                    CArg::Dbl(min_pred as f64)
                );
                num_times_best[bb][bf] += 1;
            }

            // Find FRC crossing and mean near half-nyquist
            if self.m_test_mode == 0 && self.m_getting_frc {
                self.s_fa.analyze_frc_crossings(
                    &ring_corrs,
                    frc_delta_r,
                    &mut half_cross,
                    &mut quart_cross,
                    &mut eighth_cross,
                    &mut half_nyq,
                );

                if num_summary_lines > 2 {
                    printf!(
                        " FRC crossings 0.5: %.4f  0.25: %.4f  0.125: %.4f  is %.4f at 0.25/pix\n",
                        CArg::Dbl(half_cross as f64),
                        CArg::Dbl(quart_cross as f64),
                        CArg::Dbl(eighth_cross as f64),
                        CArg::Dbl(half_nyq as f64)
                    );
                }
            }
            if ifile < ending_file - 1 && self.m_num_bin_tests * self.m_num_filt_tests[0] > 1 {
                printf!("\n");
            }

            // Output the transforms
            if let Some(ext) = xf_ext.as_deref() {
                // The source opens on the first set (`!ifile`) and closes after set
                // `numSets - 1` of the whole run, so a range of sets not starting at
                // the first, or a second file broken into sets, wrote through an
                // unopened or closed stream (BUGS.md).  Here it opens when none is
                // open and closes after the last set of each file.
                if num_sets == 0 || !xf_open {
                    xf_name = filename.clone();
                    if let Some(t) = find_last_of(&xf_name, b".") {
                        if t + 5 >= xf_name.len() && t > 1 {
                            xf_name.truncate(t + 1);
                        }
                    }
                    xf_name.extend_from_slice(ext);
                    imod_backup_file(&String::from_utf8_lossy(&xf_name));
                    extra_fp = ImodFile::open(os_path(&xf_name), "w");
                    if extra_fp.is_none() {
                        exit_error_fmt!(
                            "Opening output file %s for transforms",
                            CArg::Bytes(&xf_name)
                        );
                    }
                    xf_open = true;
                }
                num_fetch = z_dir * (z_end - z_start) + 1;
                let fp = extra_fp.as_mut().unwrap();
                for group in 0..nz_align {
                    if reverse != 0 {
                        balanced_group_limits(
                            num_fetch,
                            nz_align,
                            group,
                            &mut group_start,
                            &mut group_end,
                        );
                        ind = nz_align - 1 - group;
                    } else {
                        balanced_group_limits(
                            num_fetch,
                            nz_align,
                            nz_align - 1 - group,
                            &mut group_start,
                            &mut group_end,
                        );
                        ind = group;
                    }
                    if start_frame > 0 && group == 0 {
                        group_end += start_frame - 1;
                    }
                    for _ in group_start..=group_end {
                        let _ = fp.write_all(&c_format_bytes(
                            " 1.00000    0.00000    0.00000   1.00000  %8.3f %8.3f\n",
                            &[
                                CArg::Dbl(best_xshifts[ind as usize] as f64),
                                CArg::Dbl(best_yshifts[ind as usize] as f64),
                            ],
                        ));
                    }
                }
                if num_sets == 0 || set_in_file == 0 {
                    extra_fp = None;
                    xf_open = false;
                }
            }

            // Output plottable shifts
            if let Some(fp) = plot_fp.as_mut() {
                for ind in 0..num_done as usize {
                    let _ = fp.write_all(&c_format_bytes(
                        "%3d  %.3f  %.3f\n",
                        &[
                            CArg::Int((10 * ifile + 10) as i64),
                            CArg::Dbl(best_xraw[ind] as f64),
                            CArg::Dbl(best_yraw[ind] as f64),
                        ],
                    ));
                }
                if self.m_do_spline != 0 {
                    for ind in 0..num_done as usize {
                        let _ = fp.write_all(&c_format_bytes(
                            "%3d  %.3f  %.3f\n",
                            &[
                                CArg::Int((10 * ifile + 11) as i64),
                                CArg::Dbl(best_xshifts[ind] as f64),
                                CArg::Dbl(best_yshifts[ind] as f64),
                            ],
                        ));
                    }
                }
            }

            // Output the FRC
            if let Some(fp) = frc_fp.as_mut() {
                for ind in 0..(0.5 / frc_delta_r as f64).floor() as i32 {
                    let _ = fp.write_all(&c_format_bytes(
                        "%2d  %.4f %10.6f\n",
                        &[
                            CArg::Int((ifile + 1) as i64),
                            CArg::Dbl((ind as f64 + 0.5) * frc_delta_r as f64),
                            CArg::Dbl(ring_corrs[ind as usize] as f64),
                        ],
                    ));
                }
            }

            out_sec_num += 1;
            ifile += 1;
        }
        drop(extra_fp);

        if num_files_to_do > 1 && self.m_num_bin_tests * self.m_num_filt_tests[0] > 1 {
            printf!("\nNumber of times each condition is best  (rad2 in parentheses):\n");
            for itest in 0..self.m_num_bin_tests as usize {
                printf!("bin = %d  ", CArg::Int(bins_to_test[itest] as i64));
                for filt in 0..self.m_num_filt_tests[itest] as usize {
                    printf!(
                        "  %3d (%.3f)",
                        CArg::Int(num_times_best[itest][filt] as i64),
                        CArg::Dbl(vary_radius2[itest][filt] as f64)
                    );
                }
                printf!("\n");
            }
        }

        // Write the new mdoc file, possibly with re-ordering
        if adjust_mdoc != 0 {
            sstr = self.m_out_names[0].clone().unwrap_or_default();
            sstr.extend_from_slice(b".mdoc");
            if changed_set_order && adoc_order_write_by_value(Some(ADOC_ZVALUE_NAME)) != 0 {
                exit_error(b"Memory problem in AdocOrderWriteByValue");
            }
            if adoc_write(&sstr) != 0 {
                exit_error(b"Writing adjusted mdoc file");
            }
        }

        // Finish up
        self.s_fa.cleanup();
        drop(summed);
        drop(rot_sum);
        drop(unwgt_sum);
        drop(even_sum);
        drop(odd_sum);
        for ind in 0..3 {
            self.m_partial_scan_bufs[ind] = Vec::new();
        }
        if self.m_test_mode == 0 {
            for out_num in 0..self.m_num_out_files as usize {
                let hp = self.m_out_heads[out_num];
                let mut fp = out_fps[out_num].take().unwrap();
                if mrc_head_write(&mut fp, &mut self.m_heads[hp]) != 0 {
                    exit_error_fmt!(
                        "Writing header to %s output file",
                        CArg::Bytes(title_descs[out_num].as_deref().unwrap_or(b""))
                    );
                }
                self.m_heads[hp].fp = None;
                ii_fclose(&mut fp);
                if let Some(name) = open_ts_names[out_num].as_deref() {
                    let _ = std::fs::remove_file(os_path(name));
                }
            }
        }
        drop(frc_fp);
        drop(plot_fp);
        drop(stack_fp);
        exit(0);
    }

    /// `AliFrame::getNextFilename` (`alignframes.cpp:2170`).
    ///
    /// Return the next filename (for ind) by whatever means they are available
    pub fn get_next_filename(
        &mut self,
        ind: i32,
        num_in_by_opt: i32,
        file_list_fp: Option<&mut ImodFile>,
        frame_path: Option<&[u8]>,
        list_name: &[u8],
        end_reached: &mut bool,
    ) -> Option<Vec<u8>> {
        let mut iz: i32 = 0;
        let ierr: i32;
        let mut sstr: Vec<u8> = Vec::new();
        let mut filename: Vec<u8> = Vec::new();
        let mut tempname: Vec<u8> = Vec::new();

        if self.m_names_from_mdoc {
            // Get the name from the mdoc file and extract it from the path
            if self.m_ignore_zvalue != 0 {
                iz = ind;
            } else {
                iz = adoc_lookup_by_name_value(ADOC_ZVALUE_NAME, ind);
            }
            if iz < 0 {
                exit_error_fmt!(
                    "Looking up section with Z value %d in mdoc file",
                    CArg::Int(ind as i64)
                );
            }

            // The source passes (ind, string) to "%s %d" (BUGS.md).
            if adoc_get_string(ADOC_ZVALUE_NAME, iz, b"SubFramePath", &mut filename) != 0 {
                exit_error_fmt!(
                    "Getting SubFramePath for %s %d in mdoc file",
                    CArg::Str(if self.m_ignore_zvalue != 0 {
                        "section"
                    } else {
                        "Z value"
                    }),
                    CArg::Int(ind as i64)
                );
            }
            self.extract_file_tail(&filename, &mut sstr);
            if let Some(path) = frame_path {
                sstr.insert(0, b'/');
                let mut joined = path.to_vec();
                joined.extend_from_slice(&sstr);
                sstr = joined;
            }
            filename = sstr;

            // Try to get a frame dose line from this section, store the dose data
            if self.m_dose_file_type == 4 {
                ierr = adoc_get_string(
                    ADOC_ZVALUE_NAME,
                    iz,
                    FRAME_DOSE_KEY.as_bytes(),
                    &mut tempname,
                );
                if ierr < 0 {
                    exit_error_fmt!(
                        "Trying to access FrameDosesAndNumber for %s %d in mdoc file",
                        CArg::Str(if self.m_ignore_zvalue != 0 {
                            "section"
                        } else {
                            "Z value"
                        }),
                        CArg::Int(ind as i64)
                    );
                }
                if ierr == 0 {
                    self.m_frame_dose_lines[ind as usize] = tempname;
                }
                self.m_total_dose_vec
                    .push(self.m_dose_from_mdoc[iz as usize]);
                self.m_prior_dose_vec.push(if self.m_dose_accumulates > 0 {
                    self.m_prior_from_mdoc[iz as usize]
                } else {
                    0.
                });
            }
        } else if let Some(fp) = file_list_fp {
            // Or get name from file
            if *end_reached {
                return None;
            }
            loop {
                iz = fgetline(fp, &mut self.m_in_line, MAX_LINE as i32);
                if iz != 0 {
                    break;
                }
            }
            if iz == -2 {
                return None;
            }
            if iz == -1 {
                exit_error_fmt!(
                    "Reading line %d of list of input files %s",
                    CArg::Int((ind + 1) as i64),
                    CArg::Bytes(list_name)
                );
            }
            if iz < 0 {
                *end_reached = true;
            }
            filename = c_str(&self.m_in_line).to_vec();
        } else if ind < num_in_by_opt {
            // Or get arguments
            pip_get_string(b"InputFile", &mut filename);
        } else {
            filename = pip_get_non_option_arg(ind - num_in_by_opt).unwrap_or_default();
        }
        Some(filename)
    }

    /// `AliFrame::openAndReadHeader` (`alignframes.cpp:2244`).
    ///
    /// Open an MRC file and read its header
    pub fn open_and_read_header(
        &self,
        filename: &[u8],
        head: &mut MrcHeader,
        descrip: &[u8],
        test_mode: bool,
    ) -> ImodFile {
        let Some(mut in_fp) = ii_fopen(filename, "rb") else {
            exit_error_fmt!(
                "Opening %s file %s%s",
                CArg::Bytes(descrip),
                CArg::Bytes(filename),
                CArg::Str(if test_mode {
                    "; do not specify an output file when not making sums"
                } else {
                    ""
                })
            );
        };
        if mrc_head_read(&mut in_fp, head) != 0 {
            exit_error_fmt!(
                "Reading header of %s file %s",
                CArg::Bytes(descrip),
                CArg::Bytes(filename)
            );
        }
        in_fp
    }

    /// `AliFrame::checkInputFile` (`alignframes.cpp:2259`).
    ///
    /// Do basic checks on size and mode for a file
    pub fn check_input_file(
        &self,
        filename: &[u8],
        head: &MrcHeader,
        nx: i32,
        ny: i32,
        combine: i32,
    ) {
        if slice_mode_if_real(head.mode) < 0 {
            exit_error_fmt!(
                "File mode for %s is %d; only byte, short, float allowed",
                CArg::Bytes(filename),
                CArg::Int(head.mode as i64)
            );
        }
        if nx > 0 && (nx != head.nx || ny != head.ny) {
            exit_error_fmt!(
                "File %s has a different size (%d x %d) from previous files (%d x %d)",
                CArg::Bytes(filename),
                CArg::Int(head.nx as i64),
                CArg::Int(head.ny as i64),
                CArg::Int(nx as i64),
                CArg::Int(ny as i64)
            );
        }
        if combine != 0 && head.nz > 1 {
            exit_error_fmt!(
                "File %s has more than one slice and cannot be used with -combine",
                CArg::Bytes(filename)
            );
        }
    }

    /// `AliFrame::readAnalyzeSavedFrameList` (`alignframes.cpp:2276`).
    ///
    /// Get a list of frames saved from SEMCCD in a single exposure tilt series
    pub fn read_analyze_saved_frame_list(&mut self, break_set_size: i32) -> bool {
        let mut ierr: i32;
        let mut saved_num: i32;
        let mut last_kept_num: i32;
        let mut ind: usize;
        let mut start_of_set = 0usize;
        let mut max_gap_in_frame_set = 0i32;
        let num_in_list: usize;
        let mut saved_list_name: Vec<u8> = Vec::new();
        let mut in_frame_set: bool;
        let mut single_sets_ok: bool;

        let have_name = pip_get_string(b"SavedFrameListFile", &mut saved_list_name) == 0;
        pip_get_integer(b"MaxGapWithinFrameSet", &mut max_gap_in_frame_set);
        if !have_name {
            return false;
        }
        if self.m_num_in_files != 1 {
            exit_error(b"There must be only a single input file with a saved frame list");
        }
        if break_set_size != 0 {
            exit_error(b"You cannot use the -break option with a saved frame list");
        }
        let Some(mut fp) = ImodFile::open(os_path(&saved_list_name), "r") else {
            exit_error_fmt!(
                "Opening saved frame list file %s",
                CArg::Bytes(&saved_list_name)
            );
        };
        loop {
            ierr = fgetline(&mut fp, &mut self.m_in_line, MAX_LINE as i32);
            if ierr == 0 {
                continue;
            }
            if ierr == -2 {
                break;
            }
            if ierr == -1 {
                exit_error_fmt!(
                    "Reading saved frame list file %s",
                    CArg::Bytes(&saved_list_name)
                );
            }
            let mut end = 0;
            self.m_saved_frames
                .push(strtol(c_str(&self.m_in_line), &mut end, 10) as i32);
            if ierr < 0 {
                break;
            }
        }
        drop(fp);

        num_in_list = self.m_saved_frames.len();
        if num_in_list < 10 {
            exit_error_fmt!(
                "There are only %d numbers in the saved frame list file %s",
                CArg::Int(num_in_list as i64),
                CArg::Bytes(&saved_list_name)
            );
        }

        // Analyze it now so the number is known if tilt angles come in
        // Keep track if negative numbers seen
        in_frame_set = false;
        single_sets_ok = false;
        last_kept_num = -1;
        ind = 0;
        while ind < num_in_list {
            saved_num = self.m_saved_frames[ind];
            if saved_num < 0
                || (last_kept_num < 0 && saved_num >= 0)
                || (in_frame_set && saved_num > last_kept_num + max_gap_in_frame_set + 1)
            {
                if in_frame_set {
                    self.m_set_starts.push(start_of_set as i32);
                    self.m_num_in_sets.push((ind - start_of_set) as i32);
                    in_frame_set = false;
                }
                if saved_num >= 0 {
                    in_frame_set = true;
                    start_of_set = ind;
                } else if ind < num_in_list - 1 {
                    single_sets_ok = true;
                }
            }
            if saved_num >= 0 {
                last_kept_num = saved_num;
            }
            ind += 1;
        }

        // Allow single frame set at start or end if negative values were seen or if there
        // are fewer than 4 frames per set on average (having -1 in file is less ambiguous)
        if !single_sets_ok {
            single_sets_ok = num_in_list < 4 * self.m_num_in_sets.len();
        }
        if in_frame_set && (ind - start_of_set > 1 || single_sets_ok) {
            self.m_set_starts.push(start_of_set as i32);
            self.m_num_in_sets.push((ind - start_of_set) as i32);
        }

        // `mNumInSets[0]` is read natively even when no set was found
        // (BUGS.md); there is nothing to drop then.
        if !self.m_num_in_sets.is_empty() && self.m_num_in_sets[0] == 1 && !single_sets_ok {
            self.m_set_starts.remove(0);
            self.m_num_in_sets.remove(0);
        }
        true
    }

    /// `AliFrame::readTiltAngleFile` (`alignframes.cpp:2356`).
    ///
    /// Read tilt angles from a file
    pub fn read_tilt_angle_file(&mut self, tilt_name: Option<&[u8]>) {
        let mut ierr: i32;
        let mut ix: i32 = 0;
        let mut iy: i32 = 0;
        let Some(tilt_name) = tilt_name else {
            return;
        };
        let Some(mut fp) = ImodFile::open(os_path(tilt_name), "r") else {
            exit_error_fmt!("Opening tilt angle file %s", CArg::Bytes(tilt_name));
        };
        loop {
            ierr = fgetline(&mut fp, &mut self.m_in_line, MAX_LINE as i32);
            if ierr == -2 {
                break;
            }
            if ierr == -1 {
                exit_error_fmt!("Reading tilt angle file %s", CArg::Bytes(tilt_name));
            }
            if ierr > 0 {
                // Convert the angle as a float then look for two integers for frame start/end
                let line = c_str(&self.m_in_line).to_vec();
                let mut end = 0usize;
                self.m_tilt_angles.push(strtod(&line, &mut end) as f32);
                let newptr = end;
                let mut end2 = 0usize;
                ix = strtol(&line[newptr..], &mut end2, 10) as i32;
                iy = -1;
                if end2 == 0 {
                    ix = -1;
                } else {
                    let newptr = newptr + end2;
                    let mut end3 = 0usize;
                    iy = strtol(&line[newptr..], &mut end3, 10) as i32;
                    if end3 == 0 {
                        iy = -1;
                    } else {
                        self.m_rel_frame_starts_found = true;
                    }
                }
                self.m_tilt_rel_start_frame.push(ix);
                self.m_tilt_rel_end_frame.push(iy);
            }
            if ierr < 0 {
                break;
            }
        }
    }

    /// `AliFrame::getAnglesAndTitlesFromMdoc` (`alignframes.cpp:2400`).
    ///
    /// Get tilt angles, axis angle, and titles from an mdoc file
    pub fn get_angles_and_titles_from_mdoc(
        &mut self,
        tilt_name: Option<&[u8]>,
        mut axis_angle: f32,
        angles_only: bool,
    ) {
        let mut ix: i32 = 0;
        let mut iy: i32 = 0;
        let mut iz: i32 = 0;
        let ind: i32;
        let mut ierr: i32;
        let mut got_angle: bool;
        let mut title_angle: f32;
        let mut rot_angle = 0f32;
        let mut sstr: Vec<u8>;
        let mut axstr: Vec<u8>;

        // Get tilt angles if not already got them
        if tilt_name.is_none() {
            self.m_tilt_angles
                .resize(self.m_num_sect.max(0) as usize, 0.);
            for ind in 0..self.m_num_sect {
                if self.m_ignore_zvalue != 0 {
                    iz = ind;
                } else {
                    iz = adoc_lookup_by_name_value(ADOC_ZVALUE_NAME, ind);
                }
                if iz < 0 {
                    exit_error_fmt!(
                        "Looking up section with Z value %d in mdoc file",
                        CArg::Int(ind as i64)
                    );
                }

                // The source passes (ind, string) to "%s %d" (BUGS.md).
                if adoc_get_float(
                    ADOC_ZVALUE_NAME,
                    iz,
                    b"TiltAngle",
                    &mut self.m_tilt_angles[ind as usize],
                ) != 0
                {
                    exit_error_fmt!(
                        "Getting tilt angle for %s %d in mdoc file",
                        CArg::Str(if self.m_ignore_zvalue != 0 {
                            "section"
                        } else {
                            "Z value"
                        }),
                        CArg::Int(ind as i64)
                    );
                }

                // Get start and end frames for a saved frame list
                if self.m_doing_frame_ts && !angles_only {
                    ix = -1;
                    iy = -1;
                    ierr = adoc_get_two_integers(
                        ADOC_ZVALUE_NAME,
                        iz,
                        b"FrameTSStartEndFrames",
                        &mut ix,
                        &mut iy,
                    );
                    if ierr < 0 {
                        exit_error(b"Looking up frame starts and ends in mdoc file");
                    }
                    if ierr == 0 {
                        self.m_rel_frame_starts_found = true;
                    }
                    self.m_tilt_rel_start_frame.push(ix);
                    self.m_tilt_rel_end_frame.push(iy);
                }
            }
        }
        if angles_only {
            return;
        }

        // Get Pixel and size
        adoc_get_float(ADOC_GLOBAL_NAME, 0, b"PixelSpacing", &mut self.m_mdoc_pixel);
        adoc_get_two_integers(
            ADOC_GLOBAL_NAME,
            0,
            b"ImageSize",
            &mut self.m_mdoc_xsize,
            &mut self.m_mdoc_ysize,
        );

        // Get titles
        ind = adoc_get_number_of_sections(b"T").unwrap_or(-1);
        if ind < 0 {
            exit_error(b"Looking up titles in mdoc file");
        }

        for out_num in 0..self.m_num_out_files as usize {
            let hp = self.m_out_heads[out_num];
            self.m_heads[hp].nlabl = 0;

            // Transfer real titles, keep track if axis angle was gotten
            got_angle = false;
            iz = 0;
            while iz < ind {
                let Ok(sect_name) = adoc_get_section_name(b"T", iz) else {
                    exit_error(b"Getting title from mdoc file");
                };
                sstr = sect_name;
                if find_bytes(&sstr, b"TiltAxisAngle").is_some() {
                    // Got a title apparently from FEI software
                    self.m_are_feiframes = true;
                    if axis_angle as f64 > -990. {
                        iz += 1;
                        continue;
                    }

                    // Try to extract the axis angle
                    ix = find_bytes(&sstr, b"=").map_or(-1, |p| p as i32);
                    axstr = sstr.clone();
                    axstr.drain(..(ix + 1) as usize);
                    let mut end = 0usize;
                    title_angle = strtod(&axstr, &mut end) as f32;
                    let end_char = axstr.get(end).copied().unwrap_or(0);
                    if end_char == b' ' || end_char == b',' {
                        // If that checks out correctly, extract a RotationAngle to check against
                        // and get result if it fits different cases
                        if adoc_get_float(ADOC_ZVALUE_NAME, 0, b"RotationAngle", &mut rot_angle)
                            == 0
                        {
                            ix = 1;

                            // The current wrong FEI implementation (4/23/24)
                            if ((-(rot_angle + 90.) - title_angle).abs() as f64) < 0.11 {
                                axis_angle = rot_angle;
                            }
                            // If they corrected it to match SerialEM
                            else if (((rot_angle - 90.) - title_angle).abs() as f64) < 0.11 {
                                axis_angle = title_angle;
                            }
                            // If they sorta corrected it but kept it inverted as in TS file
                            else if ((-(rot_angle - 90.) - title_angle).abs() as f64) < 0.11 {
                                axis_angle = -title_angle;
                            } else {
                                ix = 0;
                            }

                            // Fix the title to be recognized by Etomo
                            if ix != 0 {
                                got_angle = true;
                                let buffer = c_format_bytes(
                                    "  Tilt axis angle = %.2f",
                                    &[CArg::Dbl(axis_angle as f64)],
                                );
                                axstr.drain(..end);
                                sstr = buffer;
                                sstr.extend_from_slice(&axstr);
                            }
                        }
                    }
                } else if find_bytes(&sstr, b"Tilt axis angle").is_some()
                    && sstr.first().copied().unwrap_or(0) != b' '
                {
                    if axis_angle as f64 > -990. {
                        iz += 1;
                        continue;
                    }
                    got_angle = true;
                    let mut padded = b"    ".to_vec();
                    padded.extend_from_slice(&sstr);
                    sstr = padded;
                }

                // The source copies into `labels[iz]` but pads and counts
                // `labels[nlabl]`, which differ once a title is skipped, and has
                // no bound at the ten label slots (BUGS.md).
                let head = &mut self.m_heads[hp];
                if (head.nlabl as usize) < MRC_NLABELS {
                    let slot = head.nlabl as usize;
                    strncpy_label(&mut head.labels[slot], &sstr);
                    fix_title_padding(&mut head.labels[slot]);
                    head.nlabl += 1;
                }
                iz += 1;
            }

            // If no real titles, look for a T = at top of frame stack mdoc
            let mut sect_name = Vec::new();
            if ind == 0 && adoc_get_string(ADOC_GLOBAL_NAME, 0, b"T", &mut sect_name) == 0 {
                // The source pads `labels[iz++]` and leaves `nlabl` at 0, so the
                // title was overwritten by the next one added (BUGS.md).
                let head = &mut self.m_heads[hp];
                let slot = head.nlabl as usize;
                strncpy_label(&mut head.labels[slot], &sect_name);
                fix_title_padding(&mut head.labels[slot]);
                head.nlabl += 1;
            }

            // And if no axis angle gotten, get the rotation angle
            if !got_angle {
                if (axis_angle as f64) < -990.
                    && adoc_get_float(b"FrameSet", 0, b"RotationAngle", &mut axis_angle) == 0
                {
                    axis_angle -= 90.;
                }
                if axis_angle as f64 > -990. {
                    self.add_axis_angle_title(out_num as i32, axis_angle);
                }
            }
        }
    }

    /// `AliFrame::checkTitlesForRefNames` (`alignframes.cpp:2533`).
    ///
    /// Looks in the titles of a frame file for gain reference name and possible defect name
    pub fn check_titles_for_ref_names(
        &mut self,
        filename: &[u8],
        ii_frames: Option<*mut ImodImageFile>,
    ) {
        let mut ind: i32;
        let xml_ind: i32;
        let mut start_ind = 0i32;
        let mut num_item = 0i32;
        let mut err: i32;
        let mut sstr: Vec<u8>;
        let mut file_str: Vec<u8>;
        let mut gain_ref: bool;
        let mut defect: bool;
        let mut value: Option<Vec<u8>> = None;
        let mut root_element: Option<Vec<u8>> = None;

        // Look up metadata for EER file: anything that goes wrong is an error
        if self.m_frames_are_eer {
            let strng = match ii_frames {
                Some(frames) => tiff_get_array(unsafe { &mut *frames }, FEI_EER_METADATA_TAG),
                None => Err(-1),
            };
            let Ok(strng) = strng.and_then(|s| if s.is_empty() { Err(0) } else { Ok(s) }) else {
                exit_error(b"The EER file has no metadata for looking up gain reference");
            };

            // The source copies the `count` bytes into a terminated buffer and then
            // parses the unterminated tag memory instead (BUGS.md); the copy is
            // what is parsed here.
            let str_copy = c_str(&strng).to_vec();
            xml_ind = ixml_load_string(&str_copy, 0, &mut root_element);
            if xml_ind < 0 {
                exit_error_fmt!(
                    "Parsing metadata string from EER file (error %d)",
                    CArg::Int(xml_ind as i64)
                );
            }
            if root_element.as_deref() != Some(b"metadata".as_slice()) {
                exit_error_fmt!(
                    "Root element %s in XML string from EER file",
                    CArg::Str(if root_element.is_some() {
                        "is not \"metadata\""
                    } else {
                        "was not found"
                    })
                );
            }

            // All the elements are item and the have different attributes
            ind = ixml_find_elements(xml_ind, 0, b"item", &mut start_ind, &mut num_item);

            // The source's format has a %d and no argument (BUGS.md).
            if ind != 0 {
                exit_error_fmt!(
                    "Error %d trying to find \"item\" nodes in metadata string",
                    CArg::Int(ind as i64)
                );
            }
            for ind in 0..num_item {
                let mut attr = Vec::new();
                err = ixml_get_string_attribute(xml_ind, start_ind + ind, b"name", &mut attr);
                if err < 0 {
                    exit_error_fmt!(
                        "Error %d getting attribute from item node in EER metadata",
                        CArg::Int(err as i64)
                    );
                }
                if err != 0 {
                    continue;
                }

                // When find the reference oce, get its value
                if attr == b"eerGainReference" {
                    let mut text = Vec::new();
                    err = ixml_get_string_value(xml_ind, start_ind + ind, &mut text);
                    if err != 0 {
                        exit_error_fmt!(
                            "Error %d getting value for \"eerGainReference\" in metadata string",
                            CArg::Int(err as i64)
                        );
                    }
                    value = Some(text);
                    break;
                }
            }

            // It is an error for an EER file not to have a gain reference if this option is
            // given
            let Some(value) = value else {
                exit_error(
                    b"Could not find the gain reference in the metadata string from the EER file",
                );
            };

            // Strip the path and assign to gain name
            file_str = c_str(&value).to_vec();
            if let Some(pos) = find_last_of(&file_str, b"/\\") {
                file_str = file_str[pos + 1..].to_vec();
            }
            self.m_gain_name = Some(file_str);
            ixml_clear(xml_ind);
            return;
        }

        for ind in 0..self.m_in_head.nlabl.clamp(0, MRC_NLABELS as i32) as usize {
            sstr = c_str(&self.m_in_head.labels[ind]).to_vec();

            // Strip the blanks
            let first = sstr.iter().position(|&c| c != b' ');
            let last = sstr.iter().rposition(|&c| c != b' ');
            if let (Some(first), Some(last)) = (first, last) {
                if last > first {
                    sstr = sstr[first..last + 1].to_vec();
                    let ext = sstr.len().wrapping_sub(4);

                    // See if it qualifies as a gain ref (and none already) or defect file
                    gain_ref = (find_bytes(&sstr, b"ref").is_some()
                        || find_bytes(&sstr, b"Ref").is_some())
                        && (find_bytes(&sstr, b".mrc") == Some(ext)
                            || find_bytes(&sstr, b".dm4") == Some(ext)
                            || find_bytes(&sstr, b".tif") == Some(ext))
                        && self.m_gain_name.is_none();
                    defect = find_bytes(&sstr, b"defect").is_some()
                        && find_bytes(&sstr, b".txt") == Some(ext)
                        && self.m_defect_name.is_none();
                    if defect || gain_ref {
                        file_str = filename.to_vec();

                        // If there is a path to the frame filename, add it to front
                        if let Some(pos) = find_last_of(&file_str, b"/\\") {
                            let mut joined = file_str[..pos + 1].to_vec();
                            joined.extend_from_slice(&sstr);
                            sstr = joined;
                        }

                        // Look for the file and accept it
                        if std::fs::metadata(os_path(&sstr)).is_ok() {
                            if gain_ref {
                                self.m_gain_name = Some(sstr.clone());
                            } else {
                                self.m_defect_name = Some(sstr.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    /// `AliFrame::addAxisAngleTitle` (`alignframes.cpp:2642`).
    ///
    /// Adds title with tilt axis angle to one of the output headers
    pub fn add_axis_angle_title(&mut self, out_num: i32, axis_angle: f32) {
        let hp = self.m_out_heads[out_num as usize];
        let mut iz = b3dmin!(self.m_heads[hp].nlabl, 8);
        let line = c_format_bytes(
            "    Tilt axis angle = %.1f",
            &[CArg::Dbl(axis_angle as f64)],
        );
        self.m_in_line = [0; MAX_LINE + 1];
        self.m_in_line[..line.len().min(MAX_LINE)]
            .copy_from_slice(&line[..line.len().min(MAX_LINE)]);
        let head = &mut self.m_heads[hp];
        strncpy_label(&mut head.labels[iz as usize], &self.m_in_line);
        fix_title_padding(&mut head.labels[iz as usize]);
        iz += 1;
        head.nlabl = iz;
    }

    /// `AliFrame::handleTooManyTiltAngles` (`alignframes.cpp:2655`).
    ///
    /// If there are too many tilt angles howevere they came in, try to sort out what is
    /// missing from a frame tilt series and give warnings
    pub fn handle_too_many_tilt_angles(&mut self, progname: &[u8]) {
        let mut trim_tilt = false;
        let ix: i32;
        let mut iz: i32 = 0;
        if self.m_tilt_angles.len() as i32 > self.m_num_in_files
            && self.m_doing_frame_ts
            && self.m_rel_frame_starts_found
        {
            ix = self.m_saved_frames[self.m_set_starts[0] as usize];
            printf!(
                "There are more tilt angles than frame sets: analyzing starting and ending frames\n"
            );

            // For each tilt angle, look at relative frame starts and find overlap with actual
            // relative frame starts
            for ind in (0..self.m_tilt_angles.len()).rev() {
                if self.m_tilt_rel_start_frame[ind] < 0 || self.m_tilt_rel_end_frame[ind] < 0 {
                    continue;
                }
                trim_tilt = true;
                for iy in 0..self.m_set_starts.len() {
                    iz = self.m_saved_frames[self.m_set_starts[iy] as usize] - ix;

                    // If overlap is found, it is good
                    if !(iz > self.m_tilt_rel_end_frame[ind]
                        || iz + self.m_num_in_sets[iy] < self.m_tilt_rel_start_frame[ind])
                    {
                        trim_tilt = false;
                        break;
                    }
                }

                // No overlap, remove the tilt
                if trim_tilt {
                    printf!(
                        "All frames seem to be lost from %.1f deg tilt\n",
                        CArg::Dbl(self.m_tilt_angles[ind] as f64)
                    );
                    self.m_tilt_angles.remove(ind);
                    self.m_tilt_rel_start_frame.remove(ind);
                    self.m_tilt_rel_end_frame.remove(ind);
                    if !self.m_total_dose_vec.is_empty() && self.m_total_dose_vec.len() > ind {
                        self.m_total_dose_vec.remove(ind);
                        self.m_prior_dose_vec.remove(ind);
                    }
                }
            }
        }
        if (self.m_tilt_angles.len() as i32) < self.m_num_in_files {
            exit_error_fmt!(
                "There are %sfewer tilt angles in the file (%d) than frame files or sets (%d)",
                CArg::Str(if trim_tilt { "now " } else { "" }),
                CArg::Int(self.m_tilt_angles.len() as i64),
                CArg::Int(self.m_num_in_files as i64)
            );
        }
        if self.m_tilt_angles.len() as i32 > self.m_num_in_files {
            printf!(
                "WARNING: %s - There are %sfewer frame files or sets (%d) than tilt angles in the file (%d)\n",
                CArg::Bytes(progname),
                CArg::Str(if trim_tilt { "still " } else { "" }),
                CArg::Int(self.m_num_in_files as i64),
                CArg::Int(self.m_tilt_angles.len() as i64)
            );
        }
    }

    /// `AliFrame::assessGpuNeeds` (`alignframes.cpp:2706`).
    ///
    /// Determine what if anything can be done on the GPU
    pub fn assess_gpu_needs(&mut self, use_shr_mem: i32, frc_name: Option<&[u8]>) {
        let gpu_frac_mem = 0.85f32;
        let mut gpu_memory = 0f32;
        let mut needed: f32;
        let mut need_for_ali = 0f32;
        let mut gpu_usable_mem: f32;
        let mut need_for_gpusum = 0f32;
        let tot_need: f32;
        let need_for_pre_ops: f32;
        let mut nz_align: i32;
        let mut ind: i32;
        let mut sum_with_align: bool;
        let normalize = self.m_gain_slice.is_some() && !self.m_antialias_eer;
        let need_preprocess = self.m_trunc_limit != 0. || self.m_cam_size_x > 0 || normalize;

        if self.m_use_gpu < 0 {
            return;
        }
        let _ = use_shr_mem;
        if self
            .s_fa
            .gpu_available(self.m_use_gpu, &mut gpu_memory, self.m_debug % 10)
            == 0
        {
            self.m_use_gpu = -1;
        }
        if self.m_use_gpu < 0 {
            return;
        }

        gpu_usable_mem = gpu_memory * gpu_frac_mem;
        if self.m_gpu_mem_limit > 0. {
            gpu_usable_mem = (1024. * 1024. * 1024. * self.m_gpu_mem_limit as f64) as f32;
        } else if self.m_gpu_mem_limit < 0. {
            gpu_usable_mem = -gpu_memory * self.m_gpu_mem_limit;
        }
        needed = 0.;
        nz_align = self.m_max_num_z;
        if self.m_use_block_group {
            nz_align = b3dmax!(1, nz_align / self.m_group_size);
        }
        FrameAlign::gpu_memory_needs(
            self.m_full_pad_size,
            self.m_sum_pad_size,
            self.m_align_pad_size,
            self.m_num_all_vs_all,
            nz_align,
            self.m_refine_at_end,
            self.m_group_size,
            &mut need_for_gpusum,
            &mut need_for_ali,
        );
        if self.m_num_out_files > 1 {
            need_for_gpusum += self.m_sum_pad_size;
        }

        // First see if summing can be done
        if self.m_test_mode == 0 {
            if need_for_gpusum > gpu_usable_mem {
                printf!(
                    "Insufficient memory on GPU to use it for summing (%.0f MB needed of %.0f MB total)\n",
                    CArg::Dbl(need_for_gpusum as f64 / 1.048e6),
                    CArg::Dbl(gpu_memory as f64 / 1.048e6)
                );
                self.m_use_gpu = -1;
                return;
            } else {
                self.m_gpu_flags = GPU_FOR_SUMMING
                    + if self.m_num_out_files > 1 {
                        GPU_DO_UNWGT_SUM
                    } else {
                        0
                    };
                needed = need_for_gpusum;
            }
        }

        // Summing will be done with alignment if only one binning, only one filter or
        // using hybrid shift, and not assessing or doing spline or refining at end
        sum_with_align = ((self.m_num_bin_tests == 1
            && (self.m_hybrid_shifts != 0 || self.m_num_filt_tests[0] == 1))
            || self.m_start_assess >= 0)
            && self.m_test_mode == 0
            && self.m_do_spline == 0
            && self.m_refine_at_end == 0;

        // If that is the case, and alignment alone would fit but both would not,
        // then see if deferring the sum will require less memory than the limit and if
        // so, then defer the summing
        // Call this first unconditionally to get mNumHoldFull set properly
        FrameAlign::total_memory_needs(
            self.m_full_pad_size,
            4,
            self.m_sum_pad_size,
            self.m_align_pad_size,
            self.m_num_all_vs_all,
            nz_align,
            self.m_refine_at_end,
            self.m_num_bin_tests,
            self.m_num_filt_tests[0],
            self.m_hybrid_shifts,
            self.m_group_size,
            self.m_do_spline,
            GPU_FOR_ALIGNING,
            0,
            self.m_test_mode,
            self.m_start_assess,
            &mut self.m_sum_in_one_pass,
            &mut self.m_num_hold_full,
        );
        if sum_with_align && need_for_ali < gpu_usable_mem && need_for_ali + needed > gpu_usable_mem
        {
            tot_need = FrameAlign::total_memory_needs(
                self.m_full_pad_size,
                4,
                self.m_sum_pad_size,
                self.m_align_pad_size,
                self.m_num_all_vs_all,
                nz_align,
                self.m_refine_at_end,
                self.m_num_bin_tests,
                self.m_num_filt_tests[0],
                self.m_hybrid_shifts,
                self.m_group_size,
                self.m_do_spline,
                GPU_FOR_ALIGNING,
                1,
                self.m_test_mode,
                self.m_start_assess,
                &mut self.m_sum_in_one_pass,
                &mut self.m_num_hold_full,
            );
            if tot_need < self.m_memory_limit {
                sum_with_align = false;
                self.m_defer_sum = 1;
            }
        }

        // If summing is done with aligning add the usage for summing if any
        if sum_with_align {
            need_for_ali += needed;
        }

        // Decide if alignment can be done there
        if need_for_ali > gpu_usable_mem {
            printf!(
                "Insufficient memory on GPU to do alignment (%.0f MB needed of %.0f MB total)\n",
                CArg::Dbl(need_for_ali as f64 / 1.048e6),
                CArg::Dbl(gpu_memory as f64 / 1.048e6)
            );
        } else {
            if sum_with_align {
                needed = need_for_ali;
            }
            self.m_gpu_flags |= GPU_FOR_ALIGNING;
        }

        // Now if summing, see if there is room for even/odd
        if (self.m_gpu_flags & GPU_FOR_SUMMING) != 0 && self.m_getting_frc {
            needed += self.m_sum_pad_size;
            if needed > gpu_usable_mem {
                printf!(
                    "Insufficient memory on GPU to get even/odd sums for FRC (%.0f MB needed of %.0f MB total)\n",
                    CArg::Dbl(needed as f64 / 1.048e6),
                    CArg::Dbl(gpu_memory as f64 / 1.048e6)
                );
                if frc_name.is_some() {
                    exit_error(b"Insufficient memory on GPU to get FRC output");
                }
                self.m_getting_frc = false;
                needed -= self.m_sum_pad_size;
            } else {
                self.m_gpu_flags |= GPU_DO_EVEN_ODD;
            }
        }

        // Now see about adding bin, noise, and preproc to GPU
        if self.m_gpu_flags != 0 {
            // If option is entered, just use it to set the flags
            ind = 0;
            if pip_get_integer(b"FlagsForGPU", &mut ind) == 0 {
                if ind % 10 != 0 {
                    self.m_gpu_flags |= GPU_DO_NOISE_TAPER;
                }
                if (ind / 10) % 10 != 0 {
                    self.m_gpu_flags |= GPU_DO_BIN_PAD;
                }
                if (self.m_gpu_flags & (GPU_DO_NOISE_TAPER | GPU_DO_BIN_PAD)) != 0 {
                    if (ind / 100) % 10 != 0 {
                        self.m_gpu_flags |= STACK_FULL_ON_GPU;
                        if ind / 10000 != 0 {
                            self.m_gpu_flags |=
                                GPU_STACK_LIMITED + ((ind / 10000) << GPU_STACK_LIM_SHIFT);
                        }
                    }
                    if (ind / 1000 % 10) != 0 && need_preprocess {
                        self.m_gpu_flags |= GPU_DO_PREPROCESS;
                        if normalize {
                            self.m_gpu_flags |= GPU_DO_GAIN_NORM;
                        }
                        if self.m_cam_size_x > 0 {
                            self.m_gpu_flags |= GPU_CORRECT_DEFECTS;
                        }
                    }
                }
            } else {
                // Otherwise, need to analyze which preprocessing/bin/pad operations to perform
                let in_flags = self.m_gpu_flags;
                need_for_pre_ops = FrameAlign::find_preproc_pad_gpu_flags(
                    self.m_nx,
                    self.m_ny,
                    if self.m_dark_slice.is_some() {
                        std::mem::size_of::<f32>() as i32
                    } else {
                        self.m_max_data_size
                    },
                    self.m_min_binning_to_test,
                    self.m_dark_slice.is_none() && normalize,
                    self.m_dark_slice.is_none() && self.m_cam_size_x > 0,
                    self.m_dark_slice.is_none() && self.m_trunc_limit != 0.,
                    b3dmax!(self.m_num_hold_full, 1),
                    gpu_usable_mem - needed,
                    (0.5 * (1. - gpu_frac_mem as f64) * gpu_usable_mem as f64) as f32,
                    in_flags,
                    &mut self.m_gpu_flags,
                );
                needed += need_for_pre_ops;
            }
            let _ = needed;
            ind = (self.m_gpu_flags >> GPU_STACK_LIM_SHIFT) & GPU_STACK_LIM_MASK;
            if self.m_debug != 0 && (self.m_gpu_flags & (GPU_DO_NOISE_TAPER | GPU_DO_BIN_PAD)) != 0
            {
                printf!(
                    "%s  %s  %s  %s %s %d on GPU\n",
                    CArg::Str(if (self.m_gpu_flags & GPU_DO_NOISE_TAPER) != 0 {
                        "noise-pad"
                    } else {
                        ""
                    }),
                    CArg::Str(if (self.m_gpu_flags & GPU_DO_BIN_PAD) != 0 {
                        "bin-pad"
                    } else {
                        ""
                    }),
                    CArg::Str(if (self.m_gpu_flags & GPU_DO_PREPROCESS) != 0 {
                        "preprocess"
                    } else {
                        ""
                    }),
                    CArg::Str(if (self.m_gpu_flags & STACK_FULL_ON_GPU) != 0 {
                        "stack"
                    } else {
                        ""
                    }),
                    CArg::Str(if (self.m_gpu_flags & GPU_STACK_LIMITED) != 0 {
                        "limit"
                    } else {
                        "   "
                    }),
                    CArg::Int(if (self.m_gpu_flags & GPU_STACK_LIMITED) != 0 {
                        ind as i64
                    } else {
                        0
                    })
                );
            }

            // Can stack smaller size if doing either operation on GPU and either there is
            // no preprocess or preprocessing is on GPU
            if (self.m_gpu_flags & (GPU_DO_BIN_PAD | GPU_DO_NOISE_TAPER)) != 0
                && (!need_preprocess || (self.m_gpu_flags & GPU_DO_PREPROCESS) != 0)
            {
                self.m_full_data_size = self.m_max_data_size;
            }
        }
    }

    /// `AliFrame::readOneFrame` (`alignframes.cpp:2867`).
    ///
    /// Read one frame of data in parallel for TIFF file or with regular call
    pub fn read_one_frame(&mut self, read_buf: &mut [u8], iz_read: i32, ifile: i32) {
        let wall_start = wall_time();
        if self.m_parallel_read {
            if unsafe {
                tiff_parallel_read(
                    self.m_file_copies.as_mut_ptr(),
                    self.m_num_read_threads,
                    0,
                    self.m_nx - 1,
                    0,
                    self.m_ny - 1,
                    self.m_in_data_size,
                    read_buf.as_mut_ptr(),
                    iz_read,
                    MRSA_NOPROC,
                )
            } != 0
            {
                exit_error_fmt!(
                    "Reading frame %d from file # %d: %s",
                    CArg::Int(iz_read as i64),
                    CArg::Int((ifile + 1) as i64),
                    CArg::Str(&b3d_get_error())
                );
            }
        } else if mrc_read_slice(
            read_buf,
            self.m_in_fp.as_mut().unwrap(),
            &mut self.m_in_head,
            iz_read,
            b'Z',
        ) != 0
        {
            exit_error_fmt!(
                "Reading frame %d from file # %d",
                CArg::Int(iz_read as i64),
                CArg::Int((ifile + 1) as i64)
            );
        }
        self.m_wall_read += wall_time() - wall_start;
    }

    /// `AliFrame::addToSumBuffer` (`alignframes.cpp:2886`).
    ///
    /// Add a read-in image to a sum buffer of the proper type.  Both buffers
    /// are frame storage (`f32`-aligned), so the typed views are aligned.
    pub fn add_to_sum_buffer(
        &self,
        read_buf: &[u8],
        in_mode: i32,
        sum_buf: &mut [u8],
        use_mode: &mut i32,
        nxy: i32,
    ) {
        let nxy = nxy as usize;
        let b_data = read_buf;
        let s_data = unsafe {
            std::slice::from_raw_parts(read_buf.as_ptr().cast::<i16>(), read_buf.len() / 2)
        };
        let us_data = unsafe {
            std::slice::from_raw_parts(read_buf.as_ptr().cast::<u16>(), read_buf.len() / 2)
        };
        let f_data = unsafe {
            std::slice::from_raw_parts(read_buf.as_ptr().cast::<f32>(), read_buf.len() / 4)
        };
        let sum_ptr = sum_buf.as_mut_ptr();
        let sum_len = sum_buf.len();
        match in_mode {
            MRC_MODE_BYTE => {
                *use_mode = SLICE_MODE_SHORT;
                let s_buf =
                    unsafe { std::slice::from_raw_parts_mut(sum_ptr.cast::<i16>(), sum_len / 2) };
                for ix in 0..nxy {
                    s_buf[ix] = (s_buf[ix] as i32 + b_data[ix] as i32) as i16;
                }
            }
            MRC_MODE_SHORT => {
                *use_mode = SLICE_MODE_SHORT;
                let s_buf =
                    unsafe { std::slice::from_raw_parts_mut(sum_ptr.cast::<i16>(), sum_len / 2) };
                for ix in 0..nxy {
                    s_buf[ix] = (s_buf[ix] as i32 + s_data[ix] as i32) as i16;
                }
            }
            MRC_MODE_USHORT => {
                *use_mode = SLICE_MODE_USHORT;
                let us_buf =
                    unsafe { std::slice::from_raw_parts_mut(sum_ptr.cast::<u16>(), sum_len / 2) };
                for ix in 0..nxy {
                    us_buf[ix] = (us_buf[ix] as i32 + us_data[ix] as i32) as u16;
                }
            }
            MRC_MODE_FLOAT => {
                *use_mode = SLICE_MODE_FLOAT;
                let f_buf =
                    unsafe { std::slice::from_raw_parts_mut(sum_ptr.cast::<f32>(), sum_len / 4) };
                for ix in 0..nxy {
                    f_buf[ix] += f_data[ix];
                }
            }
            _ => {}
        }
    }

    /// `AliFrame::extractFileTail` (`alignframes.cpp:2922`).
    ///
    /// Get the tail from a filename, i.e. strip the path
    pub fn extract_file_tail(&self, filename: &[u8], sstr: &mut Vec<u8>) {
        *sstr = filename.to_vec();
        let iz = match find_last_of(sstr, b"/\\") {
            None => 0,
            Some(pos) => pos + 1,
        };
        *sstr = sstr[iz..].to_vec();
    }

    /// `AliFrame::adjustTitleBinning` (`alignframes.cpp:2937`).
    ///
    /// Checks if the given title has the binning value in it, adjusts by relBinning and places
    /// into sstr, and gives message if printChange true.  Returns 1 if binning found
    pub fn adjust_title_binning(
        &self,
        title: &[u8],
        sstr: &mut Vec<u8>,
        mut rel_binning: f32,
        print_change: bool,
    ) -> i32 {
        let title = c_str(title);
        let at = |i: usize| title.get(i).copied().unwrap_or(0);
        let ierr: usize;
        let stack_bin: i32;
        let bin_text: Vec<u8>;
        let mut iz: usize;
        *sstr = title.to_vec();

        // Find binning = string
        match find_bytes(sstr, b"binning =") {
            None | Some(0) => return 0,
            Some(pos) => iz = pos,
        }
        iz += 9;

        // Find beginning and end of the binning number
        while at(iz) == b' ' {
            iz += 1;
        }
        if at(iz) == 0x00 {
            return 0;
        }
        ierr = iz;
        while at(iz) != b' ' && at(iz) != 0x00 {
            iz += 1;
        }
        let mut end = 0;
        stack_bin = strtol(&title[ierr..], &mut end, 10) as i32;

        // Make sure it reads, then adjust and make sure that is a sensible value
        if stack_bin == 0 {
            return 0;
        }
        rel_binning *= stack_bin as f32;
        if (rel_binning as f64) < 0.46
            || (rel_binning as f64 > 0.54 && (rel_binning as f64) < 0.96)
            || (rel_binning as f64 > 0.96
                && ((b3dnint!(rel_binning) as f32 - rel_binning).abs() as f64) > 0.04)
        {
            return 0;
        }

        // Replace with new value
        if (rel_binning as f64) < 0.54 {
            bin_text = c_format_bytes("%.1f", &[CArg::Dbl(rel_binning as f64)]);
            if print_change {
                printf!(
                    "Adjusted binning in header title to %.1f\n",
                    CArg::Dbl(rel_binning as f64)
                );
            }
        } else {
            bin_text = c_format_bytes("%d", &[CArg::Int(b3dnint!(rel_binning) as i64)]);
            if print_change {
                printf!(
                    "Adjusted binning in header title to %d\n",
                    CArg::Int(b3dnint!(rel_binning) as i64)
                );
            }
        }
        sstr.splice(ierr..iz, bin_text);
        if sstr.first().copied().unwrap_or(0) != b' ' {
            sstr.splice(0..0, b"    ".iter().copied());
        }
        1
    }

    /// `AliFrame::minMaxSetSize` (`alignframes.cpp:2989`).
    ///
    /// Determine actual minimum and maximum size for sets or groups given the nominal size and
    /// the total to be divided into the groups.
    pub fn min_max_set_size(
        &self,
        basic_size: i32,
        num_frames: i32,
        min_size: &mut i32,
        max_size: &mut i32,
    ) {
        let num_sets = num_frames / basic_size;
        let remainder = num_frames % basic_size;
        *min_size = basic_size + remainder / num_sets;
        *max_size = *min_size + if remainder % num_sets > 0 { 1 } else { 0 };
    }

    /// `AliFrame::getGainDarkDefects` (`alignframes.cpp:3000`).
    ///
    /// Read in gain reference, dark reference, and defect file
    pub fn get_gain_dark_defects(&mut self, use_shr_mem: i32) {
        let mut gain_head = MrcHeader::default();
        let mut dark_head = MrcHeader::default();
        let mut extra_name: Vec<u8> = Vec::new();
        let mut ind: i32;
        let yfac: i32;
        let super_fac: i32;
        let retval: i32;
        let mut scale_defects = 0i32;
        let use_fac: i32;
        let mut images_binned = -1.0f32;
        let mut mess_buf = String::new();
        let ii_gain: Option<*mut ImodImageFile>;
        let super_res_ok: bool;
        let gain_is_tiff: bool;
        let mut ref_temp: Vec<f32>;
        let mut num_in_x = 0i32;
        let mut x_start = 0i32;
        let mut x_interval = 0i32;
        let mut num_in_y = 0i32;
        let mut y_start = 0i32;
        let mut y_interval = 0i32;
        let mut biases: Vec<Vec<f32>> = Vec::new();
        let _ = use_shr_mem;

        // Defect file first to override defects in Falcon gain ref
        if let Some(defect_name) = self.m_defect_name.clone() {
            let defects = Rc::get_mut(&mut self.m_defects).unwrap();
            let ierr = cor_def_parse_defects(
                &String::from_utf8_lossy(&defect_name),
                false,
                defects,
                &mut self.m_cam_size_x,
                &mut self.m_cam_size_y,
            );
            if ierr != 0 {
                exit_error_fmt!(
                    "%s defect file %s\n",
                    CArg::Str(if ierr == 1 {
                        "Opening"
                    } else {
                        "Reading or parsing lines in"
                    }),
                    CArg::Bytes(&defect_name)
                );
            }
            if self.m_cam_size_x == 0 || self.m_cam_size_y == 0 {
                exit_error(b"Defect list file must have CameraSizeX and CameraSizeY entries");
            }
            pip_get_float(b"ImagesAreBinned", &mut images_binned);
            pip_get_integer(b"DoubleDefectCoords", &mut scale_defects);
            cor_def_flip_defects_in_y(defects, self.m_cam_size_x, self.m_cam_size_y, 0);
            cor_def_find_touching_pixels(defects, self.m_cam_size_x, self.m_cam_size_y, 0);
            if cor_def_setup_to_correct(
                self.m_in_head.nx,
                self.m_in_head.ny,
                defects,
                &mut self.m_cam_size_x,
                &mut self.m_cam_size_y,
                scale_defects,
                images_binned,
                &mut self.m_cor_def_binning,
                Some("-imagebinned"),
            ) != 0
            {
                exit_error(
                    b"Image size is more than twice the size stored in the camera defect list",
                );
            }
        }

        // Gain reference
        self.m_nx_gain = 0;
        self.m_ny_gain = 0;
        self.m_super_fac_for_defects = 0;
        if let Some(gain_name) = self.m_gain_name.clone() {
            let mut extra_fp =
                self.open_and_read_header(&gain_name, &mut gain_head, b"gain reference", false);
            if gain_head.mode != MRC_MODE_FLOAT {
                exit_error(b"Gain reference must be floating point");
            }
            let Some(slice) = slice_read_mrc(&mut gain_head, 0, b'Z') else {
                exit_error(b"Reading gain reference file");
            };
            let MrcData::F(gain_data) = slice.data else {
                exit_error(b"Reading gain reference file");
            };
            let mut gain_data = gain_data;

            // General evaluation of super-resolution relative to the gain
            super_fac = self.m_in_head.nx / gain_head.nx;
            yfac = self.m_in_head.ny / gain_head.ny;
            super_res_ok = !(yfac != super_fac
                || super_fac * gain_head.nx != self.m_in_head.nx
                || super_fac * gain_head.ny != self.m_in_head.ny
                || (super_fac != 1 && super_fac != 2 && super_fac != 4));

            // Find out if frames are in an EER file and if gain is in a TIFF file
            ii_gain = ii_lookup_file_from_fp(&extra_fp);
            gain_is_tiff = ii_gain.is_some() && unsafe { (*ii_gain.unwrap()).file } == IIFILE_TIFF;

            if (gain_is_tiff || self.m_frames_are_eer) && !super_res_ok {
                exit_error_fmt!(
                    "Image file size (%d x %d) must be exactly the same, twice, or 4 times the gain reference size (%d x %d)",
                    CArg::Int(self.m_in_head.nx as i64),
                    CArg::Int(self.m_in_head.ny as i64),
                    CArg::Int(gain_head.nx as i64),
                    CArg::Int(gain_head.ny as i64)
                );
            }

            // If no defects entered, look for defects in a TIFF gain file
            if self.m_cam_size_x == 0 && gain_is_tiff {
                let defects = Rc::get_mut(&mut self.m_defects).unwrap();
                retval = cor_def_process_fei_defects(
                    unsafe { &mut *ii_gain.unwrap() },
                    defects,
                    gain_head.nx,
                    gain_head.ny,
                    true,
                    super_fac,
                    self.m_fei_defect_pad,
                    None,
                    &mut mess_buf,
                    256,
                );
                if retval > 0 {
                    exit_error(mess_buf.as_bytes());
                }
                if retval == 0 {
                    if super_fac > 1 && defects.falcon_type != 0 && defects.num_avg_super_res > 0 {
                        self.m_super_fac_for_defects = super_fac;
                    }
                    self.m_cam_size_x = self.m_in_head.nx;
                    self.m_cam_size_y = self.m_in_head.ny;
                }
            }

            // Finish up with the gain file and apply rotation
            ii_fclose(&mut extra_fp);
            gain_head.fp = None;
            self.m_nx_gain = gain_head.nx;
            self.m_ny_gain = gain_head.ny;
            if self.m_rotation_flip != 0 {
                self.rotate_flip_gain_reference(&mut gain_data);
            }

            // Now expand the gain reference for super-resolution
            if (gain_is_tiff || self.m_frames_are_eer) && (super_fac > 1 || self.m_antialias_eer) {
                use_fac = if self.m_antialias_eer { 4 } else { super_fac };
                ref_temp = vec![0.; (self.m_nx_gain * self.m_ny_gain * use_fac * use_fac) as usize];
                cor_def_expand_gain_reference(
                    &gain_data,
                    gain_head.nx,
                    gain_head.ny,
                    use_fac,
                    &mut ref_temp,
                );
                self.m_nx_gain *= use_fac;
                self.m_ny_gain *= use_fac;
                gain_data = ref_temp;

                if pip_get_string(b"SuperGainFactorFile", &mut extra_name) == 0 {
                    ind = cor_def_read_super_gain(
                        &String::from_utf8_lossy(&extra_name),
                        use_fac,
                        &mut biases,
                        &mut num_in_x,
                        &mut x_start,
                        &mut x_interval,
                        &mut num_in_y,
                        &mut y_start,
                        &mut y_interval,
                    );
                    if ind != 0 {
                        exit_error_fmt!(
                            "Reading file with super-resolution gain adjustments (error %d)",
                            CArg::Int(ind as i64)
                        );
                    }
                    cor_def_refine_super_res_ref(
                        &mut gain_data,
                        self.m_nx_gain,
                        self.m_ny_gain,
                        use_fac,
                        &biases,
                        num_in_x,
                        x_start,
                        x_interval,
                        num_in_y,
                        y_start,
                        y_interval,
                    );
                }
            }
            let mut gain_rc = Rc::new(gain_data);

            // `tiffGainReferenceForEER(refTemp)` precedes the refinement in the
            // source, but it only stores the pointer, which reads the refined
            // reference when frames are read; registered once the data has its
            // final home.
            if (gain_is_tiff || self.m_frames_are_eer)
                && (super_fac > 1 || self.m_antialias_eer)
                && self.m_antialias_eer
            {
                let data = Rc::get_mut(&mut gain_rc).unwrap();
                tiff_gain_reference_for_eer_bytes(buf_bytes_mut(data));
            }
            self.m_gain_slice = Some(gain_rc);

            if self.m_nx_gain < self.m_nx || self.m_ny_gain < self.m_ny {
                exit_error_fmt!(
                    "Gain reference is smaller than image in %s%s%s",
                    CArg::Str(if self.m_nx_gain < self.m_nx { "X" } else { "" }),
                    CArg::Str(
                        if self.m_nx_gain < self.m_nx && self.m_ny_gain < self.m_ny {
                            " and "
                        } else {
                            ""
                        }
                    ),
                    CArg::Str(if self.m_ny_gain < self.m_ny { "Y" } else { "" })
                );
            }

            // Recognize K3 and set scaling to 32
            for ind in 1..=2 {
                if (self.m_nx_gain == ind * 5760 && self.m_ny_gain == ind * 4092)
                    || (self.m_ny_gain == ind * 5760 && self.m_nx_gain == ind * 4092)
                {
                    self.m_default_byte_scale = 32.;
                }
            }
        } else if self.m_extra_has_gain_ref != 0 {
            let Some(slice) = slice_create(self.m_nx, self.m_ny, SLICE_MODE_FLOAT) else {
                exit_error(b"Allocating memory for gain reference");
            };
            let MrcData::F(data) = slice.data else {
                exit_error(b"Allocating memory for gain reference");
            };
            self.m_gain_slice = Some(Rc::new(data));
        }

        // Dark reference
        if pip_get_string(b"DarkReferenceFile", &mut extra_name) == 0 {
            let mut extra_fp =
                self.open_and_read_header(&extra_name, &mut dark_head, b"dark reference", false);
            if dark_head.mode != MRC_MODE_SHORT && dark_head.mode != MRC_MODE_USHORT {
                exit_error(b"Dark reference must be signed or unsigned short integers");
            }
            if dark_head.nx != self.m_nx || dark_head.ny != self.m_ny {
                exit_error(b"Dark reference is not the same size as the image");
            }
            self.m_dark_slice = slice_read_mrc(&mut dark_head, 0, b'Z');
            if self.m_dark_slice.is_none() {
                exit_error(b"Reading dark reference file");
            }
            dark_head.fp = None;
            ii_fclose(&mut extra_fp);
        }
    }

    /// `AliFrame::analyzeExtraHeader` (`alignframes.cpp:3151`).
    ///
    /// Analyze the extra header from a UCSFtomo file for pixel size, rotation angle, and
    /// tilt angles and determine division into groups by tilt angle
    pub fn analyze_extra_header(
        &mut self,
        in_fp: &mut ImodFile,
        head: &mut MrcHeader,
        start_frame: i32,
        end_frame: i32,
        axis_pix_only: bool,
        buffer: &mut Vec<f32>,
        buf_size: &mut i32,
        tilts: &mut Vec<f32>,
        min_set: &mut i32,
        max_set: &mut i32,
        axis_angle: &mut f32,
        pix_size: &mut f32,
    ) -> i32 {
        let num_int_real = head.nint as i32 + head.nreal as i32;
        let num = num_int_real * head.nz;
        let mut iz: i32 = 0;
        let mut set: i32;
        let iz_start: i32;
        let iz_end: i32;
        let mut size: i32;
        let mut last_size = 0i32;

        // `numAtSize` is read uninitialised natively at the first tilt
        // change (BUGS.md); it cannot change the outcome there.
        let mut num_at_size = 0i32;
        let mut ret_val = 0i32;
        let mut got_double = 0i32;
        let mut temp: f32;
        let mut last_tilt = 0f32;
        if num == 0 {
            return 0;
        }
        *axis_angle = -999.;
        if *buf_size < num {
            *buffer = vec![0.; num as usize];
            *buf_size = num;
        }
        if b3d_fseek(in_fp, MRC_HEADER_SIZE as i32, SEEK_SET) != 0
            || b3d_fread(
                buf_bytes_mut(&mut buffer[..num as usize]),
                4,
                num as usize,
                in_fp,
            ) as i32
                != num
        {
            exit_error(b"Reading extended header data");
        }

        // Look for a legal pixel size and rotation angle
        if head.nreal >= 12 {
            *pix_size = 0.;
            temp = buffer[(head.nint + 11) as usize];

            // UCSF tomo puts out angstroms, but it could still be meters.  These are limits for
            // angstroms in flib/image/header.f90
            if temp as f64 > 0.05 && (temp as f64) < 100000. {
                *pix_size = temp;
            } else {
                temp = (temp as f64 * 1.0e10) as f32;
                if temp as f64 > 0.05 && (temp as f64) < 100000. {
                    *pix_size = temp;
                }
            }
            temp = buffer[(head.nint + 10) as usize];

            // set return to 1 if both are legal
            if *pix_size > 0. && temp as f64 >= -360. && temp as f64 <= 360. {
                ret_val = 1;
                if (temp as f64) < -180. {
                    temp = (temp as f64 + 360.) as f32;
                }
                if temp as f64 > 180. {
                    temp = (temp as f64 - 360.) as f32;
                }
                *axis_angle = temp;
            }
        }

        if axis_pix_only {
            return ret_val;
        }

        // Now analyze tilt angles in slot 1.  Loop on the subset of frames if any
        // (an ending frame past the file is clamped to it: the source reads past
        // the extended header buffer, BUGS.md)
        if start_frame > 0 {
            iz_start = start_frame - 1;
            iz_end = b3dmin!(end_frame, head.nz) - 1;
        } else {
            iz_start = 0;
            iz_end = head.nz - 1;
        }
        tilts.clear();
        self.m_set_starts.clear();

        // The set sizes of the previous file are not cleared in the source, so
        // a second file's sets were sized from the first's (BUGS.md).
        self.m_num_in_sets.clear();

        // Look for each place where tilt angle changes and save the angle and set start
        iz = iz_start;
        while iz <= iz_end {
            temp = buffer[(head.nint as i32 + iz * num_int_real) as usize];
            if (temp as f64) < -180. || temp as f64 > 180. {
                return ret_val;
            }
            if iz == iz_start || ((temp - last_tilt).abs() as f64) > 0.01 {
                if iz > iz_start {
                    // Keep track of size of sets and number of ones at the last size
                    // If there have been at least 5 in a row at a size and there is one at twice
                    // the size, it must be the repeated one at the starting angle
                    size = iz - *self.m_set_starts.last().unwrap();
                    if num_at_size > 5 && size == 2 * last_size && got_double == 0 {
                        self.m_set_starts.push(iz - last_size);
                        tilts.push(temp);
                        size = last_size;
                        got_double = 1;
                    }
                    if size == last_size {
                        num_at_size += 1;
                    } else {
                        num_at_size = 1;
                        last_size = size;
                    }
                }
                self.m_set_starts.push(iz);
                tilts.push(temp);
                last_tilt = temp;
            }
            iz += 1;
        }
        self.m_set_starts.push(iz);

        // Get the min and max set size
        for iz in 0..tilts.len() {
            set = self.m_set_starts[iz + 1] - self.m_set_starts[iz];
            self.m_num_in_sets.push(set);
            if iz == 0 {
                *min_set = set;
                *max_set = set;
            }
            *min_set = b3dmin!(*min_set, set);
            *max_set = b3dmax!(*max_set, set);
        }
        2 + ret_val
    }

    /// `AliFrame::rotateFlipGainReference` (`alignframes.cpp:3262`).
    ///
    /// Apply rotation and flip operation to gain reference: now it's simple
    pub fn rotate_flip_gain_reference(&mut self, reference: &mut [f32]) {
        let nx_in = self.m_nx_gain;
        let ny_in = self.m_ny_gain;
        let n = (self.m_nx_gain * self.m_ny_gain) as usize;
        let mut summed = vec![0f32; n];
        if rotate_flip_image(
            RotateFlipData::Float {
                array: &reference[..n],
                brray: &mut summed,
            },
            nx_in,
            ny_in,
            self.m_rotation_flip,
            0,
            0,
            0,
            &mut self.m_nx_gain,
            &mut self.m_ny_gain,
            0,
        ) != 0
        {
            exit_error_fmt!(
                "Inappropriate rotation/flip value %d entered",
                CArg::Int(self.m_rotation_flip as i64)
            );
        }
        let n = (self.m_nx_gain * self.m_ny_gain) as usize;
        reference[..n].copy_from_slice(&summed[..n]);
    }

    /// `AliFrame::openMdocFile` (`alignframes.cpp:3278`).
    ///
    /// Common operations when opening mdoc for either option
    pub fn open_mdoc_file(&mut self, filename: &[u8], num_sect: &mut i32, adoc_type: &mut i32) {
        let mut ind = 0;
        self.m_adoc_ind = adoc_open_image_metadata(filename, 0, &mut ind, num_sect, adoc_type);
        if self.m_adoc_ind == -1 {
            exit_error_fmt!("Opening or reading mdoc file %s", CArg::Bytes(filename));
        }
        if self.m_adoc_ind == -2 {
            exit_error_fmt!("Metadata file %s does not exist", CArg::Bytes(filename));
        }
        if self.m_adoc_ind == -3 {
            exit_error_fmt!(
                "Metadata file %s does not have image stack information",
                CArg::Bytes(filename)
            );
        }
    }

    /// `AliFrame::processDoseWeightingOptions` (`alignframes.cpp:3293`).
    ///
    /// Read some dose weighting options and get information from the dose weighting file
    pub fn process_dose_weighting_options(
        &mut self,
        frame_doses: Option<Vec<u8>>,
        adoc_type: &mut i32,
    ) {
        let mut dose_name: Option<Vec<u8>> = None;
        let mut iz: i32 = 0;
        let mut ierr: i32;
        let mut dose_temp = 0f32;
        let mut prior_temp = 0f32;
        let crit_dose_scale200_kv = 0.8f32;
        let mut voltage = 300i32;

        pip_get_float(b"InitialPriorDose", &mut self.m_initial_dose);
        pip_get_integer(b"Voltage", &mut voltage);
        pip_get_integer(b"BidirectionalNumViews", &mut self.m_num_bidir);
        pip_get_float(b"OptimalDoseScaling", &mut self.m_dose_scaling);
        iz = 0;
        pip_get_boolean(b"NormalizeDoseWeighting", &mut iz);
        if iz != 0 {
            self.m_reweight_ones.resize(9000, 1.);
            self.m_reweight_filt = true;
        }

        // Incorporate voltage info into scaling factor
        if voltage != 300 {
            if voltage != 200 {
                exit_error(b"Voltage must be either 200 or 300");
            }
            self.m_dose_scaling *= crit_dose_scale200_kv;
        }
        if let Some(frame_doses) = frame_doses {
            self.m_fixed_frame_doses = frame_doses;
        }

        // Get dose file of various kinds
        if self.m_dose_file_type > 0 {
            let mut temp = Vec::new();
            if pip_get_string(b"DoseWeightingFile", &mut temp) == 0 {
                dose_name = Some(temp);
            }
            if self.m_dose_file_type == 4 {
                if dose_name.is_none() && self.m_adoc_ind < 0 {
                    exit_error(
                        b"You cannot specify dose file type 4 without entering the name of an .mdoc file",
                    );
                }

                // Mdoc file, either existing one or specified one
                if dose_name.is_some() && self.m_adoc_ind >= 0 {
                    exit_error(
                        b"You cannot enter a dose weighting mdoc file name if you also enter the -mdoc option",
                    );
                }
                if self.m_adoc_ind >= 0 {
                    self.m_num_mdoc_sect = self.m_num_sect;
                } else {
                    let mut num_mdoc_sect = 0;
                    self.open_mdoc_file(
                        dose_name.as_deref().unwrap(),
                        &mut num_mdoc_sect,
                        adoc_type,
                    );
                    self.m_num_mdoc_sect = num_mdoc_sect;
                }

                ierr = self.get_doses_from_mdoc(*adoc_type);
                if ierr == 1 {
                    exit_error_fmt!(
                        "Problems occurred accessing dose data in the mdoc file: %s",
                        CArg::Str(&b3d_get_error())
                    );
                }
                if ierr == 2 {
                    exit_error_fmt!(
                        "The dose information in the mdoc file was not usable: %s",
                        CArg::Str(&b3d_get_error())
                    );
                }

                // If the PriorRecordDose entries are missing, add them to protect against
                // excludeviews being used, which would invalidate date-time analysis
                if self.m_adoc_ind >= 0
                    && adoc_get_float(ADOC_ZVALUE_NAME, 0, PRIOR_DOSE_KEY, &mut prior_temp) != 0
                {
                    for iz in 0..self.m_num_mdoc_sect {
                        if adoc_set_float(
                            ADOC_ZVALUE_NAME,
                            iz,
                            PRIOR_DOSE_KEY,
                            self.m_prior_from_mdoc[iz as usize],
                        ) != 0
                        {
                            exit_error(b"Setting the new accumulated dose into the mdoc structure");
                        }
                    }
                }
            } else {
                // Other text files, the name must be provided; open the file and read lines
                let Some(dose_name) = dose_name.as_deref() else {
                    exit_error(b"You must enter a dose weighting file also");
                };
                let Some(mut fp) = ImodFile::open(os_path(dose_name), "r") else {
                    exit_error_fmt!("Opening dose file %s", CArg::Bytes(dose_name));
                };
                loop {
                    ierr = fgetline(&mut fp, &mut self.m_in_line, MAX_LINE as i32);
                    if ierr == 0 {
                        continue;
                    }
                    if ierr == -2 {
                        break;
                    }
                    if ierr == -1 {
                        exit_error_fmt!("Reading dose file %s", CArg::Bytes(dose_name));
                    }
                    let line = c_str(&self.m_in_line).to_vec();

                    // A single dose value for type 1, a line for type 4, or prior and another value
                    if self.m_dose_file_type == 1 {
                        let mut end = 0;
                        self.m_total_dose_vec.push(strtod(&line, &mut end) as f32);
                        self.m_prior_dose_vec.push(0.);
                    } else if self.m_dose_file_type > 4 {
                        self.m_frame_dose_lines.push(line);
                    } else {
                        crate::imod::clip::clip::sscanf(
                            &String::from_utf8_lossy(&line),
                            "%f %f",
                            &mut [
                                crate::imod::clip::clip::ScanArg::Flt(&mut prior_temp),
                                crate::imod::clip::clip::ScanArg::Flt(&mut dose_temp),
                            ],
                        );
                        if self.m_dose_file_type == 3 {
                            dose_temp -= prior_temp;
                        }
                        self.m_total_dose_vec.push(dose_temp);
                        self.m_prior_dose_vec.push(prior_temp);
                    }
                    if ierr < 0 {
                        break;
                    }
                }
            }
        }
    }

    /// `AliFrame::getDosesFromMdoc` (`alignframes.cpp:3398`).
    ///
    /// Set up vectors to receive doses from mdoc file and call function to get them
    pub fn get_doses_from_mdoc(&mut self, adoc_type: i32) -> i32 {
        // Set up vectors to call for doses from mdoc
        let n = self.m_num_mdoc_sect.max(0) as usize;
        self.m_dose_from_mdoc.resize(n, 0.);
        self.m_prior_from_mdoc.resize(n, 0.);
        self.m_iz_piece.resize(n, 0);
        self.m_frame_dose_lines
            .resize(self.m_num_in_files.max(0) as usize, Vec::new());
        for iz in 0..n {
            self.m_iz_piece[iz] = iz as i32;
        }
        if self.m_zero_dose_thresh > 0. {
            set_zero_dose_thresh_and_accum(self.m_zero_dose_thresh, self.m_zero_dose_accum);
        }
        get_metadata_weighting_doses(
            self.m_adoc_ind,
            adoc_type,
            self.m_num_mdoc_sect,
            &self.m_iz_piece,
            self.m_num_bidir,
            &mut self.m_prior_from_mdoc,
            &mut self.m_dose_from_mdoc,
        )
    }

    /// `AliFrame::unifyDoseInformation` (`alignframes.cpp:3419`).
    ///
    /// Process dose information some more so all pathways end up in mTotalDoseVec and
    /// mPriorDoseVec
    pub fn unify_dose_information(
        &mut self,
        break_set_size: i32,
        combine_files: i32,
        adoc_type: i32,
    ) {
        let mut mdoc_file_tails: Vec<Vec<u8>> = Vec::new();
        let mut full_frame_paths: Vec<Option<String>>;
        let mut sstr: Vec<u8> = Vec::new();
        let mut ind: i32 = 0;
        let mut iz: i32 = 0;
        let mut ierr: i32;
        let mut tempname: Vec<u8> = Vec::new();
        let mut dose_temp = 0f32;
        let mut had_priors: bool;
        let mut got_prior = false;
        let mut priors_with_mdoc_prior: Vec<f32> = Vec::new();
        let mut mdoc_prior_index: Vec<i32> = Vec::new();
        let mut dt_prior_index: Vec<i32> = Vec::new();

        if self.m_total_dose > 0.
            || self.m_dose_file_type > 0
            || !self.m_fixed_frame_doses.is_empty()
        {
            self.m_frame_doses
                .resize(self.m_max_frame_doses.max(0) as usize, 0.);
            self.m_temp_val1
                .resize(2 * self.m_max_frame_doses.max(0) as usize, 0.);
        }
        if self.m_dose_file_type > 0 {
            ind = if self.m_dose_file_type < 4 {
                self.m_total_dose_vec.len() as i32
            } else {
                self.m_frame_dose_lines.len() as i32
            };
            if self.m_dose_file_type != 4 && ind < self.m_num_in_files {
                exit_error_fmt!(
                    "The dose file has fewer lines (%d) than frame sets to be aligned %d",
                    CArg::Int(ind as i64),
                    CArg::Int(self.m_num_in_files as i64)
                );
            }
            if (combine_files > 0 || break_set_size > 0) && self.m_dose_file_type == 4 {
                exit_error(
                    b"You cannot use an mdoc for a dose file when combining files or breaking frames into sets",
                );
            }
            if self.m_dose_file_type == 4 && !self.m_names_from_mdoc && !self.m_doing_frame_ts {
                // Need to match frame paths in mdoc to actual files being aligned
                let n = self.m_num_mdoc_sect.max(0) as usize;
                full_frame_paths = vec![None; n];
                if self.m_temp_val1.len() < n {
                    self.m_temp_val1.resize(n, 0.);
                }
                let mut val2: Vec<f32> = Vec::new();
                let mut val3: Vec<f32> = Vec::new();
                if get_metadata_by_key(
                    self.m_adoc_ind,
                    adoc_type,
                    self.m_num_mdoc_sect,
                    "SubFramePath",
                    0,
                    &mut self.m_temp_val1,
                    &mut val2,
                    &mut val3,
                    Some(&mut full_frame_paths),
                    &mut ind,
                    &mut iz,
                    self.m_num_mdoc_sect,
                    &self.m_iz_piece,
                ) != 0
                    || iz == 0
                {
                    exit_error(b"Getting all frame paths from mdoc file");
                }

                // Got some names: reduce them all to filename only
                mdoc_file_tails.resize(n, Vec::new());
                for ind in 0..n {
                    if let Some(path) = full_frame_paths[ind].take() {
                        self.extract_file_tail(path.as_bytes(), &mut sstr);
                        mdoc_file_tails[ind] = sstr.clone();
                    }
                }

                // Loop on the filenames
                for ifile in 0..self.m_num_in_files as usize {
                    let in_file = self.m_in_files[ifile].clone();
                    self.extract_file_tail(&in_file, &mut sstr);
                    ind = 0;
                    while (ind as usize) < n {
                        let i = ind as usize;
                        if mdoc_file_tails[i] == sstr {
                            self.m_total_dose_vec.push(self.m_dose_from_mdoc[i]);
                            self.m_prior_dose_vec.push(
                                if (self.m_adoc_ind >= 0 && self.m_dose_accumulates > 0)
                                    || self.m_dose_accumulates > 1
                                {
                                    self.m_prior_from_mdoc[i]
                                } else {
                                    0.
                                },
                            );
                            mdoc_file_tails[i] = Vec::new();
                            break;
                        }
                        ind += 1;
                    }
                    if ind >= self.m_num_mdoc_sect {
                        exit_error_fmt!(
                            "No section was found in the mdoc file with a filename matching input file %s",
                            CArg::Bytes(&in_file)
                        );
                    }
                    ierr = adoc_get_string(
                        ADOC_ZVALUE_NAME,
                        ind,
                        FRAME_DOSE_KEY.as_bytes(),
                        &mut tempname,
                    );
                    if ierr < 0 {
                        exit_error_fmt!(
                            "Trying to access FrameDosesAndNumber from section %d in mdoc file",
                            CArg::Int(ind as i64)
                        );
                    }
                    if ierr == 0 {
                        self.m_frame_dose_lines[ifile] = tempname.clone();
                    }
                }
            }

            // When there is a saved frame list, just copy the entries over
            if self.m_dose_file_type == 4 && self.m_doing_frame_ts {
                self.m_total_dose_vec = self.m_dose_from_mdoc.clone();
                self.m_prior_dose_vec = self.m_prior_from_mdoc.clone();
            }

            // Now want to check mFrameDoses for reasonableness and get a total dose for type 5
            if self.m_dose_file_type > 3 && !self.m_doing_frame_ts {
                for ifile in 0..self.m_num_in_files as usize {
                    let line = self.m_frame_dose_lines[ifile].clone();
                    let mut temp_arr = std::mem::take(&mut self.m_temp_val1);
                    self.expand_frame_doses_numbers(
                        &line,
                        &mut temp_arr,
                        self.m_max_frame_doses,
                        &mut ind,
                        &mut dose_temp,
                        None,
                    );
                    self.m_temp_val1 = temp_arr;
                    if self.m_dose_file_type > 4 {
                        self.m_total_dose_vec.push(dose_temp);
                        self.m_prior_dose_vec.push(0.);
                    }
                }
            }
        }

        // Single frame dose entry: check it, fill arrays, and pretend it is type 5
        if !self.m_fixed_frame_doses.is_empty() {
            let line = self.m_fixed_frame_doses.clone();
            let mut temp_arr = std::mem::take(&mut self.m_temp_val1);
            self.expand_frame_doses_numbers(
                &line,
                &mut temp_arr,
                self.m_max_frame_doses,
                &mut ind,
                &mut dose_temp,
                None,
            );
            self.m_temp_val1 = temp_arr;
            for _ in 0..self.m_num_in_files {
                self.m_frame_dose_lines
                    .push(self.m_fixed_frame_doses.clone());
                self.m_total_dose_vec.push(dose_temp);
                self.m_prior_dose_vec.push(0.);
            }
            self.m_dose_file_type = 5;
        }

        // Fixed dose, fill the arrays
        if self.m_total_dose > 0. {
            self.m_total_dose_vec
                .resize(self.m_num_in_files.max(0) as usize, self.m_total_dose);
            self.m_prior_dose_vec
                .resize(self.m_num_in_files.max(0) as usize, 0.);

            // If there is an mdoc file, try to insert this exposure dose and use it to compute
            // the prior doses.  First make sure every entry has date-time
            if self.m_adoc_ind >= 0 && self.m_dose_accumulates > 0 {
                iz = 0;
                while iz < self.m_num_sect {
                    if adoc_get_string(ADOC_ZVALUE_NAME, iz, b"DateTime", &mut tempname) != 0 {
                        break;
                    }
                    iz += 1;
                }
                if iz == self.m_num_sect {
                    // If they are all there, get the doses first in case there are priors, and save
                    self.m_num_mdoc_sect = self.m_num_sect;
                    ierr = self.get_doses_from_mdoc(adoc_type);
                    had_priors = ierr == 0;
                    if had_priors {
                        priors_with_mdoc_prior = self.m_prior_from_mdoc.clone();
                    }

                    // Wipe out the prior Record doses if any and insert/update the dose in each sect
                    for iz in 0..self.m_num_sect {
                        if adoc_delete_key_value(ADOC_ZVALUE_NAME, iz, PRIOR_DOSE_KEY).is_err() {
                            had_priors = false;
                        }
                        if adoc_set_float(ADOC_ZVALUE_NAME, iz, b"ExposureDose", self.m_total_dose)
                            != 0
                        {
                            exit_error(
                                b"Setting the entered fixed total dose into the mdoc structure",
                            );
                        }
                    }

                    // Get the doses back from mdoc, accessing the date-time info
                    ierr = self.get_doses_from_mdoc(adoc_type);

                    got_prior = ierr == 0;
                    if ierr == 1 {
                        exit_error(b"Problems occurred accessing data in the mdoc file");
                    }
                    if ierr != 0 {
                        printf!(
                            "WARNING: The date-time information in the mdoc file was not usable for computing accumulated doses\n"
                        );
                    } else {
                        self.m_total_dose_vec = self.m_dose_from_mdoc.clone();
                        self.m_prior_dose_vec = self.m_prior_from_mdoc.clone();

                        // Check consistency of ordering between original mdoc if it has priors,
                        // and the order based on date-time
                        if had_priors && !priors_with_mdoc_prior.is_empty() {
                            for iz in 0..self.m_num_sect {
                                mdoc_prior_index.push(iz);
                                dt_prior_index.push(iz);
                            }
                            rs_sort_indexed_floats(
                                &priors_with_mdoc_prior,
                                &mut mdoc_prior_index,
                                self.m_num_sect,
                            );
                            rs_sort_indexed_floats(
                                &self.m_prior_from_mdoc,
                                &mut dt_prior_index,
                                self.m_num_sect,
                            );
                            for iz in 0..self.m_num_sect as usize {
                                if mdoc_prior_index[iz] != dt_prior_index[iz] {
                                    printf!(
                                        "WARNING: Inconsistency between image ordering implied by pre-existing PriorRecordDose entries and new ordering from DateTime entries in mdoc\n"
                                    );
                                    break;
                                }
                            }
                        }

                        // (Re)insert the PriorRecordDose entries to protect against excludeviews
                        for iz in 0..self.m_num_sect {
                            if adoc_set_float(
                                ADOC_ZVALUE_NAME,
                                iz,
                                PRIOR_DOSE_KEY,
                                self.m_prior_from_mdoc[iz as usize],
                            ) != 0
                            {
                                exit_error(
                                    b"Setting the new accumulated dose into the mdoc structure",
                                );
                            }
                        }
                    }
                }
            }
        }

        // If assuming a tilt series, assign prior doses when none available
        if (self.m_dose_accumulates > 0
            && (self.m_dose_file_type == 1
                || self.m_dose_file_type > 4
                || (self.m_total_dose > 0. && !got_prior)))
            || (self.m_dose_accumulates == 1 && self.m_dose_file_type == 4 && self.m_adoc_ind < 0)
        {
            let n = self.m_total_dose_vec.len();
            self.m_prior_dose_vec
                .resize(b3dmax!(n, self.m_prior_dose_vec.len()), 0.);
            prior_doses_from_image_doses(
                &self.m_total_dose_vec,
                self.m_num_bidir,
                &mut self.m_prior_dose_vec[..n],
            );
        }
    }

    /// `AliFrame::expandFrameDosesNumbers` (`alignframes.cpp:3605`).
    ///
    /// Convert a text line for frame doses and number into an array of doses.
    /// `tempArr` is grown to the `2 * maxFrames` values the parse may store
    /// (natively a write within the vector's capacity but past its size, BUGS.md).
    pub fn expand_frame_doses_numbers(
        &self,
        line: &[u8],
        temp_arr: &mut Vec<f32>,
        max_frames: i32,
        total_frames: &mut i32,
        total_dose: &mut f32,
        mut frame_doses: Option<&mut Vec<f32>>,
    ) {
        let mut num_same: i32;
        let mut num_to_get = 0i32;
        *total_frames = 0;
        *total_dose = 0.;
        if line.is_empty() {
            return;
        }
        let need = (2 * max_frames).max(0) as usize;
        if temp_arr.len() < need {
            temp_arr.resize(need, 0.);
        }
        if pip_get_line_of_values(
            FRAME_DOSE_KEY.as_bytes(),
            line,
            PipValueArray::Float(&mut temp_arr[..]),
            PIP_FLOAT,
            &mut num_to_get,
            2 * max_frames,
        )
        .is_err()
        {
            exit_error_fmt!(
                "Processing an entry for frame doses and numbers: %s",
                CArg::Bytes(line)
            );
        }
        if num_to_get % 2 != 0 {
            exit_error_fmt!(
                "Odd number of numbers in entry for frame doses and numbers: %s",
                CArg::Bytes(line)
            );
        }
        let mut ind = 0usize;
        while (ind as i32) < num_to_get {
            num_same = b3dnint!(temp_arr[ind + 1]);
            if ((num_same as f32 - temp_arr[ind + 1]).abs() as f64) > 1.0e-3 {
                exit_error_fmt!(
                    "Non-integer value for count in entry for frame doses and numbers: %s",
                    CArg::Bytes(line)
                );
            }
            if *total_frames + num_same > max_frames {
                exit_error_fmt!(
                    "Frame numbers add up to more than maximum expected number of frames in: %s",
                    CArg::Bytes(line)
                );
            }
            *total_dose += num_same as f32 * temp_arr[ind];
            if let Some(doses) = frame_doses.as_deref_mut() {
                for _ in 0..num_same {
                    doses[*total_frames as usize] = temp_arr[ind];
                    *total_frames += 1;
                }
            } else {
                *total_frames += num_same;
            }
            ind += 2;
        }
    }

    /// `AliFrame::frameGroupLimits` (`alignframes.cpp:3640`).
    ///
    /// Compute limits for looping over frame groups.  Yes, class members would be easier
    #[allow(clippy::too_many_arguments)]
    pub fn frame_group_limits(
        &self,
        num_fetch: i32,
        nz_align: i32,
        group: i32,
        group_start: &mut i32,
        group_end: &mut i32,
        z_start: i32,
        z_dir: i32,
        iz_low: &mut i32,
        iz_high: &mut i32,
    ) {
        let mut iz = 0;
        balanced_group_limits(
            num_fetch,
            nz_align,
            nz_align - 1 - group,
            group_start,
            &mut iz,
        );
        *group_end = num_fetch - 1 - *group_start;
        *group_start = num_fetch - 1 - iz;
        *iz_low = z_start + z_dir * *group_start;
        *iz_high = z_start + z_dir * *group_end;
    }

    /// `AliFrame::analyzeForPartialFrames` (`alignframes.cpp:3656`).
    ///
    /// Load in first, last, and middle frame of a frame set and compare their means to
    /// see if first or last should be skipped
    pub fn analyze_for_partial_frames(
        &mut self,
        nz: &mut i32,
        start_combine: &mut i32,
        end_combine: &mut i32,
        ifile: i32,
        dropped: &mut [i32; 2],
    ) {
        let mut drop_ind = 0usize;
        let num_sample = 40000i32;
        let sample: f32;
        let mut means = [0f32; 3];
        let trim = b3dmin!(self.m_nx, self.m_ny) / 20;
        let nx_use = self.m_nx - 2 * trim;
        let ny_use = self.m_ny - 2 * trim;
        let typ = type_for_sample_mean(self.m_in_head.mode);
        dropped[0] = -1;
        dropped[1] = -1;

        // Skip if no partial thresholds or too few frames
        if (self.m_partial_thresh[0] <= 0. && self.m_partial_thresh[1] <= 0.) || *nz < 3 {
            return;
        }

        self.m_in_data_size = data_size_for_mode(self.m_in_head.mode).map_or(0, |(d, _)| d);

        // Allocation buffers and their line pointers the first time
        let line_bytes = (self.m_in_data_size * self.m_nx) as usize;
        if self.m_partial_scan_bufs[0].is_empty() {
            for ind in 0..3 {
                self.m_partial_scan_bufs[ind] = frame_storage(line_bytes * self.m_ny as usize);
            }
        }

        //Keep track of what frames these are
        sample = b3dmin!(num_sample, nx_use * ny_use) as f32 / (nx_use * ny_use) as f32;
        self.m_zin_partial_bufs[0] = *start_combine;
        self.m_zin_partial_bufs[1] = *start_combine + *nz / 2;
        self.m_zin_partial_bufs[2] = *end_combine;

        // Read and get the sample mean
        for ind in 0..3 {
            let mut buf = std::mem::take(&mut self.m_partial_scan_bufs[ind]);
            self.read_one_frame(buf_bytes_mut(&mut buf), self.m_zin_partial_bufs[ind], ifile);
            let bytes = buf_bytes(&buf);
            let lines: Vec<&[u8]> = (0..self.m_ny as usize)
                .map(|iy| &bytes[iy * line_bytes..(iy + 1) * line_bytes])
                .collect();
            if sample_mean_only(
                Some(&lines),
                typ,
                self.m_nx,
                self.m_ny,
                sample,
                trim,
                trim,
                nx_use,
                ny_use,
                Some(&mut means[ind]),
            ) != 0
            {
                exit_error(b"Error computing mean with sampling for partial frame analysis");
            }
            drop(lines);
            self.m_partial_scan_bufs[ind] = buf;
        }

        // Make decisions and adjust the frame set
        if self.m_partial_thresh[0] > 0. && means[0] < self.m_partial_thresh[0] * means[1] {
            if self.m_debug != 0 {
                printf!(
                    "Skipping frame %d  mean %.3f  ref  %.3f\n",
                    CArg::Int((*start_combine + 1) as i64),
                    CArg::Dbl(means[0] as f64),
                    CArg::Dbl(means[1] as f64)
                );
            }
            dropped[drop_ind] = *start_combine;
            drop_ind += 1;
            *start_combine += 1;
        }
        if self.m_partial_thresh[1] > 0.
            && means[2] < self.m_partial_thresh[1] * means[1]
            && *end_combine - *start_combine > 1
        {
            if self.m_debug != 0 {
                printf!(
                    "Skipping frame %d  mean %.3f  ref  %.3f\n",
                    CArg::Int((*end_combine + 1) as i64),
                    CArg::Dbl(means[2] as f64),
                    CArg::Dbl(means[1] as f64)
                );
            }
            dropped[drop_ind] = *end_combine;
            *end_combine -= 1;
        }
        *nz = *end_combine + 1 - *start_combine;
    }
}
