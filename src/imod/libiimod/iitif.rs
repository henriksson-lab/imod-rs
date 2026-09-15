//! Translation scaffold for `IMOD/libiimod/iitif.c`.
//!
//! The source owns TIFF-specific file opening, pixel conversion, EER decoding, and
//! serial/parallel TIFF writing.  Each function below deliberately corresponds to
//! one function in that C source; libtiff calls are filled in as its ABI is brought
//! into the crate.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_fread, b3d_rewind, c_format, c_format_bytes,
};
use crate::imod::libcfshr::b3dutil::{
    b3d_error, b3d_shift_bytes, group_limits_remainder_at_end, imod_getpid, make_all_big_tiff,
    num_omp_threads,
};
use crate::imod::libcfshr::parse_params::{strtod, strtol};
use crate::imod::libcfshr::zoomdown::{select_zoom_filter, zoom_raw_filt_value};
use crate::imod::libiimod::iilikemrc::{
    RAW_MODE_BYTE, RAW_MODE_FLOAT, RAW_MODE_SBYTE, RAW_MODE_SHORT, RAW_MODE_USHORT,
    ii_setup_raw_headers,
};
use crate::imod::libiimod::iimage::{
    IIFILE_TIFF, IIFORMAT_COLORMAP, IIFORMAT_COMPLEX, IIFORMAT_LUMINANCE, IIFORMAT_RGB,
    IISTATE_BUSY, IISTATE_READY, IITYPE_BYTE, IITYPE_FLOAT, IITYPE_INT, IITYPE_SHORT, IITYPE_UBYTE,
    IITYPE_UINT, IITYPE_USHORT, ImodImageFile, MRSA_BYTE, MRSA_FLOAT, MRSA_NOPROC, MRSA_USHORT,
    RawImageInfo, ii_best_tile_size, ii_close, ii_delete, ii_make_buffer_convert_if_float, ii_new,
    ii_reopen,
};
use crate::imod::libiimod::mrcfiles::{
    IIUNIT_4BIT_MODE, IIUNIT_HALF_XSIZE, MRC_LABEL_SIZE, MRC_MODE_BYTE, MRC_MODE_FLOAT,
    MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT, MRC_NLABELS, MrcHeader, PACKED_4BIT_MODE,
    PACKED_HALF_XSIZE, fix_title_padding, get_byte_map, get_short_map, mrc_head_new, mrc_set_scale,
    size_can_be_4_bit_k2_super_res,
};
use chrono::Local;
// `c_char` survives only below the libtiff line: the `TIFF*` entry points, the
// `TIFFFieldInfo` libtiff itself reads, and the `va_list` message handler.
use core::ffi::{c_char, c_void};
use core::sync::atomic::{AtomicI32, AtomicPtr, Ordering};
use std::io::Write;
use std::sync::Mutex;

unsafe extern "C" {
    // `warningHandler` formats libtiff's own `va_list` message.  On this ABI a
    // `va_list` argument decays to a pointer, so the source call is direct.
    fn vsnprintf(str: *mut c_char, size: usize, format: *const c_char, ap: *mut c_void) -> i32;
}

pub const IICOMPRESSION_NONE: i32 = 1;
pub const IICOMPRESSION_LZW: i32 = 5;
pub const IICOMPRESSION_JPEG: i32 = 7;
pub const IICOMPRESSION_ZIP: i32 = 8;
pub const IICOMPRESSION_EER_8BIT: i32 = 65_000;
pub const IICOMPRESSION_EER_7BIT: i32 = 65_001;
pub const IIFLAG_SKIP_EER_DIRS: i32 = 1;
pub const IIFLAG_ADD_TO_EER_SUM: i32 = 2;
pub const IIFLAG_START_END_EER_SUM: i32 = 4;
pub const IIFLAG_IGNORE_BAD_EER_END: i32 = 8;
pub const IIFLAG_ANTIALIAS_EER: i32 = 16;
pub const IIFLAG_EER_USE_LANCZOS: i32 = 32;
pub const EER_AA_SCALING_BIT_SHIFT: i32 = 6;
pub const EER_AA_SCALING_MASK: i32 = 7;
pub const MAX_TIFF_THREADS: usize = 16;
pub const IIERR_BAD_CALL: i32 = -1;
pub const IIERR_NOT_FORMAT: i32 = 1;
pub const IIERR_IO_ERROR: i32 = 2;
pub const IIERR_MEMORY_ERR: i32 = 3;
pub const IIERR_NO_SUPPORT: i32 = 4;
pub const IIERR_NOT_PRESENT: i32 = 6;
/// `iimage.h:116`
pub const RAW_MODE_RGB: i32 = 6;
/// `iimage.h:92`
pub const IIFLAG_BYTES_SWAPPED: u32 = 1;
/// `iimage.h:93`
pub const IIFLAG_TVIPS_DATA: u32 = 2;

/// Opaque libtiff handle.  Its representation is private to libtiff.
#[repr(C)]
pub struct Tiff {
    _private: [u8; 0],
}

#[repr(C)]
struct TiffFieldInfo {
    field_tag: u32,
    field_readcount: i16,
    field_writecount: i16,
    field_type: i32,
    field_bit: u16,
    field_oktochange: u8,
    field_passcount: u8,
    field_name: *mut c_char,
}

// `TIFFMergeFieldInfo` receives this structure through its `const` ABI and
// copies it into libtiff's directory state.  The sole pointer points at static
// C string data and neither Rust nor libtiff mutates these entries.
unsafe impl Sync for TiffFieldInfo {}

// This is deliberately the libtiff C ABI used by iitif.c.  It is not a Rust TIFF
// abstraction: IMOD relies on variadic tag access and on TIFF's codec behaviour.
#[link(name = "tiff")]
unsafe extern "C" {
    fn TIFFOpen(name: *const c_char, mode: *const c_char) -> *mut Tiff;
    fn TIFFClientOpen(
        name: *const c_char,
        mode: *const c_char,
        client_data: *mut c_void,
        read_proc: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, isize) -> isize>,
        write_proc: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, isize) -> isize>,
        seek_proc: Option<unsafe extern "C" fn(*mut c_void, u64, i32) -> u64>,
        close_proc: Option<unsafe extern "C" fn(*mut c_void) -> i32>,
        size_proc: Option<unsafe extern "C" fn(*mut c_void) -> u64>,
        map_proc: Option<unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut u64) -> i32>,
        unmap_proc: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, u64)>,
    ) -> *mut Tiff;
    fn TIFFClose(tif: *mut Tiff);
    fn TIFFSetDirectory(tif: *mut Tiff, directory: u16) -> i32;
    fn TIFFReadDirectory(tif: *mut Tiff) -> i32;
    fn TIFFSetField(tif: *mut Tiff, tag: u32, ...) -> i32;
    fn TIFFWriteDirectory(tif: *mut Tiff) -> i32;
    fn TIFFWriteEncodedStrip(tif: *mut Tiff, strip: u32, data: *mut c_void, size: isize) -> isize;
    fn TIFFWriteEncodedTile(tif: *mut Tiff, tile: u32, data: *mut c_void, size: isize) -> isize;
    fn TIFFReadEncodedStrip(tif: *mut Tiff, strip: u32, data: *mut c_void, size: isize) -> isize;
    fn TIFFStripSize(tif: *mut Tiff) -> isize;
    fn TIFFNumberOfStrips(tif: *mut Tiff) -> u32;
    fn TIFFRawStripSize(tif: *mut Tiff, strip: u32) -> isize;
    fn TIFFReadRawStrip(tif: *mut Tiff, strip: u32, data: *mut c_void, size: isize) -> isize;
    fn TIFFWriteRawStrip(tif: *mut Tiff, strip: u32, data: *mut c_void, size: isize) -> isize;
    fn TIFFMergeFieldInfo(tif: *mut Tiff, info: *const TiffFieldInfo, count: u32) -> i32;
    fn TIFFSetTagExtender(
        extender: Option<unsafe extern "C" fn(*mut Tiff)>,
    ) -> Option<unsafe extern "C" fn(*mut Tiff)>;
    fn TIFFReadEncodedTile(tif: *mut Tiff, tile: u32, data: *mut c_void, size: isize) -> isize;
    fn TIFFTileSize(tif: *mut Tiff) -> isize;
    fn TIFFGetField(tif: *mut Tiff, tag: u32, ...) -> i32;
    fn TIFFSetErrorHandler(handler: *mut c_void) -> *mut c_void;
    fn TIFFSetWarningHandler(handler: *mut c_void) -> *mut c_void;
    fn TIFFGetVersion() -> *const c_char;
    fn TIFFIsByteSwapped(tif: *mut Tiff) -> i32;
}

const TIFFTAG_IMAGE_DESCRIPTION: u32 = 270;
const TIFFTAG_DATETIME: u32 = 306;
const TIFFTAG_IMAGEWIDTH: u32 = 256;
const TIFFTAG_IMAGELENGTH: u32 = 257;
const TIFFTAG_BITSPERSAMPLE: u32 = 258;
const TIFFTAG_COMPRESSION: u32 = 259;
const TIFFTAG_PHOTOMETRIC: u32 = 262;
const TIFFTAG_SAMPLESPERPIXEL: u32 = 277;
const TIFFTAG_ROWSPERSTRIP: u32 = 278;
const TIFFTAG_PLANARCONFIG: u32 = 284;
const TIFFTAG_XRESOLUTION: u32 = 282;
const TIFFTAG_YRESOLUTION: u32 = 283;
const TIFFTAG_RESOLUTIONUNIT: u32 = 296;
const TIFFTAG_SAMPLEFORMAT: u32 = 339;
const TIFFTAG_SMINSAMPLEVALUE: u32 = 340;
const TIFFTAG_SMAXSAMPLEVALUE: u32 = 341;
const TIFFTAG_FILLORDER: u32 = 266;
const FILLORDER_MSB2LSB: i32 = 1;
const FILLORDER_LSB2MSB: i32 = 2;
const PHOTOMETRIC_PALETTE: i32 = 3;
const TIFFTAG_COLORMAP: u32 = 320;
const TIFFTAG_STRIPOFFSETS: u32 = 273;
/// `mrcfiles.h:70`
const MRC_RAMP_LIN: i32 = 1;
const TIFFTAG_DM_ORIGIN_0: u32 = 65006;
const TIFFTAG_DM_SCALE_0: u32 = 65009;
const TIFFTAG_DM_UINFO_UNIT_0: u32 = 65012;
const TIFFTAG_DM_UINFO_POWER_0: u32 = 65015;
const TIFFTAG_JPEGQUALITY: u32 = 65537;
const TIFFTAG_ZIPQUALITY: u32 = 65557;
const TIFF_SLONG: i32 = 9;
const TIFF_DOUBLE: i32 = 12;
const TIFF_ASCII: i32 = 2;
const FIELD_CUSTOM: u16 = 65;
const TIFFTAG_TILEWIDTH: u32 = 322;
const TIFFTAG_TILELENGTH: u32 = 323;
const PLANARCONFIG_CONTIG: i32 = 1;
const PLANARCONFIG_SEPARATE: i32 = 2;
const PHOTOMETRIC_MINISBLACK: i32 = 1;
const PHOTOMETRIC_RGB: i32 = 2;
const SAMPLEFORMAT_UINT: i32 = 1;
const SAMPLEFORMAT_INT: i32 = 2;
const SAMPLEFORMAT_IEEEFP: i32 = 3;
const RESUNIT_NONE: i32 = 1;
const RESUNIT_INCH: i32 = 2;
const RESUNIT_CENTIMETER: i32 = 3;

static S_USE_MAPPING: AtomicI32 = AtomicI32::new(2);
static S_WARNINGS_SUPPRESSED: AtomicI32 = AtomicI32::new(0);
static S_OLD_HANDLER: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
static S_OLD_ERR_HANDLER: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
static S_MAX_EER_SUPER_RESOLUTION: AtomicI32 = AtomicI32::new(2);
static S_MIN_EER_SUPER_RESOLUTION: AtomicI32 = AtomicI32::new(-3);
static S_READ_EER_AS_SUPER_RES: AtomicI32 = AtomicI32::new(-1);
static S_EER_FLAGS: AtomicI32 = AtomicI32::new(0);
static S_AUTOGROUP_EER: AtomicI32 = AtomicI32::new(-999_999);
static S_IGNORE_BAD_EER_END: AtomicI32 = AtomicI32::new(0);
static S_ANTIALIAS_EER: AtomicI32 = AtomicI32::new(0);
static S_EER_KERNEL_SCALE: AtomicI32 = AtomicI32::new(0);
static S_GAIN_REFERENCE: AtomicPtr<f32> = AtomicPtr::new(core::ptr::null_mut());
static S_TAG_TO_PRINT: AtomicI32 = AtomicI32::new(0);
/// `ignoreFromVar`, the function-static of `tiffSetEERreadProperties`.
static S_IGNORE_FROM_VAR: AtomicI32 = AtomicI32::new(-999);
static S_FILE_BUF_SIZE: AtomicI32 = AtomicI32::new(0);
static S_SETTING_UP_PARALLEL: AtomicI32 = AtomicI32::new(0);
/// Crate-owned client-I/O buffers and positions for parallel TIFF writers.
/// libtiff receives only a file-slot cursor through its callback ABI.
struct ParallelTiffBuffers {
    file_buf: [Vec<u8>; MAX_TIFF_THREADS],
    cur_buf_ind: [u64; MAX_TIFF_THREADS],
    max_buf_ind: [u64; MAX_TIFF_THREADS],
}

/// Per-write strip and tile layout.  This is crate-owned state shared by the
/// setup, strip, finish, and parallel-write paths; it is never part of the
/// libtiff callback ABI.
#[derive(Clone, Copy, Default)]
struct StripTileState {
    rows_per_strip: i32,
    line_bytes: i32,
    strip_bytes: i32,
    x_tile_size: i32,
    lines_done: i32,
    num_strips: i32,
    already_inverted: i32,
    pix_size: i32,
    num_x_tiles: i32,
}

static S_PARALLEL_BUFFERS: Mutex<ParallelTiffBuffers> = Mutex::new(ParallelTiffBuffers {
    file_buf: [const { Vec::new() }; MAX_TIFF_THREADS],
    cur_buf_ind: [0; MAX_TIFF_THREADS],
    max_buf_ind: [0; MAX_TIFF_THREADS],
});
static S_STRIP_TILE_STATE: Mutex<StripTileState> = Mutex::new(StripTileState {
    rows_per_strip: 0,
    line_bytes: 0,
    strip_bytes: 0,
    x_tile_size: 0,
    lines_done: 0,
    num_strips: 0,
    already_inverted: 0,
    pix_size: 0,
    num_x_tiles: 0,
});
static S_TMP_BUF: Mutex<Vec<u8>> = Mutex::new(Vec::new());
/// C `static char *sDescription` (`iitif.c:2448`): the `ImageDescription`
/// waiting to be written into the next directory.  It is held as bytes and
/// keeps the terminating NUL `strdup` copies, because the only thing done with
/// it is to hand its pointer to `TIFFSetField`.
static S_DESCRIPTION: Mutex<Option<Vec<u8>>> = Mutex::new(None);
#[derive(Default)]
struct EerFilters {
    all: Vec<i32>,
    x_start: Vec<i32>,
    y_start: Vec<i32>,
}
static S_EER_FILTERS: Mutex<EerFilters> = Mutex::new(EerFilters {
    all: Vec::new(),
    x_start: Vec::new(),
    y_start: Vec::new(),
});
/// The tag extender installed before ours.  It crosses the libtiff callback
/// boundary, but is otherwise crate-owned synchronized state.
static S_PARENT_EXTENDER: Mutex<Option<unsafe extern "C" fn(*mut Tiff)>> = Mutex::new(None);
static S_AUGMENTED_TAGS: AtomicI32 = AtomicI32::new(0);
/// C `xtiffFieldInfo` (`iitif.c:3170`).  Below the libtiff line: libtiff reads
/// these `TIFFFieldInfo` entries itself, so `field_name` is a C string.
static S_XTIFF_FIELD_INFO: [TiffFieldInfo; 8] = [
    TiffFieldInfo {
        field_tag: TIFFTAG_DM_UINFO_POWER_0,
        field_readcount: 1,
        field_writecount: 1,
        field_type: TIFF_SLONG,
        field_bit: FIELD_CUSTOM,
        field_oktochange: 0,
        field_passcount: 0,
        field_name: c"UnitInfoPower0".as_ptr().cast_mut(),
    },
    TiffFieldInfo {
        field_tag: TIFFTAG_DM_UINFO_POWER_0 + 1,
        field_readcount: 1,
        field_writecount: 1,
        field_type: TIFF_SLONG,
        field_bit: FIELD_CUSTOM,
        field_oktochange: 0,
        field_passcount: 0,
        field_name: c"UnitInfoPower1".as_ptr().cast_mut(),
    },
    TiffFieldInfo {
        field_tag: TIFFTAG_DM_ORIGIN_0,
        field_readcount: 1,
        field_writecount: 1,
        field_type: TIFF_DOUBLE,
        field_bit: FIELD_CUSTOM,
        field_oktochange: 0,
        field_passcount: 0,
        field_name: c"Origin0".as_ptr().cast_mut(),
    },
    TiffFieldInfo {
        field_tag: TIFFTAG_DM_ORIGIN_0 + 1,
        field_readcount: 1,
        field_writecount: 1,
        field_type: TIFF_DOUBLE,
        field_bit: FIELD_CUSTOM,
        field_oktochange: 0,
        field_passcount: 0,
        field_name: c"Origin1".as_ptr().cast_mut(),
    },
    TiffFieldInfo {
        field_tag: TIFFTAG_DM_SCALE_0,
        field_readcount: 1,
        field_writecount: 1,
        field_type: TIFF_DOUBLE,
        field_bit: FIELD_CUSTOM,
        field_oktochange: 0,
        field_passcount: 0,
        field_name: c"Scale0".as_ptr().cast_mut(),
    },
    TiffFieldInfo {
        field_tag: TIFFTAG_DM_SCALE_0 + 1,
        field_readcount: 1,
        field_writecount: 1,
        field_type: TIFF_DOUBLE,
        field_bit: FIELD_CUSTOM,
        field_oktochange: 0,
        field_passcount: 0,
        field_name: c"Scale1".as_ptr().cast_mut(),
    },
    TiffFieldInfo {
        field_tag: TIFFTAG_DM_UINFO_UNIT_0,
        field_readcount: 1,
        field_writecount: 1,
        field_type: TIFF_ASCII,
        field_bit: FIELD_CUSTOM,
        field_oktochange: 0,
        field_passcount: 0,
        field_name: c"Unit0".as_ptr().cast_mut(),
    },
    TiffFieldInfo {
        field_tag: TIFFTAG_DM_UINFO_UNIT_0 + 1,
        field_readcount: 1,
        field_writecount: 1,
        field_type: TIFF_ASCII,
        field_bit: FIELD_CUSTOM,
        field_oktochange: 0,
        field_passcount: 0,
        field_name: c"Unit1".as_ptr().cast_mut(),
    },
];

/// C `iiTIFFCheck` (`iitif.c:111`).
pub unsafe fn ii_tiff_check(in_file: *mut ImodImageFile) -> i32 {
    let tif: *mut Tiff;
    let mut buf: u16 = 0;
    let mut dirnum = 0i32;
    let mut file_dir_num = 0i32;
    let mut bits: u16 = 0;
    let mut samples: u16 = 0;
    let mut photometric: u16 = 0;
    let mut sampleformat: u16 = 0;
    let mut planar_config: u16 = 0;
    let mut compression: u16 = 0;
    let mut bits_im: u16 = 0;
    let mut samples_im: u16 = 0;
    let mut photo_im: u16 = 0;
    let mut format_im: u16 = 0;
    let mut planar_im: u16 = 0;
    let mut res_unit: u16;
    let mut rows_per_strip: u32 = 0;
    let mut pr_count: u32 = 0;
    let mut offsets: *mut u32 = core::ptr::null_mut();
    let mut nxim: i32 = 0;
    let mut nyim: i32 = 0;
    let mut format_def = 0i32;
    let mut tile_width: i32 = 0;
    let mut tile_length: i32 = 0;
    let mut got_min = 0i32;
    let mut got_max = 0i32;
    let mut defined: i32;
    let mut i: i32 = 0;
    let mut j: i32;
    let mut has_pixel_im: i32;
    let mut has_pixel = 0i32;
    let mut mismatch = 0i32;
    let mut err = 0i32;
    let mut compression_im: i32;
    let mut x_resol: f32 = 0.;
    let mut y_resol: f32 = 0.;
    let mut x_pixel_im: f32 = 0.;
    let mut y_pixel_im: f32 = 0.;
    let mut x_pixel: f32 = 0.;
    let mut y_pixel: f32 = 0.;
    let mut res_scale: f32;
    let mut last_min: f64 = 0.;
    let mut last_max: f64 = 0.;
    let mut redp: *mut u16 = core::ptr::null_mut();
    let mut greenp: *mut u16 = core::ptr::null_mut();
    let mut bluep: *mut u16 = core::ptr::null_mut();
    let tvips_tag: u16 = 37708;
    let mut pixel_limit: f32 = 3.;
    // `description` and `prStrng` are the `char *` libtiff fills in; the bytes
    // behind them are read as bytes the moment the call returns.
    let mut description: *mut u8 = core::ptr::null_mut();
    let mut ptr: usize;
    let mut blank: Option<usize>;
    let mut colon: Option<usize>;
    let mut pr_strng: *mut u8 = core::ptr::null_mut();
    let mut end: usize;
    let mut image_j = 0i32;
    let mut im_jslices = 0i32;
    let mut im_jimages = 0i32;
    let mut im_jmin: f32 = 9.0e37;
    let mut im_jmax: f32 = -9.0e37;
    let mut im_jpixel: f32 = -1.;
    let format_defined: i32;
    let mut has_date_time = 0i32;
    let mut from_semccd = 0i32;
    let mut byte_max: i32;
    let mut eer_file: i32;
    let skip_eer = if S_EER_FLAGS.load(Ordering::SeqCst) & IIFLAG_SKIP_EER_DIRS != 0 {
        1
    } else {
        0
    };
    let mut num_eer_images = 0i32;
    let mut num_non_eer_images = 0i32;
    let mut skip_file: i32;
    let mut max_electrons: i32 = 0;
    let mut num_in_autogroup: i32 = 1;
    let mut red_fac: i32;
    let max_fac: i32;
    let mut frame_mean: f32 = 0.;
    let mut info = RawImageInfo {
        type_: 0,
        nx: 0,
        ny: 0,
        nz: 0,
        swap_bytes: 0,
        header_size: 0,
        amin: 0.,
        amax: 0.,
        scan_min_max: 0,
        all_match: 0,
        section_skip: 0,
        y_inverted: 0,
        pixel: 0.,
        z_pixel: 0.,
    };

    if S_AUTOGROUP_EER.load(Ordering::SeqCst) == -999_999 {
        S_READ_EER_AS_SUPER_RES.store(
            S_MAX_EER_SUPER_RESOLUTION.load(Ordering::SeqCst),
            Ordering::SeqCst,
        );
        S_AUTOGROUP_EER.store(1, Ordering::SeqCst);
        if let Some(resvar) = std::env::var_os("IMOD_READ_EER_SUPER_RES") {
            end = 0;
            S_READ_EER_AS_SUPER_RES.store(
                strtol(resvar.as_encoded_bytes(), &mut end, 10) as i32,
                Ordering::SeqCst,
            );
            S_READ_EER_AS_SUPER_RES.store(
                S_READ_EER_AS_SUPER_RES.load(Ordering::SeqCst).clamp(
                    S_MIN_EER_SUPER_RESOLUTION.load(Ordering::SeqCst),
                    S_MAX_EER_SUPER_RESOLUTION.load(Ordering::SeqCst),
                ),
                Ordering::SeqCst,
            );
            if S_READ_EER_AS_SUPER_RES.load(Ordering::SeqCst) < 0 {
                S_ANTIALIAS_EER.store(4, Ordering::SeqCst);
            }
        }
        if let Some(resvar) = std::env::var_os("IMOD_READ_EER_Z_SUMMING") {
            end = 0;
            S_AUTOGROUP_EER.store(
                strtol(resvar.as_encoded_bytes(), &mut end, 10) as i32,
                Ordering::SeqCst,
            );
        }
        if let Some(resvar) = std::env::var_os("IMOD_READ_EER_ANTIALIASED") {
            end = 0;
            red_fac = strtol(resvar.as_encoded_bytes(), &mut end, 10) as i32;
            if red_fac != 0 {
                red_fac = red_fac.clamp(1, 2);
                S_ANTIALIAS_EER.store(5 - red_fac, Ordering::SeqCst);
            }
        }
    }

    if in_file.is_null() {
        return IIERR_BAD_CALL;
    }
    let Some(mut fp) = (*in_file).fp.clone() else {
        return IIERR_BAD_CALL;
    };

    if S_READ_EER_AS_SUPER_RES.load(Ordering::SeqCst) < 0
        && S_ANTIALIAS_EER.load(Ordering::SeqCst) == 0
    {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiTIFFCheck - A negative super-resolution factor can be used only with antialiased EER reading\n"
            ),
        );
        return IIERR_BAD_CALL;
    }

    b3d_rewind(&mut fp);
    let mut stamp = [0u8; 2];
    if b3d_fread(&mut stamp, core::mem::size_of::<u16>(), 1, &mut fp) < 1 {
        err = IIERR_IO_ERROR;
    }
    buf = u16::from_ne_bytes(stamp);
    if err == 0 && buf != 0x4949 && buf != 0x4d4d {
        err = IIERR_NOT_FORMAT;
    }
    if err == 0 && b3d_fread(&mut stamp, core::mem::size_of::<u16>(), 1, &mut fp) < 1 {
        err = IIERR_IO_ERROR;
    }
    buf = u16::from_ne_bytes(stamp);
    if err == 0 && buf != 0x002a && buf != 0x2a00 && buf != 0x002b && buf != 0x2b00 {
        err = IIERR_NOT_FORMAT;
    }
    if err != 0 {
        if err == IIERR_IO_ERROR {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: iiTIFFCheck - Reading file {}\n",
                    (*in_file).filename.as_deref().unwrap_or("")
                ),
            );
        }
        return err;
    }

    /* Close file now, but reopen it if there is a TIFF failure */
    drop(fp);
    (*in_file).fp = None;
    tif = open_without_b_mode(in_file);
    if tif.is_null() {
        (*in_file).fp = ImodFile::open(
            (*in_file).filename.as_deref().unwrap_or(""),
            &(*in_file).fmode,
        );
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiTIFFCheck - Calling TIFFOpen on file {}\n",
                (*in_file).filename.as_deref().unwrap_or("")
            ),
        );
        return IIERR_IO_ERROR;
    }

    (*in_file).backend_handle = tif.cast();
    (*in_file).nx = 0;
    (*in_file).ny = 0;
    (*in_file).multiple_sizes = 0;
    (*in_file).planes_per_image = 1;
    (*in_file).contig_samples = 1;
    (*in_file).directory_nums = Some(Vec::with_capacity(20));
    if let Some(resvar) = std::env::var_os("TIFF_RES_PIXEL_LIMIT") {
        end = 0;
        pixel_limit = strtod(resvar.as_encoded_bytes(), &mut end) as f32;
    }

    // If no tag to print set by program, look for environment variable
    if S_TAG_TO_PRINT.load(Ordering::SeqCst) == 0 {
        if let Some(resvar) = std::env::var_os("TIFF_STRING_TAG_TO_PRINT") {
            end = 0;
            S_TAG_TO_PRINT.store(
                strtol(resvar.as_encoded_bytes(), &mut end, 10) as i32,
                Ordering::SeqCst,
            );
        }
    }

    // Try to get tag to print
    if S_TAG_TO_PRINT.load(Ordering::SeqCst) != 0 {
        if TIFFGetField(
            tif,
            S_TAG_TO_PRINT.load(Ordering::SeqCst) as u32,
            &raw mut pr_count,
            &raw mut pr_strng,
        ) > 0
            && pr_count != 0
        {
            // `iitif.c:227-232` copies `prCount` bytes with `strncpy` and
            // terminates them, so what `%s` prints is the tag's bytes up to
            // `prCount` or to an earlier NUL, whichever comes first.
            let mut pr_len = 0usize;
            while pr_len < pr_count as usize && *pr_strng.add(pr_len) != 0 {
                pr_len += 1;
            }
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "Tag %d: %s\n",
                &[
                    CArg::Int(S_TAG_TO_PRINT.load(Ordering::SeqCst) as i64),
                    CArg::Bytes(core::slice::from_raw_parts(pr_strng, pr_len)),
                ],
            ));
        }
    }

    /* Read each directory of the file, get properties and count usable images */
    loop {
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &raw mut nxim);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &raw mut nyim);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &raw mut bits_im);
        TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &raw mut photo_im);

        /* DNM 11/18/01: field need not be defined, set a default */
        defined = TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &raw mut samples_im);
        if defined == 0 {
            samples_im = 1;
        }

        TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &raw mut planar_im);
        // The source assigns PLANARCONFIG_CONTIG to `photoIm`, not `planarIm`,
        // when SAMPLESPERPIXEL was absent (`iitif.c:248`).  Preserved as written.
        if defined == 0 {
            photo_im = PLANARCONFIG_CONTIG as u16;
        }

        defined = TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &raw mut format_im);

        /* Handle pixel sizes */
        has_pixel_im = TIFFGetField(tif, TIFFTAG_XRESOLUTION, &raw mut x_resol);
        res_unit = 0;
        TIFFGetField(tif, TIFFTAG_RESOLUTIONUNIT, &raw mut res_unit);
        if res_unit > 1 && has_pixel_im != 0 {
            if TIFFGetField(tif, TIFFTAG_YRESOLUTION, &raw mut y_resol) == 0 {
                y_resol = x_resol;
            }
            res_scale = 1.0e8 * if res_unit == 2 { 2.54 } else { 1. };
            x_pixel_im = res_scale / x_resol;
            y_pixel_im = res_scale / y_resol;
        } else {
            has_pixel_im = 0;
        }

        if dirnum == 0 && TIFFGetField(tif, TIFFTAG_DATETIME, &raw mut description) != 0 {
            has_date_time = 1;
        }

        /* For the first directory, get the description and check if it ImageJ
        If so look for various values that are useful.
        Also check for SerialEMCCD title for packed 4 bit data */
        if dirnum == 0 && TIFFGetField(tif, TIFFTAG_IMAGE_DESCRIPTION, &raw mut description) != 0 {
            // The tag's value is libtiff's own NUL-terminated buffer; length it
            // once here and do the rest of `iitif.c:274-308` on the bytes.
            let mut desc_len = 0usize;
            while *description.add(desc_len) != 0 {
                desc_len += 1;
            }
            let text = core::slice::from_raw_parts(description, desc_len);
            if text
                .windows(b"SerialEMCCD".len())
                .any(|w| w == b"SerialEMCCD")
                && text
                    .windows(b"4 bits packed".len())
                    .any(|w| w == b"4 bits packed")
            {
                from_semccd = 1;

                /* But if you find it, check for multiple titles from other programs, i.e.,
                a line that has a colon before a blank */
                ptr = 0;
                while let Some(newline) = text[ptr..].iter().position(|byte| *byte == b'\n') {
                    ptr = ptr + newline + 1;
                    blank = text[ptr..].iter().position(|byte| *byte == b' ');
                    colon = text[ptr..].iter().position(|byte| *byte == b':');
                    if let (Some(colon), Some(blank)) = (colon, blank) {
                        if blank > colon {
                            from_semccd = 0;
                            break;
                        }
                    }
                }
            }
            if text.starts_with(b"ImageJ=") {
                image_j = 1;
                if let Some(found) = text.windows(b"slices=".len()).position(|w| w == b"slices=") {
                    end = 0;
                    im_jslices = strtol(&text[found + 7..], &mut end, 10) as i32;
                }
                if let Some(found) = text.windows(b"images=".len()).position(|w| w == b"images=") {
                    end = 0;
                    im_jimages = strtol(&text[found + 7..], &mut end, 10) as i32;
                }
                if let Some(found) = text.windows(b"min=".len()).position(|w| w == b"min=") {
                    end = 0;
                    im_jmin = strtol(&text[found + 4..], &mut end, 10) as i32 as f32;
                }
                if let Some(found) = text.windows(b"max=".len()).position(|w| w == b"max=") {
                    end = 0;
                    im_jmax = strtol(&text[found + 4..], &mut end, 10) as i32 as f32;
                }
                if let Some(found) = text
                    .windows(b"spacing=".len())
                    .position(|w| w == b"spacing=")
                {
                    if text
                        .windows(b"unit=micron".len())
                        .any(|w| w == b"unit=micron")
                    {
                        end = 0;
                        im_jpixel = (1.0e4 * strtod(&text[found + 8..], &mut end)) as f32;
                    }
                }
            }
        }

        compression_im = IICOMPRESSION_NONE;
        TIFFGetField(tif, TIFFTAG_COMPRESSION, &raw mut compression_im);
        eer_file = if compression_im == IICOMPRESSION_EER_7BIT
            || compression_im == IICOMPRESSION_EER_8BIT
        {
            1
        } else {
            0
        };
        if eer_file != 0 {
            num_eer_images += 1;
        } else {
            num_non_eer_images += 1;
        }

        if eer_file != 0 && skip_eer == 0 && (num_eer_images == 2 || num_eer_images == 20) {
            count_eer_bytes_and_electrons(
                tif,
                if compression_im == IICOMPRESSION_EER_7BIT {
                    1
                } else {
                    0
                },
                &raw mut i,
                &raw mut max_electrons,
            );
            frame_mean = max_electrons as f32 / (nxim * nyim) as f32;
        }

        /* If this is a bigger image, it is a new standard, so set all the
        properties and reset to one directory */
        skip_file = if (skip_eer != 0 && eer_file != 0)
            || (skip_eer == 0 && eer_file == 0 && num_eer_images > 0)
        {
            1
        } else {
            0
        };

        if (nxim as f32 * nyim as f32 > (*in_file).nx as f32 * (*in_file).ny as f32
            || (nxim == (*in_file).nx
                && nyim == (*in_file).ny
                && eer_file != 0
                && num_eer_images == 1))
            && skip_file == 0
        {
            (*in_file).nx = nxim;
            (*in_file).ny = nyim;

            /* Record the strip and tile size for 3dmod caching */
            (*in_file).tile_size_x = 0;
            (*in_file).tile_size_y = 0;
            if eer_file == 0 {
                if TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &raw mut rows_per_strip) != 0 {
                    (*in_file).tile_size_y = rows_per_strip as i32;
                } else if TIFFGetField(tif, TIFFTAG_TILEWIDTH, &raw mut tile_width) != 0
                    && TIFFGetField(tif, TIFFTAG_TILELENGTH, &raw mut tile_length) != 0
                {
                    (*in_file).tile_size_x = tile_width;
                    (*in_file).tile_size_y = tile_length;
                }
            }
            bits = bits_im;
            photometric = photo_im;
            planar_config = planar_im;
            samples = samples_im;
            format_def = defined;
            sampleformat = format_im;
            has_pixel = has_pixel_im;
            x_pixel = x_pixel_im;
            y_pixel = y_pixel_im;
            compression = compression_im as u16;

            if dirnum != 0 && !(eer_file != 0 || num_non_eer_images > 0) {
                (*in_file).multiple_sizes = 1;
            }
            dirnum = 1;
            let Some(directory_nums) = (*in_file).directory_nums.as_mut() else {
                close_with_error(in_file, "Memory error adding to directory list\n");
                return IIERR_MEMORY_ERR;
            };
            directory_nums.clear();
            directory_nums.push(file_dir_num);
        } else if nxim == (*in_file).nx && nyim == (*in_file).ny && skip_file == 0 {
            dirnum += 1;
            let Some(directory_nums) = (*in_file).directory_nums.as_mut() else {
                close_with_error(in_file, "Memory error adding to directory list\n");
                return IIERR_MEMORY_ERR;
            };
            directory_nums.push(file_dir_num);

            /* If size matches, check that everything matches */
            if bits_im != bits
                || photo_im != photometric
                || planar_im != planar_config
                || samples != samples_im
                || defined != format_def
                || (defined != 0 && format_im != sampleformat)
            {
                mismatch = 1;
                break;
            }
        } else if dirnum != 0 && skip_file == 0 {
            (*in_file).multiple_sizes = 1;
        }
        file_dir_num += 1;
        if TIFFReadDirectory(tif) == 0 {
            break;
        }
    }

    if skip_eer != 0 && num_non_eer_images == 0 {
        close_with_error(
            in_file,
            "ERROR: iiTIFFCheck - No non-EER images present in EER file\n",
        );
        return IIERR_NOT_PRESENT;
    }
    eer_file = if compression as i32 == IICOMPRESSION_EER_7BIT
        || compression as i32 == IICOMPRESSION_EER_8BIT
    {
        1
    } else {
        0
    };
    (*in_file).tiff_compression = compression as i32;

    /* get the min and max from last directory; if we wrote it, it applies to whole file */
    if TIFFGetField(tif, TIFFTAG_SMINSAMPLEVALUE, &raw mut last_min) != 0 {
        got_min = 1;
    }
    if TIFFGetField(tif, TIFFTAG_SMAXSAMPLEVALUE, &raw mut last_max) != 0 {
        got_max = 1;
    }
    if got_min == 0 && image_j != 0 && im_jmin < 8.0e37 {
        last_min = im_jmin as f64;
        got_min = 1;
    }
    if got_max == 0 && image_j != 0 && im_jmax > -8.0e37 {
        last_max = im_jmax as f64;
        got_max = 1;
    }
    if has_pixel == 0 && image_j != 0 && im_jpixel > 0. {
        x_pixel = im_jpixel;
        y_pixel = im_jpixel;
        has_pixel = 1;
    }

    TIFFSetDirectory(tif, 0);
    format_defined = TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &raw mut sampleformat);
    defined = TIFFGetField(tif, TIFFTAG_FILLORDER, &raw mut (*in_file).fill_order);
    if defined == 0 {
        (*in_file).fill_order = FILLORDER_MSB2LSB;
    }

    /* Don't know how to get the multiple bit entries from libtiff, so can't test
    if they are all 8.  Allow 4 bit if it is not signed and one plane */
    if mismatch != 0
        || !((((bits == 4
            && !(format_defined != 0 && sampleformat as i32 == SAMPLEFORMAT_INT)
            && samples == 1)
            || bits == 8
            || bits == 16
            || bits == 32)
            && (photometric as i32) < PHOTOMETRIC_RGB)
            || eer_file != 0
            || (bits == 8
                && (photometric as i32 == PHOTOMETRIC_RGB
                    || photometric as i32 == PHOTOMETRIC_PALETTE)))
    {
        close_with_error(
            in_file,
            "ERROR: iiTIFFCheck - Unsupported type of TIFF file\n",
        );
        return IIERR_NO_SUPPORT;
    }

    /* Recognize K2 files from SerialEMCCD and mark as 4-bit: require either that it has
    the exact dimensions of the full image with half size in X, or that it has the
    description added by SerialEMCCD */
    if bits == 8
        && (photometric as i32) < PHOTOMETRIC_RGB
        && samples == 1
        && has_date_time != 0
        && !(format_defined != 0 && sampleformat as i32 == SAMPLEFORMAT_INT)
        && (size_can_be_4_bit_k2_super_res((*in_file).nx, (*in_file).ny) != 0 || from_semccd != 0)
    {
        (*in_file).packed4bits = PACKED_HALF_XSIZE;
        (*in_file).nx *= 2;
        (*in_file).fill_order = FILLORDER_LSB2MSB;
    }

    /* And switch true 4-bit files to 8 bit with the flag set */
    if bits == 4 {
        (*in_file).packed4bits = PACKED_4BIT_MODE;
        bits = 8;
    }

    byte_max = 255;
    (*in_file).num_frames_in_eerfile = 0;
    if eer_file != 0 {
        /* Set up autogrouping: save the full # of frames, compute the number of groups
        and adjust the estimated frame mean for summing and super-res */
        (*in_file).num_frames_in_eerfile = dirnum;
        let autogroup = S_AUTOGROUP_EER.load(Ordering::SeqCst);
        if autogroup > 1 || (autogroup < 0 && -autogroup < dirnum) {
            if autogroup > 1 {
                dirnum = (dirnum + autogroup - 1) / autogroup;
            } else {
                dirnum = -autogroup;
            }
            num_in_autogroup = ((*in_file).num_frames_in_eerfile + dirnum - 1) / dirnum;
            frame_mean *= num_in_autogroup as f32;
        }

        /* Set size based on super-res and adjust maximum for that */
        max_fac = (2.0_f64.powf(S_MAX_EER_SUPER_RESOLUTION.load(Ordering::SeqCst) as f64) + 0.5)
            .floor() as i32;
        red_fac = (2.0_f64.powf(
            (S_MAX_EER_SUPER_RESOLUTION.load(Ordering::SeqCst)
                - S_READ_EER_AS_SUPER_RES.load(Ordering::SeqCst)) as f64,
        ) + 0.5)
            .floor() as i32;
        (*in_file).nx = ((*in_file).nx * max_fac) / red_fac;
        (*in_file).ny = ((*in_file).ny * max_fac) / red_fac;
        (*in_file).tile_size_y = ((*in_file).tile_size_y * max_fac) / red_fac;
        frame_mean = (frame_mean * red_fac as f32 * red_fac as f32) / (max_fac * max_fac) as f32;
        x_pixel = (x_pixel * red_fac as f32) / max_fac as f32;
        y_pixel = (y_pixel * red_fac as f32) / max_fac as f32;
        (*in_file).read_eer_as_super_res = S_READ_EER_AS_SUPER_RES.load(Ordering::SeqCst);

        /* Set the max as 3 SDs above mean for Poisson sample where SD = mean */
        byte_max = (4. * frame_mean as f64).ceil() as i32;
        byte_max = byte_max.clamp(1, 255);
        if S_ANTIALIAS_EER.load(Ordering::SeqCst) != 0 {
            byte_max = (4. * frame_mean as f64 * S_EER_KERNEL_SCALE.load(Ordering::SeqCst) as f64)
                .ceil() as i32;
            byte_max = byte_max.clamp(1, 32767);
        }
        (*in_file).antialias_eerfilter = S_ANTIALIAS_EER.load(Ordering::SeqCst);
        (*in_file).eerkernel_scale = S_EER_KERNEL_SCALE.load(Ordering::SeqCst);
    }
    if num_eer_images > 0 {
        S_EER_FLAGS.store(0, Ordering::SeqCst);
    }

    (*in_file).nz = dirnum;
    (*in_file).file = IIFILE_TIFF;
    (*in_file).format = IIFORMAT_LUMINANCE;

    /* Samples are assumed to be channels for RGB; record samples appropriately
    and set Z size otherwise for multiple samples */
    (*in_file).rgb_samples = samples as i32;
    if (photometric as i32) < PHOTOMETRIC_RGB {
        if planar_config as i32 == PLANARCONFIG_SEPARATE {
            (*in_file).planes_per_image = samples as i32;
        } else {
            (*in_file).contig_samples = samples as i32;
        }
        (*in_file).nz = dirnum * samples as i32;
    }

    /* 11/22/08: define this for all types, not just for 3-sample data */
    (*in_file).read_section = Some(tiff_read_section);
    (*in_file).read_section_ushort = Some(tiff_read_section_ushort);
    (*in_file).read_section_byte = Some(tiff_read_section_byte);
    (*in_file).read_section_float = Some(tiff_read_section_float);

    /* Set up file mode and default properties; fill in mode for raw info at same time */
    if bits == 8 || eer_file != 0 {
        (*in_file).type_ = if eer_file != 0 && S_ANTIALIAS_EER.load(Ordering::SeqCst) != 0 {
            IITYPE_SHORT
        } else {
            IITYPE_UBYTE
        };
        (*in_file).amin = 0.;
        (*in_file).amean = if eer_file != 0 {
            frame_mean
        } else {
            byte_max as f32 / 2.
        };
        (*in_file).amax = byte_max as f32;
        (*in_file).mode = MRC_MODE_BYTE;
        info.type_ = RAW_MODE_BYTE;
        if eer_file != 0 && S_ANTIALIAS_EER.load(Ordering::SeqCst) != 0 {
            (*in_file).mode = MRC_MODE_SHORT;
            info.type_ = RAW_MODE_SHORT;
        }

        /* Get the max right for 4-bit data, it matters for very low count data */
        if (*in_file).packed4bits != 0 {
            (*in_file).amean = 7.5;
            (*in_file).amax = 15.;
        }
        if photometric as i32 == PHOTOMETRIC_RGB {
            (*in_file).format = IIFORMAT_RGB;
            (*in_file).mode = MRC_MODE_RGB;
            info.type_ = RAW_MODE_RGB;
        } else if photometric as i32 == PHOTOMETRIC_PALETTE {
            /* For palette images, define as colormap, get the colormap and convert
            it to bytes */
            (*in_file).format = IIFORMAT_COLORMAP;
            (*in_file).mode = MRC_MODE_RGB;
            info.type_ = RAW_MODE_RGB;
            let mut colormap = Vec::new();
            if colormap
                .try_reserve_exact(3 * 256 * dirnum as usize)
                .is_err()
            {
                close_with_error(
                    in_file,
                    "ERROR: iiTIFFCheck - Getting memory for colormap\n",
                );
                return IIERR_MEMORY_ERR;
            }
            colormap.resize(3 * 256 * dirnum as usize, 0);
            (*in_file).colormap = Some(colormap);
            j = 0;
            while j < dirnum {
                if set_matching_directory(in_file, j) != 0 {
                    close_with_error(
                        in_file,
                        "ERROR: iiTIFFCheck - getting directory for colormap\n",
                    );
                    return IIERR_IO_ERROR;
                }

                TIFFGetField(
                    tif,
                    TIFFTAG_COLORMAP,
                    &raw mut redp,
                    &raw mut greenp,
                    &raw mut bluep,
                );
                i = 0;
                while i < 256 {
                    (*in_file).colormap.as_mut().unwrap()[(j * 768 + i) as usize] =
                        (*redp.add(i as usize) >> 8) as u8;
                    (*in_file).colormap.as_mut().unwrap()[(j * 768 + i + 256) as usize] =
                        (*greenp.add(i as usize) >> 8) as u8;
                    (*in_file).colormap.as_mut().unwrap()[(j * 768 + i + 512) as usize] =
                        (*bluep.add(i as usize) >> 8) as u8;
                    i += 1;
                }
                j += 1;
            }
            TIFFSetDirectory(tif, 0);
        } else if format_defined != 0 && sampleformat as i32 == SAMPLEFORMAT_INT {
            (*in_file).type_ = IITYPE_BYTE;
            info.type_ = RAW_MODE_SBYTE;
        }
    } else {
        /* If there is a field specifying signed numbers, set up for signed;
        otherwise set up for unsigned */
        if bits == 16 {
            if format_defined != 0 && sampleformat as i32 == SAMPLEFORMAT_INT {
                (*in_file).type_ = IITYPE_SHORT;
                (*in_file).amean = 0.;
                (*in_file).amin = -32767.;
                (*in_file).amax = 32767.;
                (*in_file).mode = MRC_MODE_SHORT;
                info.type_ = RAW_MODE_SHORT;
            } else {
                (*in_file).type_ = IITYPE_USHORT;
                (*in_file).amean = 32767.;
                (*in_file).amin = 0.;
                (*in_file).amax = 65535.;
                (*in_file).mode = MRC_MODE_USHORT; /* Why was this SHORT for both? */
                info.type_ = RAW_MODE_USHORT;
            }
        } else {
            /* Set up for integer data: until there is an MRC mode, set to -1 */
            if format_defined != 0 && sampleformat as i32 == SAMPLEFORMAT_INT {
                (*in_file).type_ = IITYPE_INT;
                (*in_file).amean = 0.;
                (*in_file).amin = -65536.;
                (*in_file).amax = 65536.;
                (*in_file).mode = -1;
            } else if format_defined != 0 && sampleformat as i32 == SAMPLEFORMAT_UINT {
                (*in_file).type_ = IITYPE_UINT;
                (*in_file).amean = 65536.;
                (*in_file).amin = 0.;
                (*in_file).amax = 130000.;
                (*in_file).mode = -1;
            } else if format_defined != 0 && sampleformat as i32 == SAMPLEFORMAT_IEEEFP {
                (*in_file).type_ = IITYPE_FLOAT;
                (*in_file).amean = 128.;
                (*in_file).amin = 0.;
                (*in_file).amax = 255.;
                (*in_file).mode = MRC_MODE_FLOAT;
                info.type_ = RAW_MODE_FLOAT;
            } else {
                close_with_error(
                    in_file,
                    "ERROR: iiTIFFCheck - 32-bit TIFF file with no data type defined\n",
                );
                return IIERR_NO_SUPPORT;
            }
        }
    }

    if TIFFGetField(
        tif,
        tvips_tag as u32,
        &raw mut bits,
        &raw mut (*in_file).user_data,
    ) > 0
    {
        (*in_file).user_count = bits as i32;
        (*in_file).user_flags = IIFLAG_TVIPS_DATA;
        if TIFFIsByteSwapped(tif) != 0 {
            (*in_file).user_flags |= IIFLAG_BYTES_SWAPPED;
        }
    }

    /* Use min and max from file if defined (better be there for float/int) */
    if got_min != 0 {
        if (*in_file).type_ == IITYPE_BYTE {
            last_min += 128.;
        }
        (*in_file).amin = last_min as f32;
    }
    if got_max != 0 {
        if (*in_file).type_ == IITYPE_BYTE {
            last_max += 128.;
        }
        (*in_file).amax = last_max as f32;
    }
    (*in_file).amean = if eer_file != 0 {
        frame_mean
    } else {
        ((*in_file).amin as f64 + (*in_file).amax as f64) as f32 / 2.
    };

    /* Handle pixel size if it is above threshold */
    if has_pixel != 0 && ((*in_file).any_tiff_pix_size != 0 || x_pixel / 1.0e4 <= pixel_limit) {
        (*in_file).xscale = x_pixel;
        (*in_file).yscale = y_pixel;
        (*in_file).zscale = x_pixel
            * if eer_file != 0 {
                num_in_autogroup as f32
            } else {
                1.
            };
    }

    (*in_file).smin = (*in_file).amin;
    (*in_file).smax = (*in_file).amax;

    /* Intercept ImageJ big tiff and try to convert to mrc-like */
    if image_j != 0
        && dirnum == 1
        && im_jslices == im_jimages
        && im_jimages > 1
        && (*in_file).planes_per_image == 1
        && photometric as i32 != PHOTOMETRIC_PALETTE
        && (*in_file).mode >= 0
    {
        info.nx = (*in_file).nx;
        info.ny = (*in_file).ny;
        info.nz = im_jimages;
        info.swap_bytes = TIFFIsByteSwapped(tif);
        info.section_skip = 0;
        info.y_inverted = 1;
        info.amin = (*in_file).amin;
        info.amax = (*in_file).amax;
        info.pixel = if has_pixel != 0 { x_pixel } else { 0. };
        info.z_pixel = info.pixel;
        if TIFFGetField(tif, TIFFTAG_STRIPOFFSETS, &raw mut offsets) > 0 {
            info.header_size = *offsets as i32;
            TIFFClose(tif);
            (*in_file).fp = ImodFile::open(
                (*in_file).filename.as_deref().unwrap_or(""),
                &(*in_file).fmode,
            );
            if (*in_file).fp.is_none() {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: iiTIFFCheck - Reopening file {} for treatment as MRC-like\n",
                        (*in_file).filename.as_deref().unwrap_or("")
                    ),
                );
                return IIERR_IO_ERROR;
            }
            return ii_setup_raw_headers(&mut *in_file, &info);
        }
    }

    /* Otherwise proceed to return as a TIFF file */
    (*in_file).header_size = 8;
    (*in_file).section_skip = 0;
    // `iitif.c:670`: `(FILE *)tif` — the libtiff handle used as this file's
    // identity in `sOpenedFiles`, never for I/O.
    (*in_file).fp = Some(ImodFile::Token(tif as usize));
    (*in_file).clean_up = Some(tiff_delete_callback);
    (*in_file).reopen = Some(tiff_reopen_callback);
    (*in_file).close = Some(tiff_close_callback);
    (*in_file).fill_mrc_header = Some(tiff_fill_mrc_header_callback);
    (*in_file).last_written_z = (*in_file).nz - 1;
    0
}
/// C `tiffReopen` (`iitif.c:679`).
pub fn tiff_reopen(in_file: &mut ImodImageFile) -> i32 {
    let tif = unsafe { open_without_b_mode(in_file) };
    if tif.is_null() {
        return 1;
    }
    in_file.header_size = 8;
    in_file.section_skip = 0;
    in_file.backend_handle = tif.cast();
    in_file.fp = Some(ImodFile::Token(tif as usize));
    0
}

unsafe fn tiff_reopen_callback(in_file: *mut ImodImageFile) -> i32 {
    let Some(in_file) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    tiff_reopen(in_file)
}
/// C `tiffClose` (`iitif.c:692`).
pub fn tiff_close(in_file: &mut ImodImageFile) {
    let tif = in_file.backend_handle.cast::<Tiff>();
    if in_file.new_file != 0
        && in_file.format != IIFORMAT_RGB
        && !tif.is_null()
        && in_file.amax > in_file.amin
    {
        unsafe { constrain_and_store_min_max(in_file) };
    }
    if !tif.is_null() {
        unsafe { TIFFClose(tif) };
    }
    in_file.backend_handle = core::ptr::null_mut();
    in_file.fp = None;
}

unsafe fn tiff_close_callback(in_file: *mut ImodImageFile) {
    if let Some(in_file) = unsafe { in_file.as_mut() } {
        tiff_close(in_file);
    }
}
/// C `tiffDelete` (`iitif.c:708`).
pub fn tiff_delete(in_file: &mut ImodImageFile) {
    in_file.directory_nums = None;
    tiff_close(in_file);
}

unsafe fn tiff_delete_callback(in_file: *mut ImodImageFile) {
    if let Some(in_file) = unsafe { in_file.as_mut() } {
        tiff_delete(in_file);
    }
}
/// C `tiffFillMrcHeader` (`iitif.c:715`).
pub fn tiff_fill_mrc_header(in_file: &ImodImageFile, hdata: &mut MrcHeader) -> i32 {
    mrc_head_new(hdata, in_file.nx, in_file.ny, in_file.nz, in_file.mode);
    hdata.bytes_signed = if in_file.type_ == IITYPE_BYTE { 1 } else { 0 };
    hdata.amin = in_file.amin;
    hdata.amean = in_file.amean;
    hdata.amax = in_file.amax;
    mrc_set_scale(
        hdata,
        in_file.xscale as f64,
        in_file.yscale as f64,
        in_file.zscale as f64,
    );
    hdata.fp = in_file.fp.clone();
    hdata.packed4bits = in_file.packed4bits;
    hdata.half_floats = 0;
    if hdata.packed4bits == PACKED_4BIT_MODE {
        hdata.iiu_flags |= IIUNIT_4BIT_MODE;
    }
    if hdata.packed4bits == PACKED_HALF_XSIZE {
        hdata.iiu_flags |= IIUNIT_HALF_XSIZE;
    }
    let tif = in_file.backend_handle.cast::<Tiff>();
    let mut description: *mut u8 = core::ptr::null_mut();
    if !tif.is_null()
        && unsafe { TIFFSetDirectory(tif, 0) } != 0
        && unsafe { TIFFGetField(tif, TIFFTAG_IMAGE_DESCRIPTION, &mut description) } != 0
        && !description.is_null()
    {
        hdata.nlabl = 0;
        // `iitif.c:741` takes `strlen(description)` of libtiff's buffer and
        // then indexes it; the bytes are read as bytes from here on.
        let mut desc_len = 0usize;
        while unsafe { *description.add(desc_len) } != 0 {
            desc_len += 1;
        }
        let description = unsafe { core::slice::from_raw_parts(description, desc_len) };
        let mut start_ind = 0;
        while hdata.nlabl < MRC_NLABELS as i32 && start_ind < description.len() {
            let mut end_ind = start_ind;
            while end_ind < description.len() && description[end_ind] != b'\n' {
                end_ind += 1;
            }
            if end_ind == description.len() {
                let dest = &mut hdata.labels[hdata.nlabl as usize];
                let count = (end_ind - start_ind).min(MRC_LABEL_SIZE);
                dest[..count].copy_from_slice(&description[start_ind..start_ind + count]);
                fix_title_padding(dest);
                hdata.nlabl += 1;
                break;
            }
            if end_ind != 0 && description[end_ind - 1] == b'\r' {
                end_ind -= 1;
            }
            if end_ind - start_ind > 0 {
                let dest = &mut hdata.labels[hdata.nlabl as usize];
                let count = (end_ind - start_ind).min(MRC_LABEL_SIZE);
                dest[..count].copy_from_slice(&description[start_ind..start_ind + count]);
                if end_ind - start_ind < MRC_LABEL_SIZE {
                    dest[end_ind - start_ind] = 0;
                }
                fix_title_padding(dest);
                hdata.nlabl += 1;
            }
            if description[end_ind] == b'\r' {
                end_ind += 1;
            }
            start_ind = end_ind + 1;
        }
    }
    0
}
unsafe fn tiff_fill_mrc_header_callback(
    in_file: *mut ImodImageFile,
    hdata: *mut MrcHeader,
) -> i32 {
    let Some(in_file) = (unsafe { in_file.as_ref() }) else {
        return IIERR_BAD_CALL;
    };
    let Some(hdata) = (unsafe { hdata.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    tiff_fill_mrc_header(in_file, hdata)
}
/// C `tiffSyncFromMrcHeader` (`iitif.c:771`).
fn tiff_sync_from_mrc_header(in_file: &mut ImodImageFile, hdata: &MrcHeader) -> i32 {
    in_file.description = None;
    if hdata.nlabl == 0 {
        return 0;
    }
    // Store Rust metadata bytes only. `tiff_add_description` creates its own
    // terminated copy at the libtiff boundary.
    in_file.description = Some(Vec::with_capacity(
        hdata.nlabl as usize * (MRC_LABEL_SIZE + 1),
    ));
    let description = in_file.description.as_mut().unwrap();
    let mut out_ind = 0;
    for lab in 0..hdata.nlabl as usize {
        let mut true_len = 0;
        for ind in 0..MRC_LABEL_SIZE {
            if hdata.labels[lab][ind] == b'\n' {
                break;
            }
            if hdata.labels[lab][ind] != b' ' {
                true_len = ind + 1;
            }
        }
        if true_len != 0 {
            let start_ind = out_ind;
            description.extend_from_slice(&hdata.labels[lab][..true_len]);
            out_ind += true_len;
            let line = &mut description[start_ind..start_ind + true_len];
            if let Some(bit_ind) = line
                .windows(b"4 bits packed".len())
                .position(|part| part == b"4 bits packed")
            {
                if line
                    .windows(b"SerialEMCCD".len())
                    .any(|part| part == b"SerialEMCCD")
                {
                    line[bit_ind + 1] = b'-';
                }
            }
            if lab != hdata.nlabl as usize - 1 {
                description.push(b'\n');
                out_ind += 1;
            }
        }
    }
    tiff_add_description(in_file.description.as_deref());
    0
}

unsafe fn tiff_sync_from_mrc_header_callback(
    in_file: *mut ImodImageFile,
    hdata: *mut MrcHeader,
) -> i32 {
    let Some(in_file) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    let Some(hdata) = (unsafe { hdata.as_ref() }) else {
        return IIERR_BAD_CALL;
    };
    tiff_sync_from_mrc_header(in_file, hdata)
}
/// C `tiffGetField` (`iitif.c:814`).
pub unsafe fn tiff_get_field(in_file: *mut ImodImageFile, tag: i32, value: *mut c_void) -> i32 {
    if in_file.is_null() {
        return -1;
    }
    if (*in_file).backend_handle.is_null() {
        let _ = tiff_reopen(&mut *in_file);
    }
    let tif = (*in_file).backend_handle.cast::<Tiff>();
    if tif.is_null() {
        return -1;
    }
    TIFFGetField(tif, tag as u32, value)
}
/// C `tiffGetArray` (`iitif.c:829`).
///
/// libtiff owns the returned tag memory. Copy it at this boundary so the Rust
/// image-processing path never stores or exposes that borrowed C pointer.
pub fn tiff_get_array(in_file: &mut ImodImageFile, tag: i32) -> Result<Vec<u8>, i32> {
    if in_file.backend_handle.is_null() {
        let _ = unsafe { tiff_reopen(in_file) };
    }
    let tif = in_file.backend_handle.cast::<Tiff>();
    if tif.is_null() {
        return Err(-1);
    }
    let mut count = 0_u32;
    let mut value: *mut u8 = core::ptr::null_mut();
    if unsafe {
        TIFFGetField(
            tif,
            tag as u32,
            &mut count,
            (&mut value as *mut *mut u8).cast::<c_void>(),
        )
    } <= 0
        || value.is_null()
    {
        return Err(-1);
    }
    Ok(unsafe { core::slice::from_raw_parts(value, count as usize) }.to_vec())
}
/// C `tiffSuppressErrors` (`iitif.c:843`).
pub fn tiff_suppress_errors() {
    if S_OLD_ERR_HANDLER.load(Ordering::SeqCst).is_null() {
        S_OLD_ERR_HANDLER.store(
            unsafe { TIFFSetErrorHandler(core::ptr::null_mut()) },
            Ordering::SeqCst,
        );
    } else {
        let _ = unsafe { TIFFSetErrorHandler(core::ptr::null_mut()) };
    }
}
/// C `tiffRestoreErrors` (`iitif.c:851`).
pub fn tiff_restore_errors() {
    let _ = unsafe { TIFFSetErrorHandler(S_OLD_ERR_HANDLER.load(Ordering::SeqCst)) };
}
/// C `tiffSuppressWarnings` (`iitif.c:856`).
pub fn tiff_suppress_warnings() {
    let _ = unsafe { TIFFSetWarningHandler(core::ptr::null_mut()) };
    S_WARNINGS_SUPPRESSED.store(1, Ordering::SeqCst);
}
/// C `warningHandler` (`iitif.c:862`).
unsafe extern "C" fn warning_handler(
    module: *const c_char,
    format: *const c_char,
    ap: *mut c_void,
) {
    // `vsnprintf` and the `va_list` it reads are libtiff's own message format,
    // the one thing in this unit that has to stay a C string; the bytes it
    // leaves in `buffer` are bytes from here on.
    let mut buffer = [0u8; 160];
    vsnprintf(buffer.as_mut_ptr().cast(), 159, format, ap);
    /* va_end(ap) is a no-op in this ABI */

    /* It didn't work to call the old handler with some errors, so print it
    ourselves to stderr.  It was "unknown" in libtiff 3 and "Unknown" in 4 */
    let text = &buffer[..buffer
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(buffer.len())];
    if !text
        .windows(b"nknown field with tag".len())
        .any(|part| part == b"nknown field with tag")
    {
        if !module.is_null() {
            let mut mod_len = 0usize;
            while *module.add(mod_len) != 0 {
                mod_len += 1;
            }
            let _ = ImodFile::Stderr.write_all(&c_format_bytes(
                "%s: Warning, %s\n",
                &[
                    CArg::Bytes(core::slice::from_raw_parts(module.cast::<u8>(), mod_len)),
                    CArg::Bytes(text),
                ],
            ));
        } else {
            let _ =
                ImodFile::Stderr.write_all(&c_format_bytes("Warning, %s\n", &[CArg::Bytes(text)]));
        }
    }
}
/// C `tiffFilterWarnings` (`iitif.c:879`).
pub fn tiff_filter_warnings() {
    if S_WARNINGS_SUPPRESSED.load(Ordering::SeqCst) == 0 {
        S_OLD_HANDLER.store(
            unsafe { TIFFSetWarningHandler(warning_handler as *mut c_void) },
            Ordering::SeqCst,
        );
    }
}
/// C `tiffSetMapping` (`iitif.c:885`).
pub fn tiff_set_mapping(value: i32) {
    S_USE_MAPPING.store(value, Ordering::SeqCst);
}
/// C `tiffSetEERreadProperties` (`iitif.c:896`).
///
/// Set the properties for opening EER files and flags for opening or reading.
/// superRes can be 0 to the maximum allowed value; autogroup specifies the frame
/// summing where a negative value specifies the number of frames to sum to;
/// flags are defined in iimage.h.  Flags are cleared when opening or reading from
/// file.  The flag to ignore bad endings is stored separately, not cleared between
/// invocations, and can be overridden by an environment variable.
pub fn tiff_set_eer_read_properties(super_res: i32, autogroup: i32, flags: i32) {
    if S_IGNORE_FROM_VAR.load(Ordering::SeqCst) == -999 {
        if let Some(ignore_var) = std::env::var_os("IGNORE_BAD_EER_ENDING") {
            let mut end = 0usize;
            S_IGNORE_FROM_VAR.store(
                strtol(ignore_var.as_encoded_bytes(), &mut end, 10) as i32,
                Ordering::SeqCst,
            );
        } else {
            S_IGNORE_FROM_VAR.store(-997, Ordering::SeqCst);
        }
    }
    S_READ_EER_AS_SUPER_RES.store(super_res, Ordering::SeqCst);
    S_EER_FLAGS.store(flags, Ordering::SeqCst);
    S_IGNORE_BAD_EER_END.store(
        (flags & IIFLAG_IGNORE_BAD_EER_END != 0) as i32,
        Ordering::SeqCst,
    );
    if S_IGNORE_FROM_VAR.load(Ordering::SeqCst) > -990 {
        S_IGNORE_BAD_EER_END.store(S_IGNORE_FROM_VAR.load(Ordering::SeqCst), Ordering::SeqCst);
    }
    S_AUTOGROUP_EER.store(autogroup, Ordering::SeqCst);
    let super_res = super_res.clamp(
        S_MIN_EER_SUPER_RESOLUTION.load(Ordering::SeqCst),
        S_MAX_EER_SUPER_RESOLUTION.load(Ordering::SeqCst),
    );
    S_READ_EER_AS_SUPER_RES.store(super_res, Ordering::SeqCst);
    let antialias = if flags & IIFLAG_ANTIALIAS_EER != 0 {
        3 + (flags & IIFLAG_EER_USE_LANCZOS != 0) as i32
    } else {
        0
    };
    S_ANTIALIAS_EER.store(antialias, Ordering::SeqCst);
    S_EER_KERNEL_SCALE.store(
        (((flags >> EER_AA_SCALING_BIT_SHIFT) & EER_AA_SCALING_MASK) + 1) * 100,
        Ordering::SeqCst,
    );
    S_GAIN_REFERENCE.store(core::ptr::null_mut(), Ordering::SeqCst);
}
/// C `tiffGainReferenceForEER` (`iitif.c:922`).
pub fn tiff_gain_reference_for_eer(reference: *mut f32) {
    S_GAIN_REFERENCE.store(reference, Ordering::SeqCst);
}
/// C `tiffGetMaxEERsuperRes` (`iitif.c:928`).
pub fn tiff_get_max_eer_super_res() -> i32 {
    S_MAX_EER_SUPER_RESOLUTION.load(Ordering::SeqCst)
}
/// C `tiffGetMinEERsuperRes` (`iitif.c:933`).
pub fn tiff_get_min_eer_super_res() -> i32 {
    S_MIN_EER_SUPER_RESOLUTION.load(Ordering::SeqCst)
}
/// C `tiffSetStringTagToPrint` (`iitif.c:938`).
pub fn tiff_set_string_tag_to_print(tag: i32) {
    S_TAG_TO_PRINT.store(tag, Ordering::SeqCst);
}
/// C `openWithoutBMode` (`iitif.c:944`).
unsafe fn open_without_b_mode(in_file: *mut ImodImageFile) -> *mut Tiff {
    if in_file.is_null() {
        return core::ptr::null_mut();
    }
    let Some(filename) = (&(*in_file).filename).as_deref() else {
        return core::ptr::null_mut();
    };
    let mode = (*in_file).fmode.as_bytes();
    if mode.is_empty() {
        return core::ptr::null_mut();
    }
    if S_USE_MAPPING.load(Ordering::SeqCst) == 2 {
        S_USE_MAPPING.store(
            if std::env::var_os("IMOD_NO_TIFF_MEM_MAP").is_none() {
                1
            } else {
                0
            },
            Ordering::SeqCst,
        );
    }
    let mut open_mode = mode.to_vec();
    if open_mode.last() == Some(&b'b') {
        open_mode.pop();
    }
    if S_USE_MAPPING.load(Ordering::SeqCst) == 0 {
        open_mode.push(b'm');
    }
    open_mode.push(0);
    // The libtiff line: the name and the mode become C strings here, and
    // nowhere above this call.
    let mut name = filename.as_bytes().to_vec();
    name.push(0);
    TIFFOpen(name.as_ptr().cast(), open_mode.as_ptr().cast())
}
/// C `setMatchingDirectory` (`iitif.c:984`).
unsafe fn set_matching_directory(in_file: *mut ImodImageFile, dirnum: i32) -> i32 {
    if in_file.is_null() || (*in_file).backend_handle.is_null() || dirnum < 0 {
        return 1;
    }
    let Some(directory_list) = (*in_file).directory_nums.as_ref() else {
        return 1;
    };
    let Some(&directory) = directory_list.get(dirnum as usize) else {
        return 1;
    };
    (TIFFSetDirectory((*in_file).backend_handle.cast(), directory as u16) == 0) as i32
}
/// C `closeWithError` (`iitif.c:994`).
unsafe fn close_with_error(in_file: *mut ImodImageFile, message: &str) {
    TIFFClose((*in_file).backend_handle.cast());
    (*in_file).fp = ImodFile::open(
        (*in_file).filename.as_deref().unwrap_or(""),
        &(*in_file).fmode,
    );
    b3d_error(Some(&mut ImodFile::Stderr), format_args!("{}", message));
}
/// C `countEERBytesAndElectrons` (`iitif.c:1005`).
unsafe fn count_eer_bytes_and_electrons(
    tif: *mut Tiff,
    is_7bit_eer: i32,
    raw_total_bytes: *mut i32,
    max_electrons: *mut i32,
) {
    let nstrip = TIFFNumberOfStrips(tif) as i32;
    *raw_total_bytes = 0;

    /* Add up the strip sizes to get buffer needs */
    let mut si = 0;
    while si < nstrip {
        *raw_total_bytes = (*raw_total_bytes).wrapping_add(TIFFRawStripSize(tif, si as u32) as i32);
        si += 1;
    }

    /* maximum number of electrons is 11 or 12 bits each */
    *max_electrons =
        8i32.wrapping_mul(*raw_total_bytes) / (if is_7bit_eer != 0 { 7 } else { 8 } + 4);
}
/// C `ReadSection` (`iitif.c:1023`).
///
/// DNM 12/24/00: Got this working for bytes, shorts, and RGBs, for whole
/// images or subsets, and using maps for scaling.
/// DNM 11/18/01: Added ability to read tiles, made tiffReadSection and
/// tiffReadSectionByte call a common routine to reduce duplicate code.
unsafe fn read_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    convert: i32,
) -> i32 {
    let mut nstrip: i32;
    let mut si: i32;
    let plane: i32;
    let sample_offset: i32;
    let xout: i32;
    let mut xcopy: i32 = 0;
    let xsize = (*in_file).nx;
    let ysize = (*in_file).ny;
    let samples = (*in_file).contig_samples;
    let mut row: i32;
    let mut xstart: i32;
    let mut xend: i32;
    let mut ystart: i32;
    let mut yend: i32;
    let mut y: i32;
    let mut ofsin: i32;
    let mut red_fac: i32;
    let mut ofsout: usize;
    let doscale: i32;
    let byte = if convert == MRSA_BYTE { 1 } else { 0 };
    let to_short = if convert == MRSA_USHORT { 1 } else { 0 };
    let to_float = if convert == MRSA_FLOAT { 1 } else { 0 };
    let signed_bytes = if (*in_file).type_ == IITYPE_BYTE {
        1
    } else {
        0
    };
    let slope = (*in_file).slope;
    let offset = (*in_file).offset;
    let outmin = 0;
    let outmax = if to_short != 0 { 65535 } else { 255 };
    let eps: f32 = if to_short != 0 { 0.005 / 256. } else { 0.005 };
    let pad_left = 0.max((*in_file).pad_left);
    let pad_right = 0.max((*in_file).pad_right);
    let mut stripsize: i32;
    let mut max_electrons: i32 = 0;
    let is_7bit_eer: i32;
    let mut raw_total_bytes: i32 = 0;
    let mut out_buf_size: i32;
    let is_eerfile: i32;
    let mut num_electrons: i32 = 0;
    let byte_buf_size: i32;
    let mut chip_size: i32;
    let start_finish: i32;
    let add_to_sum: i32;
    let auto_group_eer: i32;
    let mut sec_start: i32 = 0;
    let mut sec_end: i32 = 0;
    let mut raw_arr_size: i32;
    let mut elec_arr_size: i32;
    let need_convert: i32;
    let mut positions_data = Vec::<i32>::new();
    let mut symbols_data = Vec::<u8>::new();
    let mut positions: *mut i32 = core::ptr::null_mut();
    let mut symbols: *mut u8 = core::ptr::null_mut();
    let xmin: i32;
    let xmax: i32;
    let ymin: i32;
    let ymax: i32;
    let mut pixsize: i32 = 1;
    let mut move_size: i32 = if to_short != 0 { 2 } else { 1 };
    let mut obuf: *mut u8;
    let mut tmp_data = Vec::<u8>::new();
    let mut eer_data = Vec::<u8>::new();
    let mut tmp: *mut u8 = core::ptr::null_mut();
    let mut eer_buf: *mut u8 = core::ptr::null_mut();
    let mut use_buf: *mut u8 = core::ptr::null_mut();
    let mut bdata: *mut u8;
    let mut sobuf: *mut i16;
    let mut sdata: *mut i16;
    let mut map_data = Vec::<u8>::new();
    let mut map: *mut u8 = core::ptr::null_mut();
    let mut first_4bits_map = [0u8; 256];
    let mut second_4bits_map = [0u8; 256];

    /* Send a color map to line converter for a colormap image if converting or
    returning RGB values, but not if the actual bytes are wanted */
    let colormap: *mut u8 = if (*in_file).format == IIFORMAT_COLORMAP
        && (convert != 0 || (*in_file).raw_palette_bytes == 0)
    {
        (*in_file)
            .colormap
            .as_mut()
            .map_or(core::ptr::null_mut(), |map| {
                map.as_mut_ptr().add(768 * in_section as usize)
            })
    } else {
        core::ptr::null_mut()
    };
    let mut rowsperstrip: u32 = 0;
    let mut nread: isize;
    let mut line_bytes: i32;
    let mut line_offset: i32;
    let mut skip_half: i32;
    let tilesize: isize;
    let mut tilewidth: i32 = 0;
    let mut tilelength: i32 = 0;
    let xtiles: i32;
    let ytiles: i32;
    let mut xti: i32;
    let mut yti: i32;
    let x_dimension: i32;

    let mut tif = (*in_file).backend_handle.cast::<Tiff>();
    if (*in_file).axis == 2 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: tiffReadSection - Cannot read Y planes from a TIFF file\n"),
        );
        return -1;
    }

    if tif.is_null() {
        ii_reopen(&mut *in_file);
    }
    tif = (*in_file).backend_handle.cast::<Tiff>();
    if tif.is_null() {
        return -1;
    }

    is_eerfile = if (*in_file).tiff_compression == IICOMPRESSION_EER_7BIT
        || (*in_file).tiff_compression == IICOMPRESSION_EER_8BIT
    {
        1
    } else {
        0
    };
    auto_group_eer = if is_eerfile != 0 && (*in_file).num_frames_in_eerfile > (*in_file).nz {
        1
    } else {
        0
    };
    if auto_group_eer != 0
        && S_EER_FLAGS.load(Ordering::SeqCst) & (IIFLAG_ADD_TO_EER_SUM | IIFLAG_START_END_EER_SUM)
            != 0
    {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: tiffReadSection - Cannot set flags for summing frames when autogrouping an EER file\n"
            ),
        );
        return -1;
    }

    /* set the dimensions to read in */
    /* DNM 2/26/03: replace upper right only if negative */
    xmin = (*in_file).llx;
    ymin = (*in_file).lly;
    if (*in_file).urx < 0 {
        xmax = (*in_file).nx - 1;
    } else {
        xmax = (*in_file).urx;
    }
    if (*in_file).ury < 0 {
        ymax = (*in_file).ny - 1;
    } else {
        ymax = (*in_file).ury;
    }
    xout = xmax + 1 - xmin;
    x_dimension = xout + pad_left + pad_right;
    doscale = if convert != 0
        && (offset <= -1.0 || offset >= 1.0 || slope < 1. - eps || slope > 1. + eps)
    {
        1
    } else {
        0
    };
    line_bytes = xsize;
    line_offset = xmin;
    skip_half = 0;

    /* Modify bytes and offset appropriately and set flag to skip half byte at start of line
    for 4-bit, also set up byte to first/second pixel maps appropriate for fill order */
    if (*in_file).packed4bits != 0 {
        line_bytes = (xsize + 1) / 2;
        line_offset = xmin / 2;
        skip_half = xmin % 2;
        let low_map: *mut u8;
        let high_map: *mut u8;
        if (*in_file).fill_order == FILLORDER_LSB2MSB {
            low_map = first_4bits_map.as_mut_ptr();
            high_map = second_4bits_map.as_mut_ptr();
        } else {
            high_map = first_4bits_map.as_mut_ptr();
            low_map = second_4bits_map.as_mut_ptr();
        }
        xti = 0;
        while xti < 16 {
            yti = 0;
            while yti < 16 {
                *low_map.add((xti + 16 * yti) as usize) = xti as u8;
                *high_map.add((xti + 16 * yti) as usize) = yti as u8;
                yti += 1;
            }
            xti += 1;
        }
    }

    row = in_section / ((*in_file).planes_per_image * samples);
    if auto_group_eer == 0 && set_matching_directory(in_file, row) != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: TIFF ReadSection - Cannot find directory {}\n", row),
        );
        return -1;
    }
    plane = in_section % (*in_file).planes_per_image;
    sample_offset = in_section % samples;

    /* Set up pixsize which is the number of bytes in the input data, and
    moveSize which is number of bytes of output.  Also get scale maps */
    if (convert != 0 && to_float == 0) || signed_bytes != 0 {
        if (*in_file).type_ == IITYPE_SHORT {
            pixsize = 2;
            map_data = get_short_map(slope, offset, outmin, outmax, MRC_RAMP_LIN, 0, 1);
            map = map_data.as_mut_ptr();
        } else if (*in_file).type_ == IITYPE_USHORT {
            pixsize = 2;
            if byte != 0 || doscale != 0 {
                map_data = get_short_map(slope, offset, outmin, outmax, MRC_RAMP_LIN, 0, 0);
                map = map_data.as_mut_ptr();
            }
        } else if (*in_file).type_ == IITYPE_FLOAT
            || (*in_file).type_ == IITYPE_INT
            || (*in_file).type_ == IITYPE_UINT
        {
            pixsize = 4;
        } else if ((to_short != 0 || doscale != 0) && colormap.is_null()) || signed_bytes != 0 {
            map_data = get_byte_map(
                slope,
                offset,
                outmin,
                outmax,
                if signed_bytes != 0 { 1 } else { 0 },
            );
            map = map_data.as_mut_ptr();
            if (*in_file).format == IIFORMAT_RGB {
                pixsize = (*in_file).rgb_samples;
            }
        } else if (*in_file).format == IIFORMAT_RGB {
            pixsize = (*in_file).rgb_samples;
        }
        if to_float != 0 {
            move_size = 4;
        }
    } else {
        if (*in_file).format == IIFORMAT_RGB {
            pixsize = (*in_file).rgb_samples;
        } else if (*in_file).type_ == IITYPE_SHORT || (*in_file).type_ == IITYPE_USHORT {
            pixsize = 2;
        } else if (*in_file).type_ == IITYPE_FLOAT
            || (*in_file).type_ == IITYPE_INT
            || (*in_file).type_ == IITYPE_UINT
        {
            pixsize = 4;
        }
        if to_float != 0 {
            move_size = 4;
        } else {
            move_size = if (*in_file).format == IIFORMAT_RGB || !colormap.is_null() {
                3
            } else {
                pixsize
            };
        }
    }

    if is_eerfile != 0 {
        is_7bit_eer = if (*in_file).tiff_compression == IICOMPRESSION_EER_7BIT {
            1
        } else {
            0
        };
        sec_start = in_section;
        sec_end = in_section;
        if auto_group_eer != 0 {
            group_limits_remainder_at_end(
                (*in_file).num_frames_in_eerfile,
                (*in_file).nz,
                in_section,
                &mut sec_start,
                &mut sec_end,
            );
        }
        raw_arr_size = 0;
        elec_arr_size = 0;
        byte_buf_size = xout * (ymax + 1 - ymin) * pixsize;
        if pixsize > move_size {
            eer_data.resize(byte_buf_size as usize, 0);
            eer_buf = eer_data.as_mut_ptr();
            use_buf = eer_buf;
        }

        /* Allocate arrays for antialiasing */
        if (*in_file).antialias_eerfilter != 0 && (*in_file).read_eer_as_super_res < 2 {
            red_fac =
                (2.0_f64.powf((2 - (*in_file).read_eer_as_super_res) as f64) + 0.5).floor() as i32;
            si = red_fac * red_fac;
            let mut filters = S_EER_FILTERS.lock().unwrap();
            filters.all = vec![0; (si * 16) as usize];
            filters.x_start = vec![0; si as usize];
            filters.y_start = vec![0; si as usize];

            if select_zoom_filter(
                (*in_file).antialias_eerfilter,
                1. / red_fac as f64,
                &mut row,
            ) != 0
            {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!("ERROR: tiffReadSection - Setting up antialias filter\n"),
                );
                return -1;
            }
        }

        row = sec_start;
        while row <= sec_end {
            if auto_group_eer != 0 && set_matching_directory(in_file, row) != 0 {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!("ERROR: tiffReadSection - Cannot find directory {}\n", row),
                );
                cleanup_from_eer(tmp, core::ptr::null_mut(), positions, symbols, eer_buf);
                return -1;
            }

            /* Get buffer needs */
            nstrip = TIFFNumberOfStrips(tif) as i32;
            count_eer_bytes_and_electrons(
                tif,
                is_7bit_eer,
                &mut raw_total_bytes,
                &mut max_electrons,
            );

            /* Allocate arrays; make sure they are big enough */
            if raw_total_bytes + 10 > raw_arr_size {
                raw_arr_size = (1.05 * raw_total_bytes as f64) as i32 + 10;
                tmp_data.resize(raw_arr_size as usize, 0);
                tmp = tmp_data.as_mut_ptr();
            }

            if max_electrons + 10 > elec_arr_size {
                elec_arr_size = (1.05 * max_electrons as f64) as i32 + 10;
                positions_data.resize(elec_arr_size as usize, 0);
                symbols_data.resize(elec_arr_size as usize, 0);
                positions = positions_data.as_mut_ptr();
                symbols = symbols_data.as_mut_ptr();
            }

            /* Read in the data */
            ystart = 0;
            si = 0;
            while si < nstrip {
                stripsize = TIFFRawStripSize(tif, si as u32) as i32;
                nread = TIFFReadRawStrip(
                    tif,
                    si as u32,
                    tmp.add(ystart as usize).cast(),
                    stripsize as isize,
                );
                if nread < 0 {
                    cleanup_from_eer(tmp, core::ptr::null_mut(), positions, symbols, eer_buf);
                    return IIERR_IO_ERROR;
                }
                ystart += stripsize;
                si += 1;
            }
            chip_size = xsize * ysize;
            if (*in_file).read_eer_as_super_res >= 0 {
                si = 0;
                while si < (*in_file).read_eer_as_super_res {
                    chip_size /= 4;
                    si += 1;
                }
            } else {
                si = (*in_file).read_eer_as_super_res;
                while si < 0 {
                    chip_size *= 4;
                    si += 1;
                }
            }

            /* Set the flags for this frame */
            if auto_group_eer != 0 {
                let mut flags = 0;
                if row > sec_start {
                    flags = IIFLAG_ADD_TO_EER_SUM;
                }
                if row == sec_start || row == sec_end {
                    flags |= IIFLAG_START_END_EER_SUM;
                }
                S_EER_FLAGS.store(flags, Ordering::SeqCst);
            }

            /* Decode it */
            if decode_eer_image(
                tmp,
                positions,
                symbols,
                is_7bit_eer,
                chip_size,
                raw_total_bytes,
                &mut num_electrons,
            ) != 0
            {
                cleanup_from_eer(tmp, core::ptr::null_mut(), positions, symbols, eer_buf);
                return IIERR_IO_ERROR;
            }

            out_buf_size = move_size * x_dimension * (ymax + 1 - ymin);
            if eer_buf.is_null() {
                use_buf = buf
                    .cast::<u8>()
                    .offset((out_buf_size - byte_buf_size) as isize);
            }
            convert_eer_positions(
                in_file,
                positions,
                symbols,
                num_electrons,
                use_buf,
                xmin,
                xmax,
                ymin,
                ymax,
            );
            row += 1;
        }

        /* The data need conversion/copying if convert is set, and not to a byte with no map,
        or if there is any padding */
        start_finish = S_EER_FLAGS.load(Ordering::SeqCst) & IIFLAG_START_END_EER_SUM;
        add_to_sum = S_EER_FLAGS.load(Ordering::SeqCst) & IIFLAG_ADD_TO_EER_SUM;
        need_convert =
            if convert != 0 && (convert != MRSA_BYTE || !map.is_null() || !eer_buf.is_null()) {
                1
            } else {
                0
            };
        if (need_convert != 0 || pad_left != 0 || pad_right != 0)
            && ((start_finish == 0 && add_to_sum == 0) || (start_finish != 0 && add_to_sum != 0))
        {
            y = 0;
            while y <= ymax - ymin {
                ofsin = y * xout * pixsize;
                bdata = use_buf.add(ofsin as usize);
                ofsout = (move_size * (y * x_dimension + pad_left)) as usize;
                obuf = buf.cast::<u8>().add(ofsout);

                /* But if there is just byte data to be copied because of padding, it has to be
                done explicitly due to overlap at some point */
                if need_convert == 0 {
                    if pixsize == 1 {
                        si = 0;
                        while si < xout {
                            *obuf.add(si as usize) = *bdata.add(si as usize);
                            si += 1;
                        }
                    } else {
                        sdata = bdata.cast::<i16>();
                        sobuf = obuf.cast::<i16>();
                        si = 0;
                        while si < xout {
                            *sobuf.add(si as usize) = *sdata.add(si as usize);
                            si += 1;
                        }
                    }
                } else {
                    /* Otherwise use the copy routine */
                    copy_line(
                        bdata,
                        obuf,
                        xout,
                        convert,
                        pixsize,
                        (*in_file).type_,
                        (*in_file).format,
                        samples,
                        slope,
                        offset,
                        doscale,
                        0,
                        map,
                        core::ptr::null_mut(),
                        core::ptr::null_mut(),
                        core::ptr::null_mut(),
                    );
                }
                y += 1;
            }
        }
        cleanup_from_eer(tmp, core::ptr::null_mut(), positions, symbols, eer_buf);
        return 0;
    } else if TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &mut rowsperstrip) != 0 {
        /* if data are in strips, get strip size and memory for it */
        stripsize = TIFFStripSize(tif) as i32;
        let Ok(buffer_size) = usize::try_from(stripsize) else {
            return -1;
        };
        if tmp_data.try_reserve_exact(buffer_size).is_err() {
            return -1;
        }
        tmp_data.resize(buffer_size, 0);
        tmp = tmp_data.as_mut_ptr();

        nstrip = TIFFNumberOfStrips(tif) as i32 / (*in_file).planes_per_image;

        si = 0;
        while si < nstrip {
            /* Compute starting and ending Y values to use in each strip */
            ystart = ysize - 1 - (rowsperstrip as i32 * (si + 1) - 1);
            yend = ysize - 1 - (rowsperstrip as i32 * si);
            if ymin > ystart {
                ystart = ymin;
            }
            if ymax < yend {
                yend = ymax;
            }
            if ystart > yend {
                si += 1;
                continue;
            }

            /* Read the strip if necessary */
            nread = TIFFReadEncodedStrip(
                tif,
                (si + plane * nstrip) as u32,
                tmp.cast(),
                stripsize as isize,
            );
            let _ = nread;
            y = ystart;
            while y <= yend {
                /* for each y, compute back to row, and get offsets into
                input and output arrays */
                row = ysize - 1 - y - rowsperstrip as i32 * si;
                ofsin =
                    samples * pixsize * (row * line_bytes + line_offset) + sample_offset * pixsize;
                ofsout = (move_size as usize)
                    * ((y - ymin) as usize * x_dimension as usize + pad_left as usize);
                obuf = buf.cast::<u8>().add(ofsout);
                bdata = tmp.add(ofsin as usize);
                copy_line(
                    bdata,
                    obuf,
                    xout,
                    convert,
                    pixsize,
                    (*in_file).type_,
                    (*in_file).format,
                    samples,
                    slope,
                    offset,
                    doscale,
                    if (*in_file).packed4bits != 0 {
                        1 + skip_half
                    } else {
                        0
                    },
                    map,
                    first_4bits_map.as_mut_ptr(),
                    second_4bits_map.as_mut_ptr(),
                    colormap,
                );
                y += 1;
            }
            si += 1;
        }
    } else {
        /* Otherwise make sure there are tiles, if not return with error */
        if TIFFGetField(tif, TIFFTAG_TILEWIDTH, &mut tilewidth) != 0 {
            tilesize = TIFFTileSize(tif);
            let Ok(buffer_size) = usize::try_from(tilesize) else {
                return -1;
            };
            if tmp_data.try_reserve_exact(buffer_size).is_err() {
                return -1;
            }
            tmp_data.resize(buffer_size, 0);
            tmp = tmp_data.as_mut_ptr();
        } else {
            tilesize = 0;
        }
        let _ = tilesize;
        if tmp.is_null() {
            return -1;
        }
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &mut tilelength);
        xtiles = (xsize + tilewidth - 1) / tilewidth;
        ytiles = (ysize + tilelength - 1) / tilelength;

        yti = 0;
        while yti < ytiles {
            xti = 0;
            while xti < xtiles {
                /* Compute starting and ending Y then X values to use in
                this tile */
                ystart = ysize - 1 - (tilelength * (yti + 1) - 1);
                yend = ysize - 1 - (tilelength * yti);
                if ymin > ystart {
                    ystart = ymin;
                }
                if ymax < yend {
                    yend = ymax;
                }
                if ystart > yend {
                    xti += 1;
                    continue;
                }

                xstart = xti * tilewidth;
                xend = xstart + tilewidth - 1;
                if xmin > xstart {
                    xstart = xmin;
                }
                if xmax < xend {
                    xend = xmax;
                }
                if xstart > xend {
                    xti += 1;
                    continue;
                }

                /* Read the tile if necessary */
                si = xti + yti * xtiles + plane * xtiles * ytiles;
                nread = TIFFReadEncodedTile(tif, si as u32, tmp.cast(), tilesize);
                let _ = nread;
                xcopy = xend + 1 - xstart;

                /* Set up bytes and offset appropriately for this tile */
                line_bytes = tilewidth;
                line_offset = xstart - xti * tilewidth;
                skip_half = 0;
                if (*in_file).packed4bits != 0 {
                    line_bytes = (tilewidth + 1) / 2;
                    skip_half = line_offset % 2;
                    line_offset /= 2;
                }

                y = ystart;
                while y <= yend {
                    /* for each y, compute back to row, and get offsets
                    into input and output arrays */
                    row = ysize - 1 - y - tilelength * yti;
                    ofsin = pixsize * samples * (row * line_bytes + line_offset)
                        + sample_offset * pixsize;
                    ofsout = (move_size as usize)
                        * ((y - ymin) as usize * x_dimension as usize
                            + pad_left as usize
                            + (xstart - xmin) as usize);
                    obuf = buf.cast::<u8>().add(ofsout);
                    bdata = tmp.add(ofsin as usize);
                    copy_line(
                        bdata,
                        obuf,
                        xcopy,
                        convert,
                        pixsize,
                        (*in_file).type_,
                        (*in_file).format,
                        samples,
                        slope,
                        offset,
                        doscale,
                        if (*in_file).packed4bits != 0 {
                            1 + skip_half
                        } else {
                            0
                        },
                        map,
                        first_4bits_map.as_mut_ptr(),
                        second_4bits_map.as_mut_ptr(),
                        colormap,
                    );
                    y += 1;
                }
                xti += 1;
            }
            yti += 1;
        }
    }
    0
}
/// C `decodeEERimage` (`iitif.c:1493`).
///
/// Uncompress the run-length encoded EER image.  The source is compiled with
/// `NO_WASTED_BITS` defined (`iitif.c:100`), so the misalignment handling in the
/// 8-bit branch is the selected variant.
unsafe fn decode_eer_image(
    buf: *mut u8,
    positions: *mut i32,
    symbols: *mut u8,
    is_7bit_eer: i32,
    eer_image_pixels: i32,
    pos_limit: i32,
    num_electrons: *mut i32,
) -> i32 {
    let mut bit_pos: u32 = 0;
    let first_byte: u32;
    let mut pos: u32 = 0;
    let mut n_pix: u32 = 0;
    let mut n_electron: u32 = 0;
    let mut mis_aligned: u32 = 0;
    let _ = first_byte;

    if is_7bit_eer != 0 {
        loop {
            /* Fetch 32 bits and unpack up to 2 chunks of 7 + 4 bits.
            This is faster than unpack 7 and 4 bits sequentially.
            Since the size of buf is larger than the actual size of data read,
            it is always safe to read ahead. */

            let first_byte = bit_pos >> 3;
            let bit_offset_in_first_byte = bit_pos & 7; // 7 = 00000111 (same as % 8)
            let chunk = buf.add(first_byte as usize).cast::<u32>().read_unaligned()
                >> bit_offset_in_first_byte;

            let mut p = (chunk & 127) as u8; /* 127 = 01111111 */
            bit_pos += 7;
            n_pix += p as u32;
            if n_pix >= eer_image_pixels as u32 {
                break;
            }

            /* this should be rare. */
            if p == 127 {
                continue;
            }

            /* If it is a valid jump to an electron, save the position and sup-pixel bits */
            /* 15 = 00001111; See below for 0x0A */
            let mut sym = (((chunk >> 7) & 15) as u8) ^ 0x0A;
            bit_pos += 4;
            *positions.add(n_electron as usize) = n_pix as i32;
            *symbols.add(n_electron as usize) = sym;
            n_electron += 1;
            n_pix += 1;

            /* Repeat on second chunk */
            p = ((chunk >> 11) & 127) as u8;
            bit_pos += 7;
            n_pix += p as u32;
            if n_pix >= eer_image_pixels as u32 {
                break;
            }
            if p == 127 {
                continue;
            }

            sym = (((chunk >> 18) & 15) as u8) ^ 0x0A;
            bit_pos += 4;
            *positions.add(n_electron as usize) = n_pix as i32;
            *symbols.add(n_electron as usize) = sym;
            n_electron += 1;
            n_pix += 1;
        }
    } else {
        /* 8 bit code is untested with or without wasted bits */
        /* The array was oversized, so it is safe to go beyond the limit by two bytes. */
        while pos < pos_limit as u32 {
            if mis_aligned != 0 {
                let p1 = (*buf.add(pos as usize) >> 4) | (*buf.add(pos as usize + 1) << 4);
                let s1 = (*buf.add(pos as usize + 1) >> 4) ^ 0x0A;
                n_pix += p1 as u32;
                if n_pix >= eer_image_pixels as u32 {
                    break;
                }
                if p1 < 255 {
                    *positions.add(n_electron as usize) = n_pix as i32;
                    *symbols.add(n_electron as usize) = s1;
                    n_electron += 1;
                    n_pix += 1;
                    mis_aligned = 0;
                } else {
                    pos += 1;
                    continue;
                }
            }

            /* symbol is bit tricky. 0000YyXx; Y and X must be flipped. */
            let p1 = *buf.add(pos as usize);
            let s1 = (*buf.add(pos as usize + 1) & 0x0F) ^ 0x0A; // 0x0F = 00001111, 0x0A = 00001010

            let p2 = (*buf.add(pos as usize + 1) >> 4) | (*buf.add(pos as usize + 2) << 4);
            let s2 = (*buf.add(pos as usize + 2) >> 4) ^ 0x0A;

            // Note the order. Add p before checking the size and placing a new electron.
            n_pix += p1 as u32;
            if n_pix >= eer_image_pixels as u32 {
                break;
            }
            if p1 < 255 {
                *positions.add(n_electron as usize) = n_pix as i32;
                *symbols.add(n_electron as usize) = s1;
                n_electron += 1;
                n_pix += 1;
            } else {
                pos += 1;
                continue;
            }

            n_pix += p2 as u32;
            if n_pix >= eer_image_pixels as u32 {
                break;
            }
            if p2 < 255 {
                *positions.add(n_electron as usize) = n_pix as i32;
                *symbols.add(n_electron as usize) = s2;
                n_electron += 1;
                n_pix += 1;
            } else {
                pos += 2;
                mis_aligned = 1;
                continue;
            }
            pos += 3;
        }
    }

    *num_electrons = n_electron as i32;
    if n_pix != eer_image_pixels as u32 && S_IGNORE_BAD_EER_END.load(Ordering::SeqCst) == 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: TIFFReadSection - Final pixel position in EER image is not right ({} instead of {})\n.",
                n_pix as i32, eer_image_pixels
            ),
        );
        return 1;
    }
    0
}
/// C `convertEERpositions` (`iitif.c:1639`).
///
/// Convert the list of electron positions into counts in an image buffer with optional
/// gain normalization and antialiasing.  The source `#pragma omp parallel for`
/// directives select thread counts through `numOMPthreads`; this crate's
/// `num_omp_threads` reports a serial build, so the loops run in source order.
unsafe fn convert_eer_positions(
    in_file: *mut ImodImageFile,
    positions: *mut i32,
    symbols: *mut u8,
    num_electrons: i32,
    buf: *mut u8,
    xmin: i32,
    xmax: i32,
    ymin: i32,
    ymax: i32,
) {
    let mut filters = S_EER_FILTERS.lock().unwrap();
    let out_xsize = xmax + 1 - xmin;
    let out_ysize = ymax + 1 - ymin;
    let mut chip_xsize = (*in_file).nx;
    let mut chip_ysize = (*in_file).ny;
    let mut y_offset;
    if (*in_file).read_eer_as_super_res >= 0 {
        for _ind in 0..(*in_file).read_eer_as_super_res {
            chip_xsize /= 2;
            chip_ysize /= 2;
        }
    } else {
        for _ind in 0..-(*in_file).read_eer_as_super_res {
            chip_xsize *= 2;
            chip_ysize *= 2;
        }
    }
    let _ = chip_ysize;
    if S_EER_FLAGS.load(Ordering::SeqCst) & IIFLAG_ADD_TO_EER_SUM == 0 {
        core::ptr::write_bytes(
            buf,
            0,
            (out_xsize
                * out_ysize
                * if (*in_file).mode == MRC_MODE_BYTE {
                    1
                } else {
                    2
                }) as usize,
        );
    }
    y_offset = ((*in_file).ny - 1) - ymin;

    if (*in_file).antialias_eerfilter == 0 {
        /* Returning full-super-resolution image */

        /* The if test actually makes little difference,
        but using bit shifts helps when the size is standard */
        if (*in_file).read_eer_as_super_res == 2 {
            if chip_xsize == 4096 {
                for ind in 0..num_electrons as usize {
                    let x = ((((*positions.add(ind) & 4095) << 2) | (*symbols.add(ind) as i32 & 3))
                        - xmin) as i32;
                    let y = y_offset
                        - (((*positions.add(ind) >> 12) << 2)
                            | ((*symbols.add(ind) as i32 & 12) >> 2));
                    if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                        let cell = buf.add((x + y * out_xsize) as usize);
                        *cell = (*cell).wrapping_add(1);
                    }
                }
            } else {
                for ind in 0..num_electrons as usize {
                    let x = (((*positions.add(ind) % chip_xsize) << 2)
                        | (*symbols.add(ind) as i32 & 3))
                        - xmin;
                    let y = y_offset
                        - (((*positions.add(ind) / chip_xsize) << 2)
                            | ((*symbols.add(ind) as i32 & 12) >> 2));
                    if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                        let cell = buf.add((x + y * out_xsize) as usize);
                        *cell = (*cell).wrapping_add(1);
                    }
                }
            }
        } else if (*in_file).read_eer_as_super_res == 1 {
            /* Returning super-resolution 1 image */
            if chip_xsize == 4096 {
                for ind in 0..num_electrons as usize {
                    let x = (((*positions.add(ind) & 4095) << 1)
                        | ((*symbols.add(ind) as i32 & 2) >> 1))
                        - xmin;
                    let y = y_offset
                        - (((*positions.add(ind) >> 12) << 1)
                            | ((*symbols.add(ind) as i32 & 8) >> 3));
                    if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                        let cell = buf.add((x + y * out_xsize) as usize);
                        *cell = (*cell).wrapping_add(1);
                    }
                }
            } else {
                for ind in 0..num_electrons as usize {
                    let x = (((*positions.add(ind) % chip_xsize) << 1)
                        | ((*symbols.add(ind) as i32 & 2) >> 1))
                        - xmin;
                    let y = y_offset
                        - (((*positions.add(ind) / chip_xsize) << 1)
                            | ((*symbols.add(ind) as i32 & 8) >> 3));
                    if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                        let cell = buf.add((x + y * out_xsize) as usize);
                        *cell = (*cell).wrapping_add(1);
                    }
                }
            }
        } else {
            /* Returning image with no super-resolution */
            if chip_xsize == 4096 {
                for ind in 0..num_electrons as usize {
                    let x = (*positions.add(ind) & 4095) - xmin;
                    let y = y_offset - (*positions.add(ind) >> 12);
                    if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                        let cell = buf.add((x + y * out_xsize) as usize);
                        *cell = (*cell).wrapping_add(1);
                    }
                }
            } else {
                for ind in 0..num_electrons as usize {
                    let x = (*positions.add(ind) % chip_xsize) - xmin;
                    let y = y_offset - (*positions.add(ind) / chip_xsize);
                    if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                        let cell = buf.add((x + y * out_xsize) as usize);
                        *cell = (*cell).wrapping_add(1);
                    }
                }
            }
        }
    } else {
        /* COMPOSING A SCALED SHORT IMAGE WITH POSSIBLE ANTIALIASING AND NORMALIZATION */
        let sbuf = buf.cast::<i16>();
        let gain_reference = S_GAIN_REFERENCE.load(Ordering::SeqCst);

        /* Returning full super-resolution image with or without gain normalization */
        if (*in_file).read_eer_as_super_res == 2 {
            /* With gain normalization */
            if !gain_reference.is_null() {
                y_offset = (*in_file).ny - 1;
                let nx_gain = (*in_file).nx;
                let scale = (*in_file).eerkernel_scale as f32;
                for ind in 0..num_electrons as usize {
                    let x_gain =
                        ((*positions.add(ind) % chip_xsize) << 2) | (*symbols.add(ind) as i32 & 3);
                    let x = x_gain - xmin;
                    let y_gain = y_offset
                        - (((*positions.add(ind) / chip_xsize) << 2)
                            | ((*symbols.add(ind) as i32 & 12) >> 2));
                    let y = y_gain - ymin;
                    if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                        let cell = sbuf.add((x + y * out_xsize) as usize);
                        let value = ((scale
                            * *gain_reference.add((x_gain + y_gain * nx_gain) as usize))
                            as f64
                            + 0.5)
                            .floor() as i32;
                        *cell = (*cell).wrapping_add(value as i16);
                    }
                }
            } else {
                /* Without gain normalization */
                let gain_scale = (*in_file).eerkernel_scale;

                /* Variant for standard size, somewhat faster */
                if chip_xsize == 4096 {
                    for ind in 0..num_electrons as usize {
                        let x = (((*positions.add(ind) & 4095) << 2)
                            | (*symbols.add(ind) as i32 & 3))
                            - xmin;
                        let y = y_offset
                            - (((*positions.add(ind) >> 12) << 2)
                                | ((*symbols.add(ind) as i32 & 12) >> 2));
                        if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                            let cell = sbuf.add((x + y * out_xsize) as usize);
                            *cell = (*cell).wrapping_add(gain_scale as i16);
                        }
                    }
                } else {
                    /* Or handle a different size */
                    for ind in 0..num_electrons as usize {
                        let x = (((*positions.add(ind) % chip_xsize) << 2)
                            | (*symbols.add(ind) as i32 & 3))
                            - xmin;
                        let y = y_offset
                            - (((*positions.add(ind) / chip_xsize) << 2)
                                | ((*symbols.add(ind) as i32 & 12) >> 2));
                        if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                            let cell = sbuf.add((x + y * out_xsize) as usize);
                            *cell = (*cell).wrapping_add(gain_scale as i16);
                        }
                    }
                }
            }
        } else {
            /* ANTIALIASED REDUCTION with or without gain reference */
            /* Set up filters */
            let red_fac =
                (2.0_f64.powf((2 - (*in_file).read_eer_as_super_res) as f64) + 0.5).floor() as i32;
            let scale = (*in_file).eerkernel_scale as f32;
            let max_xout = out_xsize - 2;
            let max_yout = out_ysize - 2;
            for ix in 0..red_fac {
                for iy in 0..red_fac {
                    let k_ind = (ix + red_fac * iy) as usize;
                    filters.x_start[k_ind] = if ix < red_fac / 2 { -2 } else { -1 };
                    filters.y_start[k_ind] = if iy < red_fac / 2 { -2 } else { -1 };
                    let xcen = (ix as f32 + 0.5) / red_fac as f32;
                    let ycen = (iy as f32 + 0.5) / red_fac as f32;
                    let mut tsum = 0.0f64;
                    for oy in 0..4 {
                        let y_wgt = zoom_raw_filt_value(
                            filters.y_start[k_ind] as f32 + oy as f32 + 0.5 - ycen,
                        );
                        for ox in 0..4 {
                            let x_wgt = zoom_raw_filt_value(
                                filters.x_start[k_ind] as f32 + ox as f32 + 0.5 - xcen,
                            );
                            tsum += x_wgt * y_wgt;
                        }
                    }
                    for oy in 0..4 {
                        let y_wgt = zoom_raw_filt_value(
                            filters.y_start[k_ind] as f32 + oy as f32 + 0.5 - ycen,
                        );
                        for ox in 0..4 {
                            let x_wgt = zoom_raw_filt_value(
                                filters.x_start[k_ind] as f32 + ox as f32 + 0.5 - xcen,
                            );
                            let slot = k_ind * 16 + (ox + 4 * oy) as usize;
                            if !gain_reference.is_null() {
                                filters.all[slot] =
                                    ((scale as f64 * x_wgt * y_wgt / tsum) as f32).to_bits() as i32;
                            } else {
                                filters.all[slot] =
                                    ((scale as f64 * x_wgt * y_wgt / tsum) + 0.5).floor() as i32;
                            }
                        }
                    }
                }
            }

            // Compose image with no gain reference
            if gain_reference.is_null() {
                y_offset = (*in_file).ny * red_fac - 1;
                let gain_scale = (*in_file).eerkernel_scale;
                for ind in 0..num_electrons as usize {
                    let xsr =
                        ((*positions.add(ind) % chip_xsize) << 2) | (*symbols.add(ind) as i32 & 3);
                    let ysr = y_offset
                        - (((*positions.add(ind) / chip_xsize) << 2)
                            | ((*symbols.add(ind) as i32 & 12) >> 2));
                    let x = xsr / red_fac - xmin;
                    let y = ysr / red_fac - ymin;

                    /* Deposit packet within inner limits, or just add electron on edges */
                    if x >= 2 && x < max_xout && y >= 2 && y < max_yout {
                        let k_ind = ((xsr % red_fac) + red_fac * (ysr % red_fac)) as usize;
                        for iy in 0..4 {
                            let ybase = (y + iy + filters.y_start[k_ind]) * out_xsize;
                            let mut ix = x + filters.x_start[k_ind];
                            while ix < x + filters.x_start[k_ind] + 4 {
                                let cell = sbuf.add((ix + ybase) as usize);
                                let value = filters.all[k_ind * 16
                                    + (ix - x - filters.x_start[k_ind]) as usize
                                    + iy as usize * 4];
                                *cell = (*cell).wrapping_add(value as i16);
                                ix += 1;
                            }
                        }
                    } else if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                        let cell = sbuf.add((x + y * out_xsize) as usize);
                        *cell = (*cell).wrapping_add(gain_scale as i16);
                    }
                }
            } else {
                /* Compose gain normalized image: the most complex operation can use most
                threads */
                y_offset = (*in_file).ny * red_fac - 1;
                let nx_gain = red_fac * (*in_file).nx;
                // The source declares `refVal` outside the loop and assigns it only in
                // the interior branch, so the edge branch below reuses the last value.
                let mut ref_val: f32 = 0.0;
                for ind in 0..num_electrons as usize {
                    let xsr =
                        ((*positions.add(ind) % chip_xsize) << 2) | (*symbols.add(ind) as i32 & 3);
                    let ysr = y_offset
                        - (((*positions.add(ind) / chip_xsize) << 2)
                            | ((*symbols.add(ind) as i32 & 12) >> 2));
                    let x = xsr / red_fac - xmin;
                    let y = ysr / red_fac - ymin;

                    if x >= 2 && x < max_xout && y >= 2 && y < max_yout {
                        let k_ind = ((xsr % red_fac) + red_fac * (ysr % red_fac)) as usize;
                        ref_val = *gain_reference.add((xsr + ysr * nx_gain) as usize);
                        for iy in 0..4 {
                            let ybase = (y + iy + filters.y_start[k_ind]) * out_xsize;
                            let mut ix = x + filters.x_start[k_ind];
                            while ix < x + filters.x_start[k_ind] + 4 {
                                let cell = sbuf.add((ix + ybase) as usize);
                                let weight = f32::from_bits(
                                    filters.all[k_ind * 16
                                        + (ix - x - filters.x_start[k_ind]) as usize
                                        + iy as usize * 4]
                                        as u32,
                                );
                                let value = (weight * ref_val + 0.5f32).floor() as i32;
                                *cell = (*cell).wrapping_add(value as i16);
                                ix += 1;
                            }
                        }
                    } else if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                        let cell = sbuf.add((x + y * out_xsize) as usize);
                        let value = ((scale * ref_val) as f64 + 0.5).floor() as i32;
                        *cell = (*cell).wrapping_add(value as i16);
                    }
                }
            }
        }
    }
}
/// C `cleanupFromEER` (`iitif.c:2148`).
unsafe fn cleanup_from_eer(
    _tmp: *mut u8,
    _map: *mut u8,
    _positions: *mut i32,
    _symbols: *mut u8,
    _eer_buf: *mut u8,
) {
    S_EER_FLAGS.store(0, Ordering::SeqCst);
    *S_EER_FILTERS.lock().unwrap() = EerFilters::default();
}
/// C `copyLine` (`iitif.c:1954`).
///
/// Copy one line of data appropriately for the data type and conversion.
unsafe fn copy_line(
    bdata: *mut u8,
    obuf: *mut u8,
    xout: i32,
    convert: i32,
    pixsize: i32,
    type_: i32,
    format: i32,
    samples: i32,
    slope: f32,
    offset: f32,
    doscale: i32,
    unpack_4bits: i32,
    map: *mut u8,
    first_4bits_map: *mut u8,
    second_4bits_map: *mut u8,
    colormap: *mut u8,
) {
    let mut bdata = bdata;
    let mut obuf = obuf;
    let mut xout = xout;
    let mut usdata: *mut u16;
    let mut sdata: *mut i16;
    let mut uldata: *mut u32;
    let mut ldata: *mut i32;
    let mut fdata: *mut f32;
    let usmap = map.cast::<u16>();
    let mut usobuf = obuf.cast::<u16>();
    let mut fobuf = obuf.cast::<f32>();
    let to_short = if convert == MRSA_USHORT { 1 } else { 0 };
    let to_float = if convert == MRSA_FLOAT { 1 } else { 0 };
    let outmax = if to_short != 0 { 65535 } else { 255 };
    let signed_bytes = if type_ == IITYPE_BYTE { 1 } else { 0 };
    let mut i: i32;
    let mut j: i32;
    let mut ival: i32;
    let mut fpixel: f32;

    if convert != 0 || signed_bytes != 0 || unpack_4bits != 0 {
        /* Converted data */
        if pixsize == 1 {
            /* Converted from bytes: 4 bits */
            if unpack_4bits != 0 {
                i = 0;

                /* If there is an odd pixel at start, get it */
                if unpack_4bits > 1 {
                    if to_float != 0 {
                        *fobuf = *second_4bits_map.add(*bdata.add(i as usize) as usize) as f32;
                        fobuf = fobuf.add(1);
                        i += 1;
                    } else if to_short != 0 {
                        *usobuf = *usmap
                            .add(*second_4bits_map.add(*bdata.add(i as usize) as usize) as usize);
                        usobuf = usobuf.add(1);
                        i += 1;
                    } else if doscale != 0 {
                        *obuf = *map
                            .add(*second_4bits_map.add(*bdata.add(i as usize) as usize) as usize);
                        obuf = obuf.add(1);
                        i += 1;
                    } else {
                        *obuf = *second_4bits_map.add(*bdata.add(i as usize) as usize);
                        obuf = obuf.add(1);
                        i += 1;
                    }
                    xout -= 1;
                }

                /* Process pairs of pixels */
                if to_float != 0 {
                    while i < xout / 2 {
                        *fobuf = *first_4bits_map.add(*bdata.add(i as usize) as usize) as f32;
                        fobuf = fobuf.add(1);
                        *fobuf = *second_4bits_map.add(*bdata.add(i as usize) as usize) as f32;
                        fobuf = fobuf.add(1);
                        i += 1;
                    }
                } else if to_short != 0 {
                    while i < xout / 2 {
                        *usobuf = *usmap
                            .add(*first_4bits_map.add(*bdata.add(i as usize) as usize) as usize);
                        usobuf = usobuf.add(1);
                        *usobuf = *usmap
                            .add(*second_4bits_map.add(*bdata.add(i as usize) as usize) as usize);
                        usobuf = usobuf.add(1);
                        i += 1;
                    }
                } else if doscale != 0 {
                    while i < xout / 2 {
                        *obuf = *map
                            .add(*first_4bits_map.add(*bdata.add(i as usize) as usize) as usize);
                        obuf = obuf.add(1);
                        *obuf = *map
                            .add(*second_4bits_map.add(*bdata.add(i as usize) as usize) as usize);
                        obuf = obuf.add(1);
                        i += 1;
                    }
                } else {
                    while i < xout / 2 {
                        *obuf = *first_4bits_map.add(*bdata.add(i as usize) as usize);
                        obuf = obuf.add(1);
                        *obuf = *second_4bits_map.add(*bdata.add(i as usize) as usize);
                        obuf = obuf.add(1);
                        i += 1;
                    }
                }

                /* Then do odd pixel at end */
                if xout % 2 != 0 {
                    if to_float != 0 {
                        *fobuf = *first_4bits_map.add(*bdata.add(i as usize) as usize) as f32;
                        fobuf = fobuf.add(1);
                    } else if to_short != 0 {
                        *usobuf = *usmap
                            .add(*first_4bits_map.add(*bdata.add(i as usize) as usize) as usize);
                        usobuf = usobuf.add(1);
                    } else if doscale != 0 {
                        *obuf = *map
                            .add(*first_4bits_map.add(*bdata.add(i as usize) as usize) as usize);
                        obuf = obuf.add(1);
                    } else {
                        *obuf = *first_4bits_map.add(*bdata.add(i as usize) as usize);
                        obuf = obuf.add(1);
                    }
                }
            } else if format == IIFORMAT_COLORMAP {
                /* RGB in a colormap with conversion to floats, shorts or bytes */
                if to_float != 0 {
                    i = 0;
                    while i < xout {
                        fpixel = (0.3 * *colormap.add(*bdata as usize) as f64) as f32;
                        fpixel = (fpixel as f64
                            + 0.59 * *colormap.add(256 + *bdata as usize) as f64)
                            as f32;
                        fpixel = (fpixel as f64
                            + 0.11 * *colormap.add(512 + *bdata as usize) as f64)
                            as f32;
                        bdata = bdata.add(1);
                        *fobuf = fpixel;
                        fobuf = fobuf.add(1);
                        i += 1;
                    }
                } else if to_short != 0 {
                    i = 0;
                    while i < xout {
                        fpixel = (255. * 0.3 * *colormap.add(*bdata as usize) as f64) as f32;
                        fpixel = (fpixel as f64
                            + 255. * 0.59 * *colormap.add(256 + *bdata as usize) as f64)
                            as f32;
                        fpixel = (fpixel as f64
                            + 255. * 0.11 * *colormap.add(512 + *bdata as usize) as f64)
                            as f32;
                        bdata = bdata.add(1);
                        *usobuf = (fpixel + 0.5f32) as i32 as u16;
                        usobuf = usobuf.add(1);
                        i += 1;
                    }
                } else {
                    i = 0;
                    while i < xout {
                        fpixel = (0.3 * *colormap.add(*bdata as usize) as f64) as f32;
                        fpixel = (fpixel as f64
                            + 0.59 * *colormap.add(256 + *bdata as usize) as f64)
                            as f32;
                        fpixel = (fpixel as f64
                            + 0.11 * *colormap.add(512 + *bdata as usize) as f64)
                            as f32;
                        bdata = bdata.add(1);
                        *obuf = (fpixel + 0.5f32) as i32 as u8;
                        obuf = obuf.add(1);
                        i += 1;
                    }
                }
            } else if samples == 1 {
                /* Single-sample conversions of bytes to float, short, mapped or unmapped bytes */
                if to_float != 0 && signed_bytes != 0 {
                    i = 0;
                    while i < xout {
                        *fobuf = *map.add(*bdata as usize) as f32;
                        fobuf = fobuf.add(1);
                        bdata = bdata.add(1);
                        i += 1;
                    }
                } else if to_float != 0 {
                    i = 0;
                    while i < xout {
                        *fobuf = *bdata as f32;
                        fobuf = fobuf.add(1);
                        bdata = bdata.add(1);
                        i += 1;
                    }
                } else if to_short != 0 {
                    i = 0;
                    while i < xout {
                        *usobuf = *usmap.add(*bdata as usize);
                        usobuf = usobuf.add(1);
                        bdata = bdata.add(1);
                        i += 1;
                    }
                } else if doscale != 0 || signed_bytes != 0 {
                    i = 0;
                    while i < xout {
                        *obuf = *map.add(*bdata as usize);
                        obuf = obuf.add(1);
                        bdata = bdata.add(1);
                        i += 1;
                    }
                } else {
                    // Both buffers belong to the caller of this unsafe pixel
                    // conversion routine.  Model the contiguous byte transfer as
                    // slices instead of routing crate-owned work through libc.
                    core::slice::from_raw_parts_mut(obuf, xout as usize)
                        .copy_from_slice(core::slice::from_raw_parts(bdata, xout as usize));
                }
            } else {
                /* Interleaved sample conversions of bytes to float, short, mapped or unmapped
                bytes */
                if to_float != 0 && signed_bytes != 0 {
                    i = 0;
                    while i < xout {
                        *fobuf = *map.add(*bdata as usize) as f32;
                        fobuf = fobuf.add(1);
                        bdata = bdata.add(samples as usize);
                        i += 1;
                    }
                } else if to_float != 0 {
                    i = 0;
                    while i < xout {
                        *fobuf = *bdata as f32;
                        fobuf = fobuf.add(1);
                        bdata = bdata.add(samples as usize);
                        i += 1;
                    }
                } else if to_short != 0 {
                    i = 0;
                    while i < xout {
                        *usobuf = *usmap.add(*bdata as usize);
                        usobuf = usobuf.add(1);
                        bdata = bdata.add(samples as usize);
                        i += 1;
                    }
                } else if doscale != 0 || signed_bytes != 0 {
                    i = 0;
                    while i < xout {
                        *obuf = *map.add(*bdata as usize);
                        obuf = obuf.add(1);
                        bdata = bdata.add(samples as usize);
                        i += 1;
                    }
                } else {
                    i = 0;
                    while i < xout {
                        *obuf = *bdata;
                        obuf = obuf.add(1);
                        bdata = bdata.add(samples as usize);
                        i += 1;
                    }
                }
            }
        } else if pixsize == 2 {
            /* Integers converted */
            usdata = bdata.cast::<u16>();
            sdata = bdata.cast::<i16>();
            if samples == 1 {
                /* single-sample integer conversions to mapped short, float, unmapped short, or
                mapped byte */
                if to_short != 0 && !map.is_null() {
                    i = 0;
                    while i < xout {
                        *usobuf = *usmap.add(*usdata as usize);
                        usobuf = usobuf.add(1);
                        usdata = usdata.add(1);
                        i += 1;
                    }
                } else if to_float != 0 && type_ == IITYPE_SHORT {
                    i = 0;
                    while i < xout {
                        *fobuf = *sdata as f32;
                        fobuf = fobuf.add(1);
                        sdata = sdata.add(1);
                        i += 1;
                    }
                } else if to_float != 0 {
                    i = 0;
                    while i < xout {
                        *fobuf = *usdata as f32;
                        fobuf = fobuf.add(1);
                        usdata = usdata.add(1);
                        i += 1;
                    }
                } else if to_short != 0 {
                    i = 0;
                    while i < xout {
                        *usobuf = *usdata;
                        usobuf = usobuf.add(1);
                        usdata = usdata.add(1);
                        i += 1;
                    }
                } else {
                    i = 0;
                    while i < xout {
                        *obuf = *map.add(*usdata as usize);
                        obuf = obuf.add(1);
                        usdata = usdata.add(1);
                        i += 1;
                    }
                }
            } else {
                /* interleaved sample conversions to mapped short, float, unmapped short, or
                mapped byte */
                if to_short != 0 && !map.is_null() {
                    i = 0;
                    while i < xout {
                        *usobuf = *usmap.add(*usdata as usize);
                        usobuf = usobuf.add(1);
                        usdata = usdata.add(samples as usize);
                        i += 1;
                    }
                } else if to_float != 0 && type_ == IITYPE_SHORT {
                    i = 0;
                    while i < xout {
                        *fobuf = *sdata as f32;
                        fobuf = fobuf.add(1);
                        sdata = sdata.add(samples as usize);
                        i += 1;
                    }
                } else if to_float != 0 {
                    i = 0;
                    while i < xout {
                        *fobuf = *usdata as f32;
                        fobuf = fobuf.add(1);
                        usdata = usdata.add(samples as usize);
                        i += 1;
                    }
                } else if to_short != 0 {
                    i = 0;
                    while i < xout {
                        *usobuf = *usdata;
                        usobuf = usobuf.add(1);
                        usdata = usdata.add(samples as usize);
                        i += 1;
                    }
                } else {
                    i = 0;
                    while i < xout {
                        *obuf = *map.add(*usdata as usize);
                        obuf = obuf.add(1);
                        usdata = usdata.add(samples as usize);
                        i += 1;
                    }
                }
            }
        } else if type_ == IITYPE_INT {
            /* Long ints */
            ldata = bdata.cast::<i32>();
            if to_float != 0 {
                i = 0;
                while i < xout {
                    *fobuf = *ldata as f32;
                    fobuf = fobuf.add(1);
                    ldata = ldata.add(samples as usize);
                    i += 1;
                }
            } else {
                i = 0;
                while i < xout {
                    ival = (slope * (*ldata as f32) + offset) as i32;
                    if to_short != 0 {
                        *usobuf = 0.max(outmax.min(ival)) as u16;
                        usobuf = usobuf.add(1);
                    } else {
                        *obuf = 0.max(outmax.min(ival)) as u8;
                        obuf = obuf.add(1);
                    }
                    ldata = ldata.add(samples as usize);
                    i += 1;
                }
            }
        } else if type_ == IITYPE_UINT {
            /* Unsigned Long ints */
            uldata = bdata.cast::<u32>();
            if to_float != 0 {
                i = 0;
                while i < xout {
                    *fobuf = *uldata as f32;
                    fobuf = fobuf.add(1);
                    uldata = uldata.add(samples as usize);
                    i += 1;
                }
            } else {
                i = 0;
                while i < xout {
                    ival = (slope * (*uldata as f32) + offset) as i32;
                    if to_short != 0 {
                        *usobuf = 0.max(outmax.min(ival)) as u16;
                        usobuf = usobuf.add(1);
                    } else {
                        *obuf = 0.max(outmax.min(ival)) as u8;
                        obuf = obuf.add(1);
                    }
                    uldata = uldata.add(samples as usize);
                    i += 1;
                }
            }
        } else if format == IIFORMAT_RGB {
            /* RGB conversions to float, mapped short, mapped byte, unmapped short or byte */
            if to_float != 0 {
                i = 0;
                while i < xout {
                    fpixel = (0.3 * *bdata.add(0) as f64) as f32;
                    fpixel = (fpixel as f64 + 0.59 * *bdata.add(1) as f64) as f32;
                    fpixel = (fpixel as f64 + 0.11 * *bdata.add(2) as f64) as f32;
                    bdata = bdata.add(pixsize as usize);
                    *fobuf = fpixel;
                    fobuf = fobuf.add(1);
                    i += 1;
                }
            } else if doscale != 0 && to_short != 0 {
                i = 0;
                while i < xout {
                    fpixel = (0.3 * *bdata.add(0) as f64) as f32;
                    fpixel = (fpixel as f64 + 0.59 * *bdata.add(1) as f64) as f32;
                    fpixel = (fpixel as f64 + 0.11 * *bdata.add(2) as f64) as f32;
                    bdata = bdata.add(pixsize as usize);
                    *usobuf = *usmap.add((fpixel + 0.499f32) as i32 as usize);
                    usobuf = usobuf.add(1);
                    i += 1;
                }
            } else if doscale != 0 {
                i = 0;
                while i < xout {
                    fpixel = (0.3 * *bdata.add(0) as f64) as f32;
                    fpixel = (fpixel as f64 + 0.59 * *bdata.add(1) as f64) as f32;
                    fpixel = (fpixel as f64 + 0.11 * *bdata.add(2) as f64) as f32;
                    bdata = bdata.add(pixsize as usize);
                    *obuf = *map.add((fpixel + 0.499f32) as i32 as usize);
                    obuf = obuf.add(1);
                    i += 1;
                }
            } else if to_short != 0 {
                i = 0;
                while i < xout {
                    fpixel = (255. * 0.3 * *bdata.add(0) as f64) as f32;
                    fpixel = (fpixel as f64 + 255. * 0.59 * *bdata.add(1) as f64) as f32;
                    fpixel = (fpixel as f64 + 255. * 0.11 * *bdata.add(2) as f64) as f32;
                    bdata = bdata.add(pixsize as usize);
                    *usobuf = (fpixel + 0.5f32) as i32 as u16;
                    usobuf = usobuf.add(1);
                    i += 1;
                }
            } else {
                i = 0;
                while i < xout {
                    fpixel = (0.3 * *bdata.add(0) as f64) as f32;
                    fpixel = (fpixel as f64 + 0.59 * *bdata.add(1) as f64) as f32;
                    fpixel = (fpixel as f64 + 0.11 * *bdata.add(2) as f64) as f32;
                    bdata = bdata.add(pixsize as usize);
                    *obuf = (fpixel + 0.5f32) as i32 as u8;
                    obuf = obuf.add(1);
                    i += 1;
                }
            }
        } else {
            /* Floats with interleaved samples or converted to short or byte */
            fdata = bdata.cast::<f32>();
            if to_float != 0 {
                i = 0;
                while i < xout {
                    *fobuf = *fdata;
                    fobuf = fobuf.add(1);
                    fdata = fdata.add(samples as usize);
                    i += 1;
                }
            } else if to_short != 0 {
                i = 0;
                while i < xout {
                    ival = (slope * (*fdata) + offset) as i32;
                    *usobuf = 0.max(outmax.min(ival)) as u16;
                    usobuf = usobuf.add(1);
                    fdata = fdata.add(samples as usize);
                    i += 1;
                }
            } else {
                i = 0;
                while i < xout {
                    ival = (slope * (*fdata) + offset) as i32;
                    *obuf = 0.max(outmax.min(ival)) as u8;
                    obuf = obuf.add(1);
                    fdata = fdata.add(samples as usize);
                    i += 1;
                }
            }
        }
    } else {
        /* Non-converted data */
        if samples == 1 {
            /* RGB with extra samples - skip them */
            if format == IIFORMAT_RGB && pixsize > 3 {
                i = 0;
                while i < xout {
                    j = 0;
                    while j < 3 {
                        *obuf = *bdata;
                        obuf = obuf.add(1);
                        bdata = bdata.add(1);
                        j += 1;
                    }
                    bdata = bdata.add((pixsize - 3) as usize);
                    i += 1;
                }

            /* Colormap lookup */
            } else if !colormap.is_null() {
                i = 0;
                while i < xout {
                    *obuf = *colormap.add(*bdata as usize);
                    obuf = obuf.add(1);
                    *obuf = *colormap.add(256 + *bdata as usize);
                    obuf = obuf.add(1);
                    *obuf = *colormap.add(512 + *bdata as usize);
                    obuf = obuf.add(1);
                    bdata = bdata.add(1);
                    i += 1;
                }

            /* Straight copy */
            } else {
                let byte_count = (xout * pixsize) as usize;
                core::slice::from_raw_parts_mut(obuf, byte_count)
                    .copy_from_slice(core::slice::from_raw_parts(bdata, byte_count));
            }
        } else {
            /* Multiple samples (planes) interleaved - skip the other samples */
            i = 0;
            while i < xout {
                j = 0;
                while j < pixsize {
                    *obuf = *bdata;
                    obuf = obuf.add(1);
                    bdata = bdata.add(1);
                    j += 1;
                }
                bdata = bdata.add(((samples - 1) * pixsize) as usize);
                i += 1;
            }
        }
    }
}
/// C `tiffReadSectionByte` (`iitif.c:2294`).
pub unsafe fn tiff_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    read_section(in_file, buf, in_section, MRSA_BYTE)
}
/// C `tiffReadSectionUShort` (`iitif.c:2299`).
pub unsafe fn tiff_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    read_section(in_file, buf, in_section, MRSA_USHORT)
}
/// C `tiffReadSectionFloat` (`iitif.c:2304`).
pub unsafe fn tiff_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    read_section(in_file, buf, in_section, MRSA_FLOAT)
}
/// C `tiffReadSection` (`iitif.c:2309`).
pub unsafe fn tiff_read_section(in_file: *mut ImodImageFile, buf: *mut u8, in_section: i32) -> i32 {
    read_section(in_file, buf, in_section, MRSA_NOPROC)
}
/// C `bufReadProc` (`iitif.c:2329`).
unsafe extern "C" fn buf_read_proc(fd: *mut c_void, buf: *mut c_void, size: isize) -> isize {
    let fd = fd as usize;
    let mut buffers = S_PARALLEL_BUFFERS.lock().unwrap();
    if fd >= MAX_TIFF_THREADS
        || size < 0
        || buffers.cur_buf_ind[fd].saturating_add(size as u64)
            > S_FILE_BUF_SIZE.load(Ordering::SeqCst) as u64
    {
        // This is a libtiff callback, so errno is part of its C ABI failure
        // contract rather than crate-owned error handling.
        *libc::__errno_location() = libc::EINVAL;
        return -1;
    }
    let start = buffers.cur_buf_ind[fd] as usize;
    let end = start + size as usize;
    let source = &buffers.file_buf[fd][start..end];
    if source.is_empty() {
        // `from_raw_parts_mut` requires a non-null pointer even for an empty
        // slice, while libtiff permits a null buffer for a zero-byte request.
    } else if buf.is_null() {
        *libc::__errno_location() = libc::EINVAL;
        return -1;
    } else {
        core::slice::from_raw_parts_mut(buf.cast::<u8>(), source.len()).copy_from_slice(source);
    }
    buffers.cur_buf_ind[fd] += size as u64;
    buffers.max_buf_ind[fd] = buffers.max_buf_ind[fd].max(buffers.cur_buf_ind[fd]);
    size
}
/// C `bufWriteProc` (`iitif.c:2343`).
unsafe extern "C" fn buf_write_proc(fd: *mut c_void, buf: *mut c_void, size: isize) -> isize {
    let fd = fd as usize;
    let mut buffers = S_PARALLEL_BUFFERS.lock().unwrap();
    if fd >= MAX_TIFF_THREADS
        || size < 0
        || buffers.cur_buf_ind[fd].saturating_add(size as u64)
            >= S_FILE_BUF_SIZE.load(Ordering::SeqCst) as u64
    {
        *libc::__errno_location() = libc::EINVAL;
        return -1;
    }
    let start = buffers.cur_buf_ind[fd] as usize;
    let end = start + size as usize;
    let destination = &mut buffers.file_buf[fd][start..end];
    if destination.is_empty() {
        // See `buf_read_proc`: a zero-byte libtiff request need not supply a
        // valid buffer pointer.
    } else if buf.is_null() {
        *libc::__errno_location() = libc::EINVAL;
        return -1;
    } else {
        destination.copy_from_slice(core::slice::from_raw_parts(
            buf.cast::<u8>(),
            destination.len(),
        ));
    }
    buffers.cur_buf_ind[fd] += size as u64;
    buffers.max_buf_ind[fd] = buffers.max_buf_ind[fd].max(buffers.cur_buf_ind[fd]);
    size
}
/// C `bufSeekProc` (`iitif.c:2357`).
unsafe extern "C" fn buf_seek_proc(fd: *mut c_void, off: u64, whence: i32) -> u64 {
    let fd = fd as usize;
    if fd >= MAX_TIFF_THREADS {
        *libc::__errno_location() = libc::EINVAL;
        return u64::MAX;
    }
    let mut buffers = S_PARALLEL_BUFFERS.lock().unwrap();
    let signed = off as i64;
    let new_pos = match whence {
        0 if signed >= 0 => signed as u64,
        1 if signed >= 0 || buffers.cur_buf_ind[fd] >= (-signed) as u64 => {
            buffers.cur_buf_ind[fd].wrapping_add_signed(signed)
        }
        2 => buffers.max_buf_ind[fd],
        _ => {
            *libc::__errno_location() = libc::EINVAL;
            return u64::MAX;
        }
    };
    if new_pos >= S_FILE_BUF_SIZE.load(Ordering::SeqCst) as u64 {
        *libc::__errno_location() = libc::EINVAL;
        return u64::MAX;
    }
    buffers.cur_buf_ind[fd] = new_pos;
    buffers.max_buf_ind[fd] = buffers.max_buf_ind[fd].max(new_pos);
    new_pos
}
/// C `bufCloseProc` (`iitif.c:2393`).
unsafe extern "C" fn buf_close_proc(_fd: *mut c_void) -> i32 {
    0
}
/// C `bufSizeProc` (`iitif.c:2398`).
unsafe extern "C" fn buf_size_proc(fd: *mut c_void) -> u64 {
    S_PARALLEL_BUFFERS.lock().unwrap().max_buf_ind[fd as usize]
}
/// C `tiffOpenNew` (`iitif.c:2406`).
pub unsafe fn tiff_open_new(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() {
        return IIERR_BAD_CALL;
    }
    let Some(filename) = (&(*in_file).filename).as_deref() else {
        return IIERR_BAD_CALL;
    };
    augment_libtiff_with_custom_tags();
    let (version, _) = tiff_version();
    let mut pixel_size = 1;
    if matches!((*in_file).type_, IITYPE_SHORT | IITYPE_USHORT) {
        pixel_size = 2;
    }
    if matches!((*in_file).type_, IITYPE_INT | IITYPE_UINT | IITYPE_FLOAT) {
        pixel_size = 4;
    }
    if (*in_file).format == IIFORMAT_RGB {
        pixel_size *= 3;
    }
    let use_w8 = version > 3
        && ((((*in_file).nx as f64 * (*in_file).ny as f64)
            * (*in_file).nz as f64
            * pixel_size as f64)
            > 4.0e9
            || make_all_big_tiff() != 0);
    // The libtiff line: the name and the mode become C strings here.
    let mode: &[u8] = if use_w8 { b"w8\0" } else { b"w\0" };
    let mut name = filename.as_bytes().to_vec();
    name.push(0);
    let tif = if S_SETTING_UP_PARALLEL.load(Ordering::SeqCst) <= 0 {
        TIFFOpen(name.as_ptr().cast(), mode.as_ptr().cast())
    } else {
        let file_number = (S_SETTING_UP_PARALLEL.load(Ordering::SeqCst) - 1) as usize;
        TIFFClientOpen(
            name.as_ptr().cast(),
            mode.as_ptr().cast(),
            file_number as *mut c_void,
            Some(buf_read_proc),
            Some(buf_write_proc),
            Some(buf_seek_proc),
            Some(buf_close_proc),
            Some(buf_size_proc),
            None,
            None,
        )
    };
    if tif.is_null() {
        return IIERR_IO_ERROR;
    }
    (*in_file).backend_handle = tif.cast();
    (*in_file).fp = Some(ImodFile::Token(tif as usize));
    (*in_file).state = IISTATE_READY;
    (*in_file).clean_up = Some(tiff_delete_callback);
    (*in_file).close = Some(tiff_close_callback);
    (*in_file).fill_mrc_header = Some(tiff_fill_mrc_header_callback);
    (*in_file).sync_from_mrc_header = Some(tiff_sync_from_mrc_header_callback);
    (*in_file).write_section = Some(ii_tiff_write_section);
    (*in_file).write_section_float = Some(ii_tiff_write_section_float);
    0
}
/// C `tiffWriteSection` (`iitif.c:2456`).
pub fn tiff_write_section(
    in_file: &mut ImodImageFile,
    buf: &mut [u8],
    compression: i32,
    inverted: i32,
    resolution: i32,
    quality: i32,
) -> i32 {
    if in_file.backend_handle.is_null() {
        return IIERR_BAD_CALL;
    }
    let pixel_bytes = match in_file.format {
        IIFORMAT_RGB => 3usize,
        _ => match in_file.type_ {
            IITYPE_UBYTE | IITYPE_BYTE => 1,
            IITYPE_USHORT | IITYPE_SHORT => 2,
            IITYPE_UINT | IITYPE_INT | IITYPE_FLOAT => 4,
            _ => return IIERR_NO_SUPPORT,
        },
    };
    let Some(required) = usize::try_from(in_file.nx)
        .ok()
        .and_then(|nx| {
            usize::try_from(in_file.ny)
                .ok()
                .and_then(|ny| nx.checked_mul(ny))
        })
        .and_then(|pixels| pixels.checked_mul(pixel_bytes))
    else {
        return IIERR_BAD_CALL;
    };
    if buf.len() < required {
        return IIERR_BAD_CALL;
    }
    let mut rows = 0;
    let mut number = 0;
    let mut tile_x = 0;
    let error = unsafe {
        tiff_write_setup(
            in_file,
            compression,
            inverted,
            resolution,
            quality,
            &mut rows,
            &mut number,
            &mut tile_x,
        )
    };
    if error != 0 {
        return error;
    }
    let mut state = *S_STRIP_TILE_STATE.lock().unwrap();
    for strip in 0..state.num_strips {
        let lines = state.rows_per_strip.min(in_file.ny - state.lines_done);
        let source_offset = if inverted != 0 {
            strip as usize * state.strip_bytes as usize
        } else {
            (in_file.ny - (state.lines_done + lines)) as usize * state.line_bytes as usize
        };
        let Some(source) = buf.get_mut(source_offset..) else {
            return IIERR_BAD_CALL;
        };
        let error = unsafe { tiff_write_strip(in_file, strip, source.as_mut_ptr().cast()) };
        if error != 0 {
            return error;
        }
        state.lines_done += lines;
    }
    unsafe { tiff_write_finish(in_file) };
    0
}
/// C `tiffWriteSetup` (`iitif.c:2490`).
pub unsafe fn tiff_write_setup(
    in_file: *mut ImodImageFile,
    compression: i32,
    inverted: i32,
    resolution: i32,
    quality: i32,
    out_rows: *mut i32,
    out_num: *mut i32,
    tile_size_x: *mut i32,
) -> i32 {
    let mut tmp_buf = S_TMP_BUF.lock().unwrap();
    if in_file.is_null() || (*in_file).backend_handle.is_null() {
        return IIERR_BAD_CALL;
    }
    if (*in_file).format != IIFORMAT_RGB
        && ((*in_file).format != IIFORMAT_LUMINANCE
            || !matches!(
                (*in_file).type_,
                IITYPE_UBYTE | IITYPE_BYTE | IITYPE_USHORT | IITYPE_SHORT | IITYPE_FLOAT
            ))
    {
        return IIERR_NO_SUPPORT;
    }
    let tif = (*in_file).backend_handle.cast::<Tiff>();
    if (*in_file).state == IISTATE_BUSY {
        TIFFWriteDirectory(tif);
    }
    (*in_file).state = IISTATE_READY;
    (*in_file).tiff_compression = compression;
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, (*in_file).nx as u32);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, (*in_file).ny as u32);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, compression);
    TIFFSetField(
        tif,
        TIFFTAG_RESOLUTIONUNIT,
        if resolution == 0 {
            RESUNIT_NONE
        } else if resolution < 0 {
            RESUNIT_INCH
        } else {
            RESUNIT_CENTIMETER
        },
    );
    if resolution != 0 {
        TIFFSetField(tif, TIFFTAG_XRESOLUTION, (resolution as f64).abs());
        TIFFSetField(tif, TIFFTAG_YRESOLUTION, (resolution as f64).abs());
        let mut dm_pixel = if resolution > 0 { 1.0e4 } else { -2.54e4 } / resolution as f64;
        let mut nano = 0;
        if dm_pixel < 0.01 {
            dm_pixel *= 1000.;
            nano = 1;
        }
        let mut suppressed = 0;
        if S_OLD_ERR_HANDLER.load(Ordering::SeqCst).is_null() {
            tiff_suppress_errors();
            suppressed = 1;
        }
        TIFFSetField(tif, TIFFTAG_DM_SCALE_0, dm_pixel);
        TIFFSetField(tif, TIFFTAG_DM_SCALE_0 + 1, dm_pixel);
        TIFFSetField(tif, TIFFTAG_DM_ORIGIN_0, 0.0_f64);
        TIFFSetField(tif, TIFFTAG_DM_ORIGIN_0 + 1, 0.0_f64);
        TIFFSetField(tif, TIFFTAG_DM_UINFO_POWER_0, 1);
        TIFFSetField(tif, TIFFTAG_DM_UINFO_POWER_0 + 1, 1);
        // Below the libtiff line: an ASCII tag value is a C string.
        let unit: &[u8] = if nano != 0 {
            b"nanometer\0"
        } else {
            b"micrometer\0"
        };
        TIFFSetField(tif, TIFFTAG_DM_UINFO_UNIT_0, unit.as_ptr().cast::<c_char>());
        TIFFSetField(
            tif,
            TIFFTAG_DM_UINFO_UNIT_0 + 1,
            unit.as_ptr().cast::<c_char>(),
        );
        if suppressed != 0 {
            tiff_restore_errors();
        }
    }
    if quality >= 0 && compression == IICOMPRESSION_JPEG {
        TIFFSetField(tif, TIFFTAG_JPEGQUALITY, quality.min(100));
    }
    if quality > 0 && compression == IICOMPRESSION_ZIP {
        TIFFSetField(tif, TIFFTAG_ZIPQUALITY, quality.min(9));
    }
    let (samples, bits, photometric, sample_format) = if (*in_file).format == IIFORMAT_RGB {
        (3, 8, PHOTOMETRIC_RGB, SAMPLEFORMAT_UINT)
    } else {
        match (*in_file).type_ {
            IITYPE_BYTE => (1, 8, PHOTOMETRIC_MINISBLACK, SAMPLEFORMAT_INT),
            IITYPE_UBYTE => (1, 8, PHOTOMETRIC_MINISBLACK, SAMPLEFORMAT_UINT),
            IITYPE_SHORT => (1, 16, PHOTOMETRIC_MINISBLACK, SAMPLEFORMAT_INT),
            IITYPE_USHORT => (1, 16, PHOTOMETRIC_MINISBLACK, SAMPLEFORMAT_UINT),
            IITYPE_FLOAT => (1, 32, PHOTOMETRIC_MINISBLACK, SAMPLEFORMAT_IEEEFP),
            _ => return IIERR_NO_SUPPORT,
        }
    };
    if (*in_file).format == IIFORMAT_RGB {
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits, bits, bits);
    } else {
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sample_format);
        if (*in_file).amax > (*in_file).amin {
            constrain_and_store_min_max(in_file);
        }
    }
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, photometric);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples);
    {
        let mut description = S_DESCRIPTION.lock().unwrap();
        if let Some(text) = description.as_deref() {
            if S_SETTING_UP_PARALLEL.load(Ordering::SeqCst) <= 0 {
                // Below the libtiff line: this staging buffer owns the sole
                // C terminator required by the foreign API.
                TIFFSetField(
                    tif,
                    TIFFTAG_IMAGE_DESCRIPTION,
                    text.as_ptr().cast::<c_char>(),
                );
            }
        }
        if S_SETTING_UP_PARALLEL.load(Ordering::SeqCst) == 0 {
            *description = None;
        }
    }
    // Do all layout arithmetic on a local copy.  In particular, do not hold
    // the state mutex while calling libtiff, whose error paths may reenter us.
    let mut state = *S_STRIP_TILE_STATE.lock().unwrap();
    state.pix_size = samples * bits / 8;
    if !tile_size_x.is_null() && *tile_size_x != 0 {
        let mut rows = *out_rows;
        let mut xtiles = 0;
        let mut ytiles = 0;
        ii_best_tile_size((*in_file).nx, &mut *tile_size_x, &mut xtiles, 16);
        ii_best_tile_size((*in_file).ny, &mut rows, &mut ytiles, 16);
        state.rows_per_strip = rows;
        state.num_x_tiles = xtiles;
        state.num_strips = ytiles;
        state.x_tile_size = *tile_size_x;
        state.line_bytes = state.pix_size * state.x_tile_size;
        TIFFSetField(tif, TIFFTAG_TILEWIDTH, state.x_tile_size as u32);
        TIFFSetField(tif, TIFFTAG_TILELENGTH, state.rows_per_strip as u32);
    } else {
        state.x_tile_size = 0;
        state.line_bytes = state.pix_size * (*in_file).nx;
        let mut target = if compression != IICOMPRESSION_NONE {
            16384
        } else {
            8192
        };
        if (*in_file).ny > 4096 {
            target = (1 + (*in_file).ny / 4096) * state.line_bytes;
        }
        if S_SETTING_UP_PARALLEL.load(Ordering::SeqCst) <= 0 {
            state.rows_per_strip = ((target + state.line_bytes / 2) / state.line_bytes).max(1);
        }
        if compression == IICOMPRESSION_JPEG && state.rows_per_strip % 8 != 0 {
            state.rows_per_strip = if state.rows_per_strip < 5 || state.rows_per_strip % 8 > 4 {
                8 * ((state.rows_per_strip + 7) / 8)
            } else {
                8 * (state.rows_per_strip / 8)
            };
        }
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, state.rows_per_strip as u32);
    }
    state.num_strips = ((*in_file).ny + state.rows_per_strip - 1) / state.rows_per_strip;
    state.strip_bytes = state.rows_per_strip * state.line_bytes;
    if !out_rows.is_null() {
        *out_rows = state.rows_per_strip;
    }
    if !out_num.is_null() {
        *out_num = state.num_strips;
    }
    // TIFF DateTime is a 19-byte local-time ASCII field.  The terminating NUL
    // exists only for this immediate libtiff call; all time storage stays Rust-native.
    let datetime = format!("{}\0", Local::now().format("%Y:%m:%d %H:%M:%S"));
    TIFFSetField(tif, TIFFTAG_DATETIME, datetime.as_ptr().cast::<c_char>());
    *tmp_buf = Vec::new();
    if inverted == 0 || state.x_tile_size != 0 {
        let Ok(buffer_size) = usize::try_from(state.strip_bytes) else {
            return IIERR_MEMORY_ERR;
        };
        if tmp_buf.try_reserve_exact(buffer_size).is_err() {
            return IIERR_MEMORY_ERR;
        }
        tmp_buf.resize(buffer_size, 0);
    }
    state.lines_done = 0;
    state.already_inverted = inverted;
    *S_STRIP_TILE_STATE.lock().unwrap() = state;
    0
}
/// C `tiffWriteStrip` (`iitif.c:2660`).
pub unsafe fn tiff_write_strip(in_file: *mut ImodImageFile, strip: i32, buf: *mut c_void) -> i32 {
    let mut tmp_buf = S_TMP_BUF.lock().unwrap();
    if in_file.is_null() || (*in_file).backend_handle.is_null() || buf.is_null() {
        return IIERR_BAD_CALL;
    }
    let state = *S_STRIP_TILE_STATE.lock().unwrap();
    let lines = state.rows_per_strip.min((*in_file).ny - state.lines_done);
    b3d_shift_bytes(
        core::slice::from_raw_parts_mut(buf.cast::<u8>(), (state.line_bytes * lines) as usize),
        state.line_bytes,
        lines,
        1,
        if (*in_file).type_ == IITYPE_BYTE {
            1
        } else {
            0
        },
    );
    if state.x_tile_size != 0 {
        for x_tile in 0..state.num_x_tiles {
            let x_offset = x_tile * state.x_tile_size * state.pix_size;
            let num_bytes = (state
                .x_tile_size
                .min((*in_file).nx - x_tile * state.x_tile_size))
                * state.pix_size;
            for line in 0..lines {
                let source_line = if state.already_inverted != 0 {
                    line
                } else {
                    lines - line - 1
                };
                let source = core::slice::from_raw_parts(
                    buf.cast::<u8>()
                        .add((source_line * (*in_file).nx * state.pix_size + x_offset) as usize),
                    num_bytes as usize,
                );
                let destination_start = (line * state.line_bytes) as usize;
                tmp_buf[destination_start..destination_start + num_bytes as usize]
                    .copy_from_slice(source);
            }
            if TIFFWriteEncodedTile(
                (*in_file).backend_handle.cast(),
                (x_tile + strip * state.num_x_tiles) as u32,
                tmp_buf.as_mut_ptr().cast(),
                state.strip_bytes as isize,
            ) < 0
            {
                *tmp_buf = Vec::new();
                return IIERR_IO_ERROR;
            }
        }
        b3d_shift_bytes(
            core::slice::from_raw_parts_mut(buf.cast::<u8>(), (state.line_bytes * lines) as usize),
            state.line_bytes,
            lines,
            -1,
            if (*in_file).type_ == IITYPE_BYTE {
                1
            } else {
                0
            },
        );
        S_STRIP_TILE_STATE.lock().unwrap().lines_done += lines;
        return 0;
    }
    let output = if state.already_inverted != 0 {
        buf.cast::<u8>()
    } else {
        for line in 0..lines as usize {
            let source = core::slice::from_raw_parts(
                buf.cast::<u8>()
                    .add((lines as usize - line - 1) * state.line_bytes as usize),
                state.line_bytes as usize,
            );
            let destination_start = line * state.line_bytes as usize;
            tmp_buf[destination_start..destination_start + state.line_bytes as usize]
                .copy_from_slice(source);
        }
        tmp_buf.as_mut_ptr()
    };
    if TIFFWriteEncodedStrip(
        (*in_file).backend_handle.cast(),
        strip as u32,
        output.cast(),
        (state.line_bytes * lines) as isize,
    ) < 0
    {
        if state.already_inverted == 0 {
            *tmp_buf = Vec::new();
        }
        return IIERR_IO_ERROR;
    }
    b3d_shift_bytes(
        core::slice::from_raw_parts_mut(buf.cast::<u8>(), (state.line_bytes * lines) as usize),
        state.line_bytes,
        lines,
        -1,
        if (*in_file).type_ == IITYPE_BYTE {
            1
        } else {
            0
        },
    );
    S_STRIP_TILE_STATE.lock().unwrap().lines_done += lines;
    0
}
/// C `tiffWriteFinish` (`iitif.c:2714`).
pub unsafe fn tiff_write_finish(in_file: *mut ImodImageFile) {
    if in_file.is_null() {
        return;
    }
    (*in_file).state = IISTATE_BUSY;
    let mut tmp_buf = S_TMP_BUF.lock().unwrap();
    let already_inverted = S_STRIP_TILE_STATE.lock().unwrap().already_inverted;
    if already_inverted == 0 && !tmp_buf.is_empty() {
        *tmp_buf = Vec::new();
    }
}
/// C `iiTiffWriteSection` (`iitif.c:2721`).
pub unsafe fn ii_tiff_write_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    if in_file.is_null() || buf.is_null() {
        return IIERR_BAD_CALL;
    }
    let file = &mut *in_file;
    let pixel_bytes = if file.format == IIFORMAT_RGB {
        3
    } else if matches!(file.type_, IITYPE_UBYTE | IITYPE_BYTE) {
        1
    } else if matches!(file.type_, IITYPE_USHORT | IITYPE_SHORT) {
        2
    } else {
        4
    };
    let Some(byte_count) = usize::try_from(file.nx)
        .ok()
        .and_then(|nx| {
            usize::try_from(file.ny)
                .ok()
                .and_then(|ny| nx.checked_mul(ny))
        })
        .and_then(|pixels| pixels.checked_mul(pixel_bytes))
    else {
        return IIERR_BAD_CALL;
    };
    // The stored image callback is the ABI boundary; all native section writing
    // below receives a bounded Rust slice.
    tiff_write_section_any(
        file,
        core::slice::from_raw_parts_mut(buf, byte_count),
        in_section,
        0,
    )
}
/// C `iiTiffWriteSectionFloat` (`iitif.c:2726`).
pub unsafe fn ii_tiff_write_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    if in_file.is_null() || buf.is_null() {
        return IIERR_BAD_CALL;
    }
    let file = &mut *in_file;
    let pixel_bytes = if file.format == IIFORMAT_RGB {
        3
    } else if matches!(file.type_, IITYPE_UBYTE | IITYPE_BYTE) {
        1
    } else if matches!(file.type_, IITYPE_USHORT | IITYPE_SHORT) {
        2
    } else {
        4
    };
    let Some(byte_count) = usize::try_from(file.nx)
        .ok()
        .and_then(|nx| {
            usize::try_from(file.ny)
                .ok()
                .and_then(|ny| nx.checked_mul(ny))
        })
        .and_then(|pixels| pixels.checked_mul(pixel_bytes))
    else {
        return IIERR_BAD_CALL;
    };
    tiff_write_section_any(
        file,
        core::slice::from_raw_parts_mut(buf, byte_count),
        in_section,
        1,
    )
}
/// C `tiffAddDescription` (`iitif.c:2732`).
pub fn tiff_add_description(text: Option<&[u8]>) {
    let mut description = S_DESCRIPTION.lock().unwrap();
    *description = None;
    if let Some(text) = text {
        // `strdup` copies through the first NUL, and the stored copy keeps it
        // because `TIFFSetField` reads a C string.
        let end = text
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(text.len());
        let mut copy = text[..end].to_vec();
        copy.push(0);
        *description = Some(copy);
    }
}
/// C `tiffWriteSectionAny` (`iitif.c:2738`).
fn tiff_write_section_any(
    in_file: &mut ImodImageFile,
    buf: &mut [u8],
    in_section: i32,
    if_float: i32,
) -> i32 {
    if in_file.pad_left != 0 || in_file.pad_right != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: tiffWriteSectionAny - Cannot write from a subset of an array\n"),
        );
        return -1;
    }
    if in_section != in_file.last_written_z + 1 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: tiffWriteSectionAny - Can only write sequential sections to a TIFF file (last Z = {}, requested Z = {})\n",
                in_file.last_written_z, in_section
            ),
        );
        return -1;
    }
    if in_file.llx != 0
        || in_file.lly != 0
        || in_file.urx != in_file.nx - 1
        || in_file.ury != in_file.ny - 1
    {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: tiffWriteSectionAny - Can only write a whole section at once\n"),
        );
        return -1;
    }
    if in_file.format == IIFORMAT_COMPLEX {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: tiffWriteSectionAny - Cannot write complex data\n"),
        );
        return -1;
    }
    let mut inverted = false;
    let float_buffer = if if_float != 0 {
        if buf.len() % core::mem::size_of::<f32>() != 0
            || buf.as_ptr().align_offset(core::mem::align_of::<f32>()) != 0
        {
            return IIERR_BAD_CALL;
        }
        // The safe caller supplied the owned section buffer; interpret it as
        // floats only for the existing scalar conversion routine.
        Some(unsafe {
            core::slice::from_raw_parts(
                buf.as_ptr().cast::<f32>(),
                buf.len() / core::mem::size_of::<f32>(),
            )
        })
    } else {
        None
    };
    let converted = match ii_make_buffer_convert_if_float(
        in_file,
        float_buffer,
        &mut inverted,
        "tiffWriteSectionAny",
    ) {
        Ok(converted) => converted,
        Err(()) => return IIERR_MEMORY_ERR,
    };
    let resolution = if in_file.xscale == 1.0 {
        0
    } else {
        // `iitif.c:2775-2776`: `1.e8` and `-2.54e8` are double literals and
        // `inFile->xscale` is a `float` promoted to double for the division, so
        // the quotient is formed in double and only then truncated to the `int`
        // `resolution`.  Dividing in f32 instead rounds first: an `xscale` of
        // 1.5 gives 66666668 where the source gives 66666666, and the
        // `XResolution` rational libtiff writes differs.
        (if std::env::var_os("IMOD_TIFF_RESOL_PER_INCH").is_none() {
            1.0e8_f64
        } else {
            -2.54e8_f64
        } / in_file.xscale as f64) as i32
    };
    let compression = match std::env::var_os("IMOD_TIFF_COMPRESSION") {
        None => 1,
        Some(value) => {
            let mut end = 0usize;
            (strtol(value.as_encoded_bytes(), &mut end, 10) as i32).max(1)
        }
    };
    let quality = match std::env::var_os("IMOD_TIFF_QUALITY") {
        None => -1,
        Some(value) => {
            let mut end = 0usize;
            (strtol(value.as_encoded_bytes(), &mut end, 10) as i32).max(1)
        }
    };
    let result = if let Some(mut converted) = converted {
        tiff_write_section(
            in_file,
            converted.as_mut_slice(),
            compression,
            i32::from(inverted),
            resolution,
            quality,
        )
    } else {
        tiff_write_section(
            in_file,
            buf,
            compression,
            i32::from(inverted),
            resolution,
            quality,
        )
    };
    if result == 0 {
        in_file.last_written_z += 1;
    }
    result
}
/// C `tiffVersion` (`iitif.c:2793`).
pub fn tiff_version() -> (i32, i32) {
    let version = unsafe { TIFFGetVersion() };
    if version.is_null() {
        return (0, 0);
    }
    // `TIFFGetVersion` hands back libtiff's own C string; length it here and
    // read it as bytes.
    let mut version_len = 0usize;
    while unsafe { *version.add(version_len) } != 0 {
        version_len += 1;
    }
    let bytes = unsafe { core::slice::from_raw_parts(version.cast::<u8>(), version_len) };
    if bytes.windows(4).any(|word| word == b"IMOD") {
        return (0, 0);
    }
    let Some(start) = bytes.windows(6).position(|word| word == b"ersion") else {
        return (0, 0);
    };
    // `sscanf(substr + 7, "%d.%d.%d", ...)` in the C source skips the
    // whitespace following "Version" before accepting the first integer.
    let numbers = &bytes[start + 7..];
    let mut values = [0i32; 3];
    let mut index = 0;
    for part in numbers.split(|byte| *byte == b'.').take(3) {
        if let Ok(text) = core::str::from_utf8(part) {
            let digits = text
                .bytes()
                .skip_while(|byte| byte.is_ascii_whitespace())
                .take_while(|byte| byte.is_ascii_digit())
                .collect::<Vec<_>>();
            if let Ok(text) = core::str::from_utf8(&digits) {
                if let Ok(value) = text.parse() {
                    values[index] = value;
                }
            }
        }
        index += 1;
    }
    (values[0], values[1])
}
/// C `tiffNumReadThreads` (`iitif.c:2813`).
pub fn tiff_num_read_threads(nx: i32, ny: i32, compression: i32, max_threads: i32) -> i32 {
    if compression == IICOMPRESSION_NONE
        || compression == IICOMPRESSION_EER_7BIT
        || compression == IICOMPRESSION_EER_8BIT
    {
        return 1;
    }
    let scale = if compression == IICOMPRESSION_LZW || compression == IICOMPRESSION_ZIP {
        1.0
    } else if compression == IICOMPRESSION_JPEG {
        0.67
    } else {
        0.5
    };
    let mut threads = (4.0_f64 * scale * (((nx as f64 * ny as f64).sqrt() / 944.0).ln())
        / 2.0_f64.ln())
    .round() as i32;
    threads = threads.clamp(1, max_threads).clamp(1, ny / 2);
    num_omp_threads(threads).min(MAX_TIFF_THREADS as i32)
}
/// C `tiffParallelRead` (`iitif.c:2842`).
pub unsafe fn tiff_parallel_read(
    file_copies: *mut *mut ImodImageFile,
    max_threads: i32,
    llx: i32,
    urx: i32,
    lly: i32,
    ury: i32,
    data_size: i32,
    read_buf: *mut u8,
    iz_read: i32,
    convert: i32,
) -> i32 {
    if file_copies.is_null() || max_threads < 1 {
        return IIERR_BAD_CALL;
    }
    let first = *file_copies;
    if first.is_null() {
        return IIERR_BAD_CALL;
    }
    let saved = ((*first).llx, (*first).urx, (*first).lly, (*first).ury);
    let ny = ury + 1 - lly;
    let nx = urx + 1 - llx;
    let num_threads = tiff_num_read_threads(nx, ny, (*first).tiff_compression, max_threads);
    for ind in 0..num_threads {
        let file = *file_copies.add(ind as usize);
        if file.is_null() {
            return IIERR_BAD_CALL;
        }
        (*file).llx = llx;
        (*file).urx = urx;
        (*file).lly = lly + ind * (ny / num_threads);
        (*file).ury = lly + (ind + 1) * (ny / num_threads) - 1;
    }
    let last = *file_copies.add((num_threads - 1) as usize);
    (*last).ury = ury;
    let mut result = 0;
    for ind in 0..num_threads {
        let file = *file_copies.add(ind as usize);
        let error = read_section(
            file,
            read_buf.add(((*file).lly - lly) as usize * nx as usize * data_size as usize),
            iz_read,
            convert,
        );
        if error != 0 {
            result = error;
        }
    }
    (*first).llx = saved.0;
    (*first).urx = saved.1;
    (*first).lly = saved.2;
    (*first).ury = saved.3;
    let _ = (urx, ury, data_size);
    result
}
/// C `tiffParallelWrite` (`iitif.c:2899`).
pub unsafe fn tiff_parallel_write(
    in_file: *mut ImodImageFile,
    buf: *mut c_void,
    compression: i32,
    inverted: i32,
    resolution: i32,
    quality: i32,
    did_parallel: *mut i32,
) -> i32 {
    if in_file.is_null() || buf.is_null() || did_parallel.is_null() {
        return IIERR_BAD_CALL;
    }
    *did_parallel = 0;
    let mut pixel_size = 1;
    if (*in_file).type_ == IITYPE_SHORT || (*in_file).type_ == IITYPE_USHORT {
        pixel_size = 2;
    }
    if matches!((*in_file).type_, IITYPE_INT | IITYPE_UINT | IITYPE_FLOAT) {
        pixel_size = 4;
    }
    if (*in_file).format == IIFORMAT_RGB {
        pixel_size *= 3;
    }
    let strip_target = if compression != IICOMPRESSION_NONE {
        16_384
    } else {
        8_192
    };
    {
        let mut state = S_STRIP_TILE_STATE.lock().unwrap();
        state.line_bytes = (*in_file).nx * pixel_size;
        state.rows_per_strip = ((strip_target + state.line_bytes / 2) / state.line_bytes).max(1);
    }
    let rows_per_strip = S_STRIP_TILE_STATE.lock().unwrap().rows_per_strip;
    let mut zip_scale = 1.0_f64;
    if compression == IICOMPRESSION_ZIP {
        zip_scale = (1.0
            + ((((*in_file).nx as f64 * (*in_file).ny as f64).sqrt() / 2048.0).ln()
                / 4.0_f64.ln()))
        .clamp(1.0, 2.0);
    }
    let mut num_threads = (zip_scale * 2.0 * ((*in_file).nx as f64 * (*in_file).ny as f64).sqrt()
        / 944.0)
        .round() as i32;
    num_threads = num_threads.clamp(1, MAX_TIFF_THREADS as i32);
    // C `B3DCLAMP(numThreads, 1, ny / sRowsPerStrip / 4)` expands to
    // `B3DMAX(1, B3DMIN(max, value))`.  Unlike Rust's `clamp`, that is
    // defined when the small-image upper bound is zero: it yields one.
    num_threads = num_threads.min(((*in_file).ny / rows_per_strip) / 4).max(1);
    if let Some(limit) = std::env::var_os("TIFF_WRITE_THREAD_LIMIT") {
        let mut end = 0usize;
        let thread_limit = strtol(limit.as_encoded_bytes(), &mut end, 10) as i32;
        if thread_limit > 0 {
            num_threads = num_threads.min(thread_limit);
        }
    }
    num_threads = num_omp_threads(num_threads).min(MAX_TIFF_THREADS as i32);
    let (version, _) = tiff_version();
    if num_threads < 2
        || (*in_file).format == IIFORMAT_RGB
        || version < 4
        || compression == IICOMPRESSION_JPEG
        || compression == IICOMPRESSION_NONE
    {
        let pixel_bytes = if (*in_file).format == IIFORMAT_RGB {
            3
        } else if matches!((*in_file).type_, IITYPE_UBYTE | IITYPE_BYTE) {
            1
        } else if matches!((*in_file).type_, IITYPE_USHORT | IITYPE_SHORT) {
            2
        } else {
            4
        };
        let Some(byte_count) = usize::try_from((*in_file).nx)
            .ok()
            .and_then(|nx| {
                usize::try_from((*in_file).ny)
                    .ok()
                    .and_then(|ny| nx.checked_mul(ny))
            })
            .and_then(|pixels| pixels.checked_mul(pixel_bytes))
        else {
            return IIERR_BAD_CALL;
        };
        return tiff_write_section(
            &mut *in_file,
            core::slice::from_raw_parts_mut(buf.cast(), byte_count),
            compression,
            inverted,
            resolution,
            quality,
        );
    }

    *did_parallel = 1;
    S_SETTING_UP_PARALLEL.store(-1, Ordering::SeqCst);
    let mut strip = 0;
    let mut lines = 0;
    let mut tile_x = 0;
    let mut err = tiff_write_setup(
        in_file,
        compression,
        inverted,
        resolution,
        quality,
        &mut strip,
        &mut lines,
        &mut tile_x,
    );
    if inverted == 0 {
        *S_TMP_BUF.lock().unwrap() = Vec::new();
    }
    S_SETTING_UP_PARALLEL.store(0, Ordering::SeqCst);
    if err != 0 {
        return err;
    }
    let state = *S_STRIP_TILE_STATE.lock().unwrap();
    let num_strips = state.num_strips;
    let rows_per_strip = state.rows_per_strip;
    let line_bytes = state.line_bytes;
    let strip_bytes = state.strip_bytes;
    let mut lines_per_file = (*in_file).ny / num_threads;
    lines_per_file = rows_per_strip * (lines_per_file / rows_per_strip);
    let last_file_strips = num_strips - (num_threads - 1) * lines_per_file / rows_per_strip;
    S_FILE_BUF_SIZE.store(
        (4096.0 + (1.2_f64 * strip_bytes as f64 + 8.0) * last_file_strips as f64) as i32,
        Ordering::SeqCst,
    );
    let mut temp_files = [core::ptr::null_mut::<ImodImageFile>(); MAX_TIFF_THREADS];
    let mut thread_num_strips = [0_i32; MAX_TIFF_THREADS];
    let mut thread_cum_lines = [0_i32; MAX_TIFF_THREADS];
    let mut thread_tmp_buf: [Vec<u8>; MAX_TIFF_THREADS] = [const { Vec::new() }; MAX_TIFF_THREADS];
    let mut cumulative_lines = 0;
    let mut cumulative_strips = 0;
    let mut last_clean = -1_i32;
    for file in 0..num_threads as usize {
        let number_lines = if file + 1 == num_threads as usize {
            (*in_file).ny - cumulative_lines
        } else {
            lines_per_file
        };
        last_clean = file as i32;
        let Ok(buffer_size) = usize::try_from(S_FILE_BUF_SIZE.load(Ordering::SeqCst)) else {
            err = IIERR_MEMORY_ERR;
            break;
        };
        let mut file_buffer = Vec::new();
        if file_buffer.try_reserve_exact(buffer_size).is_err() {
            err = IIERR_MEMORY_ERR;
            break;
        }
        file_buffer.resize(buffer_size, 0);
        let mut buffers = S_PARALLEL_BUFFERS.lock().unwrap();
        buffers.file_buf[file] = file_buffer;
        buffers.cur_buf_ind[file] = 0;
        buffers.max_buf_ind[file] = 0;
        drop(buffers);
        temp_files[file] = ii_new();
        if temp_files[file].is_null() {
            err = IIERR_MEMORY_ERR;
            break;
        }
        // `iitif.c:2993`: `sprintf(..., "%s.%d.%d", inFile->filename, imodGetpid(), file)`.
        (*temp_files[file]).filename = Some(
            String::from_utf8(c_format_bytes(
                "%s.%d.%d",
                &[
                    CArg::Bytes((*in_file).filename.as_deref().unwrap_or("").as_bytes()),
                    CArg::Int(imod_getpid() as i64),
                    CArg::Int(file as i64),
                ],
            ))
            .expect("filename constructed from valid UTF-8 components"),
        );
        (*temp_files[file]).nx = (*in_file).nx;
        (*temp_files[file]).ny = number_lines;
        (*temp_files[file]).file = IIFILE_TIFF;
        (*temp_files[file]).mode = (*in_file).mode;
        (*temp_files[file]).type_ = (*in_file).type_;
        (*temp_files[file]).nz = 1;
        S_SETTING_UP_PARALLEL.store(file as i32 + 1, Ordering::SeqCst);
        err = tiff_open_new(temp_files[file]);
        if err != 0 {
            break;
        }
        tile_x = 0;
        err = tiff_write_setup(
            temp_files[file],
            compression,
            inverted,
            resolution,
            quality,
            &mut strip,
            &mut lines,
            &mut tile_x,
        );
        if err != 0 {
            break;
        }
        let thread_state = *S_STRIP_TILE_STATE.lock().unwrap();
        thread_num_strips[file] = thread_state.num_strips;
        thread_cum_lines[file] = if inverted != 0 {
            cumulative_lines
        } else {
            (*in_file).ny - cumulative_lines - number_lines
        };
        if inverted == 0 {
            thread_tmp_buf[file] = core::mem::take(&mut *S_TMP_BUF.lock().unwrap());
        }
        cumulative_strips += thread_state.num_strips;
        cumulative_lines += number_lines;
    }
    S_SETTING_UP_PARALLEL.store(0, Ordering::SeqCst);
    if err == 0 && num_strips != cumulative_strips {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "tiffParallelWrite: number of strips for full file ({}) not equal to total for temp files ({})\n",
                num_strips, cumulative_strips
            ),
        );
        err = IIERR_IO_ERROR;
    }
    if err == 0 {
        for file in 0..num_threads as usize {
            let mut lines_done = 0;
            for strip_index in 0..thread_num_strips[file] {
                let number_lines = rows_per_strip.min((*temp_files[file]).ny - lines_done);
                let source = if inverted != 0 {
                    (buf as *mut u8).add(
                        (strip_index * strip_bytes + thread_cum_lines[file] * line_bytes) as usize,
                    )
                } else {
                    (buf as *mut u8).add(
                        (thread_cum_lines[file] + (*temp_files[file]).ny
                            - (lines_done + number_lines)) as usize
                            * line_bytes as usize,
                    )
                };
                b3d_shift_bytes(
                    core::slice::from_raw_parts_mut(source, (line_bytes * number_lines) as usize),
                    line_bytes,
                    number_lines,
                    1,
                    if (*in_file).type_ == IITYPE_BYTE {
                        1
                    } else {
                        0
                    },
                );
                let mut use_buf = source.cast::<c_void>();
                if inverted == 0 {
                    use_buf = thread_tmp_buf[file].as_mut_ptr().cast();
                    let line_bytes = line_bytes as usize;
                    let strip_bytes = number_lines as usize * line_bytes;
                    let source_lines = core::slice::from_raw_parts(source, strip_bytes);
                    let destination_lines = &mut thread_tmp_buf[file][..strip_bytes];
                    for (destination, source) in destination_lines
                        .chunks_exact_mut(line_bytes)
                        .zip(source_lines.rchunks_exact(line_bytes))
                    {
                        destination.copy_from_slice(source);
                    }
                }
                let written = TIFFWriteEncodedStrip(
                    (*temp_files[file]).backend_handle.cast(),
                    strip_index as u32,
                    use_buf,
                    (line_bytes * number_lines) as isize,
                );
                b3d_shift_bytes(
                    core::slice::from_raw_parts_mut(source, (line_bytes * number_lines) as usize),
                    line_bytes,
                    number_lines,
                    -1,
                    if (*in_file).type_ == IITYPE_BYTE {
                        1
                    } else {
                        0
                    },
                );
                if written < 0 {
                    err = IIERR_IO_ERROR;
                    break;
                }
                lines_done += number_lines;
            }
        }
    }
    if err == 0 {
        let buffer_size = match usize::try_from(2 * strip_bytes) {
            Ok(buffer_size) => buffer_size,
            Err(_) => {
                err = IIERR_MEMORY_ERR;
                0
            }
        };
        if err == 0 {
            let mut tmp_buf = S_TMP_BUF.lock().unwrap();
            *tmp_buf = Vec::new();
            if tmp_buf.try_reserve_exact(buffer_size).is_err() {
                err = IIERR_MEMORY_ERR;
            } else {
                tmp_buf.resize(buffer_size, 0);
                let mut copied_strips = 0;
                let mut lines_done = 0;
                for file in 0..num_threads as usize {
                    for strip_index in 0..thread_num_strips[file] {
                        let lines = rows_per_strip.min((*in_file).ny - lines_done);
                        let bytes = TIFFReadRawStrip(
                            (*temp_files[file]).backend_handle.cast(),
                            strip_index as u32,
                            tmp_buf.as_mut_ptr().cast(),
                            (2 * strip_bytes) as isize,
                        );
                        if bytes <= 0 {
                            b3d_error(
                                Some(&mut ImodFile::Stderr),
                                format_args!(
                                    "tiffParallelWrite: Read error getting raw strip {} from file {}\n",
                                    strip_index, file
                                ),
                            );
                            err = IIERR_IO_ERROR;
                            break;
                        }
                        if TIFFWriteRawStrip(
                            (*in_file).backend_handle.cast(),
                            (strip_index + copied_strips) as u32,
                            tmp_buf.as_mut_ptr().cast(),
                            bytes,
                        ) <= 0
                        {
                            b3d_error(
                                Some(&mut ImodFile::Stderr),
                                format_args!(
                                    "tiffParallelWrite: Error rewriting raw strip {} from file {}\n",
                                    strip_index, file
                                ),
                            );
                            err = IIERR_IO_ERROR;
                            break;
                        }
                        lines_done += lines;
                    }
                    copied_strips += thread_num_strips[file];
                    // The source clears the error at the end of every file iteration
                    // (`iitif.c:3110`), so a raw-strip failure is reported to stderr
                    // but not returned.  That behaviour is preserved deliberately.
                    err = 0;
                }
                *tmp_buf = Vec::new();
            }
        }
    }
    if last_clean >= 0 {
        for index in 0..=last_clean as usize {
            if !temp_files[index].is_null() {
                ii_close(temp_files[index]);
                if let Some(name) = (&(*temp_files[index]).filename).as_deref() {
                    let _ = std::fs::remove_file(name);
                }
                ii_delete(temp_files[index]);
                thread_tmp_buf[index] = Vec::new();
            }
            S_PARALLEL_BUFFERS.lock().unwrap().file_buf[index] = Vec::new();
        }
    }
    (*in_file).state = IISTATE_BUSY;
    if err == 0 {
        *S_DESCRIPTION.lock().unwrap() = None;
    }
    err
}
/// C `constrainAndStoreMinMax` (`iitif.c:3134`).
unsafe fn constrain_and_store_min_max(in_file: *mut ImodImageFile) {
    if in_file.is_null() {
        return;
    }
    let mut minimum = (*in_file).amin as f64;
    let mut maximum = (*in_file).amax as f64;
    if (*in_file).format == IIFORMAT_RGB {
        minimum = 0.0;
        maximum = 255.0;
    } else {
        match (*in_file).type_ {
            IITYPE_BYTE => {
                minimum = minimum.clamp(0., 255.) - 128.;
                maximum = maximum.clamp(0., 255.) - 128.;
            }
            IITYPE_UBYTE => {
                minimum = minimum.clamp(0., 255.);
                maximum = maximum.clamp(0., 255.);
            }
            IITYPE_SHORT => {
                minimum = minimum.clamp(-32768., 32767.);
                maximum = maximum.clamp(-32768., 32767.);
            }
            IITYPE_USHORT => {
                minimum = minimum.clamp(0., 65535.);
                maximum = maximum.clamp(0., 65535.);
            }
            _ => {}
        }
    }
    TIFFSetField(
        (*in_file).backend_handle.cast(),
        TIFFTAG_SMINSAMPLEVALUE,
        minimum,
    );
    TIFFSetField(
        (*in_file).backend_handle.cast(),
        TIFFTAG_SMAXSAMPLEVALUE,
        maximum,
    );
}
/// C `registerCustomTIFFTags` (`iitif.c:3191`).
unsafe extern "C" fn register_custom_tiff_tags(tif: *mut Tiff) {
    TIFFMergeFieldInfo(tif, S_XTIFF_FIELD_INFO.as_ptr(), 8);
    let parent = *S_PARENT_EXTENDER.lock().unwrap();
    if let Some(parent) = parent {
        parent(tif);
    }
}
/// C `augment_libtiff_with_custom_tags` (`iitif.c:3201`).
unsafe fn augment_libtiff_with_custom_tags() {
    if S_AUGMENTED_TAGS.load(Ordering::SeqCst) != 0 {
        return;
    }
    let (version, minor) = tiff_version();
    if version < 4 || (version == 4 && minor < 5) {
        return;
    }
    S_AUGMENTED_TAGS.store(1, Ordering::SeqCst);
    let parent = TIFFSetTagExtender(Some(register_custom_tiff_tags));
    *S_PARENT_EXTENDER.lock().unwrap() = parent;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TIFF_IO_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn eer_configuration_clamps_and_resets_gain_reference() {
        let mut gain = [1.0_f32];
        tiff_gain_reference_for_eer(gain.as_mut_ptr());
        tiff_set_eer_read_properties(99, 7, IIFLAG_ANTIALIAS_EER | IIFLAG_EER_USE_LANCZOS);
        assert_eq!(S_READ_EER_AS_SUPER_RES.load(Ordering::SeqCst), 2);
        assert_eq!(S_AUTOGROUP_EER.load(Ordering::SeqCst), 7);
        assert_eq!(S_ANTIALIAS_EER.load(Ordering::SeqCst), 4);
        assert!(S_GAIN_REFERENCE.load(Ordering::SeqCst).is_null());
    }

    #[test]
    fn suppress_and_restore_errors_reinstates_source_saved_handler() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _guard = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let original = TIFFSetErrorHandler(core::ptr::null_mut());
            let sentinel = warning_handler as *const () as *mut c_void;
            let _ = TIFFSetErrorHandler(sentinel);
            S_OLD_ERR_HANDLER.store(core::ptr::null_mut(), Ordering::SeqCst);
            tiff_suppress_errors();
            tiff_restore_errors();
            assert_eq!(TIFFSetErrorHandler(core::ptr::null_mut()), sentinel);
            let _ = TIFFSetErrorHandler(original);
            S_OLD_ERR_HANDLER.store(core::ptr::null_mut(), Ordering::SeqCst);
        }
    }

    #[test]
    fn cleanup_from_eer_drops_owned_filter_state_and_resets_flags() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            S_EER_FLAGS.store(IIFLAG_SKIP_EER_DIRS, Ordering::SeqCst);
            *S_EER_FILTERS.lock().unwrap() = EerFilters {
                all: vec![0],
                x_start: vec![0],
                y_start: vec![0],
            };
            cleanup_from_eer(
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            assert_eq!(S_EER_FLAGS.load(Ordering::SeqCst), 0);
            assert!(
                S_EER_FILTERS.lock().unwrap().all.is_empty()
                    && S_EER_FILTERS.lock().unwrap().x_start.is_empty()
                    && S_EER_FILTERS.lock().unwrap().y_start.is_empty()
            );
        }
    }

    #[test]
    fn convert_eer_positions_accumulates_unfiltered_native_pixels() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let image = ii_new();
            (*image).nx = 4;
            (*image).ny = 4;
            (*image).mode = MRC_MODE_BYTE;
            (*image).read_eer_as_super_res = 0;
            (*image).antialias_eerfilter = 0;
            S_EER_FLAGS.store(0, Ordering::SeqCst);
            let positions = [0_i32, 0, 5];
            let symbols = [0_u8; 3];
            let mut output = [99_u8; 16];
            convert_eer_positions(
                image,
                positions.as_ptr().cast_mut(),
                symbols.as_ptr().cast_mut(),
                3,
                output.as_mut_ptr(),
                0,
                3,
                0,
                3,
            );
            assert_eq!(output[12], 2);
            assert_eq!(output[9], 1);
            assert_eq!(output.iter().copied().sum::<u8>(), 3);
            ii_delete(image);
        }
    }

    #[test]
    fn convert_eer_positions_accumulates_antialiased_full_super_res_shorts() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let image = ii_new();
            (*image).nx = 8;
            (*image).ny = 8;
            (*image).mode = MRC_MODE_SHORT;
            (*image).read_eer_as_super_res = 2;
            (*image).antialias_eerfilter = 1;
            (*image).eerkernel_scale = 5;
            S_EER_FLAGS.store(0, Ordering::SeqCst);
            S_GAIN_REFERENCE.store(core::ptr::null_mut(), Ordering::SeqCst);
            let positions = [0_i32, 0];
            let symbols = [0_u8; 2];
            let mut output = [0_i16; 64];
            convert_eer_positions(
                image,
                positions.as_ptr().cast_mut(),
                symbols.as_ptr().cast_mut(),
                2,
                output.as_mut_ptr().cast(),
                0,
                7,
                0,
                7,
            );
            assert_eq!(output[56], 10);
            ii_delete(image);
        }
    }

    #[test]
    fn convert_eer_positions_deposits_reduced_resolution_filter_packet() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let mut width = 0;
            assert_eq!(select_zoom_filter(3, 0.5, &mut width), 0);
            // `ReadSection` owns these vectors for the reduced-resolution filters.
            let filter_count = 2 * 2;
            *S_EER_FILTERS.lock().unwrap() = EerFilters {
                all: vec![0; filter_count * 16],
                x_start: vec![0; filter_count],
                y_start: vec![0; filter_count],
            };
            let image = ii_new();
            (*image).nx = 16;
            (*image).ny = 16;
            (*image).mode = MRC_MODE_SHORT;
            (*image).read_eer_as_super_res = 1;
            (*image).antialias_eerfilter = 3;
            (*image).eerkernel_scale = 100;
            S_EER_FLAGS.store(0, Ordering::SeqCst);
            S_GAIN_REFERENCE.store(core::ptr::null_mut(), Ordering::SeqCst);
            let positions = [36_i32];
            let symbols = [0_u8];
            let mut output = [0_i16; 256];
            convert_eer_positions(
                image,
                positions.as_ptr().cast_mut(),
                symbols.as_ptr().cast_mut(),
                1,
                output.as_mut_ptr().cast(),
                0,
                15,
                0,
                15,
            );
            assert_eq!(output.iter().map(|v| *v as i32).sum::<i32>(), 100);
            assert!(output.iter().filter(|v| **v != 0).count() > 1);
            cleanup_from_eer(
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            ii_delete(image);
        }
    }

    #[test]
    fn convert_eer_positions_gain_normalizes_reduced_resolution_filter_packet() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let mut width = 0;
            assert_eq!(select_zoom_filter(3, 0.5, &mut width), 0);
            // `ReadSection` owns these vectors for the reduced-resolution filters.
            let filter_count = 2 * 2;
            *S_EER_FILTERS.lock().unwrap() = EerFilters {
                all: vec![0; filter_count * 16],
                x_start: vec![0; filter_count],
                y_start: vec![0; filter_count],
            };
            let image = ii_new();
            (*image).nx = 16;
            (*image).ny = 16;
            (*image).mode = MRC_MODE_SHORT;
            (*image).read_eer_as_super_res = 1;
            (*image).antialias_eerfilter = 3;
            (*image).eerkernel_scale = 100;
            S_EER_FLAGS.store(0, Ordering::SeqCst);
            let mut gain = [0_f32; 1024];
            gain[496] = 200.0;
            S_GAIN_REFERENCE.store(gain.as_mut_ptr(), Ordering::SeqCst);
            let positions = [36_i32];
            let symbols = [0_u8];
            let mut output = [0_i16; 256];
            convert_eer_positions(
                image,
                positions.as_ptr().cast_mut(),
                symbols.as_ptr().cast_mut(),
                1,
                output.as_mut_ptr().cast(),
                0,
                15,
                0,
                15,
            );
            // The source filter values carry the EER kernel scale (100 here), and it
            // rounds each of the 16 weighted contributions independently, so the
            // deposited total is one short of `scale * refVal` = 100 * 200.
            assert_eq!(output.iter().map(|v| *v as i32).sum::<i32>(), 19999);
            S_GAIN_REFERENCE.store(core::ptr::null_mut(), Ordering::SeqCst);
            cleanup_from_eer(
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            ii_delete(image);
        }
    }

    /// Pack a stream of (bits, count) fields LSB-first, as `decodeEERimage` reads them.
    fn eer_bits(fields: &[(u32, u32)]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut bit = 0usize;
        for (value, width) in fields {
            for i in 0..*width {
                if bit % 8 == 0 {
                    out.push(0);
                }
                if value >> i & 1 != 0 {
                    let last = out.len() - 1;
                    out[last] |= 1 << (bit % 8);
                }
                bit += 1;
            }
        }
        // The source reads four bytes ahead of the current byte on every chunk.
        out.extend_from_slice(&[0u8; 8]);
        out
    }

    #[test]
    fn eer_7_bit_decoder_uses_source_eleven_bit_advance_and_127_skip() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            S_IGNORE_BAD_EER_END.store(0, Ordering::SeqCst);
            // Two electrons, then a bare 127 run (7 bits only, no symbol), then a
            // third electron, then the run that lands exactly on the pixel count.
            // Only the source's 7+4 bit accounting decodes this stream.
            let total: i32 = 3 + 1 + 10 + 1 + 127 + 5 + 1;
            let mut positions = [0i32; 8];
            let mut symbols = [0u8; 8];
            let mut electrons = -1;
            // The final run terminates exactly on the image pixel count.
            let mut full = eer_bits(&[
                (3, 7),
                (0x0A ^ 0x01, 4),
                (10, 7),
                (0x0A ^ 0x0C, 4),
                (127, 7),
                (5, 7),
                (0x0A ^ 0x07, 4),
                (9, 7),
            ]);
            assert_eq!(
                decode_eer_image(
                    full.as_mut_ptr(),
                    positions.as_mut_ptr(),
                    symbols.as_mut_ptr(),
                    1,
                    total + 9,
                    full.len() as i32,
                    &mut electrons,
                ),
                0
            );
            assert_eq!(electrons, 3);
            assert_eq!(&positions[..3], &[3, 14, 147]);
            assert_eq!(&symbols[..3], &[1, 12, 7]);
        }
    }

    #[test]
    fn eer_7_bit_decoder_reports_source_bad_final_position() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            S_IGNORE_BAD_EER_END.store(0, Ordering::SeqCst);
            let mut stream = eer_bits(&[(3, 7), (0x0A ^ 0x01, 4), (20, 7)]);
            let mut positions = [0i32; 4];
            let mut symbols = [0u8; 4];
            let mut electrons = -1;
            // The stream overshoots the image size, so the source returns 1.
            assert_eq!(
                decode_eer_image(
                    stream.as_mut_ptr(),
                    positions.as_mut_ptr(),
                    symbols.as_mut_ptr(),
                    1,
                    10,
                    stream.len() as i32,
                    &mut electrons,
                ),
                1
            );
            assert_eq!(electrons, 1);
            // The source's ignore flag turns the same stream into success.
            S_IGNORE_BAD_EER_END.store(1, Ordering::SeqCst);
            assert_eq!(
                decode_eer_image(
                    stream.as_mut_ptr(),
                    positions.as_mut_ptr(),
                    symbols.as_mut_ptr(),
                    1,
                    10,
                    stream.len() as i32,
                    &mut electrons,
                ),
                0
            );
            S_IGNORE_BAD_EER_END.store(0, Ordering::SeqCst);
        }
    }

    #[test]
    fn eer_8_bit_decoder_realigns_after_a_source_wasted_bit_run() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            S_IGNORE_BAD_EER_END.store(1, Ordering::SeqCst);
            // p1 = 255 makes the source advance one byte and restart without
            // placing an electron; p2 = 255 advances two bytes and sets the
            // misaligned flag for the nibble-shifted read on the next pass.
            let mut stream = eer_bits(&[
                (4, 8),
                (0x0A ^ 0x02, 4),
                (255, 8),
                (0, 4),
                (7, 8),
                (0x0A ^ 0x09, 4),
            ]);
            let mut positions = [0i32; 16];
            let mut symbols = [0u8; 16];
            let mut electrons = -1;
            // Stop at the five real bytes; the trailing pad is only read ahead.
            assert_eq!(
                decode_eer_image(
                    stream.as_mut_ptr(),
                    positions.as_mut_ptr(),
                    symbols.as_mut_ptr(),
                    0,
                    1_000_000,
                    5,
                    &mut electrons,
                ),
                0
            );
            assert_eq!(electrons, 4);
            // 4 from the aligned pair, then the 255 run advances two bytes and
            // the next pass reads the nibble-shifted 112 run, after which the
            // aligned pair resumes at the same byte.
            assert_eq!(&positions[..4], &[4, 372, 388, 437]);
            assert_eq!(&symbols[..4], &[2, 10, 13, 10]);
            S_IGNORE_BAD_EER_END.store(0, Ordering::SeqCst);
        }
    }

    /// Write a minimal single-strip TIFF whose IFDs carry EER compression and the
    /// given raw strip payloads, the way an EER camera file is laid out.
    fn write_eer_tiff(path: &std::path::Path, nx: u32, ny: u32, comp: u32, strips: &[Vec<u8>]) {
        let tag_count: u16 = 9;
        let ifd_size = 2 + 12 * tag_count as usize + 4;
        let mut ifd_offsets = Vec::new();
        let mut at = 8usize;
        for _ in strips {
            ifd_offsets.push(at);
            at += ifd_size;
        }
        let mut data_offsets = Vec::new();
        for strip in strips {
            data_offsets.push(at);
            at += strip.len();
        }
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"II");
        out.extend_from_slice(&42u16.to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        for (index, strip) in strips.iter().enumerate() {
            let next = if index + 1 < strips.len() {
                ifd_offsets[index + 1] as u32
            } else {
                0
            };
            let tags: [(u16, u16, u32, u32); 9] = [
                (256, 3, 1, nx),
                (257, 3, 1, ny),
                (258, 3, 1, 8),
                (259, 3, 1, comp),
                (262, 3, 1, 1),
                (273, 4, 1, data_offsets[index] as u32),
                (277, 3, 1, 1),
                (278, 3, 1, ny),
                (279, 4, 1, strip.len() as u32),
            ];
            out.extend_from_slice(&tag_count.to_le_bytes());
            for (tag, kind, count, value) in tags {
                out.extend_from_slice(&tag.to_le_bytes());
                out.extend_from_slice(&kind.to_le_bytes());
                out.extend_from_slice(&count.to_le_bytes());
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&next.to_le_bytes());
        }
        for strip in strips {
            out.extend_from_slice(strip);
        }
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn tiff_check_and_read_section_decode_a_native_eer_file() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let chip = 8u32;
            let total = (chip * chip) as i32;
            // One electron at sensor pixel 9 with sub-pixel symbol 0, then the
            // terminating run that lands exactly on the sensor pixel count.
            let frame = eer_bits(&[(9, 7), (0x0A ^ 0x00, 4), ((total - 10) as u32, 7)]);
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-eer-{}-{:?}.tif",
                std::process::id(),
                std::thread::current().id()
            ));
            write_eer_tiff(&path, chip, chip, IICOMPRESSION_EER_7BIT as u32, &[frame]);
            let name = path.to_string_lossy().into_owned();

            // Match the source defaults that iiTIFFCheck installs from the
            // environment on its first call in a process.
            S_AUTOGROUP_EER.store(1, Ordering::SeqCst);
            S_READ_EER_AS_SUPER_RES.store(2, Ordering::SeqCst);
            S_ANTIALIAS_EER.store(0, Ordering::SeqCst);
            S_EER_FLAGS.store(0, Ordering::SeqCst);
            S_GAIN_REFERENCE.store(core::ptr::null_mut(), Ordering::SeqCst);

            let reader = ii_new();
            (*reader).filename = Some(name.clone());
            (*reader).fmode = "rb".into();
            (*reader).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(&name, "rb");
            assert_eq!(ii_tiff_check(reader), 0);
            // Super-resolution 2 quadruples each sensor axis.
            assert_eq!(((*reader).nx, (*reader).ny, (*reader).nz), (32, 32, 1));
            assert_eq!((*reader).tiff_compression, IICOMPRESSION_EER_7BIT);
            assert_eq!((*reader).num_frames_in_eerfile, 1);
            assert_eq!((*reader).read_eer_as_super_res, 2);
            assert_eq!((*reader).type_, IITYPE_UBYTE);

            (*reader).llx = 0;
            (*reader).lly = 0;
            (*reader).urx = (*reader).nx - 1;
            (*reader).ury = (*reader).ny - 1;
            let mut image = vec![0u8; 32 * 32];
            assert_eq!(tiff_read_section(reader, image.as_mut_ptr().cast(), 0), 0);
            // Sensor pixel 9 is chip column 1, chip row 1; symbol 0 keeps the
            // low sub-pixel, and the source inverts Y on output.
            let x = ((9 % chip as i32) << 2) | 0;
            let y = ((*reader).ny - 1) - (((9 / chip as i32) << 2) | 0);
            assert_eq!(image[(x + y * 32) as usize], 1);
            assert_eq!(image.iter().map(|v| *v as i32).sum::<i32>(), 1);
            ii_delete(reader);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn read_section_scales_unsigned_shorts_through_the_source_byte_map() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-scale-{}-{:?}.tif",
                std::process::id(),
                std::thread::current().id()
            ));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            (*writer).filename = Some(name.clone());
            (*writer).nx = 4;
            (*writer).ny = 2;
            (*writer).nz = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_USHORT;
            (*writer).amin = 0.;
            (*writer).amax = 1000.;
            assert_eq!(tiff_open_new(writer), 0);
            let pixels: [u16; 8] = [0, 100, 200, 400, 600, 800, 900, 1000];
            let mut pixel_bytes: Vec<u8> = pixels
                .iter()
                .flat_map(|pixel| pixel.to_ne_bytes())
                .collect();
            assert_eq!(
                tiff_write_section(&mut *writer, &mut pixel_bytes, 1, 0, 0, -1),
                0
            );
            tiff_close(&mut *writer);
            ii_delete(writer);

            let reader = ii_new();
            (*reader).filename = Some(name.clone());
            (*reader).fmode = "rb".into();
            (*reader).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(&name, "rb");
            assert_eq!(ii_tiff_check(reader), 0);
            assert_eq!((*reader).type_, IITYPE_USHORT);
            // The source builds a short-to-byte map from slope and offset when
            // reading a 16-bit directory as bytes.  A unit slope leaves the map
            // out of the byte path, so use the scaling the loader would set.
            (*reader).slope = 255. / 1000.;
            (*reader).offset = 0.;
            (*reader).llx = 0;
            (*reader).lly = 0;
            (*reader).urx = 3;
            (*reader).ury = 1;
            let mut bytes = [0u8; 8];
            assert_eq!(
                tiff_read_section_byte(reader, bytes.as_mut_ptr().cast(), 0),
                0
            );
            // `get_short_map` covers the whole 16-bit domain, so the result is
            // monotone and reaches the source clamp limits at the endpoints.
            assert_eq!(bytes[0], 0);
            assert_eq!(bytes[7], 255);
            assert!(bytes.windows(2).all(|pair| pair[0] <= pair[1]));
            ii_delete(reader);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn filter_warnings_installs_the_source_handler_and_saves_the_old_one() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let original = TIFFSetWarningHandler(core::ptr::null_mut());
            let sentinel = warning_handler as *mut c_void;
            let _ = TIFFSetWarningHandler(sentinel);
            S_WARNINGS_SUPPRESSED.store(0, Ordering::SeqCst);
            S_OLD_HANDLER.store(core::ptr::null_mut(), Ordering::SeqCst);
            tiff_filter_warnings();
            // The source records the previous handler and installs its own.
            assert_eq!(S_OLD_HANDLER.load(Ordering::SeqCst), sentinel);
            let installed = TIFFSetWarningHandler(core::ptr::null_mut());
            assert_eq!(installed, warning_handler as *mut c_void);
            // With warnings suppressed the source leaves the handler alone.
            S_WARNINGS_SUPPRESSED.store(1, Ordering::SeqCst);
            let _ = TIFFSetWarningHandler(sentinel);
            tiff_filter_warnings();
            assert_eq!(TIFFSetWarningHandler(original), sentinel);
            S_WARNINGS_SUPPRESSED.store(0, Ordering::SeqCst);
            S_OLD_HANDLER.store(core::ptr::null_mut(), Ordering::SeqCst);
        }
    }

    #[test]
    fn write_setup_leaves_an_already_saved_error_handler_installed() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-dmtags-{}-{:?}.tif",
                std::process::id(),
                std::thread::current().id()
            ));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            (*writer).filename = Some(name.clone());
            (*writer).nx = 2;
            (*writer).ny = 2;
            (*writer).nz = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            (*writer).amin = 0.;
            (*writer).amax = 3.;
            assert_eq!(tiff_open_new(writer), 0);

            // A handler already saved by an earlier suppress means the source
            // does not suppress or restore around its DigitalMicrograph tags.
            let original = TIFFSetErrorHandler(core::ptr::null_mut());
            let sentinel = warning_handler as *mut c_void;
            let _ = TIFFSetErrorHandler(sentinel);
            S_OLD_ERR_HANDLER.store(sentinel, Ordering::SeqCst);
            let mut pixels = [1_u8, 2, 3, 0];
            assert_eq!(
                tiff_write_section(&mut *writer, &mut pixels, 1, 0, 100, -1),
                0
            );
            assert_eq!(TIFFSetErrorHandler(core::ptr::null_mut()), sentinel);
            let _ = TIFFSetErrorHandler(original);
            S_OLD_ERR_HANDLER.store(core::ptr::null_mut(), Ordering::SeqCst);
            tiff_close(&mut *writer);
            ii_delete(writer);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn copy_line_preserves_rgb_and_converts_rgb_to_float_luminance() {
        unsafe {
            let rgb = [10_u8, 20, 30, 100, 50, 0];
            let mut native = [0_u8; 6];
            copy_line(
                rgb.as_ptr().cast_mut(),
                native.as_mut_ptr(),
                2,
                MRSA_NOPROC,
                3,
                IITYPE_UBYTE,
                IIFORMAT_RGB,
                1,
                1.,
                0.,
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            assert_eq!(native, rgb);
            let mut luminance = [0_f32; 2];
            copy_line(
                rgb.as_ptr().cast_mut(),
                luminance.as_mut_ptr().cast(),
                2,
                MRSA_FLOAT,
                3,
                IITYPE_UBYTE,
                IIFORMAT_RGB,
                1,
                1.,
                0.,
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            assert!((luminance[0] - 18.1).abs() < 1.0e-5 && (luminance[1] - 59.5).abs() < 1.0e-5);
        }
    }

    #[test]
    fn copy_line_expands_colormap_or_converts_its_luminance() {
        unsafe {
            let indexes = [1_u8, 2];
            let mut colormap = [0_u8; 768];
            colormap[1] = 10;
            colormap[257] = 20;
            colormap[513] = 30;
            colormap[2] = 100;
            colormap[258] = 50;
            let mut rgb = [0_u8; 6];
            copy_line(
                indexes.as_ptr().cast_mut(),
                rgb.as_mut_ptr(),
                2,
                MRSA_NOPROC,
                1,
                IITYPE_UBYTE,
                IIFORMAT_COLORMAP,
                1,
                1.,
                0.,
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                colormap.as_mut_ptr(),
            );
            assert_eq!(rgb, [10, 20, 30, 100, 50, 0]);
            let mut gray = [0_u8; 2];
            copy_line(
                indexes.as_ptr().cast_mut(),
                gray.as_mut_ptr(),
                2,
                MRSA_BYTE,
                1,
                IITYPE_UBYTE,
                IIFORMAT_COLORMAP,
                1,
                1.,
                0.,
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                colormap.as_mut_ptr(),
            );
            assert_eq!(gray, [18, 60]);
        }
    }

    #[test]
    fn copy_line_converts_short_float_and_packed_four_bit_values() {
        unsafe {
            let signed = [-2_i16, 7];
            let mut floats = [0_f32; 2];
            copy_line(
                signed.as_ptr().cast_mut().cast(),
                floats.as_mut_ptr().cast(),
                2,
                MRSA_FLOAT,
                2,
                IITYPE_SHORT,
                IIFORMAT_LUMINANCE,
                1,
                1.,
                0.,
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            assert_eq!(floats, [-2., 7.]);
            let source = [1.5_f32, 300.];
            let mut bytes = [0_u8; 2];
            copy_line(
                source.as_ptr().cast_mut().cast(),
                bytes.as_mut_ptr(),
                2,
                MRSA_BYTE,
                4,
                IITYPE_FLOAT,
                IIFORMAT_LUMINANCE,
                1,
                2.,
                1.,
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            assert_eq!(bytes, [4, 255]);
            let packed = [0x2b_u8, 0x40];
            let mut first = [0_u8; 256];
            let mut second = [0_u8; 256];
            for value in 0..256 {
                first[value] = (value & 15) as u8;
                second[value] = (value >> 4) as u8;
            }
            let mut unpacked = [0_u8; 3];
            copy_line(
                packed.as_ptr().cast_mut(),
                unpacked.as_mut_ptr(),
                3,
                MRSA_BYTE,
                1,
                IITYPE_UBYTE,
                IIFORMAT_LUMINANCE,
                1,
                1.,
                0.,
                0,
                1,
                core::ptr::null_mut(),
                first.as_mut_ptr(),
                second.as_mut_ptr(),
                core::ptr::null_mut(),
            );
            assert_eq!(unpacked, [11, 2, 0]);
        }
    }

    #[test]
    fn copy_line_converts_signed_byte_and_32_bit_integer_scalars() {
        unsafe {
            let mut map = [0_u8; 256];
            map[130] = 2;
            map[5] = 200;
            let bytes = [130_u8, 5];
            let mut converted = [0_u8; 2];
            copy_line(
                bytes.as_ptr().cast_mut(),
                converted.as_mut_ptr(),
                2,
                MRSA_BYTE,
                1,
                IITYPE_BYTE,
                IIFORMAT_LUMINANCE,
                1,
                1.,
                0.,
                0,
                0,
                map.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            assert_eq!(converted, [2, 200]);
            let signed = [-3_i32, 200];
            let mut out = [0_u8; 2];
            copy_line(
                signed.as_ptr().cast_mut().cast(),
                out.as_mut_ptr(),
                2,
                MRSA_BYTE,
                4,
                IITYPE_INT,
                IIFORMAT_LUMINANCE,
                1,
                2.,
                10.,
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            assert_eq!(out, [4, 255]);
            let unsigned = [1_u32, 65_000];
            let mut ushort = [0_u16; 2];
            copy_line(
                unsigned.as_ptr().cast_mut().cast(),
                ushort.as_mut_ptr().cast(),
                2,
                MRSA_USHORT,
                4,
                IITYPE_UINT,
                IIFORMAT_LUMINANCE,
                1,
                2.,
                0.,
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            assert_eq!(ushort, [2, 65535]);
        }
    }

    #[test]
    fn read_thread_choice_matches_source_boundaries() {
        assert_eq!(tiff_num_read_threads(4096, 4096, IICOMPRESSION_NONE, 8), 1);
        assert_eq!(
            tiff_num_read_threads(4096, 4096, IICOMPRESSION_EER_8BIT, 8),
            1
        );
        assert!(
            (1..=MAX_TIFF_THREADS as i32).contains(&tiff_num_read_threads(
                4096,
                4096,
                IICOMPRESSION_ZIP,
                8
            ))
        );
    }

    #[test]
    fn native_tiff_buffer_callbacks_set_source_einval_on_bounds_failure() {
        unsafe {
            S_FILE_BUF_SIZE.store(1, Ordering::SeqCst);
            S_PARALLEL_BUFFERS.lock().unwrap().cur_buf_ind[0] = 1;
            let mut byte = 0_u8;
            *libc::__errno_location() = 0;
            assert_eq!(
                buf_write_proc(core::ptr::null_mut(), (&mut byte as *mut u8).cast(), 1),
                -1
            );
            assert_eq!(*libc::__errno_location(), libc::EINVAL);
            *libc::__errno_location() = 0;
            assert_eq!(buf_seek_proc(core::ptr::null_mut(), 1, 0), u64::MAX);
            assert_eq!(*libc::__errno_location(), libc::EINVAL);
            S_FILE_BUF_SIZE.store(0, Ordering::SeqCst);
            S_PARALLEL_BUFFERS.lock().unwrap().cur_buf_ind[0] = 0;
        }
    }

    #[test]
    fn eer_8_bit_decoder_preserves_position_and_symbol_transform() {
        let bytes = [0_u8, 0_u8, 0_u8, 0_u8];
        let mut positions = [99_i32; 2];
        let mut symbols = [99_u8; 2];
        let mut count = -1;
        unsafe {
            assert_eq!(
                decode_eer_image(
                    bytes.as_ptr().cast_mut(),
                    positions.as_mut_ptr(),
                    symbols.as_mut_ptr(),
                    0,
                    1,
                    1,
                    &mut count,
                ),
                0
            );
        }
        assert_eq!((count, positions[0], symbols[0]), (1, 0, 0x0a));
    }

    #[test]
    fn eer_7_bit_decoder_stops_on_image_boundary() {
        let bytes = [0_u8; 8];
        let mut positions = [99_i32; 2];
        let mut symbols = [99_u8; 2];
        let mut count = -1;
        unsafe {
            assert_eq!(
                decode_eer_image(
                    bytes.as_ptr().cast_mut(),
                    positions.as_mut_ptr(),
                    symbols.as_mut_ptr(),
                    1,
                    1,
                    0,
                    &mut count,
                ),
                0
            );
        }
        assert_eq!((count, positions[0], symbols[0]), (1, 0, 0x0a));
    }

    #[test]
    fn native_libtiff_write_and_read_section_roundtrip() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path =
                std::env::temp_dir().join(format!("imod-rs-iitif-{}.tif", std::process::id()));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = Some(name.clone());
            (*writer).nx = 2;
            (*writer).ny = 2;
            (*writer).nz = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            (*writer).amin = 2.;
            (*writer).amax = 5.;
            assert_eq!(tiff_open_new(writer), 0);
            let mut pixels = [3_u8, 1, 4, 1];
            assert_eq!(
                tiff_write_section(&mut *writer, &mut pixels, 1, 0, 0, -1),
                0
            );
            tiff_close(&mut *writer);
            ii_delete(writer);

            let reader = ii_new();
            assert!(!reader.is_null());
            (*reader).filename = Some(name.clone());
            (*reader).fmode = "rb".into();
            (*reader).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(&name, "rb");
            assert_eq!(ii_tiff_check(reader), 0);
            let mut minimum = 0.0f64;
            let mut maximum = 0.0f64;
            assert_ne!(
                tiff_get_field(
                    reader,
                    TIFFTAG_SMINSAMPLEVALUE as i32,
                    (&mut minimum as *mut f64).cast()
                ),
                0
            );
            assert_ne!(
                tiff_get_field(
                    reader,
                    TIFFTAG_SMAXSAMPLEVALUE as i32,
                    (&mut maximum as *mut f64).cast()
                ),
                0
            );
            assert_eq!((minimum, maximum), (2., 5.));
            let mut raw_bytes = 0;
            let mut max_electrons = 0;
            count_eer_bytes_and_electrons(
                (*reader).backend_handle.cast(),
                0,
                &mut raw_bytes,
                &mut max_electrons,
            );
            assert!(raw_bytes >= 4);
            assert_eq!(max_electrons, 8 * raw_bytes / 12);
            let mut decoded = [0_u8; 4];
            assert_eq!(tiff_read_section(reader, decoded.as_mut_ptr().cast(), 0), 0);
            assert_eq!(decoded, pixels);
            let mut floats = [0_f32; 4];
            assert_eq!(
                tiff_read_section_float(reader, floats.as_mut_ptr().cast(), 0),
                0
            );
            assert_eq!(floats, [3.0, 1.0, 4.0, 1.0]);
            ii_delete(reader);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn tiff_check_reads_separate_grayscale_planes_as_source_sections() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-separate-gray-{}.tif",
                std::process::id()
            ));
            let name = path.to_string_lossy().into_owned();
            let tiff_name = std::ffi::CString::new(name.clone()).unwrap();
            let writer = TIFFOpen(tiff_name.as_ptr(), c"w".as_ptr());
            assert!(!writer.is_null());
            assert_ne!(TIFFSetField(writer, TIFFTAG_IMAGEWIDTH, 2_u32), 0);
            assert_ne!(TIFFSetField(writer, TIFFTAG_IMAGELENGTH, 2_u32), 0);
            assert_ne!(TIFFSetField(writer, TIFFTAG_BITSPERSAMPLE, 8_i32), 0);
            assert_ne!(TIFFSetField(writer, TIFFTAG_SAMPLESPERPIXEL, 2_i32), 0);
            assert_ne!(
                TIFFSetField(writer, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK),
                0
            );
            assert_ne!(
                TIFFSetField(writer, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE),
                0
            );
            assert_ne!(TIFFSetField(writer, TIFFTAG_ROWSPERSTRIP, 2_u32), 0);
            let first_plane = [1_u8, 2, 3, 4];
            let second_plane = [5_u8, 6, 7, 8];
            assert_eq!(
                TIFFWriteEncodedStrip(
                    writer,
                    0,
                    first_plane.as_ptr().cast_mut().cast(),
                    first_plane.len() as isize,
                ),
                first_plane.len() as isize
            );
            assert_eq!(
                TIFFWriteEncodedStrip(
                    writer,
                    1,
                    second_plane.as_ptr().cast_mut().cast(),
                    second_plane.len() as isize,
                ),
                second_plane.len() as isize
            );
            TIFFClose(writer);

            let reader = ii_new();
            assert!(!reader.is_null());
            (*reader).filename = Some(name.clone());
            (*reader).fmode = "rb".into();
            (*reader).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(&name, "rb");
            assert_eq!(ii_tiff_check(reader), 0);
            assert_eq!(
                (
                    (*reader).nz,
                    (*reader).contig_samples,
                    (*reader).planes_per_image,
                    (*reader).rgb_samples,
                ),
                (2, 1, 2, 2)
            );
            let mut first = [0_u8; 4];
            let mut second = [0_u8; 4];
            assert_eq!(tiff_read_section(reader, first.as_mut_ptr().cast(), 0), 0);
            assert_eq!(tiff_read_section(reader, second.as_mut_ptr().cast(), 1), 0);
            assert_eq!(first, [3, 4, 1, 2]);
            assert_eq!(second, [7, 8, 5, 6]);
            ii_delete(reader);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn tiff_check_uses_the_largest_ifd_instead_of_a_leading_thumbnail() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-thumbnail-{}.tif",
                std::process::id()
            ));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = Some(name.clone());
            (*writer).nx = 1;
            (*writer).ny = 1;
            (*writer).nz = 2;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            assert_eq!(tiff_open_new(writer), 0);
            let mut thumbnail = [17_u8];
            assert_eq!(
                tiff_write_section(&mut *writer, &mut thumbnail, 1, 0, 0, -1),
                0
            );
            (*writer).nx = 2;
            (*writer).ny = 2;
            let mut science = [3_u8, 1, 4, 1];
            assert_eq!(
                tiff_write_section(&mut *writer, &mut science, 1, 0, 0, -1),
                0
            );
            tiff_close(&mut *writer);
            ii_delete(writer);

            let reader = ii_new();
            (*reader).filename = Some(name.clone());
            (*reader).fmode = "rb".into();
            (*reader).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(&name, "rb");
            assert_eq!(ii_tiff_check(reader), 0);
            assert_eq!(((*reader).nx, (*reader).ny, (*reader).nz), (2, 2, 1));
            assert_eq!((*reader).directory_nums.as_ref().unwrap()[0], 1);
            let mut decoded = [0_u8; 4];
            assert_eq!(tiff_read_section(reader, decoded.as_mut_ptr().cast(), 0), 0);
            assert_eq!(decoded, science);
            ii_delete(reader);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn tiff_section_callback_rejects_complex_data_before_writing() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-complex-reject-{}.tif",
                std::process::id()
            ));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = Some(name.clone());
            (*writer).nx = 1;
            (*writer).ny = 1;
            (*writer).nz = 1;
            (*writer).last_written_z = -1;
            (*writer).format = IIFORMAT_COMPLEX;
            (*writer).type_ = IITYPE_FLOAT;
            assert_eq!(tiff_open_new(writer), 0);
            let mut pixel = [1_f32, 2.];
            assert_eq!(
                ii_tiff_write_section_float(writer, pixel.as_mut_ptr().cast(), 0),
                -1
            );
            assert_eq!((*writer).last_written_z, -1);
            tiff_close(&mut *writer);
            ii_delete(writer);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn tiff_section_callback_rejects_nonsequential_real_tiff_section() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-sequence-reject-{}.tif",
                std::process::id()
            ));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            (*writer).filename = Some(name.clone());
            (*writer).nx = 1;
            (*writer).ny = 1;
            (*writer).nz = 1;
            (*writer).last_written_z = -1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            assert_eq!(tiff_open_new(writer), 0);
            let mut pixel = [1_u8];
            assert_eq!(
                ii_tiff_write_section(writer, pixel.as_mut_ptr().cast(), 1),
                -1
            );
            assert_eq!((*writer).last_written_z, -1);
            tiff_close(&mut *writer);
            ii_delete(writer);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn tiff_close_stores_source_final_min_max_for_new_luminance_file() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-close-minmax-{}.tif",
                std::process::id()
            ));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = Some(name.clone());
            (*writer).nx = 2;
            (*writer).ny = 2;
            (*writer).nz = 1;
            (*writer).new_file = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            assert_eq!(tiff_open_new(writer), 0);
            let mut pixels = [1_u8, 2, 3, 4];
            assert_eq!(
                tiff_write_section(&mut *writer, &mut pixels, 1, 0, 0, -1),
                0
            );
            (*writer).amin = 1.;
            (*writer).amax = 4.;
            tiff_close(&mut *writer);
            ii_delete(writer);

            let reader = ii_new();
            assert!(!reader.is_null());
            (*reader).filename = Some(name.clone());
            (*reader).fmode = "rb".into();
            (*reader).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(&name, "rb");
            assert_eq!(ii_tiff_check(reader), 0);
            let mut minimum = 0.0_f64;
            let mut maximum = 0.0_f64;
            assert_ne!(
                tiff_get_field(
                    reader,
                    TIFFTAG_SMINSAMPLEVALUE as i32,
                    (&mut minimum as *mut f64).cast()
                ),
                0
            );
            assert_ne!(
                tiff_get_field(
                    reader,
                    TIFFTAG_SMAXSAMPLEVALUE as i32,
                    (&mut maximum as *mut f64).cast()
                ),
                0
            );
            assert_eq!((minimum, maximum), (1., 4.));
            ii_delete(reader);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn native_libtiff_tiled_write_and_read_section_roundtrip() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path =
                std::env::temp_dir().join(format!("imod-rs-iitif-tile-{}.tif", std::process::id()));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            (*writer).filename = Some(name.clone());
            (*writer).nx = 2;
            (*writer).ny = 2;
            (*writer).nz = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            assert_eq!(tiff_open_new(writer), 0);
            let mut rows = 2;
            let mut tiles = 0;
            let mut tile_x = 2;
            assert_eq!(
                tiff_write_setup(writer, 1, 0, 0, -1, &mut rows, &mut tiles, &mut tile_x),
                0
            );
            let pixels = [2_u8, 7, 1, 8];
            assert_eq!(
                tiff_write_strip(writer, 0, pixels.as_ptr().cast_mut().cast()),
                0
            );
            tiff_write_finish(writer);
            tiff_close(&mut *writer);
            ii_delete(writer);
            let reader = ii_new();
            (*reader).filename = Some(name.clone());
            (*reader).fmode = "rb".into();
            (*reader).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(&name, "rb");
            assert_eq!(ii_tiff_check(reader), 0);
            let mut decoded = [0_u8; 4];
            assert_eq!(tiff_read_section(reader, decoded.as_mut_ptr().cast(), 0), 0);
            assert_eq!(decoded, pixels);
            ii_delete(reader);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn tiff_write_setup_caps_deflate_quality_at_source_tag_limit() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-zip-quality-{}.tif",
                std::process::id()
            ));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = Some(name.clone());
            (*writer).nx = 1;
            (*writer).ny = 1;
            (*writer).nz = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            assert_eq!(tiff_open_new(writer), 0);
            let mut rows = 0;
            let mut strips = 0;
            let mut tiles = 0;
            assert_eq!(
                tiff_write_setup(
                    writer,
                    IICOMPRESSION_ZIP,
                    0,
                    0,
                    100,
                    &mut rows,
                    &mut strips,
                    &mut tiles,
                ),
                0
            );
            let mut quality = 0_i32;
            assert_ne!(
                TIFFGetField(
                    (*writer).backend_handle.cast::<Tiff>(),
                    TIFFTAG_ZIPQUALITY,
                    &mut quality
                ),
                0
            );
            assert_eq!(quality, 9);
            tiff_close(&mut *writer);
            ii_delete(writer);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn lzw_parallel_write_falls_back_without_openmp_and_roundtrips() {
        // These expectations were captured from the reference with a single
        // thread.  `numOMPthreads` now returns the reference's OpenMP count
        // rather than a hard-coded 1, and that count selects work partitions
        // and pseudo-random seeds — so pin it here, otherwise the result
        // depends on the core count of whatever machine runs the suite.
        unsafe { std::env::set_var("OMP_NUM_THREADS", "1") };
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir()
                .join(format!("imod-rs-iitif-parallel-{}.tif", std::process::id()));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = Some(name.clone());
            (*writer).nx = 1024;
            (*writer).ny = 1024;
            (*writer).nz = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            assert_eq!(tiff_open_new(writer), 0);
            let pixels = (0..1024 * 1024)
                .map(|index| (index % 251) as u8)
                .collect::<Vec<_>>();
            let mut did_parallel = 0;
            assert_eq!(
                tiff_parallel_write(
                    writer,
                    pixels.as_ptr().cast_mut().cast(),
                    IICOMPRESSION_LZW,
                    0,
                    0,
                    -1,
                    &mut did_parallel
                ),
                0
            );
            // The source's numOMPthreads mapping is a no-OpenMP build here, so
            // the source eligibility path correctly selects its serial writer.
            assert_eq!(did_parallel, 0);
            tiff_close(&mut *writer);
            ii_delete(writer);
            let reader = ii_new();
            (*reader).filename = Some(name.clone());
            (*reader).fmode = "rb".into();
            (*reader).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(&name, "rb");
            assert_eq!(ii_tiff_check(reader), 0);
            let mut decoded = vec![0_u8; pixels.len()];
            assert_eq!(tiff_read_section(reader, decoded.as_mut_ptr().cast(), 0), 0);
            assert_eq!(decoded, pixels);
            ii_delete(reader);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn custom_digital_micrograph_scale_tag_registers_and_roundtrips() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let (version, minor) = tiff_version();
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-custom-tag-{}.tif",
                std::process::id()
            ));
            let name = path.to_string_lossy().into_owned();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = Some(name.clone());
            (*writer).nx = 1;
            (*writer).ny = 1;
            (*writer).nz = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            assert_eq!(tiff_open_new(writer), 0);
            let tif = (*writer).backend_handle.cast::<Tiff>();
            // The source deliberately declines to globally extend libtiff before
            // 4.5.  Current Debian libtiff is 4.3, so call the source callback
            // directly in that compatibility branch to exercise the ABI table.
            if version < 4 || (version == 4 && minor < 5) {
                register_custom_tiff_tags(tif);
            }
            assert_ne!(TIFFSetField(tif, TIFFTAG_DM_SCALE_0, 1.25_f64), 0);
            let mut value = 0.0_f64;
            assert_ne!(TIFFGetField(tif, TIFFTAG_DM_SCALE_0, &mut value), 0);
            assert_eq!(value, 1.25);
            let mut pixel = [0_u8];
            assert_eq!(
                tiff_write_section(&mut *writer, &mut pixel, 1, 0, 2_000_000, -1,),
                0
            );
            assert_ne!(TIFFGetField(tif, TIFFTAG_DM_SCALE_0, &mut value), 0);
            assert_eq!(value, 5.);
            let mut unit = core::ptr::null_mut::<u8>();
            assert_ne!(TIFFGetField(tif, TIFFTAG_DM_UINFO_UNIT_0, &mut unit), 0);
            assert_eq!(
                core::slice::from_raw_parts(unit, b"nanometer".len()),
                b"nanometer"
            );
            let mut datetime = core::ptr::null_mut::<u8>();
            assert_ne!(TIFFGetField(tif, TIFFTAG_DATETIME, &mut datetime), 0);
            let mut datetime_len = 0usize;
            while *datetime.add(datetime_len) != 0 {
                datetime_len += 1;
            }
            assert_eq!(datetime_len, 19);
            tiff_close(&mut *writer);
            ii_delete(writer);
            std::fs::remove_file(path).unwrap();
        }
    }
}
