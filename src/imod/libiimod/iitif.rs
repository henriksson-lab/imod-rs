//! Translation scaffold for `IMOD/libiimod/iitif.c`.
//!
//! The source owns TIFF-specific file opening, pixel conversion, EER decoding, and
//! serial/parallel TIFF writing.  Each function below deliberately corresponds to
//! one function in that C source; libtiff calls are filled in as its ABI is brought
//! into the crate.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::b3dutil::{
    b3d_error, b3d_shift_bytes, make_all_big_tiff, num_omp_threads,
};
use crate::imod::libcfshr::ilist::{ilist_append, ilist_delete, ilist_item, ilist_new};
use crate::imod::libcfshr::zoomdown::{select_zoom_filter, zoom_raw_filt_value};
use crate::imod::libiimod::iimage::{
    IIFILE_TIFF, IIFORMAT_COLORMAP, IIFORMAT_COMPLEX, IIFORMAT_LUMINANCE, IIFORMAT_RGB,
    IISTATE_BUSY, IISTATE_READY, IITYPE_BYTE, IITYPE_FLOAT, IITYPE_INT, IITYPE_SHORT, IITYPE_UBYTE,
    IITYPE_UINT, IITYPE_USHORT, ImodImageFile, MRSA_BYTE, MRSA_FLOAT, MRSA_NOPROC, MRSA_USHORT,
    ii_best_tile_size, ii_close, ii_delete, ii_make_buffer_convert_if_float, ii_new,
};
use crate::imod::libiimod::mrcfiles::{
    IIUNIT_4BIT_MODE, IIUNIT_HALF_XSIZE, MRC_LABEL_SIZE, MRC_MODE_BYTE, MRC_MODE_FLOAT,
    MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT, MRC_NLABELS, MrcHeader, PACKED_4BIT_MODE,
    PACKED_HALF_XSIZE, fix_title_padding, mrc_head_new, mrc_set_scale,
};
use core::ffi::{CStr, c_char, c_void};
use core::sync::atomic::{AtomicI32, AtomicPtr, Ordering};

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
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
    fn _TIFFfree(memory: *mut c_void);
    fn _TIFFmalloc(size: isize) -> *mut c_void;
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
static S_FILE_BUF_SIZE: AtomicI32 = AtomicI32::new(0);
static S_SETTING_UP_PARALLEL: AtomicI32 = AtomicI32::new(0);
static mut S_FILE_BUF: [*mut c_char; MAX_TIFF_THREADS] = [core::ptr::null_mut(); MAX_TIFF_THREADS];
static mut S_CUR_BUF_IND: [u64; MAX_TIFF_THREADS] = [0; MAX_TIFF_THREADS];
static mut S_MAX_BUF_IND: [u64; MAX_TIFF_THREADS] = [0; MAX_TIFF_THREADS];
static mut S_ROWS_PER_STRIP: i32 = 0;
static mut S_LINE_BYTES: i32 = 0;
static mut S_STRIP_BYTES: i32 = 0;
static mut S_X_TILE_SIZE: i32 = 0;
static mut S_LINES_DONE: i32 = 0;
static mut S_NUM_STRIPS: i32 = 0;
static mut S_ALREADY_INVERTED: i32 = 0;
static mut S_PIX_SIZE: i32 = 0;
static mut S_NUM_X_TILES: i32 = 0;
static mut S_TMP_BUF: *mut c_char = core::ptr::null_mut();
static mut S_DESCRIPTION: *mut c_char = core::ptr::null_mut();
static mut S_ALL_FILTERS: *mut i32 = core::ptr::null_mut();
static mut S_FILTER_PTRS: *mut *mut i32 = core::ptr::null_mut();
static mut S_FILT_X_START: *mut i32 = core::ptr::null_mut();
static mut S_FILT_Y_START: *mut i32 = core::ptr::null_mut();
static mut S_PARENT_EXTENDER: Option<unsafe extern "C" fn(*mut Tiff)> = None;
static S_AUGMENTED_TAGS: AtomicI32 = AtomicI32::new(0);
static mut S_XTIFF_FIELD_INFO: [TiffFieldInfo; 8] = [
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
    // The complete directory scan is a libtiff ABI operation.  Opening here retains
    // the source's format gate and lets TIFF report malformed directory chains.
    if in_file.is_null() || (*in_file).filename.is_null() {
        return IIERR_BAD_CALL;
    }
    // The source probes the two TIFF magic words before asking libtiff to open
    // the file.  Besides preserving its result distinction, this keeps a
    // non-TIFF MRC input from producing libtiff diagnostics while probing.
    let mut magic = [0u8; 4];
    libc::rewind((*in_file).fp);
    if libc::fread(magic.as_mut_ptr().cast(), 1, magic.len(), (*in_file).fp) != magic.len() {
        return IIERR_IO_ERROR;
    }
    let byte_order = u16::from_ne_bytes([magic[0], magic[1]]);
    if byte_order != 0x4949 && byte_order != 0x4d4d {
        return IIERR_NOT_FORMAT;
    }
    let version = u16::from_ne_bytes([magic[2], magic[3]]);
    if version != 0x002a && version != 0x2a00 && version != 0x002b && version != 0x2b00 {
        return IIERR_NOT_FORMAT;
    }
    libc::fclose((*in_file).fp);
    (*in_file).fp = core::ptr::null_mut();
    let tif = open_without_b_mode(in_file);
    if tif.is_null() {
        (*in_file).fp = libc::fopen((*in_file).filename, (*in_file).fmode.as_ptr());
        return IIERR_IO_ERROR;
    }
    (*in_file).header = tif.cast();
    (*in_file).fp = tif.cast();
    (*in_file).file = IIFILE_TIFF;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut bits = 8u16;
    let mut samples = 1u16;
    let mut sample_format = SAMPLEFORMAT_UINT as u16;
    let mut photometric = PHOTOMETRIC_MINISBLACK as u16;
    let mut planar_config = PLANARCONFIG_CONTIG as u16;
    let mut rows = 0u32;
    let mut compression = IICOMPRESSION_NONE as u16;
    // `iiTIFFCheck` obtains physical pixel sizes from the directory resolution
    // tags.  TIFF stores pixels per inch or centimetre while IMOD stores
    // Angstroms per pixel, hence the source's 1.e8 conversion (and 2.54 for
    // inches).  A missing Y resolution deliberately inherits X resolution.
    // Preserve the physical directory number for each uniform image in the
    // source stack.  Source `setMatchingDirectory` indexes this list, rather
    // than assuming an output section is the same as a TIFF directory number.
    let directories = ilist_new(core::mem::size_of::<i32>() as i32, 4);
    if directories.is_null() {
        tiff_close(in_file);
        return IIERR_MEMORY_ERR;
    }
    let mut directory = 0i32;
    loop {
        let mut next_width = 0u32;
        let mut next_height = 0u32;
        let mut next_bits = 8u16;
        let mut next_samples = 1u16;
        let mut next_format = SAMPLEFORMAT_UINT as u16;
        let mut next_photometric = PHOTOMETRIC_MINISBLACK as u16;
        let mut next_planar_config = PLANARCONFIG_CONTIG as u16;
        let mut next_rows = 0u32;
        let mut next_compression = IICOMPRESSION_NONE as u16;
        if TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &mut next_width) == 0
            || TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &mut next_height) == 0
            || TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &mut next_bits) == 0
        {
            ilist_delete(directories);
            tiff_close(in_file);
            return IIERR_NO_SUPPORT;
        }
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &mut next_samples);
        TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &mut next_photometric);
        TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &mut next_planar_config);
        TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &mut next_format);
        TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &mut next_rows);
        TIFFGetField(tif, TIFFTAG_COMPRESSION, &mut next_compression);
        // `iitif.c` makes a larger IFD the standard image and drops earlier
        // thumbnail directories from the section map.  It is not valid to
        // assume that directory zero is the science image.
        if next_width.saturating_mul(next_height) > width.saturating_mul(height) {
            width = next_width;
            height = next_height;
            bits = next_bits;
            samples = next_samples;
            sample_format = next_format;
            photometric = next_photometric;
            planar_config = next_planar_config;
            rows = next_rows;
            compression = next_compression;
            (*directories).size = 0;
        }
        if next_width == width
            && next_height == height
            && next_bits == bits
            && next_samples == samples
            && next_format == sample_format
            && next_photometric == photometric
            && next_planar_config == planar_config
            && ilist_append(directories, (&mut directory as *mut i32).cast()) != 0
        {
            ilist_delete(directories);
            tiff_close(in_file);
            return IIERR_MEMORY_ERR;
        }
        if next_width == width
            && next_height == height
            && (next_bits != bits
                || next_samples != samples
                || next_format != sample_format
                || next_photometric != photometric
                || next_planar_config != planar_config)
        {
            ilist_delete(directories);
            tiff_close(in_file);
            return IIERR_NO_SUPPORT;
        }
        if TIFFReadDirectory(tif) == 0 {
            break;
        }
        directory += 1;
    }
    (*in_file).tiff_compression = compression as i32;
    let selected_directory = *(ilist_item(directories, 0).cast::<i32>());
    TIFFSetDirectory(tif, selected_directory as u16);
    let mut x_resolution = 0.0f32;
    let mut y_resolution = 0.0f32;
    let mut resolution_unit = 0u16;
    let has_pixel_size = TIFFGetField(tif, TIFFTAG_XRESOLUTION, &mut x_resolution) != 0;
    TIFFGetField(tif, TIFFTAG_RESOLUTIONUNIT, &mut resolution_unit);
    if has_pixel_size && resolution_unit > 1 && x_resolution > 0.0 {
        if TIFFGetField(tif, TIFFTAG_YRESOLUTION, &mut y_resolution) == 0 {
            y_resolution = x_resolution;
        }
        let resolution_scale = if resolution_unit == RESUNIT_INCH as u16 {
            2.54e8
        } else {
            1.0e8
        };
        let x_pixel = resolution_scale / x_resolution;
        let pixel_limit = if libc::getenv(c"TIFF_RES_PIXEL_LIMIT".as_ptr()).is_null() {
            3.0
        } else {
            CStr::from_ptr(libc::getenv(c"TIFF_RES_PIXEL_LIMIT".as_ptr()))
                .to_string_lossy()
                .parse::<f32>()
                .unwrap_or(0.0)
        };
        if y_resolution > 0.0
            && ((*in_file).any_tiff_pix_size != 0 || x_pixel / 1.0e4 <= pixel_limit)
        {
            (*in_file).xscale = x_pixel;
            (*in_file).yscale = resolution_scale / y_resolution;
            (*in_file).zscale = x_pixel;
        }
    }
    TIFFSetDirectory(tif, 0);
    (*in_file).directory_nums = directories.cast();
    (*in_file).nx = width as i32;
    (*in_file).ny = height as i32;
    (*in_file).nz = (*directories).size;
    (*in_file).contig_samples = 1;
    (*in_file).planes_per_image = 1;
    (*in_file).rgb_samples = samples as i32;
    if photometric < PHOTOMETRIC_RGB as u16 {
        if planar_config == PLANARCONFIG_SEPARATE as u16 {
            (*in_file).planes_per_image = samples as i32;
        } else {
            (*in_file).contig_samples = samples as i32;
        }
        (*in_file).nz *= samples as i32;
    }
    (*in_file).tile_size_y = rows as i32;
    (*in_file).type_ = match (bits, sample_format) {
        (8, 2) => IITYPE_BYTE,
        (8, _) => IITYPE_UBYTE,
        (16, 2) => IITYPE_SHORT,
        (16, _) => IITYPE_USHORT,
        (32, 2) => IITYPE_INT,
        (32, 3) => IITYPE_FLOAT,
        _ => return IIERR_NO_SUPPORT,
    };
    (*in_file).format = if photometric == PHOTOMETRIC_RGB as u16 {
        IIFORMAT_RGB
    } else {
        IIFORMAT_LUMINANCE
    };
    (*in_file).mode = if samples >= 3 {
        MRC_MODE_RGB
    } else {
        match (*in_file).type_ {
            IITYPE_SHORT => MRC_MODE_SHORT,
            IITYPE_USHORT => MRC_MODE_USHORT,
            IITYPE_FLOAT => MRC_MODE_FLOAT,
            IITYPE_INT | IITYPE_UINT => -1,
            _ => MRC_MODE_BYTE,
        }
    };
    (*in_file).read_section = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
        unsafe extern "C" fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
    >(tiff_read_section));
    (*in_file).read_section_byte = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
        unsafe extern "C" fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
    >(tiff_read_section_byte));
    (*in_file).read_section_ushort = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
        unsafe extern "C" fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
    >(tiff_read_section_ushort));
    (*in_file).read_section_float = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
        unsafe extern "C" fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
    >(tiff_read_section_float));
    (*in_file).clean_up = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile),
        unsafe extern "C" fn(*mut ImodImageFile),
    >(tiff_delete));
    (*in_file).close = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile),
        unsafe extern "C" fn(*mut ImodImageFile),
    >(tiff_close));
    (*in_file).reopen = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile) -> i32,
        unsafe extern "C" fn(*mut ImodImageFile) -> i32,
    >(tiff_reopen));
    (*in_file).fill_mrc_header = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile, *mut MrcHeader) -> i32,
        unsafe extern "C" fn(*mut ImodImageFile, *mut MrcHeader) -> i32,
    >(tiff_fill_mrc_header));
    (*in_file).last_written_z = (*in_file).nz - 1;
    0
}
/// C `tiffReopen` (`iitif.c:679`).
pub unsafe fn tiff_reopen(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() {
        return IIERR_BAD_CALL;
    }
    let tif = open_without_b_mode(in_file);
    if tif.is_null() {
        return 1;
    }
    (*in_file).header_size = 8;
    (*in_file).section_skip = 0;
    (*in_file).header = tif.cast();
    (*in_file).fp = tif.cast();
    0
}
/// C `tiffClose` (`iitif.c:692`).
pub unsafe fn tiff_close(in_file: *mut ImodImageFile) {
    if in_file.is_null() {
        return;
    }
    let tif = (*in_file).header.cast::<Tiff>();
    if (*in_file).new_file != 0
        && (*in_file).format != IIFORMAT_RGB
        && !tif.is_null()
        && (*in_file).amax > (*in_file).amin
    {
        constrain_and_store_min_max(in_file);
    }
    if !tif.is_null() {
        TIFFClose(tif);
    }
    (*in_file).header = core::ptr::null_mut();
    (*in_file).fp = core::ptr::null_mut();
}
/// C `tiffDelete` (`iitif.c:708`).
pub unsafe fn tiff_delete(in_file: *mut ImodImageFile) {
    if !in_file.is_null() {
        tiff_close(in_file);
        if !(*in_file).directory_nums.is_null() {
            ilist_delete((*in_file).directory_nums.cast());
            (*in_file).directory_nums = core::ptr::null_mut();
        }
    }
}
/// C `tiffFillMrcHeader` (`iitif.c:715`).
pub unsafe fn tiff_fill_mrc_header(in_file: *mut ImodImageFile, hdata: *mut MrcHeader) -> i32 {
    if in_file.is_null() || hdata.is_null() {
        return IIERR_BAD_CALL;
    }
    mrc_head_new(
        &mut *hdata,
        (*in_file).nx,
        (*in_file).ny,
        (*in_file).nz,
        (*in_file).mode,
    );
    (*hdata).bytes_signed = if (*in_file).type_ == IITYPE_BYTE {
        1
    } else {
        0
    };
    (*hdata).amin = (*in_file).amin;
    (*hdata).amean = (*in_file).amean;
    (*hdata).amax = (*in_file).amax;
    mrc_set_scale(
        &mut *hdata,
        (*in_file).xscale as f64,
        (*in_file).yscale as f64,
        (*in_file).zscale as f64,
    );
    (*hdata).fp = (*in_file).fp.cast();
    (*hdata).packed4bits = (*in_file).packed4bits;
    (*hdata).half_floats = 0;
    if (*hdata).packed4bits == PACKED_4BIT_MODE {
        (*hdata).iiu_flags |= IIUNIT_4BIT_MODE;
    }
    if (*hdata).packed4bits == PACKED_HALF_XSIZE {
        (*hdata).iiu_flags |= IIUNIT_HALF_XSIZE;
    }
    let tif = (*in_file).header.cast::<Tiff>();
    let mut description: *mut c_char = core::ptr::null_mut();
    if !tif.is_null()
        && TIFFSetDirectory(tif, 0) != 0
        && TIFFGetField(tif, TIFFTAG_IMAGE_DESCRIPTION, &mut description) != 0
        && !description.is_null()
    {
        (*hdata).nlabl = 0;
        let description = CStr::from_ptr(description).to_bytes();
        let mut start_ind = 0;
        while (*hdata).nlabl < MRC_NLABELS as i32 && start_ind < description.len() {
            let mut end_ind = start_ind;
            while end_ind < description.len() && description[end_ind] != b'\n' {
                end_ind += 1;
            }
            if end_ind == description.len() {
                let dest = &mut (*hdata).labels[(*hdata).nlabl as usize];
                let count = (end_ind - start_ind).min(MRC_LABEL_SIZE);
                dest[..count].copy_from_slice(&description[start_ind..start_ind + count]);
                fix_title_padding(dest);
                (*hdata).nlabl += 1;
                break;
            }
            if end_ind != 0 && description[end_ind - 1] == b'\r' {
                end_ind -= 1;
            }
            if end_ind - start_ind > 0 {
                let dest = &mut (*hdata).labels[(*hdata).nlabl as usize];
                let count = (end_ind - start_ind).min(MRC_LABEL_SIZE);
                dest[..count].copy_from_slice(&description[start_ind..start_ind + count]);
                if end_ind - start_ind < MRC_LABEL_SIZE {
                    dest[end_ind - start_ind] = 0;
                }
                fix_title_padding(dest);
                (*hdata).nlabl += 1;
            }
            if description[end_ind] == b'\r' {
                end_ind += 1;
            }
            start_ind = end_ind + 1;
        }
    }
    0
}
/// C `tiffSyncFromMrcHeader` (`iitif.c:771`).
unsafe fn tiff_sync_from_mrc_header(in_file: *mut ImodImageFile, hdata: *mut MrcHeader) -> i32 {
    if in_file.is_null() || hdata.is_null() {
        return IIERR_BAD_CALL;
    }
    libc::free((*in_file).description.cast());
    (*in_file).description = core::ptr::null_mut();
    if (*hdata).nlabl == 0 {
        return 0;
    }
    (*in_file).description = libc::malloc((*hdata).nlabl as usize * (MRC_LABEL_SIZE + 1)).cast();
    if (*in_file).description.is_null() {
        return 1;
    }
    let mut out_ind = 0;
    for lab in 0..(*hdata).nlabl as usize {
        let mut true_len = 0;
        for ind in 0..MRC_LABEL_SIZE {
            if (*hdata).labels[lab][ind] == b'\n' {
                break;
            }
            if (*hdata).labels[lab][ind] != b' ' {
                true_len = ind + 1;
            }
        }
        if true_len != 0 {
            let start_ind = out_ind;
            for ind in 0..true_len {
                *(*in_file).description.add(out_ind) = (*hdata).labels[lab][ind] as c_char;
                out_ind += 1;
            }
            let description = core::slice::from_raw_parts_mut(
                (*in_file).description.cast::<u8>().add(start_ind),
                true_len,
            );
            if let Some(bit_ind) = description
                .windows(b"4 bits packed".len())
                .position(|part| part == b"4 bits packed")
            {
                if description
                    .windows(b"SerialEMCCD".len())
                    .any(|part| part == b"SerialEMCCD")
                {
                    description[bit_ind + 1] = b'-';
                }
            }
            *(*in_file).description.add(out_ind) = if lab == (*hdata).nlabl as usize - 1 {
                0
            } else {
                b'\n' as c_char
            };
            out_ind += 1;
        } else if lab == (*hdata).nlabl as usize - 1 {
            *(*in_file).description.add(out_ind) = 0;
            out_ind += 1;
        }
    }
    tiff_add_description((*in_file).description);
    0
}
/// C `tiffGetField` (`iitif.c:814`).
pub unsafe fn tiff_get_field(in_file: *mut ImodImageFile, tag: i32, value: *mut c_void) -> i32 {
    if in_file.is_null() {
        return -1;
    }
    if (*in_file).header.is_null() {
        let _ = tiff_reopen(in_file);
    }
    let tif = (*in_file).header.cast::<Tiff>();
    if tif.is_null() {
        return -1;
    }
    TIFFGetField(tif, tag as u32, value)
}
/// C `tiffGetArray` (`iitif.c:829`).
pub unsafe fn tiff_get_array(
    in_file: *mut ImodImageFile,
    tag: i32,
    count: *mut u32,
    value: *mut c_void,
) -> i32 {
    if !count.is_null() {
        *count = 0;
    }
    if in_file.is_null() {
        return -1;
    }
    if (*in_file).header.is_null() {
        let _ = tiff_reopen(in_file);
    }
    let tif = (*in_file).header.cast::<Tiff>();
    if tif.is_null() {
        return -1;
    }
    TIFFGetField(tif, tag as u32, count, value)
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
unsafe fn warning_handler(module: *const c_char, format: *const c_char, ap: *mut c_void) {
    let _ = (module, format, ap);
}
/// C `tiffFilterWarnings` (`iitif.c:879`).
pub fn tiff_filter_warnings() {
    if S_WARNINGS_SUPPRESSED.load(Ordering::SeqCst) == 0 {
        let _ = unsafe { TIFFSetWarningHandler(core::ptr::null_mut()) };
    }
}
/// C `tiffSetMapping` (`iitif.c:885`).
pub fn tiff_set_mapping(value: i32) {
    S_USE_MAPPING.store(value, Ordering::SeqCst);
}
/// C `tiffSetEERreadProperties` (`iitif.c:896`).
pub fn tiff_set_eer_read_properties(super_res: i32, autogroup: i32, flags: i32) {
    let super_res = super_res.clamp(
        S_MIN_EER_SUPER_RESOLUTION.load(Ordering::SeqCst),
        S_MAX_EER_SUPER_RESOLUTION.load(Ordering::SeqCst),
    );
    S_READ_EER_AS_SUPER_RES.store(super_res, Ordering::SeqCst);
    S_EER_FLAGS.store(flags, Ordering::SeqCst);
    S_IGNORE_BAD_EER_END.store(
        (flags & IIFLAG_IGNORE_BAD_EER_END != 0) as i32,
        Ordering::SeqCst,
    );
    S_AUTOGROUP_EER.store(autogroup, Ordering::SeqCst);
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
    if in_file.is_null() || (*in_file).filename.is_null() {
        return core::ptr::null_mut();
    }
    let mode = CStr::from_ptr((*in_file).fmode.as_ptr()).to_bytes();
    if mode.is_empty() {
        return core::ptr::null_mut();
    }
    if S_USE_MAPPING.load(Ordering::SeqCst) == 2 {
        S_USE_MAPPING.store(
            if libc::getenv(c"IMOD_NO_TIFF_MEM_MAP".as_ptr()).is_null() {
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
    TIFFOpen((*in_file).filename, open_mode.as_ptr().cast())
}
/// C `setMatchingDirectory` (`iitif.c:984`).
unsafe fn set_matching_directory(in_file: *mut ImodImageFile, dirnum: i32) -> i32 {
    if in_file.is_null() || (*in_file).header.is_null() || dirnum < 0 {
        return 1;
    }
    let directory = if (*in_file).directory_nums.is_null() {
        core::ptr::null_mut()
    } else {
        ilist_item((*in_file).directory_nums.cast(), dirnum).cast::<i32>()
    };
    if directory.is_null() {
        return 1;
    }
    (TIFFSetDirectory((*in_file).header.cast(), *directory as u16) == 0) as i32
}
/// C `closeWithError` (`iitif.c:994`).
unsafe fn close_with_error(in_file: *mut ImodImageFile, message: *const c_char) {
    if in_file.is_null() {
        return;
    }
    tiff_close(in_file);
}
/// C `countEERBytesAndElectrons` (`iitif.c:1005`).
unsafe fn count_eer_bytes_and_electrons(
    tif: *mut Tiff,
    is_7bit_eer: i32,
    raw_total_bytes: *mut i32,
    max_electrons: *mut i32,
) {
    if tif.is_null() {
        return;
    }
    let mut total = 0i32;
    for strip in 0..TIFFNumberOfStrips(tif) {
        total = total.saturating_add(TIFFRawStripSize(tif, strip) as i32);
    }
    if !raw_total_bytes.is_null() {
        *raw_total_bytes = total;
    }
    if !max_electrons.is_null() {
        *max_electrons = 8 * total / (if is_7bit_eer != 0 { 7 } else { 8 } + 4);
    }
}
/// C `ReadSection` (`iitif.c:1023`).
unsafe fn read_section(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
    convert: i32,
) -> i32 {
    if in_file.is_null() || buf.is_null() || (*in_file).axis == 2 {
        return -1;
    }
    let tif = (*in_file).header.cast::<Tiff>();
    let samples = (*in_file).contig_samples.max(1);
    let planes = (*in_file).planes_per_image.max(1);
    let row = in_section / (planes * samples);
    let plane = in_section % planes;
    let sample_offset = in_section % samples;
    if tif.is_null() || set_matching_directory(in_file, row) != 0 {
        return -1;
    }
    let mut rows = 0u32;
    let has_strips = TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &mut rows) != 0 && rows != 0;
    let xmin = (*in_file).llx;
    let ymin = (*in_file).lly;
    let xmax = if (*in_file).urx < 0 {
        (*in_file).nx - 1
    } else {
        (*in_file).urx
    };
    let ymax = if (*in_file).ury < 0 {
        (*in_file).ny - 1
    } else {
        (*in_file).ury
    };
    let xout = xmax + 1 - xmin;
    let mut pixsize = match (*in_file).type_ {
        IITYPE_SHORT | IITYPE_USHORT => 2,
        IITYPE_FLOAT | IITYPE_INT | IITYPE_UINT => 4,
        _ => 1,
    };
    if (*in_file).format == IIFORMAT_RGB {
        pixsize = (*in_file).rgb_samples;
    }
    let move_size = if convert == MRSA_FLOAT {
        4
    } else if convert == MRSA_USHORT {
        2
    } else if (*in_file).format == IIFORMAT_RGB || (*in_file).format == IIFORMAT_COLORMAP {
        3
    } else {
        pixsize
    };
    if !has_strips {
        let mut tile_width = 0u32;
        let mut tile_length = 0u32;
        if TIFFGetField(tif, TIFFTAG_TILEWIDTH, &mut tile_width) == 0
            || TIFFGetField(tif, TIFFTAG_TILELENGTH, &mut tile_length) == 0
        {
            return IIERR_NO_SUPPORT;
        }
        let tile_size = TIFFTileSize(tif);
        if tile_size <= 0 {
            return IIERR_IO_ERROR;
        }
        let mut tmp = vec![0u8; tile_size as usize];
        let x_tiles = ((*in_file).nx + tile_width as i32 - 1) / tile_width as i32;
        let y_tiles = ((*in_file).ny + tile_length as i32 - 1) / tile_length as i32;
        for y_tile in 0..y_tiles {
            for x_tile in 0..x_tiles {
                let start_y = ymin.max((*in_file).ny - tile_length as i32 * (y_tile + 1));
                let end_y = ymax.min((*in_file).ny - 1 - tile_length as i32 * y_tile);
                let start_x = xmin.max(x_tile * tile_width as i32);
                let end_x = xmax.min((x_tile + 1) * tile_width as i32 - 1);
                if start_y > end_y || start_x > end_x {
                    continue;
                }
                if TIFFReadEncodedTile(
                    tif,
                    (x_tile + y_tile * x_tiles + plane * x_tiles * y_tiles) as u32,
                    tmp.as_mut_ptr().cast(),
                    tile_size,
                ) < 0
                {
                    return IIERR_IO_ERROR;
                }
                for y in start_y..=end_y {
                    let row = (*in_file).ny - 1 - y - tile_length as i32 * y_tile;
                    let input = tmp.as_ptr().add(
                        ((row * tile_width as i32 + start_x - x_tile * tile_width as i32)
                            * samples
                            * pixsize
                            + sample_offset * pixsize) as usize,
                    );
                    let output = buf
                        .cast::<u8>()
                        .add((((y - ymin) * xout + start_x - xmin) * move_size) as usize);
                    let count = end_x + 1 - start_x;
                    if convert == MRSA_NOPROC {
                        core::ptr::copy_nonoverlapping(input, output, (count * pixsize) as usize);
                    } else {
                        copy_line(
                            input.cast_mut(),
                            output,
                            count,
                            convert,
                            pixsize,
                            (*in_file).type_,
                            (*in_file).format,
                            samples,
                            (*in_file).slope,
                            (*in_file).offset,
                            0,
                            0,
                            core::ptr::null_mut(),
                            core::ptr::null_mut(),
                            core::ptr::null_mut(),
                            core::ptr::null_mut(),
                        );
                    }
                }
            }
        }
        return 0;
    }
    let strip_size = TIFFStripSize(tif);
    if strip_size <= 0 {
        return IIERR_IO_ERROR;
    }
    let mut tmp = vec![0u8; strip_size as usize];
    let strips = TIFFNumberOfStrips(tif) as i32 / planes;
    for strip in 0..strips {
        if TIFFReadEncodedStrip(
            tif,
            (strip + plane * strips) as u32,
            tmp.as_mut_ptr().cast(),
            strip_size,
        ) < 0
        {
            return IIERR_IO_ERROR;
        }
        let strip_y_start = (*in_file).ny - rows as i32 * (strip + 1);
        let start = ymin.max(strip_y_start);
        let end = ymax.min((*in_file).ny - 1 - rows as i32 * strip);
        for y in start..=end {
            let row = (*in_file).ny - 1 - y - rows as i32 * strip;
            let input = tmp.as_ptr().add(
                ((row * (*in_file).nx + xmin) * samples * pixsize + sample_offset * pixsize)
                    as usize,
            );
            let output = buf
                .cast::<u8>()
                .add(((y - ymin) * xout * move_size) as usize);
            if convert == MRSA_NOPROC {
                core::ptr::copy_nonoverlapping(input, output, (xout * pixsize) as usize);
            } else {
                copy_line(
                    input.cast_mut(),
                    output,
                    xout,
                    convert,
                    pixsize,
                    (*in_file).type_,
                    (*in_file).format,
                    samples,
                    (*in_file).slope,
                    (*in_file).offset,
                    0,
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                );
            }
        }
    }
    0
}
/// C `decodeEERimage` (`iitif.c:1715`).
unsafe fn decode_eer_image(
    buf: *mut u8,
    positions: *mut i32,
    symbols: *mut u8,
    is_7bit_eer: i32,
    eer_image_pixels: i32,
    pos_limit: i32,
    num_electrons: *mut i32,
) -> i32 {
    let mut bit_pos = 0usize;
    let mut pixels = 0i32;
    let mut electrons = 0usize;
    if is_7bit_eer != 0 {
        loop {
            let first = bit_pos >> 3;
            let shift = bit_pos & 7;
            let chunk = u32::from_le_bytes([
                *buf.add(first),
                *buf.add(first + 1),
                *buf.add(first + 2),
                *buf.add(first + 3),
            ]) >> shift;
            for (entry, (jump, symbol)) in [
                (chunk & 127, (chunk >> 7) & 15),
                ((chunk >> 11) & 127, (chunk >> 18) & 15),
            ]
            .into_iter()
            .enumerate()
            {
                bit_pos += if entry == 0 { 7 } else { 11 };
                pixels += jump as i32;
                if pixels >= eer_image_pixels {
                    if !num_electrons.is_null() {
                        *num_electrons = electrons as i32
                    };
                    return if pixels != eer_image_pixels
                        && S_IGNORE_BAD_EER_END.load(Ordering::SeqCst) == 0
                    {
                        1
                    } else {
                        0
                    };
                }
                if jump != 127 {
                    *positions.add(electrons) = pixels;
                    *symbols.add(electrons) = (symbol as u8) ^ 0x0a;
                    electrons += 1;
                    pixels += 1;
                }
            }
        }
    }
    let mut pos = 0usize;
    while pos < pos_limit as usize {
        let p1 = *buf.add(pos);
        let s1 = (*buf.add(pos + 1) & 15) ^ 0x0a;
        let p2 = (*buf.add(pos + 1) >> 4) | (*buf.add(pos + 2) << 4);
        let s2 = (*buf.add(pos + 2) >> 4) ^ 0x0a;
        for (jump, symbol) in [(p1, s1), (p2, s2)] {
            pixels += jump as i32;
            if pixels >= eer_image_pixels {
                if !num_electrons.is_null() {
                    *num_electrons = electrons as i32
                };
                return if pixels != eer_image_pixels
                    && S_IGNORE_BAD_EER_END.load(Ordering::SeqCst) == 0
                {
                    1
                } else {
                    0
                };
            }
            if jump < 255 {
                *positions.add(electrons) = pixels;
                *symbols.add(electrons) = symbol;
                electrons += 1;
                pixels += 1;
            }
        }
        pos += 3;
    }
    if !num_electrons.is_null() {
        *num_electrons = electrons as i32;
    }
    if pixels != eer_image_pixels && S_IGNORE_BAD_EER_END.load(Ordering::SeqCst) == 0 {
        1
    } else {
        0
    }
}
/// C `convertEERpositions` (`iitif.c:1841`).
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
    if in_file.is_null() || positions.is_null() || symbols.is_null() || buf.is_null() {
        return;
    }
    let out_xsize = xmax + 1 - xmin;
    let out_ysize = ymax + 1 - ymin;
    let mut chip_xsize = (*in_file).nx;
    let mut chip_ysize = (*in_file).ny;
    if (*in_file).read_eer_as_super_res >= 0 {
        for _ in 0..(*in_file).read_eer_as_super_res {
            chip_xsize /= 2;
            chip_ysize /= 2;
        }
    } else {
        for _ in (*in_file).read_eer_as_super_res..0 {
            chip_xsize *= 2;
            chip_ysize *= 2;
        }
    }
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
    if (*in_file).antialias_eerfilter != 0 {
        if (*in_file).read_eer_as_super_res == 2 {
            let short_buf = buf.cast::<i16>();
            let gain = S_GAIN_REFERENCE.load(Ordering::SeqCst);
            let gain_scale = (*in_file).eerkernel_scale;
            for ind in 0..num_electrons as usize {
                let position = *positions.add(ind);
                let symbol = *symbols.add(ind) as i32;
                let gain_x = ((position % chip_xsize) << 2) | (symbol & 3);
                let gain_y =
                    (*in_file).ny - 1 - (((position / chip_xsize) << 2) | ((symbol & 12) >> 2));
                let x = gain_x - xmin;
                let y = gain_y - ymin;
                if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                    let value = if gain.is_null() {
                        gain_scale
                    } else {
                        (gain_scale as f32 * *gain.add((gain_x + gain_y * (*in_file).nx) as usize))
                            .round() as i32
                    };
                    *short_buf.add((x + y * out_xsize) as usize) += value as i16;
                }
            }
        } else if S_GAIN_REFERENCE.load(Ordering::SeqCst).is_null() {
            let red_fac = 2_i32.pow((2 - (*in_file).read_eer_as_super_res) as u32);
            let gain_scale = (*in_file).eerkernel_scale;
            let short_buf = buf.cast::<i16>();
            let y_super_offset = (*in_file).ny * red_fac - 1;
            for ind in 0..num_electrons as usize {
                let position = *positions.add(ind);
                let symbol = *symbols.add(ind) as i32;
                let x_super = ((position % chip_xsize) << 2) | (symbol & 3);
                let y_super =
                    y_super_offset - (((position / chip_xsize) << 2) | ((symbol & 12) >> 2));
                let x = x_super / red_fac - xmin;
                let y = y_super / red_fac - ymin;
                if x >= 2 && x < out_xsize - 2 && y >= 2 && y < out_ysize - 2 {
                    let phase_x = x_super % red_fac;
                    let phase_y = y_super % red_fac;
                    let start_x = if phase_x < red_fac / 2 { -2 } else { -1 };
                    let start_y = if phase_y < red_fac / 2 { -2 } else { -1 };
                    let center_x = (phase_x as f32 + 0.5) / red_fac as f32;
                    let center_y = (phase_y as f32 + 0.5) / red_fac as f32;
                    let mut total = 0.0f64;
                    for oy in 0..4 {
                        for ox in 0..4 {
                            total +=
                                zoom_raw_filt_value(start_x as f32 + ox as f32 + 0.5 - center_x)
                                    * zoom_raw_filt_value(
                                        start_y as f32 + oy as f32 + 0.5 - center_y,
                                    );
                        }
                    }
                    for oy in 0..4 {
                        for ox in 0..4 {
                            let value = (gain_scale as f64
                                * zoom_raw_filt_value(start_x as f32 + ox as f32 + 0.5 - center_x)
                                * zoom_raw_filt_value(start_y as f32 + oy as f32 + 0.5 - center_y)
                                / total)
                                .round() as i16;
                            let index =
                                (x + start_x + ox + (y + start_y + oy) * out_xsize) as usize;
                            *short_buf.add(index) += value;
                        }
                    }
                } else if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                    *short_buf.add((x + y * out_xsize) as usize) += gain_scale as i16;
                }
            }
        } else {
            let red_fac = 2_i32.pow((2 - (*in_file).read_eer_as_super_res) as u32);
            let gain = S_GAIN_REFERENCE.load(Ordering::SeqCst);
            let short_buf = buf.cast::<i16>();
            let y_super_offset = (*in_file).ny * red_fac - 1;
            let nx_gain = red_fac * (*in_file).nx;
            for ind in 0..num_electrons as usize {
                let position = *positions.add(ind);
                let symbol = *symbols.add(ind) as i32;
                let x_super = ((position % chip_xsize) << 2) | (symbol & 3);
                let y_super =
                    y_super_offset - (((position / chip_xsize) << 2) | ((symbol & 12) >> 2));
                let x = x_super / red_fac - xmin;
                let y = y_super / red_fac - ymin;
                let reference = *gain.add((x_super + y_super * nx_gain) as usize);
                if x >= 2 && x < out_xsize - 2 && y >= 2 && y < out_ysize - 2 {
                    let phase_x = x_super % red_fac;
                    let phase_y = y_super % red_fac;
                    let start_x = if phase_x < red_fac / 2 { -2 } else { -1 };
                    let start_y = if phase_y < red_fac / 2 { -2 } else { -1 };
                    let center_x = (phase_x as f32 + 0.5) / red_fac as f32;
                    let center_y = (phase_y as f32 + 0.5) / red_fac as f32;
                    let mut total = 0.0f64;
                    for oy in 0..4 {
                        for ox in 0..4 {
                            total +=
                                zoom_raw_filt_value(start_x as f32 + ox as f32 + 0.5 - center_x)
                                    * zoom_raw_filt_value(
                                        start_y as f32 + oy as f32 + 0.5 - center_y,
                                    );
                        }
                    }
                    for oy in 0..4 {
                        for ox in 0..4 {
                            let value =
                                ((zoom_raw_filt_value(start_x as f32 + ox as f32 + 0.5 - center_x)
                                    * zoom_raw_filt_value(
                                        start_y as f32 + oy as f32 + 0.5 - center_y,
                                    )
                                    / total)
                                    * reference as f64)
                                    .round() as i16;
                            let index =
                                (x + start_x + ox + (y + start_y + oy) * out_xsize) as usize;
                            *short_buf.add(index) += value;
                        }
                    }
                } else if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
                    *short_buf.add((x + y * out_xsize) as usize) += reference.round() as i16;
                }
            }
        }
        return;
    }
    let y_offset = (*in_file).ny - 1 - ymin;
    for ind in 0..num_electrons as usize {
        let position = *positions.add(ind);
        let symbol = *symbols.add(ind) as i32;
        let (x, y) = if (*in_file).read_eer_as_super_res == 2 {
            (
                ((position % chip_xsize) << 2) | (symbol & 3) - xmin,
                y_offset - (((position / chip_xsize) << 2) | ((symbol & 12) >> 2)),
            )
        } else if (*in_file).read_eer_as_super_res == 1 {
            (
                ((position % chip_xsize) << 1) | ((symbol & 2) >> 1) - xmin,
                y_offset - (((position / chip_xsize) << 1) | ((symbol & 8) >> 3)),
            )
        } else {
            (
                position % chip_xsize - xmin,
                y_offset - position / chip_xsize,
            )
        };
        if x >= 0 && x < out_xsize && y >= 0 && y < out_ysize {
            *buf.add((x + y * out_xsize) as usize) =
                (*buf.add((x + y * out_xsize) as usize)).wrapping_add(1);
        }
    }
}
/// C `cleanupFromEER` (`iitif.c:2148`).
unsafe fn cleanup_from_eer(
    tmp: *mut u8,
    map: *mut u8,
    positions: *mut i32,
    symbols: *mut u8,
    eer_buf: *mut u8,
) {
    if !tmp.is_null() {
        _TIFFfree(tmp.cast());
    }
    libc::free(map.cast());
    libc::free(positions.cast());
    libc::free(symbols.cast());
    libc::free(eer_buf.cast());
    S_EER_FLAGS.store(0, Ordering::SeqCst);
    if !S_ALL_FILTERS.is_null() {
        libc::free(S_ALL_FILTERS.cast());
        S_ALL_FILTERS = core::ptr::null_mut();
    }
    if !S_FILTER_PTRS.is_null() {
        libc::free(S_FILTER_PTRS.cast());
        S_FILTER_PTRS = core::ptr::null_mut();
    }
    if !S_FILT_X_START.is_null() {
        libc::free(S_FILT_X_START.cast());
        S_FILT_X_START = core::ptr::null_mut();
    }
    if !S_FILT_Y_START.is_null() {
        libc::free(S_FILT_Y_START.cast());
        S_FILT_Y_START = core::ptr::null_mut();
    }
}
/// C `copyLine` (`iitif.c:2171`).
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
    let _ = (first_4bits_map, second_4bits_map);
    if bdata.is_null() || obuf.is_null() || xout <= 0 {
        return;
    }
    if format == IIFORMAT_RGB {
        if convert == MRSA_NOPROC {
            core::ptr::copy_nonoverlapping(bdata, obuf, (xout * pixsize) as usize);
            return;
        }
        for ind in 0..xout as usize {
            let pixel = 0.3 * *bdata.add(ind * pixsize as usize) as f32
                + 0.59 * *bdata.add(ind * pixsize as usize + 1) as f32
                + 0.11 * *bdata.add(ind * pixsize as usize + 2) as f32;
            if convert == MRSA_FLOAT {
                *obuf.cast::<f32>().add(ind) = pixel;
            } else if convert == MRSA_USHORT {
                let value = if doscale != 0 && !map.is_null() {
                    *(map.cast::<u16>()).add((pixel + 0.499) as usize)
                } else {
                    (255. * pixel + 0.5) as u16
                };
                *obuf.cast::<u16>().add(ind) = value;
            } else {
                *obuf.add(ind) = if doscale != 0 && !map.is_null() {
                    *map.add((pixel + 0.499) as usize)
                } else {
                    (pixel + 0.5) as u8
                };
            }
        }
        return;
    }
    if format == IIFORMAT_COLORMAP && !colormap.is_null() {
        for ind in 0..xout as usize {
            let entry = *bdata.add(ind) as usize;
            if convert == MRSA_NOPROC {
                *obuf.add(3 * ind) = *colormap.add(entry);
                *obuf.add(3 * ind + 1) = *colormap.add(256 + entry);
                *obuf.add(3 * ind + 2) = *colormap.add(512 + entry);
            } else {
                let pixel = 0.3 * *colormap.add(entry) as f32
                    + 0.59 * *colormap.add(256 + entry) as f32
                    + 0.11 * *colormap.add(512 + entry) as f32;
                if convert == MRSA_FLOAT {
                    *obuf.cast::<f32>().add(ind) = pixel;
                } else if convert == MRSA_USHORT {
                    *obuf.cast::<u16>().add(ind) = (255. * pixel + 0.5) as u16;
                } else {
                    *obuf.add(ind) = (pixel + 0.5) as u8;
                }
            }
        }
        return;
    }
    if unpack_4bits != 0
        && pixsize == 1
        && !first_4bits_map.is_null()
        && !second_4bits_map.is_null()
    {
        let mut input = 0usize;
        let mut output = 0usize;
        if unpack_4bits > 1 {
            let value = *second_4bits_map.add(*bdata as usize);
            if convert == MRSA_FLOAT {
                *obuf.cast::<f32>().add(output) = value as f32;
            } else if convert == MRSA_USHORT {
                *obuf.cast::<u16>().add(output) = value as u16;
            } else {
                *obuf.add(output) = value;
            }
            input = 1;
            output = 1;
        }
        while output < xout as usize {
            let packed = *bdata.add(input);
            let first = *first_4bits_map.add(packed as usize);
            if convert == MRSA_FLOAT {
                *obuf.cast::<f32>().add(output) = first as f32;
            } else if convert == MRSA_USHORT {
                *obuf.cast::<u16>().add(output) = first as u16;
            } else {
                *obuf.add(output) = first;
            }
            output += 1;
            if output < xout as usize {
                let second = *second_4bits_map.add(packed as usize);
                if convert == MRSA_FLOAT {
                    *obuf.cast::<f32>().add(output) = second as f32;
                } else if convert == MRSA_USHORT {
                    *obuf.cast::<u16>().add(output) = second as u16;
                } else {
                    *obuf.add(output) = second;
                }
                output += 1;
            }
            input += 1;
        }
        return;
    }
    if pixsize == 2 {
        for ind in 0..xout as usize {
            let source = bdata.cast::<u16>().add(ind * samples as usize);
            if convert == MRSA_FLOAT {
                *obuf.cast::<f32>().add(ind) = if type_ == IITYPE_SHORT {
                    *source.cast::<i16>() as f32
                } else {
                    *source as f32
                };
            } else if convert == MRSA_USHORT {
                *obuf.cast::<u16>().add(ind) = if !map.is_null() {
                    *map.cast::<u16>().add(*source as usize)
                } else {
                    *source
                };
            } else if !map.is_null() {
                *obuf.add(ind) = *map.add(*source as usize);
            }
        }
        return;
    }
    if pixsize == 1 && type_ == IITYPE_BYTE && !map.is_null() {
        for ind in 0..xout as usize {
            let value = *map.add(*bdata.add(ind * samples as usize) as usize);
            if convert == MRSA_FLOAT {
                *obuf.cast::<f32>().add(ind) = value as f32;
            } else if convert == MRSA_USHORT {
                *obuf.cast::<u16>().add(ind) = *map
                    .cast::<u16>()
                    .add(*bdata.add(ind * samples as usize) as usize);
            } else {
                *obuf.add(ind) = value;
            }
        }
        return;
    }
    if pixsize == 4 && matches!(type_, IITYPE_INT | IITYPE_UINT) {
        for ind in 0..xout as usize {
            let value = if type_ == IITYPE_INT {
                *bdata.cast::<i32>().add(ind * samples as usize) as f32
            } else {
                *bdata.cast::<u32>().add(ind * samples as usize) as f32
            };
            if convert == MRSA_FLOAT {
                *obuf.cast::<f32>().add(ind) = value;
            } else if convert == MRSA_USHORT {
                *obuf.cast::<u16>().add(ind) = (slope * value + offset).clamp(0., 65535.) as u16;
            } else {
                *obuf.add(ind) = (slope * value + offset).clamp(0., 255.) as u8;
            }
        }
        return;
    }
    if pixsize == 4 && type_ == IITYPE_FLOAT {
        for ind in 0..xout as usize {
            let source = *bdata.cast::<f32>().add(ind * samples as usize);
            if convert == MRSA_FLOAT {
                *obuf.cast::<f32>().add(ind) = source;
            } else if convert == MRSA_USHORT {
                *obuf.cast::<u16>().add(ind) = (slope * source + offset).clamp(0., 65535.) as u16;
            } else {
                *obuf.add(ind) = (slope * source + offset).clamp(0., 255.) as u8;
            }
        }
        return;
    }
    for ind in 0..xout as usize {
        let source = *bdata.add(ind * samples.max(1) as usize * pixsize.max(1) as usize);
        let value = if !colormap.is_null() && format == 4 {
            let index = source as usize;
            (0.3 * (*colormap.add(index) as f32)
                + 0.59 * (*colormap.add(256 + index) as f32)
                + 0.11 * (*colormap.add(512 + index) as f32)) as u8
        } else if doscale != 0 && !map.is_null() {
            *map.add(source as usize)
        } else {
            source
        };
        if convert == MRSA_FLOAT {
            *(obuf.cast::<f32>().add(ind)) = if doscale != 0 {
                slope * value as f32 + offset
            } else {
                value as f32
            };
        } else if convert == MRSA_USHORT {
            *(obuf.cast::<u16>().add(ind)) = value as u16;
        } else {
            *obuf.add(ind) = value;
        }
    }
    let _ = (type_, unpack_4bits);
}
/// C `tiffReadSectionByte` (`iitif.c:2294`).
pub unsafe fn tiff_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
) -> i32 {
    read_section(in_file, buf, in_section, MRSA_BYTE)
}
/// C `tiffReadSectionUShort` (`iitif.c:2299`).
pub unsafe fn tiff_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
) -> i32 {
    read_section(in_file, buf, in_section, MRSA_USHORT)
}
/// C `tiffReadSectionFloat` (`iitif.c:2304`).
pub unsafe fn tiff_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
) -> i32 {
    read_section(in_file, buf, in_section, MRSA_FLOAT)
}
/// C `tiffReadSection` (`iitif.c:2309`).
pub unsafe fn tiff_read_section(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
) -> i32 {
    read_section(in_file, buf, in_section, MRSA_NOPROC)
}
/// C `bufReadProc` (`iitif.c:2329`).
unsafe extern "C" fn buf_read_proc(fd: *mut c_void, buf: *mut c_void, size: isize) -> isize {
    let fd = fd as usize;
    if fd >= MAX_TIFF_THREADS
        || size < 0
        || S_CUR_BUF_IND[fd].saturating_add(size as u64)
            > S_FILE_BUF_SIZE.load(Ordering::SeqCst) as u64
    {
        *libc::__errno_location() = libc::EINVAL;
        return -1;
    }
    core::ptr::copy_nonoverlapping(
        S_FILE_BUF[fd].add(S_CUR_BUF_IND[fd] as usize),
        buf.cast(),
        size as usize,
    );
    S_CUR_BUF_IND[fd] += size as u64;
    S_MAX_BUF_IND[fd] = S_MAX_BUF_IND[fd].max(S_CUR_BUF_IND[fd]);
    size
}
/// C `bufWriteProc` (`iitif.c:2343`).
unsafe extern "C" fn buf_write_proc(fd: *mut c_void, buf: *mut c_void, size: isize) -> isize {
    let fd = fd as usize;
    if fd >= MAX_TIFF_THREADS
        || size < 0
        || S_CUR_BUF_IND[fd].saturating_add(size as u64)
            >= S_FILE_BUF_SIZE.load(Ordering::SeqCst) as u64
    {
        *libc::__errno_location() = libc::EINVAL;
        return -1;
    }
    core::ptr::copy_nonoverlapping(
        buf.cast::<c_char>(),
        S_FILE_BUF[fd].add(S_CUR_BUF_IND[fd] as usize),
        size as usize,
    );
    S_CUR_BUF_IND[fd] += size as u64;
    S_MAX_BUF_IND[fd] = S_MAX_BUF_IND[fd].max(S_CUR_BUF_IND[fd]);
    size
}
/// C `bufSeekProc` (`iitif.c:2357`).
unsafe extern "C" fn buf_seek_proc(fd: *mut c_void, off: u64, whence: i32) -> u64 {
    let fd = fd as usize;
    if fd >= MAX_TIFF_THREADS {
        return u64::MAX;
    }
    let signed = off as i64;
    let new_pos = match whence {
        0 if signed >= 0 => signed as u64,
        1 if signed >= 0 || S_CUR_BUF_IND[fd] >= (-signed) as u64 => {
            S_CUR_BUF_IND[fd].wrapping_add_signed(signed)
        }
        2 => S_MAX_BUF_IND[fd],
        _ => return u64::MAX,
    };
    if new_pos >= S_FILE_BUF_SIZE.load(Ordering::SeqCst) as u64 {
        *libc::__errno_location() = libc::EINVAL;
        return u64::MAX;
    }
    S_CUR_BUF_IND[fd] = new_pos;
    S_MAX_BUF_IND[fd] = S_MAX_BUF_IND[fd].max(new_pos);
    new_pos
}
/// C `bufCloseProc` (`iitif.c:2393`).
unsafe extern "C" fn buf_close_proc(_fd: *mut c_void) -> i32 {
    0
}
/// C `bufSizeProc` (`iitif.c:2398`).
unsafe extern "C" fn buf_size_proc(fd: *mut c_void) -> u64 {
    unsafe { S_MAX_BUF_IND[fd as usize] }
}
/// C `tiffOpenNew` (`iitif.c:2406`).
pub unsafe fn tiff_open_new(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() || (*in_file).filename.is_null() {
        return IIERR_BAD_CALL;
    }
    augment_libtiff_with_custom_tags();
    let mut minor = 0;
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
    let use_w8 = tiff_version(&mut minor) > 3
        && ((((*in_file).nx as f64 * (*in_file).ny as f64)
            * (*in_file).nz as f64
            * pixel_size as f64)
            > 4.0e9
            || make_all_big_tiff() != 0);
    let mode = if use_w8 {
        c"w8".as_ptr()
    } else {
        c"w".as_ptr()
    };
    let tif = if S_SETTING_UP_PARALLEL.load(Ordering::SeqCst) <= 0 {
        TIFFOpen((*in_file).filename, mode)
    } else {
        let file_number = (S_SETTING_UP_PARALLEL.load(Ordering::SeqCst) - 1) as usize;
        TIFFClientOpen(
            (*in_file).filename,
            mode,
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
    (*in_file).header = tif.cast();
    (*in_file).fp = tif.cast();
    (*in_file).state = IISTATE_READY;
    (*in_file).clean_up = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile),
        unsafe extern "C" fn(*mut ImodImageFile),
    >(tiff_delete));
    (*in_file).close = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile),
        unsafe extern "C" fn(*mut ImodImageFile),
    >(tiff_close));
    (*in_file).fill_mrc_header = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile, *mut MrcHeader) -> i32,
        unsafe extern "C" fn(*mut ImodImageFile, *mut MrcHeader) -> i32,
    >(tiff_fill_mrc_header));
    (*in_file).sync_from_mrc_header = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile, *mut MrcHeader) -> i32,
        unsafe extern "C" fn(*mut ImodImageFile, *mut MrcHeader) -> i32,
    >(tiff_sync_from_mrc_header));
    (*in_file).write_section = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
        unsafe extern "C" fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
    >(ii_tiff_write_section));
    (*in_file).write_section_float = Some(core::mem::transmute::<
        unsafe fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
        unsafe extern "C" fn(*mut ImodImageFile, *mut c_char, i32) -> i32,
    >(ii_tiff_write_section_float));
    0
}
/// C `tiffWriteSection` (`iitif.c:2456`).
pub unsafe fn tiff_write_section(
    in_file: *mut ImodImageFile,
    buf: *mut c_void,
    compression: i32,
    inverted: i32,
    resolution: i32,
    quality: i32,
) -> i32 {
    if in_file.is_null() || (*in_file).header.is_null() {
        return IIERR_BAD_CALL;
    }
    let mut rows = 0;
    let mut number = 0;
    let mut tile_x = 0;
    let error = tiff_write_setup(
        in_file,
        compression,
        inverted,
        resolution,
        quality,
        &mut rows,
        &mut number,
        &mut tile_x,
    );
    if error != 0 {
        return error;
    }
    for strip in 0..S_NUM_STRIPS {
        let lines = S_ROWS_PER_STRIP.min((*in_file).ny - S_LINES_DONE);
        let source = if inverted != 0 {
            (buf as *mut u8).add(strip as usize * S_STRIP_BYTES as usize)
        } else {
            (buf as *mut u8)
                .add(((*in_file).ny - (S_LINES_DONE + lines)) as usize * S_LINE_BYTES as usize)
        };
        let error = tiff_write_strip(in_file, strip, source.cast());
        if error != 0 {
            return error;
        }
    }
    tiff_write_finish(in_file);
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
    if in_file.is_null() || (*in_file).header.is_null() {
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
    let tif = (*in_file).header.cast::<Tiff>();
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
        tiff_suppress_errors();
        TIFFSetField(tif, TIFFTAG_DM_SCALE_0, dm_pixel);
        TIFFSetField(tif, TIFFTAG_DM_SCALE_0 + 1, dm_pixel);
        TIFFSetField(tif, TIFFTAG_DM_ORIGIN_0, 0.0_f64);
        TIFFSetField(tif, TIFFTAG_DM_ORIGIN_0 + 1, 0.0_f64);
        TIFFSetField(tif, TIFFTAG_DM_UINFO_POWER_0, 1);
        TIFFSetField(tif, TIFFTAG_DM_UINFO_POWER_0 + 1, 1);
        TIFFSetField(
            tif,
            TIFFTAG_DM_UINFO_UNIT_0,
            if nano != 0 {
                c"nanometer".as_ptr()
            } else {
                c"micrometer".as_ptr()
            },
        );
        TIFFSetField(
            tif,
            TIFFTAG_DM_UINFO_UNIT_0 + 1,
            if nano != 0 {
                c"nanometer".as_ptr()
            } else {
                c"micrometer".as_ptr()
            },
        );
        tiff_restore_errors();
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
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits);
    if (*in_file).format != IIFORMAT_RGB {
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sample_format);
        if (*in_file).amax > (*in_file).amin {
            constrain_and_store_min_max(in_file);
        }
    }
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, photometric);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples);
    if !S_DESCRIPTION.is_null() && S_SETTING_UP_PARALLEL.load(Ordering::SeqCst) <= 0 {
        TIFFSetField(tif, TIFFTAG_IMAGE_DESCRIPTION, S_DESCRIPTION);
        libc::free(S_DESCRIPTION.cast());
        S_DESCRIPTION = core::ptr::null_mut();
    }
    S_PIX_SIZE = samples * bits / 8;
    if !tile_size_x.is_null() && *tile_size_x != 0 {
        let mut rows = *out_rows;
        let mut xtiles = 0;
        let mut ytiles = 0;
        ii_best_tile_size((*in_file).nx, &mut *tile_size_x, &mut xtiles, 16);
        ii_best_tile_size((*in_file).ny, &mut rows, &mut ytiles, 16);
        S_ROWS_PER_STRIP = rows;
        S_NUM_X_TILES = xtiles;
        S_NUM_STRIPS = ytiles;
        S_X_TILE_SIZE = *tile_size_x;
        S_LINE_BYTES = S_PIX_SIZE * S_X_TILE_SIZE;
        TIFFSetField(tif, TIFFTAG_TILEWIDTH, S_X_TILE_SIZE as u32);
        TIFFSetField(tif, TIFFTAG_TILELENGTH, S_ROWS_PER_STRIP as u32);
    } else {
        S_X_TILE_SIZE = 0;
        S_LINE_BYTES = S_PIX_SIZE * (*in_file).nx;
        let mut target = if compression != IICOMPRESSION_NONE {
            16384
        } else {
            8192
        };
        if (*in_file).ny > 4096 {
            target = (1 + (*in_file).ny / 4096) * S_LINE_BYTES;
        }
        if S_SETTING_UP_PARALLEL.load(Ordering::SeqCst) <= 0 {
            S_ROWS_PER_STRIP = ((target + S_LINE_BYTES / 2) / S_LINE_BYTES).max(1);
        }
        if compression == IICOMPRESSION_JPEG && S_ROWS_PER_STRIP % 8 != 0 {
            S_ROWS_PER_STRIP = if S_ROWS_PER_STRIP < 5 || S_ROWS_PER_STRIP % 8 > 4 {
                8 * ((S_ROWS_PER_STRIP + 7) / 8)
            } else {
                8 * (S_ROWS_PER_STRIP / 8)
            };
        }
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, S_ROWS_PER_STRIP as u32);
    }
    S_NUM_STRIPS = ((*in_file).ny + S_ROWS_PER_STRIP - 1) / S_ROWS_PER_STRIP;
    S_STRIP_BYTES = S_ROWS_PER_STRIP * S_LINE_BYTES;
    if !out_rows.is_null() {
        *out_rows = S_ROWS_PER_STRIP;
    }
    if !out_num.is_null() {
        *out_num = S_NUM_STRIPS;
    }
    let mut current_time = 0;
    libc::time(&mut current_time);
    let time_info = libc::localtime(&current_time);
    if !time_info.is_null() {
        let datetime = std::ffi::CString::new(format!(
            "{:04}:{:02}:{:02} {:02}:{:02}:{:02}",
            (*time_info).tm_year + 1900,
            (*time_info).tm_mon,
            (*time_info).tm_mday,
            (*time_info).tm_hour,
            (*time_info).tm_min,
            (*time_info).tm_sec,
        ))
        .unwrap();
        TIFFSetField(tif, TIFFTAG_DATETIME, datetime.as_ptr());
    }
    S_TMP_BUF = if inverted == 0 || S_X_TILE_SIZE != 0 {
        _TIFFmalloc(S_STRIP_BYTES as isize).cast()
    } else {
        core::ptr::null_mut()
    };
    if (inverted == 0 || S_X_TILE_SIZE != 0) && S_TMP_BUF.is_null() {
        return IIERR_MEMORY_ERR;
    }
    S_LINES_DONE = 0;
    S_ALREADY_INVERTED = inverted;
    0
}
/// C `tiffWriteStrip` (`iitif.c:2660`).
pub unsafe fn tiff_write_strip(in_file: *mut ImodImageFile, strip: i32, buf: *mut c_void) -> i32 {
    if in_file.is_null() || (*in_file).header.is_null() || buf.is_null() {
        return IIERR_BAD_CALL;
    }
    let lines = S_ROWS_PER_STRIP.min((*in_file).ny - S_LINES_DONE);
    b3d_shift_bytes(
        buf.cast(),
        buf.cast(),
        S_LINE_BYTES,
        lines,
        1,
        if (*in_file).type_ == IITYPE_BYTE {
            1
        } else {
            0
        },
    );
    if S_X_TILE_SIZE != 0 {
        for x_tile in 0..S_NUM_X_TILES {
            let x_offset = x_tile * S_X_TILE_SIZE * S_PIX_SIZE;
            let num_bytes =
                (S_X_TILE_SIZE.min((*in_file).nx - x_tile * S_X_TILE_SIZE)) * S_PIX_SIZE;
            for line in 0..lines {
                let source_line = if S_ALREADY_INVERTED != 0 {
                    line
                } else {
                    lines - line - 1
                };
                core::ptr::copy_nonoverlapping(
                    buf.cast::<c_char>()
                        .add((source_line * (*in_file).nx * S_PIX_SIZE + x_offset) as usize),
                    S_TMP_BUF.add((line * S_LINE_BYTES) as usize),
                    num_bytes as usize,
                );
            }
            if TIFFWriteEncodedTile(
                (*in_file).header.cast(),
                (x_tile + strip * S_NUM_X_TILES) as u32,
                S_TMP_BUF.cast(),
                S_STRIP_BYTES as isize,
            ) < 0
            {
                _TIFFfree(S_TMP_BUF.cast());
                S_TMP_BUF = core::ptr::null_mut();
                return IIERR_IO_ERROR;
            }
        }
        b3d_shift_bytes(
            buf.cast(),
            buf.cast(),
            S_LINE_BYTES,
            lines,
            -1,
            if (*in_file).type_ == IITYPE_BYTE {
                1
            } else {
                0
            },
        );
        S_LINES_DONE += lines;
        return 0;
    }
    let output = if S_ALREADY_INVERTED != 0 {
        buf.cast::<c_char>()
    } else {
        for line in 0..lines as usize {
            core::ptr::copy_nonoverlapping(
                buf.cast::<c_char>()
                    .add((lines as usize - line - 1) * S_LINE_BYTES as usize),
                S_TMP_BUF.add(line * S_LINE_BYTES as usize),
                S_LINE_BYTES as usize,
            );
        }
        S_TMP_BUF
    };
    if TIFFWriteEncodedStrip(
        (*in_file).header.cast(),
        strip as u32,
        output.cast(),
        (S_LINE_BYTES * lines) as isize,
    ) < 0
    {
        if S_ALREADY_INVERTED == 0 {
            _TIFFfree(S_TMP_BUF.cast());
            S_TMP_BUF = core::ptr::null_mut();
        }
        return IIERR_IO_ERROR;
    }
    b3d_shift_bytes(
        buf.cast(),
        buf.cast(),
        S_LINE_BYTES,
        lines,
        -1,
        if (*in_file).type_ == IITYPE_BYTE {
            1
        } else {
            0
        },
    );
    S_LINES_DONE += lines;
    0
}
/// C `tiffWriteFinish` (`iitif.c:2714`).
pub unsafe fn tiff_write_finish(in_file: *mut ImodImageFile) {
    if in_file.is_null() {
        return;
    }
    (*in_file).state = IISTATE_BUSY;
    if S_ALREADY_INVERTED == 0 && !S_TMP_BUF.is_null() {
        _TIFFfree(S_TMP_BUF.cast());
        S_TMP_BUF = core::ptr::null_mut();
    }
}
/// C `iiTiffWriteSection` (`iitif.c:2721`).
pub unsafe fn ii_tiff_write_section(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
) -> i32 {
    tiff_write_section_any(in_file, buf, in_section, 0)
}
/// C `iiTiffWriteSectionFloat` (`iitif.c:2726`).
pub unsafe fn ii_tiff_write_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
) -> i32 {
    tiff_write_section_any(in_file, buf, in_section, 1)
}
/// C `tiffAddDescription` (`iitif.c:2732`).
pub unsafe fn tiff_add_description(text: *const c_char) {
    if !S_DESCRIPTION.is_null() {
        libc::free(S_DESCRIPTION.cast());
    }
    S_DESCRIPTION = if text.is_null() {
        core::ptr::null_mut()
    } else {
        libc::strdup(text)
    };
}
/// C `tiffWriteSectionAny` (`iitif.c:2738`).
unsafe fn tiff_write_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
    if_float: i32,
) -> i32 {
    if in_file.is_null() || buf.is_null() {
        return IIERR_BAD_CALL;
    }
    if (*in_file).pad_left != 0 || (*in_file).pad_right != 0 {
        b3d_error(
            stderr,
            format_args!("ERROR: tiffWriteSectionAny - Cannot write from a subset of an array\n"),
        );
        return -1;
    }
    if in_section != (*in_file).last_written_z + 1 {
        b3d_error(
            stderr,
            format_args!(
                "ERROR: tiffWriteSectionAny - Can only write sequential sections to a TIFF file (last Z = {}, requested Z = {})\n",
                (*in_file).last_written_z,
                in_section
            ),
        );
        return -1;
    }
    if (*in_file).llx != 0
        || (*in_file).lly != 0
        || (*in_file).urx != (*in_file).nx - 1
        || (*in_file).ury != (*in_file).ny - 1
    {
        b3d_error(
            stderr,
            format_args!("ERROR: tiffWriteSectionAny - Can only write a whole section at once\n"),
        );
        return -1;
    }
    if (*in_file).format == IIFORMAT_COMPLEX {
        b3d_error(
            stderr,
            format_args!("ERROR: tiffWriteSectionAny - Cannot write complex data\n"),
        );
        return -1;
    }
    let mut inverted = 0;
    let use_buf = ii_make_buffer_convert_if_float(
        in_file,
        buf,
        if_float,
        &mut inverted,
        c"tiffWriteSectionAny".as_ptr(),
    );
    if use_buf.is_null() {
        return IIERR_MEMORY_ERR;
    }
    let resolution = if (*in_file).xscale == 1.0 {
        0
    } else {
        (if libc::getenv(c"IMOD_TIFF_RESOL_PER_INCH".as_ptr()).is_null() {
            1.0e8
        } else {
            -2.54e8
        } / (*in_file).xscale) as i32
    };
    let comp_env = libc::getenv(c"IMOD_TIFF_COMPRESSION".as_ptr());
    let compression = if comp_env.is_null() {
        1
    } else {
        libc::atoi(comp_env).max(1)
    };
    let quality_env = libc::getenv(c"IMOD_TIFF_QUALITY".as_ptr());
    let quality = if quality_env.is_null() {
        -1
    } else {
        libc::atoi(quality_env).max(1)
    };
    let result = tiff_write_section(
        in_file,
        use_buf.cast(),
        compression,
        inverted,
        resolution,
        quality,
    );
    if result == 0 {
        (*in_file).last_written_z += 1;
    }
    result
}
/// C `tiffVersion` (`iitif.c:2793`).
pub unsafe fn tiff_version(minor: *mut i32) -> i32 {
    if !minor.is_null() {
        *minor = 0;
    }
    let version = TIFFGetVersion();
    if version.is_null() {
        return 0;
    }
    let bytes = CStr::from_ptr(version).to_bytes();
    if bytes.windows(4).any(|word| word == b"IMOD") {
        return 0;
    }
    let Some(start) = bytes.windows(6).position(|word| word == b"ersion") else {
        return 0;
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
    if !minor.is_null() {
        *minor = values[1];
    }
    values[0]
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
    read_buf: *mut c_char,
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
    S_LINE_BYTES = (*in_file).nx * pixel_size;
    S_ROWS_PER_STRIP = ((strip_target + S_LINE_BYTES / 2) / S_LINE_BYTES).max(1);
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
    num_threads = num_threads
        .min(((*in_file).ny / S_ROWS_PER_STRIP) / 4)
        .max(1);
    let limit = libc::getenv(c"TIFF_WRITE_THREAD_LIMIT".as_ptr());
    if !limit.is_null() {
        let thread_limit = libc::atoi(limit);
        if thread_limit > 0 {
            num_threads = num_threads.min(thread_limit);
        }
    }
    num_threads = num_omp_threads(num_threads).min(MAX_TIFF_THREADS as i32);
    let mut minor = 0;
    if num_threads < 2
        || (*in_file).format == IIFORMAT_RGB
        || tiff_version(&mut minor) < 4
        || compression == IICOMPRESSION_JPEG
        || compression == IICOMPRESSION_NONE
    {
        return tiff_write_section(in_file, buf, compression, inverted, resolution, quality);
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
    if inverted == 0 && !S_TMP_BUF.is_null() {
        _TIFFfree(S_TMP_BUF.cast());
        S_TMP_BUF = core::ptr::null_mut();
    }
    S_SETTING_UP_PARALLEL.store(0, Ordering::SeqCst);
    if err != 0 {
        return err;
    }
    let num_strips = S_NUM_STRIPS;
    let rows_per_strip = S_ROWS_PER_STRIP;
    let line_bytes = S_LINE_BYTES;
    let strip_bytes = S_STRIP_BYTES;
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
    let mut thread_tmp_buf = [core::ptr::null_mut::<c_char>(); MAX_TIFF_THREADS];
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
        S_FILE_BUF[file] = _TIFFmalloc(S_FILE_BUF_SIZE.load(Ordering::SeqCst) as isize).cast();
        S_CUR_BUF_IND[file] = 0;
        S_MAX_BUF_IND[file] = 0;
        temp_files[file] = ii_new();
        if S_FILE_BUF[file].is_null() || temp_files[file].is_null() {
            err = IIERR_MEMORY_ERR;
            break;
        }
        let temporary_name = match CStr::from_ptr((*in_file).filename).to_str() {
            Ok(name) => std::ffi::CString::new(format!("{name}.{}.{}", libc::getpid(), file)).ok(),
            Err(_) => None,
        };
        let Some(temporary_name) = temporary_name else {
            err = IIERR_MEMORY_ERR;
            break;
        };
        (*temp_files[file]).filename = libc::strdup(temporary_name.as_ptr());
        if (*temp_files[file]).filename.is_null() {
            err = IIERR_MEMORY_ERR;
            break;
        }
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
        thread_num_strips[file] = S_NUM_STRIPS;
        thread_cum_lines[file] = if inverted != 0 {
            cumulative_lines
        } else {
            (*in_file).ny - cumulative_lines - number_lines
        };
        if inverted == 0 {
            thread_tmp_buf[file] = S_TMP_BUF;
        }
        cumulative_strips += S_NUM_STRIPS;
        cumulative_lines += number_lines;
    }
    S_SETTING_UP_PARALLEL.store(0, Ordering::SeqCst);
    if err == 0 && num_strips != cumulative_strips {
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
                    source,
                    source.cast(),
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
                    use_buf = thread_tmp_buf[file].cast();
                    for line in 0..number_lines as usize {
                        libc::memcpy(
                            (use_buf as *mut u8).add(line * line_bytes as usize).cast(),
                            source
                                .add((number_lines as usize - line - 1) * line_bytes as usize)
                                .cast(),
                            line_bytes as usize,
                        );
                    }
                }
                let written = TIFFWriteEncodedStrip(
                    (*temp_files[file]).header.cast(),
                    strip_index as u32,
                    use_buf,
                    (line_bytes * number_lines) as isize,
                );
                b3d_shift_bytes(
                    source,
                    source.cast(),
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
            if err != 0 {
                break;
            }
        }
    }
    if err == 0 {
        S_TMP_BUF = _TIFFmalloc((2 * strip_bytes) as isize).cast();
        if S_TMP_BUF.is_null() {
            err = IIERR_MEMORY_ERR;
        } else {
            let mut copied_strips = 0;
            for file in 0..num_threads as usize {
                for strip_index in 0..thread_num_strips[file] {
                    let bytes = TIFFReadRawStrip(
                        (*temp_files[file]).header.cast(),
                        strip_index as u32,
                        S_TMP_BUF.cast(),
                        (2 * strip_bytes) as isize,
                    );
                    if bytes <= 0
                        || TIFFWriteRawStrip(
                            (*in_file).header.cast(),
                            (strip_index + copied_strips) as u32,
                            S_TMP_BUF.cast(),
                            bytes,
                        ) <= 0
                    {
                        err = IIERR_IO_ERROR;
                        break;
                    }
                }
                copied_strips += thread_num_strips[file];
                if err != 0 {
                    break;
                }
            }
            _TIFFfree(S_TMP_BUF.cast());
            S_TMP_BUF = core::ptr::null_mut();
        }
    }
    if last_clean >= 0 {
        for index in 0..=last_clean as usize {
            if !temp_files[index].is_null() {
                ii_close(temp_files[index]);
                ii_delete(temp_files[index]);
                if inverted == 0 && !thread_tmp_buf[index].is_null() {
                    _TIFFfree(thread_tmp_buf[index].cast());
                }
            }
            if !S_FILE_BUF[index].is_null() {
                _TIFFfree(S_FILE_BUF[index].cast());
                S_FILE_BUF[index] = core::ptr::null_mut();
            }
        }
    }
    (*in_file).state = IISTATE_BUSY;
    if err == 0 && !S_DESCRIPTION.is_null() {
        libc::free(S_DESCRIPTION.cast());
        S_DESCRIPTION = core::ptr::null_mut();
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
    TIFFSetField((*in_file).header.cast(), TIFFTAG_SMINSAMPLEVALUE, minimum);
    TIFFSetField((*in_file).header.cast(), TIFFTAG_SMAXSAMPLEVALUE, maximum);
}
/// C `registerCustomTIFFTags` (`iitif.c:3191`).
unsafe extern "C" fn register_custom_tiff_tags(tif: *mut Tiff) {
    TIFFMergeFieldInfo(tif, (&raw const S_XTIFF_FIELD_INFO).cast(), 8);
    if let Some(parent) = S_PARENT_EXTENDER {
        parent(tif);
    }
}
/// C `augment_libtiff_with_custom_tags` (`iitif.c:3201`).
unsafe fn augment_libtiff_with_custom_tags() {
    if S_AUGMENTED_TAGS.load(Ordering::SeqCst) != 0 {
        return;
    }
    let mut minor = 0;
    let version = tiff_version(&mut minor);
    if version < 4 || (version == 4 && minor < 5) {
        return;
    }
    S_AUGMENTED_TAGS.store(1, Ordering::SeqCst);
    S_PARENT_EXTENDER = TIFFSetTagExtender(Some(register_custom_tiff_tags));
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
    fn cleanup_from_eer_releases_source_owned_buffers_and_resets_flags() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            S_EER_FLAGS.store(IIFLAG_SKIP_EER_DIRS, Ordering::SeqCst);
            S_ALL_FILTERS = libc::malloc(1).cast();
            S_FILTER_PTRS = libc::malloc(1).cast();
            S_FILT_X_START = libc::malloc(1).cast();
            S_FILT_Y_START = libc::malloc(1).cast();
            cleanup_from_eer(
                _TIFFmalloc(1).cast(),
                libc::malloc(1).cast(),
                libc::malloc(1).cast(),
                libc::malloc(1).cast(),
                libc::malloc(1).cast(),
            );
            assert_eq!(S_EER_FLAGS.load(Ordering::SeqCst), 0);
            assert!(
                S_ALL_FILTERS.is_null()
                    && S_FILTER_PTRS.is_null()
                    && S_FILT_X_START.is_null()
                    && S_FILT_Y_START.is_null()
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
            ii_delete(image);
        }
    }

    #[test]
    fn convert_eer_positions_gain_normalizes_reduced_resolution_filter_packet() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let mut width = 0;
            assert_eq!(select_zoom_filter(3, 0.5, &mut width), 0);
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
            // Source rounds each of the 16 weighted contributions independently.
            assert_eq!(output.iter().map(|v| *v as i32).sum::<i32>(), 199);
            S_GAIN_REFERENCE.store(core::ptr::null_mut(), Ordering::SeqCst);
            ii_delete(image);
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
            S_CUR_BUF_IND[0] = 1;
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
            S_CUR_BUF_IND[0] = 0;
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
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path =
                std::env::temp_dir().join(format!("imod-rs-iitif-{}.tif", std::process::id()));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = libc::strdup(name.as_ptr());
            (*writer).nx = 2;
            (*writer).ny = 2;
            (*writer).nz = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            (*writer).amin = 2.;
            (*writer).amax = 5.;
            assert_eq!(tiff_open_new(writer), 0);
            let pixels = [3_u8, 1, 4, 1];
            assert_eq!(
                tiff_write_section(writer, pixels.as_ptr().cast_mut().cast(), 1, 0, 0, -1),
                0
            );
            tiff_close(writer);
            ii_delete(writer);

            let reader = ii_new();
            assert!(!reader.is_null());
            (*reader).filename = libc::strdup(name.as_ptr());
            (*reader).fmode = [b'r' as i8, b'b' as i8, 0, 0];
            (*reader).fp = libc::fopen(name.as_ptr(), c"rb".as_ptr());
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
                (*reader).header.cast(),
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
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-separate-gray-{}.tif",
                std::process::id()
            ));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let writer = TIFFOpen(name.as_ptr(), c"w".as_ptr());
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
            (*reader).filename = libc::strdup(name.as_ptr());
            (*reader).fmode = [b'r' as i8, b'b' as i8, 0, 0];
            (*reader).fp = libc::fopen(name.as_ptr(), c"rb".as_ptr());
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
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-thumbnail-{}.tif",
                std::process::id()
            ));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = libc::strdup(name.as_ptr());
            (*writer).nx = 1;
            (*writer).ny = 1;
            (*writer).nz = 2;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            assert_eq!(tiff_open_new(writer), 0);
            let thumbnail = [17_u8];
            assert_eq!(
                tiff_write_section(writer, thumbnail.as_ptr().cast_mut().cast(), 1, 0, 0, -1),
                0
            );
            (*writer).nx = 2;
            (*writer).ny = 2;
            let science = [3_u8, 1, 4, 1];
            assert_eq!(
                tiff_write_section(writer, science.as_ptr().cast_mut().cast(), 1, 0, 0, -1),
                0
            );
            tiff_close(writer);
            ii_delete(writer);

            let reader = ii_new();
            (*reader).filename = libc::strdup(name.as_ptr());
            (*reader).fmode = [b'r' as i8, b'b' as i8, 0, 0];
            (*reader).fp = libc::fopen(name.as_ptr(), c"rb".as_ptr());
            assert_eq!(ii_tiff_check(reader), 0);
            assert_eq!(((*reader).nx, (*reader).ny, (*reader).nz), (2, 2, 1));
            assert_eq!(
                *(ilist_item((*reader).directory_nums.cast(), 0).cast::<i32>()),
                1
            );
            let mut decoded = [0_u8; 4];
            assert_eq!(tiff_read_section(reader, decoded.as_mut_ptr().cast(), 0), 0);
            assert_eq!(decoded, science);
            ii_delete(reader);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn tiff_section_callback_rejects_complex_data_before_writing() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-complex-reject-{}.tif",
                std::process::id()
            ));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = libc::strdup(name.as_ptr());
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
            tiff_close(writer);
            ii_delete(writer);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn tiff_section_callback_rejects_nonsequential_real_tiff_section() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-sequence-reject-{}.tif",
                std::process::id()
            ));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let writer = ii_new();
            (*writer).filename = libc::strdup(name.as_ptr());
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
            tiff_close(writer);
            ii_delete(writer);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn tiff_close_stores_source_final_min_max_for_new_luminance_file() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-close-minmax-{}.tif",
                std::process::id()
            ));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = libc::strdup(name.as_ptr());
            (*writer).nx = 2;
            (*writer).ny = 2;
            (*writer).nz = 1;
            (*writer).new_file = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            assert_eq!(tiff_open_new(writer), 0);
            let pixels = [1_u8, 2, 3, 4];
            assert_eq!(
                tiff_write_section(writer, pixels.as_ptr().cast_mut().cast(), 1, 0, 0, -1),
                0
            );
            (*writer).amin = 1.;
            (*writer).amax = 4.;
            tiff_close(writer);
            ii_delete(writer);

            let reader = ii_new();
            assert!(!reader.is_null());
            (*reader).filename = libc::strdup(name.as_ptr());
            (*reader).fmode = [b'r' as i8, b'b' as i8, 0, 0];
            (*reader).fp = libc::fopen(name.as_ptr(), c"rb".as_ptr());
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
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path =
                std::env::temp_dir().join(format!("imod-rs-iitif-tile-{}.tif", std::process::id()));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let writer = ii_new();
            (*writer).filename = libc::strdup(name.as_ptr());
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
            tiff_close(writer);
            ii_delete(writer);
            let reader = ii_new();
            (*reader).filename = libc::strdup(name.as_ptr());
            (*reader).fmode = [b'r' as i8, b'b' as i8, 0, 0];
            (*reader).fp = libc::fopen(name.as_ptr(), c"rb".as_ptr());
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
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-zip-quality-{}.tif",
                std::process::id()
            ));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = libc::strdup(name.as_ptr());
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
                    (*writer).header.cast::<Tiff>(),
                    TIFFTAG_ZIPQUALITY,
                    &mut quality
                ),
                0
            );
            assert_eq!(quality, 9);
            tiff_close(writer);
            ii_delete(writer);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn lzw_parallel_write_falls_back_without_openmp_and_roundtrips() {
        let _lock = TIFF_IO_TEST_LOCK.lock().unwrap();
        unsafe {
            let path = std::env::temp_dir()
                .join(format!("imod-rs-iitif-parallel-{}.tif", std::process::id()));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = libc::strdup(name.as_ptr());
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
            tiff_close(writer);
            ii_delete(writer);
            let reader = ii_new();
            (*reader).filename = libc::strdup(name.as_ptr());
            (*reader).fmode = [b'r' as i8, b'b' as i8, 0, 0];
            (*reader).fp = libc::fopen(name.as_ptr(), c"rb".as_ptr());
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
            let mut minor = 0;
            let version = tiff_version(&mut minor);
            let path = std::env::temp_dir().join(format!(
                "imod-rs-iitif-custom-tag-{}.tif",
                std::process::id()
            ));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let writer = ii_new();
            assert!(!writer.is_null());
            (*writer).filename = libc::strdup(name.as_ptr());
            (*writer).nx = 1;
            (*writer).ny = 1;
            (*writer).nz = 1;
            (*writer).format = IIFORMAT_LUMINANCE;
            (*writer).type_ = IITYPE_UBYTE;
            assert_eq!(tiff_open_new(writer), 0);
            let tif = (*writer).header.cast::<Tiff>();
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
            let pixel = [0_u8];
            assert_eq!(
                tiff_write_section(
                    writer,
                    pixel.as_ptr().cast_mut().cast(),
                    1,
                    0,
                    2_000_000,
                    -1,
                ),
                0
            );
            assert_ne!(TIFFGetField(tif, TIFFTAG_DM_SCALE_0, &mut value), 0);
            assert_eq!(value, 5.);
            let mut unit = core::ptr::null_mut::<c_char>();
            assert_ne!(TIFFGetField(tif, TIFFTAG_DM_UINFO_UNIT_0, &mut unit), 0);
            assert_eq!(CStr::from_ptr(unit).to_bytes(), b"nanometer");
            let mut datetime = core::ptr::null_mut::<c_char>();
            assert_ne!(TIFFGetField(tif, TIFFTAG_DATETIME, &mut datetime), 0);
            assert_eq!(CStr::from_ptr(datetime).to_bytes().len(), 19);
            tiff_close(writer);
            ii_delete(writer);
            std::fs::remove_file(path).unwrap();
        }
    }
}
