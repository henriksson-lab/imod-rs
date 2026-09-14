//! Translation of `IMOD/include/iimage.h` and `IMOD/libiimod/iimage.c`.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, ADOC_ZVALUE_NAME, adoc_get_collection_name, adoc_get_image_meta_info,
    adoc_get_num_collections, adoc_get_number_of_sections, adoc_get_section_name, adoc_new,
    adoc_read, adoc_set_current, adoc_transfer_section,
};
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{b3d_error, b3d_output_file_type};
use crate::imod::libcfshr::ilist::{
    Ilist, ilist_append, ilist_delete, ilist_dup, ilist_insert, ilist_item, ilist_new,
    ilist_remove, ilist_size,
};
use crate::imod::libiimod::halffloat::imnp_floatbuf_to_halfs;
use crate::imod::libiimod::hdf_imageio::{
    hdf_read_section_any as native_hdf_read_section_any,
    hdf_write_section_any as native_hdf_write_section_any,
    init_new_hdf_file as native_init_new_hdf_file,
};
use crate::imod::libiimod::iiadoc::ii_adoc_check;
use crate::imod::libiimod::iihdf::{
    hdf_write_dummy_section as native_hdf_write_dummy_section,
    hdf_write_global_adoc as native_hdf_write_global_adoc, ii_hdf_check as native_ii_hdf_check,
    ii_hdf_open_new, ii_reorder_hdf_stack, ii_test_if_hdf as native_ii_test_if_hdf,
};
use crate::imod::libiimod::iijpeg::{ii_jpeg_check, jpeg_open_new};
use crate::imod::libiimod::iilikemrc::ii_like_mrc_check;
use crate::imod::libiimod::iimrc::{
    ii_mrc_check, ii_mrc_load_pcoord, ii_mrc_mode_to_format_type, ii_mrc_open_new,
};
use crate::imod::libiimod::iishrmem::{
    IIFILE_SHR_MEM, SHR_MEM_NAME_TAG, ii_shr_mem_check_size, ii_shr_mem_open,
};
use crate::imod::libiimod::iitif::{
    MAX_TIFF_THREADS, ii_tiff_check, tiff_filter_warnings, tiff_get_max_eer_super_res,
    tiff_num_read_threads, tiff_open_new, tiff_parallel_read, tiff_set_eer_read_properties,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_HALF_FLOAT,
    MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader, mrc_complex_smin_smax, mrc_getdcsize, mrc_head_new,
};
use core::ffi::{c_char, c_void};
use core::sync::atomic::{AtomicI32, Ordering};

/// C `IISectionFunc` (`iimage.h`): `int (*)(ImodImageFile *, char *buf, int)`.
///
/// `buf` is a pixel buffer, not a string, and it stays a raw pointer: the
/// Fortran bridge in `unit_fileio.rs` receives the array from Fortran with no
/// length at all (`iiuReadSecPart` takes `array` and a line width), so there is
/// no slice to build there.  It is `*mut u8` rather than `*mut c_char` because
/// nothing about it is a C string.
pub type IiSectionFunc = Option<unsafe extern "C" fn(*mut ImodImageFile, *mut u8, i32) -> i32>;
pub type IiFileCheckFunction = Option<unsafe extern "C" fn(*mut ImodImageFile) -> i32>;
/// C `IIRawCheckFunction` (`iimage.h`).  A plain Rust `fn` pointer rather than
/// `extern "C"`: the table it lives in is private to `iilikemrc.c` and no C
/// caller ever installs an entry, so nothing needs the C calling convention —
/// and [`ImodFile`] is not an FFI type.
pub type IiRawCheckFunction = Option<unsafe fn(&mut ImodFile, &[u8], *mut RawImageInfo) -> i32>;

pub const IITYPE_UBYTE: i32 = 0;
pub const IITYPE_BYTE: i32 = 1;
pub const IITYPE_SHORT: i32 = 2;
pub const IITYPE_USHORT: i32 = 3;
pub const IITYPE_INT: i32 = 4;
pub const IITYPE_UINT: i32 = 5;
pub const IITYPE_FLOAT: i32 = 6;
pub const IIFORMAT_LUMINANCE: i32 = 0;
pub const IIFORMAT_RGB: i32 = 1;
pub const IIFORMAT_COMPLEX: i32 = 3;
pub const IIFORMAT_COLORMAP: i32 = 4;
pub const IIFILE_UNKNOWN: i32 = 0;
pub const IIFILE_DEFAULT: i32 = -1;
pub const IIFILE_TIFF: i32 = 1;
pub const IIFILE_MRC: i32 = 2;
pub const IIFILE_QIMAGE: i32 = 3;
pub const IIFILE_HDF: i32 = 5;
pub const IIFILE_JPEG: i32 = 6;
pub const IIFILE_ADOC: i32 = 7;
pub const IIFILE_RAW: i32 = 4;
pub const IIERR_BAD_CALL: i32 = -1;
pub const IIERR_NOT_FORMAT: i32 = 1;
pub const IIERR_IO_ERROR: i32 = 2;
pub const IIERR_NO_SUPPORT: i32 = 4;
pub const IIERR_QUITTING: i32 = 5;
pub const IISTATE_NOTINIT: i32 = 0;
pub const IISTATE_PARK: i32 = 1;
pub const IISTATE_READY: i32 = 2;
pub const IISTATE_UNUSED: i32 = 3;
pub const IISTATE_BUSY: i32 = 4;
pub const MRSA_NOPROC: i32 = 0;
pub const MRSA_BYTE: i32 = 1;
pub const MRSA_FLOAT: i32 = 2;
pub const MRSA_USHORT: i32 = 3;
static S_RW_CALL_COUNT: AtomicI32 = AtomicI32::new(0);
static S_ALLOW_MULTI_VOLUME: AtomicI32 = AtomicI32::new(0);
static mut S_OPENED_FILES: *mut Ilist = core::ptr::null_mut();
static mut S_CHECK_LIST: *mut Ilist = core::ptr::null_mut();
static mut S_II_TIFFS: [*mut ImodImageFile; MAX_TIFF_THREADS] =
    [core::ptr::null_mut(); MAX_TIFF_THREADS];
static mut S_MAX_TIFF_THREADS: i32 = 0;
static mut S_QUIT_CHECK_FUNC: Option<unsafe extern "C" fn(i32) -> i32> = None;

/// C `ImodImageFile` (`iimage.h`), in declaration order.
///
/// `Clone` stands in for `iiCopyOpen`'s `memcpy` (`iimage.c:541`): three of the
/// fields the source copies bitwise now own heap storage, so a bitwise copy
/// would give two structs the same buffer and a double free (NATIVE.md 4d,
/// disguise 3).  Cloning duplicates them instead, and the four fields the
/// source clears right afterwards are cleared just the same.
#[derive(Clone)]
#[repr(C)]
pub struct ImodImageFile {
    /// C `char *filename`, `strdup`ed from the caller and `free`d by
    /// `iiDelete`.  `None` is the source's NULL; the bytes never include a
    /// terminator, which is added only where a real C library is called.
    pub filename: Option<Vec<u8>>,
    /// C `char fmode[4]`: the `fopen` mode, `strncpy`ed with a length of 3, so
    /// the fourth byte is the NUL `iiNew`'s `memset` left there.
    pub fmode: [u8; 4],
    /// C `FILE *fp`.  See [`ImodFile`]; `None` is NULL, and the four places
    /// the source stores a non-file identity here use [`ImodFile::Token`].
    pub fp: Option<ImodFile>,
    /// C `char *description`: the TIFF `ImageDescription`, built out of the MRC
    /// labels by `tiffSyncFromMrcHeader` (`iitif.c:775-809`) and `free`d by
    /// `iiDelete`.  It holds the whole `nlabl * (MRC_LABEL_SIZE + 1)` buffer
    /// the source allocates, NUL bytes included, because that is what goes to
    /// libtiff.
    pub description: Option<Vec<u8>>,
    pub state: i32,
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub file: i32,
    pub format: i32,
    pub type_: i32,
    pub mode: i32,
    pub new_file: i32,
    pub amin: f32,
    pub amax: f32,
    pub amean: f32,
    pub rms: f32,
    pub xscale: f32,
    pub yscale: f32,
    pub zscale: f32,
    pub xtrans: f32,
    pub ytrans: f32,
    pub ztrans: f32,
    pub xrot: f32,
    pub yrot: f32,
    pub zrot: f32,
    pub time: i32,
    pub wave: i32,
    pub llx: i32,
    pub lly: i32,
    pub llz: i32,
    pub urx: i32,
    pub ury: i32,
    pub urz: i32,
    pub slope: f32,
    pub offset: f32,
    pub smin: f32,
    pub smax: f32,
    pub axis: i32,
    pub mirror_fft: i32,
    pub pad_left: i32,
    pub pad_right: i32,
    pub header_size: i32,
    pub section_skip: i32,
    pub has_piece_coords: i32,
    /// C `char *header`, which every user casts to something else: an
    /// `MrcHeader *` for MRC and shared memory, a `TfInfo *` for TIFF.  It is
    /// not a string and never was.
    pub header: *mut c_void,
    // `HANDLE` on Windows and an `int` file descriptor on POSIX (`iimage.h`).
    // Keeping a pointer-sized slot is required for the Windows mapping handle.
    pub shr_mem_file: isize,
    /// C `char *userData`: `iishrmem.c` stores the `mmap` base address here and
    /// does pointer arithmetic on it.  Not a string.
    pub user_data: *mut u8,
    pub user_flags: u32,
    pub user_count: i32,
    pub colormap: *mut u8,
    pub planes_per_image: i32,
    pub contig_samples: i32,
    pub multiple_sizes: i32,
    pub rgb_samples: i32,
    pub any_tiff_pix_size: i32,
    pub raw_palette_bytes: i32,
    pub tiff_compression: i32,
    pub read_eer_as_super_res: i32,
    pub num_frames_in_eerfile: i32,
    pub antialias_eerfilter: i32,
    pub eerkernel_scale: i32,
    pub tile_size_x: i32,
    pub tile_size_y: i32,
    pub last_written_z: i32,
    pub packed4bits: i32,
    pub fill_order: i32,
    pub half_floats: i32,
    pub directory_nums: *mut c_void,
    pub stack_set_list: *mut c_void,
    pub z_to_data_set_map: *mut i32,
    pub z_map_size: i32,
    /// C `char *datasetName`, `strdup`ed in `hdf_imageio.c:521` and passed to
    /// `H5Dopen`.
    pub dataset_name: Option<Vec<u8>>,
    // `iimage.h` declared this `int`, but it is passed to HDF5 as a `hid_t`.
    // Keep the native HDF identifier width so IDs from current HDF5 are not truncated.
    pub dataset_id: i64,
    pub dataset_is_open: i32,
    pub num_volumes: i32,
    pub ii_volumes: *mut *mut ImodImageFile,
    pub adoc_index: i32,
    pub global_adoc_index: i32,
    pub hdf_source: i32,
    // See `dataset_id`: this is semantically HDF5 `hid_t` storage.
    pub hdf_file_id: i64,
    pub z_chunk_size: i32,
    pub hdf_compression: i32,
    pub read_section: IiSectionFunc,
    pub read_section_byte: IiSectionFunc,
    pub read_section_ushort: IiSectionFunc,
    pub read_section_float: IiSectionFunc,
    pub write_section: IiSectionFunc,
    pub write_section_float: IiSectionFunc,
    pub clean_up: Option<unsafe extern "C" fn(*mut ImodImageFile)>,
    pub close: Option<unsafe extern "C" fn(*mut ImodImageFile)>,
    pub reopen: Option<unsafe extern "C" fn(*mut ImodImageFile) -> i32>,
    pub fill_mrc_header: Option<unsafe extern "C" fn(*mut ImodImageFile, *mut MrcHeader) -> i32>,
    pub sync_from_mrc_header:
        Option<unsafe extern "C" fn(*mut ImodImageFile, *mut MrcHeader) -> i32>,
    pub write_header: Option<unsafe extern "C" fn(*mut ImodImageFile) -> i32>,
}

impl Default for ImodImageFile {
    /// `iiNew` (`iimage.c:96`) `malloc`s and then `memset`s the whole struct to
    /// zero before setting its non-zero fields; this is that `memset`, written
    /// out because a struct carrying an `Option<ImodFile>` cannot be produced
    /// by `mem::zeroed` (NATIVE.md 4b).
    fn default() -> ImodImageFile {
        ImodImageFile {
            filename: None,
            fmode: [0; 4],
            fp: None,
            description: None,
            state: 0,
            nx: 0,
            ny: 0,
            nz: 0,
            file: 0,
            format: 0,
            type_: 0,
            mode: 0,
            new_file: 0,
            amin: 0.,
            amax: 0.,
            amean: 0.,
            rms: 0.,
            xscale: 0.,
            yscale: 0.,
            zscale: 0.,
            xtrans: 0.,
            ytrans: 0.,
            ztrans: 0.,
            xrot: 0.,
            yrot: 0.,
            zrot: 0.,
            time: 0,
            wave: 0,
            llx: 0,
            lly: 0,
            llz: 0,
            urx: 0,
            ury: 0,
            urz: 0,
            slope: 0.,
            offset: 0.,
            smin: 0.,
            smax: 0.,
            axis: 0,
            mirror_fft: 0,
            pad_left: 0,
            pad_right: 0,
            header_size: 0,
            section_skip: 0,
            has_piece_coords: 0,
            header: core::ptr::null_mut(),
            shr_mem_file: 0,
            user_data: core::ptr::null_mut(),
            user_flags: 0,
            user_count: 0,
            colormap: core::ptr::null_mut(),
            planes_per_image: 0,
            contig_samples: 0,
            multiple_sizes: 0,
            rgb_samples: 0,
            any_tiff_pix_size: 0,
            raw_palette_bytes: 0,
            tiff_compression: 0,
            read_eer_as_super_res: 0,
            num_frames_in_eerfile: 0,
            antialias_eerfilter: 0,
            eerkernel_scale: 0,
            tile_size_x: 0,
            tile_size_y: 0,
            last_written_z: 0,
            packed4bits: 0,
            fill_order: 0,
            half_floats: 0,
            directory_nums: core::ptr::null_mut(),
            stack_set_list: core::ptr::null_mut(),
            z_to_data_set_map: core::ptr::null_mut(),
            z_map_size: 0,
            dataset_name: None,
            dataset_id: 0,
            dataset_is_open: 0,
            num_volumes: 0,
            ii_volumes: core::ptr::null_mut(),
            adoc_index: 0,
            global_adoc_index: 0,
            hdf_source: 0,
            hdf_file_id: 0,
            z_chunk_size: 0,
            hdf_compression: 0,
            read_section: None,
            read_section_byte: None,
            read_section_ushort: None,
            read_section_float: None,
            write_section: None,
            write_section_float: None,
            clean_up: None,
            close: None,
            reopen: None,
            fill_mrc_header: None,
            sync_from_mrc_header: None,
            write_header: None,
        }
    }
}

/// C `RawImageInfo` (`iimage.h`), in declaration order.
#[repr(C)]
pub struct RawImageInfo {
    pub type_: i32,
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub swap_bytes: i32,
    pub header_size: i32,
    pub amin: f32,
    pub amax: f32,
    pub scan_min_max: i32,
    pub all_match: i32,
    pub section_skip: i32,
    pub y_inverted: i32,
    pub pixel: f32,
    pub z_pixel: f32,
}

/// C `LineProcData` (`iimage.h`), in declaration order.
#[repr(C)]
pub struct LineProcData {
    pub x_start: i32,
    pub x_end: i32,
    pub convert: i32,
    pub bdata: *mut u8,
    pub buf: *mut u8,
    pub bufp: *mut u8,
    pub usbufp: *mut u16,
    pub fbufp: *mut f32,
    pub fft: *mut u8,
    pub usfft: *mut u16,
    pub map: *mut u8,
    pub usmap: *mut u16,
    pub byte: i32,
    pub to_short: i32,
    pub map_sbytes: i32,
    pub do_scale: i32,
    pub need_data: i32,
    pub type_: i32,
    pub x_dimension: i32,
    pub xsize: i32,
    pub delta_y_sign: i32,
    pub pix_index: u32,
    pub cz: i32,
    pub read_y: i32,
    pub line: i32,
    pub ymin: i32,
    pub ymax: i32,
    pub y_start: i32,
    pub im_ymin: i32,
    pub im_ymax: i32,
    pub im_xsize: i32,
    pub toggle_y: i32,
    pub seek_end_y: i32,
    pub pix_size: i32,
    pub swapped: i32,
    pub packed4bits: i32,
    pub half_floats: i32,
    pub bytes_since_check: i32,
}

pub unsafe fn init_check_list() -> i32 {
    if !S_CHECK_LIST.is_null() {
        return 0;
    }
    S_CHECK_LIST = ilist_new(core::mem::size_of::<IiFileCheckFunction>() as i32, 6)
        .map_or(core::ptr::null_mut(), Box::into_raw);
    if S_CHECK_LIST.is_null() {
        return 1;
    }
    for func in [
        Some(core::mem::transmute::<
            unsafe fn(*mut ImodImageFile) -> i32,
            unsafe extern "C" fn(*mut ImodImageFile) -> i32,
        >(ii_tiff_check)),
        Some(ii_mrc_check),
        Some(core::mem::transmute::<
            unsafe extern "C" fn(*mut crate::imod::libiimod::iilikemrc::ImodImageFile) -> i32,
            unsafe extern "C" fn(*mut ImodImageFile) -> i32,
        >(ii_like_mrc_check)),
        Some(core::mem::transmute::<
            unsafe fn(*mut ImodImageFile) -> i32,
            unsafe extern "C" fn(*mut ImodImageFile) -> i32,
        >(native_ii_hdf_check)),
        Some(ii_jpeg_check),
        Some(ii_adoc_check),
    ] {
        ilist_append(
            &mut *S_CHECK_LIST,
            core::slice::from_raw_parts(
                (&raw const func).cast::<u8>(),
                core::mem::size_of::<IiFileCheckFunction>(),
            ),
        );
    }
    tiff_filter_warnings();
    0
}
pub unsafe fn ii_add_check_function(func: IiFileCheckFunction) {
    if init_check_list() == 0 {
        ilist_append(
            &mut *S_CHECK_LIST,
            core::slice::from_raw_parts(
                (&raw const func).cast::<u8>(),
                core::mem::size_of::<IiFileCheckFunction>(),
            ),
        );
    }
}
pub unsafe fn ii_insert_check_function(func: IiFileCheckFunction, index: i32) {
    if init_check_list() != 0 {
        return;
    }
    if index < ilist_size(S_CHECK_LIST.as_ref()) {
        ilist_insert(
            &mut *S_CHECK_LIST,
            core::slice::from_raw_parts(
                (&raw const func).cast::<u8>(),
                core::mem::size_of::<IiFileCheckFunction>(),
            ),
            index,
        );
    } else {
        ilist_append(
            &mut *S_CHECK_LIST,
            core::slice::from_raw_parts(
                (&raw const func).cast::<u8>(),
                core::mem::size_of::<IiFileCheckFunction>(),
            ),
        );
    }
}
pub unsafe fn ii_delete_check_list() {
    if !S_CHECK_LIST.is_null() {
        ilist_delete(Some(Box::from_raw(S_CHECK_LIST)));
        S_CHECK_LIST = core::ptr::null_mut();
    }
}
/// Matches C `iiRegisterQuitCheck(int (*)(int))` (`iimage.c:121`).
pub unsafe fn ii_register_quit_check(func: Option<unsafe extern "C" fn(i32) -> i32>) {
    unsafe { S_QUIT_CHECK_FUNC = func };
}
/// Matches C `iiCheckForQuit(int)` (`iimage.c:130`).
pub unsafe fn ii_check_for_quit(param: i32) -> i32 {
    if let Some(func) = unsafe { S_QUIT_CHECK_FUNC } {
        if unsafe { func(param) } != 0 {
            return IIERR_QUITTING;
        }
    }
    0
}
/// Matches C `iiNew(void)` (`iimage.c:141`).
///
/// `iiNew` (`iimage.c:143`) `malloc`s and `memset`s; this `Box`es a
/// [`ImodImageFile::default`], which is the same all-zero starting point.  It
/// must be a `Box` and not a `malloc`: the moment `fp` became a non-`Copy`
/// `Option<ImodFile>`, `(*ofile).fp = None` on uninitialised memory stopped
/// being a store and became a *drop of garbage*, which segfaults in
/// `Rc<File>::drop` on the first command.  The struct is leaked back out as a
/// raw pointer because every caller still holds one; `iiDelete` reclaims it
/// with `Box::from_raw`.
pub fn ii_new() -> *mut ImodImageFile {
    let ofile = Box::into_raw(Box::new(ImodImageFile::default()));
    unsafe {
        (*ofile).xscale = 1.0;
        (*ofile).yscale = 1.0;
        (*ofile).zscale = 1.0;
        (*ofile).slope = 1.0;
        (*ofile).smax = 255.0;
        (*ofile).axis = 3;
        (*ofile).mirror_fft = 0;
        (*ofile).any_tiff_pix_size = 0;
        (*ofile).raw_palette_bytes = 0;
        (*ofile).tiff_compression = 1;
        (*ofile).format = IIFILE_UNKNOWN;
        (*ofile).fp = None;
        (*ofile).read_section = None;
        (*ofile).read_section_byte = None;
        (*ofile).read_section_ushort = None;
        (*ofile).read_section_float = None;
        (*ofile).write_section = None;
        (*ofile).write_section_float = None;
        (*ofile).fill_mrc_header = None;
        (*ofile).sync_from_mrc_header = None;
        (*ofile).write_header = None;
        (*ofile).clean_up = None;
        (*ofile).reopen = None;
        (*ofile).close = None;
        (*ofile).write_section = None;
        (*ofile).colormap = core::ptr::null_mut();
        (*ofile).user_data = core::ptr::null_mut();
        (*ofile).shr_mem_file = 0;
        (*ofile).llx = 0;
        (*ofile).lly = 0;
        (*ofile).llz = 0;
        (*ofile).urx = -1;
        (*ofile).ury = -1;
        (*ofile).urz = -1;
        (*ofile).pad_left = 0;
        (*ofile).pad_right = 0;
        (*ofile).nx = 0;
        (*ofile).ny = 0;
        (*ofile).nz = 0;
        (*ofile).rms = -1.0;
        (*ofile).last_written_z = -1;
        (*ofile).packed4bits = 0;
        (*ofile).half_floats = 0;
        (*ofile).read_eer_as_super_res = 0;
        (*ofile).num_frames_in_eerfile = 0;
        (*ofile).antialias_eerfilter = 0;
        (*ofile).directory_nums = core::ptr::null_mut();
        (*ofile).adoc_index = -1;
        (*ofile).global_adoc_index = -1;
        (*ofile).stack_set_list = core::ptr::null_mut();
        (*ofile).z_to_data_set_map = core::ptr::null_mut();
        (*ofile).dataset_name = None;
        (*ofile).ii_volumes = core::ptr::null_mut();
        (*ofile).num_volumes = 0;
        (*ofile).hdf_compression = -1;
    }
    ofile
}

/// Matches C `iiInit(ImodImageFile *, int, int, int, int, int, int)` (`iimage.c:215`).
pub unsafe fn ii_init(
    image_file: *mut ImodImageFile,
    x_size: i32,
    y_size: i32,
    z_size: i32,
    file: i32,
    format: i32,
    type_: i32,
) -> i32 {
    if image_file.is_null() {
        return -1;
    }
    unsafe {
        (*image_file).nx = x_size;
        (*image_file).ny = y_size;
        (*image_file).nz = z_size;
        (*image_file).file = file;
        (*image_file).format = format;
        (*image_file).type_ = type_;
    }
    0
}
/// C `iiOpen`.  Format probing is deliberately kept in the format units; this
/// common-unit portion owns the file and registers the returned descriptor.
pub unsafe fn ii_open(filename: &[u8], mode: &str) -> *mut ImodImageFile {
    if mode.contains('w') {
        return ii_open_new(filename, mode, IIFILE_DEFAULT);
    }
    if filename.starts_with(SHR_MEM_NAME_TAG) {
        if ii_shr_mem_check_size(filename) != 0 {
            let file = ii_shr_mem_open(filename, mode);
            if file.is_null() {
                return core::ptr::null_mut();
            }
            (*file).state = IISTATE_READY;
            if add_to_opened_list(file) == 0 {
                return file;
            }
            ii_delete(file);
            return core::ptr::null_mut();
        }
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiOpen - {} has shared memory prefix but does not have correct form: {}\n",
                String::from_utf8_lossy(filename),
                // `iimage.c:284` passes a single argument for two `%s`
                // conversions; the reference reads whatever follows in the
                // varargs list.  Repeat the name rather than emulate that
                // undefined read.
                String::from_utf8_lossy(filename)
            ),
        );
        return core::ptr::null_mut();
    }
    let file = ii_new();
    if file.is_null() {
        return core::ptr::null_mut();
    }
    unsafe {
        *libc::__errno_location() = 0;
        (*file).fp = if filename.is_empty() {
            Some(ImodFile::Stdin)
        } else {
            ImodFile::open(&String::from_utf8_lossy(filename), mode)
        };
        if (*file).fp.is_none() || init_check_list() != 0 {
            if (*file).fp.is_none() {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: iiOpen - Opening file {} ({})\n",
                        String::from_utf8_lossy(filename),
                        core::ffi::CStr::from_ptr(libc::strerror(*libc::__errno_location()))
                            .to_string_lossy()
                    ),
                );
            } else {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: iiOpen - Opening file {}\n",
                        String::from_utf8_lossy(filename)
                    ),
                );
            }
            ii_delete(file);
            return core::ptr::null_mut();
        }
        (*file).format = IIFILE_UNKNOWN;
        (*file).filename = Some(filename.to_vec());
        // `iimage.c:284` is `strncpy(ofile->fmode, mode, 3)`: at most three
        // bytes, and the fourth stays the NUL `iiNew`'s `memset` left.
        for (dst, src) in (*file).fmode.iter_mut().zip(mode.bytes().take(3)) {
            *dst = src;
        }
        // `iimage.c:239` declares `err = 0`, so an empty check list leaves it
        // zero and the "unknown format" report below is not made.
        let mut err = 0;
        for index in 0..ilist_size(S_CHECK_LIST.as_ref()) {
            let check = *ilist_item(S_CHECK_LIST.as_mut(), index)
                .map_or(core::ptr::null_mut(), |item| {
                    item.as_mut_ptr().cast::<IiFileCheckFunction>()
                });
            if (*file).fp.is_none() {
                // `iimage.c:292-294`.
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: iiOpen - {} could not be reopened\n",
                        String::from_utf8_lossy(filename)
                    ),
                );
                break;
            }
            err = check.unwrap()(file);
            if err == 0 {
                if (*file).num_volumes <= 1 || S_ALLOW_MULTI_VOLUME.load(Ordering::SeqCst) != 0 {
                    (*file).state = IISTATE_READY;
                    if add_to_opened_list(file) == 0 {
                        return file;
                    }
                } else {
                    // `iimage.c:299-302`.
                    b3d_error(
                        Some(&mut ImodFile::Stderr),
                        format_args!(
                            "ERROR: iiOpen - {} is an HDF file with multiple volumes and cannot be opened by this program or with current options to the program\n",
                            String::from_utf8_lossy(filename)
                        ),
                    );
                }
                ii_delete(file);
                return core::ptr::null_mut();
            }
            if err != IIERR_NOT_FORMAT {
                break;
            }
        }
        // `iimage.c:314-315`.
        if err == IIERR_NOT_FORMAT {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: iiOpen - {} has unknown format.\n",
                    String::from_utf8_lossy(filename)
                ),
            );
        }
        ii_delete(file);
    }
    core::ptr::null_mut()
}
pub unsafe fn ii_open_new(filename: &[u8], mode: &str, mut file_kind: i32) -> *mut ImodImageFile {
    const IIFILE_DEFAULT: i32 = -1;
    if !mode.contains('w') {
        return core::ptr::null_mut();
    }
    let mut err = 0;
    let file = if filename.starts_with(SHR_MEM_NAME_TAG) {
        if ii_shr_mem_check_size(filename) == 0 {
            return core::ptr::null_mut();
        }
        file_kind = IIFILE_SHR_MEM;
        ii_shr_mem_open(filename, mode)
    } else {
        let new_file = ii_new();
        if new_file.is_null() {
            return core::ptr::null_mut();
        }
        (*new_file).filename = Some(filename.to_vec());
        new_file
    };
    if file.is_null() {
        return core::ptr::null_mut();
    }
    if file_kind == IIFILE_DEFAULT {
        file_kind = b3d_output_file_type();
    }
    if err == 0 {
        err = match file_kind {
            IIFILE_MRC => ii_mrc_open_new(file, mode),
            IIFILE_HDF => ii_hdf_open_new(file, mode),
            IIFILE_TIFF => tiff_open_new(file),
            IIFILE_JPEG => jpeg_open_new(file),
            IIFILE_SHR_MEM => 0,
            _ => 1,
        };
    }
    if err == 0 {
        (*file).file = file_kind;
        (*file).new_file = 1;
        // `iimage.c:387` is `strncpy(ofile->fmode, "rb+", 3)`.
        (&mut (*file).fmode)[..3].copy_from_slice(b"rb+");
        (*file).state = IISTATE_READY;
        if add_to_opened_list(file) == 0 {
            return file;
        }
    }
    ii_delete(file);
    core::ptr::null_mut()
}
pub unsafe fn ii_reopen(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() {
        return -1;
    }
    if unsafe { (*in_file).fp.is_some() } {
        return 1;
    }
    unsafe {
        if (*in_file).fmode[0] == 0 {
            // `iimage.c:411-412`.
            (&mut (*in_file).fmode)[..3].copy_from_slice(b"rb+");
        }
        if let Some(reopen) = (*in_file).reopen {
            if reopen(in_file) != 0 {
                return 2;
            }
            (*in_file).state = IISTATE_READY;
            add_to_opened_list(in_file);
            return 0;
        }
        let Some(name) = (*in_file).filename.clone() else {
            return 2;
        };
        (*in_file).fp = ImodFile::open(
            &String::from_utf8_lossy(&name),
            &String::from_utf8_lossy({
                let fmode = &(*in_file).fmode;
                &fmode[..fmode.iter().position(|b| *b == 0).unwrap_or(4)]
            }),
        );
        if (*in_file).fp.is_none() {
            return 2;
        }
        add_to_opened_list(in_file);
        if (*in_file).state != IISTATE_NOTINIT {
            (*in_file).state = IISTATE_READY;
            return 0;
        }
        (*in_file).format = IIFILE_UNKNOWN;
        for index in 0..ilist_size(S_CHECK_LIST.as_ref()) {
            let check = *ilist_item(S_CHECK_LIST.as_mut(), index)
                .map_or(core::ptr::null_mut(), |item| {
                    item.as_mut_ptr().cast::<IiFileCheckFunction>()
                });
            if check.unwrap()(in_file) == 0 {
                (*in_file).state = IISTATE_READY;
                return 0;
            }
        }
    }
    -1
}
/// Matches C `iiSetMM` (`iimage.c:454`).
pub unsafe fn ii_set_mm(
    in_file: *mut ImodImageFile,
    mut in_min: f32,
    mut in_max: f32,
    scale_max: f32,
) -> i32 {
    unsafe {
        if in_min != in_max {
            (*in_file).smin = in_min;
            (*in_file).smax = in_max;
        }
        if (*in_file).smin == (*in_file).smax {
            (*in_file).smin = 0.;
            (*in_file).smax = 255.;
        }
        in_min = (*in_file).smin;
        in_max = (*in_file).smax;
        if (*in_file).format == IIFORMAT_COMPLEX {
            (in_min, in_max) = mrc_complex_smin_smax(in_min, in_max);
        }
        (*in_file).slope = scale_max / (in_max - in_min);
        (*in_file).offset = -in_min * (*in_file).slope;
    }
    0
}
pub unsafe fn ii_close(in_file: *mut ImodImageFile) {
    if in_file.is_null() {
        return;
    }
    unsafe {
        if let Some(close) = (*in_file).close {
            close(in_file);
        } else if let Some(fp) = (*in_file).fp.take() {
            drop(fp);
        }
        (*in_file).fp = None;
        remove_from_opened_list(in_file);
        if (*in_file).state != IISTATE_NOTINIT {
            (*in_file).state = IISTATE_PARK;
        }
    }
}
pub unsafe fn ii_delete(in_file: *mut ImodImageFile) {
    if in_file.is_null() {
        return;
    }
    unsafe {
        ii_close(in_file);
        (*in_file).filename = None;
        if let Some(clean_up) = (*in_file).clean_up {
            clean_up(in_file);
        }
        (*in_file).description = None;
        libc::free((*in_file).colormap.cast());
        // `iiNew` hands out a `Box`; reclaim it the same way, which is also
        // what runs the `Option<ImodFile>` destructor and closes the file.
        drop(Box::from_raw(in_file));
    }
}
pub unsafe fn ii_copy_open(in_file: *mut ImodImageFile) -> *mut ImodImageFile {
    if in_file.is_null() {
        return core::ptr::null_mut();
    }
    if !(*in_file).colormap.is_null() {
        // `iimage.c:535`.
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiCopyOpen - Not allowed for a file with a colormap\n"),
        );
        return core::ptr::null_mut();
    }
    let copy = ii_new();
    if copy.is_null() {
        return copy;
    }
    // `iimage.c:541` is `memcpy(copy, inFile, sizeof(ImodImageFile))`; see the
    // type's `Clone` note for why this is a clone rather than a byte copy.
    *copy = (*in_file).clone();
    (*copy).fp = None;
    (*copy).header = core::ptr::null_mut();
    (*copy).filename = None;
    (*copy).description = None;
    if (*in_file).state != IISTATE_NOTINIT {
        (*in_file).state = IISTATE_PARK;
    }
    (*copy).filename = (*in_file).filename.clone();
    if (*copy).filename.is_none() {
        ii_delete(copy);
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiCopyOpen - Memory error copying filename\n"),
        );
        return core::ptr::null_mut();
    }
    if !(*in_file).directory_nums.is_null() {
        (*copy).directory_nums = ilist_dup((*in_file).directory_nums.cast::<Ilist>().as_ref())
            .map_or(core::ptr::null_mut(), Box::into_raw)
            .cast();
        if (*copy).directory_nums.is_null() {
            ii_delete(copy);
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: iiCopyOpen - Memory error copying directory list\n"),
            );
            return core::ptr::null_mut();
        }
    }
    let err = ii_reopen(copy);
    if err != 0 {
        ii_delete(copy);
        // `iimage.c:569` writes `%d` with no argument at all, so the reference
        // prints whatever is in the next varargs slot.  Pass the error code the
        // source plainly meant; the garbage it actually reads is not matchable.
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiCopyOpen - Error {err} calling iiReopen on copy of file\n"),
        );
        return core::ptr::null_mut();
    }
    copy
}
pub unsafe fn ii_use_tiff_threads(in_file: *mut ImodImageFile, mut max_threads: i32) -> i32 {
    if (*in_file).file != IIFILE_TIFF || S_MAX_TIFF_THREADS > 0 {
        return 0;
    }
    if max_threads <= 0 {
        max_threads = MAX_TIFF_THREADS as i32;
    }
    max_threads = tiff_num_read_threads(
        (*in_file).nx,
        (*in_file).ny,
        (*in_file).tiff_compression,
        max_threads,
    );
    if max_threads > 1 {
        S_II_TIFFS[0] = in_file;
        max_threads =
            ii_open_copies_for_threads(core::ptr::addr_of_mut!(S_II_TIFFS).cast(), max_threads);
    }
    if max_threads > 1 {
        S_MAX_TIFF_THREADS = max_threads;
    }
    S_MAX_TIFF_THREADS
}
pub unsafe fn ii_use_tiff_threads_for_fp(fp: &ImodFile, max_threads: i32) -> i32 {
    match ii_lookup_file_from_fp(fp) {
        None => 0,
        Some(file) => ii_use_tiff_threads(file, max_threads),
    }
}
pub unsafe fn ii_close_tiff_copies(in_file: *mut ImodImageFile) {
    if S_MAX_TIFF_THREADS <= 0 || in_file != S_II_TIFFS[0] {
        return;
    }
    for index in 1..S_MAX_TIFF_THREADS {
        ii_delete(S_II_TIFFS[index as usize]);
    }
    S_MAX_TIFF_THREADS = 0;
}
pub unsafe fn ii_close_tiff_copies_for_fp(fp: &ImodFile) {
    if let Some(file) = ii_lookup_file_from_fp(fp) {
        ii_close_tiff_copies(file);
    }
}
pub unsafe fn ii_open_copies_for_threads(
    file_copies: *mut *mut ImodImageFile,
    max_threads: i32,
) -> i32 {
    for index in 1..max_threads {
        *file_copies.add(index as usize) = ii_copy_open(*file_copies);
        if (*file_copies.add(index as usize)).is_null() {
            return index;
        }
    }
    max_threads
}
/// Matches C `iiFillMrcHeader(ImodImageFile *, MrcHeader *)` (`iimage.c:661`).
pub unsafe extern "C" fn ii_fill_mrc_header(
    in_file: *mut ImodImageFile,
    hdata: *mut MrcHeader,
) -> i32 {
    if in_file.is_null() {
        return 1;
    }
    let Some(fill_mrc_header) = (unsafe { (*in_file).fill_mrc_header }) else {
        return 1;
    };
    unsafe { fill_mrc_header(in_file, hdata) }
}

/// Matches C `iiSimpleFillMrcHeader(ImodImageFile *, MrcHeader *)` (`iimage.c:674`).
pub unsafe fn ii_simple_fill_mrc_header(in_file: *mut ImodImageFile, hdata: *mut MrcHeader) -> i32 {
    unsafe {
        mrc_head_new(
            &mut *hdata,
            (*in_file).nx,
            (*in_file).ny,
            (*in_file).nz,
            (*in_file).mode,
        );
        (*hdata).bytes_signed = 0;
        (*hdata).fp = (*in_file).fp.clone();
        (*hdata).amin = (*in_file).amin;
        (*hdata).amax = (*in_file).amax;
        (*hdata).amean = (*in_file).amean;
        (*hdata).xlen = (*in_file).nx as f32 * (*in_file).xscale;
        (*hdata).ylen = (*in_file).ny as f32 * (*in_file).yscale;
        (*hdata).zlen = (*in_file).nz as f32 * (*in_file).zscale;
    }
    0
}
/// Matches C `iiSyncFromMrcHeader(ImodImageFile *, MrcHeader *)` (`iimage.c:693`).
pub unsafe fn ii_sync_from_mrc_header(in_file: *mut ImodImageFile, hdata: *mut MrcHeader) {
    unsafe {
        let bytes_signed = if (*hdata).bytes_signed != 0
            && !((*in_file).file == IIFILE_TIFF && (*in_file).new_file != 0)
        {
            1
        } else {
            0
        };
        if (*hdata).mode != (*in_file).mode || (*hdata).mode == 0 {
            ii_mrc_mode_to_format_type(in_file, (*hdata).mode, bytes_signed);
        }
        (*in_file).nx = (*hdata).nx;
        (*in_file).ny = (*hdata).ny;
        (*in_file).nz = (*hdata).nz;
        (*in_file).amin = (*hdata).amin;
        (*in_file).amax = (*hdata).amax;
        (*in_file).amean = (*hdata).amean;
        (*in_file).rms = (*hdata).rms;
        (*in_file).xscale = 1.0;
        (*in_file).yscale = 1.0;
        (*in_file).zscale = 1.0;
        if (*hdata).xlen != 0.0 && (*hdata).mx != 0 {
            (*in_file).xscale = (*hdata).xlen / (*hdata).mx as f32;
        }
        if (*hdata).ylen != 0.0 && (*hdata).my != 0 {
            (*in_file).yscale = (*hdata).ylen / (*hdata).my as f32;
        }
        if (*hdata).xlen != 0.0 && (*hdata).mz != 0 {
            (*in_file).zscale = (*hdata).zlen / (*hdata).mz as f32;
        }
        (*in_file).xtrans = (*hdata).xorg;
        (*in_file).ytrans = (*hdata).yorg;
        (*in_file).ztrans = (*hdata).zorg;
        (*in_file).xrot = (*hdata).tiltangles[3];
        (*in_file).yrot = (*hdata).tiltangles[4];
        (*in_file).zrot = (*hdata).tiltangles[5];
        (*in_file).header_size = (*hdata).header_size;
        (*in_file).section_skip = (*hdata).section_skip;
        if let Some(sync_from_mrc_header) = (*in_file).sync_from_mrc_header {
            sync_from_mrc_header(in_file, hdata);
        }
    }
}
/// Matches C `iiDefaultMinMaxMean(int, float *, float *, float *)` (`iimage.c:741`).
pub fn ii_default_min_max_mean(type_: i32, amin: &mut f32, amax: &mut f32, amean: &mut f32) -> i32 {
    match type_ {
        IITYPE_UBYTE | IITYPE_FLOAT => {
            *amin = 0.0;
            *amax = 255.0;
        }
        IITYPE_BYTE => {
            *amin = -128.0;
            *amax = 127.0;
        }
        IITYPE_SHORT => {
            *amin = -32767.0;
            *amax = 32767.0;
        }
        IITYPE_USHORT => {
            *amin = 0.0;
            *amax = 65535.0;
        }
        _ => return 1,
    }
    *amean = (*amax + *amin) / 2.0;
    0
}

/// Matches C `iiWriteHeader(ImodImageFile *)` (`iimage.c:772`).
pub unsafe fn ii_write_header(in_file: *mut ImodImageFile) -> i32 {
    if let Some(write_header) = unsafe { (*in_file).write_header } {
        return unsafe { write_header(in_file) };
    }
    0
}
/// Matches C `iiAddToOpenedList(ImodImageFile *)` (`iimage.c:783`).
pub unsafe fn ii_add_to_opened_list(ii_file: *mut ImodImageFile) -> i32 {
    unsafe { add_to_opened_list(ii_file) }
}

/// Matches C static `addToOpenedList(ImodImageFile *)` (`iimage.c:791`).
pub unsafe fn add_to_opened_list(mut ii_file: *mut ImodImageFile) -> i32 {
    unsafe {
        if S_OPENED_FILES.is_null() {
            S_OPENED_FILES = ilist_new(core::mem::size_of::<*mut ImodImageFile>() as i32, 4)
                .map_or(core::ptr::null_mut(), Box::into_raw);
        }
        if !S_OPENED_FILES.is_null()
            && ilist_append(
                &mut *S_OPENED_FILES,
                core::slice::from_raw_parts(
                    (&raw const ii_file).cast::<u8>(),
                    core::mem::size_of::<*mut ImodImageFile>(),
                ),
            ) == 0
        {
            return 0;
        }
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiOpen - Memory error adding new file to master list\n"),
        );
    }
    1
}

/// Matches C static `removeFromOpenedList(ImodImageFile *)` (`iimage.c:801`).
pub unsafe fn remove_from_opened_list(ii_file: *mut ImodImageFile) {
    let index = unsafe { find_file_in_list(ii_file, None) };
    if index >= 0 {
        unsafe { ilist_remove(&mut *S_OPENED_FILES, index) };
    }
}

/// Matches C static `findFileInList(ImodImageFile *, FILE *)` (`iimage.c:811`).
///
/// The source's `(*listPtr)->fp == fp` is an identity test on the C library's
/// own handle; [`ImodFile::ptr_eq`] is that test.
pub unsafe fn find_file_in_list(ii_file: *mut ImodImageFile, fp: Option<&ImodFile>) -> i32 {
    let mut index = 0;
    while index < unsafe { ilist_size(S_OPENED_FILES.as_ref()) } {
        let list_pointer = unsafe {
            ilist_item(S_OPENED_FILES.as_mut(), index).map_or(core::ptr::null_mut(), |item| {
                item.as_mut_ptr().cast::<*mut ImodImageFile>()
            })
        };
        if (!ii_file.is_null() && unsafe { *list_pointer } == ii_file)
            || fp.is_some_and(|fp| {
                unsafe { (**list_pointer).fp.as_ref() }.is_some_and(|other| other.ptr_eq(fp))
            })
        {
            return index;
        }
        index += 1;
    }
    -1
}
/// Matches C `iiFileChangeAddress(ImodImageFile *, ImodImageFile *)` (`iimage.c:829`).
pub unsafe fn ii_file_change_address(old_file: *mut ImodImageFile, new_file: *mut ImodImageFile) {
    let change_list = if unsafe { (*old_file).fp.is_some() } {
        1
    } else {
        0
    };
    if change_list != 0 {
        unsafe { remove_from_opened_list(old_file) };
    }
    // `iimage.c:834`: an HDF file carries its own address in `fp` as an
    // identity token, so relocating the struct means restamping the token.
    if unsafe { (*new_file).fp.as_ref() }
        .is_some_and(|fp| fp.ptr_eq(&ImodFile::Token(old_file as usize)))
    {
        unsafe {
            (*new_file).fp = Some(ImodFile::Token(new_file as usize));
            if (*new_file).file == IIFILE_HDF {
                (*(*new_file).header.cast::<MrcHeader>()).fp = (*new_file).fp.clone();
            }
        }
    }
    if !unsafe { (*new_file).ii_volumes }.is_null() && unsafe { (*new_file).num_volumes } != 0 {
        for index in 0..unsafe { (*new_file).num_volumes } {
            if unsafe { *(*new_file).ii_volumes.add(index as usize) } == old_file {
                unsafe { *(*new_file).ii_volumes.add(index as usize) = new_file };
            }
        }
    }
    if change_list != 0 {
        unsafe { add_to_opened_list(new_file) };
    }
}
pub unsafe fn ii_fopen(filename: &[u8], mode: &str) -> Option<ImodFile> {
    let file = unsafe { ii_open(filename, mode) };
    if file.is_null() {
        None
    } else {
        unsafe { (*file).fp.clone() }
    }
}
/// Matches C `iiLookupFileFromFP(FILE *)` (`iimage.c:866`).
pub fn ii_lookup_file_from_fp(fp: &ImodFile) -> Option<*mut ImodImageFile> {
    let index = unsafe { find_file_in_list(core::ptr::null_mut(), Some(fp)) };
    if index < 0 {
        return None;
    }
    let file = unsafe {
        *ilist_item(S_OPENED_FILES.as_mut(), index).map_or(core::ptr::null_mut(), |item| {
            item.as_mut_ptr().cast::<*mut ImodImageFile>()
        })
    };
    if file.is_null() { None } else { Some(file) }
}
pub fn ii_fclose(fp: &mut ImodFile) {
    match ii_lookup_file_from_fp(fp) {
        // `fclose` on a handle this layer does not own: dropping the last
        // clone of the `Rc<File>` closes the descriptor.  Where another clone
        // survives, C would leave that alias dangling and using it would be
        // undefined; here it stays usable.  Recorded as a deviation.
        None => drop(core::mem::replace(fp, ImodFile::Token(0))),
        Some(file) => unsafe { ii_delete(file) },
    }
}
/// Matches C `iiFOpenVolume` (`iimage.c:896`).
pub unsafe fn ii_fopen_volume(in_file: *mut ImodImageFile, vol_index: i32) -> Option<ImodFile> {
    if in_file.is_null() {
        return None;
    }
    if (*in_file).file != IIFILE_HDF || (*in_file).num_volumes < 2 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiFOpenVolume - Attempting to open a secondary volume for a non-HDF file or an HDF file with only a stack or one volume"
            ),
        );
        return None;
    }
    if vol_index < 1 || vol_index >= (*in_file).num_volumes {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiFOpenVolume - Requested volume index {vol_index} out of range\n"
            ),
        );
        return None;
    }
    let volume = *(*in_file).ii_volumes.add(vol_index as usize);
    if ii_reopen(volume) != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiFOpenVolume - Error calling iiReopen on volume at index {vol_index}\n"
            ),
        );
        return None;
    }
    (*volume).fp.clone()
}
/// Matches C `iiFOpenNewVolume` (`iimage.c:925`).
pub unsafe fn ii_fopen_new_volume(in_file: *mut ImodImageFile) -> Option<ImodFile> {
    if in_file.is_null() {
        return None;
    }
    if (*in_file).file != IIFILE_HDF || (!(*in_file).stack_set_list.is_null() && (*in_file).nz > 1)
    {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiFOpenNewVolume - Attempting to create an additional volume for a non-HDF file or an HDF file with a stack in it\n"
            ),
        );
        return None;
    }
    if ii_hdf_open_new(in_file, "wb+") != 0 {
        return None;
    }
    let file = *(*in_file)
        .ii_volumes
        .add(((*in_file).num_volumes - 1) as usize);
    if add_to_opened_list(file) != 0 {
        return None;
    }
    (*file).fp.clone()
}
/// Matches C `iiChangeCallCount(int)` (`iimage.c:944`).
pub fn ii_change_call_count(delta: i32) {
    S_RW_CALL_COUNT
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
            Some((value + delta).max(0))
        })
        .unwrap();
}

/// Matches C `iiCallingReadOrWrite(void)` (`iimage.c:949`).
pub fn ii_calling_read_or_write() -> i32 {
    S_RW_CALL_COUNT.load(Ordering::SeqCst)
}

/// Matches C `iiAllowMultiVolume(int)` (`iimage.c:958`).
pub fn ii_allow_multi_volume(allow: i32) {
    S_ALLOW_MULTI_VOLUME.store(allow, Ordering::SeqCst);
}

/// Matches C `iiSetChunkSizes(ImodImageFile *, int, int, int)` (`iimage.c:970`).
pub unsafe fn ii_set_chunk_sizes(
    in_file: *mut ImodImageFile,
    x_size: i32,
    y_size: i32,
    z_size: i32,
) -> i32 {
    if in_file.is_null()
        || unsafe { (*in_file).file } != IIFILE_HDF
        || !unsafe { (*in_file).stack_set_list }.is_null()
    {
        unsafe {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: iiSetChunkSizes - Attempting to set chunk sizes for a non-HDF file or an HDF file with a stack in it\n"
                ),
            );
        }
        return 1;
    }
    if unsafe { (*in_file).dataset_name.is_some() } {
        unsafe {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: iiSetChunkSizes - The volume dataset properties have already been set and cannot be changed\n"
                ),
            );
        }
        return 1;
    }
    if x_size < 0 || y_size < 0 || z_size <= 0 {
        unsafe {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: iiSetChunkSizes - X and Y chunk sizes must be non-negative and Z size must be positive\n"
                ),
            );
        }
        return 1;
    }
    unsafe {
        (*in_file).tile_size_x = x_size;
        (*in_file).tile_size_y = y_size;
        (*in_file).z_chunk_size = z_size;
    }
    0
}
pub unsafe fn ii_get_adoc_index(
    in_file: *mut ImodImageFile,
    global: i32,
    open_mdoc_or_new: i32,
) -> i32 {
    if in_file.is_null() {
        return -2;
    }
    if (*in_file).file == IIFILE_HDF {
        return if !(*in_file).stack_set_list.is_null()
            || (*in_file).global_adoc_index < 0
            || global == 0
        {
            (*in_file).adoc_index
        } else {
            (*in_file).global_adoc_index
        };
    }
    if (*in_file).adoc_index >= 0 || open_mdoc_or_new == 0 {
        return (*in_file).adoc_index;
    }
    if open_mdoc_or_new < 0 {
        (*in_file).adoc_index = adoc_new();
    } else {
        // `iimage.c:1020-1026`: the image name with ".mdoc" appended.
        let mut name = (*in_file).filename.clone().unwrap_or_default();
        name.extend_from_slice(b".mdoc");
        (*in_file).adoc_index = adoc_read(&name);
    }
    if (*in_file).adoc_index < 0 {
        -2
    } else {
        (*in_file).adoc_index
    }
}
pub unsafe fn ii_transfer_adoc_sections(
    from_file: *mut ImodImageFile,
    to_file: *mut ImodImageFile,
) -> i32 {
    if (*from_file).adoc_index < 0 || (*to_file).adoc_index < 0 {
        return 1;
    }
    if adoc_set_current((*from_file).adoc_index) != 0
        || adoc_transfer_section(
            ADOC_GLOBAL_NAME,
            0,
            (*to_file).adoc_index,
            Some(ADOC_GLOBAL_NAME),
            0,
        ) != 0
    {
        return 1;
    }
    let from_doc = if (*from_file).global_adoc_index >= 0 {
        (*from_file).global_adoc_index
    } else {
        (*from_file).adoc_index
    };
    let to_doc = if (*to_file).global_adoc_index >= 0 {
        (*to_file).global_adoc_index
    } else {
        (*to_file).adoc_index
    };
    if adoc_set_current(from_doc) != 0 {
        return 1;
    }
    for coll in 0..adoc_get_num_collections() {
        let mut coll_name = Vec::new();
        if adoc_get_collection_name(coll, &mut coll_name) != 0 {
            return 1;
        }
        let mut err = 0;
        if coll_name == ADOC_ZVALUE_NAME {
            for section in 0..adoc_get_number_of_sections(&coll_name) {
                let mut section_name = Vec::new();
                if adoc_get_section_name(&coll_name, section, &mut section_name) != 0 {
                    err = 1;
                } else {
                    err =
                        adoc_transfer_section(&coll_name, section, to_doc, Some(&section_name), 0);
                }
                if err != 0 {
                    break;
                }
            }
        }
        if err != 0 {
            return err;
        }
    }
    0
}
pub unsafe extern "C" fn ii_read_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe {
        read_write_section(
            in_file,
            buf,
            in_section,
            (*in_file).read_section,
            "reading from",
        )
    }
}
pub unsafe extern "C" fn ii_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe {
        read_write_section(
            in_file,
            buf,
            in_section,
            (*in_file).read_section_byte,
            "reading and converting to bytes for",
        )
    }
}
pub unsafe extern "C" fn ii_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe {
        read_write_section(
            in_file,
            buf,
            in_section,
            (*in_file).read_section_ushort,
            "reading and converting to shorts for",
        )
    }
}
pub unsafe extern "C" fn ii_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe {
        read_write_section(
            in_file,
            buf,
            in_section,
            (*in_file).read_section_float,
            "reading and converting to floats for",
        )
    }
}
pub unsafe fn ii_read_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    convert_to: i32,
) -> i32 {
    unsafe {
        match convert_to {
            MRSA_NOPROC => ii_read_section(in_file, buf, in_section),
            MRSA_BYTE => ii_read_section_byte(in_file, buf, in_section),
            MRSA_USHORT => ii_read_section_ushort(in_file, buf, in_section),
            MRSA_FLOAT => ii_read_section_float(in_file, buf, in_section),
            _ => {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: iiReadSectionAny - Invalid value {} for convertTo parameter\n",
                        convert_to
                    ),
                );
                -1
            }
        }
    }
}
pub unsafe fn ii_write_section(in_file: *mut ImodImageFile, buf: *mut u8, in_section: i32) -> i32 {
    unsafe {
        read_write_section(
            in_file,
            buf,
            in_section,
            (*in_file).write_section,
            "writing to",
        )
    }
}
pub unsafe fn ii_write_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut f32,
    in_section: i32,
) -> i32 {
    unsafe {
        read_write_section(
            in_file,
            buf.cast(),
            in_section,
            (*in_file).write_section_float,
            "converting floats to write to",
        )
    }
}
pub unsafe fn read_write_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    func: IiSectionFunc,
    mess: &str,
) -> i32 {
    let Some(func) = func else {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiRead/WriteSection - There is no function for {} this type of file\n",
                mess
            ),
        );
        return -1;
    };
    unsafe {
        if (*in_file).fp.is_none() && ii_reopen(in_file) != 0 {
            return -1;
        }
        let mut data_size = 0;
        let mut convert = 0;
        if S_MAX_TIFF_THREADS > 1 && in_file == S_II_TIFFS[0] && (*in_file).axis == 3 {
            let mut dsize = 0;
            let mut csize = 0;
            if mrc_getdcsize((*in_file).mode, &mut dsize, &mut csize) == 0 {
                if Some(func) == (*in_file).read_section {
                    data_size = dsize * csize;
                    convert = MRSA_NOPROC;
                } else if Some(func) == (*in_file).read_section_byte {
                    data_size = 1;
                    convert = MRSA_BYTE;
                } else if Some(func) == (*in_file).read_section_ushort {
                    data_size = 2;
                    convert = MRSA_USHORT;
                } else if Some(func) == (*in_file).read_section_float {
                    data_size = 4;
                    convert = MRSA_FLOAT;
                }
            }
        }
        ii_change_call_count(1);
        let err = if data_size != 0 {
            tiff_parallel_read(
                core::ptr::addr_of_mut!(S_II_TIFFS).cast(),
                S_MAX_TIFF_THREADS,
                (*in_file).llx,
                (*in_file).urx,
                (*in_file).lly,
                (*in_file).ury,
                data_size,
                buf,
                in_section,
                convert,
            )
        } else {
            func(in_file, buf, in_section)
        };
        ii_change_call_count(-1);
        err
    }
}
/// Matches C `iiReadPoint` (`iimage.c:1214`).
pub unsafe fn ii_read_point(in_file: *mut ImodImageFile, x: i32, y: i32, z: i32) -> f32 {
    unsafe {
        let mut value = (*in_file).amin;
        if x < 0 || y < 0 || z < 0 || x >= (*in_file).nx || y >= (*in_file).ny || z >= (*in_file).nz
        {
            return value;
        }
        let mut save = ImodImageFile::default();
        ii_save_load_params(in_file, &mut save);
        (*in_file).llx = x;
        (*in_file).urx = x;
        (*in_file).lly = y;
        (*in_file).ury = y;
        (*in_file).axis = 3;
        if (*in_file).mode == MRC_MODE_COMPLEX_SHORT {
            let mut data = [0i16; 2];
            if ii_read_section(in_file, data.as_mut_ptr().cast(), z) == 0 {
                value = ((data[0] as f64 * data[0] as f64 + data[1] as f64 * data[1] as f64).sqrt())
                    as f32;
            }
        } else if (*in_file).mode == MRC_MODE_COMPLEX_FLOAT {
            let mut data = [0f32; 2];
            if ii_read_section(in_file, data.as_mut_ptr().cast(), z) == 0 {
                value = ((data[0] as f64 * data[0] as f64 + data[1] as f64 * data[1] as f64).sqrt())
                    as f32;
            }
        } else {
            ii_read_section_float(in_file, (&mut value as *mut f32).cast(), z);
        }
        ii_restore_load_params(0, in_file, &mut save);
        value
    }
}
pub unsafe fn ii_load_pcoord(
    in_file: *mut ImodImageFile,
    use_mdoc: i32,
    li: *mut crate::imod::libiimod::mrcfiles::LoadInfo,
    nx: i32,
    ny: i32,
    nz: i32,
) -> i32 {
    if (*in_file).file == IIFILE_HDF || (*in_file).file == IIFILE_ADOC {
        let adoc_index = ii_get_adoc_index(in_file, 0, 0);
        let mut montage = 0;
        let mut num_sect = 0;
        let mut sect_type = 0;
        if adoc_index >= 0
            && adoc_set_current(adoc_index) == 0
            && adoc_get_image_meta_info(&mut montage, &mut num_sect, &mut sect_type) == 0
        {
            crate::imod::libiimod::plist::ii_plist_from_autodoc(
                adoc_index, 0, li, nx, ny, nz, montage, num_sect, sect_type,
            );
        }
        return 0;
    }
    if ii_mrc_check(in_file) != 0 {
        return 0;
    }
    if use_mdoc < 2 {
        ii_mrc_load_pcoord(in_file, li, nx, ny, nz);
    }
    if (*li).plist == 0 && use_mdoc != 0 {
        crate::imod::libiimod::plist::ii_plist_from_metadata(
            (*in_file).filename.as_deref().unwrap_or_default(),
            1,
            li,
            nx,
            ny,
            nz,
        );
    }
    0
}
pub unsafe fn ii_make_buffer_convert_if_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    if_float: i32,
    inverted: *mut i32,
    routine: &str,
) -> *mut u8 {
    unsafe {
        let mut use_buf = buf;
        let mut fbufp = buf.cast::<f32>();
        let nx = (*in_file).nx;
        let ny = (*in_file).ny;
        let pixsize = if (*in_file).mode == MRC_MODE_BYTE {
            1
        } else {
            2
        };
        let data_size = nx as usize * ny as usize * pixsize as usize;
        if if_float != 0 && (*in_file).type_ != IITYPE_FLOAT {
            use_buf = libc::malloc(data_size).cast();
            if use_buf.is_null() {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: {} - Allocating array for converting floats\n",
                        routine
                    ),
                );
                return core::ptr::null_mut();
            }
            *inverted = 1;
            for iy in 0..ny {
                let bdata = use_buf
                    .cast::<u8>()
                    .add((nx * pixsize * (ny - 1 - iy)) as usize);
                ii_convert_line_of_floats(fbufp, bdata, nx, (*in_file).mode, 0, 0);
                fbufp = fbufp.add(nx as usize);
            }
        }
        use_buf
    }
}
pub unsafe fn ii_convert_line_of_floats(
    fbufp: *const f32,
    bdata: *mut u8,
    nx: i32,
    mrc_mode: i32,
    bytes_signed: i32,
    pack_4bits: i32,
) {
    unsafe {
        let sdata = bdata.cast::<i16>();
        let usdata = bdata.cast::<u16>();
        let sbdata = bdata.cast::<i8>();
        match mrc_mode {
            MRC_MODE_BYTE => {
                if pack_4bits != 0 {
                    let mut i = 0;
                    while i < nx / 2 {
                        let mut ival = (*fbufp.add((2 * i) as usize) + 0.5) as i32;
                        ival = ival.clamp(0, 15);
                        let mut hval = (*fbufp.add((2 * i + 1) as usize) + 0.5) as i32;
                        hval = hval.clamp(0, 15);
                        *bdata.add(i as usize) = (ival + (hval << 4)) as u8;
                        i += 1;
                    }
                    if nx % 2 != 0 {
                        let mut ival = (*fbufp.add((nx - 1) as usize) + 0.5) as i32;
                        ival = ival.clamp(0, 15);
                        *bdata.add(i as usize) = ival as u8;
                    }
                } else if bytes_signed != 0 {
                    for i in 0..nx {
                        // `iimage.c:1339`: `127.5` is a double literal and `floor`
                        // is the double version, so the float promotes and the
                        // whole expression evaluates in double.  Doing it in f32
                        // lands one off on values near a .5 boundary.
                        let mut ival = (*fbufp.add(i as usize) as f64 - 127.5).floor() as i32;
                        ival = ival.clamp(-128, 127);
                        *sbdata.add(i as usize) = ival as i8;
                    }
                } else {
                    for i in 0..nx {
                        let mut ival = (*fbufp.add(i as usize) + 0.5) as i32;
                        ival = ival.clamp(0, 255);
                        *bdata.add(i as usize) = ival as u8;
                    }
                }
            }
            MRC_MODE_SHORT => {
                for i in 0..nx {
                    // `iimage.c:1357`: `0.5` is a double literal and `floor` is the
                    // double version -- unlike the unsigned arms, which use `0.5f`.
                    let mut ival = (*fbufp.add(i as usize) as f64 + 0.5).floor() as i32;
                    ival = ival.clamp(-32768, 32767);
                    *sdata.add(i as usize) = ival as i16;
                }
            }
            MRC_MODE_USHORT => {
                for i in 0..nx {
                    let mut ival = (*fbufp.add(i as usize) + 0.5) as i32;
                    ival = ival.clamp(0, 65535);
                    *usdata.add(i as usize) = ival as u16;
                }
            }
            MRC_MODE_HALF_FLOAT => imnp_floatbuf_to_halfs(
                core::slice::from_raw_parts(fbufp, nx as usize),
                core::slice::from_raw_parts_mut(usdata, nx as usize),
                nx,
            ),
            _ => {}
        }
    }
}
pub unsafe fn ii_save_load_params(ii_file: *mut ImodImageFile, ii_save: *mut ImodImageFile) {
    unsafe {
        (*ii_save).llx = (*ii_file).llx;
        (*ii_save).urx = (*ii_file).urx;
        (*ii_save).lly = (*ii_file).lly;
        (*ii_save).ury = (*ii_file).ury;
        (*ii_save).llz = (*ii_file).llz;
        (*ii_save).urz = (*ii_file).urz;
        (*ii_save).axis = (*ii_file).axis;
        (*ii_save).pad_left = (*ii_file).pad_left;
        (*ii_save).pad_right = (*ii_file).pad_right;
        (*ii_save).slope = (*ii_file).slope;
        (*ii_save).offset = (*ii_file).offset;
    }
}
pub unsafe fn ii_restore_load_params(
    ret_val: i32,
    ii_file: *mut ImodImageFile,
    ii_save: *mut ImodImageFile,
) -> i32 {
    unsafe {
        (*ii_file).llx = (*ii_save).llx;
        (*ii_file).urx = (*ii_save).urx;
        (*ii_file).lly = (*ii_save).lly;
        (*ii_file).ury = (*ii_save).ury;
        (*ii_file).llz = (*ii_save).llz;
        (*ii_file).urz = (*ii_save).urz;
        (*ii_file).axis = (*ii_save).axis;
        (*ii_file).pad_left = (*ii_save).pad_left;
        (*ii_file).pad_right = (*ii_save).pad_right;
        (*ii_file).slope = (*ii_save).slope;
        (*ii_file).offset = (*ii_save).offset;
    }
    ret_val
}
/// Matches C `iiBestTileSize(int, int *, int *, int)` (`iimage.c:1425`).
pub fn ii_best_tile_size(im_size: i32, tile_size: &mut i32, num_tiles: &mut i32, multiple_of: i32) {
    ii_limited_tile_size(im_size, tile_size, num_tiles, multiple_of, 0);
}

/// Matches C `iibesttilesize(int *, int *, int *, int *)` (`iimage.c:1430`).
pub fn iibesttilesize(im_size: i32, tile_size: &mut i32, num_tiles: &mut i32, multiple_of: i32) {
    ii_best_tile_size(im_size, tile_size, num_tiles, multiple_of);
}

/// Matches C `iiLimitedTileSize(int, int *, int *, int, int)` (`iimage.c:1444`).
pub fn ii_limited_tile_size(
    im_size: i32,
    tile_size: &mut i32,
    num_tiles: &mut i32,
    multiple_of: i32,
    limit: i32,
) {
    if *tile_size == 0 {
        *tile_size = im_size;
    }
    let target = *tile_size;
    *num_tiles = (im_size + *tile_size - 1) / *tile_size;
    *tile_size = multiple_of * (im_size as f32 / (multiple_of * *num_tiles) as f32).ceil() as i32;
    *num_tiles = (im_size + *tile_size - 1) / *tile_size;
    if *num_tiles > 1 {
        let mut num_less = *num_tiles - 1;
        let tile_less =
            multiple_of * (im_size as f32 / (multiple_of * num_less) as f32).ceil() as i32;
        num_less = (im_size + tile_less - 1) / tile_less;
        if *num_tiles * *tile_size > num_less * tile_less
            && (limit <= 0 || tile_less <= limit)
            && (target - *tile_size).abs() > (target - tile_less).abs()
        {
            *num_tiles = num_less;
            *tile_size = tile_less;
        }
    }
}

/// Matches C `iilimitedtilesize(int *, int *, int *, int *, int *)` (`iimage.c:1466`).
pub fn iilimitedtilesize(
    im_size: i32,
    tile_size: &mut i32,
    num_tiles: &mut i32,
    multiple_of: i32,
    limit: i32,
) {
    ii_limited_tile_size(im_size, tile_size, num_tiles, multiple_of, limit);
}
/// The Fortran bridge (NATIVE.md 7): `f2cString` carries the hidden string
/// length, so this entry point keeps the C calling convention until both sides
/// of that bridge move together.
pub unsafe fn iitestifhdf(filename: *const c_char, name_len: i32) -> i32 {
    let cstr = crate::imod::libcfshr::b3dutil::f2c_string(filename, name_len);
    if cstr.is_null() {
        return -1;
    }
    let name = core::ffi::CStr::from_ptr(cstr).to_bytes().to_vec();
    libc::free(cstr.cast());
    native_ii_test_if_hdf(&name)
}
pub fn tiffseteerreadproperties(super_res: i32, auto_group: i32, flags: i32) {
    tiff_set_eer_read_properties(super_res, auto_group, flags);
}
pub fn get_dflt_eersumming_from_env(super_res: &mut i32, z_summing: &mut i32) {
    // `iimage.c:1509-1519`.  `atoi` on a string with no leading number is 0,
    // which is what `str::parse` failing stands in for here.
    if let Ok(var) = std::env::var("IMOD_DFLT_EER_SUPER_RES") {
        *super_res = var.trim_start().parse::<i32>().unwrap_or(0).clamp(-3, 2);
    }
    if let Ok(var) = std::env::var("IMOD_DFLT_EER_Z_SUMMING") {
        *z_summing = var.trim_start().parse::<i32>().unwrap_or(0);
        if *z_summing == 0 {
            *z_summing = 1;
        }
    }
}
pub fn tiffgetmaxeersuperres() -> i32 {
    tiff_get_max_eer_super_res()
}
pub unsafe fn ii_hdf_check(in_file: *mut ImodImageFile) -> i32 {
    native_ii_hdf_check(in_file)
}
pub unsafe fn ii_hdfopen_new(in_file: *mut ImodImageFile, mode: &str) -> i32 {
    ii_hdf_open_new(in_file, mode)
}
pub unsafe fn hdf_write_global_adoc(in_file: *mut ImodImageFile) -> i32 {
    native_hdf_write_global_adoc(in_file)
}
pub unsafe fn hdf_write_dummy_section(in_file: *mut ImodImageFile, buf: *mut u8, cz: i32) -> i32 {
    native_hdf_write_dummy_section(in_file, buf, cz)
}
pub unsafe fn ii_test_if_hdf(filename: &[u8]) -> i32 {
    native_ii_test_if_hdf(filename)
}
pub unsafe fn ii_reorder_hdfstack(in_file: *mut ImodImageFile, sect_order: *mut i32) -> i32 {
    ii_reorder_hdf_stack(in_file, sect_order)
}
pub unsafe fn hdf_read_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    cz: i32,
    type_: i32,
) -> i32 {
    native_hdf_read_section_any(in_file, buf, cz, type_)
}
pub unsafe fn hdf_write_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    cz: i32,
    from_float: i32,
) -> i32 {
    native_hdf_write_section_any(in_file, buf, cz, from_float)
}
pub unsafe fn init_new_hdffile(in_file: *mut ImodImageFile) -> i32 {
    native_init_new_hdf_file(in_file)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_min_max_mean_retains_all_supported_type_ranges() {
        let mut amin = 0.0;
        let mut amax = 0.0;
        let mut amean = 0.0;
        assert_eq!(
            ii_default_min_max_mean(IITYPE_BYTE, &mut amin, &mut amax, &mut amean),
            0
        );
        assert_eq!((amin, amax, amean), (-128.0, 127.0, -0.5));
        assert_eq!(
            ii_default_min_max_mean(IITYPE_USHORT, &mut amin, &mut amax, &mut amean),
            0
        );
        assert_eq!((amin, amax, amean), (0.0, 65535.0, 32767.5));
        assert_eq!(
            ii_default_min_max_mean(99, &mut amin, &mut amax, &mut amean),
            1
        );
    }

    #[test]
    fn convert_line_of_floats_preserves_source_rounding_clamping_and_packing() {
        unsafe {
            let floats = [-2., 0.49, 255.6];
            let mut bytes = [0u8; 3];
            ii_convert_line_of_floats(floats.as_ptr(), bytes.as_mut_ptr(), 3, MRC_MODE_BYTE, 0, 0);
            assert_eq!(bytes, [0, 0, 255]);

            let signed = [0., 127.5, 255.];
            ii_convert_line_of_floats(signed.as_ptr(), bytes.as_mut_ptr(), 3, MRC_MODE_BYTE, 1, 0);
            assert_eq!(bytes.map(|value| value as i8), [-128, 0, 127]);

            let packed = [1., 2., 15.];
            ii_convert_line_of_floats(packed.as_ptr(), bytes.as_mut_ptr(), 3, MRC_MODE_BYTE, 0, 1);
            assert_eq!(&bytes[..2], &[0x21, 15]);

            let values = [-1.1, 32767.9];
            let mut shorts = [0i16; 2];
            ii_convert_line_of_floats(
                values.as_ptr(),
                shorts.as_mut_ptr().cast(),
                2,
                MRC_MODE_SHORT,
                0,
                0,
            );
            assert_eq!(shorts, [-1, 32767]);

            let mut ushorts = [0u16; 2];
            ii_convert_line_of_floats(
                values.as_ptr(),
                ushorts.as_mut_ptr().cast(),
                2,
                MRC_MODE_USHORT,
                0,
                0,
            );
            assert_eq!(ushorts, [0, 32768]);

            let half_values = [0., 1.];
            ii_convert_line_of_floats(
                half_values.as_ptr(),
                ushorts.as_mut_ptr().cast(),
                2,
                MRC_MODE_HALF_FLOAT,
                0,
                0,
            );
            assert_eq!(ushorts, [0, 0x3c00]);
        }
    }

    #[test]
    fn make_buffer_convert_if_float_preserves_source_y_inversion_and_identity_return() {
        unsafe {
            let in_file = ii_new();
            (*in_file).nx = 2;
            (*in_file).ny = 2;
            (*in_file).mode = MRC_MODE_BYTE;
            (*in_file).type_ = IITYPE_UBYTE;
            let mut floats = [1f32, 2., 3., 4.];
            let mut inverted = 0;
            let converted = ii_make_buffer_convert_if_float(
                in_file,
                floats.as_mut_ptr().cast(),
                1,
                &mut inverted,
                "test",
            );
            assert!(!converted.is_null());
            assert_eq!(inverted, 1);
            assert_eq!(
                core::slice::from_raw_parts(converted.cast::<u8>(), 4),
                [3, 4, 1, 2]
            );
            libc::free(converted.cast());

            (*in_file).type_ = IITYPE_FLOAT;
            inverted = 0;
            assert_eq!(
                ii_make_buffer_convert_if_float(
                    in_file,
                    floats.as_mut_ptr().cast(),
                    1,
                    &mut inverted,
                    "test",
                ),
                floats.as_mut_ptr().cast()
            );
            assert_eq!(inverted, 0);
            drop(Box::from_raw(in_file));
        }
    }

    #[test]
    fn limited_tile_size_retains_source_rounding_and_preference_rule() {
        let mut tile_size = 256;
        let mut num_tiles = 0;
        ii_limited_tile_size(1000, &mut tile_size, &mut num_tiles, 16, 0);
        assert_eq!((tile_size, num_tiles), (256, 4));
        tile_size = 0;
        ii_best_tile_size(1000, &mut tile_size, &mut num_tiles, 64);
        assert_eq!((tile_size, num_tiles), (1024, 1));
    }

    #[test]
    fn simple_mrc_header_callback_copies_source_image_metadata() {
        unsafe {
            let mut image_file = ImodImageFile::default();
            image_file.nx = 4;
            image_file.ny = 5;
            image_file.nz = 6;
            image_file.mode = 2;
            image_file.amin = -2.0;
            image_file.amax = 9.0;
            image_file.amean = 3.5;
            image_file.xscale = 1.5;
            image_file.yscale = 2.0;
            image_file.zscale = 2.5;
            let mut header = MrcHeader::default();
            assert_eq!(ii_simple_fill_mrc_header(&mut image_file, &mut header), 0);
            assert_eq!((header.nx, header.ny, header.nz, header.mode), (4, 5, 6, 2));
            assert_eq!((header.amin, header.amax, header.amean), (-2.0, 9.0, 3.5));
            assert_eq!((header.xlen, header.ylen, header.zlen), (6.0, 10.0, 15.0));
            assert_eq!(header.bytes_signed, 0);
        }
    }

    #[test]
    fn sync_from_mrc_header_preserves_source_metadata_and_z_scale_condition() {
        unsafe {
            let image_file = ii_new();
            (*image_file).mode = crate::imod::libiimod::mrcfiles::MRC_MODE_SHORT;
            let mut header = MrcHeader::default();
            header.nx = 4;
            header.ny = 5;
            header.nz = 6;
            header.mode = crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE;
            header.bytes_signed = 1;
            header.amin = -3.0;
            header.amax = 18.0;
            header.amean = 7.5;
            header.rms = 2.0;
            header.mx = 2;
            header.my = 5;
            header.mz = 3;
            header.xlen = 10.0;
            header.ylen = 15.0;
            header.zlen = 21.0;
            header.xorg = 1.0;
            header.yorg = 2.0;
            header.zorg = 3.0;
            header.tiltangles[3] = 4.0;
            header.tiltangles[4] = 5.0;
            header.tiltangles[5] = 6.0;
            header.header_size = 1024;
            header.section_skip = 88;
            ii_sync_from_mrc_header(image_file, &mut header);
            assert_eq!(
                ((*image_file).nx, (*image_file).ny, (*image_file).nz),
                (4, 5, 6)
            );
            assert_eq!(
                ((*image_file).format, (*image_file).type_),
                (IIFORMAT_LUMINANCE, IITYPE_BYTE)
            );
            assert_eq!(
                (
                    (*image_file).amin,
                    (*image_file).amax,
                    (*image_file).amean,
                    (*image_file).rms
                ),
                (-3.0, 18.0, 7.5, 2.0)
            );
            assert_eq!(
                (
                    (*image_file).xscale,
                    (*image_file).yscale,
                    (*image_file).zscale
                ),
                (5.0, 3.0, 7.0)
            );
            assert_eq!(
                (
                    (*image_file).xtrans,
                    (*image_file).ytrans,
                    (*image_file).ztrans
                ),
                (1.0, 2.0, 3.0)
            );
            assert_eq!(
                ((*image_file).xrot, (*image_file).yrot, (*image_file).zrot),
                (4.0, 5.0, 6.0)
            );
            assert_eq!(
                ((*image_file).header_size, (*image_file).section_skip),
                (1024, 88)
            );
            header.xlen = 0.0;
            header.zlen = 30.0;
            ii_sync_from_mrc_header(image_file, &mut header);
            assert_eq!((*image_file).zscale, 1.0);
            libc::free(image_file.cast());
        }
    }

    #[test]
    fn read_write_count_and_hdf_chunk_size_guards_follow_source() {
        ii_change_call_count(-1);
        assert_eq!(ii_calling_read_or_write(), 0);
        ii_change_call_count(3);
        ii_change_call_count(-1);
        assert_eq!(ii_calling_read_or_write(), 2);
        ii_change_call_count(-9);
        assert_eq!(ii_calling_read_or_write(), 0);

        unsafe {
            let mut image_file = ImodImageFile::default();
            image_file.file = IIFILE_HDF;
            assert_eq!(ii_set_chunk_sizes(&mut image_file, 64, 32, 2), 0);
            assert_eq!(
                (
                    image_file.tile_size_x,
                    image_file.tile_size_y,
                    image_file.z_chunk_size
                ),
                (64, 32, 2)
            );
            crate::imod::libcfshr::b3dutil::b3d_set_store_error(1);
            assert_eq!(ii_set_chunk_sizes(&mut image_file, -1, 32, 2), 1);
            crate::imod::libcfshr::b3dutil::b3d_set_store_error(0);
            assert_eq!(
                (
                    image_file.tile_size_x,
                    image_file.tile_size_y,
                    image_file.z_chunk_size
                ),
                (64, 32, 2)
            );
        }
    }

    #[test]
    fn image_file_constructor_and_initializer_match_source_defaults() {
        unsafe {
            let image_file = ii_new();
            assert_eq!((*image_file).xscale, 1.0);
            assert_eq!((*image_file).smax, 255.0);
            assert_eq!(
                ((*image_file).urx, (*image_file).ury, (*image_file).urz),
                (-1, -1, -1)
            );
            assert_eq!(
                (
                    (*image_file).adoc_index,
                    (*image_file).global_adoc_index,
                    (*image_file).hdf_compression
                ),
                (-1, -1, -1)
            );
            assert_eq!(ii_init(image_file, 4, 5, 6, IIFILE_HDF, 3, IITYPE_FLOAT), 0);
            assert_eq!(
                ((*image_file).nx, (*image_file).ny, (*image_file).nz),
                (4, 5, 6)
            );
            assert_eq!(
                (
                    (*image_file).file,
                    (*image_file).format,
                    (*image_file).type_
                ),
                (IIFILE_HDF, 3, IITYPE_FLOAT)
            );
            assert_eq!(ii_init(core::ptr::null_mut(), 0, 0, 0, 0, 0, 0), -1);
            libc::free(image_file.cast());
        }
    }

    #[test]
    fn opened_file_registry_uses_the_original_ilist_pointer_identity_rules() {
        unsafe {
            let first = ii_new();
            let second = ii_new();
            (*first).fp = Some(ImodFile::Token(1));
            (*second).fp = Some(ImodFile::Token(2));
            assert_eq!(add_to_opened_list(first), 0);
            assert_eq!(ii_add_to_opened_list(second), 0);
            assert_eq!(find_file_in_list(first, None), 0);
            assert_eq!(
                find_file_in_list(core::ptr::null_mut(), (*second).fp.as_ref()),
                1
            );
            assert_eq!(
                ii_lookup_file_from_fp((*first).fp.as_ref().unwrap()),
                Some(first)
            );
            remove_from_opened_list(first);
            assert!(ii_lookup_file_from_fp((*first).fp.as_ref().unwrap()).is_none());
            assert_eq!(
                ii_lookup_file_from_fp((*second).fp.as_ref().unwrap()),
                Some(second)
            );
            remove_from_opened_list(second);
            drop(Box::from_raw(first));
            drop(Box::from_raw(second));
        }
    }

    #[test]
    fn file_address_change_replaces_the_opened_file_registry_entry() {
        unsafe {
            let old_file = ii_new();
            let new_file = ii_new();
            (*old_file).fp = Some(ImodFile::Token(1));
            (*new_file).fp = Some(ImodFile::Token(old_file as usize));
            assert_eq!(add_to_opened_list(old_file), 0);
            ii_file_change_address(old_file, new_file);
            assert!(
                (*new_file)
                    .fp
                    .as_ref()
                    .unwrap()
                    .ptr_eq(&ImodFile::Token(new_file as usize))
            );
            assert_eq!(
                ii_lookup_file_from_fp((*new_file).fp.as_ref().unwrap()),
                Some(new_file)
            );
            remove_from_opened_list(new_file);
            drop(Box::from_raw(old_file));
            drop(Box::from_raw(new_file));
        }
    }

    #[test]
    fn save_and_restore_load_params_copy_only_the_source_load_fields() {
        unsafe {
            let ii_file = ii_new();
            let ii_save = ii_new();
            (*ii_file).llx = 1;
            (*ii_file).urx = 2;
            (*ii_file).lly = 3;
            (*ii_file).ury = 4;
            (*ii_file).llz = 5;
            (*ii_file).urz = 6;
            (*ii_file).axis = 2;
            (*ii_file).pad_left = 7;
            (*ii_file).pad_right = 8;
            (*ii_file).slope = 1.5;
            (*ii_file).offset = -2.5;
            ii_save_load_params(ii_file, ii_save);
            (*ii_file).llx = 0;
            (*ii_file).urx = 0;
            (*ii_file).lly = 0;
            (*ii_file).ury = 0;
            (*ii_file).llz = 0;
            (*ii_file).urz = 0;
            (*ii_file).axis = 0;
            (*ii_file).pad_left = 0;
            (*ii_file).pad_right = 0;
            (*ii_file).slope = 0.;
            (*ii_file).offset = 0.;
            assert_eq!(ii_restore_load_params(-7, ii_file, ii_save), -7);
            assert_eq!(
                (
                    (*ii_file).llx,
                    (*ii_file).urx,
                    (*ii_file).lly,
                    (*ii_file).ury,
                    (*ii_file).llz,
                    (*ii_file).urz,
                    (*ii_file).axis,
                    (*ii_file).pad_left,
                    (*ii_file).pad_right,
                    (*ii_file).slope,
                    (*ii_file).offset,
                ),
                (1, 2, 3, 4, 5, 6, 2, 7, 8, 1.5, -2.5)
            );
            drop(Box::from_raw(ii_file));
            libc::free(ii_save.cast());
        }
    }

    #[test]
    fn lookup_ii_file_saves_and_sets_axis_specific_load_parameters() {
        unsafe {
            let ii_file = ii_new();
            (*ii_file).fp = Some(ImodFile::Token(1));
            (*ii_file).file = IIFILE_TIFF;
            (*ii_file).llx = 99;
            (*ii_file).lly = 98;
            (*ii_file).axis = 1;
            assert_eq!(add_to_opened_list(ii_file), 0);
            let mut header = MrcHeader::default();
            header.fp = (*ii_file).fp.clone();
            let mut load = crate::imod::libiimod::mrcfiles::LoadInfo::default();
            load.xmin = 1;
            load.xmax = 2;
            load.ymin = 3;
            load.ymax = 4;
            load.pad_left = 5;
            load.pad_right = 6;
            load.slope = 1.5;
            load.offset = -2.5;
            let mut save = ImodImageFile::default();
            assert_eq!(
                crate::imod::libiimod::mrcsec::lookup_ii_file(&mut header, &mut load, 3, &mut save),
                ii_file
            );
            assert_eq!((save.llx, save.lly, save.axis), (99, 98, 1));
            assert_eq!(
                (
                    (*ii_file).llx,
                    (*ii_file).urx,
                    (*ii_file).lly,
                    (*ii_file).ury,
                    (*ii_file).axis,
                    (*ii_file).pad_left,
                    (*ii_file).pad_right,
                    (*ii_file).slope,
                    (*ii_file).offset,
                ),
                (1, 2, 3, 4, 3, 5, 6, 1.5, -2.5)
            );
            ii_change_call_count(1);
            assert!(
                crate::imod::libiimod::mrcsec::lookup_ii_file(&mut header, &mut load, 3, &mut save)
                    .is_null()
            );
            ii_change_call_count(-1);
            remove_from_opened_list(ii_file);
            drop(Box::from_raw(ii_file));
        }
    }

    #[test]
    fn quit_callback_returns_the_source_quitting_status() {
        unsafe extern "C" fn quit_on_seven(value: i32) -> i32 {
            (value == 7) as i32
        }
        unsafe {
            ii_register_quit_check(None);
            assert_eq!(ii_check_for_quit(7), 0);
            ii_register_quit_check(Some(quit_on_seven));
            assert_eq!(ii_check_for_quit(6), 0);
            assert_eq!(ii_check_for_quit(7), IIERR_QUITTING);
            ii_register_quit_check(None);
        }
    }

    #[test]
    fn ii_open_dispatches_real_mrc_through_checker_list_and_generic_reader() {
        unsafe {
            use crate::imod::libiimod::mrcfiles::{MRC_MODE_BYTE, mrc_head_new, mrc_head_write};

            let mut path = b"/tmp/imod-rs-iimage-open-XXXXXX\0".to_vec();
            let fd = libc::mkstemp(path.as_mut_ptr().cast());
            assert!(fd >= 0);
            assert_eq!(libc::close(fd), 0);
            let mut fp = crate::imod::libcfshr::b3dutil::ImodFile::open(
                std::str::from_utf8(&path[..path.len() - 1]).unwrap(),
                "wb",
            )
            .unwrap();
            let mut header = MrcHeader::default();
            mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE);
            header.fp = Some(fp.clone());
            assert_eq!(mrc_head_write(&mut fp, &mut header), 0);
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(&[1_u8, 2, 3, 4], 1, 4, &mut fp),
                4
            );
            drop(fp);

            ii_delete_check_list();
            let image = ii_open(&path[..path.len() - 1], "rb");
            assert!(!image.is_null());
            assert_eq!(
                ((*image).state, (*image).file, (*image).nx, (*image).ny),
                (IISTATE_READY, IIFILE_MRC, 2, 2)
            );
            let mut pixels = [0_u8; 4];
            assert_eq!(ii_read_section(image, pixels.as_mut_ptr().cast(), 0), 0);
            assert_eq!(pixels, [129, 130, 131, 132]);
            assert_eq!(ii_read_point(image, 1, 1, 0), 132.);
            assert_eq!(ii_read_point(image, -1, 1, 0), (*image).amin);
            ii_delete(image);
            ii_delete_check_list();
            assert_eq!(libc::unlink(path.as_ptr().cast()), 0);
        }
    }
}
